"""Module 1 — evaluation of every cached embedding of a family on its fixed label-stratified subsample.

Metrics (all in [0, 1], higher = better):
  bio:   scib-metrics isolated labels, k-means NMI/ARI, label silhouette, cLISI; 5-fold kNN label-transfer
         balanced accuracy; lineage stage-order conservation (Spearman of diffusion pseudotime vs stage order for
         neutrophil GMP→Band→Mature→Aged/Activated and B Pro→Pre→Transitional→Naïve/Mature).
  batch (joint only, shared immune types only): BRAS, iLISI, graph connectivity, PCR comparison vs baseline PCA.
  time:  PCR-time (variance of the embedding explained by cos/sin ZT within tissue × cell type) and kNN
         same-sample excess purity within cell type — reported raw and as ratio to the family's baseline PCA
         (capped at 1); integration that erases the time-of-day signal is penalised.
  cross-tissue (joint): nearest-neighbour label consistency between cochlear and BM cells of shared types.

Usage: python scripts/22_dr_metrics.py <family> [baseline_embedding_name]
Outputs: cache/20_dr/<family>/metrics.parquet (incremental), results/20_dr/dr_benchmark_metrics_<family>.csv
"""
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
NJ = int(__import__('os').environ.get('NUMBA_NUM_THREADS', '6'))   # container CPU quota (24 cores)
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import cache_dir, log, write_csv  # noqa: E402

warnings.filterwarnings("ignore")
FAM = sys.argv[1]
BASELINE = sys.argv[2] if len(sys.argv) > 2 else None
root = cache_dir(f"20_dr/{FAM}")
obs_all = pd.read_parquet(root / "obs.parquet")
eval_idx = np.load(root / "eval_idx.npy")
obs = obs_all.iloc[eval_idx].reset_index(drop=True)
mpath = root / "metrics.parquet"
done = pd.read_parquet(mpath) if mpath.exists() else pd.DataFrame(columns=["embedding", "metric", "value"])

LINEAGES = {"neutrophil": {"Granulocyte-Monocyte Progenitor": 0, "Neutrophil (Band)": 1, "Neutrophil (Mature)": 2,
                           "Neutrophil (Aged)": 3, "Neutrophil (Activated)": 3},
            "B": {"Pro-B Cell (Cycling)": 0, "Pre-B Cell": 1, "B Cell (Transitional)": 2, "B Cell (Naïve)": 3,
                  "B Cell (Mature)": 4}}
W = 2 * np.pi / 24


def pcr_time(Z, ob):
    vals, wts = [], []
    for (t, ct), g in ob.groupby(["tissue_code", "cell_type"], observed=True):
        if g["sample_id"].nunique() < 5 or len(g) < 60:
            continue
        X = np.column_stack([np.ones(len(g)), np.cos(W * g["zt"]), np.sin(W * g["zt"])])
        Y = Z[g.index.to_numpy()]
        Yc = Y - Y.mean(0)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        fitted = X @ beta - Y.mean(0)
        vals.append((fitted ** 2).sum() / max((Yc ** 2).sum(), 1e-12))
        wts.append(np.sqrt(len(g)))
    return float(np.average(vals, weights=wts)) if vals else np.nan


def sample_purity_excess(Z, ob, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    ind = ind[:, 1:]
    ct = ob["cell_type"].to_numpy()
    smp = ob["sample_id"].to_numpy()
    same_ct = ct[ind] == ct[:, None]
    same_s = (smp[ind] == smp[:, None]) & same_ct
    with np.errstate(invalid="ignore", divide="ignore"):
        obs_frac = same_s.sum(1) / same_ct.sum(1)
    key = pd.Series(list(zip(ct, smp)))
    n_cs = key.map(key.value_counts()).to_numpy()
    n_c = pd.Series(ct).map(pd.Series(ct).value_counts()).to_numpy()
    exp = (n_cs - 1) / np.maximum(n_c - 1, 1)
    ok = same_ct.sum(1) > 0
    ex = (obs_frac[ok] - exp[ok]) / np.maximum(1 - exp[ok], 1e-6)
    per_ct = pd.Series(ex).groupby(ct[ok]).mean()
    w = np.sqrt(pd.Series(ct[ok]).value_counts().reindex(per_ct.index))
    return float(np.average(per_ct, weights=w))


def knn_transfer(Z, ob, k=15):
    y = ob["cell_type"].astype(str).to_numpy()
    pred = np.empty_like(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(Z, y):
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=NJ).fit(Z[tr], y[tr])
        pred[te] = clf.predict(Z[te])
    return float(balanced_accuracy_score(y, pred))


def stage_order(Z, ob):
    out = {}
    for lin, stages in LINEAGES.items():
        rhos = []
        for t, g in ob[ob["cell_type"].isin(stages)].groupby("tissue_code", observed=True):
            if g["cell_type"].nunique() < 3 or len(g) < 200:
                continue
            sub = ad.AnnData(np.zeros((len(g), 1)), obsm={"Z": Z[g.index.to_numpy()]})
            sc.pp.neighbors(sub, use_rep="Z", n_neighbors=15)
            sc.tl.diffmap(sub, n_comps=10)
            stage = g["cell_type"].map(stages).to_numpy()
            first = np.where(stage == 0)[0]
            dc1 = sub.obsm["X_diffmap"][:, 1]
            sub.uns["iroot"] = int(first[np.argmax(dc1[first])] if np.corrcoef(dc1, stage)[0, 1] < 0
                                   else first[np.argmin(dc1[first])])
            sc.tl.dpt(sub)
            rho = spearmanr(sub.obs["dpt_pseudotime"], stage).correlation
            rhos.append(max(rho, 0))
        out[f"stage_order_{lin}"] = float(np.mean(rhos)) if rhos else np.nan
    return out


def cross_tissue_consistency(Z, ob, k=15):
    vc = ob.groupby(["cell_type", "tissue_code"], observed=True).size().unstack(fill_value=0)
    shared = vc.index[(vc.get("cochlea", 0) >= 50) & (vc.get("bm", 0) >= 50)]
    m = ob["cell_type"].isin(shared).to_numpy()
    res = []
    for src, dst in [("cochlea", "bm"), ("bm", "cochlea")]:
        s = np.where(m & (ob["tissue_code"] == src).to_numpy())[0]
        d = np.where(m & (ob["tissue_code"] == dst).to_numpy())[0]
        nn = NearestNeighbors(n_neighbors=k).fit(Z[d])
        _, ind = nn.kneighbors(Z[s])
        lab = ob["cell_type"].to_numpy()
        res.append((lab[d][ind] == lab[s][:, None]).mean())
    return float(np.mean(res)), list(shared)


embs = sorted(p.stem for p in (root / "emb").glob("*.npy"))
todo = [e for e in embs if e not in set(done["embedding"])]
log(f"{FAM}: {len(embs)} embeddings, {len(todo)} to evaluate; eval cells {len(obs)}")
rows = []
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection  # noqa: E402

if todo:
    Zs = {e: np.load(root / "emb" / f"{e}.npy")[eval_idx] for e in todo}
    adata = ad.AnnData(np.zeros((len(obs), 1)), obs=obs.astype({"cell_type": str, "sample_id": str, "tissue_code": str}))
    for e, Z in Zs.items():
        adata.obsm[e] = Z
    t0 = time.time()
    bm = Benchmarker(adata, batch_key="sample_id", label_key="cell_type", embedding_obsm_keys=list(Zs),
                     bio_conservation_metrics=BioConservation(isolated_labels=True, nmi_ari_cluster_labels_kmeans=True,
                                                              nmi_ari_cluster_labels_leiden=False,
                                                              silhouette_label=True, clisi_knn=True),
                     batch_correction_metrics=None, pre_integrated_embedding_obsm_key=list(Zs)[0],
                     n_jobs=NJ, progress_bar=False)
    bm.benchmark()
    bio = bm.get_results(min_max_scale=False).drop(index="Metric Type", errors="ignore")
    log(f"bio benchmark {time.time() - t0:.0f}s")
    for e in Zs:
        for metric, v in bio.loc[e].items():
            if metric not in ("Total", "Bio conservation", "Batch correction"):
                rows.append({"embedding": e, "metric": metric, "value": float(v)})
    if FAM == "joint":
        _, shared = cross_tissue_consistency(next(iter(Zs.values())), obs)
        sm = obs["cell_type"].isin(shared).to_numpy()
        sub = adata[sm].copy()
        base_key = BASELINE if BASELINE in Zs else None
        if BASELINE and BASELINE not in Zs:
            sub.obsm[BASELINE] = np.load(root / "emb" / f"{BASELINE}.npy")[eval_idx][sm]
        bm2 = Benchmarker(sub, batch_key="tissue_code", label_key="cell_type", embedding_obsm_keys=list(Zs),
                          bio_conservation_metrics=None,
                          batch_correction_metrics=BatchCorrection(bras=True, ilisi_knn=True, kbet_per_label=False,
                                                                   graph_connectivity=True,
                                                                   pcr_comparison=BASELINE is not None, sbee=False),
                          pre_integrated_embedding_obsm_key=BASELINE or list(Zs)[0], n_jobs=NJ,
                          progress_bar=False)
        bm2.benchmark()
        bat = bm2.get_results(min_max_scale=False).drop(index="Metric Type", errors="ignore")
        for e in Zs:
            for metric, v in bat.loc[e].items():
                if metric not in ("Total", "Bio conservation", "Batch correction"):
                    rows.append({"embedding": e, "metric": f"platform_{metric}", "value": float(v)})
    for e, Z in Zs.items():
        t0 = time.time()
        extra = {"knn_transfer_bal_acc": knn_transfer(Z, obs), "pcr_time_raw": pcr_time(Z, obs),
                 "sample_purity_excess_raw": sample_purity_excess(Z, obs)}
        extra.update(stage_order(Z, obs))
        if FAM == "joint":
            extra["cross_tissue_label_consistency"] = cross_tissue_consistency(Z, obs)[0]
        rows += [{"embedding": e, "metric": k, "value": v} for k, v in extra.items()]
        log(f"  {e}: custom metrics {time.time() - t0:.0f}s")
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_parquet(mpath)

# ---- composite scores (recomputed over all evaluated embeddings)
wide = done.pivot_table(index="embedding", columns="metric", values="value")
if BASELINE and BASELINE in wide.index:
    for m in ["pcr_time_raw", "sample_purity_excess_raw"]:
        wide[m.replace("_raw", "_ratio")] = np.minimum(wide[m] / wide.loc[BASELINE, m], 1.0)
bio_cols = [c for c in ["Isolated labels", "KMeans NMI", "KMeans ARI", "Silhouette label", "cLISI",
                        "knn_transfer_bal_acc", "stage_order_neutrophil", "stage_order_B"] if c in wide]
wide["score_bio"] = wide[bio_cols].mean(1)
tcols = [c for c in ["pcr_time_ratio", "sample_purity_excess_ratio"] if c in wide]
if tcols:
    wide["score_time"] = wide[tcols].mean(1)
pcols = [c for c in wide if c.startswith("platform_")]
if pcols:
    wide["score_platform"] = wide[pcols].mean(1)
long = wide.reset_index().melt(id_vars="embedding", var_name="metric", value_name="value")
long.insert(0, "family", FAM)
write_csv(long, "20_dr", f"dr_benchmark_metrics_{FAM}.csv", f"Embedding benchmark metrics, family {FAM}",
          required=["family", "embedding", "metric", "value"])
show = [c for c in ["score_bio", "score_time", "score_platform", "pcr_time_raw", "sample_purity_excess_raw",
                    "knn_transfer_bal_acc", "stage_order_neutrophil", "stage_order_B",
                    "cross_tissue_label_consistency"] if c in wide]
print(wide[show].sort_values("score_bio", ascending=False).round(3).to_string())
log("done")
