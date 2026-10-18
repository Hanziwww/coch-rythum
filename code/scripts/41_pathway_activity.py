"""Module 3a — pathway and TF-activity rhythms per tissue × cell type.

Pseudobulk level (all resources): genes passing the expression filter are z-scored across the 6 samples;
  - unweighted sets (MSigDB Hallmark/Reactome/GO-BP, curated YAML): per-sample mean z of member genes
    (>= 5 expressed members; MSigDB sets restricted to 10–500 members);
  - weighted, signed resources (PROGENy mouse top-200 genes/pathway; CollecTRI mapped human→mouse 1:1 orthologs):
    univariate linear model (decoupler ULM) activity per sample.
Cell level (curated + Hallmark only): AUCell scores per cell, per-sample median per cell type.
Every activity series (6 ZTs) gets a cosinor fit, 60-class permutation p, LOTO; BH within tissue×group×resource.
Sensitivity mode (env DEPTH_MATCH=1): each cell type's 6 pseudobulk samples are binomially thinned to the smallest
library among them (seed from the cell-type name) before scoring, so log-transform/detection biases that follow
sequencing depth cannot create time effects; writes *_depth_matched.csv files and skips the cell-level part.

Outputs (results/41_pathway/): pathway_scores.csv, pathway_rhythm.csv, cell_score_sample_medians.csv,
                               cell_score_rhythm.csv, resource_summary.csv
"""
import os
import re
import sys
import zlib
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, log, sample_table, write_csv  # noqa: E402

MOD = "41_pathway"
DEPTH_MATCH = os.environ.get("DEPTH_MATCH", "0") == "1"
SFX = "_depth_matched" if DEPTH_MATCH else ""
RES = EXTERNAL / "resources"
SCRIPTS = Path(__file__).resolve().parent


def read_gmt(path, min_size=10, max_size=500):
    sets = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if min_size <= len(p) - 2 <= max_size:
            sets[p[0]] = p[2:]
    return sets


def curated_sets():
    y = yaml.safe_load(open(SCRIPTS / "resources/curated_sets.yaml"))
    return {f"{cat}:{name}": genes for cat, d in y.items() for name, genes in d.items()}


def mgi_orthologs():
    hom = pd.read_csv(RES / "mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
    hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
    mm = hom[hom["org"].str.startswith("mouse")][["key", "symbol"]]
    hs = hom[hom["org"].str.startswith("human")][["key", "symbol"]]
    one = (mm.groupby("key").size() == 1) & (hs.groupby("key").size().reindex(mm.groupby("key").size().index) == 1)
    keys = one[one].index
    m = hs[hs["key"].isin(keys)].merge(mm[mm["key"].isin(keys)], on="key", suffixes=("_hs", "_mm"))
    return dict(zip(m["symbol_hs"], m["symbol_mm"]))


def collectri_mouse(atlas_genes):
    df = pd.read_csv(RES / "collectri_human.tsv", sep="\t")
    if (RES / "mgi_hom_mouse_human.rpt").exists():
        o = mgi_orthologs()
        log("CollecTRI mapped with MGI 1:1 orthologues")
    else:   # fallback: human symbol -> mouse capitalisation, restricted to atlas genes
        cap = {g.upper(): g for g in atlas_genes}
        o = {h: cap[h] for h in set(df["source_genesymbol"]) | set(df["target_genesymbol"]) if h in cap}
        log("CollecTRI mapped by symbol case conversion (MGI file unavailable)")
    df["source"] = df["source_genesymbol"].map(o)
    df["target"] = df["target_genesymbol"].map(o)
    df = df.dropna(subset=["source", "target"])
    df["weight"] = np.where(df["is_inhibition"].astype(str) == "True", -1.0, 1.0)
    return df[["source", "target", "weight"]].drop_duplicates(["source", "target"])


def progeny_mouse(top=200):
    p = RES / "progeny_mouse.csv"     # written from the .rda by 41_convert_progeny.R
    df = pd.read_csv(p)
    df = df.sort_values("p.value").groupby("pathway").head(top)
    return df.rename(columns={"gene": "target", "pathway": "source"})[["source", "target", "weight"]]


resources = {"curated": curated_sets()}
for name, fn, lo, hi in [("hallmark", "msigdb_mh_hallmark.gmt", 10, 500), ("reactome", "msigdb_m2_reactome.gmt", 10, 500),
                         ("gobp", "msigdb_m5_gobp.gmt", 15, 300)]:
    if (RES / fn).exists():
        resources[name] = read_gmt(RES / fn, lo, hi)
    else:
        log(f"resource missing, skipped: {fn}")
weighted = {}
if (RES / "progeny_mouse.csv").exists():
    weighted["progeny"] = progeny_mouse()
if (RES / "collectri_human.tsv").exists():
    _genes = pd.read_parquet(CACHE / "30_pseudobulk" / "bm__cell_class.parquet", columns=[]).index
    weighted["collectri"] = collectri_mouse(_genes)
log("resources: " + ", ".join(f"{k}={len(v)}" for k, v in {**resources, **weighted}.items()))

design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
incl = pd.read_csv(RESULTS / "30_pseudobulk/groups_included.csv")
incl = incl[(incl["level"] == "cell_type") & (incl["status"] != "excluded")]

score_rows, rhythm_rows = [], []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    for _, inc in incl[incl["tissue"] == tissue].iterrows():
        group = inc["group"]
        d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type") & (design["group"] == group)]
        d = d.sort_values("zt")
        cnt = pb[[f"{group}|{s}" for s in d["sample_id"]]]
        if DEPTH_MATCH:
            lib0 = cnt.sum(0).to_numpy().astype(float)
            rs = np.random.default_rng(zlib.crc32(f"{tissue}|{group}".encode()))
            cnt = pd.DataFrame(rs.binomial(cnt.to_numpy().astype(np.int64), (lib0.min() / lib0)[None, :]),
                               index=cnt.index, columns=cnt.columns)
        lib = cnt.sum(0)
        cpm = cnt / lib * 1e6
        keep = ((cpm >= 10 / lib.median() * 1e6).sum(1) >= 3) & (cnt.sum(1) >= 20)
        lcpm = np.log2(cpm[keep] + 1)
        z = lcpm.sub(lcpm.mean(1), axis=0).div(lcpm.std(1).replace(0, np.nan), axis=0).dropna()
        zt = d["zt"].to_numpy()
        genes = set(z.index)
        series = {}
        for rname, sets in resources.items():
            for sname, members in sets.items():
                m = [g for g in members if g in genes]
                if len(m) >= 5:
                    series[(rname, sname, len(m))] = z.loc[m].mean(0).to_numpy()
        for rname, net in weighted.items():
            mat = pd.DataFrame(z.T.to_numpy(), index=d["sample_id"], columns=z.index)
            es, _ = dc.mt.ulm(mat, net, tmin=5, verbose=False)
            sizes = net[net["target"].isin(genes)].groupby("source").size()
            for src in es.columns:
                series[(rname, src, int(sizes.get(src, 0)))] = es[src].to_numpy()
        keys = list(series)
        Y = np.column_stack([series[k] for k in keys])
        f = cosinor.fit(Y, zt)
        _, _, p_perm, _ = cosinor.permutation_test(Y, zt)
        lo = cosinor.loto(Y, zt)
        for j, (rname, sname, n) in enumerate(keys):
            for s, v in zip(d["sample_id"], Y[:, j]):
                score_rows.append((tissue, group, s, rname, sname, v))
            rhythm_rows.append({"tissue": tissue, "cell_type": group, "resource": rname, "pathway": sname,
                                "n_genes": n, "mesor": f["mesor"][j], "amplitude": f["amp"][j],
                                "peak_zt": f["peak"][j], "r2": f["r2"][j], "F": f["F"][j], "p": f["p"][j],
                                "p_perm": p_perm[j], "loto_phase_sd_h": lo["loto_phase_sd_h"][j],
                                "loto_min_amp_ratio": lo["loto_min_amp_ratio"][j]})
        log(f"{tissue} {group}: {len(keys)} pathway series")

scores = pd.DataFrame(score_rows, columns=["tissue", "cell_type", "sample_id", "resource", "pathway", "score"])
zt_map = design.drop_duplicates("sample_id").set_index("sample_id")["zt"]
scores["zt"] = scores["sample_id"].map(zt_map)
write_csv(scores, MOD, f"pathway_scores{SFX}.csv", "Per-sample pathway/TF activity scores from pseudobulk (mean z or ULM)"
          + (" — depth-matched (binomial thinning)" if DEPTH_MATCH else ""),
          required=["tissue", "cell_type", "sample_id", "zt", "resource", "pathway", "score"])
rh = pd.DataFrame(rhythm_rows)
rh["q_BH"] = rh.groupby(["tissue", "cell_type", "resource"])["p"].transform(
    lambda p: multipletests(p, method="fdr_bh")[1])
rh["robust"] = (rh["q_BH"] < 0.1) & (rh["p_perm"] <= 0.05) & (rh["loto_phase_sd_h"] < 3)
write_csv(rh, MOD, f"pathway_rhythm{SFX}.csv", "Cosinor rhythm tests of pathway/TF activity per tissue × cell type"
          + (" — depth-matched (binomial thinning)" if DEPTH_MATCH else ""),
          required=["tissue", "cell_type", "resource", "pathway", "amplitude", "peak_zt", "p", "p_perm", "q_BH"])
summ = rh.groupby(["tissue", "cell_type", "resource"]).agg(n_tested=("p", "size"), n_robust=("robust", "sum"),
                                                          n_p01=("p", lambda x: int((x < 0.01).sum()))).reset_index()
write_csv(summ, MOD, f"resource_summary{SFX}.csv", "Counts of rhythmic pathways per tissue × cell type × resource",
          required=["tissue", "cell_type", "resource", "n_tested", "n_robust"])

if DEPTH_MATCH:
    log("done (depth-matched sensitivity; cell-level part skipped)")
    sys.exit(0)

# ---- cell-level AUCell scores for curated + Hallmark sets (sample medians per cell type)
log("cell-level AUCell scoring")
a = ad.read_h5ad(H5AD)
st = sample_table(a.obs).set_index("sample_id")
net = pd.DataFrame([(s, g) for name in ["curated", "hallmark"] if name in resources
                    for s, gl in resources[name].items() for g in gl],
                   columns=["source", "target"]).drop_duplicates()
net["weight"] = 1.0
lib = np.asarray(a.X.sum(1)).ravel()
import scipy.sparse as sp  # noqa: E402
X = sp.diags(1e4 / np.maximum(lib, 1)) @ a.X.tocsr().astype(np.float32)
X.data = np.log1p(X.data)
b = ad.AnnData(X.tocsr(), obs=a.obs[["batch_id", "cell_type", "tissue"]].copy(), var=pd.DataFrame(index=a.var_names))
dc.mt.aucell(b, net, tmin=5, verbose=False)
sc_scores = b.obsm["score_aucell"] if "score_aucell" in b.obsm else b.obsm["aucell_estimate"]
sc_scores = pd.DataFrame(np.asarray(sc_scores), index=b.obs_names,
                         columns=getattr(sc_scores, "columns", None))
sc_scores["sample_id"] = b.obs["batch_id"].astype(str).values
sc_scores["cell_type"] = b.obs["cell_type"].astype(str).values
med = sc_scores.groupby(["sample_id", "cell_type"]).median().reset_index()
med["tissue"] = med["sample_id"].map(st["tissue"])
med["zt"] = med["sample_id"].map(st["zt"])
long = med.melt(id_vars=["tissue", "sample_id", "zt", "cell_type"], var_name="pathway", value_name="median_aucell")
write_csv(long, MOD, "cell_score_sample_medians.csv", "Per-sample median AUCell scores per cell type",
          required=["tissue", "sample_id", "zt", "cell_type", "pathway", "median_aucell"])
cr = []
for (tissue, ctype), g in long.merge(incl[["tissue", "group"]].rename(columns={"group": "cell_type"})).groupby(
        ["tissue", "cell_type"]):
    w = g.pivot(index="zt", columns="pathway", values="median_aucell").sort_index().dropna(axis=1)
    if len(w) < 6:
        continue
    f = cosinor.fit(w.to_numpy(), w.index.to_numpy())
    _, _, pp, _ = cosinor.permutation_test(w.to_numpy(), w.index.to_numpy())
    for j, pw in enumerate(w.columns):
        cr.append({"tissue": tissue, "cell_type": ctype, "pathway": pw, "mesor": f["mesor"][j],
                   "amplitude": f["amp"][j], "rel_amplitude": f["amp"][j] / max(abs(f["mesor"][j]), 1e-9),
                   "peak_zt": f["peak"][j], "p": f["p"][j], "p_perm": pp[j]})
cr = pd.DataFrame(cr)
cr["q_BH"] = cr.groupby(["tissue", "cell_type"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(cr, MOD, "cell_score_rhythm.csv", "Cosinor tests of per-sample median AUCell scores (curated + Hallmark)",
          required=["tissue", "cell_type", "pathway", "peak_zt", "p", "p_perm", "q_BH"])
log("done")
