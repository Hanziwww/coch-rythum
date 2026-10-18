"""Round 9 — verification of the Milo exact-null result (I100), Python part (after 53e_milo_verification.R).

V3  Per-neighbourhood exact null (both embeddings): within each neighbourhood the observed class (1) is ranked among
    the 60 relabelling classes by p; per-neighbourhood exact p = share of classes with p <= observed. Global tests
    against the same 60 classes: number of neighbourhoods whose best class is the observed one, and the rank sum.
    Subsets: all; not dominated by one sample (largest sample < 50 % of cells; primary embedding); majority types with
    >= 60 neighbourhoods.
V4  Biology vs batch: dominance and purity per sample and cell class; rhythmic calls per sample.
V5  External replication: for every sample whose cells dominate >= 10 rhythmic neighbourhoods (spatial FDR < 0.1;
    primary embedding), a signature (top 50 non-clock genes, mean log1p CP10k of the sample's cells in those
    neighbourhoods minus the same cell types in the other samples, weighted over types) is scored in an independent,
    replicated time series of the same tissue (marrow: GSE198770 controls, ZT1-21, n = 3; cochlea: GSE88954 sham,
    CT0-20, n = 3) and its peak compared with the expected time (sample ZT aligned by the tissue clock offsets of
    f2c_clock_offsets). Association across signatures: C = mean cos(peak - expected), exact p over all permutations of
    the expected times; per signature, the external cosinor p and the same statistic for 1,000 expression-matched
    random gene sets.
V6  Sample-level tests that do not use Milo (collected from existing tables).
Outputs (results/53_milo/): milo_verification_exact.csv, milo_verification_dominance.csv, milo_verification_external.csv,
milo_verification_summary.csv
Run: python scripts/53f_milo_verification.py
"""
import itertools
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import H5AD, RESULTS, ROOT, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402

MOD = "53_milo"
M = RESULTS / MOD
CLOCK = {"Arntl", "Clock", "Npas2", "Nr1d1", "Nr1d2", "Rora", "Rorc", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Dbp", "Tef",
         "Hlf", "Nfil3", "Ciart", "Bhlhe40", "Bhlhe41"}
rng = np.random.default_rng(53)
cls = pd.read_csv(RESULTS / "r9_figures/palette_celltypes.csv").set_index("cell_type")


# ------------------------------------------------------------------------------------------------ V3
def exact_rows(P, label, tissue, emb):
    """P: neighbourhoods x 60 classes of p (class 1 = observed)."""
    n = P.shape[0]
    L = -np.log10(np.clip(P, 1e-300, 1))
    nb_p = (L >= L[:, [0]] - 1e-12).mean(1)                         # per-neighbourhood exact p (ties count)
    best = np.argmax(L + rng.uniform(0, 1e-9, L.shape), axis=1)    # random tie-break
    nbest = np.bincount(best, minlength=60)
    rsum = stats.rankdata(L, axis=1).sum(0)
    out = []
    for name, v in (("best-class count", nbest.astype(float)), ("rank sum of -log10 p", rsum)):
        out.append({"tissue": tissue, "embedding": emb, "subset": label, "n_nhoods": n, "statistic": name,
                    "observed": float(v[0]), "null_mean": float(v[1:].mean()), "null_q95": float(np.quantile(v[1:], 0.95)),
                    "p_exact": float((v >= v[0] - 1e-9).mean()), "n_nhood_exact_p_min": int((nb_p <= 1 / 60 + 1e-12).sum()),
                    "expected_n_nhood_exact_p_min": n / 60})
    return out


ex_rows = []
for t in ["bm", "cochlea"]:
    rob = pd.read_csv(M / f"milo_robustness_{t}.csv")
    for emb in ["scanvi", "scvi"]:
        f = pd.read_csv(M / f"milo_relabel_fits_{t}_{emb}.csv")
        P = f.pivot(index="nhood", columns="relabel_class", values="p").sort_index(axis=1)
        assert P.shape[1] == 60
        da = pd.read_csv(M / f"milo_da_results_{t}{'' if emb == 'scanvi' else '_scvi'}.csv").set_index("nhood")
        assert np.allclose(P[1].to_numpy(), da.loc[P.index, "p"].to_numpy(), rtol=0, atol=1e-12)
        ex_rows += exact_rows(P.to_numpy(), "all", t, emb)
        if emb == "scanvi":
            nd = rob.set_index("nhood").loc[P.index, "dominated"].to_numpy()
            ex_rows += exact_rows(P.to_numpy()[~nd], "not dominated by one sample", t, emb)
            ex_rows += exact_rows(P.to_numpy()[nd], "dominated by one sample", t, emb)
        maj = da.loc[P.index, "majority_cell_type"]
        for ct_, cnt in maj.value_counts().items():
            if cnt >= 60:
                ex_rows += exact_rows(P.to_numpy()[(maj == ct_).to_numpy()], f"majority: {ct_}", t, emb)
    log(f"V3 {t} done")
ex = pd.DataFrame(ex_rows)

# ------------------------------------------------------------------------------------------------ V4
dom_rows = []
for t in ["bm", "cochlea"]:
    rob = pd.read_csv(M / f"milo_robustness_{t}.csv")
    rob["cell_class"] = rob.majority_cell_type.map(cls["cell_class"])
    rob["structural"] = rob.majority_cell_type.map(cls["structural"]).astype(bool)
    for key, g in [("all", rob)] + [(f"class: {c}", g) for c, g in rob.groupby("cell_class")] + \
                  [("structural", rob[rob.structural]), ("haematopoietic", rob[~rob.structural])]:
        dom_rows.append({"tissue": t, "group": key, "kind": "cell class", "n_nhoods": len(g),
                         "share_dominated": float(g.dominated.mean()), "median_top_share": float(g.top_sample_share.median()),
                         "share_pure_ge90": float((g.top_sample_share >= 0.9).mean()),
                         "n_rhythmic": int((g.spatial_fdr < 0.1).sum())})
    for s, g in rob[rob.dominated].groupby("top_sample"):
        dom_rows.append({"tissue": t, "group": f"{s} (ZT{int(g.top_sample_zt.iloc[0])})", "kind": "dominant sample",
                         "n_nhoods": len(g), "share_dominated": np.nan, "median_top_share": float(g.top_sample_share.median()),
                         "share_pure_ge90": float((g.top_sample_share >= 0.9).mean()),
                         "n_rhythmic": int((g.spatial_fdr < 0.1).sum())})
dom = pd.DataFrame(dom_rows)
log("V4 done")

# ------------------------------------------------------------------------------------------------ V5
off = pd.read_csv(RESULTS / "r9_figures/f2c_clock_offsets.csv").set_index("label_short")["estimate_h"]
EXT = {"bm": ("external/prepared_r2/GSE198770_bone_marrow__expr.csv", "external/prepared_r2/GSE198770_bone_marrow__meta.csv",
              "control", off["Marrow pool"], off["GSE198770 marrow"], "GSE198770 marrow (control)"),
       "cochlea": ("external/prepared/GSE88954__expr.csv", "external/prepared/GSE88954__meta.csv", "sham",
                   off["Cochlea pool"], off["GSE88954 sham"], "GSE88954 cochlea (sham)")}


def cosinor(y, tt):
    X = np.column_stack([np.ones_like(tt), np.cos(phase.W * tt), np.sin(phase.W * tt)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss1 = float(((y - X @ b) ** 2).sum()); rss0 = float(((y - y.mean()) ** 2).sum())
    df2 = len(y) - 3
    F = ((rss0 - rss1) / 2) / (rss1 / df2)
    return float(np.arctan2(b[2], b[1]) % (2 * np.pi) / phase.W), float(np.hypot(b[1], b[2])), float(stats.f.sf(F, 2, df2))


A = ad.read_h5ad(H5AD, backed="r")
obs_all = A.obs.copy()
obs_all["tcode"] = obs_all["tissue"].astype(str).map(TISSUE_FROM_OBS)
st = sample_table(A.obs)
ext_rows, assoc_rows = [], []
for t in ["bm", "cochlea"]:
    rob = pd.read_csv(M / f"milo_robustness_{t}.csv")
    mem = pd.read_csv(M / f"milo_membership_{t}.csv")
    calls = rob[rob.spatial_fdr < 0.1][["nhood", "top_sample", "top_sample_zt"]]
    mem = mem.merge(calls, on="nhood")
    o = obs_all[obs_all["tcode"] == t]
    o_sample = o["batch_id"].astype(str)
    sub = A[o.index].to_memory()
    X = sub.X.tocsr().astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)
    genes = np.asarray(sub.var_names.astype(str))
    pos = pd.Series(np.arange(sub.n_obs), index=sub.obs_names)
    ctype = sub.obs["cell_type"].astype(str).to_numpy()
    smp = o_sample.reindex(sub.obs_names).to_numpy()
    fx, fm, cond, off_ours, off_ext, ext_name = EXT[t]
    E = pd.read_csv(ROOT / fx, index_col=0)
    meta = pd.read_csv(ROOT / fm)
    meta = meta[meta["condition"] == cond]
    E = E[meta["sample"]]
    if E.to_numpy().max() > 50:                                   # raw counts -> log2 CPM
        E = np.log2(E / E.sum(0) * 1e6 + 1)
    E = E[(E.mean(1) > 1)]
    Z = E.sub(E.mean(1), axis=0).div(E.std(1).replace(0, np.nan), axis=0).dropna()
    tt = meta.set_index("sample").loc[Z.columns, "time_h"].to_numpy(float)
    edec = pd.qcut(E.loc[Z.index].mean(1), 10, labels=False)
    sig_rows = []
    for s, g in mem.groupby("top_sample"):
        if g.nhood.nunique() < 10:
            continue
        cells = g.loc[g.cell_id.isin(sub.obs_names) & (g.cell_id.map(o_sample) == s), "cell_id"].unique()
        idx_d = pos[cells].to_numpy()
        types = pd.Series(ctype[idx_d]).value_counts()
        lfc, wsum = np.zeros(len(genes)), 0.0
        det = np.asarray((X[idx_d] > 0).mean(0)).ravel()
        for ty, nty in types.items():
            if nty < 20:
                continue
            d_i = idx_d[ctype[idx_d] == ty]
            c_i = np.where((ctype == ty) & (smp != s))[0]
            if len(c_i) < 50:
                continue
            lfc += nty * (np.asarray(X[d_i].mean(0)).ravel() - np.asarray(X[c_i].mean(0)).ravel())
            wsum += nty
        if wsum == 0:
            continue
        lfc /= wsum
        ok = (det >= 0.1) & ~np.isin(genes, list(CLOCK)) & np.isin(genes, Z.index)
        order_ = np.argsort(-np.where(ok, lfc, -np.inf))[:50]
        sig = genes[order_]
        zt_s = float(g.top_sample_zt.iloc[0])
        expected = (zt_s - off_ours + off_ext) % 24
        score = Z.loc[sig].mean(0).to_numpy()
        pk, amp, p_ = cosinor(score, tt)
        d = float(phase.circ_diff(pk, expected))
        # expression-matched random gene sets
        dec_sig = edec.loc[sig].value_counts()
        rc = []
        for _ in range(1000):
            rs = np.concatenate([rng.choice(edec.index[edec == k].to_numpy(), n_, replace=False) for k, n_ in dec_sig.items()])
            pk_r, amp_r, p_r = cosinor(Z.loc[rs].mean(0).to_numpy(), tt)
            rc.append(np.cos(phase.W * phase.circ_diff(pk_r, expected)) * (p_r < 0.05))
        obs_c = np.cos(phase.W * d) * (p_ < 0.05)
        sig_rows.append({"tissue": t, "external": ext_name, "dominant_sample": s, "sample_zt": zt_s,
                         "n_rhythmic_nhoods": int(g.nhood.nunique()), "n_cells": len(cells),
                         "cell_types": "; ".join(f"{k} {v}" for k, v in types.head(4).items()),
                         "signature_top10": "; ".join(sig[:10]), "expected_external_peak": expected,
                         "external_peak": pk, "external_amplitude_z": amp, "external_p": p_, "phase_diff_h": d,
                         "p_random_sets": float((1 + np.sum(np.array(rc) >= obs_c - 1e-12)) / 1001)})
    sr = pd.DataFrame(sig_rows)
    ext_rows.append(sr)
    if len(sr) >= 2:
        exp_ = sr["expected_external_peak"].to_numpy(); pk_ = sr["external_peak"].to_numpy()
        C0 = float(np.mean(np.cos(phase.W * phase.circ_diff(pk_, exp_))))
        perms = list(itertools.permutations(range(len(sr))))
        Cs = np.array([np.mean(np.cos(phase.W * phase.circ_diff(pk_, exp_[list(pp)]))) for pp in perms])
        assoc_rows.append({"tissue": t, "external": ext_name, "n_signatures": len(sr), "C_mean_cos": C0,
                           "p_exact_permutation": float((Cs >= C0 - 1e-12).mean()), "n_permutations": len(perms)})
    log(f"V5 {t} done")
ext = pd.concat(ext_rows, ignore_index=True)
assoc = pd.DataFrame(assoc_rows)

# ------------------------------------------------------------------------------------------------ V6 + summary
mt = pd.read_csv(RESULTS / "r6_myeloid/m_trajectory_rhythm.csv")
m3 = mt[mt["feature"].str.contains("M3 primary", regex=False)]
mm = pd.read_csv(RESULTS / "r6_myeloid/m_multimethod_matrix.csv")
ot = mm[mm["method"].str.startswith("Wasserstein")]
summ = []
for _, r in m3.iterrows():
    summ.append({"line": "V6 sample-level: M3 median neutrophil-branch pseudotime", "tissue": r.tissue,
                 "result": f"exact p {r.p_perm:.3f}; OT antiphase test p {r.ot_p_perm:.3f}"})
for _, r in ot.iterrows():
    summ.append({"line": "V6 sample-level: M1 OT shift of the maturation axis", "tissue": r.tissue, "result": str(r.value)})
for _, r in assoc.iterrows():
    summ.append({"line": "V5 external replication of sample-dominated rhythmic-neighbourhood signatures", "tissue": r.tissue,
                 "result": f"{r.external}: mean cos {r.C_mean_cos:+.2f} over {int(r.n_signatures)} signatures, exact permutation "
                           f"p {r.p_exact_permutation:.3f}"})
for _, r in ex[(ex.subset.isin(["all", "not dominated by one sample"])) & (ex.statistic == "best-class count")].iterrows():
    summ.append({"line": f"V3 per-neighbourhood exact null ({r.embedding}; {r.subset})", "tissue": r.tissue,
                 "result": f"observed best-class count {int(r.observed)} vs null mean {r.null_mean:.1f} (q95 {r.null_q95:.1f}); "
                           f"exact p {r.p_exact:.3f}; neighbourhoods at exact p 1/60: {r.n_nhood_exact_p_min} "
                           f"(expected {r.expected_n_nhood_exact_p_min:.1f})"})
write_csv(ex, MOD, "milo_verification_exact.csv", "Milo verification V3: per-neighbourhood exact relabelling null (best-class "
          "count and rank sum vs the 60 classes) per tissue, embedding and subset", required=["tissue", "embedding", "subset",
                                                                                            "statistic", "p_exact"])
write_csv(dom, MOD, "milo_verification_dominance.csv", "Milo verification V4: single-sample dominance and purity per cell "
          "class and dominant sample, with rhythmic calls", required=["tissue", "group", "kind", "n_nhoods"])
write_csv(pd.concat([ext.assign(level="signature"), assoc.assign(level="association")], ignore_index=True), MOD,
          "milo_verification_external.csv", "Milo verification V5: signatures of sample-dominated rhythmic neighbourhoods "
          "scored in independent replicated time series (peak vs the sample's aligned time; exact permutation and "
          "random-gene-set controls)", required=["tissue", "external", "level"])
write_csv(pd.DataFrame(summ), MOD, "milo_verification_summary.csv", "Milo verification: one line per check (V3, V5, V6)",
          required=["line", "tissue", "result"])
print(ex.round(3).to_string())
print(dom.round(3).to_string())
print(ext.drop(columns=["signature_top10"]).round(3).to_string())
print(assoc.round(3).to_string())
log("done")
