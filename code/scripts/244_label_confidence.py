"""Round 2, WP5 — sensitivity of key findings to annotation confidence.

High-confidence cells: label-support margin (00_audit/label_support_qc.csv: own-label marker score minus the best
alternative) > 0, i.e. cells whose own-label signature is the strongest (labels are never changed).
(a) Abundance rhythms (% of tissue cells) recomputed from high-confidence cells only: OLS cosinor (figure model) and
    a quasi-binomial GLM F test (statsmodels), compared with all cells (peak shift, p).
(b) Gene rhythms of BM Tier-1 genes in the main rhythmic BM cell types, recomputed from high-confidence-cell pseudobulks
    (OLS cosinor on log2 CPM): fraction keeping p < 0.05 and median peak shift vs the all-cell fit.
Outputs (results/r2_audit/): label_confidence_abundance.csv, label_confidence_genes.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import H5AD, RESULTS, log, sample_table, write_csv  # noqa: E402

MOD = "r2_audit"
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["cell_id", "tissue", "sample_id", "cell_type", "margin"])
qc["confident"] = qc["margin"] > 0
st = qc.drop_duplicates("sample_id")[["sample_id", "tissue"]]
a = ad.read_h5ad(H5AD)
smp = sample_table(a.obs).set_index("sample_id")
zt_of = smp["zt"].to_dict()


def fit_counts(cnt):
    """cnt: samples × types counts (DataFrame, rows sorted by ZT)."""
    zt = np.array([zt_of[s] for s in cnt.index], float)
    pct = 100 * cnt.div(cnt.sum(1), axis=0)
    f = cosinor.fit(pct.to_numpy(), zt)
    out = {}
    for j, c in enumerate(cnt.columns):
        X = sm.add_constant(np.column_stack([np.cos(phase.W * zt), np.sin(phase.W * zt)]))
        y = np.column_stack([cnt[c].to_numpy(), cnt.sum(1).to_numpy() - cnt[c].to_numpy()])
        try:
            m1 = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            m0 = sm.GLM(y, X[:, :1], family=sm.families.Binomial()).fit()
            # dispersion = Pearson X2 / residual df on the count scale (statsmodels' scale="X2" is on the proportion
            # scale for two-column binomial endog, which made every F test p ≈ 0)
            disp = m1.pearson_chi2 / m1.df_resid
            dev = (m0.deviance - m1.deviance) / 2 / disp
            pq = float(stats.f.sf(dev, 2, len(zt) - 3))
        except Exception:
            pq = np.nan
        out[c] = (float(f["peak"][j]), float(f["p"][j]), pq)
    return out


rows = []
for tissue, q in qc.groupby("tissue"):
    for label, sub in [("all_cells", q), ("high_confidence", q[q["confident"]])]:
        cnt = sub.groupby(["sample_id", "cell_type"]).size().unstack(fill_value=0)
        cnt = cnt.loc[sorted(cnt.index, key=lambda s: zt_of[s])]
        cnt = cnt.loc[:, (cnt.sum(0) >= 20) & ((cnt > 0).sum(0) >= 3)]
        for c, (pk, p, pq) in fit_counts(cnt).items():
            rows.append({"tissue": tissue, "cell_set": label, "cell_type": c, "n_cells": int(cnt[c].sum()),
                         "peak_zt": pk, "p_ols_pct": p, "p_quasibinomial": pq})
ab = pd.DataFrame(rows)
w = ab.pivot_table(index=["tissue", "cell_type"], columns="cell_set", values=["peak_zt", "p_ols_pct", "p_quasibinomial", "n_cells"])
w.columns = [f"{a_}_{b_}" for a_, b_ in w.columns]
w = w.reset_index()
w["peak_shift_h"] = phase.circ_diff(w["peak_zt_high_confidence"], w["peak_zt_all_cells"])
w["frac_cells_kept"] = w["n_cells_high_confidence"] / w["n_cells_all_cells"]
write_csv(w, MOD, "label_confidence_abundance.csv",
          "Abundance rhythms from all cells vs high-confidence cells (label-support margin > 0)",
          required=["tissue", "cell_type", "peak_shift_h", "p_ols_pct_all_cells", "p_ols_pct_high_confidence"])

# (b) gene rhythms in key BM cell types
tiers = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "peak_zt", "p"])
key = tiers[tiers["tier"] == "tier1"]["group"].value_counts().head(6).index.tolist()
conf = set(qc.loc[qc["confident"], "cell_id"])
grow = []
for ct in key:
    genes = tiers[(tiers["group"] == ct) & (tiers["tier"] == "tier1")]["gene"].tolist()
    m = (a.obs["tissue"] == "bone marrow").to_numpy() & (a.obs["cell_type"].astype(str) == ct).to_numpy()
    idx = np.where(m)[0]
    keep = np.array([c in conf for c in a.obs_names[idx]])
    for label, ii in [("all_cells", idx), ("high_confidence", idx[keep])]:
        sub = a[ii]
        samples = sorted(sub.obs["batch_id"].astype(str).unique(), key=lambda s: zt_of[s])
        X = sub.X.tocsr()
        pb = np.vstack([np.asarray(X[(sub.obs["batch_id"].astype(str) == s).to_numpy()].sum(0)).ravel() for s in samples])
        cpm = pb / pb.sum(1, keepdims=True) * 1e6
        gi = [sub.var_names.get_loc(g) for g in genes if g in sub.var_names]
        Y = np.log2(cpm[:, gi] + 1)
        f = cosinor.fit(Y, np.array([zt_of[s] for s in samples], float))
        for j, g_i in enumerate(gi):
            grow.append({"cell_type": ct, "gene": sub.var_names[g_i], "cell_set": label, "n_cells": len(ii),
                         "peak_zt": float(f["peak"][j]), "p": float(f["p"][j])})
    log(f"{ct}: {keep.mean():.2f} of cells high-confidence")
g = pd.DataFrame(grow).pivot_table(index=["cell_type", "gene"], columns="cell_set", values=["peak_zt", "p", "n_cells"])
g.columns = [f"{a_}_{b_}" for a_, b_ in g.columns]
g = g.reset_index()
g["peak_shift_h"] = phase.circ_diff(g["peak_zt_high_confidence"], g["peak_zt_all_cells"])
write_csv(g, MOD, "label_confidence_genes.csv",
          "BM Tier-1 gene rhythms (OLS on log2 CPM) from all vs high-confidence cells of the main rhythmic cell types",
          required=["cell_type", "gene", "p_all_cells", "p_high_confidence", "peak_shift_h"])
s = g.groupby("cell_type").agg(n=("gene", "size"), frac_p05_all=("p_all_cells", lambda x: (x < 0.05).mean()),
                               frac_p05_conf=("p_high_confidence", lambda x: (x < 0.05).mean()),
                               median_abs_shift_h=("peak_shift_h", lambda x: np.median(np.abs(x))))
log("\n" + s.round(3).to_string())
log("done")
