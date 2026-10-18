"""Round 3, WP2 — response of the cochlea to noise in external single-cell data, in our label space.

Usage: python scripts/310_external_sc_response.py <key>      key ∈ {gse346691, gse167078, ma2025}
Contrasts: gse346691 WT_noise vs WT_control (3 vs 2; knockouts descriptive, n = 1); gse167078 sham / 3 / 7 / 14 d
(one hashed pool each: descriptive); ma2025 day_noise vs control, night_noise vs control, night_noise vs day_noise.
(a) Composition: share of each transferred label per sample (all cells; confident cells as sensitivity); per contrast
    log2 ratio of group means and, with >= 2 samples per group, a Welch t test on logit shares (BH).
(b) Link to our cochlear oscillations: each label's noise log2 ratio next to our unadjusted cochlear abundance rhythm
    (r2_figure: peak ZT, verdict). Ma 2025 only: the night/day difference expected from exposure times (ZT2–4 vs
    ZT14–16) = log2 of our fitted cochlear share at ZT15 vs ZT3, correlated (Spearman) with the observed
    night_noise / day_noise log2 ratio across immune labels; permutation over labels (10,000).
(c) Pseudobulk export (per sample × label with >= 20 cells) for limma-voom (311_external_pseudobulk_de.R).
Outputs (results/r3_<key>/): composition_by_sample.csv, composition_contrasts.csv, composition_rhythm_links.csv;
cache/r3_query/<key>_pseudobulk.parquet + <key>_pseudobulk_design.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

KEY = sys.argv[1]
MOD = f"r3_{KEY}"
W = phase.W
CONTRASTS = {"gse346691": [("WT_noise", "WT_control"), ("Spp1_KO_noise", "WT_noise"), ("Ccr2_KO_noise", "WT_noise")],
             "gse167078": [("noise_3d", "sham"), ("noise_7d", "sham"), ("noise_14d", "sham")],
             "ma2025": [("day_noise", "control"), ("night_noise", "control"), ("night_noise", "day_noise")]}[KEY.replace("ma2025dry", "ma2025")]
q = ad.read_h5ad(CACHE / "r3_query" / f"{KEY}_mapped.h5ad")
cls = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
IMMUNE = set(cls[cls.isin(["Myeloid", "Lymphoid", "Progenitor"])].index)

comp, crow = [], []
for subset, obs in [("all cells", q.obs), ("confident cells", q.obs[q.obs["confident"].astype(bool)])]:
    tab = pd.crosstab(obs["sample"], obs["label"])
    share = tab.div(tab.sum(1), axis=0)
    grp = obs.drop_duplicates("sample").set_index("sample")["condition"]
    for s in tab.index:
        for lab in tab.columns:
            comp.append({"subset": subset, "sample": s, "condition": grp[s], "label": lab, "n_cells": int(tab.loc[s, lab]),
                         "share": float(share.loc[s, lab])})
    for a_, b_ in CONTRASTS:
        sa, sb = grp.index[grp == a_], grp.index[grp == b_]
        if not len(sa) or not len(sb):
            continue
        for lab in tab.columns:
            xa, xb = share.loc[sa, lab].to_numpy(), share.loc[sb, lab].to_numpy()
            na, nb = tab.loc[sa].sum(1).to_numpy(), tab.loc[sb].sum(1).to_numpy()
            la = np.log((xa * na + 0.5) / (na - xa * na + 0.5))
            lb = np.log((xb * nb + 0.5) / (nb - xb * nb + 0.5))
            p = float(stats.ttest_ind(la, lb, equal_var=False).pvalue) if len(sa) >= 2 and len(sb) >= 2 else np.nan
            crow.append({"subset": subset, "contrast": f"{a_}_vs_{b_}", "label": lab, "immune": lab in IMMUNE,
                         "mean_share_a": float(xa.mean()), "mean_share_b": float(xb.mean()),
                         "log2_ratio": float(np.log2((xa.mean() + 1e-4) / (xb.mean() + 1e-4))), "n_a": len(sa), "n_b": len(sb),
                         "p_welch_logit": p})
comp = pd.DataFrame(comp)
cc = pd.DataFrame(crow)
cc["q_BH"] = np.nan
for k, g in cc.groupby(["subset", "contrast"]):
    m = g["p_welch_logit"].notna()
    if m.any():
        cc.loc[g.index[m], "q_BH"] = multipletests(g.loc[m, "p_welch_logit"], method="fdr_bh")[1]
write_csv(comp, MOD, "composition_by_sample.csv", f"{KEY}: share of each transferred label per sample (figure-ready)",
          required=["subset", "sample", "condition", "label", "share"])
write_csv(cc, MOD, "composition_contrasts.csv", f"{KEY}: composition contrasts (log2 ratio of mean shares; Welch t on logit shares)",
          required=["subset", "contrast", "label", "log2_ratio"])

# ---- (b) links to our cochlear oscillations (unadjusted shares, r2_figure)
pr = pd.read_csv(RESULTS / "r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")
ms = pd.read_csv(RESULTS / "r2_figure/abundance_multimodel_summary.csv").query("tissue == 'cochlea'").set_index("cell_type")


def fitted_share(ct, zt):
    r = pr.loc[ct]
    pk = r["peak_zt"]
    return float(r["mesor_pct"] + r["amplitude_pct"] * np.cos(W * (zt - pk)))


links = []
for (sub, con), g in cc.groupby(["subset", "contrast"]):
    for r in g.itertuples():
        if r.label not in pr.index:
            continue
        row = {"subset": sub, "contrast": con, "label": r.label, "immune": r.immune, "noise_log2_ratio": r.log2_ratio,
               "p_welch_logit": r.p_welch_logit, "our_peak_zt": pr.loc[r.label, "peak_zt"],
               "our_peak_clock": pr.loc[r.label, "peak_clock"], "our_p_ols": pr.loc[r.label, "p_ols"],
               "our_verdict": ms["verdict"].get(r.label, np.nan)}
        if KEY.startswith("ma2025"):
            s3, s15 = fitted_share(r.label, 3.0), fitted_share(r.label, 15.0)
            row["our_expected_log2_ZT15_vs_ZT3"] = float(np.log2(max(s15, 1e-3) / max(s3, 1e-3)))
        links.append(row)
lk = pd.DataFrame(links)
if KEY.startswith("ma2025") and len(lk):
    rng = np.random.default_rng(310)
    for sub in lk["subset"].unique():
        x = lk[(lk["subset"] == sub) & (lk["contrast"] == "night_noise_vs_day_noise") & lk["immune"]].dropna(subset=["our_expected_log2_ZT15_vs_ZT3"])
        if len(x) >= 5:
            rho = stats.spearmanr(x["our_expected_log2_ZT15_vs_ZT3"], x["noise_log2_ratio"]).statistic
            null = np.array([stats.spearmanr(rng.permutation(x["our_expected_log2_ZT15_vs_ZT3"].to_numpy()), x["noise_log2_ratio"]).statistic
                             for _ in range(10000)])
            lk.loc[x.index, "spearman_expected_vs_observed_immune"] = rho
            lk.loc[x.index, "p_perm_labels"] = float((1 + (null >= rho).sum()) / 10001)
write_csv(lk, MOD, "composition_rhythm_links.csv",
          f"{KEY}: noise-induced composition changes next to our cochlear abundance oscillations (unadjusted shares)",
          required=["subset", "contrast", "label", "noise_log2_ratio", "our_peak_zt"])

# ---- (c) pseudobulk export
X = q.X.tocsr()
keys = q.obs["sample"].astype(str) + "|" + q.obs["label"].astype(str)
cols, mats = [], []
for k, idx in pd.Series(np.arange(q.n_obs)).groupby(keys.to_numpy()).groups.items():
    if len(idx) < 20:
        continue
    cols.append(k)
    mats.append(np.asarray(X[np.asarray(idx)].sum(0)).ravel())
pb = pd.DataFrame(np.vstack(mats).T, index=q.var_names, columns=cols)
pb.to_parquet(CACHE / "r3_query" / f"{KEY}_pseudobulk.parquet")
des = pd.DataFrame({"column": cols, "sample": [c.split("|")[0] for c in cols], "label": [c.split("|", 1)[1] for c in cols]})
des["condition"] = des["sample"].map(q.obs.drop_duplicates("sample").set_index("sample")["condition"])
des["n_cells"] = [int((keys == c).sum()) for c in cols]
des.to_csv(CACHE / "r3_query" / f"{KEY}_pseudobulk_design.csv", index=False)
log(f"{KEY}: {len(cols)} pseudobulk columns; composition contrasts {len(cc)} rows")
