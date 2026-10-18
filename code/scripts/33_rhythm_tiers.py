"""Module 2.5 — merge primary and independent rhythm results, assign evidence tiers, summarise per cell type.

Tier 1: limma BH q < 0.05 & permutation q < 0.10 & peak-to-trough FC >= 1.5 & >= 1 independent method p < 0.05
        (edgeR-QL, RAIN, JTK, LR) & LOTO phase circular SD < 3 h & not single-sample-driven.
Tier 2: limma BH q < 0.10 & FC >= 1.5 (not Tier 1).
Exploratory: limma p < 0.05 (neither tier).
single_sample_driven: nominally rhythmic (p < 0.05) but the rhythm collapses when one ZT is left out
        (LOTO max p > 0.25 or LOTO min amplitude ratio < 0.5).
pi1 (proportion of rhythmic genes) by Storey's estimator on limma p (lambda = 0.5).

Usage: python scripts/33_rhythm_tiers.py [cell_type|cell_class]
Outputs (results/33_rhythm_tiers/): rhythm_genes_all_<level>_<tissue>.csv, rhythm_summary_by_celltype_<level>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CFG, RESULTS, log, write_csv  # noqa: E402

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "cell_type"
MOD = "33_rhythm_tiers"
R = CFG["rhythm"]
glob = pd.read_csv(RESULTS / f"31_rhythm_genes/rhythm_global_test_{LEVEL}.csv")
summ_rows = []
for tissue in ["cochlea", "bm"]:
    lim = pd.read_csv(RESULTS / f"31_rhythm_genes/rhythm_limma_{LEVEL}_{tissue}.csv")
    alt_path = RESULTS / f"32_rhythm_alt/rhythm_alt_{LEVEL}_{tissue}.csv"
    if alt_path.exists():
        alt = pd.read_csv(alt_path).drop(columns=["tissue", "level"], errors="ignore")
        df = lim.merge(alt, on=["group", "gene"], how="left")
    else:
        log(f"WARNING: {alt_path} missing; independent-method columns empty")
        df = lim
    for c in ["rain_p", "jtk_p", "lr_p", "edgeR_p"]:
        if c not in df:
            df[c] = np.nan
    indep = (df[["edgeR_p", "rain_p", "jtk_p", "lr_p"]] < 0.05).any(axis=1)
    df["n_independent_p05"] = (df[["edgeR_p", "rain_p", "jtk_p", "lr_p"]] < 0.05).sum(axis=1)
    df["single_sample_driven"] = (df["p"] < 0.05) & ((df["loto_max_p"] > 0.25) | (df["loto_min_amp_ratio"] < 0.5))
    fc_ok = df["ptt_fc"] >= R["fc_threshold"]
    t1 = ((df["q_BH"] < R["q_tier1"]) & (df["q_perm"] < R["perm_fdr_tier1"]) & fc_ok & indep
          & (df["loto_phase_sd_h"] < R["loto_phase_sd_h"]) & ~df["single_sample_driven"])
    t2 = (df["q_BH"] < R["q_tier2"]) & fc_ok & ~t1
    df["tier"] = np.select([t1, t2, df["p"] < 0.05], ["tier1", "tier2", "exploratory"], default="none")
    write_csv(df, MOD, f"rhythm_genes_all_{LEVEL}_{tissue}.csv",
              f"Per-gene rhythm statistics, independent methods, robustness and evidence tier ({LEVEL}, {tissue})",
              required=["tissue", "group", "gene", "mesor", "amp_log2", "ptt_fc", "peak_zt", "p", "q_BH", "q_perm",
                        "loto_phase_sd_h", "tier"])
    for group, g in df.groupby("group"):
        p = g["p"].dropna()
        pi0 = min(1.0, (p > 0.5).sum() / (0.5 * len(p))) if len(p) else np.nan
        t1g = g[g["tier"] == "tier1"]
        gl = glob[(glob["tissue"] == tissue) & (glob["group"] == group)]
        summ_rows.append({
            "tissue": tissue, "level": LEVEL, "group": group, "status": g["status"].iloc[0],
            "n_cells_min": g["n_cells_min"].iloc[0], "n_cells_total": g["n_cells_total"].iloc[0],
            "n_genes_tested": len(g), "pi1_storey": 1 - pi0,
            "n_tier1": int((g["tier"] == "tier1").sum()), "n_tier2": int((g["tier"] == "tier2").sum()),
            "n_exploratory": int((g["tier"] == "exploratory").sum()),
            "n_single_sample_driven": int(g["single_sample_driven"].sum()),
            "median_ptt_fc_tier1": t1g["ptt_fc"].median() if len(t1g) else np.nan,
            "perm_global_p": gl["p_global"].iloc[0] if len(gl) else np.nan,
            "perm_empirical_fdr": gl["empirical_fdr"].iloc[0] if len(gl) else np.nan,
            "perm_obs_hits_q10_fc15": gl["obs_hits"].iloc[0] if len(gl) else np.nan,
        })
summ = pd.DataFrame(summ_rows).sort_values(["tissue", "n_tier1"], ascending=[True, False])
write_csv(summ, MOD, f"rhythm_summary_by_celltype_{LEVEL}.csv",
          f"Rhythmic-gene counts per tier, pi1 and permutation global tests per tissue × {LEVEL}",
          required=["tissue", "group", "n_genes_tested", "pi1_storey", "n_tier1", "n_tier2", "perm_global_p"])
pd.set_option("display.width", 250)
print(summ.round(3).to_string(index=False))
log("done")
