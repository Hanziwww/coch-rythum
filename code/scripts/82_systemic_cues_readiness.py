"""Module 6.4–6.5 — systemic-cue decomposition and time-of-day immune readiness.

Inputs: results/41_pathway/pathway_rhythm.csv (curated sets, pseudobulk mean-z activity) and
        results/33_rhythm_tiers (gene tiers), results/41_pathway/cell_score_rhythm.csv (AUCell sample medians).
(1) Systemic cues (GC, adrenergic/cAMP, cold-inducible, heat-shock, hypoxia, feeding/insulin-mTOR): per tissue ×
    cell type rhythm; across cell types, amplitude-weighted mean resultant (common phase), Rayleigh p, and V-test
    against the expected peak (GC ZT12 at the light–dark transition; adrenergic ZT14; cold-inducible ZT8 in the late
    rest phase when body temperature is low; heat shock ZT18; feeding/insulin ZT16; hypoxia none).
(2) Attribution: share of each cell type's Tier-1/2 genes that belong to a cue set and peak within ±3 h of the
    cue's common phase in that tissue.
(3) Immune readiness: rhythms of PRR, inflammasome, cytokine, MHC-II, IFN, NF-κB, phagocytosis sets per cell type.
Outputs (results/82_systemic/): systemic_cue_rhythm.csv, systemic_cue_common_phase.csv, systemic_cue_attribution.csv,
                                immune_readiness_rhythm.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "82_systemic"
W = 2 * np.pi / 24
EXPECTED = {"glucocorticoid_response": 12, "adrenergic_camp": 14, "temperature_cold_inducible": 8,
            "heat_shock_hsf1": 18, "feeding_insulin_mtor": 16, "hypoxia": None}
SC = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
pr = pd.read_csv(RESULTS / "41_pathway/pathway_rhythm.csv")
cur = pr[pr["resource"] == "curated"].copy()
cur["category"] = cur["pathway"].str.split(":").str[0]
cur["set"] = cur["pathway"].str.split(":").str[1]
cues = cur[cur["category"] == "systemic_cues"].copy()
write_csv(cues, MOD, "systemic_cue_rhythm.csv", "Rhythm of systemic-cue activity scores per tissue × cell type",
          required=["tissue", "cell_type", "set", "amplitude", "peak_zt", "p", "q_BH"])
rows = []
for (t, s), g in cues.groupby(["tissue", "set"]):
    w = g["amplitude"].to_numpy()
    ang = W * g["peak_zt"].to_numpy()
    z = (w * np.exp(1j * ang)).sum() / w.sum()
    n = len(g)
    R = np.abs(np.exp(1j * ang).mean())
    p_ray = float(np.exp(-n * R ** 2))
    exp_ = EXPECTED.get(s)
    if exp_ is not None:
        u = np.sqrt(2 * n) * np.mean(np.cos(ang - W * exp_))
        p_v = float(1 - norm.cdf(u))
    else:
        u, p_v = np.nan, np.nan
    rows.append({"tissue": t, "set": s, "n_celltypes": n, "weighted_common_peak_zt": (np.angle(z) / W) % 24,
                 "weighted_resultant": float(np.abs(z)), "rayleigh_p": p_ray, "expected_peak_zt": exp_,
                 "v_test_u": u, "v_test_p": p_v, "n_celltypes_q10": int((g["q_BH"] < 0.1).sum())})
common = pd.DataFrame(rows)
write_csv(common, MOD, "systemic_cue_common_phase.csv",
          "Common phase of each systemic cue across cell types (weighted resultant, Rayleigh, V-test vs expectation)",
          required=["tissue", "set", "weighted_common_peak_zt", "rayleigh_p"])

att = []
for t in ["cochlea", "bm"]:
    g = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv",
                    usecols=["group", "gene", "tier", "peak_zt"])
    g = g[g["tier"].isin(["tier1", "tier2"])]
    for s, genes in SC["systemic_cues"].items():
        cphase = common[(common["tissue"] == t) & (common["set"] == s)]["weighted_common_peak_zt"]
        cphase = float(cphase.iloc[0]) if len(cphase) else np.nan
        for ct, gg in g.groupby("group"):
            mem = gg[gg["gene"].isin(genes)]
            d = np.abs(((mem["peak_zt"] - cphase + 12) % 24) - 12) if not np.isnan(cphase) else pd.Series(dtype=float)
            att.append({"tissue": t, "cell_type": ct, "set": s, "n_rhythmic_genes": len(gg),
                        "n_cue_members_rhythmic": len(mem), "n_cue_members_in_common_phase": int((d < 3).sum()),
                        "frac_of_rhythmic_genes": len(mem) / max(len(gg), 1), "cue_common_peak_zt": cphase,
                        "members": ";".join(mem["gene"])})
write_csv(pd.DataFrame(att), MOD, "systemic_cue_attribution.csv",
          "Rhythmic genes per cell type that belong to systemic-cue sets and share the cue's common phase",
          required=["tissue", "cell_type", "set", "n_cue_members_rhythmic"])
imm = cur[cur["category"] == "immune_readiness"]
write_csv(imm, MOD, "immune_readiness_rhythm.csv",
          "Rhythm of immune-readiness programmes (PRR, inflammasome, cytokines, MHC-II, IFN, NF-κB, phagocytosis)",
          required=["tissue", "cell_type", "set", "amplitude", "peak_zt", "p", "q_BH"])
log("done")
