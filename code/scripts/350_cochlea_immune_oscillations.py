"""Round 3, WP6 — cochlea-only synthesis of cell-type oscillations (user decision: bone marrow is the sole source of
cochlear immune cells, so BM⇄cochlea egress/return orderings are not tested; intra-cochlear marrow content is part of
the cochlear immune compartment → unadjusted shares are primary, marrow-adjusted fits are a sensitivity column).

Per cochlear cell type (immune types flagged): unadjusted share rhythm (r2_figure: peak ZT / clock, amplitude, peak/trough,
95 % phase CI half-width, OLS p, exact permutation p, BH q, LOTO phase SD, bootstrap fraction p < 0.05), 10-model
verdict (202), marrow-adjusted sensitivity (52: P(residual FC > 1.5), residual peak), correlation with the tissue-wide
sample pattern (210 PC1), Milo neighbourhoods (scANVI / scVI: spatial FDR < 0.1 and mean peak), label-free sub-states
with rhythmic shares (332), within-type gene rhythms (Tier 1/2, Tier C), external cochlea signature rhythm (GSE88954
sham; 222), literature phase (222), validation-register verdict (241), and noise responses in external data
(GSE167078 3 d, GSE346691 WT noise vs control, Ma 2025 night vs day noise when available; log2 share ratios).
Outputs (results/r3_cochlea/): cochlea_celltype_oscillations.csv, cochlea_schematic_peaks.csv, cochlea_immune_curves.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r3_cochlea"


def rd(rel, **kw):
    p = RESULTS / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


pr = rd("r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")
ms = rd("r2_figure/abundance_multimodel_summary.csv").query("tissue == 'cochlea'").set_index("cell_type")
mr = rd("52_marrow_decomposition/tissue_residual_rhythm.csv")
mra = mr[mr["model"] == "marrow_adjusted"].set_index("cell_type") if len(mr) else pd.DataFrame()
pts = rd("r2_figure/figure_points.csv").query("tissue == 'cochlea'")
pat = rd("r2_cochlea/sample_pattern_scores.csv").query("tissue == 'cochlea' and pc == 1").set_index("sample_id")["score"]
milo = {e: rd(f"53_milo/milo_da_results_cochlea{s}.csv") for e, s in [("scANVI", ""), ("scVI", "_scvi")]}
sub = rd("r3_audit/immune_substates.csv").query("tissue == 'cochlea'")
t33 = rd("33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv", usecols=["group", "tier"])
tcs = rd("r2_cochlea/tierC_celltype_summary.csv").set_index("cell_type")
sig = rd("r2_external/signature_rhythm.csv")
sig = sig[(sig["dataset"] == "GSE88954") & (sig["condition"] == "sham")] if len(sig) else sig
lit = rd("r2_external/literature_comparison.csv")
lit = lit[lit["tissue"] == "cochlea"] if len(lit) else lit
reg = rd("r2_audit/positive_findings_register.csv")
cls = rd("00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
off = rd("r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
noise = {}
for key, con in [("gse167078", "noise_3d_vs_sham"), ("gse346691", "WT_noise_vs_WT_control"), ("ma2025", "night_noise_vs_day_noise"),
                 ("ma2025", "day_noise_vs_control"), ("ma2025", "night_noise_vs_control")]:
    c = rd(f"r3_{key}/composition_contrasts.csv")
    if len(c):
        c = c[(c["subset"] == "all cells") & (c["contrast"] == con)].set_index("label")
        noise[f"{key}:{con}"] = c
rows = []
for ct, r in pr.iterrows():
    immune = cls.get(ct, "") in ("Myeloid", "Lymphoid", "Progenitor")
    row = {"cell_type": ct, "cell_class": cls.get(ct, ""), "immune": immune, "peak_zt": r["peak_zt"], "peak_clock": r["peak_clock"],
           "mean_pct": r["mean_pct"], "amplitude_pct": r["amplitude_pct"], "ptt_ratio": r["ptt_ratio"],
           "peak_ci_halfwidth_h": r["peak_ci_halfwidth_h"], "phase_determinate": r["phase_determinate"], "p_ols": r["p_ols"],
           "p_perm": r["p_perm"], "q_BH": r["q_BH"], "loto_phase_sd_h": r["loto_phase_sd_h"], "boot_frac_p05": r["boot_frac_p05"],
           "multimodel_verdict": ms["verdict"].get(ct, np.nan), "core_models_supporting": ms["core_supporting"].get(ct, np.nan)}
    if ct in getattr(mra, "index", []):
        row.update({"sensitivity_marrow_adjusted_prob_fc_gt_1p5": mra.loc[ct, "prob_fc_gt_1p5"],
                    "sensitivity_marrow_adjusted_peak_zt": mra.loc[ct, "resid_peak_zt"]})
    pp = pts[pts["cell_type"] == ct].set_index("sample_id")["pct"]
    cm = pp.index.intersection(pat.index)
    row["r_with_sample_pattern_pc1"] = float(np.corrcoef(pp[cm], pat[cm])[0, 1]) if len(cm) >= 5 else np.nan
    for e, m in milo.items():
        if len(m):
            x = m[m["majority_cell_type"] == ct]
            s_ = x[x["spatial_fdr"] < 0.1]
            row[f"milo_{e}_n_nhoods"] = len(x)
            row[f"milo_{e}_n_sfdr10"] = len(s_)
            row[f"milo_{e}_mean_peak_zt"] = phase.circ_mean(s_["peak_zt"])[0] if len(s_) else np.nan
    if len(sub):
        x = sub[sub["cell_type"] == ct]
        row["substates_n"] = len(x)
        row["substates_rhythmic_pperm05"] = int((x["p_perm"] <= 0.05).sum())
        row["substates_rhythmic_detail"] = "; ".join(f"{a.substate.split('|')[1]} ZT{a.peak_zt:.1f} [{a.markers.split(';')[0]},{a.markers.split(';')[1]}]"
                                                    for a in x[x["p_perm"] <= 0.05].itertuples())
    row["gene_tier1_2_calls"] = int(t33[(t33["group"] == ct) & t33["tier"].isin(["tier1", "tier2"])].shape[0]) if len(t33) else np.nan
    row["gene_tierC_calls"] = tcs["n_calls"].get(ct, np.nan) if len(tcs) else np.nan
    sname = ("Neutrophil immature (Band/Mature)" if ct in ("Neutrophil (Band)", "Neutrophil (Mature)") else
             "Neutrophil mature (Aged/Activated)" if ct in ("Neutrophil (Aged)", "Neutrophil (Activated)") else ct)
    s_ = sig[sig["signature"] == sname] if len(sig) else sig
    if len(s_):
        row.update({"external_GSE88954_signature_p": float(s_["p"].iloc[0]),
                    "external_delta_internal_h": float(phase.circ_diff(phase.wrap24(r["peak_zt"] - O_C), s_["internal_peak"].iloc[0]))})
    lt = lit[lit["cell_type"] == ct] if len(lit) else lit
    if len(lt):
        row["literature"] = "; ".join(f"{a.source}: exp ZT{a.expected_peak_zt:.0f} ({a.agreement})" for a in lt.itertuples())
    if len(reg):
        rr = reg[(reg["tissue"] == "cochlea") & reg["statement"].str.startswith(ct + ":")]
        row["validation_register_verdict"] = rr["verdict"].iloc[0] if len(rr) else np.nan
    for k, c in noise.items():
        row[f"noise_log2_ratio[{k}]"] = c["log2_ratio"].get(ct, np.nan)
        row[f"noise_p[{k}]"] = c["p_welch_logit"].get(ct, np.nan)
    rows.append(row)
tab = pd.DataFrame(rows).sort_values(["immune", "p_ols"], ascending=[False, True])
write_csv(tab, MOD, "cochlea_celltype_oscillations.csv",
          "Cochlea-only synthesis per cell type: unadjusted share rhythm (primary), marrow-adjusted sensitivity, sub-states, "
          "external / literature support, noise responses", required=["cell_type", "immune", "peak_zt", "peak_clock", "p_ols"])
sch = tab[tab["multimodel_verdict"].isin(["robust", "supported"])].copy()
sch["peak_window_zt"] = [f"ZT{(p - h) % 24:.1f}–ZT{(p + h) % 24:.1f}" for p, h in zip(sch["peak_zt"], sch["peak_ci_halfwidth_h"].clip(upper=12))]
sch["peak_window_clock"] = [f"{float(phase.zt_to_clock(p - h)):.1f}–{float(phase.zt_to_clock(p + h)):.1f} h"
                            for p, h in zip(sch["peak_zt"], sch["peak_ci_halfwidth_h"].clip(upper=12))]
write_csv(sch[["cell_type", "cell_class", "immune", "peak_zt", "peak_clock", "peak_window_zt", "peak_window_clock", "amplitude_pct",
               "ptt_ratio", "multimodel_verdict", "validation_register_verdict"]].sort_values("peak_zt"), MOD, "cochlea_schematic_peaks.csv",
          "Figure-ready cochlea-only schematic: cell types with robust / supported share rhythms, peak (ZT and clock) and 95 % window",
          required=["cell_type", "peak_zt", "peak_clock", "peak_window_zt"])
cur = rd("r2_figure/figure_curves.csv")
if len(cur):
    cur = cur[(cur["tissue"] == "cochlea") & cur["cell_type"].isin(tab.loc[tab["immune"], "cell_type"])]
    write_csv(cur, MOD, "cochlea_immune_curves.csv", "Figure-ready fitted share curves (0.1-h grid, 95 % bands, ZT and clock) of cochlear immune types",
              required=["cell_type", "zt", "clock", "fit_pct", "band_lo", "band_hi"])
log("\n" + tab[tab["immune"]][["cell_type", "peak_zt", "p_ols", "multimodel_verdict", "r_with_sample_pattern_pc1"]].round(3).to_string())
