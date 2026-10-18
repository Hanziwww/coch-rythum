"""Round 6, WP-M7 — synthesis of the myeloid hypotheses M1–M8 (D45 decision rule) and summary tables.

Decision rule (D45, pre-specified): per hypothesis, the per-tissue primary exact permutation p values are combined by
Fisher (M2: paired contrast, single p; M8, M9: cochlea only) → BH across the 8 hypotheses (9 once M9 is tested by 316).
  supported      = BH q < 0.1 AND >= 2 agreeing methods AND the pre-specified direction (M2: paired difference peaks in
                   ZT12–24; M6: Cxcr4 / Cxcr2 peak within ± 4 h of the M2 timer peak in every tissue; others: none);
  partial        = q < 0.1 or >= 2 agreeing methods, but not the full rule;
  not supported  = neither.
Agreeing methods are counted within a tissue (D48): a hypothesis meets the method criterion when >= 2 distinct
methods agree in the same tissue (the paired unit for M2; cochlea for M8). Every component is listed in
m_multimethod_matrix.csv with its rule and value.
Also: m_timeline_24h.csv (peak phases of all round-6 myeloid quantities, ZT and clock time, with their evidence) and
m_paired_summary.csv (cochlea − BM contrasts and phase differences from 620 / 640 / 650 / 660).
Outputs (results/r6_myeloid/): m_primary_hypotheses.csv, m_multimethod_matrix.csv, m_timeline_24h.csv, m_paired_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r6_myeloid"
R = RESULTS / MOD
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
rd = lambda n: pd.read_csv(R / n)  # noqa: E731
ps, m3, fc, m6 = rd("m_primary_states.csv"), rd("m_primary_m3.csv"), rd("m_primary_function_clock.csv"), rd("m_primary_m6.csv")
axr, pcd, trj, milo = rd("m_axes_rhythm.csv"), rd("m_paired_contrast.csv"), rd("m_trajectory_rhythm.csv"), rd("m_milo_wave.csv")
cmp_, cpair, m2x = rd("m_composition_summary.csv"), rd("m_composition_paired.csv"), rd("m_m2_external_checks.csv")
fnr, fgl, cko = rd("m_function_rhythm.csv"), rd("m_function_global.csv"), rd("m_clock_offsets.csv")
nrr, nint, ext, ours = rd("m_niche_ratio_rhythm.csv"), rd("m_niche_interactions.csv"), rd("m_external_agreeing.csv"), rd("m_external_axes_ours.csv")
conc77 = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_concordance.csv")

HYP = {"M1": "M1 maturation / ageing timing", "M2": "M2 cochlear circulating-like state (paired)", "M3": "M3 granulopoiesis timing",
       "M4": "M4 functional readiness", "M5": "M5 monocyte → macrophage", "M6": "M6 retention / release axis", "M7": "M7 myeloid clocks",
       "M8": "M8 cochlear macrophage identity"}
# ---------------- primary p values per tissue
prim = []
def addp(h, t, p, stat, pattern_r=np.nan, peak=np.nan):
    prim.append({"hypothesis": h, "tissue": t, "primary_p": float(p) if pd.notna(p) else np.nan, "primary_statistic": stat,
                 "peak_zt": peak, "r_with_sample_pattern_pc1": pattern_r})


def axrow(t, ax, g):
    r = axr[(axr["tissue"] == t) & (axr["axis"] == ax) & (axr["group"] == g) & (axr["subset"] == "all cells")]
    return r.iloc[0]


for t in ["bm", "cochlea"]:
    r = ps[(ps["tissue"] == t) & ps["hypothesis"].str.startswith("M1")].iloc[0]
    addp("M1", t, r["primary_p"], r["primary_statistic"], axrow(t, "maturation (Xie)", "pooled neutrophils")["r_with_sample_pattern_pc1"], r["peak_zt"])
    r = m3[m3["tissue"] == t].iloc[0]
    tr = trj[(trj["tissue"] == t) & trj["feature"].str.contains(r"\[M3 primary\]", regex=True)].iloc[0]
    addp("M3", t, r["primary_p"], r["primary_statistic"], tr["r_with_sample_pattern_pc1"], r["peak_zt"])
    r = fc[(fc["tissue"] == t) & fc["hypothesis"].str.startswith("M4")].iloc[0]
    addp("M4", t, r["primary_p"], r["primary_statistic"])
    r = ps[(ps["tissue"] == t) & ps["hypothesis"].str.startswith("M5")].iloc[0]
    addp("M5", t, r["primary_p"], r["primary_statistic"], axrow(t, "mono→mac (curated)", "monocytes + macrophages")["r_with_sample_pattern_pc1"], r["peak_zt"])
    r = m6[m6["tissue"] == t].iloc[0]
    nr = nrr[(nrr["tissue"] == t) & (nrr["score"] == "cxcr4_cxcr2") & (nrr["subset"] == "all cells")].iloc[0]
    addp("M6", t, r["primary_p"], r["primary_statistic"], nr["r_with_sample_pattern_pc1"], r["ratio_peak_zt"])
    r = fc[(fc["tissue"] == t) & fc["hypothesis"].str.startswith("M7")].iloc[0]
    addp("M7", t, r["primary_p"], r["primary_statistic"])
r = ps[ps["hypothesis"].str.startswith("M2")].iloc[0]
addp("M2", "paired", r["primary_p"], r["primary_statistic"], np.nan, r["peak_zt"])
r = ps[ps["hypothesis"].str.startswith("M8")].iloc[0]
addp("M8", "cochlea", r["primary_p"], r["primary_statistic"], axrow("cochlea", "resident macrophage (cochlear)", "Macrophage")["r_with_sample_pattern_pc1"], r["peak_zt"])
m9p = RESULTS / "r3_ma2025/m9_tests.csv"                 # M9 (D46): Ma 2025, tested by 316 once the matrices are in
m9 = pd.read_csv(m9p) if m9p.exists() else pd.DataFrame()
if len(m9):
    HYP["M9"] = "M9 Ma 2025 noise-exposure readiness vs time of day (cochlea)"
    r = m9[m9["component"] == "primary"].iloc[0]
    addp("M9", "cochlea", r["p_perm_one_sided"], r["statistic"])
prim = pd.DataFrame(prim)

# ---------------- agreeing methods (in-house + external)
mm = []
def add(h, t, method, agrees, value, rule, source):
    mm.append({"hypothesis": h, "tissue": t, "method": method, "agrees": bool(agrees), "value": value, "rule": rule, "source": source})


for t in ["bm", "cochlea"]:
    # M1
    r = axrow(t, "maturation (Xie)", "pooled neutrophils")
    add("M1", t, "Wasserstein (OT) shift of the per-cell maturation axis", r["ot_p_perm"] <= 0.05, f"OT p {r['ot_p_perm']:.3f}", "OT exact p <= 0.05", "640 m_axes_rhythm")
    mw = milo[(milo["tissue"] == t) & (milo["selection"] == "all neutrophil-lineage neighbourhoods")].iloc[0]
    for h in ["M1", "M3"]:
        add(h, t, "Milo neighbourhood wave along neutrophil pseudotime (label-free)", mw["p_perm"] <= 0.05, f"r {mw['circ_linear_r']:.3f}, p {mw['p_perm']:.4f}",
            "circular–linear r permutation p <= 0.05", "630 m_milo_wave")
    # M3
    cr = trj[(trj["tissue"] == t) & trj["feature"].str.startswith("CellRank neutrophil fate")].iloc[0]
    add("M3", t, "CellRank neutrophil-fate probability of GMPs by sample", cr["p_perm"] <= 0.05, f"p {cr['p_perm']:.3f}, peak ZT{cr['peak_zt']:.1f}",
        "exact p <= 0.05", "630 m_trajectory_rhythm")
    cw = cmp_[(cmp_["tissue"] == t) & (cmp_["frame"] == "within_neutrophil") & cmp_["feature"].isin(["Neutrophil (Band)", "Neutrophil (Mature)"])]
    add("M3", t, "within-neutrophil composition multi-model (band / mature shares)", cw["verdict"].isin(["robust", "supported"]).any(),
        "; ".join(f"{a}: {b} (ZT{c:.1f})" for a, b, c in zip(cw["feature"], cw["verdict"], cw["peak_zt"])), "verdict robust or supported", "620 m_composition_summary")
    # M4
    for ver, kind, lab in [("primary", "centred", "within-sample-centred version"), ("depth_matched", "uncentred", "depth-matched version")]:
        g = fgl[(fgl["tissue"] == t) & (fgl["version"] == ver) & (fgl["kind"] == kind)].iloc[0]
        add("M4", t, f"global readiness test, {lab}", g["p_global"] <= 0.05, f"{g['n_rhythmic_p_perm_le_0.05']} of {g['n_programmes']}, p {g['p_global']:.3f}",
            "global exact p <= 0.05", "645 m_function_global")
    st_ = fnr[(fnr["tissue"] == t) & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred") & fnr["unit"].isin(NEU + MONO) & (fnr["p_perm"] <= 0.05)]
    z, p_ray = phase.rayleigh(st_["peak_zt"].to_numpy()) if len(st_) >= 5 else (np.nan, np.nan)
    add("M4", t, "per-state programme peaks clustered (Rayleigh)", pd.notna(p_ray) and p_ray < 0.05,
        f"{len(st_)} state × programme rhythms (p <= 0.05); Rayleigh p {p_ray:.3g}" if pd.notna(p_ray) else f"{len(st_)} rhythms (< 5)",
        "Rayleigh p < 0.05 over >= 5 rhythmic state × programme peaks", "645 m_function_rhythm")
    # M5
    cr = trj[(trj["tissue"] == t) & trj["feature"].str.startswith("CellRank macrophage fate")].iloc[0]
    add("M5", t, "CellRank macrophage-fate probability of classical monocytes by sample", cr["p_perm"] <= 0.05, f"p {cr['p_perm']:.3f}, peak ZT{cr['peak_zt']:.1f}",
        "exact p <= 0.05", "630 m_trajectory_rhythm")
    cw = cmp_[(cmp_["tissue"] == t) & (cmp_["frame"] == "within_mono_mac") & (cmp_["feature"] == "Monocyte (Classical)")].iloc[0]
    add("M5", t, "composition multi-model (classical monocytes within monocytes + macrophages)", cw["verdict"] in ("robust", "supported"),
        f"{cw['verdict']} (ZT{cw['peak_zt']:.1f})", "verdict robust or supported", "620 m_composition_summary")
    r = axrow(t, "mono→mac (curated)", "monocytes + macrophages")
    add("M5", t, "Wasserstein (OT) shift of the mono→mac axis", r["ot_p_perm"] <= 0.05, f"OT p {r['ot_p_perm']:.3f}", "OT exact p <= 0.05", "640 m_axes_rhythm")
    # M6
    r = m6[m6["tissue"] == t].iloc[0]
    for col, lab in [("agree_sender_cxcl12", "Cxcl12 mRNA rhythm in the representative sender"), ("agree_nichenet_cxcl12", "NicheNet CXCL12 activity percentile >= 0.9"),
                     ("agree_gc_cue", "glucocorticoid-response phase aligned with Cxcr4 / Cxcr2")]:
        add("M6", t, lab, bool(r[col]), str(r[col]), "see 650 docstring", "650 m_primary_m6")
    if t == "bm":
        add("M6", t, "GSE198770 whole-BM Cxcl12 rhythm", bool(r["agree_gse198770_cxcl12"]), str(r["agree_gse198770_cxcl12"]), "cosinor p < 0.05", "650 m_niche_external")
    # M7
    co = cko[(cko["tissue"] == t) & cko["unit"].isin(["neutrophils (pooled)", "monocytes + macrophages (pooled)", "myeloid (pooled)"])]
    k = int((co["ci_halfwidth_h"] <= 3).sum())
    add("M7", t, "pseudobulk clock offsets determinate (gene bootstrap CI half-width <= 3 h)", k >= 2,
        "; ".join(f"{u}: {o:+.1f} ± {c:.1f} h" for u, o, c in zip(co["unit"], co["offset_h"], co["ci_halfwidth_h"])), ">= 2 of 3 myeloid units determinate",
        "645 m_clock_offsets")
# M8
r = axrow("cochlea", "resident macrophage (cochlear)", "Macrophage")
add("M8", "cochlea", "Wasserstein (OT) shift of the resident axis", r["ot_p_perm"] <= 0.05, f"OT p {r['ot_p_perm']:.3f}", "OT exact p <= 0.05", "640 m_axes_rhythm")
r = pcd[(pcd["axis"] == "resident macrophage (cochlear)") & (pcd["group"] == "Macrophage")].iloc[0]
add("M8", "cochlea", "cochlea − BM macrophage contrast of the resident axis", r["p_perm"] <= 0.05, f"p {r['p_perm']:.3f}; cochlea higher at {int(r['n_zt_cochlea_higher'])}/6 ZT",
    "paired-difference exact p <= 0.05", "640 m_paired_contrast")
# M2 (paired)
both = m2x.iloc[:2]
add("M2", "paired", "per-tissue timer-score phases in the expected halves (cochlea ZT12–24, BM ZT0–12)", bool(both["agrees"].all()),
    "; ".join(f"{c.split(' timer')[0]} ZT{v:.1f} (p {p:.3f})" for c, v, p in zip(both["check"], both["value"], both["p"])), "both tissues", "640 m_m2_external_checks")
cc = conc77[conc77["cell_type"].isin(NEU)].pivot_table(index="cell_type", columns="tissue", values=["spearman_all", "p_perm_all"])
k = int(sum((cc[("spearman_all", "bm")] < 0) & (cc[("p_perm_all", "bm")] < 0.05) & (cc[("spearman_all", "cochlea")] > 0) & (cc[("p_perm_all", "cochlea")] < 0.05)))
add("M2", "paired", "per-state transcriptome-wide timer projection (77 / R14): BM negative, cochlea positive", k >= 3, f"{k} of 4 neutrophil states split",
    ">= 3 of 4 states with BM r < 0 and cochlea r > 0 (each p < 0.05)", "77 neutrophil_timer_concordance")
b = m2x[m2x["check"].str.startswith("Ballesteros")].iloc[0]
add("M2", "paired", "Ballesteros blood neutrophils: timer score ZT13 > ZT5 (independent scRNA)", bool(b["agrees"]), f"Δ {b['value']:.3f}, p {b['p']:.3g}",
    "Mann–Whitney p < 0.05 and both ZT13 samples above ZT5", "640 m_m2_external_checks")
for r in ext[ext["hypothesis"].isin(list(HYP))].itertuples():
    add(r.hypothesis, r.tissue, r.method, str(r.agrees) == "True", r.detail, "see 660 docstring", "660 m_external_agreeing")
for r in m9[m9["component"].str.startswith("agreeing")].itertuples() if len(m9) else []:
    add("M9", "cochlea", r.component.replace("agreeing: ", ""), str(r.agrees) == "True", r.detail, r.statistic, "316 m9_tests")
mm = pd.DataFrame(mm)

# ---------------- decision
rows = []
for h, name in HYP.items():
    pp = prim[prim["hypothesis"] == h]
    pv = pp["primary_p"].dropna().to_numpy()
    p_comb = float(stats.chi2.sf(-2 * np.log(pv).sum(), 2 * len(pv))) if len(pv) > 1 else (float(pv[0]) if len(pv) else np.nan)
    m_ = mm[mm["hypothesis"] == h]
    per_t = m_.groupby("tissue")["agrees"].sum()
    best_t = per_t.idxmax() if len(per_t) else None
    if h == "M2":
        dir_ok, dir_txt = bool(12 <= pp["peak_zt"].iloc[0] < 24), f"paired difference peaks ZT{pp['peak_zt'].iloc[0]:.1f} (needs ZT12–24)"
    elif h == "M6":
        d6 = m6.set_index("tissue")
        dir_ok = bool(d6["direction_ok"].astype(str).eq("True").all())
        dir_txt = "; ".join(f"{t}: Cxcr4/Cxcr2 − timer lag {d6.loc[t, 'lag_ratio_minus_timer_h']:+.1f} h" for t in d6.index) + " (needs |lag| <= 4 h in every tissue)"
    elif h == "M9":
        r9 = m9[m9["component"] == "primary"].iloc[0]
        dir_ok, dir_txt = bool(r9["r"] > 0), f"Spearman {r9['r']:+.2f} (needs > 0)"
    else:
        dir_ok, dir_txt = True, "no direction pre-specified"
    rows.append({"hypothesis": h, "name": name, "tissues": "; ".join(pp["tissue"]), "p_by_tissue": "; ".join(f"{t} {p:.3f}" for t, p in zip(pp["tissue"], pp["primary_p"])),
                 "combined_p": p_comb, "combination": "Fisher" if len(pv) > 1 else "single", "n_agreeing_methods_best_tissue": int(per_t.max()) if len(per_t) else 0,
                 "best_tissue": best_t, "agreeing_methods_best_tissue": "; ".join(m_[(m_["tissue"] == best_t) & m_["agrees"]]["method"]) if best_t else "",
                 "n_methods_evaluated": len(m_), "n_agreeing_any_tissue": int(m_.groupby("method")["agrees"].any().sum()),
                 "direction_ok": dir_ok, "direction_detail": dir_txt,
                 "max_abs_r_sample_pattern": float(pp["r_with_sample_pattern_pc1"].abs().max()) if pp["r_with_sample_pattern_pc1"].notna().any() else np.nan})
dec = pd.DataFrame(rows)
dec["q_BH"] = multipletests(dec["combined_p"].fillna(1.0), method="fdr_bh")[1]
dec["methods_ok"] = dec["n_agreeing_methods_best_tissue"] >= 2
dec["q_ok"] = dec["q_BH"] < 0.1
dec["decision"] = np.where(dec["q_ok"] & dec["methods_ok"] & dec["direction_ok"], "supported",
                           np.where(dec["q_ok"] | dec["methods_ok"], "partial", "not supported"))
dec["note"] = np.where(dec["max_abs_r_sample_pattern"] >= 0.9, "primary statistic tracks the sample pattern (|r| >= 0.9; 210 PC1)", "")
# robustness (reported, not part of D45): the state-axis primaries after within-sample centring (640 pseudobulk; removes
# shifts shared by all cell types of a sample)
CEN = {"M1": ("maturation (Xie)", "pooled neutrophils"), "M5": ("mono→mac (curated)", "monocytes + macrophages"),
       "M8": ("resident macrophage (cochlear)", "Macrophage")}
cen_txt = {}
for h, (ax, g) in CEN.items():
    c = axr[(axr["axis"] == ax) & (axr["group"] == g) & (axr["subset"] == "pseudobulk, within-sample centred")]
    if h == "M8":
        c = c[c["tissue"] == "cochlea"]
    cen_txt[h] = "; ".join(f"{r.tissue} ZT{r.peak_zt:.1f} p {r.p_perm:.3f}" for r in c.itertuples())
c = pcd[(pcd["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (pcd["group"] == "pooled neutrophils (pseudobulk, within-sample centred)")]
t2 = axr[(axr["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (axr["group"] == "pooled neutrophils") & (axr["subset"] == "pseudobulk, within-sample centred")]
if len(c):
    cen_txt["M2"] = (f"paired ZT{c['peak_zt'].iloc[0]:.1f} p {c['p_perm'].iloc[0]:.3f}; " +
                     "; ".join(f"{r.tissue} ZT{r.peak_zt:.1f} p {r.p_perm:.3f}" for r in t2.itertuples()))
dec["within_sample_centred_check"] = dec["hypothesis"].map(cen_txt).fillna("")
if not len(m9):
    dec = pd.concat([dec, pd.DataFrame([{"hypothesis": "M9", "name": "M9 Ma 2025 noise-exposure readiness vs time of day (cochlea)", "decision": "pending",
                                         "note": "tested when the Ma 2025 matrices are uploaded (O01); joins the BH set"}])], ignore_index=True)

# ---------------- 24 h timeline
tl = []
def tadd(t, domain, quantity, pk, p, evidence, source):
    if pd.notna(pk):
        tl.append({"tissue": t, "domain": domain, "quantity": quantity, "peak_zt": float(pk), "peak_clock": float(phase.zt_to_clock(pk)),
                   "p_perm": p, "evidence": evidence, "source": source})


for r in cmp_[cmp_["frame"].isin(["within_myeloid", "within_neutrophil", "within_mono_mac", "ilr_balance"])].itertuples():
    tadd(r.tissue, f"composition ({r.frame})", r.feature, r.peak_zt, r.p_perm, r.verdict, "620")
for r in axr[(axr["subset"] == "all cells") & axr["group"].isin(["pooled neutrophils", "monocytes + macrophages", "Macrophage"])].itertuples():
    tadd(r.tissue, "state axis", f"{r.axis} | {r.group}", r.peak_zt, r.p_perm, f"OT p {r.ot_p_perm:.3f}", "640")
for r in trj.itertuples():
    tadd(r.tissue, "trajectory", r.feature, r.peak_zt, r.p_perm, f"OT p {r.ot_p_perm:.3f}", "630")
for r in fnr[(fnr["version"] == "primary") & (fnr["kind"] == "uncentred") & (fnr["unit"] == "neutrophils + monocytes (pooled)")].itertuples():
    tadd(r.tissue, "function programme", r.programme, r.peak_zt, r.p_perm, f"LOTO SD {r.loto_phase_sd_h:.1f} h", "645")
for r in nrr[(nrr["subset"] == "all cells") & nrr["tissue"].isin(["bm", "cochlea"])].itertuples():
    tadd(r.tissue, "retention / release", f"{r.score} (pooled neutrophils, per-sample mean)", r.peak_zt, r.p_perm, f"OT p {r.ot_p_perm:.3f}", "650")
for r in nint[nint["magnitude_rank_in_axis"] == 1].itertuples():
    tadd(r.tissue, "retention / release", f"{r.axis}: {r.source} → {r.target}", r.peak_zt, r.p_perm, f"mechanism {r.mechanism}", "650")
for r in ours[ours["kind"] == "rhythm of per-sample medians"].itertuples():
    tadd(r.tissue, "state axis (NeuMAP)", f"{r.axis} | {r.group}", r.peak_zt, r.p_perm, f"OT p {r.ot_p_perm:.3f}", "660")
tl = pd.DataFrame(tl).sort_values(["tissue", "peak_zt"])

# ---------------- paired summary
pr = []
for r in pcd.itertuples():
    pr.append({"source": "640 axes", "quantity": f"{r.axis} | {r.group}", "difference_peak_zt": r.peak_zt, "difference_p_perm": r.p_perm,
               "mean_cochlea_minus_bm": r.mean_difference, "n_zt_cochlea_higher": r.n_zt_cochlea_higher})
for r in cpair.itertuples():
    pr.append({"source": f"620 composition ({r.frame})", "quantity": r.feature, "peak_bm_zt": r.peak_bm_zt, "peak_cochlea_zt": r.peak_cochlea_zt,
               "phase_diff_cochlea_minus_bm_h": r.diff_cochlea_minus_bm_h, "verdict_bm": r.verdict_bm, "verdict_cochlea": r.verdict_cochlea})
for r in nrr[nrr["tissue"].str.startswith("paired")].itertuples():
    pr.append({"source": "650 retention / release", "quantity": f"{r.score} (per-sample mean)", "difference_peak_zt": r.peak_zt, "difference_p_perm": r.p_perm,
               "mean_cochlea_minus_bm": r.mean_difference, "n_zt_cochlea_higher": r.n_zt_cochlea_higher})
for r in ours[ours["kind"] == "cochlea − BM per ZT"].itertuples():
    pr.append({"source": "660 NeuMAP axes", "quantity": r.axis, "mean_cochlea_minus_bm": r.value, "n_zt_cochlea_higher": r.n_zt_cochlea_higher,
               "sign_test_p": r.p_sign_test})
pr = pd.DataFrame(pr)

write_csv(dec, MOD, "m_primary_hypotheses.csv", "Round-6 primary hypotheses M1–M8 (M9 pending): Fisher-combined primary p, BH q, agreeing methods, direction, decision (D45 / D48)",
          required=["hypothesis", "combined_p", "q_BH", "n_agreeing_methods_best_tissue", "direction_ok", "decision"])
write_csv(mm, MOD, "m_multimethod_matrix.csv", "Agreeing-method components per hypothesis × tissue with rule, value and source",
          required=["hypothesis", "tissue", "method", "agrees", "rule", "source"])
write_csv(tl, MOD, "m_timeline_24h.csv", "24 h timeline of round-6 myeloid quantities (peak ZT and clock time with evidence)",
          required=["tissue", "domain", "quantity", "peak_zt", "peak_clock", "p_perm"])
write_csv(pr, MOD, "m_paired_summary.csv", "Cochlea − BM paired contrasts and phase differences across round-6 analyses", required=["source", "quantity"])
log("\n" + prim.round(3).to_string(max_colwidth=50))
log("\n" + mm.to_string(max_colwidth=70))
log("\n" + dec.drop(columns=["agreeing_methods_best_tissue"]).round(4).to_string(max_colwidth=70))
