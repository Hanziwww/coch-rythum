"""Round 2, WP4 — harmonised phase estimates and phase comparisons.

Every key series gets: peak ZT and clock time, a 95 % delta-method phase CI (t with the model's residual df),
the phase-determinate flag (half-width <= 3 h) and an internal (clock-aligned) phase = peak − tissue/dataset offset
(r2_phase/clock_alignment_offsets.csv, tissue pools for our cell types).
Series: abundance rhythms (% of tissue cells; r2_figure), BM Tier-1 genes, cochlear Tier-C genes (211), core-clock
genes in every cell type, external immune-signature rhythms (222).
Comparisons:
  (a) BM vs cochlea for the same cell type and gene (Tier 1 in BM and Tier C or p < 0.05 in the cochlea):
      Δ internal phase with a delta-method CI (independent cohorts, variances add);
  (b) our cell types vs matched external units (GSE198770 control BM for BM types; GSE88954 sham cochlea for cochlear
      types): gene-level Δ internal phase with CI for genes rhythmic in both;
  (c) dispersion of each core-clock gene's peak across cell types within a tissue (resultant length, circular SD;
      one peak per cell type, so no within-group test is possible).
12-h harmonics (external replicated data, r2_external): share of genes with a significant 12-h component.
Outputs (results/r2_phase/): phase_estimates_harmonised.csv, phase_comparisons.csv, clock_phase_equality.csv,
harmonic_12h_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_phase"
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")


def o(dataset, tissue, condition="WT", unit="tissue_pool"):
    r = off[(off["dataset"] == dataset) & (off["tissue"] == tissue) & (off["condition"] == condition) & (off["unit"] == unit)]
    return float(r["offset_canonical_h"].iloc[0]) if len(r) else 0.0


O = {"bm": o("this_atlas", "bm"), "cochlea": o("this_atlas", "cochlea")}
rows = []
ab = pd.read_csv(RESULTS / "r2_figure/abundance_pct_rhythm.csv")
for r in ab.itertuples():
    rows.append({"series": "abundance_pct", "tissue": r.tissue, "cell_type": r.cell_type, "feature": "share of tissue cells",
                 "peak_zt": r.peak_zt, "peak_clock": phase.zt_to_clock(r.peak_zt), "ci_halfwidth_h": r.peak_ci_halfwidth_h,
                 "phase_determinate": r.peak_ci_halfwidth_h <= 3, "internal_phase": phase.wrap24(r.peak_zt - O[r.tissue]),
                 "p": r.p_ols, "evidence": f"OLS-% p {r.p_ols:.3g}; perm p {r.p_perm:.3f}"})
gene_sets = {}
for tissue in ["bm", "cochlea"]:
    t = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                    usecols=["group", "gene", "tier", "a", "b", "se_a", "se_b", "df_total", "p", "ptt_fc"])
    clock = t["gene"].isin(phase.CANONICAL_CLOCK)
    sel = t[(t["tier"] == "tier1") | clock] if tissue == "bm" else t[clock]
    gene_sets[tissue] = t
    pk, half = phase.phase_ci_delta(sel["a"], sel["b"], sel["se_a"] ** 2, sel["se_b"] ** 2, 0.0, df=sel["df_total"])
    for r, p_, h_ in zip(sel.itertuples(), pk, half):
        rows.append({"series": "gene_tier1" if r.tier == "tier1" else "gene_clock", "tissue": tissue, "cell_type": r.group,
                     "feature": r.gene, "peak_zt": p_, "peak_clock": phase.zt_to_clock(p_), "ci_halfwidth_h": h_,
                     "phase_determinate": h_ <= 3, "internal_phase": phase.wrap24(p_ - O[tissue]), "p": r.p,
                     "evidence": f"tier {r.tier}; FC {r.ptt_fc:.2f}"})
tc_path = RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv"
tc = pd.read_csv(tc_path) if tc_path.exists() else pd.DataFrame()
if len(tc):
    tc = tc[tc["tierC"].astype(str) == "True"]
    pk, half = phase.phase_ci_delta(tc["a"], tc["b"], tc["se_a"] ** 2, tc["se_b"] ** 2, 0.0, df=tc["df_total"])
    for r, p_, h_ in zip(tc.itertuples(), pk, half):
        rows.append({"series": "gene_tierC", "tissue": "cochlea", "cell_type": r.cell_type, "feature": r.gene, "peak_zt": p_,
                     "peak_clock": phase.zt_to_clock(p_), "ci_halfwidth_h": h_, "phase_determinate": h_ <= 3,
                     "internal_phase": phase.wrap24(p_ - O["cochlea"]), "p": r.p,
                     "evidence": f"Tier C; pattern-like {r.tissue_pattern_like}"})
sr_path = RESULTS / "r2_external/signature_rhythm.csv"
if sr_path.exists():
    for r in pd.read_csv(sr_path).itertuples():
        rows.append({"series": "external_signature", "tissue": f"{r.dataset}|{r.tissue}|{r.condition}", "cell_type": r.signature,
                     "feature": "signature score", "peak_zt": r.peak_h, "peak_clock": np.nan, "ci_halfwidth_h": np.nan,
                     "phase_determinate": np.nan, "internal_phase": r.internal_peak, "p": r.p,
                     "evidence": f"{r.time_type}; q {r.q_BH:.3f}"})
est = pd.DataFrame(rows)
write_csv(est, MOD, "phase_estimates_harmonised.csv",
          "Harmonised phase estimates (ZT, clock, 95 % CI, determinate flag, internal clock-aligned phase) of key series",
          required=["series", "tissue", "cell_type", "feature", "peak_zt", "ci_halfwidth_h", "internal_phase"])

# ---- (a) BM vs cochlea, same cell type and gene
comp = []
bmt = gene_sets["bm"][gene_sets["bm"]["tier"] == "tier1"]
co = gene_sets["cochlea"]
co_ok = co[(co["p"] < 0.05)]
if len(tc):
    co_ok = pd.concat([co_ok, tc[["cell_type", "gene", "a", "b", "se_a", "se_b", "df_total", "p"]].rename(columns={"cell_type": "group"})]).drop_duplicates(["group", "gene"])
m = bmt.merge(co_ok, on=["group", "gene"], suffixes=("_bm", "_co"))
for r in m.itertuples():
    pb, hb = phase.phase_ci_delta(r.a_bm, r.b_bm, r.se_a_bm ** 2, r.se_b_bm ** 2, 0.0, df=r.df_total_bm)
    pc, hc = phase.phase_ci_delta(r.a_co, r.b_co, r.se_a_co ** 2, r.se_b_co ** 2, 0.0, df=r.df_total_co)
    d = float(phase.circ_diff(pc - O["cochlea"], pb - O["bm"]))
    half = float(np.sqrt(hb ** 2 + hc ** 2))
    comp.append({"comparison": "cochlea_minus_bm_same_celltype", "cell_type": r.group, "feature": r.gene, "delta_h": d,
                 "ci_halfwidth_h": half, "consistent_with_sync": abs(d) <= half, "p_a": r.p_bm, "p_b": r.p_co})
# ---- (b) ours vs matched external units
ext_bm = pd.read_csv(RESULTS / "r2_external/external_rhythm_GSE198770_bone_marrow.csv")
ext_bm = ext_bm[(ext_bm["condition"] == "control") & (ext_bm["p"] < 0.05)].drop_duplicates("gene").set_index("gene")
ext_co = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv")
ext_co = ext_co[(ext_co["condition"] == "sham") & (ext_co["p"] < 0.05)].drop_duplicates("gene").set_index("gene")
for tissue, E, eo, name in [("bm", ext_bm, o("GSE198770_bone_marrow", "bone_marrow", "control", "bulk"), "GSE198770 BM control"),
                            ("cochlea", ext_co, o("GSE88954", "cochlea", "sham", "bulk"), "GSE88954 cochlea sham")]:
    t = gene_sets[tissue]
    t = t[(t["tier"].isin(["tier1", "tier2"]) if tissue == "bm" else (t["p"] < 0.05)) & t["gene"].isin(E.index)]
    for r in t.itertuples():
        e = E.loc[r.gene]
        po, ho = phase.phase_ci_delta(r.a, r.b, r.se_a ** 2, r.se_b ** 2, 0.0, df=r.df_total)
        pe, he = phase.phase_ci_delta(e["a"], e["b"], e["se_a"] ** 2, e["se_b"] ** 2, 0.0, df=None)
        d = float(phase.circ_diff(po - O[tissue], pe - eo))
        half = float(np.sqrt(ho ** 2 + he ** 2))
        comp.append({"comparison": f"ours_minus_external ({name})", "cell_type": r.group, "feature": r.gene, "delta_h": d,
                     "ci_halfwidth_h": half, "consistent_with_sync": abs(d) <= half, "p_a": r.p, "p_b": e["p"]})
comp = pd.DataFrame(comp)
write_csv(comp, MOD, "phase_comparisons.csv",
          "Gene-level clock-aligned phase differences with CIs (BM vs cochlea; ours vs matched external units)",
          required=["comparison", "cell_type", "feature", "delta_h", "ci_halfwidth_h"])

# ---- (c) dispersion of clock-gene phases across cell types
eq = []
for tissue, t in gene_sets.items():
    t = t[t["gene"].isin(phase.CANONICAL_CLOCK) & (t["p"] < 0.05)]
    for g, h in t.groupby("gene"):
        pk = phase.wrap24(np.degrees(np.arctan2(h["b"], h["a"])) / 15)
        if len(h) < 3:
            continue
        m_, R = phase.circ_mean(pk)
        eq.append({"tissue": tissue, "gene": g, "n_celltypes_p05": len(h), "circ_mean_peak_zt": m_, "resultant": R,
                   "circ_sd_h": phase.circ_sd(pk), "canonical_peak_zt": phase.CANONICAL_CLOCK[g],
                   "cell_types": ";".join(h["group"])})
write_csv(pd.DataFrame(eq), MOD, "clock_phase_equality.csv",
          "Dispersion of core-clock gene peaks across cell types (resultant length, circular SD) per tissue",
          required=["tissue", "gene", "circ_mean_peak_zt", "resultant"])

# ---- 12-h harmonics in replicated external data
hr = []
for f in sorted((RESULTS / "r2_external").glob("external_rhythm_*.csv")):
    d = pd.read_csv(f)
    if "p_12h" not in d:
        continue
    for (t, c), g in d.groupby(["tissue", "condition"]):
        hr.append({"dataset": d["dataset"].iloc[0], "tissue": t, "condition": c, "n_genes": len(g),
                   "n_24h_q05": int((g["q_BH"] < 0.05).sum()), "n_12h_q05": int((g["q_12h"] < 0.05).sum()),
                   "frac_12h_among_24h": float(((g["q_12h"] < 0.05) & (g["q_BH"] < 0.05)).sum() / max((g["q_BH"] < 0.05).sum(), 1))})
write_csv(pd.DataFrame(hr), MOD, "harmonic_12h_summary.csv", "12-h harmonic components in replicated external datasets",
          required=["dataset", "tissue", "condition", "n_12h_q05"])
log("done")
