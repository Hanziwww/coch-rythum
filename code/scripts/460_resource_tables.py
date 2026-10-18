"""Round 4, WP-F — submission resource tables (raw CSV) that consolidate every round into one place per entity, with a
data dictionary. Values are copied or joined from the producing tables (no new statistics).

S_gene_rhythms.csv     gene × tissue × cell type for every Tier-1/2 call (80_atlas) and cochlear Tier C call (exploratory;
                       r2_cochlea): ZT and clock phase, amplitude, CIs, p / q / permutation q, independent methods, LOTO;
                       driver flags and primary driver (r4_drivers); hearing sets and chronopharmacology window (83);
                       GWAS trait groups (84); human orthologue (MGI 1:1); replicated bulk rhythm (GSE88954 cochlea /
                       GSE198770 BM; r4_attribution) with clock-aligned phase difference; baboon spleen / bone-marrow
                       rhythm and aligned phase difference (r4_crossspecies; BOM offset imputed).
S_celltypes.csv        tissue × cell type: cells per sample (ZT columns), clock (34; cell-level 332), rhythm calls (33;
                       Tier C), abundance rhythm (r2_figure multi-model verdict, unadjusted), power (37: minimum FC for
                       80 % power by expression bin), driver shares (r4_drivers).
S_abundance.csv        abundance rhythms (r2_figure abundance_pct_rhythm + multi-model verdict; cochlear immune types with
                       the marrow-adjusted sensitivity of r3_cochlea).
S_clock.csv            core-clock genes per tissue × cell type (33 tables, all statuses) + atlas tissue offsets (230).
S_external_datasets.csv  memory/datasets.csv (curated) — every external dataset used or considered, with status / reason
                         (in-house data — the user's imaging and the flow cohort, accessions data/... — are described in
                         Methods and excluded here; round 9, D89).
S_claims_to_tables.csv   r3_conclusions.csv (claims, verdicts, evidence and figure-ready tables).
S_flow_cohort.csv        independent flow-cytometry cohort (round 9, D89; 932): pre-specified readouts per mouse × tissue
                         with exclusions and consistency flags (r9_flow/flow_readouts.csv).
S_flow_tests.csv         its pre-specified tests (P1-P3, S1-S2, sensitivity) and the digestion controls per processing day
                         (r9_flow/flow_tests.csv, flow_digestion_controls.csv).
data_dictionary.csv    every column of every S table: description, unit, source.
Outputs (results/r4_resource/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, ROOT, log, write_csv  # noqa: E402

MOD = "r4_resource"


def rd(rel, **kw):
    p = RESULTS / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


off = rd("r2_phase/clock_alignment_offsets.csv")
OFF = {t: float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == t) & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
       for t in ["bm", "cochlea"]}
hom = pd.read_csv(EXTERNAL / "resources/mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
m2h = {m[0]: h[0] for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1}

# ================================================================ S_gene_rhythms
cat = rd("80_atlas/rhythmic_catalogue.csv")
tc = rd("r2_cochlea/tierC_catalogue.csv")
tc = tc[~tc["primary_tier"].astype(str).isin(["tier1", "tier2"])]
tc = tc.rename(columns={"loto_sd": "loto_phase_sd_h"}).assign(tissue="cochlea", tier="tierC (exploratory)",
                                                                foreign_marker_of=tc["marker_of"].where(tc["lineage_foreign_marker"].astype(str) == "True", ""))
keep_tc = ["tissue", "cell_type", "gene", "tier", "ptt_fc", "peak_zt", "p", "edgeR_p", "rain_p", "jtk_p", "loto_phase_sd_h", "foreign_marker_of",
           "celltype_empirical_fdr", "rule_selected"]
g = pd.concat([cat, tc[[c for c in keep_tc if c in tc.columns]]], ignore_index=True)
g["peak_clock"] = phase.zt_to_clock(g["peak_zt"])
g["aligned_phase"] = phase.wrap24(g["peak_zt"] - g["tissue"].map(OFF))
g["human_orthologue"] = g["gene"].map(m2h)
drv = rd("r4_drivers/driver_calls.csv")
if len(drv):
    dcols = ["primary_driver", "core_clock", "temperature", "handling_linked", "composition_linked", "gc_driven", "gc_dependent_ext",
             "clock_controlled", "clock_controlled_strength", "dd_support", "dd_n_organs", "clock_elements_phase_consistent", "pattern_linked", "pattern_r"]
    drv["tier"] = drv["tier"].replace({"tierC": "tierC (exploratory)"})
    g = g.merge(drv[["tissue", "cell_type", "gene", "tier"] + dcols], on=["tissue", "cell_type", "gene", "tier"], how="left")
hr = rd("83_hearing/hearing_gene_rhythm.csv")
if len(hr):
    g = g.merge(hr.drop_duplicates(["tissue", "cell_type", "gene"])[["tissue", "cell_type", "gene", "hearing_sets"]], how="left")
cw = rd("83_hearing/chronopharmacology_windows.csv")
if len(cw):
    g = g.merge(cw.drop_duplicates(["tissue", "cell_type", "gene"])[["tissue", "cell_type", "gene", "peak_window_zt", "peak_clock_time",
                                                                      "trough_window_zt", "trough_clock_time"]], how="left")
gw = rd("84_gwas/gwas_trait_genes.csv")
if len(gw):
    g["gwas_trait_groups"] = g["gene"].map(gw.groupby("mouse_gene")["trait_group"].agg(lambda x: "; ".join(sorted(set(x)))))
ba = rd("r4_attribution/bulk_gene_attribution.csv")
if len(ba):
    b = ba[["tissue", "gene", "external", "ext_aligned_phase", "main_contributor", "main_share", "mechanism_class"]].rename(
        columns={"external": "replicated_bulk_dataset", "ext_aligned_phase": "replicated_bulk_aligned_phase",
                 "main_contributor": "replicated_bulk_main_contributor", "main_share": "replicated_bulk_main_share",
                 "mechanism_class": "replicated_bulk_mechanism"})
    g = g.merge(b, on=["tissue", "gene"], how="left")
    g["replicated_bulk_phase_diff_h"] = phase.circ_diff(g["aligned_phase"], g["replicated_bulk_aligned_phase"])
# baboon (spleen; bone marrow with imputed offset)
fp = EXTERNAL / "geo/GSE98965/GSE98965_baboon_tissue_expression_FPKM.csv.gz"
bo = rd("r4_crossspecies/baboon_clock_offsets.csv")
if fp.exists() and len(bo):
    from lib import cosinor  # noqa: E402
    import re
    raw = pd.read_csv(fp).dropna(subset=["Symbol"]).drop_duplicates("Symbol").set_index("Symbol").drop(columns="EnsemblID")
    for code, nm in [("SPL", "baboon_spleen"), ("BOM", "baboon_bm")]:
        cc = [c for c in raw.columns if re.match(rf"^{code}[._]ZT\d+", c)]
        zt = np.array([float(re.search(r"ZT(\d+)", c).group(1)) for c in cc])
        x = raw[cc]
        x = x[x.mean(1) >= 1]
        f = cosinor.fit(np.log2(x + 1).T.to_numpy(), zt)
        o = float(bo.set_index("baboon_tissue").loc[code, "offset_used_h"])
        t = pd.DataFrame({"human_orthologue": x.index, f"{nm}_p": f["p"], f"{nm}_aligned_phase": phase.wrap24(f["peak"] - o)})
        g = g.merge(t, on="human_orthologue", how="left")
        g[f"{nm}_phase_diff_h"] = phase.circ_diff(g["aligned_phase"], g[f"{nm}_aligned_phase"])
g = g.sort_values(["tissue", "cell_type", "tier", "gene"])
write_csv(g, MOD, "S_gene_rhythms.csv",
          "Resource: every rhythmic call (Tier 1/2; cochlear Tier C exploratory) with phase (ZT, clock, aligned), statistics, drivers, hearing / GWAS "
          "annotation, human orthologue, replicated bulk and baboon rhythms", required=["tissue", "cell_type", "gene", "tier", "peak_zt", "peak_clock"])

# ================================================================ S_celltypes
cnt = rd("00_audit/celltype_counts_by_sample.csv")
ct = cnt.pivot_table(index=["tissue", "cell_type", "cell_class"], columns="zt", values="n_cells", fill_value=0)
ct.columns = [f"n_cells_ZT{int(c)}" for c in ct.columns]
ct["n_cells_total"] = ct.sum(1)
ct = ct.reset_index()
cl = rd("34_clock_network/clock_network_by_celltype.csv")
if len(cl):
    ct = ct.merge(cl[["tissue", "cell_type", "status", "n_clock_genes_expressed", "n_clock_genes_p05", "clock_phase_offset_h", "phase_coherence",
                      "median_clock_amp_log2", "circ_corr_observed_vs_canonical", "p_circ_corr", "CCD", "Pvalue"]].rename(
        columns={"Pvalue": "CCD_p", "status": "rhythm_status"}), how="left")
cc_ = rd("r3_audit/cell_clock_phase_summary.csv")
if len(cc_):
    ct = ct.merge(cc_[["tissue", "cell_type", "offset_resultant", "p_perm"]].rename(
        columns={"offset_resultant": "cell_level_clock_offset_resultant", "p_perm": "cell_level_clock_p_perm"}), how="left")
rs = rd("33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv")
if len(rs):
    ct = ct.merge(rs.rename(columns={"group": "cell_type"})[["tissue", "cell_type", "n_genes_tested", "pi1_storey", "n_tier1", "n_tier2",
                                                              "perm_global_p", "perm_empirical_fdr"]], how="left")
if len(tc):
    ct = ct.merge(tc.groupby("cell_type").size().rename("n_tierC_exploratory").reset_index().assign(tissue="cochlea"), how="left")
ms = rd("r2_figure/abundance_multimodel_summary.csv")
if len(ms):
    ct = ct.merge(ms[["tissue", "cell_type", "ols_pct_peak_zt", "ols_pct_peak_clock", "ols_pct_p", "p_perm_ols", "verdict"]].rename(
        columns={"ols_pct_peak_zt": "abundance_peak_zt", "ols_pct_peak_clock": "abundance_peak_clock", "ols_pct_p": "abundance_p_ols",
                 "p_perm_ols": "abundance_p_perm", "verdict": "abundance_multimodel_verdict"}), how="left")
pw = rd("37_power/power_by_celltype.csv")
if len(pw):
    mfc = pw.dropna(subset=["min_fc_80pct_alpha001"]).groupby(["tissue", "cell_type", "expr_bin"])["min_fc_80pct_alpha001"].first().unstack()
    mfc.columns = [f"min_fc_80pct_power_{c}" for c in mfc.columns]
    ct = ct.merge(mfc.reset_index(), how="left")
land = rd("r4_drivers/driver_landscape.csv")
if len(land):
    sh = land[(land["tier_set"] == "tier1/2") & (land["cell_type"] != "ALL")].pivot_table(index=["tissue", "cell_type"], columns="driver", values="share")
    sh.columns = [f"driver_share_{c.replace(' ', '_').replace('-', '_')}" for c in sh.columns]
    ct = ct.merge(sh.reset_index(), how="left")
write_csv(ct, MOD, "S_celltypes.csv", "Resource: one row per tissue × cell type — cells per ZT, clock, rhythm calls, abundance rhythm, power, driver shares",
          required=["tissue", "cell_type", "n_cells_total"])

# ================================================================ S_abundance
pr = rd("r2_figure/abundance_pct_rhythm.csv")
ab = pr.merge(ms[["tissue", "cell_type", "n_models", "n_models_p05", "core_supporting", "verdict"]], how="left") if len(ms) else pr
co = rd("r3_cochlea/cochlea_celltype_oscillations.csv")
if len(co):
    ab = ab.merge(co[["cell_type", "immune", "sensitivity_marrow_adjusted_prob_fc_gt_1p5", "sensitivity_marrow_adjusted_peak_zt",
                      "r_with_sample_pattern_pc1"]].assign(tissue="cochlea"), how="left")
write_csv(ab, MOD, "S_abundance.csv", "Resource: abundance rhythms (% of tissue cells; OLS cosinor, permutation, LOTO, bootstrap) with multi-model verdicts; "
          "cochlear immune types with the marrow-adjusted sensitivity", required=["tissue", "cell_type", "peak_zt", "peak_clock"])

# ================================================================ S_clock
CLK = list(phase.CANONICAL_CLOCK) + ["Clock", "Rora", "Bhlhe40", "Bhlhe41"]
sc_rows = []
for t in ["bm", "cochlea"]:
    a = rd(f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "status", "AveExpr", "amp_log2", "ptt_fc", "peak_zt",
                                                                            "peak_ci_low", "peak_ci_high", "p", "q_BH", "tier"])
    a = a[a["gene"].isin(CLK)].rename(columns={"group": "cell_type"}).assign(tissue=t)
    a["canonical_zt"] = a["gene"].map(phase.CANONICAL_CLOCK)
    a["peak_clock"] = phase.zt_to_clock(a["peak_zt"])
    a["aligned_phase"] = phase.wrap24(a["peak_zt"] - OFF[t])
    a["aligned_minus_canonical_h"] = phase.circ_diff(a["aligned_phase"], a["canonical_zt"])
    a["tissue_offset_h"] = OFF[t]
    sc_rows.append(a)
write_csv(pd.concat(sc_rows, ignore_index=True), MOD, "S_clock.csv",
          "Resource: core-clock genes per tissue × cell type (all cell types) with phase (ZT, clock, aligned) vs canonical phases",
          required=["tissue", "cell_type", "gene", "peak_zt", "peak_clock", "p"])

# ================================================================ datasets and claims
ds = pd.read_csv(ROOT / "memory/datasets.csv", keep_default_na=False) if (ROOT / "memory/datasets.csv").exists() else pd.DataFrame()
if len(ds):
    ds = ds[~ds["accession"].astype(str).str.startswith("data/")]           # in-house data: Methods (D89)
    write_csv(ds, MOD, "S_external_datasets.csv", "Resource: external datasets used or considered (curated in memory/datasets.csv), role, status, scripts",
              required=["accession", "status"])
cl_ = rd("r3_conclusions.csv")
if len(cl_):
    write_csv(cl_, MOD, "S_claims_to_tables.csv", "Resource: claims → verdicts → evidence files → figure-ready tables (from r3_conclusions.csv)",
              required=["section", "claim", "verdict"])

# ================================================================ round 9: independent flow cohort (D89)
fr_, ft_, fdig_ = rd("r9_flow/flow_readouts.csv"), rd("r9_flow/flow_tests.csv"), rd("r9_flow/flow_digestion_controls.csv")
if len(fr_):
    write_csv(fr_.drop(columns=["source_md5"], errors="ignore"), MOD, "S_flow_cohort.csv",
              "Resource: independent flow-cytometry cohort (D89) - pre-specified readouts per mouse x tissue, exclusions, "
              "consistency flags (r9_flow/flow_readouts.csv)", required=["sample_id", "mouse_id", "tissue", "zt"])
if len(ft_):
    ft_ = pd.concat([ft_.drop(columns=["source_md5"], errors="ignore").assign(table="test"),
                     fdig_.drop(columns=["source_md5"], errors="ignore").assign(table="digestion control", test="digestion",
                                                                                family="control")], ignore_index=True)
    write_csv(ft_, MOD, "S_flow_tests.csv",
              "Resource: flow cohort pre-specified tests (P1-P3 with Holm, S1-S2 mixed models, D17 sensitivity) and digestion "
              "controls per processing day (r9_flow)", required=["table", "test", "family"])

# ================================================================ round 5: B cells
for src_, name_, desc_ in [("r5_bcell/primary_hypotheses.csv", "S_bcell_primary_hypotheses.csv", "Resource: round-5 B-cell primary hypotheses with decisions"),
                           ("r5_bcell/timeline_24h.csv", "S_bcell_timeline.csv", "Resource: round-5 24-h B-cell timeline (ZT and clock)"),
                           ("r5_bcell/b_composition_summary.csv", "S_bcell_composition.csv", "Resource: B-lineage composition rhythms per frame and stage"),
                           ("r5_bcell/immgen_label_mapping.csv", "S_bcell_immgen_mapping.csv", "Resource: our B labels vs ImmGen sorted fractions")]:
    d_ = rd(src_)
    if len(d_):
        write_csv(d_, MOD, name_, desc_)

# ================================================================ round 6: myeloid cells
for src_, name_, desc_ in [("r6_myeloid/m_primary_hypotheses.csv", "S_myeloid_primary_hypotheses.csv", "Resource: round-6 myeloid primary hypotheses with decisions"),
                           ("r6_myeloid/m_multimethod_matrix.csv", "S_myeloid_multimethod.csv", "Resource: round-6 agreeing-method components per hypothesis and tissue"),
                           ("r6_myeloid/m_timeline_24h.csv", "S_myeloid_timeline.csv", "Resource: round-6 24-h myeloid timeline (ZT and clock)"),
                           ("r6_myeloid/m_composition_summary.csv", "S_myeloid_composition.csv", "Resource: myeloid composition rhythms per frame and state"),
                           ("r6_myeloid/myeloid_reference_signatures.csv", "S_myeloid_reference_signatures.csv", "Resource: external myeloid signatures (round 6)"),
                           ("r6_myeloid/m_external_signatures.csv", "S_myeloid_neumap_signatures.csv", "Resource: NeuMAP / GSE200145 neutrophil-lineage signatures")]:
    d_ = rd(src_)
    if len(d_):
        write_csv(d_, MOD, name_, desc_)

# ================================================================ data dictionary
D = {
    "tissue": "bm = temporal-bone marrow; cochlea", "cell_type": "gold-standard cell type (never re-annotated)", "gene": "mouse gene symbol",
    "tier": "tier1 / tier2 (33_rhythm_tiers rules); tierC (exploratory) = calibrated relaxed cochlear tier (211; FDR not reached)",
    "peak_zt": "peak time, h after lights-on (ZT; LD 12:12, lights on 08:00)", "peak_clock": "peak time in wall-clock hours (= ZT + 8)",
    "aligned_phase": "peak − this atlas's tissue clock offset vs canonical mouse phases (230); comparable across datasets / species",
    "ptt_fc": "peak-to-trough fold change (2^(2·amplitude_log2))", "amp_log2": "cosinor amplitude (log2 units)",
    "p": "cosinor p (limma-voom moderated F)", "q_BH": "Benjamini–Hochberg q", "q_perm": "empirical FDR from the 59 dihedral null relabelings",
    "loto_phase_sd_h": "leave-one-time-point-out circular SD of the peak (h)", "human_orthologue": "1:1 human orthologue (MGI HOM)",
    "primary_driver": "precedence: core clock > temperature > handling-linked > composition-linked > glucocorticoid-driven > clock-controlled > sample-pattern-linked > unexplained (r4_drivers)",
    "dd_support": "rhythmic in ≥ 2 GSE54650 organs (constant darkness) with aligned phase within ±4 h", "clock_elements_phase_consistent": "promoter E/E′-box, D-box or RORE with the element's expected phase within ±4 h",
    "replicated_bulk_aligned_phase": "aligned peak of the gene in the replicated bulk time series (GSE88954 sham / GSE198770 BM control)",
    "replicated_bulk_mechanism": "whether our composition-only or regulation-only profile reproduces the replicated bulk rhythm (r4_attribution)",
    "baboon_spleen_aligned_phase": "aligned peak in baboon spleen (GSE98965; own clock offset)", "baboon_bm_aligned_phase": "aligned peak in baboon bone marrow (offset imputed: weak BM clock)",
    "n_tier1": "Tier-1 calls", "n_tier2": "Tier-2 calls", "abundance_multimodel_verdict": "robust / supported / suggestive / not supported (r2_figure, 10 models)",
    "accession": "repository accession", "status": "used / candidate / excluded / blocked", "claim": "conclusion", "verdict": "register verdict",
    "zt": "ZT group of the flow cohort (ZT5 = 13:00, ZT17 = 01:00)", "cohort": "flow cohort (1 or 2; each with both ZT groups)",
    "P1": "immature neutrophils (Ly6G-int CD11b-int), % of CD45+ (T1 x T2)", "P2": "classical monocytes (CD11b+ Ly6C-hi, monocyte scatter), % of CD45+ (T3 / T1)",
    "P3": "recirculating B cells (B220-hi IgD+), % of CD45+ (T6 / T1)", "S1_pct": "CD62L-hi, % of Ly6G+ (T4)", "S1_med": "CD62L median, bead-relative (T4)",
    "S2_pct": "CXCR4-hi, % of Ly6G+ (T5)", "S2_med": "CXCR4 median, bead-relative (T5)", "exclusions": "pre-set exclusions (plan 6.3) hitting this sample",
    "effect": "ZT17 / ZT5 ratio (P1-P3, medians) or odds ratio (% positive); interaction = cochlea / marrow ratio of ratios",
    "p_holm": "Holm-adjusted p across the three primaries", "verdict_flow": "replicated / direction-consistent / contradicted (plan 6.4)",
}
dd = []
for name in ["S_gene_rhythms.csv", "S_celltypes.csv", "S_abundance.csv", "S_clock.csv", "S_external_datasets.csv", "S_claims_to_tables.csv",
             "S_bcell_primary_hypotheses.csv", "S_bcell_timeline.csv", "S_bcell_composition.csv", "S_bcell_immgen_mapping.csv",
             "S_myeloid_primary_hypotheses.csv", "S_myeloid_multimethod.csv", "S_myeloid_timeline.csv", "S_myeloid_composition.csv",
             "S_myeloid_reference_signatures.csv", "S_myeloid_neumap_signatures.csv", "S_flow_cohort.csv", "S_flow_tests.csv"]:
    p = RESULTS / MOD / name
    if not p.exists():
        continue
    cols = pd.read_csv(p, nrows=0).columns
    for c in cols:
        desc = D.get(c, "")
        if not desc:
            if c.startswith("n_cells_ZT"):
                desc = f"cells of this type in the sample at {c.replace('n_cells_', '')}"
            elif c.startswith("driver_share_"):
                desc = "share of the type's Tier-1/2 calls with this primary driver"
            elif c.startswith("min_fc_80pct_power_"):
                desc = "minimum peak-to-trough FC detectable with 80 % power at alpha 0.001 in this expression bin (37_power)"
            elif c.endswith("_phase_diff_h"):
                desc = "signed circular difference of aligned phases (ours − reference), h"
            elif c.endswith("_p"):
                desc = "p value (see source table)"
            else:
                desc = "copied from the source table (see script docstring)"
        dd.append({"table": name, "column": c, "description": desc})
write_csv(pd.DataFrame(dd), MOD, "data_dictionary.csv", "Data dictionary: every column of every S table with its description",
          required=["table", "column", "description"])
log(f"S_gene_rhythms {len(g)} rows; S_celltypes {len(ct)}; dictionary {len(dd)} columns")

# ---------------- ZT / clock consistency of the S tables (clock = ZT + 8)
for name in ["S_gene_rhythms.csv", "S_abundance.csv", "S_clock.csv", "S_bcell_timeline.csv", "S_myeloid_timeline.csv", "S_myeloid_composition.csv"]:
    d = pd.read_csv(RESULTS / MOD / name, usecols=lambda c: c in ("peak_zt", "peak_clock")).dropna()
    bad = (np.abs(phase.circ_diff(d["peak_clock"] - d["peak_zt"], 8.0)) > 1e-6).sum()
    log(f"{name}: clock − ZT = 8 h check: {len(d) - bad}/{len(d)} rows consistent")
    assert bad == 0, f"{name}: {bad} rows violate clock = ZT + 8"
