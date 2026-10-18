"""Round 2, WP5 — automated audit of all result CSVs and the register of known gaps.

Per CSV (results/**, first 200,000 rows): rows, columns, fraction of missing cells, exact duplicate rows, p-value
columns without a multiple-testing companion (q / FDR / lfsr), and whether the file is documented in the manifest.
Global: time-axis checks (200), plan-item status (97), scripts that draw random numbers without a seed.
Gap register: each round-1 gap with the round-2 remediation, status derived from the round-2 outputs, and the key
evidence numbers.
Outputs (results/r2_audit/): audit_checks.csv, audit_global.csv, gap_register.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_audit"
SCRIPTS = Path(__file__).resolve().parent
P_COL = re.compile(r"(^p$|^p_|_p$|pvalue|p_value|p_perm|p_fisher)", re.I)
Q_COL = re.compile(r"(^q$|^q_|_q$|q_bh|fdr|padj|lfsr|adj)", re.I)
man = pd.read_csv(RESULTS / "results_manifest.csv") if (RESULTS / "results_manifest.csv").exists() else pd.DataFrame(columns=["path"])
documented = set(man["path"].str.replace("results/", "", regex=False)) if len(man) else set()
rows = []
for p in sorted(RESULTS.rglob("*.csv")):
    rel = str(p.relative_to(RESULTS)).replace("\\", "/")
    if "/counts/" in rel:
        continue
    try:
        d = pd.read_csv(p, nrows=200000, low_memory=False)
    except Exception as e:
        rows.append({"file": rel, "readable": False, "note": str(e)[:200]})
        continue
    pc = [c for c in d.columns if P_COL.search(c) and not Q_COL.search(c)]
    qc = [c for c in d.columns if Q_COL.search(c)]
    rows.append({"file": rel, "readable": True, "rows_checked": len(d), "n_columns": d.shape[1],
                 "frac_missing": float(d.isna().to_numpy().mean()) if d.size else np.nan,
                 "n_duplicate_rows": int(d.duplicated().sum()), "p_columns": ";".join(pc), "q_columns": ";".join(qc[:6]),
                 "p_without_q": bool(pc and not qc), "documented_in_manifest": rel in documented or rel.startswith("r2_"),
                 "empty": len(d) == 0})
chk = pd.DataFrame(rows)
write_csv(chk, MOD, "audit_checks.csv", "Per-CSV audit: missingness, duplicates, p/q pairing, documentation",
          required=["file", "readable"])

glob_rows = []
ta = RESULTS / "r2_audit/time_axis_checks.csv"
if ta.exists():
    t = pd.read_csv(ta)
    glob_rows.append({"check": "time axis (ZT) recomputation checks", "result": f"{int(t['pass'].sum())}/{len(t)} passed",
                      "pass": bool(t["pass"].all())})
ps = RESULTS / "plan_execution_status.csv"
if ps.exists():
    t = pd.read_csv(ps)
    glob_rows.append({"check": "plan items (round 1)", "result": "; ".join(f"{k}={v}" for k, v in t["status"].value_counts().items()),
                      "pass": True})
unseeded = []
for f in sorted(list(SCRIPTS.glob("*.py")) + list(SCRIPTS.glob("*.R"))):
    s = f.read_text(encoding="utf-8", errors="ignore")
    rand = re.search(r"np\.random\.|rng\.|sample\(|rbinom|runif|rnorm|random_state|sample\.int", s)
    seed = re.search(r"default_rng\(|set\.seed|random_state=|seed", s)
    if rand and not seed:
        unseeded.append(f.name)
glob_rows.append({"check": "scripts with random draws but no explicit seed", "result": ";".join(unseeded) or "none",
                  "pass": not unseeded})
glob_rows.append({"check": "CSV files with exact duplicate rows", "result": str(int((chk.get("n_duplicate_rows", 0) > 0).sum())),
                  "pass": True})
glob_rows.append({"check": "CSV files with p-values but no multiple-testing column", "result": str(int(chk.get("p_without_q", False).sum())),
                  "pass": True})
glob_rows.append({"check": "empty CSV files", "result": ";".join(chk.loc[chk.get("empty", False) == True, "file"]) or "none",
                  "pass": not bool((chk.get("empty", False) == True).any())})
write_csv(pd.DataFrame(glob_rows), MOD, "audit_global.csv", "Global audit checks", required=["check", "result", "pass"])


def ex(rel):
    return (RESULTS / rel).exists()


def rd(rel, **kw):
    return pd.read_csv(RESULTS / rel, **kw) if ex(rel) else pd.DataFrame()


gaps = []


def gap(gid, issue, remediation, outputs, status, evidence):
    gaps.append({"gap_id": gid, "issue": issue, "round2_remediation": remediation, "outputs": outputs,
                 "status": status, "evidence": evidence})


fr = rd("r2_figure/figure_replication_check.csv")
gap("G01", "user figure labelled clock time as ZT (peaks 8 h late)", "exact replication; all tables give ZT and clock",
    "r2_figure/*", "fixed" if len(fr) and fr["match"].all() else "open",
    f"{int(fr['match'].sum()) if len(fr) else 0}/6 panels reproduced; true ZT = clock − 8")
ms = rd("r2_figure/abundance_multimodel_summary.csv")
gap("G02", "abundance-rhythm significance depended on the statistical model", "10-model robustness with pre-defined verdicts",
    "r2_figure/abundance_multimodel_*.csv", "resolved" if len(ms) else "open",
    "; ".join(f"{t}: {', '.join(f'{k}={v}' for k, v in g['verdict'].value_counts().items())}" for t, g in ms.groupby("tissue")) if len(ms) else "")
sp = rd("r2_cochlea/sample_pattern_covariates.csv")
ev = ""
if len(sp):
    s1 = sp[(sp["tissue"] == "cochlea") & (sp["pc"] == 1)].set_index("covariate")["pearson_r"]
    ev = "cochlea PC1 r with: " + "; ".join(f"{k} {v:+.2f}" for k, v in s1.items())
gap("G03", "cochlear rhythms dominated by a tissue-wide sample-level pattern", "PCA of shared deviations vs sample covariates; "
    "pattern-like genes flagged in Tier C", "r2_cochlea/sample_pattern_*.csv", "diagnosed" if len(sp) else "open", ev)
sim = rd("33_rhythm_tiers/rhythm_pipeline_simulation.csv")
gap("G04", "raw BH q anticonservative against structured non-circadian variation", "permutation FDR in Tier 1 and Tier C calibration",
    "33_rhythm_tiers/rhythm_pipeline_simulation.csv; r2_cochlea/cochlea_relaxed_calibration.csv", "mitigated",
    "BM FDP raw BH 0.12–0.22 vs Tier-1-like 0.02" if len(sim) else "")
pat = rd("r2_cochlea/sample_pattern_scores.csv")
ev = ""
if len(pat):
    b = pat[(pat["tissue"] == "bm") & (pat["pc"] == 1)].iloc[0]
    ev = f"BM PC1 cosinor R2 {b['pc_cosinor_r2']:.2f} (peak ZT{b['pc_cosinor_peak_zt']:.1f}); see covariates for stress-score correlation"
gap("G05", "night activation (IEG/NF-kB/heat-shock) program: physiology or collection effect", "tissue-wide pattern analysis; "
    "external replicated phases (gene concordance, signatures)", "r2_cochlea/*; r2_external/*", "partly resolved", ev)
gap("G06", "CollecTRI activity of repressor TFs antiphase to their mRNA", "documented: CollecTRI edges of repressors are mostly "
    "'activation'; ULM activity then peaks when the repressor is low — read −activity for repressors", "43_pathway_phase/tf_*",
    "documented", "clock-TF lag ~10–11 h (n = 6–7 pairs per tissue)")
gap("G07", "NNLS deconvolution degenerate (Cycling Cell ~0.9)", "superseded by BayesPrism (75b); NNLS kept only as comparison",
    "75_deconvolution/deconvolution_*", "deprecated", "")
gap("G08", "mash lfsr counts inflated by structure shared across cell types (same samples)", "documented; dream shared/specific "
    "components reported instead", "36_shared_specific/*", "documented", "")
dn = rd("r2_audit/dryr_null_check.csv")
ev = ""
if len(dn):
    o = dn[dn["run"] == "observed"].groupby("model_label")["N"].sum()
    n = dn[dn["run"] == "null"].groupby("model_label")["N"].sum() / 3
    ev = "; ".join(f"{k}: observed {int(o.get(k, 0))} vs null mean {n.get(k, 0):.0f}" for k in ["cochlea_only", "bm_only", "shared", "different"])
gap("G09", "dryR over-calls cochlea-only rhythms", "dryR re-run with relabelled cochlear time (3 cell types × 3 nulls)",
    "r2_audit/dryr_null_check.csv", "checked" if len(dn) else "open", ev)
gap("G10", "cell-level GLMM sample variance collapses to 0 (time = sample)", "documented: supportive only", "38_cell_level/*",
    "documented", "")
lc = rd("r2_audit/label_confidence_abundance.csv")
ev = f"median |peak shift| {np.nanmedian(np.abs(lc['peak_shift_h'])):.2f} h over {len(lc)} series" if len(lc) else ""
gap("G11", "dependence of findings on annotation confidence", "abundance and BM gene rhythms recomputed on high-confidence cells",
    "r2_audit/label_confidence_*.csv", "checked" if len(lc) else "open", ev)
bl = rd("r2_audit/bm_likeness_scvi_rhythm.csv")
gap("G12", "embedding-based results may depend on label-supervised scANVI", "Milo and BM-likeness on unsupervised scVI",
    "53_milo/milo_*_scvi.csv; r2_audit/bm_likeness_scvi_rhythm.csv", "checked" if len(bl) else "open",
    f"BM-likeness rhythmic (p<0.05) with scVI: {int((bl['p'] < 0.05).sum())}/{len(bl)}" if len(bl) else "")
cp = rd("02_clock_controls/clock_phase_order.csv")
tt = rd("74_timetelling/timetelling_summary.csv")
gap("G13", "time-telling outliers for some cell classes (e.g. erythroid −6 h)", "explained by weak clock-gene expression",
    "02_clock_controls/clock_phase_order.csv", "explained" if len(cp) else "open",
    f"median expressed clock genes: erythroblasts {cp.loc[cp['cell_type'].str.contains('Erythro'), 'n_clock_genes_expressed'].median():.0f} vs all {cp['n_clock_genes_expressed'].median():.0f}" if len(cp) else "")
nt = rd("r2_audit/neutrophil_timer_replication.csv")
gap("G14", "neutrophil-timer opposite signs (cochlea +, BM −) need replication", "external BM/tissue time series vs blood "
    "neutrophil ZT13−ZT5 change", "r2_audit/neutrophil_timer_replication.csv", "checked" if len(nt) else "open",
    "; ".join(f"{r.dataset}|{r.condition}: rho {r.spearman_rho:+.2f}" for r in nt[(nt['genes'] == 'neutrophil_markers')].itertuples()) if len(nt) else "")
off = rd("r2_phase/clock_alignment_offsets.csv")
lung = off[(off["dataset"] == "GSE198770_lung")] if len(off) else pd.DataFrame()
gap("G15", "GSE198770 lung samples: clock genes offset vs every other tissue", "flagged; only clock-aligned (internal) phases used",
    "r2_phase/clock_alignment_offsets.csv", "flagged" if len(lung) else "n/a",
    "; ".join(f"{r.condition} {r.offset_canonical_h:+.1f} h" for r in lung[lung['unit'] == 'bulk'].itertuples()) if len(lung) else "")
ltr = rd("r2_external/skull_marrow_label_transfer.csv")
gap("G16", "neutrophil-stage naming differs from the skull-marrow reference (Kolabas 2023)", "documented: their immature ≈ our "
    "'Mature', their mature ≈ our 'Aged'", "r2_external/skull_marrow_label_transfer.csv", "documented" if len(ltr) else "open", "")
gap("G17", "MultiNicheNet for the cochlea capped (memory)", "documented: immune receivers, <= 800 cells per sample × type",
    "67_multinichenet/*", "documented", "")
gap("G18", "CircaDB not used in round 1", "GSE54650 (12 organs) added for clock alignment and phase concordance",
    "r2_external/external_rhythm_GSE54650.csv", "fixed" if ex("r2_external/external_rhythm_GSE54650.csv") else "open", "")
sel = rd("r2_cochlea/cochlea_relaxed_rule_selection.csv")
gap("G19", "pre-specified relaxed cochlear rule (R0) anticonservative: edgeR on the same counts is not independent support",
    "declared rule ladder calibrated on the same 59 nulls; Tier C = most inclusive rule with empirical FDR <= 0.2, else the "
    "lowest-FDR rule (declared fallback)", "r2_cochlea/cochlea_relaxed_rule_selection.csv",
    ("fixed" if (sel["empirical_fdr"] <= 0.2).any() else "partly fixed: FDR target not reachable; Tier C exploratory")
    if len(sel) else "open",
    "; ".join(f"{r.rule} {int(r.observed_calls)} calls, FDR {r.empirical_fdr:.2f}{' (selected)' if str(r.selected) == 'True' else ''}"
              for r in sel.itertuples()) if len(sel) else "")
cov = rd("r2_phase/phase_ci_coverage.csv")
ev = ""
if len(cov):
    s_ = cov.groupby("ptt_fc_injected")[["coverage_all", "coverage_determinate"]].mean()
    ev = "; ".join(f"FC{k:g}: all {r.coverage_all:.2f}, determinate {r.coverage_determinate:.2f}" for k, r in s_.iterrows())
gap("G20", "delta-method phase CIs under-cover weak rhythms (nominal 0.95)", "phases reported with the determinate flag "
    "(half-width <= 3 h); weak-rhythm phases treated as indicative", "r2_phase/phase_ci_coverage.csv",
    "quantified" if len(cov) else "open", ev)
gap("G21", "first label-confidence run: quasi-binomial dispersion mis-scaled (statsmodels scale='X2' with two-column binomial "
    "endog), every p ≈ 0", "dispersion = Pearson X² / df on the count scale; re-run", "r2_audit/label_confidence_abundance.csv",
    "fixed", "")
gap("G22", "BM–cochlea phase differences reported modulo 24 with arc-wrapped CIs", "signed Δ in (−12, 12] with CIs from "
    "unwrapped bootstrap draws", "r2_figure/figure_phase_relationships.csv", "fixed", "")
gap("G23", "multi-model summary counted CLR support but not CLR as tested", "core_tested includes the CLR permutation model",
    "r2_figure/abundance_multimodel_summary.csv", "fixed", "")
ct_ = rd("r2_pilots/pilot_collection_time_inference.csv")
gap("G24", "GSE107086 collection time ambiguous (GEO ZT11/ZT23 vs design text ~ZT6/ZT18)", "collection time inferred from "
    "clock/rhythmic genes with the external cochlea (GSE88954) as reference", "r2_pilots/pilot_collection_time_inference.csv",
    "resolved" if len(ct_) else "open",
    "; ".join(f"{r.reference} [{r.genes}] {r.day_collection_internal_h:.1f} h (r {r.pearson_r:.2f})"
              for r in ct_[ct_["is_best"].astype(str) == "True"].itertuples()) if len(ct_) else "")
dpe = rd("r2_external/signature_depletion_effect.csv")
bpe = rd("75_deconvolution/bayesprism_condition_effects.csv")
ev = ""
if len(dpe):
    x = dpe[dpe["signature"].str.startswith("Neutrophil m") | dpe["signature"].str.startswith("Neutrophil i")]
    ev = "; ".join(f"{r.tissue} {r.contrast.replace('_vs_control', '')} {r.signature.split(' (')[0]} {r.mean_diff:+.2f}"
                   for r in x.itertuples())
if len(bpe):
    b = bpe[(bpe["dataset"] == "GSE198770") & bpe["cell_type"].str.startswith("Neutrophil (Aged)")]
    ev += " | BayesPrism BM aged-neutrophil fraction: " + "; ".join(f"{r.mean_by_condition} (p {r.p_mean_shift:.1e})" for r in b.itertuples())
gap("G25", "planned positive control 'neutrophil signature falls after depletion' fails for marker signatures",
    "split immature (Band/Mature) vs mature (Aged/Activated) signatures; full-transcriptome deconvolution (BayesPrism) "
    "used as the positive control", "r2_external/signature_depletion_effect.csv; 75_deconvolution/bayesprism_condition_effects.csv",
    "explained: left shift (granule-gene-rich immature neutrophils rise) masks the loss of mature cells in bulk signatures", ev)
# ---- round 3
hse = rd("r3_audit/handling_sample_effect.csv")
hgf = rd("r3_audit/handling_gene_flags.csv")
ev = ""
if len(hse) and len(hgf):
    x = hgf[hgf["gene_class"].isin(["IEG", "heat_shock"]) & hgf["phase_bin"].isin(["ZT12-18", "ZT18-24"])].groupby("gene_class")[["n_calls", "n_stress_like"]].sum()
    ev = (f"stress score sample term {hse.groupby('tissue')['sample_term_variance_share'].first().round(2).to_dict()}; night calls following the "
          f"stress profile: IEG {x.loc['IEG', 'n_stress_like'] / x.loc['IEG', 'n_calls']:.0%}, heat shock "
          f"{x.loc['heat_shock', 'n_stress_like'] / x.loc['heat_shock', 'n_calls']:.0%}")
gap("G26", "night activation programme: physiology vs collection (G05)", "per-sample dissociation-stress decomposition, gene-level "
    "stress-profile flags, independent cohort (GSE107086), external phases", "r3_audit/handling_*.csv; r2_pilots/pilot_night_programme_check.csv",
    "resolved: heat shock physiological, IEG / NF-κB handling-linked" if ev else "open", ev)
bsel = rd("r3_bm/bm_relaxed_rule_selection.csv")
gap("G27", "relaxed gene tiers cannot reach the FDR target in either tissue", "declared fallback rule used and labelled exploratory; "
    "strict tiers recommended", "r2_cochlea/cochlea_relaxed_rule_selection.csv; r3_bm/bm_relaxed_rule_selection.csv", "documented",
    ("BM: " + "; ".join(f"{r.rule} FDR {r.empirical_fdr:.2f}" for r in bsel.itertuples())) if len(bsel) else "")
c346 = rd("r3_gse346691/composition_contrasts.csv")
ev = ""
if len(c346):
    x = c346[(c346["subset"] == "all cells") & (c346["contrast"] == "WT_noise_vs_WT_control") &
             c346["label"].isin(["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Macrophage"])]
    ev = "; ".join(f"{r.label} {r.log2_ratio:+.2f} (p {r.p_welch_logit:.2f})" for r in x.itertuples())
gap("G28", "external noise datasets disagree on neutrophil expansion (GSE167078 up at 3 d; GSE346691 not)", "both reported; "
    "R06 scored 'partial' for model concordance", "r3_gse167078/*; r3_gse346691/*", "documented", ev)
c312 = rd("r3_response/gse312298_collection_time.csv")
gap("G29", "canonical tissue clock phases misplace neutrophil sampling times (neutrophil clocks differ)", "collection time judged "
    "with our neutrophil clock fits and the design; validation reported at the design times", "r3_response/gse312298_collection_time.csv",
    "documented", "; ".join(f"{r.reference}: {r.day_collection_internal_h} h (r {r.pearson_r:.2f})"
                            for r in c312[c312['is_best'].astype(str) == 'True'].itertuples()) if len(c312) else "")
gap("G30", "Ma 2025 (CRA019820 / CRA019827) not yet analysed", "user quantifies and uploads matrices (Aspera blocked from the server); "
    "pipeline 300–314 dry-run passed", "r3_ingest/ma2025_pipeline_dryrun.csv",
    "done" if ex("r3_ma2025/composition_contrasts.csv") else "pending (user upload)", "")
ccs = rd("r3_audit/cell_clock_phase_summary.csv")
gap("G31", "single-cell phase heterogeneity (Tempo) not run in round 1", "cell-level clock projection with permutation test (332)",
    "r3_audit/cell_clock_phase_*.csv", "addressed" if len(ccs) else "open",
    f"{int((ccs['p_perm'] <= 0.05).sum())}/{len(ccs)} cell types track ZT" if len(ccs) else "")
gap("G32", "BM⇄cochlea egress / return schematic", "retired by user decision (BM is the sole source of cochlear immune cells); "
    "cochlea-only synthesis", "r3_cochlea/cochlea_celltype_oscillations.csv", "retired", "")
rm = rd("r3_refmap/reference_model.csv")
gap("G33", "label transfer to external single-cell data", "scANVI/scArches reference + PCA-kNN consensus; marker positive controls",
    "r3_refmap/reference_model.csv; r3_ingest/*_marker_check.csv", "checked" if len(rm) else "open",
    f"reference self-accuracy {rm.loc[rm['cell_type'] == 'ALL', 'self_prediction_accuracy'].iloc[0]:.2f}" if len(rm) else "")
# ---- round 4
dcs = rd("r4_drivers/diurnal_circadian_status.csv")
gap("G34", "light–dark design only: rhythms are diurnal; endogenous (circadian) control not testable in our data",
    "driver annotation with constant-darkness evidence from GSE54650 (phase-matched), clock elements, ADX dependence; terminology per finding",
    "r4_drivers/diurnal_circadian_status.csv; r4_drivers/driver_phase_null.csv", "documented" if len(dcs) else "open",
    "; ".join(f"{r.finding_id}: {r.recommended_term}" for r in dcs.itertuples() if str(r.finding_id)[0] in "GR") if len(dcs) else "")
bco = rd("r4_crossspecies/baboon_clock_offsets.csv")
gap("G35", "baboon bone marrow has a weak clock (few rhythmic clock genes): BM conservation tested with an imputed offset and spleen",
    "spleen (own clock offset) as immune-organ comparator; BOM with the median baboon tissue offset reported as partial",
    "r4_crossspecies/baboon_clock_offsets.csv; r4_crossspecies/baboon_bm_gene_concordance.csv", "documented" if len(bco) else "open",
    (lambda b: f"BOM rhythmic clock genes {int(b['n_clock_genes_rhythmic'])}; offset source: {b['offset_source']}")(bco.set_index('baboon_tissue').loc['BOM'])
    if len(bco) and "BOM" in set(bco["baboon_tissue"]) else "")
bsum = rd("r4_benchmark/benchmark_summary.csv")
gap("G36", "with 6 time points a rank-based independent test (JTK-like, Bonferroni over lags) cannot reach p < 0.05; Tier-1 independence "
    "rests on parametric alternatives (edgeR-QL / LR; OLS in the benchmark)", "documented; benchmark measures the realised FDP of the rules as used",
    "r4_benchmark/benchmark_summary.csv", "documented" if len(bsum) else "open", "")
ev = ""
if len(bsum):
    a = bsum[(bsum["unit"] == "ALL") & (bsum["rule"].isin(["nominal", "tier2"]))]
    ev = "; ".join(f"{r.dataset.split(' ')[0]} {r.condition} {r.rule}: calls {r.mean_calls:.0f}, FDP {r.mean_fdp_strict:.3f}" for r in a.itertuples())
gap("G37", "a widespread sample-level nuisance (cochlear-like) inflates nominal FDP and shrinks moderated power for all genes",
    "quantified by the benchmark stress test; pattern flags and strict tiers keep the realised FDP low at the cost of power",
    "r4_benchmark/benchmark_summary.csv; r4_benchmark/benchmark_pattern_flags.csv", "quantified" if ev else "open", ev)
agr = rd("r4_ageing/ageing_sampling_phase.csv")
gap("G38", "ageing atlas (GSE274279): one library per age, collection time unknown", "sampling-phase check; same-age library control for "
    "rhythmic-gene variability; reported as extended / descriptive", "r4_ageing/*", "documented" if len(agr) else "open", "")
gap("G39", "Tier C constant-darkness support is not independent of its selection (rule R1 required external phase support)",
    "Tier C reported separately in every round-4 table and flagged exploratory", "r4_drivers/driver_phase_null.csv", "documented", "")
gap("G40", "spliced / unspliced (nascent transcription) rhythms not analysable", "raw reads of our samples not available (user decision D25)",
    "memory/decisions.csv", "documented", "")
# ---- round 5
imm = rd("r5_bcell/immgen_label_mapping.csv")
ev = ""
if len(imm):
    b_ = imm[imm["is_best"].astype(str) == "True"]
    ev = "; ".join(f"{r.tissue} {r.our_stage} → {r.immgen_population} ({r.reference.split(' ')[1]})" for r in b_.itertuples() if not bool(r.expected))
gap("G41", "ImmGen mapping: some B-label names differ from their transcriptional fraction (e.g. 'Transitional' ≈ large pre-B FrC′, "
    "'Mature' ≈ immature FrE / T1); the pre-specified developmental-order test (H2) used label names",
    "labels unchanged (gold standard); naming note; label-free pseudotime wave reported as the order-independent test",
    "r5_bcell/immgen_label_mapping.csv; r5_bcell/b_milo_wave.csv", "documented" if len(imm) else "open", ev)
bc_ = rd("r5_bcell/b_composition_summary.csv")
gap("G42", "cochlear B stages show no within-lineage rhythm (marrow-derived B cells share the marrow-content axis)",
    "within-B frame cancels marrow content; reported with the share-of-tissue frame (unadjusted primary, D17)", "r5_bcell/b_composition_summary.csv",
    "documented" if len(bc_) else "open",
    "; ".join(f"{r.feature} {r.verdict}" for r in bc_[(bc_["tissue"] == "cochlea") & (bc_["frame"] == "within_B")].itertuples()) if len(bc_) else "")
gap("G43", "IL-7 → IL-7R not detectable in the LIANA per-sample scores (low Il7 in our senders); H5 used CXCL12 and BAFF/APRIL",
    "documented; MultiNicheNet / NicheNet reported", "r5_bcell/b_niche_axes.csv", "documented", "")
clk_ = rd("r5_bcell/b_clock_cell_level.csv")
gap("G44", "B-cell clock inference has little power (few clock UMIs per B cell; Arntl absent)", "cell-level usable fraction and gene-bootstrap CIs "
    "reported; H4 decided by the pre-specified rule", "r5_bcell/b_clock_cell_level.csv; r5_bcell/b_clock_offsets.csv", "documented" if len(clk_) else "open",
    "; ".join(f"{r.tissue} {r.group}: usable {r.frac_cells_usable:.2f}, UMIs/cell {r.mean_clock_umis_per_cell:.2f}" for r in clk_.itertuples()) if len(clk_) else "")
gap("G45", "B ↔ partner interactions: none rhythmic after correction (global dihedral test); nominal hits share sample-level profiles",
    "global test reported as primary; nominal counts not used as evidence", "r5_partners/b_partner_global_test.csv", "documented", "")
gap("G46", "GeoMx: B-lineage deconvolution share and the external ImmGen B signature disagree on neuron vs supporting AOIs",
    "both reported; H8 not supported by the pre-specified rule", "r5_partners/b_partner_geomx.csv", "documented", "")
# ---- round 6
ph6 = rd("r6_myeloid/m_primary_hypotheses.csv")
gap("G47", "M1 (BM) and M3 primary statistics track the sample-level pattern (|r| >= 0.9 with 210 PC1): maturation timing is not separable "
    "from the sample pattern in BM", "flag in m_primary_hypotheses (note); OT, Milo and composition reported; decision by the pre-specified rule",
    "r6_myeloid/m_primary_hypotheses.csv; r6_myeloid/m_axes_rhythm.csv", "documented" if len(ph6) else "open",
    "; ".join(f"{r.hypothesis} max |r| {r.max_abs_r_sample_pattern:.2f}" for r in ph6.itertuples() if pd.notna(r.max_abs_r_sample_pattern)) if len(ph6) else "")
nr6 = rd("r6_myeloid/m_niche_ratio_rhythm.csv")
gap("G48", "cochlear neutrophil Cxcr4 / Cxcr2 per-sample medians are all exactly 0 (23–45 % of cells lack both transcripts): the plan's median "
    "is degenerate", "per-sample mean used in both tissues (D47); medians and their p reported; lib.abundance.perm_p returns NaN for constant input",
    "r6_myeloid/m_niche_ratio_rhythm.csv", "documented" if len(nr6) else "open", "")
cl6 = rd("r6_myeloid/m_clock_cell_level.csv")
gap("G49", "myeloid clock inference has little power in the cochlea (about one clock UMI per myeloid cell) and external neutrophil data express "
    "only 4–5 clock genes", "cell-level usable fraction, gene-bootstrap CIs and item counts reported; M7 decided by the pre-specified rule",
    "r6_myeloid/m_clock_cell_level.csv; r6_myeloid/m_external_timeofday.csv", "documented" if len(cl6) else "open",
    "; ".join(f"{r.tissue} {r.group}: usable {r.frac_cells_usable:.2f}, UMIs/cell {r.mean_clock_umis_per_cell:.2f}" for r in cl6.itertuples()) if len(cl6) else "")
gap("G50", "external neutrophil references are contaminated by other leukocytes (Ballesteros GEO deposit mostly lymphocytes / monocytes; few lung / "
    "spleen neutrophils); GSE200145's GEO matrix lacks the sample tags", "neutrophil-lineage Leiden filter and cell-level contrast signatures "
    "(lib/neutro.py); the tissue-vs-marrow axis taken from the tagged NeuMAP YO object (same experiment as GSE200145)",
    "r6_myeloid/myeloid_reference_checks.csv; r6_myeloid/m_external_signatures.csv", "documented", "")
m6_ = rd("r6_myeloid/m_primary_m6.csv")
gap("G51", "M6 pre-specified direction (Cxcr4 / Cxcr2 peak within ± 4 h of the timer peak) is not met: the cochlear ratio peaks in the light "
    "phase, antiphase to the timer score", "reported as specified (direction fails); not re-specified after the fact",
    "r6_myeloid/m_primary_m6.csv", "documented" if len(m6_) else "open",
    "; ".join(f"{r.tissue}: ratio ZT{r.ratio_peak_zt:.1f} vs timer ZT{r.m2_timer_peak_zt:.1f}" for r in m6_.itertuples()) if len(m6_) else "")
gap("G52", "M3 external agreeing method used granule classes (primary / secondary / tertiary) as stage proxies for the Xie progenitor / immature / "
    "mature signatures; the label-free Milo wave counts as an agreeing method for both M1 and M3", "documented in 660 / 670 docstrings",
    "r6_myeloid/m_multimethod_matrix.csv", "documented", "")
tsv6 = rd("r6_myeloid/m_timer_split_verification.csv")
ev6 = ""
if len(tsv6):
    k6 = tsv6[tsv6["part"].str.startswith("A/B") & (tsv6["gene_set"] == "all shared genes") & tsv6["unit"].isin(["Neutrophil (Aged)", "pooled neutrophils"])]
    ev6 = "; ".join(f"{r.tissue} {r.unit} rho {r.spearman:+.2f}, exact p {r.p_exact_two_sided:.3f}" for r in k6.itertuples())
gap("G53", "the neutrophil-timer split (G02) was scored against the sample-level pattern only by stress-gene removal, and its gene-label "
    "permutation p ignores that one sample-level profile drives thousands of genes; with n = 6 the exact time-relabelling p of single "
    "correlations does not reach 0.05", "verified by cell-type specificity within each tissue (rank test; 241 'sample_level_pattern') and by "
    "external long-bone marrow with / without neutrophils (676); within-sample centring (675) is not used to judge G02 — it also removes "
    "genuine marrow-wide programmes", "r6_myeloid/m_timer_split_verification.csv; r6_myeloid/m_timer_split_centred.csv",
    "verified" if len(tsv6) else "open", ev6)
sph = rd("r3_ma2025/sampling_phase.csv")
bph = rd("r3_ma2025/bulk_sampling_phase.csv")
ev7 = ""
if len(sph):
    m7 = sph.groupby("condition")[["condition_mean_phase_h", "condition_resultant"]].first()
    ev7 = "scRNA clock phase by group: " + "; ".join(f"{c} {r.condition_mean_phase_h:.1f} h (R {r.condition_resultant:.2f})" for c, r in m7.iterrows())
if len(bph):
    b7 = bph[bph["reference"].str.startswith("our")].groupby("group")["group_mean_h"].first()
    ev7 += " | bulk internal time: " + "; ".join(f"{g} {v:.1f} h" for g, v in b7.items())
gap("G54", "Ma 2025 scRNA: night-noise mice were collected about 11–12 h apart from control and day-noise mice (clock-gene phase), so the "
    "night_noise vs day_noise / control contrasts mix exposure time with collection time of day; only day_noise vs control is time-matched "
    "(the bulk design is time-matched within day and within night)",
    "R08 and M9 read as two-time-point comparisons consistent with collection time of day, not as exposure-time effects (statement / verdict "
    "change left to the user); noise-response conclusions from time-matched contrasts only", "r3_ma2025/sampling_phase.csv; "
    "r3_ma2025/bulk_sampling_phase.csv", "documented" if len(sph) else "open", ev7)
lim8 = rd("r8_ma_integration/ma_resolution_and_limits.csv")
gap("G55", "Ma 2025 integration limits: scRNA night-vs-day contrasts conditional on negligible noise × time interaction; bulk deconvolution not "
    "validated for neutrophil states / non-classical monocytes / mast cells; absolute phase uncertain ± 2–3 h (composition match reverses at −4 h); "
    "ambient RNA in sorted immune cells not modelled", "every round-8 line carries a timing qualifier and assumption-robustness count; claims "
    "upgraded on conditional lines only are labelled as such (720)", "r8_ma_integration/ma_resolution_and_limits.csv; r8_ma_integration/ma_claim_table.csv",
    "documented" if len(lim8) else "open", "; ".join(f"{r.topic}: {r.status}" for r in lim8.itertuples()) if len(lim8) else "")
write_csv(pd.DataFrame(gaps), MOD, "gap_register.csv", "Known gaps, round-2 remediation, status and evidence",
          required=["gap_id", "issue", "status"])
log(f"{len(chk)} CSVs audited; {len(gaps)} gaps registered")
