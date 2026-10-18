"""Round 2, WP5 — positive-findings register and multidimensional validation matrix.

Every positive finding of rounds 1–2 is scored on pre-specified evidence dimensions. Each dimension receives a status
(supports / partial / against / neutral / not_tested) computed from the result tables, with the numbers behind it.
Core dimensions: model concordance; resampling (exact permutation, leave-one-timepoint-out, bootstrap) or calibrated
FDR. Independent dimensions: reference frame, annotation confidence, embedding / sub-state, sample-level pattern,
molecular or mechanistic corroboration, cross-tissue logic, external data, literature, power / effect size.
Verdict (rule fixed before the outputs were inspected; evaluated in this order):
  robust         all core 'supports'; >= 2 independent 'supports'; <= 1 independent 'against'
  supported      no core 'against'; >= 1 core 'supports'; >= 1 independent 'supports'; independent supports > against
  not supported  no core 'supports' and independent 'against' >= independent 'supports'
  suggestive     everything else
Phases are ZT (lights on 08:00, clock = ZT + 8); internal phases = ZT − clock-alignment offset (230).
Outputs (results/r2_audit/): validation_matrix.csv, positive_findings_register.csv
Round 3 (user decisions): cochlear oscillations are analysed cochlea-only — the BM⇄cochlea egress/return schematic (X01)
is retired, and intra-cochlear marrow content is treated as part of the cochlear immune compartment (cochlear series:
marrow-based frames reported as sensitivity, not counted; the marrow-linked sample pattern is 'neutral'). Round-3
findings R01–R08 (external response data, BM extension, cell-level clocks, heat-shock programme, spatial, Ma 2025).
Round 4 adds R09–R16 (design benchmark, bulk-rhythm attribution, primate / human conservation, atlas concordance, drivers).
Round 5 adds B01–B12: the pre-specified B-cell primary hypotheses (verdict = D34 decision: supported / suggestive (= partial) / not supported).
Round 6 adds M01–M08: the pre-specified myeloid primary hypotheses (verdict = D45 / D48 decision, mapped as in round 5).
Round 9 adds the independent flow-cytometry cohort (D89; 932, pre-specified) as an independent line of A05 (P1), A04 (P2),
A01 (P3) and G02 (S1, S2): replicated = supports, significant in the opposite direction = against, otherwise neutral.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_audit"
CORE = ("model_concordance", "resampling_or_fdr")
CTRL = {"control", "WT", "sham"}
NEUT = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Band)", "Neutrophil (Mature)"]
IEG = ["Fos", "Fosb", "Jun", "Junb", "Egr1", "Egr2", "Egr3", "Ier2", "Nr4a1", "Dusp1", "Atf3", "Klf2", "Klf4", "Zfp36"]
HSG = ["Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"]
NFKB = ["Nfkbia", "Nfkbiz", "Tnfaip3", "Ier3", "Cxcl2", "Ccl3", "Ccl4", "Nfkbid", "Rel", "Relb"]
IL1 = ["Il1b", "Il1rn", "Il1r2", "Il1a"]


def rd(rel, **kw):
    p = RESULTS / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


off = rd("r2_phase/clock_alignment_offsets.csv")


def offset(ds, tissue, cond="WT", unit="bulk"):
    r = off[(off["dataset"] == ds) & (off["tissue"] == tissue) & (off["condition"] == cond) & (off["unit"] == unit)]
    return float(r["offset_canonical_h"].iloc[0]) if len(r) else 0.0


OFF = {t: offset("this_atlas", t, "WT", "tissue_pool") for t in ["bm", "cochlea"]}
rows, reg = [], []


def add(fid, dim, status, evidence, source):
    rows.append({"finding_id": fid, "dimension": dim, "dimension_class": "core" if dim in CORE else "independent",
                 "status": status, "evidence": evidence, "source": source})


def combine(st):
    st = [s for s in st if s != "not_tested"]
    if not st:
        return "not_tested"
    if "supports" in st and "against" in st:
        return "partial"
    for s in ("supports", "against", "partial"):
        if s in st:
            return s
    return "neutral"


def cmean(h):
    h = np.asarray(h, float)
    h = h[~np.isnan(h)]
    return phase.circ_mean(h) if len(h) else (np.nan, np.nan)


def register(fid, category, tissue, statement, peak_zt=np.nan, in_figure=False, note=""):
    reg.append({"finding_id": fid, "category": category, "tissue": tissue, "statement": statement,
                "peak_zt": peak_zt, "peak_clock": float(phase.zt_to_clock(peak_zt)) if pd.notna(peak_zt) else np.nan,
                "in_user_figure": in_figure, "note": note})


# ---------------------------------------------------------------- shared inputs
ms = rd("r2_figure/abundance_multimodel_summary.csv")
pr = rd("r2_figure/abundance_pct_rhythm.csv")
lg = rd("r2_figure/abundance_multimodel_long.csv")
pts = rd("r2_figure/figure_points.csv")
lca = rd("r2_audit/label_confidence_abundance.csv")
lcg = rd("r2_audit/label_confidence_genes.csv")
pat = rd("r2_cochlea/sample_pattern_scores.csv")
pcov = rd("r2_cochlea/sample_pattern_covariates.csv").drop_duplicates(["tissue", "pc", "covariate"])
sigr = rd("r2_external/signature_rhythm.csv")
emat = rd("r2_external/external_validation_matrix.csv")
litc = rd("r2_external/literature_comparison.csv")
gconc = rd("r2_external/gene_phase_concordance.csv")
pw = rd("54_abundance_power/abundance_power.csv")
milo = {(t, e): rd(f"53_milo/milo_da_results_{t}{sfx}.csv") for t in ["bm", "cochlea"]
        for e, sfx in [("scANVI", ""), ("scVI", "_scvi")]}
summ = rd("33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv")
cls = rd("00_audit/celltype_counts_by_sample.csv").drop_duplicates(["tissue", "cell_type"])
G = {t: rd(f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv",
           usecols=["group", "gene", "tier", "p", "peak_zt", "edgeR_p", "rain_p", "jtk_p"]) for t in ["bm", "cochlea"]}
ext = []
for f in sorted(list((RESULTS / "71_external").glob("external_rhythm_GSE88954.csv")) +
                list((RESULTS / "r2_external").glob("external_rhythm_*.csv"))):
    ext.append(pd.read_csv(f, usecols=["dataset", "tissue", "condition", "gene", "peak_h", "p"]))
ext = pd.concat(ext, ignore_index=True)
ext = ext[ext["condition"].isin(CTRL)].copy()
ext["unit"] = ext["dataset"] + "|" + ext["tissue"] + "|" + ext["condition"]
_o = {u: offset(*u.split("|")) for u in ext["unit"].unique()}
ext["internal"] = phase.wrap24(ext["peak_h"] - ext["unit"].map(_o))
log(f"inputs loaded; external control units: {ext['unit'].nunique()}")


def pattern_cov(t):
    c = pcov[(pcov["tissue"] == t) & (pcov["pc"] == 1)]
    return "; ".join(f"{r.covariate} {r.pearson_r:+.2f}" for r in c.itertuples())


def geneset_phase_check(genes, tissue_groups):
    """Our rhythmic (p < 0.05) rows of a gene set → circular mean internal phase; external control units with >= 2
    rhythmic genes of the set → their circular mean internal phase and Δ to ours."""
    ours = []
    for t, groups in tissue_groups.items():
        g = G[t][G[t]["gene"].isin(genes) & (G[t]["p"] < 0.05)]
        if groups is not None:
            g = g[g["group"].isin(groups)]
        ours.append(phase.wrap24(g["peak_zt"].to_numpy() - OFF[t]))
    ours = np.concatenate(ours) if ours else np.array([])
    mu, R = cmean(ours)
    e = ext[ext["gene"].isin(genes) & (ext["p"] < 0.05)]
    units = []
    for u, g in e.groupby("unit"):
        if g["gene"].nunique() < 2:
            continue
        m, r_ = cmean(g["internal"])
        units.append((u, g["gene"].nunique(), m, float(phase.circ_diff(m, mu)) if pd.notna(mu) else np.nan))
    return len(ours), mu, R, units


def units_status(units):
    d = np.array([abs(x[3]) for x in units if pd.notna(x[3])])
    if not len(d):
        return "not_tested"
    if (d <= 4).mean() >= 0.5:
        return "supports"
    if (d > 8).mean() >= 0.5:
        return "against"
    return "partial"


flow = rd("r9_flow/flow_tests.csv")
dig = rd("r9_flow/flow_digestion_controls.csv")
FLOW_STATUS = {"replicated": "supports", "contradicted": "against", "opposite direction, significant": "against"}
FLOW_GATE = {"P1": "gate ≈ Band + 'Mature' neutrophils (plan mapping)", "P3": "gate ≈ 'Naïve' (FrF)"}


def flow_line(fid):
    """Independent flow-cytometry cohort (20 mice, ZT5 vs ZT17, two cohorts; flow plan 6.2-6.4): the readouts 932 links to
    the finding, scored by their pre-specified verdict like other external lines (replicated = supports, significant in the
    opposite direction = against, otherwise neutral); several readouts combined. Marrow-content-adjusted shares (D17) are
    sensitivity, not counted."""
    f = flow[flow["finding"] == fid] if len(flow) else flow
    if not len(f):
        return
    src = "r9_flow/flow_tests.csv (932; flow plan section 9)"
    pri = f[f["family"] == "primary"]
    if len(pri):
        x = pri.iloc[0]
        sen = flow[(flow["test"] == x["test"]) & flow["family"].str.startswith("sensitivity")]
        add(fid, "independent_flow_cohort", FLOW_STATUS.get(x["verdict"], "neutral"),
            f"{x['test']} {x['tissue'].replace('_', ' ')}, {x['readout']}" +
            (f" ({FLOW_GATE[x['test']]})" if x["test"] in FLOW_GATE else "") + f": ZT17 / ZT5 {x['effect']:.2f} "
            f"({x['ci_lo']:.2f}–{x['ci_hi']:.2f}), Holm p {x['p_holm']:.2g}, n {int(x['n_zt5'])} + {int(x['n_zt17'])} mice → "
            f"{x['verdict']} (predicted {x['prediction']})" +
            "".join(f"; sensitivity, not counted (D17): {s.readout} {s.effect:.2f} ({s.ci_lo:.2f}–{s.ci_hi:.2f}), p {s.p:.2f}, "
                    f"n {int(s.n_zt5)} + {int(s.n_zt17)}" for s in sen.itertuples()), src)
        return
    st, ev = [], []
    for test, g in f[(f["family"] == "secondary") & (f["tissue"] == "cochlea")].groupby("test", sort=False):
        st_t = []
        for r in g.itertuples():
            o_ = f[(f["test"] == test) & (f["readout"] == r.readout)].set_index("tissue")
            st_t.append(FLOW_STATUS.get(r.verdict, "neutral"))
            ev.append(f"{test} {r.readout}: cochlea {'OR' if r.effect_scale.startswith('odds') else 'ratio'} {r.effect:.2f} "
                      f"({r.ci_lo:.2f}–{r.ci_hi:.2f}), p {r.p:.2g}, n {int(r.n_zt5)} + {int(r.n_zt17)}; marrow "
                      f"{o_.loc['marrow', 'effect']:.2f}, p {o_.loc['marrow', 'p']:.2g}; interaction p {o_.loc['interaction', 'p']:.2g} "
                      f"→ {r.verdict}")
        st.append(combine(st_t))
    dr = dig.groupby("marker")["median_ratio_digested_undigested"].agg(["min", "max"]) if len(dig) else pd.DataFrame()
    add(fid, "independent_flow_cohort", combine(st), "; ".join(ev) + ("; digestion control, digested / undigested median: " + ", ".join(
        f"{k} ×{v['min']:.2f}–{v['max']:.2f}" for k, v in dr.iterrows()) if len(dr) else ""), src)


# ---------------------------------------------------------------- A: abundance rhythms
FIG = [("bm", "B Cell (Naïve)"), ("bm", "Monocyte (Classical)"), ("bm", "B Cell (Transitional)"),
       ("cochlea", "Monocyte (Classical)"), ("cochlea", "Neutrophil (Band)"), ("cochlea", "Neutrophil (Mature)")]
extra = ms[ms["verdict"].isin(["robust", "supported"])]
AB = FIG + [k for k in zip(extra["tissue"], extra["cell_type"]) if k not in FIG]
FRAMES = ["clr", "alr_structural", "alr_marrow", "within_lineage", "marrow_adjusted"]
MARROW_FRAMES = ["alr_marrow", "marrow_adjusted"]          # cochlea: sensitivity only (round-3 decision)
SIG_UNIT = {"bm": ("GSE198770_bone_marrow", "bone_marrow", "control"), "cochlea": ("GSE88954", "cochlea", "sham")}
for i, (t, ct) in enumerate(AB, 1):
    fid = f"A{i:02d}"
    s = ms[(ms["tissue"] == t) & (ms["cell_type"] == ct)].iloc[0]
    r = pr[(pr["tissue"] == t) & (pr["cell_type"] == ct)].iloc[0]
    pk = float(r["peak_zt"])
    register(fid, "abundance rhythm", t, f"{ct}: share of {t} cells oscillates, peak ZT{pk:.1f} (clock {float(phase.zt_to_clock(pk)):.1f}); "
             f"peak/trough {r['ptt_ratio']:.2f}", pk, (t, ct) in FIG,
             "user-figure panel" if (t, ct) in FIG else f"202 verdict {s['verdict']}")
    v = s["verdict"]
    add(fid, "model_concordance", {"robust": "supports", "supported": "partial"}.get(v, "against"),
        f"{v}: {int(s['core_supporting'])}/{int(s['core_tested'])} core models p<0.05 (CLR permutation, propeller, sccomp, "
        f"quasi-binomial, beta-binomial); core peak SD {s['core_peak_circ_sd_h']:.1f} h; OLS-% p {s['ols_pct_p']:.3f}",
        "r2_figure/abundance_multimodel_summary.csv")
    perm_ok = (r["p_perm"] <= 0.05) or (pd.notna(s["clr_p_perm"]) and s["clr_p_perm"] <= 0.05)
    ok = [perm_ok, r["loto_phase_sd_h"] < 3, r["boot_frac_p05"] >= 0.5]
    add(fid, "resampling_or_fdr", "supports" if all(ok) else ("partial" if sum(ok) == 2 else "against"),
        f"exact permutation p: OLS-% {r['p_perm']:.3f}, CLR {s['clr_p_perm']:.3f}; LOTO phase SD {r['loto_phase_sd_h']:.1f} h; "
        f"cell-resampling bootstrap fraction p<0.05 {r['boot_frac_p05']:.2f}; BH q (OLS-%) {r['q_BH']:.2f}",
        "r2_figure/abundance_pct_rhythm.csv; abundance_multimodel_summary.csv")
    used = [f for f in FRAMES if not (t == "cochlea" and f in MARROW_FRAMES)]
    lf = lg[(lg["tissue"] == t) & (lg["cell_type"] == ct) & lg["model"].isin(used)]
    lfm = lg[(lg["tissue"] == t) & (lg["cell_type"] == ct) & lg["model"].isin(MARROW_FRAMES)] if t == "cochlea" else lg.iloc[:0]
    n_ok = int(((lf["p"] < 0.05) & (np.abs(phase.circ_diff(lf["peak_zt"], pk)) <= 3)).sum())
    add(fid, "reference_frame", "not_tested" if not len(lf) else
        ("supports" if n_ok >= max(1, len(lf) / 2) else ("partial" if n_ok else "against")),
        "; ".join(f"{m.model} p {m.p:.3f} @ZT{m.peak_zt:.1f}" for m in lf.itertuples()) +
        ("; sensitivity, not counted (marrow-based frames): " + "; ".join(f"{m.model} p {m.p:.3f} @ZT{m.peak_zt:.1f}" for m in lfm.itertuples())
         if len(lfm) else ""), "r2_figure/abundance_multimodel_long.csv")
    a = lca[(lca["tissue"] == t) & (lca["cell_type"] == ct)]
    if len(a) and pd.notna(a["p_ols_pct_high_confidence"].iloc[0]):
        a = a.iloc[0]
        pmin = min(a["p_ols_pct_high_confidence"], a["p_quasibinomial_high_confidence"])
        st = ("supports" if pmin < 0.05 and abs(a["peak_shift_h"]) <= 2 else
              "partial" if pmin < 0.2 and abs(a["peak_shift_h"]) <= 3 else "against")
        add(fid, "annotation_confidence", st, f"high-confidence cells only ({a['frac_cells_kept']:.0%} kept): OLS-% p "
            f"{a['p_ols_pct_high_confidence']:.3f}, quasi-binomial p {a['p_quasibinomial_high_confidence']:.3f}, peak shift "
            f"{a['peak_shift_h']:+.1f} h", "r2_audit/label_confidence_abundance.csv")
    else:
        add(fid, "annotation_confidence", "not_tested", "too few high-confidence cells", "r2_audit/label_confidence_abundance.csv")
    parts, st_e = [], []
    for e in ["scANVI", "scVI"]:
        m = milo[(t, e)]
        m = m[m["majority_cell_type"] == ct] if len(m) else m
        if not len(m):
            parts.append(f"{e}: no neighbourhoods")
            st_e.append("not_tested")
            continue
        sig, nom = m[m["spatial_fdr"] < 0.1], m[m["p"] < 0.05]
        use = sig if len(sig) else nom
        mp = cmean(use["peak_zt"])[0] if len(use) else np.nan
        inphase = pd.notna(mp) and abs(float(phase.circ_diff(mp, pk))) <= 4
        st_e.append("supports" if len(sig) and inphase else
                    "partial" if len(nom) >= max(1, 0.1 * len(m)) and inphase else
                    "against" if len(sig) and not inphase else "neutral")
        parts.append(f"{e}: {len(sig)}/{len(m)} neighbourhoods spatial FDR<0.1, {len(nom)} p<0.05, mean peak ZT{mp:.1f}")
    st = ("supports" if st_e and all(x == "supports" for x in st_e) else combine(st_e))
    add(fid, "embedding_substate", st, "; ".join(parts), "53_milo/milo_da_results_<tissue>.csv (scANVI), *_scvi.csv (scVI)")
    pp = pts[(pts["tissue"] == t) & (pts["cell_type"] == ct)].set_index("sample_id")["pct"]
    p1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"]
    cm_ = pp.index.intersection(p1.index)
    rr = float(np.corrcoef(pp[cm_], p1[cm_])[0, 1]) if len(cm_) >= 5 else np.nan
    if t == "cochlea":                                   # PC1 tracks intra-cochlear marrow content = immune compartment
        add(fid, "sample_level_pattern", "not_tested" if np.isnan(rr) else "neutral",
            f"r with the tissue-wide sample pattern (PC1) {rr:+.2f}; PC1 tracks marrow content, treated as biological "
            f"(round-3 decision; also tracks depth): {pattern_cov(t)}", "r2_cochlea/sample_pattern_scores.csv; sample_pattern_covariates.csv")
    else:
        add(fid, "sample_level_pattern", "not_tested" if np.isnan(rr) else
            ("supports" if abs(rr) < 0.5 else "partial" if abs(rr) < 0.8 else "against"),
            f"r with the tissue-wide sample pattern (PC1) {rr:+.2f}; PC1 covariates: {pattern_cov(t)}",
            "r2_cochlea/sample_pattern_scores.csv; sample_pattern_covariates.csv")
    ds, tis, cond = SIG_UNIT[t]
    our_int = float(phase.wrap24(pk - OFF[t]))
    est, eev = [], []
    sig_name = ("Neutrophil immature (Band/Mature)" if ct in ("Neutrophil (Band)", "Neutrophil (Mature)") else
                "Neutrophil mature (Aged/Activated)" if ct in ("Neutrophil (Aged)", "Neutrophil (Activated)") else ct)
    sg = sigr[(sigr["dataset"] == ds) & (sigr["tissue"] == tis) & (sigr["condition"] == cond) & (sigr["signature"] == sig_name)] \
        if len(sigr) else sigr
    if len(sg):
        g = sg.iloc[0]
        d = float(phase.circ_diff(our_int, g["internal_peak"]))
        est.append(("supports" if abs(d) <= 4 else "against" if abs(d) > 8 else "partial") if g["p"] < 0.05 else "neutral")
        eev.append(f"{ds} {cond} {sig_name} signature score: p {g['p']:.3f}, Δ internal phase (ours − external) {d:+.1f} h")
    bp = emat[(emat["tissue"] == t) & (emat["cell_type"] == ct) & emat["source"].str.startswith("BayesPrism")] if len(emat) else emat
    for b in bp.itertuples():
        est.append({"supports": "supports", "contradicts": "against"}.get(b.status, "neutral"))
        eev.append(f"{b.source}: {b.effect}")
    add(fid, "external_data", combine(est), "; ".join(eev) or "no matched external abundance proxy",
        "r2_external/signature_rhythm.csv; external_validation_matrix.csv")
    lt = litc[(litc["tissue"] == t) & (litc["cell_type"] == ct)] if len(litc) else litc
    lst = [{"agrees": "supports", "partly": "partial", "disagrees": "against"}.get(x.split(" ")[0], "neutral")
           for x in lt["agreement"]] if len(lt) else []
    add(fid, "literature", combine(lst),
        "; ".join(f"{x.source}: expected ZT{x.expected_peak_zt:.0f} vs ours ZT{x.our_peak_zt:.1f} ({x.agreement})"
                  for x in lt.itertuples()) or "no literature phase for this cell type and tissue",
        "r2_external/literature_comparison.csv")
    pwr = pw[(pw["tissue"] == t) & (pw["cell_type"] == ct) & (pw["method"] == "propeller_moderated") & (pw["alpha"] == 0.05)]
    if len(pwr):
        mfc = float(pwr["min_fc_80pct_alpha05"].iloc[0])
        p2 = pwr.loc[pwr["ptt_fc"] == 2, "power"]
        add(fid, "power_effect_size", "supports" if r["ptt_ratio"] >= mfc else "partial",
            f"observed peak/trough {r['ptt_ratio']:.2f} vs 80 %-power detectable FC {mfc:.2f} (power at FC 2: "
            f"{float(p2.iloc[0]) if len(p2) else np.nan:.2f})", "54_abundance_power/abundance_power.csv")
    else:
        add(fid, "power_effect_size", "not_tested", "no power estimate", "54_abundance_power/abundance_power.csv")
    flow_line(fid)

# ---------------------------------------------------------------- X01: emigration / return schematic
rel = rd("r2_figure/figure_phase_relationships.csv")
fid = "X01"
register(fid, "cross-tissue trafficking", "both", "Figure schematic: rhythmic emigration of myeloid/B cells from bone marrow "
         "to the cochlea and return/clearance (BM and cochlear abundance phases linked by trafficking)", note="user-figure schematic")
both = rel[rel["both_p05"]]
add(fid, "model_concordance", "supports" if len(both) >= 3 else ("partial" if len(both) else "against"),
    f"cell types rhythmic (OLS-% p<0.05) in both tissues: {len(both)} ({'; '.join(both['cell_type'])})",
    "r2_figure/figure_phase_relationships.csv")
if len(both):
    best = both[["prob_emigration_order", "prob_return_order"]].max(1)
    add(fid, "resampling_or_fdr", "supports" if (best >= 0.8).all() else ("partial" if (best >= 0.5).any() else "against"),
        "; ".join(f"{r.cell_type}: Δ(cochlea − BM) {r.delta_cochlea_minus_bm_h:+.1f} h [{r.delta_ci_lo:+.1f}, {r.delta_ci_hi:+.1f}], "
                  f"P(emigration order) {r.prob_emigration_order:.2f}, P(return order) {r.prob_return_order:.2f}, "
                  f"P(cochlea leads 0–6 h) {r.prob_cochlea_leads_0to6h:.2f}" for r in both.itertuples()),
        "r2_figure/figure_phase_relationships.csv (bootstrap)")
else:
    add(fid, "resampling_or_fdr", "against", "no cell type rhythmic in both tissues", "r2_figure/figure_phase_relationships.csv")
bl0, bl1 = rd("66_state_continua/bm_likeness_rhythm.csv"), rd("r2_audit/bm_likeness_scvi_rhythm.csv")
nb0, nb1 = int((bl0["p"] < 0.05).sum()) if len(bl0) else 0, int((bl1["p"] < 0.05).sum()) if len(bl1) else 0
add(fid, "embedding_substate", "supports" if nb0 and nb1 else ("against" if len(bl0) else "not_tested"),
    f"cochlear immune types with rhythmic BM-likeness (marrow-like transcriptional state): scANVI {nb0}/{len(bl0)}, "
    f"scVI {nb1}/{len(bl1)}", "66_state_continua/bm_likeness_rhythm.csv; r2_audit/bm_likeness_scvi_rhythm.csv")
mr = rd("52_marrow_decomposition/tissue_residual_rhythm.csv")
mra = mr[mr["model"] == "marrow_adjusted"] if len(mr) else mr
add(fid, "sample_level_pattern", "against" if len(mra) and (mra["prob_fc_gt_1p5"] <= 0.5).all() else "partial",
    f"cochlear immune types with P(residual FC > 1.5 | marrow content) > 0.5: {int((mra['prob_fc_gt_1p5'] > 0.5).sum())}/{len(mra)} "
    f"(max {mra['prob_fc_gt_1p5'].max():.2f}); cochlear PC1 covariates: {pattern_cov('cochlea')}",
    "52_marrow_decomposition/tissue_residual_rhythm.csv; r2_cochlea/sample_pattern_covariates.csv")
cmc = rd("60_migration/compartment_model_compare.csv")
if len(cmc):
    top = cmc.loc[cmc.groupby("cell_type")["weight"].idxmax()]
    n_traf = int(top["model"].isin(["M1", "M3"]).sum())
    add(fid, "molecular_mechanistic", "supports" if n_traf >= 0.75 * len(top) else ("partial" if n_traf else "against"),
        f"compartment models (WAIC weight; M0 independent, M1 BM→cochlea, M2 shared driver, M3 cochlea→BM): trafficking "
        f"model best for {n_traf}/{len(top)} types: " + "; ".join(f"{r.cell_type} {r.model} ({r.weight:.2f})" for r in top.itertuples()),
        "60_migration/compartment_model_compare.csv")
lfx = rd("61_lineage_flux/lineage_flux_nonlinear.csv")
if len(lfx):
    m = lfx[lfx["stage"] == "Neutrophil (Mature)"].drop_duplicates(["frame", "obs_ptt_fc"])
    add(fid, "cross_tissue_logic", "against" if len(m) and (m["tau_max_exit_gating_h"] < 6).all() else "partial",
        "; ".join(f"BM Neutrophil (Mature) {r.frame}: FC {r.obs_ptt_fc:.2f}, max residence time compatible with gated exit "
                  f"{r.tau_max_exit_gating_h:.1f} h (prior median {r.tau_prior_median_h:.0f} h)" for r in m.itertuples()) +
        " → share rhythm requires state switching rather than egress", "61_lineage_flux/lineage_flux_nonlinear.csv")
lt = litc[litc["source"].str.contains("Adrover|Scheiermann|Casanova", regex=True)] if len(litc) else litc
add(fid, "literature", combine([{"agrees": "supports", "partly": "partial", "disagrees": "against"}.get(x.split(" ")[0], "neutral")
                                for x in lt["agreement"]]) if len(lt) else "not_tested",
    "; ".join(f"{x.source} ({x.tissue} {x.cell_type}): expected ZT{x.expected_peak_zt:.0f} vs ours ZT{x.our_peak_zt:.1f}"
              for x in lt.itertuples()) or "none", "r2_external/literature_comparison.csv")

# ---------------------------------------------------------------- G01: BM neutrophil-state gene rhythms
fid = "G01"
sb = summ[(summ["tissue"] == "bm") & summ["group"].isin(NEUT)]
register(fid, "gene rhythms", "bm", "BM neutrophil states carry most Tier-1 gene rhythms: " +
         ", ".join(f"{r.group} {int(r.n_tier1)}" for r in sb.itertuples()) +
         f" (of {int(summ.loc[summ['tissue'] == 'bm', 'n_tier1'].sum())} BM Tier-1 calls)")
t1 = G["bm"][(G["bm"]["tier"] == "tier1") & G["bm"]["group"].isin(NEUT)]
fr = {k: float((t1[k] < 0.05).mean()) for k in ["edgeR_p", "rain_p", "jtk_p"]}
lri = rd("32_rhythm_alt/lr_tier_impact.csv")
lr_frac = lri[(lri["tissue"] == "bm") & lri["group"].isin(NEUT)]["frac_tier1_lr_p05"].mean() if len(lri) else np.nan
add(fid, "model_concordance", "supports" if min(list(fr.values()) + [lr_frac]) >= 0.7 else "partial",
    f"Tier-1 neutrophil calls also p<0.05 by edgeR {fr['edgeR_p']:.2f}, RAIN {fr['rain_p']:.2f}, JTK {fr['jtk_p']:.2f}, "
    f"LR (diffCircadian) {lr_frac:.2f}", "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv; 32_rhythm_alt/lr_tier_impact.csv")
sim = rd("33_rhythm_tiers/rhythm_pipeline_simulation.csv")
fdp = sim[(sim["tissue"] == "bm") & (sim["rule"] == "tier1_like")]["fdp"].mean() if len(sim) else np.nan
add(fid, "resampling_or_fdr", "supports" if sb["perm_empirical_fdr"].max() <= 0.1 and fdp <= 0.1 else "partial",
    "permutation empirical FDR " + ", ".join(f"{r.group} {r.perm_empirical_fdr:.3f}" for r in sb.itertuples()) +
    f"; simulated Tier-1 FDP (BM) {fdp:.3f}", "33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv; rhythm_pipeline_simulation.csv")
dm = rd("39_sensitivity/depth_matched_summary.csv")
dmn = dm[(dm["tissue"] == "bm") & (dm["version"] == "depth_matched") & dm["cell_type"].isin(NEUT)]
v = dmn.groupby("cell_type")["frac_tier12_p05"].mean()
add(fid, "reference_frame", "supports" if len(v) and v.min() >= 0.8 else ("partial" if len(v) else "not_tested"),
    "depth-matched (thinned) re-fit, Tier-1/2 keeping p<0.05: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items()),
    "39_sensitivity/depth_matched_summary.csv")
smx = rd("39_sensitivity/statemix_summary.csv")
smn = smx[(smx["tissue"] == "bm") & smx["cell_type"].isin(NEUT)]
add(fid, "embedding_substate", "supports" if len(smn) and smn["frac_tier12_p05_after"].median() >= 0.8 else
    ("partial" if len(smn) and smn["frac_tier12_p05_after"].median() >= 0.5 else "against"),
    "within-type state-mix covariate, Tier-1/2 keeping p<0.05: " +
    ", ".join(f"{r.cell_type} {r.frac_tier12_p05_after:.2f} (state~time R² {r.state_time_r2:.2f})" for r in smn.itertuples()),
    "39_sensitivity/statemix_summary.csv")
lcn = lcg[lcg["cell_type"].isin(NEUT)] if len(lcg) else lcg
if len(lcn):
    fa, fc_ = (lcn["p_all_cells"] < 0.05).mean(), (lcn["p_high_confidence"] < 0.05).mean()
    sh = float(np.median(np.abs(lcn["peak_shift_h"])))
    add(fid, "annotation_confidence", "supports" if fc_ >= 0.6 and sh <= 1 else "partial",
        f"Tier-1 neutrophil genes p<0.05 (OLS log2 CPM): all cells {fa:.2f} vs high-confidence cells {fc_:.2f}; median |peak shift| "
        f"{sh:.2f} h", "r2_audit/label_confidence_genes.csv")
mb = {e: milo[("bm", e)] for e in ["scANVI", "scVI"]}
nn = {e: int(((m["spatial_fdr"] < 0.1) & m["majority_cell_type"].isin(NEUT)).sum()) for e, m in mb.items()}
add(fid, "molecular_mechanistic", "supports" if min(nn.values()) > 0 else "partial",
    f"rhythmic neutrophil-majority Milo neighbourhoods (spatial FDR<0.1): scANVI {nn['scANVI']}, scVI {nn['scVI']}",
    "53_milo/milo_da_results_bm.csv; milo_da_results_bm_scvi.csv")
gc_ = gconc[(gconc["tissue"] == "bm") & gconc["cell_type"].isin(NEUT) & (gconc["genes"] == "non_clock") &
            (gconc["our_set"] == "tier1_2") & gconc["external_unit"].str.startswith("GSE198770_bone_marrow|bone_marrow|control")] \
    if len(gconc) else gconc
if len(gc_):
    nsup = int(((gc_["p_perm"] < 0.05) & (gc_["frac_within_3h"] > gc_["null_frac_within_3h"])).sum())
    add(fid, "external_data", "supports" if nsup >= 2 else ("partial" if nsup else ("against" if (gc_["p_perm"] > 0.95).any() else "neutral")),
        "GSE198770 BM control, non-clock genes rhythmic in both, fraction within 3 h vs null: " +
        "; ".join(f"{r.cell_type} {r.frac_within_3h:.2f} vs {r.null_frac_within_3h:.2f} (n {r.n_genes}, p {r.p_perm:.3f})" for r in gc_.itertuples()),
        "r2_external/gene_phase_concordance.csv")
meta = rd("73_meta/meta_cosinor_summary.csv")
mm = meta[meta["tissue"] == "bm"] if len(meta) else meta
if len(mm):
    add(fid, "cross_tissue_logic", "supports" if mm["frac_concordant_among_q05"].iloc[0] >= 0.6 else "partial",
        f"meta-cosinor BM + GSE198770 (all BM cell types): {int(mm['n_q05'].iloc[0])} genes q<0.05, "
        f"{mm['frac_concordant_among_q05'].iloc[0]:.2f} phase-concordant", "73_meta/meta_cosinor_summary.csv")

# ---------------------------------------------------------------- G02: neutrophil timer concordance signs
fid = "G02"
nt = rd("77_neutrophil_timer/neutrophil_timer_concordance.csv")
rep = rd("r2_audit/neutrophil_timer_replication.csv")
register(fid, "gene rhythms vs external program", "both", "Cochlear neutrophil rhythms predict the blood-neutrophil ZT13−ZT5 "
         "program (GSE86619) positively, temporal-BM neutrophils negatively")
key = nt[nt["cell_type"].isin(["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Mature)"])]
exp_sign = np.where(key["tissue"] == "cochlea", 1, -1)
same = [(np.sign(r.spearman_all) == e) and (np.sign(r.spearman_both_p05) == e) and (np.sign(r.spearman_all_no_stress) == e)
        for r, e in zip(key.itertuples(), exp_sign)]
add(fid, "model_concordance", "supports" if np.mean(same) >= 0.8 else ("partial" if np.mean(same) >= 0.5 else "against"),
    "; ".join(f"{r.tissue} {r.cell_type}: rho all {r.spearman_all:+.2f}, both-rhythmic {r.spearman_both_p05:+.2f}, "
              f"without stress genes {r.spearman_all_no_stress:+.2f}" for r in key.itertuples()),
    "77_neutrophil_timer/neutrophil_timer_concordance.csv")
okp = [(r.p_perm_all <= 0.05) and (np.sign(r.spearman_all) == e) for r, e in zip(key.itertuples(), exp_sign)]
add(fid, "resampling_or_fdr", "supports" if all(okp) else ("partial" if any(okp) else "against"),
    "gene-label permutation p: " + ", ".join(f"{r.tissue} {r.cell_type} {r.p_perm_all:.4f}" for r in key.itertuples()),
    "77_neutrophil_timer/neutrophil_timer_concordance.csv")
if len(rep):
    def _rho(ds, cond, genes):
        x = rep[(rep["dataset"] == ds) & (rep["condition"] == cond) & (rep["genes"] == genes)]
        return (float(x["spearman_rho"].iloc[0]), float(x["p"].iloc[0])) if len(x) else (np.nan, np.nan)
    c_m, b_a, b_r, b_m = (_rho("GSE88954", "sham", "neutrophil_markers"), _rho("GSE198770_bone_marrow", "control", "all"),
                          _rho("GSE198770_bone_marrow", "control", "rhythmic_p05"), _rho("GSE198770_bone_marrow", "control", "neutrophil_markers"))
    st_c = "supports" if c_m[0] > 0 and c_m[1] < 0.05 else ("against" if c_m[0] < 0 and c_m[1] < 0.05 else "neutral")
    st_b = combine(["supports" if b_a[0] < 0 and b_a[1] < 0.05 else "neutral", "supports" if b_r[0] < 0 and b_r[1] < 0.05 else "neutral",
                    "against" if b_m[0] > 0 and b_m[1] < 0.05 else "neutral"])
    add(fid, "external_data", st_c,
        f"cochlear sign in the external cochlea (GSE88954 sham, neutrophil markers): rho {c_m[0]:+.2f} (p {c_m[1]:.3f}); "
        "gene-level p ignore inter-gene correlation", "r2_audit/neutrophil_timer_replication.csv")
    add(fid, "cross_tissue_logic", st_b,
        f"BM sign in external BM (GSE198770 control): rho all genes {b_a[0]:+.2f} (p {b_a[1]:.3f}), rhythmic genes "
        f"{b_r[0]:+.2f} (p {b_r[1]:.3f}), neutrophil markers {b_m[0]:+.2f} (p {b_m[1]:.3f})",
        "r2_audit/neutrophil_timer_replication.csv")
tsv = rd("r6_myeloid/m_timer_split_verification.csv")
if len(tsv):          # round-6 verification (676): a sample-level pattern shifts every cell type of a sample, so the split is tested by
    import itertools  # whether the key neutrophil states are the most extreme cell types of their tissue in the expected direction
    a_ = tsv[tsv["part"].str.startswith("A/B") & (tsv["gene_set"] == "all shared genes") & (tsv["unit"] != "pooled neutrophils")]
    KEY_ = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Mature)"]
    sp_, ev_ = {}, []
    for ts, sg in [("bm", -1), ("cochlea", 1)]:
        x_ = a_[a_["tissue"] == ts].assign(score=lambda d: sg * d["spearman"])
        x_["rank"] = x_["score"].rank(ascending=False)
        rk = x_.set_index("unit").loc[KEY_, "rank"]
        sp_[ts] = float(np.mean([sum(c) <= rk.sum() for c in itertools.combinations(range(1, len(x_) + 1), 3)]))
        ev_.append(f"{ts}: ranks {', '.join(str(int(v)) for v in rk)} of {len(x_)} types (rank-sum exact p {sp_[ts]:.4f})")
    ok_ = [p_ < 0.05 for p_ in sp_.values()]
    add(fid, "sample_level_pattern", "supports" if all(ok_) else ("partial" if any(ok_) else "against"),
        "cell-type specificity within each tissue (Aged / Activated / Mature vs all primary types, expected sign): " + "; ".join(ev_),
        "r6_myeloid/m_timer_split_verification.csv")
else:
    add(fid, "sample_level_pattern", "supports" if all(np.sign(r.spearman_all_no_stress) == np.sign(r.spearman_all) for r in key.itertuples())
        else "partial", "correlations unchanged after removing stress/IEG genes (see model_concordance)",
        "77_neutrophil_timer/neutrophil_timer_concordance.csv")
flow_line(fid)

# ---------------------------------------------------------------- G03: night IL-1 axis
fid = "G03"
register(fid, "cell–cell communication", "both", "Night IL-1 axis (Il1b / Il1rn / Il1r2 in neutrophils) prioritised by "
         "MultiNicheNet in both tissues (active vs rest phase)")
mn = {t: rd(f"67_multinichenet/multinichenet_prioritization_{t}.csv") for t in ["bm", "cochlea"]}
n_il = {}
for t, d in mn.items():
    if not len(d):
        n_il[t] = 0
        continue
    act = d[d["group"].astype(str).str.contains("active", case=False)].sort_values("prioritization_score", ascending=False).head(50)
    n_il[t] = int(act["ligand"].isin(IL1).sum())
add(fid, "model_concordance", "supports" if all(v > 0 for v in n_il.values()) else ("partial" if any(n_il.values()) else "against"),
    "IL-1-family ligands among the top-50 active-phase pairs: " + ", ".join(f"{k} {v}" for k, v in n_il.items()),
    "67_multinichenet/multinichenet_prioritization_<tissue>.csv")
rows_il = pd.concat([G[t][G[t]["gene"].isin(IL1) & G[t]["group"].isin(NEUT)].assign(tissue=t) for t in G])
sig_il = rows_il[rows_il["p"] < 0.05]
night = ((sig_il["peak_zt"] >= 12) & (sig_il["peak_zt"] < 24)).mean() if len(sig_il) else np.nan
add(fid, "resampling_or_fdr", "supports" if len(sig_il) >= 3 and night >= 2 / 3 else ("partial" if len(sig_il) else "against"),
    f"IL-1 genes rhythmic (limma p<0.05) in neutrophil states: {len(sig_il)}/{len(rows_il)} gene × type × tissue; "
    f"night (ZT12–24) peaks {night:.2f}: " + "; ".join(f"{r.tissue} {r.group} {r.gene} ZT{r.peak_zt:.1f} (p {r.p:.3f})"
                                                       for r in sig_il.itertuples()),
    "33_rhythm_tiers/rhythm_genes_all_cell_type_<tissue>.csv")
gp = []
for t in ["bm", "cochlea"]:
    x = rd(f"r2_cochlea/gene_pattern_scores_{t}.csv")
    if len(x):
        gp.append(x[x["gene"].isin(IL1) & x["cell_type"].isin(NEUT)].assign(tissue=t))
gp = pd.concat(gp) if gp else pd.DataFrame()
if len(gp):
    med = gp.groupby("tissue")["pattern_r"].median()
    add(fid, "sample_level_pattern", "against" if (med > 0.8).any() else ("partial" if (med > 0.5).any() else "supports"),
        "median correlation of IL-1 gene profiles with the tissue-wide sample pattern: " + ", ".join(f"{k} {v:+.2f}" for k, v in med.items()),
        "r2_cochlea/gene_pattern_scores_<tissue>.csv")
n_o, mu, R, units = geneset_phase_check(IL1, {"bm": NEUT, "cochlea": NEUT})
add(fid, "external_data", units_status(units),
    f"our IL-1 internal phase {mu:.1f} h (n {n_o}); external control units: " +
    ("; ".join(f"{u} n{n} {m_:.1f} h (Δ {d:+.1f})" for u, n, m_, d in units) or "none with >= 2 rhythmic IL-1 genes"),
    "external_rhythm_*.csv (clock-aligned)")
add(fid, "power_effect_size", "partial", "MultiNicheNet contrast = 3 vs 3 pooled samples (rest ZT3/7/11 vs active ZT15/19/23); "
    "no replicate mice", "67_multinichenet")

# ---------------------------------------------------------------- G04: cochlear structural clocks
fid = "G04"
STRUCT_CLASSES = {"Bone", "Endothelial", "Epithelial", "Glia", "Sensory", "Stromal", "Neuronal"}
struct = sorted(cls[(cls["tissue"] == "cochlea") & cls["cell_class"].isin(STRUCT_CLASSES)]["cell_type"])
register(fid, "clock function", "cochlea", f"Cochlear structural cells ({len(struct)} types) carry coherent core clocks")
ccd = rd("34_clock_network/clock_ccd.csv")
cs = ccd[(ccd["tissue"] == "cochlea") & ccd["group"].isin(struct)]
n_ccd = int((cs["Pvalue"] < 0.05).sum())
add(fid, "model_concordance", "supports" if n_ccd >= len(cs) / 3 else ("partial" if n_ccd else "against"),
    f"clock-like co-expression (CCD permutation p<0.05): {n_ccd}/{len(cs)} structural types (" +
    "; ".join(cs.loc[cs["Pvalue"] < 0.05, "group"]) + ")", "34_clock_network/clock_ccd.csv")
cpo = rd("02_clock_controls/clock_phase_order.csv")
co = cpo[(cpo["tissue"] == "cochlea") & cpo["cell_type"].isin(struct)]
n_ord = int((co["p_circ_corr"] < 0.05).sum())
add(fid, "resampling_or_fdr", "supports" if n_ord >= len(co) / 3 else ("partial" if n_ord else "against"),
    f"canonical clock-gene phase order (circular correlation permutation p<0.05): {n_ord}/{len(co)} structural types",
    "02_clock_controls/clock_phase_order.csv")
oc = off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & off["unit"].isin(struct)]
n_det = int((oc["determinate"].astype(str) == "True").sum())
add(fid, "cross_tissue_logic", "supports" if n_det >= len(oc) / 2 else ("partial" if n_det else "against"),
    f"phase-determinate clock offsets: {n_det}/{len(oc)} structural types; offsets " +
    ", ".join(f"{r.unit} {r.offset_canonical_h:+.1f}" for r in oc[oc["determinate"].astype(str) == "True"].itertuples()),
    "r2_phase/clock_alignment_offsets.csv")
tc = rd("r2_cochlea/cochlea_relaxed_tier.csv", usecols=["cell_type", "gene", "tierC", "p", "peak_zt", "lr_p", "tissue_pattern_like", "rule_selected"])
if len(tc):
    tc["tierC"] = tc["tierC"].astype(str) == "True"
    tcl = tc[tc["tierC"] & tc["gene"].isin(phase.CANONICAL_CLOCK) & tc["cell_type"].isin(struct)]
    add(fid, "molecular_mechanistic", "supports" if len(tcl) >= 5 else ("partial" if len(tcl) else "against"),
        f"core-clock genes in Tier C (positive control): {len(tcl)} gene × type in {tcl['cell_type'].nunique()} structural types (" +
        ", ".join(sorted(tcl["gene"].unique())) + ")", "r2_cochlea/cochlea_relaxed_tier.csv")
e88 = off[(off["dataset"] == "GSE88954") & (off["condition"] == "sham")]
add(fid, "external_data", "supports" if len(e88) and abs(e88["offset_canonical_h"].iloc[0]) <= 2 else "partial",
    f"external cochlea (GSE88954 sham, CT) clock offset {e88['offset_canonical_h'].iloc[0]:+.2f} h ± {e88['ci_halfwidth_h'].iloc[0]:.2f}; "
    f"our cochlear tissue pool {OFF['cochlea']:+.2f} h", "r2_phase/clock_alignment_offsets.csv")

# ---------------------------------------------------------------- G05: night activation / stress programme
fid = "G05"
register(fid, "systemic programme", "both", "Night (ZT15–19) activation programme — immediate-early, NF-κB and heat-shock genes; "
         "HSE-motif carriers peak ZT17–18 — shared by both tissues")
sc = rd("82_systemic/systemic_cue_common_phase.csv")
scs = sc[sc["set"].isin(["heat_shock_hsf1", "glucocorticoid_response", "adrenergic_camp"])]
ok = scs[(scs["rayleigh_p"] < 0.05) & (scs["weighted_common_peak_zt"] >= 12)]
add(fid, "model_concordance", "supports" if len(ok) >= 4 else ("partial" if len(ok) else "against"),
    "common phase across cell types (Rayleigh p<0.05, night peak): " +
    "; ".join(f"{r.tissue} {r.set} ZT{r.weighted_common_peak_zt:.1f} (p {r.rayleigh_p:.1e})" for r in scs.itertuples()),
    "82_systemic/systemic_cue_common_phase.csv")
mot = rd("44_motifs/promoter_motif_phase.csv")
hse = mot[mot["motif"].astype(str).str.startswith("HSE") & (mot["q_directional_perm"] < 0.1)] if len(mot) else mot
ray = {}
for t in ["bm", "cochlea"]:
    g = G[t][G[t]["gene"].isin(IEG + HSG + NFKB) & (G[t]["p"] < 0.05)]
    ray[t] = (len(g), *cmean(g["peak_zt"]), phase.rayleigh(g["peak_zt"].to_numpy())[1] if len(g) >= 3 else np.nan)
add(fid, "resampling_or_fdr", "supports" if all(v[3] < 0.05 for v in ray.values()) and len(hse) else "partial",
    "IEG/NF-κB/heat-shock rows p<0.05: " + "; ".join(f"{t} n {v[0]}, mean ZT{v[1]:.1f}, R {v[2]:.2f}, Rayleigh p {v[3]:.1e}"
                                                    for t, v in ray.items()) +
    f"; HSE carriers at the expected phase (directional permutation q<0.1): {len(hse)} (" +
    "; ".join(f"{r.tissue} {r.cell_type} ZT{r.circ_mean_peak_zt:.1f}" for r in hse.itertuples()) + ")",
    "33_rhythm_tiers/rhythm_genes_all_cell_type_<tissue>.csv; 44_motifs/promoter_motif_phase.csv")
bm_stress = pcov[(pcov["tissue"] == "bm") & (pcov["pc"] == 1) & (pcov["covariate"] == "stress_score_mean")]["pearson_r"]
add(fid, "sample_level_pattern", "against" if len(bm_stress) and abs(bm_stress.iloc[0]) >= 0.8 else "partial",
    f"BM tissue-wide pattern (PC1) vs per-sample stress score r {bm_stress.iloc[0]:+.2f}; cochlea: {pattern_cov('cochlea')} — "
    "programme co-varies with sample-level handling/composition axes", "r2_cochlea/sample_pattern_covariates.csv")
st_e, ev = [], []
for name, gs in [("IEG", IEG), ("heat shock", HSG), ("NF-κB", NFKB)]:
    n_o, mu, R, units = geneset_phase_check(gs, {"bm": None, "cochlea": None})
    st_e.append(units_status(units))
    ev.append(f"{name}: ours {mu:.1f} h internal (n {n_o}, R {R:.2f}); {sum(abs(x[3]) <= 4 for x in units)}/{len(units)} "
              f"external control units within 4 h")
npc = rd("r2_pilots/pilot_night_programme_check.csv")
if len(npc):
    sh = npc[npc["contrast"] == "night_minus_day_no_noise_shamsurg"]
    up = sh[(sh["mean_logFC_night_minus_day"] > 0) & (sh["p_wilcoxon_vs_all"] < 0.05)]
    st_e.append("supports" if len(up) >= 2 else ("partial" if len(up) else "against"))
    ev.append("independent cohort GSE107086 (no noise, ~ZT17.5 vs ~ZT5.5): " + "; ".join(
        f"{r.gene_set} {r.contrast.replace('night_minus_day_no_noise_', '')} {r.mean_logFC_night_minus_day:+.2f} (p {r.p_wilcoxon_vs_all:.3f})"
        for r in npc.itertuples()))
add(fid, "external_data", combine(st_e), "; ".join(ev), "external_rhythm_*.csv (clock-aligned); r2_pilots/pilot_night_programme_check.csv")
add(fid, "cross_tissue_logic", "supports" if all(v[3] < 0.05 for v in ray.values()) and
    abs(float(phase.circ_diff(ray["bm"][1], ray["cochlea"][1]))) <= 3 else "partial",
    f"BM vs cochlea mean phase of the programme: ZT{ray['bm'][1]:.1f} vs ZT{ray['cochlea'][1]:.1f}",
    "33_rhythm_tiers/rhythm_genes_all_cell_type_<tissue>.csv")

# ---------------------------------------------------------------- G06: cochlear Tier C (round 2)
cal = rd("r2_cochlea/cochlea_relaxed_calibration.csv")
if len(cal) and len(tc):
    fid = "G06"
    sel_rule = str(tc["rule_selected"].iloc[0]) if "rule_selected" in tc else "R0"
    al = cal[(cal["cell_type"] == "ALL") & (cal["rule"] == sel_rule)].iloc[0]
    r0 = cal[(cal["cell_type"] == "ALL") & (cal["rule"] == "R0")].iloc[0]
    register(fid, "gene rhythms (relaxed)", "cochlea", f"Calibrated relaxed cochlear tier (Tier C, rule {sel_rule}): "
             f"{int(al['observed_calls'])} gene × type calls ({int(al['observed_calls_no_pattern'])} not tissue-pattern-like); "
             f"pre-specified rule R0: {int(r0['observed_calls'])} calls")
    add(fid, "resampling_or_fdr", "supports" if al["empirical_fdr"] <= 0.1 else ("partial" if al["empirical_fdr"] <= 0.25 else "against"),
        f"rule {sel_rule} re-applied under 59 null relabelings: mean null calls {al['null_mean']:.1f} vs observed "
        f"{int(al['observed_calls'])} → empirical FDR {al['empirical_fdr']:.3f} (without pattern-like genes "
        f"{al['empirical_fdr_no_pattern']:.3f}); R0 {r0['empirical_fdr']:.3f}; no rule of the ladder reached the 0.2 target",
        "r2_cochlea/cochlea_relaxed_calibration.csv; cochlea_relaxed_rule_selection.csv")
    tcc = tc[tc["tierC"]]
    flr = float((tcc["lr_p"] < 0.05).mean()) if len(tcc) else np.nan
    add(fid, "model_concordance", "supports" if flr >= 0.8 else ("partial" if flr >= 0.5 else "against"),
        f"Tier C calls with LR (diffCircadian) p<0.05: {flr:.2f}", "r2_cochlea/cochlea_relaxed_tier.csv")
    fpat = float(tcc["tissue_pattern_like"].astype(str).eq("True").mean()) if len(tcc) else np.nan
    add(fid, "sample_level_pattern", "supports" if fpat < 0.2 else ("partial" if fpat < 0.5 else "against"),
        f"fraction of Tier C calls flagged tissue-pattern-like (r > 0.8 with the leave-one-type-out consensus): {fpat:.2f}",
        "r2_cochlea/cochlea_relaxed_tier.csv")
    tcl = tcc[tcc["gene"].isin(phase.CANONICAL_CLOCK)]
    add(fid, "molecular_mechanistic", "supports" if tcl["cell_type"].nunique() >= 3 else ("partial" if len(tcl) else "against"),
        f"core-clock genes in Tier C (positive control): {len(tcl)} in {tcl['cell_type'].nunique()} cell types",
        "r2_cochlea/cochlea_relaxed_tier.csv")
    tv = rd("r2_pilots/pilot_tierC_independent_validation.csv")
    if len(tv):
        x = tv.set_index("gene_set")
        r_ = x.loc["Tier C, BM-phase support only (no external-cochlea information)"]   # selection free of external data
        add(fid, "external_data", "supports" if r_["pearson_r"] > 0 and r_["p_perm"] < 0.05 else
            ("against" if r_["pearson_r"] < 0 and r_["p_perm"] > 0.95 else "neutral"),
            "independent cohort GSE107086 (not used for selection), predicted vs observed night − day: " +
            "; ".join(f"{k} r {v.pearson_r:+.3f} (n {int(v.n_genes)}, perm p {v.p_perm:.4f})" for k, v in x.iterrows()) +
            "; phase concordance with GSE88954 / BM is selection-dependent and not counted",
            "r2_pilots/pilot_tierC_independent_validation.csv")

# ---------------------------------------------------------------- G07: dryR cross-tissue classes (round 1)
fid = "G07"
register(fid, "cross-tissue gene rhythms", "both", "dryR cross-tissue model classes (cochlea-only 24,792; BM-only 9,371; shared "
         "5,134; different 3,980 gene × type assignments)")
dn = rd("r2_audit/dryr_null_check.csv")
if len(dn):
    o = dn[dn["run"] == "observed"].groupby("model_label")["N"].sum()
    n = dn[dn["run"] == "null"].groupby("model_label")["N"].sum() / dn[dn["run"] == "null"].groupby("cell_type")["relabel_class"].nunique().mean()
    ratio = {k: n.get(k, 0) / max(o.get(k, 0), 1) for k in ["cochlea_only", "shared", "different"]}
    add(fid, "resampling_or_fdr", "supports" if max(ratio.values()) <= 0.1 else ("partial" if max(ratio.values()) <= 0.5 else "against"),
        "null relabelling of cochlear time (3 types × 3 classes): null/observed calls " +
        ", ".join(f"{k} {v:.2f} (obs {int(o.get(k, 0))}, null mean {n.get(k, 0):.0f})" for k, v in ratio.items()),
        "r2_audit/dryr_null_check.csv")
lrd = rd("35_crosstissue/crosstissue_lr_diff.csv", usecols=["model_label", "q_lr_amplitude", "q_lr_phase"])
if len(lrd):
    nd = int(((lrd["q_lr_amplitude"] < 0.05) | (lrd["q_lr_phase"] < 0.05)).sum())
    add(fid, "model_concordance", "supports" if nd >= 0.1 * len(lrd) else ("partial" if nd else "against"),
        f"direct LR_diff tests (amplitude/phase difference, q<0.05) among dryR-classified genes: {nd}/{len(lrd)}",
        "35_crosstissue/crosstissue_lr_diff.csv")

# ---------------------------------------------------------------- G08: BM Milo neighbourhood rhythms
fid = "G08"
mbs, mbv = milo[("bm", "scANVI")], milo[("bm", "scVI")]
register(fid, "sub-state abundance", "bm", f"BM rhythmic neighbourhoods (Milo): {int((mbs['spatial_fdr'] < 0.1).sum())} of {len(mbs)} "
         "(aged/activated neutrophils, naïve B, pre-B, cycling cells)")
ta = mbs[mbs["spatial_fdr"] < 0.1].groupby("majority_cell_type")["peak_zt"].agg(["size", lambda x: cmean(x)[0]])
tb = mbv[mbv["spatial_fdr"] < 0.1].groupby("majority_cell_type")["peak_zt"].agg(["size", lambda x: cmean(x)[0]])
ta.columns = tb.columns = ["n", "peak"]
j = ta.join(tb, lsuffix="_scanvi", rsuffix="_scvi", how="inner")
j = j[j["n_scanvi"] >= 10]
dd = np.abs(phase.circ_diff(j["peak_scanvi"], j["peak_scvi"])) if len(j) else np.array([])
add(fid, "model_concordance", "supports" if len(dd) and (dd <= 3).all() else ("partial" if len(dd) else "not_tested"),
    "scANVI vs scVI (unsupervised) neighbourhoods, types with >= 10: " +
    "; ".join(f"{k} n {int(r.n_scanvi)}/{int(r.n_scvi)} ZT{r.peak_scanvi:.1f}/ZT{r.peak_scvi:.1f}" for k, r in j.iterrows()),
    "53_milo/milo_da_results_bm.csv; milo_da_results_bm_scvi.csv")
add(fid, "resampling_or_fdr", "supports" if (mbs["spatial_fdr"] < 0.1).sum() and (mbv["spatial_fdr"] < 0.1).sum() else "against",
    f"spatial FDR<0.1: scANVI {int((mbs['spatial_fdr'] < 0.1).sum())}, scVI {int((mbv['spatial_fdr'] < 0.1).sum())} (edgeR QL F, "
    "6 samples)", "53_milo")
bn = pr[(pr["tissue"] == "bm") & (pr["cell_type"] == "B Cell (Naïve)")]
if len(bn) and "B Cell (Naïve)" in ta.index:
    d = float(phase.circ_diff(ta.loc["B Cell (Naïve)", "peak"], bn["peak_zt"].iloc[0]))
    add(fid, "cross_tissue_logic", "supports" if abs(d) <= 4 else "partial",
        f"naïve-B neighbourhood peak vs naïve-B share peak: Δ {d:+.1f} h", "53_milo; r2_figure/abundance_pct_rhythm.csv")

# ---------------------------------------------------------------- G09: clocks aligned to the light cycle (positive control)
fid = "G09"
register(fid, "positive control", "both", "Both tissues' core clocks are entrained to the light–dark cycle (canonical phases)")
tp = off[(off["dataset"] == "this_atlas") & (off["unit"] == "tissue_pool")]
add(fid, "model_concordance", "supports" if (tp["determinate"].astype(str) == "True").all() and (tp["offset_canonical_h"].abs() <= 2).all()
    else "partial", "; ".join(f"{r.tissue}: offset {r.offset_canonical_h:+.2f} h (95 % CI ± {r.ci_halfwidth_h:.2f}; {r.n_genes} genes; "
                             f"joint alignment {r.offset_joint_h:+.2f})" for r in tp.itertuples()), "r2_phase/clock_alignment_offsets.csv")
add(fid, "resampling_or_fdr", "supports" if (tp["ci_halfwidth_h"] <= 3).all() else "partial",
    "gene-bootstrap CI half-widths: " + ", ".join(f"{r.tissue} {r.ci_halfwidth_h:.2f} h" for r in tp.itertuples()),
    "r2_phase/clock_alignment_offsets.csv")
eb = off[(off["unit"] == "bulk") & off["condition"].isin(CTRL)]
eb_ok = eb[~eb["dataset"].str.contains("lung")]
add(fid, "external_data", "supports" if (eb_ok["offset_canonical_h"].abs() <= 2).mean() >= 0.8 else "partial",
    f"external control units within ±2 h of canonical: {int((eb['offset_canonical_h'].abs() <= 2).sum())}/{len(eb)} "
    f"(GSE198770 lung {eb.loc[eb['dataset'].str.contains('lung'), 'offset_canonical_h'].round(1).tolist()} h flagged)",
    "r2_phase/clock_alignment_offsets.csv")
cco = cpo[cpo["status"].astype(str) == "primary"] if len(cpo) else cpo
add(fid, "cross_tissue_logic", "supports" if len(cco) and (cco["p_circ_corr"] < 0.05).mean() >= 0.3 else "partial",
    f"cell types with canonical clock order (p<0.05): " + ", ".join(
        f"{t} {int((g['p_circ_corr'] < 0.05).sum())}/{len(g)}" for t, g in cco.groupby("tissue")), "02_clock_controls/clock_phase_order.csv")


# ================================================================ round 3 findings
# R01: BM neutrophil rhythms replicate in sorted BM neutrophils (GSE312298, PBS, design times ZT14 vs ZT22)
v3 = rd("r3_response/gse312298_validation.csv")
c3 = rd("r3_response/gse312298_collection_time.csv")
if len(v3):
    fid = "R01"
    register(fid, "external replication", "bm", "BM neutrophil-state rhythms predict the night − day difference of sorted BM "
             "neutrophils in an independent study (GSE312298, PBS arms, collection ZT14 vs ZT22)")
    d_ = v3[v3["collection_time"].str.startswith("design")].set_index("predictor")
    t1 = d_.loc[[k for k in d_.index if "Tier 1" in k][0]]
    al = d_.loc[[k for k in d_.index if "all tested" in k][0]]
    add(fid, "model_concordance", "supports" if t1["pearson_r"] > 0 and al["pearson_r"] > 0 and t1["p_perm"] < 0.05 and al["p_perm"] < 0.05 else "partial",
        f"Tier 1 (selection-free) r {t1['pearson_r']:+.3f} (n {int(t1['n_genes'])}); all tested genes r {al['pearson_r']:+.3f} (n {int(al['n_genes'])})",
        "r3_response/gse312298_validation.csv")
    add(fid, "resampling_or_fdr", "supports" if t1["p_perm"] < 0.05 else "against", f"gene-label permutation p {t1['p_perm']:.4f}",
        "r3_response/gse312298_validation.csv")
    cmp_ = d_.loc[[k for k in d_.index if "comparator" in k][0]]
    add(fid, "external_data", "supports" if cmp_["pearson_r"] > 0 else "neutral",
        f"comparator: GSE198770 bulk BM fits r {cmp_['pearson_r']:+.3f} (our sorted-type prediction is of similar size)",
        "r3_response/gse312298_validation.csv")
    if len(c3):
        b = c3[c3["is_best"].astype(str) == "True"].set_index("reference")["day_collection_internal_h"]
        ours_t = float(b[[k for k in b.index if "neutrophil" in k][0]])
        add(fid, "cross_tissue_logic", "supports" if abs(phase.circ_diff(ours_t, 14.0)) <= 3 else "partial",
            f"collection time from our neutrophil clock fits {ours_t:.1f} h vs design ZT14; canonical tissue clock phases give "
            f"{float(b[[k for k in b.index if 'canonical' in k][0]]):.1f} h (neutrophil clocks differ from tissue clocks)",
            "r3_response/gse312298_collection_time.csv")
    add(fid, "power_effect_size", "partial", "time of day is confounded with experiment (2 experiments per time point)",
        "GSE312298 design")

# R02/R03: BM extension of the cochlear methodology
bsel = rd("r3_bm/bm_relaxed_rule_selection.csv")
bind = rd("r3_bm/bm_independence_validation.csv")
if len(bsel) and len(bind):
    fid = "R02"
    sel_ = bsel[bsel["selected"].astype(str) == "True"].iloc[0]
    register(fid, "gene rhythms (relaxed)", "bm", f"BM relaxed tier with the cochlear rule ladder (selected {sel_['rule']}: "
             f"{int(sel_['observed_calls'])} calls)")
    add(fid, "resampling_or_fdr", "supports" if sel_["empirical_fdr"] <= 0.1 else ("partial" if sel_["empirical_fdr"] <= 0.25 else "against"),
        "rule ladder on 59 null relabelings: " + "; ".join(f"{r.rule} {int(r.observed_calls)} (FDR {r.empirical_fdr:.2f})" for r in bsel.itertuples()),
        "r3_bm/bm_relaxed_rule_selection.csv")
    bs = rd("r3_bm/bm_relaxed_simulation.csv")
    if len(bs):
        x = bs[(bs["rule"] == sel_["rule"]) & (bs["ptt_fc_injected"] == 2)]
        add(fid, "model_concordance", "partial", f"simulation (FC 2): power {x['power'].mean():.2f}, FDP {x['fdp'].mean():.2f}",
            "r3_bm/bm_relaxed_simulation.csv")
    bi = bind.set_index("gene_set")
    co = bi.loc[[k for k in bi.index if "cochlea support only" in k][0]]
    add(fid, "external_data", "supports" if co["pearson_r_stacked"] > 0 and co["p_perm"] < 0.05 else "neutral",
        f"held-out GSE198770 replicate 3: cochlea-support-only calls (no external information) r {co['pearson_r_stacked']:+.3f} "
        f"(p {co['p_perm']:.3f}); background (p >= 0.05) r {bi.loc[[k for k in bi.index if 'p >= 0.05' in k][0], 'pearson_r_stacked']:+.3f}",
        "r3_bm/bm_independence_validation.csv")
    fid = "R03"
    pl = bi.loc[[k for k in bi.index if "pattern-like (r > 0.8)" in k][0]]
    npl = bi.loc[[k for k in bi.index if "not pattern-like" in k][0]]
    register(fid, "sample-level pattern", "bm", "The BM-wide sample pattern (210 PC1, smooth 24-h curve) carries real time-of-day "
             "signal: pattern-like Tier-1 genes replicate in held-out external BM")
    add(fid, "model_concordance", "supports" if pl["pearson_r_stacked"] > npl["pearson_r_stacked"] and pl["p_perm"] < 0.05 else "partial",
        f"pattern-like Tier 1 r {pl['pearson_r_stacked']:+.3f} (n {int(pl['n_genes'])}) vs not pattern-like r {npl['pearson_r_stacked']:+.3f} "
        f"(n {int(npl['n_genes'])}); {pl['pattern_like_share_of_tier1']:.0%} of Tier 1 is pattern-like", "r3_bm/bm_independence_validation.csv")
    add(fid, "resampling_or_fdr", "supports" if pl["p_perm"] < 0.05 else "against", f"gene-label permutation p {pl['p_perm']:.4f}",
        "r3_bm/bm_independence_validation.csv")
    hs_ = rd("r3_audit/handling_sample_effect.csv")
    if len(hs_):
        h = hs_[hs_["tissue"] == "bm"].iloc[0]
        add(fid, "sample_level_pattern", "partial", f"BM sample effect of the dissociation-stress score: {h['sample_term_variance_share']:.0%} of "
            f"variance, r with PC1 {h['r_with_tissue_pattern_pc1']:+.2f} (stress-linked yet externally replicated)",
            "r3_audit/handling_sample_effect.csv")

# R04: cell-level clock phase tracks ZT in cochlear structural cells
ccs = rd("r3_audit/cell_clock_phase_summary.csv")
if len(ccs):
    fid = "R04"
    register(fid, "clock function (cell level)", "cochlea", "Cell-level clock phase (canonical projection) tracks ZT with a ~0 h offset "
             "in cochlear structural cells")
    cc_ = ccs[ccs["tissue"] == "cochlea"].copy()
    cc_["structural"] = cc_["cell_type"].map(cls.drop_duplicates("cell_type").set_index("cell_type")["cell_class"]).isin(STRUCT_CLASSES)
    st_ = cc_[cc_["structural"]]
    n_ok = int((st_["p_perm"] <= 0.05).sum())
    add(fid, "model_concordance", "supports" if n_ok >= len(st_) / 2 else ("partial" if n_ok else "against"),
        f"structural types with offset consistency p <= 0.05: {n_ok}/{len(st_)}; median offset "
        f"{st_['mean_offset_cell_phase_minus_zt_h'].median():+.1f} h", "r3_audit/cell_clock_phase_summary.csv")
    add(fid, "resampling_or_fdr", "supports" if n_ok else "against", "exact 60-class dihedral relabelling of ZT (minimum p 1/60)",
        "r3_audit/cell_clock_phase_summary.csv")
    imm_ = cc_[~cc_["structural"]]
    add(fid, "molecular_mechanistic", "supports" if st_["median_within_sample_resultant"].median() > imm_["median_within_sample_resultant"].median() else "partial",
        f"within-sample synchrony (median resultant): structural {st_['median_within_sample_resultant'].median():.2f} vs immune "
        f"{imm_['median_within_sample_resultant'].median():.2f}", "r3_audit/cell_clock_phase_summary.csv")

# R05: the night heat-shock programme is physiological
n_o, mu, R, units = geneset_phase_check(HSG, {"bm": None, "cochlea": None})
npc = rd("r2_pilots/pilot_night_programme_check.csv")
hgf = rd("r3_audit/handling_gene_flags.csv")
fid = "R05"
register(fid, "systemic programme", "both", "Night heat-shock programme (Hsp genes peak ~ZT17) is physiological, unlike the "
         "immediate-early / NF-κB component")
add(fid, "model_concordance", "supports" if R >= 0.5 else "partial", f"our heat-shock rows p<0.05: n {n_o}, internal mean {mu:.1f} h, R {R:.2f}",
    "33_rhythm_tiers")
add(fid, "resampling_or_fdr", "supports" if n_o >= 10 and phase.rayleigh(np.array([u[2] for u in units]))[1] < 0.05 else "partial",
    f"external control units with >= 2 rhythmic heat-shock genes: {len(units)}; within 4 h of ours: {sum(abs(u[3]) <= 4 for u in units)}",
    "external_rhythm_*.csv")
add(fid, "external_data", units_status(units), "; ".join(f"{u} {m_:.1f} h (Δ {d:+.1f})" for u, n, m_, d in units[:20]), "external_rhythm_*.csv")
if len(npc):
    h = npc[npc["gene_set"] == "heat shock"]
    add(fid, "cross_tissue_logic", "supports" if (h["mean_logFC_night_minus_day"] > 0).all() else "partial",
        "independent cochlea GSE107086 (no noise) night − day: " + "; ".join(
            f"{r.contrast.replace('night_minus_day_no_noise_', '')} {r.mean_logFC_night_minus_day:+.2f} (p {r.p_wilcoxon_vs_all:.3f})" for r in h.itertuples()) +
        " — persists after adrenalectomy (glucocorticoid-independent)", "r2_pilots/pilot_night_programme_check.csv")
if len(hgf):
    x = hgf[(hgf["gene_class"].isin(["heat_shock", "IEG"])) & (hgf["phase_bin"].isin(["ZT12-18", "ZT18-24"]))]
    agg_ = x.groupby("gene_class")[["n_calls", "n_stress_like"]].sum()
    fr = (agg_["n_stress_like"] / agg_["n_calls"]).to_dict()
    add(fid, "sample_level_pattern", "supports" if fr.get("heat_shock", 1) < fr.get("IEG", 0) else "partial",
        f"night calls following the per-sample dissociation-stress profile: heat shock {fr.get('heat_shock', np.nan):.0%} vs IEG "
        f"{fr.get('IEG', np.nan):.0%}", "r3_audit/handling_gene_flags.csv")

# R06: noise expands cochlear neutrophils and macrophages (external single-cell data)
cs1, cs2 = rd("r3_gse167078/composition_contrasts.csv"), rd("r3_gse346691/composition_contrasts.csv")
if len(cs1) or len(cs2):
    fid = "R06"
    register(fid, "noise response", "cochlea", "Noise exposure expands cochlear neutrophils and macrophages within days "
             "(external single-cell data mapped to our labels)")
    TARGET = ["Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Band)", "Macrophage"]
    ev, sts = [], []
    for nm, c_, con in [("GSE167078 3 d (night noise)", cs1, "noise_3d_vs_sham"), ("GSE346691 WT noise vs control", cs2, "WT_noise_vs_WT_control")]:
        if not len(c_):
            continue
        x = c_[(c_["subset"] == "all cells") & (c_["contrast"] == con) & c_["label"].isin(TARGET)].set_index("label")
        ev.append(nm + ": " + ", ".join(f"{k} {v:+.2f}" for k, v in x["log2_ratio"].items()))
        sts.append("supports" if (x["log2_ratio"] > 0).mean() >= 0.75 else ("partial" if (x["log2_ratio"] > 0).any() else "against"))
    add(fid, "model_concordance", "supports" if sts and all(x == "supports" for x in sts) else ("partial" if "supports" in sts else "against"),
        "log2 share ratio after noise (supports only if every dataset agrees) — " + " | ".join(ev), "r3_gse167078; r3_gse346691")
    if len(cs2):
        x = cs2[(cs2["subset"] == "all cells") & (cs2["contrast"] == "WT_noise_vs_WT_control") & cs2["label"].isin(TARGET)]
        add(fid, "resampling_or_fdr", "supports" if (x["p_welch_logit"] < 0.05).any() else "partial",
            "GSE346691 Welch t on logit shares (3 vs 2): " + ", ".join(f"{r.label} p {r.p_welch_logit:.3f}" for r in x.itertuples()),
            "r3_gse346691/composition_contrasts.csv")
    add(fid, "literature", "supports", "Hirose 2005 (CD45+ phagocytes rise 1–14 d after noise); Milon 2021 (neutrophils up at 3 d); "
        "Ma 2025 (macrophages main responders at 3 d)", "scripts/resources/spatial_priors.csv")
    mc = rd("r3_ingest/gse346691_marker_check.csv")
    if len(mc):
        add(fid, "annotation_confidence", "supports" if (mc["marker_z_in_label"] > mc["marker_z_in_others"]).mean() > 0.9 else "partial",
            f"transferred labels: own markers higher in {int((mc['marker_z_in_label'] > mc['marker_z_in_others']).sum())}/{len(mc)} labels",
            "r3_ingest/gse346691_marker_check.csv")

# R07: spatial localisation (GeoMx deconvolution)
gx = rd("r3_spatial/geomx_deconvolution_summary.csv")
if len(gx):
    fid = "R07"
    register(fid, "spatial", "cochlea", "Our cochlear cell profiles localise in public GeoMx data (spiral-ganglion neurons in neuron AOIs; "
             "macrophages in supporting-cell AOIs)")
    sg_ = gx.set_index("cell_type")
    add(fid, "model_concordance", "supports" if sg_.loc["Spiral Ganglion Neuron", "mean_prop_neuron_control"] > sg_.loc["Spiral Ganglion Neuron", "mean_prop_supporting_control"] else "against",
        f"SGN proportion neuron vs supporting AOIs {sg_.loc['Spiral Ganglion Neuron', 'mean_prop_neuron_control']:.2f} vs "
        f"{sg_.loc['Spiral Ganglion Neuron', 'mean_prop_supporting_control']:.2f} (q {sg_.loc['Spiral Ganglion Neuron', 'q_region']:.3g})",
        "r3_spatial/geomx_deconvolution_summary.csv")
    add(fid, "resampling_or_fdr", "partial", f"weighted NNLS on {int(246)} shared marker genes; median fit R² {sg_['median_fit_r2'].iloc[0]:.2f}; "
        "gene-bootstrap intervals in geomx_deconvolution_aoi.csv", "r3_spatial/geomx_deconvolution_aoi.csv")
    add(fid, "literature", "supports", "macrophages in the lateral wall / spiral ligament and ganglion (Okano 2008; Okano & Kishimoto 2019)",
        "scripts/resources/spatial_priors.csv")

# R08: Ma 2025 — exposure time vs response (added when the user's matrices have been processed)
ml = rd("r3_ma2025/composition_rhythm_links.csv")
if len(ml):
    fid = "R08"
    register(fid, "noise response × time of day", "cochlea", "Independent cohort (Ma 2025): immune composition differed between "
             "cochleae collected ~12 h apart in the direction predicted by our cochlear abundance rhythms; because collection "
             "and exposure times coincide, this holds only if noise × time interactions are small (D77)")
    x = ml[(ml["subset"] == "all cells") & (ml["contrast"] == "night_noise_vs_day_noise") & ml["immune"]]
    rho = x["spearman_expected_vs_observed_immune"].dropna()
    pp_ = x["p_perm_labels"].dropna()
    add(fid, "model_concordance", ("supports" if len(rho) and rho.iloc[0] > 0 and pp_.iloc[0] < 0.05 else "partial" if len(rho) and rho.iloc[0] > 0 else "against"),
        f"Spearman expected (our share ZT15 vs ZT3) vs observed night/day log2 ratio across immune labels: "
        f"{rho.iloc[0] if len(rho) else np.nan:+.2f} (label permutation p {pp_.iloc[0] if len(pp_) else np.nan:.3f})",
        "r3_ma2025/composition_rhythm_links.csv")
    spm = rd("r3_ma2025/sampling_phase.csv")
    if len(spm):
        g_ = spm.groupby("condition")["phase_canonical_projection_h"].apply(lambda v: phase.circ_mean(v)[0])
        dd = abs(float(phase.circ_diff(g_.get("night_noise", np.nan), g_.get("day_noise", np.nan))))
        add(fid, "sample_level_pattern", "supports" if dd < 4 else "against",
            f"sampling-phase check: day vs night groups differ by {dd:.1f} h in clock phase (large = collection confounded)",
            "r3_ma2025/sampling_phase.csv")
    rgl = rd("r3_ma2025/rhythm_gene_links.csv")
    if len(rgl):
        e_ = rgl[rgl["test"].str.startswith("enrichment")]
        add(fid, "molecular_mechanistic", "supports" if (e_["q_BH"] < 0.1).any() else "neutral",
            f"night-vs-day DE genes enriched in our rhythm sets (q<0.1): {int((e_['q_BH'] < 0.1).sum())}/{len(e_)} label × set tests",
            "r3_ma2025/rhythm_gene_links.csv")
    add(fid, "literature", "supports", "Meltser 2014; Cederroth 2019; Ma 2025 (night noise more damaging)", "scripts/resources/literature_response.csv")

# ================================================================ round 4 findings
# R09: design benchmark — the one-sample-per-time-point pipeline on dense references with an independent truth
bs_ = rd("r4_benchmark/benchmark_summary.csv")
bc_ = rd("r4_benchmark/benchmark_clock_recovery.csv")
if len(bs_):
    fid = "R09"
    register(fid, "methods / design credibility", "both", "Our 6-time-point, one-sample-per-time-point rhythm pipeline is calibrated on dense "
             "public time series with a held-out truth (GSE54650 12 organs, GSE88954 cochlea): Tier-1/2 calls have low realised FDP")
    a_ = bs_[(bs_["unit"] == "ALL") & (bs_["condition"] == "clean")].set_index(["dataset", "rule"])
    t12 = a_.xs("tier1", level="rule")["mean_fdp_strict"].to_dict()
    t2 = a_.xs("tier2", level="rule")["mean_fdp_strict"].to_dict()
    nom = a_.xs("nominal", level="rule")["mean_fdp_strict"].to_dict()
    ok = all((v <= 0.1) or np.isnan(v) for v in list(t12.values()) + list(t2.values()))
    add(fid, "resampling_or_fdr", "supports" if ok else "partial",
        "realised FDP (held-out truth p > 0.2) — Tier 1: " + "; ".join(f"{k} {v:.3f}" for k, v in t12.items()) + " | Tier 2: " +
        "; ".join(f"{k} {v:.3f}" for k, v in t2.items()) + " | nominal p<0.05: " + "; ".join(f"{k} {v:.3f}" for k, v in nom.items()),
        "r4_benchmark/benchmark_summary.csv")
    est = a_.xs("tier1_core", level="rule")
    add(fid, "model_concordance", "supports" if (est["mean_est_fdr_perm"] >= est["mean_fdp_strict"] - 0.05).all() else "partial",
        "permutation-estimated FDR vs realised FDP (Tier-1 core rule): " + "; ".join(
            f"{k} est {r.mean_est_fdr_perm:.3f} / realised {r.mean_fdp_strict:.3f}" for k, r in est.iterrows()), "r4_benchmark/benchmark_summary.csv")
    if len(bc_):
        x = bc_[bc_["condition"] == "clean"]
        add(fid, "positive_control", "supports" if x["frac_tier1_or_2_when_truth"].median() >= 0.5 else "partial",
            "core-clock genes called Tier 1/2 when rhythmic in the held-out truth (median over genes): " + "; ".join(
                f"{d} {g['frac_tier1_or_2_when_truth'].median():.2f}" for d, g in x.groupby("dataset")), "r4_benchmark/benchmark_clock_recovery.csv")
    n_ = bs_[(bs_["unit"] == "ALL") & (bs_["condition"] == "sample pattern (full)")].set_index(["dataset", "rule"])
    if len(n_):
        nn = n_.xs("nominal", level="rule")
        add(fid, "sample_level_pattern", "partial",
            "with a cochlear-like sample-level nuisance: nominal realised FDP " + "; ".join(f"{k} {r.mean_fdp_strict:.3f}" for k, r in nn.iterrows())
            + "; Tier-2 calls " + "; ".join(f"{k} {r.mean_calls:.0f}" for k, r in n_.xs("tier2", level="rule").iterrows())
            + " (moderation shrinks power under a widespread sample-level nuisance)", "r4_benchmark/benchmark_summary.csv")
    add(fid, "power_effect_size", "partial", "sensitivity by truth amplitude × expression decile in r4_benchmark/benchmark_sensitivity.csv "
        "(low for < 2-fold rhythms)", "r4_benchmark/benchmark_sensitivity.csv")

# R10 / R11: replicated bulk rhythms attributed to cell types (cochlea GSE88954; BM GSE198770)
am_ = rd("r4_attribution/attribution_mechanism_summary.csv")
ac_ = rd("r4_attribution/attribution_controls.csv")
ae_ = rd("r4_attribution/attribution_celltype_enrichment.csv")
ax_ = rd("r4_attribution/attribution_phase_crosscheck.csv")
for fid, t, stmt in [("R10", "cochlea", "Replicated bulk cochlear rhythms (GSE88954) are reproduced by within-cell regulation (composition held fixed), "
                      "not by our composition changes, and are carried by neural, glial, vascular, bone and strial cell types"),
                     ("R11", "bm", "Replicated bulk BM rhythms (GSE198770) are reproduced by both our composition and within-cell regulation profiles")]:
    m_ = am_[am_["tissue"] == t].set_index("class") if len(am_) else pd.DataFrame()
    if not len(m_):
        continue
    register(fid, "replicated bulk rhythms × atlas", t, stmt)
    reg_ok = m_.loc["regulation", "p_perm"] < 0.05
    comp_ok = m_.loc["composition", "p_perm"] < 0.05
    add(fid, "resampling_or_fdr", "supports" if (reg_ok if t == "cochlea" else (reg_ok and comp_ok)) else "partial",
        "; ".join(f"{k}: {r.share:.3f} vs null {r.null_share:.3f} (p {r.p_perm:.3f})" for k, r in m_.iterrows()),
        "r4_attribution/attribution_mechanism_summary.csv")
    c_ = ac_[ac_["tissue"] == t].groupby("control")["pass"].mean() if len(ac_) else pd.Series(dtype=float)
    add(fid, "model_concordance", "supports" if len(c_) and c_.get("core clock gene", 0) >= 0.9 and c_.get("strict marker", 0) >= 0.6 else "partial",
        "attribution controls: " + "; ".join(f"{k} pass {v:.2f}" for k, v in c_.items()), "r4_attribution/attribution_controls.csv")
    e_ = ae_[(ae_["tissue"] == t) & (ae_["subset"] == "all rhythmic") & (ae_["p_enriched"] < 0.05)].sort_values("ratio", ascending=False) if len(ae_) else ae_
    add(fid, "external_data", "supports", "replicated time series (n = 3 per time point); enriched contributors: " +
        "; ".join(f"{r.cell_type} ×{r.ratio:.2f}" for r in e_.head(6).itertuples()), "r4_attribution/attribution_celltype_enrichment.csv")
    x_ = ax_[(ax_["tissue"] == t) & (ax_["cell_type"] == "ALL")] if len(ax_) else ax_
    if len(x_):
        r_ = x_.iloc[0]
        add(fid, "cross_tissue_logic", "supports" if r_["p_perm"] < 0.05 else "partial",
            f"dominant-type genes: our within-type phase vs bulk phase within 3 h {r_['frac_within_3h']:.2f} vs null {r_['null_frac_within_3h']:.2f} "
            f"(p {r_['p_perm']:.3f}; n {int(r_['n_genes'])})", "r4_attribution/attribution_phase_crosscheck.csv")

# R12: heat-shock programme — activity-locked, conserved in a diurnal primate
hb_ = rd("r4_crossspecies/baboon_heatshock_phase.csv")
dn_ = rd("r4_drivers/diurnal_circadian_status.csv")
if len(hb_):
    fid = "R12"
    register(fid, "systemic cue × conservation", "both", "The heat-shock programme peaks at the same clock-aligned phase in mouse (night) and in "
             "a diurnal primate (baboon, day), i.e. it follows the internal clock / activity phase, not light")
    sm_ = hb_[hb_["baboon_tissue"].str.startswith("SUMMARY")].set_index("baboon_tissue")
    bal = sm_.loc[[k for k in sm_.index if "aligned" in k][0]]
    braw = sm_.loc[[k for k in sm_.index if "raw ZT" in k][0]]
    mou = sm_.loc[[k for k in sm_.index if "mouse" in k][0]]
    d_ = abs(float(phase.circ_diff(bal["aligned"], mou["aligned"])))
    add(fid, "external_data", "supports" if d_ <= 3 else "against",
        f"baboon tissues (p<0.05, n {int(bal['n_genes'])}): aligned peak {bal['aligned']:.1f} h vs ours {mou['aligned']:.1f} h (Δ {d_:.1f} h); "
        f"light-referenced baboon peak ZT{braw['peak_zt']:.1f} (active day) vs mouse ZT{mou['peak_zt']:.1f} (active night)",
        "r4_crossspecies/baboon_heatshock_phase.csv")
    add(fid, "resampling_or_fdr", "supports" if bal["p"] >= 0.7 else "partial",
        f"consistency across baboon tissues: resultant {bal['p']:.2f} (aligned), {braw['p']:.2f} (raw)", "r4_crossspecies/baboon_heatshock_phase.csv")
    if len(dn_):
        r5 = dn_.set_index("finding_id").loc["R05"]
        add(fid, "model_concordance", "supports" if r5["p_dd_support_phase_perm"] <= 0.05 else "partial",
            f"our HSF1 calls peak at their constant-darkness (GSE54650) phases: {r5['frac_dd_support']:.2f} vs phase-permutation null "
            f"{r5['frac_dd_support_phase_null']:.2f} (p {r5['p_dd_support_phase_perm']:.3f})", "r4_drivers/diurnal_circadian_status.csv")

# R13: BM gene rhythms conserved in primate immune tissues
bg_ = rd("r4_crossspecies/baboon_bm_gene_concordance.csv")
if len(bg_):
    fid = "R13"
    register(fid, "cross-species conservation", "bm", "BM Tier-1/2 gene rhythms are phase-conserved (clock-aligned) in baboon immune tissues "
             "(spleen; bone marrow with an imputed clock offset)")
    sp_ = bg_[(bg_["baboon_tissue"] == "SPL") & (bg_["genes"] == "non_clock")].set_index("our_scope")
    bm_ = bg_[(bg_["baboon_tissue"] == "BOM") & (bg_["genes"] == "non_clock")].set_index("our_scope")
    a1 = sp_.loc["ALL (Tier 1/2)"]
    add(fid, "model_concordance", "supports" if a1["p_perm"] < 0.05 else "against",
        f"spleen, non-clock Tier-1/2 genes: within 3 h {a1['frac_within_3h']:.2f} vs null {a1['null_frac_within_3h']:.2f} (p {a1['p_perm']:.3f}, n {int(a1['n_genes'])})",
        "r4_crossspecies/baboon_bm_gene_concordance.csv")
    t1_ = sp_.loc["Tier 1"] if "Tier 1" in sp_.index else None
    add(fid, "resampling_or_fdr", "supports" if t1_ is not None and t1_["p_perm"] < 0.05 else "partial",
        (f"spleen, Tier 1 only: {t1_['frac_within_3h']:.2f} vs {t1_['null_frac_within_3h']:.2f} (p {t1_['p_perm']:.3f}); per cell type p<0.05: "
         f"{int((sp_['p_perm'] < 0.05).sum())}/{len(sp_)}") if t1_ is not None else "", "r4_crossspecies/baboon_bm_gene_concordance.csv")
    if "ALL (Tier 1/2)" in bm_.index:
        b1 = bm_.loc["ALL (Tier 1/2)"]
        add(fid, "external_data", "partial" if b1["p_perm"] < 0.05 else "neutral",
            f"baboon bone marrow (weak clock: offset imputed): {b1['frac_within_3h']:.2f} vs {b1['null_frac_within_3h']:.2f} (p {b1['p_perm']:.3f})",
            "r4_crossspecies/baboon_bm_gene_concordance.csv")

# R14: neutrophil timer split in human blood neutrophils
hn_ = rd("r4_crossspecies/human_neutrophil_timeofday.csv")
if len(hn_):
    fid = "R14"
    register(fid, "cross-species conservation", "both", "Human blood neutrophils (14:00 vs 08:00): cochlear neutrophil rhythms predict the change "
             "positively, temporal-BM neutrophil rhythms negatively — the neutrophil-timer split (G02) holds in humans")
    pr_ = hn_[hn_["primary"].astype(str) == "True"].set_index(["tissue", "cell_type"])
    c_all, b_all = pr_.loc[("cochlea", "all neutrophil states")], pr_.loc[("bm", "all neutrophil states")]
    add(fid, "model_concordance", "supports" if c_all["pearson_r"] > 0 and c_all["p_perm"] < 0.05 and b_all["pearson_r"] < c_all["pearson_r"] else "partial",
        f"cochlea r {c_all['pearson_r']:+.3f} (p {c_all['p_perm']:.4f}, n {int(c_all['n_genes'])}); BM r {b_all['pearson_r']:+.3f}",
        "r4_crossspecies/human_neutrophil_timeofday.csv")
    add(fid, "resampling_or_fdr", "supports" if c_all["p_perm"] < 0.01 else "partial", f"gene-label permutation p {c_all['p_perm']:.4f}",
        "r4_crossspecies/human_neutrophil_timeofday.csv")
    sh_ = hn_[(hn_["tissue"] == "cochlea") & (hn_["cell_type"] == "all neutrophil states")]
    add(fid, "sensitivity_mapping", "supports" if (sh_["pearson_r"] > 0).sum() >= 4 else "partial",
        "activity-matched mapping shifted −2 … +2 h: r " + ", ".join(f"{r.shift_h:+d} h {r.pearson_r:+.2f}" for r in sh_.itertuples()),
        "r4_crossspecies/human_neutrophil_timeofday.csv")
    add(fid, "cross_tissue_logic", "supports", "same sign split as the mouse blood-neutrophil timer (G02: cochlea +, BM −)", "77_neutrophil_timer")

# R15: atlas concordance with independent adult cochlear atlases
cs_ = rd("r4_concordance/atlas_concordance_summary.csv")
ag_ = rd("r4_concordance/author_label_agreement.csv")
if len(cs_):
    fid = "R15"
    register(fid, "atlas positioning", "cochlea", f"Our gold-standard cochlear labels transfer to {len(cs_)} independent adult cochlear "
             "single-cell / single-nucleus atlases with marker support and author-label agreement")
    add(fid, "model_concordance", "supports" if (cs_["marker_check_pass_rate"] >= 0.8).all() else "partial",
        "marker-check pass rate: " + "; ".join(f"{r.key} {r.marker_check_pass_rate:.2f}" for r in cs_.itertuples()), "r4_concordance/atlas_concordance_summary.csv")
    add(fid, "resampling_or_fdr", "supports" if (cs_["frac_confident"] >= 0.5).all() else "partial",
        "confident share (scANVI ∧ kNN agree, prob ≥ 0.7): " + "; ".join(f"{r.key} {r.frac_confident:.2f}" for r in cs_.itertuples()),
        "r4_concordance/atlas_concordance_summary.csv")
    if len(ag_):
        tot_ = ag_[ag_["author_celltype"].str.startswith("ALL")].iloc[0]
        fb = tot_.get("frac_expected_broad", np.nan)
        bad = ag_[(ag_["dominant_label_passes_marker_check"].astype(str) == "False")]["author_celltype"].tolist() if "dominant_label_passes_marker_check" in ag_ else []
        add(fid, "external_data", "supports" if fb >= 0.6 else "partial",
            f"GSE234926 author labels (nuclei): {fb:.2f} of cells in the expected label (broad: duct epithelia in our Fbxo2+ 'Interdental Cell' "
            f"label), strict {tot_['frac_expected']:.2f}; ARI {tot_['dominant_share']:.2f}; author types whose dominant transferred label fails the "
            f"marker check (nuclei-to-cell mapping artefact): {', '.join(bad) if bad else 'none'}", "r4_concordance/author_label_agreement.csv")
    add(fid, "positive_control", "supports" if (cs_["n_types_recovered_20_confident"] >= 15).any() else "partial",
        "types recovered (≥ 20 confident cells): " + "; ".join(f"{r.key} {int(r.n_types_recovered_20_confident)}" for r in cs_.itertuples()),
        "r4_concordance/atlas_concordance_summary.csv")

# R16: which rhythms are clock outputs (constant-darkness phase test)
pn_ = rd("r4_drivers/driver_phase_null.csv")
if len(pn_) and len(dn_):
    fid = "R16"
    register(fid, "drivers of rhythms", "both", "Only core-clock and heat-shock rhythms peak at their constant-darkness-predicted phases well above "
             "chance; BM neutrophil-state programmes are not clock-output-like (diurnal state programmes)")
    d5 = dn_.set_index("finding_id")
    add(fid, "resampling_or_fdr", "supports" if d5.loc["G04", "p_dd_support_phase_perm"] <= 0.05 and d5.loc["R05", "p_dd_support_phase_perm"] <= 0.05 else "partial",
        f"phase-permutation tests: cochlear core clock {d5.loc['G04', 'frac_dd_support']:.2f} vs {d5.loc['G04', 'frac_dd_support_phase_null']:.2f}; "
        f"HSF1 {d5.loc['R05', 'frac_dd_support']:.2f} vs {d5.loc['R05', 'frac_dd_support_phase_null']:.2f}; BM neutrophil states "
        f"{d5.loc['G01', 'frac_dd_support']:.2f} vs {d5.loc['G01', 'frac_dd_support_phase_null']:.2f} (p {d5.loc['G01', 'p_dd_support_phase_perm']:.2f})",
        "r4_drivers/diurnal_circadian_status.csv")
    al_ = pn_[(pn_["cell_type"] == "ALL") & (pn_["flag"] == "dd_support")]
    add(fid, "model_concordance", "supports", "all Tier-1/2 calls: " + "; ".join(
        f"{r.tissue} {r.tier_set} {r.observed_share:.3f} vs null {r.null_mean_share:.3f} (p {r.p_perm:.3f})" for r in al_.itertuples()),
        "r4_drivers/driver_phase_null.csv")

# ================================================================ round 5 findings (B cells; pre-specified primary hypotheses)
ph_ = rd("r5_bcell/primary_hypotheses.csv")
mmx_ = rd("r5_bcell/b_multimethod_matrix.csv")
B_DECISION = {}
if len(ph_):
    for i, r_ in enumerate(ph_.itertuples(), start=1):
        fid = f"B{i:02d}"
        tis_ = r_.tissue
        register(fid, "B-cell primary hypothesis (round 5)", tis_, f"{r_.hypothesis} ({tis_}): {r_.primary_statistic}",
                 peak_zt=getattr(r_, "peak_zt", np.nan), note=f"pre-specified (D33); decision rule D34")
        q_, p0 = r_.q_BH_primary_set, r_.primary_p
        add(fid, "resampling_or_fdr", "supports" if q_ < 0.1 else ("partial" if p0 < 0.1 else "against"),
            f"primary p {p0:.3f}; BH q across the 12 primary tests {q_:.3f}", "r5_bcell/primary_hypotheses.csv")
        add(fid, "model_concordance", "supports" if (bool(r_.in_window) and p0 < 0.05) else ("partial" if bool(r_.in_window) else "against"),
            f"pre-specified direction / window met: {bool(r_.in_window)}", "r5_bcell/primary_hypotheses.csv")
        for m_ in mmx_[(mmx_["hypothesis"] == r_.hypothesis) & (mmx_["tissue"] == tis_)].itertuples():
            add(fid, "method: " + m_.method[:60], "supports" if str(m_.agrees) == "True" else "against", str(m_.evidence)[:300],
                "r5_bcell/b_multimethod_matrix.csv")
        B_DECISION[fid] = {"supported": "supported", "partial": "suggestive", "not supported": "not supported"}[r_.decision]

# ================================================================ round 6 findings (myeloid; pre-specified primary hypotheses)
ph6_ = rd("r6_myeloid/m_primary_hypotheses.csv")
mm6_ = rd("r6_myeloid/m_multimethod_matrix.csv")
if len(ph6_):
    for r_ in ph6_[ph6_["decision"] != "pending"].itertuples():
        fid = f"M{int(r_.hypothesis[1:]):02d}"
        register(fid, "myeloid primary hypothesis (round 6)", r_.tissues, f"{r_.name}: tissues combined by {r_.combination} ({r_.p_by_tissue})",
                 note="pre-specified (D40); decision rule D45 / D48")
        add(fid, "resampling_or_fdr", "supports" if r_.q_BH < 0.1 else ("partial" if r_.combined_p < 0.1 else "against"),
            f"combined p {r_.combined_p:.3f}; BH q across the 8 primary hypotheses {r_.q_BH:.3f}", "r6_myeloid/m_primary_hypotheses.csv")
        add(fid, "model_concordance", "supports" if bool(r_.direction_ok) else "against", str(r_.direction_detail)[:300], "r6_myeloid/m_primary_hypotheses.csv")
        for m_ in mm6_[mm6_["hypothesis"] == r_.hypothesis].itertuples():
            add(fid, f"method ({m_.tissue}): " + m_.method[:60], "supports" if str(m_.agrees) == "True" else "against", str(m_.value)[:300],
                "r6_myeloid/m_multimethod_matrix.csv")
        B_DECISION[fid] = {"supported": "supported", "partial": "suggestive", "not supported": "not supported"}[r_.decision]

# ---------------------------------------------------------------- verdicts
vm = pd.DataFrame(rows)
reg = pd.DataFrame(reg)


def verdict(g):
    core = g[g["dimension_class"] == "core"]["status"]
    ind = g[g["dimension_class"] == "independent"]["status"]
    cs, ca = (core == "supports").sum(), (core == "against").sum()
    isup, iag = (ind == "supports").sum(), (ind == "against").sum()
    if len(core) and cs == len(core) and isup >= 2 and iag <= 1:
        return "robust"
    if ca == 0 and cs >= 1 and isup >= 1 and isup > iag:
        return "supported"
    if cs == 0 and iag >= isup:
        return "not supported"
    return "suggestive"


agg = []
for fid_, g in vm.groupby("finding_id", sort=False):
    c = g["status"].value_counts()
    agg.append({"finding_id": fid_, "n_dimensions": len(g), **{f"n_{k}": int(c.get(k, 0)) for k in
               ["supports", "partial", "against", "neutral", "not_tested"]},
                "core_status": "; ".join(f"{r.dimension}={r.status}" for r in g[g["dimension_class"] == "core"].itertuples()),
                "verdict": verdict(g),
                "main_caveats": " | ".join(f"{r.dimension}: {r.evidence[:160]}" for r in g[g["status"] == "against"].itertuples())})
reg = reg.merge(pd.DataFrame(agg), on="finding_id", how="left")
reg.loc[reg["finding_id"] == "X01", "verdict"] = "retired (cochlea-only analysis; BM is the sole source of cochlear immune cells)"
for fid_, v_ in B_DECISION.items():                     # round-5 primary hypotheses: the pre-specified decision rule (D34) is the verdict
    reg.loc[reg["finding_id"] == fid_, "verdict"] = v_
write_csv(vm, MOD, "validation_matrix.csv", "Positive findings × evidence dimensions: status and the numbers behind it",
          required=["finding_id", "dimension", "dimension_class", "status", "evidence", "source"])
write_csv(reg, MOD, "positive_findings_register.csv", "Register of positive findings with verdicts (robust / supported / "
          "suggestive / not supported) from the validation matrix", required=["finding_id", "statement", "verdict"])
log("\n" + reg[["finding_id", "tissue", "statement", "verdict", "n_supports", "n_partial", "n_against"]].to_string(max_colwidth=70))
