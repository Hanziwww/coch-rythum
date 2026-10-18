"""Round 5, WP-S — synthesis of the B-cell round: pre-specified primary hypotheses (D33 / D34), multi-method matrix,
24-h timeline and BM-vs-cochlea concordance.

Primary tests (12): H1–H5 in BM and cochlea, H7 and H8 in the cochlea (H6 dropped, D29). Decision rule (D34):
supported = BH q < 0.1 across the 12 primary tests AND >= 2 agreeing independent methods AND the pre-specified direction /
window met; partial = >= 2 agreeing methods (or q < 0.1 with < 2) but not both; not supported = otherwise.
Agreeing methods (pre-specified in the plan; evaluated from the round-5 tables):
  H1  share-of-tissue multi-model verdict (202) robust / supported with the peak in ZT16–24; Milo naïve-majority
      neighbourhoods (53, scANVI; nominal p < 0.05) circular-mean peak in ZT16–24; GSE198770 ImmGen FrF signature (BM
      only) p < 0.05 with the our-equivalent peak in ZT16–24; baboon spleen FrF signature p < 0.05 in window
  H2  Milo pseudotime wave (scVI) p < 0.05 with later quintiles peaking later (Q5 − Q1 forward lag 0–12 h); any
      lineage pseudotime quantile rhythmic (exact p <= 0.05); GSE198770 ImmGen order test p < 0.05
  H3  cell-level axis (pooled naïve + mature, all cells) exact p <= 0.05 in ZT12–24; identity drift (naïve → mature
      posterior) exact p <= 0.05; GSE198770 FrF-vs-FrE axis p < 0.05 in window; baboon spleen axis p < 0.05 in window
  H4  cell-level tracking p <= 0.05 with |offset vs structural| >= 3 h; shift >= 3 h with the same sign in both tissues;
      lymphoid comparators (T, NK) not shifted (|diff| < 3 h); GSE175609: our BM B rhythms predict WT ZT12 − ZT0 (p <
      0.05) better than KO
  H5  ligand gene-level rhythm in >= 2 axes; NicheNet top 10 % in >= 2 axes; MultiNicheNet B-receiver pairs for >= 2 axes
  H7  B-side gene rhythm (64) behind a nominally rhythmic curated-family interaction; NicheNet partner
      percentile >= 0.9 for a B-derived ligand of a curated family; spatial plausibility (H8 primary p < 0.05)
  H8  external ImmGen B-lineage signature p < 0.05 (neuron > supporting); prior knowledge (modiolar marrow next to
      Rosenthal's canal; scripts/resources/spatial_priors.csv)
Outputs (results/r5_bcell/): primary_hypotheses.csv, b_multimethod_matrix.csv, timeline_24h.csv, b_crosstissue_concordance.csv
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

MOD = "r5_bcell"
R5 = RESULTS / "r5_bcell"
RP = RESULTS / "r5_partners"
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]


def rd(p):
    return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()


def inwin(h, lo, hi):
    return bool(pd.notna(h) and lo <= h % 24 < hi)


prim = pd.concat([rd(R5 / "b_primary_composition.csv"), rd(R5 / "b_primary_h3.csv"), rd(R5 / "b_primary_h4.csv"),
                  rd(R5 / "b_primary_h5.csv"), rd(RP / "b_primary_h7_h8.csv")], ignore_index=True)
prim["q_BH_primary_set"] = multipletests(prim["primary_p"].fillna(1), method="fdr_bh")[1]
mm = []


def add(h, t, method, ok, evidence):
    mm.append({"hypothesis": h, "tissue": t, "method": method, "agrees": ok, "evidence": evidence})


comp = rd(R5 / "b_composition_summary.csv")
ms = rd(RESULTS / "r2_figure/abundance_multimodel_summary.csv")
milo = {t: rd(RESULTS / f"53_milo/milo_da_results_{t}.csv") for t in ["bm", "cochlea"]}
ext = rd(R5 / "b_external_signature_rhythm.csv")
wave = rd(R5 / "b_milo_wave.csv")
ptr = rd(R5 / "b_pseudotime_rhythm.csv")
eord = rd(R5 / "b_external_order_tests.csv")
axr = rd(R5 / "b_state_axes_rhythm.csv")
drift = rd(R5 / "b_identity_drift_rhythm.csv")
ck = rd(R5 / "b_clock_offsets.csv")
cl = rd(R5 / "b_clock_cell_level.csv")
bk = rd(R5 / "b_external_bmal1ko.csv")
h5 = rd(R5 / "b_primary_h5.csv").set_index("tissue") if len(rd(R5 / "b_primary_h5.csv")) else pd.DataFrame()
pint = rd(RP / "b_partner_interactions.csv")
geo = rd(RP / "b_partner_geomx.csv")
for t in ["bm", "cochlea"]:
    # ---- H1
    r = ms[(ms["tissue"] == t) & (ms["cell_type"] == "B Cell (Naïve)")]
    add("H1 naïve-B timing", t, "share-of-tissue multi-model (202)", bool(len(r) and r["verdict"].iloc[0] in ("robust", "supported") and inwin(r["ols_pct_peak_zt"].iloc[0], 16, 24)),
        f"verdict {r['verdict'].iloc[0]}, peak ZT{r['ols_pct_peak_zt'].iloc[0]:.1f}" if len(r) else "")
    m = milo[t]
    m = m[(m["majority_cell_type"] == "B Cell (Naïve)") & (m["p"] < 0.05)]
    mu = phase.circ_mean(m["peak_zt"].to_numpy())[0] if len(m) >= 5 else np.nan
    add("H1 naïve-B timing", t, "Milo naïve-majority neighbourhoods (53)", bool(len(m) >= 5 and inwin(mu, 16, 24)), f"{len(m)} nhoods p < 0.05, circular mean ZT{mu:.1f}")
    if t == "bm":
        e = ext[(ext["dataset"] == "GSE198770 BM bulk") & (ext["condition"] == "control") & (ext["signature"] == "ImmGen B.FrF.BM")]
        add("H1 naïve-B timing", t, "GSE198770 BM bulk FrF signature", bool(len(e) and e["p"].iloc[0] < 0.05 and inwin(e["peak_our_bm_zt_equiv"].iloc[0], 16, 24)),
            f"p {e['p'].iloc[0]:.4f}, our-equivalent ZT{e['peak_our_bm_zt_equiv'].iloc[0]:.1f}" if len(e) else "")
    e = ext[(ext["dataset"] == "baboon spleen") & (ext["signature"] == "ImmGen B.FrF.BM")]
    add("H1 naïve-B timing", t, "baboon spleen FrF signature", bool(len(e) and e["p"].iloc[0] < 0.05 and inwin(e["peak_our_bm_zt_equiv"].iloc[0], 16, 24)),
        f"p {e['p'].iloc[0]:.3f}, our-equivalent ZT{e['peak_our_bm_zt_equiv'].iloc[0]:.1f}" if len(e) else "")
    # ---- H2
    w = wave[(wave["tissue"] == t) & (wave["selection"] == "nominal p < 0.1")]
    ok, ev = False, ""
    if len(w):
        qs = [float(x.split("ZT")[1].split(" ")[0]) for x in w["peak_by_pseudotime_quintile"].iloc[0].split("; ")]
        lag = (qs[-1] - qs[0]) % 24
        ok = bool(w["p_perm"].iloc[0] < 0.05 and 0 < lag < 12)
        ev = f"r {w['circ_linear_r'].iloc[0]:.2f}, p {w['p_perm'].iloc[0]:.4f}; quintile peaks {', '.join(f'{q:.1f}' for q in qs)} (Q5 − Q1 {lag:.1f} h)"
    add("H2 developmental order", t, "Milo pseudotime wave (scVI)", ok, ev)
    p_ = ptr[(ptr["tissue"] == t) & (ptr["embedding"] == "scVI (unsupervised; primary)") & ptr["feature"].str.startswith("lineage_q")]
    add("H2 developmental order", t, "lineage pseudotime quantile rhythms", bool((p_["p_perm"] <= 0.05).any()),
        "; ".join(f"{r_.feature} ZT{r_.peak_zt:.1f} p {r_.p_perm:.3f}" for r_ in p_.itertuples()))
    eo = eord[(eord["dataset"] == "GSE198770 BM bulk") & (eord["condition"] == "control")]
    add("H2 developmental order", t, "GSE198770 ImmGen order test", bool(len(eo) and eo["p_exact"].iloc[0] < 0.05),
        f"T {eo['T_forward_lag_h'].iloc[0]:.1f} h, p {eo['p_exact'].iloc[0]:.3f}" if len(eo) else "")
    # ---- H3
    a = axr[(axr["tissue"] == t) & (axr["axis"] == "immgen_FrF_vs_FrE") & (axr["level"] == "cell (all cells)") & (axr["group"] == "B naïve + mature (pooled)")]
    add("H3 state alternation", t, "cell-level axis (all cells)", bool(len(a) and a["p_perm"].iloc[0] <= 0.05 and inwin(a["peak_zt"].iloc[0], 12, 24)),
        f"ZT{a['peak_zt'].iloc[0]:.1f}, p {a['p_perm'].iloc[0]:.3f}" if len(a) else "")
    d = drift[(drift["tissue"] == t) & (drift["stage"] == "B Cell (Naïve)")]
    add("H3 state alternation", t, "identity drift naïve → mature posterior", bool(len(d) and d["p_perm"].iloc[0] <= 0.05),
        f"ZT{d['peak_zt'].iloc[0]:.1f}, p {d['p_perm'].iloc[0]:.3f}" if len(d) else "")
    for ds in ["GSE198770 BM bulk", "baboon spleen"]:
        e = ext[(ext["dataset"] == ds) & (ext["signature"] == "ImmGen axis FrF_vs_FrE") & (ext["condition"].isin(["control", ""]) | ext["condition"].isna())]
        add("H3 state alternation", t, f"{ds} FrF-vs-FrE axis", bool(len(e) and e["p"].iloc[0] < 0.05 and inwin(e["peak_our_bm_zt_equiv"].iloc[0], 12, 24)),
            f"p {e['p'].iloc[0]:.3f}, our-equivalent ZT{e['peak_our_bm_zt_equiv'].iloc[0]:.1f}" if len(e) else "")
    # ---- H4
    c = cl[(cl["tissue"] == t) & (cl["group"] == "B lineage (pooled)")]
    add("H4 B-clock shift", t, "cell-level clock tracking with shift >= 3 h", bool(len(c) and c["p_perm"].iloc[0] <= 0.05 and abs(c["offset_minus_structural_h"].iloc[0]) >= 3),
        f"tracking p {c['p_perm'].iloc[0]:.3f}, offset vs structural {c['offset_minus_structural_h'].iloc[0]:+.1f} h, usable cells {c['frac_cells_usable'].iloc[0]:.2f}"
        if len(c) and pd.notna(c["p_perm"].iloc[0]) else "insufficient clock UMIs")
    bb = ck[(ck["unit"] == "B lineage (pooled)")].set_index("tissue")["diff_vs_tissue_pool_h"]
    add("H4 B-clock shift", t, "same shift in both tissues", bool(len(bb) == 2 and (bb.abs() >= 3).all() and np.sign(bb).nunique() == 1),
        "; ".join(f"{k} {v:+.1f} h" for k, v in bb.items()))
    cp = ck[(ck["tissue"] == t) & ck["unit"].isin(["T Cell (CD3+)", "NK Cell"])]
    add("H4 B-clock shift", t, "lymphoid comparators not shifted", bool(len(cp) and (cp["diff_vs_tissue_pool_h"].abs() < 3).all()),
        "; ".join(f"{r_.unit} {r_.diff_vs_tissue_pool_h:+.1f} h (±{r_.diff_ci_halfwidth_h:.1f})" for r_ in cp.itertuples()))
    k = bk[bk["analysis"].str.contains(f"our {t} rhythms predict") & (bk["gene_or_set"] == "all B stages")]
    wt_ = k[k["analysis"].str.endswith("(WT)")]
    ko_ = k[k["analysis"].str.endswith("(KO)")]
    okk = bool(len(wt_) and len(ko_) and wt_["WT_p"].iloc[0] < 0.05 and wt_["WT_value"].iloc[0] > ko_["KO_value"].iloc[0])
    add("H4 B-clock shift", t, "GSE175609 B-cell Bmal1 KO attenuates agreement", okk,
        f"WT r {wt_['WT_value'].iloc[0]:+.3f} (p {wt_['WT_p'].iloc[0]:.3f}), KO r {ko_['KO_value'].iloc[0]:+.3f}" if len(wt_) and len(ko_) else "")
    # ---- H5
    if t in getattr(h5, "index", []):
        x = h5.loc[t]
        add("H5 niche signals lead", t, "ligand gene-level rhythm (>= 2 axes)", bool(x["n_axes_ligand_gene_rhythmic"] >= 2), f"{x['n_axes_ligand_gene_rhythmic']} axes")
        add("H5 niche signals lead", t, "NicheNet top 10 % (>= 2 axes)", bool(x["n_axes_nichenet_top10pct"] >= 2), f"{x['n_axes_nichenet_top10pct']} axes")
        mh = str(x["multinichenet_axes_present"]) if pd.notna(x["multinichenet_axes_present"]) else ""
        add("H5 niche signals lead", t, "MultiNicheNet B receivers (>= 2 axes)", len([s for s in mh.split("; ") if s]) >= 2, mh)
# ---- H7 / H8 (cochlea)
if len(pint):
    # plan's method: B-side ligand / receptor gene-level rhythm (33 / 64) behind a nominally rhythmic curated-family interaction
    # (interactions share sample-level profiles, so counts of nominal hits are not independent and are not tested here)
    cur = pint[(pint["pair_family"].fillna("") != "") & (pint["p_perm"] <= 0.05)]
    hit = cur[cur["b_side_gene_rhythmic"].astype(str) == "True"]
    add("H7 B↔partner signals", "cochlea", "B-side gene rhythm behind a nominally rhythmic curated-family interaction", bool(len(hit)),
        f"{len(hit)} of {len(cur)} nominal curated-family interactions with a rhythmic B-side gene: " +
        "; ".join(f"{r_.source}→{r_.target} {r_.ligand_complex}→{r_.receptor_complex}" for r_ in hit.head(5).itertuples()))
    nnp = pint.loc[(pint["pair_family"].fillna("") != "") & (pint["direction"] == "B → partner"), "nichenet_partner_percentile"] if "nichenet_partner_percentile" in pint else pd.Series(dtype=float)
    add("H7 B↔partner signals", "cochlea", "NicheNet partner percentile >= 0.9 (curated B-derived ligands)", bool((nnp >= 0.9).any()),
        f"max percentile {nnp.max():.2f}" if nnp.notna().any() else "no curated B-derived ligand scored")
h8 = prim[prim["hypothesis"] == "H8 spatial adjacency"]
add("H7 B↔partner signals", "cochlea", "spatial plausibility (H8 primary p < 0.05)", bool(len(h8) and h8["primary_p"].iloc[0] < 0.05), f"H8 p {h8['primary_p'].iloc[0]:.3f}" if len(h8) else "")
g = geo[(geo["subset"] == "control slides") & (geo["measure"] == "immgen_b_signature_score")]
add("H8 spatial adjacency", "cochlea", "external ImmGen B-lineage signature", bool(len(g) and g["p_wilcoxon_neuron_gt_supporting"].iloc[0] < 0.05),
    f"p {g['p_wilcoxon_neuron_gt_supporting'].iloc[0]:.3f}" if len(g) else "")
add("H8 spatial adjacency", "cochlea", "prior knowledge (modiolar marrow / Rosenthal's canal)", True, "scripts/resources/spatial_priors.csv")
mmd = pd.DataFrame(mm)
write_csv(mmd, MOD, "b_multimethod_matrix.csv", "Round-5 primary hypotheses × pre-specified agreeing methods: agreement and evidence",
          required=["hypothesis", "tissue", "method", "agrees", "evidence"])
agree = mmd.groupby(["hypothesis", "tissue"])["agrees"].agg(["sum", "size"]).rename(columns={"sum": "n_methods_agree", "size": "n_methods"})
prim = prim.merge(agree.reset_index(), on=["hypothesis", "tissue"], how="left")
prim["in_window"] = prim["in_window"].fillna(False).astype(bool)


def decide(r):
    fdr = r["q_BH_primary_set"] < 0.1
    meth = r["n_methods_agree"] >= 2
    if fdr and meth and r["in_window"]:
        return "supported"
    if meth or (fdr and r["in_window"]):
        return "partial"
    return "not supported"


prim["decision"] = prim.apply(decide, axis=1)
write_csv(prim, MOD, "primary_hypotheses.csv", "Round-5 pre-specified primary hypotheses: primary p, BH q across the 12 tests, agreeing methods, decision (D34)",
          required=["hypothesis", "tissue", "primary_p", "q_BH_primary_set", "n_methods_agree", "decision"])

# ---------------- timeline
tl = []
for r in comp.itertuples():
    if r.frame in ("share_of_tissue", "within_B"):
        tl.append({"tissue": r.tissue, "layer": f"B composition ({r.frame})", "event": r.feature, "peak_zt": r.peak_zt, "p": r.p_perm,
                   "ci_halfwidth_h": r.boot_peak_ci_halfwidth_h, "status": r.verdict, "source": "r5_bcell/b_composition_summary.csv"})
for r in wave[wave["selection"] == "nominal p < 0.1"].itertuples():
    for q in r.peak_by_pseudotime_quintile.split("; "):
        tl.append({"tissue": r.tissue, "layer": "B neighbourhoods by pseudotime (Milo, scVI)", "event": q.split(" ")[0],
                   "peak_zt": float(q.split("ZT")[1].split(" ")[0]), "p": r.p_perm, "status": "wave test", "source": "r5_bcell/b_milo_wave.csv"})
for r in axr[(axr["level"] == "pseudobulk") & (axr["version"] == "primary") & (axr["kind"] == "centred")].itertuples():
    tl.append({"tissue": r.tissue, "layer": f"B state axis ({r.axis})", "event": r.group, "peak_zt": r.peak_zt, "p": r.p_perm,
               "status": "exploratory" if r.group != "B naïve + mature (pooled)" else "H3 primary", "source": "r5_bcell/b_state_axes_rhythm.csv"})
pr = rd(R5 / "b_programme_rhythm.csv")
for r in pr[(pr["version"] == "primary") & (pr["kind"] == "centred") & (pr["p_perm"] <= 0.05)].itertuples():
    tl.append({"tissue": r.tissue, "layer": "B programme (exploratory)", "event": f"{r.stage}: {r.gene_set}", "peak_zt": r.peak_zt, "p": r.p_perm,
               "status": "robust" if r.robust else "nominal", "source": "r5_bcell/b_programme_rhythm.csv"})
ax_ = rd(R5 / "b_niche_axes.csv")
for r in ax_[ax_["representative"]].itertuples():
    tl.append({"tissue": r.tissue, "layer": "niche → B interaction", "event": f"{r.axis}: {r.source} → {r.target}", "peak_zt": r.peak_zt, "p": r.p_perm,
               "status": f"ligand peak ZT{r.ligand_peak_zt:.1f}" if pd.notna(r.ligand_peak_zt) else "", "source": "r5_bcell/b_niche_axes.csv"})
for r in ext[ext["peak_our_bm_zt_equiv"].notna()].itertuples():
    tl.append({"tissue": "external", "layer": f"external {r.dataset} {r.condition}".strip(), "event": r.signature, "peak_zt": r.peak_our_bm_zt_equiv, "p": r.p,
               "status": "our BM ZT equivalent", "source": "r5_bcell/b_external_signature_rhythm.csv"})
cc = rd(RESULTS / "r3_audit/cell_clock_phase_summary.csv")
for r in cc[cc["cell_type"].isin(["Hair Cell (Outer/Inner)", "Spiral Ganglion Neuron", "Schwann Cell", "Oligodendrocyte", "SV Marginal Cell",
                                   "SV Intermediate Cell (Functional)", "SV Intermediate Cell (Pigmented)"])].itertuples():
    tl.append({"tissue": r.tissue, "layer": "partner clock (cell level; context, D29)", "event": r.cell_type, "peak_zt": np.nan, "p": r.p_perm,
               "status": f"offset {r.mean_offset_cell_phase_minus_zt_h:+.1f} h, resultant {r.offset_resultant:.2f}", "source": "r3_audit/cell_clock_phase_summary.csv"})
tl = pd.DataFrame(tl)
tl["peak_clock"] = phase.zt_to_clock(tl["peak_zt"])
write_csv(tl, MOD, "timeline_24h.csv", "Round-5 24-h timeline: B composition, pseudotime wave, state axes, programmes, niche interactions, external signatures, partner clocks (ZT and clock)",
          required=["tissue", "layer", "event", "peak_zt", "peak_clock", "p", "status"])

# ---------------- cross-tissue concordance
xc = []
for frame in ["share_of_tissue", "within_B"]:
    for st in BST:
        a = comp[(comp["frame"] == frame) & (comp["feature"] == st)].set_index("tissue")
        if {"bm", "cochlea"} <= set(a.index):
            xc.append({"feature": f"{frame}: {st}", "peak_bm_zt": a.loc["bm", "peak_zt"], "peak_cochlea_zt": a.loc["cochlea", "peak_zt"],
                       "diff_cochlea_minus_bm_h": float(phase.circ_diff(a.loc["cochlea", "peak_zt"], a.loc["bm", "peak_zt"])),
                       "verdict_bm": a.loc["bm", "verdict"], "verdict_cochlea": a.loc["cochlea", "verdict"]})
for (ax_n, grp), a in axr[(axr["level"] == "pseudobulk") & (axr["version"] == "primary") & (axr["kind"] == "centred")].groupby(["axis", "group"]):
    a = a.set_index("tissue")
    if {"bm", "cochlea"} <= set(a.index):
        xc.append({"feature": f"axis {ax_n}: {grp}", "peak_bm_zt": a.loc["bm", "peak_zt"], "peak_cochlea_zt": a.loc["cochlea", "peak_zt"],
                   "diff_cochlea_minus_bm_h": float(phase.circ_diff(a.loc["cochlea", "peak_zt"], a.loc["bm", "peak_zt"])),
                   "verdict_bm": f"p {a.loc['bm', 'p_perm']:.3f}", "verdict_cochlea": f"p {a.loc['cochlea', 'p_perm']:.3f}"})
xc = pd.DataFrame(xc)
xc["same_phase_within_3h"] = xc["diff_cochlea_minus_bm_h"].abs() <= 3
write_csv(xc, MOD, "b_crosstissue_concordance.csv", "BM vs cochlea phase concordance of B composition and state axes (equal weight, D27)",
          required=["feature", "peak_bm_zt", "peak_cochlea_zt", "diff_cochlea_minus_bm_h"])
log("\n" + prim[["hypothesis", "tissue", "primary_p", "q_BH_primary_set", "in_window", "n_methods_agree", "n_methods", "decision"]].round(3).to_string())
log("\n" + mmd[mmd["agrees"]].to_string(max_colwidth=90))
