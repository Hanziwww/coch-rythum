"""Round 4, WP-D — what drives each rhythm? Driver classes for every rhythmic call, landscape per cell type, enrichment
against expression-matched non-rhythmic genes, and the diurnal-vs-circadian evidence status of headline findings.

Calls: Tier 1/2 (80_atlas/rhythmic_catalogue.csv, both tissues) and cochlear Tier C (r2_cochlea/tierC_catalogue.csv;
exploratory). Evidence flags per gene × cell type (all reported; one primary class by precedence):
  core_clock         member of the curated clock sets (resources/curated_sets.yaml: clock)
  temperature        member of the HSF1 heat-shock set (curated) or cold-inducible (Cirbp) — body-temperature cues
  handling_linked    profile follows the cell type's dissociation-stress profile over the 6 samples (r > 0.8; 330 b)
  composition_linked lineage-foreign marker (80_atlas foreign_marker_of non-empty; phase match reported separately)
  gc_driven          GC-dependent rhythmic in cochlea GSE88954 (sham rhythmic, lost after ADX; 76 dependence_gene_classes
                     'dependent') or curated GC-response member, with clock-aligned peak within ±5 h of ZT13
  dd_support         rhythmic in ≥ 2 organs of GSE54650 (constant darkness; q < 0.05) with a clock-aligned external
                     phase (circular mean over those organs) within ±4 h of ours (aligned with 230 offsets)
  clock_motif_phase  promoter E-box / E′-box (expected ZT8), D-box (ZT12) or RORE (ZT22) present (44 scan) and the
                     aligned peak within ±4 h of that element's expected phase
  clock_controlled   dd_support and/or clock_motif_phase (both = strong)
  pattern_linked     profile follows the tissue-wide sample pattern (cochlea 210/211 pattern_r > 0.8; BM r3_bm pattern_r)
Primary precedence: core clock > temperature > handling > composition > GC > clock-controlled > sample pattern > unexplained.
Aligned phase = peak ZT − this atlas's tissue clock offset (230, canonical; cochlea / BM tissue pool).
Enrichment: for each tissue (and cell type with ≥ 30 calls), CMH odds ratio (AveExpr decile strata) of phase-free
evidence (DD-rhythmic in ≥ 2 organs; GC-dependent; promoter clock element; HSE) among calls vs tested non-rhythmic genes
(p ≥ 0.2) of the same cell type.
Phase null: phase-dependent flags (dd_support, clock_motif_phase, clock_controlled, gc_driven) recomputed after shuffling
peaks within tissue × tier set × cell type (200 draws) — a flag share above this null means genes peak at their
clock-predicted phases more often than chance.
Outputs (results/r4_drivers/): driver_calls.csv, driver_landscape.csv, driver_phase_null.csv, driver_enrichment.csv,
diurnal_circadian_status.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_drivers"
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
CLOCK = set().union(*[set(v) for v in SETS["clock"].values()])
HSF1 = set(SETS["systemic_cues"]["heat_shock_hsf1"]) | {"Dnaja1", "Hspa1b", "Hsp90ab1", "Bag3", "Chordc1", "Fkbp4", "Stip1", "Cryab"}
COLD = set(SETS["systemic_cues"]["temperature_cold_inducible"])
GCSET = set(SETS["systemic_cues"]["glucocorticoid_response"])
CLOCK_EL = {"Ebox_CACGTG": 8.0, "Eprime_CACGTT": 8.0, "Dbox_TTAYGTAA": 12.0, "RORE_WAWNTRGGTCA": 22.0}
PREC = ["core clock", "temperature", "handling-linked", "composition-linked", "glucocorticoid-driven", "clock-controlled",
        "sample-pattern-linked", "unexplained"]

off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
OFF = {t: float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == t) & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
       for t in ["bm", "cochlea"]}
EXT_OFF = off[off["dataset"] == "GSE54650"].set_index("tissue")["offset_canonical_h"].to_dict()
log(f"atlas offsets {OFF}; GSE54650 organs {len(EXT_OFF)}")

# ---------------- calls
cat = pd.read_csv(RESULTS / "80_atlas/rhythmic_catalogue.csv")
cat = cat[["tissue", "cell_type", "gene", "tier", "AveExpr", "ptt_fc", "peak_zt", "foreign_marker_of", "foreign_marker_phase_match"]]
tc = pd.read_csv(RESULTS / "r2_cochlea/tierC_catalogue.csv")
tc = tc[~tc["primary_tier"].astype(str).isin(["tier1", "tier2"])]          # Tier C calls not already Tier 1/2
tc = tc.assign(tissue="cochlea", tier="tierC", foreign_marker_of=tc["marker_of"].where(tc["lineage_foreign_marker"].astype(str) == "True", ""),
               foreign_marker_phase_match=np.nan, AveExpr=np.nan)[cat.columns]
calls = pd.concat([cat, tc], ignore_index=True)
calls["foreign_marker_of"] = calls["foreign_marker_of"].fillna("").astype(str).replace("nan", "")
log(f"calls: {calls.groupby(['tissue', 'tier']).size().to_dict()}")

# ---------------- tested genes (for enrichment and AveExpr of Tier C)
tested = {}
for t in ["bm", "cochlea"]:
    a = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "AveExpr", "p", "tier", "status"])
    tested[t] = a[a["status"] == "primary"].rename(columns={"group": "cell_type"})
ae = pd.concat([d.assign(tissue=t) for t, d in tested.items()]).set_index(["tissue", "cell_type", "gene"])["AveExpr"]
miss = calls["AveExpr"].isna()
calls.loc[miss, "AveExpr"] = [ae.get((r.tissue, r.cell_type, r.gene), np.nan) for r in calls[miss].itertuples()]

# ---------------- evidence: GSE54650 constant darkness
e = pd.read_csv(RESULTS / "r2_external/external_rhythm_GSE54650.csv", usecols=["tissue", "gene", "q_BH", "peak_h"])
e = e[e["tissue"].isin(EXT_OFF)].copy()
e["internal"] = phase.wrap24(e["peak_h"] - e["tissue"].map(EXT_OFF))
er = e[e["q_BH"] < 0.05]
dd_n = er.groupby("gene").size()
dd_phase = er.groupby("gene")["internal"].apply(lambda h: phase.circ_mean(h.to_numpy())[0])
dd_any = set(dd_n[dd_n >= 2].index)

# promoter motifs
mp = pd.read_csv(RESULTS / "44_motifs/promoter_motif_presence.csv")
mp = mp[mp["n_sites"] > 0].groupby("motif")["gene"].apply(set).to_dict()
CLOCK_EL_GENES = set().union(*[mp.get(m, set()) for m in CLOCK_EL])
# GC dependence (cochlea GSE88954)
dep = pd.read_csv(RESULTS / "76_dependence/dependence_gene_classes.csv")
gcdep = set(dep[(dep["dataset"] == "GSE88954") & (dep["class"] == "dependent")]["gene"])
gc_tested = set(dep[dep["dataset"] == "GSE88954"]["gene"])

# sample-pattern correlation
pc = pd.read_csv(RESULTS / "r2_cochlea/gene_pattern_scores_cochlea.csv", usecols=["cell_type", "gene", "pattern_r"]).assign(tissue="cochlea")
pb_ = pd.read_csv(RESULTS / "r3_bm/bm_relaxed_tier.csv", usecols=["cell_type", "gene", "pattern_r"]).assign(tissue="bm")
patt = pd.concat([pc, pb_]).drop_duplicates(["tissue", "cell_type", "gene"]).set_index(["tissue", "cell_type", "gene"])["pattern_r"]

# handling: correlation with the cell type's dissociation-stress profile (as in 330 b)
tech = pd.read_csv(RESULTS / "01_technical_audit/technical_audit.csv")
stress = tech[(tech["metric"] == "stress_log2cpm") & (tech["status"] == "primary")]
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
rstress = {}
for t in ["bm", "cochlea"]:
    pbk = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    samples = d0.drop_duplicates("sample_id").sort_values("zt")["sample_id"].tolist()
    st = stress[stress["tissue"] == t].pivot_table(index="cell_type", columns="sample_id", values="value")
    for ct, g in calls[calls["tissue"] == t].groupby("cell_type"):
        cols = [f"{ct}|{s}" for s in samples]
        if not set(cols) <= set(pbk.columns) or ct not in st.index:
            continue
        genes = pbk.index.intersection(g["gene"].unique())
        lib = pbk[cols].to_numpy(float).sum(0)
        l = np.log2(pbk.loc[genes, cols].to_numpy(float) / lib * 1e6 + 1)
        l -= l.mean(1, keepdims=True)
        s = st.loc[ct, samples].to_numpy(float)
        s = s - s.mean()
        den = np.sqrt((l ** 2).sum(1) * (s ** 2).sum())
        r = np.where(den > 0, (l * s).sum(1) / np.where(den > 0, den, 1), np.nan)
        rstress.update({(t, ct, gname): rv for gname, rv in zip(genes, r)})

# ---------------- flags
c = calls
c["peak_clock"] = phase.zt_to_clock(c["peak_zt"])
c["aligned_phase"] = phase.wrap24(c["peak_zt"] - c["tissue"].map(OFF))
c["core_clock"] = c["gene"].isin(CLOCK)
c["temperature"] = c["gene"].isin(HSF1) | (c["gene"].isin(COLD) & (c["aligned_phase"] < 12))
c["r_with_stress_profile"] = [rstress.get((r.tissue, r.cell_type, r.gene), np.nan) for r in c.itertuples()]
c["handling_linked"] = c["r_with_stress_profile"] > 0.8
c["composition_linked"] = c["foreign_marker_of"] != ""
c["gc_dependent_ext"] = c["gene"].isin(gcdep)
c["gc_driven"] = (c["gc_dependent_ext"] | c["gene"].isin(GCSET)) & (np.abs(phase.circ_diff(c["aligned_phase"], 13.0)) <= 5)
c["dd_n_organs"] = c["gene"].map(dd_n).fillna(0).astype(int)
c["dd_aligned_phase"] = c["gene"].map(dd_phase)
c["dd_phase_diff_h"] = phase.circ_diff(c["aligned_phase"], c["dd_aligned_phase"])
c["dd_support"] = (c["dd_n_organs"] >= 2) & (np.abs(c["dd_phase_diff_h"]) <= 4)
el = []
for r in c.itertuples():
    hit = [m.split("_")[0] for m, z in CLOCK_EL.items() if r.gene in mp.get(m, set()) and abs(phase.circ_diff(r.aligned_phase, z)) <= 4]
    el.append(";".join(sorted(set(hit))))
c["clock_elements_phase_consistent"] = el
c["clock_motif_phase"] = c["clock_elements_phase_consistent"] != ""
c["clock_controlled"] = c["dd_support"] | c["clock_motif_phase"]
c["clock_controlled_strength"] = np.select([c["dd_support"] & c["clock_motif_phase"], c["clock_controlled"]], ["strong", "moderate"], "")
c["pattern_r"] = [patt.get((r.tissue, r.cell_type, r.gene), np.nan) for r in c.itertuples()]
c["pattern_linked"] = c["pattern_r"] > 0.8
flags = ["core_clock", "temperature", "handling_linked", "composition_linked", "gc_driven", "clock_controlled", "pattern_linked"]
c["primary_driver"] = np.select([c[f] for f in flags], PREC[:-1], "unexplained")
c["n_driver_flags"] = c[flags].sum(1)
cols = ["tissue", "cell_type", "gene", "tier", "AveExpr", "ptt_fc", "peak_zt", "peak_clock", "aligned_phase", "primary_driver", "n_driver_flags",
        "core_clock", "temperature", "handling_linked", "r_with_stress_profile", "composition_linked", "foreign_marker_of", "foreign_marker_phase_match",
        "gc_driven", "gc_dependent_ext", "clock_controlled", "clock_controlled_strength", "dd_support", "dd_n_organs", "dd_aligned_phase",
        "dd_phase_diff_h", "clock_motif_phase", "clock_elements_phase_consistent", "pattern_linked", "pattern_r"]
write_csv(c[cols].sort_values(["tissue", "cell_type", "tier", "gene"]), MOD, "driver_calls.csv",
          "Driver evidence per rhythmic call (Tier 1/2; cochlear Tier C exploratory): flags and primary driver class (precedence order)",
          required=["tissue", "cell_type", "gene", "tier", "peak_zt", "peak_clock", "primary_driver"])

# ---------------- landscape
land = []
c["tier_set"] = np.where(c["tier"] == "tierC", "tierC (exploratory)", "tier1/2")
for keys, g in c.groupby(["tissue", "tier_set", "cell_type"]):
    n = len(g)
    for drv in PREC:
        h = g[g["primary_driver"] == drv]
        mu, R = phase.circ_mean(h["peak_zt"].to_numpy()) if len(h) else (np.nan, np.nan)
        bins = np.histogram(h["peak_zt"] % 24, bins=[0, 6, 12, 18, 24])[0]
        land.append({"tissue": keys[0], "tier_set": keys[1], "cell_type": keys[2], "n_calls": n, "driver": drv, "n": len(h),
                     "share": len(h) / n, "circ_mean_peak_zt": mu, "circ_mean_peak_clock": float(phase.zt_to_clock(mu)) if len(h) else np.nan,
                     "resultant_length": R, "n_zt0_6": bins[0], "n_zt6_12": bins[1], "n_zt12_18": bins[2], "n_zt18_24": bins[3],
                     "n_flagged_any": int(g[flags[PREC.index(drv)]].sum()) if drv in PREC[:-1] else int((g["n_driver_flags"] == 0).sum())})
for keys, g in c.groupby(["tissue", "tier_set"]):
    for drv in PREC:
        h = g[g["primary_driver"] == drv]
        mu, R = phase.circ_mean(h["peak_zt"].to_numpy()) if len(h) else (np.nan, np.nan)
        bins = np.histogram(h["peak_zt"] % 24, bins=[0, 6, 12, 18, 24])[0]
        land.append({"tissue": keys[0], "tier_set": keys[1], "cell_type": "ALL", "n_calls": len(g), "driver": drv, "n": len(h),
                     "share": len(h) / len(g), "circ_mean_peak_zt": mu, "circ_mean_peak_clock": float(phase.zt_to_clock(mu)) if len(h) else np.nan,
                     "resultant_length": R, "n_zt0_6": bins[0], "n_zt6_12": bins[1], "n_zt12_18": bins[2], "n_zt18_24": bins[3],
                     "n_flagged_any": int(g[flags[PREC.index(drv)]].sum()) if drv in PREC[:-1] else int((g["n_driver_flags"] == 0).sum())})
land = pd.DataFrame(land)
write_csv(land, MOD, "driver_landscape.csv",
          "Driver landscape per tissue × tier set × cell type: share of calls per primary driver, circular mean peak (ZT, clock), phase bins",
          required=["tissue", "tier_set", "cell_type", "driver", "n", "share", "circ_mean_peak_zt", "circ_mean_peak_clock"])
log("\n" + land[land["cell_type"] == "ALL"].pivot_table(index=["tissue", "tier_set"], columns="driver", values="share").round(3).to_string())


# ---------------- phase-permutation null for the phase-dependent flags (peaks shuffled within tissue × tier set × cell type)
def phase_flags(ap, d):
    dd = (d["dd_n_organs"].to_numpy() >= 2) & (np.abs(phase.circ_diff(ap, d["dd_aligned_phase"].to_numpy())) <= 4)
    mot = np.zeros(len(d), bool)
    for m, z in CLOCK_EL.items():
        mot |= d["gene"].isin(mp.get(m, set())).to_numpy() & (np.abs(phase.circ_diff(ap, z)) <= 4)
    gc = (d["gc_dependent_ext"] | d["gene"].isin(GCSET)).to_numpy() & (np.abs(phase.circ_diff(ap, 13.0)) <= 5)
    return {"dd_support": dd, "clock_motif_phase": mot, "clock_controlled": dd | mot, "gc_driven": gc}


rng = np.random.default_rng(4)
NPERM = 200
grp = c.groupby(["tissue", "tier_set", "cell_type"]).indices
nulls = {f: np.zeros((NPERM, len(c))) for f in ["dd_support", "clock_motif_phase", "clock_controlled", "gc_driven"]}
ap0 = c["aligned_phase"].to_numpy()
for i in range(NPERM):
    ap = ap0.copy()
    for idx in grp.values():
        ap[idx] = ap0[rng.permutation(idx)]
    for f, v in phase_flags(ap, c).items():
        nulls[f][i] = v
nrows = []
for keys, idx in list(grp.items()) + [((t, ts, "ALL"), np.where((c["tissue"] == t) & (c["tier_set"] == ts))[0])
                                      for t, ts in c[["tissue", "tier_set"]].drop_duplicates().itertuples(index=False)]:
    if len(idx) < 20:
        continue
    for f, nm in nulls.items():
        obs = float(c[f].to_numpy()[idx].mean())
        nd = nm[:, idx].mean(1)
        nrows.append({"tissue": keys[0], "tier_set": keys[1], "cell_type": keys[2], "flag": f, "n_calls": len(idx), "observed_share": obs,
                      "null_mean_share": float(nd.mean()), "null_q95_share": float(np.quantile(nd, 0.95)),
                      "obs_over_null": obs / nd.mean() if nd.mean() > 0 else np.nan, "p_perm": (1 + (nd >= obs).sum()) / (NPERM + 1)})
pn = pd.DataFrame(nrows)
write_csv(pn, MOD, "driver_phase_null.csv",
          "Phase-dependent driver flags vs a phase-permutation null (peaks shuffled within tissue × tier set × cell type, 200 draws)",
          required=["tissue", "tier_set", "cell_type", "flag", "observed_share", "null_mean_share", "p_perm"])
log("\n" + pn[pn["cell_type"] == "ALL"].round(4).to_string())

# ---------------- enrichment vs expression-matched non-rhythmic genes
def cmh(df, flag):
    tabs = []
    for _, s in df.groupby("dec", observed=True):
        t = np.array([[((s.call) & s[flag]).sum(), ((s.call) & ~s[flag]).sum()],
                      [((~s.call) & s[flag]).sum(), ((~s.call) & ~s[flag]).sum()]], float)
        if t.sum(1).min() > 0 and t.sum(0).min() > 0:
            tabs.append(t)
    if not tabs or sum(t[:, 0].sum() for t in tabs) < 10 or sum(t[0, 0] for t in tabs) < 3:
        return np.nan, np.nan
    st_ = StratifiedTable(tabs)
    return float(st_.oddsratio_pooled), float(st_.test_null_odds(correction=True).pvalue)


ev = {"dd_rhythmic_2plus_organs": dd_any, "gc_dependent_GSE88954": gcdep, "promoter_clock_element": CLOCK_EL_GENES,
      "promoter_HSE": mp.get("HSE_GAAnnTTCnnGAA", set()), "promoter_GRE": mp.get("GRE_GAACAnnnTGTTC", set())}
enr = []
for t, a in tested.items():
    cc = c[(c["tissue"] == t) & (c["tier"] != "tierC")][["cell_type", "gene"]].assign(call=True)
    d = a.merge(cc, on=["cell_type", "gene"], how="left")
    d["call"] = d["call"].notna()
    d = d[d["call"] | (d["p"] >= 0.2)]
    universe = {"dd_rhythmic_2plus_organs": set(e["gene"]), "gc_dependent_GSE88954": gc_tested}
    for scope, g in [("ALL", d)] + [(ct, s) for ct, s in d.groupby("cell_type") if s["call"].sum() >= 30]:
        g = g.copy()
        g["dec"] = g.groupby("cell_type")["AveExpr"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
        g["dec"] = g["cell_type"] + "_" + g["dec"].astype(str)
        for name, genes in ev.items():
            u = universe.get(name)
            gg = g[g["gene"].isin(u)] if u is not None else g
            gg = gg.assign(f=gg["gene"].isin(genes))
            orr, p = cmh(gg, "f")
            enr.append({"tissue": t, "cell_type": scope, "evidence": name, "n_calls": int(gg["call"].sum()),
                        "n_background": int((~gg["call"]).sum()), "frac_calls_with": float(gg.loc[gg["call"], "f"].mean()),
                        "frac_background_with": float(gg.loc[~gg["call"], "f"].mean()), "cmh_odds_ratio": orr, "cmh_p": p})
enr = pd.DataFrame(enr)
enr["cmh_q"] = np.nan
ok = enr["cmh_p"].notna()
from statsmodels.stats.multitest import multipletests  # noqa: E402
enr.loc[ok, "cmh_q"] = multipletests(enr.loc[ok, "cmh_p"], method="fdr_bh")[1]
write_csv(enr, MOD, "driver_enrichment.csv",
          "Enrichment of phase-free driver evidence among Tier-1/2 calls vs expression-matched non-rhythmic genes (CMH by AveExpr decile)",
          required=["tissue", "cell_type", "evidence", "cmh_odds_ratio", "cmh_p"])
log("\n" + enr[enr["cell_type"] == "ALL"].round(4).to_string())

# ---------------- diurnal vs circadian status of headline findings
reg = pd.read_csv(RESULTS / "r2_audit/positive_findings_register.csv").set_index("finding_id")
rows = []


def gene_status(fid, sel, label):
    m = (sel & (c["tier"] != "tierC")).to_numpy()
    g = c[m]
    rows.append({"finding_id": fid, "verdict": reg["verdict"].get(fid, ""), "evidence_unit": label, "n_calls": len(g),
                 "frac_dd_support": g["dd_support"].mean() if len(g) else np.nan,
                 "frac_dd_support_phase_null": float(nulls["dd_support"][:, m].mean()) if len(g) else np.nan,
                 "p_dd_support_phase_perm": float((1 + (nulls["dd_support"][:, m].mean(1) >= g["dd_support"].mean()).sum()) / (NPERM + 1)) if len(g) else np.nan,
                 "frac_clock_controlled": g["clock_controlled"].mean() if len(g) else np.nan,
                 "frac_clock_controlled_phase_null": float(nulls["clock_controlled"][:, m].mean()) if len(g) else np.nan,
                 "frac_core_clock": g["core_clock"].mean() if len(g) else np.nan,
                 "frac_systemic_temperature_gc": (g["temperature"] | g["gc_driven"]).mean() if len(g) else np.nan,
                 "frac_handling_or_composition": (g["handling_linked"] | g["composition_linked"]).mean() if len(g) else np.nan})


neut = c["cell_type"].str.startswith("Neutrophil")
gene_status("G01", (c["tissue"] == "bm") & neut, "BM neutrophil-state Tier-1/2 calls")
gene_status("G04", (c["tissue"] == "cochlea") & c["core_clock"], "cochlear core-clock calls")
gene_status("R05", c["gene"].isin(HSF1), "heat-shock (HSF1) calls, both tissues")
gene_status("G02", (c["tissue"] == "cochlea") & neut, "cochlear neutrophil-state calls")
gene_status("R01", (c["tissue"] == "bm") & neut, "BM neutrophil-state calls (same set as G01)")
st_ = pd.DataFrame(rows)
TERM = {"G01": "diurnal programme; clock-controlled share given", "G04": "circadian clock (core-clock genes are DD-rhythmic in every GSE54650 organ)",
        "R05": "diurnal, temperature-driven (body temperature is itself circadian; LD data cannot separate)",
        "G02": "diurnal (neutrophil timer; LD)", "R01": "diurnal, externally replicated (LD)"}
st_["recommended_term"] = st_["finding_id"].map(TERM)
for fid in [f for f in reg.index if f.startswith("A") and reg.loc[f, "verdict"] in ("robust", "supported")]:
    rows2 = {"finding_id": fid, "verdict": reg.loc[fid, "verdict"], "evidence_unit": "cell abundance (share)", "n_calls": np.nan,
             "recommended_term": "diurnal (LD 12:12); no public constant-darkness data for this compartment"}
    st_ = pd.concat([st_, pd.DataFrame([rows2])], ignore_index=True)
write_csv(st_, MOD, "diurnal_circadian_status.csv",
          "Headline findings: evidence for endogenous clock control (DD-rhythmic elsewhere, clock elements) vs systemic / technical drivers; recommended terminology",
          required=["finding_id", "evidence_unit", "recommended_term"])
log("\n" + st_.round(3).to_string())
