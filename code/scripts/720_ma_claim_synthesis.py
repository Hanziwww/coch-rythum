"""Round 8 (D53), step 3 — claim-level synthesis of the Ma 2025 × atlas integration. EXPLORATORY; rules fixed before running.

Inputs: 680 ex_claim_regrade (round-7 grades), 700 timing resolution, 710 tables. Confirmatory verdicts are only displayed.
Evidence lines (each with a timing qualifier):
  'clean'            bulk CMN − CMD (controls, inferred phases ~14 h apart, no noise) — indirect (deconvolution);
  'conditional'      scRNA night_noise − day_noise / − control (collection ≈ 12 h apart and exposure time / noise inseparable;
                     interpretable only if noise × time interaction is negligible);
  'timing-internal'  clock tracking (does not depend on absolute phase).
Line rules: multi-item (Spearman) lines support when ρ > 0 with one-sided p <= 0.10 and the sign holds under >= 5 of the
  timing assumptions (A0, T1, T3_30, offsets ± 2 / ± 4 h, exposure-time); oppose when ρ <= −0.30 with the sign holding under
  >= 5 assumptions; otherwise inconclusive. Single-item lines (one label / metric) support when observed and predicted
  differences share their sign with Welch p <= 0.20 and the predicted sign holds under >= 5 assumptions; oppose when the signs
  differ with p <= 0.20; otherwise inconclusive (sign agreement still shown). Clock tracking supports M7 when the unit's
  night − day phase difference is within ± 4 h of the structural one and |offset| <= 3 h in both groups; supports H4 (B-clock
  shift) when |offset| >= 3 h with the same sign in both groups, opposes it when |offset| <= 2 h in both.
Grade update from the round-7 grade: + 1 level (max E2) with >= 1 supporting clean / timing-internal line or >= 2 supporting
  conditional lines and no opposing line; − 1 level (min E0) with >= 1 opposing line and no supporting line; otherwise
  unchanged ('mixed' noted when both occur).
Post hoc conservative checks (added after the first run, D55; included whatever their direction): (a) marrow-content proxy
  (erythroblast + HSC / MPP share) and B shares relative to it; (b) M4 restricted to readiness programmes; (c) the clean bulk
  readiness-programme line (CMN − CMD) omitted from the first mapping. 'confound check' lines are reported, not graded.
Outputs (results/r8_ma_integration/): ma_claim_lines.csv, ma_claim_table.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r8_ma_integration"
W = phase.W
rd = lambda rel: pd.read_csv(RESULTS / rel) if (RESULTS / rel).exists() else pd.DataFrame()  # noqa: E731
rg7 = rd("r7_exploratory/ex_claim_regrade.csv")
sm = rd(f"{MOD}/ma_summary_tests.csv")
comp = rd(f"{MOD}/ma_composition_sc.csv")
bc = rd(f"{MOD}/ma_composition_bulk.csv")
st = rd(f"{MOD}/ma_states_sc.csv")
ck = rd(f"{MOD}/ma_clock_activity.csv")
sg = rd(f"{MOD}/ma_signalling.csv")
gph = rd("r8_ma_timing/ma_group_phase.csv")
pr = rd("r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")
off = rd("r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
EXPO = {"control": 3.0, "day_noise": 3.0, "night_noise": 15.0, "CMD": 3.0, "NMD": 3.0, "CMN": 15.0, "NMN": 15.0}
ASS = [("A0", "T2", 0, False), ("T1", "T1", 0, False), ("T3_30", "T3_30", 0, False), ("−4", "T2", -4, False), ("−2", "T2", -2, False),
       ("+2", "T2", 2, False), ("+4", "T2", 4, False), ("exposure", None, 0, True)]
lines = []


def ph(ds, est, d, expo):
    x = gph[(gph["dataset"].str.startswith(ds)) & (gph["estimator"] == est)].set_index("group") if est else None
    if expo:
        return {g: (v + O_C) % 24 for g, v in EXPO.items()}
    return {g: (x.loc[g, "internal_phase_h"] + d + O_C) % 24 for g in x.index} if x is not None and len(x) else {}


def share(l, t):
    r = pr.loc[l]
    return max(float(r["mesor_pct"] + r["amplitude_pct"] * np.cos(W * (t - r["peak_zt"]))), 1e-3)


def add(claim, name, timing, status, detail, n_rob=np.nan):
    lines.append({"claim": claim, "line": name, "timing": timing, "status": status, "detail": detail, "assumptions_with_same_sign": n_rob})


def multi(claim, family, contrast, scope, name, timing):
    x = sm[(sm["family"] == family) & (sm["contrast"] == contrast) & (sm["scope"] == scope)]
    a0 = x[x["assumption"].str.startswith("A0")]
    if not len(a0) or not np.isfinite(a0["spearman"].iloc[0]):
        add(claim, name, timing, "not testable", "too few items")
        return
    r, p = float(a0["spearman"].iloc[0]), float(a0["p_perm_one_sided"].iloc[0])
    same = int((np.sign(x["spearman"]) == np.sign(r)).sum())
    q = a0["q_BH_all_summary_tests"].iloc[0]
    status = ("supporting" if r > 0 and p <= 0.10 and same >= 5 else "opposing" if r <= -0.30 and same >= 5 else "inconclusive")
    add(claim, name, timing, status, f"ρ {r:+.2f}, one-sided p {p:.3f}, BH q {q:.2f}, n {int(a0['n_items'].iloc[0])}; sign stable in {same} / {len(x)} "
        f"assumptions; MC fraction ρ > 0 {a0['mc_frac_positive'].iloc[0]:.2f}", same)


def single_comp(claim, label, scale, contrast, ds, name, timing):
    a, b = contrast.split(" − ")
    if ds == "sc":
        x = comp[(comp["subset"] == "all cells") & (comp["scale"].str.startswith(scale)) & (comp["contrast"] == contrast) & (comp["label"] == label)]
        obs, p = (float(x["observed_log2_ratio"].iloc[0]), float(x["welch_p_logit"].iloc[0])) if len(x) else (np.nan, np.nan)
    else:
        x = bc[(bc["scale"].str.startswith(scale)) & (bc["contrast"] == contrast) & (bc["label"] == label)]
        obs, p = (float(x["observed_log2_ratio"].iloc[0]), float(x["welch_p"].iloc[0])) if len(x) else (np.nan, np.nan)
    if not np.isfinite(obs) or label not in pr.index:
        add(claim, name, timing, "not testable", "label not adequately represented / not validated")
        return
    signs = []
    for _, est, d, expo in ASS:
        if ds == "bulk" and est == "T3_30":
            continue
        P = ph("scRNA" if ds == "sc" else "bulk", est, d, expo)
        if a not in P or b not in P:
            continue
        sa, sb = share(label, P[a]), share(label, P[b])
        if scale.startswith("relative"):
            imm = [l for l in pr.index if l in IMMUNE]
            sa /= sum(share(l, P[a]) for l in imm)
            sb /= sum(share(l, P[b]) for l in imm)
        signs.append(np.sign(np.log2(sa / sb)))
    pred0 = signs[0]
    same = int(sum(s == pred0 for s in signs))
    if np.sign(obs) == pred0 and p <= 0.20 and same >= 5:
        status = "supporting"
    elif np.sign(obs) != pred0 and p <= 0.20:
        status = "opposing"
    else:
        status = "inconclusive"
    add(claim, name, timing, status, f"observed log2 {obs:+.2f} (p {p:.3f}); predicted sign {'+' if pred0 > 0 else '−'} (stable in {same} / {len(signs)}); "
        f"sign {'agrees' if np.sign(obs) == pred0 else 'disagrees'}", same)


def single_state(claim, unit, metric, contrast, name):
    a, b = contrast.split(" − ")
    x = st[(st["unit"] == unit) & (st["metric"] == metric) & (st["contrast"] == contrast)]
    if not len(x) or not np.isfinite(x["our_amplitude"].iloc[0]):
        add(claim, name, "conditional", "not testable", "no matching cochlear fit / metric")
        return
    r = x.iloc[0]
    signs = []
    for _, est, d, expo in ASS:
        P = ph("scRNA", est, d, expo)
        if a in P and b in P:
            signs.append(np.sign(r["our_amplitude"] * (np.cos(W * (P[a] - r["our_peak_zt"])) - np.cos(W * (P[b] - r["our_peak_zt"])))))
    pred0 = signs[0]
    same = int(sum(s == pred0 for s in signs))
    obs, p = r["observed_difference"], r["welch_p"]
    status = ("supporting" if np.sign(obs) == pred0 and p <= 0.20 and same >= 5 else "opposing" if np.sign(obs) != pred0 and p <= 0.20 else "inconclusive")
    add(claim, name, "conditional", status, f"observed {obs:+.3f} (p {p:.3f}, q {r['q_BH']:.2f}); predicted sign {'+' if pred0 > 0 else '−'} from our fit "
        f"(peak ZT{r['our_peak_zt']:.1f}, p_perm {r['our_p_perm']:.2f}; stable in {same} / {len(signs)})", same)


cls = rd("00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
IMMUNE = set(cls[cls.isin(["Myeloid", "Lymphoid", "Progenitor"])].index)
SC_ND, SC_NC, B_CC = "night_noise − day_noise", "night_noise − control", "CMN − CMD"
# ---------------- composition claims
for c in ["R08", "A05", "A04"]:
    pass
C_R08 = "R08 / headline 2: cochlear immune composition alternates with time of day"
multi(C_R08, "scRNA composition | absolute (share of all cells) | all cells", SC_ND, "all adequate labels", "scRNA absolute shares, night − day", "conditional")
multi(C_R08, "scRNA composition | relative (share of immune cells) | all cells", SC_ND, "all adequate labels", "scRNA immune-relative shares, night − day", "conditional")
multi(C_R08, "scRNA composition | absolute (share of all cells) | all cells", SC_NC, "all adequate labels", "scRNA absolute shares, night − control", "conditional")
multi(C_R08, "bulk composition | absolute (fraction of all cells)", B_CC, "validated immune labels", "bulk deconvolved absolute fractions, controls", "clean")
multi(C_R08, "bulk composition | relative (fraction of immune cells)", B_CC, "validated immune labels",
      "bulk deconvolved immune-relative fractions, controls (declared B1 follow-up)", "clean")
multi(C_R08, "bulk composition | absolute marker signature (indirect; 680 B1 definition, P excluded)", B_CC, "validated immune labels",
      "bulk absolute marker signatures, controls (indirect)", "clean")
for claim, lab in [("A05 band neutrophils peak in the rest phase (cochlea)", "Neutrophil (Band)"),
                   ("A04 classical monocytes peak in the rest phase (cochlea)", "Monocyte (Classical)"),
                   ("H1 naïve-B timing (cochlea)", "B Cell (Naïve)")]:
    for scale in ["absolute", "relative"]:
        single_comp(claim, lab, scale, SC_ND, "sc", f"scRNA {scale} share, night − day", "conditional")
        single_comp(claim, lab, scale, B_CC, "bulk", f"bulk deconvolved {scale} fraction, controls", "clean")
C_B = "headline 2 (B part): B lineage more abundant in the active phase (cochlea)"
for lab in ["B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]:
    single_comp(C_B, lab, "absolute", SC_ND, "sc", f"scRNA absolute share {lab}, night − day", "conditional")
multi(C_B, "bulk composition | absolute (fraction of all cells)", B_CC, "B lineage", "bulk deconvolved B fractions, controls", "clean")
C_M3 = "M3 granulopoiesis timing (cochlea; GMP / band shares and maturation)"
for lab in ["Granulocyte-Monocyte Progenitor", "Neutrophil (Band)"]:
    single_comp(C_M3, lab, "absolute", SC_ND, "sc", f"scRNA absolute share {lab}, night − day", "conditional")
single_comp(C_M3, "Granulocyte-Monocyte Progenitor", "absolute", B_CC, "bulk", "bulk deconvolved GMP fraction, controls", "clean")
# ---------------- state claims
for claim, unit, metric in [("M1 maturation / ageing timing (cochlea)", "pooled neutrophils", "maturation (Xie)"),
                            ("M2 cochlear circulating-like state (blood timer)", "pooled neutrophils", "blood timer ZT13−ZT5 (GSE86619)"),
                            ("M5 monocyte → macrophage (cochlea)", "monocytes + macrophages", "mono→mac (curated)"),
                            ("M6 retention / release axis (cochlea)", "pooled neutrophils", "cxcr4_cxcr2"),
                            ("M8 cochlear macrophage identity", "Macrophage", "resident macrophage (cochlear)")]:
    for con in [SC_ND, SC_NC]:
        single_state(claim, unit, metric, con, f"scRNA {unit} {metric}, {con}")
for claim, unit, metric in [("M2 cochlear circulating-like state (blood timer)", "pooled neutrophils", "tissue vs marrow (NeuMAP / GSE200145)"),
                            ("M1 maturation / ageing timing (cochlea)", "pooled neutrophils", "old vs young BM (NeuMAP / GSE200145)")]:
    single_state(claim, unit, metric, SC_ND, f"scRNA {unit} {metric}, {SC_ND}")
C_M4 = "M4 functional readiness (cochlea)"
multi(C_M4, "scRNA states | myeloid", SC_ND, "all items with a fit", "scRNA myeloid states (axes + programmes), night − day", "conditional")
multi(C_M4, "scRNA states | myeloid", SC_NC, "all items with a fit", "scRNA myeloid states (axes + programmes), night − control", "conditional")
x = sg[sg["sample"] == "summary"]
for r in x.itertuples():
    if r.condition == SC_ND:
        add("M6 retention / release axis (cochlea)", "CXCL12 → CXCR4 proxy, night − day", "conditional",
            "inconclusive" if r.welch_p > 0.20 else "descriptive", f"log2 ratio {r.cxcl12_cxcr4_proxy:+.2f} (p {r.welch_p:.3f}); no prediction (64 representative "
            "interaction not rhythmic)")
C_BP = "B-lineage programme rhythms (round-5 exploratory)"
multi(C_BP, "scRNA states | B", SC_ND, "items whose own fit has p_perm <= 0.20", "scRNA B programmes (own fit p <= 0.20), night − day", "conditional")
multi(C_BP, "scRNA states | B", SC_ND, "all items with a fit", "scRNA B programmes (all), night − day", "conditional")
# ---------------- clock tracking
for unit in ["pooled neutrophils", "monocytes + macrophages", "neutrophils + monocytes (pooled)"]:
    r = ck[ck["unit"] == unit]
    if len(r):
        r = r.iloc[0]
        ok = bool(r["tracking_within_4h"]) and abs(r["offset_vs_structural_day_h"]) <= 3 and abs(r["offset_vs_structural_night_h"]) <= 3
        add("M7 myeloid clocks (cochlea)", f"clock tracking: {unit}", "timing-internal", "supporting" if ok else "inconclusive",
            f"night − day {r['night_minus_day_h']:+.1f} h vs structural {r['structural_night_minus_day_h']:+.1f} h; offset vs structural "
            f"{r['offset_vs_structural_day_h']:+.1f} (day) / {r['offset_vs_structural_night_h']:+.1f} h (night)")
r = ck[ck["unit"] == "B lineage (pooled)"]
if len(r):
    r = r.iloc[0]
    od, on = r["offset_vs_structural_day_h"], r["offset_vs_structural_night_h"]
    status = ("supporting" if abs(od) >= 3 and abs(on) >= 3 and np.sign(od) == np.sign(on) else
              "opposing" if abs(od) <= 2 and abs(on) <= 2 else "inconclusive")
    add("H4 B-clock shift (cochlea)", "clock tracking: B lineage (pooled)", "timing-internal", status,
        f"night − day {r['night_minus_day_h']:+.1f} h vs structural {r['structural_night_minus_day_h']:+.1f} h; offsets {od:+.1f} / {on:+.1f} h (not a "
        "stable shift)" if status == "inconclusive" else f"offsets {od:+.1f} / {on:+.1f} h")
add("M9 noise-exposure readiness vs time of day (cochlea)", "scRNA: no night-collected control → readiness × exposure time not separable",
    "conditional", "not testable", "night-noise libraries differ from controls and day-noise libraries in collection phase (700); bulk B2 (680) remains the "
    "time-matched test (rank ρ +0.40, p 0.11; most readiness programmes respond less after night noise)")
# ---------------- post hoc conservative checks (added after the first 720 run, D55; included whatever their direction)
# (a) marrow content: Ma's dissection includes the modiolus, so libraries may differ in marrow content (as our cochlear sample
#     pattern). Proxy = share of erythroblasts + HSC / MPP (marrow-resident, non-B); B-lineage shares relative to the proxy.
cbs = rd("r3_ma2025/composition_by_sample.csv").query("subset == 'all cells'")
sh = cbs.pivot_table(index="sample", columns="label", values="share", fill_value=0)
grp = cbs.drop_duplicates("sample").set_index("sample")["condition"]
proxy = sh[[c for c in ["Erythroblast", "HSC / MPP"] if c in sh.columns]].sum(1)
nd = lambda v: (np.log2(v[grp == "night_noise"].mean() / v[grp == "day_noise"].mean()),  # noqa: E731
                float(__import__("scipy").stats.ttest_ind(np.log(v[grp == "night_noise"]), np.log(v[grp == "day_noise"]), equal_var=False).pvalue))
d0, p0 = nd(proxy)
add("headline 2 (B part): B lineage more abundant in the active phase (cochlea)", "post hoc: marrow-content proxy (erythroblast + HSC / MPP share), night − day",
    "conditional", "confound check", f"log2 {d0:+.2f} (p {p0:.3f}); per library: " + "; ".join(f"{k} {v:.3f}" for k, v in proxy.items()))
for claim in ["headline 2 (B part): B lineage more abundant in the active phase (cochlea)", "H1 naïve-B timing (cochlea)"]:
    for labB in (["B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"] if claim.startswith("headline") else ["B Cell (Naïve)"]):
        d1, p1 = nd(sh[labB] / proxy)
        add(claim, f"post hoc: {labB} share relative to the marrow proxy, night − day", "conditional",
            "supporting" if d1 > 0 and p1 <= 0.20 else "opposing" if d1 < 0 and p1 <= 0.20 else "inconclusive", f"log2 {d1:+.2f} (p {p1:.3f}); predicted +")
# (b) M4 restricted to readiness programmes (the first line mixed axes and programmes)
from scipy import stats as _st  # noqa: E402
rng_ = np.random.default_rng(720)
for con in [SC_ND, SC_NC]:
    a, b_ = con.split(" − ")
    x = st[(st["contrast"] == con) & (st["kind"] == "programme") & (st["lineage"] == "myeloid")].dropna(subset=["our_amplitude", "observed_difference"])
    rs = []
    for _, est, d, expo in ASS:
        P = ph("scRNA", est, d, expo)
        pred = x["our_amplitude"] * (np.cos(W * (P[a] - x["our_peak_zt"])) - np.cos(W * (P[b_] - x["our_peak_zt"])))
        rs.append(_st.spearmanr(pred, x["observed_difference"])[0])
    P = ph("scRNA", "T2", 0, False)
    pred = (x["our_amplitude"] * (np.cos(W * (P[a] - x["our_peak_zt"])) - np.cos(W * (P[b_] - x["our_peak_zt"])))).to_numpy()
    r0 = rs[0]
    null = np.array([_st.spearmanr(pred, rng_.permutation(x["observed_difference"].to_numpy()))[0] for _ in range(5000)])
    p_ = float((1 + (null >= r0).sum()) / 5001)
    same = int(sum(np.sign(v) == np.sign(r0) for v in rs))
    add("M4 functional readiness (cochlea)", f"post hoc: readiness programmes only (myeloid units), {con}", "conditional",
        "supporting" if r0 > 0 and p_ <= 0.10 and same >= 5 else "opposing" if r0 <= -0.30 and same >= 5 else "inconclusive",
        f"ρ {r0:+.2f}, one-sided p {p_:.3f}, n {len(x)}; sign stable in {same} / {len(rs)} assumptions", same)
# (c) clean bulk line for M4: baseline CMN − CMD programme mean gene t (phase genes excluded) vs our pooled-myeloid prediction
import yaml  # noqa: E402
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
hm = {}
for line in open(Path(RESULTS).parent / "external/resources/msigdb_mh_hallmark.gmt"):
    xx = line.rstrip("\n").split("\t")
    hm[xx[0]] = xx[2:]
PROG = {**{f"myeloid:{k}": v for k, v in SETS["myeloid_function"].items()},
        "hallmark:GLYCOLYSIS": hm["HALLMARK_GLYCOLYSIS"], "hallmark:OXIDATIVE_PHOSPHORYLATION": hm["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]}
PGS = set(phase.CANONICAL_CLOCK) | {"Arntl", "Clock", "Bhlhe40", "Bhlhe41"}
de_b = rd("r3_ma2025/bulk_de.csv")
dd = de_b[(de_b["contrast"] == "baseline_night_minus_day") & ~de_b["gene"].isin(PGS)].set_index("gene")["t"]
fn = rd("r6_myeloid/m_function_rhythm.csv")
fn = fn[(fn["tissue"] == "cochlea") & (fn["version"] == "primary") & (fn["kind"] == "uncentred") & (fn["unit"] == "neutrophils + monocytes (pooled)")].set_index("programme")
obs = pd.Series({k: dd[[g for g in v if g in dd.index]].mean() for k, v in PROG.items() if k in fn.index})
rs = []
for _, est, d, expo in ASS:
    if est == "T3_30":
        continue
    P = ph("bulk", est, d, expo)
    pred = fn.loc[obs.index, "amplitude"] * (np.cos(W * (P["CMN"] - fn.loc[obs.index, "peak_zt"])) - np.cos(W * (P["CMD"] - fn.loc[obs.index, "peak_zt"])))
    rs.append(_st.spearmanr(pred, obs)[0])
P = ph("bulk", "T2", 0, False)
pred = (fn.loc[obs.index, "amplitude"] * (np.cos(W * (P["CMN"] - fn.loc[obs.index, "peak_zt"])) - np.cos(W * (P["CMD"] - fn.loc[obs.index, "peak_zt"])))).to_numpy()
r0 = rs[0]
null = np.array([_st.spearmanr(pred, rng_.permutation(obs.to_numpy()))[0] for _ in range(5000)])
p_ = float((1 + (null >= r0).sum()) / 5001)
same = int(sum(np.sign(v) == np.sign(r0) for v in rs))
add("M4 functional readiness (cochlea)", "post hoc: bulk readiness programmes, controls CMN − CMD (indirect; whole tissue)", "clean",
    "supporting" if r0 > 0 and p_ <= 0.10 and same >= 5 else "opposing" if r0 <= -0.30 and same >= 5 else "inconclusive",
    f"ρ {r0:+.2f}, one-sided p {p_:.3f}, n {len(obs)}; sign stable in {same} / {len(rs)} assumptions", same)
L = pd.DataFrame(lines)

# ---------------- claim table
MAP7 = {"R08": ("R08", "cochlea"), "M1": ("M1", "cochlea"), "M2": ("M2", "paired"), "M3": ("M3", "cochlea"), "M4": ("M4", "cochlea"),
        "M5": ("M5", "cochlea"), "M6": ("M6", "cochlea"), "M7": ("M7", "cochlea"), "M8": ("M8", "cochlea"), "M9": ("M9", "cochlea"),
        "H1": ("H1", "cochlea"), "H4": ("H4", "cochlea")}
LEV = ["E0 no exploratory support", "E1 qualified exploratory support", "E2 exploratory support"]
rows = []
for claim, g in L.groupby("claim", sort=False):
    key = claim.split(" ")[0].split("/")[0].strip()
    g7 = None
    if key in MAP7 and len(rg7):
        x = rg7[(rg7["claim"].str.split(" ").str[0] == MAP7[key][0]) & (rg7["tissue"] == MAP7[key][1])]
        g7 = x["exploratory_grade"].iloc[0] if len(x) else None
        conf = x["confirmatory_decision (unchanged)"].iloc[0] if len(x) else ""
    else:
        conf = "register finding (A05 / A04 robust / supported; headline 2 descriptive)" if key in ("A05", "A04", "headline", "R08") else "not a primary hypothesis"
    sup = g[g["status"] == "supporting"]
    opp = g[g["status"] == "opposing"]
    clean_sup = int(sup["timing"].isin(["clean", "timing-internal"]).sum())
    cond_sup = int((sup["timing"] == "conditional").sum())
    up = (clean_sup >= 1 or cond_sup >= 2) and len(opp) == 0
    down = len(opp) >= 1 and len(sup) == 0
    if g7 in LEV:
        i = LEV.index(g7)
        new = LEV[min(i + 1, 2)] if up else LEV[max(i - 1, 0)] if down else g7
    else:
        new = ("Ma-supported (exploratory)" if up else "Ma-opposed (exploratory)" if down else "Ma-inconclusive / mixed")
    timing = "; ".join(sorted(set(g["timing"])))
    rows.append({"claim": claim, "confirmatory_status (unchanged)": conf, "round7_grade": g7 or "n/a", "ma_lines_supporting": len(sup),
                 "ma_lines_opposing": len(opp), "ma_lines_inconclusive": int((g["status"] == "inconclusive").sum()),
                 "ma_lines_not_testable": int((g["status"] == "not testable").sum()), "timing_basis": timing,
                 "updated_exploratory_grade": new, "mixed": bool(len(sup) and len(opp)),
                 "supporting_lines": " | ".join(f"{r.line} [{r.timing}]: {r.detail}" for r in sup.itertuples()),
                 "opposing_lines": " | ".join(f"{r.line} [{r.timing}]: {r.detail}" for r in opp.itertuples()),
                 "clean_timing_evidence": " | ".join(f"{r.line}: {r.status} ({r.detail.split(';')[0]})" for r in g[g["timing"] == "clean"].itertuples())
                 or "none (no clean-timing line available)",
                 "upgrade_basis": ("clean / timing-internal line" if clean_sup else "conditional lines only (exposure and collection inseparable)" if cond_sup >= 2
                                   else "") if up else "",
                 "robustness (assumptions with stable sign, supporting lines)": ", ".join(str(int(v)) for v in sup["assumptions_with_same_sign"].dropna())})
T = pd.DataFrame(rows)
# defensible manuscript wording (curated from these results, 2026-10-14; EXPLORATORY)
WORD = {
    "R08": "In an independent cohort (Ma et al. 2025), immune composition differed between cochleae collected ~12 h apart in the direction predicted by "
           "our cochlear abundance rhythms (single-cell shares, ρ = 0.75 absolute / 0.69 immune-relative). Because collection time and noise-exposure time "
           "coincide in that design, this holds only if noise × time interactions are small. In time-matched bulk controls, deconvolved absolute fractions "
           "agreed (ρ = 0.44, p = 0.08), immune-relative fractions did not reproduce it (ρ = 0.16).",
    "A05": "Band neutrophils formed a smaller share of immune cells in night-collected libraries (log2 −1.1, p = 0.13), as predicted by their rest-phase peak; "
           "their absolute share did not differ and bulk deconvolution could not resolve them — inconclusive.",
    "A04": "Classical monocytes formed a smaller share of immune cells in night-collected libraries (log2 −1.1, p = 0.02), as predicted; absolute shares and "
           "bulk did not differ, and relative shares are lowered by the large night increase of B cells (compositional closure) — inconclusive.",
    "H1": "Naïve B cells were ~6-fold more abundant in night-collected cochleae of an independent cohort (log2 +2.6, p = 0.001; +2.3 relative to a "
          "marrow-content proxy), as our late-night peak predicts (exploratory; conditional on noise × time); deconvolved time-matched bulk did not show it "
          "(sign opposite, n.s.).",
    "headline": "B-lineage cells (transitional, naïve, mature) were 3- to 10-fold more abundant in night-collected cochleae of an independent cohort, "
                "including relative to marrow-resident cells, matching the active-phase B-lineage peak of the atlas (exploratory; conditional); bulk "
                "deconvolution did not reproduce it (ρ = 0).",
    "M3": "No independent support: GMP and band-neutrophil shares did not differ between night- and day-collected libraries.",
    "M1": "No independent support for maturation timing (the maturation and ageing axes did not differ in the predicted direction).",
    "M2": "Cochlear neutrophils of night-collected libraries carried a higher blood ZT13 ('fresh') programme than time-matched-to-day controls (p = 0.04), "
          "consistent with the cochlear fresh-at-night timing; the night- vs day-noise difference pointed the same way (n.s.).",
    "M5": "No independent support for monocyte → macrophage timing.",
    "M6": "Neutrophil Cxcr4 / Cxcr2 was lower in night-collected libraries than in controls (p = 0.04), in line with the light-phase peak of this ratio in "
          "our cochlea; the pre-specified direction (co-phase with the timer) remains unmet and the CXCL12 → CXCR4 proxy showed no time dependence.",
    "M8": "No independent support for a time-of-day macrophage identity programme.",
    "M4": "Myeloid readiness programmes differed between night- and day-collected cochleae of an independent cohort in the direction of our rhythms "
          "(programmes only: ρ = 0.19 night − day, 0.47 night − control; stable across timing assumptions) — exploratory and conditional on noise × time; "
          "whole-tissue bulk at clean timing did not reproduce it (ρ = −0.20, n.s.).",
    "B-lineage": "B-lineage programmes that are rhythmic in our cochlea changed in the opposite direction in the independent cohort (ρ = −0.70 over nine "
                 "programmes, stable across timing assumptions): contradicted, not supported.",
    "M7": "In an independent cohort, the clocks of pooled cochlear neutrophils and monocytes / macrophages shifted with the ~11.5-h collection difference "
          "and stayed within 0–2 h of structural cells, supporting myeloid clocks entrained in phase with the tissue (exploratory; timing-internal; an "
          "ambient-RNA contribution to immune clock profiles cannot be excluded).",
    "H4": "The B-lineage clock did not follow the collection shift (+0.3 h vs −11.5 h in structural cells) and its offsets were unstable — a distinct B "
          "clock state is possible, a stable B-clock shift is not shown.",
    "M9": "Not testable in the single-cell data (no night-collected controls); the time-matched bulk test remains as in round 7 (rank ρ +0.40, p = 0.11, "
          "with most readiness programmes responding less after night noise).",
}
T["defensible_wording"] = [next((v for k, v in WORD.items() if c.startswith(k)), "") for c in T["claim"]]
LIM = pd.DataFrame([
    {"topic": "collection times", "status": "resolved (inferred)", "detail": "scRNA: controls and day-noise libraries share a phase (Δ −0.8 to +0.1 h); "
     "night-noise libraries ~11.5 h apart (exceeds the 5.5-h noise-shift bound by 2.7–4.6 h). Bulk: CMD ~ZT2–3, CMN ~ZT16–17.5 (internal time)."},
    {"topic": "annotation errors", "status": "resolved", "detail": "no sample met the correction rule (noise status and phase both contradicting)."},
    {"topic": "noise effect on the clock", "status": "partly resolved", "detail": "bulk: ~3-h shifts 1 d after noise (day +3.0 h, night −2.75 h) and 9 / 18 "
     "clock genes changed — collection drift vs biological shift not separable within the bulk design."},
    {"topic": "exposure vs collection time (scRNA)", "status": "unresolved by design", "detail": "night-noise libraries differ from all others in both; "
     "night-vs-day contrasts are interpretable only if noise × time interactions are small; no night-collected control exists."},
    {"topic": "absolute phase", "status": "uncertain ± 2–3 h", "detail": "phase estimates are relative to canonical / our clocks; CBA/CaJ vs our strain; "
     "results stable to +2 / +4 h offsets but the composition match reverses at −4 h (and is lost at −2 h)."},
    {"topic": "bulk deconvolution", "status": "limited", "detail": "validated for macrophages, classical monocytes, GMP and B labels; neutrophil states, "
     "non-classical monocytes and mast cells not recoverable (validation r < 0.5)."},
    {"topic": "bulk as evidence", "status": "indirect", "detail": "whole-tissue signatures / fractions; immune-relative fractions did not reproduce the "
     "single-cell composition match (ρ 0.16)."},
    {"topic": "gene reuse", "status": "controlled", "detail": "18 clock genes used for phase were excluded from every validation gene set and marker list."},
    {"topic": "sample size", "status": "limit", "detail": "3–4 pooled libraries per group (scRNA), n = 3 (bulk); per-item tests rarely survive BH."},
    {"topic": "ambient RNA in sorted immune cells", "status": "not modelled (project rule)", "detail": "immune clock profiles could partly reflect structural "
     "transcripts; affects M7 / H4 lines."}])
write_csv(LIM, MOD, "ma_resolution_and_limits.csv", "EXPLORATORY: what the round-8 timing / integration work resolved and the remaining limits",
          required=["topic", "status", "detail"])
write_csv(L, MOD, "ma_claim_lines.csv", "EXPLORATORY: Ma 2025 evidence lines per original claim with timing qualifier, status (supporting / opposing / "
          "inconclusive / not testable) and robustness across timing assumptions", required=["claim", "line", "timing", "status", "detail"])
write_csv(T, MOD, "ma_claim_table.csv", "EXPLORATORY: claim-level table — confirmatory status (unchanged), round-7 grade, Ma lines for / against, timing "
          "basis and the updated exploratory grade (rules in 720)", required=["claim", "updated_exploratory_grade"])
pd.set_option("display.width", 250)
log("\n" + L[["claim", "line", "timing", "status", "detail"]].to_string(index=False, max_colwidth=75))
log("\n" + T[["claim", "round7_grade", "ma_lines_supporting", "ma_lines_opposing", "ma_lines_inconclusive", "updated_exploratory_grade", "mixed"]]
    .to_string(index=False, max_colwidth=60))
