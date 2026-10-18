"""Round 7 — EXPLORATORY reassessment of the pre-specified claims (rounds 5, 6 and Ma 2025) with broader, biologically
motivated evidence and relaxed thresholds. Confirmatory verdicts (D34, D45 / D48, positive_findings_register) are not
changed; every output here is labelled exploratory. Rules below were fixed before this script was first run.

Part B — new exploratory analyses
 B1 time-matched replication of the cochlear composition rhythm in Ma 2025 bulk (CRA019827): cell-type signatures =
    our top markers (00_audit/label_markers: log2FC >= 1, adj. p < 0.05, detected in < 30 % of other cells), scored per
    sample as the mean gene z of log2 CPM within each time pair; observed night − day = CMN − CMD (controls) and NMN − NMD
    (noise). Predicted = log2 of our fitted cochlear share (r2_figure abundance_pct_rhythm) at each group's clock-inferred
    time (314 bulk_sampling_phase, our cochlear clock fits). Spearman across immune labels (and all labels); one-sided
    permutation p (10,000). Signatures mix abundance and per-cell expression (caveat).
 B2 time-of-day dependence of the noise response at pathway level (time-matched; counterpart of M9): for the M4 / M9
    readiness programmes, the mean gene t of the bulk noise × time interaction ((NMN − CMN) − (NMD − CMD); 314), gene-set
    permutation p (10,000); Spearman across programmes with our predicted ZT15 − ZT3 readiness difference (645,
    cochlea, pooled neutrophils + monocytes), one-sided permutation p. Whole-tissue bulk dilutes myeloid signal (caveat).
 B3 pathway-level phase coherence in the atlas: for each readiness programme × myeloid cell type (four neutrophil labels,
    classical monocytes; both tissues) and each B programme × B stage, member genes with gene-level cosinor p <= 0.20
    (33 tables) → Rayleigh z of their peak phases; null = 1,000 random gene sets of the same size from the genes tested
    in that cell type → p_perm. Tissue-level line for M4: number of coherent (p_perm <= 0.05) programme × type pairs vs
    the 5 % expected (one-sided binomial). Overlapping cell types are not independent (caveat).
 B4 replication lines for the timer split (M2 / G02) compiled from existing results: external cochlea (GSE88954),
    external marrow with / without neutrophils (GSE198770), human blood (GSE136632), within-tissue specificity (676),
    Ballesteros blood (640).

Part A — uniform re-grading of each claim unit (round 5: 12 tests; round 6: M1–M9 by tissue; R08)
 A1 primary trend: exact p of the pre-specified primary statistic (continuous; thresholds 0.10 / 0.20).
 A2 pre-specified direction / window (n/a where none).
 A3 supporting lines: strict agreeing methods of that claim unit + Part-B lines mapped to it (B1 label signs: H1 cochlea
    ← naïve B, M3 cochlea ← GMP + band, M5 cochlea ← classical monocytes; B1 immune ρ → R08; B2 → M9; B3 → M4; B4 → M2;
    640 within-sample-centred check → M8). A Part-B line counts when its direction matches the claim and its
    exploratory p <= 0.10 (label-sign lines: sign match only).
 A4 confounds: primary statistic tracks the sample-level pattern (|r| >= 0.9 with 210 PC1); Ma 2025 scRNA collection
    time (G54). A confound is resolved when a confound-robust line passes (M2: 676 specificity rank test; M8: centred
    check; M9 / R08: time-matched bulk lines B2 / B1).
 Grades: E2 'exploratory support' = A1 p <= 0.10, A2 met or n/a, >= 1 supporting line, no unresolved confound.
         E1 'qualified exploratory support' = (A1 p <= 0.20, A2 met or n/a, >= 1 supporting line) or (>= 2 supporting
             lines with A2 not violated) or E2 criteria with an unresolved confound.
         E0 'no exploratory support' otherwise.
Outputs (results/r7_exploratory/): ex_bulk_composition_replication.csv, ex_bulk_noise_timeofday_programmes.csv,
ex_programme_phase_coherence.csv, ex_replication_lines.csv, ex_claim_regrade.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r7_exploratory"
W = phase.W
rng = np.random.default_rng(680)
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
hm = {}
for line in open(EXTERNAL / "resources/msigdb_mh_hallmark.gmt"):
    x = line.rstrip("\n").split("\t")
    hm[x[0]] = x[2:]
PROG = {**{f"myeloid:{k}": v for k, v in SETS["myeloid_function"].items()},
        "hallmark:GLYCOLYSIS": hm["HALLMARK_GLYCOLYSIS"], "hallmark:OXIDATIVE_PHOSPHORYLATION": hm["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]}
BPROG = {f"b_cell:{k}": v for k, v in SETS["b_cell"].items()}
rd = lambda rel: pd.read_csv(RESULTS / rel) if (RESULTS / rel).exists() else pd.DataFrame()  # noqa: E731


def perm_spear(x, y, n=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (n + 1))


# ================================================================ B1 bulk composition replication (time-matched)
cls = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
IMMUNE = set(cls[cls.isin(["Myeloid", "Lymphoid", "Progenitor"])].index)
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 1) & (mk["pvals_adj"] < 0.05) & (mk["pct_nz_reference"] < 0.30)]
cnt = pd.read_csv("external/ma2025/CRA019827_bulk_counts_unstranded.tsv", sep="\t", index_col=0)
cpm = cnt / cnt.sum(0) * 1e6
L = np.log2(cpm[(cpm > 1).sum(1) >= 3] + 1)
ph = rd("r3_ma2025/bulk_sampling_phase.csv")
tg = ph[ph["reference"].str.startswith("our")].groupby("group")["group_mean_h"].first()
pr = rd("r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")


def share(ct, zt):
    r = pr.loc[ct]
    return float(r["mesor_pct"] + r["amplitude_pct"] * np.cos(W * (zt - r["peak_zt"])))


b1 = []
for pair, (dg, ng) in {"controls (CMN − CMD)": ("CMD", "CMN"), "noise (NMN − NMD)": ("NMD", "NMN")}.items():
    cols_d = [c for c in L.columns if c.startswith(dg)]
    cols_n = [c for c in L.columns if c.startswith(ng)]
    S = L[cols_d + cols_n]
    Z = S.sub(S.mean(1), axis=0).div(S.std(1).replace(0, np.nan), axis=0)
    for ct, g in mk.groupby("cell_type"):
        genes = [x for x in g["names"] if x in Z.index]
        if len(genes) < 5 or ct not in pr.index:
            continue
        sc_ = Z.loc[genes].mean(0)
        obs = float(sc_[cols_n].mean() - sc_[cols_d].mean())
        t_, p_ = stats.ttest_ind(sc_[cols_n], sc_[cols_d], equal_var=False)
        pred = float(np.log2(max(share(ct, tg[ng]), 1e-3) / max(share(ct, tg[dg]), 1e-3)))
        b1.append({"pair": pair, "label": ct, "immune": ct in IMMUNE, "n_markers": len(genes), "observed_signature_night_minus_day": obs,
                   "welch_p": float(p_), "predicted_log2_share_ratio": pred, "day_time_h": float(tg[dg]), "night_time_h": float(tg[ng]),
                   "sign_match": bool(np.sign(obs) == np.sign(pred))})
b1 = pd.DataFrame(b1)
b1s = []
for pair, g in b1.groupby("pair"):
    for scope, x in [("immune labels", g[g["immune"]]), ("all labels", g)]:
        r, p = perm_spear(x["predicted_log2_share_ratio"], x["observed_signature_night_minus_day"])
        b1s.append({"pair": pair, "scope": scope, "n_labels": len(x), "spearman": r, "p_perm_one_sided": p, "sign_agreement": float(x["sign_match"].mean())})
b1s = pd.DataFrame(b1s)
log("\n" + b1s.round(3).to_string(index=False))

# ================================================================ B2 noise × time at pathway level (time-matched)
de = rd("r3_ma2025/bulk_de.csv")
fnr = rd("r6_myeloid/m_function_rhythm.csv")
fr = fnr[(fnr["tissue"] == "cochlea") & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred") &
         (fnr["unit"] == "neutrophils + monocytes (pooled)")].set_index("programme")
b2 = []
for con in ["noise_x_time_interaction", "noise_effect_day_group", "noise_effect_night_group", "baseline_night_minus_day"]:
    d = de[de["contrast"] == con].set_index("gene")
    tv = d["t"].dropna()
    allg = tv.index.to_numpy()
    for nm, genes in {**PROG, "curated:heat_shock": ["Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"]}.items():
        g = [x for x in genes if x in tv.index]
        if len(g) < 3:
            continue
        m = float(tv[g].mean())
        null = np.array([tv.to_numpy()[rng.choice(len(allg), len(g), replace=False)].mean() for _ in range(10000)])
        rr = fr.loc[nm] if nm in fr.index else None
        b2.append({"contrast": con, "programme": nm, "n_genes": len(g), "mean_t": m, "mean_log2FC": float(d.loc[g, "log2FC"].mean()),
                   "p_perm_genes_two_sided": float((1 + (np.abs(null) >= abs(m)).sum()) / 10001),
                   "our_predicted_zt15_minus_zt3": float(rr["amplitude"] * (np.cos(W * (15 - rr["peak_zt"])) - np.cos(W * (3 - rr["peak_zt"]))))
                   if rr is not None else np.nan})
b2 = pd.DataFrame(b2)
b2s = []
for con, g in b2.dropna(subset=["our_predicted_zt15_minus_zt3"]).groupby("contrast"):
    r, p = perm_spear(g["our_predicted_zt15_minus_zt3"], g["mean_t"])
    b2s.append({"contrast": con, "n_programmes": len(g), "spearman_with_our_zt15_minus_zt3": r, "p_perm_one_sided": p})
b2s = pd.DataFrame(b2s)
log("\n" + b2s.round(3).to_string(index=False))

# ================================================================ B3 programme phase coherence (atlas)
b3 = []
for t in ["bm", "cochlea"]:
    g = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "p", "peak_zt"])
    for family, sets, types in [("readiness (M4)", PROG, NEU + ["Monocyte (Classical)"]), ("B programmes", BPROG, BST)]:
        for ct in types:
            x = g[g["group"] == ct].dropna(subset=["p", "peak_zt"]).drop_duplicates("gene").set_index("gene")
            if not len(x):
                continue
            pv, pk = x["p"].to_numpy(), x["peak_zt"].to_numpy()
            idx = {gg: i for i, gg in enumerate(x.index)}
            for nm, genes in sets.items():
                mem = np.array([idx[gg] for gg in genes if gg in idx])
                if len(mem) < 4:
                    continue

                def rz(ii):
                    sel = ii[pv[ii] <= 0.20]
                    if len(sel) < 3:
                        return 0.0, len(sel), np.nan
                    mu, R = phase.circ_mean(pk[sel])
                    return len(sel) * R ** 2, len(sel), mu
                z0, nr, mu = rz(mem)
                null = np.array([rz(rng.choice(len(pv), len(mem), replace=False))[0] for _ in range(1000)])
                n05 = int((pv[mem] <= 0.05).sum())
                fe = stats.fisher_exact([[n05, len(mem) - n05], [int((pv <= 0.05).sum()) - n05, len(pv) - len(mem) - int((pv <= 0.05).sum()) + n05]],
                                        alternative="greater")[1]
                b3.append({"tissue": t, "family": family, "cell_type": ct, "programme": nm, "n_members_tested": len(mem), "n_members_p_le_0.20": nr,
                           "rayleigh_z": z0, "p_perm_coherence": float((1 + (null >= z0).sum()) / 1001) if nr >= 3 else np.nan,
                           "mean_peak_zt": mu, "n_members_p_le_0.05": n05, "fisher_p_rhythmic_enrichment": float(fe)})
b3 = pd.DataFrame(b3)
b3s = []
for (t, fam), g in b3.groupby(["tissue", "family"]):
    gg = g.dropna(subset=["p_perm_coherence"])
    k = int((gg["p_perm_coherence"] <= 0.05).sum())
    co = gg[gg["p_perm_coherence"] <= 0.05]
    b3s.append({"tissue": t, "family": fam, "n_pairs": len(gg), "n_coherent": k, "expected": 0.05 * len(gg),
                "binomial_p": float(stats.binomtest(k, len(gg), 0.05, alternative="greater").pvalue) if len(gg) else np.nan,
                "coherent_pairs": "; ".join(f"{r.programme.split(':')[1]} @ {r.cell_type} ZT{r.mean_peak_zt:.1f}" for r in co.itertuples()),
                "circular_mean_of_coherent_peaks": phase.circ_mean(co["mean_peak_zt"].to_numpy())[0] if len(co) >= 2 else np.nan})
b3s = pd.DataFrame(b3s)
# post hoc conservative confound check (added after the first run, labelled as such): co-expressed modules move together
# under any sample-wide shift, so coherence whose mean phase lies within ± 3 h of the tissue's sample-pattern axis
# (210 PC1 peak or its antipeak; genes load with either sign) is not separable from that pattern
pat = rd("r2_cochlea/sample_pattern_scores.csv")
pc1pk = pat[pat["pc"] == 1].groupby("tissue")["pc_cosinor_peak_zt"].first()
b3s["pc1_peak_zt"] = b3s["tissue"].map(pc1pk)
b3s["h_from_pattern_axis"] = [min(abs(phase.circ_diff(m, p)), abs(phase.circ_diff(m, p + 12))) if pd.notna(m) else np.nan
                              for m, p in zip(b3s["circular_mean_of_coherent_peaks"], b3s["pc1_peak_zt"])]
b3s["aligned_with_pattern_axis_post_hoc"] = b3s["h_from_pattern_axis"] <= 3
log("\n" + b3s.round(3).to_string(index=False, max_colwidth=120))

# ================================================================ B4 replication lines for the timer split
rep = rd("r2_audit/neutrophil_timer_replication.csv")
hum = rd("r4_crossspecies/human_neutrophil_timeofday.csv")
tsv = rd("r6_myeloid/m_timer_split_verification.csv")
m2x = rd("r6_myeloid/m_m2_external_checks.csv")
b4 = []
x = rep[(rep["dataset"] == "GSE88954") & (rep["condition"] == "sham") & (rep["genes"] == "neutrophil_markers")]
if len(x):
    b4.append({"line": "external cochlea (GSE88954 sham), neutrophil markers: cochlear sign +", "value": x["spearman_rho"].iloc[0], "p": x["p"].iloc[0],
               "consistent": bool(x["spearman_rho"].iloc[0] > 0 and x["p"].iloc[0] <= 0.10)})
for cond in ["control", "neutrophil_depleted"]:
    x = rep[(rep["dataset"] == "GSE198770_bone_marrow") & (rep["condition"] == cond) & (rep["genes"] == "rhythmic_p05")]
    if len(x):
        exp_neg = cond == "control"
        v = x["spearman_rho"].iloc[0]
        b4.append({"line": f"external long-bone marrow (GSE198770 {cond}), rhythmic genes: marrow sign − {'expected' if exp_neg else 'expected lost'}",
                   "value": v, "p": x["p"].iloc[0], "consistent": bool((v < 0 and x["p"].iloc[0] <= 0.10) if exp_neg else v > -0.05)})
if len(hum):   # primary activity-matched mapping, all neutrophil states; p_perm is one-sided for a positive correlation
    for t in ["cochlea", "bm"]:
        x = hum[(hum["tissue"] == t) & hum["primary"].astype(str).eq("True") & (hum["cell_type"] == "all neutrophil states")]
        if len(x):
            v, p = float(x["pearson_r"].iloc[0]), float(x["p_perm"].iloc[0])
            b4.append({"line": f"human blood neutrophils (GSE136632): {t} predictions {'+' if t == 'cochlea' else '−'}", "value": v,
                       "p": p if t == "cochlea" else 1 - p, "consistent": bool((v > 0 and p <= 0.10) if t == "cochlea" else (v < 0 and 1 - p <= 0.10))})
if len(tsv):
    import itertools
    a_ = tsv[tsv["part"].str.startswith("A/B") & (tsv["gene_set"] == "all shared genes") & (tsv["unit"] != "pooled neutrophils")]
    for ts, sg in [("bm", -1), ("cochlea", 1)]:
        xx = a_[a_["tissue"] == ts].assign(score=lambda d: sg * d["spearman"])
        xx["rank"] = xx["score"].rank(ascending=False)
        rk = xx.set_index("unit").loc[["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Mature)"], "rank"]
        p = float(np.mean([sum(c) <= rk.sum() for c in itertools.combinations(range(1, len(xx) + 1), 3)]))
        b4.append({"line": f"within-tissue specificity (676), {ts}", "value": float(rk.sum()), "p": p, "consistent": p <= 0.10})
b = m2x[m2x["check"].str.startswith("Ballesteros")] if len(m2x) else pd.DataFrame()
if len(b):
    b4.append({"line": "Ballesteros blood scRNA: timer score ZT13 > ZT5", "value": b["value"].iloc[0], "p": b["p"].iloc[0], "consistent": bool(b["agrees"].iloc[0])})
b4 = pd.DataFrame(b4)
log("\n" + b4.round(4).to_string(index=False, max_colwidth=110))

# ================================================================ Part A — claim re-grading
r5 = rd("r5_bcell/primary_hypotheses.csv")
ph6 = rd("r6_myeloid/m_primary_hypotheses.csv")
mm6 = rd("r6_myeloid/m_multimethod_matrix.csv")
axr = rd("r6_myeloid/m_axes_rhythm.csv")
trj = rd("r6_myeloid/m_trajectory_rhythm.csv")
nrr = rd("r6_myeloid/m_niche_ratio_rhythm.csv")
m6 = rd("r6_myeloid/m_primary_m6.csv")
m9 = rd("r3_ma2025/m9_tests.csv")
pcd = rd("r6_myeloid/m_paired_contrast.csv")
R08 = rd("r3_ma2025/composition_rhythm_links.csv")


def axr_r(t, ax, g, sub="all cells", col="r_with_sample_pattern_pc1"):
    x = axr[(axr["tissue"] == t) & (axr["axis"] == ax) & (axr["group"] == g) & (axr["subset"] == sub)]
    return float(x[col].iloc[0]) if len(x) else np.nan


def b1_sign(labels):
    x = b1[(b1["pair"].str.startswith("controls")) & b1["label"].isin(labels)]
    if not len(x):
        return None
    return bool(np.sign(x["observed_signature_night_minus_day"].mean()) == np.sign(x["predicted_log2_share_ratio"].mean())), \
        f"bulk signature night − day {x['observed_signature_night_minus_day'].mean():+.2f} vs predicted {x['predicted_log2_share_ratio'].mean():+.2f}"


units = []
for r in r5.itertuples():
    units.append({"round": "5 (B cells)", "claim": r.hypothesis, "tissue": r.tissue, "confirmatory_decision": r.decision, "primary_p": r.primary_p,
                  "direction": ("met" if bool(r.in_window) else "not met") if pd.notna(r.pre_specified_window) else "n/a",
                  "n_strict_methods": int(r.n_methods_agree), "n_methods_evaluated": int(r.n_methods), "pattern_r": np.nan, "confounds": [],
                  "lines": []})
for r in ph6[ph6["decision"] != "pending"].itertuples():
    for tok in str(r.p_by_tissue).split("; "):
        t, p = tok.rsplit(" ", 1)
        m_ = mm6[(mm6["hypothesis"] == r.hypothesis) & (mm6["tissue"] == t)]
        if r.hypothesis == "M6":
            dr = "met" if str(m6.set_index("tissue").loc[t, "direction_ok"]) == "True" else "not met"
        elif r.hypothesis in ("M2", "M9"):
            dr = "met" if bool(r.direction_ok) else "not met"
        else:
            dr = "n/a"
        units.append({"round": "6 (myeloid)", "claim": f"{r.hypothesis} {r.name.split(' ', 1)[1]}", "tissue": t, "confirmatory_decision": r.decision,
                      "primary_p": float(p), "direction": dr, "n_strict_methods": int(m_["agrees"].astype(str).eq("True").sum()),
                      "n_methods_evaluated": len(m_), "pattern_r": np.nan, "confounds": [], "lines": []})
if len(R08):
    x = R08[(R08["subset"] == "all cells") & (R08["contrast"] == "night_noise_vs_day_noise")].dropna(subset=["p_perm_labels"])
    if len(x):
        units.append({"round": "3 (Ma 2025)", "claim": "R08 Ma night- vs day-noise immune composition vs our cochlear composition rhythm", "tissue": "cochlea",
                      "confirmatory_decision": "robust (register rule)", "primary_p": float(x["p_perm_labels"].iloc[0]), "direction":
                      "met" if x["spearman_expected_vs_observed_immune"].iloc[0] > 0 else "not met", "n_strict_methods": 2, "n_methods_evaluated": 3,
                      "pattern_r": np.nan, "confounds": [], "lines": []})
U = {(u["claim"].split(" ")[0], u["tissue"]): u for u in units}


def line(key, name, ok, detail, robust_for=None):
    if key in U:
        U[key]["lines"].append({"name": name, "ok": bool(ok), "detail": detail, "robust_for": robust_for})


# confounds: primary statistic tracks the sample pattern; Ma collection time
for t in ["bm", "cochlea"]:
    for h, val in [("M1", axr_r(t, "maturation (Xie)", "pooled neutrophils")), ("M5", axr_r(t, "mono→mac (curated)", "monocytes + macrophages"))]:
        if (h, t) in U:
            U[(h, t)]["pattern_r"] = val
    x = trj[(trj["tissue"] == t) & trj["feature"].str.contains(r"\[M3 primary\]", regex=True)]
    if ("M3", t) in U and len(x):
        U[("M3", t)]["pattern_r"] = float(x["r_with_sample_pattern_pc1"].iloc[0])
    x = nrr[(nrr["tissue"] == t) & (nrr["score"] == "cxcr4_cxcr2") & (nrr["subset"] == "all cells")]
    if ("M6", t) in U and len(x):
        U[("M6", t)]["pattern_r"] = float(x["r_with_sample_pattern_pc1"].iloc[0])
if ("M8", "cochlea") in U:
    U[("M8", "cochlea")]["pattern_r"] = axr_r("cochlea", "resident macrophage (cochlear)", "Macrophage")
for r in r5.itertuples():
    m_ = re.search(r"r with sample pattern ([−\-+]?\d*\.\d+)", str(r.detail))
    if m_ and (r.hypothesis.split(" ")[0], r.tissue) in U:
        U[(r.hypothesis.split(" ")[0], r.tissue)]["pattern_r"] = float(m_.group(1).replace("−", "-"))
for u in units:
    if pd.notna(u["pattern_r"]) and abs(u["pattern_r"]) >= 0.9:
        u["confounds"].append("sample pattern")
for k in [("M9", "cochlea"), ("R08", "cochlea")]:
    if k in U:
        U[k]["confounds"].append("Ma scRNA collection time")
if ("M2", "paired") in U:
    U[("M2", "paired")]["confounds"].append("sample pattern (each tissue's timer peak sits on its PC1)")

# Part-B lines mapped to claims
for lab_set, key in [(["B Cell (Naïve)"], ("H1", "cochlea")), ([GMP, "Neutrophil (Band)"], ("M3", "cochlea")), (["Monocyte (Classical)"], ("M5", "cochlea"))]:
    s = b1_sign(lab_set)
    if s:
        line(key, "B1 time-matched bulk composition sign (Ma 2025 controls)", s[0], s[1])
x = b1s[(b1s["pair"].str.startswith("controls")) & (b1s["scope"] == "immune labels")]
if len(x):
    line(("R08", "cochlea"), "B1 time-matched bulk composition replication (immune labels, controls)", x["spearman"].iloc[0] > 0 and x["p_perm_one_sided"].iloc[0] <= 0.10,
         f"Spearman {x['spearman'].iloc[0]:+.2f}, p {x['p_perm_one_sided'].iloc[0]:.3f}", robust_for="Ma scRNA collection time")
x = b2s[b2s["contrast"] == "noise_x_time_interaction"]
if len(x):
    line(("M9", "cochlea"), "B2 time-matched pathway-level noise × time vs our ZT15 − ZT3 readiness", x["spearman_with_our_zt15_minus_zt3"].iloc[0] > 0 and
         x["p_perm_one_sided"].iloc[0] <= 0.10, f"Spearman {x['spearman_with_our_zt15_minus_zt3'].iloc[0]:+.2f}, p {x['p_perm_one_sided'].iloc[0]:.3f}",
         robust_for="Ma scRNA collection time")
for r in b3s[b3s["family"] == "readiness (M4)"].itertuples():
    line(("M4", r.tissue), "B3 readiness programme phase coherence (atlas)", r.binomial_p <= 0.10,
         f"{r.n_coherent} of {r.n_pairs} programme × type pairs coherent (expected {r.expected:.1f}; binomial p {r.binomial_p:.3f}; mean phase "
         f"ZT{r.circular_mean_of_coherent_peaks:.1f}, {r.h_from_pattern_axis:.1f} h from the sample-pattern axis)")
    if bool(r.aligned_with_pattern_axis_post_hoc) and ("M4", r.tissue) in U:      # post hoc conservative check (see B3)
        U[("M4", r.tissue)]["confounds"].append("sample pattern (B3 coherence phase on the PC1 axis; post hoc)")
for r in b4.itertuples():
    line(("M2", "paired"), f"B4 {r.line}", r.consistent, f"{r.value:+.3f} (p {r.p:.3g})",
         robust_for="sample pattern (each tissue's timer peak sits on its PC1)" if r.line.startswith("within-tissue specificity") else None)
x = axr[(axr["tissue"] == "cochlea") & (axr["axis"] == "resident macrophage (cochlear)") & (axr["group"] == "Macrophage") &
        (axr["subset"] == "pseudobulk, within-sample centred")]
if len(x) and ("M8", "cochlea") in U:
    line(("M8", "cochlea"), "640 within-sample-centred resident axis (removes sample-wide shifts)", x["p_perm"].iloc[0] <= 0.10,
         f"ZT{x['peak_zt'].iloc[0]:.1f}, p {x['p_perm'].iloc[0]:.3f}", robust_for="sample pattern")
for t, h, ax, g in [("bm", "M1", "maturation (Xie)", "pooled neutrophils"), ("cochlea", "M1", "maturation (Xie)", "pooled neutrophils"),
                    ("bm", "M5", "mono→mac (curated)", "monocytes + macrophages"), ("cochlea", "M5", "mono→mac (curated)", "monocytes + macrophages")]:
    x = axr[(axr["tissue"] == t) & (axr["axis"] == ax) & (axr["group"] == g) & (axr["subset"] == "pseudobulk, within-sample centred")]
    if len(x) and (h, t) in U:
        line((h, t), "640 within-sample-centred version (removes sample-wide shifts)", x["p_perm"].iloc[0] <= 0.10, f"ZT{x['peak_zt'].iloc[0]:.1f}, p "
             f"{x['p_perm'].iloc[0]:.3f}", robust_for="sample pattern")

rows = []
for u in units:
    ok_lines = [l_ for l_ in u["lines"] if l_["ok"]]
    n_sup = u["n_strict_methods"] + len(ok_lines)
    unresolved = [c for c in u["confounds"] if not any(l_["ok"] and l_["robust_for"] and (l_["robust_for"] in c or c in l_["robust_for"]) for l_ in u["lines"])]
    dir_ok = u["direction"] != "not met"
    p = u["primary_p"]
    e2 = p <= 0.10 and dir_ok and n_sup >= 1
    e1 = (p <= 0.20 and dir_ok and n_sup >= 1) or (n_sup >= 2 and dir_ok)
    grade = ("E2 exploratory support" if e2 and not unresolved else "E1 qualified exploratory support" if (e1 or e2) else "E0 no exploratory support")
    n_eval = u["n_methods_evaluated"] + len(u["lines"]) + 1                    # + 1: the primary statistic (trend p <= 0.20)
    n_cons = u["n_strict_methods"] + len(ok_lines) + int(p <= 0.20 and dir_ok)
    rows.append({"round": u["round"], "claim": u["claim"], "tissue": u["tissue"], "confirmatory_decision (unchanged)": u["confirmatory_decision"],
                 "primary_p": p, "direction": u["direction"], "n_strict_agreeing_methods": u["n_strict_methods"],
                 "n_methods_evaluated": u["n_methods_evaluated"], "new_exploratory_lines_consistent": len(ok_lines),
                 "new_exploratory_lines_evaluated": len(u["lines"]), "n_supporting_lines": n_sup,
                 "primary_r_with_sample_pattern": u["pattern_r"], "confounds": "; ".join(u["confounds"]),
                 "unresolved_confounds": "; ".join(unresolved), "exploratory_grade": grade,
                 "directional_consistency (reported, not graded; post hoc)": f"{n_cons} of {n_eval} lines",
                 "lines_detail": " | ".join(f"{l_['name']}: {'consistent' if l_['ok'] else 'not consistent'} ({l_['detail']})" for l_ in u["lines"])})
rg = pd.DataFrame(rows)

write_csv(pd.concat([b1.assign(level="label"), b1s.assign(level="summary")], ignore_index=True), MOD, "ex_bulk_composition_replication.csv",
          "EXPLORATORY B1: Ma 2025 bulk (time-matched) cell-type signature night − day vs our cochlear share rhythm at the inferred collection times",
          required=["pair", "level"])
write_csv(pd.concat([b2.assign(level="programme"), b2s.assign(level="summary")], ignore_index=True), MOD, "ex_bulk_noise_timeofday_programmes.csv",
          "EXPLORATORY B2: Ma 2025 bulk noise × time (and time-matched noise effects) at programme level vs our ZT15 − ZT3 readiness differences",
          required=["contrast", "level"])
write_csv(pd.concat([b3.assign(level="pair"), b3s.assign(level="summary")], ignore_index=True), MOD, "ex_programme_phase_coherence.csv",
          "EXPLORATORY B3: phase coherence of readiness and B programmes in the atlas (Rayleigh vs random gene sets)", required=["tissue", "family", "level"])
write_csv(b4, MOD, "ex_replication_lines.csv", "EXPLORATORY B4: independent replication lines for the neutrophil-timer split (M2 / G02)",
          required=["line", "value", "consistent"])
write_csv(rg, MOD, "ex_claim_regrade.csv", "EXPLORATORY re-grading of the pre-specified claims (E2 / E1 / E0; rules in 680 docstring); confirmatory "
          "decisions shown unchanged", required=["claim", "tissue", "confirmatory_decision (unchanged)", "exploratory_grade"])
pd.set_option("display.width", 250)
log("\n" + rg[["claim", "tissue", "confirmatory_decision (unchanged)", "primary_p", "direction", "n_supporting_lines", "unresolved_confounds",
              "exploratory_grade"]].to_string(index=False, max_colwidth=60))
