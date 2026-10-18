"""Round 8 (D53), step 2 — Ma 2025 × atlas: composition, cell states, clock activity, signalling and noise responses at the
inferred collection phases. EXPLORATORY; every choice below was fixed before the first run.

Timing (700, D54): group internal phases τ from T2 (GSE88954 reference; primary); our ZT = τ + O_C. Phase uncertainty:
1,000 Monte Carlo draws per group (wrapped normal, sd = 95 % half-width / 1.96). Sensitivity: estimators T1 and T3_30, offsets
δ = −4, −2, +2, +4 h added to every phase, and the naive exposure-time assumption (τ = 3 for control / day groups, 15 for night).
Contrasts — scRNA: night_noise − day_noise (≈ 12 h apart; conditional on no noise × time interaction: exposure and collection
inseparable), day_noise − control (time-matched: noise biology), night_noise − control (inseparable). Bulk: CMN − CMD and
NMN − NMD (phase contrasts within noise status; NMN − NMD also carries both exposure times), NMD − CMD and NMN − CMN (1 d noise
effects; ~3 h phase difference, see 700).
Phase genes P (700) are removed from every gene set, signature and deconvolution marker list. Biological samples are the
replicates (libraries / bulk samples), never cells.
(1) scRNA composition: adequately represented labels = myeloid (GMP, four neutrophil labels, classical / non-classical
    monocytes, macrophages, mast cells) and B-lineage (five stages) labels with >= 30 cells in >= 8 of 10 libraries.
    Absolute share (of all cells) and relative share (within immune cells: myeloid, lymphoid, progenitor classes) separately.
    Observed: log2 ratio of group mean shares; Welch t on logit shares (n = 3–4). Predicted: our fitted cochlear share
    (r2_figure abundance_pct_rhythm) at the groups' phases; relative = share / Σ immune shares. Summary: Spearman across labels
    (all adequate; myeloid; B), one-sided permutation p (10,000); sign agreement. Confident cells only as sensitivity.
(2) Bulk composition (declared B1 follow-up): NNLS deconvolution with Ma's control libraries (C1–C3) as reference — labels
    with >= 50 reference cells; expected counts per cell = mean CP10k profile × mean UMIs per cell; markers = top 50 genes per
    label by log2 fold change of its mean CP10k over the mean of the other labels (mean CP10k >= 1; P, mito, ribo excluded);
    gene weights 1 / mean bulk CPM; fractions = coefficients / sum. Validation: the same procedure on the pseudobulk of all 10
    scRNA libraries; labels with validation r < 0.5 (estimated vs true fraction across libraries) are excluded from the tests.
    Immune-relative fractions and absolute fractions; Spearman of observed log2 ratios with predicted relative / absolute shares
    across validated immune labels; one-sided permutation p. Absolute marker signatures (680 B1 definition, P excluded) and the
    total immune signature are reported separately (indirect evidence).
(3) scRNA cell states per adequately represented subtype and pooled units (pooled neutrophils; monocytes + macrophages;
    neutrophils + monocytes; B lineage): state axes (640: maturation, blood timer, tissue vs marrow, resident macrophage,
    mono→mac; 660: NeuMAP tissue vs marrow, old vs young, chronological age) as per-library median of cell scores; readiness
    programmes (645) and B programmes (r5) as per-library pseudobulk mean gene z; Cxcr4 / Cxcr2 log-ratio and Sell
    (neutrophils; per-library mean). Predictions from the matching cochlear fits (640, 645, 650, 660, r5). Per item Welch p
    with BH within family × contrast; summary Spearman across items (all; items whose own fit has p_perm <= 0.20), sign
    agreement, Monte Carlo and sensitivity as above.
(4) Clock activity: per adequate subtype and pooled unit, per-library clock profile template-matched to our same-type fit
    (>= 5 phase genes; otherwise the tissue fit); group circular means; tracking = the unit's night − day phase difference
    within ± 4 h of the structural reference (osteoprogenitors + endothelial cells, same method). Uses clock genes of
    immune cells to read their own clocks — different cells from the structural phase reference.
(5) Signalling: CXCL12 → CXCR4 proxy per library = mean Cxcl12 CP10k of stromal senders × mean Cxcr4 CP10k of neutrophils.
(6) Noise responses (time-matched): scRNA day_noise − control per unit × programme (z difference, Welch p, BH) and bulk
    programme mean gene t of the 1-d noise effects (P excluded); cross-dataset Spearman across programmes.
Multiple testing: BH within each item family and contrast; BH across all summary tests (q reported).
Outputs (results/r8_ma_integration/): ma_composition_sc.csv, ma_deconvolution_validation.csv, ma_composition_bulk.csv,
ma_states_sc.csv, ma_clock_activity.csv, ma_signalling.csv, ma_noise_response.csv, ma_summary_tests.csv
"""
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r8_ma_integration"
W = phase.W
rng = np.random.default_rng(710)
NMC, NPERM = 1000, 10000
PG = set(phase.CANONICAL_CLOCK) | {"Arntl", "Clock", "Bhlhe40", "Bhlhe41"}
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
MYE = [GMP] + NEU + MONO + ["Mast Cell"]
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
rd = lambda rel: pd.read_csv(RESULTS / rel) if (RESULTS / rel).exists() else pd.DataFrame()  # noqa: E731
off = rd("r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
offt = off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea")].set_index("unit")["offset_canonical_h"]
cls = rd("00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
IMMUNE = set(cls[cls.isin(["Myeloid", "Lymphoid", "Progenitor"])].index)
okg = lambda g: g not in PG and not re.match(r"^(mt-|Rp[sl]\d)", g)  # noqa: E731

# ---------------- timing assumptions
gph = rd("r8_ma_timing/ma_group_phase.csv")
EXPO = {"control": 3.0, "day_noise": 3.0, "night_noise": 15.0, "CMD": 3.0, "NMD": 3.0, "CMN": 15.0, "NMN": 15.0}


def phases(ds, est="T2", delta=0.0, exposure=False):
    x = gph[(gph["dataset"].str.startswith(ds)) & (gph["estimator"] == est)].set_index("group")
    if exposure:
        return {g: (EXPO[g] + O_C) % 24 for g in x.index}, {g: 0.0 for g in x.index}
    return ({g: (x.loc[g, "internal_phase_h"] + delta + O_C) % 24 for g in x.index}, {g: x.loc[g, "ci95_halfwidth_h"] / 1.96 for g in x.index})


ASSUMPTIONS = [("A0 inferred (T2)", dict(est="T2")), ("T1", dict(est="T1")), ("T3_30", dict(est="T3_30")),
               ("offset −4 h", dict(est="T2", delta=-4)), ("offset −2 h", dict(est="T2", delta=-2)), ("offset +2 h", dict(est="T2", delta=2)),
               ("offset +4 h", dict(est="T2", delta=4)), ("exposure-time assumption", dict(exposure=True))]
SC_CON = [("night_noise", "day_noise"), ("day_noise", "control"), ("night_noise", "control")]
B_CON = [("CMN", "CMD"), ("NMN", "NMD"), ("NMD", "CMD"), ("NMN", "CMN")]
CON_NOTE = {("night_noise", "day_noise"): "≈ 12 h phase + exposure time (inseparable; conditional on no noise × time interaction)",
            ("day_noise", "control"): "time-matched: 3-d noise effect", ("night_noise", "control"): "phase + noise (inseparable)",
            ("CMN", "CMD"): "phase contrast, no noise (clean)", ("NMN", "NMD"): "phase contrast after noise at each group's exposure time",
            ("NMD", "CMD"): "1-d day-noise effect (~3 h phase shift)", ("NMN", "CMN"): "1-d night-noise effect (~3 h phase shift)"}


def perm_spear(x, y, n=NPERM):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    r = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (n + 1)), len(x)


def cos_diff(amp, pk, ta, tb):
    return amp * (np.cos(W * (ta - pk)) - np.cos(W * (tb - pk)))


summ = []


def summarise(family, contrast, items, pred_fn, obs_col, scope, ds, extra=None):
    """items: DataFrame with observed column; pred_fn(phase_dict) -> predicted array aligned to items"""
    for an, kw in ASSUMPTIONS:
        if ds == "bulk" and an == "T3_30":
            continue
        ph, sd = phases("scRNA" if ds == "sc" else "bulk", **kw)
        if contrast[0] not in ph or contrast[1] not in ph:
            continue
        pr_ = pred_fn(ph)
        r, p, n = perm_spear(pr_, items[obs_col].to_numpy(float), n=NPERM if an.startswith("A0") else 2000)
        rec = {"family": family, "dataset": ds, "contrast": f"{contrast[0]} − {contrast[1]}", "contrast_note": CON_NOTE.get(contrast, ""),
               "scope": scope, "assumption": an, "n_items": n, "spearman": r, "p_perm_one_sided": p,
               "sign_agreement": float(np.mean(np.sign(pr_[np.isfinite(pr_)]) == np.sign(items[obs_col].to_numpy(float)[np.isfinite(pr_)])))
               if np.isfinite(pr_).any() else np.nan}
        if an.startswith("A0"):
            mc = []
            for _ in range(NMC):
                phd = {g: (ph[g] + rng.normal(0, sd[g])) % 24 for g in ph}
                v = pred_fn(phd)
                ok = np.isfinite(v) & np.isfinite(items[obs_col].to_numpy(float))
                if ok.sum() >= 4 and np.std(v[ok]) > 0:
                    mc.append(stats.spearmanr(v[ok], items[obs_col].to_numpy(float)[ok])[0])
            mc = np.array(mc)
            rec.update({"mc_median_spearman": float(np.median(mc)) if len(mc) else np.nan, "mc_frac_positive": float((mc > 0).mean()) if len(mc) else np.nan})
        if extra:
            rec.update(extra)
        summ.append(rec)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


# ================================================================ scRNA
q = ad.read_h5ad(CACHE / "r3_query/ma2025_mapped.h5ad")
q.obs["label"] = q.obs["label"].astype(str)
S = sorted(q.obs["sample"].astype(str).unique())
cond = q.obs.groupby("sample", observed=True)["condition"].first().astype(str).reindex(S)
smp = q.obs["sample"].astype(str).to_numpy()
lab = q.obs["label"].to_numpy()
conf = q.obs["confident"].astype(bool).to_numpy()
cnt_all = q.obs.groupby(["sample", "label"], observed=True).size().unstack(fill_value=0).reindex(S).fillna(0)
ADEQ = [l for l in MYE + BST if l in cnt_all.columns and (cnt_all[l] >= 30).sum() >= 8]
log(f"adequately represented: {ADEQ}")
pr = rd("r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")


def share_fit(l, t):
    r = pr.loc[l]
    return max(float(r["mesor_pct"] + r["amplitude_pct"] * np.cos(W * (t - r["peak_zt"]))), 1e-3)


IMM_FIT = [l for l in pr.index if l in IMMUNE]
comp = []
for subset in ["all cells", "confident cells"]:
    o = q.obs[conf] if subset == "confident cells" else q.obs
    c_ = o.groupby(["sample", "label"], observed=True).size().unstack(fill_value=0).reindex(S).fillna(0)
    tot = c_.sum(1)
    imm = c_[[l for l in c_.columns if l in IMMUNE]].sum(1)
    for scale, den in [("absolute (share of all cells)", tot), ("relative (share of immune cells)", imm)]:
        sh = c_.div(den, axis=0)
        for a, b in SC_CON:
            ia, ib = cond.index[cond == a], cond.index[cond == b]
            for l in ADEQ:
                if l not in sh.columns:
                    continue
                va, vb = sh.loc[ia, l].to_numpy(), sh.loc[ib, l].to_numpy()
                lg = lambda v: np.log((v + 1e-4) / (1 - v + 1e-4))  # noqa: E731
                comp.append({"subset": subset, "scale": scale, "contrast": f"{a} − {b}", "contrast_note": CON_NOTE[(a, b)], "label": l,
                             "lineage": "B" if l in BST else "myeloid", "mean_a": va.mean(), "mean_b": vb.mean(),
                             "observed_log2_ratio": float(np.log2((va.mean() + 1e-5) / (vb.mean() + 1e-5))), "welch_p_logit": welch(lg(va), lg(vb))})
comp = pd.DataFrame(comp)
comp["q_BH"] = np.nan
for k, g in comp.groupby(["subset", "scale", "contrast"]):
    ok = g["welch_p_logit"].notna()
    comp.loc[g.index[ok], "q_BH"] = multipletests(g.loc[ok, "welch_p_logit"], method="fdr_bh")[1]


def comp_pred(items, scale, a, b):
    def f(ph):
        out = []
        for l in items["label"]:
            if l not in pr.index:
                out.append(np.nan)
                continue
            sa, sb = share_fit(l, ph[a]), share_fit(l, ph[b])
            if scale.startswith("relative"):
                sa /= sum(share_fit(x, ph[a]) for x in IMM_FIT)
                sb /= sum(share_fit(x, ph[b]) for x in IMM_FIT)
            out.append(np.log2(sa / sb))
        return np.array(out, float)
    return f


for (subset, scale, con), g in comp.groupby(["subset", "scale", "contrast"]):
    a, b = con.split(" − ")
    if (a, b) == ("day_noise", "control"):
        continue                                     # no phase difference: not a rhythm test
    for scope, gg in [("all adequate labels", g), ("myeloid", g[g["lineage"] == "myeloid"]), ("B lineage", g[g["lineage"] == "B"])]:
        if subset == "confident cells" and scope != "all adequate labels":
            continue
        summarise(f"scRNA composition | {scale} | {subset}", (a, b), gg.reset_index(drop=True), comp_pred(gg.reset_index(drop=True), scale, a, b),
                  "observed_log2_ratio", scope, "sc")
log("scRNA composition done")

# ================================================================ bulk deconvolution
Xc = q.X.tocsr()
lib_cell = np.asarray(Xc.sum(1)).ravel()
ref_mask = cond.reindex(smp).to_numpy() == "control"
ref_labels = [l for l in pd.Series(lab[ref_mask]).value_counts().loc[lambda s: s >= 50].index]
prof, size = {}, {}
for l in ref_labels:
    m = ref_mask & (lab == l)
    x = Xc[m]
    prof[l] = np.asarray((x.multiply(1 / lib_cell[m][:, None])).mean(0)).ravel() * 1e4
    size[l] = float(lib_cell[m].mean())
P = pd.DataFrame(prof, index=q.var_names)
genes_ok = [g for g in P.index if okg(g)]
mk = set()
for l in ref_labels:
    others = P[[c for c in ref_labels if c != l]].mean(1)
    fc = np.log2((P[l] + 0.1) / (others + 0.1))
    cand = fc[(P[l] >= 1)].loc[lambda s: s.index.isin(genes_ok)].sort_values(ascending=False)
    mk |= set(cand.index[:50])
mk = sorted(mk)
REF = P.loc[mk].mul(pd.Series(size), axis=1)


def deconv(y):
    y = y.reindex(mk).fillna(0).to_numpy(float)
    wts = 1 / np.maximum(y, 1)
    A = REF.to_numpy() * wts[:, None]
    coef, _ = optimize.nnls(A, y * wts)
    return pd.Series(coef / coef.sum() if coef.sum() > 0 else coef, index=REF.columns)


sc_pb = {s: pd.Series(np.asarray(Xc[smp == s].sum(0)).ravel(), index=q.var_names) for s in S}
val = []
est_sc = pd.DataFrame({s: deconv(v / v.sum() * 1e6) for s, v in sc_pb.items()}).T
true_sc = cnt_all.reindex(columns=ref_labels).fillna(0)
true_sc = true_sc.div(cnt_all.sum(1), axis=0)
for l in ref_labels:
    r = float(np.corrcoef(est_sc[l], true_sc[l])[0, 1]) if est_sc[l].std() > 0 and true_sc[l].std() > 0 else np.nan
    val.append({"label": l, "immune": l in IMMUNE, "n_reference_cells": int((ref_mask & (lab == l)).sum()), "validation_r": r,
                "mean_true_fraction": float(true_sc[l].mean()), "mean_estimated_fraction": float(est_sc[l].mean()),
                "passes": bool(np.isfinite(r) and r >= 0.5)})
val = pd.DataFrame(val)
bulk = pd.read_csv("external/ma2025/CRA019827_bulk_counts_unstranded.tsv", sep="\t", index_col=0)
names = [f"{c}{m}{i}" for c in "CN" for m in ("MD", "MN") for i in (1, 2, 3)]
bulk = bulk[names]
bcpm = bulk / bulk.sum(0) * 1e6
fr = pd.DataFrame({s: deconv(bcpm[s]) for s in names}).T
VAL_IMM = [l for l in val.loc[val["passes"] & val["immune"], "label"]]
bcomp = []
for scale in ["absolute (fraction of all cells)", "relative (fraction of immune cells)"]:
    F = fr if scale.startswith("absolute") else fr[VAL_IMM].div(fr[VAL_IMM].sum(1), axis=0)
    for a, b in B_CON:
        ia, ib = [s for s in names if s.startswith(a)], [s for s in names if s.startswith(b)]
        for l in VAL_IMM:
            va, vb = F.loc[ia, l].to_numpy(), F.loc[ib, l].to_numpy()
            bcomp.append({"scale": scale, "contrast": f"{a} − {b}", "contrast_note": CON_NOTE[(a, b)], "label": l,
                          "lineage": "B" if l in BST else "myeloid" if l in MYE else "other immune", "mean_a": va.mean(), "mean_b": vb.mean(),
                          "observed_log2_ratio": float(np.log2((va.mean() + 1e-5) / (vb.mean() + 1e-5))), "welch_p": welch(np.log(va + 1e-5), np.log(vb + 1e-5))})
# absolute marker signatures (680 B1 definition, P excluded) — indirect
mkl = rd("00_audit/label_markers.csv")
mkl = mkl[(mkl["logfoldchanges"] >= 1) & (mkl["pvals_adj"] < 0.05) & (mkl["pct_nz_reference"] < 0.30) & mkl["names"].map(okg)]
Lb = np.log2(bcpm[(bcpm > 1).sum(1) >= 3] + 1)
for a, b in B_CON:
    cols = [s for s in names if s.startswith(a) or s.startswith(b)]
    Z = Lb[cols].sub(Lb[cols].mean(1), axis=0).div(Lb[cols].std(1).replace(0, np.nan), axis=0)
    tot_imm = []
    for l, g in mkl.groupby("cell_type"):
        gg = [x for x in g["names"] if x in Z.index]
        if len(gg) < 5 or l not in IMMUNE:
            continue
        s_ = Z.loc[gg].mean(0)
        ia, ib = [c for c in cols if c.startswith(a)], [c for c in cols if c.startswith(b)]
        d = float(s_[ia].mean() - s_[ib].mean())
        tot_imm.append(d)
        bcomp.append({"scale": "absolute marker signature (indirect; 680 B1 definition, P excluded)", "contrast": f"{a} − {b}",
                      "contrast_note": CON_NOTE[(a, b)], "label": l, "lineage": "B" if l in BST else "myeloid" if l in MYE else "other immune",
                      "observed_log2_ratio": d, "welch_p": welch(s_[ia], s_[ib])})
    bcomp.append({"scale": "total immune signature (mean over immune labels; indirect)", "contrast": f"{a} − {b}", "contrast_note": CON_NOTE[(a, b)],
                  "label": "all immune labels", "observed_log2_ratio": float(np.mean(tot_imm)), "welch_p": np.nan})
bcomp = pd.DataFrame(bcomp)
bcomp["q_BH"] = np.nan
for k, g in bcomp.groupby(["scale", "contrast"]):
    ok = g["welch_p"].notna()
    if ok.any():
        bcomp.loc[g.index[ok], "q_BH"] = multipletests(g.loc[ok, "welch_p"], method="fdr_bh")[1]
for (scale, con), g in bcomp[bcomp["label"] != "all immune labels"].groupby(["scale", "contrast"]):
    a, b = con.split(" − ")
    if (a, b) not in [("CMN", "CMD"), ("NMN", "NMD")]:
        continue
    sc_scale = "relative" if scale.startswith("relative") else "absolute"
    gg = g[g["label"].isin(pr.index)].reset_index(drop=True)
    for scope, g2 in [("validated immune labels", gg), ("myeloid", gg[gg["lineage"] == "myeloid"].reset_index(drop=True)),
                      ("B lineage", gg[gg["lineage"] == "B"].reset_index(drop=True))]:
        summarise(f"bulk composition | {scale}", (a, b), g2, comp_pred(g2, sc_scale, a, b), "observed_log2_ratio", scope, "bulk",
                  extra={"note": "B1 follow-up (declared)" if scale.startswith("relative") and scope == "validated immune labels" else ""})
log("bulk deconvolution done")

# ================================================================ scRNA cell states
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
hm = {}
for line in open(EXTERNAL / "resources/msigdb_mh_hallmark.gmt"):
    x = line.rstrip("\n").split("\t")
    hm[x[0]] = x[2:]
PROG = {**{f"myeloid:{k}": v for k, v in SETS["myeloid_function"].items()},
        "hallmark:GLYCOLYSIS": hm["HALLMARK_GLYCOLYSIS"], "hallmark:OXIDATIVE_PHOSPHORYLATION": hm["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]}
BPROG = {f"b_cell:{k}": v for k, v in SETS["b_cell"].items()}
sig = rd("r6_myeloid/myeloid_reference_signatures.csv")
esig = rd("r6_myeloid/m_external_signatures.csv")


def sset(df, name):
    s_ = df[df["signature"] == name]
    return ([g for g in s_.loc[s_["direction"].str.startswith("up"), "gene"] if okg(g)],
            [g for g in s_.loc[s_["direction"].str.startswith("down"), "gene"] if okg(g)])


AXES = {"maturation (Xie)": sset(sig, "maturation_late_vs_early (Xie 2020)"), "blood timer ZT13−ZT5 (GSE86619)": sset(sig, "blood_timer_ZT13_vs_ZT5 (GSE86619)"),
        "tissue vs marrow (Ballesteros)": sset(sig, "tissue_vs_marrow (Ballesteros 2020)"),
        "resident macrophage (cochlear)": sset(sig, "cochlear_macrophage_resident_vs_infiltrating"),
        "mono→mac (curated)": ([g for g in SETS["macrophage"]["tlf_resident"] + ["Adgre1", "C1qa", "C1qb", "Apoe"] if okg(g)],
                               [g for g in SETS["macrophage"]["ccr2_monocyte_derived"] if okg(g)]),
        **{n: sset(esig, n) for n in esig["signature"].unique()}}
UNITS = {**{l: [l] for l in ADEQ}, "pooled neutrophils": [l for l in NEU if l in ADEQ], "monocytes + macrophages": [l for l in MONO if l in ADEQ],
         "neutrophils + monocytes (pooled)": [l for l in NEU + MONO[:2] if l in ADEQ], "B lineage (pooled)": [l for l in BST if l in ADEQ]}
X = q.copy()
sc.pp.normalize_total(X, target_sum=1e4)
sc.pp.log1p(X)
cell_scores = {}
for ax, (u, d) in AXES.items():
    u_, d_ = [g for g in u if g in X.var_names], [g for g in d if g in X.var_names]
    if len(u_) < 3:
        continue
    sc.tl.score_genes(X, u_, score_name="_u", random_state=0)
    s_ = X.obs["_u"].to_numpy().copy()
    if len(d_) >= 3:
        sc.tl.score_genes(X, d_, score_name="_d", random_state=0)
        s_ = s_ - X.obs["_d"].to_numpy()
    cell_scores[ax] = s_
cp10 = Xc.multiply(1 / lib_cell[:, None]).tocsr() * 1e4
gi = {g: i for i, g in enumerate(q.var_names)}
ratio = np.log2((np.asarray(cp10[:, gi["Cxcr4"]].todense()).ravel() + 1) / (np.asarray(cp10[:, gi["Cxcr2"]].todense()).ravel() + 1))
sell = np.log2(np.asarray(cp10[:, gi["Sell"]].todense()).ravel() + 1)
axr = rd("r6_myeloid/m_axes_rhythm.csv")
axr = axr[(axr["tissue"] == "cochlea") & (axr["subset"] == "all cells")]
exo = rd("r6_myeloid/m_external_axes_ours.csv")
exo = exo[(exo["tissue"] == "cochlea") & (exo["kind"] == "rhythm of per-sample medians")]
fnr = rd("r6_myeloid/m_function_rhythm.csv")
fnr = fnr[(fnr["tissue"] == "cochlea") & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred")]
bpr = rd("r5_bcell/b_programme_rhythm.csv")
bpr = bpr[(bpr["tissue"] == "cochlea") & (bpr["version"] == "primary") & (bpr["kind"] == "uncentred")]
nrr = rd("r6_myeloid/m_niche_ratio_rhythm.csv")
nrr = nrr[(nrr["tissue"] == "cochlea") & (nrr["subset"] == "all cells")]


def fit_for(kind, name, unit):
    if kind == "axis":
        x = axr[(axr["axis"] == name) & (axr["group"] == unit)]
        if not len(x) and unit == "pooled neutrophils":
            x = exo[(exo["axis"] == name) & (exo["group"] == unit)]
    elif kind == "programme":
        x = fnr[(fnr["programme"] == name) & (fnr["unit"] == unit)] if name in PROG else bpr[(bpr["gene_set"] == name) & (bpr["stage"] == unit)]
    else:
        x = nrr[(nrr["score"] == name)] if unit == "pooled neutrophils" else pd.DataFrame()
    return (float(x["amplitude"].iloc[0]), float(x["peak_zt"].iloc[0]), float(x["p_perm"].iloc[0])) if len(x) else (np.nan, np.nan, np.nan)


st_rows = []
for unit, labs_ in UNITS.items():
    if not labs_:
        continue
    m_unit = np.isin(lab, labs_)
    per = {}
    for ax, s_ in cell_scores.items():
        per[("axis", ax)] = {s: float(np.median(s_[m_unit & (smp == s)])) if (m_unit & (smp == s)).sum() >= 10 else np.nan for s in S}
    if set(labs_) <= set(NEU):
        per[("ratio", "cxcr4_cxcr2")] = {s: float(ratio[m_unit & (smp == s)].mean()) for s in S}
        per[("ratio", "sell")] = {s: float(sell[m_unit & (smp == s)].mean()) for s in S}
    pbu = pd.DataFrame({s: np.asarray(Xc[m_unit & (smp == s)].sum(0)).ravel() for s in S}, index=q.var_names)
    cpm = pbu / pbu.sum(0).replace(0, np.nan) * 1e6
    L = np.log2(cpm + 1)[cpm.mean(1) >= 1]
    Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
    sets = PROG if not set(labs_) <= set(BST) else BPROG
    for nm, genes in sets.items():
        g = [x for x in genes if x in Z.index and okg(x)]
        if len(g) >= 3:
            per[("programme", nm)] = Z.loc[g].mean(0).to_dict()
    for (kind, name), vals in per.items():
        v = pd.Series(vals).reindex(S)
        amp, pk, pp = fit_for(kind, name, unit)
        for a, b in SC_CON:
            va, vb = v[cond == a].to_numpy(float), v[cond == b].to_numpy(float)
            st_rows.append({"unit": unit, "lineage": "B" if set(labs_) <= set(BST) else "myeloid", "kind": kind, "metric": name,
                            "contrast": f"{a} − {b}", "contrast_note": CON_NOTE[(a, b)], "observed_difference": float(np.nanmean(va) - np.nanmean(vb)),
                            "welch_p": welch(va[np.isfinite(va)], vb[np.isfinite(vb)]), "our_amplitude": amp, "our_peak_zt": pk, "our_p_perm": pp})
states = pd.DataFrame(st_rows)
states["family"] = states["lineage"] + " " + states["kind"]
states["q_BH"] = np.nan
for k, g in states.groupby(["family", "contrast"]):
    ok = g["welch_p"].notna()
    if ok.any():
        states.loc[g.index[ok], "q_BH"] = multipletests(g.loc[ok, "welch_p"], method="fdr_bh")[1]


def st_pred(items, a, b):
    def f(ph):
        return np.array([cos_diff(r.our_amplitude, r.our_peak_zt, ph[a], ph[b]) for r in items.itertuples()], float)
    return f


for (lin, con), g in states.groupby(["lineage", "contrast"]):
    a, b = con.split(" − ")
    if (a, b) == ("day_noise", "control"):
        continue
    g = g.dropna(subset=["our_amplitude", "observed_difference"]).reset_index(drop=True)
    for scope, gg in [("all items with a fit", g), ("items whose own fit has p_perm <= 0.20", g[g["our_p_perm"] <= 0.20].reset_index(drop=True))]:
        summarise(f"scRNA states | {lin}", (a, b), gg, st_pred(gg, a, b), "observed_difference", scope, "sc")
log("scRNA states done")

# ================================================================ clock activity
T33 = rd("33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv")[["group", "gene", "a", "b"]]
T33 = T33[T33["gene"].isin(PG)]
TP = rd("r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").drop_duplicates("gene").set_index("gene")
TP = TP[TP.index.isin(PG)]          # clock (phase) genes only
GRID = np.arange(0, 24, 0.25)


def clock_phase(labs_, fit, o):
    m_unit = np.isin(lab, labs_)
    pbu = pd.DataFrame({s: np.asarray(Xc[m_unit & (smp == s)].sum(0)).ravel() for s in S}, index=q.var_names)
    cpm = pbu / pbu.sum(0).replace(0, np.nan) * 1e6
    L = np.log2(cpm + 1)
    g = [x for x in fit.index if x in L.index and cpm.loc[x].mean() > 1]
    if len(g) < 5:
        return None, 0
    gm = L.loc[g].T.groupby(cond.to_numpy()).mean().T
    D = L.loc[g].sub(gm.mean(1), axis=0)
    A, B = fit.loc[g, "a"].to_numpy(float), fit.loc[g, "b"].to_numpy(float)
    out = {}
    for s in S:
        prof = D[s].to_numpy(float)
        best = (np.nan, -2)
        for t in GRID:
            pred = A * np.cos(W * (t + o)) + B * np.sin(W * (t + o))
            if prof.std() > 0:
                r = np.corrcoef(prof, pred - pred.mean())[0, 1]
                if r > best[1]:
                    best = (t, r)
        out[s] = best
    return out, len(g)


ck = []
struct_fit = TP
ref, _ = clock_phase(["Osteoprogenitor", "Endothelial Cell"], struct_fit, O_C)
ref_g = {c: phase.circ_mean(np.array([ref[s][0] for s in cond.index[cond == c]]))[0] for c in ["control", "day_noise", "night_noise"]}
ref_nd = float(phase.circ_diff(ref_g["night_noise"], ref_g["day_noise"]))
for unit, labs_ in UNITS.items():
    if not labs_:
        continue
    fit = T33[T33["group"] == labs_[0]].drop_duplicates("gene").set_index("gene") if len(labs_) == 1 else pd.DataFrame()
    use, o = (fit, float(offt.get(labs_[0], O_C))) if len(fit) >= 5 else (TP, O_C)
    res, ng = clock_phase(labs_, use, o)
    if res is None:
        ck.append({"unit": unit, "n_phase_genes": ng, "note": "fewer than 5 expressed phase genes"})
        continue
    gph_ = {c: phase.circ_mean(np.array([res[s][0] for s in cond.index[cond == c]])) for c in ["control", "day_noise", "night_noise"]}
    nd = float(phase.circ_diff(gph_["night_noise"][0], gph_["day_noise"][0]))
    ck.append({"unit": unit, "lineage": "B" if set(labs_) <= set(BST) else "myeloid", "reference_fit": "same-type" if use is fit else "tissue",
               "n_phase_genes": ng, **{f"phase_{c}": gph_[c][0] for c in gph_}, **{f"resultant_{c}": gph_[c][1] for c in gph_},
               "mean_template_r": float(np.mean([res[s][1] for s in S])), "night_minus_day_h": nd, "structural_night_minus_day_h": ref_nd,
               "tracking_within_4h": bool(abs(phase.circ_diff(nd, ref_nd)) <= 4),
               "offset_vs_structural_day_h": float(phase.circ_diff(gph_["day_noise"][0], ref_g["day_noise"])),
               "offset_vs_structural_night_h": float(phase.circ_diff(gph_["night_noise"][0], ref_g["night_noise"]))})
ck = pd.DataFrame(ck)
log("clock activity done")

# ================================================================ signalling proxy
stromal = [l for l in ["Osteoprogenitor", "Fibrocyte (Cochlear)", "Fibrocyte (Spiral Ligament)", "Endothelial Cell"] if l in cnt_all.columns]
sig_rows = []
for s in S:
    ms, mn = np.isin(lab, stromal) & (smp == s), np.isin(lab, NEU) & (smp == s)
    c12 = float(np.asarray(cp10[ms][:, gi["Cxcl12"]].mean()).ravel()[0]) if ms.sum() else np.nan
    c4 = float(np.asarray(cp10[mn][:, gi["Cxcr4"]].mean()).ravel()[0]) if mn.sum() else np.nan
    sig_rows.append({"sample": s, "condition": cond[s], "stromal_Cxcl12_cp10k": c12, "neutrophil_Cxcr4_cp10k": c4, "cxcl12_cxcr4_proxy": c12 * c4})
sigd = pd.DataFrame(sig_rows)
nfit = nrr[nrr["score"] == "cxcr4_cxcr2"]
for a, b in SC_CON:
    va, vb = sigd.loc[sigd["condition"] == a, "cxcl12_cxcr4_proxy"], sigd.loc[sigd["condition"] == b, "cxcl12_cxcr4_proxy"]
    sigd = pd.concat([sigd, pd.DataFrame([{"sample": "summary", "condition": f"{a} − {b}", "cxcl12_cxcr4_proxy": float(np.log2(va.mean() / vb.mean())),
                                          "welch_p": welch(np.log(va), np.log(vb)), "contrast_note": CON_NOTE[(a, b)]}])], ignore_index=True)

# ================================================================ noise responses (time-matched)
de_b = rd("r3_ma2025/bulk_de.csv")
nz = []
for contrast in ["noise_effect_day_group", "noise_effect_night_group"]:
    d = de_b[(de_b["contrast"] == contrast) & de_b["gene"].map(okg)].set_index("gene")
    for nm, genes in {**PROG, **BPROG}.items():
        g = [x for x in genes if x in d.index]
        if len(g) >= 3:
            nz.append({"dataset": "bulk (1 d)", "unit": "whole cochlea", "contrast": contrast, "programme": nm, "effect": float(d.loc[g, "t"].mean()),
                       "effect_scale": "mean gene t", "p": np.nan})
nzs = states[(states["contrast"] == "day_noise − control") & (states["kind"] == "programme")]
for r in nzs.itertuples():
    nz.append({"dataset": "scRNA (3 d)", "unit": r.unit, "contrast": "day_noise − control", "programme": r.metric, "effect": r.observed_difference,
               "effect_scale": "pseudobulk mean z difference", "p": r.welch_p, "q_BH": r.q_BH})
nz = pd.DataFrame(nz)
xb = nz[(nz["dataset"] == "bulk (1 d)") & (nz["contrast"] == "noise_effect_day_group")].set_index("programme")["effect"]
for unit in ["neutrophils + monocytes (pooled)", "B lineage (pooled)"]:
    xs = nz[(nz["dataset"] == "scRNA (3 d)") & (nz["unit"] == unit)].set_index("programme")["effect"]
    common = [p for p in xs.index if p in xb.index]
    r, p, n = perm_spear(xb[common], xs[common])
    summ.append({"family": "noise response replication (time-matched)", "dataset": "scRNA vs bulk", "contrast": "scRNA day_noise − control vs bulk NMD − CMD",
                 "contrast_note": "different post-exposure times (3 d vs 1 d)", "scope": unit, "assumption": "time-matched (no phase assumption)",
                 "n_items": n, "spearman": r, "p_perm_one_sided": p})

sm = pd.DataFrame(summ)
ok = sm["p_perm_one_sided"].notna()
sm.loc[ok, "q_BH_all_summary_tests"] = multipletests(sm.loc[ok, "p_perm_one_sided"], method="fdr_bh")[1]
write_csv(comp, MOD, "ma_composition_sc.csv", "EXPLORATORY: Ma scRNA absolute and immune-relative shares of adequately represented myeloid / B labels "
          "by contrast (samples as replicates; Welch on logit shares, BH)", required=["subset", "scale", "contrast", "label", "observed_log2_ratio"])
write_csv(val, MOD, "ma_deconvolution_validation.csv", "EXPLORATORY: NNLS deconvolution (Ma control-library reference) validated on the 10 scRNA libraries",
          required=["label", "validation_r", "passes"])
write_csv(pd.concat([bcomp, fr.reset_index().rename(columns={"index": "sample"}).melt(id_vars="sample", var_name="label", value_name="fraction")
                     .assign(scale="estimated fraction per bulk sample")], ignore_index=True), MOD, "ma_composition_bulk.csv",
          "EXPLORATORY: Ma bulk composition — deconvolved absolute / immune-relative fractions and absolute marker signatures (indirect) by contrast",
          required=["scale", "label"])
write_csv(states, MOD, "ma_states_sc.csv", "EXPLORATORY: Ma scRNA states per unit (axes, programmes, Cxcr4 / Cxcr2, Sell) by contrast with our "
          "cochlear fits (samples as replicates; Welch, BH within family × contrast)", required=["unit", "metric", "contrast", "observed_difference"])
write_csv(ck, MOD, "ma_clock_activity.csv", "EXPLORATORY: clock phase of Ma myeloid / B units vs structural cells (tracking of the night − day "
          "collection difference)", required=["unit"])
write_csv(sigd, MOD, "ma_signalling.csv", "EXPLORATORY: CXCL12 → CXCR4 proxy (stromal Cxcl12 × neutrophil Cxcr4) per library and contrast",
          required=["sample", "condition"])
write_csv(nz, MOD, "ma_noise_response.csv", "EXPLORATORY: time-matched noise responses at programme level (scRNA 3 d, bulk 1 d)",
          required=["dataset", "unit", "contrast", "programme", "effect"])
write_csv(sm, MOD, "ma_summary_tests.csv", "EXPLORATORY: summary tests (Spearman vs our predictions at the inferred phases) with phase Monte Carlo, "
          "estimator / offset / exposure-time sensitivity and BH across all summary tests", required=["family", "contrast", "scope", "assumption", "spearman"])
pd.set_option("display.width", 260)
log("\n" + val.round(3).to_string(index=False))
a0 = sm[sm["assumption"].str.startswith("A0") | sm["assumption"].str.startswith("time")]
log("\n" + a0[["family", "contrast", "scope", "n_items", "spearman", "p_perm_one_sided", "sign_agreement", "mc_median_spearman", "mc_frac_positive",
               "q_BH_all_summary_tests"]].round(3).to_string(index=False, max_colwidth=60))
log("\n" + ck.round(2).to_string(index=False))
