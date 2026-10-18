"""Round 8 (D53), step 1 — Ma 2025 timing audit and sample-level circadian phase. EXPLORATORY; every choice below was fixed
before the first run.

(A) Annotation audit: Ma et al. 2025 (PMC12876505): male CBA/CaJ, LD 12:12, ZT0 = 06:00; 2 h of 107 dB noise at ZT2–4 (day,
    DNE) or ZT14–16 (night, NNE); scRNA (CRA019820) of cochleae before (controls C1–C3) and 3 d after noise (D1–D4, N1–N3),
    11–12 mice pooled per library, FACS-sorted live cells; bulk (CRA019827) before (CMD, CMN) and 1 d after noise (NMD,
    NMN), n = 3. Collection clock times are not reported in the paper or in the GSA records. Our atlas: LD 12:12, lights on
    08:00 = ZT0. Hypotheses for collection: exposure-matched (day groups ~ZT3, night groups ~ZT15).
(B) Phase genes P = core clock / clock-output genes (lib.phase.CANONICAL_CLOCK plus Arntl, Clock, Bhlhe40, Bhlhe41) present in
    the reference fits and expressed (mean CPM > 1). P is excluded from every validation gene set in 710 / 720.
    Estimators of internal time τ per sample (our ZT = τ + O_C, O_C = our cochlear tissue offset):
      T1 template matching (maximum Pearson r over a 0.25-h grid) of the sample's centred log2 CPM clock profile with our
         cochlear tissue-pseudobulk fits (as 312 / 314); scRNA: all-cell pseudobulk; bulk: whole tissue.
      T2 the same with GSE88954 sham bulk-cochlea fits (independent reference).
      T3 composition-robust (scRNA): every structural cell type with >= 50 cells in each library, its pseudobulk clock profile
         matched to our same-type cochlear fits (33 tables); circular mean over types weighted by r (r > 0 only).
      Noise-robust variants (suffix _nr): phase genes with a time-matched noise effect are dropped (bulk: noise effect in the
         day or the night group p < 0.05; scRNA: day_noise_vs_control p < 0.05 in >= 2 structural types, 311).
    Centring: group-balanced (each gene minus the mean of its group means).
    Uncertainty: 500 gene-bootstrap replicates per sample → circular 95 % half-width; group phase = circular mean of its
    samples with a bootstrap over samples and genes.
(C) Annotation error vs biology:
    noise status per sample = mean z of time-matched noise-up genes minus noise-down genes (phase genes excluded); for the
    scRNA libraries the genes come from the bulk (noise effect q < 0.05 with the same sign in the day and the night group);
    for the bulk libraries from the scRNA (day_noise_vs_control q < 0.05, same sign in >= 3 labels).
    Biological noise phase shift bound = bulk φ(noise) − φ(control) within each time group (1 d after noise), 95 % bootstrap.
    Rule: a label is corrected only if (i) the sample's noise status sits on the other side of the midpoint between the
    control and noise group means AND (ii) its phase CI excludes its own group mean and includes another group's mean.
    Group phase differences exceeding the bulk noise-shift bound + 3 h are attributed to collection time (design).
(D) Resolution table: assumed internal phase per sample and group (T1_nr primary; T1, T2, T3 sensitivity) with CIs, and the
    contrasts in which exposure and collection time are inseparable.
Deviations after the first run (documented, D54): (1) the noise-shift bound uses the full-clock estimators T2 (independent
    reference; primary) and T1 — T1_nr dropped 9 of 18 bulk clock genes (CI up to ± 11 h) and dropping noise-responsive clock
    genes removes exactly the genes through which a phase shift appears; T1_nr stays in the tables. (2) T3 kept only one
    structural type at >= 50 cells per library; T3_30 (>= 30 cells) is added as a sensitivity. (3) The resolution table also
    reports the excess of each group difference over the bound instead of relying on the binary call alone.
Outputs (results/r8_ma_timing/): ma_timing_audit.csv, ma_sample_phase.csv, ma_group_phase.csv, ma_annotation_check.csv,
ma_timing_resolution.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r8_ma_timing"
W = phase.W
rng = np.random.default_rng(700)
GRID = np.arange(0, 24, 0.25)
NB = 500
PG = list(phase.CANONICAL_CLOCK) + ["Arntl", "Clock", "Bhlhe40", "Bhlhe41"]
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
O_E = float(off[(off["dataset"] == "GSE88954") & (off["condition"] == "sham") & (off["unit"] == "bulk")]["offset_canonical_h"].iloc[0])
offt = off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea")].set_index("unit")["offset_canonical_h"]
TP = pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").drop_duplicates("gene").set_index("gene")
E88 = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv").query("condition == 'sham'").drop_duplicates("gene").set_index("gene")
T33 = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv", usecols=["group", "gene", "a", "b"])
T33 = T33[T33["gene"].isin(PG)]


def centred(L, groups):
    gm = L.T.groupby(np.asarray(groups)).mean().T
    return L.sub(gm.mean(1), axis=0)


def tmatch(prof, A, B, o):
    best = (np.nan, -2.0)
    for t in GRID:
        pred = A * np.cos(W * (t + o)) + B * np.sin(W * (t + o))
        if prof.std() > 0 and pred.std() > 0:
            r = np.corrcoef(prof, pred - pred.mean())[0, 1]
            if r > best[1]:
                best = (t, r)
    return best


def estimate(Dc, fit, o, genes):
    """Dc: centred log2 CPM (genes × samples); returns per-sample (phase, r, half-width)"""
    g = [x for x in genes if x in Dc.index and x in fit.index]
    A, B = fit.loc[g, "a"].to_numpy(float), fit.loc[g, "b"].to_numpy(float)
    out = {}
    for s in Dc.columns:
        prof = Dc.loc[g, s].to_numpy(float)
        t0, r0 = tmatch(prof, A, B, o)
        bs = []
        for _ in range(NB):
            i = rng.integers(0, len(g), len(g))
            bs.append(tmatch(prof[i], A[i], B[i], o)[0])
        bs = np.array(bs, float)
        hw = float(np.quantile(np.abs(phase.circ_diff(bs[np.isfinite(bs)], t0)), 0.95)) if np.isfinite(bs).any() else np.nan
        out[s] = (t0, r0, hw, len(g), bs)
    return out


rows, boots = [], {}
# ---------------- scRNA
q = ad.read_h5ad(CACHE / "r3_query/ma2025_mapped.h5ad")
q.obs["label"] = q.obs["label"].astype(str)
S = sorted(q.obs["sample"].astype(str).unique())
grp_sc = q.obs.groupby("sample", observed=True)["condition"].first().astype(str).reindex(S)
smp = q.obs["sample"].astype(str).to_numpy()


def pb(mask):
    return pd.DataFrame({s: np.asarray(q.X[mask & (smp == s)].sum(0)).ravel() for s in S}, index=q.var_names)


def logcpm(M):
    cpm = M / M.sum(0) * 1e6
    return np.log2(cpm + 1), cpm.mean(1)


L_all, m_all = logcpm(pb(np.ones(q.n_obs, bool)))
P_sc = [g for g in PG if g in L_all.index and m_all[g] > 1]
de_sc = pd.read_csv(RESULTS / "r3_ma2025/pseudobulk_de.csv")
cls = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv").drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
STRUCT = [l for l in cls.index if cls[l] not in ("Myeloid", "Lymphoid", "Progenitor")]
nz = de_sc[(de_sc["contrast"] == "day_noise_vs_control") & de_sc["label"].isin(STRUCT) & de_sc["gene"].isin(PG) & (de_sc["p"] < 0.05)]
noise_clock_sc = set(nz.groupby("gene")["label"].nunique().loc[lambda s: s >= 2].index)
Dc = centred(L_all, grp_sc.to_numpy())
for est, fit, o, genes in [("T1", TP, O_C, P_sc), ("T1_nr", TP, O_C, [g for g in P_sc if g not in noise_clock_sc]), ("T2", E88, O_E, P_sc)]:
    res = estimate(Dc, fit, o, genes)
    for s, (t0, r0, hw, ng, bs) in res.items():
        rows.append({"dataset": "scRNA CRA019820", "sample": s, "group": grp_sc[s], "estimator": est, "internal_phase_h": t0, "our_zt": (t0 + O_C) % 24,
                     "template_r": r0, "ci95_halfwidth_h": hw, "n_phase_genes": ng})
        boots[("sc", est, s)] = bs
# T3 composition-robust
cnt = q.obs.groupby(["sample", "label"], observed=True).size().unstack(fill_value=0).reindex(S)
types = [l for l in STRUCT if l in cnt.columns and (cnt[l] >= 50).all() and (T33["group"] == l).sum() >= 6]
types30 = [l for l in STRUCT if l in cnt.columns and (cnt[l] >= 30).all() and (T33["group"] == l).sum() >= 6]
for variant, drop, tlist in [("T3", set(), types), ("T3_nr", noise_clock_sc, types), ("T3_30", set(), types30)]:
    per = {s: [] for s in S}
    pert = {s: [] for s in S}
    for ty in tlist:
        Lt, mt = logcpm(pb(q.obs["label"].to_numpy() == ty))
        fit = T33[T33["group"] == ty].drop_duplicates("gene").set_index("gene")
        g = [x for x in fit.index if x in Lt.index and mt[x] > 1 and x not in drop]
        if len(g) < 5:
            continue
        Dt = centred(Lt, grp_sc.to_numpy())
        res = estimate(Dt, fit, float(offt.get(ty, O_C)), g)
        for s, (t0, r0, hw, ng, bs) in res.items():
            if r0 > 0:
                per[s].append((t0, r0))
                pert[s].append(bs)
            rows.append({"dataset": "scRNA CRA019820", "sample": s, "group": grp_sc[s], "estimator": f"{variant} | {ty}", "internal_phase_h": t0,
                         "our_zt": (t0 + O_C) % 24, "template_r": r0, "ci95_halfwidth_h": hw, "n_phase_genes": ng})
    for s in S:
        if per[s]:
            ph_, w_ = np.array([x[0] for x in per[s]]), np.array([x[1] for x in per[s]])
            m, R = phase.circ_mean(ph_, w_)
            bs = np.array([phase.circ_mean(np.array([b[k] for b in pert[s]]), w_)[0] for k in range(NB)])
            rows.append({"dataset": "scRNA CRA019820", "sample": s, "group": grp_sc[s], "estimator": variant, "internal_phase_h": m,
                         "our_zt": (m + O_C) % 24, "template_r": float(w_.mean()), "ci95_halfwidth_h": float(np.quantile(np.abs(phase.circ_diff(bs, m)), 0.95)),
                         "n_phase_genes": len(per[s]), "note": f"{len(per[s])} structural types; resultant {R:.2f}"})
            boots[("sc", variant, s)] = bs
log(f"scRNA: {len(P_sc)} phase genes; noise-responsive (dropped in _nr): {sorted(noise_clock_sc)}; T3 types: {types}; T3_30 types: {types30}")

# ---------------- bulk
bulk = pd.read_csv("external/ma2025/CRA019827_bulk_counts_unstranded.tsv", sep="\t", index_col=0)
names = [f"{c}{m}{i}" for c in "CN" for m in ("MD", "MN") for i in (1, 2, 3)]
bulk = bulk[names]
grp_b = pd.Series([n[:3] for n in names], index=names)
L_b, m_b = logcpm(bulk)
P_b = [g for g in PG if g in L_b.index and m_b[g] > 1]
de_b = pd.read_csv(RESULTS / "r3_ma2025/bulk_de.csv")
nb_ = de_b[de_b["contrast"].isin(["noise_effect_day_group", "noise_effect_night_group"]) & de_b["gene"].isin(PG) & (de_b["p"] < 0.05)]
noise_clock_b = set(nb_["gene"])
Db = centred(L_b, grp_b.to_numpy())
for est, fit, o, genes in [("T1", TP, O_C, P_b), ("T1_nr", TP, O_C, [g for g in P_b if g not in noise_clock_b]), ("T2", E88, O_E, P_b)]:
    res = estimate(Db, fit, o, genes)
    for s, (t0, r0, hw, ng, bs) in res.items():
        rows.append({"dataset": "bulk CRA019827", "sample": s, "group": grp_b[s], "estimator": est, "internal_phase_h": t0, "our_zt": (t0 + O_C) % 24,
                     "template_r": r0, "ci95_halfwidth_h": hw, "n_phase_genes": ng})
        boots[("bulk", est, s)] = bs
log(f"bulk: {len(P_b)} phase genes; noise-responsive (dropped in _nr): {sorted(noise_clock_b)}")
sp = pd.DataFrame(rows)

# ---------------- group phases with bootstrap over samples and genes
gp = []
for (ds, est, g), x in sp[~sp["estimator"].str.contains(r"\|")].groupby(["dataset", "estimator", "group"]):
    m, R = phase.circ_mean(x["internal_phase_h"].to_numpy())
    key = "sc" if ds.startswith("scRNA") else "bulk"
    ss = x["sample"].tolist()
    bs = []
    for k in range(NB):
        pick = rng.choice(ss, len(ss), replace=True)
        v = np.array([boots[(key, est, s)][rng.integers(0, NB)] for s in pick])
        v = v[np.isfinite(v)]
        if len(v):
            bs.append(phase.circ_mean(v)[0])
    bs = np.array(bs)
    gp.append({"dataset": ds, "estimator": est, "group": g, "n_samples": len(ss), "internal_phase_h": m, "our_zt": (m + O_C) % 24, "resultant": R,
               "ci95_halfwidth_h": float(np.quantile(np.abs(phase.circ_diff(bs, m)), 0.95)), "_boot": bs})
gp = pd.DataFrame(gp)


def gdiff(ds, est, a, b):
    x = gp[(gp["dataset"] == ds) & (gp["estimator"] == est)].set_index("group")
    if a not in x.index or b not in x.index:
        return np.nan, np.nan
    d = float(phase.circ_diff(x.loc[a, "internal_phase_h"], x.loc[b, "internal_phase_h"]))
    n = min(len(x.loc[a, "_boot"]), len(x.loc[b, "_boot"]))
    bd = phase.circ_diff(x.loc[a, "_boot"][:n], x.loc[b, "_boot"][:n])
    return d, float(np.quantile(np.abs(phase.circ_diff(bd, d)), 0.95))


# ---------------- noise status (annotation check)
up_b = de_b[de_b["contrast"].isin(["noise_effect_day_group", "noise_effect_night_group"]) & (de_b["q_BH"] < 0.05) & ~de_b["gene"].isin(PG)]
sg = up_b.groupby("gene")["log2FC"].agg(["count", lambda v: np.sign(v).nunique()])
both = sg[(sg["count"] == 2) & (sg.iloc[:, 1] == 1)].index
dirn = up_b[up_b["gene"].isin(both)].groupby("gene")["log2FC"].mean()
sc_noise_genes = (dirn[dirn > 0].index, dirn[dirn < 0].index)
s3 = de_sc[(de_sc["contrast"] == "day_noise_vs_control") & (de_sc["q_BH"] < 0.05) & ~de_sc["gene"].isin(PG)]
k3 = s3.groupby("gene").agg(n=("label", "nunique"), s=("logFC", lambda v: np.sign(v).nunique()), m=("logFC", "mean"))
k3 = k3[(k3["n"] >= 3) & (k3["s"] == 1)]
b_noise_genes = (k3.index[k3["m"] > 0], k3.index[k3["m"] < 0])


def nscore(L, groups, genes):
    Z = centred(L, groups)
    Z = Z.div(L.std(1).replace(0, np.nan), axis=0)
    u, d = [g for g in genes[0] if g in Z.index], [g for g in genes[1] if g in Z.index]
    return Z.loc[u].mean(0) - (Z.loc[d].mean(0) if d else 0), len(u), len(d)


ac = []
for ds, L, groups, genes, ctrl_of in [("scRNA CRA019820", L_all, grp_sc, sc_noise_genes, lambda g: g == "control"),
                                      ("bulk CRA019827", L_b, grp_b, b_noise_genes, lambda g: g.startswith("C"))]:
    sc_, nu, nd = nscore(L, groups.to_numpy(), genes)
    mc = sc_[[s for s in sc_.index if ctrl_of(groups[s])]].mean()
    mn = sc_[[s for s in sc_.index if not ctrl_of(groups[s])]].mean()
    mid = (mc + mn) / 2
    pe = sp[(sp["dataset"] == ds) & (sp["estimator"] == "T1_nr")].set_index("sample")
    gpe = gp[(gp["dataset"] == ds) & (gp["estimator"] == "T1_nr")].set_index("group")
    for s in sc_.index:
        lab_ctrl = ctrl_of(groups[s])
        status_contradicts = (sc_[s] > mid) if lab_ctrl else (sc_[s] < mid)
        own = gpe.loc[groups[s], "internal_phase_h"]
        hw = pe.loc[s, "ci95_halfwidth_h"]
        ph_ = pe.loc[s, "internal_phase_h"]
        others = [g for g in gpe.index if g != groups[s] and abs(phase.circ_diff(ph_, gpe.loc[g, "internal_phase_h"])) <= hw]
        phase_contradicts = abs(phase.circ_diff(ph_, own)) > hw and len(others) > 0
        ac.append({"dataset": ds, "sample": s, "group": groups[s], "noise_score": float(sc_[s]), "control_mean": mc, "noise_mean": mn,
                   "n_noise_up_genes": nu, "n_noise_down_genes": nd, "noise_status_contradicts_label": bool(status_contradicts),
                   "internal_phase_T1_nr": ph_, "ci95_halfwidth_h": hw, "own_group_phase": own, "phase_matches_other_group": "; ".join(others),
                   "phase_contradicts_label": bool(phase_contradicts), "label_corrected": bool(status_contradicts and phase_contradicts)})
ac = pd.DataFrame(ac)

# ---------------- resolution
res = []
bounds = {}
for est in ["T2", "T1", "T1_nr"]:
    dD, hD = gdiff("bulk CRA019827", est, "NMD", "CMD")
    dN, hN = gdiff("bulk CRA019827", est, "NMN", "CMN")
    bounds[est] = max(abs(dD) + hD, abs(dN) + hN)
    res.append({"item": f"bulk: noise-associated phase shift 1 d after day noise (NMD − CMD) [{est}]", "value_h": dD, "ci95_halfwidth_h": hD})
    res.append({"item": f"bulk: noise-associated phase shift 1 d after night noise (NMN − CMN) [{est}]", "value_h": dN, "ci95_halfwidth_h": hN})
    res.append({"item": f"upper 95 % bound of a noise-associated phase shift [{est}]", "value_h": bounds[est], "ci95_halfwidth_h": np.nan,
                "interpretation": "primary bound (D54)" if est == "T2" else "sensitivity" if est == "T1" else "uninformative (too few genes; D54)"})
bound = bounds["T2"]


def interp(d, h):
    if abs(d) - h > bound:
        return f"exceeds the noise-shift bound by {abs(d) - h - bound:.1f} h (lower CI) → collection time differs (design); exposure and collection inseparable"
    if abs(d) + h <= bound:
        return "within the noise-shift bound → same collection phase not excluded"
    return "overlaps the bound → unresolved"


for a, b in [("CMN", "CMD"), ("NMN", "NMD")]:
    for est in ["T2", "T1"]:
        d, h = gdiff("bulk CRA019827", est, a, b)
        res.append({"item": f"bulk: {a} − {b} ({est})", "value_h": d, "ci95_halfwidth_h": h, "interpretation": interp(d, h)})
for a, b in [("day_noise", "control"), ("night_noise", "control"), ("night_noise", "day_noise")]:
    for est in ["T2", "T1", "T1_nr", "T3", "T3_30"]:
        d, h = gdiff("scRNA CRA019820", est, a, b)
        if np.isfinite(d):
            res.append({"item": f"scRNA: {a} − {b} ({est})", "value_h": d, "ci95_halfwidth_h": h, "interpretation": interp(d, h)})
res.append({"item": "label corrections (rule C)", "value_h": float(ac["label_corrected"].sum()), "ci95_halfwidth_h": np.nan,
            "interpretation": "; ".join(ac.loc[ac["label_corrected"], "sample"]) or "none"})
res = pd.DataFrame(res)
audit = pd.DataFrame([
    {"item": "light–dark cycle", "ma2025": "LD 12:12; ZT0 = 06:00 (paper)", "atlas": "LD 12:12; ZT0 = 08:00", "source": "PMC12876505 methods; our metadata"},
    {"item": "noise exposure", "ma2025": "2 h, 2–20 kHz, 107 dB SPL; day ZT2–4 (DNE), night ZT14–16 (NNE)", "atlas": "none", "source": "PMC12876505"},
    {"item": "scRNA design", "ma2025": "controls C1–C3 'before noise'; D1–D4, N1–N3 3 d after noise; 11–12 mice pooled per library; FACS live cells; "
     "organ of Corti + lateral wall + modiolus", "atlas": "6 ZT × 1 pooled library", "source": "PMC12876505; GSA CRA019820"},
    {"item": "bulk design", "ma2025": "CMD / CMN before noise, NMD / NMN 1 d after; n = 3 per group", "atlas": "", "source": "PMC12876505; GSA CRA019827"},
    {"item": "collection clock time", "ma2025": "not reported (paper, GSA sample / experiment / run records)", "atlas": "ZT3, 7, 11, 15, 19, 23",
     "source": "PMC12876505; data/CRA019820.xlsx; data/CRA019827.xlsx"},
    {"item": "strain / sex / age", "ma2025": "CBA/CaJ, male, 6–8 weeks (GSA: 7 weeks)", "atlas": "see memory circadian-dataset", "source": "PMC12876505; GSA"}])
write_csv(audit, MOD, "ma_timing_audit.csv", "Ma 2025 timing audit: light cycle, exposure schedule, designs and missing collection times (EXPLORATORY round 8)",
          required=["item", "ma2025", "source"])
write_csv(sp, MOD, "ma_sample_phase.csv", "Ma 2025 sample-level internal phase by estimator (T1 / T1_nr / T2 / T3 / T3_nr) with gene-bootstrap 95 % half-widths",
          required=["dataset", "sample", "group", "estimator", "internal_phase_h", "our_zt", "ci95_halfwidth_h"])
write_csv(gp.drop(columns=["_boot"]), MOD, "ma_group_phase.csv", "Ma 2025 group phases (bootstrap over samples and genes)",
          required=["dataset", "estimator", "group", "internal_phase_h", "ci95_halfwidth_h"])
write_csv(ac, MOD, "ma_annotation_check.csv", "Ma 2025 annotation check: noise status and phase vs label (rule C); corrections only when both contradict",
          required=["dataset", "sample", "group", "noise_status_contradicts_label", "phase_contradicts_label", "label_corrected"])
write_csv(res, MOD, "ma_timing_resolution.csv", "Ma 2025 timing resolution: noise-shift bound, between-group phase differences and their interpretation",
          required=["item", "value_h"])
pd.set_option("display.width", 250)
log("\n" + sp[~sp["estimator"].str.contains(r"\|")].pivot_table(index=["dataset", "group", "sample"], columns="estimator", values="internal_phase_h").round(1).to_string())
log("\n" + gp.drop(columns=["_boot"]).round(2).to_string(index=False))
log("\n" + ac.round(3).to_string(index=False))
log("\n" + res.round(2).to_string(index=False, max_colwidth=90))
