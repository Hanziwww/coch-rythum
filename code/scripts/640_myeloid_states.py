"""Round 6, WP-M4 (part 1) — external myeloid state axes, the paired BM–cochlea contrast (M1, M2, M5, M8 primaries).

Axes (610; external, D41): maturation_late_vs_early (Xie 2020); blood_timer_ZT13_vs_ZT5 (GSE86619);
tissue_vs_marrow (Ballesteros 2020); cochlear_macrophage_resident_vs_infiltrating (literature, detected in external
cochlear macrophages); mono_to_mac (curated: macrophage tlf_resident + MHC-II-low markers vs ccr2_monocyte_derived; as 66).
Cell score = scanpy score_genes(up) − score_genes(down) on log CP10k within the tissue's myeloid cells.
Per sample: median over the cells of a group (pooled neutrophils; each neutrophil label; monocytes + macrophages;
macrophages). Rhythm: cosinor with the exact 60-class permutation p, LOTO, OT distribution test (lib/disttest);
high-confidence-only rerun (00_audit margin > 0); r with the sample pattern (210 PC1) and with per-sample median UMIs.
Pseudobulk versions of every group (subset 'pseudobulk': mean gene z of log2 CPM) and within-sample centred (subset
'pseudobulk, within-sample centred': minus the median of the same score over the tissue's primary types, as 43, removing
shifts shared by all cell types of a sample); centred cochlea − BM phase contrasts in m_paired_contrast (group suffix).
Primary tests (pre-specified):
  M1  maturation axis, pooled neutrophils, per tissue → exact permutation p (Fisher across tissues in 670)
  M2  blood-timer score: cochlea − BM difference of the per-sample medians (pooled neutrophils, matched ZT) → exact
      permutation p; direction: the difference peaks in ZT12–24
  M5  mono_to_mac axis, monocytes + macrophages, per tissue → exact permutation p
  M8  resident axis in cochlear macrophages → exact permutation p (cochlea only)
M2 agreeing evidence computed here: per-tissue timer-score peaks (cochlea ZT12–24, BM ZT0–12); Ballesteros blood
neutrophils (GSE142754; neutrophil lineage only, lib/neutro.py): timer score PBZT13 vs PBZT5 cells (Mann–Whitney;
sample-level means reported); tissue_vs_marrow
score higher in cochlear than BM neutrophils at every ZT (paired sign test).
Outputs (results/r6_myeloid/): m_axes_by_sample.csv, m_axes_rhythm.csv, m_paired_contrast.csv, m_primary_states.csv,
m_m2_external_checks.csv
"""
import gzip
import re
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.abundance import loto_sd, ols, perm_p  # noqa: E402
from lib.disttest import ot_rhythm_test  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402
from lib.neutro import neutrophil_lineage  # noqa: E402

MOD = "r6_myeloid"
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
sig = pd.read_csv(RESULTS / "r6_myeloid/myeloid_reference_signatures.csv")


def sset(name):
    s_ = sig[sig["signature"] == name]
    return (s_.loc[s_["direction"].str.startswith("up"), "gene"].tolist(), s_.loc[s_["direction"].str.startswith("down"), "gene"].tolist())


AXES = {"maturation (Xie)": sset("maturation_late_vs_early (Xie 2020)"), "blood timer ZT13−ZT5 (GSE86619)": sset("blood_timer_ZT13_vs_ZT5 (GSE86619)"),
        "tissue vs marrow (Ballesteros)": sset("tissue_vs_marrow (Ballesteros 2020)"),
        "resident macrophage (cochlear)": sset("cochlear_macrophage_resident_vs_infiltrating"),
        "mono→mac (curated)": (SETS["macrophage"]["tlf_resident"] + ["Adgre1", "C1qa", "C1qb", "Apoe"], SETS["macrophage"]["ccr2_monocyte_derived"])}
GROUPS = {"pooled neutrophils": NEU, **{n: [n] for n in NEU}, "monocytes + macrophages": MONO, "Macrophage": ["Macrophage"],
          "Monocyte (Classical)": ["Monocyte (Classical)"]}
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["cell_id", "margin"]).set_index("cell_id")["margin"]


def rrow(y, zt, pc1, depth, vals=None):
    F, p, pk, amp = ols(y, zt)
    r = {"peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p, "p_perm": perm_p(y, zt), "loto_phase_sd_h": loto_sd(y, zt),
         "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1]), "r_with_median_umis": float(np.corrcoef(y, depth)[0, 1])}
    if vals is not None:
        r.update(ot_rhythm_test(vals))
    return r


A = ad.read_h5ad(H5AD)
A.obs["tcode"] = A.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
st_all = sample_table(A.obs)
bys, rh, med_store = [], [], {}
for t in ["bm", "cochlea"]:
    st = st_all[st_all["tissue"] == t].sort_values("zt")
    samples, zt = st["sample_id"].tolist(), st["zt"].to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    X = A[(A.obs["tcode"] == t) & A.obs["cell_type"].astype(str).isin([GMP] + NEU + MONO + ["Mast Cell"])].copy()
    X.obs["sample_id"] = X.obs["batch_id"].astype(str)
    X.obs["ct"] = X.obs["cell_type"].astype(str)
    umi = pd.Series(np.asarray(X.X.sum(1)).ravel(), index=X.obs_names)
    sc.pp.normalize_total(X, target_sum=1e4)
    sc.pp.log1p(X)
    hcm = (qc.reindex(X.obs_names) > 0).to_numpy()
    for ax, (up, dn) in AXES.items():
        u = [g for g in up if g in X.var_names]
        d = [g for g in dn if g in X.var_names]
        sc.tl.score_genes(X, u, score_name="_u", random_state=0)
        score = X.obs["_u"].copy()
        if d:
            sc.tl.score_genes(X, d, score_name="_d", random_state=0)
            score = score - X.obs["_d"]
        for gname, members in GROUPS.items():
            if ax == "resident macrophage (cochlear)" and gname not in ("Macrophage", "monocytes + macrophages"):
                continue
            if ax == "mono→mac (curated)" and gname not in ("monocytes + macrophages", "Macrophage", "Monocyte (Classical)"):
                continue
            for subset, mask in [("all cells", np.ones(X.n_obs, bool)), ("high-confidence cells", hcm)]:
                sel = mask & X.obs["ct"].isin(members).to_numpy()
                vals = [score[sel & (X.obs["sample_id"] == s).to_numpy()].to_numpy() for s in samples]
                if min(len(v) for v in vals) < 10:
                    continue
                y = np.array([np.median(v) for v in vals])
                depth = np.array([umi[sel & (X.obs["sample_id"] == s).to_numpy()].median() for s in samples])
                rh.append({"tissue": t, "axis": ax, "group": gname, "subset": subset, "n_up": len(u), "n_down": len(d),
                           **rrow(y, zt, pc1, depth, vals if subset == "all cells" else None)})
                if subset == "all cells":
                    med_store[(t, ax, gname)] = pd.Series(y, index=st["zt"].to_numpy(float))
                    for s_, v_, z_ in zip(samples, y, st["zt"]):
                        bys.append({"tissue": t, "axis": ax, "group": gname, "sample_id": s_, "zt": z_, "median_score": float(v_)})
    log(f"{t}: axes scored ({X.n_obs} myeloid cells)")
    del X

# ---------------- pseudobulk versions: mean gene z of log2 CPM per group; within-sample centred = minus the median of the
# same score over the tissue's primary cell types (as 43), which removes shifts shared by all cell types of a sample
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
cen_store = {}
for t in ["bm", "cochlea"]:
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    st = st_all[st_all["tissue"] == t].sort_values("zt")
    samples, zt = st["sample_id"].tolist(), st["zt"].to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")

    def zlog(types):
        ok = [ty for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples)]
        M = sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in ok)
        cpm = M / M.sum(0) * 1e6
        L = np.log2(cpm + 1)[cpm.mean(1) >= 1]
        return L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0), M.sum(0).to_numpy(float)

    Zp = {ty: zlog([ty])[0] for ty in sorted(d0.loc[d0["status"] == "primary", "group"].unique())}
    for ax, (up, dn) in AXES.items():
        def pscore(Z):
            u, d = [g for g in up if g in Z.index], [g for g in dn if g in Z.index]
            if len(u) < 3:
                return None
            v = (Z.loc[u].mean(0) - (Z.loc[d].mean(0) if len(d) >= 3 else 0)).to_numpy(float)
            return None if np.isnan(v).any() else v

        ref = [v for v in (pscore(Z) for Z in Zp.values()) if v is not None]
        med = np.median(np.vstack(ref), axis=0)
        for gname, members in GROUPS.items():
            if ax == "resident macrophage (cochlear)" and gname not in ("Macrophage", "monocytes + macrophages"):
                continue
            if ax == "mono→mac (curated)" and gname not in ("monocytes + macrophages", "Macrophage", "Monocyte (Classical)"):
                continue
            Zg, lib_ = zlog(members)
            s_ = pscore(Zg)
            if s_ is None:
                continue
            for subset, y in [("pseudobulk", s_), ("pseudobulk, within-sample centred", s_ - med)]:
                rh.append({"tissue": t, "axis": ax, "group": gname, "subset": subset, "n_up": len([g for g in up if g in Zg.index]),
                           "n_down": len([g for g in dn if g in Zg.index]), **rrow(y, zt, pc1, lib_)})
            cen_store[(t, ax, gname)] = pd.Series(s_ - med, index=zt)
    log(f"{t}: pseudobulk and within-sample-centred axis scores done")

rh = pd.DataFrame(rh)
# ---------------- paired contrast (cochlea − BM, matched ZT)
pc_rows, prim = [], []
zts = np.array([3.0, 7.0, 11.0, 15.0, 19.0, 23.0])
for (ax, g) in sorted({(k[1], k[2]) for k in med_store}):
    if ("bm", ax, g) in med_store and ("cochlea", ax, g) in med_store:
        dlt = (med_store[("cochlea", ax, g)].reindex(zts) - med_store[("bm", ax, g)].reindex(zts)).to_numpy()
        F, p, pk, amp = ols(dlt, zts)
        pc_rows.append({"axis": ax, "group": g, "peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p,
                        "p_perm": perm_p(dlt, zts), "mean_difference": float(np.mean(dlt)), "n_zt_cochlea_higher": int((dlt > 0).sum())})
for (ax, g) in sorted({(k[1], k[2]) for k in cen_store}):             # centred scores are z-scaled within tissue: phase contrast only
    if ("bm", ax, g) in cen_store and ("cochlea", ax, g) in cen_store:
        dlt = (cen_store[("cochlea", ax, g)].reindex(zts) - cen_store[("bm", ax, g)].reindex(zts)).to_numpy()
        F, p, pk, amp = ols(dlt, zts)
        pc_rows.append({"axis": ax, "group": f"{g} (pseudobulk, within-sample centred)", "peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)),
                        "amplitude": amp, "p": p, "p_perm": perm_p(dlt, zts), "mean_difference": np.nan, "n_zt_cochlea_higher": int((dlt > 0).sum())})
pcd = pd.DataFrame(pc_rows)
gp = lambda t, ax, g, sub="all cells": rh[(rh["tissue"] == t) & (rh["axis"] == ax) & (rh["group"] == g) & (rh["subset"] == sub)].iloc[0]  # noqa: E731
for t in ["bm", "cochlea"]:
    r = gp(t, "maturation (Xie)", "pooled neutrophils")
    prim.append({"hypothesis": "M1 maturation/ageing timing", "tissue": t, "primary_statistic": "exact permutation p of the per-sample median Xie maturation "
                 "axis in pooled neutrophils", "primary_p": r["p_perm"], "peak_zt": r["peak_zt"], "peak_clock": r["peak_clock"], "in_window": True,
                 "detail": f"OT p {r['ot_p_perm']:.3f}; LOTO SD {r['loto_phase_sd_h']:.2f} h; r pattern {r['r_with_sample_pattern_pc1']:+.2f}"})
    r = gp(t, "mono→mac (curated)", "monocytes + macrophages")
    prim.append({"hypothesis": "M5 monocyte→macrophage", "tissue": t, "primary_statistic": "exact permutation p of the per-sample median mono→mac axis "
                 "in monocytes + macrophages", "primary_p": r["p_perm"], "peak_zt": r["peak_zt"], "peak_clock": r["peak_clock"], "in_window": True,
                 "detail": f"OT p {r['ot_p_perm']:.3f}; LOTO SD {r['loto_phase_sd_h']:.2f} h"})
r = gp("cochlea", "resident macrophage (cochlear)", "Macrophage")
prim.append({"hypothesis": "M8 cochlear macrophage identity", "tissue": "cochlea", "primary_statistic": "exact permutation p of the per-sample median "
             "resident-macrophage axis in cochlear macrophages", "primary_p": r["p_perm"], "peak_zt": r["peak_zt"], "peak_clock": r["peak_clock"],
             "in_window": True, "detail": f"OT p {r['ot_p_perm']:.3f}; LOTO SD {r['loto_phase_sd_h']:.2f} h"})
m2 = pcd[(pcd["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (pcd["group"] == "pooled neutrophils")].iloc[0]
prim.append({"hypothesis": "M2 cochlea circulating-like (paired)", "tissue": "paired", "primary_statistic": "exact permutation p of the cosinor of the "
             "cochlea − BM difference in the per-sample median GSE86619 blood-timer score (pooled neutrophils)", "primary_p": m2["p_perm"],
             "peak_zt": m2["peak_zt"], "peak_clock": m2["peak_clock"], "in_window": bool(12 <= m2["peak_zt"] < 24),
             "detail": f"amplitude {m2['amplitude']:.3f}; cochlea higher at {int(m2['n_zt_cochlea_higher'])}/6 ZT"})

# ---------------- M2 external / agreeing checks
ex = []
for t in ["bm", "cochlea"]:
    r = gp(t, "blood timer ZT13−ZT5 (GSE86619)", "pooled neutrophils")
    want = (12, 24) if t == "cochlea" else (0, 12)
    ex.append({"check": f"{t} timer-score peak in the expected half (cochlea ZT12–24, BM ZT0–12)", "value": r["peak_zt"], "p": r["p_perm"],
               "agrees": bool(want[0] <= r["peak_zt"] < want[1])})
tsm = pcd[(pcd["axis"] == "tissue vs marrow (Ballesteros)") & (pcd["group"] == "pooled neutrophils")]
if len(tsm):
    k = int(tsm["n_zt_cochlea_higher"].iloc[0])
    ex.append({"check": "tissue-vs-marrow score higher in cochlear than BM neutrophils (ZTs of 6; sign test)", "value": k,
               "p": float(stats.binomtest(k, 6, 0.5, alternative="greater").pvalue), "agrees": k == 6})
# Ballesteros blood ZT13 vs ZT5
tf = tarfile.open(EXTERNAL / "geo/GSE142754/GSE142754_RAW.tar")
bl = []
for nm in ["PBZT5", "PBZT13_1", "PBZT13_2"]:
    mf = [m.name for m in tf.getmembers() if m.name.endswith(f"_{nm}_mm10_matrix.mtx.gz")][0]
    gf = [m.name for m in tf.getmembers() if m.name.endswith(f"_{nm}_mm10_genes.tsv.gz")][0]
    with tf.extractfile(mf) as fh:
        M = scipy.io.mmread(gzip.GzipFile(fileobj=fh)).tocsr().T.tocsr()
    with tf.extractfile(gf) as fh:
        gn = pd.read_csv(gzip.GzipFile(fileobj=fh), sep="\t", header=None)[1].astype(str).to_numpy()
    b = ad.AnnData(M.astype(np.float32), var=pd.DataFrame(index=gn))
    b.var_names_make_unique()
    b.obs_names = [f"{nm}:{i}" for i in range(b.n_obs)]
    b.obs["sample"] = nm
    bl.append(b)
B = ad.concat(bl, join="inner")
sc.pp.filter_cells(B, min_genes=200)
B.layers["counts"] = B.X.copy()
sc.pp.normalize_total(B, target_sum=1e4)
sc.pp.log1p(B)
B, note_bal = neutrophil_lineage(B)
up, dn = AXES["blood timer ZT13−ZT5 (GSE86619)"]
sc.tl.score_genes(B, [g for g in up if g in B.var_names], score_name="_u", random_state=0)
sc.tl.score_genes(B, [g for g in dn if g in B.var_names], score_name="_d", random_state=0)
B.obs["timer"] = B.obs["_u"] - B.obs["_d"]
z13 = B.obs["sample"].str.startswith("PBZT13")
mw = stats.mannwhitneyu(B.obs.loc[z13, "timer"], B.obs.loc[~z13, "timer"], alternative="greater")
means = B.obs.groupby("sample")["timer"].mean()
ex.append({"check": "Ballesteros blood neutrophils: GSE86619 timer score ZT13 > ZT5 (cells; Mann–Whitney)",
           "value": float(B.obs.loc[z13, "timer"].mean() - B.obs.loc[~z13, "timer"].mean()), "p": float(mw.pvalue),
           "agrees": bool(mw.pvalue < 0.05 and means[[s for s in means.index if s.startswith("PBZT13")]].min() > means["PBZT5"]),
           "detail": "; ".join(f"{k} {v:.3f}" for k, v in means.items()) + f"; {note_bal}"})
exd = pd.DataFrame(ex)

write_csv(pd.DataFrame(bys), MOD, "m_axes_by_sample.csv", "External myeloid axis scores per sample (median of cell scores) by group", required=["tissue", "axis", "group", "sample_id", "zt", "median_score"])
write_csv(rh, MOD, "m_axes_rhythm.csv", "External myeloid axes: rhythm per tissue × group (cosinor, exact permutation, LOTO, OT; high-confidence; pattern / depth r)",
          required=["tissue", "axis", "group", "subset", "peak_zt", "peak_clock", "p_perm"])
write_csv(pcd, MOD, "m_paired_contrast.csv", "Paired contrast: cochlea − BM difference of per-sample median axis scores (matched ZT), rhythm",
          required=["axis", "group", "peak_zt", "peak_clock", "p_perm"])
write_csv(pd.DataFrame(prim), MOD, "m_primary_states.csv", "Primary hypotheses M1, M2, M5, M8: primary statistics", required=["hypothesis", "tissue", "primary_p"])
write_csv(exd, MOD, "m_m2_external_checks.csv", "M2 agreeing evidence: per-tissue timer phases, tissue-vs-marrow contrast, Ballesteros blood ZT13 vs ZT5",
          required=["check", "value", "agrees"])
log("\n" + pd.DataFrame(prim).round(3).to_string())
log("\n" + exd.round(4).to_string())
log("\n" + rh[(rh["subset"] == "all cells") & rh["group"].isin(["pooled neutrophils", "monocytes + macrophages", "Macrophage"])]
    [["tissue", "axis", "group", "peak_zt", "p_perm", "ot_p_perm", "r_with_sample_pattern_pc1"]].round(3).to_string())
log("\n" + pcd.round(3).to_string())
