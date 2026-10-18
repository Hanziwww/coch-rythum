"""Round 6, WP-M6 — external replication of the myeloid hypotheses (agreeing methods) and exploratory cross-dataset checks.

(1) NeuMAP (GSE266518; Garcia-Seyda / Hidalgo lab, BD Rhapsody; Seurat objects exported to cache/r6_neumap by R):
    neutrophil lineage only (lib/neutro.py), cell-level contrast signatures (lib/neutro.contrast_signature, top 50 each
    way, detected in >= 10 % of cells, IEG / mito / ribo excluded):
      tissue vs marrow  = YO chip 1 (the GSE200145 young / old multi-tissue experiment): lung + liver young vs BM young
                          (same chip; the plan's GSE200145 tissue-vs-BM programme; GEO's GSE200145 matrix lacks the tags);
      old vs young BM   = YO chip 1: BM old vs BM young (exploratory);
      chronological age = time-stamp experiment, steady state: tissue neutrophils 72 h vs 36 h after labelling (same
                          location, different age; exploratory chronological-age axis). Check: BM 24 h < tissues 36 h < 72 h.
(2) The new axes scored in our myeloid cells (as 640): per-sample medians in pooled neutrophils, exact permutation p, OT,
    LOTO; cochlea − BM difference per ZT; label means (positive control).
(3) Bulk time series: GSE198770 whole BM (controls, 6 ZT × 3; neutrophil-depleted arm descriptive) and GSE25585
    peritoneal macrophages (CT, 2 × 6): axis / programme scores (mean gene z of log2 expression), cosinor F-test.
(4) Two-time-point datasets: GSE312298 sorted BM / blood neutrophils PBS night − day (ZT22 vs ZT14 by design; 315),
    GSE86619 blood neutrophils ZT13 − ZT5 (77), Ballesteros blood scRNA PBZT13 − PBZT5 (neutrophil lineage pseudobulk):
    observed vs predicted from our fits (Δ̂ = A[cos ω(t₂ − φ) − cos ω(t₁ − φ)]); Pearson r across clock genes (M7) or
    Spearman across function programmes (M4); permutation p (5,000 label shuffles). Only 4–5 clock genes are expressed in
    the external neutrophil data (>= 4 required), so the clock comparison has low power (minimum p ≈ 1/24 – 1/120).
(5) Exploratory: baboon BOM / SPL (GSE98965) axis phases after clock alignment (440 offsets); human blood neutrophils
    (GSE136632) 14:00 − 08:00 axis differences vs our prediction at ZT13 → ZT19 (441 mapping).
Agreeing-method rules (pre-specified in the plan; window ± 4 h as for M6):
  M1  GSE198770 BM maturation-axis cosinor p < 0.05 and peak within ± 4 h of our BM M1 peak (BM);
      NeuMAP / GSE200145 tissue-vs-marrow axis rhythmic in our pooled neutrophils (p <= 0.05) with peak within ± 4 h of
      the M1 peak (per tissue).
  M2  NeuMAP / GSE200145 tissue-vs-marrow score higher in cochlear than BM neutrophils at all 6 ZT (sign test).
  M3  GSE198770 BM: primary → secondary → tertiary granule programmes peak in forward order within 12 h, >= 2 of 3 with
      p < 0.05 (granule classes as promyelocyte / myelocyte / band proxies) (BM).
  M4  GSE312298 BM PBS night − day of the function programmes vs our prediction: Spearman r > 0 and p < 0.05 (per tissue).
  M5  GSE25585 mono→mac axis cosinor p < 0.05 and peak within ± 4 h of our M5 peak (per tissue).
  M7  (a) time-of-day clock genes in external neutrophils: r > 0 and p < 0.05 in >= 2 of GSE312298 / GSE86619 /
      Ballesteros (per tissue's predictions); (b) GSE25585 macrophage clock: resultant >= 0.7 and offset within ± 4 h of
      our monocyte + macrophage pseudobulk offset (645) (per tissue).
Outputs (results/r6_myeloid/): m_external_signatures.csv, m_external_axes_ours.csv, m_external_bulk.csv,
m_external_timeofday.csv, m_external_agreeing.csv
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
import scipy.sparse as sp
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.abundance import loto_sd, ols, perm_p  # noqa: E402
from lib.disttest import ot_rhythm_test  # noqa: E402
from lib.io import CACHE, EXTERNAL, H5AD, RESULTS, TISSUE_FROM_OBS, log, sample_table, write_csv  # noqa: E402
from lib.neutro import contrast_signature, neutrophil_lineage  # noqa: E402

MOD = "r6_myeloid"
rng = np.random.default_rng(66)
W = phase.W
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
CAN = phase.CANONICAL_CLOCK
IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36", "Dusp1", "Klf2", "Klf4",
       "Klf6", "Btg2", "Ppp1r15a", "Gadd45b", "Socs3", "Cebpd", "Cd83", "Cd69", "Nr4a2", "Nr4a3", "Hspa1a", "Hspa1b"}
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
sig = pd.read_csv(RESULTS / f"{MOD}/myeloid_reference_signatures.csv")
axr = pd.read_csv(RESULTS / f"{MOD}/m_axes_rhythm.csv")
fnr = pd.read_csv(RESULTS / f"{MOD}/m_function_rhythm.csv")
cko = pd.read_csv(RESULTS / f"{MOD}/m_clock_offsets.csv")
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")


def ok_gene(g):
    return g not in IEG and not re.match(r"^(mt-|Rp[sl]\d|Gm\d|.*Rik$)", g)


def sset(name):
    s_ = sig[sig["signature"] == name]
    return (s_.loc[s_["direction"].str.startswith("up"), "gene"].tolist(), s_.loc[s_["direction"].str.startswith("down"), "gene"].tolist())


def delta(amp, pk, t1, t2):
    return amp * (np.cos(W * (t2 - pk)) - np.cos(W * (t1 - pk)))


def perm_r(x, y, kind="pearson", n=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (lambda a, b: stats.pearsonr(a, b)[0]) if kind == "pearson" else (lambda a, b: stats.spearmanr(a, b)[0])
    r = f(x, y)
    null = np.array([f(x, rng.permutation(y)) for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (n + 1))


# ---------------- (1) NeuMAP signatures
def neumap(key):
    m = scipy.io.mmread(gzip.open(CACHE / f"r6_neumap/{key}_counts.mtx.gz")).tocsr().T.tocsr()
    genes = pd.read_csv(CACHE / f"r6_neumap/{key}_genes.csv")["gene"].astype(str).to_numpy()
    meta = pd.read_csv(CACHE / f"r6_neumap/{key}_meta.csv")
    X = ad.AnnData(sp.csr_matrix(m, dtype=np.float32), obs=pd.DataFrame({"sample": meta["Sample_Name"].astype(str).to_numpy()},
                                                                         index=meta["cell"].astype(str).to_numpy()), var=pd.DataFrame(index=genes))
    X.obs_names_make_unique()
    X.var_names_make_unique()
    sc.pp.filter_cells(X, min_genes=200)
    X.layers["counts"] = X.X.copy()
    sc.pp.normalize_total(X, target_sum=1e4)
    sc.pp.log1p(X)
    return neutrophil_lineage(X)


esig, checks = [], []
AXN = {}
yo, note_yo = neumap("yo1")
ts, note_ts = neumap("timestamp")
for name, Xn, note, hi, lo in [("tissue vs marrow (NeuMAP / GSE200145)", yo, note_yo, ["Lung_Young", "Liver_Young"], ["BM_Young"]),
                               ("old vs young BM (NeuMAP / GSE200145)", yo, note_yo, ["BM_Old"], ["BM_Young"]),
                               ("chronological age 72 h vs 36 h (NeuMAP time-stamp)", ts, note_ts, ["Tissues_72H_SS"], ["Tissues_36H_SS"])]:
    s_ = Xn.obs["sample"]
    up, dn = contrast_signature(Xn, s_.isin(hi).to_numpy(), s_.isin(lo).to_numpy(), gene_ok=ok_gene)
    AXN[name] = (up.index.tolist(), dn.index.tolist())
    for direction, ser in [("up", up), ("down", dn)]:
        for g, v in ser.items():
            esig.append({"signature": name, "gene": g, "direction": direction, "mean_logcp10k_difference": float(v),
                         "contrast": f"{' + '.join(hi)} vs {' + '.join(lo)}", "filter": note,
                         "n_cells_hi": int(s_.isin(hi).sum()), "n_cells_lo": int(s_.isin(lo).sum())})
    sc.tl.score_genes(Xn, up.index.tolist(), score_name="_u", random_state=0)
    sc.tl.score_genes(Xn, dn.index.tolist(), score_name="_d", random_state=0)
    sc_ = (Xn.obs["_u"] - Xn.obs["_d"]).groupby(Xn.obs["sample"]).mean()
    checks.append({"check": f"{name}: mean score per NeuMAP sample", "detail": "; ".join(f"{k} {v:.3f}" for k, v in sc_.items()),
                   "agrees": bool(sc_.get("BM_24H_SS", -np.inf) < sc_.get("Tissues_36H_SS", np.nan) < sc_.get("Tissues_72H_SS", np.nan))
                   if "time-stamp" in name else np.nan})
log("NeuMAP signatures built: " + "; ".join(f"{k} ({len(v[0])}/{len(v[1])})" for k, v in AXN.items()))
del yo, ts

# ---------------- (2) new axes in our myeloid cells
A = ad.read_h5ad(H5AD)
A.obs["tcode"] = A.obs["tissue"].astype(str).map(TISSUE_FROM_OBS)
st_all = sample_table(A.obs)
ours, med_store = [], {}
for t in ["bm", "cochlea"]:
    st = st_all[st_all["tissue"] == t].sort_values("zt")
    samples, zt = st["sample_id"].tolist(), st["zt"].to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    X = A[(A.obs["tcode"] == t) & A.obs["cell_type"].astype(str).isin([GMP] + NEU + MONO)].copy()
    X.obs["sample_id"] = X.obs["batch_id"].astype(str)
    X.obs["ct"] = X.obs["cell_type"].astype(str)
    sc.pp.normalize_total(X, target_sum=1e4)
    sc.pp.log1p(X)
    for ax, (u, d) in AXN.items():
        sc.tl.score_genes(X, [g for g in u if g in X.var_names], score_name="_u", random_state=0)
        sc.tl.score_genes(X, [g for g in d if g in X.var_names], score_name="_d", random_state=0)
        score = X.obs["_u"] - X.obs["_d"]
        for ct_, v in score.groupby(X.obs["ct"]).mean().items():
            ours.append({"tissue": t, "axis": ax, "group": ct_, "kind": "label mean (positive control)", "value": float(v)})
        sel = X.obs["ct"].isin(NEU).to_numpy()
        vals = [score[sel & (X.obs["sample_id"] == s).to_numpy()].to_numpy() for s in samples]
        y = np.array([np.median(v) for v in vals])
        F, p, pk, amp = ols(y, zt)
        ours.append({"tissue": t, "axis": ax, "group": "pooled neutrophils", "kind": "rhythm of per-sample medians", "peak_zt": pk,
                     "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p, "p_perm": perm_p(y, zt), "loto_phase_sd_h": loto_sd(y, zt),
                     "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1]), **ot_rhythm_test(vals),
                     "values_by_zt": "; ".join(f"{a:.3f}" for a in y)})
        med_store[(t, ax)] = pd.Series(y, index=zt)
    log(f"{t}: NeuMAP axes scored")
    del X
zts = np.array([3.0, 7.0, 11.0, 15.0, 19.0, 23.0])
for ax in AXN:
    dlt = (med_store[("cochlea", ax)].reindex(zts) - med_store[("bm", ax)].reindex(zts)).to_numpy()
    k = int((dlt > 0).sum())
    ours.append({"tissue": "paired (cochlea − BM)", "axis": ax, "group": "pooled neutrophils", "kind": "cochlea − BM per ZT",
                 "value": float(dlt.mean()), "n_zt_cochlea_higher": k, "p_sign_test": float(stats.binomtest(k, 6, 0.5, alternative="greater").pvalue),
                 "values_by_zt": "; ".join(f"{a:.3f}" for a in dlt)})
ours = pd.DataFrame(ours)
del A

# ---------------- (3) bulk time series
gene_sets = {"maturation (Xie)": sset("maturation_late_vs_early (Xie 2020)"),
             "blood timer ZT13−ZT5 (GSE86619)": sset("blood_timer_ZT13_vs_ZT5 (GSE86619)"),
             "tissue vs marrow (Ballesteros)": sset("tissue_vs_marrow (Ballesteros 2020)"),
             "resident macrophage (cochlear)": sset("cochlear_macrophage_resident_vs_infiltrating"),
             "mono→mac (curated)": (SETS["macrophage"]["tlf_resident"] + ["Adgre1", "C1qa", "C1qb", "Apoe"], SETS["macrophage"]["ccr2_monocyte_derived"]),
             **{k: v for k, v in AXN.items()},
             **{f"myeloid:{k}": (v, []) for k, v in SETS["myeloid_function"].items()}}


def bulk_scores(L, sets, min_genes=3):
    Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
    out = {}
    for nm, (u, d) in sets.items():
        uu, dd = [g for g in u if g in Z.index], [g for g in d if g in Z.index]
        if len(uu) < min_genes:
            continue
        s_ = Z.loc[uu].mean(0) - (Z.loc[dd].mean(0) if len(dd) >= min_genes else 0)
        out[nm] = (s_, len(uu), len(dd))
    return out


bulk = []
meta = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__meta.csv")
ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__expr.csv", index_col=0)
for cond in ["control", "neutrophil_depleted"]:
    m = meta[meta["condition"] == cond]
    L = np.log2(ex[m["sample"]].astype(float) + 1)
    L = L[L.mean(1) >= 1]
    for nm, (s_, nu, nd) in bulk_scores(L, gene_sets).items():
        f = cosinor.fit(s_.to_numpy(float), m["time_h"].to_numpy(float))
        bulk.append({"dataset": "GSE198770 bone marrow", "condition": cond, "time_type": "ZT", "score": nm, "n_up": nu, "n_down": nd,
                     "peak": float(f["peak"][0]), "amplitude": float(f["amp"][0]), "p": float(f["p"][0])})
m25 = pd.read_csv(EXTERNAL / "prepared_r2/GSE25585__meta.csv")
e25 = pd.read_csv(EXTERNAL / "prepared_r2/GSE25585__expr.csv", index_col=0)[m25["sample"]].astype(float)
e25 = e25[e25.mean(1) >= np.quantile(e25.mean(1), 0.25)]
for nm, (s_, nu, nd) in bulk_scores(e25, gene_sets).items():
    f = cosinor.fit(s_.to_numpy(float), m25["time_h"].to_numpy(float))
    bulk.append({"dataset": "GSE25585 peritoneal macrophages", "condition": "WT", "time_type": "CT", "score": nm, "n_up": nu, "n_down": nd,
                 "peak": float(f["peak"][0]), "amplitude": float(f["amp"][0]), "p": float(f["p"][0])})
f25 = cosinor.fit(e25.T.to_numpy(), m25["time_h"].to_numpy(float))
c25 = pd.DataFrame({"gene": e25.index, "peak": f25["peak"], "p": f25["p"]})
c25 = c25[c25["gene"].isin(CAN) & (c25["p"] < 0.05)]
mac_off, mac_R = (phase.circ_mean(phase.circ_diff(c25["peak"].to_numpy(), c25["gene"].map(CAN).to_numpy()) % 24) if len(c25) >= 4 else (np.nan, np.nan))
mac_off = float(phase.circ_diff(mac_off, 0)) if np.isfinite(mac_off) else np.nan
bulk.append({"dataset": "GSE25585 peritoneal macrophages", "condition": "WT", "time_type": "CT", "score": "clock offset vs canonical (genes p < 0.05)",
             "n_up": len(c25), "peak": mac_off, "amplitude": mac_R, "p": np.nan})
# baboon (exploratory; human orthologues, clock-aligned)
hom = pd.read_csv(EXTERNAL / "resources/mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
m2h = {m[0]: h[0] for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1}
hsets = {nm: ([m2h[g] for g in u if g in m2h], [m2h[g] for g in d if g in m2h]) for nm, (u, d) in gene_sets.items()}
raw = pd.read_csv(EXTERNAL / "geo/GSE98965/GSE98965_baboon_tissue_expression_FPKM.csv.gz")
raw = raw.dropna(subset=["Symbol"]).drop_duplicates("Symbol").set_index("Symbol").drop(columns="EnsemblID")
offb = pd.read_csv(RESULTS / "r4_crossspecies/baboon_clock_offsets.csv").set_index("baboon_tissue")["offset_used_h"]
ours_off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
OFF = {t: float(ours_off[(ours_off["dataset"] == "this_atlas") & (ours_off["tissue"] == t) & (ours_off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
       for t in ["bm", "cochlea"]}
for bt in ["BOM", "SPL"]:
    cc = [c for c in raw.columns if re.match(rf"^{bt}[._]ZT(\d+)", c)]
    tt = np.array([float(re.match(rf"^{bt}[._]ZT(\d+)", c).group(1)) for c in cc])
    x = raw[cc]
    L = np.log2(x[x.mean(1) >= 1] + 1)
    for nm, (s_, nu, nd) in bulk_scores(L, hsets).items():
        f = cosinor.fit(s_.to_numpy(float), tt)
        bulk.append({"dataset": f"baboon {bt} (GSE98965)", "condition": "diurnal primate", "time_type": "ZT", "score": nm, "n_up": nu, "n_down": nd,
                     "peak": float(f["peak"][0]), "amplitude": float(f["amp"][0]), "p": float(f["p"][0]),
                     "aligned_peak": float(phase.wrap24(f["peak"][0] - offb[bt]))})
bulk = pd.DataFrame(bulk)
# our peaks for comparison (pooled neutrophils unless macrophage axes)
def our_peak(t, score):
    if score.startswith("myeloid:"):
        r = fnr[(fnr["tissue"] == t) & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred") & (fnr["unit"] == "neutrophils + monocytes (pooled)") & (fnr["programme"] == score)]
    elif score in AXN:
        r = ours[(ours["tissue"] == t) & (ours["axis"] == score) & (ours["kind"] == "rhythm of per-sample medians")]
    else:
        g = "monocytes + macrophages" if score in ("mono→mac (curated)", "resident macrophage (cochlear)") else "pooled neutrophils"
        r = axr[(axr["tissue"] == t) & (axr["axis"] == score) & (axr["group"] == g) & (axr["subset"] == "all cells")]
    return (float(r["peak_zt"].iloc[0]), float(r["p_perm"].iloc[0]), float(r["amplitude"].iloc[0])) if len(r) else (np.nan, np.nan, np.nan)


for t in ["bm", "cochlea"]:
    pk = [our_peak(t, s_) for s_ in bulk["score"]]
    bulk[f"our_{t}_peak_zt"] = [p_[0] for p_ in pk]
    bulk[f"our_{t}_p_perm"] = [p_[1] for p_ in pk]
    bulk[f"diff_vs_our_{t}_h"] = phase.circ_diff(bulk["peak"], bulk[f"our_{t}_peak_zt"])
    bulk[f"aligned_diff_vs_our_{t}_h"] = phase.circ_diff(bulk.get("aligned_peak", pd.Series(np.nan, index=bulk.index)),
                                                         phase.wrap24(bulk[f"our_{t}_peak_zt"] - OFF[t]))

# ---------------- (4) two-time-point datasets
tod = []
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
clockfit = {}
for t in ["bm", "cochlea"]:
    s0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")].drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    for unit, types in [("myeloid (pooled)", [GMP] + NEU + MONO)]:
        M = sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples))
        cpm = M / M.sum(0) * 1e6
        g = [x for x in CAN if x in cpm.index and cpm.loc[x].mean() >= 1]
        f = cosinor.fit(np.log2(cpm.loc[g] + 1).T.to_numpy(), zt)
        clockfit[(t, unit)] = pd.DataFrame({"amp": f["amp"], "peak": f["peak"], "p": f["p"]}, index=g)
de = pd.read_csv(RESULTS / "r3_response/gse312298_de.csv")
de = de[de["contrast"] == "night_minus_day_PBS"]
tt86 = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")["wt_zt13_vs_zt5_log2fc"]
# Ballesteros blood clock genes (neutrophil lineage pseudobulk)
tf = tarfile.open(EXTERNAL / "geo/GSE142754/GSE142754_RAW.tar")
bl = []
for nm in ["PBZT5", "PBZT13_1", "PBZT13_2"]:
    mf = [m.name for m in tf.getmembers() if m.name.endswith(f"_{nm}_mm10_matrix.mtx.gz")][0]
    gf = [m.name for m in tf.getmembers() if m.name.endswith(f"_{nm}_mm10_genes.tsv.gz")][0]
    with tf.extractfile(mf) as fh:
        M = scipy.io.mmread(gzip.GzipFile(fileobj=fh)).tocsr().T.tocsr()
    with tf.extractfile(gf) as fh:
        gn = pd.read_csv(gzip.GzipFile(fileobj=fh), sep="\t", header=None)[1].astype(str).to_numpy()
    b = ad.AnnData(sp.csr_matrix(M, dtype=np.float32), var=pd.DataFrame(index=gn))
    b.var_names_make_unique()
    b.obs_names = [f"{nm}:{i}" for i in range(b.n_obs)]
    b.obs["sample"] = nm
    bl.append(b)
B = ad.concat(bl, join="inner")
sc.pp.filter_cells(B, min_genes=200)
B.layers["counts"] = B.X.copy()
sc.pp.normalize_total(B, target_sum=1e4)
sc.pp.log1p(B)
B, note_b = neutrophil_lineage(B)
pbB = pd.DataFrame({nm: np.asarray(B.layers["counts"][(B.obs["sample"] == nm).to_numpy()].sum(0)).ravel() for nm in ["PBZT5", "PBZT13_1", "PBZT13_2"]},
                   index=B.var_names)
LB = np.log2(pbB / pbB.sum(0) * 1e6 + 1)
balB = (LB[["PBZT13_1", "PBZT13_2"]].mean(1) - LB["PBZT5"])[LB.mean(1) >= 1]
checks.append({"check": "Ballesteros blood neutrophil-lineage filter", "detail": note_b + "; " + "; ".join(f"{k} {v}" for k, v in B.obs["sample"].value_counts().items())})
del B
ext_obs = {"GSE312298 BM sorted neutrophils (ZT22 − ZT14)": (de[de["tissue"] == "BM"].set_index("gene")["log2FC"], 14.0, 22.0),
           "GSE312298 blood sorted neutrophils (ZT22 − ZT14)": (de[de["tissue"] == "PB"].set_index("gene")["log2FC"], 14.0, 22.0),
           "GSE86619 blood neutrophils (ZT13 − ZT5)": (tt86, 5.0, 13.0),
           "Ballesteros blood neutrophils scRNA (ZT13 − ZT5)": (balB, 5.0, 13.0)}
conc = []
for t in ["bm", "cochlea"]:
    cf = clockfit[(t, "myeloid (pooled)")]
    for dsn, (obs, t1, t2) in ext_obs.items():
        g = [x for x in cf.index if x in obs.index]
        if len(g) < 4:                      # neutrophils express few clock genes (4–5 per dataset): low power, flagged
            continue
        pred = delta(cf.loc[g, "amp"].to_numpy(), cf.loc[g, "peak"].to_numpy(), t1, t2)
        r, p = perm_r(pred, obs[g].to_numpy(float))
        can = np.array([delta(1.0, CAN[x], t1, t2) for x in g])
        rc, pc = perm_r(can, obs[g].to_numpy(float))
        conc.append({"analysis": "clock genes (M7)", "tissue_predictions": t, "dataset": dsn, "n_items": len(g), "genes": ";".join(g), "r": r, "p_perm": p,
                     "r_canonical_prediction": rc, "p_canonical": pc})
        for x, a_, b_ in zip(g, pred, obs[g].to_numpy(float)):
            tod.append({"analysis": "clock genes (M7)", "tissue_predictions": t, "dataset": dsn, "item": x, "predicted": float(a_), "observed": float(b_)})
    fr_ = fnr[(fnr["tissue"] == t) & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred") & (fnr["unit"] == "neutrophils + monocytes (pooled)")].set_index("programme")
    for dsn, (obs, t1, t2) in ext_obs.items():
        rows = []
        for prog, r_ in fr_.iterrows():
            genes = SETS["myeloid_function"].get(prog.split(":", 1)[1], []) if prog.startswith("myeloid:") else []
            g = [x for x in genes if x in obs.index]
            if len(g) >= 3:
                rows.append((prog, delta(r_["amplitude"], r_["peak_zt"], t1, t2), float(obs[g].mean()), len(g)))
        if len(rows) >= 5:
            r, p = perm_r([x[1] for x in rows], [x[2] for x in rows], kind="spearman")
            conc.append({"analysis": "function programmes (M4)", "tissue_predictions": t, "dataset": dsn, "n_items": len(rows), "r": r, "p_perm": p,
                         "sign_agreement": float(np.mean([np.sign(x[1]) == np.sign(x[2]) for x in rows]))})
            for prog, a_, b_, n_ in rows:
                tod.append({"analysis": "function programmes (M4)", "tissue_predictions": t, "dataset": dsn, "item": prog, "predicted": float(a_),
                            "observed": float(b_), "n_genes": n_
                            })
# human blood neutrophils (exploratory; 441 mapping 08:00 → ZT13, 14:00 → ZT19)
h = pd.read_csv(EXTERNAL / "geo/GSE136632/GSE136632_normalized_cpm_humanRNAseq.csv.gz")
h = h.dropna(subset=["ext_gene"]).drop_duplicates("ext_gene").set_index("ext_gene")
donors = ["C", "E", "J"]
cols = [f"{d}_CT{t_}" for d in donors for t_ in (8, 14)]
Lh = np.log2(h[cols].astype(float) + 1)
Lh = Lh[h[cols].astype(float).mean(1) >= 5]
hsc = bulk_scores(Lh, hsets)
for nm, (s_, nu, nd) in hsc.items():
    dd = np.array([s_[f"{d}_CT14"] - s_[f"{d}_CT8"] for d in donors])
    for t in ["bm", "cochlea"]:
        pk, _, amp = our_peak(t, nm)
        tod.append({"analysis": "axes / programmes in human blood neutrophils (exploratory)", "tissue_predictions": t,
                    "dataset": "GSE136632 human blood neutrophils (14:00 − 08:00; ZT13 → ZT19)", "item": nm,
                    "predicted": float(delta(amp, pk, 13.0, 19.0)) if np.isfinite(pk) else np.nan, "observed": float(dd.mean()),
                    "observed_p_paired": float(stats.ttest_1samp(dd, 0).pvalue), "n_genes": nu + nd})
conc = pd.DataFrame(conc)

# ---------------- agreeing methods
agr = []
def bget(ds, cond, score):
    r = bulk[(bulk["dataset"] == ds) & (bulk["condition"] == cond) & (bulk["score"] == score)]
    return r.iloc[0] if len(r) else None


for t in ["bm", "cochlea"]:
    m1pk = our_peak(t, "maturation (Xie)")[0]
    if t == "bm":
        r = bget("GSE198770 bone marrow", "control", "maturation (Xie)")
        agr.append({"hypothesis": "M1", "tissue": t, "method": "GSE198770 BM maturation-axis phase", "agrees": bool(r["p"] < 0.05 and abs(phase.circ_diff(r["peak"], m1pk)) <= 4),
                    "detail": f"p {r['p']:.3g}; peak ZT{r['peak']:.1f} vs ours ZT{m1pk:.1f}"})
        g3 = [bget("GSE198770 bone marrow", "control", f"myeloid:granules_{k}") for k in ["primary", "secondary", "tertiary"]]
        pk3 = np.array([x["peak"] for x in g3])
        steps = phase.wrap24(np.diff(pk3))
        ok_order = bool((steps.sum() < 12))
        nsig = int(sum(x["p"] < 0.05 for x in g3))
        agr.append({"hypothesis": "M3", "tissue": t, "method": "GSE198770 BM granule-class order (primary → secondary → tertiary)",
                    "agrees": ok_order and nsig >= 2, "detail": f"peaks ZT{pk3[0]:.1f} → {pk3[1]:.1f} → {pk3[2]:.1f} (forward span {steps.sum():.1f} h); "
                                                                 "p " + ", ".join(f"{x['p']:.3g}" for x in g3)})
    r = ours[(ours["tissue"] == t) & (ours["axis"] == "tissue vs marrow (NeuMAP / GSE200145)") & (ours["kind"] == "rhythm of per-sample medians")].iloc[0]
    agr.append({"hypothesis": "M1", "tissue": t, "method": "external tissue-vs-marrow (ageing / circulating) axis (NeuMAP / GSE200145) rhythm",
                "agrees": bool(r["p_perm"] <= 0.05 and abs(phase.circ_diff(r["peak_zt"], m1pk)) <= 4),
                "detail": f"p {r['p_perm']:.3f}; peak ZT{r['peak_zt']:.1f} vs M1 ZT{m1pk:.1f}"})
    c = conc[(conc["analysis"] == "function programmes (M4)") & (conc["tissue_predictions"] == t) & (conc["dataset"].str.startswith("GSE312298 BM"))]
    if len(c):
        c = c.iloc[0]
        agr.append({"hypothesis": "M4", "tissue": t, "method": "GSE312298 BM programme night − day vs prediction", "agrees": bool(c["r"] > 0 and c["p_perm"] < 0.05),
                    "detail": f"Spearman {c['r']:.2f}, p {c['p_perm']:.3f}, n {c['n_items']}"})
    m5pk = our_peak(t, "mono→mac (curated)")[0]
    r = bget("GSE25585 peritoneal macrophages", "WT", "mono→mac (curated)")
    agr.append({"hypothesis": "M5", "tissue": t, "method": "GSE25585 macrophage mono→mac phase", "agrees": bool(r["p"] < 0.05 and abs(phase.circ_diff(r["peak"], m5pk)) <= 4),
                "detail": f"p {r['p']:.3g}; peak CT{r['peak']:.1f} vs ours ZT{m5pk:.1f}"})
    cc = conc[(conc["analysis"] == "clock genes (M7)") & (conc["tissue_predictions"] == t) & conc["dataset"].isin(
        ["GSE312298 BM sorted neutrophils (ZT22 − ZT14)", "GSE86619 blood neutrophils (ZT13 − ZT5)", "Ballesteros blood neutrophils scRNA (ZT13 − ZT5)"])]
    k = int(((cc["r"] > 0) & (cc["p_perm"] < 0.05)).sum())
    agr.append({"hypothesis": "M7", "tissue": t, "method": "time-of-day clock genes in external neutrophils (GSE312298 / GSE86619 / Ballesteros)", "agrees": k >= 2,
                "detail": "; ".join(f"{d.split(' (')[0]} r {r_:.2f} p {p_:.3f}" for d, r_, p_ in zip(cc["dataset"], cc["r"], cc["p_perm"]))})
    mo = cko[(cko["tissue"] == t) & (cko["unit"] == "monocytes + macrophages (pooled)")]
    mo = float(mo["offset_h"].iloc[0]) if len(mo) else np.nan
    agr.append({"hypothesis": "M7", "tissue": t, "method": "GSE25585 macrophage clock offset", "agrees": bool(np.isfinite(mac_off) and mac_R >= 0.7 and abs(phase.circ_diff(mac_off, mo)) <= 4),
                "detail": f"GSE25585 offset {mac_off:+.1f} h (R {mac_R:.2f}, {len(c25)} genes) vs ours {mo:+.1f} h"})
r = ours[(ours["tissue"] == "paired (cochlea − BM)") & (ours["axis"] == "tissue vs marrow (NeuMAP / GSE200145)")].iloc[0]
agr.append({"hypothesis": "M2", "tissue": "paired", "method": "external tissue-vs-marrow (NeuMAP / GSE200145) higher in cochlear than BM neutrophils at every ZT",
            "agrees": bool(r["n_zt_cochlea_higher"] == 6), "detail": f"{int(r['n_zt_cochlea_higher'])}/6 ZT; sign-test p {r['p_sign_test']:.3f}"})

write_csv(pd.DataFrame(esig), MOD, "m_external_signatures.csv", "NeuMAP (GSE266518 / GSE200145) neutrophil-lineage contrast signatures: tissue vs marrow, old vs young BM, "
          "chronological age 72 h vs 36 h", required=["signature", "gene", "direction", "mean_logcp10k_difference"])
write_csv(ours, MOD, "m_external_axes_ours.csv", "NeuMAP axes in our myeloid cells: label means (positive control), rhythm of per-sample medians in pooled "
          "neutrophils, cochlea − BM per ZT", required=["tissue", "axis", "group", "kind"])
write_csv(bulk, MOD, "m_external_bulk.csv", "External bulk time series (GSE198770 BM, GSE25585 macrophages, baboon BOM / SPL): myeloid axis and programme "
          "phases vs ours (raw and clock-aligned)", required=["dataset", "condition", "score", "peak", "p"])
write_csv(pd.concat([conc.assign(level="summary"), pd.DataFrame(tod).assign(level="item")], ignore_index=True), MOD, "m_external_timeofday.csv",
          "Two-time-point datasets (GSE312298, GSE86619, Ballesteros, human GSE136632): observed vs predicted time-of-day differences of clock genes and "
          "function programmes", required=["analysis", "tissue_predictions", "dataset", "level"])
write_csv(pd.concat([pd.DataFrame(agr), pd.DataFrame(checks).assign(hypothesis="check")], ignore_index=True), MOD, "m_external_agreeing.csv",
          "External agreeing-method verdicts per hypothesis × tissue (pre-specified rules) and NeuMAP / Ballesteros checks",
          required=["hypothesis", "method", "agrees", "detail"])
log("\n" + ours[ours["kind"] != "label mean (positive control)"].round(3).to_string(max_colwidth=60))
log("\n" + ours[ours["kind"] == "label mean (positive control)"].pivot_table(index="group", columns=["tissue", "axis"], values="value").round(2).to_string())
log("\n" + conc.round(3).to_string())
log("\n" + bulk[bulk["dataset"].str.startswith("GSE")].round(3).to_string(max_colwidth=50))
log("\n" + pd.DataFrame(agr).to_string(max_colwidth=110))
log("\n" + pd.DataFrame(checks).to_string(max_colwidth=200))
