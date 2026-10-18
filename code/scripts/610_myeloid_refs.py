"""Round 6, WP-M1 — external myeloid reference signatures (D41: external sets primary) and label correspondence.

All signatures are derived from public data only (never from our atlas). Immediate-early / stress genes (330 IEG set,
Cd83, Cd69, Nr4a2/3), mitochondrial and ribosomal genes are excluded from every set.
(1) maturation (M1): Xie 2020 (GSE137540) untreated controls (wt_ctl BM1, BM2, PB2, SP2) restricted to neutrophil-lineage
    Leiden clusters (lib/neutro.py: neutrophil marker score above lymphoid and monocyte scores): log CP10k, HVG, PCA,
    diffusion pseudotime rooted at the cell with the highest early-granule score (Elane, Mpo, Ctsg, Prtn3, Azu1);
    axis = top 50 genes positively (late) and top 50 negatively (early) Spearman-correlated with reference pseudotime.
    Reference maturation quintiles (Q1 earliest … Q5 latest) serve as external states for the label correspondence.
(2) blood time-of-day programme (M2): GSE86619 blood neutrophils ZT13 vs ZT5 (77 contrast table; n = 3 per group) — top 50
    up and top 50 down genes by p among |log2FC| >= 0.5 and p < 0.05 (no gene reaches q < 0.25).
(3) tissue vs marrow programme (M2 agreeing): Ballesteros 2020 (GSE142754) sorted neutrophils restricted to the
    neutrophil lineage (same filter as Xie; lib/neutro.py) — cell-level mean log CP10k of lung + spleen vs BM neutrophils:
    top 50 tissue-high and top 50 BM-high genes, each detected in >= 10 % of the higher group's cells
    (lib/neutro.contrast_signature; summed pseudobulk let minority NK / B / mast contaminants dominate).
(4) cochlear macrophage resident axis (M8): external cochlear macrophages (GSE167078 / GSE346691 control cells with the
    transferred label Macrophage; cache/r3_query/*_mapped.h5ad): literature resident genes (Cx3cr1, Lyve1, Timd4, Folr2,
    Mrc1, Cd163, Csf1r, Fcrls, Pf4, Gas6) vs monocyte-derived / inflammatory genes (Ccr2, Ly6c2, Plac8, Chil3, Spp1, Il1b,
    Ms4a4c, Fn1) kept where detected in >= 10 % of external cochlear macrophages.
(5) label correspondence: our pooled neutrophil labels (BM and cochlea pseudobulk, log2 CPM) vs Xie maturation quintile
    profiles and Ballesteros tissue profiles (Spearman on the 2,000 most variable reference genes) — naming notes only.
Outputs (results/r6_myeloid/): myeloid_reference_signatures.csv, myeloid_label_mapping.csv, myeloid_reference_checks.csv
"""
import gzip
import io
import re
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402
from lib.neutro import contrast_signature, neutrophil_lineage  # noqa: E402

MOD = "r6_myeloid"
G = EXTERNAL / "geo"
rng = np.random.default_rng(61)
IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36", "Dusp1", "Klf2", "Klf4",
       "Klf6", "Btg2", "Ppp1r15a", "Gadd45b", "Socs3", "Cebpd", "Cd83", "Cd69", "Nr4a2", "Nr4a3", "Hspa1a", "Hspa1b"}
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]


def ok_gene(g):
    return g not in IEG and not re.match(r"^(mt-|Rp[sl]\d|Gm\d|.*Rik$)", g)


sigs, checks, maps = [], [], []


def add_sig(name, genes, direction, source, values=None):
    for i, g in enumerate(genes):
        sigs.append({"signature": name, "gene": g, "direction": direction, "source": source,
                     "value": float(values[i]) if values is not None else np.nan})


# ---------------- (1) Xie maturation axis
tf = tarfile.open(G / "GSE137540/GSE137540_RAW.tar")
xs = []
for m in tf.getmembers():
    if re.search(r"wt_ctl_(bm1|bm2|pb2|sp2)\.txt\.gz$", m.name):
        with tf.extractfile(m) as fh, gzip.open(fh, "rt") as g:
            df = pd.read_csv(g, sep=r"\s+", index_col=0)
        x = ad.AnnData(sp.csr_matrix(df.T.to_numpy(np.float32)), obs=pd.DataFrame(index=[f"{m.name}:{c}" for c in df.columns]),
                       var=pd.DataFrame(index=df.index.astype(str)))
        x.var_names_make_unique()
        x.obs["sample"] = re.search(r"wt_ctl_(\w+)\.txt", m.name).group(1)
        xs.append(x)
X = ad.concat(xs, join="inner")
sc.pp.filter_cells(X, min_genes=200)
X.layers["counts"] = X.X.copy()
sc.pp.normalize_total(X, target_sum=1e4)
sc.pp.log1p(X)
# keep the neutrophil lineage only (the PB / SP samples contain lymphocytes and monocytes)
X, note_xie = neutrophil_lineage(X)
sc.pp.highly_variable_genes(X, n_top_genes=2000, flavor="seurat_v3", layer="counts")
sc.pp.pca(X, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(X, n_neighbors=15, random_state=0)
sc.tl.diffmap(X, n_comps=10)
early = [g for g in ["Elane", "Mpo", "Ctsg", "Prtn3", "Azu1"] if g in X.var_names]
sc.tl.score_genes(X, early, score_name="early", random_state=0)
X.uns["iroot"] = int(np.argmax(X.obs["early"].to_numpy()))
sc.tl.dpt(X)
pt = X.obs["dpt_pseudotime"].to_numpy(float)
fin = np.isfinite(pt)
idx = rng.choice(np.where(fin)[0], size=min(15000, fin.sum()), replace=False)
Xs = X[idx]
L = Xs.X.toarray() if sp.issparse(Xs.X) else Xs.X
det = (L > 0).mean(0) >= 0.05
genes = Xs.var_names[det]
rho = np.array([stats.spearmanr(L[:, j], pt[idx])[0] for j in np.where(det)[0]])
rs = pd.Series(rho, index=genes).dropna()
rs = rs[[ok_gene(g) for g in rs.index]].sort_values()
add_sig("maturation_late_vs_early (Xie 2020)", rs.index[-50:][::-1].tolist(), "up (late)", "GSE137540 wt_ctl BM/PB/SP", rs.values[-50:][::-1])
add_sig("maturation_late_vs_early (Xie 2020)", rs.index[:50].tolist(), "down (early)", "GSE137540 wt_ctl BM/PB/SP", rs.values[:50])
X.obs["pt_q"] = pd.qcut(pd.Series(pt).rank(method="first"), 5, labels=[f"Xie maturation Q{i}" for i in range(1, 6)]).to_numpy()
checks.append({"reference": "Xie 2020", "check": "neutrophil-lineage filter", "value": X.n_obs, "detail": note_xie})
checks.append({"reference": "Xie 2020", "check": "cells (wt_ctl BM1/BM2/PB2/SP2)", "value": X.n_obs,
               "detail": "; ".join(f"{s} {n}" for s, n in X.obs["sample"].value_counts().items())})
checks.append({"reference": "Xie 2020", "check": "median pseudotime BM vs PB / SP (maturation increases outside the marrow)",
               "value": float(np.nanmedian(pt[X.obs["sample"].str.startswith("bm")]) - np.nanmedian(pt[~X.obs["sample"].str.startswith("bm")])),
               "detail": "negative = BM earlier"})
checks.append({"reference": "Xie 2020", "check": "top late / early genes", "value": np.nan,
               "detail": "late: " + ", ".join(rs.index[-10:][::-1]) + " | early: " + ", ".join(rs.index[:10])})
qprof = {}
for q in X.obs["pt_q"].unique():
    m_ = (X.obs["pt_q"] == q).to_numpy()
    v = np.asarray(X.layers["counts"][m_].sum(0)).ravel()
    qprof[q] = pd.Series(np.log2(v / v.sum() * 1e6 + 1), index=X.var_names)

# ---------------- (2) GSE86619 blood ZT13 vs ZT5 programme
tt = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")
tt = tt[[ok_gene(g) for g in tt.index]]
# n = 3 per group: no gene passes q < 0.25, so the programme uses nominal p < 0.05 with |log2FC| >= 0.5 (top 50 each way)
cand = tt[(tt["wt_zt13_vs_zt5_log2fc"].abs() >= 0.5) & (tt["wt_zt13_vs_zt5_p"] < 0.05)].sort_values("wt_zt13_vs_zt5_p")
up = cand[cand["wt_zt13_vs_zt5_log2fc"] > 0].head(50)
dn = cand[cand["wt_zt13_vs_zt5_log2fc"] < 0].head(50)
add_sig("blood_timer_ZT13_vs_ZT5 (GSE86619)", up.index.tolist(), "up (ZT13)", "GSE86619 WT blood neutrophils", up["wt_zt13_vs_zt5_log2fc"].to_numpy())
add_sig("blood_timer_ZT13_vs_ZT5 (GSE86619)", dn.index.tolist(), "down (ZT5)", "GSE86619 WT blood neutrophils", dn["wt_zt13_vs_zt5_log2fc"].to_numpy())
checks.append({"reference": "GSE86619", "check": "timer genes (p < 0.05, |log2FC| >= 0.5)", "value": len(cand),
               "detail": f"up {len(up)}, down {len(dn)}; top up: " + ", ".join(up.index[:8]) + "; top down: " + ", ".join(dn.index[:8])})

# ---------------- (3) Ballesteros tissue vs marrow programme
tf2 = tarfile.open(G / "GSE142754/GSE142754_RAW.tar")
names = {re.search(r"GSM\d+_(.+?)_mm10_matrix\.mtx\.gz", m.name).group(1): m.name for m in tf2.getmembers() if m.name.endswith("_matrix.mtx.gz")}
bal = []
import scipy.io  # noqa: E402
for nm, mfile in names.items():
    with tf2.extractfile(mfile) as fh:
        M = scipy.io.mmread(gzip.GzipFile(fileobj=fh)).tocsr().T.tocsr()
    gfile = [m.name for m in tf2.getmembers() if m.name.endswith(f"_{nm}_mm10_genes.tsv.gz")][0]
    with tf2.extractfile(gfile) as fh:
        gn = pd.read_csv(gzip.GzipFile(fileobj=fh), sep="\t", header=None)[1].astype(str).to_numpy()
    b = ad.AnnData(sp.csr_matrix(M, dtype=np.float32), var=pd.DataFrame(index=gn))
    b.var_names_make_unique()
    b.obs_names = [f"{nm}:{i}" for i in range(b.n_obs)]
    b.obs["sample"] = nm
    bal.append(b)
    checks.append({"reference": "Ballesteros 2020", "check": f"cells in {nm}", "value": M.shape[0], "detail": ""})
B = ad.concat(bal, join="inner")
sc.pp.filter_cells(B, min_genes=200)
B.layers["counts"] = B.X.copy()
sc.pp.normalize_total(B, target_sum=1e4)
sc.pp.log1p(B)
B, note_bal = neutrophil_lineage(B)
checks.append({"reference": "Ballesteros 2020", "check": "neutrophil-lineage filter", "value": B.n_obs,
               "detail": note_bal + "; " + "; ".join(f"{s_} {n}" for s_, n in B.obs["sample"].value_counts().items())})
pbs = {nm: pd.Series(np.asarray(B.layers["counts"][(B.obs["sample"] == nm).to_numpy()].sum(0)).ravel(), index=B.var_names)
       for nm in names}
P = pd.DataFrame(pbs).fillna(0)
Lp = np.log2(P / P.sum(0) * 1e6 + 1)
smp_ = B.obs["sample"].astype(str)
up, dn = contrast_signature(B, smp_.str.match(r"(Lung|Spleen)").to_numpy(), smp_.str.startswith("BM").to_numpy(), gene_ok=ok_gene)
add_sig("tissue_vs_marrow (Ballesteros 2020)", up.index.tolist(), "up (tissue)", "GSE142754 lung + spleen vs BM (neutrophil lineage; cell-level)", up.to_numpy())
add_sig("tissue_vs_marrow (Ballesteros 2020)", dn.index.tolist(), "down (marrow)", "GSE142754 lung + spleen vs BM (neutrophil lineage; cell-level)", dn.to_numpy())
checks.append({"reference": "Ballesteros 2020", "check": "top tissue / marrow genes", "value": np.nan,
               "detail": "tissue: " + ", ".join(up.index[:12]) + " | marrow: " + ", ".join(dn.index[:12])})
bal_prof = {f"Ballesteros {c}": Lp[c] for c in Lp.columns}

# ---------------- (4) cochlear macrophage resident axis
RES = ["Cx3cr1", "Lyve1", "Timd4", "Folr2", "Mrc1", "Cd163", "Csf1r", "Fcrls", "Pf4", "Gas6"]
INF = ["Ccr2", "Ly6c2", "Plac8", "Chil3", "Spp1", "Il1b", "Ms4a4c", "Fn1"]
det_rows = []
macs = []
for key in ["gse167078", "gse346691"]:
    p = CACHE / f"r3_query/{key}_mapped.h5ad"
    if not p.exists():
        continue
    q = ad.read_h5ad(p)
    ctrl = q.obs["condition"].astype(str).str.contains("sham|control", case=False)
    m_ = ctrl & (q.obs["label"].astype(str) == "Macrophage") & q.obs["confident"].astype(bool)
    if m_.sum() < 20:
        continue
    xx = q[m_.to_numpy()]
    macs.append(xx)
    for g in RES + INF:
        if g in xx.var_names:
            v = xx[:, g].X
            v = v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()
            det_rows.append({"dataset": key, "gene": g, "frac_detected": float((v > 0).mean()), "n_macrophages": int(m_.sum())})
dt = pd.DataFrame(det_rows)
keep = set(dt.groupby("gene")["frac_detected"].max().loc[lambda s: s >= 0.10].index) if len(dt) else set(RES + INF)
add_sig("cochlear_macrophage_resident_vs_infiltrating", [g for g in RES if g in keep], "up (resident)", "literature; detected in external cochlear macrophages (GSE167078 / GSE346691)")
add_sig("cochlear_macrophage_resident_vs_infiltrating", [g for g in INF if g in keep], "down (monocyte-derived / inflammatory)",
        "literature; detected in external cochlear macrophages (GSE167078 / GSE346691)")
for r in dt.itertuples() if len(dt) else []:
    checks.append({"reference": "external cochlear macrophages", "check": f"{r.dataset} {r.gene} detected", "value": r.frac_detected, "detail": f"n {r.n_macrophages}"})

# ---------------- (5) label correspondence (naming notes)
for t in ["bm", "cochlea"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    for lab in NEU + ["Granulocyte-Monocyte Progenitor"]:
        cols = [c for c in pb.columns if c.split("|")[0] == lab]
        v = pb[cols].sum(1)
        prof = np.log2(v / v.sum() * 1e6 + 1)
        for refname, R in [("Xie maturation quintiles", qprof), ("Ballesteros sorted neutrophils", bal_prof)]:
            Rm = pd.DataFrame(R)
            var = Rm.var(1).sort_values(ascending=False)
            gg = [g for g in var.index if g in prof.index][:2000]
            cors = {k: stats.spearmanr(prof[gg], Rm.loc[gg, k])[0] for k in Rm.columns}
            best = max(cors, key=cors.get)
            for k, r in cors.items():
                maps.append({"tissue": t, "our_label": lab, "reference": refname, "reference_state": k, "spearman_r": r, "is_best": k == best})
sg = pd.DataFrame(sigs)
write_csv(sg, MOD, "myeloid_reference_signatures.csv", "External myeloid signatures (Xie maturation; GSE86619 blood timer; Ballesteros tissue vs marrow; cochlear macrophage resident axis)",
          required=["signature", "gene", "direction", "source"])
write_csv(pd.DataFrame(maps), MOD, "myeloid_label_mapping.csv", "Our neutrophil labels vs external maturation quintiles and sorted tissue neutrophils (naming notes only)",
          required=["tissue", "our_label", "reference", "reference_state", "spearman_r", "is_best"])
write_csv(pd.DataFrame(checks), MOD, "myeloid_reference_checks.csv", "Checks of the external myeloid references", required=["reference", "check"])
log("\n" + sg.groupby(["signature", "direction"]).size().to_string())
log("\n" + pd.DataFrame(checks).head(12).to_string(max_colwidth=120))
mp = pd.DataFrame(maps)
log("\n" + mp[mp["is_best"]][["tissue", "our_label", "reference", "reference_state", "spearman_r"]].round(3).to_string())
