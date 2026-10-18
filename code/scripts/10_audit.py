"""Module 0.2 — data audit of the circadian subset (read-only on the data; CSV outputs only).

Outputs (results/00_audit/):
  sample_qc_summary.csv        per-sample cell numbers and QC-metric quantiles
  gene_filter_audit.csv        gene classes present/absent (per chromosome via Ensembl BioMart, key genes)
  celltype_counts_by_sample.csv  cells per sample × cell type (long)
  label_markers.csv            marker genes used for label-support scoring (derived from gold-standard labels)
  label_support_qc.csv         per cell: own-label marker score vs best alternative (flags only, labels unchanged)
  label_support_summary.csv    per tissue × cell type: fraction of cells with weaker own-label support
"""
import io
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, H5AD, LABEL_COLS, cache_dir, log, sample_table, write_csv  # noqa: E402

MOD = "00_audit"
N_MARKERS = 25

log(f"reading {H5AD}")
adata = ad.read_h5ad(H5AD)
obs = adata.obs
samples = sample_table(obs)
label_hash = pd.util.hash_pandas_object(obs[LABEL_COLS].astype(str), index=True).sum()
log(f"{adata.n_obs} cells, {adata.n_vars} genes; gold-standard label hash {label_hash}")

# ---------------------------------------------------------------- sample QC summary
qc_cols = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"]
rows = []
for sid, sub in obs.groupby("batch_id", observed=True):
    r = {"sample_id": sid}
    for c in qc_cols:
        q = sub[c].quantile([0.1, 0.5, 0.9]).to_numpy()
        r.update({f"{c}_q10": q[0], f"{c}_median": q[1], f"{c}_q90": q[2]})
    rows.append(r)
qc = samples.merge(pd.DataFrame(rows), on="sample_id")
write_csv(qc, MOD, "sample_qc_summary.csv", "Per-sample cell numbers and QC metric quantiles",
          required=["sample_id", "tissue", "zt", "n_cells"])

# ---------------------------------------------------------------- gene filter audit
genes = pd.Index(adata.var_names)
ann_path = cache_dir(MOD) / "ensembl_mouse_genes.tsv"
if not ann_path.exists():
    q = ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query virtualSchemaName="default" '
         'formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6"><Dataset name="mmusculus_gene_ensembl" '
         'interface="default"><Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>'
         '<Attribute name="chromosome_name"/><Attribute name="gene_biotype"/></Dataset></Query>')
    url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(q)
    try:
        with urllib.request.urlopen(url, timeout=600) as resp:
            ann_path.write_bytes(resp.read())
        log("downloaded Ensembl BioMart gene table")
    except Exception as e:  # network failure is non-fatal for the audit
        log(f"BioMart download failed: {e}")
audit_rows = []
if ann_path.exists() and ann_path.stat().st_size > 1000:
    ann = pd.read_csv(ann_path, sep="\t").rename(columns={"Gene name": "gene", "Chromosome/scaffold name": "chrom",
                                                           "Gene type": "biotype"})
    ann = ann.dropna(subset=["gene"]).drop_duplicates("gene")
    ann["in_atlas"] = ann["gene"].isin(genes)
    main = [str(c) for c in list(range(1, 20))] + ["X", "Y", "MT"]
    for chrom in main:
        s = ann[(ann["chrom"] == chrom) & (ann["biotype"] == "protein_coding")]
        audit_rows.append({"category": f"chr{chrom}_protein_coding", "n_reference": len(s),
                           "n_in_atlas": int(s["in_atlas"].sum()),
                           "frac_in_atlas": s["in_atlas"].mean() if len(s) else np.nan})
    atlas_unmapped = (~genes.isin(ann["gene"])).sum()
    audit_rows.append({"category": "atlas_genes_not_in_ensembl_names", "n_reference": len(genes),
                       "n_in_atlas": int(atlas_unmapped), "frac_in_atlas": atlas_unmapped / len(genes)})
key = {"core_clock": ["Arntl", "Npas2", "Clock", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Dbp",
                      "Tef", "Hlf", "Nfil3", "Bhlhe40", "Bhlhe41", "Ciart", "Rorc", "Rora"],
       "sex_chromosome_markers": ["Xist", "Tsix", "Ddx3y", "Eif2s3y", "Kdm5d", "Uty"],
       "x_linked_relevant": ["Tsc22d3", "Pou3f4", "Rbm3", "Cybb", "Il2rg", "Btk", "Foxp3", "Cd40lg", "Tlr7", "Tlr8"],
       "gc_targets": ["Fkbp5", "Zbtb16", "Sgk1", "Dusp1", "Klf9", "Per1", "Mt2", "Cdkn1a", "Errfi1", "Pdk4"]}
for cat, gl in key.items():
    present = [g for g in gl if g in genes]
    audit_rows.append({"category": cat, "n_reference": len(gl), "n_in_atlas": len(present),
                       "frac_in_atlas": len(present) / len(gl),
                       "missing_genes": ";".join(g for g in gl if g not in genes)})
patterns = {"mitochondrial_mt-": genes.str.startswith("mt-"), "ribosomal_Rps/Rpl": genes.str.match(r"^Rp[sl]\d"),
            "Gm_predicted": genes.str.match(r"^Gm\d"), "Rik_clones": genes.str.endswith("Rik")}
for cat, m in patterns.items():
    audit_rows.append({"category": cat, "n_reference": len(genes), "n_in_atlas": int(m.sum()),
                       "frac_in_atlas": m.mean()})
write_csv(pd.DataFrame(audit_rows), MOD, "gene_filter_audit.csv",
          "Gene classes present/absent in the atlas matrix (per chromosome, clock, sex-linked, GC targets)",
          required=["category", "n_in_atlas"])

# ---------------------------------------------------------------- cell-type counts per sample
ct = (obs.groupby(["batch_id", "cell_type", "cell_class", "cell_type_abbr"], observed=True).size()
      .rename("n_cells").reset_index().rename(columns={"batch_id": "sample_id"}))
ct = samples[["sample_id", "tissue", "zt", "platform"]].merge(ct, on="sample_id")
ct["frac_of_sample"] = ct["n_cells"] / ct.groupby("sample_id")["n_cells"].transform("sum")
ct = ct.sort_values(["tissue", "cell_type", "zt"])
write_csv(ct, MOD, "celltype_counts_by_sample.csv", "Cells per sample × gold-standard cell type",
          required=["sample_id", "tissue", "zt", "cell_type", "n_cells"])

# ---------------------------------------------------------------- label-support QC (flags only)
log("label-support QC: log-normalising")
X = adata.X.tocsr().astype(np.float32)
lib = np.asarray(X.sum(1)).ravel()
Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ X
Xn.data = np.log1p(Xn.data)
norm = ad.AnnData(Xn, obs=obs[["cell_type", "tissue"]].copy(), var=pd.DataFrame(index=adata.var_names))
del X
sc.tl.rank_genes_groups(norm, "cell_type", method="t-test_overestim_var", n_genes=200, pts=True)
markers = []
for ctype in norm.obs["cell_type"].cat.categories:
    df = sc.get.rank_genes_groups_df(norm, group=ctype)
    df = df[(df["logfoldchanges"] > 1) & (df["pct_nz_group"] >= 0.25)].head(N_MARKERS)
    df.insert(0, "cell_type", ctype)
    markers.append(df)
markers = pd.concat(markers, ignore_index=True)
write_csv(markers, MOD, "label_markers.csv", "Marker genes per gold-standard cell type used for label-support scores",
          required=["cell_type", "names"])
mg = pd.Index(markers["names"].unique())
gidx = norm.var_names.get_indexer(mg)
M = np.asarray(norm.X[:, gidx].todense())
M = (M - M.mean(0)) / (M.std(0) + 1e-6)
np.clip(M, -10, 10, out=M)
types = list(norm.obs["cell_type"].cat.categories)
W = np.zeros((len(mg), len(types)), dtype=np.float32)
for j, ctype in enumerate(types):
    sel = mg.get_indexer(markers.loc[markers["cell_type"] == ctype, "names"])
    W[sel, j] = 1.0 / max(len(sel), 1)
S = M @ W
own_idx = pd.Categorical(norm.obs["cell_type"], categories=types).codes
own = S[np.arange(len(own_idx)), own_idx]
S_other = S.copy()
S_other[np.arange(len(own_idx)), own_idx] = -np.inf
best = S_other.argmax(1)
lab = pd.DataFrame({
    "cell_id": norm.obs_names, "tissue": obs["tissue"].astype(str).map({"cochlea": "cochlea", "bone marrow": "bm"}).values,
    "sample_id": obs["batch_id"].astype(str).values, "cell_type": obs["cell_type"].astype(str).values,
    "own_score": own, "best_other_type": np.array(types)[best], "best_other_score": S_other.max(1),
    "landmark_mem_margin": obs["landmark_mem_margin"].values,
})
lab["margin"] = lab["own_score"] - lab["best_other_score"]
lab["flag_weaker_own_support"] = lab["margin"] < 0
write_csv(lab, MOD, "label_support_qc.csv",
          "Per cell: gold-standard label marker score vs best alternative (flags only; labels unchanged)",
          required=["cell_id", "cell_type", "margin", "flag_weaker_own_support"])
summ = (lab.groupby(["tissue", "cell_type"])
        .agg(n_cells=("cell_id", "size"), frac_flagged=("flag_weaker_own_support", "mean"),
             median_margin=("margin", "median"),
             top_alternative=("best_other_type", lambda s: s.value_counts().index[0]))
        .reset_index())
write_csv(summ, MOD, "label_support_summary.csv", "Label-support flags summarised per tissue × cell type",
          required=["tissue", "cell_type", "frac_flagged"])

assert pd.util.hash_pandas_object(adata.obs[LABEL_COLS].astype(str), index=True).sum() == label_hash
log("labels unchanged; audit done")
