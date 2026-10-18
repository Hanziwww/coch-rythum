"""Round 2, WP2 — the tissue-wide sample pattern (cochlea; BM for comparison).

Per tissue: pseudobulk log2-CPM of primary cell types (expression filter as in 31); each gene × cell type profile is
centred on its 6-sample mean. Tissue consensus per gene = median centred profile over the cell types expressing it
(>= 3). PCA (SVD) of the consensus matrix (genes × 6 samples) → sample scores and variance explained of PC1–3; a
cosinor fit of the PC1 scores (is the shared pattern a smooth 24-h curve?) and single-sample dominance
(max |score| / Σ |score|). Sample covariates (marrow content from 52, cells, median UMI/genes from 00, mean
dissociation-stress score from 01, immune fraction) are correlated with the PC scores (n = 6, descriptive only).
Per cell type: median correlation of its centred gene profiles with the consensus ("pattern loading").
Per gene × cell type: tissue-pattern score = correlation of the gene's centred profile with the
leave-this-cell-type-out consensus of the same gene (used to flag relaxed cochlear calls in 211).
LOTO worst time point: distribution among each cell type's nominal calls (p < 0.05, FC >= 1.25; table 33).
Outputs (results/r2_cochlea/): sample_pattern_scores.csv, sample_pattern_covariates.csv,
sample_pattern_celltypes.csv, gene_pattern_scores_<tissue>.csv, loto_worst_timepoint.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r2_cochlea"
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
incl = pd.read_csv(RESULTS / "30_pseudobulk/groups_included.csv")
incl = incl[(incl["level"] == "cell_type") & (incl["status"] == "primary")]
qc = pd.read_csv(RESULTS / "00_audit/sample_qc_summary.csv")
tech = pd.read_csv(RESULTS / "01_technical_audit/technical_audit.csv")
marrow = pd.read_csv(RESULTS / "52_marrow_decomposition/marrow_content_rhythm.csv")
cts = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
STRUCT = {"Structural", "Epithelial", "Sensory", "Glia", "Neural", "Stromal", "Bone", "Endothelial", "Vascular"}

score_rows, cov_rows, ct_rows, worst_rows = [], [], [], []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    d0 = design[(design["tissue"] == tissue) & (design["level"] == "cell_type")]
    samples = d0.drop_duplicates("sample_id").sort_values("zt")[["sample_id", "zt"]]
    zt = samples["zt"].to_numpy(float)
    prof = {}
    for g in incl.loc[incl["tissue"] == tissue, "group"]:
        cnt = pb[[f"{g}|{s}" for s in samples["sample_id"]]].to_numpy(float)
        lib = cnt.sum(0)
        cpm = cnt / lib * 1e6
        keep = ((cpm >= 10 / np.median(lib) * 1e6).sum(1) >= 3) & (cnt.sum(1) >= 20)
        l = np.log2(cpm[keep] + 1)
        prof[g] = pd.DataFrame(l - l.mean(1, keepdims=True), index=pb.index[keep], columns=samples["sample_id"])
    stack = pd.concat(prof, names=["cell_type", "gene"])
    n_ct = stack.groupby(level="gene").size()
    cons = stack.groupby(level="gene").median()[n_ct >= 3]
    U, S, Vt = np.linalg.svd(cons.to_numpy(), full_matrices=False)
    ve = S ** 2 / (S ** 2).sum()
    for k in range(3):
        sc = Vt[k] * S[k] / np.sqrt(len(cons))
        f = cosinor.fit(sc[:, None], zt)
        for s_id, z, v in zip(samples["sample_id"], zt, sc):
            score_rows.append({"tissue": tissue, "pc": k + 1, "var_explained": ve[k], "sample_id": s_id, "zt": z,
                               "score": v, "pc_cosinor_peak_zt": float(f["peak"][0]), "pc_cosinor_r2": float(f["r2"][0]),
                               "pc_cosinor_p": float(f["p"][0]),
                               "single_sample_dominance": float(np.abs(sc).max() / np.abs(sc).sum())})
    # sample covariates
    cv = samples.copy()
    cv = cv.merge(qc[["sample_id", "n_cells", "total_counts_median", "n_genes_by_counts_median"]], on="sample_id", how="left")
    st = tech[(tech["tissue"] == tissue) & (tech["metric"] == "stress_log2cpm")].groupby("sample_id")["value"].mean()
    cv["stress_score_mean"] = cv["sample_id"].map(st)
    imm = cts[cts["tissue"] == tissue].assign(imm=lambda x: ~x["cell_class"].isin(STRUCT))
    cv["immune_fraction"] = cv["sample_id"].map(imm.groupby("sample_id").apply(lambda x: x.loc[x["imm"], "n_cells"].sum() / x["n_cells"].sum()))
    if tissue == "cochlea":
        cv["marrow_vs_structural_log2"] = cv["sample_id"].map(marrow.set_index("sample_id")["log2_marrow_vs_structural"])
    for k in range(3):
        sc = np.array([r["score"] for r in score_rows if r["tissue"] == tissue and r["pc"] == k + 1])
        for c in [c for c in cv.columns if c not in ("sample_id", "zt")]:
            x = cv[c].to_numpy(float)
            if np.isfinite(x).all() and x.std() > 0:
                cov_rows.append({"tissue": tissue, "pc": k + 1, "covariate": c, "pearson_r": float(np.corrcoef(sc, x)[0, 1])})
    for c in [c for c in cv.columns if c not in ("sample_id", "zt")]:
        for s_id, v in zip(cv["sample_id"], cv[c]):
            cov_rows.append({"tissue": tissue, "pc": 0, "covariate": c, "sample_id": s_id, "value": v})
    # per cell type loading and per gene × cell type pattern score (leave-this-cell-type-out consensus)
    genes_scored = []
    for g, P in prof.items():
        others = stack.drop(index=g, level="cell_type")
        loo = others.groupby(level="gene").median()
        n_o = others.groupby(level="gene").size()
        loo = loo[n_o >= 3]
        common = P.index.intersection(loo.index)
        a = P.loc[common].to_numpy(); b = loo.loc[common].to_numpy()
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        r = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
        ct_rows.append({"tissue": tissue, "cell_type": g, "n_genes": len(common), "median_r_with_consensus": float(np.median(r)),
                        "frac_r_gt_0p8": float((r > 0.8).mean())})
        genes_scored.append(pd.DataFrame({"tissue": tissue, "cell_type": g, "gene": common, "pattern_r": r,
                                          "consensus_amp_log2": np.ptp(b, axis=1) / 2}))
    write_csv(pd.concat(genes_scored, ignore_index=True), MOD, f"gene_pattern_scores_{tissue}.csv",
              f"Correlation of each gene × cell type profile with the leave-one-cell-type-out tissue consensus ({tissue})",
              required=["tissue", "cell_type", "gene", "pattern_r"])
    t = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                    usecols=["group", "gene", "p", "ptt_fc", "loto_worst_zt"])
    t = t[(t["p"] < 0.05) & (t["ptt_fc"] >= 1.25)]
    for g, h in t.groupby("group"):
        vc = h["loto_worst_zt"].value_counts(normalize=True)
        worst_rows.append({"tissue": tissue, "cell_type": g, "n_nominal_calls": len(h),
                           **{f"frac_worst_zt{int(z)}": float(vc.get(z, 0)) for z in [3, 7, 11, 15, 19, 23]}})
    log(f"{tissue}: consensus of {len(cons)} genes; PC1 {ve[0]:.2f} of variance")

write_csv(pd.DataFrame(score_rows), MOD, "sample_pattern_scores.csv",
          "PCA of the tissue-wide consensus of centred gene profiles: sample scores, cosinor of PC scores, dominance",
          required=["tissue", "pc", "sample_id", "zt", "score"])
write_csv(pd.DataFrame(cov_rows), MOD, "sample_pattern_covariates.csv",
          "Sample covariates and their correlation with the tissue-wide pattern (n = 6, descriptive)",
          required=["tissue", "pc", "covariate"])
write_csv(pd.DataFrame(ct_rows), MOD, "sample_pattern_celltypes.csv",
          "How strongly each cell type follows the tissue-wide consensus pattern", required=["tissue", "cell_type", "median_r_with_consensus"])
write_csv(pd.DataFrame(worst_rows), MOD, "loto_worst_timepoint.csv",
          "Which left-out time point weakens nominal rhythm calls most (fraction of calls per worst ZT)",
          required=["tissue", "cell_type", "n_nominal_calls"])
log("done")
