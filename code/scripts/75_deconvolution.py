"""Module 5.4c — deconvolution of external circadian bulk data with our atlas as reference (NNLS; complements
InstaPrism in 75_deconvolution_instaprism.R when that package is available).

Reference per tissue (cochlea for E1, BM for E2): mean CPM per gold-standard cell type over the 6 samples
(types with >= 20 cells in every sample); marker genes = top 50 per type by log2(own mean / max other mean) among
genes with own CPM >= 10. Bulk samples are CPM-normalised; genes scaled by the reference row mean; non-negative
least squares per sample gives UMI-share fractions, converted to cell fractions by dividing by each type's mean
UMIs per cell. Rhythm of each fraction (logit) per dataset × condition with replicates (OLS cosinor, all samples);
condition effects (sham vs ADX; control vs neutrophil depletion) by a joint model fraction ~ condition*(cos+sin).

Outputs (results/75_deconvolution/): deconvolution_reference_markers.csv, deconvolution_fractions.csv,
                                     deconvolution_abundance_rhythm.csv, deconvolution_condition_effects.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "75_deconvolution"
W = 2 * np.pi / 24
TARGETS = {"cochlea": ("cochlea",), "bm": ("bone_marrow",)}
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
prep = EXTERNAL / "prepared"
mk_rows, fr_rows = [], []
for tissue, ext_tissues in TARGETS.items():
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type")]
    ok = d.groupby("group")["n_cells"].min()
    types = ok[ok >= 20].index.tolist()
    prof, umi_per_cell = {}, {}
    for t in types:
        dd = d[d["group"] == t]
        cnt = pb[[f"{t}|{s}" for s in dd["sample_id"]]]
        prof[t] = (cnt / cnt.sum(0) * 1e6).mean(1)
        umi_per_cell[t] = float(dd["lib_size"].sum() / dd["n_cells"].sum())
    P = pd.DataFrame(prof)
    markers = []
    for t in types:
        other = P.drop(columns=t).max(1)
        spec = np.log2((P[t] + 1) / (other + 1))
        cand = spec[(P[t] >= 10)].sort_values(ascending=False).head(50)
        markers += list(cand.index)
        mk_rows += [{"tissue": tissue, "cell_type": t, "gene": g, "log2_specificity": v} for g, v in cand.items()]
    markers = pd.Index(sorted(set(markers)))
    for f in prep.glob("*__meta.csv"):
        ds = f.name.replace("__meta.csv", "")
        meta = pd.read_csv(f)
        meta = meta[meta["tissue"].isin(ext_tissues) & meta["value_type"].isin(["counts", "normcounts"])]
        if not len(meta):
            continue
        ex = pd.read_csv(prep / f"{ds}__expr.csv", index_col=0)[meta["sample"]]
        g = markers.intersection(ex.index)
        cpm = ex / ex.sum(0) * 1e6
        S = P.loc[g]
        scale = S.mean(1).replace(0, np.nan)
        S_ = S.div(scale, axis=0).fillna(0).to_numpy()
        for _, row in meta.iterrows():
            b = (cpm.loc[g, row["sample"]] / scale).fillna(0).to_numpy()
            coef, rnorm = nnls(S_, b)
            umi_frac = coef / max(coef.sum(), 1e-12)
            cell = umi_frac / np.array([umi_per_cell[t] for t in types])
            cell = cell / cell.sum()
            for t, u, c in zip(types, umi_frac, cell):
                fr_rows.append({"dataset": ds, "tissue": row["tissue"], "condition": row["condition"],
                                "sample": row["sample"], "time_h": row["time_h"], "time_type": row["time_type"],
                                "cell_type": t, "umi_fraction": u, "cell_fraction": c, "nnls_residual": rnorm,
                                "n_marker_genes": len(g)})
        log(f"{tissue} ← {ds}: {len(meta)} samples, {len(g)} marker genes")
write_csv(pd.DataFrame(mk_rows), MOD, "deconvolution_reference_markers.csv",
          "Marker genes of the atlas reference used for deconvolution", required=["tissue", "cell_type", "gene"])
fr = pd.DataFrame(fr_rows)
write_csv(fr, MOD, "deconvolution_fractions.csv", "NNLS-deconvolved cell-type fractions of external bulk samples",
          required=["dataset", "condition", "sample", "time_h", "cell_type", "cell_fraction"])
rh, ce = [], []
for (ds, tis, cond, ct), g in fr.groupby(["dataset", "tissue", "condition", "cell_type"]):
    if g["cell_fraction"].mean() < 0.002:
        continue
    y = np.log((g["cell_fraction"] + 1e-4) / (1 - g["cell_fraction"] + 1e-4)).to_numpy()
    f_ = cosinor.fit(y[:, None], g["time_h"].to_numpy())
    rh.append({"dataset": ds, "tissue": tis, "condition": cond, "cell_type": ct, "n_samples": len(g),
               "mean_fraction": g["cell_fraction"].mean(), "amp_logit": f_["amp"][0], "peak_h": f_["peak"][0],
               "F": f_["F"][0], "p": f_["p"][0]})
for (ds, tis, ct), g in fr.groupby(["dataset", "tissue", "cell_type"]):
    if g["condition"].nunique() < 2 or g["cell_fraction"].mean() < 0.002:
        continue
    g = g.assign(y=np.log((g["cell_fraction"] + 1e-4) / (1 - g["cell_fraction"] + 1e-4)),
                 c=np.cos(W * g["time_h"]), s=np.sin(W * g["time_h"]))
    full = smf.ols("y ~ C(condition) * (c + s)", data=g).fit()
    red = smf.ols("y ~ C(condition) + c + s", data=g).fit()
    f_test = full.compare_f_test(red)
    cond_terms = [k for k in full.params.index if k.startswith("C(condition)") and ":" not in k]
    ce.append({"dataset": ds, "tissue": tis, "cell_type": ct, "conditions": ";".join(sorted(g["condition"].unique())),
               "mean_shift_logit": float(full.params[cond_terms[0]]) if cond_terms else np.nan,
               "p_mean_shift": float(full.pvalues[cond_terms[0]]) if cond_terms else np.nan,
               "F_rhythm_difference": f_test[0], "p_rhythm_difference": f_test[1]})
write_csv(pd.DataFrame(rh), MOD, "deconvolution_abundance_rhythm.csv",
          "24-h rhythm of deconvolved cell fractions per external dataset × condition (with replicates)",
          required=["dataset", "condition", "cell_type", "peak_h", "p"])
if ce:
    write_csv(pd.DataFrame(ce), MOD, "deconvolution_condition_effects.csv",
              "Condition effects (mean shift and rhythm difference) on deconvolved fractions",
              required=["dataset", "cell_type", "p_mean_shift", "p_rhythm_difference"])
log("done")
