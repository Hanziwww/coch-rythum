"""Module 0.3 — technical negative-control audit (reported only, never used as model covariates).

Per tissue × cell type × sample: median UMI depth, median detected genes, median %mt, and a dissociation-stress
score (log2 CPM of a core immediate-early/heat-shock gene set in the pseudobulk; GC/clock-responsive genes such as
Per1, Dusp1, Klf9, Mt1/2 are excluded to avoid circularity with genuine circadian outputs). Each metric is tested
for 24-h rhythmicity with the same cosinor + 60-class permutation + LOTO machinery used for biology.

Outputs (results/01_technical_audit/): technical_audit.csv, technical_audit_rhythm.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, H5AD, log, sample_table, write_csv  # noqa: E402

MOD = "01_technical_audit"
STRESS = ["Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36",
          "Hspa1a", "Hspa1b", "Hspa8", "Hsph1", "Dnajb1", "Btg2", "Klf2", "Klf4", "Klf6", "Socs3", "Ppp1r15a",
          "Cebpd", "Ccn1", "Cyr61", "Gadd45b", "Gadd45g", "Ier5"]

log("reading obs")
adata = ad.read_h5ad(H5AD, backed="r")
obs = adata.obs.copy()
samples = sample_table(obs)
incl = pd.read_csv(CACHE.parent / "results/30_pseudobulk/groups_included.csv")
incl = incl[(incl["level"] == "cell_type") & (incl["status"] != "excluded")]

med = (obs.groupby(["batch_id", "cell_type"], observed=True)
       .agg(median_umi=("total_counts", "median"), median_genes=("n_genes_by_counts", "median"),
            median_pct_mt=("pct_counts_mt", "median"))
       .reset_index().rename(columns={"batch_id": "sample_id"}))

rows = []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    genes = [g for g in STRESS if g in pb.index]
    lib = pb.sum(0)
    stress = np.log2(pb.loc[genes].sum(0) / lib * 1e6 + 1)
    for col, val in stress.items():
        group, sid = col.split("|")
        rows.append({"tissue": tissue, "cell_type": group, "sample_id": sid, "stress_log2cpm": val})
st = pd.DataFrame(rows)
tech = st.merge(med, on=["sample_id", "cell_type"], how="left").merge(samples[["sample_id", "zt"]], on="sample_id")
tech = tech.merge(incl[["tissue", "group", "status"]].rename(columns={"group": "cell_type"}), on=["tissue", "cell_type"])
tech["log2_median_umi"] = np.log2(tech["median_umi"])
long = tech.melt(id_vars=["tissue", "cell_type", "status", "sample_id", "zt"],
                 value_vars=["stress_log2cpm", "log2_median_umi", "median_genes", "median_pct_mt"],
                 var_name="metric", value_name="value")
write_csv(long.sort_values(["tissue", "cell_type", "metric", "zt"]), MOD, "technical_audit.csv",
          "Technical audit metrics per tissue × cell type × sample (negative controls; not model covariates)",
          required=["tissue", "cell_type", "sample_id", "zt", "metric", "value"])

res = []
for (tissue, ctype, metric), g in long.groupby(["tissue", "cell_type", "metric"]):
    g = g.sort_values("zt")
    y = g["value"].to_numpy()[:, None]
    zt = g["zt"].to_numpy()
    f = cosinor.fit(y, zt)
    _, _, p_perm, _ = cosinor.permutation_test(y, zt)
    lo = cosinor.loto(y, zt)
    res.append({"tissue": tissue, "cell_type": ctype, "status": g["status"].iloc[0], "metric": metric,
                "mesor": f["mesor"][0], "amplitude": f["amp"][0], "peak_zt": f["peak"][0], "r2": f["r2"][0],
                "F": f["F"][0], "p": f["p"][0], "p_perm": p_perm[0],
                "loto_phase_sd_h": lo["loto_phase_sd_h"][0], "loto_max_p": lo["loto_max_p"][0]})
res = pd.DataFrame(res)
res["q_BH_within_metric_tissue"] = res.groupby(["tissue", "metric"])["p"].transform(
    lambda p: multipletests(p, method="fdr_bh")[1])
res["flag_rhythmic_technical"] = (res["q_BH_within_metric_tissue"] < 0.1) & (res["p_perm"] <= 0.05)
write_csv(res, MOD, "technical_audit_rhythm.csv",
          "24-h cosinor tests of technical audit metrics (flags where a technical metric looks rhythmic)",
          required=["tissue", "cell_type", "metric", "p", "p_perm", "flag_rhythmic_technical"])
print(res.groupby(["tissue", "metric"])["flag_rhythmic_technical"].agg(["sum", "size"]).to_string())
log("done")
