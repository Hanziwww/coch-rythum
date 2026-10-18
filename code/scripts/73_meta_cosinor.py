"""Module 5.4e — meta-cosinor: combine our tissue-level rhythms with external replicated bulk data.

Per tissue (cochlea ↔ E1 GSE88954 sham; BM ↔ E2 GSE198770 control) and gene: cosinor coefficients (a, b) and SEs
from our tissue pseudobulk (OLS on log2 CPM, 3 residual df, t-based SEs) and from the external limma fit.
Inverse-variance fixed-effect and DerSimonian–Laird random-effects estimates per coefficient, Cochran's Q and I²,
combined amplitude/peak with a 2-df Wald test on the random-effects estimates, and a concordance flag
(|peak difference| < 3 h). Time scales: ZT for ours and E2; CT for E1 (documented; treated as equivalent).
Outputs (results/73_meta/): meta_cosinor_<tissue>.csv, meta_cosinor_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "73_meta"
MATCH = {"cochlea": ("cochlea", "sham"), "bm": ("bone_marrow", "control")}
ext_all = pd.concat([pd.read_csv(f) for f in (RESULTS / "71_external").glob("external_rhythm_*.csv")],
                    ignore_index=True)
summ = []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
    smp = design[(design["tissue"] == tissue)].drop_duplicates("sample_id").sort_values("zt")
    tot = pd.DataFrame({s: pb[[c for c in pb.columns if c.endswith("|" + s)]].sum(1) for s in smp["sample_id"]})
    cpm = tot / tot.sum(0) * 1e6
    lc = np.log2(cpm[cpm.mean(1) >= 5] + 1)
    zt = smp["zt"].to_numpy()
    X = cosinor.design(zt)
    beta, *_ = np.linalg.lstsq(X, lc.T.to_numpy(), rcond=None)
    resid = lc.T.to_numpy() - X @ beta
    s2 = (resid ** 2).sum(0) / 3
    xtx = np.linalg.inv(X.T @ X)
    ours = pd.DataFrame({"gene": lc.index, "a1": beta[1], "b1": beta[2], "se_a1": np.sqrt(s2 * xtx[1, 1]),
                         "se_b1": np.sqrt(s2 * xtx[2, 2])})
    et, ec = MATCH[tissue]
    ext = ext_all[(ext_all["tissue"] == et) & (ext_all["condition"] == ec)]
    if not len(ext):
        log(f"{tissue}: no matching external dataset")
        continue
    m = ours.merge(ext[["gene", "a", "b", "se_a", "se_b", "dataset", "time_type"]].rename(
        columns={"a": "a2", "b": "b2", "se_a": "se_a2", "se_b": "se_b2"}), on="gene")
    res = {"tissue": tissue, "gene": m["gene"], "external_dataset": m["dataset"]}
    comb = {}
    for k in ["a", "b"]:
        y = np.column_stack([m[f"{k}1"], m[f"{k}2"]])
        v = np.column_stack([m[f"se_{k}1"] ** 2, m[f"se_{k}2"] ** 2])
        w = 1 / v
        fe = (w * y).sum(1) / w.sum(1)
        Q = (w * (y - fe[:, None]) ** 2).sum(1)
        c = w.sum(1) - (w ** 2).sum(1) / w.sum(1)
        tau2 = np.maximum(0, (Q - 1) / c)
        wr = 1 / (v + tau2[:, None])
        re = (wr * y).sum(1) / wr.sum(1)
        se_re = np.sqrt(1 / wr.sum(1))
        comb[k] = (re, se_re)
        res.update({f"{k}_fixed": fe, f"{k}_random": re, f"se_{k}_random": se_re, f"Q_{k}": Q,
                    f"I2_{k}": np.where(Q > 0, np.maximum(0, (Q - 1) / np.maximum(Q, 1e-12)), 0)})
    amp, peak = cosinor.amp_phase(comb["a"][0], comb["b"][0])
    wald = (comb["a"][0] / comb["a"][1]) ** 2 + (comb["b"][0] / comb["b"][1]) ** 2
    _, pk1 = cosinor.amp_phase(m["a1"], m["b1"])
    _, pk2 = cosinor.amp_phase(m["a2"], m["b2"])
    res.update({"meta_amp_log2": amp, "meta_ptt_fc": 2 ** (2 * amp), "meta_peak_h": peak,
                "wald_chi2_2df": wald, "p_meta": stats.chi2.sf(wald, 2), "ours_peak_zt": pk1, "external_peak_h": pk2,
                "peak_diff_h": cosinor.circ_diff(pk1, pk2)})
    out = pd.DataFrame(res)
    out["q_meta"] = multipletests(out["p_meta"], method="fdr_bh")[1]
    out["concordant_phase"] = np.abs(out["peak_diff_h"]) < 3
    write_csv(out, MOD, f"meta_cosinor_{tissue}.csv",
              f"Meta-cosinor of our tissue pseudobulk and external replicated bulk ({tissue})",
              required=["tissue", "gene", "meta_ptt_fc", "meta_peak_h", "p_meta", "q_meta", "I2_a", "I2_b"])
    sig = out[out["q_meta"] < 0.05]
    summ.append({"tissue": tissue, "external_dataset": out["external_dataset"].iloc[0], "n_genes": len(out),
                 "n_q05": len(sig), "frac_concordant_among_q05": float(sig["concordant_phase"].mean()) if len(sig) else np.nan,
                 "median_I2_a_q05": float(sig["I2_a"].median()) if len(sig) else np.nan})
if summ:
    write_csv(pd.DataFrame(summ), MOD, "meta_cosinor_summary.csv", "Meta-cosinor summary per tissue",
              required=["tissue", "n_genes", "n_q05"])
log("done")
