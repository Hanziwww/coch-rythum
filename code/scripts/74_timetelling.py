"""Module 0.4b / 5.4g — molecular time-telling of our samples from external circadian references.

Template phase matching: for a reference (dataset × tissue × condition) take genes rhythmic there (q < 0.01,
FC >= 1.5) that are expressed in the target; centre the target's log expression per gene across its samples; for each
sample choose the time τ that maximises the amplitude-weighted correlation between the centred profile and
cos(ω(τ − φ_g)). Validation: the same estimator applied leave-one-sample-out to the reference's own samples
(reference phases refitted without the left-out sample) gives the expected accuracy.
Targets: our tissue-level pseudobulk (all cells per sample) and the cell-class pseudobulks.

Outputs (results/74_timetelling/): reference_validation.csv, sample_time_prediction.csv, timetelling_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "74_timetelling"
W = 2 * np.pi / 24
GRID = np.arange(0, 24, 0.1)


def predict(Z, amp, phase):
    """Z: samples × genes centred log expression; returns best τ per sample and its correlation."""
    T = np.cos(W * (GRID[:, None] - phase[None, :])) * amp[None, :]          # grid × genes
    Tc = T - T.mean(1, keepdims=True)
    Zc = Z - Z.mean(1, keepdims=True)
    num = Zc @ Tc.T
    den = np.linalg.norm(Zc, axis=1)[:, None] * np.linalg.norm(Tc, axis=1)[None, :]
    r = num / np.maximum(den, 1e-12)
    j = r.argmax(1)
    return GRID[j], r[np.arange(len(j)), j]


refs = []
for f in sorted((RESULTS / "71_external").glob("external_rhythm_*.csv")):
    refs.append(pd.read_csv(f))
refs = pd.concat(refs, ignore_index=True) if refs else pd.DataFrame()
prep = EXTERNAL / "prepared"
val_rows, pred_rows = [], []
targets = {}
for tissue in ["cochlea", "bm"]:
    for level in ["cell_type", "cell_class"]:
        pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__{level}.parquet")
        design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
        d = design[(design["tissue"] == tissue) & (design["level"] == level)]
        samples = d.drop_duplicates("sample_id").sort_values("zt")
        if level == "cell_type":
            mats = {s: pb[[c for c in pb.columns if c.endswith("|" + s)]].sum(1) for s in samples["sample_id"]}
            targets[(tissue, "all_cells")] = (pd.DataFrame(mats), samples["zt"].to_numpy())
        else:
            for grp in d["group"].unique():
                gd = d[(d["group"] == grp)].sort_values("zt")
                if gd["n_cells"].min() < 50:
                    continue
                targets[(tissue, f"class:{grp}")] = (pb[[f"{grp}|{s}" for s in gd["sample_id"]]]
                                                     .set_axis(gd["sample_id"].tolist(), axis=1), gd["zt"].to_numpy())

for (ds, tis, cond), ref in refs.groupby(["dataset", "tissue", "condition"]):
    sig = ref[(ref["q_BH"] < 0.01) & (ref["ptt_fc"] >= 1.5)].set_index("gene")
    if len(sig) < 20:
        continue
    # LOSO validation on the reference's own samples
    meta = pd.read_csv(prep / f"{ds}__meta.csv")
    ex = pd.read_csv(prep / f"{ds}__expr.csv", index_col=0)
    m = meta[(meta["tissue"] == tis) & (meta["condition"] == cond)]
    x = ex.loc[ex.index.intersection(sig.index), m["sample"]]
    x = np.log2(x / x.sum(0) * 1e6 + 1) if m["value_type"].iloc[0] in ("counts", "normcounts") else x
    errs = []
    for i, s in enumerate(m["sample"]):
        keep = m["sample"] != s
        fit = cosinor.fit(x.loc[:, keep.values].T.to_numpy(), m.loc[keep, "time_h"].to_numpy())
        Z = (x.T.to_numpy() - x.loc[:, keep.values].T.to_numpy().mean(0))[[i]]
        tau, r = predict(Z, fit["amp"], fit["peak"])
        errs.append(float(cosinor.circ_diff(tau[0], m["time_h"].iloc[i])))
    val_rows.append({"dataset": ds, "tissue": tis, "condition": cond, "n_reference_genes": len(sig),
                     "n_samples": len(m), "median_abs_error_h": float(np.median(np.abs(errs))),
                     "frac_within_2h": float(np.mean(np.abs(errs) <= 2))})
    for (tt, target), (mat, zt) in targets.items():
        g = mat.index.intersection(sig.index)
        if len(g) < 15:
            continue
        lc = np.log2(mat.loc[g] / mat.sum(0) * 1e6 + 1)
        Z = (lc.T - lc.T.mean(0)).to_numpy()
        tau, r = predict(Z, sig.loc[g, "amp_log2"].to_numpy(), sig.loc[g, "peak_h"].to_numpy())
        for s, z, t_, rr in zip(mat.columns, zt, tau, r):
            pred_rows.append({"reference": f"{ds}|{tis}|{cond}", "reference_time_type": ref["time_type"].iloc[0],
                              "target_tissue": tt, "target": target, "sample_id": s, "labelled_zt": z,
                              "predicted_time_h": t_, "match_r": rr, "n_genes": len(g),
                              "error_h": float(cosinor.circ_diff(t_, z))})
    log(f"reference {ds}|{tis}|{cond}: {len(sig)} genes; LOSO median |error| "
        f"{val_rows[-1]['median_abs_error_h']:.2f} h")
val = pd.DataFrame(val_rows)
pred = pd.DataFrame(pred_rows)
write_csv(val, MOD, "reference_validation.csv", "Leave-one-sample-out accuracy of template time-telling per reference",
          required=["dataset", "tissue", "condition", "median_abs_error_h"])
write_csv(pred, MOD, "sample_time_prediction.csv", "Predicted circadian time of our samples per reference and target",
          required=["reference", "target_tissue", "target", "sample_id", "labelled_zt", "predicted_time_h", "error_h"])
if len(pred):
    summ = (pred.groupby(["reference", "target_tissue", "target"])
            .apply(lambda g: pd.Series({"n": len(g), "circ_mean_offset_h":
                   (np.angle(np.exp(1j * W * g["error_h"]).mean()) / W),
                   "offset_resultant": float(np.abs(np.exp(1j * W * g["error_h"]).mean())),
                   "median_abs_error_after_offset_h": float(np.median(np.abs(cosinor.circ_diff(
                       g["error_h"], np.angle(np.exp(1j * W * g["error_h"]).mean()) / W))))}))
            .reset_index())
    write_csv(summ, MOD, "timetelling_summary.csv",
              "Per reference × target: circular mean offset (predicted − labelled ZT) and residual error",
              required=["reference", "target", "circ_mean_offset_h", "offset_resultant"])
log("done")
