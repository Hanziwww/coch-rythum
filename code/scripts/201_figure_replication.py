"""Round 2, WP1 — replication of the user's figure and its extension to every cell type.

The figure (abundance rhythms of BM naïve/transitional B cells and classical monocytes; cochlear classical
monocytes, band and mature neutrophils) is reproduced exactly by an ordinary least-squares single-component cosinor
(24 h) on the percentage of all cells of the tissue sample, F test of the cos/sin terms (2, 3 df), with the time axis
in clock hours (`day_time`). True ZT = clock − 8 (lights on 08:00); every table gives both (peak_zt, peak_clock).
For all cell types of both tissues (>= 20 cells in total):
  the same fit (BH q within tissue); exact 60-class dihedral permutation p; leave-one-time-point-out (LOTO) phase SD
  and minimal amplitude ratio; delta-method 95 % phase CI (t, 3 df) and phase-determinate flag (half-width <= 3 h);
  multinomial cell-resampling bootstrap (2000 draws; counting noise only): CIs of amplitude and peak and the fraction
  of draws with p < 0.05.
Figure-ready tables: fitted curve with 95 % confidence band on a 0.1-h grid (ZT and clock) and the observed points.
Left-panel schematic as numbers (cell types present in both tissues): Δpeak = cochlea − BM from t draws of the
cosinor coefficients (residual uncertainty, 4000 draws) with probabilities of synchrony (|Δ| < 3 h), of the
emigration ordering (cochlear peak 0–6 h after the BM trough: Δ ∈ [12, 18] h) and of the return/clearance ordering
(BM peak 0–6 h after the cochlear trough: Δ ∈ [6, 12] h).
Outputs (results/r2_figure/): figure_replication_check.csv, abundance_pct_rhythm.csv, figure_curves.csv,
figure_points.csv, figure_phase_relationships.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_figure"
W = 2 * np.pi / 24
rng = np.random.default_rng(2)
FIGURE = [("bm", "B Cell (Naïve)", 0.000, 4.5), ("bm", "Monocyte (Classical)", 0.010, 20.7),
          ("bm", "B Cell (Transitional)", 0.016, 20.8), ("cochlea", "Monocyte (Classical)", 0.000, 15.0),
          ("cochlea", "Neutrophil (Band)", 0.035, 13.2), ("cochlea", "Neutrophil (Mature)", 0.045, 16.6)]

ct = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
samples = ct.drop_duplicates("sample_id")[["sample_id", "tissue", "zt"]]
full = (samples.merge(ct[["tissue", "cell_type"]].drop_duplicates(), on="tissue")
        .merge(ct[["sample_id", "cell_type", "n_cells"]], on=["sample_id", "cell_type"], how="left").fillna({"n_cells": 0}))
full["total"] = full.groupby("sample_id")["n_cells"].transform("sum")
full["pct"] = 100 * full["n_cells"] / full["total"]
full["clock"] = phase.zt_to_clock(full["zt"])


def ols(zt, Y):
    """OLS cosinor on columns of Y (rows = samples sorted by zt); returns dict incl. coefficient covariance."""
    X = cosinor.design(zt)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R = Y - X @ B
    s2 = (R ** 2).sum(0) / (len(zt) - 3)
    XtXi = np.linalg.inv(X.T @ X)
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    F = ((tss - (R ** 2).sum(0)) / 2) / np.maximum(s2, 1e-300)
    return {"B": B, "s2": s2, "XtXi": XtXi, "F": F, "p": stats.f.sf(F, 2, len(zt) - 3)}


rows, curves, points, boot_peaks, draws = [], [], [], {}, {}
grid = np.round(np.arange(0, 24.0001, 0.1), 1)
Xg = cosinor.design(grid)
tq = stats.t.ppf(0.975, 3)
for tissue, g in full.groupby("tissue"):
    wide = g.pivot(index=["sample_id", "zt"], columns="cell_type", values="n_cells").sort_index(level="zt")
    zt = wide.index.get_level_values("zt").to_numpy(float)
    keep = wide.columns[(wide.sum(0) >= 20) & ((wide > 0).sum(0) >= 3)]
    cnt = wide.to_numpy(float)                      # all types (denominator = all cells of the sample)
    pct = 100 * cnt / cnt.sum(1, keepdims=True)
    cols = list(wide.columns)
    idx = [cols.index(c) for c in keep]
    Y = pct[:, idx]
    f = ols(zt, Y)
    a, b = f["B"][1], f["B"][2]
    amp, pk = cosinor.amp_phase(a, b)
    var_ab = f["s2"] * f["XtXi"][1, 1]
    _, half = phase.phase_ci_delta(a, b, var_ab, var_ab, 0.0, df=3)
    _, _, p_perm, _ = cosinor.permutation_test(Y, zt)
    lo = cosinor.loto(Y, zt)
    # multinomial cell-resampling bootstrap (counting noise)
    bt = phase.multinomial_resample(cnt, 2000, rng)
    bpct = 100 * bt / bt.sum(2, keepdims=True)
    bamp = np.empty((2000, len(idx))); bpk = np.empty_like(bamp); bsig = np.empty_like(bamp)
    for i in range(2000):
        fb = ols(zt, bpct[i][:, idx])
        bamp[i], bpk[i] = cosinor.amp_phase(fb["B"][1], fb["B"][2])
        bsig[i] = fb["p"] < 0.05
    q = multipletests(f["p"], method="fdr_bh")[1]
    for j, c in enumerate(keep):
        blo, bhi, bhalf = phase.phase_ci_draws(bpk[:, j], pk[j])
        rows.append({"tissue": tissue, "cell_type": c, "n_cells_total": int(cnt[:, idx[j]].sum()),
                     "mean_pct": float(Y[:, j].mean()), "mesor_pct": float(f["B"][0, j]), "amplitude_pct": float(amp[j]),
                     "rel_amplitude": float(amp[j] / max(f["B"][0, j], 1e-9)),
                     "ptt_ratio": float((f["B"][0, j] + amp[j]) / (f["B"][0, j] - amp[j]))
                     if f["B"][0, j] - amp[j] > 0 else np.nan,             # undefined when the fitted trough is <= 0 %
                     "peak_zt": float(pk[j]), "peak_clock": float(phase.zt_to_clock(pk[j])),
                     "trough_zt": float((pk[j] + 12) % 24), "peak_ci_halfwidth_h": float(half[j]),
                     "phase_determinate": bool(half[j] <= 3), "F": float(f["F"][j]), "p_ols": float(f["p"][j]),
                     "q_BH": float(q[j]), "p_perm": float(p_perm[j]), "loto_phase_sd_h": float(lo["loto_phase_sd_h"][j]),
                     "loto_min_amp_ratio": float(lo["loto_min_amp_ratio"][j]), "loto_max_p": float(lo["loto_max_p"][j]),
                     "boot_amp_lo": float(np.quantile(bamp[:, j], 0.025)), "boot_amp_hi": float(np.quantile(bamp[:, j], 0.975)),
                     "boot_peak_zt_lo": float(blo), "boot_peak_zt_hi": float(bhi), "boot_peak_halfwidth_h": float(bhalf),
                     "boot_frac_p05": float(bsig[:, j].mean())})
        cov = f["s2"][j] * f["XtXi"]
        yhat = Xg @ f["B"][:, j]
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))
        curves.append(pd.DataFrame({"tissue": tissue, "cell_type": c, "zt": grid, "clock": phase.zt_to_clock(grid),
                                    "fit_pct": yhat, "band_lo": yhat - tq * se, "band_hi": yhat + tq * se}))
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(3))
        d = f["B"][:, j] + rng.standard_t(3, size=(4000, 3)) @ L.T
        draws[(tissue, c)] = cosinor.amp_phase(d[:, 1], d[:, 2])[1]
    points.append(g[g["cell_type"].isin(keep)][["tissue", "cell_type", "sample_id", "zt", "clock", "n_cells", "total", "pct"]])
    log(f"{tissue}: {len(keep)} cell types fitted")

res = pd.DataFrame(rows)
chk = []
for tissue, c, p_fig, clock_fig in FIGURE:
    r = res[(res["tissue"] == tissue) & (res["cell_type"] == c)].iloc[0]
    chk.append({"tissue": tissue, "cell_type": c, "figure_p": p_fig, "replicated_p": round(r["p_ols"], 3),
                "figure_peak_axis_label": clock_fig, "replicated_peak_clock": round(r["peak_clock"], 1),
                "true_peak_zt": round(r["peak_zt"], 1),
                "match": bool(round(r["p_ols"], 3) == p_fig and round(r["peak_clock"], 1) == clock_fig)})
chk = pd.DataFrame(chk)
write_csv(chk, MOD, "figure_replication_check.csv",
          "Exact replication of the figure panels (OLS cosinor on % of tissue cells; figure axis = clock time)",
          required=["tissue", "cell_type", "figure_p", "replicated_p", "true_peak_zt", "match"])
assert chk["match"].all(), chk
write_csv(res, MOD, "abundance_pct_rhythm.csv",
          "Cosinor of % of tissue cells per cell type: OLS (figure model), BH, permutation, LOTO, phase CIs, bootstrap",
          required=["tissue", "cell_type", "peak_zt", "peak_clock", "p_ols", "q_BH", "p_perm"])
write_csv(pd.concat(curves, ignore_index=True), MOD, "figure_curves.csv",
          "Figure-ready fitted cosinor curves (% of tissue cells) with 95 % confidence bands on a 0.1-h grid",
          required=["tissue", "cell_type", "zt", "clock", "fit_pct", "band_lo", "band_hi"])
write_csv(pd.concat(points, ignore_index=True), MOD, "figure_points.csv",
          "Observed % of tissue cells per sample (figure-ready points)", required=["tissue", "cell_type", "zt", "pct"])

rel = []
shared = sorted({c for t, c in draws if t == "bm"} & {c for t, c in draws if t == "cochlea"})
for c in shared:
    d = phase.circ_diff(draws[("cochlea", c)], draws[("bm", c)]) % 24
    rb = res[(res["tissue"] == "bm") & (res["cell_type"] == c)].iloc[0]
    rc = res[(res["tissue"] == "cochlea") & (res["cell_type"] == c)].iloc[0]
    d0 = float(phase.circ_diff(rc["peak_zt"], rb["peak_zt"]))          # signed, (−12, 12]
    du = d0 + phase.circ_diff(d, d0)                                    # draws unwrapped around the estimate
    lo_, hi_ = (float(x) for x in np.percentile(du, [2.5, 97.5]))
    rel.append({"cell_type": c, "bm_peak_zt": rb["peak_zt"], "bm_p": rb["p_ols"], "cochlea_peak_zt": rc["peak_zt"],
                "cochlea_p": rc["p_ols"], "delta_cochlea_minus_bm_h": d0, "delta_ci_lo": lo_, "delta_ci_hi": hi_,
                "delta_ci_width_h": hi_ - lo_,
                "prob_sync_lt3h": float((np.minimum(d, 24 - d) < 3).mean()),
                "prob_emigration_order": float(((d >= 12) & (d <= 18)).mean()),
                "prob_return_order": float(((d >= 6) & (d < 12)).mean()),
                "prob_cochlea_leads_0to6h": float(((d >= 18) | (d == 0)).mean()),
                "prob_cochlea_lags_0to6h": float(((d > 0) & (d < 6)).mean()),
                "both_p05": bool(rb["p_ols"] < 0.05 and rc["p_ols"] < 0.05)})
write_csv(pd.DataFrame(rel), MOD, "figure_phase_relationships.csv",
          "BM vs cochlea phase relationships of % abundance (signed Δ = cochlea − BM in (−12, 12] h, 95 % CI from "
          "unwrapped bootstrap draws; ordering probabilities use Δ mod 24: emigration [12, 18], return [6, 12))",
          required=["cell_type", "delta_cochlea_minus_bm_h", "prob_sync_lt3h", "prob_emigration_order", "prob_return_order"])
log("done")
