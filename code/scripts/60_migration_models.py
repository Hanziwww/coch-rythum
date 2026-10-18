"""Module 5.1 — abundance-based migration modelling between temporal-bone marrow (BM) and the cochlear system.

Inputs: results/50_abundance/composition_frames.csv (log2 abundance series per frame).
Pairs per shared immune type: BM frame A (CLR within BM) with each cochlear frame — B (vs structural cells),
C (vs marrow content, primary: abundance in excess of the dissection's marrow) and A (CLR within cochlea).

(1) Phase relationships: OLS cosinor per series; parametric draws (t, 3 df) of the cos/sin coefficients give the
    distribution of Δφ = peak_cochlea − peak_BM; probabilities of synchrony (|Δφ| < 3 h), antiphase
    (|Δφ − 12| < 3 h), and of the cochlear peak lagging the BM trough by 0–6 h (egress → influx).
(2) Phasor compartment models (numpyro NUTS; compared by WAIC; complex amplitude c = a − i·b of a·cos ωt + b·sin ωt):
    M0 independent rhythms;  M1 BM→cochlea: c_C = −g·c_B/(1 + iωτ) (cochlear influx mirrors BM depletion,
    first-order residence τ); M2 shared systemic driver: c_C = g·c_B/(1 + iωτ);  M3 cochlea→BM return:
    c_B = −g·c_C/(1 + iωτ). Models compared by WAIC (Akaike-type weights); τ posteriors reported.
    Under the first-order model the cochlear lag is atan(ωτ)/ω ∈ (0, 6) h, so τ is identifiable from the lag.

Outputs (results/60_migration/): phase_relationships.csv, compartment_model_fits.csv, compartment_model_compare.csv
"""
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, log_likelihood

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

warnings.filterwarnings("ignore")
numpyro.set_host_device_count(4)
MOD = "60_migration"
W = 2 * np.pi / 24
rng = np.random.default_rng(0)
fr = pd.read_csv(RESULTS / "50_abundance/composition_frames.csv")


def series(frame, tissue, feature):
    s = fr[(fr["frame"] == frame) & (fr["tissue"] == tissue) & (fr["feature"] == feature)].sort_values("zt")
    return s["zt"].to_numpy(), s["value_log2"].to_numpy()


def coef_draws(zt, y, n=4000):
    X = cosinor.design(zt)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid ** 2).sum() / (len(y) - 3)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = rng.standard_t(len(y) - 3, size=(n, 3))
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(3))
    return beta, beta + t @ L.T


shared = sorted(set(fr[(fr["frame"] == "A_clr_all") & (fr["tissue"] == "bm")]["feature"]) &
                set(fr[(fr["frame"] == "C_alr_marrow") & (fr["tissue"] == "cochlea")]["feature"]))
heme_like = [c for c in shared if not any(k in c for k in ["SV ", "Fibrocyte", "Hair", "Spiral", "Schwann",
                                                           "Interdental", "Epithelial", "Osteo", "Endothelial",
                                                           "Oligodendrocyte", "Stromal"])]
log(f"{len(heme_like)} shared immune types")

rows = []
for ctype in heme_like:
    ztb, yb = series("A_clr_all", "bm", ctype)
    bb, db = coef_draws(ztb, yb)
    for cframe in ["C_alr_marrow", "B_alr_structural", "A_clr_all"]:
        ztc, yc = series(cframe, "cochlea", ctype)
        if len(yc) != 6:
            continue
        bc, dc_ = coef_draws(ztc, yc)
        pk_b = (np.arctan2(db[:, 2], db[:, 1]) % (2 * np.pi)) / W
        pk_c = (np.arctan2(dc_[:, 2], dc_[:, 1]) % (2 * np.pi)) / W
        dphi = cosinor.circ_diff(pk_c, pk_b)
        trough_b = (pk_b + 12) % 24
        lag = (pk_c - trough_b) % 24
        amp_b, ph_b = cosinor.amp_phase(bb[1], bb[2])
        amp_c, ph_c = cosinor.amp_phase(bc[1], bc[2])
        d0 = cosinor.circ_diff(ph_c, ph_b)
        rows.append({"cell_type": ctype, "bm_frame": "A_clr_all", "cochlea_frame": cframe,
                     "bm_peak_zt": float(ph_b), "bm_ptt_fc": float(2 ** (2 * amp_b)),
                     "cochlea_peak_zt": float(ph_c), "cochlea_ptt_fc": float(2 ** (2 * amp_c)),
                     "delta_phase_h": float(d0),
                     "delta_phase_lo": float(d0 + np.quantile(cosinor.circ_diff(dphi, d0), 0.05)),
                     "delta_phase_hi": float(d0 + np.quantile(cosinor.circ_diff(dphi, d0), 0.95)),
                     "prob_sync": float(np.mean(np.abs(dphi) < 3)),
                     "prob_antiphase": float(np.mean(np.abs(np.abs(dphi) - 12) < 3)),
                     "prob_cochlea_peak_0to6h_after_bm_trough": float(np.mean(lag <= 6)),
                     "bm_rhythm_p": float(cosinor.fit(yb[:, None], ztb)["p"][0]),
                     "cochlea_rhythm_p": float(cosinor.fit(yc[:, None], ztc)["p"][0])})
pr = pd.DataFrame(rows)
write_csv(pr, MOD, "phase_relationships.csv",
          "Cochlea-vs-BM abundance phase relationships per immune type and reference frame (parametric draws)",
          required=["cell_type", "cochlea_frame", "delta_phase_h", "prob_sync", "prob_antiphase"])


# ---------------------------------------------------------------- phasor compartment models
def model(t, yb, yc, kind):
    cosv, sinv = jnp.cos(W * t), jnp.sin(W * t)
    mu_b = numpyro.sample("mu_b", dist.Normal(0, 5))
    mu_c = numpyro.sample("mu_c", dist.Normal(0, 5))
    sb = numpyro.sample("sigma_b", dist.HalfNormal(1.0))
    sc_ = numpyro.sample("sigma_c", dist.HalfNormal(1.0))
    if kind == "M0":
        ab, bb_ = numpyro.sample("a_b", dist.Normal(0, 1)), numpyro.sample("b_b", dist.Normal(0, 1))
        ac, bc_ = numpyro.sample("a_c", dist.Normal(0, 1)), numpyro.sample("b_c", dist.Normal(0, 1))
    else:
        a_src = numpyro.sample("a_src", dist.Normal(0, 1))
        b_src = numpyro.sample("b_src", dist.Normal(0, 1))
        g = numpyro.sample("g", dist.HalfNormal(1.0))
        tau = numpyro.sample("tau_h", dist.LogNormal(jnp.log(4.0), 1.0))
        c_src = a_src - 1j * b_src
        sign = -1.0 if kind in ("M1", "M3") else 1.0
        c_dst = sign * g * c_src / (1 + 1j * W * tau)
        a_dst, b_dst = jnp.real(c_dst), -jnp.imag(c_dst)
        if kind == "M3":   # source is the cochlea
            ac, bc_, ab, bb_ = a_src, b_src, a_dst, b_dst
        else:
            ab, bb_, ac, bc_ = a_src, b_src, a_dst, b_dst
        numpyro.deterministic("lag_h", jnp.arctan(W * tau) / W)
    numpyro.sample("yb", dist.Normal(mu_b + ab * cosv + bb_ * sinv, sb), obs=yb)
    numpyro.sample("yc", dist.Normal(mu_c + ac * cosv + bc_ * sinv, sc_), obs=yc)


fits, comps = [], []
for ctype in heme_like:
    ztb, yb = series("A_clr_all", "bm", ctype)
    ztc, yc = series("C_alr_marrow", "cochlea", ctype)
    if len(yc) != 6 or not np.allclose(ztb, ztc):
        continue
    waic = {}
    for kind in ["M0", "M1", "M2", "M3"]:
        mc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=1500, num_samples=1500, num_chains=4,
                  progress_bar=False)
        mc.run(jax.random.PRNGKey(1), jnp.asarray(ztb), jnp.asarray(yb), jnp.asarray(yc), kind)
        s = mc.get_samples()
        ll = log_likelihood(model, s, jnp.asarray(ztb), jnp.asarray(yb), jnp.asarray(yc), kind)
        ll_all = np.concatenate([np.asarray(ll["yb"]), np.asarray(ll["yc"])], axis=1)   # draws × 12
        lppd = np.log(np.exp(ll_all - ll_all.max(0)).mean(0)) + ll_all.max(0)
        p_waic = ll_all.var(0)
        waic[kind] = {"elpd_waic": float((lppd - p_waic).sum()), "p_waic": float(p_waic.sum()),
                      "max_pointwise_var": float(p_waic.max())}
        div = int(mc.get_extra_fields()["diverging"].sum())
        for par in ["g", "tau_h", "lag_h"]:
            if par in s:
                v = np.asarray(s[par])
                fits.append({"cell_type": ctype, "model": kind, "parameter": par, "median": float(np.median(v)),
                             "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
                             "n_divergent": div})
    cmp = pd.DataFrame(waic).T.reset_index().rename(columns={"index": "model"})
    cmp["elpd_diff"] = cmp["elpd_waic"] - cmp["elpd_waic"].max()
    cmp["weight"] = np.exp(cmp["elpd_diff"]) / np.exp(cmp["elpd_diff"]).sum()
    cmp = cmp.sort_values("elpd_waic", ascending=False).reset_index(drop=True)
    cmp["rank"] = np.arange(len(cmp))
    cmp["warning_pointwise_var_gt_0p4"] = cmp["max_pointwise_var"] > 0.4
    cmp.insert(0, "cell_type", ctype)
    comps.append(cmp)
    log(f"  {ctype}: best {cmp.iloc[0]['model']} (weight {cmp.iloc[0]['weight']:.2f})")

write_csv(pd.DataFrame(fits), MOD, "compartment_model_fits.csv",
          "Posterior summaries of coupling gain g, residence time tau (h) and implied lag for models M1–M3",
          required=["cell_type", "model", "parameter", "median"])
write_csv(pd.concat(comps, ignore_index=True), MOD, "compartment_model_compare.csv",
          "WAIC comparison (Akaike-type weights) of independent/transfer/shared-driver/return models",
          required=["cell_type", "model", "elpd_waic", "weight"])
log("done")
