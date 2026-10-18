"""Module 4 core — marrow-vs-tissue decomposition of immune abundance in the cochlear system (Bayesian, numpyro).

For cochlear sample s (ZT t_s) and haematopoietic type k (marrow-restricted progenitors excluded):
    y_sk = log2((n_sk + 0.5) / (n_struct_s + 0.5))                       abundance vs structural cells
    z_s  = log2((n_marrow_s + 0.5) / (n_struct_s + 0.5)), centred           marrow content of the dissection
    y_sk ~ Normal(alpha_k + beta_k * z_s + a_k cos(ωt_s) + b_k sin(ωt_s), sigma_k)
    beta_k ~ Normal(mu_beta, tau_beta), mu_beta ~ Normal(1, 0.5), tau_beta ~ HalfNormal(0.5)
    a_k, b_k ~ Normal(0, tau_r), tau_r ~ HalfNormal(1);  sigma_k ~ HalfNormal(sigma0), sigma0 ~ HalfNormal(1)
beta_k ≈ 1: the type scales with marrow content (marrow-contained); the residual rhythm (a_k, b_k) is the
tissue-level 24-h component beyond marrow content. The rhythm of z_s itself is reported (dissection vs biology).
A second fit drops beta (no marrow adjustment) for comparison.

Outputs (results/52_marrow_decomposition/): mixture_model_fits.csv, tissue_residual_rhythm.csv,
                                            marrow_content_rhythm.csv
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

numpyro.set_host_device_count(4)
MOD = "52_marrow_decomposition"
W = 2 * np.pi / 24
STRUCT = ["Hair Cell (Outer/Inner)", "SV Marginal Cell", "SV Intermediate Cell (Pigmented)",
          "SV Intermediate Cell (Functional)", "Interdental Cell", "Epithelial Cell (Non-sensory)",
          "Fibrocyte (Cochlear)", "Fibrocyte (Spiral Ligament)", "Spiral Ganglion Neuron", "Schwann Cell"]
MARROW = ["HSC / MPP", "Granulocyte-Monocyte Progenitor", "Pro-B Cell (Cycling)", "Pre-B Cell", "Erythroblast"]
HEME = ["Myeloid", "Lymphoid", "Erythroid", "Progenitor", "Cycling"]

ct = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
co = ct[ct["tissue"] == "cochlea"]
wide = co.pivot_table(index=["sample_id", "zt"], columns="cell_type", values="n_cells", fill_value=0).sort_index(level="zt")
cls = co.drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
types = [c for c in wide.columns if cls[c] in HEME and c not in MARROW and wide[c].min() >= 5]
zt = wide.index.get_level_values("zt").to_numpy(float)
n_struct = wide[[c for c in STRUCT if c in wide]].sum(1).to_numpy()
n_marrow = wide[[c for c in MARROW if c in wide]].sum(1).to_numpy()
z = np.log2((n_marrow + 0.5) / (n_struct + 0.5))
zc = z - z.mean()
Y = np.log2((wide[types].to_numpy() + 0.5) / (n_struct[:, None] + 0.5))     # samples × types
log(f"{len(types)} immune types; marrow content z = {np.round(z, 2)}")


def model(Y, zc, cosv, sinv, adjust=True):
    S, K = Y.shape
    alpha = numpyro.sample("alpha", dist.Normal(0, 5).expand([K]))
    tau_r = numpyro.sample("tau_r", dist.HalfNormal(1.0))
    a = numpyro.sample("a", dist.Normal(0, tau_r).expand([K]))
    b = numpyro.sample("b", dist.Normal(0, tau_r).expand([K]))
    sigma0 = numpyro.sample("sigma0", dist.HalfNormal(1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma0).expand([K]))
    mu = alpha[None, :] + a[None, :] * cosv[:, None] + b[None, :] * sinv[:, None]
    if adjust:
        mu_beta = numpyro.sample("mu_beta", dist.Normal(1.0, 0.5))
        tau_beta = numpyro.sample("tau_beta", dist.HalfNormal(0.5))
        beta = numpyro.sample("beta", dist.Normal(mu_beta, tau_beta).expand([K]))
        mu = mu + beta[None, :] * zc[:, None]
    numpyro.sample("y", dist.Normal(mu, sigma[None, :]), obs=Y)


def fit(adjust):
    kern = NUTS(model, target_accept_prob=0.95)
    mc = MCMC(kern, num_warmup=2000, num_samples=2000, num_chains=4, progress_bar=False)
    mc.run(jax.random.PRNGKey(0), jnp.asarray(Y), jnp.asarray(zc), jnp.cos(W * zt), jnp.sin(W * zt), adjust)
    return mc


rows, rrows = [], []
for adjust in (True, False):
    mc = fit(adjust)
    s = mc.get_samples()
    div = int(mc.get_extra_fields()["diverging"].sum())
    summ = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    for k, c in enumerate(types):
        amp = np.hypot(np.asarray(s["a"][:, k]), np.asarray(s["b"][:, k]))
        pk = (np.arctan2(np.asarray(s["b"][:, k]), np.asarray(s["a"][:, k])) % (2 * np.pi)) / W
        pk_mean = (np.angle(np.exp(1j * W * pk).mean()) / W) % 24
        dev = cosinor.circ_diff(pk, pk_mean)
        rrows.append({"model": "marrow_adjusted" if adjust else "unadjusted", "cell_type": c,
                      "beta_mean": float(np.mean(s["beta"][:, k])) if adjust else np.nan,
                      "beta_lo": float(np.quantile(s["beta"][:, k], 0.05)) if adjust else np.nan,
                      "beta_hi": float(np.quantile(s["beta"][:, k], 0.95)) if adjust else np.nan,
                      "resid_ptt_fc_median": float(np.median(2 ** (2 * amp))),
                      "resid_ptt_fc_lo": float(np.quantile(2 ** (2 * amp), 0.05)),
                      "resid_ptt_fc_hi": float(np.quantile(2 ** (2 * amp), 0.95)),
                      "resid_peak_zt": float(pk_mean),
                      "resid_peak_lo": float((pk_mean + np.quantile(dev, 0.05)) % 24),
                      "resid_peak_hi": float((pk_mean + np.quantile(dev, 0.95)) % 24),
                      "prob_fc_gt_1p5": float(np.mean(2 ** (2 * amp) > 1.5)),
                      "rhat_a": float(summ["a"]["r_hat"][k]), "rhat_b": float(summ["b"]["r_hat"][k]),
                      "n_divergent": div})
    for par in ["mu_beta", "tau_beta", "tau_r", "sigma0"]:
        if par in s:
            v = np.asarray(s[par])
            rows.append({"model": "marrow_adjusted" if adjust else "unadjusted", "parameter": par,
                         "mean": float(v.mean()), "q05": float(np.quantile(v, 0.05)),
                         "q95": float(np.quantile(v, 0.95)), "r_hat": float(summ[par]["r_hat"]),
                         "n_eff": float(summ[par]["n_eff"])})
    log(f"fit adjust={adjust}: divergences {div}")

write_csv(pd.DataFrame(rows), MOD, "mixture_model_fits.csv",
          "Hyper-parameters of the marrow-vs-tissue decomposition (posterior mean, 90 % interval, R-hat, n_eff)",
          required=["model", "parameter", "mean"])
rr = pd.DataFrame(rrows)
write_csv(rr, MOD, "tissue_residual_rhythm.csv",
          "Per immune type: marrow-content scaling (beta) and residual tissue-level 24-h rhythm (posterior)",
          required=["model", "cell_type", "resid_ptt_fc_median", "resid_peak_zt", "prob_fc_gt_1p5"])
f = cosinor.fit(z[:, None], zt)
_, _, pp, _ = cosinor.permutation_test(z[:, None], zt)
mc_df = pd.DataFrame({"sample_id": wide.index.get_level_values("sample_id"), "zt": zt, "n_marrow": n_marrow,
                      "n_structural": n_struct, "log2_marrow_vs_structural": z})
mc_df["rhythm_ptt_fc"] = 2 ** (2 * f["amp"][0])
mc_df["rhythm_peak_zt"] = f["peak"][0]
mc_df["rhythm_p"] = f["p"][0]
mc_df["rhythm_p_perm"] = pp[0]
write_csv(mc_df, MOD, "marrow_content_rhythm.csv", "Marrow content of each cochlear dissection and its 24-h test",
          required=["sample_id", "zt", "log2_marrow_vs_structural", "rhythm_p"])
pd.set_option("display.width", 250)
print(rr.sort_values(["model", "prob_fc_gt_1p5"], ascending=[True, False]).round(3).to_string(index=False))
log("done")
