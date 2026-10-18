"""Module 5.1 (lineage flux) — time-gated production, maturation, release and return along temporal-BM lineages.

Chains (stage labels = gold-standard cell types):
  neutrophil: Granulocyte-Monocyte Progenitor → Neutrophil (Band) → Neutrophil (Mature) → release to blood
              (and the post-mitotic part Band → Mature alone, available in the within-neutrophil frame)
  B cell:     Pro-B Cell (Cycling) → Pre-B Cell → B Cell (Transitional) → egress
  monocyte:   Monocyte (Classical) → Monocyte (Non-classical)
  single pools with rhythmic influx (return / recirculation, constant first-order loss): Neutrophil (Aged),
  Neutrophil (Activated), B Cell (Naïve), B Cell (Mature), Macrophage
Model: x_s' = k_{s−1} x_{s−1} − k_s(t) x_s with k_s(t) = k̄_s (1 + μ_s(t)). Linearising the periodic steady state
(δ_s = log x_s − mean; complex 24-h amplitudes Δ_s = a − i·b, M_s; τ_s = 1/k̄_s = mean residence time):
  Δ_s (1 + iωτ_s) = Δ_{s−1} + M_{s−1} − M_s
so the exit-rate modulation of each stage follows recursively from the observed stage rhythms (production into the
first stage assumed constant, so M_1 = −Δ_1(1 + iωτ_1) is the first stage's net production/exit imbalance). For
single pools the required influx modulation is R = Δ (1 + iωτ). |M| = relative amplitude of the rate rhythm, its
peak ZT = time of maximal exit (or influx); |M| > 1 cannot be produced by sinusoidal gating of one rate (needs
pulsatile gating, rhythmic production or a compositional/dissection effect).
(1) Deterministic grid: common τ ∈ {3, 6, 12, 24, 48, 72} h for all stages.
(2) Monte Carlo: cosinor coefficients from their posterior (flat prior, t with 3 df, as in 60_migration_models) and
    τ_s from log-normal priors (literature-informed medians, ×/÷ 2 at 1 SD) → posterior of |M_s|, peak ZT, P(|M|>1).
(3) Nonlinear single-pool check: periodic steady state of x' = I − k̄(1 + α cos ωt)x (exit gating) and
    x' = Ī(1 + α cos ωt) − k̄x (influx gating) with α = 1 → largest attainable peak-to-trough fold change for a given
    τ, and τ_max = largest residence time that can still produce the observed fold change.
Frames: within-lineage fractions (frame D) and all-cell CLR shares (frame A); a compositional denominator shifts all
stage phasors of a frame by a common amount, so conclusions should agree across frames.
Outputs (results/61_lineage_flux/): lineage_flux_fits.csv, lineage_flux_grid.csv, lineage_flux_nonlinear.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "61_lineage_flux"
W = 2 * np.pi / 24
LN2 = np.log(2)
N_MC = 20000
rng = np.random.default_rng(0)
fr = pd.read_csv(RESULTS / "50_abundance/composition_frames.csv")
fr = fr[fr["tissue"] == "bm"]

# literature-informed median residence times (h) in BM; priors log-normal with SD ln 2
TAU = {"Granulocyte-Monocyte Progenitor": 24, "Neutrophil (Band)": 36, "Neutrophil (Mature)": 36,
       "Neutrophil (Aged)": 6, "Neutrophil (Activated)": 12, "Pro-B Cell (Cycling)": 48, "Pre-B Cell": 48,
       "B Cell (Transitional)": 24, "B Cell (Naïve)": 12, "B Cell (Mature)": 12, "Monocyte (Classical)": 24,
       "Monocyte (Non-classical)": 36, "Macrophage": 72}
CHAINS = {"neutrophil": ["Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Neutrophil (Mature)"],
          "neutrophil_postmitotic": ["Neutrophil (Band)", "Neutrophil (Mature)"],
          "B_cell": ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)"],
          "monocyte": ["Monocyte (Classical)", "Monocyte (Non-classical)"]}
POOLS = ["Neutrophil (Aged)", "Neutrophil (Activated)", "B Cell (Naïve)", "B Cell (Mature)", "Macrophage"]
FRAMES = ["A_clr_all", "D_within_neutrophil", "D_within_myeloid_progenitor", "D_within_B", "D_within_monocyte",
          "D_within_mono_mac"]
TAU_GRID = [3, 6, 12, 24, 48, 72]


def series(frame, feature):
    s = fr[(fr["frame"] == frame) & (fr["feature"] == feature)].sort_values("zt")
    return (s["zt"].to_numpy(), s["value_ln"].to_numpy()) if len(s) == 6 else (None, None)


def phasor_draws(zt, y, n):
    """Point estimate and posterior draws of the complex 24-h amplitude Δ = a − i·b (natural-log units)."""
    X = cosinor.design(zt)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid ** 2).sum() / (len(y) - 3)
    L = np.linalg.cholesky(s2 * np.linalg.inv(X.T @ X) + 1e-12 * np.eye(3))
    d = beta + rng.standard_t(len(y) - 3, size=(n, 3)) @ L.T
    return beta[1] - 1j * beta[2], d[:, 1] - 1j * d[:, 2]


def peak(z):
    return (-np.angle(z) / W) % 24


def circ_summary(h):
    z = np.exp(1j * W * h).mean()
    return (np.angle(z) / W) % 24, float(np.abs(z))


def chain_rates(deltas, taus):
    """Exit-rate modulations M_s along a chain (arrays broadcast over draws)."""
    out, prev_in = [], 0.0          # prev_in = Δ_{s−1} + M_{s−1} (constant production into the first stage)
    for d, t in zip(deltas, taus):
        m = prev_in - d * (1 + 1j * W * t)
        out.append(m)
        prev_in = d + m
    return out


def pss(tau, alpha=1.0, gate="exit", n=4001):
    """Peak-to-trough fold change of the periodic steady state of a gated first-order pool (I = k̄ = 1/τ units)."""
    t = np.linspace(0, 24, n)
    k0 = 1 / tau
    if gate == "influx":
        g = alpha / np.sqrt(1 + (W * tau) ** 2)
        return (1 + g) / (1 - g) if g < 1 else np.inf
    K = k0 * (t + alpha * np.sin(W * t) / W)                       # ∫ k(s) ds
    f = np.exp(K)
    J = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(t))]) * k0   # ∫ I e^{K} ds with I = k0
    x0 = np.exp(-K[-1]) * J[-1] / (1 - np.exp(-K[-1]))
    x = np.exp(-K) * (x0 + J)
    return x.max() / x.min()


def tau_max(fc_obs, gate):
    lo, hi = np.log(0.25), np.log(2000)
    if pss(np.exp(lo), gate=gate) < fc_obs:
        return 0.0
    if pss(np.exp(hi), gate=gate) >= fc_obs:
        return np.inf
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pss(np.exp(mid), gate=gate) >= fc_obs else (lo, mid)
    return float(np.exp((lo + hi) / 2))


fits, grid, nonlin = [], [], []
units = [(c, stages, "chain") for c, stages in CHAINS.items()] + [(p, [p], "pool") for p in POOLS]
for unit, stages, kind in units:
    for frame in FRAMES:
        data = [series(frame, s) for s in stages]
        if any(z is None for z, _ in data):
            continue
        est, draws = zip(*[phasor_draws(z, y, N_MC) for z, y in data])
        tau_draw = [TAU[s] * np.exp(LN2 * rng.standard_normal(N_MC)) for s in stages]
        if kind == "chain":
            m_hat = {tg: chain_rates(est, [tg] * len(stages)) for tg in TAU_GRID}
            m_mc = chain_rates(draws, tau_draw)
        else:
            m_hat = {tg: [est[0] * (1 + 1j * W * tg)] for tg in TAU_GRID}
            m_mc = [draws[0] * (1 + 1j * W * tau_draw[0])]
        for i, s in enumerate(stages):
            role = ("return_or_recirculating_pool" if kind == "pool" else
                    "first_stage" if i == 0 else "terminal_stage" if i == len(stages) - 1 else "intermediate_stage")
            rate = "influx" if kind == "pool" else "exit"
            for tg in TAU_GRID:
                m = m_hat[tg][i]
                grid.append({"unit": unit, "frame": frame, "stage": s, "role": role, "modulated_rate": rate,
                             "tau_h": tg, "rate_rel_amplitude": abs(m), "rate_peak_zt": peak(m),
                             "feasible_sinusoidal_gating": abs(m) <= 1})
            amp = np.abs(m_mc[i])
            pk_mean, pk_res = circ_summary(peak(m_mc[i]))
            d0 = est[i]
            fits.append({"unit": unit, "frame": frame, "stage": s, "role": role, "modulated_rate": rate,
                         "tau_prior_median_h": TAU[s], "obs_amp_ln": abs(d0), "obs_ptt_fc": float(np.exp(2 * abs(d0))),
                         "obs_peak_zt": peak(d0), "rate_rel_amp_median": np.median(amp),
                         "rate_rel_amp_q05": np.quantile(amp, 0.05), "rate_rel_amp_q95": np.quantile(amp, 0.95),
                         "rate_peak_zt_circmean": pk_mean, "rate_peak_zt_resultant": pk_res,
                         "prob_infeasible_sinusoidal": float((amp > 1).mean())})
            fc = float(np.exp(2 * abs(d0)))
            nonlin.append({"unit": unit, "frame": frame, "stage": s, "obs_ptt_fc": fc,
                           "tau_prior_median_h": TAU[s],
                           "max_fc_exit_gating_at_prior_tau": pss(TAU[s], gate="exit"),
                           "max_fc_influx_gating_at_prior_tau": pss(TAU[s], gate="influx"),
                           "tau_max_exit_gating_h": tau_max(fc, "exit"),
                           "tau_max_influx_gating_h": tau_max(fc, "influx")})
    log(f"{unit}: done")
f = pd.DataFrame(fits)
write_csv(f, MOD, "lineage_flux_fits.csv",
          "Required exit/influx-rate rhythms along BM lineage chains (linearised compartment model, Monte Carlo)",
          required=["unit", "frame", "stage", "rate_rel_amp_median", "rate_peak_zt_circmean",
                    "prob_infeasible_sinusoidal"])
write_csv(pd.DataFrame(grid), MOD, "lineage_flux_grid.csv",
          "Required rate rhythms on a grid of common residence times (point estimates)",
          required=["unit", "frame", "stage", "tau_h", "rate_rel_amplitude", "rate_peak_zt"])
write_csv(pd.DataFrame(nonlin), MOD, "lineage_flux_nonlinear.csv",
          "Nonlinear single-pool bounds: attainable fold change under exit or influx gating and maximal residence time",
          required=["unit", "frame", "stage", "obs_ptt_fc", "tau_max_exit_gating_h", "tau_max_influx_gating_h"])
log("done")
