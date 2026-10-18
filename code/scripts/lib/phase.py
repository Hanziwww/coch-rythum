"""Phase utilities for round 2: time axes, circular statistics and cosinor phase uncertainty.

All phases are in hours on [0, 24). ZT is primary (lights on at config design.lights_on_clock_h); clock time is
reported alongside via `zt_to_clock`.
"""
import numpy as np
from scipy import stats

from lib.io import LIGHTS_ON

W = 2 * np.pi / 24
# canonical peak phases (ZT) of core-clock genes in mouse peripheral tissues (same values as 13_clock_controls.py;
# Npas2 stands in for the absent Arntl)
CANONICAL_CLOCK = {"Npas2": 23.5, "Nfil3": 23.0, "Cry1": 20.0, "Rorc": 19.5, "Nr1d1": 7.0, "Nr1d2": 9.0, "Dbp": 10.0,
                   "Tef": 12.0, "Hlf": 12.0, "Per3": 10.0, "Per1": 11.5, "Per2": 14.0, "Cry2": 11.0, "Ciart": 12.5}


def wrap24(h):
    """Map hours to [0, 24) robustly (avoids 24.0 from floating-point round-off)."""
    return np.mod(np.round(np.asarray(h, float), 9), 24)


def zt_to_clock(zt):
    return wrap24(np.asarray(zt, float) + LIGHTS_ON)


def circ_mean(h, w=None):
    """Circular mean (h) and resultant length of phases h (hours), optional weights."""
    h = np.asarray(h, float)
    w = np.ones_like(h) if w is None else np.asarray(w, float)
    z = (w * np.exp(1j * W * h)).sum() / w.sum()
    return float(wrap24(np.angle(z) / W)), float(np.abs(z))


def circ_sd(h):
    _, R = circ_mean(h)
    return float(np.sqrt(-2 * np.log(max(min(R, 1.0), 1e-12))) / W)


def circ_diff(a, b):
    """Signed difference a − b in (−12, 12]."""
    d = (np.asarray(a, float) - np.asarray(b, float)) % 24
    return np.where(d > 12, d - 24, d)


def rayleigh(h):
    """Rayleigh test of uniformity (Zar approximation); returns (z, p)."""
    h = np.asarray(h, float)
    n = len(h)
    if n < 3:
        return np.nan, np.nan
    _, R = circ_mean(h)
    Rn = n * R
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    return n * R ** 2, float(min(max(p, 0.0), 1.0))


def v_test(h, mu):
    """V-test of clustering around the expected phase mu (hours); returns (u, one-sided p)."""
    h = np.asarray(h, float)
    n = len(h)
    if n < 3:
        return np.nan, np.nan
    V = np.cos(W * (h - mu)).sum() / n
    u = V * np.sqrt(2 * n)
    return float(u), float(stats.norm.sf(u))


def watson_williams(groups):
    """Watson–Williams test of equal mean phase across groups of phases (hours); returns (F, p).
    Valid for reasonably concentrated samples (mean resultant length > ~0.45)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    k = len(groups)
    if k < 2:
        return np.nan, np.nan
    n = np.array([len(g) for g in groups])
    N = n.sum()
    Rk = np.array([np.abs(np.exp(1j * W * g).sum()) for g in groups])
    R = np.abs(np.exp(1j * W * np.concatenate(groups)).sum())
    rbar = Rk.sum() / N
    kappa = _a1inv(rbar)
    corr = 1 + 3 / (8 * kappa) if kappa > 0 else 1.0
    F = corr * (N - k) * (Rk.sum() - R) / ((k - 1) * (N - Rk.sum()))
    return float(F), float(stats.f.sf(F, k - 1, N - k))


def _a1inv(r):
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def phase_ci_delta(a, b, var_a, var_b, cov_ab=0.0, df=None, level=0.95):
    """Delta-method CI of the cosinor peak phase φ = atan2(b, a) (hours); returns (peak, half-width)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    A2 = a ** 2 + b ** 2
    var_phi = (b ** 2 * var_a + a ** 2 * var_b - 2 * a * b * cov_ab) / np.maximum(A2 ** 2, 1e-300)
    if df is None:                                   # normal quantile; else t with (per-gene) residual df
        q = stats.norm.ppf(0.5 + level / 2)
    else:
        df = np.asarray(df, float)
        ok = np.isfinite(df) & (df > 0)
        q = np.where(ok, stats.t.ppf(0.5 + level / 2, np.where(ok, df, 1.0)), stats.norm.ppf(0.5 + level / 2))
    half = q * np.sqrt(var_phi) / W
    return wrap24(np.arctan2(b, a) / W), np.minimum(half, 12.0)


def phase_ci_draws(peaks, center, level=0.95):
    """Equal-tailed CI of circular draws around a point estimate; returns (low, high, half-width) in hours."""
    d = circ_diff(peaks, center)
    lo, hi = np.quantile(d, [(1 - level) / 2, (1 + level) / 2])
    return wrap24(center + lo), wrap24(center + hi), (hi - lo) / 2


def phase_determinate(half_width_h, threshold=3.0):
    return np.asarray(half_width_h, float) <= threshold


def multinomial_resample(counts, n_boot, rng):
    """Resample cell counts within each sample (rows) with the observed composition (columns).
    Returns an array (n_boot, n_samples, n_types). Captures counting noise only, not biological variation."""
    counts = np.asarray(counts, float)
    tot = counts.sum(1).astype(int)
    p = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    out = np.empty((n_boot,) + counts.shape)
    for s in range(counts.shape[0]):
        out[:, s, :] = rng.multinomial(tot[s], p[s], size=n_boot)
    return out
