"""Cosinor (24-h harmonic regression) utilities for 6-point time courses.

Used for sample-level quantities (pathway scores, composition ratios, technical scores, clock genes).
Gene-level rhythm statistics with empirical-Bayes moderation are done in R (limma); this module provides the
same model in vectorised OLS form plus the permutation and leave-one-timepoint-out (LOTO) machinery.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

OMEGA = 2 * np.pi / 24.0


def design(zt, period: float = 24.0) -> np.ndarray:
    """Columns: intercept, cos(2πt/T), sin(2πt/T)."""
    zt = np.asarray(zt, dtype=float)
    w = 2 * np.pi / period
    return np.column_stack([np.ones_like(zt), np.cos(w * zt), np.sin(w * zt)])


def amp_phase(a, b, period: float = 24.0):
    """Amplitude sqrt(a²+b²) and peak time (h, in [0, period)) of a·cos(ωt)+b·sin(ωt)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    amp = np.hypot(a, b)
    peak = (np.arctan2(b, a) % (2 * np.pi)) * period / (2 * np.pi)
    return amp, peak


def fit(Y: np.ndarray, zt, period: float = 24.0) -> dict:
    """OLS cosinor fit for each column of Y (n_samples × n_features).

    Returns mesor, a (cos), b (sin), amplitude, peak, F (2, n-3 df), p, residual variance, df_resid.
    Amplitude is in the units of Y (for log2 data, peak-to-trough log2 fold change = 2·amplitude).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    X = design(zt, period)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    df_r = n - k
    rss = (resid ** 2).sum(0)
    tss = ((Y - Y.mean(0)) ** 2).sum(0)
    s2 = rss / df_r if df_r > 0 else np.full(Y.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ((tss - rss) / 2) / s2
        p = stats.f.sf(F, 2, df_r) if df_r > 0 else np.full(Y.shape[1], np.nan)
        r2 = 1 - rss / tss
    amp, peak = amp_phase(beta[1], beta[2], period)
    return {"mesor": beta[0], "a": beta[1], "b": beta[2], "amp": amp, "peak": peak,
            "F": F, "p": p, "r2": r2, "s2": s2, "df_resid": np.full(Y.shape[1], df_r)}


def dihedral_classes(n: int = 6) -> list[tuple[int, ...]]:
    """Representatives of S_n / D_n for n equally spaced time points (identity first).

    A relabeling perm maps sample i to time label perm[i]. Relabelings that differ by a rotation or
    reflection of the time labels span the same {1, cos, sin} column space, hence give identical
    cosinor fits and F statistics. For n = 6 there are 720 / 12 = 60 classes.
    """
    group = []
    for r in range(n):
        group.append(tuple((k + r) % n for k in range(n)))      # rotations
        group.append(tuple((-k + r) % n for k in range(n)))     # reflections
    seen, reps = set(), []
    for perm in itertools.permutations(range(n)):
        if perm in seen:
            continue
        orbit = {tuple(g[perm[i]] for i in range(n)) for g in group}
        seen |= orbit
        reps.append(min(orbit))
    ident = tuple(range(n))
    reps.sort(key=lambda p: (p != ident, p))
    assert reps[0] == ident
    return reps


def design_overlap(perm, zt_sorted) -> float:
    """Squared canonical correlation between true and permuted rhythmic subspaces (1 = equivalent)."""
    zt_sorted = np.asarray(zt_sorted, float)
    Xt = design(zt_sorted)[:, 1:]
    Xp = design(zt_sorted[list(perm)])[:, 1:]
    Qt, _ = np.linalg.qr(Xt - Xt.mean(0))
    Qp, _ = np.linalg.qr(Xp - Xp.mean(0))
    s = np.linalg.svd(Qt.T @ Qp, compute_uv=False)
    return float((s ** 2).mean())


def permutation_test(Y: np.ndarray, zt_sorted, stat: str = "F", p_threshold: float | None = None):
    """Evaluate a cosinor statistic under all 60 dihedral relabeling classes.

    Y rows must be ordered by zt_sorted (ascending ZT). Returns (observed values, null matrix
    [59 × features], per-feature permutation p = (1 + #null ≥ obs) / 60, and if p_threshold is given the
    number of hits under each class).
    """
    Y = np.asarray(Y, float)
    reps = dihedral_classes(len(zt_sorted))
    zt_sorted = np.asarray(zt_sorted, float)
    vals, hits = [], []
    for perm in reps:
        res = fit(Y, zt_sorted[list(perm)])
        vals.append(res[stat])
        if p_threshold is not None:
            hits.append(int(np.nansum(res["p"] < p_threshold)))
    vals = np.vstack(vals)
    obs, null = vals[0], vals[1:]
    p_perm = (1 + (null >= obs[None, :]).sum(0)) / len(reps)
    return obs, null, p_perm, (hits if p_threshold is not None else None)


def loto(Y: np.ndarray, zt) -> dict:
    """Leave-one-timepoint-out refits: phase circular SD (h), min amplitude ratio, max p."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    zt = np.asarray(zt, float)
    full = fit(Y, zt)
    peaks, amps, ps = [], [], []
    for i in range(len(zt)):
        keep = np.arange(len(zt)) != i
        r = fit(Y[keep], zt[keep])
        peaks.append(r["peak"]); amps.append(r["amp"]); ps.append(r["p"])
    peaks, amps, ps = np.vstack(peaks), np.vstack(amps), np.vstack(ps)
    ang = peaks * OMEGA
    R = np.hypot(np.cos(ang).mean(0), np.sin(ang).mean(0))
    circ_sd_h = np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))) / OMEGA
    with np.errstate(divide="ignore", invalid="ignore"):
        min_amp_ratio = amps.min(0) / full["amp"]
    return {"loto_phase_sd_h": circ_sd_h, "loto_min_amp_ratio": min_amp_ratio, "loto_max_p": ps.max(0)}


def circ_diff(a_h, b_h, period: float = 24.0):
    """Signed circular difference a − b in hours, in (−period/2, period/2]."""
    d = (np.asarray(a_h, float) - np.asarray(b_h, float)) % period
    return np.where(d > period / 2, d - period, d)


if __name__ == "__main__":
    reps = dihedral_classes(6)
    print("dihedral classes:", len(reps))
    zt = np.array([3, 7, 11, 15, 19, 23], float)
    # invariance check: every member of the identity orbit gives the same F
    rng = np.random.default_rng(0)
    y = 2 * np.cos(OMEGA * (zt - 9)) + rng.normal(0, 0.5, 6)
    f0 = fit(y, zt)["F"][0]
    for r in range(6):
        for sgn in (1, -1):
            perm = [(sgn * k + r) % 6 for k in range(6)]
            assert np.isclose(fit(y, zt[perm])["F"][0], f0)
    ov = [design_overlap(p, zt) for p in reps]
    print("overlap identity:", round(ov[0], 3), " null overlap range:", round(min(ov[1:]), 3), round(max(ov[1:]), 3))
    print("self-test OK; peak of simulated rhythm (true 9):", round(float(fit(y, zt)["peak"][0]), 2))
