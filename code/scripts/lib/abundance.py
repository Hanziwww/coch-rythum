"""Shared abundance-rhythm models for count frames with one pooled sample per time point (rounds 5–6).

ols / perm_p / loto_sd   OLS cosinor on a per-sample value (logit share, CLR or balance), exact permutation p over the
                         60 dihedral relabeling classes, leave-one-time-point-out phase SD
quasibin                 GLM on proportions with var_weights = denominator, Pearson-X² dispersion, F test of cos / sin
betabin                  maximum-likelihood beta-binomial (logit mean, common precision), likelihood-ratio test (2 df)
balance                  isometric log-ratio balance between two groups of parts (sequential binary partition)
verdict                  robust (>= 3 core, LOTO SD < 3 h, core peaks SD < 3 h) / supported (2) / fragile / not rhythmic
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import optimize, special, stats

from lib import cosinor, phase

CLASSES = cosinor.dihedral_classes(6)


def logit(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.log((n + 0.5) / (d - n + 0.5))


def ols(y, zt):
    f = cosinor.fit(np.asarray(y, float)[:, None], zt)
    return float(f["F"][0]), float(f["p"][0]), float(f["peak"][0]), float(f["amp"][0])


def perm_p(y, zt):
    obs = ols(y, zt)[0]
    if not np.isfinite(obs):        # constant y (e.g. tied medians): no rhythm statistic
        return np.nan
    null = [ols(y, np.asarray(zt)[list(p)])[0] for p in CLASSES[1:]]
    return (1 + sum(x >= obs for x in null)) / len(CLASSES)


def loto_sd(y, zt):
    return float(cosinor.loto(np.asarray(y, float)[:, None], zt)["loto_phase_sd_h"][0])


def quasibin(n, d, zt):
    n, d = np.asarray(n, float), np.asarray(d, float)
    X = cosinor.design(zt)
    prop = n / d
    f1 = sm.GLM(prop, X, family=sm.families.Binomial(), var_weights=d).fit(scale="X2")
    f0 = sm.GLM(prop, X[:, :1], family=sm.families.Binomial(), var_weights=d).fit(scale="X2")
    F = ((f0.deviance - f1.deviance) / 2) / f1.scale
    return float(F), float(stats.f.sf(F, 2, len(n) - 3)), float(cosinor.amp_phase(f1.params[1], f1.params[2])[1])


def betabin(n, d, zt):
    n, d = np.asarray(n, float), np.asarray(d, float)
    X = cosinor.design(zt)

    def nll(par, Xm):
        b, lt = par[:-1], par[-1]
        mu = special.expit(Xm @ b)
        th = np.exp(lt)
        a, bb = mu * th, (1 - mu) * th
        return -np.sum(special.betaln(n + a, d - n + bb) - special.betaln(a, bb))

    b0 = [special.logit((n.sum() + 0.5) / (d.sum() + 1))]
    r0 = optimize.minimize(nll, np.r_[b0, 3.0], args=(X[:, :1],), method="L-BFGS-B")
    r1 = optimize.minimize(nll, np.r_[b0, 0, 0, 3.0], args=(X,), method="L-BFGS-B")
    lr = max(0.0, 2 * (r0.fun - r1.fun))
    return float(lr), float(stats.chi2.sf(lr, 2)), float(cosinor.amp_phase(r1.x[1], r1.x[2])[1])


def balance(m, first, rest):
    """m: samples × parts counts; first / rest: column index lists → ilr balance per sample (counts + 0.5)."""
    L = np.log(np.asarray(m, float) + 0.5)
    r, s = len(first), len(rest)
    return np.sqrt(r * s / (r + s)) * (L[:, first].mean(1) - L[:, rest].mean(1))


def verdict(n_core, loto_phase_sd, core_peaks, p_ols):
    csd = phase.circ_sd(np.asarray(core_peaks, float)) if len(core_peaks) >= 2 else np.nan
    if n_core >= 3 and loto_phase_sd < 3 and (np.isnan(csd) or csd < 3):
        return "robust", csd
    if n_core >= 2:
        return "supported", csd
    if n_core >= 1 or p_ols < 0.05:
        return "fragile", csd
    return "not rhythmic", csd
