"""Round 9 — dispersion-free exact relabelling test of Milo neighbourhood rhythms, with a positive control.

Why: in 53e the negative-binomial test re-estimates the dispersion under every relabelling; with neighbourhoods dominated
by single samples the dispersion reaches edgeR's cap and a few relabelling classes win in every neighbourhood, so even
injected 8-fold rhythms were not detected (milo_positive_control.csv). Here each neighbourhood's own rhythm is scored
without a shared dispersion: y = log2(count + 0.5) centred per sample across neighbourhoods (removes sample size and any
pattern shared by all neighbourhoods), cosinor R2 under each of the 60 dihedral relabelling classes (same enumeration as
scripts/lib/cosinor.R; class 1 = observed). A pure single-sample spike scores (almost) the same R2 under every class, so
it cannot win systematically; a smooth multi-sample 24-h pattern ranks first under the true times.
Tests (exact over the 60 classes): number of neighbourhoods whose observed class is the strict best (per-neighbourhood
exact p = 1/60) and the rank sum of R2; subsets: all, not dominated by one sample (< 50 % of cells), dominated,
majority types with >= 60 neighbourhoods. Positive control: smooth rhythms (2-8-fold peak-to-trough) injected by
binomial thinning into 5 % of the real neighbourhoods (all, or only non-dominated) aligned with a non-observed class
k0 (10 classes), statistics ranked for k0; fold change 1 = no injection (calibration).
Inputs: results/53_milo/milo_counts_<tissue>.csv, milo_robustness_<tissue>.csv (primary scANVI neighbourhoods).
Outputs (results/53_milo/): milo_r2_exact.csv, milo_r2_variants.csv, milo_r2_positive_control.csv
Run: python scripts/53g_milo_exact_r2.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "53_milo"
M = RESULTS / MOD
rng = np.random.default_rng(530)


def dihedral_classes(n=6):
    """Same enumeration as dihedral_classes() in scripts/lib/cosinor.R (lexicographic permutations, orbit under the
    rotations / reflections of the time axis, lexicographically smallest representative, identity first)."""
    group = [tuple((np.arange(n) + r) % n) for r in range(n)] + [tuple((-np.arange(n) + r) % n) for r in range(n)]
    seen, reps = set(), []
    for p in itertools.permutations(range(n)):
        if p in seen:
            continue
        orbit = [tuple(np.asarray(g)[list(p)]) for g in group]
        seen.update(orbit)
        keys = ["".join(map(str, o)) for o in orbit]
        reps.append(orbit[int(np.argsort(keys, kind="stable")[0])])
    ident = tuple(range(n))
    reps = [r for r in reps if r == ident] + [r for r in reps if r != ident]
    assert len(reps) == 60 and reps[0] == ident
    return reps


CLASSES = dihedral_classes()


def class_times(zt, k):
    ordr = np.argsort(zt, kind="stable")
    z = zt.copy()
    z[ordr] = zt[ordr][list(CLASSES[k])]
    return z


def r2_matrix(Y, zt):
    """Y: neighbourhoods x samples (centred values); returns neighbourhoods x 60 cosinor R2."""
    Yc = Y - Y.mean(1, keepdims=True)
    tss = (Yc ** 2).sum(1)
    out = np.empty((Y.shape[0], 60))
    for k in range(60):
        t = class_times(zt, k)
        X = np.column_stack([np.ones_like(t), np.cos(phase.W * t), np.sin(phase.W * t)])
        H = X @ np.linalg.pinv(X)
        res = Y - Y @ H.T
        out[:, k] = 1 - (res ** 2).sum(1) / np.where(tss > 0, tss, np.nan)
    return np.nan_to_num(out, nan=0.0)


def transform(C):
    L = np.log2(C + 0.5)
    return L - np.median(L, axis=0, keepdims=True)


def tests(R, k_obs):
    n = R.shape[0]
    mx = R.max(1, keepdims=True)
    strict = (R >= mx - 1e-12) & ((R >= mx - 1e-12).sum(1, keepdims=True) == 1)
    nstrict = strict.sum(0).astype(float)
    rsum = stats.rankdata(R, axis=1).sum(0)
    rows = []
    for name, v in (("neighbourhoods with the class as strict best", nstrict), ("rank sum of R2", rsum)):
        others = np.delete(v, k_obs)
        rows.append({"statistic": name, "n_nhoods": n, "observed": float(v[k_obs]), "null_mean": float(others.mean()),
                     "null_q95": float(np.quantile(others, 0.95)), "p_exact": float((v >= v[k_obs] - 1e-9).mean()),
                     "expected_under_null": n / 60 if name.startswith("neigh") else n * 30.5})
    return rows


VARIANTS = {
    "median-centred log counts (primary)": lambda C: transform(C),
    "mean-centred log counts": lambda C: np.log2(C + 0.5) - np.log2(C + 0.5).mean(0, keepdims=True),
    "library-normalised log counts, uncentred": lambda C: np.log2((C + 0.5) / C.sum(0, keepdims=True) * C.sum(0).mean()),
}   # (within-neighbourhood proportions give the primary result exactly: R2 ignores a per-neighbourhood constant)


def strict_counts(R):
    mx = R.max(1, keepdims=True)
    eq = R >= mx - 1e-12
    return (eq & (eq.sum(1, keepdims=True) == 1)).sum(0)


ex_rows, pc_rows, var_rows = [], [], []
for t in ["bm", "cochlea"]:
    cnt = pd.read_csv(M / f"milo_counts_{t}.csv")
    C = cnt.pivot(index="nhood", columns="sample_id", values="n_cells").sort_index()
    zt = cnt.drop_duplicates("sample_id").set_index("sample_id")["zt"].reindex(C.columns).to_numpy(float)
    rob = pd.read_csv(M / f"milo_robustness_{t}.csv").set_index("nhood").loc[C.index]
    dom = rob["dominated"].to_numpy(bool)
    maj = rob["majority_cell_type"].to_numpy()
    Craw = C.to_numpy(float)
    R = r2_matrix(transform(Craw), zt)
    subsets = [("all", np.ones(len(C), bool)), ("not dominated by one sample", ~dom), ("dominated by one sample", dom)]
    subsets += [(f"majority: {m}", maj == m) for m, c_ in pd.Series(maj).value_counts().items() if c_ >= 60]
    for name, msk in subsets:
        for r in tests(R[msk], 0):
            ex_rows.append({"tissue": t, "subset": name, **r})
    # ---- normalisation variants (sensitivity of the primary test)
    for vname, fn in VARIANTS.items():
        sc = strict_counts(r2_matrix(fn(Craw), zt))
        others = np.delete(sc, 0)
        top = np.argsort(-sc)[:3]
        var_rows.append({"tissue": t, "variant": vname, "observed": int(sc[0]), "null_mean": float(others.mean()),
                         "null_q95": float(np.quantile(others, 0.95)), "p_exact": float((sc >= sc[0]).mean()),
                         "rank_of_observed": int(1 + (sc > sc[0]).sum()),
                         "top_classes": "; ".join(f"class {int(k) + 1}: {int(sc[k])}" for k in top)})
    # ---- positive control
    k0s = rng.choice(np.arange(1, 60), 10, replace=False)
    for target in ("all", "not dominated"):
        pool = np.arange(len(C)) if target == "all" else np.where(~dom)[0]
        for fc in (1, 2, 3, 5, 8):
            for k0 in k0s:
                Cm = Craw.copy()
                sel = rng.choice(pool, int(round(0.05 * len(C))), replace=False)
                if fc > 1:
                    tk = class_times(zt, int(k0))
                    phi = rng.uniform(0, 24, len(sel))
                    f = 2 ** ((np.log2(fc) / 2) * np.cos(phase.W * (tk[None, :] - phi[:, None])))
                    f = f / f.max(1, keepdims=True)
                    Cm[sel] = rng.binomial(Cm[sel].astype(int), f)
                Rm = r2_matrix(transform(Cm), zt)
                for r in tests(Rm, int(k0)):
                    pc_rows.append({"tissue": t, "injected_into": target, "fc": fc, "k0": int(k0) + 1,
                                    "n_injected": 0 if fc == 1 else len(sel),
                                    "injected_strict_best_share": float(np.mean(Rm[sel].argmax(1) == k0)) if fc > 1 else np.nan,
                                    **r})
    log(f"{t}: exact R2 test and positive control done")
ex = pd.DataFrame(ex_rows)
pc = pd.DataFrame(pc_rows)
write_csv(ex, MOD, "milo_r2_exact.csv", "Milo neighbourhoods: dispersion-free exact relabelling test (cosinor R2 of "
          "per-sample-centred log counts; observed class vs the 60 dihedral classes) per tissue and subset",
          required=["tissue", "subset", "statistic", "observed", "p_exact"])
write_csv(pd.DataFrame(var_rows), MOD, "milo_r2_variants.csv", "Dispersion-free exact relabelling test under three count "
          "normalisations: neighbourhoods with the observed order as strict best vs the 59 relabellings",
          required=["tissue", "variant", "observed", "p_exact"])
write_csv(pc, MOD, "milo_r2_positive_control.csv", "Positive control of the dispersion-free exact test: smooth rhythms "
          "injected into 5 % of real neighbourhoods aligned with a non-observed class; exact p of that class per fold change",
          required=["tissue", "fc", "k0", "statistic", "p_exact"])
summ = pc.groupby(["tissue", "injected_into", "statistic", "fc"]).agg(
    median_p=("p_exact", "median"), share_p_le_0_05=("p_exact", lambda s: float((s <= 0.05).mean())),
    injected_strict_best=("injected_strict_best_share", "median")).reset_index()
print(ex[ex.subset.isin(["all", "not dominated by one sample", "dominated by one sample"])].round(3).to_string())
print(ex[ex.subset.str.startswith("majority") & (ex.statistic.str.startswith("neigh"))].round(3).to_string())
print(summ.round(3).to_string())
log("done")
