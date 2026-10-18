"""Round 10 (D113) — R2: one exact test of the neutrophil-timer split (G02).

The split is a difference between two tissues: cochlear neutrophils predict the circulating ZT13 - ZT5 programme
(GSE86619) positively, marrow neutrophils negatively. 676 tested each tissue alone against the 60 dihedral
relabelings of its six samples (exact p >= 0.067 per correlation, G53). Cochlea and marrow come from different mice,
so their time labels can be relabelled independently, and the split statistic D = rho(cochlea) - rho(marrow) has an
exact null over the joint relabelings.

Primary (fixed before running; plan round 10): 676's statistic, re-implemented (676 runs at import): Spearman's rho
between the cosinor-predicted ZT13 - ZT5 change and the blood change, on log2-CPM pseudobulk, genes with mean CPM >= 5
shared with GSE86619, no gene selection; units Neutrophil (Aged) (as Fig. 6a) and pooled neutrophils; the 60 class
representatives per tissue (identity first) as in 676, so the joint null has 60 x 60 = 3,600 relabelings; one-sided p
in the predicted direction (D >= observed), two-sided reported. Baseline: each tissue's rho and marginal exact p equal
676's (results/r6_myeloid/m_timer_split_verification.csv).
Secondary (declared before running): all 720 permutations of the six time labels per tissue (720 x 720 joint null).
Because rho depends on phase, the 720 permutations include the phase-shifted and reversed versions of the true
labels, which the 60 class representatives do not; this is the stricter null. Also the other three neutrophil states.
Rule: primary one-sided p <= 0.05 -> the Results sentence on single correlations is replaced by the joint test;
otherwise the joint test is reported in Extended Data with the current wording.
Outputs (results/r10_reanalysis/): timer_split_joint_null.csv (tests), timer_split_joint_null_distribution.csv (the
primary null distribution per unit; ED9h)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r10_reanalysis"
W = 2 * np.pi / 24
DC, DS = np.cos(13 * W) - np.cos(5 * W), np.sin(13 * W) - np.sin(5 * W)
CLS = cosinor.dihedral_classes(6)
PERMS = list(itertools.permutations(range(6)))
assert PERMS[0] == tuple(range(6)) and CLS[0] == tuple(range(6))
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
UNITS = [("Neutrophil (Aged)", ["Neutrophil (Aged)"], True), ("pooled neutrophils", NEU, True)] + \
    [(ty, [ty], False) for ty in NEU if ty != "Neutrophil (Aged)"]
obs = pd.read_csv(RESULTS / "77_neutrophil_timer/neutrophil_timer_gene_table.csv").set_index("gene")["wt_zt13_vs_zt5_log2fc"].dropna()
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
v676 = pd.read_csv(RESULTS / "r6_myeloid/m_timer_split_verification.csv")
v676 = v676[(v676["part"] == "A/B our data (raw, as G02)") & (v676["gene_set"] == "all shared genes")].set_index(["tissue", "unit"])


def lcpm(pb, types, samples):                                 # as 676
    ok = [ty for ty in types if all(f"{ty}|{s}" in pb.columns for s in samples)]
    M = sum(pb[[f"{ty}|{s}" for s in samples]].set_axis(samples, axis=1) for ty in ok)
    cpm = M / M.sum(0) * 1e6
    return np.log2(cpm + 1), cpm.mean(1)


def rho_of(Y, zt, ob):                                        # as 676
    f = cosinor.fit(Y.T, zt)
    return stats.spearmanr(f["a"] * DC + f["b"] * DS, ob)[0]


rhos = {}                                                     # (tissue, unit) -> (rho over CLS, rho over PERMS, n genes)
for t in ["bm", "cochlea"]:
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")]
    s0 = d0.drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    for unit, types, is_primary in UNITS:
        if not all(f"{ty}|{s}" in pb.columns for ty in types for s in samples):
            assert not is_primary, (t, unit)
            log(f"{t}: {unit} lacks pseudobulk columns for some samples; skipped")
            continue
        L, m = lcpm(pb, types, samples)
        genes = [g for g in L.index[(m >= 5).to_numpy()] if g in obs.index]
        Y, ob = L.loc[genes].to_numpy(float), obs[genes].to_numpy()
        r60 = np.array([rho_of(Y, zt[list(c)], ob) for c in CLS])
        r720 = np.array([rho_of(Y, zt[list(p)], ob) for p in PERMS])
        assert np.isclose(r60[0], r720[0])
        if (t, unit) in v676.index:                           # baseline: 676's rho and marginal exact p
            b = v676.loc[(t, unit)]
            assert np.isclose(r60[0], b["spearman"], atol=1e-12), (t, unit)
            assert np.isclose(np.mean(r60 >= r60[0] - 1e-12), b["p_exact_positive"], atol=1e-12), (t, unit)
            assert np.isclose(np.mean(r60 <= r60[0] + 1e-12), b["p_exact_negative"], atol=1e-12), (t, unit)
        else:
            assert not is_primary, (t, unit)
        rhos[(t, unit)] = (r60, r720, len(genes))
    log(f"{t}: rho computed for {len(UNITS)} units over 60 classes and 720 permutations; 676 baseline reproduced")

rows, dist = [], []
for unit, _, primary in UNITS:
    if ("cochlea", unit) not in rhos or ("bm", unit) not in rhos:
        continue
    for null, k in (("60 x 60 class representatives", 0), ("720 x 720 permutations", 1)):
        rc, rm = rhos[("cochlea", unit)][k], rhos[("bm", unit)][k]
        d_obs = rc[0] - rm[0]
        D = (rc[:, None] - rm[None, :]).ravel()
        if k == 0:                                            # the primary null distribution (figure-ready; ED9)
            dist.append(pd.DataFrame({"unit": unit, "null": null, "split_D": D,
                                      "is_observed": np.arange(D.size) == 0}))
        rows.append({"unit": unit, "null": null, "primary": bool(primary and k == 0), "n_relabelings": D.size,
                     "rho_cochlea": rc[0], "rho_marrow": rm[0], "split_D": d_obs,
                     "p_cochlea_positive": float(np.mean(rc >= rc[0] - 1e-12)),
                     "p_marrow_negative": float(np.mean(rm <= rm[0] + 1e-12)),
                     "p_joint_one_sided": float(np.mean(D >= d_obs - 1e-12)),
                     "p_joint_two_sided": float(np.mean(np.abs(D) >= abs(d_obs) - 1e-12)),
                     "n_genes_cochlea": rhos[("cochlea", unit)][2], "n_genes_marrow": rhos[("bm", unit)][2]})
res = pd.DataFrame(rows)
write_csv(res, MOD, "timer_split_joint_null.csv", "R2: the neutrophil-timer split as one statistic (rho cochlea - rho "
          "marrow) against the exact joint relabelling null of both tissues (60 x 60 primary; 720 x 720 secondary) (677)",
          required=["unit", "null", "primary", "split_D", "p_joint_one_sided", "p_joint_two_sided"])
write_csv(pd.concat(dist, ignore_index=True), MOD, "timer_split_joint_null_distribution.csv", "R2: the split statistic "
          "over all 3,600 joint relabelings (60 x 60 class representatives; identity = observed) per neutrophil unit (677)",
          required=["unit", "null", "split_D", "is_observed"])
pd.set_option("display.width", 250)
log("\n" + res.round(4).to_string(index=False))
