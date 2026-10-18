"""Round 10 (D133; the user asked for the noise analysis: 'I do not see any analysis related to noise') — partner-derived
(GFP+) cells in the temporal-bone marrow and cochlea of the one noise-exposed wild-type partner (31 days) compared with
the unexposed wild-type partners, from the tables of 934, counting partner-derived cells by the upgraded rule of 960
(D143; the 934 body counts are kept for reference). Run after 934 and 960 (local or server):
COCH_ROOT=/path/to/coch-rythum python scripts/937_parabiosis_noise.py

Units. Pairs are the unit of inference. The noise condition is one pair, so each of its pooled values (marrow density,
cochlear density, cochlea-to-marrow ratio; 934 rules, observed) is placed in the distribution of the unexposed pairs
(log scale): its z-score against their mean and SD, and the 95% prediction interval for one new unexposed pair
(t distribution with n - 1 df, s x sqrt(1 + 1/n)). Within animals, the analysed optical sections (>= 10 um apart; 934
planes table) are resampled with replacement within each acquisition (10,000 bootstrap draws) to give 95% intervals of
the pooled densities (bodies / tissue area) and of the noise / unexposed ratios against (i) the unexposed 31-day
partner (same pairing duration; marrow, and cochlea = osseous spiral lamina) and (ii) the cochlear apex of the
unexposed partners (same region as the noise-exposed apex field; 14-day apex acquisitions). These intervals describe
the sampling of sections within one animal per condition, not replication across animals. The perilymph of the noise
session is not compared (GFP detection failed in that session; 934). Rules fixed before computing.

Outputs (lib/io.write_csv, results/r10_parabiosis/): parabiosis_noise_sections.csv (one row per analysed section of
the compared acquisitions) and parabiosis_noise_comparison.csv (one row per comparison).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r10_parabiosis"
R10 = RESULTS / MOD
B, SEED = 10000, 937


def pooled_boot(sec, rng):
    """Pooled density and its bootstrap draws, resampling sections within each acquisition."""
    est = sec.gfp_cells.sum() / sec.tissue_mm2.sum()
    draws = np.zeros(B)
    groups = [g[["gfp_cells", "tissue_mm2"]].to_numpy() for _, g in sec.groupby("path")]
    for b in range(B):
        bod = area = 0.0
        for arr in groups:
            idx = rng.integers(0, len(arr), len(arr))
            bod += arr[idx, 0].sum()
            area += arr[idx, 1].sum()
        draws[b] = bod / area
    return est, draws


def main():
    rng = np.random.default_rng(SEED)
    pl = pd.read_csv(R10 / "parabiosis_planes.csv")
    tf = pd.read_csv(R10 / "parabiosis_tissue_fields.csv")
    rt = pd.read_csv(R10 / "parabiosis_ratio.csv")
    sm = pd.read_csv(R10 / "parabiosis_summary.csv")
    pl = pl.merge(tf[["path", "animal", "compartment", "subregion", "partner"]], on="path")
    pl = pl[(pl.partner == "WT") & pl.compartment.isin(["marrow", "cochlea"])].copy()
    # D143: partner-derived cells by the upgraded rule (960) per analysed section; sections without a body count 0
    cb = pd.read_csv(R10 / "survey_cells_bodies.csv").groupby(["path", "plane"]).cell.sum().rename("gfp_cells")
    pl = pl.merge(cb.reset_index(), on=["path", "plane"], how="left").fillna({"gfp_cells": 0})
    pl["density_per_mm2"] = pl.gfp_cells / pl.tissue_mm2
    sets = {
        "noise marrow": pl[(pl.animal == "31 d noise") & (pl.compartment == "marrow")],
        "noise cochlea (apex)": pl[(pl.animal == "31 d noise") & (pl.compartment == "cochlea")],
        "31 d marrow": pl[(pl.animal == "31 d") & (pl.compartment == "marrow")],
        "31 d cochlea (spiral lamina)": pl[(pl.animal == "31 d") & (pl.compartment == "cochlea")],
        "unexposed apex (14 d)": pl[(pl.animal == "14 d") & (pl.compartment == "cochlea") & (pl.subregion == "apex")],
    }
    sec = pd.concat([g.assign(set=k) for k, g in sets.items()], ignore_index=True)
    boot = {k: pooled_boot(g, rng) for k, g in sets.items()}
    rows = []
    for k, (est, d) in boot.items():
        rows.append(dict(comparison=f"{k}: pooled density", kind="pooled density (within-animal bootstrap)",
                         estimate=est, ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                         sections=len(sets[k]), acquisitions=sets[k].path.nunique(), unit="GFP+ cells per mm2"))
    for num, den, lab in [("noise marrow", "31 d marrow", "marrow, noise / unexposed 31 d"),
                          ("noise cochlea (apex)", "31 d cochlea (spiral lamina)",
                           "cochlea, noise apex / unexposed 31 d spiral lamina"),
                          ("noise cochlea (apex)", "unexposed apex (14 d)", "cochlear apex, noise / unexposed 14 d")]:
        (e1, d1), (e0, d0) = boot[num], boot[den]
        r = d1 / d0
        rows.append(dict(comparison=lab, kind="density ratio (within-animal bootstrap)", estimate=e1 / e0,
                         ci_low=np.percentile(r, 2.5), ci_high=np.percentile(r, 97.5),
                         sections=len(sets[num]) + len(sets[den]),
                         acquisitions=sets[num].path.nunique() + sets[den].path.nunique(), unit="ratio"))
    # pair level: the noise pair against the unexposed pairs (log scale)
    cs = pd.read_csv(R10 / "survey_cells_summary.csv")                 # D143: upgraded rule (960)
    cr = pd.read_csv(R10 / "survey_cells_ratio.csv").set_index("animal")
    vals = {"marrow density": cs[cs.compartment == "marrow"].set_index("animal").density_per_mm2,
            "cochlea density": cs[cs.compartment == "cochlea"].set_index("animal").density_per_mm2,
            "cochlea-to-marrow ratio": cr.cochlea_to_marrow}
    for lab, v in vals.items():
        x0 = np.log10(v.drop("31 d noise").astype(float))
        x1 = float(np.log10(v["31 d noise"]))
        n = len(x0)
        m, s = float(x0.mean()), float(x0.std(ddof=1))
        half = stats.t.ppf(0.975, n - 1) * s * np.sqrt(1 + 1 / n)
        rows.append(dict(comparison=f"noise pair vs unexposed pairs: {lab}", kind="pair level (log10)",
                         estimate=10 ** x1, ci_low=10 ** (m - half), ci_high=10 ** (m + half),
                         unexposed_pairs=n, unexposed_min=10 ** x0.min(), unexposed_max=10 ** x0.max(),
                         unexposed_geo_mean=10 ** m, z_vs_unexposed=(x1 - m) / s,
                         inside_prediction_interval=bool(abs(x1 - m) <= half),
                         unit="GFP+ cells per mm2" if "density" in lab else "ratio",
                         note="ci = 95% prediction interval for one new unexposed pair"))
    cmp_ = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(cmp_.round(3).to_string())
    write_csv(sec[["set", "path", "animal", "compartment", "subregion", "plane", "tissue_mm2", "gfp_bodies",
                   "gfp_cells", "density_per_mm2", "gfp_p999"]], MOD, "parabiosis_noise_sections.csv",
              "Parabiosis noise comparison (937): analysed optical sections of the noise-exposed and the compared "
              "unexposed acquisitions", required=["set", "path", "plane", "density_per_mm2"])
    write_csv(cmp_, MOD, "parabiosis_noise_comparison.csv", "Parabiosis noise comparison (937): pooled densities and "
              "noise / unexposed ratios with within-animal bootstrap intervals, and the noise pair against the "
              "unexposed pairs (z-score, 95% prediction interval)", required=["comparison", "estimate"])
    log("noise comparison written")


if __name__ == "__main__":
    main()
