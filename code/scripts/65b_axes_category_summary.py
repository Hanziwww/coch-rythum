"""Module 5.2c (summary) — category-level evidence for rhythmic migration/niche axes.

Individual axis × sender × receiver potentials are many and individually under-powered (no potential passes BH).
Here, per class × category, the fraction of potentials with permutation p <= 0.05 is compared with the 5 %
expected under the null (binomial test; potentials sharing senders/receivers are dependent, so the test is
descriptive), together with the circular mean peak ZT and resultant length of the nominally rhythmic potentials.
Output: results/65_migration_axes/migration_axes_category_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, write_csv  # noqa: E402

W = 2 * np.pi / 24
ax = pd.read_csv(RESULTS / "65_migration_axes/migration_axes_rhythm.csv")
rows = []
for (cls, cat), g in ax.groupby(["class", "category"]):
    k = int((g["p_perm"] <= 0.05).sum())
    n = len(g)
    sig = g[g["p_perm"] <= 0.05]
    z = np.exp(1j * W * sig["peak_zt"]).mean() if len(sig) else np.nan
    rows.append({"class": cls, "category": cat, "n_potentials": n, "n_perm_p05": k, "frac_perm_p05": k / n,
                 "enrichment_vs_5pct": (k / n) / 0.05, "binom_p_greater": binomtest(k, n, 0.05, alternative="greater").pvalue,
                 "circ_mean_peak_zt_sig": float((np.angle(z) / W) % 24) if len(sig) else np.nan,
                 "resultant_length_sig": float(np.abs(z)) if len(sig) else np.nan,
                 "top_axes": ";".join(sig.sort_values("p")["axis"].head(5) + "@" + sig.sort_values("p")["sender"].head(5)
                                      + "->" + sig.sort_values("p")["receiver"].head(5))})
write_csv(pd.DataFrame(rows).sort_values(["class", "binom_p_greater"]), "65_migration_axes",
          "migration_axes_category_summary.csv",
          "Per class × category: fraction of nominally rhythmic potentials vs 5 % null, common peak phase",
          required=["class", "category", "n_potentials", "frac_perm_p05", "binom_p_greater"])
