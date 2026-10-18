"""Distribution-level rhythm test for one pooled sample per time point (round 6; D42).

ot_rhythm_test: per-cell values of the 6 samples (ordered by ZT, equally spaced) → 1-D Wasserstein (optimal-transport)
distance matrix D between samples. A 24-h rhythm in the whole distribution makes antiphase samples (12 h apart) differ
more than adjacent samples (4 h apart): statistic R = mean D(antiphase pairs) / mean D(adjacent pairs). Exact p over the
60 dihedral relabeling classes (relabelings change which samples count as adjacent / antiphase); cells are never
resampled, so the sample stays the unit.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from lib import cosinor

CLASSES = cosinor.dihedral_classes(6)


def _ratio(D, lab):
    """lab[i] = time slot (0..5) of sample i."""
    pos = {s: i for i, s in enumerate(lab)}
    anti = [D[pos[k], pos[(k + 3) % 6]] for k in range(3)]
    adj = [D[pos[k], pos[(k + 1) % 6]] for k in range(6)]
    return float(np.mean(anti) / max(np.mean(adj), 1e-12))


def ot_rhythm_test(values, max_cells=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in values]
    vals = [v if len(v) <= max_cells else rng.choice(v, max_cells, replace=False) for v in vals]
    D = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            D[i, j] = D[j, i] = wasserstein_distance(vals[i], vals[j])
    obs = _ratio(D, list(range(6)))
    null = [_ratio(D, list(p)) for p in CLASSES[1:]]
    return {"ot_ratio": obs, "ot_p_perm": (1 + sum(x >= obs - 1e-12 for x in null)) / len(CLASSES),
            "ot_mean_antiphase": float(np.mean([D[k, (k + 3) % 6] for k in range(3)])),
            "ot_mean_adjacent": float(np.mean([D[k, (k + 1) % 6] for k in range(6)]))}
