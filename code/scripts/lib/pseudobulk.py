"""Pseudobulk aggregation: sum raw counts per (sample × group) without densifying the full matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def aggregate(X, groups: pd.Series) -> tuple[sp.csr_matrix, pd.Index]:
    """Sum rows of X (cells × genes) by group label. Returns (n_groups × genes CSR, group index)."""
    codes, uniques = pd.factorize(groups, sort=True)
    ind = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
                        shape=(len(uniques), len(codes)))
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    return (ind @ X).tocsr(), pd.Index(uniques)


def by_sample_and_label(adata, label: str = "cell_type", sample: str = "batch_id"):
    """Pseudobulk counts per (sample, label). Returns (counts CSR, meta DataFrame with n_cells)."""
    obs = adata.obs
    key = obs[sample].astype(str) + "||" + obs[label].astype(str)
    counts, idx = aggregate(adata.X, key)
    parts = idx.str.split(r"\|\|", regex=True)
    meta = pd.DataFrame({sample: [p[0] for p in parts], label: [p[1] for p in parts]})
    meta["n_cells"] = key.value_counts().reindex(idx).to_numpy()
    meta["total_counts"] = np.asarray(counts.sum(1)).ravel()
    return counts, meta
