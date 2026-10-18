"""Module 3b (threshold-free) — competitive, expression-stratified enrichment of rhythmicity in gene sets.

Complements the thresholded CMH test of 42 (Tier-1/2 vs rest). Per tissue × cell type, every tested gene gets a
rhythmicity score s_g = Φ⁻¹(1 − p_g/2) (normal score of the primary limma moderated-F p-value). A set's statistic is
the mean score of its tested members; under the competitive null its members are a random draw of the same size from
each expression decile (stratified sampling without replacement), so the null mean and variance follow exactly from
the decile means/variances with the finite-population correction:
  E = Σ_k n_k μ_k / n,   Var = Σ_k n_k σ_k² (N_k − n_k)/(N_k − 1) / n²,
z = (obs − E)/√Var, one-sided p (higher rhythmicity), BH within tissue × cell type × resource. The circular mean peak
phase of the set's top-quartile-scoring members is reported as a phase summary. Gene–gene correlation inflates z for
co-regulated sets (as in any competitive test without a correlation correction).
Output (results/42_enrichment/): rhythmicity_ranked_enrichment.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "42_enrichment"
RES = EXTERNAL / "resources"
W = 2 * np.pi / 24


def read_gmt(path, lo, hi):
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if lo <= len(p) - 2 <= hi:
            out[p[0]] = set(p[2:])
    return out


resources = {"curated": {f"{c}:{n}": set(g) for c, d in yaml.safe_load(
    open(Path(__file__).resolve().parent / "resources/curated_sets.yaml")).items() for n, g in d.items()}}
for name, fn, lo, hi in [("hallmark", "msigdb_mh_hallmark.gmt", 10, 500), ("reactome", "msigdb_m2_reactome.gmt", 10, 500),
                         ("gobp", "msigdb_m5_gobp.gmt", 15, 300)]:
    if (RES / fn).exists():
        resources[name] = read_gmt(RES / fn, lo, hi)


def one(tissue, ct, g):
    g = g.dropna(subset=["AveExpr", "p"]).reset_index(drop=True)
    if len(g) < 500:
        return []
    s = stats.norm.isf(np.clip(g["p"].to_numpy(), 1e-300, 1) / 2)
    dec = pd.qcut(g["AveExpr"].rank(method="first"), 10, labels=False).to_numpy()
    D = np.zeros((len(g), 10))
    D[np.arange(len(g)), dec] = 1
    Nk = D.sum(0)
    mu = (D * s[:, None]).sum(0) / Nk
    var = (D * (s[:, None] - mu) ** 2).sum(0) / Nk                       # population variance per decile
    idx = {x: i for i, x in enumerate(g["gene"])}
    top = s >= np.quantile(s, 0.75)
    ph = np.exp(1j * W * (g["peak_zt"].to_numpy() % 24))
    out = []
    for rname, sets in resources.items():
        rows, cols, names = [], [], []
        for sname, mem in sets.items():
            hit = [idx[m] for m in mem if m in idx]
            if len(hit) >= 10:
                rows += [len(names)] * len(hit)
                cols += hit
                names.append(sname)
        if not names:
            continue
        M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(g)))
        n = np.asarray(M.sum(1)).ravel()
        nk = np.asarray(M @ D)
        obs = np.asarray(M @ s).ravel() / n
        E = (nk * mu).sum(1) / n
        V = (nk * var * (Nk - nk) / np.maximum(Nk - 1, 1)).sum(1) / n ** 2
        z = (obs - E) / np.sqrt(np.maximum(V, 1e-12))
        mt = M.multiply(top[None, :]).tocsr()
        nt = np.asarray(mt.sum(1)).ravel()
        zph = np.asarray(mt @ ph).ravel() / np.maximum(nt, 1)
        r = pd.DataFrame({"tissue": tissue, "cell_type": ct, "resource": rname, "gene_set": names, "n_tested": n,
                          "mean_score": obs, "expected_score": E, "z": z, "p": stats.norm.sf(z),
                          "n_top_quartile": nt.astype(int),
                          "top_quartile_circ_mean_peak_zt": np.where(nt > 0, (np.angle(zph) / W) % 24, np.nan),
                          "top_quartile_resultant": np.where(nt > 0, np.abs(zph), np.nan)})
        r["q_BH"] = multipletests(r["p"], method="fdr_bh")[1]
        out.append(r)
    return out


tasks = []
for tissue in ["cochlea", "bm"]:
    df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                     usecols=["group", "gene", "AveExpr", "p", "peak_zt"])
    tasks += [(tissue, ct, g) for ct, g in df.groupby("group")]
res = Parallel(n_jobs=int(os.environ.get("NJOBS", "3")))(delayed(one)(t, c, g) for t, c, g in tasks)
out = pd.concat([r for o in res for r in o], ignore_index=True)
write_csv(out, MOD, "rhythmicity_ranked_enrichment.csv",
          "Threshold-free expression-stratified competitive enrichment of rhythmicity (normal scores of limma p)",
          required=["tissue", "cell_type", "resource", "gene_set", "z", "p", "q_BH"])
log(f"done: {len(out)} tests, q<0.05: {(out['q_BH'] < 0.05).sum()}")
