"""Module 3b/3c — competitive enrichment and phase-set enrichment (PSEA) of rhythmic genes per tissue × cell type.

(b) Competitive over-representation of Tier-1/2 rhythmic genes in each gene set against the cell type's tested
    genes, with expression-bias correction: Cochran–Mantel–Haenszel test stratified by expression decile
    (rhythm detection power rises with expression; Laloum 2020). MH odds ratio, CMH p, BH within tissue × cell type ×
    resource.
(c) PSEA: for sets with >= 5 rhythmic members, two-sample Kuiper test of member peak phases against the phases of
    all rhythmic genes of that cell type (the background phase distribution is not uniform), permutation p
    (2000 draws of random member sets of equal size; one null per member-set size and cell type, shared by all
    sets of that size); circular mean phase and resultant length of members. Vectorised across gene sets; cell
    types run in parallel (env NJOBS).
Outputs (results/42_enrichment/): rhythmic_gene_enrichment.csv, psea_phase_enrichment.csv
Round 2: env TIERC=1 runs the same tests for the cochlea with the calibrated relaxed tier (Tier C, 211) as the rhythmic
set → results/r2_cochlea/rhythmic_gene_enrichment_tierC.csv, psea_phase_enrichment_tierC.csv (exploratory).
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

TIERC = os.environ.get("TIERC", "0") == "1"
MOD = "r2_cochlea" if TIERC else "42_enrichment"
SFX = "_tierC" if TIERC else ""
RES = EXTERNAL / "resources"
W = 2 * np.pi / 24


def read_gmt(path, lo=10, hi=500):
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


N_NULL = 2000


def incidence(genes, sets):
    """Sparse sets × genes 0/1 matrix over the cell type's tested genes (sets with >= 5 tested members)."""
    idx = {g: i for i, g in enumerate(genes)}
    rows, cols, names = [], [], []
    for sname, members in sets.items():
        hit = [idx[m] for m in members if m in idx]
        if len(hit) < 5:
            continue
        rows += [len(names)] * len(hit)
        cols += hit
        names.append(sname)
    return names, sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(genes)))


def kuiper_masks(masks, last_idx, fb):
    """Two-sample Kuiper V of member subsets (boolean rows over the phase-sorted background) vs the background.
    Members are a subset of the background, so the pooled grid is the background's unique phases."""
    fa = np.cumsum(masks, axis=1)[:, last_idx] / masks.sum(1, keepdims=True)
    d = fa - fb
    return d.max(1) + (-d).max(1)


def one_celltype(tissue, ct, g, seed):
    rng = np.random.default_rng(seed)
    g = g.dropna(subset=["AveExpr"]).reset_index(drop=True)
    rhy = g["tier"].isin(["tier1", "tier2"]).to_numpy()
    if rhy.sum() < 30:
        return [], [], None
    dec = pd.qcut(g["AveExpr"].rank(method="first"), 10, labels=False).to_numpy()
    genes = g["gene"].to_numpy()
    D = np.zeros((len(g), 10))
    D[np.arange(len(g)), dec] = 1
    N, m1 = D.sum(0), D[rhy].sum(0)
    ok = N >= 2
    phase = g["peak_zt"].to_numpy() % 24
    rhy_idx = np.where(rhy)[0][np.argsort(phase[rhy], kind="stable")]   # rhythmic genes in phase order
    bg = phase[rhy_idx]
    last_idx = np.where(np.r_[bg[1:] != bg[:-1], True])[0]
    fb = (last_idx + 1) / len(bg)
    nulls = {}

    def null_for(n):                       # Kuiper null of random member sets of size n (shared across sets)
        if n not in nulls:
            pick = np.argpartition(rng.random((N_NULL, len(bg))), n, axis=1)[:, :n]
            masks = np.zeros((N_NULL, len(bg)), bool)
            np.put_along_axis(masks, pick, True, axis=1)
            nulls[n] = np.sort(kuiper_masks(masks, last_idx, fb))
        return nulls[n]

    enr, psea = [], []
    for rname, sets in resources.items():
        names, M = incidence(genes, sets)
        if not names:
            continue
        n1 = np.asarray(M @ D)                     # set members per expression decile
        a = np.asarray(M @ (D * rhy[:, None]))     # rhythmic set members per decile
        n1k, ak, Nk, m1k = n1[:, ok], a[:, ok], N[ok], m1[ok]
        num = (ak - n1k * m1k / Nk).sum(1)
        var = (n1k * m1k * (Nk - n1k) * (Nk - m1k) / (Nk ** 2 * (Nk - 1))).sum(1)
        b, c, d = n1k - ak, m1k - ak, Nk - n1k - m1k + ak
        orn, ord_ = (ak * d / Nk).sum(1), (b * c / Nk).sum(1)
        chi = np.where(var > 0, num ** 2 / np.where(var > 0, var, 1), 0.0)
        r = pd.DataFrame({"tissue": tissue, "cell_type": ct, "resource": rname, "gene_set": names,
                          "n_tested_in_set": np.asarray(M.sum(1)).ravel().astype(int),
                          "n_rhythmic_in_set": a.sum(1).astype(int), "n_rhythmic_total": int(rhy.sum()),
                          "n_tested_total": len(g),
                          "mh_odds_ratio": np.where(ord_ > 0, orn / np.where(ord_ > 0, ord_, 1), np.inf),
                          "cmh_chi2": chi, "p": np.where(num > 0, stats.chi2.sf(chi, 1), 1.0)})
        r["q_BH"] = multipletests(r["p"], method="fdr_bh")[1]
        enr.append(r)
        Mr = M[:, rhy_idx].toarray().astype(bool)  # sets × rhythmic genes (phase order)
        nm = Mr.sum(1)
        sel = np.where(nm >= 5)[0]
        if len(sel):
            V = kuiper_masks(Mr[sel], last_idx, fb)
            p_perm = np.array([(1 + N_NULL - np.searchsorted(null_for(int(nm[s])), v, side="left")) / (N_NULL + 1)
                               for s, v in zip(sel, V)])
            z = (Mr[sel] * np.exp(1j * W * bg)[None, :]).sum(1) / nm[sel]
            psea.append(pd.DataFrame({"tissue": tissue, "cell_type": ct, "resource": rname,
                                      "gene_set": [names[s] for s in sel], "n_rhythmic_members": nm[sel],
                                      "circ_mean_peak_zt": (np.angle(z) / W) % 24, "resultant_length": np.abs(z),
                                      "kuiper_V": V, "p_perm": p_perm,
                                      "members": [";".join(genes[np.sort(rhy_idx[Mr[s]])]) for s in sel]}))
    return enr, psea, f"{tissue} {ct}: {rhy.sum()} rhythmic genes"


tasks = []
if TIERC:
    df = pd.read_csv(RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv", usecols=["cell_type", "gene", "AveExpr", "tierC", "peak_zt"])
    df = df.rename(columns={"cell_type": "group"})
    df["tier"] = np.where(df["tierC"].astype(str) == "True", "tier1", "none")   # Tier C plays the role of Tier 1/2
    tasks += [("cochlea", ct, g) for ct, g in df.groupby("group")]
else:
    for tissue in ["cochlea", "bm"]:
        df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                         usecols=["group", "gene", "AveExpr", "tier", "peak_zt"])
        tasks += [(tissue, ct, g) for ct, g in df.groupby("group")]
enr_rows, psea_rows = [], []
for enr, psea, msg in Parallel(n_jobs=int(os.environ.get("NJOBS", "3")), return_as="generator")(
        delayed(one_celltype)(t, ct, g, i) for i, (t, ct, g) in enumerate(tasks)):
    enr_rows += enr
    psea_rows += psea
    if msg:
        log(msg)
enr = pd.concat(enr_rows, ignore_index=True)
write_csv(enr, MOD, f"rhythmic_gene_enrichment{SFX}.csv",
          "Expression-stratified (CMH) over-representation of rhythmic genes in gene sets per tissue × cell type",
          required=["tissue", "cell_type", "resource", "gene_set", "mh_odds_ratio", "p", "q_BH"])
ps = pd.concat(psea_rows, ignore_index=True) if psea_rows else pd.DataFrame()
if len(ps):
    ps["q_BH"] = ps.groupby(["tissue", "cell_type", "resource"])["p_perm"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(ps, MOD, f"psea_phase_enrichment{SFX}.csv",
          "Phase-set enrichment: Kuiper test of rhythmic member phases vs the cell type's rhythmic background",
          required=["tissue", "cell_type", "resource", "gene_set", "circ_mean_peak_zt", "kuiper_V", "p_perm"])
log("done")
