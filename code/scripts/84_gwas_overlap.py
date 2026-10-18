"""Module 6.7 — human genetic relevance of rhythmic programmes (GWAS Catalog gene overlap).

GWAS Catalog associations (all studies) are filtered by trait text into hearing (hearing loss/impairment/difficulty,
age-related hearing, deafness, tinnitus) and chronotype/circadian (chronotype, morningness, sleep timing, circadian)
gene sets (MAPPED_GENE, p <= 5e-8), mapped to mouse with 1:1 MGI orthologues. For each tissue × cell type, the
overlap with Tier-1/2 rhythmic genes is tested against the cell type's tested genes (hypergeometric), BH per trait.
GWAS-mapped genes are long on average and long genes are more often detected and called rhythmic, so a one-sided
CMH test stratified by gene-length decile (Ensembl GRCm39 gene span) × expression decile (AveExpr) is reported as the
bias-controlled test (cmh_p, cmh_q).
(Summary statistics for gene-level MAGMA were not available offline; catalogue overlap is the documented substitute.)
Outputs (results/84_gwas/): gwas_trait_genes.csv, gwas_enrichment.csv
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "84_gwas"
RES = EXTERNAL / "resources"
TRAITS = {"hearing": r"hearing loss|hearing impair|hearing difficult|age-related hearing|deafness|tinnitus|hearing",
          "chronotype_circadian": r"chronotype|morningness|morning person|sleep timing|circadian|diurnal preference"}
gw = pd.read_csv(RES / "gwas_catalog_associations.tsv", sep="\t", low_memory=False,
                 usecols=["DISEASE/TRAIT", "MAPPED_GENE", "P-VALUE", "PUBMEDID", "STUDY"])   # v1.0 file layout
gw = gw[pd.to_numeric(gw["P-VALUE"], errors="coerce") <= 5e-8]
hom = pd.read_csv(RES / "mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
one = [(h[0], m[0]) for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1]
h2m = dict(one)
trait_rows, sets = [], {}
for t, rx in TRAITS.items():
    sub = gw[gw["DISEASE/TRAIT"].str.contains(rx, case=False, na=False)]
    genes = set()
    for s in sub["MAPPED_GENE"].dropna():
        genes |= {g.strip() for g in re.split(r"[,;]| - | x ", s) if g.strip()}   # keep hyphenated symbols
    mouse = {h2m[g] for g in genes if g in h2m}
    sets[t] = mouse
    trait_rows += [{"trait_group": t, "human_gene": g, "mouse_gene": h2m.get(g, "")} for g in sorted(genes)]
    log(f"{t}: {len(sub)} associations from {sub['PUBMEDID'].nunique()} publications, {len(genes)} human genes, "
        f"{len(mouse)} mouse orthologues")
write_csv(pd.DataFrame(trait_rows), MOD, "gwas_trait_genes.csv", "GWAS Catalog genes per trait group (p <= 5e-8)",
          required=["trait_group", "human_gene", "mouse_gene"])
glen = {}
with gzip.open(RES / "Mus_musculus.GRCm39.113.gtf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("	", 9)
        if f[2] != "gene":
            continue
        m = re.search(r'gene_name "([^"]+)"', f[8])
        if m:
            glen[m.group(1)] = max(glen.get(m.group(1), 0), int(f[4]) - int(f[3]) + 1)
log(f"gene spans for {len(glen)} symbols")


def cmh_greater(x, y, strata):
    """One-sided CMH test of association between two boolean vectors over strata; returns (MH OR, p)."""
    num = var = orn = ord_ = 0.0
    for k in np.unique(strata):
        m = strata == k
        N = m.sum()
        if N < 2:
            continue
        n1, m1 = x[m].sum(), y[m].sum()
        a = (x & y & m).sum()
        num += a - n1 * m1 / N
        var += n1 * m1 * (N - n1) * (N - m1) / (N ** 2 * (N - 1))
        orn += a * (N - n1 - m1 + a) / N
        ord_ += (n1 - a) * (m1 - a) / N
    if var <= 0 or num <= 0:
        return (orn / ord_ if ord_ > 0 else np.nan), 1.0
    from scipy.stats import norm
    return (orn / ord_ if ord_ > 0 else np.inf), float(norm.sf(num / np.sqrt(var)))


rows = []
for tissue in ["cochlea", "bm"]:
    df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                     usecols=["group", "gene", "tier", "AveExpr"])
    for ct, g in df.groupby("group"):
        bg = set(g["gene"])
        rh = set(g.loc[g["tier"].isin(["tier1", "tier2"]), "gene"])
        if len(rh) < 20:
            continue
        gl = g[g["gene"].isin(glen.keys())].dropna(subset=["AveExpr"]).copy()
        gl["len_dec"] = pd.qcut(np.log10(gl["gene"].map(glen)).rank(method="first"), 10, labels=False)
        gl["expr_dec"] = pd.qcut(gl["AveExpr"].rank(method="first"), 10, labels=False)
        strata = (gl["len_dec"] * 10 + gl["expr_dec"]).to_numpy()
        is_rh = gl["tier"].isin(["tier1", "tier2"]).to_numpy()
        for t, s in sets.items():
            K = len(s & bg)
            k = len(s & rh)
            p = hypergeom.sf(k - 1, len(bg), K, len(rh)) if K else 1.0
            cor, cp = cmh_greater(gl["gene"].isin(s).to_numpy(), is_rh, strata)
            rows.append({"tissue": tissue, "cell_type": ct, "trait_group": t, "n_background": len(bg),
                         "n_trait_in_background": K, "n_rhythmic": len(rh), "n_overlap": k,
                         "fold_enrichment": (k / len(rh)) / (K / len(bg)) if K else np.nan, "p": p,
                         "cmh_or_len_expr": cor, "cmh_p": cp,
                         "median_span_kb_trait": np.median([glen[x] for x in s & bg if x in glen]) / 1e3 if K else np.nan,
                         "median_span_kb_background": np.median([glen[x] for x in bg if x in glen]) / 1e3,
                         "overlap_genes": ";".join(sorted(s & rh))})
r = pd.DataFrame(rows)
r["q_BH"] = r.groupby("trait_group")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
r["cmh_q"] = r.groupby("trait_group")["cmh_p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(r, MOD, "gwas_enrichment.csv", "Overlap of rhythmic genes with GWAS trait genes per tissue × cell type",
          required=["tissue", "cell_type", "trait_group", "n_overlap", "fold_enrichment", "p", "q_BH", "cmh_p", "cmh_q"])
log("done")
