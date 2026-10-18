"""Module 2.9 / 3d — promoter-motif driver attribution of rhythmic genes (replaces the RcisTarget step: the 0.5-GB
cisTarget ranking database could not be downloaded at the available bandwidth).

Promoters: UCSC mm10 RefSeq upstream 1 kb (upstream1000.fa.gz; one transcript per gene, lowest RefSeq accession,
NM_ preferred), gene symbols from refGene. Consensus motifs counted on both strands (overlapping matches):
  E-box CACGTG and E'-box CACGTT (CLOCK/BMAL1/NPAS2), D-box TTAYGTAA (DBP/TEF/HLF/NFIL3), RORE WAWNTRGGTCA
  (REV-ERB/ROR), GRE GAACAnnnTGTTC (glucocorticoid receptor), CRE TGACGTCA (cAMP/CREB, adrenergic), HSE
  GAAnnTTCnnGAA (HSF1, temperature), NF-kB GGGRNYYYCC (inflammation).
Per tissue × cell type (>= 30 Tier-1/2 rhythmic genes):
  (a) enrichment of motif presence among rhythmic vs tested non-rhythmic genes, CMH stratified by expression decile
      (BH over motifs within tissue × cell type); tissue-level CMH stratified by cell type × decile;
  (b) phase: circular mean peak ZT of rhythmic motif carriers, Kuiper test vs all rhythmic genes of the cell type
      (2000 random equal-size sets) and a directional test toward the literature-expected target phase
      (mean cos(peak − expected), same null; expected phases are priors: E-box ZT8, D-box ZT12, RORE ZT22, GRE/CRE
      ZT13, HSE ZT18);
  (c) motif presence by 6-h peak-phase bin among rhythmic genes (Fisher exact vs the other bins).
Outputs (results/44_motifs/): promoter_motif_presence.csv, promoter_motif_enrichment.csv,
promoter_motif_enrichment_tissue.csv, promoter_motif_phase.csv, promoter_motif_phase_bins.csv
"""
import gzip
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "44_motifs"
RES = EXTERNAL / "resources"
W = 2 * np.pi / 24
N_NULL = 2000
MOTIFS = {  # name: (regex on the forward strand, palindromic, expected target peak ZT or NaN)
    "Ebox_CACGTG": ("CACGTG", True, 8.0),
    "Eprime_CACGTT": ("CACGTT", False, 8.0),
    "Dbox_TTAYGTAA": ("TTA[CT]GTAA", False, 12.0),
    "RORE_WAWNTRGGTCA": ("[AT]A[AT][ACGT]T[AG]GGTCA", False, 22.0),
    "GRE_GAACAnnnTGTTC": ("GAACA[ACGT]{3}TGTTC", True, 13.0),
    "CRE_TGACGTCA": ("TGACGTCA", True, 13.0),
    "HSE_GAAnnTTCnnGAA": ("GAA[ACGT]{2}TTC[ACGT]{2}GAA", False, 18.0),
    "NFkB_GGGRNYYYCC": ("GGG[AG][ACGT][CT][CT][CT]CC", False, np.nan),
}
COMP = str.maketrans("ACGTN", "TGCAN")


def read_promoters():
    sym = pd.read_csv(RES / "refGene.txt.gz", sep="\t", header=None, usecols=[1, 12], names=["acc", "gene"])
    acc2gene = dict(zip(sym["acc"], sym["gene"]))
    seqs, acc, buf = {}, None, []
    with gzip.open(RES / "upstream1000.fa.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None and acc not in seqs:
                    seqs[acc] = "".join(buf).upper()
                acc, buf = line[1:].split("_up_")[0], []
            else:
                buf.append(line.strip())
    if acc is not None and acc not in seqs:
        seqs[acc] = "".join(buf).upper()
    rows = [(a, acc2gene.get(a), s) for a, s in seqs.items() if acc2gene.get(a)]
    df = pd.DataFrame(rows, columns=["accession", "gene", "seq"])
    df["pref"] = (~df["accession"].str.startswith("NM_")).astype(int)
    df["num"] = df["accession"].str.extract(r"_(\d+)")[0].astype(float)
    return df.sort_values(["gene", "pref", "num"]).drop_duplicates("gene").drop(columns=["pref", "num"])


def count(seq, rx, pal):
    n = len(rx.findall(seq))
    return n if pal else n + len(rx.findall(seq.translate(COMP)[::-1]))


def kuiper_masks(masks, last_idx, fb):
    fa = np.cumsum(masks, axis=1)[:, last_idx] / masks.sum(1, keepdims=True)
    d = fa - fb
    return d.max(1) + (-d).max(1)


def cmh(present, rhythmic, strata):
    """One-sided CMH test of motif presence × rhythmicity over strata; returns (MH OR, chi2, p)."""
    num = var = orn = ord_ = 0.0
    for k in np.unique(strata):
        m = strata == k
        N = m.sum()
        if N < 2:
            continue
        n1, m1 = present[m].sum(), rhythmic[m].sum()
        a = (present & rhythmic & m).sum()
        num += a - n1 * m1 / N
        var += n1 * m1 * (N - n1) * (N - m1) / (N ** 2 * (N - 1))
        b, c, d = n1 - a, m1 - a, N - n1 - m1 + a
        orn += a * d / N
        ord_ += b * c / N
    chi = num ** 2 / var if var > 0 else 0.0
    return (orn / ord_ if ord_ > 0 else np.inf), chi, (stats.chi2.sf(chi, 1) if num > 0 else 1.0)


def one_celltype(tissue, ct, g, pres, seed):
    rng = np.random.default_rng(seed)
    g = g.dropna(subset=["AveExpr"]).merge(pres, on="gene")
    rhy = g["tier"].isin(["tier1", "tier2"]).to_numpy()
    if rhy.sum() < 30:
        return [], [], []
    dec = pd.qcut(g["AveExpr"].rank(method="first"), 10, labels=False).to_numpy()
    phase = g["peak_zt"].to_numpy() % 24
    rhy_idx = np.where(rhy)[0][np.argsort(phase[rhy], kind="stable")]
    bg = phase[rhy_idx]
    last_idx = np.where(np.r_[bg[1:] != bg[:-1], True])[0]
    fb = (last_idx + 1) / len(bg)
    phase_bin = (bg // 6).astype(int)
    enr, ph, bins = [], [], []
    nulls = {}
    for motif, (_, _, exp_zt) in MOTIFS.items():
        present = g[motif].to_numpy() > 0
        orr, chi, p = cmh(present, rhy, dec)
        enr.append({"tissue": tissue, "cell_type": ct, "motif": motif, "n_tested": len(g), "n_rhythmic": int(rhy.sum()),
                    "frac_motif_rhythmic": present[rhy].mean(), "frac_motif_nonrhythmic": present[~rhy].mean(),
                    "mh_odds_ratio": orr, "cmh_chi2": chi, "p": p})
        carr = present[rhy_idx]
        n = int(carr.sum())
        for b in range(4):
            inb = phase_bin == b
            t = [[(carr & inb).sum(), (~carr & inb).sum()], [(carr & ~inb).sum(), (~carr & ~inb).sum()]]
            orb, pb = stats.fisher_exact(t, alternative="two-sided")
            bins.append({"tissue": tissue, "cell_type": ct, "motif": motif, "phase_bin": f"ZT{6 * b}-{6 * b + 6}",
                         "n_rhythmic_in_bin": int(inb.sum()), "n_carriers_in_bin": int((carr & inb).sum()),
                         "frac_carriers_in_bin": (carr & inb).sum() / max(inb.sum(), 1),
                         "frac_carriers_other_bins": (carr & ~inb).sum() / max((~inb).sum(), 1),
                         "odds_ratio": orb, "p_fisher": pb})
        if n < 5 or n > len(bg) - 5:
            continue
        if n not in nulls:
            pick = np.argpartition(rng.random((N_NULL, len(bg))), n, axis=1)[:, :n]
            masks = np.zeros((N_NULL, len(bg)), bool)
            np.put_along_axis(masks, pick, True, axis=1)
            nulls[n] = (kuiper_masks(masks, last_idx, fb), masks)
        nullV, masks = nulls[n]
        V = kuiper_masks(carr[None, :], last_idx, fb)[0]
        z = np.exp(1j * W * bg[carr]).mean()
        row = {"tissue": tissue, "cell_type": ct, "motif": motif, "n_rhythmic": len(bg), "n_carriers": n,
               "circ_mean_peak_zt": (np.angle(z) / W) % 24, "resultant_length": abs(z),
               "circ_mean_peak_zt_all_rhythmic": (np.angle(np.exp(1j * W * bg).mean()) / W) % 24,
               "kuiper_V": V, "p_kuiper_perm": (1 + (nullV >= V).sum()) / (N_NULL + 1), "expected_zt": exp_zt}
        if np.isfinite(exp_zt):
            c = np.cos(W * (bg - exp_zt))
            obs = c[carr].mean()
            nullc = (masks * c[None, :]).sum(1) / n
            row.update({"mean_cos_to_expected": obs, "p_directional_perm": (1 + (nullc >= obs).sum()) / (N_NULL + 1)})
        ph.append(row)
    return enr, ph, bins


pro = read_promoters()
log(f"{len(pro)} gene promoters")
for motif, (rx, pal, _) in MOTIFS.items():
    r = re.compile(f"(?=({rx}))")
    pro[motif] = [count(s, r, pal) for s in pro["seq"]]
pres = pro.drop(columns="seq")
write_csv(pres.melt(id_vars=["accession", "gene"], var_name="motif", value_name="n_sites"), MOD,
          "promoter_motif_presence.csv", "Motif site counts (both strands) in the 1-kb upstream region per gene",
          required=["gene", "motif", "n_sites"])
tasks = []
for tissue in ["cochlea", "bm"]:
    df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                     usecols=["group", "gene", "AveExpr", "tier", "peak_zt"])
    log(f"{tissue}: {df['gene'].isin(pres['gene']).mean():.3f} of tested gene rows have a promoter")
    tasks += [(tissue, ct, g) for ct, g in df.groupby("group")]
out = Parallel(n_jobs=int(os.environ.get("NJOBS", "3")))(delayed(one_celltype)(t, ct, g, pres, i)
                                                          for i, (t, ct, g) in enumerate(tasks))
enr = pd.DataFrame([r for o in out for r in o[0]])
enr["q_BH"] = enr.groupby(["tissue", "cell_type"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(enr, MOD, "promoter_motif_enrichment.csv",
          "Motif presence among rhythmic (Tier 1/2) vs non-rhythmic genes, CMH over expression deciles",
          required=["tissue", "cell_type", "motif", "mh_odds_ratio", "p", "q_BH"])
ph = pd.DataFrame([r for o in out for r in o[1]])
for col in ["p_kuiper_perm", "p_directional_perm"]:
    ph[col.replace("p_", "q_")] = ph.groupby(["tissue", "motif"])[col].transform(
        lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
write_csv(ph, MOD, "promoter_motif_phase.csv",
          "Peak phases of rhythmic motif carriers vs all rhythmic genes (Kuiper) and toward the expected phase",
          required=["tissue", "cell_type", "motif", "circ_mean_peak_zt", "p_kuiper_perm"])
bins = pd.DataFrame([r for o in out for r in o[2]])
bins["q_BH"] = bins.groupby(["tissue", "motif"])["p_fisher"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(bins, MOD, "promoter_motif_phase_bins.csv", "Motif carriers by 6-h peak-phase bin among rhythmic genes",
          required=["tissue", "cell_type", "motif", "phase_bin", "odds_ratio", "p_fisher"])
# tissue level: CMH stratified by cell type × expression decile
rows = []
for tissue in ["cochlea", "bm"]:
    df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                     usecols=["group", "gene", "AveExpr", "tier"]).dropna(subset=["AveExpr"]).merge(pres, on="gene")
    keep = df.groupby("group")["tier"].transform(lambda t: t.isin(["tier1", "tier2"]).sum() >= 30)
    df = df[keep].copy()
    df["stratum"] = df["group"] + "|" + df.groupby("group")["AveExpr"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)).astype(str)
    rhy = df["tier"].isin(["tier1", "tier2"]).to_numpy()
    for motif in MOTIFS:
        orr, chi, p = cmh(df[motif].to_numpy() > 0, rhy, df["stratum"].to_numpy())
        rows.append({"tissue": tissue, "motif": motif, "n_cell_types": df["group"].nunique(),
                     "n_gene_rows": len(df), "n_rhythmic_rows": int(rhy.sum()), "mh_odds_ratio": orr,
                     "cmh_chi2": chi, "p": p})
tis = pd.DataFrame(rows)
tis["q_BH"] = multipletests(tis["p"], method="fdr_bh")[1]
write_csv(tis, MOD, "promoter_motif_enrichment_tissue.csv",
          "Tissue-level motif enrichment among rhythmic genes, CMH over cell type × expression decile",
          required=["tissue", "motif", "mh_odds_ratio", "p", "q_BH"])
log("done")
