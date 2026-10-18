"""Round 5, WP-X (part 1) — ImmGen sorted B-lineage populations as an external reference for our B labels.

Data: ImmGen microarray phase 1 (GSE15907; RMA log2, Affymetrix MoGene 1.0 ST, probes → symbols via GPL6246.annot)
and ImmGen ULI RNA-seq (GSE109125; gene counts → log2 CPM). Populations kept: bone-marrow Hardy fractions (proB FrA,
FrB/C, preB FrC, FrD, B FrE immature, B FrF recirculating mature) and splenic T1 / T2 / T3, follicular (Fo), marginal-
zone (MZ), B1a (where present), CLP as the upstream anchor.
(1) immgen_label_mapping.csv: our pseudobulk B stages (pooled over the 6 samples, log2 CPM; BM and cochlea) vs every
    ImmGen population: Spearman correlation on the 2,000 most variable ImmGen-B genes shared with our matrix; best match
    and the positive-control expectation (pro-B → FrB/C; pre-B → FrC′/FrD; transitional → FrE / T1; naïve / mature →
    FrF / Fo).
(2) immgen_b_signatures.csv: external gene sets (no use of our data):
      stage signatures — per population, top 50 genes by mean log2 difference vs the other B populations (≥ 1,
                         expressed above the platform median in that population);
      axis FrF_vs_FrE  — recirculating-mature (FrF) vs immature (FrE): top 50 up and top 50 down (|log2FC| ≥ 1);
      axis Fo_vs_T1    — splenic follicular vs transitional-1 (RNA-seq or microarray, whichever has both).
    Immediate-early / stress genes (330 IEG set and Cd83) are removed from every set.
Outputs (results/r5_bcell/)
"""
import gzip
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r5_bcell"
G = EXTERNAL / "geo"
IEG = {"Fos", "Fosb", "Jun", "Junb", "Jund", "Egr1", "Egr2", "Egr3", "Atf3", "Ier2", "Ier3", "Nr4a1", "Zfp36", "Dusp1", "Klf2", "Klf4",
       "Klf6", "Btg2", "Ppp1r15a", "Gadd45b", "Socs3", "Cebpd", "Cd83", "Cd69", "Nr4a2", "Nr4a3"}
KEEP = re.compile(r"^(proB\.(CLP|FrA|FrBC|FrD)\.BM|preB\.(FrC|FrD)\.BM|B\.(FrE|FrF)\.BM|B\.(T1|T2|T3|Fo|MZ|B1a|Fem)\.Sp)", re.I)
BSTAGES = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
EXPECT = {"Pro-B Cell (Cycling)": ("FrBC", "FrA", "FrC"), "Pre-B Cell": ("FrC", "FrD"), "B Cell (Transitional)": ("FrE", "T1", "T2", "T3"),
          "B Cell (Naïve)": ("FrF", "Fo", "T2", "T3"), "B Cell (Mature)": ("FrF", "Fo", "MZ")}


def read_series_matrix(p):
    titles, rows = None, []
    with gzip.open(p, "rt", errors="replace") as fh:
        intable = False
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                intable = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif intable:
                rows.append(line)
    df = pd.read_csv(io.StringIO("".join(rows)), sep="\t", index_col=0)
    df.columns = titles
    return df


def pop_of(title):
    return re.sub(r"[#_].*$", "", re.sub(r"_RNA-seq$", "", title))


refs = {}
# ---- microarray
sm = G / "GSE15907/GSE15907_series_matrix.txt.gz"
an = G / "GSE15907/GPL6246.annot.gz"
if sm.exists() and an.exists():
    X = read_series_matrix(sm)
    X = X[[c for c in X.columns if KEEP.match(c)]]
    with gzip.open(an, "rt", errors="replace") as fh:
        lines = fh.readlines()
    start = [i for i, l in enumerate(lines) if l.startswith("!platform_table_begin")][0] + 1
    ann = pd.read_csv(io.StringIO("".join(lines[start:])), sep="\t", dtype=str, on_bad_lines="skip")
    sym = ann.set_index("ID")["Gene symbol"].dropna()
    sym = sym[~sym.str.contains("///")]
    X.index = X.index.astype(str)
    X = X.loc[X.index.intersection(sym.index)]
    X["symbol"] = sym.reindex(X.index).to_numpy()
    X = X.groupby("symbol").mean()
    refs["ImmGen microarray (GSE15907)"] = X
    log(f"GSE15907: {X.shape[0]} genes × {X.shape[1]} B-lineage samples: {sorted(set(pop_of(c) for c in X.columns))}")
# ---- RNA-seq
rs = G / "GSE109125/GSE109125_Gene_count_table_GENCODE_vM25.csv.gz"
if rs.exists():
    C = pd.read_csv(rs, index_col=0)
    gcol = [c for c in C.columns if not np.issubdtype(C[c].dtype, np.number)]
    if gcol:
        C = C.set_index(gcol[0])
    C = C[[c for c in C.columns if KEEP.match(c)]]
    C = C.groupby(level=0).sum()
    L = np.log2(C / C.sum(0) * 1e6 + 1)
    refs["ImmGen RNA-seq (GSE109125)"] = L
    log(f"GSE109125: {L.shape[0]} genes × {L.shape[1]} B-lineage samples: {sorted(set(pop_of(c) for c in L.columns))}")
assert refs, "no ImmGen data found"

# ---- our pooled stage profiles
ours = {}
for t in ["bm", "cochlea"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    for st in BSTAGES:
        cols = [c for c in pb.columns if c.split("|")[0] == st]
        v = pb[cols].sum(1)
        ours[(t, st)] = np.log2(v / v.sum() * 1e6 + 1)

maprows, sigrows = [], []
for rname, R in refs.items():
    popmean = R.T.groupby([pop_of(c) for c in R.columns]).mean().T
    var = popmean.var(1).sort_values(ascending=False)
    for (t, st), prof in ours.items():
        genes = [g for g in var.index if g in prof.index][:2000]
        cors = {p: stats.spearmanr(prof[genes], popmean.loc[genes, p])[0] for p in popmean.columns}
        best = max(cors, key=cors.get)
        exp = EXPECT[st]
        for p, r in cors.items():
            maprows.append({"reference": rname, "tissue": t, "our_stage": st, "immgen_population": p, "spearman_r": r,
                            "is_best": p == best, "expected": any(e.lower() in p.lower() for e in exp)})
    # signatures (external, our data not used)
    pops = list(popmean.columns)
    med = popmean.median().median()
    for p in pops:
        others = popmean.drop(columns=p).mean(1)
        d = (popmean[p] - others)
        d = d[(popmean[p] > med) & (d >= 1) & ~d.index.isin(IEG)].sort_values(ascending=False).head(50)
        for g, v in d.items():
            sigrows.append({"reference": rname, "signature": f"stage:{p}", "gene": g, "log2_diff": v, "direction": "up"})
    for a, b, name in [("FrF", "FrE", "axis:FrF_vs_FrE"), ("Fo", "T1", "axis:Fo_vs_T1")]:
        pa = [p for p in pops if re.search(rf"\.{a}\.", p + ".")]
        pb_ = [p for p in pops if re.search(rf"\.{b}\.", p + ".")]
        if not pa or not pb_:
            continue
        expressed = popmean[[pa[0], pb_[0]]].max(1) > med
        d = (popmean[pa[0]] - popmean[pb_[0]])[expressed & ~popmean.index.isin(IEG)]
        for g, v in d[d >= 1].sort_values(ascending=False).head(50).items():
            sigrows.append({"reference": rname, "signature": name, "gene": g, "log2_diff": v, "direction": "up (recirculating / mature)"})
        for g, v in d[d <= -1].sort_values().head(50).items():
            sigrows.append({"reference": rname, "signature": name, "gene": g, "log2_diff": v, "direction": "down (immature)"})
mp = pd.DataFrame(maprows)
write_csv(mp, MOD, "immgen_label_mapping.csv", "Our B stages (pooled pseudobulk) vs ImmGen sorted B-lineage populations: Spearman r, best match, expected fraction",
          required=["reference", "tissue", "our_stage", "immgen_population", "spearman_r", "is_best", "expected"])
sg = pd.DataFrame(sigrows)
write_csv(sg, MOD, "immgen_b_signatures.csv", "External B-lineage gene sets from ImmGen (stage signatures; FrF-vs-FrE and Fo-vs-T1 axes; IEG removed)",
          required=["reference", "signature", "gene", "direction"])
best = mp[mp["is_best"]].sort_values(["reference", "tissue", "our_stage"])
log("\n" + best[["reference", "tissue", "our_stage", "immgen_population", "spearman_r", "expected"]].round(3).to_string())
log("\n" + sg.groupby(["reference", "signature", "direction"]).size().to_string())
