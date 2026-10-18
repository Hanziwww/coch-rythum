"""Round 5, WP-X (part 2) — external data for the B-cell hypotheses (agreeing methods) and the exploratory screen.

Signatures: ImmGen microarray stage signatures (560; proB.FrBC, preB.FrC, preB.FrD, B.FrE, B.FrF) and the FrF-vs-FrE axis.
(A) GSE198770 bone marrow (bulk, 6 ZTs × 3, control and neutrophil-depleted; prepared_r2): per sample mean z of log2
    expression of each signature; cosinor per condition (all replicates); peak converted to our BM ZT equivalent via the
    clock offsets (230: internal = peak − dataset offset; our ZT = internal + our BM offset). Agreeing evidence:
    H1 — FrF (recirculating mature) peak in ZT16–24; H2 — forward-lag statistic over FrBC → FrC → FrD → FrE → FrF
    (ImmGen developmental order) < 24 h, exact p over the 120 permutations; H3 — FrF-vs-FrE axis peak in ZT12–24.
    Neutrophil depletion: condition × rhythm interaction F test per signature.
(B) Baboon GSE98965 spleen (own clock offset) and bone marrow (imputed offset; 440): the same signatures (human
    orthologues), cosinor over 12 time points, aligned and converted to our BM ZT equivalent.
(C) GSE175635 intestinal IgA+ plasma cells (ZT0 / 6 / 12 / 18, n = 5; WT): clock offset vs canonical phases (230
    weighting) — an external B-lineage clock (H4 context); phase concordance of genes rhythmic in IgA PCs and in our
    B stages (within 3 h after alignment; gene-permutation null).
(D) GSE175609 IgA+ PCs, ZT0 vs ZT12, WT vs Mb1-Cre Bmal1 KO (H4 agreeing method): per gene log2FC(ZT12 − ZT0) in WT and
    KO; clock-gene positive control; for our B-stage rhythmic genes (BM and cochlea Tier 1/2 and nominal p < 0.01) the
    predicted ZT12 − ZT0 difference (our cosinor) vs the WT and KO differences (Pearson r, gene-label permutation), and
    the share of genes with a WT time effect (|log2FC| ≥ 0.5, p < 0.05) that lose it in KO.
(E) GSE87467 Cry1/2 double-KO vs WT bone marrow (no time; GPL1261, log2 quantile-normalised): ImmGen stage signatures
    and our BM B-stage marker signatures (222 construction), Welch t (exploratory).
Outputs (results/r5_bcell/): b_external_signature_rhythm.csv, b_external_order_tests.csv, b_external_igapc.csv,
b_external_bmal1ko.csv, b_external_cryko.csv
"""
import gzip
import io
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r5_bcell"
G = EXTERNAL / "geo"
rng = np.random.default_rng(56)
CAN = phase.CANONICAL_CLOCK
STAGES = ["proB.FrBC.BM", "preB.FrC.BM", "preB.FrD.BM", "B.FrE.BM", "B.FrF.BM"]
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
sg = pd.read_csv(RESULTS / "r5_bcell/immgen_b_signatures.csv")
sg = sg[sg["reference"] == "ImmGen microarray (GSE15907)"]
SIGS = {f"ImmGen {s}": sg.loc[sg["signature"] == f"stage:{s}", "gene"].tolist() for s in STAGES}
ax = sg[sg["signature"] == "axis:FrF_vs_FrE"]
AXIS = (ax.loc[ax["direction"].str.startswith("up"), "gene"].tolist(), ax.loc[ax["direction"].str.startswith("down"), "gene"].tolist())
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_BM = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "bm") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
hom = pd.read_csv(EXTERNAL / "resources/mgi_hom_mouse_human.rpt", sep="\t", dtype=str).rename(
    columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
m2h = {m[0]: h[0] for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1}


def score(L, genes):
    g = [x for x in genes if x in L.index]
    Z = L.loc[g].sub(L.loc[g].mean(1), axis=0).div(L.loc[g].std(1).replace(0, np.nan), axis=0)
    return Z.mean(0), len(g)


def fit(y, t):
    f = cosinor.fit(np.asarray(y, float)[:, None], np.asarray(t, float))
    return float(f["peak"][0]), float(f["p"][0]), float(f["amp"][0])


def order_T(pk):
    return float(sum((pk[k + 1] - pk[k]) % 24 for k in range(len(pk) - 1)))


rows, order_rows = [], []
# ---------------- (A) GSE198770 BM
ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__expr.csv", index_col=0)
me = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__meta.csv")
L = np.log2(ex.clip(lower=0) + 1)
for cond, m in me.groupby("condition"):
    o_ext = float(off[(off["dataset"] == "GSE198770_bone_marrow") & (off["condition"] == cond)]["offset_canonical_h"].iloc[0])
    Lc = L[m["sample"]]
    t = m["time_h"].to_numpy(float)
    pks = {}
    for name, genes in list(SIGS.items()) + [("ImmGen axis FrF_vs_FrE", None)]:
        if genes is None:
            u, nu = score(Lc, AXIS[0]); d, nd = score(Lc, AXIS[1])
            y, n = (u - d).to_numpy(), nu + nd
        else:
            y, n = score(Lc, genes)
            y = y.to_numpy()
        pk, p, amp = fit(y, t)
        ours = float(phase.wrap24(pk - o_ext + O_BM))
        pks[name] = ours
        rows.append({"dataset": "GSE198770 BM bulk", "condition": cond, "signature": name, "n_genes": n, "peak_ext_h": pk, "p": p, "amplitude": amp,
                     "peak_our_bm_zt_equiv": ours, "peak_our_bm_clock_equiv": float(phase.zt_to_clock(ours))})
    seq = [pks[f"ImmGen {s}"] for s in STAGES]
    T = order_T(seq)
    perms = [order_T([seq[i] for i in p_]) for p_ in itertools.permutations(range(5))]
    order_rows.append({"dataset": "GSE198770 BM bulk", "condition": cond, "order": " → ".join(STAGES), "T_forward_lag_h": T,
                       "p_exact": float(np.mean([x <= T + 1e-9 for x in perms])),
                       "peaks_our_zt_equiv": "; ".join(f"{s} {v:.1f}" for s, v in zip(STAGES, seq))})
# depletion interaction
mc = me.set_index("sample")
for name, genes in list(SIGS.items()) + [("ImmGen axis FrF_vs_FrE", None)]:
    if genes is None:
        y = (score(L, AXIS[0])[0] - score(L, AXIS[1])[0])
    else:
        y = score(L, genes)[0]
    d = pd.DataFrame({"y": y, "t": mc.loc[y.index, "time_h"], "dep": (mc.loc[y.index, "condition"] != "control").astype(float)})
    X1 = np.column_stack([np.ones(len(d)), np.cos(phase.W * d["t"]), np.sin(phase.W * d["t"]), d["dep"],
                          d["dep"] * np.cos(phase.W * d["t"]), d["dep"] * np.sin(phase.W * d["t"])])
    X0 = X1[:, :4]
    r1 = np.linalg.lstsq(X1, d["y"], rcond=None)[0]
    r0 = np.linalg.lstsq(X0, d["y"], rcond=None)[0]
    rss1, rss0 = ((d["y"] - X1 @ r1) ** 2).sum(), ((d["y"] - X0 @ r0) ** 2).sum()
    F = ((rss0 - rss1) / 2) / (rss1 / (len(d) - 6))
    rows.append({"dataset": "GSE198770 BM bulk", "condition": "depletion × rhythm interaction", "signature": name, "n_genes": np.nan,
                 "peak_ext_h": np.nan, "p": float(stats.f.sf(F, 2, len(d) - 6)), "amplitude": np.nan})
# ---------------- (B) baboon
bo = pd.read_csv(RESULTS / "r4_crossspecies/baboon_clock_offsets.csv").set_index("baboon_tissue")["offset_used_h"]
fp = G / "GSE98965/GSE98965_baboon_tissue_expression_FPKM.csv.gz"
raw = pd.read_csv(fp).dropna(subset=["Symbol"]).drop_duplicates("Symbol").set_index("Symbol").drop(columns="EnsemblID")
for code, lab in [("SPL", "baboon spleen"), ("BOM", "baboon bone marrow (imputed offset)")]:
    cc = [c for c in raw.columns if re.match(rf"^{code}[._]ZT\d+", c)]
    t = np.array([float(re.search(r"ZT(\d+)", c).group(1)) for c in cc])
    Lb = np.log2(raw[cc] + 1)
    Lb = Lb[raw[cc].mean(1) >= 1]
    pks = {}
    for name, genes in list(SIGS.items()) + [("ImmGen axis FrF_vs_FrE", None)]:
        if genes is None:
            u, nu = score(Lb, [m2h.get(g) for g in AXIS[0] if m2h.get(g)]); d_, nd = score(Lb, [m2h.get(g) for g in AXIS[1] if m2h.get(g)])
            y, n = (u - d_).to_numpy(), nu + nd
        else:
            y, n = score(Lb, [m2h.get(g) for g in genes if m2h.get(g)])
            y = y.to_numpy()
        pk, p, amp = fit(y, t)
        ours = float(phase.wrap24(pk - bo[code] + O_BM))
        pks[name] = ours
        rows.append({"dataset": lab, "condition": "", "signature": name, "n_genes": n, "peak_ext_h": pk, "p": p, "amplitude": amp,
                     "peak_our_bm_zt_equiv": ours, "peak_our_bm_clock_equiv": float(phase.zt_to_clock(ours))})
    seq = [pks[f"ImmGen {s}"] for s in STAGES]
    T = order_T(seq)
    perms = [order_T([seq[i] for i in p_]) for p_ in itertools.permutations(range(5))]
    order_rows.append({"dataset": lab, "condition": "", "order": " → ".join(STAGES), "T_forward_lag_h": T,
                       "p_exact": float(np.mean([x <= T + 1e-9 for x in perms])), "peaks_our_zt_equiv": "; ".join(f"{s} {v:.1f}" for s, v in zip(STAGES, seq))})
sr = pd.DataFrame(rows)
write_csv(sr, MOD, "b_external_signature_rhythm.csv", "ImmGen B-stage signatures and the FrF-vs-FrE axis in GSE198770 BM bulk and baboon spleen / BM: peaks in our BM ZT equivalent; depletion interaction",
          required=["dataset", "condition", "signature", "p"])
write_csv(pd.DataFrame(order_rows), MOD, "b_external_order_tests.csv", "External developmental-order test (ImmGen order FrBC → FrC → FrD → FrE → FrF): forward-lag T, exact p",
          required=["dataset", "T_forward_lag_h", "p_exact"])

# ---------------- (C) GSE175635 IgA PC time series
pc = pd.read_excel(G / "GSE175635/GSE175635_IgA_PC_circadian_readcount.xlsx")
emap = pd.read_csv(EXTERNAL / "resources/ensembl_gene_map.csv", usecols=["gene_id", "symbol"]).dropna().drop_duplicates("gene_id").set_index("gene_id")["symbol"]
pc["Gene"] = pc["Gene"].map(emap).fillna(pc["Gene"])          # Ensembl IDs → symbols
pc = pc.dropna(subset=["Gene"]).drop_duplicates("Gene").set_index("Gene")
pc = pc[[c for c in pc.columns if c.startswith("PC_ZT")]].apply(pd.to_numeric, errors="coerce").fillna(0)
cpm = pc / pc.sum(0) * 1e6
Lp = np.log2(cpm[cpm.mean(1) >= 2] + 1)
tp = np.array([float(re.search(r"ZT(\d+)", c).group(1)) for c in Lp.columns])
fp_ = cosinor.fit(Lp.T.to_numpy(), tp)
pcfit = pd.DataFrame({"gene": Lp.index, "peak": fp_["peak"], "p": fp_["p"], "amp": fp_["amp"], "a": fp_["a"], "b": fp_["b"]}).set_index("gene")
cg = pcfit[pcfit.index.isin(CAN) & (pcfit["p"] < 0.05)]
dev = phase.circ_diff(cg["peak"].to_numpy(), cg.index.map(CAN).to_numpy())
o_pc, R_pc = phase.circ_mean(dev % 24) if len(cg) else (np.nan, np.nan)
ig = [{"analysis": "IgA PC clock offset vs canonical phases", "n_genes": len(cg), "value": float(phase.circ_diff(o_pc, 0)) if len(cg) else np.nan,
       "resultant": R_pc, "detail": "; ".join(f"{g} ZT{r.peak:.1f}" for g, r in cg.iterrows())}]
cat = pd.read_csv(RESULTS / "80_atlas/rhythmic_catalogue.csv")
cat = cat[cat["cell_type"].isin(BST)]
rh_pc = pcfit[pcfit["p"] < 0.05]
pool = phase.wrap24(rh_pc["peak"].to_numpy() - phase.circ_diff(o_pc, 0)) if len(cg) else rh_pc["peak"].to_numpy()
for (t_, ct), g in cat.groupby(["tissue", "cell_type"]):
    g = g[g["gene"].isin(rh_pc.index)]
    if len(g) < 10:
        continue
    o_t = O_BM if t_ == "bm" else float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
    a_ = phase.wrap24(g["peak_zt"].to_numpy() - o_t)
    b_ = phase.wrap24(rh_pc.loc[g["gene"], "peak"].to_numpy() - (phase.circ_diff(o_pc, 0) if len(cg) else 0))
    obs = float((np.abs(phase.circ_diff(a_, b_)) <= 3).mean())
    null = [float((np.abs(phase.circ_diff(a_, rng.choice(pool, len(a_), replace=False))) <= 3).mean()) for _ in range(1000)]
    ig.append({"analysis": f"phase concordance with our {t_} {ct} Tier-1/2 genes (aligned)", "n_genes": len(g), "value": obs,
               "resultant": float(np.mean(null)), "detail": f"null mean in 'resultant' column; p {(1 + sum(x >= obs for x in null)) / 1001:.3f}"})
write_csv(pd.DataFrame(ig), MOD, "b_external_igapc.csv", "GSE175635 IgA+ plasma cells (ZT0/6/12/18): clock offset vs canonical and phase concordance with our B-stage rhythms",
          required=["analysis", "n_genes", "value"])

# ---------------- (D) GSE175609 B-cell Bmal1 KO
k = pd.read_csv(G / "GSE175609/GSE175609_readcount_genename.xls.gz", sep="\t", compression="gzip")
k = k.dropna(subset=["gene_name"]).drop_duplicates("gene_name").set_index("gene_name")
sc_ = [c for c in k.columns if re.match(r"^(WT|KO)_ZT(0|12)_\d+$", c)]
K = k[sc_].apply(pd.to_numeric, errors="coerce").fillna(0)
cpm = K / K.sum(0) * 1e6
Lk = np.log2(cpm[cpm.mean(1) >= 2] + 1)
res = {}
for gt in ["WT", "KO"]:
    a = Lk[[c for c in Lk.columns if c.startswith(f"{gt}_ZT12")]]
    b = Lk[[c for c in Lk.columns if c.startswith(f"{gt}_ZT0_")]]
    tt = stats.ttest_ind(a, b, axis=1)
    res[gt] = pd.DataFrame({"lfc": a.mean(1) - b.mean(1), "p": tt.pvalue})
bk = res["WT"].add_prefix("WT_").join(res["KO"].add_prefix("KO_"))
rows_k = []
for g in [x for x in CAN if x in bk.index]:
    rows_k.append({"analysis": "clock gene ZT12 − ZT0 (positive control)", "gene_or_set": g, "n_genes": 1, "WT_value": bk.loc[g, "WT_lfc"],
                   "KO_value": bk.loc[g, "KO_lfc"], "WT_p": bk.loc[g, "WT_p"], "KO_p": bk.loc[g, "KO_p"]})
for t_ in ["bm", "cochlea"]:
    a33 = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t_}.csv", usecols=["group", "gene", "a", "b", "p", "tier", "status"])
    a33 = a33[a33["group"].isin(BST) & (a33["status"] == "primary") & ((a33["p"] < 0.01) | a33["tier"].isin(["tier1", "tier2"]))]
    o_t = O_BM if t_ == "bm" else float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
    for ct, g in list(a33.groupby("group")) + [("all B stages", a33)]:
        g = g.sort_values("p").drop_duplicates("gene")
        g = g[g["gene"].isin(bk.index)]
        if len(g) < 20:
            continue
        t1, t2 = 0 + o_t, 12 + o_t       # external ZT0 / ZT12 mapped onto our axis through our tissue offset (external assumed canonical)
        pred = (g["a"] * (np.cos(phase.W * t2) - np.cos(phase.W * t1)) + g["b"] * (np.sin(phase.W * t2) - np.sin(phase.W * t1))).to_numpy()
        for gt in ["WT", "KO"]:
            ov = bk.loc[g["gene"], f"{gt}_lfc"].to_numpy()
            r = float(np.corrcoef(pred, ov)[0, 1])
            null = np.array([np.corrcoef(pred, rng.permutation(ov))[0, 1] for _ in range(2000)])
            rows_k.append({"analysis": f"our {t_} rhythms predict IgA-PC ZT12 − ZT0 ({gt})", "gene_or_set": ct, "n_genes": len(g),
                           f"{gt}_value": r, f"{gt}_p": float((1 + (null >= r).sum()) / 2001)})
        wt_eff = (bk.loc[g["gene"], "WT_p"] < 0.05) & (bk.loc[g["gene"], "WT_lfc"].abs() >= 0.5)
        lost = wt_eff & ((bk.loc[g["gene"], "KO_p"] >= 0.05) | (bk.loc[g["gene"], "KO_lfc"].abs() < 0.25))
        rows_k.append({"analysis": f"our {t_} rhythmic genes with a WT time effect lost in B-cell Bmal1 KO", "gene_or_set": ct, "n_genes": int(wt_eff.sum()),
                       "WT_value": float(lost.sum() / max(wt_eff.sum(), 1))})
allw = (bk["WT_p"] < 0.05) & (bk["WT_lfc"].abs() >= 0.5)
rows_k.append({"analysis": "background: all expressed genes with a WT time effect lost in KO", "gene_or_set": "all", "n_genes": int(allw.sum()),
               "WT_value": float((allw & ((bk["KO_p"] >= 0.05) | (bk["KO_lfc"].abs() < 0.25))).sum() / max(allw.sum(), 1))})
write_csv(pd.DataFrame(rows_k), MOD, "b_external_bmal1ko.csv", "GSE175609 IgA+ PCs ZT12 − ZT0 in WT vs B-cell Bmal1 KO: clock controls, prediction from our B rhythms, loss of time effects",
          required=["analysis", "gene_or_set", "n_genes"])

# ---------------- (E) GSE87467 Cry1/2 DKO BM
with gzip.open(G / "GSE87467/GSE87467_series_matrix.txt.gz", "rt") as fh:
    lines = fh.readlines()
titles = [x.strip('"') for x in [l for l in lines if l.startswith("!Sample_title")][0].rstrip("\n").split("\t")[1:]]
b0 = [i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin")][0] + 1
b1 = [i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end")][0]
M = pd.read_csv(io.StringIO("".join(lines[b0:b1])), sep="\t", index_col=0)
M.columns = titles
with gzip.open(G / "GSE87467/GPL1261.annot.gz", "rt", errors="replace") as fh:
    al = fh.readlines()
s0 = [i for i, l in enumerate(al) if l.startswith("!platform_table_begin")][0] + 1
ann = pd.read_csv(io.StringIO("".join(al[s0:])), sep="\t", dtype=str, on_bad_lines="skip").set_index("ID")["Gene symbol"].dropna()
ann = ann[~ann.str.contains("///")]
M = np.log2(M.clip(lower=1))
M = pd.DataFrame(np.sort(M.to_numpy(), 0).mean(1)[np.argsort(np.argsort(M.to_numpy(), 0), 0)], index=M.index, columns=M.columns)  # quantile normalisation
M = M.loc[M.index.intersection(ann.index)]
M["sym"] = ann.reindex(M.index).to_numpy()
M = M.groupby("sym").mean()
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4)]
ours = {f"our {c} markers": g.sort_values("scores", ascending=False)["names"].head(25).tolist() for c, g in mk.groupby("cell_type") if c in BST}
rows_e = []
dko = [c for c in M.columns if c.startswith("DKO")]
wt = [c for c in M.columns if c.startswith("WT")]
for name, genes in list(SIGS.items()) + list(ours.items()):
    s, n = score(M, genes)
    tt = stats.ttest_ind(s[dko], s[wt], equal_var=False)
    rows_e.append({"signature": name, "n_genes": n, "mean_DKO": float(s[dko].mean()), "mean_WT": float(s[wt].mean()),
                   "diff_DKO_minus_WT": float(s[dko].mean() - s[wt].mean()), "p_welch": float(tt.pvalue)})
write_csv(pd.DataFrame(rows_e), MOD, "b_external_cryko.csv", "GSE87467 Cry1/2 DKO vs WT bone marrow: B-stage signature scores (exploratory; no time course)",
          required=["signature", "diff_DKO_minus_WT", "p_welch"])
log("\n" + sr.round(3).to_string())
log("\n" + pd.DataFrame(order_rows).round(3).to_string())
log("\n" + pd.DataFrame(ig).round(3).to_string())
log("\n" + pd.DataFrame(rows_k).round(3).to_string())
log("\n" + pd.DataFrame(rows_e).round(3).to_string())
