"""Ma 2025 (CRA019820) — myeloid primary hypothesis M9 (D46) and the exploratory B-lineage add-on (D36).

Design: cochleae 3 d after 2 h of noise at ZT2–4 (day_noise, D1–D4) or ZT14–16 (night_noise, N1–N3), controls C1–C3.
Our time-of-day expectation is taken at the exposure midpoints, ZT15 − ZT3 (plan: M9).
M9 primary (pre-specified): across the 11 curated readiness programmes of M4 (myeloid_function + Hallmark glycolysis /
  OXPHOS), Spearman between (i) the night_noise − day_noise difference of the programme score in Ma's cochlear
  neutrophil + monocyte pseudobulk (transferred labels: four neutrophil labels + classical / non-classical monocytes;
  per sample, log2 CPM, genes >= 1 CPM, gene z across samples, score = mean z of >= 3 member genes, as 645) and (ii) our
  predicted ZT15 − ZT3 difference of the same programme in the cochlear pooled unit (645 m_function_rhythm, primary,
  uncentred: A[cos ω(15 − φ) − cos ω(3 − φ)]). Programme-label permutation p (10,000; one-sided, direction positive).
M9 agreeing methods (pre-specified in the plan; rules fixed here before the data were opened):
  (1) myeloid composition: Spearman across myeloid labels (GMP, neutrophils, monocytes, macrophages, mast cells) between
      Ma's night / day log2 share ratio (310) and our predicted logit-share difference ZT15 − ZT3 (620 share_of_tissue,
      cochlea) > 0 with permutation p < 0.05;
  (2) NLRP3 / IL-1 axis (312 genes) in macrophages + classical monocytes: the mean night − day logFC (311) has the sign
      of our predicted mean ZT15 − ZT3 difference of the same genes in the same cochlear labels (33 cosinor fits).
Sensitivity: confident cells only (separate pseudobulk from the mapped object); controls vs each noise group reported.
B add-on (exploratory, D36): B-lineage label shares and the curated B programmes (b_cell sets) in Ma's pooled B-lineage
  pseudobulk, night − day vs our predicted ZT15 − ZT3 (r5 b_composition_summary share_of_tissue; b_programme_rhythm
  primary uncentred, pooled B lineage or the mean over stages); Spearman with permutation p; plus each B label's
  night_noise_vs_day_noise DE count (311).
Outputs (results/r3_<key>/): m9_programmes.csv, m9_tests.csv, b_addon.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

KEY = sys.argv[1] if len(sys.argv) > 1 else "ma2025"
MOD = f"r3_{KEY}"
W = phase.W
rng = np.random.default_rng(316)
T1, T2 = 3.0, 15.0
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO2 = ["Monocyte (Classical)", "Monocyte (Non-classical)"]
MYE = [GMP] + NEU + MONO2 + ["Macrophage", "Mast Cell"]
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
hm = {}
for line in open(EXTERNAL / "resources/msigdb_mh_hallmark.gmt"):
    x = line.rstrip("\n").split("\t")
    hm[x[0]] = x[2:]
PROG = {**{f"myeloid:{k}": v for k, v in SETS["myeloid_function"].items()},
        "hallmark:GLYCOLYSIS": hm["HALLMARK_GLYCOLYSIS"], "hallmark:OXIDATIVE_PHOSPHORYLATION": hm["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]}
BPROG = {f"b_cell:{k}": v for k, v in SETS["b_cell"].items()}


def pred(amp, pk):
    return amp * (np.cos(W * (T2 - pk)) - np.cos(W * (T1 - pk)))


def spear_perm(x, y, n=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
    return float(r), float((1 + (null >= r).sum()) / (n + 1))


def scores(pbu, sets):
    cpm = pbu / pbu.sum(0) * 1e6
    L = np.log2(cpm + 1)[cpm.mean(1) >= 1]
    Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
    out = {}
    for nm, genes in sets.items():
        g = [x for x in genes if x in Z.index]
        if len(g) >= 3:
            out[nm] = (Z.loc[g].mean(0), len(g))
    return out


def night_minus_day(s, cond):
    n, d = s[cond == "night_noise"], s[cond == "day_noise"]
    t = stats.ttest_ind(n, d, equal_var=False)
    return float(n.mean() - d.mean()), float(t.pvalue)


# ---------------- pseudobulk per sample for a label set (all cells; confident cells as sensitivity)
q = ad.read_h5ad(CACHE / "r3_query" / f"{KEY}_mapped.h5ad")
q.obs["label"] = q.obs["label"].astype(str)
samples = sorted(q.obs["sample"].astype(str).unique())
cond = q.obs.groupby("sample", observed=True)["condition"].first().astype(str).reindex(samples).to_numpy()


def unit_pb(labels, confident_only=False):
    m = q.obs["label"].isin(labels).to_numpy()
    if confident_only:
        m &= q.obs["confident"].astype(bool).to_numpy()
    cols = {}
    for s in samples:
        mm = m & (q.obs["sample"].astype(str) == s).to_numpy()
        cols[s] = np.asarray(q.X[mm].sum(0)).ravel()
    n = {s: int((m & (q.obs["sample"].astype(str) == s).to_numpy()).sum()) for s in samples}
    return pd.DataFrame(cols, index=q.var_names), n


fnr = pd.read_csv(RESULTS / "r6_myeloid/m_function_rhythm.csv")
fr = fnr[(fnr["tissue"] == "cochlea") & (fnr["version"] == "primary") & (fnr["kind"] == "uncentred") &
         (fnr["unit"] == "neutrophils + monocytes (pooled)")].set_index("programme")
prog_rows, tests = [], []
for subset in ["all cells", "confident cells"]:
    pbu, ncell = unit_pb(NEU + MONO2, subset == "confident cells")
    sc_ = scores(pbu, PROG)
    for nm, (s, ng) in sc_.items():
        d, p = night_minus_day(s.to_numpy(float), cond)
        rr = fr.loc[nm] if nm in fr.index else None
        prog_rows.append({"subset": subset, "programme": nm, "n_genes": ng, "ma_night_minus_day": d, "ma_welch_p": p,
                          "ma_day_noise_minus_control": float(s[cond == "day_noise"].mean() - s[cond == "control"].mean()),
                          "ma_night_noise_minus_control": float(s[cond == "night_noise"].mean() - s[cond == "control"].mean()),
                          "our_peak_zt": rr["peak_zt"] if rr is not None else np.nan, "our_amplitude": rr["amplitude"] if rr is not None else np.nan,
                          "our_p_perm": rr["p_perm"] if rr is not None else np.nan,
                          "our_predicted_zt15_minus_zt3": pred(rr["amplitude"], rr["peak_zt"]) if rr is not None else np.nan,
                          "cells_per_sample": "; ".join(f"{k} {v}" for k, v in ncell.items())})
pr = pd.DataFrame(prog_rows)
for subset, g in pr.groupby("subset"):
    g = g.dropna(subset=["our_predicted_zt15_minus_zt3"])
    r, p = spear_perm(g["our_predicted_zt15_minus_zt3"], g["ma_night_minus_day"])
    tests.append({"hypothesis": "M9", "component": "primary" if subset == "all cells" else "primary (sensitivity: confident cells)",
                  "statistic": "Spearman across readiness programmes: Ma night − day (neutrophil + monocyte pseudobulk) vs our predicted ZT15 − ZT3",
                  "n_items": len(g), "r": r, "p_perm_one_sided": p, "direction_ok": bool(r > 0), "agrees": np.nan})

# (1) composition agreeing method
cc = pd.read_csv(RESULTS / MOD / "composition_contrasts.csv")
cc = cc[(cc["contrast"].astype(str).str.contains("night_noise") & cc["contrast"].astype(str).str.contains("day_noise"))]
if "subset" in cc.columns:
    cc = cc[cc["subset"] == "all cells"]
cs = pd.read_csv(RESULTS / "r6_myeloid/m_composition_summary.csv")
cs = cs[(cs["tissue"] == "cochlea") & (cs["frame"] == "share_of_tissue")].set_index("feature")
x = cc[cc["label"].isin(MYE) & cc["label"].isin(cs.index)].copy()
x["our_predicted"] = [pred(cs.loc[l, "amplitude"], cs.loc[l, "peak_zt"]) for l in x["label"]]
r, p = spear_perm(x["our_predicted"], x["log2_ratio"]) if len(x) >= 4 else (np.nan, np.nan)
tests.append({"hypothesis": "M9", "component": "agreeing: myeloid composition (310) vs our share rhythms", "statistic":
              "Spearman across myeloid labels: Ma night / day log2 share ratio vs our predicted logit-share ZT15 − ZT3", "n_items": len(x), "r": r,
              "p_perm_one_sided": p, "direction_ok": bool(r > 0) if np.isfinite(r) else False, "agrees": bool(np.isfinite(r) and r > 0 and p < 0.05),
              "detail": "; ".join(f"{a}: {b:+.2f} vs {c:+.2f}" for a, b, c in zip(x["label"], x["log2_ratio"], x["our_predicted"]))})

# (2) NLRP3 / IL-1 axis agreeing method
AX = ["Nlrp3", "Il1b", "Il18", "Casp1", "Pycard", "Gsdmd", "Il1rn", "Il1r2"]
de = pd.read_csv(RESULTS / MOD / "pseudobulk_de.csv")
cn = [c for c in de.columns if c.lower() in ("contrast",)][0]
gcol = [c for c in de.columns if c.lower() in ("gene", "genes", "symbol")][0]
lcol = [c for c in de.columns if c.lower() in ("label", "cell_type")][0]
fc = [c for c in de.columns if c.lower() in ("logfc", "log2fc")][0]
dd = de[de[cn].astype(str).str.contains("night_noise") & de[cn].astype(str).str.contains("day_noise") &
        de[lcol].isin(["Macrophage", "Monocyte (Classical)"]) & de[gcol].isin(AX)]
rt = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_genes_all_cell_type_cochlea.csv", usecols=["group", "gene", "a", "b", "p"])
rt = rt[rt["group"].isin(["Macrophage", "Monocyte (Classical)"]) & rt["gene"].isin(AX)]
rt["pred"] = rt["a"] * (np.cos(W * T2) - np.cos(W * T1)) + rt["b"] * (np.sin(W * T2) - np.sin(W * T1))
mo, mp = float(dd[fc].mean()) if len(dd) else np.nan, float(rt["pred"].mean()) if len(rt) else np.nan
tests.append({"hypothesis": "M9", "component": "agreeing: NLRP3 / IL-1 axis (312 genes; macrophages + classical monocytes)", "statistic":
              "sign of the mean night − day logFC (311) vs sign of our mean predicted ZT15 − ZT3 (33 fits)", "n_items": len(dd), "r": np.nan,
              "p_perm_one_sided": np.nan, "direction_ok": bool(np.sign(mo) == np.sign(mp)) if np.isfinite(mo) and np.isfinite(mp) else False,
              "agrees": bool(np.isfinite(mo) and np.isfinite(mp) and np.sign(mo) == np.sign(mp)),
              "detail": f"Ma mean logFC {mo:+.3f} ({len(dd)} gene × label rows); our mean predicted {mp:+.3f} ({len(rt)} rows)"})

# ---------------- B add-on (exploratory)
ba = []
bc = pd.read_csv(RESULTS / "r5_bcell/b_composition_summary.csv")
bc = bc[(bc["tissue"] == "cochlea") & (bc["frame"] == "share_of_tissue")].set_index("feature")
x = cc[cc["label"].isin(BST) & cc["label"].isin(bc.index)].copy()
x["our_predicted"] = [pred(bc.loc[l, "amplitude"], bc.loc[l, "peak_zt"]) for l in x["label"]]
for r_ in x.itertuples():
    ba.append({"analysis": "B-lineage share: Ma night / day log2 ratio vs our predicted ZT15 − ZT3", "item": r_.label, "ma_value": r_.log2_ratio,
               "our_predicted": r_.our_predicted})
if len(x) >= 4:
    r, p = spear_perm(x["our_predicted"], x["log2_ratio"])
    ba.append({"analysis": "B-lineage share (summary)", "item": "Spearman across B labels", "r": r, "p_perm_one_sided": p, "n_items": len(x)})
bp = pd.read_csv(RESULTS / "r5_bcell/b_programme_rhythm.csv")
bp = bp[(bp["tissue"] == "cochlea") & (bp["version"] == "primary") & (bp["kind"] == "uncentred")]
pool = [s for s in bp["stage"].unique() if "pool" in str(s).lower()]
bpp = (bp[bp["stage"] == pool[0]] if pool else bp.groupby("gene_set", as_index=False)[["amplitude", "peak_zt"]].agg(
    {"amplitude": "mean", "peak_zt": lambda v: phase.circ_mean(v.to_numpy())[0]})).set_index("gene_set")
pbB, nB = unit_pb(BST)
scB = scores(pbB, BPROG)
rowsB = []
for nm, (s, ng) in scB.items():
    if nm not in bpp.index:
        continue
    d, p = night_minus_day(s.to_numpy(float), cond)
    rowsB.append((nm, d, p, pred(bpp.loc[nm, "amplitude"], bpp.loc[nm, "peak_zt"]), ng))
    ba.append({"analysis": "B programme: Ma night − day (pooled B-lineage pseudobulk) vs our predicted ZT15 − ZT3", "item": nm, "ma_value": d,
               "ma_welch_p": p, "our_predicted": rowsB[-1][3], "n_genes": ng,
               "our_unit": pool[0] if pool else "mean over B stages"})
if len(rowsB) >= 4:
    r, p = spear_perm([v[3] for v in rowsB], [v[1] for v in rowsB])
    ba.append({"analysis": "B programme (summary)", "item": "Spearman across B programmes", "r": r, "p_perm_one_sided": p, "n_items": len(rowsB)})
dB = de[de[cn].astype(str).str.contains("night_noise") & de[cn].astype(str).str.contains("day_noise") & de[lcol].isin(BST)]
pcol = [c for c in de.columns if c in ("P.Value", "p", "pvalue", "PValue")]
qcol = [c for c in de.columns if c in ("adj.P.Val", "q_BH", "FDR", "padj")]
for lab, g in dB.groupby(lcol):
    ba.append({"analysis": "B label night_noise vs day_noise DE (311)", "item": lab, "n_genes": len(g),
               "n_p_lt_0.01": int((g[pcol[0]] < 0.01).sum()) if pcol else np.nan, "n_q_lt_0.1": int((g[qcol[0]] < 0.1).sum()) if qcol else np.nan})

write_csv(pr, MOD, "m9_programmes.csv", "M9: readiness programmes in Ma 2025 neutrophil + monocyte pseudobulk (night − day, each vs control) next to "
          "our predicted ZT15 − ZT3 differences", required=["subset", "programme", "ma_night_minus_day", "our_predicted_zt15_minus_zt3"])
write_csv(pd.DataFrame(tests), MOD, "m9_tests.csv", "M9 primary statistic, sensitivity and agreeing methods (pre-specified rules)",
          required=["hypothesis", "component", "r", "p_perm_one_sided", "agrees"])
write_csv(pd.DataFrame(ba), MOD, "b_addon.csv", "Exploratory B-lineage add-on (D36): Ma 2025 night − day B shares and programmes vs our predicted "
          "ZT15 − ZT3; B DE counts", required=["analysis", "item"])
pd.set_option("display.width", 250)
log("\n" + pr.round(3).drop(columns=["cells_per_sample"]).to_string(index=False))
log("\n" + pd.DataFrame(tests).round(4).to_string(index=False, max_colwidth=90))
log("\n" + pd.DataFrame(ba).round(3).to_string(index=False, max_colwidth=70))
