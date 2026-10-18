"""Round 4, WP-E (extended) — are the atlas's rhythmic programmes age-sensitive? GSE274279 snRNA cochlea at 3, 12 and
24 months (24M has two libraries, 24M_Cochlea and 24M_Cochlea_BU); labels transferred onto our gold standard (302).

Label transfer of these nuclei is poor (most nuclei go to two labels that fail the marker check; r4_concordance), so the
whole-tissue analysis (all nuclei) is primary and label-level rows carry the marker-check flag.
Caveat built into the design: one library per age (two at 24M) and unknown collection times — differences in collection
time would make rhythmic genes look 'age-variable'. Hence:
(1) ageing_sampling_phase.csv: per library, centred core-clock profile → canonical projection phase and template match
    with our cochlear tissue fits (method of 312); large between-age phase differences flag a time-of-day confound.
(2) ageing_composition.csv: confident transferred-label shares per library; log2 ratio vs 3M.
(3) ageing_rhythmic_variability.csv: per transferred label with ≥ 30 cells in every library, pseudobulk log2 CPM;
    |log2FC| of 24M (mean of the two libraries) vs 3M and 12M vs 3M for rhythmic genes (same-type cochlear Tier 1/2 or
    Tier C; core clock; GSE88954-replicated rhythmic genes) vs expression-matched non-rhythmic genes (Mann–Whitney,
    median ratio). Control: the same comparison between the two 24M libraries (same age) — rhythmic genes that differ
    as much between same-age libraries are time-of-day / technical, not age.
Outputs (results/r4_ageing/)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_ageing"
W = phase.W
q = ad.read_h5ad(CACHE / "r3_query/gse274279_mapped.h5ad")
q.obs["age"] = q.obs["condition"].astype(str)
libs = sorted(q.obs["sample"].unique(), key=lambda s: (int(s.split("M")[0]), s))
log(f"libraries: {libs}")

# ---------------- (1) sampling phase
tot = pd.DataFrame({s: np.asarray(q[q.obs["sample"] == s].X.sum(0)).ravel() for s in libs}, index=q.var_names)
cpm = tot / tot.sum(0) * 1e6
lc = np.log2(cpm + 1)
clock = [g for g in phase.CANONICAL_CLOCK if g in lc.index and cpm.loc[g].mean() > 1]
z = lc.loc[clock].sub(lc.loc[clock].mean(1), axis=0)
z = z.div(z.std(1).replace(0, np.nan), axis=0).fillna(0)
E = np.exp(1j * W * np.array([phase.CANONICAL_CLOCK[g] for g in clock]))
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
O_C = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
tp = pd.read_csv(RESULTS / "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").set_index("gene")
tg = [g for g in clock if g in tp.index]
rows = []
for s in libs:
    v = (z[s].to_numpy() * E).sum()
    prof = z.loc[tg, s].to_numpy()
    best = max(((float(np.corrcoef(prof, (lambda p: p - p.mean())(tp.loc[tg, "a"] * np.cos(W * (t + O_C)) + tp.loc[tg, "b"] * np.sin(W * (t + O_C))))[0, 1]), t)
                for t in np.arange(0, 24, 0.25)), default=(np.nan, np.nan))
    rows.append({"sample": s, "age": s.split("_")[0], "n_clock_genes": len(clock), "phase_canonical_projection_h": float(np.mod(np.angle(v) / W, 24)),
                 "projection_length": float(np.abs(v)), "phase_template_match_h": best[1], "template_match_r": best[0],
                 **{f"centred_log2_{g}": float(lc.loc[g, s] - lc.loc[g].mean()) for g in ["Dbp", "Nr1d1", "Nr1d2", "Per3", "Npas2", "Arntl", "Cry1", "Tef"] if g in lc.index}})
sp_ = pd.DataFrame(rows)
write_csv(sp_, MOD, "ageing_sampling_phase.csv",
          "GSE274279: inferred relative collection phase per library from core-clock genes (projection; template match) and centred clock-gene levels",
          required=["sample", "age", "phase_canonical_projection_h", "projection_length"])
log("\n" + sp_.round(2).to_string())

# ---------------- (2) composition
conf = q.obs[q.obs["confident"].astype(bool)]
sh = conf.groupby(["sample", "label"], observed=True).size().unstack(fill_value=0)
sh = sh.div(sh.sum(1), axis=0)
comp = sh.stack().rename("share").reset_index()
base = sh.loc[[l for l in libs if l.startswith("3M")]].mean(0)
comp["log2_ratio_vs_3M"] = np.log2((comp["share"] + 1e-4) / (comp["label"].map(base) + 1e-4))
comp["age"] = comp["sample"].str.split("_").str[0]
write_csv(comp, MOD, "ageing_composition.csv", "GSE274279: transferred-label shares (confident cells) per library and log2 ratio vs 3 months",
          required=["sample", "age", "label", "share", "log2_ratio_vs_3M"])

# ---------------- (3) rhythmic-gene variability with a same-age control
cat = pd.read_csv(RESULTS / "80_atlas/rhythmic_catalogue.csv")
cat = cat[cat["tissue"] == "cochlea"]
tc = pd.read_csv(RESULTS / "r2_cochlea/tierC_catalogue.csv", usecols=["cell_type", "gene"])
rhy_type = pd.concat([cat[["cell_type", "gene"]], tc]).groupby("cell_type")["gene"].apply(set).to_dict()
ext = pd.read_csv(RESULTS / "71_external/external_rhythm_GSE88954.csv")
ext_rh = set(ext[(ext["condition"] == "sham") & (ext["p"] < 0.05)]["gene"])
CORE = set(phase.CANONICAL_CLOCK) | {"Arntl", "Clock", "Bhlhe40", "Bhlhe41"}
l24 = [l for l in libs if l.startswith("24M")]
l3 = [l for l in libs if l.startswith("3M")]
l12 = [l for l in libs if l.startswith("12M")]
mkc = pd.read_csv(RESULTS / "r3_ingest/gse274279_marker_check.csv").set_index("label")
mpass = ((mkc["marker_z_in_label"] >= 0.5) & (mkc["marker_z_in_label"] > mkc["marker_z_in_others"])).to_dict()
res = []
groups = [("ALL cells (whole tissue; primary)", q.obs)] + [(lab, g) for lab, g in conf.groupby("label", observed=True)]
for lab, g in groups:
    n = g.groupby("sample", observed=True).size().reindex(libs).fillna(0)
    if (n < 30).any():
        continue
    X = q[g.index]
    pb = pd.DataFrame({s: np.asarray(X[X.obs["sample"] == s].X.sum(0)).ravel() for s in libs}, index=q.var_names)
    keep = (pb / pb.sum(0) * 1e6).mean(1) >= 10
    L = np.log2(pb[keep] / pb[keep].sum(0) * 1e6 + 1)
    contrasts = {"24M_vs_3M": L[l24].mean(1) - L[l3].mean(1), "12M_vs_3M": L[l12].mean(1) - L[l3].mean(1)}
    if len(l24) == 2:
        contrasts["24M_lib1_vs_24M_lib2 (same-age control)"] = L[l24[0]] - L[l24[1]]
    dec = pd.qcut(L.mean(1).rank(method="first"), 10, labels=False)
    sets = {"same-type rhythmic (Tier 1/2 or C)": (set().union(*rhy_type.values()) if lab.startswith("ALL") else rhy_type.get(lab, set())),
            "GSE88954-replicated rhythmic": ext_rh, "core clock": CORE}
    for cname, fc in contrasts.items():
        for sname, genes in sets.items():
            m = L.index.isin(genes)
            if m.sum() < 5:
                continue
            bg_idx = []
            rng = np.random.default_rng(3)
            for d in dec[m].unique():
                pool = np.where((~m) & (dec.to_numpy() == d))[0]
                k = int((dec[m] == d).sum())
                if len(pool):
                    bg_idx += list(rng.choice(pool, size=min(len(pool), 5 * k), replace=len(pool) < 5 * k))
            a, b = np.abs(fc[m]).to_numpy(), np.abs(fc.iloc[bg_idx]).to_numpy()
            res.append({"label": lab, "label_marker_check_pass": True if lab.startswith("ALL") else mpass.get(lab, np.nan), "contrast": cname, "gene_set": sname, "n_genes": int(m.sum()), "n_background": len(b),
                        "median_abs_log2fc_set": float(np.median(a)), "median_abs_log2fc_background": float(np.median(b)),
                        "ratio": float(np.median(a) / np.median(b)) if np.median(b) > 0 else np.nan,
                        "p_mannwhitney_greater": float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)})
rv = pd.DataFrame(res)
write_csv(rv, MOD, "ageing_rhythmic_variability.csv",
          "GSE274279: age-related |log2FC| of rhythmic vs expression-matched non-rhythmic genes per transferred label, with a same-age (24M vs 24M) control",
          required=["label", "contrast", "gene_set", "n_genes", "ratio", "p_mannwhitney_greater"])
if len(rv):
    log("\n" + rv.groupby(["contrast", "gene_set"])[["ratio"]].median().round(3).to_string())
