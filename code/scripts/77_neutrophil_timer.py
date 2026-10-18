"""Module 5.4 (E3) — the neutrophil timer: GSE86619 (Adrover 2019) blood neutrophils, WT at ZT5 and ZT13 (n = 3 each)
and neutrophil-specific Bmal1 (Arntl), Cxcr2 and Cxcr4 deletion (Mrp8-Cre) at ZT5, versus our neutrophil rhythms.

Expression: the GEO normalised table (Ensembl IDs → symbols via the GRCm39 GTF map of 70), log2(x + 1), genes with
mean >= 10 in WT. Contrasts (Welch t, BH): WT ZT13 − ZT5 (diurnal change) and each knockout − WT ZT5 at ZT5
(Bmal1/Cxcr2 deletion keep neutrophils 'young', Cxcr4 deletion makes them 'aged-like'; Adrover 2019).
Concordance per tissue × neutrophil-lineage cell type: our cosinor fit predicts the log2 difference between ZT13 and
ZT5, Δ̂ = a(cos 13ω − cos 5ω) + b(sin 13ω − sin 5ω); Spearman correlation of Δ̂ with the observed WT log2FC over
shared genes (all tested genes, and genes with p < 0.05 in both), and sign agreement; permutation p for the
correlation (2000 gene-label shuffles); repeated without the dissociation/immediate-early stress genes (lib.dr.STRESS),
which are themselves time-of-day and handling sensitive. Enrichment: our Tier-1/2 rhythmic genes vs non-rhythmic
tested genes among responsive genes (p < 0.01 and |log2FC| > 0.5; with n = 2–3 per group no gene passes BH q < 0.1
in the knockout contrasts), Fisher exact test.
Outputs (results/77_neutrophil_timer/): neutrophil_timer_gene_table.csv, neutrophil_timer_concordance.csv,
neutrophil_timer_ko_enrichment.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.dr import STRESS  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "77_neutrophil_timer"
W = 2 * np.pi / 24
rng = np.random.default_rng(0)
x = pd.read_excel(EXTERNAL / "geo/GSE86619/GSE86619_matrix_table.xls", engine="xlrd")
x = x[[c for c in x.columns if not str(c).startswith("Unnamed")]].dropna(subset=["ID"])
gmap = pd.read_csv(EXTERNAL / "resources/ensembl_gene_map.csv")
idcol = [c for c in gmap.columns if "id" in c.lower()][0]
symcol = [c for c in gmap.columns if c != idcol and ("name" in c.lower() or "symbol" in c.lower())][0]
x["gene"] = x["ID"].map(dict(zip(gmap[idcol], gmap[symcol])))
x = x.dropna(subset=["gene"]).groupby("gene").mean(numeric_only=True)
grp = {g: [c for c in x.columns if c.startswith(g + "_")] for g in ["ZT5", "ZT13", "BMAL", "CXCR2", "CXCR4"]}
x = x[x[grp["ZT5"] + grp["ZT13"]].mean(1) >= 10]
lx = np.log2(x + 1)
log(f"GSE86619: {len(lx)} genes; samples per group " + ", ".join(f"{k}={len(v)}" for k, v in grp.items()))


def contrast(a, b, name):
    t, p = stats.ttest_ind(lx[grp[a]], lx[grp[b]], axis=1, equal_var=False)
    out = pd.DataFrame({f"{name}_log2fc": lx[grp[a]].mean(1) - lx[grp[b]].mean(1), f"{name}_p": p}, index=lx.index)
    out[f"{name}_q"] = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    return out


tab = pd.concat([contrast("ZT13", "ZT5", "wt_zt13_vs_zt5"), contrast("BMAL", "ZT5", "bmal1ko_vs_wt"),
                 contrast("CXCR2", "ZT5", "cxcr2ko_vs_wt"), contrast("CXCR4", "ZT5", "cxcr4ko_vs_wt")], axis=1)
tab.index.name = "gene"
dc, ds = np.cos(13 * W) - np.cos(5 * W), np.sin(13 * W) - np.sin(5 * W)
conc, enr, pred_cols = [], [], {}
for tissue in ["bm", "cochlea"]:
    t = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                    usecols=["group", "gene", "a", "b", "p", "tier"])
    t = t[t["group"].str.contains("Neutrophil|Granulocyte-Monocyte", regex=True)]
    for ct, g in t.groupby("group"):
        g = g.set_index("gene").join(tab, how="inner")
        if len(g) < 100:
            continue
        g["pred"] = g["a"] * dc + g["b"] * ds
        pred_cols[f"pred_zt13_minus_zt5_{tissue}_{ct}"] = g["pred"]
        obs = g["wt_zt13_vs_zt5_log2fc"]
        r_all = stats.spearmanr(g["pred"], obs).statistic
        null = np.array([stats.spearmanr(g["pred"].to_numpy(), rng.permutation(obs.to_numpy())).statistic
                         for _ in range(2000)])
        both = g[(g["p"] < 0.05) & (g["wt_zt13_vs_zt5_p"] < 0.05)]
        ns = g[~g.index.isin(STRESS)]
        ns_both = both[~both.index.isin(STRESS)]
        conc.append({"tissue": tissue, "cell_type": ct, "n_shared_genes": len(g), "spearman_all": r_all,
                     "p_perm_all": (1 + (np.abs(null) >= abs(r_all)).sum()) / 2001,
                     "n_both_p05": len(both),
                     "spearman_both_p05": stats.spearmanr(both["pred"], both["wt_zt13_vs_zt5_log2fc"]).statistic
                     if len(both) >= 5 else np.nan,
                     "sign_agreement_both_p05": float((np.sign(both["pred"]) == np.sign(both["wt_zt13_vs_zt5_log2fc"])).mean())
                     if len(both) else np.nan,
                     "spearman_all_no_stress": stats.spearmanr(ns["pred"], ns["wt_zt13_vs_zt5_log2fc"]).statistic,
                     "n_both_p05_no_stress": len(ns_both),
                     "spearman_both_p05_no_stress": stats.spearmanr(ns_both["pred"], ns_both["wt_zt13_vs_zt5_log2fc"]).statistic
                     if len(ns_both) >= 5 else np.nan,
                     "n_stress_genes_in_both_p05": int(both.index.isin(STRESS).sum())})
        rhy = g["tier"].isin(["tier1", "tier2"])
        for ko in ["bmal1ko_vs_wt", "cxcr2ko_vs_wt", "cxcr4ko_vs_wt", "wt_zt13_vs_zt5"]:
            resp = (g[f"{ko}_p"] < 0.01) & (g[f"{ko}_log2fc"].abs() > 0.5)
            a_, b_ = int((rhy & resp).sum()), int((rhy & ~resp).sum())
            c_, d_ = int((~rhy & resp).sum()), int((~rhy & ~resp).sum())
            orr, p = stats.fisher_exact([[a_, b_], [c_, d_]], alternative="greater")
            enr.append({"tissue": tissue, "cell_type": ct, "external_contrast": ko, "n_rhythmic": int(rhy.sum()),
                        "n_responsive": int(resp.sum()), "n_rhythmic_and_responsive": a_,
                        "frac_rhythmic_responsive": a_ / max(rhy.sum(), 1),
                        "frac_nonrhythmic_responsive": c_ / max((~rhy).sum(), 1), "odds_ratio": orr, "p_fisher": p})
gene_tab = tab.join(pd.DataFrame(pred_cols), how="left").reset_index()
write_csv(gene_tab, MOD, "neutrophil_timer_gene_table.csv",
          "GSE86619 contrasts (WT ZT13−ZT5; Bmal1/Cxcr2/Cxcr4 KO − WT at ZT5) with our predicted ZT13−ZT5 differences",
          required=["gene", "wt_zt13_vs_zt5_log2fc", "wt_zt13_vs_zt5_p"])
write_csv(pd.DataFrame(conc), MOD, "neutrophil_timer_concordance.csv",
          "Concordance of our neutrophil-lineage rhythms with the blood neutrophil ZT13−ZT5 change (GSE86619)",
          required=["tissue", "cell_type", "spearman_all", "p_perm_all"])
e = pd.DataFrame(enr)
e["q_BH"] = e.groupby("external_contrast")["p_fisher"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(e, MOD, "neutrophil_timer_ko_enrichment.csv",
          "Rhythmic neutrophil genes among Bmal1/Cxcr2/Cxcr4-knockout-responsive and diurnal blood-neutrophil genes",
          required=["tissue", "cell_type", "external_contrast", "odds_ratio", "p_fisher", "q_BH"])
log("done")
