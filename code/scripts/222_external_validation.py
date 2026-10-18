"""Round 2, WP3 — systematic external validation across cell types.

(A) Gene-rhythm phase concordance: every cell type of our atlas (BM, cochlea) × every external unit (dataset ×
    tissue × condition: GSE88954 cochlea sham/ADX; GSE198770 BM, gut, liver, lung, skin; CircaDB 12 organs;
    peritoneal macrophages GSE25585). Genes rhythmic in both (ours: Tier 1/2, cochlea also Tier C from 211;
    external: p < 0.05) are compared on clock-aligned internal phases (peak − unit offset, 230): circular correlation,
    median |Δ|, fraction within 3 h, and a gene-label permutation null (1000 draws of external rhythmic genes) for the
    fraction within 3 h; computed for all such genes and for non-clock genes only (clock genes agree by construction).
(B) Abundance proxies in external bulk data: immune signature scores (top 25 specific markers per cell type from
    00_audit/label_markers.csv: log2FC >= 2.5, adj. p < 1e-10, detected in >= 40 % of the type and <= 10 % of other
    cells; types with < 10 such markers are filled with markers detected in <= 25 % of other cells — 'relaxed'), and a
    neutrophil-lineage signature (markers of the neutrophil states with log2FC >= 4 in >= 90 % of their cells, minus
    strict markers of other types) split into an immature part (Band / Mature states: granule-protein genes) and a
    mature part (Aged / Activated states), each excluding the other part's markers (the states share markers, so
    state-level bulk signatures are not defined) = mean z-scored log2 expression per sample; cosinor per tissue ×
    condition (internal phases) and depletion shifts (expected: the mature-neutrophil signature falls after anti-Ly6G /
    DT; the immature signature may rise with the compensatory left shift / emergency granulopoiesis).
    BayesPrism fractions (75b, our atlas as reference) are added for BM (GSE198770) and cochlea (GSE88954).
(C) Literature: resources/literature_phases.csv (verified from abstracts) vs our abundance peaks (r2_figure, ZT) or,
    for gene-level statements (measure 'gene:<symbol>'), our gene peak in the named cell types (33_rhythm_tiers);
    series that are not rhythmic in our data (verdict 'not rhythmic' / gene p >= 0.05) are 'not testable'.
(D) Matrix and dossier: external_validation_matrix.csv (cell type × source: supports / contradicts / not testable)
    and celltype_validation_dossier.csv (per cell type: our evidence, external, literature, conclusion, caveats).
Outputs (results/r2_external/): gene_phase_concordance.csv, signature_scores.csv, signature_rhythm.csv,
signature_depletion_effect.csv, literature_comparison.csv, external_validation_matrix.csv,
celltype_validation_dossier.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r2_external"
SCRIPTS = Path(__file__).resolve().parent
rng = np.random.default_rng(4)
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
CLOCKG = set(phase.CANONICAL_CLOCK) | {"Clock", "Bhlhe40", "Bhlhe41", "Rora"}


def unit_offset(dataset, tissue, condition, unit="bulk"):
    r = off[(off["dataset"] == dataset) & (off["tissue"] == tissue) & (off["condition"] == condition) & (off["unit"] == unit)]
    return float(r["offset_canonical_h"].iloc[0]) if len(r) else 0.0


# ---- external rhythm tables (internal phases)
ext = []
for f in sorted(list((RESULTS / "71_external").glob("external_rhythm_GSE88954.csv")) +
                list((RESULTS / "r2_external").glob("external_rhythm_*.csv"))):
    d = pd.read_csv(f, usecols=["dataset", "tissue", "condition", "time_type", "gene", "peak_h", "p", "amp_log2"])
    for (ds, t, c), g in d.groupby(["dataset", "tissue", "condition"]):
        g = g.copy()
        g["internal"] = phase.wrap24(g["peak_h"] - unit_offset(ds, t, c))
        g["unit_label"] = f"{ds}|{t}|{c}"
        ext.append(g)
ext = pd.concat(ext, ignore_index=True)
log(f"external units: {ext['unit_label'].nunique()}")

# ---- our gene rhythms (internal phases)
ours = []
for tissue in ["bm", "cochlea"]:
    t = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                    usecols=["group", "gene", "tier", "peak_zt", "p"])
    t = t[t["tier"].isin(["tier1", "tier2"])].assign(set="tier1_2")
    ours.append(t.assign(tissue=tissue))
tc = RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv"
if tc.exists():
    c = pd.read_csv(tc, usecols=["cell_type", "gene", "tierC", "peak_zt", "p"])
    c = c[c["tierC"].astype(str) == "True"].rename(columns={"cell_type": "group"}).drop(columns="tierC")
    ours.append(c.assign(tissue="cochlea", set="tierC", tier="tierC"))
ours = pd.concat(ours, ignore_index=True)
ours["internal"] = [phase.wrap24(p - unit_offset("this_atlas", t, "WT", "tissue_pool")) for p, t in zip(ours["peak_zt"], ours["tissue"])]


def circ_corr(x, y):
    mx, my = np.angle(np.exp(1j * x).mean()), np.angle(np.exp(1j * y).mean())
    sx, sy = np.sin(x - mx), np.sin(y - my)
    den = np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
    return float((sx * sy).sum() / den) if den > 0 else np.nan


conc = []
W = 2 * np.pi / 24
for (tissue, ct, st), g in ours.groupby(["tissue", "group", "set"]):
    for u, e in ext.groupby("unit_label"):
        e = e[e["p"] < 0.05].drop_duplicates("gene").set_index("gene")
        m = g[g["gene"].isin(e.index)]
        for subset, mm in [("all", m), ("non_clock", m[~m["gene"].isin(CLOCKG)])]:
            if len(mm) < 5:
                continue
            a, b = mm["internal"].to_numpy(), e.loc[mm["gene"], "internal"].to_numpy()
            d = np.abs(phase.circ_diff(a, b))
            pool = e["internal"].to_numpy()
            null = np.array([(np.abs(phase.circ_diff(a, rng.choice(pool, len(a), replace=False))) < 3).mean()
                             for _ in range(1000)])
            obs = float((d < 3).mean())
            conc.append({"tissue": tissue, "cell_type": ct, "our_set": st, "external_unit": u, "genes": subset,
                         "n_genes": len(mm), "circ_corr": circ_corr(W * a, W * b), "median_abs_diff_h": float(np.median(d)),
                         "frac_within_3h": obs, "null_frac_within_3h": float(null.mean()),
                         "p_perm": float((1 + (null >= obs).sum()) / 1001)})
conc = pd.DataFrame(conc)
write_csv(conc, MOD, "gene_phase_concordance.csv",
          "Phase concordance (clock-aligned) of our cell-type gene rhythms with every external unit",
          required=["tissue", "cell_type", "external_unit", "genes", "n_genes", "frac_within_3h", "p_perm"])

# ---- (B) signature scores in external bulk
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4)]
sigs, spec = {}, {}
for c, g in mk.groupby("cell_type"):
    g = g.sort_values("scores", ascending=False)
    strict = g.loc[g["pct_nz_reference"] <= 0.1, "names"].tolist()
    if c.startswith("Neutrophil"):
        continue
    if len(strict) >= 10:
        sigs[c], spec[c] = strict[:25], "strict"
    else:
        relaxed = [x for x in g.loc[g["pct_nz_reference"] <= 0.25, "names"] if x not in strict]
        sigs[c], spec[c] = (strict + relaxed)[:25], f"relaxed ({len(strict)} strict markers)"
other_strict = set(mk.loc[~mk["cell_type"].str.startswith("Neutrophil") & (mk["pct_nz_reference"] <= 0.1), "names"])
lin = mk[mk["cell_type"].str.startswith("Neutrophil") & (mk["logfoldchanges"] >= 4) & (mk["pct_nz_group"] >= 0.9)]
lin = lin.sort_values("scores", ascending=False).drop_duplicates("names")
sigs["Neutrophil lineage"] = [x for x in lin["names"] if x not in other_strict][:30]
spec["Neutrophil lineage"] = "lineage (neutrophil-state markers, log2FC >= 4, >= 90 % in-state)"
NEU_GROUPS = {"Neutrophil immature (Band/Mature)": ["Neutrophil (Band)", "Neutrophil (Mature)"],
              "Neutrophil mature (Aged/Activated)": ["Neutrophil (Aged)", "Neutrophil (Activated)"]}
for name, states in NEU_GROUPS.items():
    others = [x for k, v in NEU_GROUPS.items() if k != name for x in v]
    cand = mk[mk["cell_type"].isin(states) & (mk["logfoldchanges"] >= 3) & (mk["pct_nz_group"] >= 0.8)]
    excl = set(mk.loc[mk["cell_type"].isin(others) & (mk["logfoldchanges"] >= 2) & (mk["pct_nz_group"] >= 0.6), "names"])
    cand = cand.sort_values("scores", ascending=False).drop_duplicates("names")
    sigs[name] = [x for x in cand["names"] if x not in excl and x not in other_strict][:20]
    spec[name] = f"neutrophil sub-lineage ({'/'.join(s_.split(' (')[1].rstrip(')') for s_ in states)}; excluding the other part's markers)"
log("signatures: " + "; ".join(f"{k} {len(v)} ({spec[k]})" for k, v in sigs.items()))
prep_dirs = [EXTERNAL / "prepared", EXTERNAL / "prepared_r2"]
targets = [("prepared", "GSE88954"), ("prepared_r2", "GSE198770_bone_marrow"), ("prepared_r2", "GSE198770_gut"),
           ("prepared_r2", "GSE198770_liver"), ("prepared_r2", "GSE198770_lung"), ("prepared_r2", "GSE198770_skin"),
           ("prepared_r2", "GSE54650")]
score_rows, rhy_rows, dep_rows = [], [], []
for sub, ds in targets:
    meta = pd.read_csv(EXTERNAL / sub / f"{ds}__meta.csv")
    ex = pd.read_csv(EXTERNAL / sub / f"{ds}__expr.csv", index_col=0)[meta["sample"]]
    lx = np.log2(ex + 1) if meta["value_type"].iloc[0] in ("counts", "normcounts") else ex
    if meta["value_type"].iloc[0] == "counts":
        lx = np.log2(ex / ex.sum(0) * 1e6 + 1)
    ds_name = ds.split("_")[0] if ds.startswith("GSE198770") else ds
    for tissue, mt in meta.groupby("tissue"):
        x = lx[mt["sample"]]
        z = x.sub(x.mean(1), axis=0).div(x.std(1).replace(0, np.nan), axis=0)
        for sname, genes in sigs.items():
            gg = [g for g in genes if g in z.index and z.loc[g].notna().all()]
            if len(gg) < 5:
                continue
            sc = z.loc[gg].mean(0)
            for s, v in sc.items():
                r = mt.set_index("sample").loc[s]
                score_rows.append({"dataset": ds, "tissue": tissue, "condition": r["condition"], "sample": s,
                                   "time_h": r["time_h"], "signature": sname, "n_genes": len(gg), "score": float(v)})
            for cond, mc in mt.groupby("condition"):
                y = sc[mc["sample"]].to_numpy()[:, None]
                f = cosinor.fit(y, mc["time_h"].to_numpy(float))
                o = unit_offset(ds if not ds.startswith("GSE198770") else ds, tissue, cond)
                rhy_rows.append({"dataset": ds, "tissue": tissue, "condition": cond, "time_type": mc["time_type"].iloc[0],
                                 "signature": sname, "signature_specificity": spec[sname], "n_genes": len(gg),
                                 "n_samples": len(mc), "amplitude": float(f["amp"][0]),
                                 "peak_h": float(f["peak"][0]), "unit_offset_h": o,
                                 "internal_peak": float(phase.wrap24(f["peak"][0] - o)), "p": float(f["p"][0])})
            if "control" in set(mt["condition"]):
                for cond in set(mt["condition"]) - {"control"}:
                    a = sc[mt.loc[mt["condition"] == cond, "sample"]]; b = sc[mt.loc[mt["condition"] == "control", "sample"]]
                    t_, p_ = stats.ttest_ind(a, b, equal_var=False)
                    dep_rows.append({"dataset": ds, "tissue": tissue, "contrast": f"{cond}_vs_control", "signature": sname,
                                     "mean_diff": float(a.mean() - b.mean()), "t": float(t_), "p": float(p_)})
    log(f"{ds}: signatures scored")
sr = pd.DataFrame(rhy_rows)
from statsmodels.stats.multitest import multipletests  # noqa: E402
sr["q_BH"] = sr.groupby(["dataset", "tissue", "condition"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(pd.DataFrame(score_rows), MOD, "signature_scores.csv", "Immune signature scores per external bulk sample",
          required=["dataset", "tissue", "condition", "sample", "signature", "score"])
write_csv(sr, MOD, "signature_rhythm.csv", "Cosinor of immune signature scores per external tissue × condition (internal phases)",
          required=["dataset", "tissue", "condition", "signature", "internal_peak", "p", "q_BH"])
dp = pd.DataFrame(dep_rows)
if len(dp):
    dp["q_BH"] = dp.groupby(["dataset", "tissue", "contrast"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(dp, MOD, "signature_depletion_effect.csv",
          "Effect of neutrophil depletion / neutrophil Bmal1 deletion / adrenalectomy on immune signature scores",
          required=["dataset", "tissue", "contrast", "signature", "mean_diff", "p"])

# ---- (C) literature comparison against our abundance peaks
ab = pd.read_csv(RESULTS / "r2_figure/abundance_multimodel_summary.csv")
lit = pd.read_csv(SCRIPTS / "resources/literature_phases.csv")
lc = []
lit = lit.rename(columns={"expected_peak_zt_in_our_tissue": "expected_peak_zt"})
gene_meas = {m.split(":", 1)[1] for m in lit.get("measure", pd.Series(dtype=str)).dropna() if str(m).startswith("gene:")}
gtab = {t: pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "p", "peak_zt"])
        .query("gene in @gene_meas") for t in ["bm", "cochlea"]} if gene_meas else {}
for r in lit.itertuples():
    if pd.isna(r.expected_peak_zt):
        continue
    measure = str(getattr(r, "measure", "abundance"))
    tissues = ["bm", "cochlea"] if r.our_tissue == "both" else [r.our_tissue]
    for ct in str(r.our_cell_types).split(";"):
        for t in tissues:
            if measure.startswith("gene:"):
                x = gtab[t][(gtab[t]["group"] == ct) & (gtab[t]["gene"] == measure.split(":", 1)[1])]
                if not len(x):
                    continue
                pk, rhythmic = float(x["peak_zt"].iloc[0]), bool(x["p"].iloc[0] < 0.05)
                verdict = f"{measure.split(':', 1)[1]} mRNA p {x['p'].iloc[0]:.3f}"
            else:
                a = ab[(ab["tissue"] == t) & (ab["cell_type"] == ct)]
                if not len(a):
                    continue
                pk, verdict = float(a["ols_pct_peak_zt"].iloc[0]), a["verdict"].iloc[0]
                rhythmic = verdict != "not rhythmic"
            d = float(phase.circ_diff(pk, r.expected_peak_zt))
            agreement = ("not testable (not rhythmic here)" if not rhythmic else
                         "agrees (<= 4 h)" if abs(d) <= 4 else ("partly (4–6 h)" if abs(d) <= 6 else "disagrees"))
            lc.append({"source": r.source, "doi": r.doi, "tissue": t, "cell_type": ct, "measure": measure,
                       "finding": r.finding, "expected_peak_zt": r.expected_peak_zt, "our_peak_zt": pk,
                       "our_verdict": verdict, "delta_h": d, "agreement": agreement, "confidence": r.confidence})
lc = pd.DataFrame(lc)
write_csv(lc, MOD, "literature_comparison.csv", "Literature-reported trafficking/abundance phases vs our abundance peaks",
          required=["source", "doi", "tissue", "cell_type", "expected_peak_zt", "our_peak_zt", "agreement"])

# ---- (D) matrix and dossier
mat = []
for (t, ct), g in conc[conc["genes"] == "non_clock"].groupby(["tissue", "cell_type"]):
    for r in g.itertuples():
        status = ("supports" if (r.p_perm < 0.05 and r.frac_within_3h > r.null_frac_within_3h) else
                  "contradicts" if (r.p_perm > 0.95) else "neutral")
        note = ""
        if r.our_set == "tierC":      # Tier C is selected on phase agreement with GSE88954 sham / our BM: not independent
            status, note = "neutral", " [selection-dependent: Tier C requires phase agreement with GSE88954 sham or BM]"
        mat.append({"tissue": t, "cell_type": ct, "source": f"gene phases: {r.external_unit} ({r.our_set})", "status": status,
                    "effect": f"{r.frac_within_3h:.2f} within 3 h vs {r.null_frac_within_3h:.2f} null; n={r.n_genes}{note}"})
bp = RESULTS / "75_deconvolution/bayesprism_abundance_rhythm.csv"
if bp.exists():
    b = pd.read_csv(bp)
    for r in b.itertuples():
        t = "bm" if r.dataset == "GSE198770" else "cochlea"
        o = unit_offset("GSE198770_bone_marrow" if t == "bm" else "GSE88954", "bone_marrow" if t == "bm" else "cochlea",
                        r.condition)
        a = ab[(ab["tissue"] == t) & (ab["cell_type"] == r.cell_type)]
        if not len(a) or r.p >= 0.05:
            continue
        ours_int = phase.wrap24(a["ols_pct_peak_zt"].iloc[0] - unit_offset("this_atlas", t, "WT", "tissue_pool"))
        d = float(phase.circ_diff(ours_int, phase.wrap24(r.peak_h - o)))
        mat.append({"tissue": t, "cell_type": r.cell_type, "source": f"BayesPrism fraction: {r.dataset} {r.condition}",
                    "status": "supports" if abs(d) <= 4 else "contradicts", "effect": f"Δ internal phase {d:+.1f} h (ext p {r.p:.3f})"})
for r in lc.itertuples():
    mat.append({"tissue": r.tissue, "cell_type": r.cell_type, "source": f"literature: {r.source} [{r.measure}]",
                "status": ("supports" if r.agreement.startswith("agrees") else
                           "contradicts" if r.agreement.startswith("disagrees") else "neutral"),
                "effect": f"our ZT{r.our_peak_zt:.1f} vs expected ZT{r.expected_peak_zt:.0f} ({r.agreement}; {r.confidence})"})
mat = pd.DataFrame(mat)
write_csv(mat, MOD, "external_validation_matrix.csv", "Cell type × external source validation status",
          required=["tissue", "cell_type", "source", "status"])
dos = []
cts = sorted(set(zip(ab["tissue"], ab["cell_type"])) | set(zip(ours["tissue"], ours["group"])))
_ts = pd.read_csv(RESULTS / "33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv")
tiers = {t: _ts[_ts["tissue"] == t] for t in ["bm", "cochlea"]}
for t, ct in cts:
    m = mat[(mat["tissue"] == t) & (mat["cell_type"] == ct)]
    a = ab[(ab["tissue"] == t) & (ab["cell_type"] == ct)]
    ts = tiers[t][tiers[t]["group"] == ct]
    n_sup, n_con = int((m["status"] == "supports").sum()), int((m["status"] == "contradicts").sum())
    gene_ev = f"Tier1 {int(ts['n_tier1'].iloc[0])}, Tier2 {int(ts['n_tier2'].iloc[0])}" if len(ts) else "not tested"
    if t == "cochlea" and tc.exists():
        gene_ev += f", TierC {int((ours[(ours['tissue'] == 'cochlea') & (ours['group'] == ct) & (ours['set'] == 'tierC')]).shape[0])}"
    abund = (f"{a['verdict'].iloc[0]} (peak ZT{a['ols_pct_peak_zt'].iloc[0]:.1f}, {int(a['core_supporting'].iloc[0])} core models)"
             if len(a) else "not fitted")
    conclusion = ("externally supported" if n_sup >= 2 and n_sup > n_con else
                  "mixed external evidence" if n_sup and n_con else
                  "externally contradicted" if n_con > n_sup else "not externally testable / neutral")
    dos.append({"tissue": t, "cell_type": ct, "gene_rhythm_evidence": gene_ev, "abundance_rhythm": abund,
                "n_sources_supporting": n_sup, "n_sources_contradicting": n_con, "n_sources_total": len(m),
                "supporting_sources": "; ".join(m.loc[m["status"] == "supports", "source"].head(8)),
                "contradicting_sources": "; ".join(m.loc[m["status"] == "contradicts", "source"].head(8)),
                "conclusion": conclusion,
                "caveats": "bulk external tissues dilute cell-type signals; CT (DD) vs ZT (LD) designs; phases clock-aligned"})
write_csv(pd.DataFrame(dos), MOD, "celltype_validation_dossier.csv",
          "Per cell type: our gene and abundance rhythm evidence, external and literature support, conclusion, caveats",
          required=["tissue", "cell_type", "conclusion"])
log("done")
