"""Round 4, WP-B — which cells carry the replicated bulk rhythms? Atlas-based attribution of externally replicated
tissue rhythms (cochlea: GSE88954 sham, n = 3 per time point; BM: GSE198770 bone-marrow control, n = 3).

Our side (pseudobulk counts per cell type × sample, cell_type level, primary types):
  contribution share s[g, c] = type c's share of the tissue pseudobulk counts of gene g (pooled over the 6 samples) —
  the expected share of the bulk signal of g coming from c (dissociated-cell composition).
  Three 6-sample tissue profiles per gene (log2 CPM + 1):
    observed      Σ_c counts[g, c, s]
    composition   Σ_c n[c, s] · u[c] · p[g, c]        (cell numbers vary, each type's expression fixed at its pooled mean)
    regulation    Σ_c n̄[c] · u[c] · p[g, c, s]        (cell numbers fixed at their mean, each type's expression varies)
  with n = cells, u = UMIs per cell (pooled), p = proportion of the type's UMIs on gene g.
External side: genes rhythmic in the replicated bulk (cosinor p < 0.05, ptt FC ≥ 1.2); their fitted curve is evaluated
at our six ZTs after clock alignment (230 offsets: internal phase = time − offset in each dataset).
(A) Mechanism: Pearson r of the external curve with each of our three profiles; class 'composition' (r_comp ≥ 0.6 and
    > r_reg), 'regulation' (r_reg ≥ 0.6 and > r_comp), 'both' (both ≥ 0.6 within 0.1), 'not reproduced'. Null: external
    curves reassigned to random genes of the same rhythmic set (200 draws) → expected class counts.
(B) Attribution: mean contribution share per cell type among rhythmic genes vs expression-matched non-rhythmic genes
    (external p > 0.5; matched by external mesor decile; 1000 bootstrap draws) — overall, by external aligned phase bin
    (ZT0–6 … 18–24) and, for the cochlea, by GC dependence (76: dependent / independent).
(C) Controls: core-clock genes (expected broad attribution, normalised entropy ≥ 0.5); strict markers
    (00_audit/label_markers: log2FC ≥ 3, adj. p < 1e-10, ≥ 30 % detected; top 20 per type) have their highest per-cell
    expression in their own type ('pass'); the abundance-weighted main contributor is reported alongside (abundant types
    can dominate the bulk signal of a rare type's marker).
(D) Phase cross-check: genes with one dominant type (share ≥ 0.6) and a within-type rhythm in our data (p < 0.05):
    our aligned within-type phase vs the external aligned phase (circular r, fraction within 3 h, gene-permutation null).
Outputs (results/r4_attribution/): bulk_gene_attribution.csv, attribution_mechanism_summary.csv,
attribution_celltype_enrichment.csv, attribution_controls.csv, attribution_phase_crosscheck.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_attribution"
rng = np.random.default_rng(42)
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
CORE = set(SETS["clock"]["core_positive_arm"]) | set(SETS["clock"]["core_negative_arm"])
off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")


def offset(ds, tissue, unit, cond=None):
    x = off[(off["dataset"] == ds) & (off["tissue"] == tissue) & (off["unit"] == unit)]
    if cond is not None and (x["condition"] == cond).any():
        x = x[x["condition"] == cond]
    return float(x["offset_canonical_h"].iloc[0])


EXT = {"cochlea": dict(file=RESULTS / "71_external/external_rhythm_GSE88954.csv", cond="sham", ds="GSE88954", tissue="cochlea", label="GSE88954 sham"),
       "bm": dict(file=RESULTS / "r2_external/external_rhythm_GSE198770_bone_marrow.csv", cond="control", ds="GSE198770_bone_marrow",
                  tissue="bone_marrow", label="GSE198770 BM control")}
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
markers = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
markers = markers[(markers["logfoldchanges"] >= 3) & (markers["pvals_adj"] < 1e-10) & (markers["pct_nz_group"] >= 0.3)]
dep = pd.read_csv(RESULTS / "76_dependence/dependence_gene_classes.csv")
gcclass = dep[dep["dataset"] == "GSE88954"].set_index("gene")["class"]

genes_out, mech, enr, ctrl, cross = [], [], [], [], []
for t, cfg in EXT.items():
    # ---------------- our counts
    d0 = design[(design["tissue"] == t) & (design["level"] == "cell_type") & (design["status"] == "primary")]
    samples = d0.drop_duplicates("sample_id").sort_values("zt")[["sample_id", "zt"]]
    zt = samples["zt"].to_numpy(float)
    types = sorted(d0["group"].unique())
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    cnt = np.stack([pb[[f"{c}|{s}" for s in samples["sample_id"]]].to_numpy(float) for c in types], axis=1)   # G × C × S
    ncell = d0.pivot_table(index="group", columns="sample_id", values="n_cells").loc[types, samples["sample_id"]].to_numpy(float)
    genes = pb.index.to_numpy()
    lib = cnt.sum(0)                                   # C × S
    u = lib.sum(1) / ncell.sum(1)                      # UMIs per cell (pooled)
    p_pool = cnt.sum(2) / lib.sum(1)[None, :]          # G × C
    p_s = cnt / np.where(lib > 0, lib, 1)[None]        # G × C × S
    obs = cnt.sum(1)
    comp = np.einsum("cs,c,gc->gs", ncell, u, p_pool)
    reg = np.einsum("c,c,gcs->gs", ncell.mean(1), u, p_s)
    lcpm = lambda m: np.log2(m / m.sum(0, keepdims=True) * 1e6 + 1)  # noqa: E731
    Lobs, Lcomp, Lreg = lcpm(obs), lcpm(comp), lcpm(reg)
    share = p_pool * lib.sum(1)[None, :]
    share = share / np.where(share.sum(1, keepdims=True) > 0, share.sum(1, keepdims=True), 1)
    gi = pd.Series(np.arange(len(genes)), index=genes)
    expressed = obs.sum(1) >= 50

    # ---------------- external rhythms
    e = pd.read_csv(cfg["file"])
    e = e[e["condition"] == cfg["cond"]].drop_duplicates("gene").set_index("gene")
    e = e[e.index.isin(gi.index[expressed])]
    off_ext = offset(cfg["ds"], cfg["tissue"], "bulk", cfg["cond"])
    off_our = offset("this_atlas", t, "tissue_pool")
    tau = zt - off_our + off_ext                       # external time at our samples (same internal phase)
    rh = e[(e["p"] < 0.05) & (e["ptt_fc"] >= 1.2)]
    bg = e[e["p"] > 0.5]
    log(f"{t}: external rhythmic {len(rh)}, background {len(bg)}; offsets ours {off_our:+.2f} ext {off_ext:+.2f}")
    X = cosinor.design(tau)[:, 1:]

    def curves(df):
        return (X @ df[["a", "b"]].to_numpy().T).T      # genes × 6

    def rows_r(C, M):
        Cc = C - C.mean(1, keepdims=True)
        Mc = M - M.mean(1, keepdims=True)
        den = np.sqrt((Cc ** 2).sum(1) * (Mc ** 2).sum(1))
        return np.where(den > 0, (Cc * Mc).sum(1) / np.where(den > 0, den, 1), np.nan)

    idx = gi[rh.index].to_numpy()
    C = curves(rh)
    r_obs, r_comp, r_reg = rows_r(C, Lobs[idx]), rows_r(C, Lcomp[idx]), rows_r(C, Lreg[idx])

    def classify(rc, rr):
        return np.select([(rc >= 0.6) & (rr >= 0.6) & (np.abs(rc - rr) <= 0.1), (rc >= 0.6) & (rc > rr), (rr >= 0.6) & (rr > rc)],
                         ["both", "composition", "regulation"], "not reproduced")

    cls = classify(r_comp, r_reg)
    null_cls = {k: [] for k in ["composition", "regulation", "both", "not reproduced"]}
    null_obs = []
    for _ in range(200):
        perm = rng.permutation(len(idx))
        rc_, rr_ = rows_r(C[perm], Lcomp[idx]), rows_r(C[perm], Lreg[idx])
        cc_ = classify(rc_, rr_)
        for k in null_cls:
            null_cls[k].append(float((cc_ == k).mean()))
        null_obs.append(float((rows_r(C[perm], Lobs[idx]) >= 0.6).mean()))
    mech.append({"tissue": t, "external": cfg["label"], "class": "observed profile r ≥ 0.6", "n_genes": len(idx),
                 "share": float((r_obs >= 0.6).mean()), "null_share": float(np.mean(null_obs)),
                 "p_perm": (1 + sum(x >= (r_obs >= 0.6).mean() for x in null_obs)) / 201})
    for k, v in null_cls.items():
        o = float((cls == k).mean())
        mech.append({"tissue": t, "external": cfg["label"], "class": k, "n_genes": len(idx), "share": o, "null_share": float(np.mean(v)),
                     "p_perm": (1 + sum(x >= o for x in v)) / 201})
    main = np.array(types)[share[idx].argmax(1)]
    ext_int = phase.wrap24(rh["peak_h"].to_numpy() - off_ext)
    top3 = ["; ".join(f"{types[j]} {share[i, j]:.2f}" for j in np.argsort(-share[i])[:3]) for i in idx]
    genes_out.append(pd.DataFrame({
        "tissue": t, "external": cfg["label"], "gene": rh.index, "ext_p": rh["p"].to_numpy(), "ext_ptt_fc": rh["ptt_fc"].to_numpy(),
        "ext_aligned_phase": ext_int, "ext_peak_zt_equiv": phase.wrap24(ext_int + off_our),
        "ext_peak_clock_equiv": phase.zt_to_clock(ext_int + off_our),
        "gc_class_GSE88954": gcclass.reindex(rh.index).to_numpy(), "main_contributor": main, "main_share": share[idx].max(1),
        "top3_contributors": top3, "r_observed": r_obs, "r_composition": r_comp, "r_regulation": r_reg, "mechanism_class": cls}))

    # ---------------- (B) attribution enrichment with expression-matched background
    dec_edges = np.quantile(e["mesor"], np.linspace(0, 1, 11))
    dec = lambda m: np.clip(np.searchsorted(dec_edges, m, side="right") - 1, 0, 9)  # noqa: E731
    bgi = gi[bg.index].to_numpy()
    bg_dec = dec(bg["mesor"].to_numpy())
    pools = {k: bgi[bg_dec == k] for k in range(10)}

    def enrich(sub_idx, sub_dec, label):
        obs_mean = share[sub_idx].mean(0)
        boots = []
        for _ in range(1000):
            draw = np.concatenate([rng.choice(pools[k], size=(sub_dec == k).sum(), replace=True) for k in range(10) if (sub_dec == k).sum()])
            boots.append(share[draw].mean(0))
        boots = np.vstack(boots)
        for j, ct in enumerate(types):
            enr.append({"tissue": t, "external": cfg["label"], "subset": label, "n_genes": len(sub_idx), "cell_type": ct,
                        "mean_share_rhythmic": obs_mean[j], "mean_share_matched_background": boots[:, j].mean(),
                        "ratio": obs_mean[j] / boots[:, j].mean() if boots[:, j].mean() > 0 else np.nan,
                        "p_enriched": (1 + (boots[:, j] >= obs_mean[j]).sum()) / 1001,
                        "p_depleted": (1 + (boots[:, j] <= obs_mean[j]).sum()) / 1001})

    rdec = dec(rh["mesor"].to_numpy())
    enrich(idx, rdec, "all rhythmic")
    bins = pd.cut(ext_int, [0, 6, 12, 18, 24], right=False, labels=["aligned ZT0-6", "aligned ZT6-12", "aligned ZT12-18", "aligned ZT18-24"])
    for b in bins.categories:
        m = np.asarray(bins == b)
        if m.sum() >= 30:
            enrich(idx[m], rdec[m], b)
    if t == "cochlea":
        for k in ["dependent", "independent"]:
            m = gcclass.reindex(rh.index).to_numpy() == k
            if m.sum() >= 30:
                enrich(idx[m], rdec[m], f"GC {k} (GSE88954 ADX)")
    for k in ["composition", "regulation"]:
        m = cls == k
        if m.sum() >= 30:
            enrich(idx[m], rdec[m], f"mechanism: {k}")

    # ---------------- (C) controls
    ent = lambda v: float(-(v[v > 0] * np.log(v[v > 0])).sum() / np.log(len(v)))  # noqa: E731
    for gname in sorted(CORE & set(gi.index[expressed])):
        v = share[gi[gname]]
        ctrl.append({"tissue": t, "control": "core clock gene", "gene": gname, "expected_type": "broad", "main_contributor": types[v.argmax()],
                     "main_share": float(v.max()), "normalised_entropy": ent(v), "top_expressing_type": types[p_pool[gi[gname]].argmax()],
                     "pass_abundance_weighted": np.nan, "pass": ent(v) >= 0.5})
    mk = markers[markers["cell_type"].isin(types) & markers["names"].isin(gi.index[expressed])]
    mk = mk.sort_values("logfoldchanges", ascending=False).groupby("cell_type").head(20)
    for r in mk.itertuples():
        v = share[gi[r.names]]
        top = types[p_pool[gi[r.names]].argmax()]          # highest per-cell (CPM) expression, abundance-independent
        ctrl.append({"tissue": t, "control": "strict marker", "gene": r.names, "expected_type": r.cell_type, "main_contributor": types[v.argmax()],
                     "main_share": float(v.max()), "normalised_entropy": ent(v), "top_expressing_type": top,
                     "pass_abundance_weighted": types[v.argmax()] == r.cell_type, "pass": top == r.cell_type})

    # ---------------- (D) phase cross-check against our within-type rhythms
    ours = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "p", "peak_zt", "status"])
    ours = ours[ours["status"] == "primary"].set_index(["group", "gene"])
    g1 = genes_out[-1]
    dom = g1[g1["main_share"] >= 0.6]
    pairs = [(r.gene, r.main_contributor, r.ext_aligned_phase) for r in dom.itertuples() if (r.main_contributor, r.gene) in ours.index]
    if pairs:
        pp = pd.DataFrame(pairs, columns=["gene", "cell_type", "ext_aligned"])
        o = ours.loc[list(zip(pp["cell_type"], pp["gene"]))]
        pp["our_p"], pp["our_aligned"] = o["p"].to_numpy(), phase.wrap24(o["peak_zt"].to_numpy() - off_our)
        pp = pp[pp["our_p"] < 0.05]
        for ct, g in [("ALL", pp)] + [(c, s) for c, s in pp.groupby("cell_type") if len(s) >= 10]:
            if len(g) < 5:
                continue
            dif = phase.circ_diff(g["our_aligned"], g["ext_aligned"])
            fw = float((np.abs(dif) <= 3).mean())
            nul = [float((np.abs(phase.circ_diff(g["our_aligned"].to_numpy(), rng.permutation(g["ext_aligned"].to_numpy()))) <= 3).mean())
                   for _ in range(1000)]
            mu, R = phase.circ_mean(dif % 24)
            cross.append({"tissue": t, "external": cfg["label"], "cell_type": ct, "n_genes": len(g), "frac_within_3h": fw,
                          "null_frac_within_3h": float(np.mean(nul)), "p_perm": (1 + sum(x >= fw for x in nul)) / 1001,
                          "median_abs_diff_h": float(np.median(np.abs(dif))), "mean_offset_h": float(phase.circ_diff(mu, 0)),
                          "offset_resultant": R})

ga = pd.concat(genes_out, ignore_index=True)
write_csv(ga, MOD, "bulk_gene_attribution.csv",
          "Externally replicated bulk rhythmic genes: aligned phase (ZT/clock equivalents), main contributing cell type, and r with our observed / composition-only / regulation-only profiles",
          required=["tissue", "gene", "ext_aligned_phase", "ext_peak_zt_equiv", "ext_peak_clock_equiv", "main_contributor", "mechanism_class"])
mech = pd.DataFrame(mech)
write_csv(mech, MOD, "attribution_mechanism_summary.csv",
          "Share of replicated bulk rhythms reproduced by our composition-only vs within-cell regulation-only profiles, vs a gene-reassignment null",
          required=["tissue", "class", "share", "null_share", "p_perm"])
write_csv(pd.DataFrame(enr), MOD, "attribution_celltype_enrichment.csv",
          "Cell-type contribution to replicated bulk rhythmic genes vs expression-matched non-rhythmic genes (bootstrap), overall and by phase bin / GC class / mechanism",
          required=["tissue", "subset", "cell_type", "mean_share_rhythmic", "mean_share_matched_background", "ratio", "p_enriched"])
ctrl = pd.DataFrame(ctrl)
write_csv(ctrl, MOD, "attribution_controls.csv", "Positive controls of the attribution: core-clock genes broad, strict markers to their own type",
          required=["tissue", "control", "gene", "main_contributor", "pass"])
write_csv(pd.DataFrame(cross), MOD, "attribution_phase_crosscheck.csv",
          "Our within-type phase vs the replicated bulk phase for genes dominated by one cell type (clock-aligned; permutation null)",
          required=["tissue", "cell_type", "n_genes", "frac_within_3h", "null_frac_within_3h", "p_perm"])
log("\n" + mech.round(3).to_string())
log("\n" + ctrl.groupby(["tissue", "control"])["pass"].mean().round(3).to_string())
en = pd.DataFrame(enr)
log("\n" + en[(en["subset"] == "all rhythmic") & (en["p_enriched"] < 0.05)].sort_values(["tissue", "ratio"], ascending=[True, False]).round(3).to_string())
log("\n" + pd.DataFrame(cross).round(3).to_string())
