"""Round 9 (D60 / D61) — figure-ready tables for the main figures (results/r9_figures; double as Source Data).

Status codes (approved plan, revision 2): A = select / rename registered results; R = deterministic re-derivation from stored
outputs, each asserted against the stored statistic. No new inference here (N extractions from the h5ad live in 901). Every table
carries source_table, finding_id and transform columns. Display transforms (circular unwrapping, z-scores) are named in
'transform'; values are never refitted.

Run: python 900_figure_data.py [palette f2 ...]   (default: every builder below)
Outputs (results/r9_figures/): palette_celltypes.csv, palette_classes.csv, palette_tissues.csv, panel_index.csv,
  f2a_clock_phase_map.csv, f2b_clock_timecourses.csv, f2c_clock_offsets.csv, f2d_cell_phase_offsets.csv,
  f2e_tracking_strength.csv, f2f_timetelling.csv, f2g_attribution_mechanism.csv, f2h_attribution_celltype.csv,
  ed5_tissue_clock_order.csv,
  f1a_sampling_design.csv, f1c_markers_structural.csv, f1d_markers_haematopoietic.csv,
  f1f_spatial_geomx.csv (1b uses the 901 export f1b_umap_joint.csv), ed2_atlas_{columns,tree,markers,dots,validation}.csv,
  ed4_calibration.csv, f3*.csv (Fig. 3), f4a_immune_composition.csv, f4b_share_fits.csv, f4c_composition_models.csv,
  f4d_marrow_content.csv, f4d_marrow_adjusted.csv, f4e_milo_waves.csv, f4h_gse312298_genes.csv,
  f4i_baboon_concordance.csv (4i: 903 exports f4g_neutrophil_state_genes.csv), f5b_timer_genes.csv,
  f5c_timer_concordance.csv, f5d_specificity_rank.csv, f5e_timer_score{,_fit}.csv, f5f_enriched_genes.csv,
  f5g_depletion.csv, f5h_external_lines.csv, f5i_ageing_cycle.csv
  f6a_group_phase.csv, f6a_design_audit.csv, f6b_phase_separation.csv, f6c_label_transfer.csv,
  f6d_bulk_replication.csv, f6e_heat_shock_bulk.csv, f6f_noise_response.csv, f6g_composition_sc.csv,
  f6h_clock_tracking.csv, f6i_claim_ladder.csv
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.io import log, write_csv  # noqa: E402

MOD = fd.MOD
W = 2 * np.pi / 24
INDEX = []                      # panel_index rows, filled by the builders


def wrap12(x):
    """Signed circular difference in hours, in [-12, 12)."""
    return (np.asarray(x, dtype=float) + 12) % 24 - 12


def panel(fig, p, message, files, status, findings, templates):
    INDEX.append({"figure": fig, "panel": p, "message": message, "files": "; ".join(files), "status": status,
                  "finding_ids": findings, "templates_recommended_alternative": templates})


def out(df, name, desc, required=None):
    write_csv(df, MOD, name, desc, required=(required or []) + ["source_table", "finding_id", "transform"])
    return name


# ---------------------------------------------------------------- palette
def build_palette():
    ct = fd.celltypes()
    out(fd.prov(ct, "00_audit/celltype_counts_by_sample.csv", "", "display order, class palette, D49 paper labels"),
        "palette_celltypes.csv", "Cell-type display table: class, abbreviation, order, colour (class hue, lightness steps), "
        "structural flag, paper label (D49), cells per tissue", ["cell_type", "cell_class", "abbr", "order", "colour"])
    cl = pd.DataFrame({"cell_class": fd.CLASS_ORDER, "colour": [fd.CLASS_COLOUR[c] for c in fd.CLASS_ORDER],
                       "order": range(1, len(fd.CLASS_ORDER) + 1), "structural": [c in fd.STRUCTURAL for c in fd.CLASS_ORDER]})
    out(fd.prov(cl, "lib/figdata.py", "", "palette definition"), "palette_classes.csv",
        "Cell-class palette (12 classes; Tol muted + extras; avoids the tissue hues)", ["cell_class", "colour"])
    ti = pd.DataFrame({"tissue": fd.TISSUES, "label": [fd.TISSUE_LABEL[t] for t in fd.TISSUES],
                       "colour": [fd.TISSUE_COLOUR[t] for t in fd.TISSUES], "order": [1, 2]})
    out(fd.prov(ti, "lib/figdata.py", "", "palette definition"), "palette_tissues.csv",
        "Tissue palette (Okabe–Ito; cochlea left / marrow right in every figure)", ["tissue", "colour"])


# ---------------------------------------------------------------- Fig. 2 — local clocks
ARM = {"Clock": "E-box activators", "Npas2": "E-box activators",
       "Per1": "Per / Cry repressors", "Per2": "Per / Cry repressors", "Per3": "Per / Cry repressors",
       "Cry1": "Per / Cry repressors", "Cry2": "Per / Cry repressors",
       "Nr1d1": "RORE loop", "Nr1d2": "RORE loop", "Rora": "RORE loop", "Rorc": "RORE loop",
       "Dbp": "D-box", "Tef": "D-box", "Hlf": "D-box", "Nfil3": "D-box",
       "Bhlhe40": "Modifiers", "Bhlhe41": "Modifiers", "Ciart": "Modifiers"}
ARM_ORDER = ["E-box activators", "Per / Cry repressors", "RORE loop", "D-box", "Modifiers"]
ARM_ID = {"E-box activators": "ebox", "Per / Cry repressors": "percry", "RORE loop": "rore", "D-box": "dbox", "Modifiers": "mod"}
GENE_COLOUR = {"Dbp": "#E69F00", "Nr1d1": "#56B4E9", "Per2": "#009E73", "Npas2": "#CC79A7"}   # Okabe–Ito, avoids tissue hues
REF_GREY = "#7F7F7F"
GENE_ORDER = list(ARM)


def build_f2():
    src = "02_clock_controls/clock_gene_rhythm.csv"
    r = fd.read(src)
    a = r[r["gene"].isin(ARM) & (r["status"] == "primary") & r["expressed"]].copy()
    a = fd.tissue_cols(fd.annotate(a))
    a["arm"] = a["gene"].map(ARM)
    a["arm_order"] = a["arm"].map({x: i + 1 for i, x in enumerate(ARM_ORDER)})
    a["gene_order"] = a["gene"].map({g: i + 1 for i, g in enumerate(GENE_ORDER)})
    a["significant"] = a["p_perm"] <= 0.05
    a["colour_category"] = np.where(a["significant"], a["cell_class"], "n.s.")
    a["peak_minus_canonical_h"] = wrap12(a["peak_zt"] - a["canonical_peak_zt"])
    a["arm_id"] = a["arm"].map(ARM_ID)
    a["category_id"] = np.where(a["significant"], a["cell_class"], "ns")
    a["point_id"] = a["tissue"] + "|" + a["gene"] + "|" + a["cell_type"]
    a["entity_id"] = a["cell_type"] + np.where(a["significant"], "", " [p>0.05]")   # one colour category per entity (CF-SEM-002)
    cols = ["point_id", "entity_id", "tissue", "tissue_label", "tissue_order", "arm", "arm_id", "arm_order", "gene", "gene_order", "cell_type",
            "abbr", "cell_class", "order", "structural", "peak_zt", "canonical_peak_zt", "peak_minus_canonical_h", "amp_log2",
            "ptt_fc", "p_perm", "loto_phase_sd_h", "significant", "colour_category", "category_id"]
    a = a.sort_values(["tissue_order", "arm_order", "gene_order", "order"])[cols]
    f = out(fd.prov(a, src, "G04; G09", "filter: core-clock genes (Arntl absent), primary cell types, expressed; "
                    "colour_category = class if p_perm <= 0.05 else n.s."),
            "f2a_clock_phase_map.csv", "Fig. 2a: peak ZT and amplitude of each core-clock gene in each cell type (both tissues)",
            ["tissue", "gene", "cell_type", "peak_zt", "amp_log2", "p_perm"])
    for t, g in a.groupby("tissue"):
        log(f"f2a {t}: {len(g)} gene × type fits, {int(g['significant'].sum())} with p_perm <= 0.05")
    # plot registry for CF-060: both facets carry the same entity × gene registry, absent fits explicit (phase / size null)
    reg = a[["entity_id", "gene", "gene_order", "arm_id", "category_id"]].drop_duplicates(["entity_id", "gene"])
    parts = []
    for t in fd.TISSUES:
        obs = a[a["tissue"] == t][["entity_id", "gene", "peak_zt", "amp_log2"]]
        x = reg.merge(obs, on=["entity_id", "gene"], how="left")
        x["tissue"], x["tissue_label"], x["tissue_order"] = t, fd.TISSUE_LABEL[t], fd.TISSUES.index(t) + 1
        parts.append(x)
    pr_ = pd.concat(parts, ignore_index=True)
    pr_["point_id"] = pr_["tissue"] + "|" + pr_["gene"] + "|" + pr_["entity_id"]
    pr_["observed"] = pr_["peak_zt"].notna()
    pr_ = pr_.sort_values(["tissue_order", "gene_order", "entity_id"])
    f_plot = out(fd.prov(pr_, src, "G04; G09", "display registry: union of entity × gene pairs per facet; absent fits as "
                         "explicit nulls"),
                 "f2a_clock_phase_map_plot.csv", "Fig. 2a plot registry (CF-060): identical entity × gene registry in both "
                 "tissue facets, absent fits null", ["point_id", "entity_id", "gene", "category_id", "peak_zt", "amp_log2",
                                                     "tissue"])
    panel("2", "a", "Core-clock genes peak in canonical order in every structural cell type, shared with marrow",
          [f, f_plot], "A", "G04; G09", "CF-060 (CF-040 → ED5)")

    # 2b — time courses: 6 samples + stored cosinor fit
    GENES = ["Dbp", "Nr1d1", "Per2", "Npas2"]
    UNITS = [("cochlea", "Fibrocyte (Spiral Ligament)"), ("cochlea", "SV Marginal Cell"), ("cochlea", "Osteoprogenitor"),
             ("bm", "Monocyte (Classical)")]
    psrc = "02_clock_controls/clock_gene_profiles.csv"
    pr = fd.read(psrc)
    rows = []
    for t, ctype in UNITS:
        for g in GENES:
            ft = r[(r["tissue"] == t) & (r["cell_type"] == ctype) & (r["gene"] == g)]
            pt = pr[(pr["tissue"] == t) & (pr["cell_type"] == ctype) & (pr["gene"] == g)].sort_values("zt")
            assert len(ft) == 1 and len(pt) == 6, (t, ctype, g, len(ft), len(pt))
            ft = ft.iloc[0]
            # stored fit must describe these observations: OLS cosinor on the same log2cpm values reproduces it
            X = np.column_stack([np.ones(6), np.cos(W * pt["zt"]), np.sin(W * pt["zt"])])
            m, ca, sb = np.linalg.lstsq(X, pt["log2cpm"].to_numpy(), rcond=None)[0]
            amp, pk = np.hypot(ca, sb), (np.arctan2(sb, ca) / W) % 24
            assert abs(m - ft["mesor"]) < 0.02 and abs(amp - ft["amp_log2"]) < 0.02 and abs(wrap12(pk - ft["peak_zt"])) < 0.1, \
                (t, ctype, g, m, ft["mesor"], amp, ft["amp_log2"], pk, ft["peak_zt"])
            for s in pt.itertuples():
                rows.append({"series_id": f"{t}|{ctype}|{g}", "tissue": t, "cell_type": ctype, "gene": g, "sample_id": s.sample_id,
                             "zt": s.zt, "log2cpm": s.log2cpm, "mesor": ft["mesor"], "amplitude": ft["amp_log2"],
                             "peak_zt": ft["peak_zt"], "period_h": 24.0, "p_perm": ft["p_perm"]})
    b = fd.tissue_cols(fd.annotate(pd.DataFrame(rows)))
    b["facet_label"] = b["abbr"] + " (" + b["tissue_label"].str.lower() + ")"
    b["gene_colour"] = b["gene"].map(GENE_COLOUR)
    b["y_rel"] = b["log2cpm"] - b["mesor"]           # display: observations and fit centred on the stored mesor
    b["mesor_rel"] = 0.0
    unit_idx = {f"{t}|{c}": i for i, (t, c) in enumerate(UNITS)}
    b["series_order"] = [unit_idx[f"{t}|{c}"] * 10 + GENES.index(g) for t, c, g in zip(b["tissue"], b["cell_type"], b["gene"])]
    f = out(fd.prov(b, f"{psrc}; {src}", "G04; R04", "select; stored fit asserted against an OLS cosinor of the same points; y_rel = log2cpm − mesor (display)"),
            "f2b_clock_timecourses.csv", "Extended Data Fig. 5b: clock-gene log2 CPM per sample (and centred on the stored mesor, y_rel) with the stored cosinor fit (mesor, amplitude, "
            "peak, period) for representative cell types", ["series_id", "zt", "log2cpm", "mesor", "amplitude", "peak_zt"])

    # 2c — clock offset from canonical phases with gene-bootstrap 95 % CI: our cell types, tissue pools, reference series (G09)
    asrc = "r2_phase/clock_alignment_offsets.csv"
    al = fd.read(asrc)
    ours = al[al["dataset"] == "this_atlas"].copy()
    ours["cell_type"] = ours["unit"].where(ours["unit"] != "tissue_pool")
    ct = fd.celltypes().set_index("cell_type")
    ours["cell_class"] = ours["cell_type"].map(ct["cell_class"])
    ours["abbr"] = ours["cell_type"].map(ct["abbr"])
    ours["structural"] = ours["cell_type"].map(ct["structural"]).astype("boolean").fillna(False).astype(bool)
    ours["group"] = np.select([ours["unit"] == "tissue_pool", ours["structural"]],
                              ["this atlas: tissue pool", "this atlas: structural"], "this atlas: haematopoietic")
    ours["label"] = np.where(ours["unit"] == "tissue_pool", ours["tissue"].map(fd.TISSUE_LABEL) + " (all cells)",
                             ours["abbr"].fillna("") + " (" + ours["tissue"].map(fd.TISSUE_LABEL) + ")")
    ext = al[al["dataset"] != "this_atlas"].copy()
    ext["group"] = "reference: " + np.where(ext["dataset"].eq("GSE54650"), "organ atlas (GSE54650)",
                                            ext["dataset"].str.split("_").str[0] + " " + ext["tissue"].str.replace("_", " "))
    ext["label"] = (ext["dataset"].str.split("_").str[0] + " " + ext["tissue"].str.replace("_", " ") + " ("
                    + ext["condition"].str.replace("_", " ") + ")")
    c2 = pd.concat([ours, ext], ignore_index=True)
    c2["estimate_h"] = c2["offset_canonical_h"]
    c2["low_h"] = c2["offset_canonical_h"] - c2["ci_halfwidth_h"]
    c2["high_h"] = c2["offset_canonical_h"] + c2["ci_halfwidth_h"]
    main_ext = ((c2["dataset"] == "GSE88954") | ((c2["dataset"] == "GSE198770_bone_marrow") & (c2["condition"] == "control")))
    short = {"GSE88954": "GSE88954 ", "GSE198770_bone_marrow": "GSE198770 marrow "}
    c2["label_short"] = np.where(c2["dataset"] == "this_atlas",
                                 np.where(c2["unit"] == "tissue_pool", c2["tissue"].map(fd.TISSUE_LABEL) + " pool",
                                          c2["abbr"].fillna("")),
                                 c2["dataset"].map(short).fillna(c2["dataset"] + " ") + c2["condition"].str.replace("_", " "))
    # the marrow control is the only GSE198770 condition in the main panel; its label drops the condition (fits 6 pt)
    c2.loc[(c2["dataset"] == "GSE198770_bone_marrow") & (c2["condition"] == "control"), "label_short"] = "GSE198770 marrow"
    c2["colour"] = np.where(c2["dataset"] != "this_atlas", REF_GREY,
                            np.where(c2["unit"] == "tissue_pool", c2["tissue"].map(fd.TISSUE_COLOUR),
                                     c2["cell_class"].map(fd.CLASS_COLOUR)))
    tord = {t: i for i, t in enumerate(fd.TYPE_ORDER)}
    c2["plot_order"] = np.select([c2["dataset"] != "this_atlas", c2["unit"] == "tissue_pool"],
                                 [300 + c2.index.to_numpy(), 200 + (c2["tissue"] == "bm").astype(int).to_numpy()],
                                 c2["cell_type"].map(tord).fillna(99).to_numpy())
    c2["in_main_panel"] = ((c2["dataset"] == "this_atlas") & c2["determinate"].astype(bool)
                           & ((c2["unit"] == "tissue_pool") | (c2["structural"] & (c2["tissue"] == "cochlea")))) | main_ext
    f = out(fd.prov(c2, asrc, "G09; G04", "interval = offset ± reported CI half-width (gene bootstrap; display, unwrapped); "
                    "in_main_panel = determinate cochlear structural types + tissue pools + cochlear / marrow reference series"),
            "f2c_clock_offsets.csv", "Fig. 2b: clock offset from canonical phases (95 % gene-bootstrap CI) for our cell types, "
            "tissue pools and external reference series (determinate if CI half-width <= 3 h)",
            ["dataset", "unit", "estimate_h", "low_h", "high_h", "determinate", "in_main_panel"])
    m = c2[c2["in_main_panel"]]
    log(f"f2c: {len(m)} rows in main panel; structural determinate offsets {m[m['group'] == 'this atlas: structural']['estimate_h'].round(2).tolist()}")
    panel("2", "b", "Each structural cell type's clock sits at the canonical phase (offset ≈ 0 h, tight CIs), like reference "
          "organs and independent cochleae", [f], "A", "G09; G04", "CF-027 (CF-048)")

    # ED5 — tissue-level peak vs organ-atlas reference
    rsrc = "r2_phase/clock_gene_reference_phases.csv"
    ref = fd.read(rsrc)
    tps = {"cochlea": "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv", "bm": "r3_bm/bm_tissue_pseudobulk_cosinor.csv"}
    rows = []
    for t, path in tps.items():
        tp = fd.read(path).set_index("gene")
        for x in ref.itertuples():
            if x.gene not in tp.index:
                continue
            y = tp.loc[x.gene]
            rows.append({"tissue": t, "gene": x.gene, "arm": ARM.get(x.gene, ""), "reference_peak_zt": x.joint_reference_peak,
                         "canonical_peak_zt": x.canonical_peak_zt, "our_peak_zt": y["peak_zt"], "amp_log2": y["amp_log2"],
                         "p": y["p"], "q_BH": y["q_BH"],
                         "our_peak_unwrapped": x.joint_reference_peak + wrap12(y["peak_zt"] - x.joint_reference_peak),
                         "diff_h": wrap12(y["peak_zt"] - x.joint_reference_peak)})
    c = fd.tissue_cols(pd.DataFrame(rows))
    c["significant"] = c["p"] <= 0.05
    f = out(fd.prov(c, f"{rsrc}; {'; '.join(tps.values())}", "G09",
                    "join; our_peak_unwrapped = reference + circular difference (display, within ±12 h)"),
            "ed5_tissue_clock_order.csv", "ED5: tissue-pseudobulk clock-gene peak vs the joint organ-atlas reference phase",
            ["tissue", "gene", "reference_peak_zt", "our_peak_zt", "diff_h"])
    for t, g in c.groupby("tissue"):
        s = g[g["significant"]]
        log(f"ed5 {t}: {len(g)} genes; median |diff| all {g['diff_h'].abs().median():.1f} h, p <= 0.05 genes "
            f"{s['diff_h'].abs().median():.1f} h (n {len(s)})")

    # 2d / 2e — single cells track ZT
    s1 = "r3_audit/cell_clock_phase_by_sample.csv"
    d = fd.tissue_cols(fd.annotate(fd.read(s1)))
    d["offset_h"] = wrap12(d["mean_cell_phase_h"] - d["zt"])
    d["group"] = np.where(d["structural"], "structural", "haematopoietic")
    d["point_id"] = d["tissue"] + "|" + d["sample_id"] + "|" + d["cell_type"]
    d["zt_dodged"] = d["zt"] + np.where(d["structural"], -0.6, 0.6)      # display dodge so the two groups do not overplot
    f = out(fd.prov(d, s1, "R04", "offset_h = circular difference mean single-cell clock phase − ZT (display); "
                    "zt_dodged = ZT ∓ 0.6 h by group (display)"),
            "f2d_cell_phase_offsets.csv", "Fig. 2c: per-sample mean single-cell clock phase minus ZT for each cell type",
            ["tissue", "cell_type", "zt", "offset_h"])
    panel("2", "c", "Single-cell clock phase tracks ZT in structural cells (offset ≈ 0 h) but not in immune cells",
          [f], "A", "R04", "CF-023 (CF-045; CF-058 with per-cell phases, N)")
    s2 = "r3_audit/cell_clock_phase_summary.csv"
    e = fd.tissue_cols(fd.annotate(fd.read(s2)))
    e["group"] = np.where(e["structural"], "structural", "haematopoietic")
    e["significant"] = e["p_perm"] <= 0.05
    e["label_sig"] = e["abbr"] + np.where(e["significant"], " *", "")
    e = e.sort_values(["tissue_order", "order"])
    f = out(fd.prov(e, s2, "R04", "select; label_sig marks exact relabelling p <= 0.05"), "f2e_tracking_strength.csv",
            "Fig. 2d: tracking strength (offset resultant), offset, exact relabelling p and BH q per cell type",
            ["tissue", "cell_type", "offset_resultant", "p_perm", "q_BH"])
    panel("2", "d", "Tracking strength per cell type: structural types track, immune types do not",
          [f], "A", "R04", "CF-047")

    # 2f — time-telling with independent cohorts
    s3 = "74_timetelling/timetelling_summary.csv"
    tt = fd.read(s3)
    ref_parts = tt["reference"].str.split("|", expand=True)
    tt["ref_dataset"], tt["ref_tissue"], tt["ref_condition"] = ref_parts[0], ref_parts[1], ref_parts[2]
    tt["ref_label"] = tt["ref_dataset"] + " " + tt["ref_tissue"].str.replace("_", " ") + " (" + tt["ref_condition"] + ")"
    tt["target_class"] = np.where(tt["target"].eq("all_cells"), "all cells", tt["target"].str.replace("class:", "", regex=False))
    tt["structural"] = tt["target_class"].isin(fd.STRUCTURAL)
    tt = tt.rename(columns={"target_tissue": "tissue"})
    tt = fd.tissue_cols(tt)
    tt["ref_short"] = np.where(tt["ref_dataset"] == "GSE88954", "GSE88954 cochlea " + tt["ref_condition"],
                               "GSE198770 marrow " + tt["ref_condition"])
    tt["target_label"] = tt["target_class"].str.replace("all cells", "All cells", regex=False)
    tt["row_id"] = tt["reference"] + "->" + tt["tissue"] + ":" + tt["target"]
    keep_c = tt["target_class"].isin(["all cells"] + fd.STRUCTURAL)
    keep_b = tt["target_class"].isin(["all cells", "Myeloid", "Lymphoid"])
    tt["in_main_panel"] = (((tt["ref_dataset"] == "GSE88954") & (tt["tissue"] == "cochlea") & keep_c)
                           | ((tt["ref_dataset"] == "GSE198770") & (tt["tissue"] == "bm") & keep_b))
    tt["group_label"] = tt["ref_short"] + " → our " + tt["tissue_label"].str.lower()
    f = out(fd.prov(tt, s3, "G09; R09", "parse reference / target labels; in_main_panel = cochlear references → cochlear "
                    "structural classes, marrow reference → marrow all / myeloid / lymphoid"), "f2f_timetelling.csv",
            "Extended Data Fig. 5d: circular offset and resultant when independent cohorts' clocks predict our sampling times, per target class",
            ["ref_label", "tissue", "target_class", "circ_mean_offset_h", "offset_resultant"])

    # 2g / 2h — replicated bulk rhythms: within-cell regulation, attributed to cell types (R10, R11)
    s4 = "r4_attribution/attribution_mechanism_summary.csv"
    g = fd.tissue_cols(fd.read(s4))
    g["excess_over_null"] = g["share"] - g["null_share"]
    g = g[g["class"].isin(["regulation", "composition"])]
    lg = pd.concat([g.assign(series="observed", value_pct=100 * g["share"]),
                    g.assign(series="null (gene-label permutation)", value_pct=100 * g["null_share"])], ignore_index=True)
    lg["bar_label"] = lg["tissue_label"] + ": " + lg["class"]
    lg["bar_short"] = (lg["tissue"].map({"cochlea": "Coch", "bm": "Marr"}) + " "    # figure labels (fit 6 pt bars)
                       + lg["class"].map({"regulation": "reg", "composition": "comp"}))
    lg["excess_pp"] = 100 * lg["excess_over_null"]
    lg["colour"] = np.where(lg["series"] == "observed", lg["tissue"].map(fd.TISSUE_COLOUR), "#BFBFBF")
    lg = lg.sort_values(["tissue_order", "class", "series"], ascending=[True, False, False])
    f = out(fd.prov(lg, s4, "R10; R11", "filter mechanism classes regulation / composition; long format observed vs null; "
                    "value_pct = 100 × share; excess_pp = 100 × (share − null share)"), "f2g_attribution_mechanism.csv",
            "Fig. 2f: share of replicated bulk rhythms reproduced by within-cell regulation vs composition (observed vs null)",
            ["tissue", "class", "series", "value_pct", "p_perm"])
    panel("2", "f", "Replicated whole-cochlea rhythms are reproduced by within-cell regulation, not composition",
          [f], "A", "R10; R11", "CF-028 (CF-048)")
    s5 = "r4_attribution/attribution_celltype_enrichment.csv"
    h = fd.read(s5)
    h = h[h["subset"].isin(["all rhythmic", "mechanism: regulation"])].copy()
    h = fd.tissue_cols(fd.annotate(h))
    h["log2_ratio"] = np.log2(h["ratio"].clip(lower=1e-3))
    h["significant"] = h["p_enriched"] <= 0.05
    h["label_sig"] = h["abbr"] + np.where(h["significant"], " *", "")
    h["share_pct"] = 100 * h["mean_share_rhythmic"]
    h["in_main_panel"] = (h["tissue"] == "cochlea") & (h["subset"] == "all rhythmic")
    h["row_id"] = h["tissue"] + "|" + h["subset"] + "|" + h["cell_type"]
    h = h.sort_values(["tissue_order", "subset", "order"])
    f = out(fd.prov(h, s5, "R10; R11", "filter subsets; log2_ratio for display"), "f2h_attribution_celltype.csv",
            "Fig. 2e: enrichment of cell types among contributors to replicated bulk rhythms (ratio vs matched background)",
            ["tissue", "cell_type", "subset", "ratio", "p_enriched"])
    panel("2", "e", "Replicated cochlear rhythms are carried by neural, glial, vascular, bone and strial cells",
          [f], "A", "R10", "CF-047 (CF-026)")


# ---------------------------------------------------------------- Fig. 1 — atlas foundation (D61, D64)
# Atlas-card lineage markers (D64: ~18-20 rows). Chosen among the 901 candidates (one canonical marker per type + top-3
# label markers per type) by lineage specificity = min mean over the target types / max mean over all other types, scope
# 'all'; the canonical gene is kept where it is specific (ratio >= 2), otherwise the most specific candidate (Coch is
# barely detected in spiral-ligament fibrocytes here, so Foxp2). The full one-per-type set is ED2. Rows must keep
# lineage specificity > 1 (asserted).
LINEAGE_MARKERS = [
    ("Ptprq", "Hair cells", ["Hair Cell (Outer/Inner)"]), ("Snap25", "Neurons", ["Spiral Ganglion Neuron"]),
    ("Mpz", "Schwann cells", ["Schwann Cell"]), ("Kcnq1", "Strial marginal cells", ["SV Marginal Cell"]),
    ("Dct", "Strial intermediate cells", ["SV Intermediate Cell (Pigmented)", "SV Intermediate Cell (Functional)"]),
    ("Wfdc2", "Non-sensory epithelium", ["Epithelial Cell (Non-sensory)"]),
    ("Foxp2", "Spiral-ligament fibrocytes", ["Fibrocyte (Spiral Ligament)"]), ("Ptn", "Osteoprogenitors", ["Osteoprogenitor"]),
    ("Ibsp", "Osteoblasts", ["Osteoblast"]), ("Cdh5", "Endothelium", ["Endothelial Cell"]), ("Kit", "HSC / MPP", ["HSC / MPP"]),
    ("Ermap", "Erythroid", ["Erythroblast"]),
    ("Cd79a", "B lineage", ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Mature)", "B Cell (Naïve)"]),
    ("Cd3e", "T / NK", ["T Cell (CD3+)", "NK Cell"]), ("Elane", "Granulocyte-monocyte progenitors", ["Granulocyte-Monocyte Progenitor"]),
    ("Lcn2", "Neutrophils", ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]),   # Ngp / Camp are higher in proliferating marrow myelocytes (Cycling Cell)
    ("F13a1", "Monocytes", ["Monocyte (Classical)", "Monocyte (Non-classical)"]), ("C1qa", "Macrophages", ["Macrophage"]),
    ("Siglech", "pDC", ["pDC"]), ("Cpa3", "Mast cells", ["Mast Cell"])]
# Subtype markers (Fig. 1c / d, D66): one gene per subtype from that subtype's candidates in f1c_marker_expression.csv
# (the canonical gene of 901 plus the top-3 label markers of 00_audit/label_markers.csv). Specificity is measured within
# the compartment the dot matrix shows (mean in the subtype / max mean over the compartment's other subtypes, scope all);
# the canonical gene is kept where it is highest in its subtype within the compartment (ratio >= 1) and detected
# (fraction >= 0.25), otherwise the most specific detected candidate; a gene is used once. Neutrophil, B-lineage and
# strial intermediate states share programmes, so some ratios stay below 1; they are reported, not hidden.
SUBTYPE_MIN_FRAC = 0.25
SUBTYPE_CANONICAL_SPEC = 1.0
ATLAS_SHORT = {"gse283534": "GSE283534", "gse135913": "GSE135913", "gse274279": "GSE274279", "gse234926": "GSE234926"}
MIN_ATLAS_CELLS = 10           # fewer transferred cells -> heat cell left empty (unmeasured), not scored
CAL_RULES = {"nominal": "Nominal p < 0.05", "tier1": "Tier 1", "tier1_core": "Tier 1 core", "tier2": "Tier 2"}
CAL_DATA = {"GSE54650 (DD, microarray)": "Organs (GSE54650)", "GSE88954 (cochlea sham, RNA-seq)": "Cochlea (GSE88954)"}


def _flip_tree(mg: pd.DataFrame):
    """Order each merge's children by the mean display rank (TYPE_ORDER) of their leaves: same tree, structural first."""
    rank = {t: i for i, t in enumerate(fd.TYPE_ORDER)}
    kids = {r.merge_id: (r.left, r.right, r.height) for r in mg.itertuples()}
    root = mg.loc[mg["n_leaves"].idxmax(), "merge_id"]
    branches = []

    def walk(n):
        if n not in kids:
            return [n]
        a, b, h = kids[n]
        la, lb = walk(a), walk(b)
        if np.mean([rank[x] for x in la]) > np.mean([rank[x] for x in lb]):
            a, b, la, lb = b, a, lb, la
        branches.append({"branch_id": n, "left": a, "right": b, "height": h})
        return la + lb
    leaves = walk(root)
    return root, leaves, pd.DataFrame(branches)


def _subtype_markers(types, M, F, roles):
    """One marker per subtype of one compartment (rule above). Returns the gene table in subtype display order."""
    used, rows = set(), []
    for t in types:
        others = [o for o in types if o != t]
        cand = []
        for g, role in roles.get(t, []):
            spec = M.loc[g, t] / max(M.loc[g, others].max(), 0.05)
            cand.append({"gene": g, "role": role, "spec": spec, "frac": F.loc[g, t]})
        cand = pd.DataFrame(cand)
        cand = cand[~cand["gene"].isin(used)]
        assert len(cand), f"no unused marker candidate for {t}"
        canon = cand[(cand["role"] == "canonical") & (cand["spec"] >= SUBTYPE_CANONICAL_SPEC)
                     & (cand["frac"] >= SUBTYPE_MIN_FRAC)]
        ok = cand[cand["frac"] >= SUBTYPE_MIN_FRAC]
        pick = canon.iloc[0] if len(canon) else (ok if len(ok) else cand).sort_values("spec", ascending=False).iloc[0]
        used.add(pick["gene"])
        rows.append({"gene": pick["gene"], "target_type": t, "chosen_as": pick["role"],
                     "within_compartment_specificity": pick["spec"], "frac_in_target": pick["frac"]})
    return pd.DataFrame(rows)


def build_f1():
    ct = fd.celltypes().set_index("cell_type")
    cnt = fd.read("00_audit/celltype_counts_by_sample.csv")

    # 1a — sampling design: six ZTs, one pooled sample per tissue (v7: the user's study-design illustration, D119; this
    # table documents its six time points)
    s = cnt.groupby(["sample_id", "tissue", "zt", "platform"], as_index=False)["n_cells"].sum()
    a = s.pivot_table(index="zt", columns="tissue", values="n_cells", aggfunc="sum").rename(
        columns={"cochlea": "n_cells_cochlea", "bm": "n_cells_bm"}).reset_index()
    a["zt_label"] = "ZT" + a["zt"].astype(int).astype(str)
    a["clock_time"] = ((a["zt"] + 8) % 24).astype(int).map(lambda h: f"{h:02d}:00")
    a["phase_light"] = np.where(a["zt"] < 12, "light (ZT0-12)", "dark (ZT12-24)")
    assert sorted(a["zt"]) == [3.0, 7.0, 11.0, 15.0, 19.0, 23.0]
    f1a = out(fd.prov(a, "00_audit/celltype_counts_by_sample.csv", "", "one row per ZT: cells per tissue; clock = ZT + 8"),
              "f1a_sampling_design.csv", "Fig. 1a: sampling design (six ZTs, one pooled sample per tissue)",
              ["zt", "n_cells_cochlea", "n_cells_bm"])

    # 1b-d — the joint UMAP is exported by 901 (f1b_umap_joint.csv); panel rows of Fig. 1 v5 close this builder

    # 1c / 1d — one marker per subtype, structural and haematopoietic compartments (D66; rule at SUBTYPE_MIN_FRAC)
    me = fd.read(f"{MOD}/f1c_marker_expression.csv")
    me = me[me["scope"] == "all"]
    M = me.pivot(index="gene", columns="cell_type", values="mean_log1p_cp10k")
    F = me.pivot(index="gene", columns="cell_type", values="frac_expressing")
    by_abbr = {a: t for t, a in ct["abbr"].items()}
    roles = {}
    for g, cf in me.drop_duplicates("gene")[["gene", "candidate_for"]].itertuples(index=False):
        for item in str(cf).split("; "):
            role, a = item.split(":")
            roles.setdefault(by_abbr[a], []).append((g, role))
    for t in roles:                                    # a gene listed twice for one subtype keeps its canonical role
        best = {}
        for g, role in roles[t]:
            if g not in best or role == "canonical":
                best[g] = role
        roles[t] = sorted(best.items(), key=lambda x: (x[1] != "canonical", x[0]))
    order = ct.sort_values("order")
    compartments = [("structural", "c", "f1c_markers_structural.csv",
                     list(order.index[order["structural"]]), "Fig. 1c: one marker per structural subtype"),
                    ("haematopoietic", "d", "f1d_markers_haematopoietic.csv",
                     list(order.index[~order["structural"]]), "Fig. 1d: one marker per haematopoietic subtype")]
    marker_files = []
    for comp, letter, fname, types, desc in compartments:
        sel = _subtype_markers(types, M, F, roles)
        sel["row_order"] = range(1, len(sel) + 1)
        sel["target_abbr"] = sel["target_type"].map(ct["abbr"])
        log(f"1{letter} {comp}: " + ", ".join(f"{r.target_abbr}={r.gene} ({r.within_compartment_specificity:.2f})"
                                               for r in sel.itertuples()))
        dots = []
        for r in sel.itertuples():
            lo, hi = M.loc[r.gene, types].min(), M.loc[r.gene, types].max()
            for j, t in enumerate(types):
                dots.append({"gene": r.gene, "row_order": r.row_order, "target_type": r.target_type,
                             "target_abbr": r.target_abbr, "chosen_as": r.chosen_as,
                             "within_compartment_specificity": r.within_compartment_specificity,
                             "cell_type": t, "abbr": ct.loc[t, "abbr"], "column_order": j + 1, "cell_class": ct.loc[t, "cell_class"],
                             "colour": ct.loc[t, "colour"], "mean_log1p_cp10k": M.loc[r.gene, t],
                             "frac_expressing": F.loc[r.gene, t], "scaled_mean": (M.loc[r.gene, t] - lo) / (hi - lo)})
        dots = pd.DataFrame(dots)
        assert dots["scaled_mean"].between(0, 1).all() and dots["frac_expressing"].between(0, 1).all()
        assert len(dots) == len(types) ** 2 and sel["gene"].is_unique
        f = out(fd.prov(dots, f"{MOD}/f1c_marker_expression.csv", "",
                        f"one marker per {comp} subtype (canonical if within-compartment specificity >= "
                        f"{SUBTYPE_CANONICAL_SPEC:g} and fraction >= {SUBTYPE_MIN_FRAC:g}, else the most specific detected "
                        "candidate); scaled_mean = per-gene min-max across the compartment's subtypes"), fname,
                desc + " (rows: genes; columns: subtypes; scaled mean expression and fraction expressing)",
                ["gene", "cell_type", "scaled_mean", "frac_expressing", "mean_log1p_cp10k", "within_compartment_specificity"])
        marker_files.append(f)

    # external atlases (R15): our label markers' z-score in cells of four cochlear atlases transferred to each subtype;
    # shown in the atlas card (Fig. 1f / ED2) as ed2_atlas_validation.csv below
    ac = fd.read("r4_concordance/atlas_coverage.csv")
    ac["atlas"] = ac["key"].map(ATLAS_SHORT)
    assert ac["atlas"].notna().all()
    ac["value"] = np.where(ac["n_cells"] >= MIN_ATLAS_CELLS, ac["marker_z_in_label"], np.nan)
    ac["abbr"] = ac["cell_type"].map(ct["abbr"])
    ac["column_order"] = ac["cell_type"].map(ct["order"])
    ac["colour"] = ac["cell_type"].map(ct["colour"])
    ac["atlas_order"] = ac["atlas"].map({v: i + 1 for i, v in enumerate(ATLAS_SHORT.values())})
    assert ac["column_order"].notna().all() and len(ac) == len(ATLAS_SHORT) * len(fd.TYPE_ORDER)

    # 1f — spatial localisation (GeoMx deconvolution; types with region q < 0.05)
    sp_ = fd.annotate(fd.read("r3_spatial/geomx_deconvolution_summary.csv").drop(columns=["cell_class"]))
    sp_ = sp_[sp_["q_region"] < 0.05].copy()
    sp_["diff_pp"] = 100 * (sp_["mean_prop_neuron_control"] - sp_["mean_prop_supporting_control"])
    sp_["mean_pct"] = 50 * (sp_["mean_prop_neuron_control"] + sp_["mean_prop_supporting_control"])
    sp_["group"] = np.where(sp_["structural"], "structural", "haematopoietic")
    sp_ = sp_.sort_values("diff_pp", ascending=False)
    f1f = out(fd.prov(sp_, "r3_spatial/geomx_deconvolution_summary.csv", "R07",
                      "filter q_region < 0.05; diff_pp = 100 x (neuron-region - supporting-region mean proportion); "
                      "mean_pct = mean of both x 100"), "f1f_spatial_geomx.csv",
              "Fig. 1f: deconvolved cell-type proportions differing between neuron and supporting-cell regions (control GeoMx)",
              ["cell_type", "diff_pp", "mean_pct", "q_region"])

    # ED2 — atlas card (moved from main Fig. 1, D66): tree, class / group tracks, tissue and ZT composition, lineage markers,
    # external validation
    root, leaves, br = _flip_tree(fd.read(f"{MOD}/f1c_dendrogram_merges.csv"))
    assert sorted(leaves) == sorted(fd.TYPE_ORDER)
    col = pd.DataFrame({"cell_type": leaves, "column_order": range(1, len(leaves) + 1)})
    col["abbr"] = col["cell_type"].map(ct["abbr"])
    col["cell_class"] = col["cell_type"].map(ct["cell_class"])
    col["group"] = np.where(col["cell_type"].map(ct["structural"]), "structural", "haematopoietic")
    byt = cnt.pivot_table(index="cell_type", columns="tissue", values="n_cells", aggfunc="sum").fillna(0).astype(int)
    byz = cnt.pivot_table(index="cell_type", columns="zt", values="n_cells", aggfunc="sum").fillna(0).astype(int)
    col["n_cochlea"], col["n_bm"] = col["cell_type"].map(byt["cochlea"]), col["cell_type"].map(byt["bm"])
    for z in byz.columns:
        col[f"n_zt{int(z)}"] = col["cell_type"].map(byz[z])
    col["total"] = col["n_cochlea"] + col["n_bm"]
    assert (col["total"] == col[[f"n_zt{int(z)}" for z in byz.columns]].sum(axis=1)).all()
    assert (col["total"] == col["cell_type"].map(ct["n_cells_cochlea"] + ct["n_cells_bm"])).all()
    groups = col.groupby("group", sort=False)["column_order"].agg(["min", "max", "count"])
    assert (groups["max"] - groups["min"] + 1 == groups["count"]).all(), "group bands must be contiguous"
    f_cols = out(fd.prov(col, "00_audit/celltype_counts_by_sample.csv; f1c_dendrogram_merges.csv", "",
                         "column order = dendrogram leaves with children ordered by mean display rank (same tree)"),
                 "ed2_atlas_columns.csv", "ED2: atlas-card columns (35 types): order, class, group, cells by tissue and ZT",
                 ["cell_type", "column_order", "total"])
    br["root"] = root
    f_tree = out(fd.prov(br, f"{MOD}/f1c_dendrogram_merges.csv", "",
                         "children ordered by mean display rank of leaves (tree unchanged); height = 1 - Pearson r"),
                 "ed2_atlas_tree.csv", "ED2: cell-type dendrogram branches in display order",
                 ["branch_id", "left", "right", "height"])
    rows, mk = [], []
    pos = dict(zip(col["cell_type"], col["column_order"]))
    for g, lin, targets in LINEAGE_MARKERS:
        others = [t for t in fd.TYPE_ORDER if t not in targets]
        spec = M.loc[g, targets].min() / max(M.loc[g, others].max(), 0.05)
        mk.append({"gene": g, "lineage": lin, "target_types": "; ".join(targets), "lineage_specificity": spec,
                   "min_frac_in_targets": F.loc[g, targets].min(), "first_target_column": min(pos[t] for t in targets)})
        lo, hi = M.loc[g].min(), M.loc[g].max()
        for t in fd.TYPE_ORDER:
            rows.append({"gene": g, "cell_type": t, "mean_log1p_cp10k": M.loc[g, t], "frac_expressing": F.loc[g, t],
                         "scaled_mean": (M.loc[g, t] - lo) / (hi - lo)})
    mk = pd.DataFrame(mk).sort_values("first_target_column").reset_index(drop=True)
    mk["row_order"] = range(1, len(mk) + 1)
    assert (mk["lineage_specificity"] > 1.0).all(), mk.loc[mk["lineage_specificity"] <= 1, "gene"].tolist()
    f_mk = out(fd.prov(mk, f"{MOD}/f1c_marker_expression.csv", "",
                       "lineage specificity = min mean over target types / max mean over other types (scope all)"),
               "ed2_atlas_markers.csv", "ED2: lineage markers of the atlas card (rows) with specificity",
               ["gene", "lineage", "lineage_specificity", "row_order"])
    dots = pd.DataFrame(rows).merge(mk[["gene", "row_order"]], on="gene").merge(col[["cell_type", "column_order"]], on="cell_type")
    f_dots = out(fd.prov(dots.sort_values(["row_order", "column_order"]), f"{MOD}/f1c_marker_expression.csv", "",
                         "scaled_mean = per-gene min-max of the type means across the 35 types (display; dot area = colour)"),
                 "ed2_atlas_dots.csv", "ED2: lineage-marker expression per cell type (scaled mean; raw mean and fraction kept)",
                 ["gene", "cell_type", "scaled_mean", "mean_log1p_cp10k", "frac_expressing"])
    acd = ac.drop(columns=["column_order"]).merge(col[["cell_type", "column_order"]], on="cell_type")
    f_heat = out(fd.prov(acd.sort_values(["atlas_order", "column_order"]), "r4_concordance/atlas_coverage.csv", "R15",
                         f"value = marker_z_in_label where >= {MIN_ATLAS_CELLS} cells were transferred to the label, else empty"),
                 "ed2_atlas_validation.csv", "ED2: external-atlas validation in atlas-card (dendrogram) column order",
                 ["atlas", "cell_type", "value", "n_cells"])

    # ED4 — calibration of rhythm calls on dense held-out series (R09)
    b = fd.read("r4_benchmark/benchmark_summary.csv")
    b = b[(b["condition"] == "clean") & (b["unit"] == "ALL") & b["rule"].isin(CAL_RULES)].copy()
    b["dataset_label"], b["rule_label"] = b["dataset"].map(CAL_DATA), b["rule"].map(CAL_RULES)
    b["fdp_strict_pct"], b["sensitivity_pct"] = 100 * b["mean_fdp_strict"], 100 * b["mean_sensitivity"]
    b["row_id"] = b["dataset_label"] + "|" + b["rule"]
    b["rule_order"] = b["rule"].map({k: i for i, k in enumerate(CAL_RULES)})
    b = b.sort_values(["dataset", "rule_order"])
    assert len(b) == 8
    f_cal = out(fd.prov(b, "r4_benchmark/benchmark_summary.csv", "R09",
                        "filter clean draws, all genes, four rules; percentages"), "ed4_calibration.csv",
                "ED4: realised false-discovery proportion (strict truth) and sensitivity of our calling rules on dense "
                "held-out series", ["dataset", "rule", "fdp_strict_pct", "sensitivity_pct"])

    # Fig. 1 v7 (D67 / D68, D119; scripts/910_fig1_atlas.py): panel rows in figure order (D79)
    umap = ["f1b_umap_joint.csv", "palette_celltypes.csv", "palette_classes.csv"]
    panel("1", "a", "Study design (the user's illustration, D119): collection every 4 h at ZT3-ZT23 (LD 12:12), "
          "single-cell RNA sequencing, bioinformatic analysis, validation by confocal imaging and flow cytometry; "
          "the table gives the six time points and cells per tissue", [f1a], "A", "", "custom (910)")
    panel("1", "b", "160,643 cells of cochlea and adjacent temporal-bone marrow resolve 35 subtypes in 12 classes (joint "
          "UMAP; class hue, subtype lightness; class keylines; boxes mark the zooms in d)", umap, "N", "", "custom (910)")
    panel("1", "c", "Each sample on the same embedding (cells of that sample coloured by tissue over all cells): time points "
          "intermix within subtypes", ["f1b_umap_joint.csv"], "N", "", "custom (910)")
    panel("1", "d", "Zoomed windows of the joint UMAP (i, lymphoid; ii, myeloid and progenitors) with every subtype named",
          umap[:2], "N", "", "custom (910)")
    panel("1", "e", "Each subtype carries its own marker (one gene per subtype; structural | haematopoietic; dot area = "
          "fraction expressing, colour = scaled mean)", marker_files, "N", "", "custom (910)")
    panel("1", "f", "Atlas card: subtypes in dendrogram order with class, cells, tissue and ZT shares and marker support in "
          "four independent cochlear atlases (per-sample composition rows dropped in v7, D119; see ED1b, ED1f)",
          [f_cols, f_tree, f_heat], "A + N", "R15", "custom (910)")
    panel("1", "g", "Spatial check: neurons, glia and immune cells sit in their expected cochlear regions (control GeoMx "
          "deconvolution; types with q < 0.05)", [f1f], "A", "R07", "custom (910)")


# ---------------------------------------------------------------- Fig. 3 — systemic night heat-shock programme (D68)
SYSTEMIC_LABEL = {"heat_shock_hsf1": "Heat shock (HSF1)", "feeding_insulin_mtor": "Feeding (insulin–mTOR)",
                  "hypoxia": "Hypoxia (HIF)", "adrenergic_camp": "Adrenergic (cAMP)",
                  "glucocorticoid_response": "Glucocorticoid response"}
NIGHT_SETS = {   # gene sets of the independent-cohort check (252_pilot_noise_linking.py)
    "heat shock": ["Hspa1a", "Hspa1b", "Hsph1", "Dnajb1", "Hspa8", "Hsp90aa1", "Hspb1", "Hspe1", "Hspd1", "Dnaja1"],
    "IEG": ["Fos", "Fosb", "Jun", "Junb", "Egr1", "Egr2", "Egr3", "Ier2", "Nr4a1", "Dusp1", "Atf3", "Klf2", "Klf4", "Zfp36"],
    "NF-kB": ["Nfkbia", "Nfkbiz", "Tnfaip3", "Ier3", "Cxcl2", "Ccl3", "Ccl4", "Nfkbid", "Rel", "Relb"]}
NIGHT_CONTRASTS = {"night_minus_day_no_noise_shamsurg": "Sham surgery", "night_minus_day_no_noise_adx": "Adrenalectomy"}
MOTIF_LABEL = {"HSE_GAAnnTTCnnGAA": "HSE (HSF1)", "Ebox_CACGTG": "E-box", "Eprime_CACGTT": "E′-box",
               "Dbox_TTAYGTAA": "D-box", "RORE_WAWNTRGGTCA": "RORE", "CRE_TGACGTCA": "CRE", "NFkB_GGGRNYYYCC": "NF-κB"}
DRIVER_ORDER = ["core clock", "clock-controlled", "temperature", "glucocorticoid-driven", "handling-linked",
                "composition-linked", "sample-pattern-linked", "unexplained"]


def build_f3():
    # 3a — systemic programmes: per-type cosinor of each cue set and the weighted common phase per tissue
    sr = fd.annotate(fd.read("82_systemic/systemic_cue_rhythm.csv"))
    sr["set_label"] = sr["set"].map(SYSTEMIC_LABEL)
    sr["set_order"] = sr["set"].map({k: i + 1 for i, k in enumerate(SYSTEMIC_LABEL)})
    assert sr["set_label"].notna().all()
    f3a = out(fd.prov(sr, "82_systemic/systemic_cue_rhythm.csv", "G05; R05", "select; labels"), "f3a_systemic_phases.csv",
              "Fig. 3a: peak phase and amplitude of five systemic-cue gene sets in every cell type and tissue",
              ["tissue", "cell_type", "set", "peak_zt", "amplitude", "p_perm", "q_BH"])
    cp = fd.read("82_systemic/systemic_cue_common_phase.csv")
    cp["set_label"] = cp["set"].map(SYSTEMIC_LABEL)
    cp["set_order"] = cp["set"].map({k: i + 1 for i, k in enumerate(SYSTEMIC_LABEL)})
    hs = cp.set_index(["tissue", "set"]).loc[[("bm", "heat_shock_hsf1"), ("cochlea", "heat_shock_hsf1")]]
    assert np.allclose(hs["weighted_common_peak_zt"], [17.06, 17.35], atol=0.01)
    f3a2 = out(fd.prov(cp, "82_systemic/systemic_cue_common_phase.csv", "G05; R05", "select; labels"),
               "f3a_systemic_common.csv", "Fig. 3a: amplitude-weighted common peak phase of each cue set per tissue "
               "(Rayleigh test; V-test against the expected phase)", ["tissue", "set", "weighted_common_peak_zt",
                                                                      "weighted_resultant", "rayleigh_p"])
    panel("3", "a", "Five systemic-cue programmes peak in the active phase across the cell types of both tissues "
          "(amplitude-weighted common peaks ZT14.9-19.6); heat shock is the most phase-concentrated programme in marrow "
          "(ZT17.1, resultant 0.97) and peaks at ZT17.3 in cochlea (0.83; feeding 0.89). Cell types share the six samples, "
          "so the convergence is descriptive", [f3a, f3a2], "A", "G05; R05", "custom (911)")
    panel("3", "b", "Whole-tissue pseudobulk: 5 of 10 heat-shock genes are rhythmic in marrow (q < 0.1, peaks "
          "ZT16.4-17.9); in cochlea none reaches q < 0.1 (Hsph1 p 0.036 at ZT18.8), so the cochlear night programme is "
          "resolved only per cell type (a)", ["f3b_tissue_timecourses.csv"], "R", "R05", "custom (903 R extraction)")

    # 3c — separable from handling: independent cohort (GSE107086, no noise) and stress-like share of calls per class
    de = fd.read("r2_pilots/pilot_GSE107086_de.csv")
    chk = fd.read("r2_pilots/pilot_night_programme_check.csv").set_index(["contrast", "gene_set"])
    rows = []
    for k, lab in NIGHT_CONTRASTS.items():
        d = de[de["contrast"] == k].set_index("gene")
        for s, genes in NIGHT_SETS.items():
            x = d[d.index.isin(genes)]
            ref = chk.loc[(k, s)]
            assert len(x) == ref["n_genes"] and abs(x["logFC"].mean() - ref["mean_logFC_night_minus_day"]) < 1e-9
            for g, r in x.iterrows():
                rows.append({"contrast": k, "condition": lab, "gene_set": s, "gene": g, "logFC_night_minus_day": r["logFC"],
                             "p": r["p"], "q_BH": r["q_BH"], "set_mean": ref["mean_logFC_night_minus_day"],
                             "set_frac_positive": ref["frac_positive"], "set_p_wilcoxon": ref["p_wilcoxon_vs_all"]})
    f3c = out(fd.prov(pd.DataFrame(rows), "r2_pilots/pilot_GSE107086_de.csv; r2_pilots/pilot_night_programme_check.csv",
                      "R05", "per-gene night − day log2FC (no noise; ~ZT17.5 vs ~ZT5.5) for the three programme sets; set "
                      "summaries asserted equal to the pilot check"), "f3c_night_external.csv",
              "Fig. 3c: independent cochlea cohort (GSE107086): night − day change of heat-shock, immediate-early and NF-κB "
              "genes without noise, sham surgery and adrenalectomy", ["condition", "gene_set", "gene",
                                                                      "logFC_night_minus_day", "p"])
    hf = fd.read("r3_audit/handling_gene_flags.csv")
    hf = hf[hf["gene_class"] != "ALL"]
    tiers = {"bm": ["tier1", "tier2"], "cochlea": ["tier2", "tierC"]}
    pooled = []
    for t, ts in tiers.items():
        x = hf[(hf["tissue"] == t) & hf["tier"].isin(ts)].groupby("gene_class")[["n_calls", "n_stress_like"]].sum()
        for c, r in x.iterrows():
            pooled.append({"tissue": t, "tiers": "+".join(ts), "gene_class": c, "n_calls": int(r["n_calls"]),
                           "n_stress_like": int(r["n_stress_like"]), "frac_stress_like": r["n_stress_like"] / r["n_calls"]})
    f3c2 = out(fd.prov(pd.DataFrame(pooled), "r3_audit/handling_gene_flags.csv", "R05",
                       "sum over phase bins and tiers (marrow Tier 1 + 2; cochlea Tier 2 + C) per gene class"),
               "f3c_handling_classes.csv", "Fig. 3c: share of rhythm calls that follow the per-sample handling (stress) "
               "profile, by gene class", ["tissue", "gene_class", "n_calls", "frac_stress_like"])
    panel("3", "c", "The night heat-shock rise recurs in an independent cochlea cohort (9 / 10 genes up after sham "
          "surgery, 10 / 10 after adrenalectomy) whereas immediate-early genes do not rise; in our data immediate-early "
          "calls are the most handling-like (53-65 % vs 32-36 % of heat-shock and 33-40 % of NF-κB calls)",
          [f3c, f3c2], "A + R", "R05", "custom")

    # 3d — HSE-motif carriers peak at ZT17.8-18.4 (directional permutation q < 0.1)
    mo = fd.annotate(fd.read("44_motifs/promoter_motif_phase.csv"))
    mo["motif_label"] = mo["motif"].map(MOTIF_LABEL)
    assert mo["motif_label"].notna().all()
    f3d = out(fd.prov(mo, "44_motifs/promoter_motif_phase.csv", "G05", "select; labels"), "f3d_motif_phase.csv",
              "Fig. 3d: circular mean peak of rhythmic promoter-motif carriers per cell type vs the motif's expected phase",
              ["tissue", "cell_type", "motif", "circ_mean_peak_zt", "resultant_length", "q_directional_perm"])
    panel("3", "d", "Promoter HSE carriers peak at ZT17.8-18.4 in marrow progenitors and monocytes (directional "
          "permutation q < 0.05); other motifs are rarely phase-concentrated", [f3d], "A", "G05", "custom")

    # 3e — clock-output test: share of calls at the constant-darkness-predicted phase vs a phase-permuted null
    dd = fd.read("r4_drivers/diurnal_circadian_status.csv")
    dd = dd[dd["n_calls"].notna() & (dd["finding_id"] != "R01")].copy()      # R01 repeats the G01 call set
    dd = dd.rename(columns={"finding_id": "call_set_finding"})               # keep the call-set ID beside provenance
    f3e = out(fd.prov(dd, "r4_drivers/diurnal_circadian_status.csv", "R16", "rows with call sets; R01 duplicates G01"),
              "f3e_clock_output.csv", "Fig. 3e: share of rhythm calls peaking at their constant-darkness-predicted phase "
              "(GSE54650 organs) vs a phase-permutation null", ["call_set_finding", "n_calls", "frac_dd_support",
                                                                "frac_dd_support_phase_null", "p_dd_support_phase_perm"])
    pn = fd.read("r4_drivers/driver_phase_null.csv")
    pn = fd.annotate(pn[(pn["flag"] == "dd_support") & (pn["tier_set"] == "tier1/2") & (pn["cell_type"] != "ALL")])
    f3e2 = out(fd.prov(pn, "r4_drivers/driver_phase_null.csv", "R16", "constant-darkness flag, Tier 1/2"),
               "f3e_dd_by_type.csv", "Fig. 3e: per cell type, observed vs phase-permuted share of calls at the "
               "constant-darkness-predicted phase", ["tissue", "cell_type", "observed_share", "null_mean_share", "p_perm"])
    panel("3", "e", "Only core-clock (94 % vs 40 %) and heat-shock (84 % vs 34 %) calls peak at their constant-darkness "
          "phases above chance; neutrophil-state programmes do not", [f3e, f3e2], "A", "R16", "custom")

    # 3f — mouse (night) vs baboon (day): same internal phase
    bb = fd.read("r4_crossspecies/baboon_heatshock_phase.csv")
    bb["row_type"] = np.where(bb["baboon_tissue"].str.startswith("SUMMARY"), "summary", "tissue")
    f3f = out(fd.prov(bb, "r4_crossspecies/baboon_heatshock_phase.csv", "R12", "select; row_type"),
              "f3f_baboon_heatshock.csv", "Fig. 3f: heat-shock programme phase per baboon tissue (raw ZT and clock-aligned) "
              "and the mouse summary", ["baboon_tissue", "peak_zt", "p", "aligned", "row_type"])
    panel("3", "f", "Heat shock peaks at night in mouse (ZT17.6) and by day in baboon (ZT7.8) but at the same internal "
          "phase once clocks are aligned (ZT17.0 vs 17.7)", [f3f], "A", "R12", "custom")

    # 3g — driver landscape of rhythm calls per cell type
    dl = fd.read("r4_drivers/driver_landscape.csv")
    dl = fd.annotate(dl[dl["cell_type"] != "ALL"])
    dl["driver_order"] = dl["driver"].map({k: i + 1 for i, k in enumerate(DRIVER_ORDER)})
    assert dl["driver_order"].notna().all()
    f3g = out(fd.prov(dl, "r4_drivers/driver_landscape.csv", "R16", "select; driver order"), "f3g_driver_landscape.csv",
              "Fig. 3g: primary driver of each rhythm call, share per cell type (Tier 1/2; cochlea Tier C exploratory)",
              ["tissue", "tier_set", "cell_type", "driver", "share", "n_calls"])
    panel("3", "g", "Sample-pattern-linked (36 % in both tissues) and clock-controlled calls (25 % marrow, 20 % cochlea) "
          "dominate the Tier-1/2 calls; handling-linked calls are frequent in marrow (17 %, up to 35 % in pre-B cells) but "
          "not in cochlea (6 %); core-clock, temperature and glucocorticoid drivers are each <= 1.2 %. Cell types with < 20 "
          "calls are not drawn", [f3g], "A", "R16", "custom")


# ---------------------------------------------------------------- Fig. 4 — marrow-set immunity (D68)
CORE_MODELS = {"clr": "CLR (permutation)", "propeller": "propeller", "sccomp": "sccomp", "quasibinomial": "quasi-binomial",
               "betabinomial": "beta-binomial"}
NEU_STATES = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Band)", "Neutrophil (Mature)",
              "Granulocyte-Monocyte Progenitor"]


def _flow_supported():
    """Findings the independent flow cohort supports (validation-matrix line 'independent_flow_cohort'; D90)."""
    vm = fd.read("r2_audit/validation_matrix.csv")
    return set(vm.loc[(vm["dimension"] == "independent_flow_cohort") & (vm["status"] == "supports"), "finding_id"])


def _abundance_register():
    """Register verdicts of the abundance-rhythm findings (A..), keyed by tissue and cell type; flow_replicated = the
    independent flow cohort supports the finding (D90, drawn as a diamond in Fig. 4e; D91, D97)."""
    r = fd.read("r2_audit/positive_findings_register.csv")
    r = r[r["category"] == "abundance rhythm"].copy()
    r["cell_type"] = r["statement"].str.split(":").str[0]
    assert not r.duplicated(["tissue", "cell_type"]).any()
    r["flow_replicated"] = r["finding_id"].isin(_flow_supported())
    return r[["tissue", "cell_type", "finding_id", "verdict", "peak_zt", "flow_replicated"]].rename(
        columns={"finding_id": "register_id", "verdict": "register_verdict", "peak_zt": "register_peak_zt"})


def _quintiles(s):
    """'Q1 ZT8.3 (n 122); Q2 ...' -> [(1, 8.3, 122), ...]."""
    rows = []
    for part in str(s).split(";"):
        m = re.match(r"\s*Q(\d) ZT([\d.]+) \(n (\d+)\)\s*$", part)
        assert m, part
        rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
    assert [q for q, _, _ in rows] == [1, 2, 3, 4, 5]
    return rows


def build_f4():
    """Fig. 4 (D134; v3 D142) — the marrow-cochlea immune channel: 3D route, neutrophils at the cochlea and spatial states
    (930), neutrophils in the cochlear fluids (931; former Fig. 5a) and the parabiosis survey (930, 931, 935) with the
    upgraded quantification: partner-derived cells by the upgraded body rule (960) and immune cells typed on segmented
    nuclei (961), both checked against blinded visual labels (944, 946, 960, 961)."""
    from scipy.stats import beta
    # 4a-c — the user's imaging (D82; scripts/930_image_panels.py): pixels as exported; crops and scale in the log
    img, pn = "image_panels_log.csv", "figures/panels/"
    panel("4", "a", "BM-cochlear immune channel (D84): two 3D surface renders with the user's manual annotation (teal) of "
          "vessels extending from the cochlea, and the same views unannotated",
          [pn + "vessels_view1_annotated.png", pn + "vessels_view2_annotated.png", pn + "vessels_view1_original.png",
           pn + "vessels_view2_original.png", img], "M", "", "custom (914)")
    panel("4", "b", "Neutrophils (green) along vessels (blue) in the temporal bone beside the cochlea: 3D overview with the "
          "zoomed region and the zoom", [pn + "marrow_neutrophils_overview.png", pn + "marrow_neutrophils_zoom.png", img],
          "M", "", "custom (914)")
    panel("4", "c", "Spatial transcriptomics: deconvolved neutrophil states (band, mature, activated, aged) across a tissue "
          "section", [pn + "spatial_neutrophil_states.png", "image_spatial_key.csv", img], "M", "", "custom (914)")
    panel("4", "d", "FUT4+ neutrophils in bone-marrow, perilymph and endolymph preparations of wild-type mice (Olympus "
          "FV3000, 60x, D134; DAPI, FUT4, CD45, merge; one linear display range per channel; pixel size per field; D82, D86; "
          "perilymph CD45 acquired with the 561-nm line at 0 % but a real membrane signal, presence only, D93) and "
          "per-cell FUT4 and CD45 over local background (931): FUT4+ (>= 2) in 29/30 marrow, 23/23 perilymph and 56/64 "
          "endolymph cells; CD45+ in 29/30, 22/23 and 54/64",
          ["figures/panels/fut4_bone_marrow_f3_*.png", "figures/panels/fut4_perilymph_f8_*.png",
           "figures/panels/fut4_endolymph_f5_*.png", "image_panels_log.csv", "f5a_fut4_cells.csv",
           "f5a_cd45_channel_check.csv"], "M", "",
          "custom (914)")
    # 4e-i — parabiosis (D87, D132-D134, D142): the 54-day images (930) and FUT4 status of the lateral wall (931), the
    # survey of every wild-type partner with the upgraded body rule (960; perilymph crops from 935)
    pq = fd.read("r9_figures/f4_parabiont_fields.csv")
    wl = pq[(pq["field"] == "60x_sl2") & (pq["compartment"] == "cochlea")].iloc[0]
    assert int(wl["marker_cells"]) == 13 and int(wl["marker_gfp_cells"]) == 0
    rt = fd.read("r10_parabiosis/survey_cells_ratio.csv")
    vr = fd.read("r10_parabiosis/survey_cells_verified.csv").set_index("animal")
    pr = rt[~rt["noise"].astype(bool)].copy()                    # the noise-exposed pair is Supplementary Fig. 3
    for c_ in ("verified_ratio", "ci_lo", "ci_hi", "p_ratio_ge_1"):
        pr[c_] = pr["animal"].map(vr[c_])
    pr = pr.sort_values("days")[["animal", "days", "marrow_bodies", "marrow_cells", "marrow_density", "cochlea_bodies",
                                 "cochlea_cells", "cochlea_density", "cochlea_to_marrow", "cochlea_corrected_density",
                                 "corrected_cochlea_to_marrow", "marrow_best_plane_density", "cochlea_best_plane_density",
                                 "best_plane_cochlea_to_marrow", "verified_ratio", "ci_lo", "ci_hi", "p_ratio_ge_1"]]
    assert len(pr) == 3 and (pr[["cochlea_to_marrow", "corrected_cochlea_to_marrow",
                                 "best_plane_cochlea_to_marrow"]] < 1).all().all()
    f4dp = out(fd.prov(pr, "r10_parabiosis/survey_cells_ratio.csv; r10_parabiosis/survey_cells_verified.csv", "",
                       "wild-type partners with marrow and cochlea, noise-exposed pair excluded; upgraded body rule "
                       "(960); verified = label-corrected ratio with stratified bootstrap 95% CI"),
               "f4d_parabiosis_pairs.csv", "Fig. 4g: partner-derived (GFP+) cells per mm2 of temporal-bone marrow and "
               "cochlea per wild-type parabiont (one pair per pairing duration; upgraded body rule), observed, "
               "detection-corrected and brightest section, and the label-corrected cochlea / marrow ratio",
               ["animal", "marrow_density", "cochlea_density", "verified_ratio"])
    pf = fd.read("r10_parabiosis/parabiosis_perilymph_fields.csv")
    pf = pf[pf["animal"].isin(["14 d", "28 d"]) & pf["partner_label"].isin(["WT", "GFP"])]
    pf = pf.assign(acquisition=pf["path"].str.rsplit("/", n=1).str[-1]).rename(columns={"partner_label": "partner"})
    pf = pf[["animal", "partner", "acquisition", "nuclei", "gfp_pos", "gfp_pos_pct", "sensitivity_vs_gfp_partner"]]
    pool = pf.groupby(["animal", "partner"]).apply(lambda g: 100 * g["gfp_pos"].sum() / g["nuclei"].sum())
    assert (pool.xs("WT", level="partner") < 10).all() and (pool.xs("GFP", level="partner") > 50).all()
    f4dl = out(fd.prov(pf, "r10_parabiosis/parabiosis_perilymph_fields.csv", "", "14- and 28-day pairs"),
               "f4d_perilymph_fields.csv", "Fig. 4h: perilymph fields of wild-type and GFP partners (14 and 28 days), "
               "nucleated cells, GFP+ cells and their share", ["animal", "partner", "nuclei", "gfp_pos"])
    # 4i — partner-derived share by immune cell type in the 54-day wild-type partner: nucleus-anchored typing (961; the
    # 20x cochlear FUT4 calls are not used, 1 of 10 confirmed) and the 60x lateral-wall FUT4+ cells (931); verified =
    # typed cells confirmed as single marker+ cells in the blinded reviews (945 development + 961 validation; 944 for 60x)
    cs = fd.read("r10_parabiosis/celltype_v2_summary.csv")
    dv = fd.read("r10_parabiosis/celltype_v2_development.csv")
    va = fd.read("r10_parabiosis/verification_v2_agreement.csv")
    v944 = fd.read("r10_parabiosis/verification_items_labelled.csv")

    def verified(mk, comps):
        d1 = dv[(dv.rule == "v2") & (dv.question == "typed: one marker+ cell") & (dv.marker == mk) &
                dv.compartment.isin(comps)]
        d2 = va[(va.question == "typed: one marker+ cell") & (va.marker == mk) & va.compartment.isin(comps)]
        return int(d1.n_agree.sum() + d2.n_agree.sum()), int(d1.n_decided.sum() + d2.n_decided.sum())

    lw = v944[v944.stratum == "typed FUT4 cochlea (60x lateral wall, 931)"]
    rows = [dict(row="Cochlea, neutrophils (FUT4+; 60x lateral wall)", label="Neutrophils", compartment="cochlea",
                 marker="FUT4", typed_cells=int(wl["marker_cells"]), gfp_pos_cells=int(wl["marker_gfp_cells"]),
                 confirmed=int((lw.q1_cell == "yes").sum()), decided=int((lw.q1_cell != "unsure").sum()),
                 source="931 (60x)")]
    for lab, label, cp, mk, comps in [("Cochlea, leukocytes (CD45+)", "Leukocytes", "cochlea", "CD45", ["cochlea"]),
                                      ("Marrow, neutrophils (FUT4+)", "Neutrophils", "marrow", "FUT4", ["marrow"]),
                                      ("Marrow, monocytes (CD14+)", "Monocytes", "marrow", "CD14", ["marrow"]),
                                      ("Marrow, leukocytes (CD45+)", "Leukocytes", "marrow", "CD45",
                                       ["marrow", "vestibular bone"]),
                                      ("Marrow, B cells (CD19+; reporter-negative)", "B cells (control)", "marrow",
                                       "CD19", ["marrow"])]:
        g = cs[(cs.marker == mk) & cs.compartment.isin(comps)]
        k_, n_ = verified(mk, comps)
        rows.append(dict(row=lab, label=label, compartment=cp, marker=mk, typed_cells=int(g.typed_cells.sum()),
                         gfp_pos_cells=int(g.gfp_pos_cells.sum()), confirmed=k_, decided=n_, source="961"))
    fc = pd.DataFrame(rows)
    fc["gfp_pos_pct"] = 100 * fc.gfp_pos_cells / fc.typed_cells
    ci = [(0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))
          for k, n in zip(fc.gfp_pos_cells, fc.typed_cells)]
    fc["ci_lo_pct"], fc["ci_hi_pct"] = [100 * a for a, _ in ci], [100 * b for _, b in ci]
    fc["verified"] = [f"{k}/{n}" for k, n in zip(fc.confirmed, fc.decided)]
    assert int(fc.iloc[0].gfp_pos_cells) == 0 and (fc.typed_cells > 0).all()
    f4df = out(fd.prov(fc, "r10_parabiosis/celltype_v2_summary.csv; r10_parabiosis/celltype_v2_development.csv; "
                           "r10_parabiosis/verification_v2_agreement.csv; r10_parabiosis/verification_items_labelled.csv",
                       "", "nucleus-anchored typing (961); marrow leukocytes = temporal-bone and vestibular-bone marrow; "
                       "Clopper-Pearson 95% CI; verified = confirmed / decided in the blinded reviews"),
               "f4i_celltype_share.csv", "Fig. 4i: partner-derived (GFP+) share of immune cells of the 54-day "
               "wild-type partner by cell type and compartment (nucleus-anchored typing, 961; cochlear neutrophils from "
               "the 60x lateral wall, 931) with Clopper-Pearson 95% CI and the share of typed cells confirmed in the "
               "blinded review", ["row", "typed_cells", "gfp_pos_cells", "ci_hi_pct", "verified"])
    panel("4", "e", "Parabiosis design: wild-type C57BL/6J mice joined to Lyz2-EGFP partners, one pair per pairing "
          "duration (D133); in a wild-type partner GFP+ cells are partner-derived (entered from the shared circulation) and "
          "GFP- myeloid cells host-derived", [f4dp], "M", "", "custom (914)")
    panel("4", "f", "Raw images: 54-day marrow and cochlear lateral wall of the wild-type partner (GFP, FUT4, DAPI; "
          f"all {int(wl['marker_cells'])} lateral-wall FUT4+ cells GFP-negative, asterisks) and perilymph cells of the "
          "28-day wild-type and GFP partners (GFP, DAPI; GFP+ cells circled)",
          [pn + "parabiont_marrow_merge.png", pn + "parabiont_cochlea_merge.png", img, "f4_parabiont_objects.csv",
           "figures/panels/pb_s2b_3.png", "figures/panels/pb_s2b_4.png", "sfig_parabiosis_images.csv",
           "sfig_parabiosis_crop_objects.csv"], "M", "", "custom (914)")
    # 4g, 4i (D150-D153) — the ear against brain, liver and blood: every tissue counted in the brain image's format
    # (966; "I would like the number of cochleae to be comparable to the number of brains") by the strict rule of 968
    # ("the current quantitative criteria for the cochlea are too broad"; chosen on the blinded labels, applied to every
    # tissue): sections at 2.773 um per pixel, the brain from its projection; bars, mean of the pairs (D153: "Create a
    # combined bar and line chart"); published blood chimerism as the band of 4i
    # D155: detection on the local background (969; fixed before its run and chosen by the use rule recorded before it:
    # the better label agreement), with the strict call of 968 (contrast >= 6)
    lb = fd.read("r10_parabiosis/localbg_densities.csv")
    lb = lb[lb["main"] & lb["animal"].isin(pr["animal"]) & lb["compartment"].isin(["liver", "marrow", "cochlea", "brain"])]
    tg = lb.rename(columns={"cells": "cells_upgraded", "cells_strict": "cells_strict",
                            "cell_density_per_mm2": "density_upgraded_per_mm2",
                            "strict_density_per_mm2": "density_strict_per_mm2"})
    tg = tg.assign(conversion=np.where(tg["compartment"] == "brain", "projection", "section"))[
        ["animal", "days", "compartment", "conversion", "um_per_px", "fields", "tissue_mm2", "bodies", "cells_upgraded",
         "cells_strict", "density_upgraded_per_mm2", "density_strict_per_mm2"]].reset_index(drop=True)
    piv = tg.pivot_table(index="animal", columns="compartment", values="density_strict_per_mm2")
    tg["cochlea_to_marrow"] = tg["animal"].map(piv["cochlea"] / piv["marrow"])
    bar = tg.groupby("compartment")["density_strict_per_mm2"].agg(["mean", "size"])
    tg["tissue_mean_per_mm2"] = tg["compartment"].map(bar["mean"])
    tg["tissue_pairs"] = tg["compartment"].map(bar["size"])
    assert len(tg) == 8 and set(tg.loc[tg["animal"] == "31 d", "compartment"]) == {"liver", "marrow", "cochlea", "brain"}
    assert (piv["cochlea"] < piv["marrow"]).all() and (tg["density_strict_per_mm2"] > 0).all()
    f4dt = out(fd.prov(tg, "r10_parabiosis/localbg_densities.csv", "", "brain-image format, 2.773 um per pixel, local-"
                       "background detection (969), strict call (contrast >= 6, 968): sections for liver, marrow and "
                       "cochlea, projection for the brain; tissue mean over pairs"),
               "f4g_tissue_densities.csv", "Fig. 4g: partner-derived (GFP+) cells per mm2 of liver, marrow, cochlea and "
               "brain of the wild-type partners, every tissue counted in the brain image's format with local-background "
               "detection and the strict call (966, 968, 969), per pair and as the mean of the pairs",
               ["animal", "compartment", "density_strict_per_mm2", "tissue_mean_per_mm2", "cochlea_to_marrow"])
    bl = fd.read("r10_parabiosis/tissue_comparison_blood_external.csv")
    assert len(bl) == 2 and bl["partner_derived_pct"].between(20, 31).all()
    f4db = out(fd.prov(bl, "r10_parabiosis/tissue_comparison_blood_external.csv", "", "published values"),
               "f4i_blood_reference.csv", "Fig. 4i: partner-derived share of blood neutrophils (and Ly6C-high monocytes) "
               "of wild-type parabionts in published parabiosis (external, public; shaded band)",
               ["source", "doi", "pairing_days", "partner_derived_pct"])
    vv = pr.set_index("animal")
    tgm = tg[tg["animal"] == "31 d"].set_index("compartment")["density_strict_per_mm2"]
    panel("4", "g", f"Partner-derived cells per mm2 (upgraded body rule, 960) sparser in the cochlea than in the adjacent "
          f"marrow in all {len(pr)} pairs (ratio {pr['cochlea_to_marrow'].min():.2f}-{pr['cochlea_to_marrow'].max():.2f}; "
          f"detection-corrected {pr['corrected_cochlea_to_marrow'].min():.2f}-"
          f"{pr['corrected_cochlea_to_marrow'].max():.2f}; brightest section "
          f"{pr['best_plane_cochlea_to_marrow'].min():.2f}-{pr['best_plane_cochlea_to_marrow'].max():.2f}); "
          "label-corrected ratio " + "; ".join(f"{a} {vv.loc[a, 'verified_ratio']:.2f} ({vv.loc[a, 'ci_lo']:.2f}-"
                                              f"{vv.loc[a, 'ci_hi']:.2f})" for a in pr.animal)
          + " (D140: fewer in the cochlea at 14 and 54 days; the 31-day pair inconclusive); plotted in the brain "
          "image's format, local-background detection, strict call (D152, D153, D155): 31-day partner liver "
          + f"{tgm['liver']:.0f}, marrow "
          f"{tgm['marrow']:.0f}, cochlea {tgm['cochlea']:.0f} and brain {tgm['brain']:.1f} per mm2",
          [f4dp, f4dt], "M", "", "custom (914)")
    panel("4", "h", f"Perilymph GFP+ {pool.xs('WT', level='partner').min():.0f}-"
          f"{pool.xs('WT', level='partner').max():.0f}% in wild-type vs {pool.xs('GFP', level='partner').min():.0f}-"
          f"{pool.xs('GFP', level='partner').max():.0f}% in GFP partners (14 and 28 days)", [f4dl], "M", "",
          "custom (914)")
    nb = fc.set_index("row")
    panel("4", "i", "Partner-derived share by cell type, 54 days (nucleus-anchored typing): " + "; ".join(
        f"{r} {int(nb.loc[r, 'gfp_pos_cells'])}/{int(nb.loc[r, 'typed_cells'])} (verified {nb.loc[r, 'verified']})"
        for r in nb.index) + f"; band: blood neutrophils in published parabiosis {bl['partner_derived_pct'].min():.0f}-"
        f"{bl['partner_derived_pct'].max():.0f}% (D150)", [f4df, f4db], "M", "", "custom (914)")


def build_f5():
    """Fig. 5 (D134) — the cochlear immune compartment alternates over the day (former Fig. 4e-k) with an
    overview dial."""
    ct = fd.celltypes()
    immune = set(ct.loc[~ct["structural"], "cell_type"])
    reg = _abundance_register()

    # 4a — haematopoietic cells per sample: % of all tissue cells and % of the haematopoietic cells
    pts = fd.read("r2_figure/figure_points.csv")
    tot = pts.groupby(["tissue", "sample_id"])["pct"].sum()
    assert tot.between(99.5, 100 + 1e-6).all()                   # rare types absent from a tissue's table only
    a = pts[pts["cell_type"].isin(immune)].copy()
    a["pct_of_immune"] = 100 * a["n_cells"] / a.groupby(["tissue", "sample_id"])["n_cells"].transform("sum")
    a = fd.annotate(a)
    imm_c = a[a["tissue"] == "cochlea"].groupby("zt")["pct"].sum()
    f4a = out(fd.prov(a, "r2_figure/figure_points.csv", "A05; A04; A01; A03",
                      "haematopoietic cell types; % of all tissue cells and % of the haematopoietic cells per sample"),
              "f4a_immune_composition.csv", "Extended Data Fig. 1f (Fig. 4 v1 a): haematopoietic cell types per sample (% of all cells of the tissue "
              "and % of its haematopoietic cells)", ["tissue", "cell_type", "sample_id", "zt", "pct", "pct_of_immune"])
    cls = a[a["tissue"] == "cochlea"].pivot_table(index="zt", columns="cell_class", values="pct_of_immune", aggfunc="sum")
    my, ly = cls["Myeloid"], cls["Lymphoid"]
    # 4b — cyclic fits of the haematopoietic shares (OLS cosinor on % of tissue cells, 95 % band), register verdicts
    rh = fd.read("r2_figure/abundance_pct_rhythm.csv")
    mm = fd.read("r2_figure/abundance_multimodel_summary.csv")
    st = rh[rh["cell_type"].isin(immune)][["tissue", "cell_type", "mean_pct", "amplitude_pct", "ptt_ratio", "peak_zt",
                                           "peak_clock", "p_ols", "q_BH", "p_perm", "loto_phase_sd_h",
                                           "peak_ci_halfwidth_h"]]
    st = st.merge(mm[["tissue", "cell_type", "core_tested", "core_supporting", "verdict"]].rename(
        columns={"verdict": "multimodel_verdict"}), on=["tissue", "cell_type"], how="left")
    st = st.merge(reg, on=["tissue", "cell_type"], how="left")
    chk = st[st["register_id"].notna()]
    assert (np.abs(wrap12(chk["peak_zt"] - chk["register_peak_zt"])) < 0.06).all()   # register peaks = these fits
    st["flow_replicated"] = st["flow_replicated"].fillna(False).astype(bool)
    assert st.loc[st["flow_replicated"], "register_id"].tolist() and set(st.loc[st["flow_replicated"], "register_id"]) <= \
        {"A05", "A04", "A01"}, st.loc[st["flow_replicated"], "register_id"].tolist()
    cur = fd.read("r2_figure/figure_curves.csv")
    cur = cur[cur["cell_type"].isin(immune)].merge(st, on=["tissue", "cell_type"], how="left")
    assert cur["p_ols"].notna().all()
    f4b = out(fd.prov(fd.annotate(cur), "r2_figure/figure_curves.csv; r2_figure/abundance_pct_rhythm.csv; "
                      "r2_figure/abundance_multimodel_summary.csv; r2_audit/positive_findings_register.csv",
                      "A05; A04; A01; A03; A07; A09; A15; A21; A16", "fitted curve with 95 % band (0.1-h grid) joined "
                      "with the fit statistics and register verdicts"), "f4b_share_fits.csv",
              "Fig. 5b: cosinor fit (% of tissue cells, 95 % confidence band) of every haematopoietic share with its "
              "statistics, multi-model support and register verdict", ["tissue", "cell_type", "zt", "fit_pct", "band_lo",
                                                                        "band_hi", "peak_zt", "p_ols", "register_verdict"])
    panel("5", "b", "Cochlear band neutrophils (ZT5.2, robust) and classical monocytes (ZT7.0, supported) peak in the rest "
          "phase, GMP (ZT5.8) and aged neutrophils (ZT17.9) are suggestive and naive B (ZT17.1) is not supported; in marrow, "
          "naive B (ZT20.5, robust) and transitional B (ZT12.8, supported) shares are timed, mature neutrophils (ZT4.9) "
          "suggestive; diamond = replicated in the independent flow cohort (register line, D90; cochlear band neutrophils, "
          "classical monocytes, marrow naive B; D91)", [f4a, f4b], "A", "A05; A04; A15; A21; A16; A01; A03; A07",
          "custom (912)")

    # 4c — robustness: peak and support in the five core composition models, LOTO phase SD
    ml = fd.read("r2_figure/abundance_multimodel_long.csv")
    c = ml[ml["cell_type"].isin(immune) & ml["model"].isin(list(CORE_MODELS) + ["ols_pct"])].copy()
    assert not c.duplicated(["tissue", "cell_type", "model"]).any()
    c = c.merge(mm[["tissue", "cell_type", "core_tested", "core_supporting", "core_peak_circ_sd_h", "loto_phase_sd_h",
                    "clr_p_perm", "verdict"]].rename(columns={"verdict": "multimodel_verdict"}),
                on=["tissue", "cell_type"], how="left").merge(reg, on=["tissue", "cell_type"], how="left")
    c["p_support"] = np.where(c["model"] == "clr", c["clr_p_perm"], c["p"])      # CLR counts by its permutation p
    c["supports"] = np.where(c["model"] == "clr", c["p_support"] <= 0.05, c["p_support"] < 0.05)
    n_sup = c[c["model"].isin(list(CORE_MODELS))].groupby(["tissue", "cell_type"])["supports"].sum()
    assert (n_sup == c.groupby(["tissue", "cell_type"])["core_supporting"].first().reindex(n_sup.index)).all()
    c["model_label"] = c["model"].map({**CORE_MODELS, "ols_pct": "OLS % (figure model)"})
    f4c = out(fd.prov(fd.annotate(c), "r2_figure/abundance_multimodel_long.csv; r2_figure/abundance_multimodel_summary.csv",
                      "A05; A04; A01; A03", "five core composition models + the figure model; support = p < 0.05 "
                      "(CLR: permutation p <= 0.05); support counts asserted equal to the summary"),
              "f4c_composition_models.csv", "Fig. 5a: peak phase and support of each haematopoietic share in the five core "
              "composition models, with the leave-one-time-point-out phase SD", ["tissue", "cell_type", "model", "peak_zt",
                                                                                   "supports", "core_supporting"])
    # 4d — marrow content of the cochlear samples (sensitivity, D17): unadjusted vs marrow-adjusted phases
    mc = fd.read("52_marrow_decomposition/marrow_content_rhythm.csv")
    f4d = out(fd.prov(mc, "52_marrow_decomposition/marrow_content_rhythm.csv", "",
                      "per cochlear sample: marrow-derived vs structural cells"), "f4d_marrow_content.csv",
              "Extended Data Fig. 9g: marrow content of each cochlear sample (log2 marrow-derived / structural cells) and its cosinor test",
              ["sample_id", "zt", "log2_marrow_vs_structural", "rhythm_p"])
    adj = ml[(ml["tissue"] == "cochlea") & ml["cell_type"].isin(immune)
             & ml["model"].isin(["ols_pct", "alr_marrow", "marrow_adjusted"])]
    adj = adj.pivot_table(index="cell_type", columns="model", values=["p", "peak_zt", "stat"], aggfunc="first")
    adj.columns = [f"{v}_{m}" for v, m in adj.columns]
    adj = adj.reset_index()
    beta = fd.read("52_marrow_decomposition/tissue_residual_rhythm.csv")
    beta = beta[beta["model"] == "marrow_adjusted"][["cell_type", "beta_mean", "beta_lo", "beta_hi", "resid_ptt_fc_median",
                                                     "prob_fc_gt_1p5"]]
    adj = adj.merge(beta, on="cell_type", how="left").merge(reg[reg["tissue"] == "cochlea"].drop(columns="tissue"),
                                                             on="cell_type", how="left")
    adj["tissue"] = "cochlea"
    adj["shift_alr_h"] = wrap12(adj["peak_zt_alr_marrow"] - adj["peak_zt_ols_pct"])
    f4d2 = out(fd.prov(fd.annotate(adj), "r2_figure/abundance_multimodel_long.csv; "
                       "52_marrow_decomposition/tissue_residual_rhythm.csv", "A05; A04; A15; A21; A16",
                       "cochlear haematopoietic shares: unadjusted (OLS %, primary) vs marrow-adjusted (ALR against the "
                       "marrow-resident precursors HSPC, GMP, pro-B, pre-B, erythroblasts; Bayesian marrow-content model) "
                       "peak and p; coupling beta"),
               "f4d_marrow_adjusted.csv", "Extended Data Fig. 9g: cochlear haematopoietic shares before and after adjusting for the marrow "
               "content of the sample (sensitivity, D17)", ["cell_type", "peak_zt_ols_pct", "p_ols_pct",
                                                             "peak_zt_alr_marrow", "p_alr_marrow", "beta_mean"])
    # (the marrow-content sensitivity drawn from f4d / f4d2 is Extended Data Fig. 9g since D89; its row is in build_ed9)

    # 4f — independent flow cohort (D89; 932; pre-specified primaries P1-P3 of manuscript/flow_validation_plan.md)
    fs = fd.read("r9_figures/f4f_flow_shares.csv").drop_duplicates("readout").set_index("readout")
    assert set(fs.verdict) == {"replicated"}, fs.verdict.to_dict()
    panel("5", "c", "Independent flow cohort (D89; pre-specified, n = 9-10 mice per time): cochlear immature neutrophils "
          f"ZT17 / ZT5 {fs.loc['P1', 'ratio_zt17_zt5']:.2f} ({fs.loc['P1', 'ci_lo']:.2f}-{fs.loc['P1', 'ci_hi']:.2f}), "
          f"cochlear classical monocytes {fs.loc['P2', 'ratio_zt17_zt5']:.2f} ({fs.loc['P2', 'ci_lo']:.2f}-"
          f"{fs.loc['P2', 'ci_hi']:.2f}), marrow recirculating B {fs.loc['P3', 'ratio_zt17_zt5']:.2f} "
          f"({fs.loc['P3', 'ci_lo']:.2f}-{fs.loc['P3', 'ci_hi']:.2f}); Holm p <= {fs.p_holm.max():.4f}: all three "
          "primaries replicated; representative contour plots with outliers (D92)", ["f4f_flow_shares.csv", "flow_plots.csv", "flow_contours.csv", "flow_outliers.csv", "flow_gates.csv"], "A",
          "A05; A04; A01", "custom (912)")

    # 4e — label-free waves: Milo neighbourhood peak by pseudotime quintile (neutrophil and B lineages)
    rows = []
    for lin, f in [("neutrophil", "r6_myeloid/m_milo_wave.csv"), ("B", "r5_bcell/b_milo_wave.csv")]:
        w = fd.read(f)
        for _, r in w.iterrows():
            for q, pk, n in _quintiles(r["peak_by_pseudotime_quintile"]):
                rows.append({"lineage": lin, "tissue": r["tissue"], "selection": r["selection"],
                             "selection_free": not r["selection"].startswith("nominal"), "quintile": q, "peak_zt": pk,
                             "n_nhoods_quintile": n, "n_nhoods": r["n_nhoods"], "circ_linear_r": r["circ_linear_r"],
                             "p_perm": r["p_perm"], "source_file": f})
    wv = pd.DataFrame(rows)
    f4e = out(fd.prov(wv, "r6_myeloid/m_milo_wave.csv; r5_bcell/b_milo_wave.csv", "G08; M03; B02",
                      "quintile peaks parsed from 'peak_by_pseudotime_quintile'"), "f4e_milo_waves.csv",
              "Fig. 5d: circular mean peak of the Milo neighbourhoods by pseudotime quintile (neutrophil and B lineages, "
              "both tissues) with the circular-linear correlation", ["lineage", "tissue", "selection", "quintile", "peak_zt",
                                                                      "circ_linear_r", "p_perm"])
    sf = wv[wv["selection_free"]].drop_duplicates(["lineage", "tissue"]).set_index(["lineage", "tissue"])
    panel("5", "d", "Label-free waves: along pseudotime, neighbourhoods peak progressively later in the neutrophil lineage "
          f"(marrow r {sf.loc[('neutrophil', 'bm'), 'circ_linear_r']:.2f}, cochlea "
          f"{sf.loc[('neutrophil', 'cochlea'), 'circ_linear_r']:.2f}) and the marrow B lineage "
          f"(r {sf.loc[('B', 'bm'), 'circ_linear_r']:.2f}; all neighbourhoods, permutation p <= "
          f"{sf.loc[[('neutrophil', 'bm'), ('neutrophil', 'cochlea'), ('B', 'bm')], 'p_perm'].max():.4f}); granulopoiesis "
          "and B-development order tests remain suggestive (M03, B02)", [f4e], "A", "G08; M03; B02", "custom (912)")

    # 5a — timetable (D134): peak, above-mesor window and model robustness of every timed haematopoietic share of 5b
    dial_types = {"cochlea": ["Neutrophil (Band)", "Monocyte (Classical)", "Granulocyte-Monocyte Progenitor",
                              "Neutrophil (Mature)", "Neutrophil (Aged)", "B Cell (Naïve)"],
                  "bm": ["B Cell (Naïve)", "B Cell (Transitional)", "Pro-B Cell (Cycling)", "HSC / MPP",
                         "Granulocyte-Monocyte Progenitor", "Neutrophil (Mature)"]}
    dl = pd.concat([st[(st["tissue"] == t) & st["cell_type"].isin(v)] for t, v in dial_types.items()], ignore_index=True)
    assert len(dl) == 12, len(dl)
    dl["verdict"] = dl["register_verdict"].fillna(dl["multimodel_verdict"])
    dl = fd.annotate(dl)
    dl = dl[["tissue", "cell_type", "abbr", "cell_class", "peak_zt", "peak_clock", "amplitude_pct", "mean_pct", "p_ols",
             "verdict", "flow_replicated"]]
    f5d = out(fd.prov(dl, "r2_figure/abundance_pct_rhythm.csv; r2_figure/abundance_multimodel_summary.csv; "
                      "r2_audit/positive_findings_register.csv", "A05; A04; A01; A03",
                      "the twelve shares of Fig. 5b: peak, class and verdict"), "f5a_share_peaks.csv",
              "Fig. 5a: peak time of each timed haematopoietic share (cochlea, marrow) with its cell class and verdict "
              "(above-mesor windows from f4b_share_fits, model peaks from f4c_composition_models)",
              ["tissue", "cell_type", "peak_zt", "verdict"])
    sup = dl[dl["verdict"].isin(["robust", "supported"])]
    rob = mm[(mm["verdict"] == "robust") & mm["cell_type"].isin(immune)]
    panel("5", "a", "Day-night timetable: in the cochlea the robust / supported myeloid shares peak in the rest phase "
          f"(ZT{sup[(sup.tissue == 'cochlea') & (sup.cell_class == 'Myeloid')].peak_zt.min():.1f}-"
          f"{sup[(sup.tissue == 'cochlea') & (sup.cell_class == 'Myeloid')].peak_zt.max():.1f}); in the marrow the "
          f"robust / supported B shares peak at ZT{sup[(sup.tissue == 'bm')].peak_zt.min():.1f}-"
          f"{sup[(sup.tissue == 'bm')].peak_zt.max():.1f}; robust shares keep their phase in every core composition model "
          f"(circular SD of the core peaks <= {rob['core_peak_circ_sd_h'].max():.1f} h) and when any time point is left "
          f"out (LOTO phase SD <= {rob['loto_phase_sd_h'].max():.1f} h)", [f5d, f4c, f4b], "A", "A05; A04; A01; A03; A07",
          "custom (912)")

    # 4g — phase-ordered Tier-1 genes of aged / activated marrow neutrophils (903 exports the per-sample expression)
    t33 = fd.read("33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "a", "b", "peak_zt"])
    g1 = t33[t33["group"].isin(["Neutrophil (Aged)", "Neutrophil (Activated)"]) & (t33["tier"] == "tier1")]
    night = (g1["peak_zt"] >= 12).groupby(g1["group"]).mean()
    panel("5", "e", "Tier-1 genes of aged (649) and activated (465) marrow neutrophils ordered by peak phase: "
          f"{100 * night['Neutrophil (Aged)']:.0f} % and {100 * night['Neutrophil (Activated)']:.0f} % peak in the dark "
          "phase (per-sample z of voom log2 CPM; refit asserted equal to the stored fits)",
          ["f4g_neutrophil_state_genes.csv"], "R", "G01", "custom (903 R extraction)")

    # 4h — sorted marrow neutrophils (GSE312298): our predicted vs observed night - day, per gene (re-derived, asserted)
    t33 = fd.read("33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv", usecols=["group", "gene", "tier", "a", "b"])
    neu = t33[t33["group"].isin(NEU_STATES)]
    neu_t1 = neu[neu["tier"] == "tier1"].groupby("gene")[["a", "b"]].mean()
    states = neu[neu["tier"] == "tier1"].groupby("gene")["group"].apply(lambda s: "; ".join(sorted(s)))
    off = fd.read("r2_phase/clock_alignment_offsets.csv")
    d_bm = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "bm") & (off["unit"] == "tissue_pool")]
                 ["offset_canonical_h"].iloc[0])
    de = fd.read("r3_response/gse312298_de.csv")
    obs = de[(de["tissue"] == "BM") & (de["contrast"] == "night_minus_day_PBS")].set_index("gene")
    g = neu_t1.index.intersection(obs.index)
    td, tn = 14.0 + d_bm, 22.0 + d_bm                                   # design collection ZT14 vs ZT22 (internal time)
    pred = (neu_t1.loc[g, "a"] * (np.cos(W * tn) - np.cos(W * td)) + neu_t1.loc[g, "b"] * (np.sin(W * tn) - np.sin(W * td)))
    r = float(np.corrcoef(pred, obs.loc[g, "log2FC"])[0, 1])
    val = fd.read("r3_response/gse312298_validation.csv")
    ref = val[(val["collection_time"] == "design (ZT14 vs ZT22)")
              & val["predictor"].str.startswith("our BM neutrophil-state Tier 1")].iloc[0]
    assert len(g) == ref["n_genes"] and abs(r - ref["pearson_r"]) < 1e-9, (len(g), r)
    hh = pd.DataFrame({"gene": g, "predicted_night_minus_day": pred.to_numpy(), "observed_log2FC": obs.loc[g, "log2FC"].to_numpy(),
                       "observed_p": obs.loc[g, "p"].to_numpy(), "observed_q_BH": obs.loc[g, "q_BH"].to_numpy(),
                       "tier1_states": states.reindex(g).to_numpy(), "pearson_r": r, "p_perm": ref["p_perm"],
                       "sign_agreement": ref["sign_agreement"]})
    comp = val[(val["collection_time"] == "design (ZT14 vs ZT22)") & val["predictor"].str.startswith("GSE198770")].iloc[0]
    hh["comparator_r_GSE198770"] = comp["pearson_r"]
    f4h = out(fd.prov(hh, "33_rhythm_tiers/rhythm_genes_all_cell_type_bm.csv; r3_response/gse312298_de.csv; "
                      "r2_phase/clock_alignment_offsets.csv", "R01", "mean neutrophil-state Tier-1 fit per gene -> predicted "
                      "ZT22 - ZT14 on the marrow clock axis (315 formula); r asserted equal to gse312298_validation.csv"),
              "f4h_gse312298_genes.csv", "Fig. 5f: per gene, predicted (our marrow neutrophil-state fits) vs observed night - "
              "day difference in sorted marrow neutrophils of an independent study (GSE312298, PBS)",
              ["gene", "predicted_night_minus_day", "observed_log2FC", "pearson_r"])
    panel("5", "f", f"Our marrow neutrophil-state rhythms predict the night - day difference of independently sorted marrow "
          f"neutrophils (GSE312298, PBS, ZT22 vs ZT14): r {r:.2f} over {len(g)} Tier-1 genes (gene-permutation p "
          f"{ref['p_perm']:.4f}; bulk-marrow comparator r {comp['pearson_r']:.2f})", [f4h], "R", "R01", "custom (912)")

    # 4i — baboon spleen and marrow: phase concordance of our marrow gene rhythms (clock-aligned), non-clock genes
    bb = fd.read("r4_crossspecies/baboon_bm_gene_concordance.csv")
    bb = bb[bb["genes"] == "non_clock"].copy()
    bb["scope_type"] = np.where(bb["our_scope"].isin(immune | set(ct["cell_type"])), "cell type", "summary")
    bb = bb.merge(ct[["cell_type", "abbr", "cell_class", "colour", "order"]], left_on="our_scope", right_on="cell_type",
                  how="left")
    f4i = out(fd.prov(bb, "r4_crossspecies/baboon_bm_gene_concordance.csv", "R13", "non-clock genes; summary and "
                      "per-cell-type rows"), "f4i_baboon_concordance.csv", "Fig. 5g: share of our marrow rhythm genes whose "
                      "clock-aligned peak lies within 3 h of the baboon peak (spleen; marrow with an imputed clock offset) vs "
                      "the permutation null", ["baboon_tissue", "our_scope", "n_genes", "frac_within_3h",
                                                "null_frac_within_3h", "p_perm"])
    s = bb[bb["our_scope"] == "ALL (Tier 1/2)"].set_index("baboon_tissue")
    panel("5", "g", f"Suggestive primate conservation: {100 * s.loc['SPL', 'frac_within_3h']:.0f} % of our marrow Tier-1/2 "
          f"genes peak within 3 h of baboon spleen (null {100 * s.loc['SPL', 'null_frac_within_3h']:.0f} %, p "
          f"{s.loc['SPL', 'p_perm']:.3f}) and {100 * s.loc['BOM', 'frac_within_3h']:.0f} % of baboon marrow (imputed clock "
          f"offset; null {100 * s.loc['BOM', 'null_frac_within_3h']:.0f} %, p {s.loc['BOM', 'p_perm']:.3f})", [f4i], "A",
          "R13", "custom (912)")


# ---------------------------------------------------------------- Fig. 6 — neutrophil timer split (D68; Fig. 5 until D134)
TIMER_STATES = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)",
                "Granulocyte-Monocyte Progenitor"]
KEY_STATES = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Mature)"]


def build_f6():
    from scipy import stats
    gt = fd.read("77_neutrophil_timer/neutrophil_timer_gene_table.csv")
    cc = fd.read("77_neutrophil_timer/neutrophil_timer_concordance.csv")
    blood_q = gt.drop_duplicates("gene").set_index("gene")["wt_zt13_vs_zt5_q"]

    # 5b — gene-level concordance of aged / activated neutrophils with the blood ZT13 - ZT5 programme (asserted)
    rows = []
    for t in ["cochlea", "bm"]:
        our = fd.read(f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv", usecols=["group", "gene", "p", "tier"])
        for ct in ["Neutrophil (Aged)", "Neutrophil (Activated)"]:
            col = f"pred_zt13_minus_zt5_{t}_{ct}"
            d = gt[["gene", "wt_zt13_vs_zt5_log2fc", "wt_zt13_vs_zt5_p", col]].dropna().rename(
                columns={col: "predicted_zt13_minus_zt5", "wt_zt13_vs_zt5_log2fc": "blood_zt13_minus_zt5_log2fc",
                         "wt_zt13_vs_zt5_p": "blood_p"})
            o = our[our["group"] == ct].set_index("gene")
            d["blood_q"] = d["gene"].map(blood_q)
            d["our_p"], d["our_tier"] = d["gene"].map(o["p"]), d["gene"].map(o["tier"])
            d["both_p05"] = (d["blood_p"] < 0.05) & (d["our_p"] < 0.05)
            r_all = stats.spearmanr(d["blood_zt13_minus_zt5_log2fc"], d["predicted_zt13_minus_zt5"])[0]
            b = d[d["both_p05"]]
            r_b = stats.spearmanr(b["blood_zt13_minus_zt5_log2fc"], b["predicted_zt13_minus_zt5"])[0]
            ref = cc[(cc["tissue"] == t) & (cc["cell_type"] == ct)].iloc[0]
            assert len(d) == ref["n_shared_genes"] and abs(r_all - ref["spearman_all"]) < 1e-9, (t, ct)
            assert len(b) == ref["n_both_p05"] and abs(r_b - ref["spearman_both_p05"]) < 1e-9, (t, ct)
            d["tissue"], d["cell_type"] = t, ct
            d["spearman_all"], d["spearman_both_p05"], d["p_perm_all"] = r_all, r_b, ref["p_perm_all"]
            rows.append(d)
    gb = fd.annotate(pd.concat(rows, ignore_index=True))
    gb["g02_flow_supported"] = "G02" in _flow_supported()           # diamond on the Fig. 6a titles (D90, D91)
    f5b = out(fd.prov(gb, "77_neutrophil_timer/neutrophil_timer_gene_table.csv; 33_rhythm_tiers/rhythm_genes_all_cell_type_"
                      "{cochlea,bm}.csv", "G02", "per gene: blood ZT13 - ZT5 (GSE86619) vs our predicted ZT13 - ZT5; "
                      "Spearman rho (all genes; genes p < 0.05 in both) asserted equal to neutrophil_timer_concordance.csv"),
              "f5b_timer_genes.csv", "Fig. 6a: per gene, the blood-neutrophil day-night change (GSE86619, ZT13 - ZT5) vs "
              "the change predicted by our aged / activated neutrophil fits, cochlea and marrow",
              ["tissue", "cell_type", "gene", "blood_zt13_minus_zt5_log2fc", "predicted_zt13_minus_zt5", "both_p05"])
    k = cc.set_index(["tissue", "cell_type"])
    ca, ba = k.loc[("cochlea", "Neutrophil (Aged)")], k.loc[("bm", "Neutrophil (Aged)")]
    panel("6", "a", f"Gene by gene, cochlear aged neutrophils predict the blood-neutrophil ZT13 - ZT5 programme (rho "
          f"{ca['spearman_all']:.2f} over {int(ca['n_shared_genes']):,} genes; {ca['spearman_both_p05']:.2f} for the "
          f"{int(ca['n_both_p05'])} genes p < 0.05 in both) and marrow aged neutrophils its inverse (rho "
          f"{ba['spearman_all']:.2f}; {ba['spearman_both_p05']:.2f} for {int(ba['n_both_p05'])} genes); diamond = the split "
          "is supported by the independent flow cohort (CXCR4, register line, D90; D91)", [f5b], "R", "G02", "custom (913)")

    # 5c — concordance per neutrophil state and tissue, with the exact time-relabelling p of each single correlation
    v = fd.read("r6_myeloid/m_timer_split_verification.csv")
    ab = v[v["part"].str.startswith("A/B")]
    ex = ab[ab["gene_set"] == "all shared genes"].set_index(["tissue", "unit"])["p_exact_two_sided"]
    c5 = cc.copy()
    c5["p_exact_two_sided"] = [ex.get((t, ct), np.nan) for t, ct in zip(c5["tissue"], c5["cell_type"])]
    f5c = out(fd.prov(fd.annotate(c5), "77_neutrophil_timer/neutrophil_timer_concordance.csv; "
                      "r6_myeloid/m_timer_split_verification.csv", "G02; G53", "select; exact time-relabelling p joined"),
              "f5c_timer_concordance.csv", "Fig. 6b: blood-timer concordance per neutrophil state and tissue (all genes, "
              "genes p < 0.05 in both, without stress genes) with gene-permutation and exact time-relabelling p",
              ["tissue", "cell_type", "spearman_all", "spearman_both_p05", "spearman_all_no_stress", "p_perm_all"])
    key = c5[c5["cell_type"].isin(KEY_STATES)]
    panel("6", "b", "Positive in cochlear aged / activated / mature neutrophils and negative in marrow aged / activated / "
          "mature neutrophils and GMP; band neutrophils null in both tissues; unchanged without stress / immediate-early "
          "genes. Single correlations "
          f"(n = 6) reach exact time-relabelling p {key['p_exact_two_sided'].min():.3f} at best (G53), so specificity (d) "
          "and external data (f-h) carry the claim", [f5c], "A", "G02; G53", "custom (913)")

    # 5d — specificity: rank of every cell type's correlation within its tissue; exact rank-sum p (re-derived, asserted)
    vm = fd.read("r2_audit/validation_matrix.csv")
    evd = vm[(vm["finding_id"] == "G02") & (vm["dimension"] == "sample_level_pattern")]["evidence"].iloc[0]
    a_ = ab[(ab["gene_set"] == "all shared genes") & (ab["unit"] != "pooled neutrophils")].copy()
    sp_rows = []
    for ts, sg in [("bm", -1), ("cochlea", 1)]:
        x_ = a_[a_["tissue"] == ts].copy()
        x_["score"] = sg * x_["spearman"]
        x_["rank"] = x_["score"].rank(ascending=False)
        rk = x_.set_index("unit").loc[KEY_STATES, "rank"]
        p_ = float(np.mean([sum(cmb) <= rk.sum() for cmb in itertools.combinations(range(1, len(x_) + 1), 3)]))
        m = re.search(ts + r": ranks ([\d, ]+) of (\d+) types \(rank-sum exact p ([\d.]+)\)", evd)
        assert m and [int(s) for s in m.group(1).split(",")] == [int(r) for r in rk] and int(m.group(2)) == len(x_)
        assert abs(p_ - float(m.group(3))) < 5e-5, (ts, p_, m.group(3))
        x_["rank_sum_exact_p"], x_["key_state"], x_["expected_sign"] = p_, x_["unit"].isin(KEY_STATES), sg
        sp_rows.append(x_)
    sp = pd.concat(sp_rows, ignore_index=True).rename(columns={"unit": "cell_type"})
    f5d = out(fd.prov(fd.annotate(sp), "r6_myeloid/m_timer_split_verification.csv; r2_audit/validation_matrix.csv", "G02; G53",
                      "rank of every cell type's correlation within its tissue (expected sign); exact rank-sum p of the "
                      "three key states over all 3-subsets, asserted equal to 241"), "f5d_specificity_rank.csv",
              "Fig. 6c: blood-timer correlation of every cell type, ranked within its tissue in the expected direction",
              ["tissue", "cell_type", "spearman", "rank", "key_state", "rank_sum_exact_p"])
    pr = sp.groupby("tissue")["rank_sum_exact_p"].first()
    rks = sp[sp["key_state"]].groupby("tissue")["rank"].apply(lambda s: ", ".join(str(int(v)) for v in sorted(s)))
    nn = sp.groupby("tissue").size()
    panel("6", "c", f"Specific to neutrophils: the key states are the most extreme cell types of their tissue in the expected "
          f"direction (cochlea ranks {rks['cochlea']} of {nn['cochlea']}, exact rank-sum p {pr['cochlea']:.4f}; marrow ranks "
          f"{rks['bm']} of {nn['bm']}, p {pr['bm']:.3f}), which a sample-wide pattern cannot produce", [f5d], "R",
          "G02; G53", "custom (913)")

    # 5e — the blood-timer score of pooled neutrophils across the day (OLS cosinor refit asserted equal to 640)
    axs = fd.read("r6_myeloid/m_axes_by_sample.csv")
    axr = fd.read("r6_myeloid/m_axes_rhythm.csv")
    e = axs[axs["axis"].str.startswith("blood timer") & (axs["group"] == "pooled neutrophils")].copy()
    curves = []
    for t, d in e.groupby("tissue"):
        d = d.sort_values("zt")
        assert len(d) == 6
        X = np.column_stack([np.ones(6), np.cos(W * d["zt"]), np.sin(W * d["zt"])])
        mes, ca_, sb_ = np.linalg.lstsq(X, d["median_score"].to_numpy(), rcond=None)[0]
        ref = axr[(axr["tissue"] == t) & axr["axis"].str.startswith("blood timer") & (axr["group"] == "pooled neutrophils")
                  & (axr["subset"] == "all cells")].iloc[0]
        assert abs(np.hypot(ca_, sb_) - ref["amplitude"]) < 1e-9 and abs(wrap12((np.arctan2(sb_, ca_) / W) % 24 - ref["peak_zt"])) < 1e-6
        g = np.round(np.arange(0, 24.01, 0.1), 1)
        curves.append(pd.DataFrame({"tissue": t, "zt": g, "fit_score": mes + ca_ * np.cos(W * g) + sb_ * np.sin(W * g),
                                    "mesor": mes, "peak_zt": ref["peak_zt"], "amplitude": ref["amplitude"], "p": ref["p"],
                                    "p_perm": ref["p_perm"], "loto_phase_sd_h": ref["loto_phase_sd_h"]}))
    ps = fd.read("r6_myeloid/m_paired_summary.csv")
    paired = ps[ps["quantity"] == "blood timer ZT13−ZT5 (GSE86619) | pooled neutrophils"].iloc[0]
    cur = pd.concat(curves, ignore_index=True)
    cur["paired_difference_p_perm"] = paired["difference_p_perm"]
    f5e = out(fd.prov(e, "r6_myeloid/m_axes_by_sample.csv", "M02", "blood-timer score, pooled neutrophils, per sample"),
              "f5e_timer_score.csv", "Fig. 6d: blood-neutrophil timer score (GSE86619 ZT13 - ZT5 signature) of pooled "
              "neutrophils per sample, cochlea and marrow", ["tissue", "sample_id", "zt", "median_score"])
    f5e2 = out(fd.prov(cur, "r6_myeloid/m_axes_rhythm.csv; r6_myeloid/m_paired_summary.csv", "M02",
                       "OLS cosinor refit of the six samples on a 0.1-h grid, asserted equal to the stored amplitude / peak"),
               "f5e_timer_score_fit.csv", "Fig. 6d: cosinor fit of the timer score with peak, permutation p and the paired "
               "cochlea - marrow difference test", ["tissue", "zt", "fit_score", "peak_zt", "p_perm"])
    pk = cur.groupby("tissue")[["peak_zt", "p_perm"]].first()
    panel("6", "d", f"Opposite halves of the day: the blood-timer score of marrow neutrophils peaks at ZT{pk.loc['bm', 'peak_zt']:.1f} "
          f"(permutation p {pk.loc['bm', 'p_perm']:.3f}) and of cochlear neutrophils at ZT{pk.loc['cochlea', 'peak_zt']:.1f} "
          f"(p {pk.loc['cochlea', 'p_perm']:.3f}); paired difference p {paired['difference_p_perm']:.3f} (M02 suggestive)",
          [f5e, f5e2], "R", "M02", "custom (913)")

    # 5f — neutrophil-enriched vs non-enriched genes
    fb = ab[ab["unit_class"] == "neutrophil lineage"].rename(columns={"unit": "cell_type"})
    fb = fb[fb["cell_type"] != "pooled neutrophils"]
    f5f = out(fd.prov(fd.annotate(fb), "r6_myeloid/m_timer_split_verification.csv", "G02",
                      "part A/B: all shared, neutrophil-enriched and non-enriched genes"), "f5f_enriched_genes.csv",
              "Fig. 6e: blood-timer correlation of each neutrophil state computed on neutrophil-enriched and non-enriched "
              "genes", ["tissue", "cell_type", "gene_set", "n_genes", "spearman", "p_exact_two_sided"])
    ne = fb[fb["gene_set"] == "non-enriched genes"].set_index(["tissue", "cell_type"])["spearman"]
    panel("6", "e", "Not neutrophil identity genes: the split holds in the non-enriched genes (aged neutrophils rho "
          f"{ne[('cochlea', 'Neutrophil (Aged)')]:.2f} cochlea, {ne[('bm', 'Neutrophil (Aged)')]:.2f} marrow) and is weaker in "
          "neutrophil-enriched genes", [f5f], "A", "G02", "custom (913)")

    # 5g — independent long-bone marrow with / without neutrophils (GSE198770)
    gd = v[v["part"].str.startswith("D ")].copy()
    gd["condition"] = gd["unit"].str.replace("whole marrow, ", "", regex=False)
    f5g = out(fd.prov(gd, "r6_myeloid/m_timer_split_verification.csv", "G02", "part D (GSE198770 long-bone marrow)"),
              "f5g_depletion.csv", "Fig. 6f: blood-timer correlation of the rhythms of independent long-bone marrow, control "
              "vs neutrophil-depleted (GSE198770)", ["condition", "gene_set", "n_genes", "spearman", "p_exact_two_sided"])
    gx = gd.set_index(["condition", "gene_set"])["spearman"]
    panel("6", "f", f"Neutrophil-dependent: in independent long-bone marrow the rhythmic genes carry the marrow sign "
          f"(rho {gx[('control', 'rhythmic genes (p < 0.05)')]:.2f}), lost after neutrophil depletion "
          f"({gx[('neutrophil_depleted', 'rhythmic genes (p < 0.05)')]:.2f})", [f5g], "A", "G02", "custom (913)")

    # 5h — human blood neutrophils (14:00 vs 08:00), bulk cochlea (GSE88954) and the blood scRNA check of the score
    hu = fd.read("r4_crossspecies/human_neutrophil_timeofday.csv")
    hu = hu[hu["primary"]].copy()
    rep = fd.read("r2_audit/neutrophil_timer_replication.csv")
    rep = rep[(rep["dataset"] == "GSE88954")].copy()
    chk = fd.read("r6_myeloid/m_m2_external_checks.csv")
    bal = chk[chk["check"].str.startswith("Ballesteros")].iloc[0]
    hrows = [{"source": "human blood (14:00 vs 08:00)", "tissue": r.tissue, "unit": r.cell_type, "statistic": "pearson_r",
              "value": r.pearson_r, "p": r.p_perm, "p_note": "one-sided permutation (positive)", "n_genes": r.n_genes}
             for r in hu.itertuples()]
    hrows += [{"source": f"bulk cochlea GSE88954 ({r.condition})", "tissue": "cochlea", "unit": r.genes, "statistic":
               "spearman_rho", "value": r.spearman_rho, "p": r.p, "p_note": "gene-level", "n_genes": r.n_genes}
              for r in rep.itertuples()]
    hrows.append({"source": "blood scRNA (Ballesteros): timer score ZT13 > ZT5", "tissue": "blood", "unit": "neutrophils",
                  "statistic": "score difference", "value": bal["value"], "p": bal["p"], "p_note": "Mann-Whitney, cells",
                  "n_genes": np.nan})
    hh = pd.DataFrame(hrows)
    f5h = out(fd.prov(hh, "r4_crossspecies/human_neutrophil_timeofday.csv; r2_audit/neutrophil_timer_replication.csv; "
                      "r6_myeloid/m_m2_external_checks.csv", "R14; G02", "primary human mapping; GSE88954 rows; blood "
                      "scRNA check"), "f5h_external_lines.csv", "Fig. 6g: the split in human blood neutrophils (our "
                      "fits vs 14:00 - 08:00), bulk cochlea (GSE88954) and the blood scRNA check of the timer score",
              ["source", "tissue", "unit", "value", "p"])
    hx = hu.set_index(["tissue", "cell_type"])["pearson_r"]
    g88 = rep[(rep["condition"] == "sham") & (rep["genes"] == "neutrophil_markers")].iloc[0]
    panel("6", "g", f"Holds in human blood: cochlear neutrophil fits predict the 14:00 - 08:00 change positively (all states r "
          f"{hx[('cochlea', 'all neutrophil states')]:.2f}) and marrow fits negatively ({hx[('bm', 'all neutrophil states')]:.2f}); "
          f"bulk cochlea keeps the cochlear sign (GSE88954 sham, rho {g88['spearman_rho']:.2f})", [f5h], "A", "R14; G02",
          "custom (913)")

    # 5i (exploratory) — cochlear neutrophil ageing cycle: Cxcr4 / Cxcr2 ratio, Sell and the timer score
    nr = fd.read("r6_myeloid/m_niche_ratio_rhythm.csv")
    nr = nr[(nr["subset"] == "all cells") & nr["tissue"].isin(["bm", "cochlea"])].copy()
    tm = cur.groupby("tissue")[["peak_zt", "amplitude", "p", "p_perm"]].first().reset_index()
    tm["score"] = "blood timer"
    ii = pd.concat([nr[["tissue", "score", "peak_zt", "amplitude", "p", "p_perm", "loto_phase_sd_h"]], tm],
                   ignore_index=True)
    f5i = out(fd.prov(ii, "r6_myeloid/m_niche_ratio_rhythm.csv; r6_myeloid/m_axes_rhythm.csv", "M01; M06; M02",
                      "EXPLORATORY: per-tissue peak of the Cxcr4 / Cxcr2 ratio, Sell (CD62L) and the blood-timer score"),
              "f5i_ageing_cycle.csv", "Fig. 6h (exploratory): peak phase of the neutrophil ageing markers and the timer "
              "score in each tissue", ["tissue", "score", "peak_zt", "p", "p_perm"])
    # (the exploratory ageing cycle drawn from f5i is Extended Data Fig. 9f since D89; its row is in build_ed9)

    # 5i — independent flow cohort (D89; 932; pre-specified secondaries S1-S2)
    mk = fd.read("r9_figures/f5i_flow_markers.csv").drop_duplicates("readout").set_index("readout")
    panel("6", "h", "Independent flow cohort (D89; pre-specified): the CXCR4-hi share of cochlear neutrophils falls at night "
          f"(odds ratio ZT17 / ZT5 {mk.loc['S2_pct', 'cochlea_effect']:.2f}, p {mk.loc['S2_pct', 'cochlea_p']:.1g}; marrow "
          f"{mk.loc['S2_pct', 'marrow_effect']:.2f}, p {mk.loc['S2_pct', 'marrow_p']:.2f}; interaction p "
          f"{mk.loc['S2_pct', 'interaction_p']:.1g}): S2 replicated; the CD62L-hi share does not change (cochlea "
          f"{mk.loc['S1_pct', 'cochlea_effect']:.2f}, p {mk.loc['S1_pct', 'cochlea_p']:.2f}; interaction p "
          f"{mk.loc['S1_pct', 'interaction_p']:.2f}): S1 not replicated",
          ["f5i_flow_markers.csv", "flow_plots.csv", "flow_contours.csv", "flow_outliers.csv", "flow_gates.csv"], "A", "G02", "custom (913)")

# ---------------------------------------------------------------- Fig. 6 — independent cohort (Ma 2025; D68)
MA_GROUP_LABEL = {"control": "scRNA control", "day_noise": "scRNA day noise", "night_noise": "scRNA night noise",
                  "CMD": "bulk control, day", "CMN": "bulk control, night", "NMD": "bulk day noise",
                  "NMN": "bulk night noise"}
CLAIM_SHORT = {
    "R08 / headline 2: cochlear immune composition alternates with time of day": "Immune composition",
    "A05 band neutrophils peak in the rest phase (cochlea)": "Band neutrophils, rest phase",
    "A04 classical monocytes peak in the rest phase (cochlea)": "Classical monocytes, rest phase",
    "H1 naïve-B timing (cochlea)": "Naïve-B timing",
    "headline 2 (B part): B lineage more abundant in the active phase (cochlea)": "B lineage, active phase",
    "M3 granulopoiesis timing (cochlea; GMP / band shares and maturation)": "Granulopoiesis timing",
    "M1 maturation / ageing timing (cochlea)": "Maturation / ageing",
    "M2 cochlear circulating-like state (blood timer)": "Blood-timer state",
    "M5 monocyte → macrophage (cochlea)": "Monocyte → macrophage",
    "M6 retention / release axis (cochlea)": "Retention / release",
    "M8 cochlear macrophage identity": "Macrophage identity",
    "M4 functional readiness (cochlea)": "Functional readiness",
    "B-lineage programme rhythms (round-5 exploratory)": "B programme rhythms",
    "M7 myeloid clocks (cochlea)": "Myeloid clocks",
    "H4 B-clock shift (cochlea)": "B-clock shift",
    "M9 noise-exposure readiness vs time of day (cochlea)": "Noise readiness × time"}


def build_f7():
    from scipy import stats
    from lib import phase
    off = fd.read("r2_phase/clock_alignment_offsets.csv")
    o_c = float(off[(off["dataset"] == "this_atlas") & (off["tissue"] == "cochlea") & (off["unit"] == "tissue_pool")]
                ["offset_canonical_h"].iloc[0])

    # 6a — design and inferred collection phase per group (T2 primary, T1 sensitivity; D54)
    gp = fd.read("r8_ma_timing/ma_group_phase.csv")
    gp = gp[gp["estimator"].isin(["T2", "T1"])].copy()
    gp["assay"] = np.where(gp["dataset"].str.startswith("scRNA"), "scRNA", "bulk")
    gp["group_label"] = gp["group"].map(MA_GROUP_LABEL)
    assert gp["group_label"].notna().all()
    assert np.allclose((gp["internal_phase_h"] + o_c) % 24, gp["our_zt"] % 24, atol=1e-6)
    gp["exposure_window"] = gp["group"].map({"day_noise": "ZT2-4", "NMD": "ZT2-4", "night_noise": "ZT14-16",
                                             "NMN": "ZT14-16"}).fillna("none")
    f6a = out(fd.prov(gp, "r8_ma_timing/ma_group_phase.csv", "R08; I090",
                      "T2 (primary) and T1 (sensitivity) estimators; our_zt = internal phase + our cochlear clock offset"),
              "f6a_group_phase.csv", "Fig. 7a: inferred collection phase of each sample group of the independent cohort "
              "(Ma 2025) on our ZT axis, with the noise-exposure windows", ["assay", "group", "estimator", "our_zt",
                                                                          "ci95_halfwidth_h", "exposure_window"])
    au = fd.read("r8_ma_timing/ma_timing_audit.csv")
    f6a2 = out(fd.prov(au, "r8_ma_timing/ma_timing_audit.csv", "I090", "select"), "f6a_design_audit.csv",
               "Fig. 7a: design facts of the independent cohort (light cycle, exposure, scRNA / bulk design, collection "
               "time)", ["item", "ma2025", "atlas"])
    t2 = gp[gp["estimator"] == "T2"].set_index("group")
    panel("7", "a", "Clock-based dating of the independent cohort (T2): scRNA controls ZT"
          f"{t2.loc['control', 'our_zt']:.1f} and day-noise libraries ZT{t2.loc['day_noise', 'our_zt']:.1f} share a phase, "
          f"night-noise libraries sit at ZT{t2.loc['night_noise', 'our_zt']:.1f}; bulk controls ZT{t2.loc['CMD', 'our_zt']:.1f}"
          f" / ZT{t2.loc['CMN', 'our_zt']:.1f}, noise groups ZT{t2.loc['NMD', 'our_zt']:.1f} / "
          f"ZT{t2.loc['NMN', 'our_zt']:.1f} (exposure ZT2-4 / ZT14-16; collection times unreported)", [f6a, f6a2], "A",
          "R08", "custom (916)")

    # 6b — phase separations between groups vs the upper bound of a noise-induced clock shift (T2)
    tr = fd.read("r8_ma_timing/ma_timing_resolution.csv")
    sel = tr[tr["item"].str.contains(r"\(T2\)|\[T2\]", regex=True)].copy()
    bound = float(sel[sel["item"].str.startswith("upper 95 % bound")]["value_h"].iloc[0])
    sel = sel[~sel["item"].str.startswith("upper")]
    sel["kind"] = np.where(sel["item"].str.contains("noise-associated"), "noise-associated shift", "group separation")
    sel["abs_separation_h"] = np.abs(wrap12(sel["value_h"]))
    sel["comparison"] = (sel["item"].str.replace(r"\s*\[T2\]|\s*\(T2\)", "", regex=True)
                         .str.replace("bulk: noise-associated phase shift 1 d after ", "bulk: ", regex=False))
    sel["noise_shift_bound_h"] = bound
    f6b = out(fd.prov(sel, "r8_ma_timing/ma_timing_resolution.csv", "I090", "T2 rows; absolute circular separation; "
                      "bound = upper 95 % limit of a noise-associated clock shift (D54)"), "f6b_phase_separation.csv",
              "Fig. 7b: phase separation between sample groups vs the largest clock shift noise alone could cause",
              ["comparison", "kind", "value_h", "abs_separation_h", "ci95_halfwidth_h", "noise_shift_bound_h"])
    nn = sel[sel["comparison"].str.contains("night_noise − day_noise")].iloc[0]
    panel("7", "b", f"Night-noise libraries were collected {nn['abs_separation_h']:.1f} h (±{nn['ci95_halfwidth_h']:.1f}) apart "
          f"from day-noise libraries, beyond the {bound:.1f}-h upper bound of a noise-induced clock shift: exposure and "
          "collection time are inseparable for the night-noise scRNA contrasts (conditional evidence)", [f6b], "A", "R08",
          "custom (916)")

    # 6c — label transfer: cells per label in the cohort vs our cochlear composition
    lt = fd.read("r3_ingest/ma2025_label_transfer.csv")
    assert lt["n_cells"].sum() == 110912
    agg = lt.groupby("label").apply(lambda g: pd.Series({
        "n_cells_ma": g["n_cells"].sum(), "frac_confident": np.average(g["frac_confident"], weights=g["n_cells"])}))
    agg = agg.reset_index().rename(columns={"label": "cell_type"})
    agg["share_ma"] = agg["n_cells_ma"] / agg["n_cells_ma"].sum()
    ct = fd.celltypes().set_index("cell_type")
    agg["share_atlas_cochlea"] = agg["cell_type"].map(ct["n_cells_cochlea"]) / ct["n_cells_cochlea"].sum()
    f6c = out(fd.prov(fd.annotate(agg), "r3_ingest/ma2025_label_transfer.csv; atlas cell counts", "R08",
                      "cells per transferred label (all samples); cell-weighted fraction confident; shares of all cells"),
              "f6c_label_transfer.csv", "Fig. 7c: cohort cells per transferred label vs the share of that label in our cochlea",
              ["cell_type", "n_cells_ma", "frac_confident", "share_ma", "share_atlas_cochlea"])
    conf = float(np.average(agg["frac_confident"], weights=agg["n_cells_ma"]))
    panel("7", "c", f"110,912 cohort cells mapped onto our 35 labels ({100 * conf:.0f} % confident)", [f6c], "A", "R08",
          "custom (916)")

    # 6d — time-matched bulk controls: our predicted vs observed night - day per gene (re-derived; asserted)
    de = fd.read("r3_ma2025/bulk_de.csv")
    tp = fd.read("r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv").set_index("gene")
    bsp = fd.read("r3_ma2025/bulk_sampling_phase.csv")
    ref = bsp[bsp["reference"].str.startswith("GSE88954")].drop_duplicates("group").set_index("group")["group_mean_h"]
    t_day, t_night = float(ref["CMD"]), float(ref["CMN"])
    val = fd.read("r3_ma2025/bulk_validation.csv")
    v0 = val[val["predictor"].str.startswith("our cochlear")].iloc[0]
    assert abs(t_day - v0["day_internal_h"]) < 1e-9 and abs(t_night - v0["night_internal_h"]) < 1e-9
    obs = de[de["contrast"] == "baseline_night_minus_day"].set_index("gene")
    sel_ = tp[(tp["p"] < 0.05) & ~tp.index.isin(phase.CANONICAL_CLOCK)]
    g = sel_.index.intersection(obs.index)
    pred = (sel_.loc[g, "a"] * (np.cos(W * (t_night + o_c)) - np.cos(W * (t_day + o_c)))
            + sel_.loc[g, "b"] * (np.sin(W * (t_night + o_c)) - np.sin(W * (t_day + o_c))))
    r = float(stats.pearsonr(pred, obs.loc[g, "log2FC"])[0])
    assert len(g) == v0["n_genes"] and abs(r - v0["pearson_r"]) < 1e-9, (len(g), r)
    comp = val[val["predictor"].str.startswith("GSE88954")].iloc[0]
    dd = pd.DataFrame({"gene": g, "predicted_night_minus_day": pred.to_numpy(), "observed_log2FC": obs.loc[g, "log2FC"].to_numpy(),
                       "observed_p": obs.loc[g, "p"].to_numpy(), "pearson_r": r, "p_perm": v0["p_perm"],
                       "comparator_r_GSE88954": comp["pearson_r"], "day_internal_h": t_day, "night_internal_h": t_night})
    f6d = out(fd.prov(dd, "r2_pilots/cochlea_tissue_pseudobulk_cosinor.csv; r3_ma2025/bulk_de.csv; "
                      "r3_ma2025/bulk_sampling_phase.csv", "R10; R08", "314 formula re-derived: predicted night - day at the "
                      "inferred bulk control phases (non-clock genes p < 0.05); r asserted equal to bulk_validation.csv"),
              "f6d_bulk_replication.csv", "Fig. 7d: per gene, the night - day difference of time-matched untreated cochleae "
              "(Ma 2025 bulk) predicted from our cochlear tissue rhythms vs observed",
              ["gene", "predicted_night_minus_day", "observed_log2FC", "pearson_r"])
    panel("7", "d", f"Time-matched bulk controls replicate our cochlear rhythms: predicted vs observed night - day, r {r:.2f} "
          f"over {len(g):,} non-clock genes (permutation p {v0['p_perm']:.4f}; GSE88954 comparator r "
          f"{comp['pearson_r']:.2f})", [f6d], "R", "R10", "custom (916)")

    # 6e — heat-shock genes: baseline night - day, the noise x time interaction and night - day after noise
    hs = NIGHT_SETS["heat shock"]
    lk = fd.read("r3_ma2025/bulk_links.csv").set_index(["contrast", "gene_set"])
    base = de[(de["contrast"] == "baseline_night_minus_day") & de["gene"].isin(hs)].set_index("gene")
    inter = de[(de["contrast"] == "noise_x_time_interaction") & de["gene"].isin(hs)].set_index("gene")
    assert len(base) == lk.loc[("baseline_night_minus_day", "heat shock"), "n_genes"] == 10
    assert abs(base["log2FC"].mean() - lk.loc[("baseline_night_minus_day", "heat shock"), "mean_log2FC"]) < 1e-9
    assert abs(inter["log2FC"].mean() - lk.loc[("noise_x_time_interaction", "heat shock"), "mean_log2FC"]) < 1e-9
    he = pd.DataFrame({"gene": base.index, "baseline_night_minus_day": base["log2FC"].to_numpy(),
                       "baseline_p": base["p"].to_numpy(), "interaction": inter.loc[base.index, "log2FC"].to_numpy(),
                       "interaction_p": inter.loc[base.index, "p"].to_numpy()})
    he["after_noise_night_minus_day"] = he["baseline_night_minus_day"] + he["interaction"]
    f6e = out(fd.prov(he, "r3_ma2025/bulk_de.csv; r3_ma2025/bulk_links.csv", "R05", "heat-shock genes; after-noise "
                      "night - day = baseline (CMN - CMD) + interaction = NMN - NMD; set means asserted equal to bulk_links"),
              "f6e_heat_shock_bulk.csv", "Fig. 7e: heat-shock genes in time-matched bulk: night - day in untreated "
              "cochleae and after noise", ["gene", "baseline_night_minus_day", "after_noise_night_minus_day", "interaction"])
    panel("7", "e", f"All {int((he['baseline_night_minus_day'] > 0).sum())} heat-shock genes are higher at night in untreated "
          f"cochleae (mean +{he['baseline_night_minus_day'].mean():.2f} log2) and the night elevation is lost after noise "
          f"(interaction {he['interaction'].mean():+.2f}; night - day after noise {he['after_noise_night_minus_day'].mean():+.2f})",
          [f6e], "R", "R05", "custom (916)")

    # 6f — size of the noise response by exposure time (q < 0.05)
    rows = []
    for c, lab in [("noise_effect_day_group", "day noise (ZT2-4)"), ("noise_effect_night_group", "night noise (ZT14-16)")]:
        d = de[de["contrast"] == c]
        s = d["q_BH"] < 0.05
        rows.append({"contrast": c, "exposure": lab, "n_up": int((s & (d["log2FC"] > 0)).sum()),
                     "n_down": int((s & (d["log2FC"] < 0)).sum()), "n_tested": len(d), "threshold": "q_BH < 0.05"})
    nr = pd.DataFrame(rows)
    nr["n_de"] = nr["n_up"] + nr["n_down"]
    assert nr.set_index("contrast").loc["noise_effect_day_group", "n_de"] == 183
    assert nr.set_index("contrast").loc["noise_effect_night_group", "n_de"] == 817
    f6f = out(fd.prov(nr, "r3_ma2025/bulk_de.csv", "R08", "count of genes q < 0.05 by direction"), "f6f_noise_response.csv",
              "Fig. 7f: genes changed 1 d after day vs night noise (time-matched bulk, q < 0.05)",
              ["exposure", "n_up", "n_down", "n_de"])
    panel("7", "f", "Night noise changes 817 genes (524 up, 293 down; q < 0.05) against 183 after day noise (162 up, 21 down)",
          [f6f], "A", "R08", "custom (916)")

    # 6g (exploratory) — scRNA shares of night- vs day-noise libraries vs our predicted shares (re-derived; asserted)
    cs = fd.read("r8_ma_integration/ma_composition_sc.csv")
    cs = cs[(cs["subset"] == "all cells") & (cs["scale"] == "absolute (share of all cells)")
            & (cs["contrast"] == "night_noise − day_noise")].copy()
    pr = fd.read("r2_figure/abundance_pct_rhythm.csv").query("tissue == 'cochlea'").set_index("cell_type")
    sc2 = gp[(gp["assay"] == "scRNA") & (gp["estimator"] == "T2")].set_index("group")
    ph_n, ph_d = (sc2.loc["night_noise", "internal_phase_h"] + o_c) % 24, (sc2.loc["day_noise", "internal_phase_h"] + o_c) % 24

    def share_fit(l, t):
        r_ = pr.loc[l]
        return max(float(r_["mesor_pct"] + r_["amplitude_pct"] * np.cos(W * (t - r_["peak_zt"]))), 1e-3)

    cs["predicted_log2_ratio"] = [np.log2(share_fit(l, ph_n) / share_fit(l, ph_d)) if l in pr.index else np.nan
                                  for l in cs["label"]]
    cs = cs[np.isfinite(cs["predicted_log2_ratio"]) & np.isfinite(cs["observed_log2_ratio"])].copy()   # as perm_spear
    sm = fd.read("r8_ma_integration/ma_summary_tests.csv")
    ref_ = sm[(sm["family"] == "scRNA composition | absolute (share of all cells) | all cells")
              & (sm["contrast"] == "night_noise − day_noise") & (sm["scope"] == "all adequate labels")
              & (sm["assumption"] == "A0 inferred (T2)")].iloc[0]
    rho = float(stats.spearmanr(cs["predicted_log2_ratio"], cs["observed_log2_ratio"])[0])
    assert len(cs) == ref_["n_items"] and abs(rho - ref_["spearman"]) < 1e-9, (len(cs), rho)
    rel = sm[(sm["family"] == "scRNA composition | relative (share of immune cells) | all cells")
             & (sm["contrast"] == "night_noise − day_noise") & (sm["scope"] == "all adequate labels")
             & (sm["assumption"] == "A0 inferred (T2)")].iloc[0]
    cs["spearman_absolute"], cs["p_perm_one_sided"] = rho, ref_["p_perm_one_sided"]
    cs["spearman_relative"], cs["p_relative"] = rel["spearman"], rel["p_perm_one_sided"]
    cs = cs.rename(columns={"label": "cell_type"})
    f6g = out(fd.prov(fd.annotate(cs), "r8_ma_integration/ma_composition_sc.csv; r2_figure/abundance_pct_rhythm.csv; "
                      "r8_ma_timing/ma_group_phase.csv", "R08", "EXPLORATORY: predicted log2 share ratio at the inferred "
                      "night- / day-noise phases (710 formula); Spearman asserted equal to ma_summary_tests.csv"),
              "f6g_composition_sc.csv", "Fig. 7g (exploratory): observed vs predicted change in cell-type share between "
              "night- and day-noise libraries", ["cell_type", "predicted_log2_ratio", "observed_log2_ratio",
                                                  "spearman_absolute"])
    panel("7", "g", f"EXPLORATORY, conditional on no noise x time interaction: shares in night- vs day-noise libraries follow "
          f"our predicted shares (rho {rho:.2f} absolute, {rel['spearman']:.2f} immune-relative); time-matched bulk "
          "deconvolution rho 0.44 absolute, 0.16 immune-relative (declared follow-up)", [f6g], "R", "R08", "custom (916)")

    # 6h (exploratory) — myeloid / B clocks follow the ~11.5-h collection shift of structural cells?
    ca = fd.read("r8_ma_integration/ma_clock_activity.csv")
    ca["deviation_from_structural_h"] = wrap12(ca["night_minus_day_h"] - ca["structural_night_minus_day_h"])
    assert ((np.abs(ca["deviation_from_structural_h"]) <= 4 + 1e-9) == ca["tracking_within_4h"]).all()
    ca["pooled"] = ~ca["unit"].isin(set(fd.celltypes()["cell_type"]))
    f6h = out(fd.prov(ca, "r8_ma_integration/ma_clock_activity.csv", "M07", "EXPLORATORY: deviation of each unit's "
                      "night - day clock shift from the structural shift (circular, h); tracking = |deviation| <= 4 h "
                      "(asserted equal to the stored flag)"), "f6h_clock_tracking.csv",
              "Fig. 7h (exploratory): clock-phase shift between night- and day-noise libraries per immune cell type, "
              "relative to the shift of structural cells", ["unit", "lineage", "night_minus_day_h", "deviation_from_structural_h",
                                                            "tracking_within_4h"])
    my = ca[(ca["lineage"] == "myeloid") & ~ca["pooled"]]
    panel("7", "h", f"EXPLORATORY: clocks of {int(my['tracking_within_4h'].sum())} of {len(my)} myeloid types shift with the "
          f"~{abs(ca['structural_night_minus_day_h'].iloc[0]):.1f}-h collection difference of structural cells (within 4 h), "
          "as do pooled neutrophils and monocytes", [f6h], "A", "M07", "custom (916)")

    # 6i (exploratory) — evidence ladder per claim
    cl = fd.read("r8_ma_integration/ma_claim_lines.csv")
    cnt = cl.groupby(["claim", "status"]).size().unstack(fill_value=0).reset_index()
    ctab = fd.read("r8_ma_integration/ma_claim_table.csv")[["claim", "confirmatory_status (unchanged)",
                                                             "updated_exploratory_grade", "timing_basis"]]
    lad = ctab.merge(cnt, on="claim", how="left")
    lad["claim_short"] = lad["claim"].map(CLAIM_SHORT)
    assert lad["claim_short"].notna().all()
    lad["row_order"] = range(1, len(lad) + 1)
    f6i = out(fd.prov(lad, "r8_ma_integration/ma_claim_table.csv; r8_ma_integration/ma_claim_lines.csv", "R08; H1; M04; M07",
                      "EXPLORATORY: number of lines per status and the updated exploratory grade per claim"),
              "f6i_claim_ladder.csv", "Fig. 7i (exploratory): evidence lines from the independent cohort per claim and the "
              "exploratory grade", ["claim_short", "updated_exploratory_grade"])
    panel("7", "i", "EXPLORATORY evidence ladder: lines per claim (supporting / inconclusive / opposing / not testable) and "
          "the exploratory grade; B-lineage programme rhythms are opposed (rho -0.70); confirmatory verdicts unchanged",
          [f6i], "A", "R08; H1; M04; M07", "custom (916)")

# ---------------------------------------------------------------- Extended Data 1-5 (D68, D69)
def build_ed1():
    # ED1b — cells per cell type and sample (counts asserted against the per-sample QC summary)
    cc = fd.read("00_audit/celltype_counts_by_sample.csv")
    qs = fd.read("00_audit/sample_qc_summary.csv").set_index("sample_id")
    tot = cc.groupby("sample_id")["n_cells"].sum()
    assert (tot == qs.loc[tot.index, "n_cells"]).all()
    e1b = out(fd.prov(fd.annotate(cc.drop(columns=["cell_class", "cell_type_abbr"])), "00_audit/celltype_counts_by_sample.csv",
                      "", "select; labels; totals asserted equal "
                      "to sample_qc_summary"), "ed1_counts_by_sample.csv",
              "ED1b: cells per cell type and sample", ["sample_id", "tissue", "zt", "cell_type", "n_cells"])
    # ED1c — label support (own-label score margin over the best alternative label)
    ls = fd.read("00_audit/label_support_summary.csv")
    e1c = out(fd.prov(fd.annotate(ls), "00_audit/label_support_summary.csv", "", "select; labels"),
              "ed1_label_support.csv", "ED1c: label support per cell type and tissue (median margin, share of cells with "
              "weaker own-label support, top alternative)", ["tissue", "cell_type", "median_margin", "frac_flagged"])
    lq = fd.read("00_audit/label_support_qc.csv", usecols=["cell_id", "margin", "flag_weaker_own_support"])
    assert len(lq) == 160643 and lq["cell_id"].is_unique
    e1e = out(fd.prov(lq, "00_audit/label_support_qc.csv", "", "per-cell margin of the own-label score over the best "
                      "alternative label (select)"), "ed1_cell_margin.csv", "ED1e: per-cell label-support margin",
              ["cell_id", "margin"])
    panel("ED1", "a", "Per-sample QC distributions (UMIs, genes, mitochondrial and ribosomal %, doublet score)",
          ["ed1_sample_qc_quantiles.csv"], "N", "", "custom (920)")
    panel("ED1", "b", "Every cell type is sampled at every time point in its tissue", [e1b], "A", "", "custom (920)")
    panel("ED1", "c", "Label support per cell type (median margin of own over best alternative label)", [e1c], "A", "",
          "custom (920)")
    panel("ED1", "d", "Cells of each type intermix across sampling times (kNN, observed / expected)",
          ["ed1_time_intermixing.csv"], "N", "", "custom (920)")
    panel("ED1", "e", "Cochlear and marrow cells on the joint embedding; per-cell label-support margin",
          ["f1b_umap_joint.csv", e1e], "N + A", "", "custom (920)")
    panel("ED1", "f", "Haematopoietic cells per sample: share of all cells and subtype composition of the haematopoietic "
          "cells, cochlea and marrow (moved from main Fig. 4, D82)", ["f4a_immune_composition.csv", "palette_classes.csv"],
          "A", "A05; A04", "custom (920)")


def build_ed2():
    # ED2a — three top markers per label: expression of all top-3 genes in every cell type (scope all)
    me = fd.read("r9_figures/f1c_marker_expression.csv")
    me = me[me["scope"] == "all"].copy()
    lab_genes = []
    for g, s in me.dropna(subset=["candidate_for"]).drop_duplicates("gene").set_index("gene")["candidate_for"].items():
        for part in s.split("; "):
            if part.startswith("label_top3:"):
                lab_genes.append({"gene": g, "top3_for_abbr": part.split(":", 1)[1]})
    lg = pd.DataFrame(lab_genes)
    assert lg["top3_for_abbr"].nunique() == 35
    keep = me[me["gene"].isin(set(lg["gene"]))].merge(
        lg.groupby("gene")["top3_for_abbr"].apply(lambda s: "; ".join(sorted(s))).rename("top3_for").reset_index(), on="gene")
    keep["scaled_mean"] = keep.groupby("gene")["mean_log1p_cp10k"].transform(lambda v: (v - v.min()) / (v.max() - v.min()))
    e2a = out(fd.prov(fd.annotate(keep), "r9_figures/f1c_marker_expression.csv", "R15", "top-3 markers per label (901 "
                      "candidates); scaled_mean = per-gene min-max across the 35 types"), "ed2_top3_markers.csv",
              "ED2a: expression (fraction, scaled mean) of every label's top-3 markers across all 35 cell types",
              ["cell_type", "gene", "frac_expressing", "scaled_mean", "top3_for"])
    panel("ED2", "a", "Each of the 35 subtypes has its own top-3 markers (fraction expressing, scaled mean)", [e2a], "N",
          "R15", "custom (921)")
    panel("ED2", "b", "Lineage markers are specific to their target types", ["ed2_atlas_markers.csv", "ed2_atlas_dots.csv"],
          "A + N", "R15", "custom (921)")
    panel("ED2", "c", "Label transfer to four external cochlear atlases: transferred cells, confidence and marker check",
          ["ed2_atlas_validation.csv"], "A", "R15", "custom (921)")


def build_ed3():
    # ED3a — atlas-level summary; ED3b — GSE234926 author labels vs our labels (contingency + agreement); ED3c — composition
    # similarity; ED3d — GeoMx proportions per area; ED3e — GeoMx region contrast of every type (control areas)
    sm = fd.read("r4_concordance/atlas_concordance_summary.csv")
    e3a = out(fd.prov(sm, "r4_concordance/atlas_concordance_summary.csv", "R15", "select"), "ed3_atlas_summary.csv",
              "ED3a: per external atlas, cells, confident share, types recovered and marker-check pass rate",
              ["dataset", "n_cells_qc", "frac_confident", "n_types_recovered_20_confident", "marker_check_pass_rate"])
    au = fd.read("r4_concordance/author_label_agreement.csv")
    e3b = out(fd.prov(au, "r4_concordance/author_label_agreement.csv", "R15", "select"), "ed3_author_labels.csv",
              "ED3b: author cell types of the external atlas vs our transferred labels (dominant label and share, share in "
              "the expected labels)", ["author_celltype", "n_cells", "dominant_label", "dominant_share", "frac_expected"])
    cc = fd.read("r4_concordance/author_label_concordance.csv").rename(columns={"celltype": "author_celltype",
                                                                                 "label": "cell_type"})
    tot = cc.groupby("author_celltype")["share_of_author_type"].sum()
    assert (tot - 1).abs().max() < 1e-9
    agg = au[au["author_celltype"] != "ALL (cell-weighted)"].set_index("author_celltype")
    n_ct = cc.groupby("author_celltype")["n_cells"].sum()
    assert (n_ct.reindex(agg.index) == agg["n_cells"]).all(), "contingency totals differ from the agreement table"
    e3b2 = out(fd.prov(fd.annotate(cc), "r4_concordance/author_label_concordance.csv", "R15", "select; labels"),
               "ed3_author_contingency.csv", "ED3b: GSE234926 author cell types x transferred labels (cells, share of the "
               "author type)", ["author_celltype", "cell_type", "n_cells", "share_of_author_type"])
    cs = fd.read("r4_concordance/composition_similarity.csv")
    e3c = out(fd.prov(cs, "r4_concordance/composition_similarity.csv", "R15", "select"), "ed3_composition_similarity.csv",
              "ED3c: per external sample, Spearman correlation of structural cell-type shares with ours",
              ["dataset", "sample", "spearman_rho_structural_shares", "p"])
    ao = fd.read("r3_spatial/geomx_deconvolution_aoi.csv")
    e3d = out(fd.prov(fd.annotate(ao.drop(columns=["cell_class"])), "r3_spatial/geomx_deconvolution_aoi.csv", "R07",
                      "select; labels"),
              "ed3_geomx_aoi.csv", "ED3d: deconvolved proportion of each cell type per GeoMx area of illumination (region, "
              "noise)", ["aoi", "region", "noise", "cell_type", "proportion"])
    gs = fd.read("r3_spatial/geomx_deconvolution_summary.csv")
    assert len(gs) == 35
    e3e = out(fd.prov(fd.annotate(gs.drop(columns=["cell_class"])), "r3_spatial/geomx_deconvolution_summary.csv", "R07",
                      "select; labels (all types)"),
              "ed3_geomx_summary.csv", "ED3e: GeoMx control areas, mean deconvolved proportion of every type in neuron vs "
              "supporting-cell regions with q", ["cell_type", "mean_prop_neuron_control", "mean_prop_supporting_control",
                                                 "q_region"])
    panel("ED3", "a", "Four external cochlear atlases recover our types (confident share, types with >= 20 confident cells, "
          "marker check)", [e3a], "A", "R15", "custom (922)")
    panel("ED3", "b", "GSE234926 author cell types map to the expected labels (cell-weighted agreement 0.33 strict, 0.77 "
          "broad; ARI 0.17)", [e3b2, e3b], "A", "R15", "custom (922)")
    panel("ED3", "c", "Structural composition of external samples vs ours (rho -0.08 to 0.50; none p < 0.05)", [e3c], "A",
          "R15", "custom (922)")
    panel("ED3", "d", "GeoMx: deconvolved proportions per area, neuron vs supporting-cell regions, control vs noise", [e3d],
          "A", "R07", "custom (922)")
    panel("ED3", "e", "GeoMx control areas: region contrast of every type (SGN 0.49 vs 0.15; Schwann 0.10 vs 0.45)", [e3e],
          "A", "R07", "custom (922)")


def build_ed4():
    bs = fd.read("r4_benchmark/benchmark_sensitivity.csv")
    e4b = out(fd.prov(bs, "r4_benchmark/benchmark_sensitivity.csv", "R09", "select"), "ed4_sensitivity.csv",
              "ED4b: sensitivity of each rule by true fold change and expression decile (held-out dense series)",
              ["dataset", "rule", "truth_fc_bin", "expr_decile", "sensitivity"])
    cr = fd.read("r4_benchmark/benchmark_clock_recovery.csv")
    e4c = out(fd.prov(cr, "r4_benchmark/benchmark_clock_recovery.csv", "R09", "select"), "ed4_clock_recovery.csv",
              "ED4c: recovery of core-clock genes as Tier 1 / Tier 1-2 calls in the dense held-out series",
              ["dataset", "gene", "frac_tier1", "frac_tier1_or_2"])
    pw = fd.read("37_power/power_by_celltype.csv")
    pw = pw[(pw["alpha"] == 0.001) & (pw["status"] == "primary")]
    e4d = out(fd.prov(fd.annotate(pw), "37_power/power_by_celltype.csv", "R09", "primary types; alpha 0.001"),
              "ed4_gene_power.csv", "ED4d: power to detect a gene rhythm per cell type, expression bin and fold change "
              "(alpha 0.001)", ["tissue", "cell_type", "expr_bin", "ptt_fc", "power"])
    ap = fd.read("54_abundance_power/abundance_power.csv")
    e4e = out(fd.prov(fd.annotate(ap), "54_abundance_power/abundance_power.csv", "R09", "select"), "ed4_abundance_power.csv",
              "ED4e: smallest peak-to-trough fold change of a share detectable with 80 % power (alpha 0.05) per type",
              ["tissue", "cell_type", "method", "min_fc_80pct_alpha05"])
    sim = fd.read("33_rhythm_tiers/rhythm_pipeline_simulation.csv")
    e4f = out(fd.prov(fd.annotate(sim), "33_rhythm_tiers/rhythm_pipeline_simulation.csv", "R09", "select"),
              "ed4_simulation.csv", "ED4f: pipeline simulation on real pseudobulk with injected rhythms: power, FDP and "
              "phase error per rule and fold change", ["cell_type", "ptt_fc_injected", "rule", "power", "fdp"])
    panel("ED4", "a", "Realised FDP and sensitivity of the rhythm rules on held-out dense series: Tier 1-2 FDP 3-14 % "
          "vs 32-37 % for nominal p", ["ed4_calibration.csv"], "A", "R09", "custom (923)")
    panel("ED4", "b", "Sensitivity grows with fold change and expression", [e4b], "A", "R09", "custom (923)")
    panel("ED4", "c", "Core-clock genes are recovered as Tier 1-2 calls", [e4c], "A", "R09", "custom (923)")
    panel("ED4", "d", "Smallest gene (alpha 0.001, by expression quintile) and share (alpha 0.05) rhythm detectable with 80 % "
          "power per cell type (analytic; gene power corrected for voom weights in round 9)", [e4d, e4e], "A", "R09",
          "custom (923)")
    panel("ED4", "e", "Pipeline simulation with injected rhythms on the data's own noise: power, FDP and phase error per "
          "rule", [e4f], "A", "R09", "custom (923)")


def build_ed5():
    cg = fd.read("02_clock_controls/clock_gene_rhythm.csv")
    cg = cg[cg["status"] == "primary"]
    e5a = out(fd.prov(fd.annotate(cg), "02_clock_controls/clock_gene_rhythm.csv", "G04; G09", "primary types"),
              "ed5_clock_matrix.csv", "ED5a: every clock gene in every primary cell type: peak, amplitude, permutation p",
              ["tissue", "cell_type", "gene", "peak_zt", "amp_log2", "p_perm"])
    nw = fd.read("34_clock_network/clock_network_by_celltype.csv")
    nw = nw[nw["status"] == "primary"]
    e5d = out(fd.prov(fd.annotate(nw), "34_clock_network/clock_network_by_celltype.csv", "G09", "primary types"),
              "ed5_clock_network.csv", "ED5d: clock correlation distance (CCD, z vs null) and canonical-order circular "
              "correlation per cell type", ["tissue", "cell_type", "z", "Pvalue", "circ_corr_observed_vs_canonical"])
    xt = fd.read("02_clock_controls/clock_crosstissue_concordance.csv")
    e5e = out(fd.prov(fd.annotate(xt), "02_clock_controls/clock_crosstissue_concordance.csv", "G09", "select"),
              "ed5_crosstissue_offsets.csv", "ED5e: clock phase offset of the same cell type in cochlea vs marrow",
              ["cell_type", "clock_phase_offset_h_cochlea", "clock_phase_offset_h_bm", "offset_diff_cochlea_minus_bm_h"])
    eq = fd.read("r2_phase/clock_phase_equality.csv")
    e5f = out(fd.prov(eq, "r2_phase/clock_phase_equality.csv", "G09", "select"), "ed5_phase_equality.csv",
              "ED5f: per clock gene and tissue, circular mean peak across significant cell types, spread and canonical "
              "phase", ["tissue", "gene", "n_celltypes_p05", "circ_mean_peak_zt", "circ_sd_h", "canonical_peak_zt"])
    panel("ED5", "a", "Clock genes in every primary cell type of both tissues (peak, amplitude, permutation p)", [e5a], "A",
          "G04; G09", "custom (924)")
    panel("ED5", "b", "Clock-gene time courses in single cell types (moved from Fig. 2, D63)", ["f2b_clock_timecourses.csv"],
          "A", "G04; R04", "custom (924)")
    panel("ED5", "d", "Clocks of independent series tell our sampling times (moved from Fig. 2, D63)",
          ["f2f_timetelling.csv"], "A", "G09", "custom (924)")
    panel("ED5", "e", "Clock network order per cell type (CCD z, canonical-order correlation)", [e5d], "A", "G09",
          "custom (924)")
    panel("ED5", "f", "Same clock phase in cochlea and marrow for shared types", [e5e], "A", "G09", "custom (924)")
    panel("ED5", "c", "Each clock gene peaks at one phase across cell types, near its canonical phase",
          [e5f, "ed5_tissue_clock_order.csv"], "A", "G09", "custom (924)")
    # g — round 10 R1 (333, D113 / D115): single-cell clock tracking at the clock-UMI depth of haematopoietic cells
    dm = fd.read("r10_reanalysis/cell_clock_depth_matched.csv")
    dec = fd.read("r10_reanalysis/cell_clock_depth_decision.csv").iloc[0]
    e5g = out(fd.prov(fd.annotate(dm), "r10_reanalysis/cell_clock_depth_matched.csv", "R04; G04",
                      "select; joined with the palette"), "ed5_clock_depth_matched.csv",
              "ED5g: single-cell clock tracking of every cochlear cell type at its own depth, at the clock-UMI depth of "
              "the cochlear haematopoietic cells (20 thinnings; with and without equal cell numbers) and in the natural "
              "3-5 clock-UMI stratum (333)", ["cell_type", "group", "variant", "frac_tracking", "n_testable"])
    panel("ED5", "g", f"Single-cell clocks at matched depth: {dec['n_tracking_at_haematopoietic_depth']} of "
          f"{dec['n_structural_types']} structural types track at haematopoietic depth "
          f"({dec['n_structural_untestable_at_haematopoietic_depth']} untestable), so the structural vs immune contrast "
          f"is power-limited (D115); among cells with 3-5 clock UMIs {dec['stratum_structural_tracking']} of 12 "
          f"structural and {dec['stratum_haematopoietic_tracking']} of 10 haematopoietic types track", [e5g], "A",
          "R04; G04", "custom (924)")



def build_ed6():
    # ED6 — rhythm programmes: calls and chronotype per cell type, pathway rhythms by analysis version, TF activity vs
    # mRNA lags, promoter motifs (carrier phases and tissue-level enrichment)
    rs = fd.read("33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv")
    rs = rs[(rs["level"] == "cell_type") & (rs["status"] == "primary")].rename(columns={"group": "cell_type"})
    assert len(rs) == 54
    e6a = out(fd.prov(fd.annotate(rs), "33_rhythm_tiers/rhythm_summary_by_celltype_cell_type.csv", "G01; R09",
                      "primary cell types"), "ed6_rhythm_calls.csv",
              "ED6a: Tier-1 / Tier-2 calls, pi1 and permutation global test per primary cell type",
              ["tissue", "cell_type", "n_tier1", "n_tier2", "pi1_storey", "perm_global_p"])
    ch = fd.read("80_atlas/celltype_chronotype.csv")
    bins = [f"n_peak_zt{h:02d}" for h in range(24)]
    assert (ch[bins].sum(axis=1) == ch["n_rhythmic"]).all()
    lo = ch.melt(id_vars=["tissue", "cell_type", "n_rhythmic", "n_tier1", "circ_mean_peak_zt", "mean_resultant_length",
                          "rayleigh_p", "frac_peak_light_zt0_12"], value_vars=bins, var_name="bin", value_name="n_peak")
    lo["zt_bin"] = lo["bin"].str[-2:].astype(int)
    lo["share"] = lo["n_peak"] / lo["n_rhythmic"]
    lo = lo.drop(columns="bin").sort_values(["tissue", "cell_type", "zt_bin"])
    e6b = out(fd.prov(fd.annotate(lo), "80_atlas/celltype_chronotype.csv", "G01; G05",
                      "long format; share = genes peaking in the 1-h bin / rhythmic genes of the type"),
              "ed6_chronotype.csv", "ED6a: peak-phase distribution (1-h bins) of Tier-1/2 genes per cell type",
              ["tissue", "cell_type", "zt_bin", "n_peak", "share"])
    rc = fd.read("43_pathway_phase/pathway_robust_counts.csv")
    e6c = out(fd.prov(rc, "43_pathway_phase/pathway_robust_counts.csv", "G05", "select"), "ed6_pathway_versions.csv",
              "ED6b: robust pathway / TF-activity rhythms per analysis version, tissue and resource with the modal 3-h bin",
              ["version", "tissue", "resource", "n_robust", "modal_phase_bin_zt"])
    pm = fd.read("43_pathway_phase/pathway_phase_map.csv")
    pm["bin3"] = (pm["peak_zt"] % 24 // 3).astype(int) * 3
    hp = pm.groupby(["version", "tissue", "bin3"]).size().rename("n").reset_index()
    full = pd.MultiIndex.from_product([sorted(pm["version"].unique()), sorted(pm["tissue"].unique()), range(0, 24, 3)],
                                      names=["version", "tissue", "bin3"])
    hp = hp.set_index(["version", "tissue", "bin3"]).reindex(full, fill_value=0).reset_index()
    hp["share"] = hp["n"] / hp.groupby(["version", "tissue"])["n"].transform("sum")
    tot = rc.groupby(["version", "tissue"])["n_robust"].sum()
    assert (hp.groupby(["version", "tissue"])["n"].sum().reindex(tot.index) == tot).all(), "phase map != robust counts"
    e6d = out(fd.prov(hp.rename(columns={"bin3": "zt_bin_start"}), "43_pathway_phase/pathway_phase_map.csv", "G05",
                      "robust pathway rhythms counted per 3-h peak bin; share within version x tissue"),
              "ed6_pathway_peaks.csv", "ED6c: peak-time distribution of robust pathway rhythms per analysis version",
              ["version", "tissue", "zt_bin_start", "n", "share"])
    tl = fd.read("43_pathway_phase/tf_activity_mrna_lag.csv")
    e6e = out(fd.prov(fd.annotate(tl), "43_pathway_phase/tf_activity_mrna_lag.csv", "", "select; labels"),
              "ed6_tf_lags.csv", "ED6d: TF activity (CollecTRI) peak minus TF mRNA peak per TF x cell type (both p < 0.05)",
              ["tissue", "cell_type", "tf", "clock_tf", "lag_h"])
    ts = fd.read("43_pathway_phase/tf_lag_summary.csv")
    e6e2 = out(fd.prov(ts, "43_pathway_phase/tf_lag_summary.csv", "", "select"), "ed6_tf_lag_summary.csv",
               "ED6d: circular mean TF activity - mRNA lag per tissue (clock vs other TFs)",
               ["tissue", "clock_tf", "n_pairs", "circ_mean_lag_h", "rayleigh_p"])
    mp = fd.read("44_motifs/promoter_motif_phase.csv")
    e6f = out(fd.prov(fd.annotate(mp), "44_motifs/promoter_motif_phase.csv", "G05", "select; labels"),
              "ed6_motif_phase.csv", "ED6e: circular mean peak of rhythmic promoter-motif carriers per cell type vs the "
              "motif's expected phase", ["tissue", "cell_type", "motif", "n_carriers", "circ_mean_peak_zt", "expected_zt",
                                         "q_directional_perm"])
    me = fd.read("44_motifs/promoter_motif_enrichment_tissue.csv")
    e6f2 = out(fd.prov(me, "44_motifs/promoter_motif_enrichment_tissue.csv", "G05", "select"),
               "ed6_motif_enrichment.csv", "ED6e: tissue-level promoter-motif enrichment among rhythmic genes "
               "(Cochran-Mantel-Haenszel over cell type x expression decile)", ["tissue", "motif", "mh_odds_ratio", "q_BH"])
    panel("ED6", "a", "Rhythm calls per cell type (Tier 1 / 2, pi1) and their peak-time distribution (chronotype)",
          [e6a, e6b], "A", "G01; G05", "custom (925)")
    panel("ED6", "b", "Robust pathway / TF-activity rhythms per analysis version (primary, depth-matched, centred, both) "
          "and their modal phase", [e6c], "A", "G05", "custom (925)")
    panel("ED6", "c", "Peak-time distribution of robust pathway rhythms by analysis version", [e6d], "A", "G05",
          "custom (925)")
    panel("ED6", "d", "TF activity follows TF mRNA by ~1 h in the cochlea (circular mean, Rayleigh)", [e6e, e6e2], "A", "",
          "custom (925)")
    panel("ED6", "e", "Promoter-motif carriers: phase per cell type vs expected and tissue-level enrichment (NF-kB "
          "enriched in both tissues)", [e6f, e6f2], "A", "G05", "custom (925)")


def _annotate_with_all(df):
    """fd.annotate for tables that also carry tissue-level 'ALL' rows (kept, without type annotation)."""
    typed = fd.annotate(df[df["cell_type"] != "ALL"])
    return pd.concat([df[df["cell_type"] == "ALL"], typed], ignore_index=True)


def build_ed7():
    # ED7 — heat shock and drivers: per-sample handling effect, glucocorticoid dependence, driver-evidence enrichment,
    # phase-dependent driver flags vs a phase null per type, baboon heat-shock phase per tissue, systemic-cue shares
    hs = fd.read("r3_audit/handling_sample_effect.csv")
    e7a = out(fd.prov(hs, "r3_audit/handling_sample_effect.csv", "R05", "select"), "ed7_handling_samples.csv",
              "ED7a: per-sample dissociation-stress effect (sample term of the stress-profile model) per tissue",
              ["tissue", "sample_id", "zt", "sample_effect_stress", "sample_effect_cosinor_p"])
    gp = fd.read("r3_audit/gc_response_phases.csv")
    e7b = out(fd.prov(gp, "r3_audit/gc_response_phases.csv", "R05", "select"), "ed7_gc_phases.csv",
              "ED7b: circular mean peak of glucocorticoid-response sets and GC-dependent / -independent rhythmic genes "
              "(GSE88954 adrenalectomy) per tissue", ["tissue", "gene_set", "circ_mean_peak_zt", "resultant", "rayleigh_p"])
    gd = fd.read("r3_audit/gc_dependence_by_phase.csv")
    e7b2 = out(fd.prov(gd, "r3_audit/gc_dependence_by_phase.csv", "R05", "select"), "ed7_gc_dependence.csv",
               "ED7b: night- vs day-peaking rhythmic genes that lose their rhythm after adrenalectomy (Fisher)",
               ["tissue", "odds_ratio_night_gc_dependent", "p_fisher"])
    de = fd.read("r4_drivers/driver_enrichment.csv")
    e7c = out(fd.prov(_annotate_with_all(de), "r4_drivers/driver_enrichment.csv", "R16", "select; labels"),
              "ed7_driver_enrichment.csv", "ED7c: driver evidence among rhythmic calls vs background (CMH over expression "
              "deciles) per tissue and cell type", ["tissue", "cell_type", "evidence", "cmh_odds_ratio", "cmh_q"])
    dn = fd.read("r4_drivers/driver_phase_null.csv")
    e7d = out(fd.prov(_annotate_with_all(dn), "r4_drivers/driver_phase_null.csv", "R16", "select; labels"),
              "ed7_driver_phase_null.csv", "ED7d: phase-dependent driver flags (constant-darkness support, clock-motif "
              "phase, clock-controlled, glucocorticoid-driven): observed share vs peak-shuffled null per type",
              ["tissue", "tier_set", "cell_type", "flag", "obs_over_null", "p_perm"])
    sa = fd.read("82_systemic/systemic_cue_attribution.csv")
    e7f = out(fd.prov(fd.annotate(sa), "82_systemic/systemic_cue_attribution.csv", "R05", "select; labels"),
              "ed7_systemic_attribution.csv", "ED7f: share of each cell type's rhythmic genes that are systemic-cue "
              "programme members peaking in the programme's common phase",
              ["tissue", "cell_type", "set", "n_rhythmic_genes", "frac_of_rhythmic_genes"])
    panel("ED7", "a", "Dissociation-stress sample effect per sample: night-high in both tissues but not rhythmic "
          "(cosinor p 0.11-0.13)", [e7a], "A", "R05", "custom (926)")
    panel("ED7", "b", "Glucocorticoid-response genes peak near ZT16-18 in marrow; night rhythms are not enriched for "
          "adrenalectomy-dependent genes (OR 0.64 marrow, 0.75 cochlea)", [e7b, e7b2], "A", "R05", "custom (926)")
    panel("ED7", "c", "Driver evidence among rhythmic calls vs background per tissue and cell type", [e7c], "A", "R16",
          "custom (926)")
    panel("ED7", "d", "Phase-dependent driver flags vs a peak-shuffled null per cell type", [e7d], "A", "R16",
          "custom (926)")
    panel("ED7", "e", "Baboon heat-shock programme phase in every tissue (raw and clock-aligned) vs mouse",
          ["f3f_baboon_heatshock.csv"], "A", "R12", "custom (926)")
    panel("ED7", "f", "Systemic-cue programme members in the common phase are a small share of each type's rhythmic genes",
          [e7f], "A", "R05", "custom (926)")


def build_ed8():
    # ED8 — marrow and B lineage: Milo rhythmic neighbourhoods (scANVI, primary), B-stage share peaks (within B and of
    # tissue), the 12 pre-specified B hypotheses (D34) with their method lines, ImmGen correspondence of the B labels (D49)
    mi = []
    for tis in ("bm", "cochlea"):
        m = fd.read(f"53_milo/milo_da_results_{tis}.csv")
        mi.append(m[["tissue", "nhood", "majority_cell_type", "majority_fraction", "mean_count", "amp_log2", "peak_zt",
                     "p", "spatial_fdr"]])
    mi = pd.concat(mi, ignore_index=True).rename(columns={"majority_cell_type": "cell_type"})
    assert int(((mi["tissue"] == "bm") & (mi["spatial_fdr"] < 0.1)).sum()) == 481
    e8a = out(fd.prov(fd.annotate(mi), "53_milo/milo_da_results_{bm,cochlea}.csv", "G08",
                      "select; labels of the neighbourhood's majority type"), "ed8_milo_nhoods.csv",
              "ED8a: Milo neighbourhood rhythms (scANVI embedding) per tissue with majority cell type, peak and spatial FDR",
              ["tissue", "nhood", "cell_type", "peak_zt", "amp_log2", "spatial_fdr"])
    bc = fd.read("r5_bcell/b_composition_summary.csv").rename(columns={"feature": "cell_type"})
    bc = bc[~bc["cell_type"].str.startswith("balance:")]          # ILR balances stay in the source table
    e8b = out(fd.prov(fd.annotate(bc), "r5_bcell/b_composition_summary.csv", "B01; B02",
                      "select B-stage rows (balances omitted); labels"),
              "ed8_b_stage_peaks.csv", "ED8b: B-stage share rhythms (within B lineage and share of tissue) with "
              "multi-model verdict", ["tissue", "frame", "cell_type", "peak_zt", "p_perm", "verdict"])
    ph = fd.read("r5_bcell/primary_hypotheses.csv")
    assert len(ph) == 12
    e8c = out(fd.prov(ph, "r5_bcell/primary_hypotheses.csv", "B01-B12", "select"), "ed8_b_scorecard.csv",
              "ED8c: the 12 pre-specified B-lineage hypotheses (D34): primary statistic, exact p, BH q, agreeing methods, "
              "decision", ["hypothesis", "tissue", "primary_p", "q_BH_primary_set", "n_methods_agree", "decision"])
    mm = fd.read("r5_bcell/b_multimethod_matrix.csv")
    e8c2 = out(fd.prov(mm, "r5_bcell/b_multimethod_matrix.csv", "B01-B12", "select"), "ed8_b_methods.csv",
               "ED8c: method lines per B hypothesis (agrees / not)", ["hypothesis", "tissue", "method", "agrees"])
    im = fd.read("r5_bcell/immgen_label_mapping.csv")
    e8d = out(fd.prov(im, "r5_bcell/immgen_label_mapping.csv", "", "select"), "ed8_immgen.csv",
              "ED8d: Spearman correlation of our B-stage pseudobulk with ImmGen B populations (microarray, RNA-seq)",
              ["reference", "tissue", "our_stage", "immgen_population", "spearman_r", "is_best", "expected"])
    panel("ED8", "a", "Rhythmic Milo neighbourhoods per majority cell type (marrow 481 of 3,386; cochlea 266 of 5,034)",
          [e8a], "A", "G08", "custom (927)")
    panel("ED8", "b", "B-stage share peaks within the B lineage and of the tissue (naive B late night in marrow)", [e8b],
          "A", "B01; B02", "custom (927)")
    panel("ED8", "c", "Twelve pre-specified B hypotheses: exact p, q and method lines; none reaches the D34 bar", [e8c, e8c2],
          "A", "B01-B12", "custom (927)")
    panel("ED8", "d", "ImmGen correspondence of the B labels (D49 bracket names)", [e8d], "A", "", "custom (927)")


def build_ed9():
    # ED9 — myeloid hypotheses and timer robustness: M1-M9 scorecard (D45 / D48) with method lines, timer split raw vs
    # within-sample centred and without stress / IEG genes (with the sample-pattern-implied correlation), M3 per-sample
    # neutrophil-branch pseudotime, M1 per-sample maturation axis per neutrophil state
    mh = fd.read("r6_myeloid/m_primary_hypotheses.csv")
    assert len(mh) == 9
    e9a = out(fd.prov(mh, "r6_myeloid/m_primary_hypotheses.csv", "M01-M09", "select"), "ed9_m_scorecard.csv",
              "ED9a: the nine pre-specified myeloid hypotheses (D45 / D48): combined exact p, BH q, agreeing methods, "
              "direction, decision", ["hypothesis", "name", "combined_p", "q_BH", "decision"])
    mm = fd.read("r6_myeloid/m_multimethod_matrix.csv")
    e9a2 = out(fd.prov(mm, "r6_myeloid/m_multimethod_matrix.csv", "M01-M09", "select"), "ed9_m_methods.csv",
               "ED9a: method lines per myeloid hypothesis and tissue (agrees / not)", ["hypothesis", "tissue", "method",
                                                                                   "agrees"])
    tc = fd.read("r6_myeloid/m_timer_split_centred.csv")
    e9b = out(fd.prov(tc, "r6_myeloid/m_timer_split_centred.csv", "G02; M02; G53", "select"), "ed9_timer_robustness.csv",
              "ED9b: neutrophil-state timer concordance (Spearman vs GSE86619 ZT13 - ZT5) raw and within-sample centred, "
              "all genes and without stress / IEG genes, with the sample-pattern-implied correlation",
              ["tissue", "state", "version", "gene_set", "spearman_vs_gse86619", "p_perm",
               "spearman_with_pattern_implied_prediction"])
    tb = fd.read("r6_myeloid/m_trajectory_by_sample.csv")
    tr = fd.read("r6_myeloid/m_trajectory_rhythm.csv")
    tr = tr[tr["feature"].str.contains("M3 primary", regex=False)]
    assert len(tr) == 2
    tb = tb.merge(tr[["tissue", "peak_zt", "p_perm", "ot_ratio", "ot_p_perm"]], on="tissue", how="left")
    e9c = out(fd.prov(tb, "r6_myeloid/m_trajectory_by_sample.csv; m_trajectory_rhythm.csv", "M03",
                      "per-sample median / quartiles joined with the M3 primary rhythm test"), "ed9_m3_pseudotime.csv",
              "ED9c: per-sample median neutrophil-branch pseudotime (GMP + band + mature) with the M3 rhythm and OT tests",
              ["tissue", "zt", "median", "q25", "q75", "p_perm", "ot_p_perm"])
    ax_ = fd.read("r6_myeloid/m_axes_by_sample.csv")
    ax_ = ax_[ax_["axis"] == "maturation (Xie)"]
    e9d = out(fd.prov(ax_, "r6_myeloid/m_axes_by_sample.csv", "M01", "maturation (Xie) axis only"),
              "ed9_m1_maturation.csv", "ED9d: per-sample median maturation score (Xie et al.) per neutrophil group and tissue",
              ["tissue", "group", "zt", "median_score"])
    panel("ED9", "a", "Nine pre-specified myeloid hypotheses: none reaches the D45 / D48 bar (M1-M3 partial)", [e9a, e9a2],
          "A", "M01-M09", "custom (928)")
    panel("ED9", "b", "Timer split per neutrophil state: raw vs within-sample centred, with and without stress / IEG genes; "
          "the raw prediction is largely sample-pattern-implied (G53), centring is not the specificity test (Fig. 6c)",
          [e9b], "A", "G02; M02; G53", "custom (928)")
    panel("ED9", "c", "M3: neutrophil-branch pseudotime per sample (OT test marrow p 0.033)", [e9c], "A", "M03",
          "custom (928)")
    panel("ED9", "d", "M1: maturation axis per sample and neutrophil state", [e9d], "A", "M01", "custom (928)")
    panel("ED9", "e", "Test design of the neutrophil timer split: blood ZT13 - ZT5 programme vs the change each neutrophil "
          "state predicts between the same times (idealised curves; moved from main Fig. 5, D82)", [], "I", "G02",
          "custom (928)")
    # f, g — moved from Fig. 5 v2 i and Fig. 4 v2 f when the flow cohort took those places (D89); tables from build_f5 /
    # build_f4
    q = fd.read("r9_figures/f5i_ageing_cycle.csv").set_index(["tissue", "score"])
    panel("ED9", "f", "EXPLORATORY: cochlear neutrophils look fresh (Sell / CD62L-high) at night and aged (Cxcr4 / "
          f"Cxcr2-high) by day (ZT{q.loc[('cochlea', 'sell'), 'peak_zt']:.1f} vs "
          f"ZT{q.loc[('cochlea', 'cxcr4_cxcr2'), 'peak_zt']:.1f}, permutation p {q.loc[('cochlea', 'sell'), 'p_perm']:.3f} / "
          f"{q.loc[('cochlea', 'cxcr4_cxcr2'), 'p_perm']:.3f}); the marrow Cxcr4 / Cxcr2 ratio is not rhythmic "
          f"(p {q.loc[('bm', 'cxcr4_cxcr2'), 'p_perm']:.2f}); moved from the former Fig. 5i (D89)", ["f5i_ageing_cycle.csv"], "A",
          "M01; M06", "custom (928)")
    mc = fd.read("r9_figures/f4d_marrow_content.csv")
    adj = fd.read("r9_figures/f4d_marrow_adjusted.csv")
    key = adj[adj["register_verdict"].isin(["robust", "supported", "suggestive"])]
    span = 2 ** (mc["log2_marrow_vs_structural"].max() - mc["log2_marrow_vs_structural"].min())
    panel("ED9", "g", f"Cochlear samples differ {span:.1f}-fold in marrow content (marrow-resident precursors / structural "
          f"cells) with no daily rhythm (p {mc['rhythm_p'].iloc[0]:.2f}); against the marrow-resident precursors "
          f"(sensitivity, D17) the timed cochlear shares keep their phase (shift <= {key['shift_alr_h'].abs().max():.1f} h) "
          "while p weakens; moved from Fig. 4f (D89)", ["f4d_marrow_content.csv", "f4d_marrow_adjusted.csv"], "A",
          "A05; A04; A15; A21", "custom (928)")
    # h — round 10 R2 (677, D113 / D116): the split as one statistic against the joint relabelling of both tissues
    js = fd.read("r10_reanalysis/timer_split_joint_null.csv")
    jd = fd.read("r10_reanalysis/timer_split_joint_null_distribution.csv")
    jd = jd[jd["unit"].isin(["Neutrophil (Aged)", "pooled neutrophils"])]
    e9h = out(fd.prov(js, "r10_reanalysis/timer_split_joint_null.csv", "G02; G53", "select"), "ed9_timer_joint_null.csv",
              "ED9h: the neutrophil-timer split D = rho(cochlea) - rho(marrow) per neutrophil unit with its exact joint "
              "relabelling p (60 x 60 primary; 720 x 720 secondary) (677)",
              ["unit", "null", "split_D", "p_joint_one_sided", "p_joint_two_sided"])
    e9h2 = out(fd.prov(jd, "r10_reanalysis/timer_split_joint_null_distribution.csv", "G02; G53", "aged and pooled"),
               "ed9_timer_joint_null_distribution.csv", "ED9h: the split over all 3,600 joint relabelings (identity = "
               "observed) for aged and pooled neutrophils (677)", ["unit", "split_D", "is_observed"])
    a_ = js[(js["unit"] == "Neutrophil (Aged)") & js["primary"]].iloc[0]
    p_ = js[(js["unit"] == "pooled neutrophils") & js["primary"]].iloc[0]
    panel("ED9", "h", f"The split as one exact test (cochlea and marrow relabelled independently): aged neutrophils p "
          f"{a_['p_joint_one_sided']:.3f}, pooled neutrophils p {p_['p_joint_one_sided']:.3f} (D116)", [e9h, e9h2], "A",
          "G02; G53", "custom (928)")


def build_ed10():
    # ED10 — Ma 2025 detail: library phase by estimator, group separations vs the noise-shift bound under every
    # estimator, label check, bulk deconvolution validation, summary tests under the eight timing assumptions
    sp = fd.read("r8_ma_timing/ma_sample_phase.csv")
    e10a = out(fd.prov(sp, "r8_ma_timing/ma_sample_phase.csv", "R08; I090", "select"), "ed10_library_phase.csv",
               "ED10a: inferred collection phase of every Ma 2025 library under each clock estimator (our ZT)",
               ["dataset", "sample", "group", "estimator", "our_zt", "ci95_halfwidth_h"])
    tr = fd.read("r8_ma_timing/ma_timing_resolution.csv")
    e10b = out(fd.prov(tr, "r8_ma_timing/ma_timing_resolution.csv", "R08; I090", "select"), "ed10_separations.csv",
               "ED10b: group phase differences per estimator vs the upper 95 % bound of a noise-associated shift",
               ["item", "value_h", "ci95_halfwidth_h"])
    an = fd.read("r8_ma_timing/ma_annotation_check.csv")
    e10c = out(fd.prov(an, "r8_ma_timing/ma_annotation_check.csv", "R08", "select"), "ed10_annotation_check.csv",
               "ED10c: per library noise-response score and clock phase vs its label (no correction needed)",
               ["dataset", "sample", "group", "noise_score", "internal_phase_T1_nr", "label_corrected"])
    dv = fd.read("r8_ma_integration/ma_deconvolution_validation.csv").rename(columns={"label": "cell_type"})
    e10d = out(fd.prov(fd.annotate(dv), "r8_ma_integration/ma_deconvolution_validation.csv", "R08", "select; labels"),
               "ed10_deconvolution_validation.csv", "ED10d: bulk deconvolution validated on pseudo-bulk mixtures of Ma "
               "single cells: correlation of true and estimated fractions per label", ["cell_type", "validation_r",
                                                                                     "mean_true_fraction", "passes"])
    st = fd.read("r8_ma_integration/ma_summary_tests.csv")
    e10e = out(fd.prov(st, "r8_ma_integration/ma_summary_tests.csv", "R08", "select"), "ed10_assumption_tests.csv",
               "ED10e: Ma 2025 summary tests (Spearman of observed vs our predicted change) under each timing assumption",
               ["family", "contrast", "scope", "assumption", "spearman", "p_perm_one_sided"])
    panel("ED10", "a", "Collection phase of every Ma 2025 library under each clock estimator (T2 primary)", [e10a], "A",
          "R08", "custom (929)")
    panel("ED10", "b", "Group separations under every estimator vs the 5.5-h noise-shift bound", [e10b], "A", "R08",
          "custom (929)")
    panel("ED10", "c", "Library labels agree with noise response and clock phase (no corrections)", [e10c], "A", "R08",
          "custom (929)")
    panel("ED10", "d", "Bulk deconvolution validation per label (25 of 34 pass)", [e10d], "A", "R08", "custom (929)")
    panel("ED10", "e", "Summary tests under the eight timing assumptions (exploratory)", [e10e], "A", "R08",
          "custom (929)")


def build_s4():
    """Supplementary cell-type figure (D136; nucleus-anchored typing D142, 961; drawn by 935): raw images, typed cells
    with their GFP call, partner-derived share per cell type and compartment with Clopper-Pearson 95% CI and blinded
    verification counts (B cells as the reporter-negative control), typed cells per mm2."""
    from scipy.stats import beta
    sm = fd.read("r10_parabiosis/celltype_v2_summary.csv").set_index(["marker", "compartment"])
    ce = fd.read("r10_parabiosis/celltype_v2_cells.csv")
    fl = fd.read("r10_parabiosis/parabiosis_celltype_fields.csv")
    dv = fd.read("r10_parabiosis/celltype_v2_development.csv")
    va = fd.read("r10_parabiosis/verification_v2_agreement.csv")
    v944 = fd.read("r10_parabiosis/verification_items_labelled.csv")
    pq = fd.read("r9_figures/f4_parabiont_fields.csv")
    wl = pq[(pq["field"] == "60x_sl2") & (pq["compartment"] == "cochlea")].iloc[0]

    def verified(mk, cp):
        d1 = dv[(dv.rule == "v2") & (dv.question == "typed: one marker+ cell") & (dv.marker == mk) & (dv.compartment == cp)]
        d2 = va[(va.question == "typed: one marker+ cell") & (va.marker == mk) & (va.compartment == cp)]
        return f"{int(d1.n_agree.sum() + d2.n_agree.sum())}/{int(d1.n_decided.sum() + d2.n_decided.sum())}"

    lw = v944[v944.stratum == "typed FUT4 cochlea (60x lateral wall, 931)"]
    rows = [dict(label="Neutrophils, cochlea (60x)", compartment="cochlea (60x lateral wall)", marker="FUT4",
                 typed_cells=int(wl["marker_cells"]), gfp_pos_cells=int(wl["marker_gfp_cells"]),
                 verified=f"{int((lw.q1_cell == 'yes').sum())}/{int((lw.q1_cell != 'unsure').sum())}", used=True,
                 typed_per_mm2=int(wl["marker_cells"]) / float(wl["tissue_mm2"]), source="931 (60x)")]
    for lab, mk, cp, used in [("Neutrophils, cochlea (20x)", "FUT4", "cochlea", False),
                              ("Leukocytes, cochlea", "CD45", "cochlea", True),
                              ("Neutrophils, marrow", "FUT4", "marrow", True),
                              ("Monocytes, marrow", "CD14", "marrow", True),
                              ("Leukocytes, marrow", "CD45", "marrow", True),
                              ("Leukocytes, vestibular marrow", "CD45", "vestibular bone", True),
                              ("B cells, marrow (control)", "CD19", "marrow", True)]:
        r = sm.loc[(mk, cp)]
        rows.append(dict(label=lab, compartment=cp if cp != "vestibular bone" else "marrow (vestibular bone)", marker=mk,
                         typed_cells=int(r.typed_cells), gfp_pos_cells=int(r.gfp_pos_cells), verified=verified(mk, cp),
                         used=used, typed_per_mm2=float(r.typed_per_mm2), source="961"))
    sh = pd.DataFrame(rows)
    sh["gfp_pos_pct"] = 100 * sh.gfp_pos_cells / sh.typed_cells
    ci = [(0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))
          for k, n in zip(sh.gfp_pos_cells, sh.typed_cells)]
    sh["ci_lo_pct"], sh["ci_hi_pct"] = [100 * a for a, _ in ci], [100 * b for _, b in ci]
    s4s = out(fd.prov(sh, "r10_parabiosis/celltype_v2_summary.csv; celltype_v2_development.csv; "
                          "verification_v2_agreement.csv; verification_items_labelled.csv; r9_figures/f4_parabiont_fields.csv",
                      "", "nucleus-anchored typing (961); Clopper-Pearson 95% CI; verified = confirmed / decided in the "
                      "blinded reviews; 20x cochlear FUT4 calls not used (not confirmed)"), "pb_celltype_share.csv",
              "Supplementary cell-type figure: partner-derived (GFP+) share of typed immune cells of the 54-day "
              "wild-type partner per cell type and compartment, with Clopper-Pearson 95% CI, blinded verification and "
              "typed cells per mm2", ["label", "typed_cells", "gfp_pos_cells", "ci_hi_pct", "verified", "used"])
    t = ce[ce.included.astype(bool) & ce.typed.astype(bool)]
    cnt = t.groupby(["field", "marker", "compartment"]).agg(typed_cells_v2=("typed", "size"),
                                                           gfp_pos_cells_v2=("gfp_cell", "sum")).reset_index()
    fl = fl.drop(columns=[c for c in ("typed_cells", "typed_gfp_pos", "marker_objects", "coverage") if c in fl.columns])
    fl = fl.merge(cnt, on=["field", "marker", "compartment"], how="left").fillna({"typed_cells_v2": 0,
                                                                                 "gfp_pos_cells_v2": 0})
    s4f = out(fd.prov(fl, "r10_parabiosis/parabiosis_celltype_fields.csv; r10_parabiosis/celltype_v2_cells.csv", "",
                      "acquisition attributes (938 loaders) with nucleus-anchored typed counts (961)"),
              "pb_celltype_fields.csv", "Supplementary cell-type figure: every acquisition and stain with its threshold, "
              "separability, tissue area, nuclei, inclusion decision and nucleus-anchored typed / GFP+ cells",
              ["field", "marker", "included", "typed_cells_v2"])
    cells = t[["source", "series", "field", "plane", "marker", "compartment", "x", "y", "area_um2", "nucleus_area_um2",
               "ring_pos_frac", "marker_contrast", "ring_sectors", "gfp_overlap", "gfp_contrast", "gfp_cell"]]
    s4c = out(fd.prov(cells, "r10_parabiosis/celltype_v2_cells.csv", "", "typed cells (nucleus-anchored)"),
              "pb_celltype_cells.csv", "Supplementary cell-type figure: every typed immune cell (nucleus-anchored) with its "
              "marker and GFP measures and GFP call", ["field", "marker", "gfp_cell"])
    b = sh.set_index("label")
    panel("S4", "a", "Raw images of the 54-day wild-type partner: neutrophils (FUT4), leukocytes (CD45), B cells (CD19) "
          "and monocytes (CD14) in marrow and cochlea, with GFP and DAPI", ["figures/panels/pb_s4a_*.png", PB_IMG],
          "M", "", "custom (935)")
    panel("S4", "b", "Typed cells (nucleus-anchored) circled by their GFP call: partner-derived, touching GFP only, or "
          "GFP-negative; CD3e stained the labyrinth epithelium and is not interpreted",
          ["figures/panels/pb_s4b_*.png", PB_IMG, PB_OBJ, s4c], "M", "", "custom (935)")
    panel("S4", "c", "Partner-derived share: " + "; ".join(
        f"{lab} {int(b.loc[lab, 'gfp_pos_cells'])}/{int(b.loc[lab, 'typed_cells'])} (verified {b.loc[lab, 'verified']})"
        for lab in b.index), [s4s, s4f], "M", "", "custom (935)")
    panel("S4", "d", "Typed cells per mm2 of tissue per cell type, marrow and cochlea (cochlear neutrophils from the 60x "
          "lateral wall)", [s4s], "M", "", "custom (935)")


def build_s5():
    """Supplementary Fig. 5 (Supplementary Fig. 1 until D133, 4 until D136) — flow-cytometry gating strategy and controls (D92; 932
    tables, drawn by 917)."""
    pl = fd.read("r9_figures/flow_plots.csv")
    gt = fd.read("r9_figures/flow_gates.csv")
    gm = pl[pl.role == "gating"].iloc[0]
    for letter, tag, name in (("a", "co", "cochlea"), ("b", "bm", "temporal-bone marrow")):
        d = pl[pl["plot"].str.startswith(f"gating_{tag}_")]
        panel("S5", letter, f"Gating strategy, {name} of one mouse ({gm.mouse_id}, {gm.zt}, cohort {int(gm.cohort)}; complete in "
              f"both tissues, cochlear neutrophil share closest to the median): {len(d)} contour plots with outliers, every "
              "gate with its frequency (% of the plotted parent, equal to the tube counts)", [FLOW_S1], "A", "",
              "custom (917)")
    c = gt[gt["plot"].str.startswith("control_") & ~((gt["plot"] == "control_FMO_T7") & (gt.gate_name == "Pro/pre-B"))]
    assert (c.pct_shown == 0).all(), c[c.pct_shown != 0]
    panel("S5", "c", "Controls of one processing day (cohort 1, ZT5): unstained cochlea and the FMO control of every gated "
          f"marker (CD11b, Ly6C, CD62L, CXCR4, IgD, IgM) with the same gates: all {len(c)} target gates 0 %", [FLOW_S1],
          "A", "", "custom (917)")


PB_IMG, PB_OBJ = "sfig_parabiosis_images.csv", "sfig_parabiosis_crop_objects.csv"   # written by 935 (displayed tiles)


def _pb_fields():
    """934's tissue fields with the partner-derived cells of the upgraded body rule (D143: 960, and 963 for the fields
    960 did not analyse); a field without a 934 body has no cell."""
    tf = fd.read("r10_parabiosis/parabiosis_tissue_fields.csv")
    b = pd.concat([fd.read("r10_parabiosis/survey_cells_bodies.csv"),
                   fd.read("r10_parabiosis/survey_cells_bodies_extra.csv")], ignore_index=True)
    assert not b.duplicated(["path", "plane", "y", "x"]).any()
    c = b.groupby("path").agg(gfp_cells=("cell", "sum"), bodies_checked=("cell", "size"))
    tf = tf.join(c, on="path")
    wt = tf["partner"] == "WT"
    assert (tf.loc[wt, "bodies_checked"].fillna(0) == tf.loc[wt, "gfp_bodies"]).all()
    tf["gfp_cells"] = tf["gfp_cells"].fillna(0).astype(int)
    tf["cell_density_per_mm2"] = tf["gfp_cells"] / tf["tissue_mm2"]
    return tf.drop(columns="bodies_checked").assign(acquisition=tf["path"], noise=tf["noise"].astype(bool))


def _pb_pairs():
    """Per pair (D143): marrow and cochlear densities of partner-derived cells by the upgraded body rule (960), observed,
    detection-corrected and brightest section, with the label-corrected ratio and its bootstrap 95% CI."""
    rt = fd.read("r10_parabiosis/survey_cells_ratio.csv")
    vr = fd.read("r10_parabiosis/survey_cells_verified.csv").set_index("animal")
    for c_ in ("verified_ratio", "ci_lo", "ci_hi", "p_ratio_ge_1"):
        rt[c_] = rt["animal"].map(vr[c_])
    return rt.sort_values(["noise", "days"]).reset_index(drop=True)


def build_s1():
    """Supplementary Fig. 1 — partner-derived (GFP+) cells across the tissues of the wild-type parabionts (934, D133;
    counted by the upgraded body rule of 960 / 963, D143); raw images, crops and the FUT4 tiles are drawn and logged by
    935 (sfig_parabiosis_images / _crop_objects)."""
    from scipy.stats import binomtest
    tf = _pb_fields()
    tf = tf[(tf["partner"] == "WT") & ~tf["noise"]]
    cols = ["acquisition", "animal", "compartment", "subregion", "compartment_basis", "um_per_px", "n_z", "planes_analysed",
            "tissue_mm2", "gfp_bodies", "gfp_cells", "cell_density_per_mm2", "broad_spectrum", "density_per_mm2",
            "gfp_area_pct", "gfp_sep_r", "gfp_laser_pct", "gfp_detector_v", "gfp_offset"]
    s2f = out(fd.prov(tf[cols], "r10_parabiosis/parabiosis_tissue_fields.csv; r10_parabiosis/survey_cells_bodies.csv; "
                      "r10_parabiosis/survey_cells_bodies_extra.csv", "", "wild-type partners, noise-exposed pair "
                      "excluded (Supplementary Fig. 3); gfp_cells = GFP+ bodies kept by the upgraded rule (960 / 963)"),
              "pb_tissue_fields.csv",
              "Supplementary Fig. 1: every analysed tissue field of the wild-type parabionts (marrow, cochlea, vestibular "
              "bone, liver): sections, tissue area, GFP+ bodies (934), partner-derived cells by the upgraded rule and "
              "their density, and the GFP acquisition settings",
              ["acquisition", "animal", "compartment", "gfp_cells", "cell_density_per_mm2"])
    rt = _pb_pairs()
    rt = rt[~rt["noise"].astype(bool)].copy()
    k, n = int((rt["cochlea_to_marrow"] < 1).sum()), len(rt)
    rt["sign_k_of_n"], rt["sign_p"] = f"{k}/{n}", binomtest(k, n, 0.5).pvalue
    s2p = out(fd.prov(rt, "r10_parabiosis/survey_cells_ratio.csv; r10_parabiosis/survey_cells_verified.csv", "",
                      "non-noise pairs; upgraded body rule (960); exact two-sided sign test across pairs"), "pb_pairs.csv",
              "Supplementary Fig. 1: cochlea and marrow densities of partner-derived cells and their ratio per wild-type "
              "pair (upgraded body rule), observed, detection-corrected and brightest section, the label-corrected "
              "ratio with its bootstrap 95% CI, and the exact sign test across pairs",
              ["animal", "cochlea_to_marrow", "verified_ratio"])
    det = fd.read("r10_parabiosis/parabiosis_detection.csv")
    det = det[~det["animal"].str.contains("noise")].rename(columns={"path": "acquisition"})
    s2d = out(fd.prov(det, "r10_parabiosis/parabiosis_detection.csv", "", "non-noise pairs"), "pb_detection.csv",
              "Supplementary Fig. 1: detection check per cochlear field (share of the animal's marrow GFP+ bodies "
              "detectable at the field's GFP settings) and the corrected count", ["acquisition", "detectable_share"])
    gr = fd.read("r10_parabiosis/parabiosis_gain_reference.csv")
    s2g = out(fd.prov(gr, "r10_parabiosis/parabiosis_gain_reference.csv"), "pb_gain_reference.csv",
              "Supplementary Fig. 1: detector-gain factor from one 54-day position imaged at two GFP voltages",
              ["signal_ratio_median", "beta_per_volt"])
    nf = tf.groupby("compartment").size()
    aq = fd.read("r10_parabiosis/parabiosis_acquisitions.csv")
    aq = aq.rename(columns={"path": "acquisition", "partner_label": "partner_in_name"})
    aq = aq[["acquisition", "animal", "partner_in_name", "compartment", "subregion", "compartment_basis", "um_per_px",
             "n_z", "channels", "gfp_laser_pct", "gfp_detector_v", "gfp_offset", "included", "reason",
             "stage_duplicate_of"]]
    s2a = out(fd.prov(aq, "r10_parabiosis/parabiosis_acquisitions.csv", "", "select"), "pb_acquisitions.csv",
              "Supplementary Fig. 1: every acquisition of the parabiosis image collection with its labels, pixel size, "
              "GFP settings and the inclusion decision with its reason", ["acquisition", "included", "reason"])
    panel("S1", "a", "Raw images: one full optical section per animal and tissue of the wild-type partners (marrow, "
          f"cochlea, vestibular bone, liver), linear display per channel; {int(aq.included.sum())} of {len(aq)} "
          "acquisitions included", ["figures/panels/pb_s1a_*.png", PB_IMG, s2a], "M", "", "custom (935)")
    panel("S1", "b", "GFP channel of 100-um crops with the GFP+ bodies kept (yellow) or rejected (grey) by the upgraded "
          "rule, marrow and cochlea of the 14-, 31- and 54-day pairs", ["figures/panels/pb_s1b_*.png", PB_IMG, PB_OBJ],
          "M", "", "custom (935)")
    pool = tf.groupby(["compartment", "animal"]).apply(lambda g: g["gfp_cells"].sum() / g["tissue_mm2"].sum())
    panel("S1", "c", f"Partner-derived cells per mm2 of tissue per field and pooled per animal ({nf.get('marrow', 0)} "
          f"marrow, {nf.get('cochlea', 0)} cochlear, {nf.get('vestibular bone', 0)} vestibular-bone and "
          f"{nf.get('liver', 0)} liver fields; upgraded rule): pooled marrow {pool['marrow'].min():.0f}-"
          f"{pool['marrow'].max():.0f}, cochlea {pool['cochlea'].min():.0f}-{pool['cochlea'].max():.0f}, vestibular bone "
          f"{pool['vestibular bone'].max():.0f}, liver {pool['liver'].max():.0f}", [s2f], "M", "", "custom (935)")
    panel("S1", "d", f"Cochlea vs adjacent marrow per pair: lower in {k} of {n} (exact sign test P = "
          f"{float(rt['sign_p'].iloc[0]):.2f}); observed ratios {rt['cochlea_to_marrow'].min():.2f}-"
          f"{rt['cochlea_to_marrow'].max():.2f}", [s2p], "M", "", "custom (935)")
    rob = rt[["cochlea_to_marrow", "corrected_cochlea_to_marrow", "best_plane_cochlea_to_marrow"]]
    assert (rob < 1).all().all()
    vv = rt.set_index("animal")
    panel("S1", "e", "Cochlea-to-marrow ratio per pair as observed, detection-corrected (marrow GFP+ bodies detectable "
          f"at each cochlear field's settings, median {det['detectable_share'].median():.2f}; gain factor "
          f"{float(gr['signal_ratio_median'].iloc[0]):.2f} for {float(gr['voltage_low'].iloc[0]):.0f} vs "
          f"{float(gr['voltage_high'].iloc[0]):.0f} V), in each acquisition's brightest section (all below 1, "
          f"{rob.min().min():.2f}-{rob.max().max():.2f}) and label-corrected with bootstrap 95% CI: " + "; ".join(
              f"{a} {vv.loc[a, 'verified_ratio']:.2f} ({vv.loc[a, 'ci_lo']:.2f}-{vv.loc[a, 'ci_hi']:.2f})"
              for a in rt.animal), [s2p, s2d, s2g], "M", "", "custom (935)")


def build_s2():
    """Supplementary Fig. 2 — perilymph cells of the wild-type and the GFP partners (14 and 28 days; 934, drawn by 935)."""
    pf = fd.read("r10_parabiosis/parabiosis_perilymph_fields.csv")
    pf = pf[pf["animal"].isin(["14 d", "28 d"]) & pf["partner_label"].isin(["WT", "GFP"])]
    pf = pf.rename(columns={"path": "acquisition", "partner_label": "partner"})
    s3f = out(fd.prov(pf, "r10_parabiosis/parabiosis_perilymph_fields.csv", "", "14- and 28-day pairs"),
              "pb_perilymph_fields.csv", "Supplementary Fig. 2: perilymph fields of the wild-type and GFP partners, "
              "nucleated cells, GFP+ cells, share and relative GFP sensitivity", ["acquisition", "partner", "nuclei"])
    cl = fd.read("r10_parabiosis/parabiosis_perilymph_cells.csv")
    cl = cl[cl["path"].isin(set(pf["acquisition"]))].rename(columns={"path": "acquisition", "partner_label": "partner"})
    s3c = out(fd.prov(cl, "r10_parabiosis/parabiosis_perilymph_cells.csv", "", "14- and 28-day pairs"),
              "pb_perilymph_cells.csv", "Supplementary Fig. 2: perilymph nucleated cells with cytoplasmic-ring GFP, "
              "local background and GFP status", ["acquisition", "ring_gfp", "background", "gfp_pos"])
    pool = pf.groupby(["animal", "partner"]).apply(lambda g: 100 * g["gfp_pos"].sum() / g["nuclei"].sum())
    wt = pf[pf["partner"] == "WT"]
    assert (wt["sensitivity_vs_gfp_partner"] >= 1).all()
    panel("S2", "a", "Raw images: full fields of perilymph preparations (wild-type and GFP partners, 14 and 28 days), "
          "GFP and DAPI, linear display", ["figures/panels/pb_s2a_*.png", PB_IMG], "M", "", "custom (935)")
    panel("S2", "b", "Cell calls in 150-um crops: nuclei outlined, GFP+ cells marked", ["figures/panels/pb_s2b_*.png",
          PB_IMG, PB_OBJ], "M", "", "custom (935)")
    panel("S2", "c", "GFP+ share of nucleated cells per field and pooled: wild-type "
          f"{pool.xs('WT', level='partner').min():.1f}-{pool.xs('WT', level='partner').max():.1f}% vs GFP partner "
          f"{pool.xs('GFP', level='partner').min():.0f}-{pool.xs('GFP', level='partner').max():.0f}%", [s3f], "M", "",
          "custom (935)")
    panel("S2", "d", f"Cytoplasmic-ring GFP over local background per cell ({len(cl)} cells) with the pre-set rule",
          [s3c], "M", "", "custom (935)")
    panel("S2", "e", "GFP sensitivity of the wild-type fields relative to the same pair's GFP-partner fields "
          f"({wt['sensitivity_vs_gfp_partner'].min():.1f}-{wt['sensitivity_vs_gfp_partner'].max():.1f}-fold)", [s3f],
          "M", "", "custom (935)")


def build_s3():
    """Supplementary Fig. 3 — the one noise-exposed pair (31 days; 934, D133), drawn by 935."""
    tf = _pb_fields()
    tf = tf[(tf["partner"] == "WT") & tf["compartment"].isin(["marrow", "cochlea"])]
    cols = ["acquisition", "animal", "noise", "compartment", "subregion", "um_per_px", "planes_analysed", "tissue_mm2",
            "gfp_bodies", "gfp_cells", "cell_density_per_mm2", "gfp_laser_pct", "gfp_detector_v", "gfp_offset"]
    s4f = out(fd.prov(tf[cols], "r10_parabiosis/parabiosis_tissue_fields.csv; r10_parabiosis/survey_cells_bodies.csv",
                      "", "wild-type marrow and cochlea; gfp_cells = upgraded body rule (960)"),
              "pb_noise_tissue_fields.csv", "Supplementary Fig. 3: marrow and cochlear fields of the noise-exposed and the "
              "other wild-type partners, partner-derived cells (upgraded rule) per mm2",
              ["acquisition", "animal", "noise", "cell_density_per_mm2"])
    rt = _pb_pairs()[["animal", "noise", "marrow_cells", "marrow_density", "cochlea_cells", "cochlea_density",
                      "cochlea_to_marrow", "cochlea_corrected_density", "corrected_cochlea_to_marrow", "verified_ratio",
                      "ci_lo", "ci_hi"]]
    s4p = out(fd.prov(rt, "r10_parabiosis/survey_cells_ratio.csv; r10_parabiosis/survey_cells_verified.csv", "",
                      "upgraded body rule (960)"), "pb_noise_pairs.csv", "Supplementary Fig. 3: pooled marrow and "
              "cochlear densities of partner-derived cells (upgraded rule) and their ratio, noise-exposed vs other "
              "wild-type partners", ["animal", "noise", "cochlea_to_marrow"])
    pf = fd.read("r10_parabiosis/parabiosis_perilymph_fields.csv")
    pf = pf[pf["partner_label"].isin(["WT", "GFP"])].rename(columns={"path": "acquisition", "partner_label": "partner"})
    pf["noise"] = pf["animal"].str.contains("noise")
    s4l = out(fd.prov(pf, "r10_parabiosis/parabiosis_perilymph_fields.csv"), "pb_noise_perilymph_fields.csv",
              "Supplementary Fig. 3: perilymph fields of the noise-exposed pair with the other pairs' fields for "
              "reference (GFP-partner share = labelling and detection check)", ["acquisition", "animal", "partner"])
    sec = fd.read("r10_parabiosis/parabiosis_noise_sections.csv").rename(columns={"path": "acquisition"})
    s4s = out(fd.prov(sec, "r10_parabiosis/parabiosis_noise_sections.csv"), "pb_noise_sections.csv",
              "Supplementary Fig. 3: analysed optical sections of the noise-exposed and the compared unexposed "
              "acquisitions, GFP+ cells per mm2", ["set", "acquisition", "plane", "density_per_mm2"])
    cm = fd.read("r10_parabiosis/parabiosis_noise_comparison.csv")
    s4c = out(fd.prov(cm, "r10_parabiosis/parabiosis_noise_comparison.csv"), "pb_noise_comparison.csv",
              "Supplementary Fig. 3: pooled densities and noise / unexposed ratios with within-animal bootstrap 95% "
              "intervals; the noise pair against the unexposed pairs (z-score, 95% prediction interval)",
              ["comparison", "estimate"])
    nz = rt[rt["noise"].astype(bool)].iloc[0]
    rest = rt[~rt["noise"].astype(bool)]
    assert nz["cochlea_to_marrow"] < 1
    c_ = cm.set_index("comparison")
    pair = c_[c_["kind"] == "pair level (log10)"]
    assert pair["inside_prediction_interval"].astype(bool).all()
    gfp = pf[pf["partner"] == "GFP"].groupby("animal").apply(lambda g: 100 * g["gfp_pos"].sum() / g["nuclei"].sum())
    panel("S3", "a", "Raw images (full optical sections, linear display) above the counted GFP+ cells (GFP channel of "
          "100-um crops): marrow and apex of the noise-exposed pair, marrow and spiral lamina of the unexposed 31-day "
          "pair, apex of the unexposed 14-day pair", ["figures/panels/pb_s3a_*.png", PB_IMG, PB_OBJ], "M", "",
          "custom (935)")
    zc = pair.loc["noise pair vs unexposed pairs: cochlea density"]
    zm = pair.loc["noise pair vs unexposed pairs: marrow density"]
    panel("S3", "b", "GFP+ cells per mm2 per optical section with pooled values and within-animal bootstrap 95% "
          f"intervals: noise marrow {c_.loc['noise marrow: pooled density', 'estimate']:.0f}, unexposed 31-day marrow "
          f"{c_.loc['31 d marrow: pooled density', 'estimate']:.0f}; noise apex "
          f"{c_.loc['noise cochlea (apex): pooled density', 'estimate']:.0f}, 31-day spiral lamina "
          f"{c_.loc['31 d cochlea (spiral lamina): pooled density', 'estimate']:.0f}, 14-day apex "
          f"{c_.loc['unexposed apex (14 d): pooled density', 'estimate']:.0f}", [s4s, s4c], "M", "", "custom (935)")
    panel("S3", "c", f"Noise-exposed pair among the unexposed pairs: cochlea {nz['cochlea_density']:.0f} (z "
          f"{zc['z_vs_unexposed']:.2f}) and marrow {nz['marrow_density']:.0f} (z {zm['z_vs_unexposed']:.1f}) GFP+ cells "
          f"per mm2, inside the 95% prediction intervals; other pairs cochlea {rest['cochlea_density'].min():.0f}-"
          f"{rest['cochlea_density'].max():.0f}", [s4f, s4p, s4c], "M", "", "custom (935)")
    rr = c_[c_["kind"] == "density ratio (within-animal bootstrap)"]
    panel("S3", "d", "Noise / unexposed density ratios with within-animal bootstrap 95% intervals: " + "; ".join(
          f"{k} {r_['estimate']:.2f} ({r_['ci_low']:.2f}-{r_['ci_high']:.2f})" for k, r_ in rr.iterrows()), [s4c], "M",
          "", "custom (935)")
    panel("S3", "e", f"Perilymph: the noise pair's GFP partner {gfp.get('31 d noise', np.nan):.0f}% GFP+ against "
          f"{gfp.drop('31 d noise').min():.0f}-{gfp.drop('31 d noise').max():.0f}% in the other GFP partners (detection "
          "failed in that session; not interpreted)", [s4l], "M", "", "custom (935)")


FLOW_S1 = "flow_plots.csv; flow_contours.csv; flow_outliers.csv; flow_gates.csv"

def build_srob():
    """Supplementary figure (D137 / D142) — robustness and blinded verification of the quantification behind Fig. 4 and
    Fig. 5c: GFP+ body detection (939), perilymph ring rule (939), fluid positivity (939), upgraded survey cell rule (962
    sweep; 960 verification and label-corrected ratios), flow gates (940), nucleus-anchored typing (962 sweep; 945 / 961
    verification)."""
    gb = fd.read("r10_parabiosis/robustness_gfp_bodies.csv")
    a = gb.groupby(["animal", "k"]).agg(ratio_median=("cochlea_to_marrow", "median"),
                                        ratio_min=("cochlea_to_marrow", "min"),
                                        ratio_max=("cochlea_to_marrow", "max"), settings=("cochlea_to_marrow", "size")
                                        ).reset_index()
    base = gb[gb.baseline.astype(bool)].set_index("animal")["cochlea_to_marrow"]
    a["published"] = a["animal"].map(base)
    sa = out(fd.prov(a, "r10_parabiosis/robustness_gfp_bodies.csv", "", "per pair and threshold k: median and range "
                     "over minimum area (10, 15, 25 um2) and solidity (0.6, 0.7, 0.8)"), "srob_a_gfp_bodies.csv",
             "Robustness SI a: cochlea / marrow density of GFP+ bodies per pair across the detection settings (939)",
             ["animal", "k", "ratio_median", "ratio_min", "ratio_max"])
    pe = fd.read("r10_parabiosis/robustness_perilymph.csv")
    w_ = pe.pivot_table(index=["ring_um", "ratio", "animal"], columns="partner", values="gfp_pos_pct").reset_index()
    w_["wt_over_gfp"] = w_["WT"] / w_["GFP"]
    w_ = w_.rename(columns={"WT": "wt_pct", "GFP": "gfp_pct"})
    sb = out(fd.prov(w_, "r10_parabiosis/robustness_perilymph.csv", "", "wild-type / GFP-partner share per setting"),
             "srob_b_perilymph.csv", "Robustness SI b: perilymph GFP+ share of wild-type and GFP partners and their ratio "
             "across ring widths and ratio thresholds (939)", ["ring_um", "ratio", "animal", "wt_over_gfp"])
    fl = fd.read("r10_parabiosis/robustness_fluids.csv")
    c = fl.groupby(["compartment", "ratio"]).agg(fut4_median=("fut4_pos_pct", "median"), fut4_min=("fut4_pos_pct", "min"),
                                                 fut4_max=("fut4_pos_pct", "max"), cd45_median=("cd45_pos_pct", "median"),
                                                 cd45_min=("cd45_pos_pct", "min"), cd45_max=("cd45_pos_pct", "max"),
                                                 cells=("cells", "first")).reset_index()
    sc = out(fd.prov(c, "r10_parabiosis/robustness_fluids.csv", "", "per compartment and positivity ratio: median and "
                     "range over cell expansion (1-2 um) and background annulus"), "srob_c_fluids.csv",
             "Robustness SI c: FUT4+ and CD45+ shares of the cochlear-fluid cells across the positivity settings (931 rule; "
             "939)", ["compartment", "ratio", "fut4_median"])
    sv = fd.read("r10_parabiosis/robustness_survey_cells.csv")
    sd = out(fd.prov(sv, "r10_parabiosis/robustness_survey_cells.csv", "", "select"), "srob_d_survey_sweep.csv",
             "Robustness SI d: cochlea / marrow ratio of partner-derived cells per pair at each of 27 settings of the "
             "upgraded survey cell rule (962)", ["animal", "cochlea_to_marrow", "baseline"])
    vr = fd.read("r10_parabiosis/survey_cells_verified.csv")
    se = out(fd.prov(vr, "r10_parabiosis/survey_cells_verified.csv", "", "select"), "srob_e_survey_verified.csv",
             "Robustness SI e: per pair the cochlea / marrow ratio by the upgraded rule and corrected with the blinded "
             "labels (kept x precision + rejected x miss rate; stratified bootstrap 95% CI; 960)",
             ["animal", "rule_ratio", "verified_ratio", "ci_lo", "ci_hi"])
    fg = fd.read("r9_flow/robustness_flow_gates.csv")
    sf = out(fd.prov(fg, "r9_flow/robustness_flow_gates.csv", "", "select"), "srob_f_flow.csv",
             "Robustness SI f: ZT17 / ZT5 ratio, 95% CI and Holm P of P1-P3 at the published gates and with each gate "
             "threshold moved by -10% / +10% (940)", ["perturbation", "readout", "ratio_zt17_zt5", "p_holm"])
    ct = fd.read("r10_parabiosis/robustness_celltype_v2.csv")
    sg = out(fd.prov(ct, "r10_parabiosis/robustness_celltype_v2.csv", "", "select"), "srob_g_typing_sweep.csv",
             "Robustness SI g: partner-derived share per cell type and compartment at each of 243 settings of the "
             "nucleus-anchored typing and GFP call (962)", ["marker", "compartment", "gfp_pos_pct", "baseline"])
    dv = fd.read("r10_parabiosis/celltype_v2_development.csv")
    va = fd.read("r10_parabiosis/verification_v2_agreement.csv")
    v944 = fd.read("r10_parabiosis/verification_items_labelled.csv")
    lw = v944[v944.stratum == "typed FUT4 cochlea (60x lateral wall, 931)"]
    rows = [dict(marker="FUT4", compartment="cochlea (60x)", question="typed: one marker+ cell",
                 n_agree=int((lw.q1_cell == "yes").sum()), n_decided=int((lw.q1_cell != "unsure").sum()))]
    q_map = {"typed: one marker+ cell": "typed: one marker+ cell", "untyped: not a marker+ cell":
             "untyped: not a marker+ cell", "GFP call agrees (confirmed cells)": "GFP call agrees (confirmed cells)"}
    d1 = dv[dv.rule == "v2"][["marker", "compartment", "question", "n_agree", "n_decided"]]
    d2 = va[va.question.isin(q_map)][["marker", "compartment", "question", "n_agree", "n_decided"]]
    pooled = pd.concat([d1, d2]).assign(compartment=lambda d: d.compartment.replace({"vestibular bone": "marrow"}))
    pooled = pooled.groupby(["marker", "compartment", "question"])[["n_agree", "n_decided"]].sum().reset_index()
    pooled = pd.concat([pd.DataFrame(rows), pooled], ignore_index=True)
    pooled = pooled[pooled.n_decided > 0]
    from scipy.stats import norm
    z = norm.ppf(0.975)
    p_ = pooled.n_agree / pooled.n_decided
    n_ = pooled.n_decided
    cen = (p_ + z * z / (2 * n_)) / (1 + z * z / n_)
    hw = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
    pooled["share"], pooled["ci_lo"], pooled["ci_hi"] = p_, (cen - hw).clip(lower=0), (cen + hw).clip(upper=1)
    sh = out(fd.prov(pooled, "r10_parabiosis/celltype_v2_development.csv; r10_parabiosis/verification_v2_agreement.csv; "
                     "r10_parabiosis/verification_items_labelled.csv", "", "development (945) and validation (961) "
                     "samples pooled; vestibular-bone marrow with marrow; Wilson 95% CI"), "srob_h_typing_verified.csv",
             "Robustness SI h: blinded verification of the nucleus-anchored typing per cell type and compartment (typed "
             "cells confirmed, untyped cells correctly negative, GFP calls of confirmed cells)",
             ["marker", "compartment", "question", "share", "ci_lo", "ci_hi"])
    cl = fd.read("r10_parabiosis/robustness_upgraded_claims.csv").set_index("claim")
    panel("SROB", "a", "GFP+ body detection: cochlea / marrow below 1 in every pair across threshold, area and solidity",
          [sa], "A", "", "custom (949)")
    panel("SROB", "b", "Perilymph ring rule: wild-type share below the GFP partner's at every setting", [sb], "A", "",
          "custom (949)")
    panel("SROB", "c", "Fluid positivity: FUT4+ share of perilymph cells >= 91% at every setting; endolymph 47-94%", [sc],
          "A", "", "custom (949)")
    r = cl.loc["cochlea / marrow < 1 in every pair"]
    panel("SROB", "d", f"Upgraded survey cell rule: cochlea / marrow below 1 in every pair at {int(r.holds)}/"
          f"{int(r.settings)} settings ({r.estimate_min:.2f}-{r.estimate_max:.2f})", [sd], "A", "", "custom (949)")
    v = vr.set_index("animal")
    panel("SROB", "e", "Label-corrected ratios: " + "; ".join(f"{a_} {v.loc[a_, 'verified_ratio']:.2f} "
                                                             f"({v.loc[a_, 'ci_lo']:.2f}-{v.loc[a_, 'ci_hi']:.2f})"
                                                             for a_ in v.index), [se], "A", "", "custom (949)")
    panel("SROB", "f", "Flow gates: P2 and P3 replicated at all 31 gate settings, P1 at 30/31", [sf], "A", "",
          "custom (949)")
    r2 = cl.loc["marrow leukocytes above the B-cell background"]
    panel("SROB", "g", "Typing settings: no cochlear leukocyte partner-derived at any of 243 settings; marrow neutrophils "
          f"< 5% at all; marrow leukocytes above the reporter-negative B-cell background at {int(r2.holds)}/"
          f"{int(r2.settings)}", [sg], "A", "", "custom (949)")
    panel("SROB", "h", "Blinded verification of the nucleus-anchored typing (945 development, 961 validation; 944 for "
          "the 60x lateral wall)", [sh], "A", "", "custom (949)")


def build_simg():
    """Supplementary figures (D137): raw route and spatial images with the verification of the tracing (930, 941), and
    the cochlear-fluid raw channels with per-cell values and the robustness of the positivity call (930, 931, 939)."""
    tn = fd.read("r9_figures/trace_verification_null.csv")
    tv = fd.read("r9_figures/trace_verification.csv")
    tn = tn.merge(tv[["view", "thin_on_ridge_share", "p_perm"]], on="view")
    a = out(fd.prov(tn, "r9_figures/trace_verification_null.csv; r9_figures/trace_verification.csv", "", "select"),
            "simg_trace_null.csv", "Route SI: on-ridge share of each randomly rotated and translated copy of the traced "
            "centrelines (999 per view) with the observed share (941)", ["view", "on_ridge_share", "thin_on_ridge_share"])
    pn = "figures/panels/"
    panel("SIMG1", "a", "View 1: traced render, unannotated render and the trace check (vessel ridges of the unannotated "
          "render grey; traced centrelines on a ridge green, off a ridge red; broad paint blue)",
          [pn + "vessels_view1_annotated.png", pn + "vessels_view1_original.png", pn + "trace_check_view1.png"], "M", "",
          "custom (948)")
    panel("SIMG1", "b", "View 2: as a", [pn + "vessels_view2_annotated.png", pn + "vessels_view2_original.png",
                                       pn + "trace_check_view2.png"], "M", "", "custom (948)")
    panel("SIMG1", "c", "On-ridge share of randomly placed traces (999 per view) and the observed share", [a], "A", "",
          "custom (948)")
    panel("SIMG1", "d", "Spatial transcriptomics map of deconvolved neutrophil states (the authors' export)",
          [pn + "spatial_neutrophil_states.png", "image_spatial_key.csv"], "M", "", "custom (948)")
    panel("SIMG2", "a", "Cochlear-fluid preparations: DAPI, FUT4, CD45 and merge per compartment (Olympus FV3000, 60x; "
          "one linear display range per channel)", ["figures/panels/fut4_*_{DAPI,FUT4,CD45,merge}.png",
                                                    "image_panels_log.csv"], "M", "", "custom (948)")
    panel("SIMG2", "b", "FUT4 and CD45 of every nucleated cell over its local background (931)", ["f5a_fut4_cells.csv"],
          "A", "", "custom (948)")
    panel("SIMG2", "c", "Positivity across settings (939): FUT4+ and CD45+ shares per compartment against the ratio "
          "threshold", ["srob_c_fluids.csv"], "A", "", "custom (948)")


def build_stc():
    """Supplementary Fig. 5 (D152) — the brain and the brain-image format: the brain projection of the 31-day partner
    with its GFP+ bodies (964), six random counted brain cells, the 31-day cochlea, marrow and liver sections converted to
    the brain's format with their bodies (966 tiles), native against brain-format densities per animal and tissue, and
    the densities of the 31-day partner across conversions and assumed pixel sizes (966)."""
    lg = fd.read("r10_parabiosis/localbg_display_tiles.csv")                 # D155: the local-background detector
    ob = fd.read("r10_parabiosis/localbg_display_objects.csv")
    a = out(fd.prov(lg, "r10_parabiosis/localbg_display_tiles.csv", "", "select"), "stc_display_tiles.csv",
            "Supplementary Fig. 5a-c: display tiles in the brain image's format (field, section, pixel size, display "
            "ranges, crop windows)", ["tile", "compartment", "um_per_px"])
    b = out(fd.prov(ob, "r10_parabiosis/localbg_display_objects.csv", "", "select"), "stc_display_objects.csv",
            "Supplementary Fig. 5a-c: GFP+ bodies in each display tile with their call (tile pixels)",
            ["tile", "y", "x", "cell"])
    st = fd.read("r10_parabiosis/strict_densities.csv")
    nat = st[st["format"] == "native"].set_index(["animal", "compartment"])
    lbd = fd.read("r10_parabiosis/localbg_densities.csv")
    bfm = lbd[lbd["main"] & (lbd["compartment"] != "brain")].set_index(["animal", "compartment"])
    nv = bfm[["days", "fields", "tissue_mm2", "cells_strict", "strict_density_per_mm2"]].rename(
        columns={"tissue_mm2": "brain_format_tissue_mm2", "cells_strict": "brain_format_cells",
                 "strict_density_per_mm2": "brain_format_density_per_mm2"})
    nv = nv.join(nat[["cells_strict", "density_strict_per_mm2", "density_upgraded_per_mm2"]].rename(
        columns={"cells_strict": "native_cells", "density_strict_per_mm2": "native_density_per_mm2",
                 "density_upgraded_per_mm2": "native_upgraded_density_per_mm2"})).reset_index()
    assert len(nv) == 10 and nv["native_density_per_mm2"].notna().all()
    c = out(fd.prov(nv, "r10_parabiosis/strict_densities.csv; r10_parabiosis/localbg_densities.csv", "", "strict calls: "
                    "native (20 robust SD, 968) vs brain-image format (local-background detection, contrast >= 6, 969)"),
            "stc_native_vs_brain_format.csv", "Supplementary Fig. 5d: partner-derived cells per mm2 per wild-type animal "
            "and tissue by the strict calls, native against the brain-image format",
            ["animal", "compartment", "native_density_per_mm2", "brain_format_density_per_mm2"])
    sv = lbd[lbd["animal"] == "31 d"][["animal", "compartment", "um_per_px", "fields", "tissue_mm2", "bodies", "cells",
                                       "cells_strict", "cell_density_per_mm2", "strict_density_per_mm2",
                                       "main"]].reset_index(drop=True)
    sv["conversion"] = np.where(sv["compartment"] == "brain", "projection", "section")
    d = out(fd.prov(sv, "r10_parabiosis/localbg_densities.csv", "", "31-day wild-type partner, brain-image format, "
                    "local-background detection"),
            "stc_sensitivity.csv", "Supplementary Fig. 5e: partner-derived cells per mm2 of the 31-day partner's tissues "
            "in the brain-image format (local-background detection) by call and assumed pixel size",
            ["compartment", "um_per_px", "cell_density_per_mm2", "strict_density_per_mm2"])
    ch = fd.read("r10_parabiosis/localbg_agreement.csv")
    ch["f1"] = 2 * ch["tp"] / (2 * ch["tp"] + ch["fp"] + ch["fn"])
    e = out(fd.prov(ch, "r10_parabiosis/localbg_agreement.csv", "", "F1 added"), "stc_rule_calibration.csv",
            "Supplementary Fig. 5f: agreement of the survey and the local-background detector, each with the upgraded "
            "and the strict call, with the blinded labels (precision, sensitivity, F1; overall and by compartment)",
            ["call", "scope", "precision", "sensitivity", "f1"])
    br = sv[(sv["compartment"] == "brain")]
    n_cells = int(ob[(ob["tile"] == "mf_brain_31d.png") & ob["strict"].astype(bool)].shape[0])      # D155
    panel("STC", "a", f"Brain of the 31-day wild-type partner (rendered projection; GFP, lectin) with its {n_cells} "
          "partner-derived cells (local-background detection, strict call; 2.773 um per pixel, no nucleus "
          "criterion)",
          ["figures/panels/mf_brain_31d.png", a, b], "M", "", "custom (967)")
    panel("STC", "b", "Six counted brain cells drawn at random (seed 966), 120-um crops",
          ["figures/panels/mf_brain_31d_lb_crop*.png", a, b], "M", "", "custom (967)")
    panel("STC", "c", "31-day cochlea, marrow and liver sections of Supplementary Fig. 2a in the brain image's format, "
          "with their counted cells", ["figures/panels/mf_cochlea_31d.png", "figures/panels/mf_marrow_31d.png",
                                       "figures/panels/mf_liver_31d.png", a, b], "M", "", "custom (967)")
    panel("STC", "d", "Strict rule, native vs brain-format density per animal and tissue: cochlea "
          + "; ".join(f"{r.animal} {r.native_density_per_mm2:.0f} vs {r.brain_format_density_per_mm2:.0f}"
                      for r in nv[nv["compartment"] == "cochlea"].itertuples()), [c], "A", "", "custom (967)")
    panel("STC", "e", "31-day partner across assumed pixel sizes (local-background detection, strict call): brain "
          f"{br['strict_density_per_mm2'].min():.1f}-{br['strict_density_per_mm2'].max():.1f} per mm2", [d], "A", "",
          "custom (967)")
    sc = ch[(ch["scope"] == "all labels")].set_index("call")
    panel("STC", "f", "Agreement with the blinded labels (F1): " + "; ".join(
        f"{k_} {v:.2f}" for k_, v in sc["f1"].items()), [e], "A", "", "custom (967)")


BUILDERS = {"palette": build_palette, "f2": build_f2, "f1": build_f1, "f3": build_f3, "f4": build_f4,
            "f5": build_f5, "f6": build_f6, "f7": build_f7, "ed1": build_ed1, "ed2": build_ed2, "ed3": build_ed3, "ed4": build_ed4,
            "ed5": build_ed5, "ed6": build_ed6, "ed7": build_ed7, "ed8": build_ed8, "ed9": build_ed9, "ed10": build_ed10,
            "s1": build_s1, "s2": build_s2, "s3": build_s3, "s4": build_s4, "s5": build_s5, "srob": build_srob, "simg": build_simg, "stc": build_stc}

if __name__ == "__main__":
    # --index-only: run the builders for their panel rows without writing any figure table (D79); with no builder named,
    # the index is rebuilt from scratch so no stale rows survive
    index_only = "--index-only" in sys.argv[1:]
    todo = [a_ for a_ in sys.argv[1:] if a_ != "--index-only"] or list(BUILDERS)
    write_table = out
    if index_only:
        out = lambda df, name, desc, required=None: name        # noqa: E731
    for k in todo:
        log(f"building {k}" + (" (index only)" if index_only else ""))
        BUILDERS[k]()
    out = write_table
    idx = pd.DataFrame(INDEX)
    if len(idx):
        old = fd.RESULTS / MOD / "panel_index.csv"
        fresh = index_only and set(todo) == set(BUILDERS)
        if old.exists() and not fresh:                           # keep rows of figures not rebuilt in this run
            prev = pd.read_csv(old)
            idx = pd.concat([prev[~prev["figure"].astype(str).isin(set(idx["figure"]))], idx], ignore_index=True)
        idx = idx.sort_values(["figure", "panel"], key=lambda s: s.astype(str)).reset_index(drop=True)
        out(fd.prov(idx, "900_figure_data.py", "", "index"), "panel_index.csv",
            "Index of figure panels: message, data files, status (A / R / N / P / I; M = the user's imaging), finding IDs, template shortlist",
            ["figure", "panel", "files", "status"])
