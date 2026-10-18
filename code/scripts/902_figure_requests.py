"""Round 9 (D60 / D66) — coca-fig request files for the main figures, generated from results/r9_figures (no values typed by
hand). Each panel gets <panel>_data.json (when the template needs a nested object) and <panel>_request.json (template,
semantics, options, output, sources + mapping) in figures/requests/<figure>/. Runs locally next to the coca-fig workspace.

Run: python scripts/902_figure_requests.py [fig1 ed]
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / fd.MOD
REQ = ROOT / "figures" / "requests"
FIGS = ROOT / "figures" / "figures"
CONTRACT = "2.0.0"
ZT_COLOUR = {3: "#FFE8A3", 7: "#FFD25E", 11: "#F2B134", 15: "#8C8C8C", 19: "#5E5E5E", 23: "#2E2E2E"}  # light: warm; dark: grey
PURPLE5 = ["#F2F0F7", "#CBC9E2", "#9E9AC8", "#756BB1", "#54278F"]          # dot colour (scaled mean)
PURPLES = ["#FFFFFF", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8", "#807DBA", "#6A51A3", "#54278F", "#3F007D"]


def csv(name):
    return pd.read_csv(R9 / name)


def num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def write(fig, name, obj):
    d = REQ / fig
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print(f"[req] {p.relative_to(ROOT)}")
    return p


def request(fig, panel, template, semantics, options, data_path=None, sources=None, mapping=None, size=(100, 80),
            name=None):
    req = {"templateId": template, "contractVersion": CONTRACT, "semantics": semantics, "options": options,
           "output": {"name": name or f"fig{panel}", "widthMm": size[0], "heightMm": size[1], "directory": str(FIGS)}}
    if data_path is not None:
        req["sources"] = [{"id": "input", "path": str(data_path), "format": "json"}]
        req["mapping"] = {"steps": [], "bindings": {"data": "input"}}
    else:
        req["sources"], req["mapping"] = sources, mapping
    return write(fig, f"{panel}_request.json", req)


def fig1():
    """Fig. 1 (D66): a joint UMAP by subtype, b sampling clock, c / d subtype markers, e external atlases, f spatial."""
    ct = fd.celltypes().set_index("cell_type")
    order = ct.sort_values("order")
    f = "fig1"

    # 1a joint UMAP coloured by the 35 subtypes (class hue, subtype lightness); key grouped by the 12 classes (CF-037)
    request(f, "1a", "CF-037", {
        "valueMeaning": "Gold-standard joint UMAP of all 160,643 cells (cochlea and temporal-bone marrow, six ZTs) coloured by "
                        "the 35 annotated cell types (subtypes); the key groups them by the 12 cell classes "
                        "(results/r9_figures/f1b_umap_joint.csv; colours results/r9_figures/palette_celltypes.csv)",
        "unit": "UMAP units (arbitrary)",
        "coordinates": "Supplied X_umap of the annotated atlas; no dimensionality reduction is performed"},
        {"title": "", "views": ["category"], "viewLabels": {"category": "Cell type"},
         "groupOrder": list(fd.CLASS_ORDER), "categoryOrder": list(order["abbr"]),
         "palette": dict(zip(order["abbr"], order["colour"])),
         "groupPalette": {c: fd.CLASS_COLOUR[c] for c in fd.CLASS_ORDER},
         "categoryLabels": {a: a for a in order["abbr"]}, "groupLabels": {c: c for c in fd.CLASS_ORDER},
         "legendColumns": 5, "plotSize": 220, "pointRadius": 0.45, "pointOpacity": 0.8},
        sources=[{"id": "umap", "path": str(R9 / "f1b_umap_joint.csv"), "format": "csv"}],
        mapping={"steps": [{"op": "select", "from": "umap", "as": "sel", "columns": ["cell_id", "umap_1", "umap_2", "abbr",
                                                                                  "cell_class"]},
                           {"op": "rename", "from": "sel", "as": "pts", "columns": {"cell_id": "id", "umap_1": "x", "umap_2": "y",
                                                                                   "abbr": "category", "cell_class": "group"}},
                           {"op": "cast", "from": "pts", "as": "pts_num", "columns": {"x": "number", "y": "number"}},
                           {"op": "select", "from": "pts_num", "as": "out", "columns": ["id", "x", "y", "group", "category"]}],
                 "bindings": {"data": "out"}}, size=(183, 46), name="fig1a_umap_subtypes")

    # 1b sampling clock (CF-056)
    a = csv("f1a_sampling_design.csv")
    d = write(f, "1b_data.json", {"observations": [{"id": f"{r.zt_label} {t}", "phase": float(r.zt)}
                                                   for r in a.itertuples() for t in ("cochlea", "marrow")]})
    request(f, "1b", "CF-056", {
        "valueMeaning": "Sampling times of the 12 pooled samples: one cochlea and one temporal-bone marrow sample at each of six "
                        "ZTs (two coincident points per ZT; results/r9_figures/f1a_sampling_design.csv)",
        "unit": "Zeitgeber time (h); lights on 08:00 = ZT0, clock time = ZT + 8",
        "missingAndZero": "No missing time points; ZT0 is lights-on",
        "ordering": "Observations in ZT order", "structureMeaning": "Design schematic of the sampling scheme; no statistics",
        "normalization": "None", "thresholdMeaning": "None", "angleDirection": "clockwise", "phaseOrigin": 0, "period": 24,
        "phaseUnit": "h"},
        {"title": "", "origin": 0, "period": 24,
         "ticks": [{"value": int(z), "label": f"ZT{int(z)}"} for z in a["zt"]],
         "annotations": [{"value": 6, "label": "", "icon": "sun"}, {"value": 18, "label": "", "icon": "moon"}],
         "centerLabel": "LD 12:12", "color": "#222222", "radius": 95, "pointRadius": 4.5, "labelFontSize": 12},
        data_path=d, size=(40, 45), name="fig1b_clock")

    # 1c / 1d one marker per subtype (CF-026); rows = subtypes (colour strip = UMAP colour), columns = genes (strip =
    # colour of the subtype the gene marks)
    for panel_id, fname, name, comp in [("1c", "f1c_markers_structural.csv", "fig1c_markers_structural", "structural"),
                                        ("1d", "f1d_markers_haematopoietic.csv", "fig1d_markers_haematopoietic",
                                         "haematopoietic")]:
        m = csv(fname)
        genes = m.drop_duplicates("gene").sort_values("row_order")
        types = m.drop_duplicates("abbr").sort_values("column_order")
        assert list(genes["target_abbr"]) == list(types["abbr"]), "one marker per subtype, in subtype order"
        colour = dict(zip(types["abbr"], types["colour"]))
        d = write(f, f"{panel_id}_data.json", [{"row": r.abbr, "column": r.gene, "value": round(float(r.scaled_mean), 6),
                                                "size": round(float(r.frac_expressing), 6)}
                                               for r in m.sort_values(["column_order", "row_order"]).itertuples()])
        n = len(types)
        request(f, panel_id, "CF-026", {
            "valueMeaning": f"Mean expression of one marker gene per {comp} subtype, scaled per gene to 0-1 across the {n} "
                            f"{comp} subtypes (mean log1p counts per 10k; results/r9_figures/{fname})",
            "unit": "scaled mean expression (0-1)",
            "sizeMeaning": "Fraction of the subtype's cells with at least one UMI of the gene, encoded by dot area",
            "sizeUnit": "fraction", "normalization": "Per-gene min-max of the subtype means within the compartment",
            "ordering": "Rows: subtypes in display order; columns: the marker of each row's subtype, in the same order"},
            {"title": "", "rowOrder": list(types["abbr"]), "columnOrder": list(genes["gene"]),
             "rowColors": colour, "columnColors": {g: colour[t] for g, t in zip(genes["gene"], genes["target_abbr"])},
             "scheme": PURPLE5, "colorDomain": [0, 1], "colorLabel": "Scaled mean", "labelRotation": -45,
             "cellSize": 10, "width": 86 + 12 * n + 90, "height": 56 + 10 * n + 60,
             "margin": {"top": 56, "right": 90, "bottom": 60, "left": 86}},
            data_path=d, size=(64 if n < 18 else 75, 50 if n < 18 else 58), name=name)

    # 1e (external atlases, CF-025) retired: Fig. 1 v5 shows atlas support in the atlas card (ed2_atlas_validation); its table
    # f1e_external_atlases was removed (D80). Earlier request files in figures/requests/fig1/ are self-contained.

    # 1f spatial (CF-047)
    s = csv("f1f_spatial_geomx.csv")
    grp = {"structural": "Structural", "haematopoietic": "Haematopoietic"}
    d = write(f, "1f_data.json", [{"id": r.cell_type, "label": r.abbr, "group": grp[r.group], "value": round(float(r.diff_pp), 6),
                                   "size": round(float(r.mean_pct), 6)} for r in s.itertuples()])
    request(f, "1f", "CF-047", {
        "valueMeaning": "Difference in deconvolved cell-type proportion between neuron and supporting-cell regions of control "
                        "cochleae (public GeoMx, our profiles as reference); types with region q < 0.05 "
                        "(results/r9_figures/f1f_spatial_geomx.csv)",
        "unit": "percentage points (neuron region minus supporting-cell region)",
        "sizeMeaning": "Mean deconvolved proportion across both regions (%)",
        "thresholdMeaning": "0 = equal proportion in both regions"},
        {"title": "", "groupOrder": ["Structural", "Haematopoietic"],
         "palette": {"Structural": fd.GROUP_COLOUR["structural"], "Haematopoietic": fd.GROUP_COLOUR["haematopoietic"]},
         "threshold": 0, "xDomain": [-40, 40], "xLabel": "Neuron − supporting region (pp)", "sizeLabel": "Mean share (%)",
         "maxRadius": 6, "rowHeight": 19, "plotWidth": 160},
        data_path=d, size=(62, 46), name="fig1f_spatial")


def ed():
    """Extended Data requests that moved out of Fig. 1: ED2 atlas card (CF-055, D66) and ED4 calibration (CF-047, D65)."""
    ct = fd.celltypes().set_index("cell_type")  # noqa: F841 (kept for symmetry with fig1)
    f = "ed"
    col = csv("ed2_atlas_columns.csv").sort_values("column_order")
    tree = csv("ed2_atlas_tree.csv")
    mk = csv("ed2_atlas_markers.csv").sort_values("row_order")
    dots = csv("ed2_atlas_dots.csv")
    val = csv("ed2_atlas_validation.csv")
    zts = sorted(int(c[4:]) for c in col.columns if c.startswith("n_zt"))
    card = {
        "columns": [{"id": r.cell_type, "categoryA": r.cell_class, "categoryB": r.group, "total": int(r.total),
                     "compositionA": {"cochlea": int(r.n_cochlea), "bm": int(r.n_bm)},
                     "compositionB": {str(z): int(getattr(r, f"n_zt{z}")) for z in zts}} for r in col.itertuples()],
        "dotRows": [{"id": g, "label": g} for g in mk["gene"]],
        "heatRows": [{"id": a, "label": a} for a in val.sort_values("atlas_order")["atlas"].unique()],
        "dots": [{"row": r.gene, "column": r.cell_type, "value": round(float(r.scaled_mean), 6)}
                 for r in dots.sort_values(["row_order", "column_order"]).itertuples()],
        "heatmap": [{"row": r.atlas, "column": r.cell_type, "value": None if pd.isna(r.value) else round(float(r.value), 6)}
                    for r in val.sort_values(["atlas_order", "column_order"]).itertuples()],
        "tree": {"root": tree["root"].iloc[0],
                 "branches": [{"id": r.branch_id, "left": r.left, "right": r.right, "height": round(float(r.height), 6)}
                              for r in tree.itertuples()]},
        "bands": [{"id": f"band-{i + 1:02d}", "label": r.abbr, "columns": [r.cell_type]} for i, r in enumerate(col.itertuples())]}
    assert len(card["dots"]) == len(mk) * len(col) and len(card["heatmap"]) == card["heatRows"].__len__() * len(col)
    d = write(f, "ed2_card_data.json", card)
    request(f, "ed2_card", "CF-055", {
        "valueMeaning": "Atlas card of the 35 gold-standard cell types: dendrogram, class and group tracks, cells by tissue and "
                        "by ZT, lineage-marker expression and external-atlas validation (results/r9_figures/ed2_atlas_*.csv)",
        "unit": "Dots: scaled mean expression (0-1); heat: marker z-score; compositions: cell counts",
        "normalization": "Dot value = per-gene min-max of mean log1p(counts / total x 1e4) across the 35 types; heat values "
                         "are supplied z-scores (results/r4_concordance/atlas_coverage.csv); no normalisation in rendering",
        "denominator": "Each composition partitions the type's cells (both tissues, all six samples) and sums to its total",
        "structureMeaning": "Average-linkage tree on 1 - Pearson r of pseudobulk log2 CPM over 2,000 variable genes (display "
                            "ordering only); children ordered structural-first; leaves match the column order",
        "sizeMeaning": "Dot area and colour encode the same scaled mean expression, zero area at zero",
        "ordering": "Columns in dendrogram order; marker rows ordered by the column of their target type; atlases fixed",
        "missingAndZero": "Empty heat cells: fewer than 10 cells of that external atlas were transferred to the label "
                          "(unmeasured); zeros are measured",
        "firstMeaning": "Scaled mean expression of the lineage marker (0-1)",
        "secondMeaning": "z-score of our label's marker genes in external-atlas cells transferred to that label",
        "thresholdMeaning": "No significance claims; descriptive display"},
        {"title": "", "cellWidth": 14, "dotRowHeight": 12, "heatRowHeight": 12, "treeHeight": 50, "categoryHeight": 10,
         "compositionHeight": 22, "labelFontSize": 10.5, "legendGap": 16, "bandOpacity": 0.06, "italicRows": True,
         "categoryA": [{"id": c, "label": c, "color": fd.CLASS_COLOUR[c]} for c in fd.CLASS_ORDER],
         "categoryB": [{"id": "structural", "label": "Structural", "color": fd.GROUP_COLOUR["structural"]},
                       {"id": "haematopoietic", "label": "Haematopoietic", "color": fd.GROUP_COLOUR["haematopoietic"]}],
         "compositionA": [{"id": "cochlea", "label": "Cochlea", "color": fd.TISSUE_COLOUR["cochlea"]},
                          {"id": "bm", "label": "Marrow", "color": fd.TISSUE_COLOUR["bm"]}],
         "compositionB": [{"id": str(z), "label": f"ZT{z}", "color": ZT_COLOUR[z]} for z in zts],
         "trackLabels": {"categoryA": "Class", "categoryB": "Group", "compositionA": "Tissue", "compositionB": "ZT"},
         "legendLabels": {"categoryA": "Class", "categoryB": "Group", "compositionA": "Tissue", "compositionB": "Sampling time",
                          "dot": "Scaled mean expression", "heat": "Marker z (external atlas)"},
         "dotLabel": "Lineage markers", "heatLabel": "External atlases", "columnLabel": "Cell types",
         "dotDomain": [0, 1], "dotColors": ["#FFFFFF", "#252525"], "dotTicks": [0, 0.5, 1], "dotSizeTicks": [1, 0.75, 0.5, 0.25],
         "heatDomain": [-1, 9], "heatColors": PURPLES, "heatTicks": [0, 3, 6, 9]},
        data_path=d, size=(183, 112), name="ed2_atlas_card")

    c = csv("ed4_calibration.csv")
    d = write(f, "ed4_calibration_data.json", [{"id": r.row_id, "label": r.rule_label, "group": r.dataset_label,
                                                "value": round(float(r.fdp_strict_pct), 6), "size": round(float(r.sensitivity_pct), 6)}
                                               for r in c.itertuples()])
    request(f, "ed4_calibration", "CF-047", {
        "valueMeaning": "Realised false-discovery proportion of our rhythm-calling rules when six-sample draws (one sample per "
                        "time point) from dense held-out series are scored against an independent truth; strict truth = "
                        "held-out p > 0.2 (results/r9_figures/ed4_calibration.csv)",
        "unit": "percent of calls", "sizeMeaning": "Sensitivity: share of truth-rhythmic genes called (%)",
        "thresholdMeaning": "10 % reference line (the permutation-FDR target used by Tier 1)"},
        {"title": "", "groupOrder": ["Organs (GSE54650)", "Cochlea (GSE88954)"],
         "palette": {"Organs (GSE54650)": "#222222", "Cochlea (GSE88954)": fd.GROUP_COLOUR["haematopoietic"]},
         "threshold": 10, "xDomain": [0, 40], "xLabel": "Realised FDP (%)", "sizeLabel": "Sensitivity (%)",
         "maxRadius": 6, "rowHeight": 22, "plotWidth": 200},
        data_path=d, size=(60, 45), name="ed4_calibration")


BUILDERS = {"fig1": fig1, "ed": ed}

if __name__ == "__main__":
    for k in sys.argv[1:] or list(BUILDERS):
        BUILDERS[k]()
