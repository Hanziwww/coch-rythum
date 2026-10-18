"""Shared helpers for the round-9 figure data (results/r9_figures; D60 / D61): cell-type order, class palette, paper labels,
provenance columns. Display conventions only — no statistics are computed here."""
from __future__ import annotations

import colorsys

import pandas as pd

from .io import RESULTS

MOD = "r9_figures"
TISSUES = ["cochlea", "bm"]                                   # fixed facet order: cochlea left, marrow right
TISSUE_LABEL = {"cochlea": "Cochlea", "bm": "Marrow"}
TISSUE_COLOUR = {"cochlea": "#0072B2", "bm": "#D55E00"}      # Okabe–Ito blue / vermillion (Nature figure guide palette)
CLASS_COLOUR = {                                              # Paul Tol 'muted' + extras; avoids the two tissue hues
    "Sensory": "#CC6677", "Neuronal": "#882255", "Glia": "#AA4499", "Epithelial": "#44AA99", "Stromal": "#999933",
    "Bone": "#DDCC77", "Endothelial": "#88CCEE", "Progenitor": "#332288", "Cycling": "#222222", "Erythroid": "#7A0403",
    "Lymphoid": "#117733", "Myeloid": "#E69F00"}
# Cycling / Erythroid revised 2026-10-15 (906 colour-vision QC): the former #5A5A5A / #EE3377 merged with Neuronal /
# Sensory under deuteranopia (CIE76 dE 3.5 / 1.7); now every class pair keeps dE >= 14 under all three dichromacies.
GROUP_COLOUR = {"structural": "#0E5A4E", "haematopoietic": "#767676"}   # dark teal vs grey: dE >= 15 under dichromacy
# and in greyscale (the earlier #348275 had the grey's lightness); both pass 4.5:1 text contrast (group headings)
CLASS_ORDER = list(CLASS_COLOUR)
STRUCTURAL = CLASS_ORDER[:7]
STRUCTURAL_EXTRA = {"Cycling Cell (Stromal)"}   # display grouping only: proliferating stromal cells count as structural
TYPE_ORDER = [
    "Hair Cell (Outer/Inner)", "Spiral Ganglion Neuron", "Schwann Cell", "Oligodendrocyte",
    "SV Marginal Cell", "SV Intermediate Cell (Pigmented)", "SV Intermediate Cell (Functional)", "Epithelial Cell (Non-sensory)",
    "Fibrocyte (Spiral Ligament)", "Fibrocyte (Cochlear)", "Interdental Cell", "Osteoprogenitor", "Osteoblast", "Endothelial Cell",
    "HSC / MPP", "Cycling Cell", "Cycling Cell (Stromal)", "Erythroblast",
    "Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Mature)", "B Cell (Naïve)", "T Cell (CD3+)", "NK Cell",
    "Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)",
    "Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage", "pDC", "Mast Cell"]
# D49: B stages carry the ImmGen equivalent in brackets (labels themselves unchanged, G41)
PAPER_LABEL = {
    "Pro-B Cell (Cycling)": "Pro-B Cell (Cycling) (pro-B, FrB–C)", "Pre-B Cell": "Pre-B Cell (small pre-B, FrD)",
    "B Cell (Transitional)": "B Cell (Transitional) (large pre-B, FrC′)", "B Cell (Mature)": "B Cell (Mature) (immature, FrE / T1)",
    "B Cell (Naïve)": "B Cell (Naïve) (recirculating, FrF / Fo)"}


def read(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel, **kw)


def _shade(hex_colour: str, k: int, n: int) -> str:
    """k-th of n lightness steps around a class hue (single type keeps the class colour). Classes with four or more
    types (D67) get a wider lightness range and a small hue sweep (+-0.035) so neighbouring subtypes separate."""
    if n == 1:
        return hex_colour
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n >= 4:
        lo, hi = max(0.15, l - 0.18), min(0.78, l + 0.38)
        h = (h - 0.035 + 0.07 * k / (n - 1)) % 1.0
    else:
        lo, hi = max(0.18, l - 0.20), min(0.82, l + 0.22)
    l2 = lo + (hi - lo) * k / (n - 1)
    r, g, b = colorsys.hls_to_rgb(h, l2, s)
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def celltypes() -> pd.DataFrame:
    """One row per cell type: class, abbreviation, display order, colour, structural flag, paper label, cells per tissue."""
    c = read("00_audit/celltype_counts_by_sample.csv")
    n = c.pivot_table(index="cell_type", columns="tissue", values="n_cells", aggfunc="sum").fillna(0).astype(int)
    m = c.drop_duplicates("cell_type").set_index("cell_type")[["cell_class", "cell_type_abbr"]]
    assert set(m.index) == set(TYPE_ORDER) and set(m["cell_class"]) == set(CLASS_ORDER), "label set changed"
    rows = []
    for i, t in enumerate(TYPE_ORDER):
        cl = m.loc[t, "cell_class"]
        sib = [x for x in TYPE_ORDER if m.loc[x, "cell_class"] == cl]
        rows.append({"cell_type": t, "cell_class": cl, "abbr": m.loc[t, "cell_type_abbr"], "order": i + 1,
                     "class_order": CLASS_ORDER.index(cl) + 1, "colour": _shade(CLASS_COLOUR[cl], sib.index(t), len(sib)),
                     "class_colour": CLASS_COLOUR[cl], "structural": cl in STRUCTURAL or t in STRUCTURAL_EXTRA,
                     "paper_label": PAPER_LABEL.get(t, t),
                     "n_cells_cochlea": int(n.loc[t, "cochlea"]), "n_cells_bm": int(n.loc[t, "bm"])})
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame, col: str = "cell_type") -> pd.DataFrame:
    """Join class, abbreviation, order and colour onto a table keyed by cell type (left join; unknown types fail)."""
    ct = celltypes().set_index("cell_type")[["cell_class", "abbr", "order", "colour", "structural", "paper_label"]]
    out = df.join(ct, on=col)
    assert out["cell_class"].notna().all(), f"unknown cell types: {sorted(set(df[col]) - set(ct.index))}"
    return out


def prov(df: pd.DataFrame, source: str, finding: str = "", transform: str = "select / rename") -> pd.DataFrame:
    """Provenance columns required on every figure table (source_table, finding_id, transform)."""
    out = df.copy()
    out["source_table"], out["finding_id"], out["transform"] = source, finding, transform
    return out


def tissue_cols(df: pd.DataFrame, col: str = "tissue") -> pd.DataFrame:
    out = df.copy()
    out["tissue_label"] = out[col].map(TISSUE_LABEL)
    out["tissue_order"] = out[col].map({t: i + 1 for i, t in enumerate(TISSUES)})
    return out
