"""Round 9 (D68 / D69) — Extended Data Fig. 2 (180 x 170 mm): the full marker atlas (supports Fig. 1e, f).

a  Top-3 markers of every subtype (96 genes) in all 35 subtypes: dot area = fraction expressing, colour = mean
   expression min-max scaled per gene; rows and gene blocks in atlas (dendrogram) order.
b  Lineage markers (20) across the 35 subtypes, with the lineage specificity (target mean / highest other mean).
c  Transfer to four external cochlear atlases: cells transferred to each label (size) and fraction confident (colour);
   outlined = marker check passed.

Inputs: results/r9_figures/ed2_*.csv, palette tables. Output: figures/figures/ed2_markers_v1.*
Run: python scripts/921_ed2_markers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import PT_SMALL, TEXT2, Canvas, fits, header_height, text_width, type_header  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed2_markers_v1"
W_ED = 180.0
PURPLES = LinearSegmentedColormap.from_list("p", ["#F2EEF7", "#CBC9E2", "#9E9AC8", "#6A51A3", "#3F007D"])
GREYS = LinearSegmentedColormap.from_list("g", ["#F2F2F2", "#BDBDBD", "#6E6E6E", "#1A1A1A"])
GREY = "#777777"


def dot_size(f):
    return 0.5 + 9.5 * np.clip(f, 0, 1)


def n_size(n):
    return 1.0 + 6.0 * np.log10(np.maximum(n, 1))




def size_key(cv, x, y, title):
    cv.text(x, y, title, size=PT_SMALL, va="center")
    xx = x + text_width(title, PT_SMALL) + 1.6
    for f in (0.25, 0.5, 1.0):
        cv.mark(xx, y, dot_size(f), fc=GREY)
        cv.text(xx + 1.2, y, f"{f:g}", size=PT_SMALL, va="center")
        xx += 1.2 + text_width(f"{f:g}", PT_SMALL) + 2.0
    return xx


def colour_key(cv, x, y, cmap, ticks, labels, title, w=12.0):
    ax = cv.axes(x, y - 0.6, w, 1.2)
    ax.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    ax.set_yticks([])
    ax.set_xticks(ticks, labels)
    ax.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    cv.text(x + w + 1.0, y, title, size=PT_SMALL, va="center")


# ------------------------------------------------------------------------------------------------ a top-3 markers
def panel_a(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Top-3 markers of every subtype")
    m = pd.read_csv(R9 / "ed2_top3_markers.csv")
    types = cols.sort_values("column_order")
    order = {a: i for i, a in enumerate(types.abbr)}
    g = m.drop_duplicates("gene")[["gene", "top3_for"]].copy()
    g["first"] = g.top3_for.str.split("; ").apply(lambda xs: min(order[x] for x in xs))
    g = g.sort_values(["first", "gene"])
    genes = g.gene.tolist()
    lab_w = max(text_width(a, PT_SMALL) for a in types.abbr) + 3.0
    top = y0 + 15.0
    q = (w - lab_w - 1.0) / len(genes)
    rp = (h - (top - y0) - 6.0) / len(types)
    ax = cv.axes(x0 + lab_w, top, len(genes) * q, len(types) * rp)
    gi = {gg: k for k, gg in enumerate(genes)}
    mm = m[m.abbr.isin(order)]
    ax.scatter(mm.gene.map(gi), mm.abbr.map(order), s=dot_size(mm.frac_expressing), c=PURPLES(mm.scaled_mean.clip(0, 1)),
               linewidths=0, zorder=3)
    for gg, t3 in zip(g.gene, g.top3_for):              # each type's own top-3 cells
        for a in t3.split("; "):
            ax.add_patch(Rectangle((gi[gg] - 0.5, order[a] - 0.5), 1, 1, facecolor="#FFF2C2", edgecolor="none", zorder=1))
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(len(types) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, gg in enumerate(genes):
        cv.text(x0 + lab_w + (k + 0.5) * q, top - 0.5, gg, size=PT_SMALL, ha="center", va="bottom", rotation=90,
                style="italic")
    for i, (_, r) in enumerate(types.iterrows()):
        yy = top + (i + 0.5) * rp
        cv.box(x0 + lab_w - 1.9, yy - rp * 0.4, 1.2, rp * 0.8, r.colour)
        cv.text(x0 + lab_w - 2.5, yy, r.abbr, size=PT_SMALL, ha="right", va="center")
    ky = y0 + h - 2.0
    xe = size_key(cv, x0 + lab_w, ky, "Fraction expressing")
    colour_key(cv, xe + 4.0, ky, PURPLES, [0, 1], ["0", "1"], "Scaled mean")


# ------------------------------------------------------------------------------------------------ b lineage markers
def panel_b(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Lineage markers")
    d = pd.read_csv(R9 / "ed2_atlas_dots.csv")
    mk = pd.read_csv(R9 / "ed2_atlas_markers.csv").sort_values("row_order")
    types = cols.sort_values("column_order")
    order = {ct: i for i, ct in enumerate(types.cell_type)}
    lab_w = max(text_width(gg, PT_SMALL, style="italic") for gg in mk.gene) + 1.6
    right_w = 9.0
    top = y0 + header_height(types.abbr)
    q = (w - lab_w - right_w) / len(types)
    rp = (h - (top - y0) - 0.6) / len(mk)
    ax = cv.axes(x0 + lab_w, top, len(types) * q, len(mk) * rp)
    ri = {gg: k for k, gg in enumerate(mk.gene)}
    dd = d[d.gene.isin(ri) & d.cell_type.isin(order)]
    ax.scatter(dd.cell_type.map(order), dd.gene.map(ri), s=dot_size(dd.frac_expressing),
               c=PURPLES(dd.scaled_mean.clip(0, 1)), linewidths=0, zorder=3)
    ax.set_xlim(-0.6, len(types) - 0.4)
    ax.set_ylim(len(mk) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, (_, r) in enumerate(mk.iterrows()):
        yy = top + (k + 0.5) * rp
        cv.text(x0 + lab_w - 0.8, yy, r.gene, size=PT_SMALL, ha="right", va="center", style="italic")
        cv.text(x0 + w - 0.3, yy, f"{r.lineage_specificity:.1f}", size=PT_SMALL, ha="right", va="center", color=TEXT2)
    cv.text(x0 + w - 0.3, top - 0.6, "spec.", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    type_header(cv, types.abbr, types.colour, x0 + lab_w, top, q)


# ------------------------------------------------------------------------------------------------ c external atlases
def panel_c(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "c", "External atlases")
    v = pd.read_csv(R9 / "ed2_atlas_validation.csv")
    types = cols.sort_values("column_order")
    order = {ct: i for i, ct in enumerate(types.cell_type)}
    atl = v.drop_duplicates("atlas").sort_values("atlas_order").atlas.tolist()
    lab_w = max(text_width(a, PT_SMALL) for a in atl) + 1.6
    top = y0 + header_height(types.abbr)
    q = (w - lab_w - 0.8) / len(types)
    rp = 6.2
    ax = cv.axes(x0 + lab_w, top, len(types) * q, len(atl) * rp)
    assert v.n_cells.max() < 10 ** 4
    for k, a in enumerate(atl):
        d = v[(v.atlas == a) & v.cell_type.isin(order) & (v.n_cells > 0)]
        assert d.frac_confident_of_label.notna().all()
        ax.scatter(d.cell_type.map(order), [k] * len(d), s=n_size(d.n_cells), c=GREYS(d.frac_confident_of_label.clip(0, 1)),
                   edgecolors=np.where(d.marker_check_pass.astype(bool), "#6A51A3", "none"), linewidths=0.6, zorder=3)
        cv.text(x0 + lab_w - 0.8, top + (k + 0.5) * rp, a, size=PT_SMALL, ha="right", va="center")
    ax.set_xlim(-0.6, len(types) - 0.4)
    ax.set_ylim(len(atl) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    type_header(cv, types.abbr, types.colour, x0 + lab_w, top, q)
    ky = top + len(atl) * rp + 4.0
    cv.text(x0 + lab_w, ky, "Cells transferred", size=PT_SMALL, va="center")
    xx = x0 + lab_w + text_width("Cells transferred", PT_SMALL) + 1.8
    for nn, lab in [(10, "10"), (100, "10²"), (1000, "10³"), (10000, "10⁴")]:
        cv.mark(xx, ky, n_size(nn), fc=GREY)
        cv.text(xx + 1.6, ky, lab, size=PT_SMALL, va="center")
        xx += 1.6 + text_width(lab, PT_SMALL) + 2.4
    colour_key(cv, x0 + lab_w, ky + 4.4, GREYS, [0, 1], ["0", "1"], "Fraction confident")
    x = cv.key(x0 + lab_w, ky + 9.0, [([{"s": 9, "fc": "white", "ec": "#6A51A3", "lw": 0.6}], "marker check passed")])
    fits(x, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all()
    cv = Canvas(width=W_ED)
    panel_a(cv, cols, 0.0, 0.0, W_ED, 100.0)
    panel_b(cv, cols, 0.0, 102.0, 96.0, 68.0)
    panel_c(cv, cols, 99.0, 102.0, 81.0, 68.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
