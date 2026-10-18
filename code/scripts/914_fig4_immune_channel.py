"""Round 10 (D134; v3 D142) — Fig. 4: an immune channel connects the cochlea to the temporal-bone marrow (180 x 170 mm,
5-7 pt). Built from the former Fig. 4a-d (912) and Fig. 5a (913) when Fig. 4 was split (user: reorganise the two figures
to make them more informative); raw images first, then their statistics.
v3 (D142; user: "I am not satisfied with version v2. Please revert to the previous version, while only incorporating
improvements to the layout and an upgraded quantitative algorithm."): the v1 panels, with layout refinements (black text
with coloured keys, horizontal category labels, the cochlea / marrow ratio in the pair key, confidence intervals and
verification counts on the cell-type panel) and the upgraded quantification of the parabiosis panels: partner-derived
cells counted by the upgraded body rule (960) and immune cells typed on segmented nuclei (961), both checked against
blinded visual labels.

a  BM-cochlear immune channel (D84): two 3D surface renders with the authors' manual annotation (teal) of vessels from
   the marrow onto the cochlea; right, the same views unannotated (930 exports).
b  Neutrophils (green) along vessels (blue) beside the cochlea: 3D overview with the zoomed region and the zoom.
c  Spatial transcriptomics: deconvolved neutrophil state per spot (the authors' map and colours).
d  Neutrophils in the cochlear fluids (D82, D86, D93): FUT4 / CD45 / DAPI images (Olympus FV3000, 60x; 36-um fields of
   the 930 crops; one linear display range per channel) of marrow, perilymph and endolymph preparations, and FUT4 and
   CD45 of every segmented cell over its local background (931; filled, >= 2; bar, median; positive / all above).
e  Parabiosis design (D133): wild type joined to a Lyz2-EGFP partner; GFP+ cells in the wild-type partner are
   partner-derived; one pair per pairing duration.
f  Raw images: 54-day marrow and cochlear lateral wall of the wild-type partner (930 crops of the 931 sections; GFP
   green, FUT4 magenta, DAPI blue; asterisks, FUT4+ cells, all GFP-negative) and perilymph cells of the 28-day
   wild-type and GFP partners (935 crops; GFP green, DAPI blue; circles, cells called GFP+ (yellow) or GFP- (white)).
g  Partner-derived (GFP+) cells per mm2 of marrow and cochlea per wild-type pair (960, upgraded rule; lines), with the
   range of the cochlear value's detection-corrected and brightest-section estimates (vertical bar); log axis; key, the
   cochlea / marrow ratio.
h  Perilymph GFP+ share of nucleated cells, wild-type vs GFP partner (pooled bars; points, fields).
i  Partner-derived (GFP+) share of immune cells of the 54-day wild-type partner by cell type and compartment (961,
   nucleus-anchored typing; cochlear neutrophils from the 60x lateral wall, 931; bar, share; line, Clopper-Pearson 95%
   CI; B cells, which lack the reporter, as control); right, GFP+ / typed cells and typed cells confirmed as single
   marker+ cells in the blinded review.
Inputs: results/r9_figures (900 build_f4: image_panels_log, image_spatial_key, f5a_fut4_cells, f4_parabiont_objects,
f4d_parabiosis_pairs, f4d_perilymph_fields, f4i_celltype_share, sfig_parabiosis_images / _crop_objects) and
figures/panels (930, 935). No statistics are computed here.
Output: figures/figures/fig4_immune_channel_v3.*   Run: COCH_ROOT=/path/to/coch-rythum python scripts/914_fig4_immune_channel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (AXIS, GRID, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, halo,  # noqa: E402
                          minor, text_width)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
PANELS = ROOT / "figures" / "panels"
NAME = "fig4_immune_channel_v3"
GUT = 0.6                                               # gutter between image tiles (mm)
C_COCH, C_MARROW = fd.TISSUE_COLOUR["cochlea"], fd.TISSUE_COLOUR["bm"]
C_WT, C_GFP = "#333333", "#2CA02C"
DAY_COL = {14: "#56B4E9", 31: "#CC79A7", 54: "#000000"}
YELLOW = "#FFD700"


# ------------------------------------------------------------------------------------------------ image helpers
def img_tile(cv, x, y, size, name, um_per_px=None, bar_um=None, frame=False):
    """A square image tile from figures/panels (pixels as exported). Scale bar bottom-right (white, haloed label above,
    right-aligned); frame = dashed white outline marking a zoomed view."""
    a = np.asarray(Image.open(PANELS / f"{name}.png").convert("RGB"))
    ax = cv.axes(x, y, size, size)
    ax.imshow(a, interpolation="none")
    blank(ax)
    if bar_um:
        bar = bar_um / um_per_px / a.shape[1] * size
        bx1, by = x + size - 1.0, y + size - 1.1
        cv.line([bx1 - bar, bx1], [by, by], "#000000", 1.7, solid_capstyle="butt")
        cv.line([bx1 - bar, bx1], [by, by], "#FFFFFF", 0.9, solid_capstyle="butt")
        cv.text(bx1, by - 0.5, f"{bar_um} µm", size=PT_SMALL, color="#FFFFFF", ha="right", va="bottom",
                path_effects=halo(1.1, "#000000"))
    if frame:
        ax.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, transform=ax.transAxes, fill=False, edgecolor="#FFFFFF",
                               linewidth=0.6, linestyle=(0, (2.0, 1.4))))
    return ax


def logaxis(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    dec = [10.0 ** k for k in range(-3, 5) if lo <= 10.0 ** k <= hi]
    ax.yaxis.set_major_locator(FixedLocator(dec))
    ax.set_yticklabels([f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in dec])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------------------------------------ a-c 3D imaging
def panel_imaging(cv, x0, y0, w, h):
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    key = pd.read_csv(R9 / "image_spatial_key.csv")
    cv.letter(x0, y0, "a", "BM-cochlear immune channel")
    top = y0 + 6.2
    s0 = (w - 4 * GUT) / 5                               # unannotated tile; annotated tiles = 2 x s0 + gutter
    s1 = 2 * s0 + GUT
    for k in (1, 2):
        nm = f"vessels_view{k}_annotated"
        img_tile(cv, x0 + (k - 1) * (s1 + GUT), top, s1, nm, lg.loc[nm, "um_per_px"], 100)
    ox = x0 + 2 * (s1 + GUT)
    for k in (1, 2):
        img_tile(cv, ox, top + (k - 1) * (s0 + GUT), s0, f"vessels_view{k}_original")
    fits(ox + s0, x0 + w, "a tiles")
    s2 = (w - 2 * GUT - 0.8) / 3
    top2 = top + s1 + 5.2
    cv.letter(x0, top2 - 5.0, "b", "Neutrophils at the cochlea")
    nm = "marrow_neutrophils_overview"
    ax = img_tile(cv, x0, top2, s2, nm, lg.loc[nm, "um_per_px"], 300)
    bx, by, bs = (int(v) for v in str(lg.loc[nm, "zoom_box_x_y_size_px"]).split(","))
    ax.add_patch(Rectangle((bx, by), bs, bs, fill=False, edgecolor="#FFFFFF", linewidth=0.6, linestyle=(0, (2.0, 1.4))))
    for i, (txt, colour) in enumerate([("Neutrophils", "#3EE63E"), ("Vessels", "#6FB4FF")]):
        cv.text(x0 + s2 - 1.0, top2 + 1.0 + i * 2.2, txt, size=PT_SMALL, color=colour, ha="right", va="top",
                path_effects=halo(1.1, "#000000"))
    nm = "marrow_neutrophils_zoom"
    img_tile(cv, x0 + s2 + GUT, top2, s2, nm, lg.loc[nm, "um_per_px"], 100, frame=True)
    cx = x0 + 2 * (s2 + GUT) + 0.8
    cv.letter(cx - 0.6, top2 - 5.0, "c", "Neutrophil states")
    img_tile(cv, cx, top2, s2, "spatial_neutrophil_states")
    fits(cx + s2, x0 + w + 1e-6, "c tile")
    col = dict(zip(key.state, key.colour))
    for r, pair in enumerate([("Band", "Mature"), ("Activated", "Aged")]):
        xk = cv.key(cx, top2 + s2 + 1.7 + r * 2.4,
                    [([{"kind": "box", "fc": col[st], "ec": AXIS, "lw": 0.3}], st) for st in pair], gap=1.4)
        fits(xk, x0 + w + 1.0, f"c key {r}")
    assert top2 + s2 + 1.7 + 2.4 + 1.0 <= y0 + h + 1.5, "imaging block height"


# ------------------------------------------------------------------------------------------------ d cochlear fluids
FUT4_COLS = [("bone_marrow_f3", "Marrow", "bone_marrow"), ("perilymph_f8", "Perilymph", "perilymph"),
             ("endolymph_f5", "Endolymph", "endolymph")]
FUT4_ROWS = [("DAPI", "DAPI", "#1F3FFF"), ("FUT4", "FUT4", "#008F00"), ("CD45", "CD45", "#B000B0"), ("merge", "Merge", TEXT)]
FIELD_UM = 36.0
FUT4_POS = 2.0                                          # FUT4+ = cell / local background >= 2 (931)


def panel_fluids(cv, x0, y0, w, h, tile):
    cv.letter(x0, y0, "d", "Neutrophils in the cochlear fluids")
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    gap, lab = 0.6, 3.6
    top = y0 + 8.6
    for j, (tag, name, _) in enumerate(FUT4_COLS):
        um = float(lg.loc[f"fut4_{tag}", "um_per_px"])        # per field, from the acquisition sidecar (930)
        x = x0 + lab + j * (tile + gap)
        cv.text(x + tile / 2, top - 0.6, name, size=PT_SMALL, ha="center", va="bottom")
        n = int(round(FIELD_UM / um))                          # central 36-um field of the 48-um crop
        for i, (ch, _, _) in enumerate(FUT4_ROWS):
            yt = top + i * (tile + gap)
            ax = cv.axes(x, yt, tile, tile)
            src = PANELS / f"fut4_{tag}_{ch}.png"
            assert src.exists(), src.name
            a = np.asarray(Image.open(src))
            o = (a.shape[0] - n) // 2
            ax.imshow(a[o:o + n, o:o + n], cmap="gray" if a.ndim == 2 else None, vmin=0, vmax=255,
                      interpolation="none")
            blank(ax)
        if j == len(FUT4_COLS) - 1:
            yb = top + 3 * (tile + gap) + tile - 1.2
            bar = 10.0 / um / n * tile
            cv.line([x + tile - 1.0 - bar, x + tile - 1.0], [yb, yb], "#FFFFFF", 0.9)
            cv.text(x + tile - 1.0, yb - 0.5, "10 µm", size=PT_SMALL, color="#FFFFFF", ha="right", va="bottom",
                    path_effects=halo(1.1, "#000000"))
    gx = x0 + lab + 3 * tile + 2 * gap
    assert top + 4 * tile + 3 * gap <= y0 + h, "d grid height"
    for i, (_, lab_, colour) in enumerate(FUT4_ROWS):        # channel names in the channel colours (micrograph key)
        cv.text(x0 + lab - 1.0, top + i * (tile + gap) + tile / 2, lab_, size=PT_SMALL, rotation=90, ha="center",
                va="center", color=colour, fontweight="bold")
    q = pd.read_csv(R9 / "f5a_fut4_cells.csv")
    qx, qt = gx + 11.6, top + 6.4
    qw, qh = x0 + w - qx, 3 * (tile + gap) - 6.4
    ax = cv.axes(qx, qt, qw, qh)
    yl = (0.4, 50.0)
    for col_ in ("fut4_ratio", "cd45_ratio"):
        assert q[col_].between(*yl).all(), f"d quantification outside the axis ({col_})"
    rng = np.random.default_rng(0)
    marks = (("fut4", FUT4_ROWS[1][2], -0.21), ("cd45", FUT4_ROWS[2][2], 0.21))
    ncol = len(FUT4_COLS)
    for k, (_, name, comp) in enumerate(FUT4_COLS):
        d = q[q.compartment == comp]
        xm = qx + (k + 0.5) / ncol * qw
        for j, (m, colour, off) in enumerate(marks):
            v = d[f"{m}_ratio"]
            pos = v >= FUT4_POS
            assert (pos == d[f"{m}_positive"]).all()
            xj = k + off + rng.uniform(-0.12, 0.12, len(d))
            ax.scatter(xj[pos.values], v[pos], s=2.0, color=colour, linewidths=0, zorder=3)
            ax.scatter(xj[~pos.values], v[~pos], s=2.0, facecolor="none", edgecolor="#777777", linewidths=0.35,
                       zorder=3)
            med = float(v.median())
            ax.plot([k + off - 0.17, k + off + 0.17], [med, med], color=TEXT, linewidth=0.7, zorder=4)
            cnt = f"{int(pos.sum())}/{len(d)}"
            assert text_width(cnt, PT_SMALL) + 0.2 < qw / ncol, "d counts overlap"
            cv.text(xm, qt - 0.5 - (1 - j) * 2.3, cnt, size=PT_SMALL, ha="center", va="bottom")
    for j, (m, colour, _) in enumerate(marks):                # count rows keyed by their marker glyph (black text)
        yk = qt - 1.35 - (1 - j) * 2.3
        cv.mark(qx - 1.4, yk, 6.0, "o", colour, colour)
        cv.text(qx - 2.3, yk, ("FUT4+", "CD45+")[j], size=PT_SMALL, ha="right", va="center")
    assert gx + 0.4 <= qx - 2.3 - text_width("FUT4+", PT_SMALL), "d count key"
    ax.axhline(FUT4_POS, color="#999999", linewidth=0.5, linestyle=(0, (2.0, 1.4)), zorder=1)
    logaxis(ax, *yl)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(["Marrow", "Peri-\nlymph", "Endo-\nlymph"], fontsize=PT_SMALL, linespacing=0.95)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_ylabel("Cell / background (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    fits(qx + qw, x0 + w + 1e-6, "d quantification")
    assert qt + qh + 0.8 + 2 * 2.4 <= y0 + h, "d quantification labels"


# ------------------------------------------------------------------------------------------------ e-i parabiosis
MOUSE_CELLS = [(-1.7, -2.4), (1.7, -2.4), (0.0, 0.2), (-1.7, 2.8), (1.7, 2.8)]   # cell glyphs in the body (mouse units)


def mouse(ax, cx, cy, fill, edge, s=1.0, tail=1, z=0.0):
    """A mouse seen from above, head up, body centred at (cx, cy) in an axes whose units are mm with y pointing down;
    `tail` curls the tail to the right (+1) or left (-1). Ears and tail are drawn behind; head and body are drawn as one
    silhouette (thick outline first, then the fills, so the seam between them disappears)."""
    lw = 0.55

    def P(x, y):
        return (cx + x * s, cy + y * s)
    for side in (-1, 1):
        ax.add_patch(Circle(P(2.55 * side, -7.9), 1.55 * s, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))
        ax.add_patch(Circle(P(2.55 * side, -7.9), 0.85 * s, facecolor="#F2C9D0", edgecolor="none", zorder=z + 0.1))
    C4 = MPath.CURVE4
    tl = MPath([P(0, 5.0), P(0, 8.0), P(1.9 * tail, 8.6), P(1.9 * tail, 10.6), P(1.9 * tail, 12.6), P(0.2 * tail, 12.8),
                P(0.7 * tail, 14.4)], [MPath.MOVETO, C4, C4, C4, C4, C4, C4])
    ax.add_patch(PathPatch(tl, facecolor="none", edgecolor=edge, linewidth=0.8, capstyle="round", zorder=z + 0.2))
    hd = MPath([P(-2.7, -6.4), P(-2.9, -9.3), P(-1.0, -11.5), P(0, -11.7), P(1.0, -11.5), P(2.9, -9.3), P(2.7, -6.4),
                P(2.5, -4.6), P(-2.5, -4.6), P(-2.7, -6.4), P(-2.7, -6.4)],
               [MPath.MOVETO, C4, C4, C4, C4, C4, C4, C4, C4, C4, MPath.CLOSEPOLY])
    for k, (ec, w_) in enumerate(((edge, 2 * lw), ("none", 0.0))):
        ax.add_patch(Ellipse((cx, cy), 7.6 * s, 11.5 * s, facecolor=fill, edgecolor=ec, linewidth=w_, zorder=z + 1 + k))
        ax.add_patch(PathPatch(hd, facecolor=fill, edgecolor=ec, linewidth=w_, zorder=z + 1 + k))
    for side in (-1, 1):
        ax.add_patch(Circle(P(1.15 * side, -9.0), 0.32 * s, facecolor=TEXT, edgecolor="none", zorder=z + 3))
    ax.add_patch(Circle(P(0, -11.45), 0.3 * s, facecolor="#C9707F", edgecolor="none", zorder=z + 3))


def panel_design(cv, x0, y0, w, h):
    """Two mice joined at the flank (sutured skin, shared circulation): in the wild-type partner the cells are
    host-derived (grey) except the partner-derived GFP+ cells (green) that entered from the Lyz2-EGFP partner, whose
    myeloid cells are all GFP+ (green)."""
    cv.letter(x0, y0, "e", "Parabiosis")
    s, ah = 0.9, 24.0
    at = y0 + 4.6
    ax = cv.axes(x0, at, w, ah)
    ax.set_xlim(0, w)
    ax.set_ylim(ah, 0)
    ax.axis("off")
    jx, cy = w / 2, 11.7 * s + 0.3
    half = 3.8 * s - 0.12                                   # bodies overlap slightly at the joined flanks
    cw, cg = jx - half, jx + half
    mouse(ax, cw, cy, "#FFFFFF", C_WT, s, tail=-1, z=0)
    mouse(ax, cg, cy, "#E3F5E3", C_GFP, s, tail=1, z=5)
    for xs_, cols in ((cw, ["#9A9A9A", "#9A9A9A", "#9A9A9A", "#9A9A9A", C_GFP]), (cg, [C_GFP] * 5)):
        for (dx, dy), col in zip(MOUSE_CELLS, cols):
            ax.add_patch(Circle((xs_ + dx * s, cy + dy * s), 0.55 * s, facecolor=col, edgecolor="none", zorder=12))
    ax.plot([jx, jx], [cy - 4.6 * s, cy + 4.6 * s], color=TEXT, linewidth=0.5, zorder=13)          # skin suture
    for yy in np.arange(cy - 4.0 * s, cy + 4.0 * s + 1e-6, 1.6 * s):
        ax.plot([jx - 0.55, jx + 0.55], [yy, yy], color=TEXT, linewidth=0.45, zorder=13)
    # under the joined hips: partner-derived cells reach the wild-type partner through the shared circulation
    ax.add_patch(FancyArrowPatch((cg - 1.0, cy + 6.4 * s), (cw + 1.0, cy + 6.4 * s), connectionstyle="arc3,rad=-0.6",
                                 arrowstyle="-|>", mutation_scale=4.6, linewidth=0.75, color=C_GFP, zorder=14,
                                 shrinkA=0, shrinkB=0))
    cv.text(x0 + jx, at + ah + 0.6, "shared circulation", size=PT_SMALL, ha="center", va="top")
    lab_y = at + cy + 1.0 * s
    cv.text(x0 + cw - half - 1.0, lab_y, "Wild type", size=PT_SMALL, ha="right", va="center")
    cv.text(x0 + cg + half + 1.0, lab_y, "Lyz2-EGFP", size=PT_SMALL, ha="left", va="center")
    fits(x0 + cg + half + 1.0 + text_width("Lyz2-EGFP", PT_SMALL), x0 + w + 0.5, "e right label")
    assert cw - half - 1.0 - text_width("Wild type", PT_SMALL) >= 0, "e left label"
    yk = at + ah + 5.0
    cv.mark(x0 + 1.9, yk, 9.0, "o", C_GFP, C_GFP)
    cv.text(x0 + 3.2, yk, "partner-derived (GFP+)", size=PT_SMALL, va="center")
    cv.mark(x0 + 1.9, yk + 2.5, 9.0, "o", "#9A9A9A", "#9A9A9A")
    cv.text(x0 + 3.2, yk + 2.5, "host-derived", size=PT_SMALL, va="center")
    cv.text(x0 + 1.0, yk + 5.6, "Joined 5–54 days, one pair each", size=PT_SMALL, va="center", color=TEXT2)
    fits(x0 + 1.0 + text_width("Joined 5–54 days, one pair each", PT_SMALL), x0 + w + 0.5, "e note")
    assert yk + 5.6 + 1.2 <= y0 + h + 0.3, "e height"


def panel_raw(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Raw images")
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    ob = pd.read_csv(R9 / "f4_parabiont_objects.csv")
    im = pd.read_csv(R9 / "sfig_parabiosis_images.csv")
    co = pd.read_csv(R9 / "sfig_parabiosis_crop_objects.csv")
    top = y0 + 8.4
    T = min((w - 3 * GUT) / 4, h - 8.4)
    xs = [x0 + k * (T + GUT) for k in range(4)]
    heads = [("Marrow, 54 d", ""), ("Lateral wall, 54 d", "GFP\u2212"), ("Perilymph, WT, 28 d", ""),
             ("Perilymph, GFP partner, 28 d", "")]
    for x, (s, bold) in zip(xs, heads):                    # column headers above the tiles (images stay unlabelled)
        tw = text_width(s, PT_SMALL) + (text_width(" " + bold, PT_SMALL, "bold") if bold else 0.0)
        assert tw <= T + GUT, f"f header {s}"
        cv.text(x + (T - tw) / 2, top - 0.6, s, size=PT_SMALL, ha="left", va="bottom")
        if bold:
            cv.text(x + (T - tw) / 2 + text_width(s + " ", PT_SMALL), top - 0.6, bold, size=PT_SMALL, ha="left",
                    va="bottom", fontweight="bold")
    # 1-2: 54-day marrow and lateral wall (930 crops), asterisks on the lateral wall's FUT4+ cells (931; GFP-negative)
    img_tile(cv, xs[0], top, T, "parabiont_marrow_merge", float(lg.loc["parabiont_marrow", "um_per_px"]), 10)
    img_tile(cv, xs[1], top, T, "parabiont_cochlea_merge", float(lg.loc["parabiont_cochlea", "um_per_px"]), 10)
    assert abs(float(lg.loc["parabiont_marrow", "crop_um"]) - float(lg.loc["parabiont_cochlea", "crop_um"])) < 0.5
    y_0, y_1, x_0, x_1 = (int(v) for v in str(lg.loc["parabiont_cochlea", "crop_y0_y1_x0_x1"]).replace(",", "-")
                          .replace(" ", "").split("-"))
    mm = T / (x_1 - x_0)
    um = float(lg.loc["parabiont_cochlea", "um_per_px"])
    cells = ob[(ob.field == "60x_sl2") & (ob.object == "FUT4+ cell")]
    inside = cells[cells.centroid_x_px.between(x_0, x_1 - 1) & cells.centroid_y_px.between(y_0, y_1 - 1)]
    assert len(inside) >= 1 and not inside.gfp_positive.astype(bool).any(), "f asterisks"
    wall = pd.read_csv(R9 / "f4_parabiont_fields.csv")
    wall = wall[(wall.field == "60x_sl2") & (wall.compartment == "cochlea")].iloc[0]
    assert int(wall.marker_cells) == len(cells) and int(wall.marker_gfp_cells) == 0, "f lateral-wall FUT4+ cells"
    # scale bar and its label: asterisks stay out of this box (D142)
    lab_x0 = xs[1] + T - 1.0 - max(10 / um / (x_1 - x_0) * T, text_width("10 µm", PT_SMALL)) - 1.2
    lab_y0 = top + T - 1.1 - 0.5 - 2.4
    for _, c in inside.iterrows():
        r = np.sqrt(c.area_um2 / np.pi) / um * mm
        cx_, cy_ = xs[1] + (c.centroid_x_px - x_0) * mm, top + (c.centroid_y_px - y_0) * mm
        ax_ = np.clip(cx_ + 0.55 * r + 0.5, xs[1] + 0.8, xs[1] + T - 0.8)
        ay_ = np.clip(cy_ - 0.55 * r - 0.3, top + 1.0, top + T - 1.0)
        if ax_ >= lab_x0 and ay_ >= lab_y0:                            # in the scale-bar zone: mark it on the left
            ax_ = max(cx_ - 0.55 * r - 0.5, xs[1] + 0.8)
            if ax_ >= lab_x0:
                ay_ = lab_y0 - 0.6
        assert not (ax_ >= lab_x0 and ay_ >= lab_y0), "f asterisk in the scale-bar zone"
        cv.text(ax_, ay_, "*", size=PT, color="#FFFFFF", ha="center", va="center", fontweight="bold",
                path_effects=halo(1.1, "#000000"))
    # 3-4: perilymph of the 28-day wild-type and GFP partners (935 crops of Supplementary Fig. 2b; one GFP display range
    # for both partners), nucleated cells circled by their call
    for k, tile in enumerate((3, 4)):
        row = im[(im.figure == "S2") & (im.panel == "b") & (im.tile == tile)].iloc[0]
        a = np.asarray(Image.open(PANELS / row.image_file).convert("RGB"))
        x = xs[2 + k]
        ax = cv.axes(x, top, T, T)
        ax.imshow(a, interpolation="none")
        o = co[(co.figure == "S2") & (co.panel == "b") & (co.tile == tile)]
        for r in o.itertuples():
            rad = max(np.sqrt(r.area_um2 / np.pi) / row.um_per_px * 1.9, 3.0)
            ax.add_patch(Circle((r.x_in_crop_px, r.y_in_crop_px), rad, fill=False,
                                edgecolor=YELLOW if r.gfp_pos else "#FFFFFF", linewidth=0.45))
        ax.set_xlim(-0.5, a.shape[1] - 0.5)
        ax.set_ylim(a.shape[0] - 0.5, -0.5)
        blank(ax)
        bar = 20 / row.um_per_px / a.shape[1] * T
        bx1, by = x + T - 1.0, top + T - 1.1
        cv.line([bx1 - bar, bx1], [by, by], "#000000", 1.7, solid_capstyle="butt")
        cv.line([bx1 - bar, bx1], [by, by], "#FFFFFF", 0.9, solid_capstyle="butt")
        cv.text(bx1, by - 0.5, "20 µm", size=PT_SMALL, color="#FFFFFF", ha="right", va="bottom",
                path_effects=halo(1.1, "#000000"))
    shared = im[(im.figure == "S2") & (im.panel == "b") & (im.tile.isin([3, 4]))].display_range_grey_levels
    assert shared.str.extract(r"gfp ([0-9-]+)")[0].nunique() == 1, "f perilymph GFP display differs between partners"
    fits(xs[3] + T, x0 + w + 1e-6, "f tiles")
    kx = x0 + 3.0 + text_width("Raw images", PT, "bold") + 4.0
    x = cv.key(kx, y0 + 2.6, [([{"kind": "box", "fc": "#008F00"}], "GFP"), ([{"kind": "box", "fc": "#B000B0"}], "FUT4"),
                              ([{"kind": "box", "fc": "#1F3FFF"}], "DAPI")])
    cv.text(x + 2.6, y0 + 2.6, "*", size=PT, ha="left", va="center", fontweight="bold")
    x = x + 2.6 + text_width("*", PT, "bold") + 0.5
    cv.text(x, y0 + 2.6, "FUT4+ GFP\u2212", size=PT_SMALL, ha="left", va="center")
    x += text_width("FUT4+ GFP\u2212", PT_SMALL) + 2.6
    cv.mark(x + 0.9, y0 + 2.6, 10.0, "o", "none", "#C9A800", 0.6)
    cv.text(x + 2.2, y0 + 2.6, "GFP+", size=PT_SMALL, va="center")
    x += 2.2 + text_width("GFP+", PT_SMALL) + 2.0
    cv.mark(x + 0.9, y0 + 2.6, 10.0, "o", "none", "#777777", 0.6)
    cv.text(x + 2.2, y0 + 2.6, "GFP\u2212 (perilymph)", size=PT_SMALL, va="center")
    fits(x + 2.2 + text_width("GFP\u2212 (perilymph)", PT_SMALL), x0 + w + 1e-6, "f key")


ROW_TOP, ROW_BOTTOM = 7.5, 7.5                            # g-i: plot areas share their top and bottom (mm)


def panel_pairs(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Marrow vs cochlea, per pair")
    pr = pd.read_csv(R9 / "f4d_parabiosis_pairs.csv").sort_values("days")
    ax = cv.axes(x0 + 12.0, y0 + ROW_TOP, w - 33.0, h - ROW_TOP - ROW_BOTTOM)
    vals = pr[["marrow_density", "cochlea_density", "cochlea_corrected_density", "cochlea_best_plane_density"]].stack()
    lo_lim = 10.0 ** np.floor(np.log10(vals[vals > 0].min()))
    hi_lim = 10.0 ** np.ceil(np.log10(vals.max()))
    for r in pr.itertuples():
        col = DAY_COL[int(r.days)]
        ax.plot([0, 1], [r.marrow_density, r.cochlea_density], color=col, linewidth=0.8, zorder=2)
        ax.scatter([0, 1], [r.marrow_density, r.cochlea_density], s=10, c=col, linewidths=0, zorder=3)
        lo = min(r.cochlea_density, r.cochlea_corrected_density, r.cochlea_best_plane_density)
        hi = max(r.cochlea_density, r.cochlea_corrected_density, r.cochlea_best_plane_density)
        dx = {14: -0.06, 31: 0.06, 54: 0.0}[int(r.days)]
        ax.plot([1.12 + dx, 1.12 + dx], [lo, hi], color=col, linewidth=0.9, zorder=2, solid_capstyle="butt")
        for v in (lo, hi):                                                # caps, so a short range still reads as one
            ax.plot([1.12 + dx - 0.025, 1.12 + dx + 0.025], [v, v], color=col, linewidth=0.6, zorder=2)
    assert vals.between(lo_lim, hi_lim).all(), "g values outside the axis"
    ax.set_xlim(-0.3, 1.35)
    ax.set_xticks([0, 1], ["Marrow", "Cochlea"], fontsize=PT_SMALL)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    logaxis(ax, lo_lim, hi_lim)
    clean(ax)
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    # key: pair, and its cochlea / marrow ratio by the rule and corrected with the blinded labels (D140, D142)
    kx = x0 + w - 19.4
    c1 = kx + 2.7 + text_width("00 d", PT_SMALL) + 1.0 + text_width("0.00", PT_SMALL)      # right edges of the columns
    c2 = c1 + 1.2 + text_width("verified", PT_SMALL)
    cv.text(c1, y0 + ROW_TOP - 0.4, "rule", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    cv.text(c2, y0 + ROW_TOP - 0.4, "verified", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    cv.text(c2, y0 + ROW_TOP - 2.8, "cochlea / marrow", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    for i, r in enumerate(pr.itertuples()):
        ky = y0 + ROW_TOP + 1.0 + 2.6 * i
        cv.line([kx, kx + 2.2], [ky, ky], DAY_COL[int(r.days)], 0.8)
        cv.text(kx + 2.7, ky, f"{int(r.days)} d", size=PT_SMALL, va="center")
        cv.text(c1, ky, f"{r.cochlea_to_marrow:.2f}", size=PT_SMALL, ha="right", va="center")
        cv.text(c2, ky, f"{r.verified_ratio:.2f}", size=PT_SMALL, ha="right", va="center")
    fits(c2, x0 + w + 0.5, "g key")
    ky = y0 + ROW_TOP + 1.0 + 2.6 * len(pr) + 1.2
    cv.line([kx + 1.1, kx + 1.1], [ky - 1.0, ky + 1.0], TEXT, 0.9)
    cv.text(kx + 2.7, ky, "corrected", size=PT_SMALL, va="center")
    cv.text(kx + 2.7, ky + 2.4, "range", size=PT_SMALL, va="center")
    assert ky + 2.4 + 1.0 <= y0 + h - ROW_BOTTOM + 3.0, "g key height"


def panel_perilymph(cv, x0, y0, w, h):
    cv.letter(x0, y0, "h", "Perilymph cells")
    pf = pd.read_csv(R9 / "f4d_perilymph_fields.csv")
    ax = cv.axes(x0 + 11.0, y0 + ROW_TOP, w - 13.0, h - ROW_TOP - ROW_BOTTOM)
    pos = {("14 d", "WT"): 0, ("14 d", "GFP"): 1, ("28 d", "WT"): 2.4, ("28 d", "GFP"): 3.4}
    rng = np.random.default_rng(1)
    for (an, prt), xx in pos.items():
        g = pf[(pf.animal == an) & (pf.partner == prt)]
        col = C_WT if prt == "WT" else C_GFP
        pooled = 100 * g.gfp_pos.sum() / g.nuclei.sum()
        ax.bar(xx, pooled, width=0.72, color="#E6E6E6", edgecolor=col, linewidth=0.5, zorder=1)
        ax.scatter(xx + rng.uniform(-0.14, 0.14, len(g)), g.gfp_pos_pct, s=6, c=col, linewidths=0, zorder=3)
        ax.text(xx, max(pooled, g.gfp_pos_pct.max()) + 3.0, f"{pooled:.0f}%", fontsize=PT_SMALL, ha="center",
                va="bottom")
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 50, 100])
    minor(ax, x=False, y=True)
    ax.set_xticks(list(pos.values()), ["WT", "GFP", "WT", "GFP"], fontsize=PT_SMALL)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_ylabel("GFP+ (% of nucleated)", fontsize=PT_SMALL, labelpad=0.8)
    for xx, lab in [(0.5, "14 d"), (2.9, "28 d")]:
        cv.text(x0 + 11.0 + (xx + 0.6) / 4.6 * (w - 13.0), y0 + h - ROW_BOTTOM + 3.3, lab, size=PT_SMALL, ha="center",
                va="top")


def panel_celltypes(cv, x0, y0, w, h):
    """i — partner-derived (GFP+) share of immune cells of the 54-day wild-type partner (961, nucleus-anchored typing;
    cochlear neutrophils from the 60x lateral wall, 931), cochlea and marrow; B cells do not carry the reporter and give
    the call's background; right, GFP+ / typed cells and typed cells confirmed as single marker+ cells in the blinded
    review."""
    cv.letter(x0, y0, "i", "Partner-derived share, 54 d")
    fc = pd.read_csv(R9 / "f4i_celltype_share.csv")
    rows = list(fc.itertuples())
    lab_w = max(text_width(r.label, PT_SMALL) for r in rows) + 5.0
    cnt_w = text_width("000/000", PT_SMALL) + 1.4
    ver_w = text_width("verified", PT_SMALL) + 1.2
    pw = w - lab_w - cnt_w - ver_w
    ah = h - ROW_TOP - ROW_BOTTOM
    ax = cv.axes(x0 + lab_w, y0 + ROW_TOP, pw, ah)
    xmax = 30.0
    for k, r in enumerate(rows):
        col = C_COCH if r.compartment == "cochlea" else C_MARROW
        ax.barh(k, r.gfp_pos_pct, height=0.6, color=col, zorder=2)
        hi = min(r.ci_hi_pct, xmax)
        ax.plot([r.ci_lo_pct, hi], [k, k], color=TEXT, linewidth=0.55, zorder=3, solid_capstyle="butt")
        ax.plot([r.ci_lo_pct] * 2, [k - 0.13, k + 0.13], color=TEXT, linewidth=0.55, zorder=3)
        if r.ci_hi_pct > xmax:                                   # interval runs past the axis: arrowhead at the edge
            ax.scatter([xmax], [k], marker=">", s=6, color=TEXT, linewidths=0, zorder=3, clip_on=False)
        else:
            ax.plot([hi] * 2, [k - 0.13, k + 0.13], color=TEXT, linewidth=0.55, zorder=3)
        yy = y0 + ROW_TOP + (k + 0.5) * ah / len(rows)
        cv.text(x0 + lab_w + pw + cnt_w, yy, f"{int(r.gfp_pos_cells)}/{int(r.typed_cells)}", size=PT_SMALL,
                ha="right", va="center")
        cv.text(x0 + w, yy, r.verified, size=PT_SMALL, ha="right", va="center", color=TEXT2)
    cv.text(x0 + lab_w + pw + cnt_w, y0 + ROW_TOP - 0.5, "GFP+", size=PT_SMALL, ha="right", va="bottom")
    cv.text(x0 + w, y0 + ROW_TOP - 0.5, "verified", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r.label for r in rows], fontsize=PT_SMALL)
    n_coch = int((fc.compartment == "cochlea").sum())
    ax.axhline(n_coch - 0.5, color=GRID, linewidth=0.4, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 10, 20, 30])
    minor(ax, x=True, y=False)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.set_xlabel("GFP+ (% of typed cells)", fontsize=PT_SMALL, labelpad=0.8)
    rh = ah / len(rows)
    for (a_, b_), lab, col in (((0, n_coch), "Cochlea", C_COCH), ((n_coch, len(rows)), "Marrow", C_MARROW)):
        ya, yb = y0 + ROW_TOP + a_ * rh + 0.4, y0 + ROW_TOP + b_ * rh - 0.4
        cv.line([x0 + 2.4, x0 + 2.4], [ya, yb], col, 0.8)
        cv.text(x0 + 1.4, (ya + yb) / 2, lab, size=PT_SMALL, ha="center", va="center", rotation=90,
                fontweight="bold")


def main():
    cv = Canvas()
    panel_imaging(cv, 0.0, 0.0, 96.0, 85.0)
    panel_fluids(cv, 98.0, 0.0, 82.0, 85.0, tile=16.0)
    panel_design(cv, 0.0, 87.5, 40.0, 42.0)
    panel_raw(cv, 42.0, 87.5, 138.0, 42.0)
    panel_pairs(cv, 0.0, 132.0, 60.0, 38.0)
    panel_perilymph(cv, 62.0, 132.0, 48.0, 38.0)
    panel_celltypes(cv, 112.0, 132.0, 68.0, 38.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
