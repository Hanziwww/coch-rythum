"""Round 9 (D67) — Fig. 1, built in code as one intricate composite at Nature size (180 x 170 mm, 5-7 pt).

a  Study design (D119): the user's illustration (data/photos/fig1a_workflow/fig1a_workflow_pasted.png, pasted in chat
   2026-10-16), full width. Its small raster text is removed and re-set as vector text >= 5 pt: headings ('Day-night
   sample collection' replaces 'Circadian sample collection'), the bar text, the timeline labels at the atlas sampling
   times ZT3-ZT23 (the illustration read ZT 5 ... ZT 1), the two instrument captions; the mini-charts' ZT tick labels
   are removed (wrong times, unreadable). Icons and their own micro-labels are left as drawn. Until round 10 panel a was
   the 24-h light-dark sampling ring with cells per sample (v6).
b  Joint UMAP of all 160,643 cells coloured by the 35 gold-standard subtypes (class hue, subtype lightness); classes
   named by keylines; dashed windows mark the zooms in d.
c  The same embedding per sampling time: cells of that sample coloured by tissue over all cells in grey.
d  Zoomed windows of the joint UMAP (i B / T / NK, ii myeloid and progenitors) with every subtype named.
e  One marker per subtype (structural | haematopoietic): dot area = fraction expressing, colour = scaled mean.
f  Atlas card, columns = subtypes in dendrogram order: tree, class, cells, tissue and ZT shares, marker z-scores in four
   external cochlear atlases (the per-sample composition rows of v6 are dropped to make room for a, D119; per-sample
   counts and composition are in ED1b and ED1f).
g  Spatial check (control GeoMx): deconvolved share in neuron vs supporting-cell regions (types with q < 0.05).

Inputs (registered CSVs, no recomputation of statistics): results/r9_figures/{f1b_umap_joint, palette_celltypes,
f1c_markers_structural, f1d_markers_haematopoietic, ed2_atlas_columns, ed2_atlas_tree, ed2_atlas_validation,
f1f_spatial_geomx}.csv and the study-design illustration of a.
Display-only steps: random draw order of points (seed 7); removal of the illustration's raster text (boxes below).
Outputs: figures/figures/fig1_atlas_v7.{pdf,svg,jpg,png}.
Run: python scripts/910_fig1_atlas.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (AXIS, LW, LW_THIN, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean,  # noqa: E402
                          halo)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "fig1_atlas_v7"
SEED = 7
ZT = [3, 7, 11, 15, 19, 23]
ZT_COLOUR = {3: "#FFE8A3", 7: "#FFD25E", 11: "#F2B134", 15: "#8C8C8C", 19: "#5E5E5E", 23: "#2E2E2E"}
TISSUE = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
PURPLES = LinearSegmentedColormap.from_list("p", ["#F2F0F7", "#CBC9E2", "#9E9AC8", "#756BB1", "#54278F"])
GREYS = LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#D9D9D9", "#969696", "#525252", "#000000"])
EMPTY = "#EFEFEF"
WIN = {"i": ((-7.8, 8.0), (10.6, 21.4)), "ii": ((-13.0, 9.0), (-3.6, 13.2))}
WIN_TYPES = {"i": ["B_Pro_Cyc", "B_Pre", "B_Trans", "B_Mat", "B_Naive", "T_Cell", "NK"],
             "ii": ["HSPC", "Cycling", "Erythro", "GMP", "Neu_Band", "Neu_Mat", "Neu_Aged", "Neu_Act", "Mono_Class",
                    "Mono_NC", "Mac", "pDC", "Mast"]}
WIN_TITLE = {"i": "i  Lymphoid", "ii": "ii  Myeloid and progenitors"}


def fmt(n):
    return f"{int(n):,}"


# ------------------------------------------------------------------------------------------------ data
def load():
    pal = pd.read_csv(R9 / "palette_celltypes.csv").sort_values("order")
    u = pd.read_csv(R9 / "f1b_umap_joint.csv", usecols=["umap_1", "umap_2", "abbr", "cell_class", "tissue", "zt"])
    u = u.sample(frac=1.0, random_state=SEED).reset_index(drop=True)          # draw order only
    u["colour"] = u["abbr"].map(dict(zip(pal["abbr"], pal["colour"])))
    assert u["colour"].notna().all() and len(u) == 160643
    global SX
    SX = ((u.umap_2.max() - u.umap_2.min() + 0.8) / (u.umap_1.max() - u.umap_1.min() + 0.8))   # D143: square frame
    return pal, u


# ------------------------------------------------------------------------------------------------ a design
BANNER = ROOT / "data" / "photos" / "fig1a_workflow" / "fig1a_workflow_pasted.png"   # the user's illustration (D119)
BANNER_X = (2, 1995)        # columns kept: the grey frame lines at both edges (and their compression rim) dropped
# Raster text removed from the illustration (image px: x0, y0, x1, y1) and re-set below as vector text >= 5 pt: the
# four headings, the timeline labels (read ZT 5 ... ZT 1; the atlas was sampled at ZT3 ... ZT23), the mini-charts' ZT
# tick labels (removed, not re-set) and the confocal caption. Text on coloured ground is refilled from its background.
BANNER_WHITE = [(100, 0, 330, 21), (790, 0, 896, 21), (745, 186, 936, 209), (1578, 0, 1666, 19),
                (12, 108, 408, 125), (960, 324, 1084, 343), (1871, 129, 1998, 146), (1352, 202, 1478, 219)]
BANNER_BAR = (116, 43, 318, 64)       # '6 time points every 4 hours' in the light-dark bar: refilled column by column
BANNER_FACE = (1834, 177, 1926, 196)  # 'Flow Cytometer' on the instrument face: refilled with the face grey
BANNER_ARROWS = (35.0, 105.5, 175.75, 246.25, 316.25, 386.25)   # x of the six sampling arrows (image px)
BANNER_TEXT = [                       # (text, x px, y px, size, vertical alignment); centred horizontally
    ("Day–night sample collection", 212.5, 15.5, PT, "baseline"),
    ("scRNA-seq", 842.5, 15.5, PT, "baseline"),
    ("Bioinformatics analysis", 840.5, 203.0, PT, "baseline"),
    ("Validation", 1622.0, 15.5, PT, "baseline"),
    ("6 time points, every 4 hours", 213.0, 53.5, PT_SMALL, "center"),
    ("Confocal microscopy", 1414.5, 210.5, PT_SMALL, "center"),
    ("Flow cytometry", 1835.0, 370.0, PT_SMALL, "center"),
] + [(f"ZT{z}", x, 116.5, PT_SMALL, "center") for z, x in zip(ZT, BANNER_ARROWS)]


def banner():
    """The study-design illustration with its small raster text removed (display only)."""
    a = np.asarray(Image.open(BANNER).convert("RGB")).copy()
    assert a.shape == (383, 2000, 3), a.shape
    for x0, y0, x1, y1 in BANNER_WHITE:
        a[y0:y1, x0:x1] = 255
    x0, y0, x1, y1 = BANNER_BAR                      # the bar's colour depends on x only: use the rows above the text
    a[y0:y1, x0:x1] = np.median(a[37:42, x0:x1], axis=0).astype(a.dtype)[None, :, :]
    x0, y0, x1, y1 = BANNER_FACE
    ground = np.concatenate([a[y0 - 6:y0 - 2, x0:x1].reshape(-1, 3), a[y1 + 2:y1 + 6, x0:x1].reshape(-1, 3)])
    a[y0:y1, x0:x1] = np.median(ground, axis=0).astype(a.dtype)
    return a[:, BANNER_X[0]:BANNER_X[1]]


def panel_a(cv, x0, y0, w):
    """Full-width illustration; returns its bottom edge (mm)."""
    cv.letter(x0, y0, "a")
    img = banner()
    s = w / img.shape[1]                              # mm per image pixel
    top = y0 + 1.0                                    # room above the image for the re-set headings
    ax = cv.axes(x0, top, w, img.shape[0] * s)
    ax.imshow(img, interpolation="none")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    blank(ax)
    for txt, px, py, size, va in BANNER_TEXT:
        cv.text(x0 + (px - BANNER_X[0]) * s, top + py * s, txt, size=size, ha="center", va=va, color=TEXT)
    return top + img.shape[0] * s


# ------------------------------------------------------------------------------------------------ helpers
SX = 1.0           # x stretch of the joint embedding (D143: square frame); set in load() from the joint frame


def umap_axes(cv, x0, y0, w, h, xlim, ylim, sx=None):
    """Embedding axes centred in the box (x0, y0, w, h) mm, x scaled sx times y (D143: the joint frame is shown 1:1);
    returns the axes, its mm rectangle and the scales (mm per unit in x, in y)."""
    sx = SX if sx is None else sx
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    k = min(w / (dx * sx), h / dy)
    aw, ah = dx * sx * k, dy * k
    ax0, ay0 = x0 + (w - aw) / 2, y0 + (h - ah) / 2
    ax = cv.axes(ax0, ay0, aw, ah)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    blank(ax)
    return ax, (ax0, ay0, aw, ah), (k * sx, k)


def to_mm(rect, xlim, ylim, k, x, y):
    ax0, ay0, aw, ah = rect
    kx, ky = k if isinstance(k, tuple) else (k, k)
    return ax0 + (np.asarray(x) - xlim[0]) * kx, ay0 + ah - (np.asarray(y) - ylim[0]) * ky


def spread(vals, lo, hi, gap):
    """Shift sorted label positions apart by at least gap within [lo, hi] (order preserved)."""
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order].copy()
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + gap)
    if len(v) and v[-1] > hi:
        v -= v[-1] - hi
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], v[i + 1] - gap)
    v = np.clip(v, lo, hi)
    out = np.empty_like(v)
    out[order] = v
    return out


# ------------------------------------------------------------------------------------------------ b joint UMAP
def panel_b(cv, u, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Joint atlas")
    xlim = (u.umap_1.min() - 0.4, u.umap_1.max() + 0.4)
    ylim = (u.umap_2.min() - 0.4, u.umap_2.max() + 0.4)
    ax, rect, k = umap_axes(cv, x0 + 15.5, y0 + 4.5, w - 27.5, h - 5.0, xlim, ylim)
    ax.scatter(u.umap_1, u.umap_2, s=0.55, c=u.colour, linewidths=0, rasterized=True)
    for key, ((xa, xb), (ya, yb)) in WIN.items():
        ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, linestyle=(0, (2.2, 1.4)), linewidth=0.6,
                               edgecolor=TEXT, zorder=4))
        ax.text(xa + 0.3, yb - 0.25, key, ha="left", va="top", fontsize=PT_SMALL, fontweight="bold", zorder=5,
                path_effects=halo(1.0))
    ax0, ay0, aw, ah = rect
    # class keylines: label on the nearer side, line to the class cell closest to the label
    cls = u.groupby("cell_class").agg(x=("umap_1", "median"), y=("umap_2", "median"))
    mx, my = to_mm(rect, xlim, ylim, k, cls.x, cls.y)
    cls["mx"], cls["my"] = mx, my
    centre = ax0 + aw / 2
    cls["side"] = np.where(cls.mx < centre, "L", "R")
    for side, g in cls.groupby("side"):
        ys = spread(g.my.values, y0 + 6.0, y0 + h - 1.5, 3.0)
        for (c, r), ly in zip(g.iterrows(), ys):
            lx = ax0 - 2.0 if side == "L" else ax0 + aw + 2.0
            pts = u.loc[u.cell_class == c, ["umap_1", "umap_2"]]
            px, py = to_mm(rect, xlim, ylim, k, pts.umap_1.values, pts.umap_2.values)
            j = np.argmin((px - lx) ** 2 + (py - ly) ** 2)
            cv.fig.add_artist(_line(cv, lx, ly, px[j], py[j]))
            sq = 1.5
            sx = lx - sq - 0.5 if side == "L" else lx + 0.5
            cv.fig.patches.append(Rectangle((sx / cv.w, 1 - (ly + sq / 2) / cv.h), sq / cv.w, sq / cv.h,
                                            transform=cv.fig.transFigure, facecolor=fd.CLASS_COLOUR[c],
                                            edgecolor="none"))
            tx = sx - 0.6 if side == "L" else sx + sq + 0.6
            cv.text(tx, ly, c, size=PT_SMALL, va="center", ha="right" if side == "L" else "left")
    return rect, xlim, ylim, k


def _line(cv, xa, ya, xb, yb):
    from matplotlib.lines import Line2D
    return Line2D([xa / cv.w, xb / cv.w], [1 - ya / cv.h, 1 - yb / cv.h], transform=cv.fig.transFigure,
                  color=TEXT2, linewidth=LW_THIN)


def _arrow(cv, xa, ya, xb, yb):
    from matplotlib.patches import FancyArrowPatch
    return FancyArrowPatch((xa / cv.w, 1 - ya / cv.h), (xb / cv.w, 1 - yb / cv.h), transform=cv.fig.transFigure,
                           arrowstyle="-|>,head_length=1.2,head_width=0.6", mutation_scale=1, color=TEXT2,
                           linewidth=LW_THIN)


# ------------------------------------------------------------------------------------------------ c time x tissue
def panel_c(cv, u, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Every subtype at every time")
    xlim = (u.umap_1.min() - 0.4, u.umap_1.max() + 0.4)
    ylim = (u.umap_2.min() - 0.4, u.umap_2.max() + 0.4)
    cw, gap = (w - 2 * 2.0) / 3, 2.0
    rh = (h - 9.0) / 2
    bg = u.sample(n=40000, random_state=SEED)
    for i, z in enumerate(ZT):
        r, c = divmod(i, 3)
        bx, by = x0 + c * (cw + gap), y0 + 4.5 + r * rh
        ax, rect, k = umap_axes(cv, bx, by + 3.0, cw, rh - 3.2, xlim, ylim)
        ax.scatter(bg.umap_1, bg.umap_2, s=0.06, c="#DADADA", linewidths=0, rasterized=True)
        sub = u[u.zt == z]
        ax.scatter(sub.umap_1, sub.umap_2, s=0.09, c=sub.tissue.map({t: v[1] for t, v in TISSUE.items()}),
                   linewidths=0, rasterized=True)
        sw = 1.4
        cv.fig.patches.append(Rectangle((bx / cv.w, 1 - (by + 1.3 + sw / 2) / cv.h), sw / cv.w, sw / cv.h,
                                        transform=cv.fig.transFigure, facecolor=ZT_COLOUR[z], edgecolor=TEXT,
                                        linewidth=0.25))
        cv.text(bx + sw + 0.6, by + 1.3, f"ZT{z} · {(z + 8) % 24:02d}:00", size=PT_SMALL, va="center")
        cv.text(bx + cw - 0.5, by + 1.3, fmt(len(sub)), size=PT_SMALL, va="center", ha="right", color=TEXT2)
    ly = y0 + h - 1.6
    lx = x0
    for t in ["cochlea", "bm"]:
        lab, col = TISSUE[t]
        cv.fig.patches.append(Rectangle((lx / cv.w, 1 - (ly + 0.7) / cv.h), 1.4 / cv.w, 1.4 / cv.h,
                                        transform=cv.fig.transFigure, facecolor=col, edgecolor="none"))
        cv.text(lx + 2.0, ly, lab, size=PT_SMALL, va="center")
        lx += 13.0
    cv.fig.patches.append(Rectangle((lx / cv.w, 1 - (ly + 0.7) / cv.h), 1.4 / cv.w, 1.4 / cv.h,
                                    transform=cv.fig.transFigure, facecolor="#DADADA", edgecolor="none"))
    cv.text(lx + 2.0, ly, "other samples", size=PT_SMALL, va="center")


# ------------------------------------------------------------------------------------------------ d zooms
def panel_d(cv, u, pal, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Immune subtypes")
    asp = {k: SX * (v[0][1] - v[0][0]) / (v[1][1] - v[1][0]) for k, v in WIN.items()}
    hh = (w - 3.0) / (asp["i"] + asp["ii"])
    hh = min(hh, h - 8.0)
    bx = x0
    colour = dict(zip(pal["abbr"], pal["colour"]))
    for key in ["i", "ii"]:
        (xa, xb), (ya, yb) = WIN[key]
        ww = hh * asp[key]
        cv.text(bx, y0 + 5.2, WIN_TITLE[key], size=PT_SMALL, va="top")
        ax, rect, k = umap_axes(cv, bx, y0 + 8.0, ww, hh, (xa, xb), (ya, yb))
        for sp in ["top", "bottom", "left", "right"]:
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_linestyle((0, (2.2, 1.4)))
            ax.spines[sp].set_linewidth(0.6)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        inwin = u[(u.umap_1 >= xa) & (u.umap_1 <= xb) & (u.umap_2 >= ya) & (u.umap_2 <= yb)]
        own = inwin.abbr.isin(WIN_TYPES[key])
        ax.scatter(inwin.umap_1[~own], inwin.umap_2[~own], s=0.9, c="#E3E3E3", linewidths=0, rasterized=True)
        ax.scatter(inwin.umap_1[own], inwin.umap_2[own], s=0.9, c=inwin.colour[own], linewidths=0, rasterized=True)
        cen = inwin[own].groupby("abbr").agg(x=("umap_1", "median"), y=("umap_2", "median"))
        mx, my = to_mm(rect, (xa, xb), (ya, yb), k, cen.x, cen.y)
        lab = pd.DataFrame({"x": mx, "y": my}, index=cen.index)
        lab = _repel(lab, rect)
        for a, r in lab.iterrows():
            tx, ty = r.x, r.y
            cv.text(tx, ty, a, size=PT_SMALL, ha="center", va="center", color=TEXT,
                    path_effects=halo(1.1))
            dot = to_mm(rect, (xa, xb), (ya, yb), k, cen.loc[a, "x"], cen.loc[a, "y"])
            if np.hypot(dot[0] - tx, dot[1] - ty) > 1.6:
                cv.fig.add_artist(_line(cv, dot[0], dot[1], tx, ty))
        bx += ww + 3.0


def _repel(lab, rect, iters=300, pad=(4.6, 1.9)):
    """Push label centres apart (mm) until their boxes (pad = half-width, half-height) do not overlap; keep inside."""
    ax0, ay0, aw, ah = rect
    p = lab[["x", "y"]].to_numpy(float).copy()
    half = np.array([[0.5 * len(s) * 0.97 + 0.7, pad[1]] for s in lab.index])
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx, dy = p[j] - p[i]
                ox = half[i, 0] + half[j, 0] - abs(dx)
                oy = half[i, 1] + half[j, 1] - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if oy < ox:
                        s = np.sign(dy) or 1.0
                        p[i, 1] -= s * oy / 2 + 0.05
                        p[j, 1] += s * oy / 2 + 0.05
                    else:
                        s = np.sign(dx) or 1.0
                        p[i, 0] -= s * ox / 2 + 0.05
                        p[j, 0] += s * ox / 2 + 0.05
        p[:, 0] = np.clip(p[:, 0], ax0 + half[:, 0] + 0.3, ax0 + aw - half[:, 0] - 0.3)
        p[:, 1] = np.clip(p[:, 1], ay0 + half[:, 1] + 0.3, ay0 + ah - half[:, 1] - 0.3)
        if not moved:
            break
    return pd.DataFrame(p, index=lab.index, columns=["x", "y"])


# ------------------------------------------------------------------------------------------------ e markers
def panel_e(cv, x0, y0, w, h, p=1.62):
    cv.letter(x0, y0, "e", "One marker per subtype")
    q, lab = 1.72, 10.6                                # row pitch p, column pitch, row-label width (mm)
    top = y0 + 11.0                                    # first row (gene labels above)
    bx = x0
    blocks = [("f1c_markers_structural.csv", "Structural"), ("f1d_markers_haematopoietic.csv", "Haematopoietic")]
    for fname, title in blocks:
        m = pd.read_csv(R9 / fname)
        types = m.drop_duplicates("abbr").sort_values("column_order")
        genes = m.drop_duplicates("gene").sort_values("row_order")
        nr, nc = len(types), len(genes)
        gx0 = bx + lab
        ax = cv.axes(gx0, top, nc * q, nr * p)
        ax.set_xlim(-0.5, nc - 0.5)
        ax.set_ylim(nr - 0.5, -0.5)
        blank(ax)
        ax.add_patch(Rectangle((-0.5, -0.5), nc, nr, fill=False, edgecolor=GRID_E, linewidth=0.3))
        col_of = {g: i for i, g in enumerate(genes.gene)}
        row_of = {a: i for i, a in enumerate(types.abbr)}
        size = (np.sqrt(m.frac_expressing.clip(0, 1)) * 1.45 / 25.4 * 72) ** 2      # area ~ fraction, max 1.45 mm
        ax.scatter(m.gene.map(col_of), m.abbr.map(row_of), s=size, c=PURPLES(m.scaled_mean.clip(0, 1)),
                   linewidths=0, zorder=2)
        for a, i in row_of.items():
            cv.text(gx0 - 1.25, top + (i + 0.5) * p, a, size=PT_SMALL, va="center", ha="right")
            cv.fig.patches.append(Rectangle(((gx0 - 1.05) / cv.w, 1 - (top + (i + 1) * p - 0.1) / cv.h), 0.7 / cv.w,
                                            (p - 0.2) / cv.h, transform=cv.fig.transFigure,
                                            facecolor=types.set_index("abbr").loc[a, "colour"], edgecolor="none"))
        for g, j in col_of.items():
            cv.text(gx0 + (j + 0.5) * q, top - 0.5, g, size=PT_SMALL, va="bottom", ha="center", rotation=90,
                    style="italic")
        bx = gx0 + nc * q + 3.2
    # keys: dot area and colour
    kx, ky = x0 + 1.0, y0 + h - 7.6
    cv.text(kx, ky, "Fraction expressing", size=PT_SMALL, va="top")
    axk = cv.axes(kx, ky + 2.4, 19, 3.2)
    blank(axk)
    axk.set_xlim(0, 19)
    axk.set_ylim(0, 3.2)
    for i, f in enumerate([0.25, 0.5, 1.0]):
        d = np.sqrt(f) * 1.45
        axk.scatter([1.0 + i * 6.4], [1.6], s=(d / 25.4 * 72) ** 2, c="#7F7F7F", linewidths=0)
        axk.text(2.1 + i * 6.4, 1.6, f"{f:g}", fontsize=PT_SMALL, va="center")
    cx = kx + 21.5
    cv.text(cx, ky, "Scaled mean", size=PT_SMALL, va="top")
    axc = cv.axes(cx, ky + 2.9, 12, 1.3)
    axc.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=PURPLES, extent=(0, 1, 0, 1))
    axc.set_yticks([])
    axc.set_xticks([0, 1], ["0", "1"])
    axc.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL)
    for sp in axc.spines.values():
        sp.set_linewidth(0.3)


GRID_E = "#CFCFCF"


# ------------------------------------------------------------------------------------------------ f atlas card
def _tree(ax, tree, leaves):
    pos = {lf: i for i, lf in enumerate(leaves)}
    kids = {r.branch_id: (r.left, r.right, r.height) for r in tree.itertuples()}
    root = tree["root"].iloc[0]

    def walk(n):
        if n not in kids:
            return pos[n], 0.0
        a, b, hgt = kids[n]
        xa, ha = walk(a)
        xb, hb = walk(b)
        ax.plot([xa, xa, xb, xb], [ha, hgt, hgt, hb], color=AXIS, linewidth=0.35, solid_capstyle="butt")
        return (xa + xb) / 2, hgt
    walk(root)


def panel_f(cv, pal, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Atlas card")
    col = pd.read_csv(R9 / "ed2_atlas_columns.csv").sort_values("column_order")
    tree = pd.read_csv(R9 / "ed2_atlas_tree.csv")
    val = pd.read_csv(R9 / "ed2_atlas_validation.csv")
    types = list(col.cell_type)
    n = len(types)
    lab_w, keys_w = 19.5, 8.0
    cw = (w - lab_w - keys_w) / n
    gx0 = x0 + lab_w
    colour = dict(zip(pal["cell_type"], pal["colour"]))
    y = y0 + 4.2
    rows = {}

    def track(name, hgt, gap=0.7):
        nonlocal y
        rows[name] = (y, hgt)
        y += hgt + gap
        return rows[name]

    def label(name, txt, size=PT_SMALL):
        ty, th = rows[name]
        cv.text(gx0 - 1.0, ty + th / 2, txt, size=size, va="center", ha="right")

    # dendrogram
    ty, th = track("tree", 4.6, 0.4)
    ax = cv.axes(gx0, ty, n * cw, th)
    _tree(ax, tree, types)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, tree.height.max() * 1.02)
    blank(ax)
    label("tree", "1 − Pearson r")
    # class strip
    ty, th = track("class", 1.5)
    for i, t in enumerate(types):
        cv.fig.patches.append(Rectangle(((gx0 + i * cw + 0.06) / cv.w, 1 - (ty + th) / cv.h), (cw - 0.12) / cv.w,
                                        th / cv.w * cv.w / cv.h, transform=cv.fig.transFigure,
                                        facecolor=fd.CLASS_COLOUR[col.cell_class.iloc[i]], edgecolor="none"))
    label("class", "Class")
    # cells per subtype (log10)
    ty, th = track("cells", 5.6, 0.9)
    ax = cv.axes(gx0, ty, n * cw, th)
    ax.bar(np.arange(n), np.log10(col.total), width=0.72, color=[colour[t] for t in types], linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(2, 4.3)
    ax.set_yticks([2, 3, 4], ["10²", "10³", "10⁴"])
    ax.tick_params(axis="y", labelsize=PT_SMALL, length=1.0, pad=0.6)
    clean(ax, bottom=False)
    cv.text(gx0 - 5.2, ty + th / 2, "Cells", size=PT_SMALL, va="center", ha="right")
    # tissue share
    ty, th = track("tissue", 2.4)
    ax = cv.axes(gx0, ty, n * cw, th)
    fr = col.n_cochlea / col.total
    ax.bar(np.arange(n), fr, width=0.86, color=TISSUE["cochlea"][1], linewidth=0)
    ax.bar(np.arange(n), 1 - fr, bottom=fr, width=0.86, color=TISSUE["bm"][1], linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    blank(ax)
    label("tissue", "Tissue share")
    # ZT share
    ty, th = track("zt", 2.4, 2.2)
    ax = cv.axes(gx0, ty, n * cw, th)
    bottom = np.zeros(n)
    for z in ZT:
        f = col[f"n_zt{z}"] / col.total
        ax.bar(np.arange(n), f, bottom=bottom, width=0.86, color=ZT_COLOUR[z], linewidth=0)
        bottom += f
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    blank(ax)
    label("zt", "ZT share")
    # external atlases (marker z); per-sample composition rows dropped in v7 (D119; ED1b, ED1f)
    he = 1.75
    ty, th = track("ext", 4 * he, 0.8)
    cv.text(x0 + 0.3, ty - 0.35, "External atlases", size=PT_SMALL, va="bottom")
    ax = cv.axes(gx0, ty, n * cw, th)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(th, 0)
    blank(ax)
    atl = val.drop_duplicates("atlas").sort_values("atlas_order").atlas.tolist()
    vv = val.set_index(["atlas", "cell_type"]).value
    for r_i, a in enumerate(atl):
        for i, ct_ in enumerate(types):
            v = vv.get((a, ct_), np.nan)
            fc = GREYS(np.clip((v + 1) / 10, 0, 1)) if np.isfinite(v) else EMPTY
            ax.add_patch(Rectangle((i - 0.47, r_i * he + 0.05), 0.94, he - 0.1, facecolor=fc, edgecolor="none"))
        cv.text(gx0 - 1.0, ty + r_i * he + he / 2, a, size=PT_SMALL, va="center", ha="right")
    # column labels
    for i, t in enumerate(types):
        cv.text(gx0 + (i + 0.5) * cw, y + 0.2, col.abbr.iloc[i], size=PT_SMALL, va="top", ha="center", rotation=90)
    # colour key (right)
    kx = gx0 + n * cw + 2.2
    ty, th = rows["ext"]
    _cbar(cv, kx, ty + 0.2, 1.4, th - 0.4, GREYS, ["9", "", "−1"], "")
    cv.text(kx + 4.6, ty + th / 2, "z", size=PT_SMALL, va="center", style="italic")


def _cbar(cv, x, y, w, h, cmap, ticks, title):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.linspace(1, 0, 100)[:, None], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([1, 0.5, 0][:len(ticks)], ticks)
    ax.yaxis.tick_right()
    ax.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    if title:
        cv.text(x - 0.2, y - 0.5, title, size=PT_SMALL, va="bottom", ha="left")


# ------------------------------------------------------------------------------------------------ g spatial
def panel_g(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Spatial (GeoMx)")
    s = pd.read_csv(R9 / "f1f_spatial_geomx.csv").sort_values("mean_prop_neuron_control", ascending=True)
    s = s.sort_values("diff_pp")
    lab_w = 9.5
    ax = cv.axes(x0 + lab_w, y0 + 15.0, w - lab_w - 1.0, h - 21.5)
    n = len(s)
    grp = {"structural": fd.GROUP_COLOUR["structural"], "haematopoietic": fd.GROUP_COLOUR["haematopoietic"]}
    for i, r in enumerate(s.itertuples()):
        a, b = 100 * r.mean_prop_supporting_control, 100 * r.mean_prop_neuron_control
        c = grp[r.group]
        ax.plot([a, b], [i, i], color=c, linewidth=0.7, zorder=1)
        ax.scatter([b], [i], s=7, color=c, zorder=2, linewidths=0)
        ax.scatter([a], [i], s=7, facecolor="white", edgecolor=c, linewidths=0.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 80)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(n), s.abbr)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0, pad=1.0, labelsize=PT_SMALL)
    ax.tick_params(axis="x", labelsize=PT_SMALL)
    ax.set_xlabel("Deconvolved share (%)", fontsize=PT_SMALL, labelpad=0.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ky = y0 + 9.3
    axk = cv.axes(x0 + 1.0, ky - 1.2, 3.0, 2.4)
    blank(axk)
    axk.set_xlim(0, 3)
    axk.set_ylim(0, 2.4)
    axk.scatter([0.8], [1.2], s=7, color=TEXT2, linewidths=0)
    cv.text(x0 + 3.8, ky, "Neuron region", size=PT_SMALL, va="center")
    axk2 = cv.axes(x0 + 1.0, ky + 1.3, 3.0, 2.4)
    blank(axk2)
    axk2.set_xlim(0, 3)
    axk2.set_ylim(0, 2.4)
    axk2.scatter([0.8], [1.2], s=7, facecolor="white", edgecolor=TEXT2, linewidths=0.5)
    cv.text(x0 + 3.8, ky + 2.5, "Supporting region", size=PT_SMALL, va="center")


# ------------------------------------------------------------------------------------------------ main
def main():
    pal, u = load()
    cv = Canvas()
    gap = 1.5
    r2 = panel_a(cv, 0.0, 0.0, cv.w) + gap            # a: full-width illustration (D119); b-g compressed below it
    panel_b(cv, u, 0.0, r2, 93.0, 42.5)
    panel_c(cv, u, 95.0, r2, 85.0, 42.5)
    r3 = r2 + 42.5 + gap
    panel_d(cv, u, pal, 0.0, r3, 91.5, 42.5)
    panel_e(cv, 94.0, r3, 86.0, 42.5, p=1.48)
    r4 = r3 + 42.5 + gap
    panel_f(cv, pal, 0.0, r4, 148.0, cv.h - r4)
    panel_g(cv, 150.0, r4, 30.0, cv.h - r4)
    paths = cv.save(OUT, NAME)
    for p in paths:
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
