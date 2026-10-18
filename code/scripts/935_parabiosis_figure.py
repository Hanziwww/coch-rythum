"""Round 10 (D133) — Supplementary Figs 1-3: the parabiosis survey (934), raw images first, then the statistics
(the user asked that the figures show the original images followed by their analyses). 180 mm wide, 5-7 pt.

Supplementary Fig. 1 — partner-derived (GFP+) cells across tissues of the wild-type partners:
a  raw images: one full optical section per animal and tissue (marrow 5, 7, 14, 31 and 54 d; vestibular bone 54 d;
   cochlea 14 d (whole section and a 63x stria section), 31 d, 54 d (lateral wall, duct); liver 31 d); for z-stacks
   the analysed section with the strongest GFP signal (as in the depth analysis of 934);
b  GFP channel of 100-um crops (marrow and cochlea of the 14-, 31- and 54-day pairs) with the GFP+ bodies of 934 kept
   (yellow) or rejected (grey) by the upgraded cell rule (960 / 963, D143); GFP displayed linearly from the section's
   tissue median to the larger of the crop's 99.9th percentile and the section's detection threshold;
c  partner-derived cells (upgraded rule) per mm2 per field (points; colour, pairing duration; fields without a cell at
   the axis floor) and pooled per animal (diamonds);
d  pooled marrow and cochlear densities per pair (lines) with the exact sign test across pairs;
e  cochlea-to-marrow ratio per pair as observed, detection-corrected, in each acquisition's brightest section and
   label-corrected with the blinded labels (960; bootstrap 95% CI, drawn to the axis floor when it reaches 0).
(The typed immune cells of the 54-day partner are Supplementary Fig. 4.)
Supplementary Fig. 2 — perilymph: a raw full fields (GFP green, DAPI blue) of the wild-type and GFP partners at 14
and 28 d; b 150-um crops with the nuclei called GFP+ (yellow) or GFP- (white); c GFP+ share per field (points, size =
nucleated cells) and pooled (bar); d cytoplasmic-ring GFP over local background per cell with the pre-set rule;
e GFP sensitivity of each wild-type field relative to the same pair's GFP-partner fields.
Supplementary Fig. 3 — the noise-exposed pair (31 d), counted by the upgraded rule as in Fig. 4g: a raw full fields
of its marrow and cochlea and of the unexposed pairs above GFP crops with the kept / rejected bodies; b densities per
optical section with pooled values; c pair level (noise pair among the unexposed pairs); d noise / unexposed ratios;
e perilymph GFP+ share of every partner (the noise session's GFP partner shows the failed detection).
Display: each channel linear between its 0.5th and 99.9th percentile in the shown optical section (crops keep the
range of their full section, except the GFP crops of S1b and S3b, as above); acquisition settings were set per sample
(934), so intensities are not compared between tiles. Every displayed image is saved as shown (RGB, native pixels) to figures/panels/pb_<figure><panel>_<k>.png and
logged (acquisition, plane, crop, display range) in results/r9_figures/sfig_parabiosis_images.csv; circled objects in
sfig_parabiosis_crop_objects.csv. Run locally after 934 and 900:
COCH_ROOT=/path/to/coch-rythum python scripts/935_parabiosis_figure.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Circle
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import GRID, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, halo, text_width  # noqa: E402
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402

R9, R10 = RESULTS / "r9_figures", RESULTS / "r10_parabiosis"
PANELS = ROOT / "figures" / "panels"
OUT = ROOT / "figures" / "figures"
NAMES = {"S1": "sfig_parabiosis_tissues_v1", "S2": "sfig_parabiosis_perilymph_v1", "S3": "sfig_parabiosis_noise_v1",
         "S4": "sfig_parabiosis_cell_types_v2"}
_spec = importlib.util.spec_from_file_location("s934", str(Path(__file__).resolve().parent / "934_parabiosis_survey.py"))
S934 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S934)
SRC = S934.SRC
C_MARROW, C_COCH = fd.TISSUE_COLOUR["bm"], fd.TISSUE_COLOUR["cochlea"]
C_VEST, C_LIVER = "#7F7F7F", "#009E73"
DAY_COL = {"5 d": "#F0E442", "7 d": "#E69F00", "14 d": "#56B4E9", "28 d": "#009E73", "31 d": "#CC79A7",
           "54 d": "#000000", "31 d noise": "#000000"}
GREEN, BLUE, MAG = np.array([0.1, 1.0, 0.1]), np.array([0.25, 0.45, 1.0]), np.array([0.95, 0.2, 0.95])
TILE, BIG, GUT, X0 = 28.6, 43.4, 0.6, 2.7
FIELD = "2025-01-15-2/TIFF-25.1.15/"
IMG_LOG, OBJ_LOG = [], []
_CACHE = {}


# ------------------------------------------------------------------------------------------------ image helpers
def load(rel):
    if rel not in _CACHE:
        _CACHE.clear()                                   # one acquisition in memory at a time
        if rel.startswith(FIELD):                        # rendered liver export: c1 lectin, c2 GFP, c3 DAPI
            stem = rel.rsplit("/", 1)[-1]
            d = SRC / "2025-01-15-2" / "TIFF-25.1.15" / stem
            chs = [tifffile.imread(d / f"{stem}_c{k}.tif").astype(np.float32).max(axis=-1) for k in (1, 2, 3)]
            meta = dict(um_x=S934.LIVER_UM, um_z=None, channels=["Alexa Fluor 647", "Alexa Fluor 488", "DAPI"])
            _CACHE[rel] = (np.stack(chs)[:, None], meta)
        else:
            _CACHE[rel] = S934.read_czi(SRC / rel)
    return _CACHE[rel]


def roles(meta):
    gi, di, fi = S934.channel_roles(meta["channels"])
    return {"gfp": gi, "dapi": di, "far": fi}


def far_name(meta, rel):
    fi = roles(meta)["far"]
    if fi is None:
        return ""
    n = meta["channels"][fi]
    low = rel.lower()
    if "lectin" in low:
        return "lectin"
    if "fut4" in low:
        return "FUT4"
    if "cd19" in low:
        return "CD19"
    if "cd45" in low:
        return "CD45"
    if "phalloidin" in low:
        return "phalloidin"
    return n


PLANES = None


def pick_plane(rel, a, meta, objects=None):
    """The analysed optical section with the strongest GFP signal (99.9th percentile within tissue; 934 planes table),
    as used by the depth sensitivity analysis; single planes as acquired."""
    global PLANES
    if a.shape[1] == 1:
        return 0
    if PLANES is None:
        PLANES = pd.read_csv(R10 / "parabiosis_planes.csv")
    p = PLANES[PLANES.path == rel]
    return int(p.loc[p.gfp_p999.idxmax(), "plane"]) if len(p) else 0


def ranges(a, z, idx):
    out = {}
    for k, c in idx.items():
        if c is not None:
            ch = a[c, z].astype(np.float32)
            out[k] = (float(np.percentile(ch, 0.5)), float(np.percentile(ch, 99.9)))
    return out


def scaled(a, z, c, rng, box):
    ch = a[c, z].astype(np.float32)
    if box is not None:
        y0, y1, x0, x1 = box
        ch = ch[y0:y1, x0:x1]
    lo, hi = rng
    return np.clip((ch - lo) / max(hi - lo, 1.0), 0, 1)


def merge(a, z, idx, rng, keys=("far", "dapi", "gfp"), box=None):
    shape = a.shape[2:] if box is None else (box[1] - box[0], box[3] - box[2])
    rgb = np.zeros(tuple(shape) + (3,))
    for k, col in (("far", MAG), ("dapi", BLUE), ("gfp", GREEN)):
        if k in keys and idx.get(k) is not None:
            rgb += scaled(a, z, idx[k], rng[k], box)[..., None] * col
    return np.clip(rgb, 0, 1)


def grey(a, z, idx, rng, box):
    g = scaled(a, z, idx["gfp"], rng["gfp"], box)
    return np.repeat(g[..., None], 3, axis=2)


def bar_len(field_um):
    return 100 if field_um > 400 else (50 if field_um > 150 else 20)


def show(cv, x, y, size, rgb, um, label=None, circles=(), bar=None):
    """Image tile at (x, y) mm, size mm square (non-square images letterboxed); white scale bar with its length."""
    h, w = rgb.shape[:2]
    n = max(h, w)
    ax = cv.axes(x, y, size, size)
    ax.set_facecolor("#000000")
    ax.imshow(rgb, interpolation="none", extent=(-0.5 + (n - w) / 2, w - 0.5 + (n - w) / 2, h - 0.5 + (n - h) / 2,
                                                  -0.5 + (n - h) / 2))
    for cy, cx, r, col in circles:
        ax.add_patch(Circle((cx + (n - w) / 2, cy + (n - h) / 2), r, fill=False, edgecolor=col, linewidth=0.45))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    blank(ax)
    bar = bar or bar_len(n * um)
    blen = bar / um / n * size
    bx1, by = x + size - 1.0, y + size - 1.1
    cv.line([bx1 - blen, bx1], [by, by], "#000000", 1.7, solid_capstyle="butt")
    cv.line([bx1 - blen, bx1], [by, by], "#FFFFFF", 0.9, solid_capstyle="butt")
    cv.text(bx1, by - 0.5, f"{bar} µm", size=PT_SMALL, color="#FFFFFF", ha="right", va="bottom",
            path_effects=halo(1.1, "#000000"))
    if label:
        cv.text(x + 0.8, y + 0.7, label, size=PT_SMALL, color="#FFFFFF", ha="left", va="top",
                path_effects=halo(1.1, "#000000"))


def save_tile(fig, panel, k, rgb):
    name = f"pb_{fig.lower()}{panel}_{k}"
    PANELS.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(PANELS / f"{name}.png")
    return f"{name}.png"


def log_tile(fig, panel, k, png, rel, z, um, box, rng, keys, label, shown=0):
    disp = "; ".join(f"{k_} {lo:.0f}-{hi:.0f}" for k_, (lo, hi) in rng.items() if k_ in keys)
    IMG_LOG.append(dict(figure=fig, panel=panel, tile=k, label=label, image_file=png, acquisition=rel, plane=z,
                        um_per_px=um, crop_y0_y1_x0_x1="full field" if box is None else "-".join(map(str, box)),
                        channels_shown=", ".join(keys), display_range_grey_levels=disp, objects_marked=shown))


def full_tile(cv, fig, panel, k, x, y, size, rel, label, objects, keys=("far", "dapi", "gfp"), gfp_rng=None):
    a, meta = load(rel)
    z = pick_plane(rel, a, meta, objects)
    idx = roles(meta)
    rng = ranges(a, z, idx)
    if gfp_rng is not None:                              # one GFP display range shared by both partners of a pair
        rng["gfp"] = gfp_rng
    rgb = merge(a, z, idx, rng, keys)
    png = save_tile(fig, panel, k, rgb)
    show(cv, x, y, size, rgb, meta["um_x"], label)
    log_tile(fig, panel, k, png, rel, z, meta["um_x"], None, rng, keys, label)
    return far_name(meta, rel)


def best_box(objs, shape, n, target, avoid_label=False, others=None, image_edge_ok=False):
    """Crop (n x n px) whose number of objects is closest to target; ties to the window with the fewest objects cut by
    or near its border (within 12% of n), then to the most central window; avoid_label also keeps the objects out of
    the tile label (top left); others (e.g. rejected bodies) do not count toward the target but are kept off the
    border, the scale bar and the label in the same way; image_edge_ok: a window side on the image's own edge needs no
    margin (an object there is cut by the acquisition, not by the crop)."""
    best, key = (0, 0), None
    ys = np.linspace(0, max(shape[0] - n, 0), 17).astype(int)
    xs = np.linspace(0, max(shape[1] - n, 0), 17).astype(int)
    m = 0.10 * n

    def zones(o, y0, x0):
        inside = (o.y >= y0) & (o.y < y0 + n) & (o.x >= x0) & (o.x < x0 + n)
        mt, mb, ml, mr = m, m, m, m
        if image_edge_ok:
            mt, mb = (m if y0 > 0 else -1.0), (m if y0 + n < shape[0] else -1.0)
            ml, mr = (m if x0 > 0 else -1.0), (m if x0 + n < shape[1] else -1.0)
        core = (o.y >= y0 + mt) & (o.y < y0 + n - mb) & (o.x >= x0 + ml) & (o.x < x0 + n - mr)
        near = (o.y >= y0 - m) & (o.y < y0 + n + m) & (o.x >= x0 - m) & (o.x < x0 + n + m)
        bar = inside & (o.x >= x0 + 0.55 * n) & (o.y >= y0 + 0.8 * n)          # under the scale bar
        if avoid_label:
            bar = bar | (inside & (o.x < x0 + 0.62 * n) & (o.y < y0 + 0.16 * n))
        return inside, core & ~bar, near

    cand = []
    for y0 in ys:
        for x0 in xs:
            inside, core, near = zones(objs, y0, x0)
            edge = int((near & ~core).sum())
            if others is not None and len(others):
                _, oc, on = zones(others, y0, x0)
                edge += int((on & ~oc).sum())
            cand.append((int(y0), int(x0), int(inside.sum()), int(core.sum()), edge))
    if target >= 1 and any(c[3] >= 1 for c in cand):     # show at least one counted object away from the border
        cand = [c for c in cand if c[3] >= 1]
    for y0, x0, k, kc, edge in cand:
        c = (abs(k - target) + 0.5 * edge, edge, abs(y0 + n / 2 - shape[0] / 2) + abs(x0 + n / 2 - shape[1] / 2))
        if key is None or c < key:
            best, key = (y0, x0), c
    return best


def crop_tile(cv, fig, panel, k, x, y, size, rel, label, objects, tf, crop_um=100.0):
    """GFP channel (grey) of a crop holding about the field's expected number of partner-derived cells (upgraded rule,
    D143): the 934 GFP+ bodies kept by the rule circled yellow, those it rejected grey."""
    a, meta = load(rel)
    um = meta["um_x"]
    z = pick_plane(rel, a, meta, objects)
    idx = roles(meta)
    rng = ranges(a, z, idx)
    n = min(int(round(crop_um / um)), a.shape[2], a.shape[3])
    op = objects[(objects.path == rel) & (objects.plane == z)]
    assert op.cell.notna().all(), rel
    dens = float(objects.loc[objects.path == rel, "cell"].astype(bool).sum()) / float(tf.loc[rel, "tissue_mm2"])
    kept = op[op.cell.astype(bool)]                     # a field without a kept cell shows its rejected bodies
    y0, x0 = best_box(kept if len(kept) else op, a.shape[2:], n, max(dens * (n * um) ** 2 / 1e6, 1.0),
                      avoid_label=True, others=op[~op.cell.astype(bool)] if len(kept) else None,
                      image_edge_ok=True)
    box = (y0, y0 + n, x0, x0 + n)
    # GFP displayed linearly from the section's tissue median to the larger of the crop's 99.9th percentile and the
    # section's detection threshold (934 planes table), so counted bodies read bright and sub-threshold signal dim
    global PLANES
    if PLANES is None:
        PLANES = pd.read_csv(R10 / "parabiosis_planes.csv")
    pr_ = PLANES[(PLANES.path == rel) & (PLANES.plane == z)]
    if len(pr_):
        crop = a[idx["gfp"], z, y0:y0 + n, x0:x0 + n].astype(np.float32)
        rng = dict(rng, gfp=(float(pr_["median"].iloc[0]),
                             max(float(np.percentile(crop, 99.9)), float(pr_["threshold"].iloc[0]))))
    rgb = grey(a, z, idx, rng, box)
    inside = op[(op.y >= y0) & (op.y < y0 + n) & (op.x >= x0) & (op.x < x0 + n)]
    circles = [(r.y - y0, r.x - x0, max(np.sqrt(r.area_um2 / np.pi) / um * 1.6, 3.0),
                C_KEPT if r.cell else C_REJ) for r in inside.itertuples()]
    png = save_tile(fig, panel, k, rgb)
    show(cv, x, y, size, rgb, um, label, circles, bar=20)
    log_tile(fig, panel, k, png, rel, z, um, box, rng, ("gfp",), label, len(inside))
    for r in inside.itertuples():
        OBJ_LOG.append(dict(figure=fig, panel=panel, tile=k, acquisition=rel, plane=z, object="GFP+ body",
                            x_in_crop_px=r.x - x0, y_in_crop_px=r.y - y0, area_um2=r.area_um2, gfp_pos=True,
                            kept_by_upgraded_rule=bool(r.cell)))


C_KEPT, C_REJ = "#FFD700", "#8C8C8C"                     # body kept as a cell by the upgraded rule / rejected


def kept_key(cv, x, y):
    """Key of the crop circles: kept (counted) vs rejected bodies; returns the right edge."""
    for col, lab in (("#C9A800", "GFP+ cell (counted)"), (C_REJ, "rejected body")):
        cv.mark(x + 0.9, y, 10.0, "o", "none", col, 0.6)
        cv.text(x + 2.2, y, lab, size=PT_SMALL, va="center")
        x += 2.2 + text_width(lab, PT_SMALL) + 2.0
    return x


def fmt_dec(t):
    return f"{t:,.0f}" if t >= 1 else f"{t:g}"


def logaxis(ax, lo, hi, ticks=None, labels=None, axis="y"):
    """Log axis made explicit: major ticks at decades (labelled), unlabelled minor ticks at 2-9 x each decade."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)
    dec = [10.0 ** k for k in range(-4, 6) if lo <= 10.0 ** k <= hi]
    ticks = ticks or dec
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter(labels or [fmt_dec(t) for t in ticks]))
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    a.set_minor_formatter(NullFormatter())
    ax.tick_params(axis=axis, which="major", length=1.5, pad=0.8, labelsize=PT_SMALL)
    ax.tick_params(axis=axis, which="minor", length=1.0, width=0.4)
    clean(ax)


def lin(ax, ylim, ticks):
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)


def channel_key(cv, x, y, far_label):
    items = [([{"kind": "box", "fc": "#1AFF1A"}], "GFP"), ([{"kind": "box", "fc": "#4073FF"}], "DAPI")]
    if far_label:
        items.append(([{"kind": "box", "fc": "#F233F2"}], far_label))
    return cv.key(x, y, items)


# ------------------------------------------------------------------------------------------------ Supplementary Fig. 1 (tissues)
S2_RAW = [  # (tile label, acquisition)
    ("Marrow 5 d", "2025-01-09/2025-01-09/20x-phalloidin-gfp-wt-5D-bone-02.czi"),
    ("Marrow 7 d", "2025-01-17-2/20x-7d-WT-GFP-LECTIN-DAPI-bone-01.czi"),
    ("Marrow 14 d", "2025.1.23/2025-01-23/14d-infect-gfp-vessal-10x-0_45x-bone-01.czi"),
    ("Marrow 31 d", "2025-01-15-2/20X-0_45X-31d-lectin--GFP-BONE-02.czi"),
    ("Marrow 54 d", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-02.czi"),
    ("Vestibular bone 54 d", "immune_cell_typing/2026-02-09/parabiosis54d-gfp-647CD45-dapi-10x-qiantingbone-01.czi"),
    ("Cochlea 14 d", "2025.1.6-14D/2025-01-06/Cochlear-14D-lectin-gfp-dapi-20x-01.czi"),
    ("Stria 14 d", "day14_immune_cell_thick_sections/2025-03-31/2-cd19-63x-sv-1-01.czi"),
    ("Spiral lamina 31 d", "2025-01-15-2/20X-0_45X-31d-GFP-lectin-dapi-LUOXUANBAN-01.czi"),
    ("Lateral wall 54 d", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-05.czi"),
    ("Duct 54 d", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-12.czi"),
    ("Liver 31 d", FIELD + "WT-31D-LIVER-20X-2_4X-02")]
S2_CROP = [("Marrow 14 d", "2025.1.23/2025-01-23/14d-infect-gfp-vessal-10x-0_45x-bone-01.czi"),
           ("Cochlea 14 d", "2025.1.6-14D/2025-01-06/Cochlear-14D-lectin-gfp-dapi-20x-01.czi"),
           ("Marrow 31 d", "2025-01-15-2/20X-0_45X-31d-lectin--GFP-BONE-02.czi"),
           ("Cochlea 31 d", "2025-01-15-2/20X-0_45X-31d-GFP-lectin-dapi-LUOXUANBAN-01.czi"),
           ("Marrow 54 d", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-02.czi"),
           ("Cochlea 54 d", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-05.czi")]


def fut4_tile(cv, fig, panel, k, x, y, size, rel, label, fut, crop_um=80.0):
    a, meta = load(rel)
    um = meta["um_x"]
    idx = roles(meta)
    idx["far"] = next(i for i, n in enumerate(meta["channels"]) if "647" in n)
    f = fut[fut.acquisition == rel]
    z = int(f.plane.mode().iloc[0])
    f = f[f.plane == z]
    rng = ranges(a, z, idx)
    n = min(int(round(crop_um / um)), a.shape[2])
    if "marrow" in label.lower():                        # marrow: the window holding the most GFP+ FUT4+ cells
        y0, x0 = best_box(f[f.gfp_pos], a.shape[2:], n, max(int(f.gfp_pos.sum()), 1) + 99)
    else:                                                # cochlea: the window holding the most FUT4+ cells
        y0, x0 = best_box(f[f.compartment == "cochlea"], a.shape[2:], n, 99)
    box = (y0, y0 + n, x0, x0 + n)
    rgb = merge(a, z, idx, rng, box=box)
    inside = f[(f.y >= y0) & (f.y < y0 + n) & (f.x >= x0) & (f.x < x0 + n)]
    circles = [(r.y - y0, r.x - x0, max(np.sqrt(r.area_um2 / np.pi) / um * 1.6, 3.0),
                "#FFD700" if r.gfp_pos else "#FFFFFF") for r in inside.itertuples()]
    png = save_tile(fig, panel, k, rgb)
    show(cv, x, y, size, rgb, um, label, circles, bar=20)
    log_tile(fig, panel, k, png, rel, z, um, box, rng, ("far", "dapi", "gfp"), label, len(inside))
    for r in inside.itertuples():
        OBJ_LOG.append(dict(figure=fig, panel=panel, tile=k, acquisition=rel, plane=z, object="FUT4+ cell",
                            x_in_crop_px=r.x - x0, y_in_crop_px=r.y - y0, area_um2=r.area_um2, gfp_pos=bool(r.gfp_pos)))


def fig_s1(ob):
    tf = pd.read_csv(R9 / "pb_tissue_fields.csv")
    tfi = pd.read_csv(R10 / "parabiosis_tissue_fields.csv").set_index("path")
    pr = pd.read_csv(R9 / "pb_pairs.csv")
    det = pd.read_csv(R9 / "pb_detection.csv")
    H = 6.2 + 2 * TILE + GUT + 3.6 + 6.2 + TILE + 3.6 + 6.2 + 36.0 + 9.5
    cv = Canvas(180.0, H)
    rng = np.random.default_rng(2)

    # a raw images
    cv.letter(0, 0, "a", "Raw images (wild-type partners)")
    fars = set()
    for k, (lab, rel) in enumerate(S2_RAW):
        x = X0 + (k % 6) * (TILE + GUT)
        y = 6.2 + (k // 6) * (TILE + GUT)
        fars.add(full_tile(cv, "S1", "a", k + 1, x, y, TILE, rel, lab, ob))
    kx = 3.0 + text_width("Raw images (wild-type partners)", PT, "bold") + 4.0
    x = cv.key(kx, 2.6, [([{"kind": "box", "fc": "#1AFF1A"}], "GFP"), ([{"kind": "box", "fc": "#4073FF"}], "DAPI"),
                         ([{"kind": "box", "fc": "#F233F2"}], "lectin, FUT4, CD45 or CD19")])

    # b GFP crops with the counted bodies
    yb = 6.2 + 2 * TILE + GUT + 3.6
    cv.letter(0, yb, "b", "GFP channel, counted cells")
    for k, (lab, rel) in enumerate(S2_CROP):
        crop_tile(cv, "S1", "b", k + 1, X0 + k * (TILE + GUT), yb + 6.2, TILE, rel, lab, ob, tfi)
    kx = 3.0 + text_width("GFP channel, counted cells", PT, "bold") + 4.0
    kept_key(cv, kx, yb + 2.6)

    # c densities per field and animal
    yc = yb + 6.2 + TILE + 3.6
    cv.letter(0, yc, "c", "Partner-derived cells")
    ax = cv.axes(10.0, yc + 7.0, 46.0, 36.0)
    groups = [("marrow", "Marrow"), ("cochlea", "Cochlea"), ("vestibular bone", "Vestib.\nbone"), ("liver", "Liver")]
    floor = 3.0
    for k, (comp, _) in enumerate(groups):
        g = tf[tf.compartment == comp]
        v = g.cell_density_per_mm2.to_numpy()
        ax.scatter(k - 0.1 + rng.uniform(-0.16, 0.16, len(v)), np.maximum(v, floor), s=5,
                   c=[DAY_COL.get(a_, "#888888") for a_ in g.animal], edgecolors="#333333",
                   linewidths=[0.25 if a_ == "5 d" else 0 for a_ in g.animal], alpha=0.9, zorder=3)
        for an, ga in g.groupby("animal"):
            pooled = ga.gfp_cells.sum() / ga.tissue_mm2.sum()
            ax.scatter(k + 0.3, max(pooled, floor), s=16, marker="D", c=DAY_COL.get(an, "#888888"),
                       edgecolors="#333333", linewidths=0.35, zorder=4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=PT_SMALL)
    logaxis(ax, 2.2, 4000, [3, 10, 100, 1000], ["0", "10", "100", "1,000"])
    ax.yaxis.set_minor_locator(FixedLocator([v * d for d in (10, 100, 1000) for v in range(2, 10) if v * d < 4000]))
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    # axis break between the zero floor and 10 (fields without a counted cell sit on the floor)
    for yy in (4.6, 5.4):
        ax.plot([-0.035, 0.035], [yy * 0.93, yy * 1.07], transform=ax.get_yaxis_transform(), color=TEXT,
                linewidth=0.5, clip_on=False, zorder=5)
    days = [d for d in ["5 d", "7 d", "14 d", "31 d", "54 d"] if d in set(tf.animal)]
    lx = 59.0
    for i, d in enumerate(days):
        cv.mark(lx, yc + 8.0 + 2.5 * i, 8.0, "o", DAY_COL[d], "#333333" if d == "5 d" else None, 0.3)
        cv.text(lx + 1.4, yc + 8.0 + 2.5 * i, d, size=PT_SMALL, va="center")
    cv.key(lx - 0.9, yc + 8.0 + 2.5 * len(days) + 0.8, [([{"kind": "mark", "s": 5, "fc": "#888888"}], "Field")])
    cv.key(lx - 0.9, yc + 8.0 + 2.5 * len(days) + 3.3, [([{"kind": "mark", "s": 16, "marker": "D", "fc": "#BBBBBB",
                                                         "ec": "#333333", "lw": 0.35}], "Animal")])

    # d pairs: marrow -> cochlea (observed, upgraded rule); key shared with e
    xd = 74.0
    cv.letter(xd, yc, "d", "Per pair")
    ax = cv.axes(xd + 10.5, yc + 7.0, 24.0, 36.0)
    for r in pr.itertuples():
        col = DAY_COL.get(r.animal, "#888888")
        ax.plot([0, 1], [r.marrow_density, r.cochlea_density], color=col, linewidth=0.7, zorder=2)
        ax.scatter([0, 1], [r.marrow_density, r.cochlea_density], s=10, c=col, linewidths=0, zorder=3)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Marrow", "Cochlea"], fontsize=PT_SMALL)
    logaxis(ax, 10, 1500)
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    prim = pr.sort_values("days")
    p = float(prim.sign_p.iloc[0])
    ax.text(1.32, 1350, f"{int((prim.cochlea_to_marrow < 1).sum())}/{len(prim)} pairs\nP = {p:.2f}",
            fontsize=PT_SMALL, ha="right", va="top", color=TEXT, linespacing=1.15)
    kx, ky = xd + 36.5, yc + 8.0
    for i, an in enumerate(prim.animal):
        cv.line([kx, kx + 2.2], [ky + 2.5 * i, ky + 2.5 * i], DAY_COL.get(an, "#888888"), 0.7)
        cv.text(kx + 2.7, ky + 2.5 * i, an, size=PT_SMALL, va="center")

    # e robustness: cochlea / marrow per pair as observed, detection-corrected and in the brightest section
    xe = 124.0
    cv.letter(xe, yc, "e", "Cochlea / marrow")
    ax = cv.axes(xe + 10.5, yc + 7.0, 44.0, 36.0)
    cols = [("cochlea_to_marrow", "Observed"), ("corrected_cochlea_to_marrow", "Detection-\ncorrected"),
            ("best_plane_cochlea_to_marrow", "Brightest\nsection"), ("verified_ratio", "Label-\ncorrected")]
    lo_ax, hi_ax = 0.01, 10.0
    for r in prim.itertuples():
        ys = [getattr(r, c_) for c_, _ in cols]
        col = DAY_COL.get(r.animal, "#888888")
        ax.plot(range(4), ys, color=col, linewidth=0.7, zorder=2)
        ax.scatter(range(4), ys, s=10, c=col, linewidths=0, zorder=3)
        dx = {14: -0.14, 31: 0.0, 54: 0.14}[int(r.days)]           # label-corrected 95% CI (blinded labels, 960)
        assert r.ci_hi < hi_ax and r.verified_ratio > lo_ax
        ax.plot([3 + dx] * 2, [max(r.ci_lo, lo_ax), r.ci_hi], color=col, linewidth=0.6, zorder=2,
                solid_capstyle="butt", clip_on=False)
        if r.ci_lo <= lo_ax:                                       # interval reaches 0: arrowhead at the axis floor
            ax.scatter([3 + dx], [lo_ax * 1.12], s=7, marker="v", c=col, linewidths=0, zorder=3, clip_on=False)
    ax.axhline(1.0, color=TEXT2, linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(4), [c_[1] for c_ in cols], fontsize=PT_SMALL, linespacing=1.0)
    logaxis(ax, lo_ax, hi_ax)
    ax.set_ylabel("Density ratio (log scale)", fontsize=PT_SMALL, labelpad=0.8)

    return cv


# ------------------------------------------------------------------------------------------------ Supplementary Fig. 2 (perilymph)
S3_RAW = [("WT 14 d", "2025.1.23/2025-01-24/14d-wt-out-20x-0_45x-01.czi"),
          ("GFP partner 14 d", "2025.1.23/2025-01-24/14d-gfp-out-20x-0_45x-01.czi"),
          ("WT 28 d", "2025.1.23/2025-01-24/28d-wt-out-20x-0_45x-02.czi"),
          ("GFP partner 28 d", "2025.1.23/2025-01-24/28d-gfp-out-20x-0_45x-06.czi")]


def cell_crop(cv, fig, panel, k, x, y, size, rel, label, cells, crop_um=150.0, gfp_rng=None):
    a, meta = load(rel)
    um = meta["um_x"]
    idx = roles(meta)
    rng = ranges(a, 0, idx)
    if gfp_rng is not None:
        rng["gfp"] = gfp_rng
    n = min(int(round(crop_um / um)), a.shape[2])
    c = cells[cells.acquisition == rel]
    y0, x0 = best_box(c, a.shape[2:], n, len(c) + 99)        # the window with the most nucleated cells
    box = (y0, y0 + n, x0, x0 + n)
    rgb = merge(a, 0, idx, rng, ("dapi", "gfp"), box)
    inside = c[(c.y >= y0) & (c.y < y0 + n) & (c.x >= x0) & (c.x < x0 + n)]
    circles = [(r.y - y0, r.x - x0, max(np.sqrt(r.nucleus_um2 / np.pi) / um * 1.9, 3.0),
                "#FFD700" if r.gfp_pos else "#FFFFFF") for r in inside.itertuples()]
    png = save_tile(fig, panel, k, rgb)
    show(cv, x, y, size, rgb, um, label, circles, bar=20)
    log_tile(fig, panel, k, png, rel, 0, um, box, rng, ("dapi", "gfp"), label, len(inside))
    for r in inside.itertuples():
        OBJ_LOG.append(dict(figure=fig, panel=panel, tile=k, acquisition=rel, plane=0, object="nucleated cell",
                            x_in_crop_px=r.x - x0, y_in_crop_px=r.y - y0, area_um2=r.nucleus_um2,
                            gfp_pos=bool(r.gfp_pos)))


def fig_s2():
    pf = pd.read_csv(R9 / "pb_perilymph_fields.csv")
    cl = pd.read_csv(R9 / "pb_perilymph_cells.csv")
    H = 6.2 + BIG + 3.6 + 6.2 + TILE + 4.0 + 6.2 + 30.0 + 7.0
    cv = Canvas(180.0, H)
    rng = np.random.default_rng(3)
    # GFP is displayed with the GFP partner's range (0.5th-99.9th percentile of its field) for both partners of a pair,
    # so wild-type and GFP-partner fields are compared at one display setting (the wild-type fields were acquired at
    # higher sensitivity, panel e); DAPI keeps each field's own range
    pair_gfp = {}
    for lab, rel in S3_RAW:
        if lab.startswith("GFP partner"):
            a, meta = load(rel)
            pair_gfp[" ".join(lab.split()[-2:])] = ranges(a, 0, roles(meta))["gfp"]
    cv.letter(0, 0, "a", "Raw images (perilymph)")
    for k, (lab, rel) in enumerate(S3_RAW):
        full_tile(cv, "S2", "a", k + 1, X0 + k * (BIG + GUT), 6.2, BIG, rel, lab, pd.DataFrame(columns=["path", "plane"]),
                  keys=("dapi", "gfp"), gfp_rng=pair_gfp[" ".join(lab.split()[-2:])])
    kx = 3.0 + text_width("Raw images (perilymph)", PT, "bold") + 4.0
    cv.key(kx, 2.6, [([{"kind": "box", "fc": "#1AFF1A"}], "GFP"), ([{"kind": "box", "fc": "#4073FF"}], "DAPI")])
    yb = 6.2 + BIG + 3.6
    cv.letter(0, yb, "b", "Cell calls")
    for k, (lab, rel) in enumerate(S3_RAW):
        cell_crop(cv, "S2", "b", k + 1, X0 + k * (TILE + GUT), yb + 6.2, TILE, rel, lab, cl,
                  gfp_rng=pair_gfp[" ".join(lab.split()[-2:])])
    kx = 3.0 + text_width("Cell calls", PT, "bold") + 4.0
    cv.mark(kx + 0.9, yb + 2.6, 10.0, "o", "none", "#C9A800", 0.6)
    cv.text(kx + 2.2, yb + 2.6, "GFP+", size=PT_SMALL, va="center")
    cv.mark(kx + 9.4, yb + 2.6, 10.0, "o", "none", "#777777", 0.6)
    cv.text(kx + 10.7, yb + 2.6, "GFP−", size=PT_SMALL, va="center")

    # c shares per field and pooled
    xc = X0 + 4 * (TILE + GUT) + 3.0
    cv.letter(xc, yb, "c", "GFP+ share")
    ax = cv.axes(xc + 10.0, yb + 7.0, 46.0, TILE - 1.0)
    pos = {("14 d", "WT"): 0, ("14 d", "GFP"): 1, ("28 d", "WT"): 2.4, ("28 d", "GFP"): 3.4}
    for (an, prt), xx in pos.items():
        g = pf[(pf.animal == an) & (pf.partner == prt)]
        col = "#333333" if prt == "WT" else "#2CA02C"
        pooled = 100 * g.gfp_pos.sum() / g.nuclei.sum()
        ax.bar(xx, pooled, width=0.7, color="#E6E6E6", edgecolor=col, linewidth=0.5, zorder=1)
        ax.scatter(xx + rng.uniform(-0.15, 0.15, len(g)), g.gfp_pos_pct, s=np.clip(g.nuclei / 5, 3, 24), c=col,
                   linewidths=0, alpha=0.85, zorder=3)
        ax.text(xx, max(pooled, g.gfp_pos_pct.max()) + 4, f"{pooled:.0f}%", fontsize=PT_SMALL, ha="center",
                va="bottom")
    ax.set_xticks(list(pos.values()), ["WT", "GFP", "WT", "GFP"], fontsize=PT_SMALL)
    ax.set_xlim(-0.6, 4.0)
    lin(ax, (0, 110), [0, 50, 100])
    ax.set_ylabel("GFP+ (% nucleated)", fontsize=PT_SMALL, labelpad=0.8)
    for xx, lab in [(0.5, "14 d"), (2.9, "28 d")]:
        cv.text(xc + 10.0 + (xx + 0.6) / 4.6 * 46.0, yb + 7.0 + TILE - 1.0 + 3.6, lab, size=PT_SMALL, ha="center",
                va="top")

    # d ring / background per cell
    yd = yb + 6.2 + TILE + 4.0
    cv.letter(0, yd, "d", "Cytoplasmic GFP per cell")
    ax = cv.axes(12.0, yd + 7.0, 80.0, 30.0)
    for k, ((an, prt), g) in enumerate(cl.groupby(["animal", "partner"], sort=True)):
        xx = {"14 d": 0, "28 d": 2.4}[an] + (1 if prt == "GFP" else 0)
        ratio = g.ring_gfp / np.maximum(g.background, 2.0)
        cols = np.where(g.gfp_pos.astype(bool), "#C9A800", "#777777")
        ax.scatter(xx + rng.uniform(-0.28, 0.28, len(g)), np.clip(ratio, 0.3, 30), s=2.5, c=cols, linewidths=0,
                   alpha=0.8, zorder=3, rasterized=True)
    ax.axhline(2.0, color=TEXT2, linewidth=0.5, linestyle="--", zorder=2)
    ax.set_xticks([0, 1, 2.4, 3.4], ["WT 14 d", "GFP 14 d", "WT 28 d", "GFP 28 d"], fontsize=PT_SMALL)
    ax.set_xlim(-0.6, 4.0)
    logaxis(ax, 0.3, 30)
    ax.set_ylabel("Ring GFP / background (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    ax.text(3.95, 2.15, "2×", fontsize=PT_SMALL, ha="right", va="bottom", color=TEXT2)

    # e sensitivity of the wild-type fields
    xe = 104.0
    cv.letter(xe, yd, "e", "GFP sensitivity, WT fields")
    ax = cv.axes(xe + 11.0, yd + 7.0, 36.0, 30.0)
    wt = pf[pf.partner == "WT"]
    for k, an in enumerate(["14 d", "28 d"]):
        g = wt[wt.animal == an]
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(g)), g.sensitivity_vs_gfp_partner, s=9, c="#333333", linewidths=0,
                   zorder=3)
    ax.axhline(1.0, color=TEXT2, linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["14 d", "28 d"], fontsize=PT_SMALL)
    top = float(np.ceil(wt.sensitivity_vs_gfp_partner.max() * 1.1))
    logaxis(ax, 0.5, max(top, 4))
    ax.set_ylabel("Relative to GFP partner (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    return cv


# ------------------------------------------------------------------------------------------------ Supplementary Fig. 3 (noise)
NOISE_RAW = [("Marrow, noise", "2025-01-17-2/20x-0_45x-31d-NOISE-WT-GFP-LECTIN-bone-01.czi"),
          ("Apex, noise", "2025-01-17-2/20x-0_45x-31d-NOISE-WT-GFP-LECTIN-apex-01.czi"),
          ("Marrow, 31 d", "2025-01-15-2/20X-0_45X-31d-lectin--GFP-BONE-02.czi"),
          ("Spiral lamina, 31 d", "2025-01-15-2/20X-0_45X-31d-GFP-lectin-dapi-LUOXUANBAN-01.czi"),
          ("Apex, 14 d", "2025.1.11/2025-01-11/Cochlear-14D-lectin-gfp-dapi-20x-apex-01.czi")]
C_NOISE, C_UNEXP = "#000000", "#9A9A9A"


def fig_s3(ob):
    tfi = pd.read_csv(R10 / "parabiosis_tissue_fields.csv").set_index("path")
    pr = pd.read_csv(R9 / "pb_noise_pairs.csv")
    pf = pd.read_csv(R9 / "pb_noise_perilymph_fields.csv")
    sec = pd.read_csv(R9 / "pb_noise_sections.csv")
    cmp_ = pd.read_csv(R9 / "pb_noise_comparison.csv").set_index("comparison")
    T = (180.0 - 2 * X0 - 4 * GUT) / 5
    H = 6.2 + 2 * T + GUT + 4.6 + 3.0 + 6.2 + 34.0 + 10.0
    cv = Canvas(180.0, H)
    rng = np.random.default_rng(4)
    cv.letter(0, 0, "a", "Raw images (top) and counted GFP+ cells (bottom)")
    for k, (lab, rel) in enumerate(NOISE_RAW):
        x = X0 + k * (T + GUT)
        full_tile(cv, "S3", "a", k + 1, x, 6.2, T, rel, lab, ob)
        crop_tile(cv, "S3", "a", k + 6, x, 6.2 + T + GUT, T, rel, lab, ob, tfi)
    kx = 3.0 + text_width("Raw images (top) and counted GFP+ cells (bottom)", PT, "bold") + 4.0
    x = channel_key(cv, kx, 2.6, "lectin")
    kept_key(cv, x + 2.2, 2.6)
    by = 6.2 + 2 * T + GUT + 1.4                           # brackets: which columns belong to which pair
    for c0, c1, lab in [(0, 1, "Noise-exposed pair (31 d)"), (2, 4, "Unexposed pairs (31 d, 14 d)")]:
        xa, xb = X0 + c0 * (T + GUT) + 1.0, X0 + c1 * (T + GUT) + T - 1.0
        cv.line([xa, xa, xb, xb], [by - 0.6, by, by, by - 0.6], TEXT2, 0.5)
        cv.text((xa + xb) / 2, by + 0.5, lab, size=PT_SMALL, ha="center", va="top", color=TEXT)

    # b per-section densities with pooled values and within-animal bootstrap intervals
    yc = 6.2 + 2 * T + GUT + 4.6 + 3.0
    cv.letter(0, yc, "b", "Per optical section")
    ax = cv.axes(11.0, yc + 7.0, 52.0, 34.0)
    order = [("noise marrow", 0, C_NOISE), ("31 d marrow", 1, C_UNEXP), ("noise cochlea (apex)", 2.6, C_NOISE),
             ("31 d cochlea (spiral lamina)", 3.6, C_UNEXP), ("unexposed apex (14 d)", 4.6, C_UNEXP)]
    floor = 3.0
    for key, xx, col in order:
        g = sec[sec.set == key]
        ax.scatter(xx - 0.12 + rng.uniform(-0.13, 0.13, len(g)), np.maximum(g.density_per_mm2, floor), s=4, c=col,
                   linewidths=0, alpha=0.7, zorder=3, rasterized=True)
        r = cmp_.loc[f"{key}: pooled density"]
        ax.plot([xx + 0.22, xx + 0.22], [r.ci_low, r.ci_high], color=col, linewidth=0.8, zorder=4)
        ax.scatter([xx + 0.22], [r.estimate], s=16, marker="D", c=col, edgecolors="#FFFFFF" if col == C_NOISE else
                   "#333333", linewidths=0.35, zorder=5)
    ax.set_xlim(-0.6, 5.2)
    ax.set_xticks([0, 1, 2.6, 3.6, 4.6], ["Noise", "31 d", "Noise\n(apex)", "31 d\n(lamina)", "14 d\n(apex)"],
                  fontsize=PT_SMALL, linespacing=1.0)
    logaxis(ax, 2.2, 2000, [3, 10, 100, 1000], ["0", "10", "100", "1,000"])
    ax.yaxis.set_minor_locator(FixedLocator([v * d for d in (10, 100, 1000) for v in range(2, 10) if v * d < 2000]))
    for yy in (4.6, 5.4):
        ax.plot([-0.035, 0.035], [yy * 0.93, yy * 1.07], transform=ax.get_yaxis_transform(), color=TEXT,
                linewidth=0.5, clip_on=False, zorder=5)
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    for xx, lab in [(0.5, "Marrow"), (3.6, "Cochlea")]:
        cv.text(11.0 + (xx + 0.6) / 5.8 * 52.0, yc + 7.0 + 34.0 + 5.4, lab, size=PT_SMALL, ha="center", va="top")
    cv.key(36.0, yc + 4.4, [([{"kind": "mark", "s": 4, "fc": "#777777"}], "Section"),
                            ([{"kind": "mark", "s": 16, "marker": "D", "fc": "#BBBBBB", "ec": "#333333", "lw": 0.35},
                              {"kind": "vline", "h": 2.0, "lw": 0.8, "colour": "#777777"}], "Pooled, 95% CI")])

    # c the noise pair among the unexposed pairs (pair level)
    xc = 68.0
    cv.letter(xc, yc, "c", "Per pair")
    ax = cv.axes(xc + 10.5, yc + 7.0, 20.0, 34.0)
    for r in pr.itertuples():
        nz = bool(r.noise)
        col = C_NOISE if nz else C_UNEXP
        ax.plot([0, 1], [r.marrow_density, r.cochlea_density], color=col, linewidth=0.8 if nz else 0.5,
                zorder=3 if nz else 2)
        ax.scatter([0, 1], [r.marrow_density, r.cochlea_density], s=16 if nz else 8, c=col,
                   marker="D" if nz else "o", linewidths=0, zorder=4 if nz else 3)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Marrow", "Cochlea"], fontsize=PT_SMALL)
    logaxis(ax, 10, 1000)
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    zc = cmp_.loc["noise pair vs unexposed pairs: cochlea density"]
    zm = cmp_.loc["noise pair vs unexposed pairs: marrow density"]
    def zf(v):
        return f"{v:.1f}".replace("-0.0", "0.0").replace("-", "\u2212")
    ax.text(1.32, 950, f"z = {zf(zm.z_vs_unexposed)} (marrow)\nz = {zf(zc.z_vs_unexposed)} (cochlea)",
            fontsize=PT_SMALL, ha="right", va="top", color=TEXT, linespacing=1.15)
    cv.key(xc + 3.0, yc + 4.4, [([{"kind": "mark", "s": 16, "marker": "D", "fc": C_NOISE}], "Noise"),
                                ([{"kind": "mark", "s": 8, "fc": C_UNEXP}], "Unexposed")])

    # d noise / unexposed density ratios (within-animal bootstrap, 95% CI)
    xd = 104.0
    cv.letter(xd, yc, "d", "Noise / unexposed")
    rows = [("marrow, noise / unexposed 31 d", "Marrow vs 31 d"),
            ("cochlea, noise apex / unexposed 31 d spiral lamina", "Cochlea vs 31 d"),
            ("cochlear apex, noise / unexposed 14 d", "Apex vs 14 d apex")]
    ax = cv.axes(xd + 21.0, yc + 8.0, 20.0, 18.0)
    for i, (key, _) in enumerate(rows):
        r = cmp_.loc[key]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=TEXT, linewidth=0.8, zorder=2)
        ax.scatter([r.estimate], [i], s=14, marker="D", c=C_NOISE, linewidths=0, zorder=3)
    ax.axvline(1.0, color=TEXT2, linewidth=0.5, linestyle="--", zorder=1)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks(range(len(rows)), [r_[1] for r_ in rows], fontsize=PT_SMALL)
    logaxis(ax, 0.2, 5, [0.2, 1, 5], ["0.2", "1", "5"], axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Density ratio (log scale)", fontsize=PT_SMALL, labelpad=0.8)

    # e perilymph: GFP+ share of every partner (the noise session's detection check)
    xe = 147.0
    cv.letter(xe, yc, "e", "Perilymph")
    ax = cv.axes(xe + 9.0, yc + 7.0, 23.5, 34.0)
    order = [("14 d", "WT"), ("14 d", "GFP"), ("28 d", "WT"), ("28 d", "GFP"), ("31 d noise", "WT"),
             ("31 d noise", "GFP")]
    xs = [0, 1, 2.4, 3.4, 4.8, 5.8]
    for (an, prt), xx in zip(order, xs):
        g = pf[(pf.animal == an) & (pf.partner == prt)]
        if not len(g):
            continue
        col = "#333333" if prt == "WT" else "#2CA02C"
        pooled = 100 * g.gfp_pos.sum() / g.nuclei.sum()
        ax.bar(xx, pooled, width=0.72, color="#E6E6E6", edgecolor=col, linewidth=0.5, zorder=1)
        ax.scatter(xx + rng.uniform(-0.15, 0.15, len(g)), g.gfp_pos_pct, s=3, c=col, linewidths=0, alpha=0.85,
                   zorder=3)
    ax.set_xticks(xs, ["WT", "GFP", "WT", "GFP", "WT", "GFP"], fontsize=PT_SMALL)
    ax.set_xlim(-0.6, 6.4)
    lin(ax, (0, 105), [0, 50, 100])
    ax.set_ylabel("GFP+ (% nucleated)", fontsize=PT_SMALL, labelpad=0.8)
    for xx, lab in [(0.5, "14 d"), (2.9, "28 d"), (5.3, "Noise")]:
        cv.text(xe + 9.0 + (xx + 0.6) / 7.0 * 23.5, yc + 7.0 + 34.0 + 3.4, lab, size=PT_SMALL, ha="center",
                va="top")
    return cv


# ------------------------------------------------------------------------------------------------ Supplementary Fig. 4 (cell types)
_spec938 = importlib.util.spec_from_file_location("s938", str(Path(__file__).resolve().parent /
                                                               "938_parabiosis_cell_types.py"))
S938 = importlib.util.module_from_spec(_spec938)
_spec938.loader.exec_module(S938)
YEL, ORA = "#FFD700", "#FF8C00"                          # GFP+ (majority of the cell) / contact only
S4_RAW = [  # (label, source, field, marker channel)
    ("FUT4, marrow", "ZEISS", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-02.czi", "647"),
    ("FUT4, lateral wall", "ZEISS", "immune_cell_typing/immunecelltype-54d/647fut4-568cd14-gfp-54d-20x-05.czi", "647"),
    ("CD45, marrow and cochlea", "Olympus", "20x_cd45_2", 4),
    ("CD45, cochlea", "Olympus", "20x_cd45_3", 4),
    ("CD19, marrow", "ZEISS", "immune_cell_typing/immunecelltype-54d/647cd19-gfp-54d-5x-04.czi", "647"),
    ("CD14, marrow", "Olympus", "20x_fut4_1", 3)]
# sparse cochlear Olympus fields show GFP with the range of the same series' marrow field (identical settings)
GFP_REF = {"20x_cd45_3": "20x_cd45_2", "20x_fut4_4": "20x_fut4_1", "20x_cd3e_ml": "20x_cd3e_1"}
S4_CROP = [  # (label, source, field, marker, channel, compartment, focus: which typed cells the crop should hold)
    ("FUT4, marrow", "Olympus", "20x_fut4_2", "FUT4", 4, "marrow", "majority"),
    ("CD19, marrow (control)", "ZEISS", "immune_cell_typing/immunecelltype-54d/647cd19-gfp-54d-5x-04.czi", "CD19", "647",
     "marrow", "contact"),
    ("CD45, marrow", "ZEISS", "immune_cell_typing/2026-02-09/parabiosis54d-gfp-647CD45-dapi-20x-ear-bone-02.czi", "CD45",
     "647", "marrow", "majority"),
    ("CD45, cochlea", "Olympus", "20x_cd45_2", "CD45", 4, "cochlea", "typed"),
    ("CD14, marrow", "Olympus", "20x_fut4_1", "CD14", 3, "marrow", "typed"),
    ("CD3e, labyrinth", "Olympus", "20x_cd3e_ml", "CD3e", 4, "cochlea", "none")]
OLY = {fid: (series, folder, stem, comp, box) for series, folder, flist, _ in S938.OSERIES
       for fid, stem, comp, reg, box in flist}


def oly_channels(fid, marker_ch):
    series, folder, stem, comp, box = OLY[fid]
    ch, um = S938.load_olympus(folder, stem, [marker_ch])
    return {"far": ch[marker_ch], "gfp": ch[2], "dapi": ch[1]}, um


def rgb_of(chs, rng, box=None):
    out = None
    for k, col in (("far", MAG), ("dapi", BLUE), ("gfp", GREEN)):
        c = chs[k].astype(np.float32)
        if box is not None:
            c = c[box[0]:box[1], box[2]:box[3]]
        lo, hi = rng[k]
        v = np.clip((c - lo) / max(hi - lo, 1.0), 0, 1)[..., None] * col
        out = v if out is None else out + v
    return np.clip(out, 0, 1)


def ranges_of(chs, field=None):
    rng = {k: (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.9))) for k, v in chs.items()}
    if field in GFP_REF:
        ref, _ = oly_channels(GFP_REF[field], 4)
        rng["gfp"] = (float(np.percentile(ref["gfp"], 0.5)), float(np.percentile(ref["gfp"], 99.9)))
    return rng


def raw_channels(src, field, mk):
    if src == "ZEISS":
        a, meta = load(field)
        idx = roles(meta)
        fi = next(i for i, n in enumerate(meta["channels"]) if mk in n)
        z = a.shape[1] // 2 if a.shape[1] > 1 else 0
        c = CT[(CT.field == field) & CT.typed]
        if len(c):
            z = int(c.plane.mode().iloc[0])
        return {"far": a[fi, z], "gfp": a[idx["gfp"], z], "dapi": a[idx["dapi"], z]}, meta["um_x"], z
    chs, um = oly_channels(field, mk)
    return chs, um, 0


def cell_circles(c, y0, x0, um):
    out = []
    for r in c.itertuples():
        col = YEL if r.gfp_majority else (ORA if r.gfp_contact else "#FFFFFF")
        out.append((r.y - y0, r.x - x0, max(np.sqrt(r.area_um2 / np.pi) / um * 1.1, 3.0), col))   # D143: v2 cells
    return out


def fig_s4():
    """Supplementary cell-type figure (D136; nucleus-anchored typing D142, 961): a raw images, b typed cells circled by
    their GFP call, c partner-derived share per cell type and compartment with Clopper-Pearson 95% CI and the blinded
    verification counts, d typed cells per mm2."""
    global CT
    CT = pd.read_csv(R10 / "celltype_v2_cells.csv")
    CT = CT[CT.included.astype(bool) & CT.typed.astype(bool)].copy()
    CT["gfp_majority"] = CT.gfp_cell.astype(bool)              # the v2 GFP call (body overlap >= 0.5, contrast >= 2)
    CT["gfp_contact"] = CT.gfp_pos.astype(bool)                 # any overlap with a GFP+ body
    sh = pd.read_csv(R9 / "pb_celltype_share.csv")
    H = 6.2 + TILE + 3.6 + 6.2 + TILE + 3.6 + 6.2 + 40.0 + 8.0
    cv = Canvas(180.0, H)
    cv.letter(0, 0, "a", "Raw images (54-day wild-type partner)")
    kx = 3.0 + text_width("Raw images (54-day wild-type partner)", PT, "bold") + 4.0
    cv.key(kx, 2.6, [([{"kind": "box", "fc": "#1AFF1A"}], "GFP"), ([{"kind": "box", "fc": "#4073FF"}], "DAPI"),
                     ([{"kind": "box", "fc": "#F233F2"}], "FUT4, CD45, CD19 or CD14")])
    for k, (lab, src, field, mk) in enumerate(S4_RAW):
        chs, um, z = raw_channels(src, field, mk)
        rng = ranges_of(chs, field)
        rgb = rgb_of(chs, rng)
        png = save_tile("S4", "a", k + 1, rgb)
        show(cv, X0 + k * (TILE + GUT), 6.2, TILE, rgb, um, lab)
        log_tile("S4", "a", k + 1, png, field, z, um, None, rng, ("far", "dapi", "gfp"), lab)
    yb = 6.2 + TILE + 3.6
    cv.letter(0, yb, "b", "Typed cells")
    kx = 3.0 + text_width("Typed cells", PT, "bold") + 4.0
    x = kx
    for col, lab in ((YEL, "GFP+ (partner-derived)"), (ORA, "touching GFP only"), ("#777777", "GFP\u2212")):
        cv.mark(x + 0.9, yb + 2.6, 10.0, "o", "none", "#C9A800" if col == YEL else col, 0.6)
        cv.text(x + 2.2, yb + 2.6, lab, size=PT_SMALL, va="center")
        x += 2.2 + text_width(lab, PT_SMALL) + 3.0
    for k, (lab, src, field, mk, ch, comp, focus) in enumerate(S4_CROP):
        if src == "ZEISS":
            chs, um, z = raw_channels("ZEISS", field, ch)
        else:
            chs, um = oly_channels(field, ch)
            z = 0
        rng = ranges_of(chs, field)
        c = CT[(CT.field == field) & (CT.marker == mk) & (CT.plane == z) & (CT.compartment == comp)]
        n = min(int(round(60.0 / um)), chs["far"].shape[0], chs["far"].shape[1])
        pick = {"contact": c[c.gfp_contact], "majority": c[c.gfp_majority], "typed": c}.get(focus, c.iloc[:0])
        if focus == "none" or not len(pick):
            assert focus == "none", (lab, "no cells to show")
            y0 = (chs["far"].shape[0] - n) // 2
            x0 = (chs["far"].shape[1] - n) // 2
        else:
            y0, x0 = best_box(pick, chs["far"].shape, n, max(len(pick), 1) + 99)
        inside = c[(c.y >= y0) & (c.y < y0 + n) & (c.x >= x0) & (c.x < x0 + n)]
        rgb = rgb_of(chs, rng, (y0, y0 + n, x0, x0 + n))
        png = save_tile("S4", "b", k + 1, rgb)
        show(cv, X0 + k * (TILE + GUT), yb + 6.2, TILE, rgb, um, lab,
             cell_circles(inside, y0, x0, um) if focus != "none" else (), bar=20)
        log_tile("S4", "b", k + 1, png, field, z, um, (y0, y0 + n, x0, x0 + n), rng, ("far", "dapi", "gfp"), lab,
                 len(inside) if focus != "none" else 0)
        if focus != "none":
            for r in inside.itertuples():
                OBJ_LOG.append(dict(figure="S4", panel="b", tile=k + 1, acquisition=field, plane=z,
                                    object=f"{mk}+ cell", x_in_crop_px=r.x - x0, y_in_crop_px=r.y - y0,
                                    area_um2=r.area_um2, gfp_pos=bool(r.gfp_majority)))

    # c partner-derived share per cell type and compartment (Clopper-Pearson 95% CI; blinded verification counts)
    yc = yb + 6.2 + TILE + 3.6
    cv.letter(0, yc, "c", "Partner-derived share")
    rows = list(sh.itertuples())
    lab_w = max(text_width(r.label, PT_SMALL) for r in rows) + 1.2
    ax = cv.axes(4.0 + lab_w, yc + 7.0, 104.0 - 4.0 - lab_w - 22.0, 36.0)
    xmax = 30.0
    for i, r in enumerate(rows):
        col = C_COCH if r.compartment.startswith("cochlea") else C_MARROW
        used = bool(r.used)
        ax.barh(i, r.gfp_pos_pct, height=0.6, color=col if used else "white", edgecolor=col, linewidth=0.5, zorder=2)
        hi = min(r.ci_hi_pct, xmax)
        ax.plot([r.ci_lo_pct, hi], [i, i], color=TEXT, linewidth=0.55, zorder=3, solid_capstyle="butt")
        if r.ci_hi_pct > xmax:
            ax.scatter([xmax], [i], marker=">", s=6, color=TEXT, linewidths=0, zorder=3, clip_on=False)
        yy = yc + 7.0 + (i + 0.5) * 36.0 / len(rows)
        cv.text(3.0 + lab_w, yy, r.label, size=PT_SMALL, ha="right", va="center", color=TEXT if used else TEXT2)
        cv.text(104.0 - 10.0, yy, f"{int(r.gfp_pos_cells)}/{int(r.typed_cells)}", size=PT_SMALL, ha="right",
                va="center")
        cv.text(104.0, yy, r.verified, size=PT_SMALL, ha="right", va="center", color=TEXT2)
    cv.text(104.0 - 10.0, yc + 6.4, "GFP+", size=PT_SMALL, ha="right", va="bottom")
    cv.text(104.0, yc + 6.4, "verified", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    n_c = int(sh.compartment.str.startswith("cochlea").sum())
    ax.axhline(n_c - 0.5, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("GFP+ (% of typed cells)", fontsize=PT_SMALL, labelpad=0.8)
    cv.key(4.0 + lab_w, yc + 7.0 + 36.0 + 7.0, [([{"kind": "box", "fc": "#9A9A9A"}], "share"),
                                                ([{"kind": "hline", "w": 2.4, "lw": 0.55}], "95% CI"),
                                                ([{"kind": "box", "fc": "white", "ec": "#9A9A9A", "lw": 0.5}],
                                                 "not used (calls not confirmed)")])

    # d typed cells per mm2 of tissue
    xd = 112.0
    cv.letter(xd, yc, "d", "Typed cells per mm²")
    ax = cv.axes(xd + 11.0, yc + 7.0, 56.0, 36.0)
    order = [("FUT4", "Neutrophils"), ("CD45", "Leukocytes"), ("CD14", "Monocytes"), ("CD19", "B cells")]
    dens = sh[sh.used.astype(bool)].copy()
    for j, (mk, nm) in enumerate(order):
        for cp, dx in (("marrow", -0.18), ("cochlea", 0.18)):
            r = dens[(dens.marker == mk) & dens.compartment.str.startswith(cp) & dens.typed_per_mm2.notna()]
            if mk == "CD45" and cp == "marrow":
                r = r[r.compartment == "marrow"]
            if not len(r):
                ax.text(j + dx, 1.25, "none" if cp == "cochlea" and mk == "CD14" else "not\nimaged", fontsize=PT_SMALL,
                        ha="center", va="bottom", color=TEXT2, linespacing=1.0)
                continue
            r = r.iloc[0]
            col = C_COCH if cp == "cochlea" else C_MARROW
            ax.bar(j + dx, r.typed_per_mm2, width=0.32, color=col, zorder=2)
    logaxis(ax, 1, 3000)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=PT_SMALL)
    ax.set_ylabel("Cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    cv.key(xd + 13.0, yc + 7.0 + 36.0 + 7.0, [([{"kind": "box", "fc": C_MARROW}], "marrow"),
                                               ([{"kind": "box", "fc": C_COCH}], "cochlea")])
    return cv


def upgraded_objects():
    """934's GFP+ bodies with the cell call of the upgraded rule (960; 963 for the fields 960 did not analyse; D143);
    bodies of fields neither analysed (GFP partners) keep cell = NaN."""
    ob = pd.read_csv(R10 / "parabiosis_objects.csv")
    b = pd.concat([pd.read_csv(R10 / "survey_cells_bodies.csv"), pd.read_csv(R10 / "survey_cells_bodies_extra.csv")],
                  ignore_index=True)
    for d in (ob, b):
        d["_y"], d["_x"] = d.y.round(3), d.x.round(3)
    m = ob.merge(b[["path", "plane", "_y", "_x", "cell"]], on=["path", "plane", "_y", "_x"], how="left")
    assert len(m) == len(ob) and int(m.cell.notna().sum()) == len(b)
    return m.drop(columns=["_y", "_x"])


def main():
    ob = upgraded_objects()
    for fig, build in (("S1", lambda: fig_s1(ob)), ("S2", fig_s2), ("S3", lambda: fig_s3(ob)), ("S4", fig_s4)):
        cv = build()
        for p in cv.save(OUT, NAMES[fig]):
            log(f"saved {p.relative_to(ROOT)}")
    write_csv(fd.prov(pd.DataFrame(IMG_LOG), "r10_parabiosis (934); raw acquisitions", "", "display as described"),
              "r9_figures", "sfig_parabiosis_images.csv", "Supplementary Figs 1-4: every displayed parabiosis image "
              "(file as shown, source acquisition, optical section, crop, linear display range per channel)",
              required=["figure", "panel", "image_file", "acquisition"])
    write_csv(fd.prov(pd.DataFrame(OBJ_LOG), "r10_parabiosis (934)", "", "objects inside the displayed crops"),
              "r9_figures", "sfig_parabiosis_crop_objects.csv", "Supplementary Figs 1-4: objects marked in the "
              "displayed crops (GFP+ bodies, perilymph cells, typed immune cells) with crop coordinates and GFP status",
              required=["figure", "panel", "tile", "object"])


if __name__ == "__main__":
    main()
