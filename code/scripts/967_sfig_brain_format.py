"""Round 10 (D152, D153, D155) — Supplementary Fig. 5: the brain, the brain-image format and the counting (180 x 160 mm).
D155: GFP+ bodies are detected on the local background (969; fixed before its run and used because it agreed better with
the blinded labels) and called by the strict criteria (contrast >= 6).
User: "Relevant photographic evidence and similar materials can be added to the supplementary figures", "I would like
the number of cochleae to be comparable to the number of brains" and "The current quantitative criteria for the cochlea
are too broad; they should be made more stringent".
a  Brain of the 31-day wild-type partner: the rendered maximum projection (GFP green, lectin magenta; each linear from
   its tissue median to its 99.9th percentile, GFP at least to the body threshold) with every GFP+ body (yellow, counted
   as a partner-derived cell by the strict rule of 968 in the brain format at the assumed 2.773 um per pixel; grey, not
   counted; 966, 969).
b  Six counted brain cells drawn at random among the strict-call cells (seed 969), 120-um crops.
c  The 31-day cochlea (osseous spiral lamina), marrow and liver sections of Supplementary Fig. 2a converted to the brain's
   format (box-averaged to 2.773 um per pixel; GFP and lectin, no DAPI) with their bodies (966), and the central 710-um
   window of the brain, the field size of the cochlear section; yellow, strict-rule cells.
d  Strict-rule density, native (20 robust SD) against the brain format (contrast >= 6, sections at 2.773 um per pixel),
   per wild-type animal and tissue; line, identity.
e  The 31-day partner's tissues in the brain format across the assumed pixel sizes (left to right 2.773, 1.387, 1.248
   and 1.040 um): filled, strict call; open, upgraded criteria (the brain, a projection, as diamonds).
f  Agreement with the blinded labels (152 decided labels of 946 / 960), brain format: precision (filled) and
   sensitivity (open) of the survey and the local-background detector, each with the upgraded and the strict call.
Inputs: results/r9_figures (900 build_stc: stc_display_tiles, stc_display_objects, stc_native_vs_brain_format,
stc_sensitivity, stc_rule_calibration) and figures/panels/mf_*.png (966). No statistics are computed here.
Output: figures/figures/sfig_brain_format_v1.*   Run: COCH_ROOT=D:/coch-rythum python scripts/967_sfig_brain_format.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, text_width  # noqa: E402
from lib.io import ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("s935", str(Path(__file__).resolve().parent / "935_parabiosis_figure.py"))
S935 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S935)
R9 = ROOT / "results" / "r9_figures"
PANELS = ROOT / "figures" / "panels"
OUT = ROOT / "figures" / "figures"
NAME = "sfig_brain_format_v1"
YELLOW, GREY = "#FFD700", "#B3B3B3"
C_TISSUE = {"cochlea": fd.TISSUE_COLOUR["cochlea"], "marrow": fd.TISSUE_COLOUR["bm"], "liver": "#009E73",
            "vestibular bone": "#7F7F7F", "brain": "#CC79A7"}
LABEL = {"cochlea": "Cochlea", "marrow": "Marrow", "liver": "Liver", "vestibular bone": "Vestibular bone",
         "brain": "Brain"}
PIX = [2.773, 1.387, 1.248, 1.04]


def tile(name):
    return np.asarray(Image.open(PANELS / name).convert("RGB"))


def circles(ob, tile_name, r_um, um, y0=0, x0=0):
    o = ob[ob["tile"] == tile_name]
    return [(r.y - y0, r.x - x0, r_um / um, YELLOW if r.strict else GREY) for r in o.itertuples()]


def logaxis(ax, lo, hi, axis="y"):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}lim")(lo, hi)
    dec = [10.0 ** k for k in range(-3, 5) if lo <= 10.0 ** k <= hi]
    a_ = ax.yaxis if axis == "y" else ax.xaxis
    a_.set_major_locator(FixedLocator(dec))
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)([f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in dec])
    a_.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    a_.set_minor_formatter(NullFormatter())


def panel_brain(cv, x0, y0, size, lg, ob):
    cv.letter(x0, y0, "a", "Brain, 31 d (projection)")
    r = lg.set_index("tile").loc["mf_brain_31d.png"]
    S935.show(cv, x0, y0 + 5.0, size, tile("mf_brain_31d.png"), r.um_per_px,
              circles=circles(ob, "mf_brain_31d.png", 30.0, r.um_per_px), bar=500)


def panel_crops(cv, x0, y0, w, lg, ob):
    cv.letter(x0, y0, "b", "Counted brain cells (random)")
    crops = lg[lg["tile"].str.contains("crop")].reset_index(drop=True)
    assert len(crops) == 6
    size, gap = (w - 2 * 2.0) / 3, 2.0
    for k, r in crops.iterrows():
        x, y = x0 + (k % 3) * (size + gap), y0 + 5.0 + (k // 3) * (size + gap)
        S935.show(cv, x, y, size, tile(r.tile), r.um_per_px, circles=circles(ob, r.tile, 9.0, r.um_per_px), bar=20)


def panel_format(cv, x0, y0, w, lg, ob):
    cv.letter(x0, y0, "c", "Brain-image format, 31 d")
    li = lg.set_index("tile")
    size, gap = (w - 3 * 3.0) / 4, 3.0
    for k, comp in enumerate(["cochlea", "marrow", "liver"]):
        name = f"mf_{comp}_31d.png"
        S935.show(cv, x0 + k * (size + gap), y0 + 5.0, size, tile(name), li.loc[name, "um_per_px"], label=LABEL[comp],
                  circles=circles(ob, name, 6.0 if comp == "liver" else 12.0, li.loc[name, "um_per_px"]))
    um = li.loc["mf_brain_31d.png", "um_per_px"]                    # brain window of the cochlear section's field size
    side = li.loc["mf_cochlea_31d.png", "width_px"]
    full = tile("mf_brain_31d.png")
    c0 = (full.shape[0] - side) // 2
    sub = full[c0:c0 + side, c0:c0 + side]
    cir = [c for c in circles(ob, "mf_brain_31d.png", 12.0, um, c0, c0) if 0 <= c[0] < side and 0 <= c[1] < side]
    S935.show(cv, x0 + 3 * (size + gap), y0 + 5.0, size, sub, um, label="Brain", circles=cir)


def panel_native(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Native vs brain format")
    nv = pd.read_csv(R9 / "stc_native_vs_brain_format.csv")
    ax = cv.axes(x0 + 11.0, y0 + 6.0, h - 13.0, h - 13.0)
    lo, hi = 1.0, 1000.0
    ax.plot([lo, hi], [lo, hi], color=TEXT2, linewidth=0.5, linestyle=(0, (2, 1.5)), zorder=1)
    for r in nv.itertuples():
        ax.scatter([r.native_density_per_mm2], [r.brain_format_density_per_mm2], s=10, c=C_TISSUE[r.compartment],
                   linewidths=0, zorder=3)
    assert nv[["native_density_per_mm2", "brain_format_density_per_mm2"]].stack().between(lo, hi).all()
    logaxis(ax, lo, hi, "x")
    logaxis(ax, lo, hi, "y")
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Native", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Brain format", fontsize=PT_SMALL, labelpad=0.8)
    kx = x0 + 11.0 + (h - 13.0) + 3.0
    for i, comp in enumerate(["liver", "marrow", "vestibular bone", "cochlea"]):
        ky = y0 + 7.0 + 2.6 * i
        cv.mark(kx + 0.8, ky, s=10, fc=C_TISSUE[comp])
        cv.text(kx + 2.2, ky, LABEL[comp].replace("Vestibular bone", "Vestib. bone"), size=PT_SMALL, va="center")
    fits(kx + 2.2 + text_width("Vestib. bone", PT_SMALL), x0 + w + 0.5, "d key")


def panel_sensitivity(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Pixel size and call, 31 d")
    sv = pd.read_csv(R9 / "stc_sensitivity.csv")
    order = ["liver", "marrow", "cochlea", "brain"]
    ax = cv.axes(x0 + 12.0, y0 + 6.0, w - 36.0, h - 13.0)
    off = dict(zip(PIX, [-0.27, -0.09, 0.09, 0.27]))
    vals = pd.concat([sv["strict_density_per_mm2"], sv["cell_density_per_mm2"]])
    lo, hi = 0.5, 1000.0
    assert vals[vals > 0].between(lo, hi).all()
    floor = lo * 1.12                                                   # no counted cell: drawn at the axis floor
    for r in sv.itertuples():
        x = order.index(r.compartment) + off[min(PIX, key=lambda u: abs(u - r.um_per_px))]
        col = C_TISSUE[r.compartment]
        mk = "D" if r.compartment == "brain" else "o"
        ys = r.strict_density_per_mm2 if r.strict_density_per_mm2 > 0 else floor
        yu = r.cell_density_per_mm2 if r.cell_density_per_mm2 > 0 else floor
        ax.plot([x, x], [ys, yu], color=col, linewidth=0.4, zorder=2)
        ax.scatter([x], [yu], s=10, marker=mk, facecolors="white", edgecolors=col, linewidths=0.6, zorder=3,
                   clip_on=False)
        ax.scatter([x], [ys], s=10, marker=mk, c=col, linewidths=0, zorder=4, clip_on=False)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(range(len(order)), [LABEL[c] for c in order], fontsize=PT_SMALL)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    logaxis(ax, lo, hi)
    ax.set_yticks([floor, 10, 100, 1000], ["0", "10", "100", "1,000"])
    clean(ax)
    ax.set_ylabel("Cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    kx = x0 + w - 22.0
    cv.mark(kx + 0.8, y0 + 7.0, s=10, fc=TEXT2)
    cv.text(kx + 2.2, y0 + 7.0, "strict", size=PT_SMALL, va="center")
    cv.mark(kx + 0.8, y0 + 9.6, s=10, fc="white", ec=TEXT2, lw=0.6)
    cv.text(kx + 2.2, y0 + 9.6, "upgraded", size=PT_SMALL, va="center")
    cv.text(kx, y0 + 13.0, "pixel size (µm),", size=PT_SMALL, va="center", color=TEXT2)
    cv.text(kx, y0 + 15.6, "left to right:", size=PT_SMALL, va="center", color=TEXT2)
    cv.text(kx, y0 + 18.2, "2.77, 1.39,", size=PT_SMALL, va="center", color=TEXT2)
    cv.text(kx, y0 + 20.8, "1.25, 1.04", size=PT_SMALL, va="center", color=TEXT2)
    fits(kx + text_width("pixel size (µm),", PT_SMALL), x0 + w + 0.5, "e key")


def panel_calibration(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Agreement with blinded labels")
    ch = pd.read_csv(R9 / "stc_rule_calibration.csv")
    ch = ch[ch["scope"] == "all labels"].set_index("call")
    rows = [("survey detector, upgraded criteria (966)", "Survey, upgraded"),
            ("survey detector, strict rule (968)", "Survey, strict"),
            ("local background, upgraded criteria (primary)", "Local bg., upgraded"),
            ("local background, contrast >= 6", "Local bg., strict")]
    lab_w = max(text_width(r[1], PT_SMALL) for r in rows) + 1.5
    ax = cv.axes(x0 + lab_w, y0 + 6.0, w - lab_w - 3.5, h - 13.0)
    for k, (call, _) in enumerate(rows):
        pr_, se = ch.loc[call, "precision"], ch.loc[call, "sensitivity"]
        ax.plot([pr_, se], [k, k], color=GREY, linewidth=0.6, zorder=1)
        ax.scatter([pr_], [k], s=10, c=TEXT, linewidths=0, zorder=3)
        ax.scatter([se], [k], s=10, facecolors="white", edgecolors=TEXT, linewidths=0.6, zorder=3)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=PT_SMALL)
    ax.set_xlim(0.3, 1.0)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.set_xlabel("Precision (filled), sensitivity (open)", fontsize=PT_SMALL, labelpad=0.8)


def main():
    lg = pd.read_csv(R9 / "stc_display_tiles.csv")
    ob = pd.read_csv(R9 / "stc_display_objects.csv")
    cv = Canvas(180.0, 160.0)
    panel_brain(cv, 0.0, 0.0, 70.0, lg, ob)
    panel_crops(cv, 74.0, 0.0, 106.0, lg, ob)
    panel_format(cv, 0.0, 79.0, 180.0, lg, ob)
    panel_native(cv, 0.0, 128.0, 52.0, 32.0)
    panel_sensitivity(cv, 54.0, 128.0, 70.0, 32.0)
    panel_calibration(cv, 126.0, 128.0, 54.0, 32.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
