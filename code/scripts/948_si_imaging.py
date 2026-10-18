"""Round 10 (D137) — two Supplementary figures with the raw images behind Fig. 4b-d (180 mm wide, 5-7 pt; user: images
"integrated seamlessly within the figures, rather than ... a standalone, large-scale image in the main text"; full-size
raw images belong to the Supplement).

sfig_route_v1
a-b  Per render view: the authors' traced render (teal), the unannotated render and the trace check (941): vessel
     ridges of the unannotated render grey, traced centrelines on a ridge (within 2 px) green, off a ridge red, broad
     paint (wide vessel, bony region) blue. Common 900 x 900-px region (the burned-in scale annotations excluded).
c    On-ridge share of 999 randomly rotated and translated copies of the traces (histogram) and the observed share
     (line), per view.
d    Spatial transcriptomics: deconvolved neutrophil states (the authors' export and colours).
sfig_fluids_v1
a    Cochlear-fluid preparations (Olympus FV3000, 60x): DAPI, FUT4, CD45 and merge of marrow, perilymph and endolymph
     (36-um fields; one linear display range per channel).
b    FUT4 and CD45 of every nucleated cell over its local background (931; filled, >= 2; bar, median; counts above).
c    Positivity across settings (939): FUT4+ and CD45+ shares against the ratio threshold (median, range).
Inputs: figures/panels (930, 941), results/r9_figures (image_panels_log, image_spatial_key, simg_trace_null,
f5a_fut4_cells, srob_c_fluids). Output: figures/figures/sfig_route_v1.*, sfig_fluids_v1.*
Run: COCH_ROOT=/path/to/coch-rythum python scripts/948_si_imaging.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (AXIS, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, halo, minor,  # noqa: E402
                          text_width)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
PANELS = ROOT / "figures" / "panels"
GUT = 0.8
C_MARROW = fd.TISSUE_COLOUR["bm"]
COMP = [("bone_marrow", "Marrow", C_MARROW, "bone_marrow_f3"), ("perilymph", "Perilymph", "#56B4E9", "perilymph_f8"),
        ("endolymph", "Endolymph", "#0B3C8C", "endolymph_f5")]


def tile(cv, x, y, size, arr, um_per_px=None, bar_um=None, label=None):
    ax = cv.axes(x, y, size, size)
    ax.imshow(arr, interpolation="none", cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
    blank(ax)
    if bar_um:
        bar = bar_um / um_per_px / arr.shape[1] * size
        bx1, by = x + size - 1.2, y + size - 1.3
        cv.line([bx1 - bar, bx1], [by, by], "#000000", 1.8, solid_capstyle="butt")
        cv.line([bx1 - bar, bx1], [by, by], "#FFFFFF", 1.0, solid_capstyle="butt")
        cv.text(bx1, by - 0.5, f"{bar_um} µm", size=PT_SMALL, color="#FFFFFF", ha="right", va="bottom",
                path_effects=halo(1.1, "#000000"))
    if label:
        cv.text(x + 1.0, y + 1.0, label, size=PT_SMALL, color="#FFFFFF", ha="left", va="top",
                path_effects=halo(1.1, "#000000"))
    return ax


def route():
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    tn = pd.read_csv(R9 / "simg_trace_null.csv")
    key = pd.read_csv(R9 / "image_spatial_key.csv")
    cv = Canvas(180.0, 170.0)
    S = (180.0 - 2 * GUT) / 3
    for k, letter in ((1, "a"), (2, "b")):
        y = (k - 1) * (S + 6.4)
        cv.letter(0.0, y, letter, f"View {k}")
        um = float(lg.loc[f"vessels_view{k}_annotated", "um_per_px"])
        for j, (nm, lab) in enumerate(((f"vessels_view{k}_annotated", "traced (authors)"),
                                       (f"vessels_view{k}_original", "unannotated"),
                                       (f"trace_check_view{k}", "trace check"))):
            a = np.asarray(Image.open(PANELS / f"{nm}.png").convert("RGB"))[:900, :900]
            tile(cv, j * (S + GUT), y + 5.4, S, a, um, 100, lab)
    yk = 2 * (S + 6.4) + 0.8
    x = cv.key(2 * (S + GUT), yk, [([{"kind": "box", "fc": "#5F5F5F"}], "vessel ridge"),
                                   ([{"kind": "box", "fc": "#28C846"}], "trace on ridge"),
                                   ([{"kind": "box", "fc": "#E62828"}], "off ridge"),
                                   ([{"kind": "box", "fc": "#466EC8"}], "broad paint")], gap=1.6)
    fits(x, 180.5, "route key")
    # c: null distributions
    yc = yk + 4.0
    cv.letter(0.0, yc, "c", "Random placements")
    for k in (1, 2):
        g = tn[tn.view == k]
        obs = 100 * float(g.thin_on_ridge_share.iloc[0])
        ax = cv.axes(10.0 + (k - 1) * 44.0, yc + 7.0, 34.0, 170.0 - yc - 7.0 - 8.0)
        ax.hist(100 * g.on_ridge_share, bins=np.arange(15, 101, 2.5), color="#9A9A9A", lw=0)
        ax.axvline(obs, color=TEXT, lw=0.9)
        ax.text(obs - 1.0, ax.get_ylim()[1] * 0.96, f"traced\n{obs:.0f}%", fontsize=PT_SMALL, ha="right", va="top")
        ax.set_xlim(15, 100)
        ax.set_xticks([20, 40, 60, 80, 100])
        minor(ax, x=True, y=False)
        clean(ax)
        ax.set_xlabel(f"Trace on vessel ridge (%), view {k}", fontsize=PT_SMALL, labelpad=0.8)
        if k == 1:
            ax.set_ylabel("Placements", fontsize=PT_SMALL, labelpad=0.8)
    # d: spatial states
    xd = 98.0
    cv.letter(xd, yc, "d", "Neutrophil states, spatial")
    side = 170.0 - yc - 6.0
    a = np.asarray(Image.open(PANELS / "spatial_neutrophil_states.png").convert("RGB"))
    tile(cv, xd, yc + 5.4, side, a)
    col = dict(zip(key.state, key.colour))
    for i, st in enumerate(["Band", "Mature", "Activated", "Aged"]):
        cv.key(xd + side + 2.0, yc + 7.4 + i * 2.6, [([{"kind": "box", "fc": col[st], "ec": AXIS, "lw": 0.3}], st)])
    for p in cv.save(OUT, "sfig_route_v1"):
        print(f"[fig] {p.relative_to(ROOT)}")


def fluids():
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    q = pd.read_csv(R9 / "f5a_fut4_cells.csv")
    rb = pd.read_csv(R9 / "srob_c_fluids.csv")
    cv = Canvas(180.0, 102.0)
    cv.letter(0.0, 0.0, "a", "Cochlear-fluid preparations")
    T, lab = 22.0, 4.0
    rows = [("DAPI", "DAPI"), ("FUT4", "FUT4"), ("CD45", "CD45"), ("merge", "Merge")]
    top = 9.0
    for j, (comp, name, colour, tag) in enumerate(COMP):
        um = float(lg.loc[f"fut4_{tag}", "um_per_px"])
        n = int(round(36.0 / um))
        x = lab + j * (T + GUT)
        cv.text(x + T / 2, top - 0.6, name, size=PT_SMALL, ha="center", va="bottom")
        for i, (ch, _) in enumerate(rows):
            a = np.asarray(Image.open(PANELS / f"fut4_{tag}_{ch}.png"))
            o = (a.shape[0] - n) // 2
            tile(cv, x, top + i * (T + GUT), T, a[o:o + n, o:o + n], um if (j == 2 and i == 3) else None,
                 10 if (j == 2 and i == 3) else None)
    for i, (_, lab_) in enumerate(rows):
        cv.text(lab - 1.2, top + i * (T + GUT) + T / 2, lab_, size=PT_SMALL, rotation=90, ha="center", va="center")
    # b: per-cell values over local background
    xb = lab + 3 * (T + GUT) + 8.0
    cv.letter(xb - 6.0, 0.0, "b", "Per cell")
    for k, (m, name) in enumerate((("fut4", "FUT4"), ("cd45", "CD45"))):
        ax = cv.axes(xb + 4.0 + k * 26.0, top + 3.0, 20.0, 50.0)
        rng = np.random.default_rng(k)
        for i, (comp, cname, colour, _) in enumerate(COMP):
            d = q[q.compartment == comp]
            v = d[f"{m}_ratio"]
            pos = v >= 2.0
            xj = i + rng.uniform(-0.18, 0.18, len(d))
            ax.scatter(xj[pos.values], v[pos], s=3.0, color=colour, linewidths=0, zorder=3)
            ax.scatter(xj[~pos.values], v[~pos], s=3.0, facecolor="none", edgecolor="#777777", linewidths=0.35,
                       zorder=3)
            ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=TEXT, lw=0.8, zorder=4)
            ax.text(i, 52.0, f"{int(pos.sum())}/{len(d)}", fontsize=PT_SMALL, ha="center", va="bottom")
        ax.axhline(2.0, color="#8C8C8C", lw=0.5, ls=(0, (2.0, 1.4)), zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 50.0)
        ax.yaxis.set_major_locator(FixedLocator([1, 10]))
        ax.set_yticklabels(["1", "10"])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks(range(3), [c[1] for c in COMP], rotation=90)
        clean(ax)
        ax.set_title(name, fontsize=PT_SMALL, fontweight="bold", pad=7.0)
        if k == 0:
            ax.set_ylabel("Cell / background", fontsize=PT_SMALL, labelpad=0.8)
    # c: robustness of the positivity threshold
    xc = xb + 4.0 + 2 * 26.0 + 4.0
    cv.letter(xc - 3.0, 0.0, "c", "Across settings")
    for k, (m, name) in enumerate((("fut4", "FUT4+"), ("cd45", "CD45+"))):
        ax = cv.axes(xc + 7.0, top + 3.0 + k * 30.0, 180.0 - xc - 8.0, 22.0)
        for comp, cname, colour, _ in COMP:
            g = rb[rb.compartment == comp].sort_values("ratio")
            ax.fill_between(g.ratio, g[f"{m}_min"], g[f"{m}_max"], color=colour, alpha=0.18, lw=0)
            ax.plot(g.ratio, g[f"{m}_median"], color=colour, lw=0.9, marker="o", ms=2.0)
        ax.axvline(2.0, color="#8C8C8C", lw=0.5, ls=(0, (2.0, 1.4)), zorder=0)
        ax.set_ylim(40, 100)
        ax.set_yticks([40, 70, 100])
        ax.set_xticks([1.5, 2, 3], ["1.5", "2", "3"] if k == 1 else ["", "", ""])
        clean(ax)
        ax.set_ylabel(f"{name} (%)", fontsize=PT_SMALL, labelpad=0.8)
        if k == 1:
            ax.set_xlabel("Ratio threshold", fontsize=PT_SMALL, labelpad=0.8)
    yk = top + 3.0 + 2 * 30.0 + 2.0
    for i, (comp, cname, colour, _) in enumerate(COMP):
        cv.line([xc + 7.0, xc + 9.2], [yk + i * 2.5, yk + i * 2.5], colour, 0.9)
        cv.text(xc + 9.8, yk + i * 2.5, cname, size=PT_SMALL, va="center")
    for p in cv.save(OUT, "sfig_fluids_v1"):
        print(f"[fig] {p.relative_to(ROOT)}")


def main():
    route()
    fluids()


if __name__ == "__main__":
    main()
