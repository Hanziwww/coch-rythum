"""Round 10 (D150) — discussion draft of the ear (cochlea) versus brain, liver and blood comparison (964), drawn before
its placement is decided (figures/drafts/tissue_comparison_draft_v1.*; not a manuscript figure).
  a  Partner-derived cells per mm2 of tissue in the wild-type partners (upgraded rule): liver, marrow, cochlea and brain
     of the 31-day partner (one animal; the brain from its rendered projection, range over the collection's overview
     pixel sizes), with the marrow and cochlea of the 14- and 54-day partners.
  b  Partner-derived share of cells: published blood neutrophils (shaded, 22-30%; external) against the perilymph cells
     (14 and 28 days; filled, raw; open, per labelled cell), the cochlear neutrophils and the marrow neutrophils of the
     54-day partner; Clopper-Pearson 95% CI (blood, mean +/- 1.96 SEM).
Run: COCH_ROOT=D:/coch-rythum python scripts/965_tissue_comparison_draft.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, LW, LW_THIN, PT, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, minor,  # noqa: E402
                          text_width)
from lib.io import RESULTS, ROOT, log  # noqa: E402

R10 = RESULTS / "r10_parabiosis"
OUT = ROOT / "figures" / "drafts"
DAY_COL = {14: "#56B4E9", 31: "#CC79A7", 54: "#000000"}              # as Fig. 4g
C_COCH, C_MARROW, C_BLOOD = "#0072B2", "#D55E00", "#7A0403"
TISSUES = [("liver", "Liver"), ("marrow", "Marrow"), ("cochlea", "Cochlea"), ("brain", "Brain")]
TOP, BOTTOM = 7.5, 9.0


def logaxis(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    dec = [10.0 ** k for k in range(-3, 5) if lo <= 10.0 ** k <= hi]
    ax.yaxis.set_major_locator(FixedLocator(dec))
    ax.set_yticklabels([f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in dec])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel_density(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Partner-derived cells by tissue")
    d = pd.read_csv(R10 / "tissue_comparison_density.csv")
    pos = {k: i for i, (k, _) in enumerate(TISSUES)}
    ax = cv.axes(x0 + 12.0, y0 + TOP, w - 26.0, h - TOP - BOTTOM)
    lo_lim, hi_lim = 0.5, 1000.0
    for days, col in DAY_COL.items():
        g = d[(d.days == days) & d.main & d.compartment.isin(pos)].copy()
        g["x"] = g.compartment.map(pos)
        g = g.sort_values("x")
        ax.plot(g.x, g.cell_density_per_mm2, color=col, linewidth=0.7, zorder=2)
        for r in g.itertuples():
            proj = r.compartment == "brain"
            ax.scatter([r.x], [r.cell_density_per_mm2], s=12 if not proj else 14, marker="D" if proj else "o",
                       c=col, linewidths=0, zorder=3)
    br = d[(d.compartment == "brain")]
    ax.plot([pos["brain"] + 0.16] * 2, [br.cell_density_per_mm2.min(), br.cell_density_per_mm2.max()], color=TEXT,
            linewidth=0.7, solid_capstyle="butt", zorder=2)
    for v in (br.cell_density_per_mm2.min(), br.cell_density_per_mm2.max()):
        ax.plot([pos["brain"] + 0.12, pos["brain"] + 0.20], [v, v], color=TEXT, linewidth=0.5, zorder=2)
    vals = d[d.main & d.compartment.isin(pos)].cell_density_per_mm2
    assert vals.between(lo_lim, hi_lim).all() and br.cell_density_per_mm2.between(lo_lim, hi_lim).all()
    ax.set_xlim(-0.45, 3.45)
    ax.set_xticks(range(len(TISSUES)), [lab for _, lab in TISSUES], fontsize=PT_SMALL)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    logaxis(ax, lo_lim, hi_lim)
    clean(ax)
    ax.set_ylabel("GFP+ cells per mm² (log scale)", fontsize=PT_SMALL, labelpad=0.8)
    ref = d[(d.days == 31) & (d.compartment == "cochlea")].cell_density_per_mm2.iloc[0]
    for k in ("liver", "brain"):
        v = d[(d.days == 31) & d.main & (d.compartment == k)].cell_density_per_mm2.iloc[0]
        ax.text(pos[k] - 0.16, v, f"×{v / ref:.2g}", fontsize=PT_SMALL, ha="right", va="center", color=TEXT2)
    kx = x0 + w - 12.5                                                  # key: pairs; brain = projection
    for i, (days, col) in enumerate(DAY_COL.items()):
        ky = y0 + TOP + 1.0 + 2.6 * i
        cv.line([kx, kx + 2.2], [ky, ky], col, 0.8)
        cv.text(kx + 2.7, ky, f"{days} d", size=PT_SMALL, va="center")
    ky = y0 + TOP + 1.0 + 2.6 * 3 + 0.6
    cv.mark(kx + 1.1, ky, s=10, marker="D", fc=TEXT2)
    cv.text(kx + 2.7, ky, "projection", size=PT_SMALL, va="center")
    cv.line([kx + 1.1, kx + 1.1], [ky + 1.6, ky + 3.6], TEXT, 0.7)
    cv.text(kx + 2.7, ky + 2.6, "pixel size", size=PT_SMALL, va="center")
    fits(kx + 2.7 + text_width("projection", PT_SMALL), x0 + w + 0.5, "a key")


def panel_share(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Partner-derived share")
    s = pd.read_csv(R10 / "tissue_comparison_share.csv")
    blood = s[s.tissue == "blood"].sort_values("pct")
    peri = s[s.population.str.startswith("perilymph")].sort_values("animal")
    rows = [("Neutrophils, 1 mo", "blood", blood.iloc[0]), ("Neutrophils, 60 d", "blood", blood.iloc[1]),
            ("Perilymph cells, 14 d", "cochlea", peri.iloc[0]), ("Perilymph cells, 28 d", "cochlea", peri.iloc[1]),
            ("Neutrophils, 54 d", "cochlea", s[s.population.str.startswith("cochlear")].iloc[0]),
            ("Neutrophils, 54 d", "marrow", s[s.population.str.startswith("marrow")].iloc[0])]
    lab_w = max(text_width(lab, PT_SMALL) for lab, _, _ in rows) + 5.0
    ah = h - TOP - BOTTOM
    ax = cv.axes(x0 + lab_w, y0 + TOP, w - lab_w - 1.0, ah)
    xmax = 40.0
    ax.axvspan(blood.pct.min(), blood.pct.max(), color="#F2E1E1", zorder=0, linewidth=0)
    col = {"blood": C_BLOOD, "cochlea": C_COCH, "marrow": C_MARROW}
    for k, (lab, grp, r) in enumerate(rows):
        c = col[grp]
        if np.isfinite(r.ci_lo_pct):
            ax.plot([r.ci_lo_pct, r.ci_hi_pct], [k, k], color=c, linewidth=0.6, zorder=2, solid_capstyle="butt")
        ax.scatter([r.pct], [k], s=12, c=c, linewidths=0, zorder=3, clip_on=False)
        if grp == "cochlea" and "per_labelled_pct" in r and np.isfinite(r.per_labelled_pct):
            ax.plot([r.per_labelled_lo_pct, r.per_labelled_hi_pct], [k + 0.28, k + 0.28], color=c, linewidth=0.45,
                    zorder=2, linestyle=(0, (1.5, 1.0)))
            ax.scatter([r.per_labelled_pct], [k + 0.28], s=10, facecolors="white", edgecolors=c, linewidths=0.6,
                       zorder=3)
        if r.tissue != "blood" and np.isfinite(r.cells):
            ax.text(xmax, k, f"{int(r.partner_derived)}/{int(r.cells)}", fontsize=PT_SMALL, ha="right", va="center",
                    color=TEXT2)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks(range(len(rows)), [lab for lab, _, _ in rows], fontsize=PT_SMALL)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 10, 20, 30, 40])
    minor(ax, x=True, y=False)
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.set_xlabel("Partner-derived (% of cells)", fontsize=PT_SMALL, labelpad=0.8)
    kx = x0 + w - 1.0 - text_width("per labelled cell", PT_SMALL) - 2.7         # key: raw / per labelled cell
    for i, (lab, fc) in enumerate((("raw", C_COCH), ("per labelled cell", "white"))):
        cv.mark(kx + 1.0, y0 + 2.0 + 2.6 * i, s=10, fc=fc, ec=C_COCH, lw=0.6)
        cv.text(kx + 2.7, y0 + 2.0 + 2.6 * i, lab, size=PT_SMALL, va="center")
    rh = ah / len(rows)
    for (a_, b_), lab, c in (((0, 2), "Blood*", C_BLOOD), ((2, 5), "Cochlea", C_COCH), ((5, 6), "Marrow", C_MARROW)):
        ya, yb = y0 + TOP + a_ * rh + 0.4, y0 + TOP + b_ * rh - 0.4
        cv.line([x0 + 2.4, x0 + 2.4], [ya, yb], c, 0.8)
        cv.text(x0 + 1.2, (ya + yb) / 2, lab, size=PT_SMALL, ha="center", va="center", rotation=90, fontweight="bold")


def main():
    cv = Canvas(180.0, 58.0)
    panel_density(cv, 0.0, 0.0, 84.0, 58.0)
    panel_share(cv, 88.0, 0.0, 92.0, 58.0)
    cv.text(88.0, 57.0, "*published parabiosis (Casanova-Acebes 2018; Cugurra 2021)", size=PT_SMALL, va="bottom",
            color=TEXT2)
    paths = cv.save(OUT, "tissue_comparison_draft_v1")
    log("draft: " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
