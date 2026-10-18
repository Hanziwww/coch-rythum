"""Round 10 (D137 / D142) — Supplementary figure: robustness of the quantification and blinded verification of the
image calls behind Fig. 4 and Fig. 5c (180 x 170 mm, 5-7 pt; user: "Please ensure that the current quantification and
labeling algorithms are sufficiently robust (or verify that the manual labeling is accurate)").

a  GFP+ body detection (934): cochlea / marrow density per wild-type pair against the threshold k (x robust SD); line,
   median over minimum area and solidity; band, range; diamond, published setting; dashed, 1 (log axis).
b  Perilymph ring rule: wild-type / GFP-partner GFP+ share at every ring width and ratio threshold (log axis).
c  Cochlear fluids: FUT4+ share against the positivity ratio per preparation (line, median; band, range over cell
   expansion and background annulus); dashed, the published ratio 2.
d  Upgraded survey cell rule (960): cochlea / marrow ratio per pair at each of 27 settings (dots; local contrast,
   minimum area, nucleus share) and at the published setting (bar); log axis; dashed, 1.
e  Label-corrected cochlea / marrow ratio per pair (filled; kept x precision + rejected x miss rate from the blinded
   labels, stratified bootstrap 95% CI) beside the rule's ratio (open); log axis.
f  Flow gates (932): ZT17 / ZT5 ratio of P1-P3 at each of 31 gate settings (dots; open, not significant after Holm)
   and at the published gates with its 95% CI (log axis; dashed, 1).
g  Nucleus-anchored typing (961): partner-derived share per cell type at each of 243 settings (dots; ring coverage,
   marker contrast, ring sectors, GFP overlap, GFP contrast) and the published setting (bar).
h  Blinded verification of the typing (945 development and 961 validation samples pooled; 944 for the 60x lateral
   wall): typed cells confirmed as single marker+ cells, untyped cells correctly negative, and GFP calls of confirmed
   cells (Wilson 95% CI; unsure excluded).
Inputs: results/r9_figures/srob_*.csv (900 build_srob). Output: figures/figures/sfig_robustness_v2.*
Run: COCH_ROOT=/path/to/coch-rythum python scripts/949_si_robustness.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, minor, ref_line,  # noqa: E402
                          text_width)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "sfig_robustness_v2"
C_COCH, C_MARROW = fd.TISSUE_COLOUR["cochlea"], fd.TISSUE_COLOUR["bm"]
PAIR_COL = {"14 d": "#56B4E9", "31 d": "#CC79A7", "31 d noise": "#CC79A7", "54 d": "#000000"}
COMP_COL = {"bone_marrow": C_MARROW, "perilymph": "#56B4E9", "endolymph": "#0B3C8C"}
TOP, BOT = 7.5, 7.5


def logaxis(ax, lo, hi, axis="y"):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}lim")(lo, hi)
    a = ax.yaxis if axis == "y" else ax.xaxis
    dec = [10.0 ** k for k in range(-3, 4) if lo <= 10.0 ** k <= hi]
    a.set_major_locator(FixedLocator(dec))
    a.set_ticklabels([f"{t:g}" for t in dec])
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    a.set_minor_formatter(NullFormatter())


def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "GFP+ body detection")
    d = pd.read_csv(R9 / "srob_a_gfp_bodies.csv")
    ax = cv.axes(x0 + 10.0, y0 + TOP, w - 22.0, h - TOP - BOT)
    for an, g in d.groupby("animal"):
        g = g.sort_values("k")
        col = PAIR_COL[an]
        ls = (0, (2.0, 1.2)) if "noise" in an else "-"
        ax.fill_between(g.k, g.ratio_min, g.ratio_max, color=col, alpha=0.15, lw=0)
        ax.plot(g.k, g.ratio_median, color=col, lw=0.9, ls=ls)
        ax.scatter([10], [g.published.iloc[0]], marker="D", s=9, color=col, zorder=4, linewidths=0)
    ref_line(ax, 1.0, "h", "#8C8C8C", 0.5)
    logaxis(ax, 0.03, 3.0)
    ax.set_xticks([6, 8, 10, 12, 14])
    clean(ax)
    ax.set_xlabel("Threshold (× robust SD)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Cochlea / marrow", fontsize=PT_SMALL, labelpad=0.8)
    for i, an in enumerate(["14 d", "31 d", "31 d noise", "54 d"]):
        yy = y0 + TOP + 1.0 + i * 2.5
        cv.line([x0 + w - 11.0, x0 + w - 8.8], [yy, yy], PAIR_COL[an], 0.9,
                linestyle=(0, (2.0, 1.2)) if "noise" in an else "-")
        cv.text(x0 + w - 8.2, yy, an, size=PT_SMALL, va="center")


def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Perilymph ring rule")
    d = pd.read_csv(R9 / "srob_b_perilymph.csv")
    ax = cv.axes(x0 + 10.0, y0 + TOP, w - 11.0, h - TOP - BOT)
    rng = np.random.default_rng(2)
    for i, an in enumerate(["14 d", "28 d"]):
        g = d[d.animal == an]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g.wt_over_gfp, s=6, color=TEXT, linewidths=0, zorder=3)
        b_ = g[(g.ring_um == 2.0) & (g.ratio == 2.0)]
        ax.plot([i - 0.28, i + 0.28], [b_.wt_over_gfp.iloc[0]] * 2, color=TEXT, lw=0.9, zorder=4)
    ref_line(ax, 1.0, "h", "#8C8C8C", 0.5)
    logaxis(ax, 0.01, 1.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["14 d", "28 d"])
    clean(ax)
    ax.set_ylabel("WT / GFP partner", fontsize=PT_SMALL, labelpad=0.8)
    cv.text(x0 + 11.0, y0 + h - 2.0, "bar, published; dots, 9 settings", size=PT_SMALL, va="center", color=TEXT2)


def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Fluid-cell positivity")
    d = pd.read_csv(R9 / "srob_c_fluids.csv")
    ax = cv.axes(x0 + 10.0, y0 + TOP, w - 30.0, h - TOP - BOT)
    for comp, g in d.groupby("compartment"):
        g = g.sort_values("ratio")
        col = COMP_COL[comp]
        ax.fill_between(g.ratio, g.fut4_min, g.fut4_max, color=col, alpha=0.18, lw=0)
        ax.plot(g.ratio, g.fut4_median, color=col, lw=0.9, marker="o", ms=2.2)
    ax.axvline(2.0, color="#8C8C8C", lw=0.5, ls=(0, (2.0, 1.4)), zorder=0)
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 60, 80, 100])
    minor(ax, x=False, y=True)
    ax.set_xticks([1.5, 2, 3], ["1.5", "2", "3"])
    clean(ax)
    ax.set_xlabel("Positivity ratio (cell / background)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("FUT4+ (% of cells)", fontsize=PT_SMALL, labelpad=0.8)
    for i, (comp, name) in enumerate([("bone_marrow", "Marrow"), ("perilymph", "Perilymph"),
                                      ("endolymph", "Endolymph")]):
        yy = y0 + TOP + 1.0 + i * 2.5
        cv.line([x0 + w - 18.0, x0 + w - 15.8], [yy, yy], COMP_COL[comp], 0.9)
        cv.text(x0 + w - 15.2, yy, name, size=PT_SMALL, va="center")


def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Survey cell rule settings")
    d = pd.read_csv(R9 / "srob_d_survey_sweep.csv")
    order = ["14 d", "31 d", "31 d noise", "54 d"]
    ax = cv.axes(x0 + 10.0, y0 + TOP, w - 11.0, h - TOP - BOT)
    rng = np.random.default_rng(3)
    for i, an in enumerate(order):
        g = d[d.animal == an]
        col = PAIR_COL[an]
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(g)), g.cochlea_to_marrow, s=4, color=col, alpha=0.75, linewidths=0,
                   zorder=3)
        b_ = g[g.baseline.astype(bool)]
        ax.plot([i - 0.32, i + 0.32], [b_.cochlea_to_marrow.iloc[0]] * 2, color=TEXT, lw=0.9, zorder=4)
    ref_line(ax, 1.0, "h", "#8C8C8C", 0.5)
    logaxis(ax, 0.01, 2.0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)), [a.replace(" d noise", " d\nnoise") for a in order])
    clean(ax)
    ax.set_ylabel("Cochlea / marrow", fontsize=PT_SMALL, labelpad=0.8)


def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Label-corrected ratio")
    d = pd.read_csv(R9 / "srob_e_survey_verified.csv").set_index("animal")
    order = ["14 d", "31 d", "31 d noise", "54 d"]
    ax = cv.axes(x0 + 10.0, y0 + TOP, w - 11.0, h - TOP - BOT)
    lo_lim, hi_lim = 0.01, 10.0
    for i, an in enumerate(order):
        r = d.loc[an]
        col = PAIR_COL[an]
        ax.scatter([i - 0.15], [r.rule_ratio], s=11, facecolor="white", edgecolor=col, linewidths=0.7, zorder=4)
        lo, hi = max(r.ci_lo, lo_lim), min(r.ci_hi, hi_lim)
        ax.plot([i + 0.15, i + 0.15], [lo, hi], color=col, lw=0.9, zorder=3, solid_capstyle="butt")
        if r.ci_lo < lo_lim:
            ax.scatter([i + 0.15], [lo_lim * 1.15], marker="v", s=7, color=col, linewidths=0, zorder=4)
        ax.scatter([i + 0.15], [max(r.verified_ratio, lo_lim * 1.4)], s=11, color=col, linewidths=0, zorder=5)
    ref_line(ax, 1.0, "h", "#8C8C8C", 0.5)
    logaxis(ax, lo_lim, hi_lim)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)), [a.replace(" d noise", " d\nnoise") for a in order])
    clean(ax)
    ax.set_ylabel("Cochlea / marrow", fontsize=PT_SMALL, labelpad=0.8)
    ky = y0 + h - 1.6
    cv.mark(x0 + 11.0, ky, 10.0, "o", "white", TEXT, 0.7)
    cv.text(x0 + 12.2, ky, "rule", size=PT_SMALL, va="center")
    x2 = x0 + 12.2 + text_width("rule", PT_SMALL) + 2.4
    cv.mark(x2, ky, 10.0, "o", TEXT, TEXT)
    cv.text(x2 + 1.2, ky, "label-corrected, 95% CI", size=PT_SMALL, va="center")
    fits(x2 + 1.2 + text_width("label-corrected, 95% CI", PT_SMALL), x0 + w + 0.5, "e key")


def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Flow gate thresholds ±10%")
    d = pd.read_csv(R9 / "srob_f_flow.csv")
    lab = {"P1": "P1 cochlear immature neutrophils", "P2": "P2 cochlear classical monocytes",
           "P3": "P3 marrow recirculating B cells"}
    col = {"P1": C_COCH, "P2": C_COCH, "P3": C_MARROW}
    ax = cv.axes(x0 + 2.0, y0 + TOP, w - 3.0, h - TOP - BOT)
    rng = np.random.default_rng(5)
    for i, rd in enumerate(["P1", "P2", "P3"]):
        g = d[d.readout == rd]
        sig = g.verdict == "replicated"
        ax.scatter(g.ratio_zt17_zt5[sig], i + 0.12 + rng.uniform(-0.1, 0.1, int(sig.sum())), s=5, color=col[rd],
                   linewidths=0, alpha=0.8, zorder=3)
        ax.scatter(g.ratio_zt17_zt5[~sig], i + 0.12 + rng.uniform(-0.1, 0.1, int((~sig).sum())), s=7,
                   facecolor="white", edgecolor=col[rd], linewidths=0.5, zorder=3)
        b_ = g[g.perturbation == "published gates"].iloc[0]
        ax.plot([b_.ci_lo, b_.ci_hi], [i - 0.2, i - 0.2], color=TEXT, lw=0.8, zorder=4)
        ax.scatter([b_.ratio_zt17_zt5], [i - 0.2], s=10, color=TEXT, zorder=5, linewidths=0)
        ax.text(0.27, i - 0.45, lab[rd], fontsize=PT_SMALL, ha="left", va="center")
    ref_line(ax, 1.0, "v", "#8C8C8C", 0.5)
    ax.set_ylim(2.6, -0.75)
    ax.set_yticks([])
    logaxis(ax, 0.25, 4.0, "x")
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    clean(ax, left=False)
    ax.set_xlabel("ZT17 / ZT5", fontsize=PT_SMALL, labelpad=0.8)


TYPE_ROWS = [("FUT4", "cochlea (60x)", "Neutrophils, cochlea (60x)", C_COCH),
             ("FUT4", "cochlea", "Neutrophils, cochlea (20x)", C_COCH),
             ("CD45", "cochlea", "Leukocytes, cochlea", C_COCH), ("FUT4", "marrow", "Neutrophils, marrow", C_MARROW),
             ("CD14", "marrow", "Monocytes, marrow", C_MARROW), ("CD45", "marrow", "Leukocytes, marrow", C_MARROW),
             ("CD19", "marrow", "B cells, marrow (control)", C_MARROW)]


def panel_g(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Typing and GFP-call settings")
    d = pd.read_csv(R9 / "srob_g_typing_sweep.csv")
    rows = [r for r in TYPE_ROWS if r[1] in ("cochlea", "marrow") and not (r[0] == "FUT4" and r[1] == "cochlea")]
    lab_w = max(text_width(r[2], PT_SMALL) for r in rows) + 2.0
    ax = cv.axes(x0 + lab_w, y0 + TOP, w - lab_w - 1.0, h - TOP - BOT)
    rng = np.random.default_rng(6)
    for i, (mk, cp, lab, col) in enumerate(rows):
        g = d[(d.marker == mk) & (d.compartment == cp)]
        ax.scatter(g.gfp_pos_pct, i + rng.uniform(-0.22, 0.22, len(g)), s=2.5, color=col, alpha=0.45, linewidths=0,
                   zorder=3)
        b_ = g[g.baseline.astype(bool)]
        ax.plot([b_.gfp_pos_pct.iloc[0]] * 2, [i - 0.36, i + 0.36], color=TEXT, lw=0.9, zorder=4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    minor(ax, x=True, y=False)
    clean(ax)
    ax.set_xlabel("Partner-derived (% of typed cells)", fontsize=PT_SMALL, labelpad=0.8)
    assert d[d.marker.isin([r[0] for r in rows])].gfp_pos_pct.max() <= 20, "g axis"


def panel_h(cv, x0, y0, w, h):
    cv.letter(x0, y0, "h", "Blinded verification of the typing")
    d = pd.read_csv(R9 / "srob_h_typing_verified.csv")
    qs = [("typed: one marker+ cell", "o", None), ("GFP call agrees (confirmed cells)", "D", None)]
    lab_w = max(text_width(r[2], PT_SMALL) for r in TYPE_ROWS) + 2.0
    cnt_w = text_width("00/00  00/00", PT_SMALL) + 1.0
    ax = cv.axes(x0 + lab_w, y0 + TOP, w - lab_w - cnt_w - 1.0, h - TOP - BOT)
    for i, (mk, cp, lab, col) in enumerate(TYPE_ROWS):
        txt = []
        for j, (q, m, _) in enumerate(qs):
            r = d[(d.marker == mk) & (d.compartment == cp) & (d.question == q)]
            if not len(r):
                txt.append("–")
                continue
            r = r.iloc[0]
            yy = i + (-0.16 if j == 0 else 0.16)
            ax.plot([100 * r.ci_lo, 100 * r.ci_hi], [yy, yy], color=col, lw=0.8, zorder=3, solid_capstyle="butt")
            ax.scatter([100 * r.share], [yy], s=9 if m == "o" else 7, marker=m,
                       facecolor=col if j == 0 else "white", edgecolor=col, linewidths=0.6, zorder=4)
            txt.append(f"{int(r.n_agree)}/{int(r.n_decided)}")
        cv.text(x0 + w, y0 + TOP + (i + 0.5) * (h - TOP - BOT) / len(TYPE_ROWS), "  ".join(txt), size=PT_SMALL,
                ha="right", va="center")
    cv.text(x0 + w, y0 + TOP - 0.5, "cell  GFP", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    ax.set_ylim(len(TYPE_ROWS) - 0.5, -0.5)
    ax.set_yticks(range(len(TYPE_ROWS)), [r[2] for r in TYPE_ROWS])
    ax.tick_params(axis="y", length=0)
    ax.axhline(2.5, color=GRID, lw=0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    minor(ax, x=True, y=False)
    clean(ax)
    ax.set_xlabel("Agreement with the visual call (%)", fontsize=PT_SMALL, labelpad=0.8)
    ky = y0 + h - 1.4
    cv.mark(x0 + lab_w + 0.8, ky, 9.0, "o", TEXT, TEXT)
    cv.text(x0 + lab_w + 1.9, ky, "typed cell confirmed", size=PT_SMALL, va="center")
    x2 = x0 + lab_w + 1.9 + text_width("typed cell confirmed", PT_SMALL) + 2.4
    cv.mark(x2, ky, 7.0, "D", "white", TEXT, 0.6)
    cv.text(x2 + 1.1, ky, "GFP call agrees", size=PT_SMALL, va="center")


def main():
    cv = Canvas(180.0, 170.0)
    panel_a(cv, 0.0, 0.0, 62.0, 50.0)
    panel_b(cv, 64.0, 0.0, 40.0, 50.0)
    panel_c(cv, 106.0, 0.0, 74.0, 50.0)
    panel_d(cv, 0.0, 54.0, 52.0, 52.0)
    panel_e(cv, 54.0, 54.0, 52.0, 52.0)
    panel_f(cv, 110.0, 54.0, 70.0, 52.0)
    panel_g(cv, 0.0, 110.0, 84.0, 60.0)
    panel_h(cv, 88.0, 110.0, 92.0, 60.0)
    for p_ in cv.save(OUT, NAME):
        print(f"[fig] {p_.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
