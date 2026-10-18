"""Round 10 (D134; from the Fig. 5 of D68) — Fig. 6, one composite at Nature size (180 x 170 mm, 5-7 pt): cochlear
neutrophils keep the circulating (blood) timetable while marrow neutrophils oppose it. The fluids panel of the former
Fig. 5 (a) moved to Fig. 4d when Fig. 4 was split (D134); the other panels were relettered a-h.

a  Anchor: per gene, blood ZT13 - ZT5 vs our predicted ZT13 - ZT5, aged neutrophils of cochlea and marrow (grey: all
   shared genes; colour: P < 0.05 in both); rho for both gene sets.
b  rho per neutrophil state and tissue (all genes: circles, filled if gene-permutation P < 0.05; genes P < 0.05 in both:
   diamonds) and the exact time-relabelling P of each single correlation (n = 6; G53).
c  Specificity: every cell type's rho ranked within its tissue in the expected direction; key states highlighted; exact
   rank-sum P.
d  The blood-timer score of pooled neutrophils across the day (six samples, cosinor fit), marrow and cochlea.
e  rho on all, non-enriched and neutrophil-enriched genes (aged / activated / mature states).
f  Independent long-bone marrow (GSE198770): rho of control vs neutrophil-depleted marrow rhythms.
g  External lines: human blood neutrophils 14:00 vs 08:00 (our cochlear and marrow fits) and bulk cochlea (GSE88954).
h  Independent flow cohort (D89; 932; pre-specified secondaries S1-S2): representative contour plots at ZT5 and ZT17
   (T5: Ly6G vs CXCR4 of Ly6G+ SSC-hi cells; mouse closest to its group median; % CXCR4-hi of the plotted parent) and the
   per-mouse % CXCR4-hi (S2) and % CD62L-hi (S1) of Ly6G+ cells in cochlea and marrow (bar, geometric mean; P, the
   mixed-model ZT17 vs ZT5 contrast within each tissue). The exploratory transcript ageing cycle is Extended Data
   Fig. 9f; the test-design sketch is Extended Data Fig. 9e.

Inputs: results/r9_figures/f5*.csv (900 build_f6; f5i_flow_markers and flow_* from 932). The table names keep their v3
prefix (f5b-f5i). No statistics are computed here. Output: figures/figures/fig6_timer_split_v4.*
Run: python scripts/913_fig6_timer_split.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge
from matplotlib.ticker import MaxNLocator
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.flowplot import axis_title, flow_contour, flow_dots, load_tables, p_label  # noqa: E402
from lib.figstyle import (AXIS, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, fmt_p, ld_bar,  # noqa: E402
                          text_width, zt_axis)

from lib.figstyle import P_SYM, R_SYM  # noqa: E402,F401  (italic P / r, D99)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
PANELS = ROOT / "figures" / "panels"
NAME = "fig6_timer_split_v5"
TIS = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
BLOOD = "#7A0403"
GREY, RULE = "#777777", "#E6E6E6"
STATE_ABBR = {"Neutrophil (Band)": "Band", "Neutrophil (Mature)": "Mature", "Neutrophil (Aged)": "Aged",
              "Neutrophil (Activated)": "Activated", "Granulocyte-Monocyte Progenitor": "GMP"}


def spread(ys, gap, lo, hi):
    ys = np.array(ys, float)
    for _ in range(300):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= d
                ys[i] += d
                moved = True
        ys = np.clip(ys, lo, hi)
        if not moved:
            break
    return ys


def mm_axes(cv, x0, y0, w, h):
    """An invisible axes whose data coordinates are millimetres from the panel's top-left corner (y downwards)."""
    ax = cv.axes(x0, y0, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    blank(ax)
    return ax


# ------------------------------------------------------------------------------------------------ a anchor
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Gene-level concordance")
    gb = pd.read_csv(R9 / "f5b_timer_genes.csv")
    gb = gb[gb.cell_type == "Neutrophil (Aged)"]
    fh, top = 42.0, y0 + 9.6
    fw = (w - 9.4 - 13.6 - 3.0) / 2                       # two facets fill the panel width
    xs = gb.blood_zt13_minus_zt5_log2fc
    xl = (np.floor(xs.min() * 2) / 2, np.ceil(xs.max() * 2) / 2)
    for j, t in enumerate(["cochlea", "bm"]):
        d = gb[gb.tissue == t]
        bx = x0 + 9.4 + j * (fw + 13.6)
        ax = cv.axes(bx, top, fw, fh)
        yv = d.predicted_zt13_minus_zt5
        yl = (np.floor(yv.min() * 4) / 4 - 0.05, np.ceil(yv.max() * 4) / 4 + 0.05)
        assert d.blood_zt13_minus_zt5_log2fc.between(*xl).all() and yv.between(*yl).all()
        ax.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
        o = d[~d.both_p05]
        b = d[d.both_p05]
        ax.scatter(o.blood_zt13_minus_zt5_log2fc, o.predicted_zt13_minus_zt5, s=1.2, color="#9A9A9A", alpha=0.35,
                   linewidths=0, rasterized=True, zorder=1)
        ax.scatter(b.blood_zt13_minus_zt5_log2fc, b.predicted_zt13_minus_zt5, s=5, color=TIS[t][1], alpha=0.9,
                   linewidths=0, zorder=3)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
        clean(ax)
        ax.set_xlabel("Blood neutrophils, ZT13 − ZT5 (log₂)", fontsize=PT_SMALL, labelpad=0.8)
        ax.set_ylabel("Predicted ZT13 − ZT5 (log₂)", fontsize=PT_SMALL, labelpad=0.8)
        cv.mark(bx + 0.9, top - 2.1, 10.0, "o", TIS[t][1], TIS[t][1])
        cv.text(bx + 2.2, top - 1.0, f"{TIS[t][0]}, aged neutrophils", size=PT, va="bottom", fontweight="bold",
                color=TEXT)
        if bool(gb.g02_flow_supported.iloc[0]):                  # D91: split supported by the flow cohort (h; D90)
            cv.mark(bx + 2.2 + text_width(f"{TIS[t][0]}, aged neutrophils", PT, "bold") + 1.6, top - 2.0, 5.0, "D",
                    TEXT)
        r0 = d.iloc[0]
        ax.text(0.02 if j == 0 else 0.98, 0.98, f"ρ = {r0.spearman_all:.2f}, all\nρ = {r0.spearman_both_p05:.2f}, coloured",
                transform=ax.transAxes,
                fontsize=PT_SMALL, va="top", ha="left" if j == 0 else "right",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        # label the most concordant genes (largest product with the expected sign), spread vertically
        sg = 1 if t == "cochlea" else -1
        lab = b.assign(s=sg * b.blood_zt13_minus_zt5_log2fc * b.predicted_zt13_minus_zt5).nlargest(6, "s")

        def to_mm(px, py):
            return (bx + (px - xl[0]) / (xl[1] - xl[0]) * fw, top + (yl[1] - py) / (yl[1] - yl[0]) * fh)

        pm = np.array([to_mm(a_, b_) for a_, b_ in zip(lab.blood_zt13_minus_zt5_log2fc, lab.predicted_zt13_minus_zt5)])
        right = pm[:, 0] < bx + fw * 0.62
        for side in (True, False):
            sel = np.where(right == side)[0]
            if not len(sel):
                continue
            srt = sel[np.argsort(pm[sel, 1])]
            ys = spread(pm[srt, 1], 2.2, top + 1.0, top + fh - 1.0)
            for k, yl_ in zip(srt, ys):
                px, py = pm[k]
                g = lab.gene.iloc[k]
                if side:
                    cv.line([px, px + 1.4, px + 2.2], [py, yl_, yl_], GREY, 0.3)
                    cv.text(px + 2.4, yl_, g, size=PT_SMALL, va="center", style="italic")
                else:
                    cv.line([px, px - 1.4, px - 2.2], [py, yl_, yl_], GREY, 0.3)
                    cv.text(px - 2.4, yl_, g, size=PT_SMALL, va="center", ha="right", style="italic")
    x = cv.key(x0 + 9.4, y0 + h - 1.6, [([{"s": 2.0, "fc": "#9A9A9A"}], "all genes"),
                                        ([{"s": 5, "fc": TIS["cochlea"][1]}, {"s": 5, "fc": TIS["bm"][1]}],
                                         f"{P_SYM} < 0.05 in both"),
                                        ([{"kind": "mark", "s": 5.0, "marker": "D", "fc": TEXT}],
                                         "supported by the flow cohort (h)")])
    fits(x, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b per state
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Every neutrophil state")
    cc = pd.read_csv(R9 / "f5c_timer_concordance.csv")
    states = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)",
              "Granulocyte-Monocyte Progenitor"]
    lab_w, right_w = 12.0, 12.6
    top, ah = y0 + 9.6, h - 19.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - right_w, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    rh = ah / len(states)
    cv.text(x0 + w - 0.3, top - 0.6, f"exact {P_SYM}", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    for i, ct in enumerate(states):
        if i:
            ax.axhline(i - 0.5, color=RULE, linewidth=0.4, zorder=0)
        ps = []
        for t, off in [("cochlea", -0.18), ("bm", 0.18)]:
            r = cc[(cc.tissue == t) & (cc.cell_type == ct)].iloc[0]
            col = TIS[t][1]
            ax.scatter([r.spearman_both_p05], [i + off], s=12, marker="D", facecolor="white", edgecolor=col,
                       linewidths=0.6, zorder=3)
            ax.scatter([r.spearman_all], [i + off], s=11, facecolor=col if r.p_perm_all < 0.05 else "white",
                       edgecolor=col, linewidths=0.6, zorder=4)
            ax.plot([r.spearman_all, r.spearman_both_p05], [i + off] * 2, color=col, linewidth=0.4, zorder=2)
            ps.append(r.p_exact_two_sided)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) * rh, STATE_ABBR[ct], size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if ct in ("Neutrophil (Aged)", "Neutrophil (Activated)") else "normal")
        cv.text(x0 + w - 0.3, top + (i + 0.5) * rh - 1.0, f"{fmt_p(ps[0])}", size=PT_SMALL, ha="right", va="center",
                color=TEXT)
        cv.text(x0 + w - 0.3, top + (i + 0.5) * rh + 1.0, f"{fmt_p(ps[1])}", size=PT_SMALL, ha="right", va="center",
                color=TEXT)
    lim = 0.75
    assert cc[["spearman_all", "spearman_both_p05"]].abs().max().max() <= lim
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(states) - 0.5, -0.5)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6], ["−0.6", "−0.3", "0", "0.3", "0.6"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("ρ with the blood programme", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 2.2, [([{"s": 11, "fc": GREY}], "all genes"),
                                        ([{"s": 12, "marker": "D", "fc": "white", "ec": GREY}], f"{P_SYM} < 0.05 in both")])
    fits(x, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c specificity
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Specificity within each tissue")
    sp = pd.read_csv(R9 / "f5d_specificity_rank.csv")
    lab_w = 10.6
    sh = 14.6
    for k, (t, top) in enumerate([("cochlea", y0 + 8.4), ("bm", y0 + 29.4)]):
        d = sp[sp.tissue == t].sort_values("rank")
        n = len(d)
        ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, sh)
        ax.axhline(0, color="#999999", linewidth=0.45, zorder=1)
        col = TIS[t][1]
        for _, r in d.iterrows():
            c = col if r.key_state else "#B3B3B3"
            ax.plot([r["rank"]] * 2, [0, r.spearman], color=c, linewidth=0.8 if r.key_state else 0.6, zorder=2)
            ax.scatter([r["rank"]], [r.spearman], s=8 if r.key_state else 4, color=c, linewidths=0, zorder=3)
        ax.set_xlim(0.2, 33.8)
        lo, hi = (-0.2, 0.42) if t == "cochlea" else (-0.42, 0.2)
        assert d.spearman.between(lo, hi).all()
        ax.set_ylim(lo, hi)
        ax.set_yticks([-0.2, 0, 0.2, 0.4] if t == "cochlea" else [-0.4, -0.2, 0, 0.2])
        ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
        ax.set_xticks([1, 10, 20, 30] if n > 30 else [1, 10, 20])
        clean(ax)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("ρ", fontsize=PT_SMALL, labelpad=0.6, rotation=0, va="center")
        cv.text(x0 + 1.8, top + sh / 2, TIS[t][0], size=PT_SMALL, ha="center", va="center", rotation=90, color=TEXT,
                fontweight="bold")
        ks = d[d.key_state].sort_values("spearman", ascending=False)   # stacked labels ordered by rho: no crossings
        pw_ = w - lab_w - 1.0
        for m_, (_, r) in enumerate(ks.iterrows()):
            xs_ = x0 + lab_w + (r["rank"] - 0.2) / 33.6 * pw_
            ys_ = top + (hi - r.spearman) / (hi - lo) * sh
            if t == "cochlea":                                          # stacked right of the three top stems
                xl_, yl_ = x0 + lab_w + (ks["rank"].max() + 2.4) / 33.6 * pw_, top + 1.4 + m_ * 2.3
            else:                                                       # stacked in the empty right part of the strip
                xl_, yl_ = x0 + lab_w + 23.2 / 33.6 * pw_, top + 5.6 + m_ * 2.3
            cv.line([xs_ + 0.4, xl_ - 0.4], [ys_, yl_], col, 0.3)
            cv.text(xl_, yl_, STATE_ABBR[r.cell_type], size=PT_SMALL, va="center", color=TEXT, fontweight="bold")
        rk = ", ".join(str(int(v)) for v in sorted(d[d.key_state]["rank"]))
        p = d.rank_sum_exact_p.iloc[0]
        cv.text(x0 + w - 1.0, top + 0.6, f"{P_SYM} = {fmt_p(p)}" if p >= 0.001 else f"{P_SYM} = {p:#.2g}", size=PT_SMALL,
                ha="right", va="top", color=TEXT)
    cv.text(x0 + lab_w + (w - lab_w - 1.0) / 2, y0 + 29.4 + sh + 3.0, "Rank", size=PT_SMALL, ha="center", va="top")


# ------------------------------------------------------------------------------------------------ d timer score
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Timer score across the day")
    pts = pd.read_csv(R9 / "f5e_timer_score.csv")
    fit = pd.read_csv(R9 / "f5e_timer_score_fit.csv")
    lab_w, mh = 9.6, 13.6
    pw = w - lab_w - 2.2
    for k, (t, top) in enumerate([("bm", y0 + 8.6), ("cochlea", y0 + 28.4)]):
        p_ = pts[pts.tissue == t].sort_values("zt")
        f_ = fit[fit.tissue == t].sort_values("zt")
        col = TIS[t][1]
        ax = cv.axes(x0 + lab_w, top, pw, mh)
        ax.plot(f_.zt, f_.fit_score, color=col, linewidth=1.0, zorder=2)
        ax.scatter(p_.zt, p_.median_score, s=10, color=col, linewidths=0, zorder=3)
        lo = min(p_.median_score.min(), f_.fit_score.min())
        hi = max(p_.median_score.max(), f_.fit_score.max())
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad * 2.6)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="y", length=1.2, pad=0.8, labelsize=PT_SMALL)
        clean(ax)
        zt_axis(ax)
        ld_bar(cv, x0 + lab_w, top + mh, pw, 0.8)
        r0 = f_.iloc[0]
        ax.scatter([r0.peak_zt], [hi + pad * 2.1], marker="v", s=10, color=col, linewidths=0, zorder=4)
        ax.text(r0.peak_zt + (0.8 if r0.peak_zt < 12 else -0.8), hi + pad * 2.1, f"ZT{r0.peak_zt:.1f}, {P_SYM} {fmt_p(r0.p_perm)}",
                fontsize=PT_SMALL, va="center", ha="left" if r0.peak_zt < 12 else "right", color=TEXT)
        cv.text(x0 + 2.0, top + mh / 2, TIS[t][0], size=PT_SMALL, ha="center", va="center", rotation=90, color=TEXT,
                fontweight="bold")
    cv.text(x0 + lab_w + pw / 2, y0 + 28.4 + mh + 4.6, "ZT (h)", size=PT_SMALL, ha="center", va="top")


# ------------------------------------------------------------------------------------------------ e identity genes
def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Not identity genes")
    fb = pd.read_csv(R9 / "f5f_enriched_genes.csv")
    states = ["Neutrophil (Aged)", "Neutrophil (Activated)", "Neutrophil (Mature)"]
    sets = [("all shared genes", "o", 11), ("non-enriched genes", "s", 10), ("neutrophil-enriched genes", "^", 12)]
    lab_w, top, ah = 13.4, y0 + 8.4, h - 23.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    rows = [(t, s) for t in ["cochlea", "bm"] for s in states]
    for i, (t, s) in enumerate(rows):
        col = TIS[t][1]
        d = fb[(fb.tissue == t) & (fb.cell_type == s)].set_index("gene_set")
        for gs, mk, sz in sets:
            ax.scatter([d.loc[gs, "spearman"]], [i], s=sz, marker=mk, facecolor=col if gs == "non-enriched genes" else
                       ("white" if gs.startswith("neutrophil") else col), edgecolor=col, linewidths=0.6, zorder=3)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) * ah / len(rows), f"{STATE_ABBR[s]}", size=PT_SMALL, ha="right",
                va="center")
    for g, t in enumerate(["cochlea", "bm"]):
        ya, yb = top + g * 3 * ah / 6 + 0.4, top + (g + 1) * 3 * ah / 6 - 0.4
        cv.line([x0 + 2.4, x0 + 2.4], [ya, yb], TIS[t][1], 0.8)
        cv.text(x0 + 1.4, (ya + yb) / 2, TIS[t][0], size=PT_SMALL, ha="center", va="center", rotation=90,
                color=TEXT, fontweight="bold")
    ax.axhline(2.5, color="#BBBBBB", linewidth=0.4)
    lim = 0.45
    assert fb.spearman.abs().max() <= lim
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([-0.3, 0, 0.3], ["−0.3", "0", "0.3"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("ρ with the blood programme", fontsize=PT_SMALL, labelpad=0.8)
    ky = y0 + h - 8.4
    for i, (gs, mk, sz) in enumerate(sets):
        x = cv.key(x0 + 1.0, ky + i * 2.9, [([{"s": sz, "marker": mk, "fc": GREY if gs != "neutrophil-enriched genes"
                                               else "white", "ec": GREY}], gs.replace(" genes", "").replace("shared", "genes"))])
        fits(x, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ f depletion
def panel_g(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Lost without neutrophils")
    gd = pd.read_csv(R9 / "f5g_depletion.csv")
    sets = [("all genes", "All genes"), ("rhythmic genes (p < 0.05)", "Rhythmic genes")]
    lab_w, top, ah = max(12.6, text_width("Rhythmic genes", PT_SMALL) + 1.8), y0 + 8.4, 30.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    col = TIS["bm"][1]
    for i, (gs, lab) in enumerate(sets):
        c_ = gd[(gd.gene_set == gs) & (gd.condition == "control")].iloc[0]
        d_ = gd[(gd.gene_set == gs) & (gd.condition == "neutrophil_depleted")].iloc[0]
        ax.annotate("", xy=(d_.spearman, i), xytext=(c_.spearman, i), arrowprops=dict(
            arrowstyle="-|>,head_length=0.3,head_width=0.18", color=col, linewidth=0.7, shrinkA=2.2, shrinkB=2.0))
        ax.scatter([c_.spearman], [i], s=14, color=col, linewidths=0, zorder=3)
        ax.scatter([d_.spearman], [i], s=14, facecolor="white", edgecolor=col, linewidths=0.6, zorder=3)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) * ah / 2, lab, size=PT_SMALL, ha="right", va="center")
    ax.set_xlim(-0.25, 0.1)
    assert gd.spearman.between(-0.25, 0.1).all()
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([-0.2, -0.1, 0, 0.1], ["−0.2", "−0.1", "0", "0.1"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("ρ with the blood programme", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 2.2, [([{"s": 14, "fc": col}], "control"),
                                        ([{"s": 14, "fc": "white", "ec": col}], "neutrophil-depleted")])
    fits(x, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ g external lines
def panel_h(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Human blood and bulk cochlea")
    ex = pd.read_csv(R9 / "f5h_external_lines.csv")
    hu = ex[ex.source.str.startswith("human")]
    bk = ex[ex.source.str.startswith("bulk") & (ex.unit == "neutrophil_markers")]
    rows = [("human", "Neutrophil (Band)", "Band"), ("human", "Neutrophil (Mature)", "Mature"),
            ("human", "Neutrophil (Aged)", "Aged"), ("human", "Neutrophil (Activated)", "Activated"),
            ("human", "all neutrophil states", "All states"), ("bulk", "sham", "Sham"), ("bulk", "adx", "Adrenalect.")]
    lab_w, top = 14.6, y0 + 9.8
    span = len(rows) + 0.7                                      # data rows plus the gap before the bulk rows
    ah = 31.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    for i, (src, u, lab) in enumerate(rows):
        yy = i + (0.7 if src == "bulk" else 0.0)
        if src == "human":
            for t, off in [("cochlea", -0.17), ("bm", 0.17)]:
                r = hu[(hu.tissue == t) & (hu.unit == u)].iloc[0]
                filled = (r.p < 0.05) if t == "cochlea" else True      # one-sided p stored for the positive direction
                ax.plot([0, r.value], [yy + off] * 2, color=TIS[t][1], linewidth=0.6, zorder=2)
                ax.scatter([r.value], [yy + off], s=10, facecolor=TIS[t][1] if filled else "white", edgecolor=TIS[t][1],
                           linewidths=0.6, zorder=3)
        else:
            r = bk[bk.source.str.contains(f"({u})", regex=False)].iloc[0]
            ax.plot([0, r.value], [yy, yy], color=TIS["cochlea"][1], linewidth=0.6, zorder=2)
            ax.scatter([r.value], [yy], s=10, marker="s", facecolor=TIS["cochlea"][1] if r.p < 0.05 else "white",
                       edgecolor=TIS["cochlea"][1], linewidths=0.6, zorder=3)
            ax.text(r.value + 0.025, yy, f"{P_SYM} {fmt_p(r.p)}", fontsize=PT_SMALL, va="center", color=TEXT2)
        cv.text(x0 + lab_w - 1.0, top + (yy + 0.5) / span * ah, lab, size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if lab == "All states" else "normal")
    ax.axhline(4.85, color="#BBBBBB", linewidth=0.4)
    lim = (-0.2, 0.62)
    assert ex[ex.source.str.startswith(("human", "bulk"))].value.between(*lim).all()
    ax.set_xlim(*lim)
    ax.set_ylim(span - 0.5, -0.5)
    ax.set_xticks([-0.1, 0, 0.2, 0.4], ["−0.1", "0", "0.2", "0.4"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("Correlation", fontsize=PT_SMALL, labelpad=0.8)
    cv.text(x0 + lab_w - 1.0, top - 1.6, "Human blood", size=PT_SMALL, ha="right", va="center", color=TEXT2,
            style="italic")
    cv.text(x0 + lab_w - 1.0, top + (5.25 + 0.5) / span * ah - 0.6, "Bulk cochlea", size=PT_SMALL, ha="right",
            va="center", color=TEXT2, style="italic")
    x = cv.key(x0 + 1.0, y0 + h - 2.2, [([{"s": 10, "fc": TIS["cochlea"][1]}], "cochlear fits"),
                                        ([{"s": 10, "fc": TIS["bm"][1]}], "marrow fits"),
                                        ([{"s": 10, "marker": "s", "fc": TIS["cochlea"][1]}], "bulk")])
    fits(x, x0 + w, "g key")


# ------------------------------------------------------------------------------------------------ h ageing cycle
def panel_i(cv, x0, y0, w, h):
    """h — independent flow cohort, neutrophil markers (D89; D92 restyle; 932): CXCR4 contour plots with outliers of
    Ly6G+ SSC-hi cells, cochlea and marrow x ZT5 and ZT17 (one marker group: scale on the bottom-left plot only), and
    the per-mouse CXCR4-hi (S2) and CD62L-hi (S1) shares with the mixed-model p within each tissue."""
    cv.letter(x0, y0, "h", "Neutrophil markers (flow)")
    tabs = load_tables(R9)
    mk = pd.read_csv(R9 / "f5i_flow_markers.csv")
    co, bm = TIS["cochlea"][1], TIS["bm"][1]
    gp, tg, ya = 0.7, 1.8, 5.4                             # gap in a pair, gap between tissues, y-axis area
    s_ = min(12.6, (w - ya - 2 * gp - tg - 0.5) / 4)        # plot side
    top = y0 + 10.4
    for i, (tag, tname, col) in enumerate([("co", "Cochlea", co), ("bm", "Marrow", bm)]):
        px = x0 + ya + i * (2 * s_ + gp + tg)
        cv.text(px + s_ + gp / 2, top - 3.3, tname, size=PT_SMALL, ha="center", va="bottom", fontweight="bold",
                color=TEXT)
        cv.mark(px + s_ + gp / 2 - text_width(tname, PT_SMALL, "bold") / 2 - 1.2, top - 4.3, 9.0, "o", col, col)
        for k, zt in enumerate(("ZT5", "ZT17")):
            xx = px + k * (s_ + gp)
            cv.text(xx + s_ / 2, top - 0.5, zt, size=PT_SMALL, ha="center", va="bottom")
            first = i == 0 and k == 0                         # bottom-left plot of the group carries the scale
            flow_contour(cv, xx, top, s_, f"main_T5_{tag}_{zt}", tabs, numbers=(first, first), titles=(False, first))
    row_w = 4 * s_ + 2 * gp + tg
    fits(x0 + ya + row_w, x0 + w + 1e-6, "h contour row")
    pr = tabs["plots"].loc["main_T5_co_ZT5"]
    cv.text(x0 + ya + row_w / 2, top + s_ + 2.5, axis_title(pr.x_label, pr.x_axis), size=PT_SMALL, ha="center",
            va="top")
    # S2 and S1: % positive of Ly6G+ per mouse, cochlea | marrow x ZT5 | ZT17
    qt = top + s_ + 9.2
    qh = y0 + h - qt - 5.6
    qx0, gq = x0 + 6.0, 6.6
    qw = (w - 6.0 - gq) / 2
    for k, (key, name) in enumerate([("S2_pct", "CXCR4-hi"), ("S1_pct", "CD62L-hi")]):
        d = mk[(mk.readout == key) & mk.included.astype(bool)]
        grp = [d[(d.tissue == t) & (d.zt == z)].value for t in ("cochlea", "temporal_bone_marrow") for z in ("ZT5", "ZT17")]
        xq = qx0 + k * (qw + gq)
        r0 = d.iloc[0]
        br = [(2 * j, 2 * j + 1, p_label(float(r0[f"{part}_p"]), fmt_p) if float(r0[f"{part}_p"]) < 0.05 else "n.s.")
              for j, part in enumerate(("cochlea", "marrow"))]
        flow_dots(cv, xq, qt, qw, qh, grp, [co, co, bm, bm], ["5", "17", "5", "17"],
                  ylab="% of Ly6G⁺" if k == 0 else None, seed=10 + k, ymax=100, brackets=br)
        cv.text(xq + qw / 2, qt - 1.0, name, size=PT_SMALL, ha="center", va="bottom", fontweight="bold")
        for j, (t, col) in enumerate([("Cochlea", co), ("Marrow", bm)]):
            cv.text(xq + qw * (0.25 + 0.5 * j), qt + qh + 2.6, t, size=PT_SMALL, ha="center", va="top", color=TEXT)
        fits(text_width("Cochlea", PT_SMALL) + text_width("Marrow", PT_SMALL), qw + 1.0, f"h tissue labels {key}")


# ------------------------------------------------------------------------------------------------ main
def main():
    cv = Canvas()
    panel_b(cv, 0.0, 0.0, 118.0, 60.0)                  # a anchor, b every state (function names keep the v3 letters)
    panel_c(cv, 120.0, 0.0, 60.0, 60.0)
    panel_d(cv, 0.0, 62.0, 64.0, 52.0)                  # c specificity, d timer score, e identity genes
    panel_e(cv, 66.0, 62.0, 62.0, 52.0)
    panel_f(cv, 130.0, 62.0, 50.0, 52.0)
    panel_g(cv, 0.0, 116.0, 40.0, 54.0)                 # f depletion, g external lines, h flow markers
    panel_h(cv, 42.0, 116.0, 62.0, 54.0)
    panel_i(cv, 106.0, 116.0, 74.0, 54.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
