"""Round 9 (D68 / D69) — Extended Data Fig. 9 (180 x 214 mm since round 10; a-g as in v2): myeloid hypotheses and timer
robustness (supports Fig. 6).

a  The nine pre-specified myeloid hypotheses (D45 / D48): combined exact p (Fisher over tissues), BH q, method lines per
   tissue (filled, agrees), pre-specified direction and decision.
b  Neutrophil-state timer concordance (Spearman with the blood night - day programme, GSE86619): raw and within-sample
   centred, all genes and without stress / immediate-early genes; right, correlation of the raw prediction with the
   change implied by the tissue's sample-level pattern alone.
c  M3: per-sample median neutrophil-branch pseudotime of GMP + band + mature cells (bar, interquartile range).
d  M1: per-sample median maturation score (Xie et al.) per neutrophil group (marrow above cochlea).
e  Timer test design (D82; from Fig. 5 v1 a).
f  EXPLORATORY ageing cycle (D89; from Fig. 5 v2 i, whose place the flow test of CXCR4 / CD62L took): peak phase of
   the blood-timer score, Sell (CD62L) and the Cxcr4 / Cxcr2 ratio in each tissue (solid, permutation p < 0.05).
g  Marrow-content sensitivity (D89; from Fig. 4 v2 f, whose place the flow cohort took): marrow content of the cochlear
   samples (marrow-resident precursors / structural cells; D17) and the phase and p of the cochlear shares before
   (primary) and after expressing them against the marrow-resident precursors.
h  Round 10 R2 (677; D116): the neutrophil-timer split as one statistic, rho(cochlea) - rho(marrow), against all 3,600
   joint relabelings of the two tissues' samples, for aged and pooled neutrophils (line, observed; P, one-sided exact).

Inputs: results/r9_figures/ed9_*.csv, palettes. Output: figures/figures/ed9_myeloid_v1.*
Run: python scripts/928_ed9_myeloid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402
from lib.figstyle import (AXIS, GRID, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, fmt_p, ld_bar,  # noqa: E402
                          text_width, zt_axis)

from lib.figstyle import P_SYM, Q_SYM, R_SYM  # noqa: E402,F401  (italic P / q / r, D99, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed9_myeloid_v3"                     # v3 (round 10, D112 / D116): panel h added below a-g; a-g unchanged
W_ED, H_ED = 180.0, 214.0
SPLIT_UNITS = [("Neutrophil (Aged)", "Aged neutrophils"), ("pooled neutrophils", "Pooled neutrophils")]
TIS = {"cochlea": ("Cochlea", "#0072B2"), "bm": ("Marrow", "#D55E00")}
DECISION = {"supported": "#1A1A1A", "partial": "#9A9A9A", "not supported": "white"}
STATES = ["pooled neutrophils", "Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Neutrophil (Mature)",
          "Neutrophil (Aged)", "Neutrophil (Activated)"]
STATE_LAB = {"pooled neutrophils": "pooled", "Granulocyte-Monocyte Progenitor": "GMP", "Neutrophil (Band)": "Neu_Band",
             "Neutrophil (Mature)": "Neu_Mat", "Neutrophil (Aged)": "Neu_Aged", "Neutrophil (Activated)": "Neu_Act"}
METHOD_TIS = {"bm": "M", "cochlea": "C", "paired": "P"}
GREY = "#777777"
AGEING = {"blood timer": ("Timer score", "#000000"), "sell": ("Sell (CD62L)", "#009E73"),
          "cxcr4_cxcr2": ("Cxcr4 / Cxcr2", "#CC79A7")}
MC_SHARES = ["Neutrophil (Band)", "Monocyte (Classical)", "Granulocyte-Monocyte Progenitor", "Neutrophil (Mature)",
             "Neutrophil (Aged)", "B Cell (Naïve)"]                 # cochlear shares of Fig. 5b
SHORT = {"M1": "M1 maturation / ageing timing", "M2": "M2 circulating-like state", "M3": "M3 granulopoiesis timing",
         "M4": "M4 functional readiness", "M5": "M5 monocyte → macrophage", "M6": "M6 retention / release",
         "M7": "M7 myeloid clocks", "M8": "M8 macrophage identity", "M9": "M9 noise readiness (noise cohort)"}


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def phase_axis(ax):
    zt_axis(ax, ticks=(0, 6, 12, 18, 24))
    ax.spines["bottom"].set_visible(False)
    tick_style(ax)
    ax.tick_params(axis="x", length=0, pad=2.6)


# ------------------------------------------------------------------------------------------------ a scorecard
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Pre-specified myeloid hypotheses")
    sc = pd.read_csv(R9 / "ed9_m_scorecard.csv")
    mm = pd.read_csv(R9 / "ed9_m_methods.csv")
    sc["label"] = sc.hypothesis.map(SHORT)
    assert sc.label.notna().all()
    lab_w = max(text_width(n, PT_SMALL) for n in sc.label) + 1.4
    top = y0 + 8.0
    rp = (h - (top - y0) - 7.0) / len(sc)
    cw_p, cw_q, cw_m, cw_dir, cw_d = 12.0, 10.0, 28.0, 5.0, 16.0
    xp = x0 + lab_w
    xq = xp + cw_p + 4.4
    xm = xq + cw_q + 2.0
    xr = xm + cw_m
    xd = xr + cw_dir
    fits(xd + cw_d, x0 + w, "a columns")
    ny = len(sc) * rp
    for xx, cw, col, thr in ((xp, cw_p, "combined_p", 0.05), (xq, cw_q, "q_BH", 0.1)):
        ax = cv.axes(xx, top, cw, ny)
        ax.scatter(-np.log10(sc[col].clip(lower=0.01)), np.arange(len(sc)), s=8, c=TEXT, linewidths=0, zorder=3)
        ax.axvline(-np.log10(thr), color="#E69F00", lw=0.6, zorder=1)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 1, 2], ["1", "0.1", "0.01"])
        ax.set_ylim(len(sc) - 0.5, -0.5)
        for i in range(len(sc)):
            ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        ax.set_yticks([])
        clean(ax, left=False)
        tick_style(ax)
    for xx, t in ((xp, f"Combined {P_SYM}"), (xq, Q_SYM), (xm, "Methods"), (xr, "Dir."), (xd, "Decision")):
        cv.text(xx, top - 0.8, t, size=PT_SMALL, va="bottom", color=TEXT2)
    for i, r in sc.iterrows():
        yy = top + (i + 0.5) * rp
        cv.text(x0 + lab_w - 0.8, yy, r.label, size=PT_SMALL, ha="right", va="center")
        xx = xm + 0.4
        for tis in ("bm", "cochlea", "paired"):
            lines = mm[(mm.hypothesis == r.hypothesis) & (mm.tissue == tis)]
            if not len(lines):
                continue
            cv.text(xx, yy, METHOD_TIS[tis], size=PT_SMALL, va="center", color=TEXT2)
            xx += text_width(METHOD_TIS[tis], PT_SMALL) + 0.8
            for ag in lines.agrees.astype(bool):
                cv.mark(xx + 0.9, yy, 7, fc=TEXT if ag else "white", ec=TEXT, lw=0.45)
                xx += 2.4
            xx += 1.4
        fits(xx, xr, f"a methods {r.hypothesis}")
        ok = bool(r.direction_ok)
        cv.mark(xr + 1.6, yy, 9, marker="o", fc=TEXT if ok else "white", ec=TEXT, lw=0.5)
        cv.box(xd, yy - 1.0, 2.0, 2.0, DECISION[r.decision], TEXT, 0.4)
        cv.text(xd + 2.8, yy, r.decision, size=PT_SMALL, va="center")
    items = [([{"s": 7, "fc": TEXT}], "agrees"), ([{"s": 7, "fc": "white", "ec": TEXT, "lw": 0.45}], "does not"),
             ([{"kind": "vline", "colour": "#E69F00", "lw": 0.6, "h": 2.0}], f"{P_SYM} 0.05 / {Q_SYM} 0.1")]
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, items, gap=1.8)
    fits(xx, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b timer robustness
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Timer split: centring and gene subsets")
    d = pd.read_csv(R9 / "ed9_timer_robustness.csv")
    lab_w = max(text_width(v, PT_SMALL) for v in STATE_LAB.values()) + 1.4
    top = y0 + 8.0
    pat_w = 13.0
    aw = w - lab_w - pat_w - 3.0
    gap = 1.4
    rows = [(t, s_) for t in TIS for s_ in STATES]
    rp = (h - (top - y0) - gap - 11.0) / len(rows)
    yy, y = {}, 0.0
    for i, (t, s_) in enumerate(rows):
        if i and t != rows[i - 1][0]:
            y += gap / rp
        yy[(t, s_)] = y
        y += 1
    ax = cv.axes(x0 + lab_w, top, aw, y * rp)
    axp = cv.axes(x0 + w - pat_w, top, pat_w - 1.0, y * rp)
    ax.axvline(0, color=TEXT2, lw=0.35, zorder=1)
    marks = {("raw (as G02)", "all shared genes"): ("o", True, -0.27), ("raw (as G02)", "without stress / IEG genes"):
             ("o", False, -0.09), ("within-sample centred", "all shared genes"): ("D", True, 0.09),
             ("within-sample centred", "without stress / IEG genes"): ("D", False, 0.27)}
    for (t, s_), yv in yy.items():
        col = TIS[t][1]
        for a_ in (ax, axp):
            a_.axhline(yv, color=GRID, lw=0.25, zorder=0)
        sub = d[(d.tissue == t) & (d.state == s_)]
        for _, r in sub.iterrows():
            mk, filled, off = marks[(r.version, r.gene_set)]
            ax.scatter([r.spearman_vs_gse86619], [yv + off], marker=mk, s=9 if filled else 6,
                       facecolors=col if filled else "white", edgecolors=col, linewidths=0.5, zorder=3 if filled else 4)
        pr = sub[(sub.version == "raw (as G02)") & (sub.gene_set == "all shared genes")]
        axp.scatter(pr.spearman_with_pattern_implied_prediction, [yv] * len(pr), s=8, c=TEXT2, linewidths=0, zorder=3)
        cv.text(x0 + lab_w - 0.8, top + (yv + 0.5) * rp, STATE_LAB[s_], size=PT_SMALL, ha="right", va="center")
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4], ["−0.4", "−0.2", "0", "0.2", "0.4"])
    axp.set_xlim(0, 1)
    axp.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    for a_ in (ax, axp):
        a_.set_ylim(y - 0.5, -0.5)
        a_.set_yticks([])
        clean(a_, left=False)
        tick_style(a_)
    ax.set_xlabel("Spearman ρ with blood timer", fontsize=PT_SMALL, labelpad=1.0)
    cv.text(x0 + w - pat_w, top - 0.8, "Pattern ρ", size=PT_SMALL, va="bottom", color=TEXT2)
    for t in TIS:
        ys = [v for (t_, _), v in yy.items() if t_ == t]
        cv.text(x0 + lab_w + 0.6, top + (min(ys) + 0.5) * rp, TIS[t][0], size=PT_SMALL, va="center",
                color=TEXT, fontweight="bold")
    items = [([{"s": 10, "fc": TEXT2}], "raw"), ([{"s": 10, "marker": "D", "fc": TEXT2}], "centred"),
             ([{"s": 6, "fc": "white", "ec": TEXT2, "lw": 0.5}], "open, no stress genes")]
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, items, gap=1.8)
    fits(xx, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c M3 pseudotime
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "M3 neutrophil-branch pseudotime")
    d = pd.read_csv(R9 / "ed9_m3_pseudotime.csv")
    top = y0 + 6.4
    gap = 7.0
    fh = (h - (top - y0) - gap - 4.0) / 2
    for k, t in enumerate(["bm", "cochlea"]):
        ay = top + k * (fh + gap)
        ax = cv.axes(x0 + 9.0, ay + 2.4, w - 10.0, fh - 2.4)
        s = d[d.tissue == t].sort_values("zt")
        col = TIS[t][1]
        ax.vlines(s.zt, s.q25, s.q75, color=col, lw=0.8, alpha=0.6)
        ax.plot(s.zt, s["median"], color=col, lw=0.6, alpha=0.6)
        ax.scatter(s.zt, s["median"], s=10, c=col, linewidths=0, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        clean(ax)
        phase_axis(ax)
        ax.set_ylabel("Pseudotime", fontsize=PT_SMALL, labelpad=0.8)
        ld_bar(cv, x0 + 9.0, ay + fh + 0.1, w - 10.0, 0.7)
        cv.text(x0 + 9.0, ay + 1.6, TIS[t][0], size=PT_SMALL, va="bottom", fontweight="bold", color=TEXT)


# ------------------------------------------------------------------------------------------------ d M1 maturation
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "M1 maturation score per sample")
    d = pd.read_csv(R9 / "ed9_m1_maturation.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("cell_type").colour
    groups = [g for g in STATES if g in set(d.group)]
    gcol = {g: (TEXT if g == "pooled neutrophils" else pal[g]) for g in groups}
    top = y0 + 6.4
    stack = w < 90                                        # D89: marrow above cochlea in a narrow column
    if stack:
        fw, gy = w - 10.0, 5.0
        fh = (h - (top - y0) - gy - 9.0) / 2
        pos = [(x0 + 9.0, top + k * (fh + gy)) for k in range(2)]
    else:
        gx = 6.0
        fw = (w - 9.0 - gx - 3.0) / 2
        fh = h - (top - y0) - 12.0
        pos = [(x0 + 9.0 + k * (fw + gx), top) for k in range(2)]
    lo, hi = d.median_score.min(), d.median_score.max()
    pad = 0.08 * (hi - lo)
    for k, t in enumerate(["bm", "cochlea"]):
        fx, fy = pos[k]
        ax = cv.axes(fx, fy + 2.4, fw, fh - 2.4)
        for g in groups:
            s = d[(d.tissue == t) & (d.group == g)].sort_values("zt")
            if not len(s):
                continue
            lw = 1.0 if g == "pooled neutrophils" else 0.7
            ax.plot(s.zt, s.median_score, color=gcol[g], lw=lw, zorder=2)
            ax.scatter(s.zt, s.median_score, s=7, c=gcol[g], linewidths=0, zorder=3)
        ax.set_ylim(lo - pad, hi + pad)
        clean(ax)
        phase_axis(ax)
        if k == 0 or stack:
            ax.set_ylabel("Maturation score", fontsize=PT_SMALL, labelpad=0.8)
        else:
            ax.set_yticklabels([])
        ld_bar(cv, fx, fy + fh + 0.1, fw, 0.7)
        cv.text(fx, fy + 1.6, TIS[t][0], size=PT_SMALL, va="bottom", fontweight="bold", color=TEXT)
    items = [([{"kind": "hline", "w": 2.4, "lw": 0.9, "colour": gcol[g]}], STATE_LAB[g]) for g in groups]
    rows = [items[:3], items[3:]] if stack else [items]
    for i, it in enumerate(rows):
        xx = cv.key(x0 + 9.0, y0 + h - 1.4 - (len(rows) - 1 - i) * 3.0, it, gap=1.8)
        fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ f ageing cycle (D89)
def _ring(cv, cx, cy, r, title, arrows, col_title):
    ax = cv.axes(cx - r - 2.8, cy - r - 2.8, 2 * r + 5.6, 2 * r + 5.6)
    lim = r + 2.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    blank(ax)
    for start, end, col in [(0, 12, "#F7F7F7"), (12, 24, "#BDBDBD")]:
        ax.add_patch(Wedge((0, 0), r + 0.4, start * 15 - 90, end * 15 - 90, width=0.8, facecolor=col, edgecolor=AXIS,
                           linewidth=0.3))
    for z in [0, 6, 12, 18]:
        a = np.deg2rad(z * 15 - 90)
        ax.text((r + 1.9) * np.cos(a), (r + 1.9) * np.sin(a), str(z), ha="center", va="center", fontsize=PT_SMALL,
                color=TEXT2)
    for ph, col, solid in arrows:
        a = np.deg2rad(ph * 15 - 90)
        L = r - 1.4
        ax.annotate("", xy=(L * np.cos(a), L * np.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=col, linewidth=1.0,
                                    linestyle="-" if solid else (0, (2.0, 1.2)), shrinkA=0, shrinkB=0), zorder=4)
    cv.text(cx, cy - r - 3.6, title, size=PT_SMALL, ha="center", va="bottom", fontweight="bold", color=col_title)


def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Ageing cycle")            # exploratory: stated in the legend (D100, D103)
    ag = pd.read_csv(R9 / "f5i_ageing_cycle.csv")
    r = 7.8
    for j, t in enumerate(["cochlea", "bm"]):
        d = ag[ag.tissue == t].set_index("score")
        arrows = [(d.loc[s, "peak_zt"], AGEING[s][1], d.loc[s, "p_perm"] < 0.05) for s in AGEING]
        _ring(cv, x0 + 10.8 + j * 21.4, y0 + 22.4, r, TIS[t][0], arrows, TEXT)
    ky = y0 + 40.4
    for i, (s, (lab, col)) in enumerate(AGEING.items()):
        x = cv.key(x0 + 1.0, ky + i * 3.0, [([{"kind": "hline", "w": 3.4, "lw": 1.0, "colour": col}], lab)])
        fits(x, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ g marrow content (D89)
def panel_g(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Marrow-content sensitivity")
    mc = pd.read_csv(R9 / "f4d_marrow_content.csv").sort_values("zt")
    ad = pd.read_csv(R9 / "f4d_marrow_adjusted.csv").set_index("cell_type")
    lab_w, right_w = 13.0, 16.2
    pw = w - lab_w - right_w
    # marrow content per cochlear sample
    t1, h1 = y0 + 7.6, 10.0
    ax1 = cv.axes(x0 + lab_w, t1, pw, h1)
    yv = mc.log2_marrow_vs_structural
    ax1.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax1.plot(mc.zt, yv, color=TIS["cochlea"][1], linewidth=0.6, zorder=2)
    ax1.scatter(mc.zt, yv, s=9, color=TIS["cochlea"][1], linewidths=0, zorder=3)
    lo, hi = np.floor(yv.min()) - 0.2, np.ceil(yv.max()) + 0.2
    ax1.set_ylim(lo, hi)
    assert yv.between(lo, hi).all()
    ax1.set_yticks([-1, 0], ["−1", "0"])
    ax1.tick_params(axis="y", length=1.0, pad=0.6, labelsize=PT_SMALL)
    clean(ax1)
    zt_axis(ax1)
    ld_bar(cv, x0 + lab_w, t1 + h1, pw, 0.8)
    for line_i, s in enumerate(["Marrow", "content", "(log₂)"]):
        cv.text(x0 + lab_w - 4.4, t1 + 2.2 + line_i * 2.2, s, size=PT_SMALL, ha="right", va="center")
    p = mc.rhythm_p.iloc[0]
    cv.text(x0 + lab_w + pw + 1.2, t1 + h1 / 2, f"{P_SYM} = {p:.2f}", size=PT_SMALL, va="center", color=TEXT2)
    # phases before / after expressing each share against the marrow-resident precursors (D17 sensitivity)
    t2, rp = y0 + 26.0, 2.3
    rows = MC_SHARES
    h2 = len(rows) * rp
    ax2 = cv.axes(x0 + lab_w, t2, pw, h2)
    ax2.set_ylim(len(rows) - 0.5, -0.5)
    clean(ax2, left=False)
    zt_axis(ax2)
    cv.text(x0 + w - 0.3, t2 - 0.6, "P", size=PT_SMALL, ha="right", va="bottom", color=TEXT2, style="italic")
    for i, c in enumerate(rows):
        r = ad.loc[c]
        col = r.colour
        u, a_ = r.peak_zt_ols_pct, r.peak_zt_alr_marrow
        if np.isfinite(a_) and abs(a_ - u) < 12:
            ax2.plot([u, a_], [i, i], color=col, linewidth=0.7, zorder=2)
        ax2.scatter([u], [i], s=10, facecolor=col if r.p_ols_pct < 0.05 else "white", edgecolor=col, linewidths=0.5,
                    zorder=4)                                 # primary drawn on top of the sensitivity marker
        if np.isfinite(a_):
            ax2.scatter([a_], [i], s=9, marker="s", facecolor=col if r.p_alr_marrow < 0.05 else "white", edgecolor=col,
                        linewidths=0.5, zorder=3)
            txt = f"{fmt_p(r.p_ols_pct)} → {fmt_p(r.p_alr_marrow)}"
        else:
            txt = f"{fmt_p(r.p_ols_pct)}"
        yy = t2 + (i + 0.5) * rp
        cv.text(x0 + lab_w - 1.0, yy, r.abbr, size=PT_SMALL, ha="right", va="center")
        cv.text(x0 + w - 0.3, yy, txt, size=PT_SMALL, ha="right", va="center", color=TEXT2)
        fits(text_width(txt, PT_SMALL), right_w + 0.0, "d p column")
    ax2.set_xlabel("Peak (ZT, h)", fontsize=PT_SMALL, labelpad=1.0)
    ld_bar(cv, x0 + lab_w, t2 + h2, pw)
    ky = y0 + h - 2.2
    x = cv.key(x0 + 1.0, ky, [([{"s": 10, "fc": GREY}], "unadjusted"),
                              ([{"s": 9, "marker": "s", "fc": "white", "ec": GREY}], "precursor-adjusted")])
    fits(x, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e test design (D82)
BLOOD = "#7A0403"


def panel_e(cv, x0, y0, w, h):
    """Test design of the neutrophil timer split (idealised curves; moved from main Fig. 5 v1 panel a, D82)."""
    cv.letter(x0, y0, "e", "Timer test design")
    ax = cv.axes(x0, y0, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    blank(ax)

    def box(x, y, bw, bh, lines, ec):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.8", facecolor="#F7F7F7",
                                    edgecolor=ec, linewidth=0.6))
        for i, (txt, bold) in enumerate(lines):
            ax.text(x + bw / 2, y + 1.6 + i * 2.1, txt, ha="center", va="center", fontsize=PT_SMALL,
                    fontweight="bold" if bold else "normal", color=TEXT)

    box(0.8, 6.0, 22.4, 5.2, [("Blood neutrophils", True), ("ZT13 − ZT5", False)], BLOOD)
    box(0.8, 12.6, 22.4, 5.2, [("Our neutrophil fit", True), ("predicted ZT13 − ZT5", False)], AXIS)
    ax.add_patch(FancyArrowPatch((12.0, 17.9), (12.0, 21.6), arrowstyle="-|>,head_length=1.4,head_width=0.8",
                                 color=TEXT2, linewidth=0.6, mutation_scale=1.0, shrinkA=0, shrinkB=0))
    ax.text(12.0, 23.4, "ρ across genes", ha="center", va="center", fontsize=PT_SMALL, fontweight="bold")
    ix0, iy0, iw, ih = 28.0, 8.4, 26.5, 11.6
    axc = cv.axes(x0 + ix0, y0 + iy0, iw, ih)
    t = np.linspace(0, 24, 241)
    for pk, colour in [(17.0, TIS["cochlea"][1]), (4.0, TIS["bm"][1])]:
        axc.plot(t, np.cos(2 * np.pi * (t - pk) / 24), color=colour, linewidth=1.0)
        y5, y13 = np.cos(2 * np.pi * (5 - pk) / 24), np.cos(2 * np.pi * (13 - pk) / 24)
        axc.annotate("", xy=(13, y13), xytext=(5, y5),
                     arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", color=colour, linewidth=0.6,
                                     linestyle=(0, (2, 1.2)), shrinkA=1.5, shrinkB=1.5))
    for zt in (5, 13):
        axc.axvline(zt, color=BLOOD, linewidth=0.5, linestyle=(0, (1.5, 1.2)), zorder=0)
        axc.text(zt, 1.32, f"ZT{zt}", ha="center", va="bottom", fontsize=PT_SMALL, color=TEXT)
    axc.set_ylim(-1.25, 1.25)
    axc.set_yticks([])
    clean(axc, left=False)
    zt_axis(axc, ticks=(0, 12, 24))
    ld_bar(cv, x0 + ix0, y0 + iy0 + ih, iw, 0.8)
    for i, (colour, txt) in enumerate([(TIS["cochlea"][1], "ρ > 0: follows blood"), (TIS["bm"][1], "ρ < 0: opposes blood")]):
        xk = cv.key(x0 + ix0, y0 + 26.6 + i * 2.6, [([{"kind": "hline", "w": 3.4, "lw": 1.0, "colour": colour}], txt)])
        fits(xk, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ main
def panel_h(cv, x0, y0, w, h):
    """Round 10 R2 (677): the split statistic D = rho(cochlea) - rho(marrow) over all 3,600 joint relabelings of the two
    tissues' samples (60 x 60 class representatives); line, observed; P, one-sided exact (share of relabelings >= D)."""
    cv.letter(x0, y0, "h", "The split as one exact test")
    dist = pd.read_csv(R9 / "ed9_timer_joint_null_distribution.csv")
    tests = pd.read_csv(R9 / "ed9_timer_joint_null.csv")
    lo, hi = np.floor(dist.split_D.min() * 4) / 4, np.ceil(dist.split_D.max() * 4) / 4
    bins = np.linspace(lo, hi, 41)
    half = w / 2
    for k, (unit, label) in enumerate(SPLIT_UNITS):
        d = dist[dist.unit == unit]
        obs = float(d.loc[d.is_observed, "split_D"].iloc[0])
        t = tests[(tests.unit == unit) & tests.primary].iloc[0]
        assert np.isclose(t.split_D, obs)
        xa = x0 + k * half + 10.0
        cv.text(xa, y0 + 7.4, label, size=PT_SMALL, ha="left", va="bottom", fontweight="bold")
        ax = cv.axes(xa, y0 + 8.4, half - 16.0, h - 8.4 - 9.0)
        ax.hist(d.split_D, bins=bins, color="#BDBDBD", edgecolor="white", linewidth=0.2)
        ax.axvline(obs, color=TEXT, lw=0.8)
        ax.text(0.02, 0.95, f"{P_SYM} = {fmt_p(t.p_joint_one_sided)}", transform=ax.transAxes, ha="left", va="top",
                fontsize=PT_SMALL)
        ax.set_xlim(lo, hi)
        clean(ax)
        tick_style(ax)
        ax.set_xlabel(r"$\rho$ cochlea $-$ $\rho$ marrow", fontsize=PT_SMALL, labelpad=1.0)
        ax.set_ylabel("Relabelings", fontsize=PT_SMALL, labelpad=1.0)


def main():
    cv = Canvas(width=W_ED, height=H_ED)
    panel_a(cv, 0.0, 0.0, 112.0, 64.0)
    panel_b(cv, 116.0, 0.0, 64.0, 92.0)
    panel_c(cv, 0.0, 68.0, 52.0, 52.0)
    panel_f(cv, 0.0, 122.0, 52.0, 48.0)
    panel_e(cv, 56.0, 66.0, 56.0, 32.0)
    panel_d(cv, 56.0, 100.0, 60.0, 70.0)
    panel_g(cv, 120.0, 96.0, 60.0, 52.0)
    panel_h(cv, 0.0, 174.0, 180.0, 40.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
