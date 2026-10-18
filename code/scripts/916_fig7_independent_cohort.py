"""Round 9 (D68 / D70; Fig. 6 until the Fig. 4 split, D134) — Fig. 7, one composite at Nature size (180 x 170 mm,
5-7 pt; minimal in-figure text, D69): an
independent cohort (Ma et al. 2025) replicates the tissue rhythms and time-of-day noise responses; composition and immune
clocks agree only conditionally (bottom row g-i, exploratory; stated in the legend). D103: no background band or daggers,
panel a on a ZT -3..21 window (no interval wraps around midnight), d-f x-axes on one baseline.

a  Design and clock-based dating: inferred collection phase of every sample group (T2 primary, T1 open) on our ZT axis
   (window ZT -3..21; negative values = the preceding night), with the noise-exposure windows.
b  Phase separation between groups (absolute, T2, 95 % CI) vs the upper bound of a noise-induced clock shift.
c  Label transfer: share of each transferred label among the cohort's cells vs its share in our cochlea.
d  Time-matched untreated bulk: night - day change predicted from our cochlear tissue rhythms vs observed, per gene.
e  Heat-shock genes: night - day in untreated cochleae (filled) and after noise (open).
f  Genes changed after day vs night noise (q < 0.05), up and down.
g  EXPLORATORY: observed vs predicted change in cell-type share between night- and day-noise libraries.
h  EXPLORATORY: deviation of each immune clock's night - day shift from the structural shift (tracking within 4 h).
i  EXPLORATORY: evidence lines per claim and the exploratory grade.

Inputs: results/r9_figures/f6*.csv (900 build_f7; the table names keep their v3 prefix). No statistics are computed
here (the least-squares line in d is a display). Output: figures/figures/fig7_independent_cohort_v4.*
Run: python scripts/916_fig7_independent_cohort.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (AXIS, DARK, LIGHT, PT, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, fmt_p,  # noqa: E402
                          text_width)
from matplotlib.ticker import NullLocator  # noqa: E402

from lib.figstyle import P_SYM, Q_SYM, R_SYM  # noqa: E402,F401  (italic P / q / r, D99, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "fig7_independent_cohort_v5"
TIS = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
GROUP_COL = {"control": "#1A1A1A", "day_noise": "#E69F00", "night_noise": "#6B4C9A",
             "CMD": "#1A1A1A", "CMN": "#1A1A1A", "NMD": "#E69F00", "NMN": "#6B4C9A"}
GROUP_ROWS = [("scRNA", "control", "control"), ("scRNA", "day_noise", "day noise"), ("scRNA", "night_noise", "night noise"),
              ("bulk", "CMD", "control, day"), ("bulk", "CMN", "control, night"), ("bulk", "NMD", "day noise"),
              ("bulk", "NMN", "night noise")]
DAY_BAND, NIGHT_BAND = "#FBE7C6", "#E6DEF0"
HS = "#B2182B"
UP, DOWN = "#B2182B", "#2166AC"
STATUS = [("supporting", "#1A1A1A"), ("inconclusive", "#BDBDBD"), ("opposing", "#CC79A7"),
          ("not testable", "#FFFFFF"), ("confound check", "#E69F00")]
CLAIM_DISPLAY = {"Immune composition": "Immune composition", "Band neutrophils, rest phase": "Band neutrophils",
                 "Classical monocytes, rest phase": "Classical monocytes", "Naïve-B timing": "Naïve-B timing",
                 "B lineage, active phase": "B lineage, night", "Granulopoiesis timing": "Granulopoiesis",
                 "Maturation / ageing": "Maturation / ageing", "Blood-timer state": "Blood-timer state",
                 "Monocyte → macrophage": "Monocyte → macrophage", "Retention / release": "Retention / release",
                 "Macrophage identity": "Macrophage identity", "Functional readiness": "Functional readiness",
                 "B programme rhythms": "B programmes", "Myeloid clocks": "Myeloid clocks",
                 "B-clock shift": "B-clock shift", "Noise readiness × time": "Noise readiness × time"}
GRADE_SHORT = {"E2 exploratory support": "E2", "E1 qualified exploratory support": "E1", "E0 no exploratory support": "E0",
               "Ma-inconclusive / mixed": "mixed", "Ma-supported (exploratory)": "support",
               "Ma-opposed (exploratory)": "opposed"}
GREY = "#777777"


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


# ------------------------------------------------------------------------------------------------ a design / dating
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Design and collection times")
    gp = pd.read_csv(R9 / "f6a_group_phase.csv")
    lab_w, top = 21.0, y0 + 10.6
    span = len(GROUP_ROWS) + 0.6
    ah = h - (top - y0) - 12.0
    pw = w - lab_w - 1.0
    z0, z1 = -3.0, 21.0                                   # 24-h window; every 95 % interval fits without wrapping (D103)
    ax = cv.axes(x0 + lab_w, top, pw, ah)
    for lo, hi, col in [(2, 4, DAY_BAND), (14, 16, NIGHT_BAND)]:
        ax.axvspan(lo, hi, color=col, zorder=0, linewidth=0)
    for z, s in [(3, "day noise"), (15, "night noise")]:
        cv.text(x0 + lab_w + (z - z0) / 24 * pw, top - 0.6, s, size=PT_SMALL, ha="center", va="bottom", color=TEXT2)
    for i, (assay, g, lab) in enumerate(GROUP_ROWS):
        yy = i + (0.6 if assay == "bulk" else 0.0)
        col = GROUP_COL[g]
        for est, filled, off in [("T2", True, 0.0), ("T1", False, 0.28)]:
            r = gp[(gp.assay == assay) & (gp.group == g) & (gp.estimator == est)]
            assert len(r) == 1, (assay, g, est)
            r = r.iloc[0]
            c, hw = r.our_zt % 24, r.ci95_halfwidth_h
            c = c - 24 if c > z1 else c
            assert z0 <= c - hw and c + hw <= z1, (assay, g, est, c, hw)
            ax.plot([c - hw, c + hw], [yy + off] * 2, color=col, linewidth=0.9 if filled else 0.5, zorder=2,
                    solid_capstyle="butt")
            ax.scatter([c], [yy + off], s=14 if filled else 8, facecolor=col if filled else "white", edgecolor=col,
                       linewidths=0.6, zorder=3)
        n = int(gp[(gp.assay == assay) & (gp.group == g)].n_samples.iloc[0])
        cv.text(x0 + lab_w - 1.0, top + (yy + 0.5 + 0.14) / span * ah, f"{lab} ({n})", size=PT_SMALL, ha="right",
                va="center")
    ax.set_ylim(span - 0.5 + 0.28, -0.5)
    ax.axhline(2.8 + 0.14, color="#BBBBBB", linewidth=0.4)
    clean(ax, left=False)
    ax.set_xlim(z0, z1)                                  # as zt_axis, on the shifted window
    ax.set_xticks([0, 6, 12, 18], ["0", "6", "12", "18"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4.2, labelsize=PT_SMALL)
    for a_, b_, fc in [(z0, 0.0, DARK), (0.0, 12.0, LIGHT), (12.0, z1, DARK)]:     # LD bar: preceding night, day, night
        cv.box(x0 + lab_w + (a_ - z0) / 24 * pw, top + ah, (b_ - a_) / 24 * pw, 0.9, fc, AXIS, 0.35)
    ax.set_xlabel("Inferred collection time (ZT, h)", fontsize=PT_SMALL, labelpad=1.0)
    for assay, ya, yb in [("scRNA", -0.5, 2.8), ("bulk", 2.9, span - 0.5 + 0.28)]:
        y_a = top + (ya + 0.5) / (span + 0.28) * ah
        y_b = top + (yb + 0.5) / (span + 0.28) * ah
        cv.text(x0 + 1.4, (y_a + y_b) / 2, assay, size=PT_SMALL, ha="center", va="center", rotation=90,
                fontweight="bold")
    x = cv.key(x0 + 1.0, y0 + h - 1.8, [([{"s": 14, "fc": GREY}], "T2"), ([{"s": 8, "fc": "white", "ec": GREY}], "T1")])
    fits(x, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b separation
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Phase separation")
    sp = pd.read_csv(R9 / "f6b_phase_separation.csv")
    order = ["bulk: day noise (NMD − CMD)", "bulk: night noise (NMN − CMN)", "bulk: CMN − CMD", "bulk: NMN − NMD",
             "scRNA: day_noise − control", "scRNA: night_noise − control", "scRNA: night_noise − day_noise"]
    lab = {"bulk: day noise (NMD − CMD)": "NMD − CMD", "bulk: night noise (NMN − CMN)": "NMN − CMN",
           "bulk: CMN − CMD": "CMN − CMD", "bulk: NMN − NMD": "NMN − NMD", "scRNA: day_noise − control": "day − control",
           "scRNA: night_noise − control": "night − control", "scRNA: night_noise − day_noise": "night − day"}
    assert set(order) == set(sp.comparison), set(sp.comparison) ^ set(order)
    bound = float(sp.noise_shift_bound_h.iloc[0])
    lab_w, top = 17.8, y0 + 8.6
    ah = h - (top - y0) - 12.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvspan(0, bound, color="#EFEFEF", zorder=0, linewidth=0)
    ax.axvline(bound, color=GREY, linewidth=0.6, linestyle=(0, (2, 1.5)), zorder=1)
    for i, c in enumerate(order):
        r = sp[sp.comparison == c].iloc[0]
        noise = r.kind.startswith("noise")
        col = "#6B4C9A" if "night" in c.split(":")[1] and not noise else ("#1A1A1A" if not noise else GREY)
        lo_ = max(r.abs_separation_h - r.ci95_halfwidth_h, 0)
        ax.plot([lo_, r.abs_separation_h + r.ci95_halfwidth_h], [i, i], color=col, linewidth=0.9, zorder=2)
        beyond = lo_ > bound
        ax.scatter([r.abs_separation_h], [i], s=12, facecolor=col if beyond else "white", edgecolor=col, linewidths=0.6,
                   zorder=3)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) / len(order) * ah, lab[c], size=PT_SMALL, ha="right", va="center")
    ax.axhline(3.5, color="#BBBBBB", linewidth=0.4)
    hi = 15.0
    assert (sp.abs_separation_h + sp.ci95_halfwidth_h).max() <= hi
    ax.set_xlim(0, hi)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks([0, 4, 8, 12])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("Phase separation (h)", fontsize=PT_SMALL, labelpad=0.8)
    for assay, ya, yb in [("bulk", -0.5, 3.5), ("scRNA", 3.5, len(order) - 0.5)]:
        y_a = top + (ya + 0.5) / len(order) * ah
        y_b = top + (yb + 0.5) / len(order) * ah
        cv.text(x0 + 1.4, (y_a + y_b) / 2, assay, size=PT_SMALL, ha="center", va="center", rotation=90,
                fontweight="bold")
    x = cv.key(x0 + 1.0, y0 + h - 1.8, [([{"kind": "box", "fc": "#EFEFEF", "ec": GREY, "lw": 0.3}], "noise-shift bound")])
    fits(x, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c label transfer
def panel_c(cv, col, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Label transfer")
    lt = pd.read_csv(R9 / "f6c_label_transfer.csv")
    lt = lt[(lt.share_ma > 0) & (lt.share_atlas_cochlea > 0)]
    ax = cv.axes(x0 + 9.0, y0 + 7.6, w - 10.0, h - 7.6 - 12.0)
    lo, hi = 3e-5, 0.3
    assert lt.share_ma.between(lo, hi).all() and lt.share_atlas_cochlea.between(lo, hi).all()
    ax.plot([lo, hi], [lo, hi], color="#BBBBBB", linewidth=0.5, linestyle=(0, (2, 1.5)), zorder=0)
    conf = lt.frac_confident >= 0.5
    for sel, z in [(~conf, 2), (conf, 3)]:
        ax.scatter(lt.share_atlas_cochlea[sel], lt.share_ma[sel], s=10, facecolor=[col[a] for a in lt.abbr[sel]] if z == 3
                   else "white", edgecolor=[col[a] for a in lt.abbr[sel]], linewidths=0.6, zorder=z)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1]
    labs = ["0.01", "0.1", "1", "10"]
    ax.set_xticks(ticks, labs)
    ax.set_yticks(ticks, labs)
    ax.minorticks_off()
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Our cochlea (% of cells)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Independent cohort (% of cells)", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 1.8, [([{"s": 10, "fc": GREY}], "≥ 50 % confident"),
                                        ([{"s": 10, "fc": "white", "ec": GREY}], "< 50 %")])
    fits(x, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d bulk replication
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Bulk rhythms replicate")
    d = pd.read_csv(R9 / "f6d_bulk_replication.csv")
    axl, axt, axw, axh = x0 + 9.0, y0 + 7.4, w - 10.0, h - 7.4 - 12.0       # baseline shared with e and f (D103)
    ax = cv.axes(axl, axt, axw, axh)
    x, y = d.predicted_night_minus_day, d.observed_log2FC
    ax.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.scatter(x, y, s=2.0, color="#7A7A7A", alpha=0.5, linewidths=0, zorder=2, rasterized=True)
    k, c0 = np.polyfit(x, y, 1)                               # display line only
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, k * xx + c0, color=TIS["cochlea"][1], linewidth=1.0, zorder=3)
    xl = (np.floor(x.min() * 2) / 2, np.ceil(x.max() * 2) / 2)
    yl = (np.floor(y.min() * 2) / 2, np.ceil(y.max() * 2) / 2)
    assert x.between(*xl).all() and y.between(*yl).all()
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Predicted night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Observed night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    r0 = d.iloc[0]
    ax.text(0.03, 0.98, f"{R_SYM} = {r0.pearson_r:.2f}\n" + (f"{P_SYM} < 0.001" if r0.p_perm < 0.001 else f"{P_SYM} = {fmt_p(r0.p_perm)}"), transform=ax.transAxes, fontsize=PT_SMALL,
            va="top", ha="left")


# ------------------------------------------------------------------------------------------------ e heat shock
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Night heat shock lost after noise")
    he = pd.read_csv(R9 / "f6e_heat_shock_bulk.csv").sort_values("baseline_night_minus_day", ascending=False)
    lab_w, top = 13.0, y0 + 8.4
    ah = h - (top - y0) - 12.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    n = len(he)
    for i, (_, r) in enumerate(he.iterrows()):
        ax.annotate("", xy=(r.after_noise_night_minus_day, i), xytext=(r.baseline_night_minus_day, i),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.16", color=HS, linewidth=0.6,
                                    shrinkA=2.0, shrinkB=2.2))
        ax.scatter([r.baseline_night_minus_day], [i], s=12, color=HS, linewidths=0, zorder=3)
        ax.scatter([r.after_noise_night_minus_day], [i], s=12, facecolor="white", edgecolor=HS, linewidths=0.6, zorder=3)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) / n * ah, r.gene, size=PT_SMALL, ha="right", va="center",
                style="italic")
    vals = pd.concat([he.baseline_night_minus_day, he.after_noise_night_minus_day])
    lo, hi = np.floor(vals.min() * 4) / 4 - 0.1, np.ceil(vals.max() * 4) / 4 + 0.1
    assert vals.between(lo, hi).all()
    ax.set_xlim(lo, hi)
    ax.set_ylim(n - 0.5, -0.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("Night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 1.8, [([{"s": 12, "fc": HS}], "untreated"),
                                        ([{"s": 12, "fc": "white", "ec": HS}], "after noise")])
    fits(x, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ f noise response
def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Larger response to night noise")
    nr = pd.read_csv(R9 / "f6f_noise_response.csv").set_index("contrast")
    rows = [("noise_effect_day_group", "Day noise"), ("noise_effect_night_group", "Night noise")]
    lab_w, top = 13.0, y0 + 10.0
    ah = h - (top - y0) - 12.0                            # same baseline as d and e (D103)
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 2.0, ah)
    ax.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    lim = 600
    assert nr.n_up.max() <= lim and nr.n_down.max() <= lim
    for i, (c, lab) in enumerate(rows):
        r = nr.loc[c]
        ax.barh(i, r.n_up, height=0.5, color=UP, linewidth=0)
        ax.barh(i, -r.n_down, height=0.5, color=DOWN, linewidth=0)
        ax.text(r.n_up + 12, i, f"{int(r.n_up)}", fontsize=PT_SMALL, va="center", ha="left", color=TEXT2)
        ax.text(-r.n_down - 12, i, f"{int(r.n_down)}", fontsize=PT_SMALL, va="center", ha="right", color=TEXT2)
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) * ah / 2, lab, size=PT_SMALL, ha="right", va="center")
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) * ah / 2 + 2.3, f"{int(r.n_de)} genes", size=PT_SMALL, ha="right",
                va="center", color=TEXT2)
    ax.set_xlim(-lim, lim + 60)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([-400, 0, 400], ["400", "0", "400"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel(f"Genes, {Q_SYM} < 0.05", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 1.8, [([{"kind": "box", "fc": DOWN}], "down"), ([{"kind": "box", "fc": UP}], "up")])
    fits(x, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ g composition (expl.)
def panel_g(cv, col, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Composition, night vs day noise")
    g = pd.read_csv(R9 / "f6g_composition_sc.csv")
    axl, axt, axw, axh = x0 + 9.0, y0 + 8.0, w - 10.0, h - 8.0 - 9.6
    ax = cv.axes(axl, axt, axw, axh)
    x, y = g.predicted_log2_ratio, g.observed_log2_ratio
    ax.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.scatter(x, y, s=16, color=[col[a] for a in g.abbr], edgecolor="white", linewidths=0.3, zorder=3)
    xl = (np.floor(x.min() * 2) / 2 - 0.2, np.ceil(x.max() * 2) / 2 + 0.2)
    yl = (np.floor(y.min() * 2) / 2 - 0.2, np.ceil(y.max() * 2) / 2 + 0.2)
    assert x.between(*xl).all() and y.between(*yl).all()
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Predicted night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Observed night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)

    def to_mm(px, py):
        return axl + (px - xl[0]) / (xl[1] - xl[0]) * axw, axt + (yl[1] - py) / (yl[1] - yl[0]) * axh

    pm = np.array([to_mm(a_, b_) for a_, b_ in zip(x, y)])
    srt = np.argsort(pm[:, 1])
    ys = spread(pm[srt, 1], 2.05, axt + 0.8, axt + axh - 0.8)
    for k, yl_ in zip(srt, ys):
        px, py = pm[k]
        right = px < axl + axw * 0.62
        xt = px + 2.4 if right else px - 2.4
        cv.line([px, xt - (0.3 if right else -0.3)], [py, yl_], GREY, 0.3)
        cv.text(xt, yl_, g.abbr.iloc[k], size=PT_SMALL, va="center", ha="left" if right else "right")
    r0 = g.iloc[0]
    ax.text(0.03, 0.98, f"ρ = {r0.spearman_absolute:.2f}\n{P_SYM} = {fmt_p(r0.p_perm_one_sided)}", transform=ax.transAxes,
            fontsize=PT_SMALL, va="top", ha="left")


# ------------------------------------------------------------------------------------------------ h clocks (expl.)
def panel_h(cv, col, x0, y0, w, h):
    cv.letter(x0, y0, "h", "Immune clock shifts")
    ca = pd.read_csv(R9 / "f6h_clock_tracking.csv")
    ind = ca[~ca.pooled].copy()
    ind["abbr"] = ind.unit.map(dict(zip(fd.celltypes().cell_type, fd.celltypes().abbr)))
    ind = ind.sort_values(["lineage", "deviation_from_structural_h"], ascending=[False, True])
    pool = ca[ca.pooled].copy()
    pool["abbr"] = pool.unit.map({"pooled neutrophils": "Neutrophils", "monocytes + macrophages": "Mono + Mac",
                                  "neutrophils + monocytes (pooled)": "Neu + Mono", "B lineage (pooled)": "B lineage"})
    assert pool.abbr.notna().all()
    rows = [(r, False) for _, r in ind.iterrows()] + [(r, True) for _, r in pool.iterrows()]
    lab_w, top = 13.6, y0 + 8.0
    span = len(rows) + 0.6
    ah = h - (top - y0) - 9.8
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, ah)
    ax.axvspan(-4, 4, color="#EFEFEF", zorder=0, linewidth=0)
    ax.axvline(0, color=GREY, linewidth=0.5, linestyle=(0, (2, 1.5)), zorder=1)
    for i, (r, pooled) in enumerate(rows):
        yy = i + (0.6 if pooled else 0.0)
        c = col[r.abbr] if not pooled else ("#B37400" if r.lineage == "myeloid" else "#1B7A3A")
        ax.scatter([r.deviation_from_structural_h], [yy], s=11 if not pooled else 14,
                   marker="o" if not pooled else "D", facecolor=c if r.tracking_within_4h else "white", edgecolor=c,
                   linewidths=0.6, zorder=3)
        cv.text(x0 + lab_w - 1.0, top + (yy + 0.5) / span * ah, r.abbr, size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if pooled else "normal")
    ax.axhline(len(ind) - 0.5 + 0.3, color="#BBBBBB", linewidth=0.4)
    assert ca.deviation_from_structural_h.abs().max() <= 12
    ax.set_xlim(-12.8, 12.8)
    ax.set_ylim(span - 0.5, -0.5)
    ax.set_xticks([-12, -6, 0, 6, 12], ["−12", "−6", "0", "6", "12"])
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("Shift vs structural clocks (h)", fontsize=PT_SMALL, labelpad=0.8)


# ------------------------------------------------------------------------------------------------ i ladder (expl.)
def panel_i(cv, x0, y0, w, h):
    cv.letter(x0, y0, "i", "Evidence per claim")
    la = pd.read_csv(R9 / "f6i_claim_ladder.csv").sort_values("row_order")
    la["display"] = la.claim_short.map(CLAIM_DISPLAY)
    la["grade"] = la.updated_exploratory_grade.map(GRADE_SHORT)
    assert la.display.notna().all() and la.grade.notna().all()
    lab_w = max(text_width(s, PT_SMALL) for s in la.display) + 1.4
    gw = max(text_width(s, PT_SMALL) for s in la.grade) + 1.0
    top = y0 + 8.0
    two_rows = w < 68.0                                   # 180-mm layout (D95): key in two rows
    ah = h - (top - y0) - 9.8 - (2.4 if two_rows else 0.0)
    pw = w - lab_w - gw - 1.4
    ax = cv.axes(x0 + lab_w, top, pw, ah)
    n = len(la)
    cols = [c for c, _ in STATUS if c in la.columns]
    tot_max = int(la[cols].sum(1).max())
    for i, (_, r) in enumerate(la.iterrows()):
        left = 0
        for st, colr in STATUS:
            v = int(r.get(st, 0) or 0)
            if v:
                ax.barh(i, v, left=left, height=0.7, color=colr, edgecolor="#9A9A9A" if st == "not testable" else "none",
                        linewidth=0.3)
                left += v
        cv.text(x0 + lab_w - 1.0, top + (i + 0.5) / n * ah, r.display, size=PT_SMALL, ha="right", va="center")
        cv.text(x0 + w - 0.3, top + (i + 0.5) / n * ah, r.grade, size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if r.grade in ("E2", "support") else "normal",
                color=TEXT if r.grade in ("E2", "support", "E1") else TEXT2)
    ax.set_xlim(0, tot_max + 0.4)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(0, tot_max + 1, 2))
    ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax, left=False)
    ax.set_xlabel("Lines", fontsize=PT_SMALL, labelpad=0.8)
    items = [([{"kind": "box", "fc": c, "ec": "#9A9A9A" if s == "not testable" else "none", "lw": 0.3}],
              s.replace("confound check", "confound")) for s, c in STATUS]
    rows = [items[:3], items[3:]] if two_rows else [items]
    for r_, it in enumerate(rows):
        x = cv.key(x0 + 1.0, y0 + h - 1.8 - 2.4 * (len(rows) - 1 - r_), it)
        fits(x, x0 + w, f"i key {r_}")


# ------------------------------------------------------------------------------------------------ main
def main():
    col = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"].to_dict()
    cv = Canvas()
    panel_a(cv, 0.0, 0.0, 70.0, 60.0)
    panel_b(cv, 72.0, 0.0, 52.0, 60.0)
    panel_c(cv, col, 126.0, 0.0, 54.0, 60.0)
    panel_d(cv, 0.0, 62.0, 58.0, 50.0)
    panel_e(cv, 60.0, 62.0, 58.0, 50.0)
    panel_f(cv, 120.0, 62.0, 60.0, 50.0)
    panel_g(cv, col, 0.0, 115.0, 58.0, 55.0)
    panel_h(cv, col, 60.0, 115.0, 52.0, 55.0)
    panel_i(cv, 114.0, 115.0, 66.0, 55.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
