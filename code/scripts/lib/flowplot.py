"""Flow-cytometry panel helpers (D89; restyled D92 to the Nature Portfolio reporting-summary conventions): contour plots
with outliers rebuilt from the tables exported by 932 (probability contours, outlier events, gate polygons and gate
frequencies; no event data are read by the figure scripts), marker-fluorochrome axis labels, scales on the bottom-left
plot of each group, thin black gates with their frequency (% of the plotted parent); per-mouse dot plots with the
geometric mean, its 95 % CI and a comparison bracket."""
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from .figstyle import AXIS, LW_THIN, PT_SMALL, TEXT, TEXT2, clean, halo, text_width
from .figstyle import P_SYM

COF = 0.05                                                # asinh cofactor of the 932 display (bead-relative units)
FL_MAJOR = (0.0, 0.1, 1.0, 10.0)                          # bead-relative decade ticks (labels 0, 1, 10)
FL_MINOR = tuple(k * 10.0 ** e for e in (-2, -1, 0, 1) for k in range(2, 10) if k * 10.0 ** e <= 30)
SC_MAJOR, SC_MINOR = (0, 100, 200), (50, 150, 250)        # scatter, x 10^3
CONTOUR_LW, GATE_LW = 0.3, 0.5
SCATTER = {"FSC", "SSC", "FSCH"}


def load_tables(r9):
    """The four 932 plot tables, indexed for drawing."""
    plots = pd.read_csv(r9 / "flow_plots.csv").set_index("plot")
    cont = pd.read_csv(r9 / "flow_contours.csv")
    outl = pd.read_csv(r9 / "flow_outliers.csv")
    gates = pd.read_csv(r9 / "flow_gates.csv")
    return {"plots": plots, "cont": {k: d for k, d in cont.groupby("plot")}, "outl": {k: d for k, d in outl.groupby("plot")},
            "gates": {k: d for k, d in gates.groupby("plot")}}


def _fl(v):
    return np.arcsinh(np.asarray(v, float) / COF)


def axis_title(label, axis):
    return f"{label} (×10³)" if axis in SCATTER else label


def p_label(p, fmt):
    """'P = 0.002' / 'P < 0.001' (italic P) from a formatted p ('<0.001')."""
    t = fmt(p)
    return f"{P_SYM} < {t[1:]}" if t.startswith("<") else f"{P_SYM} = {t}"


def fmt_freq(v):
    """Gate frequency with three significant digits, as in FlowJo (61.4, 9.18, 0.52); 0 for an empty gate."""
    if v == 0:
        return "0"
    return f"{v:.3g}" if v < 100 else f"{v:.0f}"


def _ticks(ax, which, axis, numbers):
    set_major, set_minor, set_labels = ((ax.set_xticks, ax.xaxis.set_minor_locator, ax.set_xticklabels) if which == "x"
                                        else (ax.set_yticks, ax.yaxis.set_minor_locator, ax.set_yticklabels))
    if axis in SCATTER:
        set_major(list(SC_MAJOR))
        set_minor(FixedLocator(list(SC_MINOR)))
        labels = [f"{v:g}" for v in SC_MAJOR]
    else:                                                 # 0.1 keeps its decade tick; its label would collide with 0
        set_major(list(_fl(FL_MAJOR)))
        set_minor(FixedLocator(list(_fl(FL_MINOR))))
        labels = ["" if v == 0.1 else f"{v:g}" for v in FL_MAJOR]
    set_labels(labels if numbers else [""] * len(labels))


def flow_contour(cv, x, y, s, pid, tabs, numbers=(True, True), titles=(True, True), names=False, gates_shown=None,
                 label_size=PT_SMALL):
    """One contour plot with outliers (932 tables) at (x, y) mm, side s mm. numbers / titles: (x axis, y axis) tick
    numbers and axis titles; names: print the gate name above its frequency; gates_shown: gate names to draw (default
    all). Returns the plot's row of flow_plots.csv."""
    pr = tabs["plots"].loc[pid]
    ax = cv.axes(x, y, s, s)
    for (_, _), d in tabs["cont"].get(pid, pd.DataFrame(columns=["level", "path", "x", "y"])).groupby(["level", "path"]):
        ax.plot(d.x.to_numpy(), d.y.to_numpy(), color=TEXT, linewidth=CONTOUR_LW, solid_joinstyle="round", zorder=2)
    o = tabs["outl"].get(pid)
    if o is not None and len(o):
        ax.scatter(o.x, o.y, s=0.22, color=TEXT, linewidths=0, zorder=2, rasterized=False)
    gx_range, gy_range = pr.x_max - pr.x_min, pr.y_max - pr.y_min
    for g in tabs["gates"][pid].itertuples():
        if gates_shown is not None and g.gate_name not in gates_shown:
            continue
        vx = np.array([float(v) for v in g.vertices_x.split(";")])
        vy = np.array([float(v) for v in g.vertices_y.split(";")])
        ax.add_patch(Polygon(np.column_stack([vx, vy]), closed=True, fill=False, edgecolor=TEXT, linewidth=GATE_LW,
                             zorder=3))
        x0, x1, y0, y1 = vx.min(), vx.max(), vy.min(), vy.max()
        px, py = 0.035 * gx_range, 0.04 * gy_range
        if g.label_corner in ("above", "below"):           # small gates: label outside, left-aligned with the gate
            cx, ha = x0 + px * 0.3, "left"
            cy, va = (y1 + py * 0.5, "bottom") if g.label_corner == "above" else (y0 - py * 0.5, "top")
        else:
            cx, ha = (x0 + px, "left") if g.label_corner[1] == "l" else (x1 - px, "right")
            cy, va = (y1 - py, "top") if g.label_corner[0] == "t" else (y0 + py, "bottom")
        txt = f"{g.gate_name}\n{fmt_freq(g.pct_shown)}" if names else fmt_freq(g.pct_shown)
        ax.text(cx, cy, txt, ha=ha, va=va, fontsize=label_size, color=TEXT, linespacing=1.0, zorder=4,
                path_effects=halo(1.0))
    ax.set_xlim(pr.x_min, pr.x_max)
    ax.set_ylim(pr.y_min, pr.y_max)
    _ticks(ax, "x", pr.x_axis, numbers[0])
    _ticks(ax, "y", pr.y_axis, numbers[1])
    ax.tick_params(which="major", length=1.4, width=LW_THIN, pad=0.6, labelsize=PT_SMALL, colors=AXIS)
    ax.tick_params(which="minor", length=0.7, width=LW_THIN * 0.8, colors=AXIS)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    for sp in ax.spines.values():
        sp.set_linewidth(LW_THIN)
        sp.set_color(AXIS)
    if titles[0]:
        ax.set_xlabel(axis_title(pr.x_label, pr.x_axis), fontsize=PT_SMALL, labelpad=0.8)
    if titles[1]:
        ax.set_ylabel(axis_title(pr.y_label, pr.y_axis), fontsize=PT_SMALL, labelpad=0.8)
    return pr


def arrow(cv, x0, y0, x1, y1, lw=0.5, colour=TEXT):
    """A thin gating arrow between two points (mm)."""
    cv.fig.add_artist(FancyArrowPatch((x0 / cv.w, 1 - y0 / cv.h), (x1 / cv.w, 1 - y1 / cv.h), transform=cv.fig.transFigure,
                                      arrowstyle="-|>,head_length=2.2,head_width=1.2", mutation_scale=1.0, linewidth=lw,
                                      color=colour, shrinkA=0, shrinkB=0))


def gmean_ci(v):
    """Geometric mean and its t-based 95 % CI (log scale)."""
    from scipy import stats
    lv = np.log(np.asarray(v, float))
    m, se = lv.mean(), lv.std(ddof=1) / np.sqrt(len(lv))
    t = stats.t.ppf(0.975, len(lv) - 1)
    return float(np.exp(m)), float(np.exp(m - t * se)), float(np.exp(m + t * se))


def flow_dots(cv, x, y, w, h, groups, colours, xlabels, ylab=None, seed=0, ymax=None, brackets=()):
    """Per-mouse values per group (list of arrays), jittered; black bar = geometric mean, thin line = its 95 % CI.
    brackets: (i, j, text) comparisons drawn above the groups. Returns (axes, y top)."""
    ax = cv.axes(x, y, w, h)
    rng = np.random.default_rng(seed)
    gtop = []
    for k, (v, col) in enumerate(zip(groups, colours)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=4.0, color=col, linewidths=0, zorder=3, alpha=0.95)
        gm, lo, hi = gmean_ci(v)
        ax.plot([k - 0.28, k + 0.28], [gm, gm], color=TEXT, linewidth=0.8, zorder=4, solid_capstyle="butt")
        ax.plot([k, k], [lo, hi], color=TEXT, linewidth=0.45, zorder=4)
        for yv in (lo, hi):
            ax.plot([k - 0.1, k + 0.1], [yv, yv], color=TEXT, linewidth=0.45, zorder=4)
        gtop.append(max(float(v.max()), hi))
    span = ymax or max(gtop) * 1.12
    if brackets:                                          # room for each bracket and its label above the groups
        span = max(span, max(max(gtop[i], gtop[j]) for i, j, _ in brackets) * 1.06 / 0.80)
    ax.set_ylim(0, span)
    for i, j, txt in brackets:
        yb = max(gtop[i], gtop[j]) + 0.06 * span
        tick = span * 0.025
        ax.plot([i, i, j, j], [yb - tick, yb, yb, yb - tick], color=TEXT, linewidth=0.45, zorder=4, clip_on=False)
        ax.text((i + j) / 2, yb + span * 0.015, txt, ha="center", va="bottom", fontsize=PT_SMALL, color=TEXT2,
                clip_on=False)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(xlabels), fontsize=PT_SMALL)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(length=1.2, pad=0.6, labelsize=PT_SMALL, width=LW_THIN)
    clean(ax)
    if ylab:
        ax.set_ylabel(ylab, fontsize=PT_SMALL, labelpad=0.8)
    return ax, span


def text_fits(s, size, width):
    return text_width(s, size) <= width
