"""Round 9 (D68) — Fig. 3, one composite at Nature size (180 x 170 mm, 5-7 pt): a systemic night heat-shock programme
follows the activity phase.

a  Five systemic-cue programmes in every cell type of both tissues: peak phase (x), amplitude (size), q < 0.1 (filled),
   amplitude-weighted common phase (black) with its resultant length R, literature-expected phase (triangle). Cell types
   share the six samples, so the convergence is descriptive.
b  Ten heat-shock genes in the whole-tissue pseudobulk (z-scored voom log2 CPM per sample; stored cosinor p / q, peak)
   and the programme mean (average z of the ten genes).
c  Separable from handling: independent cochlea cohort (GSE107086, no noise), night - day change of heat-shock,
   immediate-early and NF-kB genes after sham surgery and adrenalectomy (count of genes up); share of our rhythm calls
   that follow the per-sample handling profile, per gene class.
d  Promoter-motif carriers: circular mean peak per cell type vs the motif's expected phase (directional permutation q).
e  Clock-output test: share of calls at the constant-darkness-predicted phase vs a phase-permuted null, per call set and
   per cell type (observed / null).
f  Mouse (night) vs baboon (day) heat-shock phase on the clock-time ring and after aligning the clocks.
g  Primary driver of every Tier-1/2 rhythm call per cell type; cell types with fewer than MIN_CALLS calls are not shown
   (threshold fixed before plotting; counts of shown / all types in the headers).

Inputs: results/r9_figures/f3*.csv (900 build_f3, 903) and palette_celltypes.csv. No statistics are computed here (the
programme mean in b is a plotted average of the per-gene z-scores). Output: figures/figures/fig3_heat_shock_v1.*
Run: python scripts/911_fig3_heat_shock.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import AXIS, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fmt_p, text_width  # noqa: E402

from lib.figstyle import P_SYM, Q_SYM, R_SYM  # noqa: E402,F401  (italic P / q / r, D99, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "fig3_heat_shock_v3"
TIS = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
HS = "#B2182B"
SET_COL = {"heat shock": HS, "IEG": "#7F7F7F", "NF-kB": "#2C5C8A"}
SET_LAB = {"heat shock": "Heat shock", "IEG": "IEG", "NF-kB": "NF-κB"}
DRIVER_COL = {"core clock": "#000000", "clock-controlled": "#AA3377", "temperature": "#E69F00",   # min CIE76 dE 17
              "glucocorticoid-driven": "#F0E442", "handling-linked": "#56B4E9", "composition-linked": "#228833",
              "sample-pattern-linked": "#9A9A9A", "unexplained": "#E3E3E3"}     # under every CVD simulation (906)
DRIVER_LAB = {"core clock": "Core clock", "clock-controlled": "Clock-controlled", "temperature": "Temperature",
              "glucocorticoid-driven": "Glucocorticoid", "handling-linked": "Handling-linked",
              "composition-linked": "Composition", "sample-pattern-linked": "Sample pattern",
              "unexplained": "Unexplained"}
RDBU = LinearSegmentedColormap.from_list("rdbu", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62",
                                                  "#B2182B"])
LIGHT, DARK = "#FFFFFF", "#4D4D4D"
GREY, EXPECTED, RULE = "#777777", "#9A9A9A", "#E6E6E6"
MIN_CALLS = 20                  # g: cell types with fewer Tier-1/2 calls are not shown (declared before plotting)


def ld_bar(cv, x, y, w, h=0.9):
    """Light (ZT0-12, open) and dark (ZT12-24, filled) bar along a 0-24 h axis spanning w mm."""
    cv.box(x, y, w / 2, h, LIGHT, AXIS, 0.35)
    cv.box(x + w / 2, y, w / 2, h, DARK, AXIS, 0.35)


def zt_axis(ax):
    """0-24 h x-axis whose bottom spine is replaced by the LD bar; tick labels sit below the bar."""
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4.2, labelsize=PT_SMALL)


def fits(x_end, right, what):
    assert x_end <= right + 1e-6, f"{what}: ends at {x_end:.1f} mm, beyond {right:.1f} mm"


# ------------------------------------------------------------------------------------------------ a programmes
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Systemic programme phases")
    ph = pd.read_csv(R9 / "f3a_systemic_phases.csv")
    cp = pd.read_csv(R9 / "f3a_systemic_common.csv").set_index(["tissue", "set"])
    sets = ph.drop_duplicates("set").sort_values("set_order")[["set", "set_label"]].values.tolist()
    lab_w, rcol = 23.0, 7.4
    bw = (w - lab_w - 2 * rcol) / 2
    top = y0 + 9.6
    rh = (h - 8.0 - 12.5) / len(sets)
    amax = ph.amplitude.quantile(0.98)

    def size(a):
        return 2.0 + 16.0 * np.minimum(a, amax) / amax

    for j, t in enumerate(["cochlea", "bm"]):
        bx = x0 + lab_w + j * (bw + rcol)
        cv.text(bx + bw / 2, y0 + 8.4, TIS[t][0], size=PT, ha="center", va="bottom", fontweight="bold")
        cv.text(bx + bw + 1.2, y0 + 8.4, "R", size=PT_SMALL, ha="left", va="bottom", style="italic", color=TEXT2)
        ax = cv.axes(bx, top, bw, rh * len(sets))
        ax.set_ylim(len(sets), 0)
        clean(ax, left=False)
        zt_axis(ax)
        for k in range(1, len(sets)):
            ax.axhline(k, color=RULE, linewidth=0.4, zorder=0)
        for k, (s, _) in enumerate(sets):
            d = ph[(ph.tissue == t) & (ph.set == s)].sort_values("order")
            n = len(d)
            yj = k + 0.5 + (np.arange(n) - (n - 1) / 2) / max(n - 1, 1) * 0.62
            f = (d.q_BH < 0.1).values
            for sel, z in [(~f, 3), (f, 3.5)]:                    # significant types drawn on top
                ax.scatter(d.peak_zt[sel], yj[sel], s=size(d.amplitude.values[sel]),
                           facecolor=d.colour.values[sel] if z > 3 else "white", edgecolor=d.colour.values[sel],
                           linewidths=0.5, zorder=z, clip_on=False)
            c0 = cp.loc[(t, s)]
            ax.plot([c0.weighted_common_peak_zt] * 2, [k + 0.08, k + 0.92], color=TEXT, linewidth=1.1, zorder=4)
            if np.isfinite(c0.expected_peak_zt):
                ax.scatter([c0.expected_peak_zt], [k + 0.1], marker="v", s=9, color=EXPECTED, zorder=4, linewidths=0)
            cv.text(bx + bw + 1.2, top + (k + 0.5) * rh, f"{c0.weighted_resultant:.2f}", size=PT_SMALL, ha="left",
                    va="center", color=TEXT2)
        ax.set_xlabel("Peak (ZT, h)", fontsize=PT_SMALL, labelpad=1.0)
        ld_bar(cv, bx, top + rh * len(sets), bw)
    for k, (s, slab) in enumerate(sets):
        bold = s == "heat_shock_hsf1"
        fits(text_width(slab, PT_SMALL, "bold" if bold else "normal"), lab_w - 1.7, "a row label")
        cv.text(x0 + lab_w - 1.5, top + (k + 0.5) * rh, slab, size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if bold else "normal")
    ky = y0 + h - 2.0
    x = cv.key(x0 + 1.0, ky, [
        ([{"s": 10, "fc": GREY}], f"{Q_SYM} < 0.1"),
        ([{"s": 10, "fc": "white", "ec": GREY}], "n.s."),
        ([{"s": float(size(a)), "fc": GREY, "space": 0.6} for a in (0.15 * amax, 0.5 * amax, amax)], "amplitude"),
        ([{"kind": "vline", "h": 2.4, "lw": 1.1}], "common phase"),
        ([{"marker": "v", "s": 9, "fc": EXPECTED, "lw": 0}], "expected")])
    fits(x, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b heat-shock genes
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Heat-shock genes")
    tc = pd.read_csv(R9 / "f3b_tissue_timecourses.csv")
    hs = tc[tc.gene_set == "heat shock"].copy()
    hs["z"] = hs.groupby(["tissue", "gene"]).log2cpm.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
    genes = hs[hs.tissue == "bm"].drop_duplicates("gene").sort_values("peak_zt").gene.tolist()   # marrow peak order
    assert set(genes) == set(hs.gene)
    zts = sorted(hs.zt.unique())
    assert len(zts) == 6
    cw, rh, lab_w, mean_h = 3.0, 2.6, 10.5, 9.0
    lim = max(1.3, np.ceil((hs.groupby(["tissue", "zt"]).z.mean().abs().max() + 0.15) * 5) / 5)
    for j, t in enumerate(["cochlea", "bm"]):
        bx = x0 + lab_w + j * (6 * cw + 15.0)
        d = hs[hs.tissue == t]
        cv.text(bx + 3 * cw, y0 + 8.4, TIS[t][0], size=PT, ha="center", va="bottom", fontweight="bold")
        axm = cv.axes(bx, y0 + 9.6, 6 * cw, mean_h)               # programme mean: average z of the ten genes
        mz = d.groupby("zt").z.mean().reindex(zts)
        axm.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
        axm.plot(np.arange(6), mz.values, color=HS, linewidth=0.9, marker="o", markersize=1.8, zorder=3)
        axm.set_xlim(-0.5, 5.5)
        axm.set_ylim(-lim, lim)
        clean(axm, bottom=False)
        axm.set_yticks([-1, 0, 1], ["−1", "0", "1"] if j == 0 else [])
        if j == 0:
            axm.set_ylabel("Mean z", fontsize=PT_SMALL, labelpad=0.8)
        hy = y0 + 9.6 + mean_h + 2.6
        ld_bar(cv, bx, hy - 1.5, 6 * cw, 0.8)
        ax = cv.axes(bx, hy, 6 * cw, len(genes) * rh)
        mat = np.vstack([d[d.gene == g].sort_values("zt").z.values for g in genes])
        ax.imshow(np.clip(mat, -2, 2), cmap=RDBU, vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(6), [str(int(z)) for z in zts])
        ax.set_yticks([])
        ax.tick_params(axis="x", length=1.0, pad=0.6, labelsize=PT_SMALL)
        for sp in ax.spines.values():
            sp.set_linewidth(0.35)
        cv.text(bx + 3 * cw, hy + len(genes) * rh + 3.4, "ZT (h)", size=PT_SMALL, ha="center", va="top")
        mk_x = bx + 6 * cw + 1.6
        cv.text(mk_x - 0.9, hy - 0.4, "peak", size=PT_SMALL, ha="left", va="bottom", color=TEXT2)
        for i, g in enumerate(genes):                            # stored fit: q < 0.1 filled, p < 0.05 open
            r = d[d.gene == g].iloc[0]
            yy = hy + (i + 0.5) * rh
            if r.q_BH < 0.1 or r.p < 0.05:
                cv.mark(mk_x, yy, 6, fc=HS if r.q_BH < 0.1 else "white", ec=HS)
            cv.text(mk_x + 1.3, yy, f"{r.peak_zt:.1f}", size=PT_SMALL, va="center", color=TEXT2)
        fits(mk_x + 1.3 + text_width("00.0", PT_SMALL), x0 + w, "b peak column")
        if j == 0:
            for i, g in enumerate(genes):
                cv.text(bx - 1.0, hy + (i + 0.5) * rh, g, size=PT_SMALL, ha="right", va="center", style="italic")
    ky = y0 + h - 3.6
    axc = cv.axes(x0 + lab_w, ky, 13, 1.2)
    axc.imshow(np.linspace(-2, 2, 100)[None, :], aspect="auto", cmap=RDBU, extent=(-2, 2, 0, 1))
    axc.set_yticks([])
    axc.set_xticks([-2, 0, 2], ["−2", "0", "2"])
    axc.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in axc.spines.values():
        sp.set_linewidth(0.3)
    x = cv.key(x0 + lab_w + 13.6, ky + 0.6, [([], "z (log₂ CPM)")])
    x = cv.key(x + 5.0, ky + 0.6, [([{"s": 6, "fc": HS, "ec": HS}], f"{Q_SYM} < 0.1"),
                                   ([{"s": 6, "fc": "white", "ec": HS}], f"{P_SYM} < 0.05")])
    fits(x, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c handling
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Separable from handling")
    ex = pd.read_csv(R9 / "f3c_night_external.csv")
    conds = ["Sham surgery", "Adrenalectomy"]
    sets = ["heat shock", "IEG", "NF-kB"]
    n_genes = ex.groupby(["condition", "gene_set"]).gene.nunique().unstack(0)
    assert (n_genes.nunique(axis=1) == 1).all()            # same genes measured in both conditions
    n_genes = n_genes.iloc[:, 0]
    cv.text(x0 + 1.0, y0 + 4.6, "External cohort", size=PT_SMALL, va="top", color=TEXT2)
    sx, sw, ah = x0 + 7.5, 28.0, h - 21.0
    ax = cv.axes(sx, y0 + 9.5, sw, ah)
    rng = np.random.default_rng(3)
    for i, cnd in enumerate(conds):
        for j, s in enumerate(sets):
            xpos = i * 3.5 + j * 1.0
            d = ex[(ex.condition == cnd) & (ex.gene_set == s)]
            jit = (rng.random(len(d)) - 0.5) * 0.5
            ax.scatter(xpos + jit, d.logFC_night_minus_day, s=4.5, color=SET_COL[s], linewidths=0, alpha=0.9, zorder=3)
            m = d.set_mean.iloc[0]
            ax.plot([xpos - 0.36, xpos + 0.36], [m, m], color=TEXT, linewidth=0.9, zorder=4)
            ax.text(xpos, 1.02, str(int((d.logFC_night_minus_day > 0).sum())), ha="center", va="bottom",
                    fontsize=PT_SMALL, color=TEXT2, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(-0.75, 1.02, "n up", ha="right", va="bottom", fontsize=PT_SMALL, color=TEXT2, clip_on=False,
            transform=ax.get_xaxis_transform())
    ax.axhline(0, color="#999999", linewidth=0.45, zorder=1)
    ax.set_xlim(-0.6, 3.5 + 2.0 + 0.6)
    lo, hi = ex.logFC_night_minus_day.min(), ex.logFC_night_minus_day.max()
    ax.set_ylim(min(-0.9, lo - 0.08), max(1.0, hi + 0.08))                    # every gene inside the axis
    ax.set_yticks([-0.5, 0, 0.5, 1.0], ["−0.5", "0", "0.5", "1"])
    ax.set_xticks([1.0, 3.5 + 1.0], ["Sham\nsurgery", "Adrenal-\nectomy"])
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylabel("Night − day (log₂)", fontsize=PT_SMALL, labelpad=1)
    clean(ax)
    x = cv.key(x0 + 1.0, y0 + h - 2.4, [([{"s": 6, "fc": SET_COL[s]}], SET_LAB[s]) for s in sets]
               + [([{"kind": "hline", "w": 2.4}], "mean")])
    fits(x, x0 + w, "c key")
    # handling-like share of our calls, per gene class
    hc = pd.read_csv(R9 / "f3c_handling_classes.csv")
    cls = ["IEG", "NF-kB", "heat_shock", "other"]
    lab = {"IEG": "IEG", "NF-kB": "NF-κB", "heat_shock": "Heat shock", "other": "Other"}
    lw_ = max(text_width(v, PT_SMALL) for v in lab.values())
    bx = sx + sw + 2.0 + lw_ + 1.2
    axh = cv.axes(bx, y0 + 9.5, x0 + w - 1.0 - bx, ah)
    cv.text(bx - lw_ - 1.2, y0 + 4.6, "This atlas", size=PT_SMALL, va="top", color=TEXT2)
    for i, c in enumerate(cls):
        for t, off in [("cochlea", -0.17), ("bm", 0.17)]:
            r = hc[(hc.tissue == t) & (hc.gene_class == c)]
            assert len(r) == 1
            v = 100 * r.frac_stress_like.iloc[0]
            axh.barh(i + off, v, height=0.32, color=TIS[t][1], linewidth=0)
            if c == "other":                                     # direct tissue labels on the shortest bars
                axh.text(v + 2.0, i + off, TIS[t][0], va="center", ha="left", fontsize=PT_SMALL, color=TEXT)
    axh.set_ylim(len(cls) - 0.45, -0.55)
    axh.set_yticks(range(len(cls)), [lab[c] for c in cls])
    axh.tick_params(axis="y", length=0, pad=1.0, labelsize=PT_SMALL)
    axh.set_xlim(0, 70)
    axh.set_xticks([0, 30, 60])
    axh.set_xlabel("Handling-like (%)", fontsize=PT_SMALL, labelpad=0.8)
    clean(axh)


# ------------------------------------------------------------------------------------------------ d motifs
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Promoter-motif carriers")
    mo = pd.read_csv(R9 / "f3d_motif_phase.csv")
    motifs = ["HSE (HSF1)", "D-box", "E-box", "E′-box", "RORE", "CRE", "NF-κB"]
    assert set(motifs) == set(mo.motif_label)
    lab_w, ah = 13.0, h - 17.5
    ax = cv.axes(x0 + lab_w, y0 + 7.0, w - lab_w - 1.0, ah)
    ax.set_ylim(len(motifs), 0)
    clean(ax, left=False)
    zt_axis(ax)
    for k in range(1, len(motifs)):
        ax.axhline(k, color=RULE, linewidth=0.4, zorder=0)
    for k, m in enumerate(motifs):
        for j, t in enumerate(["cochlea", "bm"]):
            d = mo[(mo.motif_label == m) & (mo.tissue == t)]
            n = len(d)
            yy = k + 0.3 + 0.4 * j + (np.arange(n) - (n - 1) / 2) / max(n - 1, 1) * 0.16
            f = (d.q_directional_perm < 0.1).values
            for sel, z in [(~f, 3), (f, 3.5)]:
                ax.scatter(d.circ_mean_peak_zt[sel], yy[sel], s=7, facecolor=TIS[t][1] if z > 3 else "white",
                           edgecolor=TIS[t][1], linewidths=0.45, zorder=z, clip_on=False)
        e = mo[mo.motif_label == m].expected_zt.dropna()
        if len(e):
            ax.scatter([e.iloc[0]], [k + 0.08], marker="v", s=9, color=EXPECTED, linewidths=0, zorder=4)
        bold = m.startswith("HSE")
        fits(text_width(m, PT_SMALL, "bold" if bold else "normal"), lab_w - 1.4, "d row label")
        cv.text(x0 + lab_w - 1.2, y0 + 7.0 + (k + 0.5) * ah / len(motifs), m, size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if bold else "normal")
    ax.set_xlabel("Mean peak (ZT, h)", fontsize=PT_SMALL, labelpad=1.0)
    ld_bar(cv, x0 + lab_w, y0 + 7.0 + ah, w - lab_w - 1.0)
    sig = mo[(mo.motif_label == "HSE (HSF1)") & (mo.q_directional_perm < 0.1)].sort_values("circ_mean_peak_zt")
    assert set(sig.tissue) == {"bm"}
    cv.text(x0 + w - 1.0, y0 + 6.6, ", ".join(sig.abbr), size=PT_SMALL,
            ha="right", va="bottom", color=TEXT)
    x = cv.key(x0 + 1.0, y0 + h - 2.4, [([{"kind": "box", "fc": TIS["cochlea"][1]}], "Cochlea"),
                                        ([{"kind": "box", "fc": TIS["bm"][1]}], "Marrow"),
                                        ([{"s": 7, "fc": GREY}], f"{Q_SYM} < 0.1"),
                                        ([{"marker": "v", "s": 9, "fc": EXPECTED, "lw": 0}], "expected")])
    fits(x, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e clock output
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Clock-output test")
    co = pd.read_csv(R9 / "f3e_clock_output.csv")
    lab = {"G04": "Core clock", "R05": "Heat shock", "G02": "Neutrophil, cochlea", "G01": "Neutrophil, marrow"}
    order = ["G04", "R05", "G02", "G01"]
    lw_ = max(text_width(v, PT_SMALL) for v in lab.values())
    nw, pw_ = text_width("4,134", PT_SMALL), text_width("0.005", PT_SMALL)
    ax_l = x0 + 0.3 + lw_ + 1.2 + nw + 1.6
    p_r = x0 + w - 0.4
    ax_w = p_r - pw_ - 1.6 - ax_l
    top, ah = y0 + 10.4, 12.0
    ax = cv.axes(ax_l, top, ax_w, ah)
    cv.text(ax_l - 1.6, top - 0.4, "Calls", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    cv.text(p_r, top - 0.4, "P", size=PT_SMALL, ha="right", va="bottom", color=TEXT2, style="italic")
    for i, f in enumerate(order):
        r = co[co.call_set_finding == f]
        assert len(r) == 1
        r = r.iloc[0]
        o, n0 = 100 * r.frac_dd_support, 100 * r.frac_dd_support_phase_null
        c = HS if f == "R05" else ("#000000" if f == "G04" else "#7F7F7F")
        ax.plot([n0, o], [i, i], color=c, linewidth=0.8, zorder=1)
        ax.scatter([o], [i], s=9, color=c, zorder=3, linewidths=0)
        ax.scatter([n0], [i], s=9, facecolor="white", edgecolor=c, linewidths=0.5, zorder=3)
        yy = top + (i + 0.5) * ah / 4
        cv.text(x0 + 0.3 + lw_, yy, lab[f], size=PT_SMALL, ha="right", va="center",
                fontweight="bold" if f == "R05" else "normal")
        cv.text(ax_l - 1.6, yy, f"{int(r.n_calls):,}", size=PT_SMALL, ha="right", va="center", color=TEXT2)
        cv.text(p_r, yy, fmt_p(r.p_dd_support_phase_perm), size=PT_SMALL, ha="right", va="center", color=TEXT2)
    ax.set_ylim(3.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("At predicted phase (%)", fontsize=PT_SMALL, labelpad=0.8)
    clean(ax, left=False)
    x = cv.key(x0 + 1.0, top + ah + 7.4, [([{"s": 9, "fc": "#555555", "lw": 0}], "observed"),
                                      ([{"s": 9, "fc": "white", "ec": "#555555"}], "null")])
    fits(x, x0 + w, "e key")
    # per cell type: observed / phase-null ratio
    bt = pd.read_csv(R9 / "f3e_dd_by_type.csv")
    yb, bh = y0 + 33.4, 10.0
    axb = cv.axes(ax_l, yb, ax_w, bh)
    for k, t in enumerate(["cochlea", "bm"]):
        d = bt[bt.tissue == t].sort_values("obs_over_null")
        n = len(d)
        yy = k + (np.arange(n) - (n - 1) / 2) / max(n - 1, 1) * 0.5
        f = (d.p_perm < 0.05).values
        for sel, z in [(~f, 3), (f, 3.5)]:
            axb.scatter(d.obs_over_null[sel], yy[sel], s=7, facecolor=TIS[t][1] if z > 3 else "white",
                        edgecolor=TIS[t][1], linewidths=0.45, zorder=z)
        cv.text(ax_l - 1.6, yb + (k + 0.5) * bh / 2, TIS[t][0], size=PT_SMALL, ha="right",
                va="center")
    axb.axvline(1.0, color="#999999", linewidth=0.45, zorder=1)
    axb.set_ylim(1.5, -0.5)
    axb.set_xlim(0.6, 1.6)
    axb.set_xticks([0.75, 1.0, 1.25, 1.5], ["0.75", "1", "1.25", "1.5"])
    axb.set_xlabel("Observed / null", fontsize=PT_SMALL, labelpad=0.8)
    clean(axb, left=False)


# ------------------------------------------------------------------------------------------------ f baboon
def _ring(cv, cx, cy, r, title, points, summaries):
    """24-h ring (ZT0 at the top, clockwise); points = (phase, filled, colour); summaries = (phase, length, colour)."""
    ax = cv.axes(cx - r - 3.2, cy - r - 3.2, 2 * r + 6.4, 2 * r + 6.4)
    lim = r + 3.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    blank(ax)
    ring = 1.0
    for start, end, col in [(0, 12, "#F7F7F7"), (12, 24, "#BDBDBD")]:
        ax.add_patch(Wedge((0, 0), r + ring / 2, start * 15 - 90, end * 15 - 90, width=ring, facecolor=col,
                           edgecolor=AXIS, linewidth=0.3))
    for z in [0, 6, 12, 18]:
        a = np.deg2rad(z * 15 - 90)
        ax.text((r + 2.1) * np.cos(a), (r + 2.1) * np.sin(a), str(z), ha="center", va="center", fontsize=PT_SMALL,
                color=TEXT2)
    for ph, filled, col in points:
        a = np.deg2rad(ph * 15 - 90)
        rr = r - 1.6
        ax.scatter([rr * np.cos(a)], [rr * np.sin(a)], s=5, facecolor=col if filled else "white", edgecolor=col,
                   linewidths=0.4, zorder=3)
    for ph, length, col in summaries:
        a = np.deg2rad(ph * 15 - 90)
        L = (r - 2.6) * length
        ax.annotate("", xy=(L * np.cos(a), L * np.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=col, linewidth=1.0,
                                    shrinkA=0, shrinkB=0), zorder=4)
    cv.text(cx, cy - r - 4.2, title, size=PT_SMALL, ha="center", va="bottom")


def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Mouse night = baboon day")
    bb = pd.read_csv(R9 / "f3f_baboon_heatshock.csv")
    tis = bb[bb.row_type == "tissue"]
    summ = bb[bb.row_type == "summary"].set_index("baboon_tissue")
    s_raw = summ.loc["SUMMARY baboon (tissues p < 0.05): raw ZT"]
    s_al = summ.loc["SUMMARY baboon (tissues p < 0.05): aligned"]
    s_mo = summ.loc["SUMMARY mouse (this atlas, HSF1 Tier-1/2 calls)"]
    bcol, mcol = "#6B4C9A", HS
    r = 12.0
    cy = y0 + 24.0
    pts_raw = [(p, pv < 0.05, bcol) for p, pv in zip(tis.peak_zt, tis.p)]
    pts_al = [(p, pv < 0.05, bcol) for p, pv in zip(tis.aligned, tis.p)]
    _ring(cv, x0 + 15.4, cy, r, "Clock time (ZT)", pts_raw, [(s_raw.peak_zt, s_raw.p, bcol), (s_mo.peak_zt, s_mo.p, mcol)])
    _ring(cv, x0 + 45.6, cy, r, "Clocks aligned", pts_al, [(s_al.aligned, s_al.p, bcol), (s_mo.aligned, s_mo.p, mcol)])
    x = cv.key(x0 + 1.0, cy + r + 6.4, [([{"kind": "box", "fc": mcol}], "Mouse"),
                                        ([{"kind": "box", "fc": bcol}], "Baboon")])
    fits(x, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ g drivers
def panel_g(cv, pal, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Drivers of rhythm calls")
    dl = pd.read_csv(R9 / "f3g_driver_landscape.csv")
    dl = dl[dl.tier_set == "tier1/2"]
    n_all = dl.groupby("tissue").cell_type.nunique()
    dl = dl[dl.n_calls >= MIN_CALLS]
    order = pal.set_index("cell_type")["order"]
    drivers = list(DRIVER_COL)
    assert set(dl.driver) <= set(drivers)
    lab_w, key_w, nh = 9.0, 24.0, 6.0
    pw = w - lab_w - key_w
    top = y0 + 10.6
    ax_n = cv.axes(x0 + lab_w, top, pw, nh)
    ax = cv.axes(x0 + lab_w, top + nh + 1.6, pw, h - (top - y0) - nh - 1.6 - 12.0)
    xpos, ticks, labels, sep = 0.0, [], [], []
    for t in ["cochlea", "bm"]:
        d = dl[dl.tissue == t]
        types = sorted(d.cell_type.unique(), key=lambda c: order[c])
        start = xpos
        for ct_ in types:
            dd = d[d.cell_type == ct_].set_index("driver")
            assert abs(dd.share.sum() - 1) < 1e-9
            bottom = 0.0
            for dr in drivers:
                v = dd.share.get(dr, 0.0) * 100
                ax.bar(xpos, v, bottom=bottom, width=0.82, color=DRIVER_COL[dr], linewidth=0)
                bottom += v
            ax_n.bar(xpos, np.log10(dd.n_calls.iloc[0]), width=0.82, color=TIS[t][1], linewidth=0)
            ticks.append(xpos)
            labels.append(dd.abbr.iloc[0])
            xpos += 1
        sep.append((start, xpos - 1, t, len(types)))
        xpos += 1.4
    xmax = xpos - 1.4 + 0.2
    for a_ in (ax, ax_n):
        a_.set_xlim(-0.7, xmax)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(ticks, labels, rotation=90)
    ax.tick_params(axis="x", length=0, pad=0.8, labelsize=PT_SMALL)
    ax.set_ylabel("Share of calls (%)", fontsize=PT_SMALL, labelpad=1)
    clean(ax)
    ax.spines["bottom"].set_visible(False)
    ax_n.set_ylim(1, 3.4)
    ax_n.set_yticks([2, 3], ["10²", "10³"])
    ax_n.tick_params(axis="y", length=1.0, pad=0.6, labelsize=PT_SMALL)
    clean(ax_n, bottom=False)
    cv.text(x0 + lab_w - 4.6, top + nh / 2, "Calls", size=PT_SMALL, ha="right", va="center")
    for a_, b_, t, n in sep:
        xa = x0 + lab_w + (a_ - 0.41 + 0.7) / (xmax + 0.7) * pw
        xb = x0 + lab_w + (b_ + 0.41 + 0.7) / (xmax + 0.7) * pw
        cv.box(xa, top - 1.5, xb - xa, 0.6, TIS[t][1])
        cv.text((xa + xb) / 2, top - 1.9, TIS[t][0], size=PT_SMALL, ha="center",
                va="bottom")
    kx = x0 + w - key_w + 2.0
    for i, dr in enumerate(drivers):
        x = cv.key(kx, top + 1.0 + i * 3.0, [([{"kind": "box", "fc": DRIVER_COL[dr],
                                               "ec": "#999999" if dr == "unexplained" else "none", "lw": 0.3}],
                                             DRIVER_LAB[dr])])
        fits(x, x0 + w, "g key")



# ------------------------------------------------------------------------------------------------ main
def main():
    pal = pd.read_csv(R9 / "palette_celltypes.csv")
    cv = Canvas()
    panel_a(cv, 0.0, 0.0, 105.0, 60.0)
    panel_b(cv, 107.0, 0.0, 73.0, 60.0)
    panel_c(cv, 0.0, 62.0, 63.0, 50.0)
    panel_d(cv, 65.0, 62.0, 57.0, 50.0)
    panel_e(cv, 124.0, 62.0, 56.0, 50.0)
    panel_f(cv, 0.0, 114.0, 60.0, 56.0)
    panel_g(cv, pal, 62.0, 114.0, 118.0, 56.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
