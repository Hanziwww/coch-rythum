"""Round 10 (D134; from the Fig. 4 of D68-D97) — Fig. 5: the cochlear immune compartment alternates over the day
(180 x 170 mm, 5-7 pt). Built from the former Fig. 4e-k when Fig. 4 was split (user: reorganise the two figures to make
them more informative): the former phase-robustness panel (4f) became the timetable a, and the flow cohort (c) now shows
the raw plots of all three pre-specified readouts.

a  Day-night timetable of the twelve timed haematopoietic shares of b: high phase (fit above its mesor; bar, shaded by
   register verdict), peak of the figure model (black tick), peaks of the five core composition models (dots; filled,
   P < 0.05; CLR by permutation), number of supporting core models and the leave-one-time-point-out (LOTO) phase SD;
   diamond, replicated in the flow cohort (c; D91).
b  Timed haematopoietic shares (anchor): six samples (% of tissue cells), OLS cosinor fit with its 95 % confidence band,
   register verdict (robust / supported: filled points, solid fit; suggestive: open, dashed; not supported: open,
   dotted); header, peak (ZT, h); diamond, replicated in the flow cohort.
c  Independent flow cohort (D89; 932; manuscript/flow_validation_plan.md, pre-specified): for each primary readout, the
   ZT5 and ZT17 contour plots of the mouse closest to its group median (P1 cochlear immature neutrophils, T2: CD11b vs
   Ly6G of Ly6G+ SSC-hi cells; P2 cochlear classical monocytes, T3: Ly6C vs CD11b in the monocyte scatter gate; P3
   marrow recirculating B cells, T6: B220 vs IgD of B220+ cells; gate frequency = % of the plotted parent) and the
   per-mouse values (% of CD45+; bar, geometric mean with its 95 % CI; ZT17 / ZT5 ratio and Holm P).
d  Label-free production waves: circular mean peak of the Milo neighbourhoods per pseudotime quintile, neutrophil and B
   lineages, selection-free (solid) and nominal P < 0.1 neighbourhoods (dashed).
e  Tier-1 genes of aged and activated marrow neutrophils ordered by peak phase: per-sample z of voom log2 CPM.
f  Sorted marrow neutrophils of an independent study (GSE312298, PBS): predicted vs observed night - day per gene.
g  Baboon spleen and marrow: share of our marrow rhythm genes peaking within 3 h of the clock-aligned baboon peak vs the
   permutation null, per cell type and overall.
The marrow-content sensitivity is Extended Data Fig. 9g; the per-sample haematopoietic composition is Extended Data
Fig. 1f.

Inputs: results/r9_figures (900 build_f5: f5a_share_peaks, f4b_share_fits, f4c_composition_models, f4a, f4e, f4h, f4i;
903: f4g; 932: f4f_flow_shares, flow_plots / _contours / _outliers / _gates). No statistics are computed here (z-scores,
the least-squares line in f and the above-mesor windows of a are display transforms of exported fits).
Output: figures/figures/fig5_immune_timetable_v1.*   Run: python scripts/912_fig5_immune_timetable.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.flowplot import axis_title, flow_contour, flow_dots, load_tables, p_label  # noqa: E402
from lib.figstyle import (AXIS, LW_THIN, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, fmt_p, ld_bar,  # noqa: E402
                          text_width, zt_axis)

from lib.figstyle import P_SYM, R_SYM  # noqa: E402,F401  (italic P / r, D99)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
PANELS = ROOT / "figures" / "panels"
NAME = "fig5_immune_timetable_v3"
TIS = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
RDBU = LinearSegmentedColormap.from_list("rdbu", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62",
                                                  "#B2182B"])
GREY, RULE = "#777777", "#E6E6E6"
SHARES = {"cochlea": ["Neutrophil (Band)", "Monocyte (Classical)", "Granulocyte-Monocyte Progenitor",
                      "Neutrophil (Mature)", "Neutrophil (Aged)", "B Cell (Naïve)"],
          "bm": ["B Cell (Naïve)", "B Cell (Transitional)", "Pro-B Cell (Cycling)", "HSC / MPP",
                 "Granulocyte-Monocyte Progenitor", "Neutrophil (Mature)"]}
VERDICT = {"robust": ("-", True, TEXT), "supported": ("-", True, TEXT), "suggestive": ((0, (3, 1.6)), False, TEXT2),
           "not supported": ((0, (0.8, 1.2)), False, "#8C8C8C")}
CLASS_ORDER = ["Lymphoid", "Progenitor", "Cycling", "Erythroid", "Myeloid"]      # stacking order, bottom to top
LABEL_GENES = {"Neutrophil (Aged)": ["Fos", "Ly6g", "Fkbp5", "Camp", "Hsph1", "Cdkn1a", "Cebpb", "Il1b"],
               "Neutrophil (Activated)": ["Fos", "Fkbp5", "Mki67", "Hspa8", "Cebpb", "Spp1", "Egr2", "Il1r2"]}


def spread(ys, gap, lo, hi):
    """Push sorted label positions apart to at least `gap`, inside [lo, hi] (order kept)."""
    ys = np.array(ys, float)
    for _ in range(200):
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


# ------------------------------------------------------------------------------------------------ a timetable
VERDICT_BAR = {"robust": (0.85, None), "supported": (0.85, None), "suggestive": (0.35, None),
               "not supported": (0.0, 0.5)}                 # (fill alpha, outline width) of the high-phase bar


def segments(zt, on):
    """Runs of True in a boolean series over a ZT grid -> [(start, end)] in hours."""
    out, start = [], None
    for z, v in zip(zt, on):
        if v and start is None:
            start = z
        if not v and start is not None:
            out.append((start, z))
            start = None
    if start is not None:
        out.append((start, zt[-1]))
    return out


def panel_timetable(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Day\u2013night timetable")
    fitd = pd.read_csv(R9 / "f4b_share_fits.csv")
    cm = pd.read_csv(R9 / "f4c_composition_models.csv")
    pk = pd.read_csv(R9 / "f5a_share_peaks.csv").set_index(["tissue", "cell_type"])
    rows = [(t, c) for t in ["cochlea", "bm"] for c in SHARES[t]]
    assert len(pk) == len(rows) and all(r in pk.index for r in rows), "a rows"
    lab_w, right_w = 17.0, 12.5
    top = y0 + 9.4
    ah = h - 9.4 - 18.0
    pw = w - lab_w - right_w
    ax = cv.axes(x0 + lab_w, top, pw, ah)
    n = len(rows)
    rh = ah / n
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 24)
    ax.axvspan(12, 24, color="#EDEDED", linewidth=0, zorder=0)              # dark phase
    clean(ax, left=False)
    zt_axis(ax)
    ax.set_xlabel("ZT (h)", fontsize=PT_SMALL, labelpad=1.0)
    xr = x0 + w - right_w
    cv.text(xr + 5.2, top - 0.6, "core", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    cv.text(x0 + w - 0.3, top - 0.6, "LOTO", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)
    offs = np.linspace(0.02, 0.38, 5)                         # core models on their own sub-lines below the bar
    for i, (t, c) in enumerate(rows):
        f = fitd[(fitd.tissue == t) & (fitd.cell_type == c)].sort_values("zt")
        st = f.iloc[0]
        verdict = pk.loc[(t, c), "verdict"]
        alpha, edge = VERDICT_BAR[verdict]
        col = st.colour
        assert abs(float(pk.loc[(t, c), "peak_zt"]) - float(st.peak_zt)) < 1e-6, (t, c)
        for a_, b_ in segments(f.zt.to_numpy(), (f.fit_pct >= f.mean_pct).to_numpy()):
            ax.add_patch(Rectangle((a_, i - 0.36), b_ - a_, 0.3, facecolor=col if alpha else "none",
                                   alpha=alpha if alpha else 1.0, edgecolor=col if edge else "none",
                                   linewidth=edge or 0.0, zorder=2))
        dd = cm[(cm.tissue == t) & (cm.cell_type == c)]
        core = dd[dd.model != "ols_pct"].sort_values("model")
        assert len(core) == 5
        for k, (_, r) in enumerate(core.iterrows()):
            ax.scatter([r.peak_zt], [i + offs[k]], s=3.2, facecolor=col if r.supports else "white", edgecolor=col,
                       linewidths=0.35, zorder=3, clip_on=False)
        ols = dd[dd.model == "ols_pct"].iloc[0]
        ax.plot([ols.peak_zt] * 2, [i - 0.44, i + 0.44], color=TEXT, linewidth=0.9, zorder=4)
        if i == 6:
            ax.axhline(5.5, color="#BBBBBB", linewidth=0.4, zorder=1)
        yy = top + (i + 0.5) * rh
        cv.text(x0 + lab_w - 1.0, yy, st.abbr, size=PT_SMALL, ha="right", va="center")
        if bool(pk.loc[(t, c), "flow_replicated"]):
            cv.mark(x0 + lab_w - 2.2 - text_width(st.abbr, PT_SMALL), yy, 5.0, "D", TEXT)
        cv.text(xr + 5.2, yy, f"{int(ols.core_supporting)}/{int(ols.core_tested)}", size=PT_SMALL, ha="right",
                va="center", color=TEXT2)
        cv.text(x0 + w - 0.3, yy, f"{ols.loto_phase_sd_h:.1f} h", size=PT_SMALL, ha="right", va="center", color=TEXT2)
    for g, t in enumerate(["cochlea", "bm"]):
        ya, yb = top + g * 6 * rh + 0.5, top + (g + 1) * 6 * rh - 0.5
        cv.line([x0 + 2.6, x0 + 2.6], [ya, yb], TIS[t][1], 0.8)
        cv.text(x0 + 1.6, (ya + yb) / 2, TIS[t][0], size=PT_SMALL, ha="center", va="center", rotation=90,
                color=TEXT, fontweight="bold")
    ld_bar(cv, x0 + lab_w, top + ah, pw)
    ky = top + ah + 8.4
    rows_k = [[([{"kind": "box", "fc": GREY, "size": 1.4}], "high phase, robust / supported"),
               ([{"kind": "box", "fc": "#CDCDCD", "size": 1.4}], "suggestive")],
              [([{"kind": "box", "fc": "white", "ec": GREY, "lw": 0.5, "size": 1.4}], "not supported"),
               ([{"kind": "vline", "h": 2.0, "lw": 0.9}], "peak (figure model)")],
              [([{"s": 3.2, "fc": GREY}], f"core model, {P_SYM} < 0.05"), ([{"s": 3.2, "fc": "white", "ec": GREY}], "n.s."),
               ([{"kind": "mark", "s": 5.0, "marker": "D", "fc": TEXT}], "flow-replicated (c)")]]
    for r_, it in enumerate(rows_k):
        x = cv.key(x0 + 1.0, ky + 2.4 * r_, it)
        fits(x, x0 + w, f"a key {r_}")
    assert ky + 2.4 * (len(rows_k) - 1) + 1.0 <= y0 + h + 0.5, "a height"


# ------------------------------------------------------------------------------------------------ b timed shares
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Timed haematopoietic shares")
    fitd = pd.read_csv(R9 / "f4b_share_fits.csv")
    comp = pd.read_csv(R9 / "f4a_immune_composition.csv")
    lab_w = 4.0
    cell_w = (w - lab_w - 1.2) / 6                       # 1.2 mm right margin keeps the last '24' on the page
    extra = max(0.0, (h - 62.0) / 2)                     # taller band: taller plots, same layout
    ax_w, ax_h = cell_w - 3.4, 17.6 + extra
    for k, t in enumerate(["cochlea", "bm"]):
        ry = y0 + 9.2 + k * (26.4 + extra)
        cv.text(x0 + 1.9, ry + ax_h / 2, f"{TIS[t][0]} (% of cells)", size=PT_SMALL, ha="center", va="center",
                rotation=90, color=TEXT, fontweight="bold")
        for i, ct_ in enumerate(SHARES[t]):
            cx = x0 + lab_w + i * cell_w + 3.4
            cur = fitd[(fitd.tissue == t) & (fitd.cell_type == ct_)].sort_values("zt")
            pts = comp[(comp.tissue == t) & (comp.cell_type == ct_)]
            assert len(cur) > 100 and len(pts) == 6, (t, ct_)
            st = cur.iloc[0]
            verdict = st.register_verdict if isinstance(st.register_verdict, str) else "not supported"
            ls, filled, vcol = VERDICT[verdict]
            col = st.colour
            ax = cv.axes(cx, ry, ax_w, ax_h)
            ymax = max(cur.band_hi.max(), pts.pct.max()) * 1.06
            ax.fill_between(cur.zt, np.clip(cur.band_lo, 0, None), cur.band_hi, color=col, alpha=0.2, linewidth=0)
            ax.plot(cur.zt, cur.fit_pct, color=col, linewidth=1.0, linestyle=ls)
            ax.scatter(pts.zt, pts.pct, s=9, facecolor=col if filled else "white", edgecolor=col, linewidths=0.6,
                       zorder=3)
            assert pts.pct.min() >= 0 and pts.pct.max() <= ymax
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.tick_params(axis="y", length=1.0, pad=0.6, labelsize=PT_SMALL)
            clean(ax)
            zt_axis(ax, ticks=(0, 12, 24))
            ld_bar(cv, cx, ry + ax_h, ax_w, 0.8)
            cv.text(cx, ry - 0.6, st.abbr, size=PT_SMALL, ha="left", va="bottom", fontweight="bold")
            if bool(st.flow_replicated):                         # D91: replicated in the flow cohort (c; register D90)
                cv.mark(cx - 1.3, ry - 1.45, 5.0, "D", TEXT)
            tag = f"{st.peak_zt:.1f}"
            cv.text(cx + ax_w + 1.0, ry - 0.6, tag, size=PT_SMALL, ha="right", va="bottom", color=TEXT2)   # 1 mm into the gutter
            fits(cx + text_width(st.abbr, PT_SMALL, "bold") + 0.8 + text_width(tag, PT_SMALL), cx + ax_w + 1.01,
                 f"b header {st.abbr}")
    ky = y0 + h - 2.0
    x = cv.key(x0 + lab_w, ky, [
        ([{"kind": "hline", "w": 4.0, "lw": 1.0, "colour": GREY, "mark_fc": GREY}], "robust / supported"),
        ([{"kind": "hline", "w": 4.0, "lw": 1.0, "colour": GREY, "mark_fc": "white", "ls": VERDICT["suggestive"][0]}],
         "suggestive"),
        ([{"kind": "hline", "w": 4.0, "lw": 1.0, "colour": GREY, "mark_fc": "white",
           "ls": VERDICT["not supported"][0]}], "not supported"),
        ([{"kind": "mark", "s": 5.0, "marker": "D", "fc": TEXT}], "replicated in the flow cohort (c)")])
    fits(x, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c flow cohort
FLOW_BLOCKS = [("P1", "T2", "co", "Cochlear immature neutrophils"), ("P2", "T3", "co", "Cochlear classical monocytes"),
               ("P3", "T6", "bm", "Marrow recirculating B cells")]


def panel_flow(cv, x0, y0, w, h):
    """c — independent flow cohort (D89; D92 style; 932): per primary readout, the ZT5 and ZT17 contour plots with
    outliers (marker-fluorochrome axes; numbers and the y title on the left plot; gate frequency = % of the plotted
    parent) and the per-mouse values (% of CD45+; geometric mean with its 95 % CI; ZT17 / ZT5 ratio and Holm P).
    D142: block titles in black with a tissue glyph."""
    cv.letter(x0, y0, "c", "Independent flow cohort")
    tabs = load_tables(R9)
    sh = pd.read_csv(R9 / "f4f_flow_shares.csv")
    gapb, ya, gp, gd, yd, wd = 3.0, 5.6, 0.7, 3.4, 5.6, 12.6
    bw = (w - 2 * gapb) / 3
    s_ = min(16.0, (bw - ya - gp - gd - yd - wd) / 2)
    top = y0 + 8.4
    assert top + s_ + 5.6 <= y0 + h + 0.2, f"c height ({top + s_ + 5.6 - y0:.1f} > {h})"
    for j, (key, tube, tis, name) in enumerate(FLOW_BLOCKS):
        bx = x0 + j * (bw + gapb)
        col = TIS["cochlea" if tis == "co" else "bm"][1]
        px = bx + ya
        tw_ = text_width(name, PT_SMALL, "bold")
        cv.text(px + s_ + gp / 2, top - 3.3, name, size=PT_SMALL, ha="center", va="bottom", fontweight="bold")
        cv.mark(px + s_ + gp / 2 - tw_ / 2 - 1.2, top - 4.3, 9.0, "o", col, col)
        fits(tw_, 2 * s_ + gp + gd + yd + wd, f"c title {key}")
        for k, zt in enumerate(("ZT5", "ZT17")):
            xx = px + k * (s_ + gp)
            cv.text(xx + s_ / 2, top - 0.5, zt, size=PT_SMALL, ha="center", va="bottom")
            flow_contour(cv, xx, top, s_, f"main_{tube}_{tis}_{zt}", tabs, numbers=(k == 0, k == 0),
                         titles=(False, k == 0))
        pr = tabs["plots"].loc[f"main_{tube}_{tis}_ZT5"]
        cv.text(px + s_ + gp / 2, top + s_ + 2.5, axis_title(pr.x_label, pr.x_axis), size=PT_SMALL, ha="center",
                va="top")
        d = sh[(sh.readout == key) & sh.included.astype(bool)]
        r0 = d.iloc[0]
        xq = px + 2 * s_ + gp + gd + yd
        # ZT17 / ZT5 ratio and Holm P as a two-line header over the dots (the bracket marks the comparison)
        for txt, yy in ((f"×{r0.ratio_zt17_zt5:.2f}", top - 3.3), (p_label(r0.p_holm, fmt_p), top - 0.5)):
            cv.text(xq + wd / 2, yy, txt, size=PT_SMALL, ha="center", va="bottom", color=TEXT2)
            fits(xq + wd / 2 + text_width(txt, PT_SMALL) / 2, x0 + w + 1e-6, f"c header {key}")
        flow_dots(cv, xq, top, wd, s_, [d[d.zt == "ZT5"].value, d[d.zt == "ZT17"].value], [col, col], ["ZT5", "ZT17"],
                  ylab="% of CD45⁺", seed=j, brackets=[(0, 1, "")])
        fits(xq + wd, x0 + w + 1e-6, f"c dots {key}")


# ------------------------------------------------------------------------------------------------ d waves
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Production waves along pseudotime")
    wv = pd.read_csv(R9 / "f4e_milo_waves.csv")
    mh, top = 27.0, y0 + 9.4
    rgap = 7.2                                            # room for the r labels of facet 1 before facet 2 (D92 width)
    mw = min(15.2, (w - 8.8 - rgap - text_width("r 0.00", PT_SMALL) - 0.8) / 2)
    for j, (lin, lab) in enumerate([("neutrophil", "Neutrophil lineage"), ("B", "B lineage")]):
        bx = x0 + 8.8 + j * (mw + rgap)
        fits(bx + mw + 0.8 + text_width("r 0.00", PT_SMALL), (x0 + w + 1e-6) if j else (bx + mw + rgap - 1.0),
             f"h r labels {lin}")
        cv.text(bx + mw / 2, top - 2.6, lab, size=PT_SMALL, ha="center", va="bottom", fontweight="bold")
        ax = cv.axes(bx, top, mw, mh)
        for t in ["bm", "cochlea"]:
            for free, ls, lw in [(False, (0, (2.2, 1.4)), 0.6), (True, "-", 1.0)]:
                d = wv[(wv.lineage == lin) & (wv.tissue == t) & (wv.selection_free == free)].sort_values("quintile")
                assert len(d) == 5
                ax.plot(d.quintile, d.peak_zt, linestyle=ls, linewidth=lw, color=TIS[t][1], zorder=2 + free,
                        marker="o" if free else None, markersize=2.2, markeredgewidth=0)
                if free:
                    r0 = d.iloc[0]
                    cv.text(bx + mw + 0.8, top + mh * (1 - d.peak_zt.iloc[-1] / 24), f"{R_SYM} {r0.circ_linear_r:.2f}",
                            size=PT_SMALL, va="center", color=TEXT)
        ax.set_xlim(0.6, 5.4)
        ax.set_ylim(0, 24)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=4.0, labelsize=PT_SMALL)
        ax.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
        clean(ax)
        ax.spines["left"].set_visible(False)
        ld_bar(cv, bx - 0.9, top, mh, 0.9, vertical=True)
        ax.set_xlabel("Pseudotime quintile", fontsize=PT_SMALL, labelpad=1.0)
        if j == 0:
            ax.set_ylabel("Peak (ZT, h)", fontsize=PT_SMALL, labelpad=1.0)
    items = [([{"kind": "hline", "w": 4.0, "lw": 1.0, "colour": GREY, "mark_fc": GREY, "s": 4.8}], "all neighbourhoods"),
             ([{"kind": "hline", "w": 4.0, "lw": 0.6, "colour": GREY, "ls": (0, (2.2, 1.4))}], f"nominal {P_SYM} < 0.1 only")]
    rows = [items[:1], items[1:]] if w < 50.0 else [items]
    for r_, it in enumerate(rows):
        x = cv.key(x0 + 1.0, y0 + h - 2.2 - 2.4 * (len(rows) - 1 - r_), it)
        fits(x, x0 + w, f"h key {r_}")


# ------------------------------------------------------------------------------------------------ e Tier-1 calls
def panel_g(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Neutrophil-state programmes")
    gx = pd.read_csv(R9 / "f4g_neutrophil_state_genes.csv")
    gx["z"] = gx.groupby(["group", "gene"]).log2cpm.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
    zts = sorted(gx.zt.unique())
    cw, top, hh = (2.5 if w >= 60.0 else 2.35), y0 + 9.4, h - 19.0
    lab_w = max(text_width(g, PT_SMALL, style="italic") for v in LABEL_GENES.values() for g in v)
    pitch = min(30.6, w - 6.2 - 6 * cw - 2.1 - lab_w - 0.2)  # second heat map ends at the panel edge
    for j, (grp, lab) in enumerate([("Neutrophil (Aged)", "Aged"), ("Neutrophil (Activated)", "Activated")]):
        bx = x0 + 6.2 + j * pitch
        d = gx[gx.group == grp]
        order = d.drop_duplicates("gene").sort_values(["peak_zt", "gene"]).gene.tolist()
        mat = d.pivot(index="gene", columns="zt", values="z").loc[order, zts]
        n = len(order)
        ax = cv.axes(bx, top, 6 * cw, hh)
        ax.imshow(np.clip(mat.values, -2, 2), cmap=RDBU, vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(6), [str(int(z)) for z in zts])
        ax.set_yticks([])
        ax.tick_params(axis="x", length=1.0, pad=0.6, labelsize=PT_SMALL)
        for sp in ax.spines.values():
            sp.set_linewidth(0.35)
        ld_bar(cv, bx, top - 1.5, 6 * cw, 0.8)
        cv.text(bx + 3 * cw, top - 2.2, f"{lab} ({n})", size=PT_SMALL, ha="center", va="bottom", fontweight="bold")
        pk = d.drop_duplicates("gene").set_index("gene").loc[order, "peak_zt"].to_numpy()
        for zb in [6, 12, 18]:                               # peak-phase ruler on the left
            yrow = np.searchsorted(pk, zb) / n * hh
            cv.line([bx - 1.0, bx], [top + yrow, top + yrow], TEXT, 0.4)
            cv.text(bx - 1.3, top + yrow, f"ZT{zb}", size=PT_SMALL, ha="right", va="center", color=TEXT2)
        genes = [g for g in LABEL_GENES[grp] if g in set(order)]
        ys = np.array([(order.index(g) + 0.5) / n * hh for g in genes])
        lab_y = spread(np.sort(ys), 2.15, 0.6, hh - 0.6)
        srt = [genes[k] for k in np.argsort(ys)]
        for g, ya, yl in zip(srt, np.sort(ys), lab_y):
            cv.line([bx + 6 * cw, bx + 6 * cw + 1.0, bx + 6 * cw + 1.8], [top + ya, top + ya, top + yl], GREY, 0.3)
            cv.text(bx + 6 * cw + 2.1, top + yl, g, size=PT_SMALL, va="center", style="italic")
        fits(bx + 6 * cw + 2.1 + max(text_width(g, PT_SMALL, style="italic") for g in genes), x0 + w, "e labels")
        cv.text(bx + 3 * cw, top + hh + 3.4, "ZT (h)", size=PT_SMALL, ha="center", va="top")
    ky = y0 + h - 3.4
    axc = cv.axes(x0 + 6.2, ky - 0.6, 13, 1.2)
    axc.imshow(np.linspace(-2, 2, 100)[None, :], aspect="auto", cmap=RDBU, extent=(-2, 2, 0, 1))
    axc.set_yticks([])
    axc.set_xticks([-2, 0, 2], ["−2", "0", "2"])
    axc.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in axc.spines.values():
        sp.set_linewidth(0.3)
    cv.text(x0 + 6.2 + 13.6, ky, "z (log₂ CPM)", size=PT_SMALL, va="center")


# ------------------------------------------------------------------------------------------------ f GSE312298
def panel_h(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Sorted marrow neutrophils")
    d = pd.read_csv(R9 / "f4h_gse312298_genes.csv")
    ax = cv.axes(x0 + 8.4, y0 + 7.4, w - 9.4, 37.0)
    x, y = d.predicted_night_minus_day, d.observed_log2FC
    ax.axhline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.4, zorder=0)
    ax.scatter(x, y, s=2.2, color="#7A7A7A", alpha=0.55, linewidths=0, zorder=2, rasterized=True)
    k, c0 = np.polyfit(x, y, 1)                              # display line only
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, k * xx + c0, color=TIS["bm"][1], linewidth=1.0, zorder=3)
    xl, yl = (np.floor(x.min() * 2) / 2, np.ceil(x.max() * 2) / 2), (np.floor(y.min() * 2) / 2, np.ceil(y.max() * 2) / 2)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    assert x.between(*xl).all() and y.between(*yl).all()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Predicted night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Observed night − day (log₂)", fontsize=PT_SMALL, labelpad=0.8)
    r0 = d.iloc[0]
    ax.text(0.03, 0.98, f"{R_SYM} = {r0.pearson_r:.2f}\n{p_label(r0.p_perm, fmt_p)}", transform=ax.transAxes,
            fontsize=PT_SMALL, va="top", ha="left")
    lab = d.assign(s=x * y).sort_values("s", ascending=False).head(4)       # most concordant genes
    axl, axt, axw, axh = x0 + 8.4, y0 + 7.4, w - 9.4, 37.0

    def to_mm(px, py):
        return axl + (px - xl[0]) / (xl[1] - xl[0]) * axw, axt + (yl[1] - py) / (yl[1] - yl[0]) * axh

    pm = np.array([to_mm(a_, b_) for a_, b_ in zip(lab.predicted_night_minus_day, lab.observed_log2FC)])
    srt = np.argsort(pm[:, 1])
    ys = spread(pm[srt, 1] + 1.5, 2.2, axt + 1.0, axt + axh - 1.0)
    for k, yl_ in zip(srt, ys):
        px, py = pm[k]
        cv.line([px, px + 1.6, px + 2.4], [py, yl_, yl_], GREY, 0.3)
        cv.text(px + 2.6, yl_, lab.gene.iloc[k], size=PT_SMALL, va="center", style="italic")


# ------------------------------------------------------------------------------------------------ g baboon
def panel_i(cv, x0, y0, w, h):
    cv.letter(x0, y0, "g", "Baboon spleen, marrow")
    fits(x0 + 3.0 + text_width("Baboon spleen, marrow", PT, "bold"), cv.w, "g title")
    bb = pd.read_csv(R9 / "f4i_baboon_concordance.csv")
    ax = cv.axes(x0 + 8.4, y0 + 7.4, w - 9.4, 33.0)
    ct_rows = bb[bb.scope_type == "cell type"]
    mk = {"SPL": "o", "BOM": "s"}
    for t in ["SPL", "BOM"]:
        d = ct_rows[ct_rows.baboon_tissue == t]
        f = (d.p_perm < 0.05).values
        for sel, z in [(~f, 3), (f, 3.5)]:
            ax.scatter(100 * d.null_frac_within_3h[sel], 100 * d.frac_within_3h[sel], s=8, marker=mk[t],
                       facecolor=d.colour.values[sel] if z > 3 else "white", edgecolor=d.colour.values[sel],
                       linewidths=0.5, zorder=z)
    allr = bb[bb.our_scope == "ALL (Tier 1/2)"].set_index("baboon_tissue")
    for t, lab, dy in [("SPL", "all, spleen", 1.0), ("BOM", "all, marrow", -1.0)]:
        r = allr.loc[t]
        ax.scatter([100 * r.null_frac_within_3h], [100 * r.frac_within_3h], s=22, marker=mk[t], facecolor="black",
                   edgecolor="black", zorder=4)
        ax.annotate(lab, (100 * r.null_frac_within_3h, 100 * r.frac_within_3h), xytext=(4, 3 * dy),
                    textcoords="offset points", fontsize=PT_SMALL, ha="left", va="center")
    lo, hi = 15, 70
    allv = 100 * pd.concat([bb.null_frac_within_3h, bb.frac_within_3h])
    assert allv.between(lo, hi).all(), (allv.min(), allv.max())
    ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.5, linestyle=(0, (2, 1.5)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([20, 40, 60])
    ax.set_yticks([20, 40, 60])
    ax.tick_params(length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(ax)
    ax.set_xlabel("Within 3 h, expected (%)", fontsize=PT_SMALL, labelpad=0.8)
    ax.set_ylabel("Within 3 h, observed (%)", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + h - 2.2, [([{"s": 8, "fc": GREY}], "spleen"),
                                        ([{"s": 8, "marker": "s", "fc": GREY}], "marrow")])
    fits(x, x0 + w, "g key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cv = Canvas()
    panel_timetable(cv, 0.0, 0.0, 64.0, 82.0)                # top: timing (a, b)
    panel_b(cv, 66.0, 0.0, cv.w - 66.0, 82.0)
    panel_flow(cv, 0.0, 84.0, cv.w, 32.0)                    # middle: the pre-specified flow cohort (c)
    panel_e(cv, 0.0, 118.0, 46.0, 52.0)                      # bottom band: marrow programmes (d-g)
    panel_g(cv, 48.0, 118.0, 58.0, 52.0)
    panel_h(cv, 108.0, 118.0, 36.0, 52.0)
    panel_i(cv, 146.0, 118.0, 34.0, 52.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
