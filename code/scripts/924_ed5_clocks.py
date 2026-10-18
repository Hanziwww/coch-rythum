"""Round 9 (D68 / D69) — Extended Data Fig. 5 (180 x 214 mm since round 10; a-f 180 x 170 mm as in v1): clock detail
(supports Fig. 2).

a  Every clock and systemic-cue gene in every primary cell type of both tissues: colour = peak ZT (cyclic), area =
   amplitude, outlined and opaque = permutation p <= 0.05; genes grouped by loop.
b  Raw clock time courses (6 pooled samples, relative to the mesor) with the cosinor fit in three cochlear structural
   types and marrow classical monocytes.
c  Peak of each core-clock gene: tissue pseudobulk (filled = p < 0.05), circular mean across significant cell types
   (diamond, +/- circular s.d.) and the organ-atlas reference (grey tick).
d  Clock time-telling: offset of our samples' clock phase predicted by external clock references, per class
   (area = resultant); diamond = all cells.
e  Clock network per cell type: correlation-structure distance to the reference (CCD z vs null) and circular
   correlation of the peak order with the canonical order.
f  Clock phase offset of the same cell type in cochlea and marrow (23 shared types).
g  Round 10 R1 (333; D115): single-cell clock tracking per cochlear cell type at its own depth, at the clock-UMI depth
   of the cochlear haematopoietic cells (20 thinnings; with and without equal cell numbers) and among cells with 3-5
   clock UMIs (dot area, share of replicates with exact p <= 0.05; x, untestable).

Inputs: results/r9_figures/ed5_*.csv, f2b_clock_timecourses.csv, f2f_timetelling.csv, palettes. Output: figures/figures/ed5_clocks_v1.*
Run: python scripts/924_ed5_clocks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, header_height, ld_bar,  # noqa: E402
                          text_width, type_header, zt_axis)

from lib.figstyle import P_SYM, R_SYM  # noqa: E402,F401  (italic P / r, D99)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed5_clocks_v2"                      # v2 (round 10, D112 / D115): panel g added below a-f; a-f unchanged
W_ED, H_ED = 180.0, 214.0
DEPTH_ROWS = [("own depth (332)", "Own depth"), ("depth matched", "Haematopoietic depth"),
              ("depth and cell number matched", "+ equal cell numbers"), ("natural stratum 3-5 clock UMIs", "3–5 clock UMIs")]
ARMS = [("E-box activators", ["Npas2", "Clock"]), ("RORE loop", ["Nr1d1", "Nr1d2", "Rora", "Rorc"]),
        ("Per / Cry repressors", ["Per1", "Per2", "Per3", "Cry1", "Cry2"]), ("D-box", ["Dbp", "Tef", "Hlf", "Nfil3"]),
        ("Modifiers", ["Ciart", "Bhlhe40", "Bhlhe41"]), ("Systemic cues", ["Cirbp", "Fkbp5", "Zbtb16", "Sgk1", "Klf9", "Dusp1"])]
# muted cyclic phase scale: dawn gold -> midday orange -> dusk purple -> midnight navy -> dawn (no rainbow, no red/green)
PHASE = LinearSegmentedColormap.from_list("phase", [(0.0, "#E9C46A"), (0.25, "#D9822B"), (0.5, "#8E5B9E"),
                                                    (0.75, "#2E4A7D"), (1.0, "#E9C46A")])
TIS = {"cochlea": "Cochlea", "bm": "Marrow"}
REF_GREY = "#9A9A9A"


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def amp_size(a):
    return 0.6 + 18.0 * np.clip(np.asarray(a, float), 0, 1.5) / 1.5


# ------------------------------------------------------------------------------------------------ a clock matrix
def panel_a(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Clock genes in every cell type")
    m = pd.read_csv(R9 / "ed5_clock_matrix.csv")
    genes = [g for _, gs in ARMS for g in gs]
    assert set(genes) == set(m.gene), set(m.gene) ^ set(genes)
    xpos, x = {}, 0.0
    for k, (_, gs) in enumerate(ARMS):
        if k:
            x += 0.7
        for g in gs:
            xpos[g] = x
            x += 1.0
    ncol = x
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    top = y0 + 6.4 + max(text_width(g, PT_SMALL, style="italic") for g in genes) + 1.2
    bw = (w - 3.0) / 2
    nmax = max(m[m.tissue == t].abbr.nunique() for t in TIS)
    rp = (h - (top - y0) - 13.0) / nmax
    for b, tis in enumerate(TIS):
        bx = x0 + b * (bw + 3.0)
        d = m[m.tissue == tis]
        types = sorted(d.abbr.unique(), key=corder.get)
        lab_w = max(text_width(a, PT_SMALL) for a in types) + 3.0
        q = (bw - lab_w - 0.5) / ncol
        ax = cv.axes(bx + lab_w, top, ncol * q, len(types) * rp)
        ri = {a: i for i, a in enumerate(types)}
        e = d[d.expressed]
        sig = e.p_perm <= 0.05
        for flag, alpha, ec in ((False, 0.35, "none"), (True, 1.0, TEXT)):
            s = e[sig == flag]
            ax.scatter(s.gene.map(xpos), s.abbr.map(ri), s=amp_size(s.amp_log2), c=PHASE((s.peak_zt % 24) / 24),
                       alpha=alpha, edgecolors=ec, linewidths=0.3, zorder=3 if flag else 2)
        ax.set_xlim(-0.6, ncol - 0.4)
        ax.set_ylim(len(types) - 0.5, -0.5)
        ax.set_axis_off()
        for i, a in enumerate(types):
            yy = top + (i + 0.5) * rp
            cv.box(bx + lab_w - 1.9, yy - rp * 0.36, 1.2, rp * 0.72, colour[a])
            cv.text(bx + lab_w - 2.4, yy, a, size=PT_SMALL, ha="right", va="center")
        for g in genes:
            cv.text(bx + lab_w + (xpos[g] + 0.6) * q, top - 0.6, g, size=PT_SMALL, ha="center", va="bottom",
                    rotation=90, style="italic")
        cv.text(bx, y0 + 5.6, TIS[tis], size=PT_SMALL, va="bottom", fontweight="bold")
    # keys: cyclic phase bar with LD bar, amplitude, significance
    ky = y0 + h - 7.6
    kx = x0 + 9.0
    axc = cv.axes(kx, ky - 0.7, 20.0, 1.4)
    axc.imshow(np.linspace(0, 1, 240)[None, :], aspect="auto", cmap=PHASE, extent=(0, 24, 0, 1))
    axc.set_yticks([])
    axc.set_xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
    tick_style(axc)
    ld_bar(cv, kx, ky + 3.4, 20.0, 0.7)
    cv.text(kx - 1.0, ky, "Peak ZT", size=PT_SMALL, ha="right", va="center")
    xx = kx + 24.0
    cv.text(xx, ky, "Amplitude", size=PT_SMALL, va="center")
    xx += text_width("Amplitude", PT_SMALL) + 1.4
    xx = cv.key(xx, ky, [([{"s": amp_size(a), "fc": TEXT2}], f"{a:g}") for a in (0.25, 0.75, 1.5)], gap=1.4)
    xx = cv.key(xx + 3.0, ky, [([{"s": 9, "fc": PHASE(0.5), "ec": TEXT, "lw": 0.3}], f"{P_SYM} ≤ 0.05")])
    fits(xx, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b time courses
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Clock time courses")
    d = pd.read_csv(R9 / "f2b_clock_timecourses.csv")
    facets = d.drop_duplicates("facet_label").sort_values(["tissue_order", "order"]).facet_label.tolist()
    genes = d.drop_duplicates("gene").sort_values("series_order").gene.tolist()
    gcol = dict(zip(d.gene, d.gene_colour))
    top = y0 + 5.0
    gx, gy = 3.4, 6.6
    fw = (w - 8.0 - gx - 2.0) / 2
    fh = (h - (top - y0) - 6.0 - gy) / 2
    t = np.linspace(0, 24, 200)
    for k, fl in enumerate(facets):
        r, c = divmod(k, 2)
        fx, fy = x0 + 8.0 + c * (fw + gx), top + r * (fh + gy)
        ax = cv.axes(fx, fy + 2.4, fw, fh - 2.4)
        sub = d[d.facet_label == fl]
        for g in genes:
            s = sub[sub.gene == g].sort_values("zt")
            a, pk = s.amplitude.iloc[0], s.peak_zt.iloc[0]
            ax.plot(t, a * np.cos(2 * np.pi * (t - pk) / 24), color=gcol[g], lw=0.7, zorder=2)
            filled = s.p_perm.iloc[0] <= 0.05
            ax.scatter(s.zt, s.y_rel, s=6, facecolors=gcol[g] if filled else "white", edgecolors=gcol[g],
                       linewidths=0.5, zorder=3)
        ax.axhline(0, color=GRID, lw=0.35, zorder=0)
        ax.set_ylim(-2.6, 2.6)
        ax.set_yticks([-2, 0, 2], ["−2", "0", "2"])
        zt_axis(ax, ticks=(0, 12, 24))
        clean(ax)
        ax.spines["bottom"].set_visible(False)
        tick_style(ax)
        ax.tick_params(axis="x", length=0, pad=2.6)
        if c == 0:
            ax.set_ylabel("log2 vs mesor", fontsize=PT_SMALL, labelpad=0.8)
        else:
            ax.set_yticklabels([])
        ld_bar(cv, fx, fy + fh + 0.1, fw, 0.7)
        cv.text(fx + fw / 2, fy + 1.6, fl, size=PT_SMALL, ha="center", va="bottom")
    ky = y0 + h - 1.4
    xx = x0 + 8.0
    for g in genes:
        cv.line([xx, xx + 2.4], [ky, ky], gcol[g], 0.7)
        cv.mark(xx + 1.2, ky, 6, fc=gcol[g])
        cv.text(xx + 3.0, ky, g, size=PT_SMALL, va="center", style="italic")
        xx += 3.0 + text_width(g, PT_SMALL, style="italic") + 2.2
    fits(xx - 2.2, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c phase order
def wrap_segments(lo, hi):
    """Segments of [lo, hi] on a 0-24 axis, split at the wrap."""
    lo, hi = lo % 24, hi % 24
    return [(lo, hi)] if lo <= hi else [(lo, 24.0), (0.0, hi)]


def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Peak order of the core clock")
    o = pd.read_csv(R9 / "ed5_tissue_clock_order.csv")
    e = pd.read_csv(R9 / "ed5_phase_equality.csv")
    ref = o.drop_duplicates("gene").set_index("gene").reference_peak_zt.sort_values()
    genes = ref.index.tolist()
    lab_w = max(text_width(g, PT_SMALL, style="italic") for g in genes) + 1.2
    top = y0 + 8.0
    aw = (w - lab_w - 2.0) / 2
    rp = (h - (top - y0) - 12.0) / len(genes)
    ri = {g: i for i, g in enumerate(genes)}
    for b, tis in enumerate(TIS):
        ax = cv.axes(x0 + lab_w + b * (aw + 2.0), top, aw, len(genes) * rp)
        for g in genes:
            ax.axhline(ri[g], color=GRID, lw=0.25, zorder=0)
            ax.plot([ref[g]] * 2, [ri[g] - 0.36, ri[g] + 0.36], color=REF_GREY, lw=1.1, zorder=1,
                    solid_capstyle="butt")
        ee = e[e.tissue == tis]
        for _, r in ee.iterrows():
            for a_, b_ in wrap_segments(r.circ_mean_peak_zt - r.circ_sd_h, r.circ_mean_peak_zt + r.circ_sd_h):
                ax.plot([a_, b_], [ri[r.gene] + 0.22] * 2, color=TEXT2, lw=0.5, zorder=2)
        ax.scatter(ee.circ_mean_peak_zt % 24, ee.gene.map(ri) + 0.22, marker="D", s=5, facecolors="white",
                   edgecolors=TEXT, linewidths=0.45, zorder=3)
        oo = o[o.tissue == tis]
        sg = oo.significant.astype(bool)
        ax.scatter(oo.our_peak_zt[sg] % 24, oo.gene[sg].map(ri) - 0.18, s=7, c=TEXT, linewidths=0, zorder=4)
        ax.scatter(oo.our_peak_zt[~sg] % 24, oo.gene[~sg].map(ri) - 0.18, s=7, facecolors="white", edgecolors=TEXT,
                   linewidths=0.45, zorder=4)
        ax.set_ylim(len(genes) - 0.5, -0.5)
        ax.set_yticks([])
        zt_axis(ax, ticks=(0, 12, 24))
        clean(ax, left=False)
        ax.spines["bottom"].set_visible(False)
        tick_style(ax)
        ax.tick_params(axis="x", length=0, pad=2.6)
        ld_bar(cv, x0 + lab_w + b * (aw + 2.0), top + len(genes) * rp + 0.1, aw, 0.7)
        cv.text(x0 + lab_w + b * (aw + 2.0) + aw / 2, top - 0.8, TIS[tis], size=PT_SMALL, ha="center",
                va="bottom", fontweight="bold")
    for g in genes:
        cv.text(x0 + lab_w - 0.8, top + (ri[g] + 0.5) * rp, g, size=PT_SMALL, ha="right", va="center",
                style="italic")
    cv.text(x0 + lab_w + aw + 1.0, top + len(genes) * rp + 4.4, "ZT", size=PT_SMALL, ha="center", va="top")
    ky = y0 + h - 1.4
    items = [([{"s": 7, "fc": TEXT}], "tissue"), ([{"s": 5, "marker": "D", "fc": "white", "ec": TEXT, "lw": 0.45}],
                                                  "across types"),
             ([{"kind": "vline", "colour": REF_GREY, "lw": 1.1, "h": 1.8}], "organs")]
    xx = cv.key(x0 + 1.0, ky, items, gap=1.8)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d time-telling
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "External clocks tell our time")
    f = pd.read_csv(R9 / "f2f_timetelling.csv")
    ccol = pd.read_csv(R9 / "palette_classes.csv")
    ccol = dict(zip(ccol.iloc[:, 0], ccol.loc[:, [c for c in ccol.columns if "colour" in c][0]]))
    refs = f.drop_duplicates("ref_short").ref_short.tolist()
    rows = [(r, t) for r in refs for t in ("cochlea", "bm")]
    lab_w = max(text_width(TIS[t], PT_SMALL) for t in TIS) + 1.4
    top = y0 + 5.6
    hdr = 3.2
    rp = (h - (top - y0) - len(refs) * hdr - 10.0) / len(rows)
    ax_w = w - lab_w - 1.0
    ys, y = {}, 0.0
    for i, (r, t) in enumerate(rows):
        if t == "cochlea":
            cv.text(x0, top + y + hdr - 0.6, r, size=PT_SMALL, va="bottom", color=TEXT2)
            y += hdr
        ys[(r, t)] = top + y + rp / 2
        y += rp
    ax = cv.axes(x0 + lab_w, top, ax_w, y)
    ax.set_ylim(top + y, top)
    ax.set_xlim(-8, 8)
    ax.axvline(0, color=TEXT2, lw=0.35, zorder=0)
    for (r, t), yy in ys.items():
        ax.axhline(yy, color=GRID, lw=0.25, zorder=0)
        s = f[(f.ref_short == r) & (f.tissue == t)]
        cl = s[s.target != "all_cells"]
        ax.scatter(cl.circ_mean_offset_h, [yy] * len(cl), s=1.0 + 16.0 * cl.offset_resultant,
                   c=[ccol[c] for c in cl.target_class], linewidths=0.3, edgecolors="white", zorder=3)
        al = s[s.target == "all_cells"]
        ax.scatter(al.circ_mean_offset_h, [yy] * len(al), marker="D", s=9, facecolors="white", edgecolors=TEXT,
                   linewidths=0.6, zorder=4)
        cv.text(x0 + lab_w - 0.8, yy, TIS[t], size=PT_SMALL, ha="right", va="center")
    ax.set_xticks([-8, -4, 0, 4, 8], ["−8", "−4", "0", "4", "8"])
    ax.set_yticks([])
    clean(ax, left=False)
    tick_style(ax)
    ax.set_xlabel("Offset (h)", fontsize=PT_SMALL, labelpad=1.0)
    ky = y0 + h - 1.4
    xx = cv.key(x0 + lab_w, ky, [([{"s": 1.0 + 16.0 * a, "fc": TEXT2}], f"{a:g}") for a in (0.25, 0.5, 1.0)],
                gap=1.4)
    xx = cv.key(xx + 2.4, ky, [([{"s": 9, "marker": "D", "fc": "white", "ec": TEXT, "lw": 0.6}], "all cells")])
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e clock network
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Clock network per cell type")
    n = pd.read_csv(R9 / "ed5_clock_network.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr").colour
    ax = cv.axes(x0 + 9.0, y0 + 5.6, w - 12.0, h - 5.6 - 10.0)
    ax.axhline(0, color=GRID, lw=0.4, zorder=0)
    ax.axvline(0, color=GRID, lw=0.4, zorder=0)
    for tis, (fc_on, mk) in {"cochlea": (True, "o"), "bm": (False, "o")}.items():
        s = n[n.tissue == tis]
        cc = s.abbr.map(pal)
        ax.scatter(s.circ_corr_observed_vs_canonical, s.z, s=10, marker=mk,
                   facecolors=cc if fc_on else "white", edgecolors=cc, linewidths=0.7, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1], ["−1", "0", "1"])
    lo = np.floor(n.z.min()) - 0.5
    hi = np.ceil(n.z.max()) + 0.5
    ax.set_ylim(hi, lo)
    clean(ax)
    tick_style(ax)
    ax.set_xlabel("Canonical-order correlation", fontsize=PT_SMALL, labelpad=1.0)
    ax.set_ylabel("CCD z", fontsize=PT_SMALL, labelpad=1.0)
    xx = cv.key(x0 + 9.0, y0 + h - 1.4, [([{"s": 10, "fc": TEXT2}], "cochlea"),
                                         ([{"s": 10, "fc": "white", "ec": TEXT2, "lw": 0.7}], "marrow")], gap=2.0)
    fits(xx, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ f cross-tissue
def panel_f(cv, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Same type, two tissues")
    x = pd.read_csv(R9 / "ed5_crosstissue_offsets.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr").colour
    ax = cv.axes(x0 + 9.0, y0 + 5.6, w - 12.0, h - 5.6 - 10.0)
    lo, hi = -12, 12
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=0.6, zorder=0)
    assert x.clock_phase_offset_h_bm.between(lo, hi).all() and x.clock_phase_offset_h_cochlea.between(lo, hi).all()
    good = np.minimum(x.phase_coherence_bm, x.phase_coherence_cochlea) >= 0.5
    cc = x.abbr.map(pal)
    ax.scatter(x.clock_phase_offset_h_bm[good], x.clock_phase_offset_h_cochlea[good], s=10, c=cc[good],
               linewidths=0.3, edgecolors="white", zorder=3)
    ax.scatter(x.clock_phase_offset_h_bm[~good], x.clock_phase_offset_h_cochlea[~good], s=10, facecolors="white",
               edgecolors=cc[~good], linewidths=0.7, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    tk = [-12, -6, 0, 6, 12]
    ax.set_xticks(tk, [f"{t:g}".replace("-", "−") for t in tk])
    ax.set_yticks(tk, [f"{t:g}".replace("-", "−") for t in tk])
    clean(ax)
    tick_style(ax)
    ax.set_xlabel("Marrow offset (h)", fontsize=PT_SMALL, labelpad=1.0)
    ax.set_ylabel("Cochlea offset (h)", fontsize=PT_SMALL, labelpad=1.0)
    xx = cv.key(x0 + 9.0, y0 + h - 1.4, [([{"s": 10, "fc": TEXT2}], "coherence ≥ 0.5 in both"),
                                         ([{"s": 10, "fc": "white", "ec": TEXT2, "lw": 0.7}], "lower")], gap=2.0)
    fits(xx, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ g depth-matched clocks
def panel_g(cv, x0, y0, w, h):
    """Round 10 R1 (333): share of replicates in which single-cell clock phase tracks ZT (exact p <= 0.05), per cochlear
    cell type, at its own depth, at haematopoietic clock-UMI depth (with and without equal cell numbers) and among cells
    with 3-5 clock UMIs; x, untestable (< 300 cells keep >= 3 clock UMIs); dash, not applicable."""
    cv.letter(x0, y0, "g", "Single-cell clocks at matched depth")
    d = pd.read_csv(R9 / "ed5_clock_depth_matched.csv")
    cols = d.drop_duplicates("cell_type").sort_values(["group", "order"], key=lambda s: s.map(
        {"structural": 0, "haematopoietic": 1}) if s.name == "group" else s)
    abbrs, colours = cols.abbr.tolist(), cols.colour.tolist()
    assert len(abbrs) == 22
    lab_w = max(text_width(lbl, PT_SMALL) for _, lbl in DEPTH_ROWS) + 1.5
    key_w = 30.0
    xm = x0 + lab_w
    q = (w - lab_w - key_w) / len(abbrs)
    top = y0 + header_height(abbrs)
    rp = 3.4
    type_header(cv, abbrs, colours, xm, top, q)
    n_struct = int((cols.group == "structural").sum())
    cv.line([xm + n_struct * q, xm + n_struct * q], [top - 1.0, top + len(DEPTH_ROWS) * rp], GRID, 0.5)
    vals = d.set_index(["cell_type", "variant"])
    smax = 14.0
    for i, (variant, label) in enumerate(DEPTH_ROWS):
        yy = top + (i + 0.5) * rp
        cv.text(xm - 1.0, yy, label, size=PT_SMALL, ha="right", va="center")
        for j, ct in enumerate(cols.cell_type):
            xx = xm + (j + 0.5) * q
            if (ct, variant) not in vals.index:
                cv.text(xx, yy, "–", size=PT_SMALL, ha="center", va="center", color=TEXT2)
                continue
            r = vals.loc[(ct, variant)]
            if r["n_testable"] == 0:
                cv.text(xx, yy, "×", size=PT_SMALL, ha="center", va="center", color=TEXT2)
            elif r["frac_tracking"] > 0:
                cv.mark(xx, yy, s=max(1.5, smax * r["frac_tracking"]), fc=TEXT)
            else:
                cv.mark(xx, yy, s=4.0, fc="white", ec=TEXT2, lw=0.5)
    xk = x0 + w - key_w + 2.0
    yk = top + 0.6 * rp
    for k, (glyph, label) in enumerate([([{"s": smax, "fc": TEXT}], "tracks in all replicates"),
                                       ([{"s": smax * 0.5, "fc": TEXT}], "in half"),
                                       ([{"s": 4.0, "fc": "white", "ec": TEXT2, "lw": 0.5}], "does not track")]):
        xx = cv.key(xk, yk + k * rp, [(glyph, label)], gap=1.0)
        fits(xx, x0 + w, "g key")
    xx = cv.key(xk, yk + 3 * rp, [([], "× untestable, – not applicable")], gap=1.0)
    fits(xx, x0 + w, "g key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED, height=H_ED)
    panel_a(cv, cols, 0.0, 0.0, 116.0, 92.0)
    panel_b(cv, 119.0, 0.0, 61.0, 92.0)
    panel_c(cv, 0.0, 96.0, 58.0, 74.0)
    panel_d(cv, 61.0, 96.0, 56.0, 74.0)
    panel_e(cv, 120.0, 96.0, 60.0, 37.0)
    panel_f(cv, 120.0, 135.0, 60.0, 35.0)
    panel_g(cv, 0.0, 174.0, 180.0, 40.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
