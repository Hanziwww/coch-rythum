"""Round 9 (D68 / D69) — Extended Data Fig. 8 (180 x 170 mm): marrow neighbourhoods and the B lineage (supports Fig. 4).

a  Rhythmic Milo neighbourhoods (spatial FDR < 0.1; scANVI embedding): peak ZT of each neighbourhood by its majority
   cell type, marrow and cochlea; right, number of rhythmic neighbourhoods.
b  B-stage share rhythms within the B lineage (circle) and as share of the tissue (square); filled, multi-model verdict
   robust or supported; stages in ImmGen order (D49 bracket names in the legend).
c  The 12 pre-specified B hypotheses (D34): exact permutation p of the primary statistic, BH q across the primary set,
   independent method lines (filled, agrees) and decision (supported needs q < 0.1 and >= 2 agreeing methods).
d  Spearman correlation of our B-stage pseudobulk profiles with ImmGen B populations (microarray, RNA-seq; marrow,
   cochlea); dot, best match; outline, expected population.

Inputs: results/r9_figures/ed8_*.csv, ed2_atlas_columns.csv, palette_celltypes.csv. Output: figures/figures/ed8_b_lineage_v1.*
Run: python scripts/927_ed8_b_lineage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, ld_bar, text_width,  # noqa: E402
                          zt_axis)

from lib.figstyle import P_SYM, Q_SYM, R_SYM  # noqa: E402,F401  (italic P / q / r, D99, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed8_b_lineage_v1"
W_ED = 180.0
TIS = {"bm": ("Marrow", "#D55E00"), "cochlea": ("Cochlea", "#0072B2")}
B_ORDER = ["B_Pro_Cyc", "B_Trans", "B_Pre", "B_Mat", "B_Naive"]
GOOD = {"robust", "supported"}
DECISION = {"supported": ("#1A1A1A", TEXT), "partial": ("#9A9A9A", TEXT), "not supported": ("white", TEXT)}
IMMGEN = ["proB.CLP.BM", "proB.FrA.BM", "proB.FrBC.BM", "preB.FrC.BM", "preB.FrD.BM", "B.FrE.BM", "B.T1.Sp", "B.T2.Sp",
          "B.T3.Sp", "B.FrF.BM", "B.Fo.Sp", "B.MZ.Sp", "B.Fem.Sp"]
REFS = {"ImmGen microarray (GSE15907)": "Microarray", "ImmGen RNA-seq (GSE109125)": "RNA-seq"}
CORR = LinearSegmentedColormap.from_list("corr", ["#F7F7F7", "#BCBDDC", "#756BB1", "#3F007D"])


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def phase_axis(ax):
    zt_axis(ax, ticks=(0, 6, 12, 18, 24))
    ax.spines["bottom"].set_visible(False)
    tick_style(ax)
    ax.tick_params(axis="x", length=0, pad=2.6)


# ------------------------------------------------------------------------------------------------ a Milo
def panel_a(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Rhythmic neighbourhoods")
    m = pd.read_csv(R9 / "ed8_milo_nhoods.csv")
    s = m[m.spatial_fdr < 0.1]
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    top = y0 + 9.4
    bw = (w - 4.0) / 2
    nmax = s.groupby("tissue").abbr.nunique().max()
    rp = (h - (top - y0) - 8.0) / nmax
    rng = np.random.default_rng(11)
    for b, tis in enumerate(TIS):
        bx = x0 + b * (bw + 4.0)
        d = s[s.tissue == tis]
        types = sorted(d.abbr.unique(), key=corder.get)
        ri = {a: i for i, a in enumerate(types)}
        lab_w = max(text_width(a, PT_SMALL) for a in types) + 3.0
        n_w = 7.0
        aw = bw - lab_w - n_w
        ax = cv.axes(bx + lab_w, top, aw, len(types) * rp)
        for i in range(len(types)):
            ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        y = d.abbr.map(ri) + rng.uniform(-0.28, 0.28, len(d))
        ax.scatter(d.peak_zt % 24, y, s=3.0, c=d.abbr.map(colour), linewidths=0, alpha=0.85, zorder=2)
        ax.set_ylim(len(types) - 0.5, -0.5)
        ax.set_yticks([])
        clean(ax, left=False)
        phase_axis(ax)
        ld_bar(cv, bx + lab_w, top + len(types) * rp + 0.1, aw, 0.7)
        cnt = d.abbr.value_counts()
        for a in types:
            yy = top + (ri[a] + 0.5) * rp
            cv.box(bx + lab_w - 1.9, yy - rp * 0.34, 1.2, rp * 0.68, colour[a])
            cv.text(bx + lab_w - 2.4, yy, a, size=PT_SMALL, ha="right", va="center")
            cv.text(bx + bw - 0.2, yy, f"{cnt[a]}", size=PT_SMALL, ha="right", va="center", color=TEXT2)
        tot = m[m.tissue == tis]
        cv.text(bx, top - 2.6, f"{TIS[tis][0]} ({len(d)} of {len(tot):,})", size=PT_SMALL, va="bottom",
                fontweight="bold", color=TEXT)
        cv.text(bx + bw - 0.2, top - 2.6, "n", size=PT_SMALL, ha="right", va="bottom", color=TEXT2)


# ------------------------------------------------------------------------------------------------ b B stages
def panel_b(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "b", "B-stage share rhythms")
    d = pd.read_csv(R9 / "ed8_b_stage_peaks.csv")
    colour = dict(zip(cols.abbr, cols.colour))
    lab_w = max(text_width(a, PT_SMALL) for a in B_ORDER) + 3.0
    top = y0 + 8.0
    gap = 6.6
    fh = (h - (top - y0) - gap - 9.0) / 2
    ri = {a: i for i, a in enumerate(B_ORDER)}
    for k, tis in enumerate(TIS):
        ay = top + k * (fh + gap)
        ax = cv.axes(x0 + lab_w, ay, w - lab_w - 2.2, fh)
        for i in range(len(B_ORDER)):
            ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        s = d[d.tissue == tis]
        for frame, mk, off in (("within_B", "o", -0.17), ("share_of_tissue", "s", 0.17)):
            f = s[s.frame == frame]
            good = f.verdict.isin(GOOD).to_numpy()
            y = f.abbr.map(ri).to_numpy() + off
            cc = f.abbr.map(colour).to_numpy()
            ax.scatter(f.peak_zt[good] % 24, y[good], marker=mk, s=12, c=cc[good], edgecolors=TEXT, linewidths=0.4,
                       zorder=3)
            ax.scatter(f.peak_zt[~good] % 24, y[~good], marker=mk, s=10, facecolors="white", edgecolors=cc[~good],
                       linewidths=0.7, zorder=3)
        ax.set_ylim(len(B_ORDER) - 0.5, -0.5)
        ax.set_yticks([])
        clean(ax, left=False)
        phase_axis(ax)
        ld_bar(cv, x0 + lab_w, ay + fh + 0.1, w - lab_w - 2.2, 0.7)
        for a in B_ORDER:
            yy = ay + (ri[a] + 0.5) * fh / len(B_ORDER)
            cv.box(x0 + lab_w - 1.9, yy - 0.7, 1.2, 1.4, colour[a])
            cv.text(x0 + lab_w - 2.4, yy, a, size=PT_SMALL, ha="right", va="center")
        cv.text(x0 + lab_w, ay - 0.6, TIS[tis][0], size=PT_SMALL, va="bottom", fontweight="bold", color=TEXT)
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, [([{"s": 12, "fc": TEXT2, "ec": TEXT, "lw": 0.4}], "within B"),
                                         ([{"s": 12, "marker": "s", "fc": TEXT2, "ec": TEXT, "lw": 0.4}], "of tissue"),
                                         ([{"s": 10, "fc": "white", "ec": TEXT2, "lw": 0.7}], "not robust")], gap=1.6)
    fits(xx, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c scorecard
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Pre-specified B hypotheses")
    sc = pd.read_csv(R9 / "ed8_b_scorecard.csv")
    mm = pd.read_csv(R9 / "ed8_b_methods.csv")
    sc["label"] = sc.hypothesis + " · " + sc.tissue.map({t: v[0].lower() for t, v in TIS.items()})
    lab_w = max(text_width(t, PT_SMALL) for t in sc.label) + 1.4
    top = y0 + 8.0
    rp = (h - (top - y0) - 7.0) / len(sc)
    cw_p, cw_q, cw_m, cw_d = 22.0, 15.0, 16.0, 14.0
    xp = x0 + lab_w
    xq = xp + cw_p + 4.4
    xm = xq + cw_q + 2.0
    xd = xm + cw_m + 1.0
    fits(xd + cw_d, x0 + w, "c columns")
    ny = len(sc) * rp
    axp = cv.axes(xp, top, cw_p, ny)
    axq = cv.axes(xq, top, cw_q, ny)
    for ax, col, lo in ((axp, "primary_p", 0.01), (axq, "q_BH_primary_set", 0.01)):
        v = sc[col].clip(lower=lo)
        ax.scatter(-np.log10(v), np.arange(len(sc)), s=8, c=TEXT, linewidths=0, zorder=3)
        ax.axvline(-np.log10(0.05 if col == "primary_p" else 0.1), color="#E69F00", lw=0.6, zorder=1)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 1, 2], ["1", "0.1", "0.01"])
        ax.set_ylim(len(sc) - 0.5, -0.5)
        for i in range(len(sc)):
            ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        ax.set_yticks([])
        clean(ax, left=False)
        tick_style(ax)
    cv.text(xp, top - 0.8, f"Exact {P_SYM}", size=PT_SMALL, va="bottom", color=TEXT2)
    cv.text(xq, top - 0.8, Q_SYM, size=PT_SMALL, va="bottom", color=TEXT2)
    cv.text(xm, top - 0.8, "Methods", size=PT_SMALL, va="bottom", color=TEXT2)
    cv.text(xd, top - 0.8, "Decision", size=PT_SMALL, va="bottom", color=TEXT2)
    for i, r in sc.iterrows():
        yy = top + (i + 0.5) * rp
        cv.text(x0 + lab_w - 0.8, yy, r.label, size=PT_SMALL, ha="right", va="center")
        lines = mm[(mm.hypothesis == r.hypothesis) & (mm.tissue == r.tissue)]
        assert len(lines) == r.n_methods, r.label
        for j, ag in enumerate(lines.agrees.astype(bool)):
            cv.mark(xm + 1.4 + j * 3.2, yy, 9, fc=TEXT if ag else "white", ec=TEXT, lw=0.5)
        fc, ec = DECISION[r.decision]
        cv.box(xd, yy - 1.0, 2.0, 2.0, fc, ec, 0.4)
        cv.text(xd + 2.8, yy, r.decision, size=PT_SMALL, va="center")
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, [([{"s": 9, "fc": TEXT}], "agrees"),
                                         ([{"s": 9, "fc": "white", "ec": TEXT, "lw": 0.5}], "does not"),
                                         ([{"kind": "vline", "colour": "#E69F00", "lw": 0.6, "h": 2.0}],
                                          f"{P_SYM} 0.05 / {Q_SYM} 0.1")], gap=1.8)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d ImmGen
def panel_d(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "d", "ImmGen correspondence")
    im = pd.read_csv(R9 / "ed8_immgen.csv")
    name2abbr = dict(zip(cols.cell_type, cols.abbr))
    im["abbr"] = im.our_stage.map(name2abbr)
    assert im.abbr.notna().all() and set(im.immgen_population) <= set(IMMGEN)
    short = {p: p.split(".")[1] for p in IMMGEN}
    lab_w = max(text_width(a, PT_SMALL) for a in B_ORDER) + 1.2
    top = y0 + 4.6 + max(text_width(v, PT_SMALL) for v in short.values()) + 1.0
    gx, gy = 2.4, 5.4
    mw = (w - lab_w - gx - 0.5) / 2
    q = mw / len(IMMGEN)
    mh = (h - (top - y0) - gy - 14.0) / 2
    rp = mh / len(B_ORDER)
    lo, hi = 0.2, 0.85
    for i, (ref, rlab) in enumerate(REFS.items()):
        for j, tis in enumerate(TIS):
            mx, my = x0 + lab_w + j * (mw + gx), top + i * (mh + gy)
            ax = cv.axes(mx, my, mw, mh)
            d = im[(im.reference == ref) & (im.tissue == tis)]
            mat = np.full((len(B_ORDER), len(IMMGEN)), np.nan)
            for _, r in d.iterrows():
                mat[B_ORDER.index(r.abbr), IMMGEN.index(r.immgen_population)] = r.spearman_r
            img = CORR(np.clip((np.nan_to_num(mat, nan=lo) - lo) / (hi - lo), 0, 1))
            img[np.isnan(mat)] = (1, 1, 1, 0)
            ax.imshow(img, aspect="auto", interpolation="nearest", extent=(-0.5, len(IMMGEN) - 0.5, len(B_ORDER) - 0.5,
                                                                            -0.5))
            for _, r in d.iterrows():
                xi, yi = IMMGEN.index(r.immgen_population), B_ORDER.index(r.abbr)
                if r.expected:
                    ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, ec=TEXT, lw=0.5))
                if r.is_best:
                    ax.scatter([xi], [yi], s=4, c="white", edgecolors=TEXT, linewidths=0.4, zorder=3)
            ax.set_xlim(-0.5, len(IMMGEN) - 0.5)
            ax.set_ylim(len(B_ORDER) - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.35)
            if i == 0:
                for k, p in enumerate(IMMGEN):
                    cv.text(mx + (k + 0.5) * q, top - 0.6, short[p], size=PT_SMALL, ha="center", va="bottom",
                            rotation=90)
                cv.text(mx + mw / 2, y0 + 4.4, TIS[tis][0], size=PT_SMALL, ha="center", va="bottom",
                        fontweight="bold", color=TEXT)
            if j == 0:
                for a in B_ORDER:
                    cv.text(mx - 0.8, my + (B_ORDER.index(a) + 0.5) * rp, a, size=PT_SMALL, ha="right", va="center")
            cv.text(mx + mw, my + mh + 0.4, rlab, size=PT_SMALL, ha="right", va="top", color=TEXT2)
    ky = y0 + h - 4.4
    axk = cv.axes(x0 + 8.0, ky - 0.6, 12.0, 1.2)
    axk.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=CORR, extent=(lo, hi, 0, 1))
    axk.set_yticks([])
    axk.set_xticks([lo, hi], [f"{lo:g}", f"{hi:g}"])
    tick_style(axk)
    cv.text(x0 + 7.4, ky, "ρ", size=PT_SMALL, ha="right", va="center")
    xx = cv.key(x0 + 23.0, ky, [([{"s": 4, "fc": "white", "ec": TEXT, "lw": 0.4}], "best"),
                                ([{"kind": "box", "fc": "white", "ec": TEXT, "lw": 0.5, "size": 1.6}], "expected")],
                gap=1.6)
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED)
    panel_a(cv, cols, 0.0, 0.0, 116.0, 76.0)
    panel_b(cv, cols, 120.0, 0.0, 60.0, 76.0)
    panel_c(cv, 0.0, 80.0, 112.0, 90.0)
    panel_d(cv, cols, 116.0, 80.0, 64.0, 90.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
