"""Round 9 (D68 / D69) — Extended Data Fig. 1 (180 x 170 mm): sampling and quality control (supports Fig. 1).

a  Per-sample QC distributions (box q25-q75, whiskers q05-q95, median): UMIs, genes, mitochondrial %, ribosomal %,
   doublet score; cochlea and marrow, six times each.
b  Cells per cell type and sample (log10), rows in atlas order.
c  Label support per cell type: median margin of the own-label score over the best alternative (cochlea, marrow).
d  Time intermixing per cell type: share of same-type kNN neighbours from other samples, observed / expected.
e  The joint embedding split by tissue (cells of the tissue coloured by subtype over all cells in grey).

Inputs: results/r9_figures/ed1_*.csv, f1b_umap_joint.csv, palette tables. Output: figures/figures/ed1_qc_v1.*
Run: python scripts/920_ed1_qc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import AXIS, PT, PT_SMALL, TEXT, TEXT2, Canvas, blank, clean, fits, text_width  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed1_qc_v2"
W_ED = 180.0
TIS = {"cochlea": ("Cochlea", fd.TISSUE_COLOUR["cochlea"]), "bm": ("Marrow", fd.TISSUE_COLOUR["bm"])}
ZTS = [3, 7, 11, 15, 19, 23]
METRICS = [("total_counts", "UMIs", True), ("n_genes_by_counts", "Genes", True), ("pct_counts_mt", "Mito. (%)", False),
           ("pct_counts_ribo", "Ribo. (%)", False), ("doublet_score", "Doublet score", False)]
GREYS = LinearSegmentedColormap.from_list("g", ["#F2F2F2", "#BDBDBD", "#6E6E6E", "#1A1A1A"])
GREY = "#777777"


# ------------------------------------------------------------------------------------------------ a QC
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Per-sample quality")
    q = pd.read_csv(R9 / "ed1_sample_qc_quantiles.csv")
    gap, lab = 3.2, 7.6
    mw = (w - len(METRICS) * (lab + gap) + gap) / len(METRICS)
    top, ah = y0 + 7.4, h - 7.4 - 9.0
    for j, (m, title, logy) in enumerate(METRICS):
        bx = x0 + lab + j * (mw + lab + gap)
        ax = cv.axes(bx, top, mw, ah)
        d = q[q.metric == m]
        xs = []
        for k, t in enumerate(["cochlea", "bm"]):
            for i, z in enumerate(ZTS):
                r = d[(d.tissue == t) & (d.zt == z)]
                assert len(r) == 1, (m, t, z)
                r = r.iloc[0]
                xp = k * 7 + i
                xs.append(xp)
                col = TIS[t][1]
                ax.plot([xp, xp], [r.q05, r.q25], color=col, linewidth=0.5)
                ax.plot([xp, xp], [r.q75, r.q95], color=col, linewidth=0.5)
                ax.add_patch(__import__("matplotlib").patches.Rectangle((xp - 0.34, r.q25), 0.68, r.q75 - r.q25,
                                                                         facecolor=col, alpha=0.35, edgecolor=col,
                                                                         linewidth=0.4))
                ax.plot([xp - 0.34, xp + 0.34], [r.q50] * 2, color=col, linewidth=0.9)
        if logy:
            ax.set_yscale("log")
            tk = [t_ for t_ in (300, 1000, 3000, 10000, 30000) if d.q05.min() * 0.8 <= t_ <= d.q95.max() * 1.25]
            ax.set_yticks(tk, [f"{t_ // 1000}k" if t_ >= 1000 else str(t_) for t_ in tk])
        ax.set_xlim(-0.8, 12.8)
        ax.set_xticks([2.5, 9.5], ["C", "M"])
        ax.tick_params(axis="x", length=0, pad=0.8, labelsize=PT_SMALL)
        ax.tick_params(axis="y", length=1.2, pad=0.6, labelsize=PT_SMALL, which="both")
        ax.minorticks_off()
        clean(ax)
        ax.set_title(title, fontsize=PT_SMALL, pad=1.5)
    x = cv.key(x0 + lab, y0 + h - 1.6, [([{"kind": "box", "fc": TIS["cochlea"][1]}], "cochlea (C)"),
                                        ([{"kind": "box", "fc": TIS["bm"][1]}], "marrow (M)")])
    fits(x, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b-d card
def panels_bcd(cv, pal, x0, y0, w, h):
    cnt = pd.read_csv(R9 / "ed1_counts_by_sample.csv")
    ls = pd.read_csv(R9 / "ed1_label_support.csv")
    mix = pd.read_csv(R9 / "ed1_time_intermixing.csv")
    types = pal.sort_values("order")
    n = len(types)
    top = y0 + 10.4
    rp = (h - (top - y0) - 6.4) / n
    lab_w = max(text_width(a, PT_SMALL) for a in types.abbr) + 3.2
    # b — counts heat map
    cv.letter(x0, y0, "b", "Cells per sample")
    cw = 4.9
    hx = x0 + lab_w
    ax = cv.axes(hx, top, 13 * cw, n * rp)
    mat = np.full((n, 13), np.nan)
    for i, ct in enumerate(types.cell_type):
        for k, t in enumerate(["cochlea", "bm"]):
            for j, z in enumerate(ZTS):
                r = cnt[(cnt.cell_type == ct) & (cnt.tissue == t) & (cnt.zt == z)]
                mat[i, k * 7 + j] = np.log10(r.n_cells.iloc[0]) if len(r) and r.n_cells.iloc[0] > 0 else np.nan
    ax.imshow(np.ma.masked_invalid(mat), cmap=GREYS, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    ax.set_xticks([k * 7 + j for k in range(2) for j in range(6)], [str(z) for _ in range(2) for z in ZTS], rotation=90)
    ax.tick_params(axis="x", length=0, pad=0.6, labelsize=PT_SMALL)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, t in enumerate(["cochlea", "bm"]):
        xa = hx + (k * 7) * cw
        cv.box(xa, top - 2.0, 6 * cw, 0.6, TIS[t][1])
        cv.text(xa + 3 * cw, top - 2.4, TIS[t][0], size=PT_SMALL, ha="center", va="bottom")
    for i, (_, r) in enumerate(types.iterrows()):
        yy = top + (i + 0.5) * rp
        cv.box(x0 + lab_w - 2.0, yy - rp * 0.4, 1.2, rp * 0.8, r.colour)
        cv.text(x0 + lab_w - 2.6, yy, r.abbr, size=PT_SMALL, ha="right", va="center")
    cbx = hx + 13 * cw + 1.0
    axc = cv.axes(cbx, top + n * rp - 14.0, 1.2, 14.0)
    axc.imshow(np.linspace(4, 0, 100)[:, None], aspect="auto", cmap=GREYS, extent=(0, 1, 0, 4))
    axc.set_xticks([])
    axc.set_yticks([0, 2, 4], ["1", "10²", "10⁴"])
    axc.yaxis.tick_right()
    axc.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in axc.spines.values():
        sp.set_linewidth(0.3)
    cv.text(cbx - 0.2, top + n * rp - 15.0, "cells", size=PT_SMALL, va="bottom")
    # c — label support
    cx0 = cbx + 9.0
    cv.letter(cx0 - 1.0, y0, "c", "Label support")
    cww = (x0 + w - cx0 - 3.0) / 2
    axs = cv.axes(cx0, top, cww, n * rp)
    v = ls.median_margin
    lo, hi = np.floor(v.min() * 2) / 2, np.ceil(v.max() * 2) / 2
    axs.axvline(0, color="#999999", linewidth=0.45, zorder=1)
    for t, mk, off in [("cochlea", "o", -0.17), ("bm", "s", 0.17)]:
        d = ls[ls.tissue == t].set_index("cell_type")
        for i, ct in enumerate(types.cell_type):
            if ct in d.index:
                axs.scatter([d.loc[ct, "median_margin"]], [i + off], s=6, marker=mk, color=TIS[t][1], linewidths=0,
                            zorder=3)
    axs.set_xlim(lo, hi)
    axs.set_ylim(n - 0.5, -0.5)
    axs.set_yticks([])
    axs.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(axs, left=False)
    axs.set_xlabel("Median margin", fontsize=PT_SMALL, labelpad=0.8)
    for k in range(1, n):
        axs.axhline(k - 0.5, color="#F0F0F0", linewidth=0.3, zorder=0)
    # d — time intermixing
    dx0 = cx0 + cww + 3.0
    cv.letter(dx0 - 1.0, y0, "d", "Time intermixing")
    axm = cv.axes(dx0, top, cww, n * rp)
    axm.axvline(1.0, color="#999999", linewidth=0.45, linestyle=(0, (2, 1.5)), zorder=1)
    for t, mk, off in [("cochlea", "o", -0.17), ("bm", "s", 0.17)]:
        d = mix[mix.tissue == t].set_index("cell_type")
        for i, ct in enumerate(types.cell_type):
            if ct in d.index:
                axm.scatter([d.loc[ct, "intermixing_ratio"]], [i + off], s=6, marker=mk, color=TIS[t][1], linewidths=0,
                            zorder=3)
    assert mix.intermixing_ratio.between(0, 1.3).all()
    axm.set_xlim(0, 1.3)
    axm.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    axm.set_ylim(n - 0.5, -0.5)
    axm.set_yticks([])
    axm.tick_params(axis="x", length=1.2, pad=0.8, labelsize=PT_SMALL)
    clean(axm, left=False)
    axm.set_xlabel("Observed / expected", fontsize=PT_SMALL, labelpad=0.8)
    for k in range(1, n):
        axm.axhline(k - 0.5, color="#F0F0F0", linewidth=0.3, zorder=0)
    x = cv.key(cx0, top - 2.0, [([{"s": 6, "fc": TIS["cochlea"][1]}], "cochlea"),
                                ([{"s": 6, "marker": "s", "fc": TIS["bm"][1]}], "marrow")])
    fits(x, x0 + w, "cd key")


# ------------------------------------------------------------------------------------------------ e embedding
MARGIN = LinearSegmentedColormap.from_list("m", ["#B35806", "#F1A340", "#F7F7F7", "#998EC3", "#542788"])


def panel_e(cv, pal, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Joint embedding")
    u = pd.read_csv(R9 / "f1b_umap_joint.csv", usecols=["cell_id", "umap_1", "umap_2", "abbr", "tissue"])
    mg = pd.read_csv(R9 / "ed1_cell_margin.csv", usecols=["cell_id", "margin"])
    u = u.merge(mg, on="cell_id", how="left")
    assert u.margin.notna().all() and len(u) == 160643
    col = pal.set_index("abbr")["colour"]
    xl = (u.umap_1.min() - 0.4, u.umap_1.max() + 0.4)
    yl = (u.umap_2.min() - 0.4, u.umap_2.max() + 0.4)
    asp = 1.0                    # D143: the joint embedding is shown in a square frame (x stretched), as in Fig. 1
    top = y0 + 8.4
    fh = h - (top - y0) - 0.5
    fw = min((w - 2 * 3.0 - 8.0) / 3, fh / asp)
    rng = np.random.default_rng(7)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=3.0)
    for j, key in enumerate(["cochlea", "bm", "margin"]):
        bx = x0 + j * (fw + 3.0)
        ax = cv.axes(bx, top, fw, fw * asp)
        if key == "margin":
            d = u.iloc[rng.permutation(len(u))]
            ax.scatter(d.umap_1, d.umap_2, s=0.15, c=np.clip(d.margin, -1, 3), cmap=MARGIN, norm=norm, linewidths=0,
                       rasterized=True)
            title, tc = "Label margin (all cells)", TEXT
        else:
            ax.scatter(u.umap_1, u.umap_2, s=0.15, color="#E4E4E4", linewidths=0, rasterized=True)
            d = u[u.tissue == key]
            d = d.iloc[rng.permutation(len(d))]
            ax.scatter(d.umap_1, d.umap_2, s=0.22, color=d.abbr.map(col).to_numpy(), linewidths=0, rasterized=True)
            title, tc = f"{TIS[key][0]} ({len(d):,} cells)", TEXT
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        blank(ax)
        cv.text(bx + fw / 2, top - 0.8, title, size=PT_SMALL, ha="center", va="bottom", color=tc, fontweight="bold")
    cbx = x0 + 3 * fw + 2 * 3.0 + 1.6
    axc = cv.axes(cbx, top + 4.0, 1.2, 16.0)
    axc.imshow(np.linspace(3, -1, 200)[:, None], aspect="auto", cmap=MARGIN, norm=norm, extent=(0, 1, -1, 3))
    axc.set_xticks([])
    axc.set_yticks([-1, 0, 1, 2, 3], ["−1", "0", "1", "2", "≥3"])
    axc.yaxis.tick_right()
    axc.tick_params(length=0.8, pad=0.5, labelsize=PT_SMALL)
    for sp in axc.spines.values():
        sp.set_linewidth(0.3)
    cv.text(cbx - 0.2, top + 3.0, "margin", size=PT_SMALL, va="bottom")


# ------------------------------------------------------------------------------------------------ f composition (D82)
CLASS_ORDER = ["Lymphoid", "Progenitor", "Cycling", "Erythroid", "Myeloid"]      # stacking order, bottom to top


def panel_f(cv, x0, y0, w, h):
    """Haematopoietic cells per sample (moved from main Fig. 4 v1 panel a, D82)."""
    cv.letter(x0, y0, "f", "Haematopoietic cells per sample")
    d = pd.read_csv(R9 / "f4a_immune_composition.csv")
    d["cls_rank"] = d.cell_class.map({c: i for i, c in enumerate(CLASS_ORDER)})
    assert d.cls_rank.notna().all(), set(d.cell_class) - set(CLASS_ORDER)
    cls_col = pd.read_csv(R9 / "palette_classes.csv").set_index("cell_class")["colour"]
    zts = sorted(d.zt.unique())
    assert len(zts) == 6
    bw, pitch, gap = 2.4, 3.2, 3.6
    left = x0 + 7.0
    top_s, sh = y0 + 9.4, 4.6
    top_m, mh = y0 + 16.2, 23.0
    for j, t in enumerate(["cochlea", "bm"]):
        bx = left + j * (6 * pitch + gap)
        dt = d[d.tissue == t]
        cv.text(bx + 3 * pitch, y0 + 8.2, TIS[t][0], size=PT_SMALL, ha="center", va="bottom", fontweight="bold",
                color=TEXT)
        ax_s = cv.axes(bx, top_s, 6 * pitch, sh)
        tot = dt.groupby("zt").pct.sum().reindex(zts)
        assert tot.max() <= 100 + 1e-6
        ax_s.bar(range(6), tot.values, width=bw / pitch, color=TIS[t][1], linewidth=0)
        ax_s.set_xlim(-0.5, 5.5)
        ax_s.set_ylim(0, 100)
        clean(ax_s, bottom=False)
        ax_s.set_yticks([0, 100], ["0", "100"] if j == 0 else [])
        ax_s.tick_params(axis="y", length=1.0, pad=0.6, labelsize=PT_SMALL)
        ax = cv.axes(bx, top_m, 6 * pitch, mh)
        for i, z in enumerate(zts):
            ss = dt[dt.zt == z].sort_values(["cls_rank", "order"])
            assert abs(ss.pct_of_immune.sum() - 100) < 1e-6
            ax.bar(i, ss.pct_of_immune.values, bottom=np.r_[0, ss.pct_of_immune.cumsum().values[:-1]],
                   width=bw / pitch, color=ss.colour.values, linewidth=0)
            cv.box(bx + (i + 0.5) * pitch - bw / 2, top_m + mh + 0.4, bw, 0.8, "#FFFFFF" if z < 12 else "#4D4D4D",
                   AXIS, 0.35)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(6), [str(int(z)) for z in zts])
        ax.tick_params(axis="x", length=0, pad=4.0, labelsize=PT_SMALL)
        clean(ax)
        ax.spines["bottom"].set_visible(False)
        ax.set_yticks([0, 50, 100], ["0", "50", "100"] if j == 0 else [])
        if j == 0:
            ax.set_ylabel("% of haematopoietic cells", fontsize=PT_SMALL, labelpad=1.0)
            ax_s.set_ylabel("% all", fontsize=PT_SMALL, labelpad=1.0)
    fits(left + 2 * 6 * pitch + gap, x0 + w, "f bars")
    cv.text(left + 6 * pitch + gap / 2, top_m + mh + 5.0, "ZT (h)", size=PT_SMALL, ha="center", va="top")
    ky = y0 + h - 5.0
    x = cv.key(x0 + 1.0, ky, [([{"kind": "box", "fc": cls_col[c]}], c) for c in ["Lymphoid", "Myeloid", "Progenitor"]])
    fits(x, x0 + w, "f key 1")
    x = cv.key(x0 + 1.0, ky + 3.0, [([{"kind": "box", "fc": cls_col[c]}], c) for c in ["Erythroid", "Cycling"]])
    fits(x, x0 + w, "f key 2")


# ------------------------------------------------------------------------------------------------ main
def main():
    pal = pd.read_csv(R9 / "palette_celltypes.csv")
    cv = Canvas(width=W_ED)
    panel_a(cv, 0.0, 0.0, W_ED, 36.0)
    panels_bcd(cv, pal, 0.0, 38.0, W_ED, 74.0)
    panel_e(cv, pal, 0.0, 114.0, 128.0, 56.0)
    panel_f(cv, 130.0, 114.0, 50.0, 56.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
