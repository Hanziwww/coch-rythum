"""Round 9 (D68 / D69) — Extended Data Fig. 3 (180 x 170 mm): external positioning of the atlas (supports Fig. 1e, f).

a  Four external cochlear atlases: cells, share confidently transferred, our types recovered (>= 20 confident cells),
   marker-check pass rate of the labels checked.
b  GSE234926 author cell types x transferred labels (dot area = share of the author type); outlined = expected label
   (solid strict, dashed broad); bar = author cells.
c  Structural-type composition of each external sample vs ours (Spearman rho; dot area = structural cells).
d  GeoMx: deconvolved composition of every area of illumination by region and exposure (types >= 1 % on average;
   the rest pooled), with the weighted fit r2 below.
e  GeoMx control areas: mean proportion of every type in the neuron vs the supporting-cell region (dark link q < 0.05).

Inputs: results/r9_figures/ed3_*.csv, ed2_atlas_columns.csv, ed2_atlas_validation.csv. Output: figures/figures/ed3_external_v1.*
Run: python scripts/922_ed3_external.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, header_height, text_width,  # noqa: E402
                          type_header)

from lib.figstyle import Q_SYM  # noqa: E402  (italic q, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed3_external_v1"
W_ED = 180.0
BAR = "#6E6E6E"
DOT = "#3D3D3D"
OTHER = "#E3E3E3"
REGION = {"spiral_ganglion_neurons": "SGN region", "supporting_cells": "Supporting-cell region"}


def acc(dataset):
    return dataset.split(" ")[0]


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


# ------------------------------------------------------------------------------------------------ a atlas summary
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "External atlases")
    s = pd.read_csv(R9 / "ed3_atlas_summary.csv")
    order = pd.read_csv(R9 / "ed2_atlas_validation.csv").drop_duplicates("key").set_index("key")["atlas_order"]
    s["o"] = s.key.map(order)
    assert s.o.notna().all()
    s = s.sort_values("o").reset_index(drop=True)
    s["acc"] = s.dataset.map(acc)
    metrics = [("Cells", s.n_cells_qc / 30000, [f"{v / 1000:.1f}k" for v in s.n_cells_qc]),
               ("Confident", s.frac_confident, [f"{v:.2f}" for v in s.frac_confident]),
               ("Types recovered", s.n_types_recovered_20_confident / s.n_our_types,
                [f"{a}/{b}" for a, b in zip(s.n_types_recovered_20_confident, s.n_our_types)]),
               ("Marker check", s.marker_check_pass_rate,
                [f"{v:.2f}" for v in s.marker_check_pass_rate])]
    lab_w = max(text_width(a, PT_SMALL) for a in s.acc) + 1.4
    top, rp = y0 + 8.4, 4.6
    cw = (w - lab_w) / len(metrics)
    for k, (_, r) in enumerate(s.iterrows()):
        cv.text(x0 + lab_w - 0.8, top + (k + 0.5) * rp, r.acc, size=PT_SMALL, ha="right", va="center")
    for j, (title, frac, labels) in enumerate(metrics):
        bx = x0 + lab_w + j * cw
        bw = cw - 8.0
        cv.text(bx, top - 1.0, title, size=PT_SMALL, va="bottom", color=TEXT2)
        cv.line([bx, bx], [top, top + len(s) * rp], TEXT, 0.35)
        for k, (f, lab) in enumerate(zip(frac, labels)):
            yy = top + (k + 0.5) * rp
            cv.box(bx, yy - 1.1, bw * float(np.clip(f, 0, 1)), 2.2, BAR)
            cv.text(bx + bw * float(np.clip(f, 0, 1)) + 0.6, yy, lab, size=PT_SMALL, va="center")
        fits(bx + bw + 0.6 + max(text_width(t, PT_SMALL) for t in labels), x0 + lab_w + (j + 1) * cw, f"a {title}")


# ------------------------------------------------------------------------------------------------ b author labels
def panel_b(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Author labels, GSE234926")
    ct = pd.read_csv(R9 / "ed3_author_contingency.csv")
    ag = pd.read_csv(R9 / "ed3_author_labels.csv")
    ag = ag[ag.author_celltype != "ALL (cell-weighted)"].set_index("author_celltype")
    name2abbr = dict(zip(cols.cell_type, cols.abbr))
    colpos = dict(zip(cols.abbr, cols.column_order))

    def split(v):
        return [name2abbr[t] for t in str(v).split("; ")] if isinstance(v, str) and v else []

    strict = {a: split(r.expected_labels) for a, r in ag.iterrows()}
    broad = {a: [t for t in split(r.expected_labels_broad) if t not in strict[a]] for a, r in ag.iterrows()}
    labels = set(ct.abbr) | {t for v in strict.values() for t in v} | {t for v in broad.values() for t in v}
    labs = sorted(labels, key=colpos.get)
    dom = ct.sort_values("share_of_author_type", ascending=False).drop_duplicates("author_celltype")
    dom = dict(zip(dom.author_celltype, dom.abbr))
    rows = sorted(ag.index, key=lambda a: (min(colpos[t] for t in (strict[a] or broad[a] or [dom[a]])),
                                           colpos[dom[a]], a))
    lab_w = max(text_width(a, PT_SMALL) for a in rows) + 1.4
    n_w = 13.0
    top = y0 + header_height(labs)
    q = (w - lab_w - n_w - 1.5) / len(labs)
    rp = (h - (top - y0) - 6.4) / len(rows)
    ax = cv.axes(x0 + lab_w, top, len(labs) * q, len(rows) * rp)
    ci = {t: i for i, t in enumerate(labs)}
    ri = {a: i for i, a in enumerate(rows)}
    for k in range(len(rows)):
        ax.axhline(k, color=GRID, lw=0.25, zorder=0)
    for a in rows:
        for t in strict[a]:
            ax.add_patch(Rectangle((ci[t] - 0.5, ri[a] - 0.5), 1, 1, fill=False, ec=TEXT, lw=0.5, zorder=2))
        for t in broad[a]:
            ax.add_patch(Rectangle((ci[t] - 0.5, ri[a] - 0.5), 1, 1, fill=False, ec=TEXT, lw=0.5, ls=(0, (1.6, 1.0)),
                                   zorder=2))
    ax.scatter(ct.abbr.map(ci), ct.author_celltype.map(ri), s=0.4 + 16.0 * ct.share_of_author_type, c=DOT,
               linewidths=0, zorder=3)
    ax.set_xlim(-0.5, len(labs) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_axis_off()
    for a in rows:
        cv.text(x0 + lab_w - 0.8, top + (ri[a] + 0.5) * rp, a, size=PT_SMALL, ha="right", va="center")
    type_header(cv, labs, [cols.set_index("abbr").colour[t] for t in labs], x0 + lab_w, top, q)
    # author cells per row
    bx = x0 + w - n_w
    axn = cv.axes(bx, top, n_w - 1.0, len(rows) * rp)
    axn.barh([ri[a] for a in rows], [ag.n_cells[a] for a in rows], height=0.62, color=BAR, linewidth=0)
    axn.set_ylim(len(rows) - 0.5, -0.5)
    axn.set_xlim(0, 450)
    axn.set_yticks([])
    axn.set_xticks([0, 200, 400], ["0", "200", "400"])
    clean(axn, left=False)
    tick_style(axn)
    cv.text(bx, top - 1.0, "Cells", size=PT_SMALL, va="bottom", color=TEXT2)
    ky = y0 + h - 1.4
    xx = cv.key(x0 + lab_w, ky, [([{"s": 0.4 + 16.0 * f, "fc": DOT}], f"{f:g}") for f in (0.1, 0.5, 1.0)], gap=1.6)
    xx = cv.key(xx + 2.4, ky, [([{"kind": "box", "fc": "white", "ec": TEXT, "lw": 0.5, "size": 1.6}], "expected"),
                               ([{"kind": "hline", "w": 2.4, "ls": (0, (1.6, 1.0)), "lw": 0.5}], "broad")])
    fits(xx, x0 + w, "b key")
    cv.text(x0 + lab_w - 0.8, ky, "Share", size=PT_SMALL, ha="right", va="center", color=TEXT2)


# ------------------------------------------------------------------------------------------------ c composition
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Structural composition vs ours")
    d = pd.read_csv(R9 / "ed3_composition_similarity.csv")
    assert (d.p >= 0.05).all()
    d["acc"] = d.dataset.map(acc)
    d["short"] = d["sample"].str.replace("Cochlea_10weeks_Bio_", "", regex=False).str.replace("_Cochlea", "", regex=False)
    grp_w = max(text_width(a, PT_SMALL) for a in d.acc) + 1.6
    lab_w = grp_w + max(text_width(a, PT_SMALL) for a in d.short) + 1.6
    top = y0 + 5.2
    gap = 1.0
    rp = (h - (top - y0) - 6.0 - gap) / len(d)
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 2.0, len(d) * rp + gap)
    ypos, k = [], 0.0
    prev = None
    for _, r in d.iterrows():
        if prev is not None and r.acc != prev:
            k += gap / rp
        ypos.append(k)
        prev = r.acc
        k += 1
    d["y"] = ypos
    ax.axvline(0, color=TEXT2, lw=0.35, zorder=1)
    ax.scatter(d.spearman_rho_structural_shares, d.y, s=3.0 + 22.0 * d.n_structural_cells / 4200.0, facecolors="white",
               edgecolors=DOT, linewidths=0.6, zorder=3)
    ax.set_ylim(d.y.max() + 0.6, -0.6)
    ax.set_xlim(-0.3, 0.7)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6], ["−0.2", "0", "0.2", "0.4", "0.6"])
    ax.set_yticks([])
    clean(ax, left=False)
    tick_style(ax)
    ax.set_xlabel("Spearman ρ, structural shares", fontsize=PT_SMALL, labelpad=1.0)
    for _, r in d.iterrows():
        cv.text(x0 + lab_w - 0.8, top + (r.y + 0.5) * rp, r.short, size=PT_SMALL, ha="right", va="center")
    for a, g in d.groupby("acc", sort=False):
        ymid = top + (g.y.mean() + 0.5) * rp
        cv.text(x0, ymid, a, size=PT_SMALL, va="center", color=TEXT2)
        cv.line([x0 + grp_w - 0.6] * 2, [top + g.y.min() * rp + 0.4, top + (g.y.max() + 1) * rp - 0.4], TEXT2, 0.35)
    xx = cv.key(x0 + w - 2.0 - 26.0, top - 2.6,
                [([{"s": 3.0 + 22.0 * n / 4200.0, "fc": "white", "ec": DOT, "lw": 0.6}], f"{n:,}") for n in (500, 2000, 4000)],
                gap=1.4)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d GeoMx per area
def panel_d(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "d", "GeoMx areas of illumination")
    g = pd.read_csv(R9 / "ed3_geomx_aoi.csv")
    assert (g.groupby("aoi").proportion.sum() - 1).abs().max() < 1e-6
    mean = g.groupby("abbr").proportion.mean()
    shown = [a for a in cols.sort_values("column_order").abbr if mean.get(a, 0) >= 0.01]
    colour = dict(zip(cols.abbr, cols.colour))
    wide = g.pivot_table(index="aoi", columns="abbr", values="proportion", fill_value=0.0)
    wide["other"] = wide.drop(columns=shown).sum(axis=1)
    meta = g.drop_duplicates("aoi").set_index("aoi")[["region", "noise", "fit_r2_weighted"]]
    groups = [("spiral_ganglion_neurons", "control"), ("spiral_ganglion_neurons", "noise"),
              ("supporting_cells", "control"), ("supporting_cells", "noise")]
    sort_by = {"spiral_ganglion_neurons": "SGN", "supporting_cells": "Schwann"}
    xs, order, spans = [], [], []
    x = 0.0
    for gi_, (reg, noi) in enumerate(groups):
        if gi_:
            x += 3.0 if groups[gi_ - 1][0] != reg else 1.4
        ids = meta[(meta.region == reg) & (meta.noise == noi)].index
        ids = wide.loc[ids].sort_values(sort_by[reg], ascending=False).index.tolist()
        spans.append((reg, noi, x, x + len(ids) - 1))
        for a in ids:
            xs.append(x)
            order.append(a)
            x += 1.0
    assert len(order) == len(meta) == 51
    lab_w = 9.0
    top = y0 + 5.4
    bh, sh = 34.0, 6.0
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 0.5, bh)
    bottom = np.zeros(len(order))
    for a in shown + ["other"]:
        v = wide.loc[order, a].to_numpy()
        ax.bar(xs, v, bottom=bottom, width=0.82, color=colour.get(a, OTHER), linewidth=0)
        bottom += v
    ax.set_xlim(-0.7, x - 0.3)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xticks([])
    clean(ax, bottom=False)
    tick_style(ax)
    ax.set_ylabel("Proportion", fontsize=PT_SMALL, labelpad=1.0)
    axr = cv.axes(x0 + lab_w, top + bh + 1.2, w - lab_w - 0.5, sh)
    r2 = meta.loc[order, "fit_r2_weighted"].to_numpy()
    floor = -0.3
    axr.bar(xs, np.clip(r2, floor, None), width=0.82, color=BAR, linewidth=0)
    low = r2 < floor
    axr.scatter(np.asarray(xs)[low], [floor] * int(low.sum()), marker="v", s=5, c=TEXT, linewidths=0, zorder=3,
                clip_on=False)
    axr.axhline(0, color=TEXT2, lw=0.35)
    axr.set_xlim(-0.7, x - 0.3)
    axr.set_ylim(floor, 0.55)
    axr.set_yticks([0, 0.5], ["0", "0.5"])
    axr.set_xticks([])
    clean(axr, bottom=False)
    tick_style(axr)
    axr.set_ylabel("Fit r²", fontsize=PT_SMALL, labelpad=1.0)
    pitch = (w - lab_w - 0.5) / (x - 0.3 + 0.7)

    def xmm(v):
        return x0 + lab_w + (v + 0.7) * pitch

    gy = top + bh + 1.2 + sh + 1.6
    for reg, noi, a, b in spans:
        cv.text((xmm(a - 0.4) + xmm(b + 0.4)) / 2, gy, noi, size=PT_SMALL, ha="center", va="top", color=TEXT2)
    for reg in REGION:
        sp = [(a, b) for r_, n_, a, b in spans if r_ == reg]
        xa, xb = xmm(sp[0][0] - 0.4), xmm(sp[-1][1] + 0.4)
        cv.line([xa, xb], [gy + 2.6, gy + 2.6], TEXT, 0.35)
        cv.text((xa + xb) / 2, gy + 3.0, REGION[reg], size=PT_SMALL, ha="center", va="top")
    items = [([{"kind": "box", "fc": colour[a]}], a) for a in shown] + [([{"kind": "box", "fc": OTHER}], "other")]
    xx = cv.key(x0 + lab_w, y0 + h - 1.4, items, gap=1.8)
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e GeoMx contrast
def panel_e(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Region contrast, control areas")
    s = pd.read_csv(R9 / "ed3_geomx_summary.csv")
    s = s.merge(cols[["abbr", "column_order"]], on="abbr").sort_values("column_order").reset_index(drop=True)
    assert len(s) == 35
    lab_w = 9.0
    top = y0 + 5.0
    lab_h = max(text_width(a, PT_SMALL) for a in s.abbr) + 3.2
    ph = h - (top - y0) - lab_h
    q = (w - lab_w - 0.5) / len(s)
    ax = cv.axes(x0 + lab_w, top, len(s) * q, ph)
    xs = np.arange(len(s))
    sig = s.q_region < 0.05
    for i, r in s.iterrows():
        ax.plot([i, i], [r.mean_prop_neuron_control, r.mean_prop_supporting_control], color=TEXT if sig[i] else GRID,
                lw=0.8 if sig[i] else 0.6, zorder=1, solid_capstyle="butt")
    ax.scatter(xs, s.mean_prop_neuron_control, s=7, c=TEXT, linewidths=0, zorder=3, clip_on=False)
    ax.scatter(xs, s.mean_prop_supporting_control, s=7, facecolors="white", edgecolors=TEXT, linewidths=0.5, zorder=3,
               clip_on=False)
    ax.set_yscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), lambda v: v ** 2))
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5], ["0", "0.01", "0.05", "0.1", "0.25", "0.5"])
    ax.set_xlim(-0.5, len(s) - 0.5)
    ax.set_xticks([])
    ax.yaxis.grid(True, color=GRID, lw=0.25)
    ax.set_axisbelow(True)
    clean(ax, bottom=False)
    tick_style(ax)
    ax.set_ylabel("Proportion", fontsize=PT_SMALL, labelpad=1.0)
    yb = top + ph + 1.2
    for i, r in s.iterrows():
        cv.box(x0 + lab_w + (i + 0.5) * q - 0.8, yb, 1.6, 1.0, r.colour)
        cv.text(x0 + lab_w + (i + 0.5) * q, yb + 1.6, r.abbr, size=PT_SMALL, ha="center", va="top", rotation=90)
    xx = cv.key(x0 + w - 62.0, y0 + 1.6, [([{"s": 7, "fc": TEXT}], "SGN region"),
                                         ([{"s": 7, "fc": "white", "ec": TEXT, "lw": 0.5}], "supporting-cell region"),
                                         ([{"kind": "hline", "w": 2.4, "lw": 0.8}], f"{Q_SYM} < 0.05")], gap=2.0)
    fits(xx, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED)
    panel_a(cv, 0.0, 0.0, 86.0, 30.0)
    panel_b(cv, cols, 90.0, 0.0, 90.0, 70.0)
    panel_c(cv, 0.0, 33.0, 86.0, 37.0)
    panel_d(cv, cols, 0.0, 73.0, W_ED, 57.0)
    panel_e(cv, cols, 0.0, 132.0, W_ED, 38.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
