"""Round 9 (D68 / D69) — Extended Data Fig. 10 (180 x 170 mm): the independent cohort in detail (supports Fig. 6).

a  Inferred collection phase (our ZT) of every Ma 2025 library under each clock estimator: T2 (primary; bar, 95 %
   interval), T1, noise-robust T1, T3 variants (composition-robust, per structural cell type).
b  Phase separation of each group contrast under each estimator (bar, 95 % interval) vs the upper 95 % bound of a
   noise-associated clock shift (grey; T2 bound); filled, interval beyond the bound.
c  Label check: noise-response score and clock phase of every library, coloured by its label (no corrections).
d  Bulk deconvolution validated on pseudo-bulk mixtures of Ma single cells: correlation of true and estimated fractions
   per label against the label's mean fraction (filled, passes).
e  Summary tests (Spearman of observed change vs our predicted change) under the eight timing assumptions (and the
   time-matched noise-response check): colour, rho; outlined, one-sided permutation p < 0.05.

Inputs: results/r9_figures/ed10_*.csv, palette_celltypes.csv. Output: figures/figures/ed10_ma2025_v1.*
Run: python scripts/929_ed10_ma2025.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, ld_bar, text_width,  # noqa: E402
                          zt_axis)

from lib.figstyle import P_SYM, R_SYM  # noqa: E402,F401  (italic P / r, D99)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed10_ma2025_v1"
W_ED = 180.0
GROUPS = {"control": ("control", "#1A1A1A"), "day_noise": ("day noise", "#E69F00"), "night_noise": ("night noise", "#6B4C9A"),
          "CMD": ("CMD", "#1A1A1A"), "CMN": ("CMN", "#7F7F7F"), "NMD": ("NMD", "#E69F00"), "NMN": ("NMN", "#6B4C9A")}
EST = {"T2": ("T2 (primary)", "#1A1A1A", "o", 9), "T1": ("T1", "#7F7F7F", "o", 6), "T1_nr": ("T1 noise-robust", "#BDBDBD", "o", 5),
       "T3": ("T3", "#56B4E9", "s", 5)}
DIV = LinearSegmentedColormap.from_list("div", ["#B35806", "#F1A340", "#F7F7F7", "#998EC3", "#542788"])
ASSUME = ["A0 inferred (T2)", "T1", "T3_30", "offset −4 h", "offset −2 h", "offset +2 h", "offset +4 h",
          "exposure-time assumption", "time-matched (no phase assumption)"]
ASSUME_LAB = ["T2", "T1", "T3", "−4 h", "−2 h", "+2 h", "+4 h", "exposure", "matched"]
FAMILY = [("scRNA composition | absolute (share of all cells) | all cells", "sc abs"),
          ("scRNA composition | relative (share of immune cells) | all cells", "sc rel"),
          ("scRNA composition | absolute (share of all cells) | confident cells", "sc abs conf"),
          ("scRNA composition | relative (share of immune cells) | confident cells", "sc rel conf"),
          ("bulk composition | absolute (fraction of all cells)", "bulk abs"),
          ("bulk composition | relative (fraction of immune cells)", "bulk rel"),
          ("bulk composition | absolute marker signature (indirect; 680 B1 definition, P excluded)", "bulk markers"),
          ("scRNA states | B", "sc B states"), ("scRNA states | myeloid", "sc myeloid"),
          ("noise response replication (time-matched)", "noise")]
CONTRAST = {"night_noise − control": "N − C", "night_noise − day_noise": "N − D", "CMN − CMD": "CMN − CMD",
            "NMN − NMD": "NMN − NMD", "scRNA day_noise − control vs bulk NMD − CMD": "sc vs bulk"}
SCOPE = {"all adequate labels": "all", "myeloid": "myeloid", "B lineage": "B", "validated immune labels": "validated",
         "all items with a fit": "all", "items whose own fit has p_perm <= 0.20": f"{P_SYM} ≤ 0.2",
         "neutrophils + monocytes (pooled)": "myeloid", "B lineage (pooled)": "B"}


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def est_family(e):
    return "T3" if str(e).startswith("T3") else e


# ------------------------------------------------------------------------------------------------ a library phases
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Library collection phase")
    d = pd.read_csv(R9 / "ed10_library_phase.csv")
    d["fam"] = d.estimator.map(est_family)
    libs = d.drop_duplicates(["dataset", "sample"])[["dataset", "sample", "group"]]
    order = {g: i for i, g in enumerate(GROUPS)}
    libs = libs.assign(o=libs.group.map(order)).sort_values(["dataset", "o", "sample"], ascending=[False, True, True])
    lab_w = max(text_width(s_, PT_SMALL) for s_ in libs["sample"]) + 3.0
    top = y0 + 8.0
    gap = 2.0
    rp = (h - (top - y0) - gap - 10.0) / len(libs)
    ys, y = {}, 0.0
    prev = None
    for _, r in libs.iterrows():
        if prev is not None and r.dataset != prev:
            y += gap / rp
        ys[(r.dataset, r["sample"])] = y
        prev = r.dataset
        y += 1
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, y * rp)
    for (ds, smp), yv in ys.items():
        ax.axhline(yv, color=GRID, lw=0.25, zorder=0)
        s = d[(d.dataset == ds) & (d["sample"] == smp)]
        for fam, (lab, col, mk, sz) in EST.items():
            f = s[s.fam == fam]
            if not len(f):
                continue
            off = {"T2": 0.0, "T1": -0.22, "T1_nr": 0.22, "T3": 0.36}[fam]
            if fam == "T2":
                r = f.iloc[0]
                lo, hi = (r.our_zt - r.ci95_halfwidth_h) % 24, (r.our_zt + r.ci95_halfwidth_h) % 24
                segs = [(lo, hi)] if lo <= hi else [(lo, 24.0), (0.0, hi)]
                for a_, b_ in segs:
                    ax.plot([a_, b_], [yv, yv], color=col, lw=0.6, zorder=2)
            ax.scatter(f.our_zt % 24, [yv + off] * len(f), marker=mk, s=sz, c=col, linewidths=0, zorder=3)
        grp = s.group.iloc[0]
        cv.box(x0 + lab_w - 1.9, top + (yv + 0.5) * rp - rp * 0.34, 1.2, rp * 0.68, GROUPS[grp][1])
        cv.text(x0 + lab_w - 2.4, top + (yv + 0.5) * rp, smp, size=PT_SMALL, ha="right", va="center")
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_yticks([])
    zt_axis(ax, ticks=(0, 6, 12, 18, 24))
    clean(ax, left=False)
    ax.spines["bottom"].set_visible(False)
    tick_style(ax)
    ax.tick_params(axis="x", length=0, pad=2.6)
    ld_bar(cv, x0 + lab_w, top + y * rp + 0.1, w - lab_w - 1.0, 0.7)
    for ds in libs.dataset.unique():
        yv = min(v for (dd, _), v in ys.items() if dd == ds)
        cv.text(x0 + lab_w + 0.6, top + yv * rp - 0.2, "scRNA" if ds.startswith("scRNA") else "bulk", size=PT_SMALL,
                va="bottom", color=TEXT2)
    items = [([{"s": sz, "marker": mk, "fc": col}], lab) for lab, col, mk, sz in EST.values()]
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, items, gap=1.6)
    fits(xx, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b separations
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Group separation vs noise-shift bound")
    r = pd.read_csv(R9 / "ed10_separations.csv")
    bound = float(r[r["item"].str.startswith("upper 95 % bound of a noise-associated phase shift [T2]")].value_h.iloc[0])
    rows = r[r["item"].str.contains(r" − .*\(T", regex=True) & ~r["item"].str.contains("noise-associated")].copy()
    rows["contrast"] = rows["item"].str.replace(r"\s*\((T[^)]*)\)\s*$", "", regex=True)
    rows["estimator"] = rows["item"].str.extract(r"\((T[^)]*)\)\s*$")[0]
    assert rows.estimator.notna().all()
    rows["absv"] = rows.value_h.abs()
    lab_c = max(text_width(c, PT_SMALL) for c in rows.contrast.unique()) + 1.4
    lab_e = max(text_width(e, PT_SMALL) for e in rows.estimator) + 1.0
    top = y0 + 6.0
    rp = (h - (top - y0) - 10.0) / len(rows)
    ax = cv.axes(x0 + lab_c + lab_e, top, w - lab_c - lab_e - 2.4, len(rows) * rp)
    ax.axvspan(0, bound, color="#E6E6E6", lw=0, zorder=0)
    prev = None
    for i, (_, q) in enumerate(rows.iterrows()):
        ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        lo = q.absv - q.ci95_halfwidth_h
        beyond = lo > bound
        ax.plot([max(lo, 0), q.absv + q.ci95_halfwidth_h], [i, i], color=TEXT, lw=0.6, zorder=2)
        ax.scatter([q.absv], [i], s=9, facecolors=TEXT if beyond else "white", edgecolors=TEXT, linewidths=0.6,
                   zorder=3)
        yy = top + (i + 0.5) * rp
        cv.text(x0 + lab_c + lab_e - 0.6, yy, q.estimator, size=PT_SMALL, ha="right", va="center", color=TEXT2)
        if q.contrast != prev:
            cv.text(x0 + lab_c - 0.6, yy, q.contrast, size=PT_SMALL, ha="right", va="center")
        prev = q.contrast
    ax.set_xlim(0, 15)
    ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    clean(ax, left=False)
    tick_style(ax)
    ax.set_xlabel("Phase difference (h)", fontsize=PT_SMALL, labelpad=1.0)


# ------------------------------------------------------------------------------------------------ c label check
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Library labels")
    d = pd.read_csv(R9 / "ed10_annotation_check.csv")
    assert not d.label_corrected.astype(bool).any()
    ax = cv.axes(x0 + 9.0, y0 + 6.0, w - 10.0, h - 6.0 - 12.0)
    for ds, mk in (("scRNA", "o"), ("bulk", "s")):
        s = d[d.dataset.str.startswith(ds)]
        ax.scatter(s.noise_score, s.internal_phase_T1_nr % 24, marker=mk, s=10, c=[GROUPS[g][1] for g in s.group],
                   linewidths=0.3, edgecolors="white", zorder=3)
    ax.set_ylim(0, 24)
    ax.set_yticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
    clean(ax)
    tick_style(ax)
    ax.set_xlabel("Noise-response score", fontsize=PT_SMALL, labelpad=1.0)
    ax.set_ylabel("Clock phase (h)", fontsize=PT_SMALL, labelpad=1.0)
    items = [([{"s": 10, "fc": GROUPS[g][1]}], GROUPS[g][0]) for g in ("control", "day_noise", "night_noise")]
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, items + [([{"s": 10, "marker": "s", "fc": TEXT2}], "bulk")], gap=1.4)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d deconvolution
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Bulk deconvolution check")
    d = pd.read_csv(R9 / "ed10_deconvolution_validation.csv")
    ax = cv.axes(x0 + 9.0, y0 + 6.0, w - 10.0, h - 6.0 - 12.0)
    ok = d.passes.astype(bool)
    x = d.mean_true_fraction.clip(lower=1e-4)
    ax.scatter(x[ok], d.validation_r[ok], s=10, c=d.colour[ok], linewidths=0.3, edgecolors="white", zorder=3)
    ax.scatter(x[~ok], d.validation_r[~ok], s=9, facecolors="white", edgecolors=d.colour[~ok], linewidths=0.7, zorder=3)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.5)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1], ["10⁻⁴", "10⁻³", "0.01", "0.1"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.6, 1.05)
    ax.set_yticks([-0.5, 0, 0.5, 1], ["−0.5", "0", "0.5", "1"])
    clean(ax)
    tick_style(ax)
    ax.set_xlabel("Mean true fraction", fontsize=PT_SMALL, labelpad=1.0)
    ax.set_ylabel("True vs estimated r", fontsize=PT_SMALL, labelpad=1.0)
    xx = cv.key(x0 + 9.0, y0 + h - 1.4, [([{"s": 10, "fc": TEXT2}], "passes"),
                                         ([{"s": 9, "fc": "white", "ec": TEXT2, "lw": 0.7}], "fails")], gap=1.6)
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e assumptions
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Summary tests under timing assumptions")
    t = pd.read_csv(R9 / "ed10_assumption_tests.csv")
    fam = dict(FAMILY)
    assert set(t.family) == set(fam), set(t.family) ^ set(fam)
    assert set(t.assumption) <= set(ASSUME) and set(t.contrast) <= set(CONTRAST) and set(t.scope) <= set(SCOPE)
    keys = t.drop_duplicates(["family", "contrast", "scope"])[["family", "contrast", "scope"]]
    forder = {f: i for i, (f, _) in enumerate(FAMILY)}
    keys = keys.assign(o=keys.family.map(forder)).sort_values("o", kind="stable").reset_index(drop=True)
    labels = [f"{fam[f]} · {CONTRAST[c]} · {SCOPE[s_]}" for f, c, s_ in zip(keys.family, keys.contrast, keys.scope)]
    row_w = max(text_width(a, PT_SMALL) for a in ASSUME_LAB) + 1.4
    top = y0 + 4.6 + max(text_width(a, PT_SMALL) for a in labels) + 1.0
    q = (w - row_w - 1.0) / len(keys)
    rp = (h - (top - y0) - 9.0) / len(ASSUME)
    ax = cv.axes(x0 + row_w, top, len(keys) * q, len(ASSUME) * rp)
    prev = None
    for i, k in keys.iterrows():
        sub = t[(t.family == k.family) & (t.contrast == k.contrast) & (t.scope == k.scope)]
        for _, r in sub.iterrows():
            if not np.isfinite(r.spearman):
                continue
            j = ASSUME.index(r.assumption)
            sig = r.p_perm_one_sided < 0.05
            ax.scatter([i], [j], marker="s", s=13, c=[DIV((np.clip(r.spearman, -1, 1) + 1) / 2)],
                       edgecolors=TEXT if sig else "none", linewidths=0.7 if sig else 0, zorder=3)
        if prev is not None and k.family != prev:
            ax.axvline(i - 0.5, color=GRID, lw=0.5, zorder=0)
        prev = k.family
        cv.text(x0 + row_w + (i + 0.5) * q, top - 0.8, labels[i], size=PT_SMALL, ha="center", va="bottom", rotation=90)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(len(ASSUME) - 0.5, -0.5)
    ax.set_axis_off()
    for j, a in enumerate(ASSUME_LAB):
        cv.text(x0 + row_w - 0.8, top + (j + 0.5) * rp, a, size=PT_SMALL, ha="right", va="center", color=TEXT2)
    ky = y0 + h - 4.4
    axk = cv.axes(x0 + 8.0, ky - 0.6, 14.0, 1.2)
    axk.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=DIV, extent=(-1, 1, 0, 1))
    axk.set_yticks([])
    axk.set_xticks([-1, 0, 1], ["−1", "0", "1"])
    tick_style(axk)
    cv.text(x0 + 7.4, ky, "ρ", size=PT_SMALL, ha="right", va="center")
    xx = cv.key(x0 + 26.0, ky, [([{"s": 13, "marker": "s", "fc": DIV(0.8), "ec": TEXT, "lw": 0.7}], f"{P_SYM} < 0.05")])
    fits(xx, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cv = Canvas(width=W_ED)
    panel_a(cv, 0.0, 0.0, 88.0, 80.0)
    panel_b(cv, 92.0, 0.0, 88.0, 80.0)
    panel_e(cv, 0.0, 84.0, 128.0, 86.0)
    panel_c(cv, 132.0, 84.0, 48.0, 42.0)
    panel_d(cv, 132.0, 128.0, 48.0, 42.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
