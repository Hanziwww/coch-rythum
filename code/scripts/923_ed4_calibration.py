"""Round 9 (D68 / D69) — Extended Data Fig. 4 (180 x 170 mm): calibration and power of the rhythm calls (R09).

a  Dense held-out series (GSE54650 organs, GSE88954 cochlea) subsampled to our design: realised false-discovery
   proportion (strict truth), sensitivity and median phase error per calling rule.
b  Sensitivity of Tier 1-2 calls by true fold change, without and with our sample-level pattern (half, full strength).
c  Core-clock genes recovered as Tier 1-2 calls per dataset and sample-pattern strength.
d  Per cell type, the smallest peak-to-trough fold change detectable with 80 % power: gene rhythms (alpha 0.001; one
   dot per expression quintile) and share rhythms (alpha 0.05; propeller and CLR). Triangles: not reached.
e  Simulation on the data's own noise (time relabelled, rhythms injected by thinning): power, false-discovery
   proportion and phase error per rule (mean of 8 cell types x 3 replicates; open = < 1 call per run).

Inputs: results/r9_figures/ed4_*.csv, ed2_atlas_columns.csv, palette_celltypes.csv. Output: figures/figures/ed4_calibration_v1.*
Run: python scripts/923_ed4_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, text_width  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed4_calibration_v1"
W_ED = 180.0
BAR = "#6E6E6E"
COND = {"clean": ("none", "#1A1A1A"), "sample pattern (half)": ("half", "#8C8C8C"),
        "sample pattern (full)": ("full", "#CFCFCF")}
DATA = {"GSE54650 (DD, microarray)": "Organs", "GSE88954 (cochlea sham, RNA-seq)": "Cochlea"}
QUINT = LinearSegmentedColormap.from_list("q", ["#CBC9E2", "#3F007D"])
RULES = {"q_BH_05": ("BH q < 0.05", "#1A1A1A"), "q_BH_10": ("BH q < 0.1", "#56B4E9"),
         "q_BH_05_qperm_10": ("+ permutation q < 0.1", "#E69F00"), "tier1_like": ("Tier-1-like", "#CC79A7")}
TIS = {"cochlea": "Cochlea", "bm": "Marrow"}


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


# ------------------------------------------------------------------------------------------------ a calibration
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Dense held-out series")
    d = pd.read_csv(R9 / "ed4_calibration.csv").sort_values(["dataset", "rule_order"], ascending=[False, True])
    d["grp"] = d.dataset.map(DATA)
    assert d.grp.notna().all() and len(d) == 8
    rule_w = max(text_width(r, PT_SMALL) for r in d.rule_label) + 1.2
    grp_w = max(text_width(g, PT_SMALL) for g in d.grp) + 1.6
    lab_w = grp_w + rule_w
    cols = [("FDP (%)", d.fdp_strict_pct, 40.0, "{:.1f}"), ("Sensitivity (%)", d.sensitivity_pct, 100.0, "{:.0f}"),
            ("Phase error (h)", d.median_phase_error_h, 1.0, "{:.2f}")]
    top, rp, gap = y0 + 8.4, 4.4, 2.0
    cw = (w - lab_w) / len(cols)
    ys, k = [], 0.0
    for i, g in enumerate(d.grp):
        if i and g != d.grp.iloc[i - 1]:
            k += gap
        ys.append(top + k + 0.5 * rp)
        k += rp
    for yy, r in zip(ys, d.rule_label):
        cv.text(x0 + lab_w - 0.8, yy, r, size=PT_SMALL, ha="right", va="center")
    for g in d.grp.unique():
        yy = [y for y, gg in zip(ys, d.grp) if gg == g]
        cv.text(x0, (min(yy) + max(yy)) / 2, g, size=PT_SMALL, va="center", color=TEXT2)
        cv.line([x0 + grp_w - 0.6] * 2, [min(yy) - 1.6, max(yy) + 1.6], TEXT2, 0.35)
    for j, (title, vals, vmax, fmt) in enumerate(cols):
        bx = x0 + lab_w + j * cw + 0.6
        bw = cw - 9.0
        cv.text(bx, top - 1.0, title, size=PT_SMALL, va="bottom", color=TEXT2)
        cv.line([bx, bx], [top, ys[-1] + 0.5 * rp], TEXT, 0.35)
        for yy, v in zip(ys, vals):
            ww = bw * float(np.clip(v / vmax, 0, 1))
            cv.box(bx, yy - 1.1, ww, 2.2, BAR)
            cv.text(bx + ww + 0.6, yy, fmt.format(v), size=PT_SMALL, va="center")
        assert vals.max() <= vmax


# ------------------------------------------------------------------------------------------------ b sensitivity
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Tier 1–2 sensitivity")
    s = pd.read_csv(R9 / "ed4_sensitivity.csv")
    s = s[s.rule == "tier1_or_tier2"]
    g = s.groupby(["dataset", "condition", "truth_fc_bin"])[["called", "n"]].sum()
    g["sens"] = g.called / g.n
    g = g.reset_index()
    bins = ["1.5-2", "2-3", ">=3"]
    blab = ["1.5–2", "2–3", "≥3"]
    top = y0 + 8.0
    fh = (h - (top - y0) - 9.0 - 3.0) / 2
    lab_w = 8.0
    for k, (ds, name) in enumerate(DATA.items()):
        ay = top + k * (fh + 3.0)
        ax = cv.axes(x0 + lab_w, ay, w - lab_w - 1.0, fh)
        for j, (cond, (lab, col)) in enumerate(COND.items()):
            v = [g[(g.dataset == ds) & (g.condition == cond) & (g.truth_fc_bin == b)].sens.iloc[0] for b in bins]
            ax.bar(np.arange(3) + (j - 1) * 0.26, v, width=0.24, color=col, linewidth=0.35, edgecolor=TEXT)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
        ax.set_xticks(range(3), blab if k == 1 else [""] * 3)
        clean(ax)
        tick_style(ax)
        ax.set_ylabel("Sensitivity", fontsize=PT_SMALL, labelpad=1.0)
        cv.text(x0 + lab_w + 1.0, ay + 0.2, name, size=PT_SMALL, va="top", color=TEXT2)
    cv.text(x0 + lab_w + (w - lab_w - 1.0) / 2, top + 2 * fh + 3.0 + 3.4, "True fold change", size=PT_SMALL,
            ha="center", va="top")
    xx = cv.key(x0 + lab_w, y0 + h - 1.2, [([{"kind": "box", "fc": c, "ec": TEXT, "lw": 0.35}], lab)
                                          for lab, c in COND.values()], gap=1.8)
    cv.text(x0 + lab_w - 0.6, y0 + h - 1.2, "pattern", size=PT_SMALL, ha="right", va="center", color=TEXT2)
    fits(xx, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c clock genes
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Core-clock recovery")
    c = pd.read_csv(R9 / "ed4_clock_recovery.csv")
    genes = sorted(c.gene.unique())
    combos = [(ds, cond) for ds in DATA for cond in COND]
    lab_w = max(text_width(gg, PT_SMALL, style="italic") for gg in genes) + 1.4
    top = y0 + 12.4
    q = (w - lab_w - 0.5) / len(combos)
    rp = (h - (top - y0) - 6.0) / len(genes)
    ax = cv.axes(x0 + lab_w, top, len(combos) * q, len(genes) * rp)
    for j, (ds, cond) in enumerate(combos):
        d = c[(c.dataset == ds) & (c.condition == cond)].set_index("gene").reindex(genes)
        ok = d.frac_tier1_or_2.notna()
        ax.scatter(np.full(ok.sum(), j), np.arange(len(genes))[ok.to_numpy()],
                   s=0.4 + 14.0 * d.frac_tier1_or_2[ok], c=TEXT, linewidths=0, zorder=3)
        ax.scatter(np.full((~ok).sum(), j), np.arange(len(genes))[(~ok).to_numpy()], marker="x", s=3, c=TEXT2,
                   linewidths=0.4, zorder=3)
    ax.axvline(2.5, color=GRID, lw=0.5)
    ax.set_xlim(-0.5, len(combos) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_axis_off()
    for i, gg in enumerate(genes):
        cv.text(x0 + lab_w - 0.8, top + (i + 0.5) * rp, gg, size=PT_SMALL, ha="right", va="center", style="italic")
    for j, (ds, cond) in enumerate(combos):
        cv.text(x0 + lab_w + (j + 0.5) * q, top - 0.6, COND[cond][0], size=PT_SMALL, ha="center", va="bottom",
                rotation=90, color=TEXT2)
    for k, name in enumerate(DATA.values()):
        xa, xb = x0 + lab_w + 3 * k * q + 0.3, x0 + lab_w + 3 * (k + 1) * q - 0.3
        cv.line([xa, xb], [top - 6.4, top - 6.4], TEXT, 0.35)
        cv.text((xa + xb) / 2, top - 6.8, name, size=PT_SMALL, ha="center", va="bottom")
    ky = y0 + h - 1.2
    xx = cv.key(x0 + lab_w - 4.0, ky, [([{"s": 0.4 + 14.0 * f, "fc": TEXT}], f"{f:g}") for f in (0.25, 0.5, 1.0)],
                gap=1.4)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d power per type
def fc_axis(ax, lo, hi, ticks):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    clean(ax, left=False)
    tick_style(ax)


def panel_d(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Smallest rhythm detectable per cell type")
    gp = pd.read_csv(R9 / "ed4_gene_power.csv")
    gp = gp[gp.ptt_fc == gp.ptt_fc.iloc[0]].drop_duplicates(["tissue", "abbr", "expr_bin"])
    assert "coef_var_median" in gp.columns, "gene power predates the round-9 voom-weight correction"
    ab = pd.read_csv(R9 / "ed4_abundance_power.csv").drop_duplicates(["tissue", "abbr", "method"])
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    G_LO, G_HI, A_LO, A_HI = 1.5, 40.0, 1.5, 150.0
    top = y0 + 9.6
    bw = (w - 3.0) / 2
    for b, tis in enumerate(["cochlea", "bm"]):
        bx = x0 + b * (bw + 3.0)
        types = sorted(set(ab[ab.tissue == tis].abbr) | set(gp[gp.tissue == tis].abbr), key=corder.get)
        lab_w = max(text_width(a, PT_SMALL) for a in types) + 3.0
        rp = min(2.9, (h - (top - y0) - 8.0) / len(types))
        cw = (bw - lab_w - 2.0) / 2
        axg = cv.axes(bx + lab_w, top, cw - 1.0, len(types) * rp)
        axa = cv.axes(bx + lab_w + cw + 1.0, top, cw - 1.0, len(types) * rp)
        ri = {a: i for i, a in enumerate(types)}
        for ax in (axg, axa):
            ax.set_ylim(len(types) - 0.5, -0.5)
            for i in range(len(types)):
                ax.axhline(i, color=GRID, lw=0.25, zorder=0)
        g = gp[gp.tissue == tis]
        for a, gg in g.groupby("abbr"):
            v = gg.sort_values("expr_bin").min_fc_80pct_alpha001.to_numpy()
            y = ri[a]
            fin = np.isfinite(v)
            if fin.any():
                axg.plot([v[fin].min(), min(v[fin].max(), G_HI)], [y, y], color=TEXT2, lw=0.5, zorder=1)
            axg.scatter(np.clip(v[fin], G_LO, G_HI), [y] * int(fin.sum()), s=5,
                        c=[QUINT(k / 4) for k in np.where(fin)[0]], linewidths=0, zorder=3)
            if (~fin).any():
                axg.scatter([G_HI], [y], marker=">", s=6, c=TEXT2, linewidths=0, zorder=3, clip_on=False)
        for m, (fc_, ec) in {"propeller_moderated": (TEXT, TEXT), "clr_unmoderated": ("white", TEXT)}.items():
            aa = ab[(ab.tissue == tis) & (ab.method == m)]
            v = aa.min_fc_80pct_alpha05.to_numpy()
            y = aa.abbr.map(ri).to_numpy()
            inr = np.isfinite(v) & (v <= A_HI)
            axa.scatter(v[inr], y[inr], s=6, facecolors=fc_, edgecolors=ec, linewidths=0.45, zorder=3)
            axa.scatter(np.full((~inr).sum(), A_HI), y[~inr], marker=">", s=6, facecolors=fc_, edgecolors=ec,
                        linewidths=0.45, zorder=3, clip_on=False)
        fc_axis(axg, G_LO, G_HI, [2, 5, 10, 20])
        fc_axis(axa, A_LO, A_HI, [2, 10, 100])
        for i, a in enumerate(types):
            yy = top + (i + 0.5) * rp
            cv.box(bx + lab_w - 2.0, yy - rp * 0.32, 1.2, rp * 0.64, colour[a])
            cv.text(bx + lab_w - 2.5, yy, a, size=PT_SMALL, ha="right", va="center")
        cv.text(bx, top - 4.6, TIS[tis], size=PT_SMALL, va="bottom", fontweight="bold")
        cv.text(bx + lab_w, top - 0.8, "Gene", size=PT_SMALL, va="bottom", color=TEXT2)
        cv.text(bx + lab_w + cw + 1.0, top - 0.8, "Share", size=PT_SMALL, va="bottom", color=TEXT2)
        cv.text(bx + lab_w + cw, top + len(types) * rp + 3.6, "Fold change, 80 % power", size=PT_SMALL, ha="center",
                va="top")
    ky = y0 + h - 1.2
    items = [([{"s": 5, "fc": QUINT(k / 4)} for k in range(5)], "expression Q1→Q5"),
             ([{"s": 6, "fc": TEXT}], "propeller"), ([{"s": 6, "fc": "white", "ec": TEXT, "lw": 0.45}], "CLR"),
             ([{"s": 6, "marker": ">", "fc": TEXT2}], "not reached")]
    xx = cv.key(x0 + bw + 3.0, ky, items, gap=2.0)
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e simulation
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Simulation on our data")
    s = pd.read_csv(R9 / "ed4_simulation.csv")
    runs = s.groupby(["rule", "ptt_fc_injected"])
    assert runs.size().nunique() == 1                    # 8 cell types x 3 replicates per rule and fold change
    m = runs[["power", "fdp", "median_abs_phase_error_h", "n_calls"]].mean().reset_index()
    metrics = [("power", "Power", (0, 0.3), [0, 0.1, 0.2, 0.3]), ("fdp", "FDP", (0, 0.3), [0, 0.1, 0.2, 0.3]),
               ("median_abs_phase_error_h", "Phase error (h)", (0, 4.0), [0, 2, 4])]
    top = y0 + 6.0
    lab_w = 9.0
    fh = (h - (top - y0) - 18.0 - 2 * 3.4) / 3
    fcs = sorted(s.ptt_fc_injected.unique())
    x = np.log2(np.array(fcs))
    style = {"q_BH_05": ("-", 0.9), "q_BH_10": ("-", 0.9), "q_BH_05_qperm_10": ("-", 1.6),
             "tier1_like": ((0, (2, 1.4)), 0.8)}
    for k, (col, lab, yl, yt) in enumerate(metrics):
        ay = top + k * (fh + 3.4)
        ax = cv.axes(x0 + lab_w, ay, w - lab_w - 1.0, fh)
        for rule, (rl, rc) in RULES.items():
            u = m[m.rule == rule].set_index("ptt_fc_injected").reindex(fcs)
            v = u[col].to_numpy()
            few = (u.n_calls < 1).to_numpy() & (col != "power")    # < 1 call per run: shown open, not linked
            ok = np.isfinite(v) & ~few
            assert np.max(v[ok], initial=0) <= yl[1], (col, rule)
            ax.plot(x[ok], v[ok], color=rc, ls=style[rule][0], lw=style[rule][1], marker="o",
                    ms=3.2 if rule == "q_BH_05_qperm_10" else 1.8, mew=0, zorder=3 if rule == "tier1_like" else 2)
            fo = np.isfinite(v) & few
            ax.scatter(x[fo], np.clip(v[fo], *yl), s=6, facecolors="white", edgecolors=rc, linewidths=0.5, zorder=4,
                       clip_on=False)
        ax.set_xlim(np.log2(1.2), np.log2(3.15))
        ax.set_xticks(x, [f"{f:g}" for f in fcs] if k == 2 else [""] * len(fcs))
        ax.set_ylim(*yl)
        ax.set_yticks(yt, [f"{t:g}" for t in yt])
        clean(ax)
        tick_style(ax)
        ax.set_ylabel(lab, fontsize=PT_SMALL, labelpad=1.0)
    cv.text(x0 + lab_w + (w - lab_w - 1.0) / 2, top + 3 * fh + 2 * 3.4 + 3.4, "Injected fold change", size=PT_SMALL,
            ha="center", va="top")
    ky = y0 + h - 9.4
    items = [([{"kind": "hline", "w": 2.4, "lw": style[r][1], "ls": style[r][0], "colour": c}], lab)
             for r, (lab, c) in RULES.items()]
    xx = cv.key(x0 + lab_w, ky, items[:2], gap=2.0)
    fits(xx, x0 + w, "e key 1")
    xx = cv.key(x0 + lab_w, ky + 3.4, items[2:], gap=2.0)
    fits(xx, x0 + w, "e key 2")
    xx = cv.key(x0 + lab_w, ky + 6.8, [([{"s": 6, "fc": "white", "ec": TEXT, "lw": 0.5}], "< 1 call per run")])
    fits(xx, x0 + w, "e key 3")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED)
    panel_a(cv, 0.0, 0.0, 76.0, 56.0)
    panel_b(cv, 79.0, 0.0, 50.0, 56.0)
    panel_c(cv, 132.0, 0.0, 48.0, 56.0)
    panel_d(cv, cols, 0.0, 60.0, 124.0, 110.0)
    panel_e(cv, 128.0, 60.0, 52.0, 110.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
