"""Round 9 (D68 / D69) — Extended Data Fig. 7 (180 x 170 mm): heat shock and rhythm drivers (supports Fig. 3).

a  Per-sample dissociation-stress effect (sample term of the stress-profile model) across the day, both tissues.
b  Glucocorticoid (GC) checks: circular mean peak of GC-response genes and of rhythmic genes that lose (adx-dependent)
   or keep their rhythm after adrenalectomy (GSE88954; filled, Rayleigh p < 0.05; grey tick, canonical peak), and the
   odds that night-peaking rhythmic genes are adx-dependent (vs day-peaking; filled, Fisher p < 0.05).
c  Driver evidence among rhythmic calls vs background (Cochran-Mantel-Haenszel odds ratio over expression deciles):
   per cell type (colour; filled, q < 0.05) and all calls (diamond).
d  Phase-dependent driver flags vs a peak-shuffled null per cell type (Tier 1-2): colour, observed / null share;
   outlined, permutation p < 0.05.
e  Baboon heat-shock programme phase in every tissue (filled, p < 0.05): clock-aligned (purple) and raw ZT (grey);
   line, mouse heat-shock phase (aligned).
f  Share of each cell type's rhythmic genes that are members of a systemic-cue programme peaking in its common phase
   (faded, < 20 rhythmic genes).

Inputs: results/r9_figures/ed7_*.csv, f3f_baboon_heatshock.csv, palettes. Output: figures/figures/ed7_heat_shock_v1.*
Run: python scripts/926_ed7_heat_shock.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import (GRID, PT_SMALL, TEXT, TEXT2, Canvas, clean, fits, ld_bar, text_width,  # noqa: E402
                          zt_axis)

from lib.figstyle import P_SYM, Q_SYM, R_SYM  # noqa: E402,F401  (italic P / q / r, D99, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed7_heat_shock_v1"
W_ED = 180.0
TIS = {"cochlea": ("Cochlea", "#0072B2"), "bm": ("Marrow", "#D55E00")}
BABOON = "#6B4C9A"
DIV = LinearSegmentedColormap.from_list("div", ["#B35806", "#F1A340", "#F7F7F7", "#998EC3", "#542788"])
GC_SETS = {"glucocorticoid_response (curated)": "GC response", "Per1": "Per1", "Nfil3": "Nfil3",
           "GSE88954 GC-dependent rhythmic": "adx-dependent", "GSE88954 GC-independent rhythmic": "adx-independent"}
EVIDENCE = {"dd_rhythmic_2plus_organs": "DD-rhythmic", "gc_dependent_GSE88954": "adx-dependent",
            "promoter_clock_element": "clock element", "promoter_HSE": "HSE", "promoter_GRE": "GRE"}
FLAGS = {"dd_support": "DD phase", "clock_motif_phase": "motif phase", "clock_controlled": "clock-controlled",
         "gc_driven": "GC-driven"}
CUES = {"heat_shock_hsf1": "heat shock", "glucocorticoid_response": "glucocorticoid", "adrenergic_camp": "adrenergic",
        "hypoxia": "hypoxia", "feeding_insulin_mtor": "feeding", "temperature_cold_inducible": "cold-inducible"}
MIN_GENES = 20


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def phase_axis(ax):
    zt_axis(ax, ticks=(0, 12, 24))
    ax.spines["bottom"].set_visible(False)
    tick_style(ax)
    ax.tick_params(axis="x", length=0, pad=2.6)


# ------------------------------------------------------------------------------------------------ a handling
def panel_a(cv, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Dissociation stress per sample")
    d = pd.read_csv(R9 / "ed7_handling_samples.csv")
    top = y0 + 6.0
    ph = h - (top - y0) - 12.0
    ax = cv.axes(x0 + 9.0, top, w - 10.0, ph)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0)
    for tis, (lab, col) in TIS.items():
        s = d[d.tissue == tis].sort_values("zt")
        ax.plot(s.zt, s.sample_effect_stress, color=col, lw=0.6, alpha=0.6, zorder=1)
        ax.scatter(s.zt, s.sample_effect_stress, s=9, c=col, linewidths=0, zorder=3)
    ax.set_ylim(-0.6, 0.4)
    ax.set_yticks([-0.5, -0.25, 0, 0.25], ["−0.5", "−0.25", "0", "0.25"])
    clean(ax)
    phase_axis(ax)
    ax.set_xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
    ax.set_ylabel("Sample effect", fontsize=PT_SMALL, labelpad=1.0)
    ld_bar(cv, x0 + 9.0, top + ph + 0.1, w - 10.0, 0.7)
    xx = cv.key(x0 + 9.0, y0 + h - 1.4, [([{"s": 9, "fc": c}], lab) for lab, c in TIS.values()], gap=2.0)
    fits(xx, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b glucocorticoids
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Glucocorticoid checks")
    g = pd.read_csv(R9 / "ed7_gc_phases.csv")
    o = pd.read_csv(R9 / "ed7_gc_dependence.csv")
    sets = list(GC_SETS)
    assert set(g.gene_set) == set(sets)
    lab_w = max(text_width(GC_SETS[k], PT_SMALL) for k in sets) + 1.4
    top = y0 + 6.0
    or_w = 15.0
    aw = w - lab_w - or_w - 2.6
    rp = (h - (top - y0) - 12.0) / len(sets)
    ax = cv.axes(x0 + lab_w, top, aw, len(sets) * rp)
    ri = {k: i for i, k in enumerate(sets)}
    for k in sets:
        ax.axhline(ri[k], color=GRID, lw=0.25, zorder=0)
        cz = g[g.gene_set == k].canonical_peak_zt.dropna()
        if len(cz):
            ax.plot([cz.iloc[0]] * 2, [ri[k] - 0.38, ri[k] + 0.38], color="#9A9A9A", lw=1.1, zorder=1,
                    solid_capstyle="butt")
    for j, (tis, (lab, col)) in enumerate(TIS.items()):
        s = g[g.tissue == tis]
        y = s.gene_set.map(ri) + (j - 0.5) * 0.3
        sig = s.rayleigh_p < 0.05
        mk = "o" if tis == "cochlea" else "s"
        ax.scatter(s.circ_mean_peak_zt[sig] % 24, y[sig], marker=mk, s=4 + 12 * s.resultant[sig], c=col, linewidths=0,
                   zorder=3)
        ax.scatter(s.circ_mean_peak_zt[~sig] % 24, y[~sig], marker=mk, s=4 + 12 * s.resultant[~sig],
                   facecolors="white", edgecolors=col, linewidths=0.5, zorder=3)
    ax.set_ylim(len(sets) - 0.5, -0.5)
    ax.set_yticks([])
    clean(ax, left=False)
    phase_axis(ax)
    ld_bar(cv, x0 + lab_w, top + len(sets) * rp + 0.1, aw, 0.7)
    for k in sets:
        cv.text(x0 + lab_w - 0.8, top + (ri[k] + 0.5) * rp, GC_SETS[k], size=PT_SMALL, ha="right", va="center",
                style="italic" if k in ("Per1", "Nfil3") else "normal")
    axo = cv.axes(x0 + w - or_w, top + 1.5 * rp, or_w - 1.0, 2 * rp)
    for j, (tis, (lab, col)) in enumerate(TIS.items()):
        r = o[o.tissue == tis].iloc[0]
        filled = r.p_fisher < 0.05
        axo.scatter([r.odds_ratio_night_gc_dependent], [j], marker="o" if tis == "cochlea" else "s", s=12,
                    facecolors=col if filled else "white", edgecolors=col, linewidths=0.6, zorder=3)
    axo.axvline(1, color=TEXT2, lw=0.35, ls=(0, (2, 1.2)))
    axo.set_xscale("log")
    axo.set_xlim(0.5, 2)
    axo.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
    axo.xaxis.set_minor_locator(NullLocator())
    axo.set_ylim(1.6, -0.6)
    axo.set_yticks([])
    clean(axo, left=False)
    tick_style(axo)
    cv.text(x0 + w - or_w, top + 1.5 * rp - 0.8, "Night OR", size=PT_SMALL, va="bottom", color=TEXT2)
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, [([{"s": 9, "fc": TIS["cochlea"][1]}], "cochlea"),
                                         ([{"s": 9, "marker": "s", "fc": TIS["bm"][1]}], "marrow"),
                                         ([{"kind": "vline", "colour": "#9A9A9A", "lw": 1.1, "h": 1.8}], "canonical")],
                gap=1.8)
    fits(xx, x0 + w, "b key")


# ------------------------------------------------------------------------------------------------ c driver evidence
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Driver evidence in rhythmic calls")
    e = pd.read_csv(R9 / "ed7_driver_enrichment.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr").colour
    ev = list(EVIDENCE)
    lab_w = max(text_width(EVIDENCE[k], PT_SMALL) for k in ev) + 1.4
    top = y0 + 6.0
    gx = 4.0
    aw = (w - lab_w - gx - 2.0) / 2
    rp = (h - (top - y0) - 9.0) / len(ev)
    ri = {k: i for i, k in enumerate(ev)}
    LO, HI = 0.25, 4.0
    rng = np.random.default_rng(3)
    for b, (tis, (lab, col)) in enumerate(TIS.items()):
        ax = cv.axes(x0 + lab_w + b * (aw + gx), top, aw, len(ev) * rp)
        d = e[(e.tissue == tis) & e.cmh_odds_ratio.notna() & (e.cmh_odds_ratio > 0)]
        for k in ev:
            ax.axhline(ri[k], color=GRID, lw=0.25, zorder=0)
        ty = d[d.cell_type != "ALL"]
        y = ty.evidence.map(ri) + rng.uniform(-0.22, 0.22, len(ty))
        sig = (ty.cmh_q < 0.05).to_numpy()
        cc = ty.abbr.map(pal).to_numpy()
        xv = np.clip(ty.cmh_odds_ratio, LO, HI).to_numpy()
        ax.scatter(xv[~sig], y[~sig], s=5, facecolors="white", edgecolors=cc[~sig], linewidths=0.45, zorder=2)
        ax.scatter(xv[sig], y[sig], s=6, c=cc[sig], linewidths=0, zorder=3)
        al = d[d.cell_type == "ALL"]
        asig = (al.cmh_q < 0.05).to_numpy()
        ax.scatter(np.clip(al.cmh_odds_ratio, LO, HI), al.evidence.map(ri), marker="D", s=12,
                   facecolors=np.where(asig, TEXT, "white"), edgecolors=TEXT, linewidths=0.6, zorder=4)
        ax.axvline(1, color=TEXT2, lw=0.35, ls=(0, (2, 1.2)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(LO, HI)
        ax.set_xticks([0.25, 1, 4], ["0.25", "1", "4"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(len(ev) - 0.5, -0.5)
        ax.set_yticks([])
        clean(ax, left=False)
        tick_style(ax)
        cv.text(x0 + lab_w + b * (aw + gx) + aw / 2, top - 0.8, lab, size=PT_SMALL, ha="center", va="bottom",
                fontweight="bold")
    for k in ev:
        cv.text(x0 + lab_w - 0.8, top + (ri[k] + 0.5) * rp, EVIDENCE[k], size=PT_SMALL, ha="right", va="center")
    cv.text(x0 + lab_w + aw + gx / 2, top + len(ev) * rp + 3.6, "Odds ratio", size=PT_SMALL, ha="center", va="top")
    xx = cv.key(x0 + 1.0, y0 + h - 1.4, [([{"s": 12, "marker": "D", "fc": TEXT}], "all calls"),
                                         ([{"s": 6, "fc": TEXT2}], f"cell type, {Q_SYM} < 0.05")], gap=1.8)
    fits(xx, x0 + w, "c key")


# ------------------------------------------------------------------------------------------------ d phase null
def panel_d(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "d", "Driver flags vs phase null")
    d = pd.read_csv(R9 / "ed7_driver_phase_null.csv")
    d = d[d.tier_set == "tier1/2"]
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    rows = []
    for tis in TIS:
        s = d[d.tissue == tis]
        types = sorted(s[s.cell_type != "ALL"].abbr.dropna().unique(), key=corder.get)
        rows += [(tis, "ALL")] + [(tis, a) for a in types]
    flags = list(FLAGS)
    top = y0 + 4.6 + max(text_width(FLAGS[f], PT_SMALL) for f in flags) + 1.0
    lab_w = max(text_width(a, PT_SMALL) for _, a in rows) + 3.0
    q = (w - lab_w - 1.0) / len(flags)
    gap = 1.6
    rp = (h - (top - y0) - gap - 12.0) / len(rows)
    ax = cv.axes(x0 + lab_w, top, len(flags) * q, len(rows) * rp + gap)
    yy, y = {}, 0.0
    for i, (tis, a) in enumerate(rows):
        if i and tis != rows[i - 1][0]:
            y += gap / rp
        yy[(tis, a)] = y
        y += 1
    for (tis, a), yv in yy.items():
        s = d[(d.tissue == tis) & ((d.cell_type == "ALL") if a == "ALL" else (d.abbr == a))]
        for _, r in s.iterrows():
            v = r.obs_over_null
            if not np.isfinite(v):
                continue
            sig = r.p_perm < 0.05
            ax.scatter([flags.index(r.flag)], [yv], s=14, c=[DIV(np.clip((v - 0.5) / 1.0, 0, 1))],
                       edgecolors=TEXT if sig else "none", linewidths=0.8 if sig else 0, zorder=3)
        ytxt = top + (yv + 0.5) * rp
        if a == "ALL":
            cv.text(x0 + lab_w - 0.8, ytxt, f"{TIS[tis][0]} (all)", size=PT_SMALL, ha="right", va="center",
                    fontweight="bold")
        else:
            cv.box(x0 + lab_w - 1.9, ytxt - rp * 0.34, 1.2, rp * 0.68, colour[a])
            cv.text(x0 + lab_w - 2.4, ytxt, a, size=PT_SMALL, ha="right", va="center")
    ax.set_xlim(-0.5, len(flags) - 0.5)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_axis_off()
    for j, f in enumerate(flags):
        cv.text(x0 + lab_w + (j + 0.5) * q, top - 0.8, FLAGS[f], size=PT_SMALL, ha="center", va="bottom", rotation=90,
                color=TEXT2)
    ky = y0 + h - 4.4
    axk = cv.axes(x0 + 14.0, ky - 0.6, 14.0, 1.2)
    axk.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=DIV, extent=(0.5, 1.5, 0, 1))
    axk.set_yticks([])
    axk.set_xticks([0.5, 1, 1.5], ["≤0.5", "1", "≥1.5"])
    tick_style(axk)
    cv.text(x0 + 13.4, ky, "obs / null", size=PT_SMALL, ha="right", va="center")
    xx = cv.key(x0 + 31.0, ky, [([{"s": 14, "fc": DIV(0.5), "ec": TEXT, "lw": 0.8}], f"{P_SYM} < 0.05")])
    fits(xx, x0 + w, "d key")


# ------------------------------------------------------------------------------------------------ e baboon
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Baboon tissues")
    b = pd.read_csv(R9 / "f3f_baboon_heatshock.csv")
    mouse = b[b.baboon_tissue.str.startswith("SUMMARY mouse")].iloc[0]
    t = b[~b.baboon_tissue.str.startswith("SUMMARY")].copy()
    assert len(t) == 45
    t = t.sort_values("aligned").reset_index(drop=True)
    lab_w = max(text_width(a, PT_SMALL) for a in t.baboon_tissue) + 1.4
    top = y0 + 6.0
    rp = (h - (top - y0) - 13.0) / len(t)
    ax = cv.axes(x0 + lab_w, top, w - lab_w - 1.0, len(t) * rp)
    for i in range(len(t)):
        ax.axhline(i, color=GRID, lw=0.2, zorder=0)
    ax.axvline(mouse.aligned, color=TEXT, lw=0.7, ls=(0, (2, 1.2)), zorder=1)
    sig = t.p < 0.05
    ax.scatter(t.peak_zt % 24, t.index, s=4, c="#B0B0B0", linewidths=0, zorder=2)
    ax.scatter(t.aligned[sig] % 24, t.index[sig], s=8, c=BABOON, linewidths=0, zorder=3)
    ax.scatter(t.aligned[~sig] % 24, t.index[~sig], s=7, facecolors="white", edgecolors=BABOON, linewidths=0.5,
               zorder=3)
    ax.set_ylim(len(t) - 0.5, -0.5)
    ax.set_yticks([])
    clean(ax, left=False)
    phase_axis(ax)
    ld_bar(cv, x0 + lab_w, top + len(t) * rp + 0.1, w - lab_w - 1.0, 0.7)
    for i, r in t.iterrows():
        cv.text(x0 + lab_w - 0.8, top + (i + 0.5) * rp, r.baboon_tissue, size=PT_SMALL, ha="right", va="center")
    cv.text(x0 + lab_w + (w - lab_w - 1.0) / 2, top + len(t) * rp + 4.4, "Phase (h)", size=PT_SMALL, ha="center",
            va="top")
    ky = y0 + h - 1.4
    xx = cv.key(x0 + 1.0, ky, [([{"s": 8, "fc": BABOON}], "aligned"), ([{"s": 4, "fc": "#B0B0B0"}], "raw"),
                               ([{"kind": "hline", "w": 2.4, "lw": 0.7, "ls": (0, (2, 1.2))}], "mouse")], gap=1.6)
    fits(xx, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ f systemic cues
def panel_f(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "f", "Systemic-cue share of rhythms")
    a = pd.read_csv(R9 / "ed7_systemic_attribution.csv")
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    cues = list(CUES)
    assert set(a.set) == set(cues)
    rows = [(tis, ab) for tis in TIS for ab in sorted(a[a.tissue == tis].abbr.unique(), key=corder.get)]
    top = y0 + 4.6 + max(text_width(CUES[c], PT_SMALL) for c in cues) + 1.0
    lab_w = max(text_width(ab, PT_SMALL) for _, ab in rows) + 3.0
    q = (w - lab_w - 1.0) / len(cues)
    gap = 1.6
    rp = (h - (top - y0) - gap - 9.0) / len(rows)
    ax = cv.axes(x0 + lab_w, top, len(cues) * q, len(rows) * rp + gap)
    yy, y = {}, 0.0
    for i, (tis, ab) in enumerate(rows):
        if i and tis != rows[i - 1][0]:
            y += gap / rp
        yy[(tis, ab)] = y
        y += 1
    for (tis, ab), yv in yy.items():
        s = a[(a.tissue == tis) & (a.abbr == ab)]
        few = s.n_rhythmic_genes.iloc[0] < MIN_GENES
        for _, r in s.iterrows():
            if r.frac_of_rhythmic_genes <= 0:
                continue
            ax.scatter([cues.index(r.set)], [yv], s=0.8 + 40.0 * np.sqrt(min(r.frac_of_rhythmic_genes, 0.5) / 0.5),
                       c=TEXT, alpha=0.3 if few else 1.0, linewidths=0, zorder=3)
        ytxt = top + (yv + 0.5) * rp
        cv.box(x0 + lab_w - 1.9, ytxt - rp * 0.34, 1.2, rp * 0.68, colour[ab])
        cv.text(x0 + lab_w - 2.4, ytxt, ab, size=PT_SMALL, ha="right", va="center")
    ax.set_xlim(-0.5, len(cues) - 0.5)
    ax.set_ylim(y - 0.5, -0.5)
    ax.set_axis_off()
    for j, c in enumerate(cues):
        cv.text(x0 + lab_w + (j + 0.5) * q, top - 0.8, CUES[c], size=PT_SMALL, ha="center", va="bottom", rotation=90,
                color=TEXT2)
    for tis in TIS:
        ys = [v for (t_, _), v in yy.items() if t_ == tis]
        cv.text(x0 + w - 0.2, top + (min(ys) + 0.5) * rp, TIS[tis][0], size=PT_SMALL, ha="right", va="center",
                color=TEXT, fontweight="bold")
    xx = cv.key(x0 + 1.0, y0 + h - 1.4,
                [([{"s": 0.8 + 40.0 * np.sqrt(f / 0.5), "fc": TEXT}], f"{f:g}") for f in (0.01, 0.1, 0.5)], gap=1.4)
    fits(xx, x0 + w, "f key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED)
    panel_a(cv, 0.0, 0.0, 52.0, 50.0)
    panel_b(cv, 56.0, 0.0, 66.0, 50.0)
    panel_c(cv, 126.0, 0.0, 54.0, 50.0)
    panel_d(cv, cols, 0.0, 54.0, 56.0, 116.0)
    panel_e(cv, 60.0, 54.0, 50.0, 116.0)
    panel_f(cv, cols, 114.0, 54.0, 66.0, 116.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
