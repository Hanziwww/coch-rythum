"""Round 9 (D68 / D69) — Extended Data Fig. 6 (180 x 170 mm): rhythm programmes (supports Figs 2-4).

a  Per primary cell type: Tier-1 and Tier-2 calls (log scale), pi1 (Storey) and the peak-time distribution of its
   Tier-1/2 genes in 1-h bins (chronotype; faded, < 20 rhythmic genes; dot, circular mean when Rayleigh p < 0.05).
b  Robust pathway / TF-activity rhythms per resource and analysis version (primary; depth-matched; centred on the
   per-sample tissue median; both): dot area, number (log); colour, modal 3-h peak bin.
c  Peak-time distribution of the robust pathway rhythms of each version (share per 3-h bin).
d  TF activity (CollecTRI) minus TF mRNA peak per TF x cell type (both p < 0.05); line, circular mean lag of
   non-clock TFs; ticks, clock TFs.
e  Promoter motifs: circular mean peak of rhythmic motif carriers per cell type (filled, directional permutation
   q < 0.1; grey tick, expected phase) and tissue-level enrichment among rhythmic genes (Mantel-Haenszel odds ratio;
   filled, q < 0.05).

Inputs: results/r9_figures/ed6_*.csv, ed2_atlas_columns.csv, palette_celltypes.csv. Output: figures/figures/ed6_programmes_v1.*
Run: python scripts/925_ed6_programmes.py
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

from lib.figstyle import Q_SYM  # noqa: E402  (italic q, D102)

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "ed6_programmes_v1"
W_ED = 180.0
TIS = {"cochlea": "Cochlea", "bm": "Marrow"}
SHARE = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#BDBDBD", "#5A5A5A", "#141414"])
PHASE = LinearSegmentedColormap.from_list("phase", [(0.0, "#E9C46A"), (0.25, "#D9822B"), (0.5, "#8E5B9E"),
                                                    (0.75, "#2E4A7D"), (1.0, "#E9C46A")])
VERSIONS = {"primary": ("primary", "#1A1A1A", "-"), "depth_matched": ("depth-matched", "#1A1A1A", (0, (2, 1.2))),
            "centred": ("centred", "#E69F00", "-"), "centred_depth_matched": ("centred + depth", "#E69F00", (0, (2, 1.2)))}
RESOURCES = {"hallmark": "Hallmark", "reactome": "Reactome", "gobp": "GO BP", "curated": "Curated", "progeny": "PROGENy",
             "collectri": "CollecTRI"}
MOTIFS = {"Ebox_CACGTG": "E-box", "Eprime_CACGTT": "E′-box", "Dbox_TTAYGTAA": "D-box", "RORE_WAWNTRGGTCA": "RORE",
          "CRE_TGACGTCA": "CRE", "HSE_GAAnnTTCnnGAA": "HSE", "GRE_GAACAnnnTGTTC": "GRE", "NFkB_GGGRNYYYCC": "NF-κB"}
MIN_GENES = 20
SHARE_MAX = 0.3


def tick_style(ax):
    ax.tick_params(length=1.0, pad=0.6, labelsize=PT_SMALL, width=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.35)


def n_area(n):
    return 1.0 + 7.0 * np.log10(np.asarray(n, float) + 1)


# ------------------------------------------------------------------------------------------------ a calls + chronotype
def panel_a(cv, cols, x0, y0, w, h):
    cv.letter(x0, y0, "a", "Rhythm calls and their timing per cell type")
    rc = pd.read_csv(R9 / "ed6_rhythm_calls.csv")
    ch = pd.read_csv(R9 / "ed6_chronotype.csv")
    corder = dict(zip(cols.abbr, cols.column_order))
    colour = dict(zip(cols.abbr, cols.colour))
    top = y0 + 11.0
    bw = (w - 3.0) / 2
    nmax = rc.groupby("tissue").size().max()
    rp = (h - (top - y0) - 6.5) / nmax
    for b, tis in enumerate(TIS):
        bx = x0 + b * (bw + 3.0)
        r = rc[rc.tissue == tis].copy()
        types = sorted(r.abbr, key=corder.get)
        ri = {a: i for i, a in enumerate(types)}
        lab_w = max(text_width(a, PT_SMALL) for a in types) + 3.0
        cw_calls, cw_pi = 17.0, 9.0
        hm_x = bx + lab_w + cw_calls + cw_pi + 2.4
        hm_w = bw - (hm_x - bx) - 2.0
        ny = len(types) * rp
        # calls (log), Tier 1 filled / Tier 2 open; zero at the left edge
        axc = cv.axes(bx + lab_w, top, cw_calls - 1.2, ny)
        for col_, fc in (("n_tier2", "white"), ("n_tier1", TEXT)):
            v = r[col_].to_numpy().astype(float)
            axc.scatter(np.where(v > 0, v, 0.5), r.abbr.map(ri), s=7, facecolors=fc, edgecolors=TEXT, linewidths=0.5,
                        zorder=3, clip_on=False)
        axc.set_xscale("log")
        axc.set_xlim(0.5, 3000)
        axc.set_xticks([0.5, 10, 1000], ["0", "10", "10³"])
        axc.xaxis.set_minor_locator(NullLocator())
        # pi1
        axp = cv.axes(bx + lab_w + cw_calls, top, cw_pi - 1.2, ny)
        axp.scatter(r.pi1_storey, r.abbr.map(ri), s=6, c=TEXT2, linewidths=0, zorder=3)
        axp.set_xlim(0, 0.6)
        axp.set_xticks([0, 0.5], ["0", "0.5"])
        for ax in (axc, axp):
            ax.set_ylim(len(types) - 0.5, -0.5)
            for i in range(len(types)):
                ax.axhline(i, color=GRID, lw=0.25, zorder=0)
            ax.set_yticks([])
            clean(ax, left=False)
            tick_style(ax)
        # chronotype heat map
        axh = cv.axes(hm_x, top, hm_w, ny)
        c = ch[ch.tissue == tis]
        mat = np.full((len(types), 24), np.nan)
        faded = np.zeros(len(types), bool)
        for a, g in c.groupby("abbr"):
            if a not in ri:
                continue
            mat[ri[a], g.zt_bin.to_numpy()] = g.share.to_numpy()
            faded[ri[a]] = g.n_rhythmic.iloc[0] < MIN_GENES
        img = SHARE(np.clip(np.nan_to_num(mat, nan=0.0) / SHARE_MAX, 0, 1))
        img[np.isnan(mat)] = (1, 1, 1, 0)
        img[faded, :, 3] = np.where(np.isnan(mat[faded]), 0, 0.3)
        axh.imshow(img, aspect="auto", extent=(0, 24, len(types) - 0.5, -0.5), interpolation="nearest")
        cm = c.drop_duplicates("abbr")
        cm = cm[cm.abbr.isin(ri) & (cm.rayleigh_p < 0.05) & (cm.n_rhythmic >= MIN_GENES)]
        axh.scatter(cm.circ_mean_peak_zt % 24, cm.abbr.map(ri), s=6, c="#E69F00", edgecolors="white", linewidths=0.4,
                    zorder=3)
        axh.set_xlim(0, 24)
        axh.set_ylim(len(types) - 0.5, -0.5)
        axh.set_yticks([])
        zt_axis(axh, ticks=(0, 6, 12, 18, 24))
        clean(axh, left=False)
        axh.spines["bottom"].set_visible(False)
        tick_style(axh)
        axh.tick_params(axis="x", length=0, pad=2.6)
        ld_bar(cv, hm_x, top + ny + 0.1, hm_w, 0.7)
        for i, a in enumerate(types):
            yy = top + (i + 0.5) * rp
            cv.box(bx + lab_w - 1.9, yy - rp * 0.36, 1.2, rp * 0.72, colour[a])
            cv.text(bx + lab_w - 2.4, yy, a, size=PT_SMALL, ha="right", va="center")
        cv.text(bx, top - 6.0, TIS[tis], size=PT_SMALL, va="bottom", fontweight="bold")
        for xx_, t_ in ((bx + lab_w, "Calls"), (bx + lab_w + cw_calls, "π₁"), (hm_x, "Peak ZT of rhythmic genes")):
            cv.text(xx_, top - 0.8, t_, size=PT_SMALL, va="bottom", color=TEXT2)
    ky = top + rc[rc.tissue == "bm"].shape[0] * rp + 10.0
    kx = x0 + bw + 3.0 + 14.0
    xx = cv.key(kx, ky, [([{"s": 7, "fc": TEXT, "ec": TEXT}], "Tier 1"),
                         ([{"s": 7, "fc": "white", "ec": TEXT, "lw": 0.5}], "Tier 2"),
                         ([{"s": 6, "fc": "#E69F00", "ec": "white", "lw": 0.4}], "circular mean")], gap=2.0)
    axk = cv.axes(xx + 9.0, ky - 0.6, 12.0, 1.2)
    axk.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=SHARE, extent=(0, SHARE_MAX, 0, 1))
    axk.set_yticks([])
    axk.set_xticks([0, SHARE_MAX], ["0", f"≥{SHARE_MAX:g}"])
    tick_style(axk)
    cv.text(xx + 8.4, ky, "share", size=PT_SMALL, ha="right", va="center")
    fits(xx + 9.0 + 12.0 + 3.0, x0 + w, "a key")


# ------------------------------------------------------------------------------------------------ b pathway versions
def panel_b(cv, x0, y0, w, h):
    cv.letter(x0, y0, "b", "Robust pathway rhythms")
    v = pd.read_csv(R9 / "ed6_pathway_versions.csv")
    vers, res = list(VERSIONS), list(RESOURCES)
    lab_w = max(text_width(t, PT_SMALL) for t in RESOURCES.values()) + 1.4
    top = y0 + 4.6 + max(text_width(t[0], PT_SMALL) for t in VERSIONS.values()) + 1.0
    q = (w - lab_w - 1.0) / len(vers)
    bh = (h - (top - y0) - 12.0 - 4.0) / 2
    rp = bh / len(res)
    for b, tis in enumerate(TIS):
        by = top + b * (bh + 4.0)
        ax = cv.axes(x0 + lab_w, by, len(vers) * q, bh)
        d = v[v.tissue == tis]
        for _, r in d.iterrows():
            xi, yi = vers.index(r.version), res.index(r.resource)
            if r.n_robust == 0:
                ax.text(xi, yi, "–", ha="center", va="center", fontsize=PT_SMALL, color=TEXT2)
                continue
            mid = (int(str(r.modal_phase_bin_zt).split("-")[0][2:]) + 1.5) / 24
            ax.scatter([xi], [yi], s=n_area(r.n_robust) ** 1.25, c=[PHASE(mid)], edgecolors=TEXT, linewidths=0.35,
                       zorder=3)
        ax.set_xlim(-0.5, len(vers) - 0.5)
        ax.set_ylim(len(res) - 0.5, -0.5)
        ax.set_axis_off()
        for i, rr in enumerate(res):
            cv.text(x0 + lab_w - 0.8, by + (i + 0.5) * rp, RESOURCES[rr], size=PT_SMALL, ha="right", va="center")
        cv.text(x0, by - 0.4, TIS[tis], size=PT_SMALL, va="bottom", fontweight="bold")
    for j, vv in enumerate(vers):
        cv.text(x0 + lab_w + (j + 0.5) * q, top - 0.8, VERSIONS[vv][0], size=PT_SMALL, ha="center", va="bottom",
                rotation=90, color=TEXT2)
    ky = y0 + h - 8.0
    xx = cv.key(x0 + 1.0, ky, [([{"s": n_area(n) ** 1.25, "fc": TEXT2}], f"{n:,}") for n in (1, 10, 100, 1000)],
                gap=1.4)
    fits(xx, x0 + w, "b key 1")
    axk = cv.axes(x0 + 13.0, ky + 4.4, 16.0, 1.2)
    axk.imshow(np.linspace(0, 1, 240)[None, :], aspect="auto", cmap=PHASE, extent=(0, 24, 0, 1))
    axk.set_yticks([])
    axk.set_xticks([0, 12, 24], ["0", "12", "24"])
    tick_style(axk)
    cv.text(x0 + 12.4, ky + 5.0, "modal ZT", size=PT_SMALL, ha="right", va="center")


# ------------------------------------------------------------------------------------------------ c pathway peaks
def panel_c(cv, x0, y0, w, h):
    cv.letter(x0, y0, "c", "Pathway peak times")
    p = pd.read_csv(R9 / "ed6_pathway_peaks.csv")
    top = y0 + 5.2
    gap = 7.0
    fh = (h - (top - y0) - gap - 12.0) / 2
    for b, tis in enumerate(TIS):
        ay = top + b * (fh + gap)
        ax = cv.axes(x0 + 8.0, ay, w - 9.0, fh)
        for vv, (lab, col, ls) in VERSIONS.items():
            d = p[(p.tissue == tis) & (p.version == vv)].sort_values("zt_bin_start")
            if d.n.sum() == 0:
                continue
            x = d.zt_bin_start.to_numpy() + 1.5
            ax.plot(x, d.share, color=col, ls=ls, lw=0.8, marker="o", ms=1.6, mew=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        zt_axis(ax, ticks=(0, 12, 24))
        clean(ax)
        ax.spines["bottom"].set_visible(False)
        tick_style(ax)
        ax.tick_params(axis="x", length=0, pad=2.6)
        ax.set_ylabel("Share", fontsize=PT_SMALL, labelpad=0.8)
        ld_bar(cv, x0 + 8.0, ay + fh + 0.1, w - 9.0, 0.7)
        cv.text(x0 + 9.0, ay + 0.2, TIS[tis], size=PT_SMALL, va="top", fontweight="bold")
    ky = y0 + h - 4.6
    items = [([{"kind": "hline", "w": 2.4, "lw": 0.8, "ls": ls, "colour": col}], lab) for lab, col, ls in VERSIONS.values()]
    xx = cv.key(x0 + 1.0, ky, items[:2], gap=1.6)
    fits(xx, x0 + w, "c key 1")
    xx = cv.key(x0 + 1.0, ky + 3.4, items[2:], gap=1.6)
    fits(xx, x0 + w, "c key 2")


# ------------------------------------------------------------------------------------------------ d TF lags
def panel_d(cv, x0, y0, w, h):
    cv.letter(x0, y0, "d", "TF activity after mRNA")
    t = pd.read_csv(R9 / "ed6_tf_lags.csv")
    s = pd.read_csv(R9 / "ed6_tf_lag_summary.csv")
    top = y0 + 7.6
    gap = 5.0
    fh = (h - (top - y0) - gap - 6.0) / 2
    edges = np.arange(-12, 12.01, 2)
    for b, tis in enumerate(TIS):
        ay = top + b * (fh + gap)
        ax = cv.axes(x0 + 8.0, ay, w - 9.0, fh)
        nc = t[(t.tissue == tis) & ~t.clock_tf.astype(bool)].lag_h
        ax.hist(nc, bins=edges, color="#9A9A9A", edgecolor="white", linewidth=0.3)
        ck = t[(t.tissue == tis) & t.clock_tf.astype(bool)].lag_h
        ymax = np.histogram(nc, bins=edges)[0].max() * 1.25
        ax.scatter(ck, np.full(len(ck), ymax * 0.93), marker="|", s=14, c=TEXT, linewidths=0.6, zorder=3)
        m = s[(s.tissue == tis) & ~s.clock_tf.astype(bool)].circ_mean_lag_h.iloc[0]
        ax.axvline(m, color="#E69F00", lw=0.9, zorder=2)
        ax.axvline(0, color=TEXT2, lw=0.35, ls=(0, (2, 1.2)), zorder=1)
        ax.set_xlim(-12, 12)
        ax.set_ylim(0, ymax)
        ax.set_xticks([-12, -6, 0, 6, 12], ["−12", "−6", "0", "6", "12"] if b == 1 else [""] * 5)
        clean(ax)
        tick_style(ax)
        ax.set_ylabel("Pairs", fontsize=PT_SMALL, labelpad=0.8)
        cv.text(x0 + 8.0, ay - 0.6, TIS[tis], size=PT_SMALL, va="bottom", fontweight="bold")
    cv.text(x0 + 8.0 + (w - 9.0) / 2, top + 2 * fh + gap + 3.4, "Activity − mRNA peak (h)", size=PT_SMALL,
            ha="center", va="top")


# ------------------------------------------------------------------------------------------------ e motifs
def panel_e(cv, x0, y0, w, h):
    cv.letter(x0, y0, "e", "Promoter motifs")
    mp = pd.read_csv(R9 / "ed6_motif_phase.csv")
    me = pd.read_csv(R9 / "ed6_motif_enrichment.csv")
    motifs = [m for m in MOTIFS if m in set(me.motif)]
    assert set(mp.motif) <= set(motifs)
    lab_w = max(text_width(MOTIFS[m], PT_SMALL) for m in motifs) + 1.4
    top = y0 + 9.0
    or_w = 15.0
    gx = 3.2
    aw = (w - lab_w - or_w - 2 * gx - 2.0) / 2
    rp = (h - (top - y0) - 12.0) / len(motifs)
    ri = {m: i for i, m in enumerate(motifs)}
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr").colour
    for b, tis in enumerate(TIS):
        ax = cv.axes(x0 + lab_w + b * (aw + gx), top, aw, len(motifs) * rp)
        d = mp[mp.tissue == tis]
        for m in motifs:
            ax.axhline(ri[m], color=GRID, lw=0.25, zorder=0)
            e = d[d.motif == m].expected_zt.dropna()
            if len(e):
                ax.plot([e.iloc[0]] * 2, [ri[m] - 0.38, ri[m] + 0.38], color="#9A9A9A", lw=1.1, zorder=1,
                        solid_capstyle="butt")
        sig = d.q_directional_perm < 0.1
        cc = d.abbr.map(pal)
        ax.scatter(d.circ_mean_peak_zt[~sig] % 24, d.motif[~sig].map(ri), s=5, facecolors="white", edgecolors=cc[~sig],
                   linewidths=0.5, zorder=2)
        ax.scatter(d.circ_mean_peak_zt[sig] % 24, d.motif[sig].map(ri), s=9, c=cc[sig], edgecolors=TEXT,
                   linewidths=0.4, zorder=3)
        ax.set_ylim(len(motifs) - 0.5, -0.5)
        ax.set_yticks([])
        zt_axis(ax, ticks=(0, 12, 24))
        clean(ax, left=False)
        ax.spines["bottom"].set_visible(False)
        tick_style(ax)
        ax.tick_params(axis="x", length=0, pad=2.6)
        ld_bar(cv, x0 + lab_w + b * (aw + gx), top + len(motifs) * rp + 0.1, aw, 0.7)
        cv.text(x0 + lab_w + b * (aw + gx) + aw / 2, top - 0.8, TIS[tis], size=PT_SMALL, ha="center", va="bottom",
                fontweight="bold")
    axo = cv.axes(x0 + w - or_w - 2.0 + 1.0, top, or_w - 1.0, len(motifs) * rp)
    lo_or, hi_or = 0.5, 2.0
    for k, (tis, mk) in enumerate((("cochlea", "o"), ("bm", "s"))):
        d = me[me.tissue == tis].set_index("motif").reindex(motifs)
        y = np.arange(len(motifs)) + (k - 0.5) * 0.3
        o = d.mh_odds_ratio.to_numpy()
        sig = (d.q_BH < 0.05).to_numpy()
        oc = np.clip(np.where(o > 0, o, lo_or), lo_or, hi_or)
        axo.scatter(oc[sig], y[sig], marker=mk, s=8, c=TEXT, linewidths=0, zorder=3, clip_on=False)
        axo.scatter(oc[~sig], y[~sig], marker=mk, s=8, facecolors="white", edgecolors=TEXT, linewidths=0.5, zorder=3,
                    clip_on=False)
    axo.axvline(1, color=TEXT2, lw=0.35, ls=(0, (2, 1.2)))
    axo.set_xscale("log")
    axo.set_xlim(lo_or, hi_or)
    axo.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
    axo.xaxis.set_minor_locator(NullLocator())
    axo.set_ylim(len(motifs) - 0.5, -0.5)
    axo.set_yticks([])
    for m in motifs:
        axo.axhline(ri[m], color=GRID, lw=0.25, zorder=0)
    clean(axo, left=False)
    tick_style(axo)
    cv.text(x0 + w - or_w - 1.0, top - 0.8, "Odds ratio", size=PT_SMALL, va="bottom", color=TEXT2)
    for m in motifs:
        cv.text(x0 + lab_w - 0.8, top + (ri[m] + 0.5) * rp, MOTIFS[m], size=PT_SMALL, ha="right", va="center")
    ky = y0 + h - 1.4
    items = [([{"s": 9, "fc": TEXT2, "ec": TEXT, "lw": 0.4}], f"{Q_SYM} < 0.1"),
             ([{"kind": "vline", "colour": "#9A9A9A", "lw": 1.1, "h": 1.8}], "expected"),
             ([{"s": 8, "fc": TEXT2}], "cochlea"), ([{"s": 8, "marker": "s", "fc": TEXT2}], "marrow")]
    xx = cv.key(x0 + 1.0, ky, items, gap=1.8)
    fits(xx, x0 + w, "e key")


# ------------------------------------------------------------------------------------------------ main
def main():
    cols = pd.read_csv(R9 / "ed2_atlas_columns.csv")
    pal = pd.read_csv(R9 / "palette_celltypes.csv").set_index("abbr")["colour"]
    cols["colour"] = cols.abbr.map(pal)
    assert cols.colour.notna().all() and len(cols) == 35
    cv = Canvas(width=W_ED)
    panel_a(cv, cols, 0.0, 0.0, W_ED, 82.0)
    panel_b(cv, 0.0, 85.0, 48.0, 85.0)
    panel_c(cv, 52.0, 85.0, 52.0, 43.0)
    panel_d(cv, 52.0, 130.0, 52.0, 40.0)
    panel_e(cv, 108.0, 85.0, 72.0, 85.0)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
