"""Round 10 (D143 / D144) — the five main figures of the Science Research Article, composed from the panels of the
Nature Immunology figures (same data tables, same panel code; nothing is re-analysed) in Science style
(COCH_FIG_STYLE=science: upper-case panel letters assigned per composite, American spelling; written to
figures/science/ under the names below). 180 mm wide, 5-7 pt, as the source panels.

Fig. 1 sci_fig1_compartment_v1   A study design (910 a), B joint atlas (910 b), C immune subtypes (910 d),
                                 D hematopoietic cells per sample (920 f), E neutrophils in the cochlear fluids (914 d)
Fig. 2 sci_fig2_local_supply_v2  A-C marrow-cochlea route, neutrophils at the cochlea, neutrophil states (914 a-c),
                                 D check of the hand tracing (941 data, drawn here), E-I parabiosis (914 e-i)
Fig. 3 sci_fig3_daily_rhythms_v1 A day-night timetable, B timed shares, C independent flow cohort (912 a-c)
Fig. 4 sci_fig4_release_time_v1  A-H the neutrophil split (913, all panels, layout of the Nature Immunology Fig. 6)
Fig. 5 sci_fig5_tissue_time_v1   A clock offsets and B single-cell clock tracking (Nature Immunology Fig. 2b and 2d data,
                                 drawn here), C heat-shock genes (911 b), D mouse night = baboon day (911 f), E-F the
                                 independent noise cohort (916 e, f)
Run: COCH_ROOT=D:/coch-rythum COCH_FIG_STYLE=science python scripts/970_science_figures.py [1 2 3 4 5]
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("COCH_FIG_STYLE", "science")      # before lib.figstyle is imported (style read at import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import figdata as fd  # noqa: E402
from lib.figstyle import (AXIS, PT, PT_SMALL, STYLE, TEXT, TEXT2, Canvas, clean, fits, minor,  # noqa: E402
                          text_width)

assert STYLE == "science", "run with COCH_FIG_STYLE=science"
ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
PANELS = ROOT / "figures" / "panels"
OUT = ROOT / "figures" / "science"
C_STRUCT, C_HAEM = fd.GROUP_COLOUR["structural"], fd.GROUP_COLOUR["haematopoietic"]
_MODS = {}


def mod(name):
    """A Nature Immunology figure script as a module (its panel functions; main() is not run)."""
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(f"m{name[:3]}", str(ROOT / "scripts" / f"{name}.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[name] = m
    return _MODS[name]


def save(cv, name):
    for p in cv.save(OUT, name):
        print(f"[fig] {p.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------------ Fig. 1
def sci_fig1():
    m910, m920, m914 = mod("910_fig1_atlas"), mod("920_ed1_qc"), mod("914_fig4_immune_channel")
    pal, u = m910.load()
    img = m910.banner()
    h_a = 1.0 + img.shape[0] * 180.0 / img.shape[1]
    r2 = h_a + 1.5
    r3 = r2 + 42.5 + 1.5
    cv = Canvas(180.0, r3 + 75.0)
    cv.relabel = ["A"]
    m910.panel_a(cv, 0.0, 0.0, cv.w)
    cv.relabel = ["B"]
    m910.panel_b(cv, u, 0.0, r2, 88.0, 42.5)
    cv.relabel = ["C"]
    m910.panel_d(cv, u, pal, 90.0, r2, 90.0, 42.5)
    cv.relabel = ["D"]
    m920.panel_f(cv, 0.0, r3, 54.0, 60.0)
    cv.relabel = ["E"]
    m914.panel_fluids(cv, 56.0, r3, 124.0, 75.0, tile=16.0)
    save(cv, "sci_fig1_compartment_v1")


# ------------------------------------------------------------------------------------------------ Fig. 2
def panel_trace(cv, x0, y0, w, h):
    """Check of the authors' hand tracing (941): trace check of both render views and the on-ridge share of the traces
    against 999 random placements."""
    m948 = mod("948_si_imaging")
    cv.letter(x0, y0, "d", "Check of the vessel tracing")
    lg = pd.read_csv(R9 / "image_panels_log.csv").set_index("panel")
    tn = pd.read_csv(R9 / "simg_trace_null.csv")
    side, gut = (w - 1.0) / 2, 1.0
    for k in (1, 2):
        um = float(lg.loc[f"vessels_view{k}_annotated", "um_per_px"])
        a = np.asarray(Image.open(PANELS / f"trace_check_view{k}.png").convert("RGB"))[:900, :900]
        m948.tile(cv, x0 + (k - 1) * (side + gut), y0 + 5.4, side, a, um, 100, f"View {k}")
    yk = y0 + 5.4 + side + 2.2
    items = [([{"kind": "box", "fc": "#5F5F5F"}], "vessel ridge"), ([{"kind": "box", "fc": "#28C846"}], "on ridge"),
             ([{"kind": "box", "fc": "#E62828"}], "off ridge"), ([{"kind": "box", "fc": "#466EC8"}], "broad paint")]
    x = cv.key(x0, yk, items, gap=1.4)
    fits(x, x0 + w + 0.5, "trace key")
    top = yk + 4.2
    hh = y0 + h - top - 6.0
    for k in (1, 2):
        g = tn[tn.view == k]
        obs = 100 * float(g.thin_on_ridge_share.iloc[0])
        ax = cv.axes(x0 + 8.0 + (k - 1) * (side + gut), top, side - 11.0, hh)
        ax.hist(100 * g.on_ridge_share, bins=np.arange(15, 101, 2.5), color="#9A9A9A", lw=0)
        ax.axvline(obs, color=TEXT, lw=0.9)
        ax.text(obs - 1.5, ax.get_ylim()[1] * 0.97, f"traced\n{obs:.0f}%", fontsize=PT_SMALL, ha="right", va="top")
        ax.set_xlim(15, 100)
        ax.set_xticks([20, 60, 100])
        minor(ax, x=True, y=False)
        clean(ax)
        ax.tick_params(labelsize=PT_SMALL)
        ax.set_xlabel(f"On ridge (%), view {k}", fontsize=PT_SMALL, labelpad=0.8)
        if k == 1:
            ax.set_ylabel("Random placements", fontsize=PT_SMALL, labelpad=0.8)


def sci_fig2():
    m914 = mod("914_fig4_immune_channel")
    cv = Canvas(180.0, 170.0)
    cv.relabel = ["A", "B", "C"]
    m914.panel_imaging(cv, 0.0, 0.0, 96.0, 85.0)
    cv.relabel = ["D"]
    panel_trace(cv, 98.0, 0.0, 82.0, 85.0)
    cv.relabel = ["E"]
    m914.panel_design(cv, 0.0, 87.5, 40.0, 42.0)
    cv.relabel = ["F"]
    m914.panel_raw(cv, 42.0, 87.5, 138.0, 42.0)
    cv.relabel = ["G"]
    m914.panel_pairs(cv, 0.0, 132.0, 72.0, 38.0)                  # D151: liver and brain added (914 v4)
    cv.relabel = ["H"]
    m914.panel_perilymph(cv, 74.0, 132.0, 40.0, 38.0)
    cv.relabel = ["I"]
    m914.panel_celltypes(cv, 116.0, 132.0, 64.0, 38.0)            # D151: published blood band
    save(cv, "sci_fig2_local_supply_v2")


# ------------------------------------------------------------------------------------------------ Fig. 3
def sci_fig3():
    m912 = mod("912_fig5_immune_timetable")
    cv = Canvas(180.0, 117.0)
    cv.relabel = ["A"]
    m912.panel_timetable(cv, 0.0, 0.0, 64.0, 82.0)
    cv.relabel = ["B"]
    m912.panel_b(cv, 66.0, 0.0, cv.w - 66.0, 82.0)
    cv.relabel = ["C"]
    m912.panel_flow(cv, 0.0, 84.0, cv.w, 32.0)
    save(cv, "sci_fig3_daily_rhythms_v1")


# ------------------------------------------------------------------------------------------------ Fig. 4
def sci_fig4():
    m913 = mod("913_fig6_timer_split")
    cv = Canvas()
    for f, box, letter in [(m913.panel_b, (0.0, 0.0, 118.0, 60.0), "A"), (m913.panel_c, (120.0, 0.0, 60.0, 60.0), "B"),
                           (m913.panel_d, (0.0, 62.0, 64.0, 52.0), "C"), (m913.panel_e, (66.0, 62.0, 62.0, 52.0), "D"),
                           (m913.panel_f, (130.0, 62.0, 50.0, 52.0), "E"), (m913.panel_g, (0.0, 116.0, 40.0, 54.0), "F"),
                           (m913.panel_h, (42.0, 116.0, 62.0, 54.0), "G"),
                           (m913.panel_i, (106.0, 116.0, 74.0, 54.0), "H")]:
        cv.relabel = [letter]
        f(cv, *box)
    save(cv, "sci_fig4_release_time_v1")


# ------------------------------------------------------------------------------------------------ Fig. 5
def panel_offsets(cv, x0, y0, w, h):
    """Clock offset of each cochlear structural cell type from the canonical light-entrained phase (gene-bootstrap 95%
    interval), the tissue pools and the independent reference series (Nature Immunology Fig. 2b data)."""
    cv.letter(x0, y0, "a", "Light-entrained structural clocks")
    c = pd.read_csv(R9 / "f2c_clock_offsets.csv")
    m = c[c.in_main_panel.astype(bool)].sort_values("plot_order").reset_index(drop=True)
    grp = np.where(m.group.str.endswith("structural"), 0, np.where(m.group.str.contains("pool"), 1, 2))
    ypos = np.arange(len(m)) + grp * 0.6                     # small gaps between structural types, pools, references
    lab_w = max(text_width(s, PT_SMALL) for s in m.label_short) + 1.2
    left = x0 + lab_w + 0.6
    ax = cv.axes(left, y0 + 6.0, x0 + w - left, h - 6.0 - 7.2)
    col = {0: C_STRUCT, 1: TEXT, 2: "#7F7F7F"}
    for yv, g, r in zip(ypos, grp, m.itertuples()):
        assert -8 <= r.low_h and r.high_h <= 8, r.label_short
        ax.plot([r.low_h, r.high_h], [yv, yv], color=col[g], lw=0.6, solid_capstyle="butt", zorder=2)
        ax.scatter([r.estimate_h], [yv], s=7, color=col[g], linewidths=0, zorder=3)
    ax.axvline(0, color=TEXT2, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ax.set_ylim(ypos.max() + 0.6, -0.6)
    ax.set_yticks(ypos, m.label_short)
    ax.tick_params(axis="y", length=0, pad=0.8, labelsize=PT_SMALL)
    ax.set_xlim(-8, 8)
    ax.set_xticks([-8, -4, 0, 4, 8])
    minor(ax, x=True, y=False)
    ax.tick_params(axis="x", labelsize=PT_SMALL)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Offset from the canonical phase (h)", fontsize=PT_SMALL, labelpad=0.8)


def panel_tracking(cv, x0, y0, w, h):
    """Clock tracking of every cochlear cell type: resultant length of the single-cell clock offsets across the six
    samples (asterisk, exact time-relabelling P <= 0.05; dot area, cells) (Nature Immunology Fig. 2d data)."""
    cv.letter(x0, y0, "b", "Single-cell clock tracking")
    e = pd.read_csv(R9 / "f2e_tracking_strength.csv")
    e = e[e.tissue == "cochlea"]
    e = pd.concat([e[e.structural.astype(bool)].sort_values("order"), e[~e.structural.astype(bool)].sort_values("order")])
    ns = int(e.structural.astype(bool).sum())
    ypos = np.r_[np.arange(ns), np.arange(ns, len(e)) + 0.9]
    lab_w = max(text_width(s, PT_SMALL) for s in e.label_sig) + 1.2
    left = x0 + lab_w + 0.6
    top = y0 + 9.0
    ax = cv.axes(left, top, x0 + w - left - 1.0, h - 9.0 - 7.2)
    sz = 1.5 + 14.0 * np.sqrt(e.n_cells_used / e.n_cells_used.max())
    for yv, r, s_ in zip(ypos, e.itertuples(), sz):
        colour = C_STRUCT if r.structural else C_HAEM
        ax.plot([0, r.offset_resultant], [yv, yv], color=colour, lw=0.45, zorder=2)
        ax.scatter([r.offset_resultant], [yv], s=s_, color=colour, linewidths=0, zorder=3)
    ax.set_ylim(ypos.max() + 0.6, -0.6)
    ax.set_yticks(ypos, e.label_sig)
    ax.tick_params(axis="y", length=0, pad=0.8, labelsize=PT_SMALL)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    minor(ax, x=True, y=False)
    ax.tick_params(axis="x", labelsize=PT_SMALL)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Clock tracking (resultant length)", fontsize=PT_SMALL, labelpad=0.8)
    x = cv.key(x0 + 1.0, y0 + 5.2, [([{"kind": "mark", "s": 7, "fc": C_STRUCT}], "structural"),
                                   ([{"kind": "mark", "s": 7, "fc": C_HAEM}], "hematopoietic")], gap=1.8)
    fits(x, x0 + w, "tracking key")


def sci_fig5():
    m911, m916 = mod("911_fig3_heat_shock"), mod("916_fig7_independent_cohort")
    cv = Canvas(180.0, 118.0)
    cv.relabel = None
    panel_offsets(cv, 0.0, 0.0, 52.0, 60.0)
    panel_tracking(cv, 54.0, 0.0, 52.0, 60.0)
    cv.relabel = ["C"]
    m911.panel_b(cv, 108.0, 0.0, 72.0, 60.0)
    cv.relabel = ["D"]
    m911.panel_f(cv, 0.0, 62.0, 60.0, 56.0)
    cv.relabel = ["E"]
    m916.panel_e(cv, 62.0, 62.0, 58.0, 56.0)
    cv.relabel = ["F"]
    m916.panel_f(cv, 122.0, 62.0, 58.0, 56.0)
    save(cv, "sci_fig5_tissue_time_v1")


BUILD = {"1": sci_fig1, "2": sci_fig2, "3": sci_fig3, "4": sci_fig4, "5": sci_fig5}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(BUILD)):
        BUILD[k]()
