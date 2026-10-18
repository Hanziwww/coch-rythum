"""Round 9 (D92) — Supplementary Fig. 5 (Supplementary Fig. 1 until D133, 4 until D136): flow-cytometry gating strategy and controls (Nature Portfolio reporting summary:
'a figure exemplifying the gating strategy is provided in the Supplementary Information'), 183 mm wide, 5-7 pt.

a  Cochlea and b temporal-bone marrow of one mouse (932: complete in both tissues, cochlear neutrophil share closest to
   the median of the complete mice): FSC-A vs SSC-A (cells) -> FSC-A vs FSC-H (single cells) -> T1 CD45-FITC vs SSC-A
   (CD45+) -> T1 Ly6G-PE vs SSC-A (neutrophils); T2 CD11b-FITC vs Ly6G-PE of Ly6G+ SSC-hi cells (immature, mature); T3
   monocyte scatter -> Ly6C-FITC vs CD11b-PE (classical monocytes); T4 CD62L-hi and T5 CXCR4-hi of Ly6G+ SSC-hi cells;
   T6 B220-FITC vs IgD-PE and T7 B220-FITC vs IgM-PE of B220+ cells (marrow; the cochlear T6 / T7 tubes were run only
   when cells sufficed and feed no primary readout); T8 CD45-FITC vs PI (viability).
c  Controls of one processing day (cohort 1, ZT5): unstained cochlea (CD45 gate) and the fluorescence-minus-one (FMO)
   control of every gated marker, with the same gates.
Contour plots with outliers, gates and frequencies (% of the plotted parent) from the 932 tables (flow_plots, _contours,
_outliers, _gates); every plot carries its own scale (each is its own marker group). Arrows link gates of one tube.
Output: figures/figures/sfig1_flow_gating_v1.{pdf,svg,jpg,png}. Run: COCH_ROOT=/path/to/coch-rythum python scripts/917_sfig1_flow_gating.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import PT_SMALL, TEXT, Canvas, fits, text_width  # noqa: E402
from lib.flowplot import arrow, flow_contour, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "r9_figures"
OUT = ROOT / "figures" / "figures"
NAME = "sfig1_flow_gating_v2"
CO, BM = "#0072B2", "#D55E00"
PARENT = {"all events": "all events", "cells": "cells", "single cells": "single cells", "CD45+ single cells": "CD45⁺",
          "Ly6G+ SSC-hi single cells": "Ly6G⁺ SSC-hi", "monocyte-scatter single cells": "monocyte scatter",
          "B220+ single cells": "B220⁺"}
GATE = {"CD45+": "CD45⁺", "PI-": "PI⁻", "Classical monocytes": "Cl. monocytes", "Monocyte scatter": "Mono. scatter",
        "Recirculating B": "Recirc. B", "Immature B": "Imm. B"}
FMO = {"FMO_T2": "FMO CD11b", "FMO_T3": "FMO Ly6C", "FMO_T4": "FMO CD62L", "FMO_T5": "FMO CXCR4", "FMO_T6": "FMO IgD",
       "FMO_T7": "FMO IgM", "unstained_CO": "Unstained cochlea"}
ROW1 = ["cells", "singlets", "T1_cd45", "T1", "T2", "T3_scatter", "T3"]
LINKS = {("cells", "singlets"), ("singlets", "T1_cd45"), ("T1_cd45", "T1"), ("T3_scatter", "T3")}
S, YA, GAP = 16.4, 6.4, 2.3                               # plot side, y-axis area, gap (mm)
PITCH = YA + S + GAP
ROW_H = 3.4 + S + 4.6 + 2.0                               # title, plot, x numbers + title, spacing


def draw_row(cv, tabs, x0, top, pids, titles, links=()):
    """One row of plots starting at x0 (left edge of the first y-axis area), plot tops at `top`; titles may have two
    lines (the widest line must fit over the plot and its y-axis area)."""
    ends = []
    for k, (pid, title) in enumerate(zip(pids, titles)):
        px = x0 + YA + k * PITCH
        cv.text(px + S / 2, top - 0.6, title, size=PT_SMALL, ha="center", va="bottom", linespacing=1.05)
        tw = max(text_width(t_, PT_SMALL) for t_ in title.split("\n"))
        fits(tw, S + YA, f"title {pid}")
        ends.append((px + S / 2 - tw / 2, px + S / 2 + tw / 2))
        g = tabs["gates"][pid]
        tabs["gates"][pid] = g.assign(gate_name=g.gate_name.map(lambda n: GATE.get(n, n)))
        flow_contour(cv, px, top, S, pid, tabs, names=True)
    for k in links:                                       # arrow in the title row, between the two titles
        xa = max(x0 + YA + k * PITCH + S, ends[k][1]) + 0.6
        xb = min(x0 + YA + (k + 1) * PITCH, ends[k + 1][0]) - 0.6
        assert xb - xa >= 2.5, f"arrow {pids[k]} too short ({xb - xa:.1f} mm)"
        arrow(cv, xa, top - 1.55, xb, top - 1.55)
    fits(x0 + YA + len(pids) * PITCH - GAP, 180.0, "row width")


def main():
    tabs = load_tables(R9)
    pl = tabs["plots"]
    cv = Canvas(180.0, 2 * (4.2 + 2 * ROW_H) + 4.2 + 2.0 + ROW_H)
    y = 0.0
    for letter, tag, tname, col in (("a", "co", "Cochlea", CO), ("b", "bm", "Temporal-bone marrow", BM)):
        mouse = pl.loc[f"gating_{tag}_cells"]
        cv.letter(0.0, y, letter, f"{tname} (one mouse, {mouse.zt})")
        top = y + 4.2 + 3.4
        r1 = [f"gating_{tag}_{k}" for k in ROW1]
        t1 = [f"{pl.loc[p].tube} · {PARENT[pl.loc[p].population]}" for p in r1]
        draw_row(cv, tabs, 0.0, top, r1, t1, links=[i for i in range(len(ROW1) - 1) if (ROW1[i], ROW1[i + 1]) in LINKS])
        r2 = [f"gating_{tag}_{k}" for k in ("T4", "T5", "T6", "T7", "T8") if f"gating_{tag}_{k}" in pl.index]
        t2 = [f"{pl.loc[p].tube} · {PARENT[pl.loc[p].population]}" for p in r2]
        draw_row(cv, tabs, 0.0, top + ROW_H, r2, t2)
        y += 4.2 + 2 * ROW_H
    cv.letter(0.0, y, "c", "Controls (one processing day)")
    top = y + 4.2 + 3.4 + 2.0                             # two-line titles
    rc = [f"control_{c}" for c in ("unstained_CO", "FMO_T2", "FMO_T3", "FMO_T4", "FMO_T5", "FMO_T6", "FMO_T7")]
    tc = [f"{FMO[p.replace('control_', '')]}\n{pl.loc[p].tube} · {PARENT[pl.loc[p].population]}" for p in rc]
    g7 = tabs["gates"]["control_FMO_T7"]                  # the IgM FMO is read on the IgM+ gate only
    tabs["gates"]["control_FMO_T7"] = g7[g7.gate_name.isin(["Immature B", "Imm. B"])]
    draw_row(cv, tabs, 0.0, top, rc, tc)
    for p in cv.save(OUT, NAME):
        print(f"[fig] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
