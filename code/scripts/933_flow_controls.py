"""Round 9 writing (D99, plan section 4) — flow-cytometry controls on all four processing days, for the Supplementary
Fig. 5 legend (Supplementary Fig. 1 until D133, 4 until D136). Supplementary Fig. 5c draws the unstained and fluorescence-minus-one (FMO) controls of one processing day
(cohort 1, ZT5); this script applies the same fixed gates (gates.csv, via 932's gate definitions) to the controls of
every processing day (cohort x ZT group = 4 days) and reports, per control and target gate, the plotted parent events,
the events falling in the target gate (recomputed from events.csv) next to the export's tube count, and the largest
value of the omitted marker among the parent events against its gate threshold.

Controls -> target gate: unstained cochlea (T1, CD45+ of single cells); FMO CD11b (T2, immature and mature neutrophils
of Ly6G+ SSC-hi cells); FMO Ly6C (T3, classical monocytes of monocyte-scatter cells); FMO CD62L (T4, CD62L-hi of Ly6G+
SSC-hi); FMO CXCR4 (T5, CXCR4-hi of Ly6G+ SSC-hi); FMO IgD (T6, recirculating B of B220+); FMO IgM (T7, immature B of
B220+; its IgM-negative pro/pre-B gate is where every FMO event belongs and is not a background readout).
Gate membership follows 932 ([lo, hi) in bead-relative units); the export's tube counts can differ by boundary
rounding (932 asserts agreement within 0.25 percentage points).

Output (lib/io.write_csv): results/r9_flow/flow_fmo_controls.csv.
Run (local; the flow data are not on the server): COCH_ROOT=/path/to/coch-rythum python scripts/933_flow_controls.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("flow932", Path(__file__).resolve().parent / "932_flow_validation.py")
f932 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f932)

# control -> (gate key in 932's gate_specs, omitted marker, its channel, gate names read as background)
CONTROLS = {"unstained_CO": ("T1_cd45", "CD45", "ch1", ["CD45+"]),
            "FMO_T2": ("T2", "CD11b", "ch1", ["Immature", "Mature"]),
            "FMO_T3": ("T3", "Ly6C", "ch1", ["Classical monocytes"]),
            "FMO_T4": ("T4", "CD62L", "ch1", ["CD62L-hi"]),
            "FMO_T5": ("T5", "CXCR4", "ch2", ["CXCR4-hi"]),
            "FMO_T6": ("T6", "IgD", "ch2", ["Recirculating B"]),
            "FMO_T7": ("T7", "IgM", "ch2", ["Immature B"])}


def main():
    g = pd.read_csv(f932.FLOW / "gates.csv", encoding="utf-8-sig").set_index("parameter")["value"]
    tv = pd.read_csv(f932.FLOW / "tube_values.csv", encoding="utf-8-sig")
    ctl = tv[(tv.record_type == "control") & tv.control_type.isin(CONTROLS)].copy()
    assert len(ctl) == 4 * len(CONTROLS), len(ctl)
    recs = set(ctl.record_id)
    cols = ["record_id", "tube", "FSC_A", "SSC_A", "FSC_H", "channel1_bead_relative", "channel2_bead_relative",
            "gate_cells", "gate_singlets"]
    parts = []
    for ch in pd.read_csv(f932.FLOW / "events.csv", usecols=cols, encoding="utf-8-sig", chunksize=1_000_000,
                          dtype={"record_id": str, "tube": str, "gate_cells": "int8", "gate_singlets": "int8"}):
        parts.append(ch[ch.record_id.isin(recs)])
    ev = pd.concat(parts, ignore_index=True)
    spec = f932.gate_specs(g)
    rows = []
    for c in ctl.sort_values(["cohort", "zt_group", "control_type"]).itertuples():
        key, marker, chan, read = CONTROLS[c.control_type]
        tube = "T1" if key == "T1_cd45" else key
        e = ev[(ev.record_id == c.record_id) & (ev.tube == tube)].reset_index(drop=True)
        assert len(e), (c.record_id, tube)
        parent, xa, ya, glist = spec[key]
        ep = e[f932.population(e, parent, g)]
        om = ep[f932.AXCOL[chan]].to_numpy()
        for gname, gx, gy, num, den, _ in glist:
            if gname not in read:
                continue
            xv, yv = ep[f932.AXCOL[xa]].to_numpy(), ep[f932.AXCOL[ya]].to_numpy()
            inside = (xv >= gx[0]) & (xv < gx[1]) & (yv >= gy[0]) & (yv < gy[1])
            thr = gx[0] if chan == xa else gy[0]                 # lower bound of the gate on the omitted marker
            tc, td = getattr(c, num), getattr(c, den)
            rows.append(dict(processing_day=f"cohort {c.cohort} {c.zt_group}", cohort=int(c.cohort), zt_group=c.zt_group,
                             control_type=c.control_type, record_id=c.record_id, tissue=c.tissue, tube=tube,
                             omitted_marker=marker, parent_population=parent, gate_name=gname,
                             parent_events=int(len(ep)), events_in_gate=int(inside.sum()),
                             pct_events=100.0 * inside.sum() / max(len(ep), 1),
                             tube_count_in_gate=float(tc), tube_count_parent=float(td),
                             pct_tube=100.0 * tc / td if np.isfinite(tc) and np.isfinite(td) and td > 0 else np.nan,
                             omitted_marker_max=float(np.max(om)) if len(om) else np.nan,
                             omitted_marker_q99=float(np.quantile(om, 0.99)) if len(om) else np.nan,
                             gate_threshold=float(thr)))
    out = pd.DataFrame(rows)
    assert out.groupby("control_type").processing_day.nunique().eq(4).all()
    n_pos = int((out.events_in_gate > 0).sum())
    log(f"controls x days x target gates: {len(out)}; with any event in the target gate: {n_pos}; "
        f"max {out.pct_events.max():.2f} % ({out.loc[out.pct_events.idxmax(), ['processing_day', 'control_type', 'gate_name', 'events_in_gate', 'parent_events']].tolist()}); "
        f"tube counts in gate: {int(out.tube_count_in_gate.fillna(0).sum())}")
    for r in out.itertuples():
        log(f"  {r.processing_day:16s} {r.control_type:13s} {r.gate_name:20s} {r.events_in_gate:3d} / {r.parent_events:5d}"
            f"  max {r.omitted_marker} {r.omitted_marker_max:.3f} vs {r.gate_threshold:.3f}")
    write_csv(out, "r9_flow", "flow_fmo_controls.csv",
              "Flow validation cohort: unstained and FMO controls on all four processing days (cohort x ZT group) - "
              "plotted parent events, events in the target gate recomputed with the fixed gates (vs the export's tube "
              "count) and the largest omitted-marker value vs its gate threshold; Supplementary Fig. 1 legend (933)",
              required=["processing_day", "control_type", "gate_name", "parent_events", "events_in_gate", "pct_events",
                        "omitted_marker_max", "gate_threshold"])


if __name__ == "__main__":
    main()
