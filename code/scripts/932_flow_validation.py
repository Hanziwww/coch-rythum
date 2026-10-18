"""Round 9 (D89) — flow-cytometry validation cohort (open item O02), analysed exactly as pre-specified in
manuscript/flow_validation_plan.md (revision 2, sections 6.2-6.4). Local run (the flow data are not on the server).

Data: the user's export data/Cochlear_Flow_Plotting_Only.zip, extracted to data/flow_cochlea (events.csv: 6,568,178
events with scatter, two fluorescence channels and fixed gate flags; tube_values.csv: per tube counts; sample_values.csv:
per mouse x tissue summaries; gates.csv: fixed thresholds; channels.csv: tubes T1-T8, FITC / PE). 20 male C57BL/6J
mice (10 weeks), ZT5 and ZT17, two cohorts of 5 + 5; cochlea and temporal-bone marrow of every mouse.

Readouts per mouse x tissue (section 6.2), recomputed from tube_values.csv (they agree with the user's
sample_values.csv to rounding, asserted): CD45 % of single cells and neutrophils (Ly6G+ SSC-hi) % of CD45+ (T1);
immature fraction = Ly6G-int CD11b-int / Ly6G+ (T2); P1 = neutrophils % CD45 x immature fraction; P2 = CD11b+ Ly6C-hi
monocyte-scatter cells / single cells (T3) / CD45 % of single cells; P3 = B220-hi IgD+ / single cells (T6) / CD45 %;
S1 = % CD62L-hi of Ly6G+ and CD62L median (T4); S2 = % CXCR4-hi of Ly6G+ and CXCR4 median (T5); marrow-content index
(cochlea) = [B220-lo IgM- / single cells] (T7) / [CD45- / single cells] (T1).
Exclusions (6.3, fixed in advance): a cochlear tube with < 3,000 single-cell events drops the readouts that use it
(M005 cochlea: T1 -> P1, P2; T5 -> S2); a parent gate with < 100 events (none); viability < 50 % (trypan blue, because
every cochlear T8 tube falls under the event floor; all >= 78 %: none); perfusion quality is not in the export;
consistency checks (CD11b+ % of single cells T2 vs T3 within 20 % relative; B220+ T6 vs T7) are reported, and P1-P2 are
repeated without flagged samples.
Statistics (6.4): P1-P3, OLS on log(% of CD45+) with ZT + cohort; ZT17 / ZT5 ratio with its 95 % CI; two-sided p;
Holm across the three primaries; verdict replicated (Holm p < 0.05, predicted direction) / direction-consistent /
contradicted. S1-S2, linear mixed model (statsmodels MixedLM, REML) on logit(% positive) and on log(median
bead-relative FI) with tissue x ZT + cohort and a random intercept per mouse; the interaction and the within-tissue
ZT17 - ZT5 contrasts (Wald). Sensitivity (D17): P1-P2 divided by the marrow-content index, for the cochleae with T7.
Plots (D92; Nature Portfolio reporting-summary conventions: contour plots with outliers, marker-fluorochrome axis labels,
visible scales, gate frequencies; gating strategy as a supplementary figure). Main panels: per group the mouse whose
readout is closest to the group median (Fig. 5c since D134: cochlear T2 Ly6G+ SSC-hi cells, CD11b-FITC vs Ly6G-PE,
T3 monocyte-scatter cells, Ly6C-FITC vs CD11b-PE, and marrow T6 B220+ cells, B220-FITC vs IgD-PE; Fig. 6h: T5 Ly6G+
SSC-hi cells of cochlea and marrow, Ly6G-FITC vs CXCR4-PE). Gating
strategy (Supplementary Fig. 5): every gate of tubes T1-T8 in the cochlea and marrow of one mouse (complete in both
tissues, cochlear neutrophil share closest to the median of the complete mice) and the FMO / unstained controls of one
processing day (cohort 1, ZT5). Display: fluorescence asinh(bead-relative / 0.05) on [-0.05, 30] (all events inside),
scatter x 10^-3 on [0, 262.144] (clipped). Probability contours: events binned on a 160 x 160 grid, Gaussian-smoothed
(sigma = 1 + 3.5 (500 / n)^0.3 bins, 1.2-4.5), density looked up per event; contour lines at the event-density
quantiles 5, 10, 20, ..., 90 % (the outermost line encloses 95 % of events); the 5 % of events outside it are exported
as outlier dots (real events). Gates are drawn from gates.csv; their membership reproduces the export's gate flags
(>= 99.99 % of events; boundary rounding), and the frequency shown (% of the plotted parent) is the tube count
(tube_values.csv), asserted to equal the event count within 0.25 points.

Digestion controls (section 5.4, one per processing day = cohort x ZT group): marrow cells taken through the cochlear
digestion vs undigested, stained with T4 / T5; median and % positive of Ly6G+ cells, digested / undigested.

Outputs (lib/io.write_csv): results/r9_flow/flow_readouts.csv, flow_tests.csv, flow_consistency.csv,
flow_digestion_controls.csv;
results/r9_figures/f4f_flow_shares.csv, f5i_flow_markers.csv, flow_plots.csv, flow_contours.csv, flow_outliers.csv,
flow_gates.csv (D92).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/932_flow_validation.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "data" / "flow_cochlea"
ZIP = ROOT / "data" / "Cochlear_Flow_Plotting_Only.zip"
CO, BM = "cochlea", "temporal_bone_marrow"
COF = 0.05                                               # asinh cofactor (bead-relative units)
NB = 96
PRIMARY = [("P1", CO, "ZT5 > ZT17", "Immature neutrophils (Ly6G-int CD11b-int), % of CD45+", "A05"),
           ("P2", CO, "ZT5 > ZT17", "Classical monocytes (CD11b+ Ly6C-hi), % of CD45+", "A04"),
           ("P3", BM, "ZT17 > ZT5", "Recirculating B (B220-hi IgD+), % of CD45+", "A01")]
SECONDARY = [("S1_pct", "S1", "CD62L-hi, % of Ly6G+", "logit"), ("S1_med", "S1", "CD62L median (bead-relative)", "log"),
             ("S2_pct", "S2", "CXCR4-hi, % of Ly6G+", "logit"), ("S2_med", "S2", "CXCR4 median (bead-relative)", "log")]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def readouts(sv, tv):
    bio = tv[tv.record_type == "biological"].set_index(["record_id", "tube"])
    rows = []
    for _, s in sv.iterrows():
        co = s.tissue == CO
        t = {}
        why = []
        for k in range(1, 9):
            key = (s.sample_id, f"T{k}")
            r = bio.loc[key] if key in bio.index else None
            if r is None or r.status != "run":
                t[k] = None
                continue
            if co and k != 8 and r.singlets_events < 3000:          # 6.3 event floor (T8 = viability only)
                why.append(f"T{k} {int(r.singlets_events)} single cells < 3,000")
                t[k] = None
                continue
            t[k] = r
        d = dict(sample_id=s.sample_id, mouse_id=s.mouse_id, tissue=s.tissue, cohort=int(s.cohort), zt=s.zt_group,
                 trypan_viability_pct=s.trypan_viability_pct, exclusions="; ".join(why))
        if t[1] is not None:
            d["cd45_pct_cells"] = 100 * t[1].cd45_events / t[1].singlets_events
            d["cd45neg_pct_cells"] = 100 * t[1].cd45_neg_events / t[1].singlets_events
            d["neut_pct_cd45"] = 100 * t[1].ly6g_events / t[1].cd45_events
        if t[1] is not None and t[2] is not None and t[2].ly6g_events >= 100:
            d["imm_frac_pct"] = 100 * t[2].int_events / t[2].ly6g_events
            d["P1"] = d["neut_pct_cd45"] * d["imm_frac_pct"] / 100
            d["cd11b_pct_cells_T2"] = 100 * t[2].cd11b_events / t[2].singlets_events
        if t[1] is not None and t[3] is not None:
            d["P2"] = 100 * (t[3].mono_events / t[3].singlets_events) / (d["cd45_pct_cells"] / 100)
            d["cd11b_pct_cells_T3"] = 100 * t[3].cd11b_events / t[3].singlets_events
        if t[1] is not None and t[6] is not None and t[6].b220_events >= 100:
            d["P3"] = 100 * (t[6].target_events / t[6].singlets_events) / (d["cd45_pct_cells"] / 100)
            d["recirc_frac_pct"] = 100 * t[6].target_events / t[6].b220_events
            d["b220_pct_cells_T6"] = 100 * t[6].b220_events / t[6].singlets_events
        if t[7] is not None and t[7].b220_events >= 100:
            d["b220_pct_cells_T7"] = 100 * t[7].b220_events / t[7].singlets_events
            if co and t[1] is not None:                  # B220-lo IgM- (igm_neg_events) / CD45- (T1)
                d["marrow_content_index"] = ((t[7].igm_neg_events / t[7].singlets_events)
                                             / (t[1].cd45_neg_events / t[1].singlets_events))
        for k, (pre, col) in [(4, ("S1", "CD62L")), (5, ("S2", "CXCR4"))]:
            if t[k] is not None and t[k].ly6g_events >= 100:
                d[f"{pre}_pct"] = 100 * t[k].above_threshold_events / t[k].ly6g_events
                d[f"{pre}_med"] = t[k].median_bead_relative_FI
        rows.append(d)
    r = pd.DataFrame(rows)
    # agreement with the user's per-sample summaries (rounding only)
    m = r.merge(sv, on="sample_id")
    for a, b in [("P1", "P1_int_neutrophil_pct_CD45"), ("P2", "P2_monocyte_pct_CD45"),
                 ("P3", "P3_B220high_IgDpos_pct_CD45"), ("S1_pct", "CD62L_above_threshold_pct"),
                 ("S2_pct", "CXCR4_above_threshold_pct"), ("neut_pct_cd45", "Ly6G_pct_CD45"),
                 ("cd45_pct_cells", "CD45_pct_scatter_singlets")]:
        x = m[[a, b]].dropna()
        assert len(x) and np.allclose(x[a], x[b], rtol=1e-3), (a, b)
    # consistency checks (6.2)
    r["flag_cd11b_T2_T3"] = (np.abs(r.cd11b_pct_cells_T2 / r.cd11b_pct_cells_T3 - 1) > 0.20)
    r["flag_b220_T6_T7"] = (np.abs(r.b220_pct_cells_T6 / r.b220_pct_cells_T7 - 1) > 0.20)
    return r


def primary_tests(r, drop_flagged=False):
    out = []
    for key, tis, pred, lab, fid in PRIMARY:
        d = r[r.tissue == tis].dropna(subset=[key]).copy()
        if drop_flagged:
            d = d[~d.flag_cd11b_T2_T3.fillna(False).astype(bool)]
        d["y"], d["night"] = np.log(d[key]), (d.zt == "ZT17").astype(int)
        f = smf.ols("y ~ night + C(cohort)", data=d).fit()
        lo, hi = f.conf_int().loc["night"]
        out.append(dict(test=key, family="primary", tissue=tis, readout=lab, prediction=pred, finding=fid,
                        model="OLS log(% of CD45+) ~ ZT + cohort", n_zt5=int((d.night == 0).sum()),
                        n_zt17=int(d.night.sum()), gmean_zt5=float(np.exp(d.y[d.night == 0].mean())),
                        gmean_zt17=float(np.exp(d.y[d.night == 1].mean())), effect=float(np.exp(f.params["night"])),
                        effect_scale="ZT17 / ZT5 ratio", ci_lo=float(np.exp(lo)), ci_hi=float(np.exp(hi)),
                        p=float(f.pvalues["night"])))
    t = pd.DataFrame(out)
    t["p_holm"] = multipletests(t.p, method="holm")[1]
    up = t.prediction.str.startswith("ZT17")
    ok = np.where(up, t.effect > 1, t.effect < 1)
    t["verdict"] = np.where(ok & (t.p_holm < 0.05), "replicated", np.where(ok, "direction-consistent",
                            np.where(t.p_holm < 0.05, "contradicted", "opposite direction, not significant")))
    return t


def secondary_tests(r):
    out = []
    for key, test, lab, link in SECONDARY:
        d = r.dropna(subset=[key]).copy()
        v = d[key] / 100 if link == "logit" else d[key]
        d["y"] = np.log(v / (1 - v)) if link == "logit" else np.log(v)
        d["night"], d["coch"] = (d.zt == "ZT17").astype(int), (d.tissue == CO).astype(int)
        m = smf.mixedlm("y ~ coch * night + C(cohort)", d, groups=d["mouse_id"]).fit(reml=True)
        cov = m.cov_params()
        b_m, se_m = m.params["night"], np.sqrt(cov.loc["night", "night"])
        b_c = m.params["night"] + m.params["coch:night"]
        se_c = np.sqrt(cov.loc["night", "night"] + cov.loc["coch:night", "coch:night"] + 2 * cov.loc["night", "coch:night"])
        scale = "odds ratio, ZT17 / ZT5" if link == "logit" else "ratio, ZT17 / ZT5"
        for part, b, se, p in [("cochlea", b_c, se_c, 2 * norm.sf(abs(b_c / se_c))),
                               ("marrow", b_m, se_m, 2 * norm.sf(abs(b_m / se_m))),
                               ("interaction", m.params["coch:night"], np.sqrt(cov.loc["coch:night", "coch:night"]),
                                m.pvalues["coch:night"])]:
            out.append(dict(test=test, family="secondary", tissue=part, readout=lab,
                            prediction="cochlea ZT17 > ZT5, marrow flat" if test == "S1" else
                            "cochlea ZT5 > ZT17, marrow flat or opposite",
                            finding="G02", model=f"MixedLM {link}(y) ~ tissue x ZT + cohort, (1 | mouse)",
                            n_zt5=int(((d.night == 0) & ((d.coch == 1) if part == "cochlea" else
                                                         (d.coch == 0) if part == "marrow" else True)).sum()),
                            n_zt17=int(((d.night == 1) & ((d.coch == 1) if part == "cochlea" else
                                                          (d.coch == 0) if part == "marrow" else True)).sum()),
                            effect=float(np.exp(b)), effect_scale=scale if part != "interaction" else
                            scale.replace("ZT17 / ZT5", "cochlea / marrow of ZT17 / ZT5"),
                            ci_lo=float(np.exp(b - 1.96 * se)), ci_hi=float(np.exp(b + 1.96 * se)), p=float(p)))
    t = pd.DataFrame(out)
    t["p_holm"] = np.nan
    v = []
    for _, x in t.iterrows():
        if x.tissue != "cochlea":
            v.append("")
            continue
        inter = t[(t.readout == x.readout) & (t.tissue == "interaction")].iloc[0]
        want_up = x.test == "S1"
        ok = (x.effect > 1) if want_up else (x.effect < 1)
        sig = x.p < 0.05 and inter.p < 0.05
        v.append("replicated" if ok and sig else "direction-consistent" if ok else
                 "opposite direction" + (", significant" if x.p < 0.05 else ", not significant"))
    t["verdict"] = v
    return t


def representative(r, tissue, zt, key):
    d = r[(r.tissue == tissue) & (r.zt == zt)].dropna(subset=[key])
    return d.iloc[int((d[key] - d[key].median()).abs().argmin())]


NG = 160                                                 # probability-contour grid (bins per axis)
LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)  # share of events outside each contour line
FL_RANGE = (-0.05, 30.0)                                 # bead-relative display range (every event inside)
SC_MAX = 262.144                                         # scatter display maximum (x 10^3, 18-bit scale)
AXCOL = {"ch1": "channel1_bead_relative", "ch2": "channel2_bead_relative", "FSC": "FSC_A", "SSC": "SSC_A",
         "FSCH": "FSC_H"}
SCATTER_LABEL = {"FSC": "FSC-A", "SSC": "SSC-A", "FSCH": "FSC-H"}
INF = np.inf


def tf(v):
    return np.arcsinh(np.asarray(v, float) / COF)


def axis_range(a):
    return (float(tf(FL_RANGE[0])), float(tf(FL_RANGE[1]))) if a in ("ch1", "ch2") else (0.0, SC_MAX)


def to_disp(a, v):
    """Raw value(s) (bead-relative or scatter units; +-inf = axis end) -> display coordinate within the axis range."""
    lo, hi = axis_range(a)
    v = np.atleast_1d(np.asarray(v, float))
    fin = np.where(np.isfinite(v), v, 0.0)
    d = tf(fin) if a in ("ch1", "ch2") else fin / 1000.0
    d = np.where(np.isposinf(v), hi, np.where(np.isneginf(v), lo, d))
    return np.clip(d, lo, hi)


def population(e, name, g):
    s_ = (e.gate_singlets == 1).to_numpy()
    c1, c2 = e.channel1_bead_relative.to_numpy(), e.channel2_bead_relative.to_numpy()
    hi = (e.SSC_A >= g["neutrophil_SSC_min"]).to_numpy()
    mono = (e.FSC_A.between(g["mono_FSC_low"], g["mono_FSC_high"])
            & e.SSC_A.between(g["mono_SSC_low"], g["mono_SSC_high"])).to_numpy()
    return {"all events": np.ones(len(e), bool), "cells": (e.gate_cells == 1).to_numpy(), "single cells": s_,
            "CD45+ single cells": s_ & (c1 >= g["CD45_pos"]),
            "Ly6G+ SSC-hi single cells": s_ & hi & ((c2 if e.tube.iloc[0] in ("T2", "T4") else c1) >= g["Ly6G_pos"]),
            "monocyte-scatter single cells": s_ & mono, "B220+ single cells": s_ & (c1 >= g["B220_pos"])}[name]


def gate_specs(g):
    """Per tube: (plotted parent, x axis, y axis, gates); gate = (name, x range, y range, tube count, tube parent, label
    corner). Ranges in raw units ([lo, hi), +-inf = open); 'singlets' is the FSC-H / FSC-A band."""
    return {
        "cells": ("all events", "FSC", "SSC", [("Cells", (g["FSC_min"], g["FSC_max"]), (g["SSC_min"], g["SSC_max"]),
                                                  "cells_events", "acquired_events", "tl")]),
        "singlets": ("cells", "FSC", "FSCH", [("Single cells", "band", "band", "singlets_events", "cells_events", "tl")]),
        "T1_cd45": ("single cells", "ch1", "SSC", [("CD45+", (g["CD45_pos"], INF), (-INF, INF), "cd45_events",
                                                    "singlets_events", "tr")]),
        "T1": ("CD45+ single cells", "ch2", "SSC", [("Neutrophils", (g["Ly6G_pos"], INF), (g["neutrophil_SSC_min"], INF),
                                                     "ly6g_events", "cd45_events", "tr")]),
        "T2": ("Ly6G+ SSC-hi single cells", "ch1", "ch2", [
            ("Immature", (g["CD11b_pos"], g["CD11b_split"]), (g["Ly6G_pos"], g["Ly6G_split"]), "int_events", "ly6g_events",
             "below"),
            ("Mature", (g["CD11b_split"], INF), (g["Ly6G_split"], INF), "high_events", "ly6g_events", "tr")]),
        "T3_scatter": ("single cells", "FSC", "SSC", [("Monocyte scatter", (g["mono_FSC_low"], g["mono_FSC_high"]),
                                                       (g["mono_SSC_low"], g["mono_SSC_high"]), "mono_scatter_events",
                                                       "singlets_events", "above")]),
        "T3": ("monocyte-scatter single cells", "ch1", "ch2", [("Classical monocytes", (g["Ly6C_hi"], INF),
                                                                 (g["CD11b_pos"], INF), "mono_events",
                                                                 "mono_scatter_events", "tr")]),
        "T4": ("Ly6G+ SSC-hi single cells", "ch1", "ch2", [("CD62L-hi", (g["CD62L_threshold"], INF), (-INF, INF),
                                                            "above_threshold_events", "ly6g_events", "tr")]),
        "T5": ("Ly6G+ SSC-hi single cells", "ch1", "ch2", [("CXCR4-hi", (-INF, INF), (g["CXCR4_threshold"], INF),
                                                            "above_threshold_events", "ly6g_events", "tl")]),
        "T6": ("B220+ single cells", "ch1", "ch2", [("Recirculating B", (g["B220_hi"], INF), (g["IgD_pos"], INF),
                                                     "target_events", "b220_events", "tr")]),
        "T7": ("B220+ single cells", "ch1", "ch2", [
            ("Immature B", (g["B220_pos"], g["B220_hi"]), (g["IgM_pos"], INF), "igm_pos_events", "b220_events", "tl"),
            ("Pro/pre-B", (g["B220_pos"], g["B220_hi"]), (-INF, g["IgM_pos"]), "igm_neg_events", "b220_events", "bl")]),
        "T8": ("single cells", "ch1", "ch2", [("PI-", (-INF, INF), (-INF, g["PI_positive"]), "pi_negative_events",
                                               "singlets_events", "bl")])}


def probability_contours(x, y, xr, yr):
    """Contour lines at event-density quantiles (LEVELS) and the outlier mask (events outside the outermost line)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import approximate_polygon, find_contours
    n = len(x)
    sig = float(np.clip(1.0 + 3.5 * (500.0 / max(n, 1)) ** 0.3, 1.2, 4.5))
    h, _, _ = np.histogram2d(x, y, bins=NG, range=[xr, yr])
    dens = gaussian_filter(h, sig, mode="constant")
    ix = np.clip(((x - xr[0]) / (xr[1] - xr[0]) * NG).astype(int), 0, NG - 1)
    iy = np.clip(((y - yr[0]) / (yr[1] - yr[0]) * NG).astype(int), 0, NG - 1)
    de = dens[ix, iy]
    lev = np.quantile(de, LEVELS) if n else np.zeros(len(LEVELS))
    z = np.pad(dens.T, 1)                                  # rows = y, columns = x; padding closes edge contours
    paths, prev = [], None
    for k, (q, lv) in enumerate(zip(LEVELS, lev)):
        if n < 20 or lv <= 0 or (prev is not None and np.isclose(lv, prev)):
            continue
        prev = lv
        for j, c in enumerate(find_contours(z, lv)):
            c = approximate_polygon(c, tolerance=0.3)
            yy = np.clip(yr[0] + (c[:, 0] - 0.5) / NG * (yr[1] - yr[0]), *yr)
            xx = np.clip(xr[0] + (c[:, 1] - 0.5) / NG * (xr[1] - xr[0]), *xr)
            paths.append((k, q, j, xx, yy))
    out = de < lev[0] if n >= 20 else np.ones(n, bool)
    return paths, out, sig


def plot_export(ev, gates, r, tv, chan):
    """Main-panel and gating-figure plots: contours, outlier events, gates with tube-count frequencies (D92)."""
    g = gates
    spec = gate_specs(g)
    label = {(t.tube, "ch1"): f"{t.channel1_marker}-{t.channel1_fluorochrome}" for t in chan.itertuples()}
    label.update({(t.tube, "ch2"): f"{t.channel2_marker}-{t.channel2_fluorochrome}" for t in chan.itertuples()})
    label[("T8", "ch2")] = "PI"
    tvx = tv.set_index(["record_id", "tube"])
    by_rec = {k: d for k, d in ev.groupby(["record_id", "tube"], observed=True)}
    plots, cont, outl, gts = [], [], [], []

    def tube_of(key):
        return "T1" if key in ("cells", "singlets", "T1_cd45") else ("T3" if key == "T3_scatter" else key)

    def add(pid, role, fig, key, rec, tissue, zt, mouse, cohort, choice):
        tube = tube_of(key)
        parent, xa, ya, glist = spec[key]
        e = by_rec[(rec, tube)].reset_index(drop=True)
        m = population(e, parent, g)
        ep = e[m]
        x, y = to_disp(xa, ep[AXCOL[xa]].to_numpy()), to_disp(ya, ep[AXCOL[ya]].to_numpy())
        xr, yr = axis_range(xa), axis_range(ya)
        paths, out, sig = probability_contours(x, y, xr, yr)
        for k, q, j, xx, yy in paths:
            cont.extend(dict(plot=pid, level=k, share_outside=q, path=j, vertex=v, x=round(float(a), 4),
                             y=round(float(b), 4)) for v, (a, b) in enumerate(zip(xx, yy)))
        outl.extend(dict(plot=pid, x=round(float(a), 4), y=round(float(b), 4)) for a, b in zip(x[out], y[out]))
        t = tvx.loc[(rec, tube)] if (rec, tube) in tvx.index else None
        for gi, (gname, gx, gy, num, den, corner) in enumerate(glist):
            if gx == "band":                                # singlets: FSC-H / FSC-A in [low, high] within the cells gate
                ratio = ep.FSC_H / ep.FSC_A
                inside = ratio.between(g["singlet_low"], g["singlet_high"]).to_numpy()
                f0, f1 = g["FSC_min"], g["FSC_max"]
                vx = to_disp("FSC", [f0, f1, f1, f0])
                vy = to_disp("FSCH", [g["singlet_low"] * f0, g["singlet_low"] * f1, g["singlet_high"] * f1,
                                      g["singlet_high"] * f0])
            else:
                xv, yv = ep[AXCOL[xa]].to_numpy(), ep[AXCOL[ya]].to_numpy()
                inside = (xv >= gx[0]) & (xv < gx[1]) & (yv >= gy[0]) & (yv < gy[1])
                x0, x1 = to_disp(xa, [gx[0], gx[1]])
                y0, y1 = to_disp(ya, [gy[0], gy[1]])
                vx, vy = np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1])
            pct_ev = 100.0 * inside.sum() / max(len(ep), 1)
            pct_tube = np.nan
            if t is not None and np.isfinite(t.get(num, np.nan)) and np.isfinite(t.get(den, np.nan)) and t[den] > 0:
                pct_tube = 100.0 * t[num] / t[den]
                assert int(t[den]) == len(ep) or abs(int(t[den]) - len(ep)) <= max(3, 0.001 * len(ep)), \
                    (pid, gname, int(t[den]), len(ep))
                assert abs(pct_tube - pct_ev) < 0.25, (pid, gname, pct_tube, pct_ev)
            gts.append(dict(plot=pid, gate=gi, gate_name=gname, label_corner=corner,
                            vertices_x=";".join(f"{v:.4f}" for v in vx), vertices_y=";".join(f"{v:.4f}" for v in vy),
                            events_in_gate=int(inside.sum()), events_plotted=len(ep), pct_events=pct_ev,
                            pct_tube=pct_tube, pct_shown=pct_tube if np.isfinite(pct_tube) else pct_ev,
                            tube_count=f"{num} / {den}"))
        plots.append(dict(plot=pid, role=role, figure=fig, tissue=tissue, zt=zt, tube=tube, record_id=rec,
                          mouse_id=mouse, cohort=cohort, population=parent, n_events=len(ep), x_axis=xa, y_axis=ya,
                          x_label=label.get((tube, xa), SCATTER_LABEL.get(xa)),
                          y_label=label.get((tube, ya), SCATTER_LABEL.get(ya)),
                          x_min=xr[0], x_max=xr[1], y_min=yr[0], y_max=yr[1], smoothing_sigma_bins=round(sig, 2),
                          n_outliers=int(out.sum()), choice_rule=choice))

    # main panels: per group the mouse closest to the group median of the panel's readout. D134: the timetable figure
    # (Fig. 5c) shows the plot behind each primary readout (P1 immature neutrophils, T2; P2 classical monocytes, T3;
    # P3 recirculating B, marrow T6); the timer-split figure is Fig. 6h
    for key, tis, read, fig in [("T2", CO, "P1", "Fig. 5c"), ("T3", CO, "P2", "Fig. 5c"), ("T6", BM, "P3", "Fig. 5c"),
                                ("T5", CO, "S2_pct", "Fig. 6h"), ("T5", BM, "S2_pct", "Fig. 6h")]:
        for zt in ("ZT5", "ZT17"):
            s_ = representative(r, tis, zt, read)
            tag = "co" if tis == CO else "bm"
            add(f"main_{key}_{tag}_{zt}", "main", fig, key, s_.sample_id, tis, zt, s_.mouse_id, int(s_.cohort),
                f"closest to the {tis} {zt} median of {read}")
    # gating strategy: one mouse complete in both tissues, cochlear neutrophil share closest to the median
    keys = ["neut_pct_cd45", "P1", "P2", "P3", "S1_pct", "S2_pct"]
    run = tv[(tv.record_type == "biological") & (tv.status == "run")].groupby("record_id").tube.apply(set)
    need = {CO: {"T1", "T2", "T3", "T4", "T5", "T8"}, BM: {f"T{k}" for k in range(1, 9)}}  # cochlear T6-T7: cells permitting
    ok = r.groupby("mouse_id").apply(lambda d: len(d) == 2 and (d.exclusions == "").all()
                                     and d[["neut_pct_cd45", "S1_pct", "S2_pct"]].notna().all().all()
                                     and all(need[x.tissue] <= run.get(x.sample_id, set()) for x in d.itertuples()))
    co = r[(r.tissue == CO) & r.mouse_id.isin(ok[ok].index)]
    pick = co.iloc[int((co.neut_pct_cd45 - co.neut_pct_cd45.median()).abs().argmin())]
    for tis, tag in ((CO, "co"), (BM, "bm")):
        s_ = r[(r.mouse_id == pick.mouse_id) & (r.tissue == tis)].iloc[0]
        for key in spec:
            if (s_.sample_id, tube_of(key)) not in by_rec:          # cochlear T6 / T7 not run (insufficient cells)
                log(f"gating: {s_.sample_id} {tube_of(key)} not run, plot skipped")
                continue
            add(f"gating_{tag}_{key}", "gating", "Supplementary Fig. 5", key, s_.sample_id, tis, s_.zt, s_.mouse_id,
                int(s_.cohort), "one mouse complete in both tissues, cochlear neutrophil share closest to the median")
    # controls of one processing day (cohort 1, ZT5): unstained cochlea (CD45 gate) and the FMO of every gated marker
    ctl = tv[(tv.record_type == "control") & (tv.cohort == 1) & (tv.zt_group == "ZT5")].set_index("control_type")
    for ct, key in [("unstained_CO", "T1_cd45"), ("FMO_T2", "T2"), ("FMO_T3", "T3"), ("FMO_T4", "T4"),
                    ("FMO_T5", "T5"), ("FMO_T6", "T6"), ("FMO_T7", "T7")]:
        c = ctl.loc[ct]
        add(f"control_{ct}", "control", "Supplementary Fig. 5", key, c.record_id, c.tissue, c.zt_group, "", 1,
            f"{ct} of processing day cohort 1 ZT5")
    log(f"plots: {len(plots)} ({sum(p_['role'] == 'main' for p_ in plots)} main); gating mouse {pick.mouse_id} "
        f"({pick.zt}); contour vertices {len(cont)}, outlier events {len(outl)}, gates {len(gts)}")
    return pd.DataFrame(plots), pd.DataFrame(cont), pd.DataFrame(outl), pd.DataFrame(gts)


def main():
    sv = pd.read_csv(FLOW / "sample_values.csv", encoding="utf-8-sig")
    tv = pd.read_csv(FLOW / "tube_values.csv", encoding="utf-8-sig")
    gates = pd.read_csv(FLOW / "gates.csv", encoding="utf-8-sig").set_index("parameter")["value"]
    src = "; ".join(f"{p.name} {md5(p)}" for p in [ZIP] + sorted(FLOW.glob("*.csv")))
    r = readouts(sv, tv)
    pt = primary_tests(r)
    pt_ns = primary_tests(r, drop_flagged=True).assign(family="primary, without T2/T3-inconsistent samples")
    pt_ns["p_holm"] = multipletests(pt_ns.p, method="holm")[1]
    st = secondary_tests(r)
    # D17 sensitivity: P1, P2 / marrow-content index (cochleae with T7)
    sens = []
    for key in ("P1", "P2"):
        d = r[(r.tissue == CO)].dropna(subset=[key, "marrow_content_index"]).copy()
        if d.zt.nunique() == 2 and d.groupby("zt").size().min() >= 2:
            d["y"], d["night"] = np.log(d[key] / d.marrow_content_index), (d.zt == "ZT17").astype(int)
            f = smf.ols("y ~ night", data=d).fit()
            lo, hi = f.conf_int().loc["night"]
            sens.append(dict(test=key, family="sensitivity (D17): / marrow-content index", tissue=CO,
                             readout=f"{key} / marrow-content index", prediction="as primary", finding="",
                             model="OLS log(y) ~ ZT (cochleae with T7)", n_zt5=int((d.night == 0).sum()),
                             n_zt17=int(d.night.sum()), effect=float(np.exp(f.params["night"])),
                             effect_scale="ZT17 / ZT5 ratio", ci_lo=float(np.exp(lo)), ci_hi=float(np.exp(hi)),
                             p=float(f.pvalues["night"])))
    tests = pd.concat([pt, pt_ns, pd.DataFrame(sens), st], ignore_index=True)
    tests["source_md5"] = src
    for _, x in pt.iterrows():
        log(f"{x.test} {x.tissue}: ZT17/ZT5 {x.effect:.2f} ({x.ci_lo:.2f}-{x.ci_hi:.2f}), p {x.p:.2g}, Holm {x.p_holm:.2g}"
            f" -> {x.verdict} (n {x.n_zt5} + {x.n_zt17})")
    for _, x in st.iterrows():
        log(f"{x.test} {x.readout} {x.tissue}: {x.effect:.2f} ({x.ci_lo:.2f}-{x.ci_hi:.2f}), p {x.p:.2g} {x.verdict}")
    write_csv(r.assign(source_md5=src), "r9_flow", "flow_readouts.csv",
              "Flow validation cohort (O02, D89): pre-specified readouts per mouse x tissue recomputed from the user's tube "
              "counts (P1-P3, S1-S2, secondary shares, marrow-content index), exclusions and consistency flags (932)",
              required=["sample_id", "mouse_id", "tissue", "zt", "cohort"])
    write_csv(tests, "r9_flow", "flow_tests.csv",
              "Flow validation cohort (O02, D89): pre-specified tests - P1-P3 OLS log ratio ZT17/ZT5 + cohort with Holm; "
              "without consistency-flagged samples; D17 sensitivity; S1-S2 mixed models (tissue x ZT, mouse intercept) (932)",
              required=["test", "family", "tissue", "effect", "ci_lo", "ci_hi", "p"])
    cons = r[["sample_id", "tissue", "zt", "cd11b_pct_cells_T2", "cd11b_pct_cells_T3", "flag_cd11b_T2_T3",
              "b220_pct_cells_T6", "b220_pct_cells_T7", "flag_b220_T6_T7"]]
    write_csv(cons, "r9_flow", "flow_consistency.csv",
              "Flow validation cohort: cross-tube consistency checks of section 6.2 (CD11b+ T2 vs T3, B220+ T6 vs T7; "
              "flag = relative difference > 20 %) (932)", required=["sample_id", "flag_cd11b_T2_T3"])
    dg = tv[tv.control_type.fillna("").str.startswith("digestion")].copy()
    dg["marker"] = np.where(dg.tube == "T4", "CD62L", "CXCR4")
    dg["state"] = np.where(dg.control_type.str.endswith("undigested"), "undigested", "digested")
    dc = dg.pivot_table(index=["cohort", "zt_group", "marker"], columns="state",
                        values=["median_bead_relative_FI", "target_pct"], aggfunc="first")
    dc.columns = [f"{a}_{b}" for a, b in dc.columns]
    dc = dc.reset_index()
    dc["median_ratio_digested_undigested"] = dc.median_bead_relative_FI_digested / dc.median_bead_relative_FI_undigested
    dc["pct_pos_change_points"] = dc.target_pct_digested - dc.target_pct_undigested
    for m_ in ("CD62L", "CXCR4"):
        x = dc[dc.marker == m_]
        log(f"digestion control {m_}: median x{x.median_ratio_digested_undigested.min():.2f}-"
            f"{x.median_ratio_digested_undigested.max():.2f}, % positive {x.pct_pos_change_points.min():+.0f} to "
            f"{x.pct_pos_change_points.max():+.0f} points (4 processing days)")
    write_csv(dc.assign(source_md5=src), "r9_flow", "flow_digestion_controls.csv",
              "Flow validation cohort: digestion controls per processing day (cohort x ZT group) - marrow cells digested vs "
              "undigested, CD62L (T4) and CXCR4 (T5) median and % positive of Ly6G+ (932)",
              required=["cohort", "zt_group", "marker", "median_ratio_digested_undigested"])
    # figure tables
    sh = []
    for key, tis, pred, lab, fid in PRIMARY:
        x = pt[pt.test == key].iloc[0]
        d = r[r.tissue == tis]
        for _, s in d.iterrows():
            sh.append(dict(readout=key, label=lab, tissue=tis, sample_id=s.sample_id, mouse_id=s.mouse_id,
                           cohort=s.cohort, zt=s.zt, value=s.get(key, np.nan), included=bool(np.isfinite(s.get(key, np.nan))),
                           exclusion=s.exclusions if not np.isfinite(s.get(key, np.nan)) else "",
                           ratio_zt17_zt5=x.effect, ci_lo=x.ci_lo, ci_hi=x.ci_hi, p=x.p, p_holm=x.p_holm,
                           verdict=x.verdict, n_zt5=x.n_zt5, n_zt17=x.n_zt17))
    write_csv(pd.DataFrame(sh), "r9_figures", "f4f_flow_shares.csv",
              "Fig. 5c: flow validation of the timed immune shares - per mouse P1 (cochlear immature neutrophils), P2 "
              "(cochlear classical monocytes), P3 (marrow recirculating B), % of CD45+, with the pre-specified tests (932)",
              required=["readout", "tissue", "zt", "value", "included", "ratio_zt17_zt5", "p_holm"])
    mk = []
    for key, test, lab, link in SECONDARY:
        for _, s in r.iterrows():
            mk.append(dict(readout=key, test=test, label=lab, tissue=s.tissue, sample_id=s.sample_id, mouse_id=s.mouse_id,
                           cohort=s.cohort, zt=s.zt, value=s.get(key, np.nan),
                           included=bool(np.isfinite(s.get(key, np.nan))),
                           exclusion=s.exclusions if not np.isfinite(s.get(key, np.nan)) else ""))
    mk = pd.DataFrame(mk)
    for key, test, lab, link in SECONDARY:
        for part in ("cochlea", "marrow", "interaction"):
            x = st[(st.readout == lab) & (st.tissue == part)].iloc[0]
            mk.loc[mk.readout == key, f"{part}_effect"] = x.effect
            mk.loc[mk.readout == key, f"{part}_p"] = x.p
    write_csv(mk, "r9_figures", "f5i_flow_markers.csv",
              "Fig. 6h: flow validation of the neutrophil markers - per mouse x tissue CD62L-hi and CXCR4-hi % of Ly6G+ and "
              "medians, with the mixed-model contrasts (cochlea, marrow ZT17/ZT5; interaction) (932)",
              required=["readout", "tissue", "zt", "value", "included"])
    cols = ["record_id", "tube", "FSC_A", "SSC_A", "FSC_H", "channel1_bead_relative", "channel2_bead_relative",
            "gate_cells", "gate_singlets"]
    ev = pd.read_csv(FLOW / "events.csv", usecols=cols, encoding="utf-8-sig",
                     dtype={"record_id": "category", "tube": "category", "gate_cells": "int8", "gate_singlets": "int8"})
    chan = pd.read_csv(FLOW / "channels.csv", encoding="utf-8-sig")
    plots, cont, outl, gts = plot_export(ev, gates, r, tv, chan)
    write_csv(plots, "r9_figures", "flow_plots.csv",
              "Flow plots (Fig. 5c, 6h, Supplementary Fig. 5; D92, renumbered D133, D134, D136): sample, plotted parent population, marker-fluorochrome "
              "axis labels, display axes (asinh(bead-relative / 0.05); scatter x 10^-3) and contour smoothing (932)",
              required=["plot", "role", "figure", "record_id", "population", "x_label", "y_label", "n_events"])
    write_csv(cont, "r9_figures", "flow_contours.csv",
              "Flow plots: probability-contour vertices (event-density quantiles 5, 10, 20, ..., 90 %; display "
              "coordinates) of the plots in flow_plots.csv (932, D92)", required=["plot", "level", "path", "x", "y"])
    write_csv(outl, "r9_figures", "flow_outliers.csv",
              "Flow plots: the events outside the outermost (95 %) contour, display coordinates (932, D92)",
              required=["plot", "x", "y"])
    write_csv(gts, "r9_figures", "flow_gates.csv",
              "Flow plots: gate polygons (display coordinates), events in gate / plotted and the frequency shown (% of "
              "the plotted parent from the tube counts, equal to the event count within 0.25 points) (932, D92)",
              required=["plot", "gate_name", "vertices_x", "vertices_y", "pct_shown"])
    log("done")


if __name__ == "__main__":
    main()
