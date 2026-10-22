"""Round 10 (D153) — a stricter cell rule, calibrated on the blinded visual labels and applied to every tissue alike.
User: "The current quantitative criteria for the cochlea are too broad; they should be made more stringent, as the
current cell counts are too high." The blinded reviews had shown that the upgraded rule (960) over-counts: precision
0.57 against the labels of 946, and 2 of 11, 7 of 13 and 1 of 3 kept cochlear bodies confirmed in 960's sample. The rule
is tightened for every tissue (never for one tissue alone: the tissues must stay comparable, D152), with its thresholds
chosen on the labels, not on the counts.

Labels: every decided blinded label (yes / no; 'unsure' excluded) of 934's GFP+ bodies from 946 (a random sample within
pair and compartment, labelled before the upgraded rule existed) and from 960 (a sample within pair, compartment and
call), joined to the bodies' features (960 / 963; each section's GFP tissue robust SD from 934, for the rendered liver
exports derived from 934's objects).
Grid (fixed before any strict count was computed): stricter versions of each criterion of the upgraded rule,
  GFP signal above the tissue median   >= 10 (current), 15, 20, 30 robust SD of the section's tissue GFP
  local contrast                        >= 2 (current), 3, 4, 6 x the median GFP of the tissue 2-6 um beyond the body
  area                                  >= 25 (current), 40, 60 um2
  nucleus (native fields with DAPI)     >= 10% (current), 30%, 50% of the body inside a DAPI nucleus
  not broad-spectrum                    unchanged.
Choice: the grid rule with the highest F1 against the labels (pooled over tissues and pairs); ties, the higher precision,
then the fewer kept bodies. Out-of-sample check: the choice repeated with one animal's labels left out at a time and
scored on the left-out labels (leave-one-animal-out, pooled over folds).
Two formats: native (the survey's sections; 960 / 963 features) and the brain image's format (966: sections box-averaged
to 2.773 um per pixel, no nucleus criterion), where each labelled body is matched to the converted body of the same
section within 8 um of its position (an unmatched labelled body counts as not kept). In the brain format the signal
criterion stays at the detection threshold (10 robust SD) and only contrast and area are tightened: the brain image's
rendered GFP saturates its bodies (every brain body at the 8-bit ceiling, at most 10.7 robust SD above the tissue
median), so any higher signal threshold excludes every brain body whatever it is. This constraint was added after the
first run of the full grid had chosen 20 SD for both formats and so excluded all 116 brain bodies (disclosed, D153).
Application: the chosen rules recount every wild-type field (native: 960 / 963 bodies; brain format: 966 bodies at every
pixel size and both conversions) and the brain projection (964 bodies, brain format), beside the upgraded rule.
Outputs (lib/io.write_csv, results/r10_parabiosis/): strict_rule_labels.csv (the labelled bodies with both formats'
features), strict_rule_grid.csv (every grid rule x format: kept, TP, FP, FN, TN, precision, sensitivity, F1),
strict_rule_choice.csv (the chosen rule per format, overall and by compartment, and leave-one-animal-out),
strict_densities.csv (wild-type animal x tissue x format x conversion x pixel size: cells and density by the upgraded
and the strict rule).
Run (local; the brain image is local): COCH_ROOT=D:/coch-rythum python scripts/968_strict_rule.py
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import filters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S964 = load_module("s964", "964_tissue_comparison.py")
S934 = S964.S934
MOD = "r10_parabiosis"
R10 = RESULTS / MOD
GRID = dict(signal_sd=[10, 15, 20, 30], contrast=[2, 3, 4, 6], area=[25, 40, 60], nucleus=[0.1, 0.3, 0.5])
GRID_BF = dict(GRID, signal_sd=[10])      # brain format: signal at the detection threshold (the brain image saturates)
MATCH_UM = 8.0
UM = S964.UM_MAIN


def plane_sd():
    """GFP tissue robust SD per analysed section (934's parabiosis_planes); for the rendered liver exports, which 934
    analysed apart, derived from its objects (threshold = median + 10 SD; median = body mean - signal)."""
    pl = pd.read_csv(R10 / "parabiosis_planes.csv")[["path", "plane", "sd"]]
    ob = pd.read_csv(R10 / "parabiosis_objects.csv")
    ob = ob[~ob.path.isin(pl.path)]
    ob = ob.assign(sd=(ob.threshold - (ob.gfp_mean - ob.signal)) / 10.0)
    d = ob.groupby(["path", "plane"]).sd.agg(["min", "max"]).reset_index()
    assert np.allclose(d["min"], d["max"], rtol=1e-6)
    return pd.concat([pl, d.rename(columns={"min": "sd"})[["path", "plane", "sd"]]], ignore_index=True)


def labels():
    l1 = pd.read_csv(R10 / "verification_bodies_labelled.csv").assign(sample="946 (random, before the rule)")
    l2 = pd.read_csv(R10 / "verification_survey_labelled.csv").assign(sample="960 (by call)")
    cols = ["item_id", "sample", "animal", "compartment", "path", "plane", "y_px", "x_px", "gfp_cell"]
    lab = pd.concat([l1[cols], l2[cols]], ignore_index=True)
    lab = lab[lab.gfp_cell.isin(["yes", "no"])].copy()
    lab["truth"] = lab.gfp_cell == "yes"
    # native features: nearest 960 / 963 body of the same section (the bodies are 934's, so the match is exact)
    b = pd.concat([pd.read_csv(R10 / "survey_cells_bodies.csv"), pd.read_csv(R10 / "survey_cells_bodies_extra.csv")],
                  ignore_index=True)
    b = b.merge(plane_sd(), on=["path", "plane"], how="left")
    assert b.sd.notna().all()
    b["signal_sd"] = b.signal / b.sd
    rows = []
    for r in lab.itertuples():
        c = b[(b.path == r.path) & (b.plane == r.plane)]
        d = np.hypot(c.y - r.y_px, c.x - r.x_px)
        assert len(c) and d.min() <= 1.5, (r.item_id, d.min() if len(c) else None)
        rows.append(c.loc[d.idxmin(), ["area_um2", "signal_sd", "local_contrast", "nucleus_frac", "has_dapi",
                                       "broad_spectrum", "cell"]].to_dict())
    lab = pd.concat([lab.reset_index(drop=True), pd.DataFrame(rows).add_prefix("native_")], axis=1)
    lab = lab.drop_duplicates(["path", "plane", "y_px", "x_px"]).reset_index(drop=True)
    # brain-format features: the converted body of the same section within 8 um (966, sections at 2.773 um per pixel)
    mb = pd.read_csv(R10 / "matched_format_bodies.csv")
    mb = mb[(mb.conversion == "section") & (mb.um_per_px.round(3) == UM)].copy()
    mb["signal_sd"] = mb.signal / mb.plane_sd
    f_rows = []
    for r in lab.itertuples():
        c = mb[(mb.path == r.path) & (mb.section_z == r.plane)]
        um0 = float(c.native_um_per_px.iloc[0]) if len(c) else np.nan
        d = np.hypot(c.y * UM - r.y_px * um0, c.x * UM - r.x_px * um0) if len(c) else pd.Series(dtype=float)
        if len(c) and d.min() <= MATCH_UM:
            m = c.loc[d.idxmin()]
            f_rows.append(dict(matched=True, area_um2=m.area_um2, signal_sd=m.signal_sd,
                               local_contrast=m.local_contrast, broad_spectrum=bool(m.broad_spectrum),
                               cell=bool(m.cell), distance_um=float(d.min())))
        else:
            f_rows.append(dict(matched=False))
    return pd.concat([lab, pd.DataFrame(f_rows).add_prefix("bf_")], axis=1)


def keep_native(t, s, c, a, n):
    return ((t.signal_sd >= s) & (t.local_contrast >= c) & (t.area_um2 >= a) & ~t.broad_spectrum.astype(bool) &
            (~t.has_dapi.astype(bool) | (t.nucleus_frac >= n)))


def keep_bf(t, s, c, a):
    return (t.signal_sd >= s) & (t.local_contrast >= c) & (t.area_um2 >= a) & ~t.broad_spectrum.astype(bool)


def score(kept, truth):
    tp, fp = int((kept & truth).sum()), int((kept & ~truth).sum())
    fn, tn = int((~kept & truth).sum()), int((~kept & ~truth).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    sens = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(kept=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, sensitivity=sens, f1=f1)


def native_view(lab):
    t = lab[[c for c in lab.columns if c.startswith("native_")]].copy()
    t.columns = [c[len("native_"):] for c in t.columns]
    return t


def bf_view(lab):
    t = lab[[c for c in lab.columns if c.startswith("bf_")]].copy()
    t.columns = [c[len("bf_"):] for c in t.columns]
    t["signal_sd"] = t.signal_sd.fillna(-np.inf)
    t["local_contrast"] = t.local_contrast.fillna(-np.inf)
    t["area_um2"] = t.area_um2.fillna(-np.inf)
    t["broad_spectrum"] = t.broad_spectrum.fillna(True).astype(bool)
    return t


def rules(fmt):
    keys = ["signal_sd", "contrast", "area"] + (["nucleus"] if fmt == "native" else [])
    grid = GRID if fmt == "native" else GRID_BF
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


def kept_by(fmt, t, r):
    if fmt == "native":
        return keep_native(t, r["signal_sd"], r["contrast"], r["area"], r["nucleus"])
    return keep_bf(t, r["signal_sd"], r["contrast"], r["area"])


def choose(fmt, lab):
    t = native_view(lab) if fmt == "native" else bf_view(lab)
    best, rows = None, []
    for r in rules(fmt):
        sc = score(kept_by(fmt, t, r), lab.truth)
        rows.append(dict(format=fmt, **r, **sc))
        key = (sc["f1"], np.nan_to_num(sc["precision"]), -sc["kept"])
        if best is None or key > best[0]:
            best = (key, r)
    return best[1], pd.DataFrame(rows)


def apply_native(rule):
    tf = pd.read_csv(R10 / "parabiosis_tissue_fields.csv")
    w = tf[(tf.partner == "WT") & ~tf.noise.astype(bool)]
    b = pd.concat([pd.read_csv(R10 / "survey_cells_bodies.csv"), pd.read_csv(R10 / "survey_cells_bodies_extra.csv")],
                  ignore_index=True).merge(plane_sd(), on=["path", "plane"], how="left")
    assert b.sd.notna().all()
    b["signal_sd"] = b.signal / b.sd
    b["strict"] = keep_native(b, rule["signal_sd"], rule["contrast"], rule["area"], rule["nucleus"])
    c = b.groupby("path").agg(cells_upgraded=("cell", "sum"), cells_strict=("strict", "sum"))
    w = w.join(c, on="path")
    w[["cells_upgraded", "cells_strict"]] = w[["cells_upgraded", "cells_strict"]].fillna(0)
    g = w.groupby(["animal", "compartment"]).agg(fields=("path", "size"), tissue_mm2=("tissue_mm2", "sum"),
                                                 cells_upgraded=("cells_upgraded", "sum"),
                                                 cells_strict=("cells_strict", "sum")).reset_index()
    return g.assign(format="native", conversion="section", um_per_px=np.nan)


def brain_sd(um):
    lec, gfp = S964.rendered(S964.BRAIN, 1), S964.rendered(S964.BRAIN, 2)
    tis = S934.tissue_mask([gfp, lec], um, [max(float(np.percentile(c, 99.5)), 1.0) for c in (gfp, lec)])
    return S934.robust(filters.gaussian(gfp, sigma=0.5 / um, preserve_range=True)[tis])[1]


def apply_bf(rule):
    mf = pd.read_csv(R10 / "matched_format_fields.csv")
    mb = pd.read_csv(R10 / "matched_format_bodies.csv")
    mb["signal_sd"] = mb.signal / mb.plane_sd
    mb["strict"] = keep_bf(mb, rule["signal_sd"], rule["contrast"], rule["area"])
    c = mb.groupby(["path", "conversion", "um_per_px"]).agg(cells_strict=("strict", "sum"))
    mf = mf.join(c, on=["path", "conversion", "um_per_px"])
    mf["cells_strict"] = mf.cells_strict.fillna(0)
    g = (mf.groupby(["animal", "compartment", "conversion", "um_per_px"])
         .agg(fields=("path", "size"), tissue_mm2=("tissue_mm2", "sum"), cells_upgraded=("gfp_cells", "sum"),
              cells_strict=("cells_strict", "sum")).reset_index())
    bb = pd.read_csv(R10 / "tissue_comparison_brain_bodies.csv")
    bd = pd.read_csv(R10 / "tissue_comparison_density.csv")
    bd = bd[bd.compartment == "brain"].set_index("um_per_px")
    rows = []
    for um, x in bb.groupby("um_per_px"):
        x = x.assign(signal_sd=x.signal / brain_sd(um))
        k = keep_bf(x, rule["signal_sd"], rule["contrast"], rule["area"])
        rows.append(dict(animal="31 d", compartment="brain", conversion="projection", um_per_px=um, fields=1,
                         tissue_mm2=float(bd.loc[um, "tissue_mm2"]), cells_upgraded=int(x.cell.sum()),
                         cells_strict=int(k.sum())))
    return pd.concat([g, pd.DataFrame(rows)], ignore_index=True).assign(format="brain image")


def main():
    lab = labels()
    log(f"labels: {len(lab)} decided ({int(lab.truth.sum())} cells), brain format matched {int(lab.bf_matched.sum())}")
    out_rows, grids, chosen = [], [], {}
    for fmt in ("native", "brain image"):
        rule, grid = choose(fmt, lab)
        grids.append(grid)
        chosen[fmt] = rule
        t = native_view(lab) if fmt == "native" else bf_view(lab)
        cur = dict(signal_sd=10, contrast=2, area=25, nucleus=0.1)
        for name, r in (("upgraded (current)", cur), ("strict (chosen)", rule)):
            k = kept_by(fmt, t, r)
            out_rows.append(dict(format=fmt, rule=name, scope="all labels", **{f"thr_{k_}": v for k_, v in r.items()},
                                 **score(k, lab.truth)))
            for comp in ("cochlea", "marrow"):
                m = (lab.compartment == comp).values
                out_rows.append(dict(format=fmt, rule=name, scope=comp, **{f"thr_{k_}": v for k_, v in r.items()},
                                     **score(k[m], lab.truth[m])))
        # leave-one-animal-out: choose without an animal, score on it
        kept_cv = pd.Series(False, index=lab.index)
        for an in sorted(lab.animal.unique()):
            tr = lab[lab.animal != an]
            r_, _ = choose(fmt, tr)
            te = lab.animal == an
            kept_cv[te] = kept_by(fmt, t[te], r_).values
        out_rows.append(dict(format=fmt, rule="strict, leave one animal out", scope="all labels",
                             **score(kept_cv, lab.truth)))
        log(f"{fmt}: chosen {rule}")
    choice = pd.DataFrame(out_rows)
    grid = pd.concat(grids, ignore_index=True)
    dens = pd.concat([apply_native(chosen["native"]), apply_bf(chosen["brain image"])], ignore_index=True)
    dens["days"] = dens.animal.str.extract(r"(\d+)").astype(int)
    dens["density_upgraded_per_mm2"] = dens.cells_upgraded / dens.tissue_mm2
    dens["density_strict_per_mm2"] = dens.cells_strict / dens.tissue_mm2
    dens["main"] = (dens.format == "native") | (dens.um_per_px.round(3) == UM)
    write_csv(lab, MOD, "strict_rule_labels.csv", "Strict rule (968, D153): decided blinded labels of GFP+ bodies (946, "
              "960) with their native and brain-format features", required=["item_id", "truth", "native_cell"])
    write_csv(grid, MOD, "strict_rule_grid.csv", "Strict rule (968, D153): every grid rule per format scored against the "
              "blinded labels", required=["format", "signal_sd", "contrast", "area", "precision", "sensitivity", "f1"])
    write_csv(choice, MOD, "strict_rule_choice.csv", "Strict rule (968, D153): the current and the chosen rule per format, "
              "overall and by compartment, and the leave-one-animal-out choice", required=["format", "rule", "scope",
                                                                                          "precision", "sensitivity"])
    write_csv(dens, MOD, "strict_densities.csv", "Strict rule (968, D153): partner-derived cells per mm2 per wild-type "
              "animal and tissue by the upgraded and the strict rule, native and in the brain image's format",
              required=["animal", "compartment", "format", "conversion", "um_per_px", "density_strict_per_mm2", "main"])
    pd.set_option("display.width", 220)
    print(choice.round(3).to_string())
    m = dens[dens.main & ((dens.format == "native") | (dens.conversion == "section") | (dens.compartment == "brain"))]
    print(m[["animal", "compartment", "format", "conversion", "cells_upgraded", "cells_strict",
             "density_upgraded_per_mm2", "density_strict_per_mm2"]].round(2).to_string())


if __name__ == "__main__":
    main()
