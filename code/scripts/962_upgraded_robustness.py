"""Round 10 (D142) — robustness of the two upgraded quantification rules used in Fig. 4g and 4i, computed from the
per-object features they store (no image re-reading; the detection steps were swept in 939):
1  survey cell rule (960): local GFP contrast 1.5 / 2 / 3, minimum area 15 / 25 / 40 um2, nucleus share 0 / 0.1 / 0.25
   (fields with DAPI) -> cochlea / marrow density ratio per wild-type pair (published setting: 2, 25, 0.1).
2  nucleus-anchored typing (961): ring coverage 0.4 / 0.5 / 0.6, local marker contrast 1.5 / 2 / 3, positive ring
   sectors 4 / 5 / 6 of 8 (typing), x GFP body overlap 0.3 / 0.5 / 0.7 and local GFP contrast 1.5 / 2 / 3 (GFP call)
   -> partner-derived share per cell type and compartment (published setting: 0.5, 2, 5; 0.5, 2).
Outputs (lib/io.write_csv, results/r10_parabiosis/): robustness_survey_cells.csv, robustness_celltype_v2.csv and
robustness_upgraded_claims.csv (per claim: settings, settings at which it holds, estimate range).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/962_upgraded_robustness.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r10_parabiosis"


def survey():
    b = pd.read_csv(RESULTS / MOD / "survey_cells_bodies.csv")
    sm = pd.read_csv(RESULTS / MOD / "survey_cells_summary.csv").set_index(["animal", "compartment"])
    rows = []
    for con, amin, nf in itertools.product((1.5, 2.0, 3.0), (15.0, 25.0, 40.0), (0.0, 0.1, 0.25)):
        cell = ((b.area_um2 >= amin) & (b.local_contrast >= con) & ~b.broad_spectrum.astype(bool)
                & (~b.has_dapi.astype(bool) | (b.nucleus_frac >= nf)))
        base = (con, amin, nf) == (2.0, 25.0, 0.1)
        if base:
            assert (cell == b.cell.astype(bool)).all(), "published setting does not reproduce 960"
        cnt = b.assign(c=cell).groupby(["animal", "compartment"]).c.sum()
        for an in sorted(b.animal.unique()):
            dm = cnt.get((an, "marrow"), 0) / sm.loc[(an, "marrow"), "tissue_mm2"]
            dc = cnt.get((an, "cochlea"), 0) / sm.loc[(an, "cochlea"), "tissue_mm2"]
            rows.append(dict(contrast=con, area_min=amin, nucleus_frac=nf, animal=an, marrow_density=dm,
                             cochlea_density=dc, cochlea_to_marrow=dc / dm if dm > 0 else np.nan, baseline=base))
    return pd.DataFrame(rows)


def celltypes():
    c = pd.read_csv(RESULTS / MOD / "celltype_v2_cells.csv")
    c = c[c.included.astype(bool)].copy()
    c["comp"] = np.where(c.compartment.isin(["marrow", "vestibular bone"]), "marrow", c.compartment)
    c = c[c.comp.isin(["marrow", "cochlea"])]
    rows = []
    for rf, mc, sec, ov, gc in itertools.product((0.4, 0.5, 0.6), (1.5, 2.0, 3.0), (4, 5, 6), (0.3, 0.5, 0.7),
                                                 (1.5, 2.0, 3.0)):
        typed = (c.ring_pos_frac >= rf) & (c.marker_contrast >= mc) & (c.ring_sectors >= sec)
        gfp = (c.gfp_overlap >= ov) & (c.gfp_contrast >= gc)
        base = (rf, mc, sec, ov, gc) == (0.5, 2.0, 5, 0.5, 2.0)
        if base:
            assert (typed == c.typed.astype(bool)).all() and (gfp[typed] == c.gfp_cell[typed].astype(bool)).all()
        t = c[typed].assign(g=gfp[typed])
        for (mk, cp), g in t.groupby(["marker", "comp"]):
            rows.append(dict(ring_frac=rf, marker_contrast=mc, ring_sectors=sec, gfp_overlap=ov, gfp_contrast=gc,
                             marker=mk, compartment=cp, typed_cells=len(g), gfp_pos_cells=int(g.g.sum()),
                             gfp_pos_pct=100 * g.g.mean(), baseline=base))
    return pd.DataFrame(rows)


def main():
    sv = survey()
    ct = celltypes()
    claims = []
    for name, pairs in (("cochlea / marrow < 1 at 14 and 54 days", ["14 d", "54 d"]),
                        ("cochlea / marrow < 1 in every pair", ["14 d", "31 d", "31 d noise", "54 d"])):
        g = sv[sv.animal.isin(pairs)]
        ok = g.groupby(["contrast", "area_min", "nucleus_frac"]).cochlea_to_marrow.apply(lambda v: bool((v < 1).all()))
        claims.append(dict(claim=name, rule="survey cells (960)", settings=len(ok), holds=int(ok.sum()),
                           estimate_min=g.cochlea_to_marrow.min(), estimate_max=g.cochlea_to_marrow.max(),
                           unit="cochlea / marrow ratio"))
    marrow_b = ct[(ct.marker == "CD19") & (ct.compartment == "marrow")].set_index(
        ["ring_frac", "marker_contrast", "ring_sectors", "gfp_overlap", "gfp_contrast"]).gfp_pos_pct
    for name, mk, cp, fn in (("no cochlear leukocyte partner-derived", "CD45", "cochlea", lambda v: v == 0),
                             ("marrow leukocytes partner-derived above zero", "CD45", "marrow", lambda v: v > 0),
                             ("marrow neutrophils below 5% partner-derived", "FUT4", "marrow", lambda v: v < 5),
                             ("reporter-negative B-cell background below 10%", "CD19", "marrow", lambda v: v < 10)):
        g = ct[(ct.marker == mk) & (ct.compartment == cp)]
        ok = fn(g.gfp_pos_pct)
        claims.append(dict(claim=name, rule="nucleus-anchored typing (961)", settings=len(g), holds=int(ok.sum()),
                           estimate_min=g.gfp_pos_pct.min(), estimate_max=g.gfp_pos_pct.max(), unit="% GFP+ of typed"))
    g = ct[(ct.marker == "CD45") & (ct.compartment == "marrow")].set_index(
        ["ring_frac", "marker_contrast", "ring_sectors", "gfp_overlap", "gfp_contrast"]).gfp_pos_pct
    common = g.index.intersection(marrow_b.index)
    above = (g.loc[common] > marrow_b.loc[common])
    claims.append(dict(claim="marrow leukocytes above the B-cell background", rule="nucleus-anchored typing (961)",
                       settings=len(common), holds=int(above.sum()),
                       estimate_min=float((g.loc[common] - marrow_b.loc[common]).min()),
                       estimate_max=float((g.loc[common] - marrow_b.loc[common]).max()),
                       unit="difference, percentage points"))
    cl = pd.DataFrame(claims)
    for r in cl.itertuples():
        log(f"{r.claim}: holds at {r.holds}/{r.settings} settings ({r.estimate_min:.3g}-{r.estimate_max:.3g} {r.unit})")
    write_csv(sv, MOD, "robustness_survey_cells.csv", "Robustness (962, D142): cochlea / marrow density ratio per "
              "wild-type pair across the settings of the upgraded survey cell rule (local contrast, minimum area, "
              "nucleus share)", required=["contrast", "area_min", "nucleus_frac", "animal", "cochlea_to_marrow"])
    write_csv(ct, MOD, "robustness_celltype_v2.csv", "Robustness (962, D142): partner-derived share per cell type and "
              "compartment across the settings of the nucleus-anchored typing (ring coverage, marker contrast, ring "
              "sectors) and GFP call (body overlap, GFP contrast)", required=["marker", "compartment", "gfp_pos_pct",
                                                                             "baseline"])
    write_csv(cl, MOD, "robustness_upgraded_claims.csv", "Robustness (962, D142): each claim of Fig. 4g / 4i with the "
              "number of settings tested and at which it holds", required=["claim", "settings", "holds"])


if __name__ == "__main__":
    main()
