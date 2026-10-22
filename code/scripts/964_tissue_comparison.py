"""Round 10 (D149, D150) — the ear (cochlea) against brain, liver and blood: partner-derived (Lyz2-EGFP+) cells in the
wild-type parabionts. User: "For the brain, you can perform quantitative analysis on the TIFF images; your methodology is
correct" and "compare the ear, brain, liver, and blood (using external data for the blood)" (D150). No spleen was imaged
(D149), and no blood preparation carries an animal or partner label.

Brain. The collection's only brain image is a rendered orthogonal projection of the 31-day wild-type partner
(organs/2025-01-13_Parabiosis-WT-31d-Lectin-GFP-DAPI/Brain/New-03-Orthogonal Projection-02: _c1 lectin, rendered magenta;
_c2 GFP, rendered green; 8-bit RGB, 1,024 x 1,024 pixels; no DAPI channel, no pixel size; its source acquisition is not
in the collection). Each rendering is read as the maximum over its colour channels (as 963 reads the liver exports) and
counted by 960's body_features, i.e. 934's tissue mask and GFP+ bodies and 960's upgraded rule (area >= 25 um2, mean GFP
>= 2x the median GFP of the tissue 2-6 um beyond the body, not broad-spectrum against lectin); the nucleus criterion
applies only where a field has DAPI, so the brain is counted as the survey's fields without DAPI are. Pixel size: the
collection's dominant overview setting for 1,024-pixel frames, 2.773 um (21 acquisitions, 5x; the field then spans
2.84 mm, consistent with the striatal fibre bundles it shows); every other overview pixel size of 1,024-pixel frames in
the collection (1.387, 1.248 and 1.040 um) is reported as a sensitivity range. A projection collects the cells of its
whole depth, and coarse pixels miss small and dim cells, so the brain density is not on the footing of the optical
sections of the other tissues. The liver projection of the same session (New-01) is not countable: its GFP rendering is
bright throughout, and the body threshold (tissue median + 10 robust SD) lies above the 8-bit range; the liver is
counted on its three 20X exports with DAPI (934 / 963).

Tissue densities (one unit): partner-derived cells per mm2 of tissue by the upgraded rule per wild-type animal and tissue
(934 fields with the 960 / 963 calls; the noise-exposed pair excluded), with the rule without the nucleus criterion
alongside (the brain's rule).

Blood (external, public, D150). Published parabiosis: blood neutrophils of wild-type partners were 22.1 +/- 2.0% (mean
+/- SEM, n = 18) partner-derived after one month with Lyz2-GFP or CD45.1 partners, tissue neutrophils matched blood
except in the marrow, and no neutrophils were found in the brain (Casanova-Acebes et al. 2018, J Exp Med,
doi:10.1084/jem.20181468; methods and Fig. 1); blood neutrophils and Ly6C-high monocytes were about 30% partner-derived
after 60 days with UBC-GFP partners, with liver and spleen near blood and the dura lower (Cugurra et al. 2021, Science,
doi:10.1126/science.abf7844). Shares are compared in one unit, the partner-derived % of cells: blood neutrophils
(external) against the 54-day partner's typed cochlear neutrophils (931, 60x lateral wall) and marrow neutrophils (961),
and the perilymph cells (nucleated, mostly FUT4+ neutrophils; 14 and 28 days) raw and per labelled cell (divided by the
GFP partner's share of the same pair). Intervals: Clopper-Pearson 95% for counts; per labelled cell, the 2.5-97.5%
quantiles of the ratio of two Jeffreys posteriors (10,000 draws, seed 964).

Outputs (lib/io.write_csv, results/r10_parabiosis/): tissue_comparison_brain_bodies.csv (every GFP+ body of the brain
projection at each pixel size, with the upgraded features and call), tissue_comparison_density.csv,
tissue_comparison_blood_external.csv, tissue_comparison_share.csv.
Run (local; the images are not on the server): COCH_ROOT=D:/coch-rythum python scripts/964_tissue_comparison.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S960 = load_module("s960", "960_survey_cells.py")
S934 = S960.S934
MOD = "r10_parabiosis"
ORGANS = "organs/2025-01-13_Parabiosis-WT-31d-Lectin-GFP-DAPI"
BRAIN = f"{ORGANS}/Brain/New-03-Orthogonal Projection-02/New-03-Orthogonal Projection-02"
LIVER_PROJ = f"{ORGANS}/Liver/New-01-Orthogonal Projection-01/New-01-Orthogonal Projection-01"
UM_MAIN = 2.773
SEED, DRAWS = 964, 10000
BLOOD = [dict(source="Casanova-Acebes et al. 2018, J Exp Med", doi="10.1084/jem.20181468", pairing_days=30,
              partner="Lyz2-GFP or CD45.1", cell_type="neutrophils", partner_derived_pct=22.1, spread_pct=2.0,
              spread="SEM", n_mice=18, method="flow cytometry",
              tissues="tissue neutrophils matched blood except in the bone marrow; no neutrophils found in the brain"),
         dict(source="Cugurra et al. 2021, Science", doi="10.1126/science.abf7844", pairing_days=60,
              partner="UBC-GFP", cell_type="neutrophils and Ly6C-high monocytes", partner_derived_pct=30.0,
              spread_pct=np.nan, spread="approximate (GFP:WT about 30:70)", n_mice=np.nan, method="flow cytometry",
              tissues="liver and spleen near blood; cranial and spinal dura lower")]


def rendered(stem, k):
    return tifffile.imread(S934.SRC / f"{stem}_c{k}.tif").astype(np.float32).max(axis=-1)


def overview_pixels():
    """Pixel sizes of the collection's overview acquisitions with 1,024-pixel frames (count per size)."""
    a = pd.read_csv(RESULTS / MOD / "parabiosis_acquisitions.csv")
    a = a[(a.um_per_px > 0.70) & ((a.fov_x_um / a.um_per_px).round() == 1024)]
    n = a.um_per_px.round(3).value_counts()
    assert n.idxmax() == UM_MAIN and n.max() >= 20, n
    return n


def brain(pixels):
    lec, gfp = rendered(BRAIN, 1), rendered(BRAIN, 2)
    a = np.stack([lec, gfp])[:, None]
    rows, summ = [], []
    for um, n_acq in pixels.items():
        meta = dict(um_x=float(um), um_z=None, channels=["Alexa Fluor 647", "Alexa Fluor 488"])
        planes = list(S934.sampled_planes(a, meta))
        assert len(planes) == 1
        tis = planes[0][1]
        x = S960.body_features(a, meta, BRAIN, {}, 255.0)
        for d in x:
            d.pop("_crop")
            d.update(um_per_px=float(um), main=bool(um == UM_MAIN))
        rows += x
        mm2 = float(tis.sum()) * um * um / 1e6
        n_cell = sum(d["cell"] for d in x)
        summ.append(dict(um_per_px=float(um), overview_acquisitions=int(n_acq), tissue_mm2=mm2, gfp_bodies=len(x),
                         broad_spectrum=sum(d["broad_spectrum"] for d in x), gfp_cells=n_cell,
                         cell_density_per_mm2=n_cell / mm2))
        log(f"brain at {um} um/px: {mm2:.2f} mm2 tissue, {len(x)} bodies, {n_cell} cells ({n_cell / mm2:.2f} per mm2)")
    return pd.DataFrame(rows), pd.DataFrame(summ)


def liver_projection():
    lec, gfp = rendered(LIVER_PROJ, 1), rendered(LIVER_PROJ, 2)
    tis = S934.tissue_mask([gfp, lec], UM_MAIN, [max(float(np.percentile(c, 99.5)), 1.0) for c in (gfp, lec)])
    out, st, _, _ = S934.gfp_bodies(gfp, lec, tis, UM_MAIN)
    sat = float((gfp[tis] >= 254).mean())
    log(f"liver projection: threshold {st['threshold']:.0f} (8-bit max 255), {len(out)} bodies, {100 * sat:.1f}% of "
        "tissue saturated")
    assert st["threshold"] > 255 and not out
    return st["threshold"], sat


def densities(bsum):
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    b = pd.concat([pd.read_csv(RESULTS / MOD / "survey_cells_bodies.csv"),
                   pd.read_csv(RESULTS / MOD / "survey_cells_bodies_extra.csv")], ignore_index=True)
    assert not b.duplicated(["path", "plane", "y", "x"]).any()
    b["no_nucleus"] = ((b.area_um2 >= S960.AREA_MIN) & (b.local_contrast >= S960.CONTRAST) &
                       ~b.broad_spectrum.astype(bool))
    c = b.groupby("path").agg(gfp_cells=("cell", "sum"), cells_no_nucleus=("no_nucleus", "sum"))
    tf = tf[(tf.partner == "WT") & ~tf.noise.astype(bool)].join(c, on="path")
    tf[["gfp_cells", "cells_no_nucleus"]] = tf[["gfp_cells", "cells_no_nucleus"]].fillna(0)
    g = tf.groupby(["animal", "compartment"]).agg(fields=("path", "size"), tissue_mm2=("tissue_mm2", "sum"),
                                                  gfp_bodies=("gfp_bodies", "sum"), gfp_cells=("gfp_cells", "sum"),
                                                  cells_no_nucleus=("cells_no_nucleus", "sum")).reset_index()
    g = g.assign(source="survey fields (934; upgraded rule 960 / 963)", image="optical sections (native or 20X export)",
                 um_per_px=np.nan, main=True)
    br = bsum.assign(animal="31 d", compartment="brain", fields=1, cells_no_nucleus=bsum.gfp_cells,
                     source="brain projection (964; 960 rule, no DAPI)", image="rendered projection, pixel size assumed",
                     main=bsum.um_per_px == UM_MAIN)
    d = pd.concat([g, br[g.columns.tolist()]], ignore_index=True)
    d["days"] = d.animal.str.extract(r"(\d+)").astype(int)
    d["cell_density_per_mm2"] = d.gfp_cells / d.tissue_mm2
    d["no_nucleus_density_per_mm2"] = d.cells_no_nucleus / d.tissue_mm2
    ref = d[(d.animal == "31 d") & (d.compartment == "cochlea")].cell_density_per_mm2.iloc[0]
    d["relative_to_cochlea_31d"] = d.cell_density_per_mm2 / ref
    order = {"liver": 0, "marrow": 1, "vestibular bone": 2, "cochlea": 3, "brain": 4}
    return d.sort_values(["days", "compartment"], key=lambda s: s.map(order) if s.name == "compartment" else s)


def cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def shares():
    rng = np.random.default_rng(SEED)
    pf = pd.read_csv(RESULTS / MOD / "parabiosis_perilymph_fields.csv")
    pf = pf[pf.animal.isin(["14 d", "28 d"]) & pf.partner_label.isin(["WT", "GFP"])]
    pool = pf.groupby(["animal", "partner_label"])[["nuclei", "gfp_pos"]].sum()
    rows = []
    for an in ("14 d", "28 d"):
        k, n = (int(v) for v in pool.loc[(an, "WT"), ["gfp_pos", "nuclei"]])
        kg, ng = (int(v) for v in pool.loc[(an, "GFP"), ["gfp_pos", "nuclei"]])
        lo, hi = cp(k, n)
        r = (rng.beta(k + 0.5, n - k + 0.5, DRAWS) / rng.beta(kg + 0.5, ng - kg + 0.5, DRAWS))
        rows.append(dict(tissue="ear", population="perilymph cells (nucleated)", animal=an, cells=n, partner_derived=k,
                         pct=100 * k / n, ci_lo_pct=100 * lo, ci_hi_pct=100 * hi, gfp_partner_cells=ng,
                         gfp_partner_pos=kg, gfp_partner_pct=100 * kg / ng, per_labelled_pct=100 * (k / n) / (kg / ng),
                         per_labelled_lo_pct=100 * float(np.quantile(r, 0.025)),
                         per_labelled_hi_pct=100 * float(np.quantile(r, 0.975)), source="934 perilymph fields"))
    pq = pd.read_csv(RESULTS / "r9_figures" / "f4_parabiont_fields.csv")
    wl = pq[(pq.field == "60x_sl2") & (pq.compartment == "cochlea")].iloc[0]
    cs = pd.read_csv(RESULTS / MOD / "celltype_v2_summary.csv").set_index(["marker", "compartment"])
    for tissue, pop, k, n, src in [("ear", "cochlear neutrophils (FUT4+, 60x lateral wall)", int(wl.marker_gfp_cells),
                                    int(wl.marker_cells), "931 (60x)"),
                                   ("marrow", "marrow neutrophils (FUT4+)", int(cs.loc[("FUT4", "marrow"),
                                                                                       "gfp_pos_cells"]),
                                    int(cs.loc[("FUT4", "marrow"), "typed_cells"]), "961")]:
        lo, hi = cp(k, n)
        rows.append(dict(tissue=tissue, population=pop, animal="54 d", cells=n, partner_derived=k, pct=100 * k / n,
                         ci_lo_pct=100 * lo, ci_hi_pct=100 * hi, source=src))
    assert (rows[2]["partner_derived"], rows[2]["cells"]) == (0, 13) and (rows[3]["partner_derived"],
                                                                           rows[3]["cells"]) == (5, 306)
    for bl in BLOOD:
        rows.append(dict(tissue="blood", population=f"blood {bl['cell_type']} (published)",
                         animal=f"{bl['pairing_days']} d", pct=bl["partner_derived_pct"],
                         ci_lo_pct=bl["partner_derived_pct"] - 1.96 * bl["spread_pct"] if bl["spread"] == "SEM" else
                         np.nan,
                         ci_hi_pct=bl["partner_derived_pct"] + 1.96 * bl["spread_pct"] if bl["spread"] == "SEM" else
                         np.nan, source=f"{bl['source']} (doi:{bl['doi']})"))
    s = pd.DataFrame(rows)
    lo_blood = min(b["partner_derived_pct"] for b in BLOOD)
    s["relative_to_blood_low"] = s.pct / lo_blood
    s.loc[s.tissue == "blood", "relative_to_blood_low"] = np.nan
    return s


def main():
    pix = overview_pixels()
    bodies, bsum = brain(pix)
    thr, sat = liver_projection()
    dens = densities(bsum)
    sh = shares()
    write_csv(bodies, MOD, "tissue_comparison_brain_bodies.csv", "Ear vs brain, liver and blood (964, D150): GFP+ "
              "bodies of the 31-day wild-type partner's brain projection (rendered, no DAPI) with 960's upgraded features "
              "and call, at the collection's overview pixel sizes (main 2.773 um)",
              required=["um_per_px", "area_um2", "local_contrast", "broad_spectrum", "cell"])
    write_csv(dens, MOD, "tissue_comparison_density.csv", "Ear vs brain, liver and blood (964, D150): partner-derived "
              "cells per mm2 of tissue per wild-type animal and tissue (upgraded rule; without the nucleus criterion "
              "alongside); brain from the rendered projection at each overview pixel size (main = 2.773 um). Liver "
              f"projection not countable (body threshold {thr:.0f} > 255; {100 * sat:.1f}% of tissue saturated)",
              required=["animal", "compartment", "tissue_mm2", "gfp_cells", "cell_density_per_mm2", "main"])
    write_csv(pd.DataFrame(BLOOD), MOD, "tissue_comparison_blood_external.csv", "Ear vs brain, liver and blood (964, "
              "D150): published partner-derived share of blood myeloid cells in wild-type parabionts (external, public)",
              required=["source", "doi", "pairing_days", "partner_derived_pct"])
    write_csv(sh, MOD, "tissue_comparison_share.csv", "Ear vs brain, liver and blood (964, D150): partner-derived share "
              "of ear cells (perilymph, 14 and 28 days; cochlear neutrophils, 54 days), marrow neutrophils (54 days) and "
              "published blood neutrophils; Clopper-Pearson 95% CI; perilymph per labelled cell = divided by the GFP "
              "partner's share (Jeffreys ratio interval)", required=["tissue", "population", "pct", "ci_hi_pct"])
    m = dens[dens.main & (dens.animal == "31 d")].set_index("compartment").cell_density_per_mm2
    log("31-day partner, cells per mm2: " + ", ".join(f"{k} {v:.1f}" for k, v in m.items()))
    log("shares: " + "; ".join(f"{r.population} {r.animal} {r.pct:.1f}%" for r in sh.itertuples()))


if __name__ == "__main__":
    main()
