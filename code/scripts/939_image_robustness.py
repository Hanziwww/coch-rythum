"""Round 10 (D137) — robustness of the image quantification behind Fig. 4 and Supplementary Figs 2-6 (user: "Please
ensure that the current quantification and labeling algorithms are sufficiently robust"). Every rule is re-run over a
grid of its parameters; the published setting is reproduced first (asserted), then each claim is re-evaluated at every
setting.

1  GFP+ bodies of the parabiosis survey (934): threshold k x robust s.d. above the tissue median (6, 8, 10, 12, 14),
   minimum area (10, 15, 25 um2), minimum solidity (0.6, 0.7, 0.8) -> pooled cochlea / marrow density per wild-type
   pair (claim: < 1 in every pair).
2  Immune cell typing of the 54-day wild-type partner (938): series threshold 0.8 / 1.0 / 1.25 x Otsu, size window
   10-200 / 15-200 / 15-300 um2, GFP majority cutoff 0.3 / 0.5 / 0.7 -> partner-derived share per cell type and
   compartment, with the reporter-negative B-cell background (claims: no cochlear neutrophil GFP+; B-cell background
   low; partner-derived cochlear leukocytes present).
3  Cochlear-fluid cells (931): cell mask = nucleus expanded by 1.0 / 1.5 / 2.0 um; local background annulus 2-6 / 3-8 /
   5-10 um; positivity ratio 1.5 / 2 / 3 -> % FUT4+ and % CD45+ per compartment.
4  Perilymph cells (934): cytoplasmic ring 1.5 / 2 / 3 um; ratio 1.5 / 2 / 3 x local background -> GFP+ share of the
   wild-type and GFP partners (claim: wild type < GFP partner in every pair).
Outputs (lib/io.write_csv, results/r10_parabiosis/): robustness_gfp_bodies.csv, robustness_celltypes.csv,
robustness_fluids.csv, robustness_perilymph.csv and robustness_claims.csv (per claim: settings tested, settings at
which it holds, range of the estimate).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/939_image_robustness.py   (local; raw images in data/photos)
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S934 = load_module("s934", "934_parabiosis_survey.py")
S931 = load_module("s931", "931_image_quant.py")
S938 = load_module("s938", "938_parabiosis_cell_types.py")
MOD = "r10_parabiosis"
R10 = RESULTS / MOD
R9 = RESULTS / "r9_figures"
K, AMIN, SMIN = (6, 8, 10, 12, 14), (10.0, 15.0, 25.0), (0.6, 0.7, 0.8)
BASE_GFP = (10, 15.0, 0.7)
SCALES, WINDOWS, CUTS = (0.8, 1.0, 1.25), ((10.0, 200.0), (15.0, 200.0), (15.0, 300.0)), (0.3, 0.5, 0.7)
BASE_TYPE = (1.0, (15.0, 200.0), 0.5)
EXPAND, ANNULI, RATIOS = (1.0, 1.5, 2.0), ((2.0, 6.0), (3.0, 8.0), (5.0, 10.0)), (1.5, 2.0, 3.0)
BASE_FLUID = (1.5, (3.0, 8.0), 2.0)
RINGS, RING_RATIOS = (1.5, 2.0, 3.0), (1.5, 2.0, 3.0)
BASE_RING = (2.0, 2.0)
PAIRS = ("14 d", "31 d", "54 d", "31 d noise")


# ------------------------------------------------------------------------------------------------ 1 GFP+ bodies
def gfp_sweep():
    tf = pd.read_csv(R10 / "parabiosis_tissue_fields.csv")
    tf = tf[(tf.partner == "WT") & tf.compartment.isin(["marrow", "cochlea"]) & tf.animal.isin(PAIRS)
            & (tf.source == "CZI")]
    rows = []
    for _, r in tf.iterrows():
        a, meta = S934.read_czi(S934.SRC / r.path)
        um = meta["um_x"]
        gi, _, _ = S934.channel_roles(meta["channels"])
        for z, tis in S934.sampled_planes(a, meta):
            gs = filters.gaussian(a[gi, z].astype(np.float32), sigma=0.5 / um, preserve_range=True)
            med, sd = S934.robust(gs[tis])
            mm2 = float(tis.sum()) * um * um / 1e6
            for k in K:
                lab = measure.label(ndi.binary_fill_holes(gs > med + k * sd) & tis)
                props = measure.regionprops(lab)
                ar = np.array([p.area * um * um for p in props])
                so = np.array([p.solidity for p in props])
                for amin, smin in itertools.product(AMIN, SMIN):
                    rows.append(dict(path=r.path, animal=r.animal, compartment=r.compartment, plane=z, k=k,
                                     area_min=amin, solidity_min=smin,
                                     bodies=int(((ar >= amin) & (so >= smin)).sum()) if len(ar) else 0, tissue_mm2=mm2))
        del a
    d = pd.DataFrame(rows)
    base = d[(d.k == BASE_GFP[0]) & (d.area_min == BASE_GFP[1]) & (d.solidity_min == BASE_GFP[2])]
    chk = base.groupby("path").bodies.sum().reindex(tf.path)
    assert (chk.values == tf.gfp_bodies.values).all(), "baseline does not reproduce 934"
    log(f"1 GFP bodies: baseline reproduces 934 in {len(tf)} fields")
    g = d.groupby(["k", "area_min", "solidity_min", "animal", "compartment"]).agg(bodies=("bodies", "sum"),
                                                                               tissue_mm2=("tissue_mm2", "sum"))
    g["density_per_mm2"] = g.bodies / g.tissue_mm2
    w = g.density_per_mm2.unstack("compartment").reset_index()
    w["cochlea_to_marrow"] = w.cochlea / w.marrow
    w["baseline"] = (w.k == BASE_GFP[0]) & (w.area_min == BASE_GFP[1]) & (w.solidity_min == BASE_GFP[2])
    return w.rename(columns={"cochlea": "cochlea_density", "marrow": "marrow_density"})


# ------------------------------------------------------------------------------------------------ 2 cell typing
def celltype_sweep():
    acq = pd.read_csv(R10 / "parabiosis_acquisitions.csv")
    ob = pd.read_csv(R9 / "f4_parabiont_objects.csv")
    lw = ob[(ob.field == "60x_sl2") & (ob.object == "FUT4+ cell")]
    rows = []
    for scale in SCALES:
        S938.THR_SCALE, S938.AREA = scale, (min(w[0] for w in WINDOWS), max(w[1] for w in WINDOWS))
        zc, _ = S938.zeiss(acq)
        oc, _ = S938.olympus()
        c = pd.DataFrame(zc + oc)
        c = c[c.included.astype(bool) & ~c.series.isin(list(S938.NOT_INTERPRETED))]
        for (lo, hi), cut in itertools.product(WINDOWS, CUTS):
            t = c[(c.area_um2 >= lo) & (c.area_um2 <= hi) & (c.nucleus_frac >= S938.NUC_FRAC)].copy()
            t["gfp"] = t.gfp_overlap >= cut
            t["comp"] = t.compartment.replace({"vestibular bone": "marrow", "marrow (otic-capsule island)": "marrow"})
            for (mk, cp), g in t.groupby(["marker", "comp"]):
                n, k = len(g), int(g.gfp.sum())
                if mk == "FUT4" and cp == "cochlea":           # + the 60x lateral wall of 931, no GFP contact at all
                    n += int(((lw.area_um2 >= lo) & (lw.area_um2 <= hi)).sum())
                rows.append(dict(threshold_scale=scale, area_lo=lo, area_hi=hi, majority_cutoff=cut, marker=mk,
                                 compartment=cp, typed_cells=n, gfp_pos_cells=k, gfp_pos_pct=100 * k / n if n else np.nan))
    S938.THR_SCALE, S938.AREA = 1.0, (15.0, 200.0)
    d = pd.DataFrame(rows)
    d["baseline"] = ((d.threshold_scale == BASE_TYPE[0]) & (d.area_lo == BASE_TYPE[1][0]) &
                     (d.area_hi == BASE_TYPE[1][1]) & (d.majority_cutoff == BASE_TYPE[2]))
    f4i = pd.read_csv(R9 / "f4i_celltype_share.csv").set_index("row")
    b = d[d.baseline].set_index(["marker", "compartment"])
    for row, key in (("Cochlea, neutrophils (FUT4+)", ("FUT4", "cochlea")), ("Cochlea, leukocytes (CD45+)", ("CD45", "cochlea")),
                     ("Marrow, B cells (CD19+; reporter-negative)", ("CD19", "marrow"))):
        assert (int(b.loc[key].typed_cells), int(b.loc[key].gfp_pos_cells)) == (int(f4i.loc[row].typed_cells),
                                                                               int(f4i.loc[row].gfp_pos_cells)), row
    log("2 cell typing: baseline reproduces 938 / Fig. 4i")
    return d


# ------------------------------------------------------------------------------------------------ 3 cochlear fluids
def fluid_sweep():
    rows = []
    for info in sorted(S931.WT.rglob("*_Info.txt")):
        stem = info.name[:-len("_Info.txt")]
        f = {c: info.parent / f"{stem}_C{c}.tif" for c in (1, 2, 3)}
        if not all(p.exists() for p in f.values()):
            continue
        px, _ = S931.sidecar(info)
        dapi, fut, cd = (tifffile.imread(f[c]).astype(float) for c in (1, 2, 3))
        lab = S931.segment_nuclei(dapi, px)
        h, w = lab.shape
        for ex, (bi, bo) in itertools.product(EXPAND, ANNULI):
            cell = expand_labels(lab, int(round(ex / px)))
            zone = expand_labels(lab, int(round(bi / px))) > 0
            far = expand_labels(lab, int(round(bo / px)))
            for r in measure.regionprops(cell):
                y0, x0, y1, x1 = r.bbox
                if y0 == 0 or x0 == 0 or y1 == h or x1 == w:
                    continue
                bg = (far == r.label) & ~zone
                if bg.sum() < 50:
                    continue
                m = cell == r.label
                rows.append(dict(compartment=S931.LABEL[info.parent.name], field=int(stem.rsplit("_", 1)[1]),
                                 cell=r.label, expand_um=ex, bg_inner_um=bi, bg_outer_um=bo,
                                 fut4_ratio=float(fut[m].mean()) / max(float(np.median(fut[bg])), 1.0),
                                 cd45_ratio=float(cd[m].mean()) / max(float(np.median(cd[bg])), 1.0)))
    c = pd.DataFrame(rows)
    out = []
    for (ex, bi, bo, comp), g in c.groupby(["expand_um", "bg_inner_um", "bg_outer_um", "compartment"]):
        for rr in RATIOS:
            out.append(dict(expand_um=ex, bg_inner_um=bi, bg_outer_um=bo, ratio=rr, compartment=comp, cells=len(g),
                            fut4_pos=int((g.fut4_ratio >= rr).sum()), cd45_pos=int((g.cd45_ratio >= rr).sum()),
                            fut4_pos_pct=100 * (g.fut4_ratio >= rr).mean(), cd45_pos_pct=100 * (g.cd45_ratio >= rr).mean()))
    d = pd.DataFrame(out)
    d["baseline"] = ((d.expand_um == BASE_FLUID[0]) & (d.bg_inner_um == BASE_FLUID[1][0]) &
                     (d.bg_outer_um == BASE_FLUID[1][1]) & (d.ratio == BASE_FLUID[2]))
    pub = pd.read_csv(R9 / "f5a_fut4_cells.csv").groupby("compartment").agg(cells=("cell", "size"),
                                                                           fut4_pos=("fut4_positive", "sum"),
                                                                           cd45_pos=("cd45_positive", "sum"))
    b = d[d.baseline].set_index("compartment")[["cells", "fut4_pos", "cd45_pos"]]
    assert (b.sort_index().values == pub.sort_index().values).all(), (b, pub)
    log("3 fluids: baseline reproduces 931")
    return d


# ------------------------------------------------------------------------------------------------ 4 perilymph rings
def ring_cells(a, meta, rel, ring_um):
    """934 analyse_free with the ring width as a parameter (nuclei, background zone and rules otherwise unchanged)."""
    um = meta["um_x"]
    gi, di, _ = S934.channel_roles(meta["channels"])
    d = a[di].max(axis=0).astype(np.float32)
    g = a[gi].max(axis=0).astype(np.float32)
    ds = filters.gaussian(d, sigma=0.5 / um, preserve_range=True)
    nuc = ndi.binary_fill_holes(ds > filters.threshold_otsu(ds))
    dist = ndi.distance_transform_edt(nuc)
    pk = peak_local_max(dist, min_distance=max(1, int(round(2.5 / um))), labels=measure.label(nuc), exclude_border=False)
    mk = np.zeros(nuc.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-dist, mk, mask=nuc)
    lab = morphology.remove_small_objects(lab, min_size=max(1, int(round(10 / (um * um)))))
    lab = segmentation.clear_border(lab)
    allnuc = lab > 0
    zone3 = morphology.binary_dilation(allnuc, morphology.disk(max(1, int(round(3 / um)))))
    rows = []
    for r in measure.regionprops(lab):
        own = lab == r.label
        y0, x0, y1, x1 = r.bbox
        pad = int(round(9 / um))
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        o = own[sl]
        ring = morphology.binary_dilation(o, morphology.disk(max(1, int(round(ring_um / um))))) & ~allnuc[sl]
        dd = ndi.distance_transform_edt(~o) * um
        bg_zone = (dd >= 3) & (dd <= 8) & ~zone3[sl]
        if ring.sum() < 5 or bg_zone.sum() < 50:
            continue
        gg = g[sl]
        rows.append(dict(path=rel, ring_gfp=float(gg[ring].mean()), background=float(np.median(gg[bg_zone]))))
    return rows


def perilymph_sweep():
    pf = pd.read_csv(R10 / "parabiosis_perilymph_fields.csv")
    pf = pf[pf.animal.isin(["14 d", "28 d"])]
    rows = []
    for _, r in pf.iterrows():
        a, meta = S934.read_czi(S934.SRC / r.path)
        for ring in RINGS:
            for c in ring_cells(a, meta, r.path, ring):
                rows.append(dict(animal=r.animal, partner=r.partner_label, ring_um=ring, **c))
        del a
    c = pd.DataFrame(rows)
    out = []
    for (ring, an, prt), g in c.groupby(["ring_um", "animal", "partner"]):
        for rr in RING_RATIOS:
            pos = (g.ring_gfp >= rr * np.maximum(g.background, 2.0)) & (g.ring_gfp - g.background >= 5)
            out.append(dict(ring_um=ring, ratio=rr, animal=an, partner=prt, nuclei=len(g), gfp_pos=int(pos.sum()),
                            gfp_pos_pct=100 * pos.mean()))
    d = pd.DataFrame(out)
    d["baseline"] = (d.ring_um == BASE_RING[0]) & (d.ratio == BASE_RING[1])
    pub = pf.groupby(["animal", "partner_label"]).agg(nuclei=("nuclei", "sum"), gfp_pos=("gfp_pos", "sum"))
    b = d[d.baseline].set_index(["animal", "partner"])[["nuclei", "gfp_pos"]]
    assert (b.sort_index().values == pub.sort_index().values).all(), (b, pub)
    log("4 perilymph: baseline reproduces 934")
    return d


# ------------------------------------------------------------------------------------------------ claims
def claims(gb, ct, fl, pe):
    rows = []
    prim = gb[gb.animal.isin(["14 d", "31 d", "54 d"])]
    per = prim.groupby(["k", "area_min", "solidity_min"]).cochlea_to_marrow.max()
    rows.append(dict(claim="cochlea / marrow partner-derived density < 1 in every unexposed pair", rule="GFP+ bodies",
                     settings=len(per), holds=int((per < 1).sum()), estimate_min=prim.cochlea_to_marrow.min(),
                     estimate_max=prim.cochlea_to_marrow.max(), unit="ratio (pairs pooled over settings)"))
    allp = gb.groupby(["k", "area_min", "solidity_min"]).cochlea_to_marrow.max()
    rows.append(dict(claim="cochlea / marrow < 1 in every pair including the noise-exposed pair", rule="GFP+ bodies",
                     settings=len(allp), holds=int((allp < 1).sum()), estimate_min=gb.cochlea_to_marrow.min(),
                     estimate_max=gb.cochlea_to_marrow.max(), unit="ratio"))
    for mk, cp, claim, test in (("FUT4", "cochlea", "no cochlear neutrophil partner-derived", lambda g: g.gfp_pos_cells == 0),
                                ("CD45", "cochlea", "partner-derived leukocytes present in the cochlea",
                                 lambda g: g.gfp_pos_cells >= 1),
                                ("CD19", "marrow", "reporter-negative B-cell background below 5%",
                                 lambda g: g.gfp_pos_pct < 5),
                                ("FUT4", "marrow", "marrow neutrophils largely host-derived (< 10% GFP+)",
                                 lambda g: g.gfp_pos_pct < 10)):
        g = ct[(ct.marker == mk) & (ct.compartment == cp)]
        rows.append(dict(claim=claim, rule="cell typing", settings=len(g), holds=int(test(g).sum()),
                         estimate_min=g.gfp_pos_pct.min(), estimate_max=g.gfp_pos_pct.max(), unit="% GFP+ of typed"))
    for comp in ("perilymph", "endolymph"):
        g = fl[fl.compartment == comp]
        rows.append(dict(claim=f"most {comp} nucleated cells FUT4+ (>= 75%)", rule="fluid cells", settings=len(g),
                         holds=int((g.fut4_pos_pct >= 75).sum()), estimate_min=g.fut4_pos_pct.min(),
                         estimate_max=g.fut4_pos_pct.max(), unit="% FUT4+"))
    w = pe.pivot_table(index=["ring_um", "ratio", "animal"], columns="partner", values="gfp_pos_pct").reset_index()
    w["wt_over_gfp"] = w.WT / w.GFP
    ok = w.groupby(["ring_um", "ratio"]).apply(lambda g: bool((g.WT < g.GFP).all()))
    rows.append(dict(claim="wild-type perilymph GFP+ share below the GFP partner's in every pair", rule="perilymph ring",
                     settings=len(ok), holds=int(ok.sum()), estimate_min=w.wt_over_gfp.min(),
                     estimate_max=w.wt_over_gfp.max(), unit="wild-type / GFP-partner share"))
    return pd.DataFrame(rows)


def main():
    gb = gfp_sweep()
    ct = celltype_sweep()
    fl = fluid_sweep()
    pe = perilymph_sweep()
    cl = claims(gb, ct, fl, pe)
    for r in cl.itertuples():
        log(f"{r.claim}: holds at {r.holds}/{r.settings} settings ({r.estimate_min:.3g}-{r.estimate_max:.3g} {r.unit})")
    write_csv(gb, MOD, "robustness_gfp_bodies.csv", "Robustness (939, D137): pooled partner-derived (GFP+) densities of "
              "marrow and cochlea per wild-type pair and their ratio at every GFP-body setting (threshold k x robust s.d., "
              "minimum area, minimum solidity); baseline = the published 934 rule",
              required=["k", "area_min", "solidity_min", "animal", "cochlea_to_marrow"])
    write_csv(ct, MOD, "robustness_celltypes.csv", "Robustness (939, D137): partner-derived share of typed immune cells "
              "per cell type and compartment at every typing setting (series-threshold scale, size window, GFP majority "
              "cutoff); baseline = the published 938 rule", required=["threshold_scale", "marker", "compartment",
                                                                      "typed_cells", "gfp_pos_cells"])
    write_csv(fl, MOD, "robustness_fluids.csv", "Robustness (939, D137): FUT4+ and CD45+ shares of the cochlear-fluid "
              "and marrow cells at every cell-mask, background-annulus and ratio setting; baseline = the published 931 rule",
              required=["expand_um", "ratio", "compartment", "fut4_pos_pct"])
    write_csv(pe, MOD, "robustness_perilymph.csv", "Robustness (939, D137): GFP+ share of perilymph nucleated cells of "
              "the wild-type and GFP partners at every ring-width and ratio setting; baseline = the published 934 rule",
              required=["ring_um", "ratio", "animal", "partner", "gfp_pos_pct"])
    write_csv(cl, MOD, "robustness_claims.csv", "Robustness (939, D137): each image-based claim with the number of "
              "settings tested, the number at which it holds and the range of its estimate",
              required=["claim", "settings", "holds"])


if __name__ == "__main__":
    main()
