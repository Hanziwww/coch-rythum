"""Round 10 (D136) — immune cell types of the 54-day wild-type parabiont: which typed cells of marrow and cochlea are
partner-derived (GFP+), for every immunostained cell type, beyond the FUT4+ neutrophils typed by 934 (user: "In
addition to neutrophils, we have also captured images of other cell types. Please include these in your comprehensive
analysis").

Sources (one wild-type partner, 54 days of pairing; Lyz2-EGFP marks the partner's myeloid cells, so partner-derived
lymphocytes are GFP-negative and CD19+ / CD3e+ cells give the background rate of GFP calls):
  ZEISS LSM 900 CZIs of the 54-day series, included and assigned to compartments by 934 (parabiosis_acquisitions.csv):
    FUT4 (Alexa Fluor 647) with CD14 (568) in marrow, lateral wall and cochlear duct; CD45 (647) in temporal-bone and
    vestibular-bone marrow; CD19 (647) in marrow. Each series is typed with its own threshold.
  Olympus FV3000 20x TIFF sets of the same animal (931 loaders; C1 DAPI, C2 GFP, C3 CD14-568, C4 marker-647):
    FUT4 / CD14 fields 1-3 (marrow) and 4 (cochlear turn; marrow corner excluded as in 931); CD45 fields 1 (marrow),
    2 (marrow islands and cochlear soft tissue, split as in 931) and 3 (cochlear soft tissue); CD3e fields 1 (marrow) and
    'mojiegou' (membranous labyrinth).
  Day-14 CD19 thick sections of the stria vascularis (ZEISS; 934 inclusion): tested for separability only (no marrow
    field in that series to set a threshold).
Rules (fixed before the counts were inspected; those of the FUT4+ cells of 934 / 931):
  separability: a marker whose signal (sigma 0.5 um) correlates with GFP inside tissue at Pearson r >= 0.8 carries GFP
    or autofluorescence and is not typed (the 934 rule for GFP channels).
  threshold: Otsu of the marker (sigma 0.3 um) inside the marrow tissue of the same instrument, series and marker
    (analysed planes pooled), applied to every field of that series.
  typed cell: a marker+ object (holes filled) of 15-200 um2 with at least 10% of its pixels in a DAPI nucleus (one
    nucleated cell). GFP+ bodies: GFP smoothed at 0.5 um above the tissue median + 10 robust s.d., >= 15 um2,
    solidity >= 0.7. A typed cell is GFP+ (partner-derived) when at least half of its pixels lie inside a GFP+ body
    ('majority'); any contact ('contact', the rule of the 934 FUT4 typing) is reported alongside. The majority rule
    was adopted after the reporter-negative control: CD19+ B cells do not express Lyz2-EGFP, yet in dense marrow 14 of 63
    touched a GFP+ body, against 1 of 63 by the majority rule. Marrow islands of the otic capsule in a cochlear field
    count as marrow.
  stains not interpreted: CD3e (it labelled the sensory epithelium of the membranous labyrinth, as shown in the
    Supplementary Figure, so T cells are not quantified) and the day-14 CD19 sections (no marrow field for a
    threshold; one field not separable).
  coverage: typed cells per DAPI nucleus of the field, to judge whether a stain marks a plausible share of marrow cells
    (a lymphocyte marker on most marrow nuclei is not interpreted).
The ZEISS FUT4 typing reproduces 934 exactly (asserted). Counts are descriptive (one animal; cells are not independent
units); shares carry Clopper-Pearson 95% intervals of the binomial proportion for orientation only.
Outputs (lib/io.write_csv, results/r10_parabiosis/): parabiosis_celltype_cells.csv (every marker+ object with its
coordinates, size, nucleus share, typed flag and GFP status), parabiosis_celltype_fields.csv (per acquisition and marker:
compartment, threshold, separability r, tissue area, nuclei, typed cells, GFP+ cells, coverage, inclusion and reason)
and parabiosis_celltype_summary.csv (marker x compartment x instrument and pooled over instruments).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/938_parabiosis_cell_types.py   (local; raw images in data/photos)
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy import stats
from skimage import filters, measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402

_spec = importlib.util.spec_from_file_location("s934", str(Path(__file__).resolve().parent / "934_parabiosis_survey.py"))
S934 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S934)

MOD = "r10_parabiosis"
R10 = RESULTS / MOD
PB = ROOT / "data" / "photos" / "parabiosis_54d" / "cell-type-bone-cochlea"
SEP_R = 0.8
AREA = (15.0, 200.0)
THR_SCALE = 1.0                                          # 939 robustness sweeps scale the series thresholds
NUC_FRAC = 0.1
LYMPHOID = {"CD19", "CD3e"}
MAJORITY = 0.5
NOT_INTERPRETED = {"Olympus CD3e": "CD3e stain labels the sensory epithelium of the membranous labyrinth "
                                   "(not interpreted)"}
CELL_TYPE = {"FUT4": "neutrophils", "CD14": "monocytes and macrophages", "CD45": "leukocytes", "CD19": "B cells",
             "CD3e": "T cells"}

# ZEISS series of the 54-day animal: (series, path pattern, {marker: dye})
ZSERIES = [("ZEISS FUT4/CD14", r"647fut4-568cd14-gfp-54d-20x", {"FUT4": "647", "CD14": "568"}),
           ("ZEISS CD45", r"647CD45-dapi", {"CD45": "647"}),
           ("ZEISS CD19 20x", r"647CD19-dapi-20x", {"CD19": "647"}),
           ("ZEISS CD19 63x", r"647cd19-gfp-54d-5x", {"CD19": "647"}),
           ("ZEISS CD19 day 14", r"day14_immune_cell_thick_sections/.*cd19-63x-sv", {"CD19": "647"})]
# Olympus 20x sets: (series, folder, [(field id, stem, compartment, region, excluded box)], {marker: channel})
F20 = "647fut4-568cd14-gfp-parabiotic54d-20x"
FCD = "647cd45-gfp-parabiotic54d-20x"
FT = "T-647cd3e-gfp-dapi-parabiotic54d-20x"
OSERIES = [("Olympus FUT4/CD14", F20,
            [(f"20x_fut4_{k}", f"{F20}-cochleabone_{k}", "marrow", "temporal-bone marrow", None) for k in (1, 2, 3)]
            + [("20x_fut4_4", f"{F20}-cochleabone_4", "cochlea", "cochlear turn", (0, 120, 820, 1024))],
            {"FUT4": 4, "CD14": 3}),
           ("Olympus CD45", FCD,
            [("20x_cd45_1", f"{FCD}-cochlea-bone_1", "marrow", "temporal-bone marrow", None),
             ("20x_cd45_2", f"{FCD}-cochlea-bone_2", "split", "marrow islands and cochlear soft tissue", None),
             ("20x_cd45_3", f"{FCD}-cochlea-bone_3", "cochlea", "cochlear soft tissue", None)],
            {"CD45": 4}),
           ("Olympus CD3e", FT,
            [("20x_cd3e_1", f"{FT}_1", "marrow", "temporal-bone marrow", None),
             ("20x_cd3e_ml", f"{FT}-mojiegou_1", "cochlea", "membranous labyrinth", None)],
            {"CD3e": 4})]


# ------------------------------------------------------------------------------------------------ shared rules
def smooth(img, um, sigma):
    return filters.gaussian(img.astype(np.float32), sigma=sigma / um, preserve_range=True)


def separability(marker, gfp, tissue, um):
    m = smooth(marker, um, 0.5)[tissue][::5]
    g = smooth(gfp, um, 0.5)[tissue][::5]
    if m.std() == 0 or g.std() == 0:
        return np.nan
    return float(np.corrcoef(m, g)[0, 1])


def nuclei(dapi, tissue, um):
    ds = smooth(dapi, um, 0.5)
    nuc = (ds > filters.threshold_otsu(ds[tissue])) & tissue
    lab = measure.label(nuc)
    n = sum(1 for r in measure.regionprops(lab) if r.area * um * um >= 10.0)
    return nuc, n


def type_cells(marker, nuc, body, tissue, um, thr, island=None, gfp=None):
    """Marker+ objects in tissue: area, nucleus share, typed (15-200 um2, >= 10% nucleus), GFP+ (overlaps a body), the
    share of the object inside GFP+ bodies, its GFP relative to the tissue median, and whether it lies inside a marrow
    island of the otic capsule (more than half of its pixels; as 934)."""
    fs = smooth(marker, um, 0.3)
    gs = smooth(gfp, um, 0.5) if gfp is not None else None
    gmed = float(np.median(gs[tissue])) if gs is not None and tissue.any() else np.nan
    lab = measure.label(ndi.binary_fill_holes(fs > thr) & tissue)
    rows = []
    for r in measure.regionprops(lab, intensity_image=fs):
        area = r.area * um * um
        if area < AREA[0]:
            continue
        idx = tuple(r.coords.T)
        nf = float(nuc[idx].mean())
        rows.append(dict(y=float(r.centroid[0]), x=float(r.centroid[1]), area_um2=area, nucleus_frac=nf,
                         typed=bool(area <= AREA[1] and nf >= NUC_FRAC), marker_mean=float(r.mean_intensity),
                         gfp_pos=bool(body[idx].any()), gfp_overlap=float(body[idx].mean()),
                         gfp_rel=float(gs[idx].mean() / max(gmed, 1.0)) if gs is not None else np.nan,
                         in_island=bool(island is not None and island[idx].mean() > 0.5)))
    return rows


def clopper(k, n):
    if n == 0:
        return np.nan, np.nan
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ------------------------------------------------------------------------------------------------ ZEISS series
def zeiss(acq):
    cells, fields = [], []
    for series, pat, markers in ZSERIES:
        sel = acq[acq.path.str.contains(pat, regex=True) & acq.reason.str.startswith("included")]
        if not len(sel):
            log(f"{series}: no included acquisition")
            continue
        # thresholds from the series' marrow tissue (analysed planes pooled)
        thr = {}
        for mk, dye in markers.items():
            vals = []
            for p in sel[sel.compartment == "marrow"].path:
                a, meta = S934.read_czi(S934.SRC / p)
                ci = next(i for i, n in enumerate(meta["channels"]) if dye in n)
                for z, tis in S934.sampled_planes(a, meta):
                    vals.append(smooth(a[ci, z], meta["um_x"], 0.3)[tis][::4])
                del a
            thr[mk] = THR_SCALE * float(filters.threshold_otsu(np.concatenate(vals))) if vals else np.nan
        for _, r in sel.iterrows():
            a, meta = S934.read_czi(S934.SRC / r.path)
            um = meta["um_x"]
            gi, di, _ = S934.channel_roles(meta["channels"])
            yy, xx = np.mgrid[:a.shape[2], :a.shape[3]]
            island = np.zeros(a.shape[2:], bool)
            for cx, cy, rad in S934.ISLANDS.get(r.path, []):
                island |= (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
            for mk, dye in markers.items():
                ci = next(i for i, n in enumerate(meta["channels"]) if dye in n)
                seps, n_nuc, tissue_mm2, objs = [], 0, 0.0, []
                for z, tis in S934.sampled_planes(a, meta):
                    _, _, _, body = S934.gfp_bodies(a[gi, z], None, tis, um)
                    nuc, nn = nuclei(a[di, z], tis, um)
                    n_nuc += nn
                    tissue_mm2 += float(tis.sum()) * um * um / 1e6
                    seps.append(separability(a[ci, z], a[gi, z], tis, um))
                    if np.isfinite(thr[mk]):
                        for o in type_cells(a[ci, z], nuc, body, tis, um, thr[mk], island, a[gi, z]):
                            comp = "marrow (otic-capsule island)" if o["in_island"] else r.compartment
                            objs.append(dict(source="ZEISS", series=series, field=r.path, plane=z, marker=mk,
                                             compartment=comp, subregion=r.subregion if isinstance(r.subregion, str)
                                             else "", um_per_px=um, **o))
                sep = float(np.nanmax(seps)) if seps else np.nan
                ok = np.isfinite(thr[mk]) and not (sep >= SEP_R)
                reason = "included" if ok else ("marker not separable from GFP / autofluorescence (r >= 0.8)"
                                                if sep >= SEP_R else "no marrow field in the series for a threshold")
                for o in objs:
                    o["included"] = ok
                cells += objs
                d = pd.DataFrame(objs)
                typed = d[d.typed] if len(d) else d
                fields.append(dict(source="ZEISS", series=series, field=r.path, marker=mk, compartment=r.compartment,
                                   subregion=r.subregion if isinstance(r.subregion, str) else "", um_per_px=um,
                                   planes=len(seps), threshold=thr[mk], separability_r=sep, tissue_mm2=tissue_mm2,
                                   nuclei=n_nuc, marker_objects=len(d), typed_cells=len(typed),
                                   typed_gfp_pos=int(typed.gfp_pos.sum()) if len(typed) else 0,
                                   included=ok, reason=reason))
                log(f"{series} {mk} {r.path.rsplit('/', 1)[-1]} ({r.compartment}): r {sep:.2f}, typed {len(typed)}, "
                    f"GFP+ {int(typed.gfp_pos.sum()) if len(typed) else 0}, nuclei {n_nuc} -> {reason}")
            del a
    return cells, fields


# ------------------------------------------------------------------------------------------------ Olympus sets
def sidecar(p: Path):
    s = p.read_text(errors="replace")
    return float(re.search(r"Pixel size: ([\d.]+)um", s).group(1))


def load_olympus(folder: str, stem: str, chans):
    """Channels of one Olympus field (C1 DAPI, C2 GFP, marker channels): the sharpest DAPI section of a z-series."""
    d = PB / folder
    by = {c: (sorted(d.glob(f"{stem}_C{c}.tif")) or sorted(d.glob(f"{stem}_C{c}_Z*.tif"))) for c in [1, 2] + chans}
    z = by[1]
    k = 0 if len(z) == 1 else int(np.argmax([ndi.laplace(filters.gaussian(tifffile.imread(p).astype(float), 1)).var()
                                             for p in z]))
    out = {}
    for c, v in by.items():
        img = tifffile.imread(v[k if len(v) > 1 else 0]).astype(float)
        out[c] = img.max(0) if img.ndim == 3 else img
    return out, sidecar(d / f"{stem}_Info.txt")


def o_tissue(dapi, um, box):
    nuc = smooth(dapi, um, 0.5)
    t = ndi.binary_closing(smooth(dapi, um, 6.0) > filters.threshold_otsu(nuc) * 0.25, iterations=3)
    if box:
        t[box[2]:box[3], box[0]:box[1]] = False
    return t


def o_islands(marker, tissue, um):
    """931: marrow islands of a CD45 image (CD45 smoothed at 10 um above its Otsu threshold in tissue, dilated 5 um)."""
    sm = smooth(marker, um, 10.0)
    m = (sm > filters.threshold_otsu(sm[tissue])) & tissue
    return ndi.binary_dilation(m, iterations=int(round(5.0 / um))) & tissue


def o_bodies(gfp, tissue, um):
    gs = smooth(gfp, um, 0.5)
    tv = gs[tissue]
    med = float(np.median(tv))
    thr = med + 10 * 1.4826 * float(np.median(np.abs(tv - med)))
    lab = measure.label(ndi.binary_fill_holes(gs > thr) & tissue)
    keep = np.zeros(lab.shape, bool)
    for r in measure.regionprops(lab):
        if r.area * um * um >= 15.0 and r.solidity >= 0.7:
            keep[tuple(r.coords.T)] = True
    return keep


def olympus():
    cells, fields = [], []
    for series, folder, flist, markers in OSERIES:
        loaded = {}
        for fid, stem, comp, reg, box in flist:
            ch, um = load_olympus(folder, stem, sorted(set(markers.values())))
            t = o_tissue(ch[1], um, box)
            if comp == "split":
                mi = o_islands(ch[markers["CD45"]], t, um)
                parts = [("marrow", "temporal-bone marrow islands", mi), ("cochlea", "cochlear soft tissue", t & ~mi)]
            else:
                parts = [(comp, reg, t)]
            loaded[fid] = (ch, um, parts)
        thr = {}
        for mk, c in markers.items():
            vals = [smooth(ch[c], um, 0.3)[m][::4] for ch, um, parts in loaded.values()
                    for cp, _, m in parts if cp == "marrow" and m.any()]
            thr[mk] = THR_SCALE * float(filters.threshold_otsu(np.concatenate(vals))) if vals else np.nan
        for fid, (ch, um, parts) in loaded.items():
            for cp, reg, m in parts:
                body = o_bodies(ch[2], m, um)
                nuc, n_nuc = nuclei(ch[1], m, um)
                for mk, c in markers.items():
                    sep = separability(ch[c], ch[2], m, um)
                    ok = np.isfinite(thr[mk]) and not (sep >= SEP_R)
                    objs = ([dict(source="Olympus", series=series, field=fid, plane=0, marker=mk, compartment=cp,
                                  subregion=reg, um_per_px=um, included=ok, **o)
                             for o in type_cells(ch[c], nuc, body, m, um, thr[mk], gfp=ch[2])]
                            if np.isfinite(thr[mk]) else [])
                    cells += objs
                    d = pd.DataFrame(objs)
                    typed = d[d.typed] if len(d) else d
                    reason = "included" if ok else "marker not separable from GFP / autofluorescence (r >= 0.8)"
                    fields.append(dict(source="Olympus", series=series, field=fid, marker=mk, compartment=cp,
                                       subregion=reg, um_per_px=um, planes=1, threshold=thr[mk], separability_r=sep,
                                       tissue_mm2=float(m.sum()) * um * um / 1e6, nuclei=n_nuc,
                                       marker_objects=len(d), typed_cells=len(typed),
                                       typed_gfp_pos=int(typed.gfp_pos.sum()) if len(typed) else 0,
                                       included=ok, reason=reason))
                    log(f"{series} {mk} {fid} ({cp}): r {sep:.2f}, typed {len(typed)}, "
                        f"GFP+ {int(typed.gfp_pos.sum()) if len(typed) else 0}, nuclei {n_nuc} -> {reason}")
    return cells, fields


# ------------------------------------------------------------------------------------------------ main
def main():
    acq = pd.read_csv(R10 / "parabiosis_acquisitions.csv")
    zc, zf = zeiss(acq)
    oc, of = olympus()
    cells = pd.DataFrame(zc + oc)
    fields = pd.DataFrame(zf + of)
    fields["coverage"] = fields.typed_cells / fields.nuclei.replace(0, np.nan)
    # a lymphocyte stain that marks most nuclei of marrow is not interpreted (T and B cells are a minority of marrow)
    implaus = (fields.marker.isin(LYMPHOID) & (fields.compartment == "marrow") & (fields.coverage > 0.5)
               & fields.included)
    for s_ in fields[implaus].series.unique():
        sel = (fields.series == s_) & fields.marker.isin(LYMPHOID)
        fields.loc[sel, "included"] = False
        fields.loc[sel, "reason"] = "stain marks most marrow nuclei (implausible for lymphocytes; not interpreted)"
        cells.loc[(cells.series == s_) & cells.marker.isin(LYMPHOID), "included"] = False
    for s_, why in NOT_INTERPRETED.items():
        fields.loc[fields.series == s_, "included"] = False
        fields.loc[fields.series == s_, "reason"] = why
        cells.loc[cells.series == s_, "included"] = False
    cells["gfp_contact"] = cells.gfp_pos.astype(bool)
    cells["gfp_majority"] = cells.gfp_overlap >= MAJORITY
    # the ZEISS FUT4 typing reproduces 934 (contact rule)
    fut = pd.read_csv(R10 / "parabiosis_fut4_cells.csv")
    mine = cells[(cells.series == "ZEISS FUT4/CD14") & (cells.marker == "FUT4") & cells.typed]
    for comp in ("marrow", "cochlea", "marrow (otic-capsule island)"):
        a_ = fut[fut.typed & (fut.compartment == comp)]
        b_ = mine[mine.compartment == comp]
        assert (len(a_), int(a_.gfp_pos.sum())) == (len(b_), int(b_.gfp_pos.sum())), (comp, len(a_), len(b_))
    log("ZEISS FUT4 typing equals 934")

    typed = cells[cells.typed & cells.included].copy()
    # 931: the 13 FUT4+ cells of the 60x lateral-wall sections (Olympus), all without GFP contact
    ob = pd.read_csv(RESULTS / "r9_figures" / "f4_parabiont_objects.csv")
    lw = ob[(ob.field == "60x_sl2") & (ob.object == "FUT4+ cell")]
    wall = pd.read_csv(RESULTS / "r9_figures" / "f4_parabiont_fields.csv")
    wall = wall[(wall.field == "60x_sl2") & (wall.compartment == "cochlea")].iloc[0]
    assert len(lw) == int(wall.marker_cells) and not lw.gfp_positive.astype(bool).any()
    typed = pd.concat([typed, pd.DataFrame(dict(source="Olympus 60x (931)", series="Olympus FUT4 60x",
                                                field="60x_sl2", marker="FUT4", compartment="cochlea",
                                                subregion="lateral wall", typed=True, included=True,
                                                gfp_contact=False, gfp_majority=False, area_um2=lw.area_um2.values))],
                      ignore_index=True)
    fields = pd.concat([fields, pd.DataFrame([dict(source="Olympus 60x (931)", series="Olympus FUT4 60x",
                                                    field="60x_sl2", marker="FUT4", compartment="cochlea",
                                                    subregion="lateral wall", tissue_mm2=float(wall.tissue_mm2),
                                                    typed_cells=len(lw), typed_gfp_pos=0, included=True,
                                                    reason="931 count (FUT4+ objects of the 60x lateral wall)")])],
                       ignore_index=True)
    rows = []
    groups = [(k, g) for k, g in typed.groupby(["marker", "compartment", "source"])]
    groups += [((mk, cp, "both instruments"), g) for (mk, cp), g in typed.groupby(["marker", "compartment"])]
    for (mk, cp, src), g in groups:
        f = fields[(fields.marker == mk) & (fields.compartment == cp) & fields.included]
        if src != "both instruments":
            f = f[f.source == src]
        n, k, kc = len(g), int(g.gfp_majority.sum()), int(g.gfp_contact.sum())
        lo, hi = clopper(k, n)
        rows.append(dict(marker=mk, cell_type=CELL_TYPE[mk], compartment=cp, source=src, fields=f.field.nunique(),
                         tissue_mm2=f.tissue_mm2.sum(), typed_cells=n, gfp_pos_cells=k, gfp_pos_pct=100 * k / n,
                         gfp_contact_cells=kc, gfp_contact_pct=100 * kc / n,
                         ci_lo_pct=100 * lo, ci_hi_pct=100 * hi, typed_per_mm2=n / f.tissue_mm2.sum()
                         if f.tissue_mm2.sum() > 0 else np.nan))
    summ = pd.DataFrame(rows).sort_values(["marker", "compartment", "source"])
    for r in summ[summ.source == "both instruments"].itertuples():
        log(f"{r.marker} ({r.cell_type}), {r.compartment}: {r.gfp_pos_cells}/{r.typed_cells} GFP+ by majority, "
            f"{r.gfp_contact_cells} by contact "
            f"({r.gfp_pos_pct:.0f}%, {r.ci_lo_pct:.0f}-{r.ci_hi_pct:.0f}), {r.typed_per_mm2:.0f} per mm2")
    write_csv(cells, MOD, "parabiosis_celltype_cells.csv", "Parabiosis, 54-day wild-type partner (938, D136): every "
              "marker+ object of the immune cell-type stainings (FUT4, CD14, CD45, CD19, CD3e) with source, field, "
              "compartment, coordinates, size, nucleus share, typed flag, GFP status and inclusion",
              required=["source", "field", "marker", "compartment", "typed", "gfp_pos"])
    write_csv(fields, MOD, "parabiosis_celltype_fields.csv", "Parabiosis, 54-day wild-type partner (938, D136): per "
              "acquisition and marker - compartment, threshold, separability from GFP, tissue area, nuclei, typed "
              "cells, GFP+ typed cells, coverage, inclusion and reason",
              required=["source", "field", "marker", "compartment", "typed_cells", "included"])
    write_csv(summ, MOD, "parabiosis_celltype_summary.csv", "Parabiosis, 54-day wild-type partner (938, D136): typed "
              "cells per marker and compartment, by instrument and pooled - GFP+ (partner-derived; majority rule, "
              "contact rule alongside) share with "
              "Clopper-Pearson 95% interval (one animal; descriptive) and density per mm2 of tissue",
              required=["marker", "compartment", "source", "typed_cells", "gfp_pos_cells"])


if __name__ == "__main__":
    main()
