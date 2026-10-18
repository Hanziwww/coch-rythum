"""Round 9 (D86, D87) — quantification of the user's confocal images (data/photos; local run, the images are not on the
server).

A. Wild-type FUT4 / CD45 / DAPI preparations of bone marrow, perilymph and endolymph (Fig. 4d; Olympus FV3000, 60x, 16-bit
   single-channel TIFFs; pixel size and laser powers from each field's acquisition sidecar). Every complete field
   (9 marrow, 10 perilymph, 9 endolymph) is segmented: nuclei = Otsu threshold of the Gaussian-smoothed DAPI
   (sigma 0.5 um), holes filled, touching nuclei split by a distance-transform watershed (peaks >= 2.5 um apart),
   objects >= 15 um2; cell = nucleus expanded by 1.5 um without overlap; cells touching the image edge are dropped.
   FUT4 per cell = mean FUT4 over the cell divided by its local background (median FUT4 3-8 um from the nucleus,
   outside every cell's 3-um zone; >= 50 px required). A cell is FUT4+ when that ratio is >= 2 (pre-set; FUT4 was
   acquired with the same 488-nm power (12 %) and detector voltage (700 V) in every field except marrow field 2
   (10 %); the ratio is self-normalising). CD45 channel (C3; D93, user request to check the perilymph images): in
   perilymph (all fields) and endolymph fields 1-4 the 561-nm line was at 0 %, yet C3 carries a membrane ring on the
   cells. It is not FUT4 emission spilling into the RFP detector: relative to FUT4 (background-subtracted C3 / C2) it is
   larger than in marrow, where the 561-nm line was on, and it is rim-enriched relative to FUT4 in the same cells
   (rim / inner ratio, rim = outer 0.75 um of the cell mask; paired Wilcoxon per compartment and excitation). C3 is
   therefore read as CD45-555 excited without the 561-nm line (488-nm cross-excitation or residual 561-nm light): CD45
   presence per cell is scored with the same self-normalising ratio (cell mean / local background >= 2, as for FUT4);
   CD45 intensities are not compared between preparations (different excitation).

B. Parabiont mice, 54 days (GFP chimera; Fig. 4f,i). Design premise given by the user (D87): the bone marrow is of
   wild-type origin and peripheral cells are GFP+, so a GFP+ cell is of peripheral origin and a GFP- haematopoietic
   cell comes from the (wild-type) marrow; the premise applies to these images only. Temporal-bone sections, Olympus
   16-bit TIFFs (DAPI C1, GFP C2, marker C4: FUT4-647, CD45-647 or CD3e-647):
   - marrow: 60x FUT4 fields at distinct stage positions (1_1, 1_3, 1_5, 1_7, 1_8; zoomed re-acquisitions of a
     position are not pooled), 20x FUT4 field 1, and the marrow islands of 20x CD45 field 2; 1_11 (a zoom inside 1_8,
     acquired with the lateral wall's GFP settings, 3 % / 550 V) is the detection reference for the lateral wall;
   - cochlea: 60x lateral wall (sl-2; sl is an earlier acquisition of the same position), 20x cochlear turn (FUT4
     field 4; its marrow corner x < 120, y > 820 px, drawn on DAPI, excluded), the cochlear soft tissue of CD45 field 2
     and the 20x membranous labyrinth (CD3e set, 'mojiegou' field; GFP only).
   Z-stacks: marrow, the optical section with the sharpest DAPI (variance of the Laplacian; dense tissue); lateral
   wall, the maximum projection of all 22 sections (the most sensitive view for GFP+ cells). Tissue = smoothed DAPI
   (sigma 6 um) above a quarter of the Otsu threshold of the sigma-0.5 DAPI, closed. CD45 field 2 is split into
   marrow islands (CD45 smoothed at sigma 10 um above its Otsu threshold within tissue, dilated 5 um) and cochlear soft
   tissue (the rest). GFP+ cell body = GFP (sigma 0.5 um) above the compartment's tissue median + 10 robust SD, holes
   filled, >= 15 um2, solidity >= 0.7 (filled cell bodies, not clusters of GFP+ particles). FUT4+ cells of the lateral
   wall = FUT4 (sigma 0.3 um) above the Otsu threshold of FUT4 in the 60x marrow tissue (identical 640-nm settings),
   objects >= 15 um2; GFP+ when they overlap a GFP+ body. Cochlear CD45+ cells are not counted: their thin, dim
   processes fall below or fragment under any threshold set in the marrow, so CD45 field 2 contributes GFP+ densities
   only (its cochlear soft tissue holds 3 GFP+ bodies, one beside a CD45+ cell in a vessel-like structure). Detection
   check: the share of marrow GFP+ bodies imaged with the same GFP settings (1_11 for
   the lateral wall, field 1 for the turn, the marrow islands for CD45 field 2) whose mean GFP clears the cochlear
   compartment's GFP threshold.

Outputs: results/r9_figures/f5a_fut4_cells.csv (A, one row per cell, FUT4 and CD45 channel) and
f5a_cd45_channel_check.csv (A, the C3 cross-talk check per compartment and excitation); results/r9_figures/f4_parabiont_fields.csv
(B, one row per field and compartment) and f4_parabiont_objects.csv (B, the objects of the two Fig. 4f image fields:
GFP+ bodies of marrow field 1_11, FUT4+ cells and GFP+ bodies of the lateral wall, with centroids in image pixels;
930 picks the crop windows from them and 912 marks the GFP- FUT4+ cells, D88). Run 931 before 930:
COCH_ROOT=/path/to/coch-rythum python scripts/931_image_quant.py
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, watershed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "data" / "photos" / "FUT4_WT_2026-10-12"
PB = ROOT / "data" / "photos" / "parabiosis_54d" / "cell-type-bone-cochlea"
LABEL = {"bone marrow": "bone_marrow", "perilymphy": "perilymph", "endolymph": "endolymph"}
RATIO_POS = 2.0
F60, FSL = "parabiotic54d-gfp-647fut4-60x", "parabiotic54d-gfp-647fut4-60x-sl-2"
F20, FCD, FT = ("647fut4-568cd14-gfp-parabiotic54d-20x", "647cd45-gfp-parabiotic54d-20x",
                "T-647cd3e-gfp-dapi-parabiotic54d-20x")
S20 = "647fut4-568cd14-gfp-parabiotic54d-20x-cochleabone_{}"
PB_FIELDS = [  # (field id, folder, file stem, magnification, compartment: marrow / cochlea / split, region, marker,
    #             maximum projection?, excluded box (x0, x1, y0, y1) px, pooled in the per-field summary?)
    ("60x_1_1", F60, "1_1", "60x", "marrow", "temporal-bone marrow", "FUT4", False, None, True),
    ("60x_1_3", F60, "1_3", "60x", "marrow", "temporal-bone marrow", "FUT4", False, None, True),
    ("60x_1_5", F60, "1_5", "60x", "marrow", "temporal-bone marrow", "FUT4", False, None, True),
    ("60x_1_7", F60, "1_7", "60x", "marrow", "temporal-bone marrow", "FUT4", False, None, True),
    ("60x_1_8", F60, "1_8", "60x", "marrow", "temporal-bone marrow", "FUT4", False, None, True),
    ("60x_1_11", F60, "1_11", "60x", "marrow", "temporal-bone marrow (zoom inside 1_8)", "FUT4", False, None, False),
    ("60x_sl2", FSL, "1_1", "60x", "cochlea", "cochlear lateral wall", "FUT4", True, None, True),
    ("20x_f1", F20, S20.format(1), "20x", "marrow", "temporal-bone marrow and bone", "FUT4", False, None, True),
    ("20x_f4", F20, S20.format(4), "20x", "cochlea", "cochlear turn and otic capsule", "FUT4", False,
     (0, 120, 820, 1024), True),
    ("20x_cd45_f2", FCD, "647cd45-gfp-parabiotic54d-20x-cochlea-bone_2", "20x", "split", "CD45 field", "CD45", False,
     None, True),
    ("20x_ml", FT, "T-647cd3e-gfp-dapi-parabiotic54d-20x-mojiegou_1", "20x", "cochlea",
     "membranous labyrinth and otic capsule", "CD3e", False, None, True),
]
PB_CH = {"dapi": 1, "gfp": 2, "marker": 4}
DETECT_REF = {("60x_sl2", "cochlea"): ("60x_1_11", "marrow"), ("20x_f4", "cochlea"): ("20x_f1", "marrow"),
              ("20x_cd45_f2", "cochlea"): ("20x_cd45_f2", "marrow")}
COUNT_MARKER = {("60x_sl2", "cochlea")}                                # marker+ cells counted (FUT4)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sidecar(p: Path):
    s = p.read_text(errors="replace")
    px = float(re.search(r"Pixel size: ([\d.]+)um", s).group(1))
    las = {int(c): (float(v), int(pmt)) for c, _, v, pmt in
           re.findall(r"Channel (\d): (\d+) Laser power: ([\d.]+)%, PMTVoltage: (\d+)", s)}
    return px, las


# ------------------------------------------------------------------------------------------------ A  WT preparations
def segment_nuclei(dapi, px):
    sm = gaussian(dapi, 0.5 / px, preserve_range=True)
    fg = ndi.binary_fill_holes(sm > threshold_otsu(sm))
    dist = ndi.distance_transform_edt(fg)
    pk = peak_local_max(dist, min_distance=int(round(2.5 / px)), labels=ndi.label(fg)[0], exclude_border=False)
    mk = np.zeros(dist.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, mk, mask=fg)
    area = np.bincount(lab.ravel()) * px * px
    lab[np.isin(lab, np.where(area < 15.0)[0])] = 0
    return lab


def part_a():
    rows = []
    for info in sorted(WT.rglob("*_Info.txt")):
        stem = info.name[:-len("_Info.txt")]
        f = {c: info.parent / f"{stem}_C{c}.tif" for c in (1, 2, 3)}
        if not all(p.exists() for p in f.values()):
            continue
        px, las = sidecar(info)
        dapi, fut, cd = (tifffile.imread(f[c]).astype(float) for c in (1, 2, 3))
        lab = segment_nuclei(dapi, px)
        cell = expand_labels(lab, int(round(1.5 / px)))
        zone = expand_labels(lab, int(round(3.0 / px))) > 0
        far = expand_labels(lab, int(round(8.0 / px)))
        h, w = lab.shape
        fmd5, cmd5 = md5(f[2]), md5(f[3])
        for r in regionprops(cell):
            y0, x0, y1, x1 = r.bbox
            if y0 == 0 or x0 == 0 or y1 == h or x1 == w:
                continue
            bg = (far == r.label) & ~zone
            if bg.sum() < 50:
                continue
            m = cell == r.label
            b = float(np.median(fut[bg]))
            bc = float(np.median(cd[bg]))
            rim = m & (ndi.distance_transform_edt(m) * px <= 0.75)       # outer 0.75 um of the cell mask
            inner = m & ~rim
            d2, d3 = float(fut[m].mean()) - b, float(cd[m].mean()) - bc
            rows.append({"compartment": LABEL[info.parent.name], "field": int(stem.rsplit("_", 1)[1]), "cell": r.label,
                         "um_per_px": px, "fut4_laser_pct": las[2][0], "cd45_laser_pct": las[3][0],
                         "nucleus_um2": round(float((lab == r.label).sum() * px * px), 2),
                         "fut4_cell_mean": round(float(fut[m].mean()), 1), "fut4_local_background": round(b, 1),
                         "fut4_ratio": round(float(fut[m].mean()) / max(b, 1.0), 3),
                         "cd45_cell_mean": round(float(cd[m].mean()), 1), "cd45_local_background": round(bc, 1),
                         "cd45_ratio": round(float(cd[m].mean()) / max(bc, 1.0), 3),
                         "cd45_over_fut4": round(d3 / d2, 3) if d2 > 0 else np.nan,
                         "fut4_rim_over_inner": round((fut[rim].mean() - b) / max(fut[inner].mean() - b, 1e-9), 3),
                         "cd45_rim_over_inner": round((cd[rim].mean() - bc) / max(cd[inner].mean() - bc, 1e-9), 3),
                         "source_fut4_file": str(f[2].relative_to(ROOT)).replace("\\", "/"), "source_md5": fmd5,
                         "source_cd45_file": str(f[3].relative_to(ROOT)).replace("\\", "/"), "source_cd45_md5": cmd5})
    df = pd.DataFrame(rows)
    df["fut4_positive"] = df.fut4_ratio >= RATIO_POS
    df["cd45_excitation"] = np.where(df.cd45_laser_pct > 0, "561 nm", "561 nm at 0 % (other lines)")
    df["cd45_positive"] = df.cd45_ratio >= RATIO_POS
    order = {"bone_marrow": 0, "perilymph": 1, "endolymph": 2}
    df = df.sort_values(["compartment", "field", "cell"], key=lambda s: s.map(order) if s.name == "compartment" else s)
    for comp, d in df.groupby("compartment", sort=False):
        log(f"A {comp}: {int(d.fut4_positive.sum())} / {len(d)} cells FUT4+ (ratio >= {RATIO_POS:g}; {d.field.nunique()} "
            f"fields; median ratio {d.fut4_ratio.median():.2f})")
    write_csv(df, "r9_figures", "f5a_fut4_cells.csv",
              "Fig. 4d quantification: every segmented cell of the wild-type FUT4 / CD45 / DAPI preparations (marrow, "
              "perilymph, endolymph) with its FUT4 and CD45-channel signal relative to local background; FUT4+ / CD45+ "
              "= ratio >= 2 (931, D86, D93); CD45 in perilymph and endolymph fields 1-4 acquired with the 561-nm line at "
              "0 % (presence only)", required=["compartment", "field", "fut4_ratio", "fut4_positive", "cd45_ratio",
                                                "cd45_positive", "cd45_excitation"])
    chk = []
    for (comp, exc), d in df.groupby(["compartment", "cd45_excitation"], sort=False):
        dd = d.dropna(subset=["fut4_rim_over_inner", "cd45_rim_over_inner"])
        w = wilcoxon(dd.cd45_rim_over_inner, dd.fut4_rim_over_inner)
        chk.append({"compartment": comp, "cd45_excitation": exc, "fields": d.field.nunique(), "cells": len(d),
                    "fut4_positive": int(d.fut4_positive.sum()), "cd45_positive": int(d.cd45_positive.sum()),
                    "cd45_ratio_median": round(float(d.cd45_ratio.median()), 2),
                    "cd45_over_fut4_median": round(float(d.cd45_over_fut4.median()), 2),
                    "cd45_over_fut4_q10": round(float(d.cd45_over_fut4.quantile(0.1)), 2),
                    "cd45_over_fut4_q90": round(float(d.cd45_over_fut4.quantile(0.9)), 2),
                    "fut4_rim_over_inner_median": round(float(dd.fut4_rim_over_inner.median()), 2),
                    "cd45_rim_over_inner_median": round(float(dd.cd45_rim_over_inner.median()), 2),
                    "cells_cd45_more_rim": int((dd.cd45_rim_over_inner > dd.fut4_rim_over_inner).sum()),
                    "rim_wilcoxon_p": float(w.pvalue)})
        log(f"A {comp} CD45 ({exc}): {chk[-1]['cd45_positive']} / {len(d)} cells >= {RATIO_POS:g} x background; CD45 / "
            f"FUT4 {chk[-1]['cd45_over_fut4_median']}; rim / inner CD45 {chk[-1]['cd45_rim_over_inner_median']} vs FUT4 "
            f"{chk[-1]['fut4_rim_over_inner_median']} ({chk[-1]['cells_cd45_more_rim']} / {len(dd)}; p {w.pvalue:.2g})")
    chk = pd.DataFrame(chk)
    mar = chk[chk.compartment == "bone_marrow"].cd45_over_fut4_median.iloc[0]
    per = chk[chk.compartment == "perilymph"].cd45_over_fut4_median.iloc[0]
    assert per > mar, "perilymph CD45 channel would be explainable as FUT4 spill-over"
    assert (chk.rim_wilcoxon_p < 0.05).all() and (chk.cd45_rim_over_inner_median > chk.fut4_rim_over_inner_median).all()
    write_csv(chk, "r9_figures", "f5a_cd45_channel_check.csv",
              "Fig. 4d CD45 channel check (D93): per compartment and CD45 excitation, CD45+ cells (ratio >= 2), "
              "background-subtracted CD45 / FUT4 and rim enrichment of CD45 vs FUT4 in the same cells (paired Wilcoxon): "
              "the 561-nm-off CD45 signal is a membrane ring, not FUT4 spill-over",
              required=["compartment", "cd45_excitation", "cells", "cd45_positive", "rim_wilcoxon_p"])
    return df


# ------------------------------------------------------------------------------------------------ B  parabiont (D87)
def load_pb(d: Path, stem: str, project: bool = False):
    """Channels of one field: the sharpest optical section, or the maximum projection of all sections (project)."""
    by = {k: (sorted(d.glob(f"{stem}_C{c}.tif")) or sorted(d.glob(f"{stem}_C{c}_Z*.tif"))) for k, c in PB_CH.items()}
    z = by["dapi"]
    if project:
        return {key: np.stack([tifffile.imread(p).astype(float) for p in v]).max(0) for key, v in by.items()}, None, \
            len(z), by
    k = 0 if len(z) == 1 else int(np.argmax([ndi.laplace(gaussian(tifffile.imread(p).astype(float), 1)).var()
                                             for p in z]))
    return {key: tifffile.imread(v[k]).astype(float) for key, v in by.items()}, k, len(z), by


def tissue_mask(dapi, px, box):
    nuc = gaussian(dapi, 0.5 / px, preserve_range=True)
    t = ndi.binary_closing(gaussian(dapi, 6.0 / px, preserve_range=True) > threshold_otsu(nuc) * 0.25, iterations=3)
    if box:
        t[box[2]:box[3], box[0]:box[1]] = False
    return t


def marrow_islands(marker, tissue, px):
    """Marrow islands of a CD45 image: CD45 smoothed at 10 um above its Otsu threshold within tissue, dilated 5 um."""
    sm = gaussian(marker, 10.0 / px, preserve_range=True)
    m = (sm > threshold_otsu(sm[tissue])) & tissue
    return ndi.binary_dilation(m, iterations=int(round(5.0 / px))) & tissue


def gfp_bodies(gfp, tissue, px):
    """GFP+ cell bodies within a tissue mask: labels, count, threshold and the mean GFP of every body."""
    gs = gaussian(gfp, 0.5 / px, preserve_range=True)
    tv = gs[tissue]
    med = float(np.median(tv))
    thr = med + 10 * 1.4826 * float(np.median(np.abs(tv - med)))
    lab = label(ndi.binary_fill_holes(gs > thr) & tissue)
    keep = [r for r in regionprops(lab) if r.area * px * px >= 15.0 and r.solidity >= 0.7]
    means = [float(gfp[lab == r.label].mean()) for r in keep]
    return np.where(np.isin(lab, [r.label for r in keep]), lab, 0), len(keep), thr, means


def marker_cells(marker, tissue, px, thr, glab):
    """Marker+ objects (>= 15 um2) within tissue, how many overlap a GFP+ body, and the objects themselves."""
    fl = label((gaussian(marker, 0.3 / px, preserve_range=True) > thr) & tissue)
    fr = [r for r in regionprops(fl) if r.area * px * px >= 15.0]
    pos = {r.label: bool((glab[fl == r.label] > 0).any()) for r in fr}
    return len(fr), sum(pos.values()), [(r, pos[r.label]) for r in fr]


IMAGE_FIELDS = {"60x_1_11", "60x_sl2"}                   # the two Fig. 4f image fields (930 crops)


def part_b():
    # FUT4 threshold for cochlear tissue: Otsu of FUT4 (sigma 0.3 um) in the pooled 60x marrow tissue (same settings)
    marrow_fut = []
    for fid, sub, stem, mag, comp, reg, mk, proj, box, pooled in PB_FIELDS:
        if mag == "60x" and comp == "marrow" and pooled:
            ch, *_ = load_pb(PB / sub, stem)
            px, _ = sidecar(PB / sub / f"{stem}_Info.txt")
            t = tissue_mask(ch["dapi"], px, box)
            marrow_fut.append(gaussian(ch["marker"], 0.3 / px, preserve_range=True)[t][::7])
    fut_thr = float(threshold_otsu(np.concatenate(marrow_fut)))
    log(f"B FUT4 threshold (60x marrow tissue, Otsu): {fut_thr:.0f}")
    rows, means, objs = [], {}, []
    for fid, sub, stem, mag, comp, reg, mk, proj, box, pooled in PB_FIELDS:
        d = PB / sub
        px, las = sidecar(d / f"{stem}_Info.txt")
        ch, k, nz, by = load_pb(d, stem, project=proj)
        t = tissue_mask(ch["dapi"], px, box)
        if comp == "split":
            mi = marrow_islands(ch["marker"], t, px)
            parts = [("marrow", "temporal-bone marrow islands", mi), ("cochlea", "cochlear soft tissue", t & ~mi)]
            mk_thr = float(threshold_otsu(gaussian(ch["marker"], 0.3 / px, preserve_range=True)[mi]))
        else:
            parts = [(comp, reg, t)]
            mk_thr = fut_thr if mk == "FUT4" and mag == "60x" else np.nan
        files = [p for v in by.values() for p in (v if proj else [v[k]])]
        for cp, rg, m in parts:
            glab, n_gfp, gthr, gm = gfp_bodies(ch["gfp"], m, px)
            means[(fid, cp)] = gm
            area = float(m.sum() * px * px / 1e6)
            n_mk = n_mk_gfp = np.nan
            cells = []
            if (fid, cp) in COUNT_MARKER:
                n_mk, n_mk_gfp, cells = marker_cells(ch["marker"], m, px, mk_thr, glab)
            if fid in IMAGE_FIELDS:
                for r_ in regionprops(glab):
                    objs.append({"field": fid, "object": "GFP+ body", "id": r_.label,
                                 "centroid_x_px": round(r_.centroid[1], 1), "centroid_y_px": round(r_.centroid[0], 1),
                                 "area_um2": round(r_.area * px * px, 1), "gfp_positive": True,
                                 "mean_gfp": round(float(ch["gfp"][glab == r_.label].mean()), 1)})
                for r_, gp in cells:
                    objs.append({"field": fid, "object": "FUT4+ cell", "id": r_.label,
                                 "centroid_x_px": round(r_.centroid[1], 1), "centroid_y_px": round(r_.centroid[0], 1),
                                 "area_um2": round(r_.area * px * px, 1), "gfp_positive": gp, "mean_gfp": np.nan})
            rows.append({"field": fid, "compartment": cp, "region": rg, "magnification": mag, "um_per_px": px,
                         "optical_section": f"maximum projection of {nz}" if proj else f"{k + 1} of {nz}",
                         "pooled": pooled, "tissue_mm2": round(area, 4), "gfp_bodies": n_gfp,
                         "gfp_bodies_per_mm2": round(n_gfp / area, 1), "marker": mk, "marker_cells": n_mk,
                         "marker_gfp_cells": n_mk_gfp, "gfp_threshold": round(gthr, 1),
                         "marker_threshold": round(mk_thr, 1) if (fid, cp) in COUNT_MARKER else np.nan,
                         "gfp_laser_pct_pmt": f"{las[2][0]:g} / {las[2][1]}",
                         "marker_laser_pct_pmt": f"{las[4][0]:g} / {las[4][1]}",
                         "excluded_box_x0_x1_y0_y1_px": "" if box is None else ",".join(map(str, box)),
                         "source_files": "; ".join(str(p.relative_to(ROOT)).replace(chr(92), "/") for p in files),
                         "source_md5": "; ".join(md5(p) for p in files)})
    df = pd.DataFrame(rows)
    # detection check: would the marrow GFP+ bodies (same GFP settings) clear each cochlear compartment's threshold?
    df["detection_reference"] = ""
    df["reference_bodies_detectable"] = ""
    for (fid, cp), (rf, rc) in DETECT_REF.items():
        i = df.index[(df.field == fid) & (df.compartment == cp)][0]
        ref = means[(rf, rc)]
        thr = float(df.at[i, "gfp_threshold"])
        df.at[i, "detection_reference"] = f"{rf} {rc}"
        df.at[i, "reference_bodies_detectable"] = f"{sum(v > thr for v in ref)} / {len(ref)}"
    for _, r in df.iterrows():
        msg = f"B {r.field} {r.compartment} ({r.region}): {r.gfp_bodies} GFP+ bodies in {r.tissue_mm2:.4f} mm2"
        msg += f" ({r.gfp_bodies_per_mm2:.0f}/mm2)"
        if not np.isnan(r.marker_cells):
            msg += f"; {r.marker}+ cells {int(r.marker_cells)}, GFP+ {int(r.marker_gfp_cells)}"
        if r.detection_reference:
            msg += f"; detection check {r.reference_bodies_detectable} ({r.detection_reference})"
        log(msg)
    write_csv(df, "r9_figures", "f4_parabiont_fields.csv",
              "Fig. 4f,i: parabiont mice (54 d; premise D87: marrow of wild-type origin, peripheral cells GFP+) - GFP+ "
              "(peripheral) cell bodies per mm2 of temporal-bone marrow and cochlear tissue, GFP status of cochlear FUT4+ "
              "and CD45+ cells, and a detection check against marrow GFP+ cells imaged with the same settings (931)",
              required=["field", "compartment", "tissue_mm2", "gfp_bodies", "gfp_bodies_per_mm2"])
    ob = pd.DataFrame(objs)
    assert (ob[ob.object == "FUT4+ cell"].shape[0] == int(df.loc[df.field == "60x_sl2", "marker_cells"].iloc[0]))
    write_csv(ob, "r9_figures", "f4_parabiont_objects.csv",
              "Fig. 4f objects of the two image fields (931, D88): GFP+ (peripheral) bodies of marrow field 1_11 and the "
              "lateral wall's FUT4+ cells (GFP status) and GFP+ bodies, centroids in image pixels (crop choice in 930, "
              "asterisks in 912)", required=["field", "object", "centroid_x_px", "centroid_y_px", "gfp_positive"])
    return df


if __name__ == "__main__":
    part_a()
    part_b()
    log("done")
