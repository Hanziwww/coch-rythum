"""Round 9 — publication panels from the user's imaging data (data/photos). Pixel values are only cropped and linearly
mapped to the display range; nothing is filtered, gamma-adjusted, background-subtracted, cloned or erased.

FUT4 / CD45 / DAPI confocal fields (data/photos/FUT4_WT_2026-10-12: Olympus, 60x, 1024 x 1024; 16-bit single-channel
TIFFs C1 = DAPI, C2 = FUT4-488, C3 = CD45-555; bone marrow, endolymph and perilymph preparations of wild-type mice).
Pixel size and laser powers are read per field from its acquisition sidecar (D86: marrow 0.228 um, perilymph and
endolymph fields 1-8 0.142 um (scan zoom 1.6), endolymph field 9 0.228 um). The 561-nm line was at 0 % in every
perilymph field and in endolymph fields 1-4; their CD45 channel still carries a CD45 membrane ring (931 check, D93:
rim-enriched relative to FUT4 and larger relative to FUT4 than in marrow, so not FUT4 spill-over), excited by the
other lines. CD45 is therefore exported for every field (an earlier version showed n.a. there); its intensity is not
comparable between preparations acquired with and without the 561-nm line (logged per crop).
- display range per channel, identical for every field and compartment: [0.5th, 99.95th] percentile of all pixels of
  the complete fields (every 4th pixel in x and y); linear mapping to 0-1;
- fields chosen by visual review of all fields under that common scaling (in focus, intact isolated cells, typical of
  the compartment); each crop (48 um) is centred on the cell with the highest FUT4 / background ratio (nuclei >= 15 um2
  from an Otsu threshold of the Gaussian-smoothed DAPI; cell = nucleus dilated by 1.5 um; background = median outside
  all cells), so the cell is chosen by rule, not by eye;
- per crop: raw 16-bit channel crops (TIFF, values unchanged) and 8-bit renders at native pixel size (DAPI, FUT4,
  CD45 greyscale; merge FUT4 green, CD45 magenta, DAPI blue, additive); scale bars are drawn by the figure code.
3D reconstructions and manually annotated ('painted') renders (data/photos/cochlea_3d; 8-bit exports from the user's
rendering software): pixels unchanged; cropped to remove the burned-in scale annotation (bottom 8 % and the same
margin on the right, keeping the panel square); um per pixel from the length of the burned-in scale bar.

Spatial-transcriptomics deconvolution map (neutrophil states; the user's 8-bit export): pixels unchanged; key colours
sampled from the user's legend image.

Parabiont mice (D87; data/photos/parabiosis_54d, Olympus 16-bit TIFFs; C1 DAPI, C2 GFP, C4 FUT4-647): temporal-bone
marrow (60x field 1_11, one optical section, 0.114 um/px) and cochlear lateral wall (60x sl-2, maximum-intensity
projection of its 22 sections, 0.228 um/px), both acquired with identical settings (GFP 3 % / 550 V, FUT4 6 % / 660 V,
DAPI 10 % / 710 V). One linear display range per channel for both images ([0.5th, 99.95th] percentile of their pooled
pixels, every 2nd pixel); 60-um crops (D88) at the window holding the most objects counted by 931
(results/r9_figures/f4_parabiont_objects.csv; run 931 first): GFP+ cell bodies in the marrow field, FUT4+ cells in the
lateral wall (ties: the larger positive area).

Outputs: figures/panels/*.png, *.tif; results/r9_figures/image_panels_log.csv (source files, md5, crop box, display
ranges, um per pixel, saturated pixels in the crop), image_spatial_key.csv (colour key of the spatial map).
Run (local; the image data are not on the server): COCH_ROOT=/path/to/coch-rythum python scripts/930_image_panels.py
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUT4 = ROOT / "data" / "photos" / "FUT4_WT_2026-10-12"
R3D = ROOT / "data" / "photos" / "cochlea_3d"
OUT = ROOT / "figures" / "panels"
MOD = "r9_figures"
CROP_UM = 48.0
CH = {"C1": "DAPI", "C2": "FUT4", "C3": "CD45"}
MERGE = {"C2": (0.0, 1.0, 0.0), "C3": (1.0, 0.0, 1.0), "C1": (0.0, 0.0, 1.0)}
# (compartment folder, field, role) — chosen after review of every field under the common display range
FUT4_CROPS = [("bone marrow", 3, "main"), ("perilymphy", 8, "main"), ("endolymph", 5, "main")]
LABEL = {"bone marrow": "bone_marrow", "perilymphy": "perilymph", "endolymph": "endolymph"}
# 3D renders: (file, panel id, scale-bar um, measured bar length px, annotated?)
RENDERS = [("painted/1.png", "vessels_view1_annotated", 100, 97, True),
           ("slide_03/color_images/slide_03_image_12.png", "vessels_view1_original", 100, 97, False),
           ("painted/3.png", "vessels_view2_annotated", 100, 91, True),
           ("slide_08/color_images/slide_08_image_04.png", "vessels_view2_original", 100, 91, False),
           ("slide_08/color_images/slide_08_image_02.png", "marrow_neutrophils_overview", 300, 133, False),
           ("slide_08/color_images/slide_08_image_01.png", "marrow_neutrophils_zoom", 100, 118, False)]
# parabiont crops (D87): (panel id, folder, stem, maximum projection?, channel that picks the crop window)
PB = ROOT / "data" / "photos" / "parabiosis_54d" / "cell-type-bone-cochlea"
PARABIONT = [("parabiont_marrow", "parabiotic54d-gfp-647fut4-60x", "1_11", False, "C2", "temporal-bone marrow",
              "60x_1_11", "GFP+ body"),
             ("parabiont_cochlea", "parabiotic54d-gfp-647fut4-60x-sl-2", "1_1", True, "C4", "cochlear lateral wall",
              "60x_sl2", "FUT4+ cell")]
PB_CH = {"C1": "DAPI", "C2": "GFP", "C4": "FUT4"}
PB_MERGE = {"C2": (0.0, 1.0, 0.0), "C4": (1.0, 0.0, 1.0), "C1": (0.0, 0.0, 1.0)}
PB_CROP_UM = 60.0
# spatial-transcriptomics deconvolution map (neutrophil states; the user's export, no scale bar) and its legend colours,
# sampled at the centre of each legend dot (radius 10 px) of the user's legend image
SPATIAL = ("slide_03/spatial_transcriptomics/slide_03_image_06.png", "spatial_neutrophil_states")
LEGEND = ("slide_03/legends/slide_03_image_10.png", {"Band": 157, "Mature": 220, "Activated": 285, "Aged": 348}, 100)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sidecar(p: Path):
    """Pixel size (um) and laser power (%) per channel (C1..C3) from an Olympus acquisition sidecar."""
    s = p.read_text(errors="replace")
    px = float(re.search(r"Pixel size: ([\d.]+)um", s).group(1))
    las = {f"C{c}": float(v) for c, _, v in re.findall(r"Channel (\d): (\d+) Laser power: ([\d.]+)%", s)}
    return px, las


def fields():
    """(compartment folder, field, files, um per px, laser % per channel) of every field with all three channels."""
    for tdir in sorted(p for p in FUT4.iterdir() if p.is_dir()):
        for c1 in sorted(tdir.glob("*_C1.tif")):
            stem = c1.name[:-len("_C1.tif")]
            files = {c: tdir / f"{stem}_{c}.tif" for c in CH}
            if all(f.exists() for f in files.values()):
                px, las = sidecar(tdir / f"{stem}_Info.txt")
                yield tdir.name, int(stem.rsplit("_", 1)[1]), files, px, las


def parabiont_crops():
    """Fig. 4f crops (D87; letter d until D134): marrow and lateral wall of a parabiont, one display range per channel for both."""
    from skimage.io import imread
    imgs = {}
    obj = pd.read_csv(ROOT / "results" / MOD / "f4_parabiont_objects.csv")
    for pid, sub_, stem, proj, pick, region, fid, kind in PARABIONT:
        d = PB / sub_
        files = {c: sorted(d.glob(f"{stem}_{c}.tif")) or sorted(d.glob(f"{stem}_{c}_Z*.tif")) for c in PB_CH}
        if proj:
            im = {c: np.stack([imread(f) for f in v]).max(0) for c, v in files.items()}
            use = files
        else:
            assert all(len(v) == 1 for v in files.values()), pid
            im = {c: imread(v[0]) for c, v in files.items()}
            use = files
        px, las = sidecar(d / f"{stem}_Info.txt")
        imgs[pid] = (im, px, las, use, proj, pick, region, obj[(obj.field == fid) & (obj.object == kind)])
    pooled = {c: np.concatenate([v[0][c][::2, ::2].ravel() for v in imgs.values()]) for c in PB_CH}
    rng = {c: tuple(float(v) for v in np.percentile(pooled[c], [0.5, 99.95])) for c in PB_CH}
    rows = []
    for pid, (im, px, las, use, proj, pick, region, ob) in imgs.items():
        n = int(round(PB_CROP_UM / px))
        dapi = im["C1"].astype(np.float64)
        nuc = gaussian(dapi, 0.5 / px, preserve_range=True)
        tissue = gaussian(dapi, 6.0 / px, preserve_range=True) > threshold_otsu(nuc) * 0.25
        sm = gaussian(im[pick].astype(np.float64), 0.5 / px, preserve_range=True)
        tv = sm[tissue]
        thr = float(np.median(tv) + 10 * 1.4826 * np.median(np.abs(tv - np.median(tv))))
        # the window with the most objects counted by 931 (GFP+ bodies / FUT4+ cells; ties: the larger positive area)
        cen = np.zeros(dapi.shape)
        for _, o in ob.iterrows():
            cen[int(o.centroid_y_px), int(o.centroid_x_px)] += 1.0
        area = ndi.uniform_filter((sm > thr).astype(np.float64), size=n, mode="constant")
        score = ndi.uniform_filter(cen, size=n, mode="constant") * n * n + 1e-3 * area
        h, w = dapi.shape
        score[:n // 2, :] = score[h - n // 2:, :] = -1
        score[:, :n // 2] = score[:, w - n // 2:] = -1
        cy, cx = (int(v) for v in np.unravel_index(int(np.argmax(score)), score.shape))
        box = (cy - n // 2, cy - n // 2 + n, cx - n // 2, cx - n // 2 + n)
        sl = (slice(box[0], box[1]), slice(box[2], box[3]))
        inside = int(((ob.centroid_y_px >= box[0]) & (ob.centroid_y_px < box[1]) & (ob.centroid_x_px >= box[2])
                      & (ob.centroid_x_px < box[3])).sum())
        disp = {}
        for c in PB_CH:
            raw = np.asarray(im[c])[sl]
            Image.fromarray(raw).save(OUT / f"{pid}_{PB_CH[c]}_raw16.tif")
            lo, hi = rng[c]
            disp[c] = np.clip((raw.astype(np.float64) - lo) / (hi - lo), 0, 1)
            Image.fromarray((disp[c] * 255).round().astype(np.uint8)).save(OUT / f"{pid}_{PB_CH[c]}.png")
        merge = sum(disp[c][..., None] * np.array(col) for c, col in PB_MERGE.items())
        Image.fromarray((np.clip(merge, 0, 1) * 255).round().astype(np.uint8)).save(OUT / f"{pid}_merge.png")
        srcs = [f for v in use.values() for f in v]
        rows.append({"panel": pid, "kind": "confocal crop (parabiont)", "role": "main", "compartment": region,
                     "field": np.nan,
                     "source_files": "; ".join(str(f.relative_to(ROOT)).replace(chr(92), "/") for f in srcs),
                     "source_md5": "; ".join(md5(f) for f in srcs),
                     "crop_y0_y1_x0_x1": f"{box[0]}-{box[1]}, {box[2]}-{box[3]}", "um_per_px": px,
                     "crop_um": round(n * px, 1), "selected_cell_fut4_over_background": np.nan,
                     "laser_power_pct": "; ".join(f"{PB_CH[c]} {las.get(c, np.nan):g}" for c in PB_CH),
                     "cd45_acquired": np.nan,
                     "display_ranges": "; ".join(f"{PB_CH[c]} {rng[c][0]:.0f}-{rng[c][1]:.0f}" for c in PB_CH),
                     "saturated_px_in_crop": "; ".join(f"{PB_CH[c]} {int((np.asarray(im[c])[sl] >= 65535).sum())}"
                                                      for c in PB_CH),
                     "processing": ("maximum-intensity projection of " + str(len(use["C1"])) + " optical sections; "
                                    if proj else "one optical section; ") + "crop at the window with the most "
                                   + ("GFP+ cell bodies" if pick == "C2" else "FUT4+ cells") + " counted by 931 ("
                                   + str(inside) + "); linear display scaling "
                                   "with one "
                                   "range per channel for both parabiont crops; merge additive (GFP green, FUT4 magenta, "
                                   "DAPI blue); no filtering or gamma"})
        log(f"{pid}: {px} um/px, crop {box} ({n * px:.1f} um); {len(ob)} objects in the field, {inside} in the crop")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allf = list(fields())
    # one display range per channel from all complete fields (CD45 included where the 561-nm line was at 0 %; D93)
    pool = {c: np.concatenate([np.asarray(Image.open(f[c]))[::4, ::4].ravel() for _, _, f, _, las in allf]) for c in CH}
    rng = {c: (float(np.percentile(pool[c], 0.5)), float(np.percentile(pool[c], 99.95))) for c in CH}
    n_cd45 = sum(1 for *_, las in allf if las.get("C3", 0) > 0)
    log(f"{len(allf)} complete fields ({n_cd45} with the 561-nm line on); display ranges "
        + ", ".join(f"{CH[c]} {lo:.0f}-{hi:.0f}" for c, (lo, hi) in rng.items()))
    rows = []
    for comp, fld, role in FUT4_CROPS:
        files, px, las = next((f, p, la) for t, k, f, p, la in allf if t == comp and k == fld)
        half = int(round(CROP_UM / px / 2))
        chans = list(CH)
        im = {c: np.asarray(Image.open(files[c])).astype(np.float64) for c in CH}
        d = gaussian(im["C1"], sigma=2, preserve_range=True)
        nuc = ndi.binary_fill_holes(ndi.binary_opening(d > threshold_otsu(d), iterations=2))
        lab = label(nuc)
        ring = int(round(1.5 / px))
        cells = ndi.grey_dilation(lab, size=(2 * ring + 1, 2 * ring + 1))
        bg = float(np.median(im["C2"][ndi.binary_dilation(cells > 0, iterations=ring) == 0]))
        best = max(((im["C2"][cells == r.label].mean() / max(bg, 1.0), r.centroid) for r in regionprops(cells)
                    if r.area * px * px >= 15), key=lambda x: x[0])
        cy, cx = (int(np.clip(round(v), half, 1024 - half)) for v in best[1])
        box = (cy - half, cy + half, cx - half, cx + half)
        sl = (slice(box[0], box[1]), slice(box[2], box[3]))
        tag = f"fut4_{LABEL[comp]}_f{fld}"
        for c in CH:                                     # drop files of an earlier run (e.g. a CD45 crop now n.a.)
            for p in (OUT / f"{tag}_{CH[c]}_raw16.tif", OUT / f"{tag}_{CH[c]}.png"):
                p.unlink(missing_ok=True)
        disp = {}
        for c in chans:
            raw = np.asarray(Image.open(files[c]))[sl]
            Image.fromarray(raw).save(OUT / f"{tag}_{CH[c]}_raw16.tif")
            lo, hi = rng[c]
            disp[c] = np.clip((raw.astype(np.float64) - lo) / (hi - lo), 0, 1)
            Image.fromarray((disp[c] * 255).round().astype(np.uint8)).save(OUT / f"{tag}_{CH[c]}.png")
        merge = sum(disp[c][..., None] * np.array(col) for c, col in MERGE.items() if c in disp)
        Image.fromarray((np.clip(merge, 0, 1) * 255).round().astype(np.uint8)).save(OUT / f"{tag}_merge.png")
        cd45_note = "" if las.get("C3", 0) > 0 else ("; CD45 acquired with the 561-nm line at 0 %: membrane signal "
                                                     "excited by the other lines (presence only; D93)")
        rows.append({"panel": tag, "kind": "confocal crop", "role": role, "compartment": LABEL[comp], "field": fld,
                     "source_files": "; ".join(str(files[c].relative_to(ROOT)) for c in chans),
                     "source_md5": "; ".join(md5(files[c]) for c in chans),
                     "crop_y0_y1_x0_x1": f"{box[0]}-{box[1]}, {box[2]}-{box[3]}", "um_per_px": px,
                     "crop_um": round(2 * half * px, 1), "selected_cell_fut4_over_background": round(best[0], 2),
                     "laser_power_pct": "; ".join(f"{CH[c]} {las.get(c, np.nan):g}" for c in CH),
                     "cd45_acquired": "C3" in chans,
                     "cd45_excitation": "561 nm" if las.get("C3", 0) > 0 else "561 nm at 0 % (other lines)",
                     "display_ranges": "; ".join(f"{CH[c]} {rng[c][0]:.0f}-{rng[c][1]:.0f}" for c in chans),
                     "saturated_px_in_crop": "; ".join(f"{CH[c]} {int((np.asarray(Image.open(files[c]))[sl] >= 65535).sum())}"
                                                      for c in chans),
                     "processing": "crop; linear display scaling with the common per-channel range; merge additive "
                                   "(FUT4 green, CD45 magenta, DAPI blue); no filtering or gamma" + cd45_note})
        log(f"{tag}: {px} um/px, cell FUT4/bg {best[0]:.1f}, crop {box}{cd45_note}")
    for rel, pid, bar_um, bar_px, annotated in RENDERS:
        src = R3D / rel
        a = Image.open(src).convert("RGB")
        w, h = a.size
        k = int(round(h * 0.92))
        crop = a.crop((0, 0, k, k))                          # removes the burned-in bar + label (bottom-left)
        crop.save(OUT / f"{pid}.png")
        rows.append({"panel": pid, "kind": "3D render (annotated)" if annotated else "3D render", "role": "main",
                     "compartment": "cochlea and temporal bone", "field": np.nan,
                     "source_files": str(src.relative_to(ROOT)), "source_md5": md5(src),
                     "crop_y0_y1_x0_x1": f"0-{k}, 0-{k}", "um_per_px": round(bar_um / bar_px, 4),
                     "crop_um": round(k * bar_um / bar_px), "selected_cell_fut4_over_background": np.nan,
                     "display_ranges": "unchanged (8-bit export)", "saturated_px_in_crop": "",
                     "processing": "crop only (removes the burned-in scale annotation); pixels unchanged; um/px from the "
                                   f"burned-in {bar_um}-um bar ({bar_px} px)" + ("; colour overlay = the user's manual "
                                                                                  "vessel annotation" if annotated else "")})
    # where the higher-magnification neutrophil render sits in the overview: normalised cross-correlation of the vessel
    # (blue) channel over scales around the ratio of the two pixel sizes; the best match is recorded and drawn as a box
    from skimage.feature import match_template
    ov = np.asarray(Image.open(OUT / "marrow_neutrophils_overview.png").convert("RGB"), dtype=float)[..., 2]
    zm = Image.open(OUT / "marrow_neutrophils_zoom.png").convert("RGB")
    f0 = (100 / 118) / (300 / 133)
    best = None
    for sc in f0 * np.array([0.9, 0.95, 1.0, 1.05, 1.1]):
        n = int(round(zm.size[0] * sc))
        zs = np.asarray(zm.resize((n, n), Image.LANCZOS), dtype=float)[..., 2]
        r = match_template(ov, zs)
        if best is None or r.max() > best[0]:
            y, x = np.unravel_index(r.argmax(), r.shape)
            best = (float(r.max()), int(x), int(y), n)
    assert best[0] > 0.5, f"zoom not located in the overview (NCC {best[0]:.2f})"
    k = next(i for i, r in enumerate(rows) if r["panel"] == "marrow_neutrophils_overview")
    rows[k]["zoom_box_x_y_size_px"] = f"{best[1]},{best[2]},{best[3]}"
    rows[k]["processing"] += (f"; zoom region located by normalised cross-correlation (NCC {best[0]:.2f}) at x {best[1]}, "
                              f"y {best[2]}, size {best[3]} px")
    log(f"zoom located in overview: NCC {best[0]:.3f}, x {best[1]} y {best[2]} size {best[3]}")
    src = R3D / SPATIAL[0]
    Image.open(src).convert("RGB").save(OUT / f"{SPATIAL[1]}.png")
    leg = np.asarray(Image.open(R3D / LEGEND[0]).convert("RGB")).astype(int)
    yy, xx = np.mgrid[0:leg.shape[0], 0:leg.shape[1]]
    keys = {}
    for state, cy in LEGEND[1].items():
        m = (yy - cy) ** 2 + (xx - LEGEND[2]) ** 2 <= 100
        assert leg[m].std(axis=0).max() == 0, f"legend dot {state} not uniform"
        keys[state] = "#%02X%02X%02X" % tuple(leg[m][0])
    write_csv(pd.DataFrame([{"state": k, "colour": v, "source": LEGEND[0]} for k, v in keys.items()]), MOD,
              "image_spatial_key.csv", "Colour key of the spatial-transcriptomics neutrophil-state map (sampled from the "
              "user's legend image)", required=["state", "colour"])
    rows.append({"panel": SPATIAL[1], "kind": "spatial deconvolution map", "role": "main",
                 "compartment": "temporal bone", "field": np.nan, "source_files": f"{src.relative_to(ROOT)}; "
                 f"{(R3D / LEGEND[0]).relative_to(ROOT)}", "source_md5": f"{md5(src)}; {md5(R3D / LEGEND[0])}",
                 "crop_y0_y1_x0_x1": "full image", "um_per_px": np.nan, "crop_um": np.nan,
                 "selected_cell_fut4_over_background": np.nan, "display_ranges": "unchanged (8-bit export)",
                 "saturated_px_in_crop": "", "processing": "unchanged pixels (RGBA to RGB); key colours sampled from the "
                 "user's legend: " + "; ".join(f"{k} {v}" for k, v in keys.items())})
    log("spatial key colours " + ", ".join(f"{k} {v}" for k, v in keys.items()))
    rows += parabiont_crops()
    write_csv(pd.DataFrame(rows), MOD, "image_panels_log.csv",
              "Imaging panels: source files and md5, crop boxes, common linear display ranges, um per pixel, saturated "
              "pixels, processing steps (FUT4 / CD45 / DAPI confocal crops; 3D vascular and neutrophil renders; "
              "parabiont GFP / FUT4 crops)",
              required=["panel", "source_files", "source_md5", "um_per_px", "processing"])
    log("done")


if __name__ == "__main__":
    main()
