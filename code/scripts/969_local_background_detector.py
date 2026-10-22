"""Round 10 (D155) — a local-background detector, fixed in advance, applied to every tissue and checked against the
blinded labels. User: "Run option 1: a local-background detector fixed in advance, applied to every tissue and checked
against the blinded labels; report the result whichever way it falls." Reason (D154): the survey's detector thresholds
GFP at the tissue median + 10 robust SD of the whole image; in the rendered brain projection the autofluorescent,
textured background inflates that SD (20.7 grey levels) and the rendering saturates, so its threshold lies at 241 of 255
grey levels and dim cells cannot be detected there, while the converted cochlear sections (SD of a few grey levels) are
searched far more sensitively. A detector that subtracts the local background tests whether this asymmetry explains why
the cochlea counts above the brain.

Everything below was fixed in this docstring before the first run (the counts were not looked at before):
  format       the brain image's (966): each analysed section (934: >= 10 um apart, >= 5% tissue) of every wild-type
               survey field (the 43 fields of Supplementary Fig. 2c; noise-exposed pair excluded) box-averaged to 2.773
               um per pixel (main) and to 1.387, 1.248 and 1.040 um (sensitivity), DAPI in the tissue mask only; the
               brain projection (964) at the same assumed pixel sizes; GFP smoothed at sigma 0.5 um (934).
  background   white top-hat of the smoothed GFP with a disk of radius 15 um (removes structures wider than a cell).
  detection    top-hat above its own tissue median + 10 robust SD (1.4826 x MAD, floor 1 grey level), holes filled,
               area >= 15 um2, solidity >= 0.7 (934's conventions, applied to the background-subtracted image).
  cell call    primary: the upgraded rule's criteria without the nucleus criterion: area >= 25 um2, mean smoothed GFP
               >= 2 x the median smoothed GFP of the tissue 2-6 um beyond the body (>= 10 pixels), not broad-spectrum
               (lectin fields: mean far-red inside the body above that channel's tissue median + 10 robust SD);
               secondary: the same with the contrast of the strict brain-format rule (>= 6; 968).
  agreement    the 152 decided blinded labels (968: 946 and 960 reviews, 56 cells) matched to a detected body of the
               same converted section within 8 um (main pixel size); an unmatched labelled body counts as not kept;
               precision and sensitivity per cell call, overall and by compartment; detected bodies without a label are
               counted but their precision is not measured (labels exist only for bodies of the survey's detector).
  densities    cells per mm2 of tissue per wild-type animal and tissue (sections summed), the brain from its projection
               (an upper bound for a section: a projection collects the cells of its whole depth).
The saturation of the brain rendering remains: no detector recovers intensities above 255 grey levels.
Outputs (lib/io.write_csv, results/r10_parabiosis/): localbg_bodies.csv (every detected body: identifiers, features,
calls), localbg_densities.csv (animal x tissue x pixel size: tissue area, bodies, cells by both calls, densities),
localbg_agreement.csv (labels against both calls; the survey detector's brain-format calls of 968 alongside).
Run (local; the images are not on the server): COCH_ROOT=D:/coch-rythum python scripts/969_local_background_detector.py
[--tiles: after the run, the Supplementary Fig. 5 objects of this detector (added after the run, display only)]
Result (2026-10-18, D155): label agreement F1 0.55 (upgraded criteria) and 0.61 (contrast >= 6) against 0.54 and 0.58 for
the survey detector, so by the use rule recorded before the run this detector supplies Fig. 4g.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S966 = load_module("s966", "966_matched_format_counts.py")
S964, S934 = S966.S964, S966.S934
MOD = "r10_parabiosis"
R10 = RESULTS / MOD
TOPHAT_UM, K_SD, MIN_AREA, MIN_SOLIDITY = 15.0, 10.0, 15.0, 0.7
AREA_CELL, CONTRAST, CONTRAST_STRICT, RING_UM = 25.0, 2.0, 6.0, (2.0, 6.0)
MATCH_UM = 8.0


def detect(chs, names, rel, um):
    """Local-background detection on one converted image (channels C x H x W at um per pixel) -> tissue mm2, bodies."""
    a = np.stack(chs)[:, None]
    meta = dict(um_x=um, um_z=None, channels=names)
    planes = list(S934.sampled_planes(a, meta))
    if not planes:
        return 0.0, []
    tis = planes[0][1]
    gi, _, fi = S934.channel_roles(names)
    if "lectin" not in rel.lower():
        fi = None
    gs = filters.gaussian(chs[gi].astype(np.float32), sigma=0.5 / um, preserve_range=True)
    th = morphology.white_tophat(gs, morphology.disk(max(1, int(round(TOPHAT_UM / um)))))
    med, sd = S934.robust(th[tis])
    pos = ndi.binary_fill_holes(th > med + K_SD * sd) & tis
    lab = measure.label(pos)
    fthr, fs = None, None
    if fi is not None:
        fs = filters.gaussian(chs[fi].astype(np.float32), sigma=0.5 / um, preserve_range=True)
        fmed, fsd = S934.robust(fs[tis])
        fthr = fmed + 10 * fsd
    H, W = gs.shape
    pad = int(np.ceil(RING_UM[1] / um)) + 1
    out = []
    for r in measure.regionprops(lab):
        area = r.area * um * um
        if area < MIN_AREA or r.solidity < MIN_SOLIDITY:
            continue
        y0, x0, y1, x1 = r.bbox
        ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        bm = lab[ya:yb, xa:xb] == r.label
        dd = ndi.distance_transform_edt(~bm) * um
        ann = (dd > RING_UM[0]) & (dd <= RING_UM[1]) & tis[ya:yb, xa:xb]
        g_ = gs[ya:yb, xa:xb]
        bg = float(np.median(g_[ann])) if ann.sum() >= 10 else np.nan
        con = float(g_[bm].mean()) / max(bg, 1.0) if np.isfinite(bg) else np.nan
        broad = bool(fs is not None and float(fs[tuple(r.coords.T)].mean()) > fthr)
        ok = area >= AREA_CELL and np.isfinite(con) and not broad
        out.append(dict(y=float(r.centroid[0]), x=float(r.centroid[1]), area_um2=area,
                        tophat_sd=(float(th[tuple(r.coords.T)].mean()) - med) / sd, local_contrast=con,
                        broad_spectrum=broad, cell=bool(ok and con >= CONTRAST),
                        cell_strict=bool(ok and con >= CONTRAST_STRICT)))
    return float(tis.sum()) * um * um / 1e6, out


def fields(pixels):
    tf = pd.read_csv(R10 / "parabiosis_tissue_fields.csv")
    w = tf[(tf.partner == "WT") & ~tf.noise.astype(bool)]
    assert len(w) == 43
    rows, bodies = [], []
    for r in w.itertuples():
        a, meta = S966.read_field(r.path)
        names = list(meta["channels"])
        gi, di, fi = S934.channel_roles(names)
        m_names = [S966.MASK_ONLY if i == di else n for i, n in enumerate(names)]
        zs = [z for z, _ in S934.sampled_planes(a, meta)]
        um0 = meta["um_x"]
        for um in pixels:
            tot = [0.0, 0, 0, 0]
            for z in zs:
                mm2, x = detect([S966.to_pixel(a[c, z].astype(np.float32), um0, um) for c in range(a.shape[0])],
                                m_names, r.path, um)
                tot = [tot[0] + mm2, tot[1] + len(x), tot[2] + sum(d["cell"] for d in x),
                       tot[3] + sum(d["cell_strict"] for d in x)]
                for d in x:
                    bodies.append(dict(path=r.path, animal=r.animal, compartment=r.compartment, um_per_px=um,
                                       native_um_per_px=um0, section_z=int(z), **d))
            rows.append(dict(path=r.path, animal=r.animal, compartment=r.compartment, um_per_px=um, sections=len(zs),
                             tissue_mm2=tot[0], bodies=tot[1], cells=tot[2], cells_strict=tot[3]))
        log(f"{r.path}: at {pixels[0]} um/px {rows[-len(pixels)]['cells']} cells "
            f"({rows[-len(pixels)]['cells_strict']} strict) of {rows[-len(pixels)]['bodies']} bodies")
    return pd.DataFrame(rows), bodies


def brain(pixels):
    lec, gfp = S964.rendered(S964.BRAIN, 1), S964.rendered(S964.BRAIN, 2)
    rows, bodies = [], []
    for um in pixels:
        mm2, x = detect([lec, gfp], ["Alexa Fluor 647", "Alexa Fluor 488"], S964.BRAIN, um)
        rows.append(dict(path=S964.BRAIN, animal="31 d", compartment="brain", um_per_px=um, sections=1, tissue_mm2=mm2,
                         bodies=len(x), cells=sum(d["cell"] for d in x), cells_strict=sum(d["cell_strict"] for d in x)))
        for d in x:
            bodies.append(dict(path=S964.BRAIN, animal="31 d", compartment="brain", um_per_px=um, native_um_per_px=um,
                               section_z=0, **d))
        log(f"brain at {um} um/px: {len(x)} bodies, {rows[-1]['cells']} cells ({rows[-1]['cells_strict']} strict) in "
            f"{mm2:.2f} mm2")
    return pd.DataFrame(rows), bodies


def agreement(bd):
    lab = pd.read_csv(R10 / "strict_rule_labels.csv")
    b = bd[(bd.um_per_px.round(3) == S964.UM_MAIN) & (bd.compartment != "brain")]
    rows = []
    for r in lab.itertuples():
        c = b[(b.path == r.path) & (b.section_z == r.plane)]
        if len(c):
            um0 = float(c.native_um_per_px.iloc[0])
            d = np.hypot(c.y * S964.UM_MAIN - r.y_px * um0, c.x * S964.UM_MAIN - r.x_px * um0)
            m = c.loc[d.idxmin()] if d.min() <= MATCH_UM else None
        else:
            m = None
        rows.append(dict(item_id=r.item_id, compartment=r.compartment, truth=bool(r.truth), matched=m is not None,
                         cell=bool(m is not None and m.cell), cell_strict=bool(m is not None and m.cell_strict),
                         survey_strict=bool(r.bf_matched and r.bf_local_contrast >= 6 and r.bf_area_um2 >= 25 and
                                            not r.bf_broad_spectrum and r.bf_signal_sd >= 10),
                         survey_upgraded=bool(r.bf_matched and r.bf_cell)))
    t = pd.DataFrame(rows)
    out = []
    for call, label in [("cell", "local background, upgraded criteria (primary)"),
                        ("cell_strict", "local background, contrast >= 6"),
                        ("survey_upgraded", "survey detector, upgraded criteria (966)"),
                        ("survey_strict", "survey detector, strict rule (968)")]:
        for scope in ("all labels", "cochlea", "marrow"):
            s = t if scope == "all labels" else t[t.compartment == scope]
            k, tr = s[call], s.truth
            tp, fp, fn = int((k & tr).sum()), int((k & ~tr).sum()), int((~k & tr).sum())
            out.append(dict(call=label, scope=scope, labels=len(s), matched=int(s.matched.sum()) if call.startswith(
                "cell") else np.nan, kept=tp + fp, tp=tp, fp=fp, fn=fn, tn=int((~k & ~tr).sum()),
                precision=tp / (tp + fp) if tp + fp else np.nan, sensitivity=tp / (tp + fn) if tp + fn else np.nan))
    return pd.DataFrame(out)


def display():
    """Supplementary Fig. 5 objects (after the run; D155: the detector with the better label agreement is shown): every
    body of this detector on the display tiles of 966 (brain, and the 31-day cochlea, marrow and liver sections, same
    pixel grids) and six crops of its strict-call brain cells drawn at random (seed 969), 120 um wide."""
    from PIL import Image
    b = pd.read_csv(R10 / "localbg_bodies.csv")
    b = b[b.um_per_px.round(3) == S964.UM_MAIN]
    lg = pd.read_csv(R10 / "matched_format_display.csv").set_index("tile")
    obj, crops = [], []
    tiles = {"mf_brain_31d.png": (S964.BRAIN, 0)}
    tiles.update({f"mf_{c}_31d.png": (rel, z) for c, rel, z in S966.DISPLAY})
    for tile, (rel, z) in tiles.items():
        assert lg.loc[tile, "path"] == rel and int(lg.loc[tile, "plane"]) == z
        x = b[(b.path == rel) & (b.section_z == z)]
        assert len(x), tile
        obj += [dict(tile=tile, y=r.y, x=r.x, cell=bool(r.cell), strict=bool(r.cell_strict)) for r in x.itertuples()]
    br = b[(b.path == S964.BRAIN) & b.cell_strict].reset_index(drop=True)
    img = np.asarray(Image.open(S966.PANELS / "mf_brain_31d.png").convert("RGB"))
    h = int(round(S966.CROP_UM / S964.UM_MAIN / 2))
    allb = b[b.path == S964.BRAIN]
    for k, i in enumerate(np.sort(np.random.default_rng(969).choice(len(br), S966.N_CROPS, replace=False)), 1):
        yc, xc = int(round(br.loc[i, "y"])), int(round(br.loc[i, "x"]))
        y0, x0 = max(0, yc - h), max(0, xc - h)
        y1, x1 = min(img.shape[0], y0 + 2 * h), min(img.shape[1], x0 + 2 * h)
        y0, x0 = max(0, y1 - 2 * h), max(0, x1 - 2 * h)
        name = f"mf_brain_31d_lb_crop{k}.png"
        Image.fromarray(img[y0:y1, x0:x1]).save(S966.PANELS / name)
        crops.append(dict(tile=name, compartment="brain", path=S964.BRAIN, plane=0, um_per_px=S964.UM_MAIN,
                          height_px=y1 - y0, width_px=x1 - x0, crop_y0_y1_x0_x1=f"{y0} {y1} {x0} {x1}",
                          gfp_range=lg.loc["mf_brain_31d.png", "gfp_range"],
                          lectin_range=lg.loc["mf_brain_31d.png", "lectin_range"]))
        inside = allb[allb.y.between(y0, y1 - 1) & allb.x.between(x0, x1 - 1)]
        obj += [dict(tile=name, y=r.y - y0, x=r.x - x0, cell=bool(r.cell), strict=bool(r.cell_strict))
                for r in inside.itertuples()]
    full = lg.reset_index()
    full = pd.concat([full[~full.tile.str.contains("crop")], pd.DataFrame(crops)], ignore_index=True)
    write_csv(full, MOD, "localbg_display_tiles.csv", "Local-background detector (969, D155): Supplementary Fig. 5 "
              "display tiles (966's tiles and crops of six random strict-call brain cells)",
              required=["tile", "compartment", "um_per_px"])
    write_csv(pd.DataFrame(obj), MOD, "localbg_display_objects.csv", "Local-background detector (969, D155): its "
              "bodies on the Supplementary Fig. 5 tiles with both calls (tile pixels)", required=["tile", "y", "x",
                                                                                                 "strict"])
    log(f"display: {len(obj)} objects on {len(tiles) + len(crops)} tiles")


def main():
    if "--tiles" in sys.argv[1:]:
        display()
        return
    pix = [float(u) for u in S964.overview_pixels().index]
    assert pix[0] == S964.UM_MAIN
    f, fb = fields(pix)
    b, bb = brain(pix)
    bodies = pd.DataFrame(fb + bb)
    g = (f.groupby(["animal", "compartment", "um_per_px"])
         .agg(fields=("path", "size"), tissue_mm2=("tissue_mm2", "sum"), bodies=("bodies", "sum"),
              cells=("cells", "sum"), cells_strict=("cells_strict", "sum")).reset_index())
    dens = pd.concat([g, b.drop(columns=["path", "sections"]).assign(fields=1)], ignore_index=True)
    dens["days"] = dens.animal.str.extract(r"(\d+)").astype(int)
    dens["cell_density_per_mm2"] = dens.cells / dens.tissue_mm2
    dens["strict_density_per_mm2"] = dens.cells_strict / dens.tissue_mm2
    dens["main"] = dens.um_per_px.round(3) == S964.UM_MAIN
    dens = dens.sort_values(["um_per_px", "days", "compartment"], ascending=[False, True, True])
    ag = agreement(bodies)
    write_csv(bodies, MOD, "localbg_bodies.csv", "Local-background detector (969, D155): every GFP+ body detected on the "
              "white top-hat (15 um) of the converted sections and the brain projection, with features and both calls",
              required=["path", "compartment", "um_per_px", "section_z", "y", "x", "cell", "cell_strict"])
    write_csv(dens, MOD, "localbg_densities.csv", "Local-background detector (969, D155): partner-derived cells per mm2 "
              "per wild-type animal and tissue in the brain-image format (main 2.773 um per pixel), both calls",
              required=["animal", "compartment", "um_per_px", "cell_density_per_mm2", "strict_density_per_mm2", "main"])
    write_csv(ag, MOD, "localbg_agreement.csv", "Local-background detector (969, D155): agreement of its calls with the "
              "152 blinded labels, with the survey detector's brain-format calls alongside",
              required=["call", "scope", "precision", "sensitivity"])
    pd.set_option("display.width", 220)
    print(ag.round(3).to_string())
    print(dens[dens.main][["animal", "compartment", "tissue_mm2", "bodies", "cells", "cells_strict",
                           "cell_density_per_mm2", "strict_density_per_mm2"]].round(2).to_string())
    print(dens[dens.compartment.isin(["brain", "cochlea"]) & (dens.animal == "31 d")][
        ["compartment", "um_per_px", "cell_density_per_mm2", "strict_density_per_mm2"]].round(2).to_string())


if __name__ == "__main__":
    main()
