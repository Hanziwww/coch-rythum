"""Round 10 (D152) — every tissue counted in the brain image's format, so that the cochlear and brain counts compare
directly. User: "Improve the counting method; I would like the number of cochleae to be comparable to the number of
brains." The brain exists only as a rendered maximum projection without DAPI and without a pixel size (964), so the
other tissues are brought to that format rather than the reverse.

Fields: the wild-type survey fields of Supplementary Fig. 2c (marrow, cochlea, vestibular bone and liver of the 5-, 7-,
14-, 31- and 54-day partners; the noise-exposed pair excluded), read as in 934 / 963 (CZI; the liver from its rendered
20X exports). Each field is converted to the brain's format and counted by 960's body_features exactly as the brain was
(964): tissue mask of all channels (934; DAPI enters the mask only, renamed so that no nucleus criterion applies), GFP+
bodies (tissue median + 10 robust SD, >= 15 um2, solidity >= 0.7) and the upgraded rule without the nucleus criterion
(area >= 25 um2, mean GFP >= 2x the median GFP of the tissue 2-6 um beyond the body, not broad-spectrum against lectin
in lectin fields). Two conversions:
  projection  the maximum over every optical section of the acquisition (the brain image is an orthogonal maximum
              projection), one count per field;
  section     each analysed section of 934 (>= 10 um apart, >= 5% tissue) separately, summed per field (independent of
              stack depth; the native survey samples sections the same way).
Each channel is area-averaged (box filter) to the target pixel size and rounded to 8-bit levels. Target pixel sizes:
2.773 um (main; the collection's dominant overview setting, at which the brain is counted in 964) and the other overview
sizes of the collection (1.387, 1.248, 1.040 um), each paired with the brain counted at the same assumed size. The
brain's projection depth is unknown (its source acquisition is not in the collection); stack depths of the other fields
are reported (projection density rises with depth, so a deeper brain stack would raise, not lower, its count).
Outputs (lib/io.write_csv, results/r10_parabiosis/): matched_format_fields.csv (field x conversion x pixel size: depth,
tissue area, GFP+ bodies, cells, density), matched_format_bodies.csv (every converted GFP+ body with its features and
the image's GFP median and robust SD, for the strict rule of 968) and matched_format_summary.csv (wild-type animal x tissue x conversion x pixel
size, with the brain rows of 964 as the projection conversion).
Display tiles for Supplementary Fig. 5 (user: "Relevant photographic evidence and similar materials can be added to the
supplementary figures", D152): the brain projection and the 31-day cochlea, marrow and liver fields shown in Supplementary
Fig. 2a (same sections), converted at 2.773 um per pixel, GFP green and lectin magenta (no DAPI, as the brain), each
channel linear from its tissue median to its 99.9th percentile within tissue (GFP: or to the body threshold if higher);
crops of six counted brain cells drawn at random (seed 966; once 968 has run, among the cells of its strict rule),
120 um wide; every body carries both calls (upgraded rule; strict rule of 968, D153). Files: figures/panels/mf_*.png; logs
matched_format_display.csv (tile, field, section, pixel size, display ranges, crop window) and
matched_format_display_objects.csv (every GFP+ body of each tile with its call, in tile pixels).
Run (local; the images are not on the server): COCH_ROOT=D:/coch-rythum python scripts/966_matched_format_counts.py
[--tiles-only: the display tiles and their logs only]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage import filters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S964 = load_module("s964", "964_tissue_comparison.py")
S960, S934 = S964.S960, S964.S934
MOD = "r10_parabiosis"
MASK_ONLY = "nuclear stain (tissue mask only)"
PANELS = ROOT / "figures" / "panels"
DISPLAY = [("cochlea", "2025-01-15-2/20X-0_45X-31d-GFP-lectin-dapi-LUOXUANBAN-01.czi", 40),     # Supplementary Fig. 2a
           ("marrow", "2025-01-15-2/20X-0_45X-31d-lectin--GFP-BONE-02.czi", 40),
           ("liver", "2025-01-15-2/TIFF-25.1.15/WT-31D-LIVER-20X-2_4X-02", 0)]
CROP_UM, N_CROPS, SEED = 120.0, 6, 966


def read_field(rel):
    if rel.lower().endswith(".czi"):
        return S934.read_czi(S934.SRC / rel)
    stem = rel.rsplit("/", 1)[-1]                       # rendered liver export: c1 lectin, c2 GFP, c3 DAPI (934, 963)
    d = S934.SRC / "2025-01-15-2" / "TIFF-25.1.15" / stem
    chs = [tifffile.imread(d / f"{stem}_c{k}.tif").astype(np.float32).max(axis=-1) for k in (1, 2, 3)]
    meta = dict(um_x=S934.LIVER_UM, um_z=None, channels=["Alexa Fluor 647", "Alexa Fluor 488", "DAPI"])
    return np.stack(chs)[:, None], meta


def to_pixel(img, um_from, um_to):
    f = um_to / um_from
    h, w = img.shape
    size = (max(1, int(round(w / f))), max(1, int(round(h / f))))
    out = Image.fromarray(np.ascontiguousarray(img, dtype=np.float32), mode="F").resize(size, Image.BOX)
    return np.round(np.asarray(out, dtype=np.float32))


def count(chs, names, rel, um, keep=None, **ids):
    """960's body_features on one converted image (channels C x H x W at um per pixel) -> tissue mm2, bodies, cells;
    with keep (a list), every body is appended with its features, the image's GFP tissue median and robust SD (the
    detection statistics of 934, used by the strict rule of 968) and the identifiers in ids."""
    a = np.stack(chs)[:, None]
    meta = dict(um_x=um, um_z=None, channels=names)
    planes = list(S934.sampled_planes(a, meta))
    if not planes:                                       # < 5% tissue after conversion: no count
        return 0.0, 0, 0
    tis = planes[0][1]
    mm2 = float(tis.sum()) * um * um / 1e6
    x = S960.body_features(a, meta, rel, {}, 255.0)
    if keep is not None:
        gi = S934.channel_roles(names)[0]
        med, sd = S934.robust(filters.gaussian(chs[gi].astype(np.float32), sigma=0.5 / um, preserve_range=True)[tis])
        for d in x:
            d.pop("_crop", None)
            keep.append(dict(**ids, **d, plane_median=med, plane_sd=sd))
    return mm2, len(x), int(sum(d["cell"] for d in x))


def field_rows(r, pixels, bodies):
    a, meta = read_field(r.path)
    names = list(meta["channels"])
    gi, di, fi = S934.channel_roles(names)
    m_names = [MASK_ONLY if i == di else n for i, n in enumerate(names)]
    g2, d2, f2 = S934.channel_roles(m_names)
    assert g2 == gi and d2 is None and (fi is None or f2 == fi or "lectin" not in r.path.lower()), r.path
    zs = [z for z, _ in S934.sampled_planes(a, meta)]
    depth = (a.shape[1] - 1) * (meta["um_z"] or 0.0)
    um0 = meta["um_x"]
    mip = a.max(axis=1).astype(np.float32)
    rows = []
    for um in pixels:
        mm2, nb, nc = count([to_pixel(mip[c], um0, um) for c in range(a.shape[0])], m_names, r.path, um, bodies,
                            animal=r.animal, compartment=r.compartment, conversion="projection", um_per_px=um,
                            native_um_per_px=um0, section_z=-1)
        rows.append(dict(path=r.path, animal=r.animal, compartment=r.compartment, conversion="projection",
                         um_per_px=um, native_um_per_px=um0, n_z=a.shape[1], depth_um=depth, sections=1,
                         tissue_mm2=mm2, gfp_bodies=nb, gfp_cells=nc))
        tot = [0.0, 0, 0]
        for z in zs:
            mm2, nb, nc = count([to_pixel(a[c, z].astype(np.float32), um0, um) for c in range(a.shape[0])], m_names,
                                r.path, um, bodies, animal=r.animal, compartment=r.compartment,
                                conversion="section", um_per_px=um, native_um_per_px=um0, section_z=int(z))
            tot = [tot[0] + mm2, tot[1] + nb, tot[2] + nc]
        rows.append(dict(path=r.path, animal=r.animal, compartment=r.compartment, conversion="section",
                         um_per_px=um, native_um_per_px=um0, n_z=a.shape[1], depth_um=depth, sections=len(zs),
                         tissue_mm2=tot[0], gfp_bodies=tot[1], gfp_cells=tot[2]))
    return rows


def rgb(gfp, far, tis, thr=0.0):
    """GFP green, lectin magenta; each linear from its tissue median to its 99.9th percentile within tissue (GFP: or to
    the body threshold if higher)."""
    g_lo, g_hi = float(np.median(gfp[tis])), max(float(np.percentile(gfp[tis], 99.9)), thr)
    f_lo, f_hi = float(np.median(far[tis])), float(np.percentile(far[tis], 99.9))
    g = np.clip((gfp - g_lo) / max(g_hi - g_lo, 1e-6), 0, 1)
    f = np.clip((far - f_lo) / max(f_hi - f_lo, 1e-6), 0, 1)
    return (np.stack([f, g, f], -1) * 255).astype(np.uint8), (g_lo, g_hi, f_lo, f_hi)


def strict_rule():
    """The brain-format strict rule chosen by 968 (D153), or None before 968 has run."""
    f = RESULTS / MOD / "strict_rule_choice.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    r = c[(c["format"] == "brain image") & (c["rule"] == "strict (chosen)") & (c["scope"] == "all labels")].iloc[0]
    return dict(signal_sd=float(r.thr_signal_sd), contrast=float(r.thr_contrast), area=float(r.thr_area))


def objects(x, tile, sd, rule):
    rows = []
    for d in x:
        sig = d["signal"] / sd
        strict = (None if rule is None else bool(sig >= rule["signal_sd"] and d["local_contrast"] >= rule["contrast"]
                                                  and d["area_um2"] >= rule["area"] and not d["broad_spectrum"]))
        rows.append(dict(tile=tile, y=d["y"], x=d["x"], area_um2=d["area_um2"], local_contrast=d["local_contrast"],
                         signal_sd=sig, broad_spectrum=bool(d["broad_spectrum"]), cell=bool(d["cell"]), strict=strict))
    return rows


def gfp_sd(gfp, tis, um):
    return S934.robust(filters.gaussian(gfp.astype(np.float32), sigma=0.5 / um, preserve_range=True)[tis])[1]


def display_tiles():
    """Supplementary Fig. 5 tiles (D152): brain projection and six random counted brain cells; the 31-day cochlea,
    marrow and liver sections of Supplementary Fig. 2a converted to the brain's format, with every GFP+ body."""
    PANELS.mkdir(parents=True, exist_ok=True)
    um = S964.UM_MAIN
    log_rows, obj_rows = [], []
    lec, gfp = S964.rendered(S964.BRAIN, 1), S964.rendered(S964.BRAIN, 2)
    tis = S934.tissue_mask([gfp, lec], um, [max(float(np.percentile(c, 99.5)), 1.0) for c in (gfp, lec)])
    _, st, _, _ = S934.gfp_bodies(gfp, lec, tis, um)
    img, rng = rgb(gfp, lec, tis, st["threshold"])
    Image.fromarray(img).save(PANELS / "mf_brain_31d.png")
    b = pd.read_csv(RESULTS / MOD / "tissue_comparison_brain_bodies.csv")
    b = b[b["main"]].reset_index(drop=True)
    log_rows.append(dict(tile="mf_brain_31d.png", compartment="brain", path=S964.BRAIN, plane=0, um_per_px=um,
                         height_px=img.shape[0], width_px=img.shape[1], crop_y0_y1_x0_x1="",
                         gfp_range=f"{rng[0]:.1f}-{rng[1]:.1f}", lectin_range=f"{rng[2]:.1f}-{rng[3]:.1f}"))
    rule = strict_rule()
    sd_b = gfp_sd(gfp, tis, um)
    obj_rows += objects(b.to_dict("records"), "mf_brain_31d.png", sd_b, rule)
    b = b.assign(strict=[o["strict"] for o in obj_rows])
    cells = b[b["strict"] if rule is not None else b["cell"]].reset_index(drop=True)      # crops among counted cells
    pick = np.sort(np.random.default_rng(SEED).choice(len(cells), N_CROPS, replace=False))
    h = int(round(CROP_UM / um / 2))
    for k, i in enumerate(pick, 1):
        yc, xc = int(round(cells.loc[i, "y"])), int(round(cells.loc[i, "x"]))
        y0, x0 = max(0, yc - h), max(0, xc - h)
        y1, x1 = min(img.shape[0], y0 + 2 * h), min(img.shape[1], x0 + 2 * h)
        y0, x0 = max(0, y1 - 2 * h), max(0, x1 - 2 * h)
        name = f"mf_brain_31d_crop{k}.png"
        Image.fromarray(img[y0:y1, x0:x1]).save(PANELS / name)
        log_rows.append(dict(tile=name, compartment="brain", path=S964.BRAIN, plane=0, um_per_px=um, height_px=y1 - y0,
                             width_px=x1 - x0, crop_y0_y1_x0_x1=f"{y0} {y1} {x0} {x1}",
                             gfp_range=f"{rng[0]:.1f}-{rng[1]:.1f}", lectin_range=f"{rng[2]:.1f}-{rng[3]:.1f}"))
        inside = b[b.y.between(y0, y1 - 1) & b.x.between(x0, x1 - 1)].assign(y=lambda d_: d_.y - y0,
                                                                                 x=lambda d_: d_.x - x0)
        obj_rows += objects(inside.to_dict("records"), name, sd_b, rule)
    for comp, rel, z in DISPLAY:
        a, meta = read_field(rel)
        names = list(meta["channels"])
        gi, di, fi = S934.channel_roles(names)
        m_names = [MASK_ONLY if i == di else n for i, n in enumerate(names)]
        chs = [to_pixel(a[c, z].astype(np.float32), meta["um_x"], um) for c in range(a.shape[0])]
        aa = np.stack(chs)[:, None]
        m2 = dict(um_x=um, um_z=None, channels=m_names)
        tis = next(S934.sampled_planes(aa, m2))[1]
        _, st, _, _ = S934.gfp_bodies(chs[gi], chs[fi], tis, um)
        x = S960.body_features(aa, m2, rel, {}, 255.0)
        img, rng = rgb(chs[gi], chs[fi], tis, st["threshold"])
        name = f"mf_{comp}_31d.png"
        Image.fromarray(img).save(PANELS / name)
        log_rows.append(dict(tile=name, compartment=comp, path=rel, plane=z, um_per_px=um, height_px=img.shape[0],
                             width_px=img.shape[1], crop_y0_y1_x0_x1="", gfp_range=f"{rng[0]:.1f}-{rng[1]:.1f}",
                             lectin_range=f"{rng[2]:.1f}-{rng[3]:.1f}"))
        obj_rows += objects(x, name, gfp_sd(chs[gi], tis, um), rule)
        log(f"tile {name}: {img.shape[1]} x {img.shape[0]} px, {len(x)} bodies, {sum(d['cell'] for d in x)} cells")
    write_csv(pd.DataFrame(log_rows), MOD, "matched_format_display.csv", "Brain-format display tiles of Supplementary "
              "Fig. 5 (966, D152): field, section, pixel size, display ranges and crop windows (figures/panels/mf_*.png)",
              required=["tile", "compartment", "um_per_px", "gfp_range"])
    write_csv(pd.DataFrame(obj_rows), MOD, "matched_format_display_objects.csv", "Brain-format display tiles of "
              "Supplementary Fig. 5 (966, D152): every GFP+ body in each tile with its call (tile pixels)",
              required=["tile", "y", "x", "cell"])


def main():
    if "--tiles-only" in sys.argv[1:]:
        display_tiles()
        return
    pix = [float(u) for u in S964.overview_pixels().index]
    assert pix[0] == S964.UM_MAIN
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    w = tf[(tf.partner == "WT") & ~tf.noise.astype(bool)]
    assert len(w) == 43, len(w)
    rows, bodies = [], []
    for r in w.itertuples():
        x = field_rows(r, pix, bodies)
        rows += x
        m = x[0]
        log(f"{r.path}: {m['depth_um']:.0f} um deep; at {pix[0]} um/px projection {x[0]['gfp_cells']} cells in "
            f"{x[0]['tissue_mm2']:.3f} mm2, sections {x[1]['gfp_cells']} in {x[1]['tissue_mm2']:.3f} mm2")
    f = pd.DataFrame(rows)
    f["cell_density_per_mm2"] = f.gfp_cells / f.tissue_mm2.where(f.tissue_mm2 > 0)
    s = (f.groupby(["animal", "compartment", "conversion", "um_per_px"])
         .agg(fields=("path", "size"), tissue_mm2=("tissue_mm2", "sum"), gfp_bodies=("gfp_bodies", "sum"),
              gfp_cells=("gfp_cells", "sum"), depth_um_max=("depth_um", "max")).reset_index())
    br = pd.read_csv(RESULTS / MOD / "tissue_comparison_density.csv")
    br = br[br.compartment == "brain"]
    assert set(br.um_per_px.round(3)) == {round(u, 3) for u in pix}
    s = pd.concat([s, br.assign(conversion="projection", depth_um_max=np.nan)[s.columns]], ignore_index=True)
    s["days"] = s.animal.str.extract(r"(\d+)").astype(int)
    s["cell_density_per_mm2"] = s.gfp_cells / s.tissue_mm2
    s["main"] = s.um_per_px.round(3) == S964.UM_MAIN
    coch = (s[(s.animal == "31 d") & (s.compartment == "cochlea")].set_index(["conversion", "um_per_px"])
            ["cell_density_per_mm2"])
    s["relative_to_cochlea_31d"] = [r.cell_density_per_mm2 / coch.get((r.conversion, r.um_per_px), np.nan)
                                    for r in s.itertuples()]
    s = s.sort_values(["conversion", "um_per_px", "days", "compartment"], ascending=[True, False, True, True])
    write_csv(f, MOD, "matched_format_fields.csv", "Brain-format counts (966, D152): every wild-type survey field "
              "converted to the brain image's format (maximum projection of the acquisition, or each analysed section) "
              "at the collection's overview pixel sizes and counted by the upgraded rule without the nucleus "
              "criterion, as the brain", required=["path", "conversion", "um_per_px", "tissue_mm2", "gfp_cells"])
    bd = pd.DataFrame(bodies)
    assert int(bd.groupby(["conversion", "um_per_px"]).size().sum()) == int(f.gfp_bodies.sum())
    write_csv(bd, MOD, "matched_format_bodies.csv", "Brain-format counts (966, D152): every GFP+ body of the converted "
              "fields (both conversions, every overview pixel size) with 960's features, the call without the nucleus "
              "criterion and the image's GFP tissue median and robust SD (for the strict rule, 968)",
              required=["path", "conversion", "um_per_px", "section_z", "y", "x", "area_um2", "signal", "local_contrast",
                        "broad_spectrum", "cell", "plane_sd"])
    write_csv(s, MOD, "matched_format_summary.csv", "Brain-format counts (966, D152): partner-derived cells per mm2 per "
              "wild-type animal and tissue in the brain image's format (projection and section conversions; main = "
              "2.773 um per pixel), with the brain (964) as a projection",
              required=["animal", "compartment", "conversion", "um_per_px", "cell_density_per_mm2", "main"])
    m = s[s.main & (s.animal == "31 d")]
    for cv_, g in m.groupby("conversion"):
        log(f"31 d, {cv_}, 2.773 um/px: " + ", ".join(f"{r.compartment} {r.cell_density_per_mm2:.1f}"
                                                      for r in g.itertuples()))
    display_tiles()


if __name__ == "__main__":
    main()
