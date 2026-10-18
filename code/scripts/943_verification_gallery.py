"""Round 10 (D137) — blinded verification gallery of the image calls behind Fig. 4i and Supplementary Fig. 6 (user:
"Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that the
manual labeling is accurate)"; verification = "Gallery for author check (Recommended)").
The typing of 938 is re-run with its own functions (nuclei / type_cells wrapped to keep a crop of every object), and a
seeded sample is drawn (SEED):
  typed cells      up to N_TYPED per marker x compartment (all when fewer), CD3e not interpreted (excluded)
  rejected objects up to N_UNTYPED marker objects per marker that failed the size / nucleus rule (negative controls)
  GFP calls        every typed cell called partner-derived (most of the cell inside a GFP+ body) and N_CONTACT typed
                   cells that only touch a GFP+ body (the hard cases of the majority rule)
  60x lateral wall the 13 FUT4+ cells of 931 (outline = equivalent circle; 931 keeps no object masks)
  fields           every field whose tissue was assigned by morphology or excluded (934 MORPH, NO_MARROW)
The reviewer sees no call, compartment or field name: per cell 'one nucleated marker+ cell?' and 'GFP+ (most of the
outlined cell carries GFP)?', per field 'which tissue?'. Display: marker and GFP with one range per series taken from
its marrow fields (0.5th-99.9th percentile; as Supplementary Fig. 6), DAPI per field. The answer key stays here.
Outputs: results/r10_parabiosis/verification_gallery_items.csv (lib/io.write_csv; key with every call) and
figures/gallery/gallery_data.js + item JPEGs folded into it (review page assets).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/943_verification_gallery.py
"""
import base64
import importlib.util
import io as _io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S934 = load_module("s934", "934_parabiosis_survey.py")
S938 = load_module("s938", "938_parabiosis_cell_types.py")
MOD = "r10_parabiosis"
OUT = ROOT / "figures" / "gallery"
SEED, N_TYPED, N_UNTYPED, N_CONTACT = 943, 20, 8, 15
CROP_UM, PANEL_PX, BAR_UM = 36.0, 150, 10.0
STORE, STATS, LAST = [], {}, {}
_nuclei, _type_cells = S938.nuclei, S938.type_cells


def pct(v):
    v = np.asarray(v, float)
    return (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.9))) if v.size else (0.0, 1.0)


def nuclei_kept(dapi, tissue, um):
    LAST["dapi"] = dapi
    return _nuclei(dapi, tissue, um)


def type_cells_kept(marker, nuc, body, tissue, um, thr, island=None, gfp=None):
    """938's type_cells, keeping per object a crop (DAPI, marker, GFP, object mask) and the plane's channel ranges."""
    rows = _type_cells(marker, nuc, body, tissue, um, thr, island, gfp)
    fs = S938.smooth(marker, um, 0.3)
    lab = measure.label(ndi.binary_fill_holes(fs > thr) & tissue)
    props = measure.regionprops(lab)
    cen = np.array([r.centroid for r in props]) if props else np.zeros((0, 2))
    key = len(STATS)
    STATS[key] = dict(dapi=pct(LAST["dapi"][tissue]), marker=pct(marker[tissue]), gfp=pct(gfp[tissue]))
    h = int(round(CROP_UM / um / 2))
    H, W = marker.shape
    for r in rows:
        j = int(np.argmin(np.hypot(cen[:, 0] - r["y"], cen[:, 1] - r["x"])))
        assert np.hypot(cen[j, 0] - r["y"], cen[j, 1] - r["x"]) < 0.5, (r["y"], r["x"], cen[j])
        lb = props[j].label
        y, x = int(round(r["y"])), int(round(r["x"]))
        y0, y1, x0, x1 = max(0, y - h), min(H, y + h), max(0, x - h), min(W, x + h)
        r["_crop"] = len(STORE)
        r["_stats"] = key
        STORE.append(dict(dapi=LAST["dapi"][y0:y1, x0:x1].astype(np.float32),
                          marker=marker[y0:y1, x0:x1].astype(np.float32),
                          gfp=gfp[y0:y1, x0:x1].astype(np.float32), obj=lab[y0:y1, x0:x1] == lb,
                          pad=(y - h - y0, x - h - x0), um=um))
    return rows


def scale(v, rng):
    lo, hi = rng
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)


def panelize(c, rng_m, rng_g, rng_d):
    """Three panels (merge; marker with the object outline; GFP with the object outline), each PANEL_PX square."""
    side = int(round(CROP_UM / c["um"]))
    # crops near the image border are placed so the cell stays centred
    m, g, d = scale(c["marker"], rng_m), scale(c["gfp"], rng_g), scale(c["dapi"], rng_d)
    oy, ox = max(0, -c["pad"][0]), max(0, -c["pad"][1])
    big = lambda a: _place(a, side, oy, ox)  # noqa: E731
    m, g, d, ob = big(m), big(g), big(d), big(c["obj"].astype(np.float32)) > 0.5
    merge = np.dstack([m, g, np.clip(m + d * 0.8, 0, 1)])
    edge = segmentation.find_boundaries(ob, mode="outer")
    mk = np.dstack([m] * 3)
    gf = np.dstack([g] * 3)
    for im in (mk, gf):
        im[edge] = (1.0, 0.85, 0.1)
    tiles = [Image.fromarray((x * 255).astype(np.uint8)).resize((PANEL_PX, PANEL_PX), Image.NEAREST)
             for x in (merge, mk, gf)]
    sheet = Image.new("RGB", (3 * PANEL_PX + 8, PANEL_PX), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (PANEL_PX + 4), 0))
    dr = ImageDraw.Draw(sheet)
    bl = BAR_UM / CROP_UM * PANEL_PX
    dr.rectangle([6, PANEL_PX - 9, 6 + bl, PANEL_PX - 6], fill=(255, 255, 255))
    return sheet


def _place(a, side, oy, ox):
    out = np.zeros((side, side), np.float32)
    hh, ww = min(a.shape[0], side - oy), min(a.shape[1], side - ox)
    out[oy:oy + hh, ox:ox + ww] = a[:hh, :ww]
    return out


def jpeg(img, q=93):
    b = _io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def lw60():
    """The 13 FUT4+ cells of the 60x lateral-wall stack (931; maximum projection, equivalent-circle outline)."""
    d = ROOT / "data" / "photos" / "parabiosis_54d" / "cell-type-bone-cochlea" / "parabiotic54d-gfp-647fut4-60x-sl-2"
    ch = {c: np.stack([tifffile.imread(p) for p in sorted(d.glob(f"1_1_{c}_Z*.tif"))]).max(0).astype(np.float32)
          for c in ("C1", "C2", "C4")}
    um = float(S938.sidecar(d / "1_1_Info.txt"))
    ob = pd.read_csv(RESULTS / "r9_figures" / "f4_parabiont_objects.csv")
    ob = ob[(ob.field == "60x_sl2") & (ob.object == "FUT4+ cell")]
    assert len(ob) == 13
    yy, xx = np.mgrid[:ch["C1"].shape[0], :ch["C1"].shape[1]]
    rng = dict(dapi=pct(ch["C1"]), marker=pct(ch["C4"]), gfp=pct(ch["C2"]))
    out = []
    h = int(round(CROP_UM / um / 2))
    H, W = ch["C1"].shape
    for r in ob.itertuples():
        y, x = int(round(r.centroid_y_px)), int(round(r.centroid_x_px))
        rad = np.sqrt(r.area_um2 / np.pi) / um
        y0, y1, x0, x1 = max(0, y - h), min(H, y + h), max(0, x - h), min(W, x + h)
        disc = (yy - r.centroid_y_px) ** 2 + (xx - r.centroid_x_px) ** 2 <= rad * rad
        c = dict(dapi=ch["C1"][y0:y1, x0:x1], marker=ch["C4"][y0:y1, x0:x1], gfp=ch["C2"][y0:y1, x0:x1],
                 obj=disc[y0:y1, x0:x1], pad=(y - h - y0, x - h - x0), um=um)
        out.append((r, c))
    return out, rng


def field_thumb(rel):
    a, meta = S934.read_czi(S934.SRC / rel)
    um = meta["um_x"]
    gi, di, fi = S934.channel_roles(meta["channels"])
    best = max(S934.sampled_planes(a, meta), key=lambda t: t[1].sum(), default=(0, None))[0]
    chans = [fi, gi, di]
    rgb = []
    for c in chans:
        v = a[c, best].astype(np.float32) if c is not None else np.zeros(a.shape[2:], np.float32)
        rgb.append(scale(v, pct(v)))
    m, g, d = rgb
    im = np.dstack([np.clip(m, 0, 1), np.clip(g, 0, 1), np.clip(m + 0.8 * d, 0, 1)])
    img = Image.fromarray((im * 255).astype(np.uint8))
    w = 520
    img = img.resize((w, int(round(img.size[1] * w / img.size[0]))), Image.LANCZOS)
    bar = 50.0 / (um * a.shape[3] / w)
    dr = ImageDraw.Draw(img)
    dr.rectangle([10, img.size[1] - 14, 10 + bar, img.size[1] - 10], fill=(255, 255, 255))
    return img, dict(field_um=um * a.shape[3], plane=int(best))


def main():
    S938.nuclei, S938.type_cells = nuclei_kept, type_cells_kept
    acq = pd.read_csv(RESULTS / MOD / "parabiosis_acquisitions.csv")
    zc, _ = S938.zeiss(acq)
    oc, _ = S938.olympus()
    cells = pd.DataFrame(zc + oc)
    pub = pd.read_csv(RESULTS / MOD / "parabiosis_celltype_cells.csv")
    keys = ["source", "series", "field", "plane", "marker"]
    for d_ in (cells, pub):
        d_["yk"], d_["xk"] = d_.y.round(2), d_.x.round(2)
    m = cells.drop(columns=["included"]).merge(pub[keys + ["yk", "xk", "included", "gfp_contact", "gfp_majority"]],
                                               on=keys + ["yk", "xk"], how="inner")
    assert len(m) == len(pub) == len(cells), (len(m), len(pub), len(cells))
    log(f"re-run reproduces all {len(pub)} objects of 938")
    rng = np.random.default_rng(SEED)
    ok = m[m.included & (m.marker != "CD3e")]
    picks = []

    def take(d, n, stratum):
        idx = d.index.to_numpy()
        sel = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
        picks.extend((i, stratum) for i in sel)
    for (mk, cp), d in ok[ok.typed].groupby(["marker", "compartment"]):
        take(d, N_TYPED, f"typed {mk} {cp}")
    for mk, d in ok[~ok.typed].groupby("marker"):
        take(d, N_UNTYPED, f"rejected {mk}")
    taken = {i for i, _ in picks}
    gp = ok[ok.typed & ok.gfp_majority & ~ok.index.isin(taken)]
    picks.extend((i, "GFP+ call") for i in gp.index)
    take(ok[ok.typed & ok.gfp_contact & ~ok.gfp_majority & ~ok.index.isin(taken)], N_CONTACT, "GFP contact only")
    # series display ranges from the marrow fields
    st = pd.DataFrame([dict(_stats=k, **{f"{c}_{j}": v[c][j] for c in ("dapi", "marker", "gfp") for j in (0, 1)})
                       for k, v in STATS.items()])
    mm = m.merge(st, on="_stats")
    srng = {}
    for (se, mk), d in mm[mm.compartment == "marrow"].groupby(["series", "marker"]):
        srng[(se, mk)] = dict(marker=(d.marker_0.median(), d.marker_1.max()), gfp=(d.gfp_0.median(), d.gfp_1.max()))
    items, data = [], []
    codes = rng.permutation(np.arange(1000, 10000))
    for n_, (i, stratum) in enumerate(picks):
        r = m.loc[i]
        c = STORE[int(r._crop)]
        fr = STATS[int(r._stats)]
        sr = srng.get((r.series, r.marker), dict(marker=fr["marker"], gfp=fr["gfp"]))
        img = panelize(c, sr["marker"], sr["gfp"], fr["dapi"])
        iid = f"C{codes[n_]}"
        data.append(dict(id=iid, kind="cell", marker=r.marker, img=jpeg(img)))
        items.append(dict(item_id=iid, kind="cell", stratum=stratum, source=r.source, series=r.series, field=r.field,
                          plane=int(r.plane), marker=r.marker, compartment=r.compartment, y_px=r.y, x_px=r.x,
                          um_per_px=r.um_per_px, area_um2=r.area_um2, nucleus_frac=r.nucleus_frac, typed=bool(r.typed),
                          gfp_overlap=r.gfp_overlap, gfp_contact=bool(r.gfp_contact),
                          gfp_majority=bool(r.gfp_majority), outline="segmented object",
                          marker_range=f"{sr['marker'][0]:.1f}-{sr['marker'][1]:.1f}",
                          gfp_range=f"{sr['gfp'][0]:.1f}-{sr['gfp'][1]:.1f}", seed=SEED))
    lw, lrng = lw60()
    k0 = len(items)
    for j, (r, c) in enumerate(lw):
        iid = f"C{codes[k0 + j]}"
        data.append(dict(id=iid, kind="cell", marker="FUT4", img=jpeg(panelize(c, lrng["marker"], lrng["gfp"],
                                                                                  lrng["dapi"]))))
        items.append(dict(item_id=iid, kind="cell", stratum="typed FUT4 cochlea (60x lateral wall, 931)",
                          source="Olympus 60x (931)", series="Olympus FUT4 60x", field="60x_sl2", plane=-1,
                          marker="FUT4", compartment="cochlea", y_px=r.centroid_y_px, x_px=r.centroid_x_px,
                          um_per_px=c["um"], area_um2=r.area_um2, nucleus_frac=np.nan, typed=True,
                          gfp_overlap=np.nan, gfp_contact=bool(r.gfp_positive), gfp_majority=bool(r.gfp_positive),
                          outline="equivalent circle (931 keeps no masks)",
                          marker_range=f"{lrng['marker'][0]:.1f}-{lrng['marker'][1]:.1f}",
                          gfp_range=f"{lrng['gfp'][0]:.1f}-{lrng['gfp'][1]:.1f}", seed=SEED))
    k0 = len(items)
    flds = list(S934.MORPH) + list(S934.NO_MARROW)
    for j, rel in enumerate(flds):
        img, info = field_thumb(rel)
        iid = f"F{codes[k0 + j]}"
        if rel in S934.NO_MARROW:
            call, basis = "excluded", S934.NO_MARROW[rel]
        else:
            call, _, basis = S934.MORPH[rel]
        data.append(dict(id=iid, kind="field", marker="", img=jpeg(img, 90)))
        items.append(dict(item_id=iid, kind="field", stratum="field assignment", source="ZEISS", series="",
                          field=rel, plane=info["plane"], marker="", compartment=call, y_px=np.nan, x_px=np.nan,
                          um_per_px=np.nan, area_um2=np.nan, nucleus_frac=np.nan, typed=False, gfp_overlap=np.nan,
                          gfp_contact=False, gfp_majority=False, outline=basis, marker_range="", gfp_range="",
                          seed=SEED))
    order = rng.permutation(len(data))
    data = [data[i] for i in order]
    OUT.mkdir(parents=True, exist_ok=True)
    js = "window.GALLERY = " + json.dumps(dict(seed=SEED, crop_um=CROP_UM, bar_um=BAR_UM, items=data)) + ";\n"
    (OUT / "gallery_data.js").write_text(js, encoding="utf-8", newline="\n")
    log(f"gallery: {len(data)} items ({sum(d['kind'] == 'cell' for d in data)} cells, "
        f"{sum(d['kind'] == 'field' for d in data)} fields); data {len(js) / 1e6:.1f} MB")
    key = pd.DataFrame(items)
    log(key.groupby("stratum").size().to_string())
    write_csv(key, MOD, "verification_gallery_items.csv", "Verification gallery (943, D137): answer key of the "
              "blinded author review - per item the sampling stratum, field, plane, position, every algorithm call "
              "(typed, GFP contact / majority, tissue assignment) and display ranges (seeded sample)",
              required=["item_id", "kind", "stratum", "typed", "gfp_majority"])


if __name__ == "__main__":
    main()
