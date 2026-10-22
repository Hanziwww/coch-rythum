"""Round 10 (D141) — immune cell typing of the 54-day wild-type parabiont anchored on segmented nuclei, replacing the
marker-object typing of 938 whose objects were single cells in only about half of the cases (944; user: "Currently, the
amount of information in our graph—as well as the number of cell types included—has actually decreased.").

Fields, series, thresholds, separability checks, islands and exclusions are those of 938 (its loaders are reused; only
the per-cell call changes). Rules (fixed before any count was inspected):
  nuclei   DAPI nucleus mask of 938 (sigma 0.5 um, Otsu inside tissue) split by a distance-transform watershed (seeds
           >= 2.5 um apart); nuclei of 10-150 um2 not touching the image border.
  cell     the nucleus expanded by 1.5 um inside tissue (931 rule of the fluid cells); ring = cell minus nucleus.
  marker+  at least half of the ring above the series marker threshold of 938 (Otsu of the marker in the series'
           marrow tissue, sigma 0.3 um): a positive cytoplasm or membrane surrounds the nucleus. Ring coverage rather
           than the ring mean, so that a positive neighbour's membrane alone does not make a cell positive.
  GFP+     at least half of the cell above the field's GFP threshold of 934 (tissue median + 10 robust SD, sigma 0.5 um):
           the cell itself carries reporter-level GFP, not a neighbouring body or a speck.
CD3e is not interpreted (938); the 13 FUT4+ cells of the 60x lateral wall (931) are carried over as in 938.
Stage 1 (default): typing, per-type summary, a seeded blinded verification sample (SEED; per marker x compartment up to
N_POS marker+ cells, up to N_NEG marker- cells, and every GFP+ marker+ cell) with ID-only contact sheets under
figures/gallery/nuclei/ and the answer key. Stage 2 (--unblind): reads data/verification/manual_labels_945.csv (labelled
blinded by the analyst, criteria as 943/944) and writes the agreement per stratum and the verified summary (typed cells
x precision; GFP calls of confirmed cells).
Outputs (lib/io.write_csv, results/r10_parabiosis/): celltype_nuclei_cells.csv, celltype_nuclei_summary.csv,
verification_nuclei_items.csv; with --unblind: verification_nuclei_agreement.csv, celltype_nuclei_verified.csv.
Run: COCH_ROOT=/path/to/coch-rythum python scripts/945_celltype_nuclei.py [--unblind]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.stats import beta, norm
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels, watershed

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
OUT = ROOT / "figures" / "gallery" / "nuclei"
LABELS = ROOT / "data" / "verification" / "manual_labels_945.csv"
NUC_AREA, EXPAND_UM, SEED_SEP_UM, RING_FRAC, GFP_FRAC = (10.0, 150.0), 1.5, 2.5, 0.5, 0.5
SEED, N_POS, N_NEG = 945, 15, 6
CROP_UM, PANEL_PX = 30.0, 150
STORE, STATS, LAST = {}, {}, {}
_nuclei = S938.nuclei


def nuclei_kept(dapi, tissue, um):
    LAST["dapi"] = dapi
    return _nuclei(dapi, tissue, um)


def split_nuclei(nuc, um):
    dist = ndi.distance_transform_edt(nuc)
    lab0 = measure.label(nuc)
    pk = peak_local_max(dist, min_distance=max(1, int(round(SEED_SEP_UM / um))), labels=lab0, exclude_border=False)
    mk = np.zeros(nuc.shape, np.int32)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, mk, mask=nuc)
    return lab


def type_cells_nuclei(marker, nuc, body, tissue, um, thr, island=None, gfp=None):
    """Nucleus-anchored typing (same signature as 938's type_cells): one row per nucleus-anchored cell."""
    fs = S938.smooth(marker, um, 0.3)
    gs = S938.smooth(gfp, um, 0.5)
    med, sd = S934.robust(gs[tissue])
    gthr = med + 10 * sd
    nl = split_nuclei(nuc & tissue, um)
    cells = expand_labels(nl, distance=EXPAND_UM / um)
    cells[~tissue] = 0
    H, W = marker.shape
    key = len(STATS)
    STATS[key] = dict(dapi=pctl(LAST["dapi"][tissue]), marker=pctl(marker[tissue]), gfp=pctl(gfp[tissue]), gthr=gthr)
    rows = []
    h = int(round(CROP_UM / um / 2))
    for r in measure.regionprops(cells):
        sl = r.slice
        cm = cells[sl] == r.label
        nm = (nl[sl] == r.label) & cm
        na = nm.sum() * um * um
        if not (NUC_AREA[0] <= na <= NUC_AREA[1]):
            continue
        y0, x0, y1, x1 = r.bbox
        if y0 == 0 or x0 == 0 or y1 == H or x1 == W:
            continue
        ring = cm & ~nm
        if ring.sum() < 4:
            continue
        rf = float((fs[sl][ring] >= thr).mean())
        gf = float((gs[sl][cm] >= gthr).mean())
        pos = rf >= RING_FRAC
        row = dict(y=float(r.centroid[0]), x=float(r.centroid[1]), area_um2=float(r.area * um * um),
                   nucleus_area_um2=float(na), nucleus_frac=float(nm.sum() / cm.sum()), typed=bool(pos),
                   marker_mean=float(fs[sl][ring].mean()), ring_pos_frac=rf, gfp_frac=gf,
                   gfp_rel_thr=float(gs[sl][cm].mean() / max(gthr, 1e-6)),
                   gfp_pos=bool(body[sl][cm].any()), gfp_overlap=float(body[sl][cm].mean()), gfp_cell=bool(gf >= GFP_FRAC),
                   gfp_rel=float(gs[sl][cm].mean() / max(med, 1.0)),
                   in_island=bool(island is not None and island[sl][cm].mean() > 0.5), _stats=key, _crop=-1)
        if pos or np.random.default_rng(int(r.centroid[0] * 7919 + r.centroid[1])).uniform() < 0.05:
            yc, xc = int(round(r.centroid[0])), int(round(r.centroid[1]))
            ya, yb, xa, xb = max(0, yc - h), min(H, yc + h), max(0, xc - h), min(W, xc + h)
            ck = len(STORE)
            STORE[ck] = dict(dapi=LAST["dapi"][ya:yb, xa:xb].astype(np.float16),
                             marker=marker[ya:yb, xa:xb].astype(np.float16), gfp=gfp[ya:yb, xa:xb].astype(np.float16),
                             cell=(cells[ya:yb, xa:xb] == r.label), nucleus=(nl[ya:yb, xa:xb] == r.label),
                             pad=(max(0, h - yc), max(0, h - xc)), side=2 * h, um=um)
            row["_crop"] = ck
        rows.append(row)
    return rows


def pctl(v):
    v = np.asarray(v, float)
    return (float(np.percentile(v, 0.5)), float(np.percentile(v, 99.9))) if v.size else (0.0, 1.0)


def clopper(k, n):
    if n == 0:
        return np.nan, np.nan
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = norm.ppf(0.975)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - hw), min(1.0, c + hw)


def render(c, rng_m, rng_g, rng_d):
    side = c["side"]

    def place(a):
        o = np.zeros((side, side), np.float32)
        oy, ox = c["pad"]
        hh, ww = min(a.shape[0], side - oy), min(a.shape[1], side - ox)
        o[oy:oy + hh, ox:ox + ww] = np.asarray(a, np.float32)[:hh, :ww]
        return o

    def sc(a, r_):
        return np.clip((a - r_[0]) / max(r_[1] - r_[0], 1e-6), 0, 1)
    m, g, d = sc(place(c["marker"]), rng_m), sc(place(c["gfp"]), rng_g), sc(place(c["dapi"]), rng_d)
    cell, nucm = place(c["cell"]) > 0.5, place(c["nucleus"]) > 0.5
    e_cell = segmentation.find_boundaries(cell, mode="outer")
    e_nuc = segmentation.find_boundaries(nucm, mode="inner")
    merge = np.dstack([m, g, np.clip(m + 0.8 * d, 0, 1)])
    mk, gp = np.dstack([m] * 3), np.dstack([g] * 3)
    for im in (mk, gp):
        im[e_cell] = (1.0, 0.85, 0.1)
        im[e_nuc] = (0.3, 0.75, 1.0)
    tiles = [Image.fromarray((t * 255).astype(np.uint8)).resize((PANEL_PX, PANEL_PX), Image.NEAREST)
             for t in (merge, mk, gp)]
    sheet = Image.new("RGB", (3 * PANEL_PX + 8, PANEL_PX), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (PANEL_PX + 4), 0))
    dr = ImageDraw.Draw(sheet)
    bl = 10.0 / CROP_UM * PANEL_PX
    dr.rectangle([6, PANEL_PX - 9, 6 + bl, PANEL_PX - 6], fill=(255, 255, 255))
    return sheet


def stage1():
    S938.nuclei, S938.type_cells = nuclei_kept, type_cells_nuclei
    acq = pd.read_csv(RESULTS / MOD / "parabiosis_acquisitions.csv")
    zc, zf = S938.zeiss(acq)
    oc, of = S938.olympus()
    cells = pd.DataFrame(zc + oc)
    fields = pd.DataFrame(zf + of)
    for s_, why in S938.NOT_INTERPRETED.items():
        cells.loc[cells.series == s_, "included"] = False
        fields.loc[fields.series == s_, "included"] = False
        fields.loc[fields.series == s_, "reason"] = why
    cells["compartment"] = np.where(cells.in_island, "marrow (otic-capsule island)", cells.compartment)
    typed = cells[cells.typed & cells.included]
    # summary per marker x compartment (instruments pooled; vestibular bone kept apart)
    rows = []
    for (mk, cp), g in typed.groupby(["marker", "compartment"]):
        f = fields[(fields.marker == mk) & (fields.compartment == cp.replace(" (otic-capsule island)", "")) &
                   fields.included]
        allc = cells[(cells.marker == mk) & (cells.compartment == cp) & cells.included]
        n, k = len(g), int(g.gfp_cell.sum())
        lo, hi = clopper(k, n)
        rows.append(dict(marker=mk, cell_type=S938.CELL_TYPE[mk], compartment=cp, fields=f.field.nunique(),
                         tissue_mm2=float(f.tissue_mm2.sum()), nuclei_cells=len(allc), typed_cells=n,
                         typed_share_pct=100 * n / max(len(allc), 1), gfp_pos_cells=k, gfp_pos_pct=100 * k / n,
                         ci_lo_pct=100 * lo, ci_hi_pct=100 * hi,
                         typed_per_mm2=n / f.tissue_mm2.sum() if f.tissue_mm2.sum() > 0 else np.nan))
    summ = pd.DataFrame(rows)
    for r in summ.itertuples():
        log(f"{r.marker} {r.compartment}: typed {r.typed_cells} of {r.nuclei_cells} nucleated cells "
            f"({r.typed_share_pct:.0f}%), GFP+ {r.gfp_pos_cells} ({r.gfp_pos_pct:.1f}%, {r.ci_lo_pct:.1f}-"
            f"{r.ci_hi_pct:.1f}); {r.typed_per_mm2:.0f} per mm2")
    # verification sample: per marker x compartment N_POS typed, N_NEG untyped (stored crops), every GFP+ typed
    rng = np.random.default_rng(SEED)
    inc = cells[cells.included & (cells._crop >= 0)]
    picks = []
    for (mk, cp), g in inc.groupby(["marker", "compartment"]):
        for flag, n_, lab in ((True, N_POS, "typed"), (False, N_NEG, "untyped")):
            gg = g[(g.typed == flag) & ~(g.typed & g.gfp_cell)] if flag else g[~g.typed]
            idx = gg.index.to_numpy()
            sel = idx if len(idx) <= n_ else rng.choice(idx, n_, replace=False)
            picks += [(i, f"{lab} {mk} {cp}") for i in sel]
        gp = g[g.typed & g.gfp_cell]
        picks += [(i, f"GFP+ call {mk} {cp}") for i in gp.index]
    # series display ranges from the marrow fields
    st = pd.DataFrame([dict(_stats=k, **{f"{c}_{j}": v[c][j] for c in ("dapi", "marker", "gfp") for j in (0, 1)})
                       for k, v in STATS.items()])
    cs = cells.merge(st, on="_stats", how="left")
    srng = {}
    for (se, mk), d in cs[cs.compartment == "marrow"].groupby(["series", "marker"]):
        srng[(se, mk)] = dict(marker=(d.marker_0.median(), d.marker_1.max()), gfp=(d.gfp_0.median(), d.gfp_1.max()))
    codes = rng.permutation(np.arange(1000, 10000))
    items, crops = [], []
    for n_, (i, stratum) in enumerate(picks):
        r = cells.loc[i]
        c = STORE[int(r._crop)]
        fr = STATS[int(r._stats)]
        sr = srng.get((r.series, r.marker), dict(marker=fr["marker"], gfp=fr["gfp"]))
        iid = f"N{codes[n_]}"
        crops.append((iid, r.marker, render(c, sr["marker"], sr["gfp"], fr["dapi"])))
        items.append(dict(item_id=iid, stratum=stratum, source=r.source, series=r.series, field=r.field,
                          plane=int(r.plane), marker=r.marker, compartment=r.compartment, y_px=r.y, x_px=r.x,
                          nucleus_area_um2=r.nucleus_area_um2, ring_pos_frac=r.ring_pos_frac, typed=bool(r.typed),
                          gfp_frac=r.gfp_frac, gfp_cell=bool(r.gfp_cell), gfp_overlap=r.gfp_overlap, seed=SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    order = rng.permutation(len(crops))
    W_, H_ = 3 * PANEL_PX + 8, PANEL_PX
    for s in range(0, len(order), 12):
        grp = [crops[j] for j in order[s:s + 12]]
        sheet = Image.new("RGB", (2 * W_ + 14, 6 * (H_ + 22)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for j, (iid, mk, im) in enumerate(grp):
            cx, cy = (j % 2) * (W_ + 14), (j // 2) * (H_ + 22)
            dr.text((cx + 2, cy + 2), f"{iid}  {mk}", fill=(0, 0, 0), font=font)
            sheet.paste(im, (cx, cy + 20))
        sheet.save(OUT / f"nuclei_{s // 12 + 1:02d}.png")
    key = pd.DataFrame(items)
    log(f"verification sample: {len(key)} items on {int(np.ceil(len(key) / 12))} sheets")
    log(key.groupby("stratum").size().to_string())
    cells_out = cells.drop(columns=["_crop", "_stats"])
    write_csv(cells_out, MOD, "celltype_nuclei_cells.csv", "Parabiosis, 54-day wild-type partner (945, D141): every "
              "nucleus-anchored cell of the immune cell-type stainings with ring marker coverage, GFP coverage, typed "
              "flag and GFP call", required=["field", "marker", "compartment", "typed", "gfp_cell"])
    write_csv(summ, MOD, "celltype_nuclei_summary.csv", "Parabiosis, 54-day wild-type partner (945, D141): per marker "
              "and compartment the nucleus-anchored typed cells, their share of nucleated cells, GFP+ (partner-derived) "
              "share with Clopper-Pearson 95% CI and density per mm2", required=["marker", "compartment", "typed_cells",
                                                                                 "gfp_pos_cells"])
    write_csv(key, MOD, "verification_nuclei_items.csv", "Verification (945, D141): answer key of the blinded sample of "
              "nucleus-anchored cells (seeded; typed, untyped and every GFP+ call per marker and compartment)",
              required=["item_id", "stratum", "typed", "gfp_cell"])


def stage2():
    key = pd.read_csv(RESULTS / MOD / "verification_nuclei_items.csv")
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    assert lab.item_id.is_unique and set(lab.item_id) == set(key.item_id), "labels must cover the sample exactly"
    d = key.merge(lab, on="item_id")
    rows = []
    for (mk, cp), g in d.groupby(["marker", "compartment"]):
        ty, un = g[g.typed], g[~g.typed]
        for name, sub, ok in (("typed: one marker+ cell", ty, ty.q1_cell == "yes"),
                              ("untyped: not a marker+ cell", un, un.q1_cell == "no")):
            dec = sub.q1_cell != "unsure"
            p, lo, hi = wilson(int((ok & dec).sum()), int(dec.sum()))
            rows.append(dict(marker=mk, compartment=cp, question=name, n_items=len(sub), n_decided=int(dec.sum()),
                             n_agree=int((ok & dec).sum()), share=p, ci_lo=lo, ci_hi=hi))
        gp = g[g.typed & (g.q1_cell == "yes")]
        dec = gp.q2_gfp != "unsure"
        ok = (gp.q2_gfp == "yes") == gp.gfp_cell.astype(bool)
        p, lo, hi = wilson(int((ok & dec).sum()), int(dec.sum()))
        rows.append(dict(marker=mk, compartment=cp, question="GFP call agrees (confirmed cells)", n_items=len(gp),
                         n_decided=int(dec.sum()), n_agree=int((ok & dec).sum()), share=p, ci_lo=lo, ci_hi=hi))
    ag = pd.DataFrame(rows)
    for r in ag.itertuples():
        log(f"{r.marker} {r.compartment} | {r.question}: {r.n_agree}/{r.n_decided} ({r.share:.0%})")
    d["labeller"] = "Research personnel, blinded to calls"
    write_csv(d, MOD, "verification_nuclei_labelled.csv", "Verification (945, D141): every item of the seeded sample "
              "with its calls and the blinded visual labels", required=["item_id", "q1_cell", "q2_gfp"])
    write_csv(ag, MOD, "verification_nuclei_agreement.csv", "Verification (945, D141): agreement of the nucleus-"
              "anchored calls with the blinded visual labels per marker and compartment (Wilson 95% CI; unsure "
              "excluded)", required=["marker", "compartment", "question", "share"])


if __name__ == "__main__":
    stage2() if "--unblind" in sys.argv[1:] else stage1()
