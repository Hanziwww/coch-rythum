"""Round 10 (D141) — nucleus-anchored immune cell typing, rules v2 (final), with a fresh blinded validation sample.
945 (rules v1) put every typed cell on a segmented nucleus and made the marrow calls largely correct (blinded review:
CD45 41/41, CD19 8/8, CD14 7/8, FUT4 11/16 single marker+ cells), but in the cochlear duct field the FUT4 background and
bright edge lines passed the series threshold (0/14), and GFP calls included cells that only border GFP signal. Rules v2
add two pre-existing conventions of this paper, fixed before any v2 count was inspected:
  marker+  v1 (>= half of the ring above the series threshold) AND local contrast: ring mean >= 2 x the local floor
           (25th percentile of the marker in the tissue 2-6 um beyond the cell; the '>= 2 x local background' rule of
           the cochlear-fluid cells, 931) AND the positive ring surrounds the nucleus: >= 5 of 8 angular sectors with at
           least 30% of their ring pixels above the threshold (an edge line passing one side of a cell does not count).
  GFP+     the D136 majority rule on the nucleus-anchored cell (>= half of the cell inside 934 GFP+ bodies) AND local
           contrast: cell GFP mean >= 2 x the median GFP of the tissue 2-6 um beyond the cell (the perilymph ring rule,
           934), so that haze around bright neighbours is not a call.
Everything else (fields, thresholds, separability, islands, exclusions, nuclei, cell = nucleus + 1.5 um) is 945 / 938.
Stage 1 (default): v1 and v2 calls for every cell; v2 summary per marker x compartment; v2 scored against the 945
development labels (data/verification/manual_labels_945.csv; reported as development agreement); a fresh seeded
validation sample (SEED; cells of the 945 sample excluded): per marker x compartment up to N_POS v2-typed cells, up to
N_REM cells typed by v1 but not by v2, up to N_NEG other untyped cells, and every v2 GFP+ typed cell; ID-only contact
sheets under figures/gallery/nuclei_v2/. Stage 2 (--unblind): reads data/verification/manual_labels_961.csv (labelled
blinded by the analyst; criteria of 944: one marker+ cell = marker signal around its nucleus; GFP+ = GFP clearly fills
most of the outlined cell at the series display) and writes the validation agreement and the verified summary.
Outputs (lib/io.write_csv, results/r10_parabiosis/): celltype_v2_cells.csv, celltype_v2_summary.csv,
celltype_v2_development.csv, verification_v2_items.csv; --unblind: verification_v2_labelled.csv,
verification_v2_agreement.csv.
Run: COCH_ROOT=/path/to/coch-rythum python scripts/961_celltype_nuclei_v2.py [--unblind]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import measure
from skimage.segmentation import expand_labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S945 = load_module("s945", "945_celltype_nuclei.py")
S938, S934 = S945.S938, S945.S934
MOD = "r10_parabiosis"
OUT = ROOT / "figures" / "gallery" / "nuclei_v2"
LABELS = ROOT / "data" / "verification" / "manual_labels_961.csv"
CONTRAST, SECTORS, SECTOR_MIN, SECTOR_FRAC, ANNULUS_UM = 2.0, 8, 5, 0.3, (2.0, 6.0)
SEED, N_POS, N_REM, N_NEG = 961, 15, 6, 4
STORE, STATS = S945.STORE, S945.STATS


def type_cells_v2(marker, nuc, body, tissue, um, thr, island=None, gfp=None):
    """945 cells with the v1 and v2 calls (signature of 938's type_cells)."""
    fs = S938.smooth(marker, um, 0.3)
    gs = S938.smooth(gfp, um, 0.5)
    med, sd = S934.robust(gs[tissue])
    gthr = med + 10 * sd
    nl = S945.split_nuclei(nuc & tissue, um)
    cells = expand_labels(nl, distance=S945.EXPAND_UM / um)
    cells[~tissue] = 0
    H, W = marker.shape
    key = len(STATS)
    dapi = S945.LAST["dapi"]
    STATS[key] = dict(dapi=S945.pctl(dapi[tissue]), marker=S945.pctl(marker[tissue]), gfp=S945.pctl(gfp[tissue]),
                      gthr=gthr)
    pad = int(np.ceil(ANNULUS_UM[1] / um)) + 1
    h = int(round(S945.CROP_UM / um / 2))
    rows = []
    for r in measure.regionprops(cells):
        y0, x0, y1, x1 = r.bbox
        if y0 == 0 or x0 == 0 or y1 == H or x1 == W:
            continue
        ya_, yb_, xa_, xb_ = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        cl = cells[ya_:yb_, xa_:xb_]
        cm = cl == r.label
        nm = (nl[ya_:yb_, xa_:xb_] == r.label) & cm
        na = nm.sum() * um * um
        if not (S945.NUC_AREA[0] <= na <= S945.NUC_AREA[1]):
            continue
        ring = cm & ~nm
        if ring.sum() < 4:
            continue
        f_, g_, t_ = fs[ya_:yb_, xa_:xb_], gs[ya_:yb_, xa_:xb_], tissue[ya_:yb_, xa_:xb_]
        above = f_ >= thr
        rf = float(above[ring].mean())
        dd = ndi.distance_transform_edt(~cm) * um
        ann = (dd > ANNULUS_UM[0]) & (dd <= ANNULUS_UM[1]) & t_
        floor = float(np.percentile(f_[ann], 25)) if ann.sum() >= 10 else np.nan
        rmean = float(f_[ring].mean())
        contrast = rmean / max(floor, 1e-6) if np.isfinite(floor) else np.nan
        cy, cx = np.argwhere(nm).mean(axis=0)
        yy, xx = np.nonzero(ring)
        sec = ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
        ab = above[ring]
        n_sec = int(sum((sec == k).any() and ab[sec == k].mean() >= SECTOR_FRAC for k in range(SECTORS)))
        typed_v1 = rf >= S945.RING_FRAC
        typed_v2 = bool(typed_v1 and np.isfinite(contrast) and contrast >= CONTRAST and n_sec >= SECTOR_MIN)
        gmean = float(g_[cm].mean())
        gbg = float(np.median(g_[ann])) if ann.sum() >= 10 else np.nan
        gcon = gmean / max(gbg, 1.0) if np.isfinite(gbg) else np.nan
        overlap = float(body[ya_:yb_, xa_:xb_][cm].mean())
        gf = float((g_[cm] >= gthr).mean())
        gfp_v2 = bool(overlap >= S938.MAJORITY and np.isfinite(gcon) and gcon >= CONTRAST)
        row = dict(y=float(ya_ + np.argwhere(cm).mean(axis=0)[0]), x=float(xa_ + np.argwhere(cm).mean(axis=0)[1]),
                   area_um2=float(cm.sum() * um * um), nucleus_area_um2=float(na), ring_pos_frac=rf,
                   marker_mean=rmean, marker_floor=floor, marker_contrast=contrast, ring_sectors=n_sec,
                   typed_v1=bool(typed_v1), typed=typed_v2, gfp_frac=gf, gfp_overlap=overlap, gfp_mean=gmean,
                   gfp_background=gbg, gfp_contrast=gcon, gfp_cell_v1=bool(gf >= S945.GFP_FRAC), gfp_cell=gfp_v2,
                   gfp_pos=bool(overlap > 0), in_island=bool(island is not None and
                                                             island[ya_:yb_, xa_:xb_][cm].mean() > 0.5),
                   _stats=key, _crop=-1)
        keep = typed_v1 or typed_v2 or np.random.default_rng(int(r.centroid[0] * 7919 + r.centroid[1])).uniform() < 0.05
        if keep:
            yc, xc = int(round(r.centroid[0])), int(round(r.centroid[1]))
            ya, yb, xa, xb = max(0, yc - h), min(H, yc + h), max(0, xc - h), min(W, xc + h)
            ck = len(STORE)
            STORE[ck] = dict(dapi=dapi[ya:yb, xa:xb].astype(np.float16), marker=marker[ya:yb, xa:xb].astype(np.float16),
                             gfp=gfp[ya:yb, xa:xb].astype(np.float16), cell=(cells[ya:yb, xa:xb] == r.label),
                             nucleus=(nl[ya:yb, xa:xb] == r.label), pad=(max(0, h - yc), max(0, h - xc)), side=2 * h,
                             um=um)
            row["_crop"] = ck
        rows.append(row)
    return rows


def summarise(cells, fields, flag="typed", gflag="gfp_cell"):
    typed = cells[cells[flag] & cells.included]
    rows = []
    for (mk, cp), g in typed.groupby(["marker", "compartment"]):
        f = fields[(fields.marker == mk) & (fields.compartment == cp.replace(" (otic-capsule island)", "")) &
                   fields.included]
        allc = cells[(cells.marker == mk) & (cells.compartment == cp) & cells.included]
        n, k = len(g), int(g[gflag].sum())
        lo, hi = S945.clopper(k, n)
        mm2 = float(f.tissue_mm2.sum())
        rows.append(dict(marker=mk, cell_type=S938.CELL_TYPE[mk], compartment=cp, fields=f.field.nunique(),
                         tissue_mm2=mm2, nucleated_cells=len(allc), typed_cells=n,
                         typed_share_pct=100 * n / max(len(allc), 1), gfp_pos_cells=k, gfp_pos_pct=100 * k / n,
                         ci_lo_pct=100 * lo, ci_hi_pct=100 * hi, typed_per_mm2=n / mm2 if mm2 > 0 else np.nan))
    return pd.DataFrame(rows)


def match(a, b, cols=("field", "plane", "marker")):
    """Rows of a matched to the nearest cell of b in the same field / plane / marker (within 1.5 px)."""
    out = []
    for key, g in a.groupby(list(cols)):
        bb = b
        for c_, v in zip(cols, key):
            bb = bb[bb[c_] == v]
        for i, r in g.iterrows():
            if not len(bb):
                out.append((i, None))
                continue
            dd = np.hypot(bb.y.to_numpy() - r.y_px, bb.x.to_numpy() - r.x_px)
            j = int(np.argmin(dd))
            out.append((i, bb.index[j] if dd[j] <= 1.5 else None))
    return dict(out)


def stage1():
    S938.nuclei, S938.type_cells = S945.nuclei_kept, type_cells_v2
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
    v1 = pd.read_csv(RESULTS / MOD / "celltype_nuclei_cells.csv")
    assert len(v1) == len(cells), (len(v1), len(cells))                    # same cells as 945
    sm = summarise(cells, fields)
    s1 = summarise(cells, fields, "typed_v1", "gfp_cell_v1").set_index(["marker", "compartment"])
    for r in sm.itertuples():
        o = s1.loc[(r.marker, r.compartment)] if (r.marker, r.compartment) in s1.index else None
        log(f"{r.marker} {r.compartment}: v2 typed {r.typed_cells} ({r.typed_share_pct:.0f}% of nucleated; v1 "
            f"{int(o.typed_cells) if o is not None else 0}), GFP+ {r.gfp_pos_cells} ({r.gfp_pos_pct:.1f}%, "
            f"{r.ci_lo_pct:.1f}-{r.ci_hi_pct:.1f}; v1 {int(o.gfp_pos_cells) if o is not None else 0}); "
            f"{r.typed_per_mm2:.0f} per mm2")
    # development: v2 calls on the 945 labelled sample
    dev = pd.read_csv(RESULTS / MOD / "verification_nuclei_labelled.csv")
    mp = match(dev, cells)
    dev["typed_v2"] = [bool(cells.loc[mp[i], "typed"]) if mp[i] is not None else np.nan for i in dev.index]
    dev["gfp_v2"] = [bool(cells.loc[mp[i], "gfp_cell"]) if mp[i] is not None else np.nan for i in dev.index]
    assert dev.typed_v2.notna().all()
    drows = []
    for (mk, cp), g in dev.groupby(["marker", "compartment"]):
        for rule, col in (("v1", "typed"), ("v2", "typed_v2")):
            t = g[g[col].astype(bool)]
            dec = t.q1_cell != "unsure"
            drows.append(dict(marker=mk, compartment=cp, rule=rule, question="typed: one marker+ cell",
                              n_decided=int(dec.sum()), n_agree=int(((t.q1_cell == "yes") & dec).sum())))
        c_ = g[(g.q1_cell == "yes") & (g.q2_gfp != "unsure")]
        for rule, col in (("v1", "gfp_cell"), ("v2", "gfp_v2")):
            drows.append(dict(marker=mk, compartment=cp, rule=rule, question="GFP call agrees (confirmed cells)",
                              n_decided=len(c_), n_agree=int(((c_.q2_gfp == "yes") == c_[col].astype(bool)).sum())))
    dv = pd.DataFrame(drows)
    dv["share"] = dv.n_agree / dv.n_decided.replace(0, np.nan)
    log("development (945 labels): " + "; ".join(f"{r.marker} {r.compartment} {r.rule} {r.question[:5]} "
                                                f"{r.n_agree}/{r.n_decided}" for r in dv.itertuples()))
    # fresh validation sample (945 sample cells excluded)
    used = set(i for i in mp.values() if i is not None)
    rng = np.random.default_rng(SEED)
    inc = cells[cells.included & (cells._crop >= 0) & ~cells.index.isin(used)]
    picks = []
    for (mk, cp), g in inc.groupby(["marker", "compartment"]):
        strata = [(g[g.typed & ~g.gfp_cell], N_POS, "typed"), (g[g.typed_v1 & ~g.typed], N_REM, "removed by v2"),
                  (g[~g.typed_v1 & ~g.typed], N_NEG, "untyped")]
        for gg, n_, lab in strata:
            idx = gg.index.to_numpy()
            sel = idx if len(idx) <= n_ else rng.choice(idx, n_, replace=False)
            picks += [(i, f"{lab} {mk} {cp}") for i in sel]
        picks += [(i, f"GFP+ call {mk} {cp}") for i in g[g.typed & g.gfp_cell].index]
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
        iid = f"V{codes[n_]}"
        crops.append((iid, r.marker, S945.render(c, sr["marker"], sr["gfp"], fr["dapi"])))
        items.append(dict(item_id=iid, stratum=stratum, source=r.source, series=r.series, field=r.field,
                          plane=int(r.plane), marker=r.marker, compartment=r.compartment, y_px=r.y, x_px=r.x,
                          typed=bool(r.typed), typed_v1=bool(r.typed_v1), gfp_cell=bool(r.gfp_cell),
                          gfp_cell_v1=bool(r.gfp_cell_v1), marker_contrast=r.marker_contrast,
                          ring_sectors=int(r.ring_sectors), gfp_contrast=r.gfp_contrast, gfp_overlap=r.gfp_overlap,
                          seed=SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    for f_ in OUT.glob("*.png"):
        f_.unlink()
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    order = rng.permutation(len(crops))
    W_, H_ = 3 * S945.PANEL_PX + 8, S945.PANEL_PX
    for s in range(0, len(order), 12):
        grp = [crops[j] for j in order[s:s + 12]]
        sheet = Image.new("RGB", (2 * W_ + 14, 6 * (H_ + 22)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for j, (iid, mk, im) in enumerate(grp):
            cx, cy = (j % 2) * (W_ + 14), (j // 2) * (H_ + 22)
            dr.text((cx + 2, cy + 2), f"{iid}  {mk}", fill=(0, 0, 0), font=font)
            sheet.paste(im, (cx, cy + 20))
        sheet.save(OUT / f"v2_{s // 12 + 1:02d}.png")
    key = pd.DataFrame(items)
    log(f"validation sample: {len(key)} items on {int(np.ceil(len(key) / 12))} sheets")
    log(key.groupby("stratum").size().to_string())
    write_csv(cells.drop(columns=["_crop", "_stats"]), MOD, "celltype_v2_cells.csv", "Parabiosis, 54-day wild-type "
              "partner (961, D141): every nucleus-anchored cell of the immune cell-type stainings with the v1 (945) and "
              "v2 calls - ring coverage, local marker contrast, ring sectors, GFP body overlap and local GFP contrast",
              required=["field", "marker", "compartment", "typed", "typed_v1", "gfp_cell"])
    write_csv(sm, MOD, "celltype_v2_summary.csv", "Parabiosis, 54-day wild-type partner (961, D141): per marker and "
              "compartment the v2-typed cells, their share of nucleated cells, partner-derived (GFP+) share with "
              "Clopper-Pearson 95% CI and density per mm2", required=["marker", "compartment", "typed_cells",
                                                                      "gfp_pos_cells", "ci_hi_pct"])
    write_csv(dv, MOD, "celltype_v2_development.csv", "Parabiosis typing (961, D141): v1 and v2 calls scored against "
              "the 945 blinded labels (development sample)", required=["marker", "compartment", "rule", "n_agree"])
    write_csv(key, MOD, "verification_v2_items.csv", "Verification (961, D141): answer key of the fresh blinded "
              "validation sample of the v2 typing (seeded; 945 sample excluded)", required=["item_id", "stratum",
                                                                                          "typed", "gfp_cell"])


def stage2():
    key = pd.read_csv(RESULTS / MOD / "verification_v2_items.csv")
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    assert lab.item_id.is_unique and set(lab.item_id) == set(key.item_id), "labels must cover the sample exactly"
    d = key.merge(lab, on="item_id")
    rows = []
    for (mk, cp), g in d.groupby(["marker", "compartment"]):
        ty = g[g.typed]
        rem = g[g.typed_v1 & ~g.typed]
        un = g[~g.typed_v1 & ~g.typed]
        for name, sub, ok in (("typed: one marker+ cell", ty, ty.q1_cell == "yes"),
                              ("removed by v2: not a marker+ cell", rem, rem.q1_cell == "no"),
                              ("untyped: not a marker+ cell", un, un.q1_cell == "no")):
            dec = sub.q1_cell != "unsure"
            p, lo, hi = S945.wilson(int((ok & dec).sum()), int(dec.sum()))
            rows.append(dict(marker=mk, compartment=cp, question=name, n_items=len(sub), n_decided=int(dec.sum()),
                             n_agree=int((ok & dec).sum()), share=p, ci_lo=lo, ci_hi=hi))
        c_ = ty[ty.q1_cell == "yes"]
        dec = c_.q2_gfp != "unsure"
        ok = (c_.q2_gfp == "yes") == c_.gfp_cell.astype(bool)
        p, lo, hi = S945.wilson(int((ok & dec).sum()), int(dec.sum()))
        rows.append(dict(marker=mk, compartment=cp, question="GFP call agrees (confirmed cells)", n_items=len(c_),
                         n_decided=int(dec.sum()), n_agree=int((ok & dec).sum()), share=p, ci_lo=lo, ci_hi=hi))
    ag = pd.DataFrame(rows)
    for r in ag.itertuples():
        log(f"{r.marker} {r.compartment} | {r.question}: {r.n_agree}/{r.n_decided}")
    d["labeller"] = "Claude (AI assistant), blinded to calls"
    write_csv(d, MOD, "verification_v2_labelled.csv", "Verification (961, D141): every item of the fresh validation "
              "sample with its v1 / v2 calls and the blinded visual labels", required=["item_id", "q1_cell", "q2_gfp"])
    write_csv(ag, MOD, "verification_v2_agreement.csv", "Verification (961, D141): agreement of the v2 calls with the "
              "blinded visual labels of the fresh validation sample per marker and compartment (Wilson 95% CI; unsure "
              "excluded)", required=["marker", "compartment", "question", "share", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    stage2() if "--unblind" in sys.argv[1:] else stage1()
