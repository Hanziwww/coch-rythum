"""Round 10 (D141 / D142) — partner-derived (GFP+) cells of the parabiosis survey with an upgraded body rule (user:
"revert to the previous version, while only incorporating improvements to the layout and an upgraded quantitative
algorithm"). The blinded review of 934's GFP+ bodies (946/947) confirmed only 27% as partner-derived cells (12-57% by
pair and compartment); rejected bodies were specks, haze, bands along vessels, broad-spectrum objects and objects at the
image edge. The upgraded rule keeps 934's detection (tissue median + 10 robust SD, >= 15 um2, solidity >= 0.7, same
fields and sections) and adds pre-existing conventions of this paper, fixed before any upgraded count was inspected:
  cell-sized       area >= 25 um2 (the largest minimum area of the 939 sweep)
  local contrast   body GFP mean >= 2 x the median GFP of the tissue 2-6 um beyond the body (the perilymph ring rule)
  not broad        not broad-spectrum in lectin fields (934 flag: far-red as bright as GFP)
  nucleated        where the field has DAPI, >= 10% of the body inside a DAPI nucleus (938's nucleus rule)
The densities, the detection-corrected densities (934's detectable share per cochlear field applied to the upgraded
counts) and the brightest-section densities are recomputed per wild-type pair. The rule is scored against the 137
blinded labels of 946 (made before the rule) and against a fresh seeded validation sample (SEED; bodies of the 946
sample excluded; per pair and compartment up to N_POS bodies kept and N_NEG rejected by the rule), labelled blinded by
the analyst (data/verification/manual_labels_960.csv; criteria of 946).
Outputs (lib/io.write_csv, results/r10_parabiosis/): survey_cells_bodies.csv (every 934 body with the upgraded
features and call), survey_cells_summary.csv (animal x compartment), survey_cells_ratio.csv (per pair: observed,
corrected, brightest section), survey_cells_development.csv (agreement with the 946 labels) and
verification_survey_items.csv; --unblind: verification_survey_agreement.csv.
Run: COCH_ROOT=/path/to/coch-rythum python scripts/960_survey_cells.py [--unblind]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import filters, measure, segmentation

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
OUT = ROOT / "figures" / "gallery" / "survey_v2"
LABELS = ROOT / "data" / "verification" / "manual_labels_960.csv"
AREA_MIN, CONTRAST, ANNULUS_UM, NUC_FRAC = 25.0, 2.0, (2.0, 6.0), 0.1
SEED, N_POS, N_NEG = 960, 6, 4
ANIMALS = ["14 d", "31 d", "31 d noise", "54 d"]
CROP_UM, PANEL_PX = 36.0, 150


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = norm.ppf(0.975)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - hw), min(1.0, c + hw)


def body_features(a, meta, rel, store, gmax):
    """934 bodies of every analysed section with the upgraded features (and a crop per body for the review)."""
    um = meta["um_x"]
    gi, di, fi = S934.channel_roles(meta["channels"])
    if "lectin" not in rel.lower():
        fi = None
    rows = []
    pad = int(np.ceil(ANNULUS_UM[1] / um)) + 1
    h = int(round(CROP_UM / um / 2))
    for z, tis in S934.sampled_planes(a, meta):
        out, st, _, keep = S934.gfp_bodies(a[gi, z], a[fi, z] if fi is not None else None, tis, um)
        lab = measure.label(keep)
        props = measure.regionprops(lab)
        cen = np.array([p.centroid for p in props]) if props else np.zeros((0, 2))
        gs = filters.gaussian(a[gi, z].astype(np.float32), sigma=0.5 / um, preserve_range=True)
        nuc = S938.nuclei(a[di, z], tis, um)[0] if di is not None else None
        H, W = gs.shape
        for b in out:
            j = int(np.argmin(np.hypot(cen[:, 0] - b["y"], cen[:, 1] - b["x"])))
            r = props[j]
            y0, x0, y1, x1 = r.bbox
            ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
            bm = lab[ya:yb, xa:xb] == r.label
            dd = ndi.distance_transform_edt(~bm) * um
            ann = (dd > ANNULUS_UM[0]) & (dd <= ANNULUS_UM[1]) & tis[ya:yb, xa:xb]
            g_ = gs[ya:yb, xa:xb]
            bg = float(np.median(g_[ann])) if ann.sum() >= 10 else np.nan
            con = float(g_[bm].mean()) / max(bg, 1.0) if np.isfinite(bg) else np.nan
            nf = float(nuc[ya:yb, xa:xb][bm].mean()) if nuc is not None else np.nan
            cell = (b["area_um2"] >= AREA_MIN and np.isfinite(con) and con >= CONTRAST and not b["broad_spectrum"]
                    and (nuc is None or nf >= NUC_FRAC))
            ck = len(store)
            yc, xc = int(round(b["y"])), int(round(b["x"]))
            sy, ey, sx, ex = max(0, yc - h), min(H, yc + h), max(0, xc - h), min(W, xc + h)
            dp = a[di, z][sy:ey, sx:ex].astype(np.float32) if di is not None else np.zeros((ey - sy, ex - sx),
                                                                                           np.float32)
            fr = a[fi, z][sy:ey, sx:ex].astype(np.float32) if fi is not None else np.zeros_like(dp)
            store[ck] = dict(g=gs[sy:ey, sx:ex], d=dp, f=fr, m=(lab[sy:ey, sx:ex] == r.label),
                             pad=(max(0, h - yc), max(0, h - xc)), side=2 * h, glo=float(np.median(gs[tis])),
                             ghi=gmax, dr=S938.smooth(a[di, z], um, 0.5)[tis] if False else None,
                             dlo=float(np.percentile(dp, 0.5)) if dp.size else 0.0,
                             dhi=float(np.percentile(a[di, z][tis], 99.9)) if di is not None else 1.0,
                             flo=float(np.percentile(a[fi, z][tis], 0.5)) if fi is not None else 0.0,
                             fhi=float(np.percentile(a[fi, z][tis], 99.9)) if fi is not None else 1.0)
            rows.append(dict(path=rel, plane=int(z), y=b["y"], x=b["x"], area_um2=b["area_um2"],
                             signal=b["signal"], broad_spectrum=bool(b["broad_spectrum"]), local_background=bg,
                             local_contrast=con, nucleus_frac=nf, has_dapi=di is not None, cell=bool(cell),
                             _crop=ck))
    return rows


def render(c):
    side = c["side"]

    def place(v):
        o = np.zeros((side, side), np.float32)
        oy, ox = c["pad"]
        hh, ww = min(v.shape[0], side - oy), min(v.shape[1], side - ox)
        o[oy:oy + hh, ox:ox + ww] = v[:hh, :ww]
        return o

    def sc(v, lo, hi):
        return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    G, D, F = place(sc(c["g"], c["glo"], c["ghi"])), place(sc(c["d"], c["dlo"], c["dhi"])), \
        place(sc(c["f"], c["flo"], c["fhi"]))
    m = place(c["m"].astype(np.float32)) > 0.5
    e = segmentation.find_boundaries(m, mode="outer")
    merge = np.dstack([F, G, np.clip(F + 0.8 * D, 0, 1)])
    gp, fp = np.dstack([G] * 3), np.dstack([F] * 3)
    for im in (gp, fp):
        im[e] = (1.0, 0.85, 0.1)
    tiles = [Image.fromarray((t * 255).astype(np.uint8)).resize((PANEL_PX, PANEL_PX), Image.NEAREST)
             for t in (merge, gp, fp)]
    sheet = Image.new("RGB", (3 * PANEL_PX + 8, PANEL_PX), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (PANEL_PX + 4), 0))
    ImageDraw.Draw(sheet).rectangle([6, PANEL_PX - 9, 6 + 10 / CROP_UM * PANEL_PX, PANEL_PX - 6], fill=(255, 255, 255))
    return sheet


def stage1():
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    ob = pd.read_csv(RESULTS / MOD / "parabiosis_objects.csv")
    pl = pd.read_csv(RESULTS / MOD / "parabiosis_planes.csv")
    det = pd.read_csv(RESULTS / MOD / "parabiosis_detection.csv")
    w = tf[(tf.partner == "WT") & tf.compartment.isin(["cochlea", "marrow"]) & tf.animal.isin(ANIMALS)
           & tf.path.str.lower().str.endswith(".czi")]
    pm = pl.merge(tf[["path", "animal", "compartment", "partner"]], on="path")
    gmax = pm[(pm.partner == "WT") & (pm.compartment == "marrow")].groupby("animal").gfp_p999.max().to_dict()
    store, rows = {}, []
    for r in w.itertuples():
        a, meta = S934.read_czi(S934.SRC / r.path)
        rows_ = body_features(a, meta, r.path, store, gmax.get(r.animal, 255.0))
        n_pub = int((ob.path == r.path).sum())
        assert len(rows_) == n_pub, (r.path, len(rows_), n_pub)              # same bodies as 934
        for x in rows_:
            x.update(animal=r.animal, compartment=r.compartment)
        rows += rows_
        log(f"{r.path}: {len(rows_)} bodies, {sum(x['cell'] for x in rows_)} cells by the upgraded rule")
        del a
    b = pd.DataFrame(rows)
    # summaries and ratios (observed, detection-corrected, brightest section)
    best = pl.loc[pl.groupby("path").gfp_p999.idxmax(), ["path", "plane", "tissue_mm2"]].rename(
        columns={"plane": "best_plane", "tissue_mm2": "best_mm2"})
    per = b.groupby("path").agg(bodies=("cell", "size"), cells=("cell", "sum")).reset_index()
    bb = b.merge(best, on="path")
    per = per.merge(bb[bb.plane == bb.best_plane].groupby("path").cell.sum().rename("best_cells").reset_index(),
                    on="path", how="left").merge(best, on="path", how="left")
    per = w[["path", "animal", "compartment", "tissue_mm2", "gfp_bodies"]].merge(per, on="path", how="left").fillna(
        {"bodies": 0, "cells": 0, "best_cells": 0})
    assert (per.bodies == per.gfp_bodies).all()
    per = per.merge(det[["path", "detectable_share"]], on="path", how="left")
    summ = []
    for (an, cp), g in per.groupby(["animal", "compartment"]):
        ok = g[g.detectable_share > 0] if cp == "cochlea" else g
        summ.append(dict(animal=an, compartment=cp, fields=len(g), tissue_mm2=g.tissue_mm2.sum(),
                         bodies=int(g.bodies.sum()), cells=int(g.cells.sum()),
                         cell_share_of_bodies=g.cells.sum() / max(g.bodies.sum(), 1),
                         density_per_mm2=g.cells.sum() / g.tissue_mm2.sum(),
                         body_density_per_mm2=g.bodies.sum() / g.tissue_mm2.sum(),
                         corrected_density_per_mm2=((ok.cells / ok.detectable_share).sum() / ok.tissue_mm2.sum())
                         if cp == "cochlea" and len(ok) else np.nan,
                         best_plane_density_per_mm2=g.best_cells.sum() / g.best_mm2.sum()))
    sm = pd.DataFrame(summ)
    rr = []
    for an in ANIMALS:
        s_ = sm[sm.animal == an].set_index("compartment")
        if not {"cochlea", "marrow"} <= set(s_.index):
            continue
        m_, c_ = s_.loc["marrow"], s_.loc["cochlea"]
        rr.append(dict(animal=an, noise="noise" in an, days=int(an.split()[0]), marrow_density=m_.density_per_mm2,
                       cochlea_density=c_.density_per_mm2, cochlea_to_marrow=c_.density_per_mm2 / m_.density_per_mm2,
                       cochlea_corrected_density=c_.corrected_density_per_mm2,
                       corrected_cochlea_to_marrow=c_.corrected_density_per_mm2 / m_.density_per_mm2,
                       marrow_best_plane_density=m_.best_plane_density_per_mm2,
                       cochlea_best_plane_density=c_.best_plane_density_per_mm2,
                       best_plane_cochlea_to_marrow=c_.best_plane_density_per_mm2 / m_.best_plane_density_per_mm2,
                       marrow_cells=int(m_.cells), cochlea_cells=int(c_.cells), marrow_bodies=int(m_.bodies),
                       cochlea_bodies=int(c_.bodies)))
    rt = pd.DataFrame(rr)
    for r in rt.itertuples():
        log(f"{r.animal}: marrow {r.marrow_density:.0f} vs cochlea {r.cochlea_density:.0f} cells per mm2 "
            f"(ratio {r.cochlea_to_marrow:.2f}; corrected {r.corrected_cochlea_to_marrow:.2f}; brightest section "
            f"{r.best_plane_cochlea_to_marrow:.2f}); cells {r.marrow_cells}/{r.marrow_bodies} and "
            f"{r.cochlea_cells}/{r.cochlea_bodies} of the bodies")
    # development: agreement with the 946 labels
    lab = pd.read_csv(RESULTS / MOD / "verification_bodies_labelled.csv")
    key = []
    for r in lab.itertuples():
        g = b[(b.path == r.path) & (b.plane == r.plane)]
        j = int(np.argmin(np.hypot(g.y - r.y_px, g.x - r.x_px)))
        assert np.hypot(g.y.iloc[j] - r.y_px, g.x.iloc[j] - r.x_px) < 1.0
        key.append(g.index[j])
    lab["cell_v2"] = b.loc[key, "cell"].to_numpy()
    dec = lab[lab.gfp_cell != "unsure"]
    y = dec.gfp_cell == "yes"
    devrows = []
    for rule, pred in (("934 (all bodies)", pd.Series(True, index=dec.index)), ("upgraded", dec.cell_v2.astype(bool))):
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        devrows.append(dict(rule=rule, labelled=len(dec), tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(tp + fp, 1),
                            sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1)))
    dv = pd.DataFrame(devrows)
    log("development (946 labels): " + "; ".join(f"{r.rule}: precision {r.precision:.2f}, sensitivity "
                                                f"{r.sensitivity:.2f}, specificity {r.specificity:.2f}"
                                                for r in dv.itertuples()))
    # fresh validation sample
    rng = np.random.default_rng(SEED)
    pool = b.drop(index=key)
    picks = []
    for (an, cp), g in pool.groupby(["animal", "compartment"]):
        for flag, n_, nm in ((True, N_POS, "kept"), (False, N_NEG, "rejected")):
            idx = g[g.cell == flag].index.to_numpy()
            sel = idx if len(idx) <= n_ else rng.choice(idx, n_, replace=False)
            picks += [(i, f"{nm} {an} {cp}") for i in sel]
    codes = rng.permutation(np.arange(1000, 10000))
    items, crops = [], []
    for n_, (i, stratum) in enumerate(picks):
        r = b.loc[i]
        iid = f"S{codes[n_]}"
        crops.append((iid, render(store[int(r._crop)])))
        items.append(dict(item_id=iid, stratum=stratum, path=r.path, plane=int(r.plane), y_px=r.y, x_px=r.x,
                          animal=r.animal, compartment=r.compartment, area_um2=r.area_um2,
                          local_contrast=r.local_contrast, nucleus_frac=r.nucleus_frac,
                          broad_spectrum=bool(r.broad_spectrum), cell=bool(r.cell), seed=SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    for f_ in OUT.glob("*.png"):
        f_.unlink()
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
        for j, (iid, im) in enumerate(grp):
            cx, cy = (j % 2) * (W_ + 14), (j // 2) * (H_ + 22)
            dr.text((cx + 2, cy + 2), iid, fill=(0, 0, 0), font=font)
            sheet.paste(im, (cx, cy + 20))
        sheet.save(OUT / f"survey_{s // 12 + 1:02d}.png")
    vk = pd.DataFrame(items)
    log(f"validation sample: {len(vk)} bodies on {int(np.ceil(len(vk) / 12))} sheets")
    write_csv(b.drop(columns=["_crop"]), MOD, "survey_cells_bodies.csv", "Parabiosis survey (960, D142): every GFP+ "
              "body of 934 in the wild-type partners with the upgraded features (area, local GFP contrast, broad "
              "spectrum, nucleus share) and the upgraded cell call", required=["path", "plane", "cell"])
    write_csv(sm, MOD, "survey_cells_summary.csv", "Parabiosis survey (960, D142): per wild-type partner and "
              "compartment the partner-derived cells by the upgraded rule, their share of 934's bodies, densities per "
              "mm2 (observed, detection-corrected, brightest section)", required=["animal", "compartment", "cells",
                                                                                  "density_per_mm2"])
    write_csv(rt, MOD, "survey_cells_ratio.csv", "Parabiosis survey (960, D142): per wild-type pair the partner-derived "
              "cells per mm2 of marrow and cochlea by the upgraded rule and the cochlea / marrow ratio (observed, "
              "detection-corrected, brightest section)", required=["animal", "marrow_density", "cochlea_density",
                                                                    "cochlea_to_marrow"])
    write_csv(dv, MOD, "survey_cells_development.csv", "Parabiosis survey (960, D142): 934's bodies and the upgraded "
              "rule scored against the 946 blinded labels (precision, sensitivity, specificity)",
              required=["rule", "precision", "sensitivity"])
    write_csv(vk, MOD, "verification_survey_items.csv", "Verification (960, D142): answer key of the fresh blinded "
              "validation sample of the upgraded body rule (seeded; 946 sample excluded)",
              required=["item_id", "stratum", "cell"])


def verified_ratio(val, B=10000):
    """Label-corrected densities: per pair x compartment, estimated partner-derived cells = kept bodies x (share of kept
    labelled bodies confirmed) + rejected bodies x (share of rejected labelled bodies confirmed), from the decided labels
    of the 946 sample (random within pair x compartment, split by the rule post hoc) and the 960 sample (random within
    pair x compartment x call); ratio of the corrected densities with a stratified bootstrap over labelled bodies."""
    b = pd.read_csv(RESULTS / MOD / "survey_cells_bodies.csv")
    sm = pd.read_csv(RESULTS / MOD / "survey_cells_summary.csv").set_index(["animal", "compartment"])
    rt = pd.read_csv(RESULTS / MOD / "survey_cells_ratio.csv").set_index("animal")
    dev = pd.read_csv(RESULTS / MOD / "verification_bodies_labelled.csv")
    calls = []
    for r in dev.itertuples():
        g = b[(b.path == r.path) & (b.plane == r.plane)]
        j = int(np.argmin(np.hypot(g.y - r.y_px, g.x - r.x_px)))
        calls.append(bool(g.cell.iloc[j]))
    dev["cell"] = calls
    lab = pd.concat([dev[["animal", "compartment", "cell", "gfp_cell"]],
                     val[["animal", "compartment", "cell", "gfp_cell"]]], ignore_index=True)
    lab = lab[lab.gfp_cell != "unsure"]
    rng = np.random.default_rng(SEED)
    rows = []
    for an in rt.index:
        est, boots = {}, {}
        for cp in ("cochlea", "marrow"):
            g = b[(b.animal == an) & (b.compartment == cp)]
            n_keep, n_rej = int(g.cell.sum()), int((~g.cell).sum())
            yk = (lab[(lab.animal == an) & (lab.compartment == cp) & lab.cell].gfp_cell == "yes").to_numpy()
            yr = (lab[(lab.animal == an) & (lab.compartment == cp) & ~lab.cell].gfp_cell == "yes").to_numpy()
            pk, pr = (yk.mean() if len(yk) else np.nan), (yr.mean() if len(yr) else 0.0)
            mm2 = float(sm.loc[(an, cp), "tissue_mm2"])
            est[cp] = (n_keep * pk + n_rej * pr) / mm2
            bk = yk[rng.integers(len(yk), size=(B, len(yk)))].mean(1) if len(yk) else np.full(B, np.nan)
            br = yr[rng.integers(len(yr), size=(B, len(yr)))].mean(1) if len(yr) else np.zeros(B)
            boots[cp] = (n_keep * bk + n_rej * br) / mm2
            rows_cp = dict(kept=n_keep, rejected=n_rej, kept_labelled=len(yk), kept_confirmed=int(yk.sum()),
                           rejected_labelled=len(yr), rejected_confirmed=int(yr.sum()))
            est[cp + "_detail"] = rows_cp
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = boots["cochlea"] / boots["marrow"]
        bs = bs[np.isfinite(bs)]
        dc, dm = est["cochlea_detail"], est["marrow_detail"]
        rows.append(dict(animal=an, noise="noise" in an, rule_ratio=float(rt.loc[an, "cochlea_to_marrow"]),
                         cochlea_verified_density=est["cochlea"], marrow_verified_density=est["marrow"],
                         verified_ratio=est["cochlea"] / est["marrow"], ci_lo=float(np.percentile(bs, 2.5)),
                         ci_hi=float(np.percentile(bs, 97.5)), p_ratio_ge_1=float((bs >= 1).mean()),
                         cochlea_kept=dc["kept"], cochlea_rejected=dc["rejected"],
                         cochlea_kept_confirmed=f"{dc['kept_confirmed']}/{dc['kept_labelled']}",
                         cochlea_rejected_confirmed=f"{dc['rejected_confirmed']}/{dc['rejected_labelled']}",
                         marrow_kept=dm["kept"], marrow_rejected=dm["rejected"],
                         marrow_kept_confirmed=f"{dm['kept_confirmed']}/{dm['kept_labelled']}",
                         marrow_rejected_confirmed=f"{dm['rejected_confirmed']}/{dm['rejected_labelled']}",
                         bootstrap=B, seed=SEED))
    vr = pd.DataFrame(rows)
    for r in vr.itertuples():
        log(f"{r.animal}: rule ratio {r.rule_ratio:.2f}; label-corrected {r.verified_ratio:.2f} ({r.ci_lo:.2f}-"
            f"{r.ci_hi:.2f}; bootstrap share >= 1: {r.p_ratio_ge_1:.3f})")
    return vr


def stage2():
    key = pd.read_csv(RESULTS / MOD / "verification_survey_items.csv")
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    assert lab.item_id.is_unique and set(lab.item_id) == set(key.item_id)
    d = key.merge(lab, on="item_id")
    rows = []
    for name, sub, ok in (("kept: partner-derived cell", d[d.cell], d[d.cell].gfp_cell == "yes"),
                          ("rejected: not a partner-derived cell", d[~d.cell], d[~d.cell].gfp_cell == "no")):
        for cp in ("all", "cochlea", "marrow"):
            s_ = sub if cp == "all" else sub[sub.compartment == cp]
            o_ = ok.loc[s_.index]
            dec = s_.gfp_cell != "unsure"
            p, lo, hi = wilson(int((o_ & dec).sum()), int(dec.sum()))
            rows.append(dict(question=name, compartment=cp, n_items=len(s_), n_decided=int(dec.sum()),
                             n_agree=int((o_ & dec).sum()), share=p, ci_lo=lo, ci_hi=hi))
    ag = pd.DataFrame(rows)
    for r in ag.itertuples():
        log(f"{r.question} ({r.compartment}): {r.n_agree}/{r.n_decided}")
    vr = verified_ratio(d)
    d["labeller"] = "Claude (AI assistant), blinded to calls"
    write_csv(vr, MOD, "survey_cells_verified.csv", "Parabiosis survey (960, D142): per wild-type pair the cochlea / "
              "marrow ratio of the upgraded rule and the label-corrected ratio (kept bodies x precision + rejected "
              "bodies x miss rate per pair, compartment and call, from the blinded labels of 946 and 960; stratified "
              "bootstrap 95% CI)", required=["animal", "rule_ratio", "verified_ratio", "ci_lo", "ci_hi"])
    write_csv(d, MOD, "verification_survey_labelled.csv", "Verification (960, D142): every item of the fresh "
              "validation sample with its call and the blinded visual label", required=["item_id", "gfp_cell"])
    write_csv(ag, MOD, "verification_survey_agreement.csv", "Verification (960, D142): precision (kept) and negative "
              "agreement (rejected) of the upgraded body rule on the fresh blinded sample (Wilson 95% CI)",
              required=["question", "compartment", "share", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    stage2() if "--unblind" in sys.argv[1:] else stage1()
