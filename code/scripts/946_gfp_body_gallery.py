"""Round 10 (D137 / D139) — blinded verification sample of the partner-derived (GFP+) bodies of the parabiosis survey
(934), the quantity behind the cochlea / marrow ratio of Fig. 4 (user: "In fact, you are the one who handles all
this—ensuring accuracy and performing manual labeling."). Seeded sample (SEED): up to N_PER GFP+ bodies per wild-type
partner x compartment (cochlea, marrow; the 54-day vestibular bone is not part of the ratio). The bodies are
re-detected with 934's own gfp_bodies on the same section (asserted: same count per section) so that each crop shows
the detected outline. Panels: merge (GFP green, far-red channel magenta, DAPI blue) | GFP with the outline | far-red
with the outline (broad-spectrum autofluorescence is bright in both). GFP display: from the section's tissue median to
the highest 99.9th percentile of the same animal's marrow sections (reporter-level yardstick); far-red and DAPI per
section. The sheets carry only an item ID; the key stays in results.
Outputs: results/r10_parabiosis/verification_bodies_items.csv (lib/io.write_csv; key) and contact sheets under
figures/gallery/bodies/ (review assets).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/946_gfp_body_gallery.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage import filters, measure, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S934 = load_module("s934", "934_parabiosis_survey.py")
MOD = "r10_parabiosis"
OUT = ROOT / "figures" / "gallery" / "bodies"
SEED, N_PER, CROP_UM, PANEL_PX, BAR_UM = 946, 20, 36.0, 150, 10.0


def scale(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ob = pd.read_csv(RESULTS / MOD / "parabiosis_objects.csv")
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    pl = pd.read_csv(RESULTS / MOD / "parabiosis_planes.csv")
    m = ob.merge(tf[["path", "animal", "compartment", "partner", "um_per_px"]], on="path", how="left")
    m = m[(m.partner == "WT") & m.compartment.isin(["cochlea", "marrow"]) & m.animal.isin(["14 d", "31 d",
                                                                                         "31 d noise", "54 d"])]
    rng = np.random.default_rng(SEED)
    picks = []
    for (an, cp), g in m.groupby(["animal", "compartment"]):
        idx = g.index.to_numpy()
        picks += list(idx if len(idx) <= N_PER else rng.choice(idx, N_PER, replace=False))
    sel = m.loc[picks].copy()
    # GFP display maximum per animal: highest 99.9th percentile of its marrow sections
    pm = pl.merge(tf[["path", "animal", "compartment", "partner"]], on="path")
    gmax = pm[(pm.partner == "WT") & (pm.compartment == "marrow")].groupby("animal").gfp_p999.max().to_dict()
    codes = rng.permutation(np.arange(1000, 10000))
    rows, crops = [], []
    for k, (path, g) in enumerate(sel.groupby("path")):
        a, meta = S934.read_czi(S934.SRC / path)
        um = meta["um_x"]
        gi, di, fi = S934.channel_roles(meta["channels"])
        planes = dict(S934.sampled_planes(a, meta))
        for z, gz in g.groupby("plane"):
            tis = planes[int(z)]
            far = a[fi, z] if fi is not None else None
            out, thr, _, keep = S934.gfp_bodies(a[gi, z], far, tis, um)
            n_pub = int(((ob.path == path) & (ob.plane == z)).sum())
            assert len(out) == n_pub, (path, z, len(out), n_pub)
            lab = measure.label(keep)
            props = measure.regionprops(lab)
            cen = np.array([p.centroid for p in props])
            gs = a[gi, z].astype(np.float32)
            glo = float(np.median(filters.gaussian(gs, 0.5 / um, preserve_range=True)[tis]))
            ghi = float(gmax.get(g.animal.iloc[0], np.percentile(gs[tis], 99.9)))
            fr = a[fi, z].astype(np.float32) if fi is not None else np.zeros_like(gs)
            dp = a[di, z].astype(np.float32) if di is not None else np.zeros_like(gs)
            flo, fhi = np.percentile(fr[tis], [0.5, 99.9]) if fi is not None else (0, 1)
            dlo, dhi = np.percentile(dp[tis], [0.5, 99.9]) if di is not None else (0, 1)
            h = int(round(CROP_UM / um / 2))
            H, W = gs.shape
            for r in gz.itertuples():
                j = int(np.argmin(np.hypot(cen[:, 0] - r.y, cen[:, 1] - r.x)))
                assert np.hypot(cen[j, 0] - r.y, cen[j, 1] - r.x) < 1.0
                y, x = int(round(r.y)), int(round(r.x))
                y0, y1, x0, x1 = max(0, y - h), min(H, y + h), max(0, x - h), min(W, x + h)
                side = 2 * h
                oy, ox = max(0, h - y), max(0, h - x)

                def place(v):
                    o_ = np.zeros((side, side), np.float32)
                    c_ = v[y0:y1, x0:x1]
                    o_[oy:oy + c_.shape[0], ox:ox + c_.shape[1]] = c_[:side - oy, :side - ox]
                    return o_
                G, F, D = place(scale(gs, glo, ghi)), place(scale(fr, flo, fhi)), place(scale(dp, dlo, dhi))
                obm = place((lab == props[j].label).astype(np.float32)) > 0.5
                edge = segmentation.find_boundaries(obm, mode="outer")
                merge = np.dstack([F, G, np.clip(F + 0.8 * D, 0, 1)])
                gp, fp = np.dstack([G] * 3), np.dstack([F] * 3)
                for im in (gp, fp):
                    im[edge] = (1.0, 0.85, 0.1)
                tiles = [Image.fromarray((t * 255).astype(np.uint8)).resize((PANEL_PX, PANEL_PX), Image.NEAREST)
                         for t in (merge, gp, fp)]
                sheet = Image.new("RGB", (3 * PANEL_PX + 8, PANEL_PX), (255, 255, 255))
                for i, t in enumerate(tiles):
                    sheet.paste(t, (i * (PANEL_PX + 4), 0))
                dr = ImageDraw.Draw(sheet)
                bl = BAR_UM / CROP_UM * PANEL_PX
                dr.rectangle([6, PANEL_PX - 9, 6 + bl, PANEL_PX - 6], fill=(255, 255, 255))
                iid = f"B{codes[len(rows)]}"
                crops.append((iid, sheet))
                rows.append(dict(item_id=iid, animal=r.animal, compartment=r.compartment, path=path, plane=int(z),
                                 y_px=r.y, x_px=r.x, area_um2=r.area_um2, gfp_mean=r.gfp_mean, signal=r.signal,
                                 broad_spectrum=bool(r.broad_spectrum), threshold=thr["threshold"],
                                 gfp_display=f"{glo:.1f}-{ghi:.1f}", far_channel=meta["channels"][fi] if fi is not None
                                 else "", seed=SEED))
        del a
        log(f"{path}: {len(g)} bodies")
    # contact sheets in a seeded random order, 12 per sheet, IDs only
    OUT.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(crops))
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    W_, H_ = 3 * PANEL_PX + 8, PANEL_PX
    for s in range(0, len(order), 12):
        grp = [crops[i] for i in order[s:s + 12]]
        sheet = Image.new("RGB", (2 * W_ + 14, 6 * (H_ + 22)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for j, (iid, im) in enumerate(grp):
            cx, cy = (j % 2) * (W_ + 14), (j // 2) * (H_ + 22)
            dr.text((cx + 2, cy + 2), iid, fill=(0, 0, 0), font=font)
            sheet.paste(im, (cx, cy + 20))
        sheet.save(OUT / f"bodies_{s // 12 + 1:02d}.png")
    key = pd.DataFrame(rows)
    log(key.groupby(["animal", "compartment"]).size().to_string())
    write_csv(key, MOD, "verification_bodies_items.csv", "Verification (946, D139): answer key of the blinded sample "
              "of GFP+ bodies of the parabiosis survey (wild-type partners; cochlea and marrow) - section, position, "
              "area, GFP signal, broad-spectrum flag and display range (seeded)",
              required=["item_id", "animal", "compartment", "broad_spectrum"])


if __name__ == "__main__":
    main()
