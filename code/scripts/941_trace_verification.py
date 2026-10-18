"""Round 10 (D137) — verification of the authors' manual vessel tracing in the 3D renders of Fig. 4 (user: "Please ensure
that the current quantification and labeling algorithms are sufficiently robust (or verify that the manual labeling is
accurate)"). The traced renders (data/photos/cochlea_3d/painted/1.png, 3.png; teal paint over a grey copy of the
render) are compared with the unannotated renders of the same views (slide_03_image_12, slide_08_image_04; vessel
channel in magenta):
1  registration: phase cross-correlation of the grey layer of the traced render with the unannotated render (asserted
   shift 0 px; Pearson r of the two layers outside the paint);
2  vessel signal of the unannotated render: Sato ridge filter (sigma 2-18 px) on the magenta signal (R - G), Otsu
   threshold within the rendered tissue;
3  traced centrelines: skeleton of the teal paint; thin traces (painted half-width <= 8 px, i.e. single vessels) are
   tested, broad paint (the wide vessel at the left edge of view 1, the bony region of view 2) is reported separately;
4  statistic: share of thin-trace centreline within 2 px of a detected vessel ridge, against the same traces randomly
   rotated and translated over the rendered tissue (999 draws; P = (1 + draws >= observed) / 1000); the longest
   unsupported stretch along the traces in um (scale from the burned-in 100-um bars, as 930).
Outputs (lib/io.write_csv, results/r9_figures/): trace_verification.csv (per view), trace_verification_null.csv (null
draws, for the robustness figure) and figures/panels/trace_check_view<k>.png (vessel ridges grey, thin traces on a
ridge green, off a ridge red, broad paint blue; diagnostic panel for Supplementary Fig. 1).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/941_trace_verification.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.registration import phase_cross_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import ROOT, log, write_csv  # noqa: E402

R3D = ROOT / "data" / "photos" / "cochlea_3d"
PANELS = ROOT / "figures" / "panels"
MOD = "r9_figures"
# (view, traced render, unannotated render, measured 100-um bar in px; 930 RENDERS)
VIEWS = [(1, "painted/1.png", "slide_03/color_images/slide_03_image_12.png", 97),
         (2, "painted/3.png", "slide_08/color_images/slide_08_image_04.png", 91)]
SIGMAS = (2, 3, 4, 6, 9, 13, 18)
NEAR_PX, THIN_HALF_PX, N_NULL, SEED = 2.0, 8.0, 999, 941


def bar_mask(shape):
    m = np.zeros(shape, bool)
    m[900:, :160] = True      # burned-in scale bar and label
    return m


def view(k, traced, orig, bar_px, rng):
    a = np.asarray(Image.open(R3D / traced).convert("RGB")).astype(float)
    o = np.asarray(Image.open(R3D / orig).convert("RGB")).astype(float)
    assert a.shape == o.shape
    bar = bar_mask(a.shape[:2])
    teal = (a[..., 1] - a[..., 0] > 30) & (a[..., 2] - a[..., 0] > 30) & ~bar
    grey = a.mean(-1)
    gfill = np.where(teal, grey[~teal].mean(), grey)
    shift, _, _ = phase_cross_correlation(gfill, o[..., 0], upsample_factor=10)
    assert np.allclose(shift, 0), shift
    r_layers = float(np.corrcoef(grey[~teal & ~bar], o[..., 0][~teal & ~bar])[0, 1])
    mag = o[..., 0] - o[..., 1]
    tissue = (mag > 30) & ~bar
    ridge = filters.sato(np.clip(mag, 0, None) / mag.max(), sigmas=SIGMAS, black_ridges=False)
    thr = filters.threshold_otsu(ridge[tissue])
    vessel = (ridge > thr) & ~bar
    dist = ndi.distance_transform_edt(~vessel)
    paint = morphology.binary_closing(teal, morphology.disk(1))
    half = ndi.distance_transform_edt(paint)
    sk = morphology.skeletonize(paint)
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    size = ndi.sum(sk, lab, range(1, n + 1))
    sk = np.isin(lab, 1 + np.flatnonzero(size >= 20))
    broad = morphology.binary_dilation(sk & (half > THIN_HALF_PX), morphology.disk(int(THIN_HALF_PX)))
    thin = sk & ~broad
    yy, xx = np.nonzero(thin)
    near = dist[yy, xx] <= NEAR_PX
    obs = float(near.mean())
    um = 100.0 / bar_px
    off = thin & (dist > NEAR_PX)
    lab2, n2 = ndi.label(off, structure=np.ones((3, 3)))
    longest = float(ndi.sum(off, lab2, range(1, n2 + 1)).max() * um) if n2 else 0.0
    # null: the same traces rotated and translated at random over the rendered tissue
    cy, cx = yy.mean(), xx.mean()
    fy, fx = np.nonzero(tissue)
    H, W = tissue.shape
    null = []
    while len(null) < N_NULL:
        th = rng.uniform(0, 2 * np.pi)
        j = rng.integers(len(fy))
        dy, dx = yy - cy, xx - cx
        ny = np.round(fy[j] + np.cos(th) * dy - np.sin(th) * dx).astype(int)
        nx = np.round(fx[j] + np.sin(th) * dy + np.cos(th) * dx).astype(int)
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        ok[ok] &= tissue[ny[ok], nx[ok]]
        if ok.mean() < 0.5:          # at least half of the trace must land on rendered tissue
            continue
        null.append(float((dist[ny[ok], nx[ok]] <= NEAR_PX).mean()))
    null = np.array(null)
    p = (1 + int((null >= obs).sum())) / (1 + N_NULL)
    # broad paint: share of its area on rendered tissue (vessel-channel signal)
    bpaint = paint & broad
    row = dict(view=k, traced_render=traced, unannotated_render=orig, um_per_px=round(um, 4),
               registration_shift_px=float(np.abs(shift).max()), layer_pearson_r=r_layers,
               ridge_sigmas_px=",".join(map(str, SIGMAS)), ridge_share_of_tissue=float(vessel[tissue].mean()),
               thin_trace_length_um=float(thin.sum() * um), thin_on_ridge_share=obs, null_mean=float(null.mean()),
               null_q95=float(np.percentile(null, 95)), null_max=float(null.max()), n_null=N_NULL, p_perm=p,
               longest_off_ridge_um=longest, off_ridge_stretches=int(n2),
               broad_paint_area_um2=float(bpaint.sum() * um * um),
               broad_paint_on_tissue_share=float(tissue[bpaint].mean()) if bpaint.any() else np.nan,
               near_px=NEAR_PX, thin_half_width_px=THIN_HALF_PX, seed=SEED)
    vis = np.dstack([vessel * 95] * 3).astype(np.uint8)
    vis[bpaint] = (70, 110, 200)
    vis[morphology.binary_dilation(thin & ~off, morphology.disk(1))] = (40, 200, 70)
    vis[morphology.binary_dilation(off, morphology.disk(2))] = (230, 40, 40)
    PANELS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis[:900]).save(PANELS / f"trace_check_view{k}.png")     # crop removes the bar region
    log(f"view {k}: shift {shift}, layers r {r_layers:.2f}; thin traces {thin.sum() * um:.0f} um, on a ridge "
        f"{obs:.1%} vs null {null.mean():.1%} (95th {np.percentile(null, 95):.1%}, max {null.max():.1%}), P {p:.3f}; "
        f"longest off-ridge stretch {longest:.0f} um; broad paint on tissue {row['broad_paint_on_tissue_share']:.1%}")
    return row, pd.DataFrame(dict(view=k, draw=np.arange(1, N_NULL + 1), on_ridge_share=null))


def main():
    rng = np.random.default_rng(SEED)
    rows, nulls = zip(*(view(k, t, o, b, rng) for k, t, o, b in VIEWS))
    d = pd.DataFrame(list(rows))
    assert (d.p_perm <= 0.01).all(), d
    write_csv(d, MOD, "trace_verification.csv", "Verification of the manual vessel tracing (941, D137): per render view "
              "registration to the unannotated render, share of thin traced centrelines on vessel ridges of the "
              "unannotated render vs randomly placed traces (permutation P), longest off-ridge stretch, broad paint",
              required=["view", "thin_on_ridge_share", "null_mean", "p_perm", "longest_off_ridge_um"])
    write_csv(pd.concat(nulls), MOD, "trace_verification_null.csv", "Verification of the manual vessel tracing (941, "
              "D137): on-ridge share of each randomly rotated and translated copy of the traces (null distribution)",
              required=["view", "draw", "on_ridge_share"])


if __name__ == "__main__":
    main()
