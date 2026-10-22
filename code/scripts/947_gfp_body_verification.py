"""Round 10 (D137 / D139) — how many GFP+ bodies of the parabiosis survey (934) are partner-derived cells by eye, and
what that does to the cochlea / marrow comparison of Fig. 4 (user: "In fact, you are the one who handles all
this—ensuring accuracy and performing manual labeling.").
Labels: data/verification/manual_labels_946.csv, made by research personnel on the ID-only contact
sheets of 946, blinded to animal, compartment and section, with criteria fixed before viewing: gfp_cell = yes when the
outlined body holds reporter-level GFP (judged against the animal's marrow display maximum), is cell-sized and
cell-shaped with a nucleus inside or beside it, and is not equally bright in the far-red channel; no = near-background
GFP, a speck or granule cluster, a band, haze or image-edge sliver, or broad-spectrum (bright in far-red as well);
unsure otherwise.
Per wild-type partner x compartment: share of bodies confirmed (yes among decided, Wilson 95% CI; and yes among all,
unsure counted as not confirmed); verified density = survey density x confirmed share; verified cochlea / marrow ratio
per pair with a stratified bootstrap 95% CI over the labelled bodies (B = 10,000, seed). The published ratio is
unchanged; this is its verification.
Outputs (lib/io.write_csv, results/r10_parabiosis/): verification_bodies_labelled.csv (key + labels),
verification_bodies_summary.csv (per animal x compartment) and verification_bodies_ratio.csv (per pair).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/947_gfp_body_verification.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402

MOD = "r10_parabiosis"
LABELS = ROOT / "data" / "verification" / "manual_labels_946.csv"
B, SEED = 10000, 947


def wilson(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = norm.ppf(1 - a / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    key = pd.read_csv(RESULTS / MOD / "verification_bodies_items.csv")
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    assert lab.item_id.is_unique and set(lab.item_id) == set(key.item_id)
    d = key.merge(lab, on="item_id")
    ratio = pd.read_csv(RESULTS / MOD / "parabiosis_ratio.csv")
    ratio = ratio[ratio.primary.astype(bool)].set_index("animal")      # the published (primary) row per pair
    assert ratio.index.is_unique
    rows = []
    for (an, cp), g in d.groupby(["animal", "compartment"]):
        yes, dec = int((g.gfp_cell == "yes").sum()), int((g.gfp_cell != "unsure").sum())
        p, lo, hi = wilson(yes, dec)
        pa, alo, ahi = wilson(yes, len(g))
        dens = float(ratio.loc[an, f"{cp}_density"])
        rows.append(dict(animal=an, compartment=cp, labelled=len(g), confirmed=yes, rejected=int(dec - yes),
                         unsure=int(len(g) - dec), confirmed_share_decided=p, ci_lo=lo, ci_hi=hi,
                         confirmed_share_all=pa, ci_all_lo=alo, ci_all_hi=ahi,
                         broad_flagged=int(g.broad_spectrum.sum()),
                         broad_flagged_rejected=int((g.broad_spectrum & (g.gfp_cell == "no")).sum()),
                         survey_density_per_mm2=dens, verified_density_per_mm2=dens * pa))
    sm = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    rr = []
    for an in sorted(d.animal.unique()):
        gc, gm = d[(d.animal == an) & (d.compartment == "cochlea")], d[(d.animal == an) & (d.compartment == "marrow")]
        if not len(gc) or not len(gm):
            continue
        dc, dm = float(ratio.loc[an, "cochlea_density"]), float(ratio.loc[an, "marrow_density"])
        yc, ym = (gc.gfp_cell == "yes").to_numpy(), (gm.gfp_cell == "yes").to_numpy()
        obs = (dc * yc.mean()) / (dm * ym.mean()) if ym.mean() > 0 else np.nan
        bs = []
        for _ in range(B):
            c_ = yc[rng.integers(len(yc), size=len(yc))].mean()
            m_ = ym[rng.integers(len(ym), size=len(ym))].mean()
            bs.append((dc * c_) / (dm * m_) if m_ > 0 else np.nan)
        bs = np.array(bs)
        bs = bs[np.isfinite(bs)]
        rr.append(dict(animal=an, published_ratio=dc / dm, cochlea_confirmed=f"{int(yc.sum())}/{len(yc)}",
                       marrow_confirmed=f"{int(ym.sum())}/{len(ym)}", verified_ratio=obs,
                       ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                       p_ratio_ge_1=float((bs >= 1).mean()), bootstrap=B, seed=SEED))
    rr = pd.DataFrame(rr)
    for r in sm.itertuples():
        log(f"{r.animal} {r.compartment}: confirmed {r.confirmed}/{r.labelled} (decided {r.confirmed_share_decided:.0%}, "
            f"{r.ci_lo:.0%}-{r.ci_hi:.0%}); density {r.survey_density_per_mm2:.0f} -> {r.verified_density_per_mm2:.0f}")
    for r in rr.itertuples():
        log(f"{r.animal}: ratio {r.published_ratio:.2f} -> verified {r.verified_ratio:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f});"
            f" bootstrap share >= 1: {r.p_ratio_ge_1:.3f}")
    d["labeller"] = "Research personnel, blinded to animal and compartment"
    write_csv(d, MOD, "verification_bodies_labelled.csv", "Verification (947, D139): every GFP+ body of the seeded "
              "sample (946) with its survey measures and the blinded visual label", required=["item_id", "gfp_cell"])
    write_csv(sm, MOD, "verification_bodies_summary.csv", "Verification (947, D139): per wild-type partner and "
              "compartment the share of GFP+ bodies confirmed as partner-derived cells by eye (Wilson 95% CI) and the "
              "verified density", required=["animal", "compartment", "confirmed_share_all", "verified_density_per_mm2"])
    write_csv(rr, MOD, "verification_bodies_ratio.csv", "Verification (947, D139): per pair the published cochlea / "
              "marrow ratio and the ratio of verified densities with a stratified bootstrap 95% CI",
              required=["animal", "published_ratio", "verified_ratio", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    main()
