"""Round 10 (D137 / D139) — agreement of the image calls with blinded visual labels of the verification sample (943;
user: "In fact, you are the one who handles all this—ensuring accuracy and performing manual labeling.").
Labels: data/verification/manual_labels_943.csv, made by research personnel on ID-stamped contact sheets
built from figures/gallery/gallery_data.js only (no call, compartment, field or stratum visible), with criteria fixed
before viewing:
  q1_cell  yes = the outline holds marker signal above the local background, corresponds to one cell-sized object (or
           part of one cell, e.g. a ring segment around its nucleus) and has a nucleus inside or enclosed; no = it spans
           two or more cells, a band or layer, an edge or streak without a nucleus, or no signal above background;
           unsure otherwise.
  q2_gfp   yes = GFP above background covers most (> ~50%) of the outlined area; no = GFP at background, only at the
           edge (a touching GFP+ neighbour) or a speck (< ~25%); unsure otherwise.
  tissue   marrow (haematopoietic marrow in bone) / cochlea (cochlear soft tissue) / bone (no marrow) / other / unsure.
Agreement (Wilson 95% CIs): typed objects judged cells (q1 yes among decided; also yes among all), rejected objects
judged non-cells, the GFP majority rule and the former contact rule against q2, and tissue assignments against the
field labels; the key claims re-read with the labels (cochlear neutrophils and leukocytes, B-cell background).
Outputs (lib/io.write_csv, results/r10_parabiosis/): verification_items_labelled.csv (key + labels per item),
verification_agreement.csv (per stratum and question) and verification_claims.csv.
Run: COCH_ROOT=/path/to/coch-rythum python scripts/944_verification_agreement.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402

MOD = "r10_parabiosis"
LABELS = ROOT / "data" / "verification" / "manual_labels_943.csv"


def wilson(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan, np.nan
    z = norm.ppf(1 - a / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(d, stratum, question, success, decided, note=""):
    """Share of `success` among `decided` rows of d (booleans), with counts and the Wilson interval."""
    k, n = int((success & decided).sum()), int(decided.sum())
    p, lo, hi = wilson(k, n)
    return dict(stratum=stratum, question=question, n_items=len(d), n_decided=n, n_agree=k, share=p, ci_lo=lo,
                ci_hi=hi, n_unsure=int(len(d) - n), note=note)


def main():
    key = pd.read_csv(RESULTS / MOD / "verification_gallery_items.csv")
    lab = pd.read_csv(LABELS, dtype=str).fillna("")
    assert lab.item_id.is_unique and set(lab.item_id) == set(key.item_id), "labels must cover every item exactly once"
    d = key.merge(lab, on="item_id")
    cells = d[d.kind == "cell"].copy()
    fields = d[d.kind == "field"].copy()
    rows = []
    # marker calls: typed objects judged cells, rejected objects judged non-cells
    for st, g in cells.groupby("stratum"):
        q1 = g.q1_cell
        if g.typed.all():
            rows.append(rate(g, st, "typed object is one marker+ cell (decided)", q1 == "yes", q1 != "unsure"))
            rows.append(rate(g, st, "typed object is one marker+ cell (all; unsure = not confirmed)", q1 == "yes",
                             pd.Series(True, index=g.index)))
        elif (~g.typed).all():
            rows.append(rate(g, st, "rejected object is not a cell (decided)", q1 == "no", q1 != "unsure"))
    typed = cells[cells.typed]
    for grp, g in [("all typed", typed), ("typed, cochlea", typed[typed.compartment == "cochlea"]),
                   ("typed, marrow and vestibular bone", typed[typed.compartment != "cochlea"])]:
        rows.append(rate(g, grp, "typed object is one marker+ cell (decided)", g.q1_cell == "yes",
                         g.q1_cell != "unsure"))
    for mk, g in typed.groupby("marker"):
        rows.append(rate(g, f"typed {mk}, all compartments", "typed object is one marker+ cell (decided)",
                         g.q1_cell == "yes", g.q1_cell != "unsure"))
    # GFP calls: majority rule (published) and contact rule (former) against the visual label
    dec = cells.q2_gfp != "unsure"
    vis = cells.q2_gfp == "yes"
    for rule, col in [("majority rule (published)", "gfp_majority"), ("contact rule (former)", "gfp_contact")]:
        alg = cells[col].astype(bool)
        rows.append(rate(cells, f"all cells, {rule}", "algorithm GFP call agrees with the visual label",
                         alg == vis, dec))
        rows.append(rate(cells[vis], f"visually GFP+, {rule}", "algorithm calls GFP+ (sensitivity)", alg[vis],
                         dec[vis]))
        rows.append(rate(cells[~vis & dec], f"visually GFP-, {rule}", "algorithm calls GFP- (specificity)",
                         ~alg[~vis & dec], dec[~vis & dec]))
        rows.append(rate(cells[alg], f"algorithm GFP+, {rule}", "visual label GFP+ (precision)", vis[alg], dec[alg]))
    # tissue assignments
    exp = fields.compartment.map({"marrow": "marrow", "cochlea": "cochlea", "excluded": "not marrow"})
    got = fields.tissue.map({"marrow": "marrow", "cochlea": "cochlea", "bone": "not marrow", "other": "not marrow",
                             "unsure": "unsure"})
    agree = (exp == got) | ((exp == "not marrow") & got.isin(["cochlea"]))
    for st, g in fields.groupby("compartment"):
        m = g.index
        rows.append(rate(g, f"fields assigned {st}", "visual tissue label agrees with the assignment", agree[m],
                         got[m] != "unsure"))
    rows.append(rate(fields, "all assigned fields", "visual tissue label agrees with the assignment", agree,
                     got != "unsure"))
    ag = pd.DataFrame(rows)
    # key claims with the visual labels
    cl = []
    for name, sel in [("cochlear neutrophils (FUT4+)", (cells.typed) & (cells.marker == "FUT4")
                       & (cells.compartment == "cochlea")),
                      ("cochlear leukocytes (CD45+)", cells.typed & (cells.marker == "CD45")
                       & (cells.compartment == "cochlea")),
                      ("marrow B cells (CD19+; reporter-negative)", cells.typed & (cells.marker == "CD19")),
                      ("marrow neutrophils (FUT4+, sample)", cells.typed & (cells.marker == "FUT4")
                       & cells.compartment.str.startswith("marrow"))]:
        g = cells[sel]
        conf = g[g.q1_cell == "yes"]
        cl.append(dict(claim=name, typed_in_sample=len(g), algorithm_gfp_pos=int(g.gfp_majority.sum()),
                       visual_cell_yes=len(conf), visual_cell_unsure=int((g.q1_cell == "unsure").sum()),
                       visual_cell_no=int((g.q1_cell == "no").sum()),
                       visual_gfp_pos_all=int((g.q2_gfp == "yes").sum()),
                       visual_gfp_unsure_all=int((g.q2_gfp == "unsure").sum()),
                       confirmed_cells_gfp_pos=int((conf.q2_gfp == "yes").sum()),
                       confirmed_cells_algorithm_gfp_pos=int(conf.gfp_majority.sum())))
    cl = pd.DataFrame(cl)
    for r in ag.itertuples():
        log(f"{r.stratum} | {r.question}: {r.n_agree}/{r.n_decided} = {r.share:.0%} ({r.ci_lo:.0%}-{r.ci_hi:.0%}); "
            f"unsure {r.n_unsure}")
    log(cl.to_string())
    d["labeller"] = "Research personnel, blinded to calls"
    write_csv(d, MOD, "verification_items_labelled.csv", "Verification (944, D139): every item of the seeded sample "
              "(943) with its algorithm calls and the blinded visual labels (cell, GFP, tissue; note)",
              required=["item_id", "stratum", "q1_cell", "q2_gfp", "tissue"])
    write_csv(ag, MOD, "verification_agreement.csv", "Verification (944, D139): agreement of the image calls with the "
              "blinded visual labels per stratum and question (share with Wilson 95% CI; unsure items excluded from "
              "the decided denominator)", required=["stratum", "question", "n_decided", "share", "ci_lo", "ci_hi"])
    write_csv(cl, MOD, "verification_claims.csv", "Verification (944, D139): key cell-type claims re-read with the "
              "blinded visual labels (typed cells in the sample, confirmed cells, GFP+ by algorithm and by eye)",
              required=["claim", "typed_in_sample", "visual_cell_yes", "confirmed_cells_gfp_pos"])


if __name__ == "__main__":
    main()
