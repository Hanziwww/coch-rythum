"""Round 10 (D137) — objective evidence for the tissue assignments of the parabiosis fields that were made by eye
(user: "Please ensure that the current quantification and labeling algorithms are sufficiently robust (or verify that
the manual labeling is accurate)"). 934 assigns most fields by their file names and some by morphology (MORPH:
haematopoietic marrow inside bone vs cochlear lateral wall / duct), excludes two 'ear-bone' fields without marrow
(NO_MARROW) and types an otic-capsule marrow island inside a lateral-wall field as marrow (ISLANDS). Haematopoietic
marrow is a densely packed nucleated tissue; here every field with DAPI is measured on the analysed optical sections of
934 (sampled_planes) with the nucleus rule of 938: nuclei per mm2 of tissue, nuclear area share and median
nearest-neighbour distance between nuclei. A logistic model of marrow vs cochlea on the name-labelled fields
(leave-one-out accuracy) then scores every morphology-assigned and excluded field and the island (inside vs the rest of
its field), so each by-eye call is checked against the nuclear signature of fields labelled at acquisition.
Outputs (lib/io.write_csv, results/r10_parabiosis/): tissue_assignment_fields.csv (per field and island: measures,
assignment and its basis, model probability of marrow, agreement) and tissue_assignment_summary.csv (per class:
medians and ranges; model accuracy).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/942_tissue_assignment_evidence.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S934 = load_module("s934", "934_parabiosis_survey.py")
S938 = load_module("s938", "938_parabiosis_cell_types.py")
MOD = "r10_parabiosis"
FEATURES = ["log10_nuclei_per_mm2", "nuclear_area_share", "log10_nn_um"]


def measure_field(a, meta, region=None):
    """Nuclei per mm2 of tissue, nuclear area share and median nearest-neighbour distance (um) over the analysed
    sections; region (optional, Y x X bool) restricts the tissue to it."""
    um = meta["um_x"]
    _, di, _ = S934.channel_roles(meta["channels"])
    n_nuc, tis_px, nuc_px, nn = 0, 0, 0, []
    for z, tis in S934.sampled_planes(a, meta):
        if region is not None:
            tis = tis & region
        if tis.sum() * um * um < 2000:           # < 2,000 um2 of tissue in this section
            continue
        nuc, _ = S938.nuclei(a[di, z], tis, um)
        props = [r for r in measure.regionprops(measure.label(nuc)) if r.area * um * um >= 10.0]
        n_nuc += len(props)
        tis_px += int(tis.sum())
        nuc_px += int(nuc.sum())
        if len(props) >= 3:
            c = np.array([r.centroid for r in props]) * um
            d, _ = cKDTree(c).query(c, k=2)
            nn.extend(d[:, 1].tolist())
    if tis_px == 0 or n_nuc < 5:
        return None
    mm2 = tis_px * um * um / 1e6
    return dict(tissue_mm2=mm2, nuclei=n_nuc, nuclei_per_mm2=n_nuc / mm2, nuclear_area_share=nuc_px / tis_px,
                nn_median_um=float(np.median(nn)) if nn else np.nan)


def logistic_loo(X, y):
    """Leave-one-out accuracy of a logistic model (standardised features) and the model fitted to all fields."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    hits = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        hits.append(int(model().fit(X[m], y[m]).predict(X[i:i + 1])[0] == y[i]))
    return float(np.mean(hits)), model().fit(X, y)


def main():
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    rels = [r for r in tf.loc[tf.has_dapi.astype(bool), "path"] if r.lower().endswith(".czi")]  # liver: TIFF sets
    rels += [r for r in list(S934.MORPH) + list(S934.NO_MARROW) if r not in rels]
    rows = []
    for rel in rels:
        p = S934.SRC / rel
        if not p.exists():
            log(f"missing: {rel}")
            continue
        a, meta = S934.read_czi(p)
        if S934.channel_roles(meta["channels"])[1] is None:
            continue
        lab = S934.labels(rel)
        if rel in S934.NO_MARROW:
            assigned, basis = "excluded (no marrow)", "morphology: " + S934.NO_MARROW[rel]
        else:
            assigned, basis = lab["compartment"], lab["compartment_basis"]
        m = measure_field(a, meta)
        if m is None:
            log(f"too little tissue / nuclei: {rel}")
            continue
        rows.append(dict(path=rel, unit="field", assigned=assigned, subregion=lab["subregion"], basis=basis,
                         um_per_px=meta["um_x"], **m))
        for k, (cx, cy, rad) in enumerate(S934.ISLANDS.get(rel, [])):
            yy, xx = np.ogrid[:a.shape[2], :a.shape[3]]
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
            for unit, reg in (("island", inside), ("field without island", ~inside)):
                mi = measure_field(a, meta, reg)
                if mi is not None:
                    rows.append(dict(path=rel, unit=unit, assigned="marrow" if unit == "island" else assigned,
                                     subregion="otic-capsule marrow island" if unit == "island" else lab["subregion"],
                                     basis="morphology (marrow island drawn on DAPI)" if unit == "island" else basis,
                                     um_per_px=meta["um_x"], **mi))
        log(f"{rel}: {assigned} ({basis[:40]}); {m['nuclei_per_mm2']:.0f} nuclei/mm2, nuclear share "
            f"{m['nuclear_area_share']:.2f}, NN {m['nn_median_um']:.1f} um")
    d = pd.DataFrame(rows)
    d["log10_nuclei_per_mm2"] = np.log10(d.nuclei_per_mm2)
    d["log10_nn_um"] = np.log10(d.nn_median_um)
    ref = d[(d.unit == "field") & d.basis.eq("name label") & d.assigned.isin(["marrow", "cochlea"])].dropna(subset=FEATURES)
    X, y = ref[FEATURES].to_numpy(), (ref.assigned == "marrow").astype(int).to_numpy()
    acc, fit = logistic_loo(X, y)
    ok = d[FEATURES].notna().all(axis=1)
    d.loc[ok, "p_marrow"] = fit.predict_proba(d.loc[ok, FEATURES].to_numpy())[:, 1]
    expect = d.assigned.map({"marrow": 1, "cochlea": 0, "excluded (no marrow)": 0})
    d["model_call"] = np.where(d.p_marrow >= 0.5, "marrow-like", "non-marrow-like")
    d["agrees"] = np.where(expect.notna() & d.p_marrow.notna(), (d.p_marrow >= 0.5).astype(float) == expect, np.nan)
    d["reference"] = d.index.isin(ref.index)
    chk = d[~d.reference & d.basis.str.startswith("morphology")]
    log(f"name-labelled reference: {len(ref)} fields ({int(y.sum())} marrow), leave-one-out accuracy {acc:.0%}")
    for r in chk.itertuples():
        log(f"  {r.unit} {Path(r.path).name}: assigned {r.assigned}; P(marrow) {r.p_marrow:.2f} -> {r.model_call}")
    sm = (d.groupby(["reference", "assigned", "unit"])
          .agg(fields=("path", "size"), nuclei_per_mm2_median=("nuclei_per_mm2", "median"),
               nuclei_per_mm2_min=("nuclei_per_mm2", "min"), nuclei_per_mm2_max=("nuclei_per_mm2", "max"),
               nn_um_median=("nn_median_um", "median"), p_marrow_median=("p_marrow", "median"),
               agreeing=("agrees", "sum")).reset_index())
    sm["model_loo_accuracy"] = acc
    sm["reference_fields"] = len(ref)
    write_csv(d, MOD, "tissue_assignment_fields.csv", "Tissue assignments of the parabiosis fields (942, D137): per "
              "field (and otic-capsule island) nuclei per mm2, nuclear area share, nearest-neighbour distance, the "
              "assignment and its basis (name label / morphology / excluded), the marrow probability of a logistic "
              "model trained on name-labelled fields, and agreement", required=["path", "unit", "assigned", "basis",
                                                                                "p_marrow", "agrees"])
    write_csv(sm, MOD, "tissue_assignment_summary.csv", "Tissue assignments (942, D137): per class (reference = "
              "name-labelled) field counts, nuclear density, nearest-neighbour distance, model marrow probability and "
              "agreement; leave-one-out accuracy of the model", required=["assigned", "fields", "model_loo_accuracy"])


if __name__ == "__main__":
    main()
