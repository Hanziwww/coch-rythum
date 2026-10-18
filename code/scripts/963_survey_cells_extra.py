"""Round 10 (D143) — the upgraded survey cell rule of 960 applied to the remaining wild-type tissue fields of the
parabiosis survey (vestibular-bone marrow, 54 days; liver, 31 days; temporal-bone marrow of the 5- and 7-day partners,
which have no cochlear fields; the reference tissues of the Supplementary tissue figure), so that every density shown
counts partner-derived cells by one rule. 960's own run (and its blinded
validation sample) is not repeated: its body_features function is applied here to these fields only. The bodies are
934's (count asserted per field). Liver fields are the rendered TIFF exports read as in 934.
Output (lib/io.write_csv, results/r10_parabiosis/): survey_cells_bodies_extra.csv (one row per 934 body with the
upgraded features and call).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/963_survey_cells_extra.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S960 = load_module("s960", "960_survey_cells.py")
S934 = S960.S934
MOD = "r10_parabiosis"


def read_field(rel):
    if rel.lower().endswith(".czi"):
        return S934.read_czi(S934.SRC / rel)
    stem = rel.rsplit("/", 1)[-1]                       # rendered liver export: c1 lectin, c2 GFP, c3 DAPI (934)
    d = S934.SRC / "2025-01-15-2" / "TIFF-25.1.15" / stem
    chs = [tifffile.imread(d / f"{stem}_c{k}.tif").astype(np.float32).max(axis=-1) for k in (1, 2, 3)]
    meta = dict(um_x=S934.LIVER_UM, um_z=None, channels=["Alexa Fluor 647", "Alexa Fluor 488", "DAPI"])
    return np.stack(chs)[:, None], meta


def main():
    tf = pd.read_csv(RESULTS / MOD / "parabiosis_tissue_fields.csv")
    ob = pd.read_csv(RESULTS / MOD / "parabiosis_objects.csv")
    done = set(pd.read_csv(RESULTS / MOD / "survey_cells_bodies.csv").path)          # fields analysed by 960
    w = tf[(tf.partner == "WT") & ~tf.path.isin(done) & (tf.gfp_bodies > 0)]
    assert set(zip(w.animal, w.compartment)) == {("54 d", "vestibular bone"), ("31 d", "liver"), ("5 d", "marrow"),
                                                 ("7 d", "marrow")}, sorted(set(zip(w.animal, w.compartment)))
    rows, store = [], {}
    for r in w.itertuples():
        a, meta = read_field(r.path)
        x = S960.body_features(a, meta, r.path, store, 255.0)
        assert len(x) == int((ob.path == r.path).sum()), (r.path, len(x))
        for d in x:
            d.update(animal=r.animal, compartment=r.compartment)
        rows += x
        log(f"{r.path}: {len(x)} bodies, {sum(d['cell'] for d in x)} cells by the upgraded rule")
    b = pd.DataFrame(rows).drop(columns=["_crop"])
    write_csv(b, MOD, "survey_cells_bodies_extra.csv", "Parabiosis survey (963, D143): GFP+ bodies of the "
              "wild-type partners' fields not analysed by 960 (vestibular bone, liver, 5- and 7-day marrow) with the "
              "upgraded features and cell call of 960", required=["path", "plane", "cell"])


if __name__ == "__main__":
    main()
