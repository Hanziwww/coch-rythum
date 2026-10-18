"""Module 1, Stage A selection — choose the feature/normalisation configuration per family.

Composite = 0.7 · bio (score_bio, min–max scaled across configs) + 0.3 · time (mean of min–max scaled
pcr_time_raw and sample_purity_excess_raw). Writes results/20_dr/stageA_selection.csv and prints the winner.
Usage: python scripts/20b_select_features.py <family>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, write_csv  # noqa: E402

fam = sys.argv[1]
m = pd.read_csv(RESULTS / f"20_dr/dr_benchmark_metrics_{fam}.csv")
w = m.pivot_table(index="embedding", columns="metric", values="value")
w = w[w.index.str.startswith("A__")]
mm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0 + 1
w["bio_scaled"] = mm(w["score_bio"])
w["time_scaled"] = (mm(w["pcr_time_raw"]) + mm(w["sample_purity_excess_raw"])) / 2
w["stageA_composite"] = 0.7 * w["bio_scaled"] + 0.3 * w["time_scaled"]
w = w.sort_values("stageA_composite", ascending=False)
w.insert(0, "family", fam)
w["selected"] = False
w.iloc[0, w.columns.get_loc("selected")] = True
out = w.reset_index()[["family", "embedding", "score_bio", "pcr_time_raw", "sample_purity_excess_raw",
                       "bio_scaled", "time_scaled", "stageA_composite", "selected"]]
path = RESULTS / "20_dr/stageA_selection.csv"
prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
if len(prev):
    prev = prev[prev["family"] != fam]
write_csv(pd.concat([prev, out], ignore_index=True), "20_dr", "stageA_selection.csv",
          "Stage A feature-set composite scores and selection per family",
          required=["family", "embedding", "stageA_composite", "selected"])
print(w.index[0])
