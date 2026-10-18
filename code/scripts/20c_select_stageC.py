"""Module 1 helper — pick Stage C candidates from the benchmark metrics (pre-registered composite and time-loss rule).

Usage:
  python 20c_select_stageC.py <family> top2              -> prints the two best Stage-B methods (method names)
  python 20c_select_stageC.py <family> bestnlat <method> -> prints the best n_latent among C__<method>_nl*_s0
Composite: per-tissue 0.7·bio + 0.3·time; joint 0.5·bio + 0.2·platform + 0.3·time; time-score < 0.8 rejected;
PCA is excluded from Stage C (no tuning parameters beyond n_pcs).
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS  # noqa: E402

fam, mode = sys.argv[1], sys.argv[2]
m = pd.read_csv(RESULTS / f"20_dr/dr_benchmark_metrics_{fam}.csv")
w = m.pivot_table(index="embedding", columns="metric", values="value")
w["composite"] = (0.5 * w["score_bio"] + 0.2 * w["score_platform"] + 0.3 * w["score_time"]) if fam == "joint" \
    else (0.7 * w["score_bio"] + 0.3 * w["score_time"])
w = w[w["score_time"] >= 0.8]
if mode == "top2":
    b = w[w.index.str.startswith("B__") & ~w.index.str.contains("__pca_")].sort_values("composite", ascending=False)
    names = [re.match(r"B__(.+)_nl\d+_s\d+", e).group(1) for e in b.index[:2]]
    print(" ".join(names))
else:
    meth = sys.argv[3]
    c = w[w.index.str.match(rf"C__{re.escape(meth)}_nl\d+_s0$")].sort_values("composite", ascending=False)
    print(re.search(r"_nl(\d+)_s0", c.index[0]).group(1) if len(c) else "30")
