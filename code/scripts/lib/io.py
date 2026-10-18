"""Paths, config, sample metadata and schema-checked CSV output shared by all analysis scripts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(SCRIPTS / "config.yaml") as fh:
        return yaml.safe_load(fh)


CFG = load_config()
ROOT = Path(os.environ.get("COCH_ROOT", CFG["paths"]["root"]))
H5AD = ROOT / CFG["paths"]["h5ad"]
RESULTS = ROOT / CFG["paths"]["results"]
CACHE = ROOT / CFG["paths"]["cache"]
EXTERNAL = ROOT / CFG["paths"]["external"]
MANUSCRIPT = ROOT / CFG["paths"].get("manuscript", "manuscript/nn")   # active manuscript sources (LaTeX)
JOURNAL = CFG.get("journal", {})
ZT_VALUES = [float(z) for z in CFG["design"]["zt_values"]]
LIGHTS_ON = float(CFG["design"]["lights_on_clock_h"])


def clock_to_zt(clock_h):
    """Wall-clock sampling hour -> Zeitgeber time (h after lights-on), in [0, 24)."""
    return (np.asarray(clock_h, dtype=float) - LIGHTS_ON) % 24
TISSUE_CODES = CFG["design"]["tissue_codes"]            # code -> obs value
TISSUE_FROM_OBS = {v: k for k, v in TISSUE_CODES.items()}  # obs value -> code
LABEL_COLS = CFG["design"]["label_cols"]
MANIFEST = CACHE / "manifest_entries.jsonl"


def result_dir(module: str) -> Path:
    d = RESULTS / module
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(module: str) -> Path:
    d = CACHE / module
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_table(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per sample: sample_id (= batch_id), tissue code, clock hour, ZT, platform, n_cells.

    ZT is derived from the wall-clock sampling hour (obs["day_time"]) with lights on at LIGHTS_ON.
    Sorted by tissue then ZT.
    """
    g = obs.groupby("batch_id", observed=True)
    clock = g["day_time"].first().astype(float)
    st = pd.DataFrame({
        "tissue": g["tissue"].first().astype(str).map(TISSUE_FROM_OBS),
        "clock_time": clock,
        "zt": (clock - LIGHTS_ON) % 24,
        "platform": g["platform"].first().astype(str),
        "n_cells": g.size(),
    })
    st.index.name = "sample_id"
    assert g["day_time"].nunique().eq(1).all() and g["tissue"].nunique().eq(1).all()
    assert st["tissue"].notna().all(), "unknown tissue value"
    st = st.reset_index().sort_values(["tissue", "zt"]).reset_index(drop=True)
    for t, sub in st.groupby("tissue"):
        assert sorted(sub["zt"]) == ZT_VALUES, (t, sorted(sub["zt"]))
    return st


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, module: str, name: str, description: str,
              required: list[str] | None = None, index: bool = False) -> Path:
    """Write a deliverable CSV, check required columns, and register it in the results manifest."""
    if required:
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"{name}: missing required columns {missing}"
    path = result_dir(module) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    entry = {
        "path": str(path.relative_to(ROOT)), "rows": int(len(df)), "n_columns": int(df.shape[1] + int(index)),
        "columns": ([df.index.name or "index"] if index else []) + [str(c) for c in df.columns],
        "md5": _md5(path), "script": Path(sys.argv[0]).name, "description": description,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"[csv] {entry['path']}  rows={entry['rows']}  cols={entry['n_columns']}")
    return path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
