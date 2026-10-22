"""Round 9 writing, phase G — Source Data for the Nature Neuroscience submission (main figures, Extended Data figures and
Supplementary Figs 1-4; parabiosis Figs 1-3 and flow gating 4 since D133). D143: the Supplementary Figures are numbered
by first citation in the Results (SI_NUMBER maps the panel-index IDs to the published numbers; retired figures are
skipped), and stale Source Data folders and archives of renumbered or retired figures are removed.

For every figure, every panel of results/r9_figures/panel_index.csv gets one CSV per source table it uses
(<Fig><panel>_<table>.csv), written with lib/io.write_csv to results/r9_source_data/<Fig>/. lib/pubclean drops internal
provenance and display columns (source_table, source_file, finding_id, register_id, call_set_finding, transform, colour,
class_colour, order, plot_order, class_order) and removes internal references from text cells (asserted clean);
the shared flow-plot tables are restricted to the plots of that figure. A panel-to-file index (<Fig>_panel_files.csv)
is written alongside. Each figure is then packed with its image panels (figures/panels, as displayed) into
<manuscript>/submission/source_data/Source_Data_<Fig>.zip (config paths.manuscript), and the same tables are written
as one Excel workbook per figure, Source_Data_<Fig>.xlsx (D134: an 'index' sheet mapping panels to sheets, then one
sheet per panel table whose first row names the figure, panel and table).
Run (local; figures/panels are local): COCH_ROOT=D:/coch-rythum python scripts/909_source_data.py
"""
import glob
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import pubclean  # noqa: E402
from lib.io import MANUSCRIPT, RESULTS, ROOT, log, write_csv  # noqa: E402

R9 = RESULTS / "r9_figures"
OUT_ZIP = MANUSCRIPT / "submission" / "source_data"
FLOW = {"flow_plots.csv", "flow_contours.csv", "flow_outliers.csv", "flow_gates.csv"}
# D143: panel-index IDs of the Supplementary Figures -> published number (1 vessel-trace check, 2-4 parabiosis survey,
# 5 brain and the brain-image format (STC, D152), 6 cell types, 7 robustness and verification, 8 flow gating); SIMG2
# (fluid images, sfig_fluids_v1) was retired
SI_NUMBER = {"SIMG1": 1, "S1": 2, "S2": 3, "S3": 4, "STC": 5, "S4": 6, "SROB": 7, "S5": 8}
SI_RETIRED = {"SIMG2"}


def tag_of(fig: str):
    if fig.startswith("ED"):
        return f"ED_Fig{fig[2:]}", f"Extended Data Fig. {fig[2:]}"
    if fig.startswith("S"):
        n = SI_NUMBER[fig]
        return f"Supplementary_Fig{n}", f"Supplementary Fig. {n}"
    return f"Fig{fig}", f"Fig. {fig}"


def flow_label(fig: str, label: str) -> str:
    """Figure label used in flow_plots.csv (internal panel-index numbering for the Supplementary gating figure)."""
    return f"Supplementary Fig. {fig[1:]}" if fig.startswith("S") else label


def flow_subset(name, label, panel, plots):
    df = pd.read_csv(R9 / name)
    if name == "flow_plots.csv":
        keep = df["figure"].isin([f"{label}{panel}", label] if not label.startswith("Supp") else [label])
        return df[keep]
    return df[df["plot"].isin(plots)]


def main():
    pi = pd.read_csv(R9 / "panel_index.csv")
    pi["figure"] = pi["figure"].astype(str)
    pi = pi[~pi["figure"].isin(SI_RETIRED)]
    fp = pd.read_csv(R9 / "flow_plots.csv")
    OUT_ZIP.mkdir(parents=True, exist_ok=True)
    tags = {tag_of(f)[0] for f in pi["figure"].unique()}
    for d in (RESULTS / "r9_source_data").glob("*"):                  # renumbered or retired figures (D143)
        if d.is_dir() and d.name not in tags:
            for f in d.glob("*.csv"):
                f.unlink()
            d.rmdir()
    for f in list(OUT_ZIP.glob("Source_Data_*.zip")) + list(OUT_ZIP.glob("Source_Data_*.xlsx")):
        if f.stem.replace("Source_Data_", "") not in tags:
            f.unlink()
    total = 0
    for fig, sub in pi.groupby("figure", sort=False):
        tag, label = tag_of(fig)
        module = f"r9_source_data/{tag}"
        for old in (RESULTS / module).glob("*.csv"):            # no stale tables when figures are renumbered (D133)
            old.unlink()
        index_rows, written, images, sheets = [], [], [], []
        for r in sub.sort_values("panel").itertuples():
            files = [f.strip() for f in str(r.files).split(";") if f.strip() and f.strip() != "nan"]
            for f in files:
                if f.endswith(".png") or "*" in f:
                    hits = sorted(glob.glob(str(ROOT / f)))
                    assert hits, (fig, r.panel, f)
                    images += hits
                    index_rows += [(r.panel, "image (as displayed)", Path(h).name) for h in hits]
                    continue
                name = Path(f).name
                if name in FLOW:
                    fl = flow_label(fig, label)
                    plots = fp.loc[fp["figure"].isin([f"{fl}{r.panel}", fl]), "plot"].tolist()
                    df = flow_subset(name, fl, r.panel, plots)
                    assert len(df), (fig, r.panel, name)
                else:
                    src = R9 / name if (R9 / name).exists() else RESULTS / "r9_flow" / name
                    df = pd.read_csv(src, low_memory=False)
                df = pubclean.clean(df, name)
                left = pubclean.residual(df)
                assert not left, (fig, r.panel, name, left[:3])
                out = f"{tag}{r.panel}_{Path(name).stem}.csv"
                p = write_csv(df, module, out, f"Source Data {label}{r.panel}: {name} (909; internal provenance and display "
                              "columns dropped, internal references removed by lib/pubclean)")
                written.append(p)
                index_rows.append((r.panel, "table", out))
                sheets.append((r.panel, out, df))
        idx = pd.DataFrame(index_rows, columns=["panel", "kind", "file"]).drop_duplicates()
        written.append(write_csv(idx, module, f"{tag}_panel_files.csv", f"Source Data {label}: panel-to-file index (909)",
                                 required=["panel", "kind", "file"]))
        zp = OUT_ZIP / f"Source_Data_{tag}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in written:
                z.write(p, arcname=f"Source_Data_{tag}/{Path(p).name}")
            for im in sorted(set(images)):
                z.write(im, arcname=f"Source_Data_{tag}/images/{Path(im).name}")
        xp = OUT_ZIP / f"Source_Data_{tag}.xlsx"
        names = [f"{panel}_{k:02d}" for k, (panel, _, _) in enumerate(sheets, 1)]
        with pd.ExcelWriter(xp, engine="openpyxl") as xw:
            pd.DataFrame([(panel, n, out) for (panel, out, _), n in zip(sheets, names)] +
                         [(panel, "(zip)", f) for panel, kind, f in index_rows if kind != "table"],
                         columns=["panel", "sheet", "table or image"]).to_excel(xw, sheet_name="index", index=False)
            for (panel, out, df), n in zip(sheets, names):
                assert len(df) < 1_000_000, (fig, out, len(df))
                pd.DataFrame([[f"Source Data {label}{panel}: {Path(out).stem}"]]).to_excel(
                    xw, sheet_name=n, index=False, header=False)
                df.to_excel(xw, sheet_name=n, index=False, startrow=2)
        total += len(written)
        log(f"{label}: {len(written) - 1} tables, {len(set(images))} images -> {zp.name} ({zp.stat().st_size / 1e6:.1f} MB)")
    log(f"Source Data: {total} CSV files in {pi['figure'].nunique()} figures")


if __name__ == "__main__":
    main()
