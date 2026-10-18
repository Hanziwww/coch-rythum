"""Prepare this release for re-running the analysis scripts (Python standard library only).

The scripts run from the project root (this release folder) and expect scripts/ there: R scripts source
scripts/lib/cosinor.R and the shell chains call scripts/... The imaging scripts (93x) read folders below data/photos/
and the flow scripts data/flow_cochlea/. This helper
  1. links scripts/ to code/scripts (symbolic link; a directory junction or a copy on Windows),
  2. with --flow: copies the flow export to data/flow_cochlea/ and data/Cochlear_Flow_Plotting_Only.zip,
  3. with --images: extracts the image archives to the folders the imaging scripts read (about 60 GB),
  4. copies the Fig. 1a illustration to data/photos/fig1a_workflow/ (read by 910).
Usage, from the release folder: python code/prepare_workspace.py [--flow] [--images]
"""
import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(os.environ.get("COCH_ROOT", Path(__file__).resolve().parents[1]))
IMAGES = [
    ('data/photos/FUT4_WT_2026-10-12/', 'data/imaging/FUT4-Neutrophils-English.zip', 'FUT4-Neutrophils/WT-Celltype/2026.10.12-WT-Celltype-FUT4/'),
    ('data/photos/parabiosis_54d/', 'data/imaging/FUT4-Neutrophils-English.zip', 'FUT4-Neutrophils/Parabiotic_Mouse_Section_Staining-Immune_Cell_Types/'),
    ('data/photos/cochlea_3d/', 'data/imaging/cochlea_3d.zip', ''),
    ('data/photos/parabiosis_all/Parabiosis_Mice/', 'data/imaging/Parabiosis_Mice_ALL.zip', 'Parabiosis_Mice/'),
]
FLOW_FILES = ['README.txt', 'channels.csv', 'events.csv', 'gates.csv', 'sample_values.csv', 'tube_values.csv']
FLOW_ARCHIVE = "data/flow/Cochlear_Flow_Plotting_Only.zip"
FIG1A = ("figures/illustrations/fig1a_workflow.png", "data/photos/fig1a_workflow/fig1a_workflow_pasted.png")


def link_scripts():
    dst, src = ROOT / "scripts", ROOT / "code" / "scripts"
    if dst.exists():
        print("scripts/ exists")
        return
    try:
        dst.symlink_to(Path("code") / "scripts", target_is_directory=True)
    except OSError:
        if not (os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                                                   capture_output=True).returncode == 0):
            shutil.copytree(src, dst)
    print("scripts/ ->", src)


def extract(archive, inner, target):
    target = (ROOT / target).resolve()
    n = 0
    with zipfile.ZipFile(ROOT / archive) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(inner):
                continue
            rel = PurePosixPath(info.filename[len(inner):])
            if ".." in rel.parts or rel.is_absolute():
                continue
            out = target / Path(*rel.parts)
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 22)
            n += 1
    print(f"{archive} -> {target}: {n} files extracted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flow", action="store_true", help="copy the flow export to data/flow_cochlea")
    ap.add_argument("--images", action="store_true", help="extract the image archives (about 60 GB)")
    a = ap.parse_args()
    link_scripts()
    src, dst = (ROOT / FIG1A[0], ROOT / FIG1A[1])
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if a.flow:
        for f in FLOW_FILES:
            d = ROOT / "data" / "flow_cochlea" / f
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "data" / "flow" / f, d)
        shutil.copy2(ROOT / FLOW_ARCHIVE, ROOT / "data" / Path(FLOW_ARCHIVE).name)
        print("flow export -> data/flow_cochlea")
    if a.images:
        for target, archive, inner in IMAGES:
            extract(archive, inner, target)


if __name__ == "__main__":
    main()
