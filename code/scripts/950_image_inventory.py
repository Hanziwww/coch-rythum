"""Round 10 (D137) — inventory of the user's raw image folder $COCH_IMAGES for the 3D vessel and partner-cell analyses
(user: "There are a great many images in $COCH_IMAGES that can be used for tasks like vessel tracking; you should analyze
as many of them as possible"). $COCH_IMAGES is read only; nothing is copied into the project.

1  Every file: size, MD5, format and its metadata.
   - Imaris volumes (.ims, HDF5; precessed_imaris = Imaris-processed): voxel size from DataSetInfo/Image (ExtMax -
     ExtMin) / X, Y, Z; channel colours; the Imaris processing log (background subtraction, deconvolution, Gaussian,
     baseline, threshold, median filters: intensities are processed, not raw); saved scene (rendering mode, camera).
   - Zeiss CZI acquisitions: pixel size, z step, channels (dye), objective, zoom, date, microscope.
   - TIFF series: ImageJ metadata (slice label naming the source CZI and channel) or plain RGB frames.
   - precessed_imaris.7z: members listed (name, size, CRC32) and compared with the .ims files on disk (CRC32 of the
     disk file); identical members need no extraction.
   - SVFI_Config*.ini / task_restore.json: Squirrel video-frame-interpolation (SVFI) settings and input folder.
2  Provenance of every .ims: the source acquisition among the project's CZI collection
   (data/photos/parabiosis_all/Parabiosis_Mice; table results/r10_parabiosis/parabiosis_acquisitions.csv) is the
   candidate with the same voxel grid (X, Y, Z, pixel size) or the same name whose maximum-intensity projection (MIP)
   correlates best with the volume's MIP; each volume channel is mapped to the CZI channel (dye) it correlates with
   (Pearson r of the MIPs on the volume's 256-512-px pyramid level). A volume is linked when its best channel r >= 0.6.
   The name of the source CZI is compared with the volume's file name (name mismatches are reported, not corrected).
3  Identity checks: .ims pairs with identical voxel grids are compared voxel by voxel on a coarse level (identical /
   r); the 193-frame TIFF series are compared slice by slice with their source CZI channel; the 1,544-frame RGB
   series are compared with their SVFI input (every 8th frame vs the input frame): interpolated frames are not data.
   Raw CZIs of $COCH_IMAGES/raw are compared with the collection by name and size (MD5 where sizes match) and with each
   other (identical acquisition time, pixel data).
4  Suitability for this round: vessel label (channel role registry of 959), GFP (parabiont), nuclei (DAPI), marrow
   and cochlea in one field, calibration, axial resolution of the objective (nominal confocal axial FWHM
   0.88 x lambda / (n - sqrt(n^2 - NA^2)), lambda 0.67 um, n 1), and the use made of it (or the reason it is not used).
Outputs (lib/io.write_csv, results/r10_imaging_3d/): image_inventory.csv (one row per file), image_series.csv (one row
per volume / acquisition / series / archive), ims_provenance.csv (volume -> source CZI, channel map and r),
archive_members.csv; figures/panels/vessel3d_inventory_thumbs.png (Imaris thumbnails, for review).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/950_image_inventory.py   (local; $COCH_IMAGES is not on the server)
"""
import configparser
import hashlib
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IO = load_module("s959", "959_imaging_io.py")
MOD = IO.MOD
ACQ = RESULTS / "r10_parabiosis" / "parabiosis_acquisitions.csv"
LINK_R = 0.6
# whole-field content seen in the projections (marrow cavities of the temporal bone next to the cochlea); checked
# quantitatively by 952
MARROW_AND_COCHLEA = {"10x-0_45x-31d-NOISE-WT-GFP-LECTIN-sl-01", "10x-CD45-DAPI-PHALLOIDIN-WT-01", "10x-tile-gfp-594-405-02",
                      "14d-infect-gfp-vessal-10x-0_6x-apex-01", "14d-infect-gfp-vessal-5x-2um-02",
                      "5X-31d-pha-lectin-dapi-01", "5X-7d-lectin-GFP-01", "5x-31d-NOISE-WT-GFP-LECTIN-01",
                      "5x-lectin-gfp-wt-14D-01", "Cochlea", "Cochlear-14D-lectin-gfp-dapi-20x-01", "OUT-LINBATE-14D-12",
                      "fei-tile-z-647-01-01", "wmi19h-2-5", "WT_CD45_Lectin", "WT_phalloidin"}


def hashes(path: Path):
    h, c = hashlib.md5(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
            c = zlib.crc32(chunk, c)
    return h.hexdigest(), c


def axial_fwhm(na):
    try:
        na = float(na)
    except (TypeError, ValueError):
        return np.nan
    return 0.88 * 0.67 / (1 - np.sqrt(1 - na * na)) if 0 < na < 1 else np.nan


def norm_mip(a):
    m = a.astype(np.float32)
    lo, hi = np.percentile(m, 1), np.percentile(m, 99.5)
    return np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)


def block_mean(img, shape):
    """Resize a 2D image to `shape` by area averaging (PIL box filter)."""
    im = Image.fromarray(img.astype(np.float32))
    return np.asarray(im.resize((shape[1], shape[0]), Image.BOX), dtype=np.float32)


def norm_stem(s):
    return re.sub(r"-+", "-", s.lower())


def r_of(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------------------------------------ provenance
def ims_mips(path, h):
    lv = next(p["level"] for p in h["pyramid"] if p["valid"][2] <= 520)
    return lv, [norm_mip(IO.ims_read(path, c, lv, h).max(axis=0)) for c in range(h["n_channels"])]


def provenance(stem, path, h, acq, mips_cache):
    vx = h["voxel_um"][0]
    same_name = acq.path.map(lambda s: norm_stem(Path(s).stem) == norm_stem(stem))
    cand = acq[(np.isclose(acq.um_per_px, vx, rtol=0.01) & (acq.n_z == h["nz"])) | same_name]
    cand = cand[~cand.path.str.contains("Orthogonal|Snapshot|/Snap-", regex=True)]
    if not len(cand):
        return dict(stem=stem, source_czi="", candidates=0, linked=False), []
    lv, mips = mips_cache[stem]
    best, best_rows = None, []
    for _, c in cand.iterrows():
        cp = IO.COLL / c.path
        if not cp.exists():
            continue
        a = IO.czi_read(cp)
        cm = [norm_mip(block_mean(a[k].max(axis=0).astype(np.float32), mips[0].shape)) for k in range(a.shape[0])]
        names = c.channels.split(";") if isinstance(c.channels, str) else [f"C{k}" for k in range(a.shape[0])]
        rows = []
        for i, m in enumerate(mips):
            rs = [r_of(m, x) for x in cm]
            k = int(np.nanargmax(rs)) if np.isfinite(rs).any() else -1
            rows.append(dict(stem=stem, czi=c.path, ims_channel=i, czi_channel=k,
                             czi_dye=names[k] if 0 <= k < len(names) else "", r=rs[k] if k >= 0 else np.nan,
                             r_all=";".join(f"{x:.3f}" for x in rs)))
        score = max((r["r"] for r in rows if np.isfinite(r["r"])), default=-1)
        if best is None or score > best[0]:
            best = (score, c, rows, a.shape)
        best_rows += rows
        del a
    if best is None:
        return dict(stem=stem, source_czi="", candidates=len(cand), linked=False), best_rows
    score, c, rows, shp = best
    linked = score >= LINK_R
    src = c.path if linked else ""
    name_match = linked and norm_stem(Path(c.path).stem) == norm_stem(stem)
    return dict(stem=stem, source_czi=src, candidates=len(cand), linked=linked, best_r=score,
                czi_shape="x".join(map(str, shp)), czi_channels=c.channels, czi_animal=c.animal,
                czi_partner=c.partner_label, czi_compartment=c.compartment, czi_acquired=c.acquired,
                name_matches_source=bool(name_match),
                channel_map=";".join(f"ims{r['ims_channel']}->{r['czi_dye']} (r {r['r']:.2f})" for r in rows)), \
        best_rows


# ------------------------------------------------------------------------------------------------ main
def main():
    files = sorted(p for p in IO.IMAGES.rglob("*") if p.is_file())
    log(f"{len(files)} files under {IO.IMAGES}")
    acq = pd.read_csv(ACQ)
    acq = acq[acq.n_z.notna()].copy()
    rows = []
    for i, p in enumerate(files):
        rel = p.relative_to(IO.IMAGES).as_posix()
        md5, crc = hashes(p)
        rows.append(dict(file=rel, folder=str(Path(rel).parent.as_posix()), name=p.name, ext=p.suffix.lower(),
                         size_bytes=p.stat().st_size, md5=md5, crc32=crc))
        if i % 500 == 0:
            log(f"hashed {i}/{len(files)}")
    inv = pd.DataFrame(rows)

    # --- Imaris volumes
    ims_rows, mips_cache = {}, {}
    for p in sorted(IO.IMS_DIR.glob("*.ims")):
        h = IO.ims_header(p)
        stem = p.stem
        mips_cache[stem] = ims_mips(p, h)
        vx, vy, vz = h["voxel_um"]
        ims_rows[stem] = dict(format="Imaris 5.5 HDF5 (Imaris 10.2; processed)", nx=h["nx"], ny=h["ny"], nz=h["nz"],
                              nc=h["n_channels"], nt=1, dtype=h["pyramid"][0]["dtype"], unit=h["unit"],
                              voxel_x_um=vx if h["unit"] == "micron" else np.nan,
                              voxel_y_um=vy if h["unit"] == "micron" else np.nan,
                              voxel_z_um=vz if h["unit"] == "micron" else np.nan,
                              voxel_note="" if h["unit"] == "micron" else
                              f"uncalibrated (unit 'pixel'; z step {vz:.2f} x the xy pixel)",
                              extent_um="x".join(f"{e[1] - e[0]:.1f}" for e in h["ext"]),
                              pyramid=";".join(f"L{q['level']}:{q['valid'][2]}x{q['valid'][1]}x{q['valid'][0]}"
                                               for q in h["pyramid"]),
                              channels=";".join(f"ch{c['index']}:{c['color'] or c['color_mode']}"
                                                f"{(' em' + c['emission']) if c['emission'] else ''}"
                                                f"{(' ' + c['name']) if c['name'] != '(name not specified)' else ''}"
                                                for c in h["channels"]),
                              processing=" > ".join(h["log_ops"]) or "none recorded",
                              ims_image_name=h["image_name"], ims_recording_date=h["recording_date"],
                              render_mode=h["render_mode"], camera=h["camera"]["position"] + " " +
                              h["camera"]["orientation"])

    # --- provenance (.ims -> CZI)
    prov, chan_rows = [], []
    for stem in ims_rows:
        p = IO.IMS_DIR / f"{stem}.ims"
        h = IO.ims_header(p)
        pr, cr = provenance(stem, p, h, acq, mips_cache)
        prov.append(pr)
        chan_rows += cr
        log(f"provenance {stem}: {pr.get('source_czi') or '-'} (best r {pr.get('best_r', np.nan):.2f})")
    prov = pd.DataFrame(prov)
    chans = pd.DataFrame(chan_rows)

    # --- role registry vs source CZI dyes
    checks = []
    for _, r in prov[prov.linked].iterrows():
        sub = chans[(chans.stem == r.stem) & (chans.czi == r.source_czi)]
        for _, c in sub.iterrows():
            role = IO.ROLES[r.stem]["roles"].get(int(c.ims_channel), "")
            dye = c.czi_dye
            marker = IO.ROLES[r.stem]["marker"]
            expect = {"vessel": ("647",) if "lectin" in marker else ("594", "488"), "gfp": ("488", "FITC"),
                      "dapi": ("DAPI",), "phalloidin": ("488",)}
            ok = role in ("other", "coloc", "") or any(e in dye for e in expect.get(role, ()))
            checks.append(dict(stem=r.stem, ims_channel=int(c.ims_channel), role=role, czi_dye=dye, r=c.r,
                               consistent=ok))
    checks = pd.DataFrame(checks)
    bad = checks[~checks.consistent] if len(checks) else checks
    for _, b in bad.iterrows():
        log(f"ROLE CHECK: {b.stem} ch{b.ims_channel} role {b.role} vs CZI dye {b.czi_dye} (r {b.r:.2f})")

    # --- identical voxel grids among .ims (duplicates under different names)
    dup = {}
    stems = list(ims_rows)
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            ra, rb = ims_rows[a], ims_rows[b]
            if (ra["nx"], ra["ny"], ra["nz"], ra["nc"], ra["extent_um"]) != (rb["nx"], rb["ny"], rb["nz"], rb["nc"],
                                                                                rb["extent_um"]):
                continue
            ha, hb = IO.ims_header(IO.IMS_DIR / f"{a}.ims"), IO.ims_header(IO.IMS_DIR / f"{b}.ims")
            lv = 0
            same, rs = True, []
            for c in range(ha["n_channels"]):
                x = IO.ims_read(IO.IMS_DIR / f"{a}.ims", c, lv, ha)
                y = IO.ims_read(IO.IMS_DIR / f"{b}.ims", c, lv, hb)
                same &= bool(np.array_equal(x, y))
                rs.append(r_of(x.astype(np.float32), y.astype(np.float32)))
            note = ("identical voxels" if same else f"same grid, voxel r {', '.join(f'{v:.3f}' for v in rs)}") + \
                   f" (level {lv}); processing: {ims_rows[a]['processing'][:60]} | {ims_rows[b]['processing'][:60]}"
            dup.setdefault(a, []).append(f"{b}: {note}")
            dup.setdefault(b, []).append(f"{a}: {note}")
            log(f"grid twin {a} ~ {b}: {note}")

    # --- archive members vs disk
    import py7zr
    arch = []
    for z in sorted(IO.IMAGES.rglob("*.7z")):
        with py7zr.SevenZipFile(z, "r") as zf:
            for m in zf.list():
                disk = z.parent / m.filename
                drow = inv[inv.file == disk.relative_to(IO.IMAGES).as_posix()]
                arch.append(dict(archive=z.relative_to(IO.IMAGES).as_posix(), member=m.filename,
                                 member_bytes=m.uncompressed, member_crc32=m.crc32,
                                 disk_file=disk.relative_to(IO.IMAGES).as_posix() if disk.exists() else "",
                                 disk_bytes=int(drow.size_bytes.iloc[0]) if len(drow) else np.nan,
                                 disk_crc32=int(drow.crc32.iloc[0]) if len(drow) else np.nan))
    arch = pd.DataFrame(arch)
    arch["identical_on_disk"] = (arch.member_bytes == arch.disk_bytes) & (arch.member_crc32 == arch.disk_crc32)
    log(f"archive: {int(arch.identical_on_disk.sum())}/{len(arch)} members identical to the disk files")

    # --- CZIs (raw folder)
    czi_rows = {}
    for p in sorted(IO.IMAGES.rglob("*.czi")):
        m = IO.czi_meta(p)
        d = m["dims"]
        rel = p.relative_to(IO.IMAGES).as_posix()
        czi_rows[rel] = dict(format="Zeiss CZI (8-bit)", nx=d.get("X"), ny=d.get("Y"), nz=d.get("Z", 1), nc=d.get("C", 1),
                             nt=d.get("T", 1), dtype=m["dtype"], unit="micron", voxel_x_um=m["um_x"], voxel_y_um=m["um_y"],
                             voxel_z_um=m["um_z"] if d.get("Z", 1) > 1 else np.nan,
                             extent_um=f"{(m['um_x'] or 0) * d.get('X', 0):.1f}x{(m['um_y'] or 0) * d.get('Y', 0):.1f}",
                             channels=";".join(f"{c['fluor'] or c['name']} (ex {c['ex']} em {c['em']})"
                                               for c in m["channels"]),
                             objective=m["objective"], na=m["na"], zoom=m["zoom"], acquired=m["acquired"],
                             microscope=m["microscope"], software=m["software"],
                             axial_fwhm_um=axial_fwhm(m["na"]))
    # identical acquisition times within the raw folder
    acq_t = pd.Series({k: v["acquired"] for k, v in czi_rows.items()})
    same_t = {k: [j for j in acq_t.index if j != k and acq_t[j] == acq_t[k]] for k in acq_t.index}
    twin_note = {}
    for k, js in same_t.items():
        for j in js:
            a, b = IO.czi_read(IO.IMAGES / k), IO.czi_read(IO.IMAGES / j)
            if a.shape == b.shape:
                note = "identical pixels" if np.array_equal(a, b) else f"same shape, r {r_of(a.astype(np.float32), b.astype(np.float32)):.3f}"
            else:
                note = f"shapes {a.shape} vs {b.shape}"
            twin_note.setdefault(k, []).append(f"{j} (same acquisition time; {note})")

    # --- TIFF series and SVFI
    tif = inv[inv.ext.isin([".tif", ".tiff"])]
    series_rows, tif_meta = {}, {}
    for folder, g in tif.groupby("folder"):
        first = IO.IMAGES / g.file.iloc[0]
        with tifffile.TiffFile(first) as t:
            pg = t.pages[0]
            ij = t.imagej_metadata or {}
            xres = pg.tags.get("XResolution")
            um = (xres.value[1] / xres.value[0]) if xres and ij else np.nan
            label = (ij.get("Properties", {}) or {}).get("Slice_Label", "") if isinstance(ij, dict) else ""
            series_rows[folder] = dict(format="ImageJ TIFF sequence (8-bit palette)" if ij else
                                       "RGB TIFF frames (8-bit x 3)", nx=pg.shape[1], ny=pg.shape[0],
                                       nz=len(g), nc=1 if pg.ndim == 2 else pg.shape[2], dtype=str(pg.dtype),
                                       voxel_x_um=um, voxel_y_um=um, slice_label=label)
    svfi = {}
    for ini in IO.IMAGES.rglob("SVFI_Config*.ini"):
        cp = configparser.ConfigParser(interpolation=None, strict=False)
        cp.read(ini, encoding="utf-8-sig")
        gsec = cp["General"] if "General" in cp else {}
        tr = json.loads((ini.parent / "task_restore.json").read_text(encoding="utf-8")) if \
            (ini.parent / "task_restore.json").exists() else {}
        svfi[ini.parent.relative_to(IO.IMAGES).as_posix()] = dict(
            input_fps=gsec.get("input_fps"), target_fps=gsec.get("target_fps"), rife_exp=gsec.get("rife_exp"),
            version=gsec.get("version"), input_path=tr.get("input_path", ""), output_ext=gsec.get("output_ext"))
    # TIFF series vs source CZI (slice-by-slice) and SVFI frames vs their input
    tiff_checks = {}
    for folder, s in series_rows.items():
        m = re.search(r"c:(\d+)/(\d+) z:\d+/(\d+) - (.+?\.czi)", s["slice_label"])
        if m:
            ch, czi_name = int(m.group(1)) - 1, m.group(4).replace(" #1", "")
            src = acq[acq.path.str.endswith("/" + czi_name)]
            if len(src):
                a = IO.czi_read(IO.COLL / src.path.iloc[0])
                fr = sorted(tif[tif.folder == folder].file)
                zt = (0, len(fr) // 2, len(fr) - 1)
                eq = [np.array_equal(tifffile.imread(IO.IMAGES / fr[z]), a[ch, z]) for z in zt]
                rs = [r_of(tifffile.imread(IO.IMAGES / fr[z]).astype(np.float32), a[ch, z].astype(np.float32)) for z in zt]
                tiff_checks[folder] = (f"channel {ch + 1} of {src.path.iloc[0]} (collection): {sum(eq)}/3 test slices "
                                       f"identical, r {', '.join(f'{x:.3f}' for x in rs)}" +
                                       ("" if all(eq) else " (display-scaled export of the same acquisition)"))
                del a
    for folder, cfg in svfi.items():
        inp_name = Path(cfg["input_path"].replace("\\", "/")).name
        inp = next((f for f in set(tif.folder) if Path(f).name == inp_name), inp_name)
        frames = sorted(tif[tif.folder == folder].file)
        src = sorted(tif[tif.folder == inp].file)
        k = int(cfg["target_fps"] or 8) // max(int(cfg["input_fps"] or 1), 1)
        rs = []
        for j in (0, len(src) // 3, 2 * len(src) // 3):
            if j * k < len(frames) and src:
                fr = tifffile.imread(IO.IMAGES / frames[j * k]).astype(np.float32).mean(axis=-1)
                rs.append(r_of(fr, tifffile.imread(IO.IMAGES / src[j]).astype(np.float32)))
        mid = []
        for j in (0, len(src) // 3):
            if j * k + k // 2 < len(frames) and src:
                fr = tifffile.imread(IO.IMAGES / frames[j * k + k // 2]).astype(np.float32).mean(axis=-1)
                mid.append(r_of(fr, tifffile.imread(IO.IMAGES / src[j]).astype(np.float32)))
        tiff_checks[folder] = (f"SVFI {cfg['version']} RIFE interpolation of {inp} ({cfg['input_fps']} -> "
                               f"{cfg['target_fps']} fps: {len(src)} input -> {len(frames)} frames); frame {k}j vs input "
                               f"j r {', '.join(f'{x:.2f}' for x in rs)}; mid-interval frames r "
                               f"{', '.join(f'{x:.2f}' for x in mid)}: synthetic frames, not data")

    # --- duplicates of project images (data/photos), by name and size
    proj = []
    for p in (ROOT / "data" / "photos").rglob("*"):
        if p.is_file():
            proj.append((p.name.lower(), p.stat().st_size, p))
    proj = pd.DataFrame(proj, columns=["lname", "size", "path"])
    by_size = {s: list(g.path) for s, g in proj.groupby("size")}
    md5_cache, dup_proj = {}, []
    for _, r in inv.iterrows():
        found = ""
        for pp in by_size.get(r.size_bytes, []):
            if pp not in md5_cache:
                md5_cache[pp] = hashes(pp)[0]
            if md5_cache[pp] == r.md5:
                found = pp.relative_to(ROOT).as_posix()
                break
        dup_proj.append(found)
    inv["identical_in_project"] = dup_proj
    log(f"project files hashed for size matches: {len(md5_cache)}; identical copies: {sum(bool(x) for x in dup_proj)}")

    # --- assemble per-file and per-series tables
    def series_of(r):
        if r.ext == ".ims":
            return "ims:" + Path(r["name"]).stem
        if r.ext == ".czi":
            return "czi:" + r.file
        if r.ext in (".tif", ".tiff") or r["name"].startswith("SVFI_Config") or r["name"] == "task_restore.json" \
                or r["name"] == "scene":
            return "tiff:" + r.folder
        return "file:" + r.file
    inv["series"] = inv.apply(series_of, axis=1)
    roles = {s: IO.ROLES[s] for s in IO.ROLES}
    ser = []
    for sid, g in inv.groupby("series", sort=False):
        kind, key = sid.split(":", 1)
        row = dict(series=sid, files=len(g), bytes=int(g.size_bytes.sum()), example_file=g.file.iloc[0])
        if kind == "ims":
            stem = key
            row.update(ims_rows[stem])
            pr = prov[prov.stem == stem].iloc[0].to_dict()
            row.update(source_czi=pr.get("source_czi", ""), source_r=pr.get("best_r", np.nan),
                       source_name_matches=pr.get("name_matches_source", np.nan),
                       channel_map=pr.get("channel_map", ""), source_animal=pr.get("czi_animal", ""),
                       grid_twins=" | ".join(dup.get(stem, [])))
            row.update(IO.parse_name(stem))
            src_stem = Path(str(row["source_czi"])).stem if row["source_czi"] else ""
            an = pr.get("czi_animal")
            row.update(animal=an if isinstance(an, str) and an else ("discarded mouse (folder label)" if
                                                                     "Discarded" in str(row["source_czi"]) else ""),
                       partner="WT" if isinstance(an, str) and an else "",
                       source_region_label=IO.parse_name(src_stem)["region_label"] if src_stem else "")
            rr = roles.get(stem, {})
            row.update(channel_roles=";".join(f"ch{k}:{v}" for k, v in rr.get("roles", {}).items()),
                       vessel_marker=rr.get("marker", ""))
            src = acq[acq.path == row["source_czi"]]
            arow = arch[arch.member == f"{stem}.ims"]
            row.update(in_archive_identical=bool(arow.identical_on_disk.iloc[0]) if len(arow) else False)
            row.update(marrow_and_cochlea_in_field=stem in MARROW_AND_COCHLEA)
        elif kind == "czi":
            row.update(czi_rows[key])
            row.update(acquisition_twins=" | ".join(twin_note.get(key, [])))
            row.update(IO.parse_name(Path(key).stem))
        elif kind == "tiff":
            s = series_rows.get(key, {})
            row.update(s)
            row.update(svfi_config=json.dumps(svfi.get(key, {})) if key in svfi else "",
                       series_check=tiff_checks.get(key, ""))
        else:
            row.update(format="7z archive" if g.ext.iloc[0] == ".7z" else g.ext.iloc[0])
        ser.append(row)
    ser = pd.DataFrame(ser)
    if "axial_fwhm_um" not in ser:
        ser["axial_fwhm_um"] = np.nan
    for col in ("objective", "na", "zoom", "acquired", "microscope"):
        if col not in ser:
            ser[col] = ""
        ser[col] = ser[col].astype(object)

    # objective of every .ims from its source CZI (pixel size and NA)
    for i, r in ser[ser.series.str.startswith("ims:")].iterrows():
        if r.source_czi:
            m = IO.czi_meta(IO.COLL / r.source_czi)
            ser.loc[i, ["objective", "na", "zoom", "acquired", "microscope"]] = [m["objective"], m["na"], m["zoom"],
                                                                               m["acquired"], m["microscope"]]
            ser.loc[i, "axial_fwhm_um"] = axial_fwhm(m["na"])

    # suitability and use
    def use(r):
        sid = r.series
        if sid.startswith("ims:"):
            stem = sid[4:]
            rr = roles.get(stem, {}).get("roles", {})
            has_v = "vessel" in rr.values()
            has_g = "gfp" in rr.values()
            twins = str(r.grid_twins)
            if "identical voxels" in twins and stem in ("Cochlear-14D-lectin-gfp-dapi-20x-01", "OUT-LINBATE-14D-12"):
                return "duplicate of Cochlea.ims (identical voxels): not analysed twice"
            uses = []
            if has_v:
                uses.append("vessel network (951)")
            if r.marrow_and_cochlea_in_field and has_v:
                uses.append("bone-cochlea connections (952)")
            if has_g and has_v and str(r.source_animal) not in ("", "nan"):
                uses.append("GFP+ cells vs vessels (953)")
            if not uses:
                return ("not used: no vessel channel (" + (r.vessel_marker or "markers " + r.channel_roles) + ")")
            return "; ".join(uses)
        if sid.startswith("czi:"):
            if "DAPI-in-out-bone" in sid:
                return "free-cell preparation (nuclei, 594 membrane ring): 954"
            return "raw 594 + DAPI z-stack (594 vessel-shaped; marker not recorded): vessel network (951)"
        if sid.startswith("tiff:"):
            if "SVFI" in str(r.series_check):
                return "not used: video-frame interpolation output (synthetic frames)"
            if r.series_check:
                return "not used: unprocessed export of a CZI channel already in the collection (same pixels)"
        if sid.endswith(".7z"):
            return "not extracted: every member identical to the .ims on disk" if arch.identical_on_disk.all() else \
                "contains members not on disk"
        return "not used"
    ser["use"] = ser.apply(use, axis=1)
    inv = inv.merge(ser[["series", "format", "use"]], on="series", how="left")

    # contact sheet of thumbnails (review)
    stems = sorted(ims_rows)
    S, cols = 200, 7
    sheet = Image.new("RGB", (cols * S, ((len(stems) + cols - 1) // cols) * (S + 24)), "white")
    d = ImageDraw.Draw(sheet)
    for i, s in enumerate(stems):
        t = Image.fromarray(IO.ims_thumbnail(IO.IMS_DIR / f"{s}.ims").astype(np.uint8)).resize((S, S))
        x, y = (i % cols) * S, (i // cols) * (S + 24)
        sheet.paste(t, (x, y + 24))
        d.text((x + 2, y + 1), s[:34], fill="black")
        d.text((x + 2, y + 12), s[34:68], fill="black")
    IO.PANELS.mkdir(parents=True, exist_ok=True)
    sheet.save(IO.PANELS / "vessel3d_inventory_thumbs.png")

    write_csv(inv, MOD, "image_inventory.csv", "Imaging inventory (950): every file under $COCH_IMAGES with size, MD5, "
              "CRC32, format, series, identical copy in data/photos and the use made of it",
              required=["file", "size_bytes", "md5", "series", "use"])
    write_csv(ser, MOD, "image_series.csv", "Imaging inventory (950): one row per Imaris volume, CZI acquisition, TIFF "
              "series or archive with dimensions, voxel size, channels, processing, source acquisition, parsed sample "
              "attributes, channel roles, suitability and use", required=["series", "files", "use"])
    write_csv(prov.merge(checks.groupby("stem").consistent.all().rename("roles_consistent").reset_index(), on="stem",
                         how="left") if len(checks) else prov, MOD, "ims_provenance.csv",
              "Imaging inventory (950): source CZI of every Imaris volume (pixel-data match), channel map with r, name "
              "consistency", required=["stem", "source_czi", "linked"])
    write_csv(chans, MOD, "ims_provenance_channels.csv", "Imaging inventory (950): Pearson r of every volume channel "
              "MIP with every channel MIP of each candidate source CZI", required=["stem", "czi", "ims_channel", "r"])
    write_csv(arch, MOD, "archive_members.csv", "Imaging inventory (950): members of precessed_imaris.7z vs the files "
              "on disk (size, CRC32)", required=["member", "identical_on_disk"])
    print(ser[["series", "use"]].to_string())


if __name__ == "__main__":
    main()
