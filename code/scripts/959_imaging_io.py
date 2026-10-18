"""Round 10 (D137) — shared readers and registry for the $COCH_IMAGES analyses (scripts 950-958). Not a pipeline step:
loaded by the 95x scripts with importlib (as 939 loads 931 / 934 / 938).

Contents
- paths: the user's raw image folder (read-only; COCH_IMAGES, default $COCH_IMAGES), the project's CZI collection
  (data/photos/parabiosis_all/Parabiosis_Mice), module and panel locations, the cache folder for masks;
- Imaris (.ims, HDF5) readers: header, valid size of every pyramid level (the stored arrays are padded to the chunk
  grid), voxel size from DataSetInfo/Image ExtMin / ExtMax and X / Y / Z, processing log, saved camera, thumbnail;
- Zeiss CZI reader (czifile) with the metadata used here (pixel size, channels, objective, zoom, date, microscope);
- name parsing (duration, partner, noise, region, magnification) and the channel-role registry of the 33 volumes
  (ROLES): the vessel channel (lectin, Alexa Fluor 647 in the source CZIs; phalloidin or an unidentified marker where
  noted), GFP, DAPI. Roles of volumes with a source CZI are checked against the CZI channels by 950 (correlation of the
  projections); the others were assigned from the projections (tubular network = vessel channel).
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import ROOT, cache_dir  # noqa: E402

IMAGES = Path(os.environ.get("COCH_IMAGES", "images"))
IMS_DIR = IMAGES / "precessed_imaris"
RAW_DIR = IMAGES / "raw"
COLL = ROOT / "data" / "photos" / "parabiosis_all" / "Parabiosis_Mice"
PANELS = ROOT / "figures" / "panels"
MOD = "r10_imaging_3d"
GRID_UM = 2.773            # primary analysis grid (isotropic): the native pixel of the 5x/0.16 acquisitions


def cache(sub: str = "") -> Path:
    d = cache_dir(MOD) / sub if sub else cache_dir(MOD)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------ Imaris
def _a(v):
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return b"".join(v).decode("latin1")
    if isinstance(v, bytes):
        return v.decode("latin1")
    return str(v)


def ims_header(path: Path) -> dict:
    """Header of an Imaris file: sizes, extents, voxel size, channels, pyramid, log, camera."""
    with h5py.File(path, "r") as f:
        im = {k: _a(v) for k, v in f["DataSetInfo"]["Image"].attrs.items()}
        nx, ny, nz = int(im["X"]), int(im["Y"]), int(im["Z"])
        ext = [(float(im[f"ExtMin{i}"]), float(im[f"ExtMax{i}"])) for i in range(3)]
        vox = ((ext[0][1] - ext[0][0]) / nx, (ext[1][1] - ext[1][0]) / ny, (ext[2][1] - ext[2][0]) / nz)
        ds = f["DataSet"]
        levels = sorted(ds.keys(), key=lambda s: int(s.split()[-1]))
        tp0 = ds[levels[0]]["TimePoint 0"]
        chs = sorted(tp0.keys(), key=lambda s: int(s.split()[-1]))
        pyr = []
        for L in levels:
            g = ds[L]["TimePoint 0"][chs[0]]
            sx, sy, sz = (int(_a(g.attrs[k])) for k in ("ImageSizeX", "ImageSizeY", "ImageSizeZ"))
            d = g["Data"]
            pyr.append(dict(level=int(L.split()[-1]), valid=(sz, sy, sx), stored=tuple(d.shape),
                            chunks=tuple(d.chunks) if d.chunks else None, dtype=str(d.dtype)))
        channels = []
        for i in range(len(chs)):
            g = f["DataSetInfo"].get(f"Channel {i}")
            a = {k: _a(v) for k, v in g.attrs.items()} if g is not None else {}
            channels.append(dict(index=i, name=a.get("Name", ""), color=a.get("Color", ""),
                                 color_mode=a.get("ColorMode", ""), emission=a.get("LSMEmissionWavelength", ""),
                                 description=a.get("Description", "").replace("\n", " | ")))
        log = {k: _a(v) for k, v in f["DataSetInfo"]["Log"].attrs.items()} if "Log" in f["DataSetInfo"] else {}
        n_log = int(log.get("Entries", 0) or 0)
        ops = []
        for i in range(n_log):
            e = log.get(f"Entry{i}", "")
            m = re.match(r"<(\w+)", e)
            op = m.group(1).replace("bpIp", "").replace("Command", "") if m else "?"
            par = re.findall(r'(sigma|threshold|size|baseline) = "([^"]+)"', e)
            chans = re.search(r'channels = "([^"]+)"', e)
            ops.append(op + ("(" + ",".join(f"{k}={v}" for k, v in par) + ")" if par else "") +
                       ("[" + chans.group(1).strip() + "]" if chans else ""))
        scene = bytes(np.asarray(f["Scene8"]["Data"][0], dtype=np.uint8)).decode("latin1") if "Scene8" in f else ""
    cam = {}
    for k in ("position", "orientation", "heightAngle", "focalDistance"):
        m = re.search(r"<SurpassSoPerspectiveCamera" + k + r">(.*?)</", scene)
        cam[k] = m.group(1) if m else ""
    return dict(nx=nx, ny=ny, nz=nz, ext=ext, voxel_um=vox, unit=im.get("Unit", ""), image_name=im.get("Name", ""),
                recording_date=im.get("RecordingDate", ""), n_channels=len(chs), channels=channels, pyramid=pyr,
                log_ops=ops, camera=cam, render_mode=";".join(re.findall(r"<renderingMode>(.*?)</", scene)))


def ims_level_voxel(h: dict, level: int) -> tuple:
    """Voxel size (x, y, z um) of a pyramid level from its valid size."""
    sz, sy, sx = h["pyramid"][level]["valid"]
    (x0, x1), (y0, y1), (z0, z1) = h["ext"]
    return ((x1 - x0) / sx, (y1 - y0) / sy, (z1 - z0) / sz)


def ims_read(path: Path, channel: int, level: int, h: dict | None = None) -> np.ndarray:
    """One channel of one pyramid level, cropped to its valid size (z, y, x)."""
    h = h or ims_header(path)
    sz, sy, sx = h["pyramid"][level]["valid"]
    with h5py.File(path, "r") as f:
        d = f["DataSet"][f"ResolutionLevel {level}"]["TimePoint 0"][f"Channel {channel}"]["Data"]
        return d[:sz, :sy, :sx]


def ims_thumbnail(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        t = np.asarray(f["Thumbnail"]["Data"][()])
    if t.ndim == 2 and t.shape[1] == 4 * t.shape[0]:
        t = t.reshape(t.shape[0], t.shape[0], 4)
    return t[..., :3]


def level_for(h: dict, target_um: float) -> int:
    """Finest pyramid level whose xy voxel does not exceed the target (the analysis grid is reached by averaging)."""
    best = 0
    for p in h["pyramid"]:
        vx = ims_level_voxel(h, p["level"])[0]
        if vx <= target_um * 1.02:
            best = p["level"]
    return best


# ------------------------------------------------------------------------------------------------ CZI
def czi_meta(path: Path) -> dict:
    import czifile
    with czifile.CziFile(str(path)) as c:
        root = ET.fromstring(c.metadata())
        axes, shape, dtype = c.axes, c.shape, str(c.dtype)
    um = {d.get("Id"): float(d.find("Value").text) * 1e6 for d in root.iter("Distance")
          if d.get("Id") in ("X", "Y", "Z") and d.find("Value") is not None}
    chans = []
    chs = root.find(".//Information/Image/Dimensions/Channels")
    if chs is not None:
        for ch in chs.findall("Channel"):
            chans.append(dict(name=ch.get("Name") or "", fluor=ch.findtext("Fluor") or "",
                              ex=ch.findtext("ExcitationWavelength") or "", em=ch.findtext("EmissionWavelength") or ""))
    obj = root.find(".//Information/Instrument/Objectives/Objective")
    mic = root.find(".//Information/Instrument/Microscopes/Microscope")
    zoom = next((z.text for z in root.iter("ZoomX")), None)
    return dict(axes=axes, shape=tuple(shape), dtype=dtype, dims=dict(zip(axes, shape)),
                um_x=um.get("X"), um_y=um.get("Y"), um_z=um.get("Z"), channels=chans,
                objective=obj.get("Name") if obj is not None else "",
                magnification=root.findtext(".//Information/Instrument/Objectives/Objective/NominalMagnification"),
                na=root.findtext(".//Information/Instrument/Objectives/Objective/LensNA"),
                immersion=root.findtext(".//Information/Instrument/Objectives/Objective/Immersion"),
                zoom=zoom, acquired=root.findtext(".//Information/Image/AcquisitionDateAndTime") or "",
                microscope=(mic.get("Name") if mic is not None else "") or "",
                software=" ".join(x for x in (root.findtext(".//Information/Application/Name"),
                                              root.findtext(".//Information/Application/Version")) if x))


def czi_read(path: Path) -> np.ndarray:
    """CZI pixels as (C, Z, Y, X) (first of every other axis)."""
    import czifile
    with czifile.CziFile(str(path)) as c:
        a = c.asarray()
        axes = c.axes
    idx = tuple(slice(None) if ax in "CZYX" else 0 for ax in axes)
    a = a[idx]
    kept = [ax for ax in axes if ax in "CZYX"]
    a = np.transpose(a, [kept.index(ax) for ax in "CZYX" if ax in kept])
    if "Z" not in kept:
        a = a[:, None]
    if "C" not in kept:
        a = a[None]
    return a


# ------------------------------------------------------------------------------------------------ names
REGIONS = [("luoxuanban", "osseous spiral lamina"), ("out-linbate", "perilymph (name)"), ("utricle", "utricle"),
           ("tube", "tube (scala)"), ("apex", "apex"), ("bone", "temporal-bone marrow"), ("tile", "tile (whole cochlea)")]


def parse_name(stem: str) -> dict:
    low = stem.lower()
    m = re.search(r"(?<![0-9])(\d{1,2})\s*-?\s*d(?![a-z])", low)
    day = int(m.group(1)) if m else None
    region = ""
    if re.search(r"(-|_)sl(-|_|$)", low):
        region = "spiral ligament (lateral wall)"
    for key, lab in REGIONS:
        if key in low:
            region = (region + "; " if region else "") + lab
    mag = re.match(r"^(\d+)x", low)
    zoom = re.search(r"(\d+_\d+)x", low)
    labels = [k for k in ("lectin", "gfp", "dapi", "phalloidin", "pha", "cd45", "594", "647", "405") if
              re.search(r"(?<![a-z])" + k + r"(?![a-z])", low.replace("-", " ").replace("_", " "))]
    return dict(duration_d=day, noise="noise" in low,
                partner_label="WT" if re.search(r"(?<![a-z])wt(?![a-z])", low.replace("-", " ").replace("_", " "))
                else "", region_label=region, magnification_label=(mag.group(1) + "x") if mag else "",
                zoom_label=zoom.group(1).replace("_", ".") if zoom else "", stain_labels=";".join(labels))


# ------------------------------------------------------------------------------------------------ registry
# Channel roles of every Imaris volume (index -> role). 'vessel' = the channel whose tubular network is analysed;
# vessel_marker says what labels it. Source CZI = the acquisition the volume was converted from (verified by 950 from
# the pixel data; names in brackets differ from the volume's file name).
V, G, D, P, X, C = "vessel", "gfp", "dapi", "phalloidin", "other", "coloc"
ROLES = {
    "10x-0_45x-31d-NOISE-WT-GFP-LECTIN-sl-01": dict(roles={0: V}, marker="lectin (Alexa Fluor 647)"),
    "10x-CD45-DAPI-PHALLOIDIN-WT-01": dict(roles={0: X, 1: X, 2: D, 3: C}, marker=""),
    "10x-lectin-gfp-wt-14D-apex-05": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "10x-tile-gfp-594-405-02": dict(roles={0: X, 1: G}, marker="none identified (Alexa Fluor 594 marker not recorded)"),
    "14d-infect-gfp-vessal-10x-0_45x-sl-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "14d-infect-gfp-vessal-10x-0_6x-apex-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "14d-infect-gfp-vessal-5x-2um-02": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "20X-0_45X-14d-lectin-GFP-SL-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "20X-0_45X-31d-GFP-lectin-dapi-LUOXUANBAN-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "20X-0_45X-31d-lectin-GFP-BONE-02": dict(roles={0: G, 1: V}, marker="lectin (Alexa Fluor 647)"),
    "20x-0_45x-31d-NOISE-WT-GFP-LECTIN-apex-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "20X-1_6X-31d-GFP-lectin-dapi-BONE-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "20X-31d-pha-lectin-dapi-SL-APEX-01": dict(roles={0: V, 1: P}, marker="lectin (Alexa Fluor 647)"),
    "20X-31d-pha-lectin-dapi-SL-TUBE-01": dict(roles={0: V, 1: P}, marker="lectin (Alexa Fluor 647)"),
    "20x-phalloidin-gfp-wt-5D-apex-01": dict(roles={0: V, 1: D}, marker="phalloidin (Alexa Fluor 594; no GFP channel "
                                                                         "in the volume)"),
    "5X-31d-pha-lectin-dapi-01": dict(roles={0: V, 1: P}, marker="lectin (Alexa Fluor 647)"),
    "5X-7d-lectin-GFP-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "5x-31d-NOISE-WT-GFP-LECTIN-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "5x-CD45-DAPI-PHALLOIDIN-01": dict(roles={0: X, 1: X, 2: X}, marker=""),
    "5x-lectin-gfp-wt-14D-01": dict(roles={0: G, 1: V}, marker="lectin (Alexa Fluor 647)"),
    "Cochlea": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "Cochlear-14D-lectin-gfp-dapi-20x-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "Cochlear-14D-lectin-gfp-dapi-20x-2_4x-apex-no-01": dict(roles={0: G, 1: D, 2: C}, marker="none in the volume (lectin "
                                                                                              "only in the source CZI)"),
    "Cochlear-14D-lectin-gfp-dapi-20x-apex-01": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "Cochlear-14D-lectin-gfp-dapi-20x-apex-02": dict(roles={0: V, 1: G, 2: D, 3: C}, marker="lectin (Alexa Fluor 647)"),
    "Cochlear-14D-lectin-gfp-dapi": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "Cochlear-pha-dapi-20x-0_45x-tube-01": dict(roles={0: V}, marker="phalloidin (Alexa Fluor 488)"),
    "fei-tile-z-647-01-01": dict(roles={0: V}, marker="lectin (Alexa Fluor 647)"),
    "OUT-LINBATE-14D-12": dict(roles={0: V, 1: G}, marker="lectin (Alexa Fluor 647)"),
    "Utricle": dict(roles={0: X, 1: X, 2: X}, marker=""),
    "wmi19h-2-5": dict(roles={0: X, 1: V}, marker="unidentified (vessel-shaped channel; marker not recorded)"),
    "WT_CD45_Lectin": dict(roles={0: V}, marker="lectin or CD45 (single exported channel; vessel-shaped)"),
    "WT_phalloidin": dict(roles={0: V}, marker="phalloidin"),
}


def role_index(stem: str, role: str):
    r = ROLES.get(stem, {}).get("roles", {})
    return next((i for i, v in r.items() if v == role), None)


def robust(v):
    v = np.asarray(v, dtype=np.float64)
    med = float(np.median(v))
    return med, max(float(np.median(np.abs(v - med))) * 1.4826, 1e-6)


# ------------------------------------------------------------------------------------------------ resampling
def resample_iso(a: np.ndarray, vox: tuple, grid: float) -> np.ndarray:
    """(z, y, x) array with voxel (x, y, z) um -> isotropic `grid` um: Gaussian anti-aliasing where downsampling
    (sigma = (factor - 1) / 2 voxels), then linear interpolation."""
    from scipy import ndimage as ndi
    a = a.astype(np.float32, copy=False)
    f = (vox[2] / grid, vox[1] / grid, vox[0] / grid)
    sig = [max(0.0, (1 / q - 1) / 2) if q < 1 else 0.0 for q in f]
    if any(s > 0 for s in sig):
        a = gauss(a, sig)
    if all(abs(q - 1) < 1e-3 for q in f):
        return a
    return ndi.zoom(a, f, order=1, prefilter=False).astype(np.float32)


# ------------------------------------------------------------------------------------------------ GPU filters
_DEV = None
MAX_CHUNK_VOX = 24_000_000          # voxels per GPU chunk incl. halo (keeps the Hessian stage < ~3 GB)


def device():
    global _DEV
    if _DEV is None:
        import torch
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEV


def _kernels(sigma):
    import torch
    r = max(1, int(np.ceil(4 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device())
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g, -x / sigma ** 2 * g, (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g


def _conv(v, k, axis):
    import torch.nn.functional as F
    r = (k.numel() - 1) // 2
    shape = [1, 1, 1, 1, 1]
    shape[2 + axis] = -1
    pad = [0] * 6
    pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
    return F.conv3d(F.pad(v, pad, mode="replicate"), k.view(shape))


def _blocks(shape, halo):
    """3D tiles (core slices, padded slices) whose padded size stays within MAX_CHUNK_VOX."""
    core = 256
    while core > 32 and (core + 2 * halo) ** 3 > MAX_CHUNK_VOX:
        core //= 2
    cz = min(shape[0], max(core, MAX_CHUNK_VOX // ((core + 2 * halo) ** 2) - 2 * halo))
    for z0 in range(0, shape[0], cz):
        for y0 in range(0, shape[1], core):
            for x0 in range(0, shape[2], core):
                c = (slice(z0, min(shape[0], z0 + cz)), slice(y0, min(shape[1], y0 + core)),
                     slice(x0, min(shape[2], x0 + core)))
                p = tuple(slice(max(0, s.start - halo), min(n, s.stop + halo)) for s, n in zip(c, shape))
                inner = tuple(slice(s.start - q.start, s.stop - q.start) for s, q in zip(c, p))
                yield c, p, inner


def gauss(a: np.ndarray, sigma) -> np.ndarray:
    """Separable Gaussian smoothing on the GPU (sigma in voxels: scalar or (z, y, x)); replicate borders."""
    import torch
    s = (sigma,) * 3 if np.isscalar(sigma) else tuple(sigma)
    halo = int(np.ceil(4 * max(s))) + 1
    out = np.empty(a.shape, np.float32)
    ks = [(_kernels(q)[0] if q > 0 else None) for q in s]
    with torch.no_grad():
        for c, p, inner in _blocks(a.shape, halo):
            v = torch.from_numpy(np.ascontiguousarray(a[p], dtype=np.float32)).to(device())[None, None]
            for ax, k in enumerate(ks):
                if k is not None:
                    v = _conv(v, k, ax)
            out[c] = v[(0, 0) + inner].cpu().numpy()
            del v
    return out


def _eigs(a11, a22, a33, a12, a13, a23):
    """Ascending eigenvalues of symmetric 3x3 matrices (closed form)."""
    import torch
    q = (a11 + a22 + a33) / 3
    p1 = a12 ** 2 + a13 ** 2 + a23 ** 2
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p = torch.sqrt((b11 ** 2 + b22 ** 2 + b33 ** 2 + 2 * p1) / 6).clamp_min(1e-12)
    det = b11 * (b22 * b33 - a23 * a23) - a12 * (a12 * b33 - a23 * a13) + a13 * (a12 * a23 - b22 * a13)
    phi = torch.acos((det / (2 * p ** 3)).clamp(-1, 1)) / 3
    e3 = q + 2 * p * torch.cos(phi)
    e1 = q + 2 * p * torch.cos(phi + 2 * np.pi / 3)
    return e1, 3 * q - e1 - e3, e3


def sato(a: np.ndarray, sigmas, blob_suppression: bool = True, alpha: float = 0.25) -> tuple:
    """Multi-scale line filter for bright tubes on an isotropic volume (sigmas in voxels), maximum over scales of the
    sigma^2-normalised Hessian response; returns (tubeness, index of the best scale).
    blob_suppression=True: the original line measure of Sato et al. 1998 (Med Image Anal 2:143), eigenvalues
    l1 >= l2 >= l3, with l3 <= l2 < 0: |l3| (l2/l3)^0.5 (1 + l1/|l2|)^0.5 if l1 <= 0, |l3| (l2/l3)^0.5
    (1 - alpha l1/|l2|)^0.5 if 0 < l1 < |l2|/alpha, else 0 (blobs l1 ~ l2 ~ l3 and sheets l2 ~ 0 score ~0).
    blob_suppression=False: the simplified form of skimage.filters.sato, sqrt(|l3| |l2|)."""
    import torch
    halo = int(np.ceil(4 * max(sigmas))) + 2
    out = np.zeros(a.shape, np.float32)
    best = np.zeros(a.shape, np.uint8)
    with torch.no_grad():
        for c, p, inner in _blocks(a.shape, halo):
            v = torch.from_numpy(np.ascontiguousarray(a[p], dtype=np.float32)).to(device())[None, None]
            sl = (0, 0) + inner
            acc = torch.zeros(tuple(s.stop - s.start for s in c), device=device())
            arg = torch.zeros_like(acc, dtype=torch.uint8)
            for si, s in enumerate(sigmas):
                g, g1, g2 = _kernels(s)
                s2 = s * s
                vz = _conv(v, g, 0)
                vzy = _conv(vz, g, 1)
                hxx = _conv(vzy, g2, 2)[sl] * s2
                del vzy
                hyy = _conv(_conv(vz, g2, 1), g, 2)[sl] * s2
                hxy = _conv(_conv(vz, g1, 1), g1, 2)[sl] * s2
                del vz
                vz1 = _conv(v, g1, 0)
                hxz = _conv(_conv(vz1, g, 1), g1, 2)[sl] * s2
                hyz = _conv(_conv(vz1, g1, 1), g, 2)[sl] * s2
                del vz1
                hzz = _conv(_conv(_conv(v, g2, 0), g, 1), g, 2)[sl] * s2
                e1, e2, e3 = _eigs(hzz, hyy, hxx, hyz, hxz, hxy)          # ascending: e1 <= e2 <= e3
                del hxx, hyy, hzz, hxy, hxz, hyz
                if blob_suppression:
                    l3, l2, l1 = e1, e2, e3                                    # Sato's order l1 >= l2 >= l3
                    ok = (l2 < 0) & (l3 < 0)
                    a2 = torch.abs(l2).clamp_min(1e-12)
                    base = torch.abs(l3) * torch.sqrt((l2 / l3.clamp_max(-1e-12)).clamp(0, 1))
                    f_neg = torch.sqrt((1 + l1 / a2).clamp_min(0))
                    f_pos = torch.sqrt((1 - alpha * l1 / a2).clamp_min(0))
                    t = torch.where(ok, base * torch.where(l1 <= 0, f_neg, f_pos), torch.zeros_like(l1))
                    del l1, l2, l3, ok, a2, base, f_neg, f_pos
                else:
                    t = torch.sqrt(torch.clamp(-e1, min=0) * torch.clamp(-e2, min=0))
                del e1, e2, e3
                upd = t > acc
                acc = torch.where(upd, t, acc)
                arg = torch.where(upd, torch.full_like(arg, si), arg)
                del t, upd
            out[c] = acc.cpu().numpy()
            best[c] = arg.cpu().numpy()
            del v, acc, arg
        torch.cuda.empty_cache()
    return out, best
