"""Round 10 (user request 2026-10-17) — quantitative survey of the full parabiosis image collection: exogenous
(partner-derived, GFP+) cells across tissues and after noise. Local run (the images are not on the server):
COCH_ROOT=/path/to/coch-rythum python scripts/934_parabiosis_survey.py

Source: data/Parabiosis_Mice_ALL.zip, extracted unchanged to data/photos/parabiosis_all/Parabiosis_Mice (its INDEX.md
lists every file). Design (D128, D131, D133): wild-type C57BL/6J mice joined to Lyz2-P2A-iCre-P2A-EGFP partners, one
pair per pairing duration (5, 7, 14, 28, 31, 31 with noise and 54 days; every unlabelled bone and cochlea image is of the
wild-type partner, as stated by the authors), so in a wild-type partner GFP marks partner-derived (exogenous)
Lyz2-expressing myeloid cells. The rules of the first run (D132) were fixed before it; the extension below was added
afterwards (D133) and applies to the whole collection.

Acquisitions. Native Zeiss CZI acquisitions (LSM 900 on an Axio Observer.Z1/7; 8-bit) are used; snapshots,
orthogonal projections, Snap-series files and channel exports are not. Included: acquisitions with a GFP channel
(Alexa Fluor 488 / FITC detection) and <= 0.70 um per pixel (cell bodies need >= ~20 pixels). Excluded, with the reason
recorded: 5x / 10x overviews (coarser pixels), the folder marked as a discarded mouse, acquisitions without GFP, the
20250114-2 'pha-lectin-dapi' series (its 488-nm channel carries phalloidin), acquisitions without a tissue or duration
label (unless assigned under (iii) below), zoomed re-acquisitions inside an imaged position ('-4x-' in the name, or
perilymph fields finer than 0.5 um per pixel; not pooled with their parent field, as in 931) and second
parent-resolution images of the same perilymph position (only the first is kept). The Olympus .oir acquisitions of the
54-day animal (immune_cell_typing, January 2026) are not analysed. The GFP channel's acquisition settings (488-nm
transmission, detector voltage, offset, digital gain, pinhole) are recorded for every acquisition: they were set per
sample, so GFP intensities are not compared between acquisitions; densities use self-normalising rules and (v) checks
detection. The 31-day liver is available only as rendered TIFF exports (WT-31D-LIVER-20X-2_4X-02, -03, -05; c1 lectin,
c2 GFP, c3 DAPI, 8-bit display renderings); their pixel size, 0.130 um, is that of the 20X / 2.4x CZIs of the same
session. Brain (one rendered orthogonal projection without scale or DAPI) and spleen (no images) are not quantified.

Animals and partners. An animal is identified by its pairing duration and noise label (5, 7, 14, 28, 31, 31 with
noise, 54 days). Partner: 'WT' or 'GFP' where the name says so (wt-out / gfp-out perilymph, NOISE-WT, 7d-WT, WT-31D,
Parabiosis-WT-31d); otherwise inferred from the temporal-bone marrow of the same animal: GFP+ area below 10% of marrow
tissue = wild-type partner (a Lyz2-EGFP partner's marrow is largely GFP+), recorded as 'WT (inferred)'.

Compartments, from name labels: bone / BONE / ear-bone -> temporal-bone marrow (two 54-day 'ear-bone' fields without
marrow, bone and inner-ear soft tissue, are excluded, D136); qiantingbone -> vestibular bone; apex,
SL (spiral ligament), sv (stria vascularis), LUOXUANBAN (osseous spiral lamina), tube, and unlabelled 'Cochlear-' fields
-> cochlea (sub-region kept); out / OUT-LINBATE -> perilymph; liver. Fields may include tissue next to the labelled
region (for example small marrow islands of the otic capsule at the edge of a lateral-wall field), which can only raise
a cochlear density.

Tissue fields (marrow, cochlea, liver). Single planes as acquired; z-stacks are sampled at optical sections >= 10 um
apart (counted once each; planes with < 5% tissue skipped). Tissue = sum of the channels, each scaled to its 99.5th
percentile in the acquisition, Gaussian-smoothed (sigma 6 um), above a quarter of its Otsu threshold, closed (5-um
disk) and hole-filled. GFP+ cell body (931 rule) = GFP smoothed at sigma 0.5 um above the plane's tissue median + 10
robust SD (1.4826 x MAD, floor 1 grey level), holes filled, >= 15 um2, solidity >= 0.7. In lectin-labelled
acquisitions, broad-spectrum objects (mean far-red lectin signal inside the body above that channel's own tissue median
+ 10 robust SD; autofluorescence or GFP+ cells pressed against a vessel wall) are counted separately and removed in the
sensitivity density; in marker acquisitions (CD45, CD19, FUT4) the far-red channel is a marker and no object is
flagged. Density = GFP+ bodies per mm2 of tissue summed over the analysed sections.

Free-cell preparations (perilymph). Parent fields only. Nuclei = Otsu threshold of DAPI smoothed at sigma 0.5 um,
holes filled, touching nuclei split by a distance-transform watershed (peaks >= 2.5 um apart), objects >= 10 um2,
edge-touching nuclei dropped. Cytoplasmic ring = nucleus dilated by 2 um minus every nucleus. Local background = median
GFP 3-8 um from the nucleus outside every cell's 3-um zone (>= 50 pixels). A cell is GFP+ when its ring mean is >= 2x
the background (background floored at 2 grey levels) and >= 5 grey levels above it; nuclear-only green signal (DAPI
bleed-through) therefore does not count.

Extension (D133; the user asked to include as many additional replicates as possible):
(i) Positions. Stage coordinates (MTBStageAxisX / Y, MTBFocus) are read from every CZI. Within a session folder and
    specimen (same animal, partner label and compartment: stage coordinates are comparable only on one slide), two
    acquisitions image the same position when their fields overlap by > 50% of the smaller field and their focus
    differs by < 10 um; the coarser one is kept (a zoom inside an included field is dropped) and, at equal pixel size,
    the first acquired. Applies to every compartment, in addition to the name rules above; it replaces the first run's
    name rule for perilymph positions wherever stage coordinates exist (31d-noise-wt-out-...-03-02 is a separate
    position, 03-01 a re-acquisition of 02-01).
(ii) Labels. The day14_immune_cell_thick_sections folder gives the 14-day animal; 'sv' marks the stria vascularis.
(iii) Morphology. Unlabelled CZIs of the 54-day immune-typing series and of the 5-day animal are assigned from their
     morphology (MORPH below: haematopoietic marrow inside bone, or the cochlear lateral wall and duct); the full raw
     fields are shown in Supplementary Fig. 2, so the assignment can be checked. The 14-day CD45 63x fields (no region
     label, not localisable) stay excluded.
(iv) GFP separability. A tissue acquisition is excluded when its GFP channel correlates with another channel at Pearson
     r >= 0.8 inside tissue (in the analysed plane with most tissue): its 488-nm channel then reproduces another label
     (the unlabelled 5-day phalloidin fields, r 0.86-0.92; every other field r <= 0.75).
(v) Detection check. For every cochlear field, the share of the same animal's marrow GFP+ bodies that would clear the
    cochlear field's threshold at its settings: body signal above its plane's tissue median, scaled linearly with the
    488-nm transmission and by the detector-gain factor exp(beta x voltage difference), beta estimated from one position
    of the 54-day series imaged at two detector voltages with all else equal (fields 02 and 03: GFP+ bodies of 02's
    plane best matching 03's DAPI, median ratio of their signals), compared with the field's threshold offset (median
    over its analysed planes of threshold minus median). The cochlear count divided by that share gives a
    detection-corrected density (sensitivity analysis; fields with share 0 are left out of it). Perilymph fields get
    the same sensitivity index (transmission x exp(beta x (voltage - 700))) relative to the median of the same
    animal's GFP-partner fields.
(vi) Neutrophils in the 54-day ZEISS series (FUT4-Alexa Fluor 647 acquired with identical settings in every field):
     FUT4+ object = FUT4 smoothed at sigma 0.3 um above the Otsu threshold of FUT4 within the series' marrow tissue
     (included marrow fields, analysed planes), holes filled, >= 15 um2; GFP+ when it overlaps a GFP+ body (931 rule).
     Typed as a FUT4+ cell when it is one cell: <= 200 um2 and >= 10% of its pixels in a nucleus (DAPI smoothed at
     sigma 0.5 um above the plane's Otsu threshold within tissue); larger or anuclear objects (merged marrow clusters,
     bone autofluorescence in the 647-nm channel) are kept in the table but not typed. FUT4+ cells inside the drawn
     otic-capsule marrow island of lateral-wall field 05 (ISLANDS) are typed as marrow.
(vii) Depth. GFP fades with depth in thick z-stacks (54-day field 02: 17 counted bodies in its first analysed section,
     none in its deepest), which lowers counts in deep sections of either compartment. Sensitivity
     analysis: each acquisition's density in its brightest analysed section (highest GFP 99.9th percentile within
     tissue; chosen on signal, not on counts); single planes are unchanged. Planes table: parabiosis_planes.csv.

Statistics. Fields are not replicates: per animal and compartment the pooled density (bodies / summed tissue area),
the median and range of field densities and the number of fields are reported, without field-level tests. Across
animals (the unit), the cochlea-to-marrow density ratio is computed per wild-type partner with both compartments, and
the share of animals with a lower cochlear than marrow density is tested by an exact sign test (non-noise animals;
all animals). The 54-day animal enters from its ZEISS series (934 rules, as every other animal); 931's 60x
sections of the same animal are reported as a second acquisition, not counted twice. Noise is one animal: its cochlea
and marrow are compared descriptively with the non-noise animals.

Outputs (lib/io.write_csv, results/r10_parabiosis/): parabiosis_acquisitions.csv (every CZI, stage position,
inclusion and reason), parabiosis_tissue_fields.csv (one row per included tissue acquisition),
parabiosis_perilymph_cells.csv (one row per cell) and parabiosis_perilymph_fields.csv, parabiosis_summary.csv (animal x
compartment), parabiosis_ratio.csv (per animal cochlea / marrow, observed and detection-corrected),
parabiosis_detection.csv (per cochlear field), parabiosis_gain_reference.csv (the voltage pair),
parabiosis_fut4_cells.csv (54-day FUT4+ objects) and parabiosis_objects.csv (GFP+ bodies, for figure crops).
"""
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import czifile
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import binomtest
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import ROOT, log, write_csv  # noqa: E402

SRC = ROOT / "data" / "photos" / "parabiosis_all" / "Parabiosis_Mice"
MOD = "r10_parabiosis"
MAX_UM = 0.70
LIVER = ["WT-31D-LIVER-20X-2_4X-02", "WT-31D-LIVER-20X-2_4X-03", "WT-31D-LIVER-20X-2_4X-05"]
LIVER_UM = 0.130
OVERLAP, DFOCUS, SEP_R = 0.5, 10.0, 0.8
S54 = "immune_cell_typing/immunecelltype-54d/"
GAIN_REF = (S54 + "647fut4-568cd14-gfp-54d-20x-02.czi", S54 + "647fut4-568cd14-gfp-54d-20x-03.czi")
_LW = "lateral wall: stria vascularis and spiral ligament"
MORPH = {S54 + "647fut4-568cd14-gfp-54d-20x-01.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647fut4-568cd14-gfp-54d-20x-02.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647fut4-568cd14-gfp-54d-20x-03.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647fut4-568cd14-gfp-54d-20x-04.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647fut4-568cd14-gfp-54d-20x-05.czi": ("cochlea", "lateral wall", _LW),
         S54 + "647fut4-568cd14-gfp-54d-20x-06.czi": ("cochlea", "lateral wall", _LW),
         S54 + "647fut4-568cd14-gfp-54d-20x-07.czi": ("cochlea", "lateral wall", _LW),
         S54 + "647fut4-568cd14-gfp-54d-20x-08.czi": ("cochlea", "lateral wall", _LW + ", Reissner's membrane"),
         S54 + "647fut4-568cd14-gfp-54d-20x-09.czi": ("cochlea", "lateral wall", _LW + ", Reissner's membrane"),
         S54 + "647fut4-568cd14-gfp-54d-20x-10.czi": ("cochlea", "lateral wall", "stria vascularis, Reissner's membrane"),
         S54 + "647fut4-568cd14-gfp-54d-20x-11.czi": ("cochlea", "duct", "cochlear duct wall and Reissner's membrane"),
         S54 + "647fut4-568cd14-gfp-54d-20x-12.czi": ("cochlea", "duct", "cochlear duct, stria and otic capsule"),
         S54 + "647fut4-568cd14-gfp-54d-20x-13.czi": ("cochlea", "duct", "otic capsule beside the duct"),
         S54 + "647cd19-gfp-54d-5x-04.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647cd19-gfp-54d-5x-05.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647cd19-gfp-54d-5x-06.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         S54 + "647cd19-gfp-54d-5x-07.czi": ("marrow", "", "haematopoietic marrow inside bone"),
         "2025-01-09/2025-01-09/20x-phalloidin-gfp-wt-5D-01.czi": ("cochlea", "duct", "cochlear duct epithelium"),
         "2025-01-09/2025-01-09/5x-phalloidin-gfp-wt-5D-02.czi": ("cochlea", "duct", "cochlear turn in cross-section"),
         "2025-01-09/2025-01-09/5x-phalloidin-gfp-wt-5D-03.czi": ("cochlea", "duct", "cochlear turn in cross-section")}
# fields named for the temporal bone ('ear-bone') that hold no haematopoietic marrow (checked by eye on DAPI and the
# marker channel when the 54-day immune cell types were typed, D136): excluded rather than counted as marrow
NO_MARROW = {S54.replace("immunecelltype-54d/", "2026-02-09/") + "parabiosis54d-gfp-647CD45-dapi-20x-ear-bone-04.czi":
             "no haematopoietic marrow in the field (inner-ear soft tissue)",
             S54.replace("immunecelltype-54d/", "2026-02-09/") + "parabiosis54d-gfp-647CD45-dapi-20x-ear-bone-05.czi":
             "no haematopoietic marrow in the field (bone)"}
# marrow island of the otic capsule inside a lateral-wall field (x, y, radius in pixels; drawn on DAPI): FUT4+ cells in
# it are typed as marrow, not cochlea (as 931 excluded the marrow corner of its cochlear turn)
ISLANDS = {S54 + "647fut4-568cd14-gfp-54d-20x-05.czi": [(40, 865, 110)]}


# ------------------------------------------------------------------------------------------------ labels
def labels(rel: str) -> dict:
    low = rel.lower()
    base = low.rsplit("/", 1)[-1]
    m = re.search(r"(?<![0-9])(\d{1,2})\s*-?\s*d(?![a-z])", low)
    day = int(m.group(1)) if m else (14 if "day14_" in low else None)
    noise = "noise" in low
    if re.search(r"wt-out|parabiosis-wt|wt-31d|noise-wt|7d-wt", low):
        partner = "WT"
    elif "gfp-out" in low:
        partner = "GFP"
    else:
        partner = ""
    if "-out-" in base or "out-linbate" in base:
        comp, sub = "perilymph", ""
    elif "qiantingbone" in base:
        comp, sub = "vestibular bone", ""
    elif "bone" in base:
        comp, sub = "marrow", ""
    elif "liver" in low:
        comp, sub = "liver", ""
    elif "luoxuanban" in base:
        comp, sub = "cochlea", "spiral lamina"
    elif re.search(r"(-|_)sl(-|_|\.)", base):
        comp, sub = "cochlea", "spiral ligament"
    elif re.search(r"(-|_)sv(-|_|\.)", base):
        comp, sub = "cochlea", "stria vascularis"
    elif "apex" in base:
        comp, sub = "cochlea", "apex"
    elif "tube" in base:
        comp, sub = "cochlea", "tube"
    elif base.startswith("cochlear-"):
        comp, sub = "cochlea", "unlabelled"
    else:
        comp, sub = "", ""
    basis = "name label" if comp else ""
    if rel in MORPH:
        comp, sub, desc = MORPH[rel]
        basis = f"morphology ({desc})"
    animal = (f"{day} d" + (" noise" if noise else "")) if day else ""
    return dict(day=day, noise=noise, animal=animal, partner_label=partner, compartment=comp, subregion=sub,
                compartment_basis=basis)


def czi_meta(c) -> dict:
    root = ET.fromstring(c.metadata())
    um = {d.get("Id"): float(d.find("Value").text) * 1e6 for d in root.iter("Distance")
          if d.get("Id") in ("X", "Z") and d.find("Value") is not None}
    names, gfp_set = [], {}
    chs = root.find(".//Information/Image/Dimensions/Channels")
    if chs is not None:
        for ch in chs.findall("Channel"):
            fl = ch.find("Fluor")
            name = (fl.text if fl is not None and fl.text else ch.get("Name")) or "?"
            names.append(name)
            if not gfp_set and ("488" in name or "FITC" in name.upper()):
                ds = ch.find("DetectorSettings")
                ls = ch.find("LightSourcesSettings/LightSourceSettings")
                tr = ls.findtext("Transmission") if ls is not None else None
                gfp_set = dict(gfp_laser_pct=round(100 * float(tr), 2) if tr else None,
                               gfp_detector_v=round(float(ds.findtext("Voltage")), 0) if ds is not None and
                               ds.findtext("Voltage") else None,
                               gfp_offset=ds.findtext("Offset") if ds is not None else None,
                               gfp_digital_gain=ds.findtext("DigitalGain") if ds is not None else None,
                               gfp_pinhole_um=round(float(ch.findtext("PinholeSize")), 1) if ch.findtext("PinholeSize")
                               else None)
    stage = {}
    for pc in root.iter("ParameterCollection"):
        if pc.get("Id") in ("MTBStageAxisX", "MTBStageAxisY", "MTBFocus") and pc.findtext("Position"):
            stage[pc.get("Id")] = float(pc.findtext("Position"))
    return dict(um_x=um.get("X"), um_z=um.get("Z"), channels=names, gfp_settings=gfp_set,
                stage_x_um=stage.get("MTBStageAxisX"), stage_y_um=stage.get("MTBStageAxisY"),
                focus_um=stage.get("MTBFocus"), acquired=root.findtext(".//AcquisitionDateAndTime"))


def channel_roles(names):
    gfp = next((i for i, n in enumerate(names) if "488" in n or "FITC" in n.upper() or "GFP" in n.upper()), None)
    dapi = next((i for i, n in enumerate(names) if "DAPI" in n.upper()), None)
    far = next((i for i, n in enumerate(names) if i not in (gfp, dapi)), None)
    return gfp, dapi, far


def read_czi(path: Path):
    with czifile.CziFile(str(path)) as c:
        a = c.asarray()
        axes = c.axes
        meta = czi_meta(c)
    idx = tuple(slice(None) if ax in "CZYX" else 0 for ax in axes)
    a = a[idx]
    kept = [ax for ax in axes if ax in "CZYX"]
    order = [kept.index(ax) for ax in "CZYX" if ax in kept]
    a = np.transpose(a, order)
    if "Z" not in kept:
        a = a[:, None]
    return a, meta                                  # (C, Z, Y, X)


def overlap_frac(r1, r2):
    """Overlap of two stage-centred fields as a share of the smaller field."""
    ox = min(r1["stage_x_um"] + r1["fov_x_um"] / 2, r2["stage_x_um"] + r2["fov_x_um"] / 2) - \
        max(r1["stage_x_um"] - r1["fov_x_um"] / 2, r2["stage_x_um"] - r2["fov_x_um"] / 2)
    oy = min(r1["stage_y_um"] + r1["fov_y_um"] / 2, r2["stage_y_um"] + r2["fov_y_um"] / 2) - \
        max(r1["stage_y_um"] - r1["fov_y_um"] / 2, r2["stage_y_um"] - r2["fov_y_um"] / 2)
    small = min(r1["fov_x_um"] * r1["fov_y_um"], r2["fov_x_um"] * r2["fov_y_um"])
    return max(ox, 0) * max(oy, 0) / small


# ------------------------------------------------------------------------------------------------ tissue fields
def robust(v):
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, 1.0)


def tissue_mask(chs, um, scales):
    s = sum(ch.astype(np.float32) / sc for ch, sc in zip(chs, scales))
    sm = filters.gaussian(s, sigma=6 / um, preserve_range=True)
    if sm.max() <= sm.min():
        return np.zeros(s.shape, bool)
    t = filters.threshold_otsu(sm) / 4
    m = morphology.binary_closing(sm > t, morphology.disk(max(1, int(round(5 / um)))))
    return ndi.binary_fill_holes(m)


def gfp_bodies(g, far, tissue, um):
    gs = filters.gaussian(g.astype(np.float32), sigma=0.5 / um, preserve_range=True)
    med, sd = robust(gs[tissue])
    thr = med + 10 * sd
    pos = ndi.binary_fill_holes(gs > thr) & tissue
    lab = measure.label(pos)
    fthr = None
    if far is not None:
        fs = filters.gaussian(far.astype(np.float32), sigma=0.5 / um, preserve_range=True)
        fmed, fsd = robust(fs[tissue])
        fthr = fmed + 10 * fsd
    out, keep = [], np.zeros(lab.shape, bool)
    for r in measure.regionprops(lab, intensity_image=gs):
        area = r.area * um * um
        if area < 15 or r.solidity < 0.7:
            continue
        broad = False
        if far is not None:
            broad = float(fs[tuple(r.coords.T)].mean()) > fthr
        keep[tuple(r.coords.T)] = True
        out.append(dict(y=float(r.centroid[0]), x=float(r.centroid[1]), area_um2=area,
                        gfp_mean=float(r.mean_intensity), signal=float(r.mean_intensity) - med, broad_spectrum=broad))
    return out, dict(median=med, sd=sd, threshold=thr), float(pos.sum()) / max(int(tissue.sum()), 1), keep


def sampled_planes(a, meta):
    """(z, tissue mask) of the analysed optical sections (>= 10 um apart, >= 5% tissue)."""
    um, umz = meta["um_x"], meta["um_z"] or 1.0
    nz = a.shape[1]
    step = 1 if nz == 1 else max(1, math.ceil(10 / umz))
    scales = [max(float(np.percentile(a[c], 99.5)), 1.0) for c in range(a.shape[0])]
    for z in range(0, nz, step):
        tis = tissue_mask([a[c, z] for c in range(a.shape[0])], um, scales)
        if tis.mean() >= 0.05:
            yield z, tis


def analyse_tissue(a, meta, rel):
    um, umz = meta["um_x"], meta["um_z"] or 1.0
    gi, di, fi = channel_roles(meta["channels"])
    if "lectin" not in rel.lower():                 # far-red is a marker (CD45 / CD19 / FUT4), not lectin: no flag
        fi = None
    nz = a.shape[1]
    step = 1 if nz == 1 else max(1, math.ceil(10 / umz))
    area_px, n_all, n_broad, pos_frac, used, objs, planes = 0, 0, 0, [], [], [], []
    sep, sep_px = {}, -1
    for z, tis in sampled_planes(a, meta):
        bodies, st, pf, _ = gfp_bodies(a[gi, z], a[fi, z] if fi is not None else None, tis, um)
        area_px += int(tis.sum())
        n_all += len(bodies)
        n_broad += sum(b["broad_spectrum"] for b in bodies)
        pos_frac.append(pf)
        used.append(z)
        gs_p = filters.gaussian(a[gi, z].astype(np.float32), sigma=0.5 / um, preserve_range=True)
        planes.append(dict(path=rel, plane=z, tissue_px=int(tis.sum()), tissue_mm2=float(tis.sum()) * um * um / 1e6,
                           gfp_bodies=len(bodies), gfp_p999=float(np.percentile(gs_p[tis], 99.9)), **st))
        for b in bodies:
            objs.append(dict(path=rel, plane=z, threshold=st["threshold"], **b))
        if tis.sum() > sep_px:                      # GFP separability on the plane with most tissue
            sep_px = int(tis.sum())
            g = a[gi, z][tis].astype(np.float32)
            sep = {}
            for c in range(a.shape[0]):
                if c != gi:
                    o = a[c, z][tis].astype(np.float32)
                    sep[meta["channels"][c]] = float(np.corrcoef(g, o)[0, 1]) if g.std() > 0 and o.std() > 0 \
                        else np.nan
    area_mm2 = area_px * um * um / 1e6
    worst = max(sep, key=lambda k: -np.inf if np.isnan(sep[k]) else sep[k]) if sep else None
    return dict(um_per_px=um, um_z=umz if nz > 1 else None, n_z=nz, plane_step=step, planes_analysed=len(used),
                tissue_mm2=area_mm2, gfp_bodies=n_all, broad_spectrum=n_broad,
                density_per_mm2=n_all / area_mm2 if area_mm2 else np.nan,
                density_excl_broad_per_mm2=(n_all - n_broad) / area_mm2 if area_mm2 else np.nan,
                gfp_area_pct=100 * float(np.mean(pos_frac)) if pos_frac else np.nan,
                has_dapi=di is not None, gfp_sep_r=sep.get(worst, np.nan) if worst else np.nan,
                gfp_sep_channel=worst or ""), objs, planes


# ------------------------------------------------------------------------------------------------ free cells
def analyse_free(a, meta, rel):
    um = meta["um_x"]
    gi, di, _ = channel_roles(meta["channels"])
    d = a[di].max(axis=0).astype(np.float32)
    g = a[gi].max(axis=0).astype(np.float32)
    ds = filters.gaussian(d, sigma=0.5 / um, preserve_range=True)
    nuc = ndi.binary_fill_holes(ds > filters.threshold_otsu(ds))
    dist = ndi.distance_transform_edt(nuc)
    pk = peak_local_max(dist, min_distance=max(1, int(round(2.5 / um))), labels=measure.label(nuc), exclude_border=False)
    mk = np.zeros(nuc.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-dist, mk, mask=nuc)
    lab = morphology.remove_small_objects(lab, min_size=max(1, int(round(10 / (um * um)))))
    lab = segmentation.clear_border(lab)
    allnuc = lab > 0
    zone3 = morphology.binary_dilation(allnuc, morphology.disk(max(1, int(round(3 / um)))))
    rows = []
    for r in measure.regionprops(lab):
        own = lab == r.label
        y0, x0, y1, x1 = r.bbox
        pad = int(round(9 / um))
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        o = own[sl]
        ring = morphology.binary_dilation(o, morphology.disk(max(1, int(round(2 / um))))) & ~allnuc[sl]
        dd = ndi.distance_transform_edt(~o) * um
        bg_zone = (dd >= 3) & (dd <= 8) & ~zone3[sl]
        if ring.sum() < 5 or bg_zone.sum() < 50:
            continue
        gg = g[sl]
        rm, bg, nm = float(gg[ring].mean()), float(np.median(gg[bg_zone])), float(gg[o].mean())
        pos = rm >= 2 * max(bg, 2.0) and rm - bg >= 5
        rows.append(dict(path=rel, y=float(r.centroid[0]), x=float(r.centroid[1]), nucleus_um2=r.area * um * um,
                         ring_gfp=rm, nucleus_gfp=nm, background=bg, gfp_pos=pos))
    return rows


# ------------------------------------------------------------------------------------------------ extension helpers
def gain_reference():
    """Detector-gain factor per volt from one position imaged at two GFP voltages (all else equal)."""
    a2, m2 = read_czi(SRC / GAIN_REF[0])
    a3, m3 = read_czi(SRC / GAIN_REF[1])
    gi, di, _ = channel_roles(m2["channels"])
    um = m2["um_x"]
    d3 = a3[di, 0].astype(np.float32).ravel()
    rs = [float(np.corrcoef(a2[di, z].astype(np.float32).ravel(), d3)[0, 1]) for z in range(a2.shape[1])]
    zb = int(np.argmax(rs))
    scales = [max(float(np.percentile(a2[c], 99.5)), 1.0) for c in range(a2.shape[0])]
    tis = tissue_mask([a2[c, zb] for c in range(a2.shape[0])], um, scales)
    _, _, _, keep = gfp_bodies(a2[gi, zb], None, tis, um)
    g2 = filters.gaussian(a2[gi, zb].astype(np.float32), sigma=0.5 / um, preserve_range=True)
    g3 = filters.gaussian(a3[gi, 0].astype(np.float32), sigma=0.5 / um, preserve_range=True)
    med2, med3 = float(np.median(g2[tis])), float(np.median(g3[tis]))
    ratios = []
    for r in measure.regionprops(measure.label(keep)):
        idx = tuple(r.coords.T)
        s2, s3 = float(g2[idx].mean()) - med2, float(g3[idx].mean()) - med3
        if s2 > 0:
            ratios.append(s3 / s2)
    v2, v3 = m2["gfp_settings"]["gfp_detector_v"], m3["gfp_settings"]["gfp_detector_v"]
    t2, t3 = m2["gfp_settings"]["gfp_laser_pct"], m3["gfp_settings"]["gfp_laser_pct"]
    ratio = float(np.median(ratios))
    beta = math.log(ratio) / (v3 - v2)
    return dict(field_high_v=GAIN_REF[0], field_low_v=GAIN_REF[1], plane_high_v=zb, dapi_r=rs[zb],
                laser_high_v_pct=t2, laser_low_v_pct=t3, voltage_high=v2, voltage_low=v3, bodies=len(ratios),
                signal_ratio_median=ratio, signal_ratio_q25=float(np.percentile(ratios, 25)),
                signal_ratio_q75=float(np.percentile(ratios, 75)), beta_per_volt=beta)


def fut4_values(a, meta):
    """FUT4 (sigma 0.3 um) inside tissue of the analysed planes (every 4th pixel), for the series threshold."""
    um = meta["um_x"]
    fi = next(i for i, n in enumerate(meta["channels"]) if "647" in n)
    vals = []
    for z, tis in sampled_planes(a, meta):
        fs = filters.gaussian(a[fi, z].astype(np.float32), sigma=0.3 / um, preserve_range=True)
        vals.append(fs[tis][::4])
    return np.concatenate(vals)


def fut4_cells(a, meta, rel, comp, thr):
    um = meta["um_x"]
    gi, di, _ = channel_roles(meta["channels"])
    fi = next(i for i, n in enumerate(meta["channels"]) if "647" in n)
    yy, xx = np.mgrid[:a.shape[2], :a.shape[3]]
    island = np.zeros(a.shape[2:], bool)
    for cx, cy, rad in ISLANDS.get(rel, []):
        island |= (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
    rows = []
    for z, tis in sampled_planes(a, meta):
        _, _, _, body = gfp_bodies(a[gi, z], None, tis, um)
        ds = filters.gaussian(a[di, z].astype(np.float32), sigma=0.5 / um, preserve_range=True)
        nuc = (ds > filters.threshold_otsu(ds[tis])) & tis
        fs = filters.gaussian(a[fi, z].astype(np.float32), sigma=0.3 / um, preserve_range=True)
        lab = measure.label(ndi.binary_fill_holes(fs > thr) & tis)
        for r in measure.regionprops(lab, intensity_image=fs):
            area = r.area * um * um
            if area < 15:
                continue
            idx = tuple(r.coords.T)
            nf = float(nuc[idx].mean())
            rows.append(dict(path=rel, plane=z,
                             compartment="marrow (otic-capsule island)" if island[idx].mean() > 0.5 else comp,
                             y=float(r.centroid[0]), x=float(r.centroid[1]), area_um2=area, nucleus_frac=nf,
                             typed=bool(area <= 200 and nf >= 0.1), fut4_mean=float(r.mean_intensity),
                             gfp_pos=bool(body[idx].any())))
    return rows


# ------------------------------------------------------------------------------------------------ main
def main():
    files = sorted(SRC.rglob("*.czi"))
    acq = []
    for p in files:                                  # pass 1: metadata and name rules
        rel = p.relative_to(SRC).as_posix()
        low = rel.lower()
        lab = labels(rel)
        row = dict(path=rel, session=rel.rsplit("/", 1)[0], **lab)
        with czifile.CziFile(str(p)) as c:
            meta = czi_meta(c)
            ax = dict(zip(c.axes, c.shape))
        um = meta["um_x"]
        row.update(um_per_px=um, n_z=ax.get("Z", 1), channels=";".join(meta["channels"]),
                   stage_x_um=meta["stage_x_um"], stage_y_um=meta["stage_y_um"], focus_um=meta["focus_um"],
                   fov_x_um=um * ax.get("X", 0) if um else None, fov_y_um=um * ax.get("Y", 0) if um else None,
                   acquired=meta["acquired"], **meta["gfp_settings"])
        gi, di, _ = channel_roles(meta["channels"])
        reason = ""
        if "orthogonal projection" in low or "snapshot" in low or "/snap-" in low:
            reason = "derivative (projection or snapshot)"
        elif "discarded" in low:
            reason = "folder marked as discarded mouse"
        elif "pha-lectin-dapi" in low:
            reason = "488-nm channel carries phalloidin"
        elif gi is None:
            reason = "no GFP channel"
        elif um is None or um > MAX_UM:
            reason = "pixels coarser than 0.70 um (overview)"
        elif "-4x-" in low:
            reason = "zoomed re-acquisition inside an imaged position"
        elif rel in NO_MARROW:
            reason = NO_MARROW[rel]
        elif not lab["compartment"] or not lab["animal"]:
            reason = "no tissue or duration label"
        elif lab["compartment"] == "perilymph" and di is None:
            reason = "perilymph without DAPI"
        elif lab["compartment"] == "perilymph" and um < 0.5:
            reason = "zoomed re-acquisition inside an imaged position"
        row["reason"] = reason
        acq.append(row)
    acq = pd.DataFrame(acq)

    # perilymph: second parent-resolution image of a named position (first-run name rule), now only for CZIs without
    # stage coordinates; the stage rule (i) decides otherwise (the names' second index does not always mark one position)
    seen = set()
    for i, r in acq.iterrows():
        if r.reason or r.compartment != "perilymph" or pd.notna(r.stage_x_um):
            continue
        key = re.sub(r"(-0_45x-\d+)-\d+$", r"\1", r.path.lower().rsplit("/", 1)[-1].replace(".czi", ""))
        if key in seen:
            acq.loc[i, "reason"] = "second image of the same perilymph position"
        seen.add(key)

    # pass 2: analyse every candidate (pixel rules decide afterwards)
    cand = acq[acq.reason == ""]
    tissue_res, objects, plane_rows, cells = {}, {}, {}, []
    for _, r in cand.iterrows():
        log(f"{r.path} ({r.animal}, {r.compartment})")
        a, meta = read_czi(SRC / r.path)
        if r.compartment == "perilymph":
            cells += [dict(**{k: r[k] for k in ("animal", "partner_label")}, **meta["gfp_settings"], **c_)
                      for c_ in analyse_free(a, meta, r.path)]
        else:
            res, objs, pls = analyse_tissue(a, meta, r.path)
            tissue_res[r.path], objects[r.path], plane_rows[r.path] = res, objs, pls
        del a
    for i, r in acq.iterrows():                      # (iv) GFP separability
        res = tissue_res.get(r.path)
        if res is not None and res["gfp_sep_r"] >= SEP_R:
            acq.loc[i, "reason"] = (f"GFP channel not separable (r = {res['gfp_sep_r']:.2f} with "
                                    f"{res['gfp_sep_channel']})")
    # (i) stage de-duplication within a session and specimen (animal, partner label, compartment): coarser first, then
    # earlier; stage coordinates are comparable only on one slide
    acq["stage_duplicate_of"] = ""
    acq["specimen"] = acq.animal + "|" + acq.partner_label + "|" + acq.compartment
    pool = acq[(acq.reason == "") & acq.stage_x_um.notna()].sort_values(["um_per_px", "acquired"],
                                                                          ascending=[False, True])
    kept = []
    for i, r in pool.iterrows():
        hit = next((k for k in kept if acq.loc[k, "session"] == r.session and acq.loc[k, "specimen"] == r.specimen and
                    abs((acq.loc[k, "focus_um"] or 0) - (r.focus_um or 0)) < DFOCUS and
                    overlap_frac(acq.loc[k], r) > OVERLAP), None)
        if hit is None:
            kept.append(i)
            continue
        other = acq.loc[hit]
        acq.loc[i, "stage_duplicate_of"] = other.path
        acq.loc[i, "reason"] = ("zoom inside an included field (stage coordinates)"
                                if r.um_per_px < 0.9 * other.um_per_px else "same position re-acquired (stage coordinates)")
    acq["included"] = acq.reason == ""
    acq["reason"] = acq.reason.replace("", "included")
    inc = set(acq[acq.included].path)

    tf = pd.DataFrame([dict(path=p, source="CZI", **{k: acq.set_index("path").loc[p, k] for k in
                       ("day", "noise", "animal", "partner_label", "compartment", "subregion", "compartment_basis",
                        "gfp_laser_pct", "gfp_detector_v", "gfp_offset", "gfp_digital_gain", "gfp_pinhole_um")},
                       **res) for p, res in tissue_res.items() if p in inc])
    ob = pd.DataFrame([o for p, os_ in objects.items() if p in inc for o in os_])
    pl = pd.DataFrame([q for p, qs in plane_rows.items() if p in inc for q in qs])
    cl = pd.DataFrame([c_ for c_ in cells if c_["path"] in inc])

    liver_rows, liver_objs = [], []
    for stem in LIVER:                             # rendered exports: c1 lectin, c2 GFP, c3 DAPI (8-bit renderings)
        d = SRC / "2025-01-15-2" / "TIFF-25.1.15" / stem
        chs = [tifffile.imread(d / f"{stem}_c{k}.tif").astype(np.float32).max(axis=-1) for k in (1, 2, 3)]
        a = np.stack(chs)[:, None]
        meta = dict(um_x=LIVER_UM, um_z=None, channels=["Alexa Fluor 647", "Alexa Fluor 488", "DAPI"])
        rel = f"2025-01-15-2/TIFF-25.1.15/{stem}"
        lab = labels(rel)
        res, objs, _ = analyse_tissue(a, meta, rel)
        liver_rows.append(dict(path=rel, source="rendered TIFF export", **lab, **res))
        liver_objs += objs
        acq = pd.concat([acq, pd.DataFrame([dict(path=rel, **lab, um_per_px=LIVER_UM, n_z=1,
                                                 channels="lectin;GFP;DAPI (rendered)", included=True,
                                                 reason="included (rendered export; pixel size from the session's "
                                                        "20X/2.4x CZI)")])], ignore_index=True)
    tf = pd.concat([tf, pd.DataFrame(liver_rows)], ignore_index=True)
    ob = pd.concat([ob, pd.DataFrame(liver_objs)], ignore_index=True)

    # (vii) brightest analysed section of each acquisition (GFP 99.9th percentile within tissue): depth-robust density
    best = pl.loc[pl.groupby("path").gfp_p999.idxmax(), ["path", "plane", "gfp_bodies", "tissue_mm2", "gfp_p999"]]
    best = best.rename(columns={"plane": "best_plane", "gfp_bodies": "best_plane_bodies",
                                "tissue_mm2": "best_plane_tissue_mm2", "gfp_p999": "best_plane_gfp_p999"})
    tf = tf.merge(best, on="path", how="left")
    tf["best_plane_density_per_mm2"] = tf.best_plane_bodies / tf.best_plane_tissue_mm2
    lv = tf.source == "rendered TIFF export"                      # single rendered planes: the plane itself
    tf.loc[lv, "best_plane_bodies"] = tf.loc[lv, "gfp_bodies"]
    tf.loc[lv, "best_plane_tissue_mm2"] = tf.loc[lv, "tissue_mm2"]
    tf.loc[lv, "best_plane_density_per_mm2"] = tf.loc[lv, "density_per_mm2"]

    # partner: labelled, else inferred from the animal's marrow GFP+ area (< 10% = wild-type partner)
    marrow_pct = tf[tf.compartment == "marrow"].groupby("animal").gfp_area_pct.median()

    def partner(r):
        if r.partner_label:
            return r.partner_label, "file label"
        pct = marrow_pct.get(r.animal)
        if pct is not None and pct < 10:
            return "WT", f"inferred (marrow GFP+ area {pct:.1f}%)"
        return "unresolved", "no marrow reference"
    lab_map = {}
    for an in tf.animal.unique():
        lbl = tf[(tf.animal == an) & (tf.partner_label != "")].partner_label.unique()
        lab_map[an] = lbl[0] if len(lbl) else ""
    tf["partner_label"] = tf.apply(lambda r: r.partner_label or lab_map.get(r.animal, ""), axis=1)
    tf[["partner", "partner_basis"]] = tf.apply(lambda r: pd.Series(partner(r)), axis=1)

    # (v) detection check: can each cochlear field detect the same animal's marrow GFP+ bodies at its settings?
    gref = gain_reference()
    beta = gref["beta_per_volt"]
    log(f"gain reference: signal ratio {gref['signal_ratio_median']:.3f} for {gref['voltage_low']:.0f} vs "
        f"{gref['voltage_high']:.0f} V ({gref['bodies']} bodies); beta {beta:.4f} per V")
    set_cols = ["path", "gfp_laser_pct", "gfp_detector_v"]
    mb = ob.merge(tf[set_cols + ["animal", "compartment", "partner"]], on="path")
    mb = mb[(mb.compartment == "marrow") & mb.gfp_laser_pct.notna()]
    det = []
    for _, f in tf[(tf.compartment == "cochlea") & tf.gfp_laser_pct.notna()].iterrows():
        offs = pl[pl.path == f.path]
        d_off = float((offs.threshold - offs["median"]).median())
        m = mb[mb.animal == f.animal]
        if not len(m):
            continue
        scale = (f.gfp_laser_pct / m.gfp_laser_pct) * np.exp(beta * (f.gfp_detector_v - m.gfp_detector_v))
        share = float((m.signal * scale >= d_off).mean())
        det.append(dict(path=f.path, animal=f.animal, partner=f.partner, gfp_laser_pct=f.gfp_laser_pct,
                        gfp_detector_v=f.gfp_detector_v, threshold_offset=d_off, marrow_bodies=len(m),
                        marrow_laser_range=f"{m.gfp_laser_pct.min():g}-{m.gfp_laser_pct.max():g}",
                        marrow_voltage_range=f"{m.gfp_detector_v.min():g}-{m.gfp_detector_v.max():g}",
                        detectable_share=share, gfp_bodies=int(f.gfp_bodies), tissue_mm2=f.tissue_mm2,
                        corrected_bodies=f.gfp_bodies / share if share > 0 else np.nan))
    det = pd.DataFrame(det)

    # (vi) 54-day ZEISS series: FUT4+ objects and their GFP status in marrow and cochlea
    ser = tf[tf.path.str.startswith(S54 + "647fut4")]
    vals = []
    for p in ser[ser.compartment == "marrow"].path:
        a, meta = read_czi(SRC / p)
        vals.append(fut4_values(a, meta))
        del a
    fthr = float(filters.threshold_otsu(np.concatenate(vals)))
    fut = []
    for _, r in ser.iterrows():
        a, meta = read_czi(SRC / r.path)
        fut += fut4_cells(a, meta, r.path, r.compartment, fthr)
        del a
    fut = pd.DataFrame(fut)
    fut["fut4_threshold"] = fthr
    for comp, g in fut[fut.typed].groupby("compartment"):
        log(f"54 d FUT4+ cells, {comp}: {int(g.gfp_pos.sum())}/{len(g)} GFP+")

    # perilymph fields
    pf = (cl.groupby(["path", "animal", "partner_label"]).agg(nuclei=("gfp_pos", "size"), gfp_pos=("gfp_pos", "sum"))
          .reset_index())
    pf["gfp_pos_pct"] = 100 * pf.gfp_pos / pf.nuclei
    pf = pf.merge(acq[["path", "gfp_laser_pct", "gfp_detector_v", "gfp_offset"]], on="path", how="left")
    # relative GFP sensitivity (488-nm transmission x detector gain, as in (v)) of each field against the median of the
    # same animal's GFP-partner fields
    pf["sensitivity_index"] = pf.gfp_laser_pct * np.exp(beta * (pf.gfp_detector_v - 700))
    ref = pf[pf.partner_label == "GFP"].groupby("animal").sensitivity_index.median()
    pf["sensitivity_vs_gfp_partner"] = pf.sensitivity_index / pf.animal.map(ref)

    # summaries: animal x compartment (tissue) and animal x partner (perilymph)
    summ = []
    for (an, comp, prt), g in tf.groupby(["animal", "compartment", "partner"]):
        d = det[det.path.isin(g.path)] if comp == "cochlea" and len(det) else pd.DataFrame()
        ok = d[d.detectable_share > 0] if len(d) else d
        summ.append(dict(animal=an, compartment=comp, partner=prt, fields=len(g), sections=int(g.planes_analysed.sum()),
                         tissue_mm2=g.tissue_mm2.sum(), gfp_bodies=int(g.gfp_bodies.sum()),
                         pooled_density_per_mm2=g.gfp_bodies.sum() / g.tissue_mm2.sum(),
                         pooled_density_excl_broad=(g.gfp_bodies.sum() - g.broad_spectrum.sum()) / g.tissue_mm2.sum(),
                         field_density_median=g.density_per_mm2.median(), field_density_min=g.density_per_mm2.min(),
                         field_density_max=g.density_per_mm2.max(), measure="GFP+ bodies per mm2 tissue",
                         detectable_share_median=d.detectable_share.median() if len(d) else np.nan,
                         corrected_density_per_mm2=(ok.corrected_bodies.sum() / ok.tissue_mm2.sum())
                         if len(ok) else np.nan,
                         best_plane_density_per_mm2=g.best_plane_bodies.sum() / g.best_plane_tissue_mm2.sum()))
    for (an, prt), g in pf.groupby(["animal", "partner_label"]):
        summ.append(dict(animal=an, compartment="perilymph", partner=prt, fields=len(g), sections=len(g),
                         tissue_mm2=np.nan, gfp_bodies=int(g.gfp_pos.sum()),
                         pooled_density_per_mm2=np.nan, pooled_density_excl_broad=np.nan,
                         field_density_median=g.gfp_pos_pct.median(), field_density_min=g.gfp_pos_pct.min(),
                         field_density_max=g.gfp_pos_pct.max(), nucleated_cells=int(g.nuclei.sum()),
                         pooled_gfp_pos_pct=100 * g.gfp_pos.sum() / g.nuclei.sum(),
                         measure="GFP+ % of nucleated cells"))
    # 931: the 54-day animal's 60x temporal-bone sections (second acquisition; pooled fields, same body rule)
    f931 = pd.read_csv(ROOT / "results" / "r9_figures" / "f4_parabiont_fields.csv")
    f931 = f931[f931.pooled.astype(str).str.lower() == "true"]
    for comp, g in f931.groupby("compartment"):
        summ.append(dict(animal="54 d", compartment=comp, partner="WT", fields=len(g), sections=len(g),
                         tissue_mm2=g.tissue_mm2.sum(), gfp_bodies=int(g.gfp_bodies.sum()),
                         pooled_density_per_mm2=g.gfp_bodies.sum() / g.tissue_mm2.sum(),
                         pooled_density_excl_broad=np.nan, field_density_median=g.gfp_bodies_per_mm2.median(),
                         field_density_min=g.gfp_bodies_per_mm2.min(), field_density_max=g.gfp_bodies_per_mm2.max(),
                         measure="GFP+ bodies per mm2 tissue", source="931 (60x sections)"))
    summ = pd.DataFrame(summ)
    summ["source"] = summ.get("source", pd.Series(index=summ.index, dtype=object)).fillna("934")

    # per-animal cochlea / marrow ratio (wild-type partners with both compartments) and exact sign tests
    wt = summ[(summ.partner == "WT") & summ.compartment.isin(["marrow", "cochlea"])]
    rows = []
    for (an, src), g in wt.groupby(["animal", "source"]):
        g = g.set_index("compartment")
        if not {"marrow", "cochlea"} <= set(g.index):
            continue
        mar, coc = g.loc["marrow", "pooled_density_per_mm2"], g.loc["cochlea", "pooled_density_per_mm2"]
        cor = g.loc["cochlea"].get("corrected_density_per_mm2", np.nan)
        bm, bc = (g.loc[c_].get("best_plane_density_per_mm2", np.nan) for c_ in ("marrow", "cochlea"))
        rows.append(dict(animal=an, source=src, primary=src == "934", noise="noise" in an, marrow_density=mar,
                         cochlea_density=coc, cochlea_to_marrow=coc / mar,
                         cochlea_corrected_density=cor, corrected_cochlea_to_marrow=cor / mar,
                         cochlea_detectable_share=g.loc["cochlea"].get("detectable_share_median", np.nan),
                         marrow_best_plane_density=bm, cochlea_best_plane_density=bc,
                         best_plane_cochlea_to_marrow=bc / bm))
    ratio = pd.DataFrame(rows)
    prim = ratio[ratio.primary]
    for tag, sub in (("non_noise", prim[~prim.noise]), ("all", prim)):
        for col, lab_ in (("cochlea_to_marrow", "observed"), ("corrected_cochlea_to_marrow", "corrected"),
                          ("best_plane_cochlea_to_marrow", "best_plane")):
            v = sub[col].dropna()
            k, n = int((v < 1).sum()), len(v)
            p = binomtest(k, n, 0.5, alternative="two-sided").pvalue if n else np.nan
            ratio[f"sign_{lab_}_{tag}_k_of_n"] = f"{k}/{n}"
            ratio[f"sign_{lab_}_{tag}_p"] = p
            log(f"cochlea < marrow ({lab_}, {tag}): {k}/{n}, exact sign test P = {p:.3g}")

    acq_out = acq.drop(columns=["session", "specimen"])
    write_csv(acq_out, MOD, "parabiosis_acquisitions.csv", "Parabiosis survey (934): every CZI of the collection with "
              "labels, pixel size, channels, stage position and the inclusion decision with its reason",
              required=["path", "included", "reason"])
    write_csv(tf, MOD, "parabiosis_tissue_fields.csv", "Parabiosis survey (934): one row per included tissue "
              "acquisition (marrow, cochlea, liver): sections analysed, tissue area, GFP+ bodies, densities, partner",
              required=["path", "animal", "compartment", "partner", "density_per_mm2"])
    write_csv(cl, MOD, "parabiosis_perilymph_cells.csv", "Parabiosis survey (934): perilymph nucleated cells with "
              "cytoplasmic-ring GFP, nuclear GFP, local background and GFP status", required=["path", "gfp_pos"])
    write_csv(pf, MOD, "parabiosis_perilymph_fields.csv", "Parabiosis survey (934): perilymph fields, nucleated "
              "cells and GFP+ share", required=["path", "nuclei", "gfp_pos_pct"])
    write_csv(summ, MOD, "parabiosis_summary.csv", "Parabiosis survey (934): animal x compartment x partner summary "
              "(pooled, field and detection-corrected densities; perilymph GFP+ shares)",
              required=["animal", "compartment", "partner"])
    write_csv(ratio, MOD, "parabiosis_ratio.csv", "Parabiosis survey (934): per-animal cochlea / marrow density of "
              "wild-type partners, observed and detection-corrected, with exact sign tests", required=["animal",
              "cochlea_to_marrow"])
    write_csv(det, MOD, "parabiosis_detection.csv", "Parabiosis survey (934): detection check per cochlear field "
              "(share of the animal's marrow GFP+ bodies detectable at the field's settings; corrected count)",
              required=["path", "detectable_share"])
    write_csv(pd.DataFrame([gref]), MOD, "parabiosis_gain_reference.csv", "Parabiosis survey (934): detector-gain "
              "factor from one 54-day position imaged at two GFP voltages", required=["beta_per_volt"])
    write_csv(pl, MOD, "parabiosis_planes.csv", "Parabiosis survey (934): every analysed optical section of the included "
              "tissue acquisitions (tissue area, GFP+ bodies, GFP median, robust SD, threshold, 99.9th percentile)",
              required=["path", "plane", "gfp_bodies", "gfp_p999"])
    write_csv(fut, MOD, "parabiosis_fut4_cells.csv", "Parabiosis survey (934): FUT4+ objects of the 54-day ZEISS "
              "series (marrow, cochlea) and their GFP status", required=["path", "compartment", "gfp_pos"])
    write_csv(ob, MOD, "parabiosis_objects.csv", "Parabiosis survey (934): GFP+ bodies of the analysed sections "
              "(centroids in pixels, area, GFP, signal above median, broad-spectrum flag) for figure crops",
              required=["path", "plane"])
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string())
    print(ratio.round(3).to_string())
    print(det.round(3).to_string())


if __name__ == "__main__":
    main()
