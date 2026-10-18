"""Round 10 (D137) — robustness of the pre-specified flow readouts (Fig. 5) to the fixed gate thresholds (user: "Please
ensure that the current quantification and labeling algorithms are sufficiently robust"). Every event of the validation
cohort (data/flow_cochlea/events.csv; fixed scatter and singlet flags as exported) is re-gated with the thresholds of
gates.csv, first unchanged and then with each threshold
of the P1-P3 gates moved by -10% and +10% in turn, and with all fluorescence thresholds moved together. The export
rounds bead-relative values to four significant digits and scatter values to integers, so each event's true value lies
within half a unit of the last digit; with unchanged thresholds every exported count of tube_values.csv lies between the events certainly and the
events possibly inside the gate (asserted for every gate and tube), so the re-gating reproduces the cytometer software
up to events tied with a gate boundary at the export precision. P1-P3 are
recomputed with the pre-specified formulas and event floors and tested with 932's primary model (OLS of log % of CD45+
on ZT and cohort; Holm across P1-P3). The pre-specified result stays primary; this is a sensitivity analysis.
Outputs (lib/io.write_csv, results/r9_flow/): robustness_flow_gates.csv (per perturbation and readout: ZT17 / ZT5
ratio, 95% CI, P, Holm P, verdict) and robustness_flow_summary.csv (per readout: perturbations tested, replicated,
ratio range).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/940_flow_gate_robustness.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import log, write_csv  # noqa: E402

_spec = importlib.util.spec_from_file_location("s932", str(Path(__file__).resolve().parent / "932_flow_validation.py"))
S932 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S932)
FLOW = S932.FLOW
FLUO = ["CD45_pos", "Ly6G_pos", "CD11b_pos", "Ly6C_hi", "B220_pos", "B220_hi", "IgD_pos", "Ly6G_split", "CD11b_split"]
SCATTER = ["neutrophil_SSC_min", "mono_FSC_low", "mono_FSC_high", "mono_SSC_low", "mono_SSC_high"]
TUBES = ("T1", "T2", "T3", "T6")


def load_events():
    cols = ["record_id", "tube", "FSC_A", "SSC_A", "channel1_bead_relative", "channel2_bead_relative", "gate_singlets"]
    e = pd.read_csv(FLOW / "events.csv", usecols=cols, encoding="utf-8-sig",
                    dtype={"record_id": "category", "tube": "category", "gate_singlets": "int8"})  # float64 as 932
    e = e[e.tube.isin(TUBES) & (e.gate_singlets == 1)]
    return e


def half_unit(v):
    """Half a unit of the fourth significant digit (the export precision of the bead-relative values)."""
    a = np.abs(v)
    return np.where(a > 0, 0.5 * 10.0 ** (np.floor(np.log10(np.where(a > 0, a, 1.0))) - 3), 0.0)


def counts(e, g, mode="point"):
    """Per (record_id, tube) the event counts of the P1-P3 gates (gate_specs semantics: [lo, hi) in raw units).
    mode 'point' gates the exported values; 'certain' / 'possible' count the events certainly / possibly inside each
    gate given the export rounding (bead-relative values: four significant digits; scatter: integers)."""
    c1, c2 = e.channel1_bead_relative.to_numpy(), e.channel2_bead_relative.to_numpy()
    u1, u2 = (half_unit(c1), half_unit(c2)) if mode != "point" else (0.0, 0.0)
    us = 0.5 if mode != "point" else 0.0
    sgn = {"point": 0.0, "certain": 1.0, "possible": -1.0}[mode]

    def ge(c, u, thr):     # true value >= thr
        return c - sgn * u >= thr

    def lt(c, u, thr):     # true value < thr
        return c + sgn * u < thr
    ssc, fsc = e.SSC_A.to_numpy(), e.FSC_A.to_numpy()
    t = e.tube.astype(str).to_numpy()
    hi = ge(ssc, us, g["neutrophil_SSC_min"])
    f = {"singlets": np.ones(len(e), bool)}
    f["cd45"] = (t == "T1") & ge(c1, u1, g["CD45_pos"])
    f["ly6g_T1"] = f["cd45"] & hi & ge(c2, u2, g["Ly6G_pos"])
    f["ly6g_T2"] = (t == "T2") & hi & ge(c2, u2, g["Ly6G_pos"])
    f["int"] = (f["ly6g_T2"] & ge(c1, u1, g["CD11b_pos"]) & lt(c1, u1, g["CD11b_split"])
                & lt(c2, u2, g["Ly6G_split"]))
    mono = (ge(fsc, us, g["mono_FSC_low"]) & lt(fsc, us, g["mono_FSC_high"]) & ge(ssc, us, g["mono_SSC_low"])
            & lt(ssc, us, g["mono_SSC_high"]))
    f["mono_scatter"] = (t == "T3") & mono
    f["mono"] = f["mono_scatter"] & ge(c1, u1, g["Ly6C_hi"]) & ge(c2, u2, g["CD11b_pos"])
    f["b220"] = (t == "T6") & ge(c1, u1, g["B220_pos"])
    f["target"] = f["b220"] & ge(c1, u1, g["B220_hi"]) & ge(c2, u2, g["IgD_pos"])
    df = pd.DataFrame({k: v.astype(np.int32) for k, v in f.items()})
    df["record_id"], df["tube"] = e.record_id.astype(str).to_numpy(), t
    return df.groupby(["record_id", "tube"]).sum()


def readouts(ct, sv):
    """P1-P3 per mouse x tissue with the pre-specified formulas and floors (932 readouts)."""
    rows = []
    for _, s in sv.iterrows():
        co = s.tissue == S932.CO
        get = {}
        for tb in TUBES:
            key = (s.sample_id, tb)
            if key in ct.index:
                r = ct.loc[key]
                if co and r.singlets < 3000:
                    continue
                get[tb] = r
        d = dict(sample_id=s.sample_id, mouse_id=s.mouse_id, tissue=s.tissue, cohort=int(s.cohort), zt=s.zt_group,
                 flag_cd11b_T2_T3=False)
        if "T1" in get:
            t1 = get["T1"]
            d["cd45_pct"] = 100 * t1.cd45 / t1.singlets
            d["neut"] = 100 * t1.ly6g_T1 / t1.cd45
            if "T2" in get and get["T2"].ly6g_T2 >= 100:
                d["P1"] = d["neut"] * (100 * get["T2"].int / get["T2"].ly6g_T2) / 100
            if "T3" in get:
                d["P2"] = 100 * (get["T3"].mono / get["T3"].singlets) / (d["cd45_pct"] / 100)
            if "T6" in get and get["T6"].b220 >= 100:
                d["P3"] = 100 * (get["T6"].target / get["T6"].singlets) / (d["cd45_pct"] / 100)
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    gates = pd.read_csv(FLOW / "gates.csv", encoding="utf-8-sig").set_index("parameter")["value"].astype(float)
    sv = pd.read_csv(FLOW / "sample_values.csv", encoding="utf-8-sig")
    tv = pd.read_csv(FLOW / "tube_values.csv", encoding="utf-8-sig")
    tv = tv[(tv.record_type == "biological") & (tv.status == "run") & tv.tube.isin(TUBES)].set_index(["record_id", "tube"])
    e = load_events()
    log(f"events: {len(e):,} single-cell events in tubes {', '.join(TUBES)}")
    lo_, hi_ = counts(e, gates, "certain"), counts(e, gates, "possible")
    # with unchanged gates every exported count lies between the events certainly and possibly inside the gate
    chk = lo_.join(hi_, rsuffix="_hi").join(tv[["singlets_events", "cd45_events", "ly6g_events", "int_events",
                                               "mono_scatter_events", "mono_events", "b220_events", "target_events"]],
                                           how="inner")
    pairs = [("singlets", "singlets_events", None), ("cd45", "cd45_events", "T1"), ("ly6g_T1", "ly6g_events", "T1"),
             ("ly6g_T2", "ly6g_events", "T2"), ("int", "int_events", "T2"), ("mono_scatter", "mono_scatter_events", "T3"),
             ("mono", "mono_events", "T3"), ("b220", "b220_events", "T6"), ("target", "target_events", "T6")]
    n_chk, tied = 0, 0
    for mine, theirs, tb in pairs:
        x = chk if tb is None else chk[chk.index.get_level_values("tube") == tb]
        x = x[[mine, mine + "_hi", theirs]].dropna()
        ok = (x[mine] <= x[theirs]) & (x[theirs] <= x[mine + "_hi"])
        assert len(x) and ok.all(), (mine, theirs, x[~ok].head())
        n_chk, tied = n_chk + len(x), max(tied, int((x[mine + "_hi"] - x[mine]).max()))
    log(f"unchanged gates reproduce all {n_chk} exported gate counts of tube_values.csv (at most {tied} events per "
        "gate tied with a boundary at the export precision)")
    settings = [("published gates", None, 1.0)]
    for par in FLUO + SCATTER:
        for fac in (0.9, 1.1):
            settings.append((f"{par} x {fac:g}", par, fac))
    settings += [("all fluorescence thresholds x 0.9", "fluorescence", 0.9),
                 ("all fluorescence thresholds x 1.1", "fluorescence", 1.1)]
    rows = []
    for name, par, fac in settings:
        g = gates.copy()
        if par == "fluorescence":
            g[FLUO] = g[FLUO] * fac
        elif par is not None:
            g[par] = g[par] * fac
        r = readouts(counts(e, g), sv)
        t = S932.primary_tests(r)
        for x in t.itertuples():
            rows.append(dict(perturbation=name, parameter=par or "", factor=fac, readout=x.test, tissue=x.tissue,
                             n_zt5=x.n_zt5, n_zt17=x.n_zt17, ratio_zt17_zt5=x.effect, ci_lo=x.ci_lo, ci_hi=x.ci_hi,
                             p=x.p, p_holm=x.p_holm, verdict=x.verdict))
        log(f"{name}: " + "; ".join(f"{x.test} {x.effect:.2f} ({x.verdict})" for x in t.itertuples()))
    d = pd.DataFrame(rows)
    pub = pd.read_csv(Path(S932.ROOT) / "results" / "r9_flow" / "flow_tests.csv")
    pub = pub[pub.family == "primary"].set_index("test")
    b = d[d.perturbation == "published gates"].set_index("readout")
    dev = (b.ratio_zt17_zt5 / pub.loc[b.index, "effect"] - 1).abs()
    assert (dev < 0.01).all() and (b.verdict == pub.loc[b.index, "verdict"]).all(), (b, pub)
    log(f"re-gated baseline vs published ratios: largest relative difference {dev.max():.2%}")
    sm = d.groupby("readout").agg(perturbations=("perturbation", "size"),
                                  replicated=("verdict", lambda v: int((v == "replicated").sum())),
                                  ratio_min=("ratio_zt17_zt5", "min"), ratio_max=("ratio_zt17_zt5", "max"),
                                  holm_p_max=("p_holm", "max")).reset_index()
    for r_ in sm.itertuples():
        log(f"{r_.readout}: replicated at {r_.replicated}/{r_.perturbations} gate settings; ratio "
            f"{r_.ratio_min:.2f}-{r_.ratio_max:.2f}; Holm P <= {r_.holm_p_max:.3g}")
    write_csv(d, "r9_flow", "robustness_flow_gates.csv", "Robustness (940, D137): P1-P3 of the pre-specified flow cohort "
              "re-gated from every event with each P1-P3 gate threshold moved by -10% / +10% (and all fluorescence "
              "thresholds together): ZT17 / ZT5 ratio, 95% CI, P, Holm P and verdict (published gates reproduce 932)",
              required=["perturbation", "readout", "ratio_zt17_zt5", "p_holm", "verdict"])
    write_csv(sm, "r9_flow", "robustness_flow_summary.csv", "Robustness (940, D137): per primary readout, gate "
              "perturbations tested, replicated, ratio range and largest Holm P",
              required=["readout", "perturbations", "replicated"])


if __name__ == "__main__":
    main()
