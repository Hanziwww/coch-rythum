"""Module 5.2b — rhythmic communication.

(1) For each interaction present in >= 5 of the 6 samples of a tissue: cosinor on log2(lr_means) (CellPhoneDB-type
    magnitude) with 60-class permutation p and LOTO; BH within tissue. Mechanistic class from the gene-level rhythms
    (results/33_rhythm_tiers, primary limma fits): ligand in the sender, receptor in the receiver —
    ligand-driven / receptor-driven / both in phase (|Δφ| < 3 h) / antiphase (|Δφ − 12| < 3 h) / both other / none
    (a gene counts as rhythmic at limma p < 0.05 & FC >= 1.25; complexes use their first subunit).
(2) Tensor-cell2cell (LIANA+ → cell2cell): samples × LR × sender × receiver tensor of −log10 magnitude ranks,
    non-negative decomposition with elbow-selected rank; each factor's 6 sample loadings tested for 24-h rhythm.

Outputs (results/64_ccc_rhythm/): ccc_rhythm_<tissue>.csv, tensor_factors_<tissue>.csv (loadings per dimension),
                                  tensor_factor_rhythm.csv
"""
import sys
from pathlib import Path

import liana as li
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "64_ccc_rhythm"
KEY = ["source", "target", "ligand_complex", "receptor_complex"]
fr_rows = []
for tissue in ["cochlea", "bm"]:
    res = pd.read_parquet(CACHE / "63_ccc" / f"liana_{tissue}.parquet")
    genes = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                        usecols=["group", "gene", "p", "ptt_fc", "peak_zt", "tier"])
    g_idx = genes.set_index(["group", "gene"])
    piv = res.pivot_table(index=KEY, columns="zt", values="lr_means", aggfunc="first")
    piv = piv[piv.notna().sum(1) >= 5]
    piv = piv.apply(lambda r: r.fillna(r.min()), axis=1)
    zt = piv.columns.to_numpy(float)
    Y = np.log2(piv.to_numpy().T + 1e-3)
    f = cosinor.fit(Y, zt)
    _, _, pp, _ = cosinor.permutation_test(Y, zt)
    lo = cosinor.loto(Y, zt)
    out = piv.index.to_frame(index=False)
    out["tissue"] = tissue
    out["mean_lr_means"] = piv.mean(1).to_numpy()
    for k, v in {"amp_log2": f["amp"], "ptt_fc": 2 ** (2 * f["amp"]), "peak_zt": f["peak"], "p": f["p"],
                 "p_perm": pp, "loto_phase_sd_h": lo["loto_phase_sd_h"]}.items():
        out[k] = v
    out["q_BH"] = multipletests(out["p"], method="fdr_bh")[1]

    def gene_rhythm(cell, complex_):
        gname = str(complex_).split("_")[0]
        try:
            r = g_idx.loc[(cell, gname)]
        except KeyError:
            return np.nan, np.nan, False
        return r["peak_zt"], r["ptt_fc"], bool(r["p"] < 0.05 and r["ptt_fc"] >= 1.25)

    lig = [gene_rhythm(s, l) for s, l in zip(out["source"], out["ligand_complex"])]
    rec = [gene_rhythm(t, r) for t, r in zip(out["target"], out["receptor_complex"])]
    out["ligand_peak_zt"], out["ligand_ptt_fc"], out["ligand_rhythmic"] = zip(*lig)
    out["receptor_peak_zt"], out["receptor_ptt_fc"], out["receptor_rhythmic"] = zip(*rec)
    d = np.abs(cosinor.circ_diff(out["ligand_peak_zt"].to_numpy(float), out["receptor_peak_zt"].to_numpy(float)))
    lr_ = out["ligand_rhythmic"].to_numpy(bool)
    rr_ = out["receptor_rhythmic"].to_numpy(bool)
    out["mechanism"] = np.select([lr_ & rr_ & (d < 3), lr_ & rr_ & (np.abs(d - 12) < 3), lr_ & rr_, lr_, rr_],
                                 ["both_in_phase", "antiphase", "both_other_phase", "ligand_driven",
                                  "receptor_driven"], default="neither_rhythmic")
    write_csv(out.sort_values("p"), MOD, f"ccc_rhythm_{tissue}.csv",
              f"24-h rhythm of LIANA+ interaction magnitudes ({tissue}) with ligand/receptor mechanism classes",
              required=KEY + ["tissue", "peak_zt", "p", "p_perm", "q_BH", "mechanism"])
    log(f"{tissue}: {len(out)} interactions tested; q<0.1: {(out['q_BH'] < 0.1).sum()}")

    # ---- Tensor-cell2cell
    try:
        import tensorly as tl
        import torch
        dev = "cuda" if torch.cuda.is_available() else None      # tensorly on the PyTorch/CUDA backend
        if dev:
            tl.set_backend("pytorch")
        tensor = li.multi.to_tensor_c2c(liana_res=res, sample_key="sample_id", score_key="magnitude_rank",
                                        how="outer_cells", non_negative=True,   # magnitude_rank → −log10 (LIANA default)
                                        device=dev)
        tensor.elbow_rank_selection(upper_rank=10, runs=3, init="random", automatic_elbow=True, random_state=0)
        rank = int(getattr(tensor, "rank", None) or 6)
        tensor.compute_tensor_factorization(rank=rank, init="random", random_state=0, runs=3,
                                            normalize_loadings=True)
        facs = tensor.factors
        rows = []
        for dim, df in facs.items():
            long = df.reset_index().melt(id_vars=df.index.name or "index", var_name="factor", value_name="loading")
            long.columns = ["element", "factor", "loading"]
            long.insert(0, "dimension", dim)
            rows.append(long)
        tf = pd.concat(rows, ignore_index=True)
        tf.insert(0, "tissue", tissue)
        write_csv(tf, MOD, f"tensor_factors_{tissue}.csv", f"Tensor-cell2cell factor loadings ({tissue}, rank {rank})",
                  required=["tissue", "dimension", "element", "factor", "loading"])
        ctx = facs[list(facs)[0]]           # context (sample) loadings
        zt_ctx = res.drop_duplicates("sample_id").set_index("sample_id")["zt"].reindex(ctx.index).to_numpy(float)
        ff = cosinor.fit(ctx.to_numpy(), zt_ctx)
        _, _, fpp, _ = cosinor.permutation_test(ctx.to_numpy()[np.argsort(zt_ctx)], np.sort(zt_ctx))
        for j, fac in enumerate(ctx.columns):
            fr_rows.append({"tissue": tissue, "factor": fac, "rank": rank, "mesor": ff["mesor"][j],
                            "amplitude": ff["amp"][j], "peak_zt": ff["peak"][j], "p": ff["p"][j], "p_perm": fpp[j]})
    except Exception as e:  # tensor step is complementary; record failure and continue
        log(f"tensor-cell2cell failed for {tissue}: {type(e).__name__}: {e}")
if fr_rows:
    write_csv(pd.DataFrame(fr_rows), MOD, "tensor_factor_rhythm.csv",
              "24-h rhythm of Tensor-cell2cell factor sample loadings", required=["tissue", "factor", "peak_zt", "p"])
log("done")
