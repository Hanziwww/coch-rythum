# Software environments

The analyses ran in two places:

- **Analysis server (Linux)**: Python 3.12.3 and
  R 4.5.3, for everything from quality control to rhythm statistics,
  composition models, external data and the evidence register. GPU training (scVI / scANVI) used CUDA.
- **Workstation**: Python 3.10.14 for imaging, flow cytometry, figures,
  Supplementary Tables, Source Data and the manuscript checks (scripts 900-937 that state a local run).

| File | Content |
| --- | --- |
| `software_versions_server.csv` | every package imported by the server analyses, with the version used |
| `software_versions_local.csv` | the workstation packages, with the version used |
| `environment-server-python.yml` | conda environment pinned to the server Python versions |
| `requirements-trajectory-overlay.txt` | separate overlay (own SciPy) for the trajectory checks of script 630 |
| `environment-server-r.yml` | conda environment pinned to the server R versions (CRAN and Bioconductor packages) |
| `r_packages.csv` | R packages with source and install route; 5 packages come from GitHub |
| `environment-local-python.yml` | conda environment pinned to the workstation versions |

```bash
conda env create -f code/env/environment-server-python.yml
conda env create -f code/env/environment-server-r.yml
conda env create -f code/env/environment-local-python.yml
pip install --target envs/traj_pkgs -r code/env/requirements-trajectory-overlay.txt
```

Install the GitHub R packages with `remotes::install_github()` as listed in `r_packages.csv`. The shell chains in
`code/scripts/setup/` call the interpreters named by `COCH_PYTHON` and `COCH_RSCRIPT` (default: `python` and `Rscript`
on the PATH) and look for environments in `COCH_ENVS` (default: `envs/` in the project root). The original
installation scripts are kept in `code/scripts/setup/` (`env_python.sh`, `env_r.sh`, `env_r_deps.sh`, `env_r3.sh`,
`r_extra_pkgs.R`, `github_clone.sh`); they used package mirrors that you may replace with the default channels.
