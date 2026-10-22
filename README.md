# Cochlear and temporal-bone marrow day-night atlas: analysis code

Code accompanying the unpublished manuscript *A day-night atlas of the inner ear and its marrow reveals an imported immune timetable* and its Science submission version.

The analysis code is public under the Apache License 2.0. The accompanying data and analysis archive is deposited at Zenodo as restricted-access version 1.1, DOI **10.5281/zenodo.23555378**. Data access requires approval from the corresponding authors. This code archive includes the tissue-comparison update of 22 October 2026.

## Contents

- `code/scripts/`: the original analysis, figure, table and manuscript scripts.
- `code/env/`: recorded environments and package versions.
- `code/docs/`: pipeline documentation and script index.
- `code/prepare_workspace.py`: prepares the expected workspace layout.
- `data/external/`: public-dataset accessions and download instructions.

The original experimental data and large analysis outputs are held in the Zenodo archive, not in Git. Copy the corresponding archive folders into the same project root before running analyses.

## Reproduction

1. Recreate the appropriate environment following `code/env/README.md`.
2. Place `data/processed`, `data/flow`, `data/imaging` and the required `results/` beside `code/` using the restricted data release.
3. Run `python code/prepare_workspace.py` to create the `scripts/` workspace link; the optional image/flow extraction steps are described in the release documentation.
4. Set `COCH_ROOT` to the project root. Run scripts in the sequence documented in `code/docs/pipeline.md`; the shell chains under `code/scripts/setup/` record the original workflow.

Existing manuscript-generation scripts reflect the original workflow. The updated submission LaTeX files and compiled PDFs are distributed with the submission package and data archive.

## Authors and contributions

Han Wang, Yao Cai, Yekang Mo, Ye Tian, Weibang Chen, Dexiao Zeng, Hongyu Ning, Yue Xu, Shuangjie Dai, Xiaoyu Liu, Ruoyun He, Huafeng Liu, Junmin Huang, Junfeng Hao, Yusu Ni, Chunjie Tian.

Han Wang performed all computational work, including data curation, formal analysis, software development and visualization, and contributed to manuscript writing. Yao Cai, Yekang Mo, Ye Tian, Weibang Chen, Dexiao Zeng, Hongyu Ning, Yue Xu, Shuangjie Dai, Xiaoyu Liu and Ruoyun He performed the experiments. Huafeng Liu, Junmin Huang, Junfeng Hao, Yusu Ni and Chunjie Tian reviewed the manuscript and provided feedback.

Corresponding authors: Junfeng Hao (ygzhjf85@gmail.com), Yusu Ni (niyusu@aliyun.com), and Chunjie Tian (tianchunjie@gdmu.edu.cn).

## Funding

This work was supported by the National Natural Science Foundation of China (grant nos. 82160215, 81660174 and 82271169); the Guangdong Provincial Key Laboratory of Autophagy and Major Chronic Non-communicable Diseases (grant no. 2022B1212030003); the Research Startup Funds for High-level Talents in the Affiliated Hospital of Guangdong Medical University (grant no. GCC20220013); the College Students' Innovative Entrepreneurial Training Plan Program (grant no. S202510571025); and the 2025 Artificial Intelligence-Enabled Medical Device Innovation Task (Open Competition Mechanism, project no. 2025-AISJK-02007).

## Competing interests

The authors declare no competing interests.

## Licence

The code is licensed under Apache-2.0. Data access and reuse information are supplied with the accompanying data archive.
