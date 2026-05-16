# L5 v2 paper-fidelity research-basis wire-in - 2026-05-16

Scope: Time-Traveler L5 v2 / Cathedral autopilot planning provenance.

Status: landed locally in `src/tac/optimization/research_basis.py`; planning
only. This memo records source fidelity and the exact contest boundary. It is
not score evidence.

## Change

The canonical `time_traveler_l5_v2` research-basis family now includes three
previously missing source anchors:

- `atick_redlich_1990` for efficient coding / natural-image redundancy.
- `wyner_ziv_1976` for decoder-side information and residual coding.
- `lu_dvc_2019` for end-to-end learned video compression with motion and
  residual streams.

`lu_dvc_2019` was added as a first-class research source with charged-byte
contracts and hardening blockers:

- motion/residual payload custody;
- runtime consumption proof;
- paired CPU/CUDA empirical anchor;
- exact CUDA auth eval.

## Boundary

These sources may influence L5 v2 planning, paper citations, and dispatch
ranker descriptions. They do not create rank reward, promotion eligibility, or
score claims. Any L5 v2 result remains blocked until a byte-closed archive has
archive SHA, runtime tree SHA, consumed side-info/residual bytes, paired CPU and
CUDA eval evidence, and component recomputation.

## Source anchors

- Rao and Ballard 1999, predictive coding in visual cortex:
  https://www.nature.com/articles/nn0199_79
- Atick and Redlich 1990, early visual processing:
  https://doi.org/10.1162/neco.1990.2.3.308
- Wyner and Ziv 1976, rate-distortion with decoder side information:
  https://doi.org/10.1109/TIT.1976.1055508
- Lu et al. 2019, DVC:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html
- Tishby, Pereira, and Bialek 2000, Information Bottleneck:
  https://arxiv.org/abs/physics/0004057
- Balle et al. 2018, scale hyperprior:
  https://arxiv.org/abs/1802.01436
- HNeRV 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_research_basis.py src/tac/tests/test_l5_staircase_v2.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/research_basis.py src/tac/tests/test_research_basis.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m py_compile src/tac/optimization/research_basis.py src/tac/optimization/l5_staircase_v2.py`
