# PR75 Raw-Output Parity Harness - 2026-05-03

Scope: local-only forensic parity harness for public PR75
`qpose14_r55_segactions_minp` versus `submissions/robust_current`. No remote
GPU jobs were dispatched. The output is empirical raw/tensor parity evidence,
not score evidence.

## Artifact

- Tool: `experiments/pr75_raw_output_parity.py`
- Test: `src/tac/tests/test_pr75_raw_output_parity.py`
- Report:
  `experiments/results/pr75_raw_output_parity_20260503_codex/pr75_raw_output_parity.json`
- Public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_deep_codex/downloads/pr75_pr67_qpose14_r55_segactions_minp_archive.zip`
- Public archive SHA-256:
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`

## Finding

The public and robust decoded masks/actions/renderer bytes are not the primary
divergence. The robust path changes the PR75 pose runtime contract:

- public PR75 decodes QP1 poses directly to float32 before
  `JointFrameGenerator`;
- `robust_current` materializes the same QP1 stream as fp16
  `optimized_poses.bin`, then reloads it as float32.

The report measured `518 / 3600` pose values changed by this fp16 boundary,
with max absolute pose drift `0.015625`.

Selected render parity over pairs `0,33,36,587`:

| Pair | Native changed values | Native max abs | Raw changed bytes | First raw diff | Public-pose override raw parity |
|---:|---:|---:|---:|---:|---|
| 0 | 589307 | 0.05426788330078125 | 16261 | 211 | exact |
| 33 | 589318 | 0.053985595703125 | 17316 | 34 | exact |
| 36 | 588276 | 0.017852783203125 | 5686 | 64 | exact |
| 587 | 589649 | 0.1472015380859375 | 49152 | 13 | exact |

Counterfactual result: when the robust renderer path is fed the public QP1
float32 pose values, the selected renderer-native tensors and camera-resolution
raw bytes become byte-identical to the public PR75 runtime. This isolates the
runtime regression to pose precision materialization, not QZS3 loader parity,
mask duplication, or tile-action application.

## Commands

```bash
.venv/bin/python -m py_compile experiments/pr75_raw_output_parity.py src/tac/tests/test_pr75_raw_output_parity.py
.venv/bin/python -m pytest src/tac/tests/test_pr75_raw_output_parity.py -q
.venv/bin/python experiments/pr75_raw_output_parity.py --skip-render --force
.venv/bin/python experiments/pr75_raw_output_parity.py --pair-indices 0,33,36,587 --force
```

## Highest-EV Fix

Preserve public PR75 QP1 pose precision at inflate time in the robust runtime
contract. The least invasive next experiment is to add a charged/logical QP1
pose member or metadata-marked packed-payload path that decodes QP1 directly to
float32 for QZS3/JointFrameGenerator, instead of expanding to fp16
`optimized_poses.bin`.

Promotion remains blocked until a byte-closed archive with this runtime
contract passes exact CUDA auth eval through
`archive.zip -> inflate.sh -> upstream/evaluate.py`.
