# PR104 Exact Replay Readiness - 2026-05-08

Status: `PASS` for dispatch readiness, `NO_SCORE_CLAIM` for frontier status.
No GPU dispatch was attempted.

## Verdict

PR104 `qhnerv_ft_best` is now dispatchable through the canonical path:

```bash
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best/archive.zip --inflate-sh experiments/public_runtime_adapters/pr104_qhnerv_ft_best_adapter/inflate.sh --upstream-dir upstream --device cuda
```

Before running that command on GPU infrastructure, claim the lane with
`tools/claim_lane_dispatch.py`. The result must remain unpromoted until exact
CUDA auth eval returns through this checkout.

## Custody

- PR: `104`, `qhnerv_ft_best`, author `patattzel`, head SHA `f1c59d895325f2d2835e843ce72be3443983a4b4`.
- Archive: `178637` bytes, SHA-256 `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`.
- ZIP members: one member, `0.bin`, `178529` bytes, SHA-256 `0a0f2cac1961f3ab5128e70ff10c0287a66949d435b5ee7a4dcf2017917910a3`.
- Public report: external CUDA report over `600` samples with PoseNet `0.00016895`, SegNet `0.00070710`, rounded score `0.23`.
- Recomputed from public rounded components: `0.23076057363801453`.

## Adapter

- Path: `experiments/public_runtime_adapters/pr104_qhnerv_ft_best_adapter/inflate.sh`.
- SHA-256: `d82335c30216d49b0c80a31665feb351774236682a05069194910a1a3a8a9ea9`.
- Runtime dependency root: `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best`.
- Runtime manifest SHA-256 from `experiments/contest_auth_eval.py`: `55c4358a22974ac3c174d185e81b22db47ee0a97e6b5199d9c36c51686e134c4`.

The adapter preserves archive bytes and does not install packages or load the
scorer in inflate. It checks for `brotli`, `numpy`, and `torch`, then maps the
canonical file-list base `0.mkv` to extracted `0.bin`. It also fails closed if
both `0.bin` and `x` are present or if neither member exists.

## Local Smokes

- `bash -n experiments/public_runtime_adapters/pr104_qhnerv_ft_best_adapter/inflate.sh`: pass.
- `unzip -t .../qhnerv_ft_best/archive.zip`: pass.
- `.venv/bin/python -m pytest src/tac/tests/test_pr104_exact_replay_readiness.py`: pass, 3 tests.
- Adapter missing-member smoke: pass, fail-closed with exit code `6`.
- Operator-validated fail-closed cases from status check: empty file list,
  missing payload, and ambiguous payload. These are guard evidence only and do
  not affect score status.
- `experiments/contest_auth_eval.py` runtime manifest smoke: pass, runtime tree
  SHA-256 `55c4358a22974ac3c174d185e81b22db47ee0a97e6b5199d9c36c51686e134c4`,
  one external dependency root, 21 external runtime files.
- `scripts/lightning_exact_eval_repro.py` public preflight shape check: pass.
- CPU-safe parse/model-load smoke: pass. Parsed metadata is `latent_dim=28`,
  `base_channels=36`, `eval_size=[384,512]`, `n_pairs=600`, latent shape
  `[600,28]`, decoder tensor count `28`, strict `load_state_dict` had no
  missing or unexpected keys.

Local CUDA was unavailable, so no local CUDA score run was attempted. This is
not a negative result for PR104.

## Blockers

- Dispatch readiness blockers: none.
- Score promotion blocker: `exact_cuda_auth_eval_not_run_by_design_for_this_task`.
