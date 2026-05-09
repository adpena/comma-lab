# Delta-Epsilon-Zeta Phase 1 Targets + A1 EMA Smoke - 2026-05-09

## Classification

- Evidence grade: `[empirical_planning; local CPU sanity loop]`
- Score claim: `false`
- Remote dispatch: `none`
- Dispatch claim required: `no` (local CPU-only target build and training-driver smoke)
- Result class: implementation/custody progress, not score promotion

This pass moved the delta-epsilon-zeta Phase 1 path from memo-only strategy into
two concrete artifacts:

1. A PR106 conditional-entropy target table for per-tensor training weights.
2. A real-checkpoint sanity loop showing `tools/run_deltaepszeta_training.py`
   consumes an A1 EMA state dict and writes durable logs/checkpoints under
   `experiments/results/`.

## Operator-Facing Bug Fixed

Two operator-facing bug classes were fixed.

### Target-builder shell expansion

`tools/build_deltaepszeta_training_targets.py --shannon-json` advertised a path
or glob, but an unquoted zsh glob can expand into multiple operands before
Python sees it. The previous argparse surface accepted one operand only, so a
normal operator command failed with:

```text
error: unrecognized arguments: experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/per_tensor_shannon.json
```

Fix:

- `--shannon-json` now accepts one or more path/glob operands.
- `_resolve_shannon_json(...)` canonicalizes quoted glob, exact path, and
  shell-expanded operands through the same newest-file selector.
- The stale usage example was corrected from `--output` to `--output-dir`.

Focused test added:

```text
src/tac/tests/test_build_deltaepszeta_training_targets.py::test_resolve_shannon_json_accepts_shell_expanded_operands
```

### Preflight direct-entrypoint environment

Running `tools/all_lanes_preflight.py` directly through the shebang used the
ambient macOS `python3`, while many child preflight tools need the repo venv
dependencies (`torch`, `numpy`, `brotli`) and stable `src` imports. The first
fix compared resolved executable paths and failed because venv symlinks resolve
to the base interpreter. The final fix:

- tests `sys.prefix` against `repo/.venv`;
- re-execs through the `.venv/bin/python` symlink itself, not its resolved base
  interpreter;
- prepends `repo/src` and `repo` to `PYTHONPATH` for child tools.

Focused tests added:

```text
src/tac/tests/test_all_lanes_pr91_gate.py::test_all_lanes_reexec_uses_sys_prefix_not_resolved_executable
src/tac/tests/test_all_lanes_pr91_gate.py::test_all_lanes_reexec_skips_inside_repo_venv
```

## Target Build

Command:

```bash
.venv/bin/python tools/build_deltaepszeta_training_targets.py \
  --shannon-json experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json \
  --output-dir experiments/results/lane_deltaepszeta_targets_pr106_20260509T_seq_codex \
  --started-at-utc 2026-05-09T00:00:00Z
```

Resolved source:

```text
experiments/results/lane_per_tensor_shannon_pr106_20260507T173436Z/per_tensor_shannon.json
```

Outputs:

```text
experiments/results/lane_deltaepszeta_targets_pr106_20260509T_seq_codex/targets.json
  bytes: 13,483
  sha256: bf9b68cd7cb3c0067128c458cd43adffcbf8492b98c657223fb701fc94c116a4

experiments/results/lane_deltaepszeta_targets_pr106_20260509T_seq_codex/targets.md
  bytes: 2,832
  sha256: bd1d0eb84b9de7992d9a68e28a016e74e32c21930e8f7533bc1f6847297b5c71
```

Key target facts:

```text
total_h0_h2_gap_bytes: 78,580
total_brotli_bytes_today: 170,096
H2/H0 aggregate ratio: 0.5310
```

Top five weighted tensors:

| rank | idx | tensor | conditional-entropy prize | normalized loss weight |
|---:|---:|---|---:|---:|
| 1 | 6 | `blocks.2.weight` | 15,942 B | 0.2029 |
| 2 | 4 | `blocks.1.weight` | 14,320 B | 0.1822 |
| 3 | 2 | `blocks.0.weight` | 13,273 B | 0.1689 |
| 4 | 8 | `blocks.3.weight` | 9,536 B | 0.1214 |
| 5 | 10 | `blocks.4.weight` | 8,646 B | 0.1100 |

Interpretation: this is a training-weighting and coder-design signal. It is
not a proof that the full 78,580 B gap is reachable in a contest packet. Exact
score movement still requires a runtime-consumed archive and paired CPU/CUDA
eval custody.

## A1 EMA State Dict Sanity Loop

Input checkpoint:

```text
experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/train/checkpoint_ema.pt
bytes: 925,026
sha256: 81e46e4f86ec9a492374903b99311c50f261822cdd0e46fd465e0ae19b9a7051
schema: 28 tensor OrderedDict, float32, HNeRV-family state dict
```

Command:

```bash
.venv/bin/python tools/run_deltaepszeta_training.py \
  --state-dict experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/train/checkpoint_ema.pt \
  --targets-json experiments/results/lane_deltaepszeta_targets_pr106_20260509T_seq_codex/targets.json \
  --n-epochs 1 \
  --steps-per-epoch 2 \
  --learning-rate 1e-6 \
  --lambda-init 1e-4 \
  --lambda-step 1e-4 \
  --rate-budget-bits 7.0 \
  --log-dir experiments/results/lane_run_deltaepszeta_training_20260509T_seq_codex/run \
  --run-label a1_ema_pr106_targets_smoke \
  --seed 0
```

Outputs:

```text
experiments/results/lane_run_deltaepszeta_training_20260509T_seq_codex/run/a1_ema_pr106_targets_smoke_step_log.jsonl
  sha256: 9dbe0e27c2b18d47540e2d8a43af98ba0d28d9e77478dd8ce4d4fda74e78c90e

experiments/results/lane_run_deltaepszeta_training_20260509T_seq_codex/run/final_state_dict.pt
  sha256: edccd014e12f67ba38826c87d9cf4c9aee7523345237e0f0891bd4d74c4ccd2e
```

Step rows:

```json
{"distortion":0.0,"epoch":0,"lambda_value":0.0001,"loss":0.0006667497218586504,"rate_bits":6.667497634887695,"step":0,"timestamp_utc":"2026-05-09T02:51:27Z"}
{"distortion":1.3788278841642484e-30,"epoch":0,"lambda_value":0.0001,"loss":0.0006667497218586504,"rate_bits":6.667497634887695,"step":1,"timestamp_utc":"2026-05-09T02:51:27Z"}
```

Interpretation: the Phase 1 driver can consume the real A1 EMA checkpoint and
PR106 entropy targets without scorer loads, remote dispatch, or `/tmp` evidence
paths. This is a scaffold validation only. The near-zero distortion is expected
for a two-step MSE-vs-reference sanity loop at `lr=1e-6`; it does not imply a
score-lowering candidate.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_build_deltaepszeta_training_targets.py -q
12 passed

.venv/bin/python -m pytest src/tac/tests/test_run_deltaepszeta_training.py src/tac/tests/test_build_deltaepszeta_training_targets.py -q
24 passed

.venv/bin/python -m pytest src/tac/tests/test_all_lanes_pr91_gate.py src/tac/tests/test_build_deltaepszeta_training_targets.py src/tac/tests/test_run_deltaepszeta_training.py -q
36 passed

tools/all_lanes_preflight.py --jobs 4 --timings
ALL 29 PREFLIGHT CHECKS PASSED

git diff --check
clean
```

## Next Actions

1. Replace the sanity-loop MSE objective with a score-domain or boundary-aware
   surrogate before spending GPU on delta-epsilon-zeta.
2. Add a packet builder or runtime-consumed codec path before exact eval
   dispatch; target tables and checkpoints alone are not shippable archives.
3. Use paired CPU/CUDA eval for any produced archive. Do not apply the HNeRV
   CPU/CUDA ratio blindly to a new co-designed substrate.
4. Retain the PR106 conditional-entropy table as a solver prior for
   meta-Lagrangian and Pareto planners, not as a promoted byte claim.
