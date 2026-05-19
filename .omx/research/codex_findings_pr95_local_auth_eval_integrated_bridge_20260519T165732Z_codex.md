# Codex findings - PR95 local auth-eval bridge

**UTC:** 2026-05-19T16:57:32Z  
**Owner:** Codex  
**Task:** `operator_pr95_local_training_auth_eval_bridge_20260519::INTEGRATED_BRIDGE`  
**Axis:** `[macOS-CPU advisory]` for local auth-eval result; not a contest-CPU promotion claim.

## Finding

The PR95 local training harness now has an opt-in integrated archive replay bridge:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
.venv/bin/python tools/run_pr95_local_training_probe.py \
  --device mps \
  --stage-epochs 1=1 \
  --eval-every 1 \
  --run-codec-stage \
  --run-auth-eval \
  --auth-eval-device cpu \
  --require-auth-eval-comparable \
  --output-dir experiments/results/pr95_local_mps_integrated_auth_bridge_smoke_20260519T175800Z
```

The bridge runs:

```text
PR95 training -> codec_stage -> stored single-member archive.zip -> experiments/contest_auth_eval.py
```

It records archive custody, auth-eval JSON custody, evidence-grade boundaries,
and the absolute delta between the training-side `best_score` and canonical
auth-eval `canonical_score`.

## Empirical smoke

Artifact: `experiments/results/pr95_local_mps_integrated_auth_bridge_smoke_20260519T175800Z/manifest.json`

Key values:

```text
training_best_score:       83.40266892376289
auth_eval_canonical_score: 83.40271170242858
absolute_score_delta:      0.00004277866568713762
score_comparable:          true
archive_sha256:            17523537254adf179825294451b8e4a4ac75d0ad6e1c40078b6ba98f4bc160aa
auth_eval_archive_sha256:  17523537254adf179825294451b8e4a4ac75d0ad6e1c40078b6ba98f4bc160aa
auth_eval_json_sha256:     66019cb18f855eecbbf574d487c1df97dde496ecba6ca9058de97e2e41314173
auth_eval_elapsed_seconds: 485.7255178339992
score_axis:                macOS-CPU advisory
```

Interpretation: for the same emitted PR95 weights/archive, the local training
score and canonical auth-eval replay score agree to `4.28e-05` in this smoke.
This makes the local MPS PR95 loop a useful fast iteration oracle for training
velocity and transfer experiments, while still preserving the rule that
promotion/rank/kill requires byte-closed replay on contest axes.

## Bugs found and fixed

Two local replay hazards were found during real smoke runs:

1. The public PR95 `inflate.sh` invokes `python` literally. The Codex shell had
   no `python` on PATH for the auth-eval subprocess, so inflate failed with
   return code 127.
2. Prepending only `Path(sys.executable).resolve().parent` put the uv-managed
   base interpreter ahead of `.venv/bin`, causing `ModuleNotFoundError: torch`.

The bridge now prepends both `sys.prefix/bin` and the resolved interpreter bin
directory to the auth-eval subprocess PATH. This preserves the public PR95
runtime unchanged while making local replay deterministic under the active
virtualenv.

Failure artifacts retained for diagnosis:

```text
experiments/results/pr95_local_mps_integrated_auth_bridge_smoke_20260519T_current/auth_eval_bridge_cpu.failed.json
experiments/results/pr95_local_mps_integrated_auth_bridge_smoke_20260519T174800Z/auth_eval_bridge_cpu.failed.json
```

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_run_pr95_local_training_probe.py -q
9 passed in 0.56s

.venv/bin/python -m ruff check tools/run_pr95_local_training_probe.py src/tac/tests/test_run_pr95_local_training_probe.py
All checks passed!
```

## Next

The bridge should be used on the next longer PR95 local stage smoke before any
expensive remote campaign. A result remains advisory until paired
`[contest-CPU]` and `[contest-CUDA]` eval artifacts exist for the emitted
archive/runtime.
