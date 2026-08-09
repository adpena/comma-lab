# DDM CL1 execution receipt

Status: **QUEUED-WITH-A-FIRE-ORDER; no CL1 rung measured**. `score_claim=false`.

## What completed

- The raw n600 ANS competitor completed at 600 frames. Its 184-B result at
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/ans_n600/ans_vs_range_n600_result.json`
  has SHA-256
  `8816f91afcc21060753a6612cda4e1b7f3b483a7aa073cbfa1b9b5d7e520d451`
  and reports ideal 114,851.8 B, Range 116,980 B, ANS 114,860 B, and measured
  encode-side delta −2,120 B. The done receipt has SHA-256
  `f099a42cb2990e06b0f4614b17f1ce737ce6e8a094ff02f41d7e5ffb4d97e5af`
  and content `rc=0 elapsed=681 detached-job`. Former PID 89557 returns ESRCH.
- The registered Level-0 lane is
  `lane_ddm_cl1_hpac_capacity_20260809`. No lane maturity or empirical gate is
  claimed complete.
- The owned trainer fail-closes on the exact cache/init/source/config, a hard
  governed launch, disabled MPS CPU fallback, and live MPS. It updates canonical
  EMA after every optimizer step; deploys the shadow; writes atomic, fsynced,
  immutable periodic/stage checkpoints; embeds stable causal state and
  authoritative resume lineage; and preserves live/EMA/optimizer/scheduler/
  best/history plus Python/NumPy/Torch/MPS/shuffle RNG.
- The owned artifact runner invokes the pinned intake packer/Range codec without
  a shell, then atomically attests its exact child argv, complete imported HPAC
  source closure, runtime/environment, and every input/output/report SHA-256 and
  byte count. Same-length output replacement is rejected.
- The fitter admits only the named controls and conditional lambda rungs. It
  requires the fixed terminal epoch-60 QAT checkpoint, an actual interrupted
  safe-run exit `-9`, exact epoch-1 parent state and preserved lineage, a
  fresh-root resumed success, an uninterrupted twin, one full comparison
  identity modulo only `rate_lambda`, exact pack/decode attestations, and real
  emitted Range bytes. The duplicate lambda-1 control is excluded from OLS.

## Measured refusal

On 2026-08-09 the exact full lambda-1 argv was invoked through
`tools/safe_run.py` with `TAC_ADMISSION_ENFORCE=1`, `PYTHONHASHSEED=0`,
`PYTORCH_ENABLE_MPS_FALLBACK=0`, the pinned n600 DALI cache, the pinned P64
initialization, every receiver-closed PR130 flag, a 12-GiB governed admission,
and SSD output paths. The hard admission guard passed, then the trainer exited
1 before model allocation or output creation with:

```text
CL1TrainingError: local Metal is unavailable in this process; CPU substitution is forbidden
```

The safe-run receipt is
`/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/preflight_attempt_v2.safe_run.json`,
SHA-256
`aedc3f6dd03422200de33970c55c2dbc4aa4dc96c7bbcb645f4c2920576d0d59`.
It records `status=ok`, child `exit=1`, the complete argv, and a governed child
PID/PGID. The attempted output root
`/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/preflight_attempt_v2/`
was absent afterward. This is a bounded sandbox capability result, not a claim
that the unsandboxed host lacks Metal.

## Committed-content gate

Before any fire, require every path below to be present in `HEAD` with exactly
the named content hash; a matching dirty working-tree copy is not sufficient.
The landing commit is reported by the serializer handoff, avoiding a
self-referential commit-id field inside this receipt.

| Path | SHA-256 |
|---|---|
| `tools/train_ddm_cl1_hpac_capacity.py` | `0c1e6464173d61c5a585450310977c13822ea662bf0bf9b59548491209f3d423` |
| `tools/fit_ddm_cl1_hpac_capacity.py` | `a66a911a7b27c5cc82eaddc222320c91696f958475e84e4b7fa40e5a9e413d2b` |
| `src/tac/tests/test_train_ddm_cl1_hpac_capacity.py` | `cf4b1a9ab138da141d542a748c294fef90d7cc5c2e0d365a225e5c03e859874a` |
| `src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py` | `ab310ef3ad333ab6efebda2c45d750ed3b156939d57974618c5a4df832e2912a` |
| `.omx/research/ddm_cl1_capacity_20260809/PREREGISTRATION.md` | `3581a47d4869f3de5af2c9db808cea84bd755623adb5b668a17fed653e10a8a7` |
| `.omx/research/ddm_cl1_capacity_20260809/MAIN_METAL_FIRE_ORDER.md` | `266e69d3235eb285d44d0804045c1ca2ddee4562ae52d12596b008c97afd64a7` |
| `.omx/state/lane_registry.json` | `ee9128f081541a982055ba03aeff46cb2fbe757c88b30011e37d96c4e9944b98` |

## Verification evidence

These commands were rerun on the content hashes above:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q src/tac/tests/test_train_ddm_cl1_hpac_capacity.py src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py
30 passed in 2.13s

env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q src/tac/pr130_lift/tests/test_fx3_semantic_qat_resume.py src/tac/pr130_lift/tests/test_fx3_semantic_checkpoint_schema.py src/tac/tests/test_ema_warmup.py src/tac/tests/test_ema_decay_from_total_steps.py src/tac/tests/test_admission_coverage_gate.py
47 passed, 1 skipped in 7.92s

.venv/bin/python -m ruff check tools/train_ddm_cl1_hpac_capacity.py tools/fit_ddm_cl1_hpac_capacity.py src/tac/tests/test_train_ddm_cl1_hpac_capacity.py src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py
All checks passed!

.venv/bin/python -m ruff format --check tools/train_ddm_cl1_hpac_capacity.py tools/fit_ddm_cl1_hpac_capacity.py src/tac/tests/test_train_ddm_cl1_hpac_capacity.py src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py
4 files already formatted

.venv/bin/python -m ty check --exit-zero-on-warning tools/train_ddm_cl1_hpac_capacity.py tools/fit_ddm_cl1_hpac_capacity.py src/tac/tests/test_train_ddm_cl1_hpac_capacity.py src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py
No file diagnostic; one pre-existing pyproject unknown-rule warning for `possibly-unbound`.

.venv/bin/python tools/lane_maturity.py validate
OK — 2263 lane(s) validated cleanly.
```

`py_compile` passed for all four Python files, and scoped `git diff --check`
was clean. Each Python file received two `review_tracker.py mark-file` passes;
the per-file policy checks reported zero violations.

## RECALL EVIDENCE

The bounded original recall searched content across `.omx/research`, `docs`,
all `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, canonical
task-status/arm queues, the canonical-equations registry, design/SPEC files,
and the complete read-only PR130 intake source/pipeline. Queries included
`clean60`, `60-epoch HPAC`, the five real trainer capacity flags, `capacity
ladder`, `model-token`, `HPAC`, `PR130`, and `rate lambda`.

No controlled same-object HPAC capacity ladder or empirical
`d(tokens)/d(model)` was found in that scope. Recall changed the implementation:
`rate_lambda` became the first receiver-closed coordinate; all topology and
wire-mechanism switches were frozen; codec defaults were rejected in favor of
the complete P64/raw argv; theoretical trainer bytes were excluded; the
incomplete intake resume path was replaced with full P0 custody; and actual
pack/Range/decode artifacts became mandatory. Detailed source/query findings
are in `PREREGISTRATION.md` under its own `## RECALL EVIDENCE`.

## Verification boundary

The local suite exercises helper-level full-state/RNG equivalence and the
artifact runner/fitter's fail-closed contracts, including same-length byte
substitution and missing resume-lineage negatives. It does **not** execute
trainer `main` on MPS, literally SIGKILL a real HPAC run, or prove final packed
EMA/Range equality between resumed and uninterrupted trajectories. That
empirical P0 gate is the first action in `MAIN_METAL_FIRE_ORDER.md`.

No CL1 `d(tokens)/d(model)`, fitted uncertainty interval, terminal-QAT knee
bracket, or section win was measured. The PR130 contest row and live
own-vehicle frontier are unchanged.
