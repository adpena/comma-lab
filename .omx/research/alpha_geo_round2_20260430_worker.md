# Alpha-Geo Round 2 Worker - 2026-04-30

Scope: Alpha/NeRV geometry path only. No score claim.

## Evidence Boundary

This turn produced code/test hardening and a CPU smoke test. It is empirical
engineering evidence only. It cannot promote, rank, retire, or score any Alpha,
NeRV, INR, or mask-compression method.

Score truth remains exact CUDA auth eval on exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

No CUDA eval was run. No full Alpha-Geo-1 retraining was run. The only training
execution was a one-step CPU smoke inside a focused pytest on a 2x4x5 synthetic
decoded-baseline tensor.

## Reviewed

- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`
- `.omx/research/alpha_geo1_next_patch_20260430_worker.md`
- `.omx/research/alpha_geo_1_visual_primitives_design_20260430_agent.md`
- `experiments/train_nerv_mask.py`
- `experiments/diagnose_nerv_geometry.py`
- `scripts/remote_lane_nerv.sh`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`
- `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py`

## Inspection Result

Decoded-baseline GT support is present in `experiments/train_nerv_mask.py`:

- `--gt-masks-source decoded-baseline`
- `--decoded-baseline-path`
- `--decoded-baseline-member`
- ZIP duplicate/path safety checks
- baseline/member/source SHA-256 custody
- decoded target mask SHA-256, shape, dtype
- fail-closed shape gate against requested scorer geometry
- class-ID gate against profile `nerv_num_classes`

`scripts/remote_lane_nerv.sh` can pass `GT_MASKS_SOURCE=decoded-baseline`, but
it continues into archive rebuild and CUDA auth eval. Under the current
instruction, do not use that remote script for a smoke run unless it is split or
explicitly configured as a reviewed non-promotable training-only job.

The current safe pretraining gate still needs these controls before exact eval
spend:

1. Decoded-baseline target custody - present.
2. Non-promotable smoke provenance - landed in this turn.
3. Candidate-vs-baseline geometry diagnostics - present via
   `experiments/diagnose_nerv_geometry.py`.
4. Pose regeneration/provenance gate for mask-changing archives - still
   missing before exact eval.
5. Renderer embedding drift - still missing; required before large Alpha-Geo-1
   training per the visual-primitives design packet.

## Patch Landed

Changed `experiments/train_nerv_mask.py` provenance:

- Adds `trainer_artifact_evidence_grade = empirical`.
- Adds `trainer_score_claim_eligible = false`.
- Adds canonical score source requirement:
  `experiments/contest_auth_eval.py --device cuda`.
- Adds `trainer_smoke_run` classification.
- Adds `trainer_non_promotable_reasons` for CPU, non-full geometry, and
  short-step runs.
- Mirrors `target_mask_shape` and `target_mask_dtype` at top level next to
  `target_mask_sha256`.
- Uses `trainer_*` prefixes so later auth-eval adjudication fields are not
  contradicted by trainer-only provenance.

Changed `src/tac/tests/test_lane12_nerv_dependency_closure.py`:

- Adds a decoded-baseline CLI smoke test.
- Runs `experiments/train_nerv_mask.py` with `--device cpu`, `--steps 1`, and a
  tiny `.pt` decoded-baseline target.
- Asserts `masks.nrv` is produced.
- Asserts provenance records decoded-baseline target custody and explicit
  non-promotable trainer status.

The worktree already had pre-existing dirty state, including the lane 12 test
file appearing as delete plus untracked in git status. This turn did not
attempt to reconcile or revert that state.

## Verification

```bash
.venv/bin/python -m py_compile experiments/train_nerv_mask.py src/tac/tests/test_lane12_nerv_dependency_closure.py
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py -q
git diff --check
```

Result:

```text
10 passed in 1.35s
```

`py_compile` and `git diff --check` passed with no output.

## Exact Next Command Plan

Local smoke coverage already exercises the new gate:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_lane12_nerv_dependency_closure.py::test_train_nerv_mask_decoded_baseline_cli_smoke_records_non_promotable_provenance \
  -q
```

Before spending on a real Alpha-Geo-1 pretraining run, use the Lane G v3
decoded baseline archive as the target source:

```text
baseline archive = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
baseline member = masks.mkv
member bytes = 421483
```

Training-only command template for a reviewed non-promotable CUDA smoke
runner. Do not use `scripts/remote_lane_nerv.sh` for this, because that script
continues into auth eval:

```bash
.venv/bin/python experiments/train_nerv_mask.py \
  --profile nerv_mask_lane_g_v3 \
  --device cuda \
  --gt-masks-source decoded-baseline \
  --decoded-baseline-path experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --decoded-baseline-member masks.mkv \
  --output-dir experiments/results/alpha_geo1_decoded_baseline_train_smoke_20260430 \
  --num-frames 1200 \
  --mask-height 384 \
  --mask-width 512 \
  --steps 100 \
  --eval-every 100
```

After a candidate `masks.nrv` exists and is bundled into a deterministic archive,
run geometry diagnostics before any exact eval queueing:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --baseline-member masks.mkv \
  --candidate <alpha_geo_1_candidate_archive.zip> \
  --candidate-member masks.nrv \
  --output-json <evidence_dir>/alpha_geo_1_vs_lane_g_v3_geometry.json \
  --threshold-preset exploratory
```

Exact CUDA eval remains blocked until candidate-vs-baseline geometry is in band,
pose regeneration provenance exists for the mask-changing archive, payload
closure is clean, and the exact-eval spend gate is reviewed.
