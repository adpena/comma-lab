# Telescope-style LFV1 foveation payload candidate (2026-05-13)

Status: local byte-closed artifact, fail-closed
Score claim: false
Promotion eligible: false
Ready for exact eval dispatch: false
Dispatch attempted: false

## Purpose

Move the non-HNeRV/domain-specific foveation path from planning prose to
deterministic bytes without claiming score. This candidate is now classified
under the corrected paper taxonomy:

- Motion-prior selector: LA-Pose-like latent-action / hard-pair telemetry.
- Spatial transform family: Telescope-style hyperbolic foveation.
- Candidate role: byte-closed local payload and runtime-consumption skeleton,
  blocked on scored runtime output parity and exact CUDA auth eval.

The prior shorthand "LA-Pose foveation" is intentionally avoided here because
LA-Pose and Telescope are different papers. See
`.omx/research/lapose_telescope_online_distinction_20260513_codex.md`.

## Source evidence

Motion records came from CUDA-derived component response and pair-metric
telemetry:

```text
.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_records.json
```

The selected global response atom was the improving epsilon `-2.0` point from:

```text
experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race/official_component_response_summary.json
```

Source archive SHA-256:

```text
0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
```

The component-response source is routing evidence only. It is not a score
claim and not promotion authority.

## Build commands

```bash
OUT=experiments/results/lapose_foveation_payload_candidate_20260513_codex
rm -rf "$OUT"
mkdir -p "$OUT"

.venv/bin/python tools/build_lapose_foveation_atom_manifest.py \
  --records-json .omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_records.json \
  --base-pose-dist 0.00346442 \
  --source lane_w_component_allocated_lapose_motion_records_20260505_cuda \
  --max-atoms 30 \
  --json-out "$OUT/foveation_atom_manifest.json"

.venv/bin/python tools/build_lapose_foveation_tuple_payload.py \
  --manifest-json "$OUT/foveation_atom_manifest.json" \
  --payload-out "$OUT/lapose_foveation_tuples.lfv1" \
  --max-atoms 30 \
  --json-out "$OUT/lfv1_payload_readiness.json"

.venv/bin/python tools/build_lapose_foveation_payload_archive.py \
  --out-dir "$OUT/archive_candidate" \
  --lfv1-payload "$OUT/lapose_foveation_tuples.lfv1" \
  --source-readiness-json "$OUT/lfv1_payload_readiness.json" \
  --source-archive-sha256 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
```

## Emitted artifacts

All emitted files are under ignored experiment custody:

```text
experiments/results/lapose_foveation_payload_candidate_20260513_codex/foveation_atom_manifest.json
experiments/results/lapose_foveation_payload_candidate_20260513_codex/lapose_foveation_tuples.lfv1
experiments/results/lapose_foveation_payload_candidate_20260513_codex/lfv1_payload_readiness.json
experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/archive.zip
experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/archive_member_manifest.json
experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/candidate.json
experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/readiness.json
experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/summary.json
```

Measured local artifact facts:

```text
LFV1 payload bytes: 402
LFV1 payload SHA-256: be4d51599b715196c78cf5e2824290990e9fb1f56c835dfa4c7ea12365167e41
Archive bytes: 68423
Archive SHA-256: 8c942621b21f46c1fb55380dfc6998e82d6acf527a1e826cb1cb54a03b9e8eaf
```

Charged archive members:

```text
foveation_params.bin: 23696 bytes
inflate.sh: 275 bytes
lapose_foveation_tuples.lfv1: 402 bytes
runtime_consumer.py: 29108 bytes
runtime_consumer_proof_skeleton.json: 14314 bytes
```

## Readiness blockers

The candidate correctly remains fail-closed:

```text
runtime_loader_parity_not_passed
lapose_foveation_scorer_visible_output_parity_not_proven
lapose_foveation_runtime_output_parity_not_proven
exact_cuda_auth_eval_missing
```

## Contest interpretation

This is not a candidate submission and not an eval packet. It is a byte-closed
local proof that the domain-specific foveation lane can be lowered into
deterministic payload bytes plus runtime-consumption scaffolding. The next
score-lowering step is to replace the fail-closed runtime skeleton with a
runtime that changes scored pixels/geometry, run no-op/mutation controls, then
submit a claimed exact-eval packet only after lane-claim registration.
