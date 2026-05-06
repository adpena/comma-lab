# LA-POSE Foveation Archive Consumption Closure - 2026-05-06

Evidence grade: empirical local archive custody, fail-closed
Score claim: false
Dispatch attempted: false
Lane claim: false

## Context

Commit `30ba8e22` lowered LA-POSE/Telescope foveation transport atoms into
deterministic `LFV1` tuple bytes. That proved local payload bytes and SHA-256,
but it was still not archive evidence because no archive member or runtime
consumer was charged.

## Change

Added an LFV1 local archive candidate path scoped to `lapose_foveation`:

- `tac.lapose_foveation_payload_candidate`
- `tac.lapose_foveation_runtime_skeleton`
- `tools/build_lapose_foveation_payload_archive.py`

The builder packages:

- `inflate.sh`
- `lapose_foveation_tuples.lfv1`
- `runtime_consumer.py`
- `runtime_consumer_proof_skeleton.json`

The ZIP is deterministic and local-only. The archive-member manifest records
member order, bytes, roles, and SHA-256s. The readiness audit rereads the ZIP,
compares each charged member against the manifest, decodes the LFV1 structure,
and fails closed on any archive/member mismatch.

## Fail-Closed Boundary

This is not score evidence and is not dispatchable. The runtime skeleton reads
the charged LFV1 member and proof skeleton, verifies member bytes/SHA-256, and
then exits with status `2`. It does not reconstruct scored output.

Readiness remains blocked by:

- `runtime_loader_parity_not_passed`
- `no_op_control_not_passed:lfv1_identity_decode_control`
- `no_op_control_not_passed:lfv1_tuple_mutation_runtime_output_control`
- `no_op_control_not_passed:runtime_consumes_foveation_tuple_control`
- `exact_cuda_auth_eval_missing`

## Verification

Focused local checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_build_lapose_foveation_payload_archive.py src/tac/tests/test_lapose_foveation_payload.py src/tac/tests/test_foveation_readiness.py -q
.venv/bin/python -m ruff check src/tac/lapose_foveation_payload_candidate.py src/tac/lapose_foveation_runtime_skeleton.py tools/build_lapose_foveation_payload_archive.py src/tac/tests/test_build_lapose_foveation_payload_archive.py
```

No GPU, remote dispatch, score claim, or lane claim was performed.

## Runtime-Effect Controls Addendum

Evidence grade: empirical local structural runtime-control evidence
Score claim: false
Dispatch attempted: false
Lane claim: false

Follow-up patch adds deterministic LFV1 runtime-effect controls while keeping
the scored-output boundary fail-closed:

- Identity decode control now decodes the charged `LFV1` tuple payload and
  re-encodes it byte-exactly.
- Tuple mutation control changes one quantized tuple field and proves that the
  runtime skeleton structural output digest changes.
- Readiness now records `runtime_consumption_audit`, explicitly separating
  `structural_runtime_consumption.passed=true` from
  `scored_runtime_output_parity.passed=false`.

This does not reconstruct scorer-visible masks or frames. Readiness remains
blocked by `runtime_loader_parity_not_passed` and `exact_cuda_auth_eval_missing`.
No GPU, remote dispatch, score claim, or lane claim was performed.
