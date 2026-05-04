# CMG1 Strict Mask Grammar Prototype - Workers D/G - 2026-05-02

## Scope

Worker D initially owned:

- `experiments/build_charged_mask_grammar_candidate.py`
- `src/tac/tests/test_build_charged_mask_grammar_candidate.py`
- `.omx/research/cmg1_strict_mask_grammar_prototype_20260502_codex.md`

Worker G follow-up owns the bounded runtime integration slice:

- `submissions/robust_current/inflate_renderer.py`
- `submissions/robust_current/inflate.sh`
- `experiments/build_charged_mask_grammar_candidate.py`
- `src/tac/tests/test_build_charged_mask_grammar_candidate.py`
- `.omx/research/cmg1_strict_mask_grammar_prototype_20260502_codex.md`

The current A++ frontier reported by the operator is C-059 with score
`0.3157055307844823` and `276347` archive bytes.  This prototype makes no
score claim.

## Prototype Contract

`CMG1` is introduced here as a strict raw-stream payload scaffold.  The first
runtime integration path is intentionally narrow: inflate accepts only raw
bit-identical mode, validates the charged header/body bounds, reconstructs the
wrapped mask stream from archive bytes, and then delegates to the existing mask
loaders.  It does not load SegNet or PoseNet.

The deterministic output archive contains two single-level stored ZIP members:

- `mask.cmg1`
- `cmg1_manifest.json`

The `mask.cmg1` member charges the fixed wire constants inside its own bytes:

- magic `CMG1`
- schema version `1`
- shape `(frames=600, height=384, width=512)`
- `class_count=5`
- mode code
- byte length of the charged inner JSON header
- optional raw mask stream bytes

If `--input-mask-stream` is supplied, the raw stream is copied byte-for-byte
after the charged header and the manifest records its byte count and SHA-256.
If no input stream is supplied, the builder emits a placeholder strict manifest
with an empty payload body.

## Runtime Decode Integration - Worker G

Implemented:

- `inflate_renderer.py` routes `.cmg1` masks through `_load_masks_from_cmg1`.
- `_decode_cmg1_payload` rejects placeholder mode at runtime.
- Header checks enforce magic `CMG1`, schema version `1`, frame count `600` or
  `1200`, shape `384x512`, `class_count=5`, bounded header JSON, bounded raw
  stream bytes, manifest byte count, manifest SHA-256, and wire-contract shape.
- The decoded raw stream is written to a temporary file derived only from
  charged archive bytes, decoded by the existing `.mkv`/`.amrc`/`.stcb`/`.nrv`
  loaders, and deleted after load.
- Final tensor checks enforce decoded frame count, mask shape, and class-id
  bounds.
- `_resolve_mask_path` now discovers `masks.cmg1` only when the legacy mask
  member is absent, preserving default behavior.

Implemented builder support:

- Default standalone output is unchanged: `mask.cmg1` plus
  `cmg1_manifest.json`.
- New opt-in `--base-archive` mode copies a full archive, replaces
  `masks.mkv` with `masks.cmg1`, and writes `cmg1_manifest.json`.
- Base archive copying rejects zip-slip, hidden/system names, directory
  members, and duplicate members.
- Full archive candidates still set `score_claim=false`,
  `promotion_eligible=false`, and `exact_evaluable_archive=false`.

## Non-Promotable Status

Every emitted manifest and adjacent provenance record sets:

- `score_claim=false`
- `promotion_eligible=false`
- `exact_evaluable_archive=false`
- `evidence_grade=empirical_build_only_non_score`

The scaffold is not a promotable score artifact.  The inflate path now contains
a bounded CMG1 raw-stream decoder and the builder can assemble a full archive
candidate, but hardened exact-eval support is still blocked because
`experiments/contest_auth_eval.py` and
`experiments/canonical_local_auth_eval_smoke.py` do not yet admit `.cmg1`.
Those files were outside Worker G ownership and must be updated together by an
owner before CUDA auth eval can run through the hardened path.

## Path Hygiene

The builder accepts only single-level ZIP member names.  It rejects absolute
paths, traversal, nested paths, backslashes, NUL bytes, macOS resource-fork
sidecars, and hidden/system member names.

ZIP emission uses fixed timestamps, stored compression, fixed permissions, and
fixed member ordering.

## Verification

Completed locally:

- `.venv/bin/python -m py_compile experiments/build_charged_mask_grammar_candidate.py src/tac/tests/test_build_charged_mask_grammar_candidate.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_charged_mask_grammar_candidate.py`

Result: `13 passed in 0.09s`.

Worker G completed after the runtime patch:

- `.venv/bin/python -m py_compile submissions/robust_current/inflate_renderer.py experiments/build_charged_mask_grammar_candidate.py src/tac/tests/test_build_charged_mask_grammar_candidate.py`
- `bash -n submissions/robust_current/inflate.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_build_charged_mask_grammar_candidate.py`
- `git diff -- submissions/robust_current/inflate_renderer.py submissions/robust_current/inflate.sh experiments/build_charged_mask_grammar_candidate.py src/tac/tests/test_build_charged_mask_grammar_candidate.py .omx/research/cmg1_strict_mask_grammar_prototype_20260502_codex.md`

Result: focused pytest `18 passed in 0.46s`; py_compile and `bash -n` passed.

## Remaining Work For Exact-Evaluable Archive

1. Add `.cmg1` to `experiments/contest_auth_eval.py` and
   `experiments/canonical_local_auth_eval_smoke.py` allowlists together.
2. Run the hardened extraction/whitelist smoke on a full CMG1 candidate.
3. Validate decoded mask stream or mask tensor SHA, shape, frame parity, and
   pair ordering from the inflated archive.
4. Run exact CUDA auth eval on the exact full archive bytes and preserve JSON,
   logs, SHA-256, bytes, hardware, manifest, and recomputed score.

## Exact-Eval Allowlist Closure - 2026-05-02T04:29Z

The blocking validator gap is closed.

Implemented:

- `.cmg1` is now admitted by `experiments/contest_auth_eval.py`.
- `.cmg1` is now admitted by `experiments/canonical_local_auth_eval_smoke.py`.
- `src/tac/tests/test_runtime_guards_pass_3.py` covers the canonical CMG1
  member set: `renderer.bin`, `masks.cmg1`, and `optimized_poses.bin`.
- `src/tac/tests/test_canonical_local_e2e_smoke.py` asserts smoke/contest
  whitelist parity includes `.cmg1`.

Verification:

```text
.venv/bin/python -m py_compile experiments/contest_auth_eval.py \
  experiments/canonical_local_auth_eval_smoke.py \
  src/tac/tests/test_runtime_guards_pass_3.py \
  src/tac/tests/test_canonical_local_e2e_smoke.py
.venv/bin/python -m pytest src/tac/tests/test_runtime_guards_pass_3.py \
  src/tac/tests/test_canonical_local_e2e_smoke.py::test_smoke_whitelist_parity_with_contest_auth_eval -q
```

Result: `15 passed in 0.08s`.

Status:

- CMG1 remains non-promotable until a concrete full archive is exact CUDA
  evaluated through `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- The previous blocker is now removed; the next step is a deterministic CMG1
  full-archive exact-eval screen on fast CUDA.
