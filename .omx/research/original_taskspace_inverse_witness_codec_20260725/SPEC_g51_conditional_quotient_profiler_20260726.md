# G51 production specification — conditional selected-preimage quotient profiler

Date: 2026-07-26  
Lane: `lane_g51_conditional_selected_preimage_quotient_profiler_20260726`  
Status: implementation specification; no score, candidate, promotion, or pointer claim

## Objective

Build one resumable full-n600 encoder-side profiler that measures the exact
description debt between:

1. a strictly parsed counted V15 `CarrierComposeReceiverV1` base, rendered at
   camera resolution and reduced by C0B's exact integer resize; and
2. the custody-bound two independent C1/V10 selected scorer planes.

The profiler must answer whether any tested *exact* conditional basis can fit
the canonical pre-existing batch-16 planning headroom. The older batch-32 MS1
coordinate remains a historical comparison only. The independent G54
batch-16 replay is corroboration, not a novelty claim or replacement for the
canonical debt receipt.

## Reused seams

- `src/tac/witness_dsl/c0b_semantic_quotient.py`
  - `PlaneChunk`
  - `exact_resize_round_u8`
  - C0B raw-XOR/LZMA baseline semantics
  - SSD storage preflight and write-once-or-equal stages
- `src/tac/optimization/direct_description_carrier_compose.py`
  - strict `receive_carrier_compose_archive`
  - `CarrierComposeReceiverV1.render_camera_pairs`
- `tools/build_c0b_semantic_quotient_archive.py`
  - strict C1 chunk/custody reader
- `taskspace_fresh_teacher_materializer_v1.py`
  - strict batch-16 compile-ready target-label custody

No second semantic-to-V10 seam, scorer, target table in a candidate, or dense
quotient archive is introduced.

## Exact measured representations

For signed residuals `r0 = Y0_target - Y0_base` and
`r1 = Y1_target - Y1_base`, measure:

1. C0B bytewise XOR, independently for Y0 and Y1.
2. C0B bytewise XOR interleaved by Y0/Y1 at each sample coordinate.
3. Independent signed residuals as canonical little-endian int16.
4. Exact common/differential factorization
   `c = floor((r0+r1)/2)`, `d = r1-r0`, with inverse
   `r0 = c-floor(d/2)`, `r1 = r0+d`.
5. Exact pair-temporal deltas within each immutable chunk: first pair absolute,
   later pairs differenced from the preceding pair; chunk reset is explicit.
6. Evaluator-asymmetric layering: `Y1` is the Seg-primary layer and
   `Y0 XOR Y1` is the Pose enhancement layer. This is exact and compares the
   two-plane direct layout against the scorer dependency graph rather than
   treating the planes as symmetric RGB video.

Every representation is round-trip checked. Exact raw, zlib-9, and the existing
C0B raw-LZMA block sizes are measured. Per-pair zlib marginals, zero-run
structure, symbol entropy, per-channel behavior, and fresh batch-16
class-conditioned residual behavior are preserved.

## Functional-operator proposal surface

The scorer dependency graph defines two exact output groups:

- Seg-primary `Y1` output delta, visible to SegNet and PoseNet;
- conditional `Y0|Y1` enhancement, visible only to PoseNet.

G51 emits their per-pair exact output-space energies and block-coded marginals.
Their ambient Gram is diagonal only because the two planes occupy disjoint
array coordinates; that zero cross-term is explicitly non-authoritative for
the evaluator metric. A task-weighted quotient-atom Gram and any low-rank
merge, prune, or macro-segment eviction ranking are
`BLOCKED_MISSING_SCORER_COSTATE_EFFECTS` until receiver/R/frozen-scorer
JVP/VJP effects exist. This is proposal generation only.

The HOPE PH-1/BatchNorm closed forms are forbidden on the source-incompatible
FiLM-conditioned `tanh(sin)` V9/V15 trunk. Static parameter count is never
rate: block bytes may rank proposals, but admission requires exact same-object
archive ZIP bytes and whole-object scorer verification.

## Resumability and authority

- Stage 00: immutable full input/custody/source/config binding.
- Stage 10: one immutable sufficient-statistics receipt per chunk.
- Stage 20: immutable aggregate receipt rebuilt only from verified chunk
  stages.
- A completed chunk can be adopted on resume only when its run binding and pair
  range match exactly.
- Full runs require exactly 600 pairs and SSD storage. Synthetic tests must set
  `test_only_small_fixture=true`, remain `research_only=true`, and never become
  empirical evidence.
- The batch-16 target-label bank is conditioning/geometry custody only.
- The primary planning coordinate is the already-existing canonical
  `c1_live_target_debt_n600_batch16.json`: d_seg
  `0.00015196058485243054`, d_pose `0.00010184347386600314`, and conditional
  headroom 53,622 bytes to 0.172 / 20,582 bytes to 0.15 from a 133,941-byte V15
  base. It is advisory planning evidence with a same-decoded-raw contest-CPU
  crosscheck, not a new candidate score.
- G54 independently corroborates the same decoded raw at batch 16 with a
  distortion-term difference of about `-1.8054e-9`. It is explicitly secondary
  and cannot be presented as discovering the batch-16 coordinate.
- The older 53,621/20,581-byte batch-32 values remain historical comparison.
- `frontier_feasibility_inference_allowed=false` always: a profiler block-byte
  result is not a receiver-closed archive/eval row.

## Fresh V15 derivation custody — required before score-decision launch

The score-decision profile must consume the fresh current-source V15
derivation, not merely reopen any file having the historical archive SHA.
The typed CLI config therefore binds:

- fresh compile receipt file SHA, schema, and exact fresh `run_id`;
- fresh source-config file SHA and RFC-8785 typed-config SHA;
- adjacent selected archive path, 133,941 bytes, and SHA
  `759e2833...`;
- current producer source paths, bytes, and SHA values, all rehashed live;
- the receiver-closed archive checkpoint;
- all 38 ordered `full_p_camera_identity` checkpoints at exact batch-16
  ranges, each bound to the same typed config, byte-identical base/final camera
  digest, and false score claim; and
- the recomputed 38-checkpoint digest chain.

Equal selected archive content is not evidence of copied provenance. The input
binding records the fresh receipt/config/checkpoint derivation proof
independently from archive byte identity and refuses historical-path fallback.

## Files

- `src/tac/witness_dsl/taskspace_conditional_quotient_profiler_v1.py`
- `src/tac/witness_dsl/tests/test_taskspace_conditional_quotient_profiler_v1.py`
- `tools/profile_taskspace_conditional_quotient_n600.py`
- `tools/tests/test_profile_taskspace_conditional_quotient_n600.py`
- dated G51 findings memo

Do not edit G48, G49, or G50 deliverables.

## Acceptance

```bash
uv run ruff check \
  src/tac/witness_dsl/taskspace_conditional_quotient_profiler_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_conditional_quotient_profiler_v1.py \
  tools/profile_taskspace_conditional_quotient_n600.py \
  tools/tests/test_profile_taskspace_conditional_quotient_n600.py

uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_conditional_quotient_profiler_v1.py \
  tools/tests/test_profile_taskspace_conditional_quotient_n600.py
```

The tests must mutate real array values and prove exact reconstruction,
statistics, resume adoption, stage tamper refusal, batch-geometry
non-authority, fresh compile-receipt/archive/source/config/38-checkpoint
custody, and full-n600/test-fixture gating. They are implementation tests, not
n600 scientific evidence.

## Governed full command

The CLI consumes one typed JSON config:

```bash
.venv/bin/python tools/profile_taskspace_conditional_quotient_n600.py \
  .omx/research/configs/taskspace_conditional_quotient_n600_20260726.json \
  --preflight-only
```

The preflight strictly reopens all hashes/custody, checks the SSD waterfall,
and seals a zero-chunk immutable receipt. It does not render a pair and does
not authorize the full launch. After explicit main-agent/governor review, the
resumable full command is:

```bash
.venv/bin/python tools/profile_taskspace_conditional_quotient_n600.py \
  .omx/research/configs/taskspace_conditional_quotient_n600_20260726.json
```

The main agent must create/review the current config and run through the
governed storage/RSS launcher. G51 does not launch the heavy job.
