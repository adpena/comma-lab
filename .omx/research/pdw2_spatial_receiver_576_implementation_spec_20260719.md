# Task #576 implementation spec — counted PDW2 coefficient-to-spatial receiver

**Authority:** delegated local BUILD and $0 verification only. No paid dispatch, long launch,
score claim, promotion, submission, or pointer mutation. The preserved pointer is
`0.1910828242 [contest-CPU Linux x86_64]`, unchanged.

**Review status:** `recovery-written-UNREVIEWED`; MAIN landing review is required.

## Outcome required

Implement the smallest honest scorer-free PDW2 spatial receiver surface and decide the gate by
measurement. The strict `138 B` #553 margin packet is a global affine-head description. A spatial
partition is `P(x)=argmax_c l_c(z(x))`, so the receiver must account separately for the spatial
rank-4 field `z(x)`. It must never infer that field from the coefficients or relabel the packet as
a self-contained spatial generator.

The optimal-form negative, if confirmed, is the exact blocker
`PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY`: the same canonical packet admits two finite
rank-4 fields whose native-fp32 first-max partitions differ, so no deterministic function of the
packet bytes alone can reproduce both spatial partitions. A through-R consumer additionally owes
a scorer-free RGB/camera pullback whose decoded bytes reproduce the intended cells under the real
frozen trunk. If either input is absent, `d_seg` and `d_pose` stay unmeasured rather than being
filled with proxy values.

## Owned files

- New `src/tac/boundary_math/pdw2_spatial_receiver.py`.
- New focused tests under `src/tac/boundary_math/tests/`.
- New `tools/probe_pdw2_spatial_receiver.py`.
- Minimal integration edits to
  `src/tac/boundary_math/integer_plane_emitter_byte_close.py` only to route the exact blocker ID;
  do not enable `receiver_consumed` mode.
- New canonical equation builder and focused test for the non-identifiability law. Do not append
  the live registry row; the reviewing parent will register only after the real receipt exists.

Do not edit the DAG, findings memo, session summary, task/lane state, receipt, or manifest. The
reviewing parent owns those post-measurement artifacts. Do not commit.

## Required receiver contract

1. Strictly decode and canonical-reencode PDW2/PDP2 bytes. Reject deletion, truncation, trailing
   bytes, non-finite values, wrong dtype, wrong rank, and wrong spatial geometry.
2. Accept an explicit `float32 [N,384,512,rank]` quotient feature field and stream bounded pairs
   through the already-declared `gauge_fixed_scores_f32` / first-max arithmetic. Return immutable
   spatial labels plus content hashes and custody metadata. Do not import SegNet or PoseNet.
3. Provide a deterministic packet-only non-identifiability witness. For the canonical frozen
   packet, construct two finite constant feature fields under the same bytes whose labels differ.
   Record the witness points/classes and verify both via the real receiver arithmetic.
4. Provide positive and negative canaries: packet mutation changes the spatial partition on a
   supplied field; packet deletion/refusal fails closed. A mutation that has no observed effect
   must not count as a canary.
5. The probe must operate on read-only `.npy` memmaps, stream pair-by-pair, record peak RSS, and
   emit canonical JSON. It must support n24 and n600 in one invocation without writing the large
   label arrays. It may emit hashes/counts only.
6. Receipt fields must separate: `packet_to_partition_consumed=true`,
   `coefficient_only_through_r_equivalent=false`, `through_r_authority=false`,
   `d_seg=null`, `d_pose=null`, `score_claim=false`, `promotion_eligible=false`, and the exact
   blocker ID. Do not call quotient-space label agreement a through-R score.

## Exact verification command

Run at minimum:

```bash
PYTHONPATH=src python3 -m pytest -q \
  src/tac/boundary_math/tests/test_pdw2_spatial_receiver.py \
  src/tac/boundary_math/tests/test_integer_plane_banded_glue.py \
  src/tac/boundary_math/tests/test_power_diagram_witness.py \
  src/tac/canonical_equations/tests/test_pdw2_spatial_identifiability_law_20260719.py
```

Also run selected Ruff and `py_compile` on every changed Python file. The parent will run the real
packet against the preserved n600 quotient memmap and will own receipt, round-1 review, DAG, equation
registration, serializer commit, and MAIN handoff.

## Acceptance

- Existing #553 packet arithmetic and M1 byte-close tests remain green.
- The partial receiver proves real coefficient consumption but cannot be confused with RGB/R
  realization.
- The packet-only impossibility is executable, deterministic, and formulation-scoped: it closes
  only `packet bytes -> arbitrary spatial partition` without spatial generator state. It leaves
  the broader `packet + counted spatial generator -> through-R RGB` family open.
- No Fourier carrier, per-flip patch, scorer/runtime payload, new flag, launch, or large local
  artifact is introduced.

## STORES CONSULTED

Loaded: delegated Task #576 authority; `CLAUDE.md`; `AGENTS.md`; operating manual; vehicle OS;
v7.5/v8/v10 specs; #553 packet memo/receipt; M1 #575 findings; `power_diagram_witness.py`;
`integer_plane_emitter.py`; `integer_plane_emitter_byte_close.py`; PDW1 realization source;
FEED-STEP2-CONVERGENCE; durable corpus query; task and lane state; operator inbox through
`2026-07-19T19:48:01Z`. Deliberately not loaded or re-derived: dominated per-flip receipts, paid
provider surfaces, and unrelated vehicle families.
