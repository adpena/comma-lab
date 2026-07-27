# G105 exact V9 dual-head SemanticRootY1 adapter

Status: **receiver/ABI implementation complete; fresh producer takeoff P0-blocked**.  
Lane: `lane_g105_exact_v9_dualhead_semantic_root_adapter_20260727`.  
Variant: `tac.semantic_root_y1.v9_hosc_dual_head_odd_y1.v1`.  
Wire magic: `SV9Y1V1\0`.

## Exact architecture boundary

The adapter in
`src/tac/witness_dsl/taskspace_g105_exact_v9_semantic_root_adapter_v1.py`
is a separate typed packet. It does not cast V9 into G103's
`ORIGINAL_COORDINR_FILM_MLP_V1`.

It preserves the repository V9 graph op-for-op:

1. deterministic polar-directional Fourier positional features;
2. `in_proj`;
3. shared FiLM, with optional `film_pl` and `concat_pl`;
4. every hidden activation as `tanh(beta * sin(omega * u))`;
5. `out_sdf` plus `softmax(phi/tau) @ palette`;
6. additive `out_tex`;
7. sigmoid RGB, with the exact optional luma projection.

Only `code[2*p+1]` is serialized. The 600 even Y0 rows are excluded and remain
G94-V2's responsibility. Phase advection is external training evidence, not a
decoder operand or candidate byte.

The public plugin ABI is:

```python
VARIANT_ID: str
parse_packet(payload: bytes) -> ExactV9SemanticRootY1ProgramV1
render_scorer_y1(parsed, pair_id: int) -> uint8[384, 512, 3]
```

G108 reserved this variant, so no edit or intent patch to the committed G103
wire is needed.

## Counted wire

The packet has an exact header/CRC/EOF and ordered `CONF`, `MODL`, and `Y1CD`
sections. Learned weights use typed power-of-two int8, learned biases/palette
and Y1 rows use little-endian power-of-two int16. Evidence and metadata remain
outside candidate bytes.

The deterministic tiny behavior fixture is 4,813 packet bytes:

- header 24; section directory 24; config 517;
- model 632 = 186 tensor data + 446 tensor metadata;
- Y1 3,616 = 3,600 odd-row data + 16 metadata.

Its SHA-256 is
`889f3947c5fc116bb45950be40ef970a3a8b8e06cfe0964a8ed2669ecc5b0568`.
This is an ABI fixture, not a real checkpoint, outer ZIP, candidate, score, or
rate claim. A real fresh V9 model's bytes remain unmeasured.

## P0 fresh-producer refusal

Checkpoint admission requires both of these conditions and fails closed
otherwise:

1. **G46 batch-16 target fiber.** The exact G46 labels have SHA-256
   `6d2ca48a...e65b85`; legacy `gt_n600.npz` labels have
   `f2c8be94...05557` and differ at exactly three cells:
   `(11,286,399) 0->4`, `(18,286,448) 0->4`,
   `(381,206,433) 0->2`. More importantly, V9 consumes target margins in its
   saliency, tie-locus, phase-advection, horizon, birth, and costate paths.
   G46 currently materializes argmax only. Admission therefore requires a
   typed `tac.taskspace_batch16_margin_base_scorer_aggregate.v1` margin receipt
   proving labels and margins came from the same batch-16 forwards, plus exact
   target/source/consumer identities in the checkpoint. The in-tree
   `taskspace_batch16_margin_base_scorer_cache_v1` materializer is the existing
   producer seam; no qualifying aggregate was found or launched in this unit.
   The adapter delegates receipt validation to the canonical
   `load_compile_ready_materialization_receipt` gate. Root adversarial review
   caught and removed an earlier parallel-parser schema typo that would have
   rejected the real `tac.taskspace_fresh_teacher_materialization.v1`
   receipt. The corrected intake reverified the actual full-n600 receipt,
   all 600 pair checkpoints, target-label bytes, portable upstream closure,
   and source-pair chain.
2. **Live verdict fiber.** Checkpoint `__cfg_verdict_batch` must equal 16,
   derived from the portable upstream closure. The historical V9 value 32 is
   refused because it also governed realized argmax/pose, controllers,
   costates, and checkpoint selection.

Consequently there is no fresh-compatible real checkpoint, archive, exact
score, or pointer movement in this unit. At approximately 81,027 total archive
bytes, the coupled competitive envelope requires roughly `d_seg <= 8e-4`
unless G94 reaches `d_pose < 1.45e-4`; an old V9-scale `d_seg ~0.0035` is
structurally noncompetitive.

## Verification

```text
.venv/bin/python -m pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g105_exact_v9_semantic_root_adapter_v1.py
.......... [100%]
10 passed in 0.39s

.venv/bin/ruff check \
  src/tac/witness_dsl/taskspace_g105_exact_v9_semantic_root_adapter_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g105_exact_v9_semantic_root_adapter_v1.py
All checks passed!
```

The tests prove exact equality with the repository NumPy V9 reference on a
fresh deterministic synthetic state, with and without the optional per-layer
routes; strict parse/re-emit; corruption refusal; odd-row-only counting; and
the batch-16 target/verdict blocker. They also prove the canonical compile-ready
receipt loader owns G46 schema validation and that all three persisted names for
the exact polar-directional control normalize through the canonical basis
registry rather than a duplicate alias table.

Pointer delta: **none**. Triality: this spec/DAG edge, the typed adapter DSL,
and the exact V9 forward equations above. No heavy job, paid dispatch, archive
evaluation, commit, or staging occurred.
