# G42 findings — G17 actuator IR and donor contract

Date: 2026-07-26  
Lane: `lane_g42_actuator_ir_donor_contract_20260726`  
Mode: bounded implementation-feasibility review; `research_only=true`; no run, eval, dispatch, archive mutation, pointer mutation, or implementation edit  
HEAD inspected: `0058123af31779d83d1fc10a728389b0ce7823ec`

## Executive verdict

The smallest honest executable path to a real changed-state n600 row is **not** to reuse the existing V1 generative receiver entrypoint. That receiver is production-shaped and already streams full n600 Y1, but its admitted predictor state is `PredictorSemanticStateV1` with explicit V9 Pose6. The selected ep725/LVLS1 predictor is `TaskspacePredictorStateV2` with `NoTransportV2`. A G29/ep725-to-V1 cast, fabricated Pose6, or zero transport would violate the actual state contract and would be a NO-FAKE failure.

The first patch series should instead build a new **pairwise streaming V2 label-local G delegate**:

1. expose the exact ep725 per-pair `(Y0, Y1, phi_argmax)` state from the already-proven runtime computation;
2. form an exact per-pair `TaskspacePredictorStateV2` carrying the current counted P member and `NoTransportV2`;
3. admit only label-local G through `require_g_transport_admission` and `apply_generative_taskspace_correction_v2`;
4. render the changed semantic state with the bounded donor `overlay_g_on_predictor_camera_y1`, called on an n1 slice;
5. preserve Y0 and every G-unowned Y1 sample byte-for-byte, and stream chronological pairs into one output raw;
6. byte-close exact P plus an ordered directory of counted per-pair G pages, then score all 600 pairs locally through the frozen CPU scorer.

This is the shortest route to an honest actuator-economics row. G8, A3, PASS, and R10 can then be added behind the same IR and pairwise executor. Waiting for their optimizers before measuring G alone adds no necessary proof and delays the first score-relevant row.

## What is already production-real

### Full n600 ep725 predictor runtime

The existing ep725 replay path is not a toy:

- `tools/replay_ep725_xcodec_n600_equality.py` loads the frozen runtime, invokes `_setup(src)` once per worker, calls `_render_pair(pi)`, writes chronological raw bytes, and checkpoints chunks.
- The full n600 decode receipt exists at `g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json`.
- Its exact raw SHA-256 is `8565df10...`; source and selected raw were each exactly `3,662,409,600` bytes before certified cleanup.
- The measured two-arm replay wall was `1896.191 s`; the four-worker child maximum RSS was `1,484,832,768` bytes on macOS.
- G28's single selected-arm decode chunks total `959.426 s`; its scorer wall is `524.016 s`. Thus the present local decode-plus-score baseline is about 24.7 minutes, while public decode alone has measured room inside the 30-minute constraint. Overlay overhead still requires measurement before any runtime claim.

The frozen `_render_pair` computation already creates the semantic `phi.argmax` internally but discards it after producing camera frames. This is the exact under-nose donor surface. The correct extension is to expose that already-computed state without changing its mathematics.

### Full n600 V1 generative receiver

`decode_generative_taskspace_fragment_to_y1_raw` is also real and useful as an execution-pattern donor:

- it streams pairwise rather than allocating the whole output object;
- it has durable checkpoint/hash/storage custody;
- it has full n600 output geometry.

It is **not** the correct semantic entrypoint for ep725. Its receipt explicitly reports `ep725_frame1_predictor_compatible=false`, and its input type requires a V1 predictor with explicit V9 Pose6. Reuse the streaming, checkpoint, and evidence pattern; do not reuse or weaken the V1 state contract.

### Existing bounded physics donors

These donor functions already implement actual state changes on real arrays and should remain the physics authorities:

- label-local G admission/application: `require_g_transport_admission`, `apply_generative_taskspace_correction_v2`;
- camera realization: `overlay_g_on_predictor_camera_y1`;
- same-class G8: `decode_same_class_realization_repair_packet`;
- coupled A3: `decode_predictor_preserving_a3_packet`;
- post-G8 conditional A: `decode_post_g8_conditional_a_packet`;
- PASS semantic G: `decode_pass_semantic_g_envelope`;
- PASS conditional A: `decode_pass_conditional_a_packet`;
- R10 operand provenance: `build_r10_selected_solution_adapter`.

Their bounded pair limits are proof-harness constraints, not evidence that the physical action is restricted to n4. The production executor should call each unchanged donor on an n1 per-pair state slice. Do **not** increase the caps on the existing in-memory wrappers and do not materialize 600-pair states merely to satisfy their current signatures.

## Exact first-slice state flow

For pair `p` in chronological order `0..599`:

1. The exact counted P member and generic runtime produce `(Y0_p, Y1_p, phi_p)` from the same calculation used by the proven ep725 replay.
2. Construct the exact V2 state:

   ```text
   TaskspacePredictorStateV2(
       predictor_program=<exact counted P member foreign key>,
       source_pair_ids=(p,),
       labels=phi_p[None, ...],
       transport=NoTransportV2(),
   )
   ```

3. Reopen and verify the counted per-pair G page. Its predictor binding must name the exact P program/state slice; it cannot be obtained by slicing a packet whose source binding names another population.
4. Call `require_g_transport_admission`. Under `NoTransportV2`, admit only the label-local G formulation. Any transport-dependent atom must fail before mutation.
5. Apply the semantic correction through `apply_generative_taskspace_correction_v2`.
6. Call `overlay_g_on_predictor_camera_y1` on this n1 slice. Verify Y0 unchanged and every unowned Y1 cell unchanged.
7. Write `Y0_p` followed by changed `Y1_p` to the output raw. Update input/output aggregate hashes and the per-pair checkpoint atomically.

The present G packet wire has a `uint16`-scale total-event ceiling. A monolithic 600-pair packet is therefore the wrong first container even if it fits by accident. Compile one G page per pair, or a bounded page of at most four pairs only where the existing wire and exact binding make that lawful. The outer selected-solution container stores P once and an ordered page directory. Each page must refuse if its actual event count, section bytes, predictor binding, or ownership support exceeds the contract.

## Ep725 runtime extension

Add one generic, deterministic per-pair state API to the emitted/frozen runtime, conceptually:

```python
def _render_pair_state(pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact camera Y0, camera Y1, and internal phi argmax for one pair."""

def _render_pair(pair_index: int) -> bytes:
    y0, y1, _phi = _render_pair_state(pair_index)
    return y0.tobytes(order="C") + y1.tobytes(order="C")
```

The name is not binding; the compatibility properties are:

- old `_render_pair` output remains byte-identical;
- `phi_argmax` is the same array already computed by the old path, not recomputed by a new proxy;
- setup, weights, quantization, preprocessing, camera conversion, and exact member/runtime identities remain unchanged;
- the generic runtime implementation is free code, while every learned/video-derived G operand remains in the counted archive;
- exact old raw SHA equality is re-established before any actuator row is accepted.

`numpy_oracle_reference_frames` already returns frames plus argmax and is a useful algebra donor, but it materializes prefix lists and is not the n600 execution interface. Refactor its internal calculation into the same per-pair iterator/callback or use it only as a bounded parity oracle.

## G17 actuator IR V1

The selected-solution byte VM should remain the packaging and exact-byte reconstruction layer. Physics should **not** be encoded as loosely named byte opcodes or arbitrary Python import/function strings. G17 actuator IR must use a closed enum and an audited dispatch map to the donor callables above.

### `G17ActuatorKindV1`

Closed values:

```text
SEMANTIC_G_V2_LABEL_LOCAL
SAME_CLASS_G8
COUPLED_A3
POST_G8_CONDITIONAL_A
PASS_SEMANTIC_G
PASS_CONDITIONAL_A
R10_FEATURE_RELAY
```

No unrecognized kind can be ignored or interpreted dynamically.

### `G17ActuatorOperandRefV1`

Required fields:

```text
operand_id
kind
physical_coding_group_id
member_name
member_offset
byte_length
operand_sha256
section_name
pair_start
pair_count
counted = true
```

Execution must reopen the exact member bytes, verify the exact span and SHA-256, reject overlap, reject gaps/unowned bytes in a declared physical coding group, and reject a section or pair range that disagrees with the parsed donor packet. The IR cannot rely on a member name or string attestation alone.

### `G17ActuatorStepV1`

Required fields:

```text
step_id
kind
operand_ids
generic_receiver_operation
input_state_sha256
expected_output_state_contract
pair_start
pair_count
frame_roles
transport_requirement
state_contract_id
predecessor_step_id
```

The realized output state hash belongs in the execution receipt. If the compiler also emits an expected output hash, label it as an executable expectation until the receiver has computed it. `generic_receiver_operation` is a closed dispatch identifier, not a claim that execution occurred.

The ordered state chain must make each step's input the predecessor's realized output. This is especially important for `POST_G8_CONDITIONAL_A`: the A condition must be evaluated on the realized post-G8 state, never on the pre-G8 state or a compiler proxy.

### `G17ActuatorProgramV1`

Required fields:

```text
program_id
schema_version
pair_population
ordered_steps
operand_refs
archive_sha256
archive_size_bytes
member_identities
runtime_source_closure_sha256
input_selected_state_receipt_sha256
input_whole_state_receipt_sha256
chronological_pair_order
standalone_public_output_contract
program_sha256
manifest_sha256
```

`pair_population` must be the exact n600 population, and `chronological_pair_order` must be exactly `0..599`. The program must bind to the strict whole-object state receipt from G41 rather than selecting only the convenient compatible leaf.

### Path-backed `G17ActuatorExecutionReceiptV1`

The existing G17/C0B receipt shapes that retain `decoded_output_bytes` must not be used for a 3.66 GB result. Add a path-backed/stream-backed receipt with:

```text
program_sha256
input_archive/member/runtime identities
input_base_raw_aggregate_sha256
output_raw_path_identity
output_raw_size_bytes
output_raw_sha256
pair_count
frame_count
chronological_pair_order
per_step_and_pair_changed_cell_counts
per_step_and_pair_changed_value_counts
per_step_and_pair_input_output_state_hashes
checkpoint_manifest_root_sha256
deterministic_double_replay_root_or_local_equality_receipt
wall_seconds
peak_rss_bytes
storage_tier_and_free_space_receipt
runtime_source_closure_sha256
exact_archive_size_bytes
exact_archive_sha256
scorer_receipt_foreign_key
authority_axis
research_only
cleanup_certificate_foreign_key
```

The output raw path is evidence only while its exact hash, size, and durable cleanup/rebuild record are jointly present. Public evaluation runs once, so double replay equality is a local deterministic-reproducibility proof, not a fabricated public-eval property.

## Donor API disposition

### Keep unchanged

Keep all bounded physical donors listed above unchanged and call them on n1 slices. Keep `build_r10_selected_solution_adapter` as the exact operand-span foreign-key mapper. Preserve every existing receipt parser and bounded proof test.

### Extend compatibly

1. **Ep725 runtime:** add the per-pair `(Y0,Y1,phi)` API; retain old `_render_pair` as a byte-identical wrapper.
2. **Selected-solution compiler:** add the closed actuator IR types and exact span verification alongside the existing byte reconstruction program.
3. **Executor:** add a path-backed, checkpointed n600 pair iterator. Do not add a full-array V2 adapter.
4. **R10:** factor a shared `_decode_r10_pair` and add `iter_decode_r10_pairs` or `decode_r10_packet_to_raw`; implement current `decode_r10_packet` as a compatibility allocating wrapper over the shared pair decoder.
5. **Receipts:** add path-backed realized-output evidence without removing or silently changing bounded byte-backed receipts.

### Do not extend by cap bump

Do not raise `MAX_BOUNDED_PAIRS=4`, the ep725 ephemeral n2 cap, or the V1 receiver state admission. Their bounded guarantees remain useful. Production scale comes from streaming lawful n1 slices, not from making the proof fixtures resident at n600.

## R10 integration contract

R10 is a later actuator behind the same IR, not a prerequisite for the first G-only row.

The current full-array `decode_r10_packet` validates `realization_sha256(base)` and then allocates a contiguous base plus an output copy. A full camera raw is `3,662,409,600` bytes; base plus output alone is `7,324,819,200` bytes, about 6.82 GiB, before Python, runtime, scorer, packet, and temporary allocations. This can be dangerous on 16 GB hosts and is unnecessary.

The streaming extension must:

- decode and mutate one pair at a time;
- preserve the exact float64 geometry/rounding path unless parity proves a replacement;
- compute the canonical base realization domain in order: `b"R10_REALIZED_PAIR_V1\0" + shape<5I> + base_bytes`;
- verify the complete base aggregate against the expected realization hash;
- write only one output raw;
- on interruption, regenerate and rehash an incomplete pair rather than trusting partial output;
- preserve the existing array decoder as a compatibility oracle.

Because the complete base hash is known only after the stream, the safe executor either completes a base-verification stage before mutation, or records exact immutable per-pair base hashes and refuses finalization until the aggregate base hash matches the bound current-P receipt. It may regenerate the base instead of storing a second raw. It must never claim R10 execution from `receiver_operation` metadata alone.

## Fresh-teacher and originality boundary

The production G compiler requires a fresh teacher under current custody. `bounded_target_g_encoder.compile_bounded_target_g_v2` is a useful compiler donor, but its current `FrozenTargetSliceCustodyV1` names a historical cache. That historical payload must not become the current candidate's target or fitted operand source.

The fresh encoder-only stage must recompute and preserve:

1. source target labels from exact `upstream/videos/0.mkv` through the frozen SegNet/evaluation preprocessing for all 600 scored Y1 frames;
2. current predictor labels and camera frames from the exact selected P member/runtime through the new per-pair state API;
3. exact source video, scorer, preprocessing, archive, member, runtime, and code hashes;
4. the exact obligation IR, atom/action candidates, optimizer trace, and selected counted G operands;
5. immutable per-pair or sharded teacher-evidence hashes on the SSD tier.

Populate `EncoderOnlyTeacherEvidenceV1` with those actual hashes and actual freshly computed target-label bytes. Historical MS1/G21/PBR/SENSE artifacts may inform atom-family ranking or falsifiers only. They must not contribute target frames, label tables, packet bytes, factors, fitted operands, branch selectors, or output payload. Teacher arrays and source labels are encoder-only and cannot ship in the archive or be hidden in inflate code.

## n1/n2 proof versus n600 evidence

The correct split is:

- n1/n2/n4: deterministic API, parsing, span, refusal, state-transition, and donor parity proofs;
- n600: the only decision evidence for changed-state distortion, runtime, memory, bytes, and score.

Required bounded proofs:

### Extend `src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py`

- `test_actuator_ir_reopens_exact_counted_spans_and_orders_state_chain`
- `test_actuator_ir_refuses_v1_pose6_cross_cast_to_ep725_none_transport`
- `test_path_backed_n600_execution_receipt_refuses_bytes_size_order_or_hash_drift`
- `test_r10_adapter_spans_map_without_name_attestation`

### Add `src/tac/witness_dsl/tests/test_taskspace_selected_solution_actuator_executor.py`

- `test_ep725_v2_label_local_g_page_preserves_y0_and_unowned_y1`
- `test_transport_dependent_g_refuses_no_transport_before_mutation`
- `test_pairwise_executor_matches_existing_bounded_monolithic_pg`
- `test_resume_rehashes_committed_pair_ranges_and_refuses_gap`
- `test_full_n600_geometry_is_path_backed_not_resident`

### Extend `src/tac/witness_dsl/tests/test_taskspace_r10_feature_texture_relay.py`

- `test_streaming_pair_decoder_equals_existing_array_decoder`
- `test_streaming_realization_hash_matches_array_domain`

### Add `tools/tests/test_run_taskspace_selected_solution_actuator_n600.py`

Cover storage refusal, stage/pair resume, fresh teacher lineage, historical-payload rejection, cleanup certification, exact population enforcement, and refusal to mutate a pointer. These tests establish mechanics; they do not substitute for the full n600 execution.

## Storage, memory, and checkpoints

Exact geometry:

```text
camera frame:          874 * 1164 * 3 = 3,052,008 bytes
chronological pair:    2 * 3,052,008    = 6,104,016 bytes
full n600 raw:         600 * 6,104,016  = 3,662,409,600 bytes
full Y1-only raw:      600 * 3,052,008  = 1,831,204,800 bytes
semantic labels:       600 * 384 * 512  =   117,964,800 bytes per label bank
```

The first executor should keep one output raw, one pair's runtime arrays, and bounded scorer batches resident. It should not retain a full base raw beside the output. Hash the exact base pair before G mutation, aggregate in chronological order, mutate the in-memory Y1 pair, and write the output.

Before launch, use the storage waterfall `/Volumes/VertigoDataTier/pact`, then `/Volumes/APDataStore/pact`, then local only by explicit opt-in. Reserve at least approximately 11.9 GB plus archive/temp margin: one 3.66 GB raw, both label banks and teacher/checkpoint data, and an 8 GiB safety reserve. The preflight must use actual projected bytes and fail closed.

Mandatory atomic resumable stages:

```text
00 custody + storage preflight
10 fresh source-target shards
20 exact current-P semantic/page source
30 compiled counted G pages
40 exact selected-solution archive
50 full n600 decode, checkpointed by pair/chunk
60 local frozen-CPU scorer, checkpointed across 38 batches at batch size 16
70 cleanup/cold-store certificate
```

Every stage completion is a distinct, atomic, no-replace checkpoint. Long stages also checkpoint per pair/chunk. Resume reopens and rehashes committed ranges and refuses gaps, overlaps, input drift, or output-prefix drift. The raw may be deleted or cold-stored only after its exact hash, size, rebuild argv/config/code/input/archive/runtime identities, score receipt, and cleanup certificate are durable. Preserve the counted archive and receipts.

## First local score row

The first row should be one actual exact P+G archive, not a predicted score or subset proxy:

1. Freshly compile all 600 label-local G pages against the exact current P state.
2. Build the actual outer `archive.zip`; record exact bytes and SHA-256.
3. Inflate through the new public-shaped streaming V2 P/G runtime into exactly `3,662,409,600` chronological bytes.
4. Run `_score_raw_cpu` from `tools/measure_r1b_boundary_generator_n600.py` on all 600 pairs with the frozen CPU scorer and batch size 16.
5. Report measured `d_seg`, measured `d_pose`, exact archive bytes, and

   ```text
   100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489
   ```

6. Compare honestly with the live effective frontier (currently about 0.172; refresh at execution time). Label the local row `[macOS-CPU advisory]`, never contest authority.
7. If the row is economically promising and public-runtime budget is still measured inside 30 minutes, move the exact same archive bytes to contest-CPU/CUDA authority under the governed lane/dispatch process.

The existing G28 ep725 row (`d_seg=0.003512717`, `d_pose=127.359558`, score about 36.09) is only an execution/custody donor, not a quality anchor. The new row must measure the changed output produced by the counted G operands. Pointer movement remains forbidden from this research-only lane.

## Patch sequence and dependency DAG

Recommended smallest executable sequence:

```text
G41 strict whole-state receipt
  -> fresh current-P n600 teacher/semantic stream
  -> G17 closed actuator IR + path-backed receipt
  -> V2 label-local per-pair G page compiler
  -> streaming P/G executor with one raw
  -> exact P+G archive + local n600 scorer row
  -> G8/A3/PASS per-pair adapters
  -> R10 pair iterator + freshly fitted operands
  -> joint descent over admitted actuators
  -> public runtime proof
  -> exact contest CPU/CUDA row
```

The strict receipt, fresh teacher, and IR type work can be prepared in parallel, but the implementation owner must integrate them into one state chain. The immediate implementation priority is the ep725 V2 per-pair semantic surface and label-local G stream, not R10.

## Hard refusal conditions

Fail closed on any of the following:

- V1 Pose6 receiver used for the ep725 `NoTransportV2` state;
- fabricated or zero transport, pose, semantics, or state hashes;
- transport-dependent G admitted under `NoTransportV2`;
- historical target/teacher payload copied into current counted operands;
- a packet sliced or rebound without its exact source-state binding being recompiled;
- name-only operand claims without exact member span and hash reopening;
- arbitrary receiver import/function strings outside the closed dispatch map;
- n600 output bytes retained in a Python receipt object;
- full base plus full output allocation when a streaming path exists;
- cap bump presented as scale proof;
- subset score used as a verdict or frontier claim;
- score claimed without actual outer archive bytes and the exact changed raw;
- resume that trusts an un-rehashed partial range;
- cleanup without a machine-readable reconstruction and custody certificate.

## Bottom line

The missing composition layer is not another abstract opcode or another bounded optimizer. It is the **exact V2 state-preserving seam between the proven ep725 predictor runtime, counted per-pair label-local G operands, the real camera overlay donor, and a path-backed n600 receiver**. That seam can be built without changing the predictor, inventing transport, loading two full raws, or waiting for every later actuator. It yields the first honest answer to the only immediate question that matters: how many exact score units does freshly compiled G buy per counted archive byte on the current selected P object?
