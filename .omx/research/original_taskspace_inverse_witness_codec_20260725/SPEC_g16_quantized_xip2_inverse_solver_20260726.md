# G16 frozen spec — counted quantized-XIP2 inverse solver

Date: 2026-07-26  
Lane: `lane_g16_quantized_xip2_inverse_solver_20260726`  
Phase: 2.0 research-only build/local verification  
Authority: operator G16 handoff, subordinate to `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md`

## 0. Frozen scope and ownership

G16 may add only:

- `src/tac/witness_dsl/taskspace_quantized_xip2_inverse_solver.py`
- `src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py`
- this dated spec and G16-local receipts/checkpoints

G16 MUST NOT edit G10, G13, G14, or G15. In particular, these are immutable inputs:

- `src/tac/witness_dsl/taskspace_counted_xip2_chronological_a3.py` (G13)
- `tools/run_taskspace_g8_a3_n2_allocator.py` (G14)
- their tests/specs/receipts

No scorer, archive rebuild, evaluator, n2, n600, remote, paid, or long-running job may be executed in this
lane. Public or historical pose payloads are forbidden. Tests use freshly constructed synthetic xi and
synthetic callback observations only. The module is an encoder-side optimizer and makes no score,
candidate, originality, or promotion claim.

The spec is frozen before implementation. Any material contract change requires a new dated amendment;
implementation convenience is not permission to weaken a fail-closed condition.

## 1. Problem and settled negatives

G13 made a counted XIP2 packet executable through the real uint8 chronological-A receiver on the exact
`PassConditionalASurfaceV1`. It did not choose useful xi. G16 supplies the missing encoder-side inverse
solve: directly search the *quantized counted packet coordinates* that G13 actually receives.

The prior affine xi-to-PoseNet calibration is a measured negative (`R^2 = -0.215`). Therefore G16 MUST NOT
fit, serialize, or consult an affine xi-to-pose map, a Jacobian surrogate, or an assumed pose-space metric.
Every active proposal is serialized, compiled, parsed, CRC-checked, decoded through G13, and only then
presented to the objective callback. The callback may later run the actual pooled frozen PoseNet on the
real decoded chronological frames after root review. Synthetic tests prove only mechanism and closure.

The lawful lessons reused from `tools/pose_frame0_inverse_solve_probe.py` are bounded proposal generation,
global-copy/zero controls, and direct post-receiver scoring. Its module is not imported: it mutates process
state and binds heavyweight scorer machinery. `xi_pose_coder` supplies the quantized grid and exact XIP2
serialization. G9 supplies the closed coder enum. G13 is the sole active receiver. V10 supplies the
store-the-seed discipline: counted low-dimensional xi is the solution statistic; generic search/receiver
code is not video payload.

## 2. Public V1 contract

The implementation SHALL expose, with exact-type validation:

1. `QuantizedXIP2InverseError`.
2. `QuantizedXIP2ObjectiveAuthorityV1`, closed to:
   - `SYNTHETIC_TEST_ONLY`
   - `FROZEN_POSENET_CALLBACK_ADVISORY`
3. `QuantizedXIP2SearchConfigV1` containing:
   - one exact integer deterministic `seed`;
   - nonempty, unique, explicitly ordered `q_levels` in `[1,32767]`;
   - nonempty, unique, explicitly ordered G9 `XIP2Coder` choices;
   - nonempty, unique, explicitly ordered G13 `CountedXIP2A3Interpretation` choices, active domains only;
   - a nonempty positive integer coordinate `step_schedule`;
   - positive exact `sweeps_per_step` and `max_callback_evaluations` search-resource bounds;
   - finite fp32-canonical `pitch`.
4. `QuantizedXIP2EvaluationRequestV1`. It contains the exact compiled G13 object in memory plus arm/q/scales
   metadata and stable hashes. It deliberately has no `as_dict`/JSON serializer.
5. `QuantizedXIP2ObjectiveObservationV1` containing scalars/hashes only:
   - exact per-pair Pose6 MSE tuple, from which the core derives pooled `d_pose` by arithmetic mean;
   - exact finite `d_seg`;
   - Seg prediction SHA-256;
   - exact rebuilt `archive_bytes` and archive SHA-256;
   - authority enum and frozen scorer/target/measurement custody SHA-256 values.
6. A callable protocol accepting exactly one request and returning exactly one observation.
7. immutable arm/candidate/result/receipt records with canonical JSON for nondense state only.
8. `run_quantized_xip2_inverse_search(...)`, with exact keyword inputs:
   - live `PassConditionalASurfaceV1`;
   - fresh numeric finite `initial_xi` of shape `[pair,6]`;
   - config;
   - objective callback;
   - one explicit durable `run_dir`;
   - `resume_from: Path | None` (new run iff absent; resume iff it equals `run_dir`).

The public API MUST NOT accept a serialized historic/public XIP2 payload, dense target frame, dense scorer
output, pose target table, scorer weights, GT labels, or an affine model. The only initializer is fresh
numeric xi supplied under the live source custody. The module computes a canonical lineage digest from
the live G13 source binding, fresh-xi hash, config hash, and implementation identity; it uses that digest
only in G13's explicitly opaque `guidance_source_binding_sha256` field. It MUST NOT upgrade that field to
receiver-verified or source authority.

## 3. Exact arm construction

Controls are compiled and callback-evaluated first through G13:

- `PASS_P0_V1` (zero-A control);
- `COPY_CONDITIONAL_Y1_V1` (global-copy control).

Both controls have canonical empty bodies. Their decoded Y1 hashes MUST equal the exact live conditional-Y1
source hash. Their callback Seg prediction hash and `d_seg` MUST be exactly equal; otherwise the run fails
closed because the claimed Seg invariance is not established. The lower complete-score control is the
reference baseline, with canonical mode-order tie break.

For every Cartesian-product arm `(q_level, coder, interpretation)` in the explicit config order:

1. Call `quantize_xi(initial_xi, q_levels=q_level)` once.
2. Freeze that arm's returned fp32 scales for V1. The negative scope is only fixed-scale direct integer
   coordinate search; it is not a family-level verdict on learned/adaptive scales.
3. Serialize current int16 q with the explicitly enumerated coder.
4. Strictly parse the XIP2 body and require exact q/scales round trip.
5. Build G13 `CountedXIP2ChronologicalA3ProgramV1` in `XIP2_WARP_V1`, with the arm domain and pitch.
6. Compile through `compile_counted_xip2_chronological_a3`, inheriting G13's strict parse/re-encode, nested
   and outer CRC, double decode, real uint8 receiver, and source/Y1 closure.
7. Present only that compiled object to the callback.

Coder choice is never estimated by raw-array size. Each coder produces its actual packet and the callback
prices its actual rebuilt archive. Both warp domains are real G13 domains and never aliases.

## 4. Direct deterministic search (no affine assumption)

For each arm, search the integer q tensor directly. V1 uses coordinate descent:

- coordinates are ordered per `(seed, arm, step, sweep)` by sorting SHA-256 ranks, avoiding host-randomized
  hashes and mutable RNG state;
- from the current q, evaluate each legal `q_i + step` and `q_i - step` inside the arm's closed
  `[-q_level,+q_level]` lattice;
- every neighbor traverses the full serialization -> G13 receiver -> callback chain;
- choose the lower complete exact score among current/plus/minus; equal scores use packet SHA-256 as the
  deterministic tie break;
- proceed through the configured step schedule and sweeps until the callback budget is exhausted or the
  schedule completes.

The callback budget and finite schedule are resource limits, never admission thresholds. A run that finds
no improving proposal reports an honest bounded-search negative with formulation scope.

No affine regression, finite-difference Jacobian, pose-coordinate norm, proxy RGB loss, or unreceived xi
may rank or admit a proposal.

## 5. Objective, invariants, and finite byte ceiling

The core derives:

`d_pose = mean(per_pair_pose6_mse)`

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`

For candidate `c` versus selected control `b`:

`delta_distortion = 100*(d_seg_c-d_seg_b) + sqrt(10*d_pose_c)-sqrt(10*d_pose_b)`

`delta_S = delta_distortion + 25*(archive_bytes_c-archive_bytes_b)/37_545_489`

If `delta_distortion < 0`, the strict maximum admissible *absolute* archive byte count is:

`strict_archive_ceiling = ceil(archive_bytes_b - (37_545_489/25)*delta_distortion) - 1`

This `ceil(x)-1` form is required because equality is not an improvement. If distortion does not improve,
the ceiling is absent and the candidate can improve only if its directly computed complete `delta_S < 0`
through lower bytes. Admission is exactly finite `delta_S < 0`; no arbitrary Pose, Seg, byte, or score
threshold is allowed. Floating inputs must be finite; calculations use Python binary64 and canonical JSON
must reject NaN/Inf.

Every candidate MUST satisfy all of:

- G13 source binding unchanged;
- decoded conditional Y1 SHA-256 exactly equals the source and both controls;
- callback Seg prediction SHA-256 exactly equals both controls;
- callback `d_seg` exactly equals both controls;
- callback per-pair loss length equals the live pair count;
- exact nonnegative integer archive bytes and canonical archive/custody hashes;
- archive bytes are callback-supplied rebuilt bytes, never packet bytes relabeled as archive bytes.

Any failure aborts the run. Packet bytes remain separately reported for rate attribution.

## 6. Crash resume and stage persistence

Every run is durable and deterministic. The small run directory contains:

- write-once canonical `manifest.json`, binding schema, implementation hash, config, seed, live source hash,
  initial-xi hash, pair IDs, and allowed callback authority/custody;
- one distinct canonical `stage-XXXXXXXX-evaluation.json` after every completed callback;
- one distinct terminal stage record.

All files are written atomically with same-directory temporary file, file fsync, rename/publish, and
directory fsync. Existing files are accepted only when byte-identical. Unexpected temporary/partial files,
gaps, overwritten records, noncanonical JSON, chain breaks, foreign config/source/initializer, duplicate
packet observations with differing values, or custody changes fail closed; the module does not delete or
repair evidence.

Each evaluation stage includes the previous stage SHA-256 and stores only:

- arm/control identity, q/scales where applicable, packet/payload/output hashes and byte counts;
- per-pair scalar Pose MSE, pooled derived d_pose, d_seg, Seg hash;
- exact archive bytes/hash, authority and custody hashes;
- complete score/delta/strict ceiling and accept/reject reason when derivable.

Dense decoded frames, dense targets, scorer tensors/weights, target labels, GT, public payloads, and callback
objects are never serialized. On resume, the caller supplies the same live surface, fresh initializer,
config, and callback; manifest/chain custody is revalidated. The deterministic algorithm may replay its
decisions from the beginning, but any packet already present in a valid evaluation stage MUST use the
cached observation and MUST NOT call the callback again. G13 recompilation/receiver replay remains allowed
and required for closure. This guarantees crash resume without scorer reevaluation.

## 7. Truthful receipt

The terminal receipt/result fixes, at minimum:

- `research_only=true`;
- `callback_external=true`;
- `direct_quantized_receiver_search=true`;
- `affine_xi_pose_model_used=false`;
- `public_or_historical_payload_used=false`;
- `dense_target_serialized=false`;
- `scorer_or_weights_serialized=false`;
- `actual_scorer_invoked_by_core=false`;
- `n2_claim=false`, `n600_claim=false`, `exact_score_claim=false`;
- `candidate_claim=false`, `originality_claim=false`, `promotion_eligible=false`;
- `through_g13_uint8_receiver=true` only after every evaluated active packet closed through G13;
- callback authority copied exactly, never upgraded.

`FROZEN_POSENET_CALLBACK_ADVISORY` means only that a separately reviewed callback says it invoked the
frozen scorer under supplied custody. It is not upstream exact-eval authority and cannot promote anything.

## 8. Required synthetic tests

Focused tests MUST cover at least:

1. a deterministic nonlinear callback surface whose local affine slope at the initializer is zero (or
   whose affine ranking is wrong), while direct q-neighbor search finds the known lower complete score;
2. exact enumeration of multiple q levels and all four G9 coders through real serialize/parse/G13 decode;
3. both G13 warp domains with nonaliasing packets/outputs;
4. PASS-P0 and global-copy controls, deterministic best-control selection, and control Seg agreement;
5. exact nonlinear score delta and strict finite ceiling, including integral-boundary `ceil(x)-1` cases;
6. G13 packet CRC/parse/re-encode/receiver closure and corruption rejection;
7. Y1 or Seg drift rejection;
8. resume from atomic stages without callback reevaluation, plus foreign config/source, corrupt chain,
   partial-file, and differing duplicate-observation rejection;
9. deterministic equality for identical seed/config and a deterministic coordinate-order change for a
   changed seed without claiming improvement;
10. checkpoint/receipt inspection proving dense target/scorer data and public payloads are absent;
11. callback pair-count/hash/type/nonfinite/negative-byte failures;
12. bounded no-improvement result with formulation-scoped truth labels.

Tests are mechanism tests only and MUST use synthetic source surfaces and synthetic callback observations.

## 9. Reviewed n2 integration status

A new scorer runner is deliberately **not authorized in G16**. The already frozen G14 runner owns the
reviewed n2 scorer/archive reconstruction path and exposes the additional-A provider seam. Making a second
runner would duplicate custody or, without real archive reconstruction, fake `archive_bytes`. G16 lands a
pure callback/provider API that G14 can consume after root review. The exact blocker to execution is:

`ROOT_REVIEW_REQUIRED_TO_BIND_G16_CALLBACK_PROVIDER_INTO_FROZEN_G14_N2_ARCHIVE_SCORER_PATH`

No reviewed command is advertised until that integration is explicitly authorized and implemented.

## 10. Verification and handoff

Required local-only commands:

```text
python3 -m pytest -q src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
python3 -m pytest -q src/tac/witness_dsl/tests/test_taskspace_counted_xip2_chronological_a3.py src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
python3 -m ruff format --check src/tac/witness_dsl/taskspace_quantized_xip2_inverse_solver.py src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
python3 -m ruff check src/tac/witness_dsl/taskspace_quantized_xip2_inverse_solver.py src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
python3 -m py_compile src/tac/witness_dsl/taskspace_quantized_xip2_inverse_solver.py src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
```

Handoff reports exact file SHA-256 values, focused and combined test counts, format/lint/compile status,
the n2 integration blocker above, pointer unchanged, and no scorer/eval/archive job executed.
