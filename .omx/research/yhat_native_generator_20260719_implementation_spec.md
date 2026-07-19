# Yhat-native generator implementation spec — 2026-07-19

Status: `IMPLEMENTATION_SPEC_ONLY`

Lane: `lane_yhat_native_generator_20260719`

Authority: delegated arm `yhat_native_generator_20260719`; no launch, paid-dispatch,
score, promotion, or pointer authority. The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`.

This file is the bounded implementation handoff required by the active
arbitrage contract. It does not authorize edits outside the ownership list.

## 1. Owned files

The implementation worker owns only these additive/narrow surfaces:

1. `src/tac/witness_dsl/yhat_native_generator_policy.py` (new)
2. `src/tac/witness_dsl/curriculum_dsl.py` (one additive factory only)
3. `src/tac/witness_dsl/tests/test_yhat_native_generator_policy.py` (new)
4. `tools/measure_yhat_native_equivalence.py` (new)
5. `src/tac/tests/test_measure_yhat_native_equivalence.py` (new, pure helper and
   receipt/resume tests; no frozen scorer invocation)

The worker must not edit canonical v10 specifications, `#539` artifacts,
production receiver code, trainer code, launcher code, activation ledgers,
lane/state ledgers, or the final findings memo. Other agents are working in the
repository; preserve and accommodate their edits, and never revert them.

## 2. Default-OFF typed DSL policy

Create an immutable `YhatNativeGeneratorPolicy` following existing policy
modules such as `shared_resize_joint_coupling_policy.py` and the fail-closed v10
capstone policy. Its compiled contract must be JSON-serializable and must seal:

- schema/name/version and lane id;
- `research_only=true`;
- activation state `BUILT_NOT_ACTIVATED_RECEIVER_ARCHIVE_GATES_OWED`;
- geometry `camera_hw=(874,1164)`, `scorer_hw=(384,512)`, `channels=3`, and
  both-plane PoseNet ownership;
- frozen scorer order: bilinear RGB resize to `(384,512)` before PoseNet's
  `rgb_to_yuv6`, and the shared resized frame-1 RGB plane for SegNet;
- deterministic NumPy-fp32 expander requirement;
- exact uint8 lattice realization and exact rational-numerator verification;
- a compact counted-description boundary and generic free expander boundary;
- `live_trainer_argv=()`, empty overrides, zero epoch delta;
- every authority flag false: trainer activation, live v10 integration, launch,
  paid dispatch, score claim, promotion, and pointer movement;
- source/value provenance for geometry and scorer order, citing
  `upstream/modules.py` and the existing lattice module;
- completed gate: n24 exact-rational-plane/native-float32-ULP receipt;
- owed gates: compact-description receiver closure, n600 decode-time custody
  within 30 minutes, exact archive parse-back, and contest CPU/CUDA replay as
  separate axes.

Any attempt to compile a live/trainer/launch/score/promotion/pointer-authorizing
contract must raise a typed, descriptive exception. Do not invent CLI flags.

In `curriculum_dsl.py`, add exactly one factory:

```python
def YhatNativeGenerator(*, policy: YhatNativeGeneratorPolicy) -> Lever:
    ...
```

The required keyword-only argument is deliberate. The factory must return a
Lever named `YhatNativeGenerator` with empty overrides and `epochs_delta=0`
after rechecking containment. Because it is not nilary, it must not become a
composable launcher lever or emit a fake `FIRED` event. Keep law/provenance maps
empty on the Lever because no flag is emitted; the policy contract owns the
constant provenance.

Tests must prove:

- sealed geometry/order/provenance and both-plane ownership;
- all authority flags remain false;
- escalation attempts fail closed;
- factory output has empty overrides and zero epoch delta;
- compiling a baseline with the inert Lever leaves argv byte-for-byte equal to
  the control compile;
- AST discovery sees it in known/duty tracking but not in nilary/composable
  launcher factories;
- the generic name resolver refuses construction without the required policy.

## 3. Equivalence measurement tool

Implement a deterministic, resumable, stage-checkpointed local CPU measurement
tool. It is a measurement harness, not a trainer and not an archive scorer.

### 3.1 Required inputs and defaults

- Sacred donor checkpoint default:
  `/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z/levelset_witness_ema_BEST.npz`
- Prepared real n24 donor frames default:
  `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/selected_n24_packet/inflated/0.raw`
- Prepared pair IDs default:
  `0,10,50,60,100,110,150,160,200,210,250,260,300,310,350,360,400,410,450,460,500,510,550,560`
- GT cache default:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
- Frozen scorer source default:
  `/Users/adpena/Projects/pact/upstream`
- Durable evidence root default:
  `/Volumes/VertigoDataTier/pact/evidence/yhat_native_20260719`

Require at least 24 real pairs. Bind source file sizes/SHA-256 values, relevant
source-module hashes, seed, thread count, dtype/order, hardware/axis, argv, and
git commit. Read the sacred donor tree and checkpoint, but never write beneath
that run directory. Capture a deterministic metadata/hash snapshot of the
sacred run before and after and fail if it changes.

### 3.2 Scientific comparison

For both donor frames of every pair:

1. Apply the exact separable resize operator from
   `src/tac/optimization/uint8_lattice_feasibility.py` to obtain target rational
   numerators and denominators for `yhat=A(frame)`.
2. Solve an exact uint8 preimage using that module's certified block solver.
3. Reapply the operator and require exact numerator equality for every channel
   and output sample. A non-exact block is a hard failure, never a fallback.
4. Compare the original donor pair and the re-realized pair through the frozen
   full DistortionNet CPU path with the matching GT pair. PoseNet must consume
   both re-realized frames; SegNet must consume the re-realized frame 1.

Report per pair and aggregate:

- exact blocks/samples and any failures;
- direct and yhat-native `d_seg`, `d_pose`, and deltas;
- direct-versus-yhat-native SegNet argmax disagreement count;
- scorer-input, PoseNet scored-output, and SegNet-logit bit equality, maximum
  absolute delta, and maximum native-f32 ULP distance;
- a narrow equivalence classification. Use `BIT_IDENTICAL` only when the
  compared frozen-oracle tensors and authoritative metrics are bit-identical.
  Otherwise use `EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS` only when exact
  rational plane equality is proven and every observed difference is fully
  described by recorded native-f32 deltas/ULPs. Do not broaden this verdict to
  other receivers, hardware, batches, or archive bytes.

The receipt must include an explicit `verdict_scope` and must not claim a score,
promotion, launch readiness, or pointer movement.

### 3.3 Resumability and durability

- Use atomic JSON writes (`tmp` in the same directory plus `os.replace`).
- Preserve a distinct stage receipt after input custody, after each completed
  pair, and after aggregate verification. Never overwrite completed per-pair
  stages.
- A state file may advance atomically but cannot replace stage receipts.
- `--resume` must verify the scientific/config/input binding before reusing
  stages; otherwise fail closed.
- A resumed run must reconstitute the same aggregate scientific payload as an
  uninterrupted run.
- Do not persist reconstructed raw frames or scorer tensors. Keep them bounded
  to one pair and release them after each stage.
- Print progress after every pair so a supervising process can remain visible.

It is acceptable to import the exact existing frozen-oracle loading helpers
from `tools/measure_yhat_rd_ladder.py`, but the receipt must bind that helper's
source hash and import failures must be descriptive.

### 3.4 Tests

Pure tests must cover:

- float32 ULP distance for equal, adjacent positive/negative, signed-zero, and
  nonfinite inputs;
- exact receipt classification rules;
- atomic stage preservation and strict resume-binding mismatch refusal;
- deterministic pair-ID parsing and minimum-pair refusal;
- sacred snapshot comparison logic on a temporary fixture;
- no scorer/GPU/network dependency in the test suite.

Do not run the real n24 measurement in the implementation worker. The parent
agent owns measurement custody and review.

## 4. Acceptance command

At minimum, the worker must run:

```bash
python3 -m pytest -q \
  src/tac/witness_dsl/tests/test_yhat_native_generator_policy.py \
  src/tac/tests/test_measure_yhat_native_equivalence.py
```

Return a concise summary of files changed, tests run, and any limitation. Do not
commit; the parent owns adversarial review, clean-pass counting, serializer
landing, and MAIN handoff.

## 5. Review pass 1 corrections — required before acceptance

The first implementation return is not accepted. Apply all of the following in
the same five owned surfaces, then rerun the focused suite. Do not touch
`uv.lock` or create/update a project lock file.

1. **Resume binding must actually resume.** The current binding includes raw
   `args.argv`, so adding `--resume` changes the hash and guarantees refusal.
   Store a normalized scientific invocation/config whose fields are explicit
   and exclude only the transport bit `resume`. Add a pure helper and test that
   identical scientific bindings pass while a changed pair/thread/node-budget/
   input hash fails. Preserve/re-derive completed stages; do not silently trust
   them.
2. **Use only PoseNet's scored outputs.** Upstream PoseNet returns a mapping.
   Concatenate, in declared hydra-head order, each scored head's first
   `head.out // 2` values. Do not label all 12 `pose` outputs as scored. Continue
   to compare both actual scorer inputs (`posenet_yuv6`, `segnet_rgb`) and full
   SegNet logits.
3. **Do not hardcode the ULP-class predicate.** Compute
   `native_f32_deltas_described` from exact rational proofs plus finite recorded
   metrics/comparison summaries and absence of the nonfinite sentinel. Record
   the fixed preimage policy and
   `f32_receiver_arithmetic_exactness_admissibility_v1`. Rational equality alone
   never authorizes the class.
4. **Bind and pass the solver budget.** Add
   `--max-nodes-per-block` (default 4096), require it positive, place it in the
   scientific binding, and pass it to every `solve_uint8` call.
5. **Validate every input geometry.** Require the donor raw byte count to equal
   `2 * len(pair_ids) * 874 * 1164 * 3`; require `gt_f0`/`gt_f1` to be
   `uint8[600,874,1164,3]`; require the donor checkpoint's stored epoch/render
   metadata to be ep725 and `(384,512)` while leaving the sacred run read-only.
6. **Make stages timed but scientifically reproducible.** Record pair runtime
   and per-frame solve runtime. On resume, compare the deterministic scientific
   payload to the preserved stage while retaining the first environmental
   timing instead of requiring timing equality. Include total runtime in the
   final receipt.
7. **Produce the owed aggregate.** Aggregate exact block/sample counts and
   failures, classification counts, total SegNet argmax disagreements, direct
   and yhat-native mean `d_seg`/`d_pose`, their deltas, and per-surface maxima
   for absolute and native-f32 ULP differences. Include `torch` version,
   platform/python/numpy, seed, and scorer hashes. The aggregate classification
   is bit-identical only if every pair is bit-identical; otherwise it is the
   narrow exact-plane/native-f32 class only if every non-bit pair satisfies the
   computed predicate.
8. **Seal every policy field.** `validate()` must reject changed scorer order,
   deterministic-expander/realization/boundary strings, value provenance,
   owed-gate set, and authority booleans—not only geometry. Tests must mutate at
   least one non-authority semantic field and prove refusal.
9. **Test the actual activation duty surface.** Assert
   `YhatNativeGenerator` appears in `known_levers()` and in `duty_to_measure()`
   for an empty temporary ledger while remaining absent from
   `name_composable_levers()`.
10. Format the touched Python files consistently and add focused tests for the
    normalized resume binding, scored-Pose mapping extraction, finite ULP-class
    predicate, exact raw-size refusal, and aggregate math. The real n24 run
    remains parent-owned.

## 6. Completion appendix — parent-owned measurement and review

Status: `IMPLEMENTED_AND_N24_MEASURED_PENDING_MAIN_REVIEW`.

The five bounded implementation surfaces were completed and the parent-owned real measurement ran
with:

```bash
/Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_yhat_native_equivalence.py \
  --evidence-root /Volumes/VertigoDataTier/pact/evidence/yhat_native_20260719/revision2 \
  --cpu-threads 1 \
  --max-nodes-per-block 4096
```

**MEASURED:** all 24 supplied pairs classified
`EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS`; `28,311,552 / 28,311,552` rational samples replayed
exactly, with zero failures. The frozen oracle was not bit-identical: seven SegNet argmax pixels
disagreed. Mean yhat-minus-direct deltas were `Delta d_seg=-2.1175947040319443e-7` and
`Delta d_pose=-0.000240008036286099` on the non-promotable local CPU axis.

**MEASURED/DERIVED runtime blocker:** mean exact solve time was `12.4537300009 s` per plane;
the n600 arbitrary-rational two-plane projection is `249.0746000188 min`. This is a derived
projection on the supplied feasible donor-derived fractional planes, not a measured full-n600
decode. The separate fast integer-yhat receipt realized one frame-1 plane and copied it; it cannot be
transferred to two independently described Pose planes or fractional yhat without new scorer-debt
and runtime measurement.

Aggregate receipt:
`/Volumes/VertigoDataTier/pact/evidence/yhat_native_20260719/revision2/receipt.json`, `62,332` bytes,
SHA-256 `1ad1cf84672c696b46f62ca8586bb29d5c70f55de5803902b6c37666e5b85c0f`.
Sacred-run snapshot before/after:
`0bdc3e39c5eac970625f91e6803b1bb33330e412514b78ce35dab2c4c351842c`.
Cleanup status: `NO_BULK_CREATED_NO_DELETION_REQUIRED`.
The exact command with `--resume` reused all 24 pair stages and preserved the receipt, state, and
aggregate-stage hashes byte-for-byte.

The default-OFF policy records the n24 receipt as closed while receiver/archive gates remain owed;
it grants no live authority. The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`. MAIN review of this branch and the separate PDW2 commit
`edf47756ba629e079a2a63233bf8f0293cf85f3d` is required before integration.
