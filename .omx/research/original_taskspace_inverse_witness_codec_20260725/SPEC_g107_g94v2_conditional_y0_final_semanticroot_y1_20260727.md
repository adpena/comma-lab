# G107 — G94-V2 conditional Y0 over frozen final SemanticRootY1V1

Date: 2026-07-27  
Lane: `lane_g107_g94v2_conditional_y0_final_semanticroot_y1_20260727`  
Implementation: `src/tac/witness_dsl/taskspace_g107_g94v2_conditional_y0_final_semanticroot_y1_v1.py`

## Verdict

This landing is a real, strict, source/scorer-free packet and receiver
contract. It is **research-only infrastructure**, not a candidate, score,
compiler fit, or frontier movement.

The canonical frontier pointer is unchanged. No source-backed conditional
operands exist in this receipt; no public `inflate.sh` dispatch or same-archive
`upstream/evaluate.py` CPU/CUDA row exists. The three explicit blockers are:

1. `G107_FRESH_OWN_LINEAGE_CONDITIONAL_Y0_OPERANDS_OWED`
2. `G107_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED`
3. `G107_SAME_ARCHIVE_UPSTREAM_EVALUATE_PY_N600_CPU_CUDA_OWED`

## Object and ownership

The counted product has one shape:

```text
FrozenFinalSemanticRootY1V1
  × SoleConditionalY0Owner(
      global int8 residual basis,
      per-rank fp32 basis scales,
      ordered 600×rank int16 coefficients,
      per-rank fp32 coefficient scales
    )
```

There is no sum/union tag and therefore no second Y0 owner. There is no
historical V15/C1/PR payload arm. The conditional object owns Y0 once; the
embedded `SemanticRootY1V1` owns final Y1 once.

The root is admissible only when:

- `rgb_gauge_ownership == DERIVED_BY_SHARED_GENERATOR`;
- `pair_rgb_gauges == ()`.

Therefore G107 cannot duplicate a raw per-pair RGB gauge table. G107 also has
no second temporal/latent field. The only full-population conditional state is
the exact ordered coefficient matrix.

The G17 seam calls the existing `bind_semantic_root_to_g17` adapter for every
root section, then requires the exact conditional operand spelling under one
`G17ChronologicalPosePreimageV1` /
`G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE` owner. A wrong type, wrong
owner, or wrong bytes refuses.

## Frozen final-Y1 binding

The packet carries one 32-byte final-Y1 binding:

```text
SHA256(
  "FINAL-SEMANTIC-ROOT-Y1-G94-V2-BINDING\0"
  || semantic_root_packet_sha256
  || canonical_g17_n600_population_binding_sha256
  || exact_ordered_scorer_y1_population_sha256
)
```

Encoding and parse-back both:

1. require the canonical G17 identity map `0..599` in every coordinate;
2. render/hash the actual embedded root's ordered n600 scorer-Y1 population;
3. recompute the final binding;
4. compare it to the externally frozen expected binding.

A header value cannot self-attest: wrong expected binding refuses, and a
different embedded root refuses even when its dimensions remain valid.

## Counted wire

The fixed header closes:

- magic/version/zero flags;
- exact `n=600`, `384×512×3`;
- rank and residual-grid geometry;
- all five exact section lengths;
- the 32-byte final-Y1 binding.

The five sections, in one fixed order, are:

1. exact `SemanticRootY1V1` packet;
2. `rank×grid_h×grid_w×3` int8 conditional basis;
3. `rank` big-endian float32 basis scales;
4. `600×rank` big-endian int16 coefficients;
5. `rank` big-endian float32 coefficient scales.

The packet ends with one CRC32 over the section body and exact EOF. Parse-back
requires canonical re-emission and verifies that parsed arrays exactly spell
their retained section bytes and packet. There are no charged proof hashes,
source hashes, scorer hashes, per-chunk selectors, pair IDs, target hashes, or
whole-camera hashes in the wire.

Known historical whole-payload identities and nested ZIP/V15/PVSA/G95 payload
magics are denied in every section. This is defense in depth; fixed typed
lengths already make those older wire shapes unrepresentable.

## Receiver equation

For pair `p`, rank `r`, and residual-grid coordinate `(u,v,c)`:

```text
w[p,r] = coefficient_q[p,r]
         * coefficient_scale[r]
         * basis_scale[r]

delta_grid[p,u,v,c] = sum_r w[p,r] * basis_q[r,u,v,c]

delta[p] = bilinear_align_corners_false(delta_grid[p], 384, 512)
Y0[p] = uint8(clip(round(float32(Y1[p]) + delta[p]), 0, 255))
```

The receiver imports no source, scorer, target, or teacher. It validates Y1's
exact uint8 scorer ABI, changes Y0, and returns Y1 byte-identically.

All four semantic conditional operands are exposed and receiver-live:

- every basis rank is nonzero;
- every ordered n600 coefficient row is nonzero;
- every rank is used by the coefficient population;
- both scale vectors are positive finite float32;
- every row's decoded residual grid is finite and nonzero.

`audit_conditional_operand_effects` perturbs each of the four typed operands
and requires realized Y0 to change. This is section-level behavior evidence,
not a claim that envelope, integrity, or binding bytes are image operands.

## Ordered n600 and V10

Streaming accepts only contiguous chronological slices, caps batch size at 16,
and traverses exactly `0..599` without gap, overlap, or reordering.

For each pair, both scorer planes are independently realized through the
reviewed `DisjointResizeOperator` factor-2 construction to
`uint8[874,1164,3]`. Both `verify_factor2_uint8_scorer_plane` receipts must be
`certified_exact`; a camera-only residual is not used.

## Pose batch-geometry P0

PoseNet is numerically batch-sensitive. The legacy `gt_n600.npz` `gt_poses`
were produced at batch 1, while upstream evaluates chronological batches of
16. The canonical V9 trainer's `--verdict-batch 32` is also not upstream
authority.

G107 therefore requires external encoder evidence to bind either:

- `FRESH_UPSTREAM_BATCH16_TARGET`, an exact ordered n600 upstream-batch16
  source-pose target; or
- `EXACT_BATCH1_TO_BATCH16_PARITY_RECEIPT`, an exact parity receipt proving
  equivalence for the bound population.

The lineage also binds the final candidate Y0/Y1 chronological-batch16
evaluation contract. It hard-codes:

- source pose batch size = 16;
- candidate pose batch size = 16;
- legacy batch1 `gt_poses` direct reuse = false;
- batch32 verdict/costate rows as authority = false.

The feasibility API independently refuses any pose measurement batch size
other than 16. These fields remain external encoder/evaluator custody and do
not enter candidate bytes.

## Candidate byte accounting

`account_g107_candidate_bytes` reports:

- every wire section's exact bytes;
- the 32-byte frozen-Y1 binding;
- envelope/integrity overhead;
- exact STORE and raw-DEFLATE deterministic ZIP prices;
- exact selected `archive.zip` bytes and SHA-256;
- exact counted-member bytes and SHA-256.

It uses `taskspace_outer_archive_codec` and selects only between strict
parse-backed STORE and DEFLATE alternatives. Its labels are permanently:

```text
research_only = true
candidate_claim = false
score_claim = false
```

## One coupled feasibility set

There is no independent Seg or Pose threshold. Given exact external
measurements and the exact selected outer ZIP bytes `B`, the only model is:

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489
strict_score_slack = target_score - S
strict_sublevel_feasible <=> strict_score_slack > 0
```

The inverse byte ceiling is the largest integer `B_max` satisfying the same
strict inequality:

```text
B_max = max { B in nonnegative integers :
              100*d_seg + sqrt(10*d_pose)
              + 25*B/37_545_489 < target_score }
```

The implementation adjusts the integer boundary against the canonical
`compute_contest_score` operation order, rather than trusting a rounded
algebraic estimate.

Illustrative arithmetic checks at target `0.172` (not measurements, candidates,
or thresholds):

| d_seg | d_pose | strict B_max | S(B_max) | strict slack |
|---:|---:|---:|---:|---:|
| 0.0006 | 0.0003 | 85,945 | 0.17199950347660165 | 4.965233983411643e-7 |
| 0.0008 | 0.0001 | 90,675 | 0.17199953717603667 | 4.628239633208686e-7 |
| 0.0010 | 0.00003 | 82,118 | 0.17199951358817525 | 4.864118247349669e-7 |

For each row `S(B_max+1) >= 0.172`. These rows demonstrate the coupled API;
they do not privilege one distortion operating point.

## External fresh-own-lineage evidence

`G107ConditionalY0SourceLineageV1` is not serialized into the candidate
packet. Its canonical JSON binds:

- exact G107 packet, final-Y1 binding, and conditional operand identities;
- fresh source video and target custody;
- upstream-batch16 source-pose target or exact parity receipt;
- chronological-batch16 candidate pose-evaluation contract;
- compiler source, compile config, and originality declaration;
- `historical_payload_reused = false`;
- `legacy_batch1_gt_poses_reused = false`;
- `batch32_verdict_rows_are_authority = false`;
- `encoder_only = true`.

The manifest is accepted only as exact
`G17EncoderOnlyTeacherOracleEvidenceV1`. This contract cannot manufacture
fresh custody: the receipt below leaves the producer evidence explicitly owed.

## Verification scope

Focused tests cover canonical parse-back, wrong-Y1 refusal, changed-root
refusal, duplicate-gauge refusal, CRC/EOF/foreign payload refusal, dead operand
refusal, sole-Y0/Y1-preservation, deterministic decode, full ordered n600,
V10 factor-2 realization of both planes, four-operand perturbation, exact G17
seam, external fresh lineage, exact ZIP accounting, strict coupled score
arithmetic, and batch1/batch32 pose-custody refusal.

They are deterministic behavior tests with a synthetic root fixture. They are
not empirical Seg/Pose evidence and do not move the pointer.
