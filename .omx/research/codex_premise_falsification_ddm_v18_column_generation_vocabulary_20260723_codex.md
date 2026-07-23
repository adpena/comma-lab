# Codex premise falsification — DDM v18 generated-column vocabulary

- UTC date: 2026-07-23
- lane_id: `ddm_v18_column_generation_vocabulary`
- research_only: `true`
- execution_allowed: `false`
- evidence_axis: `[macOS-CPU frozen-scorer advisory]`
- score_claim: `false`
- pointer: `0.1910828242 [contest-CPU]` — **UNMOVED**
- verdict: `BLOCKED_PRECONDITION_NO_COMMON_EXACT_R_MASTER`
- verdict_scope: **PRECONDITION only**. No correction, grammar, direct-description,
  coder, or generated-vocabulary family is closed.
- MAIN landing review: **REQUIRED**

## Premise tested

Probe B assumes that the settled v12 control, G1 grammar coordinates, and
v15/v16 solve-generated template degrees of freedom can be columns in one
restricted master whose accepted sets are replayed through the same
camera-resolution paint, uint8, evaluator `R`, frozen scorers, and exact coder.
That premise is false for the bound source artifacts.

This is not a weak-score negative. It is a type/custody failure detected before
pricing, so the three-round formulation falsifier remains open.

## Re-derived evidence

1. `CarrierComposeReceiverV1.render_pairs` is explicitly the legacy V9--V13
   scorer-grid render (`direct_description_carrier_compose.py:2644-2656`).
   The bound v12 archive has no counted realization profile, while
   `render_camera_pairs` refuses that state (`direct_description_carrier_compose.py:2691-2696`).
2. The compiler refuses the intended combined archive:
   `v13 PREDICT productions cannot be mixed with post-solve correction vocabularies`
   (`direct_description_carrier_compose.py:1794-1798`). The probe reproduced
   that exact exception with bound v12 and v15 archive bytes.
3. The G1 receipt is not a receiver candidate:
   `candidate_archive=false`, `receiver_closed=false`, and
   `pose_measured=false`.
4. v15's inherited camera-\(R\) control is `d_seg=0.027470296224`, not the
   requested v12 legacy reference `d_seg=0.034003668891`; those are different
   operating points and cannot be mixed into an equal-byte curve.
5. v16 itself records `linearization_invalid=true` and scopes its fork to the
   measured instance and configured trust radii.

The deterministic receipt is
`.omx/research/ddm_v18_column_generation_vocabulary_20260723T030000Z/ddm_a1_column_generated_correction_receipt.json`
(SHA-256 `f8daae958510e6cea9cf39b499ec820d6747dd968e9a51157e0a5a3e25601a96`).

## What landed despite the blocker

- A typed restricted-master LP using actual HiGHS byte/conflict duals.
- Canonical reduced cost
  \(r_j=c_j-b_j y_b-\sum_k A_{kj}y_k\).
- Dependency/conflict-aware beam search that invokes the exact-replay callback
  for every explored selected set. LP additivity never decides acceptance.
- A strict falsifier that requires exactly three complete pricing rounds and
  four complete global equal-byte replays.
- A preregistered coder race: explicit indices versus 2-of-4 structured
  support metadata at matched realized \(d_{\rm seg}\), plus shared-scale int4
  MX blocks whenever a floating-point payload exists. These entrants are
  **QUEUED_NOT_MEASURED_PRECONDITION_BLOCKED**.
- Metal GPU, custom grouped-backward, and fused-\(R\) preflight. The small
  fused-\(R\) VJP check was bit-identical to NumPy fp32 with maximum absolute
  delta zero. This authorizes future local pricing compute, not score claims.

## Honest rows and pricing history

The v12 values below are reference-only legacy scorer-grid values, not common
exact-\(R\) Probe B controls.

| Added-byte budget | v12 reference d_seg | v12 bytes | generated d_seg | status |
|---:|---:|---:|---:|---|
| 16,384 | 0.034003668891 | 106,106 | — | NOT_MEASURED_PRECONDITION_BLOCKED |
| 49,152 | 0.034003668891 | 106,106 | — | NOT_MEASURED_PRECONDITION_BLOCKED |
| 98,304 | 0.034003668891 | 106,106 | — | NOT_MEASURED_PRECONDITION_BLOCKED |
| 147,456 | 0.034003668891 | 106,106 | — | NOT_MEASURED_PRECONDITION_BLOCKED |

Pricing rounds 1, 2, and 3 are all
`NOT_RUN_PRECONDITION_BLOCKED`; negative-reduced-cost counts are null, not zero.
The preregistered falsifier is therefore ineligible and untriggered.

## Exact reformulation queue for MAIN

1. Land and review a hybrid archive schema that composes V11 addressed
   corrections, G1 natural productions, and v15/v16 template degrees of freedom
   without weakening semantic parse-back or byte ownership.
2. Re-measure the v12 control and every selected set from that same
   camera-resolution realization through uint8, evaluator `R`, and frozen
   scorers.
3. Only then run the n64 screen and three exact pricing rounds. For any claimed
   row, replay n600 and price the real encoded archive bytes.
4. At matched realized \(d_{\rm seg}\), race unstructured indices against
   2-of-4 support metadata; include shared-scale int4 MX blocks for floating
   payloads. Decide by exact bytes, not metadata intuition.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/optimization/tests/test_ddm_column_generation.py \
  src/tac/canonical_equations/tests/test_ddm_v18_column_pricing_law_20260723.py \
  tools/tests/test_probe_ddm_a1_column_generated_correction.py

TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/adpena/Projects/pact/.venv/bin/python \
  tools/probe_ddm_a1_column_generated_correction.py \
  --config .omx/research/configs/ddm_a1_column_generated_n64_20260723.json \
  --output-directory \
  .omx/research/ddm_v18_column_generation_vocabulary_20260723T030000Z
```

Fresh result: `20 passed`; receipt verdict
`BLOCKED_PRECONDITION_NO_COMMON_EXACT_R_MASTER`.
