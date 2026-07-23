# Codex findings — DDM v19 pure-priced realized objective — 2026-07-23

`research_only=true`

`execution_allowed=false`

`score_claim=false`

`evidence_axis=[macOS-CPU frozen-scorer advisory]`

`pointer_moved=false`

`main_landing_review_required=true`

## Deliverable 1 — the 405-flip row

`FIRST_NET_IMPROVING_REALIZED_CORRECTION_ADMITTED_N600_ADVISORY`

The v17 candidate
`1x1_rowband_control_solve_02_M_preconditioned_ranked_prefix_r4` caused 405
harmful and 484 helpful Seg argmax flips. The old zero-collateral and epsilon-64
controls rejected it. The required cap-free law admits it:

| term | exact measured delta |
|---|---:|
| `Delta d_seg` | `-0.00005022684700000182` |
| `Delta d_pose` | `+0.000978855354986763` |
| `Delta archive_bytes` | `+201` |
| `100 Delta d_seg` | `-0.005022684700000182` |
| nonlinear Pose term | `+0.00012130112698827133` |
| rate term | `+0.00013383764957755644` |
| joint `Delta S` | `-0.004767545923434355` |

The proposal is not a tiny local perturbation: 308,410 camera RGB pixels and
687,223 channel values changed across all eight pairs. The 405 collateral flips
are already paid in the measured `d_seg`; applying a second collateral veto
double-counted harm.

## Proposal-source screen

| source / channel | measured | admitted | rate | best joint `Delta S` | byte/flip evidence | disposition |
|---|---:|---:|---:|---:|---:|---|
| v17 rejected-class, compact int8 | 4 | 4 | 100% | `-0.004767545923` | `201/(405+484)=0.226` for first row | use; all four quanta admitted |
| camera-Q8 pre-final-uint8 | 3 | 3 | 100% | `-0.003395210621` | `5.310`, `3.507`, `2.544` | valid stage, loses byte race to int8 |
| grammar-native exact forward | 9 | 3 | 33.3% | `-0.009188162447` | exact archive deltas `+2,-5,+2` for winners | worldsheet x−1/x+1/y−1 admitted |
| #579 tie-tight PT | 0 terminals | 0 | N/A | N/A | N/A | degenerate energy spread; family open |

The model-preconditioned and model-disabled v17 rankings each admitted 4/12
under the same exact-call budget. Therefore `M_NOT_USED_FOR_V19_RANKING`.
Large coherent extent is not sufficient by itself: worldsheet and compact-int8
moves admitted, while large template/lane-program steps could still regress.
Exact realized admission remains the authority.

The grammar winner was `worldsheet_joint_active_x_+1`:
`Delta d_seg=-0.0000896453860000021`,
`Delta d_pose=-0.0017776882600060162`, `Delta bytes=-5`,
`Delta S=-0.009188162447058265`. Joint same-frame moves were evaluated together
to preserve SE/global-gate interactions.

## Stage and scale dispositions

The 405 direction was evaluated at both required stages:

- compact post-quantization int8: `Delta S=-0.004767545923`, `Delta bytes=201`;
- camera Q8 pre-final-uint8 at full scale: the same Seg/Pose endpoint,
  `Delta S=-0.003395210621`, `Delta bytes=2,262`.

The compact representation wins this instance’s byte race. The render-grid
channel is `NOT_APPLICABLE_TO_405_CLASS`, because this proposal is a
camera-mask/template correction and projecting it to 384 would change its
class; the render-grid family is open.

The selected sequence contains exactly this one move, so its development
cumulative delta is the 405 row above. The n64 and n600 rows below are the
independently remeasured cumulative endpoint at each scale. The other nine
admitted rows are alternative single-step trials from their own baselines;
summing them would violate the measured SE/non-additivity law.

The mandatory scale gate also held:

| rung | `Delta d_seg` | `Delta d_pose` | `Delta bytes` | strict joint `Delta S` |
|---|---:|---:|---:|---:|
| n64 | `-0.0006124178570000027` | `+0.00037441184900899316` | `+1152` | `-0.06042778770627022` |
| n600 | `-0.00035960727300000245` | `+0.00028292250200934177` | `+1588` | `-0.034868311533520074` |

The exact n600 archive is 135,529 bytes,
SHA-256 `bec946cedff1bdf78525e008ad4f0cbfe999b9fe35fabc1b99bf312897832d59`;
all 38 scorer batches were checkpointed and their digest chain is
`8bc418914b9186dc9384995d653046504c84959aef39583ae77a5defb0747a53`.
These remain advisory measurements, not a contest score.

## Pair recursion and inverse-solve custody

The outer solve/diff/decompose ledger has one exact row for each of the eight
screening pairs. It records Seg/Pose deltas, helpful/harmful flips,
margin-weighted changed cells, per-class strata, recurrence class, and global
archive bytes.

No pair terminal was asserted. `terminal_state=null` with blocker
`BLOCKED_G3_PAIR_DEBT_ALLOCATION_AND_C1_SHARED_BYTE_AMORTIZATION_NOT_BOUND`
is the honest state until MAIN binds the per-pair waterfill debt and allocation
of the shared 1,588-byte archive delta.

Inverse-solve labels are deliberately narrow:

- compact int8 405: known exact receiver-preimage replay;
- camera-Q8 405: exact stage-specific preimage replay;
- grammar and #579: exact forward receiver/coder evaluation, but fallback only
  because no exact nonlinear SegNet/PoseNet target-preimage certificate is bound;
- reflected per-block trunk inversion: owed to MAIN, not claimed.

## Artifacts and review request

- final receipt:
  `.omx/research/ddm_v19_pure_priced_objective_20260723T041500Z/ddm_v19_pure_priced_objective_receipt.json`
  (SHA-256 `ec6d49b5ba89c352d1c76bbf8e4e1783374a36d61db6aa2262099a59d52294db`);
- pair ledger:
  `.omx/research/ddm_v19_pure_priced_objective_20260723T041500Z/stage_checkpoints/06_pair_recursion_ledger.json`
  (SHA-256 `f8e48b028518d2f04ee8f3798da2a6d3be28f6f0f8cc13688adb8d189011a845`);
- typed config and DAG feed;
- strict cap-free equation helper, counted Q8 receiver, measurement tool, and
  regression tests.

MAIN review must verify the SHA-bound input chain, exact archive parse-back,
batch32 frozen-scorer custody, n64→n600 conditional gate, and advisory labels
before landing. It must not convert this receipt into a score or pointer move.
