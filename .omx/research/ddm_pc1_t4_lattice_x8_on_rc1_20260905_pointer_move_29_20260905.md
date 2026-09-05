# TWENTY-NINTH POINTER MOVE — S 0.1445177913121716 @ 175,576 B [contest-CUDA T4 n600]: pc1 lattice ×8 with the full re-solve supersedes ×4: −872 B AND d_pose 5.73e-6 → 5.58e-6 — the same lever, one rung coarser (2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1SPVVJGAJ5WY373TKQY1Z6A`. Lane `ddm_pc1_t4_lattice_x8_on_rc1_20260905`. Modal wall 570.5 s. Archive sha `f7e0bb793645894b2f6885fca82b98cab3067837bd66181e222f3d4b1f43e1ff`, 175,576 B. Runtime tree `6d64e168a1171ba85d253ea06751113242ee6c853db87d415eab08041794b54b`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·175,576/37,545,489 | 0.11690885155337835 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·5.58e-06) | 0.007469939758793239 |
| **S** | **0.1445177913121716** |

| | ddm_pc1_t4_lattice_x4_on_rc1_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1451981569076111 | 0.1445177913121716 | **-0.0006803655954394916** |
| d_seg | 0.00020139 | 0.00020139 | 0.0 |
| d_pose | 5.73e-06 | 5.58e-06 | -1.5000000000000026e-07 |
| bytes | 176,448 | 175,576 | -872 |

## Projection fidelity

Projected 0.1445; realized − projected = 1.7791312171611118e-05. pc1 pre-registered ×8 succeeds V3 on both legs (d_pose 6.3e-6–7.0e-6 predicted band; measured 5.58e-6 — better than the band); byte delta exact

## The mechanism

The pose carrier's 600 × 12 coefficients were re-quantized onto a lattice coarsened ×8 (from the shipped int12 range) and RE-SOLVED for every pair with the
full n600 damped Gauss–Newton (`ddm_jg5.refine_pair`, the fs2 solver verbatim) against the frozen PoseNet on the shipped renders; the twelve basis atoms are
bit-identical to the shipped carrier. This rung SUPERSEDES the ×4 rung (28th move) on the same coefficient block — it is not additive with it. Both legs
improved: −872 B vs the ×4 archive (−2,673 B vs the rc1 base) and d_pose 5.73e-6 → 5.58e-6 (T4). pc1 pre-registered "×8 succeeds V3 on both legs" before the
n600 solve and measured it. Token tail, model sections (rc1's adaptive coding) and basis are byte-identical to the 27th-move archive; only the carrier
coefficient block moved. Seal `SEAL_ddm_pc1_v3x8_lattice_x8_resolved_on_rc1.json` (canonical digest abe2c164…), admit bar derived from the 28th-move pointer;
memo `ddm_pc1_pose_carrier_efficiency_20260905.md`; law `pose_carrier_basis_rate_fidelity_exchange_v1`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.024517791312171605. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.02760893975879324: archive ≤ 138,754.7 B → **-36,821.3 B**.
- **DISTORTION corner** at held bytes 175,576: distortion ≤ 0.0030911 → **8.9× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **4,642.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pc1_t4_lattice_x8_on_rc1_20260905/MODAL_REMOTE_RESULT.json` (sha `054cc127107915c485ad5e481d100f7fa527dea37aef853f82a9605f455db33a`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained/v3x8_on_rc1_candidate_runtime/archive.zip` (sha `f7e0bb793645894b2f6885fca82b98cab3067837bd66181e222f3d4b1f43e1ff`, 175,576 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/SEAL_ddm_pc1_v3x8_lattice_x8_resolved_on_rc1.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x8_29th_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x8_29th_move/SEAL_ddm_pc1_v3x8_lattice_x8_resolved_on_rc1.json` (sha verified: True).

## What this does NOT claim

Pose-changing move: the T4 row is authority for d_pose (5.58e-06 at 3 sig figs). Not evidence that the basis can be touched (V1/V2/V4/V5 all REFUSED with measured
numbers; the carrier's bytes buy a positioned 12-dim subspace). The lattice law's knee is not yet located: ×16 is priced (−3,484 B vs base, break-even d_pose
1.03e-5) and being solved. Three carrier moves tonight are one lever pulled three times, not three levers. `[contest-CPU]` stays RECORD-WITH-REASON. PR #140 is now
six moves behind; a public update is the operator's decision.

## Next from here

pc1's ×16 rung with the full re-solve (if it passes it supersedes ×8 the same way). sj1's multi-pass token pre-distortion continues on the seg-debt pool and must
re-solve its edited pairs' carriers FROM THIS lattice and these coefficients (its `assert_carrier_is_pointer()` guard refuses any other carrier tree). Remaining
named doors on the carrier: the per-atom quantizer step (3,731 B at fixed alphabet), the packed Rice-k field width. Closed today at family scope: temporal context
on the token coder (bd1), basis precision/rank/generated bases (pc1). md3's cell resume (448 steps) and cl3's HPAC rungs wait for the machine to clear.

Equations leg (`tac.canonical_equations`): pose_carrier_basis_rate_fidelity_exchange_v1: second T4 anchor on the lattice axis — ×8 + re-solve beats ×4 + re-solve on BOTH legs (−872 B, d_pose −2.6%); the knee is at or beyond ×8

Own-vehicle frontier: **S 0.1445177913121716 @ 175,576 B [contest-CUDA T4 n600]**, archive sha `f7e0bb79…3e1ff`.
