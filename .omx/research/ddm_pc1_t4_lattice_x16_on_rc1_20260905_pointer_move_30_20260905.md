# THIRTIETH POINTER MOVE — S 0.14411787458634504 @ 174,786 B [contest-CUDA T4 n600]: pc1 lattice ×16 with the full re-solve supersedes ×8: −790 B against d_pose 5.58e-6 → 5.77e-6 — the first rung where the pose leg turned, admitted on the exchange (2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M1SRR3JGWRKQKH6GSZ3RVTRS`. Lane `ddm_pc1_t4_lattice_x16_on_rc1_20260905`. Modal wall 545.3 s. Archive sha `1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629`, 174,786 B. Runtime tree `5218a79f9e29893a088ba453fc3e09af71a85661d890ec51635839749aff8cb8`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·174,786/37,545,489 | 0.11638282298041185 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·5.77e-06) | 0.007596051605933177 |
| **S** | **0.14411787458634504** |

| | ddm_pc1_t4_lattice_x8_on_rc1_20260905 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1445177913121716 | 0.14411787458634504 | **-0.0003999167258265657** |
| d_seg | 0.00020139 | 0.00020139 | 0.0 |
| d_pose | 5.58e-06 | 5.77e-06 | 1.8999999999999987e-07 |
| bytes | 175,576 | 174,786 | -790 |

## Projection fidelity

Projected 0.1441; realized − projected = 1.7874586345029142e-05. pc1 pre-registered the ×16 break-even at d_pose 1.03e-5 vs base; measured 5.77e-6; byte delta exact

## The mechanism

The pose carrier's 600 × 12 coefficients were re-quantized onto a lattice coarsened ×16 (from the shipped int12 range) and RE-SOLVED for every pair with the
full n600 damped Gauss–Newton (`ddm_jg5.refine_pair`, the fs2 solver verbatim) against the frozen PoseNet on the shipped renders; the twelve basis atoms are
bit-identical to the shipped carrier. This rung SUPERSEDES the ×8 rung (29th move) on the same coefficient block. The first rung where the pose leg turned:
d_pose 5.58e-6 → 5.77e-6 (+3.4 %, +1.86e-4 S) against −790 B (−5.26e-4 S) — net −3.99e-4 S; the lattice law's knee is between ×8 and ×16 on the pose leg while
the byte leg still pays. Token tail, model sections (rc1's adaptive coding) and basis are byte-identical to the 27th-move archive; only the carrier coefficient
block moved. Seal `SEAL_ddm_pc1_lattice_x16_resolved_on_rc1.json`, admit bar derived from the 29th-move pointer; memo `ddm_pc1_pose_carrier_efficiency_20260905.md`;
law `pose_carrier_basis_rate_fidelity_exchange_v1`.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.02411787458634504. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.02773505160593318: archive ≤ 138,565.3 B → **-36,220.7 B**.
- **DISTORTION corner** at held bytes 174,786: distortion ≤ 0.0036172 → **7.7× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **5,432.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pc1_t4_lattice_x16_on_rc1_20260905/MODAL_REMOTE_RESULT.json` (sha `3409b37ce3fe17cbfdf09b3a30ae6210b36bece0210f8a4e4ce7cdf92583cfcf`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained/v3x16_on_rc1_candidate_runtime/archive.zip` (sha `1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629`, 174,786 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/SEAL_ddm_pc1_lattice_x16_resolved_on_rc1.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x16_30th_move/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_pc1_lattice_x16_30th_move/SEAL_ddm_pc1_lattice_x16_resolved_on_rc1.json` (sha verified: True).

## What this does NOT claim

Pose-changing move: the T4 row is authority for d_pose (5.77e-06 at 3 sig figs). The pose leg got WORSE for the first time on this lever (5.58 → 5.77e-6); the move
is admitted on the exchange, not on fidelity. A ×32 rung is not pre-registered as paying: extrapolating the byte leg (≈ −700 B) against a pose leg now rising
(+3.4 % per doubling and accelerating) puts ×32 near break-even — price before solving. Four carrier moves tonight are one lever pulled four times. Not evidence
that the basis can be touched (V1/V2/V4/V5 REFUSED). `[contest-CPU]` stays RECORD-WITH-REASON. PR #140 is now seven moves behind; a public update is the operator's decision.

## Next from here

sj1's multi-pass token pre-distortion (pass 2a nearly complete, ~42 % of flips repaired) → jg5 admission + carrier re-solve FROM THIS lattice and these coefficients
(its pointer guard enforces it) → exact re-encode → candidate on this base. The carrier's remaining named doors: the per-atom quantizer step (3,731 B at fixed
alphabet — orthogonal to the lattice factor and untested with re-solve), the packed Rice-k field width, and pricing ×32 before solving it. Then the machine
clears for md3's resume (448 steps → partition read) and cl3's HPAC rungs.

Equations leg (`tac.canonical_equations`): pose_carrier_basis_rate_fidelity_exchange_v1: third T4 anchor on the lattice axis — ×16 pays on the exchange (−790 B vs +1.86e-4 S pose) with the pose leg rising for the first time; knee located between ×8 and ×16 on fidelity, beyond ×16 on S

Own-vehicle frontier: **S 0.14411787458634504 @ 174,786 B [contest-CUDA T4 n600]**, archive sha `1de6c5d7…5a629`.
