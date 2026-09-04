# TWENTY-FIFTH POINTER MOVE — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4, n600]: the carrier re-solve on fs1's moved pairs (2026-09-04)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)
`upstream/evaluate.py --device cuda`, Tesla T4, Linux x86_64, all 600 samples, GT lineage DALI_NVDEC. Modal call
`fc-01M1Q6W3R8WWDQPRFYSF7SWTKP`, lane `ddm_fs2_t4_carrier_resolve_20260904` (re-fire r2; the first fire hit a transient Modal
"source modified during build" error — my `ruff format` raced the image build). Harvested 22:10:31Z, 656.9 s Modal wall. Archive sha
`a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6`, 180,023 B; runtime tree 915d25f9…. `passed: true`,
`validation_errors: []`.

Recomputed FROM COMPONENTS (#877): rate 25·180023/37545489 = 0.11986992607... · seg 100·0.00020139 = 0.020139 ·
pose √(10·6.14e-6) = 0.007836134... → **0.14784474152757654**.

| | fs1 (24th) | fs2 (25th) | delta |
|---|---|---|---|
| S | 0.14786319521362173 | 0.14784474152757654 | **−1.8453686045194484e-5** |
| d_seg | 0.00020139 | 0.00020139 | 0 (structurally identical) |
| d_pose | 6.17e-6 | 6.14e-6 | −3.0e-8 |
| bytes | 180,022 | 180,023 | +1 |

Projection fidelity: fs2's macOS-CPU advisory projected 0.14784104973157752; realized − projected = +3.69e-6, inside the ±3.2e-6…
band to within a hair (the pose print is 3 sig figs). Second pure-pose authority anchor for `exchange_ratio_noise_floor_v1`.

## The mechanism
fs1 moved frame 0 on 21 pairs but their 12-dim pose-carrier codes were fitted against the OLD frame 0. fs2 re-solved those codes
(damped Gauss–Newton against the frozen PoseNet on the shipped renders): 15 of 21 pairs moved 67 int12 coordinates for **+1 B**
(the `up2` control reproduced the shipped 78,628 bits exactly). The matched control on the base body (frame 0 unmoved) changed 0/21
coordinates — 100% of the win is selector-induced staleness, MEASURED not inferred.

## Sub-0.12 arithmetic RE-DERIVED at this move
gap 0.02784474152757653. Exchange 6.658589531221714e-7 S/B (unchanged).
- RATE corner: archive ≤ 138,205.2 B → **−41,817.8 B** at held distortion 0.027974815
- DISTORTION corner: distortion ≤ 1.3007e-4 → **215.1× reduction** at held bytes.
- Zero-distortion B_max 180,218.347 B → the archive is **195.347 B under** the threshold at zero distortion.

## Custody
Harvest `/Volumes/APDataStore/pact/ddm_fs2_carrier_resolve_t4_buy_20260904_r2/MODAL_REMOTE_RESULT.json`; mirror
`experiments/results/modal_auth_eval_mirror/contest_auth_eval_fs2_carrier_resolve_t4_20260904_r2.json`; archive+seal on Vertigo
(`ddm_fs2_carrier_resolve/`) + APDataStore `custody_pointer25/`. Pointer promoted (no maturity refusals); claim closed. Public best
unchanged (PR #135, 0.162).

## Consequence for PR #140
PR #140 still carries afr1's bytes (23rd move, 180,002 B). The local frontier is now TWO moves ahead (fs1 → fs2). ps1's prepared
update packet targets the fs1 bytes; if the operator elects to update the public PR, the packet now wants a SEVENTH stage (the fs2
carrier re-solve) on top of stage 6, +1 B. Updating the public PR remains the operator's decision (p0_swap_procedure).

Own-vehicle frontier: **fs2 — S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]**, archive sha a8f3a379…0427bb6.
Equations leg (`tac.canonical_equations`): `exchange_ratio_noise_floor_v1` — second pure-pose authority anchor (projection +3.69e-6).

## ERRATUM (hv1 replay, 2026-09-04 23:45Z) — transcription errors in this memo, found by `tools/pointer_move_packet.py`
The S value and the delta are correct. Hand-typed terms were not: rate 25·180023/37545489 = **0.11986992631791266** (memo wrote 0.11986992607…);
pose √(10·6.14e-6) = **0.007835815209663893** (memo wrote 0.007836134…); gap = **0.027844741527576544**. The claim terminal row carried a 12-hex
runtime-tree sha and a non-canonical status prefix; canonical rows are now apparatus-written. Memo `ddm_hv1_harvest_to_pointer_autopilot_20260904.md`.
