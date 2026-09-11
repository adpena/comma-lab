# FORTY-EIGHTH POINTER MOVE — S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600]: pointer move 48: S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] — hpr1 even rounding on the refit HPAC prior (-248 B vs move 47; decoded field and raw byte-identical), first-measurement chain with its own measured t4_direct leg of 1,023.3 s (2026-09-11)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M29A1CWQBBDT4XXTQ1K7TG88`. Lane `ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911`. Modal wall 1079.7 s. Archive sha `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`, 179,111 B. Runtime tree `a8bc13af6d8573dc292bbcd22b125ecb252c4d66ff6d15630f95b4c528487426`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,111/37,545,489 | 0.11926266295266523 |
| seg 100·0.00010345 | 0.010345 |
| pose √(10·4.59e-06) | 0.00677495387438173 |
| **S** | **0.13638261682704697** |

| | ddm_hpr1_retrain_control_move46_contest_cuda_20260911 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13654774984742127 | 0.13638261682704697 | **-0.00016513302037429733** |
| d_seg | 0.00010345 | 0.00010345 | 0.0 |
| d_pose | 4.59e-06 | 4.59e-06 | 0.0 |
| bytes | 179,359 | 179,111 | -248 |

## Projection fidelity

Projected 0.13638261682704697; realized − projected = 0.0. conditional arithmetic from the -248 B rounding of the refit prior on move 47's field; realized minus projected = 0

## The mechanism

Move 46's lever re-applied on move 47's object: the HPAC prior's frame embedding (4,800 values) rounded to even for 2,310 of them,
on top of the RETRAINED prior that move 47 shipped. The prior is built from the same bytes at both ends before any symbol is decoded,
so the rounding conditions the tail's code length without touching a decoded symbol: `hpac` 12,262 → 11,629 B (−633), RLC1 stream
118,511 → 118,896 B (+385), archive 179,359 → 179,111 B (−248). Output-lossless by receipt: the decoded token field is byte-identical
to the shipped field (a92e7d90…) and the cold n600 public decode is byte-identical to the pointer's raw (2b762eba…; four archives now
decode to one raw), so d_seg and d_pose are unchanged and the score change is the rate term alone, 25·(−248)/37,545,489 = −1.6513e-4.
The rounding's value did not depend on the prior being stale (−245 B on the old prior, −248 B on the refit). Receiver unchanged
(behavior digest 9f6e7168… on moves 44–48); the move went through the first-measurement chain because the contract admits one
inherited row per measured leg and move 47 had consumed move 46's; its completion gives move 48 a measured T4 leg of its own. The
pre-fire risk gate ran in pr19's identity-class envelope (ceiling 1,232.42 s from moves 44/46's measured legs; stress 1,477.6 s ≤ 1,800).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.016382616827046975. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01711995387438173: archive ≤ 154,507.3 B → **-24,603.7 B**.
- **DISTORTION corner** at held bytes 179,111: distortion ≤ 0.00073734 → **23.2× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **1,107.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/VertigoDataTier/pact/ddm_hpr1_first_measurement/run1/MODAL_REMOTE_RESULT.json` (sha `9bd402caccc1648c3de44fe59b0364b0ddac0decf51fe2bf2163361685220328`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime/archive.zip` (sha `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`, 179,111 B).
- Seal: `/Users/adpena/Projects/pact/.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911/custody_pointer48/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911/custody_pointer48/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json` (sha verified: True).

## What this does NOT claim

Not claimed: any distortion change (field and raw byte-identical); additivity of −245 and −887 (two edits to one object; measured
jointly: −1,135 B against move 45, and only the exact encode says so); the step-4 rounding (LOSES: 179,417 B, +58 B — the trade turns
between step 2 and step 4); a q re-solve on the refit prior (NEGATIVE: −9,440 B held-out; the refit moved q closer to its optimum);
the local timing ratio as authority (hpr1's own matched pair had the candidate 6.6 % faster; pr19's windows had its token decode
+28.4 % slower; the gate demoted the local ratio to a 1,800 s stress bound and the T4 leg is the measurement); the CPU axis
(declaration + refusal receipt); any claim about the shape rung (falsified at +812 B against its control, verdict scope formulation).

## Next from here

Next: tmx1's refit of the tail coder's 35+5-weight mixer (60 B of counted state fit to the move-32 field, eight field moves stale;
prices the 118.9 KB stream; bounded by tc3's confounded prior at tens of bytes; the container lottery does not govern this axis —
one STORE member, tail bytes reach archive bytes 1:1), priced on both move 47's and this move's prior, held-out, receiver unchanged →
first-measurement row (pr19 refused transitive inheritance; the identity-class risk mode makes each such row cheap). After that: the
staleness audit's rung 3 (QAT refit of the 29,862 B renderer; ft1's +31 % d_seg warning; provenance to be recorded first) and hpr1's R6
/ temporal tap sets measured against the RETRAINED control. Rate corner at this move: cap 154,507 B; demand −24,604 B. Storage: sr5's
reserve derivation says the 40 GiB SSD floor is an undived boot-swap copy (measured need ~24 GiB) — an operator/second-family decision.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 47 (decoded field and cold n600 raw byte-identical) and B 179,359 → 179,111: ΔS = 25·(−248)/37,545,489 = −1.6513302e-4 exactly; realized 0.13638261682704697 = projected; T4 inflate 1,023.26 s measured (t4_direct)

Own-vehicle frontier: **S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600]**, archive sha `d830edd3…f149c`.
