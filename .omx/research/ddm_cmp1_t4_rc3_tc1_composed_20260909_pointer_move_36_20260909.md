# THIRTY-SIXTH POINTER MOVE — S 0.13817298987557713 @ 180,772 B [contest-CUDA T4 n600]: cmp1: rc3's model-row mixer + tc1's tail-coder mixer composed onto move 35 — −749 B at zero distortion (36th pointer move) (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M23TGZEB6JP6VY7GF73EC0XP`. Lane `ddm_cmp1_t4_rc3_tc1_composed_20260909`. Modal wall 1129.7 s. Archive sha `66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0`, 180,772 B. Runtime tree `0008176f5d44fc1c1d317476c3c88aae6e6755f5dac47a72229f7463a3ad7253`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,772/37,545,489 | 0.12036865467380116 |
| seg 100·0.00010698 | 0.010698 |
| pose √(10·5.05e-06) | 0.007106335201775948 |
| **S** | **0.13817298987557713** |

| | ddm_sj1_t4_token_predistortion_pass4_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13867171823146562 | 0.13817298987557713 | **-0.0004987283558884892** |
| d_seg | 0.00010698 | 0.00010698 | 0.0 |
| d_pose | 5.05e-06 | 5.05e-06 | 0.0 |
| bytes | 181,521 | 180,772 | -749 |

## Projection fidelity

Projected 0.1381729898755771; realized − projected = 2.7755575615628914e-17. zero-distortion custody replay: projection = exact row to 2.8e-17

## The mechanism

cmp1 (`.omx/research/ddm_cmp1_compose_rc3_tc1_20260909.md`, seal `SEAL_ddm_cmp1_rc3_tc1_composed.json`, landed 127a9b6c1) composed two independently sealed zero-distortion rate rows onto the move-35 tree: rc3's counted 24-weight shared-mixer successor on the IHS1 hpac model rows (−201 B; decoded rows bit-identical) and tc1's 35-weight shared-mixer coder over the shipped HPAC predictor on the token tail (−548 B on THIS field, re-encoded; decoded field byte-identical over all 117,964,800 cells). Sections touched: hpac + tail + their readers; semantic and carrier byte-identical to move 35; twin n600 encodes byte-identical; section census; paired public smokes. Rate 181,521 → 180,772 B (−749 B = 201 + 548 — additive to the byte, no container interaction). T4: d_seg 0.00010698, d_pose 5.05e-6 (both the move-35 prints), 180,772 B.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01817298987557714. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017804335201775948: archive ≤ 153,479.4 B → **-27,292.6 B**.
- **DISTORTION corner** at held bytes 180,772: distortion ≤ -0.00036865 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-553.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_cmp1_t4_rc3_tc1_composed_20260909/MODAL_REMOTE_RESULT.json` (sha `25be905d5951f457ba243eef5e36d6ba258e464798556882a3f07c21416985ab`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/candidate_runtime/archive.zip` (sha `66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0`, 180,772 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/SEAL_ddm_cmp1_rc3_tc1_composed.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_cmp1_t4_rc3_tc1_composed_20260909/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_cmp1_t4_rc3_tc1_composed_20260909/SEAL_ddm_cmp1_rc3_tc1_composed.json` (sha verified: True).

## What this does NOT claim

No distortion change (custody replay of the move-35 renders; the T4 prints are identical). Not claimed: any headroom beyond the bounds — rc3's memo leaves ≈ 700 B of the order-1 gap on the model rows behind the counted-parameter gate; tc1's bound names a joint-context architecture as its live hypothesis. Not claimed: additivity in general — here the two sections happened to add exactly; fe1's lottery law still governs container deltas of edits into range-coded sections. Projection residual: +2.8e-17 (machine precision). No CPU-axis number.

## Next from here

1. cmp2 (codex, un-parked now): sm1's semantic-section mixer coder (−263 B sealed vs move 35) re-based onto this tree, with fe1's pair-331 seg-neutral FiLM code applied before re-coding and priced per symbol under the arithmetic coder (no longer a container draw) → one seal → one T4 call.
2. sj1 pass 5 (running; singles, subset-only, stop at admission) prices its field through tc1's coder against THIS tail as control; rp1 (rate-directed pre-distortion) ranks token changes by tc1's probabilities.
3. Rate corner after this move: re-derived by the packet in the pointer line; the coder-level squeeze has now taken model rows (rc1, rc2, rc3), tail (tc1), and next the semantic section (sm1) — ≈ 2.8 KB of the −28 KB demand since move 27; the remainder is a field/representation question (gs3 Addendum 17).

Equations leg (`tac.canonical_equations`): tac.canonical_equations: model_section_adaptive_recode_ceiling_v1 (rc3 anchor: 24-weight shared mixer −201 B) + token_tail_context_mixing_bound_v1 (tc1 anchor: −548 B on the move-35 field; bound-then-mix)

Own-vehicle frontier: **S 0.13817298987557713 @ 180,772 B [contest-CUDA T4 n600]**, archive sha `66b8d5bb…1b3c0`.
