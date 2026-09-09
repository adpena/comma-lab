# THIRTY-SEVENTH POINTER MOVE — S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]: cmp2: sm1's semantic-section shared-mixer coder composed onto move 36 — −384 B at zero distortion, 38/38 tensors identical (37th pointer move) (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M23WXN0HZD42AHW11RY4X1MH`. Lane `ddm_cmp2_t4_sm1_semantic_coder_20260909`. Modal wall 1079.7 s. Archive sha `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`, 180,388 B. Runtime tree `165df8de74e584e2a5692471f2a7d5b81db1537a232f4ee0ecaf604b5347c27a`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,388/37,545,489 | 0.12011296483580225 |
| seg 100·0.00010698 | 0.010698 |
| pose √(10·5.05e-06) | 0.007106335201775948 |
| **S** | **0.13791730003757818** |

| | ddm_cmp1_t4_rc3_tc1_composed_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13817298987557713 | 0.13791730003757818 | **-0.0002556898379989514** |
| d_seg | 0.00010698 | 0.00010698 | 0.0 |
| d_pose | 5.05e-06 | 5.05e-06 | 0.0 |
| bytes | 180,772 | 180,388 | -384 |

## Projection fidelity

Projected 0.13791730003757818; realized − projected = 0.0. zero-distortion custody replay: exact row

## The mechanism

cmp2 (`.omx/research/ddm_cmp2_compose_sm1_fe1_20260909.md`, seal `SEAL_ddm_cmp2_sm1_fe1_composed.json` build A, landed 310729f13) re-based sm1's counted 24-weight shared-mixer coder for the SEMANTIC renderer section (`.omx/research/ddm_sm1_semantic_section_shared_mixer_coder_priced_closed_form_20260909.md`: bound-then-mix over rc1's adaptive per-group tree coder on the SM3R int4 codes + fp16 row scales) onto the move-36 tree. All 38 public-decoded tensors are bit-identical to the base, so the renderer, its renders, the field, the carrier and the model rows are unchanged; hpac and tail (cmp1's) untouched. Rate 180,772 → 180,388 B (−384 B on this tree vs −263 B when sealed against move 35 — the container interacted favourably by 121 B). Twin encodes byte-identical; public smoke pair on both roles. T4: d_seg 0.00010698, d_pose 5.05e-6 (the move-35 prints), 180,388 B.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017917300037578188. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017804335201775948: archive ≤ 153,479.4 B → **-26,908.6 B**.
- **DISTORTION corner** at held bytes 180,388: distortion ≤ -0.00011296 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-169.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_cmp2_t4_sm1_semantic_coder_20260909/MODAL_REMOTE_RESULT.json` (sha `14bbd5df8cbefa9d7ce2dc146f165162d37dd19728e0621bcdb07c22d9f17dfe`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime/archive.zip` (sha `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`, 180,388 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/SEAL_ddm_cmp2_sm1_fe1_composed.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_cmp2_t4_sm1_semantic_coder_20260909/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_cmp2_t4_sm1_semantic_coder_20260909/SEAL_ddm_cmp2_sm1_fe1_composed.json` (sha verified: True).

## What this does NOT claim

No distortion change (custody replay). Build B (fe1's pair-331 seg-neutral FiLM code applied before re-coding, 180,372 B = 16 B more) is retained but NOT fired: 16 B = 1.07e-5 S sits under the −2e-5 admit bar, and its neutrality on the shipped bytes was not re-proved on the scorer lane; fe1's original −61 B container draw does not carry (lottery law) and priced differently under the arithmetic coder. Not claimed: any transfer of the 121 B favourable container interaction (a draw, not a law). Projection residual: exact (custody replay). No CPU-axis number.

## Next from here

1. The coder-level squeeze has now covered every section: model rows (rc1 → rc2 → rc3), tail (tc1), semantic (rc1 → sm1), carrier (pc2 scales; lattice closed). Since move 27 it took ≈ 3.2 KB of the −28 KB rate demand; the remainder is a field/representation question (gs3 Addendum 17).
2. Live: sj1 pass 5 (field, seg objective, priced through tc1's coder; stop at admission) and rp1 (field, RATE objective at strict argmax identity; sizing re-ranked under tc1's mixer). Whoever seals second re-bases onto the other's field.
3. The operator's PR #140 question stands: the pointer is now 13 moves and 0.0100 S ahead of the posted row; the swap packet is prepared only on the operator's one-line confirm.

Equations leg (`tac.canonical_equations`): tac.canonical_equations: model_section_adaptive_recode_ceiling_v1 (sm1 anchor: semantic section −263 B alone / −384 B composed) + model_section_edit_container_break_fee_v1 (container interaction +121 B favourable — a draw)

Own-vehicle frontier: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]**, archive sha `670d38d0…ce6bc`.
