# TWENTY-FOURTH POINTER MOVE — S 0.14786319521362173 @ 180,022 B [contest-CUDA T4, n600]: the first exact row of the post-submission wave (2026-09-04)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py --device cuda`, Tesla T4, Linux x86_64, all 600 samples. Modal call
`fc-01M1PM1KR3CQN5E5BC62WE4AD7`, lane `ddm_fs1_t4_frame0_selector_20260904`, dispatched
2026-09-04T16:29:18Z, harvested 16:39:16Z (545.5 s Modal wall: inflate 492.7 s + evaluate 38.2 s).
Archive sha `50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf`, 180,022 B;
runtime tree `fbf4aaf436aa02814d0558bfbc2bf4307502bdac49a7616b66bcfa31b44ca43c` (the g8v1 public
PR tree with exactly two `inflate.py` pin lines re-pinned, proved by diff in fs1). GT lineage on the
scorer host: DALI_NVDEC (authority=True). `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded 0.15 display):

| term | value |
|---|---|
| rate 25·180,022/37,545,489 | 0.11986926045895953 |
| seg 100·0.00020139 | 0.020139 |
| pose √(10·6.17e-6) | 0.007854934754662192 |
| **S** | **0.14786319521362173** |

| | afr1 (prior pointer) | fs1 | delta |
|---|---|---|---|
| S | 0.14797617125559104 | 0.14786319521362173 | **−1.1297604196930378e-4** |
| d_seg | 0.00020139 | 0.00020139 | 0 (structurally identical, as fs1 declared) |
| d_pose | 6.37e-6 | 6.17e-6 | −2.0e-7 |
| bytes | 180,002 | 180,022 | +20 |

The move is 2.095× the afr1 move and 5.65× the 2e-5 admit bar. **Projection fidelity:** fs1's
n600 batch-8 advisory projection was 0.1478658102574271; realized − projected = −2.6e-6, inside
the 8-dp report bound (±3.2e-6 on S from the printed pose). The advisory instrument reproduced
the authority to 5 significant figures on a pure-pose edit.

## The mechanism (what changed, in one sentence)

pr1's terminal pose re-solve had never been byte-closed: the shipped runtime carried a per-pair
frame-0 selector whose ENCODER did not exist (decode-only). fs1 built the encoder
(`encode(decode(shipped))` rebuilds the 14-byte blob exactly; 300/300 fuzz round-trips), scanned
the one-dimensional byte-optimal set, and re-selected **21 of 600 pairs** at +20 B. SegNet reads
frame 2p+1 and the selector writes frame 2p, so d_seg is identical by construction; 579/600
unchanged pairs measure Δd_pose = 0.0 exactly.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.02786319521362174. Exchange 25/37,545,489 = 6.658589531221714e-7 S/B (unchanged).
- **RATE corner** at held distortion 0.027993934754662192: archive ≤ 138,176.5 B → **−41,845.5 B**
  (was −42,016; the pose gain paid 170.5 B of demand).
- **DISTORTION corner** at held bytes: distortion ≤ 1.3074e-4 → **214.1× reduction** (was 195.2×;
  the +20 B ate margin on this corner — the two corners move in opposite directions on a
  bytes-for-distortion trade).
- Zero-distortion B_max 180,218.347 B → the archive is **196.347 B under** the threshold at zero
  distortion (was 216.347).

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/t4_buy_20260904/MODAL_REMOTE_RESULT.json`
  (sha `4bcecd01f3481b29c57415a934b5e3fab5f71e823d1a8802637a986093fa54a9`); mirror
  `experiments/results/modal_auth_eval_mirror/contest_auth_eval_fs1_frame0_selector_t4_20260904.json`.
- Archive + seal: primary `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/` (seal sha
  `21b2e351e401f03a62fc48ce57cbc24a0d6e46709e115067cae1eb3f88a13483`); second copy
  `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/custody_pointer24/` (archive sha verified).
- Pointer: `tools/refresh_canonical_frontier.py --no-update-upstream` promoted the mirror; maturity
  refusals: none; effective_frontier = our_local_frontier_contest_cuda = fs1. Claim closed
  (`completed_modal_auth_eval_harvested_…`). Public leaderboard best unchanged (PR #135, 0.162).

## What this does NOT claim

- `[contest-CPU]` stays RECORD-WITH-REASON (single-axis waiver: same-object pose-only edit; the
  prior CPU attempt timed out at the 1,800 s inflation budget). No CPU score is inherited.
- PR #140 still carries the afr1 bytes (180,002 B, S 0.14797617125559104). Whether to update the
  public PR to these bytes is the **operator's decision** (p0_swap_procedure: no publish without the
  one-line confirm). The packet delta would be: archive +20 B, `inflate.py` two pin lines, README/
  report numbers; the disclosure text is unchanged.

## Next from here (the wave continues)

The post-hoc doors are otherwise closed on measurement (rf1, ft1, ar1, pr1's 41.5× residual). The
live candidates are the burn cells (ng2 area-cap, ng3 τ-band, ng4 continuous-objective) — any
composition ≥ 0.027976 S at the accuracy corner fires the candidate chain (ddm_x012 rule; the
threshold is now re-derived above). Equations leg (`tac.canonical_equations`):
`exchange_ratio_noise_floor_v1` gains its first AUTHORITY anchor on a pure-pose edit
(projection error −2.6e-6 at n600 batch 8).

Own-vehicle frontier: **fs1 — S 0.14786319521362173 @ 180,022 B [contest-CUDA T4 n600]**,
archive sha 50fcaf1a…708cf.

## ERRATUM (hv1 replay, 2026-09-04 23:45Z) — transcription errors in this memo, found by `tools/pointer_move_packet.py`
The S values and deltas are correct. Hand-typed term values were not: pose term √(10·6.17e-6) = 0.007854934754662**193** (memo wrote …192).
The claim terminal rows for this lane carried a status prefix (`completed_modal_auth_eval_harvested_…`) that fails the compliance checker's
`SUCCESSFUL_EXACT_EVAL_TERMINAL_STATUS_PREFIXES`; canonical rows (`completed_contest_cuda_exact_eval_harvested`, full shas) are now written by
the poller/packet, never by hand. Memo `ddm_hv1_harvest_to_pointer_autopilot_20260904.md`.
