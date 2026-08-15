# hv1 ep0634 T4 fire — execution record (MAIN, 2026-08-15)

## What fired
The sealed hv1 fire-order (.omx/research/ddm_hv1_t4_sealed_fire_order_ep0634_20260815.json)
executed by MAIN. Candidate hv1_ep0634_s1p25_c1p0_brotli_q10: archive 182,759 B, sha
80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e (−743 B vs the e480b v2
incumbent 183,502 B). Local full-raw decode proven byte-identical to the incumbent's decode
(sha e5539653…, CPU axis, four-thread lifted F26). Advisory projection S 0.15959729295498598
(−4.947332e-4 vs the incumbent authority row). Single-axis waiver recorded (CPU-axis
re-target owns its own row per the fire-order's NEXT_IF_RESUMED).

## LIVE call
- **Attempt 2 (LIVE): call fc-01M036FY225QC9A75CM0Y7X7NP**, dispatched 2026-08-15T17:11:12Z,
  T4 / scorer cuda, output dir experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2.
- Harvest poller pid 84314 (canonical tools/modal_harvest_poller.py via
  tools/launch_detached_process.py, deadline 9000 s, poll 20 s); poller owns terminal
  ledger closure; done receipt hv1_ep0634_r2_t4_poller.done.
- Lane lane_ddm_hv1_ep0634_harvest_exact_contest_cuda_20260815, claim r2 status=dispatching.
- Est. cost ~$0.16; #381 running total ≈ $5.9 + attempt-1 ~$0.01 + this row.

## Runtime staging (the MAIN_STAGED placeholders, resolved)
- MAIN_STAGED_CUDA_RUNTIME_DIR =
  /Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/hv1_ep0634_s1p25_c1p0_brotli_q10
  (SWAP_PROCEDURE VERIFY_SOURCE + STAGE_NEW_GENERATION; generation staged, NOT activated —
  steps 3–7 untouched, packet authority remains generation 0 / e480b).
- MAIN_STAGED_CUDA_RUNTIME_TREE_SHA256 =
  70ec7bb1a673dcc4b828b7d826603e365092d524fc74ee0e8c4f2ad66bfcf6e8 (Modal-projected, computed
  via the canonical runtime-manifest helper; content tree aafb5f6b9a80…).
- Runtime = the incumbent e480b_submission_v2 receiver byte-for-byte EXCEPT inflate.py's two
  candidate-binding pin constants (ARCHIVE_SHA256 e3e6f440→80d9c8c6, ARCHIVE_BYTES
  183_502→182_759). The device-adaptive block ("cuda" if available) is UNCHANGED — the exact
  decode path that produced the incumbent authority row.

## Attempt 1: refused BY DESIGN (~$0.01 lesson, 8.7 s remote)
Call fc-01M0367KFQ27VM3E8F7K6092DA. MAIN staged the incumbent runtime with only the archive
swapped; its inflate.py `_verify_input` (the #402 fail-closed receiver hardening) raised
"archive.zip does not match the promoted F26 artifact" — the runtime PINS the promoted
archive's sha+size, so the runtime tree is CANDIDATE-BOUND and can never be reused across
archives. **Law re-learned: on this receiver family a new candidate always requires its own
runtime generation whose inflate.py pins that candidate; "same runtime tree, new archive" is
structurally impossible and the tree sha necessarily changes per candidate.** The hv1 arm's
retained CPU receiver (lifted_submission_cpu) already carried the correct pin; it hardcodes
device "cpu" (the f26p CPU lift), so the CUDA staging took the incumbent's device-adaptive
inflate.py with pins updated instead. Local _verify_input dry-check passed before re-dispatch.

## Tool defect observed (routed to #1064 cluster, NOT fixed mid-fire)
experiments/modal_auth_eval.py refuses AFTER a successful spawn: the dispatch writes its own
'dispatched' call-ledger row, then `claim_modal_auth_eval_dispatch` → `assert_modal_single_flight`
sees that very row as a live conflict and raises — both attempts spawned successfully and then
printed a MODAL SINGLE-FLIGHT REFUSAL. The remote job is unaffected; the spawn JSON + call id
are written before the raise. Post-spawn claim/consumer steps after that point do not run
(the harvest poller + manual claim rows cover closure). Self-conflict ordering fix owed.

## On harvest (poller writes MODAL_REMOTE_RESULT.json)
- If S < 0.1600920261571558 → frontier-pointer update (refresh_canonical_frontier path),
  hot-state + memory m04 update, #1058/#1066 rows updated.
- Components expected: seg 0.029611 (identical decode) · pose 0.0082946 (identical decode) ·
  rate 25·182,759/37,545,489 = 0.12169… — any component drift from the incumbent's is signal
  (decode identity was proven on the CPU axis; the T4 row measures the CUDA-decode reality).
