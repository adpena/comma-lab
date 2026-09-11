# ddm_pr19 — adjudicate two contract questions the move-46 candidate raised: (A) may the pre-fire RISK gate be satisfied WITHOUT a local decode ratio when the receiver behavior delta is EMPTY and the counted model is smaller; (B) may decode-wall-clock inheritance follow a chain whose every link shares one receiver behavior digest (charter, MAIN 2026-09-11; PRE-WRITTEN, spawn only if the matched pair refuses)

## Why (measured, 2026-09-11)
ntb2's move-46 candidate (`ddm_ntb2_frame_even_hpac_prior_move45`, 180,001 B, −245 B, output-lossless: the HPAC prior is built from
the same bytes at both ends; cold n600 raw byte-identical to the pointer's; receiver behavior delta EMPTY, both trees 9f6e7168…)
passed all eight non-timing gates and is blocked on `validate_prefire_risk`, which projects `fraction = candidate_local/base_local − 1`
from LOCAL cold-n600 decode diagnostics onto the T4 leg (1,232.418725255 s; limit 1,260 s). MEASURED: the same move-44 bytes decoded
in 1,957.60 s (pc3's window) and 1,619.12 s (ntb2's window) — a 20.9 % spread on IDENTICAL bytes — while the candidate's true
decode-work delta is ≈ 0 (same 117,964,800 symbol decodes; model 603 B smaller). The gate's instrument is an order of magnitude
noisier than the effect it must resolve. Second fact (ntb2, receipt `.omx/research/ddm_ntb2_20260911/TIMING_INHERITANCE_CHAIN_BREAK.json`):
an inherited leg cannot be inherited again and the source must be the pointer archive, so every OTHER rate-only move needs a
first-measurement dispatch even when nothing in the receiver changed. Both are dwc1-genus questions (a gate no honest producer can
satisfy on this host is a forever refusal — memory `validator_contract_no_producer_can_satisfy_is_a_forever_refusal_test_the_producer_on_the_pass_path_20260910`).
The right answer may still be "no": the pre-fire risk gate exists so a candidate never times out on the paid T4 run; the question is
what EVIDENCE of "no added decode work" the contract accepts, not whether the guard should exist.

## Deliverable (Opus; contract text + typed amendment through the frozen file, as pr18 did)
1. Adjudicate (A) with a written rule: e.g. the risk receipt may carry `decode_work_delta: "none"` ONLY when (i) the scoped receiver
   behavior digest is EQUAL to the source leg's, (ii) the cold n600 raw is byte-identical to the pointer's (a receipt the intent already
   binds), and (iii) the counted model section that the receiver materializes is ≤ the pointer's in bytes; then the projection is the
   source leg's own value (fraction 0 by construction) and the local ratio is recorded as diagnostic, not required. Say exactly which
   timing-out failure this could let through and why (i)–(iii) exclude it, or refuse the amendment with the same rigor.
2. Adjudicate (B): whether inheritance may follow a chain to its terminating t4_direct when every link's behavior digest equals the
   candidate's, with a bound on chain length or none, and what the completed leg records. If refused, say what a cheaper legal path is.
3. Implement whatever is adjudicated YES exactly as pr18 did: versioned definitions; consumers routed; typed amendment + consumer-fix
   rows appended to `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` (validated by `_pf_freeze_history`); tests for every
   refusal and pass; the five suites green; pr17 NO-GRACE announced (in-flight intents invalidated — ntb2's would be re-emitted).
4. Memo `.omx/research/ddm_pr19_…_20260911.md` with the clause table and the measured windows above; `# FORMALIZATION_PENDING:<rationale>`.

## Boundaries
No Modal, no fires, no candidate work; never weaken the 1,260 s ceiling or the t4_direct leg's completed requirements; never edit
arms' directories or the pointer receipts; serializer commits, two review passes per .py, no co-author trailer, `[no-triality] [p0-ledger-ok]`.

## OPTIMAL FORM
- Reference form: pr12's contract + pr14/pr18's amendment pattern (definition parent, typed consumer fixes, real-tree proofs).
- Provenance pins: pr18 commits 5d2632ee4 + f1b9a0dbb; ntb2's memo + receipts; pc3's paired-concurrent timing (fraction 0, move 45);
  the four local windows (1,957.60 / 1,619.12 / 2,214.57 s and the matched pair when it lands).

## Prior negatives accounted (operator 2026-08-15)
- dwc1 (gate with no door); r9m ×5 (env-coupled digests → content-only); pr9 (stale manifest); tc4 (a receiver change NEEDS a measured
  decode wall-clock — this charter must not reopen that: (A) applies only when the receiver behavior is UNCHANGED).
