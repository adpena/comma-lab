# ddm_pr11 — adjudicate the frozen timing-admission rule against four refused cold runs (charter, MAIN 2026-09-10)

## The situation (facts, all in receipts)
ddm_pr10 (your family) replaced MAIN's timing rule with a conservative admission rule, now
implemented verbatim and frozen+hashed before launch (`src/tac/decode_timing_concurrency.py` v2,
commit 52c4962cc; rule sha b94474c6167ef51b…): a quarter-core threshold on every non-excluded
process over the whole non-settle window, ancestors included, aggregate cap 200 %, stage rates
diagnostic only, three consecutive quiet settle samples, paused-process custody.

Under that rule, and under the earlier rules, the move 40 receiver has now been timed cold FOUR
times on this host (18 logical / 6 P cores), always with bit-identical raw output
(sha c5a7986c…) and always refused:

| attempt | wall s | rule at launch | refused by |
|---|---:|---|---|
| move40_quiesced | 797.146 | pre-data (25 %) | two venv pythons 66.9 % (MAIN's own tool calls) |
| move40_quiesced2 | 785.138 | pre-data (25 %) | Codex desktop app 124 %, pythons 100 %, tar, git |
| move40_quiesced3 | 783.057 | FROZEN pr10 rule | `dasd` 95–97 % for ~3.3 min from ~6 min in, `syspolicyd` 33–64 %, aggregate ≥ 200 % once |
| move40_quiesced4 | 784.772 | FROZEN pr10 rule + `caffeinate -u` | `dasd` 95–97 % for ~3.7 min from ~8.7 min in, `syspolicyd` 27–73 %, aggregate ≥ 200 % four times |

Stage-rate diagnostics (25-pair checkpoints, 62–65 % of the wall): total excess over the run's
own median 0.71 % / 0.41 % / 0.39 % / 0.15 %. Spread of the four walls: 1.8 %. The receiver's
ACTUAL contest-T4 decode is 990.053829427 s (retained Modal receipt), limit 1,260 s.

`dasd` (macOS Duet Activity Scheduler) and `syspolicyd` recur roughly every 16 minutes for 3–5
minutes, regardless of user-activity assertion; a 13-minute decode plus one minute of settle
cannot fit between bursts by period alone. MAIN did NOT amend the rule after these refusals (that
is the defect your review named) and asks your family to adjudicate on the receipts.

## Read first
- `.omx/research/ddm_pr10_second_family_review_quiesced_timing_rule_20260910.md` (your family's review).
- `src/tac/decode_timing_concurrency.py` (v2), `tools/quiesced_decode_timing.py`, tests; `src/tac/decode_wall_clock.py`.
- Receipts (read-only): `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/`
  `move40_quiesced{,2}/CONCURRENCY.json` (v1 sampler), `move40_quiesced3/CONCURRENCY_attempt3_refused.json`,
  `move40_quiesced4/CONCURRENCY.json` (v2 sampler: frozen rule, hash, freeze/start times, paused custody),
  `move40_quiesced{3_local_refused,4_local}.json`; stage checkpoints under
  `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/move40_quiesced{,2,3_refused,4}/frame_checkpoints/`.
- `.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json` (MAIN's record incl. attempts 3–4).
- rlc1's candidate runs under the OLD rule (both refused, 829.0 / 831.5 s): `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json`.

## Questions (answer from the receipts)
1. Are `dasd`/`syspolicyd` bursts a contention class the quarter-core rule must count on this host,
   given four runs with ≤ 0.7 % measured stage excess through them? If the rule stands, state the
   expected number of attempts to obtain one admissible 13-minute window from the observed burst
   period and duration, and whether that is an instrument that can ever produce a calibration.
2. If an amendment is warranted, WRITE IT EXACTLY (fields, thresholds, comparisons) so MAIN can freeze
   it BEFORE the next run — e.g. a named-daemon class with its own per-process cap validated by the
   stage-rate instrument, or a burst allowance bounded by measured impact — and state what each
   amendment would have admitted or refused across the four runs. You are the second family; MAIN
   will implement your text verbatim and not edit it.
3. Direct-T4 leg mode: `tac.decode_wall_clock` has no way to admit the candidate's OWN completed
   contest-T4 decode as the timing authority without a local calibration (`candidate_t4_receipt` only
   binds beside a measured/calibrated leg). Your review allowed "a direct exact contest-T4 cold
   public-entrypoint time for the candidate". Specify a `mode: "t4_direct"` leg contract (bindings:
   archive sha, receiver digest, T4 runtime digest, hardware field, completed inflate seconds ≤ limit;
   inheritance semantics for a receiver-identical candidate such as rp1's) or say why it must not exist.
4. For rp1's candidate specifically (receiver byte-identical to move 40's except archive.zip and the two
   pins; move 40's actual T4 decode 990.054 s): which of (a) inheritance from a `t4_direct` move-40 leg,
   (b) inheritance from a calibrated local leg, (c) its own T4 fire, is the admissible timing authority?

## OPTIMAL FORM
- Reference form: pr10's review (per-claim evidence table, verdict_scope on every negative, typed
  follow-on dispositions). This is adjudication, not a build: no code, no receipt edits, no reruns.
- Provenance pins: commit 52c4962cc (+ the `--assert-user-activity` follow-up), rule sha b94474c6167ef51b,
  the receipt paths above (record their sha256 in your table).
- Scope deltas: none — read all four runs' samples; do not sample the sample lists.

## Prior negatives accounted (operator 2026-08-15)
- pr10: a rule fixed after the data is the defect — this charter asks your family to fix the rule,
  and MAIN freezes it before the next run.
- dwc1: a validator no producer could satisfy (gate with no door) — question 1 asks whether the
  current rule has become one on this host.
- Attempts 1–2 were refused under MAIN's rule for MAIN's own activity; 3–4 under yours for system
  daemons — both classes are in the receipts, neither is excused here.

## Deliverable
`.omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md` committed via the
serializer (`REVIEW_GATE_OVERRIDE=1`, .md only, tags `[no-triality] [p0-ledger-ok]`, no co-author
trailer): verdict in {RULE-STANDS, AMEND (exact text), T4-DIRECT (exact contract)}, what each option
admits/refuses over the four runs, and the timing authority for rp1's and rlc1's candidates.
Checkpoint as `ddm_pr11` every ~10 tool uses. Final message: verdict, the exact rule text if amended,
every boundary, and the frontier line `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
