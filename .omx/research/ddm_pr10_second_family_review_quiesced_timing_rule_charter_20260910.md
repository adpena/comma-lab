# ddm_pr10 — second-family adversarial review of the quiesced decode-timing instrument (charter, MAIN 2026-09-10)

## Why (the exposure)
MAIN built `src/tac/decode_timing_concurrency.py` + `tools/quiesced_decode_timing.py` (bf467026f,
fc42f8a52) after finding that dwc1's decode_wall_clock validator had no door (memory
`validator_contract_no_producer_can_satisfy_is_a_forever_refusal_test_the_producer_on_the_pass_path_20260910`).
The rule that decides "competing" was CHANGED TWICE while two real runs were on the table
(quarter-core pcpu → measured impact + one-core pcpu outside the instrumented stages). The
admitted calibration (move 40, 797.1 s cold, ratio 1.2420) and the refused corroboration (785.1 s)
both sit under that final rule. A rule fixed after seeing the data is exactly the tuning-to-pass
shape this program forbids; MAIN's own reasoning is recorded in the module docstring and in
`.omx/research/ddm_dwc1_decode_wall_clock_seal_leg_20260910.md` (MAIN addendum 1). Review it as an
adversary from the second model family.

## Read first
- `src/tac/decode_timing_concurrency.py` (docstring = the rule), `tools/quiesced_decode_timing.py`,
  `src/tac/tests/test_decode_timing_concurrency.py`, `src/tac/decode_wall_clock.py` (the validator).
- Receipts (read-only; do not modify or move): `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/`
  `move40_quiesced/CONCURRENCY.json`, `move40_quiesced2/CONCURRENCY.json`, `move40_quiesced_local.json`,
  `move40_quiesced2_local.json`, `move40_quiesced_calibration.json`, `move40_quiesced_decode_wall_clock.json`;
  stage checkpoints `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/move40_quiesced{,2}/frame_checkpoints/`.
- `.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json` (MAIN's consumer record) and the dwc1 memo.
- CLAUDE.md non-negotiables (NO FAKE; scores only via upstream/evaluate.py; verdict_scope on every negative).

## Questions (answer each with evidence from the receipts, not opinion)
1. Is the measured-impact rule (total stage excess over the run's own median ≤ 1 % of wall) a
   sound contention detector for THIS decode (four CPU threads on an 18-core / 6 P-core host)?
   Re-derive the per-stage table from the checkpoint mtimes yourself; state the noise floor
   (stage-to-stage jitter) and whether 1 % and the 5 % per-stage listing threshold sit above it.
2. Does the rule admit attempt 1 and refuse attempt 2 for reasons that would survive a swap of the
   two runs' contamination? Show what each attempt's verdict would be under the ORIGINAL
   quarter-core rule and under yours; if you would set different thresholds, derive them.
3. The uninstrumented ~38 % of the wall (native build, verifier, render) is judged by process
   samples at ≥ one full core. Is that defensible, or does a class of contention escape both
   instruments? Name it concretely if so.
4. Direction of error for the GATE: with T4 seconds fixed, a slower contaminated local base makes
   candidate projections OPTIMISTIC. Bound the optimism for the admitted calibration from the
   receipts (excess seconds / wall) and compare with the 0.7 safety factor in the 1,260 s limit.
5. Anything in the instrument that could manufacture a PASS: settle-sample exclusion, the ancestor
   exclusion (the launching control plane), the `--pause-pid` SIGSTOP of the dashboard, the
   re-derivation of competitors from the ≥ 5 % `visible` list at assembly time.
6. The producer/validator "gate with no door" class: grep for other validators in `src/tac/` whose
   PASS path is exercised only by synthetic fixtures (a required field no committed producer writes).
   List candidates with file:line; do not fix them.

## OPTIMAL FORM
- Reference form: a review, not a build. Family reference = pr8/pr9 second-family reviews
  (`.omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md`, `..._pr9_..._20260910.md`):
  per-claim evidence table, verdict_scope on every negative, follow-on dispositions typed.
- Scope deltas: none — read every receipt named; no sampling of the sample list.
- Provenance pins: commits bf467026f, fc42f8a52; the receipts' sha256 as recorded in
  `MOVE40_QUIESCED_LEG.json`; the sidecar sha prefix e58adf1d.
- No code changes, no receipt edits, no reruns (a rerun is MAIN's call; recommend one if warranted).

## Prior negatives accounted (operator 2026-08-15)
- pr6 found Addenda 19/20 over-generalized (13 misreadings) — every negative here carries a
  verdict_scope line and cites the receipt field it rests on.
- dwc1's own 175 tests passed on a fixture that no producer could reproduce — do not accept a test
  as evidence; read the receipts.
- The rule changed after the data: name any threshold whose only justification is that it admits
  attempt 1.

## Deliverable
`.omx/research/ddm_pr10_second_family_review_quiesced_timing_rule_20260910.md` committed via the
serializer (`REVIEW_GATE_OVERRIDE=1` allowed: .md only; message tagged `[no-triality] [p0-ledger-ok]`;
no co-author trailer), with a verdict in {CLEAR, CLEAR-WITH-CONDITIONS, RERUN-REQUIRED, REFUSE} and,
for anything but CLEAR, the exact receipt fields and the exact threshold or rule change you require.
Checkpoint as `ddm_pr10` every ~10 tool uses. Final message: verdict, the three strongest findings,
every boundary, and the frontier line
`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
