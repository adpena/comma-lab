# DDM WS2 warm-start custody producer build spec

## Objective

Materialize the preregistered WS1 `W_seg` and `W_joint` endpoints as real,
receiver-closed archives, prove that the actual J5/#366 consumer can lift and
byte-identically re-emit each archive, run the bounded four-step slope
falsifier, and reseal the selected start through
`tools/reseal_ddm_j7_366_ticket.py`.

## Authority and bounds

- Delegated authority:
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_ws2_warm_start_custody_producer_20260724T053455Z.wrapped.prompt.txt`,
  SHA-256 `07dbdd765c8010176e89420b4e4603facc31a83257dee027e8edd97817f30429`,
  8,479 bytes.
- Lane: `lane_ddm_ws2_warm_start_custody_producer_20260724`.
- Threads: four for frozen-scorer measurement.
- Evidence: `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.
- Bounded `--max-steps` runs only. No campaign launch, paid dispatch, contest
  evaluation, promotion, or frontier-pointer mutation.
- Bulk output uses `/Volumes/VertigoDataTier/pact`; all inputs are SHA-bound at
  consumption and all stage outputs are immutable or atomically published.
- MAIN must independently review the complete branch diff before landing.

## Root-cause contract

The WS1 endpoint rows price the two transforms as
`137827 + payload_bytes`, but they do not contain a receiver archive:

- `W_seg` is the strict Seglex96 nested V19C receiver plus a 204-byte temporal
  affine payload, followed by decoder-derived MyCar hood reassertion.
- `W_joint` is the joint V19C receiver plus a 974-byte local-statistics payload
  and the hard/analytic geometry composition.

J5 currently accepts only the sealed 133,941-byte V15 carrier archive. A valid
repair must preserve the nested receiver programs and the WS1 transform across
every J5 parameter recompile; merely attaching archive metadata to the old
endpoint row is forbidden.

## Implementation surfaces

1. Add a scorer-free WS1 archive/receiver module under
   `src/tac/optimization/`. Its deterministic container must:
   - preserve exact base receiver bytes and the exact counted WS1 payload;
   - parse/re-emit byte-identically;
   - apply the same decoder operations used by the WS1 measurement;
   - expose the underlying V15 carrier for J5 parameter lifting;
   - rebuild the nested coupled-margin and preuint8 wrappers around every
     mutated carrier state.
2. Generalize the J5 lift and launcher receiver calls to the typed WS1 receiver
   without weakening sealed V15 handling.
3. Add a local producer that:
   - validates both SHA-bound recipes;
   - publishes both candidates and parse-back receipts on SSD;
   - remeasures exact n600 at batch32;
   - records any metric or byte drift rather than coercing reconciliation.
4. Extend the landed resealer and typed-program allow-list with candidate-bound
   semantic hashes after archive custody exists.
5. Run the exact four-step bounded windows, arbitrate with
   `R*=4.1215446777965665`, reseal, dry-run, memory-preflight, and one bounded
   re-smoke.

## Acceptance

- Focused unit tests cover malformed/truncated payloads, exact parse/re-emit,
  receiver output parity against the settled WS1 batch hashes, nested
  rewrapping after a carrier mutation, and endpoint-only resealer refusal.
- Each candidate receipt contains real path, SHA-256, bytes, parse-back
  custody, exact batch32 `d_seg`/`d_pose`, and endpoint reconciliation.
- `lift_v15_archive(candidate).exact_reemit() == candidate` for both starts.
- Four exact realized rows per start are preserved; arbitration states its
  INSTANCE/FORMULATION/FAMILY/PARADIGM scope.
- Three consecutive clean adversarial passes follow the final fix.
- Findings memo, DAG feed, equation registry row/callable receipt, directive
  table, manifest, and branch commit(s) are ready for MAIN review.

## Do not touch

- No main-repository or sibling-worktree mutations.
- No campaign fire or unbounded run.
- No manual JSON ticket edit; the typed resealer owns ticket mutation.
- No alternate/forked resealer or launcher.
- No contest-score or pointer claim.
