# ddm_cpd1 — journaled frontier disqualification

Tokens: `[no-triality] [p0-ledger-ok]`. Apparatus only; `research_only=true`, `score_claim=false`.
# FORMALIZATION_PENDING: eligibility and custody apparatus; no new scoring equation or empirical codec law

The disqualification mechanism is implemented and tested. The live pointer is unchanged; MAIN alone
performs the actual disqualification. The canonical journal is intentionally empty. Landing status,
bundle refs and hashes are recorded in `.omx/research/ddm_cpd1_20260910/LANDING_HANDOFF.json`.

## Measured result and boundaries

All apparatus counts below are MEASURED [macOS-CPU static apparatus / scorer-free fixtures].
No training, scorer, encode, decode, paid dispatch, upstream mutation, or new contest row ran.
Historical anchor mirror files were not edited. Existing archive bytes and receiver contents were
not revalidated by this task; the compliance adjudication is the cited pr8 review and gs3 Addendum 22.

- New regression suite: 53 fixture cases covering selection, append-prefix preservation, reinstatement,
  reason classes, malformed journals, placeholder refusal, both axes, preflight positive/negative/waiver,
  packet refusal, identity aliases, concurrent decisions, writer revalidation and dispatch auto-refresh.
- Focused sister suites and quota tests: final command, counts, wall time and output in `checks.json`
  and `check_1.log` under the receipt directory. Ruff covers all nine changed Python files.
- Offline rehearsal used exact copied move-40/move-41 mirror bytes plus a copy of the live pointer.
  Before refresh the disqualification preflight refuses the copied stale pointer; afterward it passes
  and both local selection and effective selection choose move 40. The original mirrors and live
  pointer retain their hashes. Source hashes and arithmetic are in `offline_rehearsal.json`.
- Full live-source scan: 459 loaded anchors, 2 records excluded by the fixture decision; the resulting
  CUDA minimum is move 40. This is an offline eligibility projection, not a live journal mutation.
- Catalog #418 claimed. Current live journal has 0 bytes and no active decisions. Live audit and
  quota audit return zero; the former is not evidence that MAIN has completed its required action.

The charter's line prediction holds for its four named files: 97 added lines, 3 removed lines across
pointer module, refresh CLI, packet CLI and the new operator CLI. The necessary shared journal helper
adds 144 lines and scanner integration adds 11; these are disclosed separately rather than hidden
inside the four-file count. The all-source filter is required before top-five truncation. Catalog
#343's dispatch auto-refresh API remains intact and its actual hook is fixture-tested.

## Mechanism

`.omx/state/frontier_disqualifications.jsonl` is the sole mutation. Each event carries lane ID,
archive SHA-256, one of four reason classes, evidence path and SHA-256, rationale, actor, timestamp,
action, and reinstatement fields. Appends flush and fsync under locks; malformed or unterminated
history refuses replay/append. Reinstatement appends a new event and preserves all prior bytes.
This protects application writes; it is not tamper-proof storage against external manual deletion.

Identity is the pair `(lane_id, archive_sha256)`: the same archive can have distinct receivers.
If a legacy record lacks an identity field, match conservatively using what is present. A packet
checks the recorded receipt lane independently of a command-line lane override. Missing recorded
lane may conservatively refuse an archive even when the caller supplies another lane; restoring
trusted receipt identity is the remedy, not treating an override as proof.

Every scanner source receives an in-memory `extra.disqualified` projection before ranking. Mirror
bytes remain unchanged. The pointer excludes vetoed current/prior anchors on both axes, preserves
upstream competition, and exposes active decisions in `disqualified_rows` and `refresh_provenance`.
The human refresh summary prints each reason, rationale and evidence. Recompute also consumes the
stored audit list. Before writing, the pointer writer checks the current journal under the same
canonical lock used by journal appends. A decision racing a scan can refuse that refresh; dispatch
outcome persistence remains intact under the existing auto-refresh failure contract.

Packets hold a shared eligibility lock for all stages; journal appends take its exclusive lock
before the canonical pointer lock. This prevents eligibility changes midway through publication
without deadlocking the refresh subprocess. A later disqualification does not rewrite historical
packet events; MAIN then refreshes the pointer. No claim is made that this journal discovers
compliance violations: the operator's evidence-backed decisions are its input.

Catalog #418 checks effective and both local pointer rows, including an upstream-winning pointer
whose local anchor remains disqualified. A fake upstream source label cannot hide local identity.
A substantive JSON `frontier_disqualification_waiver` waives only the audit, never packet or pointer
publication; malformed journal data cannot be waived. The callsite is WARN-ONLY until MAIN completes
the charter's live decision and zero-census condition. The existing file-level quota waiver passes
#299; no live protection is retired. This umbrella covers local anchors, effective pointer and
malformed decision history rather than introducing three gates.

## MAIN fire order

Run from `/Users/adpena/Projects/pact` after landing the implementation and protection batches:

```sh
.venv/bin/python tools/frontier_disqualify.py \
  --lane ddm_tc3_t4_lane_predictor_tail_20260910 \
  --archive-sha256 299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936 \
  --reason-class rule118_content_in_code \
  --evidence .omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md \
  --who MAIN \
  --rationale 'pr8 traces Lane class 1 and rows 128..319 to video-selected constants embedded in free receiver code; gs3 Addendum 22 retracts move 41.'
.venv/bin/python tools/refresh_canonical_frontier.py --no-update-upstream
.venv/bin/python -c 'from tac.preflight import check_frontier_excludes_disqualified_rows; check_frontier_excludes_disqualified_rows(strict=True, verbose=True)'
```

Verify the selected lane is `ddm_sj1_t4_compose39_rp1_union_20260910`, with the move-41 reason surfaced.
Then flip #418's `preflight_all()` call to `strict=True`, update its catalog row, and land the live
journal, refreshed pointer and strict flip together. Reinstatement uses the same lane/SHA with
`--reinstate --rationale '...substantive new evidence...'`; `--evidence` may point to the new review.
The API does not infer that a newly rebuilt archive repairs an older lane/archive pair.

## RECALL EVIDENCE

- Charter and full common contract read, then PROGRAM, CLAUDE/AGENTS (identical), operating manual,
  live board, lane registry and task stores. Memory registry search terms `pointer.disqual`, `cpd1`,
  `common.contract` returned no relevant entry; no unverified remembered score used.
- Full `.omx/research/` content search: `disqualif|pointer.*qualif|qualif.*pointer`; targeted pr8,
  gs3 Addendum 22, pm2 and pointer-shadowing receipts reviewed. Beyond charter seeds, rv13's
  same-archive/different-runtime custody finding in `frontier_scan.py` ruled out SHA-only identity.
  The September 5 ranked-candidate repair and its behavioral test exposed the top-five shadowing
  boundary; this changed the plan to filter all scanner inputs before truncation.
- Canonical equations CLI JSON filtered for `canonical_frontier_pointer|disqualification`; six
  relevant records retained in `equations_recall.json`. The existing pointer law and wrong-baseline
  substitution law support retaining official competition separately from local custody. No new
  scoring law was registered for eligibility apparatus.
- Research index/DAG and docs/SPEC surfaces searched by the same terms. Hits on historical
  supersampling and design disqualification did not supply a journal mechanism; no additional
  mechanism found in this scope. Canonical task store had no cpd1 row; operator P0 ledger explicitly
  routed this mechanism to MAIN. Retained scoped extracts are in the receipt directory.
- Source review found a pre-existing training-target test edited only the first live leaderboard
  entry. A newly present runner-up then beat its expected test minimum. The failure reproduced
  using the unchanged HEAD pointer module; the test now controls the whole ranked table.

## Review and integration

Two final visible independent review passes cover all changed implementation surfaces; root also
independently reviewed the new tests. `reviews.json` pins exact post-edit hashes. Preflight review
is explicitly limited to the new function/callsite, not all unrelated code in that module.
`review_tracker.log` records both marks per Python file; no review override is used.
Fixed review findings: lane override bypass; packet/journal publication race; overbroad archive-only
refusal of a separately identified receiver; upstream-label bypass; audit-only recompute omission.

Apparatus lane: `ddm_cpd1_pointer_disqualification_20260910`. Canonical task rows track implementation,
MAIN's live decision, and strict flip. The training-fixture repair is FOLDED into this batch.
The six solver hooks (sensitivity, Pareto, bit allocator, candidate dispatch, empirical-score posterior,
research probe disambiguator) are N/A: this work neither changes a codec nor measures a score.
Production consumers are scanner, pointer refresh/write/recompute, packet CLI and preflight.
`council_predicted_mission_contribution=frontier_protecting`; no research efficacy claim.

The fix and self-protection are separate serializer intentions, using post-edit hashes and no
attribution trailer. Shared staged index and unrelated dirty work are preserved. Shared lane/task
metadata contains sister work; only this arm's own metadata patch is retained for MAIN to reconcile,
never whole-file-staged. Exact landing outcome and any Git-object-denial bundle refs are in the handoff.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer `.omx/research/ddm_cpd1_20260910/LANDING_HANDOFF.json`; fire on a writable Git object store if serializer fallback is used. Land both exact intention bundles and reconcile only cpd1 metadata hunks; rerun affected checks if resolution changes code.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer `.omx/state/frontier_disqualifications.jsonl` and `.omx/state/canonical_frontier_pointer.json`, task `ddm_cpd1::main_disqualify`; fire at implementation/protection harvest. Run the exact command above, refresh, and verify move 40 and the surfaced reason.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer `src/tac/preflight.py` and `docs/meta_bug_class_catalog.md`, task `ddm_cpd1::strict_flip`; fire immediately after that live refresh passes strict audit with zero violations. Flip and land in the same batch as the live decision.

**LIVE-HYPOTHESES:** The live MAIN transition should match the rehearsal because it consumes the same
retained anchors and journal predicate. This remains unexecuted on live state by design. No untested
codec or score-improvement hypothesis is introduced.

**DEAD-ENDS:** Filtering after top-five truncation can hide valid successors. Reusing a vetoed prior
restores a disqualified row. SHA-only vetoes conflate distinct receivers. A lane override cannot prove
new custody. An initial packet check alone permits concurrent invalidation. These implementation
paths are closed by the new regression suite; no research family is declared dead.

OWN-VEHICLE SUBMITTABLE FRONTIER unchanged: **move 40, S 0.13763861019288715 @ 180,233 B
[contest-CUDA T4 n600]**, archive `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
<!-- HISTORICAL_SCORE_LITERAL_OK: retained move40 source mirrors and gs3 Addendum22 rederived in cpd1 offline_rehearsal.json; this apparatus task makes no new score -->
