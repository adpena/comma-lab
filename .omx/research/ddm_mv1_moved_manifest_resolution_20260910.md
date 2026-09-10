# ddm_mv1 — moved sibling artifact resolution

Status: IMPLEMENTED_AND_TESTED_IN_ISOLATION; landing outcome is recorded in
`ddm_mv1_20260910/FINAL_HANDOFF.json`. No live source or live payload was changed.
`research_only=true` until MAIN installs the reviewed patch; `score_claim=false`;
`council_predicted_mission_contribution=apparatus_maintenance`.

## Measured result

| Surface | Result | Axis / boundary |
|---|---|---|
| Real rp1 moved raw | 3,662,409,600 B, SHA-256 `fd4b08e6aa0e967cc1453b9363c5f5ad243cb3bb57417f0f383cdb4bbfbfb2e8` verified | macOS-CPU, scorer-free custody |
| Original logical `file_fact` | Identical path, bytes and SHA after resolution | Actual rp1 raw; no fabricated replacement |
| Tests | 171 passed, 2 existing skips | File fixtures, seal document and VR3/VR5 regressions; no scorer |
| Implementation | 200 added / 12 removed production lines, plus tests | Nine Python files; five existing reader files |
| Catalog #417 final census | 176 candidate reads in 90 of 726 top-level `experiments/ddm_*.py` files; 177 before wiring | AST literal/Path-join/straight-line-alias scope; not a dynamic path census |
| Targeted reader census | jg2 0, sj1 0, rp1 0, tc1 0, tc2 0, tc3 0, vr3 0 | Counts within the same AST scope |
| Score, full public600 replay, SSD reclamation | Not measured / not run / zero bytes moved | No scorer, paid dispatch, or live tree mutation |

The before and after counts use the same final scanner and file snapshot. Earlier development
counts used different scanner scopes and are not comparable. The scoped gate count is nonzero,
so the orchestrator call remains **warn-only**, with explicit strict refusal available. A
zero count is required before the strict flip. The charter's 0–3 residual prediction is refuted
in this census; these 176 sites are migration candidates, not 176 independently confirmed
runtime failures. The five reader-file groups satisfy the predicted <=6 file-group bound,
but the patch changes 13 existing reader functions, so <=6 individual call sites was too small.

## What changes

`tac.artifact_moved.resolve` implements the actual rp1 file certificate schema: `moved_to`,
`bytes`, `sha256`, `reason`, `rebuildable_from`. Existing paths win. Missing paths follow at
most eight certificates. Every hop must agree with the terminal file's bytes and SHA. Bad
schema, missing destination, cycle/depth exhaustion, byte drift, SHA drift or hash-time
identity drift raises `MovedArtifactError`. Destination paths must be absolute.

SHA verification is lazy on first resolution. An in-process cache binds the verified digest
to path/device/inode/size/mtime_ns/ctime_ns; optional receipt output exposes that fingerprint,
SHA, manifest paths and whether verification reused the receipt. The helper writes no cache
into another owner's live tree. This assumes the owner's retained files are quiescent; a
same-stat mutation invisible to the filesystem cannot be detected by a stat-keyed cache.
There is no persistent cross-process cache or directory-manifest schema.

`move_with_manifest` uses exclusive destination creation, streams the copy, flushes/fsyncs it,
checks source stability and full source/destination SHA, then uses the existing
`tac.micro_edit.ledger.atomic_write_json` before unlinking the source. The source survives
copy/checksum/publication failures. Existing destinations and certificates are refused.
Interrupted destination copies remain for inspection, rather than being deleted without
custody. Owner quiescence, source ownership and space admission remain caller prerequisites;
this helper is not a live-job scheduler or a general storage sweeper. Only temporary fixtures
executed this writer in this arm.

jg2's `file_fact` keeps the original logical path in its dictionary while reading the resolved
file. Its hash, token load and edit-array readers also resolve. This reaches sj1 and tc pin
checks through their existing jg2 calls, without touching the live, untracked tc3 sources.
rp1 field hash/load and sj1 NPZ conversion are wired explicitly. Candidate-seal retained checks
resolve and translate refusals into the existing missing-custody verdict. The seal document,
seal digest, runtime digest definitions and receiver digest functions are unchanged.

VR3 pin verification returns the physical destination while comparing the unchanged logical
pin. Both planners emit `MOVED_CERTIFIED` rows carrying the verified receipt and zero newly
freed bytes; they never mark these rows `DELETABLE`. A move discovered after planning blocks
apply as `MOVED_CERTIFIED_RETAIN_ELSEWHERE`, rather than `RAW_MISSING_OR_NOT_REGULAR`. Mixed
legacy/retained apply ledgers retain their existing fail-closed refusal; this patch does not
expand deletion authority. An all-moved plan emits its certificate rows and no deletion set.

## Validation and review

Evidence store: `.omx/research/ddm_mv1_20260910/`.
- `final_real_resolution.json`: final-source hashes, two streamed file verifications and a cached
  repeat on the actual incident payload. `real_resolution.json` preserves the earlier check.
- `pytest.log`: 171 passed, 2 skipped; new controls cover chain, malformed schema, same-size drift,
  timestamp restoration, writer publication ordering/failures, source retention, gate positive /
  negative / waiver, unchanged logical pins, unchanged seal digest, both reclaim timing cases.
- `ruff.json`: checked with canonical `--stdin-filename` paths so repository per-file policies
  apply. Zero new diagnostics; sj1's existing diagnostics are preserved, not mechanically edited.
- `review_pass1.json`, `review_pass2.json`, and `review_tracker_pass*.log`: two visible self-review
  passes over final source hashes, marked through the actual isolated `tools/review_tracker.py`.
  Unchanged surrounding definitions were checked for byte identity, not reclassified as new work.
- `census.json`: complete scoped hit list, per-reader counts and denominator.
- `landing.patch`, `landing_manifest.json`, serializer invocation/status and final handoff: exact
  installation unit and remaining disposition. No assertion that a fallback artifact is landed.

Shared assumption challenged: a pathname need not be the permanent physical location of an
artifact. Separating logical identity from verified location repairs the incident without
changing measured digests. A manifest is custody evidence, not permission to delete its target.

## RECALL EVIDENCE

Searched the memory registry for `moved.manifest|sibling.artifact|ddm_mv1`; no hits in that scope.
Searched `src/tac`, `experiments`, `tools` for `MOVED.json`, then research memos and arm receipts
for `MOVED.json|moved_to`, and docs, SPEC/index/DAG and task-ledger surfaces for the same exact
terms plus `artifact.*resolv`. Queried `tools/list_canonical_equations.py --json`; matching recall
is retained in `equations_recall.json`. No new mathematical equation is asserted: `[no-triality]`.

Beyond charter seeds: tc3's untracked `ddm_tc3_recover_public_receipt.py` already performs a
one-off rp1 schema/hash check. It confirms the reference schema but is not a reusable canonical
resolver; preserving that live source avoided checkpoint-binding drift. The older sr2 memo
(`ddm_sr2_vertigo_space_reclaim_20260811.md`) uses whole-directory manifests and a symlink,
while sc3 (`ddm_sc3_storage_custody_move_20260813.md`) records related custody routing. Those
are different schemas: they were not silently accepted as rp1 file certificates. No existing
reusable rp1-schema helper was found in the searched Python scope. Recall changed the plan to
reuse jg2 callers, leave tc3's recovery source untouched, and reuse the existing atomic JSON
writer instead of introducing a second publication primitive.

## Catalog #417 retrospective and six integration hooks

The catalog number was atomically claimed at 2026-09-10T03:34:31Z (counter now 418).
Catalog #417 is warn-only, strict-eligible, with same-line `MOVED_READ_OK:<specific explanation>`;
placeholder and preceding-line waivers do not pass. It scans literal paths, Path joins and
straight-line local aliases, not arbitrary control-flow or interprocedural path provenance.
The code's presence alone is not proof that every historical read is now resolved.

Retrospective scope: the content searches above plus all 726 current top-level ddm scripts.
Confirmed incident: tc3's old-path `file_fact` failed after valid rp1 cold storage. Re-evaluation
priority is tc3's next normally authorized public stage, after it rebinds to the current source.
Older directory custody precedents are retained historical evidence, not new missing-file
failures. No score KILL/DEFER verdict was reversed by this apparatus work.

Sensitivity-map, Pareto, bit-allocator and cathedral-autopilot hooks: N/A, because no score,
allocation, payload format or dispatch policy changes. Continual-learning hook: the canonical
probe/task receipts consume the real resolution and nonzero census. Probe-disambiguator:
ACTIVE in the behavioral controls separating moved, absent, drifted and deletion-admissible
objects. Catalog provenance: apparatus only, score_claim=false, promotable=false. Lane:
`ddm_mv1_moved_manifest_resolution_20260910`, registered L0; no strict-maturity claim.

## Live-arm sequencing and handoff

Live consumers named before preparing the patch: **rp1 round 2**, **tc3**, **vr6**; sj1 generation
3 was finishing its memo. Shared jg2 is imported by rp1/sj1/tc; VR3 is consumed by vr6; the seal
validator is shared by receiver/seal consumers. None of those source files was modified in the
live checkout. The patch lists helper/gate/tests first and live-reader files last. MAIN must
install the reader changes only after affected stages have exited and explicitly bind the next
stage to the new producer hashes. Do not continue a checkpoint against changed producer code
by relabeling its old digest. The shared staged index and upstream were untouched.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER — MAIN: install the exact reviewed patch/bundle, live reader files last;
  consumer store `ddm_mv1_20260910/FINAL_HANDOFF.json`; fire when rp1/tc3/vr6 affected stages are
  quiescent and the serializer can update main. Preserve/recheck every post-edit source hash.
- QUEUED-WITH-A-FIRE-ORDER — MAIN routes the reader-migration arm: classify and resolve or
  substantively waive the 176 scoped candidates; consumer store `ddm_mv1_20260910/census.json`;
  fire after this patch is installed, then strict-flip #417 in the batch that measures zero.
- QUEUED-WITH-A-FIRE-ORDER — tc3 via MAIN: consume the helper in the next normal public600
  receiver stage, with freshly bound producer/source hashes; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/`; fire at the next source rebind
  and stage boundary. The old failing file-fact operation is already verified on real bytes.

## LIVE-HYPOTHESES

The remaining historical readers can use the same logical-path/physical-read split: the real
rp1 receipt and unchanged seal control demonstrate it without changing identity definitions.
The next complete tc3 public600 stage should pass the old missing-path site, since that exact
file-fact operation passed against the real raw; a full stage replay was not performed here.

## DEAD-ENDS

No symlink workaround: it does not implement the required manifest mechanism on ExFAT.
No path rewrite inside sealed facts: it changes logical receipt identity unnecessarily.
No immediate strict flip: 176 scoped candidates remain; zero was not measured.
No relabeling moved bytes as absent or deletable: the verified destination remains retained.

OWN-VEHICLE FRONTIER UNMOVED: S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600].
