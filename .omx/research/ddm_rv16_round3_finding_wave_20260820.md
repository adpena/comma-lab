# ddm_rv16 — round 3 recursive adversarial review: findings, cures, and exact boundaries

Date: 2026-08-20  
Owner: ddm_rv16  
Axis: `[source/receipt review + scorer-free apparatus controls]`  
Score claim: false  
Pointer moved: false

## Answer first

Round 3 is a **FINDING round**. Fresh review found five publication/object-scope defects in MAIN's
four landing targets and four apparatus/test defects in the live Python batch. All confirmed defects
were cured and verified in the working tree; none was refused on its merits. The governed commit was
**REFUSED BEFORE STAGING** because this sandbox cannot create temporary Git object/index files, so
no landing is claimed. The clean-pass counter nevertheless **resets to 0** on the findings.

The prior-law prediction held: fresh eyes found at least two additional defects in MAIN's landings
and at least one in the uncommitted Python surface. The central failure mode was **transitive proof**:
component evidence, a line-disjoint merge, or one instrumented timing instance was allowed to read as
proof of a clean composed shipping object. The component facts remain useful; the shipping verdicts
do not follow until that exact object executes.

The charter named 25 modified/untracked Python files. The same bounded command on the review
worktree returned **29**, because the concurrent gt2 landing added four Python surfaces after the
charter denominator was written. This review covered all 29; it does not silently preserve the stale
25-file denominator.

## Per-finding adjudication

| ID | Finding | Re-derived-at-source verdict and narrow scope | Severity | Cure/disposition | Executed control |
|---|---|---|---|---|---|
| RV16-F1 | rr8 called one instrumented T4 timing row a stable cleared shipping wall and said “the port ships.” No repeat or clean-tree row exists. | **CONFIRMED.** `verdict_scope: the stable/clean-shipping claim only`; the exact n600 instrumented observation, 464.558564563 s inflate, remains measured. | HIGH | Narrowed the rr8 memo and `main_hot_state.md` to an observed-instance pass; named repeat stability and clean composed execution as owed. | Receipt/store census found one rr8 T4 wall-clock row; the memo itself says the instrumented tree is not the shipping tree. |
| RV16-F2 | rr8 treated the 1.2855 neural-render ratio from one stage on one run as a host-variance estimator and pessimistic risk multiplier. | **CONFIRMED.** `verdict_scope: variance/bound interpretation`; the two measured stage times and their ratio remain valid point observations. | MEDIUM | Relabeled it a one-sample run/host point estimate and sensitivity calculation, not a variance bound. Removed it as shipping support in the memo and hot state. | Re-derived the ratio from the two named stage values and found no repeat population, variance model, or confidence bound. |
| RV16-F3 | rr8 said Python fallback would report `None`, and let the receipt field alone read as every-pair activation proof. | **CONFIRMED.** `verdict_scope: explanation/proof surface`; native activation on the executed row is still proved by receipt plus source. | MEDIUM | Corrected fallback to `FreeCorrector`. The memo now separates one-time selection from the hashed loop proof: the selected object is used unconditionally in all 600 frame and 190 group iterations. | Traced `_rr8_select_corrector` and the frame/group loops in the exact runtime source; compared the emitted `free_corrector="NativeFreeCorrector"` report. |
| RV16-F4 | rc1's title and answer called the 37-file composed tree “decode-proven,” although only the rider parse ran on that tree and port identity was inherited from a different tree. | **CONFIRMED.** `verdict_scope: full composed-object decode/shipping claim`; the 3-way merge and parsed `carrier_blob` restoration remain proved. | HIGH | Retitled/reframed rc1 as merge- and parse-proven. Marked S 0.14827847122030854 as conditional rate arithmetic and ~464 s as an unmeasured hypothesis until real composed execution. | Reconciled rc1's parse table with its own owed list: clean native compile, real `inflate.sh` smoke, seal, and authority row were not executed. |
| RV16-F5 | rr5's source memo still claimed 10/10 parsed parts matched, although rc1's erratum established 9/10 with `compressed_models` intentionally −169 B. | **CONFIRMED.** `verdict_scope: stale standalone proof wording`; standalone rider semantic identity and −169 B rate arithmetic remain supported. | MEDIUM | Corrected C3 at the source: 9/10 parsed fields match, the raw container is the intended delta, and restored `carrier_blob` is the semantic identity object. Explicitly excluded later port+rider composition. | Re-read the real receiver comparison and the rc1 erratum; the raw container differs while restored carrier semantics match. |
| RV16-F6 | F11's per-line cure still allowed an unrelated `.md` token on the same physical line to launder a bare harness task ID. | **CONFIRMED.** `verdict_scope: lexical recall advisory`; no historical queue row or task ownership verdict is changed. | MEDIUM | Each ID now needs an explicit linguistic association to a memo on that line (`#ID per memo.md` or `memo.md owns #ID`); mere co-location is refused. | Red control `Read unrelated_review.md and execute #1162...` now warns; the owning `#1074 per ddm_td1...md` control stays silent. |
| RV16-F7 | The GT-lineage guard skipped an entire line occupied by a docstring, hiding executable code after a same-line semicolon. Prose containing `tac.gt_lineage` could also clear a real read. | **CONFIRMED.** `verdict_scope: Catalog #351 GT-lineage scanner`; it does not regrade the already verified gt2 consumer bytes. | HIGH | AST-owned docstring spans are masked without removing later code; canonical routing now requires an executable AST import. The exact gl1 registry producer is explicitly excluded by role. | Same-line docstring+read and docstring/comment-only route fixtures both fail red before the cure and are caught after it. Live bounded census is 0 findings after declarations. |
| RV16-F8 | Certification appended and fsynced the ledger before invalidating the cache. An invalidation error returned refusal after silently committing a row beside stale cache state. | **CONFIRMED.** `verdict_scope: one certification call's failure atomicity`; no historical certification row is regraded. | HIGH | Cache invalidation now precedes the append; ledger/cache path aliasing is refused. Failure leaves the ledger untouched, while successful certification remains durable and invalidates its matching cache. | Directory-as-cache red control raises `AuditError`, preserves the cache target, and creates no ledger; success and alternate-ledger controls pass. |
| RV16-F9 | A recall-lint negative test depended on live corrections-index health and failed when the corpus correctly emitted global health advisories. | **CONFIRMED.** `verdict_scope: test isolation`; the health advisories themselves remain live and unchanged. | LOW | Isolated corpus-health output in tests that exercise charter-dependent legs; retained dedicated freshness tests for the global signal. | The first full focused run failed at this exact coupling; after isolation the full five-file suite passed 237/237. |

All cures above are present in the same reviewed working-tree batch. `tools/commit_autosha.sh` was
invoked with 39 explicit files, post-edit SHA pins, label `ddm_rv16`, and the required commit tags.
It refused before staging with `error: unable to create temporary file: Operation not permitted`;
HEAD remains `adee286defed` and the shared index remains empty. The exact reviewed source snapshot is
retained at `/Volumes/VertigoDataTier/pact/ddm_rv16/retained/DDM_RV16_REVIEWED_FILES_v3.tar.gz`;
this final receipt is retained separately beside it as `DDM_RV16_MEMO.md`. The adjacent JSON manifest
records both byte counts, SHA-256 values, member denominator, and the recovery boundary. The ignored
hot-state member is snapshot-only; no unrelated ledger or artifact path was included.

## Clean primary-target rows

- **Forced score control:** rr8 labels score identity as a forced control rather than a port result.
  No additional defect was found on that narrow wording after the correction.
- **rr5 standalone/composed boundary:** after F5's stale count is corrected, rr5 explicitly limits
  its controls and exact rate arithmetic to the standalone rider. It does not prove composition.
- **5c60d32af3 harvest:** MAIN's summary matches both arm memos. The commit landed six charter/memo/
  final-message documents and did not claim the uncommitted code was landed. gt2 says the live
  selector was already DALI-backed; rvf1 says five fixes were verified but Git-blocked.
- **F19 self-inference:** no defect found in the narrowed cure. Inference requires exact session
  identity, fresh `in_progress` status, complete file coverage, and exactly one candidate.

## F19 required proof

Three direct controls pin the intended boundary:

| checkpoint population | expected | observed |
|---|---|---|
| lone fresh file-covering checkpoint, **different session** | do not infer self | `None` |
| lone file-covering checkpoint, same session, **61 minutes stale** | do not infer self | `None` |
| unique fresh file-covering checkpoint, **same session** | infer current self | `CURRENT-SELF` |

The inference does not prove human/agent identity beyond the exact session token supplied by the
harness. `verdict_scope: serializer self-collision avoidance under the canonical session contract`.

## Mandatory axis 8 — assumption challenge

The shared assumption was: **proof transfers compositionally across nearby objects and instances**.
Its forms were “one T4 instance is a stable wall,” “one unaffected stage ratio bounds host risk,”
“two separately identity-proven transforms plus a line-disjoint merge prove their composed
receiver,” and “a parse proof stands for a full receiver execution.” Violating that assumption
changes the rr8 shipping and rc1 decode verdicts, but it does **not** erase the measured 464.5586 s
instance, the 3-way merge, the restored carrier identity, or the standalone −169 B rider result.

The same assumption appeared in apparatus form: same-line filename adjacency was treated as task
ownership, a documentation line as the entire executable line, and two state mutations as though
their ordering were failure-atomic. Executed counterexamples changed those apparatus verdicts and
required code cures.

## Mandatory axis 9 — what actually ran

| “works/ships” surface | Real execution | Modeled, inherited, or still absent | Verdict now |
|---|---|---|---|
| rr8 native port | Real `[contest-CUDA T4, n600]` instrumented tree: archive 180,625 B, inflate 464.558564563 s, evaluate 39.685 s, score 0.14839100138338618; native object selected and loop-wired. | No repeat. Clean port tree and clean composed tree unexecuted. Checkout/deps/download remain projected. | **Observed instrumented instance passes; stable clean shipping unproved.** |
| rc1 port × rider | Real 37-file merge; real receiver parse of the composed archive; restored `carrier_blob` identity. | Port bit identity inherited from rr6/rr8 on another tree; no composed native compile, full `inflate.sh`, frames, scorers, or wall-clock. | **Merge- and parse-proven only.** |
| rr5 rider | Real arithmetic round-trip, restoration, archive bytes, and standalone receiver parse. | No port+rider composition or exact evaluator row. | **Standalone semantic/rate result supported; composed result absent.** |
| 5c harvest | Real reads of both memos and the six-file commit. | The Python implementation was explicitly uncommitted at 5c. | **Summary accurate; not a code landing.** |
| rvf1/gt2 Python batch | Real synthetic red controls, live GT census, compilation, and 237 focused tests in this review. | No score, archive, Modal, or full-n600 scorer claim. | **Apparatus behavior proved at its declared surfaces.** |

## Verification

- Final expanded focused suite after the cures: **342 passed in 21.97 s** (the earlier five-file
  apparatus subset was 237/237).
- Additional live GT-lineage census: **0 findings** on the bounded live repository scope.
- `py_compile` passed for the edited implementation modules.
- `git diff --check` passed before review marking.
- Two new clean review-tracker passes were recorded for every Python file in the intended batch after the
  final edits. Earlier rvf1/gt2 marks were not counted as this round's two passes.
- No Modal dispatch, scorer forward, archive materialization, APDataStore rr8/rc1 write, protected
  runtime write, upstream write, or staged-index mutation occurred.

## RECALL EVIDENCE

Queries and stores searched beyond the charter's named commits:

- Full research corpus and receipts for `native corrector`, `free_corrector`, `decode identity`,
  `composed`, `host variance`, `checkpoint inference`, `cache invalidation`, and task-ID ownership;
- canonical equations via `tools/list_canonical_equations.py --json`, filtered for runtime,
  identity, checkpoint, and lineage laws;
- `CANONICAL_RESEARCH_INDEX*`, sub-0.15 DAG FEED blocks, canonical task status, harness bridge,
  `main_hot_state.md`, and the exact runtime source;
- the rr8 read-only T4 receipt, including its `PROJECTION` budget grade and unreported decode-path
  field; the exact runtime report and loop wiring supplied the narrower activation proof;
- rv15 and rvf1 source memos, serializer log, gt2 isolated patch manifest, and the live 29-file
  modified/untracked Python census.

Beyond the seeds, recall changed the plan in four ways: it exposed the receipt-field/fallback
explanation error; it found the same-line task-ID laundering counterexample; it found the
docstring/prose route fail-open in the new GT gate; and it showed the cache invalidation cure could
return refusal after committing a certification. It also established that current authority was
still publishing the rr8 shipping overclaim, so the hot state was corrected with the source memos.

## MAIN adjudication queue

No confirmed finding from this round is left uncured in the working tree. The Git landing itself is
queued because the canonical serializer was blocked before staging. Experiments and second-landing
promotion decisions stay explicitly non-authoritative until their fire triggers.

## NEXT_IF_RESUMED

- **QUEUED-WITH-OWNER** — disposition: land the exact reviewed rv16/rvf1/gt2 batch without absorbing unrelated worktree state; owner: MAIN/operator in a Git-writable checkout; consumer store: Git `main`; fire trigger: Git object/index writes are available, then compare current files to `/Volumes/VertigoDataTier/pact/ddm_rv16/retained/DDM_RV16_REVIEWED_FILES_MANIFEST.json`, rerun the 342-test suite, and invoke the SHA-pinned canonical serializer once.
- **QUEUED-WITH-A-FIRE-ORDER** — disposition: run recursive clean-review round 1; owner: MAIN review coordinator; consumer store: the next wave-end adversarial-review memo; fire trigger: this rv16 cure commit lands and its exact Python/content hashes remain current.
- **QUEUED-WITH-A-FIRE-ORDER** — disposition: execute and seal the clean composed port × rider object, then request one T4 authority row through MAIN's single-flight lane; owner: MAIN/rc1; consumer store: `/Volumes/APDataStore/pact/ddm_rc1/` plus the canonical frontier pointer if admitted; fire trigger: real local composed `inflate.sh` succeeds, decoded frames are byte-identical to the intended rider semantics, and the clean candidate seal passes.
- **DEFERRED-SECOND-LANDING** — disposition: consider promoting the GT-lineage host gate from warn-only to strict; owner: next custody-gate landing; consumer store: `src/tac/preflight.py` and its protection tests; fire trigger: this Python batch is committed and a fresh full-repository census still reports zero undeclared consumers.
- **QUEUED-WITH-OWNER** — disposition: classify the five SSD authored-signal rows measured by rvf1; owner: MAIN/rr8, MAIN/rr5, and gt2 custody respectively; consumer store: `.omx/research/ssd_authored_signal_certified.jsonl` or tracked source homes; fire trigger: byte-identity comparison proves generated copies or exposes unique authored deltas.

## LIVE-HYPOTHESES

- The clean composed port × rider receiver will preserve semantics and retain most of the observed
  native speedup. This is plausible because the mechanisms act at different stages, their source
  merge is line-disjoint, and the rider parse restores the carrier; it remains unproved because the
  exact composed `inflate.sh` has never run.
- The one-run host variation is unlikely to consume the full 1,295.8 s measured job-wall remainder.
  This is plausible from the large observed margin, but one unaffected stage is not a variance
  population; the clean composed row is the deciding observation.
- A content-resolvable task-ID-to-memo join can replace the remaining lexical association heuristic.
  This is plausible because the harness bridge, canonical task store, and research index already
  carry complementary keys, but no verified join currently spans all three.
- Certification/cache coordination may still deserve a shared lock if certification and a cache
  writer can run concurrently. The fixed failure ordering is safe for one call; the writer currently
  has no measured concurrent interleaving control.

## DEAD-ENDS

- Treating rr8's single instrumented T4 row as repeat-stable or clean-shipping proof is closed: no
  repeat or clean-tree execution exists.
- Treating the 1.2855 stage ratio as a variance bound is closed: it is one ratio from one pair of
  runs.
- Treating `free_corrector` alone as every-pair proof is closed: selection plus loop wiring is the
  required proof; Python fallback reports `FreeCorrector`, not `None`.
- Treating a conflict-free merge or composed parse as full receiver execution is closed: neither
  compiles the native path nor emits/scorers the frames.
- Repeating rr5's “10/10 parsed fields identical” wording is closed: 9/10 match and the tenth is the
  intended raw-container delta.
- Letting any `.md` token on the same line launder a bare task ID is closed by explicit association.
- Skipping a whole source line because it contains a docstring is closed by span masking.
- Appending certification before cache invalidation is closed because it can return refusal after a
  hidden durable state change.

**Own-vehicle frontier: S 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`, UNMOVED by
ddm_rv16. This round changed claims and apparatus, not archive bytes or score.**
