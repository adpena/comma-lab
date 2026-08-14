# DDM DT1 repeated-lesson determinizer

Date: 2026-08-14  
Disposition: **THREE ELIGIBLE RECURRENCES DETERMINIZED; REMAINDER OWNED OR ROUTED**  
Axis: `[static apparatus census and scorer-free local tests]`  
`score_claim=false`; no Modal, Metal, scorer, archive build, or `upstream/evaluate.py` run occurred.

## Outcome

The instrument-driven census adjudicated eight recurring lesson genera. It did not turn detector
mentions into incidents: an incident required a distinct dated receipt, call ID, or retained failure
artifact. Existing strict cures and active owners were removed before choosing implementation work.
The top three eligible genera now have both an executable cure and a strict source gate:

1. a recursive worker-import closure is sealed against the exact target venv before dispatch;
2. the common Modal claim helper guards before it writes and identifies its own claim by exact lane,
   agent, and job;
3. terminal dual-ledger matching is call-ID-first whenever the active claim contains a call ID.

Endpoint closure and its EC2 reference emitter landed independently as AC1 commits `4f4537d835`,
`b4404f9fa3`, `8a3207e10e`, and `f204c8fcb6` while DT1 was running. The payload-retention migration
belongs to task #1001; prefix sampling already has a strict gate. None was duplicated here.

## Census

`Recurrence after first memory/gate` counts adjudicated incidents, not AU1 candidate rows. `>=` marks
a bounded lower bound. A detector population is shown separately when it is useful, but never counted
as that many real failures.

| genus | distinct dated incidents and receipts | current enforcement | recurrence after first memory/gate | measured cost and disposition |
|---|---|---|---:|---|
| Worker-chain dependency closure versus target venv | E4 Brotli, 2026-07-24 (`ddm_e4_brotli_declared_dep_DAG_FEED_20260724.md`); RR3 unconditional `constriction`, 2026-08-09 (`RR3_HPAC_SHIPPING_AUDIT.md`); EC2 r1 Pydantic `fc-01M006HS...`, 2026-08-14; EC2 r3 Brotli `fc-01M006Y0...`, 2026-08-14 | **GATED-STRICT** by this landing | 4 after Catalog #203 | EC2 paid for three failed integration attempts before r4; RR3's bare-venv receiver failed 1/1 in 0.03 s. **FIRED** here. |
| Automatic Modal endpoint closure | Four crash-loop closures plus one skipped bg1 payload harvest, 2026-08-14, enumerated in `charter_ddm_ac1_automatic_endpoint_closure_20260814.md` | **GATED-WARN** after the four AC1 commits; generic closer and EC2 emitter/arming landed | >=5 before the cure; 0 after it in the reviewed scope | About ten hand commands and one full arm cycle. Apparatus **FIRED** by `ddm_ac1`; first live provider-backed use remains queued. |
| Dispatcher preclaim collides with dispatcher-owned claim | EC2 2026-08-14 (`active_lane_dispatch_claims.md` row 22); RE1T 2026-08-14 (row 40); QS1 2026-08-13 (row 58); SA1 2026-08-13 (row 264) | **GATED-STRICT** by this landing | >=4 after Catalog #513 | No provider spend on these refusals, but each required hand closure and retry. **FIRED** here. |
| Git-blocked arm memo handoff | RR1 2026-08-09 (`arm_final_messages/ddm_rr1_20260809T151025Z.md`); SR2 2026-08-12 (`arm_final_messages/ddm_sr2_vertigo_space_reclaim_20260812T002121Z.md`) | **UNENCODED** for the non-Modal keeper leg | >=2 after the Git-custody standing discipline | Two hand commits remained owed; RR3 additionally needed an isolated fallback clone. **QUEUED-WITH-A-FIRE-ORDER** to the existing keeper surface. |
| Prefix/subset sampling substitutes a different population | m88/m96 evidence consolidated by NA2 on 2026-08-03; 110 tools used `[:n]`; pose prefixes were 2.54-4.21x harder | **GATED-STRICT** by the canonical subset-selection gate and commit hook | 0 found in the reviewed post-gate scope | A repository backlog, not 110 separate incidents. **FOLDED** into the existing strict selector; no new cure. |
| Shared-job-ID dual-ledger terminal matching | EC2 r1, r2, and r3 terminal call IDs each falsely matched the active r4 claim on 2026-08-14 (`poller_run/run.log` lines 1-3); line 4 was the legitimate r4 blocker | **GATED-STRICT** by this landing | 3 after Catalog #513 | Three false blockers and hand adjudications. **FIRED** here. |
| Materialized payload reduced to scalars and discarded | PR130 Range/ANS n600 race, 2026-08-09; SD2 missing candidate argmax payload, 2026-08-09 | **GATED-WARN** in full preflight only | 0 newly adjudicated after the warning gate; live candidates remain untriaged | First incident spent 681 s, forced two full re-encodes, and delayed a measured 2,120 B rate win; SD2 lost an arm cycle. Current scorer-free static population is 1,076 findings / 517 files with findings / 6,504 `.py` examined / 1,365 AST candidates / 0 unreadable. **QUEUED-WITH-A-FIRE-ORDER** to task #1001. |
| Headline survives after its body has narrowed or corrected it | gk2/TZ1 vehicle conflation, 2026-08-04: 23,655 B versus live IX2 24,605 B; MP2 sandbox limitation promoted to “stack cannot train on Metal,” corrected 2026-08-09 in `probe_outcomes.jsonl` | **MEMORY-ONLY** as enforcement; AU1 is a coarse detector, not a refusal | >=1 after AU1 landed | Authority and routing errors rather than a priced provider run. **QUEUED-WITH-A-FIRE-ORDER** into NB1-R2 body adjudication. |

The prefix row's 110 sites and the payload row's 1,076 findings are detector/site populations, not
incident counts. AU1's 11,840 correction candidates and 8,157 headline/body candidates receive the
same treatment.

## Ranking and eligibility

The gross recurrence-cost ordering is endpoint closure, payload discard, dependency closure, claim
ordering, dual-ledger false matching, git handoff, stale headlines, then already-strict prefix
sampling. That ordering uses observed commands, seconds, retries, and arm cycles; it does not invent a
dollar conversion between them.

The charter also forbids duplicate cures. Removing `ddm_ac1`'s active endpoint ownership, task
#1001's payload migration, and the already-strict prefix selector leaves this implementation order:

| eligible rank | recurrence x cost / cure size rationale | built result |
|---:|---|---|
| 1 | Four distinct dependency failures, including three paid EC2 integration attempts; medium reusable cure | Recursive static closure plus exact target-venv seal and image provisioning |
| 2 | At least four dispatcher refusals; one hand closure/retry each; small common-helper cure | Agent-precise guard before dispatcher claim write |
| 3 | Three false blockers in one poller sequence; tiny matcher cure | Exact call-ID match when a claim names any call ID; lane/label fallback only for legacy rows |

Git handoff and headline adjudication remain real work, but their cure surfaces are larger than the
third-ranked call-ID correction and they have existing keeper/NB1 consumers.

## Deterministic cures

### 1. Worker-to-target-venv dependency closure

`src/tac/deploy/worker_dependency_closure.py` recursively parses the import-time module closure
reachable from worker entry points, including module-scope guarded, relative, and imported-submodule
paths. Deferred function-body imports are excluded because the sealed worker can contain optional
packaging paths it never calls; the receipt states that selection mode explicitly. The helper
separates stdlib, repository-local, retained-payload, and third-party roots; unresolved
repository-local imports also fail the closure. The deterministic receipt records every visited
source path, bytes, SHA-256, target lock identity/package inventory, available roots, missing roots,
and selection mode.

`experiments/ddm_ec2_modal_oriented_adapter_trainer.py::prepare` now seals that receipt against
`upstream/uv.lock`. The exact pinned tuple `pydantic==2.13.4`, `Brotli==1.2.0` is consumed both by the
seal and by the target venv's image-build provisioning. Runtime installation after GPU dispatch was
removed. On the real EC2 import-time worker chain the control without the tuple finds exactly
`brotli` and `pydantic` missing across 15 recursively visited local source files; the sealed tuple
closes both.

The strict gate `check_worker_target_venv_dependency_closure_is_sealed` refuses removal of the seal,
drift between the sealed tuple and image provisioning, or a return to runtime self-install.

### 2. Claim ordering

`claim_modal_auth_eval_dispatch` now calls the common single-flight guard before
`record_dispatch_claim`. It passes the dispatcher's agent and job identities. A same-lane claim is
therefore excluded only when lane, agent, and job all match; a manual, stale-job, or external
same-lane preclaim is a conflict. Legacy immediate pre-spawn rechecks that omit an agent retain
lane-only self-exclusion.

The strict gate `check_modal_dispatch_claim_guard_precedes_write` parses the helper and refuses any
state without exactly one guard before exactly one write, or without the `claim_agent` and `label`
discriminators.

### 3. Call-ID-first dual-ledger matching

`dual_ledger_terminality_blockers` extracts Modal call IDs from each active claim. If a claim names a
call ID, only that exact terminal call ID can match; label/lane fallback applies only to legacy rows
with no call ID. Thus terminal r1-r3 cannot match live r4 merely because all four share one lane/job
label, while terminal r4 still blocks until its own active claim closes.

The strict gate `check_modal_dual_ledger_matching_is_call_id_first` refuses loss of the call-ID
extraction, exact membership comparison, or guarded legacy fallback.

## Typed worklist

| disposition | cure | owner | consumer store | fire trigger |
|---|---|---|---|---|
| **FIRED** | Worker closure, target-venv seal, and strict gate | DT1 landing | EC2 sealed request, `WORKER_DEPENDENCY_CLOSURE.json`, full preflight | Every EC2 `prepare` before dispatch |
| **FIRED** | Guard-before-write with agent-precise self-claim | DT1 landing | `tac.deploy.modal.auth_eval` claim helper | Every common Modal claim transition |
| **FIRED** | Call-ID-first terminal matcher and strict gate | DT1 landing | modal call ledger terminal update and claims ledger | Every terminal call-ledger write |
| **FIRED** | Generic automatic endpoint close, payload harvest, memo handoff, and both-ledger closure | `ddm_ac1`, commit `4f4537d835` | `tools/modal_endpoint_close.py`, endpoint receipts, Modal ledgers | Landed with 131 focused tests and retained dry-run receipt |
| **QUEUED-WITH-A-FIRE-ORDER** | Exercise the landed closer on one live endpoint | next data-bearing Modal dispatcher owner | EC2 closure manifest, named SSD `endpoint_closure/`, and retained payload store | Next data-bearing Modal spawn after the global single-flight guard is clear |
| **QUEUED-WITH-A-FIRE-ORDER** | Teach the keeper to serializer-commit a declared path/SHA when Git custody is available | keeper maintainer after AC1 manifest schema lands | `tools/codex_arm_queue.py`, arm-final-message index, Git history | Next non-Modal final message declaring a Git-blocked memo path and SHA |
| **FOLDED** | Canonical representative subset selection and positive-control gate | subset-selector maintainer | commit hook and `src/tac/subset_selection_gate.py` | Reopen only on a mutation-test or positive-control failure |
| **QUEUED-WITH-A-FIRE-ORDER** | Adjudicate and retain the remaining measure-and-discard candidates; move gate into normal hook before strict flip | task #1001 successor | `.omx/state/operator_p0_ledger.jsonl` | Current source owners land, then work the executable-value order already recorded by PR1 |
| **QUEUED-WITH-A-FIRE-ORDER** | Require body adjudication before an AU1 headline candidate becomes a task/status fact | `ddm_nb1_r2` | AU1 candidate JSONL and canonical task-status body | Next AU1 candidate promotion into a canonical status row |
| **QUEUED-WITH-A-FIRE-ORDER** | Implement the recurring genus adapter described below | costate apparatus maintainer | canonical task-status/costate duty queue | Two deduplicated receipts share a registered genus ID and no strict cure or active owner exists |

## Recurring-census design

Do not add another repository scanner. Add one normalizing adapter after the existing instruments:

1. Consume AU1 correction/headline candidates, the canonical anti-pattern registry, preflight catalog
   inventory, arm-final-message receipts, Modal call ledger, and claims ledger.
2. Require a typed occurrence identity `(genus_id, receipt_path, receipt_anchor)`, where the anchor is
   a call ID, retained artifact SHA, or exact dated receipt key. Mentions of one receipt collapse to
   one occurrence.
3. Join the catalog and current owner stores before emission. Classify the genus as
   `GATED-STRICT`, `GATED-WARN`, `MEMORY-ONLY`, or `UNENCODED`; suppress a new build row when a strict
   cure or active owner already exists.
4. At two distinct occurrence identities, append one typed determinization-candidate duty to the
   existing canonical task-status/costate queue with recurrence, measured-cost fields, owner,
   consumer store, and fire trigger. Never let a coarse AU1 candidate block by itself.
5. Positive controls: the four dependency receipts become one four-incident genus; repeated mentions
   of one EC2 call remain one incident; 110 `[:n]` sites do not become 110 incidents; r1-r3 remain
   distinct because their call IDs differ.

This preserves the existing costate consumer and makes “learned twice” executable without creating a
parallel registry.

## RECALL EVIDENCE

The full-corpus recall used the charter's instruments, then extended beyond its seeds:

- AU1 receipt plus `au1_corrections_index.jsonl` and `au1_headline_vs_body.jsonl`; the receipt proves
  11,840 correction candidates from 7,592 memos and 8,157 headline/body candidates, and explicitly
  labels both outputs coarse rather than adjudicated;
- `.venv/bin/python tools/list_canonical_equations.py --json`: 435 equations; searches for
  `recurr|lesson|dependency|single.flight|claim|subset|prefix|headline|payload` found sister invariants
  but no existing recurring-genus emitter to reuse;
- `.omx/state/canonical_anti_patterns_registry.jsonl`: 114 rows, comprising 85 registrations and 29
  falsification rows; used as genus taxonomy, not as an occurrence counter;
- `docs/meta_bug_class_catalog.md`, especially #203 and #513. #203 checks hard dependencies only in
  `experiments/modal_train_lane.py`, not an arbitrary worker chain against its target interpreter.
  #513 proves single-flight presence/state but did not define agent-precise claim ordering or
  call-ID-first terminal matching;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, task/P0 ledgers, `main_hot_state.md`, design/SPEC docs,
  and `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` poison/standing
  sections, searched with `dependency closure|Brotli|constriction|preclaim|endpoint closure|git blocked|prefix|headline|payload retention|dual ledger`;
- dated receipts and stores: E4, RR3, PR1, TZ1, MP2 probe-outcome correction, arm-final-message files,
  claims ledger, call-ID ledger, and the retained EC2 poller log.

Findings beyond the charter seeds changed the plan. The payload-discard genus added an expensive
second owner-routed class; the stale-headline genus added a post-AU1 recurrence; SA1 added a fourth
current claim-order incident; older claims rows showed the genus predates this session; PR1 proved its
gate is warning-only and absent from normal commits; and the prefix class was already strict. Those
facts removed endpoint, payload, and prefix work from the build queue instead of re-curing them. No
existing strict gate covered any of the three implemented semantics.

## Validation and boundaries

- Incident-shaped focused suite plus existing EC2/single-flight coverage: 56 passed.
- Existing auth-eval, claim-dispatch, and ledger compatibility coverage: 96 passed, 1 deselected.
  The deselected test creates a Unix-domain socket and independently failed under the managed
  sandbox with `PermissionError: [Errno 1] Operation not permitted`; no product assertion failed.
- The real worker-closure negative control found exactly `['brotli', 'pydantic']`; the pinned image
  tuple passed. The import-time closure visited 15 local source files and had zero unresolved locals.
- Mutation-shaped controls made each new strict gate refuse its pre-cure source form.
- All three live strict gates and Catalog #176's strict-callsite inventory passed with zero findings.
- Ruff, CPython compilation, and `git diff --check` passed on all six changed Python files.
- Two review-tracker `mark-file --status reviewed` passes completed for every changed Python file;
  no review override was used.
- Scorer-free payload detector census: 1,076 findings / 517 files with findings / 6,504 `.py`
  examined / 1,365 parsed candidates / 0 unreadable.
- No full-video job, scorer, receiver, archive, GPU, provider spend, or score measurement was run.
- The common contract's frontier paragraph is stale relative to live authority. This memo uses
  `.omx/state/main_hot_state.md`: own-vehicle LC2 remains
  `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`. DT1 did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: next data-bearing Modal dispatcher owner; consumer store: EC2 closure manifest, named SSD `endpoint_closure/`, retained payload store, and both Modal ledgers; fire trigger: the next data-bearing Modal spawn after the global single-flight guard is clear.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: keeper maintainer; consumer store: `tools/codex_arm_queue.py`, arm-final-message index, and Git history; fire trigger: AC1's manifest schema lands or the next non-Modal arm declares a Git-blocked memo path and SHA.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: task #1001 successor; consumer store: `.omx/state/operator_p0_ledger.jsonl`; fire trigger: current source owners land, then continue PR1's executable-value-ordered payload-retention migration and earn normal-hook coverage before a strict flip.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_nb1_r2`; consumer store: AU1 candidate JSONL and canonical task-status bodies; fire trigger: the next headline candidate is proposed for canonical promotion, requiring its body/correction receipt first.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: costate apparatus maintainer; consumer store: canonical task-status/costate duty queue; fire trigger: two deduplicated incident receipts share a genus with neither a strict cure nor an active owner.

## LIVE-HYPOTHESES

- A worker closure sealed against the actual interpreter will generalize beyond EC2 because the
  repeated failures all came from confusing image-wide availability with target-venv availability;
  the receipt deliberately accepts arbitrary entry-point chains and explicit payload roots.
- The existing AU1 and costate surfaces are sufficient for recurring determinization if a typed
  incident identity is added between them; their present failure is adjudication/deduplication, not
  missing corpus coverage.
- Agent-precise ownership in the claim API may expose older callers that rely on ambiguous lane-only
  adoption; that is plausible because the claims ledger contains older wrapper/preclaim collisions,
  and such callers should be migrated rather than silently adopted.

## DEAD-ENDS

- Counting AU1 rows, payload findings, or 110 prefix sites as incidents is closed: these are
  high-recall detector populations and repeat one underlying fact many times.
- Extending Catalog #203 alone is closed: its canonical-training-image string check cannot prove the
  recursive imports of a different worker or the packages in that worker's target interpreter.
- Matching terminal calls by shared lane or job label when a claim contains a call ID is closed: EC2
  r1-r3 proved that fallback emits false blockers against a newer call.
- Rebuilding endpoint closure, payload retention, or prefix selection in DT1 is closed: each already
  has an owner or strict canonical cure, so duplication would create competing apparatus.
- This scorer-free apparatus arm is not a frontier path and produced no score row. Own-vehicle
  frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.
