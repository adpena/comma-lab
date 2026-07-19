# Harness-engineering crosswalk — Codex fresh-eyes review

**UTC:** 2026-07-19  
**Task:** #565, `lane_harness_engineering_crosswalk_20260719`  
**Authority:** delegated prompt SHA-256 `98726b296fde3aa849bed3107f24e97aebcf331819322161b81fa47831a034e7`  
**Upstream snapshot:** [`lopopolo/harness-engineering@226c8d35`](https://github.com/lopopolo/harness-engineering/tree/226c8d35fb6ea3ed55467753dba6dea2b5fd5778), tag `v1.0.0`  
**Scope:** apparatus comparison only; `research_only=true`; no launch, score, submission, promotion, or pointer authority  
**Required next authority:** MAIN landing review

## Verdict

**ADOPT four corrections first; retain the rest as either already covered or non-applicable evidence.** The highest-priority adoption is not another prose rule. It is a typed migration of Pact's harness-failure ledger. The current digest reproducibly reports **20 classes / 3 unresolved**, but that number is not lifecycle-safe: newer `class_id` rows collapse into `?`, prose in a `resolution` field can be mistaken for a closure event, and two class aliases double-count unless normalized. The next three adoptions close the genuinely owed watchdog cadence, the five-round review cap, and a provider lexical-compatibility check.

The upstream repository is useful and internally coherent, but it is a retrieval-optimized documentation and evidence bundle, not an executable orchestration harness. It explicitly reports no end-to-end application record yet (`playbooks/README.md:18-21`). Pact already has materially stronger executable custody for spawn/liveness, commits, landing diffs, review, and governed launch. Those are signals that could be contributed upstream; they are not reasons to replace Pact machinery.

`0.1910828242 [contest-CPU]` is **UNMOVED**. No run was launched. `experiments/results/levelset_n600_witness_20260717T113932Z/` was not touched.

## Method and evidence boundary

- **MEASURED:** the upstream checkout is commit `226c8d35fb6ea3ed55467753dba6dea2b5fd5778`, tag `v1.0.0`, and contains two commits at review time.
- **MEASURED:** the authored synthesis, evaluations, and playbooks contain 169 substantive H2 recommendation/evaluation sections after excluding licenses, repository instructions, and source-corpus metadata. Appendix A has one row per section.
- **MEASURED:** `ARCHITECTURE.md:3-7` calls the repository a retrieval-optimized bundle. `playbooks/README.md:3-6` calls playbooks editorial syntheses.
- **MEASURED:** every local coverage claim below was re-derived from code or state, not accepted from a memo.
- **DERIVED:** source snapshots and bibliographies are evidence for authored recommendations, not additional recommendations. They are separately dispositioned under N1; licensing and repository-edit instructions are under N2.
- **Verdict discipline:** every negative is bounded by an explicit `verdict_scope`. Following `docs/operating_manual_craft_handoff.md:119-141`, claims are labeled; following `:145-169`, the review attacked its own conclusion; and following `:173-192`, negatives state what they do not establish.

## Critical correction: “20 / 3” is a reproducible digest, not stable lifecycle truth

Running the current `tools/costate_digest.py` summarizer on `.omx/state/harness_failure_ledger.jsonl` produced:

```text
classes = 20
unresolved = ['?', 'phantom_death_buffered_log_plus_misfired_grep_liveness',
              'sigurg_144_harness_kills_bg_bash_process_group']
```

This is **MEASURED**, but the interpretation “exactly these three classes remain open” is falsified by the reader contract:

1. `tools/costate_digest.py:689-729` accepts `failure_id`, `failure_class`, `class`, and `bug_class`, but not `class_id`. Ledger rows 66-68 therefore coalesce into the synthetic class `?`.
2. The same reader treats any nonempty legacy `resolution` prose as a resolution marker when no explicit event is present. Rows 63-64 explicitly say prevention remains owed, yet can be summarized as closed.
3. The raw identifiers contain 22 strings; normalizing `codex_probe_token_limit_death_incomplete_wip` with its dated form, and `dashboard_false_FAIL_at_init` with `dashboard_hardcoded_gate_boundary_false_fail_at_init`, yields the intended 20 semantic classes.
4. The latest SIGURG recurrence row says the existing cure held, but is not a canonical typed closure event. Its appearance in the unresolved list is therefore a schema/transition ambiguity, not evidence that the cure failed.
5. `src/tac/harness_failure_ledger.py:22-31,65-71,220-243` and `tools/costate_digest.py:689-747` are two lifecycle readers with different accepted shapes. That is the ownership seam.

**DERIVED verdict:** the advertised count is suitable only as a compatibility snapshot. Until A1 lands, it must not authorize prioritization or closure. `verdict_scope=current mixed-schema harness-failure digest only; this does not negate any individually custodied fix receipt.`

## Ranked ADOPT table

| Rank | Disposition | Upstream item(s) | Concrete Pact change | Falsifiable gate | Prevented class or unprotected surface |
|---:|---|---|---|---|---|
| 1 | **ADOPT — A1** | One semantic owner; finish migrations; parse uncertainty at the boundary; recover the governing failure class; preserve failed work as evidence | Add one typed `FailureEventV2` owner in `src/tac/harness_failure_ledger.py`; explicitly migrate legacy keys including `class_id`; normalize the two known aliases; separate prose from `resolution_state`; require a typed closure event; make the digest consume only the canonical reader; emit canonical supersession rows rather than rewriting history. | Feed exact ledger rows 59 and 63-68 through migration. Assert 20 normalized classes, no `?`, no prose-only closure, and independently correct states for phantom-death, provider lexical-trigger, review-spiral, and SIGURG. Reject unknown lifecycle shapes. | Directly removes current unresolved `?`; prevents false closure of two prevention-owed classes and false reopening/closure of SIGURG. |
| 2 | **ADOPT — A2** | Repository-owned continuous loop; prove and record each iteration; place context at latest reliable point | Install a repository-owned ~15-minute watchdog schedule around existing `tools/witness_chain_watchdog.py`, with a durable receipt, idempotent deduplication, and the canonical liveness API. Buffered log silence alone must never trigger death. | Fixture A: live process tree/high CPU/frozen log -> `ALIVE`, no restart. Fixture B: absent tree/no fresh receipt -> one alert/action receipt. Replaying the same observation -> no duplicate action. | `phantom_death_buffered_log_plus_misfired_grep_liveness` (ledger row 59), one of the digest's three current unresolved labels. |
| 3 | **ADOPT — A3** | Keep review convergent; spend human attention on ambiguity/authority | Extend `tac.subagent_contract` and the delegate wrapper with `self_review_round_cap=5`, a persisted round counter, and terminal `ESCALATE_MAIN`; preserve clean-pass reset semantics separately. | Five completed self-review rounds permit escalation; a sixth self-review start is refused and emits an escalation receipt. A finding in round 5 resets only the clean-pass counter, not the hard round cap. | `arm_review_spiral_unbounded_seal_loop` (ledger row 64), whose prevention is still explicitly owed. |
| 4 | **ADOPT — A4** | Interpret instructions through the authority contract; expose context and capability together | Add a narrow pre-dispatch lexical-compatibility fixture to delegated-task composition. Detect only known provider-trigger combinations, point to neutral equivalent wording, preserve substantive constraints, and allow an explicit reviewed exception. | The exact historical synthetic phrase refuses before provider dispatch; a neutral near-neighbor passes; constraints and hashes remain unchanged after the suggested rewrite. | `provider_content_filter_false_positive_kills_arm` (ledger row 63), whose prevention is still explicitly owed. |
| 5 | **ADOPT — A5** | Fixed-worker adoption epochs; requalify upgrades; matched migration | Record a `worker_epoch` receipt: model, effort, host, tool schema, context-bundle commit, and baseline/treatment trajectory IDs. Requalify whenever one changes. | Attribution is refused when baseline and treatment cross epochs; the same fixture passes with one epoch and fresh matched trajectories. | Unprotected worker-drift attribution surface; prevents crediting a harness change for a worker/config change. |
| 6 | **ADOPT — A6** | Start with the decision; locate earliest failed handoff; fresh trajectory; recognize invalid results | Add a typed harness-intervention receipt with baseline, treatment, optional ablation, earliest failed handoff, availability/retrieval/invocation/relevance, accepted outcome, invalid reason, and retained/revised/removed verdict. | Presence-only or terminology-only changes cannot earn `RETAIN`; a fresh matched outcome plus claim-boundary proof can. A no-op remains a durable state. | Unprotected causal-attribution surface; prevents proxy-progress and guard-theater conclusions. |
| 7 | **ADOPT — A7** | Accepted outcome; fully loaded human attention; four clocks; utilization; rework/waste; lifetime cost | Add a comparable effectiveness record keyed by `worker_epoch`: accepted outcome, human-attention minutes, elapsed/active/tool/wait clocks, productive/rework/waste split, and carrying cost. | Missing dimensions mark a report `NON_COMPARABLE`; fixed fixtures reproduce aggregates and never equate activity with accepted output. | Unprotected harness-cost surface; makes review spirals and latency regressions visible even when output count rises. |
| 8 | **ADOPT — A8** | Model-native semantics and complete migration | Create host-conformance journeys for command grammar, result/error shape, partial failure, cancellation, timing, and patch semantics across supported Codex/Claude/desktop seams. | A host migration must replay the same journey corpus; any semantic delta requires an explicit compatibility disposition. | Unprotected cross-host semantic-drift surface. |
| 9 | **ADOPT — A9** | Garden routes; finish migrations; retire absorbed scaffolding; version automations | Add a route/migration registry with owner, authority, freshness, replacement, tombstone, and ablation status; generate stale/duplicate-route findings. | Two authoritative routes for one concept, an ownerless route, or a replaced route without a tombstone fails the ratchet. | Unprotected stale-context and duplicate-authority surface. |
| 10 | **ADOPT — A10** | Context curator; low-trust model commentary as one sensor; promote stable decisions to owner | Stage low-trust feedback as `CANDIDATE -> CORROBORATED -> PUBLISHED`; raw telemetry never enters agent context automatically; publication requires provenance and an owner. | Raw self-report cannot become an instruction; corroborated evidence with an owner can. Revoked evidence disappears from generated context without deleting history. | Unprotected self-referential feedback-poisoning and stale-context surface. |
| 11 | **ADOPT — A11** | Dependency ownership; capability transfer; next 5,000 changes | Add a dependency-capability record with owner, version authority, replacement parity, decommission evidence, and failure consequence. | Removal/replacement is refused without parity and an ownership transfer; stale version authorities produce one actionable finding. | Unprotected dependency-lifetime and responsibility-loss surface. |
| 12 | **ADOPT — A12** | Tool discovery/familiarity; result as context; descend to missing primitive | Generate a compact tool catalog from owners: purpose, selection cue, invocation, result/error schema, dry-run, recovery, and postcondition. | Help/catalog drift against callable interfaces fails; a representative unfamiliar-worker journey must select and recover from the intended tool. | Unprotected tool-discovery and recovery surface. |

## ALREADY-COVERED evidence catalog

Each `C#` is a reusable disposition referenced by Appendix A. “Stronger” means Pact has executable enforcement where upstream offers a practice description; it is an upstream contribution signal, not a Pact deficiency.

| Ref | Disposition and exact first-hand Pact evidence |
|---|---|
| **C1** | **ALREADY-COVERED — contextual routing and session apparatus.** `.claude/settings.json:3-63` wires the pre-tool launch guard, Stop-time triality drift/auto-push/landing checks, and SessionStart costate/P0 digest; `src/tac/subagent_contract.py:396-471` composes grounded work/review/final instructions. Pact is stronger because routing is executable. |
| **C2** | **ALREADY-COVERED — spawn/liveness.** `tools/codex_delegate.py:5-19,226-243,450-462,557-605` provides isolated PTY/session spawn, checkpointed retry, custody, and the long-lived `.omx/tmp/codex_runs/codex_events.log` `tail -n 0` (never replay) notification contract; `tools/codex_status.py:35-69,100-145,266-385` derives composite liveness and actionable fleet states. Pact is stronger; upstream has no corresponding runtime. |
| **C3** | **ALREADY-COVERED — serialized commit custody.** `tools/subagent_commit_serializer.py:22-37,54-125,399-429,466-496,582-765,1494-1588,1807-1850,1930-2039` covers `fcntl` locking, expected-content/base/post SHA, ignored-file rc13, protected-doc rc14, exact patch intent, staged SHA, and post-commit verification. Pact is stronger. |
| **C4** | **ALREADY-COVERED — typed landing receipt and findings consumption.** `src/tac/landing_diff_manifest.py:1-6,61-65,430-540,620-636` re-derives a conservative BASE..HEAD manifest with default `UNACCOUNTED`; `tools/codex_landing_review_gate.py:4-50,298-443,514-583` enforces consumer/terminal-state semantics; `src/tac/preflight.py:88227-88245,88364-88455` supplies the WARN-only migration read side. Pact is stronger. |
| **C5** | **ALREADY-COVERED — review state.** `tools/review_tracker.py:276-370,592-707,895-1008,1054-1158` identifies reviewable Python, auto-ingests unknown new files, persists and enforces the two-pass clean policy, and resets on findings. Pact is stronger on executable review bookkeeping; A3 remains owed for the hard five-round cap. |
| **C6** | **ALREADY-COVERED — launch authority/governance.** `tools/launch_witness_run.py:430-523,565-646,815-833,2740-2808,2909-2949,3094-3135` binds identity, governed admission, DSL compile hash, memory preflight, and same-outdir refusal; `tools/spawn_durable_daemon.py:824-892,1006-1055`, `tools/system_memory_governor.py:217-244`, and `src/tac/admission_guard.py:1-20,82-188` enforce config freshness rc6, `TAC_ADMISSION_ENFORCE`, and runtime admission. Pact is stronger. |
| **C7** | **ALREADY-COVERED — durable failure/finding consumption, with A1 exception.** `.omx/state/harness_failure_ledger.jsonl:1-68` preserves 20 normalized semantic classes; `tools/codex_landing_review_gate.py:4-21` encodes two-landing fix+gate logic; `src/tac/preflight.py:88227-88455` consumes it. A1 is required before aggregate lifecycle summaries are authoritative. |
| **C8** | **ALREADY-COVERED — whole-job continuity/resume.** `tools/subagent_checkpoint.py:3-52,180-241,276-305` provides locked, successor-queryable checkpoints; `tools/codex_delegate.py:226-243` resumes monitoring without replay; C4 closes the lifecycle. Pact is stronger. |
| **C9** | **ALREADY-COVERED — proof at the claim boundary.** `docs/operating_manual_craft_handoff.md:92-104,119-169,173-192` requires primary-artifact re-derivation, explicit evidence labels, self-attack, and honest negative bounds; C4 proves exact Git objects. |
| **C10** | **ALREADY-COVERED — typed/domain boundaries.** `src/tac/landing_diff_manifest.py:61-65,430-540` owns typed landing dispositions; `tools/launch_witness_run.py:2740-2808` requires the typed DSL artifact and recomputed hash instead of invented trailing flags. |
| **C11** | **ALREADY-COVERED — authority and consequence staging.** `.claude/settings.json:3-42` routes launch checks and Stop gates; `tools/launch_guard_hook.py:3-25,39-43` enforces/elevates launch ambiguity; `src/tac/admission_guard.py:144-188` makes runtime admission explicit. |
| **C12** | **ALREADY-COVERED — tool result legibility.** `tools/subagent_commit_serializer.py:107-125` has stable result codes; `tools/codex_status.py:361-385` emits operator-actionable buckets; `tools/launch_witness_run.py:815-833,2740-2808` gives refusal reasons at the actuation boundary. |
| **C13** | **ALREADY-COVERED — current world model and observability.** `.claude/settings.json:44-63` starts costate/P0 digests and the dashboard; `tools/codex_status.py:266-385` provides canonical fleet state; `tools/subagent_checkpoint.py:276-305` supports recovery queries. |
| **C14** | **ALREADY-COVERED — artifact identity.** `tools/launch_witness_run.py:430-445,565-646` records command/config/content hashes; `src/tac/landing_diff_manifest.py:490-540,620-636` binds review to exact BASE..HEAD Git objects. |
| **C15** | **ALREADY-COVERED — parallel ownership/blast-radius reasoning.** `docs/operating_manual_craft_handoff.md:18-28,42-54,68-88` requires class fixes, partitioning, ownership, seam verification, and risk by probability × blast radius × silence; C2/C3 enforce isolated execution and serialized mutation. |
| **C16** | **ALREADY-COVERED — outcome and honest handoff.** `docs/operating_manual_craft_handoff.md:10-12,173-192,196-234` centers attack-surviving claims, answer-first reporting, bounded negatives, and common failure modes; this memo therefore leaves score/pointer authority untouched. |

## NOT-APPLICABLE catalog

| Ref | Disposition |
|---|---|
| **N1** | **NOT-APPLICABLE — evidence/source-corpus maintenance.** Bibliographies, captured posts, imported case ledgers, and source-fetch scripts are provenance inputs, not an additional Pact runtime recommendation. `verdict_scope=Task #565 Pact apparatus crosswalk only; the upstream evidence corpus remains valid within its own project.` |
| **N2** | **NOT-APPLICABLE — repository license/editorial maintenance.** License text, contributor instructions, and playbook-editing mechanics do not specify a Pact harness change. `verdict_scope=Task #565 runtime and governance apparatus only; no negative judgment on upstream repository maintenance.` |
| **N3** | **NOT-APPLICABLE — named external implementation as implementation mandate.** Polytoken, ACP, Artichoke, rand_mt, Hyperbola, Ryan's homelab, and RustSec are cases supporting patterns, not dependencies Pact should import. Their transferable patterns map to A/C rows. `verdict_scope=this crosswalk's dependency decision only; not a negative verdict on the named implementation or its pattern.` |
| **N4** | **NOT-APPLICABLE — end-user delivery surface.** The delegated apparatus task has no no-code/end-user deployment surface or credential handoff to build. `verdict_scope=Task #565 research-only apparatus comparison; not a negative on last-mile deployment as a family.` |
| **N5** | **NOT-APPLICABLE — release actuation.** This arm has no publish/release authority; it can compare artifact-identity principles but cannot build or execute an upstream-style release path. `verdict_scope=Task #565 no-launch/no-release authority; not a negative on release-integrity practices.` |

## Vocabulary to fold into the failure ledger

A1 should add these typed fields rather than more free-form aliases:

| Field | Purpose |
|---|---|
| `class_id` | Stable normalized identity; legacy identifiers become explicit aliases. |
| `event_kind` | `OBSERVED`, `RECURRENCE`, `FIX_LANDED`, `GATE_LANDED`, `VERIFIED_CLOSED`, `REOPENED`, `SUPERSEDED`. |
| `resolution_state` | `OPEN`, `FIX_ONLY`, `GATE_ONLY`, `VERIFY_PENDING`, `CLOSED`, `SUPERSEDED`; never inferred from prose. |
| `earliest_failed_handoff` | `availability`, `retrieval`, `invocation`, `relevance`, `execution`, `proof`, `consumption`, `lifecycle`. |
| `failure_shape` | `capability_gap`, `context_gap`, `authority_gap`, `semantic_drift`, `custody_gap`, `liveness_misclassification`, `migration_gap`, `convergence_gap`. |
| `claim_boundary` | Exact system/environment where the cure is asserted. |
| `worker_epoch` | Fixed worker/tool/context identity for causal comparisons. |
| `intervention_id` / `ablation_id` | Links a correction to fresh treatment/control evidence. |
| `evidence_refs` | Exact receipts, commits, artifacts, or line-addressed state. |
| `next_trigger` | Falsifiable observation that reopens or advances the class. |
| `owner` | One semantic owner for transition authority. |
| `verdict_scope` | Prevents one failed formulation from killing a broader family. |

The upstream vocabulary worth preserving is the **earliest failed handoff** and the distinction between **availability, retrieval, invocation, relevance, accepted outcome, rework, and lifetime cost**. It sharpens Pact's existing class ledger without replacing its two-landing cure rule.

## Inverse crosswalk: their repository does not cover what bit Pact

These are **MEASURED absences at upstream commit `226c8d35`; verdict_scope=that repository snapshot's executable apparatus, not the validity of its design principles.**

1. No canonical concurrent-worker spawn, PTY/session custody, liveness composite, event-log `tail -n 0` contract, retry-resume checkpoint, or buffered-log false-death prevention. Pact C2/C8 exists because these failures occurred.
2. No serialized mutation protocol with `fcntl`, expected-content SHA, ignored-file rc13, protected-doc rc14, exact patch intent, staged SHA, and post-commit SHA. Pact C3 is materially beyond the anthology.
3. No typed BASE..HEAD landing receipt whose default is `UNACCOUNTED`, no required findings consumer, and no WARN-only-to-strict migration surface. Pact C4 covers this.
4. No N-pass Python review state machine, unknown-file auto-ingest, finding reset, or hard cap/escalation state. Pact C5 covers most; A3 is the honest remaining gap.
5. No executable launch governor binding memory admission, `TAC_ADMISSION_ENFORCE`, same-outdir exclusion, config freshness rc6, and a recomputed DSL compile hash. Pact C6 covers this.
6. No typed, append-only failure-lifecycle implementation. The repository recommends failure classes and durable owners, but does not confront Pact's real mixed-schema/alias/closure bug. A1 is the concrete local lesson.
7. No measured contest custody model separating `[contest-CPU]`, `[contest-CUDA]`, and advisory axes, and no pointer-preservation protocol. That domain-specific omission is expected; `verdict_scope=contest witness apparatus only.`

Upstream's strongest contribution is a cleaner causal evaluation vocabulary. Pact's strongest potential upstream contribution is executable custody under concurrent, stateful, failure-prone work.

## Appendix A — exhaustive authored-section disposition

**Inventory rule:** one row per substantive H2 in the authored synthesis, evaluations, and playbooks: 169 rows. In every row, `A#` means **ADOPT**, `C#` means **ALREADY-COVERED**, and `N#` means **NOT-APPLICABLE** with the catalog's inherited `verdict_scope`. Each disposition resolves through the ranked A table, exact local C evidence, or scoped N catalog above. Subordinate examples/failure modes/tools inherit the section's row; no source-corpus item is silently promoted into a recommendation. `COPYING.md`, `AGENTS.md`/`CLAUDE.md`, `playbooks/README.md`, and `sources/**` were inspected and are covered by N1/N2 rather than counted as recommendations.

### Repository architecture and index (9)

| Upstream item | Disposition |
|---|---|
| [`README.md` — Sources and related work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/README.md#L68) | N1 |
| [`ARCHITECTURE.md` — Retrieval boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/ARCHITECTURE.md#L9) | C1 |
| [`ARCHITECTURE.md` — Ownership](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/ARCHITECTURE.md#L24) | C10 |
| [`ARCHITECTURE.md` — Invariants](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/ARCHITECTURE.md#L45) | C9 |
| [`docs/README.md` — Start here](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/README.md#L8) | C1 |
| [`docs/README.md` — Theses](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/README.md#L18) | C1 |
| [`docs/README.md` — Applications](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/README.md#L129) | A6 |
| [`docs/README.md` — Related work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/README.md#L156) | N1 |
| [`docs/README.md` — Evidence](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/README.md#L167) | C9 |

### Fixed worker and model-native semantics (9)

| Upstream item | Disposition |
|---|---|
| [`fixed-worker` — Work in adoption epochs](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/README.md#L26) | A5 |
| [`fixed-worker` — Requalify every worker upgrade](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/README.md#L55) | A5 |
| [`fixed-worker` — Treat inner-loop latency as part of qualification](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/README.md#L104) | A7 |
| [`fixed-worker` — Retire scaffolding the worker has absorbed](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/README.md#L124) | A9 |
| [`model-native` — Treat the action language as a compatibility surface](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/model-native-semantics.md#L20) | A8 |
| [`model-native` — Preserve semantics at each migration seam](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/model-native-semantics.md#L36) | A8 |
| [`model-native` — Polytoken selects tools for the active model](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/model-native-semantics.md#L48) | N3; pattern -> A8 |
| [`model-native` — ACP keeps the coding agent behind the transport boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/model-native-semantics.md#L75) | N3; pattern -> A8 |
| [`model-native` — Evaluate the complete migration](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/fixed-worker/model-native-semantics.md#L113) | A8 |

### Last-mile deployment (9)

| Upstream item | Disposition |
|---|---|
| [`last-mile` — Discover the deployment surface](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L21) | A10 |
| [`last-mile` — Model the work around the records](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L41) | C10 |
| [`last-mile` — Keep each kind of truth with its owner](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L79) | C10 |
| [`last-mile` — Pair each goal with a context curator](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L116) | A10 |
| [`last-mile` — Promote stable decisions to their owner](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L199) | C7 |
| [`last-mile` — Expose context and capability together](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L225) | C1 |
| [`last-mile` — Give domain experts a paved lane](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L252) | C15 |
| [`last-mile` — Serve users who never see code](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L277) | N4 |
| [`last-mile` — Measure situated effectiveness](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/last-mile-deployment/README.md#L301) | A7 |

### Whole-job delegation and authority (13)

| Upstream item | Disposition |
|---|---|
| [`whole-job` — Delegate the outcome and its bar](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L10) | C8 |
| [`whole-job` — Keep durable intent sparse across compaction](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L63) | C8 |
| [`whole-job` — Recover hidden requirements](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L115) | C9 |
| [`whole-job` — Keep one primary trajectory](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L141) | C15 |
| [`whole-job` — Carry purpose across the organization and trajectories](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L172) | C8 |
| [`whole-job` — Close the lifecycle](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L205) | C4 |
| [`whole-job` — Let the outcome choose the artifact](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L244) | C9 |
| [`whole-job` — Spend human attention on ambiguity and authority](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/whole-job/README.md#L279) | A7 |
| [`authority` — Give reversible work a broad envelope](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/authority/README.md#L9) | C11 |
| [`authority` — Keep credential custody outside the trajectory](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/authority/README.md#L33) | C11 |
| [`authority` — Stage consequential effects](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/authority/README.md#L76) | C11 |
| [`authority` — Interpret instructions through the authority contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/authority/README.md#L113) | C11 |
| [`authority` — Encode settled boundaries mechanically](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/authority/README.md#L126) | C11 |

### Tool legibility and just-in-time context (13)

| Upstream item | Disposition |
|---|---|
| [`tool-legibility` — Make discovery and familiarity work together](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/tool-legibility/README.md#L16) | A12 |
| [`tool-legibility` — Design every result as context](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/tool-legibility/README.md#L56) | C12 |
| [`tool-legibility` — Expose the smallest interface that closes the job](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/tool-legibility/README.md#L90) | C12 |
| [`tool-legibility` — Descend to a missing primitive](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/tool-legibility/README.md#L108) | A12 |
| [`tool-legibility` — Give the worker the real system](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/tool-legibility/README.md#L132) | C9 |
| [`just-in-time-context` — Route across distinct knowledge layers](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L38) | C1 |
| [`just-in-time-context` — Curate the route beside each goal](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L56) | A10 |
| [`just-in-time-context` — Use AGENTS.md as a map](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L186) | C1 |
| [`just-in-time-context` — Deliver context in three phases](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L213) | C8 |
| [`just-in-time-context` — Let the worker discover the next layer](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L246) | C1 |
| [`just-in-time-context` — Place context at the latest reliable point](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L274) | C1 |
| [`just-in-time-context` — Use skills to teach and runbooks to preserve work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L288) | C1 |
| [`just-in-time-context` — Garden routes and corpus](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/just-in-time-context/README.md#L322) | A9 |

### Domain modeling and lineage (21)

| Upstream item | Disposition |
|---|---|
| [`domain-modeling` — Code is part of the prompt](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L21) | C10 |
| [`domain-modeling` — Make nonfunctional requirements recoverable](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L53) | C9 |
| [`domain-modeling` — Consistency compresses context](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L114) | C10 |
| [`domain-modeling` — One concept, one authoritative owner](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L157) | A1 |
| [`domain-modeling` — Finish migrations and install the ratchet](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L188) | A1 |
| [`domain-modeling` — Parse uncertainty at the boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L225) | A1 |
| [`domain-modeling` — Encode ownership mechanically](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L246) | C10 |
| [`domain-modeling` — Preserve meaningful variation](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/README.md#L275) | C10 |
| [`implementations` — Ryan's homelab: one semantic owner](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/implementations.md#L3) | N3; pattern -> A1 |
| [`implementations` — Artichoke: capabilities as context boundaries](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/implementations.md#L19) | N3; pattern -> C10 |
| [`implementations` — rand_mt: repository shape as operating context](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/implementations.md#L42) | N3; pattern -> C1 |
| [`hyperbola` — Policy has a package](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/hyperbola.md#L7) | N3; pattern -> C10 |
| [`hyperbola` — Domain types replace permissive primitives](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/hyperbola.md#L19) | N3; pattern -> A1 |
| [`hyperbola` — Package topology carries intent](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/hyperbola.md#L29) | N3; pattern -> C10 |
| [`hyperbola` — Documentation is an architecture contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/hyperbola.md#L40) | N3; pattern -> C1 |
| [`lineage` — A short, stable codemap](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L12) | C1 |
| [`lineage` — Capability and migration seams](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L36) | A8 |
| [`lineage` — Parse at the boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L78) | A1 |
| [`lineage` — Close feedback loops around judgment](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L101) | A6 |
| [`lineage` — Incremental adoption](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L171) | A9 |
| [`lineage` — Evolution across Ryan's work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/lineage/README.md#L204) | N3 |

### Homelab case (13)

| Upstream item | Disposition |
|---|---|
| [`homelab` — Risks the harness must own](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L10) | N3; pattern -> C9 |
| [`homelab` — Route context just in time](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L36) | N3; pattern -> C1 |
| [`homelab` — Let architecture carry instructions](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L88) | N3; pattern -> C10 |
| [`homelab` — Make repository policy semantic code](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L162) | N3; pattern -> C10 |
| [`homelab` — Let supply-chain policy follow capabilities](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L223) | N3; pattern -> A11 |
| [`homelab` — Transfer responsibility when removing a Markdown dependency](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L280) | N3; pattern -> A11 |
| [`homelab` — Model consequential operations as a state machine](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L317) | N3; pattern -> C11 |
| [`homelab` — Close feedback loop with versioned automations](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L365) | N3; pattern -> A9 |
| [`homelab` — Generate observability from source of truth](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L404) | N3; pattern -> C13 |
| [`homelab` — Match proof to the changed surface](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L429) | N3; pattern -> C9 |
| [`homelab` — How the harness evolved](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L459) | N3 |
| [`homelab` — Maintenance and supply-chain outcomes](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L477) | N3 |
| [`homelab` — What to transfer](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/domain-modeling/homelab.md#L509) | N3; transferable rows above |

### Proof and release identity (12)

| Upstream item | Disposition |
|---|---|
| [`proof` — Define success where it will be experienced](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L7) | C9 |
| [`proof` — Match evidence to the claim](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L37) | C9 |
| [`proof` — Give the agent access to the real system](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L70) | C9 |
| [`proof` — Compress the trajectory for review](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L136) | C4 |
| [`proof` — Preserve artifact identity through delivery](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L161) | C14 |
| [`proof` — Give the agent ground truth it can test against](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L182) | C9 |
| [`proof` — Say what the evidence did not establish](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/README.md#L235) | C16 |
| [`release-integrity` — Build once](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/release-integrity.md#L7) | N5; principle -> C14 |
| [`release-integrity` — Bound release authority](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/release-integrity.md#L20) | N5; principle -> C11 |
| [`release-integrity` — Prove the deployed boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/release-integrity.md#L54) | N5; principle -> C9 |
| [`RustSec case` — Reproduce the state corruption](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/rustsec.md#L13) | N3; pattern -> C9 |
| [`RustSec case` — Carry the finding through review and release](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/proof/rustsec.md#L31) | N3; pattern -> C4 |

### Feedback and low-trust signals (18)

| Upstream item | Disposition |
|---|---|
| [`feedback` — Collect observable signals](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L14) | C7 |
| [`feedback` — Recover the governing failure class](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L59) | A1 |
| [`feedback` — Promote lesson into smallest durable owner](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L101) | C7 |
| [`feedback` — Learn principles and garbage-collect drift](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L130) | A9 |
| [`feedback` — Accrue domain expertise through the work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L221) | C13 |
| [`feedback` — Keep review convergent](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L288) | A3 |
| [`feedback` — Let product feedback change the job](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L303) | A10 |
| [`feedback` — Keep successful and failed work as evidence](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L337) | C7 |
| [`feedback` — Treat MLD as one sensor](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L358) | A10 |
| [`feedback` — Close the loop on later runs](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/README.md#L376) | A6 |
| [`MLD` — The three signals](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L23) | A10 |
| [`MLD` — The builder decides what persists](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L56) | C7 |
| [`MLD` — From a run to an intervention](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L62) | A6 |
| [`MLD` — Promote signals into owned surfaces](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L90) | A10 |
| [`MLD` — Keep raw telemetry out of agent context](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L111) | A10 |
| [`MLD` — Self-report under separated incentives](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L129) | A10 |
| [`MLD` — What MLD cannot prove](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L176) | C16 |
| [`MLD` — Where MLD fits](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/feedback/mld.md#L198) | A10 |

### Durable systems and dependency ownership (11)

| Upstream item | Disposition |
|---|---|
| [`durable-systems` — Make coherence cumulative](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L14) | C13 |
| [`durable-systems` — Optimize for the next 5,000 changes](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L94) | A11 |
| [`durable-systems` — Choose what must survive a rewrite](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L168) | A11 |
| [`durable-systems` — Treat dependencies as ownership decisions](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L206) | A11 |
| [`durable-systems` — Rebuild confidence when ownership moves](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L231) | A11 |
| [`durable-systems` — Keep policy authoritative and controls proportional](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L246) | C11 |
| [`durable-systems` — Preserve release identity](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/README.md#L269) | C14 |
| [`dependency-ownership` — Record the capability](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/dependency-ownership.md#L6) | A11 |
| [`dependency-ownership` — Keep specialist dependencies when they reduce total risk](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/dependency-ownership.md#L22) | A11 |
| [`dependency-ownership` — Own narrow behavior deliberately](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/dependency-ownership.md#L33) | A11 |
| [`dependency-ownership` — Keep versions authoritative](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/durable-systems/dependency-ownership.md#L52) | A11 |

### Continuous maintenance and effectiveness (13)

| Upstream item | Disposition |
|---|---|
| [`continuous-maintenance` — Give the loop a repository-owned contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/continuous-maintenance/README.md#L76) | A2 |
| [`continuous-maintenance` — Carry a current world model across runs](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/continuous-maintenance/README.md#L128) | C13 |
| [`continuous-maintenance` — Make every iteration prove and record its outcome](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/continuous-maintenance/README.md#L198) | A6 |
| [`continuous-maintenance` — Maintain the harness with the same loop](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/continuous-maintenance/README.md#L225) | A6 |
| [`continuous-maintenance` — Keep invention in the foreground](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/continuous-maintenance/README.md#L267) | C15 |
| [`effectiveness` — Measure the accepted outcome](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L9) | A7 |
| [`effectiveness` — Count fully loaded human attention](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L44) | A7 |
| [`effectiveness` — Keep four clocks](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L90) | A7 |
| [`effectiveness` — Separate addressability, activity, and productive utilization](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L132) | A7 |
| [`effectiveness` — Distinguish exploration, rework, and waste](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L172) | A7 |
| [`effectiveness` — Price the lifetime](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L216) | A7 |
| [`effectiveness` — Measure compounding longitudinally](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L239) | A7 |
| [`effectiveness` — Keep the dimensions visible](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/docs/effectiveness/README.md#L270) | A7 |

### Evaluation framework and Artichoke evaluation (11)

| Upstream item | Disposition |
|---|---|
| [`evals` — Start with the decision](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L20) | A6 |
| [`evals` — Hold the comparison steady](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L49) | A5 |
| [`evals` — Keep truth with the target](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L77) | C9 |
| [`evals` — Grade the contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L95) | A6 |
| [`evals` — Measure whether judgment compounds](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L117) | A7 |
| [`evals` — Recognize an invalid result](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L147) | A6 |
| [`evals` — Feed results back into the harness](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/README.md#L169) | A6 |
| [`Artichoke eval` — State change became a program of work](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/artichoke-state-modeling.md#L35) | N3; pattern -> C8 |
| [`Artichoke eval` — Preparation reduced uncertainty; integration remained large](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/artichoke-state-modeling.md#L141) | N3; pattern -> A6 |
| [`Artichoke eval` — Evaluate future regret](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/artichoke-state-modeling.md#L223) | N3; pattern -> A11 |
| [`Artichoke eval` — Distinguish durable capability from lucky decomposition](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/evals/artichoke-state-modeling.md#L427) | N3; pattern -> A6 |

### Improve-harness playbook (9)

| Upstream item | Disposition |
|---|---|
| [`improve-harness` — Establish scope and authority](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L22) | C11 |
| [`improve-harness` — Record the job contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L41) | C8 |
| [`improve-harness` — Observe the baseline](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L61) | A6 |
| [`improve-harness` — Locate earliest failed handoff](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L81) | A1 |
| [`improve-harness` — State one intervention hypothesis](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L112) | A6 |
| [`improve-harness` — Implement and verify at claim boundary](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L132) | C9 |
| [`improve-harness` — Run a fresh trajectory](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L148) | A6 |
| [`improve-harness` — Retain, revise, or remove](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L169) | A6 |
| [`improve-harness` — Preserve a compact result record](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/improve-harness.md#L186) | A6 |

### Repository-review playbook (8)

| Upstream item | Disposition |
|---|---|
| [`repository-review` — Start with the outcome](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L26) | C16 |
| [`repository-review` — Define the review contract](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L43) | C8 |
| [`repository-review` — Inspect work as a trajectory](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L54) | A6 |
| [`repository-review` — Recover evidence from prior collaborations](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L75) | C13 |
| [`repository-review` — Review ownership boundaries](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L98) | C15 |
| [`repository-review` — Findings](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L227) | C9 |
| [`repository-review` — After the review](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L240) | C4 |
| [`repository-review` — Readiness blockers](https://github.com/lopopolo/harness-engineering/blob/226c8d35fb6ea3ed55467753dba6dea2b5fd5778/playbooks/repository-review.md#L251) | C4 |

## Verification status

- **CLEAN:** Appendix key comparison: expected 169, actual 169, unique 169, missing 0, extra 0; every link pins upstream commit `226c8d35` and resolves to the named H2 line.
- **CLEAN:** all 38 grouped local line citations resolve to existing files and in-range lines; `git diff --check` is clean.
- **CLEAN:** the new canonical lane row is L0, `lane_class=apparatus`, `research_only=true`, with no score or launch gates marked.
- **PRE-EXISTING BLOCKER:** full `tools/lane_maturity.py validate` reports 110 historical evidence paths absent from this worktree; none names this lane. `verdict_scope=global historical lane-evidence path validation; it does not invalidate this memo, and this arm did not repair or suppress that debt.`

## MAIN landing-review checklist

MAIN should independently review:

1. **A1 first:** confirm the exact ledger rows and reproduce the digest before accepting the schema diagnosis; decide whether the intended semantic count remains 20 after alias normalization.
2. Confirm A2/A3/A4 correspond to the genuinely prevention-owed classes rather than to stale prose-based closure.
3. Sample at least one row from every Appendix A group against the fixed upstream commit and confirm the A/C/N disposition resolves to a falsifiable local claim.
4. Re-check every “stronger” claim at the cited Pact lines; a documentation repository's lack of runtime code is not itself a defect.
5. Require a separate design/build authority before implementing any A row. This artifact is a crosswalk, not an implementation authorization.

**One-line handoff:** adopt the typed ledger migration first; it is the prerequisite for knowing which harness failures are actually unresolved.
