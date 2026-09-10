# ddm_pr17 — run4 disposition and consumer-fix freshness policy

Date: 2026-09-10  
Audit cutoff: `2026-09-10T22:22:09Z`  
Axis: `[review; scorer-free revalidation of retained authority receipts; contract policy]`  
`research_only=true; score_claim=false; promotion_eligible=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

1. **RATIFY — ffi6 landing and typed row.** Commit `ea09174bbc5f9d990e4097a0c5e1756906ddd9b3` is the reviewed consumer implementation; the current `amendments[4]` is a valid `prefire_contract_consumer_fix.v1` row with unchanged definition parent, correct previous-row chain, an exact 17-file manifest, and the pr16 adjudication pin.
2. **REFUSE — run4 as-dispatched completion under the later row.** Run4 is a real retained `[contest-CUDA T4 n600]` measurement, but intent v4 is stale under `amendments[4]`; it cannot create `candidate_seal.v3`, authorize promotion, or move the pointer. A fresh fire is required, and run4 remains corroborating evidence only.
3. **AMEND-PROSPECTIVE — explicit NO-GRACE rule.** Add the literal policy below without changing the existing validator: every consumer-fix append invalidates earlier intents at dispatch and completion, even after reservation or spawn. The worst-case cost is one additional exact fire per consumer-fix landing batch that crosses an in-flight authorization.
4. **AMEND — runs 1–5 custody is not complete.** Replay prevention holds; all three provider calls have canonical dispatch rows and both terminal calls have terminal rows. But run3 lacks pr16's exact typed terminal-reconciliation object and linked ledger event; run1's claim wording and run4's `pending_v3_completion`/P0 wording need append-only corrections; run5 is legitimately in flight and therefore has no terminal receipt at this cutoff.

`verdict_scope: CONTRACT FORMULATION + NAMED INSTANCES` — the policy governs the first-measurement contract and these retained runs. It does not alter an evaluator digest, scorer output, candidate bytes, runtime-content identity, score equation, or the canonical frontier.

## Literal AMEND — consumer fixes under an in-flight intent

> **Consumer-fix freshness — NO GRACE.** A `candidate_prefire_intent.v1` and every authorization that binds it are usable only while the intent's `latest_row_sha256` equals the canonical digest of the current last freeze row. Appending a `prefire_contract_consumer_fix.v1` row invalidates every intent pinned to an earlier row for both dispatch and completion, including an intent whose authorization was already reserved or spawned. The earlier call may finish; MAIN may harvest and retain its result under the original intent, authorization, nonce, job, output, and call lineage. That result MUST NOT create `candidate_seal.v3`, receive promotion adjudication, or move the pointer under the later contract state. After the earlier call is terminal and single-flight is clear, MAIN MUST emit a new intent after the current row's implementation timestamp, commit a new authorization, reserve a new nonce with a fresh job/output/receipt path, and re-fire. The prior nonce is never reused. Identical candidate bytes do not create an exception, and no typed provenance link may upgrade the earlier receipt across the row boundary. Batch related fixes before the new intent, or defer a non-safety consumer landing until the active lifecycle is terminal, to avoid unnecessary fires; never weaken the exact latest-row check.

This is a clarification, not a validator delta. Pr16 already says that an intent must pin the exact latest row, that a stale preceding row refuses, and that every post-amendment scored fire needs a newly emitted intent and fresh authorization. Naming the rule prospectively prevents a later operator from interpreting a passing retained-evidence replay as permission to grandfather the old authorization.

## Why run4 cannot be grandfathered

Pr12 separates three immutable objects: pre-fire intent, one-shot authorization, and the completed v3 seal. Its completion procedure revalidates every non-timing gate and current candidate identity from disk, requires the current pointer to equal the intent's zero-tolerance binding, and only then creates the first promotable object. Pr16 preserves the exact-latest-row check when it introduces typed consumer batches and explicitly says any post-amendment scored fire needs a new intent and authorization.

The live implementation makes the distinction observable:

- `_pf_completion_facts(intent_v4, authorization_v7, run4_receipt)` passes. It proves the call, request, result, timeout, n600, cold public path, score components, and 48-file retained runtime custody are internally sound.
- `validate_prefire_intent(intent_v4)` then refuses with `PREFIRE_CONTRACT_DRIFT_REFUSED: latest exact frozen amendment required`.
- `validate_prefire_intent(intent_v5)` passes and pins the current typed row digest `e77ed88e2f28e69246087310c5224c1bb0085a2f780bc75693a98cfa368b9cb3`.
- `complete_first_fire_intent` deliberately computes the retained completion facts and then calls `validate_prefire_intent` before score admission or seal creation. There is no accidental missing door to exploit.

Allowing a provenance-link grace would change the authority rule after seeing run4. It would also create two meanings of “current consumer passed”: either current code replays all evidence and the current intent pins it, or old code produced the result and new code merely approves it. The first is the contract pr12/pr16 specify; the second is a retrospective exception. Pr10's settled lesson forbids selecting the authority rule after the receipt exists. The corpus-wide freshness law likewise requires consumer-side verification at consumption and says production stamps alone are insufficient.

Run4 therefore has a narrow but valuable disposition: **retained exact corroboration, non-promotable under this lifecycle**. Its score may be cited with its exact X-state lineage, but it is never counted again as run5, never converted to v3, and never used as the source of pointer move 44. If run5 succeeds under Y, run5's distinct receipt is the only completion/pointer source.

## Clause → code → test

| Normative clause | Landed enforcement at `ea09174b` (unchanged through reviewed HEAD) | Positive / negative evidence | Verdict |
|---|---|---|---|
| Pr12: completion revalidates all non-timing gates and current identity; intent and authorization remain immutable history. | `complete_first_fire_intent` calls `_pf_completion_facts`, then `validate_prefire_intent`, then score admission; `_pf_validate_completed_seal` repeats the current-intent and custody checks. | `test_completion_runtime_custody_refuses_each_join_drift`; `test_completion_seal_rechecks_custody_and_refuses_unequal_objects`; live run4 completion facts PASS while its stale intent separately REFUSES. | **RATIFY** |
| Pr16: definition identity and consumer implementation history are separate; every intent pins both the definition parent and exact latest row. | `_pf_freeze_history` validates the typed allowlist, ancestry, unique sorted batch/fix ids, definition parent, previous-row chain, nonempty tests, and implementation manifests; `_pf_contract` requires exact equality with `amendments[-1]`. | `test_consumer_fix_batch_keeps_definition_and_refuses_stale_latest`; `test_consumer_fix_typed_refusals_with_real_git`; live rows `[0..4]` validation PASS. | **RATIFY** |
| Pr16: completion joins locally recomputed content identity to context, request, argv, result, and retained worker provenance. | `validate_first_measurement_runtime_custody` recomputes the local manifest, checks the content flag once in argv, and projects the same 48 rows onto the retained root; `_pf_completion_facts` installs the identical custody object in the direct leg. | `test_content_pin_accepts_relocation_but_refuses_one_changed_byte`; `test_run3_retained_provenance_projection_and_terminal_refusal`; live run4 dry custody PASS. | **RATIFY** |
| Pr12: reserve once before spawn; failures and ambiguity never reset or reuse nonce/job/output. | `check_first_measurement_unconsumed`, `reserve_first_measurement`, and forward-only consumption transitions use the canonical per-nonce store and refuse reused job/output paths. | `test_nonce_single_use_and_forward_transitions`; `test_missing_authorization_refuses_before_subprocess`; `test_unregistered_call_cannot_transition`; current v2–v4 shared nonce remains consumed by v4/run2. | **RATIFY** |
| Pr10/pr16: the rule is prospective; consumer batching changes representation, not freshness strength. | Existing exact-latest-row equality already implements NO GRACE at dispatch and completion. No source delta is required. | Live v4 REFUSE / v5 PASS is the real two-sided control. | **AMEND text only** |

## ffi6 and freeze-chain audit

### Provenance pins

| Object | Verified identity |
|---|---|
| pr12 memo | `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc` |
| pr16 memo | `cb678491cc6859f709828017d0ed76f35d7a4143855c5e56e42690a5a8b0cf15`, 24,850 B |
| ffi6 memo | `5c52aa5f0f6ba990a14e01ec4e7fbb4095eb7a42433b6a6ba0c392ccd4922240`, 12,592 B |
| ffi6 implementation | `ea09174bbc5f9d990e4097a0c5e1756906ddd9b3`; reviewed source/test files have no diff from that commit through HEAD at the audit cutoff |
| landed manifest | `79f2677a6025d73f87423c9f36bc91b20b43b0d9a0469cf7bdfd5c6d89a76d3f`, 2,566 B, 17 sorted rows |
| current freeze | `0f392923a8be67b930f9b3b7494bace1ee8849ba2ae911f7204dd509224a2390`, 17,633 B |
| run4 receipt | `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run4/MODAL_REMOTE_RESULT.json`, `6a7267f4efe7ab54edc6e556030dbfa174d6d027e3cb6b008ad159cce51c0136`, 242,556 B |
| move 43 | commit `48109233ecd7387db2a8a1d33484a60b9fa2fc33`; archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`, 180,466 B |

The canonical digests of freeze rows `[0..4]` are, in order:

1. definition row `[0]`: `f51c6729760705855e812c01da307a1aed1298dd14581d38c50419e7e41cae1d`;
2. legacy implementation snapshot `[1]`: `ecc62f5b9cb136f8dc9d1bef92bae743e9ddf9028f9536cc1f86fbf3de5d1b72`;
3. legacy implementation snapshot `[2]`: `0651e7360f4a95a8b5618355c40a30ad15bfce5ea4ae65cdf4917ed9e807be77`;
4. legacy implementation snapshot `[3]`: `86f452da106d023173fcd3882bc2ab080640be531d8649d0f80047ef705b46b2`;
5. typed consumer row `[4]`: `e77ed88e2f28e69246087310c5224c1bb0085a2f780bc75693a98cfa368b9cb3`.

Row `[4]` has `definition_change=false`; its `definition_parent_sha256` equals row `[0]`; its `previous_row_sha256` equals row `[3]`; all seven fix ids are sorted and unique; every cause commit is on the required ancestry chain; and its adjudication memo is the exact pr16 blob present at the implementation commit. Every manifest entry matches both `git show ea09174b:<path>` and the current live file, including `src/tac/candidate_seal.py`.

Commit `3dfbd36eb9747e99146fcfbd8d380ec2648081d8` put the draft under an unconsumed top-level `consumer_fix_batches` key instead of `amendments`; it was not a valid or authorizing freeze row. Corrective commit `014ca7bf4` removed that mis-key and appended row `[4]` to `amendments`. A semantic comparison proves rows `[0..3]` are unchanged across the incident. I ratify the current amendment history and the corrective append, not the inaccurate “FREEZE APPEND” description on `3dfbd36e`.

No digest definition changed. In particular, the pr14 timing-risk definition remains row `[0]`; runtime content identity remains the content-only tuple over the same 48 files; candidate archive SHA remains a byte identity rather than a policy digest; and the contest score equation is untouched. This clears r9m's no-digest-change boundary.

## Run4 retained row

Run4's immutable lineage is intent v4 digest `ad149549d58c06eca96ee999cc8b4eed7ca0079a888a494f292bc923c1ce02ff`, authorization v7 digest `f7e229a17d05cd216faa943e6244b54a4f2cbcc39b6e43b29e9207ac81ca6b7e`, nonce `2ca56bcc195e2749b44a4c2930953bce3cc8cd144877a73ad0780b6029f44c68`, job `ddm_rlc5_first_measurement_t4_run4_20260910`, and call `fc-01M26JVVPJX57J7YTYENQZXW6A`.

The retained receipt reports n600, Tesla T4, `d_seg=0.00010345`, `d_pose=0.00000459`, archive 180,406 B with SHA `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`, canonical `S=0.13724490417134017` (JSON binary display `0.1372449041713402`), inflate 1,124.585706526 s, evaluate 40.717213193 s, and canonical ledger elapsed 1,184.731982521 s. Its generated raw is byte-identical to move 43's, while its archive is a distinct 60-byte-smaller object.

Fresh custody revalidation found 48 runtime files, local/worker content digest `e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf`, and retained-root tree digest `8d31edd6f7c3291b37b87eb8404a5944713a80a1d0e4a66fb66bab7650599dbc`. The canonical call ledger has a full dispatched row 953 and harvested row 954. The central nonce record is `HARVESTED` and exactly pins the receipt hash and call.

These facts make run4 a real exact result, not a completed candidate. `score_claim=false`, `promotion_eligible=false`, and `adjudication_required=true` remain binding until a current lifecycle creates and validates v3. This memo does not do so.

## Custody audit — intents v1–v5, authorizations v1–v8, runs 1–5

| Lifecycle | Authorization / nonce disposition | Provider, claim, and ledger disposition | Gap / verdict |
|---|---|---|---|
| Initial/run1; intent v1 then v2; auth v1–v3 | Auth v1 nonce `791384cd…` was never reserved. Auth v2–v3 share nonce `1ad3a6b1…` and job with v4; the later v4 reservation permanently consumes that equivalence class. | Only typed pre-spawn drift/replay refusals are retained; no call id or canonical provider row exists. The 20:08 claim says `dispatched`, but no provider-dispatch receipt supports that word. | **AMEND:** append a terminal no-provider-call correction to the claim ledger; do not edit the old row. |
| Run2; intent v2; auth v4 | Central and output nonce records remain `RESERVED`; reuse is refused. | The fire path refused before Modal because `FIRE_MANIFEST` lacked `axis`; no call id or call-ledger row is expected. A later claim marks stale/superseded/no dispatch. | Custody sufficient for refusal; permanent reservation is correct. |
| Run3; intent v3; auth v5–v6 | Auth v5 nonce `bf34ec2e…` was never reserved and its job/output collide with the executed lifecycle. Auth v6 nonce `9ced38b3…` remains `RESERVED` and non-replayable after the terminal provider failure. | Call `fc-01M26G7JYMY39TVN1ET3JJV2YD` has canonical dispatched/failed rows 951–952 and terminal claim rows. | **AMEND:** `RUN3_RECONCILIATION.json` is schema `first_measurement_run_reconciliation.v1`, not required `candidate_first_measurement_terminal_reconciliation.v1`; it lacks exact intent/auth file references, explicit `nonce_reusable=false`, claim-closure custody, and hashes/line identities for the two canonical rows. No linked `reconciled_terminal_failure` ledger event exists. |
| Run4; intent v4; auth v7 | Nonce `2ca56bcc…` is `HARVESTED`; receipt/call binding is exact and non-replayable. | Call `fc-01M26JVVPJX57J7YTYENQZXW6A` has canonical dispatched/harvested rows 953–954 and terminal claim rows. | **REFUSE completion:** append a claim/P0 correction replacing the stale future action `pending_v3_completion` with retained corroboration + run5 as the only current completion route. Never edit the old rows. |
| Run5; intent v5; auth v8 | At cutoff nonce `f28b0d3b…` is `SPAWNED`, bound to fresh job/output and call `fc-01M26NNV3WR2XXXV914S8BTDR4`. Intent v5 validates against row `[4]`. | Active claim and canonical dispatched row 955 exist; the detached poller is registered. No terminal result existed at cutoff. | **OPEN, not defective yet:** harvest once, then complete only if the exact retained receipt and every current check pass. |

All five intent files and all eight authorization files are committed at the reviewed HEAD. No authorization object was edited or marked “superseded.” The absence of nonce files for v1/v5 is not permission to use them: current contract drift, shared job/output history, and the latest-row check independently refuse them. The four existing central nonce files are forward-only and account for every reserved lifecycle.

The stale claim/P0 prose is a reporting gap, not proof of an extra call. The canonical call ledger is authoritative for provider dispatches: it has run3, run4, and the current run5 dispatch, and it has no run1/run2 call. The bounded audit did not find evidence of another rlc5 provider call in the canonical ledger or retained run directories.

## Cost and lower-score routing

NO GRACE costs one replacement fire whenever a consumer-fix batch lands after an intent was authorized and before that lifecycle completes. In this instance, run5 is that replacement. The cost is real: run4 measured a 1,184.731982521 s Modal wall `[contest-CUDA T4 n600]`, so another wall of that order is a **projection**, plus its provider charge; run5 was separately authorized under the first-measurement cost ceiling.

The cost-control rule is procedural and does not weaken freshness:

1. Land every known required consumer fix in one typed, reviewed batch before emitting the next intent.
2. Once a fire is active, defer a merely optional/non-safety consumer landing until terminal completion. Never defer a fix needed to make the receipt truthful or completable.
3. If a required fix lands in flight, harvest the old call for evidence, wait for single-flight to clear, and issue exactly one new lifecycle against the latest settled row.
4. Do not repeat the new fire for each fix item inside the batch. The unit of invalidation and cost is the appended implementation row.

This is the shortest honest path to the lower exact score: it buys one current-state row rather than spending engineering time on a grace mechanism whose only function would be to promote evidence produced under an obsolete contract snapshot.

## Host validation

- `git diff --exit-code ea09174b..HEAD --` over `candidate_seal.py` and the four mandated test files: PASS; reviewed object unchanged.
- Fresh `_pf_freeze_history` over rows `[0..4]`: PASS; definition and chain digests above.
- Fresh 17-row manifest comparison against live files and `git show ea09174b:<path>`: PASS.
- Fresh intent controls: v4 `PREFIRE_CONTRACT_DRIFT_REFUSED`; v5 PASS.
- Fresh run4 `validate_first_measurement_runtime_custody`: PASS across 48 files.
- Fresh run4 `_pf_completion_facts`: PASS; no seal or other output written.
- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py src/tac/tests/test_decode_wall_clock.py -q`: **231 passed in 6.32 s**.
- `.venv/bin/ruff check` on the reviewed source and those four test files: **All checks passed**.

These are host-side contract and custody checks. I did not rerun a scorer, decoder, archive builder, or exact evaluator, and I did not infer run5's outcome.

## RECALL EVIDENCE

I read the complete pr12 memo and pr13–pr16 ruling chain, ffi6 memo, current freeze, landed implementation manifest, `ea09174b^..ea09174b`, current consumer/test source, intents v1–v5, authorizations v1–v8, run1–run5 retained control files, run3 reconciliation, nonce store, active claims, canonical call ledger, P0 ledger, hot state, and canonical pointer. I also searched the full `.omx/research/` corpus by content for `stale intent`, `latest row`, `consumer fix`, `grandfather`, `as-dispatched`, `fresh authorization`, first-measurement nonce/job/output lineage, and freshness at consumption; 118 files matched the broad search and the relevant non-seed results were inspected. I generated the canonical equation registry and searched all 483 rows for this policy surface.

Beyond the charter's named seeds:

1. `FEED-603-staleness-confound-law` in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` says producer input changes create mixed-state evidence and requires consumer-side fail-closed verification at read time; production stamps alone are insufficient. That supports revalidation, not grandfathering.
2. `ddm_pr10_second_family_review_quiesced_timing_rule_20260910.md` refuses retrospective reclassification after observed receipts and requires the authority rule to be frozen before launch. Its mechanism differs, but its prospective-rule discipline directly applies.
3. `ddm_ri1_record_integrity_read_path_20260802.md` asks what a consumer sees when a claim is false versus true and makes the read path replay the writer's admission predicate. That supports using current completion consumers, not an old producer's self-description, as the authority gate.
4. `ddm_r9m_first_contest_cpu_row_20260804/FIRST_OWN_VEHICLE_CONTEST_CPU_ROW_20260804.md` preserves runtime-files digest custody as an identity fact. The present ruling changes no digest or scoring relation and therefore does not cross r9m's boundary.
5. The 483-row equation registry contains no canonical equation that grants or prices a stale-intent grace. This procedural authority question is settled by the contract lineage and real controls, not by inventing a numeric freshness threshold.

Within that bounded corpus and the named live stores, I did not find an established precedent that upgrades a pre-fix first-measurement receipt into a post-fix completed seal. Legacy grandfather rules elsewhere apply to different schemas and do not override pr12/pr16's exact latest-row clause.

## Boundaries

- Review only. The only research deliverable changed by this arm is this memo; the charter-required checkpoint appends through the canonical checkpoint writer.
- Code, tests, freeze, intents, authorizations, claims, ledgers, pointer, and every `/Volumes/...` artifact were read-only. I launched no Modal call and no fire. Run5 was independently launched by MAIN before this ruling completed.
- Existing unrelated worktree changes and the shared staged index were not touched.
- Run4's exact score is reported from its retained receipt and canonical ledger, not remeasured here. Run5's score remains unknown at the audit cutoff.
- The current typed row changes consumer implementation history only. No timing-risk, runtime, archive, scorer, score, or canonical-equation digest was redefined.
- No candidate seal, packet, submission, upstream write, or pointer move was made.

Frontier unchanged. The canonical pointer's effective/contest-CUDA fields resolve to the move-43 source line: **“S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]”**; archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`.

## NEXT_IF_RESUMED

- **HARVEST-THEN-COMPLETE-OR-REFUSE — owner MAIN; consumer store `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/` → a v3 seal, canonical call/evaluation ledger, and pointer packet; trigger: call `fc-01M26NNV3WR2XXXV914S8BTDR4` reaches one retained terminal receipt.** Revalidate intent v5 and all completion custody from disk; if every check and score bar pass, use run5 alone for v3/pointer move 44, otherwise retain a typed refusal. Append the run4 corroboration disposition to claims/P0 in the same handoff; never complete or score run4 twice.

## LIVE-HYPOTHESES

- Because run5 carries the exact same 180,406-byte archive as run4 and the deterministic evaluator inputs are pinned, its evaluator components should reproduce run4 closely enough to clear the move-43 bar. This remains a hypothesis until run5 is harvested and parsed under the current state.
- Batching consumer fixes before authorization, then holding optional landings until terminal completion, should make the one-refire penalty rare without introducing a grace path. This is a prospective operating hypothesis, not measured cost history.
- A conforming run3 reconciliation can be appended without changing any historical object because all named retained artifacts and the two canonical ledger rows still exist. Provider/claim terminal identity and exact row hashes must be rechecked at that future write.

## DEAD-ENDS

- Grandfathering run4 with a typed provenance link is closed: it weakens exact latest-row freshness after the receipt was observed and contradicts pr16's explicit fresh-intent/fresh-authorization requirement.
- Reusing run4's nonce, authorization, job, or receipt as run5 is closed: the nonce is permanently `HARVESTED`, and immutable lifecycle identities cannot be upgraded.
- Calling run4 fake or discarding it is closed: the exact receipt, canonical call rows, 48-file custody, and completion-facts replay pass. Its proper status is real retained corroboration, not promotion authority.
- Treating identical generated raw as identical archive custody is closed: move 43 and run4 have different archive hashes and sizes even though their evaluator outputs match.
- Treating `RUN3_RECONCILIATION.json` as pr16-complete is closed: its schema and fields differ from the literal requirement, and the linked `reconciled_terminal_failure` event is absent.
- Treating the `3dfbd36e` mis-key as an amendment append is closed: it added no row to `amendments`; only the corrective `014ca7bf4` state is ratified.
- Relaxing the latest-row check to save one paid fire is closed: cost pressure may change batching and landing order, not the authority predicate.

<!-- # FORMALIZATION_PENDING: review-only prospective NO-GRACE clarification; existing code already enforces exact latest-row refusal, this memo creates no measured row or digest definition -->
