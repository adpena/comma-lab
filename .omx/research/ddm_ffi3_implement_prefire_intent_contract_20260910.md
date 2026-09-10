# ddm_ffi3 — bounded prefire intent implementation

Date: 2026-09-10. Axis: `[source inspection; scorer-free unit fixtures]`.
`research_only=true; score_claim=false; promotion_eligible=false`.
Tags: `[no-triality] [p0-ledger-ok]`. Checkpoint owner: `ddm_ffi3`.

Implementation is prepared as one code-and-test serializer unit. Actual commit custody, the freeze
status, and any landing blocker are recorded in `ddm_ffi3_20260910/DELIVERY.json`; this memo alone
is not a freeze or permission for RLC2 to resume. No exact score was measured or moved.

## Contract and decisions

Reference: `ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md`, SHA-256
`50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`, plus both MAIN
clarifications in the ffi3 bounded charter. Their verbatim text and file hashes are retained in
`ddm_ffi3_20260910/CONTRACT_INPUTS.json`. They remain described as pending family ratification,
exactly as MAIN's charter says; that is not another authorization gate for this implementation.
No same-new-object contradiction was found in this implementation review. No STOP interpretation
from ffi1 or ffi2 was reintroduced.

The two clarified decisions are implemented together:

- Authorization binds `intent.file_sha256`, `intent.file_bytes`, and `intent.digest`, independently.
  File-byte hashing is never substituted for canonical self-omitting intent hashing.
- New references require `{path, bytes, sha256}`. Legacy direct-leg interiors retain their old
  `{path, sha256}` receipt references and their unchanged validator. Completion retains the direct
  leg as a separate JSON file and binds it through `decode_wall_clock_reference`, while preserving
  the inline legacy `decode_wall_clock` object required by normal seal consumers.

The required additional surfaces are the dedicated authorization CLI, CUDA worker, poller,
recovery consumer, and a small read-only Git object reader. Git custody and the provider liveness
query run in-process so that refusal checks do not spawn either Git or a provider CLI. The Git
reader supports loose objects and both Git pack delta encodings, verifies object IDs, and does
not mutate an index. Live committed blob parity was checked independently against read-only Git.

## Per-clause compliance table

Locations below are source symbols, so they survive line-number shifts. Tests are in
`src/tac/tests/test_candidate_prefire_intent.py` unless another file is named. A fixture PASS is
implementation evidence only, not a real producer or score claim.

| pr12 clause / sentence | Decision | Code location | Test / evidence |
|---|---|---|---|
| 76–81: omit exactly the named self field; canonical UTF-8 JSON | Keep nested hashes; reject duplicate keys and non-finite JSON | `candidate_seal.prefire_digest`, `_pf_read` | `test_canonical_digest_is_not_file_digest`, `test_duplicate_json_and_nested_false_authority` |
| 84–86: every new JSON/manifest reference has path, bytes, SHA | Re-read bytes and parse named semantics | `prefire_file_reference`, `_pf_ref` | `test_evidence_byte_drift` |
| ffi3 Clarification 1: authorization binds BOTH representations | Require all three independently | `validate_first_measurement_authorization` | `test_authorization_binds_all_three_intent_values` (three mutations) |
| ffi3 Clarification 2: new-object references; unchanged legacy interiors | Retain/hash the leg FILE; do not rewrite its nested references | `complete_first_fire_intent`, `_pf_validate_completed_seal`, `validate_prefire_risk` | `test_completed_fixture_seal_uses_unchanged_direct_builder`, `test_risk_refuses_forged_arithmetic_but_keeps_legacy_leg` |
| 88–100: reachable full implementation commit, committed/live source equality, sorted complete manifest, memo pin, prospective ordering | Check committed blobs and ancestry; source commit and production timestamp precede intent | `_pf_contract`, `_pf_git`; `git_custody_read.GitCustodyReader` | `test_contract_live_source_drift`; `test_git_custody_read.py` loose/pack/ancestry controls |
| 102–191: exact intent top-level shape/state/typed fields | No additional top-level keys or placeholders | `_pf_schema` | `test_schema_refusals` |
| 197–201: intent has no candidate timing or score/promotion authority, no waiver | Reject authority keys recursively; first mode rejects all waiver/override flags | `_pf_schema`, fire `_first_measurement_main`, make `_prefire_main` | `test_false_authority_refused`, `test_first_fire_cli_override_refuses_before_subprocess` |
| 205: exact archive, ZIP/member validity | Hash actual ZIP, check members and selected member bytes | `_pf_identity`, `_pf_evidence` | `test_fixture_intent_validates_but_is_not_a_seal`, identity/evidence mutations |
| 206–207: named runtime and normalized-receiver digests, receiver pins, pin consistency | Reuse existing digest and pin functions | `_pf_identity` | `test_identity_drift`; existing candidate-seal suites |
| 208–210: complete candidate/frontier public smoke, no first-fire waiver | Reuse unchanged public-smoke validator | `_pf_evidence` → `_public_smoke_problems` | fixture intent/producer controls; existing smoke negative suite |
| 211–213: independent manifest validation and complete dependencies | External verifier, exact full shipped-file map, matching verification reference | `_pf_evidence` | manifest/evidence drift tests |
| 214–216: two full-n600 encodes and selected payload parsed from archive | Distinct retained execution receipts and payload paths; actual payload/member hashes match | `_pf_evidence` | fixture positive and twin/parseback drift tests |
| 217–220: literal full-n600 raw identity, 600 pairs, 3,662,409,600 B, no resume/cache | Bind public argv and cold report; check both retained raw identities and pointer | `_pf_evidence` | fixture positive, raw evidence mutations; raw transport caveat below |
| 221–222: complete rule-118 literal census, no uncovered new path | Re-read CLEAR/118 result and match every runtime file identity | `_pf_evidence` | census/evidence drift tests |
| 223–225: retention, no discarded payload waiver | Require archive, twins, raw, parseback and every declared retained path | `_pf_evidence` | retention/evidence drift tests |
| 226–229: current zero-tolerance pointer and re-derived rate bar | Re-read current pointer and its actual archive size; strict delta comparison | `_pf_pointer` | `test_pointer_drift`, fixture producer/consumer controls |
| 232–268: completed-T4 receiver-delta risk | Validate the source leg unchanged; rederive every normalized endpoint/path | `validate_prefire_risk`, `prefire_receiver_rows` | risk arithmetic/legacy-reference control |
| 269–275: actual refused diagnostics; maximum, fraction, projection and policy | Preserve old verdicts; rederive all arithmetic; require <=1260 <1800 | `validate_prefire_risk` | `test_risk_refuses_forged_arithmetic_but_keeps_legacy_leg` |
| 277–296: RLC2 exact historical instantiation, no sixth timing run | Bind reviewed source-leg bytes/hash, cured receiver and pinned old-rule summary; match refused diagnostic rows | `validate_prefire_risk` RLC2 branch | Source inspection only for real RLC2 instantiation; no real producer claim |
| 298–321: separate MAIN authorization producer, no typed actor/nonce flag | Dedicated CLI derives candidate, policy, receipt, nonce; no `--authorized-by`/nonce option | `authorize_candidate_first_measurement.main`, `build_first_measurement_authorization` | authorization fixture and three-field negative controls |
| 323–326: exact auth blob in HEAD/main and intent/implementation ancestry | Check all committed blobs and ancestry; record containing HEAD at consumption | `validate_first_measurement_authorization`, `reserve_first_measurement` | authorization controls; Git byte/ancestry tests |
| 326–330: current provider prices, all charged resources, strictly less than USD 5 | Re-read provider source fields; exact coverage of declared charged resources; compute every subtotal; bind one T4, four CPUs, 16 GiB hard caps | `_pf_cost`; worker `main` → `with_options` | `test_cost_fails_closed`; source inspection of installed SDK request/limit handling |
| 332–341: derived nonce, atomic RESERVED before spawn, SPAWNED/HARVESTED only forward; no retry/reset | Locked exclusive reservation, call-ledger join, durable forward state transitions | `first_measurement_nonce`, `reserve_first_measurement`, `transition_first_measurement` | `test_nonce_single_use_and_forward_transitions`, `test_unregistered_call_cannot_transition` |
| 343–383: exact mutually exclusive lifecycle flags; emitter self-validates/deletes rejected new intent | Preserve legacy modes; first mode uses only pr12 flags | make `build_parser`, `_prefire_main`; `build_prefire_intent` | `test_intent_producer_emits_and_self_validates`; existing make-seal tests |
| 385–400: exact authorization CLI and commit before fire | Derived object only; consumer requires committed bytes | authorization CLI and validator | authorization tests; no real MAIN authorization made |
| 402–419: exact first-measurement mode, forbid every retyped owned value; dry-run consumes nothing | Reject overrides even at their defaults; dry-run writes no state | fire `_first_measurement_main` | CLI override, missing auth, dry-run tests |
| 420–424: validate all gates, maturity, active MAIN claim, single flight, prices, nonce and paths before subprocess | In-process Git and SDK liveness; unknown state refuses; retain existing worker guards | fire `_first_measurement_main`, `first_measurement_cloud_apps` | `test_unknown_or_busy_cloud_refuses_before_dispatch`, `test_cloud_preflight_uses_sdk_without_subprocess` |
| 425–432: explicit worker timeouts/resources, immutable snapshot, 4800 cap, 5400 poller | Explicit `build_dispatch_argv`; snapshot before worker; local custody bound to CWD; durable poller with fixed deadline | fire `build_dispatch_argv`, `_first_measurement_main`; worker `main` | dry-run argv test; `test_worker_local_custody_survives_snapshot_import`; existing axis tests; timeout harvest mutations |
| 434–439: worker/poller quarantine, retained payloads, no normal mirror | Flags remain false through worker, poller and recovery; block mirror/pointer staging; persistent-volume inputs/raw/logs plus hashed inventory | worker `run_auth_eval`, `retain_first_measurement_worker_tree`; poller/recovery checks | `test_worker_wrapper_retains_inputs_raw_and_quarantines`, `test_remote_retention_keeps_every_materialized_file`, `test_recovery_and_mirror_cannot_unquarantine` |
| 451–462: consumed auth/call/request/result identity → unchanged direct leg → cold n600 <=1260 → all live non-timing/pointer gates | Preserve this producer order | `_pf_completion_facts`, `complete_first_fire_intent` | passing completion fixture and eight harvest mutations |
| 463–466: exact evaluator components and strict frozen score delta | Use inner evaluator components; check outer equality and archive bytes; ignore rounded final score | `_pf_completed_score` | score/result harvest mutations |
| 466–473: new v3 seal, exact references, normal consumer, immutable intent/auth | Write new leg and seal; validate all v3 bindings through normal consumer; reject overwrite | completion producer and `_pf_validate_completed_seal` | completion fixture verifies immutable intent, v3 consumer and legacy leg equality |
| 475–501: fifteen typed refusal conditions and false-authority receipts | Typed exception codes; CLI stderr and retained refusal files beside objects/output | `PrefireRefusal`, `write_prefire_refusal`, CLI consumers | Schema, authority, contract, identity, evidence, pointer, risk, authorization, replay, argument, lane, timeout, warm, timing policy and result negatives |
| 523–543: prospective committed freeze and real positive control | Implementation/freeze custody must land before RLC2 resumes; fixtures release no fire | `PREFIRE_CONTRACT_FROZEN.json` and DELIVERY custody state | Real control remains unexecuted; see next section |

## Validation and limits

MEASURED: 67 new scorer-free unit tests pass. The broader 345-test selection has 341 PASS and
four permission failures: three existing `test_decode_timing_concurrency.py` tests require `ps`,
and one existing `test_modal_auth_eval.py` test binds a Unix socket. The sandbox denies those
operations with `Operation not permitted`. They were not skipped, weakened, or relabeled PASS.
The worker's legacy-path literal regression found during review was fixed by preserving its
original normal branch. Ruff is clean. Exact commands and outputs are retained in
`ddm_ffi3_20260910/PYTEST_FINAL.txt`, `PYTEST_NEW.txt`, and `RUFF.txt`.

The four specifically required suites contributed 156 PASS and three sandbox `ps` failures in
the initial required-suite run. A fully green run of those suites is therefore still an unmet
landing requirement in this sandbox. The broader selection adds a separate socket limitation.

Two final review passes for every changed Python file are recorded in `REVIEW_FINAL_1.txt` and
`REVIEW_FINAL_2.txt`, with source hashes in `REVIEW_RECEIPT.json`. The earlier review record is
kept too; its missing legacy-leg FILE reference finding was fixed and the affected review
counter restarted. Final review also fixed the normal-consumer legacy-exception refusal and
the local worker custody root under immutable-snapshot imports; both have regression coverage.
No Python review override was used.

**Fixtures do not close the pass path.** Git history and the 3.6GB raw transport in the lifecycle
fixtures are substitutes, expressly identified in the test module. Archive/member/runtime
hashing, small evidence reads, nonce persistence, the legacy T4 builder, and the v3 consumer
execute real local code. Separate Git tests exercise real loose/packed object bytes and live
committed custody. Worker-wrapper and SDK tests mock the provider/expensive inner work; no
provider or scorer ran. They cannot establish RLC2 raw identity, receiver admissibility, real
spend, cold T4 decode time, exact score, or a passing real completed seal. The mandatory real
control remains RLC2's producer followed by MAIN's authorization, one paid fire and real harvest.

## Evidence adapter details for the real producer

Pr12 fixes intent/risk/authorization names but leaves some nested evidence receipt layouts
unspecified. The stricter implementation requires these explicit data fields; it does not
accept a success boolean in place of their byte bindings. The complete executable fixture is
`test_candidate_prefire_intent.py::fixture`, expressly synthetic.

- Each manifest, verification, twin, parseback, raw and census receipt binds
  `archive_sha256`, `runtime_sha256`, `receiver_sha256`.
- The external JSON candidate manifest lists all runtime files as `relative_path/bytes/sha256`
  and records `producer_source_commit` and `production_started_at_utc`. Verification names
  that exact manifest and both complete-hash/dependency results; the validator also redoes the
  file-map comparison itself.
- Twin receipts contain two payload references and two distinct `executions` references.
  Each execution records its ID, command, completed n600, frozen producer commit and payload.
  Parseback identifies the real archive member by `name/bytes/sha256`.
- Raw identity contains literal public `command`, `candidate_public_stdout` reference, pair/sample
  counts, resume/cache facts, and `candidate_raw`/`pointer_raw` references with the exact full byte
  count. The receiver's cold report is parsed from the retained stdout.
- Census carries `rule=118`, `verdict=CLEAR`, `complete=true` and exact full `files` coverage.
  Retention carries a `payloads` reference array, including both twins and parseback evidence.
- Receiver delta `files` enumerates the full normalized path union, with source/candidate
  `[bytes, sha256]` endpoints (null for a missing side) and both endpoint digests. Normalized
  rows are checked against the unchanged digest function.
- Price evidence is machine-readable and retained: provider URL, fetched UTC, its file reference,
  `chargeable_resources`, and per-resource `price_field` paths. Cost rows name quantities,
  unit-second prices and exact computed subtotals. Every provider-declared charged resource
  must be included; the worker hard-limits compute at one T4, four CPUs and 16 GiB. Unknown
  pricing, uncovered storage/transfer charges, stale sources or non-strict totals refuse.
- Completion writes `<seal-name>.decode_wall_clock.json` and references those exact legacy bytes
  in `decode_wall_clock_reference`. This is the operational consequence of Clarification 2.

## RECALL EVIDENCE

Searched the Codex memory registry for `prefire|ffi3|common.contract`: no relevant match in that
scope. No memory-derived project fact was promoted to current authority.

Searched the complete `.omx/research/` Markdown/JSON corpus by content for
`candidate_prefire|first.measurement|t4_direct|gate with no door`; inspected the matching
pr12/ffi1/ffi2/RLC2 material, PR9 findings, current smoke/worker/recovery code and the live board.
Searched docs/design, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` and the task ledger for
`prefire|first.measurement|intent.contract`. Generated the canonical equations registry via
`.venv/bin/python tools/list_canonical_equations.py --json` and filtered the same contract terms;
no applicable equation supplied an alternative first-measurement authorization lifecycle.

Beyond the charter seeds, PR9's stale-manifest finding reinforced independent full file coverage;
existing recovery code could restore score authority from a nested evaluator artifact, which
added the recovery quarantine guard. The current worker discarded remote-local raw custody at
container lifetime, so first measurement now writes inputs/work directly to a persistent volume
and records every retained file, blocking uncertified cleanup. The installed Modal SDK source
showed that scalar memory/CPU requests are not hard caps; request/limit tuples close the cost
binding. Existing cloud liveness shells out, motivating the in-process SDK query under the
strict pre-subprocess rule. The older DAG matches concern other training events and do not
supply a replacement contract. This is a scoped recall result, not a global nonexistence claim.

## Boundaries and solver-stack disposition

No Modal/provider call, paid fire, encoder/decoder, timing window, timing producer, scorer,
evaluator or real n600 pass ran. No candidate was materialized. No `upstream/`, PR tree, sealed
candidate, rlc2/sj1 live directory, completed `decode_wall_clock.py` requirement, protected common-
contract file, frontier pointer or shared staged index was edited. Existing lane/resource/spend
guards remain, with a stricter first-measurement branch. Unrelated dirty work is preserved.
No existing payload was moved, deleted, symlinked, or discarded. Only source, small fixtures,
review/test records and serializer custody artifacts were created. No MPS authority was used;
no GT decoding occurred. No new catalog number was required or claimed.

Sensitivity-map, Pareto, bit-allocation and posterior hooks are N/A: this is a research-only
admission/custody change with no new score or actuator measurement. The dispatch hook is the
existing fire CLI's new explicit mode. A disambiguator is N/A because both disputed clauses now
have binding answers; their implementation choices and tests are recorded above.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store:
  `ddm_ffi3_20260910/DELIVERY.json`, serializer landing artifacts, this compliance memo, and
  `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; fire trigger: harvest this
  implementation packet. Resolve any recorded landing/permission blocker, run the unchanged
  required suites in a permitted environment, verify exact committed bytes, and commit the
  real freeze receipt before releasing the already-queued RLC2 producer. No paid fire is
  authorized by this implementation packet.
- **FOLDED** — owner: RLC2 producer / MAIN; consumer store: the existing RLC2
  `CONTRACT_DECISION_FIRE_ORDER.json` and committed intent/authorization/receipt chain;
  fire trigger: a real committed freeze plus the existing pr12 admission gates. The real
  producer, single authorization/fire and harvest control remain in that existing order;
  fixtures do not replace it or create an additional candidate-family permission.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- RLC2 may pass the real intent producer after adapting its retained receipts to the frozen
  evidence fields; the reviewed receiver-delta chain makes this plausible, but it was not run.
- A real cold T4 harvest may complete v3 if every frozen gate and the score bar survives; the
  unchanged direct builder passes the synthetic control, which is not real-path proof.

## DEAD-ENDS

- One digest standing for both intent file bytes and canonical identity: closed by Clarification 1.
- Rewriting legacy timing receipts to add nested byte counts: closed by Clarification 2; bind
  the unchanged leg's file instead.
- Reusing a reserved nonce or treating an ambiguous spawn as permission to retry: closed by the
  durable one-way consumption record and real call-id join.
- Promoting a worker result, recovery artifact or fixture before completed v3 adjudication:
  closed by the quarantine consumers and pr12's real-positive-control requirement.
