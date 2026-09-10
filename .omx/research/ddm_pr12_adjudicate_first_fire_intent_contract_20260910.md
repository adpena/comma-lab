# ddm_pr12 — adjudication of the first-fire intent contract

Date: 2026-09-10  
Axis: `[review; scorer-free; retained-receipt re-derivation; contract specification]`  
`research_only=true; score_claim=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**APPROVE-WITH-CHANGES.** A separately typed pre-fire intent is a sound way to break the
seal-before-receipt cycle, but the proposal is not safe enough as written. The candidate producer must
not be able to authorize its own exception; a first-measurement dispatch must be one-shot; the worker
result must remain non-promotable until harvest completes the unchanged `t4_direct` validation; and the
dispatch must carry explicit rather than defaulted timeout and spend bounds.

The approved lifecycle has three distinct objects:

1. `candidate_prefire_intent.v1`: an immutable non-seal that freezes every candidate gate except a
   candidate timing leg. It carries no candidate timing authority, no candidate score, and no promotion
   permission.
2. `candidate_first_measurement_authorization.v1`: a separately committed, one-shot authorization made
   by MAIN after the intent exists. It binds the intent digest, candidate, lane, job, axis, receipt
   destination, resource class, timeout values, and spend ceiling.
3. `candidate_seal.v3`: created only after harvest. It names the exact intent, authorization, and remote
   receipt; contains a `candidate_decode_wall_clock.v2` leg made by the unchanged
   `build_t4_direct_leg`; revalidates every non-timing gate against live bytes and the current pointer;
   and is the first object that may unlock promotion.

`candidate_prefire_intent.v1` is not a pending `candidate_seal`, is not a `decode_wall_clock` leg, and
must be refused by the normal seal validator and normal `--seal` fire path. “No seconds” means no
candidate `measured_t4_decode_seconds`, `projected_t4_decode_seconds`, or timing-clearance assertion.
Historical diagnostic seconds and dispatch timeout seconds may appear only in the explicitly
non-authoritative risk and dispatch blocks below; they cannot satisfy `validate_decode_wall_clock`.

`verdict_scope: FORMULATION` — the exact two-stage lifecycle and refusal rules below.  
`verdict_scope: INSTANCE` — the absent RLC2 candidate, the retained move-40/RLC1 timing evidence, and
the conditional RLC2 arithmetic.

## Findings and evidence

| Claim | Evidence | Adjudication |
|---|---|---|
| The cycle is real. | Current `make_candidate_seal.py` requires one timing option; `build_seal` and `validate_seal(..., require_decode_wall_clock=True)` reject a missing leg; `build_t4_direct_leg` rejects the absent receipt. RLC2 retained all four executed refusals. | `INSTANCE`: no current sealed first-fire path exists. |
| Type separation can preserve DWC1. | `validate_decode_wall_clock` accepts only its declared measured, inherited, and completed `t4_direct` modes. A different top-level schema can be consumed by a different CLI branch without weakening those comparisons. | `FORMULATION`: approve a non-seal intent, not a pending timing mode. |
| The proposal lacks authorization and replay custody. | It says “MAIN authorized” but gives no independently hashed authorization object, one-shot nonce, pre-spawn consumption record, or rule for ambiguous spawn recovery. | `FORMULATION`: approval requires the separate authorization schema below. |
| Current fire defaults are not an adequate binding. | The CUDA worker accepts `--inflate-timeout` and `--evaluate-timeout`, but `build_dispatch_argv` currently passes neither. The defaults happen to be 1,800 s. The Modal function cap is 4,800 s and the default CUDA poller deadline is 2,400 s. | `FORMULATION`: first-measurement mode must pass and record explicit values and use a poller that outlives the worker. |
| RLC2 is not a blind 1,800-second bet if the exact evidence bundle closes. | Move 40's completed cold T4 leg is 990.053829427 s. RLC1 g3/g4, on the exact cured normalized receiver, are refused diagnostics at 829.0324737499905/831.502915124991 s. Relative to the retained 797.1459791249945 s move-40 diagnostic, the conservative observed increment is 4.3099930125356956%, giving 1,032.7250802956457 s. | `INSTANCE`, diagnostic risk only: 227.27491970435426 s below policy and 767.2749197043543 s below the hard timeout. This is not timing clearance. |
| The conditional RLC2 rate row clears its bar if and only if the real prerequisites hold. | With raw identity, `net_dS = 25*(B_candidate-180238)/37545489`. The strict `-2e-5` bar requires an integer saving of at least 31 B, hence `B_candidate <= 180207`. A real 180,178 B archive would save 60 B and give `-3.995153718733028e-5 S`. | `DERIVED, CONDITIONAL`; no RLC2 bytes or score exist. |

The DWC1 guard remains intact because the exception authorizes exactly one measurement, not a timing
claim. A result that takes 1,260.000000001 s, times out, is warm, lacks cold-cache proof, has the wrong
runtime or archive, or lands after pointer drift cannot create a completed seal. Finishing before the
worker's 1,800-second timeout is not the same as clearing the 1,260-second admission policy.

## Provenance pins

| Surface | Exact pin |
|---|---|
| completed `t4_direct` implementation | commit `0524522f024cc30d0bfe0296815cc3d9a2804c3b` |
| DWC1 pre-fire timing guard | commit `86961e48769cadb12ea8f3d39ce314995280b296` |
| frozen v3 sampler implementation | commit `6a857a1ec8dcce53fbc7068df40140a945b3d8be` |
| move-42 pointer landing | commit `d2803c2148b6153e6fc26d18428cd2d3cda3ea3e` |
| RLC2 proposed contract | SHA-256 `021b44b6c4ff0dcad7fc5d7f0cf95645b997c78fda6edb520024dfaacbbc84de` |
| RLC2 STOP memo | SHA-256 `76ba13cf2ef8f91d731a8526d65f3bd96d7d3799c7002e9a99ed56e77fa24caf` |
| current `decode_wall_clock.py` | SHA-256 `f1ce9122a0114e13481eb1f3fe8ba3951b5c0f1549ec49538994393acceb89ff` |
| current `candidate_seal.py` | SHA-256 `d94f479ef13d1cc3a5331403eddf18bc5816c5bbe16aa75772c51d64d33357e6` |
| current seal producer / fire consumer | SHA-256 `7c2aa6aa79c33d8d478e94cb9dda5c752c205ebe399775d2848bdd0f7e019288` / `8bcce76759bd10e4a79620f58223ee8c29426c09d48038ca88e5cde644deacb4` |

## Exact normative contract

The words **MUST**, **MUST NOT**, **REQUIRE**, and **REFUSE** in this section are implementation
requirements. MAIN should implement these names and comparisons verbatim.

### Canonical object hashing

Every object below carries its own digest. Omit exactly the named self field (`intent_sha256`,
`risk_sha256`, or `authorization_sha256`), retain every nested/reference SHA-256, and compute:

```text
sha256(utf8(json.dumps(object with its own *_sha256 field omitted,
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False)))
```

Paths are absolute custody paths. Every referenced JSON or manifest carries `{path, bytes, sha256}`;
the validator re-reads it, requires exact byte count and SHA-256, parses it, and checks the semantic
fields named below. A typed boolean never substitutes for re-derivation from the referenced bytes.

The contract implementation itself is frozen by `contract.implementation_commit` plus
`contract.implementation_manifest`. The manifest is a sorted array of every producer, validator,
fire, worker, poller/quarantine, and completion-consumer source path with its committed SHA-256. The
validator requires all of these:

- the full 40-hex commit is reachable from the current branch;
- each live file hashes to the manifest and `git show <commit>:<path>` is byte-identical;
- the implementation commit is an ancestor of the commit containing the exact intent, resolved from
  Git and later recorded in the authorization;
- the contract manifest digest equals `contract.implementation_manifest_sha256`;
- the adjudication memo is present in that commit at its recorded SHA-256.

A dirty or later-edited contract surface refuses. Unrelated dirty files do not matter.

### `candidate_prefire_intent.v1`

The object has exactly this top-level shape; additional top-level keys refuse in v1:

```json
{
  "schema": "candidate_prefire_intent.v1",
  "state": "PREFIRE_FIRST_MEASUREMENT_ONLY",
  "candidate_id": "<non-placeholder string>",
  "created_at_utc": "<RFC3339 UTC>",
  "created_by": "<producer id>",
  "producer_source_commit": "<40-hex source commit checked out before candidate production>",
  "score_claim": false,
  "promotion_eligible": false,
  "timing_clearance": false,
  "contract": {
    "adjudication_memo": {"path": "<this memo>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "implementation_commit": "<40 hex>",
    "implementation_manifest": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "implementation_manifest_sha256": "<64 hex>"
  },
  "candidate": {
    "archive": {"path": "<archive.zip>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "runtime": {
      "path": "<runtime directory>",
      "digest_definition": "tac.candidate_seal.measure_runtime_digest",
      "sha256": "<64 hex>",
      "file_count": "<positive int>",
      "total_bytes": "<positive int>"
    },
    "normalized_receiver": {
      "digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
      "sha256": "<64 hex>"
    },
    "receiver_pins": [
      {"relative_path": "inflate.py", "bytes": "<positive int>", "sha256": "<64 hex>"},
      {"relative_path": "inflate.sh", "bytes": "<positive int>", "sha256": "<64 hex>"}
    ],
    "archive_member": null
  },
  "admit_bar": {
    "rule": "net dS = dS_rate + 100*(d_seg_new - d_seg_base) + (sqrt(10*d_pose_new) - sqrt(10*d_pose_base)) < threshold",
    "net_dS_threshold": "<finite negative float>",
    "pointer_axis": "contest_cuda",
    "pointer_score_at_intent": "<finite float re-read from pointer>",
    "pointer_archive_sha256_at_intent": "<64 hex re-read from pointer>",
    "pointer_tolerance_abs": 0.0,
    "require_pointer_archive_identity": true,
    "rate_only_precheck": {
      "raw_identity_required": true,
      "normalizer_bytes": 37545489,
      "derived_net_dS": "<25*(candidate bytes-pointer bytes)/37545489>",
      "passed": true
    }
  },
  "public_entrypoint_smoke": {"schema": "candidate_public_entrypoint_smoke.v1", "public_path_probes": "<existing complete block>", "inflate_sh_smokes": "<existing complete block>", "public_path_probe_seconds": "<existing bound>"},
  "evidence": {
    "candidate_manifest": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "manifest_validation": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "twin_encode": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "archive_parseback": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "raw_identity_n600": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "literal_census": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "retention_manifest": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
    "timing_risk": {"path": "<candidate_prefire_timing_risk.v1 json>", "bytes": "<positive int>", "sha256": "<64 hex>"}
  },
  "dispatch_policy": {
    "axis": "contest_cuda",
    "gpu": "T4",
    "scorer_device": "cuda",
    "inflate_device": "auto",
    "inflate_timeout_seconds": 1800,
    "evaluate_timeout_seconds": 1800,
    "modal_function_timeout_seconds": 4800,
    "poller_deadline_seconds": 5400,
    "n_samples": 600,
    "cold_required": true,
    "checkpoint_resume_required": false,
    "token_cache_status_required": "DISABLED",
    "source_snapshot_required": true,
    "claim_policy": "require_active",
    "max_paid_dispatches": 1,
    "currency": "USD",
    "maximum_total_cost_usd_exclusive": 5.0,
    "cost_preflight_max_age_seconds": 86400
  },
  "retained_payload_paths": ["<at least archive, both twin payloads, candidate raw, and parseback>"],
  "falsifiers": ["<at least one candidate-specific falsifier>"],
  "intent_sha256": "<canonical object digest>"
}
```

`archive_member` is either `null` or the existing measured member-pin object; no hand-typed member hash
is accepted. The apparent strings in angle brackets describe types; they are not accepted literal
values.

The intent MUST NOT contain `decode_wall_clock`, `candidate_t4_receipt`, candidate
`measured_t4_decode_seconds`, candidate `projected_t4_decode_seconds`, `timing_passed`, a candidate
score, `score_claim=true`, or `promotion_eligible=true`. It MUST NOT use the public-smoke custody waiver,
an inheritance leg, a local timing leg, or `allow_nonpromotable_lane_id`.

### Non-timing comparisons

The intent producer and consumer both perform the following from disk:

1. Re-hash `archive.zip`; require exact path, bytes, SHA-256, ZIP/member validity, and any member pin.
2. Recompute runtime and normalized-receiver digests with the named functions; require exact equality.
   Re-hash every receiver pin and run `check_pin_consistency`.
3. Re-run the existing candidate/frontier `candidate_public_entrypoint_smoke.v1` validator. Both groups,
   both roles, outcomes, runtime digests, archive hashes, and the current frontier identity must pass.
   No first-measurement waiver exists.
4. Validate the candidate dependency manifest from outside the candidate tree. Its verification receipt
   must say every listed hash passed, every runtime dependency is listed, and the tree digest equals the
   intent. A manifest inside the candidate tree is not allowed to be its own sole verifier.
5. Require two real full-n600 encodes of the candidate field to be byte-identical, require the selected
   encoded payload to be the bytes parsed from `archive.zip`, and require the archive parse-back to name
   the same archive/runtime/receiver identities.
6. Require literal public `inflate.sh` full-n600 raw identity against the pointer output: 600 pairs,
   3,662,409,600 bytes, candidate raw SHA-256 equal to the pointer raw SHA-256, no resume, and no shared
   token cache. This is what reduces the pre-fire admission calculation to rate only. A semantic-table,
   library-only, partial, or hash-without-byte-count comparison refuses.
7. Require the PR9-style complete literal census to bind the exact runtime digest and return CLEAR for
   rule 118. Any new receiver path not covered by the census refuses.
8. Validate the retention manifest, require every named payload still exists at the recorded byte count,
   and require the twin payloads, archive, raw, and parse-back path to be retained. Missing custody
   refuses; no measurement-and-discard waiver exists.
9. Re-read the canonical pointer at intent creation and immediately before dispatch. Require exact score
   and archive identity because tolerance is zero. Recompute the rate-only precheck. `passed` is checked,
   never trusted.

### `candidate_prefire_timing_risk.v1`

This schema is a bounded spend-risk gate, not timing authority:

```json
{
  "schema": "candidate_prefire_timing_risk.v1",
  "mode": "completed_t4_receiver_delta",
  "authority": false,
  "timing_clearance": false,
  "source_t4_leg": {"path": "<candidate_decode_wall_clock.v2 t4_direct>", "bytes": "<positive int>", "sha256": "<64 hex>"},
  "source_receiver": {"sha256": "<must equal source leg receiver>"},
  "candidate_receiver": {"sha256": "<must equal live candidate receiver>"},
  "diagnostic_reference_receiver": {"path": "<retained runtime>", "sha256": "<must equal candidate receiver>"},
  "receiver_delta_manifest": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>"},
  "base_local_diagnostic": {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>", "wall_seconds": "<finite positive>", "authority": false, "actual_verdict": "REFUSED"},
  "candidate_local_diagnostics": [
    {"path": "<json>", "bytes": "<positive int>", "sha256": "<64 hex>", "wall_seconds": "<finite positive>", "cold": true, "n_samples": 600, "authority": false, "actual_verdict": "REFUSED"}
  ],
  "calculation": {
    "candidate_local_ceiling_seconds": "<max candidate diagnostic wall>",
    "local_cost_fraction_upper": "<max(0, candidate ceiling/base wall - 1)>",
    "source_t4_seconds": "<exact validated t4_direct seconds>",
    "t4_risk_ceiling_seconds": "<source T4 seconds*(1+local cost fraction upper)>",
    "policy_limit_seconds": 1260.0,
    "hard_timeout_seconds": 1800.0,
    "passed": true
  },
  "score_claim": false,
  "risk_sha256": "<canonical object digest>"
}
```

Validation requires the source leg to pass the unchanged `validate_decode_wall_clock` as completed
`t4_direct`; its receiver to equal `source_receiver.sha256`; and the diagnostic reference receiver to
equal the live candidate receiver. The delta manifest must enumerate every normalized source/candidate
receiver path and both hashes, and its recomputed endpoint digests must equal those two receivers. A
boolean “same except maps” is insufficient.

Every diagnostic receipt remains at its recorded actual verdict. The validator recomputes the maximum,
fraction, projection, and both comparisons. It requires
`t4_risk_ceiling_seconds <= 1260.0 < 1800.0`. This gate can refuse a spend but can never clear timing.
A future candidate without this exact completed-T4/source-receiver/delta/diagnostic chain requires a new
schema review; `candidate_prefire_intent.v1` has no unknown-receiver fallback.

For RLC2 the required instantiation is:

- source direct leg path
  `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json`,
  1,553 bytes, file SHA-256 `ed929b24cc876bf8ffabc3004b856decbb5e73d0fee13c1d3c87d91d659f9521`,
  receiver `6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf`, and
  T4 seconds `990.053829427`;
- live RLC2 normalized receiver equal to the retained cured-RLC1 reference receiver
  `b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d`;
- RLC1 timing summary `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json`, 3,064 bytes, SHA-256
  `a8d1e1a781a0c2f80962591288dbcc4ed023113e31766baaec368db8ef75c4ed`;
- base diagnostic `797.1459791249945`, candidate diagnostic ceiling
  `831.502915124991`, fraction `0.043099930125356956`, and risk ceiling
  `1032.7250802956457` seconds.

The RLC1 g3/g4 rows remain refused old-rule `[macOS-CPU advisory]` diagnostics. Their use here is only
as conservative spend-risk evidence. They do not become v3 receipts, a local/T4 calibration, or a
candidate timing leg. Requiring a sixth local timing run would violate the explicit suspension and
recreate a gate with no available producer; RLC2 therefore uses this exact delta branch, not a claimed
fresh-v3 branch.

### `candidate_first_measurement_authorization.v1`

MAIN creates this only after the committed intent validates:

```json
{
  "schema": "candidate_first_measurement_authorization.v1",
  "state": "AUTHORIZED_ONCE",
  "authorized_by": "MAIN",
  "authorized_at_utc": "<RFC3339 UTC>",
  "intent": {"path": "<candidate_prefire_intent.v1>", "bytes": "<positive int>", "sha256": "<intent digest>", "commit": "<40-hex commit containing that exact intent>"},
  "candidate_id": "<exact intent candidate id>",
  "axis": "contest_cuda",
  "lane_id": "<promotable lane id>",
  "instance_job_id": "<unique job id>",
  "claim_agent": "MAIN",
  "output_dir": "<durable SSD directory>",
  "receipt_path": "<output_dir>/MODAL_REMOTE_RESULT.json",
  "authorization_nonce": "<sha256(utf8(intent_sha256 + NUL + lane_id + NUL + instance_job_id + NUL + candidate-first-measurement-v1)).hexdigest()>",
  "dispatch_policy_sha256": "<canonical digest of the exact intent dispatch_policy>",
  "cost_preflight": {"path": "<real resource-price receipt>", "bytes": "<positive int>", "sha256": "<64 hex>"},
  "single_axis_waiver_reason": "candidate receiver declares linux-nvidia-t4 and refuses CPU by design; this authorization is T4-only",
  "score_claim": false,
  "promotion_eligible": false,
  "authorization_sha256": "<canonical object digest>"
}
```

The authorization has no self-referential commit field. At fire time its exact blob must already be
present at the same path in current `HEAD`; the fire receipt records that resolved containing commit.
That commit must be reachable from current `main`, contain byte-identically the intent it names, and
descend from the contract implementation and intent commits. The cost receipt must bind
T4, at most 4,800 remote seconds, exactly one paid dispatch, a current provider-price source, and a
finite computed upper bound for all charged resources strictly `< 5.0 USD`. At dispatch require
`0 <= dispatch_time_utc - provider_price_fetched_at_utc <= 86400` seconds. Unknown/stale price, a
non-T4 resource, `<= 5.0` used in place of strict `< 5.0`, or any second paid dispatch refuses.

The producer derives the deterministic 256-bit nonce from the exact UTF-8/NUL concatenation shown; it
accepts no nonce flag. The nonce is consumed atomically before spawn by a durable
`candidate_first_measurement_consumption.v1` record with state `RESERVED`. It transitions to
`SPAWNED` only after a real call id is registered and to `HARVESTED` only after receipt recovery. A
failure or ambiguous state never resets the nonce. After provider reconciliation proves no call was
created, MAIN may issue a new authorization with a new nonce and job id; the old authorization remains
consumed. Automatic retry is forbidden.

## Exact CLI and lifecycle

### Intent producer

Change `tools/make_candidate_seal.py` so its lifecycle mode is mutually exclusive:

- existing `--decode-wall-clock <leg>`;
- existing `--inherit-decode-wall-clock <leg>`;
- new `--first-fire-intent`;
- new `--complete-first-fire-intent <intent>`.

The RLC2 producer invocation is:

```text
.venv/bin/python tools/make_candidate_seal.py \
  --first-fire-intent \
  --candidate-id <candidate-id> \
  --runtime-dir <candidate-runtime> \
  --axis contest_cuda \
  --public-entrypoint-smoke <receipt> \
  --candidate-manifest <manifest> \
  --manifest-validation <receipt> \
  --twin-encode-receipt <receipt> \
  --archive-parseback-receipt <receipt> \
  --raw-identity-receipt <receipt> \
  --literal-census <receipt> \
  --retention-manifest <manifest> \
  --timing-risk-evidence <candidate_prefire_timing_risk.v1> \
  --admit-bar-net-ds -2e-5 \
  --pointer-axis contest_cuda \
  --bar-tolerance 0 \
  --retained-path <path> [--retained-path <path> ...] \
  --falsifier <text> [--falsifier <text> ...] \
  --out <candidate_prefire_intent.json>
```

The producer derives archive, runtime, receiver, pointer, and every digest from disk. It writes the
intent, validates it through the consumer validator, and deletes it if that validation fails. The
immediate producer check performs every semantic and live-byte validation except the necessarily later
“exact intent blob is committed” check; the pre-dispatch consumer requires that Git custody. The
existing command without `--first-fire-intent` still requires a real timing leg and behaves unchanged.

### Authorization producer

Add a dedicated producer; do not accept `--authorized-by MAIN` as a hand-typed substitute:

```text
.venv/bin/python tools/authorize_candidate_first_measurement.py \
  --intent <candidate_prefire_intent.json> \
  --lane-id <lane> \
  --instance-job-id <job> \
  --output-dir <durable SSD directory> \
  --cost-preflight <real cost receipt> \
  --out <candidate_first_measurement_authorization.json>
```

MAIN commits this exact object before fire. The tool derives candidate, axis, receipt destination,
dispatch-policy digest, and all limits from the intent.

### First-measurement fire

Add exactly this mode:

```text
.venv/bin/python tools/fire_modal_auth_eval.py \
  --first-measurement <candidate_prefire_intent.json> \
  --first-measurement-authorization <candidate_first_measurement_authorization.json>
```

In this mode `--seal`, `--runtime-dir`, `--archive`, `--axis`, `--output-dir`, `--lane-id`,
`--instance-job-id`, `--claim-agent`, `--require-archive-sha`, `--repin-receiver`,
`--poller-deadline-s`, `--no-source-snapshot`, either smoke/nonpromotable waiver, and any timeout or
resource override are forbidden. Everything is derived from the two committed objects. `--dry-run`
may validate and print the derived command, but it does not consume the nonce and is not a positive
producer proof.

Before any subprocess the tool validates contract, intent, authorization, current candidate bytes,
all evidence, current pointer, lane maturity, active claim, global single-flight state, cost receipt,
unconsumed nonce, and the exact output/receipt paths. It then atomically reserves the nonce. The worker
argv MUST explicitly include:

```text
--gpu T4 --scorer-device cuda --inflate-device auto
--inflate-timeout 1800 --evaluate-timeout 1800
--claim-policy require_active
```

It also carries the intent and authorization SHA-256 values into the worker request/result, passes the
authorized single-axis reason, requires an immutable source snapshot, records the fixed 4,800-second
Modal function cap, and arms a 5,400-second durable harvest poller. Any implementation whose
`build_dispatch_argv` merely relies on the present timeout defaults is noncompliant.

The worker and poller mark the first-measurement result `score_claim=false`,
`promotion_eligible=false`, `adjudication_required=true`, and record
`prefire_intent_sha256`, `first_measurement_authorization_sha256`, lane, job, call id, exact argv, and
receipt destination. They must not publish a normal promotable frontier mirror. Exact evaluator
components, logs, raw payloads, and timeout artifacts are still retained; quarantine is not discard.

### Harvest and completed seal

MAIN harvests the real receipt and runs:

```text
.venv/bin/python tools/make_candidate_seal.py \
  --complete-first-fire-intent <candidate_prefire_intent.json> \
  --first-measurement-authorization <candidate_first_measurement_authorization.json> \
  --candidate-t4-receipt <output-dir>/MODAL_REMOTE_RESULT.json \
  --out <SEAL_candidate_contest_cuda.json>
```

This mode accepts no retyped candidate, runtime, archive, axis, pointer, bar, lane, timeout, or evidence
flags. It performs, in order:

1. validate the consumed authorization, call-id lineage, intent/result digests, exact axis/lane/job,
   receipt hash, and retained worker request;
2. call the existing `build_t4_direct_leg` on that exact receipt and unchanged candidate bytes;
3. require the unchanged direct validator: successful canonical CUDA path, T4, cold public entrypoint,
   n600, no resume/cache, finite positive inflate time, and `seconds <= 1260.0`;
4. revalidate every intent non-timing gate and current candidate identity from disk;
5. require current pointer score and archive equal the intent's zero-tolerance pointer binding;
6. recompute score from exact evaluator components, require the result's archive identity, and require
   exact `net_dS < admit_bar.net_dS_threshold`;
7. create `candidate_seal.v3` containing the completed `decode_wall_clock`, plus exact
   `{path, bytes, sha256}` blocks for `prefire_intent`, `first_measurement_authorization`, and
   `first_measurement_receipt`;
8. run the normal seal consumer with `require_decode_wall_clock=True`; delete the new seal on any
   failure; only after PASS may the quarantined score receive a separate promotable adjudication record.

The intent and authorization are immutable history. Completion creates a new seal and never overwrites,
edits, or upgrades either object. Legacy `candidate_seal.v1/v2`, measured/inherited/direct legs, and
normal `--seal` semantics remain unchanged.

## Required typed refusals

Every refusal writes beside the intent/authorization and in the output directory, prints to stderr,
sets `score_claim=false` and `promotion_eligible=false`, and occurs before subprocess unless it is a
harvest fact.

| Code | Exact refusal condition |
|---|---|
| `PREFIRE_INTENT_SCHEMA_REFUSED` | Missing/extra key, wrong schema/state, placeholder, malformed type, non-finite value, or bad canonical digest. |
| `PREFIRE_INTENT_FALSE_AUTHORITY_REFUSED` | Candidate timing leg/seconds/clearance or score/promotion claim appears in the intent. |
| `PREFIRE_CONTRACT_DRIFT_REFUSED` | Implementation/memo commit, manifest, live source, reachability, or prospective ordering differs. |
| `PREFIRE_IDENTITY_DRIFT_REFUSED` | Live archive/runtime/receiver/member/pin differs from intent or any evidence endpoint. |
| `PREFIRE_NON_TIMING_GATE_REFUSED` | Manifest, smoke, twin encode, parse-back, n600 raw identity, literal census, retention, or falsifier contract fails. |
| `PREFIRE_POINTER_DRIFT_REFUSED` | Current pointer score or archive SHA differs at intent, dispatch, or completion. |
| `PREFIRE_RISK_EVIDENCE_REFUSED` | Source direct leg invalid; endpoint receiver/delta mismatch; diagnostic identity wrong; recomputed risk ceiling differs or exceeds 1,260; hard timeout not exactly 1,800. |
| `FIRST_MEASUREMENT_AUTHORIZATION_REFUSED` | Authorization missing, not committed by the required MAIN workflow, not descended from contract/intent, digest mismatch, field mismatch, or cost upper bound not strictly below USD 5. |
| `FIRST_MEASUREMENT_REPLAY_REFUSED` | Nonce already RESERVED/SPAWNED/HARVESTED, output/job already used, or provider state is ambiguous. |
| `FIRST_MEASUREMENT_ARGUMENT_REFUSED` | Any seal-owned or authorization-owned value is retyped or overridden on the CLI. |
| `FIRST_MEASUREMENT_LANE_REFUSED` | Lane not promotable, active claim missing, another scored job active, single-flight state unknown, or axis/resource is not contest-CUDA T4. |
| `FIRST_MEASUREMENT_TIMEOUT_REFUSED` | Worker/poller timeout bindings differ, receipt records a timeout, or the direct T4 receipt is incomplete. |
| `FIRST_MEASUREMENT_WARM_REFUSED` | Pair count is not 600, resume occurred, token cache is not DISABLED, or cold proof is missing/unparsable. |
| `FIRST_MEASUREMENT_T4_POLICY_REFUSED` | Exact completed inflate seconds are greater than 1,260.0 even if below 1,800. |
| `FIRST_MEASUREMENT_RESULT_REFUSED` | Remote return code/path/hardware/CUDA/archive/runtime/intent/auth/call-id binding fails, exact score cannot be recomputed, or net delta misses the frozen bar. |

Unknown state is always a refusal. No public-smoke waiver, dry run, nonpromotable-lane flag, manual
nonce reset, or hand-written receipt may weaken these rules.

## RLC2 decision

**RLC2 MAY RESUME only after the contract implementation, validators, quarantine consumers, and
negative tests are committed and hash-frozen.** Resume means materialize the real move-42 rebase,
retain both full encodes, prove full-n600 public raw identity, regenerate and independently validate the
manifest, repeat the PR9 literal census, capture candidate/frontier smoke, build the exact risk receipt,
and produce a committed intent. It does not mean RLC2 may dispatch. MAIN alone may create the separate
authorization and fire after revalidating that real intent.

The first paid fire is worth taking under the project budget **only if** the real candidate has
`archive_bytes <= 180207`, full raw identity with move 42, the exact cured receiver digest and delta
lineage above, risk ceiling at most 1,260 seconds, all other intent gates PASS, and a one-call cost upper
bound strictly below USD 5. At the conditional 180,178 B value the rate-only delta is
`-3.995153718733028e-5 S`, giving conditional `S = 0.13743655372199698` (reported by RLC2 as
`0.137436553721997`). That would clear the move-42 pointer and its `-2e-5` bar, so a real exact row is a
reasonable use of one paid T4 call. It would not clear 0.12 and is not goal completion.

If the real archive is 180,208 B or larger, raw identity fails, the normalized receiver is not exactly
the timed RLC1 reference, the pointer moves, or any evidence is absent, no first-measurement fire is
authorized. A later candidate may be re-adjudicated; the threshold is not relaxed.

## Prospective freeze and real positive control

The implementation landing precedes every RLC2 producer timestamp and is included in every later
object. Negative and synthetic fixture tests are useful but cannot close the pass path. The mandatory
real control sequence is:

1. RLC2's real candidate producer emits and self-validates `candidate_prefire_intent.v1`; the normal
   `validate_seal` and normal `--seal` path both refuse it as not a seal.
2. MAIN emits a real committed authorization. A dry-run consumes nothing. The real fire consumes the
   nonce once; a second invocation with the same authorization refuses before subprocess.
3. MAIN harvests the real call. If and only if its exact receipt passes unchanged `t4_direct`, all
   non-timing revalidation, pointer identity, and the score bar, the completion producer creates a
   real `candidate_seal.v3` that validates through the normal consumer.
4. Timeout, over-1,260 timing, pointer drift, result mismatch, or a score miss instead produces a
   retained real refusal. A fixture-only passing test must not be described as proof that the producer
   door works.

If RLC2 does not reach the passing completion branch, `candidate_prefire_intent.v1` remains implemented
but not positively proven end to end. No second candidate family may treat fixtures as that missing
proof; the contract must be re-reviewed or exercised by another named real producer first.

## RECALL EVIDENCE

I searched the complete `.omx/research/` corpus by content for
`first[-_ ]fire|prefire|pre[-_ ]fire|first[-_ ]measurement|seal.before.fire|t4_direct|decode_wall_clock|gate with no door|producer pass path|intent digest`;
searched `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED surface, design/SPEC files,
`.omx/state/canonical_task_status.jsonl`, the lane registry, and operator/task ledgers; and generated the
current canonical-equation registry with `.venv/bin/python tools/list_canonical_equations.py --json`.
The Codex memory-registry query for `ddm_pr12|first fire|intent contract|20260910` had no relevant hit.

Beyond the charter's seeds:

1. SCG1's `candidate_public_entrypoint_smoke.v1` established two reusable boundaries: the normal fire
   waiver is only for already-scored custody replay, and library-path evidence is not public-entrypoint
   proof. This changed the contract to forbid the waiver and require the existing full structured smoke
   validator for an unscored first measurement.
2. The current fire/worker path showed that inflate/evaluate timeouts are worker options but are absent
   from `build_dispatch_argv`, the worker cap is 4,800 seconds, and the current CUDA poller deadline is
   2,400 seconds. This changed the contract from “budget bound” prose to explicit argv and a 5,400-second
   poller that can retain a terminal result even near the worker cap.
3. PR9's independent literal census and manifest finding showed that exact receiver identity alone does
   not close rule 118 or dependency custody. This added separately hashed literal-census and externally
   verified manifest requirements before the paid call.
4. The task ledger confirms RLC2 stopped before materialization and routes the decision to MAIN plus the
   second family; no prior approved first-measurement schema or live RLC2 candidate was found in that
   bounded search.
5. The canonical equation registry contains the contest rate multiplier and pointer-only frontier law,
   but no equation authorizes a pre-fire timing exception or converts the refused local rows into timing
   authority. I used the rate equation only for the integer 31-byte bar and 60-byte conditional delta.

The prior DAG “first-fire” matches concern unrelated training/probe events; none supplies a seal/timing
pass path. I did not find an earlier approved intent-plus-one-shot-authorization lifecycle in the
searched scope.

## Boundaries

- Adjudication only. I created this memo and used the mandated checkpoint writer. I did not edit source,
  tests, receipts, candidate/sealed trees, payloads, pointer, ledgers, `upstream/`, the three forbidden
  files, or the shared staged index.
- No decoder, encoder, timing producer, scorer, exact evaluator, local diagnostic, Modal call, paid
  dispatch, harvest, or candidate materialization ran. No payload was created, moved, deleted,
  deduplicated, or discarded.
- The current code/proposal facts are verified at source and from retained receipts. RLC2 bytes, raw
  identity, receiver digest, T4 time, evaluator components, exact score, and completed seal are all
  unmeasured because the candidate does not exist.
- The 1,032.7250802956457-second RLC2 figure is a conservative diagnostic projection made from refused
  `[macOS-CPU advisory]` rows and one completed source-receiver T4 row. It gates spend risk only and is
  not a timing leg, cross-host calibration, confidence bound, or promotion evidence.
- The RLC1 reference receiver and RLC2 candidate must be byte-identical under the normalized-receiver
  definition. Similar source, mechanism, archive size, or prose “same maps” is insufficient.
- The one-call `< USD 5` condition is prospective. No provider-price receipt was generated or cost
  incurred here.
- A first-measurement result can contain exact evaluator components while quarantined. It cannot move
  the pointer until a new completed seal and promotion adjudication bind the same exact receipt.
- This apparatus ruling does not weaken the 1,260-second policy, the 1,800-second hard timeout, cold
  n600 requirements, single-flight claims, public smoke gate, rule-118 boundary, or normal seal path.
- The exact frontier did not move. This memo is means, not goal progress.

<!-- # FORMALIZATION_PENDING: apparatus-only lifecycle contract; the real RLC2 producer/consumer path
must be exercised before any measured anchor exists. -->

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / ddm_dwc1`; consumer store:
  `src/tac/candidate_prefire_intent.py`, `src/tac/candidate_seal.py`,
  `src/tac/decode_wall_clock.py`, `tools/make_candidate_seal.py`,
  `tools/authorize_candidate_first_measurement.py`, `tools/fire_modal_auth_eval.py`, the CUDA worker,
  poller/quarantine consumer, and focused tests; fire trigger: harvest of this committed pr12 memo.
  Implement and commit the exact contract above before RLC2 resumes; preserve normal seal and unchanged
  `t4_direct` validation.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rlc2`; consumer store:
  `.omx/research/ddm_rlc2_20260910/CONTRACT_DECISION_FIRE_ORDER.json` plus the retained RLC2 candidate
  directory; fire trigger: MAIN records the exact implementation commit and contract manifest as
  frozen. Resume the real twin encode/raw-identity/manifest/smoke/literal-census chain and emit a
  committed intent; do not dispatch.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the committed
  `candidate_first_measurement_authorization.v1` and its durable SSD output directory; fire trigger:
  RLC2's real intent validates, `archive_bytes <= 180207`, raw identity and exact receiver-delta risk
  evidence pass, the pointer remains move 42, the T4 lane is claimed/single-flight clear, and the real
  one-call cost upper bound is `< USD 5`. Authorize once, dispatch once, retain and harvest everything.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: RLC2's `candidate_seal.v3`, exact-result
  adjudication, evaluation ledger, and canonical pointer; fire trigger: the harvested receipt passes
  unchanged `t4_direct` at `<= 1260.0` seconds and every identity/non-timing/bar recheck. Create the new
  completed seal, unquarantine that same receipt only after validation, and move the pointer only if the
  exact recomputed score qualifies.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- RLC2 will preserve the ancestor cure's roughly 60-byte gain because it applies the same counted
  fixed-point receiver mechanism to a pointer field that changed only through later payload work. This
  is plausible from mechanism lineage, but no RLC2 encode exists.
- RLC2 will complete cold T4 inflate below 1,260 seconds because the completed source receiver takes
  990.053829427 seconds and the exact cured receiver's slowest retained local diagnostic adds 4.31%,
  projecting 1,032.725 seconds. The projection is non-authoritative and host transfer remains untested.
- A one-shot committed authorization plus pre-spawn consumption will preserve DWC1 while opening a real
  producer pass path. The type and state separation are source-review sound; the real RLC2
  producer/harvest path has not yet exercised them.

## DEAD-ENDS

- Adding `pending`, `intent`, or missing-seconds semantics to `candidate_decode_wall_clock` is closed:
  it would let an unmeasured object enter the timing validator and weaken DWC1's completed-leg meaning.
- Calling the intent `candidate_seal.v3` before timing is closed: normal seal consumers must keep
  refusing every object without a valid timing leg.
- Letting the candidate producer write `authorized_by: MAIN` is closed: a string is self-authorization,
  not independent custody.
- Reusable authorization or automatic retry is closed: an ambiguous spawn can otherwise buy a second
  row. Consume before spawn and require a new MAIN authorization after reconciliation.
- Relying on the worker's current 1,800-second default is closed: the dispatch must pass and retain the
  exact timeout, and 1,800 is never the 1,260-second admission threshold.
- Publishing the first-measurement result as normally promotable is closed: a 1,336-second decode can
  finish and score while still violating the timing policy.
- A sixth local calibration/timing run for RLC2 is closed by the explicit suspension. The retained
  g3/g4 rows remain diagnostics only and cannot be relabeled v3 or timing authority.
- Inheriting move-42 timing for the cured receiver is closed by normalized-receiver inequality.
- Fixture-only positive proof is closed by DWC1's gate-with-no-door incident. The real RLC2 producer,
  one-shot fire, and MAIN harvest are the required pass-path control.
- Spending on the conditional 180,178 B row before real bytes, full raw identity, receiver equality,
  and the 31-byte strict bar are established is closed. Conditional arithmetic is not a candidate.
