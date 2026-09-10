# ddm_ffi4 — scoped pre-fire timing-risk receiver amendment

Date: 2026-09-10. Axis: `[macOS-CPU scorer-free; retained-byte verification]`.
`research_only=true; score_claim=false`. `[no-triality] [p0-ledger-ok]`.

Implemented PR14's literal scoped amendment in the two authorized Python files.
MEASURED: **135 passed**, Ruff clean; exactly six new named test functions. Both real
retained trees produce the required identical 49-row risk digest. No exact score was
measured, no fire was opened, and the frontier did not move. MAIN must land the bundle
if the serializer returns rc 17, then separately append the frozen amendment; the
production contract deliberately refuses until that committed append exists.

This is apparatus work, not sub-0.12 goal progress. The common contract's older frontier
paragraph is historical; the specific charter and live hot board supply the move-42 line.

## Clause → code → test

Code below is in `src/tac/candidate_seal.py`; all T1–T6 tests are in
`src/tac/tests/test_candidate_prefire_intent.py`.

| PR14 literal clause | Code / immutable surface | Executed control |
|---|---|---|
| Exact versioned constants and only named manifest exclusion | `PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION`, `PREFIRE_RISK_RECEIVER_EXCLUDED_PATHS`, `PREFIRE_CONTRACT_AMENDMENT_SCHEMA`, `PREFIRE_CONTRACT_AMENDMENT_ID` | T1 rejects unversioned source, candidate and diagnostic endpoint fields |
| One common content-only row materializer; normalize only two literal archive-pin AST value spans | `_materialize_prefire_receiver_rows` copied verbatim | T1 equal rows; T2 one-byte outside-pin falsifier plus duplicate, nonliteral and missing-pin refusals; two real trees |
| Exact canonical compact row digest | `_prefire_receiver_rows_digest` copied verbatim | Both retained listings have identical bytes and digest |
| Legacy normalized rows remain manifest-inclusive and match legacy helper | `prefire_receiver_rows` copied verbatim | T1 manifest row present, differing legacy digests; T4 legacy refusal |
| Separate scoped risk rows and measurement entrypoint | `prefire_risk_receiver_rows`, `measure_prefire_risk_receiver_digest`, copied verbatim | T1 manifest regeneration accepted; T2 executable-file byte mutation refused |
| Exact source endpoint with scoped hash and unchanged legacy T4 hash/definition | `validate_prefire_risk.source_receiver` | T1 definition and legacy-join mutation refusals |
| Exact candidate endpoint with versioned scoped hash | `validate_prefire_risk.candidate_receiver` | T1 definition refusal; T2 live-byte refusal |
| Diagnostic reference must have candidate scoped digest and its own legacy receipt digest | `validate_prefire_risk.diagnostic_reference_receiver`; separately recomputes reference scoped digest | T1 permits raw manifest indirection and refuses forged legacy joins |
| Enumerate complete source/candidate risk rows; pin both delta endpoints | `validate_prefire_risk` risk-row maps and `wanted` comparison | T1 removed row and either endpoint hash refuse |
| Base diagnostic binds source legacy T4 digest; candidate diagnostics bind reference legacy receipt digest | `validate_prefire_risk` diagnostic loop | T1 full validator accepts preserved legacy diagnostics; forged joins refuse |
| Risk calculation and RLC2 exact numbers remain unchanged | Existing calculation and pinned-chain block | Existing arithmetic-refusal test; same literals and formulas retained. Historical `b06e59…` now checks `diagnostic_reference_receiver.receipt_sha256`, its explicit legacy join, rather than the newly scoped hash. No new real timing measurement |
| No change to full runtime digest, dependency manifest, identity or non-risk endpoints | `measure_runtime_digest`, `_pf_identity`, `_pf_evidence`, candidate normalized-receiver field unchanged | T3 full intent refuses stale raw manifest custody at `PREFIRE_NON_TIMING_GATE_REFUSED`; T4 full runtime hash changes |
| Copy exact final frozen amendment into intent | `build_prefire_intent` and `_pf_contract`; comparison preserves JSON types | Existing producer self-validation test; T6 latest-row, uncommitted-freeze and boolean-as-number refusals |
| Keep original materialization implementation manifest pinned against original committed blobs | `_pf_contract` base manifest loop, `require_live=False` | T5 old evidence accepted although live amended source differs from original source; T6 forged original manifest row refuses |
| Amended source manifest covers every required path and matches live plus committed source | `_pf_contract` amendment manifest loop, `require_live=True` | T6 live-source and amended committed-blob mismatch refusals; existing live-source drift control |
| Base commit ≤ production time ≤ intent time | `_pf_contract` original chronological check | T5 both invalid chronology directions refuse |
| Base ancestor of amendment and producer; amendment ancestor of HEAD | `_pf_contract` Git ancestry queries | T6 disconnected amendment/producer and old-HEAD refusals using real isolated Git objects |
| Amendment commit ≤ new intent timestamp; old receipts may predate amendment | `_pf_contract` amended timestamp check | T5 old receipts accepted byte-for-byte; pre-amendment timestamp and amendment-less intent refuse |
| Exact amendment row committed in HEAD; required intent bytes equal HEAD | `_pf_contract` `_pf_blob` checks on freeze and intent | T6 uncommitted freeze and even whitespace-only intent drift refuse |
| Legacy timing implementation, maker, and existing seal tests receive no source edits | `src/tac/decode_wall_clock.py`, `tools/make_candidate_seal.py`, `src/tac/tests/test_candidate_seal.py` | Original SHA-256 pins preserved; full requested regression command passes |
| Freeze append is a separate mechanical landing after implementation | Draft below, no actual freeze edit | Baseline frozen receipt hash unchanged; draft has only two permitted placeholders |

T1 `test_prefire_risk_accepts_only_regenerated_manifest_indirection`.
T2 `test_prefire_risk_refuses_nonpin_inflate_byte_change`.
T3 `test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation`.
T4 `test_prefire_amendment_keeps_legacy_t4_direct_manifest_inclusive`.
T5 `test_prefire_amendment_allows_pinned_old_evidence_but_refuses_pre_amendment_intent`.
T6 `test_prefire_amendment_requires_latest_frozen_row_and_live_committed_sources`.

T5/T6 run a passing full intent through actual Git custody reads over isolated loose
objects before attacking ordering and custody. The broader fixture still substitutes
3.6 GB raw transport and Git history; these tests do not claim a real decoder, timing,
provider or n600 scorer result. The real-tree checks below use actual retained source bytes.

## Real-tree derivation

| Endpoint | Read-only root | Risk digest | Rows | Retained listing |
|---|---|---|---:|---|
| RLC1 timed reference | `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` | 49 | `.omx/research/ddm_ffi4_20260910/reference_normalized_rows.json` |
| RLC4 candidate | `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` | 49 | `.omx/research/ddm_ffi4_20260910/candidate_normalized_rows.json` |

Both compact JSON listings are **5,081 bytes**, SHA-256
`9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890`,
and compare byte-for-byte equal. The row-list hash is the digest by definition.
The legacy digests remain respectively
`b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d` and
`27948d3d5ac0c32bd6eda8c0cfd08b2b488cfaa60c44ce3e98ba2689d7edbb8a`.
Receipt: `.omx/research/ddm_ffi4_20260910/real_tree_digests.json`.

## Validation and review

Executed on the host, `[macOS-CPU scorer-free]`:

```text
.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q
135 passed in 3.13s
.venv/bin/python -m ruff check src/tac/candidate_seal.py src/tac/tests/test_candidate_prefire_intent.py
All checks passed!
```

Retained logs: `ddm_ffi4_20260910/pytest_full.txt`, `ruff.txt`.
Two visible review passes per changed Python file were recorded with
`tools/review_tracker.py mark-file <path> --status reviewed --reviewer ddm_ffi4 --pass 1`
and `--pass 2` after scan: 87 source entities, 41 test entities in each pass.
`review_evidence.json` records the reviews, exact post-edit hashes, and an AST check:
only `_pf_contract`, `build_prefire_intent`, `prefire_receiver_rows` and
`validate_prefire_risk` changed among existing source functions; only the four new
helper functions were added. The full five-function literal PR14 block is identical.
No review override was used.

Review assumption challenged: the raw manifest can change while the behavior-bearing
receiver rows stay equal, because the manifest hashes the raw archive-pin assignment.
The single named exclusion resolves that indirection; it does not excuse another byte
change or replace independent raw custody. No unresolved review finding remains in scope.

## Freeze append DRAFT — MAIN only, after source landing

This exact draft is also retained at
`.omx/research/ddm_ffi4_20260910/FREEZE_APPEND_DRAFT.json`.
Only `implementation_commit` and `implementation_manifest.sha256` remain placeholders.
The proposed manifest destination is fixed; its **2,072-byte** size is measured from
`IMPLEMENTATION_MANIFEST_PREVIEW.json` with sorted `{path,sha256}` rows, indent 2 and
one trailing newline. The preview is not the canonical committed manifest. MAIN must
regenerate it from the landed implementation and require the recorded size or refuse.
All other manifest-covered source files currently equal HEAD; only the chartered
`candidate_seal.py` implementation differs before landing.

```json
{
  "schema": "prefire_contract_amendment.v1",
  "amendment_id": "ddm_pr14_manifest_in_receiver_risk_digest",
  "adjudication_memo": {
    "path": "/Users/adpena/Projects/pact/.omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md",
    "bytes": 48452,
    "sha256": "6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9"
  },
  "definition_change": {
    "scope": "candidate_prefire_timing_risk.v1 only; legacy decode_wall_clock unchanged",
    "digest_definition": "tac.candidate_seal.measure_prefire_risk_receiver_digest.v1",
    "excluded_relative_paths": [
      "MANIFEST.sha256"
    ],
    "raw_manifest_still_required": true,
    "executable_difference_policy": "REFUSE"
  },
  "reference_receiver": {
    "path": "/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime",
    "legacy_sha256": "b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d",
    "amended_sha256": "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890",
    "included_file_count": 49,
    "raw_input_pinset_sha256": "663b36aed2656b9e56fb5c8867b8ebd083e8b241a6d8741d091c2762318f9f3d",
    "excluded_manifest": {
      "bytes": 4570,
      "sha256": "98993a00b6f2eb0f0ef8454bfbeea40ccdae7c18012a5a25cdfd645dfa4fc8f4"
    }
  },
  "candidate_receiver": {
    "path": "/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime",
    "legacy_sha256": "27948d3d5ac0c32bd6eda8c0cfd08b2b488cfaa60c44ce3e98ba2689d7edbb8a",
    "amended_sha256": "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890",
    "included_file_count": 49,
    "raw_input_pinset_sha256": "046621cb5c4c1d5c67f2ab2d4857ee750b97e885b95713930acfcc4c46f3a9dc",
    "excluded_manifest": {
      "bytes": 4570,
      "sha256": "7c2b977a968dbd0512ce817dd7951e06d817b125875aa410a4c0f0d863345147"
    }
  },
  "implementation_commit": "${FULL_IMPLEMENTATION_LANDING_COMMIT}",
  "implementation_manifest": {
    "path": "/Users/adpena/Projects/pact/.omx/research/ddm_ffi4_20260910/PREFIRE_IMPLEMENTATION_MANIFEST.json",
    "bytes": 2072,
    "sha256": "${AMENDED_IMPLEMENTATION_MANIFEST_SHA256}"
  },
  "score_claim": false
}
```

MAIN must verify the PR14 memo bytes/hash, both retained endpoints, all
`PREFIRE_IMPLEMENTATION_PATHS` against both live files and the landed committed blobs,
and the landed PR14 memo blob. Then fill the two placeholders, generate the canonical
manifest, and append this row as the sole member of a new top-level `amendments` array.
Never rewrite an existing frozen key/value. This arm did not modify the freeze.

## RECALL EVIDENCE

Before building, searched `.omx/research/` Markdown/JSON contents using
`manifest.{0,80}(receiver|digest)|receiver.{0,80}manifest|environment.free digest|prefire`;
searched all `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` surfaces with
`prefire|receiver.digest|content.only digest`; searched docs and repository-wide
SPEC/design Markdown with `prefire|receiver.digest|dependency.manifest`; searched the
canonical task status and lane registry for `ffi4|pr14|rlc4|receiver.digest`.
Generated all **483** canonical equation rows with
`.venv/bin/python tools/list_canonical_equations.py --json` and filtered for
`prefire|receiver.digest|environment.free.digest|normalized.receiver|dependency.manifest`:
zero matching equation rows. The graph hits concern older training fire prerequisites,
not this timing-risk digest. No competing definition was found in those equation,
graph and design/SPEC scopes. Query receipts and results are in `RECALL_SUMMARY.json`
and `recall_queries.json`; full search captures remain in this arm's evidence directory.

Beyond the charter's named seeds, `ddm_seal1_candidate_seal_contract_20260818.md:29–30,56–66`
requires consumer-invariant content identity and actionable per-file pins;
`ddm_r9m_first_contest_cpu_row_20260804/runtime_tree_hash_verification_20260804.md:7–24`
shows equal raw files hashing differently because of environment/path projection.
These checks confirmed the shared helper must remain purely byte-based and legacy pins
must remain separate. No departure from PR14's literal patch was warranted. Live hot
state assigns FFI4 implementation to this arm and freeze/producer handoff to MAIN.
Recent directive files were read; none overrides this scoped implementation charter.
The memory registry query for `ffi4|PR14|risk.digest|arbitrage` found no matching entry.

## Custody and boundaries

The serializer is the last and only commit attempt, with per-file post-edit hashes,
`[no-triality] [p0-ledger-ok]`, label `ddm_ffi4`, and `--no-co-author`.
Actual rc and output are retained after the attempt in `SERIALIZER_STATUS.json`,
`serializer.stdout.txt`, `serializer.stderr.txt`; bundle/landing verification is in
`FINAL_HANDOFF.json`. A Git-object denial is not completion of main landing: retain the
verified bundle and route MAIN to land it. The fallback destination is explicitly local
under this arm's evidence directory because this charter forbids volume writes. It is
a bounded source/metadata bundle, with serializer storage/cap checks and temporary
Git-object cleanup; no large candidate work is generated.

No Modal/provider call, authorization, fire, timing window, n600 run, encoder, decoder,
scorer, evaluator, or GT decode was launched. No source edit to `upstream/`, the PR tree,
sealed trees, any `/Volumes/...` content, `src/tac/decode_wall_clock.py`,
`tools/make_candidate_seal.py`, `src/tac/tests/test_candidate_seal.py`, the frozen receipt,
or live `ddm_sj1_pass6` directories. No changes to
`.omx/research/ddm_cr1_composition_row_827_20260801.md`,
`.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`, or
`src/tac/optimization/direct_description_carrier_compose.py`.
No stash or direct shared-index operation. Unrelated dirty work is excluded from the
intended file list. No payload was discarded, moved, cold-stored or deleted; no new
candidate payload was materialized. No exclusion outside the scoped risk digest,
no MPS authority, no score/timing authority inferred from fixtures or retained hashes.

Six integration hooks: sensitivity, Pareto, bit allocator and posterior hooks are N/A
because this is a contract amendment with no measured score law; the dispatch consumer
is the existing `validate_prefire_intent` path, intentionally closed pending the committed
freeze; interpretation is fixed by PR14, so no additional disambiguator is required.
Checkpoint identity is `ddm_ffi4`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: this memo, `ddm_ffi4_20260910/FINAL_HANDOFF.json`, serializer bundle and canonical frozen receipt `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; fire trigger: harvest this verified source/test bundle and recheck its post-edit hashes and host tests. Land the implementation if rc 17, regenerate its implementation manifest from the actual landing, fill only the two draft pins, and separately commit the append-only freeze row.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN dispatching rlc5; consumer store: PR14's evidence-reuse table, retained RLC4 receipts, new risk delta/risk receipt/new intent; fire trigger: implementation and freeze append are committed, all reused path/bytes/hash pins validate and the pointer still names move 42. Re-emit and self-validate the intent and normal-seal refusal controls; authorization/fire remain separate MAIN-owned gates from PR14, not this arm's action.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- The real reused-receipt producer path can pass after MAIN's freeze append: both retained
  endpoints now agree on the exact versioned 49-row digest. Real producer/MAIN cross-actor
  custody is still untested; fixture acceptance is not that result.

## DEAD-ENDS

- FORMULATION: excluding the raw manifest from full runtime custody or legacy timing is
  closed; both remain manifest-inclusive and the legacy mutation control refuses.
- INSTANCE: a one-byte non-pin `inflate.py` change cannot inherit the diagnostic receiver;
  the real validator returns `PREFIRE_RISK_EVIDENCE_REFUSED` in the named control.
- FORMULATION: pre-amendment intents, uncommitted/stale freeze rows, forged implementation
  manifests and mismatched source history are closed by typed custody refusals.
- FORMULATION: re-timestamping old production evidence is unnecessary; old pinned receipts
  validate under the amended contract while a pre-amendment intent refuses.
