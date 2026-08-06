# ddm_fm2 Receipt - fmtools Full-Leverage Upgrade

## Verdict First

On-device generation leg: **BLOCKED-WITH-CAUSE, not VERIFIED**.

The external fmtools venv imports `apple_fm_sdk` and reports model availability `(True, None)`, but actual generation fails before Pact parsing:
- plain `LanguageModelSession.respond("Say ok")`: `GenerationError status 255: None`
- structured `respond(..., generating=Tiny)`: `GenerationError status 255` with underlying `ModelManagerServices.ModelManagerError Code=1008`

The installed external venv still has `apple-fm-sdk 0.1.1`. PyPI latest was verified as `0.2.1`, but a test install into a writable SSD venv failed because shell DNS/PyPI access is blocked. Therefore `0.2.1` runtime verification is **QUEUED-WITH-FIRE-ORDER**, not claimed.

## External fmtools Patch

The external repo was located and inspected:

| Field | Value |
|---|---|
| path | `/Users/adpena/Projects/fmtools` |
| base commit | `c9e755539da22df8aee6c5c22fa6653253456a4f` |
| status | clean `main` |
| writability | blocked in this sandbox (`test -w` rc=1) |
| patch path | `.omx/research/ddm_fm2_20260806/fmtools_patches/0001-fmtools-full-sdk-surfaces.patch` |
| patch sha256 | `9804876bec64c21b1de70470bedfd276d12e3c0ab6cace3ef3c24ecbbd6340ee` |
| apply-check | `git -C /Users/adpena/Projects/fmtools apply --check ...` clean |

Patch contents:
- bumps `apple-fm-sdk>=0.1.1` to `>=0.2.1`
- adds `fmtools.capabilities.FMCapabilityReport` and `capability_report()`
- adds `fmtools.session.FMGenerationControls`, `respond()`, and `stream_response()`
- extends backend/session plumbing for guided generation, JSON schema, options, tools, and streaming
- preserves old custom backend compatibility by keeping `SessionProtocol` respond-only and passing new kwargs only when provided
- adds mocked tests for the new SDK surfaces and updates packaging metadata expectations

Verification in SSD clone:
- `PYTHONPATH=/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/fmtools_patch_verify /Users/adpena/Projects/fmtools/.venv/bin/python -m pytest tests` -> 652 passed, 12 skipped
- `/Users/adpena/Projects/fmtools/.venv/bin/ruff check fmtools/capabilities.py fmtools/session.py fmtools/protocols.py fmtools/backends/apple_sdk.py fmtools/backends/ffi.py fmtools/decorators.py fmtools/__init__.py tests/test_full_sdk_surfaces.py tests/test_packaging_metadata.py` -> all checks passed

## Pact Wire-Through

Changed Pact files:
- `src/tac/fm_advisory.py`
  - classifier subprocess now prefers patched `fmtools.respond(..., generating=Choice)` when available
  - preserves `local_extract` fallback for current/unpatched fmtools
  - adds fail-open `capability_report()` subprocess
- `tools/costate_digest.py`
  - prints `capability: sdk=... available=... backend=... guided/tools/stream/options`
  - stores the report under `data["fm_advisory"]["capability_report"]`
- `src/tac/tests/test_fm_advisory.py`
  - covers capability-report shape and fail-open absence
- `src/tac/tests/test_costate_digest_fm_advisory.py`
  - covers capability line rendering and stored report

Pact verification:
- `.venv/bin/python -m pytest src/tac/tests/test_fm_advisory.py src/tac/tests/test_costate_digest_fm_advisory.py src/tac/tests/test_codex_arm_queue.py` -> 65 passed
- `.venv/bin/python -m ruff check src/tac/fm_advisory.py tools/costate_digest.py src/tac/tests/test_fm_advisory.py src/tac/tests/test_costate_digest_fm_advisory.py --select F` -> all checks passed
- `tools/review_tracker.py scan` plus two whole-file review mark cycles on the four edited Pact Python files. This checkout's tracker CLI has no pass-id/reviewer metadata.

## Capabilities Table

| Capability | Current Apple SDK/native surface | fmtools patch surface | Pact consumed |
|---|---|---|---|
| Availability/model controls | `SystemLanguageModel`, `is_available()`, model/control enums and signatures | `FMCapabilityReport` with SDK version, availability reason, model-control support | `fm_advisory.capability_report()` and costate digest capability line |
| Guided structured generation | `LanguageModelSession.respond(..., generating=..., schema=..., json_schema=...)` | `fmtools.respond(..., generating/schema/json_schema)` and existing `local_extract` | classifiers prefer `fmtools.respond(..., generating=Choice)` with fallback |
| Tool calls | `fm.Tool`; `LanguageModelSession(..., tools=[...])`; async `Tool.call()` | `create_session(..., tools=...)`, `local_extract(..., tools=...)`, `respond(..., tools=...)` | capability report only; no Pact tool execution consumer yet |
| Streaming | `LanguageModelSession.stream_response(...)`, text snapshots | `fmtools.stream_response(...)`; backend `AppleFMSession.stream_response` | capability report only; no Pact streaming consumer yet |
| Generation options | `GenerationOptions(temperature, sampling, maximum_response_tokens)` | `FMGenerationControls` and direct `options` passthrough | capability report only |
| Transcript/session lifecycle | `Transcript`, `LanguageModelSession.from_transcript(...)` | existing transcript helpers preserved; tools passed on resume | no new Pact consumer |

## Ranked Next-Leverage List

| Rank | SDK surface not fully consumed | Pact consumer candidate | Disposition |
|---|---|---|---|
| 1 | Live `0.2.1` generation round-trip | `src/tac/fm_advisory.py` classifiers | QUEUED-WITH-FIRE-ORDER: install `0.2.1` once shell DNS works, then rerun plain + structured probes |
| 2 | Structured JSON-schema output | `mechanism_reduction_language` / charter lint rows | FOLDED into patch: `generating=Choice` preferred now; JSON schema remains available for future multi-row outputs |
| 3 | Capability report | `tools/costate_digest.py` | FIRED: digest now prints and stores the report |
| 4 | Tool calls | future WARN-only ty1/negative-audit explainer tools | QUEUED-WITH-FIRE-ORDER: only after live generation verifies; never decision authority |
| 5 | Streaming | long digest/report summarization UI | QUEUED LOW: no current Pact caller needs streaming; capability only |
| 6 | Generation options | bounded-response advisory classifiers | QUEUED LOW: useful after live generation works; keep deterministic fail-open boundary |

## Recall Evidence

Searches/consulted:
- `rg` over `src/tac/canonical_equations`, DAG, main hot state, docs, and `.omx/research/ddm_fm1_20260806` for `fmtools`, `apple-fm-sdk`, `Foundation Models`, `LanguageModelSession`, and `fm_advisory`
- `tools/list_canonical_equations.py --json --consumer fmtools` -> `[]`
- `tools/list_canonical_equations.py --json --consumer fm_advisory` -> `[]`
- `tools/list_canonical_equations.py --json --consumer costate_digest` -> broader costate equations exist, but no fmtools-specific equation surface
- `.omx/research/fmtools_costate_organ_20260717.md`
- `.omx/research/ddm_fm1_20260806/{RECEIPT.md,NEXT_IF_RESUMED.md,CHECKPOINTS.md}`

Found beyond the charter:
- prior fmtools costate organ law is advisory-only, fail-open, non-promotable
- fm1 already identified the same live classifier generation failure under `apple-fm-sdk 0.1.1`
- canonical equations do not treat fmtools/fm_advisory as a score-law surface

What changed:
- preserved the advisory boundary
- refused to promote model availability as generation verification
- implemented capability reporting as visibility, not a gate
- staged external fmtools work as a patch series because in-place writes are blocked

## Storage And Cleanup

SSD evidence retained:
- `/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/fmtools_patch_verify` (tracked fmtools clone plus patch)
- `/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/sdk021_venv` and `uv_cache` from the failed latest-SDK install attempt

Observed size for the SSD evidence root: `9.8M`.

No deletion or movement was performed. These bytes are retained as reproducible patch/test evidence and failed-install evidence. No persisted evidence path depends on `/tmp`.

## Boundaries

- No scorer job was run or claimed.
- No `upstream/` file was touched.
- No forbidden common-contract file was edited.
- No exact eval was run.
- No archive was produced.
- No pointer moved.
- FM outputs remain advisory-only and never block/rescue/actuate.
- Own-vehicle advisory frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
- Contest pointer remains borrowed/unmoved `0.1910828242`.

## Serializer Status

Serializer completed successfully.

| Field | Value |
|---|---|
| commit | `412783cd93` |
| serializer | `.venv/bin/python tools/subagent_commit_serializer.py` |
| result | `OK head=412783cd93 label=anonymous files=9 recorded=9 temp_index=YES` |
| message | `ddm_fm2: stage fmtools full SDK surfaces [no-triality] [p0-ledger-ok]` |

Committed scope:
- `src/tac/fm_advisory.py`
- `tools/costate_digest.py`
- `src/tac/tests/test_fm_advisory.py`
- `src/tac/tests/test_costate_digest_fm_advisory.py`
- `.omx/research/ddm_fm2_20260806/RECEIPT.md`
- `.omx/research/ddm_fm2_20260806/NEXT_IF_RESUMED.md`
- `.omx/research/ddm_fm2_20260806/CHECKPOINTS.md`
- `.omx/research/ddm_fm2_20260806/fmtools_patches/APPLY.md`
- `.omx/research/ddm_fm2_20260806/fmtools_patches/0001-fmtools-full-sdk-surfaces.patch`
