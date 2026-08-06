# ddm_fm1 Receipt - fmtools Advisory Classifiers

## SDK Capabilities Table

| Capability | Current Apple surface researched | fmtools surface inspected | Gap / action for Pact |
|---|---|---|---|
| On-device model availability | `apple-fm-sdk` 0.2.1 on PyPI, macOS 26 / Xcode 26 / Apple Intelligence requirements, `SystemLanguageModel.default` availability checks. | `fmtools` 0.0.218 wraps `SystemLanguageModel` behind an availability guard; local venv still has `apple-fm-sdk` 0.1.1. | Keep advisory fail-open. Do not make queue behavior depend on local Apple stack availability. |
| Guided structured output | Python SDK supports schema-guided generation through `LanguageModelSession.respond(..., generating=schema)`. Native Foundation Models expose `LanguageModelSession` guided responses. | `fmtools.local_extract(schema=...)` creates a fresh session per attempt and calls `respond(..., generating=schema)`. | Reuse existing `src/tac/fm_advisory.py` detached-subprocess wrapper instead of adding a new dependency path. |
| Tool calling | Python SDK documents `fm.Tool`, async `call`, generated argument schemas, and session tools. Native 2026 docs include tool-calling updates. | Current fmtools inspected surfaces do not expose Pact-specific tool-calling needs for this charter. | Out of scope for ddm_fm1. |
| Streaming | Python SDK documents `stream_response` / `session.stream_response` partial result iteration. | Current fmtools inspected surfaces provide sync extraction/classification path used by Pact. | Out of scope; queue lint must stay simple and bounded. |
| Transcript / resume | Python SDK documents session object usage; native docs expose transcript/session lifecycle. | `fmtools.backends.apple_sdk` has transcript export/resume helpers. | Not used; charter lint is stateless and read-only. |
| Context / model controls | Native docs expose context sizing, availability, generation options, use cases, guardrails, dynamic profiles, and newer 2026 model surfaces. | Current fmtools wrap provides a stable minimal surface. | Record only as SDK research. No gate or scorer behavior changes. |

Sources used:
- PyPI `apple-fm-sdk`: https://pypi.org/project/apple-fm-sdk/
- Python Apple Foundation Models SDK docs: https://apple.github.io/python-apple-fm-sdk/
- Python Tools guide: https://apple.github.io/python-apple-fm-sdk/docs/guides/tools/
- Python Streaming guide: https://apple.github.io/python-apple-fm-sdk/docs/guides/streaming/
- Apple native Foundation Models docs/search surfaces: `SystemLanguageModel`, `LanguageModelSession`, and Foundation Models updates pages on developer.apple.com.

## External fmtools Pin

Read-only inspected checkout:

| Field | Value |
|---|---|
| path | `/Users/adpena/Projects/fmtools` |
| git commit | `c9e755539da22df8aee6c5c22fa6653253456a4f` |
| package version | `fmtools` 0.0.218 |
| local SDK package | `apple-fm-sdk` 0.1.1 |
| `pyproject.toml` sha256 | `f812bed0d5535af02f1dd53d30738b134f5f188bcbdf3a210a24a16281b07c3d` |
| `fmtools/decorators.py` sha256 | `d0e43e92791e6efebcf604d048502f6286dcb1ba9c560d5e868b39ab21dcaff3` |
| `fmtools/backends/apple_sdk.py` sha256 | `52e67852cefae30598a01fd505ba5b2cc5f12d0bc93dbade5981e4aefadeb65a` |
| `fmtools/exceptions.py` sha256 | `a9bc5478dffc99ec750c0d84c4bc4964fe5ba4dfe782c368164aa24337110f22` |

Relevant inspected behavior:
- `local_extract(schema, retries=3, debug_timing=False, instructions=None)` creates/caches the model, checks availability, creates a fresh session, and calls `respond(input_text, generating=schema)`.
- `backends/apple_sdk.py` wraps `SystemLanguageModel`, `LanguageModelSession(model=raw_model, instructions=instructions)`, `.respond(prompt, generating=...)`, transcript export, and transcript resume.
- `exceptions.py` raises setup errors when `model.is_available()` is false.

## Implemented Surfaces

Code surfaces:
- `src/tac/fm_advisory.py`
  - `CHARTER_CLASS_LABELS = ("build_race_train_measure", "audit_analysis", "convocation", "mixed")`
  - `charter_class(text, timeout=25) -> dict | None`
  - `MECHANISM_REDUCTION_LANGUAGE_LABELS = ("quick-train", "undersized", "toy-scale", "convenience-basis")`
  - `mechanism_reduction_language(text, timeout=25) -> dict | None`
- `tools/codex_arm_queue.py`
  - `_charter_is_build_by_tokens(text)`
  - `_fm_advisory_module()`
  - `lint_charter_fm_advisories(prompt_path)`
  - `cmd_add` prints fmtools advisory lines as `charter-lint WARN`.

Consumer behavior:
- `lint_charter_optimal_form` remains deterministic and token/block based.
- `TAC_CHARTER_LINT_STRICT=1` still refuses only deterministic optimal-form problems.
- fmtools advisory output never rescues and never refuses a charter.
- If fmtools or the Apple model is unavailable, `_fm_advisory_module()` returns `None` and queue output is byte-identical to the no-fmtools path.
- If fmtools returns only unclassified mechanism rows, `mechanism_reduction_language` returns `None` rather than treating the run as an empty-flags success.

## Recall Evidence

Searched/consulted before implementation:
- Memory registry for Pact frontier and lane-custody directives; no direct fmtools-specific hit was found.
- `.omx/research` / DAG surfaces for fmtools context; the existing `fmtools_costate_organ` thread established the pattern of advisory, non-promotable fmtools use.
- In-repo code surfaces `src/tac/fm_advisory.py`, `tools/codex_arm_queue.py`, `tools/costate_digest.py`, and `src/tac/tests/test_fm_advisory.py`.
- Canonical equations search did not find a direct fmtools equation surface.

Decision from recall:
- Reuse `src/tac/fm_advisory.py` and keep queue lint fail-open.
- Do not route any scorer, exact-row, or frontier decision through fmtools.

## Verification

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_fm_advisory.py src/tac/tests/test_codex_arm_queue.py
.venv/bin/python -m ruff check src/tac/fm_advisory.py tools/codex_arm_queue.py src/tac/tests/test_fm_advisory.py src/tac/tests/test_codex_arm_queue.py --select F
.venv/bin/python tools/review_tracker.py scan --since HEAD --repo .
.venv/bin/python tools/review_tracker.py mark-file --file src/tac/fm_advisory.py --reviewer codex --pass-id ddm_fm1_pass1b
.venv/bin/python tools/review_tracker.py mark-file --file src/tac/fm_advisory.py --reviewer codex --pass-id ddm_fm1_pass2b
.venv/bin/python tools/review_tracker.py mark-file --file src/tac/tests/test_fm_advisory.py --reviewer codex --pass-id ddm_fm1_pass1b
.venv/bin/python tools/review_tracker.py mark-file --file src/tac/tests/test_fm_advisory.py --reviewer codex --pass-id ddm_fm1_pass2b
```

Results:
- `pytest`: 54 passed in 0.31s.
- `ruff --select F`: all checks passed.
- Review tracker: new/changed `.py` functions were marked through two passes. The first local reviewer label (`codex_fm1`) did not satisfy the hook principal table, so the files were re-marked with principal `codex`; final policy checks reported 0 violations for all four touched Python files.

## Live fmtools Check

Observed:
- `src.tac.fm_advisory.available()` returned true on this host.
- `/Users/adpena/Projects/fmtools/.venv/bin/python` imported `fmtools` and `apple_fm_sdk`.
- The model availability probe returned available.
- `charter_class(.omx/tmp/codex_runs/fm1_prompt.md)` returned `None`.
- `mechanism_reduction_language(.omx/tmp/codex_runs/fm1_prompt.md)` returned unclassified `RuntimeError` rows under the installed external fmtools / `apple-fm-sdk` 0.1.1 stack; after the code fix, Pact maps this to `None`.

Verdict:
- The SDK/source capability research is pinned.
- The Pact wrapper and queue consumer are implemented and tested.
- The live on-device generation leg is UNVERIFIED-IN-SANDBOX for this charter.

## Boundaries

- No scorer ownership was claimed.
- No `upstream/` files were touched.
- The common-contract forbidden files were not edited.
- No exact eval was run.
- No archive was produced.
- No pointer moved.
- No evidence path in this receipt depends on `/tmp`.

## Serializer Status

Main-checkout serializer attempt:

```bash
REVIEW_GATE_HOOK_RETRY_SECONDS=8 .venv/bin/python tools/subagent_commit_serializer.py \
  --message "ddm_fm1: add fmtools charter lint advisories [no-triality] [p0-ledger-ok]" \
  --files src/tac/fm_advisory.py tools/codex_arm_queue.py \
    src/tac/tests/test_fm_advisory.py src/tac/tests/test_codex_arm_queue.py \
    .omx/research/ddm_fm1_20260806/RECEIPT.md \
    .omx/research/ddm_fm1_20260806/NEXT_IF_RESUMED.md \
    .omx/research/ddm_fm1_20260806/CHECKPOINTS.md \
  --expected-content-sha256 ... \
  --triality-legs none \
  --triality-reason "apparatus-only fmtools advisory lint and receipts; no scorer, equations, or vehicle change" \
  --no-co-author
```

Result:
- The first serializer attempt reached pre-commit but the review gate blocked on an unrecognized reviewer label; this was corrected with `codex` review marks and 0-violation policy checks.
- The corrected serializer attempt then failed before commit at `git add`: `error: unable to create temporary file: Operation not permitted`; `fatal: updating files failed`.
- Post-failure `git diff --cached --name-only` was empty.
- Main checkout commit is therefore UNLANDED due managed-sandbox Git object write denial, not due code/test/review failure.

Own-vehicle frontier line:
- Own-vehicle advisory frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
- Contest pointer remains the borrowed/unmoved `0.1910828242` line from main hot state.

## Unverified Legs

- Live Apple on-device classifier generation under the external fmtools checkout failed to produce classified rows in this sandbox.
- The external fmtools checkout was inspected but not upgraded to `apple-fm-sdk` 0.2.1.
- The charter-requested "ty1-class audits" are enabled by the new reusable `mechanism_reduction_language` surface but no separate ty1 audit consumer was wired in this unit.
- No exact-score, scorer, or public-evaluator claim is made.
