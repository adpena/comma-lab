# ddm_sw1 implementation specification

## Objective

Complete the two apparatus follow-ons in the ddm_sw1 charter without scorer work or SSD mutation:

1. Make the screening-law census zero by recording one evidence-bound historical exemption for each of the 93 charters that finished before Catalog #416 landed, then flip the screening lint to strict-by-default atomically.
2. Disposition every one of cs1's 182 blocked Python/shell blobs. Land only exact real-source bytes that have an owning memo, valid syntax, two visible review passes, Ruff for Python, and a ledgered repo-relative destination. Certify only exactly reproducible generated artifacts. Record every remainder as retain-with-owner with a concrete owner and reason.

## Constraints

- Do not touch `upstream/`, the scorer lane, payloads, SSD bytes, or the live trees named by the charter.
- Preserve the staged index and every unrelated dirty file.
- Historical screening exemptions must be one row per charter and bind the charter path, SHA-256, terminal evidence, terminal time/commit, and the Catalog #416 cutoff. A stale hash, missing evidence, future terminal time, duplicate conflict, or malformed row must not exempt.
- The linter must still detect new/unexempted violations. Change no screening-law semantics except making strict mode the default after the fresh zero census; `TAC_SCREENING_LAW_STRICT=0` remains the explicit emergency opt-out.
- Do not weaken `tools/audit_ssd_authored_signal.py` or treat `retain-with-owner` as preservation. Only Git reachability or a legitimate generated-artifact certificate may reduce its bucket-C count.
- Never invent a repo destination after copying: write the proposed disposition ledger first, then materialize exactly those approved destinations.
- Batch serializer landings at at most 25 files, with post-edit `--expected-content-sha256`, `[no-triality] [p0-ledger-ok]`, and no attribution trailer.
- Every changed `.py` receives two visible review passes plus Ruff. Shell receives full source review plus `bash -n`.

## Files and areas

- `tools/codex_arm_queue.py`
- `src/tac/tests/test_ddm_pm2_screening_law.py`
- `.omx/research/ddm_pm2_20260909/` screening census/fire-order evidence
- `.omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl` via append-only successor disposition rows or a clearly linked append-only successor ledger
- `.omx/research/ddm_sw1_20260910/` receipts, recovered exact sources, and manifests
- `.omx/research/ddm_sw1_screening_census_and_ssd_code_disposition_20260910.md`

## Acceptance criteria

- A fresh scan of all tracked charter Markdown produces 0 screening-law findings, with 93 individually validated historical exemption rows and no charter-body mutation required for those historical rows.
- `TAC_SCREENING_LAW_STRICT` defaults to strict and explicit `0` returns warn-only behavior.
- Focused screening tests pass, including stale-hash, post-cutoff, missing-evidence, and new-charter positive controls.
- All 182 prior BLOCKED blob SHA-1s have exactly one successor disposition: reachable/landed, certified-generated, or retain-with-owner.
- A fresh full SSD audit reports before/after denominators and does not hide unreadable files or absent roots. Target bucket C is at most 50; if source-quality gates make that impossible, preserve the exact residue and report the falsifier without weakening any rule.
- The final memo contains RECALL EVIDENCE, measured boundaries, prediction/falsifier accounting, both before/after counts, exact verification commands, the current own-vehicle frontier line, and extractor-safe NEXT_IF_RESUMED/LIVE-HYPOTHESES/DEAD-ENDS sections.

## Required tests

```text
.venv/bin/python -m pytest -q src/tac/tests/test_ddm_pm2_screening_law.py src/tac/tests/test_codex_arm_queue.py src/tac/tests/test_agent_path_charter_lint_parity.py
.venv/bin/ruff check tools/codex_arm_queue.py src/tac/tests/test_ddm_pm2_screening_law.py
.venv/bin/python tools/audit_ssd_authored_signal.py --write-cache --manifest .omx/research/ddm_sw1_20260910/ssd_audit_after.json
```

## Do not touch

- The three common-contract forbidden files.
- Any unrelated dirty state, staged index entry, live-arm source tree, SSD source byte, payload, or score pointer.
- Auditor bucket semantics or certification rules merely to improve the number.
