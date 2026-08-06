# ddm_fm2 Checkpoints

| Checkpoint | Status | Evidence |
|---|---|---|
| Governing contract read | COMPLETE | Read `.omx/tmp/codex_runs/fm2_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` excerpt in checkout, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`. |
| External repo located | COMPLETE | `/Users/adpena/Projects/fmtools`, commit `c9e755539da22df8aee6c5c22fa6653253456a4f`, clean `main`. |
| External writability | BLOCKED | `test -w /Users/adpena/Projects/fmtools` returned rc=1 under this sandbox; no in-place branch attempted. |
| Latest SDK verification | PARTIAL | Browser/PyPI verified `apple-fm-sdk` latest `0.2.1`; shell install into SSD venv failed on DNS, so latest runtime was not installed. |
| fmtools patch series | COMPLETE | `.omx/research/ddm_fm2_20260806/fmtools_patches/0001-fmtools-full-sdk-surfaces.patch`, sha256 `9804876bec64c21b1de70470bedfd276d12e3c0ab6cace3ef3c24ecbbd6340ee`; apply-check clean. |
| fmtools verification clone | COMPLETE | `/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/fmtools_patch_verify`, 7.6M tracked clone at the expected base plus patch. |
| fmtools tests | COMPLETE | `PYTHONPATH=... /Users/adpena/Projects/fmtools/.venv/bin/python -m pytest tests` -> 652 passed, 12 skipped. |
| fmtools lint | COMPLETE | `/Users/adpena/Projects/fmtools/.venv/bin/ruff check ...` -> all checks passed. |
| Pact consumer wire-through | COMPLETE | `src/tac/fm_advisory.py` prefers `fmtools.respond(..., generating=Choice)` when available, keeps `local_extract` fallback, adds `capability_report`; `tools/costate_digest.py` prints the capability line. |
| Pact tests | COMPLETE | `.venv/bin/python -m pytest src/tac/tests/test_fm_advisory.py src/tac/tests/test_costate_digest_fm_advisory.py src/tac/tests/test_codex_arm_queue.py` -> 65 passed. |
| Pact lint | COMPLETE | `.venv/bin/python -m ruff check src/tac/fm_advisory.py tools/costate_digest.py src/tac/tests/test_fm_advisory.py src/tac/tests/test_costate_digest_fm_advisory.py --select F` -> all checks passed. |
| Pact review tracking | COMPLETE | `tools/review_tracker.py scan`, then two whole-file review mark cycles for `src/tac/fm_advisory.py`, `tools/costate_digest.py`, `src/tac/tests/test_fm_advisory.py`, `src/tac/tests/test_costate_digest_fm_advisory.py`. This tracker build has no pass-id CLI. |
| Live capability report | COMPLETE | Current external venv reports `apple-fm-sdk 0.1.1`, model_available true, guided/json_schema/tools/streaming/options/transcripts/model-controls all true. |
| Live generation round-trip | BLOCKED | Plain `LanguageModelSession.respond("Say ok")` fails with `GenerationError status 255: None`; structured `respond(..., generating=Tiny)` fails with `GenerationError status 255` and underlying `ModelManagerServices.ModelManagerError Code=1008`. |
| Scorer / exact eval | NOT RUN | Charter is code+research only; main hot state already has all scorer slots occupied elsewhere; no scorer slot claimed. |
| Serializer commit | PENDING | To be attempted after receipt creation with post-edit SHA pins and `[no-triality] [p0-ledger-ok]`. |
