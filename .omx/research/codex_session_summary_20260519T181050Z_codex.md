# Codex Session Summary - Phantom-API Backfill Wave 2 Strict Flip

**UTC:** 2026-05-19T18:10:50Z
**Lane:** `lane_phantom_api_backfill_wave_2_20260519`
**Task:** `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_C`
**Score claim:** none

## Landed

- Catalog #287 direct strict scan reduced from 229 live violations to 0.
- `.omx/research` phantom-helper citations were backfilled through an
  append-only exact waiver ledger at
  `.omx/state/catalog_287_phantom_api_waivers.jsonl`, preserving historical memo
  bodies instead of inserting waiver comments into dated memos.
- Source comment/docstring multiplier and percentage claims now carry evidence
  tags such as `[prediction]`, `[MPS-research-signal]`, or `[advisory only]`.
- `tac.preflight.preflight_all()` now invokes
  `check_no_docstring_overstatement_without_evidence_tag(strict=True)`.
- File-level Catalog #287 proposal waivers were hardened after sister review:
  they no longer suppress active authority claims.
- Active authority dotted references now require the full module to import or
  the terminal symbol to exist on an importable parent module; callable-shaped
  terminal names no longer pass by appearance alone.
- HF Jobs provider capacity is now represented as a non-score-authority,
  plan-only provider contract with readiness probing.
- The Tishby IB pure remote script stage label no longer trips the wrapper
  stage implementation gate.
- `.omx/state/canonical_task_status.jsonl` is registered as append-only
  historical provenance in the artifact-kind registry.
- Catalog #206 checkpoint discipline was repaired with a checkpoint row plus a
  durable backfill memo/addendum for 32 post-cutoff serializer commits.
- `python -m tac.preflight` now invokes `_preflight_cli_main()` only after EOF,
  eliminating the CLI/import mismatch that made Catalog #286 believe later
  Catalog #324/#325/#326 callables were absent.

## Verification

- Catalog #287 direct strict invocation: `strict-ok`, 0 findings.
- `src/tac/tests/test_check_287_phantom_api_research_md_extension.py`: 56
  passed.
- Catalog #324/#325/#326 focused suite: 122 passed.
- Catalog #287/provider/readiness/task-status/checkpoint suite: 174 passed.
- `check_dispatch_wrapper_stages_implemented(strict=True)`: 0 findings.
- `bash -n scripts/remote_lane_substrate_tishby_ib_pure.sh`: clean.
- `ruff --select F821,E9` on touched Python: clean.
- `py_compile` on touched core Python: clean.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tac.preflight --no-codebase`:
  passed.
- Slow all-scope CLI preflight now progresses past Catalog #286/#287 and stops
  on the existing optimal-form dispatch blocker below.
- Timing profile captured at
  `.omx/research/codex_findings_preflight_all_timing_profile_20260519T183000Z_codex.md`:
  35.562608s wall, 89 timed steps, with broad scan hot spots led by
  `preflight_filename_contract`, `check_no_proxy_metric_drives_decision`,
  `check_no_compromised_lightning_supply_chain`, and `preflight_dead_resolvers`.

## Residual Blocker

Full `preflight_all()` now reaches
`check_substrate_at_optimal_form_before_paid_dispatch`, which reports 17
substrate lanes at LIFTED-TRAINER form with outstanding
`PROCEED_WITH_REVISIONS` verdicts. This is a real dispatch/council authority
blocker, not a Catalog #287 blocker. It should stay fail-closed until the
relevant substrate verdicts or dispatch readiness are canonically updated.

## Recommended Next Codex Step

Finish the PR body / pre-submission adversarial review surface already
materializing in `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md`,
then return to score-moving z7/PR95 local-training and export parity blockers
with the same authority-tag discipline now enforced by Catalog #287-v2.
