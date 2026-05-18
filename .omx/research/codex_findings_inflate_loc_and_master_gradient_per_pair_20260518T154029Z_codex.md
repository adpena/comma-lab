# Codex Findings: Catalog #328, Per-Pair Master-Gradient, and Latest T3 Plan Review

Date: 2026-05-18T15:40:29Z

Reviewer: Codex, with xhigh side reviews from Hubble and Ramanujan.

## Executive Verdict

PROCEED_WITH_REVISIONS.

This pass converted the current review findings into concrete hardening:

- Claimed Catalog #328 transactionally for `check_submission_inflate_py_under_loc_budget`; Catalog #327 remains master-gradient contest-axis custody.
- Added a reusable `tac.submission_inflate_loc_budget` helper, operator CLI `tools/audit_submission_inflate_py_loc_budget.py`, warn-only preflight wire-in, and focused tests.
- Fixed the stale planner text that recommended the nonexistent `tools/extract_master_gradient.py --target local-cpu` command.
- Preserved per-pair master-gradient use cases while making anchor selection depend on `gradient_tensor_kind`, not fragile `measurement_method` substrings.
- Made DuckDB `per_byte_sensitivity` rows explicitly non-promotable derived planning rows, even when their source anchor is contest-authoritative.
- Added a regression test proving the MAE patch arithmetic correction: `384x512`, `patch_size=16`, `mask_ratio=0.25` means 768 patches, 192 masked patches, 49,152 masked pixels, exactly 25 percent.

## Findings And Resolutions

### F1: Catalog #327 collision

The T3 synthesis memo proposed Catalog #327 for operator-attention cadence, and the inflate.py symposium memo also referenced #327 for `check_submission_inflate_py_under_loc_budget`. Catalog #327 is already occupied by `check_master_gradient_contest_axis_requires_authoritative_custody`.

Resolution: claimed #328 with `tools/claim_catalog_number.py claim --commit-via-serializer`, added CLAUDE.md catalog entry #328, and wired the inflate LOC audit under #328.

### F2: `--target local-cpu` dead CLI authority

The live extractor parser has `--archive`, `--inflate-py`, `--upstream-dir`, `--axis`, `--device`, and `--output-npy`; it has no `--target`.

Resolution: `src/tac/empirical_per_x_optimal_codec_planner/per_byte_strategy.py` now emits an actionable real command shape and explicitly recommends `[macOS-CPU advisory]` for local planning. The historical posterior row is left append-only; a correction anchor was appended through `tac.council_continual_learning.append_council_anchor`.

### F3: Inflate.py LOC budget is a review guard, not score evidence

The contest score charges `archive.zip` bytes, not `inflate.py` source bytes. Compressing source directly has zero rate-term effect. Large `inflate.py` files still matter because they hide runtime-closure bugs and make review harder.

Resolution: Catalog #328 is warn-only at landing. Live audit count is 14 direct tracked submissions over 200 physical lines. The gate message states that this is not score evidence and points to helper extraction or an explicit source-faithful waiver.

### F4: DuckDB per-byte sensitivity false promotion eligibility

Ramanujan found that derived `per_byte_sensitivity` rows marked `promotion_eligible=True` when sourced from contest-axis anchors. That is too strong: a per-byte sensitivity row is a planning read-model row, not leaderboard evidence.

Resolution: rows now carry `source_anchor_authoritative` separately while forcing `promotion_eligible=False`. Contest-authoritative sources get `evidence_grade="diagnostic_from_contest_authoritative_source"`.

### F5: Per-pair anchor selection was method-name fragile

The loader previously inferred aggregate vs per-pair from whether `measurement_method` contained `"per_pair"`. That can misclassify corrected/advisory rows and future extractor names.

Resolution: loaders prefer `gradient_tensor_kind` and keep method-name fallback only for legacy rows.

### F6: MAE+SAUG memo arithmetic is wrong

The untracked MAE+SAUG memo claims `384x512`, `patch_size=16`, `mask_ratio=0.25` masks 79 percent or 3.1x too many pixels. Correct arithmetic:

- `384 / 16 = 24`
- `512 / 16 = 32`
- `24 * 32 = 768` patches
- `0.25 * 768 = 192` masked patches
- `192 * 256 = 49,152` masked pixels
- `49,152 / 196,608 = 0.25`

Resolution: added a unit test so this exact geometry cannot re-enter silently. I did not edit the partner memo in place; future memo correction should append a supersession/correction section.

## Remaining Plan Risks

- The MobileNetV3-as-SegNet-surrogate veto is correctly stated in the T3 memo, but adjacent HF Jobs plans still contain MobileNetV3 smoke rows. Next hardening should add a lightweight research-plan audit that rejects `mobilenetv3` plus `SegNet surrogate` unless it is explicitly baseline-only and paired with per-pixel-logit proof.
- Public Trackio/Gradio wording is still inconsistent across memos. The implementation default should be private Space/private dashboard unless a public-release manifest approves exposure.
- Operator-attention cadence should use the existing `tools/audit_council_tier_cadence.py` surface; do not create a duplicate cadence gate unless that auditor proves insufficient.

## Online Source Cross-Check

Online sources were used only to sanity-check ecosystem claims, not to override local contest evidence:

- Python `zipapp` official docs: https://docs.python.org/3/library/zipapp.html
- Python-minifier project page: https://pypi.org/project/python-minifier/
- Hugging Face repository settings / visibility docs: https://huggingface.co/docs/hub/repositories-settings

The local contest contract remains the authority for score impact.

## Verification

- `.venv/bin/python -m py_compile src/tac/submission_inflate_loc_budget.py tools/audit_submission_inflate_py_loc_budget.py src/tac/preflight.py src/tac/master_gradient.py src/tac/master_gradient_consumers.py src/tac/canonical_duckdb/per_byte_sensitivity_ext.py src/tac/empirical_per_x_optimal_codec_planner/per_byte_strategy.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py src/tac/tests/test_master_gradient_consumers.py src/tac/canonical_duckdb/tests/test_per_byte_sensitivity_ext.py src/tac/empirical_per_x_optimal_codec_planner/tests/test_per_byte_planner_emits_sensitivity_mask_aware_quantizr_v1.py src/tac/tests/test_mae_mask_aug.py`
- `.venv/bin/python tools/audit_submission_inflate_py_loc_budget.py --json`
- `check_master_gradient_contest_axis_requires_authoritative_custody(strict=True, verbose=True)`

Focused pytest result: 72 passed.
