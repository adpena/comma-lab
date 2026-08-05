# AU1 Measurement-Integrity Audit Receipt

Date: 2026-08-05

Scope: AU1 charter (`.omx/tmp/codex_runs/au1_prompt.md`) plus the common contract. No scorer forwards, no `upstream/evaluate.py`, no launch, no paid dispatch, no protected-file edits.

## Answer First

Leg 1 (#847 consumption-side coverage): **complete as a queryable surface.** Recalled gk2 as the seed inventory, then materialized `25/25` decode-read controls into JSONL with live source-site verification. Classification: `8 NONE`, `9 swept`, `5 levered`, `3 flagged`. Both #933 positive controls are present and classified `NONE`: `token_quant_levels_default_16` and `token_range_pm1_clamp`. Destination for future measured knobs is the existing GuardedConstant/P6 apparatus, not a new gate class.

Leg 2 (#953 correction/headline instruments): **complete as coarse detector surfaces.** Corrections index scanned `7,592` research memos, matched `2,733`, and emitted `11,840` candidate rows. Headline-vs-body scanned `7,592` memos, `522` current task-ledger rows, and `213` ledger-history git subjects, emitting `8,157` candidate rows. Positive control #933 was caught from ledger history (`23,655 B` stale subject side vs `-24,605 B` corrected body side). Negative control #931 was not caught.

Leg 3 (#840 registry-first extension): **disambiguated and partially folded.** Original `ddm_cf1` is `.omx/research/ddm_cf1_coarse_framing_audit_20260731.md` (task #840 unswept coarse-framing extension). The 2026-08-05 `ddm_cf1` is `.omx/research/ddm_cf1_20260805/CF1_CROSSWALK_RECEIPT.md` (conformal-anomaly crosswalk), not #840. The registry row records the collision and denominator refinement: `1,260` workspace `.omx/research/codex_findings_*.md`, `1,313` tracked `codex_findings_*.md` repo-wide and under `.omx/research/`. AU1 folds the correction/headline genus into Leg 2; a full coarse-framing classifier over all 1,313 remains separate.

If-budget checks: #885 currently does **not** reproduce the `50 vs true` history-count symptom (`git log --oneline | wc -l` = `14,158`; `git rev-list --count HEAD` = `14,158`), but three exact command consumers are flagged in JSONL. #867 is disambiguated: naked `#425 phase carrier` names both `src/tac/boundary_math/phase_residual_carrier.py` and `src/tac/boundary_math/dash_phase_carrier.py`; future rows should say `phase_residual_carrier_425` or `dash_phase_carrier_425`.

## Artifacts

- `.omx/research/ddm_au1_20260805/au1_decode_read_coverage.jsonl` - 25 receiver decode-read rows.
- `.omx/research/ddm_au1_20260805/au1_corrections_index.jsonl` - 11,840 correction-adjacent numeric rows.
- `.omx/research/ddm_au1_20260805/au1_headline_vs_body.jsonl` - 8,157 headline/body numeric-refutation rows.
- `.omx/research/ddm_au1_20260805/au1_registry_first_extension.jsonl` - #840/ddm_cf1 collision and counts.
- `.omx/research/ddm_au1_20260805/au1_if_budget_checks.jsonl` - #885/#867 optional rows.
- `.omx/research/ddm_au1_20260805/au1_summary.json` - machine-readable denominator summary.
- `tools/au1_measurement_integrity_audit.py` - generator for all JSONL surfaces.
- `tools/tests/test_au1_measurement_integrity_audit.py` - focused tests for #933, #931, decode denominator, and correction-window extraction.

## Recall Evidence

- `ddm_gk2_decode_read_coverage_20260804.md:45-69` supplied the binding #847 denominator (`25` decode-read controls), the `8/17` split, and both #933 positive controls. AU1 reused that inventory rather than re-deriving blind.
- `ddm_gk2_decode_read_coverage_20260804.md:96-105` named the 17 laddered decode-read controls; AU1 encoded them into the decode-read JSONL alongside the 8 `NONE` controls.
- `ddm_gk1_guarded_constant_20260803.md:75-81` shows P6 is declaration-driven; that is why AU1 does not build another gate class and why #847 remains the finder.
- `ddm_iv1_inventory_drain_20260803.md:315-318` supplied the #840 denominator refinement (`1,260` vs `1,313`), which AU1 rechecked in the generated registry row.
- `ddm_cf1_coarse_framing_audit_20260731.md:1-9` and `ddm_cf1_20260805/CF1_CROSSWALK_RECEIPT.md:1-11` distinguish original #840/ddm_cf1 from the 2026-08-05 conformal-anomaly CF1.

## Verification

- `.venv/bin/python -m pytest tools/tests/test_au1_measurement_integrity_audit.py` -> `4 passed`.
- `.venv/bin/python -m compileall -q tools/au1_measurement_integrity_audit.py tools/tests/test_au1_measurement_integrity_audit.py` -> pass.
- `.venv/bin/python tools/au1_measurement_integrity_audit.py --summary-json` regenerated all AU1 JSONL surfaces and reported `scorer_evaluate_launch = not_run`.

## Honest Limits

The correction and headline outputs are coarse candidate detectors, not adjudicated truth rows. They intentionally prefer recall over precision, so date/title numerics and dense correction windows will produce noise. A successor promoting them into a gate must add row triage, source allow/deny scopes, and value-pair adjudication.

No exact score was measured. No pointer moved. No contest-CPU/CUDA row was produced. All AU1 outputs are apparatus/measurement-integrity artifacts.

## NEXT_IF_RESUMED

1. Triage `au1_corrections_index.jsonl` and `au1_headline_vs_body.jsonl` into a smaller adjudicated ledger before treating any candidate row as a blocker.
2. After a scorer window measures actual quality/rate impacts, migrate the live decode knobs from `NONE` into GuardedConstant declarations rather than inventing a new gate class.
3. Build the general #840 coarse-framing classifier only if the target is all `1,313` tracked `codex_findings_*.md`; AU1 only covers the correction/headline numeric genus.
4. For #885, inspect the three flagged command consumers before citing a history-count denominator again.
5. For #867, retag naked `#425 phase carrier` references into raster `phase_residual_carrier_425` or curve-domain `dash_phase_carrier_425` when touching those rows.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
