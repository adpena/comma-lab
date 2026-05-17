# L5 v2 TT5L exact-dispatch authority hardening

Date: 2026-05-17

Scope: Time-Traveler L5 v2 side-info effect-curve dispatch planning.

## Finding

The TT5L side-info effect-curve plan could mark five paired Modal work units as
operator-ready from variant/archive custody plus paired-command syntax alone.
That was weaker than the shared exact-dispatch authority gate used by top-k and
other exact-eval fan-out surfaces. In the live artifacts, the uploaded runtime
tree has `inflate.sh` and the variant archives exist, but the runtime lacks
`report.txt` and per-variant `archive_manifest.json` custody for the archive
being sent to `tools/dispatch_modal_paired_auth_eval.py`.

This was an L5-v2 actuator issue, not a score result. No provider dispatch was
attempted and no score/promotion/rank claim is created here.

## Change

`src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py` now
calls `tac.optimizer.exact_dispatch_authority.exact_dispatch_authority()` for
both `contest_cpu` and `contest_cuda` lanes for every TT5L side-info variant.
The work unit is operator-ready only when both axis verdicts authorize:

- target mode includes `contest_exact_eval`;
- archive bytes/SHA/path match the file on disk;
- submission runtime has executable `inflate.sh`;
- runtime dependency tree is hashable;
- `report.txt` is present;
- archive manifest is present and matches the scored archive;
- score-affecting byte change proof exists;
- same-lane active/terminal claim conflicts are checked.

## Live State After Rebuild

The durable dispatch plan was rebuilt after the patch:

- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.md`

Current result:

- `work_unit_count=5`
- `ready_work_unit_count=0`
- `ready_for_operator_dispatch=false`
- `score_claim=false`
- `dispatch_attempted=false`

Live blockers are the useful next work:

- `exact_dispatch_authority:contest_cpu:report_txt_missing`
- `exact_dispatch_authority:contest_cpu:archive_manifest_missing`
- `exact_dispatch_authority:contest_cuda:report_txt_missing`
- `exact_dispatch_authority:contest_cuda:archive_manifest_missing`

## Next Action

Before executing the TT5L side-info effect curve, materialize a small
submission-runtime `report.txt` and per-variant archive manifests whose
archive SHA/bytes/member names match the exact variant archives. Then rebuild
the dispatch plan and only run paired Modal when `ready_work_unit_count=5`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py -q`
- `.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py`
