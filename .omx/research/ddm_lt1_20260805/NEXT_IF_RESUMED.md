# ddm_lt1 NEXT_IF_RESUMED

Status: receipt written, scorer-free, no launch, no score claim.

Current artifacts:

- `.omx/research/ddm_lt1_20260805/RECEIPT.md`
- `.omx/research/ddm_lt1_20260805/NEXT_IF_RESUMED.md`
- Pre-existing unrelated `.omx/research/ddm_lt1_20260805/LT1_RECEIPT.md` was left intact.

## Resume steps

1. Re-run the registry census before acting:
   - `.venv/bin/python - <<'PY'` with `lever_registry.build_completeness()`,
     `lever_registry.completeness()`, and
     `lever_registry.completeness(Path("experiments/train_tr1_partition_renderer_mlx.py"))`.
2. Re-run `spec_v10_status(Path("."))`.
   - Create a dated v10 addendum only if the blocker set changed.
3. Do not launch from LT1. If a scorer or training action is selected, claim a
   lane/slot first and use the owning fire order:
   - WL1-LB, WL1-SR, WL1-EIK for the top three old-witness reopen rows.
   - TP1/BI1/TK1/LA1/DY2/WP1 for already-owned TR1 rows.
4. Treat `completeness(Path("experiments/train_tr1_partition_renderer_mlx.py"))`
   as diagnostic only unless `lever_registry.completeness()` is upgraded to scan
   all package factories. The current live anti-orphan denominator is
   `build_completeness().total == 198`.
5. For any STUB row, build and test the real trainer/receiver consumer first;
   no scorer A/B is admissible while `missing_flags` is non-empty.

## Current queue

```json
{
  "schema": "ddm_lt1_next_if_resumed.v1",
  "score_claim": false,
  "scorer_runs_by_lt1": 0,
  "launches_by_lt1": 0,
  "factory_denominator": 198,
  "bucket_counts": {
    "TR1-portable": 73,
    "vehicle-agnostic": 19,
    "witness-scoped": 106
  },
  "legacy_completeness": {
    "trainer": "experiments/train_levelset_witness_realized_through_R_mlx.py",
    "unmapped": 80,
    "stale": 3
  },
  "spec_v10_status": {
    "changed_since_wl1": false,
    "clear": false,
    "blocker_count": 7
  },
  "primary_next_actions": [
    "WL1-LB analytic lane-band training-lever A/B after scorer slot and clean stage boundary",
    "WL1-SR exact S_R reachability weighting A/B after gt_n600_sR.npz custody revalidation",
    "WL1-EIK fixed-guard eikonal/viscosity fair reopen after clean checkpoint provenance",
    "Recompile TP1 tickets after TK1 if a crossed PE3/cheapdct4/BI1 object is pursued"
  ],
  "do_not_do": [
    "do not treat legacy completeness(Path(TR1)) unmapped rows as live anti-orphan truth",
    "do not duplicate TP1, TK1, LA1, DY2, WP1, or WL1 owned rows",
    "do not launch or score from LT1",
    "do not fire STUB rows before missing_flags is empty"
  ]
}
```

Frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`;
contest pointer borrowed/unmoved.
