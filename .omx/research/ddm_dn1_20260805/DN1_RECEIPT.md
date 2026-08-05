# DN1 #904 Receipt - Cross-Module Declared-On-Never-Read Detector

## Verdict

Implemented and landed the DN1 #904 detector as a bounded AST/import-graph audit plus warn-only preflight wrapper.

Controls passed before sweep output:

| control | expected | result |
|---|---:|---:|
| `tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.margin_targets` | hit | PASS |
| `...DirectDescriptionJointDescentMLXModule.seg_targets` | not hit | PASS |
| `...DirectDescriptionJointDescentMLXModule.pose_targets` | not hit | PASS |
| `...DirectDescriptionJointDescentMLXModule.margin_hinge_weight` | not hit | PASS |

Full DN1-bounded sweep:

| scope | files | parsed | parse errors | constructor fields | argparse declarations | import chains traced | none-path rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dn1` | 183 | 183 | 0 | 55 | 483 | 73 | 1 |

The detector found exactly one row: the canonical `margin_targets` positive control. The charter expectation of "at least 2 more" is not supported in this DN1 namespace. I am treating that as a refuted prediction, not as evidence of broader nonexistence.

## Scope

The shipped default scope is deliberately bounded to the tractable DN1 namespace:

| root | file count |
|---|---:|
| `src/tac/witness_dsl/*.py` | 176 |
| `src/tac/optimization/direct_description_joint_descent.py` | 1 |
| `src/tac/optimization/lane_guard.py` | 1 |
| direct-description caller tools | 4 |
| `experiments/train_levelset_witness_realized_through_R_mlx.py` argparse surface | 1 |

The CLI also has a `--scope broad` mode, but this receipt's verdict is only for the `dn1` denominator above.

## Recall

Searched before design and sweep:

| query | purpose |
|---|---|
| `#904|declared-on-never-read|margin_targets|_trainer_consumers|gd5|lever_registry` | find the prior #904 diagnosis and failed detectors |
| `margin_targets|derive_margin_floor` | locate the live margin row and its intended consumer |
| `DirectDescriptionJointDescentMLXModule(` | enumerate the caller surface |

Relevant prior evidence:

| source | fact used |
|---|---|
| `.omx/research/ddm_wk3_transcript_charter_recovery_20260803.md:24` | `margin_targets` is the named positive control; two prior single-file detectors missed it. |
| `.omx/research/ddm_qd1_backlog_drain_20260803.md:111` | prior #904 diagnosis conflated two classes; the flag-never-read issue was narrower than the old broad framing. |
| `.omx/research/ddm_gd5_grade5_detector_is_not_autoderivable_20260801.md:37-44,180-183` | old F1/F2/F3 formulations missed the positive control and were not separative. |
| `.omx/research/ddm_bo1_seg_base_objective_menu_order_20260803.md:95` | `margin_targets` was measured as declared and never read across 8 hardcoded-default call sites. |
| `.omx/research/ddm_rt2_realized_dseg_discarded_20260803.md:301-339` | `derive_margin_floor` is the intended derived-floor primitive; the live follow-on is weight/floor wiring, not a new floor-only claim. |

## Hit

| rank | key | declared | assigned | external constructor calls | production reads | routing |
|---:|---|---|---|---:|---:|---|
| 103 | `tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.margin_targets` | `src/tac/optimization/direct_description_joint_descent.py:2295` | `src/tac/optimization/direct_description_joint_descent.py:2316` | 8 | 0 | Read `self.margin_targets` on the live consumer path, or remove the parameter from external callers and docs. |

The eight call sites default `margin_targets` through direct or one-hop import paths:

| path | lines |
|---|---|
| `tools/launch_ddm_joint_descent.py` | 481, 1201, 1543, 2601 |
| `tools/run_ddm_j12_receiver_coordinate_custody.py` | 455, 1772 |
| `tools/measure_ddm_fd2_posenull_gn_disambiguation.py` | 156 |
| `tools/smoke_ddm_fd1_gn_engine.py` | 59 |

## Artifacts

| artifact | SHA-256 |
|---|---|
| `.omx/research/ddm_dn1_20260805/dn1_detector_report.json` | `0949a420541a9aad01eb924504159b197a3af0e2f26e5d6ed6400e544a5f35ee` |

The report schema is `cross_module_declared_never_read.v1`. It records controls, denominator, limitations, and full hit routing.

## Code

| file | role |
|---|---|
| `tools/audit_cross_module_declared_never_read.py` | Standalone AST/import-graph detector and CLI. |
| `tools/tests/test_audit_cross_module_declared_never_read.py` | Positive/negative controls, import-hop, test-only, argparse, parse-error, determinism, and mutation-gate tests. |
| `src/tac/preflight.py` | Warn-only `preflight_all()` wrapper for #904. |
| `src/tac/tests/test_check_cross_module_declared_never_read.py` | Live wrapper tests for warn-only and strict modes. |

## Verification

Executed:

| check | result |
|---|---:|
| `tools/audit_cross_module_declared_never_read.py --controls-only` | PASS |
| detector JSON generation for the DN1 scope | PASS |
| `pytest` on `tools/tests/test_audit_cross_module_declared_never_read.py` and `src/tac/tests/test_check_cross_module_declared_never_read.py` | 12 passed |
| `pytest` on `src/tac/tests/test_check_required_component_jsonl_read_validation.py` | 10 passed |
| `pytest` on `src/tac/tests/test_build_completeness_grades.py` | 58 passed |
| combined targeted `pytest` run across all four files above | 80 passed |
| `py_compile` on changed Python files | PASS |

## Follow-Ons

| id | disposition | fire order |
|---|---|---|
| DN1-F1 | QUEUED-WITH-FIRE-ORDER | In the next direct-description margin objective A/B landing, wire `margin_targets` into `DirectDescriptionJointDescentMLXModule` by reading the live field on the consumer path and deriving or consuming the margin floor through `derive_margin_floor`; expose the weight/floor knobs through the typed config in the same exercised change. Do not land plumbing alone. |
| DN1-F2 | FOLDED | #899 residual is already covered by `41ded0a918 Add validated required-component JSONL reader preflight`; no DN1 edit reopened it. |

## NEXT_IF_RESUMED

1. Use `tools/audit_cross_module_declared_never_read.py --controls-only` as the cheap health check for #904.
2. If widening the detector, run `--scope broad` and state a new denominator; do not merge broad output into this DN1 verdict.
3. Fire DN1-F1 only inside the exercised margin objective A/B so the parameter becomes live behavior, not another declared surface.

```json
{
  "task": "DN1 #904",
  "detector": "tools/audit_cross_module_declared_never_read.py",
  "report": ".omx/research/ddm_dn1_20260805/dn1_detector_report.json",
  "scope": "dn1",
  "controls_passed": true,
  "files": 183,
  "constructor_field_declarations": 55,
  "argparse_declarations": 483,
  "none_path_rows": 1,
  "positive_control_hit": "tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.margin_targets",
  "negative_controls_clean": [
    "tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.seg_targets",
    "tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.pose_targets",
    "tac.optimization.direct_description_joint_descent.DirectDescriptionJointDescentMLXModule.margin_hinge_weight"
  ],
  "follow_ons": [
    {
      "id": "DN1-F1",
      "disposition": "QUEUED-WITH-FIRE-ORDER"
    },
    {
      "id": "DN1-F2",
      "disposition": "FOLDED"
    }
  ]
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
