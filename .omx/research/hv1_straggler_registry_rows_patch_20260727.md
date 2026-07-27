# hv1 straggler harvest — lane_registry.json rows patch (2026-07-27)

Per charter `ddm_hv1_straggler_harvest`: every lane_registry.json merge conflict on the
integration branch `hv1/straggler_integration_20260727` was resolved to MAIN's version
(main's live registry is HOT — a parallel session holds it dirty). The branch-added rows
are carried HERE, as the exact `merge-base..branch` diff of `.omx/state/lane_registry.json`
per branch, for MAIN to re-apply at a quiet boundary via `tools/lane_maturity.py add-lane`
+ `mark` (preferred; the driver `tools/merge_lane_registry.py` refused these merges with
schema-validation errors against the current registry, so raw-diff re-application is NOT
recommended without the CLI).

Lane rows owed to main (verified ABSENT from main@29632337f3 except the last):

| lane id | source branch | on main? |
|---|---|---|
| lane_closed_scorer_variational_de_20260721 | closed_scorer_variational_de_20260721T172114Z | NO — owed |
| lane_einstein_kolmogorov_ultra_20260721 | einstein_kolmogorov_ultra_20260721T150001Z | NO — owed |
| lane_direct_description_minimizer_builder_603_20260721 | direct_description_minimizer_builder_20260721T221054Z | NO — owed |
| lane_bev_staticity_developability_probe_20260721 | bev_staticity_developability_probe_20260721T165801Z | NO — owed |
| lane_bev_staticity_v2_absolute_trajectory_20260721 | bev_staticity_v2_absolute_trajectory_20260721T174021Z | NO — owed |
| lane_g2g2_joint_multichart_solve_20260721 | g2g2_joint_multichart_solve_20260721T163037Z | NO — owed |
| lane_ddm_ms1_min_description_lattice_solve_20260723 | ddm_ms1_min_description_lattice_solve_20260723T233549Z | YES — already registered, no action |

## Branch: codexwt/closed_scorer_variational_de_20260721T172114Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 47c3fb143a..3dbab9ecdd 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -82261,6 +82261,48 @@
         }
       },
       "notes": "Delegated apparatus-only removal of obsolete inflate.py line-count enforcement; pointer unmoved; MAIN review required"
+    },
+    {
+      "id": "lane_closed_scorer_variational_de_20260721",
+      "name": "Closed scorer variational differential equations",
+      "phase": 0.0,
+      "level": 0,
+      "gates": {
+        "impl_complete": {
+          "status": false,
+          "evidence": ""
+        },
+        "real_archive_empirical": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": false,
+          "evidence": ""
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "Delegated task-space scorer functional and D1 fidelity gate; research_only=true; no score or dispatch authority; MAIN review required.",
+      "research_only": true
     }
   ],
   "level_rules": {
@@ -82270,5 +82312,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T15:33:40Z"
+  "updated_at": "2026-07-21T17:28:26Z"
 }
```

## Branch: codexwt/einstein_kolmogorov_ultra_20260721T150001Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index c65d1af28b..7fb1d6bb0b 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -81990,6 +81990,52 @@
       },
       "notes": "Task #578-R5 full n600 macOS-CPU advisory reverse-waterfill: D1 premise falsified; one #336 singleton admitted; composed receipt_v3; no paid row; pointer 0.1910828242 contest-CPU unmoved; research_only; MAIN landing review required.",
       "research_only": true
+    },
+    {
+      "id": "lane_einstein_kolmogorov_ultra_20260721",
+      "name": "Einstein Kolmogorov level-attributed RD and byte-close audit",
+      "phase": 1.0,
+      "level": 1,
+      "gates": {
+        "impl_complete": {
+          "status": true,
+          "evidence": "src/tac/optimization/einstein_kolmogorov_frontier.py"
+        },
+        "real_archive_empirical": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": true,
+          "evidence": "src/tac/tests/test_einstein_kolmogorov_frontier.py"
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "Research-only strict custody compiler; four current-scope no-candidate rows; U3 description tuple refused; generic runtime code free; no paid dispatch; pointer unchanged; MAIN landing review required.",
+      "research_only": true,
+      "reactivation_criteria": [
+        "custody-qualified-n600-candidate",
+        "complete-216KB-class-receiver-tuple"
+      ]
     }
   ],
   "level_rules": {
@@ -81999,5 +82045,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T10:45:47Z"
+  "updated_at": "2026-07-21T16:22:45Z"
 }
```

## Branch: codexwt/direct_description_minimizer_builder_20260721T221054Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 707dacdcc5..b3e1750cd5 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -82388,6 +82388,49 @@
       },
       "notes": "FORMULATION-scoped same-artifact verdict; FAMILY strengthening refused because direct-description grammar remains open; direct-description minimizer PRIMARY, #366 FALLBACK-only; no dispatch; MAIN review required.",
       "research_only": true
+    },
+    {
+      "id": "lane_direct_description_minimizer_builder_603_20260721",
+      "name": "Task 603 direct-description operations-grammar minimizer builder",
+      "phase": 3.0,
+      "level": 0,
+      "gates": {
+        "impl_complete": {
+          "status": false,
+          "evidence": ""
+        },
+        "real_archive_empirical": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": false,
+          "evidence": ""
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "STATIC_SCAFFOLD_LANDED_LAUNCH_READINESS_FALSIFIED; all maturity gates remain false; execution_allowed=false; no receiver/optimizer/score/dispatch; pointer 0.1910828242 contest-CPU unmoved; MAIN landing review required",
+      "research_only": true,
+      "lane_class": "research_substrate"
     }
   ],
   "level_rules": {
@@ -82397,5 +82440,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T20:59:01Z"
+  "updated_at": "2026-07-21T23:59:14Z"
 }
```

## Branch: codexwt/bev_staticity_developability_probe_20260721T165801Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 47c3fb143a..fe1ad41e63 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -82261,6 +82261,53 @@
         }
       },
       "notes": "Delegated apparatus-only removal of obsolete inflate.py line-count enforcement; pointer unmoved; MAIN review required"
+    },
+    {
+      "id": "lane_bev_staticity_developability_probe_20260721",
+      "name": "BEV staticity and developability probe",
+      "phase": 0.0,
+      "level": 2,
+      "gates": {
+        "impl_complete": {
+          "status": true,
+          "evidence": "tools/measure_bev_staticity_developability.py"
+        },
+        "real_archive_empirical": {
+          "status": true,
+          "evidence": ".omx/research/bev_staticity_developability_probe_20260721T172426Z_receipt.json"
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": false,
+          "evidence": ""
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "",
+      "research_only": true,
+      "reactivation_criteria": [
+        "C1 full-hood n64+n600 p50<=1px",
+        "C1 full-hood n64+n600 static_fraction>=0.5",
+        "rerun Road/Lane only with same solved xi and #327 geometry"
+      ]
     }
   ],
   "level_rules": {
@@ -82270,5 +82317,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T15:33:40Z"
+  "updated_at": "2026-07-21T17:26:28Z"
 }
```

## Branch: codexwt/bev_staticity_v2_absolute_trajectory_20260721T174021Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 47c3fb143a..a004eeebe7 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -82261,6 +82261,51 @@
         }
       },
       "notes": "Delegated apparatus-only removal of obsolete inflate.py line-count enforcement; pointer unmoved; MAIN review required"
+    },
+    {
+      "id": "lane_bev_staticity_v2_absolute_trajectory_20260721",
+      "name": "BEV staticity v2 absolute trajectory",
+      "phase": 1.0,
+      "level": 1,
+      "gates": {
+        "impl_complete": {
+          "status": true,
+          "evidence": "tools/measure_bev_staticity_developability.py"
+        },
+        "real_archive_empirical": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": false,
+          "evidence": ""
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "D0 PASS at n64 and n600; exact G1-calibrated chart Road/Lane D1-D2 negative; D3 blocked; macOS-CPU advisory; no score; MAIN landing review required",
+      "research_only": true,
+      "reactivation_criteria": [
+        "New hash-custodied absolute-motion source or independently admitted calibration preserves D0 and passes both Road and Lane n600 staticity before D3"
+      ]
     }
   ],
   "level_rules": {
@@ -82270,5 +82315,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T15:33:40Z"
+  "updated_at": "2026-07-21T18:37:27Z"
 }
```

## Branch: codexwt/g2g2_joint_multichart_solve_20260721T163037Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 8bd5a9b301..52f686892a 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -82220,6 +82220,52 @@
       "reactivation_criteria": [
         "Reactivate only with depth-stratified screw scene framing plus Fisher-margin curvelet/shearlet chart coefficients under a new governed n16 gate."
       ]
+    },
+    {
+      "id": "lane_g2g2_joint_multichart_solve_20260721",
+      "name": "G2g2 joint multi-chart coefficient solve",
+      "phase": 1.0,
+      "level": 2,
+      "gates": {
+        "impl_complete": {
+          "status": true,
+          "evidence": "tools/measure_realization_g2_lattice.py"
+        },
+        "real_archive_empirical": {
+          "status": true,
+          "evidence": ".omx/research/g2g2_joint_multichart_solve_20260721T174416Z.json"
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": true,
+          "evidence": "tools/tests/test_measure_realization_g2_lattice.py"
+        },
+        "three_clean_review": {
+          "status": false,
+          "evidence": ""
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "MEASURED_G2G2_RATE_BREAK_EVEN_STOP_FAMILY_OPEN; 0/6 admitted on macOS-CPU advisory; U1/P0/n64/n600 routes false; pointer unmoved; MAIN review required.",
+      "research_only": true,
+      "lane_class": "substrate_engineering",
+      "reactivation_criteria": [
+        "Reactivate only with a materially different nonlinear or xi-coupled chart formulation behind the same exact receiver and marginal-price gates."
+      ]
     }
   ],
   "level_rules": {
@@ -82229,5 +82275,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-21T15:33:40Z"
+  "updated_at": "2026-07-21T17:47:27Z"
 }
```

## Branch: codexwt/ddm_ms1_min_description_lattice_solve_20260723T233549Z
```diff
diff --git a/.omx/state/lane_registry.json b/.omx/state/lane_registry.json
index 552883ece1..ccae44e4ec 100644
--- a/.omx/state/lane_registry.json
+++ b/.omx/state/lane_registry.json
@@ -85189,6 +85189,48 @@
       },
       "notes": "Local-only n600 [macOS-CPU frozen-scorer advisory] measurement; research_only=true; score_claim=false; no paid/remote/contest eval; MAIN landing review required.",
       "research_only": true
+    },
+    {
+      "id": "lane_ddm_ms1_min_description_lattice_solve_20260723",
+      "name": "DDM MS1 minimum-description lattice solve",
+      "phase": 1.0,
+      "level": 1,
+      "gates": {
+        "impl_complete": {
+          "status": false,
+          "evidence": ""
+        },
+        "real_archive_empirical": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cuda": {
+          "status": false,
+          "evidence": ""
+        },
+        "contest_cpu": {
+          "status": false,
+          "evidence": ""
+        },
+        "strict_preflight": {
+          "status": false,
+          "evidence": ""
+        },
+        "three_clean_review": {
+          "status": true,
+          "evidence": ".omx/research/ddm_ms1_min_description_lattice_solve_three_clean_review_20260724.json"
+        },
+        "memory_entry": {
+          "status": false,
+          "evidence": ""
+        },
+        "deploy_runbook": {
+          "status": false,
+          "evidence": ""
+        }
+      },
+      "notes": "Delegated isolated-worktree successor to #547/#549/#602; research-only local frozen-scorer advisory; pointer unchanged; MAIN landing review required.",
+      "research_only": true
     }
   ],
   "level_rules": {
@@ -85198,5 +85240,5 @@
     "3": "ALL 8 gates satisfied \u2014 FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED."
   },
   "schema_version": 1,
-  "updated_at": "2026-07-23T23:14:07Z"
+  "updated_at": "2026-07-24T02:03:08Z"
 }
```

