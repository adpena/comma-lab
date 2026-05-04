# Contest Submission Packet Checklist

This packet is metadata-only. It does not copy raw frames or archive payloads.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`.
- [x] SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- [x] Byte size: `276214`.

## Frontier Snapshot

- Score authority JSON: `contest_auth_eval.adjudicated.json`.
- Field-supported grade: `A++`.
- Recomputed score field: `0.31561703078448233`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.

## Auth Eval Fields

- [x] `contest_auth_eval.adjudicated.json` samples: `600`.
- [x] SegNet distance field present: `0.00061244`.
- [x] PoseNet distance field present: `0.00049637`.
- [x] Recomputed score field present: `0.31561703078448233`.
- [x] Device field: `cuda`.
- [x] GPU model field: `Tesla T4`.

## Optional Evidence

- [x] `component_trace.json`
- [x] `report.txt`
- [x] `eval_provenance.json`
- [x] `contest_auth_eval.adjudicated.json`
- [x] `adjudication_provenance.json`

## Non-Score Supporting Artifacts

- [x] `planner_ledgers` `experiments/results/c067_yousfi_fridrich_field_equations_20260502/top2_plan_guarded.json`: `planning_or_proxy_only`, score claim `false`.
- [x] `planner_ledgers` `experiments/results/c067_cmg3a_body200_atom_field_20260502/body200_field_plan.json`: `planning_or_proxy_only`, score claim `false`.
- [x] `visualizations` `reports/yousfi_fridrich_observability_20260502/target_gap.svg`: `visual_audit_only`, score claim `false`.
- [x] `visualizations` `reports/yousfi_fridrich_observability_20260502/score_breakdown.svg`: `visual_audit_only`, score claim `false`.
- [x] `next_action_tranches` `docs/runbooks/contest_faithful_submission_next_tranche_20260502.md`: `roadmap_only`, score claim `false`.

## Validation

- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=276214
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.31561709566216978 json=0.31561703078448233
- [x] `expected_archive_sha256`: expected=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
- [x] `expected_archive_size_bytes`: expected=276214
- [x] `expected_samples`: expected=600
- [x] `component_trace_cross_check`: component trace cross-check did not report a mismatch
- [x] `component_trace_contest_auth_eval_sha256`: component trace cross-check points at an accepted contest_auth_eval JSON
- [x] `component_trace_n_samples`: trace=600 contest=600
- [x] `component_trace_archive_bytes`: trace=276214 contest=276214
- [x] `eval_provenance_archive_sha256`: eval_provenance archive_sha256 matches archive.zip
- [x] `eval_provenance_archive_bytes`: eval_provenance archive_size_bytes matches archive.zip
- [x] `eval_provenance_device`: eval_provenance device matches contest_auth_eval provenance
- [x] `adjudicated_archive_sha256`: adjudicated JSON archive_sha256 matches archive.zip
- [x] `adjudicated_archive_bytes`: adjudicated JSON archive_size_bytes matches archive.zip
- [x] `adjudicated_n_samples`: adjudicated JSON n_samples matches contest_auth_eval.json

## Evidence Classification

- Field-supported grade: `A++`.
- Basis: `cuda_t4_full_sample_adjudicated_fields`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.
