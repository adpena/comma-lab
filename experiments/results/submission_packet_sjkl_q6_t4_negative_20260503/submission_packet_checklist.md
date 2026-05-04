# Contest Submission Packet Checklist

This packet is metadata-only. It does not copy raw frames or archive payloads.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z/archive.zip`.
- [x] SHA-256: `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`.
- [x] Byte size: `276556`.

## Frontier Snapshot

- Score authority JSON: `contest_auth_eval.adjudicated.json`.
- Field-supported grade: `A++`.
- Recomputed score field: `0.3158419419767293`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.

## Auth Eval Fields

- [x] `contest_auth_eval.adjudicated.json` samples: `600`.
- [x] SegNet distance field present: `0.00061244`.
- [x] PoseNet distance field present: `0.00049633`.
- [x] Recomputed score field present: `0.3158419419767293`.
- [x] Device field: `cuda`.
- [x] GPU model field: `Tesla T4`.

## Optional Evidence

- [x] `component_trace.json`
- [x] `report.txt`
- [x] `eval_provenance.json`
- [x] `auth_eval.log`
- [x] `contest_auth_eval.adjudicated.json`
- [x] `adjudication_provenance.json`

## Non-Score Supporting Artifacts

- [x] `planner_ledgers` `.omx/research/contest_faithful_swarm_execution_20260502_codex.md`: `planning_or_proxy_only`, score claim `false`.
- [ ] `visualizations`: none recorded.
- [x] `next_action_tranches` `docs/runbooks/contest_faithful_submission_next_tranche_20260502.md`: `roadmap_only`, score claim `false`.

## Validation

- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=276556
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.31584198061638458 json=0.31584194197672932
- [x] `expected_archive_sha256`: expected=a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d
- [x] `expected_archive_size_bytes`: expected=276556
- [x] `expected_samples`: expected=600
- [x] `component_trace_cross_check`: component trace cross-check did not report a mismatch
- [x] `component_trace_contest_auth_eval_sha256`: component trace cross-check points at an accepted contest_auth_eval JSON
- [x] `component_trace_n_samples`: trace=600 contest=600
- [x] `component_trace_archive_bytes`: trace=276556 contest=276556
- [x] `eval_provenance_archive_sha256`: eval_provenance archive_sha256 matches archive.zip
- [x] `eval_provenance_archive_bytes`: eval_provenance archive_size_bytes matches archive.zip
- [x] `eval_provenance_device`: eval_provenance device matches contest_auth_eval provenance
- [x] `adjudicated_archive_sha256`: adjudicated JSON archive_sha256 matches archive.zip
- [x] `adjudicated_archive_bytes`: adjudicated JSON archive_size_bytes matches archive.zip
- [x] `adjudicated_n_samples`: adjudicated JSON n_samples matches contest_auth_eval.json
- [x] `sjkl_loaded_payload_log_present`: archive contains charged sjkl.bin and auth_eval.log proves payload load
- [x] `sjkl_apply_log_present`: archive contains charged sjkl.bin and auth_eval.log proves runtime apply path
- [x] `sjkl_strict_contract_passed`: archive contains charged sjkl.bin and auth_eval.log proves SJKL_REQUIRE_APPLIED strict pass

## Evidence Classification

- Field-supported grade: `A++`.
- Basis: `cuda_t4_full_sample_adjudicated_fields`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.
