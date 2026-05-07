# PR106x Low-Level Brotli Promotion Review

- target_label: `PR106x-lowlevel-brotli`
- verdict: `promotable_existing_exact_control`
- existing_exact_control_promotable: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- review_basis: `existing exact CUDA artifact only`

## Exact Artifact

- archive_sha256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- archive_bytes: `186080`
- payload_sha256: `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- byte_delta_vs_source: `-151`
- eval_artifact: `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`
- runtime_tree_sha256: `bb6baee66c61781f285fee5862ab499b8eb1fec93edeea046cb3019289638fd3`
- device: `cuda` / `Tesla T4`
- n_samples: `600`

## Adjudication

- promotion_eligible: `true`
- scientific_score_eligible: `true`
- contest_equivalent_hardware: `true`
- component_gate_triggered: `false`
- score_delta_vs_baseline: `-0.00010050000000000336`

## Blockers

- none

## Warnings

- `exact_eval_pact_commit_unavailable`: exact eval provenance could not record pact_commit; runtime tree and file hashes remain recorded
- `public_preflight_runtime_tree_differs_from_exact_eval`: local public preflight runtime hash differs from exact eval runtime hash; compare future runs by runtime tree

## Checks

| check | status | requirement |
|---|---:|---|
| `entropy_ranker_selected_existing_exact_control_review` | `pass` | ranker must route to review, not dispatch or score claim |
| `ranker_exact_control_ready_for_review` | `pass` | ranker exact-control row must be ready for promotion review |
| `scorecard_target_row_exact_cuda_frontier_eligible` | `pass` | scorecard row must be A++ exact CUDA lossless-control custody |
| `candidate_and_exact_eval_archive_identity_match` | `pass` | candidate, scorecard, and exact eval must identify the same archive bytes |
| `candidate_is_rate_positive_lossless_repack` | `pass` | candidate diff audit must be blocker-free and byte-positive |
| `brotli_raw_equivalence_closed` | `pass` | all Brotli section recodes must decompress to the same raw bytes |
| `public_replay_preflight_passed` | `pass` | public replay preflight must pass on the candidate archive |
| `exact_eval_is_cuda_t4_full_sample` | `pass` | exact eval must be CUDA on contest-equivalent T4 with 600 samples |
| `exact_eval_score_recomputed_from_components` | `pass` | review must use structured component recomputation, not rounded logs |
| `exact_eval_runtime_tree_recorded` | `pass` | exact eval runtime tree hash must be recorded and match the scorecard |
| `adjudication_marks_existing_exact_control_promotable` | `pass` | adjudication must explicitly permit promotion review use |
| `adjudication_component_gates_passed` | `pass` | PoseNet and SegNet gates must pass against the source control |
| `adjudication_score_delta_is_rate_only_improvement` | `pass` | adjudication must show non-regressing score delta versus source control |
| `scorecard_source_and_candidate_components_match` | `pass` | lossless control should preserve PoseNet and SegNet components in scorecard rows |
| `candidate_archive_file_matches_manifest` | `pass` | local archive copy must match reviewed SHA, bytes, and payload |
| `exact_eval_archive_file_matches_eval_json` | `pass` | local archive copy must match reviewed SHA, bytes, and payload |

Interpretation: this review is bounded to the existing exact CUDA
artifact. It is not a new score claim, not a GPU dispatch
authorization, and not evidence for future byte-different candidates.
