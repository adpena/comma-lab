# Codex Findings - PR101 OP-7 Exact-Eval Score Response

Timestamp: 2026-05-19T11:05:00Z
Agent: codex:gpt-5.5
Scope: ITEM_7 PR101 pose-axis raw-byte-delta candidate exact-eval closure.

## Executive Finding

The measured OP-7 `raw_byte_delta` candidate is an exact-eval regression on both contest axes at unchanged archive bytes. This retires only the measured same-length raw-delta configuration; it does not kill the master-gradient, per-pair gradient, operator-packet, or procedural/deterministic-byte-derivation families.

The candidate changed scorer-visible components, but in the wrong direction:

| axis | baseline score | candidate score | total delta | seg term delta | pose term delta | rate delta | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| contest-CPU Linux x86_64 | 0.1928480127024255 | 0.19454175105500707 | +0.0016937383525815752 | +0.001580999999999999 | +0.00011273835258157269 | 0.0 | SCORE_REGRESSION |
| contest-CUDA T4 | 0.22634945874409151 | 0.2277121199933223 | +0.00136266124923079 | +0.0014279999999999987 | -0.00006533875076921575 | 0.0 | SCORE_REGRESSION |

Interpretation: the CUDA axis shows a tiny pose-term improvement, but it is overwhelmed by SegNet regression. The CPU axis regresses both SegNet and PoseNet terms. Since archive bytes are identical, this is pure scorer-response evidence.

## Custody

Source archive:

- path: `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
- bytes: `178258`
- sha256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`

Candidate archive:

- path: `experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/archive.zip`
- bytes: `178258`
- sha256: `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`

Score-response matrix:

- path: `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.json`
- sha256 after exact-eval refresh: `1314803a083e047e936d571d272aca2d13fcdbe0a1ac02bc02747e424d2aaba2`
- markdown sha256 after exact-eval refresh: `0c1947d0987a6fd825e7ffa10a82947de5638d5d753fa9e3a8b16d609d37420a`
- generator commit before this finding: `6d80327e050563bbbfde6252478042a419d4114c`
- refreshed matrix now records `dispatch_attempted=true`, `contest_exact_eval_artifacts_present=true`, empty `score_response_blockers`, and empty `dispatch_blockers`. It still carries two authority blockers inherited from the source operator manifest: `anchor_score_axis_dominance_not_persisted` and `source_anchor_score_axis_dominance_available_not_true`.

Probe outputs:

- CPU: `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length/score_response/modal_contest_cpu_linux_x86_auto.score_response.json`
- CPU sha256: `1fd3bbfe2588850cf4a8ea04f0445ca4bba0991f3be2471e918cdef7f043d973`
- CPU markdown sha256: `3de9b28873ad59ccc148f625ae01f35f13c19208919f29307af6084d55471bd9`
- CUDA: `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length/score_response/modal_contest_cuda_t4_auto.score_response.json`
- CUDA sha256: `dc71f0103224f5e4a3c5c42ea56cf6bae7aed2d2fd9d95791c985a692e990796`
- CUDA markdown sha256: `946f7c41cc29edac8ade2633f04eb62c182a40ed2613523ab88b517cb9053836`

## Modal Calls And Result Paths

| role | axis | call id | result path | runtime tree sha256 |
|---|---|---|---|---|
| baseline | contest-CPU | `fc-01KRZXK5EVDPSTQ6BMW0DQCF3S` | `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length_baseline/modal_contest_cpu_linux_x86_auto/contest_auth_eval.json` | `2660b5b21ba1ac5aac10e65bedb65230edf30c4e8314fc52f81e6039d124e5f6` |
| candidate | contest-CPU | `fc-01KRZXK55259196DKR7MKS9C09` | `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length_candidate/modal_contest_cpu_linux_x86_auto/contest_auth_eval.json` | `2660b5b21ba1ac5aac10e65bedb65230edf30c4e8314fc52f81e6039d124e5f6` |
| baseline | contest-CUDA | `fc-01KRZXK6X5A5RJ0TTCKY08J90S` | `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length_baseline/modal_contest_cuda_t4_auto/contest_auth_eval.json` | `81efad542b4e87b0d36f3990b022a57f542acd614aeca8cfd59ea7c2a0ec19f1` |
| candidate | contest-CUDA | `fc-01KRZXK6D22XG4D1NBMFZH8WAR` | `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/pr101-op7-rank1-raw-byte-delta-same-length_candidate/modal_contest_cuda_t4_auto/contest_auth_eval.json` | `81efad542b4e87b0d36f3990b022a57f542acd614aeca8cfd59ea7c2a0ec19f1` |

All four lane claims have terminal recovery rows in `.omx/state/active_lane_dispatch_claims.md`.

The four Modal call IDs were also backfilled into `.omx/state/modal_call_id_ledger.jsonl` with dispatched + harvested rows via `tac.deploy.modal.call_id_ledger`. This closes the orphan-call-id custody gap for:

- `fc-01KRZXK5EVDPSTQ6BMW0DQCF3S`
- `fc-01KRZXK55259196DKR7MKS9C09`
- `fc-01KRZXK6X5A5RJ0TTCKY08J90S`
- `fc-01KRZXK6D22XG4D1NBMFZH8WAR`

Cathedral autopilot evidence was appended to `reports/cathedral_autopilot_evidence.jsonl` as `cathedral_autopilot_score_response_evidence_v1` with `evidence_semantics=paired_exact_score_response_regression_no_family_kill`.

## Normalizer Bug Extincted

The initial CPU score-response probe blocked on `inflate_device_not_cpu` because Modal CPU exact eval records `inflate_device_policy=auto`. The auth-eval provenance is nevertheless precise CPU authority:

- `score_axis=contest_cpu`
- `evidence_grade=contest-CPU`
- `provenance.device=cpu`
- `provenance.platform_system=Linux`
- `provenance.platform_machine=x86_64`
- `provenance.cuda_available=false`
- `sys_argv` contains `--device cpu`

Patch:

- `src/tac/scorer_response_probe.py` now maps only this narrow Modal CPU auto-inflate case to `cpu(auto)` before strict custody validation.
- `src/tac/tests/test_scorer_response_probe.py` proves the accepted path and two non-authoritative variants (`cuda_available=true`, `--device cuda`) remain `auto`, preserving fail-closed behavior.

This is an authority precision fix, not a broad waiver.

## Authority Boundaries

These raw auth-eval JSONs have `score_claim_valid=true`, but remain non-promotional:

- CPU rows: `promotion_eligible=false`, `rank_or_kill_eligible=false`, `cpu_leaderboard_reproduction_eligible=true`.
- CUDA rows: `promotion_eligible=false`, `rank_or_kill_eligible=false`, `exact_cuda_eval_complete=true`.
- All rows retain blockers requiring adjudicated CUDA/CPU policy review and pre-submission/result-review surfaces before promotion or rank/kill language.

Therefore the only valid conclusion is:

`pr101-op7-rank1-raw-byte-delta-same-length` is a measured same-runtime, same-byte-count score-response regression on both exact axes.

Canonical probe-outcome ledger row:

- path: `.omx/state/probe_outcomes.jsonl`
- probe_id: `pr101_op7_raw_delta_exact_score_response_20260519T110500Z`
- verdict: `DEFER`
- blocker_status: `advisory`
- metric_value: `0.0016937383525815752`

## Reactivation Criteria

Reactivate the family only through a new measured configuration, for example:

- smaller trust-region byte deltas;
- per-pair or per-region perturbations instead of a global raw stream replacement;
- SegNet-boundary-preserving projection before archive packing;
- explicit null-space or Lagrangian dual filtering that minimizes the SegNet term;
- procedural/deterministic packet compiler variants that change representation rather than raw decoder bytes.

Do not route this specific raw-delta archive for promotion, rerank, or repeat exact eval unless a harness or custody error is found.
