# OTHER-PRIORITIES PARALLEL SWEEP — LANDED 2026-05-13

Lane: `lane_other_priorities_parallel_sweep_20260513`
Operator directive 2026-05-13: "proceed with all in the meantime while waiting on results".

## Summary

Sequential pipeline within one subagent, four priorities, ~145 min total, $0 GPU spend. Sister to GPU surface `lane_5_parallel_substrate_dispatch_orchestrator_20260513` and adversarial review `lane_adversarial_review_5_parallel_dispatch_20260513`. No file-conflict.

## Priorities (in commit order)

| # | Topic | Commit | LOC added | Tests |
|---|---|---|---|---|
| 1 | Muon optimizer port + train_substrate wire-up | `b2b1e3a5` | ~200 | 5/5 |
| 2 | 4 STRICT preflight failures in pretrained_driving_prior | `ea911f06` | 7 | 149/149 |
| 3 | Prediction-vs-empirical delta logger + tests | `666ac00e` | ~995 | 17/17 |
| 4 | Dykstra Pareto alternating-projections rerun | `8d91dadc` | ~408 | (live-runner) |

## Priority 1 — Muon port + wire-up

`src/tac/optimization/muon.py` already existed as a complete port (pre-landed by an earlier subagent). This priority closed the loop:

- `src/tac/tests/test_muon_optimizer.py` NEW with 5 tests covering NS5 shape/dtype/finiteness, 1-D-input rejection, partition routing semantics (stem/rgb/biases → AdamW; hidden weights → Muon), 2-D parameter update + momentum buffer, closure callback. 5/5 pass. (My initial 24-test file was replaced by a pre-commit hook with this canonical shorter version; the canonical set still covers the contract.)
- `src/tac/optimization/__init__.py`: lazy-exports `MuonOptimizer`, `partition_params_for_muon`, `zeropower_via_newtonschulz5`.
- `experiments/train_substrate_time_traveler_l5_autonomy.py`: `--optimizer` choices extended to `{adamw, iglt, muon, muon+iglt}`; `--muon-lr` / `--muon-weight-decay` flags added; `_CompositeOptimizer` wrapper preserves the single-optimizer train-loop contract.

Cross-refs: Keller Jordan memo (commit `d64b17cf`); PR95 hnerv_muon source under `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/optim.py`; Chen-Li-Liu arXiv:2506.15054 for the weight-decay rationale.

**Operator-routable decision** — should Muon become the default optimizer for the next time_traveler L5 training run? Recommendation: opt-in via `--optimizer muon` on the next dispatch and compare wall-clock + final val score against the AdamW baseline. The 4-channel + ND-orthogonal-update story aligns with the L5-autonomy substrate's hidden 2-D+ conv blocks; AdamW remains the default until the empirical anchor lands.

## Priority 2 — STRICT preflight backfill

Per cleanup subagent finding referenced in commit `6400e958` audit memo. Initial probe showed 3 driving_prior failures via Check #126 (`check_lane_pre_registered_before_work_starts`) + 1 via Catalog #180 (`check_substrate_trainers_use_no_grad_at_eval`):

1. `src/tac/analysis/driving_prior_readiness.py:610` — violation string `"lane_id_mismatch"` returned from `validate_readiness_manifest`. NOT a lane reference.
2. `src/tac/substrates/pretrained_driving_prior/codebook.py:173` — codebook entry name `"lane_curvature_pca"`. NOT a lane reference.
3. `src/tac/substrates/pretrained_driving_prior/codebook.py:334` — same codebook entry.
4. `experiments/train_substrate_pretrained_driving_prior.py` — scaffold trainer (Phase 2 not yet implemented) lacking `torch.no_grad`.

Fixes:
- `src/tac/preflight.py::_LANE_ID_REFERENCE_BLOCKLIST`: added `"lane_id_mismatch"` and `"lane_curvature_pca"`. Both are exact instances of the blocklist's documented purpose: "common Python identifiers / method names that happen to share the `lane_` prefix".
- `experiments/train_substrate_pretrained_driving_prior.py`: added file-level `# NO_GRAD_WAIVED:scaffold-only-trainer-Phase-2-not-yet-implemented-no-eval-scorer-forwards-on-this-path` marker. Removed when Phase 2 lands.

Verification: `check_lane_pre_registered_before_work_starts` → 0; `check_substrate_trainers_use_no_grad_at_eval` → 0. 149/149 Catalog #126 tests still pass.

## Priority 3 — Delta logger

`tools/log_prediction_vs_empirical_delta.py` NEW (~450 LOC) consumes `.omx/state/predicted_anchors_solver_stack_wire_in_20260513.jsonl` (7 predictions) and `.omx/state/continual_learning_posterior.json` (25 accepted empirical anchors). Matches via bidirectional substring + token-overlap heuristic with short-distinctive-token fallback (`a1`, `l5`) and generic-token filtering (`prior`, `model`, `world`, `host`, `substrate`, `composition`, version tags). Computes Δ = empirical − predicted_midpoint; verdicts `within_band` / `over_predicted` / `under_predicted`; emits `deltas.json` + `deltas.csv` + `calibration_report.json` (+ optional matplotlib plot).

Live run: `matched=1 unmatched=6 predictions=7 empiricals=25`. One match: `lane_a1_plus_wavelet_residual_retarget_20260513` (predicted band `[0.187, 0.194]`) vs hnerv_wavelet empirical `0.2066` → `over_predicted`, Δ = +0.0161, recommendation `widen_band`. 6 substrates await first empirical anchor (time_traveler L5, sabor, s2sbs, a1+lapose, darts supernet, pr95 lora_dora).

**Operator-routable decision** — widen the wavelet+A1 prediction band to cover the over-prediction drift, OR investigate whether the matched empirical anchor (hnerv_wavelet at 0.2066) is the correct architecture-class peer (the substring match could be too lossy; consider tightening the matcher heuristic or adding explicit `predicted_anchor_lane_id ↔ empirical_anchor_architecture_class` mapping rules).

17/17 dedicated tests pass.

## Priority 4 — Dykstra rerun

`tools/rerun_dykstra_pareto_solver_stack_wire_in.py` NEW (~340 LOC) consumes `.omx/state/pareto_constraints_solver_stack_wire_in_20260513.json` and computes the per-axis intersection of per-substrate (rate, seg, pose) upper bounds.

Live output (`reports/dykstra_solver_stack_wire_in_rerun_20260513/pareto_solver_output.json`):

| Axis | Tightest bound | Baseline value | NEWLY-ACTIVE? | Binders |
|---|---|---|---|---|
| `rate_bytes` | **4,096** | 186,822 | **YES** (45× tighter) | a1+wavelet, pr95 lora_dora |
| `seg_distortion` | 0.0007 | 0.00067 | no (≈baseline) | a1+lapose, a1+wavelet, pr95 lora_dora |
| `pose_distortion` | 3.4e-5 | 3.4e-5 | no (=baseline) | a1+lapose, a1+wavelet, pr95 lora_dora |

Feasibility verdict: 3 / 7 substrates feasible within the intersected polytope. The 4 infeasible substrates (time_traveler L5 110KB, sabor 90KB, s2sbs 80KB, a1+lapose 6144B) all exceed the 4096-byte rate ceiling forced by the smallest sidecars.

**Operator-routable decision** — the 4,096-byte rate ceiling reflects the SMALLEST per-substrate sidecar, not a hard score floor. If the operator's goal is stacking small sidecars with heavyweight substrates, the Pareto manifest's per-substrate `rate_bytes_max` fields should be reinterpreted as COMPOSITION SLOT budgets rather than standalone packet budgets. The 3 feasible substrates (a1+lapose, a1+wavelet, pr95 lora_dora) are natural first-dispatch targets — they fit the intersected polytope AND are pose-axis-binding (per CLAUDE.md "SegNet vs PoseNet — operating-point dependent" 2.71× pose-marginal at PR106 r2).

## Discipline checks

- All 4 commits via `tools/subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #117 + #157 + #174.
- Co-Authored-By trailer auto-appended by the serializer per Catalog #119.
- No `/tmp` paths in any persisted artifact (Catalog #113).
- No KILL verdicts (CLAUDE.md "KILL is LAST RESORT").
- All score claims tagged `[contest-CUDA]` / `[contest-CPU GHA Linux x86_64]` / `[prediction; planning_only]` per CLAUDE.md "Apples-to-apples evidence discipline".
- 6-hook wire-in (Catalog #125) declared per priority in commit messages.
- Lane pre-registered before work started per Catalog #126.

## File index

- `src/tac/optimization/muon.py` (pre-existing port)
- `src/tac/optimization/__init__.py` (Muon lazy exports added)
- `src/tac/tests/test_muon_optimizer.py` (NEW; 5 tests)
- `experiments/train_substrate_time_traveler_l5_autonomy.py` (Muon wire-up + `_CompositeOptimizer`)
- `src/tac/preflight.py` (2 blocklist additions)
- `experiments/train_substrate_pretrained_driving_prior.py` (NO_GRAD_WAIVED file-level marker)
- `tools/log_prediction_vs_empirical_delta.py` (NEW; ~450 LOC)
- `src/tac/tests/test_log_prediction_vs_empirical_delta.py` (NEW; 17 tests)
- `tools/rerun_dykstra_pareto_solver_stack_wire_in.py` (NEW; ~340 LOC)
- `reports/prediction_vs_empirical_delta_20260513/` (delta logger live output)
- `reports/dykstra_solver_stack_wire_in_rerun_20260513/` (Dykstra live output)

## Commits

| SHA | Subject |
|---|---|
| `b2b1e3a5` | muon: port + tests + train_substrate wire-up via --optimizer muon |
| `ea911f06` | preflight: backfill 4 STRICT failures in pretrained_driving_prior |
| `666ac00e` | delta-logger: build prediction-vs-empirical delta logger + 17 tests |
| `8d91dadc` | dykstra: rerun alternating-projections solver on solver-stack Pareto manifest |
