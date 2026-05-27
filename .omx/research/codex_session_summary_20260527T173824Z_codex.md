# Codex session summary: MLX local unlock and target-profile preservation

Date: 2026-05-27T17:38:24Z
Agent: Codex
research_only: false

## Landed commits

- `510c15375` `Refactor MLX score-aware harness`
  - Split the MLX score-aware full harness into package modules for adapter, bundle, device gate, loss, portability, targets, and harness orchestration.
  - Kept the legacy facade importable while moving behavior behind tested module boundaries.
- `51e00effa` `Unblock MLX score-aware substrate trainers`
  - Routed ATW v2 and COIN++ substrate trainers through the canonical local MLX score-aware harness.
  - Added local MLX smoke artifacts for one-pair, one-epoch non-promotional runs.
- `fc8513050` `Wire target profiles through correction queues`
  - Connected target optimization profile metadata into correction queue surfaces.
- `5b42911f2` `Unblock method-specific local substrate training`
  - Added method-specific FAISS IVF-PQ codebook fitting and MLX-backed MDL/IBPS training unlock paths.
  - Added unlock tests for ATW, COIN++, and MDL/IBPS harness routing.
- `f19851d37` `Add FAISS codebook fitting unlock tests`
  - Added FAISS archive-emission and metadata regression coverage for the codebook-fitting path.
- `e832f8cfa` `Preserve target profiles through correction pipeline`
  - Propagated target-profile metadata through targeted correction work orders, response harvests, materialization requests, chain work orders, materializer handoff, operation-chain compiler metadata, and stage inputs.
  - Added CLI JSON handoff for `tools/build_frontier_targeted_component_correction_work_order.py`.
  - Hardened metadata extraction so canonical compact metadata, full target profiles, and the legacy local target-profile metadata schema all preserve false-authority signal instead of being silently dropped.

## Verification

- `ruff` passed for the MLX harness package and substrate unlock surfaces before their commits.
- `pytest src/tac/substrates/_shared/mlx_score_aware/tests src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py -q` passed with 59 passed and 1 skipped.
- ATW and COIN++ local MLX one-pair, one-epoch smoke runs completed with `promotable=False`.
- `pytest` for ATW and COIN++ substrate basics passed with 78 passed.
- MDL/IBPS local MLX one-pair, one-epoch smoke completed with `promotable=False`.
- FAISS IVF-PQ local codebook-fitting smoke completed, emitted a non-promotional archive, and reported `promotable=false`.
- `pytest` for ATW, COIN++, and MDL/IBPS unlock tests passed with 12 passed.
- `pytest` for FAISS basic plus MDL/IBPS tests passed with 70 passed.
- `pytest src/tac/substrates/faiss_ivf_pq_residual/tests/test_mlx_harness_unlock.py -q` passed with 3 passed.
- For `e832f8cfa`: `ruff`, `py_compile`, `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` passed with 55 passed, `tools/lane_maturity.py validate` passed with 1439 clean lanes, and `tools/review_gate_hook.py` passed.

## Current state left intentionally unstaged

- Dirty operational ledgers remain in `.omx/state/`, `experiments/results/_modal_harvest_summary.json`, and `reports/cathedral_autopilot_evidence.jsonl`.
- Generated local research artifact directories remain untracked under `.omx/research/`, including MLX smoke bundles and frontier feedback refresh outputs.
- These were not committed because they are run/custody state rather than the coherent code and test slice.

## Next integration pressure

- Convert MLX smoke bundles into deterministic replay manifests with exact argv, env, git SHA, diff SHA, artifact hashes, device metadata, and rerun-diff tooling.
- Promote the repair waterfill queue from metadata propagation into measured typed-response optimization over SegNet/PoseNet marginal curves and interaction terms.
- Wire every negative and positive local MLX/CPU result into continual-learning posterior updates without granting score, rank, kill, or dispatch authority.
- Close the encoder-side final attack path from target profile to chain planner, materializer, receiver proof, repair allocator, deterministic reproduction packet, and exact-eval handoff.
