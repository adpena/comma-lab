# Codex Session Summary: MLX Full-600 Parent Contract Closure

utc: 2026-05-22T12:33:16Z
agent: codex
status: landed_locally_pending_commit
score_claim: false
score_claim_valid: false
promotion_eligible: false
promotable: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Completed

- Recovered and materialized decoder-q auth tensor cache from Modal CPU tensor
  export with zero-residual cache/auth identity.
- Ran decoder-q full-600 local MLX parent response on CPU singleton batches.
- Ran FEC6 full-600 auth-cache MLX parent response on CPU singleton batches.
- Rebuilt full-sample score calibration from FEC6 and decoder-q strict
  contest-CPU auth payloads.
- Built strict full-600 parent production contracts for FEC6 and decoder-q.
- Built the two-contract full-600 bundle and parent plan:
  - Bundle:
    `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_parent_contract_bundle_full600_fec6_decoderq.json`
  - Parent plan:
    `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/parent_production_contract_plan_full600_fec6_decoderq.json`
  - Status: `strict_pass`
  - MLX rows: `1200`
  - Covered parent groups: `2/2`
- Exported the recovered grayscale-LUT A100 `best.pt` into a local archive ZIP
  and marked the recovery lane to L2 through `tools/lane_maturity.py`.

## Findings

- Full-parent FEC6 remains better than decoder-q on both strict CPU auth eval
  and calibrated local MLX:
  - FEC6 MLX full parent: `0.1920527920355189 [macOS-MLX research-signal]`
  - Decoder-q MLX full parent: `0.1924459939299716 [macOS-MLX research-signal]`
  - MLX gap, decoder-q minus FEC6: `0.00039320189445271603`
  - CPU gap, decoder-q minus FEC6: `0.000393914325026834`
- MLX/CPU ordering matched with zero rank inversions. The calibrated minimum
  MLX gap for spend triage is `7.375772066442465e-06`.
- The decoder-q window surface still has 170 improved singleton windows versus
  FEC6. That is the useful local signal; full-parent aggregate promotion is not
  supported.
- FEC6 full-600 parity needed the same one-pixel SegNet argmax near-tie
  tolerance already used by decoder-q reference parity. Zero-pixel strict mode
  failed on two singleton windows with one pixel each; this is recorded and not
  hidden.

## Verification Run

- `tests/test_plan_mlx_parent_contract_closure.py` -> `6 passed`
- `ruff check tools/plan_mlx_parent_contract_closure.py tests/test_plan_mlx_parent_contract_closure.py` -> pass
- Full-600 FEC6 candidate parity with one-pixel argmax tolerance ->
  `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- Full-600 FEC6 profile stability -> `PASS_MLX_PROFILE_STABILITY`
- FEC6 full-600 contract -> `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- Decoder-q full-600 contract -> `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- Full-600 two-contract bundle -> `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
- Parent plan -> `strict_pass`

## Worktree Discipline

- Partner WIP in `tools/build_hfv1_sparse_sidecar_candidate.py` remains
  intentionally unstaged and untouched by the MLX commit plan.
- `.gitignore` now ignores `modal_*_cpu_tensors_*/` so recovered tensor-volume
  downloads do not churn git state.

## Next Critical Path

1. Commit and push the MLX authority/docs/state changes without absorbing
   partner HFV WIP.
2. Convert the 170 improved decoder-q singleton windows into byte-closed
   candidate edits, then filter through the full-600 MLX contract bundle.
3. Dispatch exact CPU/CUDA auth eval only for calibrated winners with active
   lane claims and preserved archive/runtime custody.
4. Apply the same auth-cache/parent-contract gate to PR110 and any new local
   substrate-training rows before they influence spend.
