# Codex Findings - DQS1 Queue Bridge Canonicalization - 2026-05-23T10:07:26Z

## Scope

Codex adversarial review and repair of the DQS1 local-first queue after the MLX
decoder-q path failed on stale normalized-objective custody.

## Findings

- The ignored bridge plan
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.json`
  was stale. Its work units lacked `source_n_samples`,
  `full_video_denominator`, and normalized full-video objective fields, which
  made `plan_decoder_q_selective_runtime_packet.py` fail closed.
- Rebuilding the queue to call the current strict MLX OOF selector directly was
  not valid for this queue. A freshly regenerated normalized OOF gate blocks
  `mlx_decoder_q` spend triage because the current prediction model is not
  spend-triage usable. That is a correct authority block, not a queue execution
  block.
- The observed MLX singleton-selection artifact is still useful as local
  candidate-generation signal, but it must be explicitly normalized to the
  full-video objective and must remain false-authority.

## Repairs Landed

- `tac.optimization.decoder_q_selective_window_bridge` now canonicalizes legacy
  observed-selection rows by deriving `source_n_samples`,
  `full_video_denominator`, normalized full-video gain, projected full-video
  score delta, normalized break-even bytes, and normalized byte margin.
- The bridge preserves `legacy_selection_basis` and stamps
  `normalized_objective_backfilled=true` so downstream planners can audit that
  this came from legacy observed MLX signal rather than a current spend-triage
  authority gate.
- The DQS1 local-first queue now rebuilds per-candidate bridge plans from the
  observed MLX selection plus the decoder-q candidate manifest, rather than
  consuming the stale ignored bridge-plan JSON.
- The generated queue is bridge-first with no `select_mlx_windows` dependency,
  but every generated artifact stays `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Verification

- Focused tests: `30 passed`.
- Ruff check: passed.
- Queue validation: `valid=true`, `step_count=9`.
- Queue reconciliation: zero blocking orphans after retiring the stale
  `select_mlx_windows` definition from active consideration.
- Queue execution:
  - `build_bridge_plan`: passed in `0.513s`.
  - `plan_packet`: passed in `0.782s`.
  - `materialize`: passed in `0.522s`.
  - `locality_controls`: passed in `137.945s`.
  - `local_cpu_advisory`: passed in `483.515s`.
  - `build_mlx_local_advisory_cache`: passed in `5.442s`.
  - `local_cpu_contest_drift_eureka`: passed in `0.262s`.
  - `local_mlx_advisory_response`: passed in `34.806s` queue wall time
    (`30.731s` tool-reported scorer time).
  - `plan_mlx_delta_cache_retention`: passed in `5.704s`.
- Emitted bridge artifact top row now records raw MLX gain
  `0.0020326847010743165`, normalized full-video gain
  `0.000003387807835123861`, and normalized break-even margin
  `5.087876072310269` bytes.
- Full queue status after execution: `9` succeeded, `0` ready steps.
- Same-candidate local advisory score:
  - `[macOS-CPU advisory]`: `0.19204061709818365`.
  - `[macOS-MLX research-signal]`: `0.19243773514914575`.
  - MLX minus CPU: `0.0003971180509620975` worse/larger.
  - Queue wall-time speedup: about `13.9x` faster MLX response than local
    CPU advisory for this candidate.
- Drift/eureka gate verdict: `eureka_trigger=false`,
  `recommended_action=observe_only`, with auth frontier pointer
  `0.19202828295713675`.
- MLX cache retention plan marked one `mlx_scorer_input_cache` candidate,
  `2831169212` bytes, `certified_rebuildable=true`, with zero blockers.

## Authority Status

No score claim was made. This remains `[macOS-MLX research-signal]` and
`[locality-control no-score]` evidence only. The next authoritative step is
local CPU advisory scoring followed by the existing drift/eureka gate; contest
CPU/CUDA auth eval still requires the dispatch-claim path.
