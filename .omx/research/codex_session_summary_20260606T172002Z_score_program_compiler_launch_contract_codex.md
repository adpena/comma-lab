# Codex Session Summary: Score-Program Compiler Launch Contract

UTC: 2026-06-06T17:20:02Z

## Scope

Codex consumed the operator synthesis that Quantizr, qrepro, PR95, and PR110 all
point at a known-receiver score-program compiler rather than ordinary video
compression. This landing converts that into a concrete launch-admission rule:
local MLX NeRV campaign rows must carry canonical receiver-surface trace and
archive parse-back selection contracts before they can be routed or admitted.

## Landed

- Live accepted MLX updates now emit canonical receiver-surface trace metrics
  from the producer path, including uint8 receiver movement, SegNet input
  movement, PoseNet output movement, fakequant/parse-back/inflate survival
  fields, and exact connected worst-region margin p50 deltas.
- `nerv_crux_trace` now consumes only canonical `receiver_surface_*` rows and
  rejects legacy aliases or pair-local smoke rows as receiver traces.
- Long-training planner rows now carry:
  - `nerv_receiver_surface_trace_contract.v1`
  - `nerv_archive_parseback_selection_contract.v1`
- The cathedral consumer preserves and validates those nested launch contracts,
  including PR95 axis trace measurement rows, instead of compacting them away.
- The admission bridge rejects selected rows missing either contract and adds
  the canonical `nerv_crux_trace_rows.json` path as a queue postcondition.

## Authority

This is `[planning/control]` and `[macOS-MLX research-signal]` infrastructure.
It is not a score claim, promotion claim, exact CPU replay proof, or exact
CUDA/T4 authority. It only makes the next local MLX launch fail closed unless
accepted updates are wired to the receiver surface and parse-back selection is
declared before launch.

## Verification

- `uv run pytest src/tac/tests/test_trace_nerv_crux.py src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py -q`
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_consumer.py src/tac/tests/test_nerv_long_training_campaign_admission.py -q`
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py -q`
- `uv run ruff check` on the touched receiver, consumer, admission, and planner surfaces
- `uv run python tools/lane_maturity.py validate`
- `git diff --check`

## Remaining Compiler Blockers

- Emit parse-back and inflate survival metrics from the actual archive/runtime
  producer, not only the launch contract.
- Build the ActionAtlas/value-per-byte compiler row that prices scorer actions
  by `100*d_seg + sqrt(10*d_pose) + 25*bytes/N`.
- Land exact CPU/CUDA replay after the receiver-closed archive proof exists.
