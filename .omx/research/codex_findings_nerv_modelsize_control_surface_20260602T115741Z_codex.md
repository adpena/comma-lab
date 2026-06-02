# Codex Findings: NeRV Model-Size Control Surface

UTC: 2026-06-02T11:57:41Z

Verdict: before this landing, the compact HiNeRV/SNeRV queue was not fully
leveraging the upstream NeRV control surface. The runner exposed useful local
knobs, but it did not treat archive bytes as the input variable the way upstream
HNeRV/SNeRV `--modelsize` encourages. That is now represented as a planner
surface, not a score claim.

Sources checked:

- HNeRV upstream: https://github.com/haochen-rye/HNeRV
- SNeRV upstream: https://github.com/qwertja/SNeRV
- HiNeRV upstream: https://github.com/hmkx/HiNeRV

High-EV controls now encoded:

- HNeRV/SNeRV parameter-budget controls: `--modelsize`, width/reduction/stride/
  kernel/fc/embedding knobs, and model/embed quantization bits.
- SNeRV-specific controls: `--fc_dim`, `--emb_size`, `--quant_embed2_bit`,
  `snerv_t`, `snerv_t_2d`, and gradient clipping.
- HiNeRV controls: grid/channel/depth/kernel/scaling schedule, patch evaluation,
  prune ratio/weight, quant levels/noise/STE, and bitstream knobs.

Local integration now landed:

- `tac.analysis.nerv_modelsize_budget` converts target byte ceilings into
  byte-plausible local HiNeRV capacity candidates.
- The same module now emits SNeRV LF/HF receiver-grammar byte candidates over
  DWT levels, LF precision, step-map precision, and HF decoder codec.
- `tac.analysis.nerv_stack_synergy_audit` scans local HiNeRV/SNeRV surfaces,
  related `.omx/research` memos, upstream OSS controls, modelsize budgets, and
  partial/stub markers into a false-authority planner payload.
- The compact renderer MLX spine runner emits `nerv_oss_flag_audit` and
  `hinerv_modelsize_budget`, `snerv_modelsize_budget`, and
  `nerv_stack_synergy_audit` in plan reports and per-ceiling campaign rows.
- `--execute-family hi_nerv` and `--execute-family snerv` now accept
  `--modelsize-candidate-id auto|none|manual|off|<candidate_id>`. `auto`
  resolves from the full planner enumeration before launch and records the
  selected candidate in `modelsize_candidate_selection`.
- HiNeRV launch now consumes candidate `latent_dim`, `embed_dim`,
  `decoder_channel`, and `decoder_codec`; SNeRV launch now consumes candidate
  `levels`, `bits_per_coeff`, `step_map_bits_per_coeff`, and
  `decoder_payload_codec`, with waterfill step-map coding enabled for planner
  candidates.
- Candidate selection preserves the quantization ladder across portfolio/int8,
  int4, and int2 instead of collapsing to the cheapest nominal codec before
  scorer replay can arbitrate.
- SNeRV selection preserves structural over-ceiling blockers for missing DWT
  levels and LF precision points instead of suppressing them.
- Auto-resolution targets the tightest viable requested byte ceiling. In the
  current full600 nominal grammar, HiNeRV has frontier-ceiling candidate points
  under 178 KB; SNeRV does not fit 178 KB under its current LF/HF grammar and
  auto-selects the tightest viable higher ceiling when provided.

False-authority boundary:

- Nominal payload bytes are not archive authority.
- Selected candidates require trained byte-closed archive export, receiver
  proof, full-video MLX prefilter, local CPU replay, and exact CPU/CUDA only for
  true local winners.

Cross-variant priors to preserve:

- HNeRV/PR95-HNeRV: control arm, archive-byte discipline, curriculum/QAT/coder
  pressure.
- SR-NeRV: encode near scorer-observed resolution, super-resolve only for output
  compliance, protect pose geometry separately.
- RNeRV/E-NeRV: spatial-temporal disentanglement and capacity distribution.
- FFNeRV: flow-guided pose-channel enhancer when byte-priced.
- BoostNeRV: conditional decoder/temporal-affine enhancer, not standalone
  carrier.
- HiNeRV upstream: pruning/QAT/bitstream schedule and grid/channel/depth
  capacity atoms.

Source-faithfulness correction:

- The local HiNeRV path is a contest adapter with a hierarchical latent pyramid,
  MLX train/export, and HIV1 archive path. It is not source-faithful upstream
  HiNeRV until hierarchical feature grids, patch/frame unification, pruning, and
  bitstream-QAT controls are executable.
- The local SNeRV path is a contest adapter that stores LF, generates HF, and
  packages SNAR1 receiver bytes. It is not source-faithful official SNeRV until
  multi-layer scalable representation, MFU/HFR/TUB-style blocks, temporal
  SNeRV-T branches, and MLX train/export are executable.

Current actuator gap after this landing:

The launch consumes byte-budget candidates, but the curriculum is not yet fully
candidate-conditioned. Next: target bytes -> candidate capacity -> curriculum
schedule selected from candidate and scorer deltas -> staged scorer-aware MLX
training or typed refusal -> trained archive byte oracle -> receiver proof ->
MLX component prefilter -> local CPU replay -> exact CPU/CUDA only for true
local winners.

Validation anchors:

- `ruff check` on the new planner/audit/runner/test slice: pass.
- `pytest src/tac/tests/test_nerv_modelsize_budget.py
  src/tac/tests/test_nerv_stack_synergy_audit.py
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q`: 49 passed.
- Standalone audit emitted
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_nerv_stack_synergy_audit_20260602T122144Z.json`.
- Plan-mode runner emitted
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_nerv_modelsize_candidate_plan_20260602T122144Z/compact_renderer_mlx_spine_runner_report.json`.
