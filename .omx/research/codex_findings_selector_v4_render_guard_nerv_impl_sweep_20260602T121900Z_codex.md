# Codex Findings - selector-v4 render guard and NeRV implementation sweep

date_utc: 2026-06-02T12:19:00Z
agent: codex
cwd: /Users/adpena/Projects/pact
lane_id: lane_codex_nerv_impl_sweep_selector_v4_guard_20260602
authority: false_local_and_advisory_only
score_claim: false
promotion_eligible: false
exact_cpu_cuda_eval_executed: false

## Summary

Selector-v4 was not dead as an MLX module/export bridge, but the default/undertrained renderer surface can be nearly flat and previously escaped into archive/profile paths. I added a render-quality gate in the selector-v4 MLX training/export path so degenerate outputs block archive profiling/export metadata instead of becoming fake evidence. The same runner now preserves the local false-authority report and blocker list.

The NeRV sweep was hardened from an informal checklist into a hashed, complete memo/source inventory. HiNeRV and SNeRV are classified as implementation-incomplete rather than method-negative. The evidence says the current bad local numbers are consistent with missing official architecture/config/bitstream surfaces and missing score-aware training, not with a mathematical retirement of the families.

## Primary source control points

- HNeRV official implementation documents `--modelsize`, `--quant_model_bit`, and `--quant_embed_bit` as real model/rate controls.
- HiNeRV official implementation documents a compressed bitstream evaluation path using `--bitstream` and `--bitstream-q`; the paper describes hierarchical positional encodings, frame/patch representation, and training/pruning/quantization as core codec machinery.
- SNeRV official implementation documents SNeRV and SNeRV-T command families with stride, block, `fc_dim`, and embedding controls; the paper describes DWT LF/HF decomposition, MFU, HFR, and temporal extension as core machinery.
- SR-NeRV is retained as a high-priority enhancer principle because scorer-side resizing creates a resolution-axis dead-zone; this is not yet wired as a promoted archive path.

## Local evidence landed

- Focused tests: `56 passed`.
- Ruff: clean for touched selector-v4 runner, selector-v4 renderer/tests, NeRV inventory, and runner tests.
- Py compile: clean for selector-v4 renderer, runner, inventory, and inventory tool.
- SSD fail-closed smoke:
  - report: `/Volumes/VertigoDataTier/pact/selector_v4_render_guard_smoke_20260602T121320Z/compact_renderer_mlx_spine_runner_report.json`
  - verdict: `RENDER_OUTPUT_DEGENERATE_BLOCK_ARCHIVE_PROFILE`
  - blocker: `pact_nerv_selector_v4_render_quality_gate_failed`
- SSD nonflat runnable smoke:
  - report: `/Volumes/VertigoDataTier/pact/selector_v4_render_guard_pass_smoke_20260602T121330Z/compact_renderer_mlx_spine_runner_report.json`
  - archive: `/Volumes/VertigoDataTier/pact/selector_v4_render_guard_pass_smoke_20260602T121330Z/pact_nerv_selector_v4_mlx_training/archive.zip`
  - archive_bytes: 24555
  - verdict: `RENDER_OUTPUT_NONDEGENERATE_LOCAL_ONLY`
  - remaining blockers: partial coverage, no full-video MLX scorer replay, no contest CPU/CUDA exact eval.
- NeRV inventory:
  - JSON: `.omx/research/nerv_control_inventory_20260602T121343Z.json`
  - Markdown: `.omx/research/nerv_control_inventory_20260602T121343Z.md`
  - controls: 18
  - binding gaps: 27
  - implementation sweep blockers: 36
  - HiNeRV memos: 111 complete hashed rows
  - SNeRV memos: 92 complete hashed rows

## Verdicts

Selector-v4:
- The constant scorer input/cache symptom is now a hard blocker, not a silent advisory score source.
- A short nonflat SSD-backed executable path exists and archives, but it is still false-authority and noncompetitive until full-video scorer replay and exact contest-axis evidence exist.

HiNeRV:
- Not fully wired. Missing official parity surfaces include hierarchical feature grid, patch/frame equivalence, 3D upsampling, real bitstream-q receiver roundtrip, pruning/quantization stack, and score-aware trainer integration.
- Next implementation target is the real trainer: decoder-weight saliency/waterfilling plus PR95-style C1a/sigma/Muon/QAT, then measured quantized byte sections and exact replay.

SNeRV:
- Not fully wired. Missing official parity surfaces include DWT/LF/HF semantics, MFU/HFR/TUB or SNeRV-T parity, measured `fc_dim`/model-size ladder, receiver-closed quant payload replay, MLX-native train/export, and score-aware full-video loop.
- Next implementation target is representation-before-coding: learned/scorer-preserving LF/HF generation, SR low-res carrier, or score-aware decoder fit, with wavelet-group saliency binding.

## Blockers preserved

- No competitive byte-closed SNeRV archive.
- No trained byte-closed HiNeRV archive.
- No paired contest CPU/CUDA replay for these local lanes.
- No MLX-native SNeRV train/export parity.
- No first-class HiNeRV score-aware full-video trainer.
- Selector-v4 is runnable and guarded, but current proof is still a tiny false-authority smoke.
- Worktree remains dirty and shared; no staging or commit was performed.
