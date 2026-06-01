# Compact PACT-NeRV-VQ Real-Scorer Smoke Findings

Generated: 2026-06-01T15:23:04Z  
Agent: codex  
Axis: `[macOS-MLX research-signal]` only

## Landed Implementation

Commit `81e7768be` wires `tools/run_compact_renderer_mlx_spine_runner.py`
to the existing real scorer-bound MLX harness:

- `--segnet-distillation-weight`
- `--pose-distillation-weight`
- `--segnet-distillation-objective`
- `--distillation-temperature`
- `--segnet-tau-boundary`
- `--segnet-hinge-margin`
- `--distillation-device`
- `--allow-segnet-only-research`

Positive SegNet/PoseNet weights now build real teacher caches and learnable
student heads through `RendererBundle`; SegNet-only training fails closed unless
explicitly tagged as research. The output remains false-authority until
receiver proof plus exact CPU/CUDA authority.

## Executed Evidence

Tiny real-teacher custody smoke:

- artifact: `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_scoreaware_2pair_smoke_codex_v1`
- archive bytes: `17131`
- archive sha256: `e1906dded4b1a4697c7114aea392bfd0e3b2a1706552a9f42823bcf4c24eaf47`
- receiver proof: passed
- both real SegNet and real PoseNet teachers: bound

64-pair score-aware smoke:

- artifact: `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_scoreaware_64pair_100ep_codex_v1`
- archive bytes: `70099`
- archive sha256: `67b8059750dd7a144c4af302b4581c2cf55cd47fcdabf249bce1da8f0cb2f890`
- training: `100` epochs, `20.13s`, real SegNet/PoseNet-bound loss
- receiver proof: passed
- inspection: `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_scoreaware_64pair_100ep_codex_v1/inspect_rendered_frames_v1/pact_nerv_vq_scoreaware_64pair_100ep_inspection.json`
- contact sheet: `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_scoreaware_64pair_100ep_codex_v1/inspect_rendered_frames_v1/pact_nerv_vq_scoreaware_64pair_100ep_contact_sheet.png`

## Verdict

The byte-custody adapter is real and reusable, but this PACT-NeRV-VQ compact
form is not yet a plausible primary carrier. Even with real SegNet/PoseNet
teacher terms, the 64-pair receiver output is still a dark mean-field image,
not a road-scene renderer. The failure is representation/training quality, not
archive custody.

Do not spend exact CPU/CUDA or long full-video budget on this exact VQ form as
a primary carrier without a new architecture/training change. Keep it as:

- a packet-spine adapter proving tiny archive grammar;
- a VQ/selector/codebook experimental branch;
- a residual-token testbed when value-per-byte is proven.

The next score-lowering carrier budget should favor PR95/HNeRV-scale,
RNeRV/SR-NeRV/BoostNeRV, or PVQ/RT-VQ-NeRV designs that can preserve road
semantics at PR95-like archive size.

