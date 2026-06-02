# Codex Findings - HiNeRV measured archive-size ladder and allocator binding

date_utc: 2026-06-02T12:37:20Z
agent: codex
cwd: /Users/adpena/Projects/pact
lane_id: lane_codex_nerv_impl_sweep_selector_v4_guard_20260602
authority: false_local_rate_evidence_only
score_claim: false
promotion_eligible: false
exact_cpu_cuda_eval_executed: false

## What landed

HiNeRV model-size now has a measured archive ZIP ladder, not only a projected payload ladder. The new `hinerv_archive_size_ladder` analysis surface exports actual receiver-shaped `archive.zip` files for the local HiNeRV size configs, records bytes/SHA/paths, and computes marginal byte-price gates.

The ladder explicitly preserves the operator reminder: measured bytes are not enough. Each row and the top-level report require adaptive quantization, ablation, waterfilling, inverse-steg saliency, packed zeros, and entropy coding before model-size selection is admissible.

The NeRV control inventory can now ingest the measured archive-size ladder via `--hinerv-archive-size-ladder-json`, so this empirical rate result becomes reusable system intelligence instead of a standalone note.

## Durable artifacts

- Measured archive-size ladder JSON: `.omx/research/hinerv_archive_size_ladder_20260602T123711Z.json`
- Measured archive-size ladder Markdown: `.omx/research/hinerv_archive_size_ladder_20260602T123711Z.md`
- SSD archive directory: `/Volumes/VertigoDataTier/pact/hinerv_archive_size_ladder_20260602T123711Z`
- Inventory with measured ladder attached: `.omx/research/nerv_control_inventory_20260602T123720Z.json`
- Inventory Markdown: `.omx/research/nerv_control_inventory_20260602T123720Z.md`

## Measured byte ladder

Decoder codec: `int8_mixed`.
Receiver proof: not executed for this rate-only ladder.
Exact CPU/CUDA replay: not executed.

- `hi_nerv_local_tiny`: 135,056 B, rate score 0.08992824677286798, archive SHA prefix `696d95950801`.
- `hi_nerv_local_small`: 247,785 B, rate score 0.16498986069937724, archive SHA prefix `4e6c282be999`.
- `hi_nerv_local_base`: 397,988 B, rate score 0.26500387303518674, archive SHA prefix `cc6d00df7385`.
- `hi_nerv_local_wide`: 812,302 B, rate score 0.540878559339046, archive SHA prefix `88e4113daa35`.

Marginal gates:
- tiny -> small adds 112,729 B; required non-rate drop >= 0.07506161392650926.
- small -> base adds 150,203 B; required non-rate drop >= 0.1000140123358095.
- base -> wide adds 414,314 B; required non-rate drop >= 0.2758746863038593.

## Required allocator bindings

- `adaptive_quantization_by_decoder_weight_group`
- `ablate_or_zero_groups_with_nonpositive_measured_value`
- `waterfill_group_bits_against_fixed_contest_byte_price`
- `inverse_steg_saliency_decoder_weight_binding`
- `packed_zero_and_entropy_coded_low_value_groups`

## Verification

- Focused pytest: `74 passed`.
- Ruff: clean on archive ladder, model-size ladder, source-parity, control inventory, HiNeRV, selector-v4, and compact runner surfaces.
- Py compile: clean on the same executable surfaces.

## Remaining blockers

- The archive-size ladder has no non-rate scorer replay attached.
- Receiver proof was intentionally not executed for the 600-pair rate ladder.
- Exact CPU/CUDA eval was not executed.
- HiNeRV still needs score-aware full-video trainer integration and decoder-weight saliency/VJP.
- The measured ladder must be combined with adaptive quantization, ablation, waterfilling, inverse-steg saliency, packed zeros, and entropy coding before any model-size selection can be treated as meaningful.
- Worktree/index remains dirty and shared; no commit was made by this memo.
