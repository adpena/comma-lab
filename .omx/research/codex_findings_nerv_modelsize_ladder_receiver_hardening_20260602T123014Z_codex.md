# Codex Findings - NeRV model-size ladder and HiNeRV receiver hardening

date_utc: 2026-06-02T12:30:14Z
agent: codex
cwd: /Users/adpena/Projects/pact
lane_id: lane_codex_nerv_impl_sweep_selector_v4_guard_20260602
authority: false_local_and_advisory_only
score_claim: false
promotion_eligible: false
exact_cpu_cuda_eval_executed: false

## What landed

The HiNeRV/SNeRV model-size control is now an executable false-authority ladder instead of a prose knob. `tac.analysis.nerv_modelsize_ladder` emits HiNeRV local size rows and SNeRV LF/depth/quant rows with section payload estimates for fp32/fp16/int8/int4/int2/configured modes, plus marginal gates that price each larger step using the fixed contest byte price.

The NeRV control inventory now embeds this ladder, so the implementation sweep, source/feature blockers, and size/rate gates travel together in one operator-facing report.

HiNeRV receiver hardening also converged: decoder-state validation is strict for non-latent weights while allowing the three latent tensors to be intentionally absent from the decoder state and loaded from the archive latent sections. The receiver-proof regression caught by the broadened tests is now closed.

## Durable artifacts

- Model-size ladder JSON: `.omx/research/nerv_modelsize_ladder_20260602T122618Z.json`
- Model-size ladder Markdown: `.omx/research/nerv_modelsize_ladder_20260602T122618Z.md`
- Updated control inventory JSON: `.omx/research/nerv_control_inventory_20260602T122618Z.json`
- Updated control inventory Markdown: `.omx/research/nerv_control_inventory_20260602T122618Z.md`
- Refreshed source-parity contract JSON: `.omx/research/nerv_source_parity_contract_20260602T123014Z_codex.json`
- Refreshed source-parity contract Markdown: `.omx/research/nerv_source_parity_contract_20260602T123014Z_codex.md`

## Key ladder numbers

Contest byte price: `6.658589531221714e-07` score per byte.

HiNeRV local ladder:
- `hi_nerv_local_tiny`: 94,764 params; int8 97,730 B; int4 50,348 B; int2 26,658 B.
- `hi_nerv_local_small`: 198,873 params; int8 205,091 B; int4 105,655 B; int2 55,937 B.
- `hi_nerv_local_base`: 340,802 params; int8 351,456 B; int4 181,055 B; int2 95,855 B.
- `hi_nerv_local_wide`: 738,120 params; int8 761,190 B; int4 392,130 B; int2 207,601 B.
- Example marginal gate: tiny -> small at int8 adds 107,361 B and requires non-rate score drop >= 0.07148728306614945.

SNeRV projected LF ladder:
- `snerv_l4_lf2_decoder_int4`: LF shape 24x32; configured payload 792,174 B.
- `snerv_l3_lf2_decoder_int4`: LF shape 48x64; configured payload 3,124,930 B.
- `snerv_l3_lf4_decoder_int4`: LF shape 48x64; configured payload 5,889,730 B.
- `snerv_l2_lf4_decoder_int4`: LF shape 96x128; configured payload 23,515,287 B.
- `snerv_l2_lf8_decoder_int8`: LF shape 96x128; configured payload 45,633,768 B.
- `snerv_l1_lf8_decoder_int8`: LF shape 192x256; configured payload 182,491,285 B.
- Example marginal gate: l4 lf2 -> l3 lf2 adds 2,332,756 B and requires non-rate score drop >= 1.553286468049464.

Interpretation: SNeRV LF storage remains rate-dangerous unless learned receiver-side generation, SR, symbolic residual grammar, or stronger wavelet-group saliency collapses the stored LF payload. HiNeRV int4/int2 ladders are in a plausible byte band, but only after score-aware fit proves non-rate improvement.

## Verification

- Focused pytest: `71 passed`.
- Environment-sensitive duplicate-campaign lock briefly failed while a live HiNeRV process was detected; after the process exited, the full suite passed.
- Ruff: clean on source-parity, model-size ladder, control inventory, HiNeRV architecture/inflate/renderer, selector-v4 renderer, and compact runner surfaces.
- Py compile: clean on the same executable surfaces.

## Remaining blockers

- Model-size ladder is payload-projection only; receiver-closed archive ZIP section measurement is still required.
- Ladder rows do not yet carry measured non-rate scorer replay, so the budget planner cannot select a promoted size.
- Source-parity contract still has 7 long-training blockers.
- SNeRV still lacks MLX-native train/export and official parity for DWT/MFU/HFR/SNeRV-T.
- HiNeRV still lacks real score-aware full-video trainer integration and bitstream-q/pruning/quantization parity.
- No paired contest CPU/CUDA exact replay was executed.
- Worktree/index remains shared and dirty; no commit was made by this memo.
