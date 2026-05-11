# PR103 global-combo top25/beam512 saturation note

Generated: 2026-05-11T01:16:01Z

## Verdict

The widened PR103 arithmetic histogram global-combo probe is **planning-only
negative evidence** for this local search basin. It does not beat the current
clean-runtime PR103 byte target.

## Evidence

- Probe artifact:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_top25_beam512_plus_latent_hi_probe.json`
- Markdown review:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_top25_beam512_plus_latent_hi_probe.md`
- Search configuration: `top_per_stream=25`, `beam_width=512`,
  `combo_workers=8`, `stream_count=6`
- Exact recomputed states: `41979`
- Wall-clock: `77.58s`
- User CPU: `568.12s`
- Best estimated member delta: `-13B`
- Best objective components:
  - `merged_ac_delta=0`
  - `ac_histograms_brotli_delta=-12`
  - `latent_hi_histogram_brotli_delta=-1`
- Selected options:
  - `blocks.0.weight:candidate13`
  - `blocks.1.weight:candidate8`
  - `blocks.2.weight:source`
  - `blocks.3.weight:candidate1`
  - `stem.weight:source`
  - `latent_hi_bytes:candidate2`

## Classification

This is not a score claim and not a dispatch authorization. It is an exact
local byte-objective probe over PR103 q8 histogram sidebands plus latent-hi
histogram sideband, using exact merged-range re-encoding and exact Brotli
section recompression. The current clean-runtime packet remains better:

- Clean-runtime PR103 target: `-16B`, archive bytes `178207`, archive SHA-256
  `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- Widened top25/beam512 local best: `-13B`

## Implication

Do not materialize or CUDA-dispatch this top25/beam512 best candidate. The next
PR103 score-lowering work should move to a different grammar-aware axis:

1. source-shell-contract parity and compliance closure for the existing `-16B`
   packet;
2. additional decoded-stream grammar transforms beyond q8/latent-hi histogram
   sideband tweaks;
3. arithmetic/range coder replacement only after conformance vectors prove the
   decoder consumes the new bytes;
4. public-frontier deconstruction or T1/T6 training work when it has higher
   expected score movement than more histogram-grid widening.

## Tooling note

The process-backed worker path materially improved wall-clock viability:
`568.12s` user CPU completed in `77.58s` wall on local macOS CPU. Keep this
worker model for future exact local byte probes, but do not treat faster local
search as evidence of score movement unless it produces a byte-closed archive
and then exact auth-eval evidence on the correct CPU/CUDA axis.
