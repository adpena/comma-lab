# Frontier Monolithic Archive Layout Correction — 2026-05-08

## Finding

Local verification confirms the medal-band HNeRV frontier archives are physical
monoliths at the ZIP layer:

- PR101 local archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
  is `178,258` bytes, SHA-256
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`,
  with one stored member `x` of `178,158` bytes.
- PR106 local archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
  is `186,239` bytes, SHA-256
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`,
  with one stored member `0.bin` of `186,131` bytes.

The user's cited `0.bin`/`178,873` PR101 figure does not match this local
PR101 custody artifact; it may refer to a different public/repacked artifact.
The conclusion that matters still holds: these frontier archives are
single-member packets, not separate `masks.mkv` / `optimized_poses.bin` /
`renderer.bin` archives.

## Parser-Proven Logical Layout

PR101 fixed-offset parser:

- `decoder_blob`: offset `0`, len `162,164`;
- `latent_blob`: offset `162,164`, len `15,387`;
- `sidecar_blob`: offset `177,551`, len `607`.

PR106 FF parser:

- `ff_header`: offset `0`, len `4`;
- `decoder_packed_brotli`: offset `4`, len `170,278`;
- `latents_and_sidecar_brotli`: offset `170,282`, len `15,849`.

## Implication

Archive-member-level component budgets are invalid for PR101/PR106-style
frontier HNeRV archives. Any budget for masks, poses, renderer, foveation,
latent sidecars, or categorical labels must be backed by parser-proven
internal sections with offsets, lengths, SHA-256s, and decode roundtrip.

The stronger statement "there is no mask/pose information anywhere" is not
proved by a single ZIP member. The proved statement is narrower and actionable:
there is no **separate ZIP-member** mask/pose budget on these substrates.

## Hardening Landed

- `src/tac/frontier_archive_layout.py` emits physical ZIP custody plus
  parser-proven PR101/PR106 logical sections.
- `tools/pr106_archive_decomposition.py` was rewritten to avoid filename
  category heuristics and to report monolithic packet implications explicitly.
- `src/tac/codec_stack_planner.py` now marks the stack target as
  `single_member_monolithic_packet_with_internal_parser_proven_logical_sections`,
  with `member_level_component_budgets_valid=false`.
- `reports/frontier_monolithic_archive_layout_20260508.json` contains the
  current local PR101/PR106 layout manifest.

Evidence grade: `empirical_archive_layout_cpu_no_score`. Score claim: false.
