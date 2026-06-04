# Codex Findings - HiNeRV Fresh-Eyes Blocker Burndown

Generated: 2026-06-04T05:43:12Z

## Authority

False-authority implementation review only. No score claim, promotion claim,
rank authority, or exact-eval dispatch authority.

## Finding

The HiNeRV stack had a narrow authority-labeling gap around the feature-grid
and ConvNeXt path. The local PyTorch/MLX receiver path exposes hierarchical
feature grids, ConvNeXt blocks, and archive roundtrip coverage, while
`official_grid.py` separately binds the official temporal-only
`GridTrilinear3D` primitive. Those are useful, but they are not the same as
official source-forward parity for the full HiNeRV core renderer.

The existing official-source audit already failed closed on
`hinerv_official_forward_parity_missing`; the implementation gap was that the
source-parity contract could still describe the feature-grid/ConvNeXt row as
official-bound without a dedicated machine-readable local-analogue risk.

## Landing

- Added `hi_nerv_feature_grid_convnext_authority_status()` with explicit
  false-authority fields and blockers.
- Split the source-parity contract so the receiver-visible local
  grid/ConvNeXt binding remains long-training-safe, while
  `hi_nerv_official_feature_grid_source_forward_parity` is a nonblocking source
  gap until official core-forward replay closes.
- Added an analogue-risk row for
  `hi_nerv_local_feature_grid_convnext_analogue`.
- Added tests that preserve launchability while preventing official
  source-forward authority laundering.

## Remaining Blockers

- `hinerv_official_feature_grid_source_forward_parity_missing`
- `hinerv_core_hierarchical_renderer_source_forward_replay_missing`
- `hinerv_local_feature_grid_sampler_differs_from_official_grid_trilinear3d`
- `hinerv_official_forward_parity_missing`
- receiver-closed full-600 replay and contest exact-eval authority remain
  separate from this local source-parity patch.

## Highest-EV Next Patch

Build the official-core-forward replay bridge that loads a tiny official
HiNeRV state/input bundle, maps or rejects local receiver parameters explicitly,
and writes a falsification/proof artifact consumed by
`build_hinerv_official_source_parity_audit()`. That is the shortest path from
local analogue to either official parity proof or a precise architecture
renaming blocker.
