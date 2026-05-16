# L5 v2 and PacketIR source-basis expansion - 2026-05-16

## Trigger

The operator asked for continued L5/L5-v2 focus, paper fidelity, production
OSS rigor, and useful latest domain research. A read-only source-fidelity
subagent recommended adding temporal-coherence INR, dense world-model,
foveation, semantic-layering, and compiler/parser references to the canonical
planning surface.

## Landing

Added planning-only research-basis rows for:

- `teconerv_2026` - temporal-coherence INR compression and residual temporal
  representation storage for L5/NeRV-family follow-up.
- `vjepa2_2025` and `vjepa2_1_dense_2026` - world-model and dense-video-feature
  teacher priors for DP1/L5-v2/pretrained-driving-prior work.
- `deepfovea_2019` - foveated neural reconstruction and sparse pixel stream
  priors for telescopic/foveation lanes.
- `dsslic_2019` - semantic segmentation as a layered compression base stream.
- `mlir_2020`, `kaitai_struct`, and `rfc7932_brotli` - PacketIR/compiler,
  archive grammar, and lossless repack source anchors.

All entries keep `score_claim=false` through the existing manifest surface and
carry charged-byte contracts plus hardening blockers. The L5-v2 family now
includes V-JEPA2/V-JEPA2.1/TeCoNeRV before the neural-codec stack; PacketIR has
its own source family for future compiler-pass documentation.

## Verification

Tests updated:

- `test_family_lookup_prefers_latest_family_specific_sources`
- `test_legacy_research_basis_aliases_resolve_to_canonical_ids`
- `test_all_registered_research_sources_satisfy_required_contract`

## Sources

- TeCoNeRV: `https://arxiv.org/abs/2602.16711`
- V-JEPA 2: `https://arxiv.org/abs/2506.09985`
- V-JEPA 2.1: `https://arxiv.org/abs/2603.14482`
- DeepFovea: `https://mmehas.github.io/publication/deepfovea`
- DSSLIC: `https://arxiv.org/abs/1806.03348`
- MLIR: `https://arxiv.org/abs/2002.11054`
- Kaitai Struct: `https://kaitai.io/`
- Brotli RFC 7932: `https://www.rfc-editor.org/rfc/rfc7932`

## Boundary

These sources are not empirical anchors and do not authorize dispatch. They
only improve source fidelity for L5-v2, DP1, foveation, semantic layering, and
PacketIR planning.
