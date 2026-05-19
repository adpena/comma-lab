# Codex Findings: DP1 Master-Gradient Projector Blocker

Timestamp UTC: 2026-05-19T06:46:06Z
Agent: codex
Scope: OP_SYN_1 DP1 projector review after PR106/PR107 projector landings.

## Verdict

DP1 should remain fail-closed for master-gradient anchor emission.

The current DP1 archive has canonical section offsets, but it does not yet have
the deterministic tensor-byte grammar needed for byte-gradient authority:

```text
DP1 header
codebook_blob          score-affecting, custom Brotli section grammar
renderer_blob          Brotli(pickle(state_dict))  <-- blocker
residual_blob          score-affecting int8 stream, Brotli-compressed
meta_blob              metadata
```

The unsafe shortcut would be to spread renderer gradients uniformly across the
whole `renderer_blob`. That would be a cargo-cult projector: useful as rough
telemetry, but not enough to authorize a master-gradient anchor or downstream
byte mutation ranking.

## Required Fix

Before DP1 can become anchor-emitting, replace or supplement
`Brotli(pickle(state_dict))` with a deterministic tensor-span serializer:

- explicit tensor order;
- explicit dtype and scale fields;
- explicit raw tensor byte offsets before compression;
- compression mapping policy documented as compressed-region approximation or
  replaced by section-local recompression probes;
- codebook and residual sections either get real Jacobians or named
  zero-gradient v1 contracts.

## Live Fixture

The current local DP1 tiny advisory archive:

- archive: `experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/archive.zip`
- archive bytes: `25914`
- archive SHA-256: `e4918b420c7b40379e432a11beb5671430f96cbaf68fa7bae70423ae0af2fc0b`
- member `0.bin` bytes: `25814`
- member SHA-256: `b11bab015fa5c0c60c4587713413511fca18e194cb3c1c36f06db3afae4ee0e9`

Section map:

| Section | Offset | Length | Role |
|---|---:|---:|---|
| `dp1_header` | 0 | 28 | control_or_metadata |
| `codebook_blob` | 28 | 8751 | decoder_weight_stream |
| `renderer_blob` | 8779 | 16538 | decoder_weight_stream |
| `residual_blob` | 25317 | 22 | latent_stream |
| `meta_blob` | 25339 | 475 | control_or_metadata |

## Landing

Updated the DP1 projection contract to make the blocker explicit:

```text
required_projector = dp1_deterministic_tensor_span_serializer_projector
```

The detection-only reason now names the exact blocker:
`Brotli(pickle(state_dict))` has no stable tensor-byte span grammar.

## Authority

No score claim. No anchor. No archive mutation.

This is a fail-closed authority-hardening artifact that prevents a false DP1
master-gradient projector from being treated as production signal.
