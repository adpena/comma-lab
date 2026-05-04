# C067/C088 Mask Topology Sub-0.314 Local Candidate - 2026-05-03 Codex

Scope: local-only build and triage for a contest-faithful candidate aimed at
compressing the dominant PR75/C088 mask stream while preserving decoded
geometry. No remote GPU job was dispatched and no `.omx/state` dispatch row was
created.

## Source Frontier

- Source archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip`
- Source archive bytes: `276386`
- Source archive SHA-256:
  `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`
- Exact T4 score source:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/contest_auth_eval.adjudicated.json`
- Exact T4 recomputed score: `0.3155226919767294`

This is the promoted C088/top40 PR75-action transfer parent. It preserves the
dominant PR75/C088 mask geometry and is the right parent for lossless mask
stream work.

## Local Build

Command:

```bash
.venv/bin/python experiments/build_pr75_lossless_repack_candidates.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip \
  --output-dir experiments/results/c067_mask_topology_sub314_c088_lossless_repack_20260503 \
  --force \
  --exhaustive-brotli-grid
```

Best local candidate:

- Candidate id: `c082_p6_delta_varint_actions_stream_resweep`
- Archive:
  `experiments/results/c067_mask_topology_sub314_c088_lossless_repack_20260503/c082_p6_delta_varint_actions_stream_resweep/archive.zip`
- Archive bytes: `276333`
- Archive SHA-256:
  `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
- Delta vs C088 source: `-53` bytes
- Formula-only rate delta vs source: `-0.00003529052451547508`
- Expected score if decoded-stream parity survives exact eval:
  `0.3154874014522139`
- Remaining unchanged-distortion byte gap to sub-`0.314`: about `2234` bytes

The build also emitted two smaller wins:

| Candidate | Bytes | Delta vs source | Payload format |
|---|---:|---:|---|
| `c082_p6_delta_varint_actions_stream_resweep` | `276333` | `-53` | `public_pr75_qzs3_qp1_segactions_p6_delta_varint` |
| `c082_p6_delta_varint_actions` | `276341` | `-45` | `public_pr75_qzs3_qp1_segactions_p6_delta_varint` |
| `c082_p3_stream_resweep` | `276372` | `-14` | `public_pr75_qzs3_qp1_segactions_p3` |

## Geometry And Compliance

The selected manifest records `decoded_stream_parity=true` and
`source_preservation.status=lossless_decoded_stream_preserving_repack`.

Decoded member parity:

- `masks.mkv`: decoded bytes `223385`, decoded SHA-256
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- `renderer.bin`: decoded bytes `59288`, decoded SHA-256
  `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`
- `seg_tile_actions.bin`: decoded bytes `160`, decoded SHA-256
  `8b34ef134e6551c83d170f07c8f6672870f09df5d38551ab24320f4a2f20a373`
- `optimized_poses.qp1`: decoded bytes `1140`, decoded SHA-256
  `1d2a6c31e836aa138bd09b3448db7e066f29f0cfcbf71b00e13357242655b583`

The candidate archive is a deterministic single-member stored ZIP with member
`p`; no sidecars are required. The mask stream itself improves from `219472`
Brotli bytes to `219465` Brotli bytes using params
`quality=11, mode=0, lgwin=19, lgblock=17`. The additional byte win comes from
lossless pose resweep and delta-varint action encoding.

## Exact Dispatch Readiness

Readiness status:

- Byte-closed archive exists: yes.
- Source-preserving decoded-stream parity: yes, as recorded by the local
  builder.
- Sidecar dependency: none.
- Score claim: no. This is empirical lossless byte-transform evidence until
  exact CUDA auth eval runs on the exact archive bytes.
- Remote dispatch performed: no.
- Dispatch claim requirement before any future remote exact eval:
  `tools/claim_lane_dispatch.py claim ...`.

Decision: technically exact-eval ready, but not a high-EV sub-`0.314` spend by
itself. At unchanged distortion the expected score is about `0.3154874`, still
roughly `2234` bytes from sub-`0.314`. It is useful as a safe local candidate or
as an additive packaging component for a larger geometry-preserving mask-stream
move, but it does not justify a standalone remote exact-eval dispatch under the
current sub-`0.314` objective.

## Negative/Blocked Alternatives

Existing exact and planning evidence still blocks the higher-byte mask-topology
replacements as standalone dispatches:

- CMG3/CMG3A and multimask replacements can save bytes but change decoded mask
  geometry and have exact-negative family evidence.
- Micro mask re-encodes save more bytes locally but have diagnostic collapse
  risk and are not source-preserving.
- Lane 12/NeRV readiness remains false for this objective without a passing
  geometry/closure packet.

Next high-EV direction: keep this lossless repack as the safe packaging base,
then search for a larger geometry-preserving mask-stream transform with at
least `~2234` additional bytes saved or a measured component improvement that
survives exact CUDA.
