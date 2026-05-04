# Geometry-Safe Mask Overlay Search - 2026-05-03 Worker

Scope: local-only search for geometry-preserving or geometry-bounded mask byte
cuts that could combine with the current PR75/QP1/P6 frontier. No remote GPU
job was dispatched and no dispatch claim row was created.

## Source

- Source archive:
  `experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/c067_pr75_actions_lag_eval_top67_p6/archive.zip`
- Source bytes: `276352`
- Source SHA-256:
  `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972`
- Exact T4 frontier score reference: `0.3154979650614253`
- Payload: single stored `p`, parsed as
  `public_pr75_qzs3_qp1_segactions_p6_delta_varint`.
- Decoded mask tensor: shape `[600, 384, 512]`, SHA-256
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`.

## Local Tooling

Added `experiments/geometry_safe_mask_overlay_search.py`, which preserves the
PR75 P6 renderer, QP1 pose, and tile-action streams while searching:

- lossless Brotli resweep of the existing AV1 OBU mask stream;
- bounded AV1 OBU mask re-encodes with decoded class-pixel disagreement
  metrics;
- exact CDO1 overlay economics for lossy bases.

Tests:

```bash
.venv/bin/python -m pytest src/tac/tests/test_geometry_safe_mask_overlay_search.py -q
```

Result: `4 passed in 0.10s`.

Search command:

```bash
.venv/bin/python experiments/geometry_safe_mask_overlay_search.py \
  --force \
  --overlay-probe-limit 2
```

Output directory:
`experiments/results/geometry_safe_mask_overlay_search_20260503_worker/`.

## Candidate Matrix

All rows are local empirical artifacts with `score_claim=false`.

| Candidate | Bytes | Delta vs source | SHA-256 | Decoded mask disagreement | Recommendation |
|---|---:|---:|---|---:|---|
| `lossless_mask_obu_brotli_resweep_p6` | `276345` | `-7` | `7d294d4c8923648ae0c56f918b0bd13576eac098c74d400c2becf311a415ee02` | `0` pixels | Exact-ready as a bundled packaging component only; standalone value too small. |
| `bounded_mask_reencode_crf52_preset13_g9999` | `263046` | `-13306` | `3ccd680aec1f8ea124ce96468c7d7dec313e9053a6bf0b498edc838fbb22ee0f` | `894813` pixels (`0.007585423787434896`) | Do not dispatch standalone; geometry changed. |
| `bounded_mask_reencode_crf55_preset13_g9999` | `204443` | `-71909` | `00887f03d32ef0187a4f43929d3f79575fd32cabd47a4c3666eaa9786bb2dcdf` | `959366` pixels (`0.008132646348741319`) | Do not dispatch standalone; geometry changed. |
| `bounded_mask_reencode_crf58_preset13_g9999` | `155782` | `-120570` | `33b559190bb302aa0993b4297d5915ec80bbaf06249c017e2a6682e83d308eed` | `1041607` pixels (`0.00882981194390191`) | Do not dispatch standalone; geometry changed. |
| `bounded_mask_reencode_crf60_preset13_g9999` | `133099` | `-143253` | `912b391a3199b6bc4358cd2b04bfbcf632e5255d454e70a5f0b22c3ba98f7f44` | `1093201` pixels (`0.009267179701063367`) | Do not dispatch standalone; geometry changed. |

Structured matrix:
`experiments/results/geometry_safe_mask_overlay_search_20260503_worker/candidate_matrix.json`.

## Overlay Economics

Exact CDO1 overlays were evaluated for the two lowest-disagreement lossy bases.
They restore the source mask tensor exactly, but the charged residual is far
larger than the byte savings:

- `crf52`: raw CDO1 `2725972` B, Brotli q11 `739820` B, zlib9 `1150284` B.
  This overwhelms the `13306` B archive saving.
- `crf55`: raw CDO1 `2777083` B, Brotli q11 `754049` B, zlib9 `1175998` B.
  This overwhelms the `71909` B archive saving.

Structured overlay record:
`experiments/results/geometry_safe_mask_overlay_search_20260503_worker/overlay_economics.json`.

## Smoke Checks

The lossless and lowest-disagreement bounded archives were unpack-smoked with
`submissions/robust_current/unpack_renderer_payload.py`.

- Lossless unpack summary:
  `experiments/results/geometry_safe_mask_overlay_search_20260503_worker/smoke_unpack_lossless/renderer_payload_unpack_summary.json`
- CRF52 unpack summary:
  `experiments/results/geometry_safe_mask_overlay_search_20260503_worker/smoke_unpack_crf52/renderer_payload_unpack_summary.json`

Both materialized `renderer.bin`, `masks.mkv`, `optimized_poses.qp1`, and
`seg_tile_actions.bin` from a single `p` member. Non-mask decoded member SHAs
match the source in all candidate manifests.

## Decision

There is no standalone geometry-safe >1KB mask/overlay candidate from this
finite local space.

- Promoteable exact-safe byte cut: only `-7` B via lossless mask Brotli
  resweep. Useful only as a bundled packaging component.
- Larger byte cuts exist, but they change 0.76 percent to 0.93 percent of the
  half-frame decoded mask tensor. Prior CMG/PMG/micro-mask negatives make this
  too collapse-prone for direct exact dispatch.
- Exact CDO1 overlays preserve geometry but cost hundreds of KB after
  compression, so they are negative evidence for residual-overlay rescue of
  these lossy bases.

Dispatch recommendation: do not remote-dispatch these bounded-mask archives
standalone. If an operator wants a cliff-mapping exact eval anyway, first claim
the lane with `tools/claim_lane_dispatch.py claim ...` and treat the result as
diagnostic negative/positive boundary evidence, not promotion evidence.
