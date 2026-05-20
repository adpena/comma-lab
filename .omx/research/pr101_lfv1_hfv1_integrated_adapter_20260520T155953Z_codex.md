# PR101 LFV1/HFV1 Integrated Adapter Control

## Verdict

Local PR101/FEC6 integration control is passed. This is not a score claim and
not an exact-eval-ready submission packet.

The copied PR101 runtime in
`experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/`
now contains a self-contained HFV1 post-selector adapter. The live PR #110
submission directory was not edited.

## Runtime Patch

- Patched file:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.py`
- Runtime SHA-256:
  `626148315a8d3016d3cc2517c516c6376a204e1268049ee32db91a3f8a4e05c7`
- Insertion point:
  after `apply_pr101_selector_to_frames(...)`, before uint8 NHWC raw write.
- Absolute frame alignment:
  `frame_start = pair_start * 2`.
- HFV1 sidecar member:
  `foveation_params.bin`, loaded next to archive member `x`.
- Dependency rule:
  no import from `tac`; HFV1 parser and radial hyperbolic warp are vendored in
  the copied runtime.
- Identity rule:
  alpha-zero rows bypass `grid_sample` entirely, preserving byte-exact raw
  output.
- Nonidentity rule:
  warped frames are `clamp(0,255).round()` before uint8 conversion.

## Composed Archives

Manifest:

- Path:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/composed_archive_manifest.json`
- SHA-256:
  `eea6b40962f5da890672b7f500a02a76dc7a29b44a218b11c8411bb296f8211b`

Identity archive:

- Path:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_identity/archive.zip`
- Bytes:
  `202649`
- SHA-256:
  `6554f76fd506cda1066e9bd6672be840914b23e2aa626364a8158bc0a6444e6f`

Nonidentity archive:

- Path:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_nonidentity/archive.zip`
- Bytes:
  `202649`
- SHA-256:
  `172da5770da4f7de530741db58eb7b3e2bdba2352e47c8d6886e79065f8b382b`

Both archives are ZIP_STORED with two charged members:

- `foveation_params.bin`: `24016` bytes
- `x`: `178417` bytes,
  SHA-256 `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`

## Official Inflate Control

Control path:

```bash
env PACT_PYTHON_BIN=/Users/adpena/Projects/pact/.venv/bin/python \
  time bash experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.sh \
  <data_dir> <output_dir> \
  experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/file_list.txt
```

Runs:

- No-sidecar baseline: `36.74 real`, `saved 1200 frames`
- Identity sidecar: `36.85 real`, `saved 1200 frames`
- Nonidentity boundary sidecar: `36.88 real`, `saved 1200 frames`

Raw byte counts from `wc -c`:

- No sidecar: `3662409600`
- Identity: `3662409600`
- Nonidentity boundary: `3662409600`

Raw SHA-256 from `shasum -a 256`:

- No sidecar:
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- Identity:
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- Nonidentity boundary:
  `5753477449bca2a440dc6890d8e8c3aa042b6d4acdbb4d2c4045022d589e3585`

`cmp` checks:

- No-sidecar vs identity: exit `0`
- No-sidecar vs nonidentity boundary: exit `1`

Raw comparison report:

- Path:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/official_inflate_raw_comparison.json`
- SHA-256:
  `1f381f1c80d600a566f1e65986da3503f3dbecdd231949f73add3dc2bcc1f0a6`
- Passed:
  `true`
- Changed frame indices:
  `[32, 33]`
- Changed pair route:
  pair `16`, frame parities `0` and `1`

This satisfies the local control bar:

- no sidecar preserves PR101 behavior;
- full-1200-frame identity HFV1 is byte-exact to no sidecar;
- nonidentity HFV1 changes scorer-visible raw bytes;
- changed frames match the intended absolute frame route.

## Sensitivity Discipline

The control keeps SegNet and PoseNet routing separate:

- SegNet-sensitive mutations are frame-indexed.
- PoseNet-sensitive mutations are pair-indexed, then expanded to frames
  `2k` and `2k+1`.
- Boundary control frames `[32, 33]` map to pair `16` with both parities.

Compress-time optimization should use orthogonal or conflict-aware updates:

- PCGrad or explicit gradient projection for PoseNet-vs-SegNet conflicts.
- Stagewise freeze/unfreeze: renderer/selector baseline first, then geometry
  sidecar, then joint fine-tune only after no-op controls stay green.
- Pair waterfilling for PoseNet-sensitive pairs.
- Per-frame or region weighting for SegNet-sensitive frames.
- Master-gradient or atom-level byte selection only after the sidecar grammar is
  byte-closed.

Inflate-time scorer, teacher-model, training, or online optimization dependency
remains forbidden.

## Source-Fidelity Notes

This implementation uses the papers as compress-time design sources, not as
inflate-time dependencies:

- LA-Pose: pose prior and uncertainty source-map candidate
  (`https://arxiv.org/abs/2604.27448`).
- RAFT / SEA-RAFT: optical-flow and motion initialization candidate
  (`https://arxiv.org/abs/2003.12039`,
  `https://arxiv.org/abs/2405.14793`).
- Telescope-style foveation: radial/hyperbolic geometry family candidate
  (`https://princeton-computational-imaging.github.io/Telescope/`).

The contest runtime remains self-contained aside from PR101's existing
stdlib/torch/numpy/brotli dependency closure.

## PR #110 Activity Check

Checked with `gh api` on 2026-05-20:

- PR state: open, mergeable.
- Head SHA: `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- Last PR `updated_at`: `2026-05-20T14:46:27Z`.
- Human reviews: none.
- Review comments: none.
- Check-runs/status contexts: none.
- Visible comments: one initial `github-actions[bot]` submission acknowledgement
  from `2026-05-20T03:29:04Z`.
- Requested reviewer: `YassineYousfi`.

No new maintainer activity was visible.

## Remaining Blockers

- Exact CUDA auth eval missing.
- No tuned HFV1 sidecar search has been run against real component response.
- No score claim from these controls.
- No live PR #110 update from this experiment.

## Next Frontier-Moving Artifact

Build the first tuned HFV1 sidecar candidate for the PR101/FEC6 runtime:

1. Use frame/pair sensitivity anchors to select a small number of candidate
   pair routes.
2. Generate a compact HFV1 sidecar with identity everywhere except selected
   atoms.
3. Run the same official `inflate.sh` identity/nonidentity raw controls.
4. Run local component-response smoke if available without scorer-at-inflate.
5. Dispatch exact CUDA only after the candidate archive is byte-closed and lane
   claim discipline is satisfied.
