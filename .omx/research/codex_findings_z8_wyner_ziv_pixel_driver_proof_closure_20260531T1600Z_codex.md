# Codex Findings: Z8 Wyner-Ziv Pixel-Driver Proof Closure

Date: 2026-05-31
Author: Codex
Scope: Z8 hierarchical predictive coding runtime/archive bridge

## Finding

Two independent read-only subagent audits converged on the same contradiction:
`archive_candidate.py` listed `wyner_ziv_blob` as pixel-consumed, while
`byte_mutation_proof.py` and the runtime bridge still described WZ/Mamba as
custody-only. The current runtime was already decoding WZ top states and
projecting them into frame-1 top-LL, so the remaining gap was proof quality.

## Fix Landed In This Slice

The proof now uses a valid semantic WZ payload mutation instead of raw byte
flips alone:

- decode original WZ top state from the archive;
- encode a deterministic alternate top state through the same WZ coder;
- repack the Z8HPC1 archive with the mutated WZ payload;
- reconstruct through the same receiver projection path used by `inflate.py`;
- require frame 1 to change while frame 0 remains stable;
- preserve false-authority fields for score, promotion, rank/kill, and exact
  dispatch.

The coefficient-domain clamp was also removed from the WZ top-LL projection.
Top-LL wavelet coefficients are not pixel values and can be outside `[0, 1]`;
clipping there saturated the projection and erased state dependence. Final
pixel clipping remains at the RGB/raw write boundary.

## Current Truth

Z8 pixel-consuming sections are now:

- `wavelet_blob`: Mallat reconstruction payload.
- `wyner_ziv_blob`: decoded WZ/Mamba top-state projection into frame-1 top-LL.

Still custody-only:

- `decoder_blob`
- `indices_blob`
- `dreamer_state_blob`

Z8 remains a partial predictive-stack archive candidate, not a score authority.
The next honest runtime bridge work is decoder/index/Dreamer pixel consumption
or a hard blocker stating why those sections remain custody only.

## Partner-Agent Hints

- Do not cite old "WZ custody-only" wording unless referencing historical
  superseded state.
- Do not bring back coefficient-domain `[0, 1]` clipping; clip pixels only at
  final render/write boundaries.
- Keep `boundary_argmax_hinge` as the active SegNet loss default unless a real
  scorer-response surface beats it.
- Treat `.omx/research/z8_hier_pc_full_stack_longrun_20260531/` as MLX training
  signal, not archive authority.
- Keep Compound C out of the validated predictive stack until provenance repair
  lands through byte-closed contest members.

