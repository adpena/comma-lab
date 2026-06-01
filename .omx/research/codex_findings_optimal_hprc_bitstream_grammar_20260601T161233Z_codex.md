# Optimal HPRC Bitstream Grammar Landing

**UTC:** 2026-06-01T16:12:33Z
**Agent:** Codex
**Authority:** false until receiver proof plus exact CPU/CUDA auth eval

## Finding

The optimal grammar is two-layered:

1. **Container/tensor entropy grammar:** use a single stored `0.bin` member,
   hardcoded or compact-varint section boundaries, homogeneous tensor streams,
   per-tensor byte-map/storage/coder selection, and entropy-gap diagnostics.
   This is near-saturated for PR101/fec6-style HNeRV payloads, where prior
   evidence says PR103 arithmetic saved only a few hundred bytes over split
   Brotli.
2. **Scorer decision-boundary grammar:** spend bytes by full-video P18/P19
   value, not visual detail. SegNet argmax-margin/boundary sensitivity and
   PoseNet inverse-variance Mahalanobis sensitivity define the atom saliency;
   reverse water-filling admits pixels/regions/subbands/pairs only when
   scorer value per byte beats the contest rate price.

This means grammar work is not expected to move a saturated HNeRV packet by
itself, but it is critical infrastructure for unsaturated substrates such as
Z8-style coefficient blobs and for every future compact-base/residual section.

## Implemented

- `src/tac/substrates/hprc/archive.py`
  - Added real bidirectional `hprc_g1_compact_bitmask_varint` packet grammar.
  - G1 keeps V0 section semantics but removes charged fixed-table hash/CRC
    overhead; section integrity stays with ZIP/proof manifests.
  - Added grammar byte profiling for V0 versus G1.
- `src/tac/substrates/hprc/bitstream_grammar.py`
  - Added contract-first optimal grammar planner for HPRC spine acquisition.
  - Emits entropy saturation diagnostics and decision-boundary allocation
    grammar for joint P18/P19/rate routing.
  - Keeps SegNet frame saliency and PoseNet pair saliency as separate channels
    until frame/pair incidence projection, then combines them for waterfill.
  - Routes saturated sections away from format churn and unsaturated sections
    toward the smallest bit-exact materializer task.
- `src/tac/substrates/hprc/spine_acquisition.py`
  - Acquisition reports now include `optimal_bitstream_grammar`.
- `tools/run_compact_renderer_mlx_spine_runner.py`
  - PR95 Stage-8 reports now reuse embedded package receiver proofs instead
    of requiring duplicate proof execution.
- `src/tac/substrates/hprc/spine_bounded_runner.py`
  - Receiver proof indexing now accepts archive/report alias fields.

## Evidence

- Focused tests: `18 passed`.
- Lint/diff hygiene: `ruff` passed and `git diff --check` passed.
- Live Stage-8 bridge adaptation on SSD:
  - report:
    `/Volumes/VertigoDataTier/pact/compact_pr95_stage8_existing_report_adapt_codex_v2/compact_renderer_mlx_spine_runner_report.json`
  - archive bytes: `178417`
  - declared pairs: `600`
  - receiver proof observed: `true`
  - receiver proof passed: `true`
  - optimal grammar saliency contract:
    `hprc_joint_p18_p19_saliency_contract.v1`
  - no raw/video artifacts left under the new SSD report dir.

## Remaining Blockers

- The Stage-8 candidate is a zero-epoch source-seed package, not trained
  source-faithfully yet.
- Full-video MLX scorer replay is not attached to the compact spine row.
- Section value-per-byte rows are missing for decoder, latents, receiver state,
  and RDO metadata.
- No contest CPU/CUDA exact auth eval has been executed.

## Next Action

Run a short nonzero PR95 Stage-8 continuation under this bridge, then attach
full-video section value replay. Only after the candidate is receiver-proven,
full-video-priced, and plausibly frontier-close should it produce exact
CPU/CUDA dispatch packets.
