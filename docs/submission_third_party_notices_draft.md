<!-- DRAFT — placeholders must be filled from measured artifacts; see docs/submission_pr_draft_witness.md Final Gate. -->
<!-- This file becomes submissions/levelset_taskspace_witness/THIRD_PARTY_NOTICES.md at packaging. -->

# Third-party notices

This submission is a new decoder family with its own archive grammar and coders.
No code, weights, latents, or archive sections from any merged or open submission
to this repository are included. What follows is everything external that this
submission builds on, uses at train time, or depends on at inflate time.

## Contest upstream scaffold (commaai) — used unchanged

- `upstream/evaluate.py` — the official scorer; run unmodified.
- The frozen SegNet and PoseNet evaluator weights — used unmodified as the
  training objective (compress-time only; nothing from them ships in the
  archive).
- The `inflate.sh` / `archive.zip` submission convention — followed as
  specified; not modified.

## Concept-level credits — no code or weights reused

- **PR #95 — `hnerv_muon` (@AaronLeslie138)**: the pivot point. It showed what
  a tiny learned decoder can do on this problem — and studying it, including
  where its approach tops out, is what pushed us to develop our own model,
  objective, architecture, and curriculum from a different formulation. We
  reuse none of its code or architecture (this is not an HNeRV/NeRV variant).
  https://github.com/commaai/comma_video_compression_challenge/pull/95
- **PR #55 / #56 — (@Quantizr)**: the stored-target / FiLM-conditioning
  concept — conditioning the render on small stored per-pair values rather
  than reconstructing them from pixels. Our pose handling follows that
  concept; the implementation and the training (joint descent) are our own.
  https://github.com/commaai/comma_video_compression_challenge/pull/55
  https://github.com/commaai/comma_video_compression_challenge/pull/56

## Train-time-only priors (commaai public stack) — never shipped, never counted

- **openpilot** lane and camera geometry (public commaai code): used as a
  compress-time geometric prior. No openpilot code or data ships in the
  archive or the inflate runtime.
- **comma10k** class conventions (the 5-class segmentation labeling scheme):
  used to interpret SegNet's classes during training. Nothing derived from
  comma10k ships.

${IF_BYTE_SHAVING_ADOPTED_AT_PACKAGING: add a "## Byte-shaving methods adopted"
section crediting the specific method/code actually used, with its PR/author
and what exactly was taken. This is a test-gated decision made at byte-close;
OMIT this section entirely if nothing external was adopted.}

## Public libraries used by `inflate.py`

- numpy (BSD)
- brotli (MIT)
- torch (BSD-style)
- scipy (BSD)

The final dependency list is verified at packaging time against the actual
import set of the shipped `inflate.py` in a clean container
(`${AUDITED_DEP_LIST}` in the PR body is filled from that audit, not from this
draft).

## New code in this submission

The decoder (coordinate network), the archive grammar and its coders, the
pose-conditioning implementation, `compress.sh`, and `inflate.py` are new code
written for this submission (© 2026 adpena, licensed per the `LICENSE` file in
this directory).
