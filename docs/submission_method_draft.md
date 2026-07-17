<!-- DRAFT — placeholders must be filled from measured artifacts; see docs/submission_pr_draft_witness.md Final Gate. -->
<!-- This file becomes submissions/levelset_taskspace_witness/method.md at packaging. -->

# How this submission works

## 1. The idea in 30 seconds

The decoder is a small coordinate network (an INR) trained **only against the
two frozen scorers** — there is no RGB reconstruction loss anywhere. The score
only cares about three things: SegNet's per-pixel class labels on frame 1,
PoseNet's 6 motion numbers per pair, and archive bytes. So the bytes go where
the score is: the SegNet class boundary and the pose-relevant image structure,
not picture quality. All the generic, deterministic code (the coordinate
transform, the network forward, the rasterization, the entropy decoder) lives
in `inflate.py`, which is free under the rules; `archive.zip` carries only the
learned payload: the network weights, the per-pair ego motion `ξ` (quantized
and entropy-coded, ~${DXI_BYTES} B), and a few small per-class seeds.

## 2. Training through the exact inflate chain

The optimizer never sees a proxy pipeline. Every training step renders at the
decode resolution, then goes through the same chain the shipped `inflate.py`
runs: bicubic upsample to 874×1164, uint8 quantization (straight-through in
the backward pass), and the real packing grid the archive stores values on.
What the optimizer sees is exactly what ships — there is no train/pack gap to
lose score to. The scorers are evaluated on the round-tripped frames, the same
way `evaluate.py` will see them.

Training is staged: ${CURRICULUM_SUMMARY — one sentence per stage, filled from
the final training config; no internal stage names}. Weights are tracked with
an EMA shadow and the shadow is what ships.

## 3. The boundary objective

SegNet distortion is an argmax disagreement rate — only pixels whose winning
class flips contribute. Its gradient is zero almost everywhere, so we train
against a margin-based surrogate derived from a level-set view of the class
boundary: the per-pixel margin (winning logit minus runner-up) is treated as a
level-set function whose zero crossing is the class boundary, and the loss is
a smooth monotone function of that margin with an annealed temperature — early
in training the surrogate is wide and moves the whole boundary region; as the
temperature anneals it concentrates gradient on the pixels that are actually
at risk of flipping. This spends the model's limited capacity on the boundary
band instead of on interior pixels that SegNet already labels stably.

## 4. Pose

Pose is not reconstructed from pixels. The 6 pose values per pair are driven
by a stored per-pair ego motion `ξ` (quantized, entropy-coded, ~${DXI_BYTES} B
total), and the render is **conditioned** on the stored `ξ` — in the spirit of
@Quantizr's #55/#56 stored-target conditioning. The conditioning is trained by
joint descent with the boundary objective (the render consumes `ξ` during
training, not as a post-hoc correction), so the shipped frames carry the
photometric structure PoseNet needs to reproduce the stored motion.

## 5. Archive layout

Single ZIP member `${ZIP_MEMBER}` (${ZIP_METHOD}), ${ARCHIVE_BYTES} bytes
total, containing length-prefixed sections:

${SECTION_TABLE}

<!-- SECTION_TABLE is filled from the byte-close manifest: one row per section
with coded size and content; every section must carry its
video-derived-but-not-scorer-output justification per the Final Gate audit. -->

No scorer weights, no stored label/argmax tables, and no per-pixel margin maps
ship in the archive — the payload is the network, the motion, and the seeds.

## 6. Reproduction

`compress.sh` rebuilds `archive.zip` from scratch: it is seeded and
deterministic (same inputs → same bytes) and asserts the final archive SHA-256
(`${ARCHIVE_SHA256}`) so a drifted rebuild fails loudly instead of shipping
different bytes.

## 7. Evaluation

The canonical path, unchanged:

```
archive.zip → inflate.sh → upstream/evaluate.py   (CPU)
```

Inflation is CPU-only (device pinned to CPU), runs in ${INFLATE_MINUTES} min
on the contest CPU runner, and depends on ${AUDITED_DEP_LIST}, all installed
by the packet. Inflation is deterministic on a given host; cross-host
reproducibility is via the fp64 forward path.
