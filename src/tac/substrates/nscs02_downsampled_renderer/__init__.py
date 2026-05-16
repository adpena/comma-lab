# SPDX-License-Identifier: MIT
"""NSCS02 — Downsampled Renderer + Inflate Upsample substrate.

NSCS02 is the highest-EV / lowest-risk shift in the
ASSUMPTIONS-CHALLENGE-AUDIT (`.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json`)
NSCS02 entry: violates ``SA03_segnet_stride_2_stem`` /
``SA05_inflate_at_camera_native`` / ``SA17_rgb_format``. Predicted
ΔS = ``[-0.010, -0.030]`` at ``$5-15`` Modal T4 smoke. STACKS with A1
sidecars per the audit composition matrix.

Math basis (verbatim from audit entry NSCS02):
  Both scorers internally interpolate to (384, 512). The (1164, 874)
  intermediate is lossy compute. SegNet stride-2 stem already discards
  half resolution. At (192, 256), 16x pixel reduction; if SegNet
  5-class argmax + PoseNet 6-dim pose preserved, ZERO d_seg / d_pose
  cost.

Per the standing directive
``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``
this substrate is built as a UNIQUE-AND-COMPLETE-PER-METHOD package
(architecture + score-aware loss + archive grammar + inflate runtime +
export contract) bound into one coherent reviewable implementation. See
the canonical-vs-unique decision section in the landing memo
``feedback_nscs02_downsampled_renderer_inflate_upsample_build_landed_20260515.md``.

Directory layout (mirrors ATW / A1 substrate convention):
- :mod:`tac.substrates.nscs02_downsampled_renderer.architecture` —
  5-stage NSCS02 decoder (renders at 192x256 vs A1's 384x512).
- :mod:`tac.substrates.nscs02_downsampled_renderer.score_aware_loss` —
  unique loss for the downsample-then-upsample composite pipeline.
- :mod:`tac.substrates.nscs02_downsampled_renderer.archive` — packed
  archive grammar + parser-section manifest + byte-identical
  encode/decode roundtrip.
- :mod:`tac.substrates.nscs02_downsampled_renderer.inflate` — package
  ``inflate(archive_bytes, output_dir, file_list)`` consumer that
  delegates to the standalone submission inflate at
  ``submissions/nscs02_downsampled_renderer/inflate.py`` per HNeRV
  parity discipline lesson 9 (runtime closure).

Catalog #220 mechanism status:
  ``score_improvement_mechanism_status=RESEARCH_ONLY`` until the resizing-chain
  ablation proves the downsampled-renderer bytes remain score-relevant under
  the actual scorer path and paired CPU+CUDA exact eval lands.

Dispatch status:
  ``research_only=true`` until ``experiments/train_substrate_nscs02_downsampled_renderer.py``
  implements the non-smoke training/export path and a strict operator recipe
  can route paired CPU+CUDA auth-eval custody. The current local ``--smoke``
  path is parser/runtime evidence only, not a promotion result.
"""

from __future__ import annotations

__all__ = [
    "NSCS02_ARCHIVE_MAGIC",
    "NSCS02_BASE_CHANNELS",
    "NSCS02_LATENT_DIM",
    "NSCS02_N_PAIRS",
    "NSCS02_RENDER_HW",
]

# Render at (H=192, W=256) — exactly half A1's (384, 512) along each axis.
# Factor-2 downscale per axis means the renderer omits ONE of A1's 6
# PixelShuffle stages (5 stages from 6x8 to 192x256 == 32x scale-up).
NSCS02_RENDER_HW: tuple[int, int] = (192, 256)
NSCS02_LATENT_DIM: int = 28
NSCS02_BASE_CHANNELS: int = 36
NSCS02_N_PAIRS: int = 600

# Archive magic byte sequence "NSCS02\x00\x01" — version 1.
NSCS02_ARCHIVE_MAGIC: bytes = b"NSCS02\x00\x01"
