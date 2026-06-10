# SPDX-License-Identifier: MIT
"""The CAPSTONE — our OWN original small VQ-NeRV basis (Task #78).

This is the first original sub-0.15 *attempt* — original frontier synthesis,
NOT a borrowed substrate. It binds three independently-validated pieces from the
clean #82/#76/#67/#68 stack into ONE coherent score-aware codec:

1. **VQ-NeRV basis** (the #67 free-inflate): per-pair latents are quantized via
   the canonical ``VectorQuantizerEMAMLX`` (van den Oord 1711.00937). The
   codebook is "free" in the native MLX decode (it lives in the decoder weights
   that are paid once); only the per-pair *index* costs bytes, and it bit-packs
   to ``ceil(log2(K))`` bits/pair. This is the rate lever that a continuous
   28-d-fp16 latent cannot match.

2. **Store-pose-FiLM** (the #81 finding + Quantizr's proven approach): the 6-dim
   GT PoseNet output is STORED explicitly (~kilobytes, brotli) and
   FiLM-INJECTED into the decoder, modulating the rendered frames so the pose
   term is inherited from the stored scalars rather than reconstructed from
   pixels. This SIDESTEPS the seg/pose antagonism the small-conv student hits
   (``smaller_student.py``): pose is not a capacity wall when it is stored, not
   regressed.

3. **The #82 working score-aware loop**: SEG descends via direct CE through the
   LIVE frozen torch SegNet, propagated to the MLX decoder via the
   ``mx.vjp`` ↔ torch-scorer bridge (the headline #82 mechanism: d_seg 0.806 →
   0.008). Muon-throughout + EMA. The pose half of the objective re-anchors the
   FiLM injection to the stored GT pose.

``borrowed_substrate_accounting``: the VQ-EMA primitive (van den Oord) and the
HNeRV decoder backbone (PR95) are PUBLIC method components reused as kernels;
the SYNTHESIS — a VQ-quantized-latent NeRV decoder with explicit-pose-FiLM
injection trained score-aware through the live scorer for the contest's exact
``100*d_seg + sqrt(10*d_pose) + rate`` objective — is OURS-ORIGINAL. No
competitor's published codec is being absorb-recoded; this is a new niche.

All MLX/scorer numerics are ``[macOS-MLX research-signal]`` / ``[macOS-CPU
advisory]`` (non-promotable per CLAUDE.md). A contest score requires
``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

from tac.capstone_vq_nerv.export import (
    CapstoneArchiveAccount,
    bit_pack_vq_indices,
    bit_unpack_vq_indices,
    build_capstone_archive_bytes,
)
from tac.capstone_vq_nerv.vq_nerv_bundle import (
    CapstoneVqNervBundle,
    CapstoneVqNervConfig,
)

__all__ = [
    "CapstoneArchiveAccount",
    "CapstoneVqNervBundle",
    "CapstoneVqNervConfig",
    "bit_pack_vq_indices",
    "bit_unpack_vq_indices",
    "build_capstone_archive_bytes",
]
