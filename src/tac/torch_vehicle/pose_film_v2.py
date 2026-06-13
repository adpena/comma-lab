# SPDX-License-Identifier: MIT
"""Lever 3 v2 (the OPTIMAL Quantizr pose-solve port) for the torch-vehicle HNeRV.

Design memo: ``.omx/research/lever3_optimal_quantizr_pose_solve_port_trackA_20260613.md``.

The v1 port (``pose_film.PoseFiLMHNeRVWrapper``) injected the pose-FiLM on the
SHARED stem channel ``x`` BEFORE the cascade. That feeds BOTH final heads
(``rgb_0`` -> frame_0, ``rgb_1`` -> frame_1), so the pose conditioning perturbs
``f1`` too. But the contest **SegNet reads ONLY the last frame ``f1``** (verified
``upstream/modules.py`` + the vendored ``model.HNeRVDecoder.forward`` =
``stack([f0, f1], dim=1)`` with SegNet on ``x[:, -1] = f1``). So the v1 stem
injection COUPLES ``d_pose`` into ``d_seg`` (pose changes f1 changes the SegNet
mask) and is high-leverage/unstable (6x8 shared trunk, multiplicative ``gamma*x``).

**The v2 OPTIMAL port** mirrors Quantizr's decoupled dual head: pose-FiLM as a
**RESIDUAL on the ``rgb_0`` head input ONLY**, leaving the ``rgb_1`` (SegNet)
path FiLM-clean::

    x  = stem(z) -> sin -> cascade -> refine     # SHARED feature, NO FiLM on the trunk
    cond = pose_mlp(stored_pose[idx])            # Linear(6->C) -> SiLU -> Linear(C->C)
    x0 = x + film_resid(x, cond)                 # RESIDUAL FiLM, applied ONLY to rgb_0
    f1 = sigmoid(rgb_1(x))  * 255                # CLEAN -> carries d_seg, INVARIANT to pose
    f0 = sigmoid(rgb_0(x0)) * 255                # pose-conditioned -> carries the f0->f1 motion
    return stack([f0, f1], dim=1)

Mapping to Quantizr: ``f1`` = the static "frame2" (seg-clean reference, carries
d_seg); ``f0`` = the pose-FiLM "frame1" (carries the relative-pose motion). The
pair's relative pose = how the pose-conditioned ``f0`` differs from the clean
``f1``.

Properties (all 3, vs v1 which fails 2):
  1. **d_seg / d_pose DECOUPLING at zero extra rate.** ``f1`` has NO FiLM, so the
     pose-FiLM physically cannot perturb its SegNet d_seg. Proved by the
     pose-invariance test (vary the stored pose -> ``f1`` is bit-identical).
  2. **Residual containment** -> bounded pose perturbation (kills v1's
     ``gamma*x+beta`` variance spikes): the identity path anchors ``x0``; the
     FiLM is a *bounded correction*, not a free modulation.
  3. **High-res, head-local injection** (post-refine, 384x512, one head) ->
     minimal leverage vs v1's 6x8 max-leverage shared trunk.

``film_resid(x, cond) = proj(x) * (1 + gamma) + beta`` where ``proj`` is a 1x1
conv ``(final_ch -> final_ch)`` and ``gamma, beta`` are per-channel
``(B, final_ch)`` heads on ``cond``. **The branch is ZERO-INIT** (``proj`` weight
+ bias = 0, and the ``beta`` head = 0): then ``proj(x) = 0`` and ``beta = 0`` so
``film_resid == 0`` exactly -> ``x0 = x`` -> ``f0`` renders BIT-EQUAL to the
vendored ``rgb_0(x)``. The byte-identity / basin-resume contract holds (the live
base_ch=20 basin is unperturbed if it resumes onto this code).

This is a THIN WRAPPER over the vendored decoder (never edits it). Default-OFF in
the driver (``cfg.pose_film_enabled=False`` builds the vendored decoder
unchanged + adds no pose section -> archive BYTE-IDENTICAL). The additive
pose-section grammar + numpy-portable inflate are REUSED verbatim from
``pose_film`` (same ``stored_pose`` buffer, same ``set_stored_pose``, same
``_FiLMEvalDecoder`` cursor adapter), since only the injection point + the
residual form + the cond builder change.

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"); NO MPS for
the exact metric. The in-loop d_pose is ``[contest-CPU advisory]`` NON-PROMOTABLE
until the byte-closed archive is run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the v1 pose-section grammar + eval-cursor adapter VERBATIM (they are
# injection-point-agnostic: they only depend on the wrapper exposing
# ``stored_pose`` (n_pairs, 6), ``set_stored_pose``, and ``forward(z, idx)``).
from tac.torch_vehicle.pose_film import (
    _POSE_DIM,
    _FiLMEvalDecoder,  # re-exported for the driver (works on the v2 wrapper too)
    build_archive_with_pose,
    decode_pose_section,
    encode_pose_section,
    parse_pose_section,
    stored_pose_bytes,
    wrapper_sd_to_archive_decoder_sd,
)

__all__ = [
    "PoseFiLMHNeRVWrapperV2",
    "_FiLMEvalDecoder",
    "_PoseCondMLP",
    "_RGB0ResidualFiLM",
    "build_archive_with_pose",
    "decode_pose_section",
    "encode_pose_section",
    "inflate_film_decoder_v2",
    "parse_pose_section",
    "stored_pose_bytes",
    "wrapper_sd_to_archive_decoder_sd",
]


class _PoseCondMLP(nn.Module):
    """Quantizr's pose-conditioning embedding builder.

        cond = Linear2(SiLU(Linear1(pose6)))     # (B, 6) -> (B, cond_dim)

    NOT zero-init (the cond embedding is allowed to be a real, nonzero function
    of the pose from the start); identity-at-init is enforced DOWNSTREAM at the
    ``_RGB0ResidualFiLM`` zero-init branch, so ``cond`` may be arbitrary while the
    residual it drives is still exactly zero at init.
    """

    def __init__(self, *, pose_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.cond_dim = int(cond_dim)
        self.fc1 = nn.Linear(self.pose_dim, self.cond_dim)
        self.fc2 = nn.Linear(self.cond_dim, self.cond_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(pose)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _RGB0ResidualFiLM(nn.Module):
    """Residual pose-FiLM applied to the ``rgb_0`` head INPUT only.

        film_resid(x, cond) = proj(x) * (1 + gamma(cond)) + beta(cond)
        x0 = x + film_resid(x, cond)

    where ``proj`` is a depthwise-friendly 1x1 conv ``(channels -> channels)`` and
    ``gamma``, ``beta`` are per-channel ``(B, channels)`` linear heads on ``cond``.

    **Identity-at-init (MANDATORY).** ``proj`` (weight + bias) AND the ``beta`` head
    (weight + bias) are zero-init, so ``proj(x) == 0`` and ``beta == 0`` -> the
    multiplicative term ``proj(x) * (1 + gamma)`` is ``0 * (1 + gamma) == 0`` and the
    additive term is ``0`` -> ``film_resid == 0`` exactly, regardless of ``gamma``.
    Thus ``x0 == x`` at init and ``f0`` is bit-equal to the vendored ``rgb_0(x)``.
    (``gamma`` need NOT be zero-init: when ``proj(x)==0`` it is multiplied by zero,
    so it has no effect at init; leaving it nonzero gives a faster-to-escape
    init while preserving the exact identity.)
    """

    def __init__(self, *, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        # 1x1 conv proj over the (B, channels, H, W) feature.
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        # per-channel gamma / beta heads on the pose-cond embedding.
        self.gamma_head = nn.Linear(self.cond_dim, self.channels)
        self.beta_head = nn.Linear(self.cond_dim, self.channels)
        # ZERO-INIT the residual branch -> film_resid == 0 at init (identity).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        # gamma_head is left at default init (it is multiplied by proj(x)==0 at
        # init, so it cannot break the identity; a nonzero gamma escapes faster).

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return the RESIDUAL ``film_resid(x, cond)`` (caller adds it to ``x``).

        ``x`` is ``(B, channels, H, W)``; ``cond`` is ``(B, cond_dim)``.
        """
        gamma = self.gamma_head(cond)[:, :, None, None]  # (B, channels, 1, 1)
        beta = self.beta_head(cond)[:, :, None, None]
        return self.proj(x) * (1.0 + gamma) + beta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PoseFiLMHNeRVWrapperV2(nn.Module):
    """WRAP (never edit) the vendored ``HNeRVDecoder`` with the OPTIMAL pose-FiLM:
    a RESIDUAL on the ``rgb_0`` head only, leaving ``rgb_1`` (the SegNet frame)
    FiLM-clean.

    Vendored ``HNeRVDecoder.forward(z)``::

        x = stem(z).view(B, channels[0], 6, 8) ; x = sin(x)
        for block, skip in zip(blocks, skips):
            identity = skip(interpolate(x, x2)); x = ps(block(x)); x = sin(x + identity)
        x = x + 0.1 * sin(refine(x))               # x: (B, final_ch, 384, 512)
        f0 = sigmoid(rgb_0(x)) * 255 ; f1 = sigmoid(rgb_1(x)) * 255
        return stack([f0, f1], dim=1)

    We REPLICATE that flow EXACTLY (calling the vendored layers unchanged) and
    inject the residual pose-FiLM on the ``rgb_0`` head input::

        ... vendored trunk unchanged -> x = x + 0.1*sin(refine(x)) ...
        cond = pose_mlp(pose6)
        x0 = x + film_resid(x, cond)               # <-- pose-FiLM, rgb_0 path ONLY
        f1 = sigmoid(rgb_1(x))  * 255              # <-- CLEAN x (NOT x0): seg-invariant
        f0 = sigmoid(rgb_0(x0)) * 255

    The wrapper holds the GT pose as a NON-TRAINABLE ``stored_pose`` buffer
    ``(n_pairs, 6)`` (the Quantizr STORE-pose payload, range-coded into the
    additive pose section ~1 KB at byte-close), and looks the per-pair pose up by
    index. ``idx is None`` renders WITHOUT FiLM (the exact vendored path).

    The FiLM submodules are named ``pose_mlp`` + ``film_resid`` so the driver's
    EMA-shadow split (``decoder.*`` -> blob, the FiLM keys -> blob, ``stored_pose``
    -> additive section) finds them; we keep an empty ``pose_film`` alias-free
    design (the driver's split keys on the ``decoder.`` prefix and the
    ``stored_pose`` buffer, both unchanged here).
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        n_pairs: int,
        pose_dim: int = _POSE_DIM,
        film_hidden: int = 8,
    ) -> None:
        super().__init__()
        self.decoder = decoder  # vendored HNeRVDecoder (its params still train)
        self.n_pairs = int(n_pairs)
        self.pose_dim = int(pose_dim)
        # The FiLM injects on the rgb_0 HEAD INPUT -> final_ch = channels[-1].
        self.head_channels = int(decoder.channels[-1])
        cond_dim = max(int(film_hidden), 1)
        self.pose_mlp = _PoseCondMLP(pose_dim=self.pose_dim, cond_dim=cond_dim)
        self.film_resid = _RGB0ResidualFiLM(
            channels=self.head_channels, cond_dim=cond_dim
        )
        self.register_buffer(
            "stored_pose", torch.zeros(self.n_pairs, self.pose_dim), persistent=True
        )

    # --- API parity with v1 (the driver + eval adapter call these unchanged) ---
    def set_stored_pose(self, pose: torch.Tensor) -> None:
        """Set the STORED per-pair 6-dim pose buffer (the Quantizr STORE-pose
        payload). ``pose`` is ``(n_pairs, pose_dim)``."""
        with torch.no_grad():
            p = pose.detach().to(self.stored_pose.dtype).to(self.stored_pose.device)
            if p.shape != self.stored_pose.shape:
                raise ValueError(
                    f"stored_pose expects {tuple(self.stored_pose.shape)}, "
                    f"got {tuple(p.shape)}"
                )
            self.stored_pose.copy_(p)

    def _trunk(self, z: torch.Tensor) -> torch.Tensor:
        """Replicate the vendored trunk EXACTLY -> the shared post-refine feature
        ``x`` (B, final_ch, 384, 512). NO FiLM on the trunk (the v1 bug removed)."""
        d = self.decoder
        B = z.shape[0]
        x = d.stem(z).view(B, d.channels[0], d.base_h, d.base_w)
        x = torch.sin(x)
        for block, skip in zip(d.blocks, d.skips, strict=False):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = d.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(d.refine(x))
        return x

    def _forward_with_film(
        self, z: torch.Tensor, pose6: torch.Tensor | None
    ) -> torch.Tensor:
        """Vendored trunk -> residual pose-FiLM on the ``rgb_0`` path ONLY.

        ``pose6 is None`` (or the zero-init branch at init) reproduces the
        vendored forward bit-for-bit (the byte-identity contract). ``f1`` ALWAYS
        renders from the CLEAN ``x`` (never ``x0``) -> it is INVARIANT to ``pose6``
        (the d_seg/d_pose decoupling)."""
        d = self.decoder
        x = self._trunk(z)
        # f1 is the SegNet frame: render from the CLEAN trunk feature, FiLM-free.
        f1 = torch.sigmoid(d.rgb_1(x)) * 255.0
        if pose6 is None:
            x0 = x
        else:
            cond = self.pose_mlp(pose6)
            x0 = x + self.film_resid(x, cond)
        f0 = torch.sigmoid(d.rgb_0(x0)) * 255.0
        return torch.stack([f0, f1], dim=1)

    def forward(self, z: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``(B, 2, 3, 384, 512)`` float in ``[0, 255]`` — the pair render
        with the ``rgb_0`` head conditioned on the per-pair stored pose and the
        ``rgb_1`` (SegNet) head FiLM-clean.

        ``idx is None`` renders WITHOUT FiLM (the exact vendored path) — used by
        the byte-identity proof."""
        pose6 = None if idx is None else self.stored_pose[idx]
        return self._forward_with_film(z, pose6)


@torch.inference_mode()
def inflate_film_decoder_v2(
    archive_bytes: bytes,
    vendored_parse_archive,
    vendored_decoder_cls,
    *,
    film_hidden: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """Inflate a v2 FiLM archive to raw pair frames — the export-first ROUND-TRIP
    proof (numpy-portable; no challenge deps).

    Parses the vendored 3 sections (decoder + latents + meta) AND the additive
    pose section, rebuilds the vendored decoder, wraps it with the v2 residual
    pose-FiLM whose weights come from the decoder blob (under the ``pose_mlp.*`` /
    ``film_resid.*`` keys), sets the stored pose, and renders every pair. Returns
    ``(n_pairs, 2, 3, 384, 512)`` float in [0, 255]."""
    decoder_sd, latents, meta = vendored_parse_archive(archive_bytes)
    pose = parse_pose_section(archive_bytes, vendored_parse_archive)
    n_pairs = int(meta["n_pairs"])
    # Split the parsed state dict into vendored-decoder keys and FiLM keys.
    film_prefixes = ("pose_mlp.", "film_resid.")
    film_sd = {
        k: v
        for k, v in decoder_sd.items()
        if any(k.startswith(p) for p in film_prefixes)
    }
    dec_sd = {
        k: v
        for k, v in decoder_sd.items()
        if not any(k.startswith(p) for p in film_prefixes)
    }
    decoder = vendored_decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(dec_sd)
    decoder.eval()
    wrapper = PoseFiLMHNeRVWrapperV2(
        decoder, n_pairs=n_pairs, pose_dim=_POSE_DIM, film_hidden=film_hidden
    ).to(device)
    if film_sd:
        # Strip the submodule prefixes already present; load into the wrapper.
        wrapper.load_state_dict({**wrapper.state_dict(), **film_sd}, strict=False)
    if pose is not None:
        wrapper.set_stored_pose(pose.to(device))
    latents = latents.to(device)
    out = []
    for i in range(0, n_pairs, 16):
        j = min(i + 16, n_pairs)
        idx = torch.arange(i, j, device=device)
        out.append(wrapper(latents[i:j], idx))
    return torch.cat(out, dim=0)
