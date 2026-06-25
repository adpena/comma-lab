# SPDX-License-Identifier: MIT
"""Boundary capacity-routing for the binding d_seg residual (the class-1 lane edge).

The binding ``d_seg`` residual at the live small-basis HNeRV basin is the class-1
LANE-marking band: only ~0.64-0.72% of pixels (a codim-1 boundary curve), yet it
dominates ``100*d_seg`` because every flipped argmax pixel on that thin band costs
the same as a flipped pixel anywhere. The DAG d_seg lever
(``.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`` FEEDs
2026-06-25b/d/j) decomposes the fix into:

    CAPACITY-ROUTING = WHERE (refine at the ~0.72% band)
                     x BASIS (directional, matched to the lane = smooth C2 curve)
                     x SHARPNESS (step-like nonlinearity across the edge)

This module is the **WHERE** + a tiny **SHARPNESS** modulator + an optional
**BASIS** helper, all at ~0 byte cost (the routing key is computed from the GT
argmax, which the decoder never needs to store — it is a *training-time* prior
that concentrates the existing ~83K conv weights on the boundary band):

* :func:`boundary_distance_map` — the WHERE prior. Per-pixel (unsigned or signed)
  Euclidean distance to the nearest class-1(lane) boundary, via
  ``scipy.ndimage.distance_transform_edt``. Computable from the GT argmax alone;
  zero archive bytes (it is a fixed function of the frozen GT targets, identical
  at train and inflate, so nothing about it is *stored*).
* :func:`boundary_proximity_feature` — squashes the distance to a bounded
  ``[0, 1]`` proximity ``exp(-d / tau)`` (the routing *key* fed to the FiLM gate;
  near-boundary -> 1, interior -> 0).
* :class:`BoundaryFiLM` — a coordinate-conditioned FiLM gate. Given a decoder
  feature map and the boundary-proximity feature it emits per-pixel
  ``(gamma, beta)`` that give near-boundary pixels MORE effective nonlinearity /
  capacity. Identity-at-init (zero-init heads) so turning it on is a no-op at
  step 0 (the live basin is unperturbed if it resumes onto this code). Tiny: the
  added params are O(C) per the FiLM heads, negligible vs the ~83K conv weights.
* :func:`local_boundary_tangent` + :func:`directional_positional_encoding` — the
  optional **BASIS** helper: an *anisotropic* (oriented) Fourier feature encoding
  whose frequency vectors are stretched ALONG the local boundary tangent (so the
  representation is high-frequency ACROSS the edge and low-frequency ALONG it).
  This is the provably-more-efficient directional inductive bias for a codim-1
  smooth curve (curvelet O(N^-2) >> isotropic wavelet O(N^-1)).

Borrowed-substrate accounting (CLAUDE.md NO-FAKE + borrowed-substrate rule):

* BORROWED: ``scipy.ndimage.distance_transform_edt`` (the exact Euclidean
  distance transform — Felzenszwalb-Huttenlocher / Maurer algorithm).
* BORROWED: the anisotropic-Fourier-feature *mechanism* — per-direction frequency
  scaling generalizing isotropic random Fourier features. AFPE, Kuckelhaus et al.
  2025 (arXiv:2509.02488); the isotropic base is Tancik et al. 2020 (Fourier
  Features Let Networks Learn High Frequency Functions); the spatially-adaptive
  capacity-localization idea is SASNet, 2025 (arXiv:2503.09750); the oriented /
  steerable directional basis lineage is Freeman-Adelson steerable filters (1991)
  + Candes-Donoho curvelets (2004).
* OURS (the novel application): using the GT-argmax class-1 boundary distance map
  as the routing KEY for a FiLM capacity-router AND as the orientation field for
  an oriented PE, specifically to concentrate a small HNeRV decoder's capacity on
  the contest's binding ``d_seg`` lane-edge band at ~0 archive cost. The
  boundary-routing-for-argmax-edge framing, the identity-at-init contest-safe
  wire-in, and the proximity-key codec-free design are ours.

Authority: this is a training-time prior + a tiny module. NO score claim here.
Any d_seg movement it produces is ``[contest-CPU advisory]`` NON-PROMOTABLE until
the byte-closed archive is run through ``upstream/evaluate.py``. NO MPS for the
metric. ``boundary_distance_map`` is pure CPU/numpy/scipy; the FiLM module is
device-agnostic torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Contest SegNet class index for the lane markings (the binding d_seg band).
LANE_CLASS: int = 1
_SEG_H = 384
_SEG_W = 512


# ---------------------------------------------------------------------------
# WHERE: boundary distance map (the ~0-byte routing prior)
# ---------------------------------------------------------------------------
def boundary_distance_map(
    seg_targets_hard: torch.Tensor,
    *,
    lane_class: int = LANE_CLASS,
    signed: bool = False,
) -> torch.Tensor:
    """Per-pixel distance to the nearest class-``lane_class`` boundary.

    Args:
        seg_targets_hard: integer GT argmax of shape ``(B, H, W)`` or ``(H, W)``
            with class indices (contest: 0..4). A single ``(H, W)`` map is
            promoted to ``(1, H, W)``.
        lane_class: the class whose boundary we route to (default 1 = lane).
        signed: if True, return a SIGNED distance (negative inside the lane
            region, positive outside) — the level-set form used by the oriented
            PE. If False (default), return the UNSIGNED distance to the boundary
            CURVE (zero exactly on the boundary band, growing away from it on
            both sides) — the form used by :func:`boundary_proximity_feature`.

    Returns:
        Float32 tensor, same leading shape as the (promoted) input
        ``(B, H, W)``, on the SAME device as ``seg_targets_hard``. Units are
        pixels.

    The "boundary" is the codim-1 curve separating ``lane_class`` from the rest.
    We compute the canonical level-set signed distance ``sdf = EDT(~lane) -
    EDT(lane)`` (negative inside the lane region, positive outside). The UNSIGNED
    distance to the boundary curve is ``|sdf|`` — small (~1 px, since the curve
    runs between pixels) on the band edges and growing on BOTH sides. (Note:
    because the interface is between pixels it is ~1, not exactly 0, on the
    pixels straddling the edge; the proximity key ``exp(-d/tau)`` is therefore
    ~1, not exactly 1, on the band — which is the intended behavior.)

    Borrowed: ``scipy.ndimage.distance_transform_edt``. Pure CPU (numpy); we move
    the result back onto the input device. Zero archive bytes — this is a fixed
    function of the frozen GT targets, identical at train and inflate.
    """
    from scipy import ndimage

    if seg_targets_hard.dim() == 2:
        seg_targets_hard = seg_targets_hard.unsqueeze(0)
    if seg_targets_hard.dim() != 3:
        raise ValueError(
            "seg_targets_hard must be (B, H, W) or (H, W); got shape "
            f"{tuple(seg_targets_hard.shape)}"
        )
    device = seg_targets_hard.device
    seg_np = seg_targets_hard.detach().to("cpu").numpy()
    out = np.empty(seg_np.shape, dtype=np.float32)
    for b in range(seg_np.shape[0]):
        lane = seg_np[b] == lane_class
        if not lane.any():
            # No lane pixels in this frame: distance is +inf everywhere. Use a
            # large finite sentinel (the diagonal) so downstream exp(-d/tau)=0
            # (interior) and the tensor stays finite. Signed stays all-positive.
            diag = float(np.hypot(seg_np[b].shape[0], seg_np[b].shape[1]))
            out[b] = diag
            continue
        if lane.all():
            diag = float(np.hypot(seg_np[b].shape[0], seg_np[b].shape[1]))
            out[b] = -diag if signed else diag
            continue
        d_out = ndimage.distance_transform_edt(~lane).astype(np.float32)
        d_in = ndimage.distance_transform_edt(lane).astype(np.float32)
        sdf = d_out - d_in  # signed: negative inside lane, positive outside
        if signed:
            out[b] = sdf
        else:
            # unsigned distance to the boundary curve = |signed distance| (small
            # ~1 on the band edges, growing on both sides). NOT min(d_out, d_in)
            # — that degenerates to 0 everywhere because d_in==0 outside the lane
            # and d_out==0 inside it.
            out[b] = np.abs(sdf)
    return torch.from_numpy(out).to(device)


def boundary_proximity_feature(
    distance_map: torch.Tensor,
    *,
    tau: float = 4.0,
) -> torch.Tensor:
    """Squash an UNSIGNED boundary distance map to a ``[0, 1]`` proximity key.

    ``proximity = exp(-distance / tau)`` — 1 exactly on the boundary band,
    decaying to 0 in the interior with length-scale ``tau`` pixels. This is the
    routing key the :class:`BoundaryFiLM` gate consumes; ``tau`` sets the width
    of the boundary band that receives extra capacity.

    Returns the same shape as ``distance_map`` (``(B, H, W)``), float, device
    preserved. Requires ``tau > 0``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0; got {tau}")
    dm = distance_map.clamp_min(0.0).to(torch.float32)
    return torch.exp(-dm / float(tau))


# ---------------------------------------------------------------------------
# SHARPNESS: boundary-conditioned FiLM capacity router
# ---------------------------------------------------------------------------
class BoundaryFiLM(nn.Module):
    r"""Per-pixel FiLM gate that routes capacity to the lane-boundary band.

    Given a decoder feature map ``x`` of shape ``(B, C, H, W)`` and a
    boundary-proximity feature ``p`` of shape ``(B, 1, H, W)`` (in ``[0, 1]``,
    1 on the boundary band), emit per-pixel per-channel ``(gamma, beta)`` and
    return ``x * (1 + p * gamma) + p * beta``.

    The ``p *`` factor makes the modulation **vanish in the interior** (``p=0``)
    and act **only on the boundary band** (``p≈1``): the FiLM concentrates the
    decoder's effective nonlinearity / dynamic range exactly where the binding
    ``d_seg`` residual lives, leaving the (already-good) interior untouched.

    Mechanism per layer: ``p`` is lifted to a small per-pixel embedding by a 1x1
    conv, then two 1x1-conv heads produce ``gamma`` and ``beta`` (per channel,
    per pixel). Borrowed mechanism: FiLM (Perez et al. 2018) + the SASNet
    spatially-adaptive masking idea (modulate the sinusoidal/conv layer by a
    learned spatial mask). Ours: keying the mask on the GT class-1 boundary
    proximity for contest d_seg routing.

    **Identity-at-init (MANDATORY).** ``gamma_head`` and ``beta_head`` are
    zero-init (weight AND bias), so at step 0 ``gamma == 0`` and ``beta == 0`` ->
    output ``== x`` exactly, regardless of ``p`` or the (default-init) embedding.
    Turning the router on therefore does NOT perturb the live basin on the first
    step; it can only HELP as training escapes the zero.

    Param count: ``embed`` (1->E: E*(1+1)) + gamma_head (E->C: C*(E+1)) +
    beta_head (E->C: C*(E+1)) = ``2E + 2C*(E+1)``. For the default ``E=8`` and a
    typical decoder stage ``C=64`` this is ``16 + 2*64*9 = 1168`` params — well
    under 2% of the ~83K conv weights. :meth:`param_count` returns the exact
    number.
    """

    def __init__(self, *, channels: int, embed_dim: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0; got {channels}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0; got {embed_dim}")
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        # 1x1 conv embedding of the scalar proximity key -> per-pixel embedding.
        self.embed = nn.Conv2d(1, self.embed_dim, kernel_size=1)
        # gamma / beta heads (1x1 conv, per-channel per-pixel modulation).
        self.gamma_head = nn.Conv2d(self.embed_dim, self.channels, kernel_size=1)
        self.beta_head = nn.Conv2d(self.embed_dim, self.channels, kernel_size=1)
        # Identity-at-init: zero the two heads (weight + bias) so the modulation
        # is exactly zero at step 0 regardless of p / embed.
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def param_count(self) -> int:
        """Exact number of learnable parameters added by this gate."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, proximity: torch.Tensor) -> torch.Tensor:
        """Apply boundary-routed FiLM.

        Args:
            x: decoder feature map ``(B, C, H, W)``.
            proximity: boundary-proximity key, broadcastable to ``(B, 1, H, W)``.
                Accepts ``(B, H, W)`` or ``(B, 1, H, W)``; it is resized
                (bilinear) to ``x``'s spatial size if it differs (decoder stages
                run at lower resolution than the 384x512 GT band).

        Returns:
            ``x * (1 + p * gamma) + p * beta`` of shape ``(B, C, H, W)``.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"x has {x.shape[1]} channels but gate was built for "
                f"{self.channels}"
            )
        if proximity.dim() == 3:
            proximity = proximity.unsqueeze(1)
        if proximity.dim() != 4 or proximity.shape[1] != 1:
            raise ValueError(
                "proximity must be (B, H, W) or (B, 1, H, W); got "
                f"{tuple(proximity.shape)}"
            )
        p = proximity.to(x.dtype)
        if p.shape[-2:] != x.shape[-2:]:
            p = F.interpolate(
                p, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        emb = self.embed(p)
        gamma = self.gamma_head(emb)
        beta = self.beta_head(emb)
        return x * (1.0 + p * gamma) + p * beta


# ---------------------------------------------------------------------------
# BASIS: oriented / anisotropic positional encoding (grounded in part A)
# ---------------------------------------------------------------------------
def local_boundary_tangent(signed_distance_map: torch.Tensor) -> torch.Tensor:
    """Estimate the local boundary TANGENT direction at every pixel.

    The gradient of a (signed) distance field points along the boundary NORMAL;
    rotating it 90 degrees gives the TANGENT. We compute the gradient via central
    differences and return the unit tangent ``(tx, ty)`` per pixel.

    Args:
        signed_distance_map: ``(B, H, W)`` (or ``(H, W)``) signed distance field
            (see :func:`boundary_distance_map` with ``signed=True``).

    Returns:
        ``(B, H, W, 2)`` float tensor of unit tangent vectors ``(tx, ty)``.
        Where the gradient is ~0 (far interior, flat field) the tangent defaults
        to ``(1, 0)`` (arbitrary but unit).
    """
    if signed_distance_map.dim() == 2:
        signed_distance_map = signed_distance_map.unsqueeze(0)
    if signed_distance_map.dim() != 3:
        raise ValueError(
            "signed_distance_map must be (B, H, W) or (H, W); got "
            f"{tuple(signed_distance_map.shape)}"
        )
    d = signed_distance_map.to(torch.float32)
    # central differences (gradient of the field): gy along H, gx along W.
    gy = torch.zeros_like(d)
    gx = torch.zeros_like(d)
    gy[:, 1:-1, :] = (d[:, 2:, :] - d[:, :-2, :]) * 0.5
    gx[:, :, 1:-1] = (d[:, :, 2:] - d[:, :, :-2]) * 0.5
    # normal = (gx, gy); tangent = rotate 90 deg -> (-gy, gx).
    tx = -gy
    ty = gx
    norm = torch.sqrt(tx * tx + ty * ty).clamp_min(1e-8)
    tx = tx / norm
    ty = ty / norm
    # where the field is flat (norm ~ 0 before clamp) default to (1, 0).
    flat = (torch.sqrt(gx * gx + gy * gy) < 1e-6)
    tx = torch.where(flat, torch.ones_like(tx), tx)
    ty = torch.where(flat, torch.zeros_like(ty), ty)
    return torch.stack([tx, ty], dim=-1)


def directional_positional_encoding(
    coords: torch.Tensor,
    tangent: torch.Tensor,
    *,
    n_freqs: int = 6,
    freq_across: float = 32.0,
    freq_along: float = 4.0,
) -> torch.Tensor:
    r"""Anisotropic (oriented) Fourier-feature positional encoding.

    Grounded in part-A research: AFPE / anisotropic Fourier features
    (arXiv:2509.02488) generalize isotropic random Fourier features
    (Tancik 2020) by scaling frequency DIFFERENTLY per direction; here we orient
    those directions to the LOCAL boundary geometry (steerable-filter /
    curvelet lineage). For each pixel we build a local orthonormal frame
    ``(tangent t, normal n)`` and encode the coordinate's projection onto each
    with DIFFERENT frequency scales: HIGH frequency across the edge (``freq_across``,
    along the normal — where the signal changes fast) and LOW frequency along the
    edge (``freq_along``, along the tangent — where the lane is a smooth curve).

    Mathematically, for octave ``k`` and projections ``u_t = <coord, t>``,
    ``u_n = <coord, n>``::

        f_k_along  = freq_along  * 2**k
        f_k_across = freq_across * 2**k
        enc += [ sin(2*pi*f_k_along*u_t),  cos(2*pi*f_k_along*u_t),
                 sin(2*pi*f_k_across*u_n), cos(2*pi*f_k_across*u_n) ]

    Args:
        coords: per-pixel coordinates ``(B, H, W, 2)`` as ``(x, y)``, typically
            normalized to ``[0, 1]`` or ``[-1, 1]``.
        tangent: per-pixel unit tangent ``(B, H, W, 2)`` (from
            :func:`local_boundary_tangent`).
        n_freqs: number of octaves.
        freq_across / freq_along: base frequency along the normal / tangent. The
            anisotropy ratio ``freq_across / freq_along`` is the directional bias.

    Returns:
        ``(B, H, W, 4 * n_freqs)`` float tensor (sin/cos x along/across per
        octave). Deterministic (fixed-frequency, no random projection) so it is
        identical at train and inflate and adds ZERO archive bytes.
    """
    if coords.shape[-1] != 2 or tangent.shape[-1] != 2:
        raise ValueError(
            "coords and tangent last dim must be 2; got coords "
            f"{tuple(coords.shape)}, tangent {tuple(tangent.shape)}"
        )
    if coords.shape[:-1] != tangent.shape[:-1]:
        raise ValueError(
            "coords and tangent must share leading shape; got "
            f"{tuple(coords.shape)} vs {tuple(tangent.shape)}"
        )
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be > 0; got {n_freqs}")
    tx = tangent[..., 0]
    ty = tangent[..., 1]
    # normal = rotate tangent -90 deg -> (ty, -tx) (orthonormal to tangent).
    nx = ty
    ny = -tx
    cx = coords[..., 0]
    cy = coords[..., 1]
    u_t = cx * tx + cy * ty  # projection onto tangent  (B,H,W)
    u_n = cx * nx + cy * ny  # projection onto normal    (B,H,W)
    feats = []
    two_pi = 2.0 * np.pi
    for k in range(n_freqs):
        f_along = float(freq_along) * (2.0**k)
        f_across = float(freq_across) * (2.0**k)
        feats.append(torch.sin(two_pi * f_along * u_t))
        feats.append(torch.cos(two_pi * f_along * u_t))
        feats.append(torch.sin(two_pi * f_across * u_n))
        feats.append(torch.cos(two_pi * f_across * u_n))
    return torch.stack(feats, dim=-1)


__all__ = [
    "LANE_CLASS",
    "BoundaryFiLM",
    "boundary_distance_map",
    "boundary_proximity_feature",
    "directional_positional_encoding",
    "local_boundary_tangent",
]
