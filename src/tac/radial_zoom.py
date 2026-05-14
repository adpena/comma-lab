# SPDX-License-Identifier: MIT
"""Radial zoom warp — information-theoretic minimum for PoseNet ego-motion.

The PoseNet Jacobian has effective rank 1.008 (verified empirically in
experiments/results/gradient_rank_analysis.json across 5 pairs).  Only one
degree of freedom matters: a scalar radial zoom centered at the Focus of
Expansion (FoE).  This replaces the 50K-param MotionPredictor with 600
learned scalars (1.2 KB at FP16, 300 bytes at FP4).

The warp for pair *i* is::

    grid = foe + exp(s_i) * (coord_grid - foe)
    frame_t = grid_sample(frame_t1, grid, mode='bilinear', align_corners=True)

where ``s_i`` is a small learned scalar (initialised near 0 = identity zoom).

Compatible with :func:`tac.renderer.warp_with_flow` — returns ``(B, 2, H, W)``
flow in normalised ``[-1, 1]`` grid-sample coordinates.

Council origin
--------------
* Hotz: "if it's rank-1 just learn a scalar."
* Fridrich: "radial zoom *is* the forward-translation FOE structure."
* Quantizr: "our affine table has 3600 params; 600 scalars is 6x cheaper."
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W, VANISHING_POINT

__all__ = [
    "RadialZoomWarp",
    "save_zoom_scalars",
    "load_zoom_scalars",
    "optimize_zoom_scalars",
]


class RadialZoomWarp(nn.Module):
    """Per-pair scalar radial zoom from the Focus of Expansion.

    The PoseNet Jacobian has effective rank 1.008 — only one degree of
    freedom matters.  That DOF is a radial zoom centered at the vanishing
    point (Focus of Expansion).  This replaces the 50K-param MotionPredictor
    with 600 learned scalars (1.2 KB at FP16).

    The warp for pair *i* is::

        grid = foe + exp(s_i) * (coord_grid - foe)
        frame_t = grid_sample(frame_t1, grid, mode='bilinear', align_corners=True)

    where ``s_i`` is a small learned scalar (initialised near 0 = identity zoom).

    Parameters
    ----------
    n_pairs : int
        Number of frame pairs (600 for the comma challenge: 1200 frames / 2).
    foe_h : float
        Focus of Expansion row in scorer coordinates (default 174.0 from
        empirical gradient analysis and camera.py VANISHING_POINT).
    foe_w : float
        Focus of Expansion column in scorer coordinates (default 256.0).
    target_h : int
        Scorer frame height (384).
    target_w : int
        Scorer frame width (512).
    max_zoom_log : float
        Bound on the log-zoom: ``|s_i| <= max_zoom_log``.
        ``exp(±0.1) ≈ 0.905..1.105``, i.e. roughly ±10 % zoom.
    learn_foe : bool
        If ``True``, the FoE position becomes a learnable parameter.
    """

    def __init__(
        self,
        n_pairs: int = 600,
        foe_h: float = float(VANISHING_POINT[1]),   # 174.0
        foe_w: float = float(VANISHING_POINT[0]),    # 256.0
        target_h: int = SEGNET_INPUT_H,              # 384
        target_w: int = SEGNET_INPUT_W,              # 512
        max_zoom_log: float = 0.15,  # Council: 0.1 tight for 70mph; 0.15 gives gradient headroom
        learn_foe: bool = False,
    ) -> None:
        super().__init__()
        self.n_pairs = n_pairs
        self.target_h = target_h
        self.target_w = target_w
        self.max_zoom_log = max_zoom_log
        self.learn_foe = learn_foe

        # 600 scalars, initialised to zero (identity zoom — no warp)
        self.zoom_scalars = nn.Parameter(torch.zeros(n_pairs))

        # FoE in *pixel* coordinates (scorer resolution)
        foe_init = torch.tensor([foe_h, foe_w], dtype=torch.float32)
        if learn_foe:
            self.foe = nn.Parameter(foe_init)
        else:
            self.register_buffer("foe", foe_init)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pair_indices: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute zoom flow fields for the given pair indices.

        The output is a *flow offset* in normalised ``[-1, 1]`` coordinates,
        compatible with :func:`tac.renderer.warp_with_flow`.  That function
        adds the flow to a base coordinate grid before calling
        ``F.grid_sample(..., align_corners=True)``.

        Parameters
        ----------
        pair_indices : torch.Tensor
            ``(B,)`` int tensor of pair indices into ``self.zoom_scalars``.
        H, W : int
            Spatial dimensions of the output flow field (typically 384, 512).

        Returns
        -------
        torch.Tensor
            ``(B, 2, H, W)`` flow field.  Channel 0 = x (horizontal),
            channel 1 = y (vertical), in normalised ``[-1, 1]`` space.
        """
        device = self.zoom_scalars.device
        B = pair_indices.shape[0]

        # ── 1. Look up & bound the zoom scalar ────────────────────────
        s = torch.tanh(self.zoom_scalars[pair_indices]) * self.max_zoom_log  # (B,)
        zoom_factor = torch.exp(s)  # (B,) — near 1.0 for small s

        # ── 2. FoE in normalised coordinates [-1, 1]  (align_corners) ─
        # align_corners=True convention:
        #   pixel 0 → -1,  pixel (N-1) → +1,  so  norm = 2*px/(N-1) - 1
        foe_h_norm = 2.0 * self.foe[0] / (H - 1) - 1.0
        foe_w_norm = 2.0 * self.foe[1] / (W - 1) - 1.0

        # ── 3. Build normalised coordinate grid ───────────────────────
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # (H, W) each

        # ── 4. Compute warped grid ────────────────────────────────────
        # warped = foe + zoom * (coord - foe)
        # flow   = warped - coord  =  (zoom - 1) * (coord - foe)
        #
        # warp_with_flow adds flow to the base grid, so we return the *delta*.
        z = (zoom_factor - 1.0).view(B, 1, 1)  # (B, 1, 1)

        flow_x = z * (grid_x.unsqueeze(0) - foe_w_norm)  # (B, H, W)
        flow_y = z * (grid_y.unsqueeze(0) - foe_h_norm)  # (B, H, W)

        # ── 5. Stack as (B, 2, H, W): channel 0 = x, channel 1 = y ──
        return torch.stack([flow_x, flow_y], dim=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Total learnable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def archive_bytes_fp16(self) -> int:
        """Estimated archive size at FP16 (2 bytes per scalar)."""
        n = self.zoom_scalars.numel()
        if self.learn_foe:
            n += 2  # foe has 2 values
        return n * 2

    def archive_bytes_fp4(self) -> int:
        """Estimated archive size at FP4 (0.5 bytes per scalar)."""
        n = self.zoom_scalars.numel()
        if self.learn_foe:
            n += 2
        return n // 2

    def __repr__(self) -> str:
        return (
            f"RadialZoomWarp(n_pairs={self.n_pairs}, "
            f"foe=({self.foe[0].item():.1f}, {self.foe[1].item():.1f}), "
            f"max_zoom_log={self.max_zoom_log}, "
            f"learn_foe={self.learn_foe}, "
            f"params={self.param_count()}, "
            f"archive_fp16={self.archive_bytes_fp16()} B)"
        )

    def warp_inverse_masks(
        self,
        masks_t1: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Warp t+1 (odd-frame) masks BACKWARD to recover t (even-frame) masks.

        Used at inflate when only odd-frame masks are stored in the archive
        (Quantizr's half-frame paradigm). For pair k:
            mask_t[k] = inverse_zoom(mask_t1[k], -zoom_scalars[k])

        Discrete classes are preserved by ``mode='nearest'`` resampling. Out-
        of-bounds samples take the border class (0 = road) which matches the
        physical interpretation: pixels beyond the camera's previous-frame
        field of view are unknown, treat as the most common class.

        Parameters
        ----------
        masks_t1 : torch.Tensor
            ``(B, H, W)`` integer tensor of class-index masks for time t+1.
        pair_indices : torch.Tensor
            ``(B,)`` int tensor of pair indices to look up zoom scalars.

        Returns
        -------
        torch.Tensor
            ``(B, H, W)`` integer tensor of warped masks for time t.
        """
        import torch.nn.functional as F

        device = masks_t1.device
        B, H, W = masks_t1.shape

        # Forward zoom for these pairs: s_k = tanh(scalars[k]) * max_zoom_log
        s = torch.tanh(self.zoom_scalars[pair_indices]) * self.max_zoom_log
        # Inverse zoom: negate s so the mapping goes t+1 → t.
        zoom_factor_inv = torch.exp(-s)  # (B,)

        # FoE in normalised coords (matches forward())
        foe_h_norm = 2.0 * self.foe[0] / (H - 1) - 1.0
        foe_w_norm = 2.0 * self.foe[1] / (W - 1) - 1.0

        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        z = (zoom_factor_inv - 1.0).view(B, 1, 1)
        flow_x = z * (grid_x.unsqueeze(0) - foe_w_norm)
        flow_y = z * (grid_y.unsqueeze(0) - foe_h_norm)

        base_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        base_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        warp_grid = torch.stack([base_x + flow_x, base_y + flow_y], dim=-1)  # (B, H, W, 2)

        # grid_sample needs float input; mask is class indices.
        masks_float = masks_t1.float().unsqueeze(1)  # (B, 1, H, W)
        sampled = F.grid_sample(
            masks_float,
            warp_grid,
            mode="nearest",          # preserve class boundaries
            padding_mode="border",   # OOB pixels take the nearest valid class
            align_corners=True,
        )
        return sampled.squeeze(1).to(masks_t1.dtype)


# ======================================================================
# Serialisation helpers
# ======================================================================


def save_zoom_scalars(zoom_warp: RadialZoomWarp, path: Path) -> int:
    """Save zoom scalars as raw FP16 binary.

    Format: ``n_pairs`` float16 values in native byte order, optionally
    followed by 2 float16 FoE values when ``learn_foe=True``.

    Parameters
    ----------
    zoom_warp : RadialZoomWarp
        Module whose scalars to save.
    path : Path
        Destination file (e.g. ``zoom_scalars.bin``).

    Returns
    -------
    int
        Number of bytes written.
    """
    path = Path(path)
    parts = [zoom_warp.zoom_scalars.detach().cpu().half()]
    if zoom_warp.learn_foe:
        parts.append(zoom_warp.foe.detach().cpu().half())
    blob = torch.cat(parts).numpy().tobytes()
    path.write_bytes(blob)
    return len(blob)


def load_zoom_scalars(
    path: Path,
    n_pairs: int = 600,
    learn_foe: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load zoom scalars from raw FP16 binary.

    Parameters
    ----------
    path : Path
        Source file written by :func:`save_zoom_scalars`.
    n_pairs : int
        Expected number of pairs (for validation).
    learn_foe : bool
        Whether the file also contains 2 FoE values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        ``(zoom_scalars, foe_or_none)`` as float32 tensors.

    Raises
    ------
    ValueError
        If the file size does not match expectations.
    """
    path = Path(path)
    raw = path.read_bytes()
    expected_n = n_pairs + (2 if learn_foe else 0)
    expected_bytes = expected_n * 2  # FP16 = 2 bytes
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes ({expected_n} FP16 values), "
            f"got {len(raw)} bytes in {path}"
        )
    import numpy as np

    data = torch.from_numpy(
        np.frombuffer(raw, dtype=np.float16).copy()
    ).float()
    scalars = data[:n_pairs]
    foe = data[n_pairs:] if learn_foe else None
    return scalars, foe


# ======================================================================
# Gradient-based optimisation of zoom scalars
# ======================================================================


def optimize_zoom_scalars(
    renderer: nn.Module,
    zoom_warp: RadialZoomWarp,
    masks: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: nn.Module,
    device: torch.device,
    steps: int = 500,
    lr: float = 0.01,
    eval_roundtrip: bool = True,
    batch_size: int = 8,
    verbose: bool = True,
) -> RadialZoomWarp:
    """Optimise zoom scalars via gradient descent through PoseNet.

    Only the ``zoom_scalars`` parameter (600 scalars) is optimised.
    The renderer weights and PoseNet weights are frozen.  This is the
    pose-space TTO reduced to its information-theoretic minimum.

    Parameters
    ----------
    renderer : nn.Module
        The :class:`AsymmetricPairGenerator` (or any pair generator whose
        ``forward()`` accepts ``ego_flow``).  Weights are frozen during
        optimisation.
    zoom_warp : RadialZoomWarp
        Module whose ``zoom_scalars`` will be optimised in-place.
    masks : torch.Tensor
        ``(1200, H, W)`` int/long tensor of segmentation masks.
    gt_frames : list[torch.Tensor]
        List of 1200 ``(H, W, 3)`` uint8 ground-truth frames.
    posenet : nn.Module
        Frozen PoseNet scorer with differentiable ``preprocess_input``.
    device : torch.device
        Compute device.
    steps : int
        Number of optimisation steps.
    lr : float
        Learning rate for Adam.
    eval_roundtrip : bool
        If ``True``, apply the contest roundtrip (384 -> 874 -> uint8 -> 384)
        before scoring, for auth-faithful optimisation.
    batch_size : int
        Number of pairs per gradient step.
    verbose : bool
        Print progress every 50 steps.

    Returns
    -------
    RadialZoomWarp
        The same module, with optimised ``zoom_scalars``.
    """
    import torch.nn.functional as F

    from tac.camera import CAMERA_H, CAMERA_W, SEGNET_INPUT_H, SEGNET_INPUT_W
    from tac.renderer import simulate_eval_roundtrip

    zoom_warp = zoom_warp.to(device)
    renderer = renderer.to(device).eval()
    posenet = posenet.to(device).eval()

    # Verify PoseNet is patched for differentiable gradients.
    # The upstream rgb_to_yuv6 has @torch.no_grad which kills ALL gradients.
    # This is the exact "GREAT GRADIENT BUG" from the project history.
    _test_input = torch.randn(1, 2, 3, 8, 8, device=device, requires_grad=True)
    _test_prep = posenet.preprocess_input(_test_input)
    assert _test_prep.requires_grad, (
        "PoseNet preprocess_input kills gradients (upstream @torch.no_grad). "
        "Call load_differentiable_scorers() or patch_scorers_for_training() first."
    )
    del _test_input, _test_prep

    # Freeze everything except zoom scalars
    for p in renderer.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in zoom_warp.parameters():
        p.requires_grad_(True)

    optimiser = torch.optim.Adam(zoom_warp.parameters(), lr=lr)

    n_frames = masks.shape[0]
    n_pairs = n_frames // 2

    H, W = masks.shape[1], masks.shape[2]

    for step in range(steps):
        # Sample a random batch of pair indices
        perm = torch.randperm(n_pairs)[:batch_size]  # CPU for indexing
        pair_idx = perm.to(device)  # GPU for zoom_scalars lookup

        # Gather masks (CPU indexing, then move to device)
        mask_t = masks[2 * perm].to(device)
        mask_t1 = masks[2 * perm + 1].to(device)

        # Gather GT pairs: (B, 2, H, W, 3) float
        gt_list = []
        for k in perm.cpu().tolist():
            gt_list.append(
                torch.stack(
                    [gt_frames[2 * k].float(), gt_frames[2 * k + 1].float()],
                    dim=0,
                )
            )
        gt_pair = torch.stack(gt_list).to(device)  # (B, 2, H, W, 3)

        # Compute zoom flow
        ego_flow = zoom_warp(pair_idx, H=H, W=W)  # (B, 2, H, W)

        # Forward through renderer with ego_flow
        pred_pair = renderer(mask_t, mask_t1, ego_flow=ego_flow)  # (B, 2, H, W, 3)

        # Convert to CHW for PoseNet
        pred_chw = pred_pair.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, 3, H, W)
        gt_chw = gt_pair.permute(0, 1, 4, 2, 3).contiguous()

        # Resize to scorer input if needed
        B_batch = pred_chw.shape[0]
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            pred_flat = pred_chw.reshape(B_batch * 2, 3, H, W)
            gt_flat = gt_chw.reshape(B_batch * 2, 3, H, W)
            pred_flat = F.interpolate(
                pred_flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear",
                align_corners=False,
            )
            gt_flat = F.interpolate(
                gt_flat,
                size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                mode="bilinear",
                align_corners=False,
            )
            pred_chw = pred_flat.reshape(B_batch, 2, 3, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B_batch, 2, 3, SEGNET_INPUT_H, SEGNET_INPUT_W)

        # Round to simulate uint8
        pred_chw = pred_chw.clamp(0, 255)

        # Eval roundtrip
        if eval_roundtrip:
            pred_flat = pred_chw.reshape(-1, 3, pred_chw.shape[-2], pred_chw.shape[-1])
            pred_flat = simulate_eval_roundtrip(pred_flat, target_h=CAMERA_H, target_w=CAMERA_W)
            pred_chw = pred_flat.reshape(B_batch, 2, 3, pred_flat.shape[-2], pred_flat.shape[-1])

        # PoseNet loss: MSE on first 6 pose dims
        pred_pose = posenet.preprocess_input(pred_chw)
        gt_pose = posenet.preprocess_input(gt_chw)

        with torch.no_grad():
            gt_out_raw = posenet(gt_pose)
            gt_pose_6 = (gt_out_raw["pose"] if isinstance(gt_out_raw, dict) else gt_out_raw)[..., :6]
        pred_out_raw = posenet(pred_pose)
        pred_pose_6 = (pred_out_raw["pose"] if isinstance(pred_out_raw, dict) else pred_out_raw)[..., :6]

        # MSE on first 6 pose dimensions (the scored ones)
        loss = F.mse_loss(pred_pose_6, gt_pose_6.detach())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if verbose and (step % 50 == 0 or step == steps - 1):
            s_vals = zoom_warp.zoom_scalars.detach()
            s_bounded = torch.tanh(s_vals) * zoom_warp.max_zoom_log
            print(
                f"[zoom-tto] step {step:4d}/{steps}  "
                f"loss={loss.item():.6f}  "
                f"|s|_mean={s_bounded.abs().mean().item():.5f}  "
                f"|s|_max={s_bounded.abs().max().item():.5f}"
            )

    return zoom_warp
