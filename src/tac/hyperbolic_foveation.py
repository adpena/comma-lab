# SPDX-License-Identifier: MIT
"""Invertible Telescope-style hyperbolic foveation.

The operator is a radial, origin-centered geometry wrap intended to live after
the renderer.  The renderer still receives the ordinary mask/pose distribution;
only the rendered RGB field is warped.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6
_FOVEATION_MAGIC = b"HFV1"
_FOVEATION_HEADER = struct.Struct("<4sIII")

__all__ = [
    "EPSILON",
    "HyperbolicFoveation",
    "functional_hyperbolic_foveation",
    "load_foveation_params",
    "save_foveation_params",
]


def _as_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError(f"image_size must be (H, W), got {image_size!r}")
    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"image_size entries must be positive, got {image_size!r}")
    return h, w


def _coerce_origin(init_o: tuple[float, float] | None) -> tuple[float, float]:
    if init_o is None:
        return (256.0, 174.0)
    if len(init_o) != 2:
        raise ValueError(f"init_o must be (ox, oy), got {init_o!r}")
    return (float(init_o[0]), float(init_o[1]))


def _identity_coordinates(
    image_size: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_size
    yy = torch.arange(h, dtype=dtype, device=device)
    xx = torch.arange(w, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return grid_x, grid_y


def _normalise_grid(coords: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    x = coords[..., 0]
    y = coords[..., 1]
    if w <= 1:
        gx = torch.zeros_like(x)
    else:
        gx = 2.0 * x / float(w - 1) - 1.0
    if h <= 1:
        gy = torch.zeros_like(y)
    else:
        gy = 2.0 * y / float(h - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _frame_indices(
    batch: int,
    n_frames: int,
    *,
    frame_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_frames == 1:
        return torch.zeros(batch, dtype=torch.long, device=device)
    if frame_indices is None:
        return (torch.arange(batch, dtype=torch.long, device=device) % n_frames)
    idx = frame_indices.to(device=device, dtype=torch.long).flatten()
    if idx.numel() != batch:
        raise ValueError(
            f"frame_indices must have {batch} entries for batch size {batch}, "
            f"got shape {tuple(frame_indices.shape)}"
        )
    if torch.any(idx < 0) or torch.any(idx >= n_frames):
        raise IndexError(
            f"frame_indices out of range for n_frames={n_frames}: "
            f"min={int(idx.min().item())}, max={int(idx.max().item())}"
        )
    return idx


def _select_params(
    alpha: torch.Tensor,
    radius: torch.Tensor,
    power: torch.Tensor,
    origin: torch.Tensor,
    *,
    batch: int,
    frame_indices: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_frames = int(alpha.shape[0])
    idx = _frame_indices(
        batch,
        n_frames,
        frame_indices=frame_indices,
        device=device,
    )
    a = alpha.to(device=device)[idx].view(batch, 1, 1)
    r = radius.to(device=device)[idx].view(batch, 1, 1)
    p = power.to(device=device)[idx].view(batch, 1, 1)
    o = origin.to(device=device)[idx].view(batch, 1, 1, 2)
    return a, r, p, o


def _radial_q(alpha: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q(r)=tanh(alpha*r)/alpha and q'(r).

    This is the identity-preserving Telescope limit: as alpha -> 0, q(r) -> r.
    """
    a = alpha.abs()
    small = a <= EPSILON
    a_safe = torch.where(small, torch.ones_like(a), a)
    ar = a_safe * r
    tanh_ar = torch.tanh(ar)
    q_raw = tanh_ar / a_safe
    # Broadcast `small` to the result shape so where() works regardless of
    # whether `alpha` and `r` share a single non-trivial dim or broadcast
    # against each other (e.g. (n_frames,1) vs (1,samples)).
    small_b = small.expand_as(q_raw)
    r_b = r.expand_as(q_raw)
    q = torch.where(small_b, r_b, q_raw)
    q_prime_raw = 1.0 - tanh_ar.square()
    q_prime = torch.where(small_b, torch.ones_like(q_prime_raw), q_prime_raw)
    return q, q_prime


def _blend_weight(radius: torch.Tensor, R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    R_safe = R.clamp_min(EPSILON)
    base = (1.0 - (radius / R_safe).clamp(min=0.0, max=1.0)).clamp(min=0.0, max=1.0)
    return base.pow(p.clamp_min(0.0))


def _radius_map(
    source_radius: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor,
    p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    R_safe = R.clamp_min(EPSILON)
    p_safe = p.clamp_min(0.0)
    q, q_prime = _radial_q(alpha, source_radius)
    base = (
        1.0 - (source_radius / R_safe).clamp(min=0.0, max=1.0)
    ).clamp(min=0.0, max=1.0)
    w = base.pow(p_safe)
    rho = source_radius + w * (q - source_radius)

    inside = (source_radius > 0.0) & (source_radius < R_safe) & (p_safe > 0.0)
    base_for_deriv = base.clamp_min(EPSILON)
    dw = -p_safe * base_for_deriv.pow(p_safe - 1.0) / R_safe
    dw = torch.where(inside, dw, torch.zeros_like(dw))
    rho_prime = 1.0 + dw * (q - source_radius) + w * (q_prime - 1.0)
    return rho, rho_prime.clamp_min(EPSILON)


def _map_coordinates(
    coords: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor,
    p: torch.Tensor,
    origin: torch.Tensor,
) -> torch.Tensor:
    delta = coords - origin
    radius = torch.linalg.vector_norm(delta, dim=-1).clamp_min(0.0)
    q, _ = _radial_q(alpha, radius)
    scale = torch.where(
        radius > EPSILON,
        q / radius.clamp_min(EPSILON),
        torch.ones_like(radius),
    )
    hyperbolic = origin + scale.unsqueeze(-1) * delta
    w = _blend_weight(radius, R, p).unsqueeze(-1)
    return coords + w * (hyperbolic - coords)


def _inverse_coordinates(
    coords: torch.Tensor,
    alpha: torch.Tensor,
    R: torch.Tensor,
    p: torch.Tensor,
    origin: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int]:
    target_delta = coords - origin
    target_radius = torch.linalg.vector_norm(target_delta, dim=-1)
    source_radius = target_radius.clone()
    converged_in = max_iter

    for i in range(max_iter):
        rho, rho_prime = _radius_map(source_radius, alpha, R, p)
        step = (rho - target_radius) / rho_prime
        source_radius = (source_radius - step).clamp_min(0.0)
        max_step = float(step.detach().abs().max().item())
        if max_step < tol:
            converged_in = i + 1
            break

    scale = torch.where(
        target_radius > EPSILON,
        source_radius / target_radius.clamp_min(EPSILON),
        torch.zeros_like(source_radius),
    )
    inv = origin + scale.unsqueeze(-1) * target_delta
    inv = torch.where(
        (target_radius <= EPSILON).unsqueeze(-1),
        origin.expand_as(inv),
        inv,
    )
    return inv, converged_in


def functional_hyperbolic_foveation(
    x: torch.Tensor,
    image_size: tuple[int, int],
    alpha: torch.Tensor,
    radius: torch.Tensor,
    power: torch.Tensor,
    origin: torch.Tensor,
    *,
    frame_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional forward warp used by gradcheck and the module wrapper."""
    image_size = _as_image_size(image_size)
    if x.ndim != 4:
        raise ValueError(f"x must have shape (B, C, H, W), got {tuple(x.shape)}")
    batch, _channels, h, w = x.shape
    if (h, w) != image_size:
        raise ValueError(f"x spatial shape {(h, w)} does not match image_size={image_size}")

    a, R, p, o = _select_params(
        alpha,
        radius,
        power,
        origin,
        batch=batch,
        frame_indices=frame_indices,
        device=x.device,
    )
    grid_x, grid_y = _identity_coordinates(image_size, dtype=x.dtype, device=x.device)
    coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    mapped = _map_coordinates(coords, a, R, p, o)
    grid = _normalise_grid(mapped, image_size)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class HyperbolicFoveation(nn.Module):
    """Per-frame invertible hyperbolic foveation geometry.

    Parameters are stored directly as learnable tensors:
      * ``alpha``: contraction strength, shape ``(n_frames,)``
      * ``R``: blend radius in pixels, shape ``(n_frames,)``
      * ``p``: blend falloff power, shape ``(n_frames,)``
      * ``o``: FoE origin ``(ox, oy)``, shape ``(n_frames, 2)``
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        n_frames: int = 1,
        init_alpha: float = 1.0,
        init_R: float = 1.0,
        init_p: float = 2.0,
        init_o: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.image_size = _as_image_size(image_size)
        self.n_frames = int(n_frames)
        if self.n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")

        ox, oy = _coerce_origin(init_o)
        self.alpha = nn.Parameter(torch.full((self.n_frames,), float(init_alpha)))
        self.R = nn.Parameter(torch.full((self.n_frames,), float(init_R)))
        self.p = nn.Parameter(torch.full((self.n_frames,), float(init_p)))
        origin = torch.tensor([ox, oy], dtype=torch.float32).repeat(self.n_frames, 1)
        self.o = nn.Parameter(origin)
        self._last_inverse_iters = 0

    def identity_coordinates(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dtype is None:
            dtype = self.alpha.dtype
        if device is None:
            device = self.alpha.device
        return _identity_coordinates(self.image_size, dtype=dtype, device=device)

    def _selected_params(
        self,
        batch: int,
        *,
        frame_indices: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _select_params(
            self.alpha,
            self.R,
            self.p,
            self.o,
            batch=batch,
            frame_indices=frame_indices,
            device=device,
        )

    def coordinate_map(
        self,
        x: torch.Tensor,
        *,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, C, H, W), got {tuple(x.shape)}")
        batch, _channels, h, w = x.shape
        if (h, w) != self.image_size:
            raise ValueError(
                f"x spatial shape {(h, w)} does not match foveation image_size={self.image_size}"
            )
        a, R, p, o = self._selected_params(
            batch,
            frame_indices=frame_indices,
            device=x.device,
        )
        grid_x, grid_y = _identity_coordinates(self.image_size, dtype=x.dtype, device=x.device)
        coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        return _map_coordinates(coords, a, R, p, o)

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return functional_hyperbolic_foveation(
            x,
            self.image_size,
            self.alpha,
            self.R,
            self.p,
            self.o,
            frame_indices=frame_indices,
        )

    def inverse(
        self,
        y: torch.Tensor,
        max_iter: int = 50,
        tol: float = 1e-5,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"y must have shape (B, C, H, W), got {tuple(y.shape)}")
        batch, _channels, h, w = y.shape
        if (h, w) != self.image_size:
            raise ValueError(
                f"y spatial shape {(h, w)} does not match foveation image_size={self.image_size}"
            )
        a, R, p, o = self._selected_params(
            batch,
            frame_indices=frame_indices,
            device=y.device,
        )
        grid_x, grid_y = _identity_coordinates(self.image_size, dtype=y.dtype, device=y.device)
        coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        inv_coords, converged_in = _inverse_coordinates(
            coords,
            a,
            R,
            p,
            o,
            max_iter=max_iter,
            tol=tol,
        )
        self._last_inverse_iters = converged_in
        grid = _normalise_grid(inv_coords, self.image_size)
        return F.grid_sample(
            y,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def invertibility_penalty(
        self,
        *,
        samples: int = 128,
        min_derivative: float = 1e-4,
    ) -> torch.Tensor:
        """Penalise parameter regions where the radial map can fold."""
        device = self.alpha.device
        dtype = self.alpha.dtype
        h, w = self.image_size
        max_radius = float((h * h + w * w) ** 0.5)
        radii = torch.linspace(0.0, max_radius, samples, dtype=dtype, device=device)
        radii = radii.view(1, samples)
        a = self.alpha.view(self.n_frames, 1)
        R = self.R.view(self.n_frames, 1)
        p = self.p.view(self.n_frames, 1)
        _rho, rho_prime = _radius_map(radii, a, R, p)
        fold_penalty = F.relu(min_derivative - rho_prime).square().mean()
        param_penalty = (
            F.relu(EPSILON - self.R).square().mean()
            + F.relu(-self.p).square().mean()
        )
        return fold_penalty + param_penalty

    def gradient_check_at(self, x: torch.Tensor) -> dict:
        y = self.forward(x)
        x_rt = self.inverse(y)
        return {
            "max_roundtrip_error": float((x_rt - x).detach().abs().max().item()),
            "converged_in": int(self._last_inverse_iters),
        }

    def resized(self, image_size: tuple[int, int]) -> "HyperbolicFoveation":
        """Return a geometry-equivalent copy for a different image size."""
        new_h, new_w = _as_image_size(image_size)
        old_h, old_w = self.image_size
        sx = float(new_w) / float(old_w)
        sy = float(new_h) / float(old_h)
        sr = 0.5 * (sx + sy)
        out = HyperbolicFoveation(
            (new_h, new_w),
            n_frames=self.n_frames,
            init_alpha=1.0,
            init_R=1.0,
            init_p=2.0,
            init_o=(0.0, 0.0),
        ).to(device=self.alpha.device, dtype=self.alpha.dtype)
        with torch.no_grad():
            out.alpha.copy_(self.alpha / max(sr, EPSILON))
            out.R.copy_(self.R * sr)
            out.p.copy_(self.p)
            out.o[:, 0].copy_(self.o[:, 0] * sx)
            out.o[:, 1].copy_(self.o[:, 1] * sy)
        return out


def _params_matrix(
    module: HyperbolicFoveation,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    def get(name: str, fallback: torch.Tensor) -> torch.Tensor:
        if state_dict is None:
            return fallback.detach()
        return state_dict.get(name, state_dict.get(f"hyperbolic_foveation.{name}", fallback)).detach()

    alpha = get("alpha", module.alpha).detach().cpu().float()
    radius = get("R", module.R).detach().cpu().float()
    power = get("p", module.p).detach().cpu().float()
    origin = get("o", module.o).detach().cpu().float()
    return torch.stack([alpha, radius, power, origin[:, 0], origin[:, 1]], dim=1)


def save_foveation_params(
    module: HyperbolicFoveation,
    path: Path | str,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> int:
    """Save pure geometry parameters to ``foveation_params.bin``.

    Format: ``HFV1`` magic, ``uint32 n_frames``, ``uint32 H``, ``uint32 W``,
    followed by float32 rows ``alpha, R, p, ox, oy``.
    """
    path = Path(path)
    matrix = _params_matrix(module, state_dict=state_dict).contiguous()
    header = _FOVEATION_HEADER.pack(
        _FOVEATION_MAGIC,
        int(matrix.shape[0]),
        int(module.image_size[0]),
        int(module.image_size[1]),
    )
    blob = header + matrix.numpy().astype("<f4", copy=False).tobytes()
    path.write_bytes(blob)
    return len(blob)


def load_foveation_params(
    path: Path | str,
    *,
    image_size: tuple[int, int] | None = None,
    device: torch.device | str | None = None,
) -> HyperbolicFoveation:
    """Load ``foveation_params.bin`` without touching scorer weights."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= _FOVEATION_HEADER.size and raw[:4] == _FOVEATION_MAGIC:
        magic, n_frames, h, w = _FOVEATION_HEADER.unpack(raw[: _FOVEATION_HEADER.size])
        if magic != _FOVEATION_MAGIC:
            raise ValueError(f"bad foveation magic in {path}: {magic!r}")
        body = raw[_FOVEATION_HEADER.size :]
        expected = int(n_frames) * 5 * 4
        if len(body) != expected:
            raise ValueError(
                f"bad foveation payload size in {path}: got {len(body)} bytes, "
                f"expected {expected}"
            )
        header_size = (int(h), int(w))
        if image_size is not None and _as_image_size(image_size) != header_size:
            raise ValueError(
                f"foveation header image_size {header_size} does not match "
                f"expected {tuple(_as_image_size(image_size))}"
            )
        size = header_size
    else:
        if len(raw) % (5 * 4) != 0:
            raise ValueError(
                f"legacy foveation payload must be a multiple of 20 bytes, got {len(raw)}"
            )
        n_frames = len(raw) // (5 * 4)
        body = raw
        size = _as_image_size(image_size or (384, 512))

    values = torch.frombuffer(bytearray(body), dtype=torch.float32).clone().view(int(n_frames), 5)
    module = HyperbolicFoveation(size, n_frames=int(n_frames))
    with torch.no_grad():
        module.alpha.copy_(values[:, 0])
        module.R.copy_(values[:, 1])
        module.p.copy_(values[:, 2])
        module.o[:, 0].copy_(values[:, 3])
        module.o[:, 1].copy_(values[:, 4])
    if device is not None:
        module = module.to(device)
    return module
