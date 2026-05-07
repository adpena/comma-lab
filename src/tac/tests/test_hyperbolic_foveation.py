from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from torch.autograd import gradcheck


def test_forward_shape_preserved() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((64, 64), n_frames=1)
    x = torch.randn(4, 3, 64, 64)

    y = hf(x)

    assert y.shape == x.shape


def test_forward_at_origin_is_identity() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((64, 64), n_frames=1, init_alpha=0.0, init_R=32.0, init_o=(32.0, 32.0))
    x = torch.randn(2, 3, 64, 64)

    y = hf(x)

    assert torch.allclose(y, x, atol=1e-5)


def test_inverse_roundtrip_within_tolerance() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((64, 64), n_frames=1)
    x = torch.randn(4, 3, 64, 64)

    x_rt = hf.inverse(hf(x))

    # float32 bilinear grid_sample roundtrip tops out at ~1e-5 over 64x64
    # even when Newton-Raphson converges to <1e-6 in coord space.
    assert (x_rt - x).abs().max().item() < 5e-5


def test_inverse_converges_in_max_iter() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((64, 64), n_frames=1, init_alpha=0.25, init_R=40.0, init_p=2.0, init_o=(32.0, 32.0))
    x = torch.randn(1, 3, 64, 64)

    stats = hf.gradient_check_at(x)

    assert stats["converged_in"] < 50


def test_forward_is_differentiable() -> None:
    from tac.hyperbolic_foveation import functional_hyperbolic_foveation

    # Use a smooth (low-frequency) input + non-integer origin so finite
    # differences don't straddle bilinear grid_sample's C^0 pixel-boundary
    # discontinuities. Otherwise gradcheck of the origin param sees noise
    # that the analytical gradient (which is exact within each cell) misses.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 8, dtype=torch.double),
        torch.linspace(-1, 1, 8, dtype=torch.double),
        indexing="ij",
    )
    base = torch.sin(xx * 1.3) * torch.cos(yy * 1.7)
    base = base.view(1, 1, 8, 8)
    alpha = torch.tensor([0.2], dtype=torch.double, requires_grad=True)
    radius = torch.tensor([12.0], dtype=torch.double, requires_grad=True)
    power = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    origin = torch.tensor([[3.5, 3.5]], dtype=torch.double, requires_grad=True)

    def fn(a: torch.Tensor, r: torch.Tensor, p: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return functional_hyperbolic_foveation(base, (8, 8), a, r, p, o)

    assert gradcheck(fn, (alpha, radius, power, origin), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_inverse_is_differentiable() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((16, 16), n_frames=1, init_alpha=0.1, init_R=20.0, init_p=2.0, init_o=(8.0, 8.0))
    x = torch.randn(1, 1, 16, 16, requires_grad=True)

    y = hf.inverse(hf(x), max_iter=20).sum()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert hf.alpha.grad is not None
    assert torch.isfinite(hf.alpha.grad).all()


def test_alpha_zero_is_identity() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((32, 32), n_frames=1, init_alpha=0.0, init_R=64.0, init_p=2.0, init_o=(16.0, 16.0))
    x = torch.randn(2, 3, 32, 32)

    assert torch.allclose(hf(x), x, atol=1e-5)


def test_p_zero_uniform_foveation() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((32, 32), n_frames=1, init_alpha=0.2, init_R=4.0, init_p=0.0, init_o=(16.0, 16.0))
    x = torch.randn(1, 1, 32, 32)
    coords = hf.coordinate_map(x, frame_indices=torch.tensor([0]))
    grid_x, grid_y = hf.identity_coordinates(dtype=coords.dtype, device=coords.device)
    identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    assert not torch.allclose(coords, identity)
    assert (coords - identity).abs().amax().item() > 0.01


def test_lane_mn_failure_avoided() -> None:
    from tac.hyperbolic_foveation import HyperbolicFoveation

    class Recorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            self.seen = z.detach().clone()
            return z * 2.0

    inp = torch.randn(2, 3, 16, 16)
    renderer_no_hf = Recorder()
    renderer_hf = Recorder()
    hf = HyperbolicFoveation((16, 16), n_frames=1, init_alpha=0.0)

    out_no_hf = renderer_no_hf(inp)
    rendered = renderer_hf(inp)
    out_hf = hf(rendered)

    assert torch.equal(renderer_hf.seen, renderer_no_hf.seen)
    assert torch.equal(renderer_hf.seen, inp)
    assert torch.allclose(out_hf, out_no_hf, atol=1e-5)


def test_archive_registration() -> None:
    from tac import submission_archive
    from tac.preflight import _ARCHIVE_ARTIFACT_FILENAMES

    manifest = submission_archive.ArchiveManifest(foveation_params_bin=True)

    assert "foveation_params.bin" in manifest.required_files()
    assert "foveation_params.bin" in _ARCHIVE_ARTIFACT_FILENAMES


def test_newton_raphson_coord_invert_100_random_points() -> None:
    """Stronger N-R inversion check: round-trip 100 random coords through
    Φ and Φ⁻¹ in coordinate space (no grid_sample bilinear floor)."""
    from tac.hyperbolic_foveation import (
        _identity_coordinates,
        _inverse_coordinates,
        _map_coordinates,
    )

    H, W = 384, 512
    torch.manual_seed(0)
    # 100 random interior points (avoid the exact corners; o-at-corner is a
    # separate edge-case test below).
    coords = torch.stack(
        [
            torch.rand(100, dtype=torch.float64) * (W - 4) + 2,
            torch.rand(100, dtype=torch.float64) * (H - 4) + 2,
        ],
        dim=-1,
    ).view(1, 100, 1, 2)
    alpha = torch.tensor([[[0.3]]], dtype=torch.float64)
    R = torch.tensor([[[120.0]]], dtype=torch.float64)
    p = torch.tensor([[[2.0]]], dtype=torch.float64)
    origin = torch.tensor([[[[W * 0.5, H * 0.5]]]], dtype=torch.float64)

    warped = _map_coordinates(coords, alpha, R, p, origin)
    inv, n_iter = _inverse_coordinates(warped, alpha, R, p, origin, max_iter=20, tol=1e-10)
    err = (inv - coords).abs().max().item()

    assert err < 1e-4, f"N-R coord roundtrip err={err:.2e} (n_iter={n_iter})"
    assert n_iter <= 20


def test_per_frame_param_packing_bit_exact_roundtrip(tmp_path) -> None:
    """save_foveation_params / load_foveation_params must be bit-exact for
    fp32 storage (the on-disk format is float32)."""
    from tac.hyperbolic_foveation import (
        HyperbolicFoveation,
        load_foveation_params,
        save_foveation_params,
    )

    torch.manual_seed(7)
    n_frames = 1200
    hf = HyperbolicFoveation((384, 512), n_frames=n_frames)
    with torch.no_grad():
        hf.alpha.copy_(torch.randn(n_frames) * 0.5)
        hf.R.copy_(torch.rand(n_frames) * 100 + 20)
        hf.p.copy_(torch.rand(n_frames) * 3 + 0.5)
        hf.o.copy_(torch.rand(n_frames, 2) * torch.tensor([512.0, 384.0]))

    path = tmp_path / "foveation_params.bin"
    n_bytes = save_foveation_params(hf, path)
    # 16-byte header + 5 floats * 4 bytes * n_frames
    assert n_bytes == 16 + 5 * 4 * n_frames
    # 1200 frames * 20 bytes/frame + 16 header = 24016 bytes (~24KB)
    assert n_bytes == 24_016

    hf2 = load_foveation_params(path)

    # bit-exact: writing fp32 to disk and reading fp32 back should be
    # identical to the fp32 view of the original parameters.
    assert torch.equal(hf.alpha.detach().float(), hf2.alpha.detach())
    assert torch.equal(hf.R.detach().float(), hf2.R.detach())
    assert torch.equal(hf.p.detach().float(), hf2.p.detach())
    assert torch.equal(hf.o.detach().float(), hf2.o.detach())
    assert hf2.image_size == (384, 512)
    assert hf2.n_frames == n_frames


def test_hfv1_header_image_size_cannot_be_overridden(tmp_path) -> None:
    from tac.hyperbolic_foveation import (
        HyperbolicFoveation,
        load_foveation_params,
        save_foveation_params,
    )

    hf = HyperbolicFoveation((384, 512), n_frames=2)
    path = tmp_path / "foveation_params.bin"
    save_foveation_params(hf, path)

    with pytest.raises(ValueError, match="header image_size"):
        load_foveation_params(path, image_size=(64, 96))


def test_origin_at_corner_does_not_nan() -> None:
    """Origin at (0,0) and (W-1,H-1) corners must not produce NaN/Inf in
    forward, inverse, or gradients."""
    from tac.hyperbolic_foveation import HyperbolicFoveation

    for ox, oy in [(0.0, 0.0), (31.0, 31.0), (0.0, 31.0), (31.0, 0.0)]:
        hf = HyperbolicFoveation(
            (32, 32),
            n_frames=1,
            init_alpha=0.4,
            init_R=20.0,
            init_p=2.0,
            init_o=(ox, oy),
        )
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        y = hf(x)
        assert torch.isfinite(y).all(), f"forward NaN at o=({ox},{oy})"
        x_rt = hf.inverse(y, max_iter=50, tol=1e-5)
        assert torch.isfinite(x_rt).all(), f"inverse NaN at o=({ox},{oy})"
        x_rt.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), (
            f"grad NaN at o=({ox},{oy})"
        )
        assert hf.alpha.grad is None or torch.isfinite(hf.alpha.grad).all()


def test_invertibility_penalty_is_finite_and_nonneg() -> None:
    """Penalty exists for the loss-balance machinery; sanity-check it."""
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((64, 64), n_frames=4, init_alpha=0.2, init_R=30.0)
    pen = hf.invertibility_penalty(samples=64)
    assert torch.isfinite(pen).all()
    assert float(pen.item()) >= 0.0


def test_per_frame_indices_select_correct_params() -> None:
    """frame_indices must route per-frame parameters through grid_sample."""
    from tac.hyperbolic_foveation import HyperbolicFoveation

    hf = HyperbolicFoveation((32, 32), n_frames=2, init_alpha=0.0, init_o=(16.0, 16.0))
    # Frame 0 stays at α=0 (identity); frame 1 gets α=0.5 (warps).
    with torch.no_grad():
        hf.alpha[0] = 0.0
        hf.alpha[1] = 0.5
        hf.R[1] = 40.0
        hf.p[1] = 2.0

    x = torch.randn(2, 3, 32, 32)
    y = hf(x, frame_indices=torch.tensor([0, 1]))

    # batch[0] used frame 0 → identity → equals input.
    assert torch.allclose(y[0], x[0], atol=1e-5)
    # batch[1] used frame 1 → warped → must differ.
    assert not torch.allclose(y[1], x[1], atol=1e-3)


def test_load_foveation_params_rejects_corrupt_payload(tmp_path) -> None:
    from tac.hyperbolic_foveation import load_foveation_params

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"HFV1" + b"\x00" * 12 + b"\x00" * 7)  # ragged tail
    import pytest

    with pytest.raises(ValueError, match="payload"):
        load_foveation_params(bad)
