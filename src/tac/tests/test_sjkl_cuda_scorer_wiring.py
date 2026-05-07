"""Track-B Omega-1 CUDA scorer wiring smoke tests.

Exercises the direct-pixel SegNet + PoseNet differentiable score_fn integration in
``experiments/build_sjkl_residual.py`` from a CPU-only test environment by
injecting stub modules through the ``_SCORE_FN_FACTORY_OVERRIDE`` hook.

Why a CPU stub path: the real CUDA scorer Fisher proxy needs the contest
SegNet + PoseNet checkpoints (~90 MB) on a CUDA device. The override-hook
approach replaces just the score_fn factory while exercising every other code
path: BuildConfig parsing, anchor reshape, Lanczos basis solve via the new
``score_fn``, alpha-block sparse encode, brotli compression, sjkl.bin
roundtrip, magic-byte / manifest correctness. The current direct-pixel contest
score formula ``100 * seg_dist + sqrt(10 * pose_dist + eps)`` is unit-tested
separately by direct invocation of ``_build_cuda_jfg_contest_score_fn`` on
nano stub modules.

Strict-scorer-rule note: this test file does NOT load a renderer at inflate
time. The build_sjkl_residual.py path under test is a COMPRESS-time path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import brotli  # noqa: F401  (decode_full_sjkl_payload pulls it via runtime)
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = REPO_ROOT / "experiments" / "build_sjkl_residual.py"
spec = importlib.util.spec_from_file_location(
    "build_sjkl_residual_for_wiring_test", _BUILDER_PATH
)
builder = importlib.util.module_from_spec(spec)
sys.modules["build_sjkl_residual_for_wiring_test"] = builder
spec.loader.exec_module(builder)

from tac.sjkl_basis import decode_full_sjkl_payload  # noqa: E402

# ---------------------------------------------------------------------------
# Stub modules: tiny stand-ins for JFG / SegNet / PoseNet that are CPU-cheap
# and produce well-defined gradients for the Lanczos HVP path.
# ---------------------------------------------------------------------------


class _StubSegNet(torch.nn.Module):
    """Stub SegNet: returns (B, K, h, w) class-logit volume.

    Linear projection from a global pooled frame to per-class logits, then
    broadcast to the full spatial extent. Output shape matches the real
    SegNet's argmax-over-classes contract.
    """

    def __init__(self, num_classes: int = 5, h: int = 4, w: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.h = h
        self.w = w
        self.proj = torch.nn.Linear(3, num_classes, bias=True)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # Real SegNet uses last frame: x[:, -1, ...]. Keep the contract.
        return x[:, -1, ...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=3, H, W). Pool to (B, 3) and project to (B, K).
        pooled = x.mean(dim=(-1, -2))  # (B, 3)
        logits = self.proj(pooled)  # (B, K)
        # Broadcast to (B, K, h, w)
        return logits.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.h, self.w)


class _StubPoseNet(torch.nn.Module):
    """Stub PoseNet: returns dict with `pose` key, shape (B, 12)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 12, bias=True)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # Real PoseNet expects (B, T, C, H, W) and returns (B, T*6, h, w).
        # Stub flattens to (B, 3) for projection - this is a SHAPE-FAITHFUL
        # stand-in, not a numerical replica.
        return x.mean(dim=(-1, -2)).mean(dim=1)  # (B, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.proj(x)}


class _StubJFG(torch.nn.Module):
    """Stub JointFrameGenerator: identity-passthrough of the rendered frame.

    Real JFG takes (mask2, pose6) -> (frame1, frame2). The stub here is used
    only to confirm the wiring path exercises the JFG forward call without
    requiring an 88K-param checkpoint.
    """

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, mask2: torch.Tensor, pose6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return two copies; ignore inputs but stay in the graph for HVP.
        b = mask2.shape[0]
        h, w = 4, 4
        f1 = torch.zeros(b, 3, h, w, dtype=mask2.dtype, device=mask2.device) + self.bias
        f2 = torch.zeros(b, 3, h, w, dtype=mask2.dtype, device=mask2.device) + self.bias
        return f1, f2


def _build_stub_score_fn(anchor_frame: torch.Tensor, cfg: object) -> tuple[object, dict]:
    """Factory hook installed via _SCORE_FN_FACTORY_OVERRIDE for tests.

    Builds the direct-pixel contest score_fn against stub SegNet+PoseNet in the
    CPU test environment. The actual ``_build_cuda_jfg_contest_score_fn``
    helper from the builder module is exercised end-to-end here.

    The CPU stub-mode anchor is a flat (D,) tensor where D may not factor as
    3*H*W for an integer (H, W). We pick a (H, W) such that 3*H*W == D and
    use those for the score_fn's frame reshape. For D=256 (the default stub
    dim), this is impossible, so the score_fn instead pools/reshapes via a
    dim-flexible projection.
    """
    flat_dim = int(anchor_frame.numel())
    # Find (H, W) so 3*H*W == flat_dim. For arbitrary D this may fail; pick
    # the closest feasible factorisation by zero-padding internally.
    # For test simplicity, use H=W=int(sqrt(D/3)); the score_fn will treat
    # flat as a (3, H, W) tensor of size 3*H*W.
    import math
    side = max(1, int(math.sqrt(flat_dim / 3.0)))
    target_h = target_w = side
    feasible_dim = 3 * target_h * target_w

    segnet = _StubSegNet(num_classes=5, h=target_h, w=target_w)
    posenet = _StubPoseNet()
    jfg = _StubJFG()

    # Build score_fn against the feasible dim using a dim-adapting wrapper.
    # We use a fresh, equally-sized anchor so the inner _pair_from_flat call
    # accepts (3, H, W).
    if feasible_dim == flat_dim:
        anchor_for_factory = anchor_frame
        inner_score_fn, meta = builder._build_cuda_jfg_contest_score_fn(
            anchor_for_factory,
            jfg,
            posenet,
            segnet,
            target_h=target_h,
            target_w=target_w,
        )
        score_fn = inner_score_fn
    else:
        anchor_padded = anchor_frame.reshape(-1)[:feasible_dim].clone()
        if anchor_padded.numel() < feasible_dim:
            pad = torch.zeros(feasible_dim - anchor_padded.numel(), dtype=anchor_padded.dtype)
            anchor_padded = torch.cat([anchor_padded, pad], dim=0)
        inner_score_fn, meta = builder._build_cuda_jfg_contest_score_fn(
            anchor_padded,
            jfg,
            posenet,
            segnet,
            target_h=target_h,
            target_w=target_w,
        )

        # Wrap to accept the original flat_dim shape from Lanczos.
        def score_fn(flat: torch.Tensor) -> torch.Tensor:
            slc = flat.reshape(-1)[:feasible_dim]
            return inner_score_fn(slc)

    meta["test_override"] = "stub_modules_cpu"
    meta["stub_target_h"] = int(target_h)
    meta["stub_target_w"] = int(target_w)
    return score_fn, meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_renderer_checkpoint_field_exists():
    """BuildConfig must expose `renderer_checkpoint` for the new CLI flag."""
    fields = builder.BuildConfig.__dataclass_fields__
    assert "renderer_checkpoint" in fields, (
        "BuildConfig must include `renderer_checkpoint` (Path | None) for "
        "Track-B Omega-1 JFG-aware Fisher basis builds."
    )
    # default is None - the field is opt-in
    cfg = builder.BuildConfig(
        pair_tensor_manifest=Path("/dev/null"),
        output_dir=Path("/dev/null"),
        device="cpu",
        rank=4,
        n_pairs=4,
        alpha_bits=4,
        basis_quant_bits=6,
        max_bytes=32768,
        allow_cpu_stub=True,
        seed=0,
    )
    assert cfg.renderer_checkpoint is None


def test_cli_flag_parses_renderer_checkpoint(tmp_path):
    parser = builder.build_parser()
    args = parser.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--device", "cpu",
            "--allow-cpu-stub",
            "--renderer-checkpoint", "/some/path/q_faithful.pt",
        ]
    )
    assert args.renderer_checkpoint == Path("/some/path/q_faithful.pt")


def test_cli_flag_default_is_none(tmp_path):
    parser = builder.build_parser()
    args = parser.parse_args(
        [
            "--output-dir", str(tmp_path),
            "--device", "cpu",
            "--allow-cpu-stub",
        ]
    )
    assert args.renderer_checkpoint is None


def test_jfg_contest_score_fn_returns_scalar_and_is_finite():
    """The contest-formula score_fn must return a finite scalar."""
    target_h, target_w = 4, 4
    dim = 3 * target_h * target_w
    anchor = torch.full((dim,), 128.0, dtype=torch.float32)  # mid-grey

    segnet = _StubSegNet(num_classes=5, h=target_h, w=target_w)
    posenet = _StubPoseNet()
    jfg = _StubJFG()

    score_fn, meta = builder._build_cuda_jfg_contest_score_fn(
        anchor, jfg, posenet, segnet, target_h=target_h, target_w=target_w
    )
    # Run on a NEW frame (slightly perturbed)
    perturbed = anchor + 0.5 * torch.randn(dim)
    s = score_fn(perturbed)
    assert s.dim() == 0, f"score_fn must return a scalar, got shape {s.shape}"
    assert torch.isfinite(s).item()
    # contest formula bounded below by 0 (both terms non-negative)
    assert float(s) >= -1e-6
    # at the exact anchor, soft seg_dist > 0 (softmax never reaches 1) but
    # pose_mse == 0 so the score should be small. Just check finiteness.
    s_anchor = score_fn(anchor)
    assert torch.isfinite(s_anchor).item()
    # meta carries provenance tags
    assert meta["scorer_fisher_mode"] == "cuda_direct_pixel_contest_score_proxy"
    assert meta["scorer_fisher_jfg_in_loop"] is False
    assert meta["scorer_fisher_jfg_argument_used"] is False
    assert "100 * seg_dist + sqrt(10 * pose_dist + 1e-12)" in meta["score_formula"]
    assert meta["scorer_fisher_score_claim"] is False


def test_jfg_contest_score_fn_is_differentiable():
    """The Lanczos HVP path needs torch.autograd.grad through score_fn."""
    target_h, target_w = 4, 4
    dim = 3 * target_h * target_w
    anchor = torch.full((dim,), 100.0, dtype=torch.float32)

    score_fn, _ = builder._build_cuda_jfg_contest_score_fn(
        anchor,
        _StubJFG(),
        _StubPoseNet(),
        _StubSegNet(num_classes=5, h=target_h, w=target_w),
        target_h=target_h,
        target_w=target_w,
    )

    f = anchor.detach().clone().requires_grad_(True)
    s = score_fn(f)
    (g,) = torch.autograd.grad(s, f, create_graph=True)
    assert g.shape == f.shape
    # The HVP needs to support double-backward
    v = torch.randn_like(f)
    gv = (g * v).sum()
    (Hv,) = torch.autograd.grad(gv, f)
    assert Hv.shape == f.shape
    assert torch.isfinite(Hv).all().item()


def test_factory_override_hook_is_module_level():
    """The override hook is the canonical test-injection point."""
    assert hasattr(builder, "_SCORE_FN_FACTORY_OVERRIDE")
    assert builder._SCORE_FN_FACTORY_OVERRIDE is None  # default


def test_full_pipeline_with_stub_factory_override(tmp_path):
    """End-to-end: install stub factory, run build, verify sjkl.bin roundtrips.

    This is the canonical "wiring landed" smoke. Exercises:
      * BuildConfig with renderer_checkpoint=None (override path bypasses it)
      * _SCORE_FN_FACTORY_OVERRIDE injection
      * Lanczos basis solve through the contest-formula score_fn
      * Alpha-block sparse encode + brotli compress
      * Full sjkl.bin payload write + magic + roundtrip decode
    """
    out_dir = tmp_path / "wiring_smoke"
    cfg = builder.BuildConfig(
        pair_tensor_manifest=Path("/dev/null"),
        output_dir=out_dir,
        device="cpu",
        rank=4,
        n_pairs=8,
        alpha_bits=4,
        basis_quant_bits=6,
        max_bytes=32768,
        allow_cpu_stub=True,
        seed=7,
        renderer_checkpoint=None,  # override hook bypasses the JFG load path
    )

    builder._SCORE_FN_FACTORY_OVERRIDE = _build_stub_score_fn
    try:
        manifest = builder.build_sjkl_residual(cfg)
    finally:
        builder._SCORE_FN_FACTORY_OVERRIDE = None

    sjkl_path = out_dir / "sjkl.bin"
    assert sjkl_path.is_file()
    payload = sjkl_path.read_bytes()
    # Magic byte
    assert payload[:4] == b"SJKL", f"first 4 bytes must be SJKL, got {payload[:4]!r}"
    # Roundtrip
    basis, decoded_meta = decode_full_sjkl_payload(payload)
    assert basis.rank == 4
    assert basis.dim == 256
    # Manifest provenance reflects override
    assert manifest["score_claim"] is False
    assert manifest["sjkl_bin_bytes"] == sjkl_path.stat().st_size
    # The stub override emits provenance fields
    assert manifest.get("test_override") == "stub_modules_cpu"
    assert manifest.get("scorer_fisher_jfg_in_loop") is False


def test_renderer_checkpoint_missing_file_raises(tmp_path):
    """Passing --renderer-checkpoint to a non-existent file must fail loud
    rather than silently fall back to the proxy."""
    # Use device=cpu with allow_cpu_stub but DON'T set the override -
    # then the renderer_checkpoint code path triggers the file-existence
    # check. (CPU branch with override=None falls back to stub, but we
    # specifically want the existence check, so we use the helper directly.)
    nonexistent = tmp_path / "does_not_exist.pt"
    assert not nonexistent.exists()

    # The check is in the cuda branch but the helper itself is callable;
    # _load_jfg_from_checkpoint will fail on read_bytes for a missing file.
    with pytest.raises((FileNotFoundError, OSError)):
        builder._load_jfg_from_checkpoint(nonexistent, torch.device("cpu"))


def test_load_jfg_from_qfai_blob(tmp_path):
    """A QFAI-formatted state-dict blob must load via the qfai branch."""
    import json as _json
    import struct as _struct

    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    # Build a real (small) JFG, snapshot its state, and pack a QFAI blob.
    gen = build_quantizr_faithful_renderer(
        num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1
    )
    state = gen.state_dict()

    import io as _io
    body_buf = _io.BytesIO()
    torch.save(state, body_buf)
    body = body_buf.getvalue()

    header = _json.dumps({
        "num_classes": 5,
        "pose_dim": 6,
        "cond_dim": 48,
        "depth_mult": 1,
    }).encode("utf-8")

    blob = b"QFAI" + _struct.pack("<I", len(header)) + header + body
    qfai_path = tmp_path / "jfg.qfai.bin"
    qfai_path.write_bytes(blob)

    loaded = builder._load_jfg_from_checkpoint(qfai_path, torch.device("cpu"))
    # Architecture matches; eval mode; frozen parameters
    assert loaded.training is False
    assert all(not p.requires_grad for p in loaded.parameters())
    # Functionally equivalent: same forward signature.
    # JFG.forward(mask2, pose6): mask2 is (B, H, W) integer class indices in
    # [0, num_classes); pose6 is (B, 6) float. Use small spatial dims so the
    # CPU smoke is fast.
    mask = torch.zeros(1, 24, 32, dtype=torch.long)
    pose = torch.zeros(1, 6, dtype=torch.float32)
    # Override the renderer's out_h/out_w so the smoke output is tiny.
    loaded.out_h = 24
    loaded.out_w = 32
    with torch.no_grad():
        f1, f2 = loaded(mask, pose)
    assert f1.shape == f2.shape
    assert f1.shape[0] == 1


def test_load_jfg_from_raw_state_dict(tmp_path):
    """A bare state_dict ``.pt`` checkpoint must load via the default-arch
    fallback branch."""
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    gen = build_quantizr_faithful_renderer(
        num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1
    )
    pt_path = tmp_path / "jfg.pt"
    torch.save(gen.state_dict(), pt_path)

    loaded = builder._load_jfg_from_checkpoint(pt_path, torch.device("cpu"))
    assert loaded.training is False
    # Same param count
    n_loaded = sum(p.numel() for p in loaded.parameters())
    n_orig = sum(p.numel() for p in gen.parameters())
    assert n_loaded == n_orig
