# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.scorer_basin_parity`.

These tests use synthetic stub modules (no real upstream PoseNet/SegNet, no
real HNeRV decoder) to keep CI fast and dependency-light. The tests verify:

  - Hutchinson trace estimator matches an analytic Hessian on a quadratic.
  - ``compute_scorer_basin_parity`` reports PASSED on a benign perturbation.
  - ``compute_scorer_basin_parity`` reports FAILED on a divergent perturbation
    (large weight kick -> both distortion and curvature blow up).
"""

from __future__ import annotations

import json
import math
import zipfile

import einops
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.scorer_basin_parity import (
    DEFAULT_POSE_DIST_DELTA_THRESHOLD,
    ParityReport,
    compute_scorer_basin_parity,
    hutchinson_trace_estimate,
    reconstruct_frames,
)

# ---------------------------------------------------------------------------
# Stub scorer modules - minimal API matching upstream PoseNet/SegNet
# ---------------------------------------------------------------------------


class _StubPoseNet(nn.Module):
    """Tiny PoseNet stub: 12-dim pose head from mean-pooled frame pairs."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.linear = nn.Linear(6, 12)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=2, 3, H, W). Reduce to a (B, 12)-like input via mean+spatial.
        # Output shape: (B, 12) (no separate 'h' axis - _build_pose_loss still works).
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w).mean(dim=(2, 3)) / 255.0  # (B, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.linear(x)}

    def compute_distortion(self, out_a: dict[str, torch.Tensor], out_b: dict[str, torch.Tensor]) -> torch.Tensor:
        # MSE on first 6 of 12 dims (matches upstream semantics)
        a, b = out_a["pose"], out_b["pose"]
        diff = (a[..., :6] - b[..., :6]).pow(2)
        return diff.mean(dim=tuple(range(1, diff.ndim)))


class _StubSegNet(nn.Module):
    """Tiny SegNet stub: 5-class logits over a tiny feature map."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # Use last frame (matches upstream SegNet)
        last = x[:, -1, ...]  # (B, 3, H, W)
        return F.interpolate(last, size=(16, 16), mode="bilinear", align_corners=False) / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def compute_distortion(self, out_a: torch.Tensor, out_b: torch.Tensor) -> torch.Tensor:
        diff = (out_a.argmax(dim=1) != out_b.argmax(dim=1)).float()
        return diff.mean(dim=tuple(range(1, diff.ndim)))


class _StubDecoder(nn.Module):
    """Tiny decoder: latent -> low-res RGB pair, upsampled by reconstruct_frames."""

    def __init__(self, latent_dim: int = 4, eval_size: tuple[int, int] = (8, 8)):
        super().__init__()
        self.eval_size = eval_size
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 2 * 3 * eval_size[0] * eval_size[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        eh, ew = self.eval_size
        return torch.sigmoid(self.linear(z)).reshape(b, 2, 3, eh, ew) * 255.0


# ---------------------------------------------------------------------------
# Hutchinson trace correctness - analytic check
# ---------------------------------------------------------------------------


def test_hutchinson_trace_recovers_quadratic_hessian() -> None:
    """For loss = 0.5 x^T A x + b^T x, Tr(H) = Tr(A) (constant in x)."""

    torch.manual_seed(7)
    n = 6
    a = torch.randn(n, n)
    a = a @ a.T  # SPD, so trace(A) is positive
    expected_trace = float(torch.diagonal(a).sum().item())

    x = nn.Parameter(torch.randn(n))

    def _loss_fn() -> torch.Tensor:
        return 0.5 * (x @ a @ x)

    gen = torch.Generator()
    gen.manual_seed(11)
    estimated = hutchinson_trace_estimate(_loss_fn, [x], n_samples=64, generator=gen)
    # 64 Rademacher samples -> ~10% relative error tolerable
    assert math.isfinite(estimated)
    assert abs(estimated - expected_trace) / max(abs(expected_trace), 1e-6) < 0.30


# ---------------------------------------------------------------------------
# Synthetic basin-parity tests
# ---------------------------------------------------------------------------


def _make_synthetic_setup(latent_dim: int = 4, n_pairs: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    decoder = _StubDecoder(latent_dim=latent_dim, eval_size=(8, 8))
    posenet = _StubPoseNet().eval()
    segnet = _StubSegNet().eval()
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False
    latents = torch.randn(n_pairs, latent_dim) * 0.5
    # Synthetic GT: small deterministic uint8 frames at the camera resolution.
    gt = torch.randint(0, 256, (n_pairs, 2, 32, 32, 3), dtype=torch.uint8)
    return decoder, posenet, segnet, latents, gt


def test_parity_passes_on_benign_perturbation() -> None:
    """Tiny weight perturbation -> reconstructed frames almost identical -> parity passes."""

    decoder, posenet, segnet, latents, gt = _make_synthetic_setup(seed=0, n_pairs=6)

    lossless = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    # Quantization-style tiny perturbation: round to a fine step.
    quantized = {
        k: (torch.round(v / 1e-3) * 1e-3) for k, v in lossless.items()
    }

    report = compute_scorer_basin_parity(
        quantized_state_dict=quantized,
        lossless_state_dict=lossless,
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        latents=latents,
        gt_frames=gt,
        n_probes=4,
        n_hessian_samples=2,
        n_hessian_pairs=2,
        absolute_seg_ceiling=0.10,
        device="cpu",
        seed=42,
    )

    assert isinstance(report, ParityReport)
    assert report.basin_parity_passed, (
        f"Expected parity PASS on benign perturbation; failure_reasons={report.failure_reasons}; "
        f"pose_delta={report.pose_dist_delta:.3e} seg_delta={report.seg_dist_delta:.3e} "
        f"log_ratio={report.hessian_log_ratio:.3f}"
    )
    # Distortion-delta should be very small (perturbation is at 1e-3 weight level)
    assert abs(report.pose_dist_delta) < DEFAULT_POSE_DIST_DELTA_THRESHOLD


def test_parity_fails_on_divergent_perturbation() -> None:
    """Large weight kick -> reconstructed frames diverge -> parity fails."""

    # Use a fresh seed-stable setup, but with synthetic frames forced to make
    # the divergence VISIBLE in segmap: we want the kicked decoder to produce
    # a different argmax distribution from the unkicked one.
    decoder, posenet, segnet, latents, gt = _make_synthetic_setup(seed=0, n_pairs=6)

    # Strengthen the SegNet distinguishability so segnet distortion is sensitive
    # to large pixel changes - give it a strong bias on one channel.
    with torch.no_grad():
        segnet.conv.weight.zero_()
        # Make class 0 sensitive to red, class 4 sensitive to blue, etc.
        for c in range(5):
            segnet.conv.weight[c, c % 3, :, :] = 1.0

    lossless = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    quantized = {k: v.detach().clone() for k, v in lossless.items()}
    # KICK the linear bias by a huge amount - this shifts every pixel away.
    quantized["linear.bias"] = lossless["linear.bias"] + 50.0
    # And add Gaussian noise to the weight to blow up curvature
    quantized["linear.weight"] = lossless["linear.weight"] + torch.randn_like(
        lossless["linear.weight"]
    ) * 5.0

    report = compute_scorer_basin_parity(
        quantized_state_dict=quantized,
        lossless_state_dict=lossless,
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        latents=latents,
        gt_frames=gt,
        n_probes=4,
        n_hessian_samples=2,
        n_hessian_pairs=2,
        # Use TIGHT thresholds so the test is deterministic in the
        # divergence direction.
        pose_threshold=1e-6,
        seg_threshold=1e-6,
        hessian_log_ratio_tolerance=0.5,
        device="cpu",
        seed=42,
    )

    assert isinstance(report, ParityReport)
    assert not report.basin_parity_passed, (
        f"Expected parity FAIL on divergent perturbation; got passed=True. "
        f"pose_delta={report.pose_dist_delta:.3e} seg_delta={report.seg_dist_delta:.3e} "
        f"log_ratio={report.hessian_log_ratio:.3f}"
    )
    assert report.failure_reasons, "failure_reasons must be populated on FAIL"


def test_reconstruct_frames_returns_uint8_correct_shape() -> None:
    """Smoke: reconstruct_frames returns expected shape/dtype."""

    decoder = _StubDecoder(latent_dim=4, eval_size=(8, 8))
    latents = torch.randn(3, 4)
    out = reconstruct_frames(
        decoder,
        decoder.state_dict(),
        latents,
        eval_size=(8, 8),
        camera_size=(32, 48),
        n_pairs=3,
    )
    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (3, 2, 32, 48, 3)


def test_parity_threshold_violation_message_is_useful() -> None:
    """A failure should include actionable failure_reasons strings."""

    decoder, posenet, segnet, latents, gt = _make_synthetic_setup(seed=1, n_pairs=4)
    lossless = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    # Perturbation enough to push deltas above microscopic thresholds.
    quantized = {k: v + 0.01 for k, v in lossless.items()}

    report = compute_scorer_basin_parity(
        quantized_state_dict=quantized,
        lossless_state_dict=lossless,
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        latents=latents,
        gt_frames=gt,
        n_probes=2,
        n_hessian_samples=2,
        n_hessian_pairs=2,
        pose_threshold=0.0,  # force fail
        seg_threshold=0.0,  # force fail
        hessian_log_ratio_tolerance=100.0,
        device="cpu",
        seed=42,
    )
    assert not report.basin_parity_passed
    msg = " ".join(report.failure_reasons)
    assert "pose_dist_delta" in msg or "seg_dist_delta" in msg


def test_n_probes_validation() -> None:
    decoder = _StubDecoder(latent_dim=4)
    posenet = _StubPoseNet().eval()
    segnet = _StubSegNet().eval()
    latents = torch.randn(2, 4)
    gt = torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="n_probes"):
        compute_scorer_basin_parity(
            quantized_state_dict=decoder.state_dict(),
            lossless_state_dict=decoder.state_dict(),
            decoder=decoder,
            posenet=posenet,
            segnet=segnet,
            latents=latents,
            gt_frames=gt,
            n_probes=10,
        )


# Make einops-rearrange import-time errors loud (the module may be missing
# in stripped envs).
def test_einops_available() -> None:
    assert hasattr(einops, "rearrange")


def test_emit_evidence_json_schema_matches_gate6(tmp_path) -> None:
    """The CLI's emit_evidence_json must produce a JSON consumable by Gate 6.

    Gate 6 (predispatch_sanity._gate_apogee_evidence_semantics) requires:
      - candidate_archive_sha256 OR archive_sha256 OR archive.sha256
      - evidence_semantics in APOGEE_ALLOWED_EVIDENCE_SEMANTICS
      - ready_for_exact_eval_dispatch is True
      - evidence_grade not in {invalid, external, prediction, "*negative*"}
      - scorer_basin_parity_status in APOGEE_PASS_STATUSES
    """

    # We don't import the CLI module top-level (it pulls in upstream codecs).
    # Inject the emit helper via importlib so the test can exercise it.
    import importlib.util as _iu
    import json
    from pathlib import Path

    cli_path = Path(__file__).resolve().parents[3] / "tools" / "build_scorer_basin_parity_evidence.py"
    spec = _iu.spec_from_file_location("_basin_parity_cli_test", cli_path)
    assert spec is not None and spec.loader is not None
    cli_mod = _iu.module_from_spec(spec)

    # Make tac/repo_io importable when loading the CLI module.
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    for cand in (str(src_root), str(repo_root)):
        if cand not in _sys.path:
            _sys.path.insert(0, cand)

    spec.loader.exec_module(cli_mod)

    decoder, posenet, segnet, latents, gt = _make_synthetic_setup(seed=2, n_pairs=4)
    lossless = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    quantized = {k: v.detach().clone() for k, v in lossless.items()}
    report = compute_scorer_basin_parity(
        quantized_state_dict=quantized,
        lossless_state_dict=lossless,
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        latents=latents,
        gt_frames=gt,
        n_probes=2,
        n_hessian_samples=2,
        n_hessian_pairs=2,
        absolute_seg_ceiling=0.10,
        device="cpu",
        seed=42,
    )

    out = tmp_path / "parity_evidence.json"
    cli_mod.emit_evidence_json(
        output_json_path=out,
        candidate_archive=tmp_path / "candidate.zip",
        candidate_archive_sha="cafebabe" * 8,
        lossless_archive=tmp_path / "lossless.zip",
        lossless_archive_sha="deadbeef" * 8,
        report=report,
        extra_notes="synthetic-test",
    )
    payload = json.loads(out.read_text())
    # Schema check - these are exactly the keys Gate 6 inspects
    assert payload["evidence_semantics"] == "scorer_basin_parity_gate"
    assert payload["candidate_archive_sha256"] == "cafebabe" * 8
    assert payload["archive_sha256"] == payload["candidate_archive_sha256"]
    assert payload["ready_for_exact_eval_dispatch"] is True
    assert payload["scorer_basin_parity_status"] == "pass"
    assert payload["evidence_grade"] == "empirical"
    assert payload["readiness_status"] == "pass"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    # The forbidden-marker scan in Gate 6 will reject any of these strings,
    # even in nested fields. Verify our notes don't accidentally include them.
    forbidden = (
        "byte_only", "byte-only", "prediction_only", "predicted_band",
        "invalid_predicted_band", "proxy_only", "distortion_proxy_local",
        "local_distortion_proxy", "[distortion-proxy:local]",
    )
    notes_lower = payload["notes"].lower()
    for marker in forbidden:
        assert marker.lower() not in notes_lower, (
            f"emit_evidence_json placed forbidden marker {marker!r} in notes"
        )
    # The full report should be embedded for forensics
    assert "parity_report" in payload
    assert payload["parity_report"]["basin_parity_passed"] is True


def test_emit_evidence_json_records_payload_and_latent_custody(tmp_path) -> None:
    import importlib.util
    import sys as _sys
    from pathlib import Path

    tool_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "build_scorer_basin_parity_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("build_scorer_basin_parity_evidence", tool_path)
    assert spec and spec.loader
    cli_mod = importlib.util.module_from_spec(spec)
    _sys.modules["build_scorer_basin_parity_evidence"] = cli_mod
    for cand in (
        str(Path(__file__).resolve().parents[3] / "src"),
        str(Path(__file__).resolve().parents[3]),
    ):
        if cand not in _sys.path:
            _sys.path.insert(0, cand)
    spec.loader.exec_module(cli_mod)

    report = ParityReport(
        pose_dist_lossless=1e-4,
        pose_dist_quantized=2e-4,
        pose_dist_delta=1e-4,
        seg_dist_lossless=1e-3,
        seg_dist_quantized=2e-3,
        seg_dist_delta=1e-3,
        hessian_trace_lossless=10.0,
        hessian_trace_quantized=11.0,
        hessian_log_ratio=0.04,
        basin_parity_passed=True,
        pose_threshold=1e-3,
        seg_threshold=5e-3,
        hessian_log_ratio_tolerance=1.0,
        absolute_pose_ceiling=1e-2,
        absolute_seg_ceiling=2e-2,
        n_probes=2,
        n_hessian_samples=1,
        anchor_frame_shas=("abc",),
        device="cpu",
        computed_utc="2026-05-07T00:00:00Z",
    )

    out = tmp_path / "custody.json"
    cli_mod.emit_evidence_json(
        output_json_path=out,
        candidate_archive=tmp_path / "candidate.zip",
        candidate_archive_sha="cafebabe" * 8,
        lossless_archive=tmp_path / "lossless.zip",
        lossless_archive_sha="deadbeef" * 8,
        report=report,
        candidate_payload_meta={"member_name": "0.bin", "member_sha256": "a"},
        lossless_payload_meta={"member_name": "0.bin", "member_sha256": "b"},
        candidate_latents_sha256="latents",
        lossless_latents_sha256="latents",
    )
    payload = json.loads(out.read_text())
    assert payload["candidate_payload_member"]["member_name"] == "0.bin"
    assert payload["lossless_payload_member"]["member_sha256"] == "b"
    assert payload["candidate_latents_sha256"] == "latents"
    assert payload["latents_match_exact"] is True


def test_zip_payload_reader_rejects_ambiguous_or_unsafe_members(tmp_path) -> None:
    import importlib.util
    import sys as _sys
    from pathlib import Path

    tool_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "build_scorer_basin_parity_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("build_scorer_basin_parity_evidence", tool_path)
    assert spec and spec.loader
    cli_mod = importlib.util.module_from_spec(spec)
    _sys.modules["build_scorer_basin_parity_evidence"] = cli_mod
    for cand in (
        str(Path(__file__).resolve().parents[3] / "src"),
        str(Path(__file__).resolve().parents[3]),
    ):
        if cand not in _sys.path:
            _sys.path.insert(0, cand)
    spec.loader.exec_module(cli_mod)

    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("0.bin", b"abc")
    info, payload, meta = cli_mod._safe_payload_member(good)
    assert info.filename == "0.bin"
    assert payload == b"abc"
    assert meta["member_bytes"] == 3

    multi = tmp_path / "multi.zip"
    with zipfile.ZipFile(multi, "w") as zf:
        zf.writestr("0.bin", b"abc")
        zf.writestr("1.bin", b"def")
    with pytest.raises(ValueError, match="exactly one"):
        cli_mod._safe_payload_member(multi)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as zf:
        zf.writestr("../0.bin", b"abc")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        cli_mod._safe_payload_member(unsafe)
