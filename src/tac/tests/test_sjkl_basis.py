"""Tests for SJ-KL basis primitive (Wave-Ω-1 Council #2).

Verifies:
  - Math correctness: Fisher-matvec is symmetric PSD; eigenvalues sorted
    descending; eigenvectors orthonormal.
  - Round-trip: encode → pack → unpack → decode reconstructs the residual
    within FP4-quant noise.
  - Determinism: same inputs produce byte-identical outputs (archive
    reproducibility).
  - Effective-rank claim: on a synthetic low-rank operator the recovered
    rank matches; on a SegNet-like proxy operator the rank is < 100.

Per CLAUDE.md "MPS auth eval is NOISE": this test runs on CPU/MPS as
implementation correctness check ONLY. Final SJ-KL bytes for any submission
must be re-derived on CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tac.sjkl_basis import (
    SJKL_MAGIC,
    SJKLBasis,
    compute_sjkl_basis,
    decode_residual,
    effective_rank,
    encode_residual,
    fisher_matvec,
    lanczos_topk,
    pack_sjkl_basis,
    unpack_sjkl_basis,
)


# ---------------------------------------------------------------------------
# Test fixtures: tiny scorer-like models that share the SegNet/PoseNet API
# ---------------------------------------------------------------------------

class TinySegNetLike(nn.Module):
    """Mimics SegNet API: preprocess_input takes (B,T,C,H,W), uses last frame,
    forward returns (B, num_classes, H_seg, W_seg).
    """

    def __init__(self, num_classes: int = 5, in_h: int = 16, in_w: int = 16):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        # Shallow conv → matches SegNet's stride-2 stem in spirit
        self.stem = nn.Conv2d(3, num_classes, kernel_size=3, padding=1, bias=False)
        # Initialize deterministically
        with torch.no_grad():
            self.stem.weight.copy_(0.01 * torch.randn_like(self.stem.weight, generator=torch.Generator().manual_seed(7)))

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x[:, -1, ...], size=(self.in_h, self.in_w), mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class TinyPoseNetLike(nn.Module):
    """Mimics PoseNet API: preprocess_input takes (B,T,C,H,W), produces
    (B, T*C, H_pose, W_pose) → forward returns dict with "pose": (B, 12).
    """

    def __init__(self, in_h: int = 16, in_w: int = 16):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        # T*C = 2*3 = 6 in the YUV6-flattened API; here we use the same 6 chans
        self.head = nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            self.head.weight.copy_(0.01 * torch.randn_like(self.head.weight, generator=torch.Generator().manual_seed(13)))

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # Mimic the rgb_to_yuv6 flatten: just concatenate the 2 frames along
        # channel dim and downsample.
        b, t, c, h, w = x.shape
        x_resized = torch.nn.functional.interpolate(
            x.flatten(0, 1), size=(self.in_h, self.in_w), mode="bilinear",
            align_corners=False,
        )
        x_resized = x_resized.view(b, t * c, self.in_h, self.in_w)
        return x_resized

    def forward(self, x: torch.Tensor) -> dict:
        out = self.head(x)
        pooled = self.pool(out).view(x.shape[0], -1)  # (B, 12)
        return {"pose": pooled}


def _make_frames(n: int = 2, h: int = 16, w: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 128.0 + 30.0 * torch.randn(n, 2, 3, h, w, generator=g)


# ---------------------------------------------------------------------------
# Math correctness: fisher matvec is PSD and symmetric
# ---------------------------------------------------------------------------

def test_fisher_matvec_is_psd():
    """For random v, v^T F v >= 0 (positive semi-definite)."""
    h, w = 16, 16
    seg = TinySegNetLike(in_h=h, in_w=w).eval()
    pose = TinyPoseNetLike(in_h=h, in_w=w).eval()
    frames = _make_frames(h=h, w=w, seed=0)
    g = torch.Generator().manual_seed(42)
    for _trial in range(5):
        v = torch.randn(3, h, w, generator=g)
        Fv = fisher_matvec(seg, pose, frames[:1], v)
        quad = float((v * Fv).sum())
        assert quad >= -1e-4, f"Fisher matvec not PSD: v^T F v = {quad}"


def test_fisher_matvec_is_symmetric():
    """For random u, v: <u, Fv> ~= <v, Fu> (symmetric operator)."""
    h, w = 16, 16
    seg = TinySegNetLike(in_h=h, in_w=w).eval()
    pose = TinyPoseNetLike(in_h=h, in_w=w).eval()
    frames = _make_frames(h=h, w=w, seed=0)
    g = torch.Generator().manual_seed(7)
    for _trial in range(3):
        u = torch.randn(3, h, w, generator=g)
        v = torch.randn(3, h, w, generator=g)
        Fv = fisher_matvec(seg, pose, frames[:1], v)
        Fu = fisher_matvec(seg, pose, frames[:1], u)
        u_Fv = float((u * Fv).sum())
        v_Fu = float((v * Fu).sum())
        # Symmetric to within FP32 noise; allow 1% relative
        rel = abs(u_Fv - v_Fu) / max(abs(u_Fv) + abs(v_Fu), 1e-10)
        assert rel < 1e-3, f"Fisher not symmetric: <u,Fv>={u_Fv}, <v,Fu>={v_Fu}"


def test_lanczos_topk_on_known_matrix():
    """Apply Lanczos to a known symmetric matrix; verify recovered eigvals
    match scipy.eigh and eigvecs are orthonormal.
    """
    n = 50
    g = torch.Generator().manual_seed(3)
    A = torch.randn(n, n, generator=g)
    A = A @ A.t()  # PSD

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return A @ v

    eigvals, eigvecs = lanczos_topk(matvec, dim=n, k=8, n_iters=30, seed=0)
    # Compare to ground truth
    ref_eigvals, _ = torch.linalg.eigh(A)
    ref_eigvals = torch.sort(ref_eigvals, descending=True).values
    # Top-8 should match within 1% relative
    for i in range(8):
        rel = abs(float(eigvals[i] - ref_eigvals[i])) / max(abs(float(ref_eigvals[i])), 1e-10)
        assert rel < 1e-2, f"top-{i} eigval mismatch: lanczos={eigvals[i]}, ref={ref_eigvals[i]}"
    # Eigvecs orthonormal
    gram = eigvecs @ eigvecs.t()
    eye = torch.eye(8, dtype=gram.dtype)
    err = float((gram - eye).abs().max())
    assert err < 1e-3, f"eigvecs not orthonormal: max off-diag = {err}"


def test_eigenvalues_sorted_descending():
    """Verify compute_sjkl_basis returns a basis with implicit eigenvalue
    ordering (encoded into the scale field as sqrt(λ)).
    """
    h, w = 16, 16
    seg = TinySegNetLike(in_h=h, in_w=w).eval()
    pose = TinyPoseNetLike(in_h=h, in_w=w).eval()
    frames = _make_frames(n=4, h=h, w=w, seed=0)
    basis = compute_sjkl_basis(seg, pose, frames, k=4, basis_grid_h=8, basis_grid_w=8)
    # scale = sqrt(eigval) so it should be non-increasing
    sc = basis.scale.tolist()
    for i in range(len(sc) - 1):
        # Allow tiny numerical noise but enforce non-increasing within 1%
        assert sc[i] >= sc[i + 1] - 1e-4 * max(sc[0], 1e-10), \
            f"scale not sorted descending at index {i}: {sc}"


# ---------------------------------------------------------------------------
# Round-trip: encode → pack → unpack → decode
# ---------------------------------------------------------------------------

def test_pack_unpack_basis_roundtrip():
    """Pack and unpack a basis; verify all fields preserved within FP16
    plus compact basis quantization precision.
    """
    g = torch.Generator().manual_seed(5)
    basis = SJKLBasis(
        basis_coarse=torch.randn(8, 3, 32, 24, generator=g),
        scale=torch.linspace(10.0, 1.0, 8),
        target_h=384,
        target_w=512,
    )
    payload = pack_sjkl_basis(basis)
    # MAGIC prefix preserved
    assert payload[:4] == SJKL_MAGIC
    restored = unpack_sjkl_basis(payload)
    assert restored.target_h == 384
    assert restored.target_w == 512
    assert restored.basis_coarse.shape == (8, 3, 32, 24)
    # FP16 scale plus compact basis quantization keeps the coarse basis close
    # while cutting the charged basis payload materially.
    assert torch.allclose(restored.scale, basis.scale, atol=1e-2), \
        f"scale mismatch: {restored.scale} vs {basis.scale}"
    err = float((restored.basis_coarse - basis.basis_coarse).abs().max())
    assert err < 1e-1, f"basis_coarse mismatch: max abs = {err}"


def test_pack_basis_supports_legacy_lossless_fp16_layout():
    g = torch.Generator().manual_seed(55)
    basis = SJKLBasis(
        basis_coarse=torch.randn(2, 3, 5, 4, generator=g),
        scale=torch.tensor([1.25, 0.5]),
        target_h=20,
        target_w=16,
    )

    payload = pack_sjkl_basis(basis, basis_quant_bits=None)
    restored = unpack_sjkl_basis(payload)

    assert restored.target_h == basis.target_h
    assert restored.target_w == basis.target_w
    assert torch.allclose(restored.scale, basis.scale, atol=1e-3)
    assert torch.allclose(restored.basis_coarse, basis.basis_coarse, atol=2e-3)


def test_quantized_basis_payload_is_smaller_than_legacy_fp16():
    g = torch.Generator().manual_seed(56)
    basis = SJKLBasis(
        basis_coarse=torch.randn(1, 3, 8, 6, generator=g),
        scale=torch.tensor([30.0]),
        target_h=384,
        target_w=512,
    )

    legacy = pack_sjkl_basis(basis, basis_quant_bits=None)
    compact = pack_sjkl_basis(basis)
    restored = unpack_sjkl_basis(compact)

    assert len(compact) < len(legacy)
    rel = float((restored.basis_coarse - basis.basis_coarse).norm() / basis.basis_coarse.norm())
    assert rel < 0.07


def test_encode_decode_residual_roundtrip():
    """Encode a known residual onto an ORTHOGONAL basis; decode; verify
    reconstruction error is dominated by the alpha-quant noise (small for
    6-bit qint over a unit-scaled coefficient range).
    """
    h, w = 32, 32
    # 3 mutually-orthogonal eigenvectors: each lives in a single channel and
    # is a constant plane in that channel. Bilinear-upsample preserves
    # orthogonality (different channels never overlap regardless of spatial
    # interpolation).
    basis_coarse = torch.zeros(3, 3, 8, 8)
    basis_coarse[0, 0] = 1.0
    basis_coarse[1, 1] = 1.0
    basis_coarse[2, 2] = 1.0
    basis = SJKLBasis(
        basis_coarse=basis_coarse,
        scale=torch.tensor([5.0, 5.0, 5.0]),
        target_h=h,
        target_w=w,
    ).renormalize()

    # Verify orthonormality of the upsampled basis (sanity check).
    full = basis.upsample()
    gram = full.flatten(1) @ full.flatten(1).t()
    assert torch.allclose(gram, torch.eye(3), atol=1e-5), \
        f"basis not orthonormal: {gram}"

    target_alpha = torch.tensor([1.5, -0.7, 0.3])
    r = sum(target_alpha[j] * basis.scale[j] * full[j] for j in range(3))

    a_q, a_min, a_step = encode_residual(r, basis, alpha_bits=6)
    r_hat = decode_residual(a_q, a_min, a_step, basis)
    err = float((r - r_hat).abs().max())
    # 6-bit qint over coefficient range ~2.2 → step ~0.035; max error per
    # eigenvector is step/2 * scale ~= 0.09; max pixel error similar magnitude.
    rel = err / float(r.abs().max().clamp_min(1e-6))
    assert rel < 0.05, f"reconstruction error too high: {rel:.4f}"


def test_pack_alpha_block_roundtrip():
    """Pack and unpack the alpha block; verify exact recovery (no quantization
    loss in the block layer itself; only the encode_residual loses precision).
    """
    from experiments.build_sjkl_residual import pack_alpha_block, unpack_alpha_block

    n_pairs = 5
    k = 8
    g = np.random.default_rng(42)
    alpha_qs = [g.integers(0, 64, size=k, dtype=np.uint8) for _ in range(n_pairs)]
    alpha_mins = [float(g.uniform(-1.0, 1.0)) for _ in range(n_pairs)]
    alpha_steps = [float(g.uniform(0.01, 0.5)) for _ in range(n_pairs)]
    payload = pack_alpha_block(alpha_qs, alpha_mins, alpha_steps, alpha_bits=6)
    qs, mins, steps, ab = unpack_alpha_block(payload)
    assert ab == 6
    assert len(qs) == n_pairs
    for i in range(n_pairs):
        assert np.array_equal(qs[i], alpha_qs[i])
        # FP16 round-trip on mins/steps
        assert abs(mins[i] - alpha_mins[i]) < 1e-2
        assert abs(steps[i] - alpha_steps[i]) < 1e-2


def test_sparse_bitpacked_alpha_block_roundtrip():
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        unpack_alpha_block_with_indices,
    )

    alpha_qs = [
        np.array([0, 1, 2], dtype=np.uint8),
        np.array([3, 4, 5], dtype=np.uint8),
        np.array([6, 7, 0], dtype=np.uint8),
    ]
    payload = pack_alpha_block(
        alpha_qs,
        [-1.0, 0.0, 1.0],
        [0.25, 0.5, 0.75],
        alpha_bits=3,
        pair_indices=[2, 10, 31],
        sparse_bitpacked=True,
    )

    qs, mins, steps, alpha_bits, pair_indices = unpack_alpha_block_with_indices(payload)

    assert alpha_bits == 3
    assert pair_indices == [2, 10, 31]
    assert len(qs) == len(alpha_qs)
    for got, expected in zip(qs, alpha_qs):
        assert np.array_equal(got, expected)
    assert mins == pytest.approx([-1.0, 0.0, 1.0], abs=1e-3)
    assert steps == pytest.approx([0.25, 0.5, 0.75], abs=1e-3)


def test_full_payload_roundtrip():
    """End-to-end: pack basis + alpha block; unpack; verify SJKLBasis fields
    and per-pair alpha tuples recover.
    """
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        pack_full_sjkl_payload,
        unpack_full_sjkl_payload,
    )

    g = torch.Generator().manual_seed(1)
    basis = SJKLBasis(
        basis_coarse=torch.randn(4, 3, 16, 12, generator=g),
        scale=torch.tensor([2.0, 1.5, 1.0, 0.5]),
        target_h=384,
        target_w=512,
    )
    basis_bytes = pack_sjkl_basis(basis)
    n_pairs = 3
    rng = np.random.default_rng(0)
    alpha_qs = [rng.integers(0, 64, size=4, dtype=np.uint8) for _ in range(n_pairs)]
    mins = [-0.5, 0.0, 0.3]
    steps = [0.01, 0.02, 0.03]
    block_bytes = pack_alpha_block(alpha_qs, mins, steps, alpha_bits=6)
    payload = pack_full_sjkl_payload(basis_bytes, block_bytes)

    basis_r, qs_r, mins_r, steps_r, ab_r = unpack_full_sjkl_payload(payload)
    assert basis_r.target_h == 384 and basis_r.target_w == 512
    assert basis_r.basis_coarse.shape == (4, 3, 16, 12)
    assert ab_r == 6
    assert len(qs_r) == n_pairs
    for i in range(n_pairs):
        assert np.array_equal(qs_r[i], alpha_qs[i])


def test_repack_sjkl_payload_can_downsample_truncate_and_select_sparse_rows(tmp_path):
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        pack_full_sjkl_payload,
        repack_sjkl_payload,
        unpack_full_sjkl_payload_with_indices,
    )

    g = torch.Generator().manual_seed(11)
    basis = SJKLBasis(
        basis_coarse=torch.randn(4, 3, 8, 6, generator=g),
        scale=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        target_h=32,
        target_w=32,
    ).renormalize()
    source_qs = [
        np.array([1, 1, 1, 1], dtype=np.uint8),
        np.array([63, 63, 63, 63], dtype=np.uint8),
        np.array([2, 2, 2, 2], dtype=np.uint8),
        np.array([31, 31, 31, 31], dtype=np.uint8),
    ]
    source_payload = pack_full_sjkl_payload(
        pack_sjkl_basis(basis),
        pack_alpha_block(source_qs, [0.0] * 4, [0.1] * 4, alpha_bits=6),
    )
    source = tmp_path / "source.sjkl.bin"
    source.write_bytes(source_payload)

    manifest = repack_sjkl_payload(
        source_sjkl_bin=source,
        out=tmp_path / "tiny.sjkl.bin",
        manifest=tmp_path / "tiny.manifest.json",
        alpha_bits=4,
        residual_gain=0.25,
        max_encoded_pairs=2,
        repack_k=2,
        repack_basis_grid_h=4,
        repack_basis_grid_w=3,
    )
    restored, qs, _mins, _steps, alpha_bits, pair_indices = unpack_full_sjkl_payload_with_indices(
        (tmp_path / "tiny.sjkl.bin").read_bytes()
    )

    assert manifest["score_claim"] is False
    assert manifest["alpha_block_format"] == "sparse_bitpacked_v2"
    assert manifest["selected_pair_count"] == 2
    assert manifest["k"] == 2
    assert restored.basis_coarse.shape == (2, 3, 4, 3)
    assert alpha_bits == 4
    assert pair_indices == manifest["selected_pair_indices"]
    assert len(qs) == 2


def test_repack_sjkl_payload_can_select_explicit_absolute_pairs(tmp_path):
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        pack_full_sjkl_payload,
        repack_sjkl_payload,
        unpack_full_sjkl_payload_with_indices,
    )

    g = torch.Generator().manual_seed(19)
    basis = SJKLBasis(
        basis_coarse=torch.randn(3, 3, 4, 4, generator=g),
        scale=torch.tensor([3.0, 2.0, 1.0]),
        target_h=16,
        target_w=16,
    ).renormalize()
    source_qs = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([4, 5, 6], dtype=np.uint8),
        np.array([7, 8, 9], dtype=np.uint8),
        np.array([10, 11, 12], dtype=np.uint8),
    ]
    source_payload = pack_full_sjkl_payload(
        pack_sjkl_basis(basis),
        pack_alpha_block(
            source_qs,
            [0.0] * 4,
            [0.1] * 4,
            alpha_bits=4,
            pair_indices=[10, 20, 30, 40],
            sparse_bitpacked=True,
        ),
    )
    source = tmp_path / "source.sjkl.bin"
    source.write_bytes(source_payload)

    manifest = repack_sjkl_payload(
        source_sjkl_bin=source,
        out=tmp_path / "explicit.sjkl.bin",
        manifest=tmp_path / "explicit.manifest.json",
        pair_selection="explicit",
        explicit_pair_indices=[30, 10],
        repack_k=2,
        alpha_bits=3,
    )
    _basis, qs, _mins, _steps, alpha_bits, pair_indices = unpack_full_sjkl_payload_with_indices(
        (tmp_path / "explicit.sjkl.bin").read_bytes()
    )

    assert manifest["pair_selection"] == "explicit"
    assert manifest["requested_pair_indices"] == [30, 10]
    assert manifest["selected_pair_indices"] == [30, 10]
    assert pair_indices == [30, 10]
    assert len(qs) == 2
    assert alpha_bits == 3


def test_repack_sjkl_payload_explicit_pairs_fail_closed(tmp_path):
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        pack_full_sjkl_payload,
        repack_sjkl_payload,
    )

    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, 2, 2),
        scale=torch.tensor([1.0]),
        target_h=8,
        target_w=8,
    ).renormalize()
    source_payload = pack_full_sjkl_payload(
        pack_sjkl_basis(basis),
        pack_alpha_block(
            [np.array([1], dtype=np.uint8), np.array([2], dtype=np.uint8)],
            [0.0, 0.0],
            [0.1, 0.1],
            alpha_bits=2,
            pair_indices=[7, 9],
            sparse_bitpacked=True,
        ),
    )
    source = tmp_path / "source.sjkl.bin"
    source.write_bytes(source_payload)

    with pytest.raises(ValueError, match="mutually exclusive"):
        repack_sjkl_payload(
            source_sjkl_bin=source,
            out=tmp_path / "bad.sjkl.bin",
            explicit_pair_indices=[7],
            max_encoded_pairs=1,
        )
    with pytest.raises(ValueError, match="not present"):
        repack_sjkl_payload(
            source_sjkl_bin=source,
            out=tmp_path / "missing.sjkl.bin",
            pair_selection="explicit",
            explicit_pair_indices=[8],
        )


def test_apply_sjkl_at_decode_respects_sparse_absolute_pair_indices():
    from experiments.build_sjkl_residual import (
        apply_sjkl_at_decode,
        pack_alpha_block,
        pack_full_sjkl_payload,
    )

    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, 1, 1),
        scale=torch.tensor([1.0]),
        target_h=2,
        target_w=2,
    ).renormalize()
    payload = pack_full_sjkl_payload(
        pack_sjkl_basis(basis),
        pack_alpha_block(
            [np.array([1], dtype=np.uint8)],
            [0.0],
            [1.0],
            alpha_bits=2,
            pair_indices=[3],
            sparse_bitpacked=True,
        ),
    )
    fake1 = torch.zeros(3, 2, 2)
    fake2 = torch.zeros(3, 2, 2)

    unchanged, _ = apply_sjkl_at_decode(fake1.clone(), fake2.clone(), payload, pair_idx=0)
    changed, _ = apply_sjkl_at_decode(fake1.clone(), fake2.clone(), payload, pair_idx=3)

    assert torch.equal(unchanged, fake1)
    assert not torch.equal(changed, fake1)


def test_residual_for_target_slot_matches_runtime_fake1_slot():
    """Builder residuals must target the same pair slot runtime corrects."""
    from experiments.build_sjkl_residual import residual_for_target_slot

    renderer_frame = torch.full((3, 2, 2), 10.0)
    gt_pair = torch.stack(
        [
            torch.full((3, 2, 2), 13.0),
            torch.full((3, 2, 2), 99.0),
        ],
        dim=0,
    )

    residual = residual_for_target_slot(renderer_frame, gt_pair, target_slot=0)

    assert torch.equal(residual, torch.full((3, 2, 2), 3.0))


def test_residual_for_target_slot_rejects_bad_slot():
    from experiments.build_sjkl_residual import residual_for_target_slot

    renderer_frame = torch.zeros(3, 2, 2)
    gt_pair = torch.zeros(2, 3, 2, 2)
    with pytest.raises(ValueError, match="target_slot"):
        residual_for_target_slot(renderer_frame, gt_pair, target_slot=2)


def test_resolve_sjkl_build_device_rejects_non_cuda_without_advisory_flag():
    from experiments.build_sjkl_residual import resolve_sjkl_build_device

    with pytest.raises(RuntimeError, match="non-CUDA"):
        resolve_sjkl_build_device("cpu", allow_non_cuda=False)


def test_resolve_sjkl_build_device_allows_explicit_advisory_cpu():
    from experiments.build_sjkl_residual import resolve_sjkl_build_device

    device = resolve_sjkl_build_device("cpu", allow_non_cuda=True)

    assert device.type == "cpu"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_pack_basis():
    """Same SJKLBasis → byte-identical payload (archive reproducibility)."""
    g = torch.Generator().manual_seed(99)
    basis = SJKLBasis(
        basis_coarse=torch.randn(4, 3, 8, 8, generator=g),
        scale=torch.tensor([1.0, 0.5, 0.25, 0.125]),
        target_h=384,
        target_w=512,
    )
    payload1 = pack_sjkl_basis(basis)
    payload2 = pack_sjkl_basis(basis)
    assert payload1 == payload2


def test_determinism_compute_basis():
    """Same scorers + frames + seed → identical basis_coarse."""
    h, w = 16, 16
    seg = TinySegNetLike(in_h=h, in_w=w).eval()
    pose = TinyPoseNetLike(in_h=h, in_w=w).eval()
    frames = _make_frames(h=h, w=w, seed=0)
    b1 = compute_sjkl_basis(seg, pose, frames, k=4, basis_grid_h=8, basis_grid_w=8, seed=0)
    b2 = compute_sjkl_basis(seg, pose, frames, k=4, basis_grid_h=8, basis_grid_w=8, seed=0)
    assert torch.allclose(b1.basis_coarse, b2.basis_coarse, atol=1e-5), \
        "basis not deterministic"
    assert torch.allclose(b1.scale, b2.scale, atol=1e-5), \
        "scale not deterministic"


# ---------------------------------------------------------------------------
# Effective-rank claim verification
# ---------------------------------------------------------------------------

def test_effective_rank_helper():
    """effective_rank counts evals above threshold * max(eval)."""
    eigvals = torch.tensor([100.0, 50.0, 10.0, 1.0, 0.001, 1e-9])
    # threshold=1e-6: above 1e-6*100=1e-4 → 100, 50, 10, 1, 0.001 (5 vals)
    assert effective_rank(eigvals, threshold=1e-6) == 5
    # threshold=1e-2: above 1e-2*100=1.0 → 100, 50, 10 (3 vals; 1.0 itself
    # is not strictly greater than 1.0)
    assert effective_rank(eigvals, threshold=1e-2) == 3
    assert effective_rank(torch.tensor([]), threshold=1e-6) == 0
    assert effective_rank(torch.tensor([0.0, 0.0]), threshold=1e-6) == 0


def test_fisher_low_rank_claim():
    """Council Section 5.3 claim: F = J_seg^T J_seg + J_pose^T J_pose has
    rank ≤ dim(J_seg.out) + dim(J_pose.out) = 5*H_seg*W_seg + 6.

    The claim is meaningful only when dim(input) > dim(scorer outputs).
    This test constructs an INPUT-DIM-DOMINANT regime: input is 3*32*32 = 3072
    pixels but the scorer aggregates massively (J_seg.out_dim = 5 * 4 * 4 = 80
    after stride-8 downsampling and J_pose.out_dim = 12). Total scorer-output
    dim = 92, so Fisher rank ≤ 92 << 3072.

    Empirical verification of THIS test is the deliverable that justifies the
    Council's "low rank" architectural claim.
    """
    # Input large; scorer aggressively downsamples its output (stride-8).
    h, w = 32, 32
    dim = 3 * h * w  # 3072
    seg = TinySegNetLike(in_h=4, in_w=4, num_classes=5).eval()  # output is 5*4*4 = 80 dims
    pose = TinyPoseNetLike(in_h=4, in_w=4).eval()  # output pooled to 12 dims
    frames = _make_frames(h=h, w=w, seed=0)

    # Estimate the scorer-output dim bottleneck for the rank bound.
    seg_out_dim = 5 * 4 * 4
    pose_out_dim = 12
    rank_bound = seg_out_dim + pose_out_dim  # 92

    # Recover top-(rank_bound + 50) eigvals via Lanczos and check the rest
    # are negligible.
    def avg_matvec(v):
        return fisher_matvec(seg, pose, frames[:1], v)

    n_probe = min(rank_bound + 30, dim)
    eigvals, _ = lanczos_topk(
        matvec=avg_matvec, dim=dim, k=n_probe, n_iters=n_probe + 10,
        seed=0, shape_hint=(3, h, w),
    )
    eff_rank = effective_rank(eigvals, threshold=1e-4)
    print(f"[fisher rank test] dim={dim}, scorer-output bound={rank_bound}, "
          f"effective rank={eff_rank}")
    # Effective rank should be ≤ scorer-output-dim bound + a few for noise.
    # Real test: rank << dim. We pin a generous bound (rank_bound + 20).
    assert eff_rank < rank_bound + 20, \
        f"Fisher rank {eff_rank} > scorer-output bound {rank_bound} + 20; " \
        f"Council low-rank claim REFUTED (would invalidate SJ-KL premise)"
    assert eff_rank < dim // 4, \
        f"Fisher rank {eff_rank} not << dim {dim}"


def test_compute_basis_smoke():
    """End-to-end smoke test: compute basis on tiny scorers, verify shapes
    and finite values.
    """
    h, w = 16, 16
    seg = TinySegNetLike(in_h=h, in_w=w).eval()
    pose = TinyPoseNetLike(in_h=h, in_w=w).eval()
    frames = _make_frames(n=4, h=h, w=w, seed=0)
    basis = compute_sjkl_basis(
        seg, pose, frames, k=6, basis_grid_h=8, basis_grid_w=8,
    )
    assert basis.basis_coarse.shape == (6, 3, 8, 8)
    assert basis.scale.shape == (6,)
    assert torch.isfinite(basis.basis_coarse).all()
    assert torch.isfinite(basis.scale).all()
    assert (basis.scale >= 0).all()
    # After renormalization, upsampled eigenvectors should be ~unit norm
    full = basis.upsample()
    norms = full.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), \
        f"upsampled eigenvectors not unit-norm: {norms}"


def test_unpack_rejects_bad_magic():
    """Unpacking with wrong magic raises a clear error."""
    with pytest.raises(ValueError, match="bad magic"):
        unpack_sjkl_basis(b"XXXX" + b"\x00" * 100)


def test_alpha_bits_8_works():
    """Encoding with alpha_bits=8 still round-trips."""
    basis_coarse = torch.zeros(2, 3, 8, 8)
    basis_coarse[0, 0] = 1.0  # red plane
    basis_coarse[1, 1] = 1.0  # green plane (orthogonal channel)
    basis = SJKLBasis(
        basis_coarse=basis_coarse,
        scale=torch.tensor([3.0, 2.0]),
        target_h=16,
        target_w=16,
    ).renormalize()
    full = basis.upsample()
    r = 0.5 * basis.scale[0] * full[0] - 0.3 * basis.scale[1] * full[1]
    a_q, a_min, a_step = encode_residual(r, basis, alpha_bits=8)
    assert a_q.dtype == np.uint8
    r_hat = decode_residual(a_q, a_min, a_step, basis)
    rel = float((r - r_hat).abs().max() / r.abs().max().clamp_min(1e-6))
    # 8-bit ~0.4% step over coefficient range; reconstruction within ~1%
    assert rel < 0.02, f"8-bit reconstruction error too high: {rel:.4f}"


def test_rgb_to_yuv6_diff_matches_upstream():
    """Sanity check: the differentiable copy of rgb_to_yuv6 in sjkl_basis
    produces the same numerical output as upstream/frame_utils.py:rgb_to_yuv6
    on shared test inputs.
    """
    from pathlib import Path
    import sys as _sys

    _UPSTREAM = Path(__file__).resolve().parent.parent.parent.parent / "upstream"
    if str(_UPSTREAM) not in _sys.path:
        _sys.path.insert(0, str(_UPSTREAM))
    try:
        from frame_utils import rgb_to_yuv6 as _upstream_rgb_to_yuv6  # type: ignore
    except ImportError:
        pytest.skip("upstream/frame_utils.py not importable in this env")
    from tac.sjkl_basis import _rgb_to_yuv6_diff

    g = torch.Generator().manual_seed(0)
    x = (128.0 + 30.0 * torch.randn(1, 3, 64, 64, generator=g)).clamp(0.0, 255.0)
    upstream_y = _upstream_rgb_to_yuv6(x.clone())
    diff_y = _rgb_to_yuv6_diff(x.clone())
    assert upstream_y.shape == diff_y.shape, \
        f"shape mismatch: upstream {upstream_y.shape} vs diff {diff_y.shape}"
    assert torch.allclose(upstream_y, diff_y, atol=1e-4), \
        f"rgb_to_yuv6 numerically differ: max abs = {(upstream_y - diff_y).abs().max()}"
    # And: differentiable variant must actually be differentiable
    x_grad = x.clone().requires_grad_(True)
    diff_y_grad = _rgb_to_yuv6_diff(x_grad)
    diff_y_grad.sum().backward()
    assert x_grad.grad is not None, "differentiable rgb_to_yuv6 broke autograd"
    assert torch.isfinite(x_grad.grad).all(), "rgb_to_yuv6 grad has NaN/Inf"


def test_byte_budget_estimate():
    """Document expected archive byte budget at the realistic deployment
    config (k=8, grid=32x24, alpha=6-bit, n_pairs=600).

    This test does NOT enforce a hard limit — it just exercises the pack
    path at production-like dimensions and prints the byte count, so the
    Council Section 5.3 ~13-18 KB target can be measured.
    """
    from experiments.build_sjkl_residual import (
        pack_alpha_block,
        pack_full_sjkl_payload,
    )

    g = torch.Generator().manual_seed(0)
    basis = SJKLBasis(
        basis_coarse=torch.randn(8, 3, 32, 24, generator=g),
        scale=torch.tensor([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]),
        target_h=384,
        target_w=512,
    )
    basis_bytes = pack_sjkl_basis(basis)
    n_pairs = 600
    rng = np.random.default_rng(0)
    alpha_qs = [rng.integers(0, 64, size=8, dtype=np.uint8) for _ in range(n_pairs)]
    mins = [0.0] * n_pairs
    steps = [0.01] * n_pairs
    block_bytes = pack_alpha_block(alpha_qs, mins, steps, alpha_bits=6)
    payload = pack_full_sjkl_payload(basis_bytes, block_bytes)
    print(f"[byte budget] basis={len(basis_bytes)}B, block={len(block_bytes)}B, "
          f"total={len(payload)}B, rate_contrib={25.0 * len(payload)/37545489:.6f}")
    # The 16 KB / 1KB ranges are SOFT — the test passes as long as the pack
    # path produces non-trivial bytes.
    assert 1000 < len(payload) < 200000, \
        f"unexpected total payload size: {len(payload)}B"
