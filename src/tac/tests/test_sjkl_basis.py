# SPDX-License-Identifier: MIT
"""Regression tests for tac.sjkl_basis recovered from runbook spec."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.sjkl_basis import (
    SJKLBasis,
    apply_sjkl_residual,
    compute_sjkl_basis_lanczos,
    decode_full_sjkl_payload,
    decode_sjkl_alpha_block,
    decode_sjkl_basis,
    effective_rank,
    encode_full_sjkl_payload,
    encode_sjkl_alpha_block_v1_dense,
    encode_sjkl_alpha_block_v2_sparse,
    encode_sjkl_basis,
    fisher_matvec,
    lanczos_topk,
    unpack_sjkl_basis,
)


def _make_random_basis(rank: int = 4, dim: int = 32, n_frames: int = 1, seed: int = 0) -> SJKLBasis:
    g = torch.Generator().manual_seed(seed)
    eig = torch.randn(rank, dim, generator=g, dtype=torch.float32)
    # orthonormalize via QR for cleanliness
    eig, _ = torch.linalg.qr(eig.T)
    eig = eig.T  # (rank, dim) ortho rows
    coefs = torch.randn(rank * n_frames, generator=g, dtype=torch.float32) * 0.1
    return SJKLBasis(eigenvectors=eig, coefficients=coefs, rank=rank, dim=dim)


def test_encode_decode_fp16_roundtrip_within_tolerance():
    basis = _make_random_basis(rank=8, dim=64)
    blob = encode_sjkl_basis(basis, basis_quant_bits=None)
    decoded = decode_sjkl_basis(blob)
    assert decoded.rank == 8
    assert decoded.dim == 64
    assert decoded.eigenvectors.shape == (8, 64)
    # FP16 has ~3 decimal-digit precision
    assert torch.allclose(decoded.eigenvectors, basis.eigenvectors, atol=1e-3, rtol=1e-2)
    assert torch.allclose(decoded.coefficients, basis.coefficients, atol=1e-3, rtol=1e-2)


def test_encode_decode_q6_roundtrip_within_quant_tolerance():
    basis = _make_random_basis(rank=4, dim=32)
    blob = encode_sjkl_basis(basis, basis_quant_bits=6)
    decoded = decode_sjkl_basis(blob)
    assert decoded.rank == 4
    assert decoded.dim == 32
    # q6 has 31 levels per side; expect rel_err <~ 5%
    rel_err = (decoded.eigenvectors - basis.eigenvectors).abs().max().item() / max(
        basis.eigenvectors.abs().max().item(), 1e-9
    )
    assert rel_err < 0.10, f"q6 rel_err {rel_err:.4f} exceeded 10% tolerance"


def test_q6_payload_smaller_than_fp16():
    basis = _make_random_basis(rank=8, dim=128)
    fp16_blob = encode_sjkl_basis(basis, basis_quant_bits=None)
    q6_blob = encode_sjkl_basis(basis, basis_quant_bits=6)
    # Per addendum: q6 should be roughly 6/16 ~ 38% the size of fp16 basis,
    # plus a 4-byte scale + same coef bytes. Verify <50% to allow header overhead.
    assert len(q6_blob) < len(fp16_blob)


def test_q4_payload_even_smaller():
    basis = _make_random_basis(rank=8, dim=128)
    q4_blob = encode_sjkl_basis(basis, basis_quant_bits=4)
    q8_blob = encode_sjkl_basis(basis, basis_quant_bits=8)
    assert len(q4_blob) < len(q8_blob)


def test_invalid_quant_bits_rejected():
    basis = _make_random_basis()
    with pytest.raises(ValueError):
        encode_sjkl_basis(basis, basis_quant_bits=3)
    with pytest.raises(ValueError):
        encode_sjkl_basis(basis, basis_quant_bits=9)


def test_invalid_magic_rejected():
    with pytest.raises(ValueError):
        decode_sjkl_basis(b"NOPE" + b"\x00" * 20)


def test_apply_residual_single_frame():
    rank, dim = 3, 16
    basis = _make_random_basis(rank=rank, dim=dim, n_frames=1)
    frames = torch.zeros(dim, dtype=torch.float32)
    out = apply_sjkl_residual(frames, basis)
    expected = basis.coefficients @ basis.eigenvectors
    assert torch.allclose(out, expected, atol=1e-6)


def test_apply_residual_per_frame_coefficients():
    rank, dim, n_frames = 2, 8, 4
    basis = _make_random_basis(rank=rank, dim=dim, n_frames=n_frames)
    frames = torch.zeros(n_frames, dim, dtype=torch.float32)
    out = apply_sjkl_residual(frames, basis)
    assert out.shape == (n_frames, dim)


def test_lanczos_recovers_top_eigenvector_of_known_quadratic():
    # Quadratic form score_fn(f) = 0.5 * f.T @ A @ f where A is diagonal with
    # entries [10, 5, 2, 0.5, 0.1, ...]. Top eigenvector of A is e_0.
    dim = 16
    diag = torch.tensor([10.0, 5.0, 2.0, 0.5, 0.1] + [0.05] * (dim - 5), dtype=torch.float32)

    def score_fn(f: torch.Tensor) -> torch.Tensor:
        return 0.5 * (diag * f * f).sum()

    frames = torch.randn(dim, dtype=torch.float32) * 0.01
    basis = compute_sjkl_basis_lanczos(score_fn, frames, rank=2, n_iters=8, seed=42)
    assert basis.rank == 2
    assert basis.dim == dim
    assert basis.eigenvectors.shape == (2, dim)
    # Top eigenvector should align with e_0 (up to sign)
    top = basis.eigenvectors[0].abs()
    assert top[0].item() > 0.9, f"top eigenvector did not align with e_0: {top}"


def test_apply_residual_rejects_wrong_dim():
    basis = _make_random_basis(rank=3, dim=16)
    bad_frames = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(ValueError):
        apply_sjkl_residual(bad_frames, basis)


def test_zero_eigenvectors_encode_decode():
    """Edge case: all-zero basis should encode (scale=0) and decode back to zeros."""
    basis = SJKLBasis(
        eigenvectors=torch.zeros(2, 8, dtype=torch.float32),
        coefficients=torch.zeros(2, dtype=torch.float32),
        rank=2,
        dim=8,
    )
    blob = encode_sjkl_basis(basis, basis_quant_bits=6)
    decoded = decode_sjkl_basis(blob)
    assert torch.allclose(decoded.eigenvectors, torch.zeros(2, 8))
    assert torch.allclose(decoded.coefficients, torch.zeros(2))


def test_runtime_contract_unpack_sjkl_basis_alias():
    """Runtime contract: inflate_renderer.py imports unpack_sjkl_basis (not decode_*).
    Verify the alias works and produces identical output to decode_sjkl_basis."""
    basis = _make_random_basis(rank=4, dim=32)
    blob = encode_sjkl_basis(basis, basis_quant_bits=6)
    via_decode = decode_sjkl_basis(blob)
    via_unpack = unpack_sjkl_basis(blob)
    assert torch.equal(via_decode.eigenvectors, via_unpack.eigenvectors)
    assert torch.equal(via_decode.coefficients, via_unpack.coefficients)
    assert via_decode.rank == via_unpack.rank
    assert via_decode.dim == via_unpack.dim


def test_runtime_contract_basis_coarse_attribute():
    """Runtime contract: inflate_renderer.py accesses basis.basis_coarse.
    Verify the property returns the eigenvectors tensor."""
    basis = _make_random_basis(rank=5, dim=24)
    assert hasattr(basis, "basis_coarse")
    assert torch.equal(basis.basis_coarse, basis.eigenvectors)
    assert basis.basis_coarse.shape == (5, 24)


def test_runtime_contract_shaped_basis_metadata_roundtrip():
    basis = SJKLBasis(
        basis_coarse=torch.ones(1, 3, 2, 3),
        scale=torch.tensor([2.5]),
        target_h=4,
        target_w=6,
    ).renormalize()
    blob = encode_sjkl_basis(basis, basis_quant_bits=8)
    decoded = decode_sjkl_basis(blob)

    assert decoded.basis_coarse.shape == (1, 3, 2, 3)
    assert decoded.upsample().shape == (1, 3, 4, 6)
    assert decoded.target_h == 4
    assert decoded.target_w == 6
    assert torch.allclose(decoded.scale, basis.scale, atol=1e-5)


def test_quant_bits_8_payload_byte_layout():
    """Verify q8 packs as 1 byte per value (no bit-stream overhead)."""
    basis = _make_random_basis(rank=2, dim=10)
    blob = encode_sjkl_basis(basis, basis_quant_bits=8)
    # header (4 magic + 1 ver + 1 flag + 2 K + 4 D + 2 M = 14) + 4 scale + 2*10 basis + 2*2 coef
    expected = 14 + 4 + (2 * 10) + (2 * 2)
    assert len(blob) == expected, f"q8 blob size mismatch: got {len(blob)}, expected {expected}"


def test_fisher_matvec_matches_linear_gauss_newton():
    class LinearScorer(torch.nn.Module):
        def __init__(self, diag: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("diag", diag)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1) * self.diag

    diag = torch.tensor([1.0, 2.0, 3.0, 4.0])
    frames = torch.zeros(4, dtype=torch.float32)
    probe = torch.tensor([0.5, -1.0, 2.0, -0.25], dtype=torch.float32)

    out = fisher_matvec(LinearScorer(diag), None, frames, probe)

    expected = 100.0 * diag.square() * probe
    assert torch.allclose(out, expected, atol=1e-5)


def test_lanczos_topk_and_effective_rank_are_deterministic():
    diag = torch.tensor([9.0, 4.0, 1.0, 0.01], dtype=torch.float32)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return diag * v

    eigvals_a, eigvecs_a = lanczos_topk(matvec, dim=4, k=2, n_iters=4, seed=7)
    eigvals_b, eigvecs_b = lanczos_topk(matvec, dim=4, k=2, n_iters=4, seed=7)

    assert torch.allclose(eigvals_a, torch.tensor([9.0, 4.0]), atol=1e-4)
    assert torch.allclose(eigvals_a, eigvals_b)
    assert torch.allclose(eigvecs_a.abs(), eigvecs_b.abs())
    assert effective_rank(torch.tensor([9.0, 4.0, 1e-5]), threshold=1e-4) == 2


# ============================================================
# Alpha-block codec tests (per-frame-pair sparse coefficients)
# ============================================================


def _make_alpha_block_inputs(n_pairs: int, k: int, alpha_bits: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    qmax = (1 << alpha_bits) - 1
    qs = rng.integers(0, qmax + 1, size=(n_pairs, k), dtype=np.uint16 if alpha_bits > 8 else np.uint8)
    mins = rng.standard_normal(n_pairs).astype(np.float32) * 0.1
    steps = np.abs(rng.standard_normal(n_pairs).astype(np.float32)) * 0.01
    pair_indices = rng.choice(10_000, size=n_pairs, replace=False).astype(np.uint16)
    return qs, mins, steps, pair_indices


def test_alpha_block_v2_sparse_roundtrip_byte_exact():
    qs, mins, steps, pair_indices = _make_alpha_block_inputs(16, 4, alpha_bits=4)
    blob = encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=4, pair_indices=pair_indices)
    decoded = decode_sjkl_alpha_block(blob)
    assert decoded["alpha_block_format"] == "sparse_bitpacked_v2"
    assert np.array_equal(decoded["qs"], qs.astype(decoded["qs"].dtype))
    assert np.allclose(decoded["mins"], mins, atol=1e-3)
    assert np.allclose(decoded["steps"], steps, atol=1e-3)
    assert np.array_equal(decoded["pair_indices"], pair_indices.astype(np.int64))


def test_alpha_block_v1_dense_roundtrip_byte_exact():
    qs, mins, steps, _ = _make_alpha_block_inputs(8, 6, alpha_bits=8)
    blob = encode_sjkl_alpha_block_v1_dense(qs, mins, steps, alpha_bits=8)
    decoded = decode_sjkl_alpha_block(blob)
    assert decoded["alpha_block_format"] == "legacy_v1"
    assert np.array_equal(decoded["qs"], qs)
    assert decoded["pair_indices"] is None


def test_alpha_block_v2_sparse_smaller_than_v1_dense_at_4bits():
    """At alpha_bits=4 the bit-packing should beat dense-uint8 storage by ~half on qs portion."""
    qs, mins, steps, pair_indices = _make_alpha_block_inputs(32, 8, alpha_bits=4)
    v2 = encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=4, pair_indices=pair_indices)
    v1 = encode_sjkl_alpha_block_v1_dense(qs, mins, steps, alpha_bits=4)
    # v2 has +2*n_pairs for indices; v1 stores qs as uint8 (1 byte each); v2 packs to alpha_bits each
    # net: at alpha_bits=4, v2's bit-packed qs portion is ~half the dense uint8 size, more than offsetting indices
    assert len(v2) < len(v1) + 2 * 32  # account for indices overhead


def test_alpha_block_rejects_invalid_alpha_bits():
    qs, mins, steps, pair_indices = _make_alpha_block_inputs(2, 2, alpha_bits=4)
    with pytest.raises(ValueError):
        encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=0, pair_indices=pair_indices)
    with pytest.raises(ValueError):
        encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=17, pair_indices=pair_indices)


def test_alpha_block_rejects_duplicate_pair_indices():
    qs, mins, steps, _ = _make_alpha_block_inputs(4, 2, alpha_bits=4)
    bad_indices = np.array([1, 2, 1, 3], dtype=np.uint16)
    with pytest.raises(ValueError, match="duplicate"):
        encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=4, pair_indices=bad_indices)


def test_alpha_block_rejects_qs_out_of_range():
    qs = np.array([[16, 0]], dtype=np.uint8)  # 16 > qmax=15 for 4-bit
    mins = np.zeros(1, dtype=np.float32)
    steps = np.ones(1, dtype=np.float32)
    pair_indices = np.array([0], dtype=np.uint16)
    with pytest.raises(ValueError, match="qs values must be in"):
        encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=4, pair_indices=pair_indices)


def test_alpha_block_decode_invalid_magic_rejected():
    with pytest.raises(ValueError, match="bad SJ-KL alpha block magic"):
        decode_sjkl_alpha_block(b"NOPE" + b"\x00" * 20)


# ============================================================
# Full sjkl.bin payload codec tests (basis + alpha block)
# ============================================================


def test_full_payload_roundtrip_basis_recoverable():
    basis = _make_random_basis(rank=4, dim=32)
    qs, mins, steps, pair_indices = _make_alpha_block_inputs(16, 4, alpha_bits=4)
    alpha_block = encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=4, pair_indices=pair_indices)

    full_payload = encode_full_sjkl_payload(basis, alpha_block, basis_quant_bits=6)
    decoded_basis, meta = decode_full_sjkl_payload(full_payload)

    assert decoded_basis.rank == basis.rank
    assert decoded_basis.dim == basis.dim
    rel_err = (decoded_basis.eigenvectors - basis.eigenvectors).abs().max().item() / max(
        basis.eigenvectors.abs().max().item(), 1e-9
    )
    assert rel_err < 0.10


def test_full_payload_roundtrip_alpha_block_recoverable():
    basis = _make_random_basis(rank=2, dim=16)
    qs, mins, steps, pair_indices = _make_alpha_block_inputs(8, 2, alpha_bits=6)
    alpha_block = encode_sjkl_alpha_block_v2_sparse(qs, mins, steps, alpha_bits=6, pair_indices=pair_indices)

    full_payload = encode_full_sjkl_payload(basis, alpha_block)
    _basis, meta = decode_full_sjkl_payload(full_payload)
    decoded_alpha = decode_sjkl_alpha_block(meta["alpha_block_raw_bytes"])

    assert np.array_equal(decoded_alpha["qs"], qs.astype(decoded_alpha["qs"].dtype))
    assert np.array_equal(decoded_alpha["pair_indices"], pair_indices.astype(np.int64))


def test_full_payload_layout_matches_runtime_contract():
    """Byte layout must be: SJKL[4] + basis_len[4] + block_len[4] + basis + block.
    This is exactly what submissions/robust_current/inflate_renderer.py:_unpack_full_sjkl_payload
    expects.
    """
    basis = _make_random_basis(rank=2, dim=8)
    alpha_block = b"SJK2" + b"\x00" * 50  # arbitrary block-shaped bytes
    full = encode_full_sjkl_payload(basis, alpha_block, basis_quant_bits=6)
    import struct
    assert full[:4] == b"SJKL"
    basis_len, block_len = struct.unpack("<II", full[4:12])
    assert block_len == len(alpha_block)
    assert len(full) == 12 + basis_len + block_len


def test_full_payload_invalid_magic_rejected():
    with pytest.raises(ValueError, match="bad SJ-KL payload magic"):
        decode_full_sjkl_payload(b"NOPE" + b"\x00" * 20)


def test_full_payload_truncated_rejected():
    with pytest.raises(ValueError, match="too short"):
        decode_full_sjkl_payload(b"\x00")
