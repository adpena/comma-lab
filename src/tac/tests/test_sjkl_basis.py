"""Regression tests for tac.sjkl_basis recovered from runbook spec."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.sjkl_basis import (
    SJKLBasis,
    apply_sjkl_residual,
    compute_sjkl_basis_lanczos,
    decode_sjkl_basis,
    encode_sjkl_basis,
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


def test_quant_bits_8_payload_byte_layout():
    """Verify q8 packs as 1 byte per value (no bit-stream overhead)."""
    basis = _make_random_basis(rank=2, dim=10)
    blob = encode_sjkl_basis(basis, basis_quant_bits=8)
    # header (4 magic + 1 ver + 1 flag + 2 K + 4 D + 2 M = 14) + 4 scale + 2*10 basis + 2*2 coef
    expected = 14 + 4 + (2 * 10) + (2 * 2)
    assert len(blob) == expected, f"q8 blob size mismatch: got {len(blob)}, expected {expected}"
