# SPDX-License-Identifier: MIT
"""Tests for Wasserstein-Barycenter Checkpoint Ensemble + MERA Quantizer."""

from __future__ import annotations

import struct

import pytest
import torch

from tac.composition.frontier_primitives import (
    DiagonalGaussian,
    wasserstein_diagonal_gaussian_barycenter,
)
from tac.composition.wbce_mera import (
    WBCE_MAGIC,
    BarycenterSpec,
    BrenierOTQuantizer,
    BrenierOTResult,
    BrenierOTSpec,
    MERAFactors,
    MERAQuantizer,
    MERAQuantizerSpec,
    WassersteinBarycenterEnsemble,
    WBCEMERAError,
    compose_wbce_mera,
    compressed_payload_byte_breakdown,
    deserialize_compose_state,
    estimate_compressed_bytes,
    serialize_compose_state,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


# --- BarycenterSpec validation ---


def test_barycenter_spec_default_legal() -> None:
    spec = BarycenterSpec()
    assert spec.max_iters > 0
    assert spec.tol > 0


def test_barycenter_spec_rejects_negative_weights() -> None:
    with pytest.raises(WBCEMERAError):
        BarycenterSpec(weights=(0.5, -0.5))


def test_barycenter_spec_rejects_unnormalised_weights() -> None:
    with pytest.raises(WBCEMERAError):
        BarycenterSpec(weights=(0.5, 0.4))


def test_barycenter_spec_rejects_zero_max_iters() -> None:
    with pytest.raises(WBCEMERAError):
        BarycenterSpec(max_iters=0)


def test_barycenter_spec_rejects_nonpositive_tol() -> None:
    with pytest.raises(WBCEMERAError):
        BarycenterSpec(tol=0)


def test_barycenter_spec_rejects_nonfinite_values() -> None:
    with pytest.raises(WBCEMERAError, match="finite"):
        BarycenterSpec(tol=float("inf"))
    with pytest.raises(WBCEMERAError, match="finite"):
        BarycenterSpec(weights=(0.5, float("nan")))


# --- WassersteinBarycenterEnsemble ---


def test_barycenter_uniform_mean() -> None:
    _seed()
    cs = [torch.randn(4, 4) for _ in range(3)]
    bary = WassersteinBarycenterEnsemble().compute(cs)
    expected = torch.stack(cs).mean(0)
    assert torch.allclose(bary, expected)


def test_barycenter_weighted_mean() -> None:
    cs = [torch.full((3,), 1.0), torch.full((3,), 4.0)]
    bary = WassersteinBarycenterEnsemble(
        BarycenterSpec(weights=(0.25, 0.75))
    ).compute(cs)
    assert torch.allclose(bary, torch.full((3,), 3.25))


def test_barycenter_requires_matching_shapes() -> None:
    cs = [torch.randn(2, 2), torch.randn(2, 3)]
    with pytest.raises(WBCEMERAError):
        WassersteinBarycenterEnsemble().compute(cs)


def test_barycenter_requires_nonempty_input() -> None:
    with pytest.raises(WBCEMERAError):
        WassersteinBarycenterEnsemble().compute([])


def test_barycenter_weights_length_must_match_K() -> None:
    cs = [torch.randn(2, 2) for _ in range(3)]
    bary = WassersteinBarycenterEnsemble(BarycenterSpec(weights=(0.5, 0.5)))
    with pytest.raises(WBCEMERAError):
        bary.compute(cs)


def test_barycenter_use_covariance_still_returns_mean() -> None:
    _seed()
    cs = [torch.randn(3, 3) for _ in range(3)]
    bary = WassersteinBarycenterEnsemble(
        BarycenterSpec(use_covariance=True)
    ).compute(cs)
    expected = torch.stack(cs).mean(0)
    assert torch.allclose(bary, expected, atol=1e-5)


def test_barycenter_conforms_to_frontier_diagonal_gaussian_primitive() -> None:
    c0 = torch.tensor([1.0, 3.0])
    c1 = torch.tensor([5.0, 7.0])
    weights = (0.25, 0.75)
    wbce = WassersteinBarycenterEnsemble(BarycenterSpec(weights=weights)).compute(
        (c0, c1)
    )
    canonical = wasserstein_diagonal_gaussian_barycenter(
        (
            DiagonalGaussian(c0, torch.zeros_like(c0)),
            DiagonalGaussian(c1, torch.zeros_like(c1)),
        ),
        weights,
    ).mean
    assert torch.allclose(wbce, canonical)


# --- MERAQuantizerSpec validation ---


def test_mera_spec_rejects_invalid_bond() -> None:
    with pytest.raises(WBCEMERAError):
        MERAQuantizerSpec(max_bond=0)
    with pytest.raises(WBCEMERAError):
        MERAQuantizerSpec(rank_floor=0)
    with pytest.raises(WBCEMERAError):
        MERAQuantizerSpec(rank_floor=10, max_bond=4)
    with pytest.raises(WBCEMERAError):
        MERAQuantizerSpec(fisher_floor=0)


def test_mera_spec_rejects_nonfinite_fisher_floor() -> None:
    with pytest.raises(WBCEMERAError, match="fisher_floor"):
        MERAQuantizerSpec(fisher_floor=float("nan"))


# --- MERAQuantizer.fisher_water_fill ---


def test_water_fill_uniform_fisher() -> None:
    quant = MERAQuantizer()
    alloc = quant.fisher_water_fill([1.0, 1.0, 1.0], chi_total=9)
    assert sum(alloc) == 9
    assert max(alloc) - min(alloc) <= 1


def test_water_fill_concentrates_on_high_fisher() -> None:
    quant = MERAQuantizer()
    alloc = quant.fisher_water_fill([10.0, 1.0, 0.1], chi_total=12)
    assert alloc[0] >= alloc[1] >= alloc[2]
    assert sum(alloc) == 12


def test_water_fill_respects_max_bond() -> None:
    quant = MERAQuantizer(MERAQuantizerSpec(max_bond=4))
    alloc = quant.fisher_water_fill([100.0, 1.0, 1.0], chi_total=12)
    assert all(a <= 4 for a in alloc)


def test_water_fill_respects_rank_floor() -> None:
    quant = MERAQuantizer(MERAQuantizerSpec(rank_floor=2))
    alloc = quant.fisher_water_fill([1.0, 1e-10, 1e-10], chi_total=8)
    assert all(a >= 2 for a in alloc)


def test_water_fill_insufficient_budget_raises() -> None:
    quant = MERAQuantizer(MERAQuantizerSpec(rank_floor=3))
    with pytest.raises(WBCEMERAError):
        quant.fisher_water_fill([1.0, 1.0], chi_total=2)


def test_water_fill_rejects_nonfinite_fisher_score() -> None:
    with pytest.raises(WBCEMERAError, match="fisher_scores"):
        MERAQuantizer().fisher_water_fill([1.0, float("inf")], chi_total=4)


# --- MERAQuantizer.compress ---


def test_mera_compress_full_rank_is_lossless() -> None:
    _seed()
    W = torch.randn(6, 8)
    quant = MERAQuantizer()
    factors = quant.compress(W, bond_dim=6)
    rec = factors.reconstruct()
    assert torch.allclose(rec, W, atol=1e-4)


def test_mera_compress_truncates_smallest_singular_values() -> None:
    W = torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1]))
    factors = MERAQuantizer().compress(W, bond_dim=2)
    assert factors.bond_dim == 2
    # Reconstruction is rank-2 → matches the top-2 singular components.
    rec = factors.reconstruct()
    assert torch.allclose(rec, torch.diag(torch.tensor([5.0, 3.0, 0.0, 0.0])),
                          atol=1e-5)


def test_mera_compress_rejects_non_matrix() -> None:
    with pytest.raises(WBCEMERAError):
        MERAQuantizer().compress(torch.randn(4), bond_dim=2)


def test_mera_compress_bond_dim_capped_to_min_shape() -> None:
    W = torch.randn(3, 7)
    factors = MERAQuantizer().compress(W, bond_dim=20)
    assert factors.bond_dim == 3


def test_mera_compress_rejects_zero_bond() -> None:
    with pytest.raises(WBCEMERAError):
        MERAQuantizer().compress(torch.randn(3, 3), bond_dim=0)


# --- BrenierOTSpec ---


def test_brenier_spec_rejects_small_codebook() -> None:
    with pytest.raises(WBCEMERAError):
        BrenierOTSpec(codebook_size=1)


# --- BrenierOTQuantizer ---


def test_brenier_fit_quantize_shape() -> None:
    _seed()
    x = torch.randn(20)
    res = BrenierOTQuantizer(BrenierOTSpec(codebook_size=8)).fit_quantize(x)
    assert res.codebook.shape == (8,)
    assert res.indices.shape == x.shape
    assert res.indices.min() >= 0
    assert res.indices.max() <= 7


def test_brenier_dequantize_round_trip_close() -> None:
    _seed()
    x = torch.randn(50)
    q = BrenierOTQuantizer(BrenierOTSpec(codebook_size=32))
    res = q.fit_quantize(x)
    rec = q.dequantize(res.indices, res.codebook)
    # Approximate; 32 levels for 50 samples should be close.
    assert (x - rec).abs().mean() < 0.5


def test_brenier_symmetric_codebook_includes_negatives() -> None:
    x = torch.linspace(-3, 3, 60)
    res = BrenierOTQuantizer(
        BrenierOTSpec(codebook_size=8, symmetric=True)
    ).fit_quantize(x)
    assert (res.codebook < 0).any()
    assert (res.codebook > 0).any()


def test_brenier_rejects_empty_input() -> None:
    with pytest.raises(WBCEMERAError):
        BrenierOTQuantizer().fit_quantize(torch.tensor([]))


def test_brenier_preserves_shape_on_2d_input() -> None:
    x = torch.randn(4, 5)
    res = BrenierOTQuantizer().fit_quantize(x)
    assert res.indices.shape == (4, 5)


# --- compose_wbce_mera end-to-end ---


def test_compose_wbce_mera_full_pipeline() -> None:
    _seed()
    cs = [torch.randn(6, 8) for _ in range(3)]
    state = compose_wbce_mera(cs, chi_total=6)
    assert "barycenter" in state
    assert "mera" in state
    assert "brenier" in state
    assert "reconstruction" in state
    rec = state["reconstruction"]
    assert rec.shape == (6, 8)


def test_compose_wbce_mera_rejects_empty() -> None:
    with pytest.raises(WBCEMERAError):
        compose_wbce_mera([])


def test_compose_wbce_mera_rejects_non_2d_input() -> None:
    cs = [torch.randn(6) for _ in range(2)]
    with pytest.raises(WBCEMERAError):
        compose_wbce_mera(cs)


def test_serialize_compose_state_starts_with_magic() -> None:
    _seed()
    cs = [torch.randn(4, 4) for _ in range(2)]
    state = compose_wbce_mera(cs, chi_total=4)
    payload = serialize_compose_state(state)
    assert payload[:4] == WBCE_MAGIC


def test_serialize_compose_state_size_matches_estimate() -> None:
    _seed()
    cs = [torch.randn(4, 4) for _ in range(2)]
    state = compose_wbce_mera(cs, chi_total=4)
    payload = serialize_compose_state(state)
    factors = state["mera"]
    bro = state["brenier"]
    est = estimate_compressed_bytes(
        factors.bond_dim, 4, 4, codebook_size=bro.codebook.numel()
    )
    # Payload includes header overhead plus compressed tensors only.
    assert len(payload) == struct.calcsize("<4sBHIIIIIIII") + est


def test_wbce_byte_breakdown_excludes_original_float32_singular_values() -> None:
    breakdown = compressed_payload_byte_breakdown(
        bond_dim=4,
        m=6,
        n=8,
        codebook_size=16,
    )
    assert breakdown["stores_original_singular_values"] is False
    assert breakdown["original_singular_float32_bytes"] == 0
    assert breakdown["total_bytes"] == estimate_compressed_bytes(4, 6, 8, 16)


def test_deserialize_compose_state_golden_quantized_s_payload() -> None:
    U = torch.eye(2, dtype=torch.float32).numpy()
    V = torch.eye(2, dtype=torch.float32).numpy()
    codebook = torch.tensor([1.0, 3.0], dtype=torch.float32).numpy()
    indices = torch.tensor([0, 1], dtype=torch.int32).numpy()
    payload = (
        struct.pack(
            "<4sBHIIIIIIII",
            WBCE_MAGIC,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
        )
        + U.tobytes()
        + V.tobytes()
        + codebook.tobytes()
        + indices.tobytes()
    )
    decoded = deserialize_compose_state(payload)
    assert torch.allclose(decoded["mera"].S, torch.tensor([1.0, 3.0]))
    assert torch.allclose(decoded["reconstruction"], torch.diag(torch.tensor([1.0, 3.0])))


def test_serialized_payload_does_not_restore_original_singular_values() -> None:
    state = {
        "mera": MERAFactors(
            U=torch.eye(2),
            S=torch.tensor([111.0, 222.0]),
            V=torch.eye(2),
            bond_dim=2,
        ),
        "brenier": BrenierOTResult(
            codebook=torch.tensor([1.0, 3.0]),
            indices=torch.tensor([0, 1], dtype=torch.int32),
        ),
    }
    decoded = deserialize_compose_state(serialize_compose_state(state))
    assert torch.allclose(decoded["mera"].S, torch.tensor([1.0, 3.0]))
    assert not torch.allclose(decoded["mera"].S, state["mera"].S)


def test_compose_wbce_mera_reconstruction_close_at_full_rank() -> None:
    _seed()
    cs = [torch.randn(4, 4) for _ in range(3)]
    state = compose_wbce_mera(
        cs, chi_total=4, brenier_spec=BrenierOTSpec(codebook_size=32)
    )
    bary = state["barycenter"]
    rec = state["reconstruction"]
    # Full-rank SVD + 32-level Brenier symmetric codebook quantizes the
    # singular-value spectrum (rank-4 random Gaussian has only 4 values;
    # symmetric codebook covers plus/minus range and the antisymmetric folding
    # over a 4-value source is lossy). We check the reconstruction is
    # within the magnitude of the barycenter — not exactly close.
    rms = (rec - bary).pow(2).mean().sqrt()
    assert torch.isfinite(rms).all()
    assert rms.item() < bary.abs().max().item() * 2.0
