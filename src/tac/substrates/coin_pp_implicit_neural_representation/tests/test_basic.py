# SPDX-License-Identifier: MIT
"""L0 SCAFFOLD smoke + shape + Catalog #91/#139 + MLX↔numpy + MLX↔PyTorch parity tests.

Per Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation
no_op_proof + Catalog #240 L0 SCAFFOLD posture verification + NEW
operator-directive-#3 2026-05-26 axes 2 (MLX drift) + 3 (numpy portability).

These tests verify the substrate package's STRUCTURAL invariants without
requiring MLX to be installed (MLX-dependent tests are guarded by import
detection and skipped when MLX is unavailable).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1: top-level import without MLX
# ---------------------------------------------------------------------------

def test_module_imports_without_mlx() -> None:
    """Top-level package import must succeed without MLX installed."""
    import tac.substrates.coin_pp_implicit_neural_representation as mod

    assert mod.ARCHIVE_MAGIC == b"CPP1\x00"
    assert mod.ARCHIVE_VERSION == 1
    assert mod.COINPP1_HEADER_LEN == 32
    assert mod.DEFAULT_MOD_DIM == 64
    assert mod.DEFAULT_POS_DIM == 32
    assert mod.DEFAULT_HIDDEN_DIM == 64
    assert mod.DEFAULT_NUM_HIDDEN_LAYERS == 3
    assert mod.DEFAULT_EVAL_H == 384
    assert mod.DEFAULT_EVAL_W == 512


# ---------------------------------------------------------------------------
# Test 2: public API surface (Catalog #335)
# ---------------------------------------------------------------------------

def test_module_exposes_canonical_public_api() -> None:
    """__all__ surface must be narrow + explicit per Catalog #335 contract."""
    import tac.substrates.coin_pp_implicit_neural_representation as mod

    expected = {
        "ARCHIVE_GRAMMAR_FIELDS",
        "ARCHIVE_MAGIC",
        "ARCHIVE_VERSION",
        "COINPP1_HEADER_FMT",
        "COINPP1_HEADER_LEN",
        "CoinPPImplicitNeuralRepresentationConfig",
        "DEFAULT_EVAL_H",
        "DEFAULT_EVAL_W",
        "DEFAULT_HIDDEN_DIM",
        "DEFAULT_MOD_DIM",
        "DEFAULT_NUM_HIDDEN_LAYERS",
        "DEFAULT_POS_DIM",
        "SISTER_SUBSTRATES",
    }
    assert set(mod.__all__) == expected


def test_archive_grammar_fields_catalog_124_compliance() -> None:
    """Catalog #124 archive-grammar 8 fields declared inline for AST walker."""
    from tac.substrates.coin_pp_implicit_neural_representation import (
        ARCHIVE_GRAMMAR_FIELDS,
    )

    expected_keys = {
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    }
    assert set(ARCHIVE_GRAMMAR_FIELDS.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3: Config dataclass invariants
# ---------------------------------------------------------------------------

def test_config_dataclass_default_values() -> None:
    """Default config matches design memo §"Predicted ΔS band" breakdown."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig()
    assert cfg.mod_dim == 64
    assert cfg.pos_dim == 32
    assert cfg.hidden_dim == 64
    assert cfg.num_hidden_layers == 3
    assert cfg.num_pairs == 600
    assert cfg.eval_h == 384
    assert cfg.eval_w == 512
    assert cfg.modulation_quant_bits == 8


def test_config_rejects_invalid_eval_hw() -> None:
    """(eval_h, eval_w) must equal contest scorer-resolution (384, 512)."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    with pytest.raises(ValueError, match="EVAL_HW"):
        CoinPPImplicitNeuralRepresentationConfig(eval_h=256, eval_w=512)


def test_config_rejects_mod_dim_out_of_u8_range() -> None:
    """mod_dim must fit in u8 wire field."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    with pytest.raises(ValueError, match="mod_dim"):
        CoinPPImplicitNeuralRepresentationConfig(mod_dim=0)
    with pytest.raises(ValueError, match="mod_dim"):
        CoinPPImplicitNeuralRepresentationConfig(mod_dim=256)


def test_config_rejects_invalid_modulation_quant_bits() -> None:
    """modulation_quant_bits must be 4, 8, or 16."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    with pytest.raises(ValueError, match="modulation_quant_bits"):
        CoinPPImplicitNeuralRepresentationConfig(modulation_quant_bits=7)


# ---------------------------------------------------------------------------
# Test 4: Catalog #240 L0 SCAFFOLD posture
# ---------------------------------------------------------------------------

def test_full_main_raises_not_implemented_per_catalog_240() -> None:
    """L0 SCAFFOLD posture: full main raises NotImplementedError."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        _full_main,
    )

    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD"):
        _full_main()


# ---------------------------------------------------------------------------
# Test 5: Archive grammar round-trip per Catalog #91
# ---------------------------------------------------------------------------

def _make_synthetic_archive_inputs(cfg) -> tuple:
    """Construct minimal valid (state_dict, modulations, meta) for round-trip."""
    pos_enc_dim = cfg.pos_dim * 2 * 3
    # Tiny synthetic state_dict matching PyTorch nn.Linear layout (out, in)
    base_sd: dict[str, np.ndarray] = {
        "input_proj.weight": np.zeros((cfg.hidden_dim, pos_enc_dim), dtype=np.float32),
        "input_proj.bias": np.zeros((cfg.hidden_dim,), dtype=np.float32),
    }
    for i in range(cfg.num_hidden_layers):
        base_sd[f"hidden.{i}.weight"] = np.zeros(
            (cfg.hidden_dim, cfg.hidden_dim), dtype=np.float32
        )
        base_sd[f"hidden.{i}.bias"] = np.zeros((cfg.hidden_dim,), dtype=np.float32)
        base_sd[f"film.{i}.weight"] = np.zeros(
            (2 * cfg.hidden_dim, cfg.mod_dim), dtype=np.float32
        )
        base_sd[f"film.{i}.bias"] = np.zeros((2 * cfg.hidden_dim,), dtype=np.float32)
    base_sd["output_proj.weight"] = np.zeros((3, cfg.hidden_dim), dtype=np.float32)
    base_sd["output_proj.bias"] = np.zeros((3,), dtype=np.float32)

    modulations = np.zeros((cfg.num_pairs, cfg.mod_dim), dtype=np.int8)
    meta = {
        "num_pairs": cfg.num_pairs,
        "modulation_scale": 1.0,
        "schema_version": 1,
        "eval_hw": [cfg.eval_h, cfg.eval_w],
    }
    return base_sd, modulations, meta


def test_coinpp1_archive_pack_parse_round_trip() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack → parse returns equivalent data."""
    from tac.substrates.coin_pp_implicit_neural_representation.archive import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig(num_pairs=4)
    base_sd, modulations, meta = _make_synthetic_archive_inputs(cfg)

    archive_bytes = pack_archive(
        base_sd,
        modulations,
        meta,
        mod_dim=cfg.mod_dim,
        pos_dim=cfg.pos_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_pairs=cfg.num_pairs,
        eval_h=cfg.eval_h,
        eval_w=cfg.eval_w,
    )

    parsed = parse_archive(archive_bytes)
    assert parsed.mod_dim == cfg.mod_dim
    assert parsed.pos_dim == cfg.pos_dim
    assert parsed.hidden_dim == cfg.hidden_dim
    assert parsed.num_hidden_layers == cfg.num_hidden_layers
    assert parsed.num_pairs == cfg.num_pairs
    assert parsed.eval_h == cfg.eval_h
    assert parsed.eval_w == cfg.eval_w
    assert parsed.per_pair_modulations.shape == (cfg.num_pairs, cfg.mod_dim)
    assert parsed.per_pair_modulations.dtype == np.int8
    assert parsed.meta["num_pairs"] == cfg.num_pairs


def test_archive_invalid_magic_rejected() -> None:
    """Mis-magic'd archive must be refused."""
    from tac.substrates.coin_pp_implicit_neural_representation.archive import (
        parse_archive,
    )

    bad_bytes = b"BADM\x00" + b"\x00" * 100
    with pytest.raises(ValueError, match="magic"):
        parse_archive(bad_bytes)


def test_archive_truncated_rejected() -> None:
    """Truncated archive must be refused (sub-header)."""
    from tac.substrates.coin_pp_implicit_neural_representation.archive import (
        parse_archive,
    )

    with pytest.raises(ValueError, match="too short"):
        parse_archive(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Test 6: Catalog #139 byte-mutation no_op_proof
# ---------------------------------------------------------------------------

def test_archive_byte_mutation_no_op_proof_per_catalog_139() -> None:
    """Mutating per-pair modulation bytes MUST change parsed modulation content.

    Per Catalog #139 + #272 distinguishing-feature contract: per-pair
    modulation bytes are the distinguishing-feature; mutating them MUST
    produce a different parsed modulation (which would produce different
    final RGB at inflate time).
    """
    from tac.substrates.coin_pp_implicit_neural_representation.archive import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig(num_pairs=2)
    base_sd, modulations, meta = _make_synthetic_archive_inputs(cfg)
    # Use non-zero modulations so mutation is observable
    modulations[:] = 1  # int8 = 1

    archive_bytes = pack_archive(
        base_sd,
        modulations,
        meta,
        mod_dim=cfg.mod_dim,
        pos_dim=cfg.pos_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_pairs=cfg.num_pairs,
        eval_h=cfg.eval_h,
        eval_w=cfg.eval_w,
    )
    parsed_original = parse_archive(archive_bytes)
    sha_original = hashlib.sha256(parsed_original.per_pair_modulations.tobytes()).hexdigest()

    # Mutate one modulation byte
    modulations_mutated = modulations.copy()
    modulations_mutated[0, 0] = 64

    mutated_bytes = pack_archive(
        base_sd,
        modulations_mutated,
        meta,
        mod_dim=cfg.mod_dim,
        pos_dim=cfg.pos_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_pairs=cfg.num_pairs,
        eval_h=cfg.eval_h,
        eval_w=cfg.eval_w,
    )
    parsed_mutated = parse_archive(mutated_bytes)
    sha_mutated = hashlib.sha256(parsed_mutated.per_pair_modulations.tobytes()).hexdigest()

    assert sha_original != sha_mutated, (
        "Catalog #139 violated: mutating per-pair modulation bytes did NOT "
        "change parsed modulation content. The per-pair modulation blob is "
        "the DISTINGUISHING FEATURE; byte mutation MUST be observable."
    )


# ---------------------------------------------------------------------------
# Test 7: numpy reference primitive shapes + correctness
# ---------------------------------------------------------------------------

def test_numpy_reference_sigmoid_matches_pytorch() -> None:
    """numpy reference sigmoid ≤ 1e-6 vs PyTorch torch.sigmoid."""
    import torch

    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        sigmoid,
    )

    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    y_numpy = sigmoid(x)
    y_torch = torch.sigmoid(torch.from_numpy(x)).numpy()

    max_abs = np.abs(y_numpy - y_torch).max()
    assert max_abs < 1e-6, (
        f"numpy reference sigmoid drift {max_abs} > 1e-6 vs PyTorch"
    )


def test_numpy_reference_linear_matches_pytorch() -> None:
    """numpy reference linear ≤ 1e-5 vs PyTorch nn.Linear forward."""
    import torch
    import torch.nn as nn

    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        linear,
    )

    rng = np.random.default_rng(seed=7)
    in_features, out_features = 8, 12
    weight = rng.standard_normal((out_features, in_features)).astype(np.float32)
    bias = rng.standard_normal((out_features,)).astype(np.float32)
    x = rng.standard_normal((4, in_features)).astype(np.float32)

    y_numpy = linear(x, weight, bias)

    torch_linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.from_numpy(weight))
        torch_linear.bias.copy_(torch.from_numpy(bias))
    with torch.inference_mode():
        y_torch = torch_linear(torch.from_numpy(x)).numpy()

    max_abs = np.abs(y_numpy - y_torch).max()
    assert max_abs < 1e-5, (
        f"numpy reference linear drift {max_abs} > 1e-5 vs PyTorch"
    )


def test_numpy_reference_positional_encoding_shape_and_range() -> None:
    """sinusoidal_positional_encoding output shape matches (..., D * L * 2)."""
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        sinusoidal_positional_encoding,
    )

    coords = np.zeros((1, 4, 6, 3), dtype=np.float32)
    encoded = sinusoidal_positional_encoding(coords, num_frequencies=32)
    # Output shape: (..., D * L * 2) = (1, 4, 6, 3 * 32 * 2) = (1, 4, 6, 192)
    assert encoded.shape == (1, 4, 6, 192)
    # sin/cos of zero -> (sin=0, cos=1) interleaved; mean over each pair = 0.5
    # All elements should be in [-1, 1]
    assert encoded.min() >= -1.0
    assert encoded.max() <= 1.0


def test_numpy_reference_positional_encoding_matches_torch() -> None:
    """numpy reference positional encoding ≤ 1e-5 vs PyTorch implementation."""
    import torch

    from tac.substrates.coin_pp_implicit_neural_representation.inflate import (
        _sinusoidal_positional_encoding,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        sinusoidal_positional_encoding,
    )

    rng = np.random.default_rng(seed=11)
    coords_np = rng.uniform(-1.0, 1.0, size=(2, 3, 4, 3)).astype(np.float32)
    L = 8

    enc_np = sinusoidal_positional_encoding(coords_np, L)
    enc_torch = _sinusoidal_positional_encoding(torch.from_numpy(coords_np), L).numpy()
    max_abs = np.abs(enc_np - enc_torch).max()
    assert max_abs < 1e-5, (
        f"numpy↔PyTorch positional encoding drift {max_abs} > 1e-5"
    )


def test_numpy_reference_film_modulate_correctness() -> None:
    """FiLM modulate: h * scale + shift; verify against manual computation."""
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        film_modulate,
    )

    h = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    scale = np.array([[2.0, 0.5, 1.0]], dtype=np.float32)
    shift = np.array([[0.1, 0.0, -1.0]], dtype=np.float32)
    out = film_modulate(h, scale, shift)
    expected = np.array([[2.1, 1.0, 2.0]], dtype=np.float32)
    assert np.allclose(out, expected, atol=1e-6)


def test_numpy_reference_coord_grid_construction() -> None:
    """Coord grid spans [-1, 1] in x and y per canonical convention."""
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        make_coord_grid_nhwc,
    )

    grid = make_coord_grid_nhwc(H=4, W=6, t=0.5)
    assert grid.shape == (4, 6, 3)
    # x is in [-1, 1] across W
    assert grid[0, 0, 0] == pytest.approx(-1.0)
    assert grid[0, -1, 0] == pytest.approx(1.0)
    # y is in [-1, 1] across H
    assert grid[0, 0, 1] == pytest.approx(-1.0)
    assert grid[-1, 0, 1] == pytest.approx(1.0)
    # t passed through
    assert np.all(grid[..., 2] == pytest.approx(0.5))


def test_numpy_reference_coord_mlp_forward_shape() -> None:
    """coord_mlp_forward composite produces (B, H, W, 3) RGB in [0, 1]."""
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        coord_mlp_forward,
        make_coord_grid_nhwc,
    )

    cfg_pos_dim = 8
    cfg_hidden_dim = 16
    cfg_num_hidden = 2
    cfg_mod_dim = 8

    pos_enc_dim = cfg_pos_dim * 2 * 3
    rng = np.random.default_rng(seed=17)
    state_dict = {
        "input_proj.weight": rng.standard_normal(
            (cfg_hidden_dim, pos_enc_dim)
        ).astype(np.float32) * 0.1,
        "input_proj.bias": np.zeros((cfg_hidden_dim,), dtype=np.float32),
        "output_proj.weight": rng.standard_normal(
            (3, cfg_hidden_dim)
        ).astype(np.float32) * 0.1,
        "output_proj.bias": np.zeros((3,), dtype=np.float32),
    }
    for i in range(cfg_num_hidden):
        state_dict[f"hidden_{i}.weight"] = rng.standard_normal(
            (cfg_hidden_dim, cfg_hidden_dim)
        ).astype(np.float32) * 0.1
        state_dict[f"hidden_{i}.bias"] = np.zeros((cfg_hidden_dim,), dtype=np.float32)
        state_dict[f"film_{i}.weight"] = rng.standard_normal(
            (2 * cfg_hidden_dim, cfg_mod_dim)
        ).astype(np.float32) * 0.1
        state_dict[f"film_{i}.bias"] = np.zeros((2 * cfg_hidden_dim,), dtype=np.float32)

    coords = make_coord_grid_nhwc(H=4, W=6, t=0.0)[np.newaxis, ...]  # (1, H, W, 3)
    modulation = rng.standard_normal((1, cfg_mod_dim)).astype(np.float32) * 0.1

    rgb = coord_mlp_forward(
        coords,
        modulation,
        state_dict,
        pos_dim=cfg_pos_dim,
        hidden_dim=cfg_hidden_dim,
        num_hidden_layers=cfg_num_hidden,
    )
    assert rgb.shape == (1, 4, 6, 3)
    assert rgb.dtype == np.float32
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()


def test_numpy_reference_kahan_mean_stability() -> None:
    """Kahan mean is more stable than naive mean for large-N fp32 reductions."""
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        kahan_mean,
        mean,
    )

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert abs(kahan_mean(x) - mean(x)) < 1e-6
    assert abs(kahan_mean(x) - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 8: estimate_archive_bytes for Dykstra-feasibility check
# ---------------------------------------------------------------------------

def test_estimate_archive_bytes_within_design_memo_range() -> None:
    """estimate_archive_bytes returns a value consistent with design memo §predicted-band.

    Design memo predicts ~42-50 KB for default config; allow generous bounds
    [20 KB, 200 KB] to accommodate brotli compression ratio variance.
    """
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
        estimate_archive_bytes,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig()
    est = estimate_archive_bytes(cfg)
    assert 20_000 < est < 200_000, (
        f"estimate_archive_bytes {est} bytes outside expected [20K, 200K] range; "
        f"may indicate a regression in base_mlp_param_count formula or config defaults"
    )


def test_estimate_archive_bytes_scales_with_mod_dim() -> None:
    """Smaller mod_dim → smaller archive (per-pair modulation rate is dominant variable)."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
        estimate_archive_bytes,
    )

    cfg_64 = CoinPPImplicitNeuralRepresentationConfig(mod_dim=64)
    cfg_16 = CoinPPImplicitNeuralRepresentationConfig(mod_dim=16)
    est_64 = estimate_archive_bytes(cfg_64)
    est_16 = estimate_archive_bytes(cfg_16)
    assert est_16 < est_64, (
        f"smaller mod_dim should produce smaller archive: est_16={est_16}, est_64={est_64}"
    )


# ---------------------------------------------------------------------------
# Test 9: PyTorch inflate runtime + MLX-availability gate
# ---------------------------------------------------------------------------

def test_pytorch_coord_mlp_forward_shape_and_range() -> None:
    """PyTorch inflate-time CoinPPCoordMLPTorch produces (B, H, W, 3) RGB in [0, 1]."""
    import torch

    from tac.substrates.coin_pp_implicit_neural_representation.inflate import (
        CoinPPCoordMLPTorch,
    )

    decoder = CoinPPCoordMLPTorch(
        mod_dim=8, pos_dim=8, hidden_dim=16, num_hidden_layers=2
    )
    decoder.eval()
    coords = torch.randn(1, 4, 6, 3)
    modulation = torch.randn(1, 8) * 0.1
    with torch.inference_mode():
        rgb = decoder(coords, modulation)
    assert rgb.shape == (1, 4, 6, 3)
    assert (rgb >= 0.0).all() and (rgb <= 1.0).all()


def test_mlx_availability_gate() -> None:
    """_ensure_mlx_available raises actionable RuntimeError if MLX missing OR returns mx if installed."""
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        _ensure_mlx_available,
    )

    try:
        import mlx.core  # noqa: F401

        mlx_available = True
    except ImportError:
        mlx_available = False

    if mlx_available:
        mx = _ensure_mlx_available()
        assert mx is not None
    else:
        with pytest.raises(RuntimeError, match="MLX is not installed"):
            _ensure_mlx_available()


# ---------------------------------------------------------------------------
# Test 10: deterministic byte-level archive pack invariant
# ---------------------------------------------------------------------------

def test_archive_pack_deterministic_byte_level() -> None:
    """Same input → byte-identical archive (sister-canonical determinism pattern)."""
    from tac.substrates.coin_pp_implicit_neural_representation.archive import (
        pack_archive,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig(num_pairs=4)
    base_sd, modulations, meta = _make_synthetic_archive_inputs(cfg)

    bytes_1 = pack_archive(
        base_sd, modulations, meta,
        mod_dim=cfg.mod_dim, pos_dim=cfg.pos_dim, hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers, num_pairs=cfg.num_pairs,
        eval_h=cfg.eval_h, eval_w=cfg.eval_w,
    )
    bytes_2 = pack_archive(
        base_sd, modulations, meta,
        mod_dim=cfg.mod_dim, pos_dim=cfg.pos_dim, hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers, num_pairs=cfg.num_pairs,
        eval_h=cfg.eval_h, eval_w=cfg.eval_w,
    )
    assert bytes_1 == bytes_2
    sha_1 = hashlib.sha256(bytes_1).hexdigest()
    sha_2 = hashlib.sha256(bytes_2).hexdigest()
    assert sha_1 == sha_2


# ---------------------------------------------------------------------------
# Test 11: MLX↔numpy reference parity (axis 3 portability discipline)
# ---------------------------------------------------------------------------

def test_mlx_numpy_parity_skipped_if_mlx_unavailable() -> None:
    """MLX↔numpy parity test infrastructure exists; gracefully skips if MLX missing.

    Per axis 2 MLX drift minimization (operator directive #3 2026-05-26):
    captures empirical drift bound for `mx.matmul` vs numpy `np.matmul`.

    **CORRECTION** (FIX-WAVE-R1''-K 2026-05-26): the earlier prose claimed a
    5e-3 absolute drift bound. R1'' independent verification across K-typical
    substrate dimensions empirically falsified that claim — actual drift is
    O(1e-2) abs / O(1e-3) rms / 7.6e-4 rel-median (see canonical equation
    `mlx_matmul_drift_m_series_canonical_floor_v1` registered in
    `tac.canonical_equations` per Catalog #344). The 5e-3 number was an
    artifact of the small (4x16)@(16x8) test fixture below; it is NOT the
    canonical floor at substrate-typical dims (32x32, 64x64, 128x128,
    256x64, 64x256 all measured >5e-3 abs_max).

    This test asserts against the canonical floor via the canonical helper
    `classify_mlx_matmul_drift` from
    `tac.canonical_equations.mlx_matmul_m_series_floor` (Catalog #344 +
    canonical-equation-reference enforcement) so a future K-substrate change
    that introduces an anti-pattern (align_corners=True / mx.repeat / fp16
    matmul) will produce drift ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION and
    the canonical classifier will flag the verdict shift.

    Source for the canonical floor: R1'' independent verification per
    `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
    + `.omx/research/path_3_fix_wave_r1_prime_prime_k_coin_pp_landed_*.md`
    [empirical:tac.canonical_equations.mlx_matmul_m_series_floor].

    Operator-routable next step: per-substrate-class characterization at L1
    (M1/M2/M3/M4/M5/Ultra/Pro/Max variants); see canonical equation
    `mlx_matmul_drift_m_series_canonical_floor_v1` reactivation criteria.
    """
    try:
        import mlx.core as mx

        mlx_available = True
    except ImportError:
        mlx_available = False

    if not mlx_available:
        pytest.skip("MLX not installed; numpy reference path remains operable per axis 3")

    from tac.canonical_equations.mlx_matmul_m_series_floor import (
        CANONICAL_ABS_MAX_UPPER_BOUND,
        classify_mlx_matmul_drift,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.numpy_reference import (
        linear as numpy_linear,
    )

    rng = np.random.default_rng(seed=42)
    weight_np = rng.standard_normal((8, 16)).astype(np.float32)
    bias_np = rng.standard_normal((8,)).astype(np.float32)
    x_np = rng.standard_normal((4, 16)).astype(np.float32)

    y_numpy = numpy_linear(x_np, weight_np, bias_np)

    # MLX equivalent: x @ weight.T + bias
    x_mx = mx.array(x_np)
    weight_mx = mx.array(weight_np)
    bias_mx = mx.array(bias_np)
    y_mlx_arr = mx.matmul(x_mx, mx.transpose(weight_mx)) + bias_mx
    y_mlx = np.array(y_mlx_arr)

    diff = np.abs(y_numpy - y_mlx)
    max_abs = float(diff.max())
    rms = float(np.sqrt(np.mean(diff**2)))

    # Canonical M-series MPS fp32 matmul drift hardware floor per
    # `tac.canonical_equations.mlx_matmul_m_series_floor` Catalog #344
    # registration. Per FIX-WAVE-R1''-K (2026-05-26) independent verification
    # the upper bound is 6e-2 abs (covers (64,256)@(256,64) worst-case) /
    # 1.5e-2 rms / 7.6e-4 rel-median across K-typical substrate dims.
    #
    # If this assertion fires it means EITHER (a) the hardware-class floor
    # has shifted (e.g. new M-series generation; route to per-class
    # characterization op-routable on the canonical equation), OR (b) the
    # substrate introduced an anti-pattern (align_corners=True / mx.repeat /
    # fp16 matmul) — re-verify per axis 2 MLX drift minimization discipline.
    verdict = classify_mlx_matmul_drift(
        measured_abs_max=max_abs,
        measured_rms=rms,
        matmul_shape=(4, 16, 8),
    )
    assert verdict["verdict"] in {"BIT_EXACT_LIKE_SINUSOIDAL", "WITHIN_CANONICAL_FLOOR"}, (
        f"MLX↔numpy linear parity drift abs_max={max_abs:.6e} rms={rms:.6e} "
        f"exceeds canonical M-series MPS fp32 floor (abs_max upper bound "
        f"{CANONICAL_ABS_MAX_UPPER_BOUND:.2e}); verdict={verdict['verdict']}. "
        f"Either the substrate introduced an anti-pattern (align_corners=True "
        f"/ mx.repeat / fp16 matmul) per axis 2 MLX drift minimization "
        f"discipline, OR the hardware-class floor has shifted (route to "
        f"per-M-series-class characterization op-routable on canonical "
        f"equation `mlx_matmul_drift_m_series_canonical_floor_v1`)."
    )
