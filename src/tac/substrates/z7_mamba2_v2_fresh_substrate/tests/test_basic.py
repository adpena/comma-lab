# SPDX-License-Identifier: MIT
"""L0 SCAFFOLD basic tests for Z7-Mamba-2-v2 fresh substrate.

Verifies the scaffold's structural contract:
- Package is RESEARCH_ONLY (Catalog #240 + #298 retirement discipline)
- Config validates per CC-D + CC-G + CC-F unwinds (a_log_init_scheme,
  training_backend)
- Substrate/Cell/Decoder skeleton classes refuse instantiation per
  CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
- Z7MCM3 archive grammar constants are well-formed
- Byte budget estimate is internally consistent with savings claim
- Phase 1/2/3 memo paths are accessible (cross-reference integrity)

Full architecture + archive + inflate tests land at L1 EMPIRICAL build
per the Phase 3 L0 SCAFFOLD design memo §7 + per Catalog #325 6-step
per-substrate symposium contract.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tac.substrates.z7_mamba2_v2_fresh_substrate import (
    DISPATCH_ENABLED,
    IMPLEMENTATION_STATUS,
    PHASE_1_AUDIT_PATH,
    PHASE_2_DECISION_PATH,
    PHASE_3_DESIGN_PATH,
    PLANNED_PUBLIC_API,
    RESEARCH_ONLY,
    SUBSTRATE_CLASS_SHIFT_HORIZON,
)
from tac.substrates.z7_mamba2_v2_fresh_substrate.architecture import (
    A_LOG_INIT_SCHEMES,
    EVAL_HW,
    NUM_PAIRS,
    TRAINING_BACKENDS,
    Mamba2TemporalDecoder,
    Mamba2V2Cell,
    Z7Mamba2V2Config,
    Z7Mamba2V2Substrate,
    normalize_a_log_init_scheme,
    normalize_training_backend,
)
from tac.substrates.z7_mamba2_v2_fresh_substrate.archive import (
    A_LOG_INIT_SCHEME_BYTE_ENUM,
    Z7MCM3_HEADER_FMT,
    Z7MCM3_HEADER_SIZE,
    Z7MCM3_MAGIC,
    Z7MCM3_SECTION_ROLES,
    Z7MCM3_VERSION,
    estimated_byte_budget,
    pack_archive,
    parse_archive,
    regenerate_a_log_from_init_scheme,
    replay_latent_sequence,
)
from tac.substrates.z7_mamba2_v2_fresh_substrate.inflate_runtime import (
    inflate_one_video,
    main_cli,
)


# === Package-level scaffold-only invariants ===


def test_substrate_is_research_only() -> None:
    """L0 SCAFFOLD MUST be research_only=True per Catalog #240."""
    assert RESEARCH_ONLY is True


def test_substrate_dispatch_disabled_at_L0() -> None:
    """L0 SCAFFOLD MUST be dispatch_disabled per Catalog #240."""
    assert DISPATCH_ENABLED is False


def test_implementation_status_signals_scaffold_only() -> None:
    """IMPLEMENTATION_STATUS surfaces scaffold-only state for autopilot ranker."""
    assert "scaffold_skeleton_only" in IMPLEMENTATION_STATUS
    assert "design_complete" in IMPLEMENTATION_STATUS


def test_horizon_class_declaration() -> None:
    """Per Catalog #309: horizon_class declared at L0."""
    assert SUBSTRATE_CLASS_SHIFT_HORIZON == "frontier_pursuit"


def test_phase_1_audit_path_exists() -> None:
    """Phase 1 cargo-cult audit memo must exist (cross-reference integrity)."""
    repo_root = Path(__file__).resolve().parents[5]
    assert (repo_root / PHASE_1_AUDIT_PATH).is_file()


def test_phase_2_decision_path_exists() -> None:
    """Phase 2 design-decision memo must exist (cross-reference integrity)."""
    repo_root = Path(__file__).resolve().parents[5]
    assert (repo_root / PHASE_2_DECISION_PATH).is_file()


def test_phase_3_design_path_exists() -> None:
    """Phase 3 L0 SCAFFOLD design memo must exist (cross-reference integrity)."""
    repo_root = Path(__file__).resolve().parents[5]
    assert (repo_root / PHASE_3_DESIGN_PATH).is_file()


def test_planned_public_api_non_empty() -> None:
    """PLANNED_PUBLIC_API declares the L1 public surface for downstream callers."""
    assert len(PLANNED_PUBLIC_API) >= 5
    assert "Z7Mamba2V2Config" in PLANNED_PUBLIC_API
    assert "Z7Mamba2V2Substrate" in PLANNED_PUBLIC_API


# === Config validates per CC-B + CC-C + CC-D + CC-G + CC-F + CC-H unwinds ===


def test_config_default_latent_dim_is_32_per_cc_b_unwind() -> None:
    """Default latent_dim=32 (was 24 in v1) per CC-B unwind."""
    config = Z7Mamba2V2Config()
    assert config.latent_dim == 32


def test_config_default_ego_motion_dim_is_16_per_cc_c_unwind() -> None:
    """Default ego_motion_dim=16 (was 8 in v1) per CC-C unwind."""
    config = Z7Mamba2V2Config()
    assert config.ego_motion_dim == 16


def test_config_default_ib_scale_is_5e_minus_4_per_cc_h_unwind() -> None:
    """Default ib_scale=5e-4 (was 1e-3 in v1) per CC-H unwind."""
    config = Z7Mamba2V2Config()
    assert config.ib_scale == 5e-4


def test_config_default_a_log_init_scheme_is_configurable_per_cc_d_unwind() -> None:
    """Default a_log_init_scheme is upstream Mamba-2 'z_plus_1'; configurable per CC-D."""
    config = Z7Mamba2V2Config()
    assert config.a_log_init_scheme == "z_plus_1"
    assert config.a_log_init_scheme in A_LOG_INIT_SCHEMES


def test_config_default_training_backend_is_mlx_per_cc_g_cc_f_unwind() -> None:
    """Default training_backend='mlx_native' per operator binding directive #1."""
    config = Z7Mamba2V2Config()
    assert config.training_backend == "mlx_native"
    assert config.training_backend in TRAINING_BACKENDS


def test_config_predictor_input_dim_property() -> None:
    """predictor_input_dim = latent_dim + ego_motion_dim per architecture spec."""
    config = Z7Mamba2V2Config()
    assert config.predictor_input_dim == config.latent_dim + config.ego_motion_dim
    assert config.predictor_input_dim == 32 + 16  # default values per CC-B + CC-C


def test_config_d_inner_property() -> None:
    """d_inner = expand * d_model per Mamba-2 §3."""
    config = Z7Mamba2V2Config()
    assert config.d_inner == config.expand * config.d_model
    assert config.d_inner == 2 * 64  # default values


def test_config_validates_latent_dim_positive() -> None:
    """Config rejects non-positive latent_dim."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        Z7Mamba2V2Config(latent_dim=0)


def test_config_validates_a_log_init_scheme_in_enum() -> None:
    """Config rejects invalid a_log_init_scheme per CC-D unwind enum."""
    with pytest.raises(ValueError, match="a_log_init_scheme must be one of"):
        Z7Mamba2V2Config(a_log_init_scheme="invalid_scheme")


def test_config_validates_training_backend_in_enum() -> None:
    """Config rejects invalid training_backend per CC-G + CC-F unwind enum."""
    with pytest.raises(ValueError, match="training_backend must be one of"):
        Z7Mamba2V2Config(training_backend="invalid_backend")


def test_config_validates_ib_scale_nonneg() -> None:
    """Config rejects negative ib_scale."""
    with pytest.raises(ValueError, match="ib_scale must be non-negative"):
        Z7Mamba2V2Config(ib_scale=-0.1)


def test_normalize_a_log_init_scheme_accepts_dash_variant() -> None:
    """normalize handles hyphen/underscore variants per canonical pattern."""
    assert normalize_a_log_init_scheme("z-plus-1") == "z_plus_1"
    assert normalize_a_log_init_scheme("Z_PLUS_1") == "z_plus_1"


def test_normalize_training_backend_accepts_dash_variant() -> None:
    """normalize handles hyphen/underscore variants per canonical pattern."""
    assert normalize_training_backend("ssd-scan-cuda") == "ssd_scan_cuda"
    assert normalize_training_backend("MLX-NATIVE") == "mlx_native"


def test_eval_hw_is_hard_earned_contest_resolution() -> None:
    """EVAL_HW is HARD-EARNED contest scorer resolution per CLAUDE.md."""
    assert EVAL_HW == (384, 512)


def test_num_pairs_is_hard_earned_contest_pair_count() -> None:
    """NUM_PAIRS is HARD-EARNED contest pair count per CLAUDE.md."""
    assert NUM_PAIRS == 600


# === Substrate/Cell/Decoder skeleton classes refuse instantiation ===


def test_substrate_skeleton_refuses_instantiation_at_L0() -> None:
    """Z7Mamba2V2Substrate raises NotImplementedError at L0 per CLAUDE.md."""
    config = Z7Mamba2V2Config()
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        Z7Mamba2V2Substrate(config)


def test_substrate_rejects_non_config_type() -> None:
    """Z7Mamba2V2Substrate rejects non-Config arg with TypeError before NotImplementedError."""
    with pytest.raises(TypeError, match="config must be Z7Mamba2V2Config"):
        Z7Mamba2V2Substrate(object())  # type: ignore[arg-type]


def test_temporal_decoder_skeleton_refuses_instantiation_at_L0() -> None:
    """Mamba2TemporalDecoder raises NotImplementedError at L0."""
    config = Z7Mamba2V2Config()
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        Mamba2TemporalDecoder(config)


def test_mamba2_v2_cell_skeleton_refuses_instantiation_at_L0() -> None:
    """Mamba2V2Cell raises NotImplementedError at L0."""
    config = Z7Mamba2V2Config()
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        Mamba2V2Cell(config)


# === Z7MCM3 archive grammar constants ===


def test_z7mcm3_magic_is_four_bytes() -> None:
    """Z7MCM3_MAGIC must be 4 bytes per Phase 3 §7.3 header layout."""
    assert Z7MCM3_MAGIC == b"Z7M3"
    assert len(Z7MCM3_MAGIC) == 4


def test_z7mcm3_version_is_3() -> None:
    """Z7MCM3_VERSION is 3 (sister to Z7MCM2=2)."""
    assert Z7MCM3_VERSION == 3


def test_z7mcm3_header_size_matches_format() -> None:
    """Z7MCM3_HEADER_SIZE matches struct.calcsize of Z7MCM3_HEADER_FMT."""
    assert Z7MCM3_HEADER_SIZE == struct.calcsize(Z7MCM3_HEADER_FMT)
    # Per Phase 3 §7.3: magic(4) + version(1) + num_pairs(2) + 6 * 1 byte fields = 13
    assert Z7MCM3_HEADER_SIZE == 13


def test_z7mcm3_section_roles_canonical_set() -> None:
    """Z7MCM3 section roles canonical per Phase 3 §7.3."""
    assert "meta_blob" in Z7MCM3_SECTION_ROLES
    assert "decoder_blob" in Z7MCM3_SECTION_ROLES
    assert "predictor_blob" in Z7MCM3_SECTION_ROLES
    assert "residuals_blob" in Z7MCM3_SECTION_ROLES
    assert "ego_motion_blob" in Z7MCM3_SECTION_ROLES


def test_a_log_init_scheme_byte_enum_covers_all_schemes() -> None:
    """A_LOG_INIT_SCHEME_BYTE_ENUM covers all enumerated schemes per CC-D + CC-J."""
    for scheme in A_LOG_INIT_SCHEMES:
        assert scheme in A_LOG_INIT_SCHEME_BYTE_ENUM
    # The byte enum is 1-byte per CC-J grammar layout
    for value in A_LOG_INIT_SCHEME_BYTE_ENUM.values():
        assert 0 <= value <= 255


# === Archive pack/parse/replay refuse to run at L0 ===


def test_pack_archive_refuses_at_L0() -> None:
    """pack_archive raises NotImplementedError at L0."""
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        pack_archive()


def test_parse_archive_refuses_at_L0() -> None:
    """parse_archive raises NotImplementedError at L0."""
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        parse_archive(b"")


def test_regenerate_a_log_validates_scheme_before_refusing() -> None:
    """regenerate_a_log validates scheme arg then raises NotImplementedError."""
    # Invalid scheme rejected first
    with pytest.raises(ValueError, match="a_log_init_scheme must be one of"):
        regenerate_a_log_from_init_scheme(
            d_inner=128, d_state=16, a_log_init_scheme="invalid"
        )
    # Valid scheme raises NotImplementedError
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        regenerate_a_log_from_init_scheme(
            d_inner=128, d_state=16, a_log_init_scheme="z_plus_1"
        )


def test_replay_latent_sequence_refuses_at_L0() -> None:
    """replay_latent_sequence raises NotImplementedError at L0."""
    with pytest.raises(TypeError, match="archive must be Z7MCM3Archive"):
        replay_latent_sequence(b"")  # type: ignore[arg-type]


# === Byte budget estimate is internally consistent ===


def test_byte_budget_estimate_self_consistent_per_cc_j_unwind() -> None:
    """Z7MCM3 byte budget vs Z7MCM2 baseline declares CC-J unwind savings."""
    budget = estimated_byte_budget()
    # CC-J unwind saves ~5 KB on predictor blob
    z7mcm3_predictor = budget["predictor_blob_z7mcm3_after_cc_j_unwind"]
    z7mcm2_baseline = budget["predictor_blob_z7mcm2_baseline_v1"]
    declared_savings = budget["predictor_blob_savings_vs_z7mcm2_per_cc_j"]
    assert z7mcm2_baseline - z7mcm3_predictor == declared_savings
    assert declared_savings >= 4 * 1024  # CC-J target: ≥ 4 KB savings


def test_byte_budget_total_matches_savings_claim() -> None:
    """Z7MCM3 total budget vs Z7MCM2 baseline matches CC-J savings."""
    budget = estimated_byte_budget()
    total_savings = (
        budget["z7mcm2_v1_baseline_total_compressed"]
        - budget["total_estimate_compressed"]
    )
    assert total_savings == budget["savings_per_cc_j_unwind"]


# === Inflate runtime refuses to run at L0 ===


def test_inflate_one_video_validates_args_before_refusing() -> None:
    """inflate_one_video validates args then raises NotImplementedError."""
    # Type validation first
    with pytest.raises(TypeError, match="archive_bytes must be bytes-like"):
        inflate_one_video("not_bytes", Path("/tmp/out.raw"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output_raw_path must be Path"):
        inflate_one_video(b"data", "not_path")  # type: ignore[arg-type]
    # Valid args raise NotImplementedError
    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD only"):
        inflate_one_video(b"data", Path("/tmp/out.raw"))


def test_main_cli_returns_2_on_missing_args() -> None:
    """main_cli returns 2 on missing CLI args per HNeRV parity L4 contract."""
    import sys as _sys

    saved_argv = _sys.argv
    try:
        _sys.argv = ["inflate.py"]
        assert main_cli() == 2
    finally:
        _sys.argv = saved_argv
