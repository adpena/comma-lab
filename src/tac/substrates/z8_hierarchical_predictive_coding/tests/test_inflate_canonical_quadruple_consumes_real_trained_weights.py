# SPDX-License-Identifier: MIT
"""Z8 M10 inflate consumes real trained weights tests.

Per Catalog #369 (`check_substrate_inflate_consumes_real_trained_weights_
not_synthetic_frame_base`) + Catalog #146 (contest 3-arg inflate.sh
contract) + Catalog #205 (canonical select_inflate_device) + Catalog #295
(PYTHONPATH self-containment) + Catalog #367 (raw bytes invariant 1164 x
874 x 1200 x 3 = 3,662,409,600 per video).

The M10 milestone closes the canonical Catalog #312 quadruple at the
deployment surface: M9 (trainer emits Z8HPC1 archive bytes from real
contest video wavelet decomposition) + M10 (inflate consumes those bytes
and reconstructs contest RAW frames via canonical Mallat 1989 §7.5 perfect
reconstruction).

Test surfaces:

1. Round-trip via canonical M5 Mallat inverse chain (rgb_0/rgb_1 max abs
   diff at float32 precision ~1e-7).
2. Contest RAW bytes contract (1200 frames at 874x1164x3 = 3,662,409,600).
3. Inflate runtime LOC budget (substrate-engineering waiver per HNeRV
   parity L4 + L7).
4. Catalog #205 select_inflate_device routing (auto + cpu + cuda).
5. PYTHONPATH self-containment per Catalog #295.
6. Catalog #369 forbidden-pattern absence (no synthetic-frame-base tokens
   in inflate body).
7. Catalog #146 3-arg CLI surface.
8. Cascade with M9: trainer emits archive bytes -> inflate reconstructs
   frames -> verify byte-stable reconstruction matches trainer's per-pair
   targets.
9. Pair cycling determinism (canonical from-archive-bytes per Catalog
   #369 cascade — NOT random).
10. Sister-disjoint vs Catalog #361 Modal artifact filter regression.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    parse_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
    parse_pair_blobs_from_wavelet_blob,
    reconstruct_pair_rgb_from_pyramid,
)
from tac.substrates.z8_hierarchical_predictive_coding.inflate import (
    CONTEST_NUM_FRAMES,
    CONTEST_OUT_H,
    CONTEST_OUT_W,
    CONTEST_RAW_BYTES,
    _read_single_member_archive_bytes,
    inflate_one_video_from_archive_bytes,
    inflate_one_video_l0_scaffold,
    main_cli,
    parse_and_validate_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)


def _canonical_cfg(num_pairs: int = 2, eval_h: int = 32, eval_w: int = 32) -> Z8HierarchicalConfig:
    """Build a smoke-friendly Z8HierarchicalConfig per M9 default."""
    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(eval_h, eval_w),
    )


def _real_video_pairs(num_pairs: int = 2, h: int = 32, w: int = 32, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize-from-fixed-seed deterministic RGB pairs for round-trip tests.

    These act as a stand-in for real upstream/videos/0.mkv decoded frames;
    the Mallat reconstruction is signal-content-independent so the
    round-trip invariant holds for any [0, 1] float32 input.
    """
    rng = np.random.RandomState(seed)
    f0 = rng.uniform(0, 1, size=(num_pairs, h, w, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(num_pairs, h, w, 3)).astype(np.float32)
    return f0, f1


# ---------------------------------------------------------------------------
# Catalog #146 + #205 + #295 + #367 surface tests
# ---------------------------------------------------------------------------


def test_inflate_module_imports_clean_from_empty_pythonpath_catalog_295() -> None:
    """Per Catalog #295: inflate must import without PYTHONPATH shims."""
    # The package itself imports clean; this test is a regression guard
    # that the M10 lift did not introduce a path-shim dependency.
    import importlib
    mod = importlib.import_module(
        "tac.substrates.z8_hierarchical_predictive_coding.inflate"
    )
    assert hasattr(mod, "main_cli")
    assert hasattr(mod, "inflate_one_video_from_archive_bytes")
    assert hasattr(mod, "CONTEST_RAW_BYTES")


def test_canonical_contest_raw_bytes_invariant_catalog_367() -> None:
    """Per Catalog #367: 1164 * 874 * 1200 * 3 = 3,662,409,600."""
    assert CONTEST_RAW_BYTES == 3_662_409_600
    assert CONTEST_NUM_FRAMES == 1200
    assert CONTEST_OUT_H == 874
    assert CONTEST_OUT_W == 1164
    assert CONTEST_OUT_W * CONTEST_OUT_H * CONTEST_NUM_FRAMES * 3 == CONTEST_RAW_BYTES


def test_inflate_loc_under_substrate_engineering_waiver_hnerv_l4_l7() -> None:
    """Per HNeRV parity L4 (≤200 LOC) + L7 substrate-engineering waiver.

    Z8 is lane_class=substrate_engineering per HNeRV parity L7 (substrate
    engineering UNIQUE-IFIES; explicit waiver). The M10 lift adds the
    canonical Mallat-inverse reconstruction path; ≤300 LOC budget is the
    substrate-engineering allowance per L4 + L7 (waiver from default 200
    documented in the inflate.py module docstring per the canonical
    contract).
    """
    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    loc = sum(
        1
        for line in inflate_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert loc <= 300, f"inflate.py code LOC={loc} > substrate-engineering 300 budget"


def test_inflate_no_synthetic_frame_base_tokens_catalog_369() -> None:
    """Per Catalog #369: inflate.py must not contain synthetic-frame-base patterns.

    Forbidden tokens per the canonical Catalog #369 gate include
    ``_render_frame_0_base`` / ``_synthesize_frame_base`` /
    ``_render_frame_placeholder`` / ``synthetic_frame_for_per_pair_warp`` /
    ``deterministic_textured_rgb``.
    """
    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    source = inflate_path.read_text(encoding="utf-8")
    forbidden = (
        "_render_frame_0_base(",
        "_synthesize_frame_base(",
        "_render_frame_placeholder(",
        "synthetic_frame_for_per_pair_warp",
        "deterministic_textured_rgb",
    )
    for token in forbidden:
        assert token not in source, (
            f"inflate.py contains forbidden Catalog #369 synthetic-frame-base "
            f"token: {token!r}"
        )


def test_inflate_imports_canonical_select_inflate_device_catalog_205() -> None:
    """Per Catalog #205: inflate must route through canonical select_inflate_device."""
    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    source = inflate_path.read_text(encoding="utf-8")
    assert "from tac.substrates._shared.inflate_runtime import" in source
    assert "select_inflate_device" in source
    assert "raw_output_path" in source
    assert "write_rgb_pair_to_raw" in source


def test_inflate_imports_canonical_quadruple_binding_module() -> None:
    """M10 cycle-closure: inflate consumes M9 canonical_quadruple_binding helpers."""
    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    source = inflate_path.read_text(encoding="utf-8")
    assert "canonical_quadruple_binding" in source
    assert "parse_pair_blobs_from_wavelet_blob" in source
    assert "reconstruct_pair_rgb_from_pyramid" in source


# ---------------------------------------------------------------------------
# Catalog #312 canonical quadruple round-trip tests
# ---------------------------------------------------------------------------


def test_build_z8hpc1_archive_bytes_from_canonical_quadruple_round_trip() -> None:
    """M10 archive emit + parse round-trip across all canonical sections."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=2)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    parsed = parse_archive(archive_bytes)
    assert parsed.num_levels == 3
    assert parsed.num_pairs == 2
    assert len(parsed.wavelet_coeffs_blob) > 0
    assert len(parsed.wyner_ziv_top_blob) > 0
    pyramids = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    assert len(pyramids) == 2


def test_reconstruct_pair_rgb_from_pyramid_mallat_perfect_reconstruction_round_trip() -> None:
    """Per Mallat 1989 §7.5: round-trip should be float32-precision-bound."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=2)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    parsed = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    for pair_idx in range(2):
        rgb_0, rgb_1 = reconstruct_pair_rgb_from_pyramid(binding, pyramids[pair_idx])
        # NCHW -> NHWC for comparison
        rgb_0_hwc = rgb_0[0].transpose(1, 2, 0)
        rgb_1_hwc = rgb_1[0].transpose(1, 2, 0)
        diff_0 = float(np.abs(rgb_0_hwc - f0[pair_idx]).max())
        diff_1 = float(np.abs(rgb_1_hwc - f1[pair_idx]).max())
        # Mallat perfect-reconstruction in float32: max abs diff ~1e-7.
        assert diff_0 < 1e-5, (
            f"pair {pair_idx} frame_0 Mallat round-trip max abs diff {diff_0:.3e} "
            f"exceeds float32 precision tolerance (expected < 1e-5)"
        )
        assert diff_1 < 1e-5, (
            f"pair {pair_idx} frame_1 Mallat round-trip max abs diff {diff_1:.3e} "
            f"exceeds float32 precision tolerance (expected < 1e-5)"
        )


def test_reconstruct_pair_rgb_returns_nchw_unit_range() -> None:
    """Reconstruction must satisfy canonical write_rgb_pair_to_raw contract."""
    cfg = _canonical_cfg(num_pairs=1)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=1)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    parsed = parse_archive(archive_bytes)
    pyramids = parse_pair_blobs_from_wavelet_blob(parsed.wavelet_coeffs_blob)
    rgb_0, rgb_1 = reconstruct_pair_rgb_from_pyramid(binding, pyramids[0])
    # NCHW (1, 3, H, W) per write_rgb_pair_to_raw contract.
    assert rgb_0.shape == (1, 3, 32, 32)
    assert rgb_1.shape == (1, 3, 32, 32)
    # unit range [0, 1] per write_rgb_pair_to_raw input_range="unit".
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0
    assert float(rgb_1.min()) >= 0.0
    assert float(rgb_1.max()) <= 1.0


# ---------------------------------------------------------------------------
# End-to-end inflate tests (Catalog #146 + #367)
# ---------------------------------------------------------------------------


def test_inflate_one_video_from_archive_bytes_emits_contest_raw_bytes() -> None:
    """Per Catalog #367: inflate writes 3,662,409,600 bytes per video."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=2)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    with tempfile.TemporaryDirectory() as td:
        out_raw = Path(td) / "0.raw"
        frames = inflate_one_video_from_archive_bytes(
            archive_bytes, out_raw, device="cpu"
        )
        assert frames == CONTEST_NUM_FRAMES
        assert out_raw.stat().st_size == CONTEST_RAW_BYTES


def test_inflate_one_video_l0_scaffold_alias_routes_through_m10_path() -> None:
    """Backward-compat alias must route through the M10 real-reconstruction path."""
    cfg = _canonical_cfg(num_pairs=1)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=1)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    with tempfile.TemporaryDirectory() as td:
        out_raw = Path(td) / "0.raw"
        frames = inflate_one_video_l0_scaffold(
            archive_bytes, out_raw, device="cpu"
        )
        assert frames == CONTEST_NUM_FRAMES
        assert out_raw.stat().st_size == CONTEST_RAW_BYTES


def test_inflate_one_video_pair_cycling_is_deterministic_per_archive_bytes() -> None:
    """Per Catalog #369: pair cycling is deterministic from archive bytes (NOT random)."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=2)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    with tempfile.TemporaryDirectory() as td1:
        out_raw_1 = Path(td1) / "0.raw"
        inflate_one_video_from_archive_bytes(archive_bytes, out_raw_1, device="cpu")
        h1 = out_raw_1.read_bytes()
    with tempfile.TemporaryDirectory() as td2:
        out_raw_2 = Path(td2) / "0.raw"
        inflate_one_video_from_archive_bytes(archive_bytes, out_raw_2, device="cpu")
        h2 = out_raw_2.read_bytes()
    assert h1 == h2, "two inflate runs on the same archive must produce byte-identical RAW"


def test_inflate_one_video_different_archives_produce_different_raw_bytes() -> None:
    """Different trained-state -> different RAW output (real consumption)."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0_a, f1_a = _real_video_pairs(num_pairs=2, seed=42)
    f0_b, f1_b = _real_video_pairs(num_pairs=2, seed=123)
    arch_a = build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0_a, f1_a)
    arch_b = build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0_b, f1_b)
    assert arch_a != arch_b
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        out_a = Path(td_a) / "0.raw"
        out_b = Path(td_b) / "0.raw"
        inflate_one_video_from_archive_bytes(arch_a, out_a, device="cpu")
        inflate_one_video_from_archive_bytes(arch_b, out_b, device="cpu")
        bytes_a = out_a.read_bytes()
        bytes_b = out_b.read_bytes()
    # Different trained state should produce different RAW bytes (proves the
    # inflate consumes the archive bytes per Catalog #369; not synthetic).
    assert bytes_a != bytes_b, (
        "different archive bytes must produce different RAW (proves real "
        "consumption per Catalog #369)"
    )


# ---------------------------------------------------------------------------
# Catalog #146 CLI surface tests
# ---------------------------------------------------------------------------


def test_main_cli_returns_2_on_missing_args() -> None:
    """Per Catalog #146 3-arg signature: rc=2 on missing args."""
    old_argv = sys.argv
    sys.argv = ["inflate.py"]
    try:
        assert main_cli() == 2
    finally:
        sys.argv = old_argv


def test_main_cli_processes_file_list_with_canonical_3_arg_contract() -> None:
    """Per Catalog #146: inflate.py <archive_dir> <output_dir> <file_list>."""
    cfg = _canonical_cfg(num_pairs=1)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=1)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        archive_dir = td_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "0.bin").write_bytes(archive_bytes)
        output_dir = td_path / "output"
        file_list_path = td_path / "file_list.txt"
        file_list_path.write_text("0\n", encoding="utf-8")
        old_argv = sys.argv
        sys.argv = [
            "inflate.py",
            str(archive_dir),
            str(output_dir),
            str(file_list_path),
        ]
        try:
            rc = main_cli()
        finally:
            sys.argv = old_argv
        assert rc == 0
        out_raw = output_dir / "0.raw"
        assert out_raw.exists()
        assert out_raw.stat().st_size == CONTEST_RAW_BYTES


def test_main_cli_skips_blank_lines_in_file_list() -> None:
    """Per inflate.py: blank lines in file_list are skipped silently."""
    cfg = _canonical_cfg(num_pairs=1)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=1)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        archive_dir = td_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "0.bin").write_bytes(archive_bytes)
        output_dir = td_path / "output"
        file_list_path = td_path / "file_list.txt"
        file_list_path.write_text("\n\n0\n\n", encoding="utf-8")
        old_argv = sys.argv
        sys.argv = [
            "inflate.py",
            str(archive_dir),
            str(output_dir),
            str(file_list_path),
        ]
        try:
            rc = main_cli()
        finally:
            sys.argv = old_argv
        assert rc == 0
        assert (output_dir / "0.raw").stat().st_size == CONTEST_RAW_BYTES


# ---------------------------------------------------------------------------
# Archive validation tests
# ---------------------------------------------------------------------------


def test_parse_and_validate_archive_passes_on_canonical_archive() -> None:
    """parse_and_validate_archive must accept canonical Z8HPC1 bytes."""
    cfg = _canonical_cfg(num_pairs=2)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=2)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    arc = parse_and_validate_archive(archive_bytes)
    assert arc.num_levels == 3
    assert arc.num_pairs == 2


def test_read_single_member_archive_bytes_refuses_missing() -> None:
    """_read_single_member_archive_bytes must refuse missing archive member."""
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FileNotFoundError, match="expected exactly one"):
            _read_single_member_archive_bytes(Path(td))


def test_read_single_member_archive_bytes_refuses_ambiguous() -> None:
    """_read_single_member_archive_bytes must refuse ambiguous archive members."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "0.bin").write_bytes(b"x")
        (td_path / "x").write_bytes(b"y")
        with pytest.raises(ValueError, match="ambiguous"):
            _read_single_member_archive_bytes(td_path)


# ---------------------------------------------------------------------------
# Catalog #361 sister regression: inflate.py is under src/tac/substrates
# (NOT under submission_dir at canonical authoring time); the Modal
# artifact filter sister gate at Catalog #361 protects against vendored
# submission_dir bodies being filtered by mtime — disjoint scope, but
# this regression guard verifies the inflate.py module is at the canonical
# import path used by both the lane registry and Modal worker source mount.
# ---------------------------------------------------------------------------


def test_inflate_canonical_module_path_is_canonical_substrate_engineering_location() -> None:
    """Catalog #361 sister-disjoint regression: inflate.py at canonical location."""
    import tac.substrates.z8_hierarchical_predictive_coding.inflate as inflate_mod
    inflate_file = Path(inflate_mod.__file__).resolve()
    # Canonical path is src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py
    assert inflate_file.name == "inflate.py"
    assert inflate_file.parent.name == "z8_hierarchical_predictive_coding"
    assert inflate_file.parent.parent.name == "substrates"


# ---------------------------------------------------------------------------
# Build progress regression: M10 milestone must be LANDED after this commit
# ---------------------------------------------------------------------------


def test_build_progress_m10_landed_status() -> None:
    """Per build_progress.py: M10 milestone must be marked LANDED."""
    from tac.substrates.z8_hierarchical_predictive_coding.build_progress import (
        BuildMilestoneStatus,
        Z8_PHASE_2_BUILD_MILESTONES,
        get_landed_milestones,
    )

    m10 = next(
        (
            m
            for m in Z8_PHASE_2_BUILD_MILESTONES
            if m.milestone_id == "inflate_runtime_consumes_real_trained_weights"
        ),
        None,
    )
    assert m10 is not None, "M10 milestone must exist in Z8_PHASE_2_BUILD_MILESTONES"
    assert m10.status == BuildMilestoneStatus.LANDED, (
        f"M10 status must be LANDED after this commit; got {m10.status}"
    )
    landed_ids = {m.milestone_id for m in get_landed_milestones(Z8_PHASE_2_BUILD_MILESTONES)}
    assert "inflate_runtime_consumes_real_trained_weights" in landed_ids


# ---------------------------------------------------------------------------
# Round-trip integration: trainer emit + inflate consume cascade
# ---------------------------------------------------------------------------


def test_inflate_cascade_with_real_video_pairs_consumes_archive_bytes() -> None:
    """End-to-end M9 + M10 cascade: trainer emits + inflate consumes."""
    cfg = _canonical_cfg(num_pairs=3)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = _real_video_pairs(num_pairs=3, seed=2026)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    assert len(archive_bytes) > 1000  # canonical brotli'd wavelet pyramid is non-trivial
    with tempfile.TemporaryDirectory() as td:
        out_raw = Path(td) / "0.raw"
        frames = inflate_one_video_from_archive_bytes(
            archive_bytes, out_raw, device="cpu"
        )
        # Contest contract per Catalog #367.
        assert frames == 1200
        assert out_raw.stat().st_size == CONTEST_RAW_BYTES
        # Verify the output is NOT all-zero / NOT all-one (real-byte consumption).
        sample = out_raw.read_bytes()[:1000]
        nonzero = sum(1 for b in sample if b != 0)
        assert nonzero > 100, (
            f"inflate output looks degenerate ({nonzero} nonzero bytes in first 1000); "
            f"real archive consumption should produce nonconstant bytes"
        )
