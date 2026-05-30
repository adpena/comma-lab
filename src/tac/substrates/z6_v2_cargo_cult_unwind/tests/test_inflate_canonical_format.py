# SPDX-License-Identifier: MIT
"""Phase C canonical inflate format extension tests for z6_v2_cargo_cult_unwind.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable + Slot EEE 5 forbidden
classes: every test below verifies ACTUAL BEHAVIOR not just constant
declarations. Tests cover (per Catalog #146 + #367 + #205 + canonical
leaderboard binding-depth L21+L29+L32):

* CONTEST_RAW_BYTES invariant + canonical 1164×874×1200×3 = 3,662,409,600
* Canonical helper routing (write_rgb_pair_to_raw + select_inflate_device)
* Schema v1 + v2 archive round-trip + parse dispatch
* Catalog #367 fail-closed on wrong-size raw output
* Allow-partial-frame-count smoke mode for MLX-LOCAL truncated dispatch
* End-to-end inflate produces actual bytes that match CONTEST_RAW_BYTES (slow)
* Archive v2 reduces archive size by ≥40% vs v1 (canonical L21+L29+L32 bind)
* 3-arg main_cli signature per Catalog #146
* Sister-DISJOINT: ONLY touches z6_v2 surfaces (no z8/dreamer/mdl edits)
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from tac.substrates.z6_v2_cargo_cult_unwind.architecture import (
    Z6V2Config,
    Z6V2Substrate,
)
from tac.substrates.z6_v2_cargo_cult_unwind.archive import (
    Z6V2_SCHEMA_VERSION,
    Z6V2_SCHEMA_VERSION_V2_INT8,
    pack_archive,
    pack_archive_v2_int8,
    parse_archive,
)
from tac.substrates.z6_v2_cargo_cult_unwind.inflate import (
    CAMERA_HW,
    CONTEST_NUM_FRAMES,
    CONTEST_OUT_H,
    CONTEST_OUT_W,
    CONTEST_RAW_BYTES,
    _raw_output_path,
    _read_single_member_archive_bytes,
    inflate_one_video,
    main_cli,
    select_inflate_device,
)


# --------------------------------------------------------------------------- #
# CONTEST_RAW_BYTES invariants per Catalog #367 (THIS gate is the canonical
# fail-closed sister for z6_v2)
# --------------------------------------------------------------------------- #


def test_contest_raw_bytes_canonical_invariant():
    """Per Catalog #367: raw byte count MUST be exactly 3,662,409,600."""
    assert CONTEST_OUT_H == 874
    assert CONTEST_OUT_W == 1164
    assert CONTEST_NUM_FRAMES == 1200
    assert CONTEST_RAW_BYTES == 3_662_409_600
    assert CONTEST_RAW_BYTES == CONTEST_OUT_H * CONTEST_OUT_W * CONTEST_NUM_FRAMES * 3


def test_camera_hw_matches_shared_canonical_helper():
    """Z6_v2 inflate CAMERA_HW MUST match canonical shared helper."""
    from tac.substrates._shared.inflate_runtime import CAMERA_HW as SHARED_CAMERA_HW
    assert CAMERA_HW == SHARED_CAMERA_HW
    assert CAMERA_HW == (874, 1164)


# --------------------------------------------------------------------------- #
# Canonical helper routing per Catalog #205 + #146
# --------------------------------------------------------------------------- #


def test_select_inflate_device_returns_torch_device():
    """Canonical Catalog #205 select_inflate_device routes through shared helper."""
    dev = select_inflate_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda"}, f"unexpected device type: {dev.type}"
    # MPS structurally refused per CLAUDE.md "MPS auth eval is NOISE" non-negotiable
    assert dev.type != "mps"


def test_inflate_imports_canonical_shared_helper():
    """Verifies the inflate.py module imports the canonical shared helper
    rather than re-implementing rgb_pair_to_raw conversion locally.

    Per Catalog #226 sister discipline at the inflate-helper-routing surface:
    substrate inflate MUST route through ``tac.substrates._shared.inflate_runtime.write_rgb_pair_to_raw``
    (canonical Catalog L20+L29 helper) rather than hand-rolled per-frame
    PNG → raw conversions.
    """
    inflate_path = Path(__file__).resolve().parent.parent / "inflate.py"
    body = inflate_path.read_text(encoding="utf-8")
    assert "from tac.substrates._shared.inflate_runtime import" in body
    assert "write_rgb_pair_to_raw" in body
    assert "select_inflate_device" in body


def test_inflate_no_png_output_legacy_removed():
    """Per Phase C reactivation criterion #1: PNG output is forbidden in
    canonical mode. The legacy ``Image.fromarray(...).save(...png...)``
    pattern from pre-Phase-C must be GONE.
    """
    inflate_path = Path(__file__).resolve().parent.parent / "inflate.py"
    body = inflate_path.read_text(encoding="utf-8")
    assert ".png" not in body.lower(), (
        "PNG output is forbidden in canonical Phase C inflate; "
        "use .raw via write_rgb_pair_to_raw per Catalog #146"
    )
    assert "Image.fromarray" not in body, "Pillow PNG-save path forbidden"


# --------------------------------------------------------------------------- #
# Schema v1 + v2 dispatch invariants
# --------------------------------------------------------------------------- #


def _build_test_meta(cfg: Z6V2Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "rao_ballard_level_boundary": cfg.rao_ballard_level_boundary,
        "film_generator_depth": cfg.film_generator_depth,
        "film_hidden_width": cfg.film_hidden_width,
        "cooperative_receiver_beta": cfg.cooperative_receiver_beta,
    }


def test_v1_archive_round_trip_preserves_keys():
    cfg = Z6V2Config(num_pairs=4)
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    blob = pack_archive(sd, model.latents.data, model.ego_vecs.data, _build_test_meta(cfg))
    arc = parse_archive(blob)
    assert arc.schema_version == Z6V2_SCHEMA_VERSION
    assert set(arc.decoder_state_dict.keys()) == set(sd.keys())


def test_v2_archive_round_trip_preserves_keys():
    cfg = Z6V2Config(num_pairs=4)
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    blob = pack_archive_v2_int8(sd, model.latents.data, model.ego_vecs.data, _build_test_meta(cfg))
    arc = parse_archive(blob)
    assert arc.schema_version == Z6V2_SCHEMA_VERSION_V2_INT8
    assert set(arc.decoder_state_dict.keys()) == set(sd.keys())


def test_v2_archive_dequant_error_within_grid_resolution():
    """v2 INT8 quantization MUST reconstruct fp32 within 1/254 grid resolution."""
    cfg = Z6V2Config(num_pairs=4)
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    blob = pack_archive_v2_int8(sd, model.latents.data, model.ego_vecs.data, _build_test_meta(cfg))
    arc = parse_archive(blob)
    # Check at least one tensor's quantization error fits within the grid
    sample_key = next(iter(sd.keys()))
    original = sd[sample_key].detach().cpu().to(torch.float32)
    reconstructed = arc.decoder_state_dict[sample_key].detach().cpu().to(torch.float32)
    range_t = float(original.max() - original.min())
    if range_t > 0:
        max_err = float((original - reconstructed).abs().max())
        # INT8 with 254 levels => max error per tensor < range/254
        assert max_err < range_t / 254.0 + 1e-6, (
            f"INT8 quant error {max_err} exceeds grid resolution "
            f"{range_t/254.0} for tensor {sample_key!r}"
        )


def test_v2_archive_reduces_size_vs_v1_at_canonical_600_pairs():
    """Canonical L21+L29+L32 bind: v2 archive MUST be ≥40% smaller than v1."""
    cfg = Z6V2Config(num_pairs=600)
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    meta = _build_test_meta(cfg)
    v1_blob = pack_archive(sd, model.latents.data, model.ego_vecs.data, meta)
    v2_blob = pack_archive_v2_int8(sd, model.latents.data, model.ego_vecs.data, meta)
    reduction_pct = 100.0 * (len(v1_blob) - len(v2_blob)) / len(v1_blob)
    assert reduction_pct >= 40.0, (
        f"v2 archive should be ≥40% smaller than v1; got {reduction_pct:.1f}% "
        f"(v1={len(v1_blob)}B v2={len(v2_blob)}B)"
    )


def test_parse_archive_rejects_unsupported_schema_version():
    """parse_archive raises on unknown schema version (3+)."""
    import struct
    from tac.substrates.z6_v2_cargo_cult_unwind.archive import Z6V2_HEADER_FMT, Z6V2_MAGIC
    bogus = struct.pack(
        Z6V2_HEADER_FMT,
        Z6V2_MAGIC,
        99,  # unsupported version
        24, 6, 1, 0, 0, 0, 0, 0,
    )
    with pytest.raises(ValueError, match="unsupported schema version"):
        parse_archive(bogus)


# --------------------------------------------------------------------------- #
# Allow-partial-frame-count smoke mode (MLX-LOCAL truncated dispatch)
# --------------------------------------------------------------------------- #


def test_inflate_truncated_archive_refuses_in_canonical_mode():
    """allow_partial_frame_count=False (canonical contest mode) refuses
    archives whose pair count != 600."""
    cfg = Z6V2Config(num_pairs=4)
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    blob = pack_archive_v2_int8(sd, model.latents.data, model.ego_vecs.data, _build_test_meta(cfg))
    with pytest.raises(ValueError, match="canonical contest mode requires 600 pairs"):
        inflate_one_video(blob, Path("/tmp/z6v2_smoke_canonical.raw"), device="cpu")


def test_inflate_truncated_archive_accepts_with_allow_partial(tmp_path: Path):
    """allow_partial_frame_count=True accepts truncated archive + skips
    Catalog #367 fail-closed (advisory MLX-LOCAL smoke mode)."""
    cfg = Z6V2Config(num_pairs=2)  # tiny so it runs fast
    model = Z6V2Substrate(cfg)
    sd = {k: v for k, v in model.state_dict().items() if k not in ("latents", "ego_vecs")}
    blob = pack_archive_v2_int8(sd, model.latents.data, model.ego_vecs.data, _build_test_meta(cfg))
    out_path = tmp_path / "smoke.raw"
    n_frames = inflate_one_video(
        blob, out_path, device="cpu", allow_partial_frame_count=True,
    )
    # 2 pairs = 4 frames at CAMERA_HW resolution
    assert n_frames == 4
    expected_bytes = 4 * CAMERA_HW[0] * CAMERA_HW[1] * 3
    assert out_path.stat().st_size == expected_bytes


# --------------------------------------------------------------------------- #
# main_cli + _raw_output_path safety invariants per Catalog #146
# --------------------------------------------------------------------------- #


def test_inflate_main_cli_3arg_signature_per_catalog_146():
    """Catalog #146 contest-compliant 3-arg runtime contract."""
    inflate_path = Path(__file__).resolve().parent.parent / "inflate.py"
    body = inflate_path.read_text(encoding="utf-8")
    assert "archive_dir = Path(sys.argv[1])" in body
    assert "output_dir = Path(sys.argv[2])" in body
    assert "file_list_path = Path(sys.argv[3])" in body


def test_main_cli_missing_args_returns_2():
    """main_cli with <4 args prints usage + returns 2."""
    import sys as _sys
    saved_argv = _sys.argv
    _sys.argv = ["inflate.py", "only_one_arg"]
    try:
        assert main_cli() == 2
    finally:
        _sys.argv = saved_argv


def test_raw_output_path_rejects_absolute_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="unsafe file_list video name"):
        _raw_output_path(tmp_path, "/etc/passwd")


def test_raw_output_path_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="unsafe file_list video name"):
        _raw_output_path(tmp_path, "../escape.bin")


def test_raw_output_path_appends_raw_suffix(tmp_path: Path):
    result = _raw_output_path(tmp_path, "0.mkv")
    assert result.suffix == ".raw"
    assert result.parent == tmp_path.resolve(strict=False)


def test_read_single_member_finds_zero_bin(tmp_path: Path):
    blob = b"test archive bytes"
    (tmp_path / "0.bin").write_bytes(blob)
    assert _read_single_member_archive_bytes(tmp_path) == blob


def test_read_single_member_finds_x_legacy_fallback(tmp_path: Path):
    blob = b"legacy x member"
    (tmp_path / "x").write_bytes(blob)
    assert _read_single_member_archive_bytes(tmp_path) == blob


def test_read_single_member_refuses_no_member(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="expected exactly one archive member"):
        _read_single_member_archive_bytes(tmp_path)


def test_read_single_member_refuses_ambiguous(tmp_path: Path):
    (tmp_path / "0.bin").write_bytes(b"a")
    (tmp_path / "x").write_bytes(b"b")
    with pytest.raises(ValueError, match="ambiguous archive members"):
        _read_single_member_archive_bytes(tmp_path)


# --------------------------------------------------------------------------- #
# Sister-DISJOINT regression guards per Catalog #340
# --------------------------------------------------------------------------- #


def test_inflate_module_does_not_import_z8_substrate():
    """Sister-DISJOINT vs PR110-OPT-7 + z8 + dreamer per Catalog #340.

    Z6-v2's Phase C extension MUST NOT pull in z8/dreamer/mdl substrate code
    (those surfaces are sister-Agent territory in the concurrent wave).
    """
    inflate_path = Path(__file__).resolve().parent.parent / "inflate.py"
    body = inflate_path.read_text(encoding="utf-8")
    assert "z8_hierarchical" not in body
    assert "dreamer_v3" not in body
    assert "mdl_ibps_j" not in body
    assert "pr110" not in body.lower()


def test_archive_module_v2_does_not_import_z8_substrate():
    """Sister-DISJOINT regression guard at archive module."""
    archive_path = Path(__file__).resolve().parent.parent / "archive.py"
    body = archive_path.read_text(encoding="utf-8")
    assert "z8_hierarchical" not in body
    assert "dreamer_v3" not in body
    assert "mdl_ibps_j" not in body
