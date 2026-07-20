"""Focused EKPR1 codec and LVLS1 receiver/oracle integration tests.

These fixtures are tiny local build checks, not score evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.boundary_math.localized_basis_frames import literal_basis_program_config
from tac.codec.levelset_palette_residual import (
    EKPR1_APPLICATION,
    LevelsetPaletteResidualError,
    apply_palette_residual,
    cap_palette_residual,
    decode_palette_residual,
    encode_palette_residual,
    validate_palette_residual_binding,
)

_REPO = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO / "tools" / "levelset_byte_close_and_eval.py"
_N_PAIRS = 2
_N_CLASSES = 5
_RH, _RW = 6, 8


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("levelset_byte_close_ekpr1", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _residuals(n_pairs: int = _N_PAIRS) -> np.ndarray:
    values = np.arange(n_pairs * _N_CLASSES * 3, dtype=np.int16).reshape(n_pairs, _N_CLASSES, 3)
    return (values - 20).astype(np.int8)


def _manifest_entry(n_pairs: int = _N_PAIRS) -> dict[str, object]:
    return {
        "codec": "EKPR1",
        "version": 1,
        "shape": [n_pairs, _N_CLASSES, 3],
        "dtype": "int8",
        "application": EKPR1_APPLICATION,
    }


def test_ekpr1_roundtrip_and_cap_are_canonical() -> None:
    source = _residuals(4)
    raw = encode_palette_residual(source)
    parsed = decode_palette_residual(raw, expected_n_pairs=4, expected_n_classes=_N_CLASSES)
    np.testing.assert_array_equal(parsed.residuals, source)
    capped = cap_palette_residual(raw, 2)
    expected = encode_palette_residual(source[:2])
    assert capped == expected
    np.testing.assert_array_equal(decode_palette_residual(capped).residuals, source[:2])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: b.__setitem__(0, ord("X")), "magic"),
        (lambda b: b.__setitem__(6, 2), "version"),
        (lambda b: b.__setitem__(13, 4), "channels"),
        (lambda b: b.__setitem__(14, 2), "dtype"),
    ],
)
def test_ekpr1_rejects_header_mutations(mutate, match: str) -> None:
    raw = bytearray(encode_palette_residual(_residuals()))
    mutate(raw)
    with pytest.raises(LevelsetPaletteResidualError, match=match):
        decode_palette_residual(raw)


def test_ekpr1_rejects_truncation_bad_length_and_trailing() -> None:
    raw = encode_palette_residual(_residuals())
    with pytest.raises(LevelsetPaletteResidualError, match="truncated"):
        decode_palette_residual(raw[:-1])
    with pytest.raises(LevelsetPaletteResidualError, match="trailing"):
        decode_palette_residual(raw + b"x")
    bad_length = bytearray(raw)
    struct.pack_into("<I", bad_length, 15, len(raw))
    with pytest.raises(LevelsetPaletteResidualError, match="length field"):
        decode_palette_residual(bad_length)


def test_manifest_section_bijection_and_shape_are_strict() -> None:
    raw = encode_palette_residual(_residuals())
    with pytest.raises(LevelsetPaletteResidualError, match="without palette_residual manifest"):
        validate_palette_residual_binding(None, raw, expected_n_pairs=_N_PAIRS, expected_n_classes=_N_CLASSES)
    with pytest.raises(LevelsetPaletteResidualError, match="without EKPR1 section"):
        validate_palette_residual_binding(
            _manifest_entry(), None, expected_n_pairs=_N_PAIRS, expected_n_classes=_N_CLASSES
        )
    extra = {**_manifest_entry(), "ignored": True}
    with pytest.raises(LevelsetPaletteResidualError, match="keys mismatch"):
        validate_palette_residual_binding(extra, raw, expected_n_pairs=_N_PAIRS, expected_n_classes=_N_CLASSES)
    with pytest.raises(LevelsetPaletteResidualError, match="n_pairs"):
        validate_palette_residual_binding(_manifest_entry(), raw, expected_n_pairs=600, expected_n_classes=_N_CLASSES)


def test_n24_section_cannot_attach_to_n600_lvls1(tool: ModuleType) -> None:
    raw = encode_palette_residual(_residuals(24))
    manifest = {
        "n_pairs": 600,
        "n_classes": _N_CLASSES,
        "palette_residual": _manifest_entry(24),
    }
    blob = tool._io_pack(
        json.dumps(manifest, separators=(",", ":")).encode(),
        b"base",
        b"code",
        None,
        None,
        None,
        None,
        raw,
    )
    with pytest.raises(LevelsetPaletteResidualError, match="n_pairs"):
        tool._read_blob_bytes_full(blob)


def test_lvls1_manifest_section_bijection_fails_closed(tool: ModuleType) -> None:
    raw = encode_palette_residual(_residuals())
    base_manifest = {"n_pairs": _N_PAIRS, "n_classes": _N_CLASSES}
    unbound = tool._io_pack(
        json.dumps(base_manifest, separators=(",", ":")).encode(),
        b"base",
        b"code",
        None,
        None,
        None,
        None,
        raw,
    )
    with pytest.raises(ValueError, match="unconsumed trailing"):
        tool._read_blob_bytes_full(unbound)
    declared = {**base_manifest, "palette_residual": _manifest_entry()}
    missing = tool._io_pack(json.dumps(declared, separators=(",", ":")).encode(), b"base", b"code", None)
    with pytest.raises(ValueError, match="without EKPR1 section"):
        tool._read_blob_bytes_full(missing)


def test_apply_uses_phi_argmax_clips_and_mutation_is_non_inert() -> None:
    rgb = np.asarray([[250.0, 1.0, 120.0], [20.0, 30.0, 40.0]], dtype=np.float32)
    phi = np.asarray([[0.0, 4.0, 1.0], [5.0, 0.0, 1.0]], dtype=np.float32)
    residual = np.zeros((1, 3, 3), dtype=np.int8)
    residual[0, 1] = [10, -5, 2]
    residual[0, 0] = [-3, 4, 100]
    got = apply_palette_residual(rgb, phi, residual, pair_index=0)
    np.testing.assert_array_equal(got, np.asarray([[255.0, 0.0, 122.0], [17.0, 34.0, 140.0]], dtype=np.float32))
    mutated = residual.copy()
    mutated[0, 1, 2] += 1
    changed = apply_palette_residual(rgb, phi, mutated, pair_index=0)
    assert not np.array_equal(got, changed), "EKPR1 payload mutation was receiver-inert"


def test_shipped_inline_apply_matches_numpy_helper(tool: ModuleType) -> None:
    namespace: dict[str, object] = {"__name__": "_ekpr_inline_test"}
    source = tool._inflate_source_for_manifest({"basis_family": "legacy_fourier_ab_control", "palette_residual": {}})
    exec(compile(source, "<inflate.py>", "exec"), namespace)
    rgb = np.asarray([[100.25, 120.5, 140.75], [200.0, 10.0, 40.0]], dtype=np.float64)
    phi = np.asarray([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]], dtype=np.float64)
    residual = np.zeros((1, 3, 3), dtype=np.int8)
    residual[0, 1] = [5, -6, 7]
    residual[0, 2] = [-8, 9, -10]
    expected = apply_palette_residual(rgb, phi, residual, pair_index=0)
    actual = namespace["_ekpr_apply"](rgb, phi, residual, 0)
    np.testing.assert_array_equal(actual, expected)


def _tiny_literal_checkpoint(tmp_path: Path) -> Path:
    rng = np.random.default_rng(19)
    program = literal_basis_program_config()

    def weight(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    hidden = 4
    n_hidden = 1
    mod_dim = 2
    payload = {
        "code": weight(2 * _N_PAIRS, mod_dim),
        "in_proj.weight": weight(hidden, 80),
        "in_proj.bias": weight(hidden),
        "film.weight": weight(n_hidden * 2 * hidden, mod_dim),
        "film.bias": weight(n_hidden * 2 * hidden),
        "hidden.0.weight": weight(hidden, hidden),
        "hidden.0.bias": weight(hidden),
        "out_sdf.weight": weight(_N_CLASSES, hidden),
        "out_sdf.bias": weight(_N_CLASSES),
        "out_tex.weight": weight(3, hidden),
        "out_tex.bias": weight(3),
        "palette": weight(_N_CLASSES, 3),
        "__cfg_basis": np.asarray("literal_polar_curvelet"),
        "__cfg_basis_program_json": np.asarray(json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))),
        "__cfg_basis_program_sha256": np.asarray(program.canonical_sha256()),
        "__cfg_basis_taper_folded": np.asarray(0),
        "__cfg_hidden_dim": np.asarray(hidden),
        "__cfg_n_hidden": np.asarray(n_hidden),
        "__cfg_activation": np.asarray("hosc"),
        "__render_hw": np.asarray([_RH, _RW]),
        "__bank_n_scales": np.asarray(4),
    }
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    np.savez(ckpt / "levelset_witness_ema.npz", **payload)
    return ckpt


def test_optional_section_absent_blob_matches_pre_ekpr_golden_and_receiver_oracle_is_bit_exact(
    tool: ModuleType, tmp_path: Path
) -> None:
    ckpt = _tiny_literal_checkpoint(tmp_path)
    params, cfg = tool._load_levelset_ckpt(ckpt, "levelset_witness_ema.npz")
    cfg["basis_program_deploy"] = cfg["basis_program"]
    so = tool.detect_self_orient(cfg, {})
    legacy, _ = tool.build_levelset_blob(params, cfg, so, None)
    explicit_absent, _ = tool.build_levelset_blob(params, cfg, so, None, palette_residual_bytes=None)
    assert legacy == explicit_absent
    # Golden from 8eee0e3bec, before EKPR landed. The absent *LVLS1 payload*
    # remains byte-identical; the shipped receiver source intentionally grows
    # to parse EKPR and is not claimed to be source-byte-identical.
    assert hashlib.sha256(legacy).hexdigest() == "7e3d094fdbb319c7d21bc91b2cfa9977a05b2253ec9a8881f9ca7230fb9fce7d"

    residual = np.full((_N_PAIRS, _N_CLASSES, 3), 64, dtype=np.int8)
    section = encode_palette_residual(residual)
    active, breakdown = tool.build_levelset_blob(params, cfg, so, None, palette_residual_bytes=section)
    assert breakdown["palette_residual_counted_bytes"] == len(section)
    manifest, base_b, code_b, _pose, _lane, _pcar, _chart, palette_b = tool._read_blob_bytes_full(active)
    parsed = validate_palette_residual_binding(
        manifest["palette_residual"],
        palette_b,
        expected_n_pairs=_N_PAIRS,
        expected_n_classes=_N_CLASSES,
    )
    assert parsed is not None
    params_d = tool._decode_base_params(manifest, base_b)
    code_d = tool._decode_code(manifest, code_b)
    frames_off, _ = tool.numpy_oracle_reference_frames(params_d, code_d, manifest, _N_PAIRS)
    frames_on, _ = tool.numpy_oracle_reference_frames(
        params_d,
        code_d,
        manifest,
        _N_PAIRS,
        palette_residual=parsed.residuals,
    )
    for pair_index in range(_N_PAIRS):
        np.testing.assert_array_equal(frames_off[2 * pair_index], frames_on[2 * pair_index])
        assert not np.array_equal(frames_off[2 * pair_index + 1], frames_on[2 * pair_index + 1]), (
            "EKPR1 did not mutate frame1 before R"
        )

    packet_dir = tmp_path / "packet"
    tool.assemble_packet(active, packet_dir)
    receipt = tool.bit_exact_roundtrip_gate(packet_dir, active, gate_pairs=1, strict=True)
    assert receipt["bit_exact"] is True
