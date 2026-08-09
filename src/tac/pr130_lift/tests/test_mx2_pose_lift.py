from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tac.pr130_lift.pose import mlx_pose_carrier, repack_race
from tac.pr130_lift.pose.source_loader import load_lifted_module


def test_vendor_manifest_records_pr130_pose_sources() -> None:
    manifest_path = Path("src/tac/pr130_lift/pose/vendor_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_repo_head"] == "e34f31bc4969042c0051ac81aa3c56884419a231"
    assert manifest["lifted_at_head"] == "2f94596bb0136d342254022a5c9584756eae0468"
    assert manifest["source_sha256_scope"].startswith("exact intake path bytes")
    assert manifest["vendored_body_authentication"].endswith(
        "test_fx2_lift_custody.py"
    )
    paths = {entry["path"]: entry for entry in manifest["files"]}
    assert paths["code/carrier_codec.py"]["sha256"] == (
        "d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed"
    )
    assert paths["code/repack_carrier.py"]["sha256"] == (
        "df6bdaa23d0bd1f717af588931ebdb5ed3777517af8b046e287835aa16a7a72e"
    )
    assert len(paths) == 8
    assert paths["code/train_pose_carrier_full.py"]["adaptation"] == (
        "governed_admission_guard_after_argparse"
    )
    assert sum(entry["adaptation"] == "none" for entry in paths.values()) == 7


def test_cpr1_codec_roundtrips_legacy_shape_symbols() -> None:
    codec = load_lifted_module("carrier_codec")
    rng = np.random.default_rng(130)
    basis_scales = np.linspace(0.25, 1.5, repack_race.CARRIER_DIM, dtype=np.float32)
    basis_codes = rng.integers(
        -8,
        8,
        size=repack_race.BASIS_COUNT,
        dtype=np.int32,
    )
    coefficient_scales = np.linspace(
        0.1,
        0.9,
        repack_race.CARRIER_DIM,
        dtype=np.float32,
    )
    encoded_coefficients = rng.integers(
        0,
        4096,
        size=(repack_race.N, repack_race.CARRIER_DIM),
        dtype=np.int32,
    )
    compact = codec.encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
    decoded = codec.decode_compact_carrier(
        compact,
        repack_race.BASIS_COUNT,
        repack_race.N,
        repack_race.CARRIER_DIM,
    )
    for actual, expected in zip(
        decoded,
        (basis_scales, basis_codes, coefficient_scales, encoded_coefficients),
        strict=True,
    ):
        assert np.array_equal(actual, expected)


def test_cpr1_rejects_non_legacy_carrier_bytes() -> None:
    result = repack_race.try_cpr1_legacy_carrier(b"PFS1WPD1" + bytes(17))
    assert result["status"] == "NOT_COMPATIBLE"
    assert "legacy carrier length mismatch" in result["reason"]
    assert result["roundtrip"] is False


def test_generic_coder_roundtrips_payload() -> None:
    payload = (b"pose-warp-section" * 64) + bytes(range(16))
    result = repack_race.best_generic_code(payload)
    assert result["roundtrip"] is True
    assert result["bytes"] <= len(payload)
    assert result["coder"] in {"stored", "deflate", "brotli", "lzma"}


def test_mlx_probe_handles_missing_runtime_without_importing_mlx() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        probe = mlx_pose_carrier.mlx_device_probe()
    assert probe == {
        "available": False,
        "status": "BLOCKED",
        "reason": "ModuleNotFoundError: No module named 'mlx'",
    }
