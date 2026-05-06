from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/profile_pr82_henosis_frontier_contract.py"
PR82_INTAKE = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex"
PR82_ARCHIVE = PR82_INTAKE / "archive.zip"
PR82_REPLAY_INFLATE = PR82_INTAKE / "replay_submission/inflate.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("profile_pr82_henosis_frontier_contract", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_single_x_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_synthetic_v5_micro_bundle_is_planning_only(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    script = _load_script()

    encoded = {
        "mask": brotli.compress(b"\x12\x00" + b"mask" * 400),
        "model": brotli.compress(b"QH0" + bytes(3200)),
        "pose": brotli.compress(b"P1D1" + bytes([2, 0, 3, 0, 1, 2, 0]) + b"abcde"),
        "post": brotli.compress(bytes(2400)),
        "shift": brotli.compress(b"SD4" + bytes(600)),
        "frac": brotli.compress(b"FV1" + (0).to_bytes(2, "little")),
        "frac2": brotli.compress(b"FH2" + bytes([4]) * 600),
        "frac3": brotli.compress(b"FD3" + bytes(600)),
        "bias": brotli.compress(b"BD1" + bytes(600)),
        "region": brotli.compress(b"RH1" + bytes(600)),
        "randmulti": brotli.compress(b"\x00"),
    }
    replay = tmp_path / "inflate.py"
    replay.write_text(
        "\n".join(
            [
                "def load_compact_archive_bundle():",
                f"    l_bias = {len(encoded['bias'])}",
                f"    l_region = {len(encoded['region'])}",
                "def main():",
                "    specs_n = [(1, 1, 1, 1)]",
            ]
        )
    )
    header_names = ["mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3"]
    payload = b"".join(len(encoded[name]).to_bytes(3, "little") for name in header_names)
    payload += b"".join(encoded[name] for name in header_names)
    payload += encoded["bias"] + encoded["region"] + encoded["randmulti"]
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    _write_single_x_zip(archive, payload)

    profile = script.build_profile(archive=archive, replay_inflate=replay, output_json=output)

    assert profile["schema"] == "pr82_henosis_frontier_static_profile_v1"
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["dispatch_performed"] is False
    assert profile["compact_bundle"]["format"] == "public_pr82_henosis_compact_v5_micro_header"
    assert profile["compact_bundle"]["header_bytes"] == 24
    assert profile["anatomy"]["pose"]["format"] == "P1D1_delta_varint"
    assert profile["anatomy"]["postprocess"]["stage_count"] == 4
    assert profile["anatomy"]["randmulti"]["valid"] is True
    assert output.exists()


@pytest.mark.skipif(not PR82_ARCHIVE.exists(), reason="PR82 intake archive missing")
def test_actual_pr82_intake_profile_pins_static_contract(tmp_path: Path) -> None:
    script = _load_script()
    output = tmp_path / "profile.json"

    profile = script.build_profile(
        archive=PR82_ARCHIVE,
        replay_inflate=PR82_REPLAY_INFLATE,
        output_json=output,
    )

    assert profile["zip_container"]["archive_bytes"] == 296789
    assert profile["zip_container"]["archive_sha256"] == (
        "a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4"
    )
    assert profile["zip_container"]["members"][0]["filename"] == "x"
    assert profile["zip_container"]["members"][0]["file_size"] == 296689
    assert profile["compact_bundle"]["header_lengths_u24"] == {
        "mask": 219472,
        "model": 57074,
        "pose": 1487,
        "post": 1400,
        "shift": 226,
        "frac": 106,
        "frac2": 149,
        "frac3": 154,
    }
    assert profile["compact_bundle"]["fixed_tail_lengths_from_replay"] == {
        "bias": 223,
        "region": 273,
    }
    assert profile["anatomy"]["model_qh0"]["format"] == "QH0"
    assert profile["anatomy"]["model_qh0"]["parse_valid"] is True
    assert profile["anatomy"]["model_qh0"]["qconv_quantization_counts"]["fp4_hilosplit"] == 40
    assert profile["anatomy"]["pose"]["dimension_count"] == 2
    assert profile["anatomy"]["postprocess"]["valid"] is True
    assert profile["anatomy"]["randmulti"]["group_count"] == 72
    assert profile["anatomy"]["randmulti"]["valid"] is True
    assert profile["static_rate_break_even_vs_reference"]["score_claim"] is False
    assert profile["static_rate_break_even_vs_reference"]["static_archive_delta_bytes_vs_reference"] == 19468
    assert json.loads(output.read_text())["score_claim"] is False


def test_argparse_exposes_no_reference_and_output() -> None:
    script = _load_script()

    args = script.build_arg_parser().parse_args(
        [
            "--archive",
            "archive.zip",
            "--replay-inflate",
            "inflate.py",
            "--output-json",
            "profile.json",
            "--no-reference",
        ]
    )

    assert str(args.archive) == "archive.zip"
    assert str(args.replay_inflate) == "inflate.py"
    assert str(args.output_json) == "profile.json"
    assert args.no_reference is True
