from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import FIXED_V5_LENGTHS, SEGMENT_ORDER, pack_pr85_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_model_payload_self_compression.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_model_payload_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_archive(path: Path, *, model: bytes, member_name: str = "x") -> dict[str, bytes]:
    segments = {
        "mask": b"QMA9" + b"m" * 600,
        "model": model,
        "pose": b"P1D1" + b"p" * 64,
        "post": b"post" * 25,
        "shift": b"shift" * 5,
        "frac": b"frac" * 3,
        "frac2": b"frac2" * 3,
        "frac3": b"frac3" * 3,
        "bias": b"B" * FIXED_V5_LENGTHS["bias"],
        "region": b"R" * FIXED_V5_LENGTHS["region"],
        "randmulti": b"randmulti" * 8,
    }
    raw = pack_pr85_bundle(segments, header_mode="v5")
    info = zipfile.ZipInfo(member_name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)
    return segments


def test_profiles_brotli_qh0_model_segment_and_routes(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    decoded_model = b"QH0" + b"\x00" * 4096 + bytes((idx % 17 for idx in range(4096)))
    model = brotli.compress(decoded_model, quality=1)
    archive = tmp_path / "archive.zip"
    segments = _write_archive(archive, model=model)

    profile = module.build_profile(
        archive,
        bit_budget_json=None,
        bundle_profile_json=None,
        window_bytes=512,
    )

    assert profile["schema"] == module.SCHEMA
    assert profile["planning_only"] is True
    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["model_segment"]["offset_in_x_member"] == 24 + len(segments["mask"])
    assert profile["model_segment"]["encoded"]["sha256"] == module._sha256_bytes(model)
    assert profile["model_segment"]["container"]["brotli_decodable"] is True
    assert profile["model_segment"]["container"]["decoded_payload_kind"] == "pr85_qh0_joint_frame_model"
    assert profile["model_segment"]["decoded"]["sha256"] == module._sha256_bytes(decoded_model)
    assert (
        profile["model_segment"]["decoded_windows"]["summary"]["zero_heavy_window_count"]
        >= 1
    )
    routes = {route["route_id"]: route for route in profile["candidate_routes"]}
    assert routes["lossless_brotli_recode_decoded_model_segment"][
        "estimated_model_segment_bytes_saved"
    ] >= 0
    assert routes["lossless_brotli_recode_decoded_model_segment"]["dispatchable_now"] is False
    assert profile["top_implementable_route"]["planning_only"] is True


def test_fails_closed_on_non_pr85_member_name(tmp_path: Path) -> None:
    brotli = pytest.importorskip("brotli")
    archive = tmp_path / "archive.zip"
    _write_archive(archive, model=brotli.compress(b"QH0payload"), member_name="p")

    with pytest.raises(module.ProfileError, match="strict PR85 archive"):
        module.build_profile(archive, bit_budget_json=None, bundle_profile_json=None)


def test_cli_writes_deterministic_json_and_markdown(tmp_path: Path, capsys) -> None:
    brotli = pytest.importorskip("brotli")
    archive = tmp_path / "archive.zip"
    json_out = tmp_path / "profile.json"
    md_out = tmp_path / "profile.md"
    _write_archive(archive, model=brotli.compress(b"QH0" + b"abc" * 512, quality=5))

    rc = module.main(
        [
            "--archive",
            str(archive),
            "--bit-budget-json",
            str(tmp_path / "missing_budget.json"),
            "--bundle-profile-json",
            str(tmp_path / "missing_bundle.json"),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
            "--window-bytes",
            "256",
        ]
    )

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["model_segment"]["container"]["recognized_runtime_model_payload"] is True
    assert payload["artifact_consistency"]["bit_budget_json_present"] is False
    assert "PR85 Model Payload Self-Compression Profile" in md_out.read_text(encoding="utf-8")
    assert '"dispatch_performed": false' in capsys.readouterr().out
