from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import FIXED_V5_LENGTHS, pack_pr85_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_nonmask_self_compression.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_nonmask_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _br(data: bytes, *, quality: int) -> bytes:
    brotli = pytest.importorskip("brotli")
    return brotli.compress(data, quality=quality)


def _write_archive(path: Path, *, post: bytes | None = None, member_name: str = "x") -> dict[str, bytes]:
    segments = {
        "mask": b"QMA9" + b"m" * 800,
        "model": _br(b"QH0" + bytes((idx % 251 for idx in range(2048))), quality=11),
        "pose": _br(b"P1D1" + bytes((idx % 17 for idx in range(512))), quality=11),
        "post": post if post is not None else _br(bytes((idx % 251 for idx in range(2048))), quality=11),
        "shift": _br(b"SD4" + b"\x00" * 300, quality=11),
        "frac": _br(b"FV1" + b"\x01\x02" * 80, quality=11),
        "frac2": _br(b"FH2" + b"\x04" * 300, quality=11),
        "frac3": _br(b"FD3" + b"\x00" * 300, quality=11),
        "bias": b"B" * FIXED_V5_LENGTHS["bias"],
        "region": b"R" * FIXED_V5_LENGTHS["region"],
        "randmulti": _br(b"\x00\x01\x02\x03" * 200, quality=11),
    }
    raw = pack_pr85_bundle(segments, header_mode="v5")
    info = zipfile.ZipInfo(member_name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)
    return segments


def _write_pr90_probe(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "payload_len": 1200,
                "split_mode": "synthetic",
                "compact_constants": {"model_body_bytes": 10},
                "slices": {"pose_qrgb_body": {"len": 20}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_rejects_noop_recompression_and_records_overhead(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    pr90 = tmp_path / "pr90.json"
    _write_archive(archive)
    _write_pr90_probe(pr90)

    profile = module.build_profile(archive, pr90_probe_json=pr90, pr91_archive=None)

    assert profile["schema"] == module.SCHEMA
    assert profile["planning_only"] is True
    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["bundle"]["mask_segment_excluded_from_candidate_search"]["reason"]
    assert profile["single_blob_container_overhead"]["zip_container_overhead_bytes"] == 100
    assert profile["single_blob_container_overhead"]["arbitrary_extra_zip_overhead_bytes"] == 0
    assert profile["single_blob_container_overhead"]["overhead_recommendation"].startswith(
        "No direct single-blob overhead candidate"
    )
    assert profile["fail_closed_summary"]["no_op_recommendations_promoted"] == 0


def test_detects_lossless_variable_segment_recode_candidate(tmp_path: Path) -> None:
    poorly_compressed_post = _br((b"POST" + b"\x00" * 64) * 256, quality=1)
    archive = tmp_path / "archive.zip"
    _write_archive(archive, post=poorly_compressed_post)

    profile = module.build_profile(archive, pr90_probe_json=None, pr91_archive=None)

    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in profile["lossless_archive_builder_candidates"]
    }
    candidate = candidates["lossless_brotli_recode_pr85_post_segment"]
    assert candidate["expected_archive_delta_bytes"] < 0
    assert candidate["expected_rate_score_delta_formula_only"] < 0
    assert candidate["no_op"] is False
    assert candidate["state_change_required"] is True
    assert candidate["runtime_risk"] == "low"
    rows = {row["name"]: row for row in profile["nonmask_segments"]}
    assert rows["post"]["candidate_assessment"]["lossless_decoded_recode_candidate"] is not None


def test_cli_writes_json_and_markdown(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "archive.zip"
    json_out = tmp_path / "audit.json"
    markdown_out = tmp_path / "audit.md"
    _write_archive(archive)

    rc = module.main(
        [
            "--archive",
            str(archive),
            "--pr90-probe-json",
            str(tmp_path / "missing_pr90.json"),
            "--pr91-archive",
            str(tmp_path / "missing_pr91.zip"),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["deterministic"] is True
    assert payload["comparisons"]["pr91_nonmask_identity"]["available"] is False
    assert "PR85 Non-Mask Self-Compression Audit" in markdown_out.read_text(encoding="utf-8")
    assert '"local_only": true' in capsys.readouterr().out
