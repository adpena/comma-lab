# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "experiments" / "profile_archive_byte_accounting.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("profile_archive_byte_accounting", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _det_bytes(label: str, n_bytes: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out.extend(hashlib.sha256(f"{label}:{counter}".encode("utf-8")).digest())
        counter += 1
    return bytes(out[:n_bytes])


def _rpk1_payload() -> bytes:
    members = [
        ("renderer.bin", b"renderer-weights" * 17),
        ("masks.mkv", b"\x00\x01\x02\x03mask-obu" * 29),
        ("optimized_poses.bin", b"pose" * 61),
    ]
    header = {
        "schema": "renderer_payload_v1",
        "members": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256(data),
                "codec": "raw",
                "decoded_bytes": len(data),
                "decoded_sha256": _sha256(data),
            }
            for name, data in members
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"RPK1" + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(
        data for _, data in members
    )


def _write_single_member_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _write_named_single_member_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_profile_rpk1_archive_records_streams_and_no_score_claim(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    _write_single_member_zip(archive, _rpk1_payload())

    profile = module.build_profile(archive=archive, output_json=output)

    assert profile["schema"] == "archive_byte_accounting_profile_v1"
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert output.exists()
    assert profile["zip_container"]["member_count"] == 1
    assert profile["zip_container"]["overhead_bytes"] > 0
    assert profile["single_payload"]["zip_member_name"] == "p"
    assert profile["single_payload"]["payload_internal_overhead_bytes"] > 0
    container_probe = profile["single_payload"]["finite_container_probe"]
    assert container_probe["score_claim"] is False
    assert container_probe["best_option"]["archive_bytes"] > 0
    assert any(
        row["option_id"] == "zip_stored_raw_p"
        and row["runtime_supported_by_current_unpacker"] is True
        for row in container_probe["options"]
    )
    stream_names = {stream["name"] for stream in profile["streams"]}
    assert stream_names == {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
    for stream in profile["streams"]:
        assert stream["encoded_entropy"]["bytes"] == stream["encoded_bytes"]
        assert stream["encoded_bitplanes"]["bits"] == stream["encoded_bytes"] * 8
        assert len(stream["encoded_bitplanes"]["planes_lsb0"]) == 8
        assert "best_probe" in stream["encoded_self_compression_probe"]
        assert stream["self_compression_signal"]["encoded_probe_is_directly_deployable"] is False


def test_profile_public_pr65_compact_bundle_records_qpost_streams(tmp_path: Path) -> None:
    brotli = __import__("brotli")
    module = _load_module()
    archive = tmp_path / "pr65.zip"
    output = tmp_path / "profile.json"
    raw_streams = [
        b"\x12\x00" + _det_bytes("mask-obu", 2200),
        b"QH0" + _det_bytes("model", 2200),
        b"P1D1" + _det_bytes("pose", 280),
        bytes([1, 2, 3]) * 600,
        b"SD4" + bytes([40]) * 600,
        b"FV1" + bytes([4]) * 120,
        b"FH2" + bytes([4]) * 600,
        b"FD3" + bytes([0]) * 600,
        b"BD1" + bytes([0]) * 600,
        b"RH1" + bytes([0]) * 600,
    ]
    encoded = [brotli.compress(stream, quality=5) for stream in raw_streams]
    lengths = [len(stream) for stream in encoded]
    header = b"".join(int(n).to_bytes(3, "little") for n in lengths)
    randmulti = brotli.compress(b"NM1" + bytes([1]) + bytes(600), quality=5)
    _write_named_single_member_zip(archive, "x", header + b"".join(encoded) + randmulti)

    profile = module.build_profile(archive=archive, output_json=output)

    assert profile["single_payload"]["zip_member_name"] == "x"
    assert profile["single_payload"]["payload_format"] == "public_pr65_qpost_compact_v4"
    assert profile["single_payload"]["payload_internal_overhead_bytes"] == 30
    names = {stream["name"] for stream in profile["streams"]}
    assert {"masks.mkv", "renderer.bin", "optimized_poses.bin", "qpost.post", "qpost.randmulti"} <= names
    qpost = [stream for stream in profile["streams"] if stream["name"].startswith("qpost.")]
    assert qpost
    assert all(stream["attackability"]["priority"] == 4 for stream in qpost)
    assert profile["score_claim"] is False


def test_profile_eval_json_adds_sub03_target_pressure(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    eval_json = tmp_path / "contest_auth_eval.json"
    _write_single_member_zip(archive, _rpk1_payload())
    eval_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.31561703078448233,
                "archive_size_bytes": archive.stat().st_size,
                "avg_segnet_dist": 0.00061244,
                "avg_posenet_dist": 0.00049637,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": module._sha256_file(archive),
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
            }
        )
    )

    profile = module.build_profile(archive=archive, output_json=output, eval_json=eval_json)

    target_gap = profile["target_gap"]
    assert target_gap["available"] is True
    assert target_gap["bytes_to_remove_if_distortion_unchanged"] > 0
    assert target_gap["score_source"] == "score_recomputed_from_components"
    assert profile["eval_json"]["target_gap"] == target_gap
    assert profile["thirty_k_foot_summary"]["bytes_to_remove_if_distortion_unchanged"] == target_gap[
        "bytes_to_remove_if_distortion_unchanged"
    ]
    assert all(stream["target_gap_pressure"] is not None for stream in profile["streams"])
    assert profile["eval_json"]["matches_profiled_archive"] is True
    assert target_gap["reference_matches_profiled_archive"] is True


def test_profile_eval_json_prefers_canonical_score_over_rounded_final(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    eval_json = tmp_path / "contest_auth_eval.json"
    _write_single_member_zip(archive, _rpk1_payload())
    eval_json.write_text(
        json.dumps(
            {
                "canonical_score": 0.31561703078448233,
                "final_score": 0.32,
                "archive_size_bytes": archive.stat().st_size,
                "avg_segnet_dist": 0.00061244,
                "avg_posenet_dist": 0.00049637,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": module._sha256_file(archive),
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
            }
        )
    )

    profile = module.build_profile(archive=archive, output_json=output, eval_json=eval_json)

    target_gap = profile["target_gap"]
    assert target_gap["available"] is True
    assert target_gap["score_source"] == "canonical_score"
    assert target_gap["score"] == pytest.approx(0.31561703078448233)


def test_profile_eval_json_mismatch_is_flagged_as_reference_only(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    eval_json = tmp_path / "contest_auth_eval.json"
    _write_single_member_zip(archive, _rpk1_payload())
    eval_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.31561703078448233,
                "archive_size_bytes": archive.stat().st_size + 7,
                "avg_segnet_dist": 0.00061244,
                "avg_posenet_dist": 0.00049637,
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": "0" * 64,
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
            }
        )
    )

    profile = module.build_profile(archive=archive, output_json=output, eval_json=eval_json)

    assert profile["eval_json"]["matches_profiled_archive"] is False
    assert profile["eval_json"]["reference_warning"]
    assert profile["target_gap"]["reference_matches_profiled_archive"] is False
    assert "reference gap only" in profile["target_gap"]["reference_warning"]


def test_component_trace_and_action_records_feed_atom_table(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    trace = tmp_path / "component_trace.json"
    actions = tmp_path / "actions.json"
    _write_single_member_zip(archive, _rpk1_payload())
    archive_sha = module._sha256_file(archive)
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "score_claim": False,
                "evidence_grade": "diagnostic_component_trace",
                "n_samples": 2,
                "archive_size_bytes": archive.stat().st_size,
                "avg_posenet_dist": 0.0002,
                "avg_segnet_dist": 0.0003,
                "score_recomputed_from_components": 0.4,
                "trace_inputs": {"archive_sha256": archive_sha},
                "samples": [
                    {
                        "pair_index": 1,
                        "frame_start": 2,
                        "frame_indices": [2, 3],
                        "posenet_dist": 0.0007,
                        "segnet_dist": 0.0009,
                        "score_pose_contribution_first_order": 0.003,
                        "score_seg_contribution_exact": 0.002,
                    },
                    {
                        "pair_index": 0,
                        "frame_start": 0,
                        "frame_indices": [0, 1],
                        "posenet_dist": 0.0001,
                        "segnet_dist": 0.0002,
                        "score_combined_contribution_first_order": 0.001,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    actions.write_text(
        json.dumps(
            {
                "action_records": [
                    {
                        "action_id": "mask_patch_001",
                        "stream": "masks.mkv",
                        "charged_bytes": 80,
                        "byte_delta": -100,
                        "score_delta": -0.001,
                        "segnet_delta": -0.00001,
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    profile = module.build_profile(
        archive=archive,
        output_json=output,
        component_trace=trace,
        action_jsons=[actions],
    )

    assert profile["component_trace"]["matches_profiled_archive"] is True
    row_types = {row["row_type"] for row in profile["rate_distortion_atom_table"]}
    assert {"packed_stream", "action_record", "component_trace_pair", "zip_member"} <= row_types
    action = next(row for row in profile["rate_distortion_atom_table"] if row["atom_id"] == "mask_patch_001")
    assert action["benefit_score_estimate"] == pytest.approx(0.001)
    assert action["benefit_per_byte_estimate"] == pytest.approx(0.001 / 80)
    hard_pair = next(
        row for row in profile["rate_distortion_atom_table"] if row["atom_id"] == "component_trace_pair:1"
    )
    assert hard_pair["benefit_score_estimate"] == pytest.approx(0.005)
    assert hard_pair["break_even_bytes_at_rate_only"] > 0


def test_collection_cli_writes_combined_json_and_csv(tmp_path: Path) -> None:
    module = _load_module()
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    output = tmp_path / "collection.json"
    output_csv = tmp_path / "collection.csv"
    output_dir = tmp_path / "profiles"
    _write_single_member_zip(archive_a, _rpk1_payload())
    _write_single_member_zip(archive_b, _rpk1_payload())

    rc = module.main(
        [
            "--archive",
            str(archive_a),
            "--archive",
            str(archive_b),
            "--output-json",
            str(output),
            "--output-csv",
            str(output_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["schema"] == "archive_byte_accounting_profile_collection_v1"
    assert payload["archive_count"] == 2
    assert len(payload["profiles"]) == 2
    assert all(Path(row["collection_profile_json"]).exists() for row in payload["archives"])
    csv_text = output_csv.read_text()
    assert "candidate_index" in csv_text
    assert "packed_stream:renderer.bin" in csv_text


def test_markdown_report_keeps_profile_empirical(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    _write_single_member_zip(archive, _rpk1_payload())

    profile = module.build_profile(archive=archive, output_json=output)
    report = module._markdown_report(profile)

    assert "Archive Byte Accounting" in report
    assert "not score evidence" in report
    assert "promotion eligible: `False`" in report
    assert "| stream | encoded bytes | decoded bytes |" in report


def test_png_report_is_reproducible_artifact(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    png = tmp_path / "profile.png"
    _write_single_member_zip(archive, _rpk1_payload())

    profile = module.build_profile(archive=archive, output_json=output)
    module._write_png_report(profile, png)

    assert png.exists()
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_profile_accepts_brotli_wrapped_single_member_payload(tmp_path: Path) -> None:
    brotli = __import__("brotli")
    module = _load_module()
    archive = tmp_path / "archive.zip"
    output = tmp_path / "profile.json"
    wrapped = brotli.compress(_rpk1_payload(), quality=5)
    _write_single_member_zip(archive, wrapped)

    profile = module.build_profile(archive=archive, output_json=output)

    assert profile["single_payload"]["payload_container_codec"] == "brotli"
    assert profile["single_payload"]["zip_member_payload_bytes"] == len(wrapped)
    assert profile["single_payload"]["payload_bytes"] > len(wrapped)
    assert {stream["name"] for stream in profile["streams"]} == {
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.bin",
    }


def test_compression_probe_records_optional_zstd_cli() -> None:
    module = _load_module()

    probe = module._compression_probe(b"mask-bytes" * 256)

    assert "best_probe" in probe
    if shutil.which("zstd") is not None:
        assert "zstd_3_bytes" in probe
