# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import shlex
import stat
import struct
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import verify_z7_exact_eval_handoff as handoff  # noqa: E402

Z7_MAMBA2_SUBSTRATE_ID = "time_traveler_l5_z7_mamba2"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _write_zip(path: Path, payload: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", payload)
    return path.stat().st_size, _sha256(path)


def _z7mcm2_payload(
    *,
    num_pairs: int,
    output_height: int = 16,
    output_width: int = 16,
    tag: str = "R",
    identity_predictor: bool = False,
) -> bytes:
    latent_dim = 2
    ego_dim = 1
    d_model = 4
    d_state = 1
    expand = 1
    d_conv = 1
    flags = 2 if identity_predictor else 0
    encoder_blob = b""
    decoder_blob = b""
    predictor_blob = b"P"
    latent_init_blob = bytes([1] * latent_dim)
    residuals_blob = bytes([2] * (num_pairs * latent_dim))
    ego_blob = bytes([3] * (num_pairs * ego_dim))
    meta_blob = json.dumps(
        {
            "output_height": output_height,
            "output_width": output_width,
            "fixture_tag": tag,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header = struct.pack(
        handoff.Z7MCM2_HEADER_FMT,
        handoff.Z7MCM2_MAGIC,
        handoff.Z7MCM2_SCHEMA_VERSION,
        latent_dim,
        ego_dim,
        num_pairs,
        d_model,
        d_state,
        expand,
        d_conv,
        flags,
        len(encoder_blob),
        len(decoder_blob),
        len(predictor_blob),
        len(latent_init_blob),
        len(residuals_blob),
        len(ego_blob),
        len(meta_blob),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + predictor_blob
        + latent_init_blob
        + residuals_blob
        + ego_blob
        + meta_blob
    )


def _runtime_geometry_positive_control(
    *,
    num_pairs: int,
    output_height: int = 16,
    output_width: int = 16,
    expected_raw_bytes_delta: int = 0,
) -> dict[str, object]:
    expected_frames = num_pairs * 2
    expected_raw_bytes = (
        expected_frames
        * 3
        * handoff.CAMERA_HW[0]
        * handoff.CAMERA_HW[1]
        + expected_raw_bytes_delta
    )
    return {
        "schema": handoff.RUNTIME_GEOMETRY_POSITIVE_CONTROL_SCHEMA,
        "num_pairs": num_pairs,
        "render_hw": [output_height, output_width],
        "camera_hw": list(handoff.CAMERA_HW),
        "expected_frames_written": expected_frames,
        "expected_raw_bytes": expected_raw_bytes,
        "sample_pair_indices": sorted({0, num_pairs // 2, num_pairs - 1}),
        "recurrent_sampled_raw_sha256": hashlib.sha256(
            b"recurrent-sample"
        ).hexdigest(),
        "static_sampled_raw_sha256": hashlib.sha256(
            b"static-sample"
        ).hexdigest(),
        "recurrent_static_sample_changed": True,
        "archive_header": {
            "num_pairs": num_pairs,
            "output_height": output_height,
            "output_width": output_width,
        },
    }


def _fixture_repo(
    tmp_path: Path,
    *,
    num_pairs: int = 600,
    lane_id: str | None = None,
    substrate_id: str | None = None,
    pair_group_id: str | None = None,
    include_runtime_geometry_positive_control: bool | None = None,
    runtime_geometry_expected_raw_bytes_delta: int = 0,
) -> Path:
    effective_substrate_id = substrate_id or handoff.SUBSTRATE_ID
    is_mamba2 = effective_substrate_id == Z7_MAMBA2_SUBSTRATE_ID
    recurrent_payload = (
        _z7mcm2_payload(num_pairs=num_pairs, tag="R")
        if is_mamba2
        else b"recurrent-z7-payload"
    )
    static_payload = (
        _z7mcm2_payload(num_pairs=num_pairs, tag="S", identity_predictor=True)
        if is_mamba2
        else b"staticctrl-z7payload"
    )
    recurrent_bytes, recurrent_sha = _write_zip(
        tmp_path / "runs/z7/archive.zip",
        recurrent_payload,
    )
    static_bytes, static_sha = _write_zip(
        tmp_path / "runs/z7/static_capacity_control/archive.zip",
        static_payload,
    )
    expected_frames = num_pairs * 2
    expected_raw_bytes = (
        expected_frames * 3 * handoff.CAMERA_HW[0] * handoff.CAMERA_HW[1]
    )
    runtime_dir = tmp_path / "runs/z7/submission_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR)
    (runtime_dir / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    stats = {
        "archive_zip_bytes": recurrent_bytes,
        "archive_zip_path": "runs/z7/archive.zip",
        "archive_zip_sha256": recurrent_sha,
        "config": {"num_pairs": num_pairs},
        "lane_id": lane_id or handoff.LANE_ID,
        "loss_mode": "score_aware",
        "device_runtime_contract": {
            "device_type": "mps",
            "inflate_verify_device": "cpu",
            "mps_research_signal_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ready_for_paid_dispatch": False,
            "rank_or_kill_eligible": False,
        },
        "inflate_verify": {
            "device": "cpu",
            "frames_written": expected_frames if is_mamba2 else 1200,
            "raw_bytes": expected_raw_bytes if is_mamba2 else 1,
            "raw_sha256": "0" * 64,
        },
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "score_aware_scorer_loss_used": True,
        "score_claim": False,
        "static_capacity_control": {
            "archive_zip_bytes": static_bytes,
            "archive_zip_path": "runs/z7/static_capacity_control/archive.zip",
            "archive_zip_sha256": static_sha,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ready_for_paid_dispatch": False,
            "inflate_verify_frames_written": (
                expected_frames if is_mamba2 else None
            ),
            "inflate_verify_raw_bytes": expected_raw_bytes if is_mamba2 else None,
            "runtime_output_byte_differences_vs_recurrent": 7,
            "runtime_output_changed_vs_recurrent": True,
            "same_archive_zip_bytes_as_recurrent": recurrent_bytes == static_bytes,
            "score_claim": False,
        },
        "submission_runtime_dir": "runs/z7/submission_runtime",
        "substrate_id": effective_substrate_id,
    }
    should_include_runtime_geometry = (
        is_mamba2
        if include_runtime_geometry_positive_control is None
        else include_runtime_geometry_positive_control
    )
    if should_include_runtime_geometry:
        stats["runtime_geometry_positive_control"] = (
            _runtime_geometry_positive_control(
                num_pairs=num_pairs,
                expected_raw_bytes_delta=runtime_geometry_expected_raw_bytes_delta,
            )
        )
    if pair_group_id is not None:
        stats["pair_group_id"] = pair_group_id
    (tmp_path / "runs/z7/stats.json").write_text(
        json.dumps(stats),
        encoding="utf-8",
    )
    return tmp_path


def test_z7_handoff_blocks_current_one_pair_packet_and_plan_commands() -> None:
    payload = handoff.build_packet(repo_root=REPO_ROOT)

    assert payload["current_pair_count"] == 1
    assert payload["required_pair_count"] == 600
    assert payload["ready_for_exact_eval_handoff"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["provider_dispatch_attempted"] is False
    assert payload["lane_claim_opened"] is False
    assert set(payload["result_review_blockers"]) == {
        "z7_exact_handoff_current_packet_not_600_pairs",
        "z7_exact_handoff_inflate_verify_device_not_cpu_or_cuda",
    }
    assert payload["same_archive_zip_bytes"] is True
    assert payload["runtime_output_changed_vs_recurrent"] is True
    assert payload["modal_plan_commands_for_current_packet"] == {}


def test_z7_handoff_ready_for_ratified_full_pair_packet(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is True
    assert payload["result_review_blockers"] == []
    assert payload["current_pair_count"] == 600
    assert payload["runtime_custody"]["inflate_sh_executable"] is True
    assert {row["axis"] for row in payload["axis_plan"]} == {
        "[contest-CUDA]",
        "[contest-CPU]",
    }
    for row in payload["axis_plan"]:
        assert row["inflate_device_policy"] == "auto"
        assert row["evaluate_device"] in {"cpu", "cuda"}
        assert row["required_hardware"] in {"linux_x86_64_cpu", "linux_x86_64_t4"}
        assert row["authority_precondition"].startswith("modal_paired_auth_eval_")
    assert set(payload["modal_execute_commands_after_ratified_full_packet"]) == {
        "recurrent_paired_contest_cpu_cuda",
        "static_control_paired_contest_cpu_cuda",
    }
    for command in payload["modal_execute_commands_after_ratified_full_packet"].values():
        assert "--execute" in command
        assert "--expected-runtime-tree-sha256 auto" in command
        assert "--skip-axis-if-promotable-anchor-exists" in command


def test_z7_handoff_surfaces_failed_inflate_verify(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["inflate_verify"] = {
        "verify_failed": "unsupported PACT_INFLATE_DEVICE='mps'; expected auto/cpu/cuda"
    }
    static = stats["static_capacity_control"]
    static.pop("runtime_output_changed_vs_recurrent")
    static.pop("runtime_output_byte_differences_vs_recurrent")
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert "z7_exact_handoff_inflate_verify_failed" in payload["result_review_blockers"]
    assert (
        "z7_exact_handoff_static_control_inflate_output_evidence_missing"
        in payload["result_review_blockers"]
    )
    assert (
        "z7_exact_handoff_static_control_runtime_output_not_changed"
        not in payload["result_review_blockers"]
    )
    assert (
        "z7_exact_handoff_static_control_byte_differences_not_positive"
        not in payload["result_review_blockers"]
    )


def test_z7_handoff_refuses_mps_inflate_verify_device(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["inflate_verify"]["device"] = "mps"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert "z7_exact_handoff_inflate_verify_device_not_cpu_or_cuda" in payload[
        "result_review_blockers"
    ]
    assert "z7_exact_handoff_mps_training_must_cpu_inflate_verify" in payload[
        "result_review_blockers"
    ]


def test_z7_handoff_derives_mamba_identity_from_stats(tmp_path: Path) -> None:
    repo = _fixture_repo(
        tmp_path,
        num_pairs=600,
        lane_id="lane_z7_as_mamba_2_full_landing_20260518",
        substrate_id=Z7_MAMBA2_SUBSTRATE_ID,
    )
    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is True
    assert payload["lane_id"] == "lane_z7_as_mamba_2_full_landing_20260518"
    assert payload["substrate_id"] == Z7_MAMBA2_SUBSTRATE_ID
    assert payload["runtime_geometry_positive_control"]["schema"] == (
        handoff.RUNTIME_GEOMETRY_POSITIVE_CONTROL_SCHEMA
    )
    assert payload["source_archive_rows"][0]["z7mcm2_header"] == {
        "num_pairs": 600,
        "output_height": 16,
        "output_width": 16,
    }
    assert (
        payload["pair_group_id"]
        == "z7_mamba2_temporal_coherence_vs_static_capacity_same_bytes"
    )
    commands = payload["modal_execute_commands_after_ratified_full_packet"]
    assert commands
    for command in commands.values():
        assert "lane_z7_as_mamba_2_full_landing_20260518_exact_eval_" in command
        assert "time_traveler_l5_z7_mamba2_" in command
        assert "z7_mamba2_temporal_coherence_vs_static_capacity_same_bytes" in command
        assert "lane_z7_lstm_predictive_coding_20260518" not in command
        assert "time_traveler_l5_z7_lstm_predictive_coding_" not in command
        parts = shlex.split(command)
        pair_group = parts[parts.index("--pair-group-id") + 1]
        assert pair_group == "z7_mamba2_temporal_coherence_vs_static_capacity_same_bytes"
        assert "_recurrent_" not in pair_group
        assert "_static-control_" not in pair_group


def test_z7_mamba2_handoff_refuses_missing_runtime_geometry_positive_control(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(
        tmp_path,
        num_pairs=600,
        substrate_id=Z7_MAMBA2_SUBSTRATE_ID,
        include_runtime_geometry_positive_control=False,
    )

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert payload["runtime_geometry_positive_control"] is None
    assert "z7_exact_handoff_runtime_geometry_positive_control_missing" in payload[
        "result_review_blockers"
    ]


def test_z7_mamba2_handoff_refuses_runtime_geometry_mismatch(tmp_path: Path) -> None:
    repo = _fixture_repo(
        tmp_path,
        num_pairs=600,
        substrate_id=Z7_MAMBA2_SUBSTRATE_ID,
        runtime_geometry_expected_raw_bytes_delta=1,
    )

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert (
        "z7_exact_handoff_runtime_geometry_expected_raw_bytes_mismatch"
        in payload["result_review_blockers"]
    )
    assert (
        "z7_exact_handoff_runtime_geometry_recurrent_raw_bytes_mismatch"
        not in payload["result_review_blockers"]
    )


def test_z7_mamba2_handoff_refuses_archive_header_geometry_mismatch(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(
        tmp_path,
        num_pairs=600,
        substrate_id=Z7_MAMBA2_SUBSTRATE_ID,
    )
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["runtime_geometry_positive_control"]["archive_header"]["output_width"] = 17
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert (
        "z7_exact_handoff_runtime_geometry_archive_header_output_width_mismatch"
        in payload["result_review_blockers"]
    )


def test_z7_handoff_refuses_false_authority_stats_and_hides_plan_commands(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["score_claim"] = True
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert "z7_exact_handoff_stats_score_claim_not_false" in payload[
        "result_review_blockers"
    ]
    assert payload["modal_plan_commands_for_current_packet"] == {}
    assert payload["modal_execute_commands_after_ratified_full_packet"] == {}


def test_z7_handoff_refuses_ready_for_exact_eval_authority_stats(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, num_pairs=600)
    stats_path = repo / "runs/z7/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["ready_for_exact_eval_dispatch"] = True
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    payload = handoff.build_packet(repo_root=repo, stats_json=Path("runs/z7/stats.json"))

    assert payload["ready_for_exact_eval_handoff"] is False
    assert "z7_exact_handoff_stats_ready_for_exact_eval_dispatch_not_false" in payload[
        "result_review_blockers"
    ]
    assert payload["modal_plan_commands_for_current_packet"] == {}
    assert payload["modal_execute_commands_after_ratified_full_packet"] == {}
