# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV advisory packet/package CLI wiring."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    build_snerv_receiver_archive_proof,
)


def test_advisory_cli_writes_real_packet_and_runtime_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )
    repo_root = Path(__file__).resolve().parents[5]
    cli_path = repo_root / "tools/run_snerv_inverse_steg_advisory.py"
    spec = importlib.util.spec_from_file_location(
        "run_snerv_inverse_steg_advisory_for_test",
        cli_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = _fake_advisory_result(archive.packet)
    captured_kwargs = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_kwargs.update(kwargs)
        return fake

    monkeypatch.setattr(module, "run_snerv_advisory", fake_run_snerv_advisory)

    report_path = tmp_path / "advisory.json"
    packet_path = tmp_path / "advisory.snar"
    package_dir = tmp_path / "package"
    rc = module.main(
        [
            "--n-pairs",
            "1",
            "--levels",
            "1",
            "--out",
            str(report_path),
            "--packet-out",
            str(packet_path),
            "--package-dir",
            str(package_dir),
            "--decoder-payload-codec",
            "mixed_magnitude_symmetric",
            "--decoder-payload-mixed-modes",
            "fp16,int4,int4",
            "--package-timeout-seconds",
            "120",
            "--trained-ladder-row-out",
            str(tmp_path / "trained_ladder_row.json"),
        ]
    )
    payload = json.loads(report_path.read_text())
    trained_row = payload["trained_ladder_row_payload"]

    assert rc == 0
    assert packet_path.read_bytes() == archive.packet
    assert (package_dir / "archive.zip").is_file()
    assert payload["receiver_archive_packet"]["redacted"] is True
    assert payload["receiver_archive_packet_path"] == str(packet_path)
    assert payload["archive_byte_closure_blockers_before_package"] == [
        "full_600_pair_receiver_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "not_packaged_as_contest_archive_zip",
    ]
    assert set(payload["archive_byte_closure_blockers"]) >= {
        "snerv_packet_not_full_600_pairs",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "pywavelets_runtime_dependency_not_contest_proven",
    }
    assert payload["runtime_package"]["receiver_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert trained_row["schema"] == "nerv_trained_ladder_row_payload.v1"
    assert trained_row["status"] == "trained_ladder_row_blocked"
    assert trained_row["archive_path_kind"] == "contest_archive_zip"
    assert trained_row["archive_custody"]["archive_path"].endswith(
        "package/archive.zip"
    )
    assert trained_row["rows"][0]["archive_bytes"] == (
        package_dir / "archive.zip"
    ).stat().st_size
    assert "sample_pair_count_below_full600" in trained_row["blockers"]
    assert Path(payload["trained_ladder_row_payload_path"]).is_file()
    assert captured_kwargs["decoder_payload_codec"] == "mixed_magnitude_symmetric"
    assert captured_kwargs["decoder_payload_mixed_modes"] == ("fp16", "int4", "int4")


def _fake_advisory_result(packet: bytes):
    base = {
        "n_pairs": 1,
        "levels": 1,
        "wavelet": "db2",
        "carrier_hw": [16, 24],
        "receiver_archive_packet": {
            "bytes": len(packet),
            "sha256": "fake",
            "redacted": True,
        },
        "archive_byte_closure_blockers": [
            "full_600_pair_receiver_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
            "not_packaged_as_contest_archive_zip",
        ],
    }
    return SimpleNamespace(
        receiver_archive_packet=packet,
        axis_tag="[macOS-CPU advisory]",
        carrier_hw=(16, 24),
        n_pairs=1,
        levels=1,
        wavelet="db2",
        adjoint_rel_residual=0.0,
        lf_coeff_count_total=384,
        lf_payload_bytes=100,
        linf_steps_payload_bytes=80,
        linf_steps_payload_codec="snerv_step_map_coder.v1",
        linf_steps_coder_mode="uniform",
        linf_steps_coder_bins=4,
        linf_steps_fp32_lzma_baseline_bytes=200,
        linf_steps_max_relative_error=0.0,
        linf_steps_coder_groups=(),
        metadata_bytes=24,
        receiver_archive_header_bytes=64,
        decoder_bytes=40,
        decoder_payload_codec="float32_lzma",
        decoder_payload_header={"schema": "snerv_decoder_payload.v1"},
        snerv_fc_dim=9,
        snerv_emb_size=0,
        snerv_patch_radius=1,
        decoder_feature_count=9,
        hf_decoder_fit_mode="least_squares",
        hf_decoder_saliency_gain=1.0,
        hf_decoder_saliency_component="combined",
        archive_bytes_total=len(packet),
        receiver_archive_sha256="fake",
        rate_term=0.001,
        pr101_frontier_bytes=178493,
        pr101_frontier_rate=0.1,
        beats_frontier_rate=True,
        d_seg_mean_linf=0.1,
        d_pose_mean_linf=0.2,
        score_linf=1.0,
        d_seg_mean_l2=0.2,
        d_pose_mean_l2=0.3,
        score_l2=2.0,
        z8_disease_detail_store_frac=0.5,
        z8_falsification_verdict="test",
        as_jsonable=lambda: dict(base),
    )
