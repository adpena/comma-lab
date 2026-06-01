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
    monkeypatch.setattr(module, "run_snerv_advisory", lambda **_kwargs: fake)

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
            "--package-timeout-seconds",
            "120",
        ]
    )
    payload = json.loads(report_path.read_text())

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
    assert payload["archive_byte_closure_blockers"] == [
        "snerv_packet_not_full_600_pairs",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "pywavelets_runtime_dependency_not_contest_proven",
    ]
    assert payload["runtime_package"]["receiver_proof"][
        "runtime_consumption_proof_passed"
    ] is True


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
