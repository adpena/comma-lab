# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV advisory packet/package CLI wiring."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
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
            "--lf-payload-codec",
            "portfolio_auto",
            "--snerv-spectra-preserving-adapter",
            "--snerv-mfu-scales",
            "1,3",
            "--snerv-hfr-gain",
            "0.25",
            "--snerv-temporal-context",
            "1",
            "--snerv-temporal-mode",
            "official_haar_dwt1d_lowpass",
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
    archive_zip_path = package_dir / "archive.zip"
    assert archive_zip_path.is_file()
    archive_zip_bytes = archive_zip_path.stat().st_size
    packet_accounting = payload["receiver_snar_packet_rate_accounting"]
    charged_accounting = payload["charged_archive_rate_accounting"]
    assert packet_accounting["archive_path_kind"] == "receiver_snar_packet"
    assert packet_accounting["archive_bytes_total"] == len(archive.packet)
    assert packet_accounting["score_linf"] == 1.0
    assert packet_accounting["chargeable_for_contest_submission"] is False
    assert charged_accounting["archive_path_kind"] == "contest_archive_zip"
    assert charged_accounting["archive_bytes"] == archive_zip_bytes
    assert charged_accounting["archive_bytes_total"] == archive_zip_bytes
    assert charged_accounting["packet_bytes_preserved_as_diagnostic"] == len(
        archive.packet
    )
    assert charged_accounting["chargeable_for_contest_submission"] is True
    assert payload["archive_bytes_total_before_package"] == len(archive.packet)
    assert payload["rate_term_before_package"] == 0.001
    assert payload["score_linf_before_package"] == 1.0
    assert payload["charged_archive_path_kind"] == "contest_archive_zip"
    assert payload["archive_bytes_total"] == archive_zip_bytes
    assert payload["rate_term"] == pytest.approx(
        module.CONTEST_BYTE_PRICE * archive_zip_bytes
    )
    assert payload["score_linf"] == pytest.approx(1.0 - 0.001 + payload["rate_term"])
    assert payload["score_l2_archive_path_kind"] == "receiver_snar_packet"
    assert "selected L-inf SNAR1 packet" in (
        payload["score_l2_package_rate_not_recomputed_reason"]
    )
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
    assert trained_row["rows"][0]["archive_bytes"] == archive_zip_bytes
    assert "sample_pair_count_below_full600" in trained_row["blockers"]
    assert Path(payload["trained_ladder_row_payload_path"]).is_file()
    assert captured_kwargs["decoder_payload_codec"] == "mixed_magnitude_symmetric"
    assert captured_kwargs["decoder_payload_mixed_modes"] == ("fp16", "int4", "int4")
    assert captured_kwargs["lf_payload_codec"] == "portfolio_auto"
    assert payload["lf_payload_codec"] == "portfolio_auto"
    assert captured_kwargs["snerv_model_size_adapter"] == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
    assert captured_kwargs["snerv_mfu_scales"] == (1, 3)
    assert captured_kwargs["snerv_hfr_gain"] == 0.25
    assert captured_kwargs["snerv_temporal_context"] == 1
    assert captured_kwargs["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"


def test_advisory_cli_forwards_official_modelsize_and_persists_solution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    cli_path = repo_root / "tools/run_snerv_inverse_steg_advisory.py"
    spec = importlib.util.spec_from_file_location(
        "run_snerv_inverse_steg_advisory_for_modelsize_test",
        cli_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official_solution = {
        "schema": "official_snerv_modelsize_to_fc_dim.v1",
        "source": "official_snerv_train_snerv_modelsize_quadratic_fc_dim_resolver_bound",
        "modelsize_mparams": 0.05,
        "full_data_length": 2,
        "final_size": 16 * 24,
        "enc_strds": [5, 4, 2, 2, 2],
        "dec_strds": [5, 4, 2, 2, 2],
        "fc_dim": 11,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    fake = _fake_advisory_result(
        b"SNAR1-modelsize",
        snerv_fc_dim=11,
        snerv_capacity_source="official_snerv_modelsize",
        official_modelsize_solution=official_solution,
    )
    captured_kwargs = {}

    def fake_run_snerv_advisory(**kwargs):
        captured_kwargs.update(kwargs)
        return fake

    monkeypatch.setattr(module, "run_snerv_advisory", fake_run_snerv_advisory)

    report_path = tmp_path / "advisory_modelsize.json"
    rc = module.main(
        [
            "--n-pairs",
            "1",
            "--levels",
            "1",
            "--out",
            str(report_path),
            "--snerv-official-modelsize-mparams",
            "0.05",
            "--snerv-official-enc-strds",
            "5,4,2,2,2",
            "--snerv-official-dec-strds",
            "5,4,2,2,2",
        ]
    )
    payload = json.loads(report_path.read_text())

    assert rc == 0
    assert captured_kwargs["snerv_official_modelsize_mparams"] == 0.05
    assert captured_kwargs["snerv_official_enc_strds"] == (5, 4, 2, 2, 2)
    assert captured_kwargs["snerv_official_dec_strds"] == (5, 4, 2, 2, 2)
    assert captured_kwargs["snerv_fc_dim"] == 9
    assert captured_kwargs["snerv_fc_dim_explicit"] is False
    assert payload["snerv_capacity_source"] == "official_snerv_modelsize"
    assert payload["official_modelsize_solution"]["modelsize_mparams"] == 0.05
    assert payload["official_modelsize_solution"]["fc_dim"] == 11
    assert payload["snerv_fc_dim"] == 11


def test_advisory_cli_records_native_mlx_training_knobs_as_unconsumed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    cli_path = repo_root / "tools/run_snerv_inverse_steg_advisory.py"
    spec = importlib.util.spec_from_file_location(
        "run_snerv_inverse_steg_advisory_for_native_control_test",
        cli_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = _fake_advisory_result(b"SNAR1-native-control")
    monkeypatch.setattr(module, "run_snerv_advisory", lambda **_kwargs: fake)

    report_path = tmp_path / "advisory_native_controls.json"
    rc = module.main(
        [
            "--n-pairs",
            "1",
            "--levels",
            "1",
            "--out",
            str(report_path),
            "--snerv-native-mlx-decoder-train-steps",
            "9",
            "--snerv-native-mlx-decoder-train-lr",
            "0.0007",
            "--snerv-native-mlx-decoder-train-ridge",
            "0.000003",
            "--snerv-native-mlx-decoder-train-optimizer",
            "sgd",
        ]
    )
    payload = json.loads(report_path.read_text())

    assert rc == 0
    controls = payload["native_mlx_decoder_training_controls"]
    assert controls["requested_steps"] == 9
    assert controls["learning_rate"] == pytest.approx(0.0007)
    assert controls["ridge"] == pytest.approx(0.000003)
    assert controls["optimizer"] == "sgd"
    assert controls["consumed_by_cli"] is False
    assert controls["native_mlx_training_executed"] is False
    assert (
        "snerv_native_mlx_decoder_training_controls_unreachable_from_cpu_advisory_cli"
        in controls["blockers"]
    )
    assert (
        "snerv_native_mlx_decoder_training_controls_unreachable_from_cpu_advisory_cli"
        in payload["blockers"]
    )


def _fake_advisory_result(
    packet: bytes,
    *,
    snerv_fc_dim: int = 9,
    snerv_capacity_source: str = "manual_fc_dim",
    official_modelsize_solution: dict[str, object] | None = None,
):
    base = {
        "n_pairs": 1,
        "levels": 1,
        "wavelet": "db2",
        "carrier_hw": [16, 24],
        "snerv_fc_dim": int(snerv_fc_dim),
        "snerv_capacity_source": snerv_capacity_source,
        "official_modelsize_solution": official_modelsize_solution,
        "receiver_archive_packet": {
            "bytes": len(packet),
            "sha256": "fake",
            "redacted": True,
        },
        "lf_payload_codec": "portfolio_auto",
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
        lf_payload_codec="portfolio_auto",
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
        snerv_fc_dim=int(snerv_fc_dim),
        snerv_emb_size=0,
        snerv_patch_radius=1,
        snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
        snerv_mfu_scales=(1, 3),
        snerv_hfr_gain=0.25,
        snerv_temporal_context=0,
        snerv_temporal_mode="delta",
        snerv_capacity_source=snerv_capacity_source,
        official_modelsize_solution=official_modelsize_solution,
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
