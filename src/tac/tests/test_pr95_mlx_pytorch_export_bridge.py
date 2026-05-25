from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.local_acceleration.pr95_hnerv_mlx import (
    HNeRVSyntheticTrainingBundleMLX,
    pytorch_state_dict_from_mlx,
    write_pr95_public_archive_zip,
)
from tools.export_pr95_mlx_to_pytorch_state_dict import (
    DEFAULT_SOURCE_MODEL,
    PR95_MLX_EXPORT_BRIDGE_SCHEMA,
    export_pr95_public_archive_to_pytorch_state_dict,
)
from tools.package_pr95_mlx_pytorch_state_dict_to_contest_archive import (
    PR95_MLX_PACKAGE_SCHEMA,
    package_pytorch_state_dict_to_contest_archive,
)


def test_pr95_mlx_pytorch_export_bridge_smoke_false_authority(
    tmp_path: Path,
) -> None:
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=8,
        seed=17,
    )
    archive_zip = tmp_path / "archive.zip"
    write_pr95_public_archive_zip(
        pytorch_state_dict_from_mlx(bundle.decoder),
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=archive_zip,
    )
    output_pt = tmp_path / "state.pt"
    report_path = tmp_path / "report.json"
    decoder_trace_path = tmp_path / "decoder_trace.json"

    report = export_pr95_public_archive_to_pytorch_state_dict(
        archive_zip=archive_zip,
        output_pytorch_state_dict=output_pt,
        source_model=DEFAULT_SOURCE_MODEL,
        report_out=report_path,
        decoder_trace_out=decoder_trace_path,
        sample_indices=[0],
    )

    assert report["schema_version"] == PR95_MLX_EXPORT_BRIDGE_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["archive_packet"]["member_name"] == "0.bin"
    assert report["pytorch_export_forward_parity_established"] is True
    assert output_pt.is_file()
    assert report_path.is_file()
    assert decoder_trace_path.is_file()
    assert report["pt_path"] == str(output_pt)
    assert report["decoder_trace"]["path"] == str(decoder_trace_path)
    assert report["decoder_trace"]["output_delta"]["max_abs_delta"] <= 1e-4
    drift_attestation = report["forward_drift_attestation"]
    assert drift_attestation["schema"] == (
        "pr95_mlx_pytorch_export_forward_drift_attestation.v1"
    )
    assert drift_attestation["attested_within_band"] is True
    assert drift_attestation["actual_class"] in {
        "byte_stable_by_default",
        "numeric_tolerance_inherent",
    }
    assert drift_attestation["score_claim"] is False
    decoder_trace = json.loads(decoder_trace_path.read_text(encoding="utf-8"))
    assert decoder_trace["score_claim"] is False
    assert decoder_trace["ready_for_exact_eval_dispatch"] is False
    parity = report["pytorch_export_forward_parity"]["forward_parity"]["parity"]
    assert parity["max_abs"] <= 1e-4


def test_pr95_mlx_pytorch_package_round_trips_archive_false_authority(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=8,
        seed=23,
    )
    state_dict = pytorch_state_dict_from_mlx(bundle.decoder)
    source_archive_zip = tmp_path / "source_archive.zip"
    write_report = write_pr95_public_archive_zip(
        state_dict,
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=source_archive_zip,
    )
    input_pt = tmp_path / "decoder_state.pt"
    torch.save(pytorch_state_dict_from_mlx(bundle.decoder, as_torch=True), input_pt)

    package_report = package_pytorch_state_dict_to_contest_archive(
        input_pt=input_pt,
        source_archive_zip=source_archive_zip,
        output_submission_dir=tmp_path / "submission_dir",
        report_out=tmp_path / "package_report.json",
    )

    assert package_report["schema_version"] == PR95_MLX_PACKAGE_SCHEMA
    assert package_report["archive_zip_sha256"] == write_report["archive_zip_sha256"]
    assert package_report["archive_zip_bytes"] == write_report["archive_zip_bytes"]
    assert package_report["archive_member_sha256"] == write_report["member_sha256"]
    assert package_report["decoder_state_dict_tensor_count"] == len(state_dict)
    assert package_report["runtime_files_emitted"]["inflate.sh"]
    assert package_report["runtime_files_emitted"]["inflate.py"]
    assert package_report["runtime_files_emitted"]["vendored_src_model.py"]
    assert package_report["runtime_files_emitted"]["vendored_src_codec.py"]
    assert package_report["score_claim"] is False
    assert package_report["promotion_eligible"] is False
    assert package_report["rank_or_kill_eligible"] is False
    assert package_report["ready_for_exact_eval_dispatch"] is False
    assert package_report["exact_readiness_refusal"]["ready"] is False
    assert "requires_full_frame_inflate_parity_before_runtime_consumption_claim" in (
        package_report["exact_readiness_refusal"]["blockers"]
    )
