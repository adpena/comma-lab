from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.boundary_math.integer_plane_emitter_byte_close import (
    C2ByteCloseError,
    build_counted_archive,
)
from tac.boundary_math.shared_receiver_admission import (
    BLOCKER_ID,
    MAX_ARCHIVE_BYTES,
    SharedReceiverAdmissionError,
    evaluate_shared_receiver_admission,
)

ROOT = Path(__file__).resolve().parents[4]


def _json(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text())
    assert isinstance(value, dict)
    return value


def _inputs() -> tuple[dict, dict, dict]:
    return (
        _json(".omx/research/pdw2_spatial_receiver_576_blocker_receipt_20260719.json"),
        _json(".omx/research/pdw1_fp32_realization_receipt_20260719.json"),
        _json(".omx/research/shared_receiver_r1_step2_prior_measurement_20260720.json"),
    )


def test_real_receipts_fail_closed_with_exact_measured_terms() -> None:
    pdw2, pdw1, step2 = _inputs()
    result = evaluate_shared_receiver_admission(
        pdw2_receipt=pdw2,
        pdw1_receipt=pdw1,
        step2_summary=step2,
    )

    assert result["success"] is False
    assert result["verdict"] == BLOCKER_ID
    assert result["through_r_authority"] is False
    assert result["score_claim"] is False
    assert result["pdw2"]["packet_brotli_bytes"] == 133
    assert result["pdw2"]["quotient_file_bytes"] == 1_887_436_928
    assert result["pdw1"]["d_a"] == 0.0
    assert result["pdw1"]["d_b"] == pytest.approx(0.00806956821017795)
    assert result["pdw1"]["mismatch_pixels"] == 38_077
    assert result["pdw1"]["confusion_rows_lstar_to_pred"][0] == {
        "lstar": "Road",
        "pred": "Lane",
        "px": 10_701,
        "share": 0.281,
    }
    assert result["pdw1"]["mismatch_within_chebyshev_r_of_lstar_boundary"]["1"]["px"] == 29_512
    assert result["pdw1"]["payload_n24_components"] == {
        "header_bytes": 17,
        "label_stream_bytes": 19_386,
        "fills_bytes": 360,
        "other_container_bytes": 96,
    }
    assert result["pdw1"]["projected_n600_bytes"] == 496_067
    assert result["step2"]["selected_ranked_cells"] == 0
    assert result["step2"]["first_prefix_net_fixed_pixels"] == 3
    assert result["step2"]["first_prefix_projected_bytes"] == 3_374
    assert result["step2"]["clears_rate_waterline"] is False
    assert sum(result["pdw2"]["partition_label_counts_by_class_0_to_4"]) == result["pdw2"]["partition_pixel_count"]
    assert result["dominant_measured_term"] == "pdw1_partition_carrier_and_realization"


def test_dense_section_is_hash_bound_and_priced_separately() -> None:
    pdw2, pdw1, step2 = _inputs()
    dense = {
        "schema": "dense_quotient_field_zip_measurement.v1",
        "source_bytes": pdw2["quotient_field"]["file_bytes"],
        "source_sha256": pdw2["quotient_field"]["file_sha256"],
        "member_uncompressed_bytes": pdw2["quotient_field"]["file_bytes"],
        "member_compressed_bytes": 899_999_832,
        "zip_bytes": 900_000_000,
        "zip_sha256": "a" * 64,
        "compression": "zip_deflate9",
    }
    result = evaluate_shared_receiver_admission(
        pdw2_receipt=pdw2,
        pdw1_receipt=pdw1,
        step2_summary=step2,
        dense_section_receipt=dense,
    )
    row = result["dense_spatial_section"]
    assert row["in_box"] is False
    assert row["over_archive_gate_bytes"] == 900_000_000 - MAX_ARCHIVE_BYTES
    assert row["zip_container_overhead_bytes"] == 168
    assert result["dominant_measured_term"] == "dense_float32_spatial_field_section"

    dense["source_sha256"] = "b" * 64
    with pytest.raises(SharedReceiverAdmissionError, match="source hash"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=pdw2,
            pdw1_receipt=pdw1,
            step2_summary=step2,
            dense_section_receipt=dense,
        )


def test_self_authored_candidate_json_cannot_confer_authority() -> None:
    pdw2, pdw1, step2 = _inputs()
    candidate = {
        "n_pairs": 600,
        "archive_bytes": 250_000,
        "archive_sha256": "c" * 64,
        "d_seg": 0.0003,
        "d_pose": 0.0001,
        "authority_axis": "[macOS-CPU advisory]",
        "exact_archive": True,
        "archive_parseback_identical": True,
        "production_receiver": True,
        "through_r_authority": True,
        "hard_cpu_torch_oracle": True,
        "packet_mutation_changes_decoded": True,
        "scorer_free_spatial_rgb_pullback": True,
        "pdw2_packet_sha256": pdw2["packet"]["raw_sha256"],
        "spatial_generator_payload_sha256": "d" * 64,
        "hard_oracle_receipt_sha256": "e" * 64,
    }
    custody = {"all_claims": True, "authority_axis": "[macOS-CPU advisory]"}
    with pytest.raises(SharedReceiverAdmissionError, match="not admission authority"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=pdw2,
            pdw1_receipt=pdw1,
            step2_summary=step2,
            exact_candidate=candidate,
            exact_candidate_custody=custody,
        )


def test_cross_receipt_counts_and_dense_bytes_fail_closed() -> None:
    pdw2, pdw1, step2 = _inputs()
    bad_step2 = copy.deepcopy(step2)
    bad_step2["n48"]["residual_reason_counts"]["STOPPED_AT_MEASURED_RATE_WATERLINE"] -= 1
    with pytest.raises(SharedReceiverAdmissionError, match="must sum"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=pdw2,
            pdw1_receipt=pdw1,
            step2_summary=bad_step2,
        )

    dense = {
        "schema": "dense_quotient_field_zip_measurement.v1",
        "source_bytes": pdw2["quotient_field"]["file_bytes"] - 1,
        "source_sha256": pdw2["quotient_field"]["file_sha256"],
        "member_uncompressed_bytes": pdw2["quotient_field"]["file_bytes"] - 1,
        "member_compressed_bytes": 899_999_832,
        "zip_bytes": 900_000_000,
        "zip_sha256": "a" * 64,
        "compression": "zip_deflate9",
    }
    with pytest.raises(SharedReceiverAdmissionError, match="source bytes"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=pdw2,
            pdw1_receipt=pdw1,
            step2_summary=step2,
            dense_section_receipt=dense,
        )


def test_production_archive_refusal_names_shared_blocker(tmp_path: Path) -> None:
    with pytest.raises(C2ByteCloseError, match=BLOCKER_ID):
        build_counted_archive(
            base_archive=tmp_path / "missing_base.zip",
            checkpoint_path=tmp_path / "missing_checkpoint.npz",
            output=tmp_path / "must_not_exist.zip",
            pdw2_packet=b"packet",
            pdw2_role="receiver_consumed",
        )
    assert not (tmp_path / "must_not_exist.zip").exists()


def test_target_only_receipt_cannot_launder_distortion_authority() -> None:
    pdw2, pdw1, step2 = _inputs()
    corrupted = copy.deepcopy(pdw2)
    corrupted["authority"]["through_r_authority"] = True
    with pytest.raises(SharedReceiverAdmissionError, match="cannot carry through-R"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=corrupted,
            pdw1_receipt=pdw1,
            step2_summary=step2,
        )

    corrupted = copy.deepcopy(pdw2)
    corrupted["authority"]["d_seg"] = 0.0
    with pytest.raises(SharedReceiverAdmissionError, match="must be null"):
        evaluate_shared_receiver_admission(
            pdw2_receipt=corrupted,
            pdw1_receipt=pdw1,
            step2_summary=step2,
        )
