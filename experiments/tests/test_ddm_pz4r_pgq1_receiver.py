from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PZ4_RUNTIME = REPO / "experiments/ddm_pz4r_runtime"
PZ3_RUNTIME = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
for path in (PZ4_RUNTIME, PZ3_RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

gauge_receiver = importlib.import_module("pose_gauge_receiver")
runner = importlib.import_module("experiments.ddm_pz4r_pgq1_receiver")

PGQ1 = Path("/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3/candidates/r6_b12_global/gauge.pgq1")
DECODED = PGQ1.with_name("decoded_outputs.float32.npy")
RESULT = Path("/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/PZ4R_RESULT.json")


def test_literal_pgq1_decode_matches_retained_winner() -> None:
    codes, decoded = gauge_receiver.decode_pgq1(PGQ1.read_bytes())
    assert codes.shape == (600, 6)
    assert codes.dtype == np.int32
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, np.load(DECODED))


def test_pgq1_strictly_rejects_geometry_and_trailing_bytes() -> None:
    payload = bytearray(PGQ1.read_bytes())
    payload[6] = 5  # rank field; the PZ4R branch is pinned to full rank six.
    with pytest.raises(ValueError, match="unsupported PGQ1 geometry"):
        gauge_receiver.decode_pgq1(payload)
    with pytest.raises(ValueError, match="payload length mismatch"):
        gauge_receiver.decode_pgq1(PGQ1.read_bytes() + b"\0")


def test_pose_object_features_consume_compensation_state() -> None:
    payload = bytearray(PGQ1.read_bytes())
    _, baseline = gauge_receiver.decode_pgq1(payload)
    # First compensation float starts after the 16-byte header and 4-byte scale.
    payload[20] ^= 1
    _, mutated = gauge_receiver.decode_pgq1(payload)
    baseline_features = gauge_receiver.quantize_pose_object(baseline, 16)
    mutated_features = gauge_receiver.quantize_pose_object(mutated, 16)
    assert not np.array_equal(baseline_features, mutated_features)


def test_coefficient_metric_uses_rendered_non_circular_geometry() -> None:
    reference = np.asarray([[-2048], [2047]], dtype=np.int32)
    predicted = np.asarray([[2047], [2047]], dtype=np.int32)
    metrics = runner.coefficient_metrics(predicted, reference)
    assert metrics["coefficient_code_mae"] == 2047.5
    assert metrics["coefficient_code_exact"] == 1
    assert metrics["endpoint_wrap_crossings"] == 1
    assert metrics["coefficient_error_geometry"] == ("ordinary_signed_difference_non_circular")


def test_receiver_rejects_corruption_and_selected_parseback_is_deterministic() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    carrier_path = Path(result["selected"]["records"]["carrier"]["path"])
    carrier = carrier_path.read_bytes()
    first = gauge_receiver.decode_pose_gauge_carrier(carrier)
    second = gauge_receiver.decode_pose_gauge_carrier(carrier)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    corrupted = bytearray(carrier)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError):
        gauge_receiver.decode_pose_gauge_carrier(corrupted)


def test_lc2_patch_has_real_pz4r_dispatch_and_early_return() -> None:
    source = (runner.LC2_RUNTIME_SOURCE / "inflate.py").read_text(encoding="utf-8")
    patched = runner.patch_inflate_source(source)
    assert "decode_pose_gauge_carrier(carrier_blob)" in patched
    assert "return semantic, basis, coeff" in patched
    compile(patched, "inflate.py", "exec")


def test_result_proves_repeat_identity_and_no_exact_residual() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    selected = result["selected"]
    assert selected["name"] == "target_quadratic_previous_f10_q20"
    assert selected["records"]["archive"] == {
        **selected["records"]["archive_repeat"],
        "path": selected["records"]["archive"]["path"],
    }
    assert selected["records"]["archive"]["sha256"] == selected["records"]["archive_repeat"]["sha256"]
    assert result["receiver_boundary"]["cpr1_packet_present"] is False
    assert result["receiver_boundary"]["exact_coefficient_residual_present"] is False
    assert result["pgq_consumption"]["rendered_frame_bytes_changed"] is True


def test_completed_resume_rehashes_receipts_and_source_bindings() -> None:
    state_path = RESULT.parent / "checkpoints/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    runner.validate_resume_state(state, RESULT.parent)

    stale_result = json.loads(json.dumps(state))
    stale_result["final_result_sha256"] = "0" * 64
    with pytest.raises(runner.PZ4RBuildError, match="final result"):
        runner.validate_resume_state(stale_result, RESULT.parent)

    stale_source = json.loads(json.dumps(state))
    stale_source["source_files"]["builder"]["sha256"] = "0" * 64
    with pytest.raises(runner.PZ4RBuildError, match="source binding drifted: builder"):
        runner.validate_resume_state(stale_source, RESULT.parent)

    truthy_non_boolean = json.loads(json.dumps(state))
    truthy_non_boolean["preflight_complete"] = 1
    with pytest.raises(runner.PZ4RBuildError, match="stage flag is not boolean"):
        runner.validate_resume_state(truthy_non_boolean, RESULT.parent)


def test_runtime_dependency_manifest_matches_actual_pz4r_tree() -> None:
    manifest = runner.validate_runtime_manifest(RESULT.parent / "submission")
    assert "denominator" not in manifest["closure_provenance"]
    assert manifest["closure_provenance"]["python_module_denominator"] == 8
    assert manifest["closure_provenance"]["entrypoint_denominator"] == 1
    assert set(manifest["source"]["copied_files"]) == set(runner.RUNTIME_SOURCE_FILES)
