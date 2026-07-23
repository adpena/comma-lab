"""Tiny-fixture tests for the DDM AT1x atlas materialization contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.scorer_analytic_atlas import (
    SourceHashStamp,
    derive_se_gate_closed_form,
    evaluate_se_gate_factor,
)
from tac.optimization.scorer_atlas_materialization import (
    OBSERVED_E2,
    SEG_BINDING,
    AtlasMaterializationError,
    TensorIndexRow,
    aggregate_contractions,
    build_atlas_manifest,
    build_calibration_blocker_receipt,
    build_calibration_receipt,
    build_environment_receipt,
    certify_tree,
    classify_gaze_v19_coverage,
    contraction_spectrum,
    parse_evaluate_report,
    require_exact_pair_coverage,
    require_locked_inventory,
    score_formula,
    storage_preflight,
    version_set_sha256,
    write_immutable_receipt,
)


def _stamp() -> SourceHashStamp:
    return SourceHashStamp(
        source_id="fixture",
        path="fixture.py",
        sha256="a" * 64,
        bytes=1,
        validity_horizon="exact input hash equality",
    )


def _inventory(*, drift: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "body": {
            "source_strata": {
                "B_imported_library_sources": {
                    "version_drift": drift or {},
                    "binding_gate": (
                        "BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED"
                        if drift
                        else "PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED"
                    ),
                }
            }
        }
    }


def _identity(token: str = "b") -> dict[str, object]:
    return {"path": "fixture", "bytes": 1, "sha256": token * 64}


def test_locked_inventory_refuses_any_version_drift() -> None:
    assert require_locked_inventory(_inventory())
    with pytest.raises(AtlasMaterializationError, match="version_drift"):
        require_locked_inventory(_inventory(drift={"torch": {"locked": "1", "observed": "2"}}))
    with pytest.raises(
        AtlasMaterializationError,
        match="PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED",
    ):
        require_locked_inventory(
            {
                "source_strata": {
                    "B_imported_library_sources": {
                        "version_drift": {},
                        "binding_gate": "BLOCKED",
                    }
                }
            }
        )


@pytest.mark.parametrize(
    ("activation", "expected_hidden"),
    [
        ("relu", np.array([[0.0, 0.3]])),
        (
            "silu",
            np.array([[-0.2 / (1.0 + np.exp(0.2)), 0.3 / (1.0 + np.exp(-0.3))]]),
        ),
    ],
)
def test_se_factor_preserves_real_activation(activation: str, expected_hidden: np.ndarray) -> None:
    w1 = np.eye(2)
    b1 = np.zeros(2)
    w2 = np.eye(2)
    b2 = np.zeros(2)
    factor = derive_se_gate_closed_form(
        layer_id=f"fixture.{activation}",
        reduce_weight=w1,
        reduce_bias=b1,
        expand_weight=w2,
        expand_bias=b2,
        source_hashes=(_stamp(),),
        activation=activation,  # type: ignore[arg-type]
    )
    assert factor.payload["hidden_activation"] == activation
    expected = 1.0 / (1.0 + np.exp(-expected_hidden))
    np.testing.assert_allclose(
        evaluate_se_gate_factor(factor, np.array([[-0.2, 0.3]])),
        expected,
    )


def test_tree_and_immutable_receipt_custody(tmp_path: Path) -> None:
    tree = tmp_path / "env"
    tree.mkdir()
    (tree / "a.txt").write_text("alpha", encoding="utf-8")
    nested = tree / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"\x00\x01")
    certificate = certify_tree(tree)
    expected_rows = [
        {
            "relative_path": "a.txt",
            "bytes": 5,
            "sha256": hashlib.sha256(b"alpha").hexdigest(),
        },
        {
            "relative_path": "nested/b.bin",
            "bytes": 2,
            "sha256": hashlib.sha256(b"\x00\x01").hexdigest(),
        },
    ]
    assert certificate["rows"] == expected_rows

    path = tmp_path / "stage.json"
    write_immutable_receipt(path, {"stage": "one", "sha256": "a" * 64})
    original = path.read_bytes()
    write_immutable_receipt(path, {"sha256": "a" * 64, "stage": "one"})
    assert path.read_bytes() == original
    with pytest.raises(AtlasMaterializationError, match="non-byte-identical"):
        write_immutable_receipt(path, {"stage": "two"})


def test_environment_receipt_records_exact_lock_contract(tmp_path: Path) -> None:
    root = tmp_path / "ssd"
    environment = root / "locked_env"
    python = environment / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (upstream / "uv.lock").write_text("version=1\n", encoding="utf-8")
    preflight = storage_preflight(environment, required_free_bytes=1)
    tree = certify_tree(environment)
    receipt = build_environment_receipt(
        environment=environment,
        upstream_root=upstream,
        python_path=python,
        uv_version="uv 1.0",
        package_versions={"numpy": "2.3.4", "torch": "2.10.0"},
        preflight=preflight,
        tree_certificate=tree,
        ssd_root=root,
    )
    assert receipt["sync"]["argv"] == [
        "uv",
        "sync",
        "--frozen",
        "--group",
        "cpu",
        "--python",
        "3.11",
    ]
    assert receipt["sync"]["environment"]["UV_LINK_MODE"] == "copy"
    assert receipt["environment_disposition"]["automatic_delete"] is False
    assert receipt["tree_certificate"]["tree_sha256"] == tree["tree_sha256"]


def test_exact_n600_coverage_rejects_duplicate_and_gap() -> None:
    require_exact_pair_coverage(range(600))
    with pytest.raises(AtlasMaterializationError, match="600 unique"):
        require_exact_pair_coverage([*range(599), 598])
    with pytest.raises(AtlasMaterializationError, match=r"exact 0\.\.599"):
        require_exact_pair_coverage(range(1, 601))


def test_pose_spectra_are_six_row_float64_gram_at_both_depths() -> None:
    pose_y = np.arange(1, 25, dtype=np.float32).reshape(6, 2, 2)
    pose_x = (pose_y * 2).astype(np.float32)
    row = contraction_spectrum(
        pair_id=3,
        pose_scorer_plane=pose_y,
        pose_camera_input=pose_x,
        seg_scorer_plane=np.ones((2, 2, 3), dtype=np.float32),
        seg_camera_input=np.full((3, 3, 3), 2, dtype=np.float32),
        head_pair_norms=np.array([[1, 2], [3, 4]], dtype=np.float32),
        seg_binding=SEG_BINDING,
    )
    expected = np.linalg.eigvalsh(pose_y.reshape(6, -1).astype(np.float64) @ pose_y.reshape(6, -1).astype(np.float64).T)
    np.testing.assert_allclose(row["pose"]["scorer_plane_y"]["eigenvalues_ascending"], expected)
    assert row["pose"]["camera_input_x"]["row_count"] == 6
    with pytest.raises(AtlasMaterializationError, match="every Pose row"):
        contraction_spectrum(
            pair_id=3,
            pose_scorer_plane=np.zeros((6, 2), dtype=np.float32),
            pose_camera_input=pose_x,
            seg_scorer_plane=np.ones(2, dtype=np.float32),
            seg_camera_input=np.ones(2, dtype=np.float32),
            head_pair_norms=np.ones(2, dtype=np.float32),
            seg_binding=SEG_BINDING,
        )


def test_seg_contraction_requires_rank4_binding() -> None:
    common = {
        "pair_id": 0,
        "pose_scorer_plane": np.eye(6, dtype=np.float32),
        "pose_camera_input": np.eye(6, dtype=np.float32),
        "seg_scorer_plane": np.array([3.0, 4.0], dtype=np.float32),
        "seg_camera_input": np.array([1.0, 2.0], dtype=np.float32),
        "head_pair_norms": np.array([1.0, 2.0], dtype=np.float32),
    }
    row = contraction_spectrum(**common, seg_binding=SEG_BINDING)
    assert row["seg"]["binding"] == SEG_BINDING
    assert row["seg"]["head_pullback_rank"] == 4
    assert row["seg"]["scorer_plane_y"]["contracted_singular_energy"] == 25.0
    with pytest.raises(AtlasMaterializationError, match=SEG_BINDING):
        contraction_spectrum(**common, seg_binding="invented")


def test_gaze_lambda_classification_is_exactly_8_plus_592() -> None:
    coverage = classify_gaze_v19_coverage(exact_v19_pair_ids=(1, 3, 5, 7, 9, 11, 13, 15))
    assert coverage["v19_exact_join_count"] == 8
    assert coverage["gaze_measured_v19_join_owed_counted_inert"] == 592
    counts: dict[str, int] = {}
    for row in coverage["rows"]:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    assert counts == {
        "V19_EXACT_JOIN_AVAILABLE": 8,
        "GAZE_MEASURED_V19_JOIN_OWED_COUNTED_INERT": 592,
    }
    with pytest.raises(AtlasMaterializationError, match="exactly 8"):
        classify_gaze_v19_coverage(exact_v19_pair_ids=range(7))


def test_tensor_index_requires_version_stamp() -> None:
    common = {
        "pair_id": 0,
        "tensor_name": "pose_j_y",
        "path": "fixture.npz",
        "archive_sha256": "a" * 64,
        "tensor_sha256": "b" * 64,
        "shape": (6, 2),
        "dtype": "float32",
        "version_set_sha256": version_set_sha256({"torch": "2.10.0"}),
    }
    with pytest.raises(AtlasMaterializationError, match="version stamp"):
        TensorIndexRow(**common, version_stamp_id="")
    row = TensorIndexRow(**common, version_stamp_id="locked-macos-cpu-v1")
    assert row.to_dict()["version_stamp_id"] == "locked-macos-cpu-v1"


def test_aggregate_requires_all_600_rows() -> None:
    template = contraction_spectrum(
        pair_id=0,
        pose_scorer_plane=np.eye(6, dtype=np.float32),
        pose_camera_input=np.eye(6, dtype=np.float32),
        seg_scorer_plane=np.ones(2, dtype=np.float32),
        seg_camera_input=np.ones(3, dtype=np.float32),
        head_pair_norms=np.ones(2, dtype=np.float32),
        seg_binding=SEG_BINDING,
    )
    rows = []
    for pair_id in range(600):
        row = json.loads(json.dumps(template))
        row["pair_id"] = pair_id
        rows.append(row)
    aggregate = aggregate_contractions(rows)
    assert aggregate["pair_count"] == 600
    with pytest.raises(AtlasMaterializationError, match="exactly 600"):
        aggregate_contractions(rows[:-1])


def test_calibration_parser_and_signed_arithmetic() -> None:
    parsed = parse_evaluate_report(
        "\n".join(
            [
                "archive_bytes: 343466",
                "d_seg: 0.02861482",
                "d_pose: 162.58094788",
                "final_score: 43.411509751432",
            ]
        )
    )
    assert parsed["archive_bytes"] == 343466
    assert score_formula(
        archive_bytes=343466,
        d_seg=0.02861482,
        d_pose=162.58094788,
    ) == pytest.approx(OBSERVED_E2["total"])
    receipt = build_calibration_receipt(
        parsed=parsed,
        argv=("bash", "evaluate.sh", "archive.zip"),
        environment={"UV_LINK_MODE": "copy"},
        archive=_identity("a"),
        runtime=_identity("b"),
        upstream=_identity("c"),
        stdout=_identity("d"),
        stderr=_identity("e"),
        report=_identity("f"),
        wallclock_seconds=12.5,
    )
    drift = receipt["signed_drift_locked_minus_observed"]
    assert drift["d_seg"] == pytest.approx(0.0)
    assert drift["d_pose"] == pytest.approx(0.0)
    assert receipt["frozen_scorer_realization"]["signed_delta_to_upstream"] == pytest.approx(0.001144523776)


def test_official_report_parser_accepts_rounded_score_and_comma_bytes() -> None:
    report = """
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 162.58094788
  Average SegNet Distortion: 0.02861482
  Submission file size: 343,466 bytes
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 43.41
"""
    parsed = parse_evaluate_report(report)
    assert parsed == {
        "archive_bytes": 343466,
        "d_seg": 0.02861482,
        "d_pose": 162.58094788,
        "total": 43.41,
    }
    receipt = build_calibration_receipt(
        parsed=report,
        argv=("bash", "evaluate.sh", "--device", "cpu"),
        environment={"UV_LINK_MODE": "copy"},
        archive=_identity("a"),
        runtime=_identity("b"),
        upstream=_identity("c"),
        stdout=_identity("d"),
        stderr=_identity("e"),
        report=_identity("f"),
        wallclock_seconds=12.5,
    )
    assert receipt["measured"]["reported_total"] == 43.41
    assert receipt["measured"]["formula_total"] == pytest.approx(OBSERVED_E2["total"])


def test_locked_calibration_blocker_prices_only_observed_realization_gap() -> None:
    receipt = build_calibration_blocker_receipt(
        stderr_text="ModuleNotFoundError: No module named 'brotli'\n",
        exit_code=1,
        argv=("bash", "evaluate.sh", "--device", "cpu"),
        environment={"PYTHON": "/locked/bin/python"},
        archive=_identity("a"),
        runtime=_identity("b"),
        upstream=_identity("c"),
        stdout=_identity("d"),
        stderr=_identity("e"),
        wallclock_seconds=7.0,
    )
    assert receipt["status"] == "BLOCKED_LOCKED_RUNTIME_DEPENDENCY"
    assert receipt["blocker"]["module"] == "brotli"
    assert receipt["calibration"]["signed_drift_locked_minus_observed"] is None
    assert receipt["frozen_scorer_realization"]["signed_observed_minus_frozen"] == pytest.approx(0.001144523776)


def test_manifest_requires_nonzero_closed_form_gaze_and_jacobian_counts() -> None:
    contraction = {
        "pair_count": 600,
        "pair_rows": [{"pair_id": value} for value in range(600)],
    }
    manifest = build_atlas_manifest(
        environment_receipt={"schema": "env"},
        factor_index={
            "factor_count": 2,
            "factors": [
                {
                    "factor_id": "bn",
                    "network": "segnet",
                    "layer_id": "block.bn",
                },
                {
                    "factor_id": "bn_silu",
                    "network": "segnet",
                    "layer_id": "block.bn",
                },
            ],
        },
        contraction_atlas=contraction,
        calibration_receipt=None,
        reconstruction_commands=("rebuild",),
    )
    assert manifest["counts"] == {
        "closed_forms": 2,
        "gaze_pairs": 600,
        "jacobian_contraction_rows": 600,
    }
    assert manifest["amplitude_factors"]["count"] == 0
    assert manifest["frequency_dead_band"]["admission"] == "REFUSE_ZERO_BYTE_TRUNCATION"
    assert manifest["nonadditive_pools"]["pool_count"] == 1
    assert manifest["nonadditive_pools"]["rows"][0]["factor_ids"] == [
        "bn",
        "bn_silu",
    ]
    assert manifest["pointer"] == "UNCHANGED"
    assert manifest["main_landing_review_required"] is True
    with pytest.raises(AtlasMaterializationError, match="nonzero"):
        build_atlas_manifest(
            environment_receipt={},
            factor_index={"factor_count": 0},
            contraction_atlas=contraction,
            calibration_receipt=None,
            reconstruction_commands=(),
        )
