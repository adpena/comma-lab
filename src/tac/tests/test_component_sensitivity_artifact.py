"""Tests for component sensitivity artifact validation."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tac.component_sensitivity_artifact import (
    ComponentSensitivityArtifactError,
    dumps_component_sensitivity_manifest,
    has_a2_certified_sensitivity_binding_reference,
    materialize_component_sensitivity_manifest,
    sha256_file,
    validate_a2_certified_sensitivity_binding,
    validate_component_sensitivity_manifest,
    write_component_sensitivity_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _pair_record(pair_index: int) -> dict[str, int]:
    return {
        "video": 0,
        "pair_index": pair_index,
        "t": 2 * pair_index,
        "t1": 2 * pair_index + 1,
    }


def _sample_plan() -> dict[str, object]:
    return {
        "calibration_pairs": [_pair_record(idx) for idx in range(480)],
        "holdout_pairs": [_pair_record(idx) for idx in range(480, 600)],
        "split_seed": 1234,
        "split_hash": SHA_D,
    }


def _file_meta(sha256: str = SHA_A) -> dict[str, object]:
    return {"path": "artifact.bin", "bytes": 123, "sha256": sha256}


def _component_map(component: str, sha256: str) -> dict[str, object]:
    return {
        "path": "sensitivity.pt",
        "bytes": 456,
        "sha256": sha256,
        "scorer_target": component,
        "map_format": "tac_score_sensitivity_map_v1",
        "certification": _component_map_certification(component),
        "tensor": {"dtype": "float32", "shape": [8, 16], "numel": 128},
    }


def _component_map_certification(component: str) -> dict[str, object]:
    return {
        "format": "component_sensitivity_map_certification_v1",
        "component": component,
        "device": "cuda",
        "official_component_response": True,
        "canonical_scorer_path": True,
        "promotion_eligible": True,
        "source_map_sha256": SHA_A,
        "official_response_curve_sha256": SHA_B,
        "stability_sha256": SHA_C,
        "sample_plan_sha256": SHA_D,
        "baseline_archive_sha256": SHA_A,
        "baseline_archive_bytes": 686635,
        "contest_auth_eval_json_sha256": SHA_B,
        "prediction_deltas_sha256": SHA_C,
        "perturbation_basis_sha256": SHA_D,
        "review_packet_sha256": SHA_A,
        "review_clean_passes": 3,
        "review_unresolved_blockers": [],
        "response_gate_results": {
            "finite_values": True,
            "coverage_passed": True,
            "zero_repro": True,
            "zero_repro_error": 0.0,
            "signal_present": True,
            "observed_delta_max": 0.01,
            "prediction_error_passed": True,
            "max_relative_prediction_error": 0.02,
            "promotion_gate_passed": True,
        },
        "stability_gate_results": {
            "passed": True,
            "cv_max": 0.04,
            "spearman_min": 0.96,
            "top_decile_overlap_min": 0.91,
        },
    }


def _response_curve(component: str, sha256: str) -> dict[str, object]:
    readouts = {
        "posenet": "official_pose_mse",
        "segnet": "official_argmax_disagreement",
        "combined": "official_component_formula",
    }
    return {
        "path": "curves.json",
        "bytes": 789,
        "sha256": sha256,
        "count": 7,
        "holdout_error": 0.02,
        "official_component_response": True,
        "passed": True,
        "gate_results": {
            "finite_values": True,
            "coverage_passed": True,
            "zero_repro": True,
            "zero_repro_error": 0.0,
            "signal_present": True,
            "observed_delta_max": 0.01,
            "prediction_error_passed": True,
            "max_relative_prediction_error": 0.02,
            "promotion_gate_passed": True,
        },
        "gate_spec": {
            "zero_repro_tolerance": 1e-7,
            "holdout_error_max": 0.05,
            "spearman_min": 0.3,
        },
        "promotion_blockers": [],
        "component_readout": readouts[component],
        "response_kind": "symmetric",
        "epsilon_ladder": [-0.002, -0.001, 0.0, 0.001, 0.002],
    }


def _response_curve_without_custody(component: str) -> dict[str, object]:
    entry = _response_curve(component, SHA_A)
    entry.pop("bytes")
    entry.pop("sha256")
    return entry


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": "cuda",
        "promotion_eligible": True,
        "evidence_grade": "A",
        "inputs": {
            "checkpoint": _file_meta(SHA_A),
            "video": _file_meta(SHA_B),
            "upstream": _file_meta(SHA_C),
        },
        "sample_plan": _sample_plan(),
        "component_maps": {
            "posenet": _component_map("posenet", SHA_A),
            "segnet": _component_map("segnet", SHA_B),
            "combined": _component_map("combined", SHA_C),
        },
        "stability": {
            "cv": {"posenet": 0.04, "segnet": 0.05, "combined": 0.03},
            "rank": {"posenet": 0.98, "segnet": 0.97, "combined": 0.96},
            "top_k": {
                "posenet": {"k": 16, "overlap": 0.91},
                "segnet": {"k": 16, "overlap": 0.89},
                "combined": {"k": 16, "overlap": 0.93},
            },
            "thresholds": {
                "cv_max": 0.35,
                "spearman_min": 0.3,
                "top_decile_overlap_min": 0.5,
            },
            "passed": True,
        },
        "response_curves": {
            "posenet": _response_curve("posenet", SHA_A),
            "segnet": _response_curve("segnet", SHA_B),
            "combined": _response_curve("combined", SHA_C),
        },
        "contest_eval": {
            "archive_bytes": 37_000_000,
            "archive_sha256": SHA_A,
            "contest_auth_eval_json": {
                "path": "contest_auth_eval.json",
                "bytes": 321,
                "sha256": SHA_B,
            },
            "device": "cuda",
            "n_samples": 600,
        },
    }


def test_minimal_valid_manifest_passes() -> None:
    validate_component_sensitivity_manifest(_valid_manifest())


def test_a2_certified_binding_requires_combined_component_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "component_sensitivity_v1.json"
    write_component_sensitivity_manifest(manifest_path, _valid_manifest())
    a2_manifest = {
        "sensitivity_artifact": {
            "path": "combined_certified_sensitivity_map.pt",
            "status": "passed",
            "allow_diagnostic_sensitivity": False,
            "metadata_blockers": [],
            "component_sensitivity_manifest": {
                "path": "component_sensitivity_v1.json",
                "sha256": sha256_file(manifest_path),
                "component": "combined",
            },
        },
        "inputs": {
            "sensitivity_map_sha256": SHA_C,
        },
    }

    assert has_a2_certified_sensitivity_binding_reference(a2_manifest)
    proof = validate_a2_certified_sensitivity_binding(
        a2_manifest,
        manifest_root=tmp_path,
    )

    assert proof["status"] == "passed"
    assert proof["source"] == "component_sensitivity_v1.combined"
    assert proof["component_map"]["sha256"] == SHA_C
    assert proof["a2_sensitivity_map_sha256"] == SHA_C


def test_a2_certified_binding_rejects_stub_and_map_sha_mismatch(tmp_path) -> None:
    manifest_path = tmp_path / "component_sensitivity_v1.json"
    write_component_sensitivity_manifest(manifest_path, _valid_manifest())
    a2_manifest = {
        "sensitivity_artifact": {
            "path": "combined_certified_sensitivity_map.pt",
            "status": "passed",
            "allow_diagnostic_sensitivity": False,
            "metadata_blockers": [],
            "component_sensitivity_manifest": {
                "path": "component_sensitivity_v1.json",
                "sha256": sha256_file(manifest_path),
                "component": "combined",
            },
        },
        "inputs": {
            "sensitivity_map_sha256": SHA_C,
        },
    }

    stub = deepcopy(a2_manifest)
    stub["sensitivity_artifact"]["metadata_blockers"] = ["is_stub=true"]  # type: ignore[index]
    with pytest.raises(ComponentSensitivityArtifactError, match="metadata_blockers"):
        validate_a2_certified_sensitivity_binding(stub, manifest_root=tmp_path)

    stale = deepcopy(a2_manifest)
    stale["inputs"]["sensitivity_map_sha256"] = SHA_A  # type: ignore[index]
    with pytest.raises(ComponentSensitivityArtifactError, match="does not match certified combined map"):
        validate_a2_certified_sensitivity_binding(stale, manifest_root=tmp_path)


def test_promotion_sample_plan_requires_full_absolute_pair_ids() -> None:
    missing = _valid_manifest()
    missing["sample_plan"]["holdout_pairs"].pop()  # type: ignore[index,union-attr]
    with pytest.raises(ComponentSensitivityArtifactError, match="exactly 600"):
        validate_component_sensitivity_manifest(missing)

    duplicate = _valid_manifest()
    duplicate["sample_plan"]["holdout_pairs"][0]["pair_index"] = 0  # type: ignore[index,union-attr]
    duplicate["sample_plan"]["holdout_pairs"][0]["t"] = 0  # type: ignore[index,union-attr]
    duplicate["sample_plan"]["holdout_pairs"][0]["t1"] = 1  # type: ignore[index,union-attr]
    with pytest.raises(ComponentSensitivityArtifactError, match="unique"):
        validate_component_sensitivity_manifest(duplicate)

    mismatch = _valid_manifest()
    mismatch["sample_plan"]["calibration_pairs"][0]["t1"] = 99  # type: ignore[index,union-attr]
    with pytest.raises(ComponentSensitivityArtifactError, match="t1 must equal"):
        validate_component_sensitivity_manifest(mismatch)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cpu_and_mps_rejected_for_promotion(device: str) -> None:
    manifest = _valid_manifest()
    manifest["device"] = device

    with pytest.raises(ComponentSensitivityArtifactError, match="CUDA"):
        validate_component_sensitivity_manifest(manifest)


def test_missing_component_map_rejected_for_promotion() -> None:
    manifest = _valid_manifest()
    del manifest["component_maps"]["segnet"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="component_maps.segnet"):
        validate_component_sensitivity_manifest(manifest)


def test_missing_component_map_bytes_rejected_for_promotion() -> None:
    manifest = _valid_manifest()
    del manifest["component_maps"]["segnet"]["bytes"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="component_maps.segnet.bytes"):
        validate_component_sensitivity_manifest(manifest)


def test_component_map_scorer_target_must_match_component() -> None:
    manifest = _valid_manifest()
    manifest["component_maps"]["combined"]["scorer_target"] = "posenet"  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="scorer_target"):
        validate_component_sensitivity_manifest(manifest)


def test_component_map_requires_certification_for_promotion() -> None:
    manifest = _valid_manifest()
    del manifest["component_maps"]["combined"]["certification"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="certification"):
        validate_component_sensitivity_manifest(manifest)


def test_component_map_rejects_insufficient_review_passes() -> None:
    manifest = _valid_manifest()
    manifest["component_maps"]["combined"]["certification"]["review_clean_passes"] = 2  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="review_clean_passes"):
        validate_component_sensitivity_manifest(manifest)


def test_component_map_requires_explicit_empty_review_blockers() -> None:
    manifest = _valid_manifest()
    del manifest["component_maps"]["combined"]["certification"]["review_unresolved_blockers"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="review_unresolved_blockers"):
        validate_component_sensitivity_manifest(manifest)

    for bad_blockers in (None, ["needs review"], "none"):
        manifest = _valid_manifest()
        manifest["component_maps"]["combined"]["certification"][  # type: ignore[index]
            "review_unresolved_blockers"
        ] = bad_blockers

        with pytest.raises(ComponentSensitivityArtifactError, match="review_unresolved_blockers"):
            validate_component_sensitivity_manifest(manifest)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("debug", True),
        ("smoke", True),
        ("fake_sensitivity", True),
        ("random_sensitivity", True),
        ("proxy_only", True),
    ],
)
def test_fake_debug_random_and_proxy_flags_rejected_for_promotion(key: str, value: bool) -> None:
    manifest = _valid_manifest()
    manifest[key] = value

    with pytest.raises(ComponentSensitivityArtifactError, match="fail-closed"):
        validate_component_sensitivity_manifest(manifest)


@pytest.mark.parametrize(
    "marker",
    [
        "diagnostic_cuda_fisher_proxy",
        "owv3_channel_sensitivity_from_fisher_v1",
        "experiments/profile_component_sensitivity.py",
        "uniform_sensitivity",
    ],
)
def test_diagnostic_proxy_and_uniform_source_strings_rejected_for_promotion(marker: str) -> None:
    manifest = _valid_manifest()
    manifest["component_maps"]["combined"]["source"] = marker  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="fail-closed"):
        validate_component_sensitivity_manifest(manifest)


def test_non_promotion_allows_diagnostic_cpu_debug_manifest() -> None:
    manifest = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": "cpu",
        "promotion_eligible": False,
        "evidence_grade": "invalid",
        "debug": True,
        "promotion_blockers": [
            {
                "code": "diagnostic_cpu_only",
                "mathematical_explanation": (
                    "CPU scorer paths do not preserve the CUDA PoseNet/SegNet "
                    "response, so this artifact cannot define a contest-grade "
                    "score-sensitivity allocation."
                ),
            }
        ],
    }

    validate_component_sensitivity_manifest(manifest, promotion=False)


def test_non_promotable_incomplete_manifest_requires_mathematical_blocker() -> None:
    manifest = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": "cpu",
        "promotion_eligible": False,
    }

    with pytest.raises(ComponentSensitivityArtifactError, match="mathematical_explanation"):
        validate_component_sensitivity_manifest(manifest, promotion=False)


def test_promotion_blockers_rejected_for_promotion() -> None:
    manifest = _valid_manifest()
    manifest["promotion_blockers"] = [
        {
            "code": "missing_combined",
            "mathematical_explanation": (
                "The combined channel is needed because the contest objective "
                "contains both 100*SegNet and sqrt(10*PoseNet) terms."
            ),
        }
    ]

    with pytest.raises(ComponentSensitivityArtifactError, match="promotion_blockers"):
        validate_component_sensitivity_manifest(manifest)


def test_nonfinite_stability_rejected() -> None:
    manifest = _valid_manifest()
    manifest["stability"]["cv"]["posenet"] = float("nan")  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="finite"):
        validate_component_sensitivity_manifest(manifest)


def test_nonfinite_response_curve_holdout_error_rejected() -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["combined"]["holdout_error"] = float("inf")  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="finite"):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_requires_official_component_response() -> None:
    manifest = _valid_manifest()
    del manifest["response_curves"]["posenet"]["official_component_response"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="official_component_response"):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_requires_passed_gate() -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["combined"]["passed"] = False  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="passed"):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_requires_gate_results_for_promotion() -> None:
    manifest = _valid_manifest()
    del manifest["response_curves"]["posenet"]["gate_results"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="gate_results"):
        validate_component_sensitivity_manifest(manifest)


@pytest.mark.parametrize(
    "gate_key",
    [
        "coverage_passed",
        "zero_repro",
        "signal_present",
        "prediction_error_passed",
        "promotion_gate_passed",
    ],
)
def test_response_curve_rejects_false_gate_results_for_promotion(gate_key: str) -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["combined"]["gate_results"][gate_key] = False  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match=gate_key):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_rejects_nonempty_promotion_blockers_for_promotion() -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["segnet"]["promotion_blockers"] = [  # type: ignore[index]
        {
            "code": "prediction_error_gate_failed",
            "mathematical_explanation": "The official response curve did not pass the prediction gate.",
        }
    ]

    with pytest.raises(ComponentSensitivityArtifactError, match="promotion_blockers"):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_requires_official_segnet_argmax_readout() -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["segnet"]["component_readout"] = "segnet_cross_entropy"  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="official segnet readout"):
        validate_component_sensitivity_manifest(manifest)


def test_response_curve_requires_symmetric_or_directional_coverage() -> None:
    manifest = _valid_manifest()
    manifest["response_curves"]["posenet"]["epsilon_ladder"] = [0.0, 0.001, 0.002]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="-eps/\\+eps"):
        validate_component_sensitivity_manifest(manifest)


def test_stability_requires_pass_thresholds() -> None:
    manifest = _valid_manifest()
    del manifest["stability"]["thresholds"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="thresholds"):
        validate_component_sensitivity_manifest(manifest)


def test_bad_sha_looking_value_rejected() -> None:
    manifest = _valid_manifest()
    manifest["inputs"]["checkpoint"]["sha256"] = "not-a-sha"  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="sha256"):
        validate_component_sensitivity_manifest(manifest)


def test_bad_contest_eval_json_sha_rejected_when_present() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["contest_eval"]["contest_auth_eval_json"]["sha256"] = "1234"  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="contest_auth_eval_json_sha256"):
        validate_component_sensitivity_manifest(manifest)


def test_contest_eval_json_bytes_required_for_promotion() -> None:
    manifest = deepcopy(_valid_manifest())
    del manifest["contest_eval"]["contest_auth_eval_json"]["bytes"]  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="contest_auth_eval_json"):
        validate_component_sensitivity_manifest(manifest)


def test_nested_non_cuda_source_device_rejected_for_promotion() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["component_maps"]["posenet"]["source_metadata"] = {"source_device": "cpu"}  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="source_device.*CUDA"):
        validate_component_sensitivity_manifest(manifest)


def test_empirical_evidence_grade_rejected_for_promotion() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["evidence_grade"] = "empirical"

    with pytest.raises(ComponentSensitivityArtifactError, match="evidence_grade"):
        validate_component_sensitivity_manifest(manifest)


def test_contest_eval_wrong_sample_count_rejected_for_promotion() -> None:
    manifest = deepcopy(_valid_manifest())
    manifest["contest_eval"]["n_samples"] = 599  # type: ignore[index]

    with pytest.raises(ComponentSensitivityArtifactError, match="n_samples"):
        validate_component_sensitivity_manifest(manifest)


def test_materialize_manifest_fills_custody_and_writes_deterministically(tmp_path) -> None:
    files = {
        "checkpoint.bin": b"checkpoint",
        "video.mkv": b"video",
        "upstream/evaluate.py": b"print('eval')\n",
        "posenet.pt": b"pose",
        "segnet.pt": b"seg",
        "combined.pt": b"combined",
        "posenet_curves.json": b"[]",
        "segnet_curves.json": b"[]",
        "combined_curves.json": b"[]",
        "archive.zip": b"zip",
        "contest_auth_eval.json": b"{}",
    }
    for rel, data in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    draft = deepcopy(_valid_manifest())
    draft["inputs"] = {
        "video": {"path": "video.mkv"},
        "upstream": {"path": "upstream"},
        "checkpoint": {"path": "checkpoint.bin"},
    }
    draft["component_maps"] = {
        "combined": {
            "path": "combined.pt",
            "scorer_target": "combined",
            "map_format": "tac_score_sensitivity_map_v1",
            "certification": _component_map_certification("combined"),
            "tensor": {"dtype": "float32", "shape": [1]},
        },
        "segnet": {
            "path": "segnet.pt",
            "scorer_target": "segnet",
            "map_format": "tac_score_sensitivity_map_v1",
            "certification": _component_map_certification("segnet"),
            "tensor": {"dtype": "float32", "shape": [1]},
        },
        "posenet": {
            "path": "posenet.pt",
            "scorer_target": "posenet",
            "map_format": "tac_score_sensitivity_map_v1",
            "certification": _component_map_certification("posenet"),
            "tensor": {"dtype": "float32", "shape": [1]},
        },
    }
    draft["response_curves"] = {
        "combined": {
            **_response_curve_without_custody("combined"),
            "path": "combined_curves.json",
            "count": 1,
            "holdout_error": 0.0,
        },
        "segnet": {
            **_response_curve_without_custody("segnet"),
            "path": "segnet_curves.json",
            "count": 1,
            "holdout_error": 0.0,
        },
        "posenet": {
            **_response_curve_without_custody("posenet"),
            "path": "posenet_curves.json",
            "count": 1,
            "holdout_error": 0.0,
        },
    }
    draft["contest_eval"] = {
        "archive": {"path": "archive.zip"},
        "contest_auth_eval_json": {"path": "contest_auth_eval.json"},
        "device": "cuda",
        "n_samples": 600,
    }

    manifest = materialize_component_sensitivity_manifest(draft, root=tmp_path)
    validate_component_sensitivity_manifest(manifest)

    assert list(manifest["inputs"]) == ["checkpoint", "video", "upstream"]
    assert list(manifest["component_maps"]) == ["posenet", "segnet", "combined"]
    assert manifest["inputs"]["upstream"]["kind"] == "directory"
    assert manifest["contest_eval"]["archive"]["bytes"] == len(files["archive.zip"])
    assert len(manifest["contest_eval"]["contest_auth_eval_json"]["sha256"]) == 64

    out_a = tmp_path / "manifest_a.json"
    out_b = tmp_path / "manifest_b.json"
    write_component_sensitivity_manifest(out_a, manifest)
    write_component_sensitivity_manifest(out_b, manifest)
    assert out_a.read_text() == out_b.read_text()
    assert out_a.read_text() == dumps_component_sensitivity_manifest(manifest)


def test_materialize_manifest_rejects_absolute_tmp_paths(tmp_path) -> None:
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"checkpoint")
    draft = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "inputs": {"checkpoint": {"path": str(artifact)}},
    }

    with pytest.raises(ComponentSensitivityArtifactError, match="transient"):
        materialize_component_sensitivity_manifest(draft, root=tmp_path, promotion=False)


def test_materialize_manifest_rejects_mismatched_custody(tmp_path) -> None:
    artifact = tmp_path / "posenet.pt"
    artifact.write_bytes(b"real-bytes")
    draft = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "component_maps": {
            "posenet": {
                "path": "posenet.pt",
                "bytes": 1,
                "sha256": SHA_A,
                "scorer_target": "posenet",
                "tensor": {"dtype": "float32", "shape": [1]},
            }
        },
        "promotion_blockers": [
            {
                "code": "partial_fixture",
                "mathematical_explanation": (
                    "Only PoseNet is present; SegNet and combined sensitivity "
                    "are required because the contest objective has separable "
                    "SegNet and PoseNet terms."
                ),
            }
        ],
    }

    with pytest.raises(ComponentSensitivityArtifactError, match="custody"):
        materialize_component_sensitivity_manifest(draft, root=tmp_path, promotion=False)
