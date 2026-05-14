# SPDX-License-Identifier: MIT
"""Sensitivity-map contract tests for OWV3."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.sensitivity_map import (
    CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT,
    SensitivityMapError,
    build_contiguous_pair_manifest,
    canonical_sensitivity_json_bytes,
    conv_weight_shapes,
    load_sensitivity_map,
    require_authoritative_device,
    resolve_layer_sensitivity,
    save_certified_sensitivity_map,
    save_sensitivity_map,
    sensitivity_cv_distance,
    sensitivity_manifest_sha256,
    validate_certified_sensitivity_map_metadata,
    validate_real_sensitivity_artifact,
    validate_sensitivity_map_for_model,
    validate_sensitivity_vector,
)


def _conv_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 5, 3, padding=1),
    )


def test_authoritative_device_rejects_non_cuda() -> None:
    require_authoritative_device("cuda:0")
    with pytest.raises(SensitivityMapError, match="record their device"):
        require_authoritative_device(None)
    with pytest.raises(SensitivityMapError, match="require CUDA"):
        require_authoritative_device("cpu")
    with pytest.raises(SensitivityMapError, match="require CUDA"):
        require_authoritative_device("mps")


def test_validate_sensitivity_vector_rejects_shape_nan_and_negative() -> None:
    valid = validate_sensitivity_vector(
        torch.tensor([0.0, 1.0, 2.0]),
        expected_channels=3,
        name="x.weight",
    )
    assert valid.dtype == torch.float32
    assert valid.device.type == "cpu"

    with pytest.raises(SensitivityMapError, match="does not match"):
        validate_sensitivity_vector(torch.ones(2), expected_channels=3, name="bad")
    with pytest.raises(SensitivityMapError, match="NaN/Inf"):
        validate_sensitivity_vector(
            torch.tensor([1.0, float("nan"), 2.0]),
            expected_channels=3,
            name="bad",
        )
    with pytest.raises(SensitivityMapError, match="non-negative"):
        validate_sensitivity_vector(
            torch.tensor([1.0, -1.0, 2.0]),
            expected_channels=3,
            name="bad",
        )


def test_validate_map_requires_every_conv_when_requested() -> None:
    model = _conv_model()
    shapes = conv_weight_shapes(model)
    assert shapes == {"0.weight": 4, "2.weight": 5}

    with pytest.raises(SensitivityMapError, match="missing"):
        validate_sensitivity_map_for_model(
            {"0.weight": torch.ones(4)},
            model,
            require_all_conv=True,
        )

    stats = validate_sensitivity_map_for_model(
        {"0.weight": torch.ones(4), "2.weight": torch.arange(5).float()},
        model,
        require_all_conv=True,
    )
    assert stats.n_layers == 2
    assert stats.n_channels == 9
    assert stats.min_value == 0.0
    assert stats.max_value == 4.0


def test_resolve_layer_sensitivity_accepts_canonical_and_bare_keys() -> None:
    model = _conv_model()
    conv0 = dict(model.named_modules())["0"]
    canonical = resolve_layer_sensitivity(
        {"0.weight": torch.ones(4)},
        module_name="0",
        weight=conv0.weight,
    )
    bare = resolve_layer_sensitivity(
        {"0": torch.ones(4) * 2.0},
        module_name="0",
        weight=conv0.weight,
    )
    assert torch.equal(canonical, torch.ones(4))
    assert torch.equal(bare, torch.ones(4) * 2.0)


def test_save_load_and_cv_distance_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sens.pt"
    sensitivities = {
        "0.weight": torch.tensor([1.0, 3.0, 0.0, 0.0]),
        "2.weight": torch.arange(5).float(),
    }
    save_sensitivity_map(
        path,
        sensitivities,
        metadata={"device": "cuda:0", "kind": "unit"},
    )
    loaded, metadata = load_sensitivity_map(path)
    assert metadata == {"device": "cuda:0", "kind": "unit"}
    assert set(loaded) == set(sensitivities)
    assert torch.equal(loaded["0.weight"], sensitivities["0.weight"])

    dist = sensitivity_cv_distance(
        {"x.weight": torch.tensor([1.0, 3.0])},
        {"x.weight": torch.tensor([2.0, 2.0])},
    )
    assert dist["x.weight"] == pytest.approx(0.5)

    with pytest.raises(SensitivityMapError, match="non-negative"):
        sensitivity_cv_distance(
            {"x.weight": torch.tensor([1.0, -1.0])},
            {"x.weight": torch.tensor([1.0, 1.0])},
        )


def test_contiguous_pair_manifest_is_deterministic_and_frame_aligned() -> None:
    manifest = build_contiguous_pair_manifest(
        3,
        latent_rows=10,
        source_bindings={"source_archive_sha256": "a" * 64},
    )
    assert manifest["format"] == "tac_sensitivity_pair_manifest_v1"
    assert manifest["n_pairs"] == 3
    assert manifest["pairs"] == [
        {"pair_index": 0, "latent_index": 0, "frame_start": 0, "frame_indices": [0, 1]},
        {"pair_index": 1, "latent_index": 1, "frame_start": 2, "frame_indices": [2, 3]},
        {"pair_index": 2, "latent_index": 2, "frame_start": 4, "frame_indices": [4, 5]},
    ]
    assert manifest["source_bindings"]["source_archive_sha256"] == "a" * 64

    text_a = canonical_sensitivity_json_bytes(manifest)
    text_b = canonical_sensitivity_json_bytes(dict(reversed(list(manifest.items()))))
    assert text_a == text_b
    assert sensitivity_manifest_sha256(manifest) == sensitivity_manifest_sha256(manifest)

    with pytest.raises(SensitivityMapError, match="exceeds latent rows"):
        build_contiguous_pair_manifest(4, latent_rows=3)


def _certification(component: str = "combined") -> dict[str, object]:
    return {
        "format": CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT,
        "component": component,
        "device": "cuda",
        "official_component_response": True,
        "canonical_scorer_path": True,
        "promotion_eligible": True,
        "source_map_sha256": "a" * 64,
        "official_response_curve_sha256": "b" * 64,
        "stability_sha256": "c" * 64,
        "sample_plan_sha256": "d" * 64,
        "baseline_archive_sha256": "e" * 64,
        "baseline_archive_bytes": 686635,
        "contest_auth_eval_json_sha256": "f" * 64,
        "review_clean_passes": 3,
        "review_unresolved_blockers": [],
    }


def test_certified_sensitivity_map_metadata_is_fail_closed(tmp_path: Path) -> None:
    cert = _certification("combined")
    metadata = {
        "component": "combined",
        "device": "cuda",
        "promotion_eligible": True,
        "official_component_response": True,
        "canonical_scorer_path": True,
        "certification": cert,
    }
    assert validate_certified_sensitivity_map_metadata(metadata, component="combined")["component"] == "combined"

    bad = dict(metadata)
    bad["certification"] = {**cert, "review_clean_passes": 2}
    with pytest.raises(SensitivityMapError, match="review_clean_passes"):
        validate_certified_sensitivity_map_metadata(bad, component="combined")

    for bad_blockers in (None, ["needs review"], "none"):
        bad = dict(metadata)
        bad["certification"] = {**cert, "review_unresolved_blockers": bad_blockers}
        with pytest.raises(SensitivityMapError, match="review_unresolved_blockers"):
            validate_certified_sensitivity_map_metadata(bad, component="combined")

    bad = dict(metadata)
    bad_cert = dict(cert)
    del bad_cert["review_unresolved_blockers"]
    bad["certification"] = bad_cert
    with pytest.raises(SensitivityMapError, match="review_unresolved_blockers"):
        validate_certified_sensitivity_map_metadata(bad, component="combined")

    path = tmp_path / "combined.pt"
    save_certified_sensitivity_map(
        path,
        {"layer.weight": torch.ones(3)},
        component="combined",
        certification=cert,
    )
    loaded, loaded_meta = load_sensitivity_map(path)
    assert torch.equal(loaded["layer.weight"], torch.ones(3))
    assert loaded_meta["official_component_response"] is True


def test_real_sensitivity_artifact_rejects_stub_uniform_and_stale_source() -> None:
    cert = _certification("combined")
    metadata = {
        "component": "combined",
        "device": "cuda",
        "promotion_eligible": True,
        "official_component_response": True,
        "canonical_scorer_path": True,
        "certification": cert,
    }
    proof = validate_real_sensitivity_artifact(
        {"layer.weight": torch.tensor([0.25, 1.0, 2.0])},
        metadata,
        source_archive_sha256="e" * 64,
        source_archive_bytes=686635,
        component="combined",
    )
    assert proof["n_tensors"] == 1
    assert proof["n_values"] == 3
    assert proof["certification"]["component"] == "combined"

    stub = {**metadata, "is_stub": True, "tag": "[stub-design-mode]"}
    with pytest.raises(SensitivityMapError, match="is_stub"):
        validate_real_sensitivity_artifact(
            {"layer.weight": torch.tensor([0.25, 1.0, 2.0])},
            stub,
            source_archive_sha256="e" * 64,
            source_archive_bytes=686635,
            component="combined",
        )

    with pytest.raises(SensitivityMapError, match="uniform"):
        validate_real_sensitivity_artifact(
            {"layer.weight": torch.ones(3)},
            metadata,
            source_archive_sha256="e" * 64,
            source_archive_bytes=686635,
            component="combined",
        )

    with pytest.raises(SensitivityMapError, match="stale or mismatched"):
        validate_real_sensitivity_artifact(
            {"layer.weight": torch.tensor([0.25, 1.0, 2.0])},
            metadata,
            source_archive_sha256="f" * 64,
            source_archive_bytes=686635,
            component="combined",
        )
