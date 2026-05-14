# SPDX-License-Identifier: MIT
"""Focused tests for section-level scorer evidence on Z1 MDL output."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.analysis.scorer_section_evidence import axis_label
from tac.analysis.scorer_conditional_mdl import (
    ArchiveInput,
    build_scorer_conditional_mdl_ablation,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_pr106_fixture(path: Path) -> bytes:
    decoder = b"D" * 12
    tail = b"L" * 24
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + tail
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload, compress_type=zipfile.ZIP_STORED)
    return payload


def _write_eval_json(
    path: Path,
    archive: Path,
    *,
    score_axis: str | None = "contest_cuda",
) -> None:
    archive_bytes = archive.read_bytes()
    payload = {
        "avg_posenet_dist": 0.000034,
        "avg_segnet_dist": 0.00062,
        "archive_size_bytes": archive.stat().st_size,
        "archive": {
            "archive_sha256": _sha256(archive_bytes),
            "archive_size_bytes": archive.stat().st_size,
        },
        "score_recomputed_from_components": 0.206,
    }
    if score_axis is not None:
        payload["score_axis"] = score_axis
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _evidence_map(tmp_path: Path) -> dict[str, Any]:
    artifact = tmp_path / "segnet_response_curve.json"
    artifact.write_text("{}", encoding="utf-8")
    return {
        "schema": "tac_section_scorer_evidence_map_v1",
        "schema_version": 1,
        "source": "unit_fixture_component_response",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "bindings": [
            {
                "archive_label": "pr106_fixture",
                "section_name": "decoder_packed_brotli",
                "component": "segnet",
                "evidence_type": "component_response_curve",
                "axis": "contest_cuda",
                "binding_strength": "section_level_component_response_fixture",
                "artifact": {
                    "artifact_type": "component_response_curve",
                    "path": artifact.as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": "a" * 64,
                    "evidence_grade": "diagnostic_cuda_direct_renderer_component_response",
                    "official_component_response": False,
                    "canonical_scorer_path": False,
                    "passed": False,
                    "promotion_eligible": False,
                    "score_claim": False,
                },
            }
        ],
    }


def _ready_evidence_map(tmp_path: Path, *, section_name: str) -> dict[str, Any]:
    artifact = tmp_path / f"{section_name}_ready_response_curve.json"
    artifact.write_text('{"passed":true}', encoding="utf-8")
    artifact_bytes = artifact.read_bytes()
    return {
        "schema": "tac_section_scorer_evidence_map_v1",
        "schema_version": 1,
        "source": "unit_fixture_true_component_response_one_section_only",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "bindings": [
            {
                "archive_label": "pr106_fixture",
                "section_name": section_name,
                "component": "segnet",
                "evidence_type": "component_response_curve",
                "axis": "contest_cuda",
                "binding_strength": "section_level_component_response_fixture",
                "artifact": {
                    "artifact_type": "component_response_curve",
                    "path": artifact.as_posix(),
                    "bytes": len(artifact_bytes),
                    "sha256": _sha256(artifact_bytes),
                    "evidence_grade": "diagnostic_cuda_direct_renderer_component_response",
                    "official_component_response": True,
                    "canonical_scorer_path": True,
                    "passed": True,
                    "promotion_eligible": False,
                    "score_claim": False,
                },
            }
        ],
    }


def _assert_no_score_claims(value: Any) -> None:
    if isinstance(value, Mapping):
        if "score_claim" in value:
            assert value["score_claim"] is False
        for child in value.values():
            _assert_no_score_claims(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_score_claims(child)


def test_missing_section_scorer_evidence_fails_closed_as_proxy(tmp_path: Path) -> None:
    archive = tmp_path / "pr106_fixture.zip"
    _write_pr106_fixture(archive)

    manifest = build_scorer_conditional_mdl_ablation(
        [ArchiveInput(label="pr106_fixture", archive_path=archive)],
        repo_root=tmp_path,
    )

    proxy_layer = manifest["measurement_layers"]["scorer_feature_proxy_conditioned"]
    section_layer = manifest["section_scorer_evidence"]
    assert proxy_layer["claim_strength"] == "proxy_not_true_scorer_conditional_entropy"
    assert section_layer["claim_strength"] == "proxy_not_true_scorer_conditional_entropy"
    assert "section_scorer_evidence_map_missing" in section_layer["blockers"]
    assert manifest["true_scorer_conditional_entropy_claim"] is False
    assert manifest["score_claim"] is False


def test_section_evidence_map_records_bindings_but_not_true_without_required_artifacts(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pr106_fixture.zip"
    _write_pr106_fixture(archive)

    manifest = build_scorer_conditional_mdl_ablation(
        [ArchiveInput(label="pr106_fixture", archive_path=archive)],
        repo_root=tmp_path,
        section_scorer_evidence=_evidence_map(tmp_path),
    )

    layer = manifest["section_scorer_evidence"]
    assert layer["component_binding_count"] == 1
    assert layer["bound_section_count"] == 1
    assert layer["unbound_section_count"] == 2
    assert "unbound_sections" in layer["blockers"]
    assert layer["claim_strength"] == (
        "section_bound_scorer_evidence_incomplete_not_true_scorer_conditional_entropy"
    )
    assert layer["true_scorer_conditioning_ready"] is False
    assert manifest["true_scorer_conditional_entropy_claim"] is False

    decoder_section = next(
        row for row in layer["sections"] if row["section_name"] == "decoder_packed_brotli"
    )
    binding = decoder_section["component_bindings"][0]
    assert binding["component"] == "segnet"
    assert binding["axis_label"] == "[contest-CUDA]"
    assert binding["true_scorer_ready"] is False
    assert "official_component_response_not_true" in binding["blockers"]
    assert "canonical_scorer_path_not_true" in binding["blockers"]
    assert "component_response_curve_not_passed" in binding["blockers"]


def test_one_ready_section_does_not_mark_whole_archive_scorer_ready(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pr106_fixture.zip"
    _write_pr106_fixture(archive)

    manifest = build_scorer_conditional_mdl_ablation(
        [ArchiveInput(label="pr106_fixture", archive_path=archive)],
        repo_root=tmp_path,
        section_scorer_evidence=_ready_evidence_map(
            tmp_path, section_name="decoder_packed_brotli"
        ),
    )

    layer = manifest["section_scorer_evidence"]
    assert layer["section_count"] == 3
    assert layer["bound_section_count"] == 1
    assert layer["true_scorer_ready_binding_count"] == 1
    assert layer["true_scorer_ready_section_count"] == 1
    assert layer["unbound_section_count"] == 2
    assert layer["unbound_sections"] == [
        "pr106_fixture:latents_and_sidecar_brotli",
        "pr106_fixture:packed_header_ff_len24",
    ]
    assert "unbound_sections" in layer["blockers"]
    assert layer["true_scorer_conditioning_ready"] is False
    assert manifest["true_scorer_conditional_entropy_claim"] is False
    assert layer["claim_strength"] == (
        "section_bound_scorer_evidence_incomplete_not_true_scorer_conditional_entropy"
    )


def test_axis_labels_are_explicit_and_no_score_claim_is_emitted(tmp_path: Path) -> None:
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "contest_eval.json"
    _write_pr106_fixture(archive)
    _write_eval_json(eval_json, archive)

    manifest = build_scorer_conditional_mdl_ablation(
        [
            ArchiveInput(
                label="pr106_fixture",
                archive_path=archive,
                eval_json_path=eval_json,
            )
        ],
        repo_root=tmp_path,
        section_scorer_evidence=_evidence_map(tmp_path),
    )

    assert "[contest-CUDA]" in manifest["axis_labels"]
    assert (
        manifest["archives"][0]["scorer_feature_summary"]["score_axis_label"]
        == "[contest-CUDA]"
    )
    assert manifest["measurement_layers"]["scorer_feature_proxy_conditioned"]["axis_labels"] == [
        "[contest-CUDA]"
    ]
    assert manifest["section_scorer_evidence"]["axis_labels"] == ["[contest-CUDA]"]
    _assert_no_score_claims(manifest)
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False


def test_mps_axis_label_is_distinct_from_macos_cpu_advisory() -> None:
    assert axis_label("mps") == "[MPS advisory]"
    assert axis_label("mps_proxy") == "[MPS advisory]"
    assert axis_label("macos_cpu_advisory") == "[macOS-CPU advisory]"


def test_mps_path_inference_is_not_reported_as_macos_cpu(tmp_path: Path) -> None:
    archive = tmp_path / "pr106_fixture.zip"
    eval_json = tmp_path / "mps_proxy_eval.json"
    _write_pr106_fixture(archive)
    _write_eval_json(eval_json, archive, score_axis=None)

    manifest = build_scorer_conditional_mdl_ablation(
        [
            ArchiveInput(
                label="pr106_fixture",
                archive_path=archive,
                eval_json_path=eval_json,
            )
        ],
        repo_root=tmp_path,
    )

    assert manifest["archives"][0]["scorer_feature_summary"]["score_axis"] == "mps"
    assert (
        manifest["archives"][0]["scorer_feature_summary"]["score_axis_label"]
        == "[MPS advisory]"
    )
