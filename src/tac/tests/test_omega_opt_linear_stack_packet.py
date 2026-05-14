# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tac.omega_opt_linear_stack_packet import (
    CANONICAL_LINEAR_STACK_LAYER_IDS,
    OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA,
    LinearStackLayer,
    build_linear_stack_packet_manifest,
    canonical_json_sha256,
    default_linear_stack_layers,
    has_exact_linear_stack_anchor,
    linear_stack_packet_status,
    validate_linear_stack_packet_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEST_DENOMINATOR = 37_545_489


def _strict_auth_eval(*, archive_bytes: int, archive_sha256: str) -> dict[str, object]:
    pose = 0.0001
    seg = 0.001
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / CONTEST_DENOMINATOR
    return {
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "rate_unscaled": archive_bytes / CONTEST_DENOMINATOR,
        "n_samples": 600,
        "provenance": {
            "archive_size_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": 1,
            "gpu_t4_match": True,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "b" * 64,
            },
        },
    }


def _write_exact_anchor_files(tmp_path: Path) -> dict[str, object]:
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"omega-linear-stack")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    runtime_packet = tmp_path / "runtime_packet.json"
    runtime_packet.write_text('{"ok": true}\n', encoding="utf-8")
    inflate_sh = tmp_path / "inflate.sh"
    inflate_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    auth_eval_json = tmp_path / "contest_auth_eval.json"
    auth_eval_json.write_text(
        json.dumps(
            _strict_auth_eval(
                archive_bytes=archive_path.stat().st_size,
                archive_sha256=archive_sha,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    anchor = tmp_path / "one_to_one_anchor.json"
    anchor.write_text('{"anchor": true}\n', encoding="utf-8")
    layers = tuple(
        LinearStackLayer(
            layer_id=layer.layer_id,
            order_index=layer.order_index,
            transform_kind=layer.transform_kind,
            input_artifact_sha256=f"{index + 1:064x}",
            output_artifact_sha256=f"{index + 11:064x}",
            charged_byte_delta=-index,
            runtime_consumed=True,
        )
        for index, layer in enumerate(default_linear_stack_layers())
    )
    return {
        "archive_path": archive_path,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "runtime_packet_path": runtime_packet,
        "inflate_path": inflate_sh,
        "contest_auth_eval_json": auth_eval_json,
        "one_to_one_anchor_artifact": anchor,
        "layers": layers,
    }


def test_linear_stack_scaffold_is_scoreless_and_non_promotable() -> None:
    manifest = build_linear_stack_packet_manifest()
    status = linear_stack_packet_status(manifest)

    assert manifest["schema"] == OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA
    assert manifest["claim_id"] == "omega_opt_linear_stack"
    assert [layer["layer_id"] for layer in manifest["linear_stack"]] == list(CANONICAL_LINEAR_STACK_LAYER_IDS)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert status["exact_anchor_complete"] is False
    assert status["promotion_eligible"] is False
    assert status["rank_or_kill_eligible"] is False
    assert status["ready_for_exact_eval_dispatch"] is False
    assert "archive_sha256_missing_or_invalid" in status["blockers"]
    assert "runtime_packet_path_missing" in status["blockers"]
    assert validate_linear_stack_packet_manifest(manifest) == []
    assert manifest["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def test_unanchored_predicted_linear_stack_row_cannot_promote() -> None:
    manifest = build_linear_stack_packet_manifest(
        evidence_grade="prediction",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        promotion_allowed=True,
        dispatchable=True,
    )

    findings = validate_linear_stack_packet_manifest(manifest)
    status = linear_stack_packet_status(manifest)

    assert "score_claim_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "promotion_eligible_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "rank_or_kill_eligible_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "ready_for_exact_eval_dispatch_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "promotion_allowed_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "dispatchable_must_be_false_without_exact_linear_stack_anchor" in findings
    assert "exact_cuda_auth_eval_json_missing" in findings
    assert has_exact_linear_stack_anchor(manifest) is False
    assert status["score_claim"] is False
    assert status["promotion_eligible"] is False
    assert status["rank_or_kill_eligible"] is False
    assert status["ready_for_exact_eval_dispatch"] is False


def test_complete_a_plus_plus_fixture_can_promote_only_with_exact_eval_files(tmp_path: Path) -> None:
    files = _write_exact_anchor_files(tmp_path)
    manifest = build_linear_stack_packet_manifest(
        archive_path=str(files["archive_path"]),
        archive_bytes=files["archive_bytes"],
        archive_sha256=str(files["archive_sha256"]),
        runtime_packet_path=str(files["runtime_packet_path"]),
        inflate_path=str(files["inflate_path"]),
        evidence_grade="A++",
        contest_auth_eval_json=str(files["contest_auth_eval_json"]),
        one_to_one_anchor_artifact=str(files["one_to_one_anchor_artifact"]),
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        promotion_allowed=True,
        dispatchable=True,
        layers=files["layers"],
    )

    assert has_exact_linear_stack_anchor(manifest) is True
    assert validate_linear_stack_packet_manifest(manifest) == []
    status = linear_stack_packet_status(manifest)
    assert status["exact_anchor_complete"] is True
    assert status["score_claim"] is True
    assert status["promotion_eligible"] is True
    assert status["rank_or_kill_eligible"] is True
    assert status["ready_for_exact_eval_dispatch"] is True

    without_eval = build_linear_stack_packet_manifest(
        archive_path=str(files["archive_path"]),
        archive_bytes=files["archive_bytes"],
        archive_sha256=str(files["archive_sha256"]),
        runtime_packet_path=str(files["runtime_packet_path"]),
        inflate_path=str(files["inflate_path"]),
        evidence_grade="A++",
        contest_auth_eval_json=None,
        one_to_one_anchor_artifact=str(files["one_to_one_anchor_artifact"]),
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        layers=files["layers"],
    )

    assert has_exact_linear_stack_anchor(without_eval) is False
    assert "exact_cuda_auth_eval_json_missing" in validate_linear_stack_packet_manifest(without_eval)
    assert linear_stack_packet_status(without_eval)["promotion_eligible"] is False


def test_linear_stack_manifest_accepts_flat_exact_anchor_aliases(tmp_path: Path) -> None:
    files = _write_exact_anchor_files(tmp_path)
    manifest = build_linear_stack_packet_manifest()
    manifest["linear_stack"] = [layer.to_manifest() for layer in files["layers"]]
    manifest.update({
        "evidence_grade": "A++",
        "archive_path": str(files["archive_path"]),
        "exact_archive_bytes": files["archive_bytes"],
        "exact_archive_sha256": str(files["archive_sha256"]),
        "runtime_packet_path": str(files["runtime_packet_path"]),
        "inflate_path": str(files["inflate_path"]),
        "exact_cuda_auth_eval_json": str(files["contest_auth_eval_json"]),
        "one_to_one_archive_eval_artifact": str(files["one_to_one_anchor_artifact"]),
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "promotion_allowed": True,
        "dispatchable": True,
    })
    manifest["promotion_status"] = linear_stack_packet_status(manifest)
    manifest["blockers"] = manifest["promotion_status"]["blockers"]
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest = json.loads(json.dumps(manifest, sort_keys=True))

    assert has_exact_linear_stack_anchor(manifest) is True
    assert validate_linear_stack_packet_manifest(manifest) == []


def test_linear_stack_packet_checker_cli_accepts_fail_closed_scaffold(tmp_path: Path) -> None:
    manifest_path = tmp_path / "linear_stack_manifest.json"
    manifest_path.write_text(json.dumps(build_linear_stack_packet_manifest()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/check_omega_opt_linear_stack_packet.py"),
            str(manifest_path),
            "--strict",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["finding_count"] == 0
    assert payload["promotion_status"]["exact_anchor_complete"] is False
    assert payload["promotion_status"]["promotion_eligible"] is False


def test_materialize_linear_stack_packet_cli_emits_default_fail_closed_scaffold(tmp_path: Path) -> None:
    manifest_path = tmp_path / "materialized_linear_stack_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/materialize_omega_opt_linear_stack_packet.py"),
            "--output",
            str(manifest_path),
            "--score-claim",
            "--promotion-eligible",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["promotion_status"]["exact_anchor_complete"] is False
    assert "exact_linear_stack_anchor_missing" in manifest["blockers"]
    assert manifest["suppressed_positive_intents"] == {
        "fields": ["promotion_eligible", "score_claim"],
        "reason": "exact_a_plus_plus_linear_stack_anchor_incomplete",
    }
    assert validate_linear_stack_packet_manifest(manifest) == []


def test_materialize_linear_stack_packet_cli_normalizes_nested_plan_claim(tmp_path: Path) -> None:
    source_plan = REPO_ROOT / "reports/hstack_vstack_multipass_plan_20260507.json"
    manifest_path = tmp_path / "from_plan_linear_stack_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/materialize_omega_opt_linear_stack_packet.py"),
            "--source-plan",
            str(source_plan),
            "--output",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_claim = manifest["source_claim"]
    normalized_claim = source_claim["normalized_claim"]

    assert source_claim["json_pointer"] == "/metadata/nested_optimization/score_band_prediction/claims/0"
    assert normalized_claim["claim_id"] == "omega_opt_linear_stack"
    assert normalized_claim["predicted_score"] == 0.13
    assert manifest["predicted_score"] == 0.13
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["promotion_status"]["exact_anchor_complete"] is False
    assert validate_linear_stack_packet_manifest(manifest) == []


def test_materialize_linear_stack_packet_cli_suppresses_positive_flags_without_layer_proof(tmp_path: Path) -> None:
    files = _write_exact_anchor_files(tmp_path)
    manifest_path = tmp_path / "exact_linear_stack_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/materialize_omega_opt_linear_stack_packet.py"),
            "--output",
            str(manifest_path),
            "--archive-path",
            str(files["archive_path"]),
            "--archive-bytes",
            str(files["archive_bytes"]),
            "--archive-sha256",
            str(files["archive_sha256"]),
            "--runtime-packet-path",
            str(files["runtime_packet_path"]),
            "--inflate-path",
            str(files["inflate_path"]),
            "--evidence-grade",
            "A++",
            "--contest-auth-eval-json",
            str(files["contest_auth_eval_json"]),
            "--one-to-one-anchor-artifact",
            str(files["one_to_one_anchor_artifact"]),
            "--score-claim",
            "--promotion-eligible",
            "--rank-or-kill-eligible",
            "--ready-for-exact-eval-dispatch",
            "--promotion-allowed",
            "--dispatchable",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["promotion_status"]["exact_anchor_complete"] is False
    assert manifest["promotion_status"]["score_claim"] is False
    assert manifest["promotion_status"]["promotion_eligible"] is False
    assert "linear_stack_layers_missing_runtime_consumption_proof" in manifest["blockers"]
    assert set(manifest["suppressed_positive_intents"]["fields"]) == {
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "promotion_allowed",
        "dispatchable",
    }
    assert validate_linear_stack_packet_manifest(manifest) == []
