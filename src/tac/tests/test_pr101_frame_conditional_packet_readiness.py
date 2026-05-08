from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.codec.frame_conditional_bit_budget import FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA
from tac.pr101_frame_conditional_packet_readiness import (
    CANDIDATE_ARCHIVE_MANIFEST,
    PACKET_RUNTIME_PATCH_MANIFEST,
    PER_PAIR_SCORE_MARGINAL_MANIFEST,
    RUNTIME_CONSUMPTION_PROOF,
    STRICT_PRE_SUBMISSION_COMPLIANCE_JSON,
    build_packet_readiness,
)
from tac.repo_io import json_text

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools" / "build_pr101_frame_conditional_packet_readiness.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_pr101_frame_conditional_packet_readiness", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")
    return path


def _a5_manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "pr101_frame_conditional_bit_anchor.v1",
            "score_claim": False,
            "byte_proxy_only": True,
            "ready_for_exact_eval_dispatch": False,
            "input_archive": "archive.zip",
            "input_archive_bytes": 1000,
            "input_archive_sha256": "a" * 64,
            "n_pairs": 2,
            "latent_dim": 4,
            "best_eta": 4.0,
            "best_archive_delta_bytes": -20,
            "frame_conditional_wire_contract_status": {
                "typed_sideinfo_wire_contract_landed": True,
            },
            "rows": [
                {
                    "eta": 4.0,
                    "archive_delta_bytes": -20,
                    "frame_conditional_wire_contract": {
                        "schema": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
                        "score_claim": False,
                        "ready_for_exact_eval_dispatch": False,
                        "decoder_helper_consumes_sideinfo_bytes": True,
                        "q_bits_sideinfo": {
                            "bytes": 1,
                            "sha256": "b" * 64,
                        },
                        "latent_wire_payload": {
                            "bytes": 5,
                            "sha256": "c" * 64,
                            "score_affecting_payload_changed": True,
                        },
                        "q_bits_roundtrip": {"passed": True},
                        "latent_decode_roundtrip": {"passed": True},
                    },
                }
            ],
        },
    )


def test_a5_packet_readiness_fails_closed_with_explicit_missing_work(
    tmp_path: Path,
) -> None:
    manifest = build_packet_readiness(
        a5_manifest_path=_a5_manifest(tmp_path / "a5.json"),
        repo_root=tmp_path,
    )

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["ready_for_exact_eval_after_lane_claim"] is False
    assert CANDIDATE_ARCHIVE_MANIFEST in manifest["missing_artifacts"]
    assert PACKET_RUNTIME_PATCH_MANIFEST in manifest["missing_artifacts"]
    assert RUNTIME_CONSUMPTION_PROOF in manifest["missing_artifacts"]
    assert PER_PAIR_SCORE_MARGINAL_MANIFEST in manifest["missing_artifacts"]
    assert STRICT_PRE_SUBMISSION_COMPLIANCE_JSON in manifest["missing_artifacts"]
    assert "missing_frame_conditional_runtime_consumption_proof" in manifest[
        "readiness_blockers"
    ]
    assert "requires_level2_dispatch_claim_before_exact_eval" in manifest[
        "dispatch_blockers"
    ]
    assert manifest["a5_manifest"]["q_bits_sideinfo_sha256"] == "b" * 64
    assert manifest["operator_missing_work"][0]["id"] == CANDIDATE_ARCHIVE_MANIFEST


def test_a5_packet_readiness_accepts_valid_local_prerequisites_but_keeps_dispatch_closed(
    tmp_path: Path,
) -> None:
    a5_manifest = _a5_manifest(tmp_path / "a5.json")
    artifacts = {
        CANDIDATE_ARCHIVE_MANIFEST: _write_json(
            tmp_path / "candidate_archive.json",
            {
                "score_claim": False,
                "charged_bits_changed": True,
                "candidate_archive": {
                    "path": "candidate/archive.zip",
                    "bytes": 900,
                    "sha256": "d" * 64,
                },
            },
        ),
        PACKET_RUNTIME_PATCH_MANIFEST: _write_json(
            tmp_path / "runtime_patch.json",
            {
                "score_claim": False,
                "packet_local_runtime_patch": {
                    "consumes_schema": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
                    "parse_archive_consumes_q_bits_sideinfo": True,
                    "decode_latents_consumes_variable_width_payload": True,
                },
            },
        ),
        RUNTIME_CONSUMPTION_PROOF: _write_json(
            tmp_path / "runtime_proof.json",
            {
                "score_claim": False,
                "ready_for_exact_eval_runtime": True,
                "consumed_q_bits_sideinfo_sha256": "b" * 64,
                "consumed_latent_wire_payload_sha256": "c" * 64,
            },
        ),
        PER_PAIR_SCORE_MARGINAL_MANIFEST: _write_json(
            tmp_path / "marginals.json",
            {
                "score_claim": False,
                "n_pairs": 2,
                "marginal_evidence_available": True,
                "per_pair_score_marginals": [0.1, 0.2],
            },
        ),
        STRICT_PRE_SUBMISSION_COMPLIANCE_JSON: _write_json(
            tmp_path / "strict_compliance.json",
            {"score_claim": False, "ok": True},
        ),
    }

    manifest = build_packet_readiness(
        a5_manifest_path=a5_manifest,
        artifact_paths=artifacts,
        repo_root=tmp_path,
    )

    assert manifest["missing_artifacts"] == []
    assert manifest["invalid_artifacts"] == []
    assert manifest["readiness_blockers"] == []
    assert manifest["ready_for_local_packet_review"] is True
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_after_lane_claim"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["exact_eval_command_status"]["command_emitted"] is False
    assert "requires_level2_dispatch_claim_before_exact_eval" in manifest[
        "dispatch_blockers"
    ]


def test_a5_packet_readiness_rejects_score_claiming_anchor(tmp_path: Path) -> None:
    path = _a5_manifest(tmp_path / "a5.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["score_claim"] = True
    path.write_text(json_text(payload), encoding="utf-8")

    manifest = build_packet_readiness(a5_manifest_path=path, repo_root=tmp_path)

    assert "a5_manifest_must_not_claim_score" in manifest["readiness_blockers"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_a5_packet_readiness_rejects_nested_authority_flags(tmp_path: Path) -> None:
    path = _a5_manifest(tmp_path / "a5.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["frame_conditional_wire_contract"]["promotion_eligible"] = True
    path.write_text(json_text(payload), encoding="utf-8")

    manifest = build_packet_readiness(a5_manifest_path=path, repo_root=tmp_path)

    assert (
        "a5_manifest_authority_flag_true:"
        "$.rows[0].frame_conditional_wire_contract.promotion_eligible"
    ) in manifest["readiness_blockers"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_a5_packet_readiness_rejects_non_numeric_score_marginals(
    tmp_path: Path,
) -> None:
    a5_manifest = _a5_manifest(tmp_path / "a5.json")
    artifacts = {
        CANDIDATE_ARCHIVE_MANIFEST: _write_json(
            tmp_path / "candidate_archive.json",
            {
                "score_claim": False,
                "charged_bits_changed": True,
                "candidate_archive": {
                    "path": "candidate/archive.zip",
                    "bytes": 900,
                    "sha256": "d" * 64,
                },
            },
        ),
        PACKET_RUNTIME_PATCH_MANIFEST: _write_json(
            tmp_path / "runtime_patch.json",
            {
                "score_claim": False,
                "packet_local_runtime_patch": {
                    "consumes_schema": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
                    "parse_archive_consumes_q_bits_sideinfo": True,
                    "decode_latents_consumes_variable_width_payload": True,
                },
            },
        ),
        RUNTIME_CONSUMPTION_PROOF: _write_json(
            tmp_path / "runtime_proof.json",
            {
                "score_claim": False,
                "ready_for_exact_eval_runtime": True,
                "consumed_q_bits_sideinfo_sha256": "b" * 64,
                "consumed_latent_wire_payload_sha256": "c" * 64,
            },
        ),
        PER_PAIR_SCORE_MARGINAL_MANIFEST: _write_json(
            tmp_path / "marginals.json",
            {
                "score_claim": False,
                "n_pairs": 2,
                "marginal_evidence_available": True,
                "per_pair_score_marginals": ["0.1", 0.2],
            },
        ),
        STRICT_PRE_SUBMISSION_COMPLIANCE_JSON: _write_json(
            tmp_path / "strict_compliance.json",
            {"score_claim": False, "ok": True},
        ),
    }

    manifest = build_packet_readiness(
        a5_manifest_path=a5_manifest,
        artifact_paths=artifacts,
        repo_root=tmp_path,
    )

    assert PER_PAIR_SCORE_MARGINAL_MANIFEST in manifest["invalid_artifacts"]
    assert (
        f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:marginals_not_finite_numeric"
        in manifest["readiness_blockers"]
    )
    assert manifest["ready_for_exact_eval_after_lane_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_a5_packet_readiness_cli_writes_fail_closed_json(tmp_path: Path) -> None:
    tool = _load_tool()
    out = tmp_path / "readiness.json"

    rc = tool.main(
        [
            "--a5-manifest",
            str(_a5_manifest(tmp_path / "a5.json")),
            "--json-out",
            str(out),
            "--fail-if-not-ready",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["schema"] == "pr101_frame_conditional_packet_readiness.v1"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["tool_run_manifest"]["score_claim"] is False
