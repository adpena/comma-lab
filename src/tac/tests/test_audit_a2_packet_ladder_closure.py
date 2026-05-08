"""Tests for the A2 packet-ladder closure audit CLI."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO_ROOT / "tools" / "audit_a2_packet_ladder_closure.py"
    spec = importlib.util.spec_from_file_location("_audit_a2_packet_ladder_closure", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def _stub_sensitivity_artifact() -> dict[str, object]:
    return {
        "path": "experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt",
        "status": "diagnostic_allowed",
        "allow_diagnostic_sensitivity": True,
        "metadata_blockers": ["is_stub=true", "tag contains 'stub'"],
        "metadata": {
            "device": "cpu",
            "is_stub": True,
            "tag": "[stub-design-mode]",
        },
    }


def test_audit_a2_packet_ladder_closure_accepts_repo_root(tmp_path: Path) -> None:
    """All-lanes preflight invokes guards with --repo-root; keep that contract."""
    mod = _load_module()
    out = tmp_path / "report.json"

    rc = mod.main(["--repo-root", str(tmp_path), "--strict", "--json-out", str(out)])

    assert rc == 0
    assert out.exists()
    assert "a2_packet_ladder_closure_audit_v1" in out.read_text(encoding="utf-8")


def test_audit_a2_packet_ladder_closure_scans_ignored_local_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_local"
        / "candidate_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_candidate_manifest.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["no_exact_cuda_auth_eval"],
                "runtime_closure": {
                    "cleared_blockers": ["packet_local_inflate_parity_not_run"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is False
    assert report["scanned_artifacts"] == [
        "experiments/results/a2_local/candidate_manifest.json"
    ]
    assert any("cleared without evidence" in item for item in report["violations"])


def test_audit_a2_packet_ladder_closure_rejects_extended_false_authority(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_false_authority"
        / "candidate_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_candidate_manifest.v1",
                "score_claim": False,
                "score_claim_valid": True,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": True,
                "evidence_grade": "A++",
                "dispatch_blockers": ["no_exact_cuda_auth_eval"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is False
    assert any("score_claim_valid must be False" in item for item in report["violations"])
    assert any("dispatch_attempted must be False" in item for item in report["violations"])
    assert any("evidence_grade" in item and "A++" in item for item in report["violations"])


def test_audit_a2_packet_ladder_closure_requires_stub_sensitivity_blockers(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_stub_missing_blockers"
        / "a2_packet_ladder_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_sensitivity_weighted_pr101_packet_ladder.v1",
                "tool": "tools/build_a2_sensitivity_weighted_pr101_packet.py",
                **_false_authority(),
                "dispatch_blockers": [
                    "no_exact_cuda_auth_eval",
                    "packet_local_inflate_parity_not_run",
                ],
                "upstream_a2_manifest": {
                    "dispatch_blockers": [
                        "cpu_local_allocator_proxy_only",
                        "diagnostic_or_stub_sensitivity_map_not_score_authority",
                        "score_sensitivity_artifact_must_be_certified_before_promotion",
                    ],
                    "sensitivity_artifact": _stub_sensitivity_artifact(),
                },
                "packet_closure": {
                    "byte_closed_packet_ladder_built": True,
                    "cleared_blockers": ["no_byte_closed_runtime_packet_built"],
                    "cleared_blockers_by_evidence": {
                        "no_byte_closed_runtime_packet_built": "packet_local_parse_smoke",
                    },
                    "inflate_parity_status": "not_run",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is False
    assert any(
        "sensitivity/proxy blockers missing from dispatch_blockers" in item
        and "is_stub=true" in item
        and "cpu_local_allocator_proxy_only" in item
        for item in report["violations"]
    )


def test_audit_a2_packet_ladder_closure_rejects_runtime_probe_clearing_parity(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_runtime_probe"
        / "a2_runtime_closure_probe.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_packet_runtime_closure_probe.v1",
                "tool": "tools/probe_a2_packet_runtime_closure.py",
                **_false_authority(),
                "dispatch_blockers": [
                    "no_exact_cuda_auth_eval",
                    "no_contest_cpu_auth_eval",
                    "no_active_level2_lane_dispatch_claim",
                    "operator_score_claim_review_not_done",
                ],
                "runtime_closure": {
                    "verified": True,
                    "cleared_blockers": ["packet_local_inflate_parity_not_run"],
                    "cleared_blockers_by_evidence": {
                        "packet_local_inflate_parity_not_run": (
                            "runtime_closure_probe_strict_model_load_and_a2_parse"
                        ),
                    },
                    "remaining_blockers": [
                        "no_exact_cuda_auth_eval",
                        "no_contest_cpu_auth_eval",
                        "no_active_level2_lane_dispatch_claim",
                        "operator_score_claim_review_not_done",
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is False
    assert any("expected one of ['inflate_parity_log']" in item for item in report["violations"])
    assert any("candidate_manifest evidence" in item for item in report["violations"])


def test_audit_a2_packet_ladder_closure_accepts_stub_when_all_blockers_remain(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    blockers = [
        "cpu_local_allocator_proxy_only",
        "diagnostic_or_stub_sensitivity_map_not_score_authority",
        "is_stub=true",
        "no_active_level2_lane_dispatch_claim",
        "no_contest_cpu_auth_eval",
        "no_exact_cuda_auth_eval",
        "operator_score_claim_review_not_done",
        "packet_local_inflate_parity_not_run",
        "score_sensitivity_artifact_must_be_certified_before_promotion",
        "tag contains 'stub'",
    ]
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_stub_blocked"
        / "candidate_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_sensitivity_weighted_pr101_packet_variant.v1",
                "tool": "tools/build_a2_sensitivity_weighted_pr101_packet.py",
                **_false_authority(),
                "dispatch_blockers": blockers,
                "upstream_a2_manifest": {
                    "dispatch_blockers": blockers,
                    "sensitivity_artifact": _stub_sensitivity_artifact(),
                },
                "packet_closure": {
                    "byte_closed_packet_built": True,
                    "runtime_consumes_changed_archive_bytes": True,
                    "cleared_blockers": ["no_byte_closed_runtime_packet_built"],
                    "cleared_blockers_by_evidence": {
                        "no_byte_closed_runtime_packet_built": "packet_local_parse_smoke",
                    },
                    "inflate_parity_status": "not_run",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is True
