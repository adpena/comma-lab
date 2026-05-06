from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.precompute_gradient_corrections import pack_sparse_corrections
from tac.engineered_correction_readiness import (
    audit_sparse_corrections,
    detector_cost_atom_from_correction_report,
)
from tac.uniward_delta import build_detector_cost_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "audit_engineered_corrections.py"


def _sparse() -> dict:
    return {
        "indices": np.array([1, 7], dtype=np.uint32),
        "values": np.array([[10, -2, 0], [0, 3, -4]], dtype=np.int8),
        "scale": 2.0,
        "shape": [1, 3, 3, 3],
        "top_k_pct": 22.2,
        "quantize_bits": 8,
        "n_kept": 2,
        "n_total": 9,
    }


def test_engineered_correction_readiness_accepts_bounded_nonzero_roundtrip() -> None:
    report = audit_sparse_corrections(_sparse(), max_packed_bytes=10_000)

    assert report.ready_for_local_patch is True
    assert report.ready_for_exact_eval_dispatch is False
    assert report.score_claim is False
    assert report.dispatch_attempted is False
    assert report.n_kept == 2
    assert report.packed_bytes > 0
    assert report.blockers == ()


def test_engineered_correction_readiness_rejects_noop_duplicate_and_4_channel() -> None:
    sparse = _sparse()
    sparse["indices"] = np.array([1, 1], dtype=np.uint32)
    sparse["values"] = np.zeros((2, 3), dtype=np.int8)
    sparse["shape"] = [1, 3, 3, 4]

    report = audit_sparse_corrections(sparse, max_packed_bytes=10_000)

    assert report.ready_for_local_patch is False
    assert "duplicate_indices" in report.blockers
    assert "all_correction_values_zero" in report.blockers
    assert "wire_format_requires_3_channels_got_4" in report.blockers


def test_engineered_correction_readiness_rejects_score_claim_and_oversize() -> None:
    report = audit_sparse_corrections(
        _sparse(),
        max_packed_bytes=1,
        manifest={"score_claim": True, "dispatch_attempted": True},
    )

    assert report.ready_for_local_patch is False
    assert "manifest_score_claim_true" in report.blockers
    assert "manifest_dispatch_attempted_true" in report.blockers
    assert any(blocker.startswith("packed_bytes_exceed_cap") for blocker in report.blockers)


def test_engineered_correction_readiness_rejects_int4_out_of_range() -> None:
    sparse = _sparse()
    sparse["quantize_bits"] = 4
    sparse["values"] = np.array([[8, 0, 0], [0, -8, 0]], dtype=np.int8)

    report = audit_sparse_corrections(sparse, max_packed_bytes=10_000)

    assert report.ready_for_local_patch is False
    assert "int4_corrections_out_of_range" in report.blockers


def test_correction_readiness_feeds_fridrich_detector_cost_manifest() -> None:
    report = audit_sparse_corrections(_sparse(), max_packed_bytes=10_000)
    atom = detector_cost_atom_from_correction_report(
        report,
        atom_id="correction:fixture",
        detector_capacity=0.75,
        positive_scorer_sensitivity=0.006,
    )

    manifest = build_detector_cost_manifest([atom], source_label="correction_fixture")

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["rows"][0]["atom_id"] == "correction:fixture"
    assert manifest["rows"][0]["charged_bytes"] == report.packed_bytes
    assert manifest["rows"][0]["allocation_priority"] > 0.0
    assert manifest["rows"][0]["promotion_eligible"] is False


def test_engineered_correction_readiness_requires_component_trace_plan() -> None:
    report = audit_sparse_corrections(
        _sparse(),
        max_packed_bytes=10_000,
        require_component_trace_plan=True,
    )

    assert report.ready_for_local_patch is False
    assert report.component_trace_signed is False
    assert "component_trace_plan_required" in report.blockers


def test_engineered_correction_readiness_accepts_cross_checked_trace_plan() -> None:
    report = audit_sparse_corrections(
        _sparse(),
        max_packed_bytes=10_000,
        component_trace_plan=_component_trace_plan(),
        require_component_trace_plan=True,
    )

    assert report.ready_for_local_patch is True
    assert report.component_trace_signed is True
    assert report.component_trace_atom_count == 1
    assert report.component_trace_plan_sha256 == _component_trace_plan()["stable_plan_digest_sha256"]


def test_engineered_correction_readiness_cli_json(tmp_path: Path) -> None:
    correction_bin = tmp_path / "gradient_corrections.bin"
    correction_bin.write_bytes(pack_sparse_corrections(_sparse()))

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(correction_bin),
            "--max-packed-bytes",
            "10000",
            "--fail-if-not-ready",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ready_for_local_patch"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["packed_bytes"] == correction_bin.stat().st_size


def test_engineered_correction_readiness_cli_requires_trace_plan(tmp_path: Path) -> None:
    correction_bin = tmp_path / "gradient_corrections.bin"
    trace_plan = tmp_path / "component_trace_atom_plan.json"
    correction_bin.write_bytes(pack_sparse_corrections(_sparse()))
    trace_plan.write_text(json.dumps(_component_trace_plan()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(correction_bin),
            "--max-packed-bytes",
            "10000",
            "--component-trace-plan",
            str(trace_plan),
            "--require-component-trace-plan",
            "--fail-if-not-ready",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ready_for_local_patch"] is True
    assert payload["component_trace_signed"] is True
    assert payload["component_trace_atom_count"] == 1


def test_engineered_correction_readiness_cli_fails_closed_on_score_claim(tmp_path: Path) -> None:
    correction_bin = tmp_path / "gradient_corrections.bin"
    manifest = tmp_path / "manifest.json"
    correction_bin.write_bytes(pack_sparse_corrections(_sparse()))
    manifest.write_text(json.dumps({"score_claim": True}), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(correction_bin),
            "--max-packed-bytes",
            "10000",
            "--manifest",
            str(manifest),
            "--fail-if-not-ready",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "manifest_score_claim_true" in proc.stdout


def test_engineered_correction_readiness_cli_self_test() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--self-test",
            "--max-packed-bytes",
            "10000",
            "--fail-if-not-ready",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ready_for_local_patch"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False


def _component_trace_plan() -> dict:
    payload = {
        "schema": "pr85_scorer_gradient_atom_opportunity_v1",
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "input_artifacts": {
            "component_traces": [
                {
                    "score_claim": False,
                    "evidence_grade": "diagnostic_component_trace",
                    "trace_cross_checked_to_exact_eval": True,
                }
            ]
        },
        "atom_ranking": [
            {
                "atom_id": "trace:pair_0001",
                "source_kind": "component_trace",
                "ranking_score": 0.0125,
                "score_claim": False,
                "promotion_eligible": False,
                "dispatch_gate": {"dispatchable": False},
            }
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    payload["stable_plan_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
