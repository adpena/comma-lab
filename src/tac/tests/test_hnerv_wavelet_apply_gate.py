from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from tac.hnerv_wavelet_apply_gate import (
    CONTEST_ORIGINAL_BYTES,
    build_wavelet_apply_gate,
)

REPO = Path(__file__).resolve().parents[3]


def test_wavelet_apply_gate_computes_rate_and_seg_break_even() -> None:
    payload = build_wavelet_apply_gate(
        sidechannel_manifest=_sidechannel_manifest(),
        stacked_metadata=_stacked_metadata(),
        required_component_margin=0.001,
    )

    expected_rate = 25.0 * 387.0 / CONTEST_ORIGINAL_BYTES
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_archive_preflight"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["stacked_archive_path"] == "candidate.zip"
    assert math.isclose(payload["rate_score_delta"], expected_rate)
    assert math.isclose(payload["break_even_score_improvement"], expected_rate + 0.001)
    assert math.isclose(payload["min_required_seg_dist_reduction"], (expected_rate + 0.001) / 100.0)
    assert payload["decoded_atom_count"] == 32
    assert "wr01_runtime_mode_is_explicit_noop" in payload["dispatch_blockers"]
    assert "requires_component_benefit_evidence_over_break_even" in payload["dispatch_blockers"]


def test_wavelet_apply_gate_computes_pose_break_even() -> None:
    payload = build_wavelet_apply_gate(
        sidechannel_manifest=_sidechannel_manifest(),
        stacked_metadata=_stacked_metadata(),
        baseline_pose_dist=0.01,
    )

    pose = payload["pose_break_even"]
    assert pose is not None
    assert pose["required_pose_dist"] < 0.01
    assert pose["min_required_pose_dist_reduction"] > 0.0


def test_audit_hnerv_wavelet_apply_gate_cli(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    metadata = tmp_path / "metadata.json"
    out = tmp_path / "gate.json"
    manifest.write_text(json.dumps(_sidechannel_manifest()), encoding="utf-8")
    metadata.write_text(json.dumps(_stacked_metadata()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_hnerv_wavelet_apply_gate.py"),
            "--sidechannel-manifest",
            str(manifest),
            "--stacked-metadata",
            str(metadata),
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["audit"] == "hnerv_wavelet_apply_gate"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["archive_byte_delta"] == 387


def test_apply_gate_propagates_source_archive_sha256_from_apply_transform_manifest() -> None:
    """Round 6 R6-3 + Round 7 R7-3 fix (2026-05-06): the gate's
    `source_archive_sha256` output must reflect whatever value the input
    sidechannel_manifest carries — including None when the apply_transform
    manifest is the input and the operator did not supply a SHA. This pins
    R6-1's contract: apply_transform no longer auto-derives a misleading
    payload-bytes hash; it propagates None or the caller-supplied SHA
    untouched.

    R7-3 hardening: use `_apply_transform_manifest()` (the actual shape
    `build_wavelet_apply_transform_candidate` emits — keyed by
    `candidate_archive_byte_delta_vs_source_estimate`, NOT the sidechannel
    manifest's `candidate_archive_byte_delta`). This exercises the
    apply_transform manifest path through the gate, not the sidechannel
    manifest path. R6-3's previous use of `_sidechannel_manifest()` was
    a false-green: a future regression of the new-key fallback would not
    fail this test.
    """
    apply_transform_with_sha = _apply_transform_manifest()
    apply_transform_with_sha["source_archive_sha256"] = "c" * 64

    payload = build_wavelet_apply_gate(
        sidechannel_manifest=apply_transform_with_sha,
        stacked_metadata=_stacked_metadata(),
    )
    assert payload["source_archive_sha256"] == "c" * 64

    apply_transform_no_sha = _apply_transform_manifest()
    apply_transform_no_sha["source_archive_sha256"] = None

    payload_none = build_wavelet_apply_gate(
        sidechannel_manifest=apply_transform_no_sha,
        stacked_metadata=_stacked_metadata(),
    )
    assert payload_none["source_archive_sha256"] is None


def _sidechannel_manifest() -> dict:
    return {
        "score_claim": False,
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "b" * 64,
        "candidate_archive_byte_delta": 388,
        "wavelet_sidechannel_bytes": 379,
        "decoded_wavelet_sidechannel": {"total_atom_count": 32},
        "runtime_consumption_proof": {
            "runtime_consumed": True,
            "decoded_atom_count": 32,
            "score_claim": False,
        },
    }


def _stacked_metadata() -> dict:
    # Round 7 R7-4 note (2026-05-06): the 387 here is the AUTHORITATIVE
    # delta the gate uses (stacked_delta wins over manifest-only delta in
    # `build_wavelet_apply_gate` line 88). The manifest fixtures use 388
    # (sidechannel) and 393 (apply_transform) intentionally — they are
    # NOT-the-same-source-of-truth markers, demonstrating that the gate
    # prefers the stacked metadata's delta when both are present. If you
    # change this 387, also recompute `expected_rate` in
    # test_wavelet_apply_gate_computes_rate_and_seg_break_even.
    return {
        "score_claim": False,
        "archive_path": "candidate.zip",
        "delta_bytes_vs_pr106_zip": 387,
        "wavelet_runtime_mode": "explicit_noop_consume_only",
        "wavelet_runtime_consumption_proof": {
            "runtime_consumed": True,
            "decoded_atom_count": 32,
            "score_claim": False,
        },
    }


def _apply_transform_manifest() -> dict:
    """Round 7 R7-3 fixture: the actual manifest shape emitted by
    `build_wavelet_apply_transform_candidate` — keyed by
    `candidate_archive_byte_delta_vs_source_estimate`, NOT
    `candidate_archive_byte_delta`. The gate's R2-2/R3-A fallback path
    reads this key when the sidechannel-shape key is absent.

    Use this fixture when testing apply_transform → gate flow. Use
    `_sidechannel_manifest()` when testing sidechannel → gate flow.
    """
    return {
        "score_claim": False,
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "b" * 64,
        "candidate_archive_byte_delta_vs_source_estimate": 393,
        "rate_score_delta_vs_source_estimate": 25.0 * 393.0 / 37_545_489.0,
        "wavelet_sidechannel_bytes": 379,
        "decoded_wavelet_sidechannel": {"total_atom_count": 32},
        "runtime_consumption_proof": {
            "runtime_consumed": True,
            "decoded_atom_count": 32,
            "score_claim": False,
        },
    }
