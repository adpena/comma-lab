from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.precompute_gradient_corrections import pack_sparse_corrections
from tac.engineered_correction_readiness import audit_sparse_corrections

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
