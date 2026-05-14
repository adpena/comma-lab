# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tac.paradigm_alpha_readiness import (
    AUDIT_NAME,
    DISPATCH_BLOCKER_AUDIT_ONLY,
    audit_paradigm_alpha,
    readiness_payload,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "audit_paradigm_alpha_readiness.py"


def test_paradigm_alpha_readiness_current_repo_identifies_on_disk_components() -> None:
    rows = {row.key: row for row in audit_paradigm_alpha(repo_root=REPO_ROOT)}

    assert set(rows) == {
        "alpha1_nerv",
        "alpha2_wavelet",
        "alpha3_vqvae",
        "alpha4_grayscale_lut",
    }
    assert "codec_module" in rows["alpha2_wavelet"].components_on_disk
    assert "tests" in rows["alpha2_wavelet"].components_on_disk
    assert "codec_module" in rows["alpha3_vqvae"].components_on_disk
    assert "tests" in rows["alpha3_vqvae"].components_on_disk
    assert rows["alpha1_nerv"].ready_for_exact_eval_dispatch is False
    assert DISPATCH_BLOCKER_AUDIT_ONLY in rows["alpha1_nerv"].dispatch_blockers


def test_paradigm_alpha_readiness_current_repo_keeps_runtime_blockers_explicit() -> None:
    rows = {row.key: row for row in audit_paradigm_alpha(repo_root=REPO_ROOT)}

    assert "runtime_missing_wmc1_loader" in rows["alpha2_wavelet"].local_blockers
    assert "runtime_missing_vqm1_loader" in rows["alpha3_vqvae"].local_blockers
    assert "missing_lane12_l2_clearance_packet" in rows["alpha1_nerv"].local_blockers
    assert rows["alpha4_grayscale_lut"].ready_for_exact_eval_dispatch is False


def test_paradigm_alpha_readiness_fake_repo_reports_byte_negative_wavelet(tmp_path: Path) -> None:
    _write(tmp_path / "src/tac/wavelet_mask_codec.py", "WMC codec\n")
    _write(tmp_path / "src/tac/tests/test_wavelet_mask_codec.py", "def test_x(): pass\n")
    _write(tmp_path / "experiments/paradigm_alpha_real_archive_eval.py", "alpha2\n")
    _write(tmp_path / "submissions/robust_current/inflate_renderer.py", "# runtime hook absent\n")
    _write(
        tmp_path / "reports/paradigm_alpha_real_archive.json",
        json.dumps(
            {
                "baseline_av1_bytes": 100,
                "candidates": {
                    "alpha2_wavelet": {
                        "encoded_bytes": 150,
                        "argmax_agreement_vs_source": 1.0,
                        "argmax_disagreement_vs_source": 0.0,
                        "pct_savings_vs_av1": -50.0,
                    }
                },
            }
        ),
    )

    row = next(row for row in audit_paradigm_alpha(repo_root=tmp_path) if row.key == "alpha2_wavelet")

    assert row.missing_core_components == ()
    assert "runtime_missing_wmc1_loader" in row.local_blockers
    assert "empirical_bytes_not_better_than_baseline_alpha2_wavelet" in row.local_blockers
    assert row.readiness_status == "blocked_missing_runtime_integration"


def test_paradigm_alpha_readiness_payload_and_markdown_are_stable() -> None:
    payload = readiness_payload(repo_root=REPO_ROOT)
    markdown = render_markdown(audit_paradigm_alpha(repo_root=REPO_ROOT))

    assert payload["audit"] == AUDIT_NAME
    assert payload["schema_version"] == 1
    assert payload["summary"]["candidate_count"] == 4
    assert payload["summary"]["ready_for_exact_eval_dispatch_count"] == 0
    assert markdown.startswith("# Paradigm-Alpha Mask Readiness\n")
    assert "| `alpha2_wavelet` |" in markdown


def test_audit_paradigm_alpha_readiness_cli_json_output() -> None:
    proc = _run_cli("--format", "json")

    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["audit"] == AUDIT_NAME
    assert [entry["key"] for entry in payload["entries"]] == [
        "alpha1_nerv",
        "alpha2_wavelet",
        "alpha3_vqvae",
        "alpha4_grayscale_lut",
    ]
    assert all(entry["ready_for_exact_eval_dispatch"] is False for entry in payload["entries"])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
