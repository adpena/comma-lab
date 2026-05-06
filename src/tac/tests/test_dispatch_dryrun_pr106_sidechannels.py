"""Focused tests for the PR106 sidechannel dispatch dry-run.

The tool is intentionally read-only: these tests exercise source/argparse
inspection, safe help probing, fail-closed real-mode selection, and production
artifact gating without dispatching or requiring CUDA.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.dispatch_dryrun_pr106_sidechannels import (  # noqa: E402
    ProductionInputs,
    run_dryrun,
)


def _zip_with_zero_bin(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("0.bin", b"\xffsynthetic")


def _manifest(path: Path, *, score_claim: bool = False) -> None:
    path.write_text(json.dumps({"score_claim": score_claim}, indent=2))


def test_default_dryrun_passes_and_reports_no_score_claim() -> None:
    report = run_dryrun(repo=REPO, run_help=True)

    assert report.ok
    assert report.score_claim is False
    assert report.dispatch_attempted is False
    assert report.gpu_required is False
    assert report.provider_state_free is True
    assert any(c.name == "latent:help" and c.ok for c in report.checks)
    assert any(c.name == "stacked:score-claim-marker" and c.ok for c in report.checks)


def test_missing_builder_files_fail_closed(tmp_path: Path) -> None:
    report = run_dryrun(repo=tmp_path, run_help=False)

    assert not report.ok
    missing = [c for c in report.checks if c.name.endswith(":file") and not c.ok]
    assert {c.name for c in missing} == {
        "latent:file",
        "yshift:file",
        "lrl1:file",
        "stacked:file",
    }


def test_real_yshift_or_lrl1_mode_selection_fails_closed() -> None:
    report = run_dryrun(
        repo=REPO,
        run_help=False,
        yshift_search_mode="gradient",
        lrl1_search_mode="brute_force",
    )

    assert not report.ok
    failures = {c.name: c.detail for c in report.checks if not c.ok}
    assert "yshift:selected-mode" in failures
    assert "lrl1:selected-mode" in failures
    assert "NotImplemented" in failures["yshift:selected-mode"]
    assert "NotImplemented" in failures["lrl1:selected-mode"]


def test_missing_optional_manifests_and_sisters_warn_outside_production(tmp_path: Path) -> None:
    report = run_dryrun(
        repo=REPO,
        run_help=False,
        production_readiness=False,
        production_inputs=ProductionInputs(
            latent_sister_archive=tmp_path / "missing_latent.zip",
            latent_manifest=tmp_path / "missing_latent_manifest.json",
        ),
    )

    assert report.ok
    assert any("ignored outside --production-readiness" in warning for warning in report.warnings)


def test_production_readiness_requires_manifests_and_sister_archives() -> None:
    report = run_dryrun(
        repo=REPO,
        run_help=False,
        production_readiness=True,
        production_inputs=ProductionInputs(),
    )

    assert not report.ok
    failures = {c.name for c in report.checks if not c.ok}
    assert "production:pr106-archive" in failures
    assert "production:latent-sister-archive" in failures
    assert "production:yshift-sister-archive" in failures
    assert "production:lrl1-sister-archive" in failures
    assert "latent:manifest" in failures
    assert "yshift:manifest" in failures
    assert "lrl1:manifest" in failures
    assert "stacked:manifest" in failures


def test_production_readiness_accepts_false_score_claim_manifests(tmp_path: Path) -> None:
    pr106 = tmp_path / "pr106.zip"
    latent = tmp_path / "latent.zip"
    yshift = tmp_path / "yshift.zip"
    lrl1 = tmp_path / "lrl1.zip"
    for archive in (pr106, latent, yshift, lrl1):
        _zip_with_zero_bin(archive)

    latent_manifest = tmp_path / "latent_manifest.json"
    yshift_manifest = tmp_path / "yshift_manifest.json"
    lrl1_manifest = tmp_path / "lrl1_manifest.json"
    stacked_manifest = tmp_path / "stacked_manifest.json"
    for manifest in (latent_manifest, yshift_manifest, lrl1_manifest, stacked_manifest):
        _manifest(manifest, score_claim=False)

    report = run_dryrun(
        repo=REPO,
        run_help=False,
        production_readiness=True,
        production_inputs=ProductionInputs(
            pr106_archive=pr106,
            latent_sister_archive=latent,
            yshift_sister_archive=yshift,
            lrl1_sister_archive=lrl1,
            latent_manifest=latent_manifest,
            yshift_manifest=yshift_manifest,
            lrl1_manifest=lrl1_manifest,
            stacked_manifest=stacked_manifest,
        ),
    )

    assert report.ok
    assert all(c.ok for c in report.checks if c.name.endswith(":manifest"))


def test_json_cli_is_deterministic_and_score_claim_false() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "dispatch_dryrun_pr106_sidechannels.py"),
            "--json",
            "--skip-help-subprocess",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)

    assert data["schema"] == "pr106_sidechannel_dispatch_dryrun_v1"
    assert data["ok"] is True
    assert data["score_claim"] is False
    assert data["dispatch_attempted"] is False
    assert data["gpu_required"] is False
    assert data["provider_state_free"] is True
