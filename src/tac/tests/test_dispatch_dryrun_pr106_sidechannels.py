"""Focused tests for the PR106 sidechannel dispatch dry-run.

The tool is intentionally read-only: these tests exercise source/argparse
inspection, safe help probing, fail-closed real-mode selection, and production
artifact gating without dispatching or requiring CUDA.
"""
from __future__ import annotations

import json
import struct
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


def _zip_with_payload(path: Path, payload: bytes, *, member_name: str = "0.bin") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(member_name, payload)


def _pr106_payload() -> bytes:
    return b"\xffsynthetic-pr106"


def _latent_payload(pr106_payload: bytes) -> bytes:
    return b"\xfe\x01" + struct.pack("<I", len(pr106_payload)) + pr106_payload + b"\x00\x00"


def _yshift_payload(pr106_payload: bytes) -> bytes:
    return b"\xfc" + len(pr106_payload).to_bytes(3, "little") + pr106_payload + b"\x01\x00\x00"


def _lrl1_payload(pr106_payload: bytes) -> bytes:
    return b"\xfb" + len(pr106_payload).to_bytes(3, "little") + pr106_payload + b"\x01\x00\x00"


def _wavelet_payload(pr106_payload: bytes) -> bytes:
    return b"\xfa\x01" + len(pr106_payload).to_bytes(3, "little") + pr106_payload + b"\x00\x00\x00\x00"


def _zip_with_single_member(path: Path, *, member_name: str = "x") -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(member_name, b"\xfa\x01synthetic")


def _manifest(
    path: Path,
    *,
    archive: Path | None = None,
    score_claim: bool = False,
    dispatch_blockers: bool = False,
) -> None:
    payload: dict[str, object] = {"score_claim": score_claim}
    if archive is not None:
        payload["archive_path"] = str(archive)
        payload["archive_zip_bytes"] = archive.stat().st_size
    if dispatch_blockers:
        payload["ready_for_exact_eval_dispatch"] = False
        payload["dispatch_blockers"] = ["requires_exact_cuda_auth_eval"]
    path.write_text(json.dumps(payload, indent=2))


def test_default_dryrun_passes_and_reports_no_score_claim() -> None:
    report = run_dryrun(repo=REPO, run_help=True)

    assert report.ok
    assert report.score_claim is False
    assert report.dispatch_attempted is False
    assert report.gpu_required is False
    assert report.provider_state_free is True
    assert any(c.name == "latent:help" and c.ok for c in report.checks)
    assert any(c.name == "stacked:score-claim-marker" and c.ok for c in report.checks)
    stacked_flags = next(c for c in report.checks if c.name == "stacked:argparse-flags")
    assert "--wavelet" in stacked_flags.detail


def test_missing_builder_files_fail_closed(tmp_path: Path) -> None:
    report = run_dryrun(repo=tmp_path, run_help=False)

    assert not report.ok
    missing = [c for c in report.checks if c.name.endswith(":file") and not c.ok]
    assert {c.name for c in missing} == {
        "latent:file",
        "latent_score_table:file",
        "yshift:file",
        "yshift_score_table:file",
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
    pr106_payload = _pr106_payload()
    _zip_with_payload(pr106, pr106_payload)
    _zip_with_payload(latent, _latent_payload(pr106_payload))
    _zip_with_payload(yshift, _yshift_payload(pr106_payload))
    _zip_with_payload(lrl1, _lrl1_payload(pr106_payload))

    latent_manifest = tmp_path / "latent_manifest.json"
    yshift_manifest = tmp_path / "yshift_manifest.json"
    lrl1_manifest = tmp_path / "lrl1_manifest.json"
    stacked_manifest = tmp_path / "stacked_manifest.json"
    _manifest(latent_manifest, archive=latent, score_claim=False)
    _manifest(yshift_manifest, archive=yshift, score_claim=False)
    _manifest(lrl1_manifest, archive=lrl1, score_claim=False)
    _manifest(stacked_manifest, score_claim=False, dispatch_blockers=True)

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


def test_production_readiness_accepts_optional_wavelet_candidate(tmp_path: Path) -> None:
    pr106 = tmp_path / "pr106.zip"
    latent = tmp_path / "latent.zip"
    yshift = tmp_path / "yshift.zip"
    lrl1 = tmp_path / "lrl1.zip"
    wavelet = tmp_path / "wavelet.zip"
    pr106_payload = _pr106_payload()
    _zip_with_payload(pr106, pr106_payload)
    _zip_with_payload(latent, _latent_payload(pr106_payload))
    _zip_with_payload(yshift, _yshift_payload(pr106_payload))
    _zip_with_payload(lrl1, _lrl1_payload(pr106_payload))
    _zip_with_payload(wavelet, _wavelet_payload(pr106_payload), member_name="x")

    latent_manifest = tmp_path / "latent_manifest.json"
    yshift_manifest = tmp_path / "yshift_manifest.json"
    lrl1_manifest = tmp_path / "lrl1_manifest.json"
    wavelet_manifest = tmp_path / "wavelet_manifest.json"
    stacked_manifest = tmp_path / "stacked_manifest.json"
    _manifest(latent_manifest, archive=latent, score_claim=False)
    _manifest(yshift_manifest, archive=yshift, score_claim=False)
    _manifest(lrl1_manifest, archive=lrl1, score_claim=False)
    _manifest(wavelet_manifest, archive=wavelet, score_claim=False, dispatch_blockers=True)
    _manifest(stacked_manifest, score_claim=False, dispatch_blockers=True)

    report = run_dryrun(
        repo=REPO,
        run_help=False,
        production_readiness=True,
        production_inputs=ProductionInputs(
            pr106_archive=pr106,
            latent_sister_archive=latent,
            yshift_sister_archive=yshift,
            lrl1_sister_archive=lrl1,
            wavelet_sister_archive=wavelet,
            latent_manifest=latent_manifest,
            yshift_manifest=yshift_manifest,
            lrl1_manifest=lrl1_manifest,
            wavelet_manifest=wavelet_manifest,
            stacked_manifest=stacked_manifest,
        ),
    )

    assert report.ok
    checks = {c.name: c for c in report.checks}
    assert checks["production:wavelet-sister-single-member"].ok
    assert checks["production:wavelet-embeds-pr106"].ok
    assert checks["wavelet:manifest"].ok


def test_production_readiness_rejects_sister_archive_pr106_drift(tmp_path: Path) -> None:
    pr106 = tmp_path / "pr106.zip"
    latent = tmp_path / "latent.zip"
    yshift = tmp_path / "yshift.zip"
    lrl1 = tmp_path / "lrl1.zip"
    pr106_payload = _pr106_payload()
    _zip_with_payload(pr106, pr106_payload)
    _zip_with_payload(latent, _latent_payload(b"\xffdifferent-pr106"))
    _zip_with_payload(yshift, _yshift_payload(pr106_payload))
    _zip_with_payload(lrl1, _lrl1_payload(pr106_payload))

    latent_manifest = tmp_path / "latent_manifest.json"
    yshift_manifest = tmp_path / "yshift_manifest.json"
    lrl1_manifest = tmp_path / "lrl1_manifest.json"
    stacked_manifest = tmp_path / "stacked_manifest.json"
    _manifest(latent_manifest, archive=latent, score_claim=False)
    _manifest(yshift_manifest, archive=yshift, score_claim=False)
    _manifest(lrl1_manifest, archive=lrl1, score_claim=False)
    _manifest(stacked_manifest, score_claim=False, dispatch_blockers=True)

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

    assert not report.ok
    failures = {c.name: c.detail for c in report.checks if not c.ok}
    assert "production:latent-embeds-pr106" in failures
    assert "not anchored to the selected PR106 payload" in failures["production:latent-embeds-pr106"]


def test_production_readiness_rejects_manifest_archive_byte_drift(tmp_path: Path) -> None:
    pr106 = tmp_path / "pr106.zip"
    latent = tmp_path / "latent.zip"
    yshift = tmp_path / "yshift.zip"
    lrl1 = tmp_path / "lrl1.zip"
    pr106_payload = _pr106_payload()
    _zip_with_payload(pr106, pr106_payload)
    _zip_with_payload(latent, _latent_payload(pr106_payload))
    _zip_with_payload(yshift, _yshift_payload(pr106_payload))
    _zip_with_payload(lrl1, _lrl1_payload(pr106_payload))

    latent_manifest = tmp_path / "latent_manifest.json"
    yshift_manifest = tmp_path / "yshift_manifest.json"
    lrl1_manifest = tmp_path / "lrl1_manifest.json"
    stacked_manifest = tmp_path / "stacked_manifest.json"
    _manifest(latent_manifest, archive=latent, score_claim=False)
    data = json.loads(latent_manifest.read_text())
    data["archive_zip_bytes"] += 1
    latent_manifest.write_text(json.dumps(data))
    _manifest(yshift_manifest, archive=yshift, score_claim=False)
    _manifest(lrl1_manifest, archive=lrl1, score_claim=False)
    _manifest(stacked_manifest, score_claim=False, dispatch_blockers=True)

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

    assert not report.ok
    failures = {c.name: c.detail for c in report.checks if not c.ok}
    assert "latent:manifest" in failures
    assert "archive byte count does not match" in failures["latent:manifest"]


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
