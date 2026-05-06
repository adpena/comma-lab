from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from tac.foveation_readiness import audit_foveation_params
from tac.hyperbolic_foveation import HyperbolicFoveation, save_foveation_params
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_audit_foveation_params_records_charged_custody_without_dispatch(tmp_path: Path) -> None:
    params = tmp_path / "foveation_params.bin"
    hf = HyperbolicFoveation((384, 512), n_frames=3, init_alpha=0.0, init_R=40.0)
    save_foveation_params(hf, params)

    manifest = audit_foveation_params(
        params,
        repo_root=REPO,
        expected_frames=3,
        expected_image_size=(384, 512),
        source_archive_sha256="a" * 64,
    )

    assert manifest["ok"] is True
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["member"] == "foveation_params.bin"
    assert manifest["bytes"] == params.stat().st_size
    assert manifest["sha256"] == sha256_file(params)
    assert manifest["wire_format"] == "HFV1"
    assert manifest["n_frames"] == 3
    assert manifest["image_size"] == {"height": 384, "width": 512}
    assert manifest["geometry"]["identity_like"] is True
    assert manifest["runtime_contract"]["runtime_consumer_required"] is True
    assert "foveation_runtime_consumer_not_proven" in manifest["dispatch_blockers"]
    assert "exact_cuda_auth_eval_required_before_score_claim" in manifest["dispatch_blockers"]


def test_audit_foveation_params_blocks_bad_geometry(tmp_path: Path) -> None:
    params = tmp_path / "bad_foveation_params.bin"
    hf = HyperbolicFoveation((32, 32), n_frames=2, init_alpha=0.2, init_R=10.0)
    with torch.no_grad():
        hf.R[1] = 0.0
        hf.p[0] = -1.0
        hf.o[1] = torch.tensor([999.0, -3.0])
    save_foveation_params(hf, params)

    manifest = audit_foveation_params(
        params,
        repo_root=REPO,
        expected_frames=3,
        expected_image_size=(32, 32),
    )

    blockers = set(manifest["dispatch_blockers"])
    assert manifest["ok"] is False
    assert "foveation_radius_nonpositive" in blockers
    assert "foveation_power_negative" in blockers
    assert "foveation_origin_outside_image" in blockers
    assert "foveation_frame_count_mismatch" in blockers


def test_audit_hyperbolic_foveation_readiness_cli_records_tool_manifest(tmp_path: Path) -> None:
    params = tmp_path / "foveation_params.bin"
    out = tmp_path / "readiness.json"
    hf = HyperbolicFoveation((64, 96), n_frames=4, init_alpha=0.1, init_R=20.0)
    save_foveation_params(hf, params)

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_hyperbolic_foveation_readiness.py"),
            "--foveation-params-bin",
            str(params),
            "--expected-frames",
            "4",
            "--expected-image-size",
            "64",
            "96",
            "--source-archive-sha256",
            "b" * 64,
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    tool_run = manifest["tool_run_manifest"]
    assert manifest["source_archive_sha256"] == "b" * 64
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert tool_run["tool"] == "tools/audit_hyperbolic_foveation_readiness.py"
    assert tool_run["input_files"] == [
        {
            "path": params.as_posix(),
            "bytes": params.stat().st_size,
            "sha256": sha256_file(params),
        }
    ]
    assert tool_run["dispatch_attempted"] is False


def test_audit_foveation_params_reports_load_error_for_corrupt_payload(tmp_path: Path) -> None:
    params = tmp_path / "corrupt.bin"
    params.write_bytes(b"HFV1" + b"\x00" * 8)

    manifest = audit_foveation_params(params, repo_root=REPO)

    assert manifest["ok"] is False
    assert "load_error" in manifest
    assert "foveation_payload_load_failed" in manifest["dispatch_blockers"]
