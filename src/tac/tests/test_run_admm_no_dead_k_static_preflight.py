from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "run_admm_no_dead_k_static_preflight.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("run_admm_no_dead_k_static_preflight", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_admm_no_dead_k_static_preflight"] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(tmp_path: Path, *, archive_sha_override: str | None = None) -> Path:
    root = tmp_path / "candidate"
    submission_dir = root / "submission_dir"
    submission_dir.mkdir(parents=True)
    (submission_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n",
        encoding="utf-8",
    )
    os.chmod(submission_dir / "inflate.sh", 0o755)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
    archive_sha = archive_sha_override or _sha(archive)
    manifest = {
        "schema_version": "admm_x_lossy_coarsening_path_b_step6_no_dead_k_build.v1",
        "lane_id": "admm_x_lossy_coarsening_path_b_step6_no_dead_k",
        "archive_relpath": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "submission_dir_relpath": submission_dir.as_posix(),
        "evidence_grade": "[CPU-build]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "section_K_bytes_in_wire_format": 0,
        "smoke_n_latent_pairs_decoded": 600,
        "dispatch_blockers": sorted(helper.REQUIRED_CPU_BUILD_BLOCKERS),
        "score_claim_blockers": sorted(helper.REQUIRED_CPU_BUILD_BLOCKERS),
    }
    manifest_path = root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def test_static_preflight_fails_closed_on_runtime_cache_leak(tmp_path: Path) -> None:
    manifest_path = _write_artifact(tmp_path)
    submission_dir = manifest_path.parent / "submission_dir"
    (submission_dir / "__pycache__").mkdir()
    (submission_dir / "__pycache__" / "inflate.cpython-312.pyc").write_bytes(b"cache")

    payload = helper.build_static_preflight(manifest_path)

    assert payload["static_archive_runtime_closure_passed"] is False
    assert "submission_runtime_python_caches_absent" in payload["static_blockers"]
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False


def test_static_preflight_verifies_archive_runtime_closure_without_score_claim(
    tmp_path: Path,
) -> None:
    manifest_path = _write_artifact(tmp_path)

    payload = helper.build_static_preflight(manifest_path)

    assert payload["static_archive_runtime_closure_passed"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["archive_mutation_attempted"] is False
    assert payload["omx_state_touched"] is False
    assert payload["static_blockers"] == []
    assert payload["bash_syntax"]["returncode"] == 0
    assert payload["pre_submission_static"]["archive"]["members"][0]["name"] == "x"
    assert "runtime_tree_sha256" in payload["pre_submission_static"]["submission_runtime"]
    assert "release_packet_archive_missing_inside_submission_dir" in payload["readiness_blockers"]
    assert "contest_auth_eval_json_missing" in payload["readiness_blockers"]
    assert "apogee_int6_contest_cuda_anchor_required_first" in payload["readiness_blockers"]


def test_static_preflight_fails_closed_on_archive_sha_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_artifact(tmp_path, archive_sha_override="b" * 64)

    payload = helper.build_static_preflight(manifest_path)

    assert payload["static_archive_runtime_closure_passed"] is False
    assert "build_manifest_archive_sha_matches_file" in payload["static_blockers"]
    assert (
        "pre_submission_static_unexpected_failure:expected_archive_sha256_matches"
        in payload["static_blockers"]
    )
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["score_claim"] is False


def test_static_preflight_cli_writes_json_and_fails_only_on_static_break(
    tmp_path: Path,
) -> None:
    good_manifest = _write_artifact(tmp_path / "good")
    bad_manifest = _write_artifact(tmp_path / "bad", archive_sha_override="c" * 64)
    out = tmp_path / "static_preflight.json"

    rc = helper.main(
        [
            "--build-manifest",
            str(good_manifest),
            "--json-out",
            str(out),
            "--fail-if-static-closure-broken",
        ]
    )
    fail_rc = helper.main(
        [
            "--build-manifest",
            str(bad_manifest),
            "--fail-if-static-closure-broken",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert fail_rc == 1
    assert payload["schema"] == "admm_no_dead_k_static_preflight_v1"
    assert payload["static_archive_runtime_closure_passed"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
