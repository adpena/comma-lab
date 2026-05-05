from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "pre_submission_compliance_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pre_submission_compliance_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_submission(root: Path, *, device: str = "cuda", t4: bool = True, runtime_tree: str | None = "a" * 64) -> dict:
    mod = _load_module()
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    inflate.chmod(0o755)
    archive_sha = _sha(archive)
    archive_bytes = archive.stat().st_size
    seg = 0.00057185
    pose = 0.0001894
    score = 100 * seg + (10 * pose) ** 0.5 + 25 * archive_bytes / 37_545_489
    auth = {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "archive_size_bytes": archive_bytes,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "n_samples": 600,
        "promotion_eligible": device == "cuda" and t4,
        "score_claim_valid": device == "cuda" and t4,
        "evidence_grade": "A++" if device == "cuda" and t4 else "invalid",
        "provenance": {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "device": device,
            "gpu_t4_match": t4,
        },
    }
    if runtime_tree:
        auth["inflate_runtime_manifest"] = {"runtime_tree_sha256": runtime_tree}
    (root / "contest_auth_eval.json").write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive) as zf:
        members = [
            {"name": info.filename, "file_size": info.file_size, "sha256": mod._bytes_sha256(zf.read(info))}
            for info in zf.infolist()
        ]
    (root / "archive_manifest.json").write_text(
        json.dumps({"archive": {"sha256": archive_sha, "size_bytes": archive_bytes, "members": members}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "report.txt").write_text(
        f"archive_sha256: {archive_sha}\narchive_size_bytes: {archive_bytes}\nscore: {score}\n",
        encoding="utf-8",
    )
    return {"archive_sha256": archive_sha, "archive_size_bytes": archive_bytes, "runtime_tree": runtime_tree}


def _failed_check_names(report: dict) -> set[str]:
    return {check["name"] for check in report["checks"] if not check["passed"]}


def test_pre_submission_check_passes_strict_happy_path(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--expect-single-member",
                "x",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
                "--expected-runtime-tree-sha256",
                expected["runtime_tree"],
            ]
        )
    )
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_pre_submission_check_fails_zip_slip_member(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("../x", b"payload")
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            ["--submission-dir", str(tmp_path / "submission"), "--require-auth-eval"]
        )
    )
    assert not report["passed"]
    assert any(check["name"].startswith("zip_member_safe") for check in report["checks"] if not check["passed"])


def test_pre_submission_check_rejects_cpu_auth_eval_for_promotion(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission", device="cpu", t4=False)
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            ["--submission-dir", str(tmp_path / "submission"), "--require-auth-eval", "--require-t4-equivalent"]
        )
    )
    assert not report["passed"]
    failed = _failed_check_names(report)
    assert "auth_eval_t4_equivalent" in failed
    assert "auth_eval_promotable_stamp" in failed


def test_pre_submission_check_contest_final_rejects_stale_archive_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    manifest = tmp_path / "submission" / "archive_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["archive"]["sha256"] = "b" * 64
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--contest-final",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )
    assert not report["passed"]
    assert "archive_manifest_sha_matches" in _failed_check_names(report)


def test_pre_submission_check_dispatch_claim_linkage_requires_terminal_row(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| ts | lane_id | platform | instance/job_id | status | notes |\n"
        "| 2026-05-04T00:00:00Z | lane-a | lightning | job-a | running | active |\n",
        encoding="utf-8",
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--require-auth-eval",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
                "--dispatch-claims-md",
                str(claims),
                "--expected-lane-id",
                "lane-a",
                "--expected-job-id",
                "job-a",
            ]
        )
    )
    assert not report["passed"]
    assert "dispatch_claim_terminal_row" in _failed_check_names(report)


def test_pre_submission_check_dispatch_claim_linkage_accepts_live_eight_column_schema(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T15:05:13Z | codex | lane-a | lightning | job-a | 2026-05-04T15:05Z | completed_score=0.209 | A++ |\n",
        encoding="utf-8",
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--require-auth-eval",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
                "--dispatch-claims-md",
                str(claims),
                "--expected-lane-id",
                "lane-a",
                "--expected-job-id",
                "job-a",
            ]
        )
    )
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_pre_submission_check_public_hygiene_flags_provider_ids(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    public_doc = tmp_path / "submission" / "supplement.md"
    public_doc.write_text("Modal call fc-01KQS22WSZ7YR3ZJYXVPPYE4VB\n", encoding="utf-8")
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            ["--submission-dir", str(tmp_path / "submission"), "--public-scan-path", str(public_doc)]
        )
    )
    assert not report["passed"]
    assert "public_scan_has_no_private_surface" in _failed_check_names(report)


def test_pre_submission_check_public_hygiene_recurses_directories(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    public_dir = tmp_path / "public_site"
    nested = public_dir / "assets" / "index.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("debug path /Users/adpena/Projects/pact/private.json\n", encoding="utf-8")
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            ["--submission-dir", str(tmp_path / "submission"), "--public-scan-path", str(public_dir)]
        )
    )
    assert not report["passed"]
    assert "public_scan_has_no_private_surface" in _failed_check_names(report)
    assert any("assets/index.md" in hit for hit in report["public_hygiene"]["hits"])


def test_pre_submission_check_detects_non_executable_inflate(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    os.chmod(tmp_path / "submission" / "inflate.sh", 0o644)
    report = mod.build_report(mod.build_arg_parser().parse_args(["--submission-dir", str(tmp_path / "submission")]))
    assert not report["passed"]
    assert "inflate_sh_executable" in _failed_check_names(report)
