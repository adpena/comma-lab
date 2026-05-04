from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "pre_submission_compliance_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pre_submission_compliance_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_submission(
    root: Path,
    *,
    unsafe_zip_member: bool = False,
    include_runtime_tree: bool = True,
    runtime_file_mismatch: bool = False,
    top_level_runtime_manifest_only: bool = False,
) -> dict[str, object]:
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        name = "../x" if unsafe_zip_member else "x"
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        zf.writestr(info, b"payload")
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    inflate.chmod(0o755)
    archive_sha = module_sha256(archive)
    archive_bytes = archive.stat().st_size
    with zipfile.ZipFile(archive) as zf:
        members = []
        for info in zf.infolist():
            members.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "sha256": module_bytes_sha256(zf.read(info.filename)),
                }
            )
    seg = 0.00057185
    pose = 0.0001894
    score = 100 * seg + (10 * pose) ** 0.5 + 25 * archive_bytes / 37_545_489
    (root / "report.txt").write_text(
        (
            "report\n"
            f"archive_sha256: {archive_sha}\n"
            f"archive_size_bytes: {archive_bytes}\n"
            f"score_recomputed_from_components: {score}\n"
        ),
        encoding="utf-8",
    )
    (root / "archive_manifest.json").write_text(
        json.dumps(
            {
                "schema": "unit_test_archive_manifest_v1",
                "archive": {
                    "sha256": archive_sha,
                    "size_bytes": archive_bytes,
                    "members": members,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "cuda_available": True,
        "device": "cuda",
        "gpu_t4_match": True,
        "tool": "experiments/contest_auth_eval.py",
    }
    runtime_manifest = None
    if include_runtime_tree:
        runtime_sha = module_sha256(inflate)
        runtime_bytes = inflate.stat().st_size
        runtime_manifest = {
            "runtime_tree_sha256": "a" * 64,
            "files": [
                {
                    "relative_path": "inflate.sh",
                    "bytes": runtime_bytes + (1 if runtime_file_mismatch else 0),
                    "sha256": runtime_sha,
                }
            ],
        }
        if not top_level_runtime_manifest_only:
            provenance["inflate_runtime_manifest"] = runtime_manifest
    auth = {
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_size_bytes": archive_bytes,
        "n_samples": 600,
        "score_recomputed_from_components": score,
        "provenance": provenance,
    }
    if runtime_manifest is not None and top_level_runtime_manifest_only:
        auth["inflate_runtime_manifest"] = runtime_manifest
    (root / "contest_auth_eval.json").write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
    return {"archive_sha256": archive_sha, "archive_size_bytes": archive_bytes}


def module_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_bytes_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _rewrite_auth_archive_identity(auth_path: Path, archive_path: Path) -> None:
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    archive_sha = module_sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    payload["archive_size_bytes"] = archive_bytes
    payload["score_recomputed_from_components"] = (
        100 * payload["avg_segnet_dist"]
        + (10 * payload["avg_posenet_dist"]) ** 0.5
        + 25 * archive_bytes / 37_545_489
    )
    payload["provenance"]["archive_sha256"] = archive_sha
    payload["provenance"]["archive_size_bytes"] = archive_bytes
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rewrite_archive_manifest_identity(manifest_path: Path, archive_path: Path) -> None:
    archive_sha = module_sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    members = []
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            members.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "sha256": module_bytes_sha256(zf.read(info.filename)),
                }
            )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "unit_test_archive_manifest_v1",
                "archive": {"sha256": archive_sha, "size_bytes": archive_bytes, "members": members},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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
                "--require-auth-eval",
                "--require-t4-equivalent",
                "--expect-single-member",
                "x",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
                "--expected-runtime-tree-sha256",
                "a" * 64,
            ]
        )
    )

    assert report["status"] == "passed"
    assert report["score_claim"] is False
    assert report["provider_agnostic"] is True
    assert not report["failed_checks"]


def test_pre_submission_check_contest_final_implies_strict_gates(tmp_path: Path) -> None:
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
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["status"] == "passed"
    assert report["expectations"]["contest_final"] is True
    assert report["expectations"]["require_auth_eval"] is True
    assert report["expectations"]["require_t4_equivalent"] is True
    assert report["expectations"]["expect_single_member"] == "x"
    assert checks["archive_expected_single_member"]["passed"] is True
    assert checks["auth_eval_t4_equivalent"]["passed"] is True
    assert checks["archive_manifest_sha256_matches_archive"]["passed"] is True
    assert checks["report_links_exact_archive_sha256"]["passed"] is True
    assert checks["public_release_hygiene"]["passed"] is True


def test_pre_submission_check_contest_final_requires_expected_archive_identity(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "contest_final_expected_archive_sha256_present" in report["failed_checks"]
    assert "contest_final_expected_archive_size_bytes_present" in report["failed_checks"]


def test_pre_submission_check_contest_final_requires_submission_runtime_match(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", runtime_file_mismatch=True)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "failed"
    assert "auth_eval_runtime_file_matches_submission:inflate.sh" in report["failed_checks"]


def test_pre_submission_check_runtime_match_flag_fails_without_contest_final(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", runtime_file_mismatch=True)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
                "--require-submission-runtime-match",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "failed"
    assert report["expectations"]["require_submission_runtime_match"] is True
    assert "auth_eval_runtime_file_matches_submission:inflate.sh" in report["failed_checks"]


def test_pre_submission_check_runtime_match_accepts_top_level_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", top_level_runtime_manifest_only=True)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
                "--require-submission-runtime-match",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "passed"
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["auth_eval_runtime_file_manifest_present"]["passed"] is True
    assert checks["auth_eval_runtime_file_matches_submission:inflate.sh"]["passed"] is True


def test_pre_submission_check_rejects_malformed_expected_hash(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--expected-archive-sha256",
                "not-a-sha",
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "failed"
    assert "expected_archive_sha256_format" in report["failed_checks"]


def test_pre_submission_check_allows_adjudicator_display_contribution_rounding(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["score_seg_contribution"] = 100 * payload["avg_segnet_dist"]
    payload["score_pose_contribution"] = (10 * payload["avg_posenet_dist"]) ** 0.5
    payload["score_rate_contribution"] = round(25 * payload["archive_size_bytes"] / 37_545_489, 6)
    payload["score_recomputed_from_components"] = (
        payload["score_seg_contribution"]
        + payload["score_pose_contribution"]
        + payload["score_rate_contribution"]
    )
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--require-auth-eval",
                "--require-t4-equivalent",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "passed"
    assert "auth_eval_score_contributions_match_components" not in report["failed_checks"]


def test_pre_submission_check_fails_zip_slip_member(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission", unsafe_zip_member=True)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert any(name.startswith("archive_member_safe") for name in report["failed_checks"])


def test_pre_submission_check_fails_hidden_zip_member(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
        zf.writestr(".env", b"SECRET=1\n")
    _rewrite_auth_archive_identity(tmp_path / "submission" / "contest_auth_eval.json", archive)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "archive_member_safe:.env" in report["failed_checks"]


def test_pre_submission_check_fails_hidden_directory_even_with_single_member(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(".cache/", b"")
        zf.writestr("x", b"payload")
    _rewrite_auth_archive_identity(tmp_path / "submission" / "contest_auth_eval.json", archive)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--expect-single-member",
                "x",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "archive_member_safe:.cache/" in report["failed_checks"]
    assert "archive_expected_single_member" in report["failed_checks"]


def test_pre_submission_check_contest_final_rejects_stale_archive_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    manifest_path = tmp_path / "submission" / "archive_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["archive"]["sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "failed"
    assert "archive_manifest_sha256_matches_archive" in report["failed_checks"]


def test_pre_submission_check_contest_final_requires_report_archive_link(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "report.txt").write_text("report without custody\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["status"] == "failed"
    assert "report_links_exact_archive_sha256" in report["failed_checks"]
    assert "report_links_exact_archive_size_bytes" in report["failed_checks"]


def test_pre_submission_check_fails_multiple_packed_payload_containers(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"payload-a")
        zf.writestr("renderer_payload.bin", b"payload-b")
    _rewrite_auth_archive_identity(tmp_path / "submission" / "contest_auth_eval.json", archive)
    _rewrite_archive_manifest_identity(tmp_path / "submission" / "archive_manifest.json", archive)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--archive-manifest-json",
                str(tmp_path / "submission" / "archive_manifest.json"),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "archive_packed_payload_singleton" in report["failed_checks"]


def test_pre_submission_check_dispatch_claim_linkage_requires_terminal_row(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "active_lane_dispatch_claims.md"
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
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
                "--archive-manifest-json",
                str(tmp_path / "submission" / "archive_manifest.json"),
                "--expected-archive-sha256",
                str(expected["archive_sha256"]),
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

    assert report["status"] == "failed"
    assert "dispatch_claim_terminal_row_present" in report["failed_checks"]


def test_pre_submission_check_public_hygiene_flags_modal_provider_ids(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    public_doc = tmp_path / "submission" / "supplement.md"
    public_doc.write_text(
        "Modal call fc-01KQS22WSZ7YR3ZJYXVPPYE4VB app ap-KoGUy9mB8TVViZbp6BIoJX\n",
        encoding="utf-8",
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--public-scan-path",
                str(public_doc),
            ]
        )
    )

    assert report["status"] == "failed"
    assert "public_release_hygiene" in report["failed_checks"]
    assert len(report["public_hygiene"]["violations"]) == 2


def test_release_docs_use_pre_submission_source_prs_argparse_flag() -> None:
    """Guard release docs against stale, invented pre-submission flags."""
    stale_flag = "--public" + "-pr-refs"
    doc_paths = [
        REPO / "docs" / "submission_template.md",
        REPO / "reports" / "writeup_working.md",
        REPO / "reports" / "graphs" / "final_writeup_draft.md",
    ]

    stale_hits = [
        str(path.relative_to(REPO))
        for path in doc_paths
        if stale_flag in path.read_text(encoding="utf-8")
    ]

    assert stale_hits == []
    assert "--source-prs" in SCRIPT.read_text(encoding="utf-8")


def test_pre_submission_check_fails_archive_override_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    _write_submission(tmp_path / "other")

    with zipfile.ZipFile(tmp_path / "other" / "archive.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"different-payload")
    _rewrite_auth_archive_identity(tmp_path / "other" / "contest_auth_eval.json", tmp_path / "other" / "archive.zip")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--archive",
                str(tmp_path / "other" / "archive.zip"),
                "--auth-eval-json",
                str(tmp_path / "other" / "contest_auth_eval.json"),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "submission_archive_matches_inspected_archive" in report["failed_checks"]


def test_pre_submission_check_fails_auth_eval_archive_identity_conflict(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["archive_sha256"] = "b" * 64
    payload["archive_size_bytes"] = payload["archive_size_bytes"] + 1
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "auth_eval_archive_sha256_fields_consistent" in report["failed_checks"]
    assert "auth_eval_archive_size_bytes_fields_consistent" in report["failed_checks"]


def test_pre_submission_check_fails_auth_eval_runtime_tree_conflict(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["inflate_runtime_manifest"] = {"runtime_tree_sha256": "b" * 64}
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "auth_eval_runtime_tree_fields_consistent" in report["failed_checks"]


def test_pre_submission_check_fails_auth_eval_without_runtime_tree(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission", include_runtime_tree=False)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--require-auth-eval",
            ]
        )
    )

    assert report["status"] == "failed"
    assert "auth_eval_runtime_tree_recorded" in report["failed_checks"]


def test_pre_submission_check_detects_non_executable_inflate(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    os.chmod(tmp_path / "submission" / "inflate.sh", 0o644)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(["--submission-dir", str(tmp_path / "submission")])
    )

    assert report["status"] == "failed"
    assert "inflate_sh_executable" in report["failed_checks"]
