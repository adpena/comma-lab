from __future__ import annotations

import importlib.util
import json
import os
import struct
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


def _write_submission(
    root: Path,
    *,
    device: str = "cuda",
    t4: bool = True,
    runtime_tree: str | None = None,
) -> dict:
    mod = _load_module()
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    inflate.chmod(0o755)
    if runtime_tree is None:
        runtime_tree = mod._submission_runtime_manifest(root)["runtime_tree_sha256"]
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


def _write_terminal_claim(path: Path, *, lane_id: str = "lane-a", job_id: str = "job-a") -> None:
    row = (
        f"| 2026-05-08T00:00:00Z | codex | {lane_id} | lightning | {job_id} | "
        "2026-05-08T00:00Z | completed_score=0.209 | A++ |\n"
    )
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + row,
        encoding="utf-8",
    )


def test_pre_submission_check_passes_strict_happy_path(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(claims)
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
    strict_formula = report["auth_eval"]["strict_formula"]
    assert strict_formula["basis"] == "auth_eval_report_components_plus_exact_archive_bytes"
    assert strict_formula["score"] == strict_formula["report_reconstructed_score"]
    assert report["auth_eval"]["anchor_proof"]["schema"] == (
        "pre_submission_compliance_anchor_proof_v1"
    )
    assert report["auth_eval"]["anchor_proof"]["score_basis"] == strict_formula
    assert report["submission_runtime"]["runtime_tree_sha256"] == expected["runtime_tree"]


def test_pre_submission_check_contest_final_rejects_runtime_tree_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", runtime_tree="b" * 64)

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert not report["passed"]
    assert "submission_runtime_tree_matches_auth_eval" in _failed_check_names(report)


def test_pre_submission_check_contest_final_rejects_inferred_promotion_stamp(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth.pop("promotion_eligible")
    auth.pop("score_claim_valid")
    auth.pop("evidence_grade")
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--contest-final",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert not report["passed"]
    assert "auth_eval_explicit_promotable_stamp" in _failed_check_names(report)


def test_pre_submission_check_records_strict_formula_when_report_score_uses_rounded_rate(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(claims)
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    strict_score = auth["score_recomputed_from_components"]
    rounded_rate_score = (
        100 * auth["avg_segnet_dist"]
        + (10 * auth["avg_posenet_dist"]) ** 0.5
        + 25 * round(auth["archive_size_bytes"] / 37_545_489, 8)
    )
    auth["canonical_score"] = rounded_rate_score
    auth["score_recomputed_from_components"] = rounded_rate_score
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--contest-final",
                "--expect-single-member",
                "x",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
                "--expected-runtime-tree-sha256",
                expected["runtime_tree"],
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
    strict_formula = report["auth_eval"]["strict_formula"]
    assert strict_formula["score"] == strict_score
    assert strict_formula["report_reconstructed_score"] == rounded_rate_score
    assert strict_formula["score_delta_vs_report_reconstruction"] == (
        strict_score - rounded_rate_score
    )
    assert report["auth_eval"]["anchor_proof"]["score_basis"]["score"] == strict_score


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


def test_pre_submission_check_rejects_local_central_method_skew(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    data = bytearray(archive.read_bytes())
    struct.pack_into("<H", data, 8, zipfile.ZIP_DEFLATED)
    archive.write_bytes(data)

    _record, checks = mod.inspect_archive(archive, expect_single_member="x")

    failed = {check.name for check in checks if not check.passed}
    assert "zip_local_header_metadata_matches:x" in failed


def test_pre_submission_check_reports_local_central_name_skew_directly(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    data = bytearray(archive.read_bytes())
    local_name_offset = 30
    assert data[local_name_offset:local_name_offset + 1] == b"x"
    data[local_name_offset:local_name_offset + 1] = b"y"
    archive.write_bytes(data)

    _record, checks = mod.inspect_archive(archive, expect_single_member="x")

    failed = {check.name for check in checks if not check.passed}
    assert "zip_local_header_matches:x" in failed
    assert "zip_local_header_metadata_matches:x" in failed
    assert "zip_member_payload_readable:x" in failed


def test_pre_submission_check_contest_final_requires_dispatch_identity(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")

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
                "--expected-runtime-tree-sha256",
                expected["runtime_tree"],
            ]
        )
    )

    assert not report["passed"]
    failed = _failed_check_names(report)
    assert "contest_final_expected_lane_id_supplied" in failed
    assert "contest_final_expected_job_id_supplied" in failed


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


def test_pre_submission_check_accepts_candidate_archive_manifest_identity(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    manifest = tmp_path / "wr01_candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": expected["archive_sha256"],
                "candidate_archive_bytes": expected["archive_size_bytes"],
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--archive-manifest-json",
                str(manifest),
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_pre_submission_check_marks_optional_missing_custody_as_warning(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "contest_auth_eval.json").unlink()
    (tmp_path / "submission" / "archive_manifest.json").unlink()

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            ["--submission-dir", str(tmp_path / "submission")]
        )
    )

    assert report["passed"], [c for c in report["checks"] if c["severity"] == "error" and not c["passed"]]
    warnings = {
        check["name"]: check
        for check in report["checks"]
        if check["severity"] == "warning" and not check["passed"]
    }
    assert "auth_eval_optional_missing" in warnings
    assert "archive_manifest_optional_missing" in warnings


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


def test_pre_submission_check_dispatch_claim_linkage_uses_newest_matching_row(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-08T01:00:00Z | codex | lane-a | lightning | job-a | "
        "2026-05-08T02:00Z | active_exact_eval | newest row still active |\n"
        "| 2026-05-08T00:00:00Z | codex | lane-a | lightning | job-a | "
        "2026-05-08T00:30Z | completed_score=0.209 | stale older terminal |\n",
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
    assert report["dispatch_claims"]["latest_matching_status"] == "active_exact_eval"


def test_pre_submission_check_dispatch_claim_linkage_accepts_live_eight_column_schema(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T15:05:13Z | codex | lane-a | lightning | job-a | "
        "2026-05-04T15:05Z | completed_score=0.209 | A++ |\n",
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


def test_pre_submission_check_dispatch_claim_linkage_accepts_claim_helper_terminal_statuses(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T15:05:13Z | codex | lane-a | lightning | job-a | "
        "2026-05-04T15:05Z | cancelled_operator_request | terminal per claim helper |\n",
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
