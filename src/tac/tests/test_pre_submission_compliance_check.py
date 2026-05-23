# SPDX-License-Identifier: MIT
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
PINNED_GIT_SHA = "a" * 40


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
    policy_statement: str | None = (
        "Competitive: this packet is intended to beat the top #1 leaderboard "
        "submission on the same contest-final evidence axis, with archive "
        "bytes, runtime custody, and exact component recomputation linked here."
    ),
) -> dict:
    mod = _load_module()
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"payload")
    inflate = root / "inflate.sh"
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "ARCHIVE_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "while IFS= read -r line; do\n"
        "  [ -z \"$line\" ] && continue\n"
        "  : \"$ARCHIVE_DIR/$line\"\n"
        "done < \"$FILE_LIST\"\n",
        encoding="utf-8",
    )
    inflate.chmod(0o755)
    if runtime_tree is None:
        runtime_tree = mod._submission_runtime_manifest(root)["runtime_tree_sha256"]
    archive_sha = _sha(archive)
    archive_bytes = archive.stat().st_size
    seg = 0.00057185
    pose = 0.0001894
    score = 100 * seg + (10 * pose) ** 0.5 + 25 * archive_bytes / 37_545_489
    platform_system = "Linux"
    platform_machine = "x86_64"
    auth = {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": archive_bytes,
        "rate_unscaled": archive_bytes / 37_545_489,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "n_samples": 600,
        "exact_cuda_eval_complete": device == "cuda" and t4,
        "score_claim": device == "cuda" and t4,
        "promotion_eligible": False,
        "score_claim_valid": device == "cuda" and t4,
        "rank_or_kill_eligible": False,
        "lane_tag": "[contest-CUDA]" if device == "cuda" and t4 else "[diagnostic-auth-eval]",
        "score_axis": "contest_cuda" if device == "cuda" and t4 else device,
        "evidence_semantics": (
            "contest_cuda_exact_auth_eval"
            if device == "cuda" and t4
            else "diagnostic_auth_eval_non_promotable"
        ),
        "evidence_grade": "contest-CUDA" if device == "cuda" and t4 else "invalid",
        "provenance": {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "device": device,
            "gpu_t4_match": t4,
            "gpu_model": "Tesla T4" if device == "cuda" and t4 else None,
            "platform_system": platform_system,
            "platform_machine": platform_machine,
        },
    }
    if runtime_tree:
        auth["inflate_runtime_manifest"] = {"runtime_tree_sha256": runtime_tree}
    (root / "contest_auth_eval.json").write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
    cpu_auth = {
        **auth,
        "exact_cuda_eval_complete": False,
        "score_claim": True,
        "promotion_eligible": False,
        "score_claim_valid": True,
        "rank_or_kill_eligible": False,
        "lane_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "evidence_grade": "contest-CPU",
        "cpu_leaderboard_reproduction_eligible": True,
        "provenance": {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "device": "cpu",
            "gpu_t4_match": False,
            "platform_system": platform_system,
            "platform_machine": platform_machine,
            "hardware": "linux x86_64 contest CPU",
        },
    }
    (root / "contest_cpu_auth_eval.json").write_text(
        json.dumps(cpu_auth, indent=2) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive) as zf:
        members = [
            {
                "name": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "crc": info.CRC,
                "sha256": mod._bytes_sha256(zf.read(info)),
            }
            for info in zf.infolist()
        ]
    (root / "archive_manifest.json").write_text(
        json.dumps({"archive": {"sha256": archive_sha, "size_bytes": archive_bytes, "members": members}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "report.txt").write_text(
        (
            f"archive_sha256: {archive_sha}\n"
            f"archive_size_bytes: {archive_bytes}\n"
            f"score: {score}\n"
            f"CUDA axis score: {score} [contest-CUDA]\n"
            f"CPU axis score: {score} [contest-CPU]\n"
            "source code: https://github.com/adpena/tac\n"
            f"pinned source: https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}\n"
            f"runtime_tree_sha256: {runtime_tree}\n"
            "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
            "--archive archive.zip --inflate-sh inflate.sh --device cuda\n"
        ),
        encoding="utf-8",
    )
    if policy_statement is not None:
        (root / "competitive_or_innovative.md").write_text(
            policy_statement + "\n",
            encoding="utf-8",
        )
    return {"archive_sha256": archive_sha, "archive_size_bytes": archive_bytes, "runtime_tree": runtime_tree}


def _auth_runtime_manifest_with_custody_files(
    mod,
    root: Path,
    *,
    corrupt_inflate_sha: bool = False,
) -> dict:
    repo_root = mod.REPO_ROOT.resolve()
    runtime_root = root.resolve()
    files = []
    for row in mod._contest_runtime_root_file_manifest(runtime_root, repo_root):
        if row["relative_path"] == "contest_auth_eval.json":
            continue
        row = dict(row)
        if corrupt_inflate_sha and row["relative_path"] == "inflate.sh":
            row["sha256"] = "0" * 64
        files.append(row)
    upstream_eval_path = repo_root / "upstream" / "evaluate.py"
    upstream_eval = None
    if upstream_eval_path.exists():
        upstream_eval = {
            "relative_path": "evaluate.py",
            "bytes": upstream_eval_path.stat().st_size,
            "sha256": mod._contest_sha256(upstream_eval_path, prefix=0),
        }
    manifest = {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": str(runtime_root),
        "runtime_file_count": len(files),
        "files": files,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": mod._contest_repo_local_tac_import_manifest(
            runtime_root,
            repo_root,
        ),
        "upstream_evaluate_py": upstream_eval,
    }
    manifest["runtime_tree_sha256"] = mod._runtime_tree_sha_from_manifest(manifest)
    return manifest


def _failed_check_names(report: dict) -> set[str]:
    return {check["name"] for check in report["checks"] if not check["passed"]}


def _write_terminal_claim(
    path: Path,
    *,
    lane_id: str = "lane-a",
    job_id: str = "job-a",
    archive_sha256: str = "0" * 64,
    runtime_tree_sha256: str | None = None,
    terminal_status: str = "completed_contest_cuda_score=0.209",
) -> None:
    runtime_note = (
        f" runtime_tree_sha256={runtime_tree_sha256}"
        if runtime_tree_sha256
        else ""
    )
    active_row = (
        f"| 2026-05-07T23:59:00Z | codex | {lane_id} | lightning | {job_id} | "
        "2026-05-08T00:30Z | active_exact_eval | claimed before dispatch |\n"
    )
    terminal_row = (
        f"| 2026-05-08T00:00:00Z | codex | {lane_id} | lightning | {job_id} | "
        f"2026-05-08T00:00Z | {terminal_status} | "
        f"A++ archive_sha256={archive_sha256}{runtime_note} |\n"
    )
    path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + terminal_row
        + active_row,
        encoding="utf-8",
    )


def _write_hosted_archive_manifest(
    path: Path,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    url: str = "https://github.com/adpena/comma_video_compression_challenge/releases/download/pr101-fec6/archive.zip",
) -> str:
    path.write_text(
        json.dumps(
            {
                "schema": "hosted_archive_manifest_v1",
                "url": url,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_size_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return url


def _rewrite_auth_score(path: Path, target_score: float) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    archive_bytes = int(payload["archive_size_bytes"])
    rate_term = 25 * archive_bytes / 37_545_489
    payload["avg_posenet_dist"] = 0.0
    payload["avg_segnet_dist"] = (target_score - rate_term) / 100
    payload["canonical_score"] = target_score
    payload["score_recomputed_from_components"] = target_score
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_pre_submission_check_passes_strict_happy_path(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
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
    assert report["contest_cpu_auth_eval"]["record"]["score_axis"] == "contest_cpu"
    assert report["contest_cpu_auth_eval"]["record"]["promotion_eligible"] is False


def test_contest_final_accepts_explicit_score_that_matches_auth_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    auth_payload = json.loads((submission / "contest_auth_eval.json").read_text())
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--auth-eval-json",
                str(submission / "contest_auth_eval.json"),
                "--contest-final",
                "--submission-score",
                f"{auth_payload['canonical_score']:.15g}",
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

    failed = _failed_check_names(report)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert "contest_final_selected_axis_auth_score_available" not in failed
    assert "contest_final_explicit_score_matches_auth_artifact" not in failed
    assert report["frontier_baseline"]["candidate"]["score_source"] == "strict_formula"


def test_contest_final_rejects_explicit_score_that_disagrees_with_auth_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--submission-score",
                "0.0",
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

    assert not report["passed"]
    assert (
        "contest_final_explicit_score_matches_auth_artifact"
        in _failed_check_names(report)
    )
    assert (
        "contest_final_selected_axis_auth_score_available"
        not in _failed_check_names(report)
    )
    assert report["frontier_baseline"]["candidate"]["score_source"] == "strict_formula"


def test_contest_final_rejects_raw_promotion_blockers_in_auth_eval(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    auth_path = submission / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["promotion_blockers"] = ["pre_submission_compliance_check_not_recorded"]
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
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

    assert not report["passed"]
    assert "auth_eval_raw_promotion_policy_blockers_absent" in _failed_check_names(report)


def test_contest_final_rejects_malformed_raw_rank_blockers(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    auth_path = submission / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["rank_or_kill_blockers"] = {"unexpected": "shape"}
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
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

    assert not report["passed"]
    assert "auth_eval_raw_promotion_policy_blockers_absent" in _failed_check_names(report)


def test_contest_final_rejects_adjudicated_raw_policy_gate_trigger(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    auth_path = submission / "contest_auth_eval.json"
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    payload["raw_promotion_policy_gate_triggered"] = True
    payload["scientific_score_eligible"] = False
    auth_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
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

    assert not report["passed"]
    assert "auth_eval_adjudicated_raw_policy_clean" in _failed_check_names(report)


def test_contest_final_frontier_scan_error_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    import tac.frontier_scan as frontier_scan

    def _boom(_repo_root):
        raise RuntimeError("synthetic frontier scan failure")

    monkeypatch.setattr(frontier_scan, "collect_all_anchors", _boom)
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

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

    assert not report["passed"]
    assert "frontier_scan_helper_available" in _failed_check_names(report)


def test_strict_contest_final_requires_hosted_archive_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(tmp_path / "submission" / "contest_auth_eval.json"),
                "--contest-final",
                "--strict",
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

    assert not report["passed"]
    assert "hosted_archive_manifest_supplied" in _failed_check_names(report)


def test_strict_contest_final_accepts_hosted_archive_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    hosted_manifest = tmp_path / "hosted_archive.json"
    url = _write_hosted_archive_manifest(
        hosted_manifest,
        archive_sha256=expected["archive_sha256"],
        archive_size_bytes=expected["archive_size_bytes"],
    )
    with (submission / "report.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"hosted archive: {url}\n")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--auth-eval-json",
                str(submission / "contest_auth_eval.json"),
                "--contest-final",
                "--strict",
                "--hosted-archive-manifest-json",
                str(hosted_manifest),
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
    assert report["hosted_archive"]["valid"] is True


def test_strict_contest_final_rejects_hosted_archive_placeholder(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    hosted_manifest = tmp_path / "hosted_archive.json"
    url = _write_hosted_archive_manifest(
        hosted_manifest,
        archive_sha256=expected["archive_sha256"],
        archive_size_bytes=expected["archive_size_bytes"],
    )
    with (submission / "report.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"hosted archive: {url}\n")
        handle.write("draft placeholder: <HOSTED_URL_PLACEHOLDER>\n")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--auth-eval-json",
                str(submission / "contest_auth_eval.json"),
                "--contest-final",
                "--strict",
                "--hosted-archive-manifest-json",
                str(hosted_manifest),
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

    assert not report["passed"]
    assert "hosted_archive_public_text_has_no_placeholder" in _failed_check_names(report)


def test_strict_contest_final_rejects_hosted_archive_placeholder_in_public_scan_path(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    hosted_manifest = tmp_path / "hosted_archive.json"
    url = _write_hosted_archive_manifest(
        hosted_manifest,
        archive_sha256=expected["archive_sha256"],
        archive_size_bytes=expected["archive_size_bytes"],
    )
    with (submission / "report.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"hosted archive: {url}\n")
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text("Hosted at: <HOSTED_URL_PLACEHOLDER>\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--auth-eval-json",
                str(submission / "contest_auth_eval.json"),
                "--contest-final",
                "--strict",
                "--hosted-archive-manifest-json",
                str(hosted_manifest),
                "--public-scan-path",
                str(public_pr_body),
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

    assert not report["passed"]
    assert "hosted_archive_public_text_has_no_placeholder" in _failed_check_names(report)
    assert mod._rel(public_pr_body) in report["hosted_archive"]["public_text_sources"]


def test_pre_submission_check_contest_final_requires_cpu_auth_eval_json(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "contest_cpu_auth_eval.json").unlink()
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    assert not report["passed"]
    assert "contest_cpu_auth_eval_exists" in _failed_check_names(report)


def test_contest_final_rejects_cuda_score_above_operator_threshold(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    _rewrite_auth_score(
        tmp_path / "submission" / "contest_auth_eval.json",
        0.1920513168811056,
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    assert not report["passed"]
    assert "auth_eval_score_at_or_below_submission_threshold" in _failed_check_names(report)


def test_contest_final_rejects_cpu_axis_score_above_operator_threshold(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    _rewrite_auth_score(
        tmp_path / "submission" / "contest_cpu_auth_eval.json",
        0.1920513168811056,
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
        terminal_status="completed_contest_cpu_score=0.1920513168811056",
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--contest-final",
                "--submission-score-axis",
                "contest_cpu",
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

    assert not report["passed"]
    assert (
        "contest_cpu_auth_eval_score_at_or_below_submission_threshold"
        in _failed_check_names(report)
    )


def test_pre_submission_check_contest_final_rejects_bad_cpu_pair(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    cpu_auth_path = tmp_path / "submission" / "contest_cpu_auth_eval.json"
    cpu_auth = json.loads(cpu_auth_path.read_text(encoding="utf-8"))
    cpu_auth["archive_size_bytes"] = expected["archive_size_bytes"] + 1
    cpu_auth["score_axis"] = "cpu_advisory"
    cpu_auth["lane_tag"] = "[macOS-CPU advisory]"
    cpu_auth["evidence_grade"] = "macOS-CPU advisory"
    cpu_auth["cpu_leaderboard_reproduction_eligible"] = False
    cpu_auth["provenance"]["archive_sha256"] = "b" * 64
    cpu_auth["provenance"]["archive_size_bytes"] = expected["archive_size_bytes"] + 1
    cpu_auth["provenance"]["platform_system"] = "Darwin"
    cpu_auth["provenance"]["platform_machine"] = "arm64"
    cpu_auth["provenance"]["hardware"] = "macOS Apple Silicon CPU advisory"
    cpu_auth["inflate_runtime_manifest"] = {"runtime_tree_sha256": "c" * 64}
    cpu_auth_path.write_text(json.dumps(cpu_auth, indent=2) + "\n", encoding="utf-8")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "contest_cpu_auth_eval_archive_sha_matches" in failed
    assert "contest_cpu_auth_eval_archive_size_matches" in failed
    assert "contest_cpu_auth_eval_linux_x86_64_provenance" in failed
    assert "contest_cpu_auth_eval_contest_cpu_axis" in failed
    assert "contest_cpu_auth_eval_runtime_tree_matches_cuda" in failed


def test_contest_final_requires_competitive_or_innovative_statement(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", policy_statement=None)
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
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
    failed = _failed_check_names(report)
    assert "post_deadline_policy_statement_present" in failed
    assert not report["passed"]


def test_competitive_or_innovative_statement_can_be_cli_file(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission", policy_statement=None)
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    statement = tmp_path / "policy.md"
    statement.write_text(
        "Innovative: this submission introduces a novel scorer-conditional "
        "archive grammar not on the leaderboard yet, and the report links the "
        "frontier score comparison plus exact-eval custody.",
        encoding="utf-8",
    )
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
                "--competitive-or-innovative-statement-file",
                str(statement),
            ]
        )
    )
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert report["post_deadline_submission_policy"]["source"].endswith("policy.md")


def test_competitive_or_innovative_statement_rejects_template_guidance(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(
        tmp_path / "submission",
        policy_statement=(
            "Is this submission competitive or innovative? Please explain whether "
            "the packet is competitive or innovative for the PR108 template and "
            "include leaderboard context here."
        ),
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "post_deadline_policy_statement_not_template" in failed
    assert "post_deadline_policy_statement_names_mode" in failed


def test_competitive_or_innovative_statement_rejects_negated_claim(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(
        tmp_path / "submission",
        policy_statement=(
            "This packet is not competitive or innovative; it only exists as a "
            "cleanup artifact with leaderboard words and score context that must "
            "not satisfy the final PR policy gate."
        ),
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "post_deadline_policy_statement_not_negated" in failed
    assert "post_deadline_policy_statement_names_mode" in failed


def test_contest_final_requires_public_repo_and_reproducibility_context(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    report_path = tmp_path / "submission" / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        "score: 0.1\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
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
    failed = _failed_check_names(report)
    assert "public_source_repo_link_present" in failed
    assert "public_source_reproducibility_context_present" in failed
    assert not report["passed"]


def test_contest_final_rejects_generic_source_link_without_pinned_revision(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    report_path = tmp_path / "submission" / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        "source code: https://github.com/adpena/tac\n"
        "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
        "--archive archive.zip --inflate-sh inflate.sh --device cuda\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    assert not report["passed"]
    assert "public_source_pinned_revision_present" in _failed_check_names(report)


def test_contest_final_accepts_pinned_source_and_reproduce_command(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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
    assert report["public_source_reproducibility"]["pinned_source_refs"]
    assert report["public_source_reproducibility"]["has_reproduce_command"] is True
    assert report["public_evidence_axis_labels"]["labels"] == {
        "[contest-CUDA]": True,
        "[contest-CPU]": True,
    }


def test_contest_final_accepts_public_source_ref_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    source_refs = tmp_path / "source_refs.json"
    source_refs.write_text(
        json.dumps(
            {
                "schema": "public_source_ref_manifest_v1",
                "repo_url": "https://github.com/adpena/tac",
                "refs": {"refs/heads/main": PINNED_GIT_SHA},
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
                "--public-source-ref-manifest-json",
                str(source_refs),
            ]
        )
    )

    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    source = report["public_source_reproducibility"]
    assert source["source_ref_manifest"]["valid"] is True
    assert source["source_ref_manifest"]["visible_pins"] == [
        f"https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}"
    ]
    assert mod._source_pin_publicly_visible(
        PINNED_GIT_SHA,
        {"refs/heads/main": PINNED_GIT_SHA},
    )


def test_contest_final_accepts_public_source_pin_in_public_scan_path(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    report_path = submission / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        f"runtime_tree_sha256: {expected['runtime_tree']}\n"
        "CUDA axis score: 0.1 [contest-CUDA]\n"
        "CPU axis score: 0.1 [contest-CPU]\n"
        "source code: https://github.com/adpena/tac\n"
        "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
        "--archive archive.zip --inflate-sh inflate.sh --device cuda\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    source_refs = tmp_path / "source_refs.json"
    source_refs.write_text(
        json.dumps(
            {
                "schema": "public_source_ref_manifest_v1",
                "repo_url": "https://github.com/adpena/tac",
                "refs": {"refs/heads/main": PINNED_GIT_SHA},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text(
        f"Public source pin: https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}\n",
        encoding="utf-8",
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
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
                "--public-source-ref-manifest-json",
                str(source_refs),
                "--public-scan-path",
                str(public_pr_body),
            ]
        )
    )

    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    source = report["public_source_reproducibility"]
    assert mod._rel(public_pr_body) in source["sources"]
    assert source["pinned_source_refs"] == [
        f"https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}"
    ]


def test_contest_final_rejects_source_pin_placeholder_in_public_scan_path(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    report_path = submission / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        f"runtime_tree_sha256: {expected['runtime_tree']}\n"
        "CUDA axis score: 0.1 [contest-CUDA]\n"
        "CPU axis score: 0.1 [contest-CPU]\n"
        "source code: https://github.com/adpena/tac\n"
        "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
        "--archive archive.zip --inflate-sh inflate.sh --device cuda\n",
        encoding="utf-8",
    )
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text(
        "Source checkout: git checkout <PINNED_COMMIT>\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--contest-final",
                "--public-scan-path",
                str(public_pr_body),
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

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "public_source_pinned_revision_present" in failed
    assert "public_source_pin_text_has_no_placeholder" in failed
    assert report["public_source_reproducibility"]["pin_placeholders"] == [
        "<PINNED_COMMIT>"
    ]
    assert any(
        mod._rel(public_pr_body) in hit
        for hit in report["public_source_reproducibility"]["pin_placeholder_locations"]
    )


def test_contest_final_rejects_unpublished_source_pin_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    source_refs = tmp_path / "source_refs.json"
    source_refs.write_text(
        json.dumps(
            {
                "schema": "public_source_ref_manifest_v1",
                "repo_url": "https://github.com/adpena/tac",
                "refs": {"refs/heads/main": "b" * 40},
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
                "--public-source-ref-manifest-json",
                str(source_refs),
            ]
        )
    )

    assert not report["passed"]
    assert "public_source_pinned_revision_publicly_visible" in _failed_check_names(report)
    assert report["public_source_reproducibility"]["source_ref_manifest"]["valid"] is False


def test_contest_final_requires_public_cpu_and_cuda_axis_labels(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    report_path = tmp_path / "submission" / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        "score: 0.1\n"
        "source code: https://github.com/adpena/tac\n"
        f"pinned source: https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}\n"
        f"runtime_tree_sha256: {expected['runtime_tree']}\n"
        "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
        "--archive archive.zip --inflate-sh inflate.sh --device cuda\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "public_evidence_contest_cuda_label_present" in failed
    assert "public_evidence_contest_cpu_label_present" in failed


def test_contest_final_accepts_public_axis_labels_in_public_scan_path(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    report_path = submission / "report.txt"
    report_path.write_text(
        f"archive_sha256: {expected['archive_sha256']}\n"
        f"archive_size_bytes: {expected['archive_size_bytes']}\n"
        "score: 0.1\n"
        "source code: https://github.com/adpena/tac\n"
        f"pinned source: https://github.com/adpena/tac/commit/{PINNED_GIT_SHA}\n"
        f"runtime_tree_sha256: {expected['runtime_tree']}\n"
        "reproduce command: .venv/bin/python experiments/contest_auth_eval.py "
        "--archive archive.zip --inflate-sh inflate.sh --device cuda\n",
        encoding="utf-8",
    )
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text(
        "CUDA evidence: 0.1 [contest-CUDA]\nCPU evidence: 0.1 [contest-CPU]\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--contest-final",
                "--public-scan-path",
                str(public_pr_body),
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
    labels = report["public_evidence_axis_labels"]
    assert labels["labels"] == {"[contest-CUDA]": True, "[contest-CPU]": True}
    assert mod._rel(public_pr_body) in labels["sources"]


def test_contest_final_rejects_generic_public_template_placeholder(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text("Release URL: <REPLACE_ME>\n", encoding="utf-8")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--contest-final",
                "--public-scan-path",
                str(public_pr_body),
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

    assert not report["passed"]
    assert "public_text_has_no_unresolved_template_placeholders" in _failed_check_names(report)
    placeholders = report["public_template_placeholders"]
    assert placeholders["placeholders"] == ["<REPLACE_ME>"]
    assert any(mod._rel(public_pr_body) in hit for hit in placeholders["placeholder_locations"])


def test_contest_final_allows_upstream_inflate_signature_placeholders(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission)
    public_pr_body = tmp_path / "PR_BODY.md"
    public_pr_body.write_text(
        "Contract: inflate.sh <archive_dir> <output_dir> <file_list>\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(submission),
                "--contest-final",
                "--public-scan-path",
                str(public_pr_body),
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
    placeholders = report["public_template_placeholders"]
    assert placeholders["placeholders"] == []
    assert "<archive_dir>" in placeholders["allowlist"]


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


def test_pre_submission_check_accepts_strict_runtime_equivalence_proof(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission, runtime_tree="b" * 64)
    current_runtime_tree = mod._submission_runtime_manifest(submission)["runtime_tree_sha256"]
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=current_runtime_tree,
    )
    proof = tmp_path / "runtime_equivalence.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "pre_submission_runtime_equivalence_proof_v1",
                "archive_sha256": expected["archive_sha256"],
                "archive_size_bytes": expected["archive_size_bytes"],
                "auth_eval_runtime_tree_sha256": "b" * 64,
                "submission_runtime_tree_sha256": current_runtime_tree,
                "equivalence_basis": "full_inflate_output_byte_identity",
                "output_equivalence": {
                    "basis": "full_inflate_output_byte_identity",
                    "baseline_output_sha256": "c" * 64,
                    "candidate_output_sha256": "c" * 64,
                    "diff_bytes": 0,
                    "files": [
                        {
                            "path": "0.raw",
                            "auth_eval_output_sha256": "c" * 64,
                            "submission_output_sha256": "c" * 64,
                            "diff_bytes": 0,
                        }
                    ],
                },
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
                current_runtime_tree,
                "--runtime-equivalence-proof-json",
                str(proof),
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
    assert report["submission_runtime"]["runtime_equivalence_proof"]["valid"] is True


def test_pre_submission_check_rejects_stale_runtime_equivalence_shape(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission, runtime_tree="b" * 64)
    current_runtime_tree = mod._submission_runtime_manifest(submission)["runtime_tree_sha256"]
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=current_runtime_tree,
    )
    proof = tmp_path / "runtime_equivalence.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "pre_submission_runtime_equivalence_proof_v1",
                "archive_sha256": expected["archive_sha256"],
                "archive_size_bytes": expected["archive_size_bytes"],
                "auth_eval_runtime_tree_sha256": "b" * 64,
                "submission_runtime_tree_sha256": current_runtime_tree,
                "submission_runtime_shape": {"inflate.sh": "0" * 64},
                "equivalence_basis": "full_inflate_output_byte_identity",
                "output_equivalence": {
                    "basis": "full_inflate_output_byte_identity",
                    "baseline_output_sha256": "c" * 64,
                    "candidate_output_sha256": "c" * 64,
                    "diff_bytes": 0,
                },
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
                current_runtime_tree,
                "--runtime-equivalence-proof-json",
                str(proof),
                "--dispatch-claims-md",
                str(claims),
                "--expected-lane-id",
                "lane-a",
                "--expected-job-id",
                "job-a",
            ]
        )
    )

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "runtime_equivalence_proof_submission_runtime_shape_matches" in failed
    diff = report["submission_runtime"]["runtime_equivalence_proof"][
        "submission_runtime_shape_diff"
    ]
    assert diff["changed"][0]["relative_path"] == "inflate.sh"
    assert report["submission_runtime"]["runtime_equivalence_proof"]["valid"] is False


def test_pre_submission_check_rejects_malformed_runtime_equivalence_proof_size(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    submission = tmp_path / "submission"
    expected = _write_submission(submission, runtime_tree="b" * 64)
    current_runtime_tree = mod._submission_runtime_manifest(submission)["runtime_tree_sha256"]
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=current_runtime_tree,
    )
    proof = tmp_path / "runtime_equivalence.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "pre_submission_runtime_equivalence_proof_v1",
                "archive_sha256": expected["archive_sha256"],
                "archive_size_bytes": "not-an-int",
                "auth_eval_runtime_tree_sha256": "b" * 64,
                "submission_runtime_tree_sha256": current_runtime_tree,
                "equivalence_basis": "full_inflate_output_byte_identity",
                "output_equivalence": {
                    "basis": "full_inflate_output_byte_identity",
                    "baseline_output_sha256": "c" * 64,
                    "candidate_output_sha256": "c" * 64,
                    "diff_bytes": 0,
                },
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
                current_runtime_tree,
                "--runtime-equivalence-proof-json",
                str(proof),
                "--dispatch-claims-md",
                str(claims),
                "--expected-lane-id",
                "lane-a",
                "--expected-job-id",
                "job-a",
            ]
        )
    )

    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "runtime_equivalence_proof_archive_matches" in failed
    assert "auth_eval_runtime_tree_expected_match" in failed
    assert "submission_runtime_tree_matches_auth_eval" in failed
    assert report["submission_runtime"]["runtime_equivalence_proof"]["valid"] is False


def test_pre_submission_check_matches_auth_runtime_after_custody_pruning_and_remote_root(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    full_auth_manifest = _auth_runtime_manifest_with_custody_files(
        mod,
        tmp_path / "submission",
    )
    assert full_auth_manifest["runtime_tree_sha256"] != expected["runtime_tree"]
    full_auth_manifest["runtime_root"] = "/tmp/modal_auth_eval/submission_dir"
    full_auth_manifest["repo_local_tac_import_manifest"]["runtime_root_name"] = "submission_dir"
    for row in full_auth_manifest["files"]:
        row["repo_relative_path"] = f"/tmp/modal_auth_eval/submission_dir/{row['relative_path']}"
    auth["inflate_runtime_manifest"] = full_auth_manifest
    auth["provenance"]["inflate_runtime_manifest"] = full_auth_manifest
    auth["provenance"]["inflate_script"] = "/tmp/modal_auth_eval/submission_dir/inflate.sh"
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")
    cpu_auth_path = tmp_path / "submission" / "contest_cpu_auth_eval.json"
    cpu_auth = json.loads(cpu_auth_path.read_text(encoding="utf-8"))
    cpu_auth["inflate_runtime_manifest"] = full_auth_manifest
    cpu_auth["provenance"]["inflate_runtime_manifest"] = full_auth_manifest
    cpu_auth["provenance"]["inflate_script"] = "/tmp/modal_auth_eval/submission_dir/inflate.sh"
    cpu_auth_path.write_text(json.dumps(cpu_auth, indent=2) + "\n", encoding="utf-8")

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
    pruned_candidates = report["auth_eval"]["runtime_tree_pruned_candidates"]
    assert (
        pruned_candidates[
            "provenance.inflate_runtime_manifest.runtime_tree_sha256_without_submission_custody_files"
        ]
        != expected["runtime_tree"]
    )
    assert (
        pruned_candidates[
            "provenance.inflate_runtime_manifest.portable_runtime_tree_sha256_without_submission_custody_files"
        ]
        == report["submission_runtime"]["portable_runtime_tree_sha256_without_custody_files"]
    )
    eval_command = report["auth_eval"]["anchor_proof"]["auth_eval"]["eval_command_sanitized"]
    assert eval_command[eval_command.index("--inflate-sh") + 1] == "submission_dir/inflate.sh"
    assert report["submission_runtime"]["runtime_tree_sha256"] == expected["runtime_tree"]


def test_pre_submission_check_does_not_relabel_provider_temp_runtime_as_default_runtime(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["provenance"]["inflate_script"] = "/tmp/modal_auth_eval/submission_dir/inflate.sh"
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
    eval_command = report["auth_eval"]["anchor_proof"]["auth_eval"]["eval_command_sanitized"]
    assert eval_command[eval_command.index("--inflate-sh") + 1] == (
        "non_repo_absolute_runtime/inflate.sh"
    )
    assert "submissions/pr103_pr106_final_runtime/inflate.sh" not in eval_command


def test_pre_submission_check_rejects_non_custody_runtime_manifest_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    full_auth_manifest = _auth_runtime_manifest_with_custody_files(
        mod,
        tmp_path / "submission",
        corrupt_inflate_sha=True,
    )
    auth["inflate_runtime_manifest"] = full_auth_manifest
    auth["provenance"]["inflate_runtime_manifest"] = full_auth_manifest
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
    assert "submission_runtime_tree_matches_auth_eval" in _failed_check_names(report)


def test_pre_submission_check_contest_final_rejects_inferred_exact_cuda_stamp(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth.pop("exact_cuda_eval_complete")
    auth.pop("lane_tag")
    auth.pop("score_axis")
    auth.pop("score_claim_valid")
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
    assert "auth_eval_explicit_exact_cuda_stamp" in _failed_check_names(report)


def test_pre_submission_check_records_strict_formula_when_report_score_uses_rounded_rate(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
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


def test_pre_submission_check_rejects_auth_eval_component_score_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    auth_path = tmp_path / "submission" / "contest_auth_eval.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["canonical_score"] = auth["score_recomputed_from_components"] + 0.01
    auth["score_recomputed_from_components"] = auth["canonical_score"]
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")

    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
                "--auth-eval-json",
                str(auth_path),
                "--require-auth-eval",
                "--expected-archive-sha256",
                expected["archive_sha256"],
                "--expected-archive-size-bytes",
                str(expected["archive_size_bytes"]),
            ]
        )
    )

    assert not report["passed"]
    failed = _failed_check_names(report)
    assert "auth_eval_schema_metric_consistency" in failed
    assert "auth_eval_score_recomputes" in failed


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
    assert "auth_eval_exact_cuda_stamp" in failed


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


def test_pre_submission_check_contest_final_rejects_stale_archive_member_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    manifest = tmp_path / "submission" / "archive_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["archive"]["members"][0]["sha256"] = "b" * 64
    payload["archive"]["members"][0]["crc"] = 0
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
    failed = _failed_check_names(report)
    assert not report["passed"]
    assert "archive_manifest_member_0_sha256_matches" in failed
    assert "archive_manifest_member_0_crc_matches" in failed


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


def test_pre_submission_check_accepts_nested_candidate_archive_manifest_with_hex_crc(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    archive = tmp_path / "submission" / "archive.zip"
    with zipfile.ZipFile(archive) as zf:
        info = zf.infolist()[0]
        member = zf.read(info)
        members = [
            {
                "name": info.filename,
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "crc": f"{info.CRC:08x}",
                "sha256": mod._bytes_sha256(member),
            }
        ]
    manifest = tmp_path / "a5_candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "score_claim": False,
                "charged_bits_changed": True,
                "candidate_archive": {
                    "sha256": expected["archive_sha256"],
                    "bytes": expected["archive_size_bytes"],
                    "members": members,
                },
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
    passed = {check["name"]: check for check in report["checks"] if check["passed"]}
    assert "auth_eval_exists" not in passed
    assert "auth_eval_present_or_optional" in passed
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
        "2026-05-04T15:05Z | completed_score=0.209 | A++ |\n"
        "| 2026-05-04T15:00:00Z | codex | lane-a | lightning | job-a | "
        "2026-05-04T15:30Z | active_exact_eval | prior active claim |\n",
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


def test_pre_submission_check_dispatch_claim_linkage_rejects_terminal_without_prior_active(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-04T15:05:13Z | codex | lane-a | lightning | job-a | "
        "2026-05-04T15:05Z | completed_score=0.209 | fabricated terminal only |\n",
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
    assert "dispatch_claim_prior_active_row" in _failed_check_names(report)


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
        "2026-05-04T15:05Z | cancelled_operator_request | terminal per claim helper |\n"
        "| 2026-05-04T15:00:00Z | codex | lane-a | lightning | job-a | "
        "2026-05-04T15:30Z | active_exact_eval | prior active claim |\n",
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


def test_pre_submission_check_contest_final_rejects_unsuccessful_terminal_claim(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-08T01:00:00Z | codex | lane-a | modal | job-a | "
        f"2026-05-08T01:00Z | cancelled_operator_request | archive_sha256={expected['archive_sha256']} |\n"
        "| 2026-05-08T00:00:00Z | codex | lane-a | modal | job-a | "
        "2026-05-08T00:30Z | active_exact_eval | prior active claim |\n",
        encoding="utf-8",
    )

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

    assert not report["passed"]
    failed = _failed_check_names(report)
    assert "dispatch_claim_terminal_row" not in failed
    assert "dispatch_claim_terminal_archive_sha_bound" not in failed
    assert "dispatch_claim_successful_exact_eval_terminal_row" in failed


def test_pre_submission_check_contest_final_requires_terminal_runtime_tree_binding(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    claims = tmp_path / "claims.md"
    _write_terminal_claim(claims, archive_sha256=expected["archive_sha256"])

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

    assert not report["passed"]
    failed = _failed_check_names(report)
    assert "dispatch_claim_terminal_archive_sha_bound" not in failed
    assert "dispatch_claim_successful_exact_eval_terminal_row" not in failed
    assert "dispatch_claim_terminal_runtime_tree_sha_bound" in failed


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


def test_pre_submission_check_requires_canonical_inflate_signature(tmp_path: Path) -> None:
    mod = _load_module()
    _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython inflate.py \"$1\" \"$2\"\n",
        encoding="utf-8",
    )
    report = mod.build_report(mod.build_arg_parser().parse_args(["--submission-dir", str(tmp_path / "submission")]))
    assert not report["passed"]
    assert "inflate_sh_uses_canonical_three_arg_contract" in _failed_check_names(report)


def test_pre_submission_check_rejects_runtime_scorer_imports(tmp_path: Path) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "inflate.py").write_text(
        "from modules import DistortionNet\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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
    assert not report["passed"]
    assert "submission_runtime_loads_no_scorers_or_eval" in _failed_check_names(report)


def test_pre_submission_check_rejects_runtime_network_install_or_local_paths(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "inflate.py").write_text(
        "import urllib.request\nurllib.request.urlopen('https://example.com/archive.zip')\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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
    assert not report["passed"]
    assert "submission_runtime_has_no_network_install_or_local_paths" in _failed_check_names(report)


def test_pre_submission_check_rejects_runtime_dependency_outside_allowlist(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "inflate.py").write_text(
        "import pandas as pd\nprint(pd.__version__)\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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
    assert not report["passed"]
    assert "submission_runtime_imports_within_allowlist" in _failed_check_names(report)
    assert any(
        hit.endswith("submission/inflate.py:pandas")
        for hit in report["submission_runtime"]["disallowed_runtime_imports"]
    )


def test_pre_submission_check_rejects_unparseable_runtime_imports(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    expected = _write_submission(tmp_path / "submission")
    (tmp_path / "submission" / "inflate.py").write_text(
        "import torch\nif True print('bad')\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    _write_terminal_claim(
        claims,
        archive_sha256=expected["archive_sha256"],
        runtime_tree_sha256=expected["runtime_tree"],
    )
    report = mod.build_report(
        mod.build_arg_parser().parse_args(
            [
                "--submission-dir",
                str(tmp_path / "submission"),
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
    assert not report["passed"]
    assert "submission_runtime_import_allowlist_parseable" in _failed_check_names(report)
    assert report["submission_runtime"]["runtime_import_parse_errors"]
