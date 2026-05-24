from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "parallel_dispatch_top_k.py"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.hnerv_frontier_defaults import (  # noqa: E402
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE,
    ACTIVE_SCORE_FRONTIER_SCORE,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest  # noqa: E402


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_parallel_dispatch_floor_preserves_active_nonpromotional_reference() -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_floor_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert (
        module.DEFAULT_ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE
        == ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE
    )
    assert module.DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE == ACTIVE_SCORE_FRONTIER_SCORE
    assert module.DEFAULT_ACTIVE_FLOOR_SCORE == module.DEFAULT_ACTIVE_SCORE_FRONTIER_SCORE


def test_parallel_dispatch_ready_flag_still_requires_live_custody(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_authority_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    runtime_manifest = runtime_dependency_manifest(
        submission,
        REPO_ROOT,
    )
    runtime_tree_sha256 = runtime_manifest["runtime_tree_sha256"]
    runtime_content_tree_sha256 = runtime_manifest["runtime_content_tree_sha256"]

    candidate = {
        "candidate_id": "ready_flag_without_live_custody",
        "lane_id": "ready_flag_without_live_custody_lane",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
        "ready_for_exact_eval_dispatch": True,
        "score_claim": False,
        "score_claim_verified": False,
        "archive_path": str(submission / "archive.zip"),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_content_tree_sha256": runtime_content_tree_sha256,
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_bytes,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }

    blockers = module._candidate_blockers(
        candidate,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
    )

    assert "exact_dispatch_authority:archive_manifest_missing" in blockers
    assert "exact_dispatch_authority:report_txt_missing" in blockers


def _live_dispatch_candidate(tmp_path: Path) -> dict[str, object]:
    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("contest custody fixture\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
            "score_claim": False,
        },
    )
    runtime_manifest = runtime_dependency_manifest(
        submission,
        REPO_ROOT,
    )
    return {
        "candidate_id": "fixture",
        "lane_id": "fixture_lane",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
        "ready_for_exact_eval_dispatch": True,
        "score_axis": "contest_cuda",
        "target_score_axis": "contest_cuda",
        "score_claim": False,
        "score_claim_verified": False,
        "archive_path": str(submission / "archive.zip"),
        "submission_dir": str(submission),
        "archive_manifest_path": str(submission / "archive_manifest.json"),
        "inflate_sh_path": str(inflate),
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_bytes,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }


def test_parallel_dispatch_blocks_truthy_authority_fields(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_truthy_authority_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = _live_dispatch_candidate(tmp_path)

    for field in (
        "rank_or_kill_eligible",
        "score_claim_valid",
        "exact_cuda_auth_eval",
        "contest_cuda_auth_eval",
        "dispatch_attempted",
        "promotable",
    ):
        row = dict(candidate)
        row[field] = True
        blockers = module._candidate_blockers(
            row,
            ranked_input_dir=tmp_path,
            active_floor_archive_bytes=999_999,
            active_floor_score=0.2,
            dispatch_claims_path=tmp_path / "claims.md",
        )
        assert f"exact_dispatch_authority:truthy_authority_field:{field}=truthy" in blockers


def test_parallel_dispatch_requires_explicit_contest_cuda_score_axis(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_score_axis_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = _live_dispatch_candidate(tmp_path)

    missing = dict(candidate)
    missing.pop("score_axis", None)
    missing.pop("target_score_axis", None)
    missing_blockers = module._candidate_blockers(
        missing,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
        dispatch_claims_path=tmp_path / "claims.md",
    )
    assert "exact_dispatch_authority:score_axis_missing:required=contest_cuda" in missing_blockers

    wrong = dict(candidate)
    wrong["score_axis"] = "contest_cpu"
    wrong["target_score_axis"] = "contest_cpu"
    wrong_blockers = module._candidate_blockers(
        wrong,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
        dispatch_claims_path=tmp_path / "claims.md",
    )
    assert any(
        blocker.startswith("exact_dispatch_authority:score_axis_required:contest_cuda:")
        for blocker in wrong_blockers
    )


def test_parallel_dispatch_does_not_treat_contest_mode_as_exact_eval(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_contest_mode_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = _live_dispatch_candidate(tmp_path)
    candidate.pop("target_modes", None)
    candidate["contest_mode"] = True

    blockers = module._candidate_blockers(
        candidate,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
        dispatch_claims_path=tmp_path / "claims.md",
    )

    assert "exact_dispatch_authority:contest_exact_eval_target_mode_missing" in blockers


def test_parallel_dispatch_does_not_treat_deployment_target_as_exact_eval(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_deployment_target_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = _live_dispatch_candidate(tmp_path)
    candidate.pop("target_modes", None)
    candidate["deployment_target"] = "contest_exact_eval"

    blockers = module._candidate_blockers(
        candidate,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
        dispatch_claims_path=tmp_path / "claims.md",
    )

    assert "exact_dispatch_authority:contest_exact_eval_target_mode_missing" in blockers
    assert any(blocker.startswith("target_modes_missing;") for blocker in blockers)


def test_parallel_dispatch_blocks_proxy_score_fields(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "parallel_dispatch_top_k_proxy_score_field_test",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = _live_dispatch_candidate(tmp_path)
    candidate["proxy_score"] = 0.1
    candidate["macos_cpu_score"] = 0.2

    blockers = module._candidate_blockers(
        candidate,
        ranked_input_dir=tmp_path,
        active_floor_archive_bytes=999_999,
        active_floor_score=0.2,
        dispatch_claims_path=tmp_path / "claims.md",
    )

    assert (
        "exact_dispatch_authority:pre_dispatch_score_field_present:"
        "macos_cpu_score,proxy_score"
    ) in blockers
    assert "predicted_score_field_present:macos_cpu_score,proxy_score" in blockers


def test_parallel_dispatch_refuses_stale_exact_ready_terminal_claim(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    archive_bytes, archive_sha = _write_archive(submission / "archive.zip")
    inflate = submission / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text("contest custody fixture\n", encoding="utf-8")
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
            "score_claim": False,
        },
    )
    runtime_manifest = runtime_dependency_manifest(
        submission,
        REPO_ROOT,
    )
    runtime_tree_sha256 = runtime_manifest["runtime_tree_sha256"]
    runtime_content_tree_sha256 = runtime_manifest["runtime_content_tree_sha256"]
    queue = _write_json(
        tmp_path / "exact_ready_queue.json",
        {
            "schema": "noncanonical_dispatch_ready_fixture_v1",
            "dispatch_ready": [
                {
                    "candidate_id": "fixture",
                    "lane_id": "fixture_lane",
                    "target_modes": ["contest_exact_eval"],
                    "deployment_target": "t4_contest_runtime",
                    "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
                    "ready_for_exact_eval_dispatch": True,
                    "score_claim": False,
                    "score_claim_verified": False,
                    "archive_path": str(submission / "archive.zip"),
                    "submission_dir": str(submission),
                    "inflate_sh_path": str(inflate),
                    "runtime_tree_sha256": runtime_tree_sha256,
                    "runtime_content_tree_sha256": runtime_content_tree_sha256,
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "source_archive_sha256": "0" * 64,
                    "score_axis": "contest_cuda",
                    "target_score_axis": "contest_cuda",
                    "predicted_band": [0.19, 0.2],
                }
            ],
            "top_k": [],
        },
    )
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job-1 | | completed_contest_cuda | score=0.226 archive_sha={archive_sha} |\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--ranked-input",
            str(queue),
            "--provider",
            "vastai",
            "--dry-run",
            "--dispatch-claims-path",
            str(claims),
            "--active-floor-score",
            "0.2",
            "--active-floor-archive-bytes",
            "999999",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "ranked-input contains non-dispatch-ready candidates" in proc.stderr
    assert "same_lane_terminal_score_not_below_active_floor_for_same_archive" in proc.stderr


def test_parallel_dispatch_refuses_paid_vastai_without_claim_enforcement(tmp_path: Path) -> None:
    ranked = _write_json(tmp_path / "ranked.json", {"dispatch_ready": []})

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--ranked-input",
            str(ranked),
            "--provider",
            "vastai",
            "--allow-above-active-floor-dispatch",
            "--operator-override-reason",
            "fixture",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "provider=vastai is disabled" in proc.stderr
    assert "claim_lane_dispatch.py" in proc.stderr
