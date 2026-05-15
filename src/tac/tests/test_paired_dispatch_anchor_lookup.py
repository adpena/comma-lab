# SPDX-License-Identifier: MIT
"""Tests for canonical paired-dispatch anchor lookup + skip-logic.

PAIRED-DISPATCH-SKIP-IF-ANCHOR-EXISTS-ENHANCEMENT 2026-05-15.

Coverage matrix:
1. Anchor exists for CUDA only -> CPU dispatched, CUDA skipped, log + repointer
2. Anchor exists for CPU only -> CUDA dispatched, CPU skipped
3. Anchor exists for both -> no dispatch fires; rc=0; both repointers written
4. Anchor exists for neither -> both dispatch, preserving current behavior
5. Custody-chain incomplete, missing evidence_grade -> no anchor; dispatch fires
6. Archive sha mismatch -> no anchor; dispatch fires
7. Axis mismatch, advisory CPU when CUDA requested -> no anchor
8. Default flag false -> skip-logic disabled; current behavior preserved
9. Flag true + ledger landed -> uses ledger lookup
10. Flag true + ledger missing -> falls back to filesystem scan
11. find_promotable_anchor_for_axis_and_sha unit tests for each lookup-path

Per the standing canonical-helpers directive, fail-safe semantics: any unclear
custody invariant returns ``None`` and lets the dispatcher fire fresh. Cost
of false-negative skip = one extra ~$0.40 dispatch; cost of false-positive
skip = corrupt promotion claim. The latter is catastrophic per Catalog
#127 / #221; the former is recoverable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_modal_paired_auth_eval.py"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.modal.anchor_lookup import (  # noqa: E402
    _archive_sha_matches,
    _grade_is_advisory,
    _grade_matches_axis,
    _normalize_axis,
    _score_claim_present,
    _score_is_finite_numeric,
    _validate_payload_for_axis_and_sha,
    find_promotable_anchor_for_axis_and_sha,
)

# Helpers: synthesize minimal anchor JSON files in a tmp repo_root.


def _write_canonical_cuda_anchor(
    repo_root: Path,
    *,
    archive_sha256: str,
    score: float = 0.19869,
    label: str = "z3_v2_full",
    grade: str = "contest-CUDA",
    score_claim_valid: bool = True,
    runtime_tree_sha256: str = "a" * 64,
) -> Path:
    """Write a sister-of-c6 canonical contest_auth_eval_cuda.json shape."""
    lane_dir = repo_root / "experiments" / "results" / f"lane_{label}_modal" / "harvested_artifacts"
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / "contest_auth_eval_cuda.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_grade": grade,
                "score_claim_valid": score_claim_valid,
                "score_claim": score_claim_valid,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": 224857,
                "score": score,
                "scorer_device": "cuda",
                "score_axis": "contest_cuda",
                "n_samples": 600,
                "runtime_tree_sha256": runtime_tree_sha256,
            },
            indent=2,
        )
    )
    return path


def _write_canonical_cpu_anchor(
    repo_root: Path,
    *,
    archive_sha256: str,
    score: float = 0.19663,
    label: str = "z3_v2_full",
    grade: str = "contest-CPU",
    runtime_tree_sha256: str = "a" * 64,
) -> Path:
    """Write a CPU-axis anchor."""
    lane_dir = repo_root / "experiments" / "results" / f"lane_{label}_modal" / "harvested_artifacts"
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / "contest_auth_eval_cpu.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_grade": grade,
                "score_claim_valid": True,
                "score_claim": True,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": 224857,
                "score": score,
                "scorer_device": "cpu",
                "score_axis": "contest_cpu",
                "n_samples": 600,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
                "runtime_tree_sha256": runtime_tree_sha256,
            },
            indent=2,
        )
    )
    return path


def _write_modal_cuda_result(
    repo_root: Path,
    *,
    archive_sha256: str,
    score: float = 0.21,
    label: str = "modal_legacy",
    runtime_tree_sha256: str = "a" * 64,
) -> Path:
    """Write the older ``modal_cuda_auth_eval_result.json`` shape."""
    out_dir = repo_root / "experiments" / "results" / "modal_auth_eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "modal_cuda_auth_eval_result.json"
    path.write_text(
        json.dumps(
            {
                "evidence_grade": "contest-CUDA",
                "score_claim": True,
                "score_axis": "contest_cuda",
                "expected_archive_sha256": archive_sha256,
                "archive_size_bytes": 186405,
                "final_score": score,
                "n_samples": 600,
                "scorer_device": "cuda",
                "runtime_tree_sha256": runtime_tree_sha256,
            },
            indent=2,
        )
    )
    return path


def _make_archive_file(repo_root: Path) -> Path:
    """Make a deterministic non-empty archive file so build_plan can hash it."""
    archive_dir = repo_root / "submissions" / "test"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / "archive.zip"
    path.write_bytes(b"PK\x03\x04" + b"a" * 1024)  # minimal non-empty
    import hashlib

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def _write_modal_call_id_ledger_row(repo_root: Path, payload: dict[str, object]) -> Path:
    """Write one canonical Modal call-id ledger row under a tmp repo root."""
    ledger = repo_root / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return ledger


# Unit tests for find_promotable_anchor_for_axis_and_sha


def test_axis_normalization_accepts_canonical_and_short_forms():
    assert _normalize_axis("cuda") == "contest_cuda"
    assert _normalize_axis("CUDA") == "contest_cuda"
    assert _normalize_axis("contest_cuda") == "contest_cuda"
    assert _normalize_axis("contest-cuda") == "contest_cuda"
    assert _normalize_axis("cpu") == "contest_cpu"
    assert _normalize_axis("CPU") == "contest_cpu"
    assert _normalize_axis("contest_cpu") == "contest_cpu"
    assert _normalize_axis("mps") is None
    assert _normalize_axis("") is None
    assert _normalize_axis(None) is None  # type: ignore[arg-type]


def test_grade_matches_axis_promotable_only():
    assert _grade_matches_axis("contest-CUDA", "contest_cuda") is True
    assert _grade_matches_axis("[contest-CUDA]", "contest_cuda") is True
    assert _grade_matches_axis("contest-CPU", "contest_cpu") is True
    assert _grade_matches_axis("[contest-CPU GHA Linux x86_64]", "contest_cpu") is True
    # Wrong axis
    assert _grade_matches_axis("contest-CUDA", "contest_cpu") is False
    assert _grade_matches_axis("contest-CPU", "contest_cuda") is False
    # Advisory grades refused
    assert _grade_matches_axis("macOS-CPU-advisory", "contest_cpu") is False
    assert _grade_matches_axis("training-only", "contest_cuda") is False
    assert _grade_matches_axis("MPS-PROXY", "contest_cuda") is False
    # Empty / non-string
    assert _grade_matches_axis("", "contest_cuda") is False
    assert _grade_matches_axis(None, "contest_cuda") is False


def test_grade_is_advisory_substring_matches():
    assert _grade_is_advisory("macos-cpu-advisory") is True
    assert _grade_is_advisory("contest-cpu (advisory only)") is True
    assert _grade_is_advisory("training-only") is True
    assert _grade_is_advisory("contest-cuda") is False
    assert _grade_is_advisory("") is False


def test_archive_sha_matches_exact_only():
    assert _archive_sha_matches("AbCDef0123", "abcdef0123") is True
    assert _archive_sha_matches("abcdef0123", "ABCDEF0123") is True
    # Partial prefix REJECTED
    assert _archive_sha_matches("abcdef", "abcdef0123") is False
    assert _archive_sha_matches("abcdef0123", "abcdef") is False
    # Empty / non-string
    assert _archive_sha_matches("", "abc") is False
    assert _archive_sha_matches("abc", "") is False
    assert _archive_sha_matches(None, "abc") is False


def test_score_is_finite_numeric_rejects_nan_inf_bool_none():
    assert _score_is_finite_numeric(0.19869) is True
    assert _score_is_finite_numeric(0) is True
    assert _score_is_finite_numeric(-1.5) is True
    assert _score_is_finite_numeric(float("nan")) is False
    assert _score_is_finite_numeric(float("inf")) is False
    assert _score_is_finite_numeric(float("-inf")) is False
    assert _score_is_finite_numeric(None) is False
    assert _score_is_finite_numeric(True) is False  # bool subclass refused
    assert _score_is_finite_numeric("0.1") is False


def test_score_claim_present_requires_explicit_true():
    assert _score_claim_present({"score_claim_valid": True}) is True
    assert _score_claim_present({"score_claim": True}) is True
    assert _score_claim_present({}) is False
    assert _score_claim_present({"score_claim_valid": False}) is False
    assert _score_claim_present({"score_claim_valid": "true"}) is False  # string-true refused
    assert _score_claim_present({"score_claim_valid": 1}) is False  # integer-truthy refused


def test_validate_payload_all_4_checks_must_pass():
    base = {
        "evidence_grade": "contest-CUDA",
        "score_claim_valid": True,
        "archive_sha256": "deadbeef" * 8,
        "score": 0.21,
        "score_axis": "contest_cuda",
        "n_samples": 600,
        "scorer_device": "cuda",
    }
    # All 4 checks pass
    assert (
        _validate_payload_for_axis_and_sha(base, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8) is True
    )
    # Missing grade
    payload = {**base}
    del payload["evidence_grade"]
    assert (
        _validate_payload_for_axis_and_sha(payload, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8)
        is False
    )
    # Missing score_claim
    payload = {**base, "score_claim_valid": False, "score_claim": False}
    assert (
        _validate_payload_for_axis_and_sha(payload, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8)
        is False
    )
    # Sha mismatch
    assert (
        _validate_payload_for_axis_and_sha(base, canonical_axis="contest_cuda", archive_sha256="cafebabe" * 8) is False
    )
    # Non-finite score
    payload = {**base, "score": float("nan")}
    assert (
        _validate_payload_for_axis_and_sha(payload, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8)
        is False
    )
    # Missing explicit score axis refuses reuse; axis cannot be inferred from grade.
    payload = {**base}
    del payload["score_axis"]
    assert (
        _validate_payload_for_axis_and_sha(payload, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8)
        is False
    )
    # Wrong sample count refuses reuse.
    payload = {**base, "n_samples": 599}
    assert (
        _validate_payload_for_axis_and_sha(payload, canonical_axis="contest_cuda", archive_sha256="deadbeef" * 8)
        is False
    )


def test_validate_payload_requires_cpu_linux_x86_64_contract():
    base = {
        "evidence_grade": "contest-CPU",
        "score_claim_valid": True,
        "archive_sha256": "deadbeef" * 8,
        "score": 0.192,
        "score_axis": "contest_cpu",
        "n_samples": 600,
        "scorer_device": "cpu",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
    }
    assert (
        _validate_payload_for_axis_and_sha(base, canonical_axis="contest_cpu", archive_sha256="deadbeef" * 8) is True
    )
    assert (
        _validate_payload_for_axis_and_sha(
            {**base, "platform_system": "Darwin"},
            canonical_axis="contest_cpu",
            archive_sha256="deadbeef" * 8,
        )
        is False
    )
    assert (
        _validate_payload_for_axis_and_sha(
            {**base, "platform_machine": ""},
            canonical_axis="contest_cpu",
            archive_sha256="deadbeef" * 8,
        )
        is False
    )
    assert (
        _validate_payload_for_axis_and_sha(
            {
                **base,
                "platform_system": "",
                "platform_machine": "",
                "hardware": "github-actions-ubuntu-latest_x86_64",
            },
            canonical_axis="contest_cpu",
            archive_sha256="deadbeef" * 8,
        )
        is True
    )


# Filesystem scan + ledger fallback tests


def test_lookup_returns_none_when_no_results_dir(tmp_path):
    sha = "deadbeef" * 8
    assert find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path) is None
    assert find_promotable_anchor_for_axis_and_sha("cpu", sha, repo_root=tmp_path) is None


def test_lookup_filesystem_finds_canonical_cuda_anchor(tmp_path):
    sha = "feedface" * 8
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha, score=0.19869)
    hit = find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path)
    assert hit is not None
    assert hit["axis"] == "contest_cuda"
    assert hit["archive_sha256"] == sha
    assert hit["score"] == pytest.approx(0.19869)
    assert hit["runtime_tree_sha256"] == "a" * 64
    assert hit["custody_match"] is True
    assert hit["source"] == "filesystem_scan"
    assert hit["evidence_grade"] == "contest-CUDA"


def test_lookup_filesystem_finds_canonical_cpu_anchor(tmp_path):
    sha = "1234abcd" * 8
    _write_canonical_cpu_anchor(tmp_path, archive_sha256=sha, score=0.19663)
    hit = find_promotable_anchor_for_axis_and_sha("cpu", sha, repo_root=tmp_path)
    assert hit is not None
    assert hit["axis"] == "contest_cpu"
    assert hit["score"] == pytest.approx(0.19663)


def test_lookup_filesystem_finds_modal_legacy_layout(tmp_path):
    sha = "abc01234" * 8
    _write_modal_cuda_result(tmp_path, archive_sha256=sha, score=0.21)
    hit = find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path)
    assert hit is not None
    assert hit["score"] == pytest.approx(0.21)


def test_lookup_ledger_finds_nested_harvest_result(tmp_path):
    sha = "c" * 64
    _write_modal_call_id_ledger_row(
        tmp_path,
        {
            "call_id": "fc-test",
            "status": "harvested",
            "event_type": "harvested",
            "harvest_result": {
                "evidence_grade": "[contest-CUDA]",
                "score_claim_valid": True,
                "score_axis": "contest_cuda",
                "scorer_device": "cuda",
                "n_samples": 600,
                "archive_sha256": sha,
                "eval_archive_size_bytes": 456,
                "score": 0.234,
                "runtime_tree_sha256": "b" * 64,
            },
        },
    )

    hit = find_promotable_anchor_for_axis_and_sha(
        "contest_cuda",
        sha,
        repo_root=tmp_path,
        expected_runtime_tree_sha256="b" * 64,
    )

    assert hit is not None
    assert hit["source"] == "modal_call_id_ledger"
    assert hit["archive_bytes"] == 456
    assert hit["score"] == pytest.approx(0.234)
    assert hit["runtime_tree_sha256"] == "b" * 64


def test_lookup_refuses_runtime_tree_mismatch_when_expected(tmp_path):
    sha = "baddcafe" * 8
    _write_canonical_cuda_anchor(
        tmp_path,
        archive_sha256=sha,
        runtime_tree_sha256="a" * 64,
    )
    assert (
        find_promotable_anchor_for_axis_and_sha(
            "cuda",
            sha,
            repo_root=tmp_path,
            expected_runtime_tree_sha256="b" * 64,
        )
        is None
    )
    hit = find_promotable_anchor_for_axis_and_sha(
        "cuda",
        sha,
        repo_root=tmp_path,
        expected_runtime_tree_sha256="a" * 64,
    )
    assert hit is not None
    assert hit["runtime_tree_sha256"] == "a" * 64


def test_lookup_ledger_preserves_falsey_numeric_zero_score(tmp_path):
    sha = "0" * 64
    _write_modal_call_id_ledger_row(
        tmp_path,
        {
            "call_id": "fc-zero-score",
            "status": "harvested",
            "harvest_result": {
                "evidence_grade": "[contest-CUDA]",
                "score_claim_valid": True,
                "score_axis": "contest_cuda",
                "scorer_device": "cuda",
                "n_samples": 600,
                "archive_sha256": sha,
                "eval_archive_size_bytes": 789,
                "score": 0,
            },
        },
    )

    hit = find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path)

    assert hit is not None
    assert hit["source"] == "modal_call_id_ledger"
    assert hit["score"] == 0.0


def test_lookup_refuses_advisory_macos_payload_for_cpu_axis(tmp_path):
    """Catalog #192 sister: macOS-CPU-advisory MUST NOT be reused as contest-CPU."""
    sha = "999abcde" * 8
    _write_canonical_cpu_anchor(tmp_path, archive_sha256=sha, grade="macOS-CPU-advisory only")
    hit = find_promotable_anchor_for_axis_and_sha("cpu", sha, repo_root=tmp_path)
    assert hit is None


def test_lookup_refuses_axis_mismatch(tmp_path):
    """CUDA anchor must not satisfy a CPU lookup and vice versa."""
    sha = "aaaa1111" * 8
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha)
    assert find_promotable_anchor_for_axis_and_sha("cpu", sha, repo_root=tmp_path) is None
    assert find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path) is not None


def test_lookup_refuses_sha_mismatch(tmp_path):
    sha_a = "aaaa1111" * 8
    sha_b = "bbbb2222" * 8
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha_a)
    assert find_promotable_anchor_for_axis_and_sha("cuda", sha_b, repo_root=tmp_path) is None


def test_lookup_refuses_incomplete_custody(tmp_path):
    """Missing evidence_grade means no anchor."""
    sha = "deadc0de" * 8
    lane_dir = tmp_path / "experiments" / "results" / "lane_partial_modal" / "harvested_artifacts"
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                # NO evidence_grade
                "score_claim_valid": True,
                "archive_sha256": sha,
                "score": 0.21,
            }
        )
    )
    assert find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path) is None


def test_lookup_refuses_score_claim_false(tmp_path):
    sha = "f00dba01" * 8
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha, score_claim_valid=False)
    assert find_promotable_anchor_for_axis_and_sha("cuda", sha, repo_root=tmp_path) is None


def test_lookup_returns_none_for_invalid_inputs(tmp_path):
    """Unknown axis, empty sha, or non-string sha returns None."""
    assert find_promotable_anchor_for_axis_and_sha("mps", "abc", repo_root=tmp_path) is None
    assert find_promotable_anchor_for_axis_and_sha("cuda", "", repo_root=tmp_path) is None
    assert find_promotable_anchor_for_axis_and_sha("cuda", None, repo_root=tmp_path) is None  # type: ignore[arg-type]


def test_lookup_returns_none_when_repo_root_missing(tmp_path):
    missing = tmp_path / "nonexistent_dir"
    assert find_promotable_anchor_for_axis_and_sha("cuda", "abc", repo_root=missing) is None


# CLI integration tests: dispatch_modal_paired_auth_eval.py --skip-axis...


def _run_cli(repo_root: Path, archive: Path, *extra_args: str) -> subprocess.CompletedProcess:
    """Invoke the CLI in plan-only mode against a tmp repo_root."""
    cmd = [
        sys.executable,
        str(TOOL_PATH),
        "--archive",
        str(archive),
        "--label",
        "test_pair",
        *extra_args,
    ]
    # We deliberately leave --execute OFF so the CLI never actually spawns
    # Modal. Plan-only mode emits the JSON dict to stdout; we parse it.
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))


def test_cli_default_disabled_no_skip_status_in_plan(tmp_path):
    """Default behavior: skip-flag absent means axes_skipped is all False."""
    archive, sha = _make_archive_file(tmp_path)
    # Pre-create a CUDA anchor, but with the flag off it must not skip.
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha)
    proc = _run_cli(tmp_path, archive)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["skip_axis_if_promotable_anchor_exists"] is False
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is False
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is False
    assert plan["existing_anchors_reused"]["contest_cuda"] is None
    assert plan["existing_anchors_reused"]["contest_cpu"] is None


def test_cli_flag_true_with_cuda_anchor_skips_cuda_only(tmp_path):
    """Anchor exists for CUDA only: cuda_skipped=True / cpu_skipped=False."""
    archive, sha = _make_archive_file(tmp_path)
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha, score=0.19869)
    proc = _run_cli(
        tmp_path,
        archive,
        "--skip-axis-if-promotable-anchor-exists",
        "--expected-runtime-tree-sha256",
        "a" * 64,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["skip_axis_if_promotable_anchor_exists"] is True
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is True
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is False
    assert plan["existing_anchors_reused"]["contest_cuda"] is not None
    assert plan["existing_anchors_reused"]["contest_cuda"]["score"] == pytest.approx(0.19869)
    assert plan["existing_anchors_reused"]["contest_cpu"] is None


def test_cli_flag_true_with_cpu_anchor_skips_cpu_only(tmp_path):
    archive, sha = _make_archive_file(tmp_path)
    _write_canonical_cpu_anchor(tmp_path, archive_sha256=sha, score=0.19663)
    proc = _run_cli(
        tmp_path,
        archive,
        "--skip-axis-if-promotable-anchor-exists",
        "--expected-runtime-tree-sha256",
        "a" * 64,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is False
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is True


def test_cli_flag_true_with_both_anchors_marks_both_skipped(tmp_path):
    """Both axes have anchors, so the plan reports both skipped."""
    archive, sha = _make_archive_file(tmp_path)
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha)
    _write_canonical_cpu_anchor(tmp_path, archive_sha256=sha)
    proc = _run_cli(
        tmp_path,
        archive,
        "--skip-axis-if-promotable-anchor-exists",
        "--expected-runtime-tree-sha256",
        "a" * 64,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is True
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is True


def test_cli_flag_true_with_neither_anchor_dispatches_normally(tmp_path):
    archive, _ = _make_archive_file(tmp_path)
    # No anchors written
    proc = _run_cli(
        tmp_path,
        archive,
        "--skip-axis-if-promotable-anchor-exists",
        "--expected-runtime-tree-sha256",
        "a" * 64,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is False
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is False
    # Commands are still in the plan (would dispatch on --execute)
    assert plan["commands"]["contest_cuda"]
    assert plan["commands"]["contest_cpu"]


def test_cli_skip_axis_when_anchor_grade_is_advisory_dispatches(tmp_path):
    """Advisory anchor must not cause skip; dispatch still fires."""
    archive, sha = _make_archive_file(tmp_path)
    _write_canonical_cpu_anchor(tmp_path, archive_sha256=sha, grade="macOS-CPU-advisory only")
    proc = _run_cli(
        tmp_path,
        archive,
        "--skip-axis-if-promotable-anchor-exists",
        "--expected-runtime-tree-sha256",
        "a" * 64,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is False
    assert plan["existing_anchors_reused"]["contest_cpu"] is None


def test_cli_skip_flag_without_runtime_sha_does_not_reuse_runtime_blind_anchor(tmp_path):
    archive, sha = _make_archive_file(tmp_path)
    _write_canonical_cuda_anchor(tmp_path, archive_sha256=sha)
    proc = _run_cli(tmp_path, archive, "--skip-axis-if-promotable-anchor-exists")
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is False
    assert plan["existing_anchors_reused"]["contest_cuda"] is None
    assert any(
        "runtime-bound anchor reuse is disabled" in note
        for note in plan["notes"]
    )
