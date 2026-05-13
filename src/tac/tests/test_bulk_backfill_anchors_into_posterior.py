"""Tests for tools/bulk_backfill_anchors_into_posterior.py.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
the operator one-touch authorization toolkit landing 2026-05-11. These tests
cover:

- orphan detection (custody intact but not in posterior)
- Catalog #127 validation per orphan (accepts CUDA T4 + GHA CPU; refuses
  Modal CPU + macOS substrates)
- dry-run produces no posterior changes
- --commit produces locked writes per Catalog #128
- substrate-class signature correctly inferred from result-dir name
- idempotent re-run produces 0 changes
- /tmp paths refused per CLAUDE.md no-/tmp-path
- malformed/missing JSON gracefully classified as parse_error
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# Make the tools/ module importable in the test environment.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import bulk_backfill_anchors_into_posterior as bbf  # noqa: E402
from tac.continual_learning import (  # noqa: E402
    ContinualLearningPosterior,
    load_posterior,
    save_posterior,
)


# ── Fixtures: minimal contest_auth_eval.json payloads ─────────────────────────


def _cuda_t4_payload(sha: str = "1" * 64) -> dict:
    """Custody-VALID CUDA T4 payload (Linux x86_64, Tesla T4, gpu_t4_match)."""
    return {
        "score_axis": "contest_cuda",
        "device": "cuda",
        "canonical_score": 0.22650343,
        "avg_segnet_dist": 0.00064,
        "avg_posenet_dist": 0.000032,
        "archive_size_bytes": 178258,
        "n_samples": 600,
        "evidence_grade": "contest-CUDA",
        "lane_tag": "[contest-CUDA]",
        "score_claim_valid": True,
        "promotion_eligible": True,
        "provenance": {
            "archive_sha256": sha,
            "archive_size_bytes": 178258,
            "device": "cuda",
            "hardware": "Modal Tesla T4 Linux x86_64",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
        },
    }


def _modal_cpu_payload(sha: str = "2" * 64) -> dict:
    """Modal CPU payload — REFUSED per Catalog #127 (cpu_tag_non_gha_linux)."""
    return {
        "score_axis": "contest_cpu",
        "device": "cpu",
        "canonical_score": 0.22809238,
        "avg_segnet_dist": 0.00063,
        "avg_posenet_dist": 0.000164,
        "archive_size_bytes": 186822,
        "n_samples": 600,
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU]",
        "score_claim_valid": False,
        "promotion_eligible": False,
        "provenance": {
            "archive_sha256": sha,
            "archive_size_bytes": 186822,
            "device": "cpu",
            "hardware": "Modal CPU Linux x86_64",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
    }


def _gha_cpu_payload(sha: str = "3" * 64) -> dict:
    """Custody-VALID GHA CPU payload — should be promotable."""
    return {
        "score_axis": "contest_cpu_gha",
        "device": "cpu",
        "canonical_score": 0.19284758,
        "avg_segnet_dist": 0.00060,
        "avg_posenet_dist": 0.000150,
        "archive_size_bytes": 178262,
        "n_samples": 600,
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU GHA Linux x86_64]",
        "score_claim_valid": True,
        "promotion_eligible": True,
        "provenance": {
            "archive_sha256": sha,
            "archive_size_bytes": 178262,
            "device": "cpu",
            "hardware": "GHA Linux x86_64 CPU",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
    }


def _macos_payload(sha: str = "4" * 64) -> dict:
    """macOS payload — REFUSED per Catalog #127 (macos_substrate)."""
    return {
        "score_axis": "contest_cpu",
        "device": "cpu",
        "canonical_score": 0.19664189,
        "avg_segnet_dist": 0.00060,
        "avg_posenet_dist": 0.000150,
        "archive_size_bytes": 178262,
        "n_samples": 600,
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU]",
        "score_claim_valid": False,
        "provenance": {
            "archive_sha256": sha,
            "archive_size_bytes": 178262,
            "device": "cpu",
            "hardware": "macOS Apple Silicon",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
        },
    }


def _make_eval_dir(
    root: Path,
    name: str,
    payload: dict,
) -> Path:
    """Write contest_auth_eval.json into root/name/."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "contest_auth_eval.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return d / "contest_auth_eval.json"


# ── infer_architecture_class ──────────────────────────────────────────────────


def test_infer_architecture_class_pr101_lossy_coarsening():
    assert (
        bbf.infer_architecture_class(
            "pr101_a5_channel_qbits_dp_qsum200_exact_cuda_modal_20260510T194943Z"
        )
        == "pr101_lossy_coarsening"
    )


def test_infer_architecture_class_pr103_arithmetic_coding():
    assert (
        bbf.infer_architecture_class(
            "pr103_global_combo_12b_exact_cuda_modal_20260510T2257Z"
        )
        == "pr103_arithmetic_coding"
    )


def test_infer_architecture_class_pr103_on_pr106():
    assert (
        bbf.infer_architecture_class(
            "pr103_pr106_dual_runtime_cuda_v2_20260511T022553Z"
        )
        == "pr103_on_pr106"
    )


def test_infer_architecture_class_pr106_latent_sidecar_r1():
    assert (
        bbf.infer_architecture_class("pr106_latent_sidecar_20260511T150517Z")
        == "lane_pr106_latent_sidecar_r1"
    )


def test_infer_architecture_class_pr106_latent_sidecar_r2():
    assert (
        bbf.infer_architecture_class("pr106_latent_sidecar_r2_20260511T160358Z")
        == "lane_pr106_latent_sidecar_r2"
    )


def test_infer_architecture_class_pr106_latent_sidecar_r2_pr101_grammar():
    assert (
        bbf.infer_architecture_class(
            "pr106_latent_sidecar_r2_pr101_grammar_20260511T200000Z"
        )
        == "lane_pr106_latent_sidecar_r2_pr101_grammar"
    )


def test_infer_architecture_class_lane_c3_residual():
    assert (
        bbf.infer_architecture_class("lane_c3_residual_pr106_sidecar_20260511T203000Z")
        == "lane_c3_residual_pr106_sidecar_dispatch_ready"
    )


def test_infer_architecture_class_l2_sparse_aware_more_specific_wins():
    assert (
        bbf.infer_architecture_class(
            "lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch_20260511"
        )
        == "lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch"
    )


def test_infer_architecture_class_unknown_returns_unknown_prefix():
    assert (
        bbf.infer_architecture_class("brand_new_substrate_2027").startswith("unknown__")
    )


# ── discover_anchors: classification verdicts ────────────────────────────────


def test_discover_anchors_finds_zero_when_search_root_empty(tmp_path: Path):
    out = bbf.discover_anchors(tmp_path)
    assert out == []


def test_discover_anchors_classifies_cuda_t4_as_promotable_orphan(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_bias_refine_cuda_modal_test_20260510", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
        posterior_path=posterior_path,
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.custody_accepted is True
    assert a.custody_refused_class is None
    assert a.in_posterior_already is False
    assert a.is_promotable_orphan is True
    assert a.architecture_class == "pr101_lossy_coarsening"
    assert a.axis == "cuda"
    assert a.evidence_tag == "[contest-CUDA]"


def test_discover_anchors_classifies_modal_cpu_as_refused_cpu_tag_non_gha_linux(
    tmp_path: Path,
):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    _make_eval_dir(
        root, "pr106_latent_sidecar_r2_modal_test_20260511", _modal_cpu_payload()
    )
    posterior_path = tmp_path / "posterior.json"

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval_cpu",),
        posterior_path=posterior_path,
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.custody_accepted is False
    assert a.custody_refused_class == "cpu_tag_non_gha_linux"
    assert a.is_promotable_orphan is False
    assert a.axis == "cpu"


def test_discover_anchors_classifies_gha_cpu_as_promotable_orphan(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    _make_eval_dir(root, "a1_gha_cpu_test_20260509", _gha_cpu_payload())
    posterior_path = tmp_path / "posterior.json"

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval_cpu",),
        posterior_path=posterior_path,
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.custody_accepted is True
    assert a.is_promotable_orphan is True
    assert a.evidence_tag == "[contest-CPU GHA Linux x86_64]"
    assert a.hardware_substrate == "linux_x86_64_gha_cpu"


def test_discover_anchors_classifies_macos_as_refused_macos_substrate(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    _make_eval_dir(root, "a1_macos_test_20260509", _macos_payload())
    posterior_path = tmp_path / "posterior.json"

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval_cpu",),
        posterior_path=posterior_path,
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.custody_accepted is False
    # macOS evidence_grade=contest-CPU but hardware=Darwin -> tag flips to
    # advisory; the verdict is still macos_substrate (highest specificity).
    assert a.custody_refused_class == "macos_substrate"
    assert a.is_promotable_orphan is False


def test_discover_anchors_handles_corrupt_json_gracefully(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    d = root / "corrupt_test_20260511"
    d.mkdir(parents=True)
    (d / "contest_auth_eval.json").write_text("{ this is not valid json", encoding="utf-8")

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.parse_error is not None
    assert "JSONDecodeError" in a.parse_error
    assert a.is_promotable_orphan is False


def test_discover_anchors_handles_missing_required_field(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    payload = _cuda_t4_payload()
    payload["provenance"].pop("archive_sha256")
    payload.pop("provenance", None)  # also drop top-level fallbacks
    payload.pop("score_axis", None)
    _make_eval_dir(root, "missing_sha_test_20260511", payload)

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.parse_error is not None
    assert a.is_promotable_orphan is False


def test_discover_anchors_marks_already_in_posterior(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    sha = "5" * 64
    _make_eval_dir(root, "pr101_test_already_in_posterior", _cuda_t4_payload(sha))

    # Pre-populate the posterior with this anchor.
    posterior_path = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    posterior.accepted_anchor_history.append(
        {"axis": "cuda", "architecture_class": "pr101_lossy_coarsening", "archive_sha256": sha}
    )
    posterior.accepted_anchor_count = 1
    save_posterior(posterior, posterior_path)

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
        posterior_path=posterior_path,
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.in_posterior_already is True
    assert a.is_promotable_orphan is False  # already in posterior is NOT an orphan


# ── existing_posterior_anchor_keys ────────────────────────────────────────────


def test_existing_posterior_anchor_keys_returns_empty_set_when_missing(tmp_path: Path):
    p = tmp_path / "missing.json"
    assert bbf.existing_posterior_anchor_keys(p) == set()


def test_existing_posterior_anchor_keys_returns_pairs(tmp_path: Path):
    p = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    posterior.accepted_anchor_history.append(
        {"axis": "cuda", "archive_sha256": "a" * 64}
    )
    posterior.accepted_anchor_history.append(
        {"axis": "cpu", "archive_sha256": "b" * 64}
    )
    save_posterior(posterior, p)

    keys = bbf.existing_posterior_anchor_keys(p)
    assert ("a" * 64, "cuda") in keys
    assert ("b" * 64, "cpu") in keys


# ── summarize_anchors ────────────────────────────────────────────────────────


def test_summarize_anchors_counts_correctly(tmp_path: Path):
    root_cuda = tmp_path / "experiments" / "results" / "modal_auth_eval"
    root_cpu = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    # 2 CUDA T4 (promotable orphans) + 1 modal CPU (refused) + 1 GHA CPU
    # (promotable orphan) + 1 macOS (refused).
    _make_eval_dir(root_cuda, "pr101_test_a", _cuda_t4_payload("a" * 64))
    _make_eval_dir(root_cuda, "pr103_test_b", _cuda_t4_payload("b" * 64))
    _make_eval_dir(root_cpu, "pr106_modal_cpu_test", _modal_cpu_payload())
    _make_eval_dir(root_cpu, "a1_gha_cpu_test", _gha_cpu_payload())
    _make_eval_dir(root_cpu, "a1_macos_test", _macos_payload())

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=(
            "experiments/results/modal_auth_eval",
            "experiments/results/modal_auth_eval_cpu",
        ),
    )
    summary = bbf.summarize_anchors(anchors)

    assert summary["total_artifacts_discovered"] == 5
    assert summary["custody_accepted"] == 3
    assert summary["custody_refused"] == 2
    assert summary["promotable_orphans"] == 3
    assert summary["promotable_orphans_by_axis"]["cuda"] == 2
    assert summary["promotable_orphans_by_axis"]["cpu"] == 1
    assert summary["custody_refused_by_class"]["cpu_tag_non_gha_linux"] == 1
    assert summary["custody_refused_by_class"]["macos_substrate"] == 1


# ── render_dry_run_table ──────────────────────────────────────────────────────


def test_render_dry_run_table_emits_header_and_rows(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_test_render", _cuda_t4_payload())
    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
    )
    table = bbf.render_dry_run_table(anchors)
    lines = table.splitlines()
    assert "axis" in lines[0]
    assert "arch_class" in lines[0]
    assert "ORPHAN_PROMOTABLE" in lines[1]
    assert "pr101_lossy_coarsening" in lines[1]


def test_render_dry_run_table_handles_in_posterior_status(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    sha = "6" * 64
    _make_eval_dir(root, "pr101_in_posterior", _cuda_t4_payload(sha))

    posterior_path = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    posterior.accepted_anchor_history.append(
        {"axis": "cuda", "archive_sha256": sha}
    )
    save_posterior(posterior, posterior_path)

    anchors = bbf.discover_anchors(
        tmp_path,
        search_roots=("experiments/results/modal_auth_eval",),
        posterior_path=posterior_path,
    )
    table = bbf.render_dry_run_table(anchors)
    assert "in_posterior" in table


# ── Dry-run: no posterior changes ────────────────────────────────────────────


def test_dry_run_main_produces_no_posterior_changes(tmp_path: Path, capsys):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_dryrun_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    summary_json = tmp_path / "summary.json"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--summary-json",
            str(summary_json),
            "--quiet",
        ]
    )
    assert rc == 0
    assert not posterior_path.exists()  # dry run did NOT write the posterior

    payload = json.loads(summary_json.read_text())
    assert payload["mode"] == "dry_run"
    assert payload["discovery_summary"]["promotable_orphans"] == 1


def test_dry_run_then_posterior_unchanged(tmp_path: Path):
    """Pre-existing posterior is byte-identical after a dry run."""
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_dryrun_unchanged_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    posterior = ContinualLearningPosterior()
    save_posterior(posterior, posterior_path)
    before = posterior_path.read_bytes()

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--quiet",
        ]
    )
    assert rc == 0
    after = posterior_path.read_bytes()
    assert before == after


# ── Commit: locked writes ────────────────────────────────────────────────────


def test_commit_main_back_fills_into_posterior(tmp_path: Path, capsys):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_commit_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"
    audit_path = tmp_path / "audit.jsonl"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--lock-path",
            str(lock_path),
            "--audit-log-path",
            str(audit_path),
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 0
    assert posterior_path.exists()
    posterior = load_posterior(posterior_path)
    assert posterior.accepted_anchor_count == 1
    assert audit_path.exists()
    rows = [json.loads(L) for L in audit_path.read_text().splitlines() if L.strip()]
    assert len(rows) == 1
    assert rows[0]["action"] == "posterior_update_locked_called"
    assert rows[0]["posterior_update_accepted"] is True


def test_commit_idempotent_re_run_no_double_count(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_idempotent_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"
    audit_path = tmp_path / "audit.jsonl"

    args = [
        "--repo-root",
        str(tmp_path),
        "--search-root",
        "experiments/results/modal_auth_eval",
        "--posterior-path",
        str(posterior_path),
        "--lock-path",
        str(lock_path),
        "--audit-log-path",
        str(audit_path),
        "--commit",
        "--quiet",
    ]
    rc1 = bbf.main(args)
    assert rc1 == 0
    posterior_after_1 = load_posterior(posterior_path)
    assert posterior_after_1.accepted_anchor_count == 1

    rc2 = bbf.main(args)
    assert rc2 == 0
    posterior_after_2 = load_posterior(posterior_path)
    # The orphan is now in posterior, so the second run finds NO orphans;
    # the row is classified as 'skipped_already_in_posterior'.
    assert posterior_after_2.accepted_anchor_count == 1


def test_commit_skips_modal_cpu_with_audit_row(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval_cpu"
    _make_eval_dir(root, "pr106_modal_cpu_skip_test", _modal_cpu_payload())
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"
    audit_path = tmp_path / "audit.jsonl"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval_cpu",
            "--posterior-path",
            str(posterior_path),
            "--lock-path",
            str(lock_path),
            "--audit-log-path",
            str(audit_path),
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 0
    rows = [json.loads(L) for L in audit_path.read_text().splitlines() if L.strip()]
    assert len(rows) == 1
    assert rows[0]["action"] == "skipped_custody_refused"
    assert rows[0]["custody_refused_class"] == "cpu_tag_non_gha_linux"
    # No posterior update should have happened.
    if posterior_path.exists():
        posterior = load_posterior(posterior_path)
        assert posterior.accepted_anchor_count == 0


def test_commit_writes_one_jsonl_row_per_anchor(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_one", _cuda_t4_payload("a" * 64))
    _make_eval_dir(root, "pr103_two", _cuda_t4_payload("b" * 64))
    _make_eval_dir(root, "pr106_three", _cuda_t4_payload("c" * 64))
    audit_path = tmp_path / "audit.jsonl"
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--lock-path",
            str(lock_path),
            "--audit-log-path",
            str(audit_path),
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 0
    rows = [json.loads(L) for L in audit_path.read_text().splitlines() if L.strip()]
    assert len(rows) == 3
    posterior = load_posterior(posterior_path)
    assert posterior.accepted_anchor_count == 3


def test_commit_summary_json_contains_counts(tmp_path: Path):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_summary_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"
    audit_path = tmp_path / "audit.jsonl"
    summary_json = tmp_path / "commit_summary.json"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--lock-path",
            str(lock_path),
            "--audit-log-path",
            str(audit_path),
            "--summary-json",
            str(summary_json),
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 0
    payload = json.loads(summary_json.read_text())
    assert payload["mode"] == "commit"
    assert payload["commit_result"]["accepted"] == 1
    assert payload["commit_result"]["refused"] == 0


# ── /tmp guard per CLAUDE.md ──────────────────────────────────────────────────


def test_commit_refuses_tmp_audit_path(tmp_path: Path, capsys):
    root = tmp_path / "experiments" / "results" / "modal_auth_eval"
    _make_eval_dir(root, "pr101_tmp_guard_test", _cuda_t4_payload())
    posterior_path = tmp_path / "posterior.json"
    lock_path = tmp_path / "posterior.lock"

    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--search-root",
            "experiments/results/modal_auth_eval",
            "--posterior-path",
            str(posterior_path),
            "--lock-path",
            str(lock_path),
            "--audit-log-path",
            "/tmp/audit.jsonl",
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "/tmp" in captured.err
    assert not posterior_path.exists()


def test_commit_refuses_var_tmp_audit_path(tmp_path: Path, capsys):
    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--audit-log-path",
            "/var/tmp/audit.jsonl",
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "/tmp" in captured.err


def test_commit_without_audit_log_path_refused(tmp_path: Path, capsys):
    rc = bbf.main(
        [
            "--repo-root",
            str(tmp_path),
            "--commit",
            "--quiet",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "audit-log-path" in captured.err
