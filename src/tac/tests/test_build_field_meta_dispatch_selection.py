from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tools.build_field_meta_dispatch_selection import build_selection_report

REPO = Path(__file__).resolve().parents[3]


def test_field_meta_selector_reports_static_ready_but_not_dispatch_ready_without_claim(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="closed_candidate",
        lane_id="lane_closed_candidate",
        job_name="job_closed_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready"] is True
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["strict_candidate_preflight_ready"] is True
    assert row["strict_candidate_static_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "passed"
    assert row["dispatch_claim_proof"]["checked"] is False
    assert "missing_active_lane_dispatch_claim" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "matching_active_level2_lane_claim_for_manifest_lane_and_job",
    ]


def test_field_meta_selector_requires_lane_and_job_identity_for_static_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="identity_missing_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_local_preflight_ready"] is True
    assert report["candidate_local_preflight_ready_count"] == 1
    assert report["candidate_static_preflight_ready"] is False
    assert report["candidate_static_preflight_ready_count"] == 0
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_local_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "blocked"
    assert row["dispatch_identity_proof"]["blockers"] == [
        "dispatch_lane_id_missing",
        "dispatch_instance_job_id_missing",
    ]
    assert "missing_dispatch_identity_for_lane_claim" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "manifest_lane_id_and_instance_job_id_for_level2_claim",
    ]


def test_field_meta_selector_requires_matching_active_claim_for_dispatch_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="claimed_candidate",
        lane_id="lane_claimed_candidate",
        job_name="job_claimed_candidate",
        kkt_proof=_kkt_proof(),
    )
    claims = _claims_file(tmp_path, lane_id="lane_claimed_candidate", job_name="job_claimed_candidate")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        claims_path=claims,
        now_utc="2026-05-06T12:00:00Z",
    )

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_candidate_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is True
    assert report["field_selection_ready_for_exact_eval_dispatch"] is True
    assert report["field_selection_ready_for_exact_eval_dispatch_count"] == 1
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["field_selection_ready_for_exact_eval_dispatch"] is True
    assert row["exact_dispatch_blockers"]["blocker_count"] == 0
    assert row["dispatch_claim_proof"]["checked"] is True
    assert row["dispatch_claim_proof"]["active_lane_claim"] is True
    assert row["dispatch_claim_proof"]["active_claim"]["instance_job_id"] == "job_claimed_candidate"
    assert row["next_required_proof"] == [
        "exact_cuda_auth_eval_on_selected_archive_bytes",
    ]


def test_field_meta_selector_keeps_active_claim_blocked_by_missing_kkt_for_field_selection(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="claimed_no_kkt",
        lane_id="lane_claimed_no_kkt",
        job_name="job_claimed_no_kkt",
    )
    claims = _claims_file(tmp_path, lane_id="lane_claimed_no_kkt", job_name="job_claimed_no_kkt")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        claims_path=claims,
        now_utc="2026-05-06T12:00:00Z",
    )

    row = report["rows"][0]
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["field_selection_ready_for_exact_eval_dispatch"] is False
    assert report["field_selection_ready_for_exact_eval_dispatch_count"] == 0
    assert "kkt:kkt_proof_or_admm_result_missing" in row["exact_dispatch_blockers"]["blockers"]
    assert (
        "passed_kkt_proof_or_converged_admm_waterline_result"
        in row["exact_dispatch_blockers"]["next_required_proof"]
    )


def test_field_meta_selector_rejects_non_zip_archive_even_with_matching_sha(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="not_a_zip",
        lane_id="lane_not_a_zip",
        job_name="job_not_a_zip",
        valid_zip=False,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["candidate_static_preflight_ready"] is False
    assert row["archive_proof"]["byte_closed"] is False
    assert row["archive_proof"]["zip_custody"]["status"] == "blocked"
    assert "archive:zip:archive_zip_unreadable" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "local_archive_file_with_matching_sha256_and_bytes"
    ]


def test_field_meta_selector_rejects_packet_without_runtime_tree_even_when_strict_preflight_passes(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="missing_runtime_tree",
        runtime_tree_sha256=None,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "missing_byte_closed_runtime_proof" in row["candidate_blockers"]
    assert row["next_required_proof"] == [
        "runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract"
    ]


def test_field_meta_selector_never_overrides_strict_candidate_preflight(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="strict_blocked_candidate",
        dispatch_gate="planning_only/no_remote_dispatch",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["strict_candidate_preflight_ready"] is False
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert any(
        blocker.startswith("strict:dispatch_gate_blocked")
        for blocker in row["candidate_blockers"]
    )


def test_field_meta_selector_is_deterministic_and_orders_static_ready_packets_first(
    tmp_path: Path,
) -> None:
    blocked = _packet_manifest(
        tmp_path / "blocked",
        candidate_id="blocked",
        runtime_tree_sha256=None,
    )
    ready = _packet_manifest(
        tmp_path / "ready",
        candidate_id="ready",
        expected_score_delta=-0.0002,
        lane_id="lane_ready",
        job_name="job_ready",
    )

    first = build_selection_report(repo_root=REPO, manifest_paths=[blocked, ready])
    second = build_selection_report(repo_root=REPO, manifest_paths=[ready, blocked])

    assert first == second
    assert first["selected_candidate"]["candidate_id"] == "ready"
    assert [row["candidate_id"] for row in first["rows"]] == ["ready", "blocked"]
    assert first["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert first["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_exposes_pareto_kkt_and_information_gain(
    tmp_path: Path,
) -> None:
    common = {
        "lane_id": "lane_pareto",
        "job_name": "job_pareto",
        "family_group": "hnerv_rate_recode",
        "pareto_scope": "hnerv_rate_recode",
        "interaction_assumptions": ["rate_only_byte_equivalent"],
        "interaction_model": "first_order_volterra_rate_recode",
        "volterra_order": 1,
        "volterra_terms": ["rate_only_linear_section"],
        "kkt_proof": _kkt_proof(),
    }
    dominator = _packet_manifest(
        tmp_path / "dominator",
        candidate_id="dominator",
        expected_score_delta=-0.0003,
        byte_delta=-30,
        expected_seg_dist_delta=-0.0002,
        expected_pose_dist_delta=-0.00002,
        expected_information_gain_nats=0.2,
        **common,
    )
    dominated = _packet_manifest(
        tmp_path / "dominated",
        candidate_id="dominated",
        expected_score_delta=-0.0001,
        byte_delta=-10,
        expected_seg_dist_delta=-0.0001,
        expected_pose_dist_delta=-0.00001,
        expected_information_gain_nats=0.9,
        **common,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[dominated, dominator])

    assert report["pareto_summary"]["frontier_count"] == 1
    assert report["pareto_summary"]["dominated_count"] == 1
    assert report["kkt_ready_for_field_planning_count"] == 2
    assert report["selected_candidate"]["candidate_id"] == "dominator"
    first = report["rows"][0]
    assert first["pareto_frontier"] is True
    assert first["field_interaction_contract"]["status"] == "passed"
    assert first["field_interaction_contract"]["assumptions"] == ["rate_only_byte_equivalent"]
    assert first["field_interaction_contract"]["interaction_model"] == "first_order_volterra_rate_recode"
    assert first["field_interaction_contract"]["volterra_order"] == 1
    assert first["field_interaction_contract"]["volterra_terms"] == ["rate_only_linear_section"]
    assert first["non_dominated_frontier_reason"]["schema"] == "non_dominated_frontier_reason_v1"
    assert first["non_dominated_frontier_reason"]["status"] == "non_dominated"
    assert first["non_dominated_frontier_reason"]["reason"] == "non_dominated_within_pareto_scope"
    assert first["kkt_ready_for_field_planning"] is True
    assert first["lexicographic_feasibility_tuple"][6] is True
    assert first["field_selection_score_delta"] == first["expected_total_score_delta"]
    assert first["expected_information_gain_nats"] == 0.2
    assert first["exact_dispatch_blockers"]["schema"] == "exact_dispatch_blockers_v1"
    assert "missing_active_lane_dispatch_claim" in first["exact_dispatch_blockers"]["blockers"]
    second = report["rows"][1]
    assert second["candidate_id"] == "dominated"
    assert second["pareto_frontier"] is False
    assert second["pareto_dominated_by"] == ["dominator"]
    assert second["non_dominated_frontier_reason"]["status"] == "dominated"
    assert second["non_dominated_frontier_reason"]["dominated_by"] == ["dominator"]
    assert second["selection_penalty_terms"]["pareto_dominated_packet"] > 0.0
    assert "pareto_dominated_within_scope" in second["exact_dispatch_blockers"]["blockers"]


def test_field_meta_selector_requires_real_kkt_proof_before_kkt_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="no_kkt_proof",
        lane_id="lane_no_kkt_proof",
        job_name="job_no_kkt_proof",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["kkt_ready_for_field_planning_count"] == 0
    assert row["candidate_static_preflight_ready"] is True
    assert row["kkt_ready_for_field_planning"] is False
    assert row["kkt_proof"]["status"] == "blocked"
    assert "kkt:kkt_proof_or_admm_result_missing" in row["kkt_blockers"]
    assert row["field_interaction_contract"]["status"] == "passed"
    assert "kkt:kkt_proof_or_admm_result_missing" in row["exact_dispatch_blockers"]["blockers"]
    assert row["selection_penalty_terms"]["kkt_not_ready_for_field_planning"] > 0.0
    assert row["field_selection_score_delta"] == row["expected_total_score_delta"]


def test_field_meta_selector_accepts_converged_admm_result_as_kkt_proof(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="admm_kkt_ready",
        lane_id="lane_admm_kkt_ready",
        job_name="job_admm_kkt_ready",
        admm_result={
            "converged": True,
            "waterline_kkt_residual": 0.002,
            "kkt_waterline_satisfied": True,
        },
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["kkt_ready_for_field_planning_count"] == 1
    assert row["kkt_ready_for_field_planning"] is True
    assert row["kkt_proof"]["kind"] == "admm_result"
    assert row["lexicographic_feasibility_tuple"][6] is True


def test_field_meta_selector_penalizes_dirty_packet_over_raw_delta(
    tmp_path: Path,
) -> None:
    dirty = _packet_manifest(
        tmp_path / "dirty",
        candidate_id="dirty_raw_best",
        lane_id="lane_dirty_raw_best",
        job_name="job_dirty_raw_best",
        expected_score_delta=-10.0,
        code_paths=["src/tac/optimization/dirty_packet_owner.py"],
        interaction_assumptions=["dirty_candidate_pending_other_worker"],
    )
    clean = _packet_manifest(
        tmp_path / "clean",
        candidate_id="clean_weaker",
        lane_id="lane_clean_weaker",
        job_name="job_clean_weaker",
        expected_score_delta=-0.0001,
        interaction_assumptions=["clean_candidate_first_order"],
    )

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[dirty, clean],
        dirty_paths=["src/tac/optimization/dirty_packet_owner.py"],
    )

    assert report["dirty_blocked_candidate_count"] == 1
    assert report["selected_candidate"]["candidate_id"] == "clean_weaker"
    dirty_row = next(row for row in report["rows"] if row["candidate_id"] == "dirty_raw_best")
    assert dirty_row["dirty_blocked"] is True
    assert dirty_row["candidate_static_preflight_ready"] is False
    assert dirty_row["selection_decision"] == "refused_dirty_worktree_overlap"
    assert dirty_row["selection_penalty_terms"]["dirty_path_blocked"] > 0.0
    assert dirty_row["field_selection_score_delta"] == dirty_row["expected_total_score_delta"]
    assert dirty_row["next_required_proof"][0] == "clean_or_isolate_dirty_candidate_paths_before_selection"


def test_build_field_meta_dispatch_selection_cli_writes_json(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="cli_candidate",
        lane_id="lane_cli_candidate",
        job_name="job_cli_candidate",
    )
    out = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_field_meta_dispatch_selection.py"),
            "--manifest",
            str(manifest),
            "--json-out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    assert payload["selected_candidate"]["candidate_id"] == "cli_candidate"
    assert payload["candidate_static_preflight_ready_count"] == 1
    assert payload["ready_candidate_count"] == 0
    assert payload["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert payload["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def _packet_manifest(
    root: Path,
    *,
    candidate_id: str,
    runtime_tree_sha256: str | None = "b" * 64,
    dispatch_gate: str = "eligible_for_cuda_auth_eval_after_lane_claim",
    expected_score_delta: float = -0.0001,
    byte_delta: int = -9,
    expected_seg_dist_delta: float = 0.0,
    expected_pose_dist_delta: float = 0.0,
    expected_information_gain_nats: float = 0.1,
    family_group: str = "fixture_family",
    pareto_scope: str = "fixture_family",
    interaction_assumptions: list[str] | None = None,
    interaction_model: str | None = None,
    volterra_order: int | None = None,
    volterra_terms: list[str] | None = None,
    code_paths: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    kkt_proof: dict[str, object] | None = None,
    admm_result: dict[str, object] | None = None,
    lane_id: str | None = None,
    job_name: str | None = None,
    valid_zip: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    if valid_zip:
        info = zipfile.ZipInfo("x")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o644 << 16
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(info, f"{candidate_id} archive bytes".encode())
    else:
        archive.write_bytes(f"{candidate_id} archive bytes".encode())
    fixed_runtime = {
        "ready_for_fixed_runtime_exact_eval": True,
        "remaining_blockers": [],
    }
    if runtime_tree_sha256 is not None:
        fixed_runtime["runtime_tree_sha256"] = runtime_tree_sha256
    payload = {
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": dispatch_gate,
        "dispatch_unlocked": True,
        "ready_for_exact_eval_dispatch_claim": True,
        "family_group": family_group,
        "pareto_scope": pareto_scope,
        "interaction_assumptions": interaction_assumptions or ["fixture_first_order"],
        "archive": {
            "path": archive.as_posix(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size,
        },
        "fixed_runtime_preflight": fixed_runtime,
        "byte_delta": byte_delta,
        "expected_total_score_delta": expected_score_delta,
        "expected_seg_dist_delta": expected_seg_dist_delta,
        "expected_pose_dist_delta": expected_pose_dist_delta,
        "expected_information_gain_nats": expected_information_gain_nats,
    }
    if interaction_model is not None:
        payload["interaction_model"] = interaction_model
    if volterra_order is not None:
        payload["volterra_order"] = volterra_order
    if volterra_terms is not None:
        payload["volterra_terms"] = volterra_terms
    if code_paths is not None:
        payload["code_paths"] = code_paths
    if evidence_paths is not None:
        payload["evidence_paths"] = evidence_paths
    if kkt_proof is not None:
        payload["kkt_proof"] = kkt_proof
    if admm_result is not None:
        payload["admm_result"] = admm_result
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if job_name is not None:
        payload["job_name"] = job_name
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _kkt_proof() -> dict[str, object]:
    return {
        "status": "passed",
        "stationarity_residual": 0.0,
        "stationarity_tolerance": 0.001,
    }


def _claims_file(root: Path, *, lane_id: str, job_name: str) -> Path:
    claims = root / "active_lane_dispatch_claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims - test fixture",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                f"| 2026-05-06T11:55:00Z | codex:test | {lane_id} | lightning | {job_name} | 2026-05-06T12:30:00Z | active_exact_eval | test claim |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
