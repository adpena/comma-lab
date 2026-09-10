"""Disqualification custody and admission regressions; all state is fixture-local."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac import canonical_frontier_pointer as cfp
from tac.frontier_disqualifications import (
    active_disqualifications,
    append_disqualification,
    disqualification_for,
    require_qualified,
)
from tac.frontier_scan import build_frontier_scan_payload, collect_all_anchors

SHA_A = "a" * 64
SHA_B = "b" * 64
JOURNAL = ".omx/state/frontier_disqualifications.jsonl"
POINTER = ".omx/state/canonical_frontier_pointer.json"
RATIONALE = "Receiver embeds video-selected lane constants outside the counted archive."
REINSTATE = "Counted archive now carries the constants and the receiver review passed."
REPO = Path(__file__).resolve().parents[3]


def _json(root: Path, rel: str, value: object) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ban(root: Path, lane: str = "lane_bad", sha: str = SHA_A, **updates):
    evidence = root / "evidence.md"
    evidence.write_text("Audited receiver stores video-derived constants in free code.\n")
    args = {
        "lane_id": lane,
        "archive_sha256": sha,
        "reason_class": "rule118_content_in_code",
        "evidence": str(evidence),
        "rationale": RATIONALE,
        "who": "fixture-reviewer",
    }
    args.update(updates)
    return append_disqualification(root, **args)


def _mirror(root: Path, lane: str, sha: str, score: float = 0.14, axis: str = "contest_cuda"):
    return _json(
        root,
        f"experiments/results/modal_auth_eval_mirror/contest_auth_eval_{lane}.json",
        {
            "schema": "modal_auth_eval_anchor_mirror.v2",
            "score": score,
            "score_axis": axis,
            "archive_sha256": sha,
            "lane_id": lane,
            "hardware_substrate": "linux_x86_64_cpu" if axis == "contest_cpu" else "linux_x86_64_t4",
            "runtime_tree_sha256": "c" * 64,
            "archive_size_bytes": 180233,
            "evidence_grade": "contest-CPU" if axis == "contest_cpu" else "contest-CUDA",
        },
    )


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"cpd1_{name}", REPO / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate(root: Path, *, strict: bool = False):
    from tac.preflight import check_frontier_excludes_disqualified_rows

    return check_frontier_excludes_disqualified_rows(repo_root=root, strict=strict, verbose=False)


def test_missing_journal_is_empty_and_does_not_create_state(tmp_path):
    assert active_disqualifications(tmp_path) == []
    require_qualified(tmp_path, "lane_good", SHA_A)
    assert not (tmp_path / JOURNAL).exists()


def test_append_preserves_exact_prefix_and_reason_evidence(tmp_path):
    first = _ban(tmp_path)
    prefix = (tmp_path / JOURNAL).read_bytes()
    _ban(tmp_path, "lane_other", SHA_B, reason_class="custody")
    assert (tmp_path / JOURNAL).read_bytes().startswith(prefix)
    assert len((tmp_path / JOURNAL).read_text().splitlines()) == 2
    assert first["reason_class"] == "rule118_content_in_code"
    assert Path(first["evidence"]).is_file()
    assert first["rationale"] == RATIONALE
    assert first["who"] == "fixture-reviewer"
    assert len(active_disqualifications(tmp_path)) == 2


@pytest.mark.parametrize("reason", ["rule118_content_in_code", "decode_budget", "determinism", "custody"])
def test_all_declared_reason_classes_are_accepted(tmp_path, reason):
    assert _ban(tmp_path, reason_class=reason)["reason_class"] == reason


@pytest.mark.parametrize("rationale", ["", "TODO", "TBD", "placeholder", "n/a"])
def test_placeholder_rationale_cannot_mutate_journal(tmp_path, rationale):
    with pytest.raises(ValueError):
        _ban(tmp_path, rationale=rationale)
    assert not (tmp_path / JOURNAL).exists() or not (tmp_path / JOURNAL).read_bytes()


@pytest.mark.parametrize("sha", ["", "a" * 63, "g" * 64, "a" * 65])
def test_invalid_archive_identity_is_rejected(tmp_path, sha):
    with pytest.raises(ValueError):
        _ban(tmp_path, sha=sha)


def test_empty_lane_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _ban(tmp_path, lane="")


def test_unknown_reason_class_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _ban(tmp_path, reason_class="operator_dislikes_score")


def test_missing_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _ban(tmp_path, evidence="missing-review.md")


def test_reinstatement_appends_and_preserves_original_reason(tmp_path):
    _ban(tmp_path)
    before = (tmp_path / JOURNAL).read_bytes()
    append_disqualification(
        tmp_path, lane_id="lane_bad", archive_sha256=SHA_A, rationale=REINSTATE, who="fixture-reviewer", reinstate=True
    )
    assert active_disqualifications(tmp_path) == []
    assert (tmp_path / JOURNAL).read_bytes().startswith(before)
    history = [json.loads(line) for line in (tmp_path / JOURNAL).read_text().splitlines()]
    assert [event["action"] for event in history] == ["disqualify", "reinstate"]
    assert history[0]["rationale"] == RATIONALE
    require_qualified(tmp_path, "lane_bad", SHA_A)


def test_unknown_reinstatement_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_disqualification(tmp_path, lane_id="lane_bad", archive_sha256=SHA_A, rationale=REINSTATE, reinstate=True)


def test_reinstatement_does_not_clear_other_archive_on_same_lane(tmp_path):
    _ban(tmp_path)
    _ban(tmp_path, sha=SHA_B)
    append_disqualification(tmp_path, lane_id="lane_bad", archive_sha256=SHA_A, rationale=REINSTATE, reinstate=True)
    active = active_disqualifications(tmp_path)
    assert len(active) == 1 and active[0]["archive_sha256"] == SHA_B


def test_repeated_ban_cycle_is_not_permanently_reinstated(tmp_path):
    _ban(tmp_path)
    append_disqualification(tmp_path, lane_id="lane_bad", archive_sha256=SHA_A, rationale=REINSTATE, reinstate=True)
    _ban(tmp_path, reason_class="decode_budget")
    assert active_disqualifications(tmp_path)[0]["reason_class"] == "decode_budget"
    assert len((tmp_path / JOURNAL).read_text().splitlines()) == 3


@pytest.mark.parametrize("contents", ["not-json\n", "[]\n", '{"action":"disqualify"}\n'])
def test_malformed_journal_fails_closed(tmp_path, contents):
    path = tmp_path / JOURNAL
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    with pytest.raises(ValueError):
        active_disqualifications(tmp_path)
    with pytest.raises(ValueError):
        _ban(tmp_path)
    assert path.read_text() == contents


def test_match_uses_pair_when_both_identity_fields_are_known(tmp_path):
    _ban(tmp_path)
    rows = active_disqualifications(tmp_path)
    assert disqualification_for(rows, "lane_bad", SHA_A)
    assert disqualification_for(rows, "lane_bad", SHA_B) is None
    assert disqualification_for(rows, "lane_other", SHA_A) is None


@pytest.mark.parametrize("lane,sha", [(None, SHA_A), ("lane_bad", None)])
def test_missing_anchor_identity_matches_known_field_fail_closed(tmp_path, lane, sha):
    _ban(tmp_path)
    assert disqualification_for(active_disqualifications(tmp_path), lane, sha)
    with pytest.raises(ValueError, match="disqualif"):
        require_qualified(tmp_path, lane, sha)


def test_projection_changes_only_loaded_anchor_not_historical_mirror(tmp_path):
    mirror = _mirror(tmp_path, "lane_bad", SHA_A)
    original = mirror.read_bytes()
    _ban(tmp_path)
    anchors = collect_all_anchors(tmp_path)
    assert len(anchors) == 1
    assert anchors[0].extra["disqualified"]
    assert not anchors[0].is_qualifying()
    assert mirror.read_bytes() == original


def test_six_better_disqualified_rows_cannot_shadow_healthy_seventh(tmp_path):
    for index in range(6):
        lane, sha = f"lane_blocked_{index}", f"{index + 1:064x}"
        _mirror(tmp_path, lane, sha, 0.13 + index * 0.001)
        _ban(tmp_path, lane, sha)
    _mirror(tmp_path, "lane_good", SHA_B, 0.14)
    scan = build_frontier_scan_payload(tmp_path)
    assert scan["best_per_axis"]["contest_cuda"]["extra"]["lane_id"] == "lane_good"
    pointer = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    assert pointer.our_local_frontier_contest_cuda.lane_id == "lane_good"
    assert pointer.effective_frontier["score"] == 0.14
    assert len(pointer.as_dict()["disqualified_rows"]) == 6
    assert pointer.refresh_provenance["disqualified_rows"]


@pytest.mark.parametrize("axis", ["contest_cpu", "contest_cuda"])
def test_disqualified_prior_is_not_preserved_when_no_replacement_exists(tmp_path, axis):
    _mirror(tmp_path, "lane_bad", SHA_A, axis=axis)
    prior = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=True)
    assert getattr(prior, f"our_local_frontier_{axis}")
    _ban(tmp_path)
    pointer = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    assert getattr(pointer, f"our_local_frontier_{axis}") is None
    assert pointer.effective_frontier is None
    assert pointer.as_dict()["disqualified_rows"]


def test_reinstatement_restores_selection(tmp_path):
    _mirror(tmp_path, "lane_bad", SHA_A, 0.13)
    _mirror(tmp_path, "lane_good", SHA_B, 0.14)
    _ban(tmp_path)
    assert (
        cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False).effective_frontier["score"]
        == 0.14
    )
    append_disqualification(tmp_path, lane_id="lane_bad", archive_sha256=SHA_A, rationale=REINSTATE, reinstate=True)
    assert (
        cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False).effective_frontier["score"]
        == 0.13
    )


def test_upstream_refresh_preserves_disqualification_audit_and_public_minimum(tmp_path):
    _mirror(tmp_path, "lane_bad", SHA_A, 0.13)
    _mirror(tmp_path, "lane_good", SHA_B, 0.15)
    _ban(tmp_path)
    snapshot = {
        "fetch_status": "ok",
        "fetched_at_utc": "2026-09-10T00:00:00Z",
        "best_entry": {"score": 0.14, "name": "public leader"},
    }
    pointer = cfp.refresh_canonical_frontier_from_upstream_leaderboard(
        repo_root=tmp_path, write=True, fetcher=lambda **_: snapshot
    )
    assert pointer.effective_frontier["score"] == 0.14
    assert pointer.our_local_frontier_contest_cuda.lane_id == "lane_good"
    assert pointer.as_dict()["disqualified_rows"]
    assert pointer.refresh_provenance["disqualified_rows"]
    local = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    assert local.upstream_leaderboard_snapshot == snapshot
    assert local.effective_frontier["score"] == 0.14


def test_dispatch_auto_refresh_skips_disqualified_row(tmp_path, monkeypatch):
    _mirror(tmp_path, "lane_bad", SHA_A, 0.13)
    _mirror(tmp_path, "lane_good", SHA_B, 0.14)
    _ban(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "tac.master_gradient_mlx_pipeline",
        SimpleNamespace(auto_schedule_mlx_per_pair_extraction_for_frontier=lambda **_: None),
    )
    pointer = cfp.auto_refresh_canonical_frontier_after_dispatch_outcome(
        status="harvested", score=0.13, archive_sha256=SHA_A, repo_root=tmp_path
    )
    assert pointer is not None
    assert pointer.our_local_frontier_contest_cuda.lane_id == "lane_good"
    assert json.loads((tmp_path / POINTER).read_text())["disqualified_rows"]


def test_preflight_accepts_empty_journal(tmp_path):
    assert _gate(tmp_path, strict=True) == []


def test_preflight_refuses_active_effective_frontier(tmp_path):
    _json(tmp_path, POINTER, {"effective_frontier": {"lane_id": "lane_bad", "archive_sha256": SHA_A}})
    _ban(tmp_path)
    violations = _gate(tmp_path)
    assert violations and "lane_bad" in str(violations)
    from tac.preflight import PreflightError

    with pytest.raises(PreflightError):
        _gate(tmp_path, strict=True)


def test_preflight_allows_different_archive_on_same_lane(tmp_path):
    _json(tmp_path, POINTER, {"effective_frontier": {"lane_id": "lane_bad", "archive_sha256": SHA_B}})
    _ban(tmp_path)
    assert _gate(tmp_path, strict=True) == []


@pytest.mark.parametrize(
    "waiver,allowed",
    [
        ("TODO", False),
        ("Historical pointer held for offline incident reconstruction; this file is not admission authority.", True),
    ],
)
def test_preflight_waiver_requires_substantive_rationale(tmp_path, waiver, allowed):
    _json(
        tmp_path,
        POINTER,
        {
            "effective_frontier": {"lane_id": "lane_bad", "archive_sha256": SHA_A},
            "frontier_disqualification_waiver": waiver,
        },
    )
    _ban(tmp_path)
    assert bool(_gate(tmp_path)) is not allowed
    with pytest.raises(ValueError):
        require_qualified(tmp_path, "lane_bad", SHA_A)


def test_packet_prior_refuses_disqualified_identity(tmp_path):
    _mirror(tmp_path, "lane_bad", SHA_A)
    cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=True)
    _ban(tmp_path)
    with pytest.raises(ValueError, match="disqualif"):
        _tool("pointer_move_packet").prior_anchor_from_pointer(tmp_path, "contest_cuda")


@pytest.mark.parametrize("override_lane", ["lane_bad", "lane_renamed", None])
def test_packet_candidate_refuses_before_any_stage_mutation(tmp_path, capsys, override_lane):
    _ban(tmp_path)
    harvest = _json(
        tmp_path,
        "harvest.json",
        {
            "avg_segnet_dist": 0.0001,
            "avg_posenet_dist": 0.00001,
            "archive_size_bytes": 180000,
            "expected_archive_sha256": SHA_A,
            "expected_runtime_tree_sha256": "c" * 64,
            "n_samples": 600,
            "score_axis": "contest_cuda",
            "passed": True,
            "validation_errors": [],
            "gpu_model": "Tesla T4",
            "lane_id": "lane_bad",
        },
    )
    rc = _tool("pointer_move_packet").main(
        [
            "--repo-root",
            str(tmp_path),
            "--harvest",
            str(harvest),
            *(["--lane-id", override_lane] if override_lane else []),
            "--move-number",
            "42",
            "--apply",
            "--no-custody",
        ]
    )
    output = capsys.readouterr()
    assert rc != 0 and "disqualif" in (output.out + output.err).lower()
    assert not (tmp_path / ".omx/state/pointer_move_events.jsonl").exists()
    assert not (tmp_path / POINTER).exists()


def test_cli_disqualification_and_reinstatement_roundtrip(tmp_path, capsys):
    evidence = tmp_path / "review.md"
    evidence.write_text("Measured receiver compliance violation.\n")
    cli = _tool("frontier_disqualify")
    common = ["--repo-root", str(tmp_path), "--lane", "lane_bad", "--archive-sha256", SHA_A]
    assert (
        cli.main(
            [
                *common,
                "--reason-class",
                "rule118_content_in_code",
                "--evidence",
                str(evidence),
                "--rationale",
                RATIONALE,
                "--who",
                "fixture-reviewer",
            ]
        )
        == 0
    )
    assert active_disqualifications(tmp_path)
    assert cli.main([*common, "--reinstate", "--rationale", REINSTATE]) == 0
    assert active_disqualifications(tmp_path) == []


def test_writer_rechecks_journal_after_pointer_computation(tmp_path):
    _mirror(tmp_path, "lane_bad", SHA_A)
    pointer = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    _ban(tmp_path)
    with pytest.raises(ValueError, match="disqualif"):
        cfp.write_canonical_frontier_pointer_locked(pointer, repo_root=tmp_path)
    assert not (tmp_path / POINTER).exists()


def test_preflight_rejects_corrupt_journal_even_with_audit_waiver(tmp_path):
    _json(tmp_path, POINTER, {"frontier_disqualification_waiver": RATIONALE})
    (tmp_path / JOURNAL).write_text("malformed journal\n")
    assert _gate(tmp_path)


def test_preflight_rejects_disqualified_local_anchor_when_upstream_wins(tmp_path):
    _json(
        tmp_path,
        POINTER,
        {
            "effective_frontier": {"source": "upstream_official_leaderboard", "score": 0.1},
            "our_local_frontier_contest_cuda": {"lane_id": "lane_bad", "archive_sha256": SHA_A},
        },
    )
    _ban(tmp_path)
    assert _gate(tmp_path)


def test_packet_blocks_concurrent_ban_until_publication_finishes(tmp_path, monkeypatch):
    from threading import Event, Thread

    harvest = _json(
        tmp_path,
        "harvest.json",
        {
            "avg_segnet_dist": 0.0001,
            "avg_posenet_dist": 0.00001,
            "archive_size_bytes": 180000,
            "expected_archive_sha256": SHA_A,
            "expected_runtime_tree_sha256": "c" * 64,
            "n_samples": 600,
            "score_axis": "contest_cuda",
            "passed": True,
            "validation_errors": [],
            "gpu_model": "Tesla T4",
            "lane_id": "lane_bad",
        },
    )
    packet = _tool("pointer_move_packet")
    attempted, finished = Event(), Event()

    def concurrent_ban():
        attempted.set()
        _ban(tmp_path)
        finished.set()

    writer = Thread(target=concurrent_ban, daemon=True)

    def run_with_concurrent_ban(cmd, *, cwd, apply, label):
        if label == "refresh pointer from local state":
            writer.start()
            assert attempted.wait(1)
            assert not finished.wait(0.05), "journal mutation interleaved with packet publication"
        return {"rc": 0, "label": label}

    monkeypatch.setattr(packet, "_run", run_with_concurrent_ban)
    try:
        rc = packet.main(
            [
                "--repo-root",
                str(tmp_path),
                "--harvest",
                str(harvest),
                "--lane-id",
                "lane_bad",
                "--move-number",
                "42",
                "--apply",
                "--no-custody",
            ]
        )
    finally:
        if writer.ident is not None:
            writer.join(timeout=2)
    assert rc == 0
    assert finished.is_set(), "journal writer did not resume after packet released its lock"
    assert active_disqualifications(tmp_path)
    assert (tmp_path / ".omx/state/pointer_move_events.jsonl").exists()


def test_packet_accepts_independently_identified_receiver_on_same_archive(tmp_path, monkeypatch):
    _ban(tmp_path, lane="lane_bad")
    harvest = _json(
        tmp_path,
        "harvest.json",
        {
            "avg_segnet_dist": 0.0001,
            "avg_posenet_dist": 0.00001,
            "archive_size_bytes": 180000,
            "expected_archive_sha256": SHA_A,
            "expected_runtime_tree_sha256": "d" * 64,
            "n_samples": 600,
            "score_axis": "contest_cuda",
            "passed": True,
            "validation_errors": [],
            "gpu_model": "Tesla T4",
            "lane_id": "lane_good",
        },
    )
    packet = _tool("pointer_move_packet")
    monkeypatch.setattr(packet, "_run", lambda *args, **kwargs: {"rc": 0})
    assert (
        packet.main(
            [
                "--repo-root",
                str(tmp_path),
                "--harvest",
                str(harvest),
                "--lane-id",
                "lane_good",
                "--move-number",
                "42",
                "--no-custody",
            ]
        )
        == 0
    )


def test_recompute_uses_pointer_disqualification_audit_without_anchor_projection(tmp_path):
    _mirror(tmp_path, "lane_bad", SHA_A)
    pointer = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    _ban(tmp_path)
    pointer = replace(pointer, disqualified_rows=active_disqualifications(tmp_path))
    assert not pointer.our_local_frontier_contest_cuda.extra.get("disqualified")
    assert cfp.recompute_effective_frontier(pointer) is None


@pytest.mark.parametrize("key", ["effective_frontier", "our_local_frontier_contest_cuda"])
def test_preflight_upstream_source_label_cannot_hide_local_disqualified_identity(tmp_path, key):
    _json(
        tmp_path,
        POINTER,
        {
            key: {"source": "upstream_official_leaderboard", "lane_id": "lane_bad", "archive_sha256": SHA_A},
        },
    )
    _ban(tmp_path)
    assert _gate(tmp_path)
