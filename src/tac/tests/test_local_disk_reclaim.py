# SPDX-License-Identifier: MIT
"""Fail-closed controls for the boot-volume certify-or-block reclaimer.

Two of these are regression tests for bugs that shipped in the first revision
and were caught only because the planner's output was read rather than trusted:

* ``test_ancestor_reference_does_not_pin_a_child`` -- an ancestor rule plus the
  repo root appearing in every ``ps`` line classified 100% of candidates as
  blocked. A fully-blocked census is a vacuous PASS, not caution.
* ``test_ref_pointing_at_a_tree_object_is_present_not_absent`` -- codex writes
  ``refs/codex/turn-diffs/*`` straight at TREE objects. Testing them with
  ``cat-file -e <obj>^{commit}`` fails even though the main repo holds every
  byte, which mis-classified 21 GiB of reclaimable trees as needing a copy.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "tools" / "local_disk_reclaim.py"
    name = "_dk1_local_disk_reclaim"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__];
    # registering before exec keeps `git: GitProof | None` from blowing up.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rec():
    return _load()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout


def _seed_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True
    )
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("payload\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-qm", "seed")
    return root


# --------------------------------------------------------------------- boundary


@pytest.mark.parametrize(
    "path",
    [
        "upstream/videos/0.mkv",
        "submissions/exact_current/inflate.py",
        ".omx/tmp/x/retained/candidates/archive.zip",
        "experiments/results/modal_auth_eval_mirror/x.json",
        "experiments/results/ddm_fr2_final_review_20260903/memo.md",
        ".claude/worktrees/agent-abc/src",
        ".omx/tmp/codex_runs/ng4_wait_then_fire.done",
        ".omx/tmp/codex_runs/arm.last.txt",
        ".omx/tmp/gt_cache_0_600.pt",
        ".omx/tmp/scorer_backend_torch_ref.npz",
        ".omx/tmp/arm_receipts_local/x/retained",
        ".omx/tmp/x/seals",
    ],
)
def test_never_touch_boundary_covers_every_charter_class(rec, path):
    assert rec.is_never_touch(path) is True


@pytest.mark.parametrize(
    "path",
    [
        ".omx/tmp/codex_worktrees/ddm_v4_stratum",
        ".omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/advisory",
        "/Users/x/pact/.omx/tmp/codex_runs/run.log",
    ],
)
def test_ordinary_bulk_is_not_swept_into_never_touch(rec, path):
    assert rec.is_never_touch(path) is False


def test_retained_matches_as_a_component_not_as_a_substring(rec):
    # "retained" the directory is protected; "retained_notes" the sibling is not.
    assert rec.is_never_touch(".omx/tmp/a/retained/b") is True
    assert rec.is_never_touch(".omx/tmp/a/retained_notes") is False


# ------------------------------------------------------------------------ pins


@pytest.mark.parametrize(
    "status,terminal",
    [
        ("completed_modal_auth_eval_harvested", True),
        ("failed_oom", True),
        ("stopped_by_operator", True),
        ("refused_dispatch_governor", True),
        ("stale_superseded_reconciled", True),
        ("active_eval", False),
        ("active_modal_auth_eval_spawned", False),
    ],
)
def test_claim_terminality_releases_only_closed_lanes(rec, status, terminal):
    assert rec.claim_is_terminal(status) is terminal


def test_active_claim_pins_its_paths_and_terminal_row_releases_them(rec):
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 600))
    md = (
        f"| {stamp} | main | lane_a | local | job1 |  | active_eval | "
        f"store /Volumes/VertigoDataTier/pact/live_arm |\n"
        f"| {stamp} | main | lane_b | local | job2 |  | completed_ok | "
        f"store /Volumes/VertigoDataTier/pact/closed_arm |\n"
    )
    pinned = rec.active_claim_paths(md, now_epoch=now)
    assert "/Volumes/VertigoDataTier/pact/live_arm" in pinned
    assert "/Volumes/VertigoDataTier/pact/closed_arm" not in pinned


def test_claim_older_than_the_ttl_no_longer_pins(rec):
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 48 * 3600))
    md = f"| {stamp} | main | lane_a | local | j |  | active_eval | /Volumes/x/old_arm |\n"
    assert rec.active_claim_paths(md, now_epoch=now, ttl_hours=24.0) == set()


def test_claim_stamps_are_read_as_utc_not_local_time(rec):
    """The 'Z' in the claim stamp is load-bearing.

    Reading it with mktime (local) and patching with time.timezone is wrong
    under DST, which silently slides the 24 h pin window by an hour.
    """
    # 2026-09-04T12:00:00Z == 1788523200 epoch. A claim 1 s inside a 1 h TTL
    # must still pin; the same row 1 s outside must not.
    md_at = "2026-09-04T12:00:00Z"
    row = f"| {md_at} | main | lane | local | j |  | active_eval | /Volumes/x/arm |\n"
    epoch = 1788523200
    assert rec.active_claim_paths(row, now_epoch=epoch + 3599, ttl_hours=1.0)
    assert rec.active_claim_paths(row, now_epoch=epoch + 3601, ttl_hours=1.0) == set()


def test_exact_reference_pins_the_candidate(rec):
    assert rec.path_is_pinned(Path("/a/b/c"), {"/a/b/c"}) is True


def test_reference_inside_the_candidate_pins_it(rec):
    # Deleting /a/b/c would destroy the live referent /a/b/c/live.pt.
    assert rec.path_is_pinned(Path("/a/b/c"), {"/a/b/c/live.pt"}) is True


def test_ancestor_reference_does_not_pin_a_child(rec):
    # Regression: the repo root appears in essentially every command line, so an
    # ancestor rule blocked the entire census and reported it as safety.
    assert rec.path_is_pinned(Path("/a/b/c"), {"/a/b"}) is False
    assert rec.path_is_pinned(Path("/a/b/c"), {"/"}) is False


def test_live_process_scan_excludes_this_tools_own_command_line(rec):
    ps = "111 python tools/local_disk_reclaim.py --roots /a/b/c\n222 python other.py /a/b/keepme\n"
    pinned = rec.live_process_paths(ps, self_pids={111})
    assert "/a/b/c" not in pinned
    assert "/a/b/keepme" in pinned


# ------------------------------------------------------------------ git proofs


def test_clean_worktree_with_present_refs_is_reconstructible(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
    proof = rec.probe_git_proof(wt, repo, {str(wt.resolve())})
    assert proof.is_git and proof.kind == "worktree"
    assert proof.porcelain_lines == 0
    assert proof.unreachable_refs == []
    assert proof.reconstructible is True


def test_uncommitted_bytes_block_reconstructibility(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
    (wt / "only_here.txt").write_text("exists nowhere else\n")
    proof = rec.probe_git_proof(wt, repo, {str(wt.resolve())})
    assert proof.porcelain_lines == 1
    assert proof.reconstructible is False


def test_ref_pointing_at_a_tree_object_is_present_not_absent(rec, tmp_path):
    # Regression: codex refs/codex/turn-diffs/* point at TREE objects. Probing
    # them with ^{commit} fails though the repo holds them -- that false BLOCK
    # mis-planned 21 GiB of reclaimable trees as a copy.
    repo = _seed_repo(tmp_path / "main")
    tree_sha = _git(repo, "rev-parse", "HEAD^{tree}").strip()
    _git(repo, "update-ref", "refs/codex/turn-diffs/1/base", tree_sha)
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
    proof = rec.probe_git_proof(wt, repo, {str(wt.resolve())})
    assert tree_sha not in proof.unreachable_refs
    assert proof.reconstructible is True


def test_commit_absent_from_the_main_repo_blocks_deletion(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(repo), str(clone)], check=True, capture_output=True
    )
    _git(clone, "config", "user.email", "t@example.invalid")
    _git(clone, "config", "user.name", "t")
    (clone / "b.txt").write_text("only in the clone\n")
    _git(clone, "add", "b.txt")
    _git(clone, "commit", "-qm", "clone-only work")
    proof = rec.probe_git_proof(clone, repo, set())
    assert proof.kind == "clone"
    assert proof.porcelain_lines == 0  # clean, but the commit exists nowhere else
    assert proof.unreachable_refs
    assert proof.reconstructible is False


def test_bulk_containing_a_retained_subtree_is_blocked_not_moved(rec, tmp_path):
    """A container of protected bytes is itself protected.

    Regression: ddm_mst1_manufactured_stage_split has an innocuous top-level
    name and 20.55 GiB living entirely under capture_r2_local/retained/. Judging
    only the candidate's own path marked it movable.
    """
    repo = _seed_repo(tmp_path / "main")
    bulk = tmp_path / "bulk_arm"
    (bulk / "capture" / "retained").mkdir(parents=True)
    (bulk / "capture" / "retained" / "payload.bin").write_bytes(b"\0" * 64)
    cand = rec.classify(bulk, repo=repo, registered=set(), pinned=set())
    assert cand.klass == rec.CLASS_BLOCKED_NEVER_TOUCH
    assert "contains a never-touch path" in cand.reason


def test_gt_cache_inside_bulk_blocks_the_whole_container(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    bulk = tmp_path / "bulk_arm2"
    (bulk / "sub").mkdir(parents=True)
    (bulk / "sub" / "gt_cache.npz").write_bytes(b"\0" * 8)
    cand = rec.classify(bulk, repo=repo, registered=set(), pinned=set())
    assert cand.klass == rec.CLASS_BLOCKED_NEVER_TOUCH


def test_clean_worktree_is_not_blocked_by_its_own_upstream_directory(rec, tmp_path):
    """Class A must survive the descendant scan.

    Every checkout contains upstream/ and submissions/. In a clean tree whose
    refs the main repo holds, those bytes are git's -- blocking on them would
    make the whole reconstructible class unreachable.
    """
    repo = _seed_repo(tmp_path / "main")
    (repo / "upstream").mkdir()
    (repo / "upstream" / "evaluate.py").write_text("# pinned\n")
    _git(repo, "add", "upstream/evaluate.py")
    _git(repo, "commit", "-qm", "add upstream")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
    assert (wt / "upstream" / "evaluate.py").is_file()
    cand = rec.classify(wt, repo=repo, registered={str(wt.resolve())}, pinned=set())
    assert cand.klass == rec.CLASS_GIT_RECONSTRUCTIBLE


def test_unscannably_large_bulk_fails_closed(rec, tmp_path):
    bulk = tmp_path / "wide"
    bulk.mkdir()
    for i in range(12):
        (bulk / f"f{i}").write_bytes(b"x")
    hit = rec.find_never_touch_descendant(bulk, limit=5)
    assert hit is not None and "not provably clear" in hit


def test_a_non_git_directory_is_never_deletable_here(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    (bulk / "big.raw").write_bytes(b"\0" * 1024)
    cand = rec.classify(bulk, repo=repo, registered=set(), pinned=set())
    assert cand.klass == rec.CLASS_CERTIFY_MOVE_REQUIRED
    assert "not a git tree" in cand.reason


def test_git_probe_failure_fails_closed_rather_than_reconstructible(rec, tmp_path):
    proof = rec.GitProof(is_git=True, kind="clone", head="abc", porcelain_lines=0, n_refs=3)
    assert proof.reconstructible is True
    proof.error = "for-each-ref failed"
    assert proof.reconstructible is False


# ------------------------------------------------------------------ cert rows


def test_never_touch_candidate_is_classified_without_measuring_it(rec, tmp_path, monkeypatch):
    repo = _seed_repo(tmp_path / "main")
    victim = tmp_path / "retained"
    victim.mkdir()

    def explode(_p):  # du must never run against a protected tree
        raise AssertionError("du ran on a never-touch path")

    monkeypatch.setattr(rec, "du_kib", explode)
    cand = rec.classify(victim, repo=repo, registered=set(), pinned=set())
    assert cand.klass == rec.CLASS_BLOCKED_NEVER_TOUCH


def test_cert_row_carries_the_full_charter_schema(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
    cand = rec.classify(wt, repo=repo, registered={str(wt.resolve())}, pinned=set())
    assert cand.klass == rec.CLASS_GIT_RECONSTRUCTIBLE
    row = cand.cert_row(repo=repo, executor="test")
    for key in (
        "schema",
        "utc",
        "original_path",
        "allocated_kib",
        "class",
        "rebuildable_reason",
        "git_proof",
        "rebuild_command",
    ):
        assert key in row, key
    assert row["schema"] == rec.SCHEMA
    assert row["rebuild_command"][:3] == ["git", "worktree", "add"]


def test_clone_rebuild_command_differs_from_worktree_rebuild_command(rec, tmp_path):
    repo = _seed_repo(tmp_path / "main")
    proof = rec.GitProof(is_git=True, kind="clone", head="deadbeef", porcelain_lines=0, n_refs=1)
    cand = rec.Candidate(tmp_path / "c", 10, rec.CLASS_GIT_RECONSTRUCTIBLE, "r", proof)
    assert cand.cert_row(repo=repo, executor="t")["rebuild_command"][:2] == ["git", "clone"]


def test_plan_and_apply_are_mutually_exclusive(rec, capsys):
    assert rec.main(["--roots", ".", "--plan", "--apply"]) == 2
    assert "exactly one of --plan or --apply" in capsys.readouterr().err


def test_ledger_rows_are_one_json_object_per_line(rec, tmp_path):
    ledger = tmp_path / "certs.jsonl"
    rec.append_ledger(ledger, {"a": 1})
    rec.append_ledger(ledger, {"b": 2})
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    assert [json.loads(x) for x in lines] == [{"a": 1}, {"b": 2}]


# --- ddm_dk2: vacuity guard -------------------------------------------------
# A census that blocks every candidate prints "0.00 GiB in 0 tree(s)" and reads
# as a considered PASS. dk1 shipped that shape once (ancestor-pin defect); dk2
# hit it again from a different door -- a concurrent read-only `du -sk <root>/*`
# names every child on its command line, so the live-process scan pins the whole
# tree. These pin the denominator report, not the pin rule itself.


def _cand(rec, klass, reason):
    return rec.Candidate(Path("/x"), 0, klass, reason)


def test_vacuity_report_names_the_denominator_when_work_exists(rec):
    lines = rec.vacuity_report(
        [
            _cand(rec, rec.CLASS_CERTIFY_MOVE_REQUIRED, "bulk"),
            _cand(rec, rec.CLASS_BLOCKED_NEVER_TOUCH, rec.REASON_PINNED),
            _cand(rec, rec.CLASS_BLOCKED_NEVER_TOUCH, "matches never-touch boundary"),
        ]
    )
    assert lines[0] == (
        "census denominator: 1 reclaimable / 1 pinned / 1 never-touch / 3 candidates"
    )
    assert not any("VACUOUS-CENSUS" in line for line in lines)


def test_fully_blocked_census_is_reported_as_vacuous_not_as_a_clean_pass(rec):
    lines = rec.vacuity_report(
        [_cand(rec, rec.CLASS_BLOCKED_NEVER_TOUCH, "matches never-touch boundary")] * 4
    )
    assert any("VACUOUS-CENSUS" in line for line in lines)
    assert any("0 of 4 candidates are reclaimable" in line for line in lines)


def test_pin_dominated_vacuous_census_names_the_concurrent_census_cause(rec):
    lines = rec.vacuity_report([_cand(rec, rec.CLASS_BLOCKED_NEVER_TOUCH, rec.REASON_PINNED)] * 5)
    joined = "\n".join(lines)
    assert "VACUOUS-CENSUS" in joined
    assert "5/5 were pinned" in joined
    assert "du/ls/find" in joined


def test_never_touch_dominated_vacuity_does_not_blame_a_concurrent_census(rec):
    lines = rec.vacuity_report(
        [_cand(rec, rec.CLASS_BLOCKED_NEVER_TOUCH, "matches never-touch boundary")] * 5
    )
    assert not any("du/ls/find" in line for line in lines)


def test_empty_root_reports_a_zero_denominator_without_a_bug_claim(rec):
    lines = rec.vacuity_report([])
    assert lines == ["census denominator: 0 candidates under the given --roots"]


def test_pin_reason_constant_is_the_string_classify_actually_emits(rec, tmp_path):
    # The guard counts rows by reason; a drift between the constant and the
    # emitted string would silently zero the "pinned" tally.
    tree = tmp_path / "pinned_tree"
    tree.mkdir()
    cand = rec.classify(tree, repo=tmp_path, registered=set(), pinned={str(tree)})
    assert cand.reason == rec.REASON_PINNED
    assert cand.klass == rec.CLASS_BLOCKED_NEVER_TOUCH
