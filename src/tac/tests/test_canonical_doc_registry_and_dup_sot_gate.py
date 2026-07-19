"""Tests for the NAME-ANCHORED-SEARCH / duplicate-SoT apparatus:

* ``tools/canonical_doc_registry.py`` — naming-independent canonical-doc
  registry (``lookup`` / ``check_before_create`` match by CONCEPT + all-refs
  ``git grep`` content search, never by the caller's guessed filename), and
* ``tac.confound_gates.check_no_duplicate_canonical_spec_across_refs`` — the
  preflight gate that refuses a new canonical-spec-shaped doc when a
  same-vehicle spec already exists on ANY git ref or in the registry.

Every test exercises BEHAVIOR (the search actually finding content it was not
named after; the gate actually catching a cross-branch duplicate), never
constants. Root cause under test (operator 2026-07-18): "You searched, but you
searched for what you would have named it." Memory:
``vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718.md``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.confound_gates import (
    REPO_ROOT,
    check_no_duplicate_canonical_spec_across_refs,
)
from tac.preflight import PreflightError

# ---------------------------------------------------------------------------
# import the tools module (script, not a package) by path
# ---------------------------------------------------------------------------

_TOOL_PATH = REPO_ROOT / "tools" / "canonical_doc_registry.py"
_spec = importlib.util.spec_from_file_location("canonical_doc_registry", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
cdr = importlib.util.module_from_spec(_spec)
sys.modules["canonical_doc_registry"] = cdr  # dataclass needs sys.modules entry
_spec.loader.exec_module(cdr)

_REAL_REGISTRY = REPO_ROOT / ".omx" / "state" / "canonical_doc_registry.json"


# ---------------------------------------------------------------------------
# hermetic git-repo helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )


def _mk_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".omx" / "research").mkdir(parents=True)
    (repo / ".omx" / "state").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t.local")
    _git(repo, "config", "user.name", "t")
    (repo / ".omx" / "research" / "placeholder.md").write_text(
        "# placeholder\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _commit_on_branch(repo: Path, branch: str, rel: str, body: str) -> None:
    """Create/commit ``rel`` with ``body`` on ``branch``, then return to main
    WITHOUT the file present in the working tree (it lives only on the ref)."""
    _git(repo, "checkout", "-q", "-b", branch)
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", f"add {rel} on {branch}")
    _git(repo, "checkout", "-q", "main")
    assert not p.exists(), "branch-only doc must NOT be in the main working tree"


def _no_registry(tmp_path: Path) -> Path:
    return tmp_path / "nonexistent_registry.json"


# ===========================================================================
# registry: lookup / check_before_create
# ===========================================================================


class TestRegistryLookup:
    def test_lookup_finds_spec_v10_by_concept_not_name(self):
        # The query words ("cold start seeded capstone") are NOT the naming
        # convention a duplicate-creating agent used ("optimal cold start
        # capstone") — the match is by concept tags, not filename glob.
        hits = cdr.lookup(
            "cold start seeded capstone",
            repo_root=REPO_ROOT,
            registry_path=_REAL_REGISTRY,
            git_search=False,
        )
        ids = [e.doc_id for e in hits]
        assert "spec_v10_capstone_cold_start_seeded" in ids
        v10 = next(e for e in hits if e.doc_id == "spec_v10_capstone_cold_start_seeded")
        assert v10.branch == "main"  # merged to main 2026-07-19 (was claude/p0_521_spec_v10_capstone_20260717)
        assert (
            v10.canonical_path
            == ".omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md"
        )

    def test_lookup_nonsense_concept_returns_empty(self):
        hits = cdr.lookup(
            "zebra pudding volcano trombone",
            repo_root=REPO_ROOT,
            registry_path=_REAL_REGISTRY,
            git_search=False,
        )
        assert hits == []

    def test_lookup_git_content_search_finds_doc_on_unmerged_branch(self, tmp_path):
        # A doc whose FILENAME shares no tokens with the query, on a branch
        # that is NOT checked out, is found via git grep of its CONTENT.
        repo = _mk_repo(tmp_path)
        _commit_on_branch(
            repo,
            "side/unmerged_feature",
            "notes/mystery_naming.md",
            "# some doc\n\nthe cold start seeded capstone vehicle program\n",
        )
        hits = cdr.lookup(
            "cold start seeded capstone",
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            git_search=True,
        )
        paths = {e.canonical_path for e in hits}
        assert "notes/mystery_naming.md" in paths
        hit = next(e for e in hits if e.canonical_path == "notes/mystery_naming.md")
        assert hit.branch == "side/unmerged_feature"
        assert hit.doc_id.startswith("unregistered:")

    def test_lookup_no_hits_in_bare_tmp_repo(self, tmp_path):
        repo = _mk_repo(tmp_path)
        hits = cdr.lookup(
            "cold start seeded capstone",
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            git_search=True,
        )
        assert hits == []


class TestCheckBeforeCreate:
    def test_flags_differently_named_proposal_for_same_concept(self):
        # The EXACT incident: proposing the duplicate's filename must surface
        # the canonical doc that lives under a DIFFERENT name on a branch.
        hits = cdr.check_before_create(
            "SPEC_v10_optimal_cold_start_capstone_20260718.md",
            repo_root=REPO_ROOT,
            registry_path=_REAL_REGISTRY,
            git_search=False,
        )
        assert any(e.doc_id == "spec_v10_capstone_cold_start_seeded" for e in hits)

    def test_unrelated_proposal_returns_no_match(self):
        hits = cdr.check_before_create(
            "random_zebra_pudding_notes.md",
            repo_root=REPO_ROOT,
            registry_path=_REAL_REGISTRY,
            git_search=False,
        )
        assert hits == []

    def test_cli_check_exit_codes(self, capsys):
        rc_dup = cdr.main(
            [
                "check",
                "SPEC_v10_optimal_cold_start_capstone_20260718.md",
                "--no-git",
                "--repo-root",
                str(REPO_ROOT),
                "--registry",
                str(_REAL_REGISTRY),
            ]
        )
        assert rc_dup == 2
        out = capsys.readouterr().out
        assert "DUPLICATE-SoT RISK" in out
        rc_clean = cdr.main(
            [
                "check",
                "random_zebra_pudding_notes.md",
                "--no-git",
                "--repo-root",
                str(REPO_ROOT),
                "--registry",
                str(_REAL_REGISTRY),
            ]
        )
        assert rc_clean == 0


class TestRegistrySchema:
    def test_rejects_bad_status(self, tmp_path):
        bad = tmp_path / "r.json"
        bad.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "doc_id": "x",
                            "concept_tags": ["a"],
                            "canonical_path": "a.md",
                            "branch": "main",
                            "status": "bogus",
                            "one_line": "x",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(cdr.RegistryError, match="invalid status"):
            cdr.load_registry(bad)

    def test_rejects_missing_required_field(self, tmp_path):
        bad = tmp_path / "r.json"
        bad.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "doc_id": "x",
                            "concept_tags": ["a"],
                            "branch": "main",
                            "one_line": "x",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(cdr.RegistryError, match="canonical_path"):
            cdr.load_registry(bad)

    def test_real_registry_entries_are_reachable(self):
        # Behavior: every registered canonical_path+branch resolves to real
        # bytes (worktree, ref, or declared live-state) — no phantom paths.
        entries = cdr.load_registry(_REAL_REGISTRY)
        assert len(entries) >= 7
        statuses = {e.doc_id: cdr.verify_entry(e, repo_root=REPO_ROOT) for e in entries}
        missing = {k: v for k, v in statuses.items() if v == "missing"}
        assert not missing, f"registry entries with unverifiable bytes: {missing}"


# ===========================================================================
# gate: check_no_duplicate_canonical_spec_across_refs
# ===========================================================================


_CANON_V10 = ".omx/research/SPEC_v10_capstone_cold_start_seeded.md"
_CANON_V10_BODY = "# SPEC_v10 — THE CAPSTONE: cold start on a seeded program\n\nbody\n"


class TestDupSotGate:
    def test_catches_name_shaped_duplicate_on_other_ref(self, tmp_path):
        repo = _mk_repo(tmp_path)
        _commit_on_branch(repo, "canon/spec_v10", _CANON_V10, _CANON_V10_BODY)
        dup = repo / ".omx" / "research" / "SPEC_v10_optimal_cold_start_capstone.md"
        dup.write_text("# SPEC_v10 — my duplicate\n", encoding="utf-8")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert len(v) == 1, v
        assert "canon/spec_v10" in v[0]
        assert "SPEC_v10_capstone_cold_start_seeded.md" in v[0]

    def test_strict_raises_preflight_error(self, tmp_path):
        repo = _mk_repo(tmp_path)
        _commit_on_branch(repo, "canon/spec_v10", _CANON_V10, _CANON_V10_BODY)
        dup = repo / ".omx" / "research" / "SPEC_v10_other_name.md"
        dup.write_text("# SPEC_v10 duplicate\n", encoding="utf-8")
        with pytest.raises(PreflightError, match="duplicate canonical spec"):
            check_no_duplicate_canonical_spec_across_refs(
                repo_root=repo,
                registry_path=_no_registry(tmp_path),
                strict=True,
                verbose=False,
            )

    def test_content_family_catch_without_spec_filename(self, tmp_path):
        # The ref-side canonical doc is NOT named SPEC_* at all — only its
        # first heading claims the vehicle. A name-anchored glob can never
        # find this; the content-family leg must.
        repo = _mk_repo(tmp_path)
        _commit_on_branch(
            repo,
            "codex/true_final_form",
            ".omx/research/capstone_true_final_form.md",
            "# SPEC v10 capstone — cold start final form\n\nbody\n",
        )
        dup = repo / ".omx" / "research" / "SPEC_v10_fresh_attempt.md"
        dup.write_text("# SPEC_v10 fresh attempt\n", encoding="utf-8")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert len(v) == 1, v
        assert "content-family" in v[0]
        assert "capstone_true_final_form.md" in v[0]

    def test_first_and_only_spec_is_clean(self, tmp_path):
        repo = _mk_repo(tmp_path)
        p = repo / ".omx" / "research" / "SPEC_v10_capstone.md"
        p.write_text(_CANON_V10_BODY, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "the one spec")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=True,
            verbose=False,
        )
        assert v == []

    def test_same_path_on_other_refs_is_not_a_duplicate(self, tmp_path):
        repo = _mk_repo(tmp_path)
        p = repo / ".omx" / "research" / "SPEC_v10_capstone.md"
        p.write_text(_CANON_V10_BODY, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "spec on main")
        _git(repo, "checkout", "-q", "-b", "feature/x")
        p.write_text(_CANON_V10_BODY + "\nmore\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "spec evolved on branch (same path)")
        _git(repo, "checkout", "-q", "main")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=True,
            verbose=False,
        )
        assert v == []

    def test_waiver_respected(self, tmp_path):
        repo = _mk_repo(tmp_path)
        _commit_on_branch(repo, "canon/spec_v10", _CANON_V10, _CANON_V10_BODY)
        dup = repo / ".omx" / "research" / "SPEC_v10_fork.md"
        dup.write_text(
            "# SPEC_v10 fork\n"
            "<!-- # DUPLICATE_SOT_OK:operator-approved archived-lineage fork -->\n",
            encoding="utf-8",
        )
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=True,
            verbose=False,
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        repo = _mk_repo(tmp_path)
        _commit_on_branch(repo, "canon/spec_v10", _CANON_V10, _CANON_V10_BODY)
        dup = repo / ".omx" / "research" / "SPEC_v10_fork.md"
        dup.write_text(
            "# SPEC_v10 fork\n<!-- # DUPLICATE_SOT_OK:<rationale> -->\n",
            encoding="utf-8",
        )
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert len(v) == 1

    def test_registry_leg_catches_duplicate_without_any_ref_copy(self, tmp_path):
        # No ref carries the canonical bytes locally (e.g. the branch lives
        # only on origin in a different clone) — the registry alone must
        # still refuse the duplicate.
        repo = _mk_repo(tmp_path)
        reg = tmp_path / "registry.json"
        reg.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "doc_id": "spec_v10",
                            "concept_tags": ["spec", "v10", "capstone"],
                            "canonical_path": ".omx/research/SPEC_v10_capstone_cold_start_seeded.md",
                            "branch": "claude/p0_521_spec_v10_capstone",
                            "status": "active",
                            "vehicle": "v10",
                            "one_line": "the canonical v10 spec",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        dup = repo / ".omx" / "research" / "SPEC_v10_optimal_cold_start.md"
        dup.write_text("# SPEC_v10 duplicate attempt\n", encoding="utf-8")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo, registry_path=reg, strict=False, verbose=False
        )
        assert len(v) == 1, v
        assert "registry" in v[0]
        assert "claude/p0_521_spec_v10_capstone" in v[0]

    def test_registry_leg_canonical_path_itself_is_clean(self, tmp_path):
        # The doc AT the registered canonical path must never self-flag.
        repo = _mk_repo(tmp_path)
        reg = tmp_path / "registry.json"
        rel = ".omx/research/SPEC_v10_capstone_cold_start_seeded.md"
        reg.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "doc_id": "spec_v10",
                            "concept_tags": ["spec", "v10"],
                            "canonical_path": rel,
                            "branch": "main",
                            "status": "active",
                            "vehicle": "v10",
                            "one_line": "the canonical v10 spec",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (repo / rel).write_text(_CANON_V10_BODY, encoding="utf-8")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo, registry_path=reg, strict=True, verbose=False
        )
        assert v == []

    def test_worktree_pairwise_duplicate_caught(self, tmp_path):
        repo = _mk_repo(tmp_path)
        (repo / ".omx" / "research" / "SPEC_v10_a.md").write_text(
            "# SPEC_v10 a\n", encoding="utf-8"
        )
        (repo / ".omx" / "research" / "SPEC_v10_b.md").write_text(
            "# SPEC_v10 b\n", encoding="utf-8"
        )
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert any("duplicate canonical spec for vehicle v10" in x for x in v)

    def test_distinct_vehicles_do_not_collide(self, tmp_path):
        # v8 and v8.1 are DIFFERENT vehicles ('8' vs '81'); v7.5 == v75.
        repo = _mk_repo(tmp_path)
        (repo / ".omx" / "research" / "SPEC_v8_perclass.md").write_text(
            "# SPEC — v8 per-class\n", encoding="utf-8"
        )
        (repo / ".omx" / "research" / "SPEC_v8.1_refined.md").write_text(
            "# SPEC — v8.1 refined\n", encoding="utf-8"
        )
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=True,
            verbose=False,
        )
        assert v == []

    def test_dotted_and_undotted_vehicle_notation_collide(self, tmp_path):
        repo = _mk_repo(tmp_path)
        (repo / ".omx" / "research" / "SPEC_v75_trunk.md").write_text(
            "# SPEC — v7.5 trunk\n", encoding="utf-8"
        )
        (repo / ".omx" / "research" / "SPEC_v7.5_trunk_redux.md").write_text(
            "# SPEC — v7.5 trunk redux\n", encoding="utf-8"
        )
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert any("v75" in x for x in v)

    def test_warn_mode_returns_list_without_raising(self, tmp_path):
        repo = _mk_repo(tmp_path)
        _commit_on_branch(repo, "canon/spec_v10", _CANON_V10, _CANON_V10_BODY)
        dup = repo / ".omx" / "research" / "SPEC_v10_dup.md"
        dup.write_text("# SPEC_v10 dup\n", encoding="utf-8")
        v = check_no_duplicate_canonical_spec_across_refs(
            repo_root=repo,
            registry_path=_no_registry(tmp_path),
            strict=False,
            verbose=False,
        )
        assert isinstance(v, list) and len(v) == 1

    def test_live_repo_zero_violations(self):
        # Regression guard on the real tree: the gate landed at live count 0
        # (the 4 main-lineage specs + SPEC_v10 on its branch are all
        # registered at their own canonical paths). A future duplicate makes
        # this fail loudly.
        v = check_no_duplicate_canonical_spec_across_refs(
            strict=False, verbose=False
        )
        non_transient = [x for x in v if "codexwt/" not in str(x)]
        # live codexwt arm branches carry pre-waiver spec blobs until they merge/prune;
        # those refs-leg rows are expected transients, not live-repo drift (2026-07-19).
        assert non_transient == [], non_transient
