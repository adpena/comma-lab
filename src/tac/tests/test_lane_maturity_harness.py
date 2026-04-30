"""Regression tests for the Lane Maturity Harness.

Covers:
  - Registry load + schema validation
  - CLI mark / unmark / validate roundtrip
  - Computed level matches gate-count rules (4 cases)
  - Validate fails on missing evidence file
  - Validate fails on level/gates mismatch
  - Audit log JSONL append
  - Report generation
  - add-lane creates a new lane at L0

The CLI is exercised both via direct function calls (for fast unit-test
coverage) and via subprocess invocation (for one happy-path integration test).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import lane_maturity as lm

REPO = Path(__file__).resolve().parents[3]


# ── Fixture: build a fake repo + registry in a tmp dir ───────────────────


def _make_repo(tmp: Path, lanes: list[dict]) -> Path:
    """Create a minimal fake repo at tmp with .omx/state/lane_registry.json
    seeded from `lanes`. Returns the repo root."""
    (tmp / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "reports").mkdir(parents=True, exist_ok=True)

    registry = {
        "schema_version": 1,
        "updated_at": "2026-04-30T00:00:00Z",
        "gate_definitions": {g: f"<def of {g}>" for g in lm.GATES},
        "lanes": lanes,
    }
    (tmp / lm.REGISTRY_REL).write_text(json.dumps(registry, indent=2))
    return tmp


def _empty_gates() -> dict:
    return {g: {"status": False, "evidence": ""} for g in lm.GATES}


def _all_true_gates(repo: Path) -> dict:
    """All 7 gates true with evidence pointing to a file we create in `repo`."""
    f = repo / "src" / "tac" / "fake_module.py"
    f.write_text("# fake\n")
    return {g: {"status": True, "evidence": "src/tac/fake_module.py"} for g in lm.GATES}


# ── Test 1: registry load + schema validation ────────────────────────────


def test_load_registry_happy_path(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    assert data["schema_version"] == 1
    assert len(data["lanes"]) == 1


def test_load_registry_schema_mismatch(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    bad = json.loads((repo / lm.REGISTRY_REL).read_text())
    bad["schema_version"] = 99
    (repo / lm.REGISTRY_REL).write_text(json.dumps(bad))
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    with pytest.raises(ValueError, match="schema_version mismatch"):
        lm.load_registry()


def test_load_registry_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lm, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        lm.load_registry()


# ── Test 2: computed level rules (4 cases) ───────────────────────────────


def test_compute_level_zero_gates():
    assert lm.compute_level(_empty_gates()) == 0


def test_compute_level_partial_one_gate():
    g = _empty_gates()
    g["impl_complete"]["status"] = True
    assert lm.compute_level(g) == 1


def test_compute_level_two_required_gates_only():
    """Level 2 specifically requires impl_complete + real_archive_empirical."""
    g = _empty_gates()
    g["impl_complete"]["status"] = True
    g["real_archive_empirical"]["status"] = True
    assert lm.compute_level(g) == 2


def test_compute_level_4_gates_but_missing_required_is_still_l1():
    """Tie-break rule: 4 gates true but missing impl_complete = L1, NOT L2."""
    g = _empty_gates()
    # 4 gates true but neither impl_complete NOR real_archive_empirical
    g["contest_cuda"]["status"] = True
    g["strict_preflight"]["status"] = True
    g["three_clean_review"]["status"] = True
    g["memory_entry"]["status"] = True
    assert lm.compute_level(g) == 1


def test_compute_level_all_seven_is_l3():
    g = {gn: {"status": True, "evidence": "x"} for gn in lm.GATES}
    assert lm.compute_level(g) == 3


def test_compute_level_six_of_seven_is_l2_or_l1():
    """6/7 is L2 if both required are present, L1 if not."""
    g = {gn: {"status": True, "evidence": "x"} for gn in lm.GATES}
    g["deploy_runbook"]["status"] = False
    assert lm.compute_level(g) == 2  # required gates still satisfied
    # Now invert: drop a required gate
    g2 = {gn: {"status": True, "evidence": "x"} for gn in lm.GATES}
    g2["impl_complete"]["status"] = False
    assert lm.compute_level(g2) == 1


# ── Test 3: validate fails on missing evidence file ──────────────────────


def test_validate_fails_on_missing_evidence_file(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 1,
         "gates": {**_empty_gates(),
                   "impl_complete": {"status": True,
                                     "evidence": "src/tac/does_not_exist.py"}},
         "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    errors = lm.validate_registry(data, repo_root=repo)
    assert any("does not exist" in e for e in errors), errors


def test_validate_passes_when_evidence_file_exists(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    (repo / "src" / "tac" / "fake.py").write_text("# fake\n")
    data = json.loads((repo / lm.REGISTRY_REL).read_text())
    data["lanes"].append({
        "id": "lane_y", "name": "Y", "phase": 2, "level": 1,
        "gates": {**_empty_gates(),
                  "impl_complete": {"status": True,
                                    "evidence": "src/tac/fake.py"}},
        "notes": "",
    })
    (repo / lm.REGISTRY_REL).write_text(json.dumps(data))
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data2 = lm.load_registry()
    assert lm.validate_registry(data2, repo_root=repo) == []


def test_validate_descriptive_text_evidence_no_path_check(tmp_path, monkeypatch):
    """Evidence that doesn't look like a path skips file-existence check."""
    repo = _make_repo(tmp_path, [
        {"id": "lane_z", "name": "Z", "phase": 1, "level": 1,
         "gates": {**_empty_gates(),
                   "impl_complete": {"status": True,
                                     "evidence": "documented in council notes"}},
         "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    assert lm.validate_registry(data, repo_root=repo) == []


# ── Test 4: validate fails on level/gates mismatch ───────────────────────


def test_validate_fails_on_level_mismatch(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_w", "name": "W", "phase": 1, "level": 3,  # claims L3
         "gates": _empty_gates(), "notes": ""}  # but 0 gates true
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    errors = lm.validate_registry(data, repo_root=repo)
    assert any("disagrees with computed" in e for e in errors), errors


def test_validate_fails_on_duplicate_lane_ids(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_dup", "name": "A", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""},
        {"id": "lane_dup", "name": "B", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""},
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    errors = lm.validate_registry(data, repo_root=repo)
    assert any("duplicate lane id" in e for e in errors), errors


def test_validate_fails_on_missing_gate(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_mg", "name": "MG", "phase": 1, "level": 0,
         "gates": {"impl_complete": {"status": False, "evidence": ""}},
         "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    errors = lm.validate_registry(data, repo_root=repo)
    assert any("missing" in e for e in errors), errors


# ── Test 5: mark / unmark roundtrip ──────────────────────────────────────


def test_mark_gate_happy_path(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_m", "name": "M", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    (repo / "src" / "tac" / "real.py").write_text("# real\n")
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lm.mark_gate(data, "lane_m", "impl_complete", "src/tac/real.py", repo_root=repo)
    assert data["lanes"][0]["gates"]["impl_complete"]["status"] is True
    assert data["lanes"][0]["level"] == 1


def test_mark_gate_evidence_path_must_exist(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_m2", "name": "M2", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="does not exist"):
        lm.mark_gate(data, "lane_m2", "impl_complete",
                     "src/tac/missing.py", repo_root=repo)


def test_mark_gate_unknown_lane_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="unknown lane id"):
        lm.mark_gate(data, "lane_nope", "impl_complete", "x", repo_root=repo)


def test_mark_gate_unknown_gate_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_g", "name": "G", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="unknown gate"):
        lm.mark_gate(data, "lane_g", "not_a_gate", "x", repo_root=repo)


def test_unmark_gate_roundtrip(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_u", "name": "U", "phase": 1, "level": 1,
         "gates": {**_empty_gates(),
                   "impl_complete": {"status": True, "evidence": "described"}},
         "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lm.unmark_gate(data, "lane_u", "impl_complete", reason="rolled back")
    assert data["lanes"][0]["gates"]["impl_complete"]["status"] is False
    assert data["lanes"][0]["level"] == 0


def test_unmark_gate_requires_reason(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_u2", "name": "U2", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="reason"):
        lm.unmark_gate(data, "lane_u2", "impl_complete", reason="")


# ── Test 6: add-lane ─────────────────────────────────────────────────────


def test_add_lane_starts_at_l0(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lm.add_lane(data, "lane_new", "New Lane", phase=2)
    assert data["lanes"][0]["level"] == 0
    assert data["lanes"][0]["id"] == "lane_new"
    # All gates present, all false
    for g in lm.GATES:
        assert data["lanes"][0]["gates"][g]["status"] is False


def test_add_lane_duplicate_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="already exists"):
        lm.add_lane(data, "lane_x", "X again", phase=2)


# ── Test 7: audit log JSONL append ───────────────────────────────────────


def test_audit_log_appends_jsonl(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    lm.append_audit_log({"x": 1}, repo_root=repo)
    lm.append_audit_log({"x": 2}, repo_root=repo)
    log = (repo / lm.AUDIT_LOG_REL).read_text().strip().splitlines()
    assert len(log) == 2
    assert json.loads(log[0]) == {"x": 1}
    assert json.loads(log[1]) == {"x": 2}


# ── Test 8: report generation ────────────────────────────────────────────


def test_render_report_md(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_r1", "name": "R1", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": "n1"},
        {"id": "lane_r2", "name": "R2", "phase": 2, "level": 0,
         "gates": _empty_gates(), "notes": "n2"},
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    md = lm.render_report_md(data)
    assert "# Lane Maturity Report" in md
    assert "lane_r1" in md
    assert "lane_r2" in md
    assert "Phase 1" in md
    assert "Phase 2" in md
    # Auto-generated stamp must be present (caller has been warned not to edit)
    assert "Auto-generated" in md


# ── Test 9: end-to-end CLI smoke (subprocess) ────────────────────────────


def test_cli_validate_on_real_registry_passes():
    """The actual repo registry must validate cleanly. This is the contract
    the preflight Check 90 enforces."""
    proc = subprocess.run(
        [sys.executable, "tools/lane_maturity.py", "validate"],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"lane_maturity validate failed on real registry:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )


def test_cli_audit_on_real_registry_runs():
    """`audit` must complete with rc=0 and produce non-empty output."""
    proc = subprocess.run(
        [sys.executable, "tools/lane_maturity.py", "audit"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "LANE MATURITY AUDIT" in proc.stdout


# ── Test 10: looks_like_filepath / extract_filepath helpers ──────────────


@pytest.mark.parametrize("ev,expected", [
    ("src/tac/foo.py", True),
    ("scripts/remote_lane_x.sh", True),
    ("reports/raw/lane_g_v3", True),
    (".omx/state/foo.json", True),
    ("just descriptive text", False),
    ("documented in council notes", False),
    ("src/tac/foo.py — with notes after", True),
    ("[empirical:src/tac/foo.py] empirical result", True),
    ("see project_lane_x_landed memory entry", False),
    ("", False),
    ("/absolute/path", True),
])
def test_looks_like_filepath_heuristic(ev, expected):
    assert lm.looks_like_filepath(ev) == expected


def test_extract_filepath_strips_bracket_tag():
    assert lm.extract_filepath("[empirical:src/tac/foo.py] x") == "src/tac/foo.py"
    assert lm.extract_filepath("src/tac/foo.py") == "src/tac/foo.py"
    assert lm.extract_filepath("plain text") is None


# ── Test 11: Preflight Check 90 wiring ───────────────────────────────────


def test_check_90_passes_on_real_registry():
    """The actual repo's preflight Check 90 must pass cleanly."""
    from tac.preflight import check_lane_registry_consistent
    violations = check_lane_registry_consistent(strict=False, verbose=False)
    if violations:
        pytest.fail(
            "Check 90 failed on real registry — fix via "
            "tools/lane_maturity.py:\n  • " + "\n  • ".join(violations)
        )


def test_check_90_strict_raises_on_inconsistent_registry(tmp_path, monkeypatch):
    """Check 90 must RAISE MetaBugViolation when registry is inconsistent."""
    from tac.preflight import check_lane_registry_consistent, MetaBugViolation
    repo = _make_repo(tmp_path, [
        {"id": "lane_bad", "name": "Bad", "phase": 1, "level": 3,  # wrong
         "gates": _empty_gates(), "notes": ""}
    ])
    with pytest.raises(MetaBugViolation, match="LANE-REGISTRY"):
        check_lane_registry_consistent(repo_root=repo, strict=True, verbose=False)


def test_check_90_warn_only_returns_list(tmp_path, monkeypatch):
    """Check 90 strict=False must return list of errors instead of raising."""
    from tac.preflight import check_lane_registry_consistent
    repo = _make_repo(tmp_path, [
        {"id": "lane_bad2", "name": "Bad2", "phase": 1, "level": 3,
         "gates": _empty_gates(), "notes": ""}
    ])
    violations = check_lane_registry_consistent(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations
    assert any("disagrees with computed" in v for v in violations)
