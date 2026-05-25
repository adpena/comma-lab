# SPDX-License-Identifier: MIT
# FAKE_LANE_OK_FILE: this test file constructs many fake lane_id fixtures
# (lane_a, lane_b, lane_g, lane_m, lane_dup, lane_r1, lane_r2, lane_u, etc.)
# to exercise the lane-maturity harness CLI behavior. Per Check #126
# file-level waiver semantics, the entire file is exempt from the
# registered-lane gate.
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
    """All gates true with evidence pointing to a file we create in `repo`."""
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


def test_load_registry_migrates_missing_contest_cpu_gate_fail_closed(
    tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path, [])
    old_gates = [g for g in lm.GATES if g != "contest_cpu"]
    data = json.loads((repo / lm.REGISTRY_REL).read_text())
    data["gate_definitions"] = {g: f"<def of {g}>" for g in old_gates}
    data.setdefault("level_rules", {})["3"] = "ALL 7 gates satisfied"
    data["lanes"].append(
        {
            "id": "lane_old_l3",
            "name": "Old L3",
            "phase": 1,
            "level": 3,
            "gates": {g: {"status": True, "evidence": "documented"} for g in old_gates},
            "notes": "",
        }
    )
    (repo / lm.REGISTRY_REL).write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(lm, "REPO_ROOT", repo)

    migrated = lm.load_registry()
    lane = migrated["lanes"][0]

    assert "contest_cpu" in migrated["gate_definitions"]
    assert lane["gates"]["contest_cpu"] == {"status": False, "evidence": ""}
    assert lane["level"] == 2
    assert lm.validate_registry(migrated, repo_root=repo) == []


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


def test_compute_level_all_gates_is_l3():
    g = {gn: {"status": True, "evidence": "x"} for gn in lm.GATES}
    assert lm.compute_level(g) == 3


def test_compute_level_one_missing_gate_is_l2_or_l1():
    """One missing gate is L2 if both required are present, L1 if not."""
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


def test_validate_rejects_l2_exact_readiness_refusal_evidence(
    tmp_path,
    monkeypatch,
):
    repo = _make_repo(tmp_path, [])
    evidence = repo / ".omx" / "research" / "package_report.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "example_package_report.v1",
                "exact_readiness_refusal": {
                    "ready": False,
                    "blockers": ["requires_exact_auth_eval"],
                },
            }
        )
    )
    gates = _empty_gates()
    gates["impl_complete"] = {"status": True, "evidence": "src/tac/fake.py"}
    gates["real_archive_empirical"] = {
        "status": True,
        "evidence": ".omx/research/package_report.json",
    }
    (repo / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "tac" / "fake.py").write_text("# fake\n")
    data = json.loads((repo / lm.REGISTRY_REL).read_text())
    data["lanes"].append(
        {
            "id": "lane_refusal",
            "name": "Refusal",
            "phase": 2,
            "level": 2,
            "gates": gates,
            "notes": "",
        }
    )
    (repo / lm.REGISTRY_REL).write_text(json.dumps(data))
    monkeypatch.setattr(lm, "REPO_ROOT", repo)

    errors = lm.validate_registry(lm.load_registry(), repo_root=repo)

    assert any("exact_readiness_refusal.ready=false" in e for e in errors), errors


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
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


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
    from tac.preflight import MetaBugViolation, check_lane_registry_consistent
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


# ── set_field mutation surface (Catalog #124 backfill) ───────────────────


def test_set_field_top_level_research_only_bool(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lane = lm.set_field(data, "lane_x", "research_only", True)
    assert lane["research_only"] is True


def test_set_field_top_level_lane_class_string(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lane = lm.set_field(data, "lane_x", "lane_class", "substrate_engineering")
    assert lane["lane_class"] == "substrate_engineering"


def test_set_field_top_level_reactivation_criteria_list(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    crit = ["criterion_a", "criterion_b"]
    lane = lm.set_field(data, "lane_x", "reactivation_criteria", crit)
    assert lane["reactivation_criteria"] == crit


def test_set_field_design_evidence_subfield_creates_dict(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lane = lm.set_field(
        data, "lane_x",
        "design_evidence.archive_grammar",
        "src/tac/foo.py:42_FOO_MAGIC",
    )
    assert "design_evidence" in lane
    assert lane["design_evidence"]["archive_grammar"] == "src/tac/foo.py:42_FOO_MAGIC"


def test_set_field_design_evidence_subfield_int(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    lane = lm.set_field(
        data, "lane_x",
        "design_evidence.inflate_runtime_loc_budget",
        100,
    )
    assert lane["design_evidence"]["inflate_runtime_loc_budget"] == 100


def test_set_field_unknown_lane_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="unknown lane id"):
        lm.set_field(data, "lane_does_not_exist", "research_only", True)


def test_set_field_unknown_top_level_field_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="unknown field"):
        lm.set_field(data, "lane_x", "level", 3)  # 'level' is computed, not settable


def test_set_field_unknown_design_evidence_subfield_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="unknown design_evidence sub-field"):
        lm.set_field(data, "lane_x", "design_evidence.bogus", "x")


def test_set_field_empty_value_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="must be non-empty"):
        lm.set_field(data, "lane_x", "research_only", "")


def test_set_field_design_evidence_existing_non_dict_raises(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": "",
         "design_evidence": "not a dict"}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    data = lm.load_registry()
    with pytest.raises(ValueError, match="not a dict"):
        lm.set_field(
            data, "lane_x",
            "design_evidence.archive_grammar", "x",
        )


def test_coerce_set_field_value_bool_yes_no():
    assert lm._coerce_set_field_value("research_only", "yes") is True
    assert lm._coerce_set_field_value("research_only", "NO") is False
    assert lm._coerce_set_field_value(
        "design_evidence.no_op_detector_planned", "true"
    ) is True


def test_coerce_set_field_value_int():
    assert lm._coerce_set_field_value(
        "design_evidence.inflate_runtime_loc_budget", "100"
    ) == 100
    with pytest.raises(ValueError, match="expects integer"):
        lm._coerce_set_field_value(
            "design_evidence.bolt_on_loc_budget", "not_a_number"
        )


def test_coerce_set_field_value_list_csv():
    assert lm._coerce_set_field_value(
        "design_evidence.runtime_dep_closure", "torch, brotli, numpy"
    ) == ["torch", "brotli", "numpy"]
    assert lm._coerce_set_field_value(
        "reactivation_criteria", "a,,b,"
    ) == ["a", "b"]


def test_check_124_passes_after_set_field_research_only_optout(tmp_path, monkeypatch):
    """End-to-end: a representation lane at level >= 1 violates Check 124,
    then research_only=true via set_field clears the violation."""
    from tac.preflight import (
        check_representation_lane_has_archive_grammar_at_design_time,
    )
    # Build repo first (creates src/tac/), then seed the fake_module.py
    # before computing the all-true gates (which assume the file exists).
    repo = _make_repo(tmp_path, [])
    gates_true = _all_true_gates(repo)
    data = json.loads((repo / lm.REGISTRY_REL).read_text())
    data["lanes"] = [
        {"id": "lane_nerv_test", "name": "Lane NeRV test", "phase": 1,
         "level": lm.compute_level(gates_true),
         "gates": gates_true, "notes": ""}
    ]
    (repo / lm.REGISTRY_REL).write_text(json.dumps(data, indent=2))

    monkeypatch.setattr(lm, "REPO_ROOT", repo)

    # Should be flagged before opt-out
    violations_before = (
        check_representation_lane_has_archive_grammar_at_design_time(
            repo_root=repo, strict=False, verbose=False,
        )
    )
    assert any("lane_nerv_test" in v for v in violations_before), violations_before

    # Apply opt-out
    data = lm.load_registry()
    lm.set_field(data, "lane_nerv_test", "research_only", True)
    lm.save_registry(data)

    # Should be clear after opt-out
    violations_after = (
        check_representation_lane_has_archive_grammar_at_design_time(
            repo_root=repo, strict=False, verbose=False,
        )
    )
    assert not any(
        "lane_nerv_test" in v for v in violations_after
    ), violations_after


def test_set_field_cli_command(tmp_path, monkeypatch):
    """End-to-end: the `set-field` subcommand persists the field + audit log."""
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "X", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": ""}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)
    rc = lm.main([
        "set-field", "lane_x",
        "--field", "research_only", "--value", "true",
    ])
    assert rc == 0
    data = lm.load_registry()
    assert data["lanes"][0]["research_only"] is True
    audit = (repo / lm.AUDIT_LOG_REL).read_text().strip().split("\n")
    assert any("set-field" in line for line in audit)


def test_set_field_updates_display_metadata_without_bare_registry_edit(
    tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path, [
        {"id": "lane_x", "name": "Old", "phase": 1, "level": 0,
         "gates": _empty_gates(), "notes": "old notes"}
    ])
    monkeypatch.setattr(lm, "REPO_ROOT", repo)

    assert lm.main([
        "set-field", "lane_x", "--field", "name", "--value", "New display name"
    ]) == 0
    assert lm.main([
        "set-field", "lane_x", "--field", "notes", "--value", "new notes"
    ]) == 0

    data = lm.load_registry()
    assert data["lanes"][0]["name"] == "New display name"
    assert data["lanes"][0]["notes"] == "new notes"
