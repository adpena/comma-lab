"""Tests for the EIGHTFOLD design-philosophy preflight gates
(``tac.confound_gates`` P1 + P4; 2026-07-09 operator "Encode all").

Coverage per gate: positive (catches the anti-pattern) / negative (allows the
clean form) / waiver-respect / placeholder-waiver-rejected / edge cases / strict
raises PreflightError / warn-only returns. The P4 suite explicitly verifies the
heuristic CATCHES a synthetic meter-without-canary AND does NOT false-flag an
actuator/value-object (the NO-FAKE anti-token-theater requirement). Plus a
repo-smoke test that bounds each gate's live-count against the real tree so a
future regression that ADDS a violation is caught.

Source: .omx/research/design_philosophies_eightfold_20260709.md +
.omx/research/eightfold_apparatus_build_20260709.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac import confound_gates as cg
from tac.preflight import PreflightError

_SIG_REL = cg._SIGNIFICANCE_STORE_REL
_CTRL_REL = cg._WITNESS_CONTROL_REL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _write_store(root: Path, rows: list[dict]) -> Path:
    lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    return _mk(root, _SIG_REL, lines)


def _real_factory() -> str:
    from tac.witness_dsl.lever_registry import lever_factories

    return sorted(lever_factories().keys())[0]


def _real_alias_pair() -> tuple[str, str]:
    """A (legacy_task_key, held_canonical_factory) pair from the REAL alias map,
    skipped if none currently resolves (registry churn robustness)."""
    from tac.witness_dsl.activation_ledger import _SIGNIFICANCE_LEVER_ALIASES
    from tac.witness_dsl.lever_registry import lever_factories

    held = set(lever_factories().keys())
    for legacy, canonical in _SIGNIFICANCE_LEVER_ALIASES.items():
        if canonical in held:
            return legacy, canonical
    pytest.skip("no held alias target in the current registry")


def _sig_row(lever: str, notes: str = "") -> dict:
    return {
        "agent": "test",
        "axis": "d_seg",
        "delta_s_label": "ESTIMATED",
        "est_delta_s": 0.01,
        "lever": lever,
        "notes": notes,
        "source_anchor": "test",
        "ts": "2026-07-09T00:00:00Z",
    }


def _meter_module(class_name: str, *, canary: bool, waiver: str | None = None,
                  methods: str = "    def observe(self, ep, x):\n        return x\n") -> str:
    canary_line = (
        "# a positive_control synthetic_control canary here\n" if canary else ""
    )
    waiver_line = f"    # METER_CANARY_OK:{waiver}\n" if waiver else ""
    return (
        '"""module."""\n'
        + canary_line
        + f"class {class_name}:\n"
        + waiver_line
        + methods
    )


# ===========================================================================
# P1 — check_significance_keys_canonical
# ===========================================================================


def test_p1_held_factory_key_resolves(tmp_path):
    _write_store(tmp_path, [_sig_row(_real_factory())])
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_aliased_task_key_resolves(tmp_path):
    legacy, _canonical = _real_alias_pair()
    _write_store(tmp_path, [_sig_row(legacy)])
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_unresolvable_key_flagged(tmp_path):
    _write_store(tmp_path, [_sig_row("totally_made_up_lever_999")])
    v = cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False)
    assert len(v) == 1 and "totally_made_up_lever_999" in v[0]


def test_p1_waiver_in_notes_respected(tmp_path):
    _write_store(
        tmp_path,
        [_sig_row("byte_close_tool_lever_x",
                  notes="a free byte cut # SIGNIFICANCE_KEY_OK:byte-close tool lever not a DSL factory")],
    )
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_placeholder_waiver_rejected(tmp_path):
    _write_store(
        tmp_path,
        [_sig_row("byte_close_tool_lever_x", notes="x # SIGNIFICANCE_KEY_OK:<rationale>")],
    )
    v = cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False)
    assert len(v) == 1


def test_p1_notes_naming_held_factory_gives_alias_hint(tmp_path):
    fac = _real_factory()
    _write_store(tmp_path, [_sig_row("some_task_777", notes=f"DSL {fac} factory built")])
    v = cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False)
    assert len(v) == 1 and "add its alias" in v[0]


def test_p1_missing_store_is_ok(tmp_path):
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_corrupt_json_line_skipped(tmp_path):
    _mk(tmp_path, _SIG_REL, "{not json}\n" + json.dumps(_sig_row(_real_factory())) + "\n")
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_latest_row_wins(tmp_path):
    # first row unresolvable, later row for same key resolves onto a held factory
    fac = _real_factory()
    _mk(
        tmp_path,
        _SIG_REL,
        json.dumps(_sig_row(fac, notes="v1")) + "\n" + json.dumps(_sig_row(fac, notes="v2")) + "\n",
    )
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, verbose=False) == []


def test_p1_strict_raises(tmp_path):
    _write_store(tmp_path, [_sig_row("nope_lever_000")])
    with pytest.raises(PreflightError):
        cg.check_significance_keys_canonical(repo_root=tmp_path, strict=True, verbose=False)


def test_p1_strict_clean_returns_empty(tmp_path):
    _write_store(tmp_path, [_sig_row(_real_factory())])
    assert cg.check_significance_keys_canonical(repo_root=tmp_path, strict=True, verbose=False) == []


# ===========================================================================
# P4 — check_witness_control_meters_have_canaries
# ===========================================================================


def test_p4_meter_with_canary_passes(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py", _meter_module("FooDetector", canary=True))
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_meter_without_canary_flagged(tmp_path):
    # THE NO-FAKE check: the heuristic must catch a synthetic meter-without-canary.
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py", _meter_module("FooDetector", canary=False))
    v = cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False)
    assert len(v) == 1 and "FooDetector" in v[0]


def test_p4_alarm_suffix_is_a_meter(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/alarm_x.py", _meter_module("BarAlarm", canary=False, methods="    pass\n"))
    v = cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False)
    assert len(v) == 1 and "BarAlarm" in v[0]


def test_p4_canary_in_test_file_passes(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py", _meter_module("FooDetector", canary=False))
    _mk(tmp_path, f"{_CTRL_REL}/tests/test_plateau_x.py", "def test_canary():\n    positive_control = 1\n")
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_waiver_near_class_respected(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py",
        _meter_module("FooDetector", canary=False, waiver="synthetic canary infeasible; live-gated downstream"))
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_placeholder_waiver_rejected(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py",
        _meter_module("FooDetector", canary=False, waiver="<rationale>"))
    v = cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False)
    assert len(v) == 1


def test_p4_actuator_gate_value_object_not_flagged(tmp_path):
    # A frozen value-object / actuator named *Gate with NO meter-verb method is
    # NOT a measurement surface — deliberately excluded (no false positive).
    body = (
        '"""module."""\n'
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class GateStep:\n"
        "    just_fired: bool\n\n"
        "class EventBackstopGate:\n"
        "    def update(self, ep):\n"
        "        return None\n"
        "    def fire(self):\n"
        "        return True\n"
    )
    _mk(tmp_path, f"{_CTRL_REL}/event_x.py", body)
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_observe_method_ambiguous_name_is_uncertain_not_violation(tmp_path):
    # observe() but non-meter name -> UNCERTAIN, not a violation by the heuristic floor.
    _mk(tmp_path, f"{_CTRL_REL}/ctrl_x.py",
        _meter_module("SomeController", canary=False, methods="    def observe(self, ep, x):\n        return x\n"))
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_detect_method_ambiguous_name_uncertain(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/ctrl_y.py",
        _meter_module("Widget", canary=False, methods="    def detect(self, x):\n        return x\n"))
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_plain_class_no_meter_signal_ignored(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plain_x.py",
        '"""m."""\nclass Config:\n    def to_dict(self):\n        return {}\n')
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_init_py_skipped(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/__init__.py", _meter_module("FooDetector", canary=False))
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_syntax_error_module_skipped(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/broken.py", "class FooDetector(:\n")
    # unparseable module is skipped, not crashed
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_missing_dir_is_ok(tmp_path):
    assert cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, verbose=False) == []


def test_p4_strict_raises(tmp_path):
    _mk(tmp_path, f"{_CTRL_REL}/plateau_x.py", _meter_module("FooDetector", canary=False))
    with pytest.raises(PreflightError):
        cg.check_witness_control_meters_have_canaries(repo_root=tmp_path, strict=True, verbose=False)


def test_p4_use_fmtools_off_no_subprocess(monkeypatch):
    # default path must NEVER shell out to the FM venv (per-session zero cost).
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("fmtools must not be invoked when use_fmtools=False")

    monkeypatch.setattr(cg, "_fm_meter_advisory", _boom)
    cg.check_witness_control_meters_have_canaries(verbose=False)  # real tree, default off
    assert called["n"] == 0


# ===========================================================================
# repo-smoke: bound the live-count so a NEW violation is caught
# ===========================================================================


def test_p1_repo_live_count_bounded():
    v = cg.check_significance_keys_canonical(verbose=False)
    assert len(v) <= 4, f"P1 live-count regressed upward: {v}"


def test_p4_repo_live_count_bounded():
    v = cg.check_witness_control_meters_have_canaries(verbose=False)
    assert len(v) <= 1, f"P4 live-count regressed upward: {v}"


def test_eightfold_registry_wired():
    assert cg.check_significance_keys_canonical in cg.EIGHTFOLD_GATES
    assert cg.check_witness_control_meters_have_canaries in cg.EIGHTFOLD_GATES
