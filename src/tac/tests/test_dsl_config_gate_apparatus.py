"""Apparatus tests for the DSL-authored-config enforcement (#353, req V):
the confound preflight gate (#403) + the triality drift-detector leg.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tac.confound_gates import check_launch_config_authored_in_dsl
from tac.preflight import PreflightError

_REPO = Path(__file__).resolve().parents[3]


def _load_tdd():
    if str(_REPO / "tools") not in sys.path:
        sys.path.insert(0, str(_REPO / "tools"))
    spec = importlib.util.spec_from_file_location("triality_drift_detector",
                                                  _REPO / "tools" / "triality_drift_detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_fake_repo(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "src" / "tac"
    d.mkdir(parents=True)
    (d / "witness_autoconfig.py").write_text(body)
    return tmp_path


# ── preflight gate #403 ──────────────────────────────────────────────────────
def test_gate_flags_untyped_derive_config(tmp_path):
    root = _make_fake_repo(tmp_path, "def derive_thing_config(gt):\n    return build_argv(gt)\n")
    v = check_launch_config_authored_in_dsl(repo_root=root, strict=False, verbose=False)
    assert len(v) == 1 and "derive_thing_config" in v[0]


def test_gate_passes_when_routed_through_typed_layer(tmp_path):
    root = _make_fake_repo(
        tmp_path,
        "def derive_thing_config(gt):\n"
        "    from tac.witness_dsl.typed_config import build_launch_manifest\n"
        "    return build_launch_manifest(program_name='x', emitted_flag_names=[], typed_config_hash='h')\n",
    )
    v = check_launch_config_authored_in_dsl(repo_root=root, strict=False, verbose=False)
    assert v == []


def test_gate_respects_waiver(tmp_path):
    root = _make_fake_repo(
        tmp_path,
        "def derive_thing_config(gt):  # DSL_CONFIG_AUTHORING_OK: migration queue req V, #353\n"
        "    return build_argv(gt)\n",
    )
    v = check_launch_config_authored_in_dsl(repo_root=root, strict=False, verbose=False)
    assert v == []


def test_gate_ignores_non_derive_config_functions(tmp_path):
    root = _make_fake_repo(tmp_path, "def helper_flags(x):\n    return [('--a', 1)]\n")
    v = check_launch_config_authored_in_dsl(repo_root=root, strict=False, verbose=False)
    assert v == []


def test_gate_strict_raises(tmp_path):
    root = _make_fake_repo(tmp_path, "def derive_x_config(gt):\n    return build_argv(gt)\n")
    with pytest.raises(PreflightError):
        check_launch_config_authored_in_dsl(repo_root=root, strict=True, verbose=False)


def test_gate_live_repo_crucible_passes_others_are_migration_queue():
    """On the REAL repo: crucible routes through the typed layer; the other 4 derive_*_config
    are the (documented) migration queue. live-count == exactly those 4."""
    v = check_launch_config_authored_in_dsl(strict=False, verbose=False)
    names = {"derive_sealed_205_config", "derive_store_nothing_205_config",
             "derive_fresh_seeded_config", "derive_config"}
    flagged = {n for n in names if any(n + "() authors" in msg for msg in v)}
    assert flagged == names, flagged
    assert not any("derive_crucible_v6_config() authors" in msg for msg in v), "crucible must be routed"


def test_gate_registered_in_confound_tuple():
    from tac.confound_gates import CONFOUND_GATES
    assert check_launch_config_authored_in_dsl in CONFOUND_GATES


# ── drift-detector leg ───────────────────────────────────────────────────────
def test_drift_leg_flags_new_untyped_emitter():
    tdd = _load_tdd()
    added = {"src/tac/witness_autoconfig.py": ["+def derive_new_config(gt):", "+    return WitnessConfig()"]}
    v = tdd.dsl_config_bypass_violations(["a commit"], added)
    assert v and "derive_new_config" in v[0]


def test_drift_leg_clean_when_window_cites_typed_layer():
    tdd = _load_tdd()
    added = {"src/tac/witness_autoconfig.py": [
        "+def derive_new_config(gt):", "+    from tac.witness_dsl.typed_config import build_launch_manifest"]}
    assert tdd.dsl_config_bypass_violations(["c"], added) == []


def test_drift_leg_respects_waiver():
    tdd = _load_tdd()
    added = {"src/tac/witness_autoconfig.py": [
        "+def derive_x_config(gt):  # DSL_CONFIG_BYPASS_OK: legacy shim migration #999"]}
    assert tdd.dsl_config_bypass_violations(["c"], added) == []


def test_drift_leg_witness_dsl_out_of_scope():
    tdd = _load_tdd()
    assert not tdd.dsl_config_file_in_scope("src/tac/witness_dsl/typed_config.py")
    assert tdd.dsl_config_file_in_scope("src/tac/witness_autoconfig.py")


def test_drift_leg_opt_out_respected():
    tdd = _load_tdd()
    added = {"src/tac/witness_autoconfig.py": ["+def derive_new_config(gt):"]}
    assert tdd.dsl_config_bypass_violations(["feat [no-triality]"], added) == []


def test_drift_leg_fail_open_on_bad_input():
    tdd = _load_tdd()
    assert tdd.dsl_config_bypass_violations_safe(["c"], None) == []  # type: ignore[arg-type]
