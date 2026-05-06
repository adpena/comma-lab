"""Tests for tac.optimizer.sweep_plugin and the apogee_intN plugin extraction.

Covers:
  - register / load / list / unregister / reset
  - duplicate registration replaces (consistent with collection semantics)
  - load_generator raises KeyError for unknown names
  - the abstract base cannot be instantiated without overrides
  - the apogee_intN plugin imports + registers via __init__
  - the apogee_intN plugin returns expected schema fields
  - the apogee_intN dispatch command shape matches the original feedback_loop_sweep behavior
  - synthetic example generator from examples/synthetic_sweep.py round-trips

Per CLAUDE.md: every test added must actually pass.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from tac.optimizer.sweep_plugin import (
    CandidateGenerator,
    DispatchSpec,
    list_generators,
    load_generator,
    register_generator,
    reset_registry,
    unregister_generator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTH_PATH = REPO_ROOT / "examples" / "synthetic_sweep.py"


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a clean registry and we re-import the apogee plugin
    when the suite needs it (it self-registers on import)."""
    reset_registry()
    yield
    reset_registry()


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        CandidateGenerator()  # type: ignore[abstract]


def test_register_and_list():
    class Dummy(CandidateGenerator):
        name = "dummy"
        def __call__(self):
            return []
        def build_dispatch(self, candidate, *, label):
            return DispatchSpec(label=label, cmd=["echo", label])

    register_generator("dummy", Dummy)
    assert "dummy" in list_generators()


def test_register_rejects_empty_name():
    class Dummy(CandidateGenerator):
        name = "dummy"
        def __call__(self):
            return []
        def build_dispatch(self, candidate, *, label):
            return DispatchSpec(label=label, cmd=["echo", label])

    with pytest.raises(ValueError):
        register_generator("", Dummy)
    with pytest.raises(ValueError):
        register_generator(None, Dummy)  # type: ignore[arg-type]


def test_load_generator_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="unknown candidate generator"):
        load_generator("does_not_exist")


def test_unregister_is_noop_when_absent():
    unregister_generator("nope")  # no exception
    assert "nope" not in list_generators()


def test_register_replaces_on_duplicate():
    class A(CandidateGenerator):
        name = "x"
        def __call__(self):
            return [{"candidate_id": "a"}]
        def build_dispatch(self, candidate, *, label):
            return DispatchSpec(label=label, cmd=["a"])

    class B(CandidateGenerator):
        name = "x"
        def __call__(self):
            return [{"candidate_id": "b"}]
        def build_dispatch(self, candidate, *, label):
            return DispatchSpec(label=label, cmd=["b"])

    register_generator("x", A)
    assert load_generator("x")()[0]["candidate_id"] == "a"
    register_generator("x", B)
    assert load_generator("x")()[0]["candidate_id"] == "b"


def test_dispatch_spec_defaults():
    s = DispatchSpec(label="foo", cmd=["python", "-c", "pass"])
    assert s.label == "foo"
    assert s.estimated_cost_usd == 0.30
    assert s.timeout_seconds == 1800.0
    assert s.cwd is None
    assert s.env == {}


def test_apogee_intn_plugin_self_registers_on_import():
    # Force a fresh import; reset_registry() + autouse cleared the registry.
    if "tac.optimizer.generators.apogee_intn" in sys.modules:
        importlib.reload(sys.modules["tac.optimizer.generators.apogee_intn"])
    else:
        importlib.import_module("tac.optimizer.generators.apogee_intn")
    assert "apogee_intN" in list_generators()


def _ensure_apogee_registered():
    if "tac.optimizer.generators.apogee_intn" in sys.modules:
        importlib.reload(sys.modules["tac.optimizer.generators.apogee_intn"])
    else:
        importlib.import_module("tac.optimizer.generators.apogee_intn")


def test_apogee_intn_returns_list():
    _ensure_apogee_registered()
    gen = load_generator("apogee_intN")
    candidates = gen()
    # It returns a list (possibly empty if no apogee_int*_repack_* dirs exist).
    # Either way, every candidate must have the canonical schema fields.
    assert isinstance(candidates, list)
    for c in candidates:
        assert "candidate_id" in c
        assert "archive_bytes" in c
        assert "rel_err_pct" in c
        assert c["evidence_semantics"] == "byte_only_forensic"
        assert c["ready_for_exact_eval_dispatch"] is False
        assert "missing_contest_faithful_distortion_model" in c["dispatch_blockers"]


def test_apogee_intn_build_dispatch_includes_lightning_args():
    _ensure_apogee_registered()
    gen = load_generator("apogee_intN")
    fake = {
        "candidate_id": "apogee_int7",
        "archive_bytes": 567890,
        "archive_sha256": "abc1234567890def",
        "archive_path": str(REPO_ROOT / "experiments/results/apogee_int7_repack/archive.zip"),
        "predicted_band": [0.9, 1.1],
    }
    spec = gen.build_dispatch(fake, label="loopcycle_int7")
    assert spec.label == "loopcycle_int7"
    cmd_str = " ".join(spec.cmd)
    assert "lightning_dispatch_pr106_stack.py" in cmd_str
    assert "--lane apogee_int7" in cmd_str
    assert "--job-name loopcycle_int7" in cmd_str
    assert "--predicted-low 0.9" in cmd_str
    assert "--predicted-high 1.1" in cmd_str
    assert "--print-only" in cmd_str
    assert "--allow-forensic-apogee-intN" in cmd_str


def test_synthetic_example_loadable_and_runs():
    """Loading examples/synthetic_sweep.py registers + generates candidates."""
    spec = importlib.util.spec_from_file_location("synthetic_sweep", SYNTH_PATH)
    assert spec and spec.loader, f"could not load module from {SYNTH_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["synthetic_sweep"] = mod
    spec.loader.exec_module(mod)
    # After import, "toy_compression" is in the registry
    assert "toy_compression" in list_generators()
    gen = load_generator("toy_compression")
    cands = gen()
    assert len(cands) == 6
    # Schema invariants
    for c in cands:
        assert c["evidence_semantics"] == "synthetic_test"
        assert c["ready_for_exact_eval_dispatch"] is False
        assert "synthetic_workload_no_real_eval" in c["dispatch_blockers"]
    spec_obj = gen.build_dispatch(cands[0], label="synthetic_test_001")
    assert spec_obj.label == "synthetic_test_001"
    assert spec_obj.estimated_cost_usd == 0.0
    assert "synthetic-dispatch label=synthetic_test_001" in " ".join(spec_obj.cmd)
