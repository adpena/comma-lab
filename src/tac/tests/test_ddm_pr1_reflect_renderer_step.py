"""Tests for ``experiments/ddm_pr1_reflect_renderer_step``.

The reflected step exists to test the ONE assumption the coupling law names as
stated-not-measured. Two things would make it useless without being visible:

* a reflection that is not a reflection (a sign or factor slip) -- the arithmetic
  is checked directly against ``2*shipped - candidate``;
* a reflection the deployed encoder cannot represent, so the REALIZED object is
  not the opposite step at all. ``alignment`` is the instrument that would show
  that, and it must report a perfect reflection as cosine -1 with norm ratio 1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO / "experiments" / "ddm_pr1_reflect_renderer_step.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pr1_reflect_undertest", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ref = _load_module()


class TestBuildReflection:
    def test_is_the_point_reflection_through_the_shipped_weights(self):
        shipped = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([[0.5]])}
        candidate = {"a": torch.tensor([1.5, 1.0]), "b": torch.tensor([[0.25]])}
        got = ref.build_reflection(shipped, candidate)
        assert torch.allclose(got["a"], torch.tensor([0.5, 3.0]))
        assert torch.allclose(got["b"], torch.tensor([[0.75]]))

    def test_reflecting_twice_returns_the_candidate(self):
        shipped = {"a": torch.tensor([1.0, 2.0, -3.0])}
        candidate = {"a": torch.tensor([1.5, 1.0, 4.0])}
        once = ref.build_reflection(shipped, candidate)
        twice = ref.build_reflection(shipped, once)
        assert torch.allclose(twice["a"], candidate["a"])

    def test_a_zero_step_reflects_to_itself(self):
        shipped = {"a": torch.tensor([1.0, 2.0])}
        got = ref.build_reflection(shipped, {"a": shipped["a"].clone()})
        assert torch.allclose(got["a"], shipped["a"])

    def test_preserves_dtype(self):
        shipped = {"a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        candidate = {"a": torch.tensor([1.5, 1.0], dtype=torch.float32)}
        assert ref.build_reflection(shipped, candidate)["a"].dtype == torch.float32

    def test_refuses_mismatched_keys(self):
        with pytest.raises(ValueError, match="different keys"):
            ref.build_reflection({"a": torch.zeros(2)}, {"b": torch.zeros(2)})

    def test_refuses_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            ref.build_reflection({"a": torch.zeros(2)}, {"a": torch.zeros(3)})


class TestAlignment:
    def test_a_perfect_reflection_is_cosine_minus_one_at_unit_norm_ratio(self):
        shipped = {"a": torch.tensor([0.0, 0.0])}
        candidate = {"a": torch.tensor([1.0, 2.0])}
        reflected = {"a": torch.tensor([-1.0, -2.0])}
        got = ref.alignment(shipped, candidate, reflected)
        assert got["cosine"] == pytest.approx(-1.0)
        assert got["norm_ratio"] == pytest.approx(1.0)

    def test_an_unrepresentable_reflection_shows_up_as_a_short_step(self):
        """If the encoder cannot carry the opposite step the norm ratio collapses."""
        shipped = {"a": torch.tensor([0.0, 0.0])}
        candidate = {"a": torch.tensor([1.0, 2.0])}
        reflected = {"a": torch.tensor([-0.05, -0.1])}
        got = ref.alignment(shipped, candidate, reflected)
        assert got["cosine"] == pytest.approx(-1.0)
        assert got["norm_ratio"] == pytest.approx(0.05)

    def test_a_reflection_that_went_the_same_way_is_positive_cosine(self):
        shipped = {"a": torch.tensor([0.0, 0.0])}
        candidate = {"a": torch.tensor([1.0, 2.0])}
        got = ref.alignment(shipped, candidate, {"a": torch.tensor([1.0, 2.0])})
        assert got["cosine"] == pytest.approx(1.0)

    def test_reports_the_reference_value_so_a_reader_need_not_recall_it(self):
        shipped = {"a": torch.tensor([0.0])}
        got = ref.alignment(shipped, {"a": torch.tensor([1.0])}, {"a": torch.tensor([-1.0])})
        assert got["perfect_reflection_cosine"] == -1.0

    def test_a_degenerate_zero_step_does_not_raise(self):
        shipped = {"a": torch.tensor([1.0])}
        got = ref.alignment(shipped, {"a": torch.tensor([1.0])}, {"a": torch.tensor([1.0])})
        assert got["forward_step_norm"] == 0.0
