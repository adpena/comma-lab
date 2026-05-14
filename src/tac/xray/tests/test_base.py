"""Tests for tac.xray.base canonical primitive protocol + result dataclass."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.xray.base import (
    CANONICAL_WIRE_IN_HOOKS,
    XRAY_PRIMITIVE_SCHEMA_VERSION,
    ComposedXRayPrimitive,
    XRayPrimitive,
    XRayPrimitiveResult,
)


class _FakePrimitive:
    """Minimal XRayPrimitive implementation for protocol tests."""

    def __init__(self, name: str, hooks, value, grade="mathematical-derivation"):
        self._name = name
        self._hooks = hooks
        self._value = value
        self._grade = grade

    @property
    def name(self) -> str:
        return self._name

    @property
    def wire_in_hooks(self):
        return self._hooks

    def compute(self, target, **kwargs):
        return XRayPrimitiveResult(
            primitive_name=self._name,
            archive_or_video_path=Path(target) if target else None,
            archive_sha256=None,
            primitive_value=self._value,
            evidence_grade=self._grade,
            confidence_band=None,
            composes_with=(),
            wire_in_hooks_engaged=self._hooks,
        )

    def compose_with(self, other):
        return ComposedXRayPrimitive(left=self, right=other)


def test_schema_version_pinned():
    assert XRAY_PRIMITIVE_SCHEMA_VERSION == "tac_xray_primitive_v1"


def test_canonical_wire_in_hooks_count():
    assert len(CANONICAL_WIRE_IN_HOOKS) == 6


def test_canonical_wire_in_hooks_values():
    assert "sensitivity_map" in CANONICAL_WIRE_IN_HOOKS
    assert "pareto_constraint" in CANONICAL_WIRE_IN_HOOKS
    assert "bit_allocator" in CANONICAL_WIRE_IN_HOOKS
    assert "cathedral_autopilot" in CANONICAL_WIRE_IN_HOOKS
    assert "continual_learning" in CANONICAL_WIRE_IN_HOOKS
    assert "probe_disambiguator" in CANONICAL_WIRE_IN_HOOKS


def test_result_rejects_empty_wire_in_hooks(tmp_path):
    with pytest.raises(ValueError, match="zero wire-in hooks"):
        XRayPrimitiveResult(
            primitive_name="x",
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=1.0,
            evidence_grade="mathematical-derivation",
            confidence_band=None,
            composes_with=(),
            wire_in_hooks_engaged=(),
        )


def test_result_rejects_empty_name():
    with pytest.raises(ValueError, match="primitive_name must be non-empty"):
        XRayPrimitiveResult(
            primitive_name="",
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=1.0,
            evidence_grade="mathematical-derivation",
            confidence_band=None,
            composes_with=(),
            wire_in_hooks_engaged=("sensitivity_map",),
        )


def test_result_rejects_unknown_hook():
    with pytest.raises(ValueError, match="unknown wire-in hook"):
        XRayPrimitiveResult(
            primitive_name="x",
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=1.0,
            evidence_grade="mathematical-derivation",
            confidence_band=None,
            composes_with=(),
            wire_in_hooks_engaged=("bogus_hook",),  # type: ignore
        )


def test_result_rejects_inverted_confidence_band():
    with pytest.raises(ValueError, match="lower > upper"):
        XRayPrimitiveResult(
            primitive_name="x",
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=1.0,
            evidence_grade="mathematical-derivation",
            confidence_band=(1.0, 0.5),
            composes_with=(),
            wire_in_hooks_engaged=("sensitivity_map",),
        )


def test_result_accepts_well_formed():
    r = XRayPrimitiveResult(
        primitive_name="test",
        archive_or_video_path=Path("/tmp/foo"),  # path object is fine, not /tmp string literal in artifact
        archive_sha256="a" * 64,
        primitive_value=1.0,
        evidence_grade="mathematical-derivation",
        confidence_band=(0.0, 1.0),
        composes_with=("other",),
        wire_in_hooks_engaged=("sensitivity_map",),
    )
    assert r.primitive_name == "test"
    assert r.hook_count() == 1


def test_result_hook_count_deduplicates():
    r = XRayPrimitiveResult(
        primitive_name="x",
        archive_or_video_path=None,
        archive_sha256=None,
        primitive_value=1.0,
        evidence_grade="mathematical-derivation",
        confidence_band=None,
        composes_with=(),
        wire_in_hooks_engaged=("sensitivity_map", "sensitivity_map", "pareto_constraint"),
    )
    assert r.hook_count() == 2


def test_primitive_protocol_runtime_check():
    p = _FakePrimitive("test", ("sensitivity_map",), 1.0)
    assert isinstance(p, XRayPrimitive)


def test_composed_primitive_name():
    a = _FakePrimitive("a", ("sensitivity_map",), 1.0)
    b = _FakePrimitive("b", ("pareto_constraint",), 2.0)
    c = ComposedXRayPrimitive(left=a, right=b)
    assert c.name == "a+b"


def test_composed_primitive_wire_in_hooks_union():
    a = _FakePrimitive("a", ("sensitivity_map", "bit_allocator"), 1.0)
    b = _FakePrimitive("b", ("pareto_constraint", "bit_allocator"), 2.0)
    c = ComposedXRayPrimitive(left=a, right=b)
    assert set(c.wire_in_hooks) == {
        "sensitivity_map",
        "bit_allocator",
        "pareto_constraint",
    }


def test_composed_primitive_compute_returns_dict_value():
    a = _FakePrimitive("a", ("sensitivity_map",), 1.0)
    b = _FakePrimitive("b", ("pareto_constraint",), 2.0)
    c = ComposedXRayPrimitive(left=a, right=b)
    result = c.compute(None)
    assert isinstance(result.primitive_value, dict)
    assert "left" in result.primitive_value
    assert "right" in result.primitive_value


def test_composed_primitive_evidence_weakest_wins():
    """When composing council-deliberation and mathematical-derivation,
    council-deliberation (weaker) wins."""
    a = _FakePrimitive("a", ("sensitivity_map",), 1.0, grade="mathematical-derivation")
    b = _FakePrimitive("b", ("pareto_constraint",), 2.0, grade="council-deliberation")
    c = ComposedXRayPrimitive(left=a, right=b)
    result = c.compute(None)
    assert result.evidence_grade == "council-deliberation"


def test_composed_primitive_chains_via_compose_with():
    a = _FakePrimitive("a", ("sensitivity_map",), 1.0)
    b = _FakePrimitive("b", ("pareto_constraint",), 2.0)
    d = _FakePrimitive("d", ("bit_allocator",), 4.0)
    c = ComposedXRayPrimitive(left=a, right=b)
    chained = c.compose_with(d)
    assert chained.name == "a+b+d"
    assert set(chained.wire_in_hooks) == {
        "sensitivity_map",
        "pareto_constraint",
        "bit_allocator",
    }


def test_result_default_metadata_is_empty_mapping():
    r = XRayPrimitiveResult(
        primitive_name="x",
        archive_or_video_path=None,
        archive_sha256=None,
        primitive_value=1.0,
        evidence_grade="mathematical-derivation",
        confidence_band=None,
        composes_with=(),
        wire_in_hooks_engaged=("sensitivity_map",),
    )
    assert r.metadata == {}


def test_result_schema_version_default():
    r = XRayPrimitiveResult(
        primitive_name="x",
        archive_or_video_path=None,
        archive_sha256=None,
        primitive_value=1.0,
        evidence_grade="mathematical-derivation",
        confidence_band=None,
        composes_with=(),
        wire_in_hooks_engaged=("sensitivity_map",),
    )
    assert r.schema_version == XRAY_PRIMITIVE_SCHEMA_VERSION
