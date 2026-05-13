"""Catalog #169 (WAVE-7-MED-FIX, REVIEW-OMNI A2-1) tests.

Bug-class anchor: REVIEW-OMNI 2026-05-12 noted that CompressAI primitives
were not in `canonical_primitive_inventory()`. FIX-J landed the 3 rows
(``compressai_factorized_prior``, ``compressai_balle_hyperprior``,
``compressai_cheng2020``) on 2026-05-12. This gate prevents regression.

Coverage targets:
- 0-violation when all 3 CompressAI primitives present
- violation when one missing
- violation when all missing
- strict mode raises PreflightError
- non-strict returns the violation list
- verbose mode prints diagnostic banner
- gracefully handles import failure
- gracefully handles invocation exception
"""

from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    check_compressai_primitives_registered_in_canonical_inventory,
)


def test_canonical_inventory_has_all_3_compressai_primitives() -> None:
    """Live-repo invariant: all 3 CompressAI primitives are present."""
    violations = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"Expected 0 violations on canonical inventory, got: {violations}"
    )


def test_strict_mode_passes_when_all_present() -> None:
    """Strict mode does not raise when inventory is complete."""
    # Should not raise.
    check_compressai_primitives_registered_in_canonical_inventory(
        strict=True, verbose=False,
    )


def test_one_missing_primitive_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing one CompressAI primitive surfaces 1 violation."""
    from tac.composition import registry as reg_mod

    real_inv = reg_mod.canonical_primitive_inventory()

    def fake_inv():
        # Drop the cheng2020 row.
        return [r for r in real_inv if r.primitive_id != "compressai_cheng2020"]

    monkeypatch.setattr(reg_mod, "canonical_primitive_inventory", fake_inv)
    # Also patch the import path within the check.
    import tac.composition.registry  # noqa: F401

    violations = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "compressai_cheng2020" in violations[0]


def test_all_missing_primitives_violations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing all CompressAI primitives surfaces 3 violations."""
    from tac.composition import registry as reg_mod

    real_inv = reg_mod.canonical_primitive_inventory()

    def fake_inv():
        return [r for r in real_inv if not r.primitive_id.startswith("compressai_")]

    monkeypatch.setattr(reg_mod, "canonical_primitive_inventory", fake_inv)

    violations = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert len(violations) == 3
    for required in ("compressai_factorized_prior",
                     "compressai_balle_hyperprior",
                     "compressai_cheng2020"):
        assert any(required in v for v in violations), (
            f"Expected {required} mentioned in violations: {violations}"
        )


def test_strict_mode_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode raises PreflightError when violations present."""
    from tac.composition import registry as reg_mod

    real_inv = reg_mod.canonical_primitive_inventory()

    def fake_inv():
        return [r for r in real_inv if r.primitive_id != "compressai_factorized_prior"]

    monkeypatch.setattr(reg_mod, "canonical_primitive_inventory", fake_inv)

    with pytest.raises(PreflightError) as excinfo:
        check_compressai_primitives_registered_in_canonical_inventory(
            strict=True, verbose=False,
        )
    assert "Catalog #169" in str(excinfo.value)
    assert "REVIEW-OMNI" in str(excinfo.value)


def test_verbose_prints_ok_when_clean(capsys: pytest.CaptureFixture) -> None:
    """Verbose mode prints OK banner when 0 violations."""
    check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[compressai-primitives-registered] OK" in out


def test_verbose_prints_violations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Verbose mode prints violation count and details."""
    from tac.composition import registry as reg_mod

    real_inv = reg_mod.canonical_primitive_inventory()

    def fake_inv():
        return [r for r in real_inv if not r.primitive_id.startswith("compressai_")]

    monkeypatch.setattr(reg_mod, "canonical_primitive_inventory", fake_inv)

    check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=True,
    )
    out = capsys.readouterr().out
    assert "[compressai-primitives-registered]" in out
    assert "violation" in out


def test_handles_inventory_invocation_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When canonical_primitive_inventory() raises, the check captures it."""
    from tac.composition import registry as reg_mod

    def boom():
        raise RuntimeError("synthetic")

    monkeypatch.setattr(reg_mod, "canonical_primitive_inventory", boom)

    violations = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "RuntimeError" in violations[0]
    assert "synthetic" in violations[0]


def test_returns_list_type() -> None:
    """Non-strict mode returns a list."""
    out = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert isinstance(out, list)


def test_required_ids_constant_has_expected_3_entries() -> None:
    """The required-IDs constant declares exactly 3 entries."""
    from tac.preflight import _CHECK_169_REQUIRED_PRIMITIVE_IDS
    assert len(_CHECK_169_REQUIRED_PRIMITIVE_IDS) == 3
    assert "compressai_factorized_prior" in _CHECK_169_REQUIRED_PRIMITIVE_IDS
    assert "compressai_balle_hyperprior" in _CHECK_169_REQUIRED_PRIMITIVE_IDS
    assert "compressai_cheng2020" in _CHECK_169_REQUIRED_PRIMITIVE_IDS


def test_non_strict_does_not_raise_with_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-strict mode returns violations rather than raising."""
    from tac.composition import registry as reg_mod
    real_inv = reg_mod.canonical_primitive_inventory()
    monkeypatch.setattr(
        reg_mod,
        "canonical_primitive_inventory",
        lambda: [r for r in real_inv if not r.primitive_id.startswith("compressai_")],
    )
    out = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert len(out) == 3  # Does not raise.


def test_partial_missing_returns_correct_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing 2 of 3 surfaces 2 violations."""
    from tac.composition import registry as reg_mod
    real_inv = reg_mod.canonical_primitive_inventory()
    keep = "compressai_balle_hyperprior"
    monkeypatch.setattr(
        reg_mod,
        "canonical_primitive_inventory",
        lambda: [r for r in real_inv if not r.primitive_id.startswith("compressai_")
                 or r.primitive_id == keep],
    )
    out = check_compressai_primitives_registered_in_canonical_inventory(
        strict=False, verbose=False,
    )
    assert len(out) == 2
