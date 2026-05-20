# SPDX-License-Identifier: MIT
"""Tests for the dual-tier cathedral consumer architecture (Dim 6 + Catalog #357).

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 6 Step 6.1-6.4 (operator blanket
approval 2026-05-20) + canonical contract extension at
``src/tac/cathedral/consumer_contract.py``.

Test coverage:

- ConsumerTier IntEnum membership + canonical-name pinning
- DEFAULT_CONSUMER_TIER backward-compat semantics
- ConsumerRegistration accepts consumer_tier field; defaults to Tier A
- ConsumerRegistration rejects wrong-type consumer_tier
- validate_consumer_module: Tier A backward-compat (omits CONSUMER_TIER)
- validate_consumer_module: Tier A explicit declaration
- validate_consumer_module: Tier B explicit declaration
- validate_consumer_module: wrong-type CONSUMER_TIER flagged
- Protocol runtime_checkable backward compat (existing 44+ consumers)
- is_tier_b_axis_tag_valid: contest/diagnostic accepted; predicted/advisory rejected
- validate_tier_b_contribution: comprehensive happy path + each rejection path
- Catalog #357 STRICT preflight gate: live-repo regression + synthetic Tier B fixture
- Catalog #357 waiver semantics: rationale accepted / placeholder rejected
- Catalog #185 sister-callable regression guard
- Orchestrator wire-in strict=True regression guard

Sister test files:

- ``src/tac/tests/test_cathedral_consumer_contract.py`` (Tier A canonical contract)
- ``src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py``
- ``src/tac/tests/test_check_341_cathedral_consumer_mps_prescreen_routing.py``
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    ConsumerRegistration,
    ConsumerTier,
    DEFAULT_CONSUMER_TIER,
    HookNumber,
    is_tier_b_axis_tag_valid,
    validate_consumer_module,
    validate_tier_b_contribution,
)
from tac.preflight import (
    PreflightError,
    check_cathedral_consumer_tier_b_declares_canonical_contract,
)


# ---------------------------------------------------------------------------
# ConsumerTier enum
# ---------------------------------------------------------------------------


def test_consumer_tier_has_exactly_two_canonical_members() -> None:
    """Dim 6 Step 6.1: dual-tier architecture has exactly Tier A + Tier B."""
    members = list(ConsumerTier)
    assert len(members) == 2
    assert ConsumerTier.TIER_A_OBSERVABILITY_ONLY in members
    assert ConsumerTier.TIER_B_SCORE_CONTRIBUTING in members


def test_consumer_tier_intenum_for_compactness() -> None:
    """Per project style, tier enum is IntEnum (mirrors HookNumber pattern)."""
    assert int(ConsumerTier.TIER_A_OBSERVABILITY_ONLY) == 1
    assert int(ConsumerTier.TIER_B_SCORE_CONTRIBUTING) == 2


def test_consumer_tier_names_pinned() -> None:
    """Canonical names must not drift (operator-mental-model gap closure relies
    on naming clarity)."""
    assert ConsumerTier.TIER_A_OBSERVABILITY_ONLY.name == "TIER_A_OBSERVABILITY_ONLY"
    assert ConsumerTier.TIER_B_SCORE_CONTRIBUTING.name == "TIER_B_SCORE_CONTRIBUTING"


def test_default_consumer_tier_is_tier_a_for_safety() -> None:
    """Backward-compat default per Dim 6 Step 6.2: Tier A is the canonical
    safe choice; existing 44+ production consumers stay Tier A by default."""
    assert DEFAULT_CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


# ---------------------------------------------------------------------------
# ConsumerRegistration accepts consumer_tier field
# ---------------------------------------------------------------------------


def test_consumer_registration_defaults_to_tier_a() -> None:
    """Backward compat: registration without explicit consumer_tier defaults
    to Tier A (preserves Catalog #341 semantics)."""
    reg = ConsumerRegistration(
        consumer_name="x",
        consumer_version="0.1.0",
        consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
        consumer_module_path="x",
        contract_compliant=True,
    )
    assert reg.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_registration_accepts_explicit_tier_a() -> None:
    reg = ConsumerRegistration(
        consumer_name="x",
        consumer_version="0.1.0",
        consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
        consumer_module_path="x",
        contract_compliant=True,
        consumer_tier=ConsumerTier.TIER_A_OBSERVABILITY_ONLY,
    )
    assert reg.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_registration_accepts_tier_b() -> None:
    reg = ConsumerRegistration(
        consumer_name="x",
        consumer_version="0.1.0",
        consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
        consumer_module_path="x",
        contract_compliant=True,
        consumer_tier=ConsumerTier.TIER_B_SCORE_CONTRIBUTING,
    )
    assert reg.consumer_tier == ConsumerTier.TIER_B_SCORE_CONTRIBUTING


def test_consumer_registration_rejects_wrong_type_tier() -> None:
    """Frozen-dataclass invariant: consumer_tier must be ConsumerTier."""
    with pytest.raises(ValueError, match="consumer_tier"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
            consumer_module_path="x",
            contract_compliant=True,
            consumer_tier="tier_a",  # type: ignore[arg-type]
        )


def test_consumer_registration_rejects_raw_int_tier() -> None:
    """Raw int (not the IntEnum member) must be rejected for type safety."""
    with pytest.raises(ValueError, match="consumer_tier"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
            consumer_module_path="x",
            contract_compliant=True,
            consumer_tier=1,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# validate_consumer_module respects CONSUMER_TIER
# ---------------------------------------------------------------------------


def _make_module(
    name: str = "fake_consumer",
    *,
    tier: ConsumerTier | None | object = "__omit__",
) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.CONSUMER_NAME = "test_consumer"
    mod.CONSUMER_VERSION = "1.0.0"
    mod.CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)
    mod.update_from_anchor = lambda anchor: None
    mod.consume_candidate = lambda candidate: {
        "predicted_delta_adjustment": 0.0,
        "rationale": "test",
        "axis_tag": "[predicted]",
    }
    if tier != "__omit__":
        mod.CONSUMER_TIER = tier  # type: ignore[assignment]
    return mod


def test_validate_omitted_consumer_tier_defaults_to_tier_a() -> None:
    """Backward compat: existing 44+ consumers that predate the dual-tier
    landing have no CONSUMER_TIER attribute — they must default to Tier A
    silently per Dim 6 Step 6.2 (zero migration burden)."""
    mod = _make_module()
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True
    assert reg.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_validate_explicit_tier_a_consumer() -> None:
    mod = _make_module(tier=ConsumerTier.TIER_A_OBSERVABILITY_ONLY)
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True
    assert reg.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_validate_explicit_tier_b_consumer() -> None:
    mod = _make_module(tier=ConsumerTier.TIER_B_SCORE_CONTRIBUTING)
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True
    assert reg.consumer_tier == ConsumerTier.TIER_B_SCORE_CONTRIBUTING


def test_validate_wrong_type_consumer_tier_flagged() -> None:
    """Present-but-wrong-type CONSUMER_TIER is a contract violation."""
    mod = _make_module(tier="tier_a")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_TIER" in err for err in reg.validation_errors)


def test_validate_raw_int_consumer_tier_flagged() -> None:
    """Per Dim 6 Step 6.2 contract: only ConsumerTier members accepted (not
    raw ints, to prevent accidental mistype)."""
    mod = _make_module(tier=2)
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_TIER" in err for err in reg.validation_errors)


# ---------------------------------------------------------------------------
# Protocol runtime_checkable backward compat (Catalog #335 sister)
# ---------------------------------------------------------------------------


def test_protocol_isinstance_check_backward_compatible() -> None:
    """Per Dim 6 Step 6.2: ``CONSUMER_TIER`` is INTENTIONALLY NOT a
    Protocol-required attribute so the runtime_checkable ``isinstance``
    check remains backward-compatible. Existing consumers omitting the
    attribute satisfy the Protocol; validation happens via
    ``validate_consumer_module``."""
    mod = _make_module()  # omits CONSUMER_TIER
    assert isinstance(mod, CathedralConsumerContract)


def test_protocol_isinstance_with_tier_b_consumer() -> None:
    """A Tier B consumer also satisfies the Protocol."""
    mod = _make_module(tier=ConsumerTier.TIER_B_SCORE_CONTRIBUTING)
    assert isinstance(mod, CathedralConsumerContract)


# ---------------------------------------------------------------------------
# is_tier_b_axis_tag_valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis_tag",
    [
        "[contest-CPU]",
        "[contest-CUDA]",
        "[diagnostic-CPU]",
        "[diagnostic-CUDA]",
    ],
)
def test_is_tier_b_axis_tag_valid_accepts_empirical(axis_tag: str) -> None:
    assert is_tier_b_axis_tag_valid(axis_tag) is True


@pytest.mark.parametrize(
    "axis_tag",
    [
        "[predicted]",
        "[advisory only]",
        "[macOS-CPU advisory]",
        "[MPS-PROXY]",
        "[MPS-research-signal]",
    ],
)
def test_is_tier_b_axis_tag_valid_rejects_speculative(axis_tag: str) -> None:
    assert is_tier_b_axis_tag_valid(axis_tag) is False


def test_is_tier_b_axis_tag_valid_rejects_empty() -> None:
    assert is_tier_b_axis_tag_valid("") is False
    assert is_tier_b_axis_tag_valid("   ") is False


def test_is_tier_b_axis_tag_valid_rejects_non_string() -> None:
    assert is_tier_b_axis_tag_valid(None) is False
    assert is_tier_b_axis_tag_valid(42) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_tier_b_contribution (runtime validator)
# ---------------------------------------------------------------------------


def _good_tier_b() -> dict:
    return {
        "predicted_delta_adjustment": -0.0015,
        "axis_tag": "[contest-CUDA]",
        "promotable": False,
        "provenance": {"kind": "CONTEST_ARCHIVE_MEMBER", "grade": "contest_cuda"},
        "rationale": "Bayesian posterior delta-S anchor",
    }


def test_validate_tier_b_contribution_happy_path() -> None:
    ok, errs = validate_tier_b_contribution(_good_tier_b())
    assert ok is True
    assert errs == ()


def test_validate_tier_b_contribution_rejects_predicted_axis() -> None:
    bad = _good_tier_b()
    bad["axis_tag"] = "[predicted]"
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("axis_tag" in e for e in errs)


def test_validate_tier_b_contribution_rejects_missing_provenance() -> None:
    bad = _good_tier_b()
    del bad["provenance"]
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("provenance" in e for e in errs)


def test_validate_tier_b_contribution_rejects_empty_provenance() -> None:
    bad = _good_tier_b()
    bad["provenance"] = {}
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("provenance" in e and "non-empty" in e for e in errs)


def test_validate_tier_b_contribution_rejects_non_mapping_provenance() -> None:
    bad = _good_tier_b()
    bad["provenance"] = ["not a mapping"]
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("provenance" in e for e in errs)


def test_validate_tier_b_contribution_rejects_promotable_true() -> None:
    bad = _good_tier_b()
    bad["promotable"] = True
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("promotable" in e for e in errs)


def test_validate_tier_b_contribution_rejects_missing_promotable() -> None:
    bad = _good_tier_b()
    del bad["promotable"]
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("promotable" in e for e in errs)


def test_validate_tier_b_contribution_rejects_nan_delta() -> None:
    bad = _good_tier_b()
    bad["predicted_delta_adjustment"] = float("nan")
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("NaN" in e for e in errs)


def test_validate_tier_b_contribution_rejects_inf_delta() -> None:
    bad = _good_tier_b()
    bad["predicted_delta_adjustment"] = float("inf")
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("infinite" in e for e in errs)


def test_validate_tier_b_contribution_rejects_bool_delta() -> None:
    """bool is a subclass of int but semantically not a score-delta."""
    bad = _good_tier_b()
    bad["predicted_delta_adjustment"] = True  # type: ignore[assignment]
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("predicted_delta_adjustment" in e for e in errs)


def test_validate_tier_b_contribution_rejects_missing_delta() -> None:
    bad = _good_tier_b()
    del bad["predicted_delta_adjustment"]
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("predicted_delta_adjustment" in e for e in errs)


def test_validate_tier_b_contribution_rejects_placeholder_rationale() -> None:
    bad = _good_tier_b()
    bad["rationale"] = "<rationale>"
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("placeholder" in e for e in errs)


def test_validate_tier_b_contribution_rejects_short_rationale() -> None:
    bad = _good_tier_b()
    bad["rationale"] = "ok"  # 2 chars < 4
    ok, errs = validate_tier_b_contribution(bad)
    assert ok is False
    assert any("rationale" in e for e in errs)


def test_validate_tier_b_contribution_rejects_non_mapping_input() -> None:
    ok, errs = validate_tier_b_contribution(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert ok is False
    assert any("Mapping" in e for e in errs)


def test_validate_tier_b_contribution_accepts_zero_delta() -> None:
    """Tier B MAY return 0.0 (no signal) without violating the contract."""
    good = _good_tier_b()
    good["predicted_delta_adjustment"] = 0.0
    ok, errs = validate_tier_b_contribution(good)
    assert ok is True


def test_validate_tier_b_contribution_accepts_positive_and_negative_delta() -> None:
    for delta in (-0.005, 0.001, -1.0, 0.5):
        good = _good_tier_b()
        good["predicted_delta_adjustment"] = delta
        ok, errs = validate_tier_b_contribution(good)
        assert ok is True, errs


# ---------------------------------------------------------------------------
# Catalog #357 STRICT preflight gate
# ---------------------------------------------------------------------------


def test_check_357_live_repo_regression_guard() -> None:
    """Live count: 0 — no Tier B consumers exist yet; gate is a structural
    foundation for future Tier B landings. If this count ever grows, EITHER
    a new Tier B consumer landed (verify it satisfies the contract) OR the
    gate regressed (fix the gate)."""
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, verbose=False
    )
    # Allow up to 5 in case sister Wave 1 lands a synthetic in-flight Tier B
    # but live anchor MUST be 0 at strict-flip per Catalog #185.
    assert len(violations) <= 5, (
        f"Catalog #357 live count grew: {len(violations)} violations. "
        f"Investigate: {violations[:3]}"
    )


def test_check_357_clean_tier_a_consumer_passes(tmp_path: Path) -> None:
    """A Tier A consumer (default; omits CONSUMER_TIER) does NOT trigger the
    Tier B contract gate."""
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/my_tier_a"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        "from tac.cathedral.consumer_contract import HookNumber\n"
        "CONSUMER_NAME = 'my_tier_a'\n"
        "CONSUMER_VERSION = '0.1.0'\n"
        "CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)\n"
        "def update_from_anchor(a): pass\n"
        "def consume_candidate(c):\n"
        "    return {'predicted_delta_adjustment': 0.0, 'rationale': 'tier a', "
        "'axis_tag': '[predicted]', 'promotable': False}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_synthetic_tier_b_compliant_passes(tmp_path: Path) -> None:
    """A well-formed Tier B consumer passes the gate."""
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/my_tier_b"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        "from tac.cathedral.consumer_contract import ConsumerTier, HookNumber\n"
        "CONSUMER_NAME = 'my_tier_b'\n"
        "CONSUMER_VERSION = '0.1.0'\n"
        "CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "def update_from_anchor(a): pass\n"
        "def consume_candidate(c):\n"
        "    return {'predicted_delta_adjustment': -0.001, 'rationale': 'tier b',\n"
        "            'axis_tag': '[contest-CUDA]', 'promotable': False,\n"
        "            'provenance': {'kind': 'CONTEST_ARCHIVE_MEMBER'}}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == [], violations


def test_check_357_tier_b_missing_provenance_flagged(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/bad_tier_b"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# no 'provenance' word + no promotable field\n"
        "def f(): pass\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert len(violations) >= 1
    assert any("provenance" in v for v in violations)


def test_check_357_tier_b_missing_promotable_flagged(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/bad_tier_b2"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# provenance present but promotable missing\n"
        "x = {'provenance': {'kind': 'x'}}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert any("promotable" in v for v in violations)


def test_check_357_tier_b_forbidden_predicted_axis_flagged(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/bad_tier_b3"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "x = {'provenance': {'k': 'v'}, 'promotable': False, "
        "'axis_tag': '[predicted]'}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert any("forbidden axis_tag form" in v for v in violations)


def test_check_357_string_literal_tier_b_not_in_scope(tmp_path: Path) -> None:
    """A string-literal mention of TIER_B_SCORE_CONTRIBUTING (e.g. in a
    docstring documenting the canonical promotion path) is NOT a Tier B
    declaration — the AST-based in-scope check (Catalog #168 sister
    discipline) requires an actual ``CONSUMER_TIER`` assignment.

    Bug class anchor: `findings_lagrangian_consumer/__init__.py` declares
    `CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY` but has the
    string `TIER_B_SCORE_CONTRIBUTING` in its docstring (documenting the
    promotion path per Dim 6 Step 6.5). A naive substring check
    false-flagged this Tier A consumer as Tier B.
    """
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/docstring_mention"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        '"""Docstring mentions TIER_B_SCORE_CONTRIBUTING for promotion path."""\n'
        "# string mention only\n"
        "MARKER = 'TIER_B_SCORE_CONTRIBUTING'\n"
        "x = {'provenance': {'k': 'v'}, 'promotable': False, "
        "'axis_tag': '[contest-CUDA]'}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == [], (
        "Docstring/string-literal mentions of TIER_B_SCORE_CONTRIBUTING "
        "must NOT be detected as Tier B declarations (AST sister of #168)"
    )


def test_check_357_tier_a_explicit_consumer_not_in_scope(tmp_path: Path) -> None:
    """A consumer that explicitly declares ``CONSUMER_TIER =
    ConsumerTier.TIER_A_OBSERVABILITY_ONLY`` (even with TIER_B mention in
    a docstring or comment) is out of scope — Tier A is covered by sister
    Catalog #341.

    Empirical anchor: `findings_lagrangian_consumer` post-Dim-6 landing.
    """
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/explicit_tier_a"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        '"""Docstring mentions TIER_B_SCORE_CONTRIBUTING promotion path."""\n'
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY\n"
        "# A Tier A consumer with axis_tag='[predicted]' is valid (Tier A)\n"
        "x = {'axis_tag': '[predicted]', 'predicted_delta_adjustment': 0.0}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/scaffold_tier_b"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# CATHEDRAL_CONSUMER_TIER_B_DEFERRED_OK:Phase 2 wire-in pending per Dim 6 Step 6.5\n"
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# missing canonical contract — waiver active\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_waiver_placeholder_rejected(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/scaffold_tier_b2"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# CATHEDRAL_CONSUMER_TIER_B_DEFERRED_OK:<rationale>\n"
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# missing canonical contract — placeholder waiver rejected\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert len(violations) > 0


def test_check_357_waiver_short_rationale_rejected(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/scaffold_tier_b3"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "# CATHEDRAL_CONSUMER_TIER_B_DEFERRED_OK:wip\n"  # 3 chars < 4
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert len(violations) > 0


def test_check_357_annotated_assignment_tier_b_in_scope(tmp_path: Path) -> None:
    """Per Catalog #168 (Assign vs AnnAssign META): the in-scope detector
    must handle both ``CONSUMER_TIER = ...`` and ``CONSUMER_TIER:
    ConsumerTier = ...`` forms."""
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/annotated_tier_b"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER: ConsumerTier = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# annotated form is in-scope but missing the canonical contract here\n"
        "x = {'axis_tag': '[predicted]'}\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert len(violations) > 0


def test_check_357_underscore_prefix_consumer_exempt(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/_reference_tier_b"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "# reference consumers (underscore prefix) exempt from contract\n"
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_strict_mode_raises(tmp_path: Path) -> None:
    consumer_dir = tmp_path / "src/tac/cathedral_consumers/bad_strict"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        "from tac.cathedral.consumer_contract import ConsumerTier\n"
        "CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING\n"
        "x = {'axis_tag': '[predicted]'}\n"
    )
    with pytest.raises(PreflightError, match="Catalog #357"):
        check_cathedral_consumer_tier_b_declares_canonical_contract(
            strict=True, repo_root=tmp_path
        )


def test_check_357_strict_silent_on_clean(tmp_path: Path) -> None:
    # Empty consumer dir => no Tier B consumers => silent on strict.
    (tmp_path / "src/tac/cathedral_consumers").mkdir(parents=True)
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=True, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_missing_consumer_dir_silent(tmp_path: Path) -> None:
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_357_string_repo_root_accepted(tmp_path: Path) -> None:
    """repo_root may be str or Path per canonical preflight conventions."""
    (tmp_path / "src/tac/cathedral_consumers").mkdir(parents=True)
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, repo_root=str(tmp_path)
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Catalog #185 sister-callable regression guard
# ---------------------------------------------------------------------------


def test_check_357_callable_via_globals() -> None:
    """Catalog #185 META-meta-meta gate requires the check function be
    callable via globals lookup; without this, the META-meta gate would
    falsely report Live count: 0 drift."""
    from tac import preflight as P

    fn = getattr(P, "check_cathedral_consumer_tier_b_declares_canonical_contract")
    assert callable(fn)
    out = fn(strict=False, verbose=False)
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Orchestrator wire-in strict=True regression guard
# ---------------------------------------------------------------------------


def test_check_357_wired_into_preflight_all_strict_true() -> None:
    """Per Catalog #176 META-meta gate: every STRICT preflight callsite must
    have a CLAUDE.md catalog row. This regression guard verifies the
    orchestrator wires this gate at strict=True."""
    pre_path = Path(__file__).resolve().parents[1] / "preflight.py"
    body = pre_path.read_text(encoding="utf-8")
    # Verify the callsite uses strict=True.
    assert (
        "check_cathedral_consumer_tier_b_declares_canonical_contract(\n"
        "            strict=True, verbose=verbose,\n"
        "        )"
    ) in body, "Catalog #357 orchestrator callsite must use strict=True"
