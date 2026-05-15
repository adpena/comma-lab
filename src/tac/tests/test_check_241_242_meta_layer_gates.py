# SPDX-License-Identifier: MIT
"""Catalog #241 + #242 META layer STRICT preflight gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_register_substrate_contract_fields_canonical,
    check_substrate_uses_register_decorator_or_explicitly_legacy_tagged,
)


def _make_substrate_trainer(
    tmp_path: Path,
    *,
    name: str,
    has_decorator: bool = False,
    waiver: str | None = None,
) -> Path:
    """Materialize a synthetic substrate trainer file under ``<tmp>/experiments/``."""
    exp = tmp_path / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    target = exp / f"train_substrate_{name}.py"
    lines: list[str] = ['"""Synthetic substrate trainer."""', "from __future__ import annotations"]
    if waiver:
        lines.append(f"# {waiver}")
    if has_decorator:
        lines.extend(
            [
                "from tac.substrate_registry import register_substrate, SubstrateContract",
                "@register_substrate(SubstrateContract(id='x', lane_id='lane_x_20260515', target_modes=('research_substrate',), deployment_target='desktop_research', council_verdict_provenance=None, archive_grammar='g', parser_section_manifest={}, inflate_runtime_loc_budget=80, runtime_dep_closure=('torch',), export_format='fp16_brotli', score_aware_loss='scorer_loss_terms_btchw', bolt_on_loc_budget=200, no_op_detector_planned=True, archive_bytes_added=None, score_improvement_mechanism_status='RESEARCH_ONLY', runtime_overlay_consumed=False, recipe_smoke_only=True, recipe_research_only=True, recipe_min_smoke_gpu='T4', recipe_min_vram_gb=16, recipe_pyav_decode_strategy='cpu_thread_async_upload', recipe_canary_status='independent_substrate', recipe_video_input_strategy='per_dispatch_local_copy', recipe_canary_dependency=None, cost_band_epochs=10, cost_band_gpu_key='T4', cost_band_platform_key='modal', cost_band_p50_usd=0.10, hook_sensitivity_contribution='not_applicable_with_rationale', hook_pareto_constraint='not_applicable_with_rationale', hook_bit_allocator_class='not_applicable_with_rationale', hook_autopilot_ranker_class_shift_token=None, hook_continual_learning_anchor_kind='not_applicable_with_rationale', hook_probe_disambiguator=None))",
                "def main(argv=None): return 0",
            ]
        )
    else:
        lines.append("def main(argv=None): return 0")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Catalog #241 (decorator-or-legacy-tagged) — WARN-ONLY at landing
# ---------------------------------------------------------------------------


def test_check_241_no_experiments_dir_returns_empty(tmp_path: Path) -> None:
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert out == []


def test_check_241_decorator_passes(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="alpha", has_decorator=True)
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert out == []


def test_check_241_legacy_waiver_with_real_rationale_passes(tmp_path: Path) -> None:
    _make_substrate_trainer(
        tmp_path,
        name="legacy_a",
        has_decorator=False,
        waiver="LEGACY_SUBSTRATE_PRE_META_LAYER:pre-2026-05-15 migration backlog item #1",
    )
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert out == []


def test_check_241_no_decorator_no_waiver_flagged(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="missing", has_decorator=False)
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert len(out) == 1
    assert "missing" in out[0]
    assert "@register_substrate" in out[0]


def test_check_241_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _make_substrate_trainer(
        tmp_path,
        name="placeholder",
        has_decorator=False,
        waiver="LEGACY_SUBSTRATE_PRE_META_LAYER:<rationale>",
    )
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert len(out) == 1
    assert "placeholder" in out[0]


def test_check_241_empty_rationale_rejected(tmp_path: Path) -> None:
    _make_substrate_trainer(
        tmp_path,
        name="empty",
        has_decorator=False,
        waiver="LEGACY_SUBSTRATE_PRE_META_LAYER:",
    )
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_check_241_strict_raises_on_violation(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="raisecase", has_decorator=False)
    with pytest.raises(PreflightError, match="Catalog #241"):
        check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
            repo_root=tmp_path, strict=True,
        )


def test_check_241_strict_silent_on_clean(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="cleanup", has_decorator=True)
    # No exception.
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path, strict=True,
    )
    assert out == []


def test_check_241_aggregates_multiple_violations(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="a", has_decorator=False)
    _make_substrate_trainer(tmp_path, name="b", has_decorator=False)
    _make_substrate_trainer(tmp_path, name="c", has_decorator=True)  # OK
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=tmp_path
    )
    assert len(out) == 2
    flagged_names = {p.split("/")[-1] for p in (v.split(":")[0] for v in out)}
    assert flagged_names == {"train_substrate_a.py", "train_substrate_b.py"}


def test_check_241_string_repo_root_accepted(tmp_path: Path) -> None:
    _make_substrate_trainer(tmp_path, name="strpath", has_decorator=True)
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(
        repo_root=str(tmp_path)
    )
    assert out == []


def test_check_241_live_repo_warn_only(tmp_path: Path) -> None:
    """The live repo currently has 31 legacy substrates; gate is WARN-ONLY.

    This test pins the warn-only behavior so a future operator strict-flip
    must update this test alongside the migration completion.
    """
    out = check_substrate_uses_register_decorator_or_explicitly_legacy_tagged()
    # Live count varies as substrates migrate; just assert >=0 (no exception).
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Catalog #242 (contract fields canonical) — STRICT-from-byte-one
# ---------------------------------------------------------------------------


def test_check_242_clean_registry_returns_empty() -> None:
    """The example_template registers cleanly — gate finds zero errors."""
    out = check_register_substrate_contract_fields_canonical()
    assert out == []


def test_check_242_strict_silent_on_clean() -> None:
    out = check_register_substrate_contract_fields_canonical(strict=True)
    assert out == []


def test_check_242_live_repo_zero_violations() -> None:
    """STRICT-from-byte-one regression guard: live count MUST be 0."""
    out = check_register_substrate_contract_fields_canonical(strict=True, verbose=True)
    assert out == []


def test_check_242_strict_raises_when_registry_corrupted(monkeypatch) -> None:
    """If a corrupted contract were ever registered, the gate raises."""
    from tac.substrate_registry import (
        SubstrateContract,
        SubstrateContractError,
        _REGISTERED_SUBSTRATES,
    )

    # Inject a SubstrateContract instance that bypasses validation by
    # constructing a corrupted to_dict view via monkeypatching.
    snapshot = dict(_REGISTERED_SUBSTRATES)
    try:

        class _Corrupted:
            id = "corrupted_test_subject"

            def to_dict(self) -> dict:
                # Missing required fields.
                return {"id": "corrupted_test_subject"}

        # Inject directly (test-only back door simulating registry mutation).
        _REGISTERED_SUBSTRATES["corrupted_test_subject"] = _Corrupted()  # type: ignore[assignment]
        with pytest.raises(PreflightError, match="Catalog #242"):
            check_register_substrate_contract_fields_canonical(strict=True)
    finally:
        # Restore.
        _REGISTERED_SUBSTRATES.clear()
        _REGISTERED_SUBSTRATES.update(snapshot)


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_wires_check_241_warn_only_and_check_242_strict() -> None:
    """preflight_all must wire #241 (strict=False) and #242 (strict=True).

    Pins the exact callsite invariants per CLAUDE.md "Strict-flip atomicity
    rule" — any future strict-flip on #241 must update this assertion in the
    same commit batch.
    """
    pre = Path("src/tac/preflight.py").read_text(encoding="utf-8")
    assert (
        "check_substrate_uses_register_decorator_or_explicitly_legacy_tagged(\n"
        "            strict=False, verbose=verbose,\n"
        "        )"
    ) in pre
    assert (
        "check_register_substrate_contract_fields_canonical(\n"
        "            strict=True, verbose=verbose,\n"
        "        )"
    ) in pre
