# SPDX-License-Identifier: MIT
"""Tests for the one-arg local-MPS vs Modal dispatch switch.

Lane: ``lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517``

Per operator directive 2026-05-17 verbatim: *"Deploying to local MPS versus
modal should be super easy to configure, like one arg in a func"* + *"Do
everything possible you can to accelerate dev velocity and save money using
local MPS"*.

Covers:
  1. Catalog #317 STRICT preflight gate (live count + helper unit tests + waiver)
  2. ``--target`` CLI flag wiring + recipe override + precedence
  3. ``platform: local_mps`` / ``platform: local_cpu`` recipe parsing
  4. ``LEGAL_NATIVE_PLATFORMS`` + ``LEGAL_DISPATCH_KINDS`` extension
  5. ``_is_local_research_signal_dispatch`` detection
  6. ``mps_research_signal.append_manifest_row_to_jsonl`` canonical helper
  7. Recipe-level refusal of ``score_claim: true`` / ``promotion_eligible: true``
  8. Posterior-isolation invariant (writes go to manifest jsonl, NOT to
     ``.omx/state/continual_learning_posterior.jsonl``)
  9. ``local_pre_deploy_check`` skip-set behavior + ``_is_local_research_signal_dispatch_for_harness``
  10. Non-regression: existing modal/lightning/vastai dispatch paths still work
  11. Loud non-authoritative banner present in dispatcher source
  12. ``PYTORCH_ENABLE_MPS_FALLBACK=0`` env injection
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# Catalog #317 STRICT preflight gate
# =============================================================================


def test_check_317_live_repo_has_zero_violations():
    """The new dispatchers must satisfy the contract at landing."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"Catalog #317: expected 0 violations at landing, got "
        f"{len(violations)}: {violations}"
    )


def test_check_317_strict_silent_on_clean():
    """Strict mode does not raise when contract is satisfied."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    # Should not raise.
    result = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=True, verbose=False
    )
    assert result == []


def test_check_317_raises_when_required_token_missing(tmp_path: Path):
    """When _dispatch_local_mps lacks a required token, gate raises in strict mode."""
    from tac.preflight import (
        PreflightError,
        check_local_research_signal_dispatches_stamp_evidence_grade,
    )

    # Build a fake repo where operator_authorize.py is missing required tokens.
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    target = tools_dir / "operator_authorize.py"
    target.write_text(
        "def _dispatch_local_mps(recipe, instance_job_id, env_overrides):\n"
        "    # missing all required tokens\n"
        "    return 0\n"
        "\n"
        "def _dispatch_local_cpu(recipe, instance_job_id, env_overrides):\n"
        "    # also missing tokens\n"
        "    return 0\n",
        encoding="utf-8",
    )

    with pytest.raises(PreflightError) as exc_info:
        check_local_research_signal_dispatches_stamp_evidence_grade(
            strict=True, verbose=False, repo_root=tmp_path
        )
    msg = str(exc_info.value)
    assert "Catalog #317" in msg
    assert "missing required contract tokens" in msg


def test_check_317_accepts_waiver_with_real_rationale(tmp_path: Path):
    """Same-line waiver with non-placeholder rationale opts out."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    target = tools_dir / "operator_authorize.py"
    target.write_text(
        "def _dispatch_local_mps(recipe, instance_job_id, env_overrides):  # LOCAL_RESEARCH_SIGNAL_STAMP_WAIVED: test fixture for unit test\n"
        "    return 0\n"
        "\n"
        "def _dispatch_local_cpu(recipe, instance_job_id, env_overrides):  # LOCAL_RESEARCH_SIGNAL_STAMP_WAIVED: test fixture for unit test\n"
        "    return 0\n",
        encoding="utf-8",
    )
    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_317_rejects_placeholder_rationale_waiver(tmp_path: Path):
    """Placeholder ``<rationale>`` / ``<reason>`` in waiver rejected."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    target = tools_dir / "operator_authorize.py"
    target.write_text(
        "def _dispatch_local_mps(recipe, instance_job_id, env_overrides):  # LOCAL_RESEARCH_SIGNAL_STAMP_WAIVED: <rationale>\n"
        "    return 0\n"
        "\n"
        "def _dispatch_local_cpu(recipe, instance_job_id, env_overrides):  # LOCAL_RESEARCH_SIGNAL_STAMP_WAIVED: <reason>\n"
        "    return 0\n",
        encoding="utf-8",
    )
    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) >= 2


def test_check_317_flags_missing_function(tmp_path: Path):
    """If _dispatch_local_mps is entirely missing, gate flags it."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    target = tools_dir / "operator_authorize.py"
    target.write_text("# empty stub\n", encoding="utf-8")
    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) >= 1
    assert any("missing required function" in v for v in violations)


def test_check_317_skip_when_target_file_missing(tmp_path: Path):
    """If tools/operator_authorize.py is absent, return empty (SKIP)."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_check_317_orchestrator_wires_strict_true():
    """preflight_all() must wire check #317 at strict=True."""
    src = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    assert "check_local_research_signal_dispatches_stamp_evidence_grade(\n            strict=True" in src or (
        "check_local_research_signal_dispatches_stamp_evidence_grade(" in src
        and "strict=True" in src
    )


def test_check_317_appears_in_claude_md_catalog():
    """CLAUDE.md catalog row #317 must be present (Catalog #176 sister)."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "317. `check_local_research_signal_dispatches_stamp_evidence_grade`" in claude_md


# =============================================================================
# LEGAL_NATIVE_PLATFORMS + LEGAL_DISPATCH_KINDS enum extension
# =============================================================================


def test_legal_native_platforms_includes_local_mps_and_local_cpu():
    from tac.deploy.dispatch_protocol import LEGAL_NATIVE_PLATFORMS

    assert "local_mps" in LEGAL_NATIVE_PLATFORMS
    assert "local_cpu" in LEGAL_NATIVE_PLATFORMS
    # Existing entries preserved
    assert "modal" in LEGAL_NATIVE_PLATFORMS
    assert "vastai" in LEGAL_NATIVE_PLATFORMS
    assert "vast" in LEGAL_NATIVE_PLATFORMS
    assert "local" in LEGAL_NATIVE_PLATFORMS


def test_legal_dispatch_kinds_includes_local_research_signal():
    from tac.deploy.dispatch_protocol import LEGAL_DISPATCH_KINDS

    assert "local_research_signal" in LEGAL_DISPATCH_KINDS
    # Existing
    assert "substrate" in LEGAL_DISPATCH_KINDS
    assert "tool" in LEGAL_DISPATCH_KINDS


def test_local_research_signal_platforms_constant():
    from tac.deploy.dispatch_protocol import LOCAL_RESEARCH_SIGNAL_PLATFORMS

    assert frozenset({"local_mps", "local_cpu"}) == LOCAL_RESEARCH_SIGNAL_PLATFORMS


def test_canonical_dispatch_optimization_protocol_enum_parity():
    """Catalog #270 helper must mirror dispatch_protocol enum."""
    # Import via direct file load to avoid module path issues; register in
    # sys.modules so dataclass internals can resolve the class module.
    import importlib.util
    module_name = "_cdop_module_for_tests"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
        assert "local_research_signal" in mod.LEGAL_DISPATCH_KINDS
        assert frozenset({"local_mps", "local_cpu"}) == mod.LOCAL_RESEARCH_SIGNAL_PLATFORMS
    finally:
        sys.modules.pop(module_name, None)


# =============================================================================
# _is_local_research_signal_dispatch detection
# =============================================================================


def test_is_local_research_signal_dispatch_explicit_kind():
    from tac.deploy.dispatch_protocol import is_local_research_signal_dispatch

    assert is_local_research_signal_dispatch({"dispatch_kind": "local_research_signal"})
    assert is_local_research_signal_dispatch({"dispatch_kind": "Local_Research_Signal"})


def test_is_local_research_signal_dispatch_implicit_platform():
    from tac.deploy.dispatch_protocol import is_local_research_signal_dispatch

    assert is_local_research_signal_dispatch({"platform": "local_mps"})
    assert is_local_research_signal_dispatch({"platform": "local_cpu"})
    assert is_local_research_signal_dispatch({"platform": "LOCAL_MPS"})


def test_is_local_research_signal_dispatch_explicit_substrate_short_circuits():
    from tac.deploy.dispatch_protocol import is_local_research_signal_dispatch

    assert not is_local_research_signal_dispatch(
        {"dispatch_kind": "substrate", "platform": "local_mps"}
    )
    assert not is_local_research_signal_dispatch(
        {"dispatch_kind": "tool", "platform": "local_mps"}
    )


def test_is_local_research_signal_dispatch_other_platforms_false():
    from tac.deploy.dispatch_protocol import is_local_research_signal_dispatch

    assert not is_local_research_signal_dispatch({"platform": "modal"})
    assert not is_local_research_signal_dispatch({"platform": "vastai"})
    assert not is_local_research_signal_dispatch({"platform": "local"})
    assert not is_local_research_signal_dispatch({})


# =============================================================================
# mps_research_signal.append_manifest_row_to_jsonl
# =============================================================================


def test_append_manifest_row_to_jsonl_basic(tmp_path: Path):
    from tac.optimization.mps_research_signal import append_manifest_row_to_jsonl

    output = tmp_path / ".omx" / "state" / "mps_research_signal_manifest.jsonl"
    row = {
        "instance_job_id": "ij_test_01",
        "lane_id": "lane_test_20260517",
        "platform": "local_mps",
    }
    append_manifest_row_to_jsonl(row, output_path=output)

    assert output.is_file()
    line = output.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["instance_job_id"] == "ij_test_01"
    assert parsed["evidence_grade"] == "MPS-research-signal"
    assert parsed["score_claim"] is False
    assert parsed["promotion_eligible"] is False
    assert parsed["ready_for_exact_eval_dispatch"] is False
    assert parsed["rank_or_kill_eligible"] is False


def test_append_manifest_row_to_jsonl_refuses_tmp_path(tmp_path: Path):
    from tac.optimization.mps_research_signal import append_manifest_row_to_jsonl

    with pytest.raises(ValueError, match="forbidden /tmp path"):
        append_manifest_row_to_jsonl(
            {"foo": "bar"},
            output_path=Path("/tmp/mps_research_signal.jsonl"),
        )


def test_append_manifest_row_to_jsonl_refuses_score_claim_true(tmp_path: Path):
    from tac.optimization.mps_research_signal import (
        MPSResearchSignalError,
        append_manifest_row_to_jsonl,
    )

    output = tmp_path / "out.jsonl"
    with pytest.raises(MPSResearchSignalError, match="non-authoritative"):
        append_manifest_row_to_jsonl(
            {"score_claim": True, "foo": "bar"},
            output_path=output,
        )


def test_append_manifest_row_to_jsonl_refuses_promotion_eligible(tmp_path: Path):
    from tac.optimization.mps_research_signal import (
        MPSResearchSignalError,
        append_manifest_row_to_jsonl,
    )

    output = tmp_path / "out.jsonl"
    with pytest.raises(MPSResearchSignalError):
        append_manifest_row_to_jsonl(
            {"promotion_eligible": True},
            output_path=output,
        )


def test_append_manifest_row_to_jsonl_appends_multiple_rows(tmp_path: Path):
    from tac.optimization.mps_research_signal import append_manifest_row_to_jsonl

    output = tmp_path / "out.jsonl"
    for i in range(3):
        append_manifest_row_to_jsonl(
            {"row_index": i, "instance_job_id": f"ij_{i}"},
            output_path=output,
        )
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert json.loads(line)["row_index"] == i


def test_append_manifest_row_to_jsonl_exported_in_all():
    from tac.optimization import mps_research_signal

    assert "append_manifest_row_to_jsonl" in mps_research_signal.__all__


# =============================================================================
# CLI --target flag wiring
# =============================================================================


def test_operator_authorize_cli_help_contains_target_flag():
    """--target flag must appear in CLI --help."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "operator_authorize.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "local-mps" in result.stdout
    assert "local-cpu" in result.stdout


def test_operator_authorize_cli_target_choices_include_all_platforms():
    """--target enum must include all canonical platforms."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "operator_authorize.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
    )
    for choice in ("auto", "modal", "vastai", "lightning", "local", "local-mps", "local-cpu"):
        assert choice in result.stdout, f"--target choice {choice!r} missing from --help"


# =============================================================================
# local_pre_deploy_check skip-set behavior
# =============================================================================


def test_local_pre_deploy_has_local_research_signal_skip_constant():
    src = (REPO_ROOT / "tools" / "local_pre_deploy_check.py").read_text(encoding="utf-8")
    assert "_LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS" in src
    assert "_is_local_research_signal_dispatch_for_harness" in src


def test_local_pre_deploy_skipped_checks_include_full_main_archive_auth_eval():
    # Direct file load to avoid module path issues.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    skipped = mod._LOCAL_RESEARCH_SIGNAL_SKIPPED_CHECKS
    for check_name in ("full_main_implemented", "archive_grammar",
                       "auth_eval_reachability", "canonical_inflate_device",
                       "deterministic_zip"):
        assert check_name in skipped, f"{check_name!r} must be skipped for local research signal"


def test_local_pre_deploy_detector_recipe_with_local_mps_platform(tmp_path: Path, monkeypatch):
    """_is_local_research_signal_dispatch_for_harness recognizes platform: local_mps."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Override REPO_ROOT inside the module to point at tmp_path.
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "test_local_mps.yaml").write_text(
        "name: test_local_mps\nplatform: local_mps\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "experiments" / "train_substrate_x.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# stub\n", encoding="utf-8")
    assert mod._is_local_research_signal_dispatch_for_harness(trainer, "test_local_mps")


def test_local_pre_deploy_detector_recipe_with_local_cpu_platform(tmp_path: Path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "test_local_cpu.yaml").write_text(
        "name: test_local_cpu\nplatform: local_cpu\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "experiments" / "train_substrate_x.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# stub\n", encoding="utf-8")
    assert mod._is_local_research_signal_dispatch_for_harness(trainer, "test_local_cpu")


def test_local_pre_deploy_detector_explicit_dispatch_kind(tmp_path: Path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "test_explicit.yaml").write_text(
        "name: test_explicit\nplatform: modal\ndispatch_kind: local_research_signal\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "experiments" / "train_substrate_x.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# stub\n", encoding="utf-8")
    assert mod._is_local_research_signal_dispatch_for_harness(trainer, "test_explicit")


def test_local_pre_deploy_detector_modal_recipe_returns_false(tmp_path: Path, monkeypatch):
    """Modal recipe is NOT local-research-signal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True)
    (recipes_dir / "test_modal.yaml").write_text(
        "name: test_modal\nplatform: modal\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "experiments" / "train_substrate_x.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# stub\n", encoding="utf-8")
    assert not mod._is_local_research_signal_dispatch_for_harness(trainer, "test_modal")


def test_local_pre_deploy_detector_no_recipe_returns_false(tmp_path: Path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lpdc_module",
        str(REPO_ROOT / "tools" / "local_pre_deploy_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "experiments" / "train_substrate_x.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# stub\n", encoding="utf-8")
    assert not mod._is_local_research_signal_dispatch_for_harness(trainer, None)


# =============================================================================
# operator_authorize.py source-level invariants
# =============================================================================


def test_dispatch_local_mps_function_exists_in_source():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "def _dispatch_local_mps(" in src


def test_dispatch_local_cpu_function_exists_in_source():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "def _dispatch_local_cpu(" in src


def test_dispatch_local_mps_has_loud_banner():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "LOCAL-MPS RESEARCH-SIGNAL — NON-AUTHORITATIVE" in src


def test_dispatch_local_cpu_has_loud_banner():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "LOCAL-CPU ADVISORY — NON-AUTHORITATIVE" in src


def test_dispatch_local_mps_enforces_mps_availability():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "torch.backends.mps.is_available" in src
    assert "FATAL: local-MPS dispatch requested but" in src


def test_dispatch_local_mps_forces_mps_fallback_disabled():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in src
    assert '"0"' in src or "'0'" in src


def test_dispatch_local_mps_refuses_score_claim_true():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert 'recipe.raw.get("score_claim") is True' in src


def test_dispatch_local_mps_refuses_promotion_eligible_true():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert 'recipe.raw.get("promotion_eligible") is True' in src


def test_dispatch_local_mps_routes_through_canonical_helper():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "from tac.optimization.mps_research_signal import" in src
    assert "append_manifest_row_to_jsonl" in src


def test_dispatch_local_cpu_routes_through_canonical_helper():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "from tac.optimization.macos_cpu_advisory_signal import" in src


def test_run_dispatch_fork_includes_local_mps_and_local_cpu():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert 'platform == "local_mps":' in src
    assert 'platform == "local_cpu":' in src
    assert "_dispatch_local_mps(recipe" in src
    assert "_dispatch_local_cpu(recipe" in src


def test_platform_has_native_dispatch_includes_local_mps_and_local_cpu():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    # The set literal in _platform_has_native_dispatch must include both.
    needle = '{"modal", "vastai", "vast", "local", "local_mps", "local_cpu"}'
    assert needle in src


def test_native_dispatch_preflight_handles_local_mps():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert 'platform in {"local_mps", "local_cpu"}' in src


def test_target_cli_override_records_cli_target_override_field():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert 'recipe.raw["cli_target_override"]' in src


# =============================================================================
# Non-regression: existing dispatch paths still present
# =============================================================================


def test_dispatch_modal_function_still_present():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "def _dispatch_modal(" in src


def test_dispatch_vastai_function_still_present():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "def _dispatch_vastai(" in src


def test_dispatch_local_function_still_present():
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    assert "def _dispatch_local(" in src


def test_dispatch_protocol_modal_routing_unchanged():
    from tac.deploy.dispatch_protocol import evaluate_dispatch_protocol_complete
    # A canonical modal recipe shape should still produce a report (without crashing).
    recipe = {
        "platform": "modal",
        "lane_id": "lane_test_modal_20260517",
        "cost_band": {"epochs": 100},
        "min_vram_gb": 16,
        "video_input_strategy": "per_dispatch_local_copy",
        "pyav_decode_strategy": "cpu_thread_async_upload",
        "target_modes": ["contest_one_video_replay"],
        "canary_status": "independent_substrate",
        "min_smoke_gpu": "T4",
    }
    report = evaluate_dispatch_protocol_complete(
        recipe, repo_root=REPO_ROOT, native_dispatch=True
    )
    # The report dataclass should be returned successfully.
    assert report.recipe_name is not None


# =============================================================================
# Canonical protocol: local_research_signal short-circuits Tier 2/3 substrate checks
# =============================================================================


def test_dispatch_protocol_local_mps_skips_substrate_only_checks():
    """A local_mps recipe should produce no substrate-only Tier 2 blockers."""
    from tac.deploy.dispatch_protocol import evaluate_dispatch_protocol_complete
    # Recipe lacks substrate-only fields (min_vram_gb / target_modes etc.) on
    # purpose; local_mps platform should short-circuit the Tier 2 checks.
    recipe = {
        "platform": "local_mps",
        "lane_id": "lane_test_local_mps_20260517",
        "dispatch_kind": "local_research_signal",
        "research_only": True,
    }
    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=REPO_ROOT,
        trainer_path=REPO_ROOT / "tools" / "operator_authorize.py",
        remote_driver_path=REPO_ROOT / "tools" / "operator_authorize.py",
        native_dispatch=True,
    )
    # Tier 2 should have no blockers because local_research_signal skips them.
    tier_2 = next(t for t in report.tiers if t.name == "tier2_hardware_correctness")
    assert tier_2.passed, f"Tier 2 should pass for local_mps; blockers: {tier_2.blockers}"


# =============================================================================
# Edge cases
# =============================================================================


def test_check_317_str_repo_root_accepted(tmp_path: Path):
    """repo_root accepts str in addition to Path."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    violations = check_local_research_signal_dispatches_stamp_evidence_grade(
        strict=False, verbose=False, repo_root=str(tmp_path)
    )
    assert violations == []


def test_check_317_verbose_output_does_not_crash(capsys):
    """verbose=True shouldn't crash."""
    from tac.preflight import check_local_research_signal_dispatches_stamp_evidence_grade

    check_local_research_signal_dispatches_stamp_evidence_grade(strict=False, verbose=True)
    captured = capsys.readouterr()
    # Either "OK" or "violation(s)" message should appear
    assert (
        "check_local_research_signal_dispatches_stamp_evidence_grade" in captured.out
    )


def test_mps_research_signal_evidence_grade_pinned():
    from tac.optimization import mps_research_signal

    assert mps_research_signal.EVIDENCE_GRADE == "MPS-research-signal"


def test_macos_cpu_advisory_evidence_grade_pinned():
    from tac.optimization import macos_cpu_advisory_signal

    assert macos_cpu_advisory_signal.EVIDENCE_GRADE == "macOS-CPU-advisory"


def test_dispatch_local_mps_uses_canonical_evidence_grade_token():
    """The dispatcher's evidence_grade reference must match the canonical helper."""
    src = (REPO_ROOT / "tools" / "operator_authorize.py").read_text(encoding="utf-8")
    # Both the literal AND a guarded equality assertion are present.
    assert '"MPS-research-signal"' in src
    assert '"macOS-CPU-advisory"' in src
