# SPDX-License-Identifier: MIT
"""Dedicated tests for Catalog #280 (codex bklem3v5j F2 self-protection).

The gate refuses any state of ``tools/canonical_dispatch_optimization_protocol.py``
where the unresolved-recipe branch sets Tier 2/3 signals to True (vacuous
PASS) without paired-env discipline. Codex review bklem3v5j HIGH (2026-05-15)
recommended ``--allow-no-recipe-advisory-mode`` + paired
``--no-recipe-rationale <text>`` per Catalog #199 sister discipline; even in
advisory mode ``--strict`` MUST exit nonzero (advisory != pass).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"


# -----------------------------------------------------------------------------
# Live-repo regression guard
# -----------------------------------------------------------------------------


def test_live_repo_clean_after_f2_fix() -> None:
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        "Live repo expected clean post Catalog #280 F2 fix; "
        f"got {len(v)} violation(s):\n" + "\n".join(v[:5])
    )


def test_live_repo_strict_clean() -> None:
    check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=REPO_ROOT, strict=True
    )


# -----------------------------------------------------------------------------
# Positive tests (gate FLAGS the F2 regression)
# -----------------------------------------------------------------------------


def _write_target(tmp_root: Path, body: str) -> None:
    target = tmp_root / "tools" / "canonical_dispatch_optimization_protocol.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_flags_vacuous_true_regression(tmp_path: Path) -> None:
    """If the F2 vacuous-True pattern returns, the gate flags it."""
    body = textwrap.dedent(
        '''
        def _verify_tier2(trainer_text, trainer_path, recipe_path, lane_driver_path):
            verdict = TierVerdict()
            if recipe_path is not None:
                pass
            else:
                verdict.pass_signals["recipe_declares_min_vram_gb"] = True  # vacuous
                verdict.pass_signals["recipe_declares_min_smoke_gpu"] = True
                verdict.pass_signals["recipe_declares_video_input_strategy"] = True
                verdict.pass_signals["recipe_declares_pyav_decode_strategy"] = True
                verdict.pass_signals["recipe_declares_target_modes"] = True
            return verdict
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    assert any("vacuous-True pattern" in s for s in v), v


def test_flags_missing_paired_env_kwargs(tmp_path: Path) -> None:
    """If the paired-env kwargs disappear from the verifier, gate flags."""
    body = textwrap.dedent(
        '''
        def verify_dispatch_protocol_complete(trainer, recipe=None, *, repo_root=None):
            return None
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    # Multiple required tokens missing.
    assert len(v) >= 3, v


def test_flags_missing_blocker_messaging(tmp_path: Path) -> None:
    """If the unresolved-recipe blocker token disappears, gate flags."""
    body = textwrap.dedent(
        '''
        def _verify_tier2(*args, allow_no_recipe_advisory_mode=False, no_recipe_rationale=None):
            pass

        def _verify_tier3(*args, allow_no_recipe_advisory_mode=False, no_recipe_rationale=None):
            pass

        # CLI:
        # --allow-no-recipe-advisory-mode
        # --no-recipe-rationale
        # Per Catalog #280 F2 fix
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    # Blocker tokens (tier2/tier3 unresolved) absent
    assert any("tier2_hardware: recipe unresolved" in s or "tier3_substrate: recipe unresolved" in s for s in v), v


def test_strict_mode_raises_with_catalog_280(tmp_path: Path) -> None:
    body = '# empty fixture\n'
    _write_target(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "Catalog #280" in msg
    assert "advisory-mode opt-in" in msg


# -----------------------------------------------------------------------------
# Negative tests (gate ACCEPTS the canonical fix)
# -----------------------------------------------------------------------------


def test_accepts_canonical_fix(tmp_path: Path) -> None:
    """The canonical fix landed in the same commit batch passes the gate."""
    body = textwrap.dedent(
        '''
        def _verify_tier2(
            trainer_text, trainer_path, recipe_path, lane_driver_path,
            *,
            allow_no_recipe_advisory_mode: bool = False,
            no_recipe_rationale: str | None = None,
        ):
            verdict = TierVerdict()
            if recipe_path is not None:
                pass
            else:
                verdict.blockers.append(
                    "tier2_hardware: recipe unresolved — cannot verify "
                    "min_vram_gb declaration."
                )
            return verdict

        def _verify_tier3(
            trainer_text, trainer_path, recipe_path,
            *,
            allow_no_recipe_advisory_mode: bool = False,
            no_recipe_rationale: str | None = None,
        ):
            verdict = TierVerdict()
            if recipe_path is not None:
                pass
            else:
                verdict.blockers.append(
                    "tier3_substrate: recipe unresolved — cannot verify "
                    "recipe-vs-trainer-state consistency."
                )
            return verdict

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--allow-no-recipe-advisory-mode", action="store_true")
            parser.add_argument("--no-recipe-rationale", type=str)
            # Per Catalog #280 F2 fix paired-env discipline.
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


# -----------------------------------------------------------------------------
# Waiver tests
# -----------------------------------------------------------------------------


def test_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        # DISPATCH_PROTOCOL_UNRESOLVED_RECIPE_VACUOUS_OK: reviewed-test-fixture
        verdict.pass_signals["recipe_declares_min_vram_gb"] = True  # vacuous
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


def test_waiver_placeholder_rejected(tmp_path: Path) -> None:
    body = textwrap.dedent(
        '''
        # DISPATCH_PROTOCOL_UNRESOLVED_RECIPE_VACUOUS_OK: <rationale>
        verdict.pass_signals["recipe_declares_min_vram_gb"] = True  # vacuous
        '''
    )
    _write_target(tmp_path, body)
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    assert any("vacuous-True pattern" in s for s in v), v


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_missing_target_file(tmp_path: Path) -> None:
    v = check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_orchestrator_strict_wireup() -> None:
    text = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    idx = text.find(
        "check_dispatch_protocol_unresolved_recipe_blocks_unless_paired_waiver("
    )
    assert idx != -1, "orchestrator callsite missing"
    block = text[idx : idx + 200]
    assert "strict=True" in block, f"strict=True expected; got: {block!r}"


# -----------------------------------------------------------------------------
# Regression tests per codex's explicit recommendations
# -----------------------------------------------------------------------------


def test_strict_cli_exits_nonzero_when_recipe_missing() -> None:
    """End-to-end: --strict on the canonical CLI exits nonzero with no recipe.

    Per codex review bklem3v5j HIGH F2 explicit recommendation:
    *"Keep --strict nonzero whenever recipe-side checks are skipped."*
    """
    trainer = REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py"
    if not trainer.is_file():
        pytest.skip("substrate trainer fixture missing in this checkout")
    result = subprocess.run(
        [
            sys.executable,
            str(TARGET),
            "--trainer",
            str(trainer),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0, (
        f"--strict with no recipe expected nonzero (F2 regression); "
        f"got rc={result.returncode}\nstdout:\n{result.stdout}"
    )


def test_bare_advisory_optin_without_rationale_rejected() -> None:
    """End-to-end: bare --allow-no-recipe-advisory-mode raises rc=11.

    Per Catalog #199 paired-env discipline + Catalog #280 F2 fix.
    """
    trainer = REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py"
    if not trainer.is_file():
        pytest.skip("substrate trainer fixture missing in this checkout")
    result = subprocess.run(
        [
            sys.executable,
            str(TARGET),
            "--trainer",
            str(trainer),
            "--strict",
            "--allow-no-recipe-advisory-mode",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 11, (
        f"bare opt-in should rc=11; got rc={result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Catalog #280" in result.stderr or "Catalog #199" in result.stderr


def test_paired_advisory_optin_strict_still_nonzero() -> None:
    """End-to-end: paired opt-in + --strict STILL exits nonzero.

    Per codex's explicit recommendation:
    *"Even with both flags supplied, --strict still exits nonzero
    (advisory != pass)."*
    """
    trainer = REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py"
    if not trainer.is_file():
        pytest.skip("substrate trainer fixture missing in this checkout")
    result = subprocess.run(
        [
            sys.executable,
            str(TARGET),
            "--trainer",
            str(trainer),
            "--strict",
            "--allow-no-recipe-advisory-mode",
            "--no-recipe-rationale",
            "operator-reviewed-test-probe",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0, (
        f"paired opt-in + --strict should still exit nonzero "
        f"(advisory != pass); got rc={result.returncode}\n"
        f"stdout:\n{result.stdout}"
    )
