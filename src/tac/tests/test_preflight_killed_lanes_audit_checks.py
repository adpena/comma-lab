"""Tests for Checks 48, 49, 50 — the killed-lanes forensic audit preflight checks.

Reference: project_killed_lanes_forensic_audit_20260428.

Three new meta-bug checks that prevent regression of the bug classes that
killed Lane V (orphan modules), Lane J-JBL (loss_mode validator allowlist
mismatch), and the silent --profile typo class (deploy script references
unregistered profile).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_deploy_script_profiles_exist_in_registry,
    check_no_orphan_src_tac_modules,
    check_profile_loss_modes_in_validator_allowlist,
)

# ── Check 48: orphan src/tac modules ─────────────────────────────────────────


def test_check_48_no_violations_in_warn_mode(tmp_path: Path) -> None:
    """Check 48 must execute without crashing on a synthetic clean repo."""
    # Build a tiny repo: src/tac/foo.py + src/tac/profiles.py importing it.
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "foo.py").write_text("# trivial module\n")
    (tmp_path / "src" / "tac" / "profiles.py").write_text(
        "from tac.foo import something\n"
    )
    (tmp_path / "src" / "tac" / "experiments").mkdir()
    (tmp_path / "src" / "tac" / "experiments" / "train_renderer.py").write_text(
        "# stub\n"
    )
    violations = check_no_orphan_src_tac_modules(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], (
        f"foo.py is referenced by profiles.py — should NOT be flagged: {violations}"
    )


def test_check_48_flags_orphan_module(tmp_path: Path) -> None:
    """Check 48 must flag a src/tac module with no reference anywhere."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "orphan_thing.py").write_text("# not referenced\n")
    (tmp_path / "src" / "tac" / "profiles.py").write_text("# nothing\n")
    (tmp_path / "src" / "tac" / "experiments").mkdir()
    (tmp_path / "src" / "tac" / "experiments" / "train_renderer.py").write_text(
        "# nothing\n"
    )
    violations = check_no_orphan_src_tac_modules(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert any("orphan_thing.py" in v for v in violations), (
        f"orphan_thing.py has no reference — must be flagged: {violations}"
    )


def test_check_48_strict_raises(tmp_path: Path) -> None:
    """Check 48 in strict mode raises MetaBugViolation when orphans exist."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "ghost.py").write_text("# nothing\n")
    (tmp_path / "src" / "tac" / "profiles.py").write_text("# empty\n")
    (tmp_path / "src" / "tac" / "experiments").mkdir()
    (tmp_path / "src" / "tac" / "experiments" / "train_renderer.py").write_text("\n")
    with pytest.raises(MetaBugViolation, match="ORPHAN SRC/TAC MODULES"):
        check_no_orphan_src_tac_modules(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_check_48_ignores_oss_export_mirror_references(tmp_path: Path) -> None:
    """OSS-publication mirrors must not keep live src/tac modules alive."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "mirror_only.py").write_text("# candidate\n")
    (tmp_path / "src" / "tac" / "profiles.py").write_text("# nothing\n")
    (tmp_path / "src" / "tac" / "experiments").mkdir()
    (tmp_path / "src" / "tac" / "experiments" / "train_renderer.py").write_text(
        "# nothing\n"
    )
    mirror = tmp_path / "experiments" / "comma_lab_public_export"
    mirror.mkdir(parents=True)
    (mirror / "consumer.py").write_text("import tac.mirror_only\n")

    violations = check_no_orphan_src_tac_modules(
        repo_root=tmp_path, strict=False, verbose=False,
    )

    assert any("mirror_only.py" in v for v in violations), (
        "comma_lab_public_export is a regenerated mirror and must not satisfy "
        f"live-code orphan references: {violations}"
    )


def test_check_48_counts_non_mirror_experiment_references(tmp_path: Path) -> None:
    """Non-mirror experiment tools remain valid wiring evidence."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "wired_by_experiment.py").write_text("# candidate\n")
    (tmp_path / "src" / "tac" / "profiles.py").write_text("# nothing\n")
    (tmp_path / "src" / "tac" / "experiments").mkdir()
    (tmp_path / "src" / "tac" / "experiments" / "train_renderer.py").write_text(
        "# nothing\n"
    )
    experiments = tmp_path / "experiments" / "live_tool"
    experiments.mkdir(parents=True)
    (experiments / "consumer.py").write_text("import tac.wired_by_experiment\n")

    violations = check_no_orphan_src_tac_modules(
        repo_root=tmp_path, strict=False, verbose=False,
    )

    assert not any("wired_by_experiment.py" in v for v in violations), violations


# ── Check 49: profile loss_mode allowlist parity ─────────────────────────────


def _write_train_renderer_with_allowlist(path: Path, allowed: list[str]) -> None:
    """Write a stub train_renderer.py with a _VALID_LOSS_MODES tuple."""
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = ", ".join(f'"{a}"' for a in allowed)
    path.write_text(f"# stub\n_VALID_LOSS_MODES = ({quoted})\n")


def test_check_49_passes_when_loss_mode_in_allowlist(tmp_path: Path) -> None:
    """Check 49 passes when every profile loss_mode is in the allowlist."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write_train_renderer_with_allowlist(
        tmp_path / "src" / "tac" / "experiments" / "train_renderer.py",
        ["standard", "kl", "jbl"],
    )
    (tmp_path / "src" / "tac" / "profiles.py").write_text(
        '''GOOD = {"loss_mode": "jbl"}\nALSO_GOOD = {"loss_mode": "kl"}\n'''
    )
    violations = check_profile_loss_modes_in_validator_allowlist(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"Both profile values are allowed: {violations}"


def test_check_49_flags_unknown_loss_mode(tmp_path: Path) -> None:
    """Check 49 must flag a profile loss_mode value not in the allowlist.

    This is the EXACT Lane J-JBL bug class — a profile sets loss_mode="jbl"
    but train_renderer.py's allowlist doesn't include it. Validator raises
    SystemExit at boot; the lane exits unexpectedly.
    """
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write_train_renderer_with_allowlist(
        tmp_path / "src" / "tac" / "experiments" / "train_renderer.py",
        ["standard", "kl"],  # NO "jbl"
    )
    (tmp_path / "src" / "tac" / "profiles.py").write_text(
        'BAD_PROFILE = {"loss_mode": "jbl"}\n'
    )
    violations = check_profile_loss_modes_in_validator_allowlist(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1, (
        f"Exactly 1 violation expected (loss_mode='jbl' missing): {violations}"
    )
    assert "'jbl'" in violations[0]
    # Strict mode raises.
    with pytest.raises(MetaBugViolation, match="LOSS_MODE NOT IN VALIDATOR"):
        check_profile_loss_modes_in_validator_allowlist(
            repo_root=tmp_path, strict=True, verbose=False,
        )


# ── Check 50: deploy script --profile X must resolve in PROFILES ─────────────


def _write_profiles_registry(path: Path, names: list[str]) -> None:
    """Write a stub src/tac/profiles.py with a PROFILES dict of given keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = "\n".join(f'    "{n}": None,' for n in names)
    path.write_text(f"PROFILES = {{\n{items}\n}}\n")


def test_check_50_passes_when_profile_registered(tmp_path: Path) -> None:
    """Check 50 passes when --profile X resolves in PROFILES."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write_profiles_registry(
        tmp_path / "src" / "tac" / "profiles.py",
        ["h_v3_joint_halfframe", "lane_a_optimized"],
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "remote_lane_test.sh").write_text(
        "python train_renderer.py --profile h_v3_joint_halfframe\n"
    )
    violations = check_deploy_script_profiles_exist_in_registry(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"h_v3_joint_halfframe IS registered: {violations}"


def test_check_50_flags_typo_profile(tmp_path: Path) -> None:
    """Check 50 must flag a deploy script that --profile's a typo'd name."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write_profiles_registry(
        tmp_path / "src" / "tac" / "profiles.py",
        ["h_v3_joint_halfframe"],
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "remote_lane_typo.sh").write_text(
        "python train.py --profile h_v3_joint_halframe\n"  # missing 'f'
    )
    violations = check_deploy_script_profiles_exist_in_registry(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) >= 1
    assert any("h_v3_joint_halframe" in v for v in violations)
    with pytest.raises(MetaBugViolation, match="REFERENCES MISSING PROFILE"):
        check_deploy_script_profiles_exist_in_registry(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_check_50_skips_bash_interpolation(tmp_path: Path) -> None:
    """Check 50 must not flag --profile $VAR (operator-supplied at runtime)."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    _write_profiles_registry(
        tmp_path / "src" / "tac" / "profiles.py",
        ["lane_x"],
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "remote_lane_dynamic.sh").write_text(
        'python train.py --profile $PROFILE_NAME\n'
    )
    violations = check_deploy_script_profiles_exist_in_registry(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], (
        f"$PROFILE_NAME is bash interpolation, not a literal: {violations}"
    )


# ── Real-codebase smoke: warn-only at this commit ────────────────────────────


def test_check_49_runs_on_live_codebase() -> None:
    """Check 49 must execute on the live repo without crashing.

    Currently warn-only (live count: 2 — posenet_embedding + segnet_kl).
    These are profile-declared loss_mode values that don't appear in
    train_renderer.py's _VALID_LOSS_MODES allowlist. They MAY be live
    bugs (Lane J-JBL class) or profiles that intentionally bypass
    train_renderer.py. STRICT promotion deferred until audit complete.
    """
    violations = check_profile_loss_modes_in_validator_allowlist(
        strict=False, verbose=False,
    )
    # Bound the violation count so a regression that adds 50 unknown
    # loss_modes is caught by this test even before STRICT promotion.
    assert len(violations) <= 5, (
        f"Live codebase has {len(violations)} loss_mode allowlist "
        f"violation(s); expected ≤ 5 (regression guard): {violations[:5]}"
    )


def test_check_50_runs_on_live_codebase() -> None:
    """Check 50 must execute on the live repo without crashing.

    Currently warn-only (live count: 4 — includes a comment false-positive
    in remote_lane_ebr_*.sh and two profiles needing registration).
    STRICT promotion deferred until lane script cleanup pass.
    """
    violations = check_deploy_script_profiles_exist_in_registry(
        strict=False, verbose=False,
    )
    # Bound the violation count to catch regressions while warn-only.
    assert len(violations) <= 10, (
        f"Live codebase has {len(violations)} deploy-script profile "
        f"violation(s); expected ≤ 10 (regression guard): {violations[:5]}"
    )


def test_check_48_runs_on_live_codebase() -> None:
    """Check 48 must execute on the live repo without crashing.

    Warn-only initially. The live violation count IS expected to be > 0
    until a cleanup pass; this test pins the current cleanup frontier so the
    backlog cannot silently regress.
    """
    violations = check_no_orphan_src_tac_modules(
        strict=False, verbose=False,
    )
    # 2026-05-07 cleanup frontier: 10 remaining unresolved modules after
    # commits 8969f6d2 and 2c30e0b8. Keep this bound tight until Check 48 can
    # promote to strict at zero.
    assert len(violations) <= 10, (
        f"Live codebase has {len(violations)} orphan module(s); "
        f"expected ≤ 10 (cleanup frontier guard): {violations[:5]}"
    )
