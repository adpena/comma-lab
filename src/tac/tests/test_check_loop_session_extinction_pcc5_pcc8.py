# SPDX-License-Identifier: MIT
"""Tests for the 4 PCC5-PCC8 loop-session permanent extinction checks.

Reference: feedback_loop_session_permanent_bug_class_extinction_20260501.md.

Each test:
  1. Verifies the check function importable + runs without crash on real
     codebase at strict=False.
  2. Verifies the live-codebase violation count for PCC6/7/8 is 0
     (we fixed every violation in the same commit).
  3. Synthetic-violation tests: create a tmp file matching the bug pattern,
     scan, assert violation reported with expected text.
  4. STRICT=True propagates a PreflightError when violations exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg,
    check_venv_creators_use_ensurepip,
    check_vastai_create_uses_min_disk_60,
    check_remote_chain_drivers_clean_inflated_per_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ─── Live-codebase smoke (no synthesis) ──────────────────────────────────────


def test_pcc5_runs_clean_on_codebase():
    """PCC5 must run without crashing. Live count is allowed > 0 (warn-only)."""
    v = check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg(
        strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_pcc6_runs_clean_and_zero_violations():
    """PCC6: every venv-creator must install pip (or annotate)."""
    v = check_venv_creators_use_ensurepip(strict=False, verbose=False)
    assert v == [], (
        "PCC6 must be at 0 violations after the loop-session extinction fix:\n"
        + "\n".join(f"  - {x}" for x in v)
    )


def test_pcc7_runs_clean_and_zero_violations():
    """PCC7: vastai create instance must use --disk >= 60GB."""
    v = check_vastai_create_uses_min_disk_60(strict=False, verbose=False)
    assert v == [], (
        "PCC7 must be at 0 violations after the loop-session extinction fix:\n"
        + "\n".join(f"  - {x}" for x in v)
    )


def test_pcc8_runs_clean_and_zero_violations():
    """PCC8: multi-candidate chain drivers must clean per-candidate."""
    v = check_remote_chain_drivers_clean_inflated_per_candidate(
        strict=False, verbose=False
    )
    assert v == [], (
        "PCC8 must be at 0 violations after the loop-session extinction fix:\n"
        + "\n".join(f"  - {x}" for x in v)
    )


# ─── Synthetic-violation regression tests (PCC5) ─────────────────────────────


def test_pcc5_flags_script_without_uv_bootstrap(tmp_path: Path):
    """A script that calls contest_auth_eval but has no uv bootstrap fires."""
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    bad = fake_root / "scripts" / "remote_lane_zz_test.sh"
    bad.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "python experiments/contest_auth_eval.py --archive a.zip\n"
    )
    v = check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert any("remote_lane_zz_test.sh" in x for x in v)


def test_pcc5_passes_with_bootstrap_runtime_deps(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "remote_lane_zz_good.sh"
    good.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "bootstrap_runtime_deps() { :; }\nbootstrap_runtime_deps\n"
        "python experiments/contest_auth_eval.py --archive a.zip\n"
    )
    v = check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc5_strict_raises_preflight_error(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    bad = fake_root / "scripts" / "remote_lane_zz_test.sh"
    bad.write_text(
        "#!/bin/bash\nset -e\npython experiments/contest_auth_eval.py\n"
    )
    with pytest.raises(PreflightError):
        check_remote_archive_eval_self_bootstraps_uv_and_ffmpeg(
            repo_root=fake_root, strict=True, verbose=False
        )


# ─── Synthetic-violation regression tests (PCC6) ─────────────────────────────


def test_pcc6_flags_uv_venv_without_ensurepip(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    bad = fake_root / "scripts" / "fake_bootstrap.sh"
    bad.write_text(
        "#!/bin/bash\nset -e\nuv venv .venv\n"
        "echo done\n"  # no ensurepip / ensure_remote_pip
    )
    v = check_venv_creators_use_ensurepip(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert any("fake_bootstrap.sh" in x for x in v)


def test_pcc6_passes_with_ensure_remote_pip(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "fake_ok.sh"
    good.write_text(
        "#!/bin/bash\nset -e\nuv venv .venv\n"
        "bash scripts/ensure_remote_pip.sh .venv/bin/python\n"
    )
    v = check_venv_creators_use_ensurepip(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc6_passes_with_inline_no_pip_needed_marker(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "fake_ok2.sh"
    good.write_text(
        "#!/bin/bash\nset -e\n"
        "uv venv .venv  # NO_PIP_NEEDED: uv-only flow downstream\n"
    )
    v = check_venv_creators_use_ensurepip(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc6_python_dash_m_venv_not_flagged(tmp_path: Path):
    """`python -m venv` ships pip in stdlib bundle; should NOT be flagged."""
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    file = fake_root / "scripts" / "stdlib_venv.sh"
    file.write_text(
        "#!/bin/bash\nset -e\npython3 -m venv .venv\n"
        "source .venv/bin/activate\n"
    )
    v = check_venv_creators_use_ensurepip(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc6_strict_raises(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    (fake_root / "scripts" / "bad.sh").write_text(
        "#!/bin/bash\nuv venv .venv\necho hi\n"
    )
    with pytest.raises(PreflightError):
        check_venv_creators_use_ensurepip(
            repo_root=fake_root, strict=True, verbose=False
        )


# ─── Synthetic-violation regression tests (PCC7) ─────────────────────────────


def test_pcc7_flags_disk_30_in_shell(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    bad = fake_root / "scripts" / "fake_create.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "vastai create instance 12345 --disk 30 --label foo\n"
    )
    v = check_vastai_create_uses_min_disk_60(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert any("fake_create.sh" in x and "30" in x for x in v)


def test_pcc7_passes_with_disk_60_in_shell(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "fake_ok.sh"
    good.write_text(
        "#!/bin/bash\n"
        "vastai create instance 12345 --disk 60 --label foo\n"
    )
    v = check_vastai_create_uses_min_disk_60(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc7_passes_with_single_candidate_waiver(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "fake_ok_waiver.sh"
    good.write_text(
        "#!/bin/bash\n"
        "vastai create instance 12345 --disk 30 --label foo  # SINGLE_CANDIDATE_DISK_OK: smoke only\n"
    )
    v = check_vastai_create_uses_min_disk_60(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc7_python_disk_variable_passes(tmp_path: Path):
    """Python source with `--disk` referencing a `disk_gb`-named variable
    is allowed because the dataclass default is policed at the InstanceSpec
    level."""
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "create.py"
    good.write_text(
        'def go(spec):\n'
        '    cmd = ["vastai", "create", "instance", "1",\n'
        '           "--disk", str(int(spec.disk_gb)),\n'
        '           "--label", "x"]\n'
    )
    v = check_vastai_create_uses_min_disk_60(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc7_strict_raises(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    (fake_root / "scripts" / "bad.sh").write_text(
        "#!/bin/bash\nvastai create instance 1 --disk 16\n"
    )
    with pytest.raises(PreflightError):
        check_vastai_create_uses_min_disk_60(
            repo_root=fake_root, strict=True, verbose=False
        )


# ─── Synthetic-violation regression tests (PCC8) ─────────────────────────────


def test_pcc8_flags_chain_without_cleanup(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    bad = fake_root / "scripts" / "wave_chain_driver.sh"
    bad.write_text(
        "#!/bin/bash\nset -e\n"
        "for entry in ${CANDIDATES[@]}; do\n"
        "  bash scripts/remote_archive_only_eval.sh\n"
        '  echo "no cleanup here"\n'
        "done\n"
    )
    v = check_remote_chain_drivers_clean_inflated_per_candidate(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert any("wave_chain_driver.sh" in x for x in v)


def test_pcc8_passes_with_rm_inflated(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "wave_chain_ok.sh"
    good.write_text(
        "#!/bin/bash\nset -e\n"
        "for entry in ${CANDIDATES[@]}; do\n"
        "  bash scripts/remote_archive_only_eval.sh\n"
        "  rm -rf $LOG_DIR/eval_work/inflated\n"
        "done\n"
    )
    v = check_remote_chain_drivers_clean_inflated_per_candidate(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc8_passes_with_waiver(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    good = fake_root / "scripts" / "wave_chain_waiver.sh"
    good.write_text(
        "#!/bin/bash\nset -e\n"
        "# NO_INFLATE_CLEANUP_NEEDED: this is a 1-candidate smoke chain\n"
        "for entry in ${CANDIDATES[@]}; do\n"
        "  bash scripts/remote_archive_only_eval.sh\n"
        "done\n"
    )
    v = check_remote_chain_drivers_clean_inflated_per_candidate(
        repo_root=fake_root, strict=False, verbose=False
    )
    assert v == []


def test_pcc8_strict_raises(tmp_path: Path):
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    (fake_root / "scripts" / "chain_bad.sh").write_text(
        "#!/bin/bash\nfor c in ${A[@]}; do\n"
        "  bash scripts/remote_archive_only_eval.sh\ndone\n"
    )
    with pytest.raises(PreflightError):
        check_remote_chain_drivers_clean_inflated_per_candidate(
            repo_root=fake_root, strict=True, verbose=False
        )


# ─── ensure_remote_pip.sh contract tests ─────────────────────────────────────


def test_ensure_remote_pip_script_exists_and_executable():
    p = REPO_ROOT / "scripts" / "ensure_remote_pip.sh"
    assert p.exists(), "scripts/ensure_remote_pip.sh missing"
    import os
    assert os.access(p, os.X_OK), "scripts/ensure_remote_pip.sh not executable"


def test_ensure_remote_pip_script_contract():
    """Verify the script honors the canonical contract:
    - prints resolved python path on stdout
    - logs to stderr
    - has set -euo pipefail
    - has a pip-already-importable fast path
    """
    text = (REPO_ROOT / "scripts" / "ensure_remote_pip.sh").read_text()
    assert "set -euo pipefail" in text
    assert 'printf \'%s\\n\' "$PYBIN"' in text
    assert "import pip" in text
    assert ">&2" in text, "logs must go to stderr"
    assert "ensurepip" in text
    assert "feedback_loop_session_permanent_bug_class_extinction_20260501" in text


def test_probe_nvdec_self_heals_missing_pip():
    """probe_nvdec.sh must auto-invoke ensure_remote_pip.sh BEFORE the
    pip-install path."""
    text = (REPO_ROOT / "scripts" / "probe_nvdec.sh").read_text()
    assert "ensure_remote_pip.sh" in text
    # Make sure self-heal is BEFORE the pip-install call (line ordering).
    self_heal_idx = text.index("ensure_remote_pip.sh")
    pip_install_idx = text.index('"$PYBIN" -m pip install')
    assert self_heal_idx < pip_install_idx, (
        "ensure_remote_pip.sh self-heal must precede the pip install path"
    )


# ─── launch_lane_on_vastai.py disk-floor tests ───────────────────────────────


def test_launcher_default_disk_is_60():
    """Both create_instance() and find_offer() must default to disk >= 60GB."""
    text = (REPO_ROOT / "scripts" / "launch_lane_on_vastai.py").read_text()
    assert "min_disk_gb: int = 60" in text
    assert "disk_gb: int = 60" in text
    assert "default=60" in text  # argparse defaults
    # Floor warning still in place
    assert "if disk_gb < 60:" in text


def test_launcher_exposes_fast_chip_preference():
    """Phase1/full must support the sprint fast-chip preference."""
    text = (REPO_ROOT / "scripts" / "launch_lane_on_vastai.py").read_text()
    assert "prefer_fast_chip: bool = False" in text
    assert "from probe_fastest_chip import probe" in text
    assert "--prefer-fast-chip" in text
    assert "H100/H200/A100" in text


def test_launcher_includes_explicit_anchor_dirs_in_tarball():
    """Explicit --anchor-dirs must change shipped bytes, not only metadata."""
    text = (REPO_ROOT / "scripts" / "launch_lane_on_vastai.py").read_text()
    assert "def build_tarball(anchor_dirs: list[str] | None = None)" in text
    assert "paths += _expand_explicit_anchor_paths(anchor_dirs or [])" in text
    assert "tar = build_tarball(args.anchor_dirs)" in text
    assert "explicit anchor path does not exist" in text
    assert '"tar", "-rzf"' not in text


def test_instance_spec_default_disk_is_60():
    """src/tac/deploy/base.py InstanceSpec.disk_gb default must be 60."""
    text = (REPO_ROOT / "src" / "tac" / "deploy" / "base.py").read_text()
    assert "disk_gb: float = 60.0" in text
