"""Tests for the --seed-poses-path flag added to experiments/optimize_poses.py.

CLAUDE.md non-negotiable: NEVER invent CLI flags. These tests verify the flag
is declared in argparse, has the right type/default, and that init_poses
loading prefers it over --gt-poses-path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OPT_POSES = REPO / "experiments" / "optimize_poses.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return OPT_POSES.read_text()


@pytest.fixture(scope="module")
def argparse_flags(script_text: str) -> set[str]:
    """All --flag names declared in optimize_poses.py argparse."""
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", script_text))


# ── Flag is declared ─────────────────────────────────────────────────


def test_seed_poses_path_flag_declared(argparse_flags: set[str]) -> None:
    """--seed-poses-path must exist in optimize_poses.py argparse."""
    assert "seed-poses-path" in argparse_flags, (
        "--seed-poses-path missing from optimize_poses.py argparse — "
        "Lane OS-A's seed file cannot be wired in. CLAUDE.md non-negotiable: "
        "NEVER invent CLI flags. The remote bootstrap script invokes this "
        "flag — it must exist here too."
    )


def test_seed_poses_path_has_string_type_default_none(script_text: str) -> None:
    """The flag must be a string with default=None (matches --gt-poses-path)."""
    m = re.search(
        r'add_argument\(\s*"--seed-poses-path"\s*,\s*type=str\s*,\s*default=None',
        script_text,
    )
    assert m is not None, (
        "--seed-poses-path must be declared as `type=str, default=None` "
        "(parity with --gt-poses-path)"
    )


def test_seed_poses_path_help_mentions_lane_os(script_text: str) -> None:
    """The help text must explain Lane OS-A so future agents understand."""
    # Find the seed-poses-path block and look for Lane OS reference.
    m = re.search(
        r'add_argument\(\s*"--seed-poses-path"(.*?)\)',
        script_text, re.DOTALL,
    )
    assert m is not None
    block = m.group(1)
    assert "Lane OS" in block or "openpilot" in block or "supercombo" in block, (
        "--seed-poses-path help must reference Lane OS / openpilot / "
        "supercombo so the flag's purpose is discoverable"
    )


# ── Loading precedence: seed > gt > extracted ────────────────────────


def test_init_poses_prefers_seed_over_gt(script_text: str) -> None:
    """When both --seed-poses-path and --gt-poses-path are set, seed wins.

    The order matters: the seed path is the explicit Lane OS warm-start,
    and the gt path is the legacy / Lane A path. If a future refactor flips
    the order, Lane OS-A silently degrades to Lane A behavior.
    """
    # Find the init_poses loading block.
    m = re.search(
        r"Step 4:\s*Load or extract initial GT poses.*?init_poses\s*=\s*init_poses\[:n_pairs\]",
        script_text, re.DOTALL,
    )
    assert m is not None, (
        "couldn't find init_poses loading block in optimize_poses.py"
    )
    block = m.group(0)

    seed_idx = block.find("seed_poses_path")
    gt_idx = block.find("gt_poses_path")
    assert seed_idx >= 0, "seed_poses_path must be checked in init_poses block"
    assert gt_idx >= 0, "gt_poses_path must remain in init_poses block"
    assert seed_idx < gt_idx, (
        "--seed-poses-path must be checked BEFORE --gt-poses-path. If the "
        "order is reversed, Lane OS-A silently degrades to baseline-pose "
        "warm-start (Lane A behavior)."
    )


def test_init_poses_loads_seed_with_weights_only(script_text: str) -> None:
    """The seed load must use weights_only=True (CLAUDE.md loader-format-safety)."""
    m = re.search(
        r"args\.seed_poses_path.*?torch\.load\([^)]*weights_only=True",
        script_text, re.DOTALL,
    )
    assert m is not None, (
        "seed_poses_path load must use weights_only=True (preflight: "
        "loader_format_safety, Mario R2 CRITICAL)"
    )


# ── No accidental forbidden patterns introduced ──────────────────────


def test_no_mps_default_in_seed_section(script_text: str) -> None:
    """No 'mps' fallback was added to the init_poses block."""
    # Just check the init_poses block — the script's --device flag retains
    # its existing choices.
    m = re.search(
        r"Step 4:\s*Load or extract initial GT poses.*?init_poses\s*=\s*init_poses\[:n_pairs\]",
        script_text, re.DOTALL,
    )
    assert m is not None
    block = m.group(0)
    assert "mps" not in block.lower(), (
        "init_poses block must not reference MPS"
    )
