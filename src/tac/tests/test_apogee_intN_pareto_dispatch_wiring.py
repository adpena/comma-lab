"""Regression: verify tools/apogee_intN_pareto.py emits ONLY real launch_lane_on_vastai.py flags.

Per CLAUDE.md NEVER-INVENT-CLI-FLAGS non-negotiable: any new wrapper that
prints subprocess one-liners must have a regression test that introspects the
target's argparse and asserts the wrapper's flag set is a subset.

This test:
  A. Runs `tools/apogee_intN_pareto.py` on the live empirical manifests
  B. Extracts every long-form flag (--xxx) from the dispatch one-liners
  C. Asserts every flag exists in `launch_lane_on_vastai.py`'s `full` subparser
  D. Asserts the `full` subcommand keyword appears (not just `phase1`)

If a future refactor renames or removes a flag from launch_lane_on_vastai.py,
this test breaks at CI time, before any operator copy-pastes a dead command.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _full_subparser_flags() -> set[str]:
    """Extract every flag added to the `full` subparser via AST."""
    src = (REPO / "scripts" / "launch_lane_on_vastai.py").read_text()
    tree = ast.parse(src)
    flags: set[str] = set()
    # Find every `pf.add_argument(...)` call (the `full` subparser is bound to `pf`)
    # and every shared loop call `p_.add_argument(...)` that gets applied to all subparsers.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        # Check the receiver is `pf` or `p_` (loop var that includes pf)
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in {"pf", "p_", "p"}:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("--"):
            flags.add(first.value)
    return flags


def _pareto_emitted_flags() -> tuple[set[str], str]:
    """Run the Pareto tool and extract flags from its emitted one-liners."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "apogee_intN_pareto.py")],
        capture_output=True, text=True, check=True,
    )
    out = proc.stdout
    # Find every `--<word>` token in the dispatch one-liner block
    flags = set(re.findall(r"--[a-z][a-z0-9-]*", out))
    return flags, out


def test_pareto_tool_runs_clean():
    """Smoke: the Pareto tool runs on the live manifests."""
    flags, out = _pareto_emitted_flags()
    assert "DISPATCH ONE-LINERS" in out, "Pareto tool did not emit dispatch one-liners section"
    assert flags, "Pareto tool emitted zero --flags (regression in dispatch generator)"


def test_pareto_emits_full_subcommand_keyword():
    """The dispatch one-liner must invoke the `full` subcommand."""
    _, out = _pareto_emitted_flags()
    # The subcommand appears as a positional arg in the launch line, e.g. `... full \`
    assert re.search(r"launch_lane_on_vastai\.py\s+full\b", out), \
        "Pareto one-liners do not invoke the `full` subcommand of launch_lane_on_vastai.py"


def test_pareto_flags_subset_of_full_subparser():
    """Every flag emitted by Pareto must exist in the `full` subparser of launch_lane_on_vastai.py."""
    emitted, _out = _pareto_emitted_flags()
    valid = _full_subparser_flags()
    invented = emitted - valid
    assert not invented, (
        f"Pareto tool emits flags that DO NOT EXIST in launch_lane_on_vastai.py `full` subparser: "
        f"{sorted(invented)}\n"
        f"Valid flags in launch_lane_on_vastai.py: {sorted(valid)}\n"
        f"This is the dead-flag-wiring bug class — see CLAUDE.md NEVER-INVENT-CLI-FLAGS."
    )


def test_apogee_intN_env_var_in_one_liners():
    """Each dispatch one-liner must export APOGEE_INTN_BITS=N to pick the bit-width."""
    _, out = _pareto_emitted_flags()
    matches = re.findall(r"APOGEE_INTN_BITS=(\d)", out)
    assert matches, "No APOGEE_INTN_BITS=N env var in any dispatch one-liner"
    bits_emitted = sorted({int(m) for m in matches})
    # int7 should be Pareto-dominated, so it should NOT be in the dispatch list
    assert 7 not in bits_emitted, (
        "int7 is Pareto-dominated by int8 (more bytes, same distortion class). "
        "Pareto tool emitted a dispatch for int7 — this is a regression in the domination check."
    )
    # All emitted bits must be in the legal range 4..8
    for b in bits_emitted:
        assert 4 <= b <= 8, f"APOGEE_INTN_BITS={b} outside legal range 4..8"


def test_lane_script_path_exists():
    """The --lane-script path emitted by Pareto must point to a real file on disk."""
    _, out = _pareto_emitted_flags()
    matches = re.findall(r"--lane-script\s+(\S+)", out)
    assert matches, "No --lane-script flag emitted"
    for path in matches:
        full = REPO / path
        assert full.exists(), (
            f"--lane-script {path} does not exist (would fail at dispatch time). "
            f"Pareto tool is referencing a stale or wrong wrapper script path."
        )
