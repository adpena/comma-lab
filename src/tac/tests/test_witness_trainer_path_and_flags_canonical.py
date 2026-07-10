# SPDX-License-Identifier: MIT
"""Drift guard for the witness trainer-path constant (#2) and the argparse
flag-scanner regex (#3) — both MUST flow from ``tac.witness_dsl.curriculum_dsl``.

De-dup audit `.omx/research/hardcode_duplication_audit_witness_stack_20260710.md`
findings #2 + #3: the levelset trainer path and the never-invent-flags regex were
copy-pasted across production files. They are now single-sourced from
``curriculum_dsl.{TRAINER_REL, TRAINER_PATH, real_trainer_flags}``.

Three protections (the self-protecting half of the two-landing fix):

1. **Value drift** — every production copy of the trainer-path constant must
   equal the canonical constant. Retype the literal anywhere and this fails.
2. **Flag-behavior drift** — ``launch_witness_run.real_trainer_flags()`` must
   still equal ``base ∪ {--no-<f> for BooleanOptionalAction f}`` (the 2026-07-07
   negation CLASS-fix), sourced from the canonical helpers — a naive whole-function
   swap that dropped the negations would fail here.
3. **Regex re-introduction ratchet** — the duplicated ``add_argument`` scanner
   regex must not reappear as a raw literal in the two migrated tool files.
"""
from __future__ import annotations

import sys
from pathlib import Path

from tac.witness_dsl.curriculum_dsl import (
    TRAINER_PATH,
    TRAINER_REL,
    real_boolean_flags,
    real_store_true_flags,
    real_trainer_flags,
)

_REPO = Path(__file__).resolve().parents[3]

# The duplicated base-scan regex literal that finding #3 canonicalized. It must
# NOT reappear as a raw string in the two migrated tool files (they import the
# canonical helper instead).
_DUP_REGEX_LITERAL = r'add_argument\(\s*"(--[a-z0-9-]+)"'


def _import_tool(name: str):
    if str(_REPO / "tools") not in sys.path:
        sys.path.insert(0, str(_REPO / "tools"))
    return __import__(name)


# ─────────────────────────── #2 trainer-path value drift ───────────────────────────
def test_gauge_levelset_trainer_rel_is_canonical():
    from tac.witness_dsl import gauge

    assert gauge.LEVELSET_TRAINER_REL == TRAINER_REL


def test_confound_gates_first_trainer_file_is_canonical():
    from tac import confound_gates

    assert confound_gates._TRAINER_FILES[0] == TRAINER_REL


def test_v2_compose_default_trainer_is_canonical():
    from tac.v2_compose import launch_command

    assert launch_command.DEFAULT_TRAINER == TRAINER_REL


def test_tool_trainer_path_constants_are_canonical():
    lwr = _import_tool("launch_witness_run")
    mgd = _import_tool("mlx_gpu_determinism_probe")
    assert lwr._TRAINER == TRAINER_PATH
    assert mgd._TRAINER == str(TRAINER_PATH)  # probe uses os.path/str argv style


# ─────────────────────────── #3 flag-scanner behavior drift ───────────────────────────
def test_launch_witness_run_flags_preserve_base_plus_negations():
    """The launcher validator = base flags ∪ ``--no-`` negations for
    BooleanOptionalAction flags (NOT store_true). Must match the canonical helpers
    exactly — this guards against a naive delegation that would drop the negations."""
    lwr = _import_tool("launch_witness_run")
    base = set(real_trainer_flags(TRAINER_PATH))
    bool_only = real_boolean_flags(TRAINER_PATH) - real_store_true_flags(TRAINER_PATH)
    expected = frozenset(base | {f.replace("--", "--no-", 1) for f in bool_only})
    got = lwr.real_trainer_flags()
    assert got == expected
    # The negation set is non-empty (the 2026-07-07 CLASS-fix must not be silently lost).
    assert any(f.startswith("--no-") for f in got)


def test_witness_autoconfig_tool_uses_canonical_base_scanner():
    wac_tool = _import_tool("witness_autoconfig")
    # The tool re-exports the canonical function (base flags only, no --no- forms).
    assert wac_tool.real_trainer_flags() == real_trainer_flags()
    assert not any(f.startswith("--no-") for f in wac_tool.real_trainer_flags())


# ─────────────────────────── #3 regex re-introduction ratchet ───────────────────────────
def test_migrated_tools_do_not_re_spell_the_scanner_regex():
    """The duplicated base-scan regex must live ONLY in curriculum_dsl now."""
    for rel in (
        "tools/launch_witness_run.py",
        "tools/witness_autoconfig.py",
    ):
        text = (_REPO / rel).read_text()
        assert _DUP_REGEX_LITERAL not in text, (
            f"{rel} re-spelled the canonicalized scanner regex — import "
            f"curriculum_dsl.real_trainer_flags instead"
        )
    # And it DOES still live in the canonical owner (sanity: we didn't delete the source).
    cdsl_text = (_REPO / "src/tac/witness_dsl/curriculum_dsl.py").read_text()
    assert _DUP_REGEX_LITERAL in cdsl_text
