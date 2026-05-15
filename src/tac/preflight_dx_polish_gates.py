# SPDX-License-Identifier: MIT
"""DX-POLISH-WAVE STRICT preflight gates (Catalog #238).

Sister module to ``src/tac/preflight.py`` per CLAUDE.md "Subagent
coherence-by-default" sister-subagent ownership map (Catalog #230). The
canonical ``src/tac/preflight.py`` is owned by the R2-CRITICAL-FIX
parallel subagent; landing this gate as a sibling module preserves the
sister-disjoint scope while keeping the gate immediately invocable by
``tools/preflight_hook.py`` and the dedicated test suite.

Catalog #238 — ``check_smoke_path_default_relaxes_clean_head``
=============================================================

Refuses any state of ``tools/run_modal_smoke_before_full.py`` that drops
the DX-POLISH-WAVE 2026-05-15 (DX-4) contract: the SMOKE phase MUST
auto-detect a dirty working tree and activate the Catalog #202 paired-
env bypass for the smoke dispatch only. The FULL phase MUST NOT
auto-relax the clean-head check. Together these surfaces extinct the
"cheap smoke probes blocked by sister-subagent _codex.md research
ledger dirty paths" bug class anchored 2026-05-13 (`feedback_grand_council_ad_hoc_dispatch_unblock_landed_20260514.md`)
+ 2026-05-14 (`feedback_harness_sigurg_kills_subagent_modal_dispatch_permanent_20260514.md`).

Required surfaces in ``tools/run_modal_smoke_before_full.py``:

  1. ``def _count_dirty_paths`` — the canonical helper that runs
     ``git status --porcelain`` and returns the count.
  2. ``OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK`` literal in the
     smoke-spawn function body (the Catalog #202 intent env var).
  3. ``OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED`` literal in
     the same function (the paired attestation env var).
  4. The dirty-count guard ``if dirty_count > 0:`` (or equivalent
     positive-count check) before the env injection so a clean tree
     does NOT get the bypass spuriously activated.
  5. Catalog reference token ``Catalog #238`` in the smoke-spawn body
     (proves the wire-in is intentional and reviewable).
  6. Catalog reference token ``Catalog #238`` in the full-spawn body
     (proves the FULL-phase non-relaxation is intentional, not an
     oversight).
  7. The full-spawn function body MUST NOT export
     ``OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK`` — that would
     re-introduce the bug class at the FULL surface.

There is no waiver. The seven surfaces ARE the contract.

Sister gates
------------

* Catalog #166 (`check_modal_dispatch_verifies_worker_source_matches_head`)
  — the worker-side sentinel hash check that runs INDEPENDENTLY of
  ``--require-clean-head``. Catalog #238 relaxes the whole-tree check
  for SMOKE only; Catalog #166's sentinel hash check still fires.
* Catalog #199 (`check_operator_authorize_bypass_requires_session_budget`)
  — the paired-env contract for ``OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE``.
* Catalog #202 (`check_catalog_202_bypass_requires_paired_env_attestation`)
  — the paired-env contract for the whole-tree clean-check bypass that
  Catalog #238 activates programmatically for the smoke phase.
* Catalog #167 (`check_substrate_dispatch_uses_smoke_before_full_pattern`)
  — the gate that mandates the wrapper exist; Catalog #238 ensures the
  wrapper does the right thing on dirty trees.

Wire-in plan
------------

Per CLAUDE.md "Strict-flip atomicity rule", this gate lands STRICT @ 0
in the same commit batch as the DX-POLISH-WAVE behavior changes. The
sister R2-CRITICAL-FIX subagent owns ``src/tac/preflight.py``; the
operator-routed integration commit will add a single line to
``preflight_all()`` once the sister wave settles:

    from tac.preflight_dx_polish_gates import check_smoke_path_default_relaxes_clean_head
    check_smoke_path_default_relaxes_clean_head(strict=True, repo_root=root)

Until then, ``tools/preflight_hook.py`` invokes the gate directly when
the DX-polish module is importable, and the dedicated test suite
``src/tac/tests/test_check_238_smoke_path_default_relaxes_clean_head.py``
guards the contract from byte zero.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "DX_POLISH_GATES_CATALOG_NUMBER",
    "PreflightDxPolishError",
    "check_smoke_path_default_relaxes_clean_head",
]

DX_POLISH_GATES_CATALOG_NUMBER = 238

_SMOKE_WRAPPER_PATH = "tools/run_modal_smoke_before_full.py"
_SMOKE_FUNCTION_NAME = "_spawn_smoke_dispatch"
_FULL_FUNCTION_NAME = "_spawn_full_dispatch"

# Required surfaces: (anchor_in_text, friendly_label).
_REQUIRED_TOP_LEVEL_SURFACES: tuple[tuple[str, str], ...] = (
    ("def _count_dirty_paths", "_count_dirty_paths helper definition"),
)

_REQUIRED_SMOKE_BODY_SURFACES: tuple[tuple[str, str], ...] = (
    (
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK",
        "Catalog #202 intent env var injection in smoke-spawn body",
    ),
    (
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED",
        "Catalog #202 attestation env var injection in smoke-spawn body",
    ),
    ("Catalog #238", "Catalog #238 reference token in smoke-spawn body"),
)

_REQUIRED_FULL_BODY_SURFACES: tuple[tuple[str, str], ...] = (
    ("Catalog #238", "Catalog #238 reference token in full-spawn body"),
)

_DIRTY_COUNT_GUARD_RX = re.compile(
    r"\bif\s+dirty_count\s*[>!]=?\s*0\s*:"
)


class PreflightDxPolishError(RuntimeError):
    """Raised when a DX-polish strict gate finds a contract violation."""


def _slice_function_body(text: str, function_name: str) -> str:
    """Return source text from ``def <function_name>`` through the next
    top-level ``def`` / ``class`` / EOF.

    Used to scope per-function surface checks so a token reference in
    one function (e.g. the smoke-spawn body) does not falsely satisfy a
    check intended for a different function (e.g. the full-spawn body).
    """

    pattern = re.compile(
        rf"^def\s+{re.escape(function_name)}\b",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    rest = text[start:]
    # Find the next top-level `def ` or `class ` after the function header.
    next_def = re.search(r"^(?:def|class)\s+\w+\b", rest[1:], re.MULTILINE)
    if next_def is None:
        return rest
    return rest[: next_def.start() + 1]


def check_smoke_path_default_relaxes_clean_head(
    *,
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Catalog #238. Refuse any state of the smoke wrapper that drops the
    DX-POLISH-WAVE 2026-05-15 dirty-tree auto-detect contract.

    Returns a list of human-readable violation strings. Empty list = clean.
    With ``strict=True``, raises :class:`PreflightDxPolishError` on any
    violation so the gate fails closed in CI.
    """

    root = (repo_root or Path.cwd()).resolve()
    path = root / _SMOKE_WRAPPER_PATH

    violations: list[str] = []

    if not path.is_file():
        violations.append(
            f"{_SMOKE_WRAPPER_PATH}: missing — the canonical smoke-before-"
            "full wrapper is gone; Catalog #167 + Catalog #238 contracts "
            "are re-exposed."
        )
        if violations and strict:
            raise PreflightDxPolishError(
                "check_smoke_path_default_relaxes_clean_head found "
                f"{len(violations)} violation(s):\n  "
                + "\n  ".join(violations)
            )
        return violations

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        violations.append(f"{_SMOKE_WRAPPER_PATH}: read error {exc!s}")
        if strict:
            raise PreflightDxPolishError(violations[-1]) from exc
        return violations

    # Top-level surfaces.
    for anchor, label in _REQUIRED_TOP_LEVEL_SURFACES:
        if anchor not in text:
            violations.append(
                f"{_SMOKE_WRAPPER_PATH}: missing required top-level surface "
                f"`{label}` (anchor `{anchor}`)."
            )

    # Smoke-spawn body surfaces.
    smoke_body = _slice_function_body(text, _SMOKE_FUNCTION_NAME)
    if not smoke_body:
        violations.append(
            f"{_SMOKE_WRAPPER_PATH}: missing function "
            f"`def {_SMOKE_FUNCTION_NAME}` — Catalog #238 cannot enforce "
            "the smoke-relaxed clean-head contract without the canonical "
            "smoke-spawn function."
        )
    else:
        for anchor, label in _REQUIRED_SMOKE_BODY_SURFACES:
            if anchor not in smoke_body:
                violations.append(
                    f"{_SMOKE_WRAPPER_PATH}::{_SMOKE_FUNCTION_NAME}: missing "
                    f"required surface `{label}` (anchor `{anchor}`)."
                )
        if not _DIRTY_COUNT_GUARD_RX.search(smoke_body):
            violations.append(
                f"{_SMOKE_WRAPPER_PATH}::{_SMOKE_FUNCTION_NAME}: missing "
                "the `if dirty_count > 0:` (or equivalent) guard before the "
                "Catalog #202 env-var injection. Without the positive-count "
                "guard a clean tree gets the bypass spuriously activated."
            )

    # Full-spawn body surfaces (and forbidden surface).
    full_body = _slice_function_body(text, _FULL_FUNCTION_NAME)
    if not full_body:
        violations.append(
            f"{_SMOKE_WRAPPER_PATH}: missing function "
            f"`def {_FULL_FUNCTION_NAME}` — Catalog #238 cannot enforce "
            "the FULL-phase non-relaxation contract without the canonical "
            "full-spawn function."
        )
    else:
        for anchor, label in _REQUIRED_FULL_BODY_SURFACES:
            if anchor not in full_body:
                violations.append(
                    f"{_SMOKE_WRAPPER_PATH}::{_FULL_FUNCTION_NAME}: missing "
                    f"required surface `{label}` (anchor `{anchor}`)."
                )
        # FORBIDDEN surface: the FULL phase MUST NOT export the Catalog #202
        # intent env var. Activating the bypass for the FULL canary would
        # re-introduce the bug class at the costliest surface.
        if "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK" in full_body:
            violations.append(
                f"{_SMOKE_WRAPPER_PATH}::{_FULL_FUNCTION_NAME}: FORBIDDEN "
                "surface `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK` "
                "found in full-spawn body. The FULL phase MUST NOT auto-"
                "activate the Catalog #202 paired-env bypass; only the "
                "SMOKE phase does. Operator override (manual env export) "
                "is fine; landing it inside the wrapper's full-spawn body "
                "re-introduces the bug class at the $5-15 dispatch surface."
            )

    if verbose:
        if violations:
            print(
                f"  [dx-polish-238] {len(violations)} violation(s) in "
                f"{_SMOKE_WRAPPER_PATH}:"
            )
            for v in violations[:10]:
                print(f"    - {v[:240]}")
        else:
            print(
                "  [dx-polish-238] OK (smoke wrapper auto-detects dirty "
                "tree + relaxes clean-head for SMOKE only; FULL stays "
                "fail-closed per default)"
            )

    if violations and strict:
        raise PreflightDxPolishError(
            "check_smoke_path_default_relaxes_clean_head found "
            f"{len(violations)} contract violation(s). Catalog #238 (DX-"
            "POLISH-WAVE 2026-05-15 / DX-4) cannot be dropped from the "
            "smoke wrapper without re-exposing the bug class:\n  "
            + "\n  ".join(v[:300] for v in violations[:5])
        )

    return violations
