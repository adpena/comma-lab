#!/usr/bin/env python3
"""Local pre-deploy harness — catches bug classes BEFORE paid Modal/Vast.ai dispatch.

Per operator 2026-05-15: *"shouldn't you run python compile locally before deploy
to find those bugs"*. Empirical anchor: Z3 v2 smoke `fc-01KRNHEGC9ZE48Y68GGJHP7FXN`
($2 wasted) + Z4 smoke `fc-01KRNHE942JSV7VRGXGR1FJGHQ` ($2 wasted) both crashed
on bugs that local 30s pre-flight would have caught:
- Z3 v2: archive ZIP member name `'x'` vs canonical `'0.bin'` (HNeRV parity L3)
- Z4: trainer didn't reach auth_eval stage (pre-auth-eval crash class)

This harness gates EVERY operator-authorize dispatch. Bug-class extinction.

Usage:
    .venv/bin/python tools/local_pre_deploy_check.py \\
        --trainer experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \\
        [--strict]   # exit 1 on any FAIL (default: warn-only)

Checks:
1. **PY-COMPILE**: `python -m py_compile <trainer>` — catches syntax errors
2. **ARCHIVE-GRAMMAR**: greps `_build_archive_zip` for `filename="0.bin"` per
   HNeRV parity discipline lesson 3 ("Archive grammar = monolithic single-file
   `0.bin`"). Refuses any other member name without `# ARCHIVE_MEMBER_OK:<reason>`.
3. **AUTH-EVAL-REACHABILITY**: greps trainer for `auth_eval_renderer` invocation
   OR `gate_auth_eval_call` (canonical helper per Catalog #226). Refuses if
   trainer has neither (per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable).
4. **CANONICAL-INFLATE-DEVICE**: per Catalog #205, refuses inline
   `torch.device("cuda" if ...)` in submission inflate.py.
5. **DETERMINISTIC-ZIP**: per Catalog #19, refuses bare `ZipFile.write` (must
   use `ZipInfo + writestr` with fixed timestamp).

Exit codes:
- 0 = all checks pass; safe to dispatch
- 1 = at least one check FAILED; do NOT dispatch (refund the $2 smoke)
- 2 = bad invocation (missing trainer / recipe)

Cost of running: ~5-15s wall-clock; $0. Cost of NOT running: $2-15 per failed
smoke + 30-60min wall-clock waiting for Modal to crash.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_ARCHIVE_MEMBER = "0.bin"
ARCHIVE_BUILD_PATTERN = re.compile(
    r'zipfile\.ZipInfo\s*\(\s*filename\s*=\s*["\']([^"\']+)["\']'
)
ARCHIVE_MEMBER_WAIVER = re.compile(r"#\s*ARCHIVE_MEMBER_OK:\s*\S")

AUTH_EVAL_REACHABILITY_PATTERNS = [
    re.compile(r"\bauth_eval_renderer\b"),
    re.compile(r"\bgate_auth_eval_call\b"),
    re.compile(r"\bcontest_auth_eval\b"),
]


def check_py_compile(trainer: Path) -> tuple[bool, str]:
    """Return (passed, message)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
            py_compile.compile(str(trainer), cfile=tmp.name, doraise=True)
        return True, f"PASS: {trainer.name} compiles cleanly"
    except py_compile.PyCompileError as e:
        return False, f"FAIL: {trainer.name} has syntax errors:\n{e.msg}"


def check_archive_grammar(trainer: Path) -> tuple[bool, str]:
    """Refuse `filename="x"` (or any non-canonical) without same-line waiver."""
    text = trainer.read_text()
    violations: list[str] = []
    for match in ARCHIVE_BUILD_PATTERN.finditer(text):
        member = match.group(1)
        if member == CANONICAL_ARCHIVE_MEMBER:
            continue
        line_no = text.count("\n", 0, match.start()) + 1
        line_text = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
        if ARCHIVE_MEMBER_WAIVER.search(line_text):
            continue
        violations.append(
            f"  {trainer.name}:{line_no}: filename={member!r} "
            f"(canonical: {CANONICAL_ARCHIVE_MEMBER!r}); "
            f"per HNeRV parity discipline lesson 3 + Catalog #146"
        )
    if violations:
        msg = "FAIL: archive ZIP member must be '0.bin' (canonical); violations:\n" + "\n".join(violations)
        msg += (
            "\n  Fix: change filename=\"x\" → filename=\"0.bin\" in _build_archive_zip"
            "\n  OR add same-line waiver: # ARCHIVE_MEMBER_OK:<rationale>"
        )
        return False, msg
    return True, f"PASS: {trainer.name} archive ZIP member is canonical"


def check_auth_eval_reachability(trainer: Path) -> tuple[bool, str]:
    """Refuse trainer with no auth_eval invocation (Z4 bug class)."""
    text = trainer.read_text()
    for pat in AUTH_EVAL_REACHABILITY_PATTERNS:
        if pat.search(text):
            return True, f"PASS: {trainer.name} invokes auth_eval (canonical helper)"
    return (
        False,
        f"FAIL: {trainer.name} has no auth_eval invocation; "
        f"trainer will reach Modal but never produce auth_eval_*.json. "
        f"Per CLAUDE.md 'Auth eval EVERYWHERE' non-negotiable, every training script must end with CUDA auth eval. "
        f"Fix: import + call gate_auth_eval_call from tac.substrates._shared.smoke_auth_eval_gate per Catalog #226.",
    )


def check_canonical_inflate_device(trainer: Path) -> tuple[bool, str]:
    """Per Catalog #205: no inline torch.device("cuda" if ...) in submission inflate.py.

    The trainer typically writes inflate.py via _write_runtime; check that
    template uses canonical select_inflate_device.
    """
    text = trainer.read_text()
    bad_pattern = re.compile(
        r'torch\.device\s*\(\s*["\']cuda["\']\s+if\s+torch\.cuda\.is_available'
    )
    matches = list(bad_pattern.finditer(text))
    if not matches:
        return True, f"PASS: {trainer.name} no inline torch.device cuda-fallback"
    line_nos = [text.count("\n", 0, m.start()) + 1 for m in matches]
    return (
        False,
        f"FAIL: {trainer.name} contains inline torch.device cuda-if-available at lines {line_nos}; "
        f"per Catalog #205 use canonical select_inflate_device from tac.substrates._shared.inflate_runtime",
    )


def check_deterministic_zip(trainer: Path) -> tuple[bool, str]:
    """Per Catalog #19: refuse bare ZipFile.write (must use ZipInfo + writestr)."""
    text = trainer.read_text()
    bad_pattern = re.compile(r'\bZipFile\s*\([^)]*\)\.write\s*\(')
    matches = list(bad_pattern.finditer(text))
    if not matches:
        return True, f"PASS: {trainer.name} uses deterministic-ZIP pattern"
    line_nos = [text.count("\n", 0, m.start()) + 1 for m in matches]
    return (
        False,
        f"FAIL: {trainer.name} uses bare ZipFile.write at lines {line_nos}; "
        f"per Catalog #19 use ZipInfo(filename, date_time=fixed) + writestr",
    )


def check_full_main_implemented(trainer: Path) -> tuple[bool, str]:
    """Refuse trainer with NotImplementedError stub `_full_main`.

    Empirical anchor 2026-05-15: Z4 + Z5 + s2sbs all have ``_full_main``
    that raises ``NotImplementedError`` — Modal dispatch reaches the trainer
    but immediately crashes pre-auth-eval. Per CLAUDE.md "Substrate scaffolds
    MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220: such trainers must be
    tagged ``research_only=true`` in their recipe AND NOT dispatchable.
    """
    text = trainer.read_text()
    # Match `def _full_main(...)` followed by docstring then `raise NotImplementedError`
    # within the function body (first ~30 lines after def).
    full_main_pat = re.compile(
        r"def\s+_full_main\s*\([^)]*\)[^:]*:\s*\n((?:[ \t]+.*\n){0,40})",
        re.MULTILINE,
    )
    match = full_main_pat.search(text)
    if not match:
        return True, f"PASS: {trainer.name} has no _full_main (smoke-only trainer; OK)"
    body = match.group(1)
    if re.search(r"\braise\s+NotImplementedError\b", body):
        line_no = text.count("\n", 0, match.start()) + 1
        return (
            False,
            f"FAIL: {trainer.name}:{line_no}: _full_main raises NotImplementedError "
            f"(stub trainer; not dispatchable). Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' + Catalog #220, "
            f"tag the recipe with research_only=true OR implement _full_main before dispatch.",
        )
    return True, f"PASS: {trainer.name} _full_main appears implemented"


CHECKS = [
    ("py_compile", check_py_compile),
    ("full_main_implemented", check_full_main_implemented),
    ("archive_grammar", check_archive_grammar),
    ("auth_eval_reachability", check_auth_eval_reachability),
    ("canonical_inflate_device", check_canonical_inflate_device),
    ("deterministic_zip", check_deterministic_zip),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local pre-deploy check; catches bug classes before paid dispatch."
    )
    parser.add_argument(
        "--trainer",
        required=True,
        type=Path,
        help="Path to experiments/train_substrate_*.py to validate",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Optional recipe name (for log clarity; not validated here)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any FAIL (default: warn-only)",
    )
    args = parser.parse_args()

    trainer = args.trainer
    if not trainer.is_absolute():
        trainer = (REPO_ROOT / trainer).resolve()
    if not trainer.is_file():
        print(f"FATAL: trainer not found: {trainer}", file=sys.stderr)
        return 2

    label = args.recipe or trainer.name
    print(f"[local-pre-deploy] validating: {trainer.name}  recipe={label}")
    print(f"[local-pre-deploy] mode: {'STRICT (exit 1 on fail)' if args.strict else 'WARN-ONLY'}")

    failed: list[str] = []
    for name, fn in CHECKS:
        ok, msg = fn(trainer)
        marker = "✓" if ok else "✗"
        print(f"  {marker} [{name}] {msg}")
        if not ok:
            failed.append(name)

    if failed:
        banner = "\n[local-pre-deploy] " + "=" * 70
        print(banner)
        print(f"[local-pre-deploy] {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
        print(
            "[local-pre-deploy] Per operator 2026-05-15 directive: every Modal/Vast.ai "
            "dispatch must pass this 30s harness BEFORE spending."
        )
        print(banner.lstrip("\n"))
        return 1 if args.strict else 0

    print(f"\n[local-pre-deploy] ALL {len(CHECKS)} CHECKS PASSED. Safe to dispatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
