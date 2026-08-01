"""Static guard: Metal kernel buffer names must not be reserved words.

WHY THIS IS STATIC AND NOT A RUNTIME CHECK
==========================================
``mx.fast.metal_kernel`` splices each ``input_names`` / ``output_names`` entry
VERBATIM into the generated Metal signature (``const device float* <name>``).
A name that is a Metal Shading Language reserved word therefore produces a
source file that cannot compile — and the failure surfaces only at DISPATCH,
on a machine that has Metal.

MEASURED (ddm_tr6, 2026-08-01, commit b02b99cecb): the phase forward + VJP
kernels in ``local_acceleration/metal_micro_batch_v9_levers.py`` named a buffer
``signed``. They FAILED TO COMPILE ON EVERY DISPATCH from 2026-07-12 until the
rename — 20 days — and nothing noticed, because:

  * CI runs ubuntu with ``.[dev,runtime]``; ``mlx`` has no Linux wheel, so every
    MLX-gated module is SKIPPED and pytest reports skip as GREEN; and
  * the landing commit's own message recorded "Metal-verify owed (codex sandbox
    has no Metal)" — a debt that was never paid.

That is the vacuity genus (``vacuity_is_indistinguishable_from_pass``): an
instrument that examined nothing emitted the same symbol as one that examined
everything cleanly. **A guard that needs Metal to run would inherit exactly that
blindness.** This one is a pure AST + string check, so it runs on any host,
including the Linux CI that cannot import mlx at all.

tr6 fixed the instance and left a code comment describing the mechanism.
CLAUDE.md is explicit that comment-only contracts are FORBIDDEN — a comment
cannot refuse the next occurrence. This module is that refusal.

RESERVED-WORD LIST PROVENANCE
=============================
DERIVED, not hand-picked: Metal Shading Language is specified as C++14 with
additions, so the set is the C++14 keyword list (which is where ``signed``
lives — the exact word that bit us) UNION the Metal-specific address-space,
function-qualifier, and texture-access keywords. Deliberately NOT included:
``half``/``float2``-style vector and texture TYPE names, because a buffer named
``float2`` is legal-if-confusing and flagging it would be a false positive. The
guard refuses what the compiler refuses, nothing more.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# --- C++14 keywords (ISO/IEC 14882:2014 §2.12). `signed` is in here. ---------
_CPP14_KEYWORDS: frozenset[str] = frozenset(
    ["alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class", "compl", "const", "const_cast", "constexpr", "continue", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"]
)

# --- Metal-specific: address spaces, function qualifiers, texture access. ----
_METAL_KEYWORDS: frozenset[str] = frozenset(
    ["kernel", "vertex", "fragment", "device", "constant", "threadgroup", "thread", "threadgroup_imageblock", "ray_data", "object_data", "mesh", "access", "read", "write", "read_write", "sample", "buffer", "texture", "sampler", "stage_in"]
)

RESERVED_METAL_IDENTIFIERS: frozenset[str] = _CPP14_KEYWORDS | _METAL_KEYWORDS

#: Kwargs whose string elements are spliced verbatim into the Metal signature.
_SPLICED_KWARGS: tuple[str, ...] = ("input_names", "output_names")

_WAIVER_TOKEN = "METAL_RESERVED_IDENTIFIER_OK:"

# --- SCOPE, DERIVED (not a generic rglob) ------------------------------------
# First cut of this guard swept `rglob("*.py")` over three roots and examined
# 62,799 files — 8,724 of them vendored public-PR intake clones, plus a vendored
# numpy tree — and TIMED OUT at 60 s. That is the generic-default reflex applied
# to a scope instead of a basis: nobody derived what to examine.
#
# CLAUDE.md forbids touching intake clones at all ("Forbidden in-place edits to
# public PR intake clones"): they are pristine forensic inputs, we do not ship
# them, and we cannot fix their kernels. Flagging them would be pure noise, and
# a noisy guard gets overridden.
#
# Mirrors `_VENDORED_PATH_MARKERS` in `tac.preflight` (module-local there, so it
# cannot be imported — SOURCE OF TRUTH is preflight.py; keep in sync).
_VENDORED_PATH_MARKERS: tuple[str, ...] = (
    "/pr_heads/",
    "/leaderboard_intel_",
    "/reverse_engineering_",
    "/public_runtime_adapters_",
    "/raw/kaggle_ingest/",
    "/vendored/",
    "_intake_",
    "/av1_crf31_bicubic/",
)

#: Directories that are never our source under any root.
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"__pycache__", ".venv", "venv", "node_modules", "site-packages", ".git"}
)


def _is_our_source(path: Path) -> bool:
    """True iff ``path`` is code WE ship and can fix."""
    if _EXCLUDED_DIR_PARTS.intersection(path.parts):
        return False
    posix = path.as_posix()
    return not any(marker in posix for marker in _VENDORED_PATH_MARKERS)


@dataclass(frozen=True)
class ReservedIdentifierViolation:
    """One kernel buffer name that the Metal compiler will reject."""

    path: str
    line: int
    kwarg: str
    name: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.kwarg} contains reserved Metal/C++ "
            f"identifier {self.name!r} — mx.fast.metal_kernel splices it verbatim "
            f"into the generated signature, so the kernel will FAIL TO COMPILE at "
            f"dispatch (silently, on any host without Metal). Rename the buffer."
        )


def _string_elements(node: ast.AST) -> list[tuple[str, int]]:
    """Literal strings in a list/tuple, with line numbers. Non-literals ignored.

    A dynamically-built name list is NOT flagged: we would be guessing, and a
    guess that fires is a false positive that trains people to override.
    """
    out: list[tuple[str, int]] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append((elt.value, getattr(elt, "lineno", 0)))
    return out


def scan_source(source: str, path: str = "<memory>") -> list[ReservedIdentifierViolation]:
    """Return every reserved-word buffer name in ``source``.

    Waiver: a same-line ``# METAL_RESERVED_IDENTIFIER_OK:<rationale>`` comment.
    Note we read the raw line rather than ``ast.get_source_segment`` — that
    helper EXCLUDES trailing comments, which is precisely the narrowed-detector
    bug ddm_tr6 hit while fixing this same family. A guard whose waiver is
    invisible is worse than no waiver.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    violations: list[ReservedIdentifierViolation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in _SPLICED_KWARGS:
                continue
            for name, lineno in _string_elements(kw.value):
                if name not in RESERVED_METAL_IDENTIFIERS:
                    continue
                idx = lineno - 1
                raw = lines[idx] if 0 <= idx < len(lines) else ""
                if _WAIVER_TOKEN in raw:
                    continue
                violations.append(
                    ReservedIdentifierViolation(
                        path=path, line=lineno, kwarg=kw.arg or "", name=name
                    )
                )
    return violations


def scan_paths(paths: Iterable[Path]) -> tuple[list[ReservedIdentifierViolation], int]:
    """Scan files; return (violations, examined_count).

    The count is returned so callers can report the DENOMINATOR — an empty scope
    must never be reported as a clean pass (see ``tac.scope_ledger`` and
    ``vacuity_is_indistinguishable_from_pass``).
    """
    violations: list[ReservedIdentifierViolation] = []
    examined = 0
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        examined += 1
        # CHEAP PRE-FILTER, not a scope reduction: a file with neither spliced
        # kwarg cannot contain a violation, so parsing it is pure cost. It stays
        # in `examined` because it WAS examined — narrowing the reported
        # denominator to "files we bothered to parse" would be the vacuity bug
        # (see tac.scope_ledger). Measured: 16,908 files, 36.7 s full-parse vs
        # ~1 s with the pre-filter; only ~20 files carry the token.
        if not any(kw in source for kw in _SPLICED_KWARGS):
            continue
        violations.extend(scan_source(source, str(path)))
    return violations, examined


def check_metal_kernel_identifiers_not_reserved(
    repo_root: Path | str = ".",
    *,
    strict: bool = True,
    roots: Sequence[str] = ("src/tac", "experiments", "tools"),
) -> tuple[list[ReservedIdentifierViolation], int]:
    """Refuse any Metal kernel buffer named a reserved Metal/C++ word.

    Returns ``(violations, examined_count)``. Raises in ``strict`` mode.
    """
    root = Path(repo_root)
    candidates = [
        p
        for r in roots
        for p in sorted((root / r).rglob("*.py"))
        if _is_our_source(p)
    ]
    violations, examined = scan_paths(candidates)

    if violations and strict:
        detail = "\n  ".join(v.render() for v in violations)
        raise ValueError(
            f"check_metal_kernel_identifiers_not_reserved: {len(violations)} "
            f"violation(s) across {examined} file(s) examined:\n  {detail}"
        )
    return violations, examined
