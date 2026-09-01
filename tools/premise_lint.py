#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""premise_lint — the CANONICAL falsified-premise matcher (one matcher, many surfaces).

Born 2026-08-31 from the fpr1/ea1 arc: the curated registry
``.omx/research/falsified_premise_registry.jsonl`` guarded exactly ONE surface — codex
charter SPAWNS (``codex_arm_queue._lint_falsified_premises``) — while the day's actual
propagation vector was MAIN-authored MEMOS (jt1 cited a number its owning memo had
published a do-not-cite list against, and passed no lint because memos never crossed the
spawn path). Seven reading-semantics instances in one day, one genus.

This module extracts the matcher so BOTH surfaces consume one implementation:

  * ``tools/codex_arm_queue.py`` delegates here (charter spawns — the original surface);
  * ``tools/subagent_commit_serializer.py`` runs the falsified-premise matcher
    advisory-only and the SHA-transcription matcher refusing over staged
    ``.omx/research/*.md`` files (the memo surface the genus actually used).

FALSIFIED-PREMISE CONTRACT (inherited from the keeper's function and BINDING on every
consumer): advisory only, and silent on every registry failure. The independent
SHA-transcription leg is refusing because its inputs are the edited text plus canonical
pins, not an optional recall store. Registry rows use ``claim_patterns`` (substring
match after lowercasing + unicode-dash normalisation) and an ``origin`` naming the
QUANTITY; this is deliberately a curated store, never the auto-scraped corrections
index, whose window-adjacency rows cannot carry quantity identity (the retired
``_lint_stale_numbers`` verdict — precision cannot be added at the consumer).

USAGE (CLI is advisory; rc=0 always unless --strict-rc):
  .venv/bin/python tools/premise_lint.py --file MEMO.md
  .venv/bin/python tools/premise_lint.py --file MEMO.md --subject "staged memo"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_RESEARCH_DIR = _REPO / ".omx" / "research"

#: Default registry union: the standing undated path + the dated path the registry was
#: born in. Consumers with their own resolution (the keeper honours test overrides)
#: pass ``registries=`` explicitly and this default is not consulted.
DEFAULT_REGISTRIES: tuple[Path, ...] = (
    _RESEARCH_DIR / "falsified_premise_registry.jsonl",
    _RESEARCH_DIR / "falsified_premise_registry_20260828.jsonl",
)

_WARN_CAP = 5
_HEX_RUN_RE = re.compile(r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{16,})(?![0-9A-Fa-f])")


def _normalise(text: str) -> str:
    """Lowercase + unicode-dash normalisation so "13-23%" matches "13–23%"."""
    return text.replace("–", "-").replace("—", "-").lower()


def lint_text(
    text: str,
    registries: list[Path] | tuple[Path, ...] | None = None,
    *,
    subject: str = "charter",
    cap: int = _WARN_CAP,
) -> list[str]:
    """Return advisory warnings for every registered falsified premise ``text`` restates.

    ``subject`` names the surface in the warning ("charter", "staged memo", ...).
    Silent on EVERY failure per the contract: any exception, missing file, or malformed
    row yields no warning for that row — never an exception to the caller.
    """
    paths = list(registries if registries is not None else DEFAULT_REGISTRIES)
    paths = [p for p in paths if isinstance(p, Path) and p.is_file()]
    if not paths:
        return []
    haystack = _normalise(text)
    warnings: list[str] = []
    try:
        rows: list[str] = []
        for registry in paths:
            rows.extend(registry.read_text(encoding="utf-8").splitlines())
        for line in rows:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            patterns = row.get("claim_patterns") or []
            if not isinstance(patterns, list):
                continue
            hit = next(
                (
                    p
                    for p in patterns
                    if isinstance(p, str) and p and _normalise(p) in haystack
                ),
                None,
            )
            if hit is None:
                continue
            origin = row.get("origin")
            if not isinstance(origin, dict):
                origin = {}
            falsifications = row.get("falsifications") or []
            measured = "; ".join(
                f"{Path(str(f.get('path', '?'))).name} -> {f.get('measured', '?')}"
                f" ({f.get('scale', '?')})"
                for f in falsifications
                if isinstance(f, dict)
            )
            warnings.append(
                f"RECALL: {subject} restates {hit!r} ({row.get('topic', '?')}) — FALSIFIED "
                f"premise. Origin {Path(str(origin.get('path', '?'))).name} "
                f"n={origin.get('n_pairs', '?')} authority={origin.get('authority_mode', '?')}"
                f", score_claim={origin.get('score_claim', '?')}. Later measurements: "
                f"{measured or 'see registry'}. verdict_scope="
                f"{row.get('verdict_scope', '?')}. Re-derive before citing."
            )
            if len(warnings) >= cap:
                break
    except Exception:
        return warnings
    return warnings


def canonical_frontier_shas(pointer_path: Path | None = None) -> set[str]:
    """Return full SHA pins recursively present in the canonical frontier pointer.

    A missing or malformed pointer returns an empty set.  This helper is shared by
    the charter and staged-memo consumers; they must not grow separate pin loaders.
    """

    path = pointer_path or (_REPO / ".omx" / "state" / "canonical_frontier_pointer.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    pins: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and "sha" in str(key).lower():
                    token = child.lower()
                    if re.fullmatch(r"[0-9a-f]{64}", token):
                        pins.add(token)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return pins


def lint_sha_prefix_divergent_tails(
    text: str,
    *,
    canonical_shas: set[str] | None = None,
    subject: str = "text",
    cap: int = _WARN_CAP,
) -> list[str]:
    """Refuse likely SHA transcription drift without policing unrelated hashes.

    A token is suspicious only when it shares at least eight leading hex digits
    with a canonical full SHA but diverges before either token ends.  Exact full
    matches and honest abbreviated prefixes therefore pass.  Full 64-hex pins in
    the same document join the frontier-pointer pins, catching an internally
    inconsistent memo even when the pointer is unavailable.
    """

    observed = [match.group(1).lower() for match in _HEX_RUN_RE.finditer(text)]
    document_pins = {token for token in observed if len(token) == 64}
    pins = set(canonical_shas if canonical_shas is not None else canonical_frontier_shas())
    pins.update(document_pins)
    warnings: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for token in observed:
        for pin in sorted(pins):
            if token == pin or token.startswith(pin) or pin.startswith(token):
                continue
            common = 0
            for left, right in zip(token, pin, strict=False):
                if left != right:
                    break
                common += 1
            if common < 8:
                continue
            pair = tuple(sorted((token, pin)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            warnings.append(
                f"SHA-TRANSCRIPTION: {subject} carries {token}; its first {common} hex "
                f"digits match canonical pin {pin}, then the tail diverges. Refuse and "
                "copy the canonical SHA verbatim."
            )
            if len(warnings) >= cap:
                return warnings
    return warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True, help="text file to lint")
    ap.add_argument("--subject", default="text", help="surface name used in warnings")
    ap.add_argument("--registry", action="append", help="explicit registry path(s)")
    ap.add_argument(
        "--strict-rc",
        action="store_true",
        help="rc=1 when any warning fires (default rc=0 always — advisory)",
    )
    args = ap.parse_args(argv)
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[premise-lint] cannot read {args.file}: {exc}", file=sys.stderr)
        return 0 if not args.strict_rc else 2
    registries = [Path(r) for r in args.registry] if args.registry else None
    warnings = lint_text(text, registries, subject=args.subject)
    for warning in warnings:
        print(f"[premise-lint] {warning}")
    return 1 if (warnings and args.strict_rc) else 0


if __name__ == "__main__":
    sys.exit(main())
