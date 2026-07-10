#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed receiver-consumption bijection gate (#417, the NO-FAKE #8 class-fix).

THE BUG CLASS. The byte-close / archive path COUNTS a set of learned param groups
(pays the rate term for them). The receiver — the generated ``inflate.py`` forward,
which lives as the ``_INFLATE_PY`` SOURCE STRING in ``tools/levelset_byte_close_and_eval.py``
— CONSUMES only the groups it actually reads (``P["in_proj.weight"]``,
``P["hidden.%d.weight" % li]``, …). A group that is COUNTED but NOT CONSUMED is a
**dead counted byte**: it pays rate yet is inert through R, so a scored A/B on that
lever silently renders the shared-head CONTROL and reports a FAKE lever verdict
(NO-FAKE #8, surrogate-≠-authority on the authority surface itself). Verified
2026-07-10: v7.5.3 ``tex_trunk.*`` / v8 ``decoupled_head.*`` are produced by the
trainer but the receiver forward never reads them. Sister: the receiver ALREADY
hand-excludes ``pose_carrier.*`` from the base blob for exactly this reason — this
gate makes that fix STRUCTURAL and self-protecting for every future group.

WHY DRIFT-PROOF (not a hand allowlist). The receiver forward is a SOURCE STRING, so
the CONSUMED set is read by STATIC AST ANALYSIS of that exact shipped string on every
call — never a maintained list that rots (a hand list would already have missed
``out_tex_hidden`` / ``out_tex_h``, the optional deep texture head). Add a new group
to the archive without teaching the forward to read it → its param key never appears
as a ``P[...]`` subscript in the source → REFUSE, naming it. Teach the forward
(``P["tex_trunk.weight"]``) → it becomes consumed → PASS. Single source of truth =
the generated inflate.py source.

INTEGRITY GUARD. If the receiver source ever accesses ``P`` with a NON-literal,
non-format key (a dynamic ``P[name]``), static analysis can no longer prove the
consumed set. The extractor flags ``dynamic_access=True`` and the gate DEGRADES TO
ADVISORY (warns, does not refuse) rather than silently pass or false-refuse — the
honest fail-open for an un-analyzable receiver. The current source has zero dynamic
accesses (full-strength gate).

USAGE (wired into ``build_levelset_blob``): ``assert_receiver_bijection(base_order,
_INFLATE_PY, context=...)`` before trusting the counted archive. Escape hatch (rare,
explicit, loud): ``TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS=<prefix,...>`` or
``allow_unconsumed=...`` names groups intentionally counted-but-not-yet-consumed
(e.g. building an archive to MEASURE an in-progress receiver fix). Default empty =
fail-closed.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

# A %-format spec inside a receiver key, e.g. the "%d" in "hidden.%d.weight".
_FMT_SPEC = re.compile(r"%[#0\- +]*[0-9]*(?:\.[0-9]+)?[diouxXeEfFgGrsa]")


class ReceiverBijectionError(RuntimeError):
    """Raised when the archive counts a param group the receiver never consumes."""


@dataclass(frozen=True)
class ConsumedVocabulary:
    """The param keys the receiver forward reads, extracted from its source string.

    ``exact`` = literal ``P["name"]`` keys. ``patterns`` = (prefix, suffix) pairs from
    format keys like ``P["hidden.%d.weight" % li]`` -> ("hidden.", ".weight"), matching
    any ``hidden.<digits>.weight``. ``dynamic_access`` = the source read ``P`` with an
    unresolvable key (gate degrades to advisory if so)."""

    exact: frozenset[str] = field(default_factory=frozenset)
    patterns: tuple[tuple[str, str], ...] = ()
    dynamic_access: bool = False

    def consumes(self, name: str) -> bool:
        if name in self.exact:
            return True
        for pre, suf in self.patterns:
            if name.startswith(pre) and name.endswith(suf) and len(name) > len(pre) + len(suf):
                mid = name[len(pre): len(name) - len(suf)] if suf else name[len(pre):]
                if mid.isdigit():  # the %d index must be an integer -> a real indexed layer
                    return True
        return False


def extract_consumed_vocabulary(receiver_src: str, *, param_var: str = "P") -> ConsumedVocabulary:
    """Static-parse the receiver source for every ``<param_var>[...]`` read."""
    tree = ast.parse(receiver_src)
    exact: set[str] = set()
    patterns: set[tuple[str, str]] = set()
    dynamic = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == param_var):
            continue
        key = node.slice  # py3.9+: the expression directly (no ast.Index wrapper)
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            exact.add(key.value)
        elif (isinstance(key, ast.BinOp) and isinstance(key.op, ast.Mod)
              and isinstance(key.left, ast.Constant) and isinstance(key.left.value, str)):
            fmt = key.left.value
            parts = _FMT_SPEC.split(fmt, maxsplit=1)
            if len(parts) == 2:
                patterns.add((parts[0], parts[1]))
            else:  # a %-key with no recognizable spec -> treat as unresolvable
                dynamic = True
        else:
            # a non-literal / non-format key: P[name], P[i], P[a + b] ... unresolvable.
            dynamic = True
    return ConsumedVocabulary(frozenset(exact), tuple(sorted(patterns)), dynamic)


def orphaned_counted_params(
    counted: Iterable[str], vocab: ConsumedVocabulary, *, allow_unconsumed: Iterable[str] = (),
) -> list[str]:
    """COUNTED groups the receiver never CONSUMES (minus the explicit allow-list).

    A counted name is orphaned if the vocabulary does not consume it AND neither it nor
    any of its dotted prefixes is in ``allow_unconsumed`` (so ``tex_trunk`` waives
    ``tex_trunk.w_tex``). Deterministic (sorted)."""
    allow = frozenset(allow_unconsumed)

    def _allowed(name: str) -> bool:
        if name in allow:
            return True
        parts = name.split(".")
        return any(".".join(parts[:i]) in allow for i in range(1, len(parts)))

    return sorted(n for n in counted if not vocab.consumes(n) and not _allowed(n))


def format_refusal(orphans: list[str], vocab: ConsumedVocabulary, context: str) -> str:
    """A rule-chain failure message (per the preflight-cites-the-rule-chain discipline)."""
    consumed_desc = f"{len(vocab.exact)} exact + {len(vocab.patterns)} indexed pattern(s)"
    return (
        f"ReceiverBijectionError [{context}]: the archive COUNTS {len(orphans)} learned "
        f"param group(s) the receiver forward NEVER CONSUMES -> they pay rate but are INERT "
        f"through R (a scored A/B would measure the CONTROL = FAKE lever verdict, NO-FAKE #8). "
        f"Orphaned counted groups: {orphans}. Receiver consumes {consumed_desc}. "
        f"FIX (owner #395/#398): teach the receiver forward (the _INFLATE_PY source) to read "
        f"these groups via P[\"...\"], torch/MLX-parity-gated; OR relocate them out of the counted "
        f"base blob (as pose_carrier.* already is); OR, to build the archive anyway for an "
        f"in-progress-fix MEASUREMENT, set TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS to the group "
        f"prefix(es) (loud + explicit)."
    )


def check_receiver_bijection(
    counted: Iterable[str], receiver_src: str, *,
    allow_unconsumed: Iterable[str] = (),
) -> tuple[list[str], ConsumedVocabulary]:
    """Return (orphaned_counted_groups, consumed_vocabulary). Does NOT raise."""
    vocab = extract_consumed_vocabulary(receiver_src)
    orphans = orphaned_counted_params(counted, vocab, allow_unconsumed=allow_unconsumed)
    return orphans, vocab


def assert_receiver_bijection(
    counted: Iterable[str], receiver_src: str, *,
    context: str = "", allow_unconsumed: Iterable[str] = (),
    warn: Callable[[str], None] | None = None,
) -> ConsumedVocabulary:
    """Fail-closed: raise ReceiverBijectionError if any counted group is unconsumed.

    If the receiver source has a dynamic (unresolvable) P-access, DEGRADE TO ADVISORY:
    emit a warning via ``warn`` (default: print to stderr) and do not raise (the honest
    fail-open for an un-analyzable receiver). Returns the consumed vocabulary."""
    counted = list(counted)
    orphans, vocab = check_receiver_bijection(counted, receiver_src, allow_unconsumed=allow_unconsumed)
    if vocab.dynamic_access:
        msg = (f"[receiver-bijection ADVISORY {context or ''}]: the receiver source reads P with a "
               f"dynamic/unresolvable key -> static consumption analysis is INCOMPLETE; the gate "
               f"cannot prove counted-but-inert groups. Orphan candidates (unverified): {orphans}.")
        (warn or _default_warn)(msg)
        return vocab
    if orphans:
        raise ReceiverBijectionError(format_refusal(orphans, vocab, context or "receiver-bijection"))
    return vocab


def _default_warn(msg: str) -> None:
    import sys
    print(msg, file=sys.stderr, flush=True)


__all__ = [
    "ConsumedVocabulary",
    "ReceiverBijectionError",
    "assert_receiver_bijection",
    "check_receiver_bijection",
    "extract_consumed_vocabulary",
    "format_refusal",
    "orphaned_counted_params",
]
