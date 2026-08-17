"""ddm_cd1 (2026-08-17): the DDM-independent recall organs must run on BOTH paths.

Anchor incident: ``tools/costate_digest.py`` ``build_digest`` opened ``if ddm_live:``
at :2200 and closed it with ``return lines, data`` at :2251, so :2253-:2330 was
unreachable whenever ddm_live was true -- the LIVE state. The branch still
re-provided a total 16-key schema, so nothing looked absent; instead FOUR keys
carried plausible WRONG values (``verdict_scope`` shadowed by an unrelated
provenance dict, ``corpus_recall`` hard-bound to ``[]`` against its own
``dict | None`` contract, ``active_convening``/``graph_memory`` bound to None).
Seven verdict-scope recall advisories reached a sink whose only reader was dead
code.

These tests are STRUCTURAL and value-contractual rather than advisory: they read
the shipped source and the built payload, so re-truncating the builder turns them
red instead of emitting a nag nobody runs.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DIGEST = REPO / "tools" / "costate_digest.py"
sys.path.insert(0, str(REPO / "tools"))

# The organs that read the research corpus, the convening ledger, the advisory
# sink and the graph memory. None of these is dominated by the live DDM receipt
# fleet, so none may sit inside a single branch of the ddm_live split.
SHARED_SECTIONS = (
    "section_verdict_scope_advisories",
    "section_active_convening",
    "section_corpus_recall",
    "section_graph_memory",
)


def _build_digest_fn() -> ast.FunctionDef:
    tree = ast.parse(DIGEST.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_digest":
            return node
    raise AssertionError("build_digest not found in tools/costate_digest.py")


def _called_names(node: ast.AST) -> set[str]:
    return {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_shared_recall_organs_are_reachable_on_every_path() -> None:
    """Each shared organ must run on BOTH paths -- REACHABILITY, not position.

    The first draft of this test asserted the organs sat at build_digest's top
    level rather than inside an ``if``. It PASSED on the pre-fix source, because
    the truncated statements were still top-level -- merely unreachable. Position
    is not the property that matters; being preceded by an unconditional-return
    guard is. That is the measured-object-vs-named-object trap, caught here by
    mutation-checking the test against the pre-fix file instead of trusting it.
    """
    fn = _build_digest_fn()
    unreachable: dict[str, str] = {}
    for name in SHARED_SECTIONS:
        first = next(
            (i for i, stmt in enumerate(fn.body) if name in _called_names(stmt)),
            None,
        )
        assert first is not None, f"{name} is no longer called by build_digest"
        for stmt in fn.body[:first]:
            # A top-level `if ...:` whose body ends in a Return and that has no
            # `else` short-circuits every later statement on the true path.
            if (
                isinstance(stmt, ast.If)
                and stmt.body
                and isinstance(stmt.body[-1], ast.Return)
                and not stmt.orelse
            ):
                unreachable[name] = (
                    f"unreachable when the guard at line {stmt.lineno} is taken"
                )
                break
    assert not unreachable, (
        "these DDM-independent recall organs do NOT run on every path: "
        f"{unreachable}. That is the ddm_cd1 truncation shape -- an early return "
        "orphaned the tail. Make the split an explicit if/else so the shared tail "
        "runs on BOTH the live-DDM and the legacy path."
    )


def test_build_digest_has_no_dead_conditional_retest() -> None:
    """The class guard must be clean on its own anchor file."""
    from tac.confound_gates import (
        check_no_dead_conditional_retest_after_early_return as gate,
    )

    violations = [v for v in gate(strict=False, verbose=False) if "costate_digest" in v]
    assert not violations, violations


def test_poisoned_key_shapes_never_ship() -> None:
    """The two keys the anchor poisoned must honour their sections' contracts.

    ``section_corpus_recall`` is typed ``tuple[list[str], dict | None]`` -- a list
    payload is out of contract by construction, which is exactly what ``[]`` was.
    ``verdict_scope`` must hold ADVISORIES; the run's authority provenance now
    lives under its own key, so seeing provenance fields here means the shadowing
    returned.
    """
    import costate_digest as cd

    _lines, data = cd.build_digest(include_fm=False)

    recall = data["corpus_recall"]
    assert recall is None or isinstance(recall, dict), (
        f"corpus_recall must be dict|None per section_corpus_recall's own "
        f"signature; got {type(recall).__name__} = {recall!r}"
    )

    scope = data["verdict_scope"]
    assert scope is None or isinstance(scope, dict)
    if isinstance(scope, dict):
        shadow = {"evidence_axis", "promotion_eligible", "main_landing_review_required"}
        assert not (shadow & set(scope)), (
            "verdict_scope is holding the run's AUTHORITY PROVENANCE again "
            f"(keys {sorted(shadow & set(scope))}). That object belongs under "
            "ddm_authority_provenance; verdict_scope owns the advisory organ."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
