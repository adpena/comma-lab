"""STRICT preflight check: ChARM-class files must actually implement
channel-conditional autoregression — not just declare a context net.

Codex finding (2026-05-08, /tmp/codex_runs/phase_a_codemath_20260508T161501Z):
> A4 is labeled ChARM but never conditions on previous channels. The model
> defines `self.context`, but the forward path derives mu, sigma only from z
> and never calls the context network or conditions channel c on channels < c.
> This is not ChARM 2020 channel-conditional autoregression on INT8 residuals;
> it is at best a factorized hyperprior proxy with uncoded raw bytes.

Scope: any class named `*Charm*`, `*ChARM*`, `*ar_codec*`, `*Hyperprior*` that
also defines a `context` attribute (or `self.context = ...` assignment) MUST
have a forward method that actually CALLS that context network with a per-
channel-prior input (e.g., `self.context(prior_summary)` inside a loop).

Gate semantics (STRICT mode):
- Detect classes matching the name patterns above.
- AST-walk each class body for both:
  - A context attribute assignment (`self.context = SomeNet(...)`) in __init__
  - A `self.context(...)` invocation in any method
- If a class declares `self.context` but never calls it, FAIL the check.

Wiring this into `preflight_all()`:
The full preflight orchestrator lives in `src/tac/preflight.py` (24990 lines,
currently being edited by 4 sister subagents in the same session). To avoid
merge conflicts on commit, this gate is published as a standalone module and
wire-in is deferred to a follow-up commit. Until then, callers can invoke
`check_charm_class_actually_implements_channel_conditional()` directly.

CLAUDE.md catalog row 114 (proposed): "ChARM/AR-codec class declares context
network but never invokes it — relabeled hyperprior masquerading as ChARM."

[empirical:experiments/train_charm_50k_toy_substrate.py — pre-fix violation;
 post-fix at commit pending]
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

CHARM_NAME_PATTERN = re.compile(
    r"(?:Charm|ChARM|charm_2020|charm2020|ar_codec|ARCodec|ar_codec_2020)",
    re.IGNORECASE,
)


class CharmClassCheckError(Exception):
    """Raised by the STRICT-mode check when a ChARM class is misnamed."""


def _walk_class_methods_for_context_call(class_node: ast.ClassDef) -> bool:
    """Return True if any method in this class invokes self.context(...)."""
    for node in ast.walk(class_node):
        if isinstance(node, ast.Call):
            func = node.func
            # Pattern: self.context(...)
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr == "context"
            ):
                return True
            # Also accept self.context_net, self.ar_context, etc.
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and (
                    "context" in func.attr.lower()
                    or "ar_predict" in func.attr.lower()
                    or "ar_decode" in func.attr.lower()
                )
            ):
                return True
    return False


def _walk_class_init_for_context_attr(class_node: ast.ClassDef) -> bool:
    """Return True if __init__ assigns self.context = SomeNet(...)."""
    for method in class_node.body:
        if isinstance(method, ast.FunctionDef) and method.name == "__init__":
            for node in ast.walk(method):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and (
                                target.attr == "context"
                                or "context" in target.attr.lower()
                                or "ar_predictor" in target.attr.lower()
                            )
                        ):
                            return True
    return False


def _scan_file_for_charm_violations(path: Path) -> list[tuple[str, str]]:
    """Return list of (class_name, reason) violations in a file."""
    violations: list[tuple[str, str]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return violations
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not CHARM_NAME_PATTERN.search(node.name):
                continue
            has_context_attr = _walk_class_init_for_context_attr(node)
            has_context_call = _walk_class_methods_for_context_call(node)
            if has_context_attr and not has_context_call:
                violations.append(
                    (
                        node.name,
                        (
                            f"class {node.name!r} declares self.context in __init__ "
                            "but never invokes it in any method — declared "
                            "channel-conditional but actually factorized "
                            "(codex_finding_charm_high_b)"
                        ),
                    )
                )
    return violations


def check_charm_class_actually_implements_channel_conditional(
    *,
    repo_root: Path | None = None,
    scan_paths: Iterable[Path] | None = None,
    strict: bool = True,
) -> dict:
    """STRICT preflight check: ChARM/AR classes must invoke their context net.

    Args:
        repo_root: project root (defaults to .)
        scan_paths: explicit list of .py files to scan; if None, sweep
            experiments/, src/tac/codec/, src/tac/optimization/, tools/.
        strict: if True, raise on any violation. If False, return result dict.

    Returns:
        {
            "files_scanned": int,
            "violations": list[{"path": str, "class": str, "reason": str}],
            "passed": bool,
        }

    Raises:
        CharmClassCheckError if strict and any violations found.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if scan_paths is None:
        roots = [
            repo_root / "experiments",
            repo_root / "src" / "tac" / "codec",
            repo_root / "src" / "tac" / "optimization",
            repo_root / "tools",
            repo_root / "submissions",
        ]
        scan_paths = []
        for root in roots:
            if root.is_dir():
                scan_paths.extend(root.rglob("*.py"))

    violations: list[dict] = []
    files_scanned = 0
    for p in scan_paths:
        if not p.is_file():
            continue
        # Skip vendored/intake clones
        s = str(p)
        if "_intake_" in s or "/vendor/" in s or "/__pycache__/" in s:
            continue
        files_scanned += 1
        for cls_name, reason in _scan_file_for_charm_violations(p):
            violations.append(
                {
                    "path": str(p.relative_to(repo_root)),
                    "class": cls_name,
                    "reason": reason,
                }
            )

    result = {
        "files_scanned": files_scanned,
        "violations": violations,
        "passed": len(violations) == 0,
    }
    if strict and violations:
        details = "\n".join(
            f"  {v['path']}::{v['class']} — {v['reason']}" for v in violations
        )
        raise CharmClassCheckError(
            "ChARM/AR-codec class declares context network but never calls it "
            f"({len(violations)} violation(s) — codex_finding_charm_high_b "
            f"meta-bug class):\n{details}"
        )
    return result


__all__ = [
    "CHARM_NAME_PATTERN",
    "CharmClassCheckError",
    "check_charm_class_actually_implements_channel_conditional",
]
