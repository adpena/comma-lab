# SPDX-License-Identifier: MIT
"""Audit substrate trainers for canonical optimization-helper use."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

CANONICAL_TRAINER_SKELETON_MODULE = "tac.substrates._shared.trainer_skeleton"
CANONICAL_OPTIMIZATION_HELPER = "build_optimized_training_context"
DIRECT_OPTIMIZATION_MODULE = "tac.training_optimization"
DIRECT_OPTIMIZATION_HELPERS = frozenset(
    {
        "GTScorerCache",
        "autocast_aware_forward",
        "build_gt_scorer_cache",
        "compile_with_fallback",
    }
)
WAIVER_RE = re.compile(
    r"#\s*OPTIMIZATION_HELPERS_WAIVED:\s*(?!<reason>|reason|TODO|TBD\b)\S.{2,}",
    re.IGNORECASE,
)
TRAINER_GLOB = "train_substrate_*.py"


@dataclass(frozen=True)
class TrainerOptimizationAuditRow:
    """One trainer's optimization-helper contract verdict."""

    path: str
    trainer_id: str
    status: str
    canonical_helper_imported: bool
    canonical_helper_assigned_call: bool
    direct_helper_imported: bool
    direct_helper_called: bool
    waiver_present: bool
    accepted: bool
    reason: str

    def to_json_obj(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainerOptimizationAudit:
    """Aggregate trainer optimization-helper audit."""

    scanned_trainers: int
    accepted_trainers: int
    missing_trainers: int
    waived_trainers: int
    rows: tuple[TrainerOptimizationAuditRow, ...]

    @property
    def violations(self) -> tuple[str, ...]:
        return tuple(
            f"{row.path}: missing canonical optimization helper/direct helper/waiver "
            f"(reason={row.reason})"
            for row in self.rows
            if not row.accepted
        )

    def to_json_obj(self) -> dict[str, object]:
        return {
            "schema": "trainer_optimization_helper_audit_v1",
            "scanned_trainers": self.scanned_trainers,
            "accepted_trainers": self.accepted_trainers,
            "missing_trainers": self.missing_trainers,
            "waived_trainers": self.waived_trainers,
            "rows": [row.to_json_obj() for row in self.rows],
            "violations": list(self.violations),
        }


def _comment_waiver_present(text: str) -> bool:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return any(tok.type == tokenize.COMMENT and WAIVER_RE.search(tok.string) for tok in tokens)
    except tokenize.TokenError:
        return False


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _assigned_call_to_alias(tree: ast.AST, aliases: set[str]) -> bool:
    if not aliases:
        return False
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
        if not isinstance(value, ast.Call):
            continue
        if _call_name(value.func) in aliases:
            return True
    return False


def _calls_any_helper(tree: ast.AST, aliases: set[str], module_aliases: set[str]) -> bool:
    if not aliases and not module_aliases:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in aliases:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr in DIRECT_OPTIMIZATION_HELPERS
            and isinstance(func.value, ast.Name)
            and func.value.id in module_aliases
        ):
            return True
    return False


def audit_trainer_file(path: Path, repo_root: Path) -> TrainerOptimizationAuditRow:
    """Audit one ``experiments/train_substrate_*.py`` file."""

    rel = path.relative_to(repo_root).as_posix()
    trainer_id = path.stem.removeprefix("train_substrate_")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return TrainerOptimizationAuditRow(
            path=rel,
            trainer_id=trainer_id,
            status="read_error",
            canonical_helper_imported=False,
            canonical_helper_assigned_call=False,
            direct_helper_imported=False,
            direct_helper_called=False,
            waiver_present=False,
            accepted=False,
            reason=f"read_error:{exc}",
        )
    waiver_present = _comment_waiver_present(text)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return TrainerOptimizationAuditRow(
            path=rel,
            trainer_id=trainer_id,
            status="syntax_error",
            canonical_helper_imported=False,
            canonical_helper_assigned_call=False,
            direct_helper_imported=False,
            direct_helper_called=False,
            waiver_present=waiver_present,
            accepted=waiver_present,
            reason=f"syntax_error:{exc.msg}",
        )

    canonical_aliases: set[str] = set()
    direct_aliases: set[str] = set()
    direct_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == CANONICAL_TRAINER_SKELETON_MODULE:
                for alias in node.names:
                    if alias.name == CANONICAL_OPTIMIZATION_HELPER:
                        canonical_aliases.add(alias.asname or alias.name)
            if module == DIRECT_OPTIMIZATION_MODULE:
                for alias in node.names:
                    if alias.name in DIRECT_OPTIMIZATION_HELPERS:
                        direct_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == DIRECT_OPTIMIZATION_MODULE:
                    direct_module_aliases.add(alias.asname or alias.name.split(".")[-1])

    canonical_assigned_call = _assigned_call_to_alias(tree, canonical_aliases)
    direct_called = _calls_any_helper(tree, direct_aliases, direct_module_aliases)
    accepted = canonical_assigned_call or direct_called or waiver_present
    if canonical_assigned_call:
        status = "canonical_helper"
        reason = "canonical_build_optimized_training_context_assigned_call"
    elif direct_called:
        status = "direct_helper"
        reason = "direct_training_optimization_helper_called"
    elif waiver_present:
        status = "waived"
        reason = "comment_waiver_present"
    else:
        status = "missing"
        reason = "no_ast_import_plus_call_contract"

    return TrainerOptimizationAuditRow(
        path=rel,
        trainer_id=trainer_id,
        status=status,
        canonical_helper_imported=bool(canonical_aliases),
        canonical_helper_assigned_call=canonical_assigned_call,
        direct_helper_imported=bool(direct_aliases or direct_module_aliases),
        direct_helper_called=direct_called,
        waiver_present=waiver_present,
        accepted=accepted,
        reason=reason,
    )


def iter_trainer_files(repo_root: Path) -> Iterable[Path]:
    experiments_dir = repo_root / "experiments"
    if not experiments_dir.is_dir():
        return ()
    return sorted(path for path in experiments_dir.glob(TRAINER_GLOB) if path.is_file())


def audit_trainer_optimization_helpers(
    repo_root: str | Path,
) -> TrainerOptimizationAudit:
    """Audit every immediate substrate trainer under ``experiments/``."""

    root = Path(repo_root).resolve()
    rows = tuple(audit_trainer_file(path, root) for path in iter_trainer_files(root))
    accepted = sum(1 for row in rows if row.accepted)
    waived = sum(1 for row in rows if row.waiver_present)
    return TrainerOptimizationAudit(
        scanned_trainers=len(rows),
        accepted_trainers=accepted,
        missing_trainers=len(rows) - accepted,
        waived_trainers=waived,
        rows=rows,
    )


__all__ = [
    "CANONICAL_OPTIMIZATION_HELPER",
    "CANONICAL_TRAINER_SKELETON_MODULE",
    "DIRECT_OPTIMIZATION_HELPERS",
    "DIRECT_OPTIMIZATION_MODULE",
    "TrainerOptimizationAudit",
    "TrainerOptimizationAuditRow",
    "audit_trainer_file",
    "audit_trainer_optimization_helpers",
    "iter_trainer_files",
]
