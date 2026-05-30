# SPDX-License-Identifier: MIT
"""Audit MLX primitive routing through canonical helper surfaces.

The audit is intentionally conservative: it does not ban direct MLX work, but
it makes direct substrate/tool use visible unless the file either routes through
one of the known canonical helper namespaces or carries an explicit
``MLX_PRIMITIVE_UNIQUE_BECAUSE_...`` waiver.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import ordered_unique
from tac.repo_io import repo_relative, sha256_file

SCHEMA = "mlx_canonicalization_audit.v1"
ROW_SCHEMA = "mlx_canonicalization_audit_row.v1"
TOOL = "tac.local_acceleration.mlx_canonicalization_audit"
WAIVER_TOKEN = "MLX_PRIMITIVE_UNIQUE_BECAUSE_"

CANONICAL_HELPER_IMPORT_PREFIXES = (
    "tac.local_acceleration.pr95_hnerv_mlx",
    "tac.local_acceleration.pr95_hnerv_mlx_training",
    "tac.local_acceleration.pr95_hnerv_mlx_long_training",
    "tac.local_acceleration.deterministic_primitives",
    "tac.portable_primitives",
    "tac.framework_agnostic",
    "tac.substrates._shared.mamba2_ssd",
    "tac.substrates._shared.mlx_score_aware",
)

CANONICAL_SOURCE_PREFIXES = (
    "src/tac/local_acceleration/",
    "src/tac/portable_primitives/",
    "src/tac/framework_agnostic/",
    "src/tac/substrates/_shared/mamba2_ssd/",
    "src/tac/substrates/_shared/mlx_score_aware/",
    "src/tac/tests/",
)

DEFAULT_SCAN_ROOTS = (
    "src/tac/substrates",
    "src/tac/composition",
    "src/tac/local_acceleration",
    "src/tac/portable_primitives",
    "tools",
)
DEFAULT_EXCLUDE_PATH_PARTS = (
    "/tests/",
    "/__pycache__/",
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


class MlxCanonicalizationAuditError(ValueError):
    """Raised when the audit cannot parse or classify source input."""


@dataclass(frozen=True)
class MlxFileSignal:
    """One source file's MLX canonicalization signal."""

    path: str
    sha256: str
    mlx_imports: tuple[str, ...]
    canonical_helper_imports: tuple[str, ...]
    waiver_lines: tuple[int, ...]
    primitive_tokens: tuple[str, ...]
    in_canonical_source_root: bool

    @property
    def uses_mlx(self) -> bool:
        return bool(self.mlx_imports)

    @property
    def routes_canonical_helper(self) -> bool:
        return bool(self.canonical_helper_imports)

    @property
    def has_unique_waiver(self) -> bool:
        return bool(self.waiver_lines)

    @property
    def requires_review(self) -> bool:
        return (
            self.uses_mlx
            and not self.in_canonical_source_root
            and not self.routes_canonical_helper
            and not self.has_unique_waiver
        )

    def to_row(self) -> dict[str, Any]:
        if self.requires_review:
            status = "needs_canonical_helper_or_unique_method_waiver"
        elif self.in_canonical_source_root:
            status = "canonical_source_root"
        elif self.routes_canonical_helper:
            status = "routes_canonical_helper"
        elif self.has_unique_waiver:
            status = "unique_method_waived"
        else:
            status = "no_mlx_signal"
        return {
            "schema": ROW_SCHEMA,
            "path": self.path,
            "sha256": self.sha256,
            "uses_mlx": self.uses_mlx,
            "status": status,
            "requires_review": self.requires_review,
            "mlx_imports": list(self.mlx_imports),
            "canonical_helper_imports": list(self.canonical_helper_imports),
            "waiver_lines": list(self.waiver_lines),
            "primitive_tokens": list(self.primitive_tokens),
            "in_canonical_source_root": self.in_canonical_source_root,
            **FALSE_AUTHORITY,
        }


def _repo_rel(path: Path, repo_root: Path) -> str:
    return repo_relative(path, repo_root).replace("\\", "/")


def _should_scan_path(
    path: Path,
    *,
    repo_root: Path,
    exclude_path_parts: Sequence[str],
) -> bool:
    rel = "/" + _repo_rel(path, repo_root)
    return not any(part in rel for part in exclude_path_parts)


def _iter_python_files(
    roots: Sequence[Path],
    *,
    repo_root: Path,
    exclude_path_parts: Sequence[str],
) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        absolute = root if root.is_absolute() else repo_root / root
        if absolute.is_file() and absolute.suffix == ".py":
            if _should_scan_path(
                absolute,
                repo_root=repo_root,
                exclude_path_parts=exclude_path_parts,
            ):
                files.append(absolute)
            continue
        if not absolute.exists():
            continue
        files.extend(
            path
            for path in absolute.rglob("*.py")
            if path.is_file()
            and _should_scan_path(
                path,
                repo_root=repo_root,
                exclude_path_parts=exclude_path_parts,
            )
        )
    return sorted(set(files), key=lambda path: _repo_rel(path, repo_root))


def _import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module
    return None


def _collect_imports(tree: ast.AST) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mlx_imports: list[str] = []
    canonical_imports: list[str] = []
    for node in ast.walk(tree):
        name = _import_name(node)
        if not name:
            continue
        imported_names = [part.strip() for part in name.split(",") if part.strip()]
        if any(item == "mlx" or item.startswith("mlx.") for item in imported_names):
            mlx_imports.extend(imported_names)
        for item in imported_names:
            if any(
                item == prefix or item.startswith(prefix + ".")
                for prefix in CANONICAL_HELPER_IMPORT_PREFIXES
            ):
                canonical_imports.append(item)
    return (
        tuple(ordered_unique(mlx_imports)),
        tuple(ordered_unique(canonical_imports)),
    )


def _waiver_lines(text: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, line in enumerate(text.splitlines(), start=1)
        if WAIVER_TOKEN in line
    )


def _primitive_tokens(text: str) -> tuple[str, ...]:
    token_map = {
        "mx.compile": "mx.compile",
        "mx.eval": "mx.eval",
        "mlx.nn.Conv": "mlx.nn.Conv",
        "mlx.nn.Conv2d": "mlx.nn.Conv2d",
        "mlx.nn.BatchNorm": "mlx.nn.BatchNorm",
        "mlx.nn.LayerNorm": "mlx.nn.LayerNorm",
        "mlx.nn.Linear": "mlx.nn.Linear",
        "pixel_shuffle": "pixel_shuffle",
        "bilinear": "bilinear_resize",
        "upsample": "upsample",
        "conv_transpose": "conv_transpose",
    }
    return tuple(
        name for needle, name in token_map.items() if needle in text
    )


def _in_canonical_source_root(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in CANONICAL_SOURCE_PREFIXES)


def scan_mlx_canonicalization_file(
    path: Path,
    *,
    repo_root: Path,
) -> MlxFileSignal | None:
    """Return a canonicalization signal for one Python source file."""

    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise MlxCanonicalizationAuditError(f"{path}: cannot parse Python") from exc
    mlx_imports, canonical_imports = _collect_imports(tree)
    rel = _repo_rel(path, repo_root)
    if not mlx_imports and WAIVER_TOKEN not in text:
        return None
    return MlxFileSignal(
        path=rel,
        sha256=sha256_file(path),
        mlx_imports=mlx_imports,
        canonical_helper_imports=canonical_imports,
        waiver_lines=_waiver_lines(text),
        primitive_tokens=_primitive_tokens(text),
        in_canonical_source_root=_in_canonical_source_root(rel),
    )


def build_mlx_canonicalization_audit(
    *,
    repo_root: str | Path,
    scan_roots: Sequence[str | Path] | None = None,
    exclude_path_parts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a false-authority MLX canonicalization audit payload."""

    root = Path(repo_root).resolve()
    roots = [Path(path) for path in (scan_roots or DEFAULT_SCAN_ROOTS)]
    excludes = DEFAULT_EXCLUDE_PATH_PARTS if exclude_path_parts is None else exclude_path_parts
    signals = [
        signal
        for path in _iter_python_files(
            roots,
            repo_root=root,
            exclude_path_parts=excludes,
        )
        if (signal := scan_mlx_canonicalization_file(path, repo_root=root)) is not None
    ]
    rows = [signal.to_row() for signal in signals]
    review_rows = [row for row in rows if row["requires_review"]]
    routed_rows = [
        row
        for row in rows
        if row["status"] in {"routes_canonical_helper", "canonical_source_root"}
    ]
    waived_rows = [row for row in rows if row["status"] == "unique_method_waived"]
    blockers = [
        f"mlx_canonicalization_review_required:{row['path']}"
        for row in review_rows
    ]
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "scan_roots": [str(path) for path in (scan_roots or DEFAULT_SCAN_ROOTS)],
        "exclude_path_parts": list(excludes),
        "canonical_helper_import_prefixes": list(CANONICAL_HELPER_IMPORT_PREFIXES),
        "canonical_source_prefixes": list(CANONICAL_SOURCE_PREFIXES),
        "waiver_token": WAIVER_TOKEN,
        "mlx_canonicalization_ready": not review_rows,
        "review_required_count": len(review_rows),
        "routed_or_canonical_count": len(routed_rows),
        "unique_waiver_count": len(waived_rows),
        "mlx_file_count": len(rows),
        "rows": rows,
        "review_required_rows": review_rows,
        "blockers": blockers,
        "allowed_use": "local_mlx_canonicalization_audit_only",
        "forbidden_use": "score_claim_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def summarize_mlx_canonicalization_audit(payload: Mapping[str, Any]) -> str:
    """Render a compact audit summary for operator surfaces."""

    lines = [
        "# MLX Canonicalization Audit",
        "",
        f"- Schema: `{payload.get('schema')}`",
        f"- Ready: `{payload.get('mlx_canonicalization_ready')}`",
        f"- MLX files: `{payload.get('mlx_file_count')}`",
        f"- Routed/canonical: `{payload.get('routed_or_canonical_count')}`",
        f"- Unique waivers: `{payload.get('unique_waiver_count')}`",
        f"- Review required: `{payload.get('review_required_count')}`",
        "",
        "## Review Required",
        "",
    ]
    for row in payload.get("review_required_rows", []):
        if not isinstance(row, Mapping):
            continue
        tokens = ", ".join(str(item) for item in row.get("primitive_tokens", []))
        lines.append(f"- `{row.get('path')}` ({tokens or 'mlx import'})")
    if not payload.get("review_required_rows"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CANONICAL_HELPER_IMPORT_PREFIXES",
    "CANONICAL_SOURCE_PREFIXES",
    "DEFAULT_EXCLUDE_PATH_PARTS",
    "DEFAULT_SCAN_ROOTS",
    "FALSE_AUTHORITY",
    "SCHEMA",
    "TOOL",
    "WAIVER_TOKEN",
    "MlxCanonicalizationAuditError",
    "MlxFileSignal",
    "build_mlx_canonicalization_audit",
    "scan_mlx_canonicalization_file",
    "summarize_mlx_canonicalization_audit",
]
