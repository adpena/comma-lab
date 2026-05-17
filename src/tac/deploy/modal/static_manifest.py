# SPDX-License-Identifier: MIT
"""Static Modal mount-manifest coverage checks.

Specialized Modal dispatchers sometimes intentionally use a fixed custody
manifest instead of ``build_training_image``. That is valid only when the
static manifest stays mechanically in sync with trainer-declared input
metadata.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _normalize_rel(path: str | Path) -> str:
    return Path(str(path).strip()).as_posix().lstrip("./")


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _bool_constant(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _dict_entries(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        return {}
    out: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=False):
        if key is None:
            continue
        key_text = _string_constant(key)
        if key_text is not None:
            out[key_text] = value
    return out


def _assignment_targets_name(node: ast.AST, names: set[str]) -> str | None:
    targets: list[ast.AST]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return None
    for target in targets:
        if isinstance(target, ast.Name) and target.id in names:
            return target.id
    return None


def _collect_static_trainer_paths(trainer_path: Path) -> list[tuple[str, str]]:
    """Return ``[(label, rel_path), ...]`` from statically-resolved metadata."""

    tree = ast.parse(trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path))
    paths: list[tuple[str, str]] = []
    unresolved: list[str] = []
    for node in ast.walk(tree):
        attr_name = _assignment_targets_name(
            node,
            {
                "TIER_1_EXTRA_MOUNT_PATHS",
                "MODAL_EXTRA_MOUNT_PATHS",
            },
        )
        if attr_name is not None and isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)):
                for elt in value.elts:
                    text = _string_constant(elt)
                    if text is None:
                        unresolved.append(f"{attr_name}@line{node.lineno}")
                        continue
                    paths.append((attr_name, text))
            elif value is not None:
                unresolved.append(f"{attr_name}@line{node.lineno}")
            continue

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target_name = _assignment_targets_name(
            node,
            {
                target.id
                for target in ast.walk(node)
                if isinstance(target, ast.Name)
                and target.id.startswith("TIER_")
                and target.id.endswith("_OPERATOR_REQUIRED_FLAGS")
            },
        )
        if target_name is None:
            continue
        manifest = _dict_entries(node.value)
        for flag, meta_node in manifest.items():
            meta = _dict_entries(meta_node)
            required = meta.get("required_input_file")
            if required is None or _bool_constant(required) is not True:
                continue
            default_node = meta.get("default")
            default = _string_constant(default_node) if default_node is not None else None
            if default is None:
                unresolved.append(f"{flag}.default@line{getattr(meta_node, 'lineno', node.lineno)}")
                continue
            paths.append((f"{flag} required_input_file", default))
    for item in unresolved:
        paths.append((f"unresolved_metadata:{item}", ""))
    return paths


def _manifest_specs(manifest: Any) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for spec in manifest:
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("kind") or "")
        local_path = spec.get("local_path")
        if kind not in {"dir", "file"} or not isinstance(local_path, str):
            continue
        specs.append((kind, _normalize_rel(local_path)))
    return specs


def _covered_by_manifest(path: str, specs: list[tuple[str, str]]) -> bool:
    rel = _normalize_rel(path)
    if not rel:
        return False
    for kind, mounted in specs:
        if kind == "file" and rel == mounted:
            return True
        if kind == "dir" and (rel == mounted or rel.startswith(f"{mounted}/")):
            return True
    return False


def validate_static_manifest_covers_trainer_metadata(
    manifest: Any,
    *,
    trainer_path: str | Path,
    repo_root: str | Path,
) -> list[str]:
    """Return coverage violations for a static Modal mount manifest.

    The check is intentionally pure source-level: it does not import the
    trainer, so dispatchers can run it at module import time without triggering
    trainer side effects or requiring CUDA/scorer dependencies.
    """

    root = Path(repo_root).resolve()
    trainer = Path(trainer_path)
    if not trainer.is_absolute():
        trainer = root / trainer
    specs = _manifest_specs(manifest)
    violations: list[str] = []
    for label, rel_path in _collect_static_trainer_paths(trainer):
        if not rel_path:
            violations.append(
                f"{trainer.relative_to(root)} has {label}; static Modal manifest "
                "coverage cannot be proven from source. Use a literal path or "
                "add a dispatcher-specific verifier."
            )
            continue
        if _covered_by_manifest(rel_path, specs):
            continue
        violations.append(
            f"{trainer.relative_to(root)} declares {label} path {rel_path!r}, "
            "but the static Modal manifest does not mount that file or a parent "
            "directory."
        )
    return violations


__all__ = ["validate_static_manifest_covers_trainer_metadata"]
