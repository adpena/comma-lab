# SPDX-License-Identifier: MIT
"""Optional Rust/Rayon bridge for pairset component-combo search.

The Python planner remains the semantic oracle. This bridge uses the native
``pairset-combo-planner`` binary only when it is already built or explicitly
provided by environment; callers must keep a Python fallback.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class PairsetComboRustBridgeError(RuntimeError):
    """Raised when an explicitly requested native combo planner cannot run."""


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def resolve_pairset_combo_planner_binary(
    binary_path: str | Path | None = None,
) -> tuple[Path | None, bool]:
    """Return ``(path, explicit)`` for an available native planner binary."""

    if binary_path:
        path = Path(binary_path)
        if not _is_executable(path):
            raise PairsetComboRustBridgeError(
                f"pairset-combo-planner binary is not executable: {path}"
            )
        return path, True

    env_path = os.environ.get("PACT_PAIRSET_COMBO_PLANNER_BIN")
    if env_path:
        path = Path(env_path)
        if not _is_executable(path):
            raise PairsetComboRustBridgeError(
                "PACT_PAIRSET_COMBO_PLANNER_BIN is not executable: "
                f"{path}"
            )
        return path, True

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "runtime-rs" / "target" / "release" / "pairset-combo-planner",
    ]
    if os.environ.get("PACT_PAIRSET_COMBO_PLANNER_ALLOW_DEBUG") == "1":
        candidates.append(
            repo_root / "runtime-rs" / "target" / "debug" / "pairset-combo-planner"
        )
    for path in candidates:
        if _is_executable(path):
            return path, False
    return None, False


def plan_pairset_component_combos_native(
    request: dict[str, Any],
    *,
    binary_path: str | Path | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any] | None:
    """Return native combo-search output, or ``None`` when no binary exists."""

    binary, explicit = resolve_pairset_combo_planner_binary(binary_path)
    if binary is None:
        return None
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(request, sort_keys=True),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        message = (
            f"pairset-combo-planner failed with returncode={proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
        if explicit:
            raise PairsetComboRustBridgeError(message)
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        if explicit:
            raise PairsetComboRustBridgeError(
                "pairset-combo-planner emitted invalid JSON"
            ) from exc
        return None
    if not isinstance(payload, dict):
        if explicit:
            raise PairsetComboRustBridgeError(
                "pairset-combo-planner JSON output must be an object"
            )
        return None
    return payload


__all__ = [
    "PairsetComboRustBridgeError",
    "plan_pairset_component_combos_native",
    "resolve_pairset_combo_planner_binary",
]
