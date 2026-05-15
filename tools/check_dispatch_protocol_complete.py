#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the dispatch_protocol_complete umbrella for one operator recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.deploy.dispatch_protocol import (  # noqa: E402
    evaluate_dispatch_protocol_complete,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SystemExit("PyYAML is required to parse operator recipes") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: recipe did not parse to a YAML mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="Recipe YAML path")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--trainer", default=None, help="Optional trainer path override")
    parser.add_argument(
        "--remote-driver",
        default=None,
        help="Optional remote driver path override",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on blockers")
    parser.add_argument("--json-out", default=None, help="Optional report JSON path")
    args = parser.parse_args(argv)

    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        recipe_path = Path(args.repo_root) / recipe_path
    recipe = _load_yaml(recipe_path)
    platform = str(recipe.get("platform") or "").strip().lower()
    native_dispatch = platform in {"modal", "vastai", "vast", "local"}
    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=args.repo_root,
        recipe_path=recipe_path,
        trainer_path=args.trainer,
        remote_driver_path=args.remote_driver,
        native_dispatch=native_dispatch,
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and not report.dispatch_protocol_complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
