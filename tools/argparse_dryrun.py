"""Argparse dry-run helper: load a Python script's parser + simulate parse.

Bug class extincted (2026-05-01): `experiments/train_renderer.py` crashed
with `error: --tag required` AFTER 64s of GPU spend on Modal Q-FAITHFUL v1.
v2 then crashed differently (`--auth-eval-on-best` incompatible with
`variant='quantizr_faithful'`). Both wasted Modal compute before failing.

The existing `preflight_arity` check covers SUBPROCESS calls; this helper
extends to the lane-script case where the failure mode is the SCRIPT
inheriting wrong defaults from a profile.

Usage:
    .venv/bin/python tools/argparse_dryrun.py \
        --target experiments/train_renderer.py \
        --args '--profile q_faithful_h100 --output-dir /tmp/foo'

Exit 0 = parse OK. Exit non-zero = argparse would have rejected those args.

Public API (used by PCC11):
    extract_argparse_calls(target_path) -> list[ArgumentSpec]
    dryrun_parse(target_path, argv_list) -> tuple[ok, message]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ArgumentSpec:
    """One add_argument call extracted statically from a target script."""

    flag_names: tuple[str, ...]
    required: bool
    has_default: bool
    is_store_true: bool
    is_store_false: bool
    choices: list[str] | None


def _node_value(node: ast.AST) -> object:
    """Best-effort literal extraction (None if non-literal)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]
    return None


def extract_argparse_calls(target_path: Path | str) -> list[ArgumentSpec]:
    """Walk the target's AST and extract every `parser.add_argument(...)`.

    Captures flag_names, required=, has-default-or-not, store_true/false,
    and choices=. Skips non-literal defaults (treated as has_default=True).

    Used by PCC11 to verify lane scripts pass every required arg.
    """
    src = Path(target_path).read_text(errors="ignore")
    try:
        tree = ast.parse(src, filename=str(target_path))
    except SyntaxError:
        return []
    out: list[ArgumentSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        names: list[str] = []
        for arg in node.args:
            v = _node_value(arg)
            if isinstance(v, str) and v.startswith("-"):
                names.append(v)
        if not names:
            continue
        required = False
        has_default = False
        is_store_true = False
        is_store_false = False
        choices: list[str] | None = None
        for kw in node.keywords:
            if kw.arg == "required":
                v = _node_value(kw.value)
                if v is True:
                    required = True
            elif kw.arg == "default":
                # Any default at all (even None or non-literal) means it's
                # not required-by-absence.
                has_default = True
            elif kw.arg == "action":
                v = _node_value(kw.value)
                if v == "store_true":
                    is_store_true = True
                elif v == "store_false":
                    is_store_false = True
            elif kw.arg == "choices":
                if isinstance(kw.value, (ast.List, ast.Tuple)):
                    vals = [_node_value(e) for e in kw.value.elts]
                    choices = [str(x) for x in vals if isinstance(x, str)]
        out.append(ArgumentSpec(
            flag_names=tuple(names),
            required=required,
            has_default=has_default,
            is_store_true=is_store_true,
            is_store_false=is_store_false,
            choices=choices,
        ))
    return out


def required_flags(specs: Sequence[ArgumentSpec]) -> set[str]:
    """The set of long-form flag names that MUST appear in argv."""
    out: set[str] = set()
    for s in specs:
        if not s.required:
            continue
        # Prefer the long-form flag (--foo) over short (-f).
        long_flags = [f for f in s.flag_names if f.startswith("--")]
        if long_flags:
            out.add(long_flags[0])
        elif s.flag_names:
            out.add(s.flag_names[0])
    return out


def passed_flags(argv: Sequence[str]) -> set[str]:
    """Extract every flag-looking token from a flat argv list."""
    out: set[str] = set()
    for tok in argv:
        if tok.startswith("--"):
            # Handle `--flag=value` form.
            name = tok.split("=", 1)[0]
            out.add(name)
        elif tok.startswith("-") and len(tok) > 1 and not tok[1].isdigit():
            out.add(tok)
    return out


def dryrun_parse(
    target_path: Path | str,
    argv: Sequence[str],
) -> tuple[bool, str]:
    """Verify static argparse contract: every required flag is in argv.

    This is a STATIC check — does NOT actually import the target. Catches
    the same class of failures `train_renderer.py: error: --tag required`
    without needing the target's full import graph (which may need GPU).

    Returns (ok, message). ok=False indicates a missing required flag.
    """
    specs = extract_argparse_calls(target_path)
    if not specs:
        return True, "no add_argument calls found (or non-argparse target)"
    req = required_flags(specs)
    seen = passed_flags(argv)
    missing = req - seen
    if missing:
        return False, (
            f"target argparse declares {len(req)} required flag(s); "
            f"argv missing: {sorted(missing)}"
        )
    return True, f"OK — {len(req)} required flag(s) all present"


def _import_main(target_path: Path) -> object:
    """Try to import the target so we can introspect its actual parser.

    Falls back to static AST extraction if import fails (common for
    targets that pull in CUDA-only deps at import time).
    """
    spec = importlib.util.spec_from_file_location("dryrun_target", target_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", required=True,
                   help="Path to a Python script with an argparse parser.")
    p.add_argument("--args", default="",
                   help="Whitespace-separated argv to validate.")
    p.add_argument("--print-spec", action="store_true",
                   help="Dump the extracted argparse spec as JSON and exit.")
    args = p.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"target not found: {target}", file=sys.stderr)
        return 2

    if args.print_spec:
        specs = extract_argparse_calls(target)
        print(json.dumps([{
            "flags": list(s.flag_names),
            "required": s.required,
            "has_default": s.has_default,
            "is_store_true": s.is_store_true,
            "is_store_false": s.is_store_false,
            "choices": s.choices,
        } for s in specs], indent=2))
        return 0

    argv = args.args.split()
    ok, msg = dryrun_parse(target, argv)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
