#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate that every `required_input_file=True` flag declared in a trainer's
``TIER_N_OPERATOR_REQUIRED_FLAGS`` manifest points to an EXISTING FILE PATH
BEFORE the GPU dispatch fires. This is the canonical wrapper-side validator
referenced by Catalog #152.

Bug-class anchor (the reason this tool exists):
    2026-05-12T17:12 Modal A100 dispatch (call_id fc-01KREJST89QHFRWJXHAKXD850C)
    burned $0.016 in 15 seconds before crashing with rc=1 because
    `--pr95-parity-profile` pointed to a non-existent file on the Modal worker.
    A local 100ms validation would have caught it before the GPU meter started.

Usage (from an operator-authorize dispatch wrapper, BEFORE the modal/vastai
launcher call):

    .venv/bin/python tools/validate_dispatch_required_inputs.py \\
        --trainer experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py \\
        || { echo "FATAL: required input missing; aborting before GPU dispatch"; exit 7; }

Exit codes:
    0 — every required-input-file flag resolves to an existing file
    1 — at least one required-input-file flag is missing or unresolvable
    2 — trainer file does not exist or has no parseable TIER_N manifest

Resolution order for each flag's value:
    1. Explicit `--flag-value FLAG=PATH` from the operator recipe
    2. The flag's `env` env-var (operator override)
    3. The flag's `default` field in the manifest

A flag whose env-var is unset AND whose default is `None` is FATAL — the
operator must explicitly set the env-var or the trainer is misconfigured.

This tool delegates to ``tac.preflight._check_151_extract_tier_manifests`` for
the trainer-manifest read so the validator and Catalog #152 stay in lock-step.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

_REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(_REPO_ROOT)

from tac.preflight import _check_151_extract_tier_manifests  # noqa: E402


def _resolve_flag_value(
    meta: dict,
    repo_root: Path,
    *,
    explicit_value: str | None = None,
) -> tuple[str | None, str]:
    """Return (resolved_path_str_or_None, source_description).

    source_description is one of: "env:<NAME>", "default", "unresolved".
    """
    if explicit_value:
        return explicit_value, "flag-value"
    env_var = meta.get("env") or ""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var], f"env:{env_var}"
    default = meta.get("default")
    if isinstance(default, str) and default:
        return default, "default"
    return None, "unresolved"


def _validate_one_flag(
    flag: str,
    meta: dict,
    repo_root: Path,
    *,
    explicit_value: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, message). `message` is operator-facing diagnostic text."""
    raw_value, source = _resolve_flag_value(
        meta,
        repo_root,
        explicit_value=explicit_value,
    )
    env_var = meta.get("env") or "<none>"
    rationale = (meta.get("rationale") or "").strip()
    generator = (meta.get("generator_command") or "").strip()
    if raw_value is None:
        msg = (
            f"FATAL: {flag} unresolved (env={env_var}; no default in manifest); "
            f"set {env_var} or update the trainer manifest"
        )
        if rationale:
            msg += f"\n  rationale: {rationale}"
        if generator:
            msg += f"\n  generate the file via: {generator}"
        return False, msg
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = (repo_root / raw_value).resolve()
    if not candidate.is_file():
        msg = (
            f"FATAL: {flag} → {candidate} does NOT exist "
            f"(source={source}; env_var={env_var}); "
            f"GPU dispatch would crash in ~15s on the remote worker"
        )
        if rationale:
            msg += f"\n  rationale: {rationale}"
        if generator:
            msg += f"\n  generate the file via: {generator}"
        return False, msg
    return True, f"OK: {flag} → {candidate} ({source})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate required-input-file flags exist before GPU dispatch. "
            "Canonical Catalog #152 wrapper-side validator."
        )
    )
    parser.add_argument(
        "--trainer",
        required=True,
        help=(
            "Trainer script path (e.g. "
            "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py). "
            "Relative paths resolve against the repo root."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (default: auto-detect from this tool's location)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print FATAL messages on stderr; suppress OK lines",
    )
    parser.add_argument(
        "--flag-value",
        action="append",
        default=[],
        metavar="FLAG=PATH",
        help=(
            "Explicit value for a required input flag, usually supplied from "
            "an operator-authorize recipe's required_input_files.default_path. "
            "May be repeated. Takes precedence over env/default."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT
    trainer_arg = Path(args.trainer)
    trainer_path = (
        trainer_arg if trainer_arg.is_absolute() else repo_root / trainer_arg
    )

    if not trainer_path.is_file():
        print(
            f"[validate-dispatch-required-inputs] FATAL: trainer file not found: "
            f"{trainer_path}",
            file=sys.stderr,
        )
        return 2

    manifest = _check_151_extract_tier_manifests(trainer_path)
    if not manifest:
        # No manifest declared → fail-open per CLAUDE.md "Subagent coherence-by-
        # default" (the trainer is opt-in to the manifest contract).
        if not args.quiet:
            print(
                f"[validate-dispatch-required-inputs] OK: {trainer_path.name} "
                f"declares no TIER_N_OPERATOR_REQUIRED_FLAGS manifest "
                f"(opt-in contract; nothing to validate)"
            )
        return 0

    required_input_flags = {
        flag: meta
        for flag, meta in manifest.items()
        if meta.get("required_input_file") is True
    }
    if not required_input_flags:
        if not args.quiet:
            print(
                f"[validate-dispatch-required-inputs] OK: {trainer_path.name} "
                f"declares {len(manifest)} tier-required flag(s) but none "
                f"carry `required_input_file=True` — nothing to validate"
            )
        return 0

    failures: list[str] = []
    ok_lines: list[str] = []
    explicit_values: dict[str, str] = {}
    for item in args.flag_value:
        if "=" not in item:
            print(
                f"[validate-dispatch-required-inputs] FATAL: --flag-value "
                f"must be FLAG=PATH, got {item!r}",
                file=sys.stderr,
            )
            return 2
        flag, value = item.split("=", 1)
        explicit_values[flag.strip()] = value.strip()
    for flag in sorted(required_input_flags):
        meta = required_input_flags[flag]
        ok, message = _validate_one_flag(
            flag,
            meta,
            repo_root,
            explicit_value=explicit_values.get(flag),
        )
        if ok:
            ok_lines.append(message)
        else:
            failures.append(message)

    if failures:
        print(
            f"[validate-dispatch-required-inputs] {len(failures)} of "
            f"{len(required_input_flags)} required-input-file flag(s) FAILED "
            f"validation for {trainer_path.name}; aborting before GPU dispatch:",
            file=sys.stderr,
        )
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"[validate-dispatch-required-inputs] OK: "
            f"{len(ok_lines)} required-input-file flag(s) validated for "
            f"{trainer_path.name}:"
        )
        for line in ok_lines:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
