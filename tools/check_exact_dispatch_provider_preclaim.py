#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed local provider preclaim gate for exact-eval dispatch queues."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

try:
    from tools.tool_bootstrap import repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
SCHEMA = "exact_dispatch_provider_preclaim_check.v1"
SUPPORTED_PROVIDERS = frozenset({"lightning", "modal"})
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an older local preclaim artifact for deterministic queue replays.",
    )
    return parser.parse_args(argv)


def build_preclaim_check(
    *,
    provider: str,
    job_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if provider not in SUPPORTED_PROVIDERS:
        return _payload(
            provider=provider,
            job_id=job_id,
            blockers=[f"unsupported_provider:{provider}"],
            env_status={},
        )
    source_env = os.environ if env is None else env
    if provider == "lightning":
        env_status, blockers = _lightning_preclaim(source_env)
        return _payload(
            provider=provider,
            job_id=job_id,
            blockers=blockers,
            env_status=env_status,
        )
    if provider == "modal":
        env_status, blockers = _modal_preclaim(source_env)
        return _payload(
            provider=provider,
            job_id=job_id,
            blockers=blockers,
            env_status=env_status,
        )
    return _payload(
        provider=provider,
        job_id=job_id,
        blockers=[f"unsupported_provider:{provider}"],
        env_status={},
    )


def _lightning_preclaim(env: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    keys = (
        "LIGHTNING_STUDIO",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_SDK_USER",
        "LIGHTNING_ORG",
        "LIGHTNING_SSH_TARGET",
    )
    env_status = {key: _presence(env.get(key, "")) for key in keys}
    blockers: list[str] = []
    if env_status["LIGHTNING_STUDIO"] != "present":
        blockers.append("lightning_studio_missing")
    if env_status["LIGHTNING_TEAMSPACE"] != "present":
        blockers.append("lightning_teamspace_missing")
    if env_status["LIGHTNING_SDK_USER"] != "present" and env_status["LIGHTNING_ORG"] != "present":
        blockers.append("lightning_owner_missing")
    if env_status["LIGHTNING_SSH_TARGET"] != "present":
        blockers.append("lightning_ssh_target_missing")
    return env_status, blockers


def _modal_preclaim(env: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    home = Path(env.get("HOME") or "").expanduser()
    modal_cli = REPO_ROOT / ".venv" / "bin" / "modal"
    config_path = Path(env.get("MODAL_CONFIG_PATH") or "") if env.get("MODAL_CONFIG_PATH") else None
    config_candidates = [
        path
        for path in (
            config_path,
            home / ".modal.toml" if str(home) else None,
            home / ".modal" / "config.toml" if str(home) else None,
        )
        if path is not None
    ]
    token_pair_present = (
        _presence(env.get("MODAL_TOKEN_ID", "")) == "present"
        and _presence(env.get("MODAL_TOKEN_SECRET", "")) == "present"
    )
    config_present = any(path.is_file() for path in config_candidates)
    env_status = {
        "MODAL_TOKEN_ID": _presence(env.get("MODAL_TOKEN_ID", "")),
        "MODAL_TOKEN_SECRET": _presence(env.get("MODAL_TOKEN_SECRET", "")),
        "MODAL_PROFILE": _presence(env.get("MODAL_PROFILE", "")),
        "MODAL_CONFIG_PATH": _presence(env.get("MODAL_CONFIG_PATH", "")),
        "modal_cli": "present"
        if modal_cli.is_file() or shutil.which("modal")
        else "missing",
        "modal_config": "present" if config_present else "missing",
    }
    blockers: list[str] = []
    if env_status["modal_cli"] != "present":
        blockers.append("modal_cli_missing")
    if not token_pair_present and not config_present:
        blockers.append("modal_auth_config_missing")
    return env_status, blockers


def _payload(
    *,
    provider: str,
    job_id: str,
    blockers: list[str],
    env_status: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "tool": "tools/check_exact_dispatch_provider_preclaim.py",
        "generated_at_utc": _utc_now(),
        "provider": provider,
        "job_id": job_id,
        "preclaim_ready": not blockers,
        "blockers": blockers,
        "env_status": dict(env_status),
        "redaction_policy": "presence_only_no_env_values_written",
        **FALSE_AUTHORITY,
    }


def _presence(value: str | None) -> str:
    return "present" if isinstance(value, str) and value.strip() else "missing"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
    output = path if path.is_absolute() else REPO_ROOT / path
    if output.exists() and not overwrite:
        raise SystemExit(f"refusing_to_overwrite_preclaim_check:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_preclaim_check(provider=args.provider, job_id=args.job_id)
    _write_json(args.output, payload, overwrite=bool(args.overwrite))
    if payload["blockers"]:
        print("exact dispatch provider preclaim blocked: " + ",".join(str(blocker) for blocker in payload["blockers"]))
        return 2
    print(f"exact dispatch provider preclaim ready: {args.provider}:{args.job_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
