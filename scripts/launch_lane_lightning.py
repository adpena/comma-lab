#!/usr/bin/env python3
"""launch_lane_lightning.py — dispatch a lane on Lightning AI Studio.

Wraps `tac.deploy.lightning.LightningDispatcher` to make Lightning Studio
dispatches one-command analogous to `launch_lane_with_retry.py` (Vast.ai)
and `experiments/modal_train_lane.py` (Modal).

Lightning is PERSISTENT — there's no spin-up cost per lane. The Studio
must already be running and the GPU tier set in the Lightning UI
(this script does NOT change Studio settings; only dispatches into the
already-attached GPU).

Usage:
    .venv/bin/python scripts/launch_lane_lightning.py \\
        --lane-script scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh \\
        --label lane_17_imp_lightning_2026-04-30 \\
        --gpu-tier H100 \\
        --predicted-band 0.85 1.00 \\
        --estimated-cost 87.50 \\
        --env IMP_QUICK_VARIANT=0 \\
        --env AUTH_EVAL_DEVICE=cuda

Status / harvest commands (run with same --label):
    .venv/bin/python scripts/launch_lane_lightning.py status --session-id <id>
    .venv/bin/python scripts/launch_lane_lightning.py harvest --session-id <id> \\
        --local-dir experiments/results/lane_17_imp_lightning_2026-04-30
    .venv/bin/python scripts/launch_lane_lightning.py teardown --session-id <id>

Cross-references:
- feedback_lightning_ai_ssh_credentials_20260430.md
- src/tac/deploy/lightning/lightning_dispatch.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.deploy.lightning import LightningDispatcher  # noqa: E402


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in items or []:
        if "=" not in s:
            raise SystemExit(f"--env requires KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _make_dispatcher(args) -> LightningDispatcher:
    if args.ssh_user or args.ssh_host:
        if not (args.ssh_user and args.ssh_host):
            raise SystemExit("--ssh-user and --ssh-host must be provided together; prefer --ssh-target alias")
        return LightningDispatcher(
            ssh_user=args.ssh_user,
            ssh_host=args.ssh_host,
            remote_workspace=args.remote_workspace,
            ssh_key=args.ssh_key,
        )
    return LightningDispatcher(
        ssh_target=args.ssh_target,
        remote_workspace=args.remote_workspace,
        ssh_key=args.ssh_key,
    )


def cmd_dispatch(args) -> int:
    if not args.lane_script:
        print("error: --lane-script is required for dispatch", file=sys.stderr)
        return 2
    if not args.label:
        print("error: --label is required for dispatch", file=sys.stderr)
        return 2

    # Predicted-band + cost are recorded for the durable lane state so the
    # operator (or downstream agent) can audit dispatched lanes after the
    # fact (CLAUDE.md "every dispatch must document predicted band + kill
    # criteria BEFORE launch").
    if args.predicted_band is None or len(args.predicted_band) != 2:
        print(
            "error: --predicted-band <low> <high> is REQUIRED per CLAUDE.md "
            "non-negotiable",
            file=sys.stderr,
        )
        return 2

    env_overrides = _parse_env_kv(args.env)
    dispatcher = _make_dispatcher(args)

    try:
        info = dispatcher.dispatch_lane(
            lane_script=args.lane_script,
            label=args.label,
            gpu_tier_required=args.gpu_tier,
            env_overrides=env_overrides,
            allow_gpu_mismatch=args.allow_gpu_mismatch,
        )
    except RuntimeError as exc:
        print(f"DISPATCH_FAILED: {exc}", file=sys.stderr)
        return 1

    # Augment the on-disk session record with predicted-band + cost.
    sessions = dispatcher.list_sessions()
    for s in sessions:
        if s.get("session_id") == info.session_id:
            s["predicted_band"] = list(args.predicted_band)
            s["estimated_cost_usd"] = float(args.estimated_cost or 0.0)
            s["kill_criteria"] = args.kill_criteria
    dispatcher._save_state(sessions)  # noqa: SLF001 (private but intentional)

    print(json.dumps(
        {
            "status": "DISPATCHED",
            "session_id": info.session_id,
            "label": info.label,
            "lane_script": info.lane_script,
            "remote_log_path": info.remote_log_path,
            "remote_workspace": info.remote_workspace,
            "started_at_utc": info.started_at_utc,
            "gpu_tier_observed": info.gpu_tier_observed,
            "env_overrides": info.env_overrides,
            "predicted_band": list(args.predicted_band),
            "estimated_cost_usd": float(args.estimated_cost or 0.0),
        },
        indent=2,
    ))
    return 0


def cmd_status(args) -> int:
    if not args.session_id:
        print("error: --session-id is required for status", file=sys.stderr)
        return 2
    dispatcher = _make_dispatcher(args)
    info = dispatcher.poll_status(args.session_id)
    print(json.dumps(info, indent=2))
    return 0


def cmd_harvest(args) -> int:
    if not args.session_id:
        print("error: --session-id is required for harvest", file=sys.stderr)
        return 2
    if not args.local_dir:
        print("error: --local-dir is required for harvest", file=sys.stderr)
        return 2
    dispatcher = _make_dispatcher(args)
    out = dispatcher.harvest(
        args.session_id,
        local_dir=args.local_dir,
        remote_subdir=args.remote_subdir,
    )
    print(json.dumps(out, indent=2))
    return 0


def cmd_teardown(args) -> int:
    if not args.session_id:
        print("error: --session-id is required for teardown", file=sys.stderr)
        return 2
    dispatcher = _make_dispatcher(args)
    killed = dispatcher.tear_down(args.session_id)
    print(json.dumps({"session_id": args.session_id, "killed": killed}))
    return 0


def cmd_list(args) -> int:
    sessions = LightningDispatcher.list_sessions()
    print(json.dumps(sessions, indent=2))
    return 0


def cmd_probe(args) -> int:
    """Quick GPU probe — just SSH in and read nvidia-smi."""
    dispatcher = _make_dispatcher(args)
    name = dispatcher.get_gpu_tier()
    tier = dispatcher._gpu_tier_normalize(name)  # noqa: SLF001
    print(json.dumps({"gpu_name_raw": name, "gpu_tier": tier}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lightning AI Studio lane dispatcher",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common_ssh = argparse.ArgumentParser(add_help=False)
    common_ssh.add_argument(
        "--ssh-target",
        default=os.environ.get("LIGHTNING_SSH_TARGET", "lightning-pact"),
        help="SSH config alias for the Lightning Studio.",
    )
    common_ssh.add_argument(
        "--ssh-user",
        default=os.environ.get("LIGHTNING_USER"),
        help="Deprecated direct Studio SSH user; prefer --ssh-target alias.",
    )
    common_ssh.add_argument(
        "--ssh-host",
        default=os.environ.get("LIGHTNING_HOST"),
        help="Deprecated direct Studio SSH host; prefer --ssh-target alias.",
    )
    common_ssh.add_argument(
        "--remote-workspace",
        default="/teamspace/studios/this_studio/pact",
    )
    common_ssh.add_argument(
        "--ssh-key",
        default=None,
        help="Optional path to SSH identity file (default: rely on ssh agent).",
    )

    p_dispatch = sub.add_parser("dispatch", parents=[common_ssh])
    p_dispatch.add_argument("--lane-script", required=True)
    p_dispatch.add_argument("--label", required=True)
    p_dispatch.add_argument(
        "--gpu-tier",
        default=None,
        help="Required GPU tier (H100/A100/L40S/T4). If set, dispatch fails when "
             "Studio's attached GPU doesn't match.",
    )
    p_dispatch.add_argument(
        "--allow-gpu-mismatch",
        action="store_true",
        help="Allow dispatch even if GPU tier doesn't match.",
    )
    p_dispatch.add_argument(
        "--predicted-band",
        nargs=2,
        type=float,
        required=True,
        metavar=("LOW", "HIGH"),
        help="Predicted score band [low, high] — required by CLAUDE.md.",
    )
    p_dispatch.add_argument(
        "--estimated-cost",
        type=float,
        default=0.0,
        help="Estimated cost in USD (Lightning credits).",
    )
    p_dispatch.add_argument(
        "--kill-criteria",
        default="",
        help="Free-form description of kill criteria (recorded for audit).",
    )
    p_dispatch.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE env overrides exported before lane script (repeatable).",
    )
    p_dispatch.set_defaults(func=cmd_dispatch)

    p_status = sub.add_parser("status", parents=[common_ssh])
    p_status.add_argument("--session-id", required=True)
    p_status.set_defaults(func=cmd_status)

    p_harvest = sub.add_parser("harvest", parents=[common_ssh])
    p_harvest.add_argument("--session-id", required=True)
    p_harvest.add_argument("--local-dir", required=True)
    p_harvest.add_argument("--remote-subdir", default=None)
    p_harvest.set_defaults(func=cmd_harvest)

    p_teardown = sub.add_parser("teardown", parents=[common_ssh])
    p_teardown.add_argument("--session-id", required=True)
    p_teardown.set_defaults(func=cmd_teardown)

    p_list = sub.add_parser("list", parents=[common_ssh])
    p_list.set_defaults(func=cmd_list)

    p_probe = sub.add_parser("probe", parents=[common_ssh])
    p_probe.set_defaults(func=cmd_probe)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
