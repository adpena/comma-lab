#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Claim, stage, and dispatch the PR106 yshift score-table lane on Lightning.

This is the canonical launcher for
``scripts/remote_lane_pr106_yshift_sidechannel.sh`` with
``PR106_YSHIFT_MODE=score_table``.

The important custody detail is order:

1. verify Lightning SSH is reachable;
2. record the local lane claim with a stable ``instance/job_id``;
3. stage the workspace, including the freshly updated claim ledger;
4. dispatch the remote script with the same job id in the environment.

The score-table producer refuses CUDA scoring unless the remote copy of
``.omx/state/active_lane_dispatch_claims.md`` contains that active claim.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    LightningBatchJobsClient,
    LightningBatchJobSpec,
)
from tac.deploy.lightning.defaults import (  # noqa: E402
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_ssh_target,
    default_studio,
    default_teamspace,
    default_user,
)
from tac.deploy.claims import dispatch_claim_command  # noqa: E402
from tac.deploy.pr106_yshift import (  # noqa: E402
    REMOTE_SCRIPT,
    SCORE_TABLE_LANE_ID,
    SCORE_TABLE_ROLE,
    Pr106YshiftScoreTableSpec,
    dispatch_claim_spec,
    score_table_env as canonical_score_table_env,
)

DEFAULT_REMOTE_PACT = DEFAULT_LIGHTNING_REMOTE_PACT
DEFAULT_PR106_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"


def default_job_name() -> str:
    from tac.deploy.claims import utc_now

    return f"lane_pr106_yshift_score_table_{utc_now().replace('-', '').replace(':', '')}"


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"FATAL: path must be inside repo root: {path}") from exc


def _score_table_spec(args: argparse.Namespace) -> Pr106YshiftScoreTableSpec:
    return Pr106YshiftScoreTableSpec(
        job_name=args.job_name,
        pr106_archive=_repo_rel(args.pr106_archive),
        candidate_radius=args.candidate_radius,
        score_step=args.score_step,
        n_pairs=args.n_pairs,
        batch_pairs=args.batch_pairs,
        candidate_batch_size=args.candidate_batch_size,
    )


def build_ssh_check_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/lightning_repro_workspace.py"),
        "--remote",
        args.ssh_target,
        "--remote-pact",
        args.remote_pact,
        "--requirements-mode",
        "verify-only",
        "--python-bin",
        args.python_bin,
        "--ssh-check-only",
        "--ssh-connect-timeout",
        str(args.ssh_connect_timeout),
    ]
    if args.require_studio_cuda:
        cmd.append("--require-cuda")
    return cmd


def build_claim_command(args: argparse.Namespace, *, status: str, notes: str) -> list[str]:
    spec = dispatch_claim_spec(
        _score_table_spec(args),
        platform="lightning",
        agent=args.agent,
        predicted_eta_hours=args.predicted_eta_hours,
        force=args.force_claim,
        notes=notes,
    )
    return dispatch_claim_command(
        spec=spec,
        status=status,
        python_executable=sys.executable,
        claim_tool=REPO_ROOT / "tools/claim_lane_dispatch.py",
    )


def build_stage_command(args: argparse.Namespace) -> list[str]:
    manifest_out = (
        REPO_ROOT
        / "experiments/results/lightning_batch"
        / args.job_name
        / "source_manifest.json"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/lightning_repro_workspace.py"),
        "--remote",
        args.ssh_target,
        "--remote-pact",
        args.remote_pact,
        "--run-id",
        args.job_name,
        "--manifest-out",
        str(manifest_out),
        "--source",
        "AGENTS.md",
        "--source",
        "README.md",
        "--source",
        "pyproject.toml",
        "--source",
        "uv.lock",
        "--source",
        "src",
        "--source",
        "experiments",
        "--source",
        "scripts",
        "--source",
        "submissions",
        "--source",
        "upstream",
        "--source",
        "tools",
        "--artifact",
        _repo_rel(args.pr106_archive),
        "--artifact",
        _repo_rel(CLAIMS_PATH),
        "--requirements-mode",
        "verify-only",
        "--python-bin",
        args.python_bin,
        "--ssh-connect-timeout",
        str(args.ssh_connect_timeout),
    ]
    if args.require_studio_cuda:
        cmd.append("--require-cuda")
    return cmd


def score_table_env(
    args: argparse.Namespace,
    *,
    output_dir: str | None = None,
    include_log_dir: bool = True,
) -> dict[str, str]:
    if output_dir is None:
        output_dir = f"{args.remote_pact}/experiments/results/lightning_batch/{args.job_name}"
    return canonical_score_table_env(
        _score_table_spec(args),
        output_dir=output_dir,
        include_log_dir=include_log_dir,
    )


def build_dispatch_command(args: argparse.Namespace) -> list[str]:
    env = score_table_env(args)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/launch_lane_lightning.py"),
        "dispatch",
        "--lane-script",
        REMOTE_SCRIPT,
        "--label",
        args.job_name,
        "--ssh-target",
        args.ssh_target,
        "--remote-workspace",
        args.remote_pact,
        "--predicted-band",
        str(args.predicted_low),
        str(args.predicted_high),
        "--estimated-cost",
        str(args.estimated_cost),
        "--kill-criteria",
        args.kill_criteria,
    ]
    if args.gpu_tier:
        cmd.extend(["--gpu-tier", args.gpu_tier])
    if args.allow_gpu_mismatch:
        cmd.append("--allow-gpu-mismatch")
    for key in sorted(env):
        cmd.extend(["--env", f"{key}={env[key]}"])
    return cmd


def build_batch_command(args: argparse.Namespace) -> str:
    env_exports = [
        f"export PYBIN={args.python_bin}",
        "export PYTHONUNBUFFERED=1",
    ]
    for key, value in sorted(score_table_env(args, include_log_dir=False).items()):
        env_exports.append(f"export {key}={value}")
    return "\n".join(
        [
            "set -euo pipefail",
            f"if [ -d pact ] && [ -f pact/pyproject.toml ]; then cd pact; elif [ -d {args.remote_pact} ]; then cd {args.remote_pact}; else echo FATAL: pact repo not found >&2; exit 2; fi",
            'export OUT_DIR="$PWD/experiments/results/lightning_batch/' + args.job_name + '"',
            'mkdir -p "$OUT_DIR"',
            'export WORKSPACE="$PWD"',
            'export TAC_UPSTREAM_DIR="$PWD/upstream"',
            'export PYTHONPATH="$PWD/src:$PWD/upstream:$PWD"',
            'export PR106_YSHIFT_LOG_DIR="$OUT_DIR/yshift_run"',
            *env_exports,
            (
                f"{args.python_bin} - <<'PY' > \"$OUT_DIR/lightning_runner_preflight.json\"\n"
                "import subprocess\n"
                "import torch\n"
                "from tac.repo_io import json_text\n"
                "gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], "
                "text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)\n"
                "payload = {\n"
                "  'cuda_available': bool(torch.cuda.is_available()),\n"
                "  'device_count': int(torch.cuda.device_count()),\n"
                "  'torch_version': torch.__version__,\n"
                "  'torch_cuda': getattr(torch.version, 'cuda', None),\n"
                "  'nvidia_smi_returncode': gpu.returncode,\n"
                "  'gpu_names': gpu.stdout.strip().splitlines(),\n"
                "}\n"
                "if not payload['cuda_available']:\n"
                "    raise SystemExit(json_text({'LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK': False, **payload}))\n"
                "print(json_text({'LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK': True, **payload}), end='')\n"
                "PY"
            ),
            f"bash {REMOTE_SCRIPT} 2>&1 | tee \"$OUT_DIR/batch_run.log\"",
            (
                f"{args.python_bin} - <<'PY'\n"
                "import os, pathlib, shutil\n"
                "from tac.repo_io import write_json\n"
                "out = pathlib.Path(os.environ['OUT_DIR'])\n"
                "run = out / 'yshift_run'\n"
                "score_json = run / 'eval' / 'contest_auth_eval.json'\n"
                "summary = {\n"
                "  'score_claim': False,\n"
                "  'promotion_requires_adjudication': True,\n"
                "  'yshift_run_dir': str(run),\n"
                "  'contest_auth_eval_json': str(score_json),\n"
                "  'contest_auth_eval_json_exists': score_json.is_file(),\n"
                "}\n"
                "if score_json.is_file():\n"
                "    shutil.copy2(score_json, out / 'contest_auth_eval.json')\n"
                "    summary['copied_contest_auth_eval_json'] = True\n"
                "write_json(out / 'pr106_yshift_score_table_batch_summary.json', summary)\n"
                "if not score_json.is_file():\n"
                "    raise SystemExit('FATAL: yshift score-table batch did not produce contest_auth_eval.json')\n"
                "PY"
            ),
        ]
    )


def build_batch_spec(args: argparse.Namespace) -> LightningBatchJobSpec:
    output_dir = f"{args.remote_pact}/experiments/results/lightning_batch/{args.job_name}"
    return LightningBatchJobSpec(
        name=args.job_name,
        machine=args.machine,
        command=build_batch_command(args),
        studio=args.studio or None,
        teamspace=args.teamspace or None,
        user=args.user or None,
        cloud_account=args.cloud_account or None,
        max_runtime=args.max_runtime_seconds,
        reuse_snapshot=False,
        role=SCORE_TABLE_ROLE,
        local_artifact_dir=f"experiments/results/lightning_batch/{args.job_name}",
        remote_output_dir=output_dir,
        queue_metadata={
            "lane": SCORE_TABLE_LANE_ID,
            "pr106_archive": _repo_rel(args.pr106_archive),
            "mode": "score_table",
            "score_claim": "false",
            "promotion_gate": "requires contest_auth_eval_json adjudication",
        },
    )


def _print_command(title: str, cmd: list[str]) -> None:
    print(f"=== {title} ===")
    print(" \\\n+  ".join(cmd))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _terminal_claim(args: argparse.Namespace, *, status: str, notes: str) -> None:
    cmd = build_claim_command(args, status=status, notes=notes)
    if "--force" not in cmd:
        cmd.append("--force")
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _needs_ssh_target(args: argparse.Namespace) -> bool:
    return not (
        args.print_only
        and args.backend == "batch"
        and args.skip_ssh_check
        and args.skip_stage
    )


def _validate_dispatch_args(args: argparse.Namespace) -> None:
    if args.skip_stage and not args.print_only:
        raise SystemExit(
            "FATAL: real dispatch with --skip-stage is refused by default. "
            "Generate a pre-staged Batch command with --print-only, or stage "
            "the current claim ledger through this launcher."
        )
    if _needs_ssh_target(args) and not str(args.ssh_target or "").strip():
        raise SystemExit(
            "FATAL: --ssh-target or LIGHTNING_SSH_TARGET is required before "
            "claim/stage dispatch. For pre-staged Lightning Batch-only command "
            "generation, pass --backend batch --skip-ssh-check --skip-stage."
        )


def _submit_batch(args: argparse.Namespace) -> int:
    spec = build_batch_spec(args)
    if args.print_only:
        print("=== batch spec ===")
        print(spec.asdict())
        return 0
    try:
        record = LightningBatchJobsClient().submit(spec, dry_run=args.dry_run_batch)
    except Exception:
        _terminal_claim(args, status="failed_batch_submit", notes="Lightning Batch Job submit failed")
        raise
    if args.dry_run_batch:
        _terminal_claim(args, status="completed_dry_run", notes="Lightning Batch dry-run only; no CUDA work dispatched")
    print(record)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    args.pr106_archive = args.pr106_archive.resolve()
    if not args.pr106_archive.is_file():
        raise SystemExit(f"FATAL: PR106 archive not found: {args.pr106_archive}")
    if not CLAIMS_PATH.is_file():
        raise SystemExit(f"FATAL: dispatch claim ledger not found: {CLAIMS_PATH}")
    _validate_dispatch_args(args)

    ssh_cmd = build_ssh_check_command(args)
    claim_cmd = build_claim_command(
        args,
        status="active_dispatching",
        notes="PR106 yshift CUDA score-table producer plus charged archive exact eval",
    )
    stage_cmd = build_stage_command(args)
    dispatch_cmd = build_dispatch_command(args)

    if args.print_only:
        if not args.skip_ssh_check:
            _print_command("ssh preflight", ssh_cmd)
        _print_command("claim", claim_cmd)
        if not args.skip_stage:
            _print_command("stage", stage_cmd)
        if args.backend == "batch":
            print("=== batch command ===")
            print(build_batch_command(args))
        else:
            _print_command("dispatch", dispatch_cmd)
        return 0

    if not args.skip_ssh_check:
        ssh_result = _run(ssh_cmd)
        if ssh_result.returncode != 0:
            return ssh_result.returncode

    claim_result = _run(claim_cmd)
    if claim_result.returncode != 0:
        return claim_result.returncode

    if not args.skip_stage:
        (REPO_ROOT / "experiments/results/lightning_batch" / args.job_name).mkdir(
            parents=True,
            exist_ok=True,
        )
        stage_result = _run(stage_cmd)
        if stage_result.returncode != 0:
            _terminal_claim(args, status="failed_stage", notes="Lightning staging failed before dispatch")
            return stage_result.returncode

    if args.backend == "batch":
        return _submit_batch(args)
    dispatch_result = _run(dispatch_cmd)
    if dispatch_result.returncode != 0:
        _terminal_claim(args, status="failed_dispatch", notes="Lightning tmux dispatch failed")
    return dispatch_result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default=default_job_name())
    parser.add_argument("--pr106-archive", type=Path, default=DEFAULT_PR106_ARCHIVE)
    parser.add_argument("--ssh-target", default=default_ssh_target())
    parser.add_argument("--remote-pact", default=DEFAULT_REMOTE_PACT)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--ssh-connect-timeout", type=int, default=30)
    parser.add_argument("--backend", choices=("batch", "studio-tmux"), default="batch")
    parser.add_argument("--gpu-tier", default="T4")
    parser.add_argument("--machine", default="g4dn.2xlarge")
    parser.add_argument("--studio", default=default_studio())
    parser.add_argument("--teamspace", default=default_teamspace())
    parser.add_argument("--user", default=default_user())
    parser.add_argument("--cloud-account", default=None)
    parser.add_argument("--allow-gpu-mismatch", action="store_true")
    parser.add_argument("--candidate-radius", type=int, default=3)
    parser.add_argument("--score-step", type=float, default=1.0)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--predicted-low", type=float, default=0.2065)
    parser.add_argument("--predicted-high", type=float, default=0.2080)
    parser.add_argument("--estimated-cost", type=float, default=2.0)
    parser.add_argument("--predicted-eta-hours", type=float, default=2.0)
    parser.add_argument("--max-runtime-seconds", type=int, default=3 * 60 * 60)
    parser.add_argument(
        "--kill-criteria",
        default="stop if CUDA auth eval fails, NVDEC/DALI probe fails, or score is not below PR106 baseline",
    )
    parser.add_argument("--agent", default="codex:gpt-5.5")
    parser.add_argument("--force-claim", action="store_true")
    parser.add_argument("--require-studio-cuda", action="store_true",
                        help="Require CUDA in the interactive Studio shell during staging; off by default because Batch Jobs request their own machine.")
    parser.add_argument("--dry-run-batch", action="store_true")
    parser.add_argument("--skip-ssh-check", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
