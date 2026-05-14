#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Provider readiness inventory for score-lowering compute.

This tool is deliberately read-only: it does not create kernels, VMs, buckets,
instances, or dispatch claims. Its job is to answer "which provider can we use
right now, and for what evidence grade?" without letting proxy substrates such
as Kaggle or MPS become score claims.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from tac.deploy.provider_contracts import provider_contracts  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "experiments/results/cloud_provider_readiness_latest.json"
DEFAULT_KAGGLE_KERNEL = "adpena/comma-gpu-lane-smoke"

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
AWS_ACCOUNT_RE = re.compile(r'("Account"\s*:\s*")([0-9]{4})[0-9]+([0-9]{2}")')


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


@dataclass
class ProviderReadiness:
    provider: str
    status: str
    score_lowering_role: str
    exact_cuda_evidence_allowed: bool
    proxy_only: bool
    command: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    checked_at_utc: str = ""


Runner = Callable[[list[str], int], CommandResult]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def redact(text: str) -> str:
    """Redact identity-bearing details while preserving useful diagnostics."""

    home = str(Path.home())
    if home and home in text:
        text = text.replace(home, "~")
    text = EMAIL_RE.sub(r"\1***\2", text)
    text = AWS_ACCOUNT_RE.sub(r"\1\2********\3", text)
    return text


def excerpt(text: str, limit: int = 600) -> str:
    text = redact(" ".join(text.strip().split()))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def run_command(command: list[str], timeout_s: int) -> CommandResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return CommandResult(
            command=command,
            returncode=int(result.returncode),
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=f"timeout after {timeout_s}s",
            duration_s=time.monotonic() - started,
        )


def _venv_bin(name: str) -> Path:
    return REPO_ROOT / ".venv" / "bin" / name


def find_cli_command(
    binary: str,
    *,
    uv_package: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str] | None:
    venv = _venv_bin(binary)
    if venv.exists():
        return [str(venv)]
    found = which(binary)
    if found:
        return [found]
    if uv_package and which("uv"):
        return ["uv", "run", "--with", uv_package, binary]
    return None


def _provider(
    *,
    provider: str,
    status: str,
    role: str,
    exact_cuda: bool,
    proxy_only: bool,
    command: list[str] | None = None,
    blockers: list[str] | None = None,
    next_actions: list[str] | None = None,
    result: CommandResult | None = None,
) -> ProviderReadiness:
    return ProviderReadiness(
        provider=provider,
        status=status,
        score_lowering_role=role,
        exact_cuda_evidence_allowed=exact_cuda,
        proxy_only=proxy_only,
        command=list(command or (result.command if result else [])),
        blockers=list(blockers or []),
        next_actions=list(next_actions or []),
        stdout_excerpt=excerpt(result.stdout) if result else "",
        stderr_excerpt=excerpt(result.stderr) if result else "",
        checked_at_utc=utc_now(),
    )


def probe_modal(*, runner: Runner = run_command, timeout_s: int = 15) -> ProviderReadiness:
    command = find_cli_command("modal")
    if command is None:
        return _provider(
            provider="modal",
            status="blocked_cli_missing",
            role="primary_short_cuda_dispatch_when_cli_and_credits_available",
            exact_cuda=False,
            proxy_only=False,
            blockers=["modal_cli_missing"],
            next_actions=["Install Modal CLI in .venv or make modal available on PATH."],
        )
    result = runner(command + ["--version"], timeout_s)
    ok = result.returncode == 0
    return _provider(
        provider="modal",
        status="ready_cli_check_runtime_probe_next" if ok else "blocked_cli_error",
        role="primary_short_cuda_dispatch_and_exact_eval_candidate",
        exact_cuda=False,
        proxy_only=False,
        result=result,
        blockers=["modal_billing_not_checked", "cuda_runtime_import_probe_not_run"] if ok else ["modal_cli_error"],
        next_actions=[
            "Harvest active Modal A1 calls before refiring.",
            "Run the Modal CUDA scorer import probe before any score-lowering job.",
            "Use claim_lane_dispatch.py before every new Modal GPU job.",
        ] if ok else ["Fix Modal CLI authentication or installation."],
    )


def probe_kaggle(
    *,
    kernel_ref: str = DEFAULT_KAGGLE_KERNEL,
    runner: Runner = run_command,
    timeout_s: int = 30,
) -> ProviderReadiness:
    command = find_cli_command("kaggle", uv_package="kaggle")
    if command is None:
        return _provider(
            provider="kaggle",
            status="blocked_cli_missing",
            role="free_proxy_sweeps_only",
            exact_cuda=False,
            proxy_only=True,
            blockers=["kaggle_cli_missing"],
            next_actions=["Install Kaggle CLI or uv so `uv run --with kaggle kaggle ...` works."],
        )
    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        return _provider(
            provider="kaggle",
            status="blocked_credentials_missing",
            role="free_proxy_sweeps_only",
            exact_cuda=False,
            proxy_only=True,
            command=command,
            blockers=["kaggle_credentials_missing"],
            next_actions=["Create ~/.kaggle/kaggle.json before pushing private GPU kernels."],
        )
    result = runner(command + ["kernels", "status", kernel_ref], timeout_s)
    ok = result.returncode == 0
    return _provider(
        provider="kaggle",
        status="ready_proxy" if ok else "blocked_status_query",
        role="free_gpu_proxy_sweeps_config_curves_and_warm_starts_only",
        exact_cuda=False,
        proxy_only=True,
        result=result,
        blockers=[] if ok else ["kaggle_status_query_failed"],
        next_actions=[
            "Use Kaggle for Optuna/CMA-ES/proxy curves only.",
            "Write score_claim=false and ready_for_exact_eval_dispatch=false into Kaggle outputs.",
            "Promote any winning config to Modal/GCP/AWS/Azure exact CUDA before score claims.",
        ] if ok else ["Run `uv run --with kaggle kaggle kernels list --mine` and fix API access."],
    )


def probe_lightning(*, runner: Runner = run_command, timeout_s: int = 20) -> ProviderReadiness:
    python = _venv_bin("python")
    command = [str(python if python.exists() else Path(sys.executable))]
    result = runner(
        command
        + [
            "-c",
            (
                "import importlib.metadata as m; "
                "print(m.version('lightning-sdk'))"
            ),
        ],
        timeout_s,
    )
    ok = result.returncode == 0
    return _provider(
        provider="lightning",
        status="ready_sdk_check_credit_quota_next" if ok else "blocked_sdk_missing",
        role="claimed_cuda_batch_jobs_when_credits_quota_and_studio_route_are_ready",
        exact_cuda=False,
        proxy_only=False,
        result=result,
        blockers=(
            ["credits_or_quota_not_checked", "studio_route_not_checked", "no_dispatch_claim"]
            if ok
            else ["lightning_sdk_missing_or_broken"]
        ),
        next_actions=(
            [
                "Run `scripts/launch_lightning_batch_job.py doctor` before any Lightning dispatch.",
                "Use claim_lane_dispatch.py before every non-dry-run Lightning job.",
                "Treat Lightning exact CUDA only after artifact custody and adjudication land.",
            ]
            if ok
            else ["Install or repair lightning-sdk in .venv, or use another CUDA provider."]
        ),
    )


def probe_vastai(*, runner: Runner = run_command, timeout_s: int = 30) -> ProviderReadiness:
    command = find_cli_command("vastai", uv_package="vastai")
    if command is None:
        return _provider(
            provider="vastai",
            status="blocked_cli_missing",
            role="claimed_cuda_dispatch_when_api_key_offer_and_heartbeat_are_ready",
            exact_cuda=False,
            proxy_only=False,
            blockers=["vastai_cli_missing"],
            next_actions=["Install Vast.ai CLI in .venv or make vastai available on PATH."],
        )
    api_key = Path.home() / ".vast_api_key"
    if not api_key.exists():
        return _provider(
            provider="vastai",
            status="blocked_credentials_missing",
            role="claimed_cuda_dispatch_when_api_key_offer_and_heartbeat_are_ready",
            exact_cuda=False,
            proxy_only=False,
            command=command,
            blockers=["vastai_api_key_missing"],
            next_actions=["Run `vastai set api-key <key>` before any Vast.ai launch."],
        )
    query = "gpu_name=RTX_4090 num_gpus=1 disk_space>60 rentable=True"
    result = runner(command + ["search", "offers", query, "--order", "dph_total", "--limit", "1", "--raw"], timeout_s)
    ok = result.returncode == 0
    return _provider(
        provider="vastai",
        status="ready_offer_query_claim_heartbeat_next" if ok else "blocked_offer_query",
        role="claimed_cuda_dispatch_with_nvdec_probe_heartbeat_and_artifact_custody",
        exact_cuda=False,
        proxy_only=False,
        result=result,
        blockers=["no_dispatch_claim", "heartbeat_not_checked", "nvdec_cuda_probe_not_run"] if ok else ["vastai_offer_query_failed"],
        next_actions=[
            "Use scripts/launch_lane_on_vastai.py only after a non-conflicting lane claim.",
            "Require NVDEC/CUDA probe, heartbeat, and artifact custody before score promotion.",
            "Prefer Modal/Azure/GCP/AWS if Vast.ai offer or network readiness is unstable.",
        ] if ok else ["Check Vast.ai API/network access and retry the read-only offer query."],
    )


def probe_aws(*, runner: Runner = run_command, timeout_s: int = 20) -> ProviderReadiness:
    command = find_cli_command("aws")
    if command is None:
        return _provider(
            provider="aws",
            status="blocked_cli_missing",
            role="future_ec2_gpu_fallback_after_login_budget_quota",
            exact_cuda=False,
            proxy_only=False,
            blockers=["aws_cli_missing"],
            next_actions=["Install AWS CLI."],
        )
    result = runner(command + ["sts", "get-caller-identity", "--output", "json"], timeout_s)
    text = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode == 0:
        return _provider(
            provider="aws",
            status="ready_identity_check_budget_quota_next",
            role="future_ec2_gpu_cuda_dispatch_after_budget_and_quota_check",
            exact_cuda=False,
            proxy_only=False,
            result=result,
            blockers=["gpu_quota_not_checked", "budget_not_checked", "no_dispatch_claim"],
            next_actions=[
                "Run AWS Free Tier and budget checks.",
                "Check G4/G5 GPU quota in target region.",
                "Only then claim and launch EC2 GPU jobs.",
            ],
        )
    blockers = ["aws_auth_failed"]
    if "session has expired" in text:
        blockers = ["aws_session_expired"]
    return _provider(
        provider="aws",
        status="blocked_auth",
        role="future_ec2_gpu_fallback_after_login_budget_quota",
        exact_cuda=False,
        proxy_only=False,
        result=result,
        blockers=blockers,
        next_actions=["Run `aws login`, then re-run this readiness tool."],
    )


def _gcloud_project(runner: Runner, timeout_s: int) -> tuple[str | None, CommandResult]:
    command = ["gcloud", "config", "get-value", "project"]
    result = runner(command, timeout_s)
    project = result.stdout.strip().splitlines()[-1] if result.returncode == 0 and result.stdout.strip() else None
    if project == "(unset)":
        project = None
    return project, result


def probe_gcp(*, runner: Runner = run_command, timeout_s: int = 20) -> ProviderReadiness:
    if shutil.which("gcloud") is None:
        return _provider(
            provider="gcp",
            status="blocked_cli_missing",
            role="future_cuda_dispatch_after_billing_and_gpu_quota",
            exact_cuda=False,
            proxy_only=False,
            blockers=["gcloud_cli_missing"],
            next_actions=["Install Google Cloud CLI."],
        )
    auth = runner(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], timeout_s)
    if auth.returncode != 0 or not auth.stdout.strip():
        return _provider(
            provider="gcp",
            status="blocked_auth",
            role="future_cuda_dispatch_after_billing_and_gpu_quota",
            exact_cuda=False,
            proxy_only=False,
            result=auth,
            blockers=["gcp_auth_missing"],
            next_actions=["Run `gcloud auth login` and select the intended project."],
        )
    project, project_result = _gcloud_project(runner, timeout_s)
    if not project:
        return _provider(
            provider="gcp",
            status="blocked_project_missing",
            role="future_cuda_dispatch_after_billing_and_gpu_quota",
            exact_cuda=False,
            proxy_only=False,
            result=project_result,
            blockers=["gcp_project_unset"],
            next_actions=["Run `gcloud config set project <project-id>`."],
        )
    billing = runner(["gcloud", "billing", "projects", "describe", project, "--format=json"], timeout_s)
    billing_text = f"{billing.stdout}\n{billing.stderr}".lower()
    if billing.returncode != 0 or '"billingenabled": false' in billing_text or "billingenabled: false" in billing_text:
        return _provider(
            provider="gcp",
            status="blocked_billing",
            role="future_cuda_dispatch_after_billing_and_gpu_quota",
            exact_cuda=False,
            proxy_only=False,
            result=billing,
            blockers=["gcp_billing_not_enabled_or_not_readable", "gpu_quota_not_checked"],
            next_actions=[
                "Enable billing on the selected GCP project or switch to a billed project.",
                "Then query T4/L4 GPU quotas before any CUDA dispatch.",
            ],
        )
    return _provider(
        provider="gcp",
        status="ready_identity_billing_check_gpu_quota_next",
        role="future_cuda_dispatch_after_gpu_quota_check",
        exact_cuda=False,
        proxy_only=False,
        result=billing,
        blockers=["gpu_quota_not_checked", "no_dispatch_claim"],
        next_actions=[
            "Query Compute Engine GPU quota for target regions.",
            "Claim lane before creating any VM or batch job.",
        ],
    )


def probe_azure(*, runner: Runner = run_command, timeout_s: int = 20) -> ProviderReadiness:
    if shutil.which("az") is None:
        return _provider(
            provider="azure",
            status="blocked_cli_missing",
            role="future_spot_gpu_fallback_after_login_budget_quota",
            exact_cuda=False,
            proxy_only=False,
            blockers=["azure_cli_missing"],
            next_actions=["Install Azure CLI."],
        )
    result = runner(["az", "account", "show", "--output", "json"], timeout_s)
    if result.returncode == 0:
        return _provider(
            provider="azure",
            status="ready_identity_check_budget_quota_next",
            role="future_spot_gpu_cuda_dispatch_after_budget_and_quota_check",
            exact_cuda=False,
            proxy_only=False,
            result=result,
            blockers=["gpu_quota_not_checked", "budget_not_checked", "no_dispatch_claim"],
            next_actions=[
                "Check credits/subscription budget and NC-series regional quota.",
                "Use scripts/launch_lane_azure.py dry-run before any --no-dry-run launch.",
            ],
        )
    return _provider(
        provider="azure",
        status="blocked_auth",
        role="future_spot_gpu_fallback_after_login_budget_quota",
        exact_cuda=False,
        proxy_only=False,
        result=result,
        blockers=["azure_not_logged_in"],
        next_actions=["Run `az login`, then re-run this readiness tool."],
    )


def collect_readiness(*, kaggle_kernel: str, timeout_s: int) -> dict[str, object]:
    probes_by_provider = {
        "modal": probe_modal(timeout_s=timeout_s),
        "kaggle": probe_kaggle(kernel_ref=kaggle_kernel, timeout_s=timeout_s),
        "lightning": probe_lightning(timeout_s=timeout_s),
        "vastai": probe_vastai(timeout_s=timeout_s),
        "aws": probe_aws(timeout_s=timeout_s),
        "azure": probe_azure(timeout_s=timeout_s),
        "gcp": probe_gcp(timeout_s=timeout_s),
    }
    probes = [
        probes_by_provider[name]
        for name in provider_contracts()
        if name in probes_by_provider
    ]
    return {
        "schema": "cloud_provider_readiness_v1",
        "generated_at_utc": utc_now(),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_semantics": "provider_inventory_no_dispatch_no_score_claim",
        "kaggle_kernel_ref": kaggle_kernel,
        "providers": [asdict(p) for p in probes],
    }


def write_markdown(payload: dict[str, object], path: Path) -> None:
    providers = payload.get("providers", [])
    lines = [
        "# Cloud provider readiness",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "This is a read-only provider inventory. It is not a dispatch, score claim, or promotion artifact.",
        "",
        "| provider | status | exact CUDA allowed now | proxy only | blockers | next action |",
        "|---|---|---:|---:|---|---|",
    ]
    if isinstance(providers, list):
        for raw in providers:
            if not isinstance(raw, dict):
                continue
            blockers = ", ".join(str(x) for x in raw.get("blockers", [])) or "-"
            next_actions = raw.get("next_actions", [])
            next_action = str(next_actions[0]) if isinstance(next_actions, list) and next_actions else "-"
            lines.append(
                "| {provider} | {status} | {exact} | {proxy} | {blockers} | {next_action} |".format(
                    provider=raw.get("provider", ""),
                    status=raw.get("status", ""),
                    exact="yes" if raw.get("exact_cuda_evidence_allowed") else "no",
                    proxy="yes" if raw.get("proxy_only") else "no",
                    blockers=blockers,
                    next_action=next_action.replace("|", "\\|"),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--kaggle-kernel", default=DEFAULT_KAGGLE_KERNEL)
    parser.add_argument("--timeout-s", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = collect_readiness(kaggle_kernel=args.kaggle_kernel, timeout_s=args.timeout_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        write_markdown(payload, args.markdown_output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
