#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a private Kaggle proxy-sweep kernel for PR101-style config search.

This tool only prepares a Kaggle script-kernel directory. It never launches a
kernel, never pushes to Kaggle, and never emits exact-eval or score-claim
metadata. The generated kernel is a proxy/config search substrate whose outputs
must be promoted through a separate exact CUDA archive/eval path before they can
affect any score claim.
"""
from __future__ import annotations

import argparse
import json
import shlex
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_DIR = REPO_ROOT / "experiments/kaggle_kernels/pr101_proxy_sweep"
DEFAULT_OWNER = "adpena"
DEFAULT_SLUG = "pr101-proxy-sweep"
DEFAULT_TITLE = "PR101 Proxy Sweep"
DEFAULT_LANE_ID = "kaggle_pr101_proxy_sweep"
DEFAULT_AGENT = "codex:kaggle_proxy_readiness"
SCRIPT_NAME = "pr101_proxy_sweep.py"
BIAS_REFINE_KERNEL_DIR = REPO_ROOT / "experiments/kaggle_kernels/pr101_bias_refine"
BIAS_REFINE_SCRIPT_NAME = "pr101_bias_refine.py"
METADATA_NAME = "kernel-metadata.json"
MANIFEST_NAME = "proxy_sweep_build_manifest.json"
README_NAME = "README.md"

EVIDENCE_SEMANTICS = (
    "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval"
)
DISPATCH_BLOCKERS = [
    "kaggle_proxy_substrate_not_contest_exact_eval",
    "no_archive_zip_emitted",
    "no_inflate_runtime_emitted",
    "no_contest_cuda_auth_eval",
    "operator_must_promote_candidate_manually",
]


@dataclass(frozen=True)
class KernelBuildProfile:
    name: str
    default_kernel_dir: Path
    default_slug: str
    default_title: str
    default_lane_id: str
    script_name: str
    lane_class: str
    candidate_family: str
    param_schema: str
    claim_notes: str


KERNEL_BUILD_PROFILES: dict[str, KernelBuildProfile] = {
    "pr101_proxy_sweep": KernelBuildProfile(
        name="pr101_proxy_sweep",
        default_kernel_dir=DEFAULT_KERNEL_DIR,
        default_slug=DEFAULT_SLUG,
        default_title=DEFAULT_TITLE,
        default_lane_id=DEFAULT_LANE_ID,
        script_name=SCRIPT_NAME,
        lane_class="pr101_kaggle_proxy_sweep",
        candidate_family="pr101_proxy_config_search",
        param_schema="pr101_kaggle_proxy_candidate_params_v1",
        claim_notes="Kaggle PR101 proxy sweep only; score_claim=false; exact CUDA promotion required",
    ),
    "pr101_bias_refine": KernelBuildProfile(
        name="pr101_bias_refine",
        default_kernel_dir=BIAS_REFINE_KERNEL_DIR,
        default_slug="pr101-bias-refine",
        default_title="PR101 Bias Refine",
        default_lane_id="kaggle_pr101_bias_refine",
        script_name=BIAS_REFINE_SCRIPT_NAME,
        lane_class="pr101_kaggle_bias_refine",
        candidate_family="pr101_runtime_consumed_bias_refinement",
        param_schema="pr101_kaggle_proxy_bias_runtime_params_v1",
        claim_notes=(
            "Kaggle PR101 bias-only refinement; all params are runtime-consumed; "
            "score_claim=false; exact CUDA promotion required"
        ),
    ),
}


KERNEL_SCRIPT = r'''#!/usr/bin/env python3
"""PR101 proxy-sweep Kaggle kernel.

This script is intentionally proxy-only. It searches PR101/A1-style runtime
configuration knobs with random, Optuna-style, or CMA-ES-style proposal loops,
then writes JSON/JSONL artifacts that can seed a real exact CUDA dispatch.

Hard evidence boundary:
- score_claim is always false
- ready_for_exact_eval_dispatch is always false
- proxy_only is always true
- no archive.zip is emitted
- no inflate runtime is emitted
- no exact auth eval is performed
- MPS auth eval is forbidden and not available on Kaggle
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "pr101_kaggle_proxy_sweep_v1"
DEFAULT_PROFILE = "__DEFAULT_PROFILE__"
EVIDENCE_SEMANTICS = "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval"
DISPATCH_BLOCKERS = [
    "kaggle_proxy_substrate_not_contest_exact_eval",
    "no_archive_zip_emitted",
    "no_inflate_runtime_emitted",
    "no_contest_cuda_auth_eval",
    "operator_must_promote_candidate_manually",
]
PROFILE_CONFIGS = {
    "pr101_proxy_sweep": {
        "schema": SCHEMA,
        "lane_id": "kaggle_pr101_proxy_sweep",
        "lane_class": "pr101_kaggle_proxy_sweep",
        "candidate_family": "pr101_proxy_config_search",
        "candidate_prefix": "proxy",
        "param_schema": "pr101_kaggle_proxy_candidate_params_v1",
        "search_space": {
            "delta_scale": {"low": 0.0085, "high": 0.0115, "anchor": 0.0100},
            "bias_r": {"low": -1.35, "high": -0.65, "anchor": -1.0},
            "bias_g": {"low": -1.35, "high": -0.65, "anchor": -1.0},
            "bias_b": {"low": -1.35, "high": -0.65, "anchor": -1.0},
            "latent_delta_scale": {"low": 0.006, "high": 0.014, "anchor": 0.010},
            "smooth_weight": {"low": 0.0, "high": 0.12, "anchor": 0.02},
        },
    },
    "pr101_bias_refine": {
        "schema": SCHEMA,
        "lane_id": "kaggle_pr101_bias_refine",
        "lane_class": "pr101_kaggle_bias_refine",
        "candidate_family": "pr101_runtime_consumed_bias_refinement",
        "candidate_prefix": "bias_refine",
        "param_schema": "pr101_kaggle_proxy_bias_runtime_params_v1",
        "search_space": {
            "bias_r": {"low": -1.08, "high": -0.92, "anchor": -1.0},
            "bias_g": {"low": -1.08, "high": -0.92, "anchor": -1.0},
            "bias_b": {"low": -1.08, "high": -0.92, "anchor": -1.0},
        },
    },
}
CONTRACT = {
    "schema": "pr101_kaggle_proxy_contract_v1",
    "proxy_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "ready_for_exact_eval_dispatch": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
    "exact_auth_eval_performed": False,
    "contest_cuda_auth_eval": False,
    "mps_auth_eval": False,
    "archive_zip_emitted": False,
    "inflate_runtime_emitted": False,
    "evidence_semantics": EVIDENCE_SEMANTICS,
    "dispatch_blockers": list(DISPATCH_BLOCKERS),
}


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    trial_index: int
    optimizer: str
    optimizer_status: str
    param_schema: str
    lane_class: str
    candidate_family: str
    params: dict[str, float]
    proxy_objective: float
    proxy_components: dict[str, float]
    score_claim: bool = False
    score_claim_valid: bool = False
    ready_for_exact_eval_dispatch: bool = False
    proxy_only: bool = True
    evidence_semantics: str = EVIDENCE_SEMANTICS


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def profile_config(profile_name: str) -> dict[str, Any]:
    profile = PROFILE_CONFIGS.get(profile_name)
    if profile is None:
        raise SystemExit(
            f"unknown profile {profile_name!r}; expected one of {sorted(PROFILE_CONFIGS)}"
        )
    return profile


def sample_uniform(rng: random.Random, search_space: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        name: rng.uniform(spec["low"], spec["high"])
        for name, spec in search_space.items()
    }


def sample_cmaes_style(
    rng: random.Random,
    mean: dict[str, float],
    sigma: float,
    search_space: dict[str, dict[str, float]],
) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, spec in search_space.items():
        width = spec["high"] - spec["low"]
        proposal = mean[name] + rng.gauss(0.0, sigma * width)
        params[name] = clamp(proposal, spec["low"], spec["high"])
    return params


def update_cmaes_style_mean(
    mean: dict[str, float],
    winners: list[CandidateResult],
    search_space: dict[str, dict[str, float]],
    learning_rate: float = 0.35,
) -> dict[str, float]:
    if not winners:
        return dict(mean)
    best = winners[0].params
    return {
        name: (1.0 - learning_rate) * mean[name] + learning_rate * best[name]
        for name in search_space
    }


def proxy_objective(
    params: dict[str, float],
    *,
    seed: int,
    search_space: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float]]:
    """Cheap deterministic proxy around known PR101/A1 local optimum.

    This is not a score. It is a shaped config-search objective that keeps
    optimizer plumbing busy on Kaggle while candidates wait for exact CUDA
    promotion elsewhere.
    """

    normalized_terms: dict[str, float] = {}
    for name, spec in search_space.items():
        half_width = (spec["high"] - spec["low"]) / 2.0
        normalized_terms[name] = ((params[name] - spec["anchor"]) / half_width) ** 2

    bias_channels = [channel for channel in ("bias_r", "bias_g", "bias_b") if channel in params]
    bias_asymmetry = 0.0
    if len(bias_channels) == 3:
        bias_mean = statistics.fmean(params[channel] for channel in bias_channels)
        bias_asymmetry = statistics.fmean(
            (params[channel] - bias_mean) ** 2
            for channel in bias_channels
        )
    latent_interaction = (
        abs(params["latent_delta_scale"] - params["delta_scale"])
        if "latent_delta_scale" in params and "delta_scale" in params
        else 0.0
    )
    smooth_penalty = (
        max(0.0, params["smooth_weight"] - 0.035) ** 2
        if "smooth_weight" in params
        else 0.0
    )

    deterministic_jitter_rng = random.Random(
        seed + int(sum(params.values()) * 1_000_000)
    )
    deterministic_jitter = deterministic_jitter_rng.uniform(0.0, 2.5e-5)

    objective = (
        0.19285
        + 0.00024 * statistics.fmean(normalized_terms.values())
        + 0.00036 * bias_asymmetry
        + 0.018 * latent_interaction
        + 0.040 * smooth_penalty
        + deterministic_jitter
    )
    return objective, {
        "anchor_proximity": statistics.fmean(normalized_terms.values()),
        "bias_asymmetry": bias_asymmetry,
        "latent_delta_mismatch": latent_interaction,
        "smooth_penalty": smooth_penalty,
        "deterministic_jitter": deterministic_jitter,
    }


def run_search(
    *,
    profile_name: str,
    optimizer: str,
    max_trials: int,
    seed: int,
) -> tuple[list[CandidateResult], str]:
    profile = profile_config(profile_name)
    search_space = profile["search_space"]
    rng = random.Random(seed)
    results: list[CandidateResult] = []
    optimizer_status = optimizer

    if optimizer == "optuna":
        try:
            import optuna  # type: ignore
        except Exception:
            optimizer_status = "optuna_missing_random_fallback"
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def objective(trial: Any) -> float:
                params = {
                    name: trial.suggest_float(name, spec["low"], spec["high"])
                    for name, spec in search_space.items()
                }
                value, components = proxy_objective(params, seed=seed, search_space=search_space)
                results.append(CandidateResult(
                    candidate_id=f"{profile['candidate_prefix']}_optuna_{len(results):04d}",
                    trial_index=len(results),
                    optimizer=optimizer,
                    optimizer_status="optuna_tpe",
                    param_schema=profile["param_schema"],
                    lane_class=profile["lane_class"],
                    candidate_family=profile["candidate_family"],
                    params=params,
                    proxy_objective=value,
                    proxy_components=components,
                ))
                return value

            study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
            return sorted(results, key=lambda row: row.proxy_objective), "optuna_tpe"

    if optimizer == "cmaes":
        mean = {name: spec["anchor"] for name, spec in search_space.items()}
        sigma = 0.50
        optimizer_status = "cmaes_style_stdlib"
        generation: list[CandidateResult] = []
        for idx in range(max_trials):
            params = sample_cmaes_style(rng, mean, sigma, search_space)
            value, components = proxy_objective(params, seed=seed, search_space=search_space)
            row = CandidateResult(
                candidate_id=f"{profile['candidate_prefix']}_cmaes_{idx:04d}",
                trial_index=idx,
                optimizer=optimizer,
                optimizer_status=optimizer_status,
                param_schema=profile["param_schema"],
                lane_class=profile["lane_class"],
                candidate_family=profile["candidate_family"],
                params=params,
                proxy_objective=value,
                proxy_components=components,
            )
            results.append(row)
            generation.append(row)
            if len(generation) >= 4:
                generation.sort(key=lambda item: item.proxy_objective)
                mean = update_cmaes_style_mean(mean, generation[:2], search_space)
                sigma *= 0.82
                generation.clear()
        return sorted(results, key=lambda row: row.proxy_objective), optimizer_status

    optimizer_status = "random_search"
    for idx in range(max_trials):
        params = sample_uniform(rng, search_space)
        value, components = proxy_objective(params, seed=seed, search_space=search_space)
        results.append(CandidateResult(
            candidate_id=f"{profile['candidate_prefix']}_random_{idx:04d}",
            trial_index=idx,
            optimizer=optimizer,
            optimizer_status=optimizer_status,
            param_schema=profile["param_schema"],
            lane_class=profile["lane_class"],
            candidate_family=profile["candidate_family"],
            params=params,
            proxy_objective=value,
            proxy_components=components,
        ))
    return sorted(results, key=lambda row: row.proxy_objective), optimizer_status


def write_outputs(
    *,
    output_dir: Path,
    profile_name: str,
    optimizer: str,
    optimizer_status: str,
    seed: int,
    max_trials: int,
    results: list[CandidateResult],
) -> dict[str, Any]:
    profile = profile_config(profile_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "proxy_sweep_results.jsonl"
    best_path = output_dir / "best_proxy_candidate.json"
    manifest_path = output_dir / "proxy_sweep_manifest.json"

    with results_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")

    best = asdict(results[0]) if results else None
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema": profile["schema"],
        "generated_at_utc": utc_now(),
        "platform": "kaggle_script_kernel",
        "profile": profile_name,
        "lane_id": profile["lane_id"],
        "lane_class": profile["lane_class"],
        "candidate_family": profile["candidate_family"],
        "param_schema": profile["param_schema"],
        "optimizer": optimizer,
        "optimizer_status": optimizer_status,
        "seed": seed,
        "max_trials": max_trials,
        "n_results": len(results),
        "best_candidate": best,
        "search_space": profile["search_space"],
        "contract": CONTRACT,
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "proxy_only": True,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "mps_auth_eval": False,
        "archive_zip_emitted": False,
        "inflate_runtime_emitted": False,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "promotion_path": (
            "Copy best_proxy_candidate.json into a real archive-builder or "
            "training dispatch, then run claimed exact CUDA auth eval outside Kaggle."
        ),
        "outputs": {
            "results_jsonl": str(results_path),
            "best_candidate_json": str(best_path),
            "manifest_json": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default=DEFAULT_PROFILE)
    parser.add_argument("--optimizer", choices=("random", "optuna", "cmaes"), default="cmaes")
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_trials < 1:
        raise SystemExit("--max-trials must be >= 1")
    output_dir = args.output_dir or Path(f"/kaggle/working/{args.profile}")
    results, optimizer_status = run_search(
        profile_name=args.profile,
        optimizer=args.optimizer,
        max_trials=args.max_trials,
        seed=args.seed,
    )
    manifest = write_outputs(
        output_dir=output_dir,
        profile_name=args.profile,
        optimizer=args.optimizer,
        optimizer_status=optimizer_status,
        seed=args.seed,
        max_trials=args.max_trials,
        results=results,
    )
    print(json.dumps({
        "schema": "pr101_kaggle_proxy_sweep_stdout_v1",
        "proxy_only": True,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "profile": args.profile,
        "param_schema": manifest["param_schema"],
        "best_candidate": manifest["best_candidate"],
        "manifest_json": manifest["outputs"]["manifest_json"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


README_TEXT = """# PR101 proxy-sweep Kaggle kernel

This directory is a private Kaggle script-kernel substrate for cheap
config-search only. It is intentionally **not** exact auth eval and cannot be a
score claim. The build manifest records the generated profile and parameter
schema so archive builders can fail closed instead of silently ignoring
searched parameters.

Generated outputs always declare:

- `score_claim=false`
- `score_claim_valid=false`
- `ready_for_exact_eval_dispatch=false`
- `proxy_only=true`
- `exact_auth_eval_performed=false`
- `archive_zip_emitted=false`
- `inflate_runtime_emitted=false`

Operator-controlled launch command:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id kaggle_pr101_proxy_sweep --platform kaggle --instance-job-id kaggle:adpena/pr101-proxy-sweep --agent codex:kaggle_proxy_readiness --status active_proxy_dispatch --notes "Kaggle PR101 proxy sweep only; score_claim=false; exact CUDA promotion required"
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id kaggle_pr101_proxy_sweep --platform kaggle --instance-job-id kaggle:adpena/pr101-proxy-sweep --agent codex:kaggle_proxy_readiness --status active_proxy_dispatch --notes "Kaggle PR101 proxy sweep only; score_claim=false; exact CUDA promotion required"
uv run --with kaggle kaggle kernels push -p experiments/kaggle_kernels/pr101_proxy_sweep
```

The first command is a claim dry-run; the second records the active proxy
dispatch claim. Do not run the push command without a successful active claim.
When the Kaggle run reaches a terminal state, close the claim with the
terminal-claim template in `proxy_sweep_build_manifest.json`.

Any promising `best_proxy_candidate.json` must be promoted through a separate
claimed exact CUDA archive/eval path before it can influence score status.
"""


@dataclass(frozen=True)
class BuildResult:
    kernel_dir: Path
    metadata_path: Path
    script_path: Path
    manifest_path: Path
    readme_path: Path
    claim_command: list[str]
    claim_dry_run_command: list[str]
    push_command: list[str]
    terminal_claim_command_template: list[str]
    score_claim: bool
    ready_for_exact_eval_dispatch: bool
    proxy_only: bool
    evidence_semantics: str


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_text(path: Path, text: str, *, executable: bool = False, force: bool = False) -> None:
    if path.exists() and not force:
        current = path.read_text(encoding="utf-8")
        if current != text:
            raise FileExistsError(
                f"{path} already exists with different content; pass --force to rewrite"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _build_profile(profile_name: str) -> KernelBuildProfile:
    try:
        return KERNEL_BUILD_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown Kaggle kernel profile {profile_name!r}; "
            f"expected one of {sorted(KERNEL_BUILD_PROFILES)}"
        ) from exc


def kernel_metadata(*, owner: str, slug: str, title: str, code_file: str) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _command_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def build_claim_command(
    *,
    owner: str,
    slug: str,
    lane_id: str,
    agent: str,
    dry_run: bool = False,
    status: str = "active_proxy_dispatch",
    notes: str = KERNEL_BUILD_PROFILES["pr101_proxy_sweep"].claim_notes,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "--lane-id",
            lane_id,
            "--platform",
            "kaggle",
            "--instance-job-id",
            f"kaggle:{owner}/{slug}",
            "--agent",
            agent,
            "--status",
            status,
            "--notes",
            notes,
        ]
    )
    return command


def build_manifest(
    *,
    owner: str,
    slug: str,
    title: str,
    kernel_dir: Path,
    lane_id: str,
    agent: str,
    profile: KernelBuildProfile,
) -> dict[str, object]:
    kernel_dir_for_command = _command_path(kernel_dir)
    claim_dry_run_command = build_claim_command(
        owner=owner,
        slug=slug,
        lane_id=lane_id,
        agent=agent,
        notes=profile.claim_notes,
        dry_run=True,
    )
    claim_command = build_claim_command(
        owner=owner,
        slug=slug,
        lane_id=lane_id,
        agent=agent,
        notes=profile.claim_notes,
    )
    terminal_claim_command_template = build_claim_command(
        owner=owner,
        slug=slug,
        lane_id=lane_id,
        agent=agent,
        status="completed_proxy_or_failed_proxy_SET_EXACT_STATUS",
        notes="Set exact Kaggle terminal status and artifact path; still score_claim=false",
    ) + ["--force"]
    push_command = [
        "uv",
        "run",
        "--with",
        "kaggle",
        "kaggle",
        "kernels",
        "push",
        "-p",
        kernel_dir_for_command,
    ]
    remote_job_manifest = f".omx/logs/remote_jobs/kaggle-{slug}.json"
    status_path = f".omx/status/kaggle-{slug}.json"
    return {
        "schema": "kaggle_proxy_sweep_kernel_build_manifest_v1",
        "generated_at_utc": utc_now(),
        "kernel_id": f"{owner}/{slug}",
        "kernel_title": title,
        "profile": profile.name,
        "lane_id": lane_id,
        "lane_class": profile.lane_class,
        "candidate_family": profile.candidate_family,
        "param_schema": profile.param_schema,
        "kernel_dir": kernel_dir_for_command,
        "kernel_ref": f"{owner}/{slug}",
        "remote_job_manifest_path": remote_job_manifest,
        "status_path": status_path,
        "remote_command": command_text(push_command),
        "code_file": profile.script_name,
        "proxy_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "exact_auth_eval_performed": False,
        "contest_cuda_auth_eval": False,
        "mps_auth_eval": False,
        "archive_zip_emitted": False,
        "inflate_runtime_emitted": False,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "dispatch_claim_required": True,
        "claim_command_dry_run": claim_dry_run_command,
        "claim_command_dry_run_text": command_text(claim_dry_run_command),
        "claim_command": claim_command,
        "claim_command_text": command_text(claim_command),
        "push_command": push_command,
        "push_command_text": command_text(push_command),
        "safe_push_sequence": [
            claim_dry_run_command,
            claim_command,
            push_command,
        ],
        "safe_push_sequence_text": [
            command_text(claim_dry_run_command),
            command_text(claim_command),
            command_text(push_command),
        ],
        "terminal_claim_command_template": terminal_claim_command_template,
        "terminal_claim_command_template_text": command_text(terminal_claim_command_template),
        "operator_controlled_launch": True,
    }


def build_remote_job_manifest(build_manifest: Mapping[str, object]) -> dict[str, object]:
    kernel_ref = str(build_manifest["kernel_ref"])
    return {
        "schema": "kaggle_remote_job_manifest_v1",
        "generated_at_utc": utc_now(),
        "provider": "kaggle",
        "kernel_ref": kernel_ref,
        "slug": kernel_ref.split("/", 1)[-1],
        "lane_id": build_manifest["lane_id"],
        "status": "queued",
        "phase": "manifest_built_pending_operator_push",
        "proxy_only": True,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "remote_command": build_manifest["remote_command"],
        "status_path": build_manifest["status_path"],
        "source_manifest_path": build_manifest["remote_job_manifest_path"],
        "notes": "Kaggle proxy/search only; byte-closed packet plus exact CUDA promotion required before score claims.",
    }


def build_kernel(
    *,
    profile_name: str = "pr101_proxy_sweep",
    kernel_dir: Path | None = None,
    owner: str = DEFAULT_OWNER,
    slug: str | None = None,
    title: str | None = None,
    lane_id: str | None = None,
    agent: str = DEFAULT_AGENT,
    force: bool = False,
    write_remote_job_manifest: bool = False,
) -> BuildResult:
    profile = _build_profile(profile_name)
    kernel_dir = kernel_dir or profile.default_kernel_dir
    slug = slug or profile.default_slug
    title = title or profile.default_title
    lane_id = lane_id or profile.default_lane_id
    metadata_path = kernel_dir / METADATA_NAME
    script_path = kernel_dir / profile.script_name
    manifest_path = kernel_dir / MANIFEST_NAME
    readme_path = kernel_dir / README_NAME

    metadata = kernel_metadata(owner=owner, slug=slug, title=title, code_file=profile.script_name)
    manifest = build_manifest(
        owner=owner,
        slug=slug,
        title=title,
        kernel_dir=kernel_dir,
        lane_id=lane_id,
        agent=agent,
        profile=profile,
    )

    _write_text(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        force=force,
    )
    _write_text(
        script_path,
        KERNEL_SCRIPT.replace("__DEFAULT_PROFILE__", profile.name),
        executable=True,
        force=force,
    )
    _write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        force=force,
    )
    if write_remote_job_manifest:
        remote_job_manifest_path = Path(str(manifest["remote_job_manifest_path"]))
        _write_text(
            remote_job_manifest_path,
            json.dumps(build_remote_job_manifest(manifest), indent=2, sort_keys=True) + "\n",
            force=True,
        )
    readme = README_TEXT.replace(
        "experiments/kaggle_kernels/pr101_proxy_sweep",
        _command_path(kernel_dir),
    ).replace(
        "kaggle_pr101_proxy_sweep",
        lane_id,
    ).replace(
        "codex:kaggle_proxy_readiness",
        agent,
    ).replace(
        "kaggle:adpena/pr101-proxy-sweep",
        f"kaggle:{owner}/{slug}",
    )
    _write_text(readme_path, readme, force=force)

    return BuildResult(
        kernel_dir=kernel_dir,
        metadata_path=metadata_path,
        script_path=script_path,
        manifest_path=manifest_path,
        readme_path=readme_path,
        claim_command=list(manifest["claim_command"]),  # type: ignore[arg-type]
        claim_dry_run_command=list(manifest["claim_command_dry_run"]),  # type: ignore[arg-type]
        push_command=list(manifest["push_command"]),  # type: ignore[arg-type]
        terminal_claim_command_template=list(manifest["terminal_claim_command_template"]),  # type: ignore[arg-type]
        score_claim=False,
        ready_for_exact_eval_dispatch=False,
        proxy_only=True,
        evidence_semantics=EVIDENCE_SEMANTICS,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(KERNEL_BUILD_PROFILES), default="pr101_proxy_sweep")
    parser.add_argument("--kernel-dir", type=Path, default=None)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--lane-id", default=None)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_kernel(
        profile_name=args.profile,
        kernel_dir=args.kernel_dir,
        owner=args.owner,
        slug=args.slug,
        title=args.title,
        lane_id=args.lane_id,
        agent=args.agent,
        force=args.force,
        write_remote_job_manifest=True,
    )
    print(json.dumps({
        "schema": "kaggle_proxy_sweep_kernel_build_stdout_v1",
        "kernel_dir": str(result.kernel_dir),
        "metadata_path": str(result.metadata_path),
        "script_path": str(result.script_path),
        "manifest_path": str(result.manifest_path),
        "readme_path": str(result.readme_path),
        "remote_job_manifest_path": str(
            json.loads(result.manifest_path.read_text(encoding="utf-8"))["remote_job_manifest_path"]
        ),
        "dispatch_claim_required": True,
        "claim_dry_run_command": command_text(result.claim_dry_run_command),
        "claim_command": command_text(result.claim_command),
        "score_claim": result.score_claim,
        "ready_for_exact_eval_dispatch": result.ready_for_exact_eval_dispatch,
        "proxy_only": result.proxy_only,
        "evidence_semantics": result.evidence_semantics,
        "push_command": command_text(result.push_command),
    }, indent=2, sort_keys=True))
    print()
    print("Claim dry-run command:")
    print(command_text(result.claim_dry_run_command))
    print()
    print("Claim command:")
    print(command_text(result.claim_command))
    print()
    print("Operator-controlled launch command after successful claim:")
    print(command_text(result.push_command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
