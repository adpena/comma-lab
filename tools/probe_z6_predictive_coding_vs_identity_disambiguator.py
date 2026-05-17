#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z6 predictive-coding vs identity-predictor disambiguator.

This is a fail-closed L5-v2 probe surface. It either emits the exact paired
smoke commands required for Z6's full-FiLM and identity-predictor regimes, or
it consumes both smoke ``stats.json`` files and records a proxy-only verdict.

The output is deliberately not score authority. Synthetic smoke loss can
detect engineering regressions and decide the next build/eval step; it cannot
promote a paradigm, rank a lane, or replace paired contest CPU/CUDA evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "z6_predictive_coding_vs_identity_disambiguator_v1"
PROBE_ID = "z6_predictive_coding_vs_identity_disambiguator"
SUBSTRATE_TAG = "time_traveler_l5_z6"
LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
DEFAULT_VIDEO_PATH = "upstream/videos/0.mkv"
DEFAULT_FULL_OUTPUT_DIR = (
    "experiments/results/time_traveler_l5_z6/"
    "disambiguator_full_film_smoke_20260516_codex"
)
DEFAULT_IDENTITY_OUTPUT_DIR = (
    "experiments/results/time_traveler_l5_z6/"
    "disambiguator_identity_smoke_20260516_codex"
)
DEFAULT_OUTPUT_JSON = (
    ".omx/research/"
    "l5_v2_z6_identity_predictor_disambiguator_20260516_codex.json"
)
DEFAULT_OUTPUT_MD = (
    ".omx/research/"
    "l5_v2_z6_identity_predictor_disambiguator_20260516_codex.md"
)
TRAINER_PATH = "experiments/train_substrate_time_traveler_l5_z6.py"
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}
SMOKE_PROXY_BLOCKERS = [
    "smoke_proxy_synthetic_no_scorer",
    "no_contest_cpu_cuda_pair",
    "no_byte_closed_score_anchor",
    "not_paradigm_claim_authority",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.expanduser().resolve()
    resolved.relative_to(repo_root)
    text = str(resolved)
    if (
        text.startswith("/tmp/")
        or "/private/tmp/" in text
        or "/var/tmp/" in text
    ):
        raise ValueError(f"refusing to write transient Z6 probe output: {text!r}")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _expect_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    value = payload.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a literal JSON boolean")
    if value is not expected:
        raise ValueError(f"{key} must be {str(expected).lower()}")


def _expect_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON number")
    return float(value)


def _expect_optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be null or a JSON number")
    return float(value)


def _validate_z6_smoke_stats(
    payload: Mapping[str, Any],
    *,
    expected_identity: bool,
    label: str,
) -> None:
    if payload.get("substrate_tag") != SUBSTRATE_TAG:
        raise ValueError(f"{label}: substrate_tag must be {SUBSTRATE_TAG!r}")
    if payload.get("lane_id") != LANE_ID:
        raise ValueError(f"{label}: lane_id must be {LANE_ID!r}")
    _expect_bool(payload, "smoke", True)
    _expect_bool(payload, "score_claim_valid", False)
    _expect_bool(payload, "promotion_eligible", False)
    _expect_bool(payload, "ready_for_exact_eval_dispatch", False)
    _expect_bool(payload, "identity_predictor", expected_identity)
    _expect_number(payload, "final_loss_proxy")
    _expect_optional_number(payload, "final_recon")
    _expect_optional_number(payload, "final_residual")
    archive_bytes = payload.get("archive_bytes")
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool):
        raise ValueError(f"{label}: archive_bytes must be an integer")
    if archive_bytes <= 0:
        raise ValueError(f"{label}: archive_bytes must be > 0")


def _require_same_config(
    full: Mapping[str, Any],
    identity: Mapping[str, Any],
    key: str,
) -> None:
    if full.get(key) != identity.get(key):
        raise ValueError(
            f"full and identity smoke stats must match {key}: "
            f"{full.get(key)!r} != {identity.get(key)!r}"
        )


def _mode_stats_row(
    payload: Mapping[str, Any],
    *,
    path: Path,
    repo_root: Path,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "path": _repo_relative(path, repo_root=repo_root),
        "sha256": _sha256_file(path),
        "identity_predictor": payload.get("identity_predictor"),
        "epochs": payload.get("epochs"),
        "requested_epochs": payload.get("requested_epochs"),
        "lambda_residual_entropy": payload.get("lambda_residual_entropy"),
        "predictor_kernel_size": payload.get("predictor_kernel_size"),
        "final_loss_proxy": payload.get("final_loss_proxy"),
        "final_recon": payload.get("final_recon"),
        "final_residual": payload.get("final_residual"),
        "archive_bytes": payload.get("archive_bytes"),
        "evidence_grade": payload.get("evidence_grade"),
        "stats_payload": dict(payload),
    }


def _delta_optional(identity_value: Any, full_value: Any) -> float | None:
    if identity_value is None or full_value is None:
        return None
    return float(identity_value) - float(full_value)


def _smoke_command(
    *,
    output_dir: str,
    identity_predictor: bool,
    epochs: int,
    device: str,
    seed: int,
) -> list[str]:
    command = [
        ".venv/bin/python",
        TRAINER_PATH,
        "--video-path",
        DEFAULT_VIDEO_PATH,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--device",
        device,
        "--seed",
        str(seed),
        "--smoke",
    ]
    if identity_predictor:
        command.append("--identity-predictor")
    return command


def build_plan_payload(
    *,
    epochs: int = 3,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Return the paired smoke command plan without asserting a result."""

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "evidence_grade": "plan_only_no_smoke_stats",
        "verdict": "pending_paired_smoke_stats",
        **FALSE_AUTHORITY_FLAGS,
        "blockers": [
            "full_film_smoke_stats_missing",
            "identity_predictor_smoke_stats_missing",
            *SMOKE_PROXY_BLOCKERS,
        ],
        "paired_smoke_commands": [
            {
                "mode": "full_film_predictor",
                "identity_predictor": False,
                "output_dir": DEFAULT_FULL_OUTPUT_DIR,
                "stats_path": f"{DEFAULT_FULL_OUTPUT_DIR}/stats.json",
                "command": _smoke_command(
                    output_dir=DEFAULT_FULL_OUTPUT_DIR,
                    identity_predictor=False,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                ),
            },
            {
                "mode": "identity_predictor",
                "identity_predictor": True,
                "output_dir": DEFAULT_IDENTITY_OUTPUT_DIR,
                "stats_path": f"{DEFAULT_IDENTITY_OUTPUT_DIR}/stats.json",
                "command": _smoke_command(
                    output_dir=DEFAULT_IDENTITY_OUTPUT_DIR,
                    identity_predictor=True,
                    epochs=epochs,
                    device=device,
                    seed=seed,
                ),
            },
        ],
        "reactivation_criteria": [
            "run both smoke commands from same git SHA and seed",
            "compare stats through this tool",
            "do not assert Z6 paradigm movement until paired contest CPU/CUDA exact eval exists",
            "keep full_main council-gated until real-video training path is implemented",
        ],
    }


def evaluate_stats_pair(
    *,
    full_stats_path: Path,
    identity_stats_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Compare paired Z6 smoke stats and emit a proxy-only verdict."""

    full = _load_json_object(full_stats_path)
    identity = _load_json_object(identity_stats_path)
    _validate_z6_smoke_stats(full, expected_identity=False, label="full")
    _validate_z6_smoke_stats(identity, expected_identity=True, label="identity")
    for key in (
        "epochs",
        "requested_epochs",
        "smoke_epoch_cap",
        "lambda_residual_entropy",
        "predictor_kernel_size",
    ):
        _require_same_config(full, identity, key)

    delta_loss = (
        float(identity["final_loss_proxy"]) - float(full["final_loss_proxy"])
    )
    delta_recon = _delta_optional(identity.get("final_recon"), full.get("final_recon"))
    delta_residual = _delta_optional(
        identity.get("final_residual"),
        full.get("final_residual"),
    )
    delta_archive = int(full["archive_bytes"]) - int(identity["archive_bytes"])
    if delta_loss > 1e-9:
        verdict = "full_film_predictor_proxy_lower_loss"
        proxy_preferred = "full_film_predictor"
    elif delta_loss < -1e-9:
        verdict = "identity_predictor_proxy_lower_loss"
        proxy_preferred = "identity_predictor"
    else:
        verdict = "indeterminate_tie_smoke_proxy_only"
        proxy_preferred = "tie"

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "evidence_grade": "smoke_proxy_synthetic_pair",
        "verdict": verdict,
        "proxy_preferred_mode": proxy_preferred,
        "paired": True,
        **FALSE_AUTHORITY_FLAGS,
        "blockers": SMOKE_PROXY_BLOCKERS,
        "source_stats": [
            _mode_stats_row(
                full,
                path=full_stats_path,
                repo_root=repo_root,
                mode="full_film_predictor",
            ),
            _mode_stats_row(
                identity,
                path=identity_stats_path,
                repo_root=repo_root,
                mode="identity_predictor",
            ),
        ],
        "deltas": {
            "identity_minus_full_loss_proxy": delta_loss,
            "identity_minus_full_recon": delta_recon,
            "identity_minus_full_residual": delta_residual,
            "full_minus_identity_archive_bytes": delta_archive,
        },
        "result_review": {
            "classification": "smoke_proxy_only",
            "score_formula_recomputed": False,
            "score_formula_recompute_blocker": "no seg_dist/pose_dist/contest archive score fields in smoke stats",
            "component_score_authority": False,
            "contest_compliance_authority": False,
            "next_authoritative_gate": "paired contest CPU/CUDA exact eval after council-gated full_main implementation",
        },
        "reactivation_criteria": [
            "if full-FiLM wins proxy, implement real-video full_main and run paired smoke on contest video",
            "if identity wins proxy, keep Z6 predictive-coding claim blocked and diagnose predictor/curriculum",
            "only promote/rank/kill after byte-closed paired contest CPU/CUDA exact eval",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown report."""

    lines = [
        "# L5 v2 Z6 identity-predictor disambiguator",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- probe_id: `{payload.get('probe_id')}`",
        f"- lane_id: `{payload.get('lane_id')}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- verdict: `{payload.get('verdict')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- ready_for_paid_dispatch: `false`",
        "- paradigm_claim_allowed: `false`",
        "",
        "This report is a Z6-specific probe surface. It can route the next "
        "engineering action, but it is not contest score evidence.",
    ]
    commands = payload.get("paired_smoke_commands")
    if isinstance(commands, list):
        lines.extend(["", "## Paired Smoke Commands"])
        for row in commands:
            if not isinstance(row, Mapping):
                continue
            command = row.get("command")
            rendered = " ".join(command) if isinstance(command, list) else ""
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- identity_predictor: `{row.get('identity_predictor')}`",
                    f"- stats_path: `{row.get('stats_path')}`",
                    "",
                    "```bash",
                    rendered,
                    "```",
                ]
            )
    stats = payload.get("source_stats")
    if isinstance(stats, list):
        lines.extend(["", "## Source Stats"])
        for row in stats:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('mode')}",
                    "",
                    f"- path: `{row.get('path')}`",
                    f"- sha256: `{row.get('sha256')}`",
                    f"- final_loss_proxy: `{row.get('final_loss_proxy')}`",
                    f"- final_recon: `{row.get('final_recon')}`",
                    f"- final_residual: `{row.get('final_residual')}`",
                    f"- archive_bytes: `{row.get('archive_bytes')}`",
                ]
            )
    deltas = payload.get("deltas")
    if isinstance(deltas, Mapping):
        lines.extend(["", "## Deltas"])
        for key, value in deltas.items():
            lines.append(f"- {key}: `{value}`")
    blockers = payload.get("blockers")
    if isinstance(blockers, list):
        lines.extend(["", "## Blockers"])
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    reactivation = payload.get("reactivation_criteria")
    if isinstance(reactivation, list):
        lines.extend(["", "## Reactivation Criteria"])
        for criterion in reactivation:
            lines.append(f"- {criterion}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--full-stats", type=Path, default=None)
    parser.add_argument("--identity-stats", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if (args.full_stats is None) != (args.identity_stats is None):
            raise ValueError("--full-stats and --identity-stats must be supplied together")
        if args.full_stats is None:
            payload = build_plan_payload(
                epochs=args.epochs,
                device=args.device,
                seed=args.seed,
            )
        else:
            full_stats_path = _resolve_repo_path(args.full_stats, repo_root=repo_root)
            identity_stats_path = _resolve_repo_path(
                args.identity_stats,
                repo_root=repo_root,
            )
            payload = evaluate_stats_pair(
                full_stats_path=full_stats_path,
                identity_stats_path=identity_stats_path,
                repo_root=repo_root,
            )
        output_json = _resolve_repo_path(args.output_json, repo_root=repo_root)
        output_md = _resolve_repo_path(args.output_md, repo_root=repo_root)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[z6-disambiguator] FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        "[z6-disambiguator] "
        f"verdict={payload['verdict']} "
        f"evidence_grade={payload['evidence_grade']} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
