#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonicalize local training runtime profiles into planning signals.

This helper is intentionally not a dispatcher. It turns CPU/MLX/MPS/CUDA
training-smoke telemetry into durable, false-authority records that can be
fed into the optimizer candidate queue and local-vs-cloud cost model. Runtime
speed is useful signal; it is not a score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.local_training_runtime_profile import (  # noqa: E402
    SCHEMA as RUNTIME_PROFILE_SCHEMA,
)
from tac.optimization.local_training_runtime_profile import (  # noqa: E402
    LocalTrainingRuntimeProfileError,
    normalize_runtime_profile_observation,
    runtime_profile_summary_from_training_manifest,
)
from tac.optimizer.candidate_queue import build_candidate_queue  # noqa: E402

CANONICALIZATION_SCHEMA = "local_training_runtime_profile_canonicalization.v1"
SOURCE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "promotable",
    "dispatch_attempted",
    "gpu_launched",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "YAML input requires PyYAML; convert the manifest to JSON or "
                "install the repo's YAML extra."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(payload: Any, *, path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{path}: expected a JSON/YAML object")
    return payload


def _source_summary(
    *,
    path: Path,
    payload: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    schema = str(payload.get("schema") or payload.get("schema_version") or "unknown")
    if schema == RUNTIME_PROFILE_SCHEMA:
        profile = normalize_runtime_profile_observation(payload)
        summary: dict[str, Any] = {
            "schema": "trainer_runtime_profile_summary.v1",
            "profile_count": 1,
            "best_local_backend": profile["training_backend"],
            "best_timing_field": profile["timing_field"],
            "best_timing_value_seconds": profile["timing_value_seconds"],
            "kernel_fusion_strategy_ids": [
                profile["kernel_fusion"]["kernel_fusion_strategy_id"]
            ],
            "operator_mix_keys": list(
                profile["kernel_fusion"]["operator_mix"].keys()
            ),
            "profiles": [profile],
            "blockers": profile["blockers"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        }
    else:
        summary = runtime_profile_summary_from_training_manifest(payload)

    return {
        "path": _repo_rel(path, repo_root),
        "input_schema": schema,
        "source_authority": {
            key: payload[key]
            for key in SOURCE_AUTHORITY_FIELDS
            if key in payload
        },
        "runtime_profile_summary": summary,
        "profile_count": summary.get("profile_count", 0),
        "best_local_backend": summary.get("best_local_backend"),
        "best_timing_field": summary.get("best_timing_field"),
        "best_timing_value_seconds": summary.get("best_timing_value_seconds"),
        "blockers": summary.get("blockers", []),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
    }


def build_canonicalization(
    source_paths: list[Path],
    *,
    repo_root: Path,
    top_k: int | None = None,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    total_profiles = 0
    for path in source_paths:
        payload = _require_mapping(_load_payload(path), path=path)
        try:
            source = _source_summary(path=path, payload=payload, repo_root=repo_root)
        except LocalTrainingRuntimeProfileError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        sources.append(source)
        count = source.get("profile_count")
        if isinstance(count, int):
            total_profiles += count

    queue = build_candidate_queue(source_paths, repo_root=repo_root, top_k=top_k)
    return {
        "schema": CANONICALIZATION_SCHEMA,
        "tool": "tools/canonicalize_local_training_runtime_profile.py",
        "generated_at_utc": _utc_now(),
        "source_count": len(source_paths),
        "runtime_profile_count": total_profiles,
        "sources": sources,
        "candidate_queue": queue,
        "evidence_boundary": {
            "runtime_profile_is_cost_signal_not_score": True,
            "local_mlx_rows_are_research_signal_only": True,
            "next_gate": (
                "pair the runtime profile with same-seed quality, byte-closed "
                "archive export, PacketIR/compiler/runtime proof, then exact auth eval"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help=(
            "JSON/YAML trainer_runtime_profile_observation.v1 or a training "
            "manifest containing runtime_profile/runtime_profiles. May repeat."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=None,
        help="Optional optimizer_candidate_queue_v1 output path.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be >= 1 when provided")
    missing = [path for path in args.source if not path.is_file()]
    if missing:
        raise SystemExit(
            "source path(s) do not exist: "
            + ", ".join(path.as_posix() for path in missing)
        )

    payload = build_canonicalization(
        args.source,
        repo_root=args.repo_root,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.queue_output is not None:
        args.queue_output.parent.mkdir(parents=True, exist_ok=True)
        args.queue_output.write_text(
            json.dumps(
                payload["candidate_queue"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        f"wrote {args.output} "
        f"(runtime_profiles={payload['runtime_profile_count']}, "
        f"queue_top_k={payload['candidate_queue']['top_k_count']}, "
        f"dispatch_ready={payload['candidate_queue']['dispatch_ready_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
