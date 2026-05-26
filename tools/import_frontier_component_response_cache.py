#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Import false-authority component-response cache artifacts into a queue run.

The targeted component-correction queue can reuse prior local CPU advisory
artifacts instead of recomputing them. This tool is deliberately narrow: it
copies only the JSON artifacts that downstream MLX cache builders already
consume, and it refuses any payload that carries score/promotion/dispatch
authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY  # noqa: E402
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    require_no_truthy_authority_fields,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    repo_relative,
    sha256_file,
    write_json_artifact,
)

SCHEMA = "frontier_component_response_cache_import.v1"
HASHES_SCHEMA_VERSION = "mlx_scorer_input_cache_hashes.v1"


class ComponentResponseCacheImportError(ValueError):
    """Raised when a cache import would violate the queue contract."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ComponentResponseCacheImportError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComponentResponseCacheImportError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentResponseCacheImportError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_advisory(payload: dict[str, Any], *, source: Path) -> None:
    require_no_truthy_authority_fields(
        payload,
        context=f"component_response_cache_advisory:{source}",
    )
    if payload.get("score_axis") != "cpu_advisory":
        raise ComponentResponseCacheImportError(
            f"{source}: local CPU advisory score_axis must be cpu_advisory"
        )


def _validate_hashes(payload: dict[str, Any], *, source: Path) -> None:
    require_no_truthy_authority_fields(
        payload,
        context=f"component_response_cache_hashes:{source}",
    )
    if payload.get("schema_version") != HASHES_SCHEMA_VERSION:
        raise ComponentResponseCacheImportError(
            f"{source}: scorer cache hashes schema_version must be {HASHES_SCHEMA_VERSION}"
        )


def _write_imported_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    existing_sha = sha256_file(path) if path.is_file() else None
    write_json_artifact(
        path,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=existing_sha,
    )


def import_component_response_cache(
    *,
    source_local_cpu_advisory: Path,
    source_scorer_input_cache_hashes: Path,
    local_cpu_advisory_out: Path,
    scorer_input_cache_hashes_out: Path,
    role: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    advisory = _load_json_object(source_local_cpu_advisory, label="source advisory")
    hashes = _load_json_object(
        source_scorer_input_cache_hashes,
        label="source scorer input cache hashes",
    )
    _validate_advisory(advisory, source=source_local_cpu_advisory)
    _validate_hashes(hashes, source=source_scorer_input_cache_hashes)

    _write_imported_json(local_cpu_advisory_out, advisory, overwrite=overwrite)
    _write_imported_json(
        scorer_input_cache_hashes_out,
        hashes,
        overwrite=overwrite,
    )
    return {
        "schema": SCHEMA,
        "role": role,
        "reuse_mode": "import_false_authority_component_response_cache",
        "source_local_cpu_advisory": repo_relative(
            source_local_cpu_advisory,
            REPO_ROOT,
        ),
        "source_local_cpu_advisory_sha256": sha256_file(source_local_cpu_advisory),
        "source_scorer_input_cache_hashes": repo_relative(
            source_scorer_input_cache_hashes,
            REPO_ROOT,
        ),
        "source_scorer_input_cache_hashes_sha256": sha256_file(
            source_scorer_input_cache_hashes,
        ),
        "local_cpu_advisory_out": repo_relative(local_cpu_advisory_out, REPO_ROOT),
        "local_cpu_advisory_out_sha256": sha256_file(local_cpu_advisory_out),
        "scorer_input_cache_hashes_out": repo_relative(
            scorer_input_cache_hashes_out,
            REPO_ROOT,
        ),
        "scorer_input_cache_hashes_out_sha256": sha256_file(
            scorer_input_cache_hashes_out,
        ),
        "allowed_use": "queue_local_cpu_advisory_cache_reuse_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-local-cpu-advisory", type=Path, required=True)
    parser.add_argument("--source-scorer-input-cache-hashes", type=Path, required=True)
    parser.add_argument("--local-cpu-advisory-out", type=Path, required=True)
    parser.add_argument("--scorer-input-cache-hashes-out", type=Path, required=True)
    parser.add_argument("--role", choices=("candidate", "reference"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = import_component_response_cache(
            source_local_cpu_advisory=args.source_local_cpu_advisory,
            source_scorer_input_cache_hashes=args.source_scorer_input_cache_hashes,
            local_cpu_advisory_out=args.local_cpu_advisory_out,
            scorer_input_cache_hashes_out=args.scorer_input_cache_hashes_out,
            role=args.role,
            overwrite=args.overwrite,
        )
    except (ArtifactWriteError, ComponentResponseCacheImportError, OSError, ValueError) as exc:
        print(f"FATAL: component response cache import failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(report), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
