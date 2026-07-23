#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the SHA-pinned Phase-0 AT1 atlas receipt without scorer execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tac.ddm_costate_organ import build_live_ddm_costate, discover_sources
from tac.optimization.scorer_analytic_atlas import (
    SourceHashStamp,
    build_manifest,
    build_r_null_band_certificate,
    build_sdwl1_e2_coordinate_bridge,
)
from tac.optimization.scorer_module_inventory import (
    canonical_json_bytes,
    read_and_validate_receipt,
    sha256_file,
)

SCHEMA = "ddm_at1_scorer_analytic_atlas_receipt.v1"


class AtlasReceiptError(ValueError):
    """The atlas receipt could not close its source custody."""


def _identity(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": (
            str(resolved.relative_to(root.resolve()))
            if resolved.is_relative_to(root.resolve())
            else str(resolved)
        ),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_wrapped(path: Path, *, expected_schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"body", "body_sha256"}:
        raise AtlasReceiptError(f"{path}: wrapped receipt fields changed")
    body = value["body"]
    if body.get("schema") != expected_schema:
        raise AtlasReceiptError(f"{path}: schema mismatch")
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if actual != value["body_sha256"]:
        raise AtlasReceiptError(f"{path}: body SHA-256 mismatch")
    return value


def _stamp(
    *,
    source_id: str,
    path: Path,
    root: Path,
    validity_horizon: str = "exact input hash equality; rederive on mismatch",
) -> SourceHashStamp:
    row = _identity(path, root=root)
    return SourceHashStamp(
        source_id=source_id,
        path=str(row["path"]),
        sha256=str(row["sha256"]),
        bytes=int(row["bytes"]),
        validity_horizon=validity_horizon,
    )


def _write_once(path: Path, receipt: dict[str, Any]) -> None:
    payload = canonical_json_bytes(receipt) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--divergence", required=True, type=Path)
    parser.add_argument("--resize-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    inventory = read_and_validate_receipt(args.inventory)
    divergence = _load_wrapped(
        args.divergence,
        expected_schema="ddm_at1_scorer_semantic_divergence_receipt.v1",
    )
    sources = discover_sources(root)
    for name in ("dv2", "e2"):
        if not sources[name]["available"]:
            raise AtlasReceiptError(f"required bridge source unavailable: {name}")
    dv2_path = root / sources["dv2"]["path"]
    e2_path = root / sources["e2"]["path"]
    bridge = build_sdwl1_e2_coordinate_bridge(
        source_hashes=(
            _stamp(source_id="sdwl1", path=dv2_path, root=root),
            _stamp(source_id="e2_manifest", path=e2_path, root=root),
        )
    )
    resize_factor = build_r_null_band_certificate(
        resize_authority=_stamp(
            source_id="resize_full_kernel_580",
            path=args.resize_receipt,
            root=root,
        ),
        requested_band_ids=(
            "horizontal_nyquist",
            "vertical_nyquist",
            "diagonal_nyquist",
        ),
    )
    factor_manifest = build_manifest(
        factors=(resize_factor,),
        pools=(),
        materialization_status=(
            "PHASE0_SCHEMA_READY; ONE_REUSED_580_FACTOR; LOCKED_LIBRARY_SOURCE_"
            "DRIFT_BLOCKS_NETWORK_CLOSED_FORMS; N600_GAZE_NOT_MATERIALIZED"
        ),
    )
    organ = build_live_ddm_costate(repo_root=root)
    if not organ.get("available"):
        raise AtlasReceiptError(f"live costate organ unavailable: {organ.get('status')}")

    inventory_drift = inventory["body"]["source_strata"][
        "B_imported_library_sources"
    ]["version_drift"]
    body = {
        "schema": SCHEMA,
        "created_at_utc": args.created_at_utc,
        "research_only": True,
        "score_claim": False,
        "execution_allowed": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "first_rung": True,
        "pair_count_required": 600,
        "n600_or_not_evidence": True,
        "phase0": {
            "inventory": {
                **_identity(args.inventory, root=root),
                "body_sha256": inventory["body_sha256"],
                "binding_status": inventory["body"]["analytic_binding"]["status"],
                "version_drift_package_count": len(inventory_drift),
            },
            "divergence": {
                **_identity(args.divergence, root=root),
                "body_sha256": divergence["body_sha256"],
                "consumer_gate": divergence["body"]["consumer_gate"]["status"],
            },
            "frozen_graph_counts": {
                network: {
                    "modules": inventory["body"]["networks"][network][
                        "module_count"
                    ],
                    "checkpoint_tensors": inventory["body"]["source_strata"][
                        "C_loaded_artifacts"
                    ][network]["tensor_count"],
                    "checkpoint_match": inventory["body"]["source_strata"][
                        "C_loaded_artifacts"
                    ][network]["module_state_match"]["status"],
                    "mechanisms": inventory["body"]["networks"][network][
                        "mechanism_summary"
                    ],
                }
                for network in ("posenet", "segnet")
            },
        },
        "atlas": {
            "typed_factor_manifest": factor_manifest,
            "materialized_factor_counts": {
                "reused_resize_certificate": 1,
                "network_closed_forms": 0,
                "gaze_fields": 0,
                "jacobian_factor_shards": 0,
                "axis_projections": 0,
                "nonadditive_pools": 0,
            },
            "materialization_gate": {
                "status": "BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED",
                "why": (
                    "The observed local library graph differs from the lock; "
                    "network closed forms cannot cite locked stratum-B bytes."
                ),
                "next_stage": (
                    "Materialize exact lock-selected sources, require zero drift, "
                    "then emit hash-stamped weight-derived factor shards and n600 "
                    "gaze/Jacobian checkpoints."
                ),
                "nonadditive_pool_policy": (
                    "pool type and KKT validation are implemented and tested; no "
                    "empty or invented pool is emitted before competing factors exist"
                ),
                "amplitude_policy": (
                    "every amplitude factor is rejected unless it carries a complete "
                    "uint8-surviving R projection receipt"
                ),
            },
            "coordinate_bridge": bridge,
            "validation_harness": {
                "status": "WAITING_SN1_MEASURED_RESIDUALS",
                "contract": (
                    "join measured and derived factors only under exact factor ID, "
                    "pair range, tensor shape/dtype, and source-hash equality"
                ),
            },
        },
        "lambda_unification": {
            "single_producer": organ["lambda"]["producer"],
            "producer_schema": organ["lambda"]["producer_schema"],
            "producer_status": organ["lambda"]["producer_status"],
            "producer_content_sha256": organ["lambda"][
                "producer_content_sha256"
            ],
            "pair_rows": len(organ["lambda"]["pair_rows"]),
            "site_rows": len(organ["lambda"]["site_rows"]),
            "missing_exact_pair_lambda_count": organ["lambda"][
                "missing_exact_pair_lambda_count"
            ],
            "missing_rows_are_counted_inert": organ["lambda"][
                "unconsumed_missing_pairs_counted_inert"
            ],
            "backtest": organ["lambda"]["backtest"],
            "organ_status": organ["status"],
            "organ_actuation": organ["actuation"],
            "organ_source_hashes": organ["source_custody"]["input_hashes"],
        },
        "triality": {
            "dsl": {
                "schema_ids": [
                    "ddm_scorer_analytic_atlas.v2",
                    "ddm_scorer_module_inventory.v1",
                    "ddm_scorer_analytic_lambda_bundle.v1",
                    "sdwl1_e2_coordinate_bridge.v1",
                ],
                "typed_module": "src/tac/optimization/scorer_analytic_atlas.py",
                "inventory_module": (
                    "src/tac/optimization/scorer_module_inventory.py"
                ),
            },
            "dag": (
                "upstream closure -> inventory gate -> frozen factors -> gaze/J "
                "composition -> axes/pools -> atlas lambda producer -> costate "
                "organ consumer/controller"
            ),
            "equations": [
                "lambda_k=dS/dz_k",
                "dS/dx=J_1^T...J_K^T lambda_K",
                "S=100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489",
                "BN(x)=gamma*(x-mu)/sqrt(var+eps)+beta",
                "SE(z)=z*sigmoid(W2*silu(W1*GAP(z)+b1)+b2)",
            ],
        },
        "pointer_delta": {
            "pointer_moved": False,
            "score_claim": False,
            "reason": (
                "research-only Phase-0 structure and partial exact-v19 backtest; "
                "no contest-CPU/CUDA replay or candidate archive"
            ),
        },
        "source_files": {
            path: _identity(root / path, root=root)
            for path in (
                "src/tac/optimization/scorer_analytic_atlas.py",
                "src/tac/optimization/scorer_module_inventory.py",
                "src/tac/ddm_costate_organ.py",
                "tools/build_scorer_module_inventory.py",
                "tools/build_ddm_at1_divergence_receipt.py",
                "tools/build_ddm_at1_atlas_receipt.py",
            )
        },
    }
    receipt = {
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    _write_once(args.output, receipt)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "body_sha256": receipt["body_sha256"],
                "producer_content_sha256": body["lambda_unification"][
                    "producer_content_sha256"
                ],
                "materialization_gate": body["atlas"]["materialization_gate"][
                    "status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
