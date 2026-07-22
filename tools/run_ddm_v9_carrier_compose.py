#!/usr/bin/env python3
"""Build and advisory-measure one receiver-closed DDM V9 carrier archive."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    EVIDENCE_AXIS,
    RESULT_SCHEMA,
    DirectDescriptionV9CarrierComposeConfigV1,
    compile_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    _load_posenet_oracle,
    _measure_evaluator_bridge,
    _storage_preflight,
)
from tac.optimization.direct_description_measurement_ladder import load_target_receipt  # noqa: E402
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    POINTER_SCORE_TEXT,
    SOURCE_BYTES,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_polytope_membership import _load_segnet_oracle  # noqa: E402


def _bound_bytes(path: Path, digest: str, name: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 mismatch")
    return payload


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(path, digest, name))
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must contain one JSON object")
    return value


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    encoded = rfc8785_canonicalize(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != encoded:
            raise DirectDescriptionError(f"checkpoint exists with different bytes: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _objective(archive_bytes: int, d_seg: str, d_pose: str) -> str:
    value = 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + 25.0 * archive_bytes / SOURCE_BYTES
    return f"{value:.12f}"


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _candidate_row(v6_receipt: dict[str, Any], config: DirectDescriptionV9CarrierComposeConfigV1) -> dict[str, Any]:
    for row in v6_receipt.get("candidates", ()):  # fixed AR(1)/hold24 is candidate index 1 by sealed v6 order.
        archive = row.get("archive", {})
        if (
            row.get("candidate_index") == 1
            and archive.get("path") == config.predictor_archive_path
            and archive.get("sha256") == config.predictor_archive_sha256
        ):
            return row
    raise DirectDescriptionError("typed predictor is not the bound v6 fixed_ar1_hold24 candidate")


def run(config: DirectDescriptionV9CarrierComposeConfigV1, output_directory: Path, semantic_argv: list[str]) -> Path:
    root = output_directory
    storage = _storage_preflight(root.resolve())
    output_tier = Path(str(storage.get("output_tier", root)))
    storage["output_tier"] = _portable_path(output_tier)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v9_carrier_compose_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed receipt typed-config hash differs")
        archive = _bound_bytes(Path(receipt["archive"]["path"]), receipt["archive"]["sha256"], "completed v9 archive")
        receive_carrier_compose_archive(archive)
        print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
        return receipt_path

    v6_receipt = _bound_json(Path(config.v6_receipt_path), config.v6_receipt_sha256, "v6 receipt")
    if v6_receipt.get("schema") != "direct_description_dseg_bridge_amortize.v1":
        raise DirectDescriptionError("input is not the governed v6 receipt")
    v6_typed = v6_receipt.get("typed_config", {})
    if (v6_typed.get("pair_start"), v6_typed.get("pair_count")) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v9 window differs from bound v6 window")
    _candidate_row(v6_receipt, config)
    predictor_archive = _bound_bytes(
        Path(config.predictor_archive_path), config.predictor_archive_sha256, "v6 predictor archive"
    )

    started = time.perf_counter()
    archive, homes = compile_carrier_compose_archive(predictor_archive, config.symbols())
    receiver = receive_carrier_compose_archive(archive)
    fail_closed = prove_carrier_archive_fail_closed(archive)
    build_seconds = time.perf_counter() - started
    archive_path = root / f"ddm_v9_carrier_compose_n{config.pair_count}.not_a_candidate.zip.receipt-bytes"
    _publish_new_bytes(archive_path, archive)
    _atomic_checkpoint(
        root / "stage_checkpoints" / "01_receiver_closed_build.json",
        {
            "schema": "ddm_v9_carrier_compose_stage_checkpoint.v1",
            "stage": "receiver_closed_build",
            "typed_config_sha256": config.typed_config_hash(),
            "archive": {"path": _portable_path(archive_path), "bytes": len(archive), "sha256": _sha256(archive)},
            "receiver_custody": dict(receiver.custody),
        },
    )

    v5_receipt = _bound_json(Path(v6_typed["v5_receipt_path"]), v6_typed["v5_receipt_sha256"], "v6-bound v5 receipt")
    v5_typed = v5_receipt.get("typed_config", {})
    target_receipt = load_target_receipt(Path(v5_typed["target_receipt_path"]), v5_typed["target_receipt_sha256"])
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("frozen scorer cache is unavailable")
    cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    cached_margins = open_stored_npy_memmap(cache_path, "margins")
    cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    measured_at = time.perf_counter()
    bridge = _measure_evaluator_bridge(
        receiver,  # duck-typed: z + render_pairs are the governed bridge surface.
        pair_start=config.pair_start,
        cached_lstars=cached_lstars,
        cached_margins=cached_margins,
        cached_poses=cached_poses,
        segnet_oracle=segnet_oracle,
        posenet_oracle=posenet_oracle,
        batch_size=config.scorer_batch_size,
    )
    measure_seconds = time.perf_counter() - measured_at
    d_seg = bridge["segmentation"]["d_seg"]
    d_pose = bridge["pose"]["d_pose"]
    class_rows = bridge["segmentation"]["strata"]["target_class"]
    byte_rows = recursive_carrier_byte_rows(archive)
    for row in byte_rows:
        stratum = row["stratum"]
        if stratum in class_rows:
            row["d_seg"] = class_rows[stratum]["d_seg"]
        elif stratum == "xi/Pose6":
            row["d_seg"] = d_seg
        else:
            row["d_seg"] = None
        row["d_pose"] = d_pose if stratum != "chart_symbol_refinement" or row["nested_unique_home_bytes"] else None
        row["causal_attribution_scope"] = (
            "target-class conditional composite error; d_pose shared-composite, not leave-one-out"
        )
    objective = _objective(len(archive), d_seg, d_pose)
    under_box = len(archive) <= 154_600 and float(d_seg) <= 0.00116
    verdict = (
        "ADVISORY_INSTANCE_MEETS_SUB015_BOX_NOT_PROMOTABLE"
        if under_box
        else "ADVISORY_INSTANCE_FAILS_SUB015_BOX_FORMULATION_OPEN"
    )
    receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "lane_id": "ddm_v9_carrier_compose_byteclose",
        "tasks": [603, 613],
        "run_id": config.run_id,
        "seed": config.seed,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "archive": {
            "path": _portable_path(archive_path),
            "bytes": len(archive),
            "sha256": _sha256(archive),
            "member_homes": list(homes),
            "parse_reencode_identical": True,
            "receiver_closed": True,
            "all_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
        },
        "per_stratum": byte_rows,
        "bridge": bridge,
        "objective_advisory": {
            "score": objective,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": len(archive),
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
        },
        "boxes": {
            "sub_0p15": {"max_bytes": 154600, "max_d_seg": "0.001160000000", "met": under_box},
            "pointer_knee_bytes": 216300,
        },
        "correction": {
            "symbols": len(config.symbols()),
            "policy": config.correction_policy,
            "pixel_residual_present": False,
            "admission": "empty unless hard-oracle improvement is preselected by typed config",
            "fisher_margin_curvature_ranker": "required upstream for nonempty symbols; no blanket fixes",
        },
        "receiver_custody": dict(receiver.custody),
        "fail_closed_mutation_proof": fail_closed,
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "receipt_path": v5_typed["target_receipt_path"],
            "receipt_sha256": v5_typed["target_receipt_sha256"],
            "cache_path": str(cache_path),
            "cache_bytes": target_receipt.source_cache.bytes,
            "cache_sha256": target_receipt.source_cache.sha256,
        },
        "wallclock": {
            "build_seconds": f"{build_seconds:.6f}",
            "measure_seconds": f"{measure_seconds:.6f}",
            "n600_projection_seconds": f"{(build_seconds + measure_seconds) * 600 / config.pair_count:.6f}",
            "n600_status": "WALLCLOCK_PROJECTION_ONLY_NOT_RUN",
        },
        "storage_preflight": storage,
        "resume": {
            "policy": config.checkpoint_policy,
            "stage_checkpoint": _portable_path(root / "stage_checkpoints" / "01_receiver_closed_build.json"),
            "all_preserved": True,
        },
        "verdict": verdict,
        "verdict_scope": (
            "This exact v6-fixed predictor plus five settled carrier payloads and empty-or-explicit G2CS1 "
            "chart refinement on the stated bridge window only. Failure does not close joint multicoefficient "
            "chart solves, xi-transported birth/death events, corrected inner-Jacobian realization, or the V9 family."
        ),
        "blocker_delta": (
            "Receiver closure and per-stratum byte/Seg/Pose accounting are discharged. Remaining primary DOF is "
            "a Fisher-margin/curvature-ranked joint chart-symbol plus birth/death-event solve with pose-tube "
            "admission; current inherited Pose6 stream is sole-owner but not yet triple-used as carrier transport."
        ),
        "stores_consulted": [
            config.v6_receipt_path,
            config.predictor_archive_path,
            v6_typed["v5_receipt_path"],
            v5_typed["target_receipt_path"],
            str(cache_path),
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _atomic_checkpoint(
        root / "stage_checkpoints" / "02_frozen_scorer_measurement.json",
        {
            "schema": "ddm_v9_carrier_compose_stage_checkpoint.v1",
            "stage": "frozen_scorer_measurement",
            "typed_config_sha256": config.typed_config_hash(),
            "archive_sha256": _sha256(archive),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        },
    )
    _publish_new_bytes(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "receipt": str(receipt_path), "verdict": verdict}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DirectDescriptionV9CarrierComposeConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = [
        _portable_path(Path(__file__)),
        "--config",
        str(args.config),
        "--output-directory",
        str(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
