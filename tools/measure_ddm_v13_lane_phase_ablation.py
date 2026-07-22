#!/usr/bin/env python3
"""Measure the isolated DDM v13 Lane phase-symbol path through the frozen receiver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_ddm_v9_carrier_compose import (  # noqa: E402
    _atomic_checkpoint,
    _bound_bytes,
    _bound_json,
    _objective,
    _portable_path,
    _publish_identical_or_new,
    _v13_lane_program_rows,
)

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_v13_lane_phase_ablation import (  # noqa: E402
    RESULT_SCHEMA,
    DDMV13LanePhaseAblationConfigV1,
    phase_only_knots,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    EVIDENCE_AXIS,
    DirectDescriptionV13WorldsheetPredictorConfigV1,
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
    DirectDescriptionError,
    _read_regular_file_once,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_polytope_membership import _load_segnet_oracle  # noqa: E402


def _bridge_row(name: str, archive: bytes, receiver: Any, bridge: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "rung": name,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "d_seg": bridge["segmentation"]["d_seg"],
        "d_pose": bridge["pose"]["d_pose"],
        "lane_conditional_d_seg": bridge["segmentation"]["strata"]["target_class"]["Lane"]["d_seg"],
        "movable_conditional_d_seg": bridge["segmentation"]["strata"]["target_class"]["Movable"]["d_seg"],
        "objective_advisory": _objective(
            len(archive),
            bridge["segmentation"]["d_seg"],
            bridge["pose"]["d_pose"],
        ),
        "byte_streams": recursive_carrier_byte_rows(archive),
        "receiver_custody": dict(receiver.custody),
        "elapsed_seconds": f"{elapsed:.6f}",
    }


def run(config: DDMV13LanePhaseAblationConfigV1, output_directory: Path, semantic_argv: list[str]) -> Path:
    root = output_directory
    storage = _storage_preflight(root.resolve())
    storage["output_tier"] = _portable_path(Path(str(storage.get("output_tier", root))))
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v13_lane_phase_ablation_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed phase-ablation receipt typed-config hash differs")
        print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
        return receipt_path

    v13_config_bytes = _bound_bytes(Path(config.v13_config_path), config.v13_config_sha256, "v13 config")
    v13_config = DirectDescriptionV13WorldsheetPredictorConfigV1.model_validate_json(v13_config_bytes)
    if (v13_config.pair_start, v13_config.pair_count) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("phase-ablation window differs from bound v13 config")
    v13_receipt = _bound_json(Path(config.v13_receipt_path), config.v13_receipt_sha256, "v13 receipt")
    if v13_receipt.get("typed_config_sha256") != v13_config.typed_config_hash():
        raise DirectDescriptionError("phase-ablation v13 receipt/config custody differs")

    predictor_archive = _bound_bytes(
        Path(v13_config.predictor_archive_path),
        v13_config.predictor_archive_sha256,
        "v13-bound predictor",
    )
    inventory = v13_receipt["natural_production_inventory"]
    programs, raw_knots = _v13_lane_program_rows(inventory)
    phase_knots = phase_only_knots(raw_knots)
    if not programs or not phase_knots:
        raise DirectDescriptionError("phase ablation requires nonempty program and phase-symbol rows")

    base_archive, _ = compile_carrier_compose_archive(predictor_archive, obligation_vocabulary=True)
    predict_archive, _ = compile_carrier_compose_archive(predictor_archive, lane_programs=programs)
    symbols_archive, _ = compile_carrier_compose_archive(
        predictor_archive,
        lane_programs=programs,
        lane_knots=phase_knots,
    )
    archives = {
        "phase_predict": (predict_archive, receive_carrier_compose_archive(predict_archive)),
        "phase_symbols": (symbols_archive, receive_carrier_compose_archive(symbols_archive)),
    }
    base_sha = next(row["archive"]["sha256"] for row in v13_receipt["composition_ladder"] if row["rung"] == "base")
    if _sha256(base_archive) != base_sha:
        raise DirectDescriptionError("phase-ablation base archive differs from bound v13 control")
    for name, (archive, _receiver) in archives.items():
        _publish_identical_or_new(
            root / f"ddm_v13_{name}_n{config.pair_count}.not_a_candidate.zip.receipt-bytes",
            archive,
        )
    _atomic_checkpoint(
        root / "stage_checkpoints" / "01_phase_derivation_inventory.json",
        {
            "schema": "ddm_v13_lane_phase_derivation_inventory.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "program_count": len(programs),
            "raw_knot_count": len(raw_knots),
            "phase_symbol_count": len(phase_knots),
            "geometry_or_width_symbol_fields_present": False,
            "programs": [asdict(row) for row in programs],
            "phase_knots": [asdict(row) for row in phase_knots],
            "ranker": config.ranker,
        },
    )
    _atomic_checkpoint(
        root / "stage_checkpoints" / "02_phase_receiver_archives.json",
        {
            "schema": "ddm_v13_lane_phase_receiver_archives.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "base_archive_sha256": _sha256(base_archive),
            "archives": {
                name: {"bytes": len(archive), "sha256": _sha256(archive), "receiver_closed": True}
                for name, (archive, _receiver) in archives.items()
            },
        },
    )

    v6_receipt = _bound_json(Path(v13_config.v6_receipt_path), v13_config.v6_receipt_sha256, "v6 receipt")
    v6_typed = v6_receipt["typed_config"]
    v5_receipt = _bound_json(Path(v6_typed["v5_receipt_path"]), v6_typed["v5_receipt_sha256"], "v5 receipt")
    v5_typed = v5_receipt["typed_config"]
    target_receipt = load_target_receipt(Path(v5_typed["target_receipt_path"]), v5_typed["target_receipt_sha256"])
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("phase-ablation evaluator cache size/path custody mismatch")
    cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    cached_margins = open_stored_npy_memmap(cache_path, "margins")
    cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    segnet, segnet_custody = _load_segnet_oracle(Path(v13_config.upstream_root), threads=config.scorer_threads)
    posenet, posenet_custody = _load_posenet_oracle(Path(v13_config.upstream_root), threads=config.scorer_threads)
    measured = 0
    for name, (archive, receiver) in archives.items():
        checkpoint = root / "stage_checkpoints" / "measurements" / f"{name}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if row.get("typed_config_sha256") != config.typed_config_hash() or row.get("archive_sha256") != _sha256(
                archive
            ):
                raise DirectDescriptionError("phase-ablation measurement checkpoint identity differs")
            continue
        started = time.perf_counter()
        bridge = _measure_evaluator_bridge(
            receiver,
            pair_start=config.pair_start,
            cached_lstars=cached_lstars,
            cached_margins=cached_margins,
            cached_poses=cached_poses,
            segnet_oracle=segnet,
            posenet_oracle=posenet,
            batch_size=config.scorer_batch_size,
        )
        _atomic_checkpoint(
            checkpoint,
            {
                "schema": "ddm_v13_lane_phase_measurement.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "archive_sha256": _sha256(archive),
                "row": _bridge_row(name, archive, receiver, bridge, time.perf_counter() - started),
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            },
        )
        measured += 1
        if measured >= config.max_measurements_per_invocation:
            break
    missing = [name for name in archives if not (root / "stage_checkpoints" / "measurements" / f"{name}.json").exists()]
    if missing:
        print(json.dumps({"complete": False, "measured": measured, "next_rung": missing[0]}))
        return root / "stage_checkpoints" / "measurements" / f"{missing[0]}.json"

    rows = {
        name: json.loads(_read_regular_file_once(root / "stage_checkpoints" / "measurements" / f"{name}.json"))["row"]
        for name in archives
    }
    predict = rows["phase_predict"]
    symbols = rows["phase_symbols"]
    delta_dseg = float(symbols["d_seg"]) - float(predict["d_seg"])
    delta_lane = float(symbols["lane_conditional_d_seg"]) - float(predict["lane_conditional_d_seg"])
    delta_score = float(symbols["objective_advisory"]) - float(predict["objective_advisory"])
    verdict = "MEASURED_PHASE_SYMBOLS_HELP" if delta_dseg < 0.0 else "MEASURED_PHASE_SYMBOLS_DO_NOT_HELP"
    receipt = {
        "schema": RESULT_SCHEMA,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "v13_control": {"receipt_path": config.v13_receipt_path, "receipt_sha256": config.v13_receipt_sha256},
        "derivation": {
            "program_count": len(programs),
            "phase_symbol_count": len(phase_knots),
            "raw_knot_count": len(raw_knots),
            "geometry_or_width_symbol_fields_present": False,
            "successor_scope": config.successor_scope,
        },
        "rungs": rows,
        "phase_symbol_delta": {
            "archive_bytes": int(symbols["archive_bytes"]) - int(predict["archive_bytes"]),
            "d_seg": f"{delta_dseg:.12f}",
            "lane_conditional_d_seg": f"{delta_lane:.12f}",
            "joint_objective": f"{delta_score:.12f}",
        },
        "g2_corrections_consumed": {
            "energy_is_not_score": True,
            "ranker": config.ranker,
            "lane_topology": "birth_dominated_dash_comb_plus_events_confirmed",
            "movable_topology": "persistence_dominated_rel_velocity_plus_deviations_required",
            "xi_only_movable_predictive": False,
            "phase_hypothesis_before_this_receipt": "BLOCKED_UNTESTED",
            "phase_hypothesis_after_this_receipt": verdict,
        },
        "fail_closed_mutation_proof": prove_carrier_archive_fail_closed(symbols_archive),
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "cache_path": str(cache_path),
            "cache_bytes": target_receipt.source_cache.bytes,
            "cache_sha256": target_receipt.source_cache.sha256,
        },
        "storage_preflight": storage,
        "verdict": verdict,
        "verdict_scope": (
            "INSTANCE raw q8 phase deviations over the pre-addendum inherited Lane chart; "
            "the anisotropic-volatility AR1-whitened BEV successor remains unmeasured and open"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    _publish_identical_or_new(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"complete": True, "receipt": str(receipt_path), "verdict": verdict}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = _read_regular_file_once(args.config)
    config = DDMV13LanePhaseAblationConfigV1.model_validate_json(config_bytes)
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
