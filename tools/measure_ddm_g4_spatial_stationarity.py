#!/usr/bin/env python3
"""Measure DDM G4 n600 flip stationarity from settled G3/V12 caches.

The command is local, scorer-free, chunk-bounded, restartable, and advisory.
It writes bulky maps/checkpoints to the SSD output tier and compact receipts to
the repository.  It never emits or evaluates an archive candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_g3_score_atlas import reconstruct_v12_state  # noqa: E402
from tac.optimization.ddm_g4_spatial_stationarity import (  # noqa: E402
    AXIS,
    HEIGHT,
    N_PAIRS,
    STRATA,
    WIDTH,
    DdmG4SpatialStationarityConfigV1,
    build_opportunities,
    build_xi_tracks,
    concentration_fractions,
    free_context_measurement,
    recurrence_histogram,
    sha256_file,
    stationarity_decomposition,
    stratum_masks,
    transition_codes,
    typed_ledger_rows,
)

MIN_FREE_BYTES = 64 << 20
POINTER = "0.1910828242 [contest-CPU]"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _canonical_json(payload))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(_canonical_json(row) for row in rows)
    _atomic_bytes(path, payload)


def _output_row(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _load_bound_json(path_value: str, expected_sha: str, label: str) -> dict[str, Any]:
    path = _resolve(path_value)
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{label} SHA mismatch: expected {expected_sha}, got {actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _checkpoint(output: Path, step: int, name: str, payload: dict[str, Any]) -> Path:
    path = output / "stage_checkpoints" / f"{step:02d}_{name}.json"
    _atomic_json(
        path,
        {
            "schema": "ddm_g4_spatial_stationarity_stage_checkpoint.v1",
            "step": step,
            "stage": name,
            **payload,
        },
    )
    return path


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_g3_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "ddm_g3_score_atlas_receipt.v1":
        raise RuntimeError("bound G3 receipt schema changed")
    atlas = next(
        (row for row in receipt.get("outputs", []) if str(row.get("path", "")).endswith("ddm_g3_score_atlas_n600.jsonl")),
        None,
    )
    if atlas is None:
        raise RuntimeError("G3 receipt does not bind the n600 atlas JSONL")
    path = Path(str(atlas["path"]))
    if not path.is_file() or path.stat().st_size != int(atlas["bytes"]) or sha256_file(path) != atlas["sha256"]:
        raise RuntimeError("G3 atlas bytes changed")
    return atlas


def _plot_map(path: Path, values: np.ndarray, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 6), dpi=144)
    image = axis.imshow(np.log1p(np.asarray(values, dtype=np.float64)), cmap="magma", interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("x (512-wide scorer grid)")
    axis.set_ylabel("y (384-high scorer grid)")
    fig.colorbar(image, ax=axis, label="log(1 + flip events)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": "pact-ddm-g4"})
    plt.close(fig)


def _compute_counts(
    target: np.ndarray, predicted: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    transition_counts = np.empty((25, HEIGHT, WIDTH), dtype=np.uint16)
    for code in range(25):
        transition_counts[code] = np.count_nonzero(codes == code, axis=0).astype(np.uint16)
    flip_frequency = np.count_nonzero(predicted != target, axis=0).astype(np.uint16)
    strata = {name: np.zeros((HEIGHT, WIDTH), dtype=np.uint32) for name in STRATA}
    for pair_index in range(N_PAIRS):
        flip = predicted[pair_index] != target[pair_index]
        masks = stratum_masks(target[pair_index], predicted[pair_index])
        for name, mask in masks.items():
            strata[name] += flip & mask
    if int(flip_frequency.sum()) != int(transition_counts[[code for code in range(25) if code // 5 != code % 5]].sum()):
        raise RuntimeError("flip frequency and transition counts do not close")
    return transition_counts, flip_frequency, strata


def _compact_summary(
    concentration: dict[str, Any],
    decomposition: dict[str, Any],
    free_context: dict[str, Any],
    opportunities: list[dict[str, Any]],
    xi_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "ddm_g4_spatial_stationarity_summary.v1",
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "concentration": concentration,
        "stationarity_decomposition": decomposition,
        "xi_registration": xi_summary,
        "free_context": free_context,
        "top5_amortization_opportunities": opportunities[:5],
        "independent_physical_bev_fraction": None,
        "bev_blocker": (
            "no independently observed pixel homography/liveCalibration custody; xi fraction is a "
            "target-cache metric-Pose6 G1 translation-only proxy whose transport side-information is not free"
        ),
        "receiver_realized_delta_d_seg": None,
        "blocker_delta_vs_603": (
            "exact n600 spatial concentration, image recurrence, xi-proxy recurrence, real-coded one-time "
            "field prices, and free-context gains are measured; current predictor-margin custody, total xi "
            "side-information bytes, receiver-realized RGB delta, and independent physical-BEV registration "
            "remain owed"
        ),
    }


def _resume_receipt(path: Path, typed_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("typed_config_sha256") != typed_hash:
        raise RuntimeError("existing G4 receipt belongs to another typed config")
    custody_rows = [
        *receipt.get("outputs", []),
        *receipt.get("compact_outputs", []),
        *receipt.get("implementation_custody", {}).get("source_files", []),
    ]
    for row in custody_rows:
        output = Path(row["path"])
        if (
            not output.is_file()
            or output.stat().st_size != int(row["bytes"])
            or sha256_file(output) != row["sha256"]
        ):
            raise RuntimeError(f"resume output custody failed: {output}")
    return receipt


def run(config: DdmG4SpatialStationarityConfigV1, *, resume: bool, argv: list[str]) -> Path:
    output = _resolve(config.output_directory)
    compact = _resolve(config.compact_receipt_directory)
    receipt_path = compact / "ddm_g4_spatial_stationarity_receipt.json"
    if resume:
        receipt = _resume_receipt(receipt_path, config.typed_hash())
        if receipt is not None:
            print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
            return receipt_path

    output.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < MIN_FREE_BYTES:
        raise RuntimeError("SSD storage preflight failed")
    g3 = _load_bound_json(config.g3_receipt_path, config.g3_receipt_sha256, "G3 receipt")
    atlas_row = _validate_g3_receipt(g3)
    v12 = _load_bound_json(config.v12_receipt_path, config.v12_receipt_sha256, "V12 receipt")
    source = {
        "g3_receipt": {
            "path": config.g3_receipt_path,
            "sha256": config.g3_receipt_sha256,
            "atlas": atlas_row,
        },
        "v12_receipt": {"path": config.v12_receipt_path, "sha256": config.v12_receipt_sha256},
        "g3_source_reuse": "settled cells/counts are reconstructed; scorer outputs are not recomputed",
    }
    _checkpoint(
        output,
        0,
        "sources_validated",
        {
            "typed_config_sha256": config.typed_hash(),
            "source_custody": source,
            "storage_preflight": {"path": str(output), "required_free_bytes": MIN_FREE_BYTES, "status": "PASS"},
            "status": "complete",
        },
    )

    state = reconstruct_v12_state(REPO, v12, n_pairs=N_PAIRS)
    target_cache = Path(v12["target_custody"]["cache_path"])
    if not target_cache.is_file() or target_cache.stat().st_size != int(v12["target_custody"]["cache_bytes"]):
        raise RuntimeError("V12 target cache custody changed")
    target = open_stored_npy_memmap(target_cache, "lstars")
    predicted = np.asarray(state.final_cells, dtype=np.uint8)
    if target.shape != (N_PAIRS, HEIGHT, WIDTH) or predicted.shape != target.shape:
        raise RuntimeError("n600 cell geometry changed")
    codes = transition_codes(predicted, target)
    transition_counts, flip_frequency, stratum_frequency = _compute_counts(target, predicted, codes)
    expected_flips = round(
        g3["summary"]["global_reconstruction"]["d_seg"] * N_PAIRS * HEIGHT * WIDTH
    )
    if int(flip_frequency.sum()) != expected_flips:
        raise RuntimeError("G4 reconstructed flip total does not match G3")
    recurrence_path = output / "stage_checkpoints" / "01_recurrence_arrays.npz"
    _atomic_npz(
        recurrence_path,
        transition_counts=transition_counts,
        flip_frequency=flip_frequency,
        **{f"stratum_{name}": value for name, value in stratum_frequency.items()},
    )
    concentration = concentration_fractions(flip_frequency)
    recurrence = recurrence_histogram(transition_counts)
    _checkpoint(
        output,
        1,
        "recurrence_complete",
        {
            "recurrence_arrays": _output_row(recurrence_path),
            "concentration": concentration,
            "status": "complete",
        },
    )

    registered_pose6 = np.asarray(open_stored_npy_memmap(target_cache, "gt_poses"), dtype=np.float64)
    xi_tracks, xi_membership, xi_summary = build_xi_tracks(
        codes, transition_counts, registered_pose6
    )
    decomposition, category_maps = stationarity_decomposition(
        codes, transition_counts, xi_membership, target, predicted
    )
    track_path = output / "xi_proxy_tracks.jsonl"
    _write_jsonl(
        track_path,
        [
            {
                "schema": "ddm_g4_xi_proxy_track.v1",
                "track_index": index,
                "start_pair": track.start_pair,
                "start_row": track.start_row,
                "start_col": track.start_col,
                "transition_code": track.transition_code,
                "length": track.length,
                "event_ids": list(track.event_ids),
                "registration_scope": xi_summary["registration_scope"],
                "score_claim": False,
            }
            for index, track in enumerate(xi_tracks)
        ],
    )
    _checkpoint(
        output,
        2,
        "stationarity_decomposition_complete",
        {
            "decomposition": decomposition,
            "xi_summary": xi_summary,
            "xi_tracks": _output_row(track_path),
            "status": "complete",
        },
    )

    free_context = free_context_measurement(codes, predicted, flip_frequency)
    opportunities = build_opportunities(
        transition_counts, flip_frequency, xi_tracks, stratum_frequency
    )
    ledger_rows = typed_ledger_rows(
        concentration=concentration,
        recurrence=recurrence,
        decomposition=decomposition,
        free_context=free_context,
        opportunities=opportunities,
        xi_summary=xi_summary,
    )
    ledger_path = output / "ddm_g4_spatial_stationarity_n600.jsonl"
    _write_jsonl(ledger_path, ledger_rows)
    summary = _compact_summary(concentration, decomposition, free_context, opportunities, xi_summary)
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)

    maps_dir = output / "maps"
    _plot_map(maps_dir / "flip_frequency_all.png", flip_frequency, "All n600 flip frequency")
    for stratum in STRATA[1:]:
        _plot_map(
            maps_dir / f"flip_frequency_{stratum}.png",
            stratum_frequency[stratum],
            f"{stratum.replace('_', ' ').title()} flip frequency",
        )
    for category, values in category_maps.items():
        _plot_map(
            maps_dir / f"stationarity_{category.lower()}.png",
            values,
            category.replace("_", " ").title(),
        )

    durable_outputs = [
        ledger_path,
        summary_path,
        recurrence_path,
        track_path,
        *sorted(maps_dir.glob("*.png")),
        *sorted(
            path
            for path in (output / "stage_checkpoints").glob("*.json")
            if path.name != "03_outputs_complete.json"
        ),
    ]
    cleanup_path = output / "cleanup_manifest.json"
    _atomic_json(
        cleanup_path,
        {
            "schema": "certified_rebuildable_artifact_manifest.v1",
            "policy": "certify-or-block",
            "artifacts": [
                {
                    **_output_row(path),
                    "rebuildable": True,
                    "delete_authorized": False,
                    "reason": "rebuildable from SHA-bound G3/V12 scorer-cache custody",
                }
                for path in durable_outputs
            ],
            "semantic_argv": argv,
            "false_authority": {"evidence_axis": AXIS, "score_claim": False, "promotion_eligible": False},
        },
    )
    durable_outputs.append(cleanup_path)

    compact_summary = compact / "summary.json"
    compact_ledger = compact / "stationarity_ledger.jsonl"
    _atomic_bytes(compact_summary, summary_path.read_bytes())
    _atomic_bytes(compact_ledger, ledger_path.read_bytes())
    outputs = [_output_row(path) for path in durable_outputs]
    receipt = {
        "schema": "ddm_g4_spatial_stationarity_receipt.v1",
        "run_id": config.run_id,
        "typed_config_sha256": config.typed_hash(),
        "semantic_argv": argv,
        "source_custody": source,
        "implementation_custody": {
            "git_head_at_build": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "source_files": [
                _output_row(REPO / "src/tac/optimization/ddm_g4_spatial_stationarity.py"),
                _output_row(REPO / "tools/measure_ddm_g4_spatial_stationarity.py"),
                _output_row(REPO / ".omx/research/configs/ddm_g4_spatial_stationarity_n600_20260722.json"),
            ],
        },
        "outputs": outputs,
        "compact_outputs": [_output_row(compact_summary), _output_row(compact_ledger)],
        "summary": summary,
        "storage_preflight": {"path": str(output), "required_free_bytes": MIN_FREE_BYTES, "status": "PASS"},
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            config.g3_receipt_path,
            config.v12_receipt_path,
            ".omx/research/per_stratum_recursive_fractal_optimal_20260721T191217Z.md",
            ".omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md",
            "src/tac/margin_saliency_map.py",
            ".omx/research/label_noise_floor_and_margin_saliency_20260618.md",
            ".omx/research/path_b_recalibration_and_resolve_audit_20260618.md",
            ".omx/state/lane_registry.json",
            ".omx/state/operator_p0_ledger.jsonl",
        ],
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "d_seg_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "verdict": "MEASURED_ADVISORY_SPATIAL_STATIONARITY_COMPLETE_XI_PROXY_SCOPED",
        "verdict_scope": (
            "exact v12 predicted-vs-target argmax cells; xi category is target-cache metric-Pose6 G1 proxy, "
            "not physical BEV, and its transport side information is unpriced; opportunity delta-dseg is "
            "cell-space intervention, not RGB receiver realization"
        ),
        "main_landing_review_required": True,
        "round1_self_review_required": True,
    }
    _atomic_json(receipt_path, receipt)
    _checkpoint(
        output,
        3,
        "outputs_complete",
        {"receipt": _output_row(receipt_path), "status": "complete"},
    )
    # The stage-3 checkpoint is written after receipt formation and therefore is
    # intentionally not part of that receipt's output list; its own hash is the
    # final resumability marker, not a scientific input.
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = DdmG4SpatialStationarityConfigV1.model_validate_json(config_path.read_text(encoding="utf-8"))
    argv = ["tools/measure_ddm_g4_spatial_stationarity.py", "--config", str(args.config)]
    if args.resume:
        argv.append("--resume")
    receipt = run(config, resume=args.resume, argv=argv)
    print(json.dumps({"receipt": str(receipt), "sha256": sha256_file(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
