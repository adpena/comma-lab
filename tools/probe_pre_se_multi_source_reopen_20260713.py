#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable $0 local n600 measurement for the #484 PRE-SE composition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TOOLS = REPO / "tools"
for entry in (SRC, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import probe_frozen_replay_convex_head as round2  # noqa: E402
import probe_pre_se_locus_20260713 as pre_se_probe  # noqa: E402
import probe_replace_round5_deeper_nonlinear as round5  # noqa: E402

from tac.scorer_surrogate.pre_se_locus_20260713 import (  # noqa: E402
    PreSECutCostLedger,
    PreSEPairGatedMLPWeights,
    capture_pre_se_teacher,
    pre_se_feature_snapshot,
    pre_se_pair_gated_logits_numpy,
)
from tac.scorer_surrogate.pre_se_multi_source_reopen_20260713 import (  # noqa: E402
    AUTHORITY_SCOPE,
    MULTI_SOURCE_FEATURE_COUNT,
    SCHEMA,
    capture_full_frame_se_gates,
    capture_variable_pre_se,
    cheap_global_cost_accounting,
    compose_protected_feature_rows,
    derive_receptive_field_to_block3_pre_se,
    donated_se_gates,
    multi_source_pair_block_features,
    ordered_upstream_se_modules,
)
from tac.scorer_surrogate.replace_round4_support_ranking import (  # noqa: E402
    ORDERED_PAIR_COUNT,
    QuadraticStatistics,
    array_sha256,
    block_scores,
    exact_support_target,
    pairwise_rank_block_statistics,
)
from tac.witness_dsl.pre_se_multi_source_reopen_policy_20260713 import (  # noqa: E402
    PreSEMultiSourceReopenPolicy,
)

LANE_ID = "lane_replace_round5_pre_se_locus_20260713"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]"
DEFAULT_OUTPUT = REPO / "experiments/results/pre_se_multi_source_reopen_20260713"
PREREGISTRATION = DEFAULT_OUTPUT / "preregistration.json"
STORAGE_PREFLIGHT = (
    REPO / ".omx/research/pre_se_multi_source_reopen_storage_preflight_20260713.json"
)
PRIOR_OUTPUT = REPO / "experiments/results/pre_se_locus_20260713"
PRIOR_RECEIPT = PRIOR_OUTPUT / "receipt.json"
ROUND5_OUTPUT = REPO / "experiments/results/replace_round5_deeper_nonlinear_20260713"

SOURCE_FILES = (
    "src/tac/witness_dsl/pre_se_multi_source_reopen_policy_20260713.py",
    "src/tac/scorer_surrogate/pre_se_multi_source_reopen_20260713.py",
    "tools/probe_pre_se_multi_source_reopen_20260713.py",
    "src/tac/witness_dsl/pre_se_locus_policy_20260713.py",
    "src/tac/scorer_surrogate/pre_se_locus_20260713.py",
    "tools/probe_pre_se_locus_20260713.py",
    "src/tac/scorer_surrogate/replace_round5_deeper_nonlinear.py",
    "tools/probe_replace_round5_deeper_nonlinear.py",
    "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
)


class ProbeError(RuntimeError):
    """Measurement or custody failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    round5._atomic_json(path, payload)


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    result = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing source {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for relative, row in _source_fingerprints().items():
        source = REPO / relative
        destination = output_dir / "source_bundle" / relative
        if destination.exists():
            if destination.stat().st_size != row["bytes"] or _sha256(destination) != row["sha256"]:
                raise ProbeError(f"source bundle drift for {relative}")
        else:
            round5._atomic_bytes(destination, source.read_bytes())
        result[relative] = {**row, "path": str(destination.relative_to(output_dir))}
    return result


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()


def _runtime_custody(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _write_preregistration(policy: PreSEMultiSourceReopenPolicy) -> dict[str, Any]:
    payload = {
        "schema": "pre_se_multi_source_reopen_preregistration.v1",
        "sealed_at_utc": _utc_now(),
        "sealed_before_any_reopen_heldout_teacher_call": True,
        "measurement_contract": policy.compile_measurement_contract(),
        "rungs": [
            {
                "name": "convex-multi-source-pair-block-mp",
                "feature_count": policy.feature_count,
                "fit": "twenty exact float64 pair-block RankRLS Moore-Penrose optima",
            },
            {
                "name": "nonlinear-multi-source-pair-gated-mlp-ensemble",
                "feature_count": policy.feature_count,
                "fit": "three deterministic 476-to-32-to-20 pair-gated ReLU MLP seeds",
            },
        ],
        "only_feature_delta": (
            "shared base-42 once + block2 PRE-SE 144 + block3 PRE-SE 288 + shared sensitivity-2 once"
        ),
        "only_tileability_delta": (
            "seven unique upstream SE gates computed once full-frame and broadcast; exact core-tile equality"
        ),
        "no_post_heldout_rung": True,
        "number_labels": "MEASURED DERIVED INFERRED ASSUMED",
        "verdict_ladder": "INSTANCE < FORMULATION < FAMILY < PARADIGM",
    }
    if PREREGISTRATION.exists():
        existing = json.loads(PREREGISTRATION.read_text())
        left = {key: value for key, value in existing.items() if key != "sealed_at_utc"}
        right = {key: value for key, value in payload.items() if key != "sealed_at_utc"}
        if left != right:
            raise ProbeError("existing preregistration differs from typed policy")
        return existing
    _atomic_json(PREREGISTRATION, payload)
    return payload


def _validate_preregistration(policy: PreSEMultiSourceReopenPolicy) -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise ProbeError("missing multi-source preregistration")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload["measurement_contract"] != json.loads(
        json.dumps(policy.compile_measurement_contract(), sort_keys=True, allow_nan=False)
    ):
        raise ProbeError("typed policy and preregistration disagree")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO)),
        "bytes": PREREGISTRATION.stat().st_size,
        "sha256": _sha256(PREREGISTRATION),
        "payload": payload,
    }


def _storage_custody(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing storage waterfall preflight")
    payload = json.loads(STORAGE_PREFLIGHT.read_text())
    if payload.get("blockers"):
        raise ProbeError(f"storage waterfall blocked: {payload['blockers']}")
    selected = Path(payload.get("selected_workload_root", "")).resolve()
    if selected != output_dir.resolve():
        raise ProbeError("storage preflight selected a different workload root")
    free_bytes = os.statvfs(output_dir).f_bavail * os.statvfs(output_dir).f_frsize
    if free_bytes < int(payload["requested_bytes"]):
        raise ProbeError("free space fell below the preregistered request")
    return {
        "path": str(STORAGE_PREFLIGHT.relative_to(REPO)),
        "bytes": STORAGE_PREFLIGHT.stat().st_size,
        "sha256": _sha256(STORAGE_PREFLIGHT),
        "selected_tier": payload["selected_tier"],
        "requested_bytes": payload["requested_bytes"],
        "free_bytes_at_run_start": free_bytes,
        "explicit_local_opt_in": payload["operator_storage_policy"]["local_disk_enabled"],
        "blockers": [],
    }


def _verify_prior_custody() -> dict[str, Any]:
    complete_path = PRIOR_OUTPUT / "complete.json"
    if not complete_path.is_file() or not PRIOR_RECEIPT.is_file():
        raise ProbeError("protected PRE-SE completion custody is missing")
    complete = json.loads(complete_path.read_text())
    if (
        complete.get("receipt") != "receipt.json"
        or complete.get("bytes") != PRIOR_RECEIPT.stat().st_size
        or complete.get("sha256") != _sha256(PRIOR_RECEIPT)
    ):
        raise ProbeError("protected PRE-SE receipt custody drifted")
    receipt = json.loads(PRIOR_RECEIPT.read_text())
    accounting = receipt["teacher_call_accounting"]
    if not accounting["all_n600_states_completed"] or accounting["campaign_honest_started_calls"] != 600:
        raise ProbeError("protected PRE-SE receipt is not a complete n600 source")
    return {
        "path": str(PRIOR_RECEIPT.relative_to(REPO)),
        "bytes": PRIOR_RECEIPT.stat().st_size,
        "sha256": _sha256(PRIOR_RECEIPT),
        "prior_verdict": receipt["verdict"]["verdict"],
        "prior_retained_mass": {
            name: {
                rung: values["retained_mass_fraction"]
                for rung, values in locus["rungs"].items()
            }
            for name, locus in receipt["verdict"]["loci"].items()
        },
    }


def _empty_accumulator() -> dict[str, np.ndarray]:
    return {
        "completed_pairs": np.empty(0, dtype=np.int64),
        "gram": np.zeros(
            (ORDERED_PAIR_COUNT, MULTI_SOURCE_FEATURE_COUNT, MULTI_SOURCE_FEATURE_COUNT),
            dtype=np.float64,
        ),
        "rhs": np.zeros((ORDERED_PAIR_COUNT, MULTI_SOURCE_FEATURE_COUNT), dtype=np.float64),
        "target_square": np.zeros(ORDERED_PAIR_COUNT, dtype=np.float64),
        "row_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "state_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
    }


def _load_accumulator(path: Path) -> dict[str, np.ndarray]:
    expected = _empty_accumulator()
    if not path.exists():
        return expected
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    if result.keys() != expected.keys() or any(
        name != "completed_pairs" and result[name].shape != expected[name].shape
        for name in expected
    ):
        raise ProbeError("multi-source accumulator schema drift")
    completed = result["completed_pairs"]
    if completed.dtype != np.int64 or not np.array_equal(completed, np.unique(completed)):
        raise ProbeError("multi-source completed-pair index drift")
    return result


def _add_pairwise(
    accumulator: dict[str, np.ndarray], pair_index: int, records: Sequence[QuadraticStatistics]
) -> None:
    for block, record in enumerate(records):
        accumulator["gram"][block] += record.gram
        accumulator["rhs"][block] += record.rhs
        accumulator["target_square"][block] += record.target_square
        accumulator["row_count"][block] += record.row_count
        accumulator["state_count"][block] += record.state_count
    accumulator["completed_pairs"] = np.sort(
        np.append(accumulator["completed_pairs"], pair_index).astype(np.int64)
    )


def _stats(accumulator: Mapping[str, np.ndarray], block: int) -> QuadraticStatistics:
    return QuadraticStatistics(
        gram=accumulator["gram"][block],
        rhs=accumulator["rhs"][block],
        target_square=float(accumulator["target_square"][block]),
        row_count=int(accumulator["row_count"][block]),
        state_count=int(accumulator["state_count"][block]),
    )


def _tileability_stage(
    *, output_dir: Path, segnet: Any, frame: Any, cut_cost_model: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    stage_path = output_dir / "stage_tileability_complete.json"
    if stage_path.exists():
        return json.loads(stage_path.read_text())
    ancestors = ordered_upstream_se_modules(segnet)
    gates = capture_full_frame_se_gates(segnet, frame, ancestors)
    rf = derive_receptive_field_to_block3_pre_se(segnet)
    halo = int(rf["aligned_halo_input_pixels"])
    # Match the inference-only donated path so CPU convolution dispatch is identical.
    with torch.no_grad():
        baseline2, baseline3 = pre_se_feature_snapshot(segnet, frame)[1:]
    with donated_se_gates(segnet, gates):
        donated2, donated3 = capture_variable_pre_se(segnet, frame)
    full_parity = {
        "block2-pre-se": {
            "bitwise_equal": bool(torch.equal(baseline2, donated2)),
            "max_abs": float((baseline2 - donated2).abs().max().item()),
        },
        "block3-pre-se": {
            "bitwise_equal": bool(torch.equal(baseline3, donated3)),
            "max_abs": float((baseline3 - donated3).abs().max().item()),
        },
    }
    if not all(row["bitwise_equal"] for row in full_parity.values()):
        raise ProbeError("donated gates do not reproduce the full-frame PRE-SE tensors")
    height, width = (int(value) for value in frame.shape[-2:])
    if (height, width) != (384, 512):
        raise ProbeError("tile proof requires the registered real 384x512 frame")
    cores = ((0, 192, 0, 256), (0, 192, 256, 512), (192, 384, 0, 256), (192, 384, 256, 512))
    tile_rows = []
    tiled_local_conv_macs = 0
    for tile_index, (y0, y1, x0, x1) in enumerate(cores):
        ey0, ey1 = max(0, y0 - halo), min(height, y1 + halo)
        ex0, ex1 = max(0, x0 - halo), min(width, x1 + halo)
        crop = frame[:, :, ey0:ey1, ex0:ex1]
        with donated_se_gates(segnet, gates), PreSECutCostLedger(segnet) as ledger:
            local2, local3 = capture_variable_pre_se(segnet, crop)
        local_macs = sum(
            int(row["forward_macs"])
            for row in ledger._conv_rows
            if ledger._under_pre_se_cut(str(row["module"]), 2, 2)
        )
        tiled_local_conv_macs += local_macs
        # A same-shape zero embedding distinguishes nonlocal dependence from the
        # shape-dependent accumulation order of CPU convolution kernels.  The
        # operational cost below remains the independently cropped tile cost.
        embedded = torch.zeros_like(frame)
        embedded[:, :, ey0:ey1, ex0:ex1] = crop
        with donated_se_gates(segnet, gates):
            embedded2, embedded3 = capture_variable_pre_se(segnet, embedded)
        comparisons = {}
        for name, stride, local, full in (
            ("block2-pre-se", 4, local2, baseline2),
            ("block3-pre-se", 8, local3, baseline3),
        ):
            local_core = local[
                :,
                :,
                (y0 - ey0) // stride : (y1 - ey0) // stride,
                (x0 - ex0) // stride : (x1 - ex0) // stride,
            ]
            full_core = full[:, :, y0 // stride : y1 // stride, x0 // stride : x1 // stride]
            embedded_source = embedded2 if name == "block2-pre-se" else embedded3
            embedded_core = embedded_source[
                :, :, y0 // stride : y1 // stride, x0 // stride : x1 // stride
            ]
            comparisons[name] = {
                "shape_invariant_zero_embedded_bitwise_equal": bool(
                    torch.equal(embedded_core, full_core)
                ),
                "shape_invariant_zero_embedded_max_abs": float(
                    (embedded_core - full_core).abs().max().item()
                ),
                "physical_crop_bitwise_equal": bool(torch.equal(local_core, full_core)),
                "physical_crop_max_abs": float((local_core - full_core).abs().max().item()),
                "shape": [int(value) for value in local_core.shape],
            }
        tile_rows.append(
            {
                "tile_index": tile_index,
                "core_input_bounds": [y0, y1, x0, x1],
                "expanded_input_bounds": [ey0, ey1, ex0, ex1],
                "local_conv_forward_macs": local_macs,
                "comparisons": comparisons,
            }
        )
    exact = all(
        row["comparisons"][name]["shape_invariant_zero_embedded_bitwise_equal"]
        for row in tile_rows
        for name in ("block2-pre-se", "block3-pre-se")
    )
    cropped_max_abs = max(
        float(row["comparisons"][name]["physical_crop_max_abs"])
        for row in tile_rows
        for name in ("block2-pre-se", "block3-pre-se")
    )
    accounting = cheap_global_cost_accounting(
        cut_cost_model,
        upstream_se_modules=ancestors,
        tile_count=len(tile_rows),
        tiled_local_conv_macs=tiled_local_conv_macs,
    )
    deepest_cut = cut_cost_model["cuts"]["block3-pre-se"]
    ideal_full_frame = (
        int(deepest_cut["forward_plus_input_vjp_conv_flops"])
        + int(deepest_cut["global_pool_forward_plus_vjp_flops"])
    )
    accounting.update(
        {
            "ideal_full_frame_deepest_cut_forward_plus_vjp_flops": ideal_full_frame,
            "ideal_equal_area_per_tile_including_amortized_globals": ideal_full_frame / len(tile_rows),
            "measured_overlap_tiling_overhead_ratio": (
                accounting["total_tiled_forward_plus_vjp_flops"] / ideal_full_frame
            ),
            "composition_cost_rule": "deepest block3 cut subsumes block2; never sum both cut costs",
        }
    )
    stage = {
        "schema": "pre_se_multi_source_tileability_stage.v1",
        "completed_at_utc": _utc_now(),
        "status": "MEASURED_CONFIRMED" if exact else "MEASURED_FAILED",
        "tileable_modulo_cheap_globals": exact,
        "proof_scope": (
            "real 384x512 state; 2x2 aligned cores; derived sufficient halo; exact donated gates; "
            "bitwise same-shape zero-embedded locality plus separately measured physical-crop rounding"
        ),
        "physical_crop_max_abs_across_sources_and_tiles": cropped_max_abs,
        "physical_crop_rounding_status": "MEASURED_SHAPE_DEPENDENT_CPU_ROUNDING",
        "unique_upstream_se_modules": list(ancestors),
        "unique_upstream_se_count": len(ancestors),
        "gate_array_sha256": {name: array_sha256(value.cpu().numpy()) for name, value in gates.items()},
        "full_frame_donated_gate_parity": full_parity,
        "receptive_field": rf,
        "tiles": tile_rows,
        "cost_accounting": accounting,
        "stage_barrier_requirement": (
            "tiles are independent only between SE barriers; each full-frame channel reduction must finish "
            "before its gate is broadcast to the next local-convolution stage"
        ),
    }
    _atomic_json(stage_path, stage)
    return stage


def _training_stage(
    *, output_dir: Path, assignments: Sequence[Any], labels: Any, margins: Any, policy: Any, segnet: Any, yopo: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    accumulator_path = output_dir / "convex" / "train_accumulator_current.npz"
    accumulator = _load_accumulator(accumulator_path)
    completed = {int(value) for value in accumulator["completed_pairs"]}
    train = [row for row in assignments if row.split == "train"]
    preserved = []
    parity_rows = []
    tile_stage = None
    cut_cost_model = json.loads((PRIOR_OUTPUT / "pre_se_cut_cost_model.json").read_text())
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.base.base.n_pairs or code.shape[0] != 2 * policy.base.base.n_pairs:
            raise ProbeError("renderer geometry drifted")
        parity = round2._checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        for assignment in (row for row in train if row.checkpoint_index == checkpoint_index):
            if assignment.pair_index in completed:
                continue
            support, _mass, pair_ids = round5._load_target(
                ROUND5_OUTPUT / "train_targets" / f"pair_{assignment.pair_index:04d}.npz", assignment
            )
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            if tile_stage is None:
                tile_stage = _tileability_stage(
                    output_dir=output_dir, segnet=segnet, frame=frame, cut_cost_model=cut_cost_model
                )
            prefix, block2, block3 = pre_se_feature_snapshot(segnet, frame)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            features, pairs = multi_source_pair_block_features(
                prefix.cpu().numpy(),
                block2.cpu().numpy(),
                block3.cpu().numpy(),
                label,
                margin,
                pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.base.base.checkpoint_count,
                stride=policy.base.base.train_lattice_stride_on_prefix,
            )
            if not np.array_equal(pairs, pair_ids.reshape(-1)):
                raise ProbeError("cached and reconstructed pair rows differ")
            records = pairwise_rank_block_statistics(
                features.astype(np.float64), pairs, support.reshape(-1)
            )
            _add_pairwise(accumulator, assignment.pair_index, records)
            round5._atomic_npz(accumulator_path, **accumulator)
            completed.add(assignment.pair_index)
        stage_checkpoint = (
            output_dir / "convex" / "stage_checkpoints" / f"train_{checkpoint_name}_complete.npz"
        )
        if not stage_checkpoint.exists():
            round5._atomic_npz(stage_checkpoint, **accumulator)
        preserved.append(
            {
                "checkpoint_name": checkpoint_name,
                "path": str(stage_checkpoint.relative_to(output_dir)),
                "bytes": stage_checkpoint.stat().st_size,
                "sha256": _sha256(stage_checkpoint),
                "completed_train_states_cumulative": int(accumulator["completed_pairs"].size),
            }
        )
    if len(completed) != len(train):
        raise ProbeError("convex training stage did not cover all 480 states")
    if tile_stage is None:
        tile_stage = json.loads((output_dir / "stage_tileability_complete.json").read_text())
    stage = {
        "schema": "pre_se_multi_source_train_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(train),
        "feature_count": MULTI_SOURCE_FEATURE_COUNT,
        "exact_targets": "MEASURED-INHERITED immutable Round-5 compact targets",
        "fresh_exact_teacher_calls": 0,
        "feature_forward_states": len(train),
        "cross_terms": "MEASURED by replay; unavailable in separate protected Gram matrices",
        "checkpoint_parity": parity_rows,
        "preserved_stage_checkpoints": preserved,
        "accumulator": {
            "path": str(accumulator_path.relative_to(output_dir)),
            "bytes": accumulator_path.stat().st_size,
            "sha256": _sha256(accumulator_path),
        },
        "tileability_stage": "stage_tileability_complete.json",
    }
    _atomic_json(output_dir / "stage_train_complete.json", stage)
    return accumulator, stage


def _convex_fit_stage(output_dir: Path, accumulator: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    fits = tuple(round5.fit_exact_quadratic(_stats(accumulator, block)) for block in range(ORDERED_PAIR_COUNT))
    weights = np.stack([fit.weights for fit in fits]).astype(np.float64)
    certificates = [fit.certificate for fit in fits]
    if not all(row["normal_equation_optimum_certified"] for row in certificates):
        raise ProbeError("joint convex fit lacks a Moore-Penrose optimum certificate")
    path = output_dir / "convex" / "weights.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["weights"], weights):
                raise ProbeError("joint convex weights drifted on resume")
    else:
        round5._atomic_npz(path, weights=weights)
    stage = {
        "schema": "pre_se_multi_source_convex_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "feature_count": MULTI_SOURCE_FEATURE_COUNT,
        "exact_optimum_class": "float64 pair-block RankRLS Moore-Penrose",
        "block_certificates": certificates,
        "weights": {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "array_sha256": array_sha256(weights),
        },
    }
    _atomic_json(output_dir / "stage_convex_fit_complete.json", stage)
    return weights, stage


def _verify_chunk_alignment(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> None:
    for key in ("checkpoint_index", "core_pair", "core_y", "dev_pair", "dev_y", "dev_mass", "dev_lengths"):
        if not np.array_equal(left[key], right[key]):
            raise ProbeError(f"protected nonlinear chunk alignment drifted for {key}")


def _nonlinear_data_stage(output_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    chunks = []
    custody = []
    for checkpoint_name, _checkpoint_path, _epoch in round2.CHECKPOINTS:
        left_path = PRIOR_OUTPUT / "loci/block2-pre-se/nonlinear/stage_chunks" / f"{checkpoint_name}.npz"
        right_path = PRIOR_OUTPUT / "loci/block3-pre-se/nonlinear/stage_chunks" / f"{checkpoint_name}.npz"
        with np.load(left_path, allow_pickle=False) as archive:
            left = {name: np.array(archive[name], copy=True) for name in archive.files}
        with np.load(right_path, allow_pickle=False) as archive:
            right = {name: np.array(archive[name], copy=True) for name in archive.files}
        _verify_chunk_alignment(left, right)
        chunks.append(
            {
                "core_x": compose_protected_feature_rows(left["core_x"], right["core_x"]),
                "core_pair": left["core_pair"],
                "core_y": left["core_y"],
                "dev_x": compose_protected_feature_rows(left["dev_x"], right["dev_x"]),
                "dev_pair": left["dev_pair"],
                "dev_y": left["dev_y"],
                "dev_mass": left["dev_mass"],
                "dev_lengths": left["dev_lengths"],
            }
        )
        custody.append(
            {
                "checkpoint_name": checkpoint_name,
                "block2": {"path": str(left_path.relative_to(REPO)), "bytes": left_path.stat().st_size, "sha256": _sha256(left_path)},
                "block3": {"path": str(right_path.relative_to(REPO)), "bytes": right_path.stat().st_size, "sha256": _sha256(right_path)},
                "alignment": "MEASURED_BITWISE_ALL_TARGET_PAIR_AND_SPLIT_COLUMNS",
            }
        )
    core_x = np.concatenate([chunk["core_x"] for chunk in chunks]).astype(np.float32)
    data = {
        "core_x": core_x,
        "core_pair": np.concatenate([chunk["core_pair"] for chunk in chunks]).astype(np.int16),
        "core_y": np.concatenate([chunk["core_y"] for chunk in chunks]).astype(np.float32),
        "dev_x": np.concatenate([chunk["dev_x"] for chunk in chunks]).astype(np.float32),
        "dev_pair": np.concatenate([chunk["dev_pair"] for chunk in chunks]).astype(np.int16),
        "dev_y": np.concatenate([chunk["dev_y"] for chunk in chunks]).astype(np.bool_),
        "dev_mass": np.concatenate([chunk["dev_mass"] for chunk in chunks]).astype(np.float32),
    }
    dev_lengths = np.concatenate([chunk["dev_lengths"] for chunk in chunks])
    data["dev_offsets"] = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(dev_lengths, dtype=np.int64)))
    mean = data["core_x"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(data["core_x"].std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1e-6))
    data["feature_mean"] = mean
    data["feature_std"] = std
    data["core_x"] = np.ascontiguousarray((data["core_x"] - mean) / std, dtype=np.float32)
    data["dev_x"] = np.ascontiguousarray((data["dev_x"] - mean) / std, dtype=np.float32)
    normalization_path = output_dir / "nonlinear" / "normalization.npz"
    round5._atomic_npz(normalization_path, feature_mean=mean, feature_std=std)
    stage = {
        "schema": "pre_se_multi_source_nonlinear_data_stage.v1",
        "completed_at_utc": _utc_now(),
        "feature_count": MULTI_SOURCE_FEATURE_COUNT,
        "core_balanced_rows": int(data["core_x"].shape[0]),
        "dev_full_rows": int(data["dev_x"].shape[0]),
        "source_chunks": custody,
        "new_raw_chunk_copy": False,
        "normalization": {
            "path": str(normalization_path.relative_to(output_dir)),
            "bytes": normalization_path.stat().st_size,
            "sha256": _sha256(normalization_path),
        },
    }
    _atomic_json(output_dir / "stage_nonlinear_data_complete.json", stage)
    return data, stage


@contextmanager
def _round5_width_adapter() -> Iterator[None]:
    original = (round5.DEEP_FEATURE_COUNT, round5.PairGatedMLPWeights, round5.pair_gated_logits_numpy)
    round5.DEEP_FEATURE_COUNT = MULTI_SOURCE_FEATURE_COUNT
    round5.PairGatedMLPWeights = PreSEPairGatedMLPWeights
    round5.pair_gated_logits_numpy = pre_se_pair_gated_logits_numpy
    try:
        yield
    finally:
        round5.DEEP_FEATURE_COUNT, round5.PairGatedMLPWeights, round5.pair_gated_logits_numpy = original


def _nonlinear_fit_stage(
    output_dir: Path, data: dict[str, np.ndarray], policy: PreSEMultiSourceReopenPolicy
) -> tuple[tuple[PreSEPairGatedMLPWeights, ...], dict[str, Any]]:
    with _round5_width_adapter():
        weights, stage = round5._fit_nonlinear_stage(
            output_dir=output_dir, data=data, policy=policy.base.base
        )
    return weights, stage


def _heldout_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout" / f"pair_{pair_index:04d}.json"


def _heldout_stage(
    *, output_dir: Path, assignments: Sequence[Any], labels: Any, margins: Any, policy: Any,
    convex_weights: np.ndarray, nonlinear_weights: Sequence[PreSEPairGatedMLPWeights],
    data: Mapping[str, np.ndarray], segnet: Any, yopo: Any
) -> dict[str, Any]:
    import torch

    records = []
    heldout = [row for row in assignments if row.split == "heldout"]
    ledger = output_dir / "teacher_calls.jsonl"
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in (row for row in heldout if row.checkpoint_index == checkpoint_index):
            path = _heldout_record_path(output_dir, assignment.pair_index)
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            prior_path = PRIOR_OUTPUT / "heldout" / f"pair_{assignment.pair_index:04d}.json"
            if not prior_path.is_file():
                raise ProbeError("protected heldout state receipt is missing")
            prior = json.loads(prior_path.read_text())
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            batch_id = f"pre-se-multi-heldout-p{assignment.pair_index:04d}-{os.getpid()}"
            round2._teacher_start(ledger, assignment, stage="pre_se_multi_source_heldout", batch_id=batch_id)
            prefix, block2, block3, costate, pair_ids, metrics, elapsed = capture_pre_se_teacher(
                segnet=segnet,
                frame_nchw=frame,
                labels=torch.as_tensor(label[None], dtype=torch.long),
            )
            round2._teacher_complete(
                ledger,
                assignment,
                stage="pre_se_multi_source_heldout",
                batch_id=batch_id,
                teacher_metrics=metrics,
                elapsed_seconds=elapsed,
            )
            costate_sha = array_sha256(costate.cpu().numpy())
            if costate_sha != prior["costate_sha256"]:
                raise ProbeError(f"heldout exact costate drift for pair {assignment.pair_index}")
            mass, support, selected = exact_support_target(
                costate.cpu().numpy(), area_fraction=policy.base.base.requested_area_fraction
            )
            if selected != policy.base.base.selected_prefix_cells:
                raise ProbeError("heldout support area drifted")
            features, pairs = multi_source_pair_block_features(
                prefix.cpu().numpy(), block2.cpu().numpy(), block3.cpu().numpy(), label, margin, pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.base.base.checkpoint_count,
                stride=policy.base.base.heldout_lattice_stride_on_prefix,
            )
            convex_scores = block_scores(features.astype(np.float64), pairs, convex_weights)
            standardized = np.ascontiguousarray(
                (features - data["feature_mean"]) / data["feature_std"], dtype=np.float32
            )
            seed_logits = np.stack(
                [pre_se_pair_gated_logits_numpy(standardized, pairs, weight) for weight in nonlinear_weights]
            )
            seed_probabilities = round5.sigmoid_probabilities(seed_logits)
            ensemble = seed_probabilities.mean(axis=0, dtype=np.float64).astype(np.float32)
            selections = {
                "convex-multi-source-pair-block-mp": round5._selection_metrics(mass, convex_scores, selected),
                "nonlinear-multi-source-pair-gated-mlp-ensemble": round5._selection_metrics(mass, ensemble, selected),
                "oracle": {
                    "total_mass": float(mass.sum(dtype=np.float64)),
                    "retained_mass": float(np.sort(mass.reshape(-1).astype(np.float64))[-selected:].sum(dtype=np.float64)),
                },
            }
            for index, seed in enumerate(policy.base.base.nonlinear_seeds):
                selections[f"nonlinear-seed-{seed}"] = round5._selection_metrics(
                    mass, seed_probabilities[index], selected
                )
            record = {
                "schema": "pre_se_multi_source_heldout_state.v1",
                "pair_index": assignment.pair_index,
                "checkpoint_index": assignment.checkpoint_index,
                "checkpoint_name": assignment.checkpoint_name,
                "selected_prefix_cells": selected,
                "realized_area_fraction": policy.base.base.realized_area_fraction,
                "selections": selections,
                "teacher_metrics": metrics,
                "teacher_elapsed_seconds": elapsed,
                "costate_sha256": costate_sha,
                "prior_costate_sha256": prior["costate_sha256"],
                "prior_hash_match": True,
            }
            _atomic_json(path, record)
            records.append(record)
    if len(records) != policy.base.base.heldout_state_count:
        raise ProbeError("heldout state count drifted")
    names = [
        "convex-multi-source-pair-block-mp",
        "nonlinear-multi-source-pair-gated-mlp-ensemble",
        *(f"nonlinear-seed-{seed}" for seed in policy.base.base.nonlinear_seeds),
        "oracle",
    ]
    aggregate = {}
    for name in names:
        total = sum(row["selections"][name]["total_mass"] for row in records)
        retained = sum(row["selections"][name]["retained_mass"] for row in records)
        aggregate[name] = {
            "total_mass": total,
            "retained_mass": retained,
            "retained_mass_fraction": retained / total,
            "conditional_masked_exact_cosine": math.sqrt(retained / total),
        }
    stage = {
        "schema": "pre_se_multi_source_heldout_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(records),
        "costate_hash_matches_prior": sum(bool(row["prior_hash_match"]) for row in records),
        "aggregate": aggregate,
    }
    _atomic_json(output_dir / "stage_heldout_complete.json", stage)
    return stage


def _teacher_accounting(output_dir: Path, assignments: Sequence[Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in (output_dir / "teacher_calls.jsonl").read_text().splitlines() if line.strip()]
    starts = [row for row in rows if row.get("event") == "exact_teacher_state_call_started"]
    completions = [row for row in rows if row.get("event") == "exact_teacher_state_call_completed"]
    started = {int(row["pair_index"]) for row in starts}
    completed = {int(row["pair_index"]) for row in completions}
    heldout = {row.pair_index for row in assignments if row.split == "heldout"}
    return {
        "schema": "pre_se_multi_source_teacher_accounting.v1",
        "inherited_round5_exact_train_targets": 480,
        "replayed_exact_heldout_started_calls": len(starts),
        "replayed_exact_heldout_completed_calls": len(completions),
        "replayed_exact_heldout_unique_started_states": len(started),
        "replayed_exact_heldout_unique_completed_states": len(completed),
        "retry_starts_charged": len(starts) - len(started),
        "campaign_honest_started_calls": 480 + len(starts),
        "all_n600_states_completed": completed == heldout,
        "all_replayed_costates_hash_matched_prior": True,
        "custody_note": (
            "480 immutable compact targets inherited; the same 120 registered heldout states were "
            "replayed because the protected receipt stores hashes/metrics but not raw support vectors"
        ),
        "ledger": {
            "path": "teacher_calls.jsonl",
            "bytes": (output_dir / "teacher_calls.jsonl").stat().st_size,
            "sha256": _sha256(output_dir / "teacher_calls.jsonl"),
        },
    }


def _decision(
    heldout: Mapping[str, Any], tileability: Mapping[str, Any], prior: Mapping[str, Any], policy: Any
) -> dict[str, Any]:
    aggregate = heldout["aggregate"]
    convex = aggregate["convex-multi-source-pair-block-mp"]["retained_mass_fraction"]
    nonlinear = aggregate["nonlinear-multi-source-pair-gated-mlp-ensemble"]["retained_mass_fraction"]
    seeds = np.asarray(
        [aggregate[f"nonlinear-seed-{seed}"]["retained_mass_fraction"] for seed in policy.base.base.nonlinear_seeds],
        dtype=np.float64,
    )
    mass_pass = bool(max(convex, nonlinear) >= policy.base.base.retained_mass_bar)
    tile_pass = bool(tileability["tileable_modulo_cheap_globals"])
    reopen = mass_pass and tile_pass
    verdict = "REOPEN-CHEAP-LOCALIZATION-FAMILY" if reopen else "RETAINED-MASS-FAMILY-KILL"
    route = (
        "#484 whole-teacher-over-boundary hedge plus concrete tileable multi-source surrogate"
        if reopen
        else "#455 whole-teacher DISTILLED student; no tileability prerequisite"
    )
    return {
        "verdict": verdict,
        "verdict_scope": (
            "FAMILY x CHEAP-PRE-SE-LOCALIZATION x SINGLE-AND-MULTI-SOURCE x "
            "CONVEX-AND-NONLINEAR-RUNGS x FIXED-n600-REPLAY x 4.70%-AREA"
        ),
        "requested_area_fraction": policy.base.base.requested_area_fraction,
        "realized_area_fraction": policy.base.base.realized_area_fraction,
        "retained_mass_bar": policy.base.base.retained_mass_bar,
        "convex_retained_mass_fraction": convex,
        "nonlinear_retained_mass_fraction": nonlinear,
        "nonlinear_seed_retained_mass": seeds.tolist(),
        "nonlinear_seed_population_std": float(seeds.std(ddof=0)),
        "retained_mass_gate_pass": mass_pass,
        "tileable_modulo_cheap_globals_gate_pass": tile_pass,
        "joint_reopen_gate_pass": reopen,
        "oracle_retained_mass_fraction": aggregate["oracle"]["retained_mass_fraction"],
        "req_R_family_evidence": {
            "requirement": "at least two formulations plus a structural reason",
            "formulations": [
                {
                    "name": "single-source block2 PRE-SE",
                    "convex": prior["prior_retained_mass"]["block2-pre-se"]["convex-deeper-pair-block-mp"],
                    "nonlinear": prior["prior_retained_mass"]["block2-pre-se"]["nonlinear-pair-gated-mlp-ensemble"],
                },
                {
                    "name": "single-source block3 PRE-SE",
                    "convex": prior["prior_retained_mass"]["block3-pre-se"]["convex-deeper-pair-block-mp"],
                    "nonlinear": prior["prior_retained_mass"]["block3-pre-se"]["nonlinear-pair-gated-mlp-ensemble"],
                },
                {"name": "joint multi-source PRE-SE", "convex": convex, "nonlinear": nonlinear},
            ],
            "structural_reason": (
                "The same-area exact oracle remains above the bar, while exact convex optima and a "
                "nonlinear ensemble over shallow+block2+block3 PRE-SE charts remain below it; the "
                "binding loss is target-ordering information absent from this cheap frozen feature family, "
                "not SE tileability or selected area."
            ),
            "family_evidence_satisfied": not mass_pass,
        },
        "route": route,
        "pointer_moved": False,
    }


def _cleanup_manifest(output_dir: Path, run_contract: Mapping[str, Any]) -> dict[str, Any]:
    source_bundle_sha = hashlib.sha256(json.dumps(run_contract["sources"], sort_keys=True).encode()).hexdigest()
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and ".tmp" not in path.name:
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "action": "PRESERVED-DURABLE",
                    "rebuildable": True,
                    "rebuild_command": ".venv/bin/python tools/probe_pre_se_multi_source_reopen_20260713.py --resume",
                    "source_bundle_sha256": source_bundle_sha,
                }
            )
    payload = {
        "schema": "pre_se_multi_source_cleanup_manifest.v1",
        "created_at_utc": _utc_now(),
        "policy": "certify-or-block",
        "artifacts": artifacts,
        "deleted": [],
        "moved": [],
        "blockers": [],
    }
    _atomic_json(output_dir / "cleanup_manifest.json", payload)
    return payload


def run(*, output_dir: Path, resume: bool, validate_only: bool) -> dict[str, Any]:
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ProbeError("the preregistered instance has one sealed output directory")
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        if not resume:
            raise ProbeError("completed receipt exists; pass --resume to verify")
        complete = json.loads(complete_path.read_text())
        receipt_path = output_dir / complete["receipt"]
        if receipt_path.stat().st_size != complete["bytes"] or _sha256(receipt_path) != complete["sha256"]:
            raise ProbeError("completed receipt custody drifted")
        return json.loads(receipt_path.read_text())
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = PreSEMultiSourceReopenPolicy()
    contract = policy.compile_measurement_contract()
    preregistration = _validate_preregistration(policy)
    storage = _storage_custody(output_dir)
    inputs = round2._verify_input_custody()
    prior = _verify_prior_custody()
    assignments = round2.deterministic_replay_assignments(
        n_pairs=policy.base.base.n_pairs,
        checkpoint_names=tuple(row[0] for row in round2.CHECKPOINTS),
        holdout_period=policy.base.base.holdout_period,
        seed=policy.base.base.seed,
    )
    inherited = pre_se_probe._verify_round5_target_custody(output_dir, assignments)
    if validate_only:
        return {
            "schema": "pre_se_multi_source_validate_only.v1",
            "compiled_policy": contract,
            "preregistration": preregistration,
            "storage": storage,
            "inputs": inputs,
            "prior": prior,
            "inherited_target_count": inherited["target_count"],
            "sources": _source_fingerprints(),
            "fresh_teacher_calls": 0,
        }
    descriptor = round2._acquire_lock(output_dir)
    try:
        contract_path = output_dir / "run_contract.json"
        if contract_path.exists():
            if not resume:
                raise ProbeError("existing run contract requires --resume")
            run_contract = json.loads(contract_path.read_text())
            if run_contract["compiled_policy"] != json.loads(json.dumps(contract, sort_keys=True, allow_nan=False)):
                raise ProbeError("resume policy drift")
            if run_contract["inputs"] != inputs or run_contract["prior"] != prior:
                raise ProbeError("resume input custody drift")
        else:
            run_contract = {
                "schema": "pre_se_multi_source_run_contract.v1",
                "created_at_utc": _utc_now(),
                "lane_id": LANE_ID,
                "compiled_policy": contract,
                "preregistration": preregistration,
                "inputs": inputs,
                "prior": prior,
                "inherited_target_manifest": {
                    "path": "inherited_round5_train_targets.json",
                    "bytes": (output_dir / "inherited_round5_train_targets.json").stat().st_size,
                    "sha256": _sha256(output_dir / "inherited_round5_train_targets.json"),
                },
                "sources": _source_bundle(output_dir),
                "runtime": _runtime_custody(torch),
                "storage_preflight": storage,
                "git_head_at_measurement": _git_head(),
                "git_status_at_measurement_start": _git_status(),
                "protected_pre_se_output_read_only": True,
                "protected_round5_output_read_only": True,
                "paid_or_remote_launch": False,
                "live_run_mutated": False,
                "witness_training_launched": False,
                "authority": AXIS,
            }
            _atomic_json(contract_path, run_contract)
        labels = round2._stored_npy_memmap(round2.GT_CACHE, "lstars.npy")
        margins = round2._stored_npy_memmap(round2.GT_CACHE, "margins.npy")
        if labels.shape != (600, 384, 512) or margins.shape != labels.shape:
            raise ProbeError("GT cache geometry drift")
        yopo = round2._load_tool_module(
            "_pre_se_multi_source_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
        )
        segnet = round2._load_cpu_segnet()
        accumulator, train_stage = _training_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        convex_weights, convex_stage = _convex_fit_stage(output_dir, accumulator)
        nonlinear_data, nonlinear_data_stage = _nonlinear_data_stage(output_dir)
        nonlinear_weights, nonlinear_fit_stage = _nonlinear_fit_stage(output_dir, nonlinear_data, policy)
        heldout = _heldout_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            convex_weights=convex_weights,
            nonlinear_weights=nonlinear_weights,
            data=nonlinear_data,
            segnet=segnet,
            yopo=yopo,
        )
        del nonlinear_data
        gc.collect()
        accounting = _teacher_accounting(output_dir, assignments)
        tileability = json.loads((output_dir / "stage_tileability_complete.json").read_text())
        decision = _decision(heldout, tileability, prior, policy)
        cleanup = _cleanup_manifest(output_dir, run_contract)
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc_now(),
            "verdict": decision,
            "authority": {
                "axis": AXIS,
                "module_scope": AUTHORITY_SCOPE,
                "research_only": True,
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "MPS_used": False,
            },
            "run_contract": run_contract,
            "train_stage": train_stage,
            "convex_fit_stage": convex_stage,
            "nonlinear_data_stage": nonlinear_data_stage,
            "nonlinear_fit_stage": nonlinear_fit_stage,
            "heldout_stage": heldout,
            "tileability_stage": tileability,
            "teacher_call_accounting": accounting,
            "cleanup_custody": {"path": "cleanup_manifest.json", "blockers": cleanup["blockers"]},
            "triality": {
                "dsl": "tac.witness_dsl.pre_se_multi_source_reopen_policy_20260713",
                "equation": "tac.canonical_equations.pre_se_multi_source_reopen_20260713",
                "dag_feed": ".omx/research/pre_se_multi_source_reopen_DAG_FEED_20260713.md",
            },
            "verdict_scope": decision["verdict_scope"],
            "route": decision["route"],
            "pointer_delta": "NONE",
        }
        receipt_path = output_dir / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(
            complete_path,
            {
                "schema": "pre_se_multi_source_completion.v1",
                "completed_at_utc": _utc_now(),
                "receipt": "receipt.json",
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
                "verdict": decision["verdict"],
            },
        )
        return receipt
    finally:
        round2._release_lock(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-preregistration", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if REPO not in output_dir.parents:
        raise ProbeError("durable evidence must remain under the repository")
    if args.write_preregistration:
        if args.resume or args.validate_only:
            raise ProbeError("preregistration write cannot be combined with run modes")
        payload = _write_preregistration(PreSEMultiSourceReopenPolicy())
        print(json.dumps({"path": str(PREREGISTRATION), "sealed_at_utc": payload["sealed_at_utc"]}, sort_keys=True))
        return 0
    receipt = run(output_dir=output_dir, resume=args.resume, validate_only=args.validate_only)
    if args.validate_only:
        print(json.dumps({"schema": receipt["schema"], "fresh_teacher_calls": 0, "inherited_target_count": receipt["inherited_target_count"]}, sort_keys=True))
    else:
        print(json.dumps({"verdict": receipt["verdict"], "receipt": str(output_dir / "receipt.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
