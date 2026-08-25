#!/usr/bin/env python3
"""Bounded, payload-retaining JO5 Pose6 determinism probe.

The probe replays one already-retained receiver-close candidate batch and the
selected candidate as a singleton.  It persists every camera, PoseNet input,
and Pose6 array that it materializes, so a failed comparison remains
reproducible and byte-auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo2_receiver_close as receiver_close
from experiments import ddm_jo3_joint_objective_entrypoint as entrypoint

AXIS: Final = "[macOS-CPU bounded determinism diagnostic; no score authority]"


class JO5ProbeError(RuntimeError):
    """A source, geometry, custody, or determinism invariant failed."""


def _state_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _materialize_forward(
    *,
    root: Path,
    label: str,
    surface: receiver_close.CarrierSurface,
    modules: receiver_close.RuntimeModules,
    posenet: Any,
    codes: np.ndarray,
    master: np.ndarray,
    pair: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    camera = receiver_close.render_frame0(surface, modules, codes, pair)
    masters = np.repeat(master[None], len(codes), axis=0)
    pose_input = np.stack((camera, masters), axis=1)
    pose6 = receiver_close.pose_vectors(posenet, pose_input)
    elapsed = time.perf_counter() - started
    forward_root = root / label
    payloads = {
        "codes": entrypoint.atomic_npy(forward_root / "codes.int32.npy", codes),
        "camera": entrypoint.atomic_npy(forward_root / "camera.uint8.npy", camera),
        "pose_input": entrypoint.atomic_npy(
            forward_root / "pose_input.uint8.npy", pose_input
        ),
        "pose6": entrypoint.atomic_npy(forward_root / "pose6.float32.npy", pose6),
    }
    return {
        "label": label,
        "candidate_denominator": len(codes),
        "elapsed_seconds": elapsed,
        "payloads": payloads,
        "raw_sha256": {
            "camera": _raw_sha256(camera),
            "pose_input": _raw_sha256(pose_input),
            "pose6": _raw_sha256(pose6),
        },
        "materialized_payloads_retained": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise JO5ProbeError(f"refusing existing output root: {args.output.resolve()}")
    if not 0 <= args.pair < design.N_PAIRS:
        raise JO5ProbeError("--pair is outside the n600 denominator")
    if not 1 <= args.threads <= 4:
        raise JO5ProbeError("--threads must be in [1,4]")
    if args.required_free_bytes < 0:
        raise JO5ProbeError("--required-free-bytes must be nonnegative")
    storage = shutil.disk_usage(args.output.parent)
    if storage.free < args.required_free_bytes:
        raise JO5ProbeError(
            "storage preflight refused: "
            f"free={storage.free},required={args.required_free_bytes}"
        )

    entrypoint.configure_determinism(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    # The diagnostic may intentionally replay a retained pre-cure batch after
    # the entrypoint source pin has advanced.  Validate the sealed config bytes
    # and workload identity without pretending its historical source record is
    # the current working tree.
    config = design.load_compiled_config(args.compiled_config, args.expected_config_sha256)
    _segnet, posenet, patch_receipt = entrypoint.load_scorers(config)  # SCORER_LOADER_ORDER_OK: jo3 entrypoint wrapper returns (segnet, posenet, receipt); unpack matches its verified order
    surface, modules = receiver_close.load_surface(
        Path(config.inputs.rc2_archive.path), Path(config.inputs.rc2_runtime.path)
    )

    batch_codes = np.load(args.source_batch_codes, allow_pickle=False)
    saved_pose6 = np.load(args.source_batch_pose6, allow_pickle=False)
    master_field = np.load(args.master_field, mmap_mode="r", allow_pickle=False)
    if (
        batch_codes.ndim != 2
        or batch_codes.shape[1] != receiver_close.D
        or batch_codes.dtype != np.int32
        or saved_pose6.shape != (len(batch_codes), receiver_close.POSE_DIMS)
        or saved_pose6.dtype != np.float32
    ):
        raise JO5ProbeError("retained exploration batch geometry differs")
    if not 0 <= args.selected_index < len(batch_codes):
        raise JO5ProbeError("--selected-index is outside the retained batch")
    if master_field.shape != (
        design.N_PAIRS,
        receiver_close.CAMERA_H,
        receiver_close.CAMERA_W,
        3,
    ) or master_field.dtype != np.uint8:
        raise JO5ProbeError("candidate master field geometry differs")

    args.output.mkdir(parents=True)
    source_payloads = {
        "batch_codes": entrypoint.atomic_npy(
            args.output / "source/batch_codes.int32.npy", batch_codes
        ),
        "saved_pose6": entrypoint.atomic_npy(
            args.output / "source/saved_pose6.float32.npy", saved_pose6
        ),
        "master": entrypoint.atomic_npy(
            args.output / "source/master.uint8.npy", np.asarray(master_field[args.pair])
        ),
    }
    python_rng_before = _state_sha256(random.getstate())
    numpy_rng_before = _state_sha256(np.random.get_state())
    torch_rng_before = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()

    exact_batch = _materialize_forward(
        root=args.output,
        label="exact_batch_repeat",
        surface=surface,
        modules=modules,
        posenet=posenet,
        codes=batch_codes,
        master=np.asarray(master_field[args.pair]),
        pair=args.pair,
    )
    selected_codes = np.ascontiguousarray(
        batch_codes[args.selected_index : args.selected_index + 1]
    )
    singleton = _materialize_forward(
        root=args.output,
        label="singleton_repeat",
        surface=surface,
        modules=modules,
        posenet=posenet,
        codes=selected_codes,
        master=np.asarray(master_field[args.pair]),
        pair=args.pair,
    )

    cure: dict[str, Any] | None = None
    if args.run_cure_recompute:
        retention = entrypoint.CertifiedCandidateRetention(
            solve_root=args.output / "cure_recompute",
            stage_id="jo5_determinism_proof",
            workload_config_sha256=str(config.workload_config_sha256),
            base_archive_sha256=config.inputs.rc2_archive.sha256,
        )
        exact_camera_batch = np.load(
            exact_batch["payloads"]["camera"]["path"], allow_pickle=False
        )
        exact_input_batch = np.load(
            exact_batch["payloads"]["pose_input"]["path"], allow_pickle=False
        )
        exact_pose6_batch = np.load(
            exact_batch["payloads"]["pose6"]["path"], allow_pickle=False
        )
        retention.retain_explored(
            root=args.output / "cure_recompute/exploration/batch",
            pair=args.pair,
            base_codes=np.asarray(surface.codes[args.pair], dtype=np.int32),
            codes=batch_codes,
            slave_camera=exact_camera_batch,
            pose_input=exact_input_batch,
            pose_vectors=exact_pose6_batch,
        )
        repeated = retention.recompute_selected_winner(
            root=args.output / "cure_recompute/repeat",
            pair=args.pair,
            base_codes=np.asarray(surface.codes[args.pair], dtype=np.int32),
            candidate_codes=tuple(batch_codes),
            selected_index=args.selected_index,
            master=np.asarray(master_field[args.pair]),
            surface=surface,
            modules=modules,
            posenet=posenet,
        )
        winner = retention.retain_winner(
            root=args.output / "cure_recompute/winner",
            pair=args.pair,
            base_codes=np.asarray(surface.codes[args.pair], dtype=np.int32),
            codes=np.asarray(repeated["codes"], dtype=np.int32),
            slave_camera=np.asarray(repeated["slave_camera"], dtype=np.uint8),
            pose_input=np.asarray(repeated["pose_input"], dtype=np.uint8),
            pose_vector=np.asarray(repeated["pose_vector"], dtype=np.float32),
        )
        retention.verify_winner(winner)
        cure = {
            "repeat_receipt": repeated["repeat_receipt"],
            "winner_retention": winner,
            "exact_batch_vs_cure_selected_camera_byte_identical": bool(
                np.array_equal(
                    exact_camera_batch[args.selected_index], repeated["slave_camera"]
                )
            ),
            "exact_batch_vs_cure_selected_pose_input_byte_identical": bool(
                np.array_equal(
                    exact_input_batch[args.selected_index], repeated["pose_input"]
                )
            ),
            "exact_batch_vs_cure_selected_pose6_byte_identical": bool(
                np.array_equal(
                    exact_pose6_batch[args.selected_index], repeated["pose_vector"]
                )
            ),
            "selected_camera_raw_sha256": _raw_sha256(
                np.asarray(repeated["slave_camera"], dtype=np.uint8)
            ),
            "selected_pose_input_raw_sha256": _raw_sha256(
                np.asarray(repeated["pose_input"], dtype=np.uint8)
            ),
            "selected_pose6_raw_sha256": _raw_sha256(
                np.asarray(repeated["pose_vector"], dtype=np.float32)
            ),
            "all_materialized_payloads_retained": True,
        }

    exact_pose6 = np.load(
        exact_batch["payloads"]["pose6"]["path"], allow_pickle=False
    )[args.selected_index]
    singleton_pose6 = np.load(
        singleton["payloads"]["pose6"]["path"], allow_pickle=False
    )[0]
    saved_selected = np.asarray(saved_pose6[args.selected_index])
    exact_camera = np.load(
        exact_batch["payloads"]["camera"]["path"], allow_pickle=False
    )[args.selected_index]
    singleton_camera = np.load(
        singleton["payloads"]["camera"]["path"], allow_pickle=False
    )[0]
    exact_input = np.load(
        exact_batch["payloads"]["pose_input"]["path"], allow_pickle=False
    )[args.selected_index]
    singleton_input = np.load(
        singleton["payloads"]["pose_input"]["path"], allow_pickle=False
    )[0]

    python_rng_after = _state_sha256(random.getstate())
    numpy_rng_after = _state_sha256(np.random.get_state())
    torch_rng_after = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
    receipt = {
        "schema": "ddm_jo5_pair_determinism_probe.v1",
        "axis": AXIS,
        "score_claim": False,
        "pair": args.pair,
        "selected_index": args.selected_index,
        "seed": args.seed,
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "storage_preflight": {
            "path": str(args.output.parent.resolve()),
            "free_bytes": storage.free,
            "required_free_bytes": args.required_free_bytes,
            "status": "PASS",
        },
        "source_records": {
            "compiled_config": entrypoint.file_record(args.compiled_config),
            "source_batch_codes": entrypoint.file_record(args.source_batch_codes),
            "source_batch_pose6": entrypoint.file_record(args.source_batch_pose6),
            "master_field": entrypoint.file_record(args.master_field),
        },
        "retained_source_payloads": source_payloads,
        "patch_receipt": patch_receipt,
        "forwards": [exact_batch, singleton],
        "cure_recompute": cure,
        "comparison": {
            "saved_vs_exact_batch_pose6_byte_identical": bool(
                np.array_equal(saved_selected, exact_pose6)
            ),
            "saved_vs_exact_batch_pose6_max_abs": float(
                np.max(np.abs(saved_selected.astype(np.float64) - exact_pose6))
            ),
            "saved_vs_singleton_pose6_byte_identical": bool(
                np.array_equal(saved_selected, singleton_pose6)
            ),
            "saved_vs_singleton_pose6_max_abs": float(
                np.max(np.abs(saved_selected.astype(np.float64) - singleton_pose6))
            ),
            "exact_batch_vs_singleton_pose6_byte_identical": bool(
                np.array_equal(exact_pose6, singleton_pose6)
            ),
            "exact_batch_vs_singleton_pose6_max_abs": float(
                np.max(np.abs(exact_pose6.astype(np.float64) - singleton_pose6))
            ),
            "exact_batch_vs_singleton_camera_byte_identical": bool(
                np.array_equal(exact_camera, singleton_camera)
            ),
            "exact_batch_vs_singleton_pose_input_byte_identical": bool(
                np.array_equal(exact_input, singleton_input)
            ),
            "exact_batch_selected_camera_raw_sha256": _raw_sha256(exact_camera),
            "singleton_camera_raw_sha256": _raw_sha256(singleton_camera),
            "exact_batch_selected_pose_input_raw_sha256": _raw_sha256(exact_input),
            "singleton_pose_input_raw_sha256": _raw_sha256(singleton_input),
            "saved_selected_pose6_raw_sha256": _raw_sha256(saved_selected),
            "exact_batch_selected_pose6_raw_sha256": _raw_sha256(exact_pose6),
            "singleton_pose6_raw_sha256": _raw_sha256(singleton_pose6),
        },
        "rng_state": {
            "python_unchanged": python_rng_before == python_rng_after,
            "numpy_unchanged": numpy_rng_before == numpy_rng_after,
            "torch_cpu_unchanged": torch_rng_before == torch_rng_after,
            "before": {
                "python": python_rng_before,
                "numpy": numpy_rng_before,
                "torch_cpu": torch_rng_before,
            },
            "after": {
                "python": python_rng_after,
                "numpy": numpy_rng_after,
                "torch_cpu": torch_rng_after,
            },
        },
        "all_materialized_payloads_retained": True,
    }
    entrypoint.atomic_json(args.output / "RESULT.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--compiled-config", required=True, type=Path)
    value.add_argument("--expected-config-sha256", required=True)
    value.add_argument("--source-batch-codes", required=True, type=Path)
    value.add_argument("--source-batch-pose6", required=True, type=Path)
    value.add_argument("--master-field", required=True, type=Path)
    value.add_argument("--pair", required=True, type=int)
    value.add_argument("--selected-index", required=True, type=int)
    value.add_argument("--threads", required=True, type=int)
    value.add_argument("--seed", default=20260821, type=int)
    value.add_argument("--required-free-bytes", default=2 * 1024**3, type=int)
    value.add_argument("--run-cure-recompute", action="store_true")
    value.add_argument("--output", required=True, type=Path)
    return value


def main() -> int:
    try:
        receipt = run(parser().parse_args())
    except (
        JO5ProbeError,
        design.JO1Error,
        entrypoint.JO3EntrypointError,
        receiver_close.JO2ReceiverCloseError,
    ) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "COMPLETE", "result": receipt["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
