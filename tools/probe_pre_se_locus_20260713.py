#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable $0 local n600 probe for the Round-5 PRE-SE feature loci."""

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
from collections.abc import Iterator, Sequence
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
import probe_replace_round5_deeper_nonlinear as round5  # noqa: E402

from tac.scorer_surrogate.frozen_replay_convex_head import (  # noqa: E402
    ReplayAssignment,
    deterministic_replay_assignments,
)
from tac.scorer_surrogate.pre_se_locus_20260713 import (  # noqa: E402
    AUTHORITY_SCOPE,
    SCHEMA,
    PreSECutCostLedger,
    PreSEPairGatedMLPWeights,
    capture_pre_se_teacher,
    pre_se_feature_snapshot,
    pre_se_pair_block_features,
    pre_se_pair_gated_logits_numpy,
    verify_pre_se_taps,
)
from tac.scorer_surrogate.replace_round4_support_ranking import (  # noqa: E402
    ORDERED_PAIR_COUNT,
    QuadraticStatistics,
    array_sha256,
    block_scores,
    exact_support_target,
    pairwise_rank_block_statistics,
)
from tac.witness_dsl.pre_se_locus_policy_20260713 import (  # noqa: E402
    LOCUS_SPECS,
    PreSELocusPolicy,
    PreSELocusSpec,
)

LANE_ID = "lane_replace_round5_pre_se_locus_20260713"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]"
DEFAULT_OUTPUT = REPO / "experiments/results/pre_se_locus_20260713"
PREREGISTRATION = DEFAULT_OUTPUT / "preregistration.json"
STORAGE_PREFLIGHT = REPO / ".omx/research/pre_se_locus_storage_preflight_20260713.json"
ROUND5_OUTPUT = REPO / "experiments/results/replace_round5_deeper_nonlinear_20260713"
ROUND5_RECEIPT = ROUND5_OUTPUT / "receipt.json"

SOURCE_FILES = (
    "src/tac/scorer_surrogate/pre_se_locus_20260713.py",
    "src/tac/witness_dsl/pre_se_locus_policy_20260713.py",
    "tools/probe_pre_se_locus_20260713.py",
    "src/tac/scorer_surrogate/replace_round5_deeper_nonlinear.py",
    "src/tac/witness_dsl/replace_round5_deeper_nonlinear_policy.py",
    "tools/probe_replace_round5_deeper_nonlinear.py",
    "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
    "src/tac/scorer_surrogate/replace_round3_fidelity_wall.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
)


class ProbeError(RuntimeError):
    """Measurement or evidence custody failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_normalized(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    result = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing measurement source {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for relative, custody in _source_fingerprints().items():
        source = REPO / relative
        destination = output_dir / "source_bundle" / relative
        if destination.exists():
            if destination.stat().st_size != custody["bytes"] or _sha256(destination) != custody[
                "sha256"
            ]:
                raise ProbeError(f"source bundle drift for {relative}")
        else:
            round5._atomic_bytes(destination, source.read_bytes())
        result[relative] = {**custody, "path": str(destination.relative_to(output_dir))}
    return result


def _write_preregistration(policy: PreSELocusPolicy) -> dict[str, Any]:
    payload = {
        "schema": "pre_se_locus_preregistration.v1",
        "sealed_at_utc": _utc_now(),
        "sealed_before_any_pre_se_heldout_teacher_call": True,
        "measurement_contract": policy.compile_measurement_contract(),
        "rungs_by_locus": {
            spec.name: [
                {
                    "name": "convex-deeper-pair-block-mp",
                    "target": "exact top-area input-costate L2-square support",
                    "feature_count": spec.feature_count,
                    "fit": (
                        "twenty exact float64 pair-block RankRLS Moore-Penrose optima"
                    ),
                },
                {
                    "name": "nonlinear-pair-gated-mlp-ensemble",
                    "target": "same exact top-area support",
                    "feature_count": spec.feature_count,
                    "fit": (
                        f"three deterministic {spec.feature_count}-to-32-to-20 "
                        "pair-gated ReLU MLP seeds"
                    ),
                },
            ]
            for spec in LOCUS_SPECS
        },
        "only_change_from_round5": (
            "separate block2/block3 last-MBConv depthwise activation captured by the target "
            "SE forward-pre hook instead of the post-SE stage outputs"
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
            raise ProbeError("existing preregistration differs from the typed policy")
        return existing
    round5._atomic_json(PREREGISTRATION, payload)
    return payload


def _validate_preregistration(policy: PreSELocusPolicy) -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise ProbeError("PRE-SE preregistration is missing")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload["measurement_contract"] != _json_normalized(
        policy.compile_measurement_contract()
    ):
        raise ProbeError("typed PRE-SE policy and preregistration disagree")
    ledger = DEFAULT_OUTPUT / "teacher_calls.jsonl"
    if (
        ledger.exists()
        and any(
            json.loads(line).get("event") == "exact_teacher_state_call_started"
            for line in ledger.read_text().splitlines()
            if line.strip()
        )
        and payload.get("sealed_before_any_pre_se_heldout_teacher_call") is not True
    ):
        raise ProbeError("PRE-SE preregistration did not precede teacher calls")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO)),
        "bytes": PREREGISTRATION.stat().st_size,
        "sha256": _sha256(PREREGISTRATION),
        "payload": payload,
    }


def _storage_custody(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing PRE-SE storage waterfall preflight")
    payload = json.loads(STORAGE_PREFLIGHT.read_text())
    if payload.get("blockers"):
        raise ProbeError(f"storage preflight blocked: {payload['blockers']}")
    selected = Path(payload.get("selected_workload_root", "")).resolve()
    if selected != output_dir.resolve():
        raise ProbeError("storage preflight selected a different workload root")
    free_bytes = os.statvfs(output_dir).f_bavail * os.statvfs(output_dir).f_frsize
    if free_bytes < int(payload["requested_bytes"]):
        raise ProbeError("storage free space fell below the preregistered request")
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


def _round5_training_target_path(pair_index: int) -> Path:
    return ROUND5_OUTPUT / "train_targets" / f"pair_{pair_index:04d}.npz"


def _verify_round5_target_custody(
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
) -> dict[str, Any]:
    manifest_path = output_dir / "inherited_round5_train_targets.json"
    complete_path = ROUND5_OUTPUT / "complete.json"
    if not complete_path.is_file() or not ROUND5_RECEIPT.is_file():
        raise ProbeError("Round-5 completion custody is missing")
    complete = json.loads(complete_path.read_text())
    if (
        complete.get("receipt") != "receipt.json"
        or ROUND5_RECEIPT.stat().st_size != complete.get("bytes")
        or _sha256(ROUND5_RECEIPT) != complete.get("sha256")
    ):
        raise ProbeError("Round-5 receipt custody drifted")
    receipt = json.loads(ROUND5_RECEIPT.read_text())
    accounting = receipt["teacher_call_accounting"]
    if not (
        accounting["all_n600_states_completed"]
        and accounting["campaign_honest_started_calls"] == 600
        and accounting["retry_starts_charged"] == 0
    ):
        raise ProbeError("Round-5 is not a complete retry-free n600 source")
    train = [assignment for assignment in assignments if assignment.split == "train"]
    if len(train) != 480:
        raise ProbeError("inherited Round-5 train split is not n480")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_round5_receipt_sha256"] != _sha256(ROUND5_RECEIPT):
            raise ProbeError("inherited Round-5 manifest receipt drifted")
        for row in manifest["targets"]:
            path = REPO / row["path"]
            if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                raise ProbeError(f"inherited target drifted: {row['path']}")
        return manifest
    rows = []
    for assignment in train:
        path = _round5_training_target_path(assignment.pair_index)
        round5._load_target(path, assignment)
        with np.load(path, allow_pickle=False) as archive:
            costate_sha = str(archive["costate_sha256"].item())
        rows.append(
            {
                "pair_index": assignment.pair_index,
                "checkpoint_index": assignment.checkpoint_index,
                "checkpoint_name": assignment.checkpoint_name,
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "costate_array_sha256": costate_sha,
            }
        )
    manifest = {
        "schema": "pre_se_inherited_round5_targets.v1",
        "created_at_utc": _utc_now(),
        "source_round5_receipt": str(ROUND5_RECEIPT.relative_to(REPO)),
        "source_round5_receipt_sha256": _sha256(ROUND5_RECEIPT),
        "source_round5_receipt_bytes": ROUND5_RECEIPT.stat().st_size,
        "target_count": len(rows),
        "exact_target_authority": (
            "MEASURED-INHERITED exact input-costate L2-square support/mass from Round 5"
        ),
        "targets": rows,
    }
    round5._atomic_json(manifest_path, manifest)
    return manifest


def _locus_root(output_dir: Path, spec: PreSELocusSpec) -> Path:
    return output_dir / "loci" / spec.name


def _empty_accumulator(feature_count: int) -> dict[str, np.ndarray]:
    return {
        "completed_pairs": np.empty(0, dtype=np.int64),
        "gram": np.zeros(
            (ORDERED_PAIR_COUNT, feature_count, feature_count), dtype=np.float64
        ),
        "rhs": np.zeros((ORDERED_PAIR_COUNT, feature_count), dtype=np.float64),
        "target_square": np.zeros(ORDERED_PAIR_COUNT, dtype=np.float64),
        "row_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "state_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
    }


def _load_accumulator(path: Path, feature_count: int) -> dict[str, np.ndarray]:
    expected = _empty_accumulator(feature_count)
    if not path.is_file():
        return expected
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    if result.keys() != expected.keys() or any(
        key != "completed_pairs" and result[key].shape != expected[key].shape
        for key in expected
    ):
        raise ProbeError("PRE-SE convex accumulator schema drift")
    completed = result["completed_pairs"]
    if completed.dtype != np.int64 or not np.array_equal(completed, np.unique(completed)):
        raise ProbeError("PRE-SE completed-pair index drift")
    return result


def _stats(accumulator: dict[str, np.ndarray], block: int) -> QuadraticStatistics:
    return QuadraticStatistics(
        gram=accumulator["gram"][block],
        rhs=accumulator["rhs"][block],
        target_square=float(accumulator["target_square"][block]),
        row_count=int(accumulator["row_count"][block]),
        state_count=int(accumulator["state_count"][block]),
    )


def _add_pairwise(
    accumulator: dict[str, np.ndarray],
    assignment: ReplayAssignment,
    records: Sequence[QuadraticStatistics],
) -> None:
    for block, record in enumerate(records):
        accumulator["gram"][block] += record.gram
        accumulator["rhs"][block] += record.rhs
        accumulator["target_square"][block] += record.target_square
        accumulator["row_count"][block] += record.row_count
        accumulator["state_count"][block] += record.state_count
    accumulator["completed_pairs"] = np.sort(
        np.append(accumulator["completed_pairs"], assignment.pair_index).astype(np.int64)
    )


def _locus_tensor(spec: PreSELocusSpec, block2: Any, block3: Any) -> Any:
    return block2 if spec.name == "block2-pre-se" else block3


def _training_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: PreSELocusPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    stage_path = output_dir / "stage_train_complete.json"
    accumulators = {
        spec.name: _load_accumulator(
            _locus_root(output_dir, spec) / "train_accumulator_current.npz",
            spec.feature_count,
        )
        for spec in LOCUS_SPECS
    }
    completed = {
        name: {int(value) for value in accumulator["completed_pairs"]}
        for name, accumulator in accumulators.items()
    }
    train = [assignment for assignment in assignments if assignment.split == "train"]
    ledger = output_dir / "teacher_calls.jsonl"
    parity_rows = []
    stage_checkpoints: dict[str, list[dict[str, Any]]] = {
        spec.name: [] for spec in LOCUS_SPECS
    }
    cost_path = output_dir / "pre_se_cut_cost_model.json"
    tap_path = output_dir / "pre_se_tap_verification.json"
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(
        round2.CHECKPOINTS
    ):
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.base.n_pairs or code.shape[0] != 2 * policy.base.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not the sealed n600 renderer")
        parity = round2._checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        for assignment in (
            row for row in train if row.checkpoint_index == checkpoint_index
        ):
            missing = [
                spec for spec in LOCUS_SPECS if assignment.pair_index not in completed[spec.name]
            ]
            if not missing:
                continue
            support, _mass, pair_ids = round5._load_target(
                _round5_training_target_path(assignment.pair_index), assignment
            )
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            if not cost_path.exists() or not tap_path.exists():
                with PreSECutCostLedger(segnet) as cost_ledger:
                    tap_verification = verify_pre_se_taps(segnet, frame)
                round5._atomic_json(cost_path, cost_ledger.summary())
                round5._atomic_json(tap_path, tap_verification)
            prefix, block2, block3 = pre_se_feature_snapshot(segnet, frame)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            event_loci = {}
            for spec in missing:
                feature_tensor = _locus_tensor(spec, block2, block3)
                features, reconstructed_pairs = pre_se_pair_block_features(
                    prefix.cpu().numpy(),
                    feature_tensor.cpu().numpy(),
                    label,
                    margin,
                    pair_ids,
                    locus=spec.name,
                    checkpoint_index=assignment.checkpoint_index,
                    checkpoint_count=policy.base.checkpoint_count,
                    stride=policy.base.train_lattice_stride_on_prefix,
                )
                if not np.array_equal(reconstructed_pairs, pair_ids.reshape(-1)):
                    raise ProbeError("cached and reconstructed pair rows differ")
                records = pairwise_rank_block_statistics(
                    features.astype(np.float64), reconstructed_pairs, support.reshape(-1)
                )
                _add_pairwise(accumulators[spec.name], assignment, records)
                accumulator_path = (
                    _locus_root(output_dir, spec) / "train_accumulator_current.npz"
                )
                round5._atomic_npz(accumulator_path, **accumulators[spec.name])
                completed[spec.name].add(assignment.pair_index)
                event_loci[spec.name] = {
                    "feature_array_sha256": array_sha256(feature_tensor.cpu().numpy()),
                    "accumulator_sha256": _sha256(accumulator_path),
                }
            round2._append_jsonl(
                ledger,
                {
                    "event": "pre_se_training_features_checkpointed",
                    "timestamp_utc": _utc_now(),
                    "pair_index": assignment.pair_index,
                    "source_target": str(
                        _round5_training_target_path(assignment.pair_index).relative_to(REPO)
                    ),
                    "loci": event_loci,
                    "exact_teacher_call": False,
                },
            )
        for spec in LOCUS_SPECS:
            preserved = (
                _locus_root(output_dir, spec)
                / "stage_checkpoints"
                / f"train_{checkpoint_name}_complete.npz"
            )
            if not preserved.exists():
                round5._atomic_npz(preserved, **accumulators[spec.name])
            stage_checkpoints[spec.name].append(
                {
                    "checkpoint_name": checkpoint_name,
                    "path": str(preserved.relative_to(output_dir)),
                    "bytes": preserved.stat().st_size,
                    "sha256": _sha256(preserved),
                    "completed_train_states_cumulative": int(
                        accumulators[spec.name]["completed_pairs"].size
                    ),
                }
            )
    if any(len(completed[spec.name]) != len(train) for spec in LOCUS_SPECS):
        raise ProbeError("training stage did not cover every registered state and locus")
    stage = {
        "schema": "pre_se_locus_train_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(train),
        "exact_targets": "MEASURED-INHERITED immutable Round-5 compact targets",
        "fresh_exact_teacher_calls": 0,
        "feature_forward_states": len(train),
        "loci": {},
        "checkpoint_parity": parity_rows,
        "pre_se_cut_cost_model": {
            "path": str(cost_path.relative_to(output_dir)),
            "bytes": cost_path.stat().st_size,
            "sha256": _sha256(cost_path),
        },
        "pre_se_tap_verification": {
            "path": str(tap_path.relative_to(output_dir)),
            "bytes": tap_path.stat().st_size,
            "sha256": _sha256(tap_path),
        },
    }
    for spec in LOCUS_SPECS:
        accumulator_path = _locus_root(output_dir, spec) / "train_accumulator_current.npz"
        stage["loci"][spec.name] = {
            "feature_count": spec.feature_count,
            "accumulator": {
                "path": str(accumulator_path.relative_to(output_dir)),
                "bytes": accumulator_path.stat().st_size,
                "sha256": _sha256(accumulator_path),
            },
            "preserved_stage_checkpoints": stage_checkpoints[spec.name],
        }
    round5._atomic_json(stage_path, stage)
    return accumulators, stage


def _fit_convex_loci(
    output_dir: Path,
    accumulators: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = {}
    stages = {}
    for spec in LOCUS_SPECS:
        fits = tuple(
            round5.fit_exact_quadratic(_stats(accumulators[spec.name], block))
            for block in range(ORDERED_PAIR_COUNT)
        )
        locus_weights = np.stack([fit.weights for fit in fits]).astype(np.float64)
        weights_path = _locus_root(output_dir, spec) / "fit" / "convex_weights.npz"
        if weights_path.exists():
            with np.load(weights_path, allow_pickle=False) as archive:
                if not np.array_equal(archive["weights"], locus_weights):
                    raise ProbeError(f"{spec.name} convex weights drifted on resume")
        else:
            round5._atomic_npz(weights_path, weights=locus_weights)
        certificates = [fit.certificate for fit in fits]
        if not all(row["normal_equation_optimum_certified"] for row in certificates):
            raise ProbeError(f"{spec.name} lacks a convex optimum certificate")
        stage_path = _locus_root(output_dir, spec) / "stage_convex_fit_complete.json"
        stage = {
            "schema": "pre_se_locus_convex_fit_stage.v1",
            "completed_at_utc": _utc_now(),
            "locus": spec.name,
            "feature_count": spec.feature_count,
            "exact_optimum_class": "float64 pair-block RankRLS Moore-Penrose",
            "block_certificates": certificates,
            "weights": {
                "path": str(weights_path.relative_to(output_dir)),
                "bytes": weights_path.stat().st_size,
                "sha256": _sha256(weights_path),
                "array_sha256": array_sha256(locus_weights),
            },
        }
        round5._atomic_json(stage_path, stage)
        weights[spec.name] = locus_weights
        stages[spec.name] = stage
    aggregate = {
        "schema": "pre_se_locus_convex_fits.v1",
        "completed_at_utc": _utc_now(),
        "loci": stages,
    }
    round5._atomic_json(output_dir / "stage_convex_fit_complete.json", aggregate)
    return weights, aggregate


def _nonlinear_chunk_path(
    output_dir: Path, spec: PreSELocusSpec, checkpoint_name: str
) -> Path:
    return _locus_root(output_dir, spec) / "nonlinear" / "stage_chunks" / f"{checkpoint_name}.npz"


def _build_nonlinear_chunk(
    *,
    output_dir: Path,
    spec: PreSELocusSpec,
    checkpoint_index: int,
    checkpoint_name: str,
    renderer: Any,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: PreSELocusPolicy,
    segnet: Any,
) -> dict[str, np.ndarray]:
    path = _nonlinear_chunk_path(output_dir, spec, checkpoint_name)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            data = {name: np.array(archive[name], copy=True) for name in archive.files}
        if int(data["checkpoint_index"].item()) != checkpoint_index:
            raise ProbeError(f"{spec.name} nonlinear chunk checkpoint drift")
        return data
    core_x = []
    core_pair = []
    core_y = []
    dev_x = []
    dev_pair = []
    dev_y = []
    dev_mass = []
    dev_lengths = []
    for assignment in assignments:
        support, mass, pair_ids = round5._load_target(
            _round5_training_target_path(assignment.pair_index), assignment
        )
        frame = round2._render_state_nchw(renderer, assignment.pair_index)
        prefix, block2, block3 = pre_se_feature_snapshot(segnet, frame)
        label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
        margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
        features, pairs = pre_se_pair_block_features(
            prefix.cpu().numpy(),
            _locus_tensor(spec, block2, block3).cpu().numpy(),
            label,
            margin,
            pair_ids,
            locus=spec.name,
            checkpoint_index=assignment.checkpoint_index,
            checkpoint_count=policy.base.checkpoint_count,
            stride=policy.base.train_lattice_stride_on_prefix,
        )
        support_rows = support.reshape(-1)
        if (
            assignment.pair_index % policy.base.nonlinear_dev_modulus
            == policy.base.nonlinear_dev_remainder
        ):
            dev_x.append(features)
            dev_pair.append(pairs)
            dev_y.append(support_rows)
            dev_mass.append(mass.reshape(-1))
            dev_lengths.append(features.shape[0])
        else:
            keep = round5._balanced_core_indices(
                support_rows, seed=policy.base.seed + assignment.pair_index
            )
            core_x.append(features[keep])
            core_pair.append(pairs[keep])
            core_y.append(support_rows[keep])
    data = {
        "checkpoint_index": np.asarray(checkpoint_index, dtype=np.int64),
        "core_x": np.concatenate(core_x).astype(np.float32),
        "core_pair": np.concatenate(core_pair).astype(np.int16),
        "core_y": np.concatenate(core_y).astype(np.float32),
        "dev_x": np.concatenate(dev_x).astype(np.float32),
        "dev_pair": np.concatenate(dev_pair).astype(np.int16),
        "dev_y": np.concatenate(dev_y).astype(np.bool_),
        "dev_mass": np.concatenate(dev_mass).astype(np.float32),
        "dev_lengths": np.asarray(dev_lengths, dtype=np.int64),
    }
    round5._atomic_npz(path, **data)
    return data


def _build_nonlinear_data(
    *,
    output_dir: Path,
    spec: PreSELocusSpec,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: PreSELocusPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    chunks = []
    chunk_custody = []
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(
        round2.CHECKPOINTS
    ):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        rows = [
            assignment
            for assignment in assignments
            if assignment.split == "train" and assignment.checkpoint_index == checkpoint_index
        ]
        chunk = _build_nonlinear_chunk(
            output_dir=output_dir,
            spec=spec,
            checkpoint_index=checkpoint_index,
            checkpoint_name=checkpoint_name,
            renderer=renderer,
            assignments=rows,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
        )
        chunks.append(chunk)
        path = _nonlinear_chunk_path(output_dir, spec, checkpoint_name)
        chunk_custody.append(
            {
                "checkpoint_name": checkpoint_name,
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
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
    data["dev_offsets"] = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(dev_lengths, dtype=np.int64))
    )
    mean = data["core_x"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = data["core_x"].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    data["feature_mean"] = mean
    data["feature_std"] = std
    data["core_x"] = np.ascontiguousarray((data["core_x"] - mean) / std, dtype=np.float32)
    data["dev_x"] = np.ascontiguousarray((data["dev_x"] - mean) / std, dtype=np.float32)
    normalization_path = _locus_root(output_dir, spec) / "nonlinear" / "normalization.npz"
    round5._atomic_npz(normalization_path, feature_mean=mean, feature_std=std)
    stage = {
        "schema": "pre_se_locus_nonlinear_data_stage.v1",
        "completed_at_utc": _utc_now(),
        "locus": spec.name,
        "core_states": policy.base.nonlinear_core_state_count,
        "dev_states": policy.base.nonlinear_dev_state_count,
        "core_balanced_rows": int(data["core_x"].shape[0]),
        "dev_full_rows": int(data["dev_x"].shape[0]),
        "feature_count": spec.feature_count,
        "preserved_renderer_stage_chunks": chunk_custody,
        "normalization": {
            "path": str(normalization_path.relative_to(output_dir)),
            "bytes": normalization_path.stat().st_size,
            "sha256": _sha256(normalization_path),
        },
    }
    stage_path = _locus_root(output_dir, spec) / "stage_nonlinear_data_complete.json"
    round5._atomic_json(stage_path, stage)
    return data, stage


@contextmanager
def _round5_width_adapter(spec: PreSELocusSpec) -> Iterator[None]:
    original = (
        round5.DEEP_FEATURE_COUNT,
        round5.PairGatedMLPWeights,
        round5.pair_gated_logits_numpy,
    )
    round5.DEEP_FEATURE_COUNT = spec.feature_count
    round5.PairGatedMLPWeights = PreSEPairGatedMLPWeights
    round5.pair_gated_logits_numpy = pre_se_pair_gated_logits_numpy
    try:
        yield
    finally:
        (
            round5.DEEP_FEATURE_COUNT,
            round5.PairGatedMLPWeights,
            round5.pair_gated_logits_numpy,
        ) = original


def _fit_nonlinear_loci(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: PreSELocusPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[
    dict[str, tuple[PreSEPairGatedMLPWeights, ...]],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    all_weights = {}
    normalizations = {}
    stages = {}
    for spec in LOCUS_SPECS:
        data, data_stage = _build_nonlinear_data(
            output_dir=output_dir,
            spec=spec,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        with _round5_width_adapter(spec):
            weights, fit_stage = round5._fit_nonlinear_stage(
                output_dir=_locus_root(output_dir, spec),
                data=data,
                policy=policy.base,
            )
        all_weights[spec.name] = weights
        normalizations[spec.name] = {
            "feature_mean": np.array(data["feature_mean"], copy=True),
            "feature_std": np.array(data["feature_std"], copy=True),
        }
        stages[spec.name] = {"data": data_stage, "fit": fit_stage}
        del data
        gc.collect()
    aggregate = {
        "schema": "pre_se_locus_nonlinear_fits.v1",
        "completed_at_utc": _utc_now(),
        "loci": stages,
    }
    round5._atomic_json(output_dir / "stage_nonlinear_fit_complete.json", aggregate)
    return all_weights, normalizations, aggregate


def _heldout_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout" / f"pair_{pair_index:04d}.json"


def _teacher_call(
    *,
    ledger: Path,
    assignment: ReplayAssignment,
    frame_nchw: Any,
    labels_t: Any,
    segnet: Any,
) -> tuple[Any, Any, Any, Any, np.ndarray, dict[str, float], float]:
    batch_id = f"pre-se-heldout-p{assignment.pair_index:04d}-{os.getpid()}"
    round2._teacher_start(
        ledger, assignment, stage="pre_se_locus_heldout", batch_id=batch_id
    )
    result = capture_pre_se_teacher(
        segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
    )
    round2._teacher_complete(
        ledger,
        assignment,
        stage="pre_se_locus_heldout",
        batch_id=batch_id,
        teacher_metrics=result[5],
        elapsed_seconds=result[6],
    )
    return result


def _heldout_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: PreSELocusPolicy,
    convex_weights: dict[str, np.ndarray],
    nonlinear_weights: dict[str, tuple[PreSEPairGatedMLPWeights, ...]],
    normalizations: dict[str, dict[str, np.ndarray]],
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    import torch

    heldout = [assignment for assignment in assignments if assignment.split == "heldout"]
    ledger = output_dir / "teacher_calls.jsonl"
    records = []
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in (
            row for row in heldout if row.checkpoint_index == checkpoint_index
        ):
            record_path = _heldout_record_path(output_dir, assignment.pair_index)
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            prefix, block2, block3, costate, pair_ids, metrics, elapsed = _teacher_call(
                ledger=ledger,
                assignment=assignment,
                frame_nchw=frame,
                labels_t=torch.as_tensor(label[None], dtype=torch.long),
                segnet=segnet,
            )
            mass, support, selected = exact_support_target(
                costate.cpu().numpy(), area_fraction=policy.base.requested_area_fraction
            )
            if selected != policy.base.selected_prefix_cells:
                raise ProbeError("heldout support area drifted")
            locus_records = {}
            for spec in LOCUS_SPECS:
                features, pairs = pre_se_pair_block_features(
                    prefix.cpu().numpy(),
                    _locus_tensor(spec, block2, block3).cpu().numpy(),
                    label,
                    margin,
                    pair_ids,
                    locus=spec.name,
                    checkpoint_index=assignment.checkpoint_index,
                    checkpoint_count=policy.base.checkpoint_count,
                    stride=policy.base.heldout_lattice_stride_on_prefix,
                )
                convex_scores = block_scores(
                    features.astype(np.float64), pairs, convex_weights[spec.name]
                )
                normalization = normalizations[spec.name]
                standardized = np.ascontiguousarray(
                    (features - normalization["feature_mean"])
                    / normalization["feature_std"],
                    dtype=np.float32,
                )
                seed_logits = np.stack(
                    [
                        pre_se_pair_gated_logits_numpy(standardized, pairs, weights)
                        for weights in nonlinear_weights[spec.name]
                    ]
                )
                seed_probabilities = round5.sigmoid_probabilities(seed_logits)
                ensemble = seed_probabilities.mean(axis=0, dtype=np.float64).astype(np.float32)
                selections = {
                    "convex-deeper-pair-block-mp": round5._selection_metrics(
                        mass, convex_scores, selected
                    ),
                    "nonlinear-pair-gated-mlp-ensemble": round5._selection_metrics(
                        mass, ensemble, selected
                    ),
                    "oracle": {
                        "total_mass": float(mass.sum(dtype=np.float64)),
                        "retained_mass": float(
                            np.sort(mass.reshape(-1).astype(np.float64))[-selected:].sum(
                                dtype=np.float64
                            )
                        ),
                    },
                }
                for index, seed in enumerate(policy.base.nonlinear_seeds):
                    selections[f"nonlinear-seed-{seed}"] = round5._selection_metrics(
                        mass, seed_probabilities[index], selected
                    )
                query = round5.disagreement_query_audit(
                    seed_probabilities,
                    support.reshape(-1),
                    seed=policy.base.seed + assignment.pair_index,
                    targeted_fraction=policy.base.query_targeted_fraction,
                    random_audit_fraction=policy.base.query_random_audit_fraction,
                )
                locus_records[spec.name] = {
                    "feature_array_sha256": array_sha256(
                        _locus_tensor(spec, block2, block3).cpu().numpy()
                    ),
                    "selections": selections,
                    "ensemble_ece_sums": round5._ece_sums(
                        ensemble, support.reshape(-1)
                    ),
                    "query_real_calibration": query,
                }
            record = {
                "schema": "pre_se_locus_heldout_state.v1",
                "pair_index": assignment.pair_index,
                "checkpoint_index": assignment.checkpoint_index,
                "checkpoint_name": assignment.checkpoint_name,
                "selected_prefix_cells": selected,
                "realized_area_fraction": policy.base.realized_area_fraction,
                "loci": locus_records,
                "teacher_metrics": metrics,
                "teacher_elapsed_seconds": elapsed,
                "costate_sha256": array_sha256(costate.cpu().numpy()),
            }
            round5._atomic_json(record_path, record)
            records.append(record)
    if len(records) != policy.base.heldout_state_count:
        raise ProbeError("heldout state count drift")
    aggregate = {}
    query_summary = {}
    for spec in LOCUS_SPECS:
        names = [
            "convex-deeper-pair-block-mp",
            "nonlinear-pair-gated-mlp-ensemble",
            *(f"nonlinear-seed-{seed}" for seed in policy.base.nonlinear_seeds),
            "oracle",
        ]
        aggregate[spec.name] = {}
        for name in names:
            total = sum(
                row["loci"][spec.name]["selections"][name]["total_mass"]
                for row in records
            )
            retained = sum(
                row["loci"][spec.name]["selections"][name]["retained_mass"]
                for row in records
            )
            aggregate[spec.name][name] = {
                "total_mass": total,
                "retained_mass": retained,
                "retained_mass_fraction": retained / total,
                "conditional_masked_exact_cosine": math.sqrt(retained / total),
            }
        query_numeric = (
            "high_to_low_error_ratio",
            "spearman_disagreement_vs_absolute_error",
            "queried_absolute_error_fraction",
            "realized_query_fraction",
            "random_audit_positive_propensity",
        )
        query_summary[spec.name] = {
            key: float(
                np.mean(
                    [
                        row["loci"][spec.name]["query_real_calibration"][key]
                        for row in records
                    ]
                )
            )
            for key in query_numeric
        }
    stage = {
        "schema": "pre_se_locus_heldout_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(records),
        "aggregate": aggregate,
        "query_real_calibration": query_summary,
    }
    round5._atomic_json(output_dir / "stage_heldout_complete.json", stage)
    return stage


def _teacher_accounting(
    output_dir: Path,
    policy: PreSELocusPolicy,
    inherited_manifest: dict[str, Any],
    assignments: Sequence[ReplayAssignment],
) -> dict[str, Any]:
    ledger = output_dir / "teacher_calls.jsonl"
    starts = []
    completions = []
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "exact_teacher_state_call_started":
            starts.append(row)
        elif row.get("event") == "exact_teacher_state_call_completed":
            completions.append(row)
    inherited = {int(row["pair_index"]) for row in inherited_manifest["targets"]}
    current_started = {int(row["pair_index"]) for row in starts}
    current_completed = {int(row["pair_index"]) for row in completions}
    heldout = {row.pair_index for row in assignments if row.split == "heldout"}
    if inherited & heldout:
        raise ProbeError("inherited train and fresh heldout target sets overlap")
    unique_completed = inherited | current_completed
    campaign_starts = len(inherited) + len(starts)
    return {
        "schema": "pre_se_locus_teacher_accounting.v1",
        "inherited_round5_exact_train_targets": len(inherited),
        "fresh_exact_heldout_started_calls": len(starts),
        "fresh_exact_heldout_completed_calls": len(completions),
        "fresh_exact_heldout_unique_started_states": len(current_started),
        "fresh_exact_heldout_unique_completed_states": len(current_completed),
        "retry_starts_charged": len(starts) - len(current_started),
        "campaign_honest_started_calls": campaign_starts,
        "started_call_budget": policy.base.teacher_started_call_budget,
        "budget_gate_pass": campaign_starts <= policy.base.teacher_started_call_budget,
        "all_n600_states_completed": unique_completed == set(range(policy.base.n_pairs)),
        "source_target_manifest_sha256": _sha256(
            output_dir / "inherited_round5_train_targets.json"
        ),
        "ledger": {
            "path": str(ledger.relative_to(output_dir)),
            "bytes": ledger.stat().st_size,
            "sha256": _sha256(ledger),
        },
    }


def _decision(
    *,
    heldout: dict[str, Any],
    accounting: dict[str, Any],
    policy: PreSELocusPolicy,
    tap_verification: dict[str, Any],
) -> dict[str, Any]:
    loci = {}
    for spec in LOCUS_SPECS:
        aggregate = heldout["aggregate"][spec.name]
        convex = aggregate["convex-deeper-pair-block-mp"]["retained_mass_fraction"]
        nonlinear = aggregate["nonlinear-pair-gated-mlp-ensemble"][
            "retained_mass_fraction"
        ]
        seeds = np.asarray(
            [
                aggregate[f"nonlinear-seed-{seed}"]["retained_mass_fraction"]
                for seed in policy.base.nonlinear_seeds
            ],
            dtype=np.float64,
        )
        seed_std = float(seeds.std(ddof=0))
        convex_pass = bool(convex >= policy.base.retained_mass_bar)
        nonlinear_pass = bool(
            nonlinear >= policy.base.retained_mass_bar
            and seed_std <= policy.base.nonlinear_seed_std_bar
            and accounting["budget_gate_pass"]
            and accounting["all_n600_states_completed"]
        )
        tileable = bool(
            tap_verification[spec.name]["strict_end_to_end_independently_tileable_from_rgb"]
        )
        loci[spec.name] = {
            "rungs": {
                "convex-deeper-pair-block-mp": {
                    "retained_mass_fraction": convex,
                    "primary_mass_gate_pass": convex_pass,
                    "exact_optimum_certificates": True,
                },
                "nonlinear-pair-gated-mlp-ensemble": {
                    "retained_mass_fraction": nonlinear,
                    "primary_mass_gate_pass": nonlinear >= policy.base.retained_mass_bar,
                    "seed_retained_mass": seeds.tolist(),
                    "seed_population_std": seed_std,
                    "stability_gate_pass": seed_std <= policy.base.nonlinear_seed_std_bar,
                    "teacher_economics_gate_pass": accounting["budget_gate_pass"],
                    "admitted": nonlinear_pass,
                },
            },
            "retained_mass_gate_pass": convex_pass or nonlinear_pass,
            "strict_tileability_gate_pass": tileable,
            "joint_reopen_gate_pass": tileable and (convex_pass or nonlinear_pass),
            "oracle_retained_mass_fraction": aggregate["oracle"]["retained_mass_fraction"],
        }
    reopen = any(row["joint_reopen_gate_pass"] for row in loci.values())
    if reopen:
        verdict = "REOPEN-CHEAP-LOCALIZATION-FAMILY"
        scope = "FORMULATION x FIXED REPLAY x STRICT RGB TILEABILITY"
    else:
        verdict = "WIDER-FAMILY-KILL"
        scope = (
            "FAMILY x TESTED-SINGLE-SOURCE-LOCI x FIXED-REPLAY x "
            "STRICT-END-TO-END-RGB-TILEABILITY"
        )
    return {
        "verdict": verdict,
        "verdict_scope": scope,
        "retained_mass_bar": policy.base.retained_mass_bar,
        "requested_area_fraction": policy.base.requested_area_fraction,
        "realized_area_fraction": policy.base.realized_area_fraction,
        "oracle_retained_mass_fraction": policy.base.compile_measurement_contract()[
            "round4_oracle_retained_mass"
        ],
        "loci": loci,
        "pointer_moved": False,
    }


def _cleanup_manifest(output_dir: Path, run_contract: dict[str, Any]) -> dict[str, Any]:
    source_bundle_sha = hashlib.sha256(
        json.dumps(run_contract["sources"], sort_keys=True).encode()
    ).hexdigest()
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
                    "rebuild_command": (
                        ".venv/bin/python tools/probe_pre_se_locus_20260713.py --resume"
                    ),
                    "source_bundle_sha256": source_bundle_sha,
                }
            )
    payload = {
        "schema": "pre_se_locus_cleanup_manifest.v1",
        "created_at_utc": _utc_now(),
        "policy": "certify-or-block",
        "artifacts": artifacts,
        "deleted": [],
        "moved": [],
        "blockers": [],
    }
    round5._atomic_json(output_dir / "cleanup_manifest.json", payload)
    return payload


def run(*, output_dir: Path, resume: bool, validate_only: bool) -> dict[str, Any]:
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ProbeError("the preregistered PRE-SE instance has one sealed output directory")
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        if not resume:
            raise ProbeError("completed PRE-SE receipt exists; pass --resume to verify")
        complete = json.loads(complete_path.read_text())
        receipt_path = output_dir / complete["receipt"]
        if (
            receipt_path.stat().st_size != complete["bytes"]
            or _sha256(receipt_path) != complete["sha256"]
        ):
            raise ProbeError("completed PRE-SE receipt custody drift")
        return json.loads(receipt_path.read_text())
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = PreSELocusPolicy()
    contract = policy.compile_measurement_contract()
    preregistration = _validate_preregistration(policy)
    storage = _storage_custody(output_dir)
    inputs = round2._verify_input_custody()
    assignments = deterministic_replay_assignments(
        n_pairs=policy.base.n_pairs,
        checkpoint_names=tuple(row[0] for row in round2.CHECKPOINTS),
        holdout_period=policy.base.holdout_period,
        seed=policy.base.seed,
    )
    inherited = _verify_round5_target_custody(output_dir, assignments)
    if validate_only:
        return {
            "schema": "pre_se_locus_validate_only.v1",
            "compiled_policy": contract,
            "preregistration": preregistration,
            "storage": storage,
            "inputs": inputs,
            "inherited_target_count": inherited["target_count"],
            "sources": _source_fingerprints(),
            "fresh_teacher_calls": 0,
        }
    descriptor = round2._acquire_lock(output_dir)
    try:
        contract_path = output_dir / "run_contract.json"
        if contract_path.exists():
            if not resume:
                raise ProbeError("existing PRE-SE run contract requires --resume")
            run_contract = json.loads(contract_path.read_text())
            if run_contract["compiled_policy"] != _json_normalized(contract):
                raise ProbeError("PRE-SE resume policy drift")
            if run_contract["inputs"] != inputs:
                raise ProbeError("PRE-SE resume input custody drift")
            for relative, row in run_contract["sources"].items():
                bundled = output_dir / row["path"]
                if bundled.stat().st_size != row["bytes"] or _sha256(bundled) != row["sha256"]:
                    raise ProbeError(f"bundled source drift for {relative}")
        else:
            sources = _source_bundle(output_dir)
            run_contract = {
                "schema": "pre_se_locus_run_contract.v1",
                "created_at_utc": _utc_now(),
                "lane_id": LANE_ID,
                "compiled_policy": contract,
                "preregistration": preregistration,
                "inputs": inputs,
                "inherited_target_manifest": {
                    "path": "inherited_round5_train_targets.json",
                    "bytes": (output_dir / "inherited_round5_train_targets.json").stat().st_size,
                    "sha256": _sha256(output_dir / "inherited_round5_train_targets.json"),
                },
                "sources": sources,
                "runtime": _runtime_custody(torch),
                "storage_preflight": storage,
                "git_head_at_measurement": _git_head(),
                "git_status_at_measurement_start": _git_status(),
                "source_round5_read_only": True,
                "paid_or_remote_launch": False,
                "live_run_mutated": False,
                "witness_training_launched": False,
                "authority": AXIS,
            }
            round5._atomic_json(contract_path, run_contract)
        labels = round2._stored_npy_memmap(round2.GT_CACHE, "lstars.npy")
        margins = round2._stored_npy_memmap(round2.GT_CACHE, "margins.npy")
        if labels.shape != (600, 384, 512) or margins.shape != labels.shape:
            raise ProbeError("GT cache geometry drift")
        yopo = round2._load_tool_module(
            "_pre_se_locus_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
        )
        segnet = round2._load_cpu_segnet()
        accumulators, train_stage = _training_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        convex_weights, convex_stage = _fit_convex_loci(output_dir, accumulators)
        nonlinear_weights, normalizations, nonlinear_stage = _fit_nonlinear_loci(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        heldout_stage = _heldout_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            convex_weights=convex_weights,
            nonlinear_weights=nonlinear_weights,
            normalizations=normalizations,
            segnet=segnet,
            yopo=yopo,
        )
        accounting = _teacher_accounting(output_dir, policy, inherited, assignments)
        tap_verification = json.loads(
            (output_dir / "pre_se_tap_verification.json").read_text()
        )
        decision = _decision(
            heldout=heldout_stage,
            accounting=accounting,
            policy=policy,
            tap_verification=tap_verification,
        )
        cost_model = json.loads((output_dir / "pre_se_cut_cost_model.json").read_text())
        economics = {}
        for spec in LOCUS_SPECS:
            cut = cost_model["cuts"][spec.name]
            fraction = cut["fraction_of_full_teacher_conv_flops"]
            c_label = fraction + (1.0 - fraction) * policy.base.realized_area_fraction
            economics[spec.name] = {
                "form": "C_teacher = A + c_label * D",
                "feature_cut_fraction_p": fraction,
                "selected_area_fraction_q": policy.base.realized_area_fraction,
                "conditional_c_label": c_label,
                "conditional_variable_cost_reduction_x": 1.0 / c_label,
                "wall_clock_claim": False,
            }
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
            "nonlinear_fit_stage": nonlinear_stage,
            "heldout_stage": heldout_stage,
            "pre_se_tap_verification": tap_verification,
            "pre_se_cut_cost_model": cost_model,
            "teacher_call_accounting": accounting,
            "economics": economics,
            "cleanup_custody": {
                "path": "cleanup_manifest.json",
                "blockers": cleanup["blockers"],
            },
            "triality": {
                "dsl": "tac.witness_dsl.pre_se_locus_policy_20260713",
                "equation": "deferred pending clean-law audit",
                "dag_feed": ".omx/research/pre_se_locus_DAG_FEED_20260713.md",
            },
            "verdict_scope": decision["verdict_scope"],
            "reformulation_queue": [
                "SE-free or local-attention deep feature extractor",
                "donated full-frame SE-gate cache with charged broadcast accounting",
                "multi-source or dense-label localizer",
                "transition-complete causal-manifest successor",
            ],
            "pointer_delta": "NONE",
        }
        receipt_path = output_dir / "receipt.json"
        round5._atomic_json(receipt_path, receipt)
        round5._atomic_json(
            complete_path,
            {
                "schema": "pre_se_locus_completion.v1",
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
        payload = _write_preregistration(PreSELocusPolicy())
        print(
            json.dumps(
                {
                    "path": str(PREREGISTRATION),
                    "bytes": PREREGISTRATION.stat().st_size,
                    "sha256": _sha256(PREREGISTRATION),
                    "sealed_at_utc": payload["sealed_at_utc"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = run(output_dir=output_dir, resume=args.resume, validate_only=args.validate_only)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema": receipt["schema"],
                    "fresh_teacher_calls": 0,
                    "inherited_target_count": receipt["inherited_target_count"],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "verdict": receipt["verdict"]["verdict"],
                    "loci": receipt["verdict"]["loci"],
                    "receipt": str(output_dir / "receipt.json"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
