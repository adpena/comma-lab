#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight, materialize, or reopen the G109 V9 batch-16 target capsule."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    load_compile_ready_materialization_receipt,
    load_json_mapping,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    reverify_preflight as reverify_g46_preflight,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (  # noqa: E402
    CONFIG_SCHEMA,
    EVIDENCE_AXIS,
    POSE_DIM,
    PREFLIGHT_SCHEMA,
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_CAMERA_HW,
    PRODUCTION_CLASS_COUNT,
    PRODUCTION_PAIR_COUNT,
    PRODUCTION_SEG_HW,
    ScoredSourceBatchV1,
    V9TrainingTargetCapsuleError,
    V9TrainingTargetCapsuleLoaderV1,
    file_identity,
    materialize_v9_training_target_capsule,
    payload_sha256,
    reverify_preflight,
    seal_preflight,
    sha256_array_bytes,
    sha256_file,
    storage_preflight,
    write_immutable_json,
)

PACKAGE_DISTRIBUTIONS: Final = (
    "av",
    "einops",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
    "torch",
)
CONFIG_KEYS: Final = {
    "schema",
    "run_id",
    "output_root",
    "g46_compile_ready_receipt",
    "posenet_weights",
    "required_free_bytes",
    "truth",
}
IDENTITY_KEYS: Final = {"path", "expected_sha256"}
TRUTH: Final = {
    "research_only": True,
    "encoder_only": True,
    "score_claim": False,
    "candidate_claim": False,
    "promotion_eligible": False,
    "dense_targets_candidate_payload_allowed": False,
    "scorer_weights_candidate_payload_allowed": False,
}


def _require_identity(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != IDENTITY_KEYS:
        raise V9TrainingTargetCapsuleError(f"{label} identity keys differ")
    path = value.get("path")
    expected = value.get("expected_sha256")
    if not isinstance(path, str) or not path or path.strip() != path:
        raise V9TrainingTargetCapsuleError(f"{label}.path is invalid")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise V9TrainingTargetCapsuleError(f"{label}.expected_sha256 is invalid")
    return {"path": path, "expected_sha256": expected}


def _open_identity(value: Any, label: str) -> Path:
    identity = _require_identity(value, label)
    candidate = Path(identity["path"]).expanduser()
    if candidate.is_symlink():
        raise V9TrainingTargetCapsuleError(f"{label} path is a symlink")
    path = candidate.resolve()
    if not path.is_file() or sha256_file(path) != identity["expected_sha256"]:
        raise V9TrainingTargetCapsuleError(f"{label} path/type/SHA-256 differs")
    return path


def load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    config = load_json_mapping(resolved)
    if set(config) != CONFIG_KEYS or config.get("schema") != CONFIG_SCHEMA:
        raise V9TrainingTargetCapsuleError("typed config keys/schema differ")
    if config.get("truth") != TRUTH:
        raise V9TrainingTargetCapsuleError("typed config truth boundary differs")
    if (
        not isinstance(config.get("run_id"), str)
        or len(config["run_id"]) < 12
        or not isinstance(config.get("output_root"), str)
        or not config["output_root"]
    ):
        raise V9TrainingTargetCapsuleError("typed config run/output identifiers differ")
    required = config.get("required_free_bytes")
    if isinstance(required, bool) or not isinstance(required, int) or not 1 <= required <= 1 << 60:
        raise V9TrainingTargetCapsuleError("required_free_bytes is invalid")
    _require_identity(config["g46_compile_ready_receipt"], "G46 receipt")
    _require_identity(config["posenet_weights"], "PoseNet weights")
    return resolved, config


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise V9TrainingTargetCapsuleError(f"required package distribution is unavailable: {distribution}") from exc
    return versions


def _input_row(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, **file_identity(path)}


def _merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    roles: dict[str, set[str]] = {}
    for row in rows:
        path = str(row["path"])
        identity = {key: row[key] for key in ("path", "bytes", "sha256")}
        if path in identities and identities[path] != identity:
            raise V9TrainingTargetCapsuleError(f"sealed input identity collision: {path}")
        identities[path] = identity
        roles.setdefault(path, set()).add(str(row["role"]))
    return [{"role": "+".join(sorted(roles[path])), **identities[path]} for path in sorted(identities)]


def build_preflight(config_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_config, config = load_config(config_path)
    output_root = Path(config["output_root"]).expanduser().resolve()
    preflight_path = output_root / "00_preflight_receipt.json"
    if preflight_path.exists():
        existing = load_json_mapping(preflight_path)
        reverify_preflight(existing)
        if (
            existing.get("config", {}).get("path") != str(resolved_config)
            or existing.get("config", {}).get("sha256") != sha256_file(resolved_config)
            or existing.get("output_root") != str(output_root)
        ):
            raise V9TrainingTargetCapsuleError("existing immutable preflight names another config/run")
        return preflight_path, existing

    g46_receipt_path = _open_identity(
        config["g46_compile_ready_receipt"],
        "G46 receipt",
    )
    posenet_path = _open_identity(config["posenet_weights"], "PoseNet weights")
    g46 = load_compile_ready_materialization_receipt(g46_receipt_path)
    target_path = Path(g46["target_labels"]["path"]).resolve()
    g46_root = target_path.parent.parent
    g46_preflight_path = g46_root / "00_custody_storage_preflight.json"
    g46_preflight = load_json_mapping(g46_preflight_path)
    reverify_g46_preflight(g46_preflight)
    if (
        g46["preflight_sha256"] != g46_preflight["preflight_sha256"]
        or g46["pair_count"] != PRODUCTION_PAIR_COUNT
        or g46["scorer_pair_batch_size"] != PRODUCTION_BATCH_PAIRS
        or g46["target_labels"]["shape"] != [PRODUCTION_PAIR_COUNT, *PRODUCTION_SEG_HW]
        or g46["target_labels"]["dtype"] != "uint8"
    ):
        raise V9TrainingTargetCapsuleError("G46 production geometry/custody differs")
    upstream_root = Path(g46["upstream_closure"]["root"]).resolve()
    expected_posenet = (upstream_root / "models/posenet.safetensors").resolve()
    if posenet_path != expected_posenet:
        raise V9TrainingTargetCapsuleError("PoseNet weights are not the exact model under G46's upstream root")
    source = Path(g46["source_video"]["path"]).resolve()
    segnet = Path(g46["segnet_weights"]["path"]).resolve()
    runtime_files = [
        (
            "g109_capsule_core",
            REPO_ROOT / "src/tac/witness_control/taskspace_v9_training_target_capsule_v1.py",
        ),
        ("g109_capsule_cli", Path(__file__).resolve()),
        (
            "g46_strict_loader",
            REPO_ROOT / "src/tac/witness_control/taskspace_fresh_teacher_materializer_v1.py",
        ),
    ]
    rows = [
        _input_row("typed_config", resolved_config),
        _input_row("g46_compile_ready_receipt", g46_receipt_path),
        _input_row("g46_preflight", g46_preflight_path),
        _input_row("g46_target_labels", target_path),
        _input_row("source_video", source),
        _input_row("segnet_weights", segnet),
        _input_row("posenet_weights", posenet_path),
    ]
    for member in g46["upstream_closure"]["members"]:
        rows.append(_input_row(f"upstream_{member['relative_path']}", Path(member["path"])))
    for role, path in runtime_files:
        rows.append(_input_row(role, path))
    sealed_inputs = _merge_rows(rows)
    storage = storage_preflight(
        output_root,
        required_free_bytes=int(config["required_free_bytes"]),
    )
    run_argv = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        str(resolved_config),
        "--materialize",
    ]
    target_binding = {
        **{key: g46["target_labels"][key] for key in ("path", "bytes", "sha256", "shape", "dtype")},
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": config["run_id"],
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "encoder_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "dense_targets_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "output_root": str(output_root),
        "pair_count": PRODUCTION_PAIR_COUNT,
        "batch_pairs": PRODUCTION_BATCH_PAIRS,
        "camera_hw": list(PRODUCTION_CAMERA_HW),
        "seg_hw": list(PRODUCTION_SEG_HW),
        "class_count": PRODUCTION_CLASS_COUNT,
        "pose_dim": POSE_DIM,
        "seed": int(g46_preflight["seed"]),
        "num_threads": int(g46_preflight["num_threads"]),
        "test_only_small_fixture": False,
        "storage_preflight": storage,
        "config": {
            **file_identity(resolved_config),
            "canonical_sha256": payload_sha256(config),
        },
        "g46_custody": {
            "receipt": file_identity(g46_receipt_path),
            "receipt_sha256": g46["receipt_sha256"],
            "preflight": file_identity(g46_preflight_path),
            "preflight_sha256": g46_preflight["preflight_sha256"],
            "target_labels": target_binding,
            "pair_checkpoint_root_sha256": g46["pair_checkpoint_root_sha256"],
            "compile_ready_reopened": True,
            "labels_reused_as_validation_authority": True,
        },
        "source_custody": {
            "source_video": file_identity(source),
            "sequence_length": 2,
            "pairing": "AVVideoDataset_NONOVERLAPPING_CONTIGUOUS_PAIRS",
            "chronological_pair_range": [0, PRODUCTION_PAIR_COUNT],
            "g46_source_identity_equal": file_identity(source) == g46["source_video"],
        },
        "scorer_custody": {
            "model": "upstream.modules.DistortionNet",
            "segnet_model": "upstream.modules.SegNet",
            "posenet_model": "upstream.modules.PoseNet",
            "segnet_weights": file_identity(segnet),
            "posenet_weights": file_identity(posenet_path),
            "upstream_closure": g46["upstream_closure"],
            "device": "cpu",
            "batch_pairs": PRODUCTION_BATCH_PAIRS,
            "final_partial_batch_pairs": 8,
            "segnet_frame_selector": "last_frame_index_1",
            "pose_output_selector": "pose_head_first_6",
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
            "package_versions": _package_versions(),
        },
        "runtime_custody": {
            "files": [{"role": role, **file_identity(path)} for role, path in runtime_files],
            "python": sys.version.split()[0],
            "upstream_closure_sha256": g46["upstream_closure"]["closure_sha256"],
        },
        "sealed_input_files": sealed_inputs,
        "run_argv": run_argv,
        "resume_contract": {
            "batch_count": 38,
            "batch_atomic": True,
            "checkpoint_committed_after_all_batch_arrays": True,
            "completed_batches_skip_scorer_forward": True,
            "completed_batch_source_bytes_rehashed": True,
            "orphan_batch_arrays_recomputed_and_byte_compared": True,
            "aggregate_rebuilt_from_verified_batches": True,
            "strict_loader_reopens_every_batch_raw_and_npz_byte": True,
        },
        "cleanup_contract": {
            "scratch_root": str((output_root / ".scratch").resolve()),
            "success_scratch_auto_deleted": True,
            "crash_scratch_certified_before_delete": True,
            "durable_delete_allowed": False,
            "cold_store_before_durable_cleanup": True,
        },
    }
    preflight = seal_preflight(body)
    output_root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(preflight_path, preflight)
    reopened = load_json_mapping(preflight_path)
    reverify_preflight(reopened)
    if reopened != preflight:
        raise V9TrainingTargetCapsuleError("preflight changed across parse-back")
    return preflight_path, preflight


def _configure_determinism(*, seed: int, num_threads: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


class _ProductionSourceAndScorer:
    def __init__(self, preflight: Mapping[str, Any]) -> None:
        import torch

        upstream_root = Path(preflight["scorer_custody"]["upstream_closure"]["root"])
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        from frame_utils import AVVideoDataset
        from modules import DistortionNet

        source = Path(preflight["source_custody"]["source_video"]["path"])
        self.source_path = source.resolve()
        self.dataset = AVVideoDataset(
            [source.name],
            data_dir=source.parent,
            batch_size=PRODUCTION_BATCH_PAIRS,
            device=torch.device("cpu"),
            num_threads=int(preflight["num_threads"]),
            seed=int(preflight["seed"]),
            prefetch_queue_depth=1,
        )
        self.dataset.prepare_data()
        self.model = DistortionNet().eval().to(device=torch.device("cpu"))
        self.model.load_state_dicts(
            Path(preflight["scorer_custody"]["posenet_weights"]["path"]),
            Path(preflight["scorer_custody"]["segnet_weights"]["path"]),
            torch.device("cpu"),
        )
        self._next_batch_index = 1

    def source_batches(self) -> Iterator[np.ndarray]:
        for path, batch_index, batch in self.dataset:
            if Path(path).resolve() != self.source_path or int(batch_index) != self._next_batch_index:
                raise V9TrainingTargetCapsuleError(
                    f"AVVideoDataset source/index differs at batch {self._next_batch_index}"
                )
            self._next_batch_index += 1
            yield np.ascontiguousarray(batch.cpu().numpy(), dtype=np.uint8)

    def score(self, source_batch: np.ndarray) -> ScoredSourceBatchV1:
        import torch

        source = torch.from_numpy(np.ascontiguousarray(source_batch))
        with torch.inference_mode():
            posenet_input, segnet_input = self.model.preprocess_input(source)
            posenet_output = self.model.posenet(posenet_input)["pose"][..., :POSE_DIM]
            segnet_output = self.model.segnet(segnet_input)
        return ScoredSourceBatchV1(
            seg_logits_f32=np.ascontiguousarray(
                segnet_output.to(torch.float32).cpu().numpy(),
                dtype=np.float32,
            ),
            source_pose6_f32=np.ascontiguousarray(
                posenet_output.to(torch.float32).cpu().numpy(),
                dtype=np.float32,
            ),
            segnet_input_sha256=sha256_array_bytes(segnet_input.to(torch.float32).cpu().numpy()),
            posenet_input_sha256=sha256_array_bytes(posenet_input.to(torch.float32).cpu().numpy()),
        )


def run_materialization(config_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_config, config = load_config(config_path)
    preflight_path = Path(config["output_root"]) / "00_preflight_receipt.json"
    if not preflight_path.is_file():
        raise V9TrainingTargetCapsuleError("materialization requires an existing strict preflight receipt")
    preflight = load_json_mapping(preflight_path)
    reverify_preflight(preflight)
    if preflight["config"]["sha256"] != sha256_file(resolved_config):
        raise V9TrainingTargetCapsuleError("typed config changed after preflight")
    _configure_determinism(
        seed=int(preflight["seed"]),
        num_threads=int(preflight["num_threads"]),
    )
    production = _ProductionSourceAndScorer(preflight)
    return materialize_v9_training_target_capsule(
        preflight=preflight,
        source_batches=production.source_batches(),
        score_source_batch=production.score,
    )


def run_status(config_path: Path) -> tuple[Path, dict[str, Any]]:
    _resolved, config = load_config(config_path)
    receipt_path = Path(config["output_root"]) / "21_v9_training_target_capsule_receipt.json"
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        receipt_path,
        expected_sha256=sha256_file(receipt_path),
    )
    return receipt_path, loader.receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight_only:
            path, receipt = build_preflight(args.config)
            kind = "preflight"
            self_hash = receipt["preflight_sha256"]
        elif args.materialize:
            assert_governed_admission(
                "taskspace_v9_training_target_capsule_n600",
                on_refuse="raise",
            )
            path, receipt = run_materialization(args.config)
            kind = "aggregate"
            self_hash = receipt["aggregate_receipt_sha256"]
        else:
            path, receipt = run_status(args.config)
            kind = "aggregate_status"
            self_hash = receipt["aggregate_receipt_sha256"]
    except (
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
        V9TrainingTargetCapsuleError,
    ) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": kind,
                "receipt": file_identity(path),
                "sealed_self_sha256": self_hash,
                "score_claim": False,
                "candidate_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
