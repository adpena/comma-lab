# SPDX-License-Identifier: MIT
"""Physical production authority for one parsed G105/G110 semantic stage.

G117 is intentionally an injection-based engine.  This wrapper supplies its
inputs only after recursively reopening the physical G112/G111 and G109
custody chains, constructing the frozen CPU SegNet itself, and sealing the
shipped G110 public plugin tree.  It then independently replays the selected
archive through that public plugin tree and compares its ordered n600
population with the repository receiver.

The expensive scorer cache and the public-wire measurement identity exclude
the competitive frontier pointer.  A fresh dynamic-frontier snapshot is bound
only to the resulting observation and is reverified before publication.
Nothing emitted here is a contest score, candidate, promotion, or pointer
mutation.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np

from tac.scorer import load_default_segnet
from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
    G112PartitionReceiptV2,
    open_g112_partition_receipt,
)
from tac.witness_control.taskspace_v9_training_target_binding_v1 import (
    CHECKPOINT_PROJECTION_KEY,
    CHECKPOINT_PROJECTION_SHA_KEY,
    reopen_v9_training_target_projection,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    PRODUCTION_SEG_HW,
    V9TrainingTargetCapsuleLoaderV1,
    canonical_json_bytes,
)
from tac.witness_dsl import (
    taskspace_g105_exact_v9_semantic_root_adapter_v1 as g105_adapter,
)
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.g111_parsed_g105_stage_selector_v1 import (
    PAIR_COUNT_N600,
    VERDICT_BATCH_SIZES,
    G111ParsedG105StageSelectionV1,
    compile_select_parsed_g105_stage_v1,
    semantic_stage_action,
    semantic_stage_conditional_observation,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    Y1WireCodecV1,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    PACKET_MEMBER,
    PUBLIC_RUNTIME_RELATIVE_ROOT,
    G110OuterZipMethodV1,
    parse_g110_public_archive,
)

SCHEMA: Final = "tac.g120_parsed_stage_production_authority.v1"
PUBLIC_MEASUREMENT_SCHEMA: Final = "tac.g120_public_plugin_measurement.v1"
PUBLIC_BATCH_SCHEMA: Final = "tac.g120_public_plugin_batch.v1"
CROSS_STAGE_ROW_SCHEMA: Final = "tac.g120_cross_stage_pareto_row.v1"
CROSS_STAGE_LEDGER_SCHEMA: Final = "tac.g120_cross_stage_pareto.v1"
BEST_POINTER_SCHEMA: Final = "tac.g120_cross_stage_best_pointer.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU exact parsed-public-wire producer screen]"
PUBLIC_RUNTIME_EXPECTED_FILES: Final = (
    "frame0_variants/conditional_lowrank_rice_v1.py",
    "frame0_variants/generated_y1_pose_xip2_v1.py",
    "inflate.py",
    "inflate.sh",
    "semantic_variants/original_coordinr_film_mlp_v1.py",
    "semantic_variants/v9_hosc_dual_head_odd_y1_v1.py",
)
PARETO_FILENAME: Final = "g105_public_wire_pareto.json"
BEST_FILENAME: Final = "g105_public_wire_best.json"
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_STAGE_TAG: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_EPHEMERAL_ROOTS: Final = (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"))
_SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

SegArgmaxBatchScorerV1 = Callable[[np.ndarray], np.ndarray]


class G120ProductionAuthorityError(RuntimeError):
    """A physical custody, public receiver, cache, or pointer gate failed."""


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G120ProductionAuthorityError("G120 value is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _LOWER_SHA256 for character in value):
        raise G120ProductionAuthorityError(f"{name} must be a lowercase SHA-256")
    return value


def _regular_file_identity(path: Path, *, name: str) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise G120ProductionAuthorityError(f"{name} must be an absolute regular non-symlink file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != before.st_size:
        raise G120ProductionAuthorityError(f"{name} changed during physical reopen")
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _durable_dir(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise G120ProductionAuthorityError(f"{name} must be an absolute pathlib.Path")
    resolved = path.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in _EPHEMERAL_ROOTS):
        raise G120ProductionAuthorityError(f"{name} must not be temporary")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise G120ProductionAuthorityError(f"{name} must be a real directory")
    return path


def _ssd_cache_dir(path: Path) -> Path:
    result = _durable_dir(path, name="measurement_cache_dir")
    resolved = result.resolve()
    if not any(resolved != root.resolve() and resolved.is_relative_to(root.resolve()) for root in _SSD_ROOTS):
        raise G120ProductionAuthorityError("production measurement cache must be below a configured SSD pact root")
    return result


def _atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    if type(payload) is not bytes or not payload:
        raise G120ProductionAuthorityError("atomic payload must be nonempty exact bytes")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise G120ProductionAuthorityError(f"refusing non-regular output path: {path.name}")
        if path.read_bytes() == payload:
            return
        if not replace:
            raise G120ProductionAuthorityError(f"refusing to overwrite different output: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                if path.is_symlink() or path.read_bytes() != payload:
                    raise G120ProductionAuthorityError(f"concurrent output differs: {path.name}") from exc
            temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _snapshot_payload(snapshot: DynamicFrontierTargetSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, DynamicFrontierTargetSnapshot):
        raise G120ProductionAuthorityError("frontier loader returned the wrong snapshot type")
    return dataclasses.asdict(snapshot)


def dynamic_snapshot_identity_sha256(snapshot: DynamicFrontierTargetSnapshot) -> str:
    """Bind the pointer object and its recomputed competitive observation."""

    return _sha256(_canonical_json(_snapshot_payload(snapshot)))


def _reopen_upstream_closure(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"root", "members", "closure_sha256"}:
        raise G120ProductionAuthorityError("G109 upstream closure has a noncanonical field set")
    root = Path(str(value["root"]))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise G120ProductionAuthorityError("G109 upstream root is not a physical absolute directory")
    expected_members = ("evaluate.py", "frame_utils.py", "modules.py", "public_test_video_names.txt")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for relative in expected_members:
        identity = _regular_file_identity(root / relative, name=f"upstream {relative}")
        row = {"relative_path": relative, **identity}
        rows.append(row)
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    reopened = {
        "root": str(root.resolve()),
        "members": rows,
        "closure_sha256": digest.hexdigest(),
    }
    if reopened != value:
        raise G120ProductionAuthorityError("G109 upstream source closure changed")
    return reopened


def public_plugin_tree_identity(repo_root: Path) -> dict[str, Any]:
    """Physically seal the exact shipped G110 runtime tree, excluding caches."""

    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise G120ProductionAuthorityError("repo_root must be an absolute pathlib.Path")
    root = repo_root / PUBLIC_RUNTIME_RELATIVE_ROOT
    if root.is_symlink() or not root.is_dir():
        raise G120ProductionAuthorityError("shipped G110 public plugin root is absent or a symlink")
    observed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise G120ProductionAuthorityError("shipped G110 public plugin tree contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        observed.append(relative.as_posix())
    if tuple(observed) != PUBLIC_RUNTIME_EXPECTED_FILES:
        raise G120ProductionAuthorityError("shipped G110 public plugin file census differs")
    rows = []
    for relative in observed:
        identity = _regular_file_identity(root / relative, name=f"public plugin {relative}")
        rows.append({"relative_path": relative, **identity})
    content_rows = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return {
        "root": str(root.resolve()),
        "files": rows,
        "tree_sha256": _sha256(_canonical_json(content_rows)),
        "physical_tree_identity_sha256": _sha256(_canonical_json(rows)),
    }


def _load_module(path: Path, *, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise G120ProductionAuthorityError(f"cannot load shipped public module {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = prior
    return module


class _PointerIndependentPredictionCache:
    """Cache SegNet argmax arrays by exact camera bytes and scorer identity."""

    def __init__(self, *, model: object, scorer_identity_sha256: str, cache_dir: Path) -> None:
        self.model = model
        self.scorer_identity_sha256 = _require_sha256(
            scorer_identity_sha256,
            name="Seg scorer identity",
        )
        self.cache_dir = cache_dir

    def _predict(self, camera_y1: np.ndarray) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(camera_y1).permute(0, 3, 1, 2).contiguous().float()
        pair = torch.stack((tensor, tensor), dim=1)
        with torch.inference_mode():
            scorer_input = self.model.preprocess_input(pair)  # type: ignore[attr-defined]
            logits = self.model(scorer_input)  # type: ignore[operator]
            labels = logits.argmax(dim=1).to(dtype=torch.uint8).cpu().numpy()
        return np.ascontiguousarray(labels, dtype=np.uint8)

    def __call__(self, camera_y1: np.ndarray) -> np.ndarray:
        raw = np.asarray(camera_y1)
        if (
            type(raw) is not np.ndarray
            or raw.dtype != np.uint8
            or raw.ndim != 4
            or raw.shape[1:] != (874, 1164, 3)
            or not raw.flags.c_contiguous
            or not 1 <= raw.shape[0] <= PRODUCTION_BATCH_PAIRS
        ):
            raise G120ProductionAuthorityError("SegNet camera batch is not exact contiguous uint8 BHWC")
        key_body = {
            "schema": "tac.g120_pointer_independent_seg_prediction_cache.v1",
            "seg_scorer_identity_sha256": self.scorer_identity_sha256,
            "camera_batch_shape": list(raw.shape),
            "camera_batch_sha256": _sha256(memoryview(raw)),
        }
        key = _sha256(_canonical_json(key_body))
        path = self.cache_dir / f"{key}.predicted_labels.npy"
        receipt_path = self.cache_dir / f"{key}.receipt.json"
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or receipt_path.is_symlink() or not receipt_path.is_file():
                raise G120ProductionAuthorityError("prediction cache data/receipt pair is incomplete or non-regular")
            receipt_raw = receipt_path.read_bytes()
            try:
                receipt = json.loads(receipt_raw)
                with path.open("rb") as stream:
                    predicted = np.load(stream, allow_pickle=False)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise G120ProductionAuthorityError("prediction cache entry cannot be reopened") from exc
            receipt_body = {item: value for item, value in receipt.items() if item != "receipt_body_sha256"}
            if (
                type(receipt) is not dict
                or _canonical_json(receipt) != receipt_raw
                or receipt.get("schema") != key_body["schema"]
                or receipt.get("cache_key_sha256") != key
                or receipt.get("identity") != key_body
                or receipt.get("prediction_file") != _regular_file_identity(path, name="prediction cache data")
                or receipt.get("receipt_body_sha256") != _sha256(_canonical_json(receipt_body))
            ):
                raise G120ProductionAuthorityError("prediction cache receipt identity differs")
        else:
            if receipt_path.exists() or receipt_path.is_symlink():
                raise G120ProductionAuthorityError("prediction cache receipt exists without data")
            predicted = self._predict(raw)
            handle = io.BytesIO()
            np.save(handle, predicted, allow_pickle=False)
            payload = handle.getvalue()
            _atomic_write(path, payload)
            receipt_body = {
                "schema": key_body["schema"],
                "cache_key_sha256": key,
                "identity": key_body,
                "prediction_file": _regular_file_identity(
                    path,
                    name="prediction cache data",
                ),
                "predicted_labels_sha256": _sha256(memoryview(predicted)),
            }
            receipt = {
                **receipt_body,
                "receipt_body_sha256": _sha256(_canonical_json(receipt_body)),
            }
            _atomic_write(receipt_path, _canonical_json(receipt))
        expected = (raw.shape[0], *PRODUCTION_SEG_HW)
        if (
            type(predicted) is not np.ndarray
            or predicted.dtype != np.uint8
            or predicted.shape != expected
            or not predicted.flags.c_contiguous
            or np.any(predicted >= 5)
        ):
            raise G120ProductionAuthorityError("prediction cache returned malformed SegNet argmax labels")
        return predicted


@dataclass(frozen=True, slots=True)
class _OpenedProductionAuthorityV1:
    g112: G112PartitionReceiptV2
    target_projection: dict[str, Any]
    target_labels: np.ndarray = field(repr=False)
    config: object = field(repr=False)
    scorer: SegArgmaxBatchScorerV1 = field(repr=False)
    seg_scorer_identity_sha256: str
    public_runtime_tree: dict[str, Any]
    g109_custody: dict[str, Any]
    stage_tag: str
    physical_stage_identity: dict[str, Any]
    physical_stage_identity_sha256: str


def _physical_stage_identity(g112: G112PartitionReceiptV2) -> tuple[dict[str, Any], str, str]:
    node = g112.source_chain.current
    pair = node.pair
    stage_tag = f"{pair.stage}.epoch_{pair.epoch}.chk_{pair.checkpoint_id_sha256[:12]}"
    identity = {
        "g112_partition_receipt": {
            "path": str(g112.receipt_path),
            "bytes": g112.receipt_bytes,
            "sha256": g112.receipt_sha256,
        },
        "g112_semantic_child": {
            "path": str(g112.semantic_child.checkpoint_path),
            "bytes": g112.semantic_child.checkpoint_bytes,
            "sha256": g112.semantic_child.checkpoint_sha256,
            "semantic_packet_sha256": g112.semantic_packet_sha256,
        },
        "g112_pose_initializer": {
            "path": str(g112.initializer.checkpoint_path),
            "bytes": g112.initializer.checkpoint_bytes,
            "sha256": g112.initializer.checkpoint_sha256,
            "target_projection_sha256": g112.initializer.target_projection_sha256,
        },
        "g111_deploy_checkpoint": {
            "path": str(pair.deploy.path),
            "bytes": pair.deploy.bytes,
            "sha256": pair.deploy.sha256,
        },
        "g111_full_state_resume_checkpoint": {
            "path": str(pair.resume.path),
            "bytes": pair.resume.bytes,
            "sha256": pair.resume.sha256,
        },
        "g111_fresh_lineage_receipt": {
            "path": str(node.receipt_path),
            "bytes": node.receipt_bytes,
            "sha256": node.receipt_sha256,
        },
        "g111_checkpoint_id_sha256": pair.checkpoint_id_sha256,
        "g111_stage": pair.stage,
        "g111_epoch": pair.epoch,
        "fresh_lineage_complete": g112.source_chain.complete_trajectory_proven,
    }
    return identity, _sha256(_canonical_json(identity)), stage_tag


def _open_production_authority(
    *,
    repo_root: Path,
    g112_partition_receipt: Path,
    expected_g112_partition_receipt_sha256: str,
    measurement_cache_dir: Path,
) -> _OpenedProductionAuthorityV1:
    expected_g112_sha = _require_sha256(
        expected_g112_partition_receipt_sha256,
        name="expected G112 partition receipt",
    )
    g112 = open_g112_partition_receipt(
        g112_partition_receipt,
        expected_sha256=expected_g112_sha,
    )
    scalars = g112.semantic_child.g105_scalars
    projection_json = scalars.get(CHECKPOINT_PROJECTION_KEY)
    projection_sha = scalars.get(CHECKPOINT_PROJECTION_SHA_KEY)
    if not isinstance(projection_json, str) or not isinstance(projection_sha, str):
        raise G120ProductionAuthorityError("G112 semantic child lacks physical G109 projection")
    projection = reopen_v9_training_target_projection(
        projection_json=projection_json,
        expected_projection_sha256=projection_sha,
    )
    pair = g112.source_chain.current.pair
    if projection_sha != g112.initializer.target_projection_sha256 or projection_sha != pair.target_projection_sha256:
        raise G120ProductionAuthorityError("G112 semantic/initializer/G111 target projections differ")
    aggregate = projection.get("aggregate_receipt")
    if not isinstance(aggregate, dict):
        raise G120ProductionAuthorityError("G109 projection lacks aggregate receipt")
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        Path(str(aggregate.get("path"))),
        expected_sha256=str(aggregate.get("sha256")),
    )
    if (
        loader.preflight.get("test_only_small_fixture") is not False
        or loader.pair_count != PRODUCTION_PAIR_COUNT
        or loader.batch_pairs != PRODUCTION_BATCH_PAIRS
        or loader.seg_hw != PRODUCTION_SEG_HW
    ):
        raise G120ProductionAuthorityError("G109 capsule is not production n600/batch16 geometry")
    target_labels = np.ascontiguousarray(loader.targets.seg_labels_u8, dtype=np.uint8)
    label_binding = projection.get("arrays", {}).get("seg_labels_u8")
    if (
        target_labels.shape != (PRODUCTION_PAIR_COUNT, *PRODUCTION_SEG_HW)
        or np.any(target_labels >= 5)
        or not isinstance(label_binding, dict)
        or _sha256(memoryview(target_labels)) != label_binding.get("sha256")
    ):
        raise G120ProductionAuthorityError("physical G109 labels differ from checkpoint projection")
    scorer_custody = loader.preflight.get("scorer_custody")
    if not isinstance(scorer_custody, dict):
        raise G120ProductionAuthorityError("G109 scorer custody is absent")
    closure = _reopen_upstream_closure(scorer_custody.get("upstream_closure"))
    upstream_root = Path(closure["root"])
    if upstream_root != (repo_root / "upstream").resolve():
        raise G120ProductionAuthorityError("G109 upstream root differs from the selected repository")
    seg_binding = scorer_custody.get("segnet_weights")
    if not isinstance(seg_binding, dict):
        raise G120ProductionAuthorityError("G109 SegNet weights binding is absent")
    expected_seg_path = upstream_root / "models" / "segnet.safetensors"
    if _regular_file_identity(expected_seg_path, name="SegNet weights") != seg_binding:
        raise G120ProductionAuthorityError("G109 SegNet weights differ from the exact upstream model")
    if scorer_custody.get("segnet_model") != "upstream.modules.SegNet":
        raise G120ProductionAuthorityError("G109 scorer model is not upstream.modules.SegNet")
    scorer_identity_payload = {
        "model": "upstream.modules.SegNet",
        "device": "cpu",
        "weights": seg_binding,
        "upstream_closure": closure,
        "package_versions": scorer_custody.get("package_versions"),
        "runtime_custody": loader.preflight.get("runtime_custody"),
    }
    scorer_identity = _sha256(_canonical_json(scorer_identity_payload))
    model = load_default_segnet(upstream_root, device="cpu")
    try:
        import torch

        model_module = sys.modules.get(model.__class__.__module__)
        if (
            model.training
            or any(parameter.requires_grad for parameter in model.parameters())
            or any(parameter.device != torch.device("cpu") for parameter in model.parameters())
            or model_module is None
            or Path(str(getattr(model_module, "__file__", ""))).resolve() != (upstream_root / "modules.py").resolve()
        ):
            raise G120ProductionAuthorityError(
                "constructed SegNet is not frozen eval-mode CPU from the sealed upstream modules.py"
            )
    except AttributeError as exc:
        raise G120ProductionAuthorityError("constructed SegNet lacks the frozen model contract") from exc
    scorer = _PointerIndependentPredictionCache(
        model=model,
        scorer_identity_sha256=scorer_identity,
        cache_dir=measurement_cache_dir,
    )
    config = g105_adapter._checkpoint_config(
        {**g112.semantic_child.shared_params, "code": g112.semantic_child.code_y1},
        scalars,
    )
    runtime_tree = public_plugin_tree_identity(repo_root)
    stage_identity, stage_identity_sha, stage_tag = _physical_stage_identity(g112)
    return _OpenedProductionAuthorityV1(
        g112=g112,
        target_projection=projection,
        target_labels=target_labels,
        config=config,
        scorer=scorer,
        seg_scorer_identity_sha256=scorer_identity,
        public_runtime_tree=runtime_tree,
        g109_custody={
            "projection_sha256": projection_sha,
            "aggregate_receipt": aggregate,
            "target_labels_sha256": label_binding["sha256"],
            "source_video": loader.receipt["source_custody"]["source_video"],
            "segnet_weights": seg_binding,
            "upstream_closure": closure,
            "scorer_runtime_identity_sha256": scorer_identity,
        },
        stage_tag=stage_tag,
        physical_stage_identity=stage_identity,
        physical_stage_identity_sha256=stage_identity_sha,
    )


def _batch_chain(rows: Sequence[Mapping[str, Any]], field_name: str) -> str:
    return _sha256(
        _canonical_json(
            [
                {
                    "batch_index": row["batch_index"],
                    field_name: row[field_name],
                }
                for row in rows
            ]
        )
    )


def _load_public_batch(
    path: Path,
    *,
    identity_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    target_batch_sha256: str,
    scorer_y1_batch_sha256: str,
    camera_y1_batch_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise G120ProductionAuthorityError("public measurement batch receipt is not a regular file")
    raw = path.read_bytes()
    try:
        row = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise G120ProductionAuthorityError("public measurement batch receipt is corrupt") from exc
    if type(row) is not dict or _canonical_json(row) != raw:
        raise G120ProductionAuthorityError("public measurement batch receipt is not canonical")
    body = {key: value for key, value in row.items() if key != "row_body_sha256"}
    if (
        row.get("schema") != PUBLIC_BATCH_SCHEMA
        or row.get("measurement_identity_sha256") != identity_sha256
        or row.get("batch_index") != batch_index
        or row.get("pair_start") != pair_start
        or row.get("pair_stop") != pair_stop
        or row.get("target_batch_sha256") != target_batch_sha256
        or row.get("scorer_y1_batch_sha256") != scorer_y1_batch_sha256
        or row.get("camera_y1_batch_sha256") != camera_y1_batch_sha256
        or row.get("row_body_sha256") != _sha256(_canonical_json(body))
    ):
        raise G120ProductionAuthorityError("public measurement batch identity differs")
    disagreements = row.get("disagreement_pixels")
    maximum = (pair_stop - pair_start) * PRODUCTION_SEG_HW[0] * PRODUCTION_SEG_HW[1]
    if type(disagreements) is not int or not 0 <= disagreements <= maximum:
        raise G120ProductionAuthorityError("public measurement batch disagreement count is invalid")
    _require_sha256(row.get("predicted_labels_batch_sha256"), name="public predicted-label batch")
    return row


@dataclass(frozen=True, slots=True)
class G120PublicPluginMeasurementV1:
    d_seg: float
    disagreement_pixels: int
    measurement_identity_sha256: str
    public_runtime_tree_sha256: str
    repository_scorer_y1_population_sha256: str
    public_scorer_y1_population_sha256: str
    repository_camera_y1_population_sha256: str
    public_camera_y1_population_sha256: str
    repository_predicted_labels_sha256: str
    public_predicted_labels_sha256: str
    batch_receipt_chain_sha256: str
    resumed_batch_count: int


def _measure_public_plugin_surface(
    *,
    authority: _OpenedProductionAuthorityV1,
    engine: G111ParsedG105StageSelectionV1,
    progress_dir: Path,
) -> G120PublicPluginMeasurementV1:
    selected = engine.selected
    packet = parse_g110_public_archive(selected.archive)
    runtime_root = Path(authority.public_runtime_tree["root"])
    tree_sha = str(authority.public_runtime_tree["tree_sha256"])
    inflate = _load_module(
        runtime_root / "inflate.py",
        module_name=f"_g120_public_inflate_{tree_sha}",
    )
    with tempfile.TemporaryDirectory(prefix="g120-public-packet-") as temporary:
        archive_root = Path(temporary)
        (archive_root / PACKET_MEMBER).write_bytes(packet)
        public_packet = inflate._load_packet(archive_root)
    if public_packet != packet:
        raise G120ProductionAuthorityError("shipped public packet loader changed the selected packet")
    semantic_plugins = inflate._load_plugins(
        runtime_root / "semantic_variants",
        calls=inflate.SEMANTIC_PLUGIN_CALLS,
        expected=inflate.EXPECTED_SEMANTIC_PLUGINS,
    )
    frame0_plugins = inflate._load_plugins(
        runtime_root / "frame0_variants",
        calls=inflate.FRAME0_PLUGIN_CALLS,
        expected=inflate.EXPECTED_FRAME0_PLUGINS,
    )
    frame0_matches = [module for module in frame0_plugins.values() if module.accepts_packet(packet) is True]
    if len(frame0_matches) != 1:
        raise G120ProductionAuthorityError("selected packet did not match exactly one shipped frame0 plugin")
    frame0 = frame0_matches[0]
    frame0_state = frame0.parse_packet(packet)
    semantic_packet = frame0.semantic_packet(frame0_state)
    semantic_matches = [
        module for module in semantic_plugins.values() if module.accepts_packet(semantic_packet) is True
    ]
    if len(semantic_matches) != 1:
        raise G120ProductionAuthorityError("selected packet did not match exactly one shipped semantic plugin")
    semantic = semantic_matches[0]
    parsed = semantic.parse_packet(semantic_packet)
    measurement_identity = {
        "schema": PUBLIC_MEASUREMENT_SCHEMA,
        "public_runtime_tree_sha256": tree_sha,
        "selected_archive_sha256": _sha256(selected.archive),
        "selected_archive_bytes": len(selected.archive),
        "selected_packet_sha256": _sha256(packet),
        "target_labels_sha256": _sha256(memoryview(authority.target_labels)),
        "seg_scorer_identity_sha256": authority.seg_scorer_identity_sha256,
        "physical_stage_identity_sha256": authority.physical_stage_identity_sha256,
        "g112_pose_initializer_sha256": authority.g112.initializer.checkpoint_sha256,
        "frontier_pointer_intentionally_excluded": True,
    }
    measurement_identity_sha = _sha256(_canonical_json(measurement_identity))
    measurement_root = _durable_dir(
        progress_dir / "g120_public_plugin" / measurement_identity_sha,
        name="public measurement progress directory",
    )
    final_y1_population = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    resumed = 0
    pair_start = 0
    for batch_index, batch_size in enumerate(VERDICT_BATCH_SIZES):
        pair_stop = pair_start + batch_size
        scorer_digest = hashlib.sha256()
        camera_digest = hashlib.sha256()
        camera_frames: list[np.ndarray] = []
        for pair_id in range(pair_start, pair_stop):
            scorer_y1 = semantic.render_scorer_y1(parsed, pair_id)
            if (
                type(scorer_y1) is not np.ndarray
                or scorer_y1.dtype != np.uint8
                or scorer_y1.shape != (*PRODUCTION_SEG_HW, 3)
            ):
                raise G120ProductionAuthorityError("shipped semantic plugin emitted malformed scorer Y1")
            scorer_y1 = np.ascontiguousarray(scorer_y1)
            final_y1_population.update(struct.pack(">H", pair_id))
            final_y1_population.update(memoryview(scorer_y1))
            camera_y1 = inflate._realize_factor2(scorer_y1)
            camera_y0 = frame0.render_camera_y0(frame0_state, pair_id, scorer_y1, camera_y1)
            if (
                type(camera_y0) is not np.ndarray
                or camera_y0.dtype != np.uint8
                or not camera_y0.flags.c_contiguous
                or not np.array_equal(camera_y0, camera_y1)
            ):
                raise G120ProductionAuthorityError("rank-zero shipped public Y0 is not exact camera Y1")
            scorer_digest.update(memoryview(scorer_y1))
            camera_digest.update(memoryview(camera_y1))
            camera_frames.append(camera_y1)
        scorer_batch_sha = scorer_digest.hexdigest()
        camera_batch_sha = camera_digest.hexdigest()
        target_batch = authority.target_labels[pair_start:pair_stop]
        target_batch_sha = _sha256(memoryview(target_batch))
        row_path = measurement_root / f"batch_{batch_index:03d}_{pair_start:03d}_{pair_stop:03d}.json"
        row = _load_public_batch(
            row_path,
            identity_sha256=measurement_identity_sha,
            batch_index=batch_index,
            pair_start=pair_start,
            pair_stop=pair_stop,
            target_batch_sha256=target_batch_sha,
            scorer_y1_batch_sha256=scorer_batch_sha,
            camera_y1_batch_sha256=camera_batch_sha,
        )
        if row is None:
            camera_batch = np.ascontiguousarray(np.stack(camera_frames), dtype=np.uint8)
            predicted = authority.scorer(camera_batch)
            expected_shape = (batch_size, *PRODUCTION_SEG_HW)
            if (
                type(predicted) is not np.ndarray
                or predicted.dtype != np.uint8
                or predicted.shape != expected_shape
                or not predicted.flags.c_contiguous
                or np.any(predicted >= 5)
            ):
                raise G120ProductionAuthorityError("frozen CPU SegNet returned malformed public labels")
            body = {
                "schema": PUBLIC_BATCH_SCHEMA,
                "measurement_identity_sha256": measurement_identity_sha,
                "batch_index": batch_index,
                "pair_start": pair_start,
                "pair_stop": pair_stop,
                "target_batch_sha256": target_batch_sha,
                "scorer_y1_batch_sha256": scorer_batch_sha,
                "camera_y1_batch_sha256": camera_batch_sha,
                "predicted_labels_batch_sha256": _sha256(memoryview(predicted)),
                "disagreement_pixels": int(np.count_nonzero(predicted != target_batch)),
            }
            row = {**body, "row_body_sha256": _sha256(_canonical_json(body))}
            _atomic_write(row_path, _canonical_json(row))
        else:
            resumed += 1
        rows.append(row)
        pair_start = pair_stop
    frame0.verify_final_y1_population(frame0_state, final_y1_population.digest())
    if pair_start != PAIR_COUNT_N600 or len(rows) != len(VERDICT_BATCH_SIZES):
        raise AssertionError("public plugin measurement lost exact n600 chronology")
    public_scorer_sha = _batch_chain(rows, "scorer_y1_batch_sha256")
    public_camera_sha = _batch_chain(rows, "camera_y1_batch_sha256")
    public_predicted_sha = _batch_chain(rows, "predicted_labels_batch_sha256")
    if (
        public_scorer_sha != selected.scorer_y1_population_sha256
        or public_camera_sha != selected.camera_y1_population_sha256
        or public_predicted_sha != selected.predicted_labels_sha256
    ):
        raise G120ProductionAuthorityError("repository/public ordered population hashes differ")
    disagreements = sum(int(row["disagreement_pixels"]) for row in rows)
    d_seg = disagreements / (PAIR_COUNT_N600 * PRODUCTION_SEG_HW[0] * PRODUCTION_SEG_HW[1])
    if d_seg != selected.d_seg:
        raise G120ProductionAuthorityError("repository/public exact d_seg differs")
    return G120PublicPluginMeasurementV1(
        d_seg=d_seg,
        disagreement_pixels=disagreements,
        measurement_identity_sha256=measurement_identity_sha,
        public_runtime_tree_sha256=tree_sha,
        repository_scorer_y1_population_sha256=selected.scorer_y1_population_sha256,
        public_scorer_y1_population_sha256=public_scorer_sha,
        repository_camera_y1_population_sha256=selected.camera_y1_population_sha256,
        public_camera_y1_population_sha256=public_camera_sha,
        repository_predicted_labels_sha256=selected.predicted_labels_sha256,
        public_predicted_labels_sha256=public_predicted_sha,
        batch_receipt_chain_sha256=_batch_chain(rows, "row_body_sha256"),
        resumed_batch_count=resumed,
    )


def _validate_engine_handoff(
    *,
    authority: _OpenedProductionAuthorityV1,
    engine: G111ParsedG105StageSelectionV1,
    pointer_identity_sha256: str,
) -> None:
    receipt = engine.receipt
    expected_matrix = {
        (codec, method)
        for codec in (
            Y1WireCodecV1.RAW_I16_LE,
            Y1WireCodecV1.DELTA_RICE_BEST_K,
        )
        for method in (
            G110OuterZipMethodV1.STORE,
            G110OuterZipMethodV1.DEFLATE,
        )
    }
    alternatives = getattr(engine, "alternatives", None)
    observed_matrix = (
        {
            (
                getattr(alternative, "y1_wire_codec", None),
                getattr(alternative, "outer_zip_method", None),
            )
            for alternative in alternatives
        }
        if type(alternatives) is tuple
        else set()
    )
    if (
        type(receipt) is not dict
        or receipt.get("engine_only") is not True
        or receipt.get("production_authority_closed") is not False
        or receipt.get("production_wrapper_required") is not True
        or receipt.get("stage_tag") != authority.stage_tag
        or receipt.get("source_checkpoint_identity_sha256") != authority.physical_stage_identity_sha256
        or receipt.get("pose_initializer_identity_sha256") != authority.g112.initializer.checkpoint_sha256
        or receipt.get("target_labels_sha256") != _sha256(memoryview(authority.target_labels))
        or receipt.get("seg_scorer_identity_sha256") != authority.seg_scorer_identity_sha256
        or receipt.get("pointer_snapshot_identity_sha256") != pointer_identity_sha256
        or type(alternatives) is not tuple
        or len(alternatives) != 4
        or observed_matrix != expected_matrix
        or not any(engine.selected is alternative for alternative in alternatives)
    ):
        raise G120ProductionAuthorityError("G117 engine handoff does not match physical G120 custody")
    archive_identity = _regular_file_identity(
        engine.archive_path,
        name="G117 selected archive",
    )
    if archive_identity["bytes"] != len(engine.selected.archive) or archive_identity["sha256"] != _sha256(
        engine.selected.archive
    ):
        raise G120ProductionAuthorityError("G117 selected archive path differs from selected exact bytes")
    _regular_file_identity(engine.receipt_path, name="G117 engine receipt")
    receipt_raw = engine.receipt_path.read_bytes()
    try:
        reopened_receipt = json.loads(receipt_raw)
    except json.JSONDecodeError as exc:
        raise G120ProductionAuthorityError("G117 engine receipt path is corrupt") from exc
    if _canonical_json(reopened_receipt) != receipt_raw or reopened_receipt != engine.receipt:
        raise G120ProductionAuthorityError("G117 engine receipt object differs from its durable canonical path")


@dataclass(frozen=True, slots=True)
class G120CrossStageParetoRowV1:
    """One physical stage retained for the later conditional-pose value function."""

    stage_tag: str
    row_identity_sha256: str
    d_seg_wire: float
    exact_archive_bytes: int
    semantic_action: float
    distortion_only_value: float
    retained_for_post_g105_pose: bool
    source_float_to_wire_regret: dict[str, Any]
    pose_initializer_identity_sha256: str
    physical_stage_identity: dict[str, Any]
    physical_stage_identity_sha256: str
    measurement_identity_sha256: str
    public_runtime_tree_sha256: str
    selected_archive: dict[str, Any]
    selected_archive_sha256: str
    pointer_snapshot_identity_sha256: str
    live_target_score: float

    def __post_init__(self) -> None:
        if type(self.stage_tag) is not str or _STAGE_TAG.fullmatch(self.stage_tag) is None:
            raise G120ProductionAuthorityError("cross-stage stage tag is unsafe")
        for name, value in (
            ("row identity", self.row_identity_sha256),
            ("pose initializer", self.pose_initializer_identity_sha256),
            ("physical stage identity", self.physical_stage_identity_sha256),
            ("measurement identity", self.measurement_identity_sha256),
            ("public runtime tree", self.public_runtime_tree_sha256),
            ("selected archive", self.selected_archive_sha256),
            ("pointer snapshot", self.pointer_snapshot_identity_sha256),
        ):
            _require_sha256(value, name=name)
        if (
            type(self.d_seg_wire) is not float
            or not math.isfinite(self.d_seg_wire)
            or not 0.0 <= self.d_seg_wire <= 1.0
            or type(self.exact_archive_bytes) is not int
            or self.exact_archive_bytes <= 0
            or self.semantic_action
            != semantic_stage_action(
                d_seg=self.d_seg_wire,
                archive_bytes=self.exact_archive_bytes,
            )
            or self.distortion_only_value != 100.0 * self.d_seg_wire
            or type(self.live_target_score) is not float
            or not math.isfinite(self.live_target_score)
            or self.live_target_score <= 0.0
            or self.retained_for_post_g105_pose is not (self.distortion_only_value < self.live_target_score)
        ):
            raise G120ProductionAuthorityError("cross-stage semantic geometry differs")
        regret = self.source_float_to_wire_regret
        if (
            type(regret) is not dict
            or regret.get("status") not in {"measured", "unmeasured"}
            or (
                regret["status"] == "unmeasured"
                and (regret.get("value") is not None or type(regret.get("reason")) is not str or not regret["reason"])
            )
            or (
                regret["status"] == "measured"
                and (
                    type(regret.get("value")) is not float
                    or not math.isfinite(regret["value"])
                    or regret["value"] < 0.0
                )
            )
        ):
            raise G120ProductionAuthorityError("cross-stage wire-regret coordinate differs")
        physical = self.physical_stage_identity
        required_physical = {
            "g112_partition_receipt",
            "g112_semantic_child",
            "g112_pose_initializer",
            "g111_deploy_checkpoint",
            "g111_full_state_resume_checkpoint",
            "g111_fresh_lineage_receipt",
            "g111_checkpoint_id_sha256",
            "g111_stage",
            "g111_epoch",
            "fresh_lineage_complete",
        }
        if (
            type(physical) is not dict
            or set(physical) != required_physical
            or physical.get("fresh_lineage_complete") is not True
            or self.physical_stage_identity_sha256 != _sha256(_canonical_json(physical))
        ):
            raise G120ProductionAuthorityError("cross-stage row lacks exact G112 deploy/resume/lineage custody")
        _require_sha256(
            physical.get("g111_checkpoint_id_sha256"),
            name="G111 checkpoint identity",
        )
        for section in (
            "g112_partition_receipt",
            "g112_semantic_child",
            "g112_pose_initializer",
            "g111_deploy_checkpoint",
            "g111_full_state_resume_checkpoint",
            "g111_fresh_lineage_receipt",
        ):
            binding = physical.get(section)
            if (
                type(binding) is not dict
                or type(binding.get("path")) is not str
                or not Path(binding["path"]).is_absolute()
                or type(binding.get("bytes")) is not int
                or binding["bytes"] <= 0
            ):
                raise G120ProductionAuthorityError(f"cross-stage physical binding differs at {section}")
            _require_sha256(binding.get("sha256"), name=f"{section} SHA-256")
        archive = self.selected_archive
        if (
            type(archive) is not dict
            or set(archive) != {"path", "bytes", "sha256"}
            or type(archive.get("path")) is not str
            or not Path(archive["path"]).is_absolute()
            or archive.get("bytes") != self.exact_archive_bytes
            or archive.get("sha256") != self.selected_archive_sha256
        ):
            raise G120ProductionAuthorityError("cross-stage row lacks its exact selected public archive binding")

    @classmethod
    def build(
        cls,
        *,
        authority: _OpenedProductionAuthorityV1,
        engine: G111ParsedG105StageSelectionV1,
        measurement: G120PublicPluginMeasurementV1,
        pointer_snapshot_identity_sha256: str,
        live_target_score: float,
    ) -> G120CrossStageParetoRowV1:
        selected = engine.selected
        body = {
            "schema": CROSS_STAGE_ROW_SCHEMA,
            "stage_tag": authority.stage_tag,
            "d_seg_wire": measurement.d_seg,
            "exact_archive_bytes": len(selected.archive),
            "semantic_action": semantic_stage_action(
                d_seg=measurement.d_seg,
                archive_bytes=len(selected.archive),
            ),
            "distortion_only_value": 100.0 * measurement.d_seg,
            "retained_for_post_g105_pose": 100.0 * measurement.d_seg < live_target_score,
            "source_float_to_wire_regret": {
                "status": "unmeasured",
                "value": None,
                "reason": "production screen begins at parsed public G105/G110 wire",
            },
            "pose_initializer_identity_sha256": authority.g112.initializer.checkpoint_sha256,
            "physical_stage_identity": authority.physical_stage_identity,
            "physical_stage_identity_sha256": authority.physical_stage_identity_sha256,
            "measurement_identity_sha256": measurement.measurement_identity_sha256,
            "public_runtime_tree_sha256": measurement.public_runtime_tree_sha256,
            "selected_archive": _regular_file_identity(
                engine.archive_path,
                name="selected G120 semantic archive",
            ),
            "selected_archive_sha256": _sha256(selected.archive),
            "pointer_snapshot_identity_sha256": pointer_snapshot_identity_sha256,
            "live_target_score": live_target_score,
        }
        return cls(
            **{key: value for key, value in body.items() if key != "schema"},
            row_identity_sha256=_sha256(_canonical_json(body)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CROSS_STAGE_ROW_SCHEMA,
            "stage_tag": self.stage_tag,
            "row_identity_sha256": self.row_identity_sha256,
            "d_seg_wire": self.d_seg_wire,
            "exact_archive_bytes": self.exact_archive_bytes,
            "semantic_action": self.semantic_action,
            "distortion_only_value": self.distortion_only_value,
            "retained_for_post_g105_pose": self.retained_for_post_g105_pose,
            "source_float_to_wire_regret": self.source_float_to_wire_regret,
            "pose_initializer_identity_sha256": self.pose_initializer_identity_sha256,
            "physical_stage_identity": self.physical_stage_identity,
            "physical_stage_identity_sha256": self.physical_stage_identity_sha256,
            "measurement_identity_sha256": self.measurement_identity_sha256,
            "public_runtime_tree_sha256": self.public_runtime_tree_sha256,
            "selected_archive": self.selected_archive,
            "selected_archive_sha256": self.selected_archive_sha256,
            "pointer_snapshot_identity_sha256": self.pointer_snapshot_identity_sha256,
            "live_target_score": self.live_target_score,
        }


def retain_cross_stage_rows(
    rows: Sequence[G120CrossStageParetoRowV1],
) -> tuple[G120CrossStageParetoRowV1, ...]:
    """Retain every strict distortion-open stage; order deterministically."""

    if not rows or any(type(row) is not G120CrossStageParetoRowV1 for row in rows):
        raise G120ProductionAuthorityError("cross-stage rows must be a nonempty typed sequence")
    by_identity: dict[str, G120CrossStageParetoRowV1] = {}
    stage_identity: dict[str, str] = {}
    for row in rows:
        prior = stage_identity.setdefault(row.stage_tag, row.row_identity_sha256)
        if prior != row.row_identity_sha256:
            raise G120ProductionAuthorityError("one physical stage tag names multiple production rows")
        by_identity[row.row_identity_sha256] = row
    retained = [row for row in by_identity.values() if row.retained_for_post_g105_pose]
    return tuple(
        sorted(
            retained,
            key=lambda row: (
                row.semantic_action,
                row.d_seg_wire,
                row.exact_archive_bytes,
                row.stage_tag,
                row.row_identity_sha256,
            ),
        )
    )


def _row_from_dict(value: object) -> G120CrossStageParetoRowV1:
    if type(value) is not dict or value.get("schema") != CROSS_STAGE_ROW_SCHEMA:
        raise G120ProductionAuthorityError("cross-stage ledger contains a malformed row")
    expected = {field.name for field in dataclasses.fields(G120CrossStageParetoRowV1)} | {"schema"}
    if set(value) != expected:
        raise G120ProductionAuthorityError("cross-stage row field census differs")
    kwargs = {key: value[key] for key in expected if key != "schema"}
    row = G120CrossStageParetoRowV1(**kwargs)
    body = row.to_dict()
    claimed = body.pop("row_identity_sha256")
    if claimed != _sha256(_canonical_json(body)):
        raise G120ProductionAuthorityError("cross-stage row self-hash differs")
    return row


def _publish_cross_stage_files(
    *,
    cross_stage_dir: Path,
    row: G120CrossStageParetoRowV1,
) -> tuple[Path, Path]:
    root = _durable_dir(cross_stage_dir, name="cross_stage_dir")
    pareto_path = root / PARETO_FILENAME
    best_path = root / BEST_FILENAME
    lock_path = root / ".g120_cross_stage.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior_rows: list[G120CrossStageParetoRowV1] = []
        if pareto_path.exists() or pareto_path.is_symlink():
            if pareto_path.is_symlink() or not pareto_path.is_file():
                raise G120ProductionAuthorityError("cross-stage Pareto path is not regular")
            raw = pareto_path.read_bytes()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise G120ProductionAuthorityError("cross-stage Pareto ledger is corrupt") from exc
            if (
                type(value) is not dict
                or _canonical_json(value) != raw
                or value.get("schema") != CROSS_STAGE_LEDGER_SCHEMA
                or type(value.get("rows")) is not list
            ):
                raise G120ProductionAuthorityError("cross-stage Pareto ledger is noncanonical")
            prior_rows = [_row_from_dict(item) for item in value["rows"]]
        if best_path.exists() or best_path.is_symlink():
            if best_path.is_symlink() or not best_path.is_file():
                raise G120ProductionAuthorityError("cross-stage BEST path is not regular")
            best_raw = best_path.read_bytes()
            try:
                best = json.loads(best_raw)
            except json.JSONDecodeError as exc:
                raise G120ProductionAuthorityError("cross-stage BEST pointer is corrupt") from exc
            if "levelset_best.json" in best_raw.decode("ascii", errors="ignore"):
                raise G120ProductionAuthorityError("legacy levelset_best.json pointer is forbidden")
            if type(best) is not dict or best.get("schema") != BEST_POINTER_SCHEMA:
                raise G120ProductionAuthorityError("cross-stage BEST pointer schema differs")
        if prior_rows and any(
            item.pointer_snapshot_identity_sha256 != row.pointer_snapshot_identity_sha256
            or item.live_target_score != row.live_target_score
            for item in prior_rows
        ):
            raise G120ProductionAuthorityError("cross-stage Pareto ledger cannot mix competitive pointer snapshots")
        retained = retain_cross_stage_rows([*prior_rows, row])
        ledger = {
            "schema": CROSS_STAGE_LEDGER_SCHEMA,
            "retention_rule": "retain_every_stage_with_100*d_seg_wire<live_target_until_post_G105_pose",
            "rows": [item.to_dict() for item in retained],
        }
        _atomic_write(pareto_path, _canonical_json(ledger), replace=True)
        if retained:
            best_row = retained[0]
            best = {
                "schema": BEST_POINTER_SCHEMA,
                "pareto_filename": PARETO_FILENAME,
                "row_identity_sha256": best_row.row_identity_sha256,
                "stage_tag": best_row.stage_tag,
                "selection_rule": "minimum_semantic_action_then_exact_deterministic_tie_break",
                "legacy_levelset_best_allowed": False,
            }
            _atomic_write(best_path, _canonical_json(best), replace=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return pareto_path, best_path


@dataclass(frozen=True, slots=True)
class G120ProductionStageResultV1:
    receipt_path: Path
    cross_stage_row_path: Path
    pareto_path: Path
    best_pointer_path: Path
    engine: G111ParsedG105StageSelectionV1
    public_measurement: G120PublicPluginMeasurementV1
    cross_stage_row: G120CrossStageParetoRowV1
    receipt: dict[str, Any]


def run_g120_parsed_stage_production_authority_v1(
    *,
    repo_root: Path,
    g112_partition_receipt: Path,
    expected_g112_partition_receipt_sha256: str,
    out_dir: Path,
    progress_dir: Path,
    measurement_cache_dir: Path,
    cross_stage_dir: Path,
) -> G120ProductionStageResultV1:
    """Physically screen one G112 stage through the shipped public receiver.

    There are intentionally no callback, label, target-score, pointer-hash,
    scorer-hash, stage-tag, or checkpoint-identity arguments.
    """

    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise G120ProductionAuthorityError("repo_root must be an absolute pathlib.Path")
    durable_out = _durable_dir(out_dir, name="out_dir")
    durable_progress = _durable_dir(progress_dir, name="progress_dir")
    cache_root = _ssd_cache_dir(measurement_cache_dir)
    authority = _open_production_authority(
        repo_root=repo_root,
        g112_partition_receipt=g112_partition_receipt,
        expected_g112_partition_receipt_sha256=expected_g112_partition_receipt_sha256,
        measurement_cache_dir=cache_root,
    )
    snapshot = load_dynamic_frontier_target(repo_root=repo_root)
    pointer_identity = dynamic_snapshot_identity_sha256(snapshot)
    engine = compile_select_parsed_g105_stage_v1(
        config=authority.config,  # type: ignore[arg-type]
        semantic_params=authority.g112.semantic_child.shared_params,
        odd_y1=authority.g112.semantic_child.code_y1,
        target_labels=authority.target_labels,
        seg_argmax_batch_scorer=authority.scorer,
        injected_inputs_are_test_only=True,
        seg_scorer_identity_sha256=authority.seg_scorer_identity_sha256,
        source_checkpoint_identity_sha256=authority.physical_stage_identity_sha256,
        pose_initializer_identity_sha256=authority.g112.initializer.checkpoint_sha256,
        effective_frontier_target=float(snapshot.target_score),
        pointer_snapshot_identity_sha256=pointer_identity,
        out_dir=durable_out,
        progress_dir=durable_progress,
        stage_tag=authority.stage_tag,
    )
    _validate_engine_handoff(
        authority=authority,
        engine=engine,
        pointer_identity_sha256=pointer_identity,
    )
    measurement = _measure_public_plugin_surface(
        authority=authority,
        engine=engine,
        progress_dir=durable_progress,
    )
    try:
        verify_dynamic_frontier_target_snapshot(snapshot)
    except Exception as exc:
        raise G120ProductionAuthorityError("live frontier pointer changed or became stale during screen") from exc
    observation = semantic_stage_conditional_observation(
        d_seg=measurement.d_seg,
        archive_bytes=len(engine.selected.archive),
        effective_frontier_target=float(snapshot.target_score),
    )
    row = G120CrossStageParetoRowV1.build(
        authority=authority,
        engine=engine,
        measurement=measurement,
        pointer_snapshot_identity_sha256=pointer_identity,
        live_target_score=float(snapshot.target_score),
    )
    pareto_path, best_path = _publish_cross_stage_files(
        cross_stage_dir=cross_stage_dir,
        row=row,
    )
    namespace = f"{authority.stage_tag}.ptr_{pointer_identity}"
    row_path = durable_out / f"{namespace}.g120_cross_stage_row.json"
    receipt_path = durable_out / f"{namespace}.g120_production_authority.receipt.json"
    _atomic_write(row_path, _canonical_json(row.to_dict()))
    receipt_body = {
        "schema": SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "production_authority_closed": True,
        "authoritative_through_R_for_semantic_screen": True,
        "exact_shipped_public_plugin_tree_scored": True,
        "public_runtime_tree": authority.public_runtime_tree,
        "repository_public_population_equal": True,
        "physical_stage_identity": authority.physical_stage_identity,
        "physical_stage_identity_sha256": authority.physical_stage_identity_sha256,
        "g109_custody": authority.g109_custody,
        "frozen_cpu_segnet_constructed_by_wrapper": True,
        "arbitrary_callback_accepted": False,
        "hardcoded_frontier_target_accepted": False,
        "pointer_snapshot": _snapshot_payload(snapshot),
        "pointer_snapshot_identity_sha256": pointer_identity,
        "pointer_reverified_after_n600": True,
        "measurement_cache": {
            "pointer_independent": True,
            "measurement_identity_sha256": measurement.measurement_identity_sha256,
            "public_batch_receipt_chain_sha256": measurement.batch_receipt_chain_sha256,
            "resumed_batch_count": measurement.resumed_batch_count,
        },
        "engine_receipt": {
            **_regular_file_identity(engine.receipt_path, name="G117 engine receipt"),
            "engine_only": True,
            "production_authority_source": False,
        },
        "selected_archive": {
            **_regular_file_identity(engine.archive_path, name="selected semantic archive"),
            "exact_receiver_valid_archive_bytes": len(engine.selected.archive),
        },
        "public_measurement": dataclasses.asdict(measurement),
        "conditional_observation": observation,
        "cross_stage_row": row.to_dict(),
        "cross_stage_files": {
            "row_path": str(row_path),
            "pareto_path": str(pareto_path),
            "best_pointer_path": str(best_path),
            "legacy_levelset_best_allowed": False,
        },
        "retained_for_post_g105_pose": row.retained_for_post_g105_pose,
        "retention_rule": "retain_every_stage_with_100*d_seg_wire<live_target_until_post_G105_pose",
        "pose_measured": False,
        "semantic_only": True,
        "nonpromotable_until_post_g105_pose_and_exact_archive_eval": True,
        "contest_score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "production_launch_performed": False,
    }
    receipt = {
        **receipt_body,
        "receipt_body_sha256": _sha256(_canonical_json(receipt_body)),
    }
    _atomic_write(receipt_path, _canonical_json(receipt))
    return G120ProductionStageResultV1(
        receipt_path=receipt_path,
        cross_stage_row_path=row_path,
        pareto_path=pareto_path,
        best_pointer_path=best_path,
        engine=engine,
        public_measurement=measurement,
        cross_stage_row=row,
        receipt=receipt,
    )


__all__ = [
    "BEST_FILENAME",
    "BEST_POINTER_SCHEMA",
    "CROSS_STAGE_LEDGER_SCHEMA",
    "CROSS_STAGE_ROW_SCHEMA",
    "EVIDENCE_AXIS",
    "PARETO_FILENAME",
    "PUBLIC_BATCH_SCHEMA",
    "PUBLIC_MEASUREMENT_SCHEMA",
    "SCHEMA",
    "G120CrossStageParetoRowV1",
    "G120ProductionAuthorityError",
    "G120ProductionStageResultV1",
    "G120PublicPluginMeasurementV1",
    "dynamic_snapshot_identity_sha256",
    "public_plugin_tree_identity",
    "retain_cross_stage_rows",
    "run_g120_parsed_stage_production_authority_v1",
]
