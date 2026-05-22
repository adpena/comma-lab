# SPDX-License-Identifier: MIT
"""State-dict layout map for a faithful MLX contest-scorer port."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_scorer_state_map.v1"

STATUS_MAPPED = "mapped_tensor_transform"
STATUS_UNUSED_EVAL_BUFFER = "unused_eval_buffer"
STATUS_REQUIRES_ADAPTER = "requires_module_adapter"
STATUS_UNMAPPED = "unmapped"


@dataclass(frozen=True)
class StateKeyMappingRow:
    """One upstream PyTorch state key and its MLX tensor-layout policy."""

    model_name: str
    state_key: str
    owner_module: str
    owner_class_path: str
    key_kind: str
    source_dtype: str
    source_shape: list[int]
    transform_policy: str
    target_shape: list[int] | None
    mapping_status: str
    note: str


def build_upstream_scorer_state_map(
    *,
    repo_root: str | Path = ".",
    load_weights: bool = False,
) -> dict[str, Any]:
    """Return a key-level PyTorch-to-MLX state mapping plan for PoseNet/SegNet."""

    root = Path(repo_root).resolve()
    upstream_dir = root / "upstream"
    if not upstream_dir.is_dir():
        raise FileNotFoundError(f"missing upstream directory: {upstream_dir}")
    with _temporary_sys_path(upstream_dir):
        modules = importlib.import_module("modules")
        models = {
            "posenet": modules.PoseNet().eval(),
            "segnet": modules.SegNet().eval(),
        }
        weights_loaded = False
        if load_weights:
            device = _torch_cpu_device()
            dist = modules.DistortionNet()
            dist.posenet = models["posenet"]
            dist.segnet = models["segnet"]
            dist.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, device)
            weights_loaded = True

    rows: list[StateKeyMappingRow] = []
    for model_name, model in models.items():
        rows.extend(_map_model_state(model_name, model))

    status_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.mapping_status] = status_counts.get(row.mapping_status, 0) + 1
        policy_counts[row.transform_policy] = policy_counts.get(row.transform_policy, 0) + 1
        dtype_counts[row.source_dtype] = dtype_counts.get(row.source_dtype, 0) + 1

    hard_unmapped = status_counts.get(STATUS_UNMAPPED, 0)
    requires_adapter = status_counts.get(STATUS_REQUIRES_ADAPTER, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "repo_root": str(root),
        "upstream_modules_path": str(upstream_dir / "modules.py"),
        "weights_loaded": weights_loaded,
        "summary": {
            "state_key_count": len(rows),
            "mapped_or_intentionally_unused_key_count": len(rows) - hard_unmapped,
            "hard_unmapped_key_count": hard_unmapped,
            "requires_module_adapter_key_count": requires_adapter,
            "status_counts": dict(sorted(status_counts.items())),
            "transform_policy_counts": dict(sorted(policy_counts.items())),
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "full_mlx_port_claim_allowed": False,
            "claim_blockers": _claim_blockers(
                hard_unmapped_key_count=hard_unmapped,
                requires_module_adapter_key_count=requires_adapter,
            ),
        },
        "rows": [asdict(row) for row in rows],
        "port_completion_contract": {
            "required_before_full_port_claim": [
                "hard_unmapped_key_count == 0",
                "all requires_module_adapter rows have passing PyTorch-vs-MLX layer tests",
                "Conv2d OIHW-to-OHWI transforms are verified for grouped/depthwise cases",
                "BatchNorm eval buffers match upstream running-stat behavior",
                "num_batches_tracked omission is covered by eval-mode parity tests",
                "full PoseNet and SegNet forward outputs match on fixed scorer-input tensors",
            ],
        },
    }


def write_state_map(payload: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _map_model_state(model_name: str, model: Any) -> list[StateKeyMappingRow]:
    modules = dict(model.named_modules())
    parameter_keys = {name for name, _ in model.named_parameters(recurse=True)}
    buffer_keys = {name for name, _ in model.named_buffers(recurse=True)}
    rows: list[StateKeyMappingRow] = []
    for key, tensor in model.state_dict().items():
        owner_name, local_name, owner = _owner_for_state_key(key, modules)
        owner_class_path = _class_path(owner) if owner is not None else ""
        key_kind = "parameter" if key in parameter_keys else "buffer" if key in buffer_keys else "state"
        policy, target_shape, status, note = _transform_policy(
            owner_class_path=owner_class_path,
            local_name=local_name,
            source_shape=[int(v) for v in tensor.shape],
            key_kind=key_kind,
        )
        rows.append(
            StateKeyMappingRow(
                model_name=model_name,
                state_key=key,
                owner_module=owner_name,
                owner_class_path=owner_class_path,
                key_kind=key_kind,
                source_dtype=str(tensor.dtype),
                source_shape=[int(v) for v in tensor.shape],
                transform_policy=policy,
                target_shape=target_shape,
                mapping_status=status,
                note=note,
            )
        )
    return sorted(rows, key=lambda row: (row.model_name, row.state_key))


def _transform_policy(
    *,
    owner_class_path: str,
    local_name: str,
    source_shape: list[int],
    key_kind: str,
) -> tuple[str, list[int] | None, str, str]:
    if local_name == "num_batches_tracked":
        return (
            "drop_eval_only_num_batches_tracked",
            None,
            STATUS_UNUSED_EVAL_BUFFER,
            "PyTorch BatchNorm training counter is unused in eval-mode scorer forward.",
        )
    if owner_class_path == "torch.nn.modules.conv.Conv2d" and local_name == "weight":
        if len(source_shape) != 4:
            return "conv2d_weight_unsupported_rank", None, STATUS_UNMAPPED, "Conv2d weight must be OIHW."
        out_c, in_per_group, kh, kw = source_shape
        return (
            "conv2d_weight_oihw_to_ohwi",
            [out_c, kh, kw, in_per_group],
            STATUS_REQUIRES_ADAPTER,
            "MLX Conv2d stores weight as O,H,W,I_per_group; adapter parity required.",
        )
    if owner_class_path == "torch.nn.modules.linear.Linear" and local_name == "weight":
        return (
            "linear_weight_identity_out_in",
            list(source_shape),
            STATUS_MAPPED,
            "PyTorch Linear and MLX Linear both store weight as out,in.",
        )
    if owner_class_path in {
        "torch.nn.modules.batchnorm.BatchNorm1d",
        "torch.nn.modules.batchnorm.BatchNorm2d",
    }:
        return (
            "batchnorm_1d_vector_identity",
            list(source_shape),
            STATUS_REQUIRES_ADAPTER,
            "BN tensor layout is identity, but eval running-stat semantics need adapter parity.",
        )
    if local_name in {"weight", "bias", "running_mean", "running_var", "_mean", "_std", "gamma"}:
        return (
            "identity",
            list(source_shape),
            STATUS_REQUIRES_ADAPTER if owner_class_path.startswith("timm.") else STATUS_MAPPED,
            "Tensor shape is preserved; owning module forward still controls parity.",
        )
    if key_kind == "buffer":
        return (
            "buffer_identity",
            list(source_shape),
            STATUS_REQUIRES_ADAPTER,
            "Buffer shape is preserved; owning module forward still controls parity.",
        )
    return (
        "unmapped",
        None,
        STATUS_UNMAPPED,
        "No state-key transform policy registered.",
    )


def _owner_for_state_key(key: str, modules: dict[str, Any]) -> tuple[str, str, Any | None]:
    parts = key.split(".")
    for idx in range(len(parts) - 1, -1, -1):
        owner_name = ".".join(parts[:idx])
        if owner_name in modules:
            local_name = ".".join(parts[idx:])
            return owner_name, local_name, modules[owner_name]
    return "", key, modules.get("")


def _claim_blockers(
    *,
    hard_unmapped_key_count: int,
    requires_module_adapter_key_count: int,
) -> list[str]:
    blockers = [
        "state_map_is_layout_plan_not_full_scorer_port",
        "full_port_requires_layer_and_e2e_forward_parity_tests",
    ]
    if hard_unmapped_key_count:
        blockers.append(f"hard_unmapped_state_keys:{hard_unmapped_key_count}")
    if requires_module_adapter_key_count:
        blockers.append(f"state_keys_require_module_adapters:{requires_module_adapter_key_count}")
    return blockers


def _class_path(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _torch_cpu_device() -> Any:
    import torch

    return torch.device("cpu")


class _temporary_sys_path:
    def __init__(self, *paths: Path):
        self._paths = [str(path) for path in paths]
        self._old_path: list[str] | None = None

    def __enter__(self) -> None:
        self._old_path = list(sys.path)
        for path in reversed(self._paths):
            if path not in sys.path:
                sys.path.insert(0, path)

    def __exit__(self, *_exc: object) -> None:
        if self._old_path is not None:
            sys.path[:] = self._old_path


__all__ = [
    "SCHEMA_VERSION",
    "STATUS_MAPPED",
    "STATUS_REQUIRES_ADAPTER",
    "STATUS_UNMAPPED",
    "STATUS_UNUSED_EVAL_BUFFER",
    "StateKeyMappingRow",
    "build_upstream_scorer_state_map",
    "write_state_map",
]
