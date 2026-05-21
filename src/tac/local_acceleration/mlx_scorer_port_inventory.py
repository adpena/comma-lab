# SPDX-License-Identifier: MIT
"""MLX port coverage inventory for the contest auth scorer.

The local MLX path is useful only if we can state exactly which upstream
PoseNet/SegNet operations have a faithful local implementation and which still
need adapters or rewrites.  This module does not claim scorer parity.  It
builds a portable JSON inventory that turns the full-port work into measurable
coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_scorer_port_inventory.v1"

STATUS_DIRECT = "direct_mlx_layer"
STATUS_ADAPTER = "adapter_required"
STATUS_COMPOSITE = "composite_rewrite_required"
STATUS_UNKNOWN = "unknown_or_unclassified"


@dataclass(frozen=True)
class LayerRule:
    """Static mapping from a PyTorch module class to MLX port status."""

    class_path: str
    status: str
    mlx_equivalent: str
    note: str


@dataclass(frozen=True)
class ModuleInventoryRow:
    """Aggregated inventory row for one module class."""

    class_path: str
    class_name: str
    count: int
    direct_parameter_count: int
    recursive_parameter_count: int
    direct_parameter_bytes: int
    direct_buffer_count: int
    direct_buffer_bytes: int
    direct_state_key_count: int
    leaf_count: int
    status: str
    mlx_equivalent: str
    note: str


@dataclass(frozen=True)
class ModelInventory:
    """Inventory for one scorer model."""

    name: str
    module_count: int
    leaf_module_count: int
    parameter_count: int
    parameter_bytes: int
    buffer_count: int
    buffer_bytes: int
    state_dict_key_count: int
    state_dict_parameter_key_count: int
    state_dict_buffer_key_count: int
    state_dict_tensor_bytes: int
    state_dict_dtypes: dict[str, int]
    status_counts: dict[str, int]
    rows: tuple[ModuleInventoryRow, ...]


DIRECT_RULES: tuple[LayerRule, ...] = (
    LayerRule(
        "torch.nn.modules.linear.Linear",
        STATUS_DIRECT,
        "mlx.nn.Linear",
        "Weight/bias transpose policy still must be tested per tensor.",
    ),
    LayerRule(
        "torch.nn.modules.activation.ReLU",
        STATUS_DIRECT,
        "mlx.nn.ReLU",
        "Elementwise activation; preserve inplace=False semantics at port boundary.",
    ),
    LayerRule(
        "torch.nn.modules.activation.SiLU",
        STATUS_DIRECT,
        "mlx.nn.SiLU",
        "Elementwise activation.",
    ),
    LayerRule(
        "torch.nn.modules.activation.Sigmoid",
        STATUS_DIRECT,
        "mlx.nn.Sigmoid",
        "Elementwise activation.",
    ),
    LayerRule(
        "torch.nn.modules.flatten.Flatten",
        STATUS_DIRECT,
        "mlx.reshape",
        "Shape-only operation; preserve start_dim/end_dim semantics.",
    ),
    LayerRule(
        "torch.nn.modules.linear.Identity",
        STATUS_DIRECT,
        "identity",
        "No-op layer.",
    ),
    LayerRule(
        "torch.nn.modules.dropout.Dropout",
        STATUS_DIRECT,
        "mlx.nn.Dropout",
        "Eval-mode dropout should be identity; training mode must not leak into scorer.",
    ),
)

ADAPTER_RULES: tuple[LayerRule, ...] = (
    LayerRule(
        "torch.nn.modules.conv.Conv2d",
        STATUS_ADAPTER,
        "mlx.nn.Conv2d",
        "Requires explicit NCHW/BHWC layout contract and grouped/depthwise parity tests.",
    ),
    LayerRule(
        "torch.nn.modules.batchnorm.BatchNorm1d",
        STATUS_ADAPTER,
        "mlx.nn.BatchNorm",
        "Requires eval-mode running-stat parity and affine parameter mapping.",
    ),
    LayerRule(
        "torch.nn.modules.batchnorm.BatchNorm2d",
        STATUS_ADAPTER,
        "mlx.nn.BatchNorm",
        "Requires eval-mode running-stat parity and layout-aware channel axis handling.",
    ),
    LayerRule(
        "torch.nn.modules.pooling.AdaptiveAvgPool2d",
        STATUS_ADAPTER,
        "mlx.core.mean or mlx pooling adapter",
        "Adaptive output-size semantics need explicit implementation.",
    ),
    LayerRule(
        "torch.nn.modules.pooling.AvgPool2d",
        STATUS_ADAPTER,
        "mlx.nn.AvgPool2d",
        "Padding/count_include_pad parity must be tested.",
    ),
    LayerRule(
        "torch.nn.modules.pooling.MaxPool2d",
        STATUS_ADAPTER,
        "mlx.nn.MaxPool2d",
        "Padding/dilation/ceil_mode parity must be tested.",
    ),
    LayerRule(
        "torch.nn.modules.upsampling.UpsamplingBilinear2d",
        STATUS_ADAPTER,
        "mlx image/array resize adapter",
        "Must match torch interpolate bilinear align_corners default exactly.",
    ),
)

COMPOSITE_RULES: tuple[LayerRule, ...] = (
    LayerRule(
        "torch.nn.modules.container.Sequential",
        STATUS_COMPOSITE,
        "container",
        "Container; port status is determined by children.",
    ),
    LayerRule(
        "torch.nn.modules.container.ModuleList",
        STATUS_COMPOSITE,
        "container",
        "Container; port status is determined by children.",
    ),
    LayerRule(
        "torch.nn.modules.container.ModuleDict",
        STATUS_COMPOSITE,
        "container",
        "Container; port status is determined by children.",
    ),
)

PREFIX_RULES: tuple[LayerRule, ...] = (
    LayerRule(
        "timm.",
        STATUS_COMPOSITE,
        "custom FastViT/EfficientNet MLX rewrite",
        "Third-party composite; inventory leaves exact block coverage explicit.",
    ),
    LayerRule(
        "segmentation_models_pytorch.",
        STATUS_COMPOSITE,
        "custom Unet/EfficientNet decoder rewrite",
        "Third-party composite; inventory leaves exact block coverage explicit.",
    ),
    LayerRule(
        "modules.",
        STATUS_COMPOSITE,
        "upstream scorer wrapper rewrite",
        "Challenge-local composite; port its children and preserve forward contract.",
    ),
)

RULES_BY_CLASS = {rule.class_path: rule for rule in (*DIRECT_RULES, *ADAPTER_RULES, *COMPOSITE_RULES)}

SCORER_PORT_BLOCKING_STATUSES = frozenset(
    {STATUS_ADAPTER, STATUS_COMPOSITE, STATUS_UNKNOWN}
)


def inspect_model(name: str, module: Any) -> ModelInventory:
    """Inspect one PyTorch module and return an aggregate MLX port inventory."""

    rows_by_class: dict[str, dict[str, Any]] = {}
    module_count = 0
    leaf_module_count = 0
    parameter_count = 0
    parameter_bytes = 0
    buffer_count = 0
    buffer_bytes = 0

    state_dict = module.state_dict()
    named_modules = dict(module.named_modules())
    state_key_kinds = _state_key_kinds(module)

    for module_name, child in named_modules.items():
        module_count += 1
        children = list(child.children())
        leaf = not children
        if leaf:
            leaf_module_count += 1
        class_path = _class_path(child)
        rule = classify_module_class(class_path)
        direct_params = list(child.parameters(recurse=False))
        recursive_params = list(child.parameters(recurse=True))
        direct_buffers = list(child.buffers(recurse=False))
        direct_state_keys = _direct_state_keys_for_module(
            module_name=module_name,
            module=child,
        )
        direct_param_count = sum(int(p.numel()) for p in direct_params)
        recursive_param_count = sum(int(p.numel()) for p in recursive_params)
        direct_param_bytes = sum(_tensor_nbytes(p) for p in direct_params)
        direct_buffer_count = sum(int(b.numel()) for b in direct_buffers)
        direct_buffer_bytes = sum(_tensor_nbytes(b) for b in direct_buffers)
        parameter_count += direct_param_count
        parameter_bytes += direct_param_bytes
        buffer_count += direct_buffer_count
        buffer_bytes += direct_buffer_bytes

        row = rows_by_class.setdefault(
            class_path,
            {
                "class_path": class_path,
                "class_name": type(child).__name__,
                "count": 0,
                "direct_parameter_count": 0,
                "recursive_parameter_count": 0,
                "direct_parameter_bytes": 0,
                "direct_buffer_count": 0,
                "direct_buffer_bytes": 0,
                "direct_state_key_count": 0,
                "leaf_count": 0,
                "status": rule.status,
                "mlx_equivalent": rule.mlx_equivalent,
                "note": rule.note,
            },
        )
        row["count"] += 1
        row["direct_parameter_count"] += direct_param_count
        row["recursive_parameter_count"] += recursive_param_count
        row["direct_parameter_bytes"] += direct_param_bytes
        row["direct_buffer_count"] += direct_buffer_count
        row["direct_buffer_bytes"] += direct_buffer_bytes
        row["direct_state_key_count"] += len(direct_state_keys)
        row["leaf_count"] += int(leaf)

    rows = tuple(
        ModuleInventoryRow(**row)
        for row in sorted(
            rows_by_class.values(),
            key=lambda item: (
                _status_sort_key(str(item["status"])),
                -int(item["direct_parameter_count"]),
                -int(item["count"]),
                str(item["class_path"]),
            ),
        )
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + row.count

    return ModelInventory(
        name=name,
        module_count=module_count,
        leaf_module_count=leaf_module_count,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        buffer_count=buffer_count,
        buffer_bytes=buffer_bytes,
        state_dict_key_count=len(state_dict),
        state_dict_parameter_key_count=sum(1 for kind in state_key_kinds.values() if kind == "parameter"),
        state_dict_buffer_key_count=sum(1 for kind in state_key_kinds.values() if kind == "buffer"),
        state_dict_tensor_bytes=sum(_tensor_nbytes(tensor) for tensor in state_dict.values()),
        state_dict_dtypes=_dtype_counts(state_dict.values()),
        status_counts=dict(sorted(status_counts.items())),
        rows=rows,
    )


def build_upstream_scorer_port_inventory(
    *,
    repo_root: str | Path = ".",
    load_weights: bool = False,
) -> dict[str, Any]:
    """Instantiate upstream scorer modules and return the MLX port inventory."""

    root = Path(repo_root).resolve()
    upstream_dir = root / "upstream"
    if not upstream_dir.is_dir():
        raise FileNotFoundError(f"missing upstream directory: {upstream_dir}")
    with _temporary_sys_path(upstream_dir):
        upstream_modules = importlib.import_module("modules")
        models = {
            "posenet": upstream_modules.PoseNet().eval(),
            "segnet": upstream_modules.SegNet().eval(),
        }
        weights_loaded = False
        if load_weights:
            device = _torch_cpu_device()
            for model in models.values():
                model.to(device)
            dist = upstream_modules.DistortionNet()
            dist.posenet = models["posenet"]
            dist.segnet = models["segnet"]
            dist.load_state_dicts(
                upstream_modules.posenet_sd_path,
                upstream_modules.segnet_sd_path,
                device,
            )
            weights_loaded = True

    inventories = tuple(inspect_model(name, model) for name, model in models.items())
    total_blocking_modules = sum(
        count
        for inv in inventories
        for status, count in inv.status_counts.items()
        if status in SCORER_PORT_BLOCKING_STATUSES
    )
    total_unknown_modules = sum(
        inv.status_counts.get(STATUS_UNKNOWN, 0) for inv in inventories
    )
    total_parameters = sum(inv.parameter_count for inv in inventories)
    total_parameter_bytes = sum(inv.parameter_bytes for inv in inventories)
    total_state_dict_keys = sum(inv.state_dict_key_count for inv in inventories)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "repo_root": str(root),
        "upstream_modules_path": str(upstream_dir / "modules.py"),
        "weights_loaded": weights_loaded,
        "models": [model_inventory_to_dict(inv) for inv in inventories],
        "summary": {
            "model_count": len(inventories),
            "total_modules": sum(inv.module_count for inv in inventories),
            "total_leaf_modules": sum(inv.leaf_module_count for inv in inventories),
            "total_parameter_count": total_parameters,
            "total_parameter_bytes": total_parameter_bytes,
            "total_state_dict_key_count": total_state_dict_keys,
            "total_state_dict_tensor_bytes": sum(
                inv.state_dict_tensor_bytes for inv in inventories
            ),
            "total_buffer_count": sum(inv.buffer_count for inv in inventories),
            "total_buffer_bytes": sum(inv.buffer_bytes for inv in inventories),
            "total_blocking_modules": total_blocking_modules,
            "total_unknown_modules": total_unknown_modules,
            "full_mlx_port_claim_allowed": False,
            "claim_blockers": _claim_blockers(
                total_blocking_modules=total_blocking_modules,
                total_unknown_modules=total_unknown_modules,
                total_state_dict_keys=total_state_dict_keys,
            ),
        },
        "blocking_statuses": sorted(SCORER_PORT_BLOCKING_STATUSES),
        "port_completion_contract": {
            "required_before_full_port_claim": [
                "all rows classified direct_mlx_layer or backed by tested adapter",
                "all upstream safetensor/state_dict keys mapped to MLX tensor names",
                "every mapped state key carries dtype, shape, layout, and transform policy",
                "NCHW/BHWC layout adapters covered by tensor-parity tests",
                "torch interpolate preprocessing parity covered by scorer-input hash identity",
                "FastViT PoseNet forward parity on fixed scorer inputs",
                "EfficientNet-B2/Unet SegNet forward parity on fixed scorer inputs",
                "distortion component parity against auth-eval scorer outputs on byte-closed raw",
                "CUDA and contest-CPU auth-eval remain the only promotion axes",
            ],
        },
    }
    return payload


def classify_module_class(class_path: str) -> LayerRule:
    """Classify a PyTorch module class for MLX port planning."""

    direct = RULES_BY_CLASS.get(class_path)
    if direct is not None:
        return direct
    for rule in PREFIX_RULES:
        if class_path.startswith(rule.class_path):
            return LayerRule(
                class_path=class_path,
                status=rule.status,
                mlx_equivalent=rule.mlx_equivalent,
                note=rule.note,
            )
    return LayerRule(
        class_path=class_path,
        status=STATUS_UNKNOWN,
        mlx_equivalent="none_registered",
        note="No MLX port rule registered; classify before claiming full scorer parity.",
    )


def model_inventory_to_dict(inventory: ModelInventory) -> dict[str, Any]:
    """Return JSON-serializable inventory data."""

    payload = asdict(inventory)
    payload["rows"] = [asdict(row) for row in inventory.rows]
    return payload


def write_inventory(payload: dict[str, Any], output: str | Path) -> None:
    """Write inventory JSON."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _claim_blockers(
    *,
    total_blocking_modules: int,
    total_unknown_modules: int,
    total_state_dict_keys: int,
) -> list[str]:
    blockers = [
        "inventory_is_coverage_plan_not_full_scorer_port",
        "full_port_requires_forward_parity_tests_on_fixed_scorer_inputs",
        f"state_dict_key_mapping_required:{total_state_dict_keys}",
    ]
    if total_blocking_modules:
        blockers.append(f"blocking_modules_present:{total_blocking_modules}")
    if total_unknown_modules:
        blockers.append(f"unknown_modules_present:{total_unknown_modules}")
    return blockers


def _class_path(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _status_sort_key(status: str) -> int:
    order = {
        STATUS_UNKNOWN: 0,
        STATUS_ADAPTER: 1,
        STATUS_COMPOSITE: 2,
        STATUS_DIRECT: 3,
    }
    return order.get(status, 99)


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _direct_state_keys_for_module(*, module_name: str, module: Any) -> list[str]:
    prefix = f"{module_name}." if module_name else ""
    keys: list[str] = []
    keys.extend(prefix + name for name, _ in module.named_parameters(recurse=False))
    keys.extend(prefix + name for name, _ in module.named_buffers(recurse=False))
    return keys


def _state_key_kinds(module: Any) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for name, _ in module.named_parameters(recurse=True):
        kinds[name] = "parameter"
    for name, _ in module.named_buffers(recurse=True):
        kinds[name] = "buffer"
    return kinds


def _dtype_counts(tensors: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in tensors:
        key = str(tensor.dtype)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _torch_cpu_device() -> Any:
    import torch

    return torch.device("cpu")


class _temporary_sys_path:
    """Temporarily prepend import paths."""

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
    "STATUS_ADAPTER",
    "STATUS_COMPOSITE",
    "STATUS_DIRECT",
    "STATUS_UNKNOWN",
    "LayerRule",
    "ModelInventory",
    "ModuleInventoryRow",
    "build_upstream_scorer_port_inventory",
    "classify_module_class",
    "inspect_model",
    "model_inventory_to_dict",
    "write_inventory",
]
