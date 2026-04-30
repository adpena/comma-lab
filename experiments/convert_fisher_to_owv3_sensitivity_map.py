#!/usr/bin/env python3
"""Convert per-weight Fisher importance into an OWV3 sensitivity map.

Input schema is the artifact emitted by ``experiments/profile_hessian_per_weight.py``:

    {
      "importance": {"<module>.weight": Tensor[same shape as weight]},
      "metadata": {...}
    }

Output schema is ``tac.sensitivity_map``:

    {
      "format": "tac_score_sensitivity_map_v1",
      "sensitivities": {"<conv_module>.weight": Tensor[out_channels]},
      "metadata": {...}
    }

The default aggregation is ``sum`` because the local second-order loss model is

    E[Delta score] ~= 0.5 * sum_i H_i * E[Delta w_i^2]

for all weights in a channel. Missing Conv2d layers default to a high
``missing_value`` so OWV3 stores them as protected FP16 rather than silently
quantizing unmeasured channels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class SensitivityConversionError(ValueError):
    """Raised when Fisher -> channel sensitivity conversion is malformed."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_importance(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SensitivityConversionError(f"{path}: expected dict payload")
    raw = payload.get("importance")
    if not isinstance(raw, dict):
        raise SensitivityConversionError(f"{path}: missing importance dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            raise SensitivityConversionError(f"{path}: {key} is not a tensor")
        t = value.detach().to(torch.float32).cpu()
        if not torch.isfinite(t).all():
            raise SensitivityConversionError(f"{path}: {key} contains NaN/Inf")
        if (t < 0).any():
            raise SensitivityConversionError(f"{path}: {key} contains negative Fisher values")
        out[str(key)] = t
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SensitivityConversionError(f"{path}: metadata must be a dict")
    return out, metadata


def _aggregate_channel_importance(tensor: torch.Tensor, aggregate: str) -> torch.Tensor:
    if tensor.dim() != 4:
        raise SensitivityConversionError(
            f"expected Conv2d Fisher tensor with dim=4, got shape {tuple(tensor.shape)}"
        )
    flat = tensor.detach().to(torch.float32).cpu().reshape(int(tensor.shape[0]), -1)
    if aggregate == "sum":
        return flat.sum(dim=1)
    if aggregate == "mean":
        return flat.mean(dim=1)
    if aggregate == "max":
        return flat.max(dim=1).values
    raise SensitivityConversionError(f"unknown aggregate={aggregate!r}")


def convert_importance_to_channel_sensitivity(
    *,
    model: nn.Module,
    importance: Mapping[str, torch.Tensor],
    aggregate: str = "sum",
    missing_policy: str = "protect",
    missing_value: float = 1e-2,
) -> dict[str, torch.Tensor]:
    """Convert per-weight Fisher tensors to per-output-channel sensitivity.

    Missing policy:
    - ``protect``: emit ``missing_value`` for every channel, making the default
      OWV3 threshold protect that layer as FP16.
    - ``error``: fail if any Conv2d weight is missing.
    - ``zero``: emit zeros; intended only for synthetic tests.
    """
    if missing_policy not in {"protect", "error", "zero"}:
        raise SensitivityConversionError(
            f"missing_policy must be protect|error|zero, got {missing_policy!r}"
        )
    if missing_value < 0 or not torch.isfinite(torch.tensor(float(missing_value))):
        raise SensitivityConversionError("missing_value must be finite and non-negative")

    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        key = f"{name}.weight"
        expected_shape = tuple(module.weight.shape)
        if key not in importance:
            if missing_policy == "error":
                raise SensitivityConversionError(f"missing Fisher tensor for {key}")
            fill = 0.0 if missing_policy == "zero" else float(missing_value)
            out[key] = torch.full((int(module.weight.shape[0]),), fill, dtype=torch.float32)
            continue

        fisher = importance[key].detach().to(torch.float32).cpu()
        if tuple(fisher.shape) != expected_shape:
            raise SensitivityConversionError(
                f"{key}: Fisher shape {tuple(fisher.shape)} != model weight shape {expected_shape}"
            )
        out[key] = _aggregate_channel_importance(fisher, aggregate)

    if not out:
        raise SensitivityConversionError("model has no Conv2d weights to convert")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Renderer checkpoint/bin used to validate Conv2d shapes.")
    parser.add_argument("--fisher", type=Path, required=True,
                        help="hessian_per_weight.pt from profile_hessian_per_weight.py.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output sensitivity_map.pt.")
    parser.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--missing-policy", choices=["protect", "error", "zero"],
                        default="protect")
    parser.add_argument("--missing-value", type=float, default=1e-2,
                        help="Value used for missing Conv2d layers when policy=protect.")
    parser.add_argument("--allow-non-authoritative", action="store_true",
                        help="Allow CPU/non-CUDA Fisher metadata. Output is smoke-only.")
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Optional metadata sidecar path.")
    args = parser.parse_args(argv)

    t_start = time.monotonic()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.fisher.exists():
        raise FileNotFoundError(args.fisher)

    from tac.renderer_export import load_any_renderer_checkpoint
    from tac.sensitivity_map import (
        require_authoritative_device,
        save_sensitivity_map,
        validate_sensitivity_map_for_model,
    )

    print("=== Fisher -> OWV3 sensitivity-map conversion ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"fisher:     {args.fisher}")
    model = load_any_renderer_checkpoint(str(args.checkpoint), device="cpu")
    model.eval()

    importance, source_metadata = _load_importance(args.fisher)
    source_device = source_metadata.get("device")
    if not args.allow_non_authoritative:
        require_authoritative_device(source_device)

    sensitivities = convert_importance_to_channel_sensitivity(
        model=model,
        importance=importance,
        aggregate=args.aggregate,
        missing_policy=args.missing_policy,
        missing_value=args.missing_value,
    )
    stats = validate_sensitivity_map_for_model(
        sensitivities,
        model,
        require_all_conv=True,
    )

    elapsed = time.monotonic() - t_start
    metadata = {
        "format": "owv3_channel_sensitivity_from_fisher_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "fisher": str(args.fisher),
        "fisher_sha256": _sha256(args.fisher),
        "source_metadata": source_metadata,
        "source_device": source_device,
        "authoritative_cuda": bool(str(source_device).startswith("cuda")),
        "non_authoritative_allowed": bool(args.allow_non_authoritative),
        "aggregate": args.aggregate,
        "missing_policy": args.missing_policy,
        "missing_value": float(args.missing_value),
        "n_layers": stats.n_layers,
        "n_channels": stats.n_channels,
        "min_value": stats.min_value,
        "max_value": stats.max_value,
        "elapsed_s": elapsed,
        "evidence_label": (
            "authoritative-cuda-artifact"
            if str(source_device).startswith("cuda")
            else "smoke-only-non-authoritative"
        ),
    }
    save_sensitivity_map(args.output, sensitivities, metadata=metadata)
    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2))
    print(
        f"wrote {args.output} with {stats.n_layers} layers, "
        f"{stats.n_channels} channels, range=[{stats.min_value:.6g}, {stats.max_value:.6g}]"
    )
    if args.metadata_json:
        print(f"wrote metadata {args.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
