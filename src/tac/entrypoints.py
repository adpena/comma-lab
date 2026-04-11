"""Shared cloud entry helpers for tac-backed training scripts.

These helpers own the checkpoint, metadata, and output conventions used by
remote launchers. Platform scripts should keep only bootstrap and env wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .quantization import save_int8

SCHEMA_VERSION = 1
DEFAULT_KAGGLE_ASSET_ROOT = Path("/kaggle/input/comma-lab-private-assets")


def resolve_cloud_output_dir(project_root: Path) -> Path:
    if os.environ.get("POSTFILTER_OUTPUT_DIR"):
        return Path(os.environ["POSTFILTER_OUTPUT_DIR"])
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / "postfilter_weights"
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/postfilter_weights")
    if Path("/content").exists():
        return Path("/content/postfilter_weights")
    return project_root / "experiments" / "postfilter_weights"


def resolve_cloud_base_dir() -> Path:
    if Path("/kaggle").exists():
        return Path("/kaggle/working")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


def resolve_cloud_asset(
    project_root: Path,
    script_path: Path,
    relative_path: str,
    *,
    input_root: Path = Path("/kaggle/input"),
) -> Path:
    basename = Path(relative_path).name
    candidates = [
        project_root / relative_path,
        script_path.parent / relative_path,
        script_path.parent / basename,
        project_root / basename,
        DEFAULT_KAGGLE_ASSET_ROOT / basename,
    ]
    if input_root.exists():
        candidates.extend(sorted(input_root.rglob(basename)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return project_root / relative_path


def normalize_archive_source_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_dir():
        mkv_files = sorted(source.glob("*.mkv"))
        if not mkv_files:
            raise FileNotFoundError(f"No .mkv files found inside extracted archive dir: {source}")
        return mkv_files[0]
    return source


def resolve_cloud_archive_source(
    project_root: Path,
    script_path: Path,
    relative_path: str,
    *,
    input_root: Path = Path("/kaggle/input"),
) -> Path:
    resolved = resolve_cloud_asset(project_root, script_path, relative_path, input_root=input_root)
    if resolved.exists():
        return normalize_archive_source_path(resolved)

    relative = Path(relative_path)
    if relative.suffix.lower() == ".zip" and input_root.exists():
        stem = relative.stem
        for candidate in sorted(input_root.rglob(stem)):
            if candidate.exists():
                return normalize_archive_source_path(candidate)
    return resolved


def resolve_cloud_asset_bundle(
    project_root: Path,
    script_path: Path,
    *,
    archive_relative_path: str,
    saliency_relative_path: str,
    input_root: Path = Path("/kaggle/input"),
) -> dict[str, Path]:
    archive_path = resolve_cloud_archive_source(
        project_root,
        script_path,
        archive_relative_path,
        input_root=input_root,
    )
    saliency_path = resolve_cloud_asset(
        project_root,
        script_path,
        saliency_relative_path,
        input_root=input_root,
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Cloud archive asset not found: {archive_path}")
    if not saliency_path.exists():
        raise FileNotFoundError(f"Cloud saliency asset not found: {saliency_path}")
    return {
        "archive_path": archive_path,
        "saliency_path": saliency_path,
    }


def build_postfilter_meta(*, variant: str, hidden: int, kernel: int, alpha: float) -> dict[str, int | float | str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "hidden": int(hidden),
        "kernel": int(kernel),
        "alpha": float(alpha),
    }


def make_dilated_default_tag(hidden: int, alpha: float) -> str:
    return f"dilated_qat_ema_h{int(hidden)}_a{int(alpha)}"


def make_fixed_h32_segnet_tag(alpha: float) -> str:
    return f"cloud_segnet_attack_h32_a{int(alpha)}"


def save_best_checkpoint(
    *,
    model: nn.Module,
    shadow_state: dict[str, torch.Tensor],
    output_dir: Path,
    tag: str,
    meta: dict[str, Any],
    epoch: int,
    scorer: float,
    per_channel_int8: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"postfilter_{tag}_best_fp32.pt"
    int8_path = output_dir / f"postfilter_{tag}_best_int8.pt"
    meta_path = output_dir / f"postfilter_{tag}_best_meta.json"

    shadow = {name: tensor.detach().cpu().clone() for name, tensor in shadow_state.items()}
    torch.save(shadow, fp32_path)

    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(shadow)
    int8_size = save_int8(
        model,
        int8_path,
        meta=meta,
        per_channel=per_channel_int8,
        fp32_bias=per_channel_int8,
    )
    model.load_state_dict(original_state)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "epoch": int(epoch),
        "scorer": float(scorer),
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "int8_size": int(int8_size),
        "meta_path": str(meta_path),
        "meta": dict(meta),
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def save_final_artifacts(
    *,
    model: nn.Module,
    output_dir: Path,
    tag: str,
    meta: dict[str, Any],
    final_metrics: dict[str, float],
    best_eval_payload: dict[str, Any] | None = None,
    per_channel_int8: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"postfilter_{tag}_fp32.pt"
    int8_path = output_dir / f"postfilter_{tag}_int8.pt"
    final_meta_path = output_dir / f"postfilter_{tag}_final_meta.json"
    best_meta_path = output_dir / f"postfilter_{tag}_best_meta.json"

    torch.save(model.state_dict(), fp32_path)
    int8_size = save_int8(
        model,
        int8_path,
        meta=meta,
        per_channel=per_channel_int8,
        fp32_bias=per_channel_int8,
    )
    if best_eval_payload is not None and not best_meta_path.exists():
        best_meta_path.write_text(json.dumps(best_eval_payload, indent=2) + "\n")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "int8_size": int(int8_size),
        "meta": dict(meta),
        "best_eval": best_eval_payload,
        "best_meta_path": str(best_meta_path) if best_eval_payload is not None else None,
        "final_meta_path": str(final_meta_path),
        **final_metrics,
    }
    final_meta_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
