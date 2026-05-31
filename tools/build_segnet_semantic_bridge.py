#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a source-vs-candidate SegNet semantic bridge artifact.

This is an analysis/optimization-input tool. It runs the contest SegNet over
source and inflated frames, compares scorer-grid argmax behavior, and emits a
false-authority JSON artifact for deterministic repair, postfilter, LoRA/DoRA,
selector-codec, and fleet-adaptable boundary-rule lanes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.segnet_boundary_marginals import logit_margin  # noqa: E402
from tac.analysis.segnet_semantic_bridge import (  # noqa: E402
    FALSE_AUTHORITY,
    GENERALIZATION_MODES,
    SegnetSemanticBridgeError,
    SemanticBridgeConfig,
    build_segnet_semantic_bridge,
    top2_class_indices,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflated-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument(
        "--video-names-file",
        type=Path,
        default=Path("upstream/public_test_video_names.txt"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--generalization-mode",
        choices=GENERALIZATION_MODES,
        default="mixed",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--boundary-dilation", type=int, default=5)
    parser.add_argument("--low-margin-threshold", type=float, default=1.0)
    parser.add_argument("--hinge-margin", type=float, default=0.25)
    parser.add_argument(
        "--pair-component-xray",
        type=Path,
        help="Optional pair_component_error_xray_v1 JSON to attach pose/seg context.",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        help="Optional PNG visualization for the highest-hinge sample.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_live_bridge(
    *,
    inflated_dir: Path,
    upstream_dir: Path,
    video_names_file: Path,
    candidate_id: str,
    generalization_mode: str,
    device_name: str,
    batch_size: int,
    max_pairs: int | None,
    num_threads: int,
    prefetch_queue_depth: int,
    boundary_dilation: int,
    low_margin_threshold: float,
    hinge_margin: float,
    pair_component_xray: Path | None,
    figure_out: Path | None,
    allow_overwrite: bool,
) -> dict[str, Any]:
    upstream_dir = _resolve(upstream_dir)
    inflated_dir = _resolve(inflated_dir)
    video_names_file = _resolve(video_names_file)
    if batch_size <= 0:
        raise SegnetSemanticBridgeError("batch_size must be positive")
    if max_pairs is not None and max_pairs <= 0:
        raise SegnetSemanticBridgeError("max_pairs must be positive when provided")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SegnetSemanticBridgeError("CUDA requested but torch.cuda.is_available() is false")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise SegnetSemanticBridgeError("MPS requested but torch.backends.mps.is_available() is false")

    prepend_paths(upstream_dir)
    from frame_utils import AVVideoDataset, DaliVideoDataset, TensorVideoDataset  # type: ignore
    from modules import SegNet, segnet_sd_path  # type: ignore
    from safetensors.torch import load_file  # type: ignore

    torch_device = torch.device(device_name)
    segnet = SegNet().eval().to(torch_device)
    segnet_model_path = upstream_dir / "models" / Path(segnet_sd_path).name
    segnet.load_state_dict(load_file(segnet_model_path, device=str(torch_device)))

    video_names = [
        line.strip()
        for line in video_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt_dataset_cls = DaliVideoDataset if torch_device.type == "cuda" else AVVideoDataset
    source_ds = gt_dataset_cls(
        video_names,
        data_dir=upstream_dir / "videos",
        batch_size=batch_size,
        device=torch_device if torch_device.type == "cuda" else torch.device("cpu"),
        num_threads=num_threads,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    candidate_ds = TensorVideoDataset(
        video_names,
        data_dir=inflated_dir,
        batch_size=batch_size,
        device=torch_device if torch_device.type == "cuda" else torch.device("cpu"),
        num_threads=num_threads,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    source_ds.prepare_data()
    candidate_ds.prepare_data()

    source_logits_parts: list[np.ndarray] = []
    candidate_logits_parts: list[np.ndarray] = []
    sample_ids: list[int] = []
    pair_base = 0
    with torch.inference_mode():
        for (_, _, source_batch), (_, _, candidate_batch) in zip(
            torch.utils.data.DataLoader(source_ds, batch_size=None, num_workers=0),
            torch.utils.data.DataLoader(candidate_ds, batch_size=None, num_workers=0),
            strict=True,
        ):
            if max_pairs is not None:
                remaining = max_pairs - len(sample_ids)
                if remaining <= 0:
                    break
                source_batch = source_batch[:remaining]
                candidate_batch = candidate_batch[:remaining]

            source_btchw = (
                source_batch.to(device=torch_device)
                .float()
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            candidate_btchw = (
                candidate_batch.to(device=torch_device)
                .float()
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            source_logits = segnet(segnet.preprocess_input(source_btchw))
            candidate_logits = segnet(segnet.preprocess_input(candidate_btchw))
            source_logits_parts.append(source_logits.detach().cpu().numpy())
            candidate_logits_parts.append(candidate_logits.detach().cpu().numpy())
            sample_ids.extend(range(pair_base, pair_base + int(source_logits.shape[0])))
            pair_base += int(source_logits.shape[0])
            if max_pairs is not None and len(sample_ids) >= max_pairs:
                break

    if not source_logits_parts:
        raise SegnetSemanticBridgeError("no pairs were loaded")
    source_logits_all = np.concatenate(source_logits_parts, axis=0)
    candidate_logits_all = np.concatenate(candidate_logits_parts, axis=0)
    pair_context = _load_pair_component_rows(pair_component_xray) if pair_component_xray else {}
    bridge = build_segnet_semantic_bridge(
        source_logits=source_logits_all,
        candidate_logits=candidate_logits_all,
        config=SemanticBridgeConfig(
            candidate_id=candidate_id,
            generalization_mode=generalization_mode,
            boundary_dilation=boundary_dilation,
            low_margin_threshold=low_margin_threshold,
            hinge_margin=hinge_margin,
            axis_tag=f"[{device_name} analysis; SegNet semantic bridge; no score authority]",
        ),
        sample_ids=sample_ids,
        pair_component_rows=pair_context,
    )
    bridge["source_artifacts"] = {
        "inflated_dir": str(inflated_dir),
        "upstream_dir": str(upstream_dir),
        "video_names_file": {
            "path": str(video_names_file),
            "sha256": sha256_file(video_names_file),
        },
        "segnet_safetensors": {
            "path": str(segnet_model_path),
            "sha256": sha256_file(segnet_model_path),
        },
        "pair_component_xray": (
            {
                "path": str(_resolve(pair_component_xray)),
                "sha256": sha256_file(_resolve(pair_component_xray)),
            }
            if pair_component_xray
            else None
        ),
    }
    if figure_out is not None:
        figure_path = _resolve(figure_out)
        _render_figure(
            bridge=bridge,
            source_logits=source_logits_all,
            candidate_logits=candidate_logits_all,
            path=figure_path,
            allow_overwrite=allow_overwrite,
        )
        bridge["visual_artifacts"] = {
            "highest_hinge_sample_figure": {
                "path": str(figure_path),
                "bytes": figure_path.stat().st_size,
                "sha256": sha256_file(figure_path),
            }
        }
    return bridge


def _load_pair_component_rows(path: Path) -> dict[int, Mapping[str, Any]]:
    resolved = _resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SegnetSemanticBridgeError("pair component xray must be a JSON object")
    if payload.get("schema") != "pair_component_error_xray_v1":
        raise SegnetSemanticBridgeError(
            "pair component xray must have schema pair_component_error_xray_v1"
        )
    rows: dict[int, Mapping[str, Any]] = {}
    for row in payload.get("rows") or []:
        if isinstance(row, Mapping) and "pair_idx" in row:
            rows[int(row["pair_idx"])] = row
    return rows


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _render_figure(
    *,
    bridge: Mapping[str, Any],
    source_logits: np.ndarray,
    candidate_logits: np.ndarray,
    path: Path,
    allow_overwrite: bool,
) -> None:
    if path.exists() and not allow_overwrite:
        raise SegnetSemanticBridgeError(f"refusing to overwrite existing figure: {path}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    source_labels = source_logits.argmax(axis=1).astype(np.int64)
    candidate_labels = candidate_logits.argmax(axis=1).astype(np.int64)
    source_top2 = top2_class_indices(source_logits)
    wrong = source_labels != candidate_labels
    margin_map = logit_margin(source_logits)
    sample_index = int(bridge["sample_rows"][0]["sample_index"])
    class_names = bridge["scorer_grid"]["class_names"]
    names = [class_names[idx] for idx in range(len(class_names))]
    cmap = ListedColormap(["#402020", "#ff0000", "#808060", "#00ff66", "#cc00ff"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    summary = bridge["summary"]
    fig.suptitle(
        "SegNet semantic bridge "
        f"{bridge['axis_tag']} | candidate={bridge['candidate_id']} | "
        f"argmax_disagreement={summary['argmax_disagreement_rate']:.6f}",
        fontsize=12,
    )
    axes[0, 0].imshow(source_labels[sample_index], cmap=cmap, vmin=0, vmax=4)
    axes[0, 0].set_title("source SegNet argmax")
    axes[0, 1].imshow(candidate_labels[sample_index], cmap=cmap, vmin=0, vmax=4)
    axes[0, 1].set_title("candidate SegNet argmax")
    axes[0, 2].imshow(wrong[sample_index], cmap="hot")
    axes[0, 2].set_title("argmax disagreement")
    for axis in axes[0]:
        axis.axis("off")

    confusion = np.asarray(bridge["confusion_matrix_source_to_candidate"], dtype=float)
    confusion_norm = confusion / max(float(confusion.sum()), 1.0)
    heat = axes[1, 0].imshow(confusion_norm, cmap="viridis")
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_yticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_yticklabels(names, fontsize=8)
    axes[1, 0].set_xlabel("candidate class")
    axes[1, 0].set_ylabel("source class")
    axes[1, 0].set_title("source -> candidate error mass")
    fig.colorbar(heat, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(margin_map[sample_index], cmap="magma")
    axes[1, 1].set_title("source top1-top2 margin")
    axes[1, 1].axis("off")

    top1_top2 = wrong[sample_index] & (
        candidate_labels[sample_index] == source_top2[sample_index]
    )
    out_of_pair = wrong[sample_index] & ~top1_top2
    axes[1, 2].bar(
        ["top1/top2", "out-of-pair"],
        [int(top1_top2.sum()), int(out_of_pair.sum())],
        color=["#1f77b4", "#ff7f0e"],
    )
    axes[1, 2].set_title(bridge["recommended_training"]["teacher_loss_verdict"])
    axes[1, 2].set_ylabel("pixels in highest-hinge sample")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        json_out = _resolve(args.json_out)
        bridge = build_live_bridge(
            inflated_dir=args.inflated_dir,
            upstream_dir=args.upstream_dir,
            video_names_file=args.video_names_file,
            candidate_id=args.candidate_id,
            generalization_mode=args.generalization_mode,
            device_name=args.device,
            batch_size=args.batch_size,
            max_pairs=args.max_pairs,
            num_threads=args.num_threads,
            prefetch_queue_depth=args.prefetch_queue_depth,
            boundary_dilation=args.boundary_dilation,
            low_margin_threshold=args.low_margin_threshold,
            hinge_margin=args.hinge_margin,
            pair_component_xray=args.pair_component_xray,
            figure_out=args.figure_out,
            allow_overwrite=bool(args.overwrite),
        )
        bridge = attach_tool_run_manifest(
            bridge,
            tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
            argv=sys.argv[1:],
            input_paths=[
                _resolve(args.video_names_file),
                *([_resolve(args.pair_component_xray)] if args.pair_component_xray else []),
            ],
            repo_root=REPO_ROOT,
            output_path=json_out,
        )
        expected_existing_sha256 = sha256_file(json_out) if json_out.exists() and args.overwrite else None
        write_result = write_json_artifact(
            json_out,
            bridge,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        SegnetSemanticBridgeError,
        ValueError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        print(f"FATAL: SegNet semantic bridge failed: {exc}", file=sys.stderr)
        return 2

    print(
        json_text(
            {
                "schema": "segnet_semantic_bridge_cli_result.v1",
                "json_out": str(args.json_out),
                "bytes_written": write_result.bytes_written,
                "sha256": write_result.sha256,
                "candidate_id": args.candidate_id,
                "generalization_mode": args.generalization_mode,
                "n_samples": bridge["summary"]["n_samples"],
                **FALSE_AUTHORITY,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
