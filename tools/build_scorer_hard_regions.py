#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local false-authority scorer hard-region records from receiver replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.receiver_replay_scorer_hard_regions import (  # noqa: E402
    build_segnet_argmax_arrays_from_cache_dirs,
    infer_cache_dirs_from_mlx_response,
    load_argmax_array,
    load_component_vector,
    load_component_vectors_from_dir,
    load_component_vectors_from_mlx_response,
    load_pair_indices,
    write_hard_region_recon_pixel_weight_artifact,
    write_receiver_replay_scorer_hard_region_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-argmax", type=Path, help="Candidate SegNet argmax .npy or JSON array.")
    parser.add_argument("--reference-argmax", type=Path, help="Reference SegNet argmax .npy or JSON array.")
    parser.add_argument("--candidate-cache-dir", type=Path, help="Candidate scorer-input cache dir.")
    parser.add_argument("--reference-cache-dir", type=Path, help="Reference scorer-input cache dir.")
    parser.add_argument("--upstream-dir", type=Path, help="Upstream scorer repo for deriving argmax from cache dirs.")
    parser.add_argument("--mlx-response", type=Path, help="MLX scorer-response JSON or HiNeRV response wrapper.")
    parser.add_argument("--components-dir", type=Path, help="Directory with posenet_distortion.npy/segnet_distortion.npy.")
    parser.add_argument("--posenet-distortion", type=Path, help="Explicit per-pair PoseNet distortion vector.")
    parser.add_argument("--segnet-distortion", type=Path, help="Explicit per-pair SegNet distortion vector.")
    parser.add_argument("--pair-indices", type=Path, help="Explicit pair_indices .npy or JSON.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", default="receiver_replay_scorer_hard_regions")
    parser.add_argument("--sample-pairs", type=int)
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-components", type=int, default=32)
    parser.add_argument("--include-solved-confusions", action="store_true")
    parser.add_argument(
        "--recon-weight-output-dir",
        type=Path,
        help=(
            "Optional output directory for a receiver-hard-region "
            "recon_pixel_weight NPZ/manifest consumable via "
            "--recon-pixel-weight-path."
        ),
    )
    parser.add_argument("--recon-weight-height", type=int, default=384)
    parser.add_argument("--recon-weight-width", type=int, default=512)
    parser.add_argument("--recon-weight-frame-index", type=int, default=1)
    parser.add_argument("--recon-weight-base", type=float, default=1.0)
    parser.add_argument("--recon-weight-score-gain", type=float, default=2.0)
    parser.add_argument("--recon-weight-component-gain", type=float, default=1.0)
    parser.add_argument("--recon-weight-normalize", choices=("mean", "none"), default="mean")
    parser.add_argument("--overwrite-recon-weight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_json = _resolve(args.output_json)
    source_artifacts: dict[str, object] = {}

    candidate_argmax = None
    reference_argmax = None
    pair_indices = None
    if args.candidate_argmax or args.reference_argmax:
        if not args.candidate_argmax or not args.reference_argmax:
            raise SystemExit("--candidate-argmax and --reference-argmax must be supplied together")
        candidate_argmax = load_argmax_array(_resolve(args.candidate_argmax))
        reference_argmax = load_argmax_array(_resolve(args.reference_argmax))
        source_artifacts.update(
            {
                "candidate_argmax_path": _resolve(args.candidate_argmax).as_posix(),
                "reference_argmax_path": _resolve(args.reference_argmax).as_posix(),
            }
        )
        if args.pair_indices:
            pair_indices = load_pair_indices(_resolve(args.pair_indices))

    posenet_distortion, segnet_distortion, component_sources = _load_component_context(args)
    source_artifacts.update(component_sources)

    candidate_cache_dir = _resolve(args.candidate_cache_dir) if args.candidate_cache_dir else None
    reference_cache_dir = _resolve(args.reference_cache_dir) if args.reference_cache_dir else None
    if args.mlx_response and (candidate_cache_dir is None or reference_cache_dir is None):
        inferred_candidate, inferred_reference = infer_cache_dirs_from_mlx_response(_resolve(args.mlx_response))
        candidate_cache_dir = candidate_cache_dir or inferred_candidate
        reference_cache_dir = reference_cache_dir or inferred_reference

    if candidate_argmax is None or reference_argmax is None:
        if candidate_cache_dir is None or reference_cache_dir is None:
            raise SystemExit(
                "supply direct --candidate-argmax/--reference-argmax or "
                "--candidate-cache-dir/--reference-cache-dir with --upstream-dir"
            )
        if args.upstream_dir is None:
            raise SystemExit("--upstream-dir is required when deriving argmax from scorer-input caches")
        candidate_argmax, reference_argmax, pair_indices, cache_sources = build_segnet_argmax_arrays_from_cache_dirs(
            candidate_cache_dir=candidate_cache_dir,
            reference_cache_dir=reference_cache_dir,
            upstream_dir=_resolve(args.upstream_dir),
            sample_pairs=args.sample_pairs,
            batch_frames=int(args.batch_frames),
            device=str(args.device),
        )
        source_artifacts.update(cache_sources)
    elif pair_indices is None and candidate_cache_dir is not None:
        candidate_pair_indices = candidate_cache_dir / "pair_indices.npy"
        if candidate_pair_indices.is_file():
            pair_indices = load_pair_indices(candidate_pair_indices)
            if args.sample_pairs is not None:
                pair_indices = pair_indices[: int(args.sample_pairs)]
            source_artifacts["pair_indices_path"] = candidate_pair_indices.as_posix()

    payload = write_receiver_replay_scorer_hard_region_report(
        output_json=output_json,
        candidate_argmax=candidate_argmax,
        reference_argmax=reference_argmax,
        pair_indices=pair_indices,
        posenet_distortion=posenet_distortion,
        segnet_distortion=segnet_distortion,
        label=str(args.label),
        top_components=int(args.top_components),
        include_solved_confusions=bool(args.include_solved_confusions),
        source_artifacts=source_artifacts,
    )
    recon_weight_manifest = None
    if args.recon_weight_output_dir:
        recon_weight_manifest = write_hard_region_recon_pixel_weight_artifact(
            report=payload,
            output_dir=_resolve(args.recon_weight_output_dir),
            output_height=int(args.recon_weight_height),
            output_width=int(args.recon_weight_width),
            pair_count=payload["pair_count"],
            frame_index=int(args.recon_weight_frame_index),
            base_weight=float(args.recon_weight_base),
            score_gain=float(args.recon_weight_score_gain),
            component_gain=float(args.recon_weight_component_gain),
            normalize=str(args.recon_weight_normalize),
            allow_overwrite=bool(args.overwrite_recon_weight),
        )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output_json": output_json.as_posix(),
                "pair_count": payload["pair_count"],
                "argmax_disagreement_rate": payload["argmax_disagreement_rate"],
                "segnet_score_contribution": payload["segnet_score_contribution"],
                "hard_region_count": len(payload["hard_region_records"]),
                "top_connected_component_count": len(payload["top_connected_components"]),
                "recon_weight_manifest_path": (
                    None
                    if recon_weight_manifest is None
                    else recon_weight_manifest["manifest_path"]
                ),
                "recon_weight_path": (
                    None
                    if recon_weight_manifest is None
                    else recon_weight_manifest["weight_path"]
                ),
                "score_claim": payload["score_claim"],
                "ready_for_exact_eval_dispatch": payload["ready_for_exact_eval_dispatch"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_component_context(args: argparse.Namespace) -> tuple[object | None, object | None, dict[str, object]]:
    source: dict[str, object] = {}
    pose = None
    seg = None
    if args.mlx_response:
        pose, seg, response_source = load_component_vectors_from_mlx_response(_resolve(args.mlx_response))
        source.update(response_source)
    if args.components_dir:
        dir_pose, dir_seg = load_component_vectors_from_dir(_resolve(args.components_dir))
        pose = dir_pose if dir_pose is not None else pose
        seg = dir_seg if dir_seg is not None else seg
        source["components_dir"] = _resolve(args.components_dir).as_posix()
    if args.posenet_distortion:
        pose = load_component_vector(_resolve(args.posenet_distortion), label="posenet_distortion")
        source["posenet_distortion_path"] = _resolve(args.posenet_distortion).as_posix()
    if args.segnet_distortion:
        seg = load_component_vector(_resolve(args.segnet_distortion), label="segnet_distortion")
        source["segnet_distortion_path"] = _resolve(args.segnet_distortion).as_posix()
    return pose, seg, source


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
