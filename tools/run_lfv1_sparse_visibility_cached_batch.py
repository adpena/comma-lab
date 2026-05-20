#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cached in-process LFV1 sparse visibility probe for materialized candidates.

This is the fast second-stage filter after candidate archives have already been
built. It loads the runtime once, decodes each required PR110 chunk once, then
applies many LFV1 sidecars against the cached rounded frames.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.probe_lfv1_sparse_visibility import (  # noqa: E402
    FRAME_BYTES,
    SparseVisibilityError,
    _baseline_frame,
    _candidate_src_bin,
    _frame_diff,
    _load_runtime,
    _read_json,
    _runtime_tree_sha256,
    _selected_pairs,
    _sha256_file,
    _sidecar_path,
    _write_json,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    src_bin: Path
    sidecar_path: Path | None
    selected_pairs: list[int]


def _candidate_dirs(args: argparse.Namespace) -> list[Path]:
    if args.sparse_batch_manifest is not None:
        batch = _read_json(args.sparse_batch_manifest)
        paths = []
        for result in batch.get("results", []):
            if not isinstance(result, dict) or result.get("returncode") != 0:
                continue
            candidate_dir = result.get("candidate_dir")
            if isinstance(candidate_dir, str):
                paths.append(Path(candidate_dir))
        return sorted(path.resolve() for path in paths)
    root = args.candidate_root.resolve()
    if not root.is_dir():
        raise SparseVisibilityError(f"--candidate-root not found: {root}")
    return sorted(path for path in root.iterdir() if (path / "manifest.json").is_file())


def _load_specs(args: argparse.Namespace) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for candidate_dir in _candidate_dirs(args):
        manifest_path = candidate_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        candidate_id = str(manifest.get("candidate_id") or candidate_dir.name)
        src_bin = _candidate_src_bin(candidate_dir, manifest)
        specs.append(
            CandidateSpec(
                candidate_id=candidate_id,
                candidate_dir=candidate_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                src_bin=src_bin,
                sidecar_path=_sidecar_path(src_bin),
                selected_pairs=_selected_pairs(manifest),
            )
        )
    if args.max_candidates > 0:
        specs = specs[: args.max_candidates]
    if not specs:
        raise SparseVisibilityError("no candidate manifests selected")
    return specs


def _decode_base_chunks(
    *,
    runtime: Any,
    src_bin: Path,
    chunk_starts: list[int],
    device_name: str,
) -> tuple[dict[int, torch.Tensor], int, dict[str, Any]]:
    source_payload, selector_kind, selector_codes, selector_specs = (
        runtime.parse_pr101_frame_selector_archive(src_bin.read_bytes())
    )
    decoder_sd, latents, meta = runtime.parse_archive(source_payload)
    device = torch.device(device_name)
    decoder = runtime.HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)
    eval_h, eval_w = meta["eval_size"]
    n_pairs = int(meta["n_pairs"])
    if len(selector_codes) != n_pairs:
        raise SparseVisibilityError(
            f"selector has {len(selector_codes)} pairs; archive requires exactly {n_pairs}"
        )
    chunks: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + 16, n_pairs)
            batch = chunk_end - chunk_start
            decoded = decoder(latents[chunk_start:chunk_end])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = runtime.F.interpolate(
                flat,
                size=(runtime.CAMERA_H, runtime.CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            up = up.reshape(batch, 2, 3, runtime.CAMERA_H, runtime.CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            rounded = up.reshape(batch * 2, 3, runtime.CAMERA_H, runtime.CAMERA_W).clamp(0, 255).round()
            rounded = runtime.apply_pr101_selector_to_frames(
                rounded,
                selector_kind,
                selector_codes,
                selector_specs,
                pair_start=chunk_start,
            )
            chunks[chunk_start] = rounded
    return chunks, n_pairs, meta


def _probe_one(
    *,
    runtime: Any,
    spec: CandidateSpec,
    base_chunks: dict[int, torch.Tensor],
    n_pairs: int,
    baseline_raw: Path,
    baseline_raw_sha256: str,
    runtime_dir: Path,
    runtime_tree_sha256: str,
) -> dict[str, Any]:
    foveation_params = runtime.load_foveation_sidecar(spec.src_bin)
    if hasattr(runtime, "validate_foveation_sidecar"):
        runtime.validate_foveation_sidecar(foveation_params, n_pairs=n_pairs)
    frame_rows = []
    selected_frames = {
        frame for pair in spec.selected_pairs for frame in (2 * pair, 2 * pair + 1)
    }
    with torch.inference_mode():
        for chunk_start in sorted({(pair // 16) * 16 for pair in spec.selected_pairs}):
            rounded = runtime.apply_hfv1_to_rounded_frames(
                base_chunks[chunk_start],
                foveation_params,
                frame_start=chunk_start * 2,
            )
            frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for local_index in range(int(frames.shape[0])):
                frame_index = chunk_start * 2 + local_index
                if frame_index not in selected_frames:
                    continue
                frame_rows.append(
                    _frame_diff(
                        frame_index,
                        _baseline_frame(baseline_raw, frame_index),
                        frames[local_index].tobytes(),
                    )
                )
    changed_frames = [row["frame_index"] for row in frame_rows if row["changed"]]
    archive = spec.manifest.get("archive") if isinstance(spec.manifest.get("archive"), dict) else {}
    cache_key = {
        "schema": "lfv1_cached_sparse_visibility_key_v1",
        "candidate_id": spec.candidate_id,
        "manifest_sha256": _sha256_file(spec.manifest_path),
        "runtime_tree_sha256": runtime_tree_sha256,
        "src_bin_sha256": _sha256_file(spec.src_bin),
        "sidecar_sha256": _sha256_file(spec.sidecar_path) if spec.sidecar_path else None,
        "baseline_raw_sha256": baseline_raw_sha256,
    }
    return {
        "schema": "lfv1_sparse_visibility_probe_v1",
        "producer": "tools/run_lfv1_sparse_visibility_cached_batch.py",
        "cache_key": cache_key,
        "candidate_id": spec.candidate_id,
        "candidate_dir": str(spec.candidate_dir),
        "manifest_path": str(spec.manifest_path),
        "runtime_dir": str(runtime_dir),
        "baseline_raw": {
            "path": str(baseline_raw),
            "bytes": baseline_raw.stat().st_size,
            "sha256": baseline_raw_sha256,
        },
        "archive": {
            "path": archive.get("path"),
            "bytes": archive.get("bytes"),
            "sha256": archive.get("sha256"),
            "delta_bytes_vs_source_archive": archive.get("delta_bytes_vs_source_archive"),
        },
        "selected_pairs": spec.selected_pairs,
        "expected_changed_frame_indices": sorted(selected_frames),
        "changed_frame_indices": changed_frames,
        "uint8_visible": bool(changed_frames),
        "selected_frames_all_unchanged": not bool(changed_frames),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "sparse_probe_only": True,
        "full_raw_sha256": None,
        "full_inflate_locality_control": "not_run_cached_sparse_filter",
        "frame_rows": frame_rows,
        "blockers": [
            "cached_sparse_visibility_filter_not_full_inflate_control",
            "exact_cuda_auth_eval_missing",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    specs = _load_specs(args)
    runtime_dir = args.runtime_dir.resolve()
    baseline_raw = args.baseline_raw.resolve()
    baseline_raw_sha256 = args.baseline_raw_sha256 or _sha256_file(baseline_raw)
    runtime_tree_sha256 = _runtime_tree_sha256(runtime_dir)
    runtime = _load_runtime(runtime_dir)
    chunk_starts = sorted(
        {
            (pair // 16) * 16
            for spec in specs
            for pair in spec.selected_pairs
        }
    )
    src_bins_by_sha256: dict[str, Path] = {}
    for spec in specs:
        src_bins_by_sha256.setdefault(_sha256_file(spec.src_bin), spec.src_bin)
    if len(src_bins_by_sha256) != 1:
        raise SparseVisibilityError(
            "cached batch currently requires one shared source x member payload"
        )
    src_bin = next(iter(src_bins_by_sha256.values()))
    base_chunks, n_pairs, _meta = _decode_base_chunks(
        runtime=runtime,
        src_bin=src_bin,
        chunk_starts=chunk_starts,
        device_name=args.device,
    )
    results = []
    for spec in specs:
        visibility = _probe_one(
            runtime=runtime,
            spec=spec,
            base_chunks=base_chunks,
            n_pairs=n_pairs,
            baseline_raw=baseline_raw,
            baseline_raw_sha256=baseline_raw_sha256,
            runtime_dir=runtime_dir,
            runtime_tree_sha256=runtime_tree_sha256,
        )
        output_path = args.output_root / "sparse_visibility" / spec.candidate_id / "visibility.json"
        _write_json(output_path, visibility)
        results.append(
            {
                "candidate_id": spec.candidate_id,
                "returncode": 0,
                "candidate_dir": str(spec.candidate_dir),
                "manifest_path": str(spec.manifest_path),
                "visibility_path": str(output_path),
                "uint8_visible": bool(visibility["uint8_visible"]),
                "changed_frame_count": len(visibility["changed_frame_indices"]),
                "archive_delta_bytes": visibility["archive"]["delta_bytes_vs_source_archive"],
            }
        )
    elapsed = time.monotonic() - start
    visible = [row for row in results if row["uint8_visible"]]
    first_visible = sorted(
        visible,
        key=lambda row: (
            int(row["archive_delta_bytes"] or 0),
            int(row["changed_frame_count"] or 0),
            str(row["candidate_id"]),
        ),
    )
    payload = {
        "schema": "lfv1_cached_sparse_visibility_batch_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_count": len(specs),
        "success_count": len(results),
        "failure_count": 0,
        "visible_count": len(visible),
        "elapsed_seconds": elapsed,
        "chunks_decoded": chunk_starts,
        "runtime_dir": str(runtime_dir),
        "runtime_tree_sha256": runtime_tree_sha256,
        "baseline_raw": str(baseline_raw),
        "baseline_raw_sha256": baseline_raw_sha256,
        "output_root": str(args.output_root),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "results": results,
        "first_visible": first_visible[0] if first_visible else None,
    }
    _write_json(args.output_root / "sparse_batch_manifest.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate-root", type=Path)
    group.add_argument("--sparse-batch-manifest", type=Path)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--baseline-raw-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-candidates", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "manifest": str(Path(payload["output_root"]) / "sparse_batch_manifest.json"),
                "candidate_count": payload["candidate_count"],
                "visible_count": payload["visible_count"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "chunks_decoded": payload["chunks_decoded"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
