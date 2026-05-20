#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sparse visibility probe for PR110-compatible LFV1/HFV1 candidates.

This is a disk-light filter before full raw inflate and scorer evaluation.  It
decodes only the selected candidate pairs, seeks the matching frames in a
baseline raw output, and reports whether the candidate is uint8-visible on the
frames it is allowed to affect.

It is not a score claim and not a replacement for full inflate locality or
auth-eval custody.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

FRAME_HEIGHT = 874
FRAME_WIDTH = 1164
FRAME_CHANNELS = 3
FRAME_BYTES = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS


class SparseVisibilityError(ValueError):
    """Raised when sparse visibility probing cannot run safely."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tree_sha256(runtime_dir: Path) -> str:
    files = [runtime_dir / "inflate.py"]
    src_dir = runtime_dir / "src"
    if src_dir.is_dir():
        files.extend(sorted(path for path in src_dir.rglob("*.py") if path.is_file()))
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(runtime_dir).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(_sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def _load_runtime(runtime_dir: Path) -> ModuleType:
    inflate_py = runtime_dir / "inflate.py"
    src_dir = runtime_dir / "src"
    if not inflate_py.is_file():
        raise SparseVisibilityError(f"inflate.py not found: {inflate_py}")
    if not src_dir.is_dir():
        raise SparseVisibilityError(f"runtime src dir not found: {src_dir}")
    sys.path.insert(0, str(src_dir))
    module_name = "_pact_sparse_visibility_runtime_inflate"
    spec = importlib.util.spec_from_file_location(module_name, inflate_py)
    if spec is None or spec.loader is None:
        raise SparseVisibilityError(f"failed to load runtime module spec: {inflate_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)
    return module


def _baseline_frame(raw_path: Path, frame_index: int) -> bytes:
    if frame_index < 0:
        raise SparseVisibilityError(f"negative frame index: {frame_index}")
    with raw_path.open("rb") as handle:
        handle.seek(int(frame_index) * FRAME_BYTES)
        raw = handle.read(FRAME_BYTES)
    if len(raw) != FRAME_BYTES:
        raise SparseVisibilityError(
            f"baseline raw missing full frame {frame_index}: {len(raw)} bytes"
        )
    return raw


def _selected_pairs(manifest: dict[str, Any]) -> list[int]:
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise SparseVisibilityError("candidate manifest missing selection")
    pairs = selection.get("selected_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise SparseVisibilityError("candidate manifest missing selected_pairs")
    out = []
    for pair in pairs:
        if isinstance(pair, bool) or not isinstance(pair, int) or pair < 0:
            raise SparseVisibilityError(f"bad selected pair: {pair!r}")
        out.append(int(pair))
    return sorted(set(out))


def _candidate_src_bin(candidate_dir: Path, manifest: dict[str, Any]) -> Path:
    archive = manifest.get("archive") if isinstance(manifest.get("archive"), dict) else {}
    official_input = archive.get("official_inflate_input")
    if isinstance(official_input, str):
        path = Path(official_input).resolve() / "x"
        if path.is_file():
            return path
    path = candidate_dir / "archive_extracted_data_dir" / "x"
    if path.is_file():
        return path.resolve()
    sidecar = manifest.get("sidecar") if isinstance(manifest.get("sidecar"), dict) else {}
    sidecar_path = sidecar.get("path")
    if isinstance(sidecar_path, str):
        path = Path(sidecar_path).resolve().with_name("x")
        if path.is_file():
            return path
    path = candidate_dir / "data_dir" / "x"
    if path.is_file():
        return path.resolve()
    raise SparseVisibilityError(f"candidate data_dir/x not found under {candidate_dir}")


def _sidecar_path(src_bin: Path) -> Path | None:
    paths = [
        src_bin.with_name("foveation_params.bin"),
        src_bin.with_name("lapose_foveation_tuples.lfv1"),
    ]
    found = [path for path in paths if path.is_file()]
    if len(found) > 1:
        raise SparseVisibilityError(f"multiple foveation sidecars present next to {src_bin}")
    return found[0] if found else None


def _cache_key(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = args.candidate_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    baseline_raw = args.baseline_raw.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else candidate_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    src_bin = _candidate_src_bin(candidate_dir, manifest)
    sidecar = _sidecar_path(src_bin)
    baseline_sha256 = args.baseline_raw_sha256 or _sha256_file(baseline_raw)
    return {
        "schema": "lfv1_sparse_visibility_cache_key_v1",
        "candidate_id": manifest.get("candidate_id") or candidate_dir.name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "runtime_dir": str(runtime_dir),
        "runtime_tree_sha256": _runtime_tree_sha256(runtime_dir),
        "src_bin_path": str(src_bin),
        "src_bin_sha256": _sha256_file(src_bin),
        "sidecar_path": str(sidecar) if sidecar else None,
        "sidecar_sha256": _sha256_file(sidecar) if sidecar else None,
        "baseline_raw_path": str(baseline_raw),
        "baseline_raw_bytes": baseline_raw.stat().st_size,
        "baseline_raw_sha256": baseline_sha256,
        "device": args.device,
    }


def _decode_pairs(
    *,
    runtime: ModuleType,
    src_bin: Path,
    pairs: list[int],
    device_name: str,
) -> dict[int, bytes]:
    source_payload, selector_kind, selector_codes, selector_specs = (
        runtime.parse_pr101_frame_selector_archive(src_bin.read_bytes())
    )
    decoder_sd, latents, meta = runtime.parse_archive(source_payload)
    foveation_params = runtime.load_foveation_sidecar(src_bin)
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
    frame_bytes_by_index: dict[int, bytes] = {}
    selected_frames = {2 * pair for pair in pairs} | {2 * pair + 1 for pair in pairs}
    chunk_starts = sorted({(pair // 16) * 16 for pair in pairs})
    n_pairs = int(meta["n_pairs"])
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
            rounded = runtime.apply_hfv1_to_rounded_frames(
                rounded,
                foveation_params,
                frame_start=chunk_start * 2,
            )
            frames = rounded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for local_index in range(batch * 2):
                frame_index = chunk_start * 2 + local_index
                if frame_index in selected_frames:
                    frame_bytes_by_index[frame_index] = frames[local_index].tobytes()
    return frame_bytes_by_index


def _frame_diff(frame_index: int, baseline: bytes, candidate: bytes) -> dict[str, Any]:
    if len(candidate) != FRAME_BYTES:
        raise SparseVisibilityError(
            f"candidate frame {frame_index} has {len(candidate)} bytes, expected {FRAME_BYTES}"
        )
    left = np.frombuffer(baseline, dtype=np.uint8)
    right = np.frombuffer(candidate, dtype=np.uint8)
    delta = right.astype(np.int16) - left.astype(np.int16)
    abs_delta = np.abs(delta)
    changed = bool(np.any(delta))
    return {
        "frame_index": int(frame_index),
        "changed": changed,
        "baseline_sha256": _sha256_bytes(baseline),
        "candidate_sha256": _sha256_bytes(candidate),
        "changed_byte_count": int(np.count_nonzero(delta)),
        "sum_abs_delta": int(abs_delta.sum()),
        "max_abs_delta": int(abs_delta.max(initial=0)),
        "mean_abs_delta": float(abs_delta.mean()),
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    cache_key = _cache_key(args)
    candidate_dir = args.candidate_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    baseline_raw = args.baseline_raw.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else candidate_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    pairs = _selected_pairs(manifest)
    runtime = _load_runtime(runtime_dir)
    src_bin = _candidate_src_bin(candidate_dir, manifest)
    frame_bytes = _decode_pairs(
        runtime=runtime,
        src_bin=src_bin,
        pairs=pairs,
        device_name=args.device,
    )
    frame_rows = [
        _frame_diff(frame_index, _baseline_frame(baseline_raw, frame_index), raw)
        for frame_index, raw in sorted(frame_bytes.items())
    ]
    changed_frames = [row["frame_index"] for row in frame_rows if row["changed"]]
    expected_frames = sorted(frame for pair in pairs for frame in (2 * pair, 2 * pair + 1))
    archive = manifest.get("archive") if isinstance(manifest.get("archive"), dict) else {}
    payload = {
        "schema": "lfv1_sparse_visibility_probe_v1",
        "cache_key": cache_key,
        "candidate_id": manifest.get("candidate_id") or candidate_dir.name,
        "candidate_dir": str(candidate_dir),
        "manifest_path": str(manifest_path),
        "runtime_dir": str(runtime_dir),
        "baseline_raw": {
            "path": str(baseline_raw),
            "bytes": baseline_raw.stat().st_size,
            "sha256": cache_key["baseline_raw_sha256"],
        },
        "archive": {
            "path": archive.get("path"),
            "bytes": archive.get("bytes"),
            "sha256": archive.get("sha256"),
            "delta_bytes_vs_source_archive": archive.get("delta_bytes_vs_source_archive"),
        },
        "selected_pairs": pairs,
        "expected_changed_frame_indices": expected_frames,
        "changed_frame_indices": changed_frames,
        "uint8_visible": bool(changed_frames),
        "selected_frames_all_unchanged": not bool(changed_frames),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "sparse_probe_only": True,
        "full_raw_sha256": None,
        "full_inflate_locality_control": "not_run_sparse_filter",
        "frame_rows": frame_rows,
        "blockers": [
            "sparse_visibility_filter_not_full_inflate_control",
            "exact_cuda_auth_eval_missing",
        ],
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "--baseline-raw-sha256",
        help="Known SHA-256 for --baseline-raw. Avoids re-hashing large raw files in batch mode.",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reuse_cache and args.output.is_file():
        payload = _read_json(args.output)
        try:
            request_key = _cache_key(args)
        except SparseVisibilityError:
            raise
        if payload.get("cache_key") == request_key:
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "cache_hit": True,
                        "candidate_id": payload.get("candidate_id"),
                        "uint8_visible": payload.get("uint8_visible"),
                        "changed_frame_count": len(payload.get("changed_frame_indices") or []),
                        "score_claim": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    payload = probe(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cache_hit": False,
                "candidate_id": payload.get("candidate_id"),
                "uint8_visible": payload.get("uint8_visible"),
                "changed_frame_count": len(payload.get("changed_frame_indices") or []),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
