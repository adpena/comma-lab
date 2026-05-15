#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Streaming same-runtime PR101 selector frame parity proof.

This compares two PR101 frame-selector archives through one selected runtime,
hashing rendered uint8 frame bytes as they are produced. It does not write
`.raw` files and does not evaluate a score. Use `--max-pairs` only for prefix
smoke tests; omit it for full-frame parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA = "pr101_same_runtime_streaming_frame_parity_v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_single_member_payload(archive: Path, *, member_name: str | None = None) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {archive}, found {len(infos)}")
        info = infos[0]
        if member_name is not None and info.filename != member_name:
            raise ValueError(
                f"expected archive member {member_name!r} in {archive}, found {info.filename!r}"
            )
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise ValueError(f"unsafe archive member name: {info.filename!r}")
        return info.filename, zf.read(info.filename)


def load_runtime(runtime_dir: Path) -> ModuleType:
    runtime_dir = runtime_dir.resolve()
    inflate_py = runtime_dir / "inflate.py"
    if not inflate_py.is_file():
        raise FileNotFoundError(f"runtime inflate.py missing: {inflate_py}")
    spec = importlib.util.spec_from_file_location("pr101_selector_parity_runtime", inflate_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runtime from {inflate_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streaming_digest(
    runtime: ModuleType,
    member_payload: bytes,
    *,
    device: str,
    batch_pairs: int | None,
    max_pairs: int | None,
) -> dict[str, Any]:
    if batch_pairs is not None and batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be positive when set; got {batch_pairs}")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError(f"max_pairs must be positive when set; got {max_pairs}")
    if device == "cuda" and not runtime.torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")

    source_payload, selector_kind, selector_codes, selector_specs = (
        runtime.parse_pr101_frame_selector_archive(member_payload)
    )
    decoder_sd, latents, meta = runtime.parse_archive(source_payload)
    torch_device = runtime.torch.device(device)
    decoder = runtime.HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(torch_device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(torch_device)

    n_pairs_total = int(meta["n_pairs"])
    if len(selector_codes) != n_pairs_total:
        raise ValueError(
            f"selector has {len(selector_codes)} pairs; archive requires {n_pairs_total}"
        )
    n_pairs_hashed = min(n_pairs_total, max_pairs) if max_pairs is not None else n_pairs_total
    eval_h, eval_w = meta["eval_size"]
    camera_h = int(getattr(runtime, "CAMERA_H", 874))
    camera_w = int(getattr(runtime, "CAMERA_W", 1164))
    pair_batch = batch_pairs or 16

    sha = hashlib.sha256()
    total_frames = 0
    total_bytes = 0
    start = time.monotonic()
    with runtime.torch.inference_mode():
        for i in range(0, n_pairs_hashed, pair_batch):
            j = min(i + pair_batch, n_pairs_hashed)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = runtime.F.interpolate(
                flat,
                size=(camera_h, camera_w),
                mode="bicubic",
                align_corners=False,
            )
            up = up.reshape(batch, 2, 3, camera_h, camera_w)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            rounded = up.reshape(batch * 2, 3, camera_h, camera_w).clamp(0, 255).round()
            rounded = runtime.apply_pr101_selector_to_frames(
                rounded,
                selector_kind,
                selector_codes,
                selector_specs,
                pair_start=i,
            )
            frames = rounded.to(runtime.torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            payload = frames.tobytes()
            sha.update(payload)
            total_frames += int(frames.shape[0])
            total_bytes += len(payload)

    full_frame = max_pairs is None and n_pairs_hashed == n_pairs_total
    return {
        "schema": "pr101_runtime_full_frame_streaming_digest_v1",
        "selector_kind": selector_kind,
        "device": device,
        "batch_pairs": pair_batch,
        "max_pairs": max_pairs,
        "n_pairs_total": n_pairs_total,
        "n_pairs_hashed": n_pairs_hashed,
        "total_frames": total_frames,
        "total_bytes": total_bytes,
        "eval_size": [int(eval_h), int(eval_w)],
        "camera_size": [camera_h, camera_w],
        "full_frame_digest": full_frame,
        "streaming_raw_sha256": sha.hexdigest(),
        "elapsed_seconds": time.monotonic() - start,
        "score_claim": False,
    }


def prove_parity(
    *,
    source_archive: Path,
    candidate_archive: Path,
    runtime_dir: Path,
    member_name: str | None,
    device: str,
    batch_pairs: int | None,
    max_pairs: int | None,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    candidate_archive = candidate_archive.resolve()
    runtime_dir = runtime_dir.resolve()
    source_archive_bytes = source_archive.read_bytes()
    candidate_archive_bytes = candidate_archive.read_bytes()
    source_member_name, source_payload = read_single_member_payload(
        source_archive,
        member_name=member_name,
    )
    candidate_member_name, candidate_payload = read_single_member_payload(
        candidate_archive,
        member_name=member_name,
    )
    runtime = load_runtime(runtime_dir)
    source = streaming_digest(
        runtime,
        source_payload,
        device=device,
        batch_pairs=batch_pairs,
        max_pairs=max_pairs,
    )
    candidate = streaming_digest(
        runtime,
        candidate_payload,
        device=device,
        batch_pairs=batch_pairs,
        max_pairs=max_pairs,
    )
    same_hash = source["streaming_raw_sha256"] == candidate["streaming_raw_sha256"]
    same_bytes = source["total_bytes"] == candidate["total_bytes"]
    full_scope = bool(source["full_frame_digest"] and candidate["full_frame_digest"])
    return {
        "schema": SCHEMA,
        "proof_scope": (
            "same_runtime_streaming_full_frame_hash"
            if full_scope
            else "same_runtime_streaming_prefix_hash"
        ),
        "runtime_dir": runtime_dir.as_posix(),
        "runtime_inflate_py_sha256": sha256_bytes((runtime_dir / "inflate.py").read_bytes()),
        "source_archive": {
            "path": source_archive.as_posix(),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_bytes(source_archive_bytes),
            "member_name": source_member_name,
        },
        "candidate_archive": {
            "path": candidate_archive.as_posix(),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_bytes(candidate_archive_bytes),
            "member_name": candidate_member_name,
        },
        "source": source,
        "candidate": candidate,
        "streaming_output_sha256_equal": same_hash,
        "streaming_output_total_bytes_equal": same_bytes,
        "full_frame_inflate_output_parity_claim": full_scope and same_hash and same_bytes,
        "prefix_parity_claim": (not full_scope) and same_hash and same_bytes,
        "device_axis_label": f"local-{device}-streaming-runtime",
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "exact contest auth eval with archive/runtime custody and explicit "
            "[contest-CUDA]/[contest-CPU] axis labels"
            if full_scope and same_hash and same_bytes
            else "rerun without --max-pairs for full-frame parity before using parity language"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--member-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-pairs", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prove_parity(
        source_archive=args.source_archive,
        candidate_archive=args.candidate_archive,
        runtime_dir=args.runtime_dir,
        member_name=args.member_name,
        device=args.device,
        batch_pairs=args.batch_pairs,
        max_pairs=args.max_pairs,
    )
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.max_pairs is not None:
        return 0 if manifest.get("prefix_parity_claim") else 1
    return 0 if manifest.get("full_frame_inflate_output_parity_claim") else 1


if __name__ == "__main__":
    raise SystemExit(main())
