#!/usr/bin/env python3
"""Extract HNeRV decoder weights from PR106 belt_and_suspenders archive (0.bin).

Reads the PR106 archive.zip → unpacks single-member 0.bin → parses the packed
fixed-schema layout via their own src/codec.py:parse_packed_archive (imported
directly so we never drift from the upstream parser) → writes the dequantized
torch state_dict .pt for downstream sensitivity-map and water-filling consumers.

Output:
    state_dict.pt  — dict[str, torch.Tensor (float32, dequantized)]
    metadata.json  — {archive_sha256, header, n_tensors, total_params, tensors[]}

CPU-only. No CUDA required. Anchor for Lane Ω-W-V3 (revival_plan_01_water_filling_codec_v2_pr106_decoder).

Usage:
    .venv/bin/python experiments/extract_pr106_decoder.py \\
        --archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\
        --out-dir experiments/results/sensitivity_map_pr106_20260504_claude/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

# Import PR106's own codec module directly. This sidesteps any drift between
# our re-implementation and the upstream parser. The intake snapshot is at:
PR106_SRC_PATH = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))

from codec import parse_packed_archive  # type: ignore[import-not-found]


def extract_pr106_decoder(
    archive: Path,
    out_dir: Path,
    *,
    verbose: bool = True,
) -> dict[str, object]:
    """Extract PR106 decoder tensors into ``out_dir`` and return metadata.

    The callable form keeps preflight and tests in-process while the CLI below
    remains the stable operator entrypoint.
    """
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_bytes = archive.read_bytes()
    sha = hashlib.sha256(archive_bytes).hexdigest()
    if verbose:
        print(f"[extract-pr106] archive: {archive} ({len(archive_bytes)} bytes, sha256={sha[:16]}...)")

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if names != ["0.bin"]:
            print(f"WARN: expected ['0.bin'] zip member, got {names}", file=sys.stderr)
        bin_bytes = z.read("0.bin")
    if verbose:
        print(f"[extract-pr106] 0.bin: {len(bin_bytes)} bytes")

    state_dict, latents, schema = parse_packed_archive(bin_bytes)
    n_tensors = len(state_dict)
    total_params = sum(t.numel() for t in state_dict.values())
    if verbose:
        print(f"[extract-pr106] decoded state_dict: {n_tensors} tensors, {total_params} params")
        print(f"[extract-pr106] schema: {schema}")
        print(f"[extract-pr106] latents shape: {tuple(latents.shape)}, dtype={latents.dtype}")
        for name, t in list(state_dict.items())[:5]:
            print(f"  {name}: shape={tuple(t.shape)}, dtype={t.dtype}, abs_max={t.abs().max().item():.4g}")
        if n_tensors > 5:
            print(f"  ... ({n_tensors - 5} more)")

    state_dict_path = out_dir / "state_dict.pt"
    torch.save(state_dict, state_dict_path)
    if verbose:
        print(f"[extract-pr106] wrote {state_dict_path} ({state_dict_path.stat().st_size} bytes)")

    latents_path = out_dir / "latents.pt"
    torch.save(latents, latents_path)
    if verbose:
        print(f"[extract-pr106] wrote {latents_path} ({latents_path.stat().st_size} bytes)")

    metadata = {
        "archive_path": str(archive),
        "archive_sha256": sha,
        "archive_size_bytes": len(archive_bytes),
        "bin_size_bytes": len(bin_bytes),
        "schema": schema,
        "n_tensors": n_tensors,
        "total_params": total_params,
        "latents_shape": list(latents.shape),
        "latents_dtype": str(latents.dtype),
        "tensors": [
            {"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
             "numel": t.numel(), "abs_max": float(t.abs().max())}
            for name, t in state_dict.items()
        ],
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if verbose:
        print(f"[extract-pr106] wrote {metadata_path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="PR106 archive.zip")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    try:
        extract_pr106_decoder(args.archive, args.out_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
