#!/usr/bin/env python3
"""Extract the canonical A1 latent table into ``extracted_frozen_latents.pt``.

The A1 archive's latents live inside the binary blob, decoded at inflate
time via ``codec.decode_latents_compact``. T1's frozen-A1 loader expects a
materialised tensor at
``experiments/results/A1_canonical/.../extracted_frozen_latents.pt``.

Usage
-----

    .venv/bin/python tools/extract_frozen_a1_latents.py

The tool resolves the canonical symlink, parses the archive blob, decodes
the latents using the **canonical archive's own codec.py** (not a copy),
and writes the result. Idempotent: re-running on an already-extracted
directory overwrites with the same bytes.

CLAUDE.md compliance
--------------------

- No /tmp paths.
- The output sha256 of the latent tensor is recorded in the saved dict so
  any drift relative to the designation memo is detectable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
A1_CANONICAL_RELATIVE = Path("experiments/results/A1_canonical")
LATENT_BLOB_LEN = 15_387


def _import_archive_codec(submission_src_dir: Path):
    """Import the archive's local codec.py module."""
    if not submission_src_dir.exists():
        raise SystemExit(f"submission_dir/src not found at {submission_src_dir}")
    spec = importlib.util.spec_from_file_location(
        "_a1_codec", submission_src_dir / "codec.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {submission_src_dir / 'codec.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_src_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(submission_src_dir) in sys.path:
            sys.path.remove(str(submission_src_dir))
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="Repository root (default: detected from this script).",
    )
    parser.add_argument(
        "--canonical-relpath", type=Path, default=A1_CANONICAL_RELATIVE,
        help="Relative path to the canonical A1 directory under repo root.",
    )
    parser.add_argument(
        "--output-relpath", default="harvested_artifacts/extracted_frozen_latents.pt",
        help="Output path RELATIVE to the canonical directory.",
    )
    args = parser.parse_args()

    canonical = (args.repo_root / args.canonical_relpath).resolve()
    archive_path = canonical / "harvested_artifacts" / "finetuned_archive" / "archive.zip"
    submission_src = (
        canonical / "harvested_artifacts" / "finetuned_archive" / "submission_dir" / "src"
    )
    output_path = canonical / args.output_relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise SystemExit(f"archive not found: {archive_path}")
    archive_bytes = archive_path.read_bytes()
    sha = hashlib.sha256(archive_bytes).hexdigest()

    a1_codec = _import_archive_codec(submission_src)

    with zipfile.ZipFile(archive_path) as zf:
        body = zf.read("x")
    section_total = struct.unpack_from("<I", body, 0)[0]
    latent_blob = body[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = body[section_total + LATENT_BLOB_LEN:]
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"latent blob length {len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )

    latents = a1_codec.decode_latents_compact(latent_blob)
    latents_full = a1_codec.apply_latent_sidecar(latents, sidecar_blob)

    payload = {
        "schema_version": 1,
        "latents": latents_full,
        "archive_sha256": sha,
        "archive_size_bytes": len(archive_bytes),
        "decoder_arch": "HNeRV-style",
        "extraction_tool": "tools/extract_frozen_a1_latents.py",
    }
    torch.save(payload, output_path)
    print(f"[extract_frozen_a1_latents] wrote {output_path}")
    print(f"  shape: {tuple(latents_full.shape)}; dtype: {latents_full.dtype}")
    print(f"  archive_sha256: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
