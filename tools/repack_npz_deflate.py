"""Lossless in-place npz repack: STORED members -> DEFLATED, arrays byte-identical.

The qbt1 verdict payloads were written with np.savez (zip method=0, no
compression); real camera/logits payloads deflate 2.07x (measured on
governed_n32_r4 verdict_0001). This tool recompresses each npz IN PLACE on the
same volume, refusing to replace a file unless every array round-trips
byte-identically, and appends a machine-readable manifest row per file
(certify-or-block: original sha256+bytes, new sha256+bytes, per-array sha256).

Usage:
    repack_npz_deflate.py --root <dir> [--apply] [--manifest <path>]

Without --apply it only reports the plan (files, bytes, projected savings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_stored_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (zipfile.BadZipFile, OSError):
        return False
    return bool(members) and any(m.compress_type == zipfile.ZIP_STORED for m in members)


def repack_one(path: Path) -> dict[str, object]:
    original = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name]) for name in loaded.files}
    array_facts = {
        name: {
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, value in arrays.items()
    }
    # NOTE: np.savez_compressed appends ".npz" to names lacking it — keep the
    # temp name .npz-suffixed so the written path is exactly `temporary`.
    temporary = path.with_name(path.stem + ".repacktmp.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as reread:
        if set(reread.files) != set(arrays):
            temporary.unlink()
            raise RuntimeError(f"repack key drift: {path}")
        for name in reread.files:
            value = np.array(reread[name])
            if (
                hashlib.sha256(value.tobytes()).hexdigest() != array_facts[name]["sha256"]
                or str(value.dtype) != array_facts[name]["dtype"]
                or list(value.shape) != array_facts[name]["shape"]
            ):
                temporary.unlink()
                raise RuntimeError(f"repack array drift: {path}::{name}")
    os.replace(temporary, path)
    return {
        "schema": "npz_deflate_repack.v1",
        "original": original,
        "repacked": {"bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "arrays": array_facts,
        "verified": "per_array_bytes_dtype_shape_identical",
        "repacked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    targets = sorted(
        p for p in args.root.rglob("*.npz")
        if not p.name.startswith("._")
        and not p.name.endswith(".repacktmp.npz")
        and is_stored_npz(p)
    )
    total = sum(p.stat().st_size for p in targets)
    print(f"stored-uncompressed npz under {args.root}: {len(targets)} files, {total/1e9:.2f} GB")
    if not args.apply:
        print("plan only; re-run with --apply to repack in place")
        return 0

    manifest = args.manifest or (args.root / "npz_deflate_repack_manifest.jsonl")
    saved = 0
    with manifest.open("a", encoding="utf-8") as ledger:
        for index, path in enumerate(targets, 1):
            row = repack_one(path)
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
            ledger.flush()
            saved += int(row["original"]["bytes"]) - int(row["repacked"]["bytes"])  # type: ignore[index]
            print(f"[{index}/{len(targets)}] {path.name}: "
                  f"{int(row['original']['bytes'])/1e6:.0f} -> {int(row['repacked']['bytes'])/1e6:.0f} MB "  # type: ignore[index]
                  f"(cumulative saved {saved/1e9:.2f} GB)", flush=True)
    print(f"DONE: {len(targets)} files repacked, {saved/1e9:.2f} GB reclaimed; manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
