#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shipped ET4 overlay runtime.

Reads the counted ET4 payload, decodes the counted parent IX2 payload with the
vendored parent decoder, and applies the counted sparse frame_1 overlay.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from base_inflate_runner import Decoder as ParentDecoder  # noqa: E402
from ddm_et4_overlay_codec import (  # noqa: E402
    decode_overlay_payload,
    decode_patch_records,
    apply_patch_to_frame1,
)


class Decoder:
    def __init__(self, archive_dir: Path) -> None:
        payload_path = archive_dir / "0.bin"
        if payload_path.exists():
            payload = payload_path.read_bytes()
        else:
            with zipfile.ZipFile(archive_dir / "archive.zip") as zf:
                names = zf.namelist()
                if names != ["0.bin"]:
                    raise SystemExit(f"ET4 archive member list differs: {names}")
                payload = zf.read("0.bin")
        parent_payload, compressed_patch, metadata = decode_overlay_payload(payload)
        self.metadata = metadata
        self.parent_dir = archive_dir / "_et4_parent"
        self.parent_dir.mkdir(exist_ok=True)
        (self.parent_dir / "0.bin").write_bytes(parent_payload)
        self.parent = ParentDecoder(self.parent_dir)
        self.patches = decode_patch_records(compressed_patch)
        self.n_pairs = int(getattr(self.parent, "n_pairs"))

    def f1(self, pair: int) -> np.ndarray:
        base = self.parent.f1(int(pair))
        return apply_patch_to_frame1(base, self.patches.get(int(pair)))

    def f0(self, pair: int, _f1_u8: np.ndarray | None = None) -> np.ndarray:
        base_f1 = self.parent.f1(int(pair))
        return self.parent.f0(int(pair), base_f1)


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    dec = Decoder(archive_dir)
    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as sink:
        for pair in range(dec.n_pairs):
            f1 = dec.f1(pair)
            f0 = dec.f0(pair, f1)
            sink.write(f0.tobytes())
            sink.write(f1.tobytes())


if __name__ == "__main__":
    main()
