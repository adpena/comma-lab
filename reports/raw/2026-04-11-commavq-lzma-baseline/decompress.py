#!/usr/bin/env python3
import os
import lzma
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))

def decompress_bytes(x: bytes):
    return np.frombuffer(lzma.decompress(x), dtype=np.int16).reshape(128, -1).T.reshape(-1, 8, 16)

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for payload in sorted(path for path in HERE.iterdir() if path.is_file() and path.name != "decompress.py"):
        tokens = decompress_bytes(payload.read_bytes())
        with (output_dir / payload.name).open("wb") as handle:
            np.save(handle, tokens)

if __name__ == "__main__":
    main()
