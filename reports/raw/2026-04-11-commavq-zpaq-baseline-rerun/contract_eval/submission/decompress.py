#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))
ZPAQ_BIN = "zpaq"

def decode_bytes(payload: bytes):
    return np.frombuffer(payload, dtype=np.int16).reshape(128, -1).T.reshape(-1, 8, 16)

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for payload in sorted(path for path in HERE.rglob("*") if path.is_file() and path.name.endswith(".zpaq")):
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([ZPAQ_BIN, "extract", str(payload), "-to", str(extract_dir)], check=True)
            candidates = sorted(path for path in extract_dir.rglob("*") if path.is_file())
            if not candidates:
                raise FileNotFoundError(f"no extracted file for {payload.name}")
            raw = candidates[0]
            rel = payload.relative_to(HERE).as_posix()[:-5]
            target = output_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.load(raw, allow_pickle=False)
            except Exception:
                with target.open("wb") as handle:
                    np.save(handle, decode_bytes(raw.read_bytes()))
            else:
                target.write_bytes(raw.read_bytes())

if __name__ == "__main__":
    main()
