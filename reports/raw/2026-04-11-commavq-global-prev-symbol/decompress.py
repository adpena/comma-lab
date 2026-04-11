#!/usr/bin/env python3
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.cwd()))
ENCODED_DIR = HERE if (HERE / "manifest.json").exists() else Path.cwd()

from _lossless_global_prev_symbol_runtime import decode_corpus_global_prev_symbol_position_major

output_dir = Path(os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed"))

def main():
    decode_corpus_global_prev_symbol_position_major(encoded_dir=ENCODED_DIR, output_dir=output_dir)

if __name__ == "__main__":
    main()
