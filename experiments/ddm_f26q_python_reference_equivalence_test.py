#!/usr/bin/env python3
"""Verify retained native F26 evidence against the Python semantics oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
EXPECTED_TOKEN_BYTES = 117_964_800
EXPECTED_BIT_POSITION = 921_964
EXPECTED_LOGIT_SHA256 = "617e9fcfc967c200f1ecc8bea93dd45a22f7af2a050092f982169b5f5e5a3523"
EXPECTED_CDF_SHA256 = "ba0d529b7eaf6e16da1f62fc1cc7ca43ccc1b989356a68b8d37988088cb7c7ff"
DEFAULT_WORK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    run = json.loads((root / "receipts/native_run_v13_n600.json").read_text())
    report = run["native_report"]
    token_path = Path(report["decoded_token_path"])
    assert run["complete"] and all(run["gates"].values())
    assert token_path.stat().st_size == EXPECTED_TOKEN_BYTES
    assert _sha256(token_path) == EXPECTED_TOKEN_SHA256 == report["decoded_token_sha256"]
    assert report["decoder_bit_position"] == EXPECTED_BIT_POSITION
    assert report["corrected_quantized_logit_sha256"] == EXPECTED_LOGIT_SHA256
    assert report["corrected_cdf_input_sha256"] == EXPECTED_CDF_SHA256

    reference_prefix = root / "retained/profiles/reference_prefix_n32.u8"
    native_prefix = root / "retained/native_tokens_v13_n32.u8"
    assert reference_prefix.read_bytes() == native_prefix.read_bytes()

    build = json.loads((root / "receipts/native_build_v13.json").read_text())
    repeat_hashes = {item["sha256"] for item in build["repeat_builds"]}
    assert build["deterministic_binary"] and len(repeat_hashes) == 1
    print(
        json.dumps(
            {
                "full_field_parity": True,
                "prefix_byte_parity": True,
                "deterministic_repeat_build": True,
                "decoded_token_sha256": EXPECTED_TOKEN_SHA256,
                "decoder_bit_position": EXPECTED_BIT_POSITION,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
