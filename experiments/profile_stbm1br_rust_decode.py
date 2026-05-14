#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile Rust STBM1BR decode parity against the Python reference.

This is a local CPU-only custody/timing tool. It does not dispatch remote work
or make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.stbm1br_mask_codec import decode_stbm1br_mask_segment  # noqa: E402


DEFAULT_SEGMENT = (
    REPO
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/mask_segment.stbm1br"
)
DEFAULT_OUTPUT_JSON = (
    REPO / "experiments/results/stbm1br_rust_decode_profile_20260504_codex/timing_report.json"
)
EXPECTED_SHAPE = (600, 384, 512)
EXPECTED_RENDER_SHA256 = "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _build_release_decoder() -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--quiet", "-p", "stbm1br-codec"],
        cwd=REPO / "runtime-rs",
        check=True,
        timeout=240,
    )
    decoder = REPO / "runtime-rs/target/release/stbm1br-codec"
    if not decoder.is_file():
        raise FileNotFoundError(f"Rust decoder build did not produce {decoder}")
    return decoder


def _time_python(segment: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    decoded = decode_stbm1br_mask_segment(segment, expected_shape=EXPECTED_SHAPE)
    elapsed = time.perf_counter() - started
    raw = decoded.tobytes()
    return {
        "implementation": "python_reference",
        "elapsed_seconds": elapsed,
        "decoded_bytes": len(raw),
        "render_order_sha256": _sha256_bytes(raw),
    }


def _time_rust(segment_path: Path, decoder: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stbm1br-profile-") as tmp:
        raw_path = Path(tmp) / "masks.raw"
        started = time.perf_counter()
        proc = subprocess.run(
            [
                str(decoder),
                "decode",
                str(segment_path),
                str(raw_path),
                "--expected-frames",
                str(EXPECTED_SHAPE[0]),
                "--expected-height",
                str(EXPECTED_SHAPE[1]),
                "--expected-width",
                str(EXPECTED_SHAPE[2]),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            raise RuntimeError(f"Rust decoder failed: {proc.stderr.strip() or proc.stdout.strip()}")
        raw = raw_path.read_bytes()
    return {
        "implementation": "rust_stbm1br_codec_release_cli",
        "elapsed_seconds": elapsed,
        "decoded_bytes": len(raw),
        "render_order_sha256": _sha256_bytes(raw),
        "stderr": proc.stderr.strip(),
        "decoder_path": _repo_rel(decoder),
        "decoder_sha256": _sha256_bytes(decoder.read_bytes()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, default=DEFAULT_SEGMENT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--decoder", type=Path, default=None)
    parser.add_argument("--skip-python", action="store_true")
    args = parser.parse_args(argv)

    segment_path = args.segment
    if not segment_path.is_file():
        raise FileNotFoundError(segment_path)
    segment = segment_path.read_bytes()
    decoder = args.decoder if args.decoder is not None else _build_release_decoder()
    rust = _time_rust(segment_path, decoder)
    python = None if args.skip_python else _time_python(segment)
    hashes = [rust["render_order_sha256"]]
    if python is not None:
        hashes.append(python["render_order_sha256"])
    parity_passed = all(h == EXPECTED_RENDER_SHA256 for h in hashes)
    report: dict[str, Any] = {
        "schema": "stbm1br_rust_decode_timing_report_v1",
        "score_claim": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "segment": {
            "path": _repo_rel(segment_path),
            "bytes": len(segment),
            "sha256": _sha256_bytes(segment),
        },
        "expected_shape": list(EXPECTED_SHAPE),
        "expected_render_order_sha256": EXPECTED_RENDER_SHA256,
        "rust": rust,
        "python_reference": python,
        "parity": {
            "passed": parity_passed,
            "hashes_equal_expected": parity_passed,
            "rust_matches_python": python is None
            or rust["render_order_sha256"] == python["render_order_sha256"],
        },
        "integration_gate": (
            "Exact inflate may set PACT_STBM1BR_RUST_DECODER only when the binary is "
            "part of the fixed runtime or deliberately bundled by the experiment; "
            "otherwise Python reference remains the default path."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if parity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
