#!/usr/bin/env python3
"""Validate Confucius's lgwin=18 PR101 byte-different candidate.

Pre-flight runtime-parity check on a CPU. Confirms:
  1. Re-encoding PR101's decoder_blob at brotli ``lgwin=18`` produces
     a DIFFERENT blob from the canonical PR101 default
  2. The new blob is the SAME size as the canonical (162,164 bytes)
  3. The new blob decodes via PR101's own ``decode_decoder_compact``
     to a state_dict that is bit-faithful with the canonical state_dict

Local parity is necessary but not sufficient for contest-CUDA dispatch.
This tool's PASS verdict means: the candidate is a real byte-different
PR101 archive that round-trips through PR101's stock inflate. It does
NOT prove the contest score equals PR101's; that requires CUDA T4
auth eval.

CLAUDE.md compliance:
  - Strict-scorer-rule: no scorer load.
  - Output to ``experiments/results/<lane>/`` per durable-state
    convention (NOT /tmp).
  - Tags any score claims ``[predicted-band only]``.

Usage::

    .venv/bin/python tools/validate_pr101_lgwin18_candidate.py \\
        --input-archive experiments/results/.../archive.zip \\
        --output-dir experiments/results/lgwin18_validation_<timestamp>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(input_archive: Path, output_dir: Path) -> dict:
    """Run the full lgwin=18 validation pipeline.

    Returns a structured report dict; also writes it to
    ``output_dir/lgwin18_validation_report.json``.
    """
    from pr101_archive_substitution_surgery import (
        _read_inner_blob,
        _split_pr101_inner_blob,
        substitute_decoder_blob,
    )
    from tac.pr101_split_brotli_codec import (
        decode_decoder_compact,
        encode_decoder_compact,
    )

    if not input_archive.is_file():
        raise SystemExit(f"input archive not found: {input_archive}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract canonical decoder_blob from PR101 archive
    inner = _read_inner_blob(input_archive)
    canonical_blob, canonical_latent, canonical_sidecar = (
        _split_pr101_inner_blob(inner)
    )
    canonical_state_dict = decode_decoder_compact(canonical_blob)
    canonical_blob_sha = _sha256(canonical_blob)

    # Step 2: re-encode with lgwin=18 (Confucius's finding)
    lgwin18_blob = encode_decoder_compact(
        canonical_state_dict,
        brotli_quality=11,
        brotli_lgwin=18,
    )
    lgwin18_blob_sha = _sha256(lgwin18_blob)

    # Step 3: verify bytes-different + same length
    bytes_different = canonical_blob != lgwin18_blob
    same_length = len(canonical_blob) == len(lgwin18_blob)

    # Step 4: roundtrip check — the lgwin=18 blob must decode
    # via PR101's stock decoder to a state_dict that is bit-faithful
    # with the canonical state_dict (both came from the same source).
    roundtripped_state_dict = decode_decoder_compact(lgwin18_blob)

    # Bit-faithful: every tensor in the canonical state_dict must equal
    # the roundtripped tensor (both came from int8 quantization grid).
    import torch
    parity_failures: list[str] = []
    for name, canonical_tensor in canonical_state_dict.items():
        if name not in roundtripped_state_dict:
            parity_failures.append(f"missing tensor {name}")
            continue
        rt_tensor = roundtripped_state_dict[name]
        if rt_tensor.shape != canonical_tensor.shape:
            parity_failures.append(
                f"shape mismatch for {name}: "
                f"{tuple(rt_tensor.shape)} vs {tuple(canonical_tensor.shape)}"
            )
            continue
        if not torch.equal(rt_tensor, canonical_tensor):
            parity_failures.append(
                f"tensor {name}: bit-faithful comparison FAILED"
            )
    parity_passed = not parity_failures

    # Step 5: produce a substituted archive at lgwin=18
    substituted_archive = output_dir / "archive_lgwin18.zip"
    substitution_report = substitute_decoder_blob(
        input_archive=input_archive,
        replacement_decoder_blob=lgwin18_blob,
        output_archive=substituted_archive,
    )

    # Step 6: assemble the final report
    report = {
        "tool": "tools/validate_pr101_lgwin18_candidate.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_archive": str(input_archive),
        "input_archive_sha256": _sha256(input_archive.read_bytes()),
        "output_dir": str(output_dir),
        "substituted_archive": str(substituted_archive),
        "substituted_archive_sha256": _sha256(substituted_archive.read_bytes()),
        "canonical_decoder_blob": {
            "len": len(canonical_blob),
            "sha256": canonical_blob_sha,
        },
        "lgwin18_decoder_blob": {
            "len": len(lgwin18_blob),
            "sha256": lgwin18_blob_sha,
        },
        "checks": {
            "bytes_different": bytes_different,
            "same_length": same_length,
            "decode_parity_passed": parity_passed,
            "decode_parity_failures": parity_failures,
        },
        "verdict": (
            "LOCAL_PASS" if (bytes_different and same_length and parity_passed)
            else "LOCAL_FAIL"
        ),
        "next_blocking_step": (
            "CUDA T4 inflate + upstream/evaluate.py on the substituted "
            "archive to prove contest-runtime parity. Local PASS is "
            "necessary but NOT sufficient for dispatch."
        ),
        "score_affecting_payload_changed": bytes_different,
        "charged_bits_changed": bytes_different,
        "evidence_grade": "[CPU-prep]",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
    }
    (output_dir / "lgwin18_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-archive", type=Path, required=True,
                        help="Path to a PR101 archive.zip")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where substituted archive + report land")
    args = parser.parse_args(argv)

    report = validate(args.input_archive, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "LOCAL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
