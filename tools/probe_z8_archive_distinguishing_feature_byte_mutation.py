#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin CLI for the Z8HPC1 archive byte-mutation consumption proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.byte_mutation_proof import (
    Z8_BYTE_MUTATION_PROOF_SCHEMA,
    probe_z8_archive_distinguishing_feature,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--proof-out", type=Path)
    parser.add_argument(
        "--require-distinguishing-feature-consumed", action="store_true"
    )
    args = parser.parse_args(argv)
    manifest = probe_z8_archive_distinguishing_feature(
        args.archive, proof_out=args.proof_out
    )
    wavelet = manifest["sections"]["wavelet_blob"]
    print(
        f"[z8-byte-mutation] wavelet_blob verdict={wavelet.get('verdict')} "
        f"max_abs_pixel_delta={wavelet.get('max_abs_pixel_delta')}"
    )
    for section in ("decoder_blob", "indices_blob", "wyner_ziv_blob", "dreamer_state_blob"):
        verdict = manifest["sections"][section]
        print(
            f"[z8-byte-mutation] {section} verdict={verdict.get('verdict')} "
            "(custody unless PIXEL_CONSUMED)"
        )
    print(
        "[z8-byte-mutation] distinguishing_feature_consumed="
        f"{manifest['distinguishing_feature_consumed']}"
    )
    print(
        "[z8-byte-mutation] mamba_dreamer_wyner_ziv_pixel_consumption_proven="
        f"{manifest['mamba_dreamer_wyner_ziv_pixel_consumption_proven']}"
    )
    if args.proof_out is not None:
        print(f"[z8-byte-mutation] proof={args.proof_out}")
    if (
        args.require_distinguishing_feature_consumed
        and not manifest["distinguishing_feature_consumed"]
    ):
        return 1
    return 0


__all__ = [
    "Z8_BYTE_MUTATION_PROOF_SCHEMA",
    "main",
    "probe_z8_archive_distinguishing_feature",
]


if __name__ == "__main__":
    raise SystemExit(main())
