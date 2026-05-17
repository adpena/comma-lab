#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit a planned master-gradient probe for packet-grammar validity."""
from __future__ import annotations

import argparse
import json

from tac.master_gradient_feasibility import audit_master_gradient_probe_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mutation-grain",
        required=True,
        choices=[
            "raw_archive_bit",
            "raw_archive_byte",
            "zip_member_payload_byte",
            "logical_section_parameter",
            "grammar_aware_operator",
            "repacked_archive_candidate",
        ],
    )
    parser.add_argument(
        "--axis-label",
        choices=["contest_cpu", "contest_cuda", "paired_contest_cpu_cuda", "diagnostic"],
    )
    parser.add_argument("--not-zip", action="store_true")
    parser.add_argument("--payload-not-entropy-coded", action="store_true")
    parser.add_argument("--updates-zip-headers", action="store_true")
    parser.add_argument("--updates-crc", action="store_true")
    parser.add_argument("--repacks-archive", action="store_true")
    parser.add_argument("--proves-inflate-success", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    verdict = audit_master_gradient_probe_plan(
        mutation_grain=args.mutation_grain,
        archive_is_zip=not args.not_zip,
        payload_is_entropy_coded=not args.payload_not_entropy_coded,
        updates_zip_headers=args.updates_zip_headers,
        updates_crc=args.updates_crc,
        repacks_archive=args.repacks_archive,
        proves_inflate_success=args.proves_inflate_success,
        axis_label=args.axis_label,
    )
    print(json.dumps(verdict.to_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
