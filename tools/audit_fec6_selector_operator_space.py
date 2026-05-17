#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit grammar-aware FEC6 selector operator rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from tac.fec6_selector_operator_space import build_fec6_selector_operator_space
from tac.repo_io import write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_PAIR_ROWS = (
    REPO_ROOT
    / "experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top16_topmodes_v2_codex/pair_component_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/fec6_selector_operator_space_20260517_codex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec6-archive", type=Path, default=DEFAULT_FEC6_ARCHIVE)
    parser.add_argument(
        "--pair-component-rows",
        action="append",
        type=Path,
        default=None,
        help="JSONL pair/mode component table. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()

    pair_rows = tuple(args.pair_component_rows or [DEFAULT_PAIR_ROWS])
    manifest = build_fec6_selector_operator_space(
        fec6_archive=args.fec6_archive,
        pair_component_rows_paths=pair_rows,
        repo_root=REPO_ROOT,
        max_rows=args.max_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "operator_space_manifest.json", manifest)
    (args.output_dir / "operator_space.md").write_text(_render_markdown(manifest), encoding="utf-8")
    print(args.output_dir / "operator_space_manifest.json")
    return 0


def _render_markdown(manifest: dict[str, object]) -> str:
    threshold = manifest["score_threshold"]
    selector = manifest["source_archive"]
    entropy = manifest["selector_entropy"]
    conclusion = manifest["conclusion"]
    rows = manifest["proxy_and_nonpositive_bit_rows"]
    lines = [
        "# FEC6 Selector Operator Space Audit",
        "",
        "score_claim: false",
        "promotion_eligible: false",
        "ready_for_exact_eval_dispatch: false",
        "dispatch_attempted: false",
        "",
        "## Source",
        "",
        f"- archive: `{selector['archive_path']}`",
        f"- archive bytes: `{selector['archive_bytes']}`",
        f"- archive sha256: `{selector['archive_sha256']}`",
        f"- current CPU score: `{threshold['current_cpu_score']}`",
        f"- target CPU score: `{threshold['target_cpu_score']}`",
        "- required charged bytes to cross target if components are unchanged: "
        f"`{threshold['required_rate_bytes_to_strictly_cross_target_if_components_unchanged']}`",
        "",
        "## Selector Entropy",
        "",
        f"- selector payload bytes: `{entropy['selector_payload_bytes']}`",
        f"- selector index bytes: `{entropy['selector_index_bytes']}`",
        f"- selector code bits: `{entropy['selector_code_bits_total']}`",
        f"- zero-header entropy floor bytes: `{entropy['zero_header_entropy_floor_bytes']}`",
        "- payload gap to zero-header entropy floor: "
        f"`{entropy['gap_payload_to_zero_header_entropy_floor_bytes']}`",
        "",
        "## Verdict",
        "",
        "- same-runtime byte-only selector polish blocked: "
        f"`{conclusion['same_runtime_byte_only_selector_polish_blocked']}`",
        f"- reason: {conclusion['same_runtime_selector_reason']}",
        f"- next packet operator: {conclusion['next_packet_operator']}",
        "",
        "## Best Proxy-Improving, Nonpositive-Bit Rows",
        "",
        "| operator | pair | current | candidate | bit delta | byte delta | proxy delta |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows[:10]:
        lines.append(
            "| `{operator}` | {pair} | `{current}` | `{candidate}` | {bits} | {bytes_} | {delta:.12g} |".format(
                operator=row["operator_id"],
                pair=row["pair"],
                current=row["current_mode_id"],
                candidate=row["candidate_mode_id"],
                bits=row["selector_code_bit_delta"],
                bytes_=row["selector_index_byte_delta_if_single_mutation"],
                delta=row["local_proxy_delta_with_rate"],
            )
        )
    lines.extend(
        [
            "",
            "These rows are proxy/advisory only. They are not score claims and are not",
            "rank-or-kill authority until a byte-different packet is materialized, runtime",
            "consumption is proven, and paired contest CPU/CUDA exact eval lands.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
