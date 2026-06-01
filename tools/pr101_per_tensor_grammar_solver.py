#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the PR101 per-tensor decoder-weight grammar solver.

The output is a profiler/solver manifest, not score authority.  It is intended
to feed packetIR/HPRC planning and future receiver-adapter work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr101_per_tensor_grammar_solver import (  # noqa: E402
    CoderName,
    build_optimizer_candidate_queue_from_solver_report,
    default_state_dict_output_path_hint,
    solve_state_dict_per_tensor_grammar,
)


def _parse_coders(raw: str) -> tuple[CoderName, ...]:
    values: list[CoderName] = []
    valid = {
        "brotli",
        "lzma_raw",
        "canonical_huffman",
        "range_ac_empirical_hist_u16",
    }
    for item in raw.split(","):
        coder = item.strip()
        if not coder:
            continue
        if coder not in valid:
            raise argparse.ArgumentTypeError(
                f"unknown coder {coder!r}; valid={sorted(valid)}"
            )
        values.append(coder)  # type: ignore[arg-type]
    if not values:
        raise argparse.ArgumentTypeError("at least one coder is required")
    return tuple(values)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {text!r}") from exc
        if not 0 <= value <= 11:
            raise argparse.ArgumentTypeError("Brotli quality values must be in [0, 11]")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return tuple(dict.fromkeys(values))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=None,
        help=(
            "Optional optimizer_candidate_queue_v1 output. Rows stay "
            "planning-only and dispatch-blocked until grouped packet and "
            "receiver proof exist."
        ),
    )
    parser.add_argument("--n-quant", type=int, default=127)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--brotli-qualities",
        type=_parse_int_csv,
        default=None,
        help="Optional comma-separated Brotli quality values to exhaustively price.",
    )
    parser.add_argument(
        "--brotli-quality-sweep",
        action="store_true",
        help="Exhaustively price Brotli qualities 0..11. Compress-time only.",
    )
    parser.add_argument("--brotli-lgwin-sweep", action="store_true")
    parser.add_argument(
        "--storage-perm-mode",
        choices=("identity", "pr101-plus-identity", "exhaustive-conv4"),
        default="pr101-plus-identity",
    )
    parser.add_argument(
        "--coders",
        type=_parse_coders,
        default=_parse_coders("brotli,lzma_raw,canonical_huffman"),
        help=(
            "Comma-separated coder set. Add range_ac_empirical_hist_u16 to "
            "price the constriction path with conservative histogram overhead."
        ),
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Optional prefix sample for cheap smoke; omitted means all 28 tensors.",
    )
    parser.add_argument(
        "--skip-current-grouped-pr101",
        action="store_true",
        help="Skip exact current split-Brotli grouped byte measurement.",
    )
    args = parser.parse_args(argv)

    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    import torch

    state_dict = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-local-state-dict-profiler-input
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded object is not a state_dict: {type(state_dict)!r}")

    brotli_quality_values = (
        tuple(range(12)) if args.brotli_quality_sweep else args.brotli_qualities
    )

    report = solve_state_dict_per_tensor_grammar(
        state_dict,
        n_quant=args.n_quant,
        storage_perm_mode=args.storage_perm_mode,
        coders=args.coders,
        brotli_quality=args.brotli_quality,
        brotli_quality_values=brotli_quality_values,
        brotli_lgwin_sweep=args.brotli_lgwin_sweep,
        max_tensors=args.max_tensors,
        include_current_grouped_pr101=not args.skip_current_grouped_pr101,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    if args.queue_output is not None:
        queue = build_optimizer_candidate_queue_from_solver_report(report)
        args.queue_output.parent.mkdir(parents=True, exist_ok=True)
        args.queue_output.write_text(
            json.dumps(queue, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
    bytes_ = report["byte_accounting"]
    print(f"Wrote PR101 per-tensor grammar report to {args.output}")
    if args.queue_output is not None:
        print(f"Wrote planning-only optimizer queue to {args.queue_output}")
    print(
        "selected isolated bytes="
        f"{bytes_['selected_isolated_tensor_bytes']}; "
        "current isolated bytes="
        f"{bytes_['current_pr101_isolated_tensor_bytes']}; "
        "floor="
        f"{bytes_['empirical_shannon_floor_bytes']:.1f}; "
        f"status={report['saturation_diagnostic']['status']}"
    )
    print(
        "Preferred bulky-output root: "
        f"{default_state_dict_output_path_hint()} "
        "(this run only wrote the requested JSON manifest)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
