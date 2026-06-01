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
    build_grouped_optimizer_candidate_queue_from_report,
    build_optimizer_candidate_queue_from_solver_report,
    build_u32_receiver_adapter_source_from_report,
    build_u32_receiver_runtime_tree_from_report,
    default_state_dict_output_path_hint,
    materialize_grouped_archive_from_report,
    materialize_grouped_decoder_blob_from_report,
    solve_grouped_brotli_packet_grammar,
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
    parser.add_argument(
        "--grouped-output",
        type=Path,
        default=None,
        help=(
            "Optional grouped split-Brotli packet report. This prices stream "
            "partition/context effects after per-tensor transforms."
        ),
    )
    parser.add_argument(
        "--grouped-queue-output",
        type=Path,
        default=None,
        help="Optional planning-only optimizer queue for the grouped packet report.",
    )
    parser.add_argument(
        "--grouped-decoder-blob-output",
        type=Path,
        default=None,
        help=(
            "Optional byte-closed grouped decoder blob output. Requires a full "
            "schema run; still not a submission archive."
        ),
    )
    parser.add_argument(
        "--grouped-decoder-proof-output",
        type=Path,
        default=None,
        help="Optional materialization proof JSON for --grouped-decoder-blob-output.",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help=(
            "Optional PR101 source archive.zip whose latent/sidecar sections are "
            "preserved when emitting --grouped-archive-output."
        ),
    )
    parser.add_argument(
        "--grouped-archive-output",
        type=Path,
        default=None,
        help=(
            "Optional deterministic single-member PR101-shaped archive.zip. "
            "Requires --source-archive and stays fail-closed unless fixed-offset "
            "stock runtime proof passes."
        ),
    )
    parser.add_argument(
        "--grouped-archive-proof-output",
        type=Path,
        default=None,
        help="Optional archive materialization proof JSON for --grouped-archive-output.",
    )
    parser.add_argument(
        "--grouped-archive-layout",
        choices=("fixed_pr101", "u32_decoder_len_adapter"),
        default="fixed_pr101",
        help=(
            "Archive inner layout for --grouped-archive-output. fixed_pr101 "
            "keeps PR101 fixed offsets; u32_decoder_len_adapter emits a "
            "length-prefixed decoder section and requires a receiver runtime adapter."
        ),
    )
    parser.add_argument(
        "--grouped-receiver-adapter-output",
        type=Path,
        default=None,
        help="Optional generated Python parser adapter source for u32 decoder-length archives.",
    )
    parser.add_argument(
        "--grouped-receiver-adapter-proof-output",
        type=Path,
        default=None,
        help="Optional receiver adapter source proof JSON.",
    )
    parser.add_argument(
        "--source-runtime-dir",
        type=Path,
        default=None,
        help=(
            "PR101-family source runtime root containing src/model.py. Required "
            "for --grouped-runtime-output-dir."
        ),
    )
    parser.add_argument(
        "--runtime-codec-source",
        type=Path,
        default=REPO_ROOT / "src" / "tac" / "pr101_split_brotli_codec.py",
        help=(
            "Override-aware codec.py source to vendor into grouped runtime trees. "
            "Defaults to src/tac/pr101_split_brotli_codec.py."
        ),
    )
    parser.add_argument(
        "--grouped-runtime-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for a self-contained u32 decoder-length "
            "receiver runtime tree. Directory must be absent or empty."
        ),
    )
    parser.add_argument(
        "--grouped-runtime-proof-output",
        type=Path,
        default=None,
        help="Optional runtime-tree materialization proof JSON.",
    )
    parser.add_argument(
        "--grouped-transform-mode",
        choices=("stock_pr101", "best_brotli_per_tensor"),
        default="best_brotli_per_tensor",
    )
    parser.add_argument(
        "--grouped-exact-stream-count",
        type=int,
        default=7,
        help="Exact split-Brotli stream count for grouped solving; use -1 for best up to max.",
    )
    parser.add_argument("--grouped-max-streams", type=int, default=7)
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
    grouped_report = None
    if (
        args.grouped_output is not None
        or args.grouped_queue_output is not None
        or args.grouped_decoder_blob_output is not None
        or args.grouped_decoder_proof_output is not None
        or args.grouped_archive_output is not None
        or args.grouped_archive_proof_output is not None
        or args.grouped_receiver_adapter_output is not None
        or args.grouped_receiver_adapter_proof_output is not None
        or args.grouped_runtime_output_dir is not None
        or args.grouped_runtime_proof_output is not None
    ):
        exact_stream_count = (
            None
            if args.grouped_exact_stream_count < 0
            else args.grouped_exact_stream_count
        )
        grouped_report = solve_grouped_brotli_packet_grammar(
            state_dict,
            n_quant=args.n_quant,
            selected_transform_mode=args.grouped_transform_mode,
            storage_perm_mode=args.storage_perm_mode,
            exact_stream_count=exact_stream_count,
            max_streams=args.grouped_max_streams,
            brotli_quality=args.brotli_quality,
            brotli_lgwin_sweep=args.brotli_lgwin_sweep,
            max_tensors=args.max_tensors,
        )
        if args.grouped_output is not None:
            args.grouped_output.parent.mkdir(parents=True, exist_ok=True)
            args.grouped_output.write_text(
                json.dumps(grouped_report, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
        if args.grouped_queue_output is not None:
            grouped_queue = build_grouped_optimizer_candidate_queue_from_report(grouped_report)
            args.grouped_queue_output.parent.mkdir(parents=True, exist_ok=True)
            args.grouped_queue_output.write_text(
                json.dumps(grouped_queue, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
        if args.grouped_decoder_blob_output is not None or args.grouped_decoder_proof_output is not None:
            blob, proof = materialize_grouped_decoder_blob_from_report(
                state_dict,
                grouped_report,
                n_quant=args.n_quant,
            )
            if args.grouped_decoder_blob_output is not None:
                args.grouped_decoder_blob_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_decoder_blob_output.write_bytes(blob)
            if args.grouped_decoder_proof_output is not None:
                args.grouped_decoder_proof_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_decoder_proof_output.write_text(
                    json.dumps(proof, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
        if args.grouped_archive_output is not None or args.grouped_archive_proof_output is not None:
            if args.source_archive is None:
                raise SystemExit("--source-archive is required for grouped archive materialization")
            if not args.source_archive.is_file():
                raise SystemExit(f"source archive not found: {args.source_archive}")
            archive_bytes, archive_proof = materialize_grouped_archive_from_report(
                state_dict,
                grouped_report,
                source_archive_zip=args.source_archive.read_bytes(),
                n_quant=args.n_quant,
                archive_layout=args.grouped_archive_layout,
            )
            if args.grouped_archive_output is not None:
                args.grouped_archive_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_archive_output.write_bytes(archive_bytes)
            if args.grouped_archive_proof_output is not None:
                args.grouped_archive_proof_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_archive_proof_output.write_text(
                    json.dumps(archive_proof, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
        if (
            args.grouped_receiver_adapter_output is not None
            or args.grouped_receiver_adapter_proof_output is not None
        ):
            source, adapter_proof = build_u32_receiver_adapter_source_from_report(grouped_report)
            if args.grouped_receiver_adapter_output is not None:
                args.grouped_receiver_adapter_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_receiver_adapter_output.write_text(source, encoding="utf-8")
            if args.grouped_receiver_adapter_proof_output is not None:
                args.grouped_receiver_adapter_proof_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_receiver_adapter_proof_output.write_text(
                    json.dumps(adapter_proof, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
        if args.grouped_runtime_output_dir is not None or args.grouped_runtime_proof_output is not None:
            if args.source_runtime_dir is None:
                raise SystemExit("--source-runtime-dir is required for grouped runtime materialization")
            model_source = args.source_runtime_dir / "src" / "model.py"
            if not model_source.is_file():
                raise SystemExit(f"source runtime model.py not found: {model_source}")
            if not args.runtime_codec_source.is_file():
                raise SystemExit(f"runtime codec source not found: {args.runtime_codec_source}")
            runtime_files, runtime_proof = build_u32_receiver_runtime_tree_from_report(
                grouped_report,
                codec_py_source=args.runtime_codec_source.read_bytes(),
                model_py_source=model_source.read_bytes(),
            )
            if args.grouped_runtime_output_dir is not None:
                _write_runtime_tree(args.grouped_runtime_output_dir, runtime_files)
            if args.grouped_runtime_proof_output is not None:
                args.grouped_runtime_proof_output.parent.mkdir(parents=True, exist_ok=True)
                args.grouped_runtime_proof_output.write_text(
                    json.dumps(runtime_proof, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
    bytes_ = report["byte_accounting"]
    print(f"Wrote PR101 per-tensor grammar report to {args.output}")
    if args.queue_output is not None:
        print(f"Wrote planning-only optimizer queue to {args.queue_output}")
    if args.grouped_output is not None:
        print(f"Wrote grouped split-Brotli packet report to {args.grouped_output}")
    if args.grouped_queue_output is not None:
        print(f"Wrote grouped planning-only optimizer queue to {args.grouped_queue_output}")
    if args.grouped_decoder_blob_output is not None:
        print(f"Wrote grouped decoder blob to {args.grouped_decoder_blob_output}")
    if args.grouped_decoder_proof_output is not None:
        print(f"Wrote grouped decoder materialization proof to {args.grouped_decoder_proof_output}")
    if args.grouped_archive_output is not None:
        print(f"Wrote grouped archive zip to {args.grouped_archive_output}")
    if args.grouped_archive_proof_output is not None:
        print(f"Wrote grouped archive materialization proof to {args.grouped_archive_proof_output}")
    if args.grouped_receiver_adapter_output is not None:
        print(f"Wrote grouped receiver adapter source to {args.grouped_receiver_adapter_output}")
    if args.grouped_receiver_adapter_proof_output is not None:
        print(
            "Wrote grouped receiver adapter source proof to "
            f"{args.grouped_receiver_adapter_proof_output}"
        )
    if args.grouped_runtime_output_dir is not None:
        print(f"Wrote grouped receiver runtime tree to {args.grouped_runtime_output_dir}")
    if args.grouped_runtime_proof_output is not None:
        print(f"Wrote grouped receiver runtime tree proof to {args.grouped_runtime_proof_output}")
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
        "(this run only wrote explicitly requested outputs)"
    )
    if grouped_report is not None:
        gbytes = grouped_report["byte_accounting"]
        print(
            "grouped selected bytes="
            f"{gbytes['selected_grouped_brotli_bytes']}; "
            "current grouped bytes="
            f"{gbytes['current_stock_pr101_grouped_bytes']}; "
            "saved="
            f"{gbytes['grouped_saved_bytes_vs_current_stock']}; "
            f"runtime={grouped_report['runtime_consumption_status']}"
        )
    return 0


def _write_runtime_tree(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"runtime output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in files.items():
        path = output_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if rel_path == "inflate.sh":
            path.chmod(0o755)


if __name__ == "__main__":
    raise SystemExit(main())
