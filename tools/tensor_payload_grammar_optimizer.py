#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the generic tensor payload grammar optimizer.

This is a planning-only rate gate for arbitrary exported tensor sets.  It writes
small JSON artifacts only; bulky checkpoints remain at their source path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.int_payload_bit_layouts import (  # noqa: E402
    DEFAULT_INT_PAYLOAD_LAYOUTS,
    VALID_INT_PAYLOAD_LAYOUTS,
    IntPayloadLayout,
)
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (  # noqa: E402
    DEFAULT_CODERS,
    CoderName,
)
from tac.packet_compiler.tensor_payload_grammar_optimizer import (  # noqa: E402
    build_tensor_payload_optimizer_queue,
    solve_tensor_payload_grammar,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _parse_coders(raw: str) -> tuple[CoderName, ...]:
    valid = set(DEFAULT_CODERS)
    values: list[CoderName] = []
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


def _parse_scale_dtypes(raw: str) -> tuple[str, ...]:
    valid = {"fp16", "fp32"}
    out: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if value not in valid:
            raise argparse.ArgumentTypeError(
                f"unknown scale dtype {value!r}; valid={sorted(valid)}"
            )
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("at least one scale dtype is required")
    return tuple(dict.fromkeys(out))


def _parse_payload_layouts(raw: str) -> tuple[IntPayloadLayout, ...]:
    out: list[IntPayloadLayout] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if value not in VALID_INT_PAYLOAD_LAYOUTS:
            raise argparse.ArgumentTypeError(
                f"unknown int payload layout {value!r}; "
                f"valid={sorted(VALID_INT_PAYLOAD_LAYOUTS)}"
            )
        out.append(value)  # type: ignore[arg-type]
    if not out:
        raise argparse.ArgumentTypeError("at least one int payload layout is required")
    return tuple(dict.fromkeys(out))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path, help="NumPy .npz tensor mapping.")
    source.add_argument(
        "--torch-state-dict",
        type=Path,
        help="Torch state_dict/checkpoint mapping. Requires torch import.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument("--campaign-id", default="tensor_payload_grammar")
    parser.add_argument("--n-quant", type=int, default=127)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--coders", type=_parse_coders, default=DEFAULT_CODERS)
    parser.add_argument("--scale-dtypes", type=_parse_scale_dtypes, default=("fp16",))
    parser.add_argument(
        "--payload-layouts",
        type=_parse_payload_layouts,
        default=DEFAULT_INT_PAYLOAD_LAYOUTS,
        help=(
            "Comma-separated lossless pre-entropy integer layouts "
            f"(valid: {','.join(sorted(VALID_INT_PAYLOAD_LAYOUTS))})."
        ),
    )
    parser.add_argument(
        "--storage-perm-mode",
        choices=("identity", "identity-plus-exhaustive4"),
        default="identity-plus-exhaustive4",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        help="Optional deterministic prefix limit for smoke/profile runs.",
    )
    return parser.parse_args(argv)


def _load_tensors(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path]:
    if args.npz is not None:
        path = Path(args.npz).expanduser()
        if not path.is_file():
            raise SystemExit(f"npz not found: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            tensors = {name: loaded[name] for name in sorted(loaded.files)}
        return _limit_tensors(tensors, args.max_tensors), "npz", path
    path = Path(args.torch_state_dict).expanduser()
    if not path.is_file():
        raise SystemExit(f"torch state_dict not found: {path}")
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-local-tensor-grammar-profiler-input
    if not isinstance(loaded, dict):
        raise SystemExit(f"loaded object is not a mapping: {type(loaded)!r}")
    state = loaded.get("state_dict") if isinstance(loaded.get("state_dict"), dict) else loaded
    tensors: dict[str, Any] = {}
    for name in sorted(state):
        value = state[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        tensors[str(name)] = np.asarray(value)
    return _limit_tensors(tensors, args.max_tensors), "torch_state_dict", path


def _limit_tensors(tensors: dict[str, Any], max_tensors: int | None) -> dict[str, Any]:
    if max_tensors is None:
        return tensors
    if max_tensors <= 0:
        raise SystemExit("--max-tensors must be positive")
    return dict(list(tensors.items())[: int(max_tensors)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tensors, source_kind, source_path = _load_tensors(args)
    report = solve_tensor_payload_grammar(
        tensors,
        n_quant=args.n_quant,
        scale_dtypes=args.scale_dtypes,
        storage_perm_mode=args.storage_perm_mode,
        payload_layouts=args.payload_layouts,
        coders=args.coders,
        brotli_quality=args.brotli_quality,
        campaign_id=args.campaign_id,
        source_kind=source_kind,
    )
    report["source_payload_manifest"] = {
        **report["source_payload_manifest"],
        "source_path": source_path.as_posix(),
    }
    try:
        report_artifact = write_json_artifact(
            args.output,
            report,
            allow_overwrite=False,
        )
        queue_artifact = None
        if args.queue_output is not None:
            queue = build_tensor_payload_optimizer_queue(report)
            queue_artifact = write_json_artifact(
                args.queue_output,
                queue,
                allow_overwrite=False,
            )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc

    bytes_ = report["byte_accounting"]
    print(
        json.dumps(
            {
                "ok": True,
                "campaign_id": report["campaign_id"],
                "source_kind": source_kind,
                "tensor_count": report["tensor_count"],
                "output": report_artifact.path,
                "queue_output": None
                if queue_artifact is None
                else queue_artifact.path,
                "selected_isolated_tensor_bytes": bytes_[
                    "selected_isolated_tensor_bytes"
                ],
                "selected_saved_bytes_vs_baseline": bytes_[
                    "selected_saved_bytes_vs_baseline"
                ],
                "saturation_status": report["saturation_diagnostic"]["status"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
