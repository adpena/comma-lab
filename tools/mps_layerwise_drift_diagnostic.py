#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CLI: measure layerwise drift of contest scorers across PyTorch backends.

Operator usage (no shortcuts; the --backend flag is REQUIRED to prevent
silent fallback per CLAUDE.md "Forbidden device-selection defaults"):

    # MPS vs CPU on PoseNet, fp32, 2-frame batch:
    .venv/bin/python tools/mps_layerwise_drift_diagnostic.py \\
        --scorer posenet \\
        --backends mps,cpu \\
        --batch-size 2 \\
        --seed 0 \\
        --out reports/raw/mps_drift_posenet_<utc>.md

    # MPS vs CPU on SegNet:
    .venv/bin/python tools/mps_layerwise_drift_diagnostic.py \\
        --scorer segnet \\
        --backends mps,cpu \\
        --batch-size 2 \\
        --seed 0 \\
        --out reports/raw/mps_drift_segnet_<utc>.md

Per CLAUDE.md "MPS auth eval is NOISE": the output is DIAGNOSTIC, not a
score claim. Every artifact carries `evidence_grade=macOS-MPS-diagnostic`
+ `score_claim=false` + `promotion_eligible=false`.

Lane: lane_mps_local_compute_frontier_diagnostic_20260518 (Catalog #126).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Repo root on sys.path so `tac.*` imports resolve without PYTHONPATH gymnastics
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
UPSTREAM_DIR = REPO_ROOT / "upstream"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from tac.mps_diagnostic.layerwise_drift import (  # noqa: E402
    emit_drift_table_markdown,
    identify_drift_cliff_layer,
    measure_layerwise_drift,
)
from tac.scorer import load_default_scorers  # noqa: E402


SCORER_INPUT_SHAPES = {
    # Canonical raw input: (B, T=2, C=3, H=384, W=512) per upstream contract.
    # Each scorer's `preprocess_input` consumes this raw shape.
    # We pass the raw (B,T,C,H,W) and the diagnostic wraps the model so
    # preprocess_input runs as part of the forward pass (matching the eval
    # contract that the contest scorer uses).
    "posenet": lambda batch_size: torch.randn(batch_size, 2, 3, 384, 512) * 64 + 128,
    "segnet": lambda batch_size: torch.randn(batch_size, 2, 3, 384, 512) * 64 + 128,
}


class _PreprocessingWrapper(torch.nn.Module):
    """Wrap a scorer so the forward pass invokes `preprocess_input` first.

    This matches the canonical eval contract used by `experiments/contest_auth_eval.py`
    and ensures the diagnostic measures the FULL forward path (preprocessing
    + backbone + head), not just the backbone.

    The wrapper itself is registered as a single module; the inner scorer's
    children remain accessible via `named_modules()` for the forward-hook
    drift capture.
    """

    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scorer.preprocess_input(x)
        return self.scorer(x)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Layerwise drift diagnostic for PyTorch scorers across "
            "MPS / CPU / CUDA backends. "
            "Diagnostic-only; non-promotable per CLAUDE.md \"MPS auth eval is NOISE\"."
        )
    )
    parser.add_argument(
        "--scorer",
        choices=sorted(SCORER_INPUT_SHAPES),
        required=True,
        help="which scorer to instrument",
    )
    parser.add_argument(
        "--backends",
        required=True,
        help=(
            "comma-separated backends to compare; pairs are computed as "
            "all (i,j) combinations. Example: --backends mps,cpu. "
            "Allowed values: mps, cpu, cuda."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch dimension for the synthetic input (default 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic seed shared across all backends (default 0)",
    )
    parser.add_argument(
        "--cliff-threshold",
        type=float,
        default=1e-3,
        help="L_inf above which a layer is considered diverged (default 1e-3)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output markdown table path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional JSON output path for machine-readable consumption",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=UPSTREAM_DIR,
        help=f"upstream repo root (default: {UPSTREAM_DIR})",
    )

    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    if len(backends) < 2:
        parser.error(
            f"--backends must specify >= 2 distinct backends; got {backends}"
        )

    # Load scorer (frozen, on CPU; we deepcopy + move per backend inside the helper)
    posenet, segnet = load_default_scorers(args.upstream_dir, device="cpu")
    scorer = posenet if args.scorer == "posenet" else segnet
    # Wrap so preprocess_input runs as part of the forward; this matches the
    # canonical eval contract.
    model = _PreprocessingWrapper(scorer)
    sample_input = SCORER_INPUT_SHAPES[args.scorer](args.batch_size)

    print(
        f"[mps-diagnostic] Scorer={args.scorer} "
        f"backends={backends} batch_size={args.batch_size} "
        f"seed={args.seed} cliff_threshold={args.cliff_threshold}",
        file=sys.stderr,
    )
    print(
        "[mps-diagnostic] evidence_grade=macOS-MPS-diagnostic "
        "score_claim=false promotion_eligible=false",
        file=sys.stderr,
    )

    result = measure_layerwise_drift(
        model,
        sample_input,
        backends=backends,
        seed=args.seed,
        sync_after_each_module=True,
        cliff_threshold=args.cliff_threshold,
    )

    # Tag the result with the scorer + UTC timestamp for downstream provenance
    result["scorer"] = args.scorer
    result["measurement_utc"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    result["batch_size"] = args.batch_size

    emit_drift_table_markdown(result, args.out)
    print(f"[mps-diagnostic] Drift table written to {args.out}", file=sys.stderr)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=str)
        print(f"[mps-diagnostic] JSON written to {args.json_out}", file=sys.stderr)

    for pair_name, pair_data in result["pairs"].items():
        cliff = pair_data.get("drift_cliff_layer")
        if cliff is not None:
            # Find the cliff record to surface its class + drift value
            cliff_rec = next(
                (r for r in pair_data["records"] if r["layer_name"] == cliff),
                None,
            )
            if cliff_rec is not None:
                print(
                    f"[mps-diagnostic] FIRST-DIVERGENCE [{pair_name}]: "
                    f"layer={cliff} class={cliff_rec['layer_class']} "
                    f"l_inf={cliff_rec['l_inf']:.3e}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[mps-diagnostic] CLEAN [{pair_name}]: no layer above "
                f"threshold {args.cliff_threshold}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
