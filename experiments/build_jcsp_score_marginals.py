# SPDX-License-Identifier: MIT
"""CLI to produce a JCSP score-marginals artifact from a model checkpoint.

The γ-JCSP dispatch path in ``experiments/pipeline.py`` requires a
score-marginals JSON artifact (path passed via
``cfg.jcsp_score_marginals_path``). This CLI is the canonical producer.

Two modes:

* ``--mode uniform`` — uniform placeholder marginals tagged
  ``placeholder_uniform``. Use only for plumbing smoke tests.
* ``--mode sensitivity`` — derives per-tensor marginals from a serialized
  SensitivityMap artifact at ``--sensitivity-path``. Tagged
  ``sensitivity_derived``.

A future ``--mode contest-cuda`` will run a finite-difference probe
against the scorer and emit ``contest_cuda_calibrated`` artifacts. That
mode requires GPU dispatch and is out of scope for this CLI.

Usage examples::

    .venv/bin/python experiments/build_jcsp_score_marginals.py \\
        --model experiments/results/lane_a_baseline/renderer.pt \\
        --out reports/jcsp_marginals_uniform.json \\
        --mode uniform \\
        --uniform-value 1e-6 \\
        --evidence "smoke test for cfg.use_joint_codec_stack=True gate"

    .venv/bin/python experiments/build_jcsp_score_marginals.py \\
        --model experiments/results/lane_a_baseline/renderer.pt \\
        --out reports/jcsp_marginals_sens.json \\
        --mode sensitivity \\
        --sensitivity-path reports/sensitivity_map.pt \\
        --evidence "Lane A baseline anchor + sensitivity sweep at git $(git rev-parse HEAD)"

NEVER loads scorers. NEVER claims a contest-CUDA score. Tags every
output ``placeholder_uniform`` or ``sensitivity_derived`` so future
agents can audit the provenance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.jcsp_score_marginals import (
    derive_marginals_from_sensitivity_map,
    derive_uniform_planning_marginals,
    save_marginals,
)


def _load_state_dict(path: Path) -> dict:
    """Load a state_dict (or {'model': sd, ...} envelope) from a .pt file.

    Uses ``weights_only=True`` by default. The
    ``JCSP_MARGINALS_ALLOW_UNSAFE_CKPT=1`` env-var is a deliberate opt-in
    for trusted internal checkpoints (per CLAUDE.md non-negotiable
    "torch.load weights_only=False on user data" rule).
    """
    if not path.exists():
        raise SystemExit(f"--model path does not exist: {path}")
    weights_only = os.environ.get("JCSP_MARGINALS_ALLOW_UNSAFE_CKPT") != "1"
    raw = torch.load(str(path), map_location="cpu", weights_only=weights_only)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    else:
        sd = raw
    if not isinstance(sd, dict):
        raise SystemExit(
            f"Could not extract a state_dict from {path}; got "
            f"{type(sd).__name__}"
        )
    return sd


def _load_sensitivities(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"--sensitivity-path does not exist: {path}")
    weights_only = os.environ.get("JCSP_MARGINALS_ALLOW_UNSAFE_CKPT") != "1"
    raw = torch.load(str(path), map_location="cpu", weights_only=weights_only)
    if isinstance(raw, dict) and "sensitivities" in raw:
        sens = raw["sensitivities"]
    elif isinstance(raw, dict):
        sens = raw
    else:
        raise SystemExit(
            f"--sensitivity-path must be a dict-like artifact; got "
            f"{type(raw).__name__}"
        )
    return sens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a JCSP score-marginals JSON artifact",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a model checkpoint (state_dict .pt). "
        "Loaded with weights_only=True unless "
        "JCSP_MARGINALS_ALLOW_UNSAFE_CKPT=1 is set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON artifact path. Parent dir must exist.",
    )
    parser.add_argument(
        "--mode",
        choices=["uniform", "sensitivity"],
        required=True,
        help="uniform: placeholder marginals tagged placeholder_uniform. "
        "sensitivity: derive from SensitivityMap, tagged sensitivity_derived.",
    )
    parser.add_argument(
        "--uniform-value",
        type=float,
        default=1e-6,
        help="Marginal value for --mode uniform (default 1e-6).",
    )
    parser.add_argument(
        "--sensitivity-path",
        type=Path,
        default=None,
        help="Path to a SensitivityMap .pt artifact (required for --mode sensitivity).",
    )
    parser.add_argument(
        "--bytes-per-element",
        type=float,
        default=0.5,
        help="Estimated codec cost per element (default 0.5 ≈ FP4).",
    )
    parser.add_argument(
        "--fallback-marginal",
        type=float,
        default=1e-9,
        help="Fallback marginal for tensors missing from sensitivity map.",
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help="Provenance string (e.g. 'lane A baseline at git deadbeef').",
    )
    args = parser.parse_args(argv)

    state_dict = _load_state_dict(args.model)
    if args.mode == "uniform":
        marginals = derive_uniform_planning_marginals(
            state_dict, value=args.uniform_value
        )
        source = "placeholder_uniform"
    else:
        if args.sensitivity_path is None:
            parser.error("--mode sensitivity requires --sensitivity-path")
        sensitivities = _load_sensitivities(args.sensitivity_path)
        marginals = derive_marginals_from_sensitivity_map(
            state_dict,
            sensitivities,
            bytes_per_element_estimate=args.bytes_per_element,
            fallback_marginal=args.fallback_marginal,
        )
        source = "sensitivity_derived"

    out_path = save_marginals(
        args.out,
        marginals,
        source=source,
        evidence=args.evidence,
    )
    print(
        f"[jcsp-marginals] wrote {len(marginals)} marginals "
        f"(source={source}) → {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
