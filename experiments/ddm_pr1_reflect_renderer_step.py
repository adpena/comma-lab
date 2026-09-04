#!/usr/bin/env python3
"""ddm_pr1 (bonus): build the REFLECTED renderer step, to test direction symmetry.

WHY.  ``renderer_seg_pose_coupling_shipped_object_v1`` carries exactly one
assumption it labels stated-not-measured:

    "DIRECTION SYMMETRY: both anchors moved d_seg UP; the closing arithmetic
     applies the same k to a seg DECREASE, i.e. it assumes local linearity of the
     realized map around the shipped weights. Cheapest falsification available."

Both registered anchors, and ddm_pr1's own candidate, moved d_seg the WRONG way.
No seg-improving renderer change exists to measure, so the whole closing
arithmetic -- ft1's and this arm's -- rests on an untested reflection.

WHAT THIS BUILDS.  The point-reflection of the realized candidate through the
shipped weights,

    W_reflected = 2 * W_shipped - W_candidate      (elementwise, per tensor)

which is the same step along the seg-only direction with the sign flipped.  If
the realized map were locally linear the reflected object would move d_seg DOWN
by about as much as the candidate moved it UP, and would pay about the same
|Delta d_pose|.  Measuring it is the cheapest available falsification of the
assumption the law itself names.

WHAT IT DOES NOT CLAIM.  The deployed SM3R encoder derives its per-tensor scales
from the VALUES, so the realized reflection is not guaranteed to be the exact
reflection of the realized candidate.  This tool therefore exports the reflected
state through the shipped encoder, parses it back through the shipped receiver,
and reports how well the REALIZED reflection actually opposes the REALIZED
candidate step (cosine and norm ratio).  A poor alignment is itself the answer:
it would mean the export cannot represent the opposite step, and the symmetry
question cannot be asked of this representation at all.

Axis: nothing here is a measurement -- it builds two artifacts and a record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ft1_identity_gate_and_caches import (
    FRONTIER_ARCHIVE,
    SEMANTIC_WIDTH,
    load_shipped_renderer_module,
    read_semantic_section,
    sha256_bytes,
)
from experiments.ddm_ft1_verdict_bhw_pose import export_section


def _flat(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).double() for _, v in sorted(delta.items())])


def build_reflection(shipped, candidate) -> dict[str, torch.Tensor]:
    """``2 * shipped - candidate`` per tensor, with the key sets gated."""
    if set(shipped) != set(candidate):
        raise ValueError("shipped and candidate state dicts have different keys")
    out = {}
    for name, base in shipped.items():
        other = candidate[name]
        if base.shape != other.shape:
            raise ValueError(f"{name}: shape {tuple(base.shape)} != {tuple(other.shape)}")
        out[name] = (2.0 * base.double() - other.double()).to(base.dtype)
    return out


def alignment(shipped, candidate, reflected) -> dict[str, float]:
    """How well the REALIZED reflection opposes the REALIZED candidate step."""
    forward = _flat({k: candidate[k] - shipped[k] for k in shipped})
    backward = _flat({k: reflected[k] - shipped[k] for k in shipped})
    fnorm = float(forward.norm())
    bnorm = float(backward.norm())
    denominator = fnorm * bnorm
    return {
        "forward_step_norm": fnorm,
        "reflected_step_norm": bnorm,
        "norm_ratio": bnorm / fnorm if fnorm else float("nan"),
        "cosine": float(forward.dot(backward)) / denominator if denominator else float("nan"),
        "perfect_reflection_cosine": -1.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-section", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=FRONTIER_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="pr1_reflected")
    args = parser.parse_args(argv)

    shipped_module = load_shipped_renderer_module()
    template = shipped_module.SemanticTokenRenderer(SEMANTIC_WIDTH).state_dict()
    shipped_blob = bytes(read_semantic_section(args.archive))
    shipped_state = shipped_module.unpack_variant_semantic_or_none(shipped_blob, template)
    candidate_blob = args.candidate_section.read_bytes()
    candidate_state = shipped_module.unpack_variant_semantic_or_none(
        candidate_blob, template
    )
    if shipped_state is None or candidate_state is None:
        raise SystemExit("the shipped receiver rejected one of the sections")

    intended = build_reflection(shipped_state, candidate_state)
    export, realized = export_section(intended, args.archive)
    payload = export.pop("payload")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # ALWAYS KEEP THE PAYLOAD: bytes before the record that describes them.
    section_path = out_dir / f"{args.label}_semantic_section.bin"
    section_path.write_bytes(payload)
    checkpoint_path = out_dir / f"{args.label}_checkpoint.pt"
    torch.save(
        {"deployment_weights": "ema_shadow", "state_dict": realized},
        checkpoint_path,
    )

    record = {
        "schema": "tac.ddm_pr1.reflection.v1",
        "label": args.label,
        "score_claim": False,
        "promotable": False,
        "archive": str(args.archive),
        "shipped_section_sha256": sha256_bytes(shipped_blob),
        "candidate_section": str(args.candidate_section),
        "candidate_section_sha256": sha256_bytes(candidate_blob),
        "reflected_section_path": str(section_path),
        "reflected_section_sha256": hashlib.sha256(payload).hexdigest(),
        "reflected_checkpoint_path": str(checkpoint_path),
        "export": export,
        "alignment_realized_vs_intended_reflection": alignment(
            shipped_state, candidate_state, realized
        ),
        "note": (
            "the checkpoint carries the REALIZED (parsed-back) reflected state, so a "
            "verdict run on it re-exports an already-realized object and its "
            "trained-vs-realized gap should be ~0"
        ),
    }
    (out_dir / f"{args.label}_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
