# SPDX-License-Identifier: MIT
"""Per-frame projection for per-pair master-gradient tensors.

The contest scorer is asymmetric at input topology:

- SegNet consumes only the last frame of each seq_len=2 sample.
- PoseNet consumes both frames of the seq_len=2 sample.

Upstream `frame_utils.py` emits non-overlapping pairs `(0, 1), (2, 3), ...`.
This module exposes that topology as a queryable per-frame sensitivity table
for bit allocators, training curricula, and cathedral consumers. It is a
decomposition of an existing per-pair tensor, not new score authority.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from tac.master_gradient import OperatingPoint, compute_marginal_coefficients

Topology = Literal["non_overlapping", "sliding"]
AXIS_LABELS = ("seg", "pose", "rate")
SCHEMA = "per_frame_decomposition_segnet_per_frame_posenet_per_pair_v1"


class MasterGradientFrameDecompositionError(ValueError):
    """Raised when per-frame decomposition inputs are malformed."""


@dataclass(frozen=True)
class FrameDecompositionConfig:
    topology: Topology = "non_overlapping"
    pose_first_frame_share: float = 0.5
    rate_first_frame_share: float = 0.5
    top_k_frames: int = 20

    def validate(self) -> None:
        if self.topology not in {"non_overlapping", "sliding"}:
            raise MasterGradientFrameDecompositionError(f"unsupported topology={self.topology!r}")
        for name, value in (
            ("pose_first_frame_share", self.pose_first_frame_share),
            ("rate_first_frame_share", self.rate_first_frame_share),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise MasterGradientFrameDecompositionError(f"{name} must be in [0, 1]")
        if self.top_k_frames <= 0:
            raise MasterGradientFrameDecompositionError("top_k_frames must be positive")


def score_axis_coefficients_from_operating_point(op: OperatingPoint) -> tuple[float, float, float]:
    """Return score marginal coefficients `(seg, pose, rate)` for a master-gradient op."""

    return compute_marginal_coefficients(op)


def load_anchor_for_gradient_path(
    *,
    gradient_path: Path,
    ledger_path: Path = Path(".omx/state/master_gradient_anchors.jsonl"),
) -> dict[str, Any] | None:
    """Return the newest ledger row whose gradient path resolves to `gradient_path`."""

    if not ledger_path.is_file():
        return None
    target = gradient_path.resolve()
    best: dict[str, Any] | None = None
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_path = row.get("gradient_array_path")
        if not isinstance(raw_path, str):
            continue
        try:
            candidate = Path(raw_path).expanduser().resolve()
        except OSError:
            continue
        if candidate == target:
            best = row
    return best


def axis_coefficients_from_anchor(anchor: dict[str, Any]) -> tuple[float, float, float]:
    op = anchor.get("operating_point")
    if not isinstance(op, dict):
        raise MasterGradientFrameDecompositionError("anchor missing operating_point")
    return score_axis_coefficients_from_operating_point(
        OperatingPoint(
            d_seg=float(op["d_seg"]),
            d_pose=float(op["d_pose"]),
            rate=float(op["rate"]),
            score=float(op["score"]),
        )
    )


def _pair_frames(pair_index: int, topology: Topology) -> tuple[int, int]:
    if topology == "non_overlapping":
        return 2 * pair_index, 2 * pair_index + 1
    if topology == "sliding":
        return pair_index, pair_index + 1
    raise MasterGradientFrameDecompositionError(f"unsupported topology={topology!r}")


def decompose_per_pair_gradient_to_frames(
    per_pair_gradient: Any,
    *,
    axis_coefficients: tuple[float, float, float],
    config: FrameDecompositionConfig | None = None,
) -> dict[str, Any]:
    """Project `(N_bytes, N_pairs, 3)` sensitivity onto frame rows.

    The tensor is collapsed with L1 magnitude after applying score marginal
    coefficients. This is appropriate for budget allocation and difficulty
    ranking; it intentionally does not claim signed byte-mutation authority.
    """

    if np is None:
        raise RuntimeError("numpy required for master-gradient frame decomposition")
    cfg = config or FrameDecompositionConfig()
    cfg.validate()
    arr = np.asarray(per_pair_gradient, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise MasterGradientFrameDecompositionError(
            f"per_pair_gradient must have shape (N_bytes, N_pairs, 3); got {arr.shape}"
        )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in axis_coefficients):
        raise MasterGradientFrameDecompositionError("axis_coefficients must be finite non-negative values")

    coeffs = np.asarray(axis_coefficients, dtype=np.float64)
    weighted_abs = np.abs(arr) * coeffs.reshape(1, 1, 3)
    pair_axis_l1 = weighted_abs.sum(axis=0)
    n_pairs = int(pair_axis_l1.shape[0])
    n_frames = 2 * n_pairs if cfg.topology == "non_overlapping" else n_pairs + 1
    frame_axis_l1 = np.zeros((n_frames, 3), dtype=np.float64)
    pose_first = float(cfg.pose_first_frame_share)
    rate_first = float(cfg.rate_first_frame_share)
    for pair_index in range(n_pairs):
        first, last = _pair_frames(pair_index, cfg.topology)
        seg, pose, rate = pair_axis_l1[pair_index]
        frame_axis_l1[last, 0] += seg
        frame_axis_l1[first, 1] += pose * pose_first
        frame_axis_l1[last, 1] += pose * (1.0 - pose_first)
        frame_axis_l1[first, 2] += rate * rate_first
        frame_axis_l1[last, 2] += rate * (1.0 - rate_first)

    frame_total = frame_axis_l1.sum(axis=1)
    order = np.argsort(-frame_total, kind="stable")[: min(cfg.top_k_frames, n_frames)]
    top_frames = [
        {
            "frame_index": int(idx),
            "rank": int(rank + 1),
            "total_l1": float(frame_total[idx]),
            "seg_l1": float(frame_axis_l1[idx, 0]),
            "pose_l1": float(frame_axis_l1[idx, 1]),
            "rate_l1": float(frame_axis_l1[idx, 2]),
        }
        for rank, idx in enumerate(order)
    ]
    pair_sum = pair_axis_l1.sum(axis=0)
    frame_sum = frame_axis_l1.sum(axis=0)
    conservation_abs_error = np.abs(pair_sum - frame_sum)
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "topology": cfg.topology,
        "scorer_input_topology": {
            "segnet": "last_frame_only",
            "posenet": "both_frames",
            "upstream_pairing": (
                "non-overlapping seq_len=2 chunks from upstream/frame_utils.py"
                if cfg.topology == "non_overlapping"
                else "exploratory sliding pair projection; not upstream eval topology"
            ),
        },
        "axis_labels": list(AXIS_LABELS),
        "axis_coefficients": {
            label: float(coeffs[idx]) for idx, label in enumerate(AXIS_LABELS)
        },
        "n_bytes": int(arr.shape[0]),
        "n_pairs": n_pairs,
        "n_frames": n_frames,
        "pose_first_frame_share": pose_first,
        "rate_first_frame_share": rate_first,
        "pair_axis_l1_sum": {
            label: float(pair_sum[idx]) for idx, label in enumerate(AXIS_LABELS)
        },
        "frame_axis_l1_sum": {
            label: float(frame_sum[idx]) for idx, label in enumerate(AXIS_LABELS)
        },
        "conservation_abs_error": {
            label: float(conservation_abs_error[idx]) for idx, label in enumerate(AXIS_LABELS)
        },
        "conservation_ok": bool(np.all(conservation_abs_error <= 1e-8 * np.maximum(1.0, pair_sum))),
        "top_frames": top_frames,
        "frame_axis_l1": frame_axis_l1,
    }


def json_ready_decomposition(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without the dense numpy array."""

    out = dict(payload)
    out.pop("frame_axis_l1", None)
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Master-Gradient Per-Frame Decomposition",
        "",
        f"- Topology: `{payload['topology']}`",
        f"- Pairs: {payload['n_pairs']}",
        f"- Frames: {payload['n_frames']}",
        f"- Upstream pairing: {payload['scorer_input_topology']['upstream_pairing']}",
        f"- SegNet input: {payload['scorer_input_topology']['segnet']}",
        f"- PoseNet input: {payload['scorer_input_topology']['posenet']}",
        f"- Conservation: {payload['conservation_ok']}",
        f"- Score claim: {payload['score_claim']}",
    ]
    if "source_gradient_npy" in payload:
        lines.append(f"- Source gradient: `{payload['source_gradient_npy']}`")
    if "coefficient_source" in payload:
        lines.append(f"- Coefficients: `{payload['coefficient_source']}`")
    lines.extend(
        [
            "",
            "This is a queryable structural decomposition of an existing per-pair tensor, not new gradient authority.",
            "",
            "## Top Frames",
            "",
        ]
    )
    for row in payload["top_frames"]:
        lines.append(
            f"- frame `{row['frame_index']}` rank={row['rank']} "
            f"total={row['total_l1']:.6g} seg={row['seg_l1']:.6g} "
            f"pose={row['pose_l1']:.6g} rate={row['rate_l1']:.6g}"
        )
    lines.append("")
    return "\n".join(lines)
