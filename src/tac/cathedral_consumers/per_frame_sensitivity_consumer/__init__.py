# SPDX-License-Identifier: MIT
"""Cathedral consumer for master-gradient per-frame sensitivity decompositions.

This consumer ingests payloads produced by
``tools/build_master_gradient_frame_decomposition.py`` /
``tac.master_gradient_frame_decomposition`` and exposes them as safe
observability-only routing metadata. The signal is useful for bit allocators,
frame-level curricula, and scorer-surrogate sampling because it mirrors the
upstream scorer input topology:

* SegNet sees only the last frame of each non-overlapping seq_len=2 sample.
* PoseNet sees both frames in the sample.

The consumer never turns the decomposition into a score claim; it only presents
top-frame ordering and axis attribution to downstream planners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_frame_sensitivity_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

_PAYLOAD_KEYS = (
    "per_frame_decomposition",
    "master_gradient_frame_decomposition",
    "frame_decomposition",
)
_PATH_KEYS = (
    "per_frame_decomposition_json",
    "master_gradient_frame_decomposition_json",
    "frame_decomposition_json",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5.

    The decomposition is built from an existing per-pair tensor by a separate
    producer. There is no consumer-local cache to mutate when a new anchor
    lands.
    """

    _ = anchor


def _load_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in _PAYLOAD_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    for key in _PATH_KEYS:
        value = candidate.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, Mapping):
            return loaded
    return None


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4: expose top per-frame sensitivities safely."""

    if not isinstance(candidate, Mapping):
        return _no_signal("candidate is not a mapping")
    payload = _load_payload(candidate)
    if payload is None:
        return _no_signal("no per-frame master-gradient decomposition payload on candidate")

    top_frames_raw = payload.get("top_frames")
    if not isinstance(top_frames_raw, list) or not top_frames_raw:
        return _no_signal("per-frame decomposition payload missing non-empty top_frames")

    top_frames: list[Mapping[str, Any]] = [
        row for row in top_frames_raw if isinstance(row, Mapping)
    ]
    if not top_frames:
        return _no_signal("per-frame decomposition top_frames rows are malformed")

    topology = str(payload.get("topology", "unknown"))
    n_frames = int(payload.get("n_frames") or 0)
    n_pairs = int(payload.get("n_pairs") or 0)
    first = top_frames[0]
    recommended_frame_order = [
        int(row["frame_index"])
        for row in top_frames
        if isinstance(row.get("frame_index"), int)
    ]
    rationale = (
        "per-frame master-gradient decomposition available; "
        f"topology={topology}; n_pairs={n_pairs}; n_frames={n_frames}; "
        f"top_frame={first.get('frame_index')} total_l1={first.get('total_l1')}"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "per_frame_sensitivity_routing",
        "topology": topology,
        "n_pairs": n_pairs,
        "n_frames": n_frames,
        "recommended_frame_order": recommended_frame_order,
        "top_frames": top_frames,
    }


def _no_signal(reason: str) -> Mapping[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"per-frame sensitivity consumer: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "per_frame_sensitivity_absent",
    }
