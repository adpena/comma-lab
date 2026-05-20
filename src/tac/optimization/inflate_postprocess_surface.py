# SPDX-License-Identifier: MIT
"""Deterministic raw-output postprocess probes for inflate-surface search.

This module is intentionally scorer-free and archive-agnostic.  It applies
small, deterministic transforms to an already-inflated contest raw file so the
operator can cheaply test whether a transform class is worth implementing in a
real ``inflate.py`` runtime.  Positive advisory results are not promotion
evidence; transform parameters must still be represented by charged archive
bytes or by a generic runtime rule before exact eval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
DEFAULT_FRAME_COUNT = 1200

FrameSelector = Literal["all", "even", "odd"]
PostprocessKind = Literal["channel_bias", "temporal_blend"]


@dataclass(frozen=True)
class RawVideoShape:
    frames: int = DEFAULT_FRAME_COUNT
    height: int = CAMERA_H
    width: int = CAMERA_W
    channels: int = CHANNELS

    @property
    def frame_bytes(self) -> int:
        return self.height * self.width * self.channels

    @property
    def total_bytes(self) -> int:
        return self.frames * self.frame_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "frame_bytes": self.frame_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class PostprocessSpec:
    spec_id: str
    kind: PostprocessKind
    frame_selector: FrameSelector = "all"
    frame_indices: tuple[int, ...] = ()
    channel_deltas: tuple[int, int, int] = (0, 0, 0)
    alpha_num: int = 0
    alpha_den: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("frame_indices must be non-negative")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("frame_indices must be unique")

    def selected(self, frame_index: int) -> bool:
        if self.frame_indices:
            return frame_index in self.frame_indices
        if self.frame_selector == "all":
            return True
        if self.frame_selector == "even":
            return frame_index % 2 == 0
        if self.frame_selector == "odd":
            return frame_index % 2 == 1
        raise ValueError(f"unsupported frame selector: {self.frame_selector}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "kind": self.kind,
            "frame_selector": self.frame_selector,
            "frame_indices": list(self.frame_indices),
            "channel_deltas": list(self.channel_deltas),
            "alpha_num": self.alpha_num,
            "alpha_den": self.alpha_den,
            "notes": self.notes,
            "authority": {
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotion_blockers": [
                    "raw_postprocess_advisory_only",
                    "not_stock_inflate_runtime_custody",
                    "transform_parameters_not_charged_archive_bytes",
                    "exact_contest_eval_missing",
                ],
            },
        }


@dataclass(frozen=True)
class PostprocessResult:
    spec: PostprocessSpec
    input_raw: Path
    output_raw: Path
    shape: RawVideoShape
    input_sha256: str
    output_sha256: str
    changed_byte_count: int
    changed_frame_count: int
    max_abs_delta: int

    @property
    def passed_visible_change(self) -> bool:
        return self.changed_byte_count > 0 and self.input_sha256 != self.output_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "input_raw": str(self.input_raw.resolve()),
            "output_raw": str(self.output_raw.resolve()),
            "shape": self.shape.as_dict(),
            "input_raw_sha256": self.input_sha256,
            "output_raw_sha256": self.output_sha256,
            "changed_byte_count": self.changed_byte_count,
            "changed_frame_count": self.changed_frame_count,
            "max_abs_delta": self.max_abs_delta,
            "passed_visible_change": self.passed_visible_change,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def builtin_specs() -> dict[str, PostprocessSpec]:
    """Return small deterministic transform probes for inflate-surface search."""

    specs = [
        PostprocessSpec(
            spec_id="odd_luma_bias_m1",
            kind="channel_bias",
            frame_selector="odd",
            channel_deltas=(-1, -1, -1),
            notes="Subtract one RGB level from odd frames only.",
        ),
        PostprocessSpec(
            spec_id="odd_luma_bias_p1",
            kind="channel_bias",
            frame_selector="odd",
            channel_deltas=(1, 1, 1),
            notes="Add one RGB level to odd frames only.",
        ),
        PostprocessSpec(
            spec_id="even_luma_bias_m1",
            kind="channel_bias",
            frame_selector="even",
            channel_deltas=(-1, -1, -1),
            notes="Subtract one RGB level from even frames only.",
        ),
        PostprocessSpec(
            spec_id="all_luma_bias_m1",
            kind="channel_bias",
            frame_selector="all",
            channel_deltas=(-1, -1, -1),
            notes="Subtract one RGB level from all frames.",
        ),
        PostprocessSpec(
            spec_id="odd_temporal_blend_a1_8",
            kind="temporal_blend",
            frame_selector="odd",
            alpha_num=1,
            alpha_den=8,
            notes="Blend odd frames 12.5% toward the average of neighboring frames.",
        ),
    ]
    return {spec.spec_id: spec for spec in specs}


def postprocess_spec_from_dict(payload: dict[str, Any]) -> PostprocessSpec:
    """Parse a JSON-safe spec payload into ``PostprocessSpec``."""

    return PostprocessSpec(
        spec_id=str(payload["spec_id"]),
        kind=payload["kind"],
        frame_selector=payload.get("frame_selector", "all"),
        frame_indices=tuple(int(value) for value in payload.get("frame_indices", ())),
        channel_deltas=tuple(int(value) for value in payload.get("channel_deltas", (0, 0, 0))),  # type: ignore[arg-type]
        alpha_num=int(payload.get("alpha_num", 0)),
        alpha_den=int(payload.get("alpha_den", 1)),
        notes=str(payload.get("notes", "")),
    )


def get_builtin_spec(spec_id: str) -> PostprocessSpec:
    specs = builtin_specs()
    try:
        return specs[spec_id]
    except KeyError as exc:
        raise ValueError(f"unknown postprocess spec_id: {spec_id}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_size(path: Path, shape: RawVideoShape) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != shape.total_bytes:
        raise ValueError(
            f"raw size mismatch for {path}: expected {shape.total_bytes}, got {size}"
        )


def apply_postprocess(
    *,
    input_raw: Path,
    output_raw: Path,
    spec: PostprocessSpec,
    shape: RawVideoShape = RawVideoShape(),
) -> PostprocessResult:
    """Apply ``spec`` to ``input_raw`` and write a full-size raw output."""

    input_raw = input_raw.resolve()
    output_raw = output_raw.resolve()
    validate_raw_size(input_raw, shape)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    if output_raw.exists():
        output_raw.unlink()
    output_raw.write_bytes(b"")
    with output_raw.open("wb") as handle:
        handle.truncate(shape.total_bytes)

    source = np.memmap(
        input_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )
    dest = np.memmap(
        output_raw,
        dtype=np.uint8,
        mode="r+",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )

    changed_byte_count = 0
    changed_frame_count = 0
    max_abs_delta = 0
    for frame_index in range(shape.frames):
        before = source[frame_index]
        after = _apply_frame(source, frame_index, spec)
        dest[frame_index] = after
        diff = after.astype(np.int16) - before.astype(np.int16)
        changed = int(np.count_nonzero(diff))
        if changed:
            changed_frame_count += 1
            changed_byte_count += changed
            frame_max = int(np.max(np.abs(diff)))
            max_abs_delta = max(max_abs_delta, frame_max)
    dest.flush()
    del dest
    del source

    return PostprocessResult(
        spec=spec,
        input_raw=input_raw,
        output_raw=output_raw,
        shape=shape,
        input_sha256=sha256_file(input_raw),
        output_sha256=sha256_file(output_raw),
        changed_byte_count=changed_byte_count,
        changed_frame_count=changed_frame_count,
        max_abs_delta=max_abs_delta,
    )


def _apply_frame(source: np.memmap, frame_index: int, spec: PostprocessSpec) -> np.ndarray:
    frame = np.asarray(source[frame_index])
    if not spec.selected(frame_index):
        return frame.copy()
    if spec.kind == "channel_bias":
        deltas = np.asarray(spec.channel_deltas, dtype=np.int16).reshape(1, 1, 3)
        return np.clip(frame.astype(np.int16) + deltas, 0, 255).astype(np.uint8)
    if spec.kind == "temporal_blend":
        if spec.alpha_den <= 0 or not (0 <= spec.alpha_num <= spec.alpha_den):
            raise ValueError("temporal_blend requires 0 <= alpha_num <= alpha_den")
        prev_i = max(0, frame_index - 1)
        next_i = min(source.shape[0] - 1, frame_index + 1)
        neighbor = (
            source[prev_i].astype(np.int16) + source[next_i].astype(np.int16) + 1
        ) // 2
        num = int(spec.alpha_num)
        den = int(spec.alpha_den)
        blended = ((den - num) * frame.astype(np.int16) + num * neighbor + den // 2) // den
        return np.clip(blended, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported postprocess kind: {spec.kind}")


def plan_payload() -> dict[str, Any]:
    return {
        "schema": "inflate_postprocess_surface_plan.v1",
        "shape": RawVideoShape().as_dict(),
        "builtin_specs": [spec.as_dict() for spec in builtin_specs().values()],
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": (
                "Raw postprocess probes are advisory only. A positive probe must be "
                "converted into a deterministic inflate.py/runtime implementation "
                "with charged archive parameters before exact contest eval."
            ),
        },
    }
