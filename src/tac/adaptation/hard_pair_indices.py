# SPDX-License-Identifier: MIT
"""Shared hard-pair index parsing for scorer-tail training loops."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class HardPairIndicesError(ValueError):
    """Raised when hard-pair indices would be ambiguous or unsafe."""


PAIR_INDEX_KEYS = (
    "prioritized_pair_indices",
    "hard_pair_indices",
    "vulnerable_pair_indices",
    "protected_pair_indices",
    "pair_indices",
    "pairs",
)
_NONNEGATIVE_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")


def normalize_pair_indices(
    value: Sequence[Any] | str | None,
    *,
    field: str = "pair_indices",
) -> tuple[int, ...]:
    """Return ordered unique non-negative pair indices.

    Duplicate hard-pair discoveries should not over-weight a pair unless a
    trainer explicitly implements weights, so this helper de-duplicates while
    preserving the first-seen order.
    """

    if value is None:
        return ()
    if isinstance(value, str):
        return parse_pair_indices_csv(value, field=field)
    if isinstance(value, (bytes, bytearray)):
        raise HardPairIndicesError(f"{field} must be text or a sequence of integers")
    if not isinstance(value, Sequence):
        raise HardPairIndicesError(f"{field} must be a sequence of integers")
    out: list[int] = []
    seen: set[int] = set()
    for raw in value:
        parsed = _parse_pair_index(raw, field=field)
        if parsed not in seen:
            seen.add(parsed)
            out.append(parsed)
    return tuple(out)


def parse_pair_indices_csv(
    value: str | None,
    *,
    field: str = "pair_indices",
) -> tuple[int, ...]:
    """Parse a comma-separated pair-index list."""

    text = str(value or "").strip()
    if not text:
        return ()
    return normalize_pair_indices(
        [part.strip() for part in text.split(",") if part.strip()],
        field=field,
    )


def merge_pair_indices(*groups: Sequence[Any] | str | None) -> tuple[int, ...]:
    """Merge ordered pair-index groups without duplicate weighting."""

    out: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for pair_idx in normalize_pair_indices(group):
            if pair_idx not in seen:
                seen.add(pair_idx)
                out.append(pair_idx)
    return tuple(out)


def validate_pair_indices_in_range(
    pair_indices: Sequence[Any] | str | None,
    *,
    num_pairs: int,
    field: str = "pair_indices",
) -> tuple[int, ...]:
    """Validate pair ids against the pair count that the trainer will decode."""

    if num_pairs <= 0:
        raise HardPairIndicesError(f"num_pairs must be positive; got {num_pairs}")
    normalized = normalize_pair_indices(pair_indices, field=field)
    invalid = [pair_idx for pair_idx in normalized if pair_idx >= num_pairs]
    if invalid:
        raise HardPairIndicesError(
            f"{field} contains out-of-range pair indices {invalid}; "
            f"expected 0 <= pair_idx < {num_pairs}"
        )
    return normalized


def pair_indices_from_mapping(mapping: Mapping[str, Any]) -> tuple[int, ...]:
    """Extract the first non-empty hard-pair list from common feedback schemas."""

    containers: list[Mapping[str, Any]] = []
    hard_pair = mapping.get("hard_pair_coverage")
    if isinstance(hard_pair, Mapping):
        containers.append(hard_pair)
    gate = mapping.get("sample_generalization_gate")
    if isinstance(gate, Mapping):
        nested = gate.get("hard_pair_coverage")
        if isinstance(nested, Mapping):
            containers.append(nested)
        containers.append(gate)
    containers.append(mapping)
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for container_idx, container in enumerate(containers):
        for key in PAIR_INDEX_KEYS:
            value = container.get(key)
            if value is None:
                continue
            normalized = normalize_pair_indices(value, field=key)
            if normalized:
                candidates.append((f"container{container_idx}.{key}", normalized))
    if not candidates:
        return ()
    selected_source, selected = candidates[0]
    conflicts = [
        source for source, value in candidates[1:] if tuple(value) != tuple(selected)
    ]
    if conflicts:
        raise HardPairIndicesError(
            "conflicting hard-pair index sources in mapping: "
            f"selected {selected_source}, conflicting {', '.join(conflicts)}"
        )
    return selected


def _parse_pair_index(raw: Any, *, field: str) -> int:
    if isinstance(raw, bool):
        raise HardPairIndicesError(f"{field} contains boolean {raw!r}")
    if isinstance(raw, int):
        if raw < 0:
            raise HardPairIndicesError(
                f"{field} contains invalid negative integer {raw!r}"
            )
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if _NONNEGATIVE_INT_RE.fullmatch(text):
            return int(text)
    raise HardPairIndicesError(
        f"{field} contains invalid non-negative integer {raw!r}"
    )


def load_pair_indices_file(
    path: str | Path | None,
    *,
    base: str | Path | None = None,
    field: str = "pair_indices_file",
) -> tuple[int, ...]:
    """Load hard-pair indices from JSON or simple CSV/text files."""

    if path is None:
        return ()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base is not None:
        resolved = Path(base).expanduser().resolve(strict=False) / resolved
    resolved = resolved.resolve(strict=False)
    if not resolved.is_file():
        raise HardPairIndicesError(f"{field} does not exist: {resolved}")
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        csv_text = ",".join(part for part in text.replace(",", " ").split() if part)
        return parse_pair_indices_csv(csv_text, field=field)
    if isinstance(payload, Mapping):
        normalized = pair_indices_from_mapping(payload)
        if normalized:
            return normalized
        raise HardPairIndicesError(
            f"{field} mapping must contain one of {', '.join(PAIR_INDEX_KEYS)}"
        )
    return normalize_pair_indices(payload, field=field)


__all__ = [
    "PAIR_INDEX_KEYS",
    "HardPairIndicesError",
    "load_pair_indices_file",
    "merge_pair_indices",
    "normalize_pair_indices",
    "pair_indices_from_mapping",
    "parse_pair_indices_csv",
    "validate_pair_indices_in_range",
]
