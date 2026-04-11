"""Quantization metadata loaders for best-meta and int8 sidecar artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _score_value(payload: Mapping[str, object]) -> float | None:
    value = payload.get("scorer", payload.get("score"))
    return None if value is None else float(value)


def _best_meta_candidates(int8_path: Path) -> tuple[Path, ...]:
    name = int8_path.name
    candidates: list[Path] = []
    if name.endswith("_best_int8.pt"):
        candidates.append(int8_path.with_name(name.replace("_best_int8.pt", "_best_meta.json")))
    if name.endswith("_int8.pt"):
        candidates.append(int8_path.with_name(name.replace("_int8.pt", "_best_meta.json")))
    candidates.append(int8_path.with_suffix(".json"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return tuple(deduped)


def _resolve_best_meta_path(source_path: Path) -> Path | None:
    if source_path.suffix == ".json":
        return source_path
    for candidate in _best_meta_candidates(source_path):
        if candidate.exists():
            return candidate
    return None


def _resolve_int8_path(source_path: Path, payload: Mapping[str, object]) -> Path:
    payload_path = payload.get("int8_path")
    if isinstance(payload_path, str) and payload_path:
        return Path(payload_path)
    if source_path.suffix == ".pt":
        return source_path
    if source_path.name.endswith("_best_meta.json"):
        return source_path.with_name(source_path.name.replace("_best_meta.json", "_best_int8.pt"))
    raise ValueError(f"Could not infer int8 artifact path from {source_path}")


@dataclass(frozen=True)
class QuantizationMetadata:
    """Normalized view of quantized post-filter artifact metadata."""

    source_path: Path
    best_meta_path: Path | None
    int8_path: Path
    fp32_path: Path | None = None
    epoch: int | None = None
    scorer: float | None = None
    int8_size: int | None = None
    variant: str | None = None
    hidden: int | None = None
    kernel: int | None = None
    alpha: float | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "QuantizationMetadata":
        source_path = Path(path)
        best_meta_path = _resolve_best_meta_path(source_path)
        payload: dict[str, object] = {}
        if best_meta_path is not None:
            payload = json.loads(best_meta_path.read_text())
        return cls.from_payload(source_path, payload, best_meta_path=best_meta_path)

    @classmethod
    def from_payload(
        cls,
        source_path: str | Path,
        payload: Mapping[str, object],
        *,
        best_meta_path: str | Path | None = None,
    ) -> "QuantizationMetadata":
        source_path = Path(source_path)
        best_meta = None if best_meta_path is None else Path(best_meta_path)
        meta = dict(payload.get("meta", {})) if isinstance(payload.get("meta"), dict) else {}
        int8_path = _resolve_int8_path(source_path, payload)
        fp32_path_value = payload.get("fp32_path")
        fp32_path = Path(fp32_path_value) if isinstance(fp32_path_value, str) and fp32_path_value else None

        int8_size = payload.get("int8_size")
        if int8_size is None and int8_path.exists():
            int8_size = int8_path.stat().st_size

        epoch = payload.get("epoch", payload.get("iteration"))
        return cls(
            source_path=source_path,
            best_meta_path=best_meta,
            int8_path=int8_path,
            fp32_path=fp32_path,
            epoch=None if epoch is None else int(epoch),
            scorer=_score_value(payload),
            int8_size=None if int8_size is None else int(int8_size),
            variant=None if meta.get("variant") is None else str(meta["variant"]),
            hidden=None if meta.get("hidden") is None else int(meta["hidden"]),
            kernel=None if meta.get("kernel") is None else int(meta["kernel"]),
            alpha=None if meta.get("alpha") is None else float(meta["alpha"]),
            meta=meta,
        )
