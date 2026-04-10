"""Resume-state loaders for best-checkpoint and final-metadata sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .quantization import QuantizationMetadata


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _best_meta_sibling(path: Path) -> Path:
    name = path.name
    if name.endswith("_final_meta.json"):
        return path.with_name(name.replace("_final_meta.json", "_best_meta.json"))
    if name.endswith("_best_meta.json"):
        return path
    raise ValueError(f"Could not infer best-meta sibling for {path}")


def _final_meta_sibling(path: Path) -> Path:
    name = path.name
    if name.endswith("_best_meta.json"):
        return path.with_name(name.replace("_best_meta.json", "_final_meta.json"))
    if name.endswith("_final_meta.json"):
        return path
    raise ValueError(f"Could not infer final-meta sibling for {path}")


@dataclass(frozen=True)
class FinalMetadata:
    """Normalized final-metadata sidecar with optional best-checkpoint payload."""

    source_path: Path
    final_meta_path: Path
    tag: str | None
    fp32_path: Path
    int8_path: Path
    int8_size: int | None
    baseline_loss: float | None
    final_loss: float | None
    final_pose: float | None
    final_seg: float | None
    meta: dict[str, object]
    best_meta_path: Path | None = None
    best_checkpoint: QuantizationMetadata | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "FinalMetadata":
        source_path = Path(path)
        payload = _read_json(source_path)

        best_meta_path = _optional_path(payload.get("best_meta_path"))
        if best_meta_path is None and source_path.name.endswith("_final_meta.json"):
            best_meta_path = _best_meta_sibling(source_path)

        best_checkpoint = None
        if best_meta_path is not None and best_meta_path.exists():
            best_checkpoint = QuantizationMetadata.from_path(best_meta_path)
        else:
            best_eval = payload.get("best_eval")
            if isinstance(best_eval, Mapping):
                nested_best_meta_path = best_meta_path if best_meta_path is not None else _best_meta_sibling(source_path)
                best_checkpoint = QuantizationMetadata.from_payload(
                    nested_best_meta_path,
                    best_eval,
                    best_meta_path=nested_best_meta_path,
                )

        fp32_path = _optional_path(payload.get("fp32_path"))
        int8_path = _optional_path(payload.get("int8_path"))
        if fp32_path is None or int8_path is None:
            raise KeyError(f"Final metadata must include fp32_path and int8_path: {source_path}")

        meta = dict(payload.get("meta", {})) if isinstance(payload.get("meta"), dict) else {}
        return cls(
            source_path=source_path,
            final_meta_path=source_path,
            tag=None if payload.get("tag") is None else str(payload["tag"]),
            fp32_path=fp32_path,
            int8_path=int8_path,
            int8_size=_optional_int(payload.get("int8_size")),
            baseline_loss=_optional_float(payload.get("baseline_loss")),
            final_loss=_optional_float(payload.get("final_loss")),
            final_pose=_optional_float(payload.get("final_pose")),
            final_seg=_optional_float(payload.get("final_seg")),
            meta=meta,
            best_meta_path=best_meta_path,
            best_checkpoint=best_checkpoint,
        )


@dataclass(frozen=True)
class ResumeState:
    """Best available checkpoint/final sidecar state for a remote relaunch."""

    source_path: Path
    checkpoint: QuantizationMetadata | None = None
    final: FinalMetadata | None = None

    @property
    def best_meta_path(self) -> Path | None:
        if self.checkpoint is not None:
            return self.checkpoint.best_meta_path
        if self.final is not None:
            return self.final.best_meta_path
        return None

    @property
    def final_meta_path(self) -> Path | None:
        if self.final is not None:
            return self.final.final_meta_path
        return None

    @property
    def preferred_checkpoint(self) -> QuantizationMetadata | None:
        if self.checkpoint is not None:
            return self.checkpoint
        if self.final is not None:
            return self.final.best_checkpoint
        return None

    @property
    def best_checkpoint(self) -> QuantizationMetadata | None:
        return self.preferred_checkpoint

    @property
    def preferred_fp32_path(self) -> Path | None:
        checkpoint = self.preferred_checkpoint
        if checkpoint is not None and checkpoint.fp32_path is not None:
            return checkpoint.fp32_path
        if self.final is not None:
            return self.final.fp32_path
        return None

    @property
    def preferred_int8_path(self) -> Path | None:
        checkpoint = self.preferred_checkpoint
        if checkpoint is not None:
            return checkpoint.int8_path
        if self.final is not None:
            return self.final.int8_path
        return None

    @classmethod
    def from_path(cls, path: str | Path) -> "ResumeState":
        source_path = Path(path)
        if source_path.is_dir():
            return cls.from_directory(source_path)

        if source_path.name.endswith("_final_meta.json"):
            final = FinalMetadata.from_path(source_path)
            checkpoint = final.best_checkpoint
            if checkpoint is None:
                best_meta_path = final.best_meta_path
                if best_meta_path is not None and best_meta_path.exists():
                    checkpoint = QuantizationMetadata.from_path(best_meta_path)
            return cls(source_path=source_path, checkpoint=checkpoint, final=final)

        if source_path.name.endswith("_best_meta.json") or source_path.suffix == ".pt":
            checkpoint = QuantizationMetadata.from_path(source_path)
            final_meta_path = None
            if checkpoint.best_meta_path is not None:
                final_meta_path = _final_meta_sibling(checkpoint.best_meta_path)
            final = FinalMetadata.from_path(final_meta_path) if final_meta_path is not None and final_meta_path.exists() else None
            return cls(source_path=source_path, checkpoint=checkpoint, final=final)

        raise ValueError(f"Unsupported resume-state path: {source_path}")

    @classmethod
    def from_directory(cls, directory: str | Path, *, slug: str | None = None) -> "ResumeState":
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)

        def _pick(paths: list[Path]) -> Path | None:
            if slug is not None:
                for candidate in paths:
                    if candidate.name.startswith(slug + "_") or candidate.name == f"{slug}_best_meta.json" or candidate.name == f"{slug}_final_meta.json":
                        return candidate
                return None
            return sorted(paths)[0] if paths else None

        best_paths = list(root.glob("*_best_meta.json"))
        final_paths = list(root.glob("*_final_meta.json"))
        best_path = _pick(best_paths)
        final_path = _pick(final_paths)

        final = FinalMetadata.from_path(final_path) if final_path is not None else None
        checkpoint = final.best_checkpoint if final is not None and final.best_checkpoint is not None else None
        if checkpoint is None and best_path is not None:
            checkpoint = QuantizationMetadata.from_path(best_path)
        if final is None and checkpoint is not None and checkpoint.best_meta_path is not None:
            final_candidate = _final_meta_sibling(checkpoint.best_meta_path)
            final = FinalMetadata.from_path(final_candidate) if final_candidate.exists() else None

        return cls(source_path=root, checkpoint=checkpoint, final=final)
