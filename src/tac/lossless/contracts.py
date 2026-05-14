# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LosslessCompressionResult:
    profile: str
    archive_path: str
    archive_bytes: int
    original_bytes: int
    compression_rate: float
    method: str
    payload_bytes: int | None = None
    record_count: int | None = None
    checked_items: int | None = None
    split: tuple[str, ...] | None = None
    evidence_root: str | None = None

    def __post_init__(self) -> None:
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be non-negative")
        if self.original_bytes <= 0:
            raise ValueError("original_bytes must be positive")
        if self.compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        if self.payload_bytes is not None and self.payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("record_count must be non-negative")
        if self.checked_items is not None and self.checked_items < 0:
            raise ValueError("checked_items must be non-negative")
        if self.split is not None and not isinstance(self.split, tuple):
            raise ValueError("split must be a tuple of strings when provided")


@dataclass(frozen=True)
class LosslessVerificationResult:
    exact_match: bool
    checked_items: int
    mismatch_count: int

    def __post_init__(self) -> None:
        if self.checked_items < 0:
            raise ValueError("checked_items must be non-negative")
        if self.mismatch_count < 0:
            raise ValueError("mismatch_count must be non-negative")
