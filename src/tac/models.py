"""Pydantic data models for structured results throughout tac.

Every function that returns structured data should return one of these
models, not a raw dict. This ensures type safety, validation, and
self-documenting return types.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    """Result from canonical_score or proxy evaluation."""

    score: float = Field(description="Final score: 100*seg + sqrt(10*pose) + 25*rate")
    pose: float = Field(description="Average PoseNet distortion (MSE on first 6 outputs)")
    seg: float = Field(description="Average SegNet distortion (argmax disagreement fraction)")
    rate: float = Field(description="Compression rate (archive_size / uncompressed_size)")
    rate_contribution: float = Field(description="25 * rate")
    pose_contribution: float = Field(description="sqrt(10 * pose)")
    seg_contribution: float = Field(description="100 * seg")
    n_samples: int = Field(description="Number of frame pairs evaluated")
    archive: str = Field(default="", description="Path to archive used")
    checkpoint: str = Field(default="", description="Path to checkpoint used")


class CheckpointMeta(BaseModel):
    """Metadata for a saved checkpoint."""

    epoch: int
    scorer: float
    fp32_path: str = ""
    int8_path: str = ""
    int8_size: int = 0
    meta_path: str = ""
    meta: dict = Field(default_factory=dict)


class AveragedCheckpoint(BaseModel):
    """Result from top-K checkpoint averaging."""

    source_epochs: list[int]
    source_scorers: list[float]
    avg_scorer: float
    int8_path: str
    int8_size: int


class SensitivityResult(BaseModel):
    """Score sensitivity analysis result."""

    pose_marginal: float = Field(description="d(score)/d(pose) at current operating point")
    seg_marginal: float = Field(description="d(score)/d(seg) — always 100")
    rate_marginal: float = Field(description="d(score)/d(rate) — always 25")
    seg_leverage: float = Field(description="seg_marginal / pose_marginal")
