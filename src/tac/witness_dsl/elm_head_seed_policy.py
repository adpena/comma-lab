# SPDX-License-Identifier: MIT
"""Frozen typed policy for the resumable ELM/INR affine-head seed.

The head solver is intentionally not configurable through loose trainer-style flags.  Every
mathematical or resource-scope value is carried through :class:`Provenanced`, validated here,
and compiled to an immutable value object before the solver opens its multi-gigabyte inputs.
The three input digests are part of the policy rather than observations recorded after the
fact: execution recomputes each digest and refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tac.witness_dsl.typed_config import Provenanced  # noqa: TC001 - pydantic resolves at runtime

__all__ = [
    "CompiledElmHeadSeedPolicy",
    "ElmHeadSeedPolicy",
    "ElmHeadSeedScope",
    "compile_elm_head_seed_policy",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ElmHeadSeedScope(StrEnum):
    """Whether a policy covers the canonical P=600 fit or a non-promotable slice."""

    FULL_P600 = "full_p600"
    DIAGNOSTIC = "diagnostic"


def _provenanced_number(field: Provenanced, name: str) -> float:
    value = field.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}.value must be a number, got {value!r}")
    return float(value)


def _provenanced_int(field: Provenanced, name: str) -> int:
    value = field.value
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}.value must be an integer, got {value!r}")
    return int(value)


class ElmHeadSeedPolicy(BaseModel):
    """Frozen, extra-forbidden authoring surface for one ELM head-seed run."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    policy_schema: Literal["elm_head_seed_policy.v1"] = Field(
        default="elm_head_seed_policy.v1",
        alias="schema",
    )
    ridge: Provenanced
    pinv_rcond: Provenanced
    label_smoothing: Provenanced
    target_temperature: Provenanced
    grid_rows: Provenanced
    grid_cols: Provenanced
    pixel_chunk: Provenanced
    scope: Provenanced
    pair_limit: Provenanced | None = None
    source_checkpoint_sha256: str
    feature_state_sha256: str
    labels_sha256: str

    @model_validator(mode="after")
    def _validate_policy(self) -> ElmHeadSeedPolicy:
        knobs = {
            "ridge": self.ridge,
            "pinv_rcond": self.pinv_rcond,
            "label_smoothing": self.label_smoothing,
            "target_temperature": self.target_temperature,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "pixel_chunk": self.pixel_chunk,
            "scope": self.scope,
        }
        if self.pair_limit is not None:
            knobs["pair_limit"] = self.pair_limit
        for name, knob in knobs.items():
            if not knob.source.strip():
                raise ValueError(f"{name}.source must cite a law, artifact, or measurement")

        ridge = _provenanced_number(self.ridge, "ridge")
        rcond = _provenanced_number(self.pinv_rcond, "pinv_rcond")
        smoothing = _provenanced_number(self.label_smoothing, "label_smoothing")
        temperature = _provenanced_number(self.target_temperature, "target_temperature")
        rows = _provenanced_int(self.grid_rows, "grid_rows")
        cols = _provenanced_int(self.grid_cols, "grid_cols")
        chunk = _provenanced_int(self.pixel_chunk, "pixel_chunk")
        if ridge < 0.0:
            raise ValueError("ridge.value must be >=0")
        if rcond <= 0.0:
            raise ValueError("pinv_rcond.value must be >0")
        if not 0.0 < smoothing < 1.0:
            raise ValueError("label_smoothing.value must be strictly between 0 and 1")
        if temperature <= 0.0:
            raise ValueError("target_temperature.value must be >0")
        if rows < 1 or cols < 1:
            raise ValueError("grid_rows/grid_cols values must be >=1")
        if chunk < 1:
            raise ValueError("pixel_chunk.value must be >=1")

        if not isinstance(self.scope.value, str):
            raise ValueError("scope.value must be a string")
        try:
            scope = ElmHeadSeedScope(self.scope.value)
        except ValueError as exc:
            raise ValueError(
                f"scope.value must be one of {[value.value for value in ElmHeadSeedScope]}"
            ) from exc
        if scope is ElmHeadSeedScope.FULL_P600:
            if self.pair_limit is not None:
                raise ValueError("full_p600 policy must omit pair_limit; canonical scope is exactly P=600")
        else:
            if self.pair_limit is None:
                raise ValueError("diagnostic policy requires a provenanced pair_limit")
            limit = _provenanced_int(self.pair_limit, "pair_limit")
            if not 1 <= limit < 600:
                raise ValueError("diagnostic pair_limit.value must be in [1,599]")

        for name in (
            "source_checkpoint_sha256",
            "feature_state_sha256",
            "labels_sha256",
        ):
            digest = getattr(self, name)
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
        return self


@dataclass(frozen=True)
class CompiledElmHeadSeedPolicy:
    """Immutable values and custody manifest produced by the typed policy compiler."""

    policy: ElmHeadSeedPolicy
    policy_path: Path
    policy_file_sha256: str
    policy_manifest_sha256: str
    manifest: dict[str, Any]
    ridge: float
    pinv_rcond: float
    label_smoothing: float
    target_temperature: float
    grid_shape: tuple[int, int]
    pixel_chunk: int
    scope: ElmHeadSeedScope
    pair_limit: int | None


def compile_elm_head_seed_policy(path: str | Path) -> CompiledElmHeadSeedPolicy:
    """Read, validate, and freeze a policy; no solver input is opened before this succeeds."""

    policy_path = Path(path).resolve()
    raw = policy_path.read_bytes()
    policy = ElmHeadSeedPolicy.model_validate_json(raw)
    manifest = policy.model_dump(mode="json", by_alias=True)
    manifest_json = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    scope = ElmHeadSeedScope(str(policy.scope.value))
    return CompiledElmHeadSeedPolicy(
        policy=policy,
        policy_path=policy_path,
        policy_file_sha256=hashlib.sha256(raw).hexdigest(),
        policy_manifest_sha256=hashlib.sha256(manifest_json).hexdigest(),
        manifest=manifest,
        ridge=_provenanced_number(policy.ridge, "ridge"),
        pinv_rcond=_provenanced_number(policy.pinv_rcond, "pinv_rcond"),
        label_smoothing=_provenanced_number(policy.label_smoothing, "label_smoothing"),
        target_temperature=_provenanced_number(policy.target_temperature, "target_temperature"),
        grid_shape=(
            _provenanced_int(policy.grid_rows, "grid_rows"),
            _provenanced_int(policy.grid_cols, "grid_cols"),
        ),
        pixel_chunk=_provenanced_int(policy.pixel_chunk, "pixel_chunk"),
        scope=scope,
        pair_limit=(
            None
            if policy.pair_limit is None
            else _provenanced_int(policy.pair_limit, "pair_limit")
        ),
    )
