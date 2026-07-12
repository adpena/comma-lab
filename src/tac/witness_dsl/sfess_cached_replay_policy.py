# SPDX-License-Identifier: MIT
"""Typed fail-closed policy for an isolated SFESS cached-objective replay.

This policy is deliberately separate from :mod:`scorer_gradient_policy`.  SFESS
optimizes a terminal, fixed-cardinality, black-box edit set; it does not emit a
frame-shaped SegNet input costate.  Consequently it can authorize lookups in a
sealed objective cache, but it can never be admitted as a live scorer-gradient
replacement.  Any attempted live use retains the existing ``full_teacher``
fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tac.sfess_cached_replay import cached_state_sha256

SFESS_CACHED_K_SUBSET_MODE = "sfess_cached_k_subset"
JsonScalar = str | int | float | bool | None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SFESSCacheStatFingerprint:
    """Mutation-relevant stat identity for one sealed replay input."""

    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> SFESSCacheStatFingerprint:
        stat = path.stat()
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )


class SFESSCacheCustody(BaseModel):
    """Immutable bytes for one terminal-objective replay input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["cache", "source_video"]
    path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("SFESS cache custody sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _reject_placeholder_path(self) -> SFESSCacheCustody:
        lowered = self.path.strip().lower()
        if not lowered or any(token in lowered for token in ("<value>", "tbd", "placeholder")):
            raise ValueError("SFESS cache custody path must be concrete")
        return self

    def verify_full(self) -> tuple[bool, str, SFESSCacheStatFingerprint | None]:
        path = Path(self.path)
        if not path.is_file():
            return False, f"SFESS cache artifact is missing: {path}", None
        before = SFESSCacheStatFingerprint.capture(path)
        if before.size_bytes != self.size_bytes:
            return False, "SFESS cache byte count changed", None
        actual_sha = _sha256_file(path)
        after = SFESSCacheStatFingerprint.capture(path)
        if before != after:
            return False, "SFESS cache changed while its SHA-256 was read", None
        if actual_sha != self.sha256:
            return False, "SFESS cache SHA-256 changed", None
        return True, "SFESS cache full SHA-256 verified", after

    def verify_stat(
        self, expected: SFESSCacheStatFingerprint | None
    ) -> tuple[bool, str, SFESSCacheStatFingerprint | None]:
        if expected is None:
            return False, "compiled SFESS cache stat fingerprint is missing", None
        path = Path(self.path)
        if not path.is_file():
            return False, f"SFESS cache artifact is missing: {path}", None
        current = SFESSCacheStatFingerprint.capture(path)
        if current != expected:
            return False, "SFESS cache stat fingerprint changed", current
        return True, "SFESS cache stat fingerprint verified", current


class SFESSObjectiveContext(BaseModel):
    """Immutable identity of the measured table and its authority joins."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_table_sha256: str
    measurement_receipt_sha256: str
    candidate_manifest_sha256: str
    fixture_archive_sha256: str
    fixture_authority_sha256: str
    source_video_sha256: str
    source_video_bytes: int = Field(gt=0)
    n_bits: int = Field(gt=0)
    state_count: int = Field(gt=0)
    mask_order: Literal["little_endian_bit_j_equals_index_shift_j"]
    axis: str = Field(min_length=1)

    @field_validator(
        "objective_table_sha256",
        "measurement_receipt_sha256",
        "candidate_manifest_sha256",
        "fixture_archive_sha256",
        "fixture_authority_sha256",
        "source_video_sha256",
    )
    @classmethod
    def _strong_sha(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("objective-context fingerprints must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _state_space_is_complete(self) -> SFESSObjectiveContext:
        if self.state_count != 1 << self.n_bits:
            raise ValueError("state_count must equal 2**n_bits for the sealed cached replay")
        if any(token in self.axis.lower() for token in ("contest-cpu", "contest-cuda")):
            raise ValueError("SFESS cached replay axis must remain local and non-promotable")
        return self

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SFESSCachedReplayPolicy(BaseModel):
    """Frozen control law for the six-bit, zero-scorer replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["sfess_cached_k_subset"]
    research_only: Literal[True]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]
    produces_costate: Literal[False]
    live_gradient_fallback: Literal["full_teacher"]
    cache_failure_action: Literal["refuse"]
    objective_context: SFESSObjectiveContext
    objective_context_fingerprint: str
    objective_table_custody: SFESSCacheCustody
    measurement_receipt_custody: SFESSCacheCustody
    candidate_manifest_custody: SFESSCacheCustody
    source_video_custody: SFESSCacheCustody
    k_values: tuple[int, ...]
    include_degenerate_k_controls: Literal[True]
    samples_per_gradient: int = Field(ge=2)
    eval_budget_per_k: int = Field(gt=0)
    seed: int = Field(ge=0)
    max_evidence_age_queries: int = Field(ge=0)
    comparison_noise_floor_s: float = Field(ge=0.0)
    initial_mask_rule: Literal["lowest_indices"]
    acceptance_rule: Literal["strict_improvement_beyond_registered_floor"]
    retention_rule: Literal["strict_gated_returned_state"]
    k_selection_status: Literal["post_hoc_exploratory"]
    control_variate_anchor: Literal["wijk_2024_five_sample_leave_one_out"]

    @field_validator("objective_context_fingerprint")
    @classmethod
    def _context_sha(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("objective_context_fingerprint must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _closed_contract(self) -> SFESSCachedReplayPolicy:
        if self.objective_context_fingerprint != self.objective_context.fingerprint():
            raise ValueError("objective_context_fingerprint does not match the context payload")
        custodies = {
            "objective table": (
                self.objective_table_custody,
                self.objective_context.objective_table_sha256,
                "cache",
            ),
            "measurement receipt": (
                self.measurement_receipt_custody,
                self.objective_context.measurement_receipt_sha256,
                "cache",
            ),
            "candidate manifest": (
                self.candidate_manifest_custody,
                self.objective_context.candidate_manifest_sha256,
                "cache",
            ),
            "source video": (
                self.source_video_custody,
                self.objective_context.source_video_sha256,
                "source_video",
            ),
        }
        for label, (custody, expected_sha, expected_kind) in custodies.items():
            if custody.kind != expected_kind:
                raise ValueError(f"{label} custody must have kind={expected_kind!r}")
            if custody.sha256 != expected_sha:
                raise ValueError(f"{label} custody SHA-256 does not match objective context")
        if self.source_video_custody.size_bytes != self.objective_context.source_video_bytes:
            raise ValueError("source video custody byte count does not match objective context")
        if not self.k_values:
            raise ValueError("k_values must contain at least one non-degenerate cardinality")
        if tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError("k_values must be unique and strictly increasing")
        if any(k <= 0 or k >= self.objective_context.n_bits for k in self.k_values):
            raise ValueError("k_values must exclude the degenerate 0 and n_bits controls")
        # FROM-LITERATURE: Wijk, Vinuesa, and Azizpour (2024), "Revisiting
        # Score Function Estimators for k-Subset Sampling", arXiv:2407.16058,
        # use five samples for SFESS with leave-one-out control variates.
        if self.samples_per_gradient != 5:
            raise ValueError("the registered Wijk-2024 replay control fixes samples_per_gradient=5")
        if self.eval_budget_per_k != self.objective_context.state_count:
            raise ValueError("matched replay budget must equal the existing 64-state arm budget")
        if self.max_evidence_age_queries != 0:
            raise ValueError("cached lookup evidence must be re-derived at the current query")
        if not math.isfinite(self.comparison_noise_floor_s):
            raise ValueError("comparison_noise_floor_s must be finite")
        return self

    def compile(self) -> CompiledSFESSCachedReplayPolicy:
        stats: dict[str, SFESSCacheStatFingerprint] = {}
        for role, custody in self._custodies().items():
            ok, reason, stat = custody.verify_full()
            if not ok or stat is None:
                raise ValueError(f"{role} custody failed full verification: {reason}")
            stats[role] = stat
        return CompiledSFESSCachedReplayPolicy(source=self, _provider_stats=stats)

    def _custodies(self) -> dict[str, SFESSCacheCustody]:
        return {
            "objective_table": self.objective_table_custody,
            "measurement_receipt": self.measurement_receipt_custody,
            "candidate_manifest": self.candidate_manifest_custody,
            "source_video": self.source_video_custody,
        }


@dataclass(frozen=True)
class SFESSCachedLookupDecision:
    """One cache-lookup authorization; it never authorizes a live gradient."""

    admitted_for_cached_lookup: bool
    live_gradient_admitted: bool
    fallback_to_full_teacher: bool
    reasons: tuple[str, ...]
    objective_context_fingerprint: str
    state_sha256: str | None
    query_index: int
    evidence_age_queries: int | None
    custody_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted_for_cached_lookup": self.admitted_for_cached_lookup,
            "live_gradient_admitted": self.live_gradient_admitted,
            "fallback_to_full_teacher": self.fallback_to_full_teacher,
            "reasons": list(self.reasons),
            "objective_context_fingerprint": self.objective_context_fingerprint,
            "state_sha256": self.state_sha256,
            "query_index": self.query_index,
            "evidence_age_queries": self.evidence_age_queries,
            "custody_checks": list(self.custody_checks),
        }


@dataclass
class CompiledSFESSCachedReplayPolicy:
    """Executable lookup gate with cheap per-decision mutation detection."""

    source: SFESSCachedReplayPolicy
    _provider_stats: dict[str, SFESSCacheStatFingerprint]

    def authorize_lookup(
        self,
        *,
        table_source_sha256: str,
        table_n_bits: int,
        mask: tuple[int, ...],
        value: float,
        declared_state_sha256: str,
        query_index: int,
        evidence_query_index: int,
        current_objective_context_fingerprint: str,
    ) -> SFESSCachedLookupDecision:
        """Re-derive custody, objective, state, and age at every cached lookup."""

        policy = self.source
        reasons: list[str] = []
        custody_checks: list[str] = []
        for role, custody in policy._custodies().items():
            ok, reason, _current = custody.verify_stat(self._provider_stats.get(role))
            custody_checks.append(f"{role}: {reason}")
            if not ok:
                reasons.append(f"{role} custody failed: {reason}")

        runtime_context_fingerprint = policy.objective_context.fingerprint()
        if runtime_context_fingerprint != policy.objective_context_fingerprint:
            reasons.append("objective context changed after compilation")
        if current_objective_context_fingerprint != runtime_context_fingerprint:
            reasons.append("current objective-context fingerprint mismatch")
        if table_source_sha256 != policy.objective_context.objective_table_sha256:
            reasons.append("objective-table provider fingerprint mismatch")
        if table_n_bits != policy.objective_context.n_bits:
            reasons.append("objective-table n_bits mismatch")
        try:
            mask_length = len(mask)
        except TypeError:
            mask_length = -1
        if mask_length != policy.objective_context.n_bits:
            reasons.append(
                "cached state mask length mismatch: "
                f"expected={policy.objective_context.n_bits}, actual={mask_length}"
            )

        valid_index = (
            isinstance(query_index, int)
            and not isinstance(query_index, bool)
            and query_index >= 0
        )
        valid_evidence_index = (
            isinstance(evidence_query_index, int)
            and not isinstance(evidence_query_index, bool)
            and evidence_query_index >= 0
        )
        evidence_age: int | None = None
        if not valid_index:
            reasons.append("query_index must be an integer >= 0")
        if not valid_evidence_index:
            reasons.append("evidence_query_index must be an integer >= 0")
        if valid_index and valid_evidence_index:
            evidence_age = query_index - evidence_query_index
            if evidence_age < 0:
                reasons.append("cached lookup evidence is from a future query")
            elif evidence_age > policy.max_evidence_age_queries:
                reasons.append(
                    "cached lookup evidence is stale: "
                    f"age={evidence_age} > {policy.max_evidence_age_queries}"
                )

        state_sha: str | None = None
        try:
            state_sha = cached_state_sha256(mask, float(value))
        except (TypeError, ValueError):
            reasons.append("cached state is malformed or nonfinite")
        if not _is_sha256(declared_state_sha256):
            reasons.append("declared state fingerprint is invalid")
        elif state_sha != declared_state_sha256:
            reasons.append("state fingerprint mismatch")
        if not math.isfinite(float(value)):
            reasons.append("cached objective value is nonfinite")

        admitted = not reasons
        return SFESSCachedLookupDecision(
            admitted_for_cached_lookup=admitted,
            live_gradient_admitted=False,
            fallback_to_full_teacher=True,
            reasons=tuple(reasons),
            objective_context_fingerprint=runtime_context_fingerprint,
            state_sha256=state_sha,
            query_index=query_index if valid_index else -1,
            evidence_age_queries=evidence_age,
            custody_checks=tuple(custody_checks),
        )


__all__ = [
    "SFESS_CACHED_K_SUBSET_MODE",
    "CompiledSFESSCachedReplayPolicy",
    "SFESSCacheCustody",
    "SFESSCacheStatFingerprint",
    "SFESSCachedLookupDecision",
    "SFESSCachedReplayPolicy",
    "SFESSObjectiveContext",
]
