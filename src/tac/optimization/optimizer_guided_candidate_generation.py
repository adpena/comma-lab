# SPDX-License-Identifier: MIT
"""Deterministic offline candidate generation for optimizer-guided sweeps.

This module is a reusable planning scaffold for low-dimensional score-lowering
searches: A1/PR101 inflate-time bias, PR95/HNeRV optimizer smokes, broader
representation-family training probes, and custom profile JSON supplied by
family-specific substrate builders. It emits ranked proxy rows only. It never
creates archives, dispatches GPU work, or claims exact scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tac.optimization.optimizer_training_signal_bridge import (
    DEFAULT_CANONICAL_EQUATION_REFS,
    DEFAULT_DETERMINISTIC_SOLUTION_REFS,
    DEFAULT_MASTER_GRADIENT_FEATURES,
    DEFAULT_PAIRED_MODES,
    DEFAULT_VARIANT_AXES,
    DEFAULT_XRAY_PRIMITIVES,
    build_optimizer_training_signal_wire_in,
)
from tac.optimization.parameter_group_lr_policy import (
    EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    build_parameter_group_lr_policy_fingerprint,
    canonical_json,
)
from tac.optimization.proxy_candidate_contract import apply_proxy_evidence_boundary

QUEUE_SCHEMA = "optimizer_guided_candidate_queue_v1"
PROFILE_SCHEMA = "optimizer_guided_candidate_profile_v1"
TOOL_NAME = "tools/build_optimizer_guided_candidate_queue.py"
DEFAULT_GENERATED_AT_UTC = "1970-01-01T00:00:00Z"
MAX_CANDIDATES = 10_000

EVIDENCE_SEMANTICS = "offline_optimizer_guided_proxy_queue_not_exact_auth_eval"
EVIDENCE_GRADE = "[offline-proxy-planning-only]"
BASE_DISPATCH_BLOCKERS = (
    "offline_optimizer_candidate_generation_only",
    "no_archive_zip_emitted",
    "no_inflate_runtime_emitted",
    "no_score_affecting_payload_change_proof",
    "no_contest_cuda_auth_eval",
    "provider_prefilter_must_preserve_proxy_boundary",
    "archive_builder_handoff_required_before_promotion",
)
OPTIMIZER_STATUSES = {
    "grid": "bounded_grid_stdlib",
    "random": "random_search_stdlib",
    "cmaes": "cmaes_style_stdlib",
    "optuna": "optuna_tpe_style_stdlib",
}


class CandidateGenerationError(ValueError):
    """Raised when a candidate-generation profile or request is invalid."""


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded search dimension."""

    name: str
    low: float
    high: float
    anchor: float
    kind: str = "float"
    step: float | None = None
    weight: float = 1.0
    runtime_slot: str | None = None
    description: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ParameterSpec:
        try:
            name = str(payload["name"])
            low = _finite_float(payload["low"], "low")
            high = _finite_float(payload["high"], "high")
            anchor = _finite_float(payload["anchor"], "anchor")
        except KeyError as exc:
            raise CandidateGenerationError(f"parameter missing {exc.args[0]}") from exc
        spec = cls(
            name=name,
            low=low,
            high=high,
            anchor=anchor,
            kind=str(payload.get("kind") or "float"),
            step=_optional_positive_float(payload.get("step"), "step"),
            weight=_positive_float(payload.get("weight", 1.0), "weight"),
            runtime_slot=_optional_str(payload.get("runtime_slot")),
            description=_optional_str(payload.get("description")),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise CandidateGenerationError(f"invalid parameter name: {self.name!r}")
        if self.kind not in {"float", "int"}:
            raise CandidateGenerationError(f"{self.name}: kind must be float or int")
        if self.high <= self.low:
            raise CandidateGenerationError(f"{self.name}: high must be greater than low")
        if not (self.low <= self.anchor <= self.high):
            raise CandidateGenerationError(f"{self.name}: anchor must be inside bounds")
        if self.step is not None and self.step <= 0:
            raise CandidateGenerationError(f"{self.name}: step must be positive")

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def coerce(self, value: float) -> float | int:
        clipped = min(self.high, max(self.low, value))
        if self.step is not None:
            steps = round((clipped - self.low) / self.step)
            clipped = self.low + steps * self.step
            clipped = min(self.high, max(self.low, clipped))
        if self.kind == "int":
            return round(clipped)
        return round(float(clipped), 12)

    def normalized_distance(self, value: float | int) -> float:
        if self.half_width <= 0:
            return 0.0
        return (float(value) - self.anchor) / self.half_width

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "name": self.name,
                "low": self.low,
                "high": self.high,
                "anchor": self.anchor,
                "kind": self.kind,
                "step": self.step,
                "weight": self.weight,
                "runtime_slot": self.runtime_slot,
                "description": self.description,
            }
        )


@dataclass(frozen=True)
class CandidateGenerationProfile:
    """Reusable low-dimensional search profile."""

    profile_id: str
    lane_id: str
    lane_class: str
    candidate_family: str
    representation_family: str
    substrate_family: str
    training_signal_kind: str
    param_schema: str
    candidate_prefix: str
    parameters: tuple[ParameterSpec, ...]
    score_lowering_hypothesis: str
    base_proxy_objective: float = 0.19285
    objective_scale: float = 0.00024
    bias_asymmetry_weight: float = 0.00036
    sidecar_l1_weight: float = 0.00004
    jitter_scale: float = 2.5e-8
    source_anchor: str | None = None
    dispatch_blockers: tuple[str, ...] = field(default_factory=tuple)
    canonical_equation_refs: tuple[str, ...] = DEFAULT_CANONICAL_EQUATION_REFS
    master_gradient_features: tuple[str, ...] = DEFAULT_MASTER_GRADIENT_FEATURES
    xray_primitives: tuple[str, ...] = DEFAULT_XRAY_PRIMITIVES
    deterministic_solution_refs: tuple[str, ...] = DEFAULT_DETERMINISTIC_SOLUTION_REFS
    variant_axes: tuple[str, ...] = DEFAULT_VARIANT_AXES
    paired_modes: tuple[str, ...] = DEFAULT_PAIRED_MODES

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CandidateGenerationProfile:
        parameters_raw = payload.get("parameters")
        if not isinstance(parameters_raw, list) or not parameters_raw:
            raise CandidateGenerationError("profile parameters must be a non-empty list")
        parameters = tuple(ParameterSpec.from_mapping(item) for item in parameters_raw)
        profile = cls(
            profile_id=str(payload.get("profile_id") or payload.get("profile") or ""),
            lane_id=str(payload.get("lane_id") or ""),
            lane_class=str(payload.get("lane_class") or ""),
            candidate_family=str(payload.get("candidate_family") or ""),
            representation_family=str(
                payload.get("representation_family")
                or payload.get("candidate_family")
                or "unspecified"
            ),
            substrate_family=str(
                payload.get("substrate_family")
                or payload.get("representation_family")
                or payload.get("candidate_family")
                or "unspecified"
            ),
            training_signal_kind=str(
                payload.get("training_signal_kind") or "optimizer_guided_proxy"
            ),
            param_schema=str(payload.get("param_schema") or ""),
            candidate_prefix=str(payload.get("candidate_prefix") or ""),
            parameters=parameters,
            score_lowering_hypothesis=str(payload.get("score_lowering_hypothesis") or ""),
            base_proxy_objective=_finite_float(
                payload.get("base_proxy_objective", 0.19285),
                "base_proxy_objective",
            ),
            objective_scale=_positive_float(
                payload.get("objective_scale", 0.00024),
                "objective_scale",
            ),
            bias_asymmetry_weight=_nonnegative_float(
                payload.get("bias_asymmetry_weight", 0.00036),
                "bias_asymmetry_weight",
            ),
            sidecar_l1_weight=_nonnegative_float(
                payload.get("sidecar_l1_weight", 0.00004),
                "sidecar_l1_weight",
            ),
            jitter_scale=_nonnegative_float(payload.get("jitter_scale", 2.5e-8), "jitter_scale"),
            source_anchor=_optional_str(payload.get("source_anchor")),
            dispatch_blockers=tuple(str(item) for item in payload.get("dispatch_blockers", []) if str(item)),
            canonical_equation_refs=_str_tuple_or_default(
                payload.get("canonical_equation_refs"),
                DEFAULT_CANONICAL_EQUATION_REFS,
            ),
            master_gradient_features=_str_tuple_or_default(
                payload.get("master_gradient_features"),
                DEFAULT_MASTER_GRADIENT_FEATURES,
            ),
            xray_primitives=_str_tuple_or_default(
                payload.get("xray_primitives"),
                DEFAULT_XRAY_PRIMITIVES,
            ),
            deterministic_solution_refs=_str_tuple_or_default(
                payload.get("deterministic_solution_refs"),
                DEFAULT_DETERMINISTIC_SOLUTION_REFS,
            ),
            variant_axes=_str_tuple_or_default(
                payload.get("variant_axes"),
                DEFAULT_VARIANT_AXES,
            ),
            paired_modes=_str_tuple_or_default(
                payload.get("paired_modes"),
                DEFAULT_PAIRED_MODES,
            ),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        for attr in (
            "profile_id",
            "lane_id",
            "lane_class",
            "candidate_family",
            "representation_family",
            "substrate_family",
            "training_signal_kind",
            "param_schema",
            "candidate_prefix",
            "score_lowering_hypothesis",
        ):
            if not getattr(self, attr):
                raise CandidateGenerationError(f"profile missing {attr}")
        names = [param.name for param in self.parameters]
        if len(names) != len(set(names)):
            raise CandidateGenerationError("profile parameter names must be unique")
        for attr in (
            "canonical_equation_refs",
            "master_gradient_features",
            "xray_primitives",
            "deterministic_solution_refs",
            "variant_axes",
            "paired_modes",
        ):
            if not getattr(self, attr):
                raise CandidateGenerationError(f"profile {attr} must be non-empty")

    @property
    def parameter_map(self) -> dict[str, ParameterSpec]:
        return {param.name: param for param in self.parameters}

    def anchor_params(self) -> dict[str, float | int]:
        return {param.name: param.coerce(param.anchor) for param in self.parameters}

    def as_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "schema": PROFILE_SCHEMA,
                "profile_id": self.profile_id,
                "lane_id": self.lane_id,
                "lane_class": self.lane_class,
                "candidate_family": self.candidate_family,
                "representation_family": self.representation_family,
                "substrate_family": self.substrate_family,
                "training_signal_kind": self.training_signal_kind,
                "param_schema": self.param_schema,
                "candidate_prefix": self.candidate_prefix,
                "parameters": [param.as_dict() for param in self.parameters],
                "score_lowering_hypothesis": self.score_lowering_hypothesis,
                "base_proxy_objective": self.base_proxy_objective,
                "objective_scale": self.objective_scale,
                "bias_asymmetry_weight": self.bias_asymmetry_weight,
                "sidecar_l1_weight": self.sidecar_l1_weight,
                "jitter_scale": self.jitter_scale,
                "source_anchor": self.source_anchor,
                "dispatch_blockers": list(self.dispatch_blockers),
                "canonical_equation_refs": list(self.canonical_equation_refs),
                "master_gradient_features": list(self.master_gradient_features),
                "xray_primitives": list(self.xray_primitives),
                "deterministic_solution_refs": list(self.deterministic_solution_refs),
                "variant_axes": list(self.variant_axes),
                "paired_modes": list(self.paired_modes),
            }
        )


@dataclass(frozen=True)
class CandidateDraft:
    """One unmaterialized optimizer proposal before queue-row wrapping."""

    candidate_id: str
    trial_index: int
    generation: int
    params: dict[str, float | int]
    proxy_objective: float
    proxy_components: dict[str, float]


def default_profiles() -> dict[str, CandidateGenerationProfile]:
    """Return built-in profiles for low-dimensional planning sweeps."""

    return {
        name: CandidateGenerationProfile.from_mapping(payload)
        for name, payload in _DEFAULT_PROFILE_PAYLOADS.items()
    }


def load_profile(profile: str = "pr101_bias_sidecar") -> CandidateGenerationProfile:
    """Load a built-in profile by name."""

    profiles = default_profiles()
    try:
        return profiles[profile]
    except KeyError as exc:
        raise CandidateGenerationError(
            f"unknown profile {profile!r}; expected one of {sorted(profiles)}"
        ) from exc


def profile_from_json(path: str | Any) -> CandidateGenerationProfile:
    """Load a custom profile JSON file."""

    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CandidateGenerationError("profile JSON must be an object")
    return CandidateGenerationProfile.from_mapping(payload)


def generate_candidate_queue(
    *,
    profile: CandidateGenerationProfile,
    optimizer: str,
    max_candidates: int,
    seed: int,
    top_k: int | None = None,
    generated_at_utc: str = DEFAULT_GENERATED_AT_UTC,
) -> dict[str, Any]:
    """Generate a deterministic planning-only ranked queue."""

    if optimizer not in OPTIMIZER_STATUSES:
        raise CandidateGenerationError(
            f"unknown optimizer {optimizer!r}; expected one of {sorted(OPTIMIZER_STATUSES)}"
        )
    if max_candidates < 1 or max_candidates > MAX_CANDIDATES:
        raise CandidateGenerationError(f"max_candidates must be in [1, {MAX_CANDIDATES}]")
    if top_k is not None and (top_k < 1 or top_k > MAX_CANDIDATES):
        raise CandidateGenerationError(f"top_k must be in [1, {MAX_CANDIDATES}]")

    optimizer_status = OPTIMIZER_STATUSES[optimizer]
    drafts = _generate_drafts(
        profile=profile,
        optimizer=optimizer,
        optimizer_status=optimizer_status,
        max_candidates=max_candidates,
        seed=seed,
    )
    rows = [
        _candidate_row(
            profile=profile,
            draft=draft,
            optimizer=optimizer,
            optimizer_status=optimizer_status,
            seed=seed,
        )
        for draft in drafts
    ]
    rows = sorted(rows, key=lambda row: (float(row["rank_score"]), str(row["candidate_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    emitted_rows = rows[:top_k] if top_k is not None else rows
    return _json_safe(
        {
            "schema": QUEUE_SCHEMA,
            "compatible_top_k_schema": "optimizer_candidate_queue_v1",
            "tool": TOOL_NAME,
            "generated_at_utc": generated_at_utc,
            "metadata_time_policy": "stable_by_default_for_seed_reproducibility",
            "profile": profile.profile_id,
            "profile_contract": profile.as_dict(),
            "optimizer": optimizer,
            "optimizer_status": optimizer_status,
            "seed": seed,
            "max_candidates": max_candidates,
            "n_candidates": len(rows),
            "top_k_count": len(emitted_rows),
            "dispatch_ready_count": 0,
            "dispatch_ready": [],
            "top_k": emitted_rows,
            "top_k_forensic": emitted_rows,
            "evidence_boundary": {
                "planning_only_by_default": True,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "proxy_objective_is_not_exact_score": True,
                "provider_agnostic": True,
                "offline_only": True,
                "next_gate": (
                    "materialize byte-closed archive/runtime custody, prove runtime "
                    "consumption, then use exact-readiness promotion before any dispatch"
                ),
            },
            "score_lowering_connection": {
                "hypothesis": profile.score_lowering_hypothesis,
                "how_this_lowers_score": (
                    "ranked params focus scarce local-training, prefilter, and "
                    "exact-eval attention on the declared representation/substrate "
                    "family before archive materialization"
                ),
                "not_a_score_claim": True,
            },
            "consumer_contract": {
                "modal_prefilters": "may consume top_k[*].candidate_params offline only",
                "kaggle_prefilters": "may seed private proxy kernels; keep proxy boundary",
                "m5_prefilters": "may perform CPU/macOS ranking; result remains advisory",
                "dispatch_actuators": "must require a separate exact-readiness queue",
            },
        }
    )


def _generate_drafts(
    *,
    profile: CandidateGenerationProfile,
    optimizer: str,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
) -> list[CandidateDraft]:
    if optimizer == "grid":
        return _generate_grid_drafts(profile, optimizer_status, max_candidates, seed)
    if optimizer == "random":
        return _generate_random_drafts(profile, optimizer_status, max_candidates, seed)
    if optimizer == "cmaes":
        return _generate_cmaes_drafts(profile, optimizer_status, max_candidates, seed)
    return _generate_optuna_style_drafts(profile, optimizer_status, max_candidates, seed)


def _generate_grid_drafts(
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
) -> list[CandidateDraft]:
    values_by_param: list[list[float | int]] = []
    for param in profile.parameters:
        candidates = [
            param.anchor,
            param.anchor - 0.25 * param.width,
            param.anchor + 0.25 * param.width,
            param.anchor - 0.50 * param.width,
            param.anchor + 0.50 * param.width,
        ]
        values_by_param.append(_ordered_unique_values(param.coerce(value) for value in candidates))

    drafts: list[CandidateDraft] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    trial_index = 0
    for combo in _bounded_product(values_by_param, limit=max(MAX_CANDIDATES, max_candidates * 16)):
        params = {param.name: value for param, value in zip(profile.parameters, combo, strict=True)}
        key = _param_key(params)
        if key in seen:
            continue
        seen.add(key)
        drafts.append(_draft(profile, optimizer_status, trial_index, 0, params, seed))
        trial_index += 1
    return sorted(drafts, key=lambda item: (item.proxy_objective, item.candidate_id))[:max_candidates]


def _generate_random_drafts(
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
) -> list[CandidateDraft]:
    rng = random.Random(seed)
    return _fill_with_sampler(
        profile=profile,
        optimizer_status=optimizer_status,
        max_candidates=max_candidates,
        seed=seed,
        sampler=lambda _seen, _rows: _sample_uniform(profile, rng),
    )


def _generate_cmaes_drafts(
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
) -> list[CandidateDraft]:
    rng = random.Random(seed)
    mean = {param.name: float(param.anchor) for param in profile.parameters}
    sigma = 0.42
    generation_size = max(4, 2 * len(profile.parameters))
    generation_rows: list[CandidateDraft] = []

    def sample(_seen: set[tuple[tuple[str, Any], ...]], rows: list[CandidateDraft]) -> dict[str, float | int]:
        nonlocal mean, sigma, generation_rows
        if generation_rows and len(generation_rows) >= generation_size:
            generation_rows = sorted(generation_rows, key=lambda item: item.proxy_objective)
            elite = generation_rows[: max(1, generation_size // 2)]
            mean = {
                param.name: (0.65 * mean[param.name])
                + (0.35 * statistics.fmean(float(row.params[param.name]) for row in elite))
                for param in profile.parameters
            }
            sigma = max(0.05, sigma * 0.82)
            generation_rows = []
        params = _sample_gaussian(profile, rng, mean=mean, sigma=sigma)
        if rows:
            generation_rows.append(rows[-1])
        return params

    rows = _fill_with_sampler(
        profile=profile,
        optimizer_status=optimizer_status,
        max_candidates=max_candidates,
        seed=seed,
        sampler=sample,
    )
    return rows


def _generate_optuna_style_drafts(
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
) -> list[CandidateDraft]:
    rng = random.Random(seed)
    warmup = min(max_candidates, max(4, 2 * len(profile.parameters)))

    def sample(_seen: set[tuple[tuple[str, Any], ...]], rows: list[CandidateDraft]) -> dict[str, float | int]:
        if len(rows) < warmup:
            return _sample_uniform(profile, rng)
        ranked = sorted(rows, key=lambda item: item.proxy_objective)
        elite = ranked[: max(2, len(ranked) // 4)]
        params: dict[str, float | int] = {}
        shrink = max(0.08, 0.35 * (1.0 - min(len(rows), max_candidates) / max_candidates))
        for param in profile.parameters:
            center = float(rng.choice(elite).params[param.name])
            params[param.name] = param.coerce(center + rng.gauss(0.0, shrink * param.width))
        return params

    return _fill_with_sampler(
        profile=profile,
        optimizer_status=optimizer_status,
        max_candidates=max_candidates,
        seed=seed,
        sampler=sample,
    )


def _fill_with_sampler(
    *,
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    max_candidates: int,
    seed: int,
    sampler: Any,
) -> list[CandidateDraft]:
    rows: list[CandidateDraft] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    def add(params: dict[str, float | int], generation: int = 0) -> None:
        if len(rows) >= max_candidates:
            return
        key = _param_key(params)
        if key in seen:
            return
        seen.add(key)
        rows.append(_draft(profile, optimizer_status, len(rows), generation, params, seed))

    add(profile.anchor_params())
    attempts = 0
    while len(rows) < max_candidates and attempts < max_candidates * 40:
        attempts += 1
        add(sampler(seen, rows), generation=attempts)
    return sorted(rows, key=lambda item: (item.proxy_objective, item.candidate_id))


def _sample_uniform(
    profile: CandidateGenerationProfile,
    rng: random.Random,
) -> dict[str, float | int]:
    return {
        param.name: param.coerce(rng.uniform(param.low, param.high))
        for param in profile.parameters
    }


def _sample_gaussian(
    profile: CandidateGenerationProfile,
    rng: random.Random,
    *,
    mean: Mapping[str, float],
    sigma: float,
) -> dict[str, float | int]:
    return {
        param.name: param.coerce(
            float(mean[param.name]) + rng.gauss(0.0, sigma * param.width)
        )
        for param in profile.parameters
    }


def _draft(
    profile: CandidateGenerationProfile,
    optimizer_status: str,
    trial_index: int,
    generation: int,
    params: dict[str, float | int],
    seed: int,
) -> CandidateDraft:
    objective, components = proxy_objective(profile, params, seed=seed)
    suffix = "anchor" if _param_key(params) == _param_key(profile.anchor_params()) else f"{trial_index:04d}"
    return CandidateDraft(
        candidate_id=f"{profile.candidate_prefix}_{optimizer_status}_{suffix}",
        trial_index=trial_index,
        generation=generation,
        params=params,
        proxy_objective=objective,
        proxy_components=components,
    )


def proxy_objective(
    profile: CandidateGenerationProfile,
    params: Mapping[str, float | int],
    *,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Return a deterministic planning objective; lower is better.

    The objective is intentionally shaped around known anchor neighborhoods and
    is not an exact score, predicted score, or promotion signal.
    """

    normalized_terms: dict[str, float] = {}
    weighted_terms: list[float] = []
    weight_sum = 0.0
    for param in profile.parameters:
        value = params[param.name]
        normalized = param.normalized_distance(value)
        term = normalized * normalized
        normalized_terms[param.name] = term
        weighted_terms.append(term * param.weight)
        weight_sum += param.weight
    anchor_proximity = sum(weighted_terms) / weight_sum if weight_sum else 0.0

    bias_names = [name for name in ("bias_b", "bias_g", "bias_r") if name in params]
    bias_asymmetry = 0.0
    if len(bias_names) == 3:
        bias_mean = statistics.fmean(float(params[name]) for name in bias_names)
        bias_asymmetry = statistics.fmean(
            (float(params[name]) - bias_mean) ** 2 for name in bias_names
        )

    sidecar_names = [
        param.name
        for param in profile.parameters
        if "sidecar" in param.name or (param.runtime_slot and "sidecar" in (param.description or "").lower())
    ]
    sidecar_l1 = 0.0
    if sidecar_names:
        sidecar_l1 = statistics.fmean(
            abs(profile.parameter_map[name].normalized_distance(params[name]))
            for name in sidecar_names
        )

    jitter = profile.jitter_scale * _stable_unit_interval(
        {"seed": seed, "profile": profile.profile_id, "params": params}
    )
    objective = (
        profile.base_proxy_objective
        + profile.objective_scale * anchor_proximity
        + profile.bias_asymmetry_weight * bias_asymmetry
        + profile.sidecar_l1_weight * sidecar_l1
        + jitter
    )
    return objective, {
        "anchor_proximity": anchor_proximity,
        "bias_asymmetry": bias_asymmetry,
        "sidecar_l1": sidecar_l1,
        "deterministic_jitter": jitter,
        **{f"normalized_{name}": value for name, value in normalized_terms.items()},
    }


def _candidate_row(
    *,
    profile: CandidateGenerationProfile,
    draft: CandidateDraft,
    optimizer: str,
    optimizer_status: str,
    seed: int,
) -> dict[str, Any]:
    parameter_group_fingerprint = _parameter_group_fingerprint_ref()
    solver_stack_wire_in = build_optimizer_training_signal_wire_in(
        candidate_id=draft.candidate_id,
        profile_id=profile.profile_id,
        lane_id=profile.lane_id,
        lane_class=profile.lane_class,
        candidate_family=profile.candidate_family,
        representation_family=profile.representation_family,
        substrate_family=profile.substrate_family,
        training_signal_kind=profile.training_signal_kind,
        param_schema=profile.param_schema,
        candidate_params=draft.params,
        source_anchor=profile.source_anchor,
        score_lowering_hypothesis=profile.score_lowering_hypothesis,
        dispatch_blockers=[*BASE_DISPATCH_BLOCKERS, *profile.dispatch_blockers],
        canonical_equation_refs=profile.canonical_equation_refs,
        master_gradient_features=profile.master_gradient_features,
        xray_primitives=profile.xray_primitives,
        deterministic_solution_refs=profile.deterministic_solution_refs,
        variant_axes=profile.variant_axes,
        paired_modes=profile.paired_modes,
    )
    row = {
        "candidate_id": draft.candidate_id,
        "lane_id": profile.lane_id,
        "lane_class": profile.lane_class,
        "candidate_family": profile.candidate_family,
        "representation_family": profile.representation_family,
        "substrate_family": profile.substrate_family,
        "training_signal_kind": profile.training_signal_kind,
        "profile": profile.profile_id,
        "param_schema": profile.param_schema,
        "optimizer": optimizer,
        "optimizer_status": optimizer_status,
        "embedding_lr_scaling_policy": parameter_group_fingerprint[
            "embedding_lr_scaling_policy"
        ],
        "parameter_group_lr_policy_id": parameter_group_fingerprint["policy_id"],
        "parameter_group_lr_policy_sha256": parameter_group_fingerprint["policy_sha256"],
        "parameter_group_fingerprint": parameter_group_fingerprint,
        "parameter_group_fingerprint_sha256": parameter_group_fingerprint[
            "fingerprint_sha256"
        ],
        "seed": seed,
        "trial_index": draft.trial_index,
        "generation": draft.generation,
        "candidate_params": dict(draft.params),
        "op_params": dict(draft.params),
        "runtime_slots": {
            param.name: param.runtime_slot
            for param in profile.parameters
            if param.runtime_slot
        },
        "proxy_objective": draft.proxy_objective,
        "proxy_components": dict(draft.proxy_components),
        "rank_score": draft.proxy_objective,
        "rank_score_field": "proxy_objective_not_score",
        "proxy_only": True,
        "dry_run_only": True,
        "provider_agnostic": True,
        "score_claim_valid": False,
        "exact_auth_eval_performed": False,
        "archive_zip_emitted": False,
        "inflate_runtime_emitted": False,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "evidence_grade": EVIDENCE_GRADE,
        "score_lowering_hypothesis": profile.score_lowering_hypothesis,
        "source_anchor": profile.source_anchor,
        "solver_stack_wire_in": solver_stack_wire_in,
        "master_gradient_features": list(profile.master_gradient_features),
        "xray_primitives": list(profile.xray_primitives),
        "canonical_equation_refs": list(profile.canonical_equation_refs),
        "deterministic_solution_refs": list(profile.deterministic_solution_refs),
        "variant_axes": list(profile.variant_axes),
        "paired_modes": list(profile.paired_modes),
        "promotion_path": (
            "builder consumes candidate_params -> byte-closed archive/runtime -> "
            "runtime-consumption proof -> exact-readiness queue -> claimed exact CUDA eval"
        ),
    }
    return apply_proxy_evidence_boundary(
        row,
        dispatch_blockers=[*BASE_DISPATCH_BLOCKERS, *profile.dispatch_blockers],
    )


def _parameter_group_fingerprint_ref() -> dict[str, Any]:
    fingerprint = build_parameter_group_lr_policy_fingerprint(
        (),
        policy=EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY,
    )
    fingerprint.update(
        {
            "fingerprint_status": "pending_model_parameter_shape_manifest",
            "fingerprint_scope": "policy_only_no_model_parameter_shapes_yet",
            "classification_records": [],
            "record_count": 0,
            "unknown_shape_count": 0,
            "embedding_lr_scaling_policy": EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY[
                "embedding_lr_scaling_policy"
            ],
        }
    )
    fingerprint.pop("fingerprint_sha256", None)
    fingerprint["fingerprint_sha256"] = hashlib.sha256(
        canonical_json(fingerprint).encode("utf-8")
    ).hexdigest()
    return fingerprint


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CandidateGenerationError(f"{name} must be finite float")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise CandidateGenerationError(f"{name} must be finite float") from None
    if not math.isfinite(parsed):
        raise CandidateGenerationError(f"{name} must be finite float")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = _finite_float(value, name)
    if parsed <= 0:
        raise CandidateGenerationError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _finite_float(value, name)
    if parsed < 0:
        raise CandidateGenerationError(f"{name} must be non-negative")
    return parsed


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _str_tuple_or_default(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Iterable):
        items = tuple(str(item) for item in value if str(item))
    else:
        raise CandidateGenerationError("wire-in reference fields must be strings or lists")
    if not items:
        return default
    return tuple(dict.fromkeys(items))


def _drop_none(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value is not None}


def _ordered_unique_values(values: Iterable[float | int]) -> list[float | int]:
    out: list[float | int] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _bounded_product(values_by_param: list[list[float | int]], *, limit: int) -> Iterable[tuple[float | int, ...]]:
    if not values_by_param:
        return
    stack: list[tuple[int, list[float | int]]] = [(0, [])]
    emitted = 0
    while stack and emitted < limit:
        index, prefix = stack.pop()
        if index == len(values_by_param):
            emitted += 1
            yield tuple(prefix)
            continue
        values = list(reversed(values_by_param[index]))
        for value in values:
            stack.append((index + 1, [*prefix, value]))


def _param_key(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(key), _json_safe(value)) for key, value in params.items()))


def _stable_unit_interval(payload: Mapping[str, Any]) -> float:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    return value


_DEFAULT_PROFILE_PAYLOADS: dict[str, dict[str, Any]] = {
    "pr101_bias_refine": {
        "profile_id": "pr101_bias_refine",
        "lane_id": "offline_pr101_bias_refine_candidate_generation",
        "lane_class": "pr101_kaggle_bias_refine",
        "candidate_family": "pr101_runtime_consumed_bias_refinement",
        "param_schema": "pr101_kaggle_proxy_bias_runtime_params_v1",
        "candidate_prefix": "bias_refine",
        "source_anchor": "PR101/A1 verified inflate-time bias anchor",
        "score_lowering_hypothesis": (
            "Small runtime-consumed channel-bias moves around PR101's -1 bias "
            "anchor may improve A1/PR101 scorer geometry once materialized and exact-evaled."
        ),
        "dispatch_blockers": [
            "bias_params_require_runtime_packet_builder",
            "runtime_consumption_proof_required",
        ],
        "parameters": [
            {
                "name": "bias_b",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 0, 2]",
                "description": "PR101 frame-0 blue-channel inflate-time bias.",
            },
            {
                "name": "bias_g",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 1, 1]",
                "description": "PR101 frame-1 green-channel inflate-time bias.",
            },
            {
                "name": "bias_r",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 0, 0]",
                "description": "PR101 frame-0 red-channel inflate-time bias.",
            },
        ],
    },
    "pr101_bias_sidecar": {
        "profile_id": "pr101_bias_sidecar",
        "lane_id": "offline_pr101_bias_sidecar_candidate_generation",
        "lane_class": "a1_pr101_bias_sidecar_prefilter",
        "candidate_family": "a1_pr101_runtime_bias_plus_sidecar_probe",
        "param_schema": "pr101_bias_sidecar_candidate_params_v1",
        "candidate_prefix": "bias_sidecar",
        "source_anchor": "A1 inherited PR101 bias plus valid frame-1 red sidecar coordinate",
        "score_lowering_hypothesis": (
            "A bounded fourth coordinate can test whether PR101's bias trick has "
            "a small sidecar interaction on A1 without broad ad hoc constants."
        ),
        "dispatch_blockers": [
            "sidecar_param_requires_archive_builder_support",
            "sidecar_runtime_slot_not_promoted_by_bias_only_builder",
            "runtime_consumption_proof_required",
        ],
        "parameters": [
            {
                "name": "bias_b",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 0, 2]",
            },
            {
                "name": "bias_g",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 1, 1]",
            },
            {
                "name": "bias_r",
                "low": -1.08,
                "high": -0.92,
                "anchor": -1.0,
                "runtime_slot": "up[:, 0, 0]",
            },
            {
                "name": "sidecar_f1_r",
                "low": -0.25,
                "high": 0.25,
                "anchor": 0.0,
                "weight": 1.5,
                "runtime_slot": "up[:, 1, 0]",
                "description": "Sidecar coordinate on valid frame-1 red slot.",
            },
        ],
    },
    "pr95_hnerv_muon_training_smoke": {
        "profile_id": "pr95_hnerv_muon_training_smoke",
        "lane_id": "offline_pr95_hnerv_muon_training_smoke_candidate_generation",
        "lane_class": "pr95_hnerv_muon_local_training_proxy",
        "candidate_family": "pr95_hnerv_muon_optimizer_recipe_smoke",
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_signal_kind": "local_representation_training_optimizer_schedule_smoke",
        "param_schema": "pr95_hnerv_muon_optimizer_recipe_params_v1",
        "candidate_prefix": "pr95_muon_hnerv",
        "base_proxy_objective": 0.19285,
        "objective_scale": 0.00018,
        "bias_asymmetry_weight": 0.0,
        "sidecar_l1_weight": 0.00006,
        "source_anchor": "PR95 HNeRV Muon stage-8 recipe; local training smoke only",
        "score_lowering_hypothesis": (
            "PR95's HNeRV Muon recipe should be reproduced faithfully, then varied "
            "through bounded optimizer/scheduler/normalization deltas under common "
            "seed and budget before any archive/runtime exact-eval gate."
        ),
        "dispatch_blockers": [
            "pr95_hnerv_training_smoke_not_archive",
            "requires_pr95_faithful_control_pair",
            "requires_common_seed_and_budget",
            "requires_optimizer_config_sha",
            "requires_trainer_runtime_sha",
            "requires_archive_export_before_exact_eval",
            "requires_master_gradient_component_marginal_anchor",
        ],
        "canonical_equation_refs": [
            "per_pair_master_gradient_score_impact_taylor_v1",
            "master_gradient_locality_violation_by_codec_v1",
            "canonical_frontier_pointer_v1",
            "pairset_component_marginal_score_decomposition_v1",
            "scorer_input_cache_hash_identity_v1",
        ],
        "master_gradient_features": [
            "pairset_component_marginal",
            "hard_pair_indices",
            "segnet_posenet_axis_dominance",
            "per_pair_training_weight_schedule",
            "optimizer_recipe_component_marginal",
        ],
        "xray_primitives": [
            "pairset_component_marginal",
            "per_pair_score_decomposition",
            "unified_action_principle",
            "score_lipschitz",
            "segnet_margin_polytope",
            "posenet_se3_lie_algebra",
            "wavelet_hf_energy",
        ],
        "deterministic_solution_refs": [
            "tac.packet_compiler.deterministic_compiler",
            "tac.hnerv_training_parity_guard",
            "tac.canonical_equations.scorer_input_cache_hash_identity",
            "tools/recover_pr95_training_curriculum.py",
        ],
        "variant_axes": [
            "pr95_source_faithful_control",
            "optimizer_recipe",
            "embedding_lr_scaling_policy",
            "parameter_group_fingerprint",
            "scheduler_recipe",
            "normalization_or_weight_decay",
            "training_curriculum",
            "archive_export",
        ],
        "paired_modes": [
            "pr95_faithful_control",
            "optimizer_variant",
            "embedding_lr_scaling_policy_variant",
            "parameter_group_fingerprint_variant",
            "scheduler_variant",
            "normalization_or_weight_decay_variant",
            "archive_export_variant",
        ],
        "parameters": [
            {
                "name": "muon_ns_steps",
                "kind": "int",
                "low": 3,
                "high": 7,
                "anchor": 5,
                "step": 1,
                "runtime_slot": "optimizer.muon.ns_steps",
                "description": "Newton-Schulz/polar iteration count for Muon-eligible HNeRV hidden weights.",
            },
            {
                "name": "muon_momentum",
                "low": 0.90,
                "high": 0.98,
                "anchor": 0.95,
                "runtime_slot": "optimizer.muon.momentum",
            },
            {
                "name": "hidden_weight_decay",
                "low": 0.0001,
                "high": 0.0010,
                "anchor": 0.0005,
                "runtime_slot": "optimizer.muon.hidden_weight_decay",
            },
            {
                "name": "adamw_lr_ratio",
                "low": 0.25,
                "high": 2.0,
                "anchor": 1.0,
                "runtime_slot": "optimizer.adamw.lr_ratio",
                "description": "LR multiplier for non-Muon params: 1D, bias, stem, RGB/head params.",
            },
            {
                "name": "warmup_fraction",
                "low": 0.0,
                "high": 0.20,
                "anchor": 0.05,
                "runtime_slot": "scheduler.warmup_fraction",
            },
            {
                "name": "polyak_swa_fraction",
                "low": 0.0,
                "high": 0.40,
                "anchor": 0.15,
                "runtime_slot": "averaging.polyak_swa_fraction",
                "description": "Late-training averaging budget; smoke-only until paired replay proves benefit.",
            },
        ],
    },
}


__all__ = [
    "BASE_DISPATCH_BLOCKERS",
    "DEFAULT_GENERATED_AT_UTC",
    "MAX_CANDIDATES",
    "OPTIMIZER_STATUSES",
    "QUEUE_SCHEMA",
    "CandidateDraft",
    "CandidateGenerationError",
    "CandidateGenerationProfile",
    "ParameterSpec",
    "default_profiles",
    "generate_candidate_queue",
    "load_profile",
    "profile_from_json",
    "proxy_objective",
]
