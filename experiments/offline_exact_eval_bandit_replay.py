#!/usr/bin/env python3
"""Deterministic offline replay controller for exact-eval lane cards.

This tool ranks unevaluated candidate configs for the next CUDA exact-eval
queue using historical lane cards. It is intentionally conservative:

* reward is accepted only from CUDA ``contest_auth_eval`` or adjudication JSON;
* CPU/MPS/proxy/local records are advisory features only;
* every emitted recommendation is labelled no-claim / proxy-only;
* candidate identity is the SHA-256 of canonical candidate config JSON.

The controller is offline: it never launches training, inflate, or auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT = "offline_exact_eval_bandit_replay_v1"
SCHEMA_VERSION = 1
PROXY_ONLY_TAG = "[proxy-only advisory]"
EXACT_CUDA_TAG = "[contest-CUDA historical reward]"
CONFIG_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

CONTEST_JSON_KEYS = (
    "contest_auth_eval_json",
    "contest_auth_eval",
    "contest_json",
)
CONTEST_PATH_KEYS = (
    "contest_auth_eval_path",
    "contest_json_path",
)
ADJUDICATION_JSON_KEYS = (
    "adjudication_json",
    "adjudication",
)
ADJUDICATION_PATH_KEYS = (
    "adjudication_path",
    "adjudication_json_path",
)
MAYBE_RESULT_PATH_KEYS = (
    "result_json",
    "result_json_path",
)
CONFIG_KEYS = (
    "candidate_config",
    "config",
    "hyperparameters",
    "params",
)
CONFIG_SHA_KEYS = (
    "candidate_config_sha256",
    "config_sha256",
)
CARD_ID_KEYS = (
    "card_id",
    "lane_id",
    "run_id",
    "job_name",
    "id",
    "name",
)
FAMILY_KEYS = (
    "lane_family",
    "family",
    "method",
    "codec",
    "lane",
    "script",
)
FEATURE_CONTAINER_KEYS = (
    "features",
    "advisory_features",
    "proxy_features",
    "diagnostics",
)
TOP_LEVEL_ADVISORY_KEYS = (
    "proxy_score",
    "score_proxy",
    "local_proxy_score",
    "mps_score",
    "cpu_score",
    "advisory_score",
    "estimated_score",
    "predicted_score",
    "forecast_score",
    "archive_bytes_estimate",
    "estimated_archive_bytes",
    "proxy_archive_bytes",
    "rate_bytes_estimate",
)
RAW_SCORE_FEATURE_PRIORITY = (
    "proxy_score",
    "local_proxy_score",
    "score_proxy",
    "advisory_score",
    "estimated_score",
    "predicted_score",
    "forecast_score",
    "mps_score",
    "cpu_score",
    "contest_auth_eval.score_recomputed",
    "adjudication.score_recomputed",
)


class ReplayBuildError(RuntimeError):
    """Raised for fail-closed card or custody errors."""


@dataclass(frozen=True)
class LoadedCard:
    raw: dict[str, Any]
    source_path: Path
    source_label: str
    source_index: int


@dataclass(frozen=True)
class RewardEvidence:
    eligible: bool
    score: float | None
    kind: str
    source: str
    reason: str
    metadata: dict[str, Any]
    advisory_features: dict[str, float]


@dataclass(frozen=True)
class LaneRecord:
    card_id: str
    source_label: str
    source_index: int
    family: str
    candidate_config_sha256: str
    config_sha_source: str
    features: dict[str, float]
    reward: RewardEvidence


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any, *, label: str) -> float:
    if not _is_finite_number(value):
        raise ReplayBuildError(f"{label} must be a finite number")
    return float(value)


def _optional_number(value: Any) -> float | None:
    return float(value) if _is_finite_number(value) else None


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReplayBuildError(f"candidate config is not canonical-JSON serializable: {exc}") from exc


def candidate_config_sha256(config: Any) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _validate_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReplayBuildError(f"{label} must be a 64-char lowercase hex SHA-256")
    lowered = value.lower()
    if not CONFIG_SHA_RE.match(lowered):
        raise ReplayBuildError(f"{label} must be a 64-char lowercase hex SHA-256")
    return lowered


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _resolve_json_path(value: str, *, source_path: Path) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.insert(0, source_path.parent / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ReplayBuildError(f"referenced JSON path does not exist: {value!r} from {source_path}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReplayBuildError(f"invalid JSON in {path}: {exc}") from exc


def _iter_json_payloads(
    card: dict[str, Any],
    *,
    source_path: Path,
    object_keys: tuple[str, ...],
    path_keys: tuple[str, ...],
) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for key in object_keys:
        value = card.get(key)
        if isinstance(value, dict):
            payloads.append((key, value))
        elif isinstance(value, str):
            payload_path = _resolve_json_path(value, source_path=source_path)
            payload = _read_json(payload_path)
            if not isinstance(payload, dict):
                raise ReplayBuildError(f"{key} JSON payload must be an object: {payload_path}")
            payloads.append((str(payload_path), payload))
    for key in path_keys:
        value = card.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ReplayBuildError(f"{key} must be a JSON path string")
        payload_path = _resolve_json_path(value, source_path=source_path)
        payload = _read_json(payload_path)
        if not isinstance(payload, dict):
            raise ReplayBuildError(f"{key} JSON payload must be an object: {payload_path}")
        payloads.append((str(payload_path), payload))
    return payloads


def _iter_result_payloads(card: dict[str, Any], *, source_path: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for key in MAYBE_RESULT_PATH_KEYS:
        value = card.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            payloads.append((key, value))
            continue
        if not isinstance(value, str):
            raise ReplayBuildError(f"{key} must be a JSON path string or object")
        payload_path = _resolve_json_path(value, source_path=source_path)
        payload = _read_json(payload_path)
        if not isinstance(payload, dict):
            raise ReplayBuildError(f"{key} JSON payload must be an object: {payload_path}")
        payloads.append((str(payload_path), payload))
    return payloads


def _contest_reward(payload: dict[str, Any], *, source: str, required_samples: int) -> RewardEvidence:
    score = _optional_number(payload.get("score_recomputed_from_components"))
    advisory: dict[str, float] = {}
    if score is not None:
        advisory["contest_auth_eval.score_recomputed"] = score
    for key in ("final_score", "avg_posenet_dist", "avg_segnet_dist", "archive_size_bytes", "rate_unscaled"):
        value = _optional_number(payload.get(key))
        if value is not None:
            advisory[f"contest_auth_eval.{key}"] = value

    if score is None:
        return RewardEvidence(
            eligible=False,
            score=None,
            kind="contest_auth_eval_json",
            source=source,
            reason="missing_score_recomputed_from_components",
            metadata={},
            advisory_features=advisory,
        )

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return RewardEvidence(
            eligible=False,
            score=score,
            kind="contest_auth_eval_json",
            source=source,
            reason="missing_provenance",
            metadata={"n_samples": payload.get("n_samples")},
            advisory_features=advisory,
        )

    device = provenance.get("device") or payload.get("device")
    n_samples = payload.get("n_samples")
    if device != "cuda":
        return RewardEvidence(
            eligible=False,
            score=score,
            kind="contest_auth_eval_json",
            source=source,
            reason=f"non_cuda_device:{device!r}",
            metadata={"device": device, "n_samples": n_samples},
            advisory_features=advisory,
        )
    if n_samples != required_samples:
        return RewardEvidence(
            eligible=False,
            score=score,
            kind="contest_auth_eval_json",
            source=source,
            reason=f"sample_count:{n_samples!r}",
            metadata={"device": device, "n_samples": n_samples},
            advisory_features=advisory,
        )

    archive_sha = provenance.get("archive_sha256") or payload.get("archive_sha256")
    archive_bytes = payload.get("archive_size_bytes")
    return RewardEvidence(
        eligible=True,
        score=score,
        kind="contest_auth_eval_json",
        source=source,
        reason="exact_cuda_contest_auth_eval",
        metadata={
            "device": device,
            "n_samples": n_samples,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "gpu_model": provenance.get("gpu_model"),
            "gpu_t4_match": provenance.get("gpu_t4_match"),
            "evidence_grade": "A++ contest T4" if provenance.get("gpu_t4_match") is True else "A score-grade",
            "score_source": "contest_auth_eval.json:score_recomputed_from_components",
        },
        advisory_features=advisory,
    )


def _adjudication_reward(payload: dict[str, Any], *, source: str, required_samples: int) -> RewardEvidence:
    advisory: dict[str, float] = {}
    for key in (
        "score_recomputed",
        "score_reported_rounded",
        "contest_cuda_score_recomputed",
        "contest_cuda_score",
        "contest_cuda_archive_bytes",
        "archive_bytes",
        "avg_posenet_dist",
        "avg_segnet_dist",
    ):
        value = _optional_number(payload.get(key))
        if value is not None:
            advisory[f"adjudication.{key}"] = value

    score = _optional_number(payload.get("contest_cuda_score_recomputed"))
    score_key = "contest_cuda_score_recomputed"
    if score is None:
        score = _optional_number(payload.get("score_recomputed"))
        score_key = "score_recomputed"
    if score is None:
        return RewardEvidence(
            eligible=False,
            score=None,
            kind="adjudication_json",
            source=source,
            reason="missing_adjudicated_score",
            metadata={},
            advisory_features=advisory,
        )

    device = payload.get("contest_cuda_device")
    n_samples = payload.get("contest_cuda_n_samples")
    if device is not None or n_samples is not None:
        if device != "cuda":
            return RewardEvidence(
                eligible=False,
                score=score,
                kind="adjudication_json",
                source=source,
                reason=f"non_cuda_device:{device!r}",
                metadata={"device": device, "n_samples": n_samples},
                advisory_features=advisory,
            )
        if n_samples != required_samples:
            return RewardEvidence(
                eligible=False,
                score=score,
                kind="adjudication_json",
                source=source,
                reason=f"sample_count:{n_samples!r}",
                metadata={"device": device, "n_samples": n_samples},
                advisory_features=advisory,
            )
        exact_reason = "exact_cuda_adjudication_provenance"
    else:
        evidence_grade = payload.get("evidence_grade")
        if not (isinstance(evidence_grade, str) and evidence_grade.startswith("A")):
            return RewardEvidence(
                eligible=False,
                score=score,
                kind="adjudication_json",
                source=source,
                reason=f"non_exact_evidence_grade:{evidence_grade!r}",
                metadata={"evidence_grade": evidence_grade},
                advisory_features=advisory,
            )
        exact_reason = "exact_cuda_adjudication_json"

    archive_sha = payload.get("contest_cuda_archive_sha256") or payload.get("archive_sha256")
    archive_bytes = payload.get("contest_cuda_archive_bytes") or payload.get("archive_bytes")
    return RewardEvidence(
        eligible=True,
        score=score,
        kind="adjudication_json",
        source=source,
        reason=exact_reason,
        metadata={
            "device": device or "cuda_via_adjudication_gate",
            "n_samples": n_samples or required_samples,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "gpu_model": payload.get("gpu_model"),
            "gpu_t4_match": payload.get("gpu_t4_match") or payload.get("contest_cuda_gpu_t4_match"),
            "evidence_grade": payload.get("evidence_grade"),
            "promotion_eligible": payload.get("promotion_eligible"),
            "allowed_use": payload.get("allowed_use"),
            "lane_status": payload.get("lane_status"),
            "score_source": f"adjudication.json:{score_key}",
        },
        advisory_features=advisory,
    )


def _best_reward(evidences: list[RewardEvidence]) -> RewardEvidence:
    eligible = [evidence for evidence in evidences if evidence.eligible]
    if eligible:
        scores = {round(float(evidence.score), 12) for evidence in eligible if evidence.score is not None}
        if len(scores) > 1:
            details = ", ".join(f"{e.source}={e.score}" for e in eligible)
            raise ReplayBuildError(f"conflicting exact CUDA rewards on one card: {details}")
        # Prefer direct contest_auth_eval custody when both surfaces are present.
        eligible.sort(key=lambda e: (0 if e.kind == "contest_auth_eval_json" else 1, e.source))
        return eligible[0]

    advisory_features: dict[str, float] = {}
    reasons: list[str] = []
    for evidence in evidences:
        advisory_features.update(evidence.advisory_features)
        if evidence.reason:
            reasons.append(f"{evidence.kind}:{evidence.reason}")
    return RewardEvidence(
        eligible=False,
        score=None,
        kind="none",
        source="none",
        reason=";".join(sorted(reasons)) or "no_exact_cuda_evidence",
        metadata={},
        advisory_features=advisory_features,
    )


def extract_reward_evidence(card: dict[str, Any], *, source_path: Path, required_samples: int = 600) -> RewardEvidence:
    evidences: list[RewardEvidence] = []

    if "score_recomputed_from_components" in card and isinstance(card.get("provenance"), dict):
        evidences.append(_contest_reward(card, source="<card>", required_samples=required_samples))
    if "score_recomputed" in card or "contest_cuda_score_recomputed" in card:
        evidences.append(_adjudication_reward(card, source="<card>", required_samples=required_samples))

    for source, payload in _iter_json_payloads(
        card,
        source_path=source_path,
        object_keys=CONTEST_JSON_KEYS,
        path_keys=CONTEST_PATH_KEYS,
    ):
        evidences.append(_contest_reward(payload, source=source, required_samples=required_samples))

    for source, payload in _iter_json_payloads(
        card,
        source_path=source_path,
        object_keys=ADJUDICATION_JSON_KEYS,
        path_keys=ADJUDICATION_PATH_KEYS,
    ):
        evidences.append(_adjudication_reward(payload, source=source, required_samples=required_samples))

    for source, payload in _iter_result_payloads(card, source_path=source_path):
        if "score_recomputed_from_components" in payload:
            evidences.append(_contest_reward(payload, source=source, required_samples=required_samples))
        elif "score_recomputed" in payload or "contest_cuda_score_recomputed" in payload:
            evidences.append(_adjudication_reward(payload, source=source, required_samples=required_samples))

    return _best_reward(evidences)


def _flatten_numeric_features(value: Any, *, prefix: str, depth: int = 3) -> dict[str, float]:
    out: dict[str, float] = {}
    if depth < 0:
        return out
    if isinstance(value, dict):
        for key in sorted(value):
            child = value[key]
            if key in CONTEST_JSON_KEYS or key in ADJUDICATION_JSON_KEYS:
                continue
            label = f"{prefix}.{key}" if prefix else str(key)
            if _is_finite_number(child):
                out[label] = float(child)
            elif isinstance(child, dict):
                out.update(_flatten_numeric_features(child, prefix=label, depth=depth - 1))
    return out


def _advisory_features(card: dict[str, Any], config: Any, reward: RewardEvidence) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in FEATURE_CONTAINER_KEYS:
        value = card.get(key)
        if isinstance(value, dict):
            features.update(_flatten_numeric_features(value, prefix="", depth=3))
    for key in TOP_LEVEL_ADVISORY_KEYS:
        value = _optional_number(card.get(key))
        if value is not None:
            features[key] = value
    if isinstance(config, dict):
        features.update(_flatten_numeric_features(config, prefix="config", depth=3))
    if reward.eligible:
        # Exact CUDA metrics are the supervised reward, not advisory inputs.
        # Feeding them back into calibration would let the replay model learn
        # an identity map from post-eval score components.
        pass
    else:
        features.update(reward.advisory_features)
    return dict(sorted(features.items()))


def _card_id(card: dict[str, Any], *, fallback: str) -> str:
    for key in CARD_ID_KEYS:
        value = card.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return fallback


def _family(card: dict[str, Any], config: Any, *, card_id: str) -> str:
    for key in FAMILY_KEYS:
        value = card.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    if isinstance(config, dict):
        for key in FAMILY_KEYS:
            value = config.get(key)
            if isinstance(value, (str, int)):
                return str(value)
    if ":" in card_id:
        return card_id.split(":", 1)[0]
    if "_" in card_id:
        return card_id.split("_", 1)[0]
    return "unknown"


def _config_and_sha(card: dict[str, Any]) -> tuple[Any | None, str, str]:
    config_key, config = _first_present(card, CONFIG_KEYS)
    sha_key, supplied_sha = _first_present(card, CONFIG_SHA_KEYS)
    computed_sha: str | None = None
    if config_key is not None:
        computed_sha = candidate_config_sha256(config)
    if sha_key is not None:
        sha = _validate_sha(supplied_sha, label=str(sha_key))
        if computed_sha is not None and sha != computed_sha:
            raise ReplayBuildError(
                "candidate config SHA mismatch: "
                f"supplied {sha_key}={sha}, computed={computed_sha}"
            )
        return config, sha, str(sha_key)
    if computed_sha is None:
        raise ReplayBuildError("card must include candidate_config/config or candidate_config_sha256")
    return config, computed_sha, str(config_key)


def _load_jsonl(path: Path) -> list[LoadedCard]:
    cards: list[LoadedCard] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayBuildError(f"invalid JSONL line {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReplayBuildError(f"JSONL line must be an object: {path}:{line_number}")
        cards.append(
            LoadedCard(
                raw=payload,
                source_path=path,
                source_label=f"{path}:{line_number}",
                source_index=len(cards),
            )
        )
    return cards


def _load_json_cards(path: Path) -> list[LoadedCard]:
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        raw_cards = payload["cards"]
    elif isinstance(payload, list):
        raw_cards = payload
    elif isinstance(payload, dict):
        raw_cards = [payload]
    else:
        raise ReplayBuildError(f"JSON input must be an object, list, or object with cards: {path}")

    cards: list[LoadedCard] = []
    for index, raw_card in enumerate(raw_cards):
        if not isinstance(raw_card, dict):
            raise ReplayBuildError(f"card must be an object: {path}[{index}]")
        cards.append(
            LoadedCard(
                raw=raw_card,
                source_path=path,
                source_label=f"{path}[{index}]",
                source_index=index,
            )
        )
    return cards


def load_cards(paths: list[Path]) -> list[LoadedCard]:
    cards: list[LoadedCard] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ReplayBuildError(f"input path does not exist: {path}")
        loaded = _load_jsonl(resolved) if resolved.suffix == ".jsonl" else _load_json_cards(resolved)
        cards.extend(loaded)
    return cards


def _records(cards: list[LoadedCard], *, required_samples: int) -> list[LaneRecord]:
    records: list[LaneRecord] = []
    for ordinal, loaded in enumerate(cards):
        config, config_sha, sha_source = _config_and_sha(loaded.raw)
        reward = extract_reward_evidence(loaded.raw, source_path=loaded.source_path, required_samples=required_samples)
        fallback = f"{loaded.source_label}"
        card_id = _card_id(loaded.raw, fallback=fallback)
        family = _family(loaded.raw, config, card_id=card_id)
        features = _advisory_features(loaded.raw, config, reward)
        records.append(
            LaneRecord(
                card_id=card_id,
                source_label=loaded.source_label,
                source_index=ordinal,
                family=family,
                candidate_config_sha256=config_sha,
                config_sha_source=sha_source,
                features=features,
                reward=reward,
            )
        )
    return records


def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-18:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = y_mean - slope * x_mean
    mse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True)) / len(xs)
    return intercept, slope, mse


def _fit_advisory_model(records: list[LaneRecord]) -> dict[str, Any]:
    exact = [record for record in records if record.reward.eligible and record.reward.score is not None]
    candidates: list[tuple[float, str, float, float, int]] = []
    feature_names = sorted({name for record in exact for name in record.features})
    for feature_name in feature_names:
        xs: list[float] = []
        ys: list[float] = []
        for record in exact:
            if feature_name in record.features:
                xs.append(record.features[feature_name])
                ys.append(float(record.reward.score))
        fit = _fit_linear(xs, ys)
        if fit is None:
            continue
        intercept, slope, mse = fit
        candidates.append((mse, feature_name, intercept, slope, len(xs)))

    if not candidates:
        return {
            "kind": "uncalibrated_proxy_only",
            "feature": None,
            "calibration_points": 0,
            "note": "No finite advisory feature had at least two exact CUDA rewards with nonzero variance.",
        }
    mse, feature_name, intercept, slope, count = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "kind": "single_feature_linear_replay",
        "feature": feature_name,
        "intercept": intercept,
        "slope": slope,
        "mse": mse,
        "calibration_points": count,
        "reward_target": "exact_cuda_score_lower_is_better",
        "claim_status": "proxy_only_advisory_model",
    }


def _aggregate_features(records: list[LaneRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.features.items():
            values.setdefault(key, []).append(value)
    return {key: sum(items) / len(items) for key, items in sorted(values.items())}


def _seeded_tie(seed: int, candidate_sha: str) -> str:
    digest = hashlib.sha256(f"{seed}:{candidate_sha}".encode("utf-8")).hexdigest()
    return digest[:16]


def _raw_proxy_prediction(features: dict[str, float]) -> tuple[float | None, str | None]:
    for key in RAW_SCORE_FEATURE_PRIORITY:
        if key in features:
            return features[key], key
    return None, None


def _family_stats(records: list[LaneRecord]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if record.reward.eligible and record.reward.score is not None:
            grouped.setdefault(record.family, []).append(float(record.reward.score))
    return {
        family: {
            "exact_cuda_count": len(scores),
            "mean_score": sum(scores) / len(scores),
            "best_score": min(scores),
        }
        for family, scores in sorted(grouped.items())
    }


def _prediction_for_group(
    records: list[LaneRecord],
    *,
    model: dict[str, Any],
    family_stats: dict[str, dict[str, Any]],
    global_mean_score: float | None,
) -> tuple[float | None, list[str], str]:
    features = _aggregate_features(records)
    used: list[str] = []

    feature_name = model.get("feature")
    if isinstance(feature_name, str) and feature_name in features:
        intercept = _number(model.get("intercept"), label="model.intercept")
        slope = _number(model.get("slope"), label="model.slope")
        used.append(feature_name)
        return intercept + slope * features[feature_name], used, "calibrated_advisory_feature"

    raw_prediction, raw_feature = _raw_proxy_prediction(features)
    if raw_prediction is not None and raw_feature is not None:
        used.append(raw_feature)
        return raw_prediction, used, "raw_advisory_feature"

    family = sorted({record.family for record in records})[0]
    stats = family_stats.get(family)
    if stats is not None:
        return float(stats["mean_score"]), used, "family_exact_cuda_mean"

    if global_mean_score is not None:
        return global_mean_score, used, "global_exact_cuda_mean"

    return None, used, "no_advisory_signal"


def build_report(
    input_paths: list[Path],
    *,
    seed: int,
    top_k: int | None = None,
    required_samples: int = 600,
    exploration_weight: float = 0.05,
) -> dict[str, Any]:
    if top_k is not None and top_k <= 0:
        raise ReplayBuildError("--top-k must be positive when provided")
    if required_samples <= 0:
        raise ReplayBuildError("--required-samples must be positive")
    if not math.isfinite(exploration_weight) or exploration_weight < 0.0:
        raise ReplayBuildError("--exploration-weight must be finite and non-negative")

    cards = load_cards(input_paths)
    records = _records(cards, required_samples=required_samples)
    exact_records = [record for record in records if record.reward.eligible and record.reward.score is not None]
    exact_scores = [float(record.reward.score) for record in exact_records]
    global_mean_score = sum(exact_scores) / len(exact_scores) if exact_scores else None
    model = _fit_advisory_model(records)
    families = _family_stats(records)

    groups: dict[str, list[LaneRecord]] = {}
    for record in records:
        groups.setdefault(record.candidate_config_sha256, []).append(record)

    historical: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    advisory_only_count = 0
    non_reward_reasons: dict[str, int] = {}
    total_exact_by_family = {family: int(stats["exact_cuda_count"]) for family, stats in families.items()}

    for config_sha in sorted(groups):
        group_records = sorted(groups[config_sha], key=lambda record: (record.card_id, record.source_label))
        group_exact = [record for record in group_records if record.reward.eligible and record.reward.score is not None]
        card_ids = sorted({record.card_id for record in group_records})
        families_for_group = sorted({record.family for record in group_records})
        if group_exact:
            best = min(group_exact, key=lambda record: (float(record.reward.score), record.card_id, record.source_label))
            historical.append(
                {
                    "candidate_config_sha256": config_sha,
                    "card_ids": card_ids,
                    "family": families_for_group[0],
                    "exact_cuda_evaluations": len(group_exact),
                    "best_observed_exact_cuda_score": float(best.reward.score),
                    "reward": -float(best.reward.score),
                    "reward_source": best.reward.kind,
                    "reward_source_path": best.reward.source,
                    "evidence_grade": best.reward.metadata.get("evidence_grade"),
                    "archive_sha256": best.reward.metadata.get("archive_sha256"),
                    "archive_bytes": best.reward.metadata.get("archive_bytes"),
                    "score_claim": False,
                    "result_tag": EXACT_CUDA_TAG,
                    "allowed_use": ["offline_bandit_reward", "exact_eval_budget_triage"],
                }
            )
            continue

        advisory_only_count += len(group_records)
        for record in group_records:
            non_reward_reasons[record.reward.reason] = non_reward_reasons.get(record.reward.reason, 0) + 1

        predicted_score, used_features, prediction_source = _prediction_for_group(
            group_records,
            model=model,
            family_stats=families,
            global_mean_score=global_mean_score,
        )
        family = families_for_group[0]
        family_exact_n = total_exact_by_family.get(family, 0)
        if exact_records:
            exploration_bonus = math.sqrt(math.log(len(exact_records) + 1.0) / (family_exact_n + 1.0))
        else:
            exploration_bonus = 1.0
        acquisition_score = (
            predicted_score - exploration_weight * exploration_bonus
            if predicted_score is not None
            else None
        )
        ranking_rows.append(
            {
                "candidate_config_sha256": config_sha,
                "card_ids": card_ids,
                "family": family,
                "already_exact_cuda_evaluated": False,
                "recommended_for_exact_eval": True,
                "predicted_score_proxy_only": predicted_score,
                "prediction_source": prediction_source,
                "acquisition_score_proxy_only": acquisition_score,
                "exploration_bonus": exploration_bonus,
                "family_exact_cuda_count": family_exact_n,
                "advisory_features_used": used_features,
                "advisory_feature_names": sorted(_aggregate_features(group_records)),
                "seed_tie_break": _seeded_tie(seed, config_sha),
                "score_claim": False,
                "promotion_eligible": False,
                "result_tag": PROXY_ONLY_TAG,
                "reward_source": "none",
                "exact_cuda_required_before_score_claim": True,
                "allowed_use": ["offline_budget_triage", "queue_prioritization"],
            }
        )

    def _ranking_key(row: dict[str, Any]) -> tuple[int, float, str, str]:
        acquisition = row["acquisition_score_proxy_only"]
        missing = 1 if acquisition is None else 0
        sortable_acquisition = float(acquisition) if acquisition is not None else 0.0
        return (
            missing,
            sortable_acquisition,
            str(row["seed_tie_break"]),
            str(row["candidate_config_sha256"]),
        )

    ranking_rows.sort(key=_ranking_key)
    if top_k is not None:
        ranking_rows = ranking_rows[:top_k]
    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index

    historical.sort(key=lambda row: (float(row["best_observed_exact_cuda_score"]), row["candidate_config_sha256"]))

    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "score_claim": False,
        "promotion_eligible": False,
        "result_tag": PROXY_ONLY_TAG,
        "reward_policy": {
            "lower_score_is_better": True,
            "reward": "negative exact CUDA contest score",
            "accepted_score_sources": [
                "contest_auth_eval.json:score_recomputed_from_components with provenance.device=cuda",
                "adjudication JSON/provenance emitted after CUDA contest_auth_eval validation",
            ],
            "required_samples": required_samples,
            "non_cuda_or_proxy_records": "advisory_features_only",
            "recommendations_are_claims": False,
        },
        "input": {
            "paths": [str(path) for path in input_paths],
            "card_count": len(records),
            "unique_candidate_config_count": len(groups),
        },
        "exact_reward_count": len(exact_records),
        "advisory_only_card_count": advisory_only_count,
        "non_reward_reasons": dict(sorted(non_reward_reasons.items())),
        "advisory_model": model,
        "family_stats": families,
        "historical_rewards": historical,
        "ranking": ranking_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Input JSONL or JSON lane-card file. May be supplied multiple times.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Fixed deterministic tie-break seed.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON replay report.")
    parser.add_argument("--top-k", type=int, help="Limit emitted recommendation ranking to K candidates.")
    parser.add_argument(
        "--required-samples",
        type=int,
        default=600,
        help="Required contest_auth_eval sample count for reward acceptance.",
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=0.05,
        help="UCB-style advisory exploration weight. Lower score remains better.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.input,
            seed=args.seed,
            top_k=args.top_k,
            required_samples=args.required_samples,
            exploration_weight=args.exploration_weight,
        )
    except ReplayBuildError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
