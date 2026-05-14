# SPDX-License-Identifier: MIT
"""Per-tensor score-marginals harness for γ-JCSP dispatch.

The ADMM coordinator and sequential codec stack need a per-tensor
``score_per_byte_marginal`` (cached dScore/dByte estimate) to allocate the
byte budget across streams. This module produces and persists those
marginals as a JSON artifact that ``experiments/pipeline.py`` consumes when
``cfg.use_joint_codec_stack=True``.

Three production paths:

* ``derive_uniform_planning_marginals(state_dict, value=1e-6)`` — placeholder
  uniform marginals. Tagged ``placeholder_uniform``. Use only for plumbing
  smoke tests; the orchestrator will allocate proportionally to byte
  estimates with no signal differentiation across streams.
* ``derive_marginals_from_sensitivity_map(state_dict, sensitivities)`` —
  derives per-tensor marginals from an existing per-channel SensitivityMap
  artifact (mean channel sensitivity / byte estimate). Tagged
  ``sensitivity_derived``.
* (Future) ``derive_marginals_from_finite_difference_probe(...)`` — runs a
  contest-CUDA finite-difference sweep against the scorer. Tagged
  ``contest_cuda_calibrated`` with a required ``model_sha256`` field.

Save/load uses a schema-versioned JSON envelope so future agents can detect
incompatible artifacts.

NO scorer is loaded at module import time. NO contest-CUDA score is claimed
without an explicit ``contest_cuda_calibrated`` source tag.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

JCSP_SCORE_MARGINALS_SCHEMA: str = "jcsp_score_marginals_v1"

ALLOWED_SOURCES: frozenset[str] = frozenset(
    {
        "placeholder_uniform",
        "sensitivity_derived",
        "contest_cuda_calibrated",
        "fixture",
    }
)


class JCSPScoreMarginalsError(ValueError):
    """Raised when score-marginals are malformed or non-authoritative."""


def _coerce_to_numpy_float32(tensor: Any) -> np.ndarray:
    if (
        hasattr(tensor, "detach")
        and hasattr(tensor, "cpu")
        and hasattr(tensor, "numpy")
    ):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


def _state_dict_items(model: Any) -> list[tuple[str, Any]]:
    if hasattr(model, "state_dict"):
        items = list(model.state_dict().items())
    elif isinstance(model, Mapping):
        items = list(model.items())
    else:
        raise JCSPScoreMarginalsError(
            "model must be torch.nn.Module or a state_dict-like Mapping"
        )
    if not items:
        raise JCSPScoreMarginalsError("state_dict is empty — cannot derive marginals")
    return [(str(k), v) for k, v in items]


def derive_uniform_planning_marginals(
    model: Any,
    *,
    value: float = 1e-6,
) -> dict[str, float]:
    """Return ``{tensor_name: value}`` for every tensor in ``model``.

    Use only for plumbing smoke tests — the orchestrator falls back to
    proportional byte-budget allocation when all marginals are equal.
    Tagged ``placeholder_uniform`` when persisted.
    """
    if not np.isfinite(value):
        raise JCSPScoreMarginalsError(
            f"derive_uniform_planning_marginals: value must be finite (got {value})"
        )
    if value <= 0.0:
        raise JCSPScoreMarginalsError(
            f"derive_uniform_planning_marginals: value must be > 0 (got {value})"
        )
    items = _state_dict_items(model)
    return {name: float(value) for name, _tensor in items}


def derive_marginals_from_sensitivity_map(
    model: Any,
    sensitivities: Mapping[str, Any],
    *,
    bytes_per_element_estimate: float = 0.5,
    fallback_marginal: float = 1e-9,
) -> dict[str, float]:
    """Derive per-tensor marginals from a per-channel SensitivityMap artifact.

    For each tensor in the model:

    * If ``sensitivities[name]`` exists, marginal = mean(sensitivity_vector)
      divided by the tensor's estimated byte cost.
    * Otherwise, marginal = ``fallback_marginal`` (small positive, so the
      orchestrator still treats the stream as encodable but de-prioritized).

    ``bytes_per_element_estimate`` is the average codec cost in bytes per
    element (default 0.5 ≈ FP4 equivalent). Tweak for your codec lane.

    Returns ``{tensor_name: float}`` with all values finite and >0.
    Tagged ``sensitivity_derived`` when persisted.
    """
    if bytes_per_element_estimate <= 0.0 or not np.isfinite(
        bytes_per_element_estimate
    ):
        raise JCSPScoreMarginalsError(
            f"bytes_per_element_estimate must be finite and >0 "
            f"(got {bytes_per_element_estimate})"
        )
    if fallback_marginal <= 0.0 or not np.isfinite(fallback_marginal):
        raise JCSPScoreMarginalsError(
            f"fallback_marginal must be finite and >0 (got {fallback_marginal})"
        )
    items = _state_dict_items(model)
    sens_by_name: dict[str, np.ndarray] = {}
    for k, v in sensitivities.items():
        sens_by_name[str(k)] = _coerce_to_numpy_float32(v)
    out: dict[str, float] = {}
    for name, tensor in items:
        arr = _coerce_to_numpy_float32(tensor)
        n = int(arr.size)
        bytes_estimate = max(1.0, float(n) * bytes_per_element_estimate)
        sens_arr = sens_by_name.get(name)
        if sens_arr is None:
            sens_arr = sens_by_name.get(f"{name}.weight")
        if sens_arr is None and name.endswith(".weight"):
            sens_arr = sens_by_name.get(name[: -len(".weight")])
        if sens_arr is not None and sens_arr.size > 0:
            sens_mean = float(np.mean(np.abs(sens_arr)))
            if sens_mean <= 0.0 or not np.isfinite(sens_mean):
                marginal = fallback_marginal
            else:
                marginal = sens_mean / bytes_estimate
        else:
            marginal = fallback_marginal
        if not np.isfinite(marginal) or marginal <= 0.0:
            marginal = fallback_marginal
        out[name] = float(marginal)
    return out


def _validate_marginals(marginals: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(marginals, Mapping):
        raise JCSPScoreMarginalsError("marginals must be a Mapping")
    if not marginals:
        raise JCSPScoreMarginalsError("marginals must be non-empty")
    out: dict[str, float] = {}
    for k, v in marginals.items():
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise JCSPScoreMarginalsError(
                f"marginals[{k!r}] is not numeric: {v!r}"
            ) from exc
        if not np.isfinite(fv):
            raise JCSPScoreMarginalsError(
                f"marginals[{k!r}] must be finite (got {fv})"
            )
        if fv <= 0.0:
            raise JCSPScoreMarginalsError(
                f"marginals[{k!r}] must be > 0 (got {fv})"
            )
        out[str(k)] = fv
    return out


def save_marginals(
    path: str | Path,
    marginals: Mapping[str, float],
    *,
    source: str,
    evidence: str,
    model_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist marginals to a schema-versioned JSON envelope.

    Args:
        path: output JSON path. Parent directory must exist.
        marginals: ``{tensor_name: float}``. All values must be finite > 0.
        source: one of ``ALLOWED_SOURCES``. ``contest_cuda_calibrated``
            requires ``model_sha256``.
        evidence: short string describing the artifact's provenance
            (e.g., "uniform 1e-6 for plumbing smoke",
            "derived from sensitivities at <git_sha>").
        model_sha256: required for ``contest_cuda_calibrated``; otherwise
            optional.
        extra: optional extra fields merged into the envelope (free-form).

    Returns the resolved Path.
    """
    if source not in ALLOWED_SOURCES:
        raise JCSPScoreMarginalsError(
            f"source must be one of {sorted(ALLOWED_SOURCES)}; got {source!r}"
        )
    if source == "contest_cuda_calibrated" and not model_sha256:
        raise JCSPScoreMarginalsError(
            "source='contest_cuda_calibrated' requires model_sha256"
        )
    if not evidence or not isinstance(evidence, str):
        raise JCSPScoreMarginalsError("evidence must be a non-empty string")
    validated = _validate_marginals(marginals)
    path = Path(path)
    if not path.parent.exists():
        raise JCSPScoreMarginalsError(
            f"parent directory does not exist: {path.parent}"
        )
    # Use ``score_marginals`` (not ``marginals``) so the
    # ``tac.jcsp_stream_builder.load_jcsp_score_marginals`` loader picks
    # up the artifact via its canonical second branch.
    envelope: dict[str, Any] = {
        "schema": JCSP_SCORE_MARGINALS_SCHEMA,
        "source": source,
        "evidence": evidence,
        "n_streams": len(validated),
        "score_marginals": validated,
    }
    if model_sha256 is not None:
        envelope["model_sha256"] = str(model_sha256)
    if extra:
        for k, v in extra.items():
            if k in envelope:
                raise JCSPScoreMarginalsError(
                    f"extra field {k!r} collides with reserved envelope field"
                )
            envelope[str(k)] = v
    serialized = json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False)
    envelope["envelope_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_marginals(path: str | Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Load and validate a marginals artifact.

    Returns ``(marginals, envelope_metadata)`` where ``envelope_metadata``
    contains every non-marginal field (schema, source, evidence, …).
    """
    path = Path(path)
    if not path.exists():
        raise JCSPScoreMarginalsError(f"marginals artifact not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise JCSPScoreMarginalsError(
            f"marginals envelope is not a JSON object: {type(raw).__name__}"
        )
    schema = raw.get("schema")
    if schema != JCSP_SCORE_MARGINALS_SCHEMA:
        raise JCSPScoreMarginalsError(
            f"unexpected schema {schema!r}; expected "
            f"{JCSP_SCORE_MARGINALS_SCHEMA!r}"
        )
    source = raw.get("source")
    if source not in ALLOWED_SOURCES:
        raise JCSPScoreMarginalsError(
            f"unknown source tag in artifact: {source!r}"
        )
    if source == "contest_cuda_calibrated" and not raw.get("model_sha256"):
        raise JCSPScoreMarginalsError(
            "contest_cuda_calibrated artifact missing model_sha256"
        )
    marginals = raw.get("score_marginals")
    if marginals is None:
        # Tolerate the legacy ``marginals`` key for older artifacts.
        marginals = raw.get("marginals")
    if not isinstance(marginals, dict):
        raise JCSPScoreMarginalsError(
            "envelope.score_marginals must be a JSON object"
        )
    validated = _validate_marginals(marginals)
    metadata = {
        k: v for k, v in raw.items() if k not in ("score_marginals", "marginals")
    }
    return validated, metadata


__all__ = [
    "ALLOWED_SOURCES",
    "JCSPScoreMarginalsError",
    "JCSP_SCORE_MARGINALS_SCHEMA",
    "derive_marginals_from_sensitivity_map",
    "derive_uniform_planning_marginals",
    "load_marginals",
    "save_marginals",
]
