#!/usr/bin/env python3
"""Build conservative A5 q-bit schedules from per-pair score marginals."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec.frame_conditional_bit_budget import pack_frame_conditional_q_bits  # noqa: E402
from tac.repo_io import json_text, read_json, repo_relative, sha256_bytes, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA = "pr101_a5_score_marginal_qbits_schedule.v1"
DEFAULT_SCORE_MARGINAL_MANIFEST = Path(
    "experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/"
    "per_pair_score_marginals.advisory.json"
)
MARGINAL_SOURCE_KEYS = {
    "boundary": "per_pair_boundary_mass",
    "low_margin": "per_pair_low_margin_mass",
    "mean_margin": "per_pair_mean_logit_margin",
    "p10_margin": "per_pair_p10_logit_margin",
    "score": "per_pair_score_marginals",
    "seg": "per_pair_seg_proxy_raw",
    "pose": "per_pair_pose_proxy_raw",
    "raw_score": "per_pair_score_proxy_raw",
}


class A5ScoreMarginalQBitsScheduleError(ValueError):
    """Raised when a score-marginal q-bit schedule cannot be built safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-marginal-manifest",
        type=Path,
        default=DEFAULT_SCORE_MARGINAL_MANIFEST,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--candidate-id", default="a5_score_marginal_trust_region_qbits")
    parser.add_argument("--base-q-bits", type=int, default=8)
    parser.add_argument("--low-q-bits", type=int, default=6)
    parser.add_argument(
        "--low-fraction",
        type=float,
        default=0.50,
        help="Fraction of lowest-marginal pairs assigned --low-q-bits.",
    )
    parser.add_argument(
        "--marginal-source",
        choices=sorted(MARGINAL_SOURCE_KEYS),
        default="score",
        help=(
            "Which per-pair marginal vector to rank. 'score' is the legacy "
            "blended score marginal; 'seg' keeps SegNet-sensitive pairs at "
            "base q bits; 'boundary' and margin sources consume manifests "
            "from tools/build_segnet_boundary_marginals.py."
        ),
    )
    parser.add_argument(
        "--blend-sources",
        default=None,
        help=(
            "Comma-separated marginal sources to rank as a blended risk vector. "
            "When set, this overrides --marginal-source. Example: seg,boundary,low_margin."
        ),
    )
    parser.add_argument(
        "--blend-mode",
        choices=("max", "mean", "min"),
        default="max",
        help=(
            "How normalized risk ranks are combined for --blend-sources. max is "
            "the conservative union: a pair is protected if any source marks it risky."
        ),
    )
    parser.add_argument("--latent-dim", type=int, default=28)
    return parser.parse_args(argv)


def build_schedule(
    *,
    score_marginal_manifest_path: Path,
    candidate_id: str = "a5_score_marginal_trust_region_qbits",
    base_q_bits: int = 8,
    low_q_bits: int = 6,
    low_fraction: float = 0.50,
    marginal_source: str = "score",
    blend_sources: str | Sequence[str] | None = None,
    blend_mode: str = "max",
    latent_dim: int = 28,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    score_marginal_manifest_path = _resolve(score_marginal_manifest_path, repo_root)
    manifest = _load_json_object(score_marginal_manifest_path)
    n_pairs = _positive_int(manifest.get("n_pairs"), "n_pairs")
    blend_source_list = _parse_blend_sources(blend_sources)
    if blend_source_list:
        marginals, blend_metadata = _blend_risk_vectors(
            manifest=manifest,
            sources=blend_source_list,
            mode=blend_mode,
            expected_len=n_pairs,
        )
        marginal_source_effective = "blend"
        marginal_key = "normalized_rank_blend"
    else:
        marginal_key = MARGINAL_SOURCE_KEYS.get(marginal_source)
        if marginal_key is None:
            raise A5ScoreMarginalQBitsScheduleError(
                f"unknown marginal_source={marginal_source!r}"
            )
        marginals = _finite_vector(
            manifest.get(marginal_key),
            marginal_key,
            expected_len=n_pairs,
        )
        marginal_source_effective = marginal_source
        blend_metadata = None
    base_q_bits = _q_bit_int(base_q_bits, "base_q_bits")
    low_q_bits = _q_bit_int(low_q_bits, "low_q_bits")
    if low_q_bits > base_q_bits:
        raise A5ScoreMarginalQBitsScheduleError("low_q_bits must be <= base_q_bits")
    if not np.isfinite(low_fraction) or low_fraction < 0.0 or low_fraction > 1.0:
        raise A5ScoreMarginalQBitsScheduleError("low_fraction must be in [0, 1]")
    latent_dim = _positive_int(latent_dim, "latent_dim")
    source_q_bits, source_q_bits_semantics = _source_q_bits_or_baseline(
        manifest=manifest,
        n_pairs=n_pairs,
        base_q_bits=base_q_bits,
    )

    low_count = int(np.floor(float(n_pairs) * float(low_fraction)))
    order = np.lexsort((np.arange(n_pairs, dtype=np.int64), marginals))
    low_pair_indices = np.sort(order[:low_count])
    q_bits = np.full(n_pairs, base_q_bits, dtype=np.uint8)
    q_bits[low_pair_indices] = low_q_bits
    sideinfo = pack_frame_conditional_q_bits(q_bits)
    raw_latent_payload_bits = int(q_bits.astype(np.int64).sum() * latent_dim)
    raw_latent_payload_bytes = (raw_latent_payload_bits + 7) // 8

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[advisory q-bit schedule; no eval]",
        "evidence_semantics": (
            "score-marginal trust-region q-bit schedule only; build a runtime "
            "packet and run exact eval before any score or promotion claim"
        ),
        "source_artifacts": {
            "score_marginal_manifest": _artifact_ref(
                score_marginal_manifest_path, repo_root
            )
        },
        "source_schema": manifest.get("schema"),
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "base_q_bits": base_q_bits,
        "low_q_bits": low_q_bits,
        "low_fraction": float(low_fraction),
        "marginal_source": marginal_source_effective,
        "marginal_source_key": marginal_key,
        "blend_sources": blend_source_list,
        "blend_mode": blend_mode if blend_source_list else None,
        "blend_metadata": blend_metadata,
        "low_pair_count": int(low_count),
        "low_pair_indices_sha256": sha256_bytes(
            low_pair_indices.astype("<i4").tobytes()
        ),
        "q_bits_sideinfo": {
            "bytes": len(sideinfo),
            "sha256": sha256_bytes(sideinfo),
        },
        "q_bits_summary": _q_bits_summary(q_bits),
        "source_q_bits_summary": _q_bits_summary(source_q_bits.astype(np.uint8)),
        "source_q_bits_semantics": source_q_bits_semantics,
        "raw_latent_payload_bits": raw_latent_payload_bits,
        "raw_latent_payload_bytes": raw_latent_payload_bytes,
        "per_pair_score_marginal_summary": {
            "min": float(marginals.min()),
            "mean": float(marginals.mean()),
            "max": float(marginals.max()),
            "low_pairs_mean": float(marginals[low_pair_indices].mean())
            if low_count
            else None,
            "base_pairs_mean": float(marginals[np.setdiff1d(np.arange(n_pairs), low_pair_indices)].mean())
            if low_count < n_pairs
            else None,
        },
        "alignment": {
            "q_bits_vs_selected_marginal_pearson": _pearson(q_bits, marginals),
            "source_q_bits_vs_selected_marginal_pearson": _pearson(
                source_q_bits, marginals
            ),
            "q_bits_vs_score_marginal_pearson": _optional_pearson(
                q_bits, manifest, "per_pair_score_marginals", n_pairs
            ),
            "q_bits_vs_seg_marginal_pearson": _optional_pearson(
                q_bits, manifest, "per_pair_seg_proxy_raw", n_pairs
            ),
            "q_bits_vs_pose_marginal_pearson": _optional_pearson(
                q_bits, manifest, "per_pair_pose_proxy_raw", n_pairs
            ),
            "q_bits_vs_boundary_mass_pearson": _optional_pearson(
                q_bits, manifest, "per_pair_boundary_mass", n_pairs
            ),
            "q_bits_vs_low_margin_mass_pearson": _optional_pearson(
                q_bits, manifest, "per_pair_low_margin_mass", n_pairs
            ),
        },
        "per_pair_q_bits": [int(value) for value in q_bits.tolist()],
        "reactivation_criteria": [
            "Build a runtime packet with --recompute-wire-contract-for-q-bits.",
            "Run local advisory CPU eval only after lane claim.",
            "Run exact contest-CUDA and contest-CPU before promotion.",
        ],
    }
    payload["manifest_sha256_excluding_self"] = sha256_bytes(
        json_text(payload).encode("utf-8")
    )
    return payload


def _q_bits_summary(q_bits: np.ndarray) -> dict[str, Any]:
    q_bits = np.asarray(q_bits, dtype=np.uint8)
    values, counts = np.unique(q_bits, return_counts=True)
    return {
        "min": int(q_bits.min()),
        "max": int(q_bits.max()),
        "mean": float(q_bits.mean()),
        "sha256": sha256_bytes(q_bits.tobytes()),
        "unique_counts": {
            str(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
    }


def _finite_vector(value: Any, label: str, *, expected_len: int) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        raise A5ScoreMarginalQBitsScheduleError(f"{label} must be an array")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise A5ScoreMarginalQBitsScheduleError(f"{label} must be 1-D")
    if arr.size != expected_len:
        raise A5ScoreMarginalQBitsScheduleError(
            f"{label} length {arr.size} != expected {expected_len}"
        )
    if not np.isfinite(arr).all():
        raise A5ScoreMarginalQBitsScheduleError(f"{label} must contain finite values")
    return arr


def _validate_q_bits(values: np.ndarray, label: str) -> None:
    if (values < 1).any() or (values > 8).any():
        raise A5ScoreMarginalQBitsScheduleError(f"{label} values must be in [1, 8]")
    if not np.array_equal(values, np.floor(values)):
        raise A5ScoreMarginalQBitsScheduleError(f"{label} values must be integer")


def _source_q_bits_or_baseline(
    *,
    manifest: dict[str, Any],
    n_pairs: int,
    base_q_bits: int,
) -> tuple[np.ndarray, str]:
    if "per_pair_q_bits" not in manifest:
        return (
            np.full(n_pairs, base_q_bits, dtype=np.int64),
            "synthetic_all_base_q_bits_missing_source_vector",
        )
    source_q_bits = _finite_vector(
        manifest.get("per_pair_q_bits"),
        "per_pair_q_bits",
        expected_len=n_pairs,
    ).astype(np.int64)
    _validate_q_bits(source_q_bits, "per_pair_q_bits")
    return source_q_bits, "manifest_per_pair_q_bits"


def _parse_blend_sources(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    sources = [part for part in parts if part]
    unknown = [source for source in sources if source not in MARGINAL_SOURCE_KEYS]
    if unknown:
        raise A5ScoreMarginalQBitsScheduleError(
            f"unknown blend source(s): {unknown}; expected {sorted(MARGINAL_SOURCE_KEYS)}"
        )
    if len(set(sources)) != len(sources):
        raise A5ScoreMarginalQBitsScheduleError("blend sources must be unique")
    return sources


def _risk_vector_for_source(
    manifest: dict[str, Any],
    source: str,
    *,
    expected_len: int,
) -> tuple[np.ndarray, str, str]:
    key = MARGINAL_SOURCE_KEYS[source]
    values = _finite_vector(manifest.get(key), key, expected_len=expected_len)
    if source in {"mean_margin", "p10_margin"}:
        # Lower logit margins are riskier; invert so larger means more protected.
        return -values, key, "lower_value_is_higher_risk"
    return values, key, "higher_value_is_higher_risk"


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.lexsort((np.arange(values.size, dtype=np.int64), values))
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / float(values.size - 1)


def _blend_risk_vectors(
    *,
    manifest: dict[str, Any],
    sources: Sequence[str],
    mode: str,
    expected_len: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode not in {"max", "mean", "min"}:
        raise A5ScoreMarginalQBitsScheduleError("blend_mode must be one of max, mean, min")
    if not sources:
        raise A5ScoreMarginalQBitsScheduleError("blend_sources must not be empty")
    ranked: list[np.ndarray] = []
    components: list[dict[str, str]] = []
    for source in sources:
        risk, key, orientation = _risk_vector_for_source(
            manifest,
            source,
            expected_len=expected_len,
        )
        ranked.append(_rank01(risk))
        components.append(
            {
                "source": source,
                "key": key,
                "orientation": orientation,
                "rank_semantics": "0=lowest_risk_low_q_candidate, 1=highest_risk_protect",
            }
        )
    stacked = np.stack(ranked, axis=0)
    if mode == "max":
        blended = stacked.max(axis=0)
    elif mode == "min":
        blended = stacked.min(axis=0)
    else:
        blended = stacked.mean(axis=0)
    return blended, {
        "sources": list(sources),
        "mode": mode,
        "components": components,
        "semantics": (
            "blended normalized risk ranks; q-bit schedule assigns low bits "
            "to lowest blended-risk pairs and keeps high-risk pairs at base bits"
        ),
    }


def _q_bit_int(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 8:
        raise A5ScoreMarginalQBitsScheduleError(f"{label} must be an integer in [1, 8]")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise A5ScoreMarginalQBitsScheduleError(f"{label} must be a positive integer")
    return value


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size == 0:
        return 0.0
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _optional_pearson(
    q_bits: np.ndarray, manifest: dict[str, Any], key: str, expected_len: int
) -> float | None:
    if key not in manifest:
        return None
    values = _finite_vector(manifest.get(key), key, expected_len=expected_len)
    return _pearson(q_bits, values)


def _artifact_ref(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise A5ScoreMarginalQBitsScheduleError(f"{path} must contain a JSON object")
    return payload


def _resolve(path: Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        payload = build_schedule(
            score_marginal_manifest_path=args.score_marginal_manifest,
            candidate_id=args.candidate_id,
            base_q_bits=args.base_q_bits,
            low_q_bits=args.low_q_bits,
            low_fraction=args.low_fraction,
            marginal_source=args.marginal_source,
            blend_sources=args.blend_sources,
            blend_mode=args.blend_mode,
            latent_dim=args.latent_dim,
            repo_root=REPO_ROOT,
        )
        payload = attach_tool_run_manifest(
            payload,
            tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
            argv=raw_argv,
            input_paths=[args.score_marginal_manifest],
            repo_root=REPO_ROOT,
            output_path=args.json_out,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FATAL: A5 q-bit schedule rejected: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
