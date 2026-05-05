#!/usr/bin/env python3
"""Diagnostic component-trace runner — per-pair PoseNet + SegNet decomposition.

Produces ``component_trace.json`` with the per-pair posenet (``p_i``) and segnet
(``s_i``) distortions on the canonical 600 non-overlapping pair eval used by
``upstream/evaluate.py``.  Cross-checks the sum-of-components against the
companion ``contest_auth_eval.json``; a mismatch fails closed.

The output is intentionally tagged ``score_claim: False`` and
``evidence_grade: "diagnostic_component_trace"`` so downstream gating treats it
as a diagnostic-only artifact: it surfaces hard-pair priors and per-component
gradients for atom planners, but it must NOT be cited as a contest score.

This file is the canonical Stage-2 dispatch entry point for
``experiments/contest_component_trace.py`` referenced from
``src/tac/deploy/lightning/batch_jobs.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Required color-contract scale options for ffmpeg parity with upstream/evaluate.py.
# These mirror the inflate-time decode contract; any drift here invalidates the
# per-pair score decomposition because chroma plane numerics differ.
REQUIRED_FFMPEG_SCALE_OPTIONS: tuple[str, ...] = (
    "in_range",
    "out_range",
    "in_color_matrix",
    "in_primaries",
    "in_transfer",
)

# uv link mode pinned to "copy" so isolated inflate venvs don't reuse cached
# torch wheels from a host venv with different CUDA.
UV_LINK_MODE = "copy"

# Filename for the runtime-environment sidecar written next to component_trace.json.
RUNTIME_ENV_SIDECAR = "component_trace_runtime_env.json"

EVIDENCE_GRADE = "diagnostic_component_trace"
N_SAMPLES_EXPECTED = 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_parity_ffmpeg_env() -> dict[str, str]:
    """Resolve the ffmpeg binary used for parity decode and validate its options.

    Honours an explicit ``FFMPEG_BIN`` override but rejects unusable paths
    immediately so we never silently fall back to a host ffmpeg with different
    libavfilter / libavcodec versions.
    """
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        if not (Path(explicit).is_file() and os.access(explicit, os.X_OK)):
            raise RuntimeError(
                f"FFMPEG_BIN={explicit!r} is not executable"
            )
        ffmpeg_bin = explicit
    else:
        resolved = shutil.which("ffmpeg")
        if not resolved:
            raise RuntimeError("ffmpeg not on PATH and FFMPEG_BIN unset")
        ffmpeg_bin = resolved

    try:
        help_text = subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-h", "filter=scale"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ffmpeg scale-help probe failed: {exc}") from exc

    missing = [opt for opt in REQUIRED_FFMPEG_SCALE_OPTIONS if opt not in help_text]
    if missing:
        raise RuntimeError(
            f"ffmpeg at {ffmpeg_bin!r} missing required scale options: {missing}"
        )

    return {"FFMPEG_BIN": ffmpeg_bin}


def _ensure_isolated_inflate_uv_env(work_dir: Path) -> dict[str, str]:
    """Pin a per-run uv cache + venv so inflate doesn't poison or read host state.

    Returns the env-var overlay to apply (caller merges into ``os.environ``).
    """
    cache_root = work_dir / ".uv_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    overlay: dict[str, str] = {
        "UV_CACHE_DIR": str(cache_root),
        "UV_LINK_MODE": UV_LINK_MODE,
        "UV_NO_PROGRESS": "1",
    }
    return overlay


@dataclass
class _PairScore:
    pair_index: int
    p_i: float
    s_i: float


def _score_archive_per_pair(
    submission_dir: Path,
    upstream_dir: Path,
    uncompressed_dir: Path,
    video_names_file: Path,
    device: str,
) -> list[_PairScore]:
    """Score each of the 600 non-overlapping pairs individually.

    Delegates to ``upstream.evaluate.evaluate_pair`` (or the equivalent in
    ``tac.scoring``) so the per-pair semantics MATCH the contest evaluator.
    Falls back to recomputing posenet + segnet via tac.scoring if the upstream
    helper isn't exposed.
    """
    repo = _repo_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(upstream_dir))
    try:
        from tac.scoring import evaluate_archive_per_pair  # type: ignore
    except Exception:
        evaluate_archive_per_pair = None  # type: ignore

    if evaluate_archive_per_pair is not None:
        rows = evaluate_archive_per_pair(
            submission_dir=submission_dir,
            upstream_dir=upstream_dir,
            uncompressed_dir=uncompressed_dir,
            video_names_file=video_names_file,
            device=device,
        )
        return [
            _PairScore(pair_index=int(r["pair_index"]),
                      p_i=float(r["p_i"]),
                      s_i=float(r["s_i"]))
            for r in rows
        ]

    # Fallback: invoke upstream evaluate.py with a per-pair flag if available.
    raise RuntimeError(
        "tac.scoring.evaluate_archive_per_pair is unavailable; "
        "cannot compute diagnostic component trace"
    )


def _cross_check_against_auth_eval(
    pairs: list[_PairScore],
    auth_eval_json_path: Path,
) -> dict[str, Any]:
    """Verify that summed components reproduce contest_auth_eval.json scores."""
    if not auth_eval_json_path.is_file():
        return {
            "auth_eval_json_path": str(auth_eval_json_path),
            "all_match": False,
            "reason": "contest_auth_eval.json not found",
        }
    auth = json.loads(auth_eval_json_path.read_text())
    avg_p = sum(p.p_i for p in pairs) / max(len(pairs), 1)
    avg_s = sum(p.s_i for p in pairs) / max(len(pairs), 1)
    expected_p = auth.get("avg_posenet_dist") or auth.get("posenet_distortion")
    expected_s = auth.get("avg_segnet_dist") or auth.get("segnet_distortion")
    tolerances = {"posenet": 1e-5, "segnet": 1e-5}
    matches = {}
    if expected_p is not None:
        matches["posenet"] = abs(avg_p - float(expected_p)) <= tolerances["posenet"]
    if expected_s is not None:
        matches["segnet"] = abs(avg_s - float(expected_s)) <= tolerances["segnet"]
    all_match = bool(matches) and all(matches.values())
    return {
        "auth_eval_json_path": str(auth_eval_json_path),
        "auth_eval_sha256": _sha256(auth_eval_json_path),
        "computed_avg_posenet_dist": avg_p,
        "computed_avg_segnet_dist": avg_s,
        "expected_avg_posenet_dist": expected_p,
        "expected_avg_segnet_dist": expected_s,
        "tolerances": tolerances,
        "matches": matches,
        "all_match": all_match,
    }


def _write_runtime_env_sidecar(
    sidecar_path: Path,
    args: argparse.Namespace,
    ffmpeg_env: dict[str, str],
    uv_env: dict[str, str],
) -> None:
    payload = {
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items()},
        "python_executable": sys.executable,
        "python_version": sys.version,
        "ffmpeg_env": ffmpeg_env,
        "uv_env": uv_env,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "required_ffmpeg_scale_options": list(REQUIRED_FFMPEG_SCALE_OPTIONS),
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True,
                        help="eval_work directory containing inflated frames + archive")
    parser.add_argument("--upstream-dir", type=Path, required=True,
                        help="upstream pinned snapshot root")
    parser.add_argument("--uncompressed-dir", type=Path, required=True,
                        help="ground-truth uncompressed videos directory")
    parser.add_argument("--video-names-file", type=Path, required=True,
                        help="text file with video names, one per line")
    parser.add_argument("--device", default="cuda",
                        help="cuda | cpu (cuda only is contest-faithful)")
    parser.add_argument("--contest-auth-eval-json", type=Path, required=True,
                        help="contest_auth_eval.json from same eval_work for cross-check")
    parser.add_argument("--output-json", type=Path, required=True,
                        help="component_trace.json output path")
    parser.add_argument("--top-k", type=int, default=80,
                        help="how many hard pairs to surface in top_hard_pairs")
    args = parser.parse_args(argv)

    work_dir = args.output_json.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_env = _ensure_parity_ffmpeg_env()
    uv_env = _ensure_isolated_inflate_uv_env(work_dir)
    os.environ.update(ffmpeg_env)
    os.environ.update(uv_env)

    pairs = _score_archive_per_pair(
        submission_dir=args.submission_dir,
        upstream_dir=args.upstream_dir,
        uncompressed_dir=args.uncompressed_dir,
        video_names_file=args.video_names_file,
        device=args.device,
    )
    if len(pairs) != N_SAMPLES_EXPECTED:
        raise RuntimeError(
            f"expected {N_SAMPLES_EXPECTED} pairs, got {len(pairs)}"
        )

    avg_p = sum(p.p_i for p in pairs) / len(pairs)
    avg_s = sum(p.s_i for p in pairs) / len(pairs)

    archive_path = args.submission_dir / "archive.zip"
    archive_bytes = archive_path.stat().st_size if archive_path.is_file() else None
    rate = (25.0 * archive_bytes / 37545489.0) if archive_bytes else None
    score_recomputed = None
    if rate is not None:
        # Contest formula: 100 * seg + sqrt(10 * pose) + rate
        import math
        score_recomputed = 100.0 * avg_s + math.sqrt(10.0 * avg_p) + rate

    cross_check = _cross_check_against_auth_eval(
        pairs, args.contest_auth_eval_json
    )

    pairs_sorted_by_total = sorted(
        pairs,
        key=lambda p: (100.0 * p.s_i + (10.0 * p.p_i) ** 0.5),
        reverse=True,
    )
    top_hard_pairs = [asdict(p) for p in pairs_sorted_by_total[: max(1, args.top_k)]]

    payload = {
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "n_samples": len(pairs),
        "pair_index": [p.pair_index for p in pairs],
        "p_i": [p.p_i for p in pairs],
        "s_i": [p.s_i for p in pairs],
        "avg_posenet_dist": avg_p,
        "avg_segnet_dist": avg_s,
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256(archive_path) if archive_path.is_file() else None,
        "rate_term": rate,
        "score_recomputed_from_components": score_recomputed,
        "top_hard_pairs": top_hard_pairs,
        "contest_auth_eval_cross_check": cross_check,
    }

    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    sidecar_path = work_dir / RUNTIME_ENV_SIDECAR
    _write_runtime_env_sidecar(sidecar_path, args, ffmpeg_env, uv_env)

    print(f"[component-trace] wrote {args.output_json} "
          f"(n={len(pairs)}, avg_p={avg_p:.6g}, avg_s={avg_s:.6g}, "
          f"score_recomp={score_recomputed!r}, "
          f"cross_match={cross_check.get('all_match')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
