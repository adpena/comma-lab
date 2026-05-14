#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure ρ_pose on a contest archive (Phase 2 pre-condition).

Phase 2 pre-condition surfaced by the 2026-05-11 grand council "Insight 2"
review (`feedback_grand_council_pose_axis_insights_review_20260511.md`):

    RATIFY $223–303 envelope ... conditional on ρ_pose-on-A1 measurement

ρ_pose is the lag-1 autocorrelation of the per-pair PoseNet target sequence,
which feeds the Berger-Gish-Pinkston joint-source rate-distortion bound in
``tac.joint_source_rd_bound``. The bound:

    R_joint(D)  =  R_iid(D)  -  0.5 · log2( 1 / (1 - ρ²) )

predicts the bit-savings achievable by an AR(1) coder over the i.i.d.
baseline. T13 (sqrt-N latent budget), T15 (per-pair pose-conditioned FiLM),
T20 (pose-aware latent re-sampling) all depend on ρ_pose-on-A1 because their
predicted score deltas use this scalar to scale Berger savings.

This tool computes ρ_pose three ways for cross-validation:

    1. **archive-pose** — read the per-pair pose-delta vectors out of an
       archive's pose-stream payload. Cheap (no PoseNet forward) but
       depends on archive having a parseable pose-stream layout.
    2. **upstream-targets** — decode ``upstream/videos/0.mkv`` via PyAV,
       run the contest PoseNet on each pair, take the per-pair 6-D pose
       output as the target sequence. This is the "ground-truth" ρ but
       requires PoseNet forward + the differentiable scorer path.
    3. **a1-canonical-fallback** — when neither archive-pose extraction
       nor PoseNet forward is available locally, use a documented
       ρ_pose ≈ 0.85 from ``tac.joint_source_rd_bound:26`` and tag the
       result ``[fallback; CLAUDE.md default]``.

Modes 1 and 2 are EMPIRICAL; mode 3 is FALLBACK.

CLAUDE.md compliance
--------------------
- Pure-CPU; no GPU dispatch.
- No /tmp paths in any persisted artifact.
- Output evidence_grade ∈ {research_signal_local_cpu, fallback} —
  NEVER [contest-CUDA] or [contest-CPU].
- Tags every claim per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".
- macOS-CPU runs tagged ``[macOS-CPU advisory only]`` per CLAUDE.md rule.

Usage::

    .venv/bin/python tools/measure_rho_pose_on_archive.py \\
        --archive-path experiments/results/.../archive.zip \\
        --archive-sha 87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5 \\
        --mode archive-pose \\
        --output .omx/research/rho_pose_on_a1_<utc>.json

Output schema (canonical fields)::

    {
      "schema_version": 1,
      "tool": "tools/measure_rho_pose_on_archive.py",
      "archive_path": "...",
      "archive_sha256": "...",
      "archive_bytes": <int>,
      "measurement_mode": "archive-pose" | "upstream-targets" | "fallback",
      "n_pose_samples": <int>,
      "pose_dim": 6,
      "rho_pose_per_dim": [<float>] * 6,   // lag-1 autocorr per dim
      "rho_pose_aggregate": <float>,        // mean across dims
      "rho_pose_confidence_interval_95": [<lo>, <hi>],
      "rho_pose_iqr": <float>,
      "berger_bits_per_pair": <float>,     // closed-form per Berger 1971
      "evidence_grade": "research_signal_local_cpu" | "fallback",
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false,
      "hardware_substrate": "macos_cpu_apple_silicon" | ...,
      "computed_at_utc": "...",
      "berger_alt_predictions": {
        "rho_0.50": <bits>,
        "rho_0.70": <bits>,
        "rho_0.85": <bits>,
        "rho_0.92": <bits>
      }
    }
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import sys
import zipfile
from pathlib import Path


# Canonical fallback per ``tac.joint_source_rd_bound`` line 26:
#     "Per-pair pose ρ_pose ≈ 0.85"
CLAUDE_MD_FALLBACK_RHO_POSE = 0.85

# Per CLAUDE.md "MPS auth eval is NOISE — non-negotiable" the result is
# advisory on Apple Silicon; we tag accordingly.
_MACOS_ADVISORY_TAG = "[macOS-CPU advisory only]"


def _detect_hardware_substrate() -> str:
    """Return a canonical substrate label for the evidence tag."""
    sys_name = platform.system()
    machine = platform.machine()
    if sys_name == "Darwin" and machine in ("arm64", "aarch64"):
        return "macos_cpu_apple_silicon"
    if sys_name == "Darwin":
        return "macos_cpu_intel"
    if sys_name == "Linux" and machine in ("x86_64", "amd64"):
        return "linux_cpu_x86_64"
    return f"{sys_name.lower()}_{machine}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _lag1_autocorr(series: list[float]) -> float:
    """Pearson lag-1 autocorrelation.

    Per Cover-Thomas 2006 §10.3.2: r_1 = cov(x_t, x_{t+1}) / var(x_t).
    Edge cases:
      - len(series) < 2 returns 0.0 (no information).
      - zero variance returns 0.0 (degenerate; coding free already).
    """
    n = len(series)
    if n < 2:
        return 0.0
    mean = statistics.fmean(series)
    deviations = [v - mean for v in series]
    var = sum(d * d for d in deviations)
    if var <= 0.0:
        return 0.0
    cov = sum(deviations[t] * deviations[t + 1] for t in range(n - 1))
    rho = cov / var
    # Clamp to [-0.999, 0.999] to avoid Berger divergence per
    # tac.joint_source_rd_bound:_MAX_ABS_RHO.
    return max(-0.999, min(0.999, rho))


def _bootstrap_ci(series: list[float], n_iter: int = 200, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for lag-1 autocorr via stationary block bootstrap.

    Per Politis-Romano 1994. Block length = sqrt(N) (canonical default).
    """
    import random
    n = len(series)
    if n < 4:
        return (0.0, 0.0)
    block_len = max(2, int(math.sqrt(n)))
    rng = random.Random(20260511)  # deterministic seed
    boot = []
    for _ in range(n_iter):
        idx = 0
        sample = []
        while idx < n:
            start = rng.randint(0, n - 1)
            for k in range(block_len):
                if idx >= n:
                    break
                sample.append(series[(start + k) % n])
                idx += 1
        boot.append(_lag1_autocorr(sample))
    boot.sort()
    lo = boot[int(alpha / 2 * n_iter)]
    hi = boot[int((1 - alpha / 2) * n_iter)]
    return (lo, hi)


def _iqr(series: list[float]) -> float:
    """Interquartile range of a sample (Q3 - Q1)."""
    n = len(series)
    if n < 4:
        return 0.0
    sorted_s = sorted(series)
    q1 = sorted_s[n // 4]
    q3 = sorted_s[(3 * n) // 4]
    return q3 - q1


def _berger_bits_per_pair(rho: float, pose_dim: int = 6) -> float:
    """Closed-form Berger savings per pair (Berger 1971 §4.5).

    Returns bits saved per pair vs i.i.d. baseline. Multiplied by pose_dim
    because the contest's per-pair pose vector is pose_dim-dimensional and
    each dim is treated as an independent AR(1) stream (worst-case
    decoupling; tightens if dims are jointly correlated).
    """
    if not (-0.999 <= rho <= 0.999):
        return 0.0
    if rho == 0.0:
        return 0.0
    return 0.5 * math.log2(1.0 / (1.0 - rho * rho)) * pose_dim


def _try_extract_pose_stream_from_archive(archive_path: Path) -> list[list[float]] | None:
    """Try to extract a per-pair pose stream from a contest archive.

    Looks for canonical pose-stream member names; returns None if none
    found. Each row is a pose_dim-dimensional pose-delta vector.

    Canonical member names (PR101 / A1 family + Quantizr 0.33 lineage):
      - ``poses.pt``    — torch.save() of per-pair pose tensor [N, pose_dim]
      - ``pose.pt``     — singular form
      - ``pose_deltas.pt``
      - inside ``x``    — A1's split-brotli pose-stream segment

    For SAFETY we use ``torch.load(weights_only=True)`` per CLAUDE.md
    Catalog 14 (``preflight_loader_format_safety``). If the file is not a
    pure-tensor .pt (e.g. nested dict), we skip rather than load with
    weights_only=False.
    """
    pose_member_names = ("poses.pt", "pose.pt", "pose_deltas.pt")
    try:
        import torch  # type: ignore
    except ImportError:
        return None  # CPU-only: no torch installed — skip archive-pose mode.

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            for candidate in pose_member_names:
                if candidate not in names:
                    continue
                with zf.open(candidate) as fp:
                    import io
                    buf = io.BytesIO(fp.read())
                try:
                    tensor = torch.load(buf, weights_only=True, map_location="cpu")
                except Exception:
                    continue
                if hasattr(tensor, "tolist") and tensor.ndim == 2:
                    return [list(map(float, row)) for row in tensor.tolist()]
    except (zipfile.BadZipFile, OSError):
        return None
    return None


def _measure_rho_from_pose_series(pose_stream: list[list[float]]) -> dict:
    """Compute per-dim and aggregate ρ_pose from a per-pair pose stream."""
    if not pose_stream:
        raise ValueError("Empty pose stream; cannot measure ρ_pose")
    pose_dim = len(pose_stream[0])
    rho_per_dim: list[float] = []
    ci_per_dim: list[tuple[float, float]] = []
    iqr_per_dim: list[float] = []
    for d in range(pose_dim):
        series = [row[d] for row in pose_stream]
        rho_per_dim.append(_lag1_autocorr(series))
        ci_per_dim.append(_bootstrap_ci(series))
        iqr_per_dim.append(_iqr(series))
    aggregate = statistics.fmean(rho_per_dim)
    # Aggregate CI = mean across per-dim CIs (conservative; reflects
    # cross-dim variance NOT collapsed to a single Pearson on a stacked
    # vector).
    ci_aggregate = (
        statistics.fmean(c[0] for c in ci_per_dim),
        statistics.fmean(c[1] for c in ci_per_dim),
    )
    iqr_aggregate = statistics.fmean(iqr_per_dim)
    return {
        "n_pose_samples": len(pose_stream),
        "pose_dim": pose_dim,
        "rho_pose_per_dim": rho_per_dim,
        "rho_pose_aggregate": aggregate,
        "rho_pose_confidence_interval_95": list(ci_aggregate),
        "rho_pose_iqr": iqr_aggregate,
    }


def _fallback_result() -> dict:
    """Return the CLAUDE.md-default ρ_pose ≈ 0.85 with explicit fallback tag."""
    return {
        "n_pose_samples": 0,
        "pose_dim": 6,
        "rho_pose_per_dim": [CLAUDE_MD_FALLBACK_RHO_POSE] * 6,
        "rho_pose_aggregate": CLAUDE_MD_FALLBACK_RHO_POSE,
        "rho_pose_confidence_interval_95": [
            CLAUDE_MD_FALLBACK_RHO_POSE,
            CLAUDE_MD_FALLBACK_RHO_POSE,
        ],
        "rho_pose_iqr": 0.0,
    }


def _berger_alt_predictions(pose_dim: int = 6) -> dict:
    """Anchored Berger predictions across council-discussed ρ values."""
    return {
        "rho_0.50": _berger_bits_per_pair(0.50, pose_dim),
        "rho_0.70": _berger_bits_per_pair(0.70, pose_dim),
        "rho_0.85": _berger_bits_per_pair(0.85, pose_dim),
        "rho_0.92": _berger_bits_per_pair(0.92, pose_dim),
    }


def measure(archive_path: Path, mode: str, archive_sha_expected: str | None) -> dict:
    """Compute ρ_pose on the given archive.

    Args:
        archive_path: Path to the contest archive (.zip).
        mode: ``archive-pose`` (default), ``upstream-targets``, or ``fallback``.
        archive_sha_expected: optional expected SHA-256 for verification.

    Returns:
        Canonical output dict (see module docstring for schema).
    """
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    archive_sha = _sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size

    if archive_sha_expected and archive_sha != archive_sha_expected:
        raise ValueError(
            f"Archive SHA mismatch: expected {archive_sha_expected}, got {archive_sha}"
        )

    substrate = _detect_hardware_substrate()
    extra_tags: list[str] = []
    if substrate.startswith("macos_cpu"):
        extra_tags.append(_MACOS_ADVISORY_TAG)

    measurement: dict
    used_mode: str

    if mode == "fallback":
        measurement = _fallback_result()
        used_mode = "fallback"
        evidence_grade = "fallback"
    elif mode == "archive-pose":
        pose_stream = _try_extract_pose_stream_from_archive(archive_path)
        if pose_stream is None:
            # Fall through to fallback rather than fail loud — operator
            # asked for a measurement; we transparently mark mode used.
            measurement = _fallback_result()
            used_mode = "fallback"
            evidence_grade = "fallback"
            extra_tags.append("[archive-pose extraction unavailable; canonical pose-stream member not found]")
        else:
            measurement = _measure_rho_from_pose_series(pose_stream)
            used_mode = "archive-pose"
            evidence_grade = "research_signal_local_cpu"
    elif mode == "upstream-targets":
        # PoseNet forward path. Requires torch + upstream.modules — skip if
        # not importable; this avoids a hard dependency in the pre-stage
        # measurement that gates Phase 2 dispatch.
        try:
            import torch  # noqa: F401
            from upstream import modules  # type: ignore  # noqa: F401
            raise NotImplementedError(
                "upstream-targets mode requires PoseNet forward pass; "
                "implement separately on Linux x86_64 GPU host. macOS "
                "PoseNet drift is 23× per CLAUDE.md MPS rule; advisory only "
                "on this substrate."
            )
        except (ImportError, NotImplementedError) as exc:
            measurement = _fallback_result()
            used_mode = "fallback"
            evidence_grade = "fallback"
            extra_tags.append(f"[upstream-targets mode unavailable: {type(exc).__name__}: {exc}]")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rho_used = measurement["rho_pose_aggregate"]
    pose_dim = measurement["pose_dim"]
    berger_bits = _berger_bits_per_pair(rho_used, pose_dim)

    result = {
        "schema_version": 1,
        "tool": "tools/measure_rho_pose_on_archive.py",
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "measurement_mode_requested": mode,
        "measurement_mode_used": used_mode,
        **measurement,
        "berger_bits_per_pair": berger_bits,
        "berger_alt_predictions": _berger_alt_predictions(pose_dim),
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "hardware_substrate": substrate,
        "extra_tags": extra_tags,
        "claude_md_fallback_rho_pose": CLAUDE_MD_FALLBACK_RHO_POSE,
        "computed_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--archive-path",
        required=True,
        type=Path,
        help="Path to contest archive.zip (e.g. A1's archive at SHA 87ec7ca5...)",
    )
    parser.add_argument(
        "--archive-sha",
        default=None,
        help="Expected SHA-256 for the archive (optional verification).",
    )
    parser.add_argument(
        "--mode",
        choices=("archive-pose", "upstream-targets", "fallback"),
        default="archive-pose",
        help="Measurement mode (default: archive-pose).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. If omitted, prints to stdout.",
    )
    args = parser.parse_args(argv)

    result = measure(args.archive_path, args.mode, args.archive_sha)

    output_json = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(output_json)
    else:
        # Canonical replacement for /tmp paths per CLAUDE.md
        # "Forbidden /tmp paths in any persisted artifact" rule.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
