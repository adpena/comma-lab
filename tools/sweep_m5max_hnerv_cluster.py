#!/usr/bin/env python3
"""M5 Max parallel CPU sweep harness for HNeRV-cluster archives.

This tool exploits the empirically-verified ε ≤ 6×10⁻⁶ bound between Apple
Silicon CPU and GHA Linux x86_64 CPU on HNeRV-cluster archives (PR #107
calibration anchor, ``feedback_macos_x86_64_epsilon_calibrated_tag_20260508``)
to provide a $0 / ~25-min-per-eval sweep substrate for inner-loop iteration
on candidate archives.

It is **NOT** a substitute for ``[contest-CPU]`` on shippable archives — the
output tag is ``[macOS-CPU calibrated]`` (HNeRV-cluster only) or
``[macOS-CPU advisory only]`` (non-HNeRV).

Usage (sweep a directory of archives):

    .venv/bin/python tools/sweep_m5max_hnerv_cluster.py \\
        --archives-dir experiments/results/some_lane/candidates/ \\
        --architecture-class hnerv \\
        --output-dir experiments/results/m5max_sweep_$(date -u +%Y%m%dT%H%M%SZ)/ \\
        --max-concurrency 4

Usage (sweep an explicit list of archives via ledger JSONL):

    .venv/bin/python tools/sweep_m5max_hnerv_cluster.py \\
        --candidates-jsonl experiments/results/.../candidates.jsonl \\
        --output-dir experiments/results/m5max_sweep_<ts>/ \\
        --max-concurrency 4

Each candidate produces:
    <output-dir>/<candidate_id>/
        archive.zip            (copy of input)
        provenance.json        (sha256, hardware, etc)
        report.txt             (upstream evaluate.py output verbatim)
        contest_auth_eval.adjudicated.json  (calibrated structured result)

Aggregate output:
    <output-dir>/results.jsonl   (one row per candidate)
    <output-dir>/atoms.jsonl     (typed atoms via sister tool, optional)
    <output-dir>/sweep_manifest.json  (sweep-level metadata)

Promotion verdicts (written to results.jsonl per candidate):
    macos_cpu_score < 0.190  → AUTO_PROMOTE_GHA
    score in [0.190, 0.195]  → OPERATOR_DECISION (silver-band proximity)
    score in (0.195, 0.200)  → LOG_ONLY (between silver and frontier)
    score >= 0.200           → LOG_ONLY (worse than known frontier)

Drift-flag:
    Any candidate whose macOS-CPU score deviates from the architecture-class
    calibration prediction by > 5×10⁻⁵ is flagged ``POTENTIAL_NEW_CLASS``
    (the ε bound may not hold; calibration must be re-validated).

Per CLAUDE.md non-negotiables:
- Tags every score ``[macOS-CPU calibrated]`` or ``[macOS-CPU advisory only]``
- NEVER tags ``[contest-CPU]`` (M5 Max is NOT 1:1 contest-compliant CPU hardware)
- NEVER tags ``[contest-CUDA]`` (M5 Max has no CUDA)
- Forces ``--device cpu`` and disables MPS fallback explicitly
- /tmp paths FORBIDDEN — outputs always under ``experiments/results/``

Cross-references:
- ``feedback_macos_x86_64_epsilon_calibrated_tag_20260508``: tag policy + ε bound
- ``feedback_cuda_cpu_axis_profile_learning_layer_20260508``: per-architecture-class registry
- ``feedback_domain_exploitation_catalog_landed_20260509``: atom A.3.1 (this exploit)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Calibration anchors (PR #107 — feedback_macos_x86_64_epsilon_calibrated_tag_20260508)
PR107_MACOS_CPU_SCORE = 0.19664189
PR107_GHA_CPU_SCORE = 0.19663589
PR107_EPSILON_BOUND = 6.0e-6  # |Δ macOS-CPU − x86_64-CPU|

# Drift threshold: if macOS-CPU score deviates from prior class calibration by
# more than this, flag POTENTIAL_NEW_CLASS. Set 8× the empirical ε (per memo
# discipline rule "calibration ε is per-archive empirical").
DRIFT_FLAG_THRESHOLD = 5.0e-5

# Promotion thresholds (silver-band reference: PR102 = 0.19538)
PROMOTION_THRESHOLDS = {
    "auto_promote_gha": 0.190,        # sub-medal-band; need authoritative confirmation
    "operator_decision_high": 0.195,  # silver-band proximity
    "log_only_high": 0.200,           # below current frontier band
}

# Architecture classes that have a calibration anchor (HNeRV cluster only)
CALIBRATED_CLASSES = {"hnerv"}
# Aliases mapped to "hnerv" by tac.optimization.cuda_cpu_axis_calibration
HNERV_ALIASES = {
    "hnerv", "hnerv_ft_microcodec", "hnerv_lc_v2", "hnerv_lc_ac",
    "hnerv_microcodec", "ff_packed_brotli_hnerv",
}

CONTEST_AUTH_EVAL_TOOL = REPO / "experiments" / "contest_auth_eval.py"
DEFAULT_UPSTREAM_DIR = REPO / "upstream"
DEFAULT_VIDEO_NAMES = REPO / "upstream" / "public_test_video_names.txt"


# ────────────────────────────────────────────────────────────────────────────
# Data shapes
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CandidateSpec:
    """One archive to evaluate."""
    candidate_id: str
    archive_path: Path
    architecture_class: str
    inflate_sh: Path | None = None  # default: discover near archive
    notes: str = ""

    def display(self) -> str:
        return f"{self.candidate_id} ({self.architecture_class}, {self.archive_path.name})"


@dataclass
class SweepResult:
    """One candidate's eval output."""
    candidate_id: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    architecture_class: str
    macos_cpu_score: float | None
    macos_cpu_avg_segnet_dist: float | None
    macos_cpu_avg_posenet_dist: float | None
    compression_rate: float | None
    n_samples: int | None
    elapsed_seconds: float
    started_utc: str
    completed_utc: str
    macos_cpu_calibrated_tag: str  # one of "[macOS-CPU calibrated]", "[macOS-CPU advisory only]"
    epsilon_bound_used: float | None
    epsilon_band_low: float | None
    epsilon_band_high: float | None
    predicted_contest_cpu_gha: float | None  # only populated for CALIBRATED_CLASSES
    drift_flag: str  # OK | POTENTIAL_NEW_CLASS | NA
    promotion_verdict: str  # AUTO_PROMOTE_GHA | OPERATOR_DECISION | LOG_ONLY | EVAL_FAILED
    eval_failure_reason: str | None
    output_dir: str
    report_text: str | None


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _sha256(path: Path, *, prefix: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:prefix] if prefix else digest


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _normalize_arch_class(arch_class: str) -> str:
    key = arch_class.strip().lower().replace("-", "_")
    if key in HNERV_ALIASES or key.startswith("hnerv"):
        return "hnerv"
    return key or "unknown"


def _calibrated_tag(arch_class: str) -> str:
    norm = _normalize_arch_class(arch_class)
    if norm in CALIBRATED_CLASSES:
        return "[macOS-CPU calibrated]"
    return "[macOS-CPU advisory only]"


def _verify_running_on_apple_silicon() -> None:
    """Guard against accidentally running this on x86_64 (where the
    calibration story is different and the tag is wrong)."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system != "darwin" or machine not in {"arm64", "aarch64"}:
        raise SystemExit(
            f"sweep_m5max_hnerv_cluster.py requires macOS Apple Silicon "
            f"(detected: system={system!r}, machine={machine!r}). "
            f"For Linux x86_64 use scripts/dispatch_cpu_eval_via_github_actions.py."
        )


def _discover_inflate_sh(archive_path: Path) -> Path | None:
    """Look for an inflate.sh adjacent to the archive (submission_dir layout)
    or in a sibling submission_dir/."""
    parent = archive_path.parent
    candidates = [
        parent / "inflate.sh",
        parent / "submission_dir" / "inflate.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _hnerv_predicted_contest_cpu(macos_cpu_score: float) -> tuple[float, float, float]:
    """For HNeRV-cluster archives, predicted_contest_cpu_gha = macos_cpu - bias
    where the bias is the calibration anchor's signed Δ (here: PR107 macOS −
    GHA = +6×10⁻⁶, i.e., macOS reads 6×10⁻⁶ higher).

    Returns (point_estimate, band_low, band_high) where band uses ε bound.
    """
    bias = PR107_MACOS_CPU_SCORE - PR107_GHA_CPU_SCORE  # +6e-6
    point = macos_cpu_score - bias
    band_low = point - PR107_EPSILON_BOUND
    band_high = point + PR107_EPSILON_BOUND
    return point, band_low, band_high


def _classify_promotion(macos_cpu_score: float | None) -> str:
    if macos_cpu_score is None:
        return "EVAL_FAILED"
    if macos_cpu_score < PROMOTION_THRESHOLDS["auto_promote_gha"]:
        return "AUTO_PROMOTE_GHA"
    if macos_cpu_score < PROMOTION_THRESHOLDS["operator_decision_high"]:
        return "OPERATOR_DECISION"
    if macos_cpu_score < PROMOTION_THRESHOLDS["log_only_high"]:
        return "LOG_ONLY"
    return "LOG_ONLY"


def _compute_drift_flag(macos_cpu_score: float, arch_class: str,
                        prior_anchor: float | None) -> str:
    """Per CLAUDE.md macos-CPU memo discipline rule #4 (per-archive empirical
    ε): if a candidate's macOS-CPU score deviates from the prior anchor for
    its architecture-class by > 5×10⁻⁵ in absolute terms relative to
    expected scoring band, flag POTENTIAL_NEW_CLASS.

    For HNeRV cluster the anchor reference is PR107 macOS-CPU. Practical
    implementation: we don't have a per-candidate "expected" score (it
    depends on the actual archive), so the drift flag fires only when an
    explicit prior anchor is supplied. Otherwise NA.
    """
    if prior_anchor is None or _normalize_arch_class(arch_class) not in CALIBRATED_CLASSES:
        return "NA"
    if abs(macos_cpu_score - prior_anchor) > DRIFT_FLAG_THRESHOLD:
        return "POTENTIAL_NEW_CLASS"
    return "OK"


# ────────────────────────────────────────────────────────────────────────────
# Eval runner (one candidate)
# ────────────────────────────────────────────────────────────────────────────


def _run_one_eval(spec: CandidateSpec, output_root: Path,
                  upstream_dir: Path, video_names_file: Path,
                  prior_anchor: float | None,
                  inflate_timeout: int, evaluate_timeout: int) -> SweepResult:
    """Run one candidate through experiments/contest_auth_eval.py with
    --device cpu on the local M5 Max. Returns a SweepResult.

    This is the per-task function called by the ThreadPoolExecutor."""
    cand_out = output_root / spec.candidate_id
    cand_out.mkdir(parents=True, exist_ok=True)

    started_utc = _now_iso()
    t0 = time.monotonic()

    inflate_sh = spec.inflate_sh or _discover_inflate_sh(spec.archive_path)
    if inflate_sh is None:
        # Fall back to robust_current; many HNeRV-cluster archives can use it.
        inflate_sh = REPO / "submissions" / "robust_current" / "inflate.sh"

    if not inflate_sh.exists():
        elapsed = time.monotonic() - t0
        return SweepResult(
            candidate_id=spec.candidate_id,
            archive_path=str(spec.archive_path),
            archive_sha256="",
            archive_size_bytes=spec.archive_path.stat().st_size if spec.archive_path.exists() else 0,
            architecture_class=spec.architecture_class,
            macos_cpu_score=None,
            macos_cpu_avg_segnet_dist=None,
            macos_cpu_avg_posenet_dist=None,
            compression_rate=None,
            n_samples=None,
            elapsed_seconds=elapsed,
            started_utc=started_utc,
            completed_utc=_now_iso(),
            macos_cpu_calibrated_tag=_calibrated_tag(spec.architecture_class),
            epsilon_bound_used=None,
            epsilon_band_low=None,
            epsilon_band_high=None,
            predicted_contest_cpu_gha=None,
            drift_flag="NA",
            promotion_verdict="EVAL_FAILED",
            eval_failure_reason=f"no inflate.sh found (tried adjacent + robust_current)",
            output_dir=str(cand_out),
            report_text=None,
        )

    archive_sha = _sha256(spec.archive_path)
    archive_size = spec.archive_path.stat().st_size

    # Use a per-candidate work dir so concurrent evals never collide.
    work_dir = cand_out / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    json_out = cand_out / "contest_auth_eval.adjudicated.json"

    # Force CPU and disable any MPS fallback path.
    env = {**os.environ}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(CONTEST_AUTH_EVAL_TOOL),
        "--archive", str(spec.archive_path),
        "--inflate-sh", str(inflate_sh),
        "--upstream-dir", str(upstream_dir),
        "--video-names-file", str(video_names_file),
        "--device", "cpu",
        "--work-dir", str(work_dir),
        "--json-out", str(json_out),
        "--inflate-timeout", str(inflate_timeout),
        "--evaluate-timeout", str(evaluate_timeout),
    ]

    log_path = cand_out / "eval.log"
    failure_reason: str | None = None
    macos_cpu_score: float | None = None
    avg_seg: float | None = None
    avg_pose: float | None = None
    rate: float | None = None
    n_samples: int | None = None
    report_text: str | None = None

    try:
        with open(log_path, "wb") as logf:
            logf.write(f"# cmd: {' '.join(cmd)}\n# started_utc: {started_utc}\n".encode())
            logf.flush()
            result = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                # Add a generous overall ceiling = inflate + evaluate + 60s overhead
                timeout=inflate_timeout + evaluate_timeout + 60,
            )
        if result.returncode != 0:
            failure_reason = f"contest_auth_eval rc={result.returncode}; see {log_path}"
        elif json_out.exists():
            with open(json_out) as f:
                payload = json.load(f)
            macos_cpu_score = payload.get("canonical_score_recomputed") or payload.get("canonical_score")
            avg_seg = payload.get("avg_segnet_dist")
            avg_pose = payload.get("avg_posenet_dist")
            rate = payload.get("compression_rate")
            n_samples = payload.get("n_samples")
            report_text = payload.get("report_text")
        else:
            failure_reason = f"contest_auth_eval rc=0 but {json_out} missing"
    except subprocess.TimeoutExpired:
        failure_reason = "subprocess.TimeoutExpired (inflate+evaluate exceeded ceiling)"
    except Exception as exc:  # pragma: no cover - defensive
        failure_reason = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - t0
    completed_utc = _now_iso()

    arch_norm = _normalize_arch_class(spec.architecture_class)
    if macos_cpu_score is not None and arch_norm in CALIBRATED_CLASSES:
        point, band_low, band_high = _hnerv_predicted_contest_cpu(macos_cpu_score)
        eps_used = PR107_EPSILON_BOUND
    else:
        point = None
        band_low = None
        band_high = None
        eps_used = None

    drift = _compute_drift_flag(macos_cpu_score, spec.architecture_class, prior_anchor) \
        if macos_cpu_score is not None else "NA"
    verdict = _classify_promotion(macos_cpu_score)

    return SweepResult(
        candidate_id=spec.candidate_id,
        archive_path=str(spec.archive_path),
        archive_sha256=archive_sha,
        archive_size_bytes=archive_size,
        architecture_class=spec.architecture_class,
        macos_cpu_score=macos_cpu_score,
        macos_cpu_avg_segnet_dist=avg_seg,
        macos_cpu_avg_posenet_dist=avg_pose,
        compression_rate=rate,
        n_samples=n_samples,
        elapsed_seconds=elapsed,
        started_utc=started_utc,
        completed_utc=completed_utc,
        macos_cpu_calibrated_tag=_calibrated_tag(spec.architecture_class),
        epsilon_bound_used=eps_used,
        epsilon_band_low=band_low,
        epsilon_band_high=band_high,
        predicted_contest_cpu_gha=point,
        drift_flag=drift,
        promotion_verdict=verdict,
        eval_failure_reason=failure_reason,
        output_dir=str(cand_out),
        report_text=report_text,
    )


# ────────────────────────────────────────────────────────────────────────────
# Candidate loaders
# ────────────────────────────────────────────────────────────────────────────


def _load_candidates_from_dir(archives_dir: Path,
                              architecture_class: str) -> list[CandidateSpec]:
    """Discover archive.zip files under archives_dir. Each subdirectory
    becomes a candidate; the candidate_id is the subdirectory name."""
    specs: list[CandidateSpec] = []
    if not archives_dir.exists():
        raise SystemExit(f"--archives-dir does not exist: {archives_dir}")
    # Direct: archives_dir/*.zip
    for z in sorted(archives_dir.glob("*.zip")):
        cid = z.stem
        specs.append(CandidateSpec(
            candidate_id=cid, archive_path=z.resolve(),
            architecture_class=architecture_class,
        ))
    # Nested: archives_dir/<id>/archive.zip
    for sub in sorted(archives_dir.iterdir()):
        if not sub.is_dir():
            continue
        z = sub / "archive.zip"
        if z.exists():
            specs.append(CandidateSpec(
                candidate_id=sub.name, archive_path=z.resolve(),
                architecture_class=architecture_class,
            ))
    if not specs:
        raise SystemExit(
            f"--archives-dir found no archive.zip files: {archives_dir}. "
            "Expected either archives_dir/*.zip OR archives_dir/<id>/archive.zip."
        )
    return specs


def _load_candidates_from_jsonl(jsonl_path: Path) -> list[CandidateSpec]:
    """Each line is a JSON object with at minimum:
        {"candidate_id": "...", "archive_path": "...", "architecture_class": "..."}
    Optional keys:
        {"inflate_sh": "...", "notes": "..."}
    """
    specs: list[CandidateSpec] = []
    with open(jsonl_path) as f:
        for ln_no, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"--candidates-jsonl line {ln_no}: bad JSON: {exc}")
            cid = row.get("candidate_id")
            ap = row.get("archive_path")
            ac = row.get("architecture_class", "unknown")
            if not cid or not ap:
                raise SystemExit(f"--candidates-jsonl line {ln_no}: missing required keys "
                                 f"candidate_id/archive_path")
            inflate_sh = row.get("inflate_sh")
            specs.append(CandidateSpec(
                candidate_id=cid,
                archive_path=Path(ap).resolve(),
                architecture_class=ac,
                inflate_sh=Path(inflate_sh).resolve() if inflate_sh else None,
                notes=row.get("notes", ""),
            ))
    if not specs:
        raise SystemExit(f"--candidates-jsonl produced 0 candidates: {jsonl_path}")
    return specs


# ────────────────────────────────────────────────────────────────────────────
# Calibration verification (mandatory before sweep)
# ────────────────────────────────────────────────────────────────────────────


def verify_calibration_anchor(anchor_archive: Path, expected_score: float,
                              tolerance: float, output_dir: Path,
                              upstream_dir: Path, video_names_file: Path,
                              architecture_class: str = "hnerv") -> tuple[bool, float | None, str]:
    """Run one M5 Max CPU eval on a known archive and check the score lands
    within ±tolerance of the expected GHA Linux x86_64 score.

    Returns (passed, observed_score, message).
    """
    print(f"[calibrate] verifying M5 Max ↔ x86_64 ε bound on {anchor_archive.name}")
    print(f"[calibrate]   expected (GHA Linux x86_64): {expected_score:.10f}")
    print(f"[calibrate]   tolerance: ±{tolerance:.2e}")

    spec = CandidateSpec(
        candidate_id="_calibration_anchor",
        archive_path=anchor_archive.resolve(),
        architecture_class=architecture_class,
    )
    result = _run_one_eval(
        spec, output_dir, upstream_dir, video_names_file,
        prior_anchor=expected_score,
        inflate_timeout=600, evaluate_timeout=2400,
    )
    if result.macos_cpu_score is None:
        return False, None, f"calibration eval failed: {result.eval_failure_reason}"

    observed = result.macos_cpu_score
    delta = observed - expected_score
    abs_delta = abs(delta)
    msg = (f"observed={observed:.10f} expected={expected_score:.10f} "
           f"Δ={delta:+.2e} (|Δ|={abs_delta:.2e}, tol={tolerance:.2e})")
    print(f"[calibrate] {msg}")

    if abs_delta <= tolerance:
        print("[calibrate] PASS — within tolerance; sweep ε bound holds")
        return True, observed, msg
    print("[calibrate] FAIL — calibration may have shifted; surface to operator")
    return False, observed, msg


# ────────────────────────────────────────────────────────────────────────────
# Main sweep driver
# ────────────────────────────────────────────────────────────────────────────


def run_sweep(specs: list[CandidateSpec], output_root: Path,
              max_concurrency: int, upstream_dir: Path,
              video_names_file: Path, prior_anchor: float | None,
              inflate_timeout: int, evaluate_timeout: int) -> list[SweepResult]:
    """Fan out N concurrent CPU evals and collect SweepResults."""
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.jsonl"
    # Truncate / start fresh
    results_path.write_text("")

    print(f"[sweep] dispatching {len(specs)} candidate(s) at max_concurrency={max_concurrency}")
    print(f"[sweep] output_root={output_root}")
    results: list[SweepResult] = []

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        future_to_spec = {
            ex.submit(_run_one_eval, spec, output_root, upstream_dir,
                      video_names_file, prior_anchor, inflate_timeout,
                      evaluate_timeout): spec
            for spec in specs
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                res = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[sweep] !! {spec.candidate_id} crashed: {exc}")
                continue
            results.append(res)
            score_str = f"{res.macos_cpu_score:.10f}" if res.macos_cpu_score is not None else "N/A"
            print(f"[sweep] {res.candidate_id}: score={score_str} "
                  f"verdict={res.promotion_verdict} drift={res.drift_flag} "
                  f"elapsed={res.elapsed_seconds:.1f}s")
            with open(results_path, "a") as f:
                f.write(json.dumps(asdict(res), default=str) + "\n")

    return results


def _emit_sweep_manifest(output_root: Path, results: list[SweepResult],
                         calibration_msg: str | None,
                         args: argparse.Namespace) -> Path:
    """Aggregate run-level metadata (one file per sweep)."""
    manifest = {
        "schema_version": 1,
        "tool": "tools/sweep_m5max_hnerv_cluster.py",
        "started_utc": _now_iso(),
        "macos_cpu_calibrated_tag_policy_ref":
            "feedback_macos_x86_64_epsilon_calibrated_tag_20260508",
        "calibration_anchor_message": calibration_msg,
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
        },
        "thresholds": {
            "drift_flag_threshold": DRIFT_FLAG_THRESHOLD,
            "promotion": PROMOTION_THRESHOLDS,
            "epsilon_bound_pr107": PR107_EPSILON_BOUND,
        },
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_candidates": len(results),
        "summary": _summarize(results),
    }
    out = output_root / "sweep_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def _summarize(results: list[SweepResult]) -> dict[str, Any]:
    n_ok = sum(1 for r in results if r.macos_cpu_score is not None)
    n_failed = sum(1 for r in results if r.macos_cpu_score is None)
    by_verdict: dict[str, int] = {}
    for r in results:
        by_verdict[r.promotion_verdict] = by_verdict.get(r.promotion_verdict, 0) + 1
    drift_flagged = [r.candidate_id for r in results if r.drift_flag == "POTENTIAL_NEW_CLASS"]
    auto_promote = [
        {"candidate_id": r.candidate_id, "macos_cpu_score": r.macos_cpu_score,
         "predicted_contest_cpu_gha": r.predicted_contest_cpu_gha,
         "tag": r.macos_cpu_calibrated_tag}
        for r in results if r.promotion_verdict == "AUTO_PROMOTE_GHA"
    ]
    operator_decision = [
        {"candidate_id": r.candidate_id, "macos_cpu_score": r.macos_cpu_score,
         "predicted_contest_cpu_gha": r.predicted_contest_cpu_gha,
         "tag": r.macos_cpu_calibrated_tag}
        for r in results if r.promotion_verdict == "OPERATOR_DECISION"
    ]
    best = sorted(
        (r for r in results if r.macos_cpu_score is not None),
        key=lambda r: r.macos_cpu_score,
    )[:5]
    return {
        "n_evaluated": n_ok,
        "n_failed": n_failed,
        "verdict_counts": by_verdict,
        "drift_flagged_candidates": drift_flagged,
        "auto_promote_gha_queue": auto_promote,
        "operator_decision_queue": operator_decision,
        "top5_lowest_macos_cpu_score": [
            {"candidate_id": r.candidate_id,
             "macos_cpu_score": r.macos_cpu_score,
             "predicted_contest_cpu_gha": r.predicted_contest_cpu_gha,
             "tag": r.macos_cpu_calibrated_tag}
            for r in best
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archives-dir", type=Path,
        help="Directory containing archive.zip files (or subdirs each holding "
             "archive.zip). Mutually exclusive with --candidates-jsonl.",
    )
    parser.add_argument(
        "--candidates-jsonl", type=Path,
        help="Newline-delimited JSON ledger; each row supplies "
             "candidate_id/archive_path/architecture_class (and optional "
             "inflate_sh/notes). Mutually exclusive with --archives-dir.",
    )
    parser.add_argument(
        "--architecture-class", default="hnerv",
        help="Architecture class for --archives-dir candidates (default: hnerv). "
             "Determines whether [macOS-CPU calibrated] vs [macOS-CPU advisory only] "
             "tag is applied. Per-candidate override via --candidates-jsonl.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output dir under experiments/results/. Will be created. "
             "/tmp paths are FORBIDDEN per CLAUDE.md.",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4,
        help="Max concurrent evals (default: 4). M5 Max has 12 perf cores; "
             "each eval pins num-threads via upstream defaults. Use 4 to leave "
             "headroom; raise to 6-8 for pure inner-loop sweeps.",
    )
    parser.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root (has evaluate.py + videos/).",
    )
    parser.add_argument(
        "--video-names-file", type=Path, default=DEFAULT_VIDEO_NAMES,
        help="Test video names list.",
    )
    parser.add_argument(
        "--inflate-timeout", type=int, default=900,
        help="Per-candidate inflate.sh timeout (seconds). Default 900.",
    )
    parser.add_argument(
        "--evaluate-timeout", type=int, default=2400,
        help="Per-candidate evaluate.py timeout (seconds). M5 Max CPU eval "
             "runs ~5-25 min depending on batch/threads. Default 2400.",
    )
    parser.add_argument(
        "--prior-anchor", type=float, default=None,
        help="Optional: previously-observed macOS-CPU score for the architecture-"
             "class. Enables drift_flag=POTENTIAL_NEW_CLASS detection.",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip the mandatory PR #107 calibration anchor verification. "
             "Use ONLY when you've verified the bound recently and are doing "
             "a follow-up batch. Default: calibration runs first.",
    )
    parser.add_argument(
        "--calibration-anchor-archive", type=Path,
        default=REPO / "experiments" / "results"
        / "track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal"
        / "harvested_artifacts" / "finetuned_archive" / "archive.zip",
        help="Archive used for calibration check (default: A1 latent-aligned "
             "0.19284757 GHA-anchored).",
    )
    parser.add_argument(
        "--calibration-expected-score", type=float, default=0.19284757743677347,
        help="Expected GHA Linux x86_64 score for the calibration anchor "
             "(default: A1 anchor).",
    )
    parser.add_argument(
        "--calibration-tolerance", type=float, default=PR107_EPSILON_BOUND,
        help=f"Calibration tolerance (default: {PR107_EPSILON_BOUND:.0e}, the PR #107 ε).",
    )
    parser.add_argument(
        "--proceed-on-calibration-fail", action="store_true",
        help="Proceed with sweep even if calibration verification fails "
             "(results will still be tagged honestly, but ε band may not hold).",
    )
    args = parser.parse_args()

    _verify_running_on_apple_silicon()

    if (args.archives_dir is None) == (args.candidates_jsonl is None):
        raise SystemExit(
            "exactly one of --archives-dir or --candidates-jsonl is required"
        )

    raw_output = args.output_dir
    output_dir = raw_output.resolve()
    if (str(raw_output).startswith("/tmp") or str(output_dir).startswith("/tmp")
            or str(output_dir).startswith("/private/tmp")):
        raise SystemExit("/tmp paths are FORBIDDEN per CLAUDE.md (transient-evidence-trap). "
                         "Use experiments/results/<lane>_<timestamp>/ instead.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration step (mandatory unless skipped)
    calibration_msg: str | None = None
    if not args.skip_calibration:
        anchor = args.calibration_anchor_archive
        if not anchor.exists():
            print(f"[calibrate] WARNING: calibration anchor archive missing: {anchor}")
            print("[calibrate] Skipping calibration (consider --skip-calibration if intentional)")
        else:
            cal_dir = output_dir / "_calibration"
            cal_dir.mkdir(parents=True, exist_ok=True)
            passed, observed, msg = verify_calibration_anchor(
                anchor, args.calibration_expected_score,
                args.calibration_tolerance, cal_dir,
                args.upstream_dir.resolve(), args.video_names_file.resolve(),
            )
            calibration_msg = msg
            (output_dir / "_calibration" / "calibration_result.json").write_text(
                json.dumps({
                    "passed": passed,
                    "observed_macos_cpu_score": observed,
                    "expected_gha_x86_64_score": args.calibration_expected_score,
                    "tolerance": args.calibration_tolerance,
                    "epsilon_bound_used": PR107_EPSILON_BOUND,
                    "anchor_archive": str(anchor),
                    "anchor_archive_sha256": _sha256(anchor),
                    "message": msg,
                    "completed_utc": _now_iso(),
                }, indent=2) + "\n"
            )
            if not passed and not args.proceed_on_calibration_fail:
                raise SystemExit(
                    "calibration verification FAILED; pass "
                    "--proceed-on-calibration-fail to sweep anyway "
                    "(results will be tagged honestly but ε band may not hold)"
                )

    # Load candidates
    if args.archives_dir:
        specs = _load_candidates_from_dir(args.archives_dir.resolve(),
                                           args.architecture_class)
    else:
        specs = _load_candidates_from_jsonl(args.candidates_jsonl.resolve())

    print(f"[sweep] {len(specs)} candidate(s) loaded")
    for s in specs:
        print(f"  - {s.display()}")

    # Run sweep
    results = run_sweep(
        specs, output_dir, args.max_concurrency,
        args.upstream_dir.resolve(), args.video_names_file.resolve(),
        args.prior_anchor, args.inflate_timeout, args.evaluate_timeout,
    )

    # Emit aggregate manifest
    manifest_path = _emit_sweep_manifest(output_dir, results, calibration_msg, args)
    print(f"[sweep] manifest: {manifest_path}")
    print(f"[sweep] results: {output_dir / 'results.jsonl'}")

    summary = _summarize(results)
    print(f"\n[sweep] SUMMARY:")
    print(f"  evaluated:     {summary['n_evaluated']}")
    print(f"  failed:        {summary['n_failed']}")
    print(f"  by verdict:    {summary['verdict_counts']}")
    print(f"  AUTO_PROMOTE_GHA queue ({len(summary['auto_promote_gha_queue'])}):")
    for r in summary["auto_promote_gha_queue"]:
        print(f"    - {r['candidate_id']}: macos={r['macos_cpu_score']:.10f} "
              f"predicted-GHA={r['predicted_contest_cpu_gha']}")
    print(f"  OPERATOR_DECISION queue ({len(summary['operator_decision_queue'])}):")
    for r in summary["operator_decision_queue"]:
        print(f"    - {r['candidate_id']}: macos={r['macos_cpu_score']:.10f} "
              f"predicted-GHA={r['predicted_contest_cpu_gha']}")
    if summary["drift_flagged_candidates"]:
        print(f"  DRIFT-FLAGGED (POTENTIAL_NEW_CLASS): "
              f"{summary['drift_flagged_candidates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
