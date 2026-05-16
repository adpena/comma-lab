#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""macOS-CPU SMOKE HARNESS for the Time-Traveler L5 Autonomy substrate.

PAIR T directive 2026-05-13: this is the $0-GPU companion runner for sister
lane ``lane_time_traveler_l5_autonomy_substrate_20260513``. It exercises the
substrate's full pipeline (trainer → archive build → inflate → auth eval
``--device cpu``) on the local macOS host, tags the result
``[macOS-CPU advisory]`` per Catalog #192, and compares the observed score
against the design memo's predicted_band ``[0.150, 0.170]`` as an advisory
escalation signal only. macOS-CPU output can never falsify or promote the
architecture.

CLAUDE.md compliance:
  * No GPU dispatch — local macOS-CPU only.
  * Output is PERMANENTLY ``score_claim=False``, ``promotion_eligible=False``,
    ``ready_for_exact_eval_dispatch=False`` (Catalog #192 + #127).
  * Per CLAUDE.md "MPS auth eval is NOISE": no MPS path; ``--device cpu``.
  * Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": all
    persisted artifacts land under ``experiments/results/<lane>_<UTC>/``;
    transient eval work-dirs are honored by ``contest_auth_eval.py``.
  * Per CLAUDE.md "macOS-CPU advisory only": evidence_grade is fixed and
    the autopilot ranker (``cathedral_autopilot_autonomous_loop``) is the
    sanctioned consumer; promotion and family-retirement require paired
    contest-axis evidence.

Sister-substrate interface contract (per design memo + sister
``src/tac/substrates/time_traveler_l5_autonomy/__init__.py``):

    from tac.substrates.time_traveler_l5_autonomy import (
        TimeTravelerArchive,
        TimeTravelerConfig,
        TimeTravelerSubstrate,
        pack_archive,
        parse_archive,
    )

If the substrate's full surface (archive + score_aware_loss + trainer) is
not yet ready when this harness runs, ``--stub-interface`` produces a
mock-archive smoke that exercises ONLY the manifest/autopilot wiring —
useful for landing the harness independently of the substrate.

Output (per run):
  * ``experiments/results/<lane>_<UTC>/`` containing
    * ``smoke_output.json`` — verdict + manifest + per-stage timing
    * ``macos_cpu_advisory_manifest.json`` — Catalog #192-compliant
    * ``stub_archive.zip`` (only if ``--stub-interface``)
  * stdout: structured JSON suitable for autopilot ingestion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.macos_cpu_advisory_signal import (  # noqa: E402
    EVIDENCE_GRADE,
    EVIDENCE_TAG,
    build_macos_cpu_advisory_signal_manifest,
    detect_macos_cpu_hardware_substrate,
    is_running_on_macos_arm64,
    json_text,
    load_calibration_model,
)

# Per design memo: predicted_band [0.150, 0.170]. We expose constants so
# tests can pin the contract without re-reading the memo.
PREDICTED_BAND_LOW: float = 0.150
PREDICTED_BAND_HIGH: float = 0.170

# Advisory escalation threshold per design memo. Above this, the architecture
# needs paired contest-axis recheck; macOS-CPU advisory output cannot falsify a
# method family.
ESCALATION_THRESHOLD: float = 0.190

LANE_ID: str = "lane_time_traveler_smoke_harness_20260513"
SISTER_LANE_ID: str = "lane_time_traveler_l5_autonomy_substrate_20260513"

# Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": persisted
# results land under experiments/results/<lane>_<UTC>/.
RESULTS_ROOT: Path = REPO_ROOT / "experiments" / "results"


class SmokeVerdict:
    """Verdict tokens returned by the harness."""

    PASS_IN_BAND = "pass_in_predicted_band"
    PASS_BELOW_BAND = "pass_below_band_better_than_predicted"
    WARN_ABOVE_BAND = "warn_above_predicted_band"
    ESCALATE_ABOVE_THRESHOLD = "escalate_above_threshold_requires_contest_axis_recheck"
    SUBSTRATE_NOT_READY = "substrate_not_ready_stub_only"
    EVAL_HARNESS_ERROR = "eval_harness_error"
    NON_DARWIN = "non_darwin_skip"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=(
            "Trainer epochs for CPU-smoke mode. Default 100 (per PAIR T "
            "directive). Sister substrate's trainer accepts --smoke and "
            "--epochs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Trainer batch size for macOS-CPU smoke (default 4; CPU-bounded).",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help=(
            "Path to a pre-built TT5L archive.zip to evaluate (skip training "
            "stage). When omitted, the harness invokes the sister substrate's "
            "trainer in CPU-smoke mode."
        ),
    )
    parser.add_argument(
        "--stub-interface",
        action="store_true",
        help=(
            "Run with a stub archive instead of invoking the sister substrate. "
            "Useful when the substrate is not yet wired end-to-end (race "
            "condition mitigation per PAIR T directive)."
        ),
    )
    parser.add_argument(
        "--predicted-band-low",
        type=float,
        default=PREDICTED_BAND_LOW,
        help=f"Lower bound of predicted band (default {PREDICTED_BAND_LOW}).",
    )
    parser.add_argument(
        "--predicted-band-high",
        type=float,
        default=PREDICTED_BAND_HIGH,
        help=f"Upper bound of predicted band (default {PREDICTED_BAND_HIGH}).",
    )
    parser.add_argument(
        "--escalation-threshold",
        type=float,
        default=ESCALATION_THRESHOLD,
        help=(
            "Advisory score above this requires paired contest-axis recheck "
            f"(default {ESCALATION_THRESHOLD}). macOS-CPU advisory output "
            "cannot falsify or promote the architecture."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir for persisted artifacts. Defaults to "
            f"experiments/results/{LANE_ID}_<UTC>/. NEVER /tmp."
        ),
    )
    parser.add_argument(
        "--allow-non-darwin",
        action="store_true",
        help="Skip the Darwin ARM64 platform check (for unit tests / CI).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a plan + verdict skeleton without invoking trainer or "
            "evaluate.py. Useful for autopilot dry-run wiring."
        ),
    )
    parser.add_argument(
        "--inflate-sh",
        type=Path,
        default=None,
        help=(
            "Optional inflate.sh override. When omitted, the harness requires "
            "the archive-local sister runtime emitted by the TT5L trainer "
            "(submission_dir/inflate.sh). It never falls back to exact_current."
        ),
    )
    return parser.parse_args(argv)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    out = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else RESULTS_ROOT / f"{LANE_ID}_{_utc_stamp()}"
    )
    out_str = str(out)
    # Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact":
    # refuse /tmp / /var/tmp / /private/tmp.
    if (
        out_str.startswith("/tmp/")
        or "/private/tmp/" in out_str
        or "/var/tmp/" in out_str
    ):
        raise ValueError(
            f"refusing to write smoke output to forbidden /tmp path: {out_str!r}"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def _try_build_stub_archive(output_dir: Path) -> Path:
    """Build a minimal stub TT5L archive for harness self-test.

    The stub is NOT a valid contest archive — it's a placeholder zip with one
    member ``0.bin`` containing the TT5L magic. The eval call is expected to
    fail; the harness then reports verdict=SUBSTRATE_NOT_READY.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the stub is tagged
    explicitly and never enters the autopilot ranking flow.
    """
    import zipfile

    archive = output_dir / "stub_archive.zip"
    # 4-byte magic followed by 1-byte version + tiny placeholder payload.
    placeholder_bytes = b"TT5L" + b"\x01" + b"\x00" * 96
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", placeholder_bytes)
    return archive


def _invoke_sister_trainer_cpu_smoke(
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Call the sister substrate's trainer in CPU-smoke mode.

    Returns a dict with ``archive_path`` (when training succeeded) plus the
    trainer's stdout tail. If the trainer does not exist yet, returns a
    structured ``substrate_not_ready=True`` record so the harness can fall
    back to the stub flow without hard-failing.
    """
    trainer_candidates = [
        REPO_ROOT / "experiments" / "train_substrate_time_traveler_l5_autonomy.py",
        REPO_ROOT / "experiments" / "train_substrate_time_traveler.py",
    ]
    trainer = next((p for p in trainer_candidates if p.is_file()), None)
    if trainer is None:
        return {
            "substrate_not_ready": True,
            "reason": "no_sister_trainer_script_found",
            "searched": [str(p) for p in trainer_candidates],
        }

    trainer_output = output_dir / "trainer_artifacts"
    trainer_output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(trainer),
        "--device",
        "cpu",
        "--smoke",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(trainer_output),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "substrate_not_ready": True,
            "reason": f"trainer_invocation_failed: {exc}",
        }
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        return {
            "substrate_not_ready": True,
            "reason": f"trainer_returncode={result.returncode}",
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-1500:],
            "elapsed_seconds": elapsed,
        }
    # Expect trainer to emit archive.zip under output_dir.
    archive_candidates = sorted(trainer_output.rglob("*.zip"))
    if not archive_candidates:
        return {
            "substrate_not_ready": True,
            "reason": "trainer_succeeded_but_no_archive_zip_emitted",
            "stdout_tail": result.stdout[-1500:],
            "elapsed_seconds": elapsed,
        }
    return {
        "substrate_not_ready": False,
        "archive_path": str(archive_candidates[0]),
        "elapsed_seconds": elapsed,
    }


def _resolve_inflate_sh_for_archive(
    archive: Path,
    *,
    output_dir: Path,
    requested: Path | None,
) -> dict[str, Any]:
    """Resolve the TT5L runtime for ``archive`` without baseline fallback.

    The smoke harness exists to test the Time-Traveler packet/runtime pair. A
    fallback to ``submissions/exact_current`` turns a bad TT5L runtime into a
    misleading baseline eval, so missing runtime is a harness error.
    """
    if requested is not None:
        candidate = requested.resolve()
        return {
            "ok": candidate.is_file(),
            "path": str(candidate),
            "source": "explicit_override",
            "reason": None if candidate.is_file() else "explicit_inflate_sh_missing",
        }

    candidates = [
        archive.parent / "submission_dir" / "inflate.sh",
        archive.parent / "inflate.sh",
        output_dir / "submission_dir" / "inflate.sh",
        output_dir / "trainer_artifacts" / "submission_dir" / "inflate.sh",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return {
                "ok": True,
                "path": str(candidate.resolve()),
                "source": "archive_local_tt5l_runtime",
                "searched": [str(p) for p in candidates],
            }
    return {
        "ok": False,
        "path": None,
        "source": "missing_archive_local_tt5l_runtime",
        "searched": [str(p) for p in candidates],
        "reason": "tt5l_inflate_sh_not_found_no_exact_current_fallback",
    }


def _run_contest_auth_eval_cpu(
    archive: Path,
    *,
    output_dir: Path,
    inflate_sh: Path | None,
) -> dict[str, Any]:
    """Run ``experiments/contest_auth_eval.py --device cpu`` on the archive.

    Returns parsed JSON with ``score`` + ``d_seg`` + ``d_pose`` when
    available, plus the rc/stderr tail on failure.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" the cpu result
    on a macOS host is tagged ``[macOS-CPU advisory]`` (NOT
    ``[contest-CPU]``) per the 1:1 hardware-compliance rule. The harness
    does NOT lift that gate.
    """
    auth_eval_py = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    if not auth_eval_py.is_file():
        return {
            "eval_ok": False,
            "reason": "contest_auth_eval.py_not_found",
            "expected_path": str(auth_eval_py),
        }

    runtime = _resolve_inflate_sh_for_archive(
        archive, output_dir=output_dir, requested=inflate_sh
    )
    if not runtime.get("ok"):
        return {
            "eval_ok": False,
            "reason": runtime.get("reason") or "tt5l_runtime_not_found",
            "inflate_runtime_resolution": runtime,
        }
    inflate_sh_path = Path(str(runtime["path"]))

    json_out = output_dir / "contest_auth_eval.json"
    cmd = [
        sys.executable,
        str(auth_eval_py),
        "--archive",
        str(archive),
        "--inflate-sh",
        str(inflate_sh_path),
        "--device",
        "cpu",
        "--json-out",
        str(json_out),
        "--allow-temp-work-dir",
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=7200,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "eval_ok": False,
            "reason": f"contest_auth_eval_invocation_failed: {exc}",
        }
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        return {
            "eval_ok": False,
            "reason": f"contest_auth_eval_returncode={result.returncode}",
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "elapsed_seconds": elapsed,
        }

    parsed: dict[str, Any] = {"eval_ok": True, "elapsed_seconds": elapsed}
    parsed["inflate_runtime_resolution"] = runtime
    if json_out.is_file():
        try:
            parsed.update(json.loads(json_out.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            parsed["json_parse_error"] = str(exc)
    return parsed


def _classify_verdict(
    score: float | None,
    *,
    band_low: float,
    band_high: float,
    escalation: float,
) -> str:
    if score is None:
        return SmokeVerdict.EVAL_HARNESS_ERROR
    if score >= escalation:
        return SmokeVerdict.ESCALATE_ABOVE_THRESHOLD
    if score > band_high:
        return SmokeVerdict.WARN_ABOVE_BAND
    if score < band_low:
        return SmokeVerdict.PASS_BELOW_BAND
    return SmokeVerdict.PASS_IN_BAND


def _build_manifest_for_smoke_row(
    *,
    archive: Path,
    archive_bytes: int,
    archive_sha256: str,
    eval_payload: dict[str, Any],
    elapsed_seconds: float,
    run_id: str,
) -> dict[str, Any]:
    """Build a Catalog #192-compliant macOS-CPU advisory manifest row."""
    n_samples = eval_payload.get("n_samples")
    observations = [
        {
            "family": "time_traveler_l5_autonomy",
            "variant_id": archive.stem,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "score": eval_payload.get("score"),
            "d_seg": eval_payload.get("d_seg"),
            "d_pose": eval_payload.get("d_pose"),
            "samples_evaluated": n_samples,
            "wall_clock_seconds": elapsed_seconds,
            "source_artifact": str(archive),
        }
    ]
    calibration_model = load_calibration_model()
    manifest = build_macos_cpu_advisory_signal_manifest(
        observations,
        source=f"smoke_time_traveler_l5_autonomy_macos_cpu.py:{archive}",
        run_id=run_id,
        hardware_substrate=detect_macos_cpu_hardware_substrate(),
        calibration_model=calibration_model,
    )
    return manifest


def _emit_summary(
    *,
    verdict: str,
    score: float | None,
    band_low: float,
    band_high: float,
    escalation: float,
    archive_bytes: int | None,
    manifest_path: Path | None,
    eval_payload: dict[str, Any],
    output_dir: Path,
    elapsed_total_seconds: float,
    stub_used: bool,
) -> dict[str, Any]:
    return {
        "lane_id": LANE_ID,
        "sister_lane_id": SISTER_LANE_ID,
        "verdict": verdict,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "score_macos_cpu": score,
        "predicted_band_low": band_low,
        "predicted_band_high": band_high,
        "escalation_threshold": escalation,
        "in_predicted_band": (
            None if score is None
            else (band_low <= score <= band_high)
        ),
        "above_escalation_threshold": (
            None if score is None
            else score >= escalation
        ),
        "samples_evaluated": eval_payload.get("n_samples"),
        "archive_bytes": archive_bytes,
        "stub_interface_used": stub_used,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "output_dir": str(output_dir),
        "eval_payload": eval_payload,
        "elapsed_total_seconds": elapsed_total_seconds,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "dispatch_blockers": [
            "macos_cpu_advisory_not_score_evidence",
            "macos_cpu_advisory_cannot_falsify_architecture",
            "not_a_11_contest_compliant_cpu_axis",
            "requires_paired_contest_cpu_gha_linux_x86_64_before_score_claim",
            "requires_paired_contest_cuda_before_dual_axis_submission",
            "not_promotion_eligible",
        ],
        "notes": (
            "Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA, ON 1:1 "
            "CONTEST-COMPLIANT HARDWARE': this row is a free macOS-CPU "
            "advisory proxy; promotion requires paired [contest-CPU GHA Linux "
            "x86_64] AND [contest-CUDA] anchors on the EXACT same archive bytes."
        ),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Orchestrate the full smoke run; return a structured summary dict."""
    started = time.perf_counter()
    if not args.allow_non_darwin and not is_running_on_macos_arm64():
        return _emit_summary(
            verdict=SmokeVerdict.NON_DARWIN,
            score=None,
            band_low=args.predicted_band_low,
            band_high=args.predicted_band_high,
            escalation=args.escalation_threshold,
            archive_bytes=None,
            manifest_path=None,
            eval_payload={"reason": "harness_must_run_on_darwin_arm64"},
            output_dir=Path.cwd(),
            elapsed_total_seconds=time.perf_counter() - started,
            stub_used=False,
        )

    output_dir = _resolve_output_dir(args)

    # Stage 1: obtain archive (either pre-built, sister-trained, or stub).
    if args.dry_run:
        return _emit_summary(
            verdict=SmokeVerdict.PASS_IN_BAND,
            score=None,
            band_low=args.predicted_band_low,
            band_high=args.predicted_band_high,
            escalation=args.escalation_threshold,
            archive_bytes=None,
            manifest_path=None,
            eval_payload={"dry_run": True},
            output_dir=output_dir,
            elapsed_total_seconds=time.perf_counter() - started,
            stub_used=False,
        )

    archive: Path | None
    stub_used = False
    trainer_record: dict[str, Any] = {}

    if args.archive_path is not None:
        archive = args.archive_path
        if not archive.is_file():
            return _emit_summary(
                verdict=SmokeVerdict.EVAL_HARNESS_ERROR,
                score=None,
                band_low=args.predicted_band_low,
                band_high=args.predicted_band_high,
                escalation=args.escalation_threshold,
                archive_bytes=None,
                manifest_path=None,
                eval_payload={
                    "reason": "archive_path_does_not_exist",
                    "archive_path": str(archive),
                },
                output_dir=output_dir,
                elapsed_total_seconds=time.perf_counter() - started,
                stub_used=False,
            )
    elif args.stub_interface:
        archive = _try_build_stub_archive(output_dir)
        stub_used = True
    else:
        trainer_record = _invoke_sister_trainer_cpu_smoke(
            output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        if trainer_record.get("substrate_not_ready"):
            # Race-condition mitigation: fall back to stub flow with explicit
            # SUBSTRATE_NOT_READY verdict per PAIR T directive.
            archive = _try_build_stub_archive(output_dir)
            stub_used = True
        else:
            archive = Path(trainer_record["archive_path"])

    # Stage 2: contest_auth_eval --device cpu (or skip on stub).
    archive_bytes = archive.stat().st_size if archive is not None else 0
    archive_sha256 = _sha256_of(archive) if archive is not None else ""

    if stub_used:
        eval_payload = {
            "eval_ok": False,
            "reason": "stub_archive_not_contest_compliant_no_eval_attempted",
            "trainer_record": trainer_record,
        }
    else:
        assert archive is not None
        eval_payload = _run_contest_auth_eval_cpu(
            archive,
            output_dir=output_dir,
            inflate_sh=args.inflate_sh,
        )

    score = None
    if eval_payload.get("eval_ok"):
        # contest_auth_eval emits structured JSON with score components.
        # See experiments/contest_auth_eval.py:1502 _emit_summary().
        # Accept either flat or nested layouts.
        score = eval_payload.get("score")
        if score is None and "evaluate_result" in eval_payload:
            sub = eval_payload["evaluate_result"]
            if isinstance(sub, dict):
                score = sub.get("score")
                eval_payload.setdefault("d_seg", sub.get("d_seg"))
                eval_payload.setdefault("d_pose", sub.get("d_pose"))

    # Stage 3: build advisory manifest (only when we have real eval results).
    manifest_path: Path | None = None
    if score is not None and archive is not None:
        run_id = f"smoke_{LANE_ID}_{_utc_stamp()}_{archive_sha256[:12]}"
        manifest = _build_manifest_for_smoke_row(
            archive=archive,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            eval_payload=eval_payload,
            elapsed_seconds=float(eval_payload.get("elapsed_seconds") or 0.0),
            run_id=run_id,
        )
        manifest_path = output_dir / "macos_cpu_advisory_manifest.json"
        manifest_path.write_text(json_text(manifest), encoding="utf-8")

    # Stage 4: verdict.
    if stub_used:
        verdict = SmokeVerdict.SUBSTRATE_NOT_READY
    elif not eval_payload.get("eval_ok"):
        verdict = SmokeVerdict.EVAL_HARNESS_ERROR
    else:
        verdict = _classify_verdict(
            score,
            band_low=args.predicted_band_low,
            band_high=args.predicted_band_high,
            escalation=args.escalation_threshold,
        )

    return _emit_summary(
        verdict=verdict,
        score=score,
        band_low=args.predicted_band_low,
        band_high=args.predicted_band_high,
        escalation=args.escalation_threshold,
        archive_bytes=archive_bytes if archive is not None else None,
        manifest_path=manifest_path,
        eval_payload=eval_payload,
        output_dir=output_dir,
        elapsed_total_seconds=time.perf_counter() - started,
        stub_used=stub_used,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_smoke(args)
    output_dir = Path(summary["output_dir"])
    if output_dir.is_dir():
        smoke_json = output_dir / "smoke_output.json"
        try:
            smoke_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    verdict = summary.get("verdict")
    # Exit codes:
    #   0 = PASS_IN_BAND / PASS_BELOW_BAND / dry-run / SUBSTRATE_NOT_READY
    #   1 = WARN_ABOVE_BAND / ESCALATE_ABOVE_THRESHOLD
    #   3 = EVAL_HARNESS_ERROR
    #   4 = NON_DARWIN
    if verdict in (SmokeVerdict.PASS_IN_BAND, SmokeVerdict.PASS_BELOW_BAND,
                   SmokeVerdict.SUBSTRATE_NOT_READY):
        return 0
    if verdict in (
        SmokeVerdict.WARN_ABOVE_BAND,
        SmokeVerdict.ESCALATE_ABOVE_THRESHOLD,
    ):
        return 1
    if verdict == SmokeVerdict.EVAL_HARNESS_ERROR:
        return 3
    if verdict == SmokeVerdict.NON_DARWIN:
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    raise SystemExit(main())
