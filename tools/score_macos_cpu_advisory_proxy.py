#!/usr/bin/env python3
"""macOS-CPU smoke ranker CLI: free first-class advisory proxy eval.

Operator routing 2026-05-13 ("training is the real roadblock; we can prepare
and run things on macos and cpu"). Cascade reframe: dev loop on macOS, deploy
to contest hardware. This tool runs ``inflate.sh`` + ``upstream/evaluate.py
--device cpu`` on a single archive (or glob of archives) on the local Darwin
ARM64 host and emits a structured advisory-signal manifest row.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + PR107 empirical calibration
(|Δ| ≤ 6e-6 vs GHA Linux x86_64), the output is:

  - evidence_grade = "macOS-CPU-advisory"
  - evidence_tag = "[macOS-CPU advisory only]"
  - score_claim = False
  - promotion_eligible = False
  - ready_for_exact_eval_dispatch = False
  - ranking_only = True

It participates in autopilot dispatch ranking (cheap pre-GPU ordering) but
NEVER promotes. Sister tool: ``tools/build_macos_cpu_advisory_signal_manifest.py``
builds full manifests from pre-existing observation files.

Per CLAUDE.md Catalog #127 the macOS-CPU evidence tag is already routed to
``refused_class="macos_substrate"`` by the custody validator. This tool
respects that contract by never producing rows the validator would accept
as authoritative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
    MacOSCPUAdvisorySignalError,
    append_manifest_row_to_jsonl,
    build_macos_cpu_advisory_signal_manifest,
    detect_macos_cpu_hardware_substrate,
    is_running_on_macos_arm64,
    json_text,
    load_calibration_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--archive",
        type=Path,
        help="Single archive.zip to evaluate.",
    )
    group.add_argument(
        "--archives",
        type=str,
        help=(
            "Glob pattern matching multiple archive.zip files. Relative to "
            "the current working directory."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help=(
            "Number of frame pairs to evaluate (default: 100 smoke). Use "
            "--samples 600 for the contest's full sample count. The contest "
            "scorer expects non-overlapping seq_len=2 batching."
        ),
    )
    parser.add_argument(
        "--inflate-sh",
        type=Path,
        default=REPO_ROOT / "submissions" / "exact_current" / "inflate.sh",
        help="Path to inflate.sh (defaults to submissions/exact_current/inflate.sh).",
    )
    parser.add_argument(
        "--evaluate-py",
        type=Path,
        default=REPO_ROOT / "upstream" / "evaluate.py",
        help="Path to upstream/evaluate.py (defaults to upstream/evaluate.py).",
    )
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        help=(
            "Architecture family tag for the manifest row (e.g. "
            "'pr106_hnerv_cluster', 'pr101_lossy_coarsening', 'lane_12_v2')."
        ),
    )
    parser.add_argument(
        "--variant-id",
        type=str,
        default=None,
        help=(
            "Variant id for the manifest row. Defaults to the archive "
            "filename stem when omitted."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Stable run id (defaults to a UTC timestamp + archive sha "
            "fingerprint when omitted)."
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Optional path to write the full structured manifest JSON. When "
            "set the macOS-CPU advisory signal manifest is emitted there."
        ),
    )
    parser.add_argument(
        "--jsonl-append",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSONL aggregator. When set each evaluated "
            "archive appends one canonical row."
        ),
    )
    parser.add_argument(
        "--allow-non-darwin",
        action="store_true",
        help=(
            "Skip the Darwin ARM64 platform check (for unit tests / CI). "
            "Production usage MUST run on a real macOS host."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Don't actually invoke inflate.sh / evaluate.py — emit a "
            "placeholder manifest with score=None for smoke / CI testing."
        ),
    )
    return parser.parse_args(argv)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_inflate(
    *,
    archive: Path,
    inflate_sh: Path,
    work_dir: Path,
) -> Path:
    """Run inflate.sh and return the output directory.

    Per CLAUDE.md "Operator gates must be wired and used" and the contest
    runtime contract: ``inflate.sh archive_dir output_dir file_list``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = work_dir / "archive_in"
    archive_dir.mkdir(parents=True, exist_ok=True)
    # Unzip the archive into archive_dir.
    import zipfile

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(archive_dir)
    output_dir = work_dir / "inflated"
    output_dir.mkdir(parents=True, exist_ok=True)
    # File list — default to the contest's public_test_video_names.txt-style
    # single-line "0" / "1" / ... convention.
    file_list = work_dir / "files.txt"
    file_list.write_text("0\n", encoding="utf-8")
    cmd = [
        str(inflate_sh),
        str(archive_dir),
        str(output_dir),
        str(file_list),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"inflate.sh failed rc={result.returncode}:\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    return output_dir


def _run_evaluate_cpu(
    *,
    evaluate_py: Path,
    inflated_dir: Path,
    samples: int,
) -> dict[str, Any]:
    """Run upstream/evaluate.py --device cpu on the inflated frames.

    Returns the parsed score + per-component breakdown. Per CLAUDE.md
    "Apples-to-apples evidence discipline" the score is tagged as macOS-CPU
    advisory only; promotion requires a paired GHA Linux x86_64 result.
    """
    cmd = [
        sys.executable,
        str(evaluate_py),
        "--inflated-dir",
        str(inflated_dir),
        "--device",
        "cpu",
        "--samples",
        str(samples),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"evaluate.py failed rc={result.returncode}:\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    # The contest's evaluate.py emits a single JSON line; parse it.
    # If the actual upstream script's output differs, callers can post-process
    # via the --manifest-output JSON.
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "{}"
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {"raw_stdout_tail": last_line, "score": None, "d_seg": None, "d_pose": None}


def _evaluate_one_archive(
    *,
    archive: Path,
    inflate_sh: Path,
    evaluate_py: Path,
    samples: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Evaluate one archive on macOS-CPU; return an observation row."""
    archive_bytes = archive.stat().st_size
    archive_sha256 = _sha256_of(archive)
    started = time.perf_counter()
    if dry_run:
        # Dry-run: emit a sentinel score=0.0 so the manifest builder can
        # construct a row. The row is still tagged non-promotable advisory;
        # callers parse the manifest and observe ``score_macos_cpu`` is the
        # sentinel value (paired with the dry_run=True eval payload).
        score: float | None = 0.0
        d_seg: float | None = 0.0
        d_pose: float | None = 0.0
        eval_payload: dict[str, Any] = {"dry_run": True}
    else:
        import tempfile

        # CLAUDE.md "Forbidden /tmp paths in any persisted artifact": the
        # inflate working dir is transient scratch (NEVER cited as evidence).
        # We honor the rule by ensuring no path written to the manifest
        # references /tmp. The working dir is cleaned at end-of-eval.
        with tempfile.TemporaryDirectory(prefix="macos_cpu_proxy_eval_") as work_str:
            work = Path(work_str)
            inflated = _run_inflate(
                archive=archive,
                inflate_sh=inflate_sh,
                work_dir=work,
            )
            eval_payload = _run_evaluate_cpu(
                evaluate_py=evaluate_py,
                inflated_dir=inflated,
                samples=samples,
            )
        score = eval_payload.get("score")
        d_seg = eval_payload.get("d_seg")
        d_pose = eval_payload.get("d_pose")
    elapsed = time.perf_counter() - started
    return {
        "family": "",  # filled in by caller
        "variant_id": "",  # filled in by caller
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "score": score,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "samples_evaluated": samples,
        "wall_clock_seconds": elapsed,
        "source_artifact": str(archive),
        "_raw_evaluate_payload": eval_payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.allow_non_darwin and not is_running_on_macos_arm64():
        print(
            "[macos-cpu-advisory-proxy] FATAL: must run on Darwin ARM64. Use "
            "--allow-non-darwin for unit tests or CI replay only.",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        if not args.inflate_sh.is_file():
            print(
                f"[macos-cpu-advisory-proxy] FATAL: inflate.sh missing: {args.inflate_sh}",
                file=sys.stderr,
            )
            return 3
        if not args.evaluate_py.is_file():
            print(
                f"[macos-cpu-advisory-proxy] FATAL: evaluate.py missing: {args.evaluate_py}",
                file=sys.stderr,
            )
            return 3

    if args.archive is not None:
        archives = [args.archive]
    else:
        # Glob mode. Use Python's glob module to support both relative and
        # absolute patterns.
        import glob as _glob

        archives = sorted(Path(p) for p in _glob.glob(args.archives))
        if not archives:
            print(
                f"[macos-cpu-advisory-proxy] no archives matched glob: {args.archives!r}",
                file=sys.stderr,
            )
            return 4

    observations: list[dict[str, Any]] = []
    for archive in archives:
        print(f"[macos-cpu-advisory-proxy] evaluating {archive} ({archive.stat().st_size} bytes)")
        row = _evaluate_one_archive(
            archive=archive,
            inflate_sh=args.inflate_sh,
            evaluate_py=args.evaluate_py,
            samples=args.samples,
            dry_run=args.dry_run,
        )
        row["family"] = args.family
        row["variant_id"] = args.variant_id or archive.stem
        observations.append(row)
        print(
            f"  score={row.get('score')}  d_seg={row.get('d_seg')} "
            f"d_pose={row.get('d_pose')}  wall_clock={row.get('wall_clock_seconds'):.2f}s"
        )

    if args.run_id:
        run_id = args.run_id
    else:
        # Auto-generate from UTC timestamp + first archive sha prefix.
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        sha_prefix = observations[0]["archive_sha256"][:12] if observations else "noarch"
        run_id = f"macos_cpu_advisory_{ts}_{sha_prefix}"

    calibration_model = load_calibration_model()
    manifest = build_macos_cpu_advisory_signal_manifest(
        observations,
        source=f"score_macos_cpu_advisory_proxy.py:{' '.join(str(a) for a in archives)}",
        run_id=run_id,
        hardware_substrate=detect_macos_cpu_hardware_substrate(),
        calibration_model=calibration_model,
    )

    payload = {
        "run_id": run_id,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "rows": manifest["rows"],
        "ranking_atoms": manifest["ranking_atoms"],
        "calibration_model_summary": {
            "drift_p90_abs": manifest["calibration_model"].get("drift_p90_abs"),
            "high_variance_multiplier": manifest["calibration_model"].get(
                "high_variance_multiplier"
            ),
            "calibration_status": manifest["calibration_model"].get("calibration_status"),
        },
    }
    # Emit to stdout as structured JSON.
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    if args.manifest_output is not None:
        output_str = str(args.manifest_output)
        if (
            output_str.startswith("/tmp/")
            or "/private/tmp/" in output_str
            or "/var/tmp/" in output_str
        ):
            print(
                f"[macos-cpu-advisory-proxy] FATAL: refusing /tmp output path: "
                f"{output_str!r}",
                file=sys.stderr,
            )
            return 5
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json_text(manifest), encoding="utf-8")
        print(
            f"[macos-cpu-advisory-proxy] wrote full manifest to {args.manifest_output}"
        )

    if args.jsonl_append is not None:
        for row in manifest["rows"]:
            try:
                append_manifest_row_to_jsonl(row, output_path=args.jsonl_append)
            except (MacOSCPUAdvisorySignalError, ValueError) as exc:
                print(
                    f"[macos-cpu-advisory-proxy] FATAL: refused JSONL append: {exc}",
                    file=sys.stderr,
                )
                return 6
        print(
            f"[macos-cpu-advisory-proxy] appended {len(manifest['rows'])} row(s) "
            f"to {args.jsonl_append}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
