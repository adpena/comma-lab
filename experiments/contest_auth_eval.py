#!/usr/bin/env python3
"""Generic contest-compliant auth evaluation for ANY submission archive.

This is the CANONICAL tool for verifying any submission against the contest
scorer. Unlike auth_eval_renderer.py (which loads a renderer checkpoint and
renders frames in-process — a development shortcut), this tool runs the
EXACT contest pipeline:

    archive.zip → submission's inflate.sh → upstream/evaluate.py → score

Works for ANY contest-compliant submission, not just renderer-shaped ones.
The inflate.sh path defaults to submissions/robust_current/inflate.sh but
can be overridden for non-renderer lanes.

This tool is what the contest scorer effectively does internally. If a
score from this tool differs from auth_eval_renderer.py, the difference
reveals an inflate-path bug or an in-process-vs-on-disk numerical drift.

Council R3 (2026-04-26) flagged that auth_eval_renderer.py is renderer-
specific; the user's binding rule is that auth eval should work with any
contest-compliant submission. This tool is the answer.

Usage:
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive submissions/baseline_dilated_h64_0_90/archive_baseline_0_9001.zip \\
        --upstream-dir upstream \\
        --device cuda

    # Override inflate.sh for a non-renderer submission:
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive my_submission.zip \\
        --inflate-sh submissions/exact_current/inflate.sh \\
        --upstream-dir upstream

    # Specify GT video names file (default: upstream/public_test_video_names.txt):
    .venv/bin/python experiments/contest_auth_eval.py \\
        --archive baseline.zip \\
        --upstream-dir upstream \\
        --video-names-file upstream/public_test_video_names.txt
"""
from __future__ import annotations

# Line-buffer stdout so progress flushes to log files immediately.
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Schema version for the JSON we emit. Bump when adding fields so downstream
# tooling (BATTLE_PLAN parsers, leaderboard, etc.) can detect compatibility.
SCHEMA_VERSION = 1


def _sha256(path: Path, *, prefix: int = 16) -> str:
    """Hash a file's contents (full SHA256, return prefix chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:prefix] if prefix else digest


def _ensure_uv_available() -> None:
    """The robust_current inflate.sh shells out to `uv run python ...`.
    Verify uv is on PATH so we fail loud here, not 200 lines deep."""
    if shutil.which("uv") is None:
        raise RuntimeError(
            "FATAL: `uv` is not on PATH. submissions/robust_current/inflate.sh "
            "uses `uv run python ...`. Install with `curl -LsSf "
            "https://astral.sh/uv/install.sh | sh` then re-run."
        )


def _record_provenance(work_dir: Path, archive: Path, inflate_sh: Path,
                       upstream_dir: Path, args: argparse.Namespace) -> dict:
    """Snapshot the env so a re-run on different hardware is detectable.
    Records gpu_model, driver, torch+cuda versions, ffmpeg+svtav1 versions,
    git commits, and SHA of every input file. Mandatory per CLAUDE.md
    'deterministic reproducibility' non-negotiable."""
    def _shell(cmd, *, timeout: int = 10) -> str | None:
        try:
            return subprocess.check_output(
                cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            return f"<error:{exc!r}>"

    prov: dict = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "experiments/contest_auth_eval.py",
        "archive_path": str(archive),
        "archive_sha256": _sha256(archive, prefix=0),
        "archive_size_bytes": archive.stat().st_size,
        "inflate_script": str(inflate_sh),
        "inflate_script_sha256": _sha256(inflate_sh, prefix=0) if inflate_sh.exists() else None,
        "upstream_dir": str(upstream_dir),
        "device": args.device,
        "video_names_file": str(args.video_names_file),
        "sys_argv": sys.argv,
        "env_vars": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONHASHSEED", "PYTORCH_CUDA_ALLOC_CONF", "LD_LIBRARY_PATH",
        )},
    }
    # GPU + driver — Council R3 #4 also requires WARN if non-T4 since
    # contest scorer runs on T4. T4-vs-A100/4090 fp16 numerics diverge
    # on FastViT softmax (≤0.5% on PoseNet historically).
    prov["gpu_model"] = _shell(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    prov["gpu_driver"] = _shell(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    gm = (prov["gpu_model"] or "").strip()
    if gm and "T4" not in gm:
        print(
            f"\n[contest_auth_eval] *** WARNING: GPU is {gm!r}, NOT 'Tesla T4'. ***"
            f"\n[contest_auth_eval] Contest scorer runs on T4. Cross-arch fp16"
            f"\n[contest_auth_eval] numerics differ on FastViT softmax (~0.5% PoseNet"
            f"\n[contest_auth_eval] historical drift). Score is ADVISORY until"
            f"\n[contest_auth_eval] re-confirmed on T4. Recorded gpu_model={gm!r}.\n",
            file=sys.stderr,
        )
        prov["gpu_t4_match"] = False
    else:
        prov["gpu_t4_match"] = bool(gm)
    # torch + cuda
    try:
        import torch
        prov["torch_version"] = torch.__version__
        prov["cuda_version"] = torch.version.cuda
        prov["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            prov["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        prov["torch_import_error"] = True
    # ffmpeg + svtav1
    ffv = _shell(["ffmpeg", "-version"])
    prov["ffmpeg_version"] = (ffv.splitlines()[0] if ffv and not ffv.startswith("<error") else ffv)
    encs = _shell(["ffmpeg", "-encoders"])
    if encs and not encs.startswith("<error"):
        svt = [ln.strip() for ln in encs.splitlines()
               if "svtav1" in ln.lower() or "svt-av1" in ln.lower()]
        prov["libsvtav1_version"] = svt[0] if svt else None
    # git commits — pact + upstream
    prov["pact_commit"] = _shell(["git", "rev-parse", "HEAD"])
    if (upstream_dir / ".git").exists() or (upstream_dir.parent / ".git").exists():
        prov["upstream_commit"] = _shell(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"]
        )

    out = work_dir / "provenance.json"
    with open(out, "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def _extract_archive(archive: Path, dest: Path) -> list[str]:
    """Extract archive.zip into dest/. Returns list of member names.
    Refuses to write outside dest (zip-slip protection)."""
    dest.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    with zipfile.ZipFile(archive, "r") as z:
        for info in z.infolist():
            # zip-slip protection
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Refusing zip-slip path: {info.filename}")
            z.extract(info, dest)
            members.append(info.filename)
    return members


def _run_inflate(inflate_sh: Path, archive_dir: Path, inflated_dir: Path,
                 video_names_file: Path, *, timeout: int = 1800) -> None:
    """Invoke the submission's inflate.sh. Contest budget: 30 min on T4.
    Default timeout here is 30 min (1800s); pass --inflate-timeout for
    longer development runs.

    Council R3 #3 (CRITICAL): validate per-file byte counts so a partial
    inflate (silent drop of 1 of N videos) is caught here, not 200 lines
    later when upstream's `zip(dl_gt, dl_comp)` truncates to min().
    """
    inflated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(inflate_sh),
        str(archive_dir), str(inflated_dir), str(video_names_file),
    ]
    print(f"[inflate] cmd: {' '.join(cmd)}")
    print(f"[inflate] timeout: {timeout}s ({timeout / 60:.1f} min)")
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[inflate] TIMED OUT after {timeout}s. Contest budget is "
            f"30 min on T4. If this is a development run, pass "
            f"--inflate-timeout 7200 (or higher) to bypass."
        )
    elapsed = time.monotonic() - t0
    print(f"[inflate] returncode={result.returncode} elapsed={elapsed:.1f}s")
    if result.returncode != 0:
        raise RuntimeError(f"[inflate] FAILED with returncode={result.returncode}")

    # Council R3 #3 fix: STRICT per-video byte-count validation.
    # Each .raw is uint8 RGB at upstream/frame_utils.py's camera_size
    # (1164w × 874h) × NUM_FRAMES (1200) × 3 channels = 3,663,237,120 B.
    test_videos = [n.strip() for n in video_names_file.read_text().splitlines()
                   if n.strip()]
    OUT_W, OUT_H, NUM_FRAMES = 1164, 874, 1200
    EXPECTED_RAW_BYTES = OUT_W * OUT_H * NUM_FRAMES * 3  # 3,663,237,120
    missing: list[str] = []
    wrong_size: list[tuple[str, int, int]] = []
    for vname in test_videos:
        # video name is e.g. "0.mkv" — the .raw is named after the stem
        stem = Path(vname).stem
        raw_path = inflated_dir / f"{stem}.raw"
        if not raw_path.exists():
            missing.append(stem)
            continue
        actual = raw_path.stat().st_size
        if actual != EXPECTED_RAW_BYTES:
            wrong_size.append((stem, actual, EXPECTED_RAW_BYTES))
    if missing:
        raise RuntimeError(
            f"[inflate] PARTIAL inflate — missing .raw for {len(missing)}/"
            f"{len(test_videos)} videos: {missing[:5]}{'…' if len(missing)>5 else ''}. "
            f"Upstream zip(dl_gt,dl_comp) would silently truncate to min(); "
            f"refusing to score."
        )
    if wrong_size:
        details = ", ".join(f"{n}={a}B (expected {e}B)" for n, a, e in wrong_size[:3])
        raise RuntimeError(
            f"[inflate] WRONG-SIZE .raw file(s): {details}. Each must be "
            f"{EXPECTED_RAW_BYTES:,} bytes (1164×874×1200×3). Likely "
            f"truncated mid-decode."
        )
    print(f"[inflate] produced {len(test_videos)} .raw file(s), each "
          f"{EXPECTED_RAW_BYTES:,} bytes — STRICT validation passed.")


def _validate_uncompressed_dir(uncompressed_dir: Path,
                               video_names_file: Path) -> None:
    """Council R3 #2 (CRITICAL): upstream/evaluate.py computes the rate
    denominator as `sum(file.size for file in uncompressed_dir.rglob('*'))`
    — every file under the dir tree. ANY extra file (e.g. .DS_Store, kaggle
    ingest leftovers, stray .raw caches) silently inflates the denominator
    and shifts the score vs the official scorer.

    Strategy: walk the dir, compare to the videos listed in --video-names-file.
    Fail loud on any extras OR missing files."""
    expected = {Path(n.strip()).name for n in video_names_file.read_text().splitlines()
                if n.strip()}
    found = set()
    extras: list[Path] = []
    for p in uncompressed_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(uncompressed_dir)
        # Hidden files / common leakage suspects
        if any(part.startswith(".") for part in rel.parts):
            extras.append(rel)
            continue
        if rel.name not in expected:
            extras.append(rel)
        else:
            found.add(rel.name)
    missing = expected - found
    if extras or missing:
        msg_parts = []
        if extras:
            msg_parts.append(
                f"EXTRA files in --uncompressed-dir ({len(extras)}): "
                f"{[str(p) for p in extras[:5]]}{'…' if len(extras)>5 else ''}"
            )
        if missing:
            msg_parts.append(f"MISSING: {sorted(missing)}")
        raise RuntimeError(
            f"[evaluate] uncompressed-dir contamination — score would drift "
            f"vs official scorer (rate denominator off): {'; '.join(msg_parts)}. "
            f"Move extras out of {uncompressed_dir} OR pass a clean --uncompressed-dir."
        )


def _run_upstream_evaluate(upstream_dir: Path, submission_dir: Path,
                           uncompressed_dir: Path, video_names_file: Path,
                           device: str, *, timeout: int = 1800,
                           batch_size: int = 16, num_threads: int = 2,
                           prefetch_queue_depth: int = 4,
                           seed: int = 1234) -> dict:
    """Invoke upstream/evaluate.py — the contest scorer. Returns the
    parsed score dict from the report.txt the script writes.

    Council R3 #1 (CRITICAL): all 4 score-affecting args (batch-size,
    num-threads, prefetch-queue-depth, seed) are pinned EXPLICITLY here
    so a future upstream default-bump doesn't silently shift our scores.
    Council R3 #2 (CRITICAL): pre-validate uncompressed-dir for contamination.
    Council R3 #4 (Medium): set determinism env vars."""
    _validate_uncompressed_dir(uncompressed_dir, video_names_file)

    report_path = submission_dir / "report.txt"
    cmd = [
        sys.executable, str(upstream_dir / "evaluate.py"),
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(uncompressed_dir),
        "--video-names-file", str(video_names_file),
        "--device", device,
        "--report", str(report_path),
        # PIN every contest-default arg explicitly (Council R3 #1):
        "--seed", str(seed),
        "--batch-size", str(batch_size),
        "--num-threads", str(num_threads),
        "--prefetch-queue-depth", str(prefetch_queue_depth),
    ]
    print(f"[evaluate] cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    env = {**os.environ}
    # upstream/evaluate.py imports modules from upstream/ at top level
    pp = env.get("PYTHONPATH", "")
    if str(upstream_dir) not in pp:
        env["PYTHONPATH"] = f"{upstream_dir}:{pp}" if pp else str(upstream_dir)
    # Determinism env (Council R3 #4) — required per CLAUDE.md
    # "deterministic reproducibility" non-negotiable. CUBLAS_WORKSPACE_CONFIG
    # is required for torch.use_deterministic_algorithms; PYTHONHASHSEED
    # affects dict iteration order in Python ≥ 3.6.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONHASHSEED"] = str(seed)

    result = subprocess.run(cmd, timeout=timeout, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    print(f"[evaluate] returncode={result.returncode} elapsed={elapsed:.1f}s")
    print(f"[evaluate] stdout (last 4KB):\n{result.stdout[-4096:]}")
    if result.returncode != 0:
        print(f"[evaluate] stderr:\n{result.stderr[-2048:]}", file=sys.stderr)
        raise RuntimeError(f"[evaluate] FAILED with returncode={result.returncode}")

    # Parse report.txt — upstream/evaluate.py writes a multi-line report
    if not report_path.exists():
        raise RuntimeError(f"[evaluate] no report.txt at {report_path}")
    return _parse_report(report_path, archive_size=(submission_dir / "archive.zip").stat().st_size)


def _parse_report(report_path: Path, *, archive_size: int) -> dict:
    """Parse upstream/evaluate.py's report.txt into a structured dict.

    The contest report format (from upstream/evaluate.py) looks like:
        === Evaluation results over 600 samples ===
          Average PoseNet Distortion: 0.0107...
          Average SegNet Distortion: 0.00240...
          Submission file size: 337748 bytes
          Original uncompressed size: 37545489 bytes
          Compression Rate: 0.00899...
          Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.90
    """
    text = report_path.read_text()

    def _grab(pattern: str, default: float | None = None) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    pose = _grab(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)")
    seg = _grab(r"Average SegNet Distortion:\s*([0-9.eE+-]+)")
    rate_unscaled = _grab(r"Compression Rate:\s*([0-9.eE+-]+)")
    final = _grab(r"Final score[^=]*=\s*([0-9.eE+-]+)")
    n_samples = re.search(r"results over (\d+) samples", text)

    if pose is None or seg is None or rate_unscaled is None or final is None:
        raise RuntimeError(
            f"[evaluate] could not parse report.txt:\n{text[:1024]}"
        )

    # Council R3 #5 (Medium): reject NaN/inf — float() parses both silently.
    # A divide-by-zero in upstream's distortion sum would slip through as
    # final_score=NaN that "looks like" a number. Refuse loud.
    import math as _math
    for label, val in (("posenet_dist", pose), ("segnet_dist", seg),
                       ("rate_unscaled", rate_unscaled), ("final_score", final)):
        if not _math.isfinite(val):
            raise RuntimeError(
                f"[evaluate] non-finite {label}={val} in report.txt — refuse "
                f"to ship a NaN/inf score. Investigate upstream evaluate run."
            )
    if pose < 0 or seg < 0 or rate_unscaled < 0 or final < 0:
        raise RuntimeError(
            f"[evaluate] negative metric in report (pose={pose}, seg={seg}, "
            f"rate={rate_unscaled}, final={final}) — distortions must be ≥0."
        )
    expected_n = 600  # contest pair count (1200 frames / seq_len=2)
    actual_n = int(n_samples.group(1)) if n_samples else None
    if actual_n != expected_n:
        raise RuntimeError(
            f"[evaluate] expected {expected_n} samples but report says "
            f"{actual_n}. Likely partial inflate (Council R3 #3) slipped "
            f"past the .raw byte-count check."
        )

    score_pose = (10.0 * pose) ** 0.5
    score_seg = 100.0 * seg
    score_rate = 25.0 * rate_unscaled
    score_recomputed = score_seg + score_pose + score_rate

    # Council R3 #6 (Medium): assert recomputed score matches reported
    # within upstream's print precision (.2f → ±0.005, generous bound 0.01).
    # A formula divergence (upstream changes the 100/√10/25 weights) would
    # otherwise slip through without notice.
    if abs(score_recomputed - final) > 0.01:
        raise RuntimeError(
            f"[evaluate] score formula divergence: reported final={final:.4f} "
            f"but recomputed (100*seg + sqrt(10*pose) + 25*rate) = "
            f"{score_recomputed:.4f}. Diff={abs(score_recomputed - final):.4f} "
            f"exceeds 0.01 tolerance. Upstream may have changed weights."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "final_score": final,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "rate_unscaled": rate_unscaled,
        "score_pose_contribution": score_pose,
        "score_seg_contribution": score_seg,
        "score_rate_contribution": score_rate,
        "score_recomputed_from_components": score_recomputed,
        "archive_size_bytes": archive_size,
        "n_samples": actual_n,
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive", type=Path, required=True,
                        help="Path to archive.zip — the submission to evaluate")
    parser.add_argument("--inflate-sh", type=Path,
                        default=Path("submissions/robust_current/inflate.sh"),
                        help="Submission's inflate.sh (default: robust_current)")
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"),
                        help="upstream/ root (has evaluate.py, modules.py, videos/)")
    parser.add_argument("--video-names-file", type=Path,
                        default=Path("upstream/public_test_video_names.txt"),
                        help="Test video names list (one per line)")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="Eval device. WARNING: mps is NOISE on PoseNet "
                             "(23x drift vs CUDA per CLAUDE.md). Use cuda only "
                             "for trustworthy scores.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Working directory (default: tempfile)")
    parser.add_argument("--inflate-timeout", type=int, default=1800,
                        help="Inflate.sh timeout in seconds. Contest budget "
                             "is 30 min (1800s) on T4. Default matches.")
    parser.add_argument("--evaluate-timeout", type=int, default=1800,
                        help="upstream/evaluate.py timeout in seconds.")
    parser.add_argument("--keep-work-dir", action="store_true",
                        help="Don't delete work dir on success (for debugging)")
    args = parser.parse_args()

    # Resolve required paths
    archive = args.archive.resolve()
    if not archive.exists():
        raise SystemExit(f"--archive does not exist: {archive}")
    inflate_sh = args.inflate_sh.resolve()
    if not inflate_sh.exists():
        raise SystemExit(f"--inflate-sh does not exist: {inflate_sh}")
    upstream_dir = args.upstream_dir.resolve()
    if not (upstream_dir / "evaluate.py").exists():
        raise SystemExit(
            f"--upstream-dir missing evaluate.py: {upstream_dir}. "
            f"Did you forget to clone the pinned upstream snapshot?"
        )
    video_names_file = args.video_names_file.resolve()
    if not video_names_file.exists():
        # Common alt path
        alt = upstream_dir / "public_test_video_names.txt"
        if alt.exists():
            video_names_file = alt
        else:
            raise SystemExit(f"--video-names-file does not exist: {video_names_file}")

    _ensure_uv_available()

    # Set up working directory in canonical contest-shape:
    #   work/
    #     archive.zip       (the submission)
    #     extracted/        (archive contents)
    #     inflated/         (inflate.sh output)
    #     report.txt        (evaluate.py output)
    #     provenance.json   (env snapshot)
    #     contest_auth_eval.json  (final result)
    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="contest_auth_"))
        cleanup = not args.keep_work_dir

    try:
        # Copy archive into work_dir so submission_dir layout matches what
        # upstream/evaluate.py expects: it reads (submission_dir / archive.zip).
        archive_in_work = work_dir / "archive.zip"
        shutil.copy2(archive, archive_in_work)

        # Provenance snapshot
        prov = _record_provenance(work_dir, archive, inflate_sh, upstream_dir, args)
        print(f"[contest_auth_eval] provenance saved: {work_dir / 'provenance.json'}")
        print(f"[contest_auth_eval] archive sha256: {prov['archive_sha256']}")

        # Stage 1: extract archive
        extracted = work_dir / "extracted"
        members = _extract_archive(archive_in_work, extracted)
        print(f"[contest_auth_eval] extracted {len(members)} member(s): {members}")

        # Stage 2: run submission's inflate.sh on the extracted archive dir
        inflated = work_dir / "inflated"
        _run_inflate(
            inflate_sh, extracted, inflated, video_names_file,
            timeout=args.inflate_timeout,
        )

        # Stage 3: run upstream/evaluate.py on submission_dir = work_dir
        # Note: evaluate.py needs (submission_dir / 'archive.zip') AND
        # (submission_dir / 'inflated/'). work_dir has both.
        result = _run_upstream_evaluate(
            upstream_dir, work_dir,
            uncompressed_dir=upstream_dir / "videos",
            video_names_file=video_names_file,
            device=args.device,
            timeout=args.evaluate_timeout,
        )

        # Save final JSON next to the work dir
        result["provenance"] = prov
        result["work_dir"] = str(work_dir)
        out_json = work_dir / "contest_auth_eval.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        # Print sentinel line for downstream parsers (matches the format
        # auth_eval_renderer.py uses, so existing log scrapers keep working)
        print(f"\nRESULT_JSON: {json.dumps(result)}")
        print(f"\n=== CONTEST AUTH EVAL ===")
        print(f"  Final score:    {result['final_score']:.4f}")
        print(f"  PoseNet dist:   {result['avg_posenet_dist']:.6f}")
        print(f"  SegNet dist:    {result['avg_segnet_dist']:.6f}")
        print(f"  Rate (unscaled): {result['rate_unscaled']:.6f}")
        print(f"  Archive bytes:  {result['archive_size_bytes']:,}")
        print(f"  Result JSON:    {out_json}")

        return 0
    finally:
        if cleanup:
            print(f"[contest_auth_eval] cleaning up {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
