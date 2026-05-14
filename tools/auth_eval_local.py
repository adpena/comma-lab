# SPDX-License-Identifier: MIT
"""Run contest_auth_eval LOCALLY against any recovered/local archive.

This is the operator's "what is this archive worth?" tool. It accepts either a
single ``archive.zip`` or a directory containing the archive members
(renderer.bin / masks.mkv / poses.pt) and runs the canonical
``experiments/contest_auth_eval.py`` flow with all the right env setup
(config.env, GT video presence, F5 0.mkv guard, video_names_file).

Use cases (canonical):

    # Score an archive that was just recovered from a crashed Vast.ai instance:
    python tools/auth_eval_local.py \
        --archive experiments/results/recovered_12345_lane_rm_d/workspace/archive.zip

    # Score a directory of artifacts (build the archive on-the-fly):
    python tools/auth_eval_local.py \
        --archive-dir experiments/results/recovered_12345_lane_rm_d/workspace

    # Force CPU (slow, unsupported for trustworthy scores; see CLAUDE.md):
    python tools/auth_eval_local.py --archive ... --device cpu

DESIGN NOTES:
* The score is canonical only when ``--device cuda`` is used. The tool emits
  a banner if CPU/MPS is selected so the operator knows the score is advisory.
* Auto-detects whether ``upstream/videos/0.mkv`` exists; if missing, emits a
  hard error pointing to the canonical fix (download from upstream snapshot).
* Idempotent: safe to re-run; uses a per-call work dir.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical paths (relative to REPO_ROOT). Same defaults as contest_auth_eval.
DEFAULT_INFLATE_SH = REPO_ROOT / "submissions" / "robust_current" / "inflate.sh"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO_NAMES = REPO_ROOT / "upstream" / "public_test_video_names.txt"
DEFAULT_GT_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"
CONTEST_AUTH_EVAL = REPO_ROOT / "experiments" / "contest_auth_eval.py"

# Files that legitimately compose an archive. Used by --archive-dir to know what
# to zip up.
ARCHIVE_MEMBER_NAMES: tuple[str, ...] = (
    "renderer.bin",
    "renderer.bin.br",
    "renderer.bin.zst",
    "masks.mkv",
    "optimized_poses.pt",
    "optimized_poses.pt.br",
    "poses.pt",
)


def _detect_archive_members(src_dir: Path) -> list[Path]:
    """Find archive-eligible files inside src_dir (non-recursive first, then
    recursive if nothing at top level)."""
    src_dir = src_dir.resolve()
    found: list[Path] = []
    for name in ARCHIVE_MEMBER_NAMES:
        candidates = [src_dir / name]
        if not candidates[0].exists():
            # Search one level deep for sub-tools that lay artifacts under
            # workspace/ or extracted/.
            candidates = list(src_dir.rglob(name))
        for c in candidates:
            if c.is_file():
                found.append(c)
                break  # one match per pattern
    return found


def _build_archive_from_dir(src_dir: Path, dest: Path) -> Path:
    """Pack a directory of artifacts into a deterministic archive.zip.

    Mirrors the deterministic-zip pattern enforced by preflight check 19
    (``check_archive_builders_use_deterministic_zip``).
    """
    members = _detect_archive_members(src_dir)
    if not members:
        raise SystemExit(
            f"FATAL: no archive members found in {src_dir}.\n"
            f"       Expected one of: {ARCHIVE_MEMBER_NAMES}.\n"
            f"       Pass --archive <path/to/archive.zip> instead."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic write: fixed mtime, sorted names, no compression-level
    # nondeterminism (DEFLATE level 9 chosen by zipfile is fine; what matters
    # for byte-determinism is the mtime field, not the compressor).
    fixed_mtime = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf:
        for m in sorted(members, key=lambda p: p.name):
            info = zipfile.ZipInfo(filename=m.name, date_time=fixed_mtime)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, m.read_bytes())
    return dest


def _verify_gt_video_present() -> None:
    if not DEFAULT_GT_VIDEO.exists():
        raise SystemExit(
            f"FATAL: GT video {DEFAULT_GT_VIDEO} missing.\n"
            f"       Re-clone the pinned upstream snapshot or restore from\n"
            f"       upstream/videos/. Without 0.mkv, evaluate.py crashes."
        )


def _verify_inflate_sh_present(inflate_sh: Path) -> None:
    if not inflate_sh.exists():
        raise SystemExit(
            f"FATAL: --inflate-sh {inflate_sh} not found.\n"
            f"       Default: {DEFAULT_INFLATE_SH} (submissions/robust_current)."
        )
    cfg = inflate_sh.parent / "config.env"
    if not cfg.exists():
        raise SystemExit(
            f"FATAL: {cfg} missing — inflate.sh would fall through to ffmpeg path.\n"
            f"       Per Codex F5 (2026-04-28) every submission_dir MUST have\n"
            f"       config.env with PYTHON_INFLATE=renderer."
        )


def run_local_eval(
    archive: Path,
    *,
    inflate_sh: Path = DEFAULT_INFLATE_SH,
    upstream_dir: Path = DEFAULT_UPSTREAM,
    video_names_file: Path = DEFAULT_VIDEO_NAMES,
    device: str = "cuda",
    work_dir: Path | None = None,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    keep_work_dir: bool = False,
) -> int:
    """Invoke experiments/contest_auth_eval.py via subprocess.

    Returns the subprocess return code. The eval emits a RESULT_JSON sentinel
    line on stdout that operator-side log scrapers can parse.
    """
    if device != "cuda":
        sys.stderr.write(
            f"[auth_eval_local] WARNING: device={device}. Score is ADVISORY ONLY.\n"
            f"  CLAUDE.md non-negotiable: only --device cuda gives a contest-CUDA score.\n"
            f"  MPS PoseNet drift is up to 23x; CPU has no formal validation.\n"
        )
    _verify_gt_video_present()
    _verify_inflate_sh_present(inflate_sh)
    if not CONTEST_AUTH_EVAL.exists():
        raise SystemExit(
            f"FATAL: {CONTEST_AUTH_EVAL} not found. Repository may be incomplete."
        )

    cmd: list[str] = [
        sys.executable, "-u", str(CONTEST_AUTH_EVAL),
        "--archive", str(archive),
        "--inflate-sh", str(inflate_sh),
        "--upstream-dir", str(upstream_dir),
        "--video-names-file", str(video_names_file),
        "--device", device,
        "--inflate-timeout", str(inflate_timeout),
        "--evaluate-timeout", str(evaluate_timeout),
    ]
    if work_dir is not None:
        cmd += ["--work-dir", str(work_dir)]
    if keep_work_dir:
        cmd += ["--keep-work-dir"]
    print(f"[auth_eval_local] invoking: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Local contest_auth_eval wrapper. Accepts --archive (zip) or "
            "--archive-dir (directory of artifacts). See module docstring."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=Path, default=None,
                     help="Path to a complete archive.zip")
    src.add_argument("--archive-dir", type=Path, default=None,
                     help="Directory of archive artifacts; tool builds the zip")
    p.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--video-names-file", type=Path, default=DEFAULT_VIDEO_NAMES)
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Working directory (default: tempfile inside contest_auth_eval)")
    p.add_argument("--inflate-timeout", type=int, default=1800)
    p.add_argument("--evaluate-timeout", type=int, default=1800)
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Don't delete the work dir on success (for debugging)")
    p.add_argument("--archive-out", type=Path, default=None,
                   help=(
                       "When --archive-dir is used, write the built zip here "
                       "instead of a temp file. Useful for handing the archive "
                       "to Modal afterwards."
                   ))
    args = p.parse_args(argv)

    if args.archive is not None:
        archive_path = args.archive.resolve()
        if not archive_path.exists():
            raise SystemExit(f"--archive does not exist: {archive_path}")
    else:
        src_dir = args.archive_dir.resolve()
        if not src_dir.is_dir():
            raise SystemExit(f"--archive-dir is not a directory: {src_dir}")
        if args.archive_out is not None:
            archive_path = args.archive_out.resolve()
        else:
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="auth_eval_local_"))
            archive_path = tmp / "archive.zip"
        _build_archive_from_dir(src_dir, archive_path)
        print(
            f"[auth_eval_local] built archive {archive_path} "
            f"({archive_path.stat().st_size:,} bytes) from {src_dir}",
            file=sys.stderr,
        )

    return run_local_eval(
        archive=archive_path,
        inflate_sh=args.inflate_sh,
        upstream_dir=args.upstream_dir,
        video_names_file=args.video_names_file,
        device=args.device,
        work_dir=args.work_dir,
        inflate_timeout=args.inflate_timeout,
        evaluate_timeout=args.evaluate_timeout,
        keep_work_dir=args.keep_work_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
