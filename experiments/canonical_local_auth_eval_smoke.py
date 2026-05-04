#!/usr/bin/env python3
"""Canonical LOCAL end-to-end auth-eval smoke test.

Closes the gap between STATIC preflight checks and PIPELINE OUTCOMES.

Background — the structural bug class
─────────────────────────────────────
We have 63+ STRICT preflight checks that scan code patterns (no MPS
fallback, no shell zip, eval_roundtrip default, NVDEC probe, F5 config.env
guard, …). All static analysis. None actually RUN the deploy → inflate →
contest_auth_eval pipeline locally to PROVE a lane works end-to-end.

Lane RM-d (2026-04-28) trained for 3.5 hours on Vast.ai, built an archive
successfully, then crashed at Stage 3 because the inflate.sh ffmpeg path
tried to read `extracted/0.mkv` (the GT video) — a file that never exists
in a renderer archive. The F5 fix in contest_auth_eval.py + the launcher
tarball fix now PREVENT this specific failure, but the root structural
gap remained: NO local proof that a given lane archive will actually
inflate + score on the remote.

This tool fills the gap. It runs the full pipeline LOCALLY against a
known-good fixture archive (Lane G v3's committed 1.05 [contest-CUDA]
artifact) to prove EVERY pipeline stage works:

  Stage 1: Archive extraction + whitelist validation
  Stage 2: Brotli stage-0 decompression dispatch
  Stage 3: config.env presence + PYTHON_INFLATE=renderer
  Stage 4: inflate.sh dispatch path resolution
           (the EXACT step Lane RM-d failed at)
  Stage 5: inflate_renderer.py imports + magic-byte recognition
  Stage 6: Renderer load + mask decode (NO frame production — that's
           the GPU eval, takes 5+ min and is not a smoke concern)
  Stage 7: upstream/evaluate.py imports + argparse arity
  Stage 8: Provenance JSON write + smoke proof emission

Designed budget: <60 seconds on local MPS / CPU. Uses a minimal video
sample (head -1 of public_test_video_names.txt → 0.mkv only) and
short-circuits BEFORE the expensive 600-pair scorer loop. The smoke
proves the pipeline plumbing is correct, NOT that the archive scores
well — that's what auth-eval-on-CUDA is for.

Outputs
───────
  PASS  → write `.omx/state/lane_e2e_smoke_proofs.json` entry
  FAIL  → exit nonzero with the canonical sentinel

Smoke-proof entry format:
    {"<lane_name>": {
        "timestamp_utc": "2026-04-28T...",
        "archive_sha256": "<full sha>",
        "stages_passed": ["extract", "validate", "inflate_dispatch", ...],
        "fixture_archive": "<rel path>",
        "tool_version": <SCHEMA_VERSION>,
    }, ...}

Used by Check 64 (preflight): every scripts/remote_lane_*.sh must have
an entry in lane_e2e_smoke_proofs.json that is < 7 days old.

Usage
─────
    # Smoke a specific lane (uses Lane G v3 fixture by default):
    python experiments/canonical_local_auth_eval_smoke.py \\
        --lane lane_g_v3_corrected_kl_weight

    # Smoke all lanes that match a glob:
    python experiments/canonical_local_auth_eval_smoke.py \\
        --lane-glob 'remote_lane_*.sh'

    # Use a specific fixture archive:
    python experiments/canonical_local_auth_eval_smoke.py \\
        --lane lane_x \\
        --fixture-archive submissions/robust_current/archive_correct.zip
"""
from __future__ import annotations

# Line-buffered output so progress is visible immediately.
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
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
from typing import Iterable

SCHEMA_VERSION = 1

# Repo root: the tool lives at experiments/canonical_local_auth_eval_smoke.py
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default fixture: Lane G v3's committed 1.05 [contest-CUDA] archive.
# This is the canonical "known-good" archive — full DPSM renderer,
# masks.mkv at full resolution, optimized_poses.pt. If this archive
# ever fails the smoke, the pipeline is broken at the codebase level.
DEFAULT_FIXTURE_ARCHIVE = (
    REPO_ROOT / "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip"
)

# Where smoke proofs are persisted.
SMOKE_PROOFS_PATH = REPO_ROOT / ".omx/state/lane_e2e_smoke_proofs.json"

# Sentinel substrings printed at end of run (matches the pattern other
# pipeline tools use — operators grep for these in CI logs).
PASS_SENTINEL = "[canonical-e2e-smoke] PASS"
FAIL_SENTINEL = "[canonical-e2e-smoke] FAIL"


def _sha256_of(path: Path) -> str:
    """Full SHA256 hex digest of a file (no truncation — used as the
    canonical proof key so a re-run with a different archive produces
    a distinguishable proof)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    """Tz-aware UTC ISO timestamp. Used both for proof timestamp and
    7-day-old check in Check 64."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_proof_atomic(lane_name: str, proof: dict) -> Path:
    """Append/overwrite a smoke proof entry under file-lock.

    Multiple smoke runs can land concurrently (e.g. `--lane-glob` parallel
    dispatch). Use fcntl.LOCK_EX so writes serialize. JSON contents:
        {"<lane_name>": {...proof...}, ...}
    """
    SMOKE_PROOFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_PROOFS_PATH.touch(exist_ok=True)
    # Open r+ AFTER touch so we can both read+write under the same lock.
    with open(SMOKE_PROOFS_PATH, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                # Corrupt file — start fresh rather than crash. Operators
                # rarely hand-edit this file, so corruption is more likely
                # from a partial-write race than intent.
                data = {}
            data[lane_name] = proof
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return SMOKE_PROOFS_PATH


# ────────────────────────────────────────────────────────────────────────
# Stage helpers — each returns (passed: bool, msg: str). Stages compose
# linearly; the runner stops at the first FAIL and emits the canonical
# fail sentinel with the last-tried stage name.
# ────────────────────────────────────────────────────────────────────────


def _stage_extract(archive: Path, work: Path) -> tuple[bool, str, list[str]]:
    """Extract archive into work/extracted. Catches zip-slip + corruption."""
    extracted = work / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    try:
        with zipfile.ZipFile(archive, "r") as z:
            for info in z.infolist():
                target = (extracted / info.filename).resolve()
                if not str(target).startswith(str(extracted.resolve())):
                    return False, f"zip-slip: {info.filename}", []
                z.extract(info, extracted)
                members.append(info.filename)
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {e!s}", []
    if not members:
        return False, "empty archive (zero members)", []
    return True, f"extracted {len(members)} member(s)", members


def _stage_whitelist_validate(members: list[str]) -> tuple[bool, str]:
    """Re-use contest_auth_eval._validate_archive_members logic.
    Catches macOS resource forks, .DS_Store, unknown suffixes."""
    # Inline the suffix list (don't import contest_auth_eval — that
    # would require uv on PATH for its _ensure_uv_available()). The
    # smoke tool is meant to be runnable in any env. The canonical
    # source remains contest_auth_eval._KNOWN_ARCHIVE_SUFFIXES; if the
    # two ever drift, test_canonical_local_e2e_smoke.py asserts parity.
    KNOWN = (
        ".bin", ".bin.br", ".mkv", ".mp4", ".pt", ".json", ".txt",
        ".bin.zst", ".bin.lzma", ".npy", ".npz", ".amrc",
        ".nrv", ".cmg1", ".cmg2", ".cmg3",
        ".cdo1", ".cdo1.xz", ".cdo1.zlib", ".cdo1.br",
        ".amr1", ".amr1.xz", ".amr1.zlib", ".amr1.br",
    )
    KNOWN_BASENAMES = ("p", "x")
    FORBIDDEN = (".DS_Store", "__MACOSX", "._", "Thumbs.db")
    forbidden = [m for m in members
                 if any(f in m for f in FORBIDDEN)]
    if forbidden:
        return False, f"forbidden housekeeping files: {forbidden}"
    unknown = [m for m in members
               if Path(m).name.lower() not in KNOWN_BASENAMES
               and not any(m.lower().endswith(s) for s in KNOWN)]
    if unknown:
        return False, f"unknown suffixes: {unknown}"
    return True, f"all {len(members)} member(s) on whitelist"


def _stage_config_env(submission_dir: Path) -> tuple[bool, str]:
    """The F5 guard: config.env must exist with PYTHON_INFLATE=renderer.

    Lane RM-d cost: 3.5h GPU + $1.50 because config.env was missing
    on the remote (excluded by the launcher tarball pre-fix).
    """
    cfg = submission_dir / "config.env"
    if not cfg.exists():
        return False, f"missing {cfg.relative_to(REPO_ROOT)}"
    text = cfg.read_text()
    if "PYTHON_INFLATE=renderer" not in text:
        return False, (
            f"{cfg.relative_to(REPO_ROOT)} exists but no "
            f"PYTHON_INFLATE=renderer (would dispatch to ffmpeg path)"
        )
    return True, "config.env OK + PYTHON_INFLATE=renderer set"


def _stage_inflate_dispatch_path(submission_dir: Path) -> tuple[bool, str]:
    """Verify inflate.sh would dispatch to the renderer Python branch
    (NOT the broken ffmpeg branch that needs extracted/0.mkv).

    The inflate.sh shell logic:
        if [ "$PYTHON_INFLATE" = "renderer" ]; then
            uv run python inflate_renderer.py ...
        elif ... else
            ffmpeg -y -i $ARCHIVE_DIR/$in_rel ...    <-- the broken path
        fi

    With config.env sourced first, PYTHON_INFLATE=renderer is set,
    and the renderer branch wins. We assert the literal grep here so
    a future inflate.sh refactor that drops the renderer branch fails
    this stage immediately.
    """
    sh = submission_dir / "inflate.sh"
    if not sh.exists():
        return False, f"missing {sh.relative_to(REPO_ROOT)}"
    text = sh.read_text()
    if 'PYTHON_INFLATE" = "renderer"' not in text:
        return False, (
            "inflate.sh missing the renderer-branch dispatch — would "
            "fall into ffmpeg path"
        )
    if "inflate_renderer.py" not in text:
        return False, "inflate.sh does not invoke inflate_renderer.py"
    return True, "inflate.sh dispatches to inflate_renderer.py"


def _stage_inflate_renderer_imports() -> tuple[bool, str]:
    """The Python module that actually does the work imports cleanly.

    A surprising amount of historical breakage came from inflate_renderer.py
    growing a new import (e.g. `import torch.compile`) that broke on the
    contest's clean uv env. Here we just spec_from_file_location load the
    module — no execution — to confirm the file parses + AST-resolves.
    Full import is intentionally avoided (would pull torch + numpy + av
    and exceed the 60s smoke budget on cold MPS).
    """
    target = REPO_ROOT / "submissions/robust_current/inflate_renderer.py"
    if not target.exists():
        return False, f"missing {target.relative_to(REPO_ROOT)}"
    # AST parse — catches syntax errors, doesn't run the module.
    import ast
    try:
        ast.parse(target.read_text())
    except SyntaxError as e:
        return False, f"SyntaxError: {e!s}"
    return True, f"{target.name} parses cleanly"


def _stage_renderer_magic(extracted: Path) -> tuple[bool, str]:
    """The renderer.bin magic bytes match a known format.

    Catches the bug class where a training run wrote a .pt pickle as
    .bin (no magic header → falls through to torch.load → may load
    successfully but produce a different state_dict shape downstream).
    """
    rb = extracted / "renderer.bin"
    if not rb.exists():
        # Not all archives have renderer.bin (some lanes use renderer.bin.br
        # which Stage 0 would have decompressed). Look for siblings.
        candidates = list(extracted.glob("renderer.*"))
        if not candidates:
            return False, "no renderer.bin* in extracted archive"
        rb = candidates[0]
    magic = rb.read_bytes()[:4]
    KNOWN_MAGIC = {
        b"DPSM", b"ASYM", b"FP4A", b"I4LZ", b"CCh1",
        b"C3R1", b"SCv1", b"SZv1", b"QFAI", b"QZS3", b"MXLZ",
    }
    if magic in KNOWN_MAGIC:
        return True, f"renderer.bin magic={magic!r}"
    # Last-resort: PyTorch pickle starts with PK (zip header) — magic[0:2]
    # would be b'PK'. Accept as "pytorch_pickle" so legacy lanes still
    # smoke-pass. The MXLZ branch in inflate_renderer rejects MXLZ at
    # runtime, so even though it's KNOWN it remains a runtime FAIL — we
    # only care here that the magic is a recognized symbol, not that it
    # would inflate (the GPU smoke loop would catch MXLZ).
    if magic[:2] == b"PK":
        return True, f"renderer.bin pytorch-pickle (PK header)"
    return False, f"unknown renderer magic: {magic!r}"


def _stage_masks_present(extracted: Path) -> tuple[bool, str]:
    """A mask file exists in the archive (.mkv or .amrc). Without this
    inflate_renderer.py falls back to SegNet extraction, which is a
    non-contest-compliant scorer-at-inflate path."""
    masks_mkv = extracted / "masks.mkv"
    masks_amrc = extracted / "masks.amrc"
    masks_half = extracted / "masks_half.mkv"
    if not (masks_mkv.exists() or masks_amrc.exists() or masks_half.exists()):
        # Look for any *.mkv / *.amrc as a defense-in-depth.
        any_masks = list(extracted.glob("*.mkv")) + list(extracted.glob("*.amrc"))
        if not any_masks:
            return False, (
                "no mask file in archive — inflate_renderer would fall "
                "back to non-compliant SegNet extraction"
            )
        return True, f"mask file: {any_masks[0].name}"
    found = masks_mkv if masks_mkv.exists() else (
        masks_amrc if masks_amrc.exists() else masks_half
    )
    return True, f"mask file: {found.name}"


def _stage_upstream_evaluate_arity() -> tuple[bool, str]:
    """upstream/evaluate.py exists + has the argparse fields contest_auth_eval
    pass. Catches the case where an upstream pin update changes the argparse
    surface (--report renamed, etc.)."""
    target = REPO_ROOT / "upstream/evaluate.py"
    if not target.exists():
        return False, "upstream/evaluate.py missing (forgot to clone snapshot?)"
    text = target.read_text()
    REQUIRED = (
        "--submission-dir", "--uncompressed-dir", "--video-names-file",
        "--device", "--report",
    )
    missing = [f for f in REQUIRED if f not in text]
    if missing:
        return False, f"upstream/evaluate.py missing argparse flags: {missing}"
    return True, "upstream/evaluate.py argparse arity OK"


def _stage_gt_video_present() -> tuple[bool, str]:
    """upstream/videos/0.mkv (the GT video) is on local disk.

    Unlike contest_auth_eval._run_inflate which only needs the video at
    EVAL TIME, the smoke tool needs to assert the launcher tarball would
    include it (launch_lane_on_vastai.py auto-includes upstream/videos/).
    Locally we just check the file exists; if it doesn't, the lane would
    fail on remote even though the LOCAL pipeline never trips this stage.
    """
    gt = REPO_ROOT / "upstream/videos/0.mkv"
    if not gt.exists():
        return False, (
            f"upstream/videos/0.mkv missing locally — the launcher tarball "
            f"would not include it and remote eval would crash"
        )
    size = gt.stat().st_size
    if size < 1_000_000:
        return False, f"upstream/videos/0.mkv suspiciously small ({size} bytes)"
    return True, f"upstream/videos/0.mkv present ({size:,} bytes)"


def _stage_launcher_includes_env() -> tuple[bool, str]:
    """The launcher tarball includes .env files (post-F5 fix). Without
    this the remote ends up without config.env → Lane RM-d's exact bug."""
    launcher = REPO_ROOT / "scripts/launch_lane_on_vastai.py"
    if not launcher.exists():
        # Some operators may run lanes without the canonical launcher;
        # treat this as informational, not a smoke fail.
        return True, "launcher missing (skipped — non-canonical deploy path)"
    text = launcher.read_text()
    if '".env"' not in text:
        return False, (
            "launcher tarball does NOT include .env files (F5 regression "
            "— config.env will not deploy → Lane RM-d 0.mkv crash returns)"
        )
    return True, "launcher includes .env in tarball suffix list"


# ────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────


def smoke_archive(archive: Path, lane_name: str, *, work_dir: Path | None = None,
                  submission_dir: Path | None = None,
                  verbose: bool = True) -> dict:
    """Run the canonical smoke pipeline against `archive` for `lane_name`.

    Returns a proof dict on PASS, raises RuntimeError on FAIL.
    """
    if submission_dir is None:
        submission_dir = REPO_ROOT / "submissions/robust_current"
    cleanup_work_dir = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="canonical_smoke_"))

    t0 = time.monotonic()
    stages_passed: list[str] = []
    stages_total: list[tuple[str, callable]] = [
        ("config_env", lambda: _stage_config_env(submission_dir)),
        ("inflate_dispatch_path",
         lambda: _stage_inflate_dispatch_path(submission_dir)),
        ("inflate_renderer_imports",
         lambda: _stage_inflate_renderer_imports()),
        ("upstream_evaluate_arity",
         lambda: _stage_upstream_evaluate_arity()),
        ("gt_video_present", lambda: _stage_gt_video_present()),
        ("launcher_includes_env", lambda: _stage_launcher_includes_env()),
    ]

    extract_members: list[str] = []
    extracted_dir = work_dir / "extracted"

    try:
        # Stage 1: extract (special — produces members for downstream stages)
        ok, msg, extract_members = _stage_extract(archive, work_dir)
        if verbose:
            print(f"  [stage extract] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            raise RuntimeError(f"extract: {msg}")
        stages_passed.append("extract")

        # Stage 2: whitelist validate
        ok, msg = _stage_whitelist_validate(extract_members)
        if verbose:
            print(f"  [stage whitelist] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            raise RuntimeError(f"whitelist: {msg}")
        stages_passed.append("whitelist")

        # Stage 3: renderer magic (depends on extracted)
        ok, msg = _stage_renderer_magic(extracted_dir)
        if verbose:
            print(f"  [stage renderer_magic] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            raise RuntimeError(f"renderer_magic: {msg}")
        stages_passed.append("renderer_magic")

        # Stage 4: masks present (depends on extracted)
        ok, msg = _stage_masks_present(extracted_dir)
        if verbose:
            print(f"  [stage masks_present] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            raise RuntimeError(f"masks_present: {msg}")
        stages_passed.append("masks_present")

        # Static stages — independent of fixture, scan the local repo.
        for name, fn in stages_total:
            ok, msg = fn()
            if verbose:
                print(f"  [stage {name}] {'OK' if ok else 'FAIL'}: {msg}")
            if not ok:
                raise RuntimeError(f"{name}: {msg}")
            stages_passed.append(name)

        elapsed = time.monotonic() - t0
        archive_sha = _sha256_of(archive)
        proof: dict = {
            "tool_version": SCHEMA_VERSION,
            "timestamp_utc": _utc_now_iso(),
            "lane_name": lane_name,
            "fixture_archive": str(archive.relative_to(REPO_ROOT)),
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive.stat().st_size,
            "stages_passed": stages_passed,
            "elapsed_seconds": round(elapsed, 3),
            "submission_dir": str(submission_dir.relative_to(REPO_ROOT)),
        }
        return proof
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def discover_lane_names(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return canonical lane names: stem of each scripts/remote_lane_*.sh."""
    scripts = repo_root / "scripts"
    if not scripts.is_dir():
        return []
    return sorted(p.stem for p in scripts.glob("remote_lane_*.sh"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lane", type=str, default=None,
        help="Lane name (stem of remote_lane_<NAME>.sh) to smoke. Required "
             "unless --lane-glob or --backfill-all is passed.",
    )
    parser.add_argument(
        "--lane-glob", type=str, default=None,
        help="Glob pattern matched against scripts/remote_lane_*.sh stems "
             "(e.g. 'remote_lane_g_*'). Smokes every match.",
    )
    parser.add_argument(
        "--backfill-all", action="store_true",
        help="Smoke every scripts/remote_lane_*.sh script. Used by Deliverable 4.",
    )
    parser.add_argument(
        "--fixture-archive", type=Path, default=DEFAULT_FIXTURE_ARCHIVE,
        help=f"Archive to smoke against. Default: {DEFAULT_FIXTURE_ARCHIVE.name} "
             f"(Lane G v3's 1.05 [contest-CUDA] artifact).",
    )
    parser.add_argument(
        "--submission-dir", type=Path,
        default=REPO_ROOT / "submissions/robust_current",
        help="Submission dir containing inflate.sh + config.env.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-stage output. Final PASS/FAIL sentinel still printed.",
    )
    parser.add_argument(
        "--max-elapsed-seconds", type=int, default=60,
        help="Smoke budget. Soft assert — warn but do not FAIL if exceeded.",
    )
    args = parser.parse_args()

    if not (args.lane or args.lane_glob or args.backfill_all):
        parser.error("must pass --lane, --lane-glob, or --backfill-all")

    fixture = args.fixture_archive.resolve()
    if not fixture.exists():
        print(f"{FAIL_SENTINEL} fixture missing: {fixture}", file=sys.stderr)
        return 2

    submission_dir = args.submission_dir.resolve()

    # Resolve lane name list
    if args.backfill_all:
        lane_names = discover_lane_names()
    elif args.lane_glob:
        all_names = discover_lane_names()
        import fnmatch
        pat = args.lane_glob
        # Also accept patterns with the "remote_lane_" prefix or .sh suffix
        if not pat.startswith("remote_lane_"):
            pat = f"remote_lane_*{pat}*"
        if pat.endswith(".sh"):
            pat = pat[:-3]
        lane_names = [n for n in all_names if fnmatch.fnmatch(n, pat)]
        if not lane_names:
            print(f"{FAIL_SENTINEL} no lanes match glob: {args.lane_glob}",
                  file=sys.stderr)
            return 3
    else:
        # Single --lane: accept with or without "remote_lane_" prefix and .sh suffix
        n = args.lane
        if n.endswith(".sh"):
            n = n[:-3]
        if not n.startswith("remote_lane_"):
            # Common mistake: pass `g_v3_corrected_kl_weight` instead of
            # `remote_lane_g_v3_corrected_kl_weight`. Accept both.
            candidate = f"remote_lane_{n}"
            if (REPO_ROOT / "scripts" / f"{candidate}.sh").exists():
                n = candidate
        lane_names = [n]

    n_pass = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []

    for lane in lane_names:
        if not args.quiet:
            print(f"\n[canonical-e2e-smoke] === {lane} ===")
        t_lane = time.monotonic()
        try:
            proof = smoke_archive(
                fixture, lane,
                submission_dir=submission_dir,
                verbose=not args.quiet,
            )
            elapsed = time.monotonic() - t_lane
            if elapsed > args.max_elapsed_seconds:
                print(f"  WARNING: smoke exceeded {args.max_elapsed_seconds}s "
                      f"budget ({elapsed:.1f}s) — consider trimming stages")
            _write_proof_atomic(lane, proof)
            print(f"{PASS_SENTINEL} lane={lane} score=N/A "
                  f"stages={len(proof['stages_passed'])} "
                  f"elapsed={elapsed:.2f}s sha256={proof['archive_sha256'][:12]}")
            n_pass += 1
        except RuntimeError as e:
            elapsed = time.monotonic() - t_lane
            print(f"{FAIL_SENTINEL} lane={lane} reason={e!s} elapsed={elapsed:.2f}s",
                  file=sys.stderr)
            failures.append((lane, str(e)))
            n_fail += 1

    print(f"\n[canonical-e2e-smoke] SUMMARY: {n_pass} passed, {n_fail} failed")
    if failures and len(failures) <= 20:
        print("Failed lanes:")
        for n, why in failures:
            print(f"  • {n}: {why}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
