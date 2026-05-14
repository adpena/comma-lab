#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download tto_frames.pt from Modal volume and preserve permanently.

Downloads TTO output frames from Modal volumes before access expires.
These are the highest-quality TTO frames (500 steps) and represent
irreplaceable experiment results.

Usage:
    python scripts/download_modal_tto_frames.py
    python scripts/download_modal_tto_frames.py --tag asym_v5_lagrangian_fixed
    python scripts/download_modal_tto_frames.py --verify-only
    python scripts/download_modal_tto_frames.py --list

Known TTO frame locations on Modal:
    asym_v5_lagrangian_fixed/tto_v5a_output_mse/tto_frames.pt   (auth 0.43, 500 steps)
    asym_v5_lagrangian_fixed/tto_v5b_embedding/tto_frames.pt    (auth 0.41, 500 steps)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "tto_frames"
VOLUME_NAME = "tac-asymmetric-results"

# Known TTO frame locations with metadata
KNOWN_FRAMES = {
    "v5a_output_mse": {
        "remote_path": "asym_v5_lagrangian_fixed/tto_v5a_output_mse/tto_frames.pt",
        "description": "TTO v5a (gradient fix) -- auth 0.43, 500 steps, MSE loss",
        "auth_score": 0.43,
        "tto_steps": 500,
        "loss_mode": "output_mse",
        "expected_shape": [1200, 384, 512, 3],
        "expected_frames": 1200,
    },
    "v5b_embedding": {
        "remote_path": "asym_v5_lagrangian_fixed/tto_v5b_embedding/tto_frames.pt",
        "description": "TTO v5b (embedding loss) -- auth 0.41, 500 steps, embedding loss",
        "auth_score": 0.41,
        "tto_steps": 500,
        "loss_mode": "embedding",
        "expected_shape": [1200, 384, 512, 3],
        "expected_frames": 1200,
    },
}

# ANSI colors
_RED = "\033[91m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _modal_bin() -> str:
    """Find the modal CLI binary."""
    venv_modal = REPO_ROOT / ".venv" / "bin" / "modal"
    if venv_modal.exists():
        return str(venv_modal)
    found = shutil.which("modal")
    if found:
        return found
    raise FileNotFoundError("modal CLI not found. Install with: uv pip install modal")


def _run_modal(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a modal CLI command."""
    modal = _modal_bin()
    return subprocess.run([modal] + args, capture_output=True, text=True, timeout=timeout)  # subprocess-no-check-OK: wrapper returns CompletedProcess; every caller inspects .returncode


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_tto_frames(filepath: Path, expected_meta: dict) -> dict:
    """Verify downloaded tto_frames.pt has correct shape, dtype, frame count.

    Returns a verification dict with results.
    """
    try:
        import torch
    except ImportError:
        return {
            "verified": False,
            "error": "torch not available for verification",
        }

    try:
        frames = torch.load(filepath, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "verified": False,
            "error": f"Failed to load: {exc}",
        }

    shape = list(frames.shape) if hasattr(frames, "shape") else None
    dtype = str(frames.dtype) if hasattr(frames, "dtype") else None
    n_frames = frames.shape[0] if shape else None

    expected_shape = expected_meta.get("expected_shape")
    expected_frames = expected_meta.get("expected_frames")

    issues: list[str] = []

    if expected_frames and n_frames != expected_frames:
        issues.append(f"Frame count: got {n_frames}, expected {expected_frames}")

    if expected_shape and shape != expected_shape:
        issues.append(f"Shape: got {shape}, expected {expected_shape}")

    return {
        "verified": len(issues) == 0,
        "shape": shape,
        "dtype": dtype,
        "n_frames": n_frames,
        "size_bytes": filepath.stat().st_size,
        "issues": issues,
    }


def download_frame(
    name: str,
    meta: dict,
    output_dir: Path,
    volume: str = VOLUME_NAME,
    force: bool = False,
) -> bool:
    """Download a single tto_frames.pt from Modal volume.

    Returns True on success.
    """
    dest_dir = output_dir / name
    dest_file = dest_dir / "tto_frames.pt"
    meta_file = dest_dir / "metadata.json"

    if dest_file.exists() and not force:
        print(f"  {_YELLOW}Already exists: {dest_file} (use --force to re-download){_RESET}")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {_BOLD}Downloading {name}...{_RESET}")
    print(f"    Remote: {volume}:/{meta['remote_path']}")
    print(f"    Local:  {dest_file}")

    # Download to a temp file first, then move (atomic)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=str(dest_dir), delete=False) as tmp:
        tmp_path = tmp.name

    result = _run_modal(
        ["volume", "get", volume, meta["remote_path"], tmp_path, "--force"],
        timeout=1800,  # 30 min for large files
    )

    if result.returncode != 0:
        Path(tmp_path).unlink(missing_ok=True)
        print(f"    {_RED}Download failed: {result.stderr.strip()}{_RESET}")
        return False

    # Move to final destination
    os.rename(tmp_path, str(dest_file))
    size_mb = dest_file.stat().st_size / (1024 * 1024)
    print(f"    {_GREEN}Downloaded: {size_mb:.1f} MB{_RESET}")

    # Verify
    print(f"    Verifying...")
    verification = _verify_tto_frames(dest_file, meta)

    if verification["verified"]:
        print(f"    {_GREEN}Verified: shape={verification['shape']}, dtype={verification['dtype']}{_RESET}")
    else:
        issues = verification.get("issues", [])
        error = verification.get("error", "")
        if error:
            print(f"    {_YELLOW}Verification skipped: {error}{_RESET}")
        for issue in issues:
            print(f"    {_RED}ISSUE: {issue}{_RESET}")

    # Compute hash
    sha = _sha256(dest_file)

    # Write metadata
    metadata = {
        "name": name,
        "description": meta["description"],
        "source": f"modal:{volume}",
        "remote_path": meta["remote_path"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "auth_score": meta.get("auth_score"),
        "tto_steps": meta.get("tto_steps"),
        "loss_mode": meta.get("loss_mode"),
        "sha256": sha,
        "size_bytes": dest_file.stat().st_size,
        "verification": verification,
    }
    meta_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"    Metadata: {meta_file}")

    return True


def list_volume(tag: str | None = None, volume: str = VOLUME_NAME) -> int:
    """List contents of the Modal volume."""
    path = tag or ""
    print(f"\n{_BOLD}Modal volume: {volume}/{path}{_RESET}\n")

    result = _run_modal(["volume", "ls", volume, path])
    if result.returncode != 0:
        print(f"{_RED}Failed to list volume: {result.stderr.strip()}{_RESET}")
        return 1

    print(result.stdout)
    return 0


def verify_local(output_dir: Path) -> int:
    """Verify all locally stored TTO frames."""
    print(f"\n{_BOLD}Verifying local TTO frames{_RESET}\n")

    any_issue = False
    for name, meta in KNOWN_FRAMES.items():
        dest_file = output_dir / name / "tto_frames.pt"
        meta_file = output_dir / name / "metadata.json"

        print(f"  {_BOLD}{name}{_RESET}")

        if not dest_file.exists():
            print(f"    {_RED}MISSING: {dest_file}{_RESET}")
            any_issue = True
            continue

        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"    File: {dest_file} ({size_mb:.1f} MB)")

        verification = _verify_tto_frames(dest_file, meta)
        if verification["verified"]:
            print(f"    {_GREEN}Verified: shape={verification['shape']}, dtype={verification['dtype']}{_RESET}")
        else:
            for issue in verification.get("issues", []):
                print(f"    {_RED}ISSUE: {issue}{_RESET}")
            error = verification.get("error", "")
            if error:
                print(f"    {_YELLOW}Note: {error}{_RESET}")
            any_issue = True

        if meta_file.exists():
            stored = json.loads(meta_file.read_text())
            print(f"    Downloaded: {stored.get('downloaded_at', '?')}")
            print(f"    SHA-256: {stored.get('sha256', '?')[:16]}...")
        else:
            print(f"    {_YELLOW}No metadata.json{_RESET}")

        print()

    return 1 if any_issue else 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and preserve tto_frames.pt from Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", default="asym_v5_lagrangian_fixed",
                        help="Experiment tag on Modal volume")
    parser.add_argument("--volume", default=VOLUME_NAME,
                        help="Modal volume name")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Local output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing local files")
    parser.add_argument("--list", action="store_true",
                        help="List volume contents")
    parser.add_argument("--name", choices=list(KNOWN_FRAMES.keys()),
                        help="Download only a specific frame set")

    args = parser.parse_args()

    volume = args.volume

    if args.list:
        return list_volume(args.tag, volume=volume)

    if args.verify_only:
        return verify_local(args.output)

    try:
        _modal_bin()
    except FileNotFoundError as exc:
        print(f"{_RED}{exc}{_RESET}")
        return 1

    print(f"\n{_BOLD}TTO Frames Download & Preservation{_RESET}")
    print("=" * 60)
    print(f"  Volume: {volume}")
    print(f"  Output: {args.output}")
    print()

    # Select which frames to download
    if args.name:
        targets = {args.name: KNOWN_FRAMES[args.name]}
    else:
        targets = KNOWN_FRAMES

    successes = 0
    failures = 0
    for name, meta in targets.items():
        if download_frame(name, meta, args.output, volume=volume, force=args.force):
            successes += 1
        else:
            failures += 1

    print(f"\n{_BOLD}Summary:{_RESET}")
    print(f"  {_GREEN}Downloaded: {successes}{_RESET}")
    if failures:
        print(f"  {_RED}Failed: {failures}{_RESET}")

    # Final verification
    print()
    verify_local(args.output)

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
