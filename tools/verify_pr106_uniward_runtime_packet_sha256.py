#!/usr/bin/env python3
# no-argparse-OK: thin verification harness; no flags needed.
"""Verify PR106 UNIWARD-Lagrangian runtime packet rebuilds deterministically.

Per BUGCLASSES B3 custody check (and the analog Codex Option (b) from the
admm step6 verifier): a build_manifest.json on disk that lists an
``archive_relpath`` is custody-clean ONLY if either the archive bytes are
committed OR the rebuild is reproducibly deterministic from committed
inputs.

The PR106 UNIWARD packet is BUILT by
``tools/build_pr106_uniward_runtime_packet.py`` from:

- the PR106 frontier source archive (committed under
  ``experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/``)
- the PR106 belt_and_suspenders source dir (committed alongside the archive)

The encoder is deterministic (numpy + brotli q=11; ZIP timestamps pinned to
1980-01-01 by ``write_stored_single_member_zip``); the
``UniwardWeightedAllocator`` uses int operations + λ-bisection that converges
on identical Ks for identical inputs. So a fresh checkout that re-runs the
build script SHOULD produce a byte-identical archive.

This verifier rebuilds and asserts SHA-256 + size identity against the
``build_manifest.json`` previously produced by the build tool.

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py

Exit codes
----------
- 0: rebuild succeeded AND SHA + bytes match expected
- 1: rebuild produced different SHA / bytes -> reproducibility hole
- 2: rebuild script failed to run
- 3: required inputs missing (build manifest, source archive, source dir)

Custody + CLAUDE.md compliance
------------------------------
- Pure-CPU; no scorer load; no CUDA/MPS; no score claim.
- ``[CPU-prep proxy]`` evidence_grade.
- Per ``forbidden_premature_class_level_falsification``: no kill verdict —
  this verifier only attests that the candidate archive can be rebuilt from
  committed source.
- Per BUGCLASSES B3: closes the custody gap that ``archive.zip`` itself is
  uncommitted (rebuildable + verifiable).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BUILD_SCRIPT_REL = "tools/build_pr106_uniward_runtime_packet.py"
SOURCE_ARCHIVE_REL = (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
SOURCE_DIR_REL = (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex"
    "/source/submissions/belt_and_suspenders"
)

# Search prefix for finding generated manifests (newest wins).
BUILD_OUTPUT_PREFIX = "experiments/results/pr106_uniward_runtime_packet_"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_latest_manifest() -> Path | None:
    """Return the newest pr106_uniward_runtime_packet_*/build_manifest.json
    on disk (the one this verifier should rebuild against), or None if none
    exists yet (caller will rebuild from scratch and use the freshly produced
    manifest as ground truth)."""
    parent = REPO_ROOT / "experiments/results"
    candidates = sorted(
        parent.glob("pr106_uniward_runtime_packet_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        m = c / "build_manifest.json"
        if m.is_file():
            return m
    return None


def _check_inputs() -> None:
    missing: list[str] = []
    for rel in (BUILD_SCRIPT_REL, SOURCE_ARCHIVE_REL):
        if not (REPO_ROOT / rel).is_file():
            missing.append(rel)
    if not (REPO_ROOT / SOURCE_DIR_REL).is_dir():
        missing.append(SOURCE_DIR_REL)
    if missing:
        sys.stderr.write(
            "[verify] FATAL: required input artifacts missing from repo:\n"
        )
        for rel in missing:
            sys.stderr.write(f"  - {rel}\n")
        sys.stderr.write("  Cannot rebuild archive without these. Aborting.\n")
        sys.exit(3)


def _rebuild(temp_root: Path, *, rms_target: float) -> tuple[Path, Path]:
    """Run the build script with --output-dir under ``temp_root``; return
    (rebuilt_archive_path, rebuilt_manifest_path)."""
    out_dir = temp_root / "rebuild"
    cmd = [
        sys.executable,
        str(REPO_ROOT / BUILD_SCRIPT_REL),
        "--rms-target",
        f"{rms_target}",
        "--output-dir",
        str(out_dir),
    ]
    print(f"[verify] running {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT}/src"},
    )
    if proc.returncode != 0:
        sys.stderr.write("[verify] FATAL: build script failed (rc != 0)\n")
        sys.stderr.write("---- stdout ----\n" + proc.stdout + "\n")
        sys.stderr.write("---- stderr ----\n" + proc.stderr + "\n")
        sys.exit(2)
    rebuilt_archive = out_dir / "archive.zip"
    rebuilt_manifest = out_dir / "build_manifest.json"
    if not rebuilt_archive.is_file():
        sys.stderr.write(
            f"[verify] FATAL: rebuilt archive missing at {rebuilt_archive}\n"
        )
        sys.exit(2)
    if not rebuilt_manifest.is_file():
        sys.stderr.write(
            f"[verify] FATAL: rebuilt manifest missing at {rebuilt_manifest}\n"
        )
        sys.exit(2)
    return rebuilt_archive, rebuilt_manifest


def main() -> int:
    _check_inputs()
    canonical_manifest_path = _find_latest_manifest()
    if canonical_manifest_path is None:
        print(
            "[verify] no pre-existing pr106_uniward_runtime_packet manifest "
            "on disk — building fresh + reporting SHA",
            file=sys.stderr,
        )
        expected_sha = None
        expected_bytes = None
        rms_target = 0.05
    else:
        canonical_manifest = json.loads(canonical_manifest_path.read_text())
        expected_sha = canonical_manifest["archive_sha256"]
        expected_bytes = int(canonical_manifest["archive_bytes"])
        rms_target = float(canonical_manifest["rms_target"])
        canonical_archive_path = (
            REPO_ROOT / canonical_manifest["archive_relpath"]
        )
        print(
            f"[verify] canonical manifest: "
            f"{canonical_manifest_path.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        if canonical_archive_path.is_file():
            on_disk_sha = _sha256_of_file(canonical_archive_path)
            on_disk_bytes = canonical_archive_path.stat().st_size
            print(
                f"[verify] canonical archive on disk: "
                f"{canonical_manifest['archive_relpath']} "
                f"size={on_disk_bytes:,} B sha256={on_disk_sha}",
                file=sys.stderr,
            )
            if on_disk_sha != expected_sha:
                sys.stderr.write(
                    f"[verify] WARN: canonical on-disk archive has "
                    f"unexpected SHA {on_disk_sha} != {expected_sha}; the "
                    "manifest SHA is treated as authoritative.\n"
                )
        else:
            print(
                f"[verify] canonical archive NOT on disk at "
                f"{canonical_manifest['archive_relpath']} — rebuilding from "
                "committed source (BUGCLASSES B3 closure)",
                file=sys.stderr,
            )

    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        rebuilt_archive, rebuilt_manifest = _rebuild(
            temp_root, rms_target=rms_target
        )
        rebuilt_sha = _sha256_of_file(rebuilt_archive)
        rebuilt_bytes = rebuilt_archive.stat().st_size
        rebuilt_manifest_data = json.loads(rebuilt_manifest.read_text())
        print(
            f"[verify] rebuilt archive size={rebuilt_bytes:,} B "
            f"sha256={rebuilt_sha}",
            file=sys.stderr,
        )

        if expected_sha is None:
            print(
                f"[verify] no expected SHA available — first build SHA "
                f"recorded as {rebuilt_sha}",
                file=sys.stderr,
            )
            return 0
        if rebuilt_sha != expected_sha:
            sys.stderr.write(
                "[verify] FAIL: rebuilt archive SHA mismatch\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {rebuilt_sha}\n"
                "  Reproducibility hole: build script is non-deterministic "
                "OR a dependency drifted (brotli/numpy/torch versions, "
                "PR106 source archive bytes). Investigate immediately.\n"
            )
            return 1
        if rebuilt_bytes != expected_bytes:
            sys.stderr.write(
                "[verify] FAIL: rebuilt archive size mismatch\n"
                f"  expected: {expected_bytes}\n"
                f"  actual:   {rebuilt_bytes}\n"
            )
            return 1
        # Also check key manifest fields match (rebuild stability).
        for field in (
            "rms_target",
            "achieved_rel_err",
            "decoder_packed_brotli_bytes",
            "per_tensor_K",
        ):
            if rebuilt_manifest_data.get(field) != json.loads(
                canonical_manifest_path.read_text()
            ).get(field):
                sys.stderr.write(
                    "[verify] FAIL: rebuilt manifest field mismatch: "
                    f"{field}\n"
                )
                return 1

    print(
        f"[verify] OK: rebuilt archive matches expected "
        f"size={expected_bytes:,} B sha256={expected_sha}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
