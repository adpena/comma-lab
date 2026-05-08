#!/usr/bin/env python3
# no-argparse-OK: thin verification harness; no flags needed.
"""Verify Path B step 6 ADMM archive can be rebuilt deterministically.

Source-of-finding
-----------------
Codex adversarial review 2026-05-08 (memory:
``feedback_codex_adversarial_review_4_landings_20260508.md``) found that
``experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/archive.zip``
(153,699 B at SHA-256
``23c662d6f7a245debbd728aa8b321f7ac347dc41dcea8ffadfda69922da54f87``)
exists only on dirty disk and is NOT in commit/HEAD. A clean checkout
cannot reproduce the candidate without re-running
``tools/build_admm_x_lossy_coarsening_path_b_step6.py``; the manifest
cites the source state_dict but the encoder is non-trivially deterministic
only if seeds + inputs are pinned (Yousfi voice: "custody is the rule").

This script is the rebuild-and-SHA-assert smoke — Codex Option (b) chosen
to preserve repo size + provide reproducibility audit. It re-runs the
canonical build script against the committed source state_dict + frontier
archive, then asserts the SHA-256 of the rebuilt archive matches the
expected value.

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/verify_admm_step6_archive_sha256.py

Exit codes:
- 0: rebuild succeeded AND SHA matches expected
- 1: rebuild produced different SHA -> reproducibility hole
- 2: rebuild script failed to run
- 3: required input artifacts missing

Determinism guarantees
----------------------
- Input state_dict pinned at
  ``experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt``
  (committed to HEAD; sha-stable).
- Frontier archive pinned via the build script's defaults.
- Brotli quality (q=11) + lgwin (default) pinned via build script defaults.
- ZIP timestamps pinned to (1980,1,1,0,0,0) by the canonical archive
  builder (deterministic_zip rule); see build_admm_x_lossy_coarsening_path_b_step6.py.

Custody + CLAUDE.md compliance
------------------------------
- ``[CPU-prep proxy]`` evidence_grade: pure-CPU rebuild + SHA assertion.
- No scorer load; no CUDA/MPS; no score claim.
- Per "forbidden_premature_class_level_falsification": this verifier does
  NOT make any KILL claim on lossy_coarsening; it only verifies that the
  candidate archive can be reproduced from committed source.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ARCHIVE_SHA256 = (
    "23c662d6f7a245debbd728aa8b321f7ac347dc41dcea8ffadfda69922da54f87"
)
EXPECTED_ARCHIVE_BYTES = 153_699
CANONICAL_ARCHIVE_REL = (
    "experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/archive.zip"
)
BUILD_SCRIPT_REL = "tools/build_admm_x_lossy_coarsening_path_b_step6.py"
INPUT_STATE_DICT_REL = (
    "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_inputs() -> None:
    missing: list[str] = []
    for rel in (BUILD_SCRIPT_REL, INPUT_STATE_DICT_REL):
        if not (REPO_ROOT / rel).is_file():
            missing.append(rel)
    if missing:
        sys.stderr.write(
            "[verify] FATAL: required input artifacts missing from repo:\n"
        )
        for rel in missing:
            sys.stderr.write(f"  - {rel}\n")
        sys.stderr.write(
            "  Cannot rebuild archive without these. Aborting.\n"
        )
        sys.exit(3)


def _rebuild_to_temp(temp_root: Path) -> Path:
    """Re-run the build script with --output-root pointed at temp_root.

    The build script writes to
    ``experiments/results/admm_x_lossy_coarsening_path_b_step6_<UTC>/`` by
    default; we override the parent dir via env var if the script supports
    it, or by invoking with --output-root if available. The current build
    script does not expose --output-root, so we run it normally and detect
    the newly created result dir.
    """
    # The current build script does not accept --output-root; it writes to
    # experiments/results/admm_x_lossy_coarsening_path_b_step6_<UTC>/ in the
    # repo. We run it from a temp working dir but it still writes into the
    # repo. To avoid polluting the canonical experiments/results/ tree on
    # every verification, we run the build script and capture its output
    # dir from stdout, then move the produced archive to temp_root for SHA
    # comparison + clean up the result dir.
    cmd = [
        sys.executable,
        str(REPO_ROOT / BUILD_SCRIPT_REL),
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

    # Find the most recently produced step6 archive directory.
    step6_root = REPO_ROOT / "experiments/results"
    candidates = sorted(
        step6_root.glob("admm_x_lossy_coarsening_path_b_step6_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.stderr.write(
            "[verify] FATAL: build script ran but produced no step6 result dir\n"
        )
        sys.exit(2)
    fresh_dir = candidates[0]
    rebuilt_archive = fresh_dir / "archive.zip"
    if not rebuilt_archive.is_file():
        sys.stderr.write(
            f"[verify] FATAL: rebuilt archive missing at {rebuilt_archive}\n"
        )
        sys.exit(2)

    # Copy the rebuilt archive into temp_root for SHA comparison; leave the
    # result dir in place (it is a fresh forensic record per CLAUDE.md
    # "Track small durable .omx/research ledgers and small structured
    # summaries"). Operators can prune stale step6_<UTC>/ dirs separately.
    target = temp_root / "rebuilt_archive.zip"
    shutil.copy2(rebuilt_archive, target)
    print(
        f"[verify] rebuilt archive at {rebuilt_archive.relative_to(REPO_ROOT)}",
        file=sys.stderr,
    )
    return target


def main() -> int:
    _check_inputs()
    canonical_path = REPO_ROOT / CANONICAL_ARCHIVE_REL
    if canonical_path.is_file():
        canonical_sha = _sha256_of_file(canonical_path)
        canonical_size = canonical_path.stat().st_size
        print(
            f"[verify] canonical archive on disk: {CANONICAL_ARCHIVE_REL} "
            f"size={canonical_size:,} B sha256={canonical_sha}",
            file=sys.stderr,
        )
        if canonical_sha != EXPECTED_ARCHIVE_SHA256:
            sys.stderr.write(
                f"[verify] WARN: canonical on-disk archive has unexpected "
                f"SHA {canonical_sha} != {EXPECTED_ARCHIVE_SHA256}; the "
                "expected SHA is treated as authoritative; rebuild will "
                "be tested against EXPECTED.\n"
            )
        if canonical_size != EXPECTED_ARCHIVE_BYTES:
            sys.stderr.write(
                f"[verify] WARN: canonical on-disk archive has unexpected "
                f"size {canonical_size} != {EXPECTED_ARCHIVE_BYTES}\n"
            )
    else:
        print(
            f"[verify] canonical archive NOT on disk at {CANONICAL_ARCHIVE_REL} "
            "(expected — Codex CRITICAL #2.1 motivated this verifier); "
            "proceeding to rebuild from committed source.",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        rebuilt = _rebuild_to_temp(temp_root)
        rebuilt_sha = _sha256_of_file(rebuilt)
        rebuilt_size = rebuilt.stat().st_size
        print(
            f"[verify] rebuilt archive size={rebuilt_size:,} B sha256={rebuilt_sha}",
            file=sys.stderr,
        )

        if rebuilt_sha != EXPECTED_ARCHIVE_SHA256:
            sys.stderr.write(
                "[verify] FAIL: rebuilt archive SHA mismatch\n"
                f"  expected: {EXPECTED_ARCHIVE_SHA256}\n"
                f"  actual:   {rebuilt_sha}\n"
                "  Reproducibility hole: build script is non-deterministic "
                "OR a dependency drifted (brotli/numpy/torch versions, "
                "state_dict bytes). Investigate immediately.\n"
            )
            return 1
        if rebuilt_size != EXPECTED_ARCHIVE_BYTES:
            sys.stderr.write(
                "[verify] FAIL: rebuilt archive size mismatch\n"
                f"  expected: {EXPECTED_ARCHIVE_BYTES}\n"
                f"  actual:   {rebuilt_size}\n"
            )
            return 1

    print(
        f"[verify] OK: rebuilt archive matches expected "
        f"size={EXPECTED_ARCHIVE_BYTES:,} B sha256={EXPECTED_ARCHIVE_SHA256}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
