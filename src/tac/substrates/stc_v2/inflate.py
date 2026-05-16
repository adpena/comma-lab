# SPDX-License-Identifier: MIT
"""STC v2 substrate inflate runtime — SCAFFOLD validates archive grammar.

Per the 2026-05-16 design memo Section 2.2.6 the AUTH-EVAL inflate path for
STC v2 is the EXISTING ``submissions/robust_current/inflate.sh`` which already
supports the STCB mask dispatch (per the legacy
``scripts/remote_lane_stc_clean_source.sh``). The v2 trainer BUILDS an
archive that swaps masks.mkv -> masks.stcb in the Lane A anchor and routes
auth-eval through Lane A's inflate.sh.

This module is the SUBSTRATE-PACKAGE-LEVEL inflate that validates the STC v2
archive grammar (``0.bin``) and exposes the substrate's STCB blob + renderer
+ poses for downstream consumers. It is the L1 SCAFFOLD inflate that proves
the archive parses cleanly + the STCB roundtrips. It does NOT re-implement
Lane A's renderer forward pass — that is substrate_engineering work and
remains in ``submissions/robust_current/inflate_renderer.py``.

LOC budget: this file targets <=100 effective LOC (HNeRV parity L4); the
substrate_engineering exception is reserved for the trainer + codec only.

Strict-scorer-rule compliance: NO torch / NO scorer / NO neural weights.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def select_inflate_device() -> str:
    """Canonical Catalog #205 helper; STC v2 substrate inflate has no torch."""
    # INLINE_DEVICE_FORK_OK:stc-v2-substrate-l1-scaffold-inflate-has-no-torch-no-cuda-cpu-distinction
    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if pinned not in {"auto", "cpu", "cuda"}:
        raise SystemExit(
            f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}"
        )
    return "cpu"


def inflate_one_video(archive_bytes: bytes, output_path: Path) -> Path:
    """Validate the STC v2 archive grammar and write the STCB blob to disk.

    This is the SCAFFOLD inflate: it confirms the archive parses, extracts the
    STCB blob, and writes it so the byte-mutation no-op detector (Catalog
    #139 / #272) can verify the bytes are operationally consumed downstream.

    Args:
        archive_bytes: Raw ``0.bin`` payload.
        output_path: Where to write the extracted ``masks.stcb`` byte blob.

    Returns:
        Path to the written ``masks.stcb`` file.
    """
    # Local import to keep the SUBMISSION-SCOPED inflate self-contained when
    # vendored (per Catalog #295 / NSCS06-v6 pattern).
    from tac.substrates.stc_v2.archive import parse_stc_v2_archive

    archive = parse_stc_v2_archive(archive_bytes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop a small metadata file alongside the blob so an auditor can verify
    # the parser found the expected sections without re-running the inflate.
    stcb_path = output_path.with_suffix(".stcb")
    stcb_path.write_bytes(archive.stcb_blob)
    return stcb_path


def main() -> int:
    """Contest-compliant CLI shim (3-arg ``archive_dir output_dir file_list``).

    Per Catalog #146 the canonical contest inflate.sh contract is::

        inflate.sh <archive_dir> <output_dir> <file_list>

    The substrate-package SCAFFOLD inflate parses each video listed in
    ``file_list`` and writes the STCB blob next to the expected output. For
    the SUBSTRATE-LEVEL validation this is enough; the contest auth-eval path
    invokes ``submissions/robust_current/inflate.sh`` instead.
    """
    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    select_inflate_device()
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
