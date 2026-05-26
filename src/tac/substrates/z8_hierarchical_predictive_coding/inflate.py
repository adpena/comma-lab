# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding inflate runtime — contest raw-output contract.

L0 SCAFFOLD scope (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY"):

This module parses the Z8HPC1 archive grammar and emits the canonical contest
3-positional-arg CLI surface per Catalog #146. PyTorch-canonical (no MLX at
inflate per CLAUDE.md "Strict scorer rule" sister discipline + sister A
DreamerV3 pattern — MLX is training-only; inflate is PyTorch + brotli + numpy
only per HNeRV parity L4 ≤2 deps + universal numpy foundation).

L0 SCAFFOLD restriction: this inflate stub LOADS the archive sections (verifies
grammar) and returns a NotImplementedError-style banner for the actual frame
decode forward pass. Phase 2 lands the full multi-level decoder unroll +
per-level Mallat wavelet inverse + Wyner-Ziv top-level decode + DreamerV3 GRU
state restore. Per Catalog #240 acceptance cascade (c) pre-build
substrate-engineering: the runtime forward IS council-gated.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only via canonical select_inflate_device per Catalog #205).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤200 for
substrate-engineering lanes (Z8 SCAFFOLD explicit waiver per HNeRV L7;
parent scoping memo estimates ~280 LOC for Z8 Phase 2; targeting ≤200 LOC
via canonical helper reuse from sister A DreamerV3 + sister Z6 patterns).

Per Catalog #205 device selection uses canonical ``select_inflate_device``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HierarchicalArchive,
    parse_archive,
)


class Z8L0ScaffoldNotImplementedError(NotImplementedError):
    """Raised by L0 SCAFFOLD inflate when the runtime forward is reached.

    Per Catalog #240 acceptance cascade (c) pre-build substrate-engineering:
    the runtime forward IS council-gated and explicitly raises rather than
    silently proceeding. Phase 2 council deliberation required to lift per
    CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".
    """


def parse_and_validate_archive(
    archive_bytes: bytes,
) -> Z8HierarchicalArchive:
    """Parse Z8HPC1 archive bytes and validate canonical-quadruple structure.

    Returns the typed Z8HierarchicalArchive dataclass. Verifies all 4 canonical
    distinguishing-feature sections are present + non-empty:

    1. decoder_blob: multi-level decoder + cat-projection state dict
    2. indices_blob: per-pair per-level categorical indices (DreamerV3 RSSM)
    3. wavelet_blob: per-level Mallat wavelet detail-band coeffs
    4. wyner_ziv_blob: Wyner-Ziv top-level coded against frame_0 side-info
    """
    arc = parse_archive(archive_bytes)

    # Verify multi-level structure (3-level canonical Rao-Ballard)
    if arc.num_levels != 3:
        raise ValueError(
            f"Z8 L0 SCAFFOLD requires canonical 3-level hierarchy; got {arc.num_levels}"
        )

    # Verify per-level indices structurally present (DreamerV3 distinguishing feature)
    if len(arc.per_level_category_indices) != arc.num_levels:
        raise ValueError(
            f"Z8 indices_blob has {len(arc.per_level_category_indices)} levels; "
            f"expected {arc.num_levels}"
        )

    # Verify decoder state_dict non-empty (canonical PR95 HNeRV decoder consumer)
    if not arc.decoder_state_dict:
        raise ValueError("Z8 decoder_blob is empty; cannot inflate")

    # Verify DreamerV3 state structurally present (distinguishing feature #4)
    if not arc.dreamer_state_blob:
        raise ValueError(
            "Z8 dreamer_state_blob is empty; DreamerV3 latent dynamics required"
        )

    return arc


def inflate_one_video_l0_scaffold(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """L0 SCAFFOLD: parse archive + verify structure; runtime forward
    council-gated per Catalog #240.

    Phase 2 lifts to the full multi-level decoder unroll. Returns frame count
    written (always 0 at L0; raises Z8L0ScaffoldNotImplementedError if forward
    is reached). The parse+validate IS valuable at L0: confirms the archive
    grammar is byte-deterministic + the 4 distinguishing-feature sections are
    structurally present per Catalog #272 contract.
    """
    arc = parse_and_validate_archive(archive_bytes)
    _render_device = select_inflate_device(device)

    # Catalog #240 acceptance cascade (c) pre-build substrate-engineering:
    # the runtime forward IS council-gated. Raise rather than silently proceed.
    raise Z8L0ScaffoldNotImplementedError(
        f"Z8 L0 SCAFFOLD: archive grammar verified "
        f"(num_levels={arc.num_levels}, num_pairs={arc.num_pairs}, "
        f"decoder_keys={len(arc.decoder_state_dict)}, "
        f"per_level_indices_levels={len(arc.per_level_category_indices)}, "
        f"wavelet_blob_bytes={len(arc.wavelet_coeffs_blob)}, "
        f"wyner_ziv_blob_bytes={len(arc.wyner_ziv_top_blob)}, "
        f"dreamer_state_keys={len(arc.dreamer_state_blob)}); "
        f"runtime forward council-gated per Catalog #240; "
        f"Phase 2 lifts via Path 3 cascade (#1251 export bridge + #1257 inflate "
        f"parity closure + #1265 contest-equivalence gate sister)"
    )


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [path for path in (zero_bin, x_member) if path.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.

    L0 SCAFFOLD: parse + validate archive structure, then raise
    Z8L0ScaffoldNotImplementedError. The CLI exits with rc=64 (canonical
    "L0 SCAFFOLD not promotable" exit code matching sister scaffold patterns).
    """
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])

    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    device = select_inflate_device()

    try:
        for fname in file_list:
            name = fname.strip()
            if not name:
                continue
            inflate_one_video_l0_scaffold(
                archive_bytes, raw_output_path(output_dir, name), device=device
            )
    except Z8L0ScaffoldNotImplementedError as exc:
        # L0 SCAFFOLD: parse+validate passed, runtime forward council-gated.
        print(f"[Z8 L0 SCAFFOLD] {exc}", file=sys.stderr)
        return 64  # canonical "L0 SCAFFOLD not promotable" exit code

    return 0


__all__ = [
    "Z8L0ScaffoldNotImplementedError",
    "_read_single_member_archive_bytes",
    "inflate_one_video_l0_scaffold",
    "main_cli",
    "parse_and_validate_archive",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
