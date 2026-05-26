# SPDX-License-Identifier: MIT
"""Z7-Mamba-2-v2 scorer-free inflate runtime — L0 SCAFFOLD skeleton.

Loads a Z7MCM3 archive, regenerates A_log procedurally per CC-J unwind,
replays the Mamba-2 autoregressive sequence sequentially (HNeRV parity
L4 ≤200 LOC constraint), consumes the Mamba2TemporalDecoder weight
stream, and writes contest-shaped raw RGB output.

Byte-closed runtime per CLAUDE.md HNeRV parity L4 (Inflate ≤200 LOC
substrate-engineering waiver; this file is the L0 SCAFFOLD skeleton).
NO scorer imports per CLAUDE.md "Strict scorer rule" + CC-A canonical
decoder consumes Mamba-2 temporal-conv pre-stage. CUDA/CPU agnostic
via canonical `select_inflate_device` per Catalog #205 (HARD-EARNED).

The runtime MUST use the sequential reference Mamba-2 cell, NOT the
SSD-scan CUDA kernel. Per SSD theorem (Dao-Gu 2024 §4) the sequential
unroll + chunk-parallel SSD scan produce IDENTICAL hidden states; the
sequential path simplifies inflate dependency closure (no mamba_ssm
package required on contest runtime).

L0 SCAFFOLD scope: contract + structure only. Full implementation
(Z7MCM3 archive parser + reference Mamba-2 cell + temporal decoder +
RGB writer) lands at L1 EMPIRICAL build per Phase 3 design memo §7.

[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md §7.4 MLX-implementation roadmap]
[verified-against: tac.substrates._shared.inflate_runtime canonical helpers (HARD-EARNED Catalog #205)]
"""

from __future__ import annotations

import sys
from pathlib import Path


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one Z7MCM3 archive's bytes into one contest ``.raw`` file.

    L0 SCAFFOLD: refuses to run; full implementation lands at L1.

    Per Phase 3 design memo §7.4: the runtime implementation will route
    through the canonical `tac.substrates._shared.inflate_runtime`
    helpers (select_inflate_device per Catalog #205; raw_output_path;
    write_rgb_pair_to_raw) so the per-file LOC budget stays ≤200 per
    HNeRV parity L4.

    Args:
        archive_bytes: serialized Z7MCM3 archive bytes
        output_raw_path: target .raw output path
        device: device override (None → canonical select_inflate_device
            per Catalog #205 HARD-EARNED)

    Returns:
        frames_written (int)
    """
    if not isinstance(archive_bytes, (bytes, bytearray)):
        raise TypeError(
            f"archive_bytes must be bytes-like; got {type(archive_bytes).__name__}"
        )
    if not isinstance(output_raw_path, Path):
        raise TypeError(
            f"output_raw_path must be Path; got {type(output_raw_path).__name__}"
        )
    raise NotImplementedError(
        "inflate_one_video is L0 SCAFFOLD only — full Z7MCM3 inflate runtime "
        "lands at L1 EMPIRICAL build per the Phase 3 L0 SCAFFOLD design memo "
        "§7.4 + per CC-A unwind temporal-conv decoder + CC-J unwind A_log "
        "procedural regeneration. Sequential reference Mamba-2 cell per SSD "
        "theorem (Dao-Gu 2024 §4)."
    )


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    L0 SCAFFOLD: refuses to run; full implementation lands at L1.
    """
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list> "
            "(L0 SCAFFOLD; full implementation pending at L1)",
            file=sys.stderr,
        )
        return 2
    raise NotImplementedError(
        "main_cli is L0 SCAFFOLD only — full Z7MCM3 inflate runtime lands at "
        "L1 EMPIRICAL build."
    )


__all__ = [
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover - CLI smoke
    sys.exit(main_cli())
