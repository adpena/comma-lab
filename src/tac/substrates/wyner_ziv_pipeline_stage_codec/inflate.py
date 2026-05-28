# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec inflate runtime (L0 SCAFFOLD).

Numpy-portable per HNeRV parity discipline L4 (≤200 LOC + ≤2 deps); MLX
intentionally NOT a runtime dep per 8th MLX-FIRST + numpy-portable standing
directive 2026-05-26 (MLX training, numpy inflate). PYTHONPATH self-
containment per Catalog #295.

L0 SCAFFOLD scope: this inflate runtime is the canonical contract DECLARED
per HNeRV parity L9 runtime closure. At L0 it operates on the SCAFFOLD
archive grammar (per ``archive.py::encode_archive_bytes_scaffold``) +
verifies the byte-identical roundtrip with the canonical primitive's
``reconstruct_from_wyner_ziv_layer``. At L1 the inflate runtime is wired
into the contest's ``inflate.sh archive_dir output_dir file_list`` 3-arg
contract per Catalog #146 + emits real per-pair reconstructed frames per
Catalog #369 + composes with a base substrate's inflate output per the
canonical 4-layer pattern.

Per CLAUDE.md "Strict scorer rule" non-negotiable + Catalog #6/#7: this
inflate runtime MUST NOT load PoseNet/SegNet/upstream/modules. The Wyner-Ziv
side-info Y is derived from canonical Y sources (Comma2k19LocalCache per
Catalog #213; ImageNet stats; torch_defaults; math_constants). The
``"scorer_compressed"`` side-info source is FORBIDDEN here per the primitive
+ Catalog #320.

Per Catalog #205 canonical select_inflate_device contract: the inflate
runtime exposes a ``select_inflate_device`` helper that honors the
``PACT_INFLATE_DEVICE`` env var (auto/cpu/cuda; ``mps`` refused) so the
inflate path is operator-pinnable per CLAUDE.md "Forbidden device-selection
defaults" non-negotiable.
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Any

from tac.substrates.wyner_ziv_pipeline_stage_codec.archive import (
    HEADER_FORMAT,
    HEADER_SIZE,
    WZPSC01_MAGIC,
    WZPSC01_VERSION,
    decode_archive_bytes_scaffold,
)


__all__ = (
    "select_inflate_device",
    "inflate_wyner_ziv_pipeline_stage_codec_scaffold",
    "load_archive_bytes_from_zip_member",
    "L0_SCAFFOLD_INFLATE_NOT_IMPLEMENTED_MESSAGE",
)


L0_SCAFFOLD_INFLATE_NOT_IMPLEMENTED_MESSAGE = (
    "Wyner-Ziv pipeline-stage codec L0 SCAFFOLD: contest-faithful inflate.sh "
    "3-arg invocation per Catalog #146 is council-gated pending L1 build. "
    "The L0 scaffold validates the archive grammar + verifies the byte-"
    "identical roundtrip with the canonical primitive's reconstruct_from_"
    "wyner_ziv_layer; the L1 inflate emits real per-pair reconstructed "
    "frames + composes with a base substrate's inflate output per the "
    "canonical 4-layer pattern."
)


def select_inflate_device(default: str = "auto") -> str:
    """Operator-pinnable inflate device per Catalog #205 canonical contract.

    Honors the ``PACT_INFLATE_DEVICE`` env var (``auto`` / ``cpu`` / ``cuda``).
    ``mps`` is REFUSED per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.

    Args:
        default: fallback device when PACT_INFLATE_DEVICE is unset.

    Returns:
        One of "cpu" or "cuda". (Never "mps".)

    Raises:
        ValueError: PACT_INFLATE_DEVICE set to "mps" or an unrecognized value.
    """
    env_device = os.environ.get("PACT_INFLATE_DEVICE", default).strip().lower()
    if env_device == "mps":
        raise ValueError(
            "PACT_INFLATE_DEVICE='mps' is REFUSED per CLAUDE.md 'MPS auth eval "
            "is NOISE' non-negotiable. Use PACT_INFLATE_DEVICE='cpu' or 'cuda'."
        )
    if env_device == "auto":
        # Numpy-portable inflate: default to cpu (no GPU dependency at runtime
        # per HNeRV parity L4 ≤2 deps). The L1 trainer may emit a sister
        # inflate path that auto-promotes to cuda when torch is available;
        # the L0 scaffold is cpu-only by construction.
        return "cpu"
    if env_device in ("cpu", "cuda"):
        return env_device
    raise ValueError(
        f"PACT_INFLATE_DEVICE={env_device!r} not in {{'auto', 'cpu', 'cuda'}}; "
        "'mps' is explicitly refused."
    )


def inflate_wyner_ziv_pipeline_stage_codec_scaffold(
    *,
    archive_bytes: bytes,
    side_info_y: bytes,
) -> dict[str, Any]:
    """Inflate WZPSC01 archive payload to reconstructed pre_entropy_bytes (L0 SCAFFOLD).

    Per HNeRV parity L9 runtime closure: routes through the canonical
    primitive's ``reconstruct_from_wyner_ziv_layer`` (the same primitive
    used at encode time per Catalog #335 canonical contract). The decoder-
    side side_info_y MUST equal the encoder-side side_info_y per Wyner 1976
    R(D|Y) determinism requirement (the contest scorer is decoder-deterministic
    so PoseNet output IS canonical Y when intercept_location=
    state_dict_serialization).

    Per CLAUDE.md "Strict scorer rule": this inflate runtime does NOT load
    PoseNet/SegNet. The side_info_y MUST be derived from canonical Y sources
    per the primitive's ``LEGAL_SIDE_INFO_SOURCES`` (Comma2k19 / ImageNet /
    torch_defaults / math_constants); the caller is responsible for the
    derivation per the canonical 4-layer pattern.

    Args:
        archive_bytes: the WZPSC01 archive payload (start of ``0.bin``).
        side_info_y: side info Y re-derived at inflate time per the
            canonical source declared in the archive's meta JSON.

    Returns:
        Dict with ``reconstructed_pre_entropy_bytes`` + ``layout`` (per
        :class:`tac.substrates.wyner_ziv_pipeline_stage_codec.archive.
        ArchiveLayout`) + ``meta`` decoded JSON.

    Raises:
        ValueError: malformed archive / side_info_y mismatch / codec drift.
    """
    decoded = decode_archive_bytes_scaffold(archive_bytes)
    layout = decoded["layout"]
    main_compressed = decoded["main_compressed"]
    side_compressed_baked = decoded["side_compressed_baked"]

    # Route through the canonical primitive's reconstruct path per Catalog
    # #335 canonical contract auto-discovery. The primitive is the single
    # source of truth for the WZ reconstruction; this inflate runtime is the
    # substrate-scope thin wrapper.
    #
    # NOTE: importing from tac.codec.wyner_ziv_layer adds a transitive dep
    # on lzma/brotli/zlib (stdlib + brotli pkg). brotli is in the substrate
    # contract's runtime_dep_closure=("numpy", "brotli"); lzma is stdlib. So
    # the inflate runtime's effective runtime_dep_closure stays within the
    # HNeRV parity L4 ≤2 deps budget.
    from tac.codec.wyner_ziv_layer import (  # local import for lazy stdlib pull
        InterceptLocation,
        WynerZivLayerConfig,
        reconstruct_from_wyner_ziv_layer,
    )

    config = WynerZivLayerConfig(
        intercept_location=InterceptLocation(layout.intercept_location),
        side_info_source=layout.side_info_source,
        side_info_max_bytes=layout.side_len,  # exact bake-in size from archive
        main_codec=layout.main_codec,
        compression_codec_for_side=layout.compression_codec_for_side,
    )
    reconstructed = reconstruct_from_wyner_ziv_layer(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        side_info_y=side_info_y,
        config=config,
    )
    return {
        "reconstructed_pre_entropy_bytes": reconstructed,
        "layout": layout,
        "meta": decoded["meta"],
    }


def load_archive_bytes_from_zip_member(
    archive_zip_path: str,
    member_name: str = "0.bin",
) -> bytes:
    """Load the WZPSC01 archive payload from a contest archive.zip member.

    Per HNeRV parity L3 monolithic single-file: the canonical member name is
    ``0.bin``. The L1 trainer emits the archive zip via the canonical archive-
    builder helper per Catalog #146 + the contest's 3-arg inflate.sh
    invocation.

    Args:
        archive_zip_path: path to the contest archive.zip.
        member_name: ZIP member name (default ``"0.bin"`` per HNeRV parity L3).

    Returns:
        The WZPSC01 archive payload bytes.

    Raises:
        FileNotFoundError: archive_zip_path does not exist.
        KeyError: member_name not in the zip.
    """
    import zipfile  # stdlib; not counted against runtime_dep_closure
    with zipfile.ZipFile(archive_zip_path, "r") as zf:
        with zf.open(member_name) as f:
            return f.read()


if __name__ == "__main__":  # pragma: no cover
    # L0 scaffold's __main__ is intentionally a no-op stub per Catalog #146
    # contest 3-arg inflate.sh contract (the L1 build wires the real CLI).
    print(L0_SCAFFOLD_INFLATE_NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    sys.exit(0)
