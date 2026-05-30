# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind inflate runtime — Phase C canonical contest format.

Per Catalog #146 contest-compliant inflate runtime + Catalog #367 raw-byte
fail-closed + canonical leaderboard binding-depth L5 (full RGB renderer at
camera resolution) + the 8th MLX-first standing directive (training MLX/torch,
inflate torch-portable).

Phase C extension 2026-05-30 (per `feedback_z6_v2_canonical_29650ep_mlx_local_full_run_landed_20260530.md`
Phase C BLOCKED reactivation criterion #1 + lane
`lane_z6_v2_phase_c_canonical_inflate_format_extension_20260530`):

* Replaces legacy PNG-per-frame output with canonical contest ``.raw`` uint8
  output (1200 frames at 874×1164×3 = 3,662,409,600 bytes per video) via the
  canonical helper ``tac.substrates._shared.inflate_runtime.write_rgb_pair_to_raw``.
* Iterates pairs from the archive (canonical full = 600 pairs → 1200 frames;
  truncated MLX-LOCAL smoke = N pairs → 2N frames, advisory-only with
  ``allow_partial_frame_count=True``).
* Applies Catalog #367 fail-closed check on the written ``.raw`` byte count:
  any production-mode dispatch (``allow_partial_frame_count=False``) that
  does NOT produce exactly ``CONTEST_RAW_BYTES`` bytes raises ``AssertionError``
  with the canonical "WRONG-SIZE" message contest_auth_eval recognizes.
* Supports BOTH schema v1 (fp16+pickle+brotli q=9; HISTORICAL_PROVENANCE per
  Catalog #110/#113) and schema v2 (INT8+fp16scales+brotli q=11; canonical
  leaderboard binding-depth L21+L29+L32; ~45% archive reduction).
* Catalog #205 canonical ``select_inflate_device`` routing (CPU/CUDA via
  ``PACT_INFLATE_DEVICE`` env var; MPS structurally refused per CLAUDE.md
  "MPS auth eval is NOISE").

Per CLAUDE.md "Complexity + LOC + boundaries UNCONSTRAINED within contest
compliance" standing directive: this substrate-engineering inflate exceeds
the ≤200 LOC HNeRV parity L4 BOLT-ON budget because Z6-v2 is
``lane_class=substrate_engineering`` per the canonical opt-out; the
~30-second-reviewable operator-facing surface is the canonical entry point
``main_cli`` + ``inflate_one_video`` signatures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    CAMERA_HW,
    select_inflate_device as _shared_select_inflate_device,
    write_rgb_pair_to_raw,
)

from .architecture import Z6V2Config, Z6V2Substrate
from .archive import parse_archive

# Canonical contest output contract per Catalog #146 + #367 fail-closed.
CONTEST_OUT_H: int = CAMERA_HW[0]  # 874
CONTEST_OUT_W: int = CAMERA_HW[1]  # 1164
CONTEST_NUM_FRAMES: int = 1200
CONTEST_RAW_BYTES: int = CONTEST_OUT_W * CONTEST_OUT_H * CONTEST_NUM_FRAMES * 3
assert CONTEST_RAW_BYTES == 3_662_409_600, (
    f"CONTEST_RAW_BYTES invariant: expected 3,662,409,600, "
    f"got {CONTEST_RAW_BYTES} (Catalog #367 raw-byte contract)"
)


def select_inflate_device() -> torch.device:
    """Canonical Catalog #205 inflate-device selector (CPU/CUDA only; no MPS).

    Mirrors the byte-for-byte canonical helper at
    ``tac.substrates._shared.inflate_runtime.select_inflate_device``; wraps
    the string return in ``torch.device`` per the local-helper canonical
    pattern (sister of every other ``submissions/*/inflate.py``).
    """
    return torch.device(_shared_select_inflate_device())


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
    allow_partial_frame_count: bool = False,
) -> int:
    """Inflate one Z6V2 archive into one contest ``.raw`` file (Phase C canonical).

    Args:
        archive_bytes: Z6V2 archive bytes (schema v1 or v2).
        output_raw_path: Path to write the contest ``.raw`` file. Parent dir
            created if needed.
        device: Override device string ("cpu"|"cuda"|None). When None, routes
            through ``select_inflate_device`` (PACT_INFLATE_DEVICE-aware).
        allow_partial_frame_count: When False (canonical contest mode), refuses
            archives whose pair count cannot produce exactly
            ``CONTEST_NUM_FRAMES`` frames at the canonical contest output
            resolution. When True (MLX-LOCAL smoke mode), writes whatever
            frame count the archive supports + skips the Catalog #367
            fail-closed byte check. Default False per contest-canonical
            invariant.

    Returns:
        Number of frames written.

    Raises:
        AssertionError: when ``allow_partial_frame_count=False`` and the
            written raw bytes != ``CONTEST_RAW_BYTES`` (Catalog #367).
        ValueError: archive parse errors, num_pairs mismatch, scorer device
            errors.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    num_archive_pairs = int(arc.latents.shape[0])

    # Sanity check: in canonical contest mode, the archive MUST have exactly
    # CONTEST_NUM_FRAMES // 2 = 600 pairs. allow_partial_frame_count=True
    # opts into truncated-smoke mode (advisory-only; will not produce a
    # valid contest .raw byte count).
    expected_contest_pairs = CONTEST_NUM_FRAMES // 2  # 600
    if not allow_partial_frame_count and num_archive_pairs != expected_contest_pairs:
        raise ValueError(
            f"canonical contest mode requires {expected_contest_pairs} pairs "
            f"(CONTEST_NUM_FRAMES // 2 = 1200 // 2); archive has "
            f"{num_archive_pairs} pairs. Pass allow_partial_frame_count=True "
            f"for MLX-LOCAL truncated-smoke advisory-only mode."
        )

    cfg = Z6V2Config(
        latent_dim=int(arc.latents.shape[1]),
        ego_dim=int(arc.ego_vecs.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        rao_ballard_level_boundary=int(meta.get("rao_ballard_level_boundary", 3)),
        film_generator_depth=int(meta.get("film_generator_depth", 3)),
        film_hidden_width=int(meta.get("film_hidden_width", 80)),
        cooperative_receiver_beta=float(meta.get("cooperative_receiver_beta", 0.5)),
        num_pairs=num_archive_pairs,
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )

    # Canonical device selection per Catalog #205. Honor explicit override
    # for tests / advisory smokes; otherwise use canonical helper.
    if device is not None:
        device_t = torch.device(device)
    else:
        device_t = select_inflate_device()

    model = Z6V2Substrate(cfg).to(device_t).eval()
    # strict=False because v2 emits fp16 weights into a fp32-default model
    # (canonical L29 fp16-scales pattern; PyTorch handles the dtype cast).
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device=device_t, dtype=model.latents.dtype)
        )
        model.ego_vecs.copy_(
            arc.ego_vecs.to(device=device_t, dtype=model.ego_vecs.dtype)
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.inference_mode(), open(output_raw_path, "wb") as fh:
        for pair_idx in range(num_archive_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device_t, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            # rgb_0, rgb_1 are shape (1, 3, H, W) at scorer resolution
            # (output_height x output_width = 384 x 512). The canonical helper
            # bicubic-upsamples to CAMERA_HW = (874, 1164) per Catalog #146.
            frames_written += write_rgb_pair_to_raw(
                fh, rgb_0, rgb_1, input_range="unit", resize_mode="bicubic",
            )

    written_bytes = output_raw_path.stat().st_size

    # Catalog #367 raw-byte fail-closed check.
    if not allow_partial_frame_count:
        if written_bytes != CONTEST_RAW_BYTES:
            raise AssertionError(
                f"[inflate] WRONG-SIZE .raw file(s): "
                f"{output_raw_path.name}={written_bytes}B "
                f"(expected {CONTEST_RAW_BYTES}B). Each must be "
                f"{CONTEST_RAW_BYTES:,} bytes "
                f"({CONTEST_OUT_W}x{CONTEST_OUT_H}x{CONTEST_NUM_FRAMES}x3). "
                f"Likely truncated mid-decode."
            )
        if frames_written != CONTEST_NUM_FRAMES:
            raise AssertionError(
                f"[inflate] frame count {frames_written} != expected "
                f"{CONTEST_NUM_FRAMES} (Catalog #367 contract)"
            )

    return frames_written


def _raw_output_path(output_dir: Path, video_name: str) -> Path:
    """Torch contest-safe raw output path for one relative file-list entry.

    Per CLAUDE.md "Operator gates must be wired and used" + sister
    ``tac.substrates._shared.numpy_portable_inflate._raw_output_path_numpy``:
    refuses absolute paths, ``..`` traversal, empty names, ``//`` double-slash,
    and any target that escapes the output directory.
    """
    raw = str(video_name).replace("\\", "/").strip()
    rel = Path(raw)
    if (
        not raw
        or "//" in raw
        or rel.is_absolute()
        or any(part in {"", ".."} for part in rel.parts)
    ):
        raise ValueError(f"unsafe file_list video name for raw output: {video_name!r}")
    root = output_dir.resolve(strict=False)
    target = (output_dir / rel.with_suffix(".raw")).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"file_list video name escapes output directory: {video_name!r}"
        ) from exc
    return target


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on missing/ambiguous input.

    Per Catalog #146 + sister z5 inflate pattern: the contest archive dir
    contains exactly one member at ``0.bin`` OR ``x`` (legacy fallback).
    """
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
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>`` (Catalog #146)."""
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
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(
            archive_bytes,
            _raw_output_path(output_dir, name),
            # allow_partial_frame_count defaults False = contest-canonical
            # Catalog #367 fail-closed; explicit override required for
            # MLX-LOCAL truncated-smoke advisory mode.
        )
    return 0


__all__ = [
    "CAMERA_HW",
    "CONTEST_NUM_FRAMES",
    "CONTEST_OUT_H",
    "CONTEST_OUT_W",
    "CONTEST_RAW_BYTES",
    "_raw_output_path",
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
    "select_inflate_device",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
