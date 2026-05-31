# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich inflate runtime — canonical contest format per Catalog #146.

Per Catalog #146 contest-compliant inflate runtime template + Catalog #367
raw-byte fail-closed + Catalog #205 canonical device selection + canonical
leaderboard binding-depth L5 (full RGB renderer at camera resolution).

Z4 distinguishing-feature consumption per Catalog #220 + #272 + #309
(horizon_class=frontier_pursuit substrate-engineering):

* The Atick-Redlich decorrelator (W_AR + b_AR) is LOADED from the archive's
  decorrelator_blob section and APPLIED at forward time via
  ``Z4AtickRedlichSubstrate.decorrelator.proj.weight/bias.copy_()``. This is
  the operational consumption that satisfies Catalog #272 distinguishing-
  feature byte-mutation smoke contract (mutate any byte of the
  decorrelator_blob ⇒ rendered RGB frames change byte-for-byte at the
  per-pixel surface).
* Forward returns ``(rgb_0, rgb_1)`` each ``(B, 3, 384, 512)`` in [0, 255];
  the canonical ``write_rgb_pair_to_raw`` helper bicubic-upsamples to the
  contest camera resolution ``(874, 1164)``.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this inflate is COMPLETE at L1 SCAFFOLD — it operationally
consumes the distinguishing decorrelator bytes via the canonical loaded-
weight + decorrelator-projection mechanism. ``research_only=true`` until
L2 paired-CUDA promotion lands.

Per CLAUDE.md "Complexity + LOC + boundaries UNCONSTRAINED within contest
compliance" standing directive: this substrate-engineering inflate may
exceed the ≤200 LOC HNeRV parity L4 BOLT-ON budget because Z4 is
``lane_class=substrate_engineering`` per the canonical opt-out. The
~30-second-reviewable operator-facing surface is the canonical entry
points ``main_cli`` + ``inflate_one_video`` + ``select_inflate_device``.
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

from .architecture import Z4AtickRedlichConfig, Z4AtickRedlichSubstrate
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

    Honors ``PACT_INFLATE_DEVICE`` env var (auto/cpu/cuda); MPS structurally
    refused per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. Mirrors
    the byte-for-byte canonical helper at
    ``tac.substrates._shared.inflate_runtime.select_inflate_device``.
    """
    return torch.device(_shared_select_inflate_device())


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
    allow_partial_frame_count: bool = False,
) -> int:
    """Inflate one Z4ATR archive into one contest ``.raw`` file.

    Args:
        archive_bytes: Z4ATR archive bytes (schema v1).
        output_raw_path: Path to write the contest ``.raw`` file. Parent
            dir is created if needed.
        device: Override device string ("cpu"|"cuda"|None). When None,
            routes through ``select_inflate_device``.
        allow_partial_frame_count: When False (canonical contest mode),
            refuses archives whose pair count cannot produce exactly
            ``CONTEST_NUM_FRAMES`` frames + applies Catalog #367 fail-
            closed byte check. When True (MLX-LOCAL smoke mode), writes
            whatever frame count the archive supports.

    Returns:
        Number of frames written.

    Raises:
        AssertionError: when ``allow_partial_frame_count=False`` and the
            written raw bytes != ``CONTEST_RAW_BYTES`` (Catalog #367).
        ValueError: archive parse errors, num_pairs mismatch, device errors.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    num_archive_pairs = int(arc.latents.shape[0])

    expected_contest_pairs = CONTEST_NUM_FRAMES // 2  # 600
    if not allow_partial_frame_count and num_archive_pairs != expected_contest_pairs:
        raise ValueError(
            f"canonical contest mode requires {expected_contest_pairs} pairs "
            f"(CONTEST_NUM_FRAMES // 2 = 1200 // 2); archive has "
            f"{num_archive_pairs} pairs. Pass allow_partial_frame_count=True "
            f"for MLX-LOCAL truncated-smoke advisory-only mode."
        )

    cfg = Z4AtickRedlichConfig(
        num_pairs=num_archive_pairs,
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        sin_frequency=float(meta["sin_frequency"]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        apply_decorrelator=bool(meta.get("apply_decorrelator", True)),
        cooperative_receiver_beta=float(
            meta.get("cooperative_receiver_beta", 0.5)
        ),
    )

    if device is not None:
        device_t = torch.device(device)
    else:
        device_t = select_inflate_device()

    model = Z4AtickRedlichSubstrate(cfg).to(device_t).eval()
    # strict=False because state_dict may carry fp16 weights into the fp32
    # default model (PyTorch handles the dtype cast transparently).
    model.load_state_dict(arc.decoder_state_dict, strict=False)

    # CANONICAL DISTINGUISHING-FEATURE CONSUMPTION per Catalog #272:
    # Load Atick-Redlich W_AR + b_AR from the decorrelator_blob section
    # into the live decorrelator submodule. This is the operational
    # mechanism per Catalog #220 — the decorrelator bytes that the
    # archive ships ARE consumed at forward time.
    with torch.no_grad():
        model.decorrelator.proj.weight.copy_(
            arc.decorrelator_weight.to(
                device=device_t, dtype=model.decorrelator.proj.weight.dtype
            )
        )
        model.decorrelator.proj.bias.copy_(
            arc.decorrelator_bias.to(
                device=device_t, dtype=model.decorrelator.proj.bias.dtype
            )
        )
        model.latents.copy_(
            arc.latents.to(device=device_t, dtype=model.latents.dtype)
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.inference_mode(), open(output_raw_path, "wb") as fh:
        for pair_idx in range(num_archive_pairs):
            idx_tensor = torch.tensor(
                [pair_idx], device=device_t, dtype=torch.long
            )
            rgb_0, rgb_1 = model(idx_tensor)
            # rgb_0, rgb_1 are (1, 3, H, W) at scorer resolution (384, 512);
            # the canonical helper bicubic-upsamples to CAMERA_HW (874,
            # 1164) per Catalog #146 + writes uint8 raw.
            frames_written += write_rgb_pair_to_raw(
                fh,
                rgb_0,
                rgb_1,
                input_range="byte",  # forward emits [0, 255]
                resize_mode="bicubic",
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
    """Torch contest-safe raw output path for one file-list entry.

    Per CLAUDE.md "Operator gates must be wired and used" + sister
    ``z6_v2_cargo_cult_unwind/inflate.py::_raw_output_path``: refuses
    absolute paths, ``..`` traversal, empty names, ``//`` double-slash,
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
        raise ValueError(
            f"unsafe file_list video name for raw output: {video_name!r}"
        )
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
    """Read the single contest archive member.

    Per Catalog #146 + sister z6_v2 + z5 inflate pattern: the contest
    archive dir contains exactly one member at ``0.bin`` OR ``x`` (legacy
    fallback). Fail-closed on missing/ambiguous input.
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
