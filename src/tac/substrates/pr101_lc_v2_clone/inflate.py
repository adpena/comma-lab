# SPDX-License-Identifier: MIT
"""pr101_lc_v2_clone inflate runtime — PR101-mirror, <= 200 LOC waiver.

This file is the contest-runtime image of the substrate. It is imported by
``submissions/pr101_lc_v2_clone/inflate.py`` (one-line passthrough) at
packet-build time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``(decoder_state_dict, latents, meta)``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. ``model.load_state_dict(decoder_state_dict)``; latents tensor in hand.
5. For each pair index i in [0, num_pairs) in batches of 16: render
   ``(B, 2, 3, H, W)`` RGB and emit per-frame uint8.

L4 budget waiver: <= 200 LOC because PR101-fidelity inflate carries the
multi-stream brotli + per-tensor byte-map + storage perm primitives in the
archive parser. This file itself is the THIN runtime (just builds model
+ writes frames). The byte-grammar lives in ``archive.py``.

CLAUDE.md compliance:
* CPU default for ``[contest-CPU]`` axis; CUDA opt-in via kwarg.
* No scorer load (strict-scorer-rule).
* No /tmp paths.
* Reviewable in 30 seconds (this file is ~60 LOC; the grammar work is
  factored into ``archive.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.codec.pr98_channel_balance_zero_byte_bolt_on import (
    Pr98ChannelBalanceConfig,
    apply_pr98_channel_balance_to_decoded_pair_torch,
)

from .architecture import Pr101LcV2CloneConfig, Pr101LcV2CloneSubstrate
from .archive import parse_archive

# PR101 source line 16: CAMERA_H, CAMERA_W = 874, 1164
_CAMERA_H = 874
_CAMERA_W = 1164
_PR98_L28_PR101_CONFIG = Pr98ChannelBalanceConfig(substrate_id="pr101_lc_v2_clone")


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> int:
    """Inflate one archive's bytes into a single raw uint8 RGB stream.

    Mirrors PR101 source line 19-65 (``inflate`` function) but writes one
    PNG per frame instead of a single .raw blob, matching the contest
    contract (``output_dir/<base>/<frame_idx>.png``).

    Args:
        archive_bytes: raw bytes of the ``0.bin`` member.
        output_dir: where to write per-frame PNGs.
        device: ``"cpu"`` (default, contest-leaderboard CPU axis) or ``"cuda"``.

    Returns:
        Number of frames written.
    """
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = Pr101LcV2CloneConfig(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta.get("base_channels", 36)),
        base_h=int(meta.get("base_h", 6)),
        base_w=int(meta.get("base_w", 8)),
        num_upsample_blocks=int(meta.get("num_upsample_blocks", 6)),
        num_pairs=int(meta["num_pairs"]),
        output_height=int(meta.get("output_height", 384)),
        output_width=int(meta.get("output_width", 512)),
    )

    model = Pr101LcV2CloneSubstrate(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)
    latents = arc.latents.to(device).float()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import PIL to keep this module's import light.
    from PIL import Image  # type: ignore[import-not-found]

    n = 0
    with torch.inference_mode():
        for i in range(0, cfg.num_pairs, 16):
            j = min(i + 16, cfg.num_pairs)
            batch = j - i
            decoded = model(latents[i:j])  # (batch, 2, 3, H, W) in [0, 255]
            flat = decoded.reshape(batch * 2, 3, cfg.output_height, cfg.output_width)
            up = F.interpolate(
                flat,
                size=(_CAMERA_H, _CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            up_pair = up.view(batch, 2, 3, _CAMERA_H, _CAMERA_W)
            apply_pr98_channel_balance_to_decoded_pair_torch(
                up_pair,
                _PR98_L28_PR101_CONFIG,
            )
            frames = (
                up_pair.reshape(batch * 2, 3, _CAMERA_H, _CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            for k in range(batch * 2):
                Image.fromarray(frames[k]).save(output_dir / f"{n + k}.png")
            n += batch * 2
    return n


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest 3-positional-arg inflate.sh contract per Catalog #146.
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
    for fname in file_list:
        base = Path(fname).stem  # "0" from "0.mkv"
        archive_bytes = (archive_dir / "0.bin").read_bytes()
        inflate_one_video(archive_bytes, output_dir / base, device="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
