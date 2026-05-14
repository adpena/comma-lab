"""C1 world-model + foveation inflate runtime -- contest raw-output contract.

Loads the C1WMFV1 archive, rehydrates the world-model + decoder + foveation
state, unrolls the world-model for N_FRAMES steps, decodes RGB per frame
plus residual surprise add-back, and writes one raw-output ``.raw`` file per
contest video.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is <= 200 for
substrate-engineering lanes (full world-model unroll + foveated decode +
residual surprise add-back).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
    write_rgb_pair_to_raw,
)
from tac.substrates.c1_world_model_foveation.archive import (
    WorldModelFoveationArchive,
    deserialize_state_dict,
    parse_archive,
)
from tac.substrates.c1_world_model_foveation.architecture import (
    FoveatedDecoderModule,
    FoveationMapModule,
    FoveationStrategy,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelModule,
    WorldModelRecurrenceMode,
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


def _build_substrate_from_archive(
    arc: WorldModelFoveationArchive,
    device: str,
) -> tuple[WorldModelModule, FoveatedDecoderModule, FoveationMapModule, torch.Tensor]:
    """Build the world-model + decoder + foveation modules from archive sections."""
    recurrence_mode = (
        WorldModelRecurrenceMode.GRU
        if arc.recurrence_mode == 0
        else WorldModelRecurrenceMode.LSTM
        if arc.recurrence_mode == 1
        else WorldModelRecurrenceMode.TRANSFORMER
    )
    foveation_strategy = (
        FoveationStrategy.UNIFORM
        if arc.foveation_strategy == 0
        else FoveationStrategy.EGO_MOTION_RADIAL
        if arc.foveation_strategy == 1
        else FoveationStrategy.LEARNED_PER_PIXEL
    )
    wm_cfg = WorldModelConfig(
        recurrence_mode=recurrence_mode,
        latent_dim=arc.latent_dim,
        hidden_dim=arc.latent_dim,
    )
    cfg = WorldModelFoveationConfig(
        world_model_cfg=wm_cfg,
        foveation_strategy=foveation_strategy,
        output_height=arc.output_h,
        output_width=arc.output_w,
        num_pairs=arc.num_pairs,
    )
    world_model = WorldModelModule(wm_cfg).to(device)
    decoder = FoveatedDecoderModule(cfg).to(device)
    foveation = FoveationMapModule(cfg).to(device)

    wm_state = deserialize_state_dict(arc.world_model_blob)
    decoder_state = deserialize_state_dict(arc.decoder_blob)
    z_init_state = deserialize_state_dict(arc.z_init_blob)

    world_model.load_state_dict(wm_state, strict=False)
    decoder.load_state_dict(decoder_state, strict=False)
    if foveation_strategy == FoveationStrategy.LEARNED_PER_PIXEL and arc.foveation_meta_blob:
        try:
            fov_state = deserialize_state_dict(arc.foveation_meta_blob)
            foveation.load_state_dict(fov_state, strict=False)
        except Exception:
            # If foveation_meta is JSON (UNIFORM/EGO_MOTION_RADIAL) we already
            # built the no-param module above; nothing to load.
            pass

    z_init = z_init_state["z_init"].to(device)
    return world_model, decoder, foveation, z_init


def _decode_residual_blob(blob: bytes, num_frames: int) -> torch.Tensor:
    """Decode the per-frame residual surprise blob (int8 quantized, brotli'd).

    V1 contract: decompressed payload is int8 residuals; total size is
    ``num_frames * 3 * h * w``. Coarse (h, w) is inferred from the byte
    count assuming a square-ish grid. Returns a zero tensor on empty blob
    (smoke / no-residual run).
    """
    if not blob:
        return torch.zeros(num_frames, 3, 1, 1)
    import brotli  # type: ignore[import-not-found]

    raw = brotli.decompress(blob)
    n_bytes_per_frame = len(raw) // max(num_frames, 1)
    if n_bytes_per_frame < 3:
        return torch.zeros(num_frames, 3, 1, 1)
    n_pixels_per_frame = n_bytes_per_frame // 3
    h = max(1, int(n_pixels_per_frame ** 0.5))
    w = max(1, n_pixels_per_frame // h)
    total_used = num_frames * 3 * h * w
    if total_used > len(raw):
        return torch.zeros(num_frames, 3, 1, 1)
    arr = torch.frombuffer(raw[:total_used], dtype=torch.int8)
    return arr.reshape(num_frames, 3, h, w).to(torch.float32) / 127.0


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one C1WMFV1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)

    world_model, decoder, foveation, z_init = _build_substrate_from_archive(
        arc, render_device
    )
    n_frames = 2 * arc.num_pairs
    residual = _decode_residual_blob(arc.residual_blob, n_frames).to(render_device)

    import torch.nn.functional as F

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        latents = world_model.unroll(z_init, n_frames)  # (T, latent_dim)
        for pair_idx in range(arc.num_pairs):
            z_0 = latents[2 * pair_idx : 2 * pair_idx + 1]
            z_1 = latents[2 * pair_idx + 1 : 2 * pair_idx + 2]
            rgb_0 = decoder.decode(z_0)
            rgb_1 = decoder.decode(z_1)
            if residual.shape[-1] > 1:
                size = (arc.output_h, arc.output_w)
                res_0 = F.interpolate(residual[2 * pair_idx : 2 * pair_idx + 1], size=size, mode="bilinear", align_corners=False)
                res_1 = F.interpolate(residual[2 * pair_idx + 1 : 2 * pair_idx + 2], size=size, mode="bilinear", align_corners=False)
                rgb_0 = (rgb_0 + res_0).clamp(0.0, 1.0)
                rgb_1 = (rgb_1 + res_1).clamp(0.0, 1.0)
            _ = foveation.map(z_0)  # research instrumentation only
            frames_written += write_rgb_pair_to_raw(fh, rgb_0, rgb_1, input_range="unit")
    return frames_written


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``."""
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
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(
            archive_bytes, raw_output_path(output_dir, name), device=device
        )
    return 0


__all__ = [
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover -- CLI smoke
    sys.exit(main_cli())
