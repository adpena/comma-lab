"""SABOR inflate runtime — ≤ 200 LOC (substrate_engineering exception).

Contest-runtime image of the substrate. ``submissions/sabor.../inflate.py``
will be a one-line passthrough to ``main_cli`` at packet-build time. Forward
path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> SaborArchive (boundary mask + boundary RGB
   + class means + segnet argmax + decoder state).
3. Build the substrate from header + ``meta`` (deterministic, no training).
4. Load state_dict; install class_means from uint8 (divide by 255).
5. For each pair index, decode (rgb_0, rgb_1); write contest ``.raw`` frames.

L4 budget: substrate-engineering exception per HNeRV parity discipline L7.
Runtime closure: torch + brotli (numpy is torch transitive).

NO scorer load at inflate time per CLAUDE.md strict-scorer-rule. The
segnet_argmax was captured at GT-time during training and stored in the
archive; inflate-time deterministically reconstructs the interior pixels
from class_means + segnet_argmax + per-pair bias, then overlays the
boundary RGB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .architecture import SaborBoundaryOnlyConfig, SaborBoundaryOnlyRenderer
from .archive import SaborArchive, parse_archive

try:  # pragma: no cover — vendored at packet-build time
    from tac.substrates._shared.inflate_runtime import (
        CAMERA_HW,
        write_rgb_pair_to_raw,
    )
except Exception:  # pragma: no cover
    CAMERA_HW = (874, 1164)
    write_rgb_pair_to_raw = None  # type: ignore[assignment]


def _reconstruct_pair(
    model: SaborBoundaryOnlyRenderer,
    arc: SaborArchive,
    pair_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically reconstruct one pair (rgb_0, rgb_1) at inflate time.

    Returns ``(rgb_0, rgb_1)`` each ``(1, 3, H, W)`` float in [0, 1].
    """
    cfg = model.cfg
    h, w = cfg.output_height, cfg.output_width

    # Resolve frame slices: pair pi -> frames (2*pi, 2*pi+1).
    f0, f1 = 2 * pair_index, 2 * pair_index + 1
    mask = arc.boundary_mask[f0 : f1 + 1].to(device)  # (2, H, W) bool
    seg = arc.segnet_argmax[f0 : f1 + 1].to(device).long()  # (2, H, W)

    # The flat boundary RGB is indexed across the entire archive. Compute the
    # per-pair offset by cumsum over earlier frame mask totals.
    prior_count = int(arc.boundary_mask[:f0].sum().item())
    pair_count_0 = int(arc.boundary_mask[f0].sum().item())
    pair_count_1 = int(arc.boundary_mask[f1].sum().item())
    brgb_0_flat = arc.boundary_rgb_flat[prior_count : prior_count + pair_count_0]
    brgb_1_flat = arc.boundary_rgb_flat[
        prior_count + pair_count_0 : prior_count + pair_count_0 + pair_count_1
    ]

    # Scatter boundary RGB back to (3, H, W) shape per frame.
    boundary_rgb_pair = torch.zeros(2, 3, h, w, dtype=torch.float32, device=device)
    for fi, (mask_f, brgb_flat) in enumerate(((mask[0], brgb_0_flat), (mask[1], brgb_1_flat))):
        if brgb_flat.shape[0] == 0:
            continue
        rgb_float = brgb_flat.to(device=device, dtype=torch.float32) / 255.0
        m = mask_f  # (H, W)
        for c in range(3):
            channel_plane = boundary_rgb_pair[fi, c]
            channel_plane[m] = rgb_float[:, c]

    boundary_rgb = boundary_rgb_pair.unsqueeze(0)  # (1, 2, 3, H, W)
    boundary_mask = mask.unsqueeze(0)  # (1, 2, H, W)
    segnet_argmax = seg.unsqueeze(0)  # (1, 2, H, W)
    pair_indices = torch.tensor([pair_index], device=device, dtype=torch.long)

    rgb_0, rgb_1 = model(pair_indices, boundary_mask, boundary_rgb, segnet_argmax)
    return rgb_0, rgb_1


def inflate_one_video(
    archive_bytes: bytes,
    dst_raw: Path,
    *,
    device_str: str = "cpu",
) -> int:
    """Inflate one archive's bytes into a contest ``.raw`` frame stream.

    Args:
        archive_bytes: raw SBO1 bytes.
        dst_raw: destination ``.raw`` path (one file per video).
        device_str: ``"cuda"`` or ``"cpu"`` (no MPS).

    Returns:
        Number of frames written (always ``num_pairs * 2``).
    """
    arc = parse_archive(archive_bytes)
    device = torch.device(device_str if device_str != "mps" else "cpu")

    cfg = SaborBoundaryOnlyConfig(
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
        num_seg_classes=arc.num_seg_classes,
        edge_threshold=arc.edge_threshold,
        refinement_hidden=arc.refinement_hidden,
        refinement_blocks=arc.refinement_blocks,
        embedding_dim=arc.embedding_dim,
        bias_dim=arc.bias_dim,
    )
    model = SaborBoundaryOnlyRenderer(cfg).to(device).eval()
    model.load_state_dict(arc.decoder_state_dict, strict=False)

    # Install class_means from uint8.
    with torch.no_grad():
        cm = arc.class_means.to(device=device, dtype=torch.float32) / 255.0
        model.class_means.copy_(cm)

    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    with dst_raw.open("wb") as fh, torch.inference_mode():
        for pi in range(cfg.num_pairs):
            rgb_0, rgb_1 = _reconstruct_pair(model, arc, pi, device)
            if write_rgb_pair_to_raw is not None:
                # The canonical raw writer upsamples to CAMERA_HW for us.
                n_frames += write_rgb_pair_to_raw(
                    fh, rgb_0, rgb_1, input_range="unit", resize_mode="bicubic"
                )
            else:  # pragma: no cover — local fallback only
                import torch.nn.functional as F

                pair = torch.cat([rgb_0, rgb_1], dim=0) * 255.0
                if tuple(pair.shape[-2:]) != CAMERA_HW:
                    pair = F.interpolate(
                        pair, size=CAMERA_HW, mode="bicubic", align_corners=False
                    )
                frames_u8 = (
                    pair.clamp(0.0, 255.0)
                    .permute(0, 2, 3, 1)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                fh.write(frames_u8.tobytes(order="C"))
                n_frames += int(frames_u8.shape[0])
    return n_frames


def main_cli() -> int:  # pragma: no cover — CLI smoke
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve archive bytes (x or 0.bin per A1 convention).
    candidates = [archive_dir / "0.bin", archive_dir / "x"]
    src_bin: Path | None = None
    for cand in candidates:
        if cand.is_file():
            src_bin = cand
            break
    if src_bin is None:
        print(f"FATAL: no archive bytes in {archive_dir}", file=sys.stderr)
        return 3
    archive_bytes = src_bin.read_bytes()

    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        base = Path(line).stem
        dst = output_dir / f"{base}.raw"
        inflate_one_video(archive_bytes, dst, device_str="cpu")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_cli())
