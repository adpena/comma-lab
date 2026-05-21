# SPDX-License-Identifier: MIT
"""Pre-trained driving prior inflate runtime — contest raw-output contract.

Loads the DP1 archive, dequantizes the frozen codebook, loads the
contest-overfit renderer, decodes per-pair int8 residual, and renders RGB
pairs for every contest pair index. Writes one ``.raw`` file per requested
video matching the contest's ``(1200, 874, 1164, 3)`` uint8 layout.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device per Catalog #1 (CPU/CUDA only).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes. This file stays under that ceiling: full codebook
parse + renderer load + per-pair residual decode + render is ~150 LOC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
    write_rgb_pair_to_raw,
)
from tac.substrates.pretrained_driving_prior.archive import (
    DrivingPriorArchive,
    parse_archive,
)
# WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — procedural-codebook
# variant detection sister helper per Catalog #328 inflate.py LOC budget
# (helper lives in a separate module so this file stays close to budget).
# When meta_blob declares procedural_codebook_variant_active=True the
# canonical 4-array DashcamCodebook is re-derived from the operator-supplied
# seed via tac.procedural_codebook_generator.derive_codebook_from_seed.
from tac.substrates.pretrained_driving_prior.procedural_codebook_inflate import (
    parse_archive_procedural_aware,
)


def _build_renderer_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    output_height: int,
    output_width: int,
    device: str,
) -> torch.nn.Module:
    """Reconstruct the small contest-overfit renderer from its FP16 state_dict.

    The renderer is a tiny coordinate-MLP (per-pair index → per-pixel RGB).
    The scaffold renderer mirrors the architecture used by the trainer; if
    the state_dict's shapes drift from the canonical architecture the load
    raises and the inflate exits non-zero.

    For the scaffold L0 path the renderer is a 2-layer MLP:
        input  : (B, 4)        per-pair (x, y, t_pair, foveation)
        hidden : (B, 64)
        output : (B, 3)        RGB

    Production training will likely use a richer renderer; the architecture
    stays in :class:`DrivingPriorRenderer` (loaded via ``state_dict`` keys).
    """
    from tac.substrates.pretrained_driving_prior.architecture import (
        DrivingPriorRenderer,
        DrivingPriorRendererConfig,
    )

    # Infer hidden dim + depth from the state_dict shapes (net.<i>.linear.weight).
    sine_layer_indices = sorted(
        {
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("net.") and ".linear.weight" in k
        }
    )
    num_sine_layers = len(sine_layer_indices) if sine_layer_indices else 3
    # hidden_dim from the first sine layer's out_features.
    first_key = f"net.{sine_layer_indices[0]}.linear.weight" if sine_layer_indices else None
    hidden_dim = int(state_dict[first_key].shape[0]) if first_key is not None and first_key in state_dict else 64

    cfg = DrivingPriorRendererConfig(
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_sine_layers),
        output_height=output_height,
        output_width=output_width,
    )
    renderer = DrivingPriorRenderer(cfg).to(device).eval()
    incompat = renderer.load_state_dict(state_dict, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"DP1 renderer state_dict mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    return renderer


def _apply_per_pair_residual(
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    residual_bytes: bytes,
    pair_idx: int,
    per_pair_bytes: int,
    *,
    int8_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-pair int8-quantized residual as additive RGB correction.

    Reads ``per_pair_bytes`` bytes starting at ``pair_idx * per_pair_bytes``
    from the residual blob, dequantizes by ``int8_scale``, and adds a small
    correction to both rgb tensors. The correction is tiled across the
    image and scaled gently (max ~3 gray levels out of 255) to keep the
    contribution within proxy-auth-stable bounds.
    """

    start = pair_idx * per_pair_bytes
    pair_bytes = residual_bytes[start : start + per_pair_bytes]
    pair_int8 = np.frombuffer(pair_bytes, dtype=np.int8).astype(np.float32)
    residual_floats = torch.from_numpy(pair_int8).to(rgb_0.device) / int8_scale

    # Use the residual as a 3-channel low-frequency RGB delta.
    n_chunks = per_pair_bytes // 3
    if n_chunks < 1:
        return rgb_0, rgb_1
    chunk = residual_floats[: n_chunks * 3].view(n_chunks, 3).mean(dim=0)  # (3,)
    correction = chunk.view(1, 3, 1, 1)
    correction_full = correction.expand_as(rgb_0)
    rgb_0_c = (rgb_0 + 0.02 * correction_full).clamp(0.0, 1.0)
    rgb_1_c = (rgb_1 + 0.02 * correction_full).clamp(0.0, 1.0)
    return rgb_0_c, rgb_1_c


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one DP1 archive's bytes into one contest ``.raw`` file.

    Returns the number of frames written. The codebook is consumed by the
    deterministic soft-prior transform before residual application, so charged
    codebook bytes affect the inflated frames rather than serving as parse-only
    provenance.
    """
    # WAVE-3-DP1-DISPATCH-READY-EXTENSION 2026-05-20 — route through the
    # procedural-aware parser. For canonical archives this delegates to
    # parse_archive; for procedural variants it re-derives the codebook
    # from meta["procedural_codebook_seed_hex"] via
    # tac.procedural_codebook_generator.derive_codebook_from_seed.
    arc: DrivingPriorArchive = parse_archive_procedural_aware(archive_bytes)
    render_device = select_inflate_device(device)
    renderer = _build_renderer_from_state_dict(
        arc.renderer_state_dict,
        output_height=arc.output_height,
        output_width=arc.output_width,
        device=render_device,
    )
    from tac.substrates.pretrained_driving_prior.prior_application import (
        DashcamPriorLoss,
        PriorApplicationWeights,
    )

    int8_scale = float(arc.meta.get("residual_int8_scale", 64.0))
    prior_inflate_strength = float(arc.meta.get("prior_inflate_strength", 1.0))
    prior = DashcamPriorLoss(
        arc.codebook,
        PriorApplicationWeights(eval_resolution=(arc.output_height, arc.output_width)),
        device=render_device,
    ).to(render_device)

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(arc.num_pairs):
            rgb_0, rgb_1 = renderer.render_pair(pair_idx, arc.num_pairs)
            rgb_0 = prior.apply_soft_prior(
                rgb_0,
                strength=prior_inflate_strength,
            )
            rgb_1 = prior.apply_soft_prior(
                rgb_1,
                strength=prior_inflate_strength,
            )
            rgb_0_c, rgb_1_c = _apply_per_pair_residual(
                rgb_0,
                rgb_1,
                arc.per_pair_residual,
                pair_idx,
                arc.per_pair_bytes,
                int8_scale=int8_scale,
            )
            frames_written += write_rgb_pair_to_raw(
                fh, rgb_0_c, rgb_1_c, input_range="unit"
            )
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
    src_path = archive_dir / "0.bin"
    if not src_path.is_file():
        src_path = archive_dir / "x"
    archive_bytes = src_path.read_bytes()
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
    "inflate_one_video",
    "main_cli",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
