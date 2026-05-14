# SPDX-License-Identifier: MIT
"""self_compress_nn inflate runtime — <= 200 LOC waiver per HNeRV L4 FAIL (δ).

This file is the contest-runtime image of the δ substrate. It is imported by
``submissions/self_compress_nn/inflate.py`` (one-line passthrough) at packet-build
time. The whole forward path is:

1. Read the archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> ``SelfCompressNnArchive``.
3. Build the substrate from ``meta`` (no training; deterministic).
4. Reconstruct each quantized layer's weight by gathering codebook rows
   indexed by the per-element cluster_indices, then reshaping to the
   target layer shape.
5. Copy latents; for each pair index render (rgb_0, rgb_1); write
   ``output_dir/<base>/<frame_idx>.png``.

L4 budget: <= 200 LOC waiver (council §4.2 δ FAIL note: codebook decode +
per-layer reshape adds ~50 LOC over α's 80 LOC). Target ~170 LOC. <= 2
external deps: ``torch`` + ``brotli``. Catalog #146 contract
(<inflate.sh archive_dir output_dir file_list> 3 positional args).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .archive import parse_archive
from .architecture import SelfCompressNnConfig, SelfCompressNnSubstrate


def _reconstruct_layer_weights(
    codebook: torch.Tensor,
    layer_cluster_indices: dict[str, torch.Tensor],
    layer_meta: list[dict],
) -> dict[str, torch.Tensor]:
    """Decode codebook + cluster indices into per-layer dense weight tensors.

    For each layer entry: gather ``cluster_indices`` rows from ``codebook``
    (each row is D_v values), then reshape into the original weight shape.

    Returns ``{tensor_name: weight_tensor}`` ready for ``load_state_dict``.
    """
    out: dict[str, torch.Tensor] = {}
    for entry in layer_meta:
        name = entry["name"]
        target_shape = tuple(int(s) for s in entry["shape"])
        idx = layer_cluster_indices[name]
        gathered = codebook[idx]  # (numel_groups, D_v)
        # Flatten gather across D_v, then reshape to target
        flat = gathered.reshape(-1)
        if flat.numel() != int(torch.tensor(target_shape).prod().item()):
            raise ValueError(
                f"layer {name} reconstruction size mismatch: got {flat.numel()} "
                f"vs target shape {target_shape}"
            )
        out[name] = flat.reshape(target_shape)
    return out


def inflate_one_video(
    archive_bytes: bytes,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> None:
    """Inflate one archive's bytes into ``output_dir/<frame_idx>.png`` files."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta

    cfg = SelfCompressNnConfig(
        latent_dim=int(arc.latents.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        codebook_k=int(arc.codebook.shape[0]),
        codebook_dv=int(arc.codebook.shape[1]),
        codebook_ema_decay=float(meta.get("codebook_ema_decay", 0.99)),
        commit_loss_weight=float(meta.get("commit_loss_weight", 0.25)),
    )

    model = SelfCompressNnSubstrate(cfg).to(device).eval()

    # Restore codebook centroids in-place (the EMA buffers are not
    # archived — inference doesn't need them; only forward-time lookup).
    with torch.no_grad():
        model.codebook.codebook.copy_(arc.codebook.to(device=device))

    # Reconstruct each quantized layer's weight tensor and load into the
    # substrate.  load_state_dict with strict=False skips per-pair learnables
    # and the persistent codebook EMA buffers (we just set codebook directly).
    reconstructed = _reconstruct_layer_weights(
        arc.codebook.to(device=device),
        {k: v.to(device=device) for k, v in arc.layer_cluster_indices.items()},
        arc.layer_meta,
    )
    # also pull any non-quantized state from meta if present
    # (latent_embed weights, head biases, etc. — stored in meta JSON for SKETCH)
    extra_state = meta.get("extra_state", {})
    if isinstance(extra_state, dict):
        for k, v in extra_state.items():
            # v is a list (json-encoded tensor); reconstruct as fp32 tensor
            if isinstance(v, list):
                reconstructed[k] = torch.tensor(v, dtype=torch.float32, device=device)

    model.load_state_dict(reconstructed, strict=False)

    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import PIL inside the function to keep this module's import light
    from PIL import Image  # type: ignore[import-not-found]

    with torch.no_grad():
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1, _commit = model(idx_tensor)
            for off, rgb in ((0, rgb_0), (1, rgb_1)):
                frame_idx = 2 * pair_idx + off
                arr = (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0)
                arr = arr.round().clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(output_dir / f"{frame_idx}.png")


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Honors the contest's 3-positional-arg inflate.sh contract per Catalog #146.
    """
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
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
