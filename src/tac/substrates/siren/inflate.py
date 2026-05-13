"""siren inflate runtime — contest raw-output contract.

Contest-runtime image of the substrate. ``submissions/siren/inflate.py`` will
be a one-line passthrough to ``main_cli`` at packet-build time. Forward path:

1. Read archive bytes for the requested archive_dir.
2. ``parse_archive(bytes)`` -> (mlp_sd, meta, hparams).
3. Build the substrate from header + ``meta`` (deterministic, no training).
4. Load state_dict.
5. For each pair index, evaluate MLP at all (x, y, t) coordinates -> RGB pair;
   append to one contest ``.raw`` tensor file.

L4 budget: <= 100 LOC, <= 2 external deps (torch, brotli; numpy is torch transitive).
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

from .architecture import SirenConfig, SirenSubstrate
from .archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    meta = arc.meta
    render_device = select_inflate_device(device)

    cfg = SirenConfig(
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
        first_omega=float(meta["first_omega"]),
        hidden_omega=float(meta["hidden_omega"]),
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
        coord_dim=int(meta["coord_dim"]),
        output_dim=int(meta["output_dim"]),
        activation_family=str(meta.get("activation_family", "siren")),
        wire_scale=float(meta.get("wire_scale", 1.0)),
        bacon_bandwidth_scale=float(meta.get("bacon_bandwidth_scale", 1.0)),
    )

    model = SirenSubstrate(cfg).to(render_device).eval()
    incompat = model.load_state_dict(arc.decoder_state_dict, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing - {"_spatial_coords"} or unexpected:
        raise RuntimeError(
            "siren archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(cfg.num_pairs):
            idx_tensor = torch.tensor([pair_idx], device=render_device, dtype=torch.long)
            rgb_0, rgb_1 = model(idx_tensor)
            frames_written += write_rgb_pair_to_raw(fh, rgb_0, rgb_1, input_range="unit")
    return frames_written


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
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    device = select_inflate_device()
    for fname in file_list:
        if not fname.strip():
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, fname), device=device)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
