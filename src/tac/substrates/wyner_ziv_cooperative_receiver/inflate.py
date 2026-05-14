"""Wyner-Ziv cooperative-receiver inflate runtime — contest raw-output contract.

Loads the WZ1 archive, rebuilds the renderer + side-info predictor, decodes
per-pair coset indices, and renders RGB pairs for every contest pair.
Writes one ``.raw`` file per requested video matching the contest's
``(1200, 874, 1164, 3)`` uint8 layout.

NO scorer code is imported per CLAUDE.md "Strict scorer rule —
non-negotiable" + Catalog #6. NO MPS device (Catalog #1; CPU/CUDA only).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes (this is a full renderer + side-info predictor
+ DISCUS coset disambiguation, not a thin codec).
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
from tac.substrates.wyner_ziv_cooperative_receiver.architecture import (
    WynerZivConfig,
    WynerZivSubstrate,
    disambiguate_coset,
)
from tac.substrates.wyner_ziv_cooperative_receiver.archive import (
    WynerZivArchive,
    parse_archive,
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
    arc: WynerZivArchive, *, device: str
) -> WynerZivSubstrate:
    """Rebuild the substrate from header + meta + load both state_dicts."""
    meta = arc.meta
    cfg = WynerZivConfig(
        hidden_dim=arc.hidden_dim,
        num_hidden_layers=arc.num_hidden_layers,
        side_info_hidden_dim=arc.side_info_hidden_dim,
        side_info_num_layers=arc.side_info_num_layers,
        coord_dim=int(meta.get("coord_dim", 3)),
        pose_dim=arc.pose_dim,
        coset_index_bits=arc.coset_index_bits,
        first_omega=float(meta.get("first_omega", 30.0)),
        hidden_omega=float(meta.get("hidden_omega", 1.0)),
        num_pairs=arc.num_pairs,
        output_height=arc.output_height,
        output_width=arc.output_width,
        coord_feature_freqs=int(meta.get("coord_feature_freqs", 4)),
        wyner_ziv_dither_std=float(meta.get("wyner_ziv_dither_std", 0.0)),
    )
    substrate = WynerZivSubstrate(cfg).to(device).eval()
    # Load renderer and side-info-predictor state dicts. Pose codes live in
    # the side-info predictor SD (the substrate's pose_codes parameter is
    # part of WynerZivSubstrate.state_dict() under the ``pose_codes`` key).
    full_sd = dict(arc.renderer_state_dict)
    for k, v in arc.side_info_predictor_state_dict.items():
        full_sd[k] = v
    incompat = substrate.load_state_dict(full_sd, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "WZ1 archive state_dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return substrate


def _wyner_ziv_reconstruct_pair(
    substrate: WynerZivSubstrate,
    pair_idx: int,
    *,
    coset_index: int,
    num_cosets: int,
    search_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run renderer + side-info disambiguation for one pair.

    Per Wyner-Ziv 1976 the receiver:
    1. Computes Y from the side-info predictor (deterministic given pose code).
    2. Looks up the transmitted coset index.
    3. Picks the coset member nearest to Y (DISCUS disambiguation).

    The renderer's output is used as the COSET-CENTER predictor; the coset
    index applies a small per-pair gain shift in [-1/num_cosets, +1/num_cosets]
    so the reconstruction is renderer + small disambiguation residual.
    """
    rgb_0_pred, rgb_1_pred = substrate.render_pair(pair_idx)
    y_0, y_1 = substrate.predict_side_info(pair_idx)
    # Coset-center coarse-grain: average renderer + side-info for candidate
    # selection, but shift the renderer relative to its own mean. Otherwise
    # disagreement between renderer and side-info can decode the right coset
    # representative and still emit a reconstruction in the wrong coset.
    mid = 0.5 * (rgb_0_pred + y_0)
    side_info_scalar = mid.mean().clamp(0.0, 1.0).unsqueeze(0)
    renderer_scalar = rgb_0_pred.mean().clamp(0.0, 1.0).unsqueeze(0)
    disamb = disambiguate_coset(
        side_info_scalar,
        coset_index=coset_index,
        num_cosets=num_cosets,
        search_grid=search_grid,
    )
    shift = disamb.item() - float(renderer_scalar.item())
    rgb_0_out = (rgb_0_pred + shift).clamp(0.0, 1.0)
    rgb_1_out = (rgb_1_pred + shift).clamp(0.0, 1.0)
    return rgb_0_out, rgb_1_out


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one WZ1 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    render_device = select_inflate_device(device)
    substrate = _build_substrate_from_archive(arc, device=render_device)
    coset_indices = arc.coset_indices.tolist()
    num_cosets = arc.num_cosets
    search_grid = max(int(arc.meta.get("search_grid_size", 32)), int(num_cosets))

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(arc.num_pairs):
            rgb_0, rgb_1 = _wyner_ziv_reconstruct_pair(
                substrate,
                pair_idx,
                coset_index=int(coset_indices[pair_idx]),
                num_cosets=num_cosets,
                search_grid=search_grid,
            )
            frames_written += write_rgb_pair_to_raw(
                fh, rgb_0, rgb_1, input_range="unit"
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


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
