# SPDX-License-Identifier: MIT
"""D1 inflate runtime: verify sidecar/base, render base, apply overlay.

Catalog #146 CLI: ``inflate.py archive_dir output_dir file_list``. No scorer
load, no score claim, no /tmp paths.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from tac.substrates._shared.inflate_runtime import select_inflate_device


def _read_file_list(file_list_path: Path) -> list[str]:
    """Read the contest file list (one safe relative video name per line)."""
    if not file_list_path.is_file():
        raise FileNotFoundError(f"file_list not found: {file_list_path}")
    return [
        line.strip()
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _locate_d1_archive(archive_dir: Path) -> Path:
    """Locate the D1 sidecar and fail closed if absent."""
    candidate = archive_dir / "d1_polytope.bin"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"D1 sidecar not found at {candidate}; archive_dir={archive_dir}"
        )
    return candidate


def _verify_base_archive_match(
    archive_dir: Path,
    *,
    base_substrate_id: str,
    base_sha_truncated: str,
) -> Path:
    """Locate the base archive and verify the truncated sha matches."""
    candidate = archive_dir / f"{base_substrate_id}.bin"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Base substrate archive not found at {candidate}; "
            f"D1 was paired with base_substrate_id={base_substrate_id!r}"
        )
    full_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if full_sha[:16] != base_sha_truncated:
        raise ValueError(
            f"Base archive sha mismatch: file={full_sha[:16]} "
            f"vs D1.expected={base_sha_truncated}. The D1 sidecar was "
            "paired with a different base archive — refuse the inflate."
        )
    return candidate


def main(argv: list[str] | None = None) -> int:
    """Catalog #146 CLI."""
    parser = argparse.ArgumentParser(
        prog="inflate", description="D1 sidecar inflate (Catalog #146)"
    )
    parser.add_argument(
        "archive_dir", type=Path,
        help="Directory containing d1_polytope.bin + base.bin",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for .raw files"
    )
    parser.add_argument(
        "file_list", type=Path,
        help="Newline-delimited list of video names",
    )
    args = parser.parse_args(argv)

    # Local imports to keep CLI invocation cheap.
    from tac.substrates.d1_segnet_margin_polytope.archive import parse_archive

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    file_list_path = args.file_list.resolve()

    if not archive_dir.is_dir():
        raise NotADirectoryError(
            f"archive_dir is not a directory: {archive_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    d1_path = _locate_d1_archive(archive_dir)
    d1_archive = parse_archive(d1_path.read_bytes())

    base_path = _verify_base_archive_match(
        archive_dir,
        base_substrate_id=d1_archive.base_substrate_id,
        base_sha_truncated=d1_archive.base_archive_sha256_truncated,
    )

    device = select_inflate_device()
    print(
        f"[d1-inflate] device={device} "
        f"base={d1_archive.base_substrate_id} "
        f"base_path={base_path.name} "
        f"margin_shape=({d1_archive.height},{d1_archive.width}) "
        f"jacobian_lipschitz={d1_archive.jacobian_lipschitz:.3f} "
        f"meta_keys={sorted(d1_archive.meta.keys())[:5]}",
        file=sys.stderr,
    )

    # Delegate to base substrate inflate; D1 applies polytope overlay after.
    video_names = _read_file_list(file_list_path)
    if not video_names:
        raise ValueError(f"file_list {file_list_path} is empty")

    base_module_name = (
        f"tac.substrates.{d1_archive.base_substrate_id}.inflate"
    )
    try:
        import importlib

        base_inflate = importlib.import_module(base_module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"D1 base substrate {d1_archive.base_substrate_id!r} does "
            f"not expose an inflate module at {base_module_name}; "
            "cannot delegate."
        ) from exc

    if not hasattr(base_inflate, "main"):
        raise RuntimeError(
            f"D1 base inflate module {base_module_name} does not expose "
            "main(argv)"
        )

    rc = base_inflate.main(
        [str(archive_dir), str(output_dir), str(file_list_path)]
    )
    if rc:
        return int(rc)

    # L2 INTEGRATION per Catalog #220 NON-NEGOTIABLE.
    from tac.substrates.d1_segnet_margin_polytope.overlay import (
        apply_l2_overlay_for_video_list,
        pair_sign_mask_from_meta,
    )

    sign_policy = str(d1_archive.meta.get("overlay_sign_policy", "payload"))
    overlay_diag = apply_l2_overlay_for_video_list(
        output_dir=output_dir,
        video_names=video_names,
        polytope_payload=d1_archive.polytope_payload,
        encoder_grid_h=d1_archive.height,
        encoder_grid_w=d1_archive.width,
        margin_map_int8=d1_archive.margin_map_int8,
        margin_map_scale=d1_archive.margin_map_scale,
        archive_jacobian_lipschitz=d1_archive.jacobian_lipschitz,
        channel_policy=str(
            d1_archive.meta.get("overlay_channel_policy", "rgb")
        ),
        amplitude_scale=float(
            d1_archive.meta.get("overlay_amplitude_scale", 1.0)
        ),
        sign_policy=sign_policy,
        pair_sign_mask=pair_sign_mask_from_meta(d1_archive.meta),
    )
    print(
        f"[d1-inflate] OVERLAY_TOTAL pairs_modified="
        f"{overlay_diag['total_pairs_modified']} bytes_changed="
        f"{overlay_diag['total_bytes_changed']} videos_processed="
        f"{overlay_diag['videos_processed']} "
        f"contract_nonzero_noise_pixels="
        f"{overlay_diag['contract_nonzero_noise_pixels']} "
        f"contract_boundary_pixels="
        f"{overlay_diag['contract_boundary_pixels']} "
        f"channel_policy={overlay_diag['channel_policy']} "
        f"amplitude_scale={overlay_diag['overlay_amplitude_scale']} "
        f"sign_policy={overlay_diag['overlay_sign_policy']}",
        file=sys.stderr,
    )
    if overlay_diag["total_bytes_changed"] <= 0:
        raise RuntimeError(
            "D1 sidecar was present but overlay changed zero bytes; "
            "refusing dead-rate packet"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
