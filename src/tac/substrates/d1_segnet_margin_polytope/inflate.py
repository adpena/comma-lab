"""D1 inflate runtime — sidecar margin-map dequant + L2 polytope overlay.

Per HNeRV parity L4: inflate.py <= 200 LOC substrate-engineering waiver.
NO scorer load (strict-scorer-rule). torch + numpy + brotli only.
Catalog #146 CLI: ``inflate.py archive_dir output_dir file_list``.

Steps: (1) locate ``d1_polytope.bin``; (2) verify base sha-truncated match;
(3) delegate to base substrate's inflate to write .raw files; (4) L2
INTEGRATION — apply polytope-interior noise overlay to every frame_1 in
each output .raw via overlay.apply_l2_overlay_for_video_list. Operational
score-improvement mechanism per Catalog #220 NON-NEGOTIABLE: byte addition
must produce real frame changes, not dead bytes paying rate-term penalty.

NO score claim. NO /tmp paths.
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
    """Locate the D1POLY1 0.bin sidecar inside ``archive_dir``.

    Convention: substrate archives live as ``<substrate_id>.bin`` per
    HNeRV parity L3 monolithic single-file rule. D1 sidecar lives as
    ``d1_polytope.bin``. Fail-closed if absent.
    """
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
    """Locate the base substrate's archive and verify the truncated sha matches.

    Returns the path to the base archive. Raises if missing or sha
    mismatch (fail-closed per CLAUDE.md "Apples-to-apples evidence
    discipline").
    """
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
    """Catalog #146 CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``.

    Delegates to ``tac.substrates.<base>.inflate.main`` then applies the
    L2 polytope overlay to every frame_1 in each .raw output. Returns 0
    on success, non-zero on failure.
    """
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
    )

    overlay_diag = apply_l2_overlay_for_video_list(
        output_dir=output_dir,
        video_names=video_names,
        polytope_payload=d1_archive.polytope_payload,
        encoder_grid_h=d1_archive.height,
        encoder_grid_w=d1_archive.width,
    )
    print(
        f"[d1-inflate] OVERLAY_TOTAL pairs_modified="
        f"{overlay_diag['total_pairs_modified']} bytes_changed="
        f"{overlay_diag['total_bytes_changed']} videos_processed="
        f"{overlay_diag['videos_processed']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
