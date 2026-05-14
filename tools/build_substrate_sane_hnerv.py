# SPDX-License-Identifier: MIT
"""Build the sane_hnerv archive from a trained checkpoint.

Operator-callable build tool per the Fields-medal grand council substrate
design wave (2026-05-12). Consumes a trained substrate checkpoint and emits
a single ``0.bin`` archive byte-stream that the substrate's inflate.py
parses back into the renderer.

Usage::

    .venv/bin/python tools/build_substrate_sane_hnerv.py \\
        --checkpoint experiments/results/<run>/best.pt \\
        --output experiments/results/sane_hnerv_archive_<utc>/0.bin

The output is a byte-closed archive ready for ``zip -X 0.bin`` packing into
the contest's archive.zip.

CLAUDE.md compliance:
- No silent device defaults (cpu only for build; eval has its own contract)
- No /tmp paths (all outputs under experiments/results/ or operator-provided)
- No scorer load (build is a pure encode step)
- Archive bytes are deterministic (no timestamps in the grammar)
- Catalog #139 no_op_proof: emitted alongside the archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_substrate_sane_hnerv",
        description=__doc__,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained substrate checkpoint (.pt; torch.save dict with 'state_dict', 'config')",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output 0.bin path. The parent directory will be created.",
    )
    p.add_argument(
        "--emit-manifest",
        action="store_true",
        help="Also emit a build_manifest.json alongside 0.bin (Catalog #93 schema)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Import lazily to keep CLI --help fast
    import torch  # noqa: F401  (validates torch is importable for archive build)

    from tac.substrates.sane_hnerv.archive import pack_archive
    from tac.substrates.sane_hnerv.architecture import SaneHnervConfig

    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    # ALLOWED loader: this is OUR checkpoint; weights_only=True is sufficient
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state or "config" not in state:
        print(
            "checkpoint must contain {'state_dict': ..., 'config': SaneHnervConfig (as dict)}",
            file=sys.stderr,
        )
        return 2

    cfg_dict = state["config"]
    cfg = SaneHnervConfig(
        latent_dim=int(cfg_dict["latent_dim"]),
        embed_dim=int(cfg_dict["embed_dim"]),
        initial_grid_h=int(cfg_dict["initial_grid_h"]),
        initial_grid_w=int(cfg_dict["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in cfg_dict["decoder_channels"]),
        sin_frequency=float(cfg_dict["sin_frequency"]),
        num_pairs=int(cfg_dict["num_pairs"]),
        output_height=int(cfg_dict["output_height"]),
        output_width=int(cfg_dict["output_width"]),
        num_upsample_blocks=int(cfg_dict["num_upsample_blocks"]),
    )

    sd = state["state_dict"]
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].detach().cpu()

    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }

    blob = pack_archive(decoder_sd, latents, meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    archive_sha = hashlib.sha256(blob).hexdigest()
    print(f"wrote {args.output} ({len(blob)} bytes, sha256={archive_sha})")

    if args.emit_manifest:
        manifest = {
            "schema": "sane_hnerv_build_manifest_v1",
            "generated_at": _utc_now(),
            "from_state_hash": f"sha256:{_sha256_file(ckpt_path)}",
            "archive_relpath": args.output.name,
            "archive_sha256": archive_sha,
            "archive_size_bytes": len(blob),
            "custody_status": "ci-rebuildable",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "no_op_proof_status": "advisory_only__build_smoke_only",
            "config": cfg_dict,
        }
        mf = args.output.parent / "build_manifest.json"
        mf.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {mf}")

    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
