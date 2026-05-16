#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an NSCS01 byte-closed split-renderer packet skeleton.

The emitted packet is byte-closed and runtime-consumption-proved, but it is
not trained and makes no score claim.  It is the concrete build contract for
the frame0 pose-heavy / frame1 seg-heavy NSCS01 substrate before any scorer or
provider work is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from experiments.train_substrate_nscs01_nullspace_split_renderer import (  # noqa: E402
    _write_runtime,
)
from tac.substrates.nscs01_nullspace_split_renderer.architecture import (  # noqa: E402
    NullspaceSplitConfig,
)
from tac.substrates.nscs01_nullspace_split_renderer.build_packet import (  # noqa: E402
    build_manifest,
    build_skeleton_archive_bytes,
    prove_runtime_consumes_score_affecting_sections,
    write_deterministic_archive_zip,
    write_manifest,
)


def build_nscs01_split_renderer_packet(
    *,
    out_dir: Path,
    seed: int = 0,
    num_pairs: int = 2,
    latent_dim: int = 8,
    head0_bits: int = 4,
    head1_bits: int = 8,
    latent_bits: int = 12,
    head0_base_channels: int = 8,
    head1_base_channels: int = 16,
    proof_device: str = "cpu",
    run_consumption_proof: bool = True,
) -> dict[str, Any]:
    """Build archive/runtime/manifest under ``out_dir`` and return manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NullspaceSplitConfig(
        latent_dim=latent_dim,
        head0_bits=head0_bits,
        head1_bits=head1_bits,
        latent_bits=latent_bits,
        head0_base_channels=head0_base_channels,
        head1_base_channels=head1_base_channels,
        num_pairs=num_pairs,
    )
    archive_bytes = build_skeleton_archive_bytes(
        cfg=cfg,
        seed=seed,
        extra_meta={
            "builder": "tools/build_nscs01_split_renderer_packet.py",
            "builder_contract": "byte_closed_skeleton_no_score_claim",
        },
    )

    payload_path = out_dir / "0.bin"
    payload_path.write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    write_deterministic_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)

    if run_consumption_proof:
        proof = prove_runtime_consumes_score_affecting_sections(
            archive_bytes,
            pair_indices=(0,),
            device=proof_device,
        )
        if not proof["all_score_affecting_sections_consumed"]:
            raise RuntimeError(
                "NSCS01 runtime consumption proof failed for one or more "
                "score-affecting sections"
            )
    else:
        proof = {
            "schema": "nscs01_runtime_consumption_proof_v1",
            "skipped": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "all_score_affecting_sections_consumed": False,
        }

    manifest = build_manifest(
        cfg=cfg,
        seed=seed,
        archive_bytes=archive_bytes,
        archive_zip_path=archive_zip_path,
        runtime_dir=submission_dir,
        consumption_proof=proof,
    )
    manifest_path = out_dir / "build_manifest.json"
    write_manifest(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--head0-bits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--head1-bits", type=int, choices=(6, 8), default=8)
    parser.add_argument("--latent-bits", type=int, choices=(8, 12), default=12)
    parser.add_argument("--head0-base-channels", type=int, default=8)
    parser.add_argument("--head1-base-channels", type=int, default=16)
    parser.add_argument("--proof-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--skip-consumption-proof",
        action="store_true",
        help="emit packet without running the runtime mutation proof",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_nscs01_split_renderer_packet(
        out_dir=args.out_dir,
        seed=args.seed,
        num_pairs=args.num_pairs,
        latent_dim=args.latent_dim,
        head0_bits=args.head0_bits,
        head1_bits=args.head1_bits,
        latent_bits=args.latent_bits,
        head0_base_channels=args.head0_base_channels,
        head1_base_channels=args.head1_base_channels,
        proof_device=args.proof_device,
        run_consumption_proof=not args.skip_consumption_proof,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_nscs01_split_renderer_packet", "main"]
