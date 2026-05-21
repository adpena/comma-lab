#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe whether ATW2 cdf_table_blob affects current inflate output."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.atw_codec_v2 import (  # noqa: E402
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2Variant,
    pack_archive,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (  # noqa: E402
    analyze_atw2_cdf_section,
    compact_atw2_cdf_table_in_archive_zip,
    prove_atw2_cdf_compaction_parity,
    prove_atw2_cdf_decode_influence,
)


def _read_archive_bytes(path: Path, *, zip_member: str) -> bytes:
    if path.is_dir():
        candidates = [path / "0.bin", path / "x"]
        present = [p for p in candidates if p.is_file()]
        if len(present) != 1:
            raise ValueError(
                f"expected exactly one ATW2 member at {candidates[0]} or {candidates[1]}"
            )
        return present[0].read_bytes()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            return zf.read(zip_member)
    return path.read_bytes()


def _synthetic_archive_bytes() -> bytes:
    cfg = ATWv2CodecConfig(
        variant=ATWv2Variant.B_WZ_ONLY,
        latent_dim=8,
        encoder_input_channels=3,
        encoder_hidden_dim=16,
        decoder_embed_dim=8,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(6, 4, 4, 4, 4, 4),
        decoder_num_upsample_blocks=2,
        num_pairs=4,
        output_height=16,
        output_width=24,
        scorer_class_prior_dim=8,
        wz_head_hidden_dim=8,
        g1_distill_hidden_dim=8,
    )
    torch.manual_seed(123)
    model = ATWv2Codec(cfg).eval()
    with torch.no_grad():
        model.scorer_class_prior_table.normal_(0.0, 0.2)
        model.cdf_table.copy_(
            torch.linspace(0.001, 0.999, model.cdf_table.numel()).view_as(
                model.cdf_table
            )
        )
    meta: dict[str, Any] = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    return pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        model.g1_distill_head.state_dict(),
        model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        model.cdf_table.detach().cpu(),
        meta,
        variant=1,
    )


def _build_inner_payload(
    archive_bytes: bytes,
    *,
    analyze_only: bool,
    compact_cdf: bool,
    device: str,
    mutation_kind: str,
) -> dict[str, object]:
    if analyze_only:
        return analyze_atw2_cdf_section(archive_bytes).to_dict()
    if compact_cdf:
        return prove_atw2_cdf_compaction_parity(
            archive_bytes,
            device=device,
        ).to_dict()
    if mutation_kind not in ("xor_ff", "zero"):
        raise ValueError(f"unsupported mutation_kind: {mutation_kind!r}")
    typed_mutation_kind: Literal["xor_ff", "zero"] = (
        "xor_ff" if mutation_kind == "xor_ff" else "zero"
    )
    return prove_atw2_cdf_decode_influence(
        archive_bytes,
        mutation_kind=typed_mutation_kind,
        device=device,
    ).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, help="ATW2 0.bin/x or archive_dir")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run against a deterministic tiny synthetic ATW2 archive.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mutation-kind", choices=("xor_ff", "zero"), default="xor_ff")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--zip-member", default="0.bin")
    parser.add_argument(
        "--output-archive-zip",
        type=Path,
        help="Write an archive.zip with --zip-member compacted; requires --compact-cdf.",
    )
    parser.add_argument(
        "--compact-cdf",
        action="store_true",
        help="Replace cdf_table_blob with the compact sentinel and prove raw parity.",
    )
    args = parser.parse_args(argv)

    if args.output_archive_zip is not None and not args.compact_cdf:
        parser.error("--output-archive-zip requires --compact-cdf")
    if args.synthetic_smoke and args.output_archive_zip is not None:
        parser.error("--output-archive-zip requires a source archive.zip path")

    if args.output_archive_zip is not None:
        if args.archive is None:
            parser.error("pass a source archive.zip path with --output-archive-zip")
        payload = compact_atw2_cdf_table_in_archive_zip(
            args.archive,
            args.output_archive_zip,
            member_name=args.zip_member,
            device=args.device,
        ).to_dict()
    elif args.synthetic_smoke:
        archive_bytes = _synthetic_archive_bytes()
        payload = _build_inner_payload(
            archive_bytes,
            analyze_only=args.analyze_only,
            compact_cdf=args.compact_cdf,
            device=args.device,
            mutation_kind=args.mutation_kind,
        )
    elif args.archive is not None:
        archive_bytes = _read_archive_bytes(args.archive, zip_member=args.zip_member)
        payload = _build_inner_payload(
            archive_bytes,
            analyze_only=args.analyze_only,
            compact_cdf=args.compact_cdf,
            device=args.device,
            mutation_kind=args.mutation_kind,
        )
    else:
        parser.error("pass an ATW2 archive path or --synthetic-smoke")

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
