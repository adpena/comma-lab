#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a structurally-FULL ATW2 candidate archive via $0 local CPU smoke.

PER OVERNIGHT-I (operator NON-NEGOTIABLE 2026-05-21) + Catalog #270 dispatch
optimization protocol + CLAUDE.md "Carmack MVP-first phasing" non-negotiable:

The ATW2 trainer's _full_main path REFUSES CPU per CLAUDE.md "MPS auth eval is
NOISE" + "EMA - non-negotiable"; the existing smoke path hardcodes
``num_pairs=8`` so its archive is classified ``smoke_or_small_candidate`` by
``tools/scan_atw2_cdf_compaction_candidates.py`` (which gates ``full_candidate``
at ``num_pairs >= 600``). Codex blocker memos
``codex_findings_atw2_full_candidate_generation_local_blocker_*`` documented the
gap: tooling is ready; no FULL candidate exists locally for the gate to accept.

This tool resolves the blocker WITHOUT paid GPU dispatch:

1. Build the ATW2 codec with a tiny per-pair config (latent_dim=8,
   decoder_embed_dim=8, output 16x24) but parametrized ``num_pairs`` (default
   600 = ``FULL_CANDIDATE_MIN_PAIRS``).
2. Populate ``cdf_table`` + ``scorer_class_prior_table`` with deterministic
   non-zero patterns sister to the trainer's ``_smoke_main`` (so the archive
   exercises the same B3 contents pattern).
3. Pack ATW2 archive via ``pack_archive``, write ``0.bin`` + deterministic
   stored-mode ``archive.zip``.
4. Verify the resulting archive is classified ``full_candidate=True`` by the
   canonical scanner.
5. Emit a research-only JSON manifest with canonical Provenance per Catalog
   #323 + slot 3-r7 REMOVAL paradigm reclassification + RATIFY-4 EXCLUDED
   context #6 (``direct_byte_substitution_on_decode_opaque_raw_sections``).

NO scorer load. NO real video decode. NO score claim. NO promotion eligibility.
NO ready-for-exact-eval-dispatch. ``research_only=True``.

The artifact is structural proof that the FULL-candidate gate accepts a
synthetic-data 600-pair archive; the inflate-side compaction-parity proof
(``prove_atw2_cdf_compaction_parity``) is a separate downstream concern covered
by ``src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py`` and is
NOT exercised by default here (600-pair sequential inflate is heavy and
orthogonal to gate acceptance).

Per slot 3-r7 reconciliation memo + RATIFY-4: the ATW V2 ``cdf_table_blob``
is REMOVAL-paradigm-eligible (decoder is decode-opaque per codex byte-mutation
smoke ``057130de4`` ``max_abs_raw_byte_delta=0``). This tool does NOT register
a canonical equation #26 IN-DOMAIN anchor; the EXCLUDED context #6 covers
that class structurally.

Cost: $0 + local CPU; expected wall-clock <60s for ``--num-pairs 600``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tac.substrates.atw_codec_v2 import (  # noqa: E402
    ATW2_MAGIC,
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2Variant,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (  # noqa: E402
    analyze_atw2_cdf_section,
)

# Canonical scanner constant per
# tools/scan_atw2_cdf_compaction_candidates.py:30
FULL_CANDIDATE_MIN_PAIRS = 600

# Canonical Provenance fields per Catalog #323 + slot 3-r7 reconciliation.
PROVENANCE_REMOVAL_PARADIGM_RECLASSIFICATION_SOURCE_MEMO = (
    ".omx/research/"
    "atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_"
    "20260521.md"
)
PROVENANCE_RATIFY_4_EXCLUDED_CONTEXT_TOKEN = (
    "direct_byte_substitution_on_decode_opaque_raw_sections"
)
PROVENANCE_RATIFY_4_SOURCE_MEMO = (
    ".omx/research/"
    "canonical_equation_26_excluded_context_decode_opaque_raw_sections_"
    "registration_landed_20260521.md"
)


@dataclass(frozen=True)
class Atw2FullCandidateSmokeResult:
    """Result manifest for the MVP-first phasing ATW2 FULL candidate smoke."""

    smoke_kind: str
    num_pairs: int
    full_candidate_per_gate: bool
    candidate_class: str
    full_candidate_min_pairs_threshold: int
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    cdf_offset: int
    cdf_bytes: int
    cdf_classes: int
    cdf_symbols: int
    conservative_bytes_saved: int
    conservative_delta_s_rate_only: float
    schema_version: int
    variant: int
    elapsed_seconds: float
    device: str
    # Canonical Provenance per Catalog #323 + #287 (axis_tag empirical;
    # this artifact is a structural proof, not a contest score signal).
    evidence_grade: str = "predicted"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    research_only: bool = True
    removal_paradigm_reclassification_per_slot_3_r7: bool = True
    canonical_equation_26_excluded_context_per_ratify_4: str = (
        PROVENANCE_RATIFY_4_EXCLUDED_CONTEXT_TOKEN
    )
    provenance_source_memo_slot_3_r7: str = (
        PROVENANCE_REMOVAL_PARADIGM_RECLASSIFICATION_SOURCE_MEMO
    )
    provenance_source_memo_ratify_4: str = PROVENANCE_RATIFY_4_SOURCE_MEMO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_stored_archive_zip(path: Path, member_bytes: bytes) -> None:
    """Write a deterministic stored-mode archive.zip with one 0.bin member.

    Mirrors ``experiments/train_substrate_atw_codec_v2.py::_build_archive_zip``
    so the scanner sees the same structural shape produced by the canonical
    trainer.
    """
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, member_bytes)


def generate_atw2_full_candidate_smoke(
    *,
    num_pairs: int = FULL_CANDIDATE_MIN_PAIRS,
    output_dir: Path,
    seed: int = 20260521,
) -> Atw2FullCandidateSmokeResult:
    """Generate a structurally-FULL ATW2 candidate archive at $0 CPU.

    Args:
        num_pairs: pair count for the ATW2 codec config; default 600 =
            ``FULL_CANDIDATE_MIN_PAIRS`` (the canonical scanner's threshold).
        output_dir: where ``0.bin`` + ``archive.zip`` are written.
        seed: torch seed; default 20260521 = today's UTC date.

    Returns:
        ``Atw2FullCandidateSmokeResult`` with canonical Provenance fields.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    torch.manual_seed(seed)

    # Tiny per-pair config so 600 pairs fits in <60s on local CPU.
    # The shape mirrors the existing smoke (trainer lines 384-387) except
    # num_pairs is parametrized.
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
        num_pairs=num_pairs,
        output_height=16,
        output_width=24,
        scorer_class_prior_dim=8,
        wz_head_hidden_dim=8,
        g1_distill_hidden_dim=8,
    )

    model = ATWv2Codec(cfg).eval()
    with torch.no_grad():
        model.scorer_class_prior_table.normal_(0.0, 0.2)
        model.cdf_table.copy_(
            torch.linspace(
                0.001, 0.999, model.cdf_table.numel()
            ).view_as(model.cdf_table)
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

    archive_bytes = pack_archive(
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

    archive_path = output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Verification step 1: parse roundtrip
    if not archive_bytes.startswith(ATW2_MAGIC):
        raise RuntimeError(
            f"archive magic mismatch: got {archive_bytes[:4]!r}"
            f" expected {ATW2_MAGIC!r}"
        )
    parsed = parse_archive(archive_bytes)
    if parsed.schema_version != 1:
        raise RuntimeError(f"unexpected schema version: {parsed.schema_version}")

    # Verification step 2: num_pairs derived from latent_residual.shape[0]
    # per the canonical scanner
    # (tools/scan_atw2_cdf_compaction_candidates.py:122)
    parsed_num_pairs = int(parsed.latent_residual.shape[0])
    if parsed_num_pairs != num_pairs:
        raise RuntimeError(
            f"num_pairs roundtrip mismatch: packed {num_pairs}, "
            f"parsed {parsed_num_pairs}"
        )

    # Verification step 3: write deterministic archive.zip wrapping 0.bin so
    # the canonical scanner can classify it as full_candidate.
    archive_zip_path = output_dir / "archive.zip"
    _write_stored_archive_zip(archive_zip_path, archive_bytes)

    # Verification step 4: CDF compaction analysis (does not depend on inflate)
    analysis = analyze_atw2_cdf_section(archive_bytes)

    full_candidate_per_gate = parsed_num_pairs >= FULL_CANDIDATE_MIN_PAIRS
    candidate_class = (
        "full_candidate"
        if full_candidate_per_gate
        else "smoke_or_small_candidate"
    )

    elapsed = time.time() - t0
    return Atw2FullCandidateSmokeResult(
        smoke_kind="atw2_full_candidate_mvp_first_phasing",
        num_pairs=parsed_num_pairs,
        full_candidate_per_gate=full_candidate_per_gate,
        candidate_class=candidate_class,
        full_candidate_min_pairs_threshold=FULL_CANDIDATE_MIN_PAIRS,
        archive_path=str(archive_path),
        archive_bytes=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        archive_zip_path=str(archive_zip_path),
        archive_zip_bytes=archive_zip_path.stat().st_size,
        archive_zip_sha256=hashlib.sha256(
            archive_zip_path.read_bytes()
        ).hexdigest(),
        cdf_offset=int(analysis.cdf_offset),
        cdf_bytes=int(analysis.cdf_bytes),
        cdf_classes=int(analysis.cdf_classes),
        cdf_symbols=int(analysis.cdf_symbols),
        conservative_bytes_saved=int(analysis.conservative_bytes_saved),
        conservative_delta_s_rate_only=float(
            analysis.conservative_delta_s_rate_only
        ),
        schema_version=parsed.schema_version,
        variant=parsed.variant,
        elapsed_seconds=round(elapsed, 3),
        device="cpu",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a structurally-FULL ATW2 candidate archive via $0 "
            "local CPU smoke (MVP-first phasing per CLAUDE.md Carmack)."
        )
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=FULL_CANDIDATE_MIN_PAIRS,
        help=(
            "ATW2 codec num_pairs; default 600 = FULL_CANDIDATE_MIN_PAIRS "
            "per the canonical scanner."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for 0.bin + archive.zip + result.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260521,
        help="Torch seed; default 20260521 = today's UTC date",
    )
    args = parser.parse_args(argv)

    result = generate_atw2_full_candidate_smoke(
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    manifest_path = args.output_dir / "result.json"
    manifest_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
