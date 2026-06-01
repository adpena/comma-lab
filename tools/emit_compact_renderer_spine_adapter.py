#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the shared packet spine for compact renderer candidates.

This is the non-PR95 entry point for RNeRV/SRNeRV/BoostNeRV and
PVQ/RT-VQ-NeRV experiments. It does not train models. It converts already
trained, charged blobs into the common HPRC representation spine so acquisition
can price decoder, latent/token, selector, codebook, and residual bytes under
one schema. Missing trained blobs stay blocked and non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from tac.substrates.hprc.representation_spine import (  # noqa: E402
    HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    write_representation_spine_projection,
)

COMPACT_RENDERER_SPINE_ADAPTER_SCHEMA = "compact_renderer_spine_adapter.v1"
BYTE_CEILINGS = (100_000, 178_417, 216_000, 285_000)

FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
}

FAMILY_ALIASES: dict[str, HprcRepresentationFamily] = {
    "pr95": HprcRepresentationFamily.PR95_HNERV,
    "pr95_hnerv": HprcRepresentationFamily.PR95_HNERV,
    "hnerv": HprcRepresentationFamily.PR95_HNERV,
    "hnerv_packed": HprcRepresentationFamily.HNERV_PACKED,
    "packed_hnerv": HprcRepresentationFamily.HNERV_PACKED,
    "rnerv": HprcRepresentationFamily.RNERV,
    "hinerv": HprcRepresentationFamily.HI_NERV,
    "hi_nerv": HprcRepresentationFamily.HI_NERV,
    "snerv": HprcRepresentationFamily.S_NERV,
    "s_nerv": HprcRepresentationFamily.S_NERV,
    "srnerv": HprcRepresentationFamily.SR_NERV,
    "sr_nerv": HprcRepresentationFamily.SR_NERV,
    "boostnerv": HprcRepresentationFamily.BOOST_NERV,
    "boost_nerv": HprcRepresentationFamily.BOOST_NERV,
    "pvq": HprcRepresentationFamily.PVQ_NERV,
    "pvq_nerv": HprcRepresentationFamily.PVQ_NERV,
    "rt_vq": HprcRepresentationFamily.RT_VQ_NERV,
    "rt_vq_nerv": HprcRepresentationFamily.RT_VQ_NERV,
    "vq_nerv": HprcRepresentationFamily.VQ_NERV,
}


class CompactRendererSpineAdapterError(ValueError):
    """Raised when a compact renderer cannot enter the shared spine."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _read_optional(path: Path | None) -> bytes:
    if path is None:
        return b""
    return Path(path).read_bytes()


def _artifact_row(path: Path | None, *, role: str) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    return {
        "role": role,
        "path": p.as_posix(),
        "bytes": p.stat().st_size,
        "sha256": _sha256_file(p),
    }


def _coerce_family(value: str) -> HprcRepresentationFamily:
    key = value.strip().lower()
    if key not in FAMILY_ALIASES:
        raise CompactRendererSpineAdapterError(
            f"unsupported compact renderer family {value!r}; "
            f"choices={sorted(FAMILY_ALIASES)}"
        )
    return FAMILY_ALIASES[key]


def _byte_ceiling_report(total_bytes: int) -> dict[str, Any]:
    return {
        "schema": "compact_renderer_byte_ceiling_report.v1",
        "hprc_projection_bytes": int(total_bytes),
        "ceilings": [
            {
                "ceiling_bytes": ceiling,
                "fits": int(total_bytes) <= ceiling,
                "delta_bytes": int(total_bytes) - ceiling,
            }
            for ceiling in BYTE_CEILINGS
        ],
    }


def emit_compact_renderer_spine_adapter(
    *,
    family: str,
    output_dir: Path,
    decoder_blob: Path,
    latents_blob: Path,
    codebooks_blob: Path | None = None,
    selectors_blob: Path | None = None,
    residual_blob: Path | None = None,
    receiver_state_blob: Path | None = None,
    trained_weights_provenance: str,
    trained_latents_provenance: str,
    allow_untrained_fixture: bool = False,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fam = _coerce_family(family)
    if not allow_untrained_fixture:
        if not trained_weights_provenance.strip():
            raise CompactRendererSpineAdapterError(
                "trained_weights_provenance_required"
            )
        if not trained_latents_provenance.strip():
            raise CompactRendererSpineAdapterError(
                "trained_latents_or_tokens_provenance_required"
            )

    source_rows = [
        row
        for row in (
            _artifact_row(decoder_blob, role="trained_decoder_weights_or_program"),
            _artifact_row(latents_blob, role="trained_latents_or_tokens"),
            _artifact_row(codebooks_blob, role="codebooks"),
            _artifact_row(selectors_blob, role="selectors"),
            _artifact_row(residual_blob, role="residual_tokens"),
            _artifact_row(receiver_state_blob, role="receiver_state"),
        )
        if row is not None
    ]
    spine = build_generic_neural_spine_packet(
        family=fam,
        decoder_blob=_read_optional(decoder_blob),
        latents_blob=_read_optional(latents_blob),
        codebooks_blob=_read_optional(codebooks_blob),
        selectors_blob=_read_optional(selectors_blob),
        residual_blob=_read_optional(residual_blob),
        receiver_state_blob=_read_optional(receiver_state_blob),
        manifest_extra={
            "adapter_schema": COMPACT_RENDERER_SPINE_ADAPTER_SCHEMA,
            "trained_weights_provenance": trained_weights_provenance,
            "trained_latents_or_tokens_provenance": trained_latents_provenance,
            "allow_untrained_fixture": bool(allow_untrained_fixture),
            "source_artifacts": source_rows,
            "promotion_rule": "receiver_proof_and_exact_gate_required; proxy wins forbidden",
            **(manifest_extra or {}),
        },
    )
    projection = write_representation_spine_projection(
        output_dir=output_dir,
        spine=spine,
        basename=f"{fam.value}_representation_spine",
    )
    blockers = [
        "archive_zip_runtime_receiver_proof_not_yet_emitted",
        "full_video_scorer_value_per_byte_not_yet_measured",
        "contest_cpu_cuda_exact_eval_missing",
    ]
    if allow_untrained_fixture:
        blockers.insert(0, "untrained_fixture_not_promotable")
    report = {
        "schema": COMPACT_RENDERER_SPINE_ADAPTER_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "family": fam.value,
        "output_dir": Path(output_dir).as_posix(),
        "projection_schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
        "projection": projection,
        "source_artifacts": source_rows,
        "byte_ceilings": _byte_ceiling_report(int(projection["hprc_bin_bytes"])),
        "exact_gate": {
            "schema": "compact_renderer_exact_gate_blocker.v1",
            "ready_for_exact_eval_dispatch": False,
            "blockers": blockers,
        },
        **FALSE_AUTHORITY,
    }
    report_path = Path(output_dir) / f"{fam.value}_spine_adapter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report_path": report_path.as_posix()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_ALIASES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-blob", type=Path, required=True)
    parser.add_argument("--latents-blob", type=Path, required=True)
    parser.add_argument("--codebooks-blob", type=Path)
    parser.add_argument("--selectors-blob", type=Path)
    parser.add_argument("--residual-blob", type=Path)
    parser.add_argument("--receiver-state-blob", type=Path)
    parser.add_argument("--trained-weights-provenance", default="")
    parser.add_argument("--trained-latents-provenance", default="")
    parser.add_argument("--allow-untrained-fixture", action="store_true")
    parser.add_argument(
        "--num-pairs",
        type=int,
        help="Declared full-video pair coverage for coverage gating.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = emit_compact_renderer_spine_adapter(
        family=args.family,
        output_dir=args.output_dir,
        decoder_blob=args.decoder_blob,
        latents_blob=args.latents_blob,
        codebooks_blob=args.codebooks_blob,
        selectors_blob=args.selectors_blob,
        residual_blob=args.residual_blob,
        receiver_state_blob=args.receiver_state_blob,
        trained_weights_provenance=args.trained_weights_provenance,
        trained_latents_provenance=args.trained_latents_provenance,
        allow_untrained_fixture=bool(args.allow_untrained_fixture),
        manifest_extra=(
            None
            if args.num_pairs is None
            else {
                "num_pairs": int(args.num_pairs),
                "coverage_source": "compact_renderer_spine_adapter_cli",
            }
        ),
    )
    print(
        "[compact-spine] "
        f"family={report['family']} bytes={report['projection']['hprc_bin_bytes']} "
        f"report={report['report_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
