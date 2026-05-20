#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WAVE-3 null-exploit probe: identify master-gradient NULL-SPACE bytes on an archive.

Operator framing 2026-05-20 [contest-CUDA T4 anchor `a1afce29`-class on fec6
frontier]: identify bytes whose master-gradient is zero across all measured
score axes — i.e., bytes the scorer is structurally insensitive to. These
are *contest-compliant procedural-codebook candidates* per
`.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md`
Q4 verdict (NULL-SPACE EXPLOITATION = REDUCES bytes INSIDE archive.zip; rate
term moves correct direction; no maintainer-rejection grounds).

What this tool does NOT do (per Catalog #318):
- It is NOT a score authority. It does not propose mutations.
- It does NOT bypass `tac.master_gradient.CandidateModificationSpec` /
  `grammar_aware_operator` discipline. Any actual archive mutation MUST
  go through the typed packet-compiler interface, not raw byte flips.
- It is an *observability-only* indexing of null candidates.

Output payload is canonical-Provenance tagged per Catalog #323
(`score_claim=False`, `promotable=False`, `axis_tag="[predicted]"`).

Usage:

    .venv/bin/python tools/probe_null_byte_master_gradient.py \
        --anchor .omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy \
        --archive-zip experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
        --inner-member-name x \
        --epsilon 1e-9 \
        --output-dir experiments/results/null_byte_probe_<utc>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# fec6 grammar magic constants — verified empirically against
# experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py
_FEC6_OUTER_MAGIC = b"FP11"
_FEC6_INNER_SELECTOR_MAGIC = b"FEC6"


@dataclass(frozen=True)
class _FECGrammarSections:
    """Section breakdown for an FP11/FEC6-wrapped fec6 archive inner member."""

    outer_magic_start: int
    outer_magic_end: int
    source_len_hdr_start: int
    source_len_hdr_end: int
    source_payload_start: int
    source_payload_end: int
    selector_len_hdr_start: int
    selector_len_hdr_end: int
    selector_payload_start: int
    selector_payload_end: int
    selector_magic: str
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "OUTER_MAGIC": [self.outer_magic_start, self.outer_magic_end],
            "source_len_hdr": [self.source_len_hdr_start, self.source_len_hdr_end],
            "source_payload": [self.source_payload_start, self.source_payload_end],
            "selector_len_hdr": [self.selector_len_hdr_start, self.selector_len_hdr_end],
            "selector_payload": [self.selector_payload_start, self.selector_payload_end],
            "selector_magic": self.selector_magic,
            "total_bytes": self.total_bytes,
        }


def parse_fec_grammar_from_inner_bytes(inner_bytes: bytes) -> _FECGrammarSections | None:
    """Parse fec6 grammar; return None if header magic mismatch (graceful no-op for
    non-fec6 archives — the probe still emits null-byte indices but skips section
    bucketing).
    """
    if len(inner_bytes) < 10 or inner_bytes[:4] != _FEC6_OUTER_MAGIC:
        return None
    pos = 4
    (source_len,) = struct.unpack_from("<I", inner_bytes, pos)
    src_start = pos + 4
    src_end = src_start + source_len
    if src_end + 2 > len(inner_bytes):
        return None
    pos = src_end
    (selector_len,) = struct.unpack_from("<H", inner_bytes, pos)
    sel_start = pos + 2
    sel_end = sel_start + selector_len
    if sel_end > len(inner_bytes):
        return None
    sel_magic = inner_bytes[sel_start : sel_start + 4]
    return _FECGrammarSections(
        outer_magic_start=0,
        outer_magic_end=4,
        source_len_hdr_start=4,
        source_len_hdr_end=8,
        source_payload_start=src_start,
        source_payload_end=src_end,
        selector_len_hdr_start=src_end,
        selector_len_hdr_end=sel_start,
        selector_payload_start=sel_start,
        selector_payload_end=sel_end,
        selector_magic=sel_magic.decode("latin-1"),
        total_bytes=len(inner_bytes),
    )


def _bucket_indices_to_sections(
    null_indices: np.ndarray,
    sections: _FECGrammarSections | None,
) -> dict[str, dict[str, Any]]:
    if sections is None:
        return {
            "_note": "no FP11/FEC6 grammar detected; section bucketing skipped",
        }
    out: dict[str, dict[str, Any]] = {}
    for name, lo, hi in [
        ("OUTER_MAGIC", sections.outer_magic_start, sections.outer_magic_end),
        ("source_len_hdr", sections.source_len_hdr_start, sections.source_len_hdr_end),
        ("source_payload", sections.source_payload_start, sections.source_payload_end),
        ("selector_len_hdr", sections.selector_len_hdr_start, sections.selector_len_hdr_end),
        ("selector_payload", sections.selector_payload_start, sections.selector_payload_end),
    ]:
        in_range_mask = (null_indices >= lo) & (null_indices < hi)
        n_null_in_section = int(in_range_mask.sum())
        section_len = hi - lo
        out[name] = {
            "range": [lo, hi],
            "length_bytes": section_len,
            "n_null": n_null_in_section,
            "null_fraction_within_section": (
                n_null_in_section / section_len if section_len > 0 else 0.0
            ),
        }
    return out


def _per_axis_null_summary(grad: np.ndarray) -> dict[str, int]:
    """Per-axis zero counts (seg, pose, rate)."""
    abs_arr = np.abs(grad)
    return {
        "seg_axis_zero_count": int((abs_arr[:, 0] == 0.0).sum()),
        "pose_axis_zero_count": int((abs_arr[:, 1] == 0.0).sum()),
        "rate_axis_zero_count": int((abs_arr[:, 2] == 0.0).sum()),
    }


def _build_canonical_provenance_dict(
    *,
    anchor_path: str,
    anchor_sha256: str,
) -> dict[str, Any]:
    """Per Catalog #323 canonical Provenance — predicted/observability-only."""
    return {
        "artifact_kind": "PREDICTED_FROM_MODEL",
        "model_id": "null_byte_master_gradient_probe.v1",
        "inputs_sha256": anchor_sha256,
        "measurement_axis": "[predicted]",
        "hardware_substrate": "unknown",
        "evidence_grade": "predicted",
        "captured_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "canonical_helper_invocation": "tools/probe_null_byte_master_gradient.py",
        "source_anchor_path": anchor_path,
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
    }


def probe_null_bytes(
    *,
    grad: np.ndarray,
    epsilon: float = 1e-9,
    inner_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Compute null-byte indices + per-section breakdown.

    Accepts either:
      * shape ``(N_bytes, 3)`` aggregate-per-byte tensor (canonical fec6
        OP3-V3 anchor format), OR
      * shape ``(N_bytes, N_pairs, 3)`` per-pair tensor — collapsed to
        per-byte via max-over-pairs before threshold.

    A "null byte" is one whose maximum absolute score-axis sensitivity is
    below ``epsilon`` across ALL 3 axes (seg / pose / rate). Rate axis on
    fp32 fec6 tensors is uniformly zero by underflow (rate coefficient
    ~6.66e-7 × per-byte gradient underflows fp32), so on those tensors
    the effective predicate is "seg == 0 AND pose == 0".
    """
    if grad.ndim == 3:
        # Per-pair: collapse to per-byte via max-abs over pairs axis
        if grad.shape[-1] != 3:
            raise ValueError(
                f"per-pair grad must have shape (N_bytes, N_pairs, 3); got {grad.shape}"
            )
        per_byte_per_axis = np.abs(grad).max(axis=1)  # (N_bytes, 3)
    elif grad.ndim == 2:
        if grad.shape[-1] != 3:
            raise ValueError(
                f"aggregate grad must have shape (N_bytes, 3); got {grad.shape}"
            )
        per_byte_per_axis = np.abs(grad)  # (N_bytes, 3)
    else:
        raise ValueError(f"grad must be 2D or 3D ndarray; got ndim={grad.ndim}")

    n_total = int(per_byte_per_axis.shape[0])
    max_per_byte = per_byte_per_axis.max(axis=1)
    null_mask = max_per_byte < epsilon
    null_indices = np.nonzero(null_mask)[0].astype(np.int64)
    null_fraction = float(null_indices.size) / max(n_total, 1)

    per_axis = _per_axis_null_summary(per_byte_per_axis)

    sections = (
        parse_fec_grammar_from_inner_bytes(inner_bytes) if inner_bytes is not None else None
    )
    section_breakdown = _bucket_indices_to_sections(null_indices, sections)

    return {
        "n_total_bytes": n_total,
        "n_null_bytes": int(null_indices.size),
        "null_fraction": null_fraction,
        "epsilon": float(epsilon),
        "per_axis_zero_counts": per_axis,
        "section_breakdown": section_breakdown,
        "grammar_detected": sections.to_dict() if sections is not None else None,
        "null_indices": null_indices,
    }


def _write_outputs(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    anchor_path: str,
    anchor_sha256: str,
    archive_zip_path: str | None,
    inner_member_name: str | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices_path = output_dir / "null_byte_indices.npy"
    summary_path = output_dir / "null_byte_summary.json"
    np.save(indices_path, summary["null_indices"])
    json_summary = {k: v for k, v in summary.items() if k != "null_indices"}
    json_summary["anchor_path"] = anchor_path
    json_summary["anchor_sha256"] = anchor_sha256
    json_summary["archive_zip_path"] = archive_zip_path
    json_summary["inner_member_name"] = inner_member_name
    json_summary["null_byte_indices_path"] = str(indices_path)
    json_summary["provenance"] = _build_canonical_provenance_dict(
        anchor_path=anchor_path, anchor_sha256=anchor_sha256
    )
    json_summary["score_claim"] = False
    json_summary["promotable"] = False
    json_summary["axis_tag"] = "[predicted]"
    json_summary["schema"] = "null_byte_master_gradient_probe_v1"
    summary_path.write_text(json.dumps(json_summary, indent=2, sort_keys=True), encoding="utf-8")
    return indices_path, summary_path


def _load_inner_member_bytes(
    archive_zip_path: Path, inner_member_name: str | None
) -> bytes | None:
    """Extract the canonical inner member; auto-detect for single-member archives."""
    try:
        with zipfile.ZipFile(archive_zip_path) as zf:
            names = [info.filename for info in zf.infolist()]
            if inner_member_name is None:
                if len(names) == 1:
                    chosen = names[0]
                else:
                    return None
            else:
                if inner_member_name not in names:
                    return None
                chosen = inner_member_name
            with zf.open(chosen) as f:
                return f.read()
    except (zipfile.BadZipFile, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe master-gradient null-space bytes (Catalog #318 typed-spec discipline; observability-only)"
    )
    parser.add_argument("--anchor", type=Path, required=True, help=".npy master-gradient anchor")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="null-byte threshold")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=None,
        help="optional canonical archive.zip for section bucketing",
    )
    parser.add_argument(
        "--inner-member-name",
        type=str,
        default=None,
        help="inner ZIP member name (default: auto-detect if archive has 1 member)",
    )
    args = parser.parse_args(argv)

    if not args.anchor.is_file():
        print(f"[null-byte-probe] FATAL: anchor not found: {args.anchor}", file=sys.stderr)
        return 1
    grad = np.load(args.anchor)
    anchor_sha256 = hashlib.sha256(args.anchor.read_bytes()).hexdigest()

    inner_bytes: bytes | None = None
    if args.archive_zip is not None:
        if not args.archive_zip.is_file():
            print(
                f"[null-byte-probe] WARNING: archive.zip not found at {args.archive_zip}; skipping section bucketing",
                file=sys.stderr,
            )
        else:
            inner_bytes = _load_inner_member_bytes(args.archive_zip, args.inner_member_name)
            if inner_bytes is None:
                print(
                    f"[null-byte-probe] WARNING: could not extract inner member from {args.archive_zip}; skipping section bucketing",
                    file=sys.stderr,
                )

    summary = probe_null_bytes(grad=grad, epsilon=args.epsilon, inner_bytes=inner_bytes)
    indices_path, summary_path = _write_outputs(
        output_dir=args.output_dir,
        summary=summary,
        anchor_path=str(args.anchor),
        anchor_sha256=anchor_sha256,
        archive_zip_path=str(args.archive_zip) if args.archive_zip is not None else None,
        inner_member_name=args.inner_member_name,
    )
    # Loud, axis-honest summary on stderr (operator-facing)
    print(
        f"[null-byte-probe] [predicted] anchor={args.anchor.name} "
        f"sha={anchor_sha256[:12]}... n_total={summary['n_total_bytes']} "
        f"n_null={summary['n_null_bytes']} ({summary['null_fraction']*100:.2f}%) "
        f"epsilon={args.epsilon}",
        file=sys.stderr,
    )
    print(f"[null-byte-probe] wrote {indices_path}", file=sys.stderr)
    print(f"[null-byte-probe] wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
