#!/usr/bin/env python3
"""Profile lossless PR106 latent-sidecar recode candidates.

This is a byte-closed planning tool. It decodes a PR106 latent-sidecar payload
into canonical ``(dim, delta_q)`` correction arrays, tries lossless alternative
byte grammars, proves each candidate decodes back to the same arrays, and emits
a no-score manifest. It does not build a candidate archive and it never claims
score movement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    decode_brotli_dim_delta_sidecar_payload,
    decode_pr101_ranked_sidecar_payload_to_dim_delta,
    lossless_pr106_sidecar_recode_candidates,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
    sha256_hex,
)

TOOL = "tools/profile_pr106_latent_sidecar_recode.py"
SCHEMA = "pr106_latent_sidecar_recode_profile_v1"
ARCHIVE_BYTES_DENOMINATOR = 37_545_489


def load_sidecar_source(args: argparse.Namespace) -> tuple[bytes, dict[str, Any]]:
    if args.sidecar_bin is not None:
        payload = args.sidecar_bin.read_bytes()
        return payload, {
            "mode": "sidecar_bin",
            "path": str(args.sidecar_bin),
            "sidecar_payload_bytes": len(payload),
            "sidecar_payload_sha256": sha256_hex(payload),
            "sidecar_format_id": "0x01",
            "framing_meta_bytes": 0,
            "framing_meta_sha256": None,
        }

    archive_bytes = args.sidecar_archive.read_bytes()
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=args.member_name,
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    source = {
        "mode": "sidecar_archive",
        "path": str(args.sidecar_archive),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_hex(archive_bytes),
        "member_name": member.name,
        "member_bytes": len(member.payload),
        "member_sha256": sha256_hex(member.payload),
        "sidecar_format_id": f"0x{packet.format_id:02X}",
        "sidecar_kind": packet.sidecar_kind,
        "pr106_inner_payload_bytes": len(packet.pr106_bytes),
        "pr106_inner_payload_sha256": sha256_hex(packet.pr106_bytes),
        "sidecar_payload_bytes": len(packet.sidecar_payload),
        "sidecar_payload_sha256": sha256_hex(packet.sidecar_payload),
        "framing_meta_bytes": 0 if packet.framing_meta is None else len(packet.framing_meta),
        "framing_meta_sha256": None
        if packet.framing_meta is None
        else sha256_hex(packet.framing_meta),
    }
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        return packet.sidecar_payload, source
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 archive has no framing_meta")
        # Re-encode to the canonical 0x01 byte source for candidate comparison;
        # the decoded arrays remain the semantic source of truth.
        dims, deltas = decode_pr101_ranked_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            packet.framing_meta,
        )
        current = next(
            candidate
            for candidate in lossless_pr106_sidecar_recode_candidates(dims, deltas)
            if candidate.name == "current_pr100_dim_delta_brotli_q11"
        )
        source["semantic_source_format"] = "pr101_ranked_no_op_decoded_then_profiled"
        return current.encoded_bytes, source
    raise ValueError(f"unsupported sidecar format_id=0x{packet.format_id:02X}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sidecar_payload, source = load_sidecar_source(args)
    dims, deltas = decode_brotli_dim_delta_sidecar_payload(sidecar_payload)
    current_charged_bytes = int(source["sidecar_payload_bytes"]) + int(
        source.get("framing_meta_bytes") or 0
    )
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    rows = []
    for candidate in candidates:
        applicable = bool(candidate.encoded_bytes)
        charged = candidate.charged_bytes if applicable else None
        delta_bytes = None if charged is None else charged - current_charged_bytes
        rows.append(
            {
                "name": candidate.name,
                "applicable": applicable,
                "charged_sidecar_bytes": charged,
                "delta_bytes_vs_current_charged_sidecar": delta_bytes,
                "rate_score_delta_if_runtime_consumed": None
                if delta_bytes is None
                else 25.0 * delta_bytes / ARCHIVE_BYTES_DENOMINATOR,
                "sidecar_format_id": None
                if candidate.sidecar_format_id is None
                else f"0x{candidate.sidecar_format_id:02X}",
                "encoded_payload_bytes": len(candidate.encoded_bytes),
                "encoded_payload_sha256": sha256_hex(candidate.encoded_bytes)
                if applicable
                else None,
                "framing_meta_bytes": len(candidate.framing_meta_bytes),
                "framing_meta_sha256": sha256_hex(candidate.framing_meta_bytes)
                if candidate.framing_meta_bytes
                else None,
                "runtime_decoder_implemented": candidate.runtime_decoder_implemented,
                "lossless_semantic_equivalence_proven": applicable,
                "notes": list(candidate.notes),
            }
        )
    best = next((row for row in rows if row["applicable"]), None)
    best_runtime = next(
        (row for row in rows if row["applicable"] and row["runtime_decoder_implemented"]),
        None,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "source": source,
        "semantic_arrays": {
            "n_pairs": int(dims.size),
            "n_corrected_pairs": int(((dims != 255) & (deltas != 0)).sum()),
            "n_noop_pairs": int(((dims == 255) | (deltas == 0)).sum()),
            "dim_unique": [int(value) for value in sorted(set(dims.astype(int).tolist()))],
            "delta_q_unique": [
                int(value) for value in sorted(set(deltas.astype(int).tolist()))
            ],
            "dim_sha256": sha256_hex(dims.astype("uint8").tobytes()),
            "delta_q_sha256": sha256_hex(deltas.astype("int8").tobytes()),
        },
        "current_charged_sidecar_bytes": current_charged_bytes,
        "candidate_rows": rows,
        "best_lossless_candidate": best,
        "best_runtime_consumed_candidate": best_runtime,
        "adversarial_claim_check": {
            "verdict": "planning_only_no_score_claim",
            "interpretation": (
                "Negative or positive byte deltas here are sidecar payload-rate "
                "signals only. A score claim requires a runtime that consumes the "
                "candidate grammar, a byte-closed archive, no-op proof, and exact "
                "contest eval on the emitted packet."
            ),
        },
        "dispatch_blockers": [
            "no_candidate_archive_emitted",
            "candidate_runtime_decoder_missing_for_noncurrent_rows",
            "missing_no_op_runtime_consumption_proof_for_new_grammar",
            "missing_exact_contest_eval_for_any_candidate",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    lines = [
        "# PR106 Latent Sidecar Recode Profile",
        "",
        f"- score_claim: `{str(report['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(report['ready_for_exact_eval_dispatch']).lower()}`",
        f"- source_mode: `{source.get('mode')}`",
        f"- source_path: `{source.get('path')}`",
        f"- current_charged_sidecar_bytes: `{report['current_charged_sidecar_bytes']}`",
        f"- n_pairs: `{report['semantic_arrays']['n_pairs']}`",
        f"- delta_q_unique: `{report['semantic_arrays']['delta_q_unique']}`",
        "",
        "## Candidates",
        "",
        (
            "| candidate | charged bytes | delta bytes | rate delta if consumed | "
            "runtime decoder | equivalence |"
        ),
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['charged_sidecar_bytes']} | "
            f"{row['delta_bytes_vs_current_charged_sidecar']} | "
            f"{row['rate_score_delta_if_runtime_consumed']} | "
            f"`{str(row['runtime_decoder_implemented']).lower()}` | "
            f"`{str(row['lossless_semantic_equivalence_proven']).lower()}` |"
        )
    claim = report["adversarial_claim_check"]
    lines.extend(
        [
            "",
            "## Adversarial Claim Check",
            "",
            f"- verdict: `{claim['verdict']}`",
            "",
            claim["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sidecar-bin", type=Path, help="Raw PR106 format-0x01 sidecar.bin")
    source.add_argument(
        "--sidecar-archive",
        type=Path,
        help="PR106 sidecar archive.zip containing one stored 0.bin/x member",
    )
    parser.add_argument("--member-name", default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.json_out}")
    if args.md_out is not None:
        print(f"wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
