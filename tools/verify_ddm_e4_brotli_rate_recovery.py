#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify E4 Brotli rate recovery, typed-tag cost, and section consumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization import ddm_runtime_exporter as exporter  # noqa: E402
from tac.optimization import ddm_runtime_receiver as receiver  # noqa: E402
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    TypedStreamTag,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)

SCORE_BYTE_DUAL = 25 / 37_545_489
PROJECTED_RECOVERY_BYTES = 95_837


class E4VerificationError(ValueError):
    """An E4 archive, dependency, or rate-custody assertion failed closed."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise E4VerificationError(f"expected JSON object: {path}")
    return value


def _archive_path(receipt: dict[str, Any]) -> Path:
    path = (REPO_ROOT / receipt["archive"]["path"]).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise E4VerificationError("archive path escaped repository") from exc
    return path


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != exporter.EXPECTED_MEMBERS:
            raise E4VerificationError("runtime member order changed")
        return {name: archive.read(name) for name in archive.namelist()}


def _member_sizes(receipt: dict[str, Any]) -> dict[str, int]:
    rows = receipt["archive"]["member_homes"]
    sizes = {row["member"]: int(row["member_payload_range"]["bytes"]) for row in rows if row["member"] is not None}
    sizes["container_overhead"] = int(receipt["archive"]["bytes"]) - sum(sizes.values())
    return sizes


def _score_delta(delta_bytes: int) -> float:
    return SCORE_BYTE_DUAL * delta_bytes


def _delta_row(name: str, before: int, after: int) -> dict[str, Any]:
    delta = after - before
    return {
        "after_bytes": after,
        "before_bytes": before,
        "delta_bytes": delta,
        "delta_score_units": _score_delta(delta),
        "section": name,
    }


def _verify_tagged_archive(
    *,
    tagged_receipt: dict[str, Any],
    untagged_receipt: dict[str, Any],
) -> dict[str, Any]:
    tagged_path = _archive_path(tagged_receipt)
    untagged_path = _archive_path(untagged_receipt)
    tagged = _members(tagged_path)
    _members(untagged_path)
    manifest = json.loads(tagged["manifest.json"])
    tags = []
    for row in manifest["sections"]:
        tag = TypedStreamTag.from_dict(row["typed_stream_tag"])
        if tag.counted_bytes != row["bytes"]:
            raise E4VerificationError("typed tag did not reconcile section bytes")
        tags.append(tag.to_dict())
        del row["typed_stream_tag"]
    stripped_manifest = rfc8785_canonicalize(manifest)
    stripped_members = {**tagged, "manifest.json": stripped_manifest}
    stripped_archive = exporter._deterministic_zip(stripped_members)
    if stripped_archive != untagged_path.read_bytes():
        raise E4VerificationError("tag stripping did not reproduce the exact untagged control archive")
    overhead = len(tagged_path.read_bytes()) - len(stripped_archive)
    if overhead != len(tagged["manifest.json"]) - len(stripped_manifest):
        raise E4VerificationError("typed-tag overhead escaped manifest byte home")
    return {
        "archive_overhead_bytes": overhead,
        "archive_overhead_score_units": _score_delta(overhead),
        "exact_stripped_archive_sha256": hashlib.sha256(stripped_archive).hexdigest(),
        "exact_untagged_control_match": True,
        "manifest_overhead_bytes": len(tagged["manifest.json"]) - len(stripped_manifest),
        "tags": tags,
    }


def _verify_consumption(
    brotli_members: dict[str, bytes],
    fallback_members: dict[str, bytes],
) -> dict[str, Any]:
    rows = []
    for name, kind in (
        ("base/chart.ddb", 0),
        ("semantic/composed.dds", 1),
    ):
        brotli_raw, brotli_shape = receiver._parse_blob(
            brotli_members[name],
            expected_kind=kind,
            label=f"brotli:{name}",
        )
        fallback_raw, fallback_shape = receiver._parse_blob(
            fallback_members[name],
            expected_kind=kind,
            label=f"fallback:{name}",
        )
        if brotli_raw != fallback_raw or brotli_shape != fallback_shape:
            raise E4VerificationError(f"decoded coder identity changed: {name}")
        tampered = bytearray(brotli_members[name])
        tampered[-1] ^= 1
        try:
            receiver._parse_blob(
                bytes(tampered),
                expected_kind=kind,
                label=f"tampered:{name}",
            )
        except receiver.ReceiverError:
            tamper_refused = True
        else:
            tamper_refused = False
        if not tamper_refused:
            raise E4VerificationError(f"tampered section was a no-op: {name}")
        rows.append(
            {
                "brotli_member_sha256": hashlib.sha256(brotli_members[name]).hexdigest(),
                "decoded_raw_bytes": len(brotli_raw),
                "decoded_raw_sha256": hashlib.sha256(brotli_raw).hexdigest(),
                "decoded_shape": list(brotli_shape),
                "fallback_member_sha256": hashlib.sha256(fallback_members[name]).hexdigest(),
                "member": name,
                "terminal_coded_byte_perturbation": "REFUSED_NOT_NOOP",
            }
        )
    return {
        "discipline": "#417 counted-section no-op detector",
        "raw_identity_across_coders": True,
        "sections": rows,
        "verdict": "CONSUMED_EXACT_AND_TAMPER_REFUSED",
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    e3 = _load_json(args.e3_receipt)
    brotli_untagged = _load_json(args.brotli_untagged_receipt)
    brotli_tagged = _load_json(args.brotli_tagged_receipt)
    fallback_untagged = _load_json(args.fallback_untagged_receipt)
    fallback_tagged = _load_json(args.fallback_tagged_receipt)
    receipts = (e3, brotli_untagged, brotli_tagged, fallback_untagged, fallback_tagged)
    if {row["output_identity"]["sha256"] for row in receipts} != {
        "4c553508b0bf92ccdc137e215799ae30a346b58e0617e5156441a7929302b4f1"
    }:
        raise E4VerificationError("E3/E4 output identity changed")
    if brotli_tagged["runtime"]["coder"]["selected"] != exporter.BROTLI_Q11_CODER:
        raise E4VerificationError("primary did not select Brotli Q11")
    if brotli_tagged["runtime"]["dependencies"] != ["torch", "brotli"]:
        raise E4VerificationError("primary dependency declaration changed")
    if fallback_tagged["runtime"]["coder"] != {
        "codec_id": 2,
        "fallback_trigger": "ImportError",
        "primary": "brotli_q11",
        "selected": "lzma1_raw_d1m_lc3_lp0_pb2",
    }:
        raise E4VerificationError("fallback did not prove exact ImportError custody")
    if fallback_tagged["runtime"]["dependencies"] != ["torch"]:
        raise E4VerificationError("fallback dependency declaration changed")

    brotli_tag_proof = _verify_tagged_archive(
        tagged_receipt=brotli_tagged,
        untagged_receipt=brotli_untagged,
    )
    fallback_tag_proof = _verify_tagged_archive(
        tagged_receipt=fallback_tagged,
        untagged_receipt=fallback_untagged,
    )
    if brotli_tag_proof["archive_overhead_bytes"] != fallback_tag_proof["archive_overhead_bytes"]:
        raise E4VerificationError("typed-tag overhead changed across coders")

    before = _member_sizes(e3)
    after = _member_sizes(brotli_tagged)
    rows = [
        _delta_row(name, before[name], after[name])
        for name in (
            "manifest.json",
            "base/chart.ddb",
            "semantic/composed.dds",
            "container_overhead",
        )
    ]
    rows.append(
        {
            **_delta_row(
                "typed_stream_tag_overhead",
                0,
                brotli_tag_proof["archive_overhead_bytes"],
            ),
            "accounting_role": "manifest.json decomposition; do not sum twice",
        }
    )
    archive_row = _delta_row(
        "archive.zip",
        int(e3["archive"]["bytes"]),
        int(brotli_tagged["archive"]["bytes"]),
    )
    if archive_row["delta_bytes"] != sum(row["delta_bytes"] for row in rows[:4]):
        raise E4VerificationError("section deltas did not reconcile archive delta")

    fallback_sizes = _member_sizes(fallback_tagged)
    coder_rows = [
        _delta_row(name, fallback_sizes[name], after[name])
        for name in (
            "manifest.json",
            "base/chart.ddb",
            "semantic/composed.dds",
            "container_overhead",
        )
    ]
    coder_archive_row = _delta_row(
        "archive.zip",
        int(fallback_tagged["archive"]["bytes"]),
        int(brotli_tagged["archive"]["bytes"]),
    )
    if coder_archive_row["delta_bytes"] != sum(row["delta_bytes"] for row in coder_rows):
        raise E4VerificationError("coder-only section deltas did not reconcile")

    correction = archive_row["delta_bytes"] - (-PROJECTED_RECOVERY_BYTES)
    return {
        "archive_custody": {
            "after_brotli_tagged": brotli_tagged["archive"],
            "before_e3_lzma1_untagged": e3["archive"],
            "fallback_e4_lzma1_tagged": fallback_tagged["archive"],
        },
        "coder_only_tagged_ab": {
            "archive": coder_archive_row,
            "sections": coder_rows,
        },
        "dependency_contract": {
            "fallback": {
                "dependencies": ["torch"],
                "fallback_trigger": "ImportError",
                "selected": exporter.E3_LZMA1_CODER,
            },
            "primary": {
                "dependencies": ["torch", "brotli"],
                "selected": exporter.BROTLI_Q11_CODER,
            },
        },
        "distortion_trade": {
            "present": False,
            "reason": "lossless coder substitution; decoded raw payload is byte-identical",
        },
        "evidence_axis": "[macOS-CPU local byte-custody advisory]",
        "metric_law": {
            "contract": "tac.optimization.ddm_min_description_contract",
            "equation": "25*archive_bytes/37_545_489",
            "id": "real_coder_archive_bytes_contest_units_v1",
            "score_byte_dual": SCORE_BYTE_DUAL,
        },
        "pointer": {
            "contest_cpu": 0.1910828242,
            "moved": False,
        },
        "projection_correction": {
            "actual_recovery_bytes": -archive_row["delta_bytes"],
            "actual_recovery_score_units": -archive_row["delta_score_units"],
            "correction_bytes": correction,
            "correction_score_units": _score_delta(correction),
            "projected_recovery_bytes": PROJECTED_RECOVERY_BYTES,
        },
        "rate_recovery_vs_e3": {
            "archive": archive_row,
            "sections": rows,
        },
        "research_only": True,
        "schema": "ddm_e4_brotli_rate_recovery_receipt.v1",
        "score_claim": False,
        "section_consumption": _verify_consumption(
            _members(_archive_path(brotli_tagged)),
            _members(_archive_path(fallback_tagged)),
        ),
        "typed_stream_tag_ab": {
            "brotli": brotli_tag_proof,
            "decision": "KEEP_AND_VERSION_BUMP_E4_MANIFEST",
            "fallback": fallback_tag_proof,
            "receiver_policy": "TAG_ABSENT_OR_MALFORMED_REFUSED",
        },
        "verdict": "PASS_E4_BROTLI_RATE_RECOVERY_ADVISORY_ONLY",
        "verdict_scope": (
            "INSTANCE:E3 PA1 composed payload x Brotli 1.2.0 Q11 x exact E3 "
            "raw-LZMA1 fallback x typed E4 manifest. Local byte custody only; "
            "locked upstream scorer receipts are separate and pointer remains unmoved."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-receipt", type=Path, required=True)
    parser.add_argument("--brotli-untagged-receipt", type=Path, required=True)
    parser.add_argument("--brotli-tagged-receipt", type=Path, required=True)
    parser.add_argument("--fallback-untagged-receipt", type=Path, required=True)
    parser.add_argument("--fallback-tagged-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args)
    exporter._publish_or_verify(
        args.output.resolve(),
        rfc8785_canonicalize(result) + b"\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
