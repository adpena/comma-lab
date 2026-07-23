#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-derive the bounded DDM M6 22,645-byte gap-closure receipt.

This command performs no scorer run, launch, dispatch, or config mutation.  It
hash-verifies the M4 and g2g2 receipts, parses the exact 177,169-byte archive,
and proves that the implicit-framing receiver reconstructs the source archive
byte-for-byte.  The resulting rate credit is structural and score-preserving by
transitive byte identity; all distortion values are reused historical custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.canonical_equations.ddm_m6_gap_closure_20260723 import (  # noqa: E402
    BASELINE_ARCHIVE_BYTES,
    DECISIVE_GAP_BYTES,
    STRICT_SUB015_CAP_BYTES,
    PoolCredit,
    compose_gap_closure,
)
from tac.packet_compiler.ddm_m6_implicit_fp11 import (  # noqa: E402
    LEGACY_GENERIC_FRAMING_BYTES,
    ImplicitFP11Parts,
    pack_implicit_member,
    reconstruct_legacy_member,
    split_legacy_member,
    stored_archive_bytes,
    unpack_implicit_member,
)

M4_RECEIPT = Path(
    ".omx/research/ddm_m4_rate_floor_einstein_avenue_20260723_receipt.json"
)
DEFAULT_G2G2_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/g2g2_joint_multichart_20260721/"
    "measurement_20260721T172244Z/receipt.json"
)
EXPECTED_G2G2_FILE_SHA256 = (
    "928d3cd74cc92ef52aa9f821229ada12fbf4c3e9dad772e8a76adffcfcfcb078"
)


class DerivationError(RuntimeError):
    """Raised when custody or a gap-closure invariant drifts."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DerivationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DerivationError(f"top-level object required: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_archive(path: Path) -> tuple[bytes, bytes]:
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != 1 or infos[0].filename != "x":
            raise DerivationError("source archive must contain only member 'x'")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise DerivationError("source member must be ZIP_STORED")
        member = archive.read(infos[0])
    return archive_bytes, member


def _flip_last(value: bytes) -> bytes:
    if not value:
        raise DerivationError("cannot mutation-test an empty retained field")
    return value[:-1] + bytes((value[-1] ^ 1,))


def _consumption_proof(parts: ImplicitFP11Parts, legacy_member: bytes) -> dict[str, Any]:
    """Prove every retained section affects reconstructed receiver input."""

    fields = (
        "decoder_section",
        "latent_section",
        "sidecar",
        "selector",
        "dqs1_tail",
    )
    rows: list[dict[str, Any]] = []
    for field in fields:
        mutated = replace(parts, **{field: _flip_last(getattr(parts, field))})
        compact = pack_implicit_member(mutated)
        reconstructed = reconstruct_legacy_member(unpack_implicit_member(compact))
        if reconstructed == legacy_member:
            raise DerivationError(f"retained field is not consumed: {field}")
        roundtrip = split_legacy_member(reconstructed)
        changed_fields = [
            name for name in fields if getattr(roundtrip, name) != getattr(parts, name)
        ]
        if changed_fields != [field]:
            raise DerivationError(
                f"mutation isolation failed for {field}: {changed_fields}"
            )
        rows.append(
            {
                "field": field,
                "mutation": "last_byte_xor_1",
                "reconstructed_legacy_member_changed": True,
                "only_target_field_changed": True,
            }
        )
    return {
        "method": "mutate each retained field then reconstruct and strict-parse legacy member",
        "all_retained_fields_consumed": True,
        "rows": rows,
    }


def _g2g2_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "realization_g2g2_joint_multichart_receipt.v1":
        raise DerivationError("unexpected g2g2 receipt schema")
    if receipt.get("verdict") != "MEASURED_G2G2_RATE_BREAK_EVEN_STOP_FAMILY_OPEN":
        raise DerivationError("unexpected g2g2 verdict")
    authority = receipt["authority"]
    if authority["score_claim"] is not False or authority["promotion_eligible"] is not False:
        raise DerivationError("g2g2 authority boundary drift")
    oracle = receipt["D2_hard_receiver_oracle"]
    if oracle["admitted_pair_count"] != 0 or oracle["minimum_admitted_by_pair"]:
        raise DerivationError("g2g2 admitted-packet count drift")

    curves = [
        row
        for pair in oracle["pair_summaries"]
        for row in pair["measured_k_curve"]
    ]
    if not curves:
        raise DerivationError("g2g2 receipt has no measured prefixes")
    predicates = [row["admission_predicates"] for row in curves]
    if not all(
        predicate["counted_bytes"]
        and predicate["receiver_RGB"]
        and predicate["factor2_uint8_exact"]
        and predicate["double_decode"]
        for predicate in predicates
    ):
        raise DerivationError("g2g2 exact receiver predicates drifted")
    if any(row["admitted"] for row in curves):
        raise DerivationError("unexpected admitted g2g2 prefix")

    pair_indices = [row["pair_index"] for row in oracle["pair_summaries"]]
    return {
        "axis": authority["axis"],
        "base_counted_bytes": receipt["config"]["base_counted_bytes"],
        "source_vehicle_matches_177169_fp11": False,
        "pair_indices": pair_indices,
        "pair_count": len(pair_indices),
        "measured_prefix_count": len(curves),
        "all_prefixes_factor2_uint8_exact": True,
        "all_prefixes_receiver_rgb": True,
        "all_prefixes_double_decode": True,
        "admitted_pair_count": 0,
        "verdict": receipt["verdict"],
        "verdict_scope": receipt["verdict_scope"],
    }


def derive(
    repo_root: Path = REPO_ROOT,
    *,
    g2g2_receipt_path: Path = DEFAULT_G2G2_RECEIPT,
) -> dict[str, Any]:
    """Return the deterministic, pool-aware M6 receipt."""

    m4_path = repo_root / M4_RECEIPT
    m4 = _load_json(m4_path)
    if m4["rate_floor"]["decisive_gap_bytes"] != DECISIVE_GAP_BYTES:
        raise DerivationError("M4 decisive gap drift")
    if (
        m4["rate_floor"]["audited_relaxed_receiver_floor"]["archive_bytes"]
        != BASELINE_ARCHIVE_BYTES
    ):
        raise DerivationError("M4 relaxed floor drift")
    if (
        m4["rate_floor"]["strict_sub015_at_settled_c1"]["max_archive_bytes"]
        != STRICT_SUB015_CAP_BYTES
    ):
        raise DerivationError("M4 strict cap drift")

    relaxed = m4["rate_floor"]["audited_relaxed_receiver_floor"]
    source_archive_path = Path(
        _load_json(repo_root / Path(m4["source_receipts"]["joint_optimum_575"]["path"]))[
            "candidate_custody"
        ]["archive_path"]
    )
    source_archive, source_member = _read_source_archive(source_archive_path)
    if len(source_archive) != BASELINE_ARCHIVE_BYTES:
        raise DerivationError("source archive byte count drift")
    if _sha256_bytes(source_archive) != relaxed["archive_sha256"]:
        raise DerivationError("source archive SHA-256 drift")
    if stored_archive_bytes(source_member) != source_archive:
        raise DerivationError("source ZIP metadata is not canonical M6 ZIP_STORED")

    parts = split_legacy_member(source_member)
    compact_member = pack_implicit_member(parts)
    parsed_compact = unpack_implicit_member(compact_member)
    reconstructed_member = reconstruct_legacy_member(parsed_compact)
    if reconstructed_member != source_member:
        raise DerivationError("implicit adapter does not reconstruct source member")
    reconstructed_archive = stored_archive_bytes(reconstructed_member)
    if reconstructed_archive != source_archive:
        raise DerivationError("implicit adapter does not reconstruct source archive")
    compact_archive = stored_archive_bytes(compact_member)
    archive_reduction = len(source_archive) - len(compact_archive)
    if archive_reduction != LEGACY_GENERIC_FRAMING_BYTES:
        raise DerivationError("implicit framing reduction drift")

    if _sha256_file(g2g2_receipt_path) != EXPECTED_G2G2_FILE_SHA256:
        raise DerivationError("g2g2 receipt file SHA-256 drift")
    g2g2 = _load_json(g2g2_receipt_path)
    g2g2_summary = _g2g2_summary(g2g2)

    m4_lattice = m4["integer_lattice"]
    if m4_lattice["unrecovered_debt_is_measured_recoverable_gain"] is not False:
        raise DerivationError("M4 lattice debt authority drift")
    if m4["ker_A"]["measured_counted_bytes_hideable_for_free"] != 0:
        raise DerivationError("M4 ker(A) byte credit drift")

    pool_credits = (
        PoolCredit(
            pool_id="P_REALIZE",
            lever_ids=("g2g2_integer_lattice_native",),
            joint_reduction_bytes=0,
            receiver_closed=True,
            evidence_scope=(
                "g2g2 already enforces exact factor2 uint8 receiver replay, is a "
                "121128-byte different vehicle, and admits 0/6 pairs; the n16 "
                "absolute-write -1.4% result is unmet score debt, not byte credit"
            ),
        ),
        PoolCredit(
            pool_id="P_TEMPORAL_DESCRIPTION",
            lever_ids=("rule118_implicit_fp11_ctxr_framing",),
            joint_reduction_bytes=archive_reduction,
            receiver_closed=True,
            evidence_scope=(
                "compact archive reconstructs the exact source member and exact source "
                "archive; only FP11/CTXR fixed magic, fixed version, and derived source "
                "length move into generic receiver code"
            ),
        ),
        PoolCredit(
            pool_id="P_NULL_GAUGE",
            lever_ids=("ker(A)_payload_hiding",),
            joint_reduction_bytes=0,
            receiver_closed=True,
            evidence_scope=(
                "80.6742315223% mathematical nullity does not identify any removable "
                "parser-consumed counted payload; measured incremental byte credit is zero"
            ),
        ),
    )
    closure = compose_gap_closure(
        pool_credits,
        final_archive_bytes=len(compact_archive),
        final_same_artifact_receiver_closed=True,
    )
    if closure.sub015_reached:
        raise DerivationError("unexpected sub-0.15 closure; delegated no-dispatch path drifted")

    section_rows = [
        {
            "section": name,
            "bytes": len(getattr(parts, field)),
            "sha256": _sha256_bytes(getattr(parts, field)),
            "rule_118_class": "COUNTED_VIDEO_DERIVED",
        }
        for name, field in (
            ("decoder_section", "decoder_section"),
            ("latent_section", "latent_section"),
            ("sidecar", "sidecar"),
            ("selector", "selector"),
            ("dqs1_tail", "dqs1_tail"),
        )
    ]

    return {
        "schema": "ddm_m6_close_22645_byte_gap_receipt.v1",
        "generated_at_utc": "2026-07-23T11:06:50Z",
        "lane_id": "ddm_m6_close_22645_byte_gap",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "execution": {
            "cost_usd": 0,
            "new_scorer_run": False,
            "new_launch": False,
            "config_changed": False,
            "exact_eval_dispatched": False,
            "method": "bounded deterministic packet parse-back and receipt re-derivation",
        },
        "source_receipts": {
            "m4": {
                "path": str(M4_RECEIPT),
                "bytes": m4_path.stat().st_size,
                "sha256": _sha256_file(m4_path),
            },
            "g2g2": {
                "path": str(g2g2_receipt_path),
                "bytes": g2g2_receipt_path.stat().st_size,
                "file_sha256": _sha256_file(g2g2_receipt_path),
                "canonical_receipt_sha256": g2g2["receipt_sha256"],
            },
        },
        "source_archive": {
            "path": str(source_archive_path),
            "bytes": len(source_archive),
            "sha256": _sha256_bytes(source_archive),
            "member_bytes": len(source_member),
            "member_sha256": _sha256_bytes(source_member),
            "zip_overhead_bytes": len(source_archive) - len(source_member),
            "historical_d_seg": relaxed["d_seg"],
            "historical_d_pose": relaxed["d_pose"],
            "historical_axis": relaxed["evidence_axis"],
            "historical_score_reused_not_remeasured": True,
        },
        "rule_118_receiver_adapter": {
            "status": "MEASURED_TRANSITIVE_EXACT_RECEIVER_PARSEBACK",
            "compact_format": (
                "3xu24 section lengths | decoder | latent | sidecar | "
                "u16 selector length | selector | DQS1 tail"
            ),
            "migrated_to_generic_receiver_code": [
                {"field": "FP11 magic", "bytes": 4},
                {"field": "derived source length", "bytes": 4},
                {"field": "CTXR magic", "bytes": 4},
                {"field": "fixed CTXR version 1", "bytes": 1},
            ],
            "retained_counted_sections": section_rows,
            "retained_boundary_bytes": {
                "three_u24_section_lengths": 9,
                "u16_selector_length": 2,
            },
            "compact_member_bytes": len(compact_member),
            "compact_member_sha256": _sha256_bytes(compact_member),
            "compact_archive_bytes": len(compact_archive),
            "compact_archive_sha256": _sha256_bytes(compact_archive),
            "archive_reduction_bytes": archive_reduction,
            "reconstructed_member_byte_identical": True,
            "reconstructed_archive_byte_identical": True,
            "reconstructed_archive_sha256": _sha256_bytes(reconstructed_archive),
            "original_receiver_input_therefore_byte_identical": True,
            "submission_runtime_staged": False,
            "candidate_archive_written": False,
            "exact_eval_required_for_13_byte_structural_identity": False,
            "consumption_proof": _consumption_proof(parts, source_member),
            "scope": (
                "fixed framing only; no video-derived payload, section boundary, ZIP "
                "overhead, decoder, latent, sidecar, selector, or DQS1 byte is free"
            ),
        },
        "lever_attribution": [
            {
                "lever": "ker(A) payload hiding",
                "pool_id": "P_NULL_GAUGE",
                "measured_reduction_bytes": 0,
                "authority": "MEASURED_ZERO",
                "scope": "nullity is geometry, not an archive-byte fraction",
            },
            {
                "lever": "rule-118 free migration",
                "pool_id": "P_TEMPORAL_DESCRIPTION",
                "measured_reduction_bytes": archive_reduction,
                "authority": "MEASURED_TRANSITIVE_EXACT_RECEIVER_PARSEBACK",
                "scope": "13 fixed/derived framing bytes only",
            },
            {
                "lever": "g2g2 integer lattice native",
                "pool_id": "P_REALIZE",
                "measured_reduction_bytes": 0,
                "authority": "MEASURED_ZERO_PREMISE_CONFLATION",
                "scope": (
                    "g2g2 factor2 uint8 is already exact; n16 -1.4% is a different "
                    "absolute-write formulation and not a recoverable byte measurement"
                ),
            },
        ],
        "g2g2_scope_check": g2g2_summary,
        "pool_aware_composition": {
            "law": (
                "one joint credit per non-additive pool; final admitted Y is the "
                "same-artifact receiver-closed archive delta, never singleton addition"
            ),
            "pool_credits": [
                {
                    **asdict(credit),
                    "lever_ids": list(credit.lever_ids),
                }
                for credit in pool_credits
            ],
            "result": asdict(closure),
            "Y_bytes": closure.admitted_reduction_bytes,
            "residual_gap_bytes": closure.residual_gap_bytes,
            "sub015_reached": closure.sub015_reached,
        },
        "candidate_and_dispatch": {
            "byte_close_candidate_flagged": False,
            "r6_exact_eval_flagged": False,
            "reason": "Y=13 is below the 22645-byte gap; no candidate archive was written",
            "maximum_authorized_future_exact_eval_cost_usd": 20,
        },
        "main_landing_review": {
            "required": True,
            "focus": [
                "confirm the 13-byte migration contains only fixed or derived framing",
                "confirm compact-to-legacy reconstruction is archive-byte-identical",
                "confirm g2g2 exact-factor2 custody falsifies the proposed -1.4% transfer",
                "confirm ker(A) remains zero byte credit and pools are not naively summed",
                "confirm no exact eval, launch, score, pointer, or promotion claim occurred",
            ],
        },
        "verdict": "SUB015_NOT_REACHED_Y13_RESIDUAL22632",
        "verdict_scope": (
            "the exact 177169-byte FP11/CTXR #575 vehicle and three delegated levers; "
            "not a global MDL or family-impossibility result"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--g2g2-receipt",
        type=Path,
        default=DEFAULT_G2G2_RECEIPT,
    )
    parser.add_argument(
        "--verify-receipt",
        type=Path,
        help="Compare the deterministic derivation with a committed JSON receipt.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = derive(
        args.repo_root.resolve(),
        g2g2_receipt_path=args.g2g2_receipt.resolve(),
    )
    if args.verify_receipt is not None:
        expected = _load_json(args.verify_receipt)
        if result != expected:
            print("RECEIPT_MISMATCH", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "receipt": str(args.verify_receipt),
                    "receipt_sha256": _sha256_file(args.verify_receipt),
                    "Y_bytes": result["pool_aware_composition"]["Y_bytes"],
                    "residual_gap_bytes": result["pool_aware_composition"][
                        "residual_gap_bytes"
                    ],
                    "sub015_reached": result["pool_aware_composition"][
                        "sub015_reached"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
