#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert the in-repo PR110/FEC6 K16 packet into one ActionEffect row.

This consumes the packet manifest / archive manifest produced by the local
PR110 build pipeline.  It does not rerun the scorer and does not promote the
row.  The row is a selector-program reproduction surface for downstream
ActionEffect planning gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import (  # noqa: E402
    ActionEffect,
    NormalizationScope,
    append_action_effect,
    exact_delta_score,
    validate_action_effect_payload,
)

CONVERSION_SCHEMA = "tac.pr110_k16_packet_action_effect_conversion.v1"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"required integer field missing: {key}")
    return int(value)


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"required numeric field missing: {key}")
    return float(value)


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"required string field missing: {key}")
    return value


def build_pr110_k16_packet_action_effect(
    packet_manifest: Mapping[str, Any],
    *,
    packet_manifest_path: Path,
    archive_manifest: Mapping[str, Any] | None = None,
    authority: str = "[contest-CPU] pr110_fec6_k16_packet_manifest",
    consumer: str = "inverse_evaluate_candidate_queue",
) -> ActionEffect:
    """Build one non-promotional K16 selector-program ActionEffect."""

    archive = _mapping(packet_manifest.get("archive"))
    selector = _mapping(packet_manifest.get("selector"))
    proxy = _mapping(packet_manifest.get("proxy"))
    archive_manifest = archive_manifest or {}
    selector_pack = _mapping(
        archive_manifest.get("selector_pack_manifest")
        or archive.get("selector_pack_manifest")
    )

    n_pairs = _required_int(selector, "n_pairs")
    if n_pairs != 600:
        raise ValueError(f"PR110 K16 packet manifest must describe 600 pairs; got {n_pairs}")
    palette_size = _required_int(selector_pack, "palette_size")
    if palette_size != 16:
        raise ValueError(f"PR110 K16 packet manifest must describe palette_size=16; got {palette_size}")
    selector_bits = _required_int(selector_pack, "selector_code_bits_total")
    selector_payload_bytes = _required_int(selector_pack, "selector_payload_bytes")
    selector_payload_sha256 = _required_text(selector_pack, "selector_payload_sha256")
    selector_codec = _required_text(selector_pack, "compact_selector_codec")

    old_bytes = _required_int(proxy, "baseline_archive_bytes")
    new_bytes = _required_int(proxy, "candidate_archive_bytes")
    old_d_seg = _required_float(proxy, "baseline_avg_segnet_dist")
    new_d_seg = _required_float(proxy, "selected_avg_segnet_dist")
    old_d_pose = _required_float(proxy, "baseline_avg_posenet_dist")
    new_d_pose = _required_float(proxy, "selected_avg_posenet_dist")

    archive_sha256 = _required_text(archive, "sha256")
    source_archive_sha256 = _required_text(selector, "baseline_archive_sha256")
    packet_action_id = (
        "pr110_fec6_k16_global600_"
        f"{archive_sha256[:12]}_{selector_payload_sha256[:12]}"
    )
    payload_sections = (
        "fec6_fixed_huffman_k16",
        "global_k=16",
        "n_pairs=600",
        f"selector_code_bits_total={selector_bits}",
        f"selector_payload_bytes={selector_payload_bytes}",
        f"selector_codec={selector_codec}",
        f"source_archive_sha256={source_archive_sha256}",
    )
    effect = ActionEffect.build(
        action_id=packet_action_id,
        family="pr110",
        action_kind="selector_program",
        inverse_source="manual_pr110_replay",
        frame_index="both",
        frame_incidence="seg_pose_joint",
        candidate_status="measured",
        authority=authority,
        normalization_scope=NormalizationScope.FULL_VIDEO_EQUIV_ESTIMATE.value,
        producer="pr110_fec6_k16_packet_manifest",
        consumer=consumer,
        pair_ids=tuple(range(n_pairs)),
        payload_sections=payload_sections,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        receiver_surface={
            "selector_bits": selector_bits,
            "selector_code_bits_total": selector_bits,
            "selector_payload_bytes": selector_payload_bytes,
        },
        parseback_survived=True,
        inflate_survived=True,
        restore_state_pass=True,
        artifact_ref=packet_manifest_path.as_posix(),
        archive_sha256=archive_sha256,
        payload_sha256=selector_payload_sha256,
        evaluator_hash=_evaluator_hash(packet_manifest),
        dependency_hash=_runtime_hash(packet_manifest),
        taint_status="clean",
    )
    expected = exact_delta_score(old_d_seg, new_d_seg, old_d_pose, new_d_pose, old_bytes, new_bytes)
    if effect.delta_score_total != expected:
        raise ValueError(
            "PR110 K16 ActionEffect did not use shared exact_delta_score: "
            f"effect={effect.delta_score_total!r} expected={expected!r}"
        )
    return effect


def _evaluator_hash(packet_manifest: Mapping[str, Any]) -> str | None:
    runtime = _mapping(packet_manifest.get("runtime"))
    manifest = _mapping(runtime.get("manifest"))
    upstream = _mapping(manifest.get("upstream_evaluate_py"))
    value = upstream.get("sha256")
    return str(value) if isinstance(value, str) and value else None


def _runtime_hash(packet_manifest: Mapping[str, Any]) -> str | None:
    runtime = _mapping(packet_manifest.get("runtime"))
    for key in ("runtime_content_tree_sha256", "runtime_tree_sha256"):
        value = runtime.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-manifest", required=True, type=Path)
    parser.add_argument("--archive-manifest", type=Path, default=None)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--authority", default="[contest-CPU] pr110_fec6_k16_packet_manifest")
    parser.add_argument("--consumer", default="inverse_evaluate_candidate_queue")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        packet_manifest_path = args.packet_manifest.resolve(strict=False)
        packet_manifest = _load_json_object(packet_manifest_path, label="packet manifest")
        archive_manifest = None
        if args.archive_manifest is not None:
            archive_manifest = _load_json_object(args.archive_manifest.resolve(strict=False), label="archive manifest")
        effect = build_pr110_k16_packet_action_effect(
            packet_manifest,
            packet_manifest_path=packet_manifest_path,
            archive_manifest=archive_manifest,
            authority=args.authority,
            consumer=args.consumer,
        )
        record = append_action_effect(effect, args.output_jsonl)
        validation = validate_action_effect_payload(record)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not convert PR110 K16 packet to ActionEffect: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": CONVERSION_SCHEMA,
        "packet_manifest": args.packet_manifest.as_posix(),
        "archive_manifest": args.archive_manifest.as_posix() if args.archive_manifest is not None else None,
        "output_jsonl": args.output_jsonl.as_posix(),
        "action_id": record["action_id"],
        "authority": record["authority"],
        "pair_count": len(record["pair_ids"]),
        "selector_bits": _required_int(
            _mapping((archive_manifest or _mapping(packet_manifest.get("archive"))).get("selector_pack_manifest")),
            "selector_code_bits_total",
        ),
        "delta_score_total": record["delta_score_total"],
        "delta_bytes": record["delta_bytes"],
        "value_per_byte": record["value_per_byte"],
        "validation": validation,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0 if validation.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
