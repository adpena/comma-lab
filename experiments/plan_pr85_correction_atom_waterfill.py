#!/usr/bin/env python3
"""Plan PR85 correction side-channel atom policies.

This is a planning-only ledger builder. It parses the PR85 single-member
payload with ``tac.pr85_bundle``, decomposes post/motion/bias/region/randmulti
correction side channels into fine-grained atoms, and emits strict policy gates.
It does not build archives, run inflate, load scorers, write dispatch state, or
launch GPU/remote jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import brotli
except ImportError:  # pragma: no cover - minimal environments
    brotli = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    PR85_HEADERLESS_RANDMULTI_SPECS,
    QPOST_STREAM_NAMES,
    Pr85BundleError,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/plan_pr85_correction_atom_waterfill.py"
SCHEMA = "pr85_correction_atom_waterfill_plan_v1"
ATOM_LEDGER_SCHEMA = "pr85_correction_atom_ledger_v1"
POLICY_SCHEMA = "pr85_correction_atom_policy_candidates_v1"
PAIR_COUNT = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PRESERVE_MANIFEST = REPO_ROOT / (
    "experiments/results/pr85_post_motion_group_policy_candidates_20260504_codex/"
    "preserve_post_all_shift_frac2_frac3/manifest.json"
)
DEFAULT_BASELINE_EVAL = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_hedge_g4dn2_20260503T2335Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_PRESERVE_EVAL = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4_20260504T0242Z/"
    "contest_auth_eval.adjudicated.json"
)

CORRECTION_STREAMS = ("post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti")
MOTION_GROUPS = {
    "shift": "motion_shift",
    "frac": "motion_frac",
    "frac2": "motion_frac2",
    "frac3": "motion_frac3",
}


class PlannerError(ValueError):
    """Raised when a planning input is missing, malformed, or unsafe."""


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _read_json_optional(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_archive_payload(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise PlannerError(f"PR85 archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise PlannerError(f"PR85 archive must contain exactly one member named 'x', got {names!r}")
        validate_pr85_member_name(infos[0].filename)
        raw = zf.read(infos[0])
    return (
        {
            "path": _rel(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "member_name": infos[0].filename,
            "member_bytes": int(len(raw)),
            "member_sha256": _sha256(raw),
            "zip_stored": infos[0].compress_type == zipfile.ZIP_STORED,
        },
        raw,
    )


def _brotli_decode(segment: bytes, name: str) -> bytes:
    if brotli is None:
        raise PlannerError("brotli is required to decode PR85 correction side-channel segments")
    try:
        return brotli.decompress(segment)
    except brotli.error as exc:
        raise PlannerError(f"PR85 segment {name!r} is not Brotli-decodable") from exc


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    for _ in range(count):
        value = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise PlannerError("truncated varint stream")
            byte = raw[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 128:
                values.append(value)
                break
            shift += 7
            if shift > 63:
                raise PlannerError("overlong varint stream")
    return values, pos


def _decode_sparse_choice(raw: bytes, *, magic: bytes, default_choice: int) -> bytes:
    if not raw.startswith(magic) or len(raw) < 5:
        raise PlannerError(f"bad sparse choice magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    if len(vals) != count or pos + count != len(raw):
        raise PlannerError("sparse choice payload length mismatch")
    out = bytearray([default_choice] * PAIR_COUNT)
    index = -1
    for gap, value in zip(gaps, vals, strict=True):
        index += gap + 1
        if not 0 <= index < PAIR_COUNT:
            raise PlannerError(f"sparse choice index out of range: {index}")
        out[index] = int(value) - 1
    return bytes(out)


def _decode_dense_choice(raw: bytes, *, default_center: int | None = None) -> bytes:
    magic = raw[:3]
    values = raw[3:]
    if len(values) != PAIR_COUNT:
        raise PlannerError(f"dense choice payload {magic!r} has {len(values)} pairs, expected {PAIR_COUNT}")
    if magic in {b"SH4", b"FH1", b"FH2", b"FH3", b"BH1", b"RH1"}:
        return bytes(values)
    if magic in {b"SD4", b"FD3", b"BD1", b"RD1"}:
        if default_center is None:
            raise PlannerError(f"default center is required for {magic!r}")
        return bytes(default_center if value == 0 else value - 1 for value in values)
    raise PlannerError(f"unsupported dense choice magic {magic!r}")


def _decode_choice_values(name: str, raw: bytes) -> tuple[bytes, int, str]:
    if name == "shift":
        return _decode_dense_choice(raw, default_center=40), 40, "SD4/SH4"
    if name == "frac":
        if raw[:3] == b"FV1":
            return _decode_sparse_choice(raw, magic=b"FV1", default_choice=4), 4, "FV1/FH1"
        return _decode_dense_choice(raw), 4, "FV1/FH1"
    if name == "frac2":
        return _decode_dense_choice(raw), 4, "FH2"
    if name == "frac3":
        return _decode_dense_choice(raw, default_center=4), 4, "FD3/FH3"
    if name == "bias":
        if raw[:3] == b"BV1":
            return _decode_sparse_choice(raw, magic=b"BV1", default_choice=13), 13, "BD1/BH1/BV1"
        return _decode_dense_choice(raw, default_center=13), 13, "BD1/BH1/BV1"
    if name == "region":
        if raw[:3] == b"RV1":
            return _decode_sparse_choice(raw, magic=b"RV1", default_choice=0), 0, "RD1/RH1/RV1"
        return _decode_dense_choice(raw, default_center=0), 0, "RD1/RH1/RV1"
    raise PlannerError(f"unsupported choice stream {name!r}")


def _post_stages(raw: bytes) -> list[tuple[int, bytes]]:
    if raw.startswith(b"PCD1"):
        if len(raw) < 5:
            raise PlannerError("PCD1 post stream is truncated")
        pos = 5
        stages: list[tuple[int, bytes]] = []
        for _ in range(raw[4]):
            if pos + 3 > len(raw):
                raise PlannerError("PCD1 post stage header is truncated")
            stage_id = int(raw[pos])
            count = int.from_bytes(raw[pos + 1 : pos + 3], "little")
            pos += 3
            choices = raw[pos : pos + count]
            pos += count
            if len(choices) != count:
                raise PlannerError("PCD1 post choices are truncated")
            stages.append((stage_id, bytes(choices)))
        if pos != len(raw):
            raise PlannerError("PCD1 post stream has trailing bytes")
        return stages
    if len(raw) % PAIR_COUNT:
        raise PlannerError("headerless post stream length is not a multiple of pair count")
    count = len(raw) // PAIR_COUNT
    if count <= 0:
        raise PlannerError("post stream contains no stages")
    return [
        (stage_id + 1, raw[stage_id * PAIR_COUNT : (stage_id + 1) * PAIR_COUNT])
        for stage_id in range(count)
    ]


def _byte_stats(values: bytes, *, default: int | None = None) -> dict[str, Any]:
    counts = Counter(values)
    if not values:
        return {"count": 0, "nondefault_count": 0, "unique_count": 0}
    dominant_symbol, dominant_count = max(counts.items(), key=lambda row: (row[1], -row[0]))
    entropy = -sum((count / len(values)) * math.log2(count / len(values)) for count in counts.values())
    nondefault = sum(1 for value in values if value != default) if default is not None else sum(1 for value in values if value)
    return {
        "count": int(len(values)),
        "default_symbol": None if default is None else int(default),
        "dominant_symbol": int(dominant_symbol),
        "dominant_symbol_fraction": round(float(dominant_count / len(values)), 6),
        "entropy_bits_per_symbol": round(float(entropy), 6),
        "ideal_entropy_bytes": round(float(entropy * len(values) / 8.0), 3),
        "max": int(max(values)),
        "min": int(min(values)),
        "nondefault_count": int(nondefault),
        "unique_count": int(len(counts)),
    }


def _compressed_share(stream_bytes: int, group_bytes: int, total_group_bytes: int) -> int:
    if total_group_bytes <= 0:
        return 0
    return int(round(stream_bytes * (group_bytes / total_group_bytes)))


def _noop_risk(nondefault_count: int, *, shared_stream: bool) -> str:
    if nondefault_count <= 0:
        return "high_no_decoded_output_change"
    if shared_stream:
        return "medium_shared_stream_byte_attribution"
    return "low_decoded_output_changes_if_neutralized"


def _atom_record(
    *,
    atom_id: str,
    stream: str,
    group_id: str,
    source_segment_bytes: int,
    estimated_group_segment_bytes: int,
    decoded_group_bytes: int,
    semantic: bytes,
    default_symbol: int | None,
    source_segment_sha256: str,
    group_kind: str,
    shared_stream: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = _byte_stats(semantic, default=default_symbol)
    nondefault_count = int(stats["nondefault_count"])
    record = {
        "atom_id": atom_id,
        "decoded_group_bytes": int(decoded_group_bytes),
        "default_symbol": default_symbol,
        "estimated_group_segment_bytes": int(estimated_group_segment_bytes),
        "group_id": group_id,
        "group_kind": group_kind,
        "neutralization_gate": "requires_exact_component_response_before_eval",
        "no_op_risk": _noop_risk(nondefault_count, shared_stream=shared_stream),
        "rate_score_cost_if_preserved_estimate": round(
            float(estimated_group_segment_bytes * RATE_SCORE_PER_BYTE),
            12,
        ),
        "recode_gate": "requires_decoded_output_parity_before_eval",
        "score_claim": False,
        "semantic_sha256": _sha256(semantic),
        "source_segment_bytes": int(source_segment_bytes),
        "source_segment_sha256": source_segment_sha256,
        "stats": stats,
        "stream": stream,
    }
    if extra:
        record.update(extra)
    return record


def _post_atoms(segment: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _brotli_decode(segment, "post")
    stages = _post_stages(raw)
    atoms = []
    for stage_id, choices in stages:
        atoms.append(
            _atom_record(
                atom_id=f"pr85_post_stage{stage_id}",
                stream="post",
                group_id=f"post_stage{stage_id}",
                source_segment_bytes=len(segment),
                estimated_group_segment_bytes=_compressed_share(len(segment), len(choices), len(raw)),
                decoded_group_bytes=len(choices),
                semantic=choices,
                default_symbol=0,
                source_segment_sha256=_sha256(segment),
                group_kind="post_choice_stage",
                shared_stream=True,
                extra={"post_stage_id": int(stage_id)},
            )
        )
    return (
        {
            "decoded_bytes": len(raw),
            "group_count": len(stages),
            "source_segment_bytes": len(segment),
            "source_segment_sha256": _sha256(segment),
        },
        atoms,
    )


def _choice_atom(name: str, segment: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _brotli_decode(segment, name)
    values, default, codec = _decode_choice_values(name, raw)
    group_id = MOTION_GROUPS.get(name, name)
    atom = _atom_record(
        atom_id=f"pr85_{group_id}",
        stream=name,
        group_id=group_id,
        source_segment_bytes=len(segment),
        estimated_group_segment_bytes=len(segment),
        decoded_group_bytes=len(values),
        semantic=values,
        default_symbol=default,
        source_segment_sha256=_sha256(segment),
        group_kind="dense_or_sparse_choice_stream",
        shared_stream=False,
        extra={"codec_family": codec, "decoded_magic": raw[:4].decode("ascii", errors="replace")},
    )
    return (
        {
            "codec_family": codec,
            "decoded_bytes": len(raw),
            "group_count": 1,
            "source_segment_bytes": len(segment),
            "source_segment_sha256": _sha256(segment),
        },
        [atom],
    )


def _decode_randmulti_row(raw: bytes, pos: int) -> tuple[bytes, int, int]:
    start = pos
    if pos >= len(raw):
        raise PlannerError("randmulti stream ended before count byte")
    count = int(raw[pos])
    pos += 1
    if count == 255:
        if pos + 2 > len(raw):
            raise PlannerError("truncated extended randmulti count")
        count = int.from_bytes(raw[pos : pos + 2], "little")
        pos += 2
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    pos += count
    if len(vals) != count:
        raise PlannerError("truncated randmulti value payload")
    row = bytearray(PAIR_COUNT)
    index = -1
    for gap, value in zip(gaps, vals, strict=True):
        index += gap + 1
        if not 0 <= index < PAIR_COUNT:
            raise PlannerError(f"randmulti index out of range: {index}")
        row[index] = value
    return bytes(row), pos - start, pos


def _randmulti_atoms(segment: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _brotli_decode(segment, "randmulti")
    pos = 0
    atoms: list[dict[str, Any]] = []
    total_rows = 0
    total_nonzero = 0
    for group_index, (height, width, amplitude, row_count) in enumerate(PR85_HEADERLESS_RANDMULTI_SPECS):
        group_start = pos
        rows = []
        row_payload_bytes = []
        for _row in range(int(row_count)):
            row, payload_bytes, pos = _decode_randmulti_row(raw, pos)
            rows.append(row)
            row_payload_bytes.append(payload_bytes)
        semantic = b"".join(rows)
        nonzero = sum(1 for value in semantic if value)
        total_rows += int(row_count)
        total_nonzero += nonzero
        raw_group_bytes = pos - group_start
        atoms.append(
            _atom_record(
                atom_id=f"pr85_randmulti_g{group_index:03d}",
                stream="randmulti",
                group_id=f"randmulti_g{group_index:03d}",
                source_segment_bytes=len(segment),
                estimated_group_segment_bytes=_compressed_share(len(segment), raw_group_bytes, len(raw)),
                decoded_group_bytes=len(semantic),
                semantic=semantic,
                default_symbol=0,
                source_segment_sha256=_sha256(segment),
                group_kind="randmulti_sparse_group",
                shared_stream=True,
                extra={
                    "amplitude": int(amplitude),
                    "height": int(height),
                    "raw_group_payload_bytes": int(raw_group_bytes),
                    "row_payload_bytes": [int(value) for value in row_payload_bytes],
                    "selection_rows": int(row_count),
                    "width": int(width),
                },
            )
        )
    if pos != len(raw):
        raise PlannerError("randmulti stream has trailing bytes after PR85 schedule")
    return (
        {
            "decoded_bytes": len(raw),
            "group_count": len(PR85_HEADERLESS_RANDMULTI_SPECS),
            "nonzero_entries": int(total_nonzero),
            "selection_rows": int(total_rows),
            "source_segment_bytes": len(segment),
            "source_segment_sha256": _sha256(segment),
        },
        atoms,
    )


def _score_from_eval(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    def number(field: str) -> float:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PlannerError(f"{_rel(path)} missing numeric field {field}")
        return float(value)

    score = payload.get("score_recomputed_from_components", payload.get("canonical_score", payload.get("final_score")))
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise PlannerError(f"{_rel(path)} missing recomputed score")
    return {
        "archive_size_bytes": int(payload.get("archive_size_bytes", 0)),
        "avg_posenet_dist": number("avg_posenet_dist"),
        "avg_segnet_dist": number("avg_segnet_dist"),
        "n_samples": int(payload.get("n_samples", 0)),
        "score": round(float(score), 12),
        "score_pose_contribution": number("score_pose_contribution"),
        "score_rate_contribution": number("score_rate_contribution"),
        "score_seg_contribution": number("score_seg_contribution"),
    }


def _exact_negative_constraints(
    *,
    baseline_eval_json: Path,
    preserve_eval_json: Path,
    preserve_manifest_json: Path,
) -> dict[str, Any]:
    manifest = _read_json_optional(preserve_manifest_json)
    baseline_payload = _read_json_optional(baseline_eval_json)
    preserve_payload = _read_json_optional(preserve_eval_json)
    constraints: dict[str, Any] = {
        "source": "preserve_post_all_shift_frac2_frac3",
        "manifest_path": _rel(preserve_manifest_json),
        "baseline_eval_path": _rel(baseline_eval_json),
        "preserve_eval_path": _rel(preserve_eval_json),
        "status": "missing_optional_inputs",
        "blocked_atoms": ["motion_frac"],
        "rule": "motion_frac and neighboring correction atoms must not be blindly deleted; require exact component-response or decoded-output parity before eval",
    }
    if isinstance(manifest, dict):
        constraints["manifest"] = {
            "byte_delta_vs_source_archive": manifest.get("byte_delta_vs_source_archive"),
            "changed_segments": manifest.get("changed_segments"),
            "neutralized_groups": manifest.get("neutralized_groups"),
            "policy_id": manifest.get("policy_id"),
            "selected_groups": manifest.get("selected_groups"),
            "whole_stream_negative_context": manifest.get("whole_stream_negative_context"),
        }
    if isinstance(baseline_payload, dict) and isinstance(preserve_payload, dict):
        baseline = _score_from_eval(baseline_payload, baseline_eval_json)
        preserve = _score_from_eval(preserve_payload, preserve_eval_json)
        score_delta = preserve["score"] - baseline["score"]
        pose_delta = preserve["score_pose_contribution"] - baseline["score_pose_contribution"]
        seg_delta = preserve["score_seg_contribution"] - baseline["score_seg_contribution"]
        rate_delta = preserve["score_rate_contribution"] - baseline["score_rate_contribution"]
        constraints.update(
            {
                "baseline": baseline,
                "preserve_post_all_shift_frac2_frac3": preserve,
                "component_score_delta_vs_baseline": round(float(pose_delta + seg_delta), 12),
                "pose_score_delta_vs_baseline": round(float(pose_delta), 12),
                "rate_score_delta_vs_baseline": round(float(rate_delta), 12),
                "score_delta_vs_baseline": round(float(score_delta), 12),
                "seg_score_delta_vs_baseline": round(float(seg_delta), 12),
                "status": "exact_negative" if score_delta > 0 else "not_negative",
            }
        )
    return constraints


def _rank_atoms(atoms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        atoms,
        key=lambda atom: (
            atom["no_op_risk"] == "high_no_decoded_output_change",
            -int(atom["stats"]["nondefault_count"]),
            int(atom["estimated_group_segment_bytes"]),
            atom["atom_id"],
        ),
    )
    return [
        {
            "atom_id": row["atom_id"],
            "estimated_group_segment_bytes": row["estimated_group_segment_bytes"],
            "group_id": row["group_id"],
            "no_op_risk": row["no_op_risk"],
            "nondefault_count": row["stats"]["nondefault_count"],
            "stream": row["stream"],
        }
        for row in rows
    ]


def _policy_record(
    policy_id: str,
    atom_ids: Iterable[str],
    *,
    intent: str,
    gate: str,
    blocked_by: str | None = None,
) -> dict[str, Any]:
    atoms = sorted(set(atom_ids))
    return {
        "blocked_by": blocked_by,
        "candidate_policy_id": policy_id,
        "dispatch_gate": "planning_only/no_remote_dispatch",
        "eval_gate": gate,
        "intent": intent,
        "no_remote_dispatch": True,
        "planning_only": True,
        "score_claim": False,
        "selected_atom_count": len(atoms),
        "selected_atom_ids": atoms,
    }


def _candidate_policies(atoms: Sequence[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = _rank_atoms(atoms)
    by_stream: dict[str, list[str]] = {}
    for atom in atoms:
        by_stream.setdefault(atom["stream"], []).append(atom["atom_id"])
    top_randmulti = [
        row["atom_id"]
        for row in ranked
        if row["stream"] == "randmulti" and row["no_op_risk"] != "high_no_decoded_output_change"
    ][:8]
    top_small = [
        row["atom_id"]
        for row in sorted(ranked, key=lambda row: (row["estimated_group_segment_bytes"], -row["nondefault_count"]))
        if row["no_op_risk"] != "high_no_decoded_output_change"
    ][:12]
    policies = [
        _policy_record(
            "decoded_parity_recode_all_correction_streams",
            [atom["atom_id"] for atom in atoms],
            intent="lossless recode/search only; decoded correction semantics must match source exactly",
            gate="requires_decoded_output_parity_report_for_every_changed_stream_before_eval",
        ),
        _policy_record(
            "component_response_motion_frac_microatoms",
            by_stream.get("frac", []),
            intent="split the known-sensitive motion_frac correction before considering any neutralization",
            gate="requires_exact_component_response_on_motion_frac_atoms_before_eval",
            blocked_by="preserve_post_all_shift_frac2_frac3 exact negative",
        ),
        _policy_record(
            "component_response_bias_region_sidechannels",
            by_stream.get("bias", []) + by_stream.get("region", []),
            intent="measure bias/region correction value; fixed-v5 length means byte-only deletion is not trusted",
            gate="requires_exact_component_response_or_runtime-decoded parity evidence_before_eval",
        ),
        _policy_record(
            "randmulti_dense_group_response_top008",
            top_randmulti,
            intent="probe high-signal randmulti groups at group granularity rather than deleting the stream",
            gate="requires_exact_component_response_for_selected_randmulti_groups_before_eval",
        ),
        _policy_record(
            "small_byte_correction_response_top012",
            top_small,
            intent="fine-grained low-byte atom screen across post/motion/bias/region/randmulti",
            gate="requires_exact_component_response_for_each_selected_atom_before_eval",
            blocked_by=constraints["source"] if constraints.get("status") == "exact_negative" else None,
        ),
    ]
    return policies


def build_plan(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    baseline_eval_json: Path = DEFAULT_BASELINE_EVAL,
    preserve_eval_json: Path = DEFAULT_PRESERVE_EVAL,
    preserve_manifest_json: Path = DEFAULT_PRESERVE_MANIFEST,
) -> dict[str, Any]:
    source_archive, raw = _read_archive_payload(archive)
    try:
        bundle = parse_pr85_bundle(raw)
    except Pr85BundleError as exc:
        raise PlannerError(f"failed to parse PR85 bundle: {exc}") from exc
    stream_summaries: dict[str, dict[str, Any]] = {}
    atoms: list[dict[str, Any]] = []
    for name in CORRECTION_STREAMS:
        segment = bytes(bundle.segments[name])
        if name == "post":
            summary, rows = _post_atoms(segment)
        elif name == "randmulti":
            summary, rows = _randmulti_atoms(segment)
        else:
            summary, rows = _choice_atom(name, segment)
        stream_summaries[name] = summary
        atoms.extend(rows)
    constraints = _exact_negative_constraints(
        baseline_eval_json=baseline_eval_json,
        preserve_eval_json=preserve_eval_json,
        preserve_manifest_json=preserve_manifest_json,
    )
    ledger = {
        "schema": ATOM_LEDGER_SCHEMA,
        "archive": source_archive,
        "atom_count": len(atoms),
        "atoms": atoms,
        "bundle": {
            "fixed_length_segments": dict(bundle.fixed_length_segments),
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "segment_offsets": dict(bundle.segment_offsets),
        },
        "correction_stream_order": list(QPOST_STREAM_NAMES),
        "exact_negative_constraints": constraints,
        "ranked_atoms": _rank_atoms(atoms),
        "score_claim": False,
        "stream_summaries": stream_summaries,
    }
    policies = {
        "schema": POLICY_SCHEMA,
        "candidate_count": 5,
        "dispatch_blockers": [
            "planning-only ledger: no archive builder output",
            "remote/GPU dispatch forbidden for this worker",
            "decoded-output-changing atoms require exact component-response before eval",
            "lossless recodes require decoded-output parity before eval",
            "preserve_post_all_shift_frac2_frac3 exact negative blocks blind motion_frac deletion",
        ],
        "policies": _candidate_policies(atoms, constraints),
        "score_claim": False,
    }
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "evidence_grade": "planning_only_static_bytes_plus_exact_negative_constraints",
        "source_archive": source_archive,
        "atom_ledger": ledger,
        "candidate_policies": policies,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--baseline-eval-json", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--preserve-eval-json", type=Path, default=DEFAULT_PRESERVE_EVAL)
    parser.add_argument("--preserve-manifest-json", type=Path, default=DEFAULT_PRESERVE_MANIFEST)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    payload = build_plan(
        archive=args.archive,
        baseline_eval_json=args.baseline_eval_json,
        preserve_eval_json=args.preserve_eval_json,
        preserve_manifest_json=args.preserve_manifest_json,
    )
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
