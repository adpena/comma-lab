#!/usr/bin/env python3
"""Build the local-only PR85 STBM1BR + QRGB randmulti stack candidate.

This builder deliberately does not reuse the QRGB transfer builder's source
anchor bypass.  It applies the PR85-derived QRGB randmulti action to an STBM
mask-recoded source only when a reviewed stack override manifest proves the
STBM archive is descended from the same public PR85 source and changed only the
mask stream with render-order parity.

No scorer is loaded, no remote job is dispatched, and no score claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import build_pr85_qrgb_transfer_archive_candidates as qrgb_builder  # noqa: E402
from experiments import preflight_pr85_fixed_runtime_readiness as fixed_preflight  # noqa: E402
from tac.pr85_bundle import SEGMENT_ORDER, parse_pr85_bundle  # noqa: E402
from tac.stbm1br_mask_codec import STBM1BR_MAGIC  # noqa: E402


TOOL = "experiments/build_pr85_stbm1br_qrgb_randmulti_stack_candidate.py"
SCHEMA = "pr85_stbm1br_qrgb_randmulti_stack_summary_v1"
MANIFEST_SCHEMA = "pr85_stbm1br_qrgb_randmulti_stack_candidate_v1"
OVERRIDE_SCHEMA = "pr85_stbm1br_qrgb_stack_source_override_v1"
DEFAULT_PR85_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_STBM_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
)
DEFAULT_STBM_MANIFEST = DEFAULT_STBM_ARCHIVE.parent / "manifest.json"
DEFAULT_PAIR_ACTION_EVIDENCE = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json"
)
DEFAULT_TRANSFER_PLAN = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/planning.json"
)
DEFAULT_QRGB_STANDALONE_MANIFEST = (
    REPO_ROOT
    / "experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/"
    "pr85_qrgb_f2_randglobal_pair_0192/manifest.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex"
)
DEFAULT_STACK_OVERRIDE = DEFAULT_OUT_DIR / "reviewed_stack_source_override.json"
DEFAULT_LEDGER = (
    REPO_ROOT / ".omx/research/pr85_stbm1br_qrgb_randmulti_stack_builder_20260504_codex.md"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
DEFAULT_CANDIDATE_ID = "pr85_qrgb_f2_randglobal_pair_0192"
STACK_CANDIDATE_ID = "pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192"
EXPECTED_STBM = {
    "archive_bytes": 229_756,
    "archive_sha256": "c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6",
    "render_order_sha256": "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45",
    "diff_pixels": 0,
}
EXPECTED_QRGB_STANDALONE = {
    "archive_bytes": 236_616,
    "archive_sha256": "228f8dff9e14bc7d3cdd445d6c7d73ed1818c0facecaa21e97ab71a523b2da40",
    "stream": "randmulti",
    "pair_index": 192,
    "candidate_value": 20,
    "source_value": 0,
}


class StackBuildError(ValueError):
    """Raised when the stack builder must abort fail-closed."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StackBuildError(f"JSON file is missing: {_rel(path)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StackBuildError(f"{_rel(path)} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StackBuildError(f"{_rel(path)} must contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _archive_sha(meta: Mapping[str, Any]) -> str | None:
    value = meta.get("archive_sha256")
    return value if isinstance(value, str) else None


def _archive_bytes(meta: Mapping[str, Any]) -> int | None:
    value = meta.get("archive_bytes")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _read_pr85_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    return qrgb_builder._read_source_archive(path)


def _archive_info(path: Path) -> dict[str, Any]:
    return qrgb_builder._archive_info(path)


def _zip_bytes(member_name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name, qrgb_builder.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _verify_deterministic_archive(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1 or infos[0].filename != "x":
            raise StackBuildError(f"candidate archive must contain exactly one member 'x': {_rel(archive_path)}")
        payload = zf.read("x")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise StackBuildError("candidate archive member 'x' is not ZIP_STORED")
    first = _zip_bytes("x", payload)
    second = _zip_bytes("x", payload)
    actual = archive_path.read_bytes()
    if first != second or actual != first:
        raise StackBuildError("candidate ZIP bytes are not deterministic under the fixed writer")
    return {
        "deterministic_rewrite_identical": True,
        "member_payload_sha256": _sha256_bytes(payload),
        "member_payload_bytes": len(payload),
        "archive_sha256": _sha256_bytes(actual),
        "archive_bytes": len(actual),
    }


def _segment_diff(left_raw: bytes, right_raw: bytes) -> list[dict[str, Any]]:
    left = parse_pr85_bundle(left_raw)
    right = parse_pr85_bundle(right_raw)
    rows: list[dict[str, Any]] = []
    for name in SEGMENT_ORDER:
        left_segment = bytes(left.segments[name])
        right_segment = bytes(right.segments[name])
        if left_segment != right_segment:
            rows.append(
                {
                    "segment": name,
                    "left_bytes": len(left_segment),
                    "right_bytes": len(right_segment),
                    "left_sha256": _sha256_bytes(left_segment),
                    "right_sha256": _sha256_bytes(right_segment),
                }
            )
    return rows


def _validate_stbm_manifest(
    *,
    stbm_manifest_path: Path,
    stbm_manifest: Mapping[str, Any],
    stbm_archive: Mapping[str, Any],
    pr85_archive: Mapping[str, Any],
    expected_render_order_sha256: str,
) -> dict[str, Any]:
    source = stbm_manifest.get("source_archive")
    candidate = stbm_manifest.get("candidate_archive")
    parity = stbm_manifest.get("parity")
    fail_closed = stbm_manifest.get("fail_closed_preflight")
    segments = stbm_manifest.get("segments")
    if not all(isinstance(row, Mapping) for row in (source, candidate, parity, fail_closed, segments)):
        raise StackBuildError("STBM manifest is missing required source/candidate/parity/preflight sections")
    assert isinstance(source, Mapping)
    assert isinstance(candidate, Mapping)
    assert isinstance(parity, Mapping)
    assert isinstance(fail_closed, Mapping)
    assert isinstance(segments, Mapping)
    if stbm_manifest.get("score_claim") is not False or stbm_manifest.get("dispatch_performed") is not False:
        raise StackBuildError("STBM manifest must be local-only with no score claim or dispatch")
    if _archive_sha(source) != _archive_sha(pr85_archive) or _archive_bytes(source) != _archive_bytes(pr85_archive):
        raise StackBuildError("STBM manifest source archive does not match selected PR85 source")
    if _archive_sha(candidate) != _archive_sha(stbm_archive) or _archive_bytes(candidate) != _archive_bytes(stbm_archive):
        raise StackBuildError("STBM manifest candidate archive does not match selected STBM source")
    if parity.get("decoded_mask_equal") is not True:
        raise StackBuildError("STBM manifest does not prove decoded mask equality")
    if int(parity.get("diff_pixels", -1)) != EXPECTED_STBM["diff_pixels"]:
        raise StackBuildError("STBM manifest diff_pixels is not zero")
    if parity.get("candidate_render_order_sha256") != expected_render_order_sha256:
        raise StackBuildError("STBM candidate render-order SHA does not match expected PR85 render-order SHA")
    if parity.get("pr85_render_order_sha256") != expected_render_order_sha256:
        raise StackBuildError("STBM PR85 render-order SHA does not match expected PR85 render-order SHA")
    checks = fail_closed.get("checks")
    if fail_closed.get("status") != "passed" or not isinstance(checks, Mapping) or not all(checks.values()):
        raise StackBuildError("STBM fail-closed preflight did not pass")
    candidate_mask = segments.get("candidate_mask")
    source_mask = segments.get("source_mask")
    if not isinstance(candidate_mask, Mapping) or not isinstance(source_mask, Mapping):
        raise StackBuildError("STBM manifest lacks source/candidate mask segment metadata")
    if not str(candidate_mask.get("codec", "")).startswith("STBM1BR"):
        raise StackBuildError("STBM candidate mask codec is not STBM1BR")
    if source_mask.get("codec") != "QMA9":
        raise StackBuildError("STBM source mask codec is not QMA9")
    return {
        "path": _rel(stbm_manifest_path),
        "sha256": _sha256_file(stbm_manifest_path),
        "candidate_id": stbm_manifest.get("candidate_id"),
        "source_archive_sha256": _archive_sha(source),
        "candidate_archive_sha256": _archive_sha(candidate),
        "mask_diff_pixels": int(parity.get("diff_pixels", -1)),
        "render_order_sha256": parity.get("candidate_render_order_sha256"),
        "candidate_mask_sha256": candidate_mask.get("sha256"),
        "source_mask_sha256": source_mask.get("sha256"),
    }


def _qrgb_standalone_report(path: Path, candidate_id: str) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("candidate_id") != candidate_id:
        raise StackBuildError("QRGB standalone manifest candidate_id does not match requested action")
    archive = payload.get("candidate_archive")
    if not isinstance(archive, Mapping):
        raise StackBuildError("QRGB standalone manifest lacks candidate_archive")
    if _archive_sha(archive) != EXPECTED_QRGB_STANDALONE["archive_sha256"]:
        raise StackBuildError("QRGB standalone archive SHA does not match the reviewed randmulti 0192 atom")
    if _archive_bytes(archive) != EXPECTED_QRGB_STANDALONE["archive_bytes"]:
        raise StackBuildError("QRGB standalone archive bytes do not match the reviewed randmulti 0192 atom")
    if payload.get("selected_streams") != [EXPECTED_QRGB_STANDALONE["stream"]]:
        raise StackBuildError("QRGB standalone atom is not limited to randmulti")
    if payload.get("selected_pair_indices") != [EXPECTED_QRGB_STANDALONE["pair_index"]]:
        raise StackBuildError("QRGB standalone atom is not pair 0192")
    proofs = payload.get("action_proofs")
    if not isinstance(proofs, list) or len(proofs) != 1 or not isinstance(proofs[0], Mapping):
        raise StackBuildError("QRGB standalone manifest must contain exactly one action proof")
    proof = proofs[0]
    expected = EXPECTED_QRGB_STANDALONE
    if (
        proof.get("stream") != expected["stream"]
        or proof.get("pair_index") != expected["pair_index"]
        or proof.get("source_value") != expected["source_value"]
        or proof.get("candidate_value") != expected["candidate_value"]
    ):
        raise StackBuildError("QRGB standalone action proof does not match randmulti pair 0192")
    preflight = payload.get("fixed_runtime_preflight")
    if isinstance(preflight, Mapping) and preflight.get("ready_for_fixed_runtime_exact_eval") is not True:
        raise StackBuildError("QRGB standalone fixed-runtime preflight is not ready")
    return {
        "path": _rel(path),
        "sha256": _sha256_file(path),
        "candidate_id": candidate_id,
        "archive": dict(archive),
        "action_proof": dict(proof),
        "fixed_runtime_preflight": dict(preflight) if isinstance(preflight, Mapping) else None,
    }


def create_reviewed_stack_override_manifest(
    *,
    output_path: Path,
    pr85_archive_meta: Mapping[str, Any],
    stbm_archive_meta: Mapping[str, Any],
    stbm_manifest_report: Mapping[str, Any],
    qrgb_candidate_id: str,
    reviewed_by: str,
) -> dict[str, Any]:
    if not reviewed_by.strip():
        raise StackBuildError("reviewed_by must be nonempty for a stack override manifest")
    payload = {
        "schema": OVERRIDE_SCHEMA,
        "reviewed": True,
        "reviewed_by": reviewed_by,
        "review_scope": "local_stack_source_lineage_override_only",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "source_sha_mismatch_override": {
            "reason": "QRGB action evidence is anchored to public PR85; stack source is STBM mask-only recode of that source",
            "allowed_candidate_ids": [qrgb_candidate_id],
            "allowed_downstream_streams": ["randmulti"],
            "forbidden_downstream_streams": ["mask"],
        },
        "qrgb_planning_source_archive": {
            "archive_bytes": _archive_bytes(pr85_archive_meta),
            "archive_sha256": _archive_sha(pr85_archive_meta),
            "path": pr85_archive_meta.get("path"),
        },
        "stack_source_archive": {
            "archive_bytes": _archive_bytes(stbm_archive_meta),
            "archive_sha256": _archive_sha(stbm_archive_meta),
            "path": stbm_archive_meta.get("path"),
        },
        "source_lineage_manifest": {
            "path": stbm_manifest_report.get("path"),
            "sha256": stbm_manifest_report.get("sha256"),
        },
        "transform_chain": [
            {
                "transform_id": "pr90_stbm1br_lossless_pr85_mask_recode",
                "input_archive_sha256": _archive_sha(pr85_archive_meta),
                "output_archive_sha256": _archive_sha(stbm_archive_meta),
                "changed_segments": ["mask"],
                "preserved_non_mask_segments": True,
                "decoded_mask_equal": True,
                "diff_pixels": stbm_manifest_report.get("mask_diff_pixels"),
                "render_order_sha256": stbm_manifest_report.get("render_order_sha256"),
                "source_mask_sha256": stbm_manifest_report.get("source_mask_sha256"),
                "candidate_mask_sha256": stbm_manifest_report.get("candidate_mask_sha256"),
            }
        ],
    }
    _write_json(output_path, payload)
    return payload


def _validate_stack_override_manifest(
    *,
    override_manifest_path: Path | None,
    pr85_archive_meta: Mapping[str, Any],
    stbm_archive_meta: Mapping[str, Any],
    stbm_manifest_report: Mapping[str, Any],
    qrgb_candidate_id: str,
) -> dict[str, Any]:
    if _archive_sha(pr85_archive_meta) == _archive_sha(stbm_archive_meta):
        return {
            "required": False,
            "status": "not_required_source_sha_matches",
            "path": None,
            "sha256": None,
        }
    if override_manifest_path is None:
        raise StackBuildError(
            "source_sha_mismatch: reviewed stack override manifest is required before applying PR85 QRGB action evidence to the STBM source"
        )
    payload = _load_json(override_manifest_path)
    if payload.get("schema") != OVERRIDE_SCHEMA:
        raise StackBuildError("stack override manifest has an unexpected schema")
    if payload.get("reviewed") is not True or not isinstance(payload.get("reviewed_by"), str):
        raise StackBuildError("stack override manifest must be explicitly reviewed")
    if payload.get("score_claim") is not False or payload.get("dispatch_performed") is not False:
        raise StackBuildError("stack override manifest must be local-only with no score claim or dispatch")
    if payload.get("remote_jobs_dispatched") is not False:
        raise StackBuildError("stack override manifest must not record remote dispatch")
    override = payload.get("source_sha_mismatch_override")
    if not isinstance(override, Mapping):
        raise StackBuildError("stack override manifest lacks source_sha_mismatch_override")
    if qrgb_candidate_id not in list(override.get("allowed_candidate_ids", [])):
        raise StackBuildError("stack override does not allow the requested QRGB candidate")
    if "randmulti" not in list(override.get("allowed_downstream_streams", [])):
        raise StackBuildError("stack override does not allow randmulti downstream edits")
    if "mask" not in list(override.get("forbidden_downstream_streams", [])):
        raise StackBuildError("stack override must forbid downstream mask edits")
    qrgb_source = payload.get("qrgb_planning_source_archive")
    stack_source = payload.get("stack_source_archive")
    lineage = payload.get("source_lineage_manifest")
    if not all(isinstance(row, Mapping) for row in (qrgb_source, stack_source, lineage)):
        raise StackBuildError("stack override manifest lacks source archive or lineage manifest records")
    assert isinstance(qrgb_source, Mapping)
    assert isinstance(stack_source, Mapping)
    assert isinstance(lineage, Mapping)
    if _archive_sha(qrgb_source) != _archive_sha(pr85_archive_meta) or _archive_bytes(qrgb_source) != _archive_bytes(pr85_archive_meta):
        raise StackBuildError("stack override PR85 source archive does not match selected PR85 source")
    if _archive_sha(stack_source) != _archive_sha(stbm_archive_meta) or _archive_bytes(stack_source) != _archive_bytes(stbm_archive_meta):
        raise StackBuildError("stack override STBM source archive does not match selected STBM source")
    if lineage.get("sha256") != stbm_manifest_report.get("sha256"):
        raise StackBuildError("stack override lineage manifest SHA does not match reviewed STBM manifest")
    chain = payload.get("transform_chain")
    if not isinstance(chain, list) or len(chain) != 1 or not isinstance(chain[0], Mapping):
        raise StackBuildError("stack override transform_chain must contain exactly one STBM transform")
    transform = chain[0]
    if transform.get("input_archive_sha256") != _archive_sha(pr85_archive_meta):
        raise StackBuildError("stack override transform input does not match PR85 source")
    if transform.get("output_archive_sha256") != _archive_sha(stbm_archive_meta):
        raise StackBuildError("stack override transform output does not match STBM source")
    if transform.get("changed_segments") != ["mask"]:
        raise StackBuildError("stack override transform must be mask-only")
    if transform.get("preserved_non_mask_segments") is not True:
        raise StackBuildError("stack override must prove non-mask segments are preserved")
    if transform.get("decoded_mask_equal") is not True or int(transform.get("diff_pixels", -1)) != 0:
        raise StackBuildError("stack override must prove decoded STBM mask parity")
    if transform.get("render_order_sha256") != stbm_manifest_report.get("render_order_sha256"):
        raise StackBuildError("stack override render-order SHA does not match STBM manifest")
    return {
        "required": True,
        "status": "passed",
        "path": _rel(override_manifest_path),
        "sha256": _sha256_file(override_manifest_path),
        "reviewed_by": payload.get("reviewed_by"),
        "allowed_candidate_ids": list(override.get("allowed_candidate_ids", [])),
        "transform_chain": chain,
    }


def _select_qrgb_spec(
    *,
    pr85_archive_meta: Mapping[str, Any],
    evidence_json: Path,
    transfer_plan_json: Path,
    candidate_id: str,
) -> tuple[Any, dict[str, Any]]:
    evidence_report, specs, blockers = qrgb_builder._load_candidate_specs(
        evidence_json=evidence_json,
        transfer_plan_json=transfer_plan_json,
        source_archive=pr85_archive_meta,
        max_candidates=10_000,
    )
    if blockers:
        first = blockers[0]
        raise StackBuildError(f"QRGB evidence failed closed: {first.get('blocker_class')}: {first.get('reason')}")
    matching = [spec for spec in specs if spec.candidate_id == candidate_id]
    if len(matching) != 1:
        raise StackBuildError(f"expected exactly one QRGB candidate spec for {candidate_id}, found {len(matching)}")
    spec = matching[0]
    if len(spec.actions) != 1:
        raise StackBuildError("stack builder only accepts a single QRGB action")
    action = spec.actions[0]
    expected = EXPECTED_QRGB_STANDALONE
    if (
        action.stream != expected["stream"]
        or action.pair_index != expected["pair_index"]
        or action.source_value != expected["source_value"]
        or action.value != expected["candidate_value"]
    ):
        raise StackBuildError("selected QRGB action is not the reviewed randmulti pair 0192 atom")
    return spec, evidence_report


def _runtime_stbm_support(runtime_dir: Path) -> dict[str, Any]:
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    text = inflate_renderer.read_text(encoding="utf-8", errors="replace") if inflate_renderer.is_file() else ""
    checks = {
        "inflate_renderer_present": inflate_renderer.is_file(),
        "stbm_magic_declared": "STBM1BR_MAGIC" in text and "STBM1BR" in text,
        "stbm_loader_present": "_load_masks_from_stbm1br" in text,
        "qma9_content_sniffs_stbm": "suffix.lower() == \".qma9\"" in text and "STBM1BR_MAGIC" in text,
        "canonical_masks_stbm1br_fallback_present": "masks.stbm1br" in text,
    }
    blockers = [
        {"code": f"runtime_stbm_support:{name}", "severity": "blocking", "detail": f"{name} failed"}
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "runtime_dir": _rel(runtime_dir),
        "checks": checks,
        "status": "passed" if not blockers else "failed",
        "remaining_blockers": blockers,
    }


def _orthogonality_report(
    *,
    pr85_raw: bytes,
    stbm_raw: bytes,
    stack_raw: bytes,
    qrgb_action_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pr85_bundle = parse_pr85_bundle(pr85_raw)
    stbm_bundle = parse_pr85_bundle(stbm_raw)
    stack_bundle = parse_pr85_bundle(stack_raw)
    stbm_vs_pr85 = _segment_diff(pr85_raw, stbm_raw)
    stack_vs_stbm = _segment_diff(stbm_raw, stack_raw)
    stack_vs_pr85 = _segment_diff(pr85_raw, stack_raw)
    qrgb_streams = sorted({str(proof.get("stream")) for proof in qrgb_action_proofs})
    qrgb_pairs = sorted({int(proof.get("pair_index")) for proof in qrgb_action_proofs})
    checks = {
        "stbm_changes_only_mask_vs_pr85": [row["segment"] for row in stbm_vs_pr85] == ["mask"],
        "qrgb_changes_only_randmulti_vs_stbm": [row["segment"] for row in stack_vs_stbm] == ["randmulti"],
        "stack_changes_mask_and_randmulti_vs_pr85": [row["segment"] for row in stack_vs_pr85] == ["mask", "randmulti"],
        "qrgb_action_streams_exclude_mask": "mask" not in qrgb_streams,
        "qrgb_action_is_randmulti_pair_0192": qrgb_streams == ["randmulti"] and qrgb_pairs == [192],
        "stbm_mask_preserved_after_qrgb": bytes(stbm_bundle.segments["mask"]) == bytes(stack_bundle.segments["mask"]),
        "non_mask_segments_preserved_by_stbm_before_qrgb": all(
            bytes(pr85_bundle.segments[name]) == bytes(stbm_bundle.segments[name])
            for name in SEGMENT_ORDER
            if name != "mask"
        ),
        "randmulti_source_preserved_by_stbm": bytes(pr85_bundle.segments["randmulti"]) == bytes(stbm_bundle.segments["randmulti"]),
    }
    blockers = [
        {"code": f"orthogonality:{name}", "severity": "blocking", "detail": f"{name} failed"}
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema": "pr85_stbm_qrgb_orthogonality_proof_v1",
        "status": "passed" if not blockers else "failed",
        "checks": checks,
        "stbm_vs_pr85_changed_segments": stbm_vs_pr85,
        "stack_vs_stbm_changed_segments": stack_vs_stbm,
        "stack_vs_pr85_changed_segments": stack_vs_pr85,
        "qrgb_action_streams": qrgb_streams,
        "qrgb_action_pair_indices": qrgb_pairs,
        "remaining_blockers": blockers,
    }


def _preflight_stack_candidate(
    *,
    candidate_archive_path: Path,
    stbm_archive_path: Path,
    robust_current_dir: Path,
    candidate_archive: Mapping[str, Any],
    stack_raw: bytes,
    orthogonality: Mapping[str, Any],
    stbm_manifest_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_preflight = fixed_preflight.build_preflight(
        candidate_archive_path,
        robust_current_dir,
        atom_source_archive=stbm_archive_path,
        expected_archive_sha256=_archive_sha(candidate_archive),
        expected_member_sha256=candidate_archive.get("member_sha256"),
    )
    runtime_stbm = _runtime_stbm_support(robust_current_dir)
    bundle = parse_pr85_bundle(stack_raw)
    mask = bytes(bundle.segments["mask"])
    atom_edit = raw_preflight.get("atom_edit_guard")
    fixed_bridge = raw_preflight.get("fixed_runtime_bridge")
    custody = raw_preflight.get("custody_expectations")
    generic_blockers = raw_preflight.get("blockers", [])
    generic_blocker_codes = [
        row.get("code")
        for row in generic_blockers
        if isinstance(row, Mapping)
    ]
    only_expected_stbm_mask_probe_blocker = generic_blocker_codes in ([], ["pr85_segment_probe_failed"])
    atom_edit_ok = (
        isinstance(atom_edit, Mapping)
        and atom_edit.get("status") == "passed"
        and [row.get("segment") for row in atom_edit.get("changed_segments", []) if isinstance(row, Mapping)]
        == ["randmulti"]
    )
    bridge_ok = (
        isinstance(fixed_bridge, Mapping)
        and fixed_bridge.get("expansion_available") is True
        and fixed_bridge.get("remaining_blockers") == []
    )
    custody_ok = isinstance(custody, Mapping) and custody.get("remaining_blockers") == []
    expansion_manifest = fixed_bridge.get("expansion_manifest") if isinstance(fixed_bridge, Mapping) else {}
    runtime_members = (
        expansion_manifest.get("runtime_members")
        if isinstance(expansion_manifest, Mapping)
        else {}
    )
    masks_member = runtime_members.get("masks.qma9") if isinstance(runtime_members, Mapping) else {}
    stbm_mask_member_ok = (
        isinstance(masks_member, Mapping)
        and masks_member.get("sha256") == _sha256_bytes(mask)
        and mask.startswith(STBM1BR_MAGIC)
    )
    checks = {
        "generic_preflight_has_no_unexpected_blockers": only_expected_stbm_mask_probe_blocker,
        "candidate_custody_expectations_pass": custody_ok,
        "candidate_atom_edit_is_randmulti_vs_stbm": atom_edit_ok,
        "fixed_runtime_bridge_expansion_passes": bridge_ok,
        "stbm_mask_member_is_materialized_for_runtime": stbm_mask_member_ok,
        "robust_current_stbm_loader_support_present": runtime_stbm["status"] == "passed",
        "stbm_lineage_parity_manifest_passed": stbm_manifest_report.get("mask_diff_pixels") == 0,
        "orthogonality_proof_passed": orthogonality.get("status") == "passed",
        "remote_dispatch_not_performed": True,
        "score_claim_not_made": True,
    }
    blockers = [
        {"code": f"stack_preflight:{name}", "severity": "blocking", "detail": f"{name} failed"}
        for name, passed in checks.items()
        if not passed
    ]
    blockers.extend(runtime_stbm["remaining_blockers"])
    if not only_expected_stbm_mask_probe_blocker:
        blockers.append(
            {
                "code": "stack_preflight:unexpected_generic_preflight_blockers",
                "severity": "blocking",
                "detail": f"generic preflight blockers={generic_blocker_codes}",
            }
        )
    ready = not blockers
    stack_preflight = {
        "schema": "pr85_stbm1br_qrgb_stack_fixed_runtime_preflight_v1",
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "planning_only": True,
        "ready_for_fixed_runtime_exact_eval_readiness": ready,
        "readiness_status": "ready" if ready else "blocked",
        "checks": checks,
        "generic_pr85_preflight": {
            "ready_for_fixed_runtime_exact_eval": raw_preflight.get("ready_for_fixed_runtime_exact_eval"),
            "readiness_status": raw_preflight.get("readiness_status"),
            "blocker_codes": generic_blocker_codes,
            "expected_stbm_mask_probe_blocker_waived": generic_blocker_codes == ["pr85_segment_probe_failed"],
        },
        "runtime_stbm_support": runtime_stbm,
        "blockers": blockers,
        "standalone_exact_positive_gates": {
            "stbm_standalone_exact_positive_required": True,
            "qrgb_randmulti_0192_standalone_exact_positive_required": True,
            "not_verified_by_this_local_builder": True,
        },
        "remote_dispatch_allowed": False,
        "exact_eval_allowed_only_after": [
            "STBM standalone exact CUDA positive lands for the exact STBM archive SHA",
            "QRGB randmulti 0192 standalone exact CUDA positive lands for the exact QRGB archive SHA",
            "A fresh lane claim is recorded before any exact eval dispatch",
        ],
    }
    return raw_preflight, stack_preflight


def build_pr85_stbm1br_qrgb_randmulti_stack_candidate(
    *,
    pr85_archive: Path = DEFAULT_PR85_ARCHIVE,
    stbm_archive: Path = DEFAULT_STBM_ARCHIVE,
    stbm_manifest_path: Path = DEFAULT_STBM_MANIFEST,
    qrgb_standalone_manifest_path: Path = DEFAULT_QRGB_STANDALONE_MANIFEST,
    pair_action_evidence_json: Path = DEFAULT_PAIR_ACTION_EVIDENCE,
    transfer_plan_json: Path = DEFAULT_TRANSFER_PLAN,
    reviewed_stack_override_manifest: Path | None = DEFAULT_STACK_OVERRIDE,
    out_dir: Path = DEFAULT_OUT_DIR,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    qrgb_candidate_id: str = DEFAULT_CANDIDATE_ID,
    expected_render_order_sha256: str = EXPECTED_STBM["render_order_sha256"],
) -> dict[str, Any]:
    pr85_meta, pr85_raw = _read_pr85_archive(pr85_archive)
    stbm_meta, stbm_raw = _read_pr85_archive(stbm_archive)
    if _archive_bytes(pr85_meta) != qrgb_builder.KNOWN_PR85["archive_bytes"]:
        raise StackBuildError("selected PR85 source bytes do not match the known PR85 QRGB source")
    if _archive_sha(pr85_meta) != qrgb_builder.KNOWN_PR85["archive_sha256"]:
        raise StackBuildError("selected PR85 source SHA does not match the known PR85 QRGB source")
    if _archive_bytes(stbm_meta) != EXPECTED_STBM["archive_bytes"]:
        raise StackBuildError("selected STBM archive bytes do not match the reviewed STBM source")
    if _archive_sha(stbm_meta) != EXPECTED_STBM["archive_sha256"]:
        raise StackBuildError("selected STBM archive SHA does not match the reviewed STBM source")

    stbm_manifest = _load_json(stbm_manifest_path)
    stbm_manifest_report = _validate_stbm_manifest(
        stbm_manifest_path=stbm_manifest_path,
        stbm_manifest=stbm_manifest,
        stbm_archive=stbm_meta,
        pr85_archive=pr85_meta,
        expected_render_order_sha256=expected_render_order_sha256,
    )
    if [row["segment"] for row in _segment_diff(pr85_raw, stbm_raw)] != ["mask"]:
        raise StackBuildError("actual STBM source differs from PR85 in a non-mask segment")
    qrgb_standalone = _qrgb_standalone_report(qrgb_standalone_manifest_path, qrgb_candidate_id)
    override_report = _validate_stack_override_manifest(
        override_manifest_path=reviewed_stack_override_manifest,
        pr85_archive_meta=pr85_meta,
        stbm_archive_meta=stbm_meta,
        stbm_manifest_report=stbm_manifest_report,
        qrgb_candidate_id=qrgb_candidate_id,
    )
    spec, evidence_report = _select_qrgb_spec(
        pr85_archive_meta=pr85_meta,
        evidence_json=pair_action_evidence_json,
        transfer_plan_json=transfer_plan_json,
        candidate_id=qrgb_candidate_id,
    )

    stack_spec = replace(spec, header_mode="v5")
    mutation_manifest = qrgb_builder._build_one_candidate(
        spec=stack_spec,
        source_archive=stbm_meta,
        source_raw=stbm_raw,
        out_dir=out_dir,
    )
    if mutation_manifest.get("build_status") != "built":
        raise StackBuildError(
            f"QRGB randmulti mutation failed closed on STBM source: {mutation_manifest.get('blocker_class')}"
        )
    generated_archive = mutation_manifest.get("candidate_archive")
    if not isinstance(generated_archive, Mapping):
        raise StackBuildError("QRGB mutation did not emit candidate archive metadata")
    generated_archive_path = REPO_ROOT / str(generated_archive.get("archive_path"))
    stack_candidate_dir = out_dir / STACK_CANDIDATE_ID
    stack_candidate_dir.mkdir(parents=True, exist_ok=True)
    final_archive_path = stack_candidate_dir / "archive.zip"
    final_archive_path.write_bytes(generated_archive_path.read_bytes())
    stack_archive = _archive_info(final_archive_path)
    deterministic = _verify_deterministic_archive(final_archive_path)
    if stack_archive["archive_sha256"] != deterministic["archive_sha256"]:
        raise StackBuildError("deterministic archive proof SHA does not match archive metadata")
    with zipfile.ZipFile(final_archive_path, "r") as zf:
        stack_raw = zf.read("x")

    action_proofs = mutation_manifest.get("action_proofs")
    if not isinstance(action_proofs, list) or not all(isinstance(row, Mapping) for row in action_proofs):
        raise StackBuildError("QRGB mutation manifest lacks action proofs")
    orthogonality = _orthogonality_report(
        pr85_raw=pr85_raw,
        stbm_raw=stbm_raw,
        stack_raw=stack_raw,
        qrgb_action_proofs=action_proofs,
    )
    if orthogonality["status"] != "passed":
        raise StackBuildError("STBM/QRGB orthogonality proof failed")
    raw_preflight, stack_preflight = _preflight_stack_candidate(
        candidate_archive_path=final_archive_path,
        stbm_archive_path=stbm_archive,
        robust_current_dir=robust_current_dir,
        candidate_archive=stack_archive,
        stack_raw=stack_raw,
        orthogonality=orthogonality,
        stbm_manifest_report=stbm_manifest_report,
    )
    if stack_preflight["readiness_status"] != "ready":
        raise StackBuildError("stack fixed-runtime preflight failed")

    qrgb_transform = {
        "transform_id": "pr85_qrgb_f2_randglobal_pair_0192",
        "input_archive_sha256": _archive_sha(stbm_meta),
        "output_archive_sha256": stack_archive["archive_sha256"],
        "changed_segments": ["randmulti"],
        "source_planning_header_mode": spec.header_mode,
        "stack_header_mode": stack_spec.header_mode,
        "header_mode_note": "v5 is required for STBM1BR mask sources because current explicit-30 PR85 parser gates mask magic to QMA9/HPM1.",
        "action_proofs": [dict(row) for row in action_proofs],
        "transforms": mutation_manifest.get("transforms"),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": STACK_CANDIDATE_ID,
        "source_qrgb_candidate_id": qrgb_candidate_id,
        "build_status": "built",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "evidence_grade": "empirical_local_archive_build_only",
        "source_archive": pr85_meta,
        "stack_source_archive": stbm_meta,
        "qrgb_standalone_atom": qrgb_standalone,
        "reviewed_stack_override": override_report,
        "candidate_archive": stack_archive,
        "deterministic_archive": deterministic,
        "source_transform_chain": [
            {
                "transform_id": "pr90_stbm1br_lossless_pr85_mask_recode",
                "input_archive": pr85_meta,
                "output_archive": stbm_meta,
                "changed_segments": ["mask"],
                "decoded_mask_equal": True,
                "diff_pixels": 0,
                "render_order_sha256": expected_render_order_sha256,
                "lineage_manifest": stbm_manifest_report,
            },
            qrgb_transform,
        ],
        "orthogonality": orthogonality,
        "qrgb_mutation_manifest": {
            "path": _rel(out_dir / qrgb_candidate_id / "manifest.json"),
            "sha256": _sha256_file(out_dir / qrgb_candidate_id / "manifest.json"),
            "source_manifest": mutation_manifest,
        },
        "fixed_runtime_preflight": {
            "path": _rel(stack_candidate_dir / "fixed_runtime_preflight.json"),
            "raw_generic_preflight_path": _rel(stack_candidate_dir / "fixed_runtime_preflight.raw_pr85.json"),
            "ready_for_fixed_runtime_exact_eval_readiness": stack_preflight[
                "ready_for_fixed_runtime_exact_eval_readiness"
            ],
            "readiness_status": stack_preflight["readiness_status"],
            "remote_dispatch_allowed": False,
        },
        "dispatch_unlocked": False,
        "dispatch_gate": "blocked_local_only_until_standalone_exact_positives_and_lane_claim",
        "exact_eval_safe_after_standalone_exact_positives": True,
        "standalone_exact_positive_gates": stack_preflight["standalone_exact_positive_gates"],
        "reactivation_criteria": stack_preflight["exact_eval_allowed_only_after"],
    }
    _write_json(stack_candidate_dir / "manifest.json", manifest)
    _write_json(stack_candidate_dir / "fixed_runtime_preflight.raw_pr85.json", raw_preflight)
    _write_json(stack_candidate_dir / "fixed_runtime_preflight.json", stack_preflight)
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidate_count": 1,
        "candidate_archive": stack_archive,
        "source_archive": pr85_meta,
        "stack_source_archive": stbm_meta,
        "qrgb_standalone_archive": qrgb_standalone["archive"],
        "candidate_manifest": _rel(stack_candidate_dir / "manifest.json"),
        "fixed_runtime_preflight": manifest["fixed_runtime_preflight"],
        "orthogonality_status": orthogonality["status"],
        "dispatch_unlocked": False,
        "exact_eval_safe_after_standalone_exact_positives": True,
        "standalone_exact_positive_gates": stack_preflight["standalone_exact_positive_gates"],
        "reactivation_criteria": manifest["reactivation_criteria"],
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def render_ledger(summary: Mapping[str, Any]) -> str:
    candidate = summary.get("candidate_archive", {}) if isinstance(summary.get("candidate_archive"), Mapping) else {}
    source = summary.get("source_archive", {}) if isinstance(summary.get("source_archive"), Mapping) else {}
    stbm = summary.get("stack_source_archive", {}) if isinstance(summary.get("stack_source_archive"), Mapping) else {}
    preflight = summary.get("fixed_runtime_preflight", {}) if isinstance(summary.get("fixed_runtime_preflight"), Mapping) else {}
    lines = [
        "# PR85 STBM1BR + QRGB Randmulti Stack Candidate - 2026-05-04",
        "",
        f"- tool: `{TOOL}`",
        "- score_claim: false",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        "- dispatch_unlocked: false",
        "",
        "## Source Chain",
        "",
        f"- PR85 source: `{source.get('archive_sha256')}` bytes `{source.get('archive_bytes')}`",
        f"- STBM source: `{stbm.get('archive_sha256')}` bytes `{stbm.get('archive_bytes')}`",
        "- STBM transform: mask-only, decoded render-order parity, diff_pixels `0`",
        "- QRGB transform: randmulti pair `0192`, source value `0`, candidate value `20`",
        "",
        "## Candidate",
        "",
        f"- archive: `{candidate.get('archive_path')}`",
        f"- bytes: `{candidate.get('archive_bytes')}`",
        f"- sha256: `{candidate.get('archive_sha256')}`",
        f"- manifest: `{summary.get('candidate_manifest')}`",
        "",
        "## Readiness",
        "",
        f"- orthogonality_status: `{summary.get('orthogonality_status')}`",
        f"- fixed_runtime_preflight: `{preflight.get('path')}`",
        f"- fixed_runtime_readiness: `{preflight.get('readiness_status')}`",
        "- exact_eval_safe_after_standalone_exact_positives: true",
        "- remote dispatch: not performed and not unlocked",
        "",
        "## Reactivation Criteria",
        "",
    ]
    for item in summary.get("reactivation_criteria", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_ledger(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(summary), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_ARCHIVE)
    parser.add_argument("--stbm-archive", type=Path, default=DEFAULT_STBM_ARCHIVE)
    parser.add_argument("--stbm-manifest", type=Path, default=DEFAULT_STBM_MANIFEST)
    parser.add_argument("--qrgb-standalone-manifest", type=Path, default=DEFAULT_QRGB_STANDALONE_MANIFEST)
    parser.add_argument("--pair-action-evidence-json", type=Path, default=DEFAULT_PAIR_ACTION_EVIDENCE)
    parser.add_argument("--transfer-plan-json", type=Path, default=DEFAULT_TRANSFER_PLAN)
    parser.add_argument("--reviewed-stack-override-manifest", type=Path, default=DEFAULT_STACK_OVERRIDE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--qrgb-candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--reviewed-by", default="codex")
    parser.add_argument("--create-reviewed-stack-override-only", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.create_reviewed_stack_override_only:
        pr85_meta, pr85_raw = _read_pr85_archive(args.pr85_archive)
        stbm_meta, stbm_raw = _read_pr85_archive(args.stbm_archive)
        stbm_manifest = _load_json(args.stbm_manifest)
        stbm_manifest_report = _validate_stbm_manifest(
            stbm_manifest_path=args.stbm_manifest,
            stbm_manifest=stbm_manifest,
            stbm_archive=stbm_meta,
            pr85_archive=pr85_meta,
            expected_render_order_sha256=EXPECTED_STBM["render_order_sha256"],
        )
        if [row["segment"] for row in _segment_diff(pr85_raw, stbm_raw)] != ["mask"]:
            raise StackBuildError("cannot write stack override: STBM source is not mask-only vs PR85")
        payload = create_reviewed_stack_override_manifest(
            output_path=args.reviewed_stack_override_manifest,
            pr85_archive_meta=pr85_meta,
            stbm_archive_meta=stbm_meta,
            stbm_manifest_report=stbm_manifest_report,
            qrgb_candidate_id=args.qrgb_candidate_id,
            reviewed_by=args.reviewed_by,
        )
        if args.stdout:
            sys.stdout.write(_json_text(payload))
        else:
            print(_json_text({"reviewed_stack_override_manifest": _rel(args.reviewed_stack_override_manifest)}), end="")
        return 0

    summary = build_pr85_stbm1br_qrgb_randmulti_stack_candidate(
        pr85_archive=args.pr85_archive,
        stbm_archive=args.stbm_archive,
        stbm_manifest_path=args.stbm_manifest,
        qrgb_standalone_manifest_path=args.qrgb_standalone_manifest,
        pair_action_evidence_json=args.pair_action_evidence_json,
        transfer_plan_json=args.transfer_plan_json,
        reviewed_stack_override_manifest=args.reviewed_stack_override_manifest,
        out_dir=args.out_dir,
        robust_current_dir=args.robust_current_dir,
        qrgb_candidate_id=args.qrgb_candidate_id,
    )
    write_ledger(args.ledger_md, summary)
    if args.stdout:
        sys.stdout.write(_json_text(summary))
    else:
        print(
            _json_text(
                {
                    "candidate_archive": summary["candidate_archive"],
                    "candidate_manifest": summary["candidate_manifest"],
                    "dispatch_unlocked": summary["dispatch_unlocked"],
                    "exact_eval_safe_after_standalone_exact_positives": summary[
                        "exact_eval_safe_after_standalone_exact_positives"
                    ],
                }
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
