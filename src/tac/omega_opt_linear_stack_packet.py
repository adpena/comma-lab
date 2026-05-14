# SPDX-License-Identifier: MIT
"""Fail-closed Omega-OPT linear-stack packet manifest contract.

This module describes the first Omega-OPT 1:1 prototype as a byte-closed
packet scaffold, not as score evidence. Promotion gates intentionally stay
closed until the exact archive bytes, archive SHA-256, runtime/inflate paths,
and exact CUDA auth-eval evidence are present in the same manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.deploy.lightning.round3_harvest import (
    contest_cuda_auth_eval_blockers,
    sha256_file,
)
from tac.omega_opt_claims import (
    CLAIMS_BY_ID,
    EXACT_1TO1_ANCHOR_FIELDS,
    EXACT_ARCHIVE_BYTES_FIELDS,
    EXACT_ARCHIVE_SHA_FIELDS,
    EXACT_AUTH_EVAL_FIELDS,
    FAIL_CLOSED_FIELDS,
    PROMOTION_ALIAS_FIELDS,
    has_exact_1to1_anchor,
)

OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA = "tac_omega_opt_linear_stack_packet_v1"
OMEGA_OPT_LINEAR_STACK_STATUS_SCHEMA = "tac_omega_opt_linear_stack_packet_status_v1"
LINEAR_STACK_CLAIM_ID = "omega_opt_linear_stack"

CANONICAL_LINEAR_STACK_LAYER_IDS: tuple[str, ...] = (
    "arch_shrink_x0_4",
    "imp_alpha_0_7",
    "lossy_coarsening",
    "brotli_pack",
)

EXACT_LINEAR_STACK_PROMOTION_GRADES: tuple[str, ...] = ("A++",)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MISSING = object()


@dataclass(frozen=True)
class LinearStackLayer:
    """One declared transform in the Omega-OPT linear stack."""

    layer_id: str
    order_index: int
    transform_kind: str
    input_artifact_sha256: str | None = None
    output_artifact_sha256: str | None = None
    charged_byte_delta: int | None = None
    runtime_consumed: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("layer_id must be non-empty")
        if self.order_index < 0:
            raise ValueError("order_index must be >= 0")
        if not self.transform_kind:
            raise ValueError("transform_kind must be non-empty")
        if self.charged_byte_delta is not None:
            object.__setattr__(self, "charged_byte_delta", int(self.charged_byte_delta))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "order_index": self.order_index,
            "transform_kind": self.transform_kind,
            "input_artifact_sha256": self.input_artifact_sha256,
            "output_artifact_sha256": self.output_artifact_sha256,
            "charged_byte_delta": self.charged_byte_delta,
            "runtime_consumed": bool(self.runtime_consumed),
            "notes": list(self.notes),
            "score_claim": False,
            "promotion_eligible": False,
        }


def default_linear_stack_layers() -> tuple[LinearStackLayer, ...]:
    """Return the declared first-tranche linear stack transform sequence."""
    return (
        LinearStackLayer(
            layer_id="arch_shrink_x0_4",
            order_index=0,
            transform_kind="architecture_shrink_x0.4",
            notes=("declared_first_tranche_transform_no_score_claim",),
        ),
        LinearStackLayer(
            layer_id="imp_alpha_0_7",
            order_index=1,
            transform_kind="iterative_magnitude_pruning_alpha_0.7",
            notes=("declared_first_tranche_transform_no_score_claim",),
        ),
        LinearStackLayer(
            layer_id="lossy_coarsening",
            order_index=2,
            transform_kind="lossy_weight_coarsening",
            notes=("declared_first_tranche_transform_no_score_claim",),
        ),
        LinearStackLayer(
            layer_id="brotli_pack",
            order_index=3,
            transform_kind="terminal_brotli_pack",
            notes=("declared_first_tranche_transform_no_score_claim",),
        ),
    )


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical JSON mapping."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_linear_stack_packet_manifest(
    *,
    prototype_id: str = "omega_opt_linear_stack_pr101_1to1_scaffold",
    archive_path: str | None = None,
    archive_bytes: int | None = None,
    archive_sha256: str | None = None,
    runtime_packet_path: str | None = None,
    inflate_path: str | None = None,
    evidence_grade: str = "prediction",
    evidence_semantics: str = "omega_opt_linear_stack_packet_scaffold_no_score",
    contest_auth_eval_json: str | None = None,
    one_to_one_anchor_artifact: str | None = None,
    score_claim: bool = False,
    promotion_eligible: bool = False,
    rank_or_kill_eligible: bool = False,
    ready_for_exact_eval_dispatch: bool = False,
    promotion_allowed: bool = False,
    dispatchable: bool = False,
    layers: Sequence[LinearStackLayer] | None = None,
) -> dict[str, Any]:
    """Build a deterministic first-tranche linear-stack packet manifest."""
    claim = CLAIMS_BY_ID[LINEAR_STACK_CLAIM_ID]
    manifest: dict[str, Any] = {
        "schema": OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA,
        "claim_id": LINEAR_STACK_CLAIM_ID,
        "prototype_id": prototype_id,
        "target_modes": ["contest_exact_eval"],
        "score_classification": "prediction",
        "predicted_score": claim.predicted_score,
        "score_claim": bool(score_claim),
        "promotion_eligible": bool(promotion_eligible),
        "rank_or_kill_eligible": bool(rank_or_kill_eligible),
        "ready_for_exact_eval_dispatch": bool(ready_for_exact_eval_dispatch),
        "promotion_allowed": bool(promotion_allowed),
        "dispatchable": bool(dispatchable),
        "requires_exact_linear_stack_anchor": True,
        "linear_stack": [
            layer.to_manifest() for layer in (tuple(layers) if layers is not None else default_linear_stack_layers())
        ],
        "packet": {
            "archive_path": archive_path,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "runtime_packet_path": runtime_packet_path,
            "inflate_path": inflate_path,
            "score_affecting_payload_changed": archive_sha256 is not None,
            "charged_bits_changed": archive_sha256 is not None,
        },
        "evidence": {
            "evidence_grade": evidence_grade,
            "evidence_semantics": evidence_semantics,
            "contest_auth_eval_json": contest_auth_eval_json,
            "one_to_one_anchor_artifact": one_to_one_anchor_artifact,
        },
    }
    manifest["promotion_status"] = linear_stack_packet_status(manifest)
    manifest["blockers"] = manifest["promotion_status"]["blockers"]
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def has_exact_linear_stack_anchor(manifest: Mapping[str, Any]) -> bool:
    """Return true only for complete A++ linear-stack archive/eval custody."""
    fields = linear_stack_exact_anchor_fields(manifest)
    grade = str(fields.get("evidence_grade") or "").strip()
    if grade not in EXACT_LINEAR_STACK_PROMOTION_GRADES:
        return False
    if not _valid_archive_bytes(fields.get("archive_bytes")):
        return False
    if not _valid_sha256(fields.get("archive_sha256")):
        return False
    if not _nonempty_text(fields.get("runtime_packet_path")):
        return False
    if not _nonempty_text(fields.get("inflate_path")):
        return False
    if exact_linear_stack_anchor_blockers(manifest):
        return False

    omega_row = {
        "claim_id": LINEAR_STACK_CLAIM_ID,
        "evidence_grade": grade,
        "exact_archive_bytes": fields.get("archive_bytes"),
        "exact_archive_sha256": fields.get("archive_sha256"),
        "contest_auth_eval_json": fields.get("contest_auth_eval_json"),
        "one_to_one_anchor_artifact": fields.get("one_to_one_anchor_artifact"),
        "source": "omega_opt_linear_stack_exact_packet_manifest",
    }
    return has_exact_1to1_anchor(omega_row)


def linear_stack_exact_anchor_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact-anchor fields from top-level or nested manifest keys."""
    return {
        "evidence_grade": _first_present(manifest, _aliases(("evidence_grade",), "evidence", "exact_eval")),
        "archive_path": _first_present(
            manifest,
            (
                "archive_path",
                "packet.archive_path",
                "archive.path",
            ),
        ),
        "archive_bytes": _first_present(manifest, _aliases(EXACT_ARCHIVE_BYTES_FIELDS, "packet", "archive")),
        "archive_sha256": _first_present(manifest, _aliases(EXACT_ARCHIVE_SHA_FIELDS, "packet", "archive")),
        "runtime_packet_path": _first_present(
            manifest,
            (
                "runtime_packet_path",
                "runtime.path",
                "runtime.packet_path",
                "packet.runtime_packet_path",
                "packet.runtime_path",
            ),
        ),
        "inflate_path": _first_present(
            manifest,
            ("inflate_path", "inflate.sh", "inflate_path", "runtime.inflate_path", "packet.inflate_path"),
        ),
        "contest_auth_eval_json": _first_present(
            manifest,
            _aliases(EXACT_AUTH_EVAL_FIELDS, "evidence", "exact_eval"),
        ),
        "one_to_one_anchor_artifact": _first_present(
            manifest,
            _aliases(EXACT_1TO1_ANCHOR_FIELDS, "evidence", "exact_eval"),
        ),
    }


def exact_linear_stack_anchor_blockers(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return missing or invalid exact-anchor blockers for the manifest."""
    fields = linear_stack_exact_anchor_fields(manifest)
    blockers: list[str] = []

    grade = str(fields.get("evidence_grade") or "").strip()
    if grade not in EXACT_LINEAR_STACK_PROMOTION_GRADES:
        blockers.append("a_plus_plus_evidence_grade_missing")
    if not _valid_archive_bytes(fields.get("archive_bytes")):
        blockers.append("archive_bytes_missing_or_invalid")
    if not _valid_sha256(fields.get("archive_sha256")):
        blockers.append("archive_sha256_missing_or_invalid")
    archive_blockers = _archive_file_blockers(fields)
    blockers.extend(archive_blockers)
    if not _nonempty_text(fields.get("runtime_packet_path")):
        blockers.append("runtime_packet_path_missing")
    elif _existing_file(fields.get("runtime_packet_path")) is None:
        blockers.append("runtime_packet_path_not_found")
    if not _nonempty_text(fields.get("inflate_path")):
        blockers.append("inflate_path_missing")
    elif _existing_file(fields.get("inflate_path")) is None:
        blockers.append("inflate_path_not_found")
    if not _nonempty_text(fields.get("contest_auth_eval_json")):
        blockers.append("exact_cuda_auth_eval_json_missing")
    else:
        blockers.extend(_auth_eval_file_blockers(fields))
    if not _nonempty_text(fields.get("one_to_one_anchor_artifact")):
        blockers.append("one_to_one_anchor_artifact_missing")
    elif _existing_file(fields.get("one_to_one_anchor_artifact")) is None:
        blockers.append("one_to_one_anchor_artifact_not_found")
    blockers.extend(_linear_stack_layer_anchor_blockers(manifest))
    return tuple(blockers)


def linear_stack_packet_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compute fail-closed promotion status for a linear-stack packet manifest."""
    blockers = list(exact_linear_stack_anchor_blockers(manifest))
    exact_anchor = not blockers and has_exact_linear_stack_anchor(manifest)
    if not exact_anchor and "exact_linear_stack_anchor_missing" not in blockers:
        blockers.append("exact_linear_stack_anchor_missing")

    return {
        "schema": OMEGA_OPT_LINEAR_STACK_STATUS_SCHEMA,
        "claim_id": LINEAR_STACK_CLAIM_ID,
        "exact_anchor_complete": exact_anchor,
        "score_claim": exact_anchor and _flag_is_true(manifest, "score_claim"),
        "promotion_eligible": exact_anchor and _flag_is_true(manifest, "promotion_eligible"),
        "rank_or_kill_eligible": exact_anchor and _flag_is_true(manifest, "rank_or_kill_eligible"),
        "ready_for_exact_eval_dispatch": exact_anchor
        and _flag_is_true(manifest, "ready_for_exact_eval_dispatch"),
        "promotion_allowed": exact_anchor and _flag_is_true(manifest, "promotion_allowed"),
        "dispatchable": exact_anchor and _flag_is_true(manifest, "dispatchable"),
        "blockers": [] if exact_anchor else blockers,
    }


def validate_linear_stack_packet_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate a linear-stack packet manifest without granting score credit."""
    findings: list[str] = []

    if manifest.get("schema") != OMEGA_OPT_LINEAR_STACK_PACKET_SCHEMA:
        findings.append("unsupported_or_missing_linear_stack_packet_schema")
    if manifest.get("claim_id") != LINEAR_STACK_CLAIM_ID:
        findings.append("claim_id_must_be_omega_opt_linear_stack")

    findings.extend(_validate_linear_stack_sequence(manifest))

    exact_anchor = has_exact_linear_stack_anchor(manifest)
    positive_closed_flags = False
    for field_name in (*FAIL_CLOSED_FIELDS, *PROMOTION_ALIAS_FIELDS):
        value = _lookup_path(manifest, field_name)
        if value is _MISSING:
            findings.append(f"{field_name}_missing_for_linear_stack_packet_manifest")
            continue
        if value not in (True, False):
            findings.append(f"{field_name}_must_be_boolean")
            continue
        if value is True:
            positive_closed_flags = True
        if value is True and not exact_anchor:
            findings.append(f"{field_name}_must_be_false_without_exact_linear_stack_anchor")

    if manifest.get("family_falsified") is True and not exact_anchor:
        findings.append("family_falsified_must_not_be_true_without_exact_linear_stack_anchor")

    if positive_closed_flags and not exact_anchor:
        findings.extend(exact_linear_stack_anchor_blockers(manifest))

    expected_sha = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    actual_sha = manifest.get("manifest_sha256")
    if actual_sha not in (None, expected_sha):
        findings.append("manifest_sha256_mismatch")

    return list(dict.fromkeys(findings))


def _validate_linear_stack_sequence(manifest: Mapping[str, Any]) -> list[str]:
    layers = manifest.get("linear_stack", manifest.get("layers"))
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return ["linear_stack_layers_missing_or_invalid"]
    layer_ids: list[str] = []
    for layer in layers:
        if not isinstance(layer, Mapping):
            return ["linear_stack_layer_must_be_mapping"]
        layer_ids.append(str(layer.get("layer_id", "")))
        if layer.get("score_claim") is True or layer.get("promotion_eligible") is True:
            return ["linear_stack_layer_score_or_promotion_claim_must_be_false"]
    if tuple(layer_ids) != CANONICAL_LINEAR_STACK_LAYER_IDS:
        return ["linear_stack_sequence_must_match_first_omega_opt_tranche"]
    return []


def _archive_file_blockers(fields: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    archive_path = fields.get("archive_path")
    if not _nonempty_text(archive_path):
        blockers.append("archive_path_missing")
        return blockers
    path = _existing_file(archive_path)
    if path is None:
        blockers.append("archive_path_not_found")
        return blockers
    archive_bytes = fields.get("archive_bytes")
    if _valid_archive_bytes(archive_bytes) and path.stat().st_size != int(archive_bytes):
        blockers.append("archive_file_bytes_mismatch")
    archive_sha256 = fields.get("archive_sha256")
    if _valid_sha256(archive_sha256) and sha256_file(path) != str(archive_sha256).lower():
        blockers.append("archive_file_sha256_mismatch")
    return blockers


def _auth_eval_file_blockers(fields: Mapping[str, Any]) -> list[str]:
    path = _existing_file(fields.get("contest_auth_eval_json"))
    if path is None:
        return ["exact_cuda_auth_eval_json_not_found"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["exact_cuda_auth_eval_json_invalid_json"]
    if not isinstance(payload, dict):
        return ["exact_cuda_auth_eval_json_not_object"]
    expected_archive_bytes = (
        int(fields["archive_bytes"])
        if _valid_archive_bytes(fields.get("archive_bytes"))
        else None
    )
    expected_archive_sha256 = (
        str(fields["archive_sha256"])
        if _valid_sha256(fields.get("archive_sha256"))
        else None
    )
    return [
        f"exact_cuda_auth_eval_{blocker}"
        for blocker in contest_cuda_auth_eval_blockers(
            payload,
            expected_archive_bytes=expected_archive_bytes,
            expected_archive_sha256=expected_archive_sha256,
        )
    ]


def _linear_stack_layer_anchor_blockers(manifest: Mapping[str, Any]) -> list[str]:
    layers = manifest.get("linear_stack", manifest.get("layers"))
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return ["linear_stack_layers_missing_or_invalid"]
    for layer in layers:
        if not isinstance(layer, Mapping):
            return ["linear_stack_layer_anchor_proof_missing"]
        if layer.get("runtime_consumed") is not True:
            return ["linear_stack_layers_missing_runtime_consumption_proof"]
        if not _valid_sha256(layer.get("input_artifact_sha256")):
            return ["linear_stack_layers_missing_input_sha256"]
        if not _valid_sha256(layer.get("output_artifact_sha256")):
            return ["linear_stack_layers_missing_output_sha256"]
        if not isinstance(layer.get("charged_byte_delta"), int) or isinstance(layer.get("charged_byte_delta"), bool):
            return ["linear_stack_layers_missing_charged_byte_delta"]
    return []


def _existing_file(value: object) -> Path | None:
    if not _nonempty_text(value):
        return None
    path = Path(str(value))
    if not path.is_file():
        return None
    return path


def _aliases(names: Sequence[str], *prefixes: str) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        out.append(name)
        out.extend(f"{prefix}.{name}" for prefix in prefixes)
    return tuple(dict.fromkeys(out))


def _lookup_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _first_present(data: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = _lookup_path(data, name)
        if value not in (_MISSING, None, "", [], {}):
            return value
    return None


def _flag_is_true(data: Mapping[str, Any], name: str) -> bool:
    return _lookup_path(data, name) is True


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _valid_archive_bytes(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False
