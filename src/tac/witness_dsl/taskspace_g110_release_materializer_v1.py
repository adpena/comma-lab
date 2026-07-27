# SPDX-License-Identifier: MIT
"""Atomic public-release materializer for one explicit G119 joint row.

This is the missing physical transducer between the exhaustive G119 ledger and
``upstream/evaluate.sh``.  It does not choose a cross-stage winner and does not
claim a score.  The caller names one nondominated, self-hashed G119 row; this
module reopens its G121/config/G112/refit custody, recompiles the exact G110
archive, requires the already-measured archive SHA-256 and byte count, snapshots
the closed public runtime allowlist, and atomically publishes a submission
directory plus a sealed release receipt.

The only counted object is ``archive.zip``.  Runtime files and the receipt stay
outside that archive.  No historical archive or payload is admitted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Final

from tac.witness_control.taskspace_post_g105_pose_refit_population_v1 import (
    JOINT_LEDGER_SCHEMA,
    JOINT_ROW_SCHEMA,
    _open_g121_retained_population,
    load_population_config,
)
from tac.witness_control.taskspace_post_g105_pose_refit_v1 import (
    AUDIT_SCHEMA,
    CANDIDATE_SCHEMA,
    POST_G105_REFIT_RUN_SCHEMA,
    _canonical_json,
    _output_binding,
    _require_sha256,
    _seal,
    _strict_file_binding,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    SSD_ROOTS,
)
from tac.witness_dsl.taskspace_g110_generated_y1_pose_product_v1 import (
    compile_g110_generated_y1_pose_v1,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    PUBLIC_RUNTIME_RELATIVE_ROOT,
)

RELEASE_RECEIPT_SCHEMA: Final = "tac.g110_atomic_public_release_receipt.v1"
ARCHIVE_BASENAME: Final = "archive.zip"
RECEIPT_BASENAME: Final = "g110_release_receipt.json"
MIN_RELEASE_FREE_BYTES: Final = (
    1200 * 874 * 1164 * 3
    + 64 * 1024 * 1024
)
PUBLIC_RUNTIME_FILES: Final = (
    "frame0_variants/conditional_lowrank_rice_v1.py",
    "frame0_variants/generated_y1_pose_xip2_v1.py",
    "inflate.py",
    "inflate.sh",
    "semantic_variants/original_coordinr_film_mlp_v1.py",
    "semantic_variants/v9_hosc_dual_head_odd_y1_v1.py",
)
RELEASE_SOURCE_ENTRY_MODULES: Final = (
    "tac.witness_control.taskspace_g121_resumable_stage_harvest_v1",
    "tac.witness_control.taskspace_post_g105_pose_refit_population_v1",
    "tac.witness_control.taskspace_post_g105_pose_refit_v1",
    "tac.witness_dsl.taskspace_g110_generated_y1_pose_product_v1",
    "tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1",
    "tac.witness_dsl.taskspace_g110_release_materializer_v1",
    "tac.witness_dsl.g120_parsed_stage_production_authority_v2",
)
RELEASE_SOURCE_STATIC_FILES: Final = (
    "tools/materialize_taskspace_g110_release_v1.py",
)


class G110ReleaseMaterializerError(RuntimeError):
    """The selected G119 row cannot become an exact public release."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_regular_file(path: Path, *, label: str) -> tuple[bytes, int]:
    if path.is_symlink() or not path.is_file():
        raise G110ReleaseMaterializerError(
            f"{label} must be a regular non-symlink file"
        )
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise G110ReleaseMaterializerError(
            f"{label} changed while being captured"
        )
    return payload, stat.S_IMODE(after.st_mode)


@dataclass(frozen=True, slots=True)
class PublicRuntimeSnapshotV1:
    root: Path
    files: tuple[dict[str, object], ...]
    payloads: tuple[tuple[str, bytes, int], ...]
    tree_sha256: str


def capture_public_runtime_v1(
    *,
    repo_root: Path,
    expected_tree_sha256: str | None = None,
) -> PublicRuntimeSnapshotV1:
    """Capture the six-file G110 runtime once, excluding no unknown sources."""

    root = (
        repo_root.expanduser().resolve()
        / PUBLIC_RUNTIME_RELATIVE_ROOT
    )
    if root.is_symlink() or not root.is_dir():
        raise G110ReleaseMaterializerError(
            "G110 public runtime root is absent or a symlink"
        )
    observed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise G110ReleaseMaterializerError(
                "G110 public runtime contains a symlink"
            )
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        observed.append(relative.as_posix())
    if tuple(observed) != PUBLIC_RUNTIME_FILES:
        raise G110ReleaseMaterializerError(
            "G110 public runtime file census differs from the release allowlist"
        )
    rows: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes, int]] = []
    for relative in observed:
        payload, mode = _stable_regular_file(
            root / relative,
            label=f"public runtime {relative}",
        )
        if mode & 0o022:
            raise G110ReleaseMaterializerError(
                f"public runtime {relative} is group/other writable"
            )
        if relative == "inflate.sh" and not mode & stat.S_IXUSR:
            raise G110ReleaseMaterializerError(
                "public inflate.sh is not owner-executable"
            )
        rows.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "mode": mode,
            }
        )
        payloads.append((relative, payload, mode))
    tree_sha256 = _sha256(_canonical_json(rows))
    if expected_tree_sha256 is not None and tree_sha256 != _require_sha256(
        expected_tree_sha256,
        name="expected public runtime tree",
    ):
        raise G110ReleaseMaterializerError(
            "public runtime tree differs from --expected-runtime-tree-sha256"
        )
    return PublicRuntimeSnapshotV1(
        root=root,
        files=tuple(rows),
        payloads=tuple(payloads),
        tree_sha256=tree_sha256,
    )


@dataclass(frozen=True, slots=True)
class SelectedG119ReleaseRowV1:
    ledger_path: Path
    ledger_bytes: int
    ledger_file_sha256: str
    ledger_body_sha256: str
    config_binding: dict[str, object]
    ledger: dict[str, object]
    rows: tuple[dict[str, object], ...]
    axis_proofs: tuple[G119RowAxisProofV1, ...]
    row: dict[str, object]


@dataclass(frozen=True, slots=True)
class G119RowAxisProofV1:
    joint_row_sha256: str
    population_config_sha256: str
    source_g112_partition_receipt_sha256: str
    target_capsule_receipt_sha256: str
    audit_receipt: dict[str, object]
    selected_candidate_receipt: dict[str, object]


def _open_bound_self_hashed_json(
    value: object,
    *,
    label: str,
    schema: str,
    hash_field: str,
) -> tuple[dict[str, object], dict[str, object]]:
    binding = _strict_file_binding(value, name=label)
    payload, _mode = _stable_regular_file(
        Path(str(binding["path"])),
        label=label,
    )
    if (
        len(payload) != binding["bytes"]
        or _sha256(payload) != binding["sha256"]
    ):
        raise G110ReleaseMaterializerError(
            f"{label} changed after binding validation"
        )
    try:
        opened = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G110ReleaseMaterializerError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(opened) is not dict or opened.get("schema") != schema:
        raise G110ReleaseMaterializerError(
            f"{label} schema differs"
        )
    identity = _require_sha256(
        opened.get(hash_field),
        name=f"{label} body",
    )
    try:
        observed = _sha256(
            _canonical_json(
                {
                    key: item
                    for key, item in opened.items()
                    if key != hash_field
                }
            )
        )
    except (TypeError, ValueError) as exc:
        raise G110ReleaseMaterializerError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if observed != identity:
        raise G110ReleaseMaterializerError(
            f"{label} self-hash differs"
        )
    return binding, opened


def _open_g119_row_axis_proof(
    row: Mapping[str, object],
    *,
    expected_config_binding: Mapping[str, object],
) -> G119RowAxisProofV1:
    checkpoint = _strict_file_binding(
        row.get("post_g105_refit_checkpoint"),
        name="G119 row post-G105 refit checkpoint",
    )
    run_binding, run = _open_bound_self_hashed_json(
        row.get("post_g105_refit_run_receipt"),
        label="G119 row post-G105 refit run receipt",
        schema=POST_G105_REFIT_RUN_SCHEMA,
        hash_field="receipt_sha256",
    )
    audit_binding, audit = _open_bound_self_hashed_json(
        row.get("post_g105_refit_audit_receipt"),
        label="G119 row post-G105 refit audit receipt",
        schema=AUDIT_SCHEMA,
        hash_field="audit_receipt_sha256",
    )
    audit_checkpoint = _strict_file_binding(
        audit.get("final_checkpoint"),
        name="post-G105 audit final checkpoint",
    )
    audit_run = _strict_file_binding(
        audit.get("run_receipt"),
        name="post-G105 audit run receipt",
    )
    audit_config = _strict_file_binding(
        audit.get("config"),
        name="post-G105 audit population config",
    )
    candidate_bindings = audit.get("candidate_rows")
    selected_q_levels = audit.get("selected_q_levels")
    if (
        checkpoint != audit_checkpoint
        or run_binding != audit_run
        or dict(expected_config_binding) != audit_config
        or run.get("final_checkpoint") != checkpoint
        or run.get("selected_xip2_coder")
        != row.get("selected_xip2_coder")
        or run.get("g110_selected_xip2_coder_abi_closed") is not True
        or run.get("exact_public_receiver_in_loop") is not True
        or run.get("research_only") is not True
        or run.get("candidate_claim") is not False
        or run.get("score_claim") is not False
        or run.get("pointer_moved") is not False
        or audit.get("selected_pose_mse") != row.get("d_pose_exact")
        or audit.get("selected_complete_archive_bytes")
        != row.get("final_archive_bytes")
        or audit.get("selected_complete_archive_sha256")
        != row.get("final_archive_sha256")
        or audit.get("selected_xip2_coder")
        != row.get("selected_xip2_coder")
        or selected_q_levels != row.get("selected_q_levels")
        or audit.get("g110_selected_xip2_coder_abi_closed") is not True
        or audit.get("exact_public_receiver_in_loop") is not True
        or audit.get("upstream_evaluate_py_not_run") is not True
        or audit.get("research_only") is not True
        or audit.get("candidate_claim") is not False
        or audit.get("score_claim") is not False
        or audit.get("pointer_moved") is not False
        or type(selected_q_levels) is not int
        or type(candidate_bindings) is not list
        or not candidate_bindings
    ):
        raise G110ReleaseMaterializerError(
            "G119 row axes differ from its physical post-G105 "
            "run/audit custody"
        )
    candidates: list[dict[str, object]] = []
    for index, candidate_binding in enumerate(candidate_bindings):
        _binding, candidate = _open_bound_self_hashed_json(
            candidate_binding,
            label=f"post-G105 candidate receipt {index}",
            schema=CANDIDATE_SCHEMA,
            hash_field="candidate_receipt_sha256",
        )
        _strict_file_binding(
            candidate.get("candidate_state"),
            name=f"post-G105 candidate state {index}",
        )
        if (
            candidate.get("exact_public_receiver_in_loop") is not True
            or candidate.get("research_only") is not True
            or candidate.get("candidate_claim") is not False
            or candidate.get("score_claim") is not False
            or candidate.get("pointer_moved") is not False
            or candidate.get("g110_selected_xip2_coder_abi_closed")
            is not True
        ):
            raise G110ReleaseMaterializerError(
                "post-G105 candidate receipt false-authority fences differ"
            )
        candidates.append(candidate)
    selected = [
        candidate
        for candidate in candidates
        if candidate.get("q_levels") == selected_q_levels
    ]
    if len(selected) != 1:
        raise G110ReleaseMaterializerError(
            "post-G105 selected candidate receipt is absent or ambiguous"
        )
    candidate = selected[0]
    source_g112_sha = _require_sha256(
        candidate.get("source_g112_partition_receipt_sha256"),
        name="post-G105 candidate source G112 receipt",
    )
    population_config_sha = _require_sha256(
        candidate.get("config_sha256"),
        name="post-G105 candidate population config",
    )
    target_capsule_sha = _require_sha256(
        candidate.get("target_capsule_receipt_sha256"),
        name="post-G105 candidate target capsule receipt",
    )
    if (
        candidate.get("pose_mse") != row.get("d_pose_exact")
        or candidate.get("complete_archive_bytes")
        != row.get("final_archive_bytes")
        or candidate.get("complete_archive_sha256")
        != row.get("final_archive_sha256")
        or candidate.get("selected_xip2_coder")
        != row.get("selected_xip2_coder")
        or candidate.get("global_wire_winner_xip2_coder")
        != row.get("selected_xip2_coder")
        or candidate.get("global_wire_winner_archive_bytes")
        != row.get("final_archive_bytes")
        or candidate.get("global_wire_winner_archive_sha256")
        != row.get("final_archive_sha256")
        or run.get("source_g112_partition_receipt_sha256")
        != source_g112_sha
        or run.get("target_capsule_receipt_sha256")
        != target_capsule_sha
    ):
        raise G110ReleaseMaterializerError(
            "G119 row axes differ from its selected physical candidate "
            "receipt"
        )
    return G119RowAxisProofV1(
        joint_row_sha256=str(row["joint_row_sha256"]),
        population_config_sha256=population_config_sha,
        source_g112_partition_receipt_sha256=source_g112_sha,
        target_capsule_receipt_sha256=target_capsule_sha,
        audit_receipt=audit_binding,
        selected_candidate_receipt=_strict_file_binding(
            candidate_bindings[candidates.index(candidate)],
            name="selected post-G105 candidate receipt",
        ),
    )


def _nondominated_row_sha256s(
    rows: Sequence[Mapping[str, object]],
) -> list[str]:
    axes = [
        (
            Fraction(
                int(row["d_seg_numerator"]),
                int(row["d_seg_denominator"]),
            ),
            Decimal(str(row["d_pose_exact"])),
            int(row["final_archive_bytes"]),
        )
        for row in rows
    ]
    result: list[str] = []
    for index, coordinates in enumerate(axes):
        if any(
            other_index != index
            and all(
                left <= right
                for left, right in zip(
                    other,
                    coordinates,
                    strict=True,
                )
            )
            and any(
                left < right
                for left, right in zip(
                    other,
                    coordinates,
                    strict=True,
                )
            )
            for other_index, other in enumerate(axes)
        ):
            continue
        result.append(str(rows[index]["joint_row_sha256"]))
    return result


def _validated_g119_rows(
    rows: object,
    *,
    expected_config_binding: Mapping[str, object],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[G119RowAxisProofV1, ...],
]:
    if type(rows) is not list or not rows:
        raise G110ReleaseMaterializerError(
            "G119 joint ledger has no exact row population"
        )
    validated: list[dict[str, object]] = []
    proofs: list[G119RowAxisProofV1] = []
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            raise G110ReleaseMaterializerError(
                "G119 joint ledger contains a non-object row"
            )
        row_sha = _require_sha256(
            row.get("joint_row_sha256"),
            name="G119 joint row",
        )
        try:
            observed_sha = _sha256(
                _canonical_json(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "joint_row_sha256"
                    }
                )
            )
        except (TypeError, ValueError) as exc:
            raise G110ReleaseMaterializerError(
                "G119 joint row is not canonical finite JSON"
            ) from exc
        d_seg_numerator = row.get("d_seg_numerator")
        d_seg_denominator = row.get("d_seg_denominator")
        d_seg_wire = row.get("d_seg_wire")
        d_pose = row.get("d_pose_exact")
        archive_bytes = row.get("final_archive_bytes")
        if (
            row.get("schema") != JOINT_ROW_SCHEMA
            or row_sha in seen
            or observed_sha != row_sha
            or type(d_seg_numerator) is not int
            or type(d_seg_denominator) is not int
            or not 0 <= d_seg_numerator <= d_seg_denominator
            or d_seg_denominator <= 0
            or type(d_seg_wire) is not float
            or not math.isfinite(d_seg_wire)
            or d_seg_wire != d_seg_numerator / d_seg_denominator
            or type(d_pose) not in {int, float}
            or not math.isfinite(float(d_pose))
            or float(d_pose) < 0.0
            or type(archive_bytes) is not int
            or not 0 < archive_bytes <= 2_100_000
            or row.get("selected_xip2_coder")
            not in {"none", "delta_ar_zlib"}
            or row.get("g110_selected_xip2_coder_abi_closed") is not True
            or row.get("exact_public_receiver_in_loop") is not True
            or row.get("upstream_evaluate_py_run") is not False
            or row.get("research_only") is not True
            or row.get("candidate_claim") is not False
            or row.get("score_claim") is not False
            or row.get("pointer_moved") is not False
        ):
            raise G110ReleaseMaterializerError(
                "G119 joint row identity, exact axes, or false-authority "
                "fences differ"
            )
        _require_sha256(
            row.get("final_archive_sha256"),
            name="G119 joint-row archive",
        )
        _require_sha256(
            row.get("g121_row_identity_sha256"),
            name="G119 joint-row G121 identity",
        )
        _require_sha256(
            row.get("physical_stage_identity_sha256"),
            name="G119 joint-row physical stage",
        )
        seen.add(row_sha)
        validated.append(dict(row))
        proofs.append(
            _open_g119_row_axis_proof(
                row,
                expected_config_binding=expected_config_binding,
            )
        )
    return tuple(validated), tuple(proofs)


def open_selected_g119_release_row_v1(
    *,
    joint_ledger_path: Path,
    expected_joint_ledger_file_sha256: str,
    joint_row_sha256: str,
) -> SelectedG119ReleaseRowV1:
    """Open one explicit nondominated row without inventing winner authority."""

    candidate = joint_ledger_path.expanduser()
    payload, _mode = _stable_regular_file(
        candidate,
        label="G119 joint ledger",
    )
    path = candidate.resolve()
    observed_file_sha = _sha256(payload)
    if observed_file_sha != _require_sha256(
        expected_joint_ledger_file_sha256,
        name="expected G119 joint-ledger file",
    ):
        raise G110ReleaseMaterializerError(
            "G119 joint-ledger file SHA-256 differs"
        )
    try:
        ledger = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G110ReleaseMaterializerError(
            "G119 joint ledger is not strict ASCII JSON"
        ) from exc
    if type(ledger) is not dict:
        raise G110ReleaseMaterializerError(
            "G119 joint ledger is not an exact object"
        )
    body_sha = _require_sha256(
        ledger.get("joint_ledger_sha256"),
        name="G119 joint-ledger body",
    )
    if _sha256(
        _canonical_json(
            {
                key: value
                for key, value in ledger.items()
                if key != "joint_ledger_sha256"
            }
        )
    ) != body_sha:
        raise G110ReleaseMaterializerError(
            "G119 joint-ledger self-hash differs"
        )
    config_binding = _strict_file_binding(
        ledger.get("config"),
        name="G119 population config",
    )
    rows, axis_proofs = _validated_g119_rows(
        ledger.get("rows"),
        expected_config_binding=config_binding,
    )
    nondominated = ledger.get("nondominated_joint_row_sha256")
    selected_sha = _require_sha256(
        joint_row_sha256,
        name="selected G119 joint row",
    )
    recomputed_nondominated = _nondominated_row_sha256s(rows)
    if (
        ledger.get("schema") != JOINT_LEDGER_SCHEMA
        or ledger.get("g121_exhaustive_enumeration_proven") is not True
        or ledger.get("every_retained_stage_processed") is not True
        or ledger.get("retained_stage_count") != len(rows)
        or ledger.get("processed_stage_count") != len(rows)
        or ledger.get("axes")
        != [
            "d_seg_numerator/d_seg_denominator",
            "d_pose_exact",
            "final_archive_bytes",
        ]
        or ledger.get("cross_stage_winner_selected") is not False
        or ledger.get("selection_deferred_to_whole_archive_evaluate")
        is not True
        or ledger.get("upstream_evaluate_py_run") is not False
        or ledger.get("research_only") is not True
        or ledger.get("candidate_claim") is not False
        or ledger.get("score_claim") is not False
        or ledger.get("pointer_moved") is not False
        or type(nondominated) is not list
        or nondominated != recomputed_nondominated
        or selected_sha not in nondominated
    ):
        raise G110ReleaseMaterializerError(
            "G119 ledger is not count-consistent, independently Pareto "
            "recomputed, false-authority fenced, and explicitly nondominated"
        )
    selected = [
        row
        for row in rows
        if type(row) is dict
        and row.get("joint_row_sha256") == selected_sha
    ]
    if len(selected) != 1:
        raise G110ReleaseMaterializerError(
            "selected G119 joint-row identity is absent or ambiguous"
        )
    row = selected[0]
    g121_binding = _strict_file_binding(
        ledger.get("g121_retained_prepose"),
        name="G119-bound G121 retained-prepose manifest",
    )
    if g121_binding["sha256"] != _require_sha256(
        ledger.get("g121_manifest_sha256"),
        name="G119 G121 retained-prepose manifest",
    ):
        raise G110ReleaseMaterializerError(
            "G119 ledger G121 manifest identities differ"
        )
    _strict_file_binding(
        ledger.get("g121_completion_receipt"),
        name="G119-bound G121 completion receipt",
    )
    return SelectedG119ReleaseRowV1(
        ledger_path=path,
        ledger_bytes=len(payload),
        ledger_file_sha256=observed_file_sha,
        ledger_body_sha256=body_sha,
        config_binding=config_binding,
        ledger=dict(ledger),
        rows=rows,
        axis_proofs=axis_proofs,
        row=dict(row),
    )


@dataclass(frozen=True, slots=True)
class G110ReleaseCompileCustodyV1:
    target_capsule_receipt: dict[str, object]
    g112_partition_receipt: dict[str, object]
    post_g105_refit_checkpoint: dict[str, object]
    post_g105_refit_run_receipt: dict[str, object]
    g121_manifest: dict[str, object]
    g121_row_identity_sha256: str
    physical_stage_identity_sha256: str


def _resolve_compile_custody(
    selected: SelectedG119ReleaseRowV1,
) -> G110ReleaseCompileCustodyV1:
    config = load_population_config(
        Path(str(selected.config_binding["path"]))
    )
    if _output_binding(config.config_path) != selected.config_binding:
        raise G110ReleaseMaterializerError(
            "G119 population config changed after ledger sealing"
        )
    opened = _open_g121_retained_population(config)
    ledger = selected.ledger
    if (
        config.g121_retained_prepose
        != _strict_file_binding(
            ledger.get("g121_retained_prepose"),
            name="G119-bound G121 retained-prepose manifest",
        )
        or opened.completion_receipt
        != _strict_file_binding(
            ledger.get("g121_completion_receipt"),
            name="G119-bound G121 completion receipt",
        )
        or opened.manifest_sha256
        != _require_sha256(
            ledger.get("g121_manifest_sha256"),
            name="G119-bound G121 manifest",
        )
        or opened.pointer_snapshot_identity_sha256
        != _require_sha256(
            ledger.get("g121_pointer_snapshot_identity_sha256"),
            name="G119-bound G121 pointer snapshot",
        )
        or opened.postverified_pointer_identity_sha256
        != _require_sha256(
            ledger.get("g121_postverified_pointer_identity_sha256"),
            name="G119-bound G121 postverified pointer",
        )
        or opened.live_target_score_decimal
        != ledger.get("g121_live_target_score_decimal")
        or opened.live_target_numerator
        != ledger.get("g121_live_target_numerator")
        or opened.live_target_denominator
        != ledger.get("g121_live_target_denominator")
    ):
        raise G110ReleaseMaterializerError(
            "G119 ledger does not bind the physically reopened G121 "
            "population/completion/target"
        )
    stages_by_identity = {
        stage.row_identity_sha256: stage
        for stage in opened.stages
    }
    rows_by_identity = {
        str(item["g121_row_identity_sha256"]): item
        for item in selected.rows
    }
    proofs_by_row = {
        proof.joint_row_sha256: proof
        for proof in selected.axis_proofs
    }
    if (
        len(stages_by_identity) != len(opened.stages)
        or len(rows_by_identity) != len(selected.rows)
        or set(stages_by_identity) != set(rows_by_identity)
        or len(proofs_by_row) != len(selected.rows)
        or set(proofs_by_row)
        != {
            str(item["joint_row_sha256"])
            for item in selected.rows
        }
    ):
        raise G110ReleaseMaterializerError(
            "G119 rows do not cover every physical G121 retained stage "
            "exactly once"
        )
    for identity, stage in stages_by_identity.items():
        joint = rows_by_identity[identity]
        proof = proofs_by_row[str(joint["joint_row_sha256"])]
        if (
            joint.get("stage_tag") != stage.stage_tag
            or joint.get("physical_stage_identity_sha256")
            != stage.physical_stage_identity_sha256
            or joint.get("d_seg_numerator")
            != stage.disagreement_pixels
            or joint.get("d_seg_denominator")
            != stage.pixel_denominator
            or joint.get("d_seg_wire") != stage.d_seg_wire
            or joint.get("live_target_score_decimal")
            != stage.live_target_score_decimal
            or joint.get("live_target_numerator")
            != stage.live_target_numerator
            or joint.get("live_target_denominator")
            != stage.live_target_denominator
            or joint.get("pointer_snapshot_identity_sha256")
            != stage.pointer_snapshot_identity_sha256
            or joint.get("postverified_pointer_identity_sha256")
            != stage.postverified_pointer_identity_sha256
            or proof.source_g112_partition_receipt_sha256
            != stage.g112_partition_receipt["sha256"]
            or proof.population_config_sha256 != config.config_sha256
            or proof.target_capsule_receipt_sha256
            != config.target_capsule_receipt["sha256"]
        ):
            raise G110ReleaseMaterializerError(
                "G119 row coordinates differ from its physical G121 stage"
            )
    row = selected.row
    g121_identity = _require_sha256(
        row.get("g121_row_identity_sha256"),
        name="G121 retained row",
    )
    stages = [
        stage
        for stage in opened.stages
        if stage.row_identity_sha256 == g121_identity
    ]
    if len(stages) != 1:
        raise G110ReleaseMaterializerError(
            "selected G119 row does not resolve to one physical G121 stage"
        )
    stage = stages[0]
    physical_sha = _require_sha256(
        row.get("physical_stage_identity_sha256"),
        name="selected physical stage",
    )
    if stage.physical_stage_identity_sha256 != physical_sha:
        raise G110ReleaseMaterializerError(
            "selected G119/G121 physical-stage identities differ"
        )
    if (
        opened.manifest_sha256
        != config.g121_retained_prepose["sha256"]
    ):
        raise G110ReleaseMaterializerError(
            "G121 retained manifest changed after G119"
        )
    return G110ReleaseCompileCustodyV1(
        target_capsule_receipt=config.target_capsule_receipt,
        g112_partition_receipt=stage.g112_partition_receipt,
        post_g105_refit_checkpoint=_strict_file_binding(
            row.get("post_g105_refit_checkpoint"),
            name="selected post-G105 refit checkpoint",
        ),
        post_g105_refit_run_receipt=_strict_file_binding(
            row.get("post_g105_refit_run_receipt"),
            name="selected post-G105 refit run receipt",
        ),
        g121_manifest=config.g121_retained_prepose,
        g121_row_identity_sha256=g121_identity,
        physical_stage_identity_sha256=physical_sha,
    )


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise G110ReleaseMaterializerError(
            "cannot bind release source Git HEAD"
        ) from exc
    if len(value) != 40 or any(
        character not in "0123456789abcdef"
        for character in value
    ):
        raise G110ReleaseMaterializerError(
            "release source Git HEAD is not canonical SHA-1"
        )
    return value


def _module_source_relative_path(
    repo_root: Path,
    module_name: str,
) -> Path | None:
    module_parts = module_name.split(".")
    source = (
        repo_root
        / "src"
        / Path(*module_parts)
    ).with_suffix(".py")
    if source.is_file():
        return source.relative_to(repo_root)
    package = repo_root / "src" / Path(*module_parts) / "__init__.py"
    if package.is_file():
        return package.relative_to(repo_root)
    return None


def _relative_import_base(
    *,
    current_module: str,
    current_is_package: bool,
    imported_module: str | None,
    level: int,
) -> str | None:
    if level == 0:
        return imported_module
    package = current_module.split(".")
    if not current_is_package:
        package = package[:-1]
    keep = len(package) - level + 1
    if keep < 0:
        return None
    parts = package[:keep]
    if imported_module:
        parts.extend(imported_module.split("."))
    return ".".join(parts)


def _existing_module_chain(
    repo_root: Path,
    module_name: str,
) -> tuple[str, ...]:
    parts = module_name.split(".")
    return tuple(
        candidate
        for stop in range(1, len(parts) + 1)
        if (
            candidate := ".".join(parts[:stop])
        )
        and _module_source_relative_path(
            repo_root,
            candidate,
        )
        is not None
    )


def _discover_release_source_paths(repo_root: Path) -> tuple[Path, ...]:
    """Statically close the in-repo Python import graph used by this release."""

    queued = [
        candidate
        for module_name in RELEASE_SOURCE_ENTRY_MODULES
        for candidate in _existing_module_chain(
            repo_root,
            module_name,
        )
    ]
    seen_modules: set[str] = set()
    paths: set[Path] = {
        Path(relative)
        for relative in RELEASE_SOURCE_STATIC_FILES
    }
    paths.update(
        Path(PUBLIC_RUNTIME_RELATIVE_ROOT) / relative
        for relative in PUBLIC_RUNTIME_FILES
    )
    while queued:
        module_name = queued.pop()
        if module_name in seen_modules:
            continue
        seen_modules.add(module_name)
        relative = _module_source_relative_path(repo_root, module_name)
        if relative is None:
            raise G110ReleaseMaterializerError(
                f"release source module is absent: {module_name}"
            )
        paths.add(relative)
        is_package = relative.name == "__init__.py"
        payload, _mode = _stable_regular_file(
            repo_root / relative,
            label=f"release source {relative.as_posix()}",
        )
        try:
            tree = ast.parse(
                payload,
                filename=relative.as_posix(),
            )
        except (SyntaxError, ValueError) as exc:
            raise G110ReleaseMaterializerError(
                f"release source cannot be parsed: {relative.as_posix()}"
            ) from exc
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _relative_import_base(
                    current_module=module_name,
                    current_is_package=is_package,
                    imported_module=node.module,
                    level=node.level,
                )
                if base:
                    candidates.append(base)
                    candidates.extend(
                        f"{base}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
            for candidate in candidates:
                if (
                    candidate.startswith("tac.")
                    or candidate == "tac"
                ):
                    queued.extend(
                        _existing_module_chain(
                            repo_root,
                            candidate,
                        )
                    )
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


@dataclass(frozen=True, slots=True)
class ReleaseSourceSnapshotV1:
    repo_root: Path
    git_sha: str
    files: tuple[dict[str, object], ...]
    tree_sha256: str
    all_files_equal_git_head: bool


def capture_release_source_closure_v1(
    *,
    repo_root: Path,
    expected_git_sha: str | None = None,
) -> ReleaseSourceSnapshotV1:
    """Bind every discovered source and its clean/dirty relation to stable HEAD."""

    root = repo_root.expanduser().resolve()
    if root != Path(__file__).resolve().parents[3]:
        raise G110ReleaseMaterializerError(
            "release source root differs from the executing checkout"
        )
    git_sha = _git_head(root)
    if expected_git_sha is not None and git_sha != expected_git_sha:
        raise G110ReleaseMaterializerError(
            "release source Git HEAD changed before source capture"
        )
    rows: list[dict[str, object]] = []
    for relative in _discover_release_source_paths(root):
        path = root / relative
        payload, mode = _stable_regular_file(
            path,
            label=f"release source {relative.as_posix()}",
        )
        try:
            committed = subprocess.run(
                ["git", "show", f"{git_sha}:{relative.as_posix()}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise G110ReleaseMaterializerError(
                "release source is not tracked at the bound Git HEAD: "
                f"{relative.as_posix()}"
            ) from exc
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "git_blob_sha256": _sha256(committed),
                "matches_git_head": committed == payload,
                "mode": mode,
            }
        )
    if _git_head(root) != git_sha:
        raise G110ReleaseMaterializerError(
            "release source Git HEAD changed during source capture"
        )
    return ReleaseSourceSnapshotV1(
        repo_root=root,
        git_sha=git_sha,
        files=tuple(rows),
        tree_sha256=_sha256(_canonical_json(rows)),
        all_files_equal_git_head=all(
            row["matches_git_head"] is True
            for row in rows
        ),
    )


def _validate_output_root(
    output_root: Path,
    *,
    allowed_output_roots: Sequence[Path],
    minimum_free_bytes: int,
) -> Path:
    candidate = output_root.expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise G110ReleaseMaterializerError(
            "release output must be an absolute non-symlink path"
        )
    lexical = Path(os.path.abspath(candidate))
    for parent in (lexical, *lexical.parents):
        if parent.exists() and parent.is_symlink():
            raise G110ReleaseMaterializerError(
                "release output ancestry contains a symlink"
            )
    resolved = candidate.resolve()
    roots: list[Path] = []
    for configured_root in allowed_output_roots:
        lexical_root = configured_root.expanduser()
        if (
            not lexical_root.is_absolute()
            or lexical_root.is_symlink()
            or not lexical_root.exists()
            or not lexical_root.is_dir()
        ):
            continue
        root = lexical_root.resolve()
        if root != lexical_root.absolute():
            continue
        roots.append(root)
    selected_root = next(
        (
            root
            for root in roots
            if resolved == root or root in resolved.parents
        ),
        None,
    )
    if selected_root is None:
        raise G110ReleaseMaterializerError(
            "release output is outside an existing non-symlink storage root"
        )
    if (
        type(minimum_free_bytes) is not int
        or minimum_free_bytes < 0
        or shutil.disk_usage(selected_root).free < minimum_free_bytes
    ):
        raise G110ReleaseMaterializerError(
            "release output storage preflight failed"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_fsynced(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exact_directory(
    *,
    output_root: Path,
    files: Mapping[str, tuple[bytes, int]],
) -> None:
    expected_names = tuple(sorted(files))
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir():
            raise G110ReleaseMaterializerError(
                "existing release output is not a regular directory"
            )
        entries = tuple(output_root.rglob("*"))
        if any(path.is_symlink() for path in entries):
            raise G110ReleaseMaterializerError(
                "existing release output contains a symlink"
            )
        observed = tuple(
            sorted(
                path.relative_to(output_root).as_posix()
                for path in entries
                if path.is_file()
            )
        )
        if observed != expected_names:
            raise G110ReleaseMaterializerError(
                "existing release output member set differs"
            )
        expected_directories = {
            parent.as_posix()
            for relative in files
            for parent in Path(relative).parents
            if parent != Path(".")
        }
        observed_directories = {
            path.relative_to(output_root).as_posix()
            for path in entries
            if path.is_dir()
        }
        if observed_directories != expected_directories:
            raise G110ReleaseMaterializerError(
                "existing release output directory set differs"
            )
        for relative, (expected_payload, expected_mode) in files.items():
            payload, mode = _stable_regular_file(
                output_root / relative,
                label=f"existing release {relative}",
            )
            if payload != expected_payload or mode != expected_mode:
                raise G110ReleaseMaterializerError(
                    f"existing release {relative} differs"
                )
        return
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.g110-release-",
            dir=output_root.parent,
        )
    )
    try:
        for relative, (payload, mode) in files.items():
            _write_fsynced(
                temporary / relative,
                payload,
                mode,
            )
        directories = {
            temporary,
            *(
                path
                for path in temporary.rglob("*")
                if path.is_dir()
            ),
        }
        for directory in sorted(
            directories,
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        os.replace(temporary, output_root)
        _fsync_directory(output_root.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True, slots=True)
class G110ReleaseMaterializationV1:
    submission_dir: Path
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    runtime_tree_sha256: str
    release_receipt_path: Path
    release_receipt_file_sha256: str
    release_receipt_body_sha256: str


def materialize_g110_release_v1(
    *,
    joint_ledger_path: Path,
    expected_joint_ledger_file_sha256: str,
    joint_row_sha256: str,
    expected_runtime_tree_sha256: str,
    output_root: Path,
    command: Sequence[str],
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
    minimum_free_bytes: int = MIN_RELEASE_FREE_BYTES,
) -> G110ReleaseMaterializationV1:
    """Atomically publish one explicit G119 row as an unevaluated submission."""

    source_root = Path(__file__).resolve().parents[3]
    initial_git_sha = _git_head(source_root)
    source = capture_release_source_closure_v1(
        repo_root=source_root,
        expected_git_sha=initial_git_sha,
    )
    command_tokens = [str(token) for token in command]
    if not command_tokens:
        raise G110ReleaseMaterializerError(
            "release command provenance is empty"
        )
    release_root = _validate_output_root(
        output_root,
        allowed_output_roots=allowed_output_roots,
        minimum_free_bytes=minimum_free_bytes,
    )
    runtime = capture_public_runtime_v1(
        repo_root=source_root,
        expected_tree_sha256=expected_runtime_tree_sha256,
    )
    selected = open_selected_g119_release_row_v1(
        joint_ledger_path=joint_ledger_path,
        expected_joint_ledger_file_sha256=(
            expected_joint_ledger_file_sha256
        ),
        joint_row_sha256=joint_row_sha256,
    )
    custody = _resolve_compile_custody(selected)
    compiled = compile_g110_generated_y1_pose_v1(
        target_capsule_receipt=Path(
            str(custody.target_capsule_receipt["path"])
        ),
        expected_target_capsule_receipt_sha256=str(
            custody.target_capsule_receipt["sha256"]
        ),
        g112_partition_receipt=Path(
            str(custody.g112_partition_receipt["path"])
        ),
        expected_g112_partition_receipt_sha256=str(
            custody.g112_partition_receipt["sha256"]
        ),
        post_g105_refit_checkpoint=Path(
            str(custody.post_g105_refit_checkpoint["path"])
        ),
        expected_post_g105_refit_checkpoint_sha256=str(
            custody.post_g105_refit_checkpoint["sha256"]
        ),
        post_g105_refit_run_receipt=Path(
            str(custody.post_g105_refit_run_receipt["path"])
        ),
        expected_post_g105_refit_run_receipt_sha256=str(
            custody.post_g105_refit_run_receipt["sha256"]
        ),
    )
    row = selected.row
    if (
        type(compiled.archive) is not bytes
        or len(compiled.archive) != compiled.archive_bytes
        or _sha256(compiled.archive) != compiled.archive_sha256
        or compiled.archive_sha256 != row["final_archive_sha256"]
        or compiled.archive_bytes != row["final_archive_bytes"]
        or compiled.selected_xip2_coder != row["selected_xip2_coder"]
        or compiled.g112_partition_receipt_sha256
        != custody.g112_partition_receipt["sha256"]
        or compiled.refit_checkpoint_sha256
        != custody.post_g105_refit_checkpoint["sha256"]
        or compiled.refit_run_receipt_sha256
        != custody.post_g105_refit_run_receipt["sha256"]
    ):
        raise G110ReleaseMaterializerError(
            "recompiled G110 archive/custody differs from selected G119 row"
        )
    reopened_selected = open_selected_g119_release_row_v1(
        joint_ledger_path=joint_ledger_path,
        expected_joint_ledger_file_sha256=(
            expected_joint_ledger_file_sha256
        ),
        joint_row_sha256=joint_row_sha256,
    )
    reopened_custody = _resolve_compile_custody(reopened_selected)
    runtime_after = capture_public_runtime_v1(
        repo_root=source_root,
        expected_tree_sha256=expected_runtime_tree_sha256,
    )
    source_after = capture_release_source_closure_v1(
        repo_root=source_root,
        expected_git_sha=initial_git_sha,
    )
    if (
        reopened_selected != selected
        or reopened_custody != custody
        or runtime_after != runtime
        or source_after != source
    ):
        raise G110ReleaseMaterializerError(
            "release ledger/custody/runtime/source changed during compilation"
        )
    receipt_body: dict[str, object] = {
        "schema": RELEASE_RECEIPT_SCHEMA,
        "source_git_sha": source.git_sha,
        "release_source_closure": {
            "repo_root": str(source.repo_root),
            "git_sha": source.git_sha,
            "files": list(source.files),
            "tree_sha256": source.tree_sha256,
            "all_files_equal_bound_git_head": (
                source.all_files_equal_git_head
            ),
            "dirty_source_file_count": sum(
                row["matches_git_head"] is False
                for row in source.files
            ),
            "git_head_stable_across_compile": True,
        },
        "command": command_tokens,
        "submission_dir": str(release_root),
        "g119_joint_ledger": {
            "path": str(selected.ledger_path),
            "bytes": selected.ledger_bytes,
            "sha256": selected.ledger_file_sha256,
        },
        "g119_joint_ledger_body_sha256": (
            selected.ledger_body_sha256
        ),
        "explicit_nondominated_joint_row_sha256": (
            row["joint_row_sha256"]
        ),
        "g119_population_coverage_recomputed_from_physical_g121": True,
        "g119_nondominance_recomputed_from_receipt_bound_axes": True,
        "g119_pose_axis_remeasured_during_release": False,
        "cross_stage_winner_selected": False,
        "selection_deferred_to_whole_archive_evaluate": True,
        "target_capsule_receipt": custody.target_capsule_receipt,
        "g121_retained_prepose": custody.g121_manifest,
        "g121_row_identity_sha256": (
            custody.g121_row_identity_sha256
        ),
        "physical_stage_identity_sha256": (
            custody.physical_stage_identity_sha256
        ),
        "g112_partition_receipt": custody.g112_partition_receipt,
        "post_g105_refit_checkpoint": (
            custody.post_g105_refit_checkpoint
        ),
        "post_g105_refit_run_receipt": (
            custody.post_g105_refit_run_receipt
        ),
        "archive": {
            "relative_path": ARCHIVE_BASENAME,
            "bytes": compiled.archive_bytes,
            "sha256": compiled.archive_sha256,
            "counted_by_upstream_evaluate_py": True,
        },
        "packet_sha256": compiled.packet_sha256,
        "final_y1_binding_sha256": compiled.final_y1_binding_sha256,
        "g111_source_checkpoint_id_sha256": (
            compiled.g111_source_checkpoint_id_sha256
        ),
        "g111_source_root_sha256": compiled.g111_source_root_sha256,
        "g112_semantic_child_sha256": (
            compiled.g112_semantic_child_sha256
        ),
        "g112_pose_initializer_sha256": (
            compiled.g112_pose_initializer_sha256
        ),
        "refit_xi_sha256": compiled.refit_xi_sha256,
        "selected_y1_wire_codec": (
            compiled.selected_y1_wire_codec.name
        ),
        "selected_xip2_coder": compiled.selected_xip2_coder,
        "selected_outer_zip_method": (
            compiled.selected_outer_zip_method.name
        ),
        "public_runtime": {
            "source_root": str(runtime.root),
            "files": list(runtime.files),
            "tree_sha256": runtime.tree_sha256,
            "expected_tree_sha256": _require_sha256(
                expected_runtime_tree_sha256,
                name="expected public runtime tree",
            ),
            "packaged_in_archive": False,
            "charged_free_source_boundary_audit_run": False,
            "video_derived_content_absent_claim": False,
            "generic_free_eligibility_claim": False,
        },
        "receiver_files_packaged": True,
        "receiver_packaging_closed": False,
        "receiver_closure_state": (
            "FILES_PACKAGED_DOUBLE_DECODE_AND_UPSTREAM_EVAL_OWED"
        ),
        "clean_public_entrypoint_double_decode_run": False,
        "upstream_evaluate_py_run": False,
        "contest_cpu_run": False,
        "contest_cuda_run": False,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    receipt = _seal(
        receipt_body,
        field="release_receipt_sha256",
    )
    receipt_payload = _canonical_json(receipt)
    files: dict[str, tuple[bytes, int]] = {
        ARCHIVE_BASENAME: (compiled.archive, 0o644),
        RECEIPT_BASENAME: (receipt_payload, 0o644),
    }
    files.update(
        {
            relative: (payload, mode)
            for relative, payload, mode in runtime.payloads
        }
    )
    _publish_exact_directory(
        output_root=release_root,
        files=files,
    )
    archive_path = release_root / ARCHIVE_BASENAME
    receipt_path = release_root / RECEIPT_BASENAME
    if (
        archive_path.stat().st_size != compiled.archive_bytes
        or _sha256_file(archive_path) != compiled.archive_sha256
    ):
        raise G110ReleaseMaterializerError(
            "published counted archive differs after atomic release"
        )
    return G110ReleaseMaterializationV1(
        submission_dir=release_root,
        archive_path=archive_path,
        archive_sha256=compiled.archive_sha256,
        archive_bytes=compiled.archive_bytes,
        runtime_tree_sha256=runtime.tree_sha256,
        release_receipt_path=receipt_path,
        release_receipt_file_sha256=_sha256_file(receipt_path),
        release_receipt_body_sha256=str(
            receipt["release_receipt_sha256"]
        ),
    )


__all__ = [
    "ARCHIVE_BASENAME",
    "PUBLIC_RUNTIME_FILES",
    "RECEIPT_BASENAME",
    "RELEASE_RECEIPT_SCHEMA",
    "G110ReleaseMaterializationV1",
    "G110ReleaseMaterializerError",
    "PublicRuntimeSnapshotV1",
    "ReleaseSourceSnapshotV1",
    "SelectedG119ReleaseRowV1",
    "capture_public_runtime_v1",
    "capture_release_source_closure_v1",
    "materialize_g110_release_v1",
    "open_selected_g119_release_row_v1",
]
