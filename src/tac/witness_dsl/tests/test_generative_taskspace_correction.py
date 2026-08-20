# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_frontier_pointer import CanonicalFrontierPointer, recompute_effective_frontier
from tac.contest_compliance import compute_upstream_snapshot_sha256
from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    BoundaryCoefficientDelta,
    BoundaryShearletAtomV1,
    DirectDescriptionError,
    IslandShapeAtomV1,
    MovableWorldsheetKnotV1,
    MovableWorldsheetTrackV1,
    ReceiverRealizationProfileV1,
    TopologyEventV1,
    _apply_boundary_shearlet_atoms,
    _event_mask,
    _island_shape_mask,
    requires_pose6_transport,
)
from tac.witness_dsl.generative_taskspace_correction import (
    CompiledGenerativeCorrectionV1,
    CorrectionResourceCountsV1,
    EncoderOnlyTeacherEvidenceV1,
    ExactEvalCustodyPathsV1,
    GenerativeCorrectionProgramV1,
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    admit_by_exact_coupled_score,
    apply_generative_taskspace_correction,
    compile_generative_taskspace_correction,
    parse_generative_taskspace_correction,
)

_G_ARCHIVE_MEMBER = "correction.g"


def _state(*, mutate_first_cell: bool = False) -> PredictorSemanticStateV1:
    labels = np.full((4, 384, 512), 2, dtype=np.uint8)
    labels[:, 210:, :] = 0
    labels[:, 275:290, 230:270] = 3
    if mutate_first_cell:
        labels[0, 0, 0] = 4
    pose6 = np.zeros((4, 6), dtype=np.int16)
    pose6[:, 0] = np.arange(4, dtype=np.int16)
    pose6[:, 1] = np.arange(4, dtype=np.int16) * 2
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(10, 11, 12, 13),
        labels=labels,
        pose6_codes=pose6,
    )


def _program() -> GenerativeCorrectionProgramV1:
    return GenerativeCorrectionProgramV1(
        # Deliberately unsorted input: the compiler must canonicalize it.
        topology_events=(TopologyEventV1(11, "Lane", "birth", "box", 2, 160, 80, 168, 104),),
        boundary_coefficients=(BoundaryCoefficientDelta(10, "Road", 0, 3.0),),
        worldsheet_tracks=(MovableWorldsheetTrackV1(7, 10, 14, 120, 140, 5, 10, 0, 0, 0, 0, 8, 4),),
        worldsheet_knots=(MovableWorldsheetKnotV1(7, 12, delta_center_y_q4=16),),
        realization_profile=ReceiverRealizationProfileV1(
            (
                (20, 80, 20),
                (240, 220, 40),
                (30, 30, 30),
                (220, 40, 40),
                (40, 80, 220),
            )
        ),
    )


def _target(*, include_residual: bool = True) -> np.ndarray:
    labels = _state().labels.copy()
    labels[0, 210:213, :] = 2
    labels[1:3, 160:168, 80:104] = 1
    if include_residual:
        labels[3, 40:42, 20:120] = 4
    return np.ascontiguousarray(labels)


def _teacher(
    *,
    target_labels: np.ndarray | None = None,
    teacher_event_count: int | None = None,
) -> EncoderOnlyTeacherEvidenceV1:
    target = _target() if target_labels is None else np.ascontiguousarray(target_labels, dtype=np.uint8)
    event_count = (
        int(np.count_nonzero(_state().labels != target)) if teacher_event_count is None else teacher_event_count
    )
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=event_count,
    )


def _compile() -> CompiledGenerativeCorrectionV1:
    return compile_generative_taskspace_correction(
        _state(),
        _program(),
        teacher_evidence=_teacher(),
    )


def _joint_score(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / 37_545_489


def test_compiles_canonical_existing_primitives_and_generates_obligations() -> None:
    state = _state()
    first = compile_generative_taskspace_correction(
        state,
        _program(),
        teacher_evidence=_teacher(),
    )
    second = compile_generative_taskspace_correction(
        state,
        _program(),
        teacher_evidence=_teacher(),
    )

    assert first.packet == second.packet
    assert (
        hashlib.sha256(first.packet).hexdigest() == "d146e5a19cba16f7ab2feff8661bc3192043fb953ef211c56a6edc2f0f17c935"
    )
    assert (
        hashlib.sha256(memoryview(first.decoded.labels).cast("B")).hexdigest()
        == "143cb722fe74e21ecd9e9f6468ce99855b82f566dd56cb9d2fb6a94bdadbc829"
    )
    assert first.receipt.packet_bytes == len(first.packet)
    assert first.receipt.total_atoms == 5
    assert first.receipt.max_active_atoms_per_pair == 4
    assert first.receipt.changed_cells > 0
    assert first.receipt.candidate_payload_eligible is True
    assert first.receipt.abi_representable is True
    assert first.receipt.arbitrary_pre_score_caps_applied is False
    assert first.receipt.exact_target_match_is_not_lineage_authority is True
    assert first.receipt.serialized_teacher_bytes == 0
    assert first.receipt.serialized_dense_semantic_bytes == 0
    assert first.receipt.serialized_dense_y_bytes == 0
    assert first.receipt.serialized_explicit_preimage_bytes == 0
    assert first.receipt.primitive_lineage_policy.startswith("original_v9_v19c_receiver_closed")
    assert "no_target_table_dense_plane_or_preimage" in first.receipt.decoder_payload_policy
    assert first.receipt.encoder_teacher_role == "pbr2_acquisition_strata_never_candidate_payload"
    assert first.receipt.decoded_obligation_scope == "generated_frame1_semantic_obligations_only"
    assert first.receipt.independent_frame0_pose_preimage_owed is True
    assert first.receipt.evaluator_realization_and_exact_score_owed is True
    assert first.receipt.research_only is True
    assert first.receipt.score_claim is False
    assert first.receipt.promotion_eligible is False
    assert not np.array_equal(first.decoded.labels, state.labels)
    assert first.decoded.paint_rgb().shape == (4, 384, 512, 3)

    parsed = parse_generative_taskspace_correction(first.packet, predictor_state=state)
    assert parsed.program.worldsheet_tracks == _program().worldsheet_tracks
    assert apply_generative_taskspace_correction(first.packet, predictor_state=state).correction_packet_sha256 == (
        first.receipt.packet_sha256
    )


def test_teacher_and_dense_payload_identities_never_cross_wire() -> None:
    evidence = _teacher()
    compiled = compile_generative_taskspace_correction(
        _state(),
        _program(),
        teacher_evidence=evidence,
    )

    for digest in (
        evidence.pbr1_sha256,
        evidence.pbr2_sha256,
        evidence.target_labels_sha256,
        evidence.obligation_ir_sha256,
        evidence.oracle_evidence_sha256,
        evidence.dense_y_sha256,
    ):
        assert bytes.fromhex(digest) not in compiled.packet
        assert digest.encode("ascii") not in compiled.packet
    assert compiled.receipt.teacher_evidence_binding_sha256 == evidence.binding_sha256
    assert compiled.receipt.pbr1_sha256 == evidence.pbr1_sha256
    assert compiled.receipt.pbr2_sha256 == evidence.pbr2_sha256
    assert compiled.receipt.target_labels_sha256 == evidence.target_labels_sha256
    assert compiled.receipt.obligation_ir_sha256 == evidence.obligation_ir_sha256
    assert compiled.receipt.oracle_evidence_sha256 == evidence.oracle_evidence_sha256
    assert compiled.receipt.dense_y_sha256 == evidence.dense_y_sha256

    for field, digest in (
        ("pbr1_sha256", "7" * 64),
        ("pbr2_sha256", "8" * 64),
        ("obligation_ir_sha256", "9" * 64),
        ("oracle_evidence_sha256", "a" * 64),
        ("dense_y_sha256", "b" * 64),
    ):
        changed_evidence = replace(evidence, **{field: digest})
        changed = compile_generative_taskspace_correction(
            _state(),
            _program(),
            teacher_evidence=changed_evidence,
        )
        assert changed.packet == compiled.packet
        assert getattr(changed.receipt, field) == digest
        assert changed.receipt.binding_sha256 != compiled.receipt.binding_sha256

    changed_target = evidence.target_labels.copy()
    changed_target[0, 0, 0] = (int(changed_target[0, 0, 0]) + 1) % 5
    changed_target = np.ascontiguousarray(changed_target)
    target_evidence = replace(
        evidence,
        target_labels=changed_target,
        target_labels_sha256=hashlib.sha256(memoryview(changed_target).cast("B")).hexdigest(),
        teacher_event_count=int(np.count_nonzero(_state().labels != changed_target)),
    )
    target_changed = compile_generative_taskspace_correction(
        _state(),
        _program(),
        teacher_evidence=target_evidence,
    )
    assert target_changed.packet == compiled.packet
    assert target_changed.receipt.target_labels_sha256 == target_evidence.target_labels_sha256
    assert target_changed.receipt.binding_sha256 != compiled.receipt.binding_sha256

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="only exact TopologyEventV1"):
        GenerativeCorrectionProgramV1(topology_events=(b"PBR2-target-table",))  # type: ignore[arg-type]
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="target label hash"):
        replace(evidence, target_labels=changed_target)


def test_existing_shearlet_and_island_codecs_drive_the_same_receiver_semantics() -> None:
    state = _state()
    shearlet = BoundaryShearletAtomV1(10, "Road", 210, 256, 8, 32, 0, 32)
    island = IslandShapeAtomV1(11, "birth", 1, 100, 140, 6, 12, 0, 0, 0, 8)
    program = GenerativeCorrectionProgramV1(
        boundary_shearlets=(shearlet,),
        island_shapes=(island,),
    )
    expected = state.labels.copy()
    masks = {role: state.labels[0] == ROLE_CLASS_IDS[role] for role in REALIZATION_PAINT_ORDER}
    masks["Road"] = _apply_boundary_shearlet_atoms(masks["Road"], (shearlet,))
    merged = np.full((384, 512), ROLE_CLASS_IDS["UndrivableBoundary"], dtype=np.uint8)
    for role in REALIZATION_PAINT_ORDER:
        merged[masks[role]] = np.uint8(ROLE_CLASS_IDS[role])
    expected[0] = merged
    island_sites = _island_shape_mask(
        island,
        source_pair_id=11,
        source_pair_start=state.pair_start,
        pose6_codes=state.pose6_codes,
    )
    expected[1, island_sites] = ROLE_CLASS_IDS["Movable"]
    target = expected.copy()
    target[3, 40:42, 20:120] = 4

    compiled = compile_generative_taskspace_correction(
        state,
        program,
        teacher_evidence=_teacher(target_labels=target),
    )

    assert np.array_equal(compiled.decoded.labels, expected)
    assert compiled.receipt.family_counts["boundary_shearlets"] == 1
    assert compiled.receipt.family_counts["island_shapes"] == 1
    assert compiled.receipt.debt_after_cells == 200


@pytest.mark.parametrize("atom_kind", ["event", "island"])
def test_zero_gain_masks_do_not_read_pose_and_nonzero_gain_none_fails_closed(atom_kind: str) -> None:
    pose_a = np.zeros((2, 6), dtype=np.int16)
    pose_b = np.array([[300, -250, 0, 0, 0, 0], [-300, 250, 0, 0, 0, 0]], dtype=np.int16)
    if atom_kind == "event":
        zero_gain = TopologyEventV1(10, "Lane", "birth", "box", 2, 10, 20, 14, 28)
        nonzero_gain = TopologyEventV1(10, "Lane", "birth", "box", 2, 10, 20, 14, 28, 1, 0)
    else:
        zero_gain = IslandShapeAtomV1(10, "birth", 2, 100, 140, 6, 12, 0, 0, 0, 8)
        nonzero_gain = IslandShapeAtomV1(10, "birth", 2, 100, 140, 6, 12, 0, 0, 0, 8, 0, 1)

    def render(
        atom: TopologyEventV1 | IslandShapeAtomV1,
        pose: np.ndarray | None,
    ) -> np.ndarray:
        if type(atom) is TopologyEventV1:
            return _event_mask(
                atom,
                source_pair_id=11,
                source_pair_start=10,
                pose6_codes=pose,
            )
        if type(atom) is IslandShapeAtomV1:
            return _island_shape_mask(
                atom,
                source_pair_id=11,
                source_pair_start=10,
                pose6_codes=pose,
            )
        raise AssertionError("closed test atom type escaped")

    assert requires_pose6_transport(zero_gain) is False
    assert requires_pose6_transport(nonzero_gain) is True
    without_pose = render(zero_gain, None)
    assert np.array_equal(without_pose, render(zero_gain, pose_a))
    assert np.array_equal(without_pose, render(zero_gain, pose_b))
    with pytest.raises(DirectDescriptionError, match="requires Pose6 transport"):
        render(nonzero_gain, None)


def test_mutation_truncation_and_identity_drift_refuse() -> None:
    state = _state()
    packet = _compile().packet
    mutated = bytearray(packet)
    mutated[-1] ^= 1
    header_mutated = bytearray(packet)
    header_mutated[48] ^= 1

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="CRC mismatch"):
        parse_generative_taskspace_correction(bytes(mutated), predictor_state=state)
    with pytest.raises(GenerativeTaskspaceCorrectionError):
        parse_generative_taskspace_correction(bytes(header_mutated), predictor_state=state)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="length"):
        parse_generative_taskspace_correction(packet[:-1], predictor_state=state)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="identity binding mismatch"):
        parse_generative_taskspace_correction(packet, predictor_state=_state(mutate_first_cell=True))
    for offset in range(len(packet)):
        single_byte_mutation = bytearray(packet)
        single_byte_mutation[offset] ^= 1
        with pytest.raises(GenerativeTaskspaceCorrectionError):
            parse_generative_taskspace_correction(bytes(single_byte_mutation), predictor_state=state)


def test_exact_target_reconstruction_by_compact_generator_is_candidate_eligible() -> None:
    exact_program = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(11, "Lane", "birth", "box", 1, 160, 80, 168, 104),)
    )
    exact_target = _state().labels.copy()
    exact_target[1, 160:168, 80:104] = 1
    compiled = compile_generative_taskspace_correction(
        _state(),
        exact_program,
        teacher_evidence=_teacher(target_labels=exact_target),
    )
    assert len(compiled.packet) == 141
    assert (
        hashlib.sha256(compiled.packet).hexdigest()
        == "1fd833ee310800688c76a3bb7c1af8ea287d8d02fb1997c9d7f0e23c3f735230"
    )
    assert compiled.receipt.debt_after_cells == 0
    assert compiled.receipt.debt_delta_cells == -compiled.receipt.debt_before_cells
    assert compiled.receipt.exact_semantic_target_reconstructed is True
    assert compiled.receipt.candidate_payload_eligible is True


def test_pbr2_event_count_is_bound_but_sparse_hard_pixel_correction_is_legal() -> None:
    one_atom = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(11, "Lane", "birth", "box", 1, 160, 80, 168, 104),)
    )
    target = _state().labels.copy()
    target[1, 160:168, 80:104] = 1
    target[3, 0, 0] = 4
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="PBR2 teacher event count differs"):
        compile_generative_taskspace_correction(
            _state(),
            one_atom,
            teacher_evidence=_teacher(target_labels=target, teacher_event_count=1),
        )

    one_cell = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(11, "Lane", "birth", "box", 1, 0, 0, 1, 1),)
    )
    one_cell_target = _state().labels.copy()
    one_cell_target[1, 0, 0] = 1
    compiled = compile_generative_taskspace_correction(
        _state(),
        one_cell,
        teacher_evidence=_teacher(target_labels=one_cell_target),
    )
    assert compiled.receipt.changed_cells == 1
    assert compiled.receipt.exact_semantic_target_reconstructed is True


def test_packet_self_describes_resources_without_arbitrary_caps() -> None:
    events = tuple(TopologyEventV1(10, "Lane", "birth", "box", 1, 0, column, 1, column + 1) for column in range(17))
    compiled = compile_generative_taskspace_correction(
        _state(),
        GenerativeCorrectionProgramV1(topology_events=events),
        teacher_evidence=_teacher(),
    )
    parsed = parse_generative_taskspace_correction(compiled.packet, predictor_state=_state())

    assert parsed.packet_bytes == len(compiled.packet)
    assert parsed.resource_counts.topology_events == 17
    assert compiled.receipt.resource_counts == parsed.resource_counts
    assert compiled.receipt.max_active_atoms_per_pair == 17
    assert compiled.receipt.changed_cells == 17
    assert compiled.receipt.arbitrary_pre_score_caps_applied is False
    assert CorrectionResourceCountsV1(0, 65_535, 0, 0, 0, 0, 0).total_atoms == 65_535
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="uint16-representable"):
        CorrectionResourceCountsV1(0, 65_536, 0, 0, 0, 0, 0)
    with pytest.raises(TypeError, match="max_changed_cells"):
        compile_generative_taskspace_correction(
            _state(),
            _program(),
            teacher_evidence=_teacher(),
            max_changed_cells=1,  # type: ignore[call-arg]
        )


def test_pair_window_and_temporal_lifecycle_refuse() -> None:
    outside = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(9, "Lane", "birth", "box", 1, 1, 1, 3, 3),)
    )
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="address escaped"):
        compile_generative_taskspace_correction(
            _state(),
            outside,
            teacher_evidence=_teacher(),
        )

    escaping_lifetime = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(13, "Lane", "birth", "box", 2, 1, 1, 3, 3),)
    )
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="lifetime escaped"):
        compile_generative_taskspace_correction(
            _state(),
            escaping_lifetime,
            teacher_evidence=_teacher(),
        )

    overlapping_lifetimes = GenerativeCorrectionProgramV1(
        topology_events=(
            TopologyEventV1(10, "Lane", "birth", "box", 2, 1, 1, 3, 3),
            TopologyEventV1(11, "Lane", "birth", "box", 1, 4, 4, 6, 6),
        )
    )
    compiled = compile_generative_taskspace_correction(
        _state(),
        overlapping_lifetimes,
        teacher_evidence=_teacher(),
    )
    assert compiled.receipt.max_active_atoms_per_pair == 2


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_exact_custody(
    root: Path,
    *,
    name: str,
    compiled: CompiledGenerativeCorrectionV1,
    d_seg: float,
    d_pose: float,
    include_packet: bool,
    archive_padding: int = 0,
    wrap_packet_in_payload: bool = False,
    archive_packet: bytes | None = None,
) -> ExactEvalCustodyPathsV1:
    custody = root / name
    runtime = custody / "runtime"
    inflated = custody / "inflated"
    upstream = custody / "upstream"
    for directory in (runtime, inflated, upstream):
        directory.mkdir(parents=True)

    evaluator_path = upstream / "evaluate.py"
    evaluator_path.write_text("# frozen upstream evaluator\n", encoding="utf-8")
    video_names_path = upstream / "public_test_video_names.txt"
    video_names_path.write_text("0.mkv\n", encoding="utf-8")
    (upstream / "modules.py").write_text("# frozen scorer module\n", encoding="utf-8")
    (runtime / "inflate.sh").write_text('#!/bin/sh\nexec python3 inflate.py "$@"\n', encoding="utf-8")
    (runtime / "inflate.py").write_text("# standalone receiver\n", encoding="utf-8")

    payload = b"B" * archive_padding
    if wrap_packet_in_payload:
        payload += b"G-PREFIX" + compiled.packet + b"G-SUFFIX"
    archive_path = custody / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.bin", payload or b"baseline")
        if include_packet:
            archive.writestr(
                _G_ARCHIVE_MEMBER,
                compiled.packet if archive_packet is None else archive_packet,
            )
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)

    runtime_rows = []
    for relative in ("inflate.py", "inflate.sh"):
        path = runtime / relative
        runtime_rows.append(
            {
                "relative_path": relative,
                "repo_relative_path": f"{name}/runtime/{relative}",
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    repo_tac = {
        "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
        "discovery": "static_ast_recursive_import_closure",
        "runtime_root_name": runtime.name,
        "tac_root_relative_path": "src/tac",
        "root_import_modules": [],
        "unresolved_modules": [],
        "parse_errors": [],
        "module_count": 0,
        "file_count": 0,
        "files": [],
    }
    evaluator_row = {
        "relative_path": "evaluate.py",
        "bytes": evaluator_path.stat().st_size,
        "sha256": _sha256_file(evaluator_path),
    }
    tree_payload = {
        "runtime_root_name": runtime.name,
        "files": runtime_rows,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": repo_tac,
        "upstream_evaluate_py": evaluator_row,
    }
    content_payload = {
        "files": [{key: row[key] for key in ("relative_path", "bytes", "sha256")} for row in runtime_rows],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {key: value for key, value in repo_tac.items() if key != "runtime_root_name"},
        "upstream_evaluate_py": evaluator_row,
    }
    runtime_manifest = {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": str(runtime),
        "runtime_file_count": len(runtime_rows),
        "runtime_tree_sha256": hashlib.sha256(
            json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "runtime_content_tree_sha256": hashlib.sha256(
            json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": runtime_rows,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": repo_tac,
        "upstream_evaluate_py": evaluator_row,
    }

    raw_path = inflated / "0.raw"
    raw_path.write_bytes((name.encode("ascii") + b"-raw") * 8)
    raw_row = {
        "video_name": "0.mkv",
        "relative_path": "0.raw",
        "exists": True,
        "bytes": raw_path.stat().st_size,
        "sha256": _sha256_file(raw_path),
    }
    raw_aggregate = hashlib.sha256(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": raw_row["relative_path"],
                        "bytes": raw_row["bytes"],
                        "sha256": raw_row["sha256"],
                    }
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    inflated_manifest = {
        "schema": "contest_auth_eval_inflated_output_manifest_v1",
        "inflated_dir": str(inflated),
        "video_names_file": str(video_names_path),
        "raw_file_count": 1,
        "total_bytes": raw_path.stat().st_size,
        "files": [raw_row],
        "aggregate_sha256": raw_aggregate,
    }
    inflated_manifest_path = custody / "inflated_outputs_manifest.json"
    _write_json(inflated_manifest_path, inflated_manifest)

    report_path = custody / "report.txt"
    report_path.write_text(
        "\n".join(
            (
                "=== Evaluation config ===",
                "  device: cpu",
                "=== Evaluation results over 600 samples ===",
                f"  Average PoseNet Distortion: {d_pose:.8f}",
                f"  Average SegNet Distortion: {d_seg:.8f}",
                f"  Submission file size: {archive_bytes:,} bytes",
                "  Original uncompressed size: 37,545,489 bytes",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    upstream_snapshot_sha = compute_upstream_snapshot_sha256(upstream, upstream_subdir=".")
    assert upstream_snapshot_sha is not None
    provenance = {
        "schema_version": 1,
        "tool": "experiments/contest_auth_eval.py",
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "inflate_script": str(runtime / "inflate.sh"),
        "inflate_script_sha256": _sha256_file(runtime / "inflate.sh"),
        "inflate_runtime_manifest": runtime_manifest,
        "inflated_output_manifest": {
            "path": str(inflated_manifest_path),
            "sha256": _sha256_file(inflated_manifest_path),
            "payload": inflated_manifest,
        },
        "upstream_dir": str(upstream),
        "upstream_snapshot_sha256": upstream_snapshot_sha,
        "upstream_commit": "a" * 40,
        "device": "cpu",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "video_names_file": str(video_names_path),
        "sys_argv": [
            "experiments/contest_auth_eval.py",
            "--archive",
            str(archive_path),
            "--inflate-sh",
            str(runtime / "inflate.sh"),
            "--upstream-dir",
            str(upstream),
            "--device",
            "cpu",
            "--work-dir",
            str(custody),
            "--keep-work-dir",
        ],
    }
    provenance_path = custody / "provenance.json"
    _write_json(provenance_path, provenance)
    score = _joint_score(d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes)
    result = {
        "schema_version": 1,
        "avg_posenet_dist": d_pose,
        "avg_segnet_dist": d_seg,
        "rate_unscaled": archive_bytes / 37_545_489,
        "original_uncompressed_size_bytes": 37_545_489,
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "archive_size_bytes": archive_bytes,
        "n_samples": 600,
        "score_axis": "contest_cpu",
        "provenance": provenance,
    }
    result_path = custody / "contest_auth_eval.json"
    _write_json(result_path, result)
    return ExactEvalCustodyPathsV1(
        custody_root=custody,
        result_json_path=result_path,
        archive_path=archive_path,
        provenance_json_path=provenance_path,
        report_path=report_path,
        runtime_root=runtime,
        inflated_root=inflated,
        inflated_outputs_manifest_path=inflated_manifest_path,
        upstream_evaluate_path=evaluator_path,
        video_names_path=video_names_path,
    )


def _write_frontier_pointer(repo_root: Path, score: float) -> Path:
    path = repo_root / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    refreshed_at = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "canonical_frontier_pointer_v1_20260519",
        "our_local_frontier_contest_cpu": None,
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": {
                "score": score,
                "rank": 1,
                "name": "test-frontier",
                "pr_number": 999,
                "pr_url": "https://example.invalid/pr/999",
            }
        },
        "upstream_leaderboard_snapshot_at_utc": refreshed_at,
        "last_refreshed_utc": refreshed_at,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "test-only",
        "refresh_provenance": {"test_fixture": True},
        "effective_frontier": None,
    }
    pointer = CanonicalFrontierPointer.from_dict(payload)
    payload["effective_frontier"] = recompute_effective_frontier(pointer)
    _write_json(path, payload)
    return path


def test_exact_custody_fails_closed_until_receiver_consumption_is_causally_proved(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    candidate = _make_exact_custody(
        tmp_path,
        name="candidate",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    with zipfile.ZipFile(candidate.archive_path) as archive:
        assert archive.read(_G_ARCHIVE_MEMBER) == compiled.packet
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    first_pointer = _write_frontier_pointer(repo_root, 0.17)
    first_pointer_sha256 = _sha256_file(first_pointer)

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="receiver_consumption_custody_absent"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    _write_frontier_pointer(repo_root, 0.18)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="receiver_consumption_custody_absent"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )
    assert first_pointer_sha256 != _sha256_file(repo_root / ".omx/state/canonical_frontier_pointer.json")


def test_fabricated_score_numbers_and_authority_booleans_cannot_admit(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline_fake",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    candidate = _make_exact_custody(
        tmp_path,
        name="candidate_fake",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    repo_root = tmp_path / "repo_fake"
    repo_root.mkdir()
    _write_frontier_pointer(repo_root, 0.18)

    with pytest.raises(TypeError):
        admit_by_exact_coupled_score(  # type: ignore[call-arg]
            compiled,
            state,
            lambda *_args: {"candidate_joint_score": 0.0, "upstream_evaluate": True},
        )

    result = json.loads(candidate.result_json_path.read_text(encoding="utf-8"))
    result.update(
        {
            "avg_segnet_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "canonical_score": 0.0,
            "score_recomputed_from_components": 0.0,
            "score_claim_valid": True,
            "upstream_evaluate": True,
            "same_object_archive_decode": True,
            "correction_packet_embedded": True,
        }
    )
    _write_json(candidate.result_json_path, result)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="distortion coordinates differ"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )


def test_missing_packet_and_mutated_runtime_fail_closed(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline_custody",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    wrapped_packet = _make_exact_custody(
        tmp_path,
        name="candidate_wrapped_packet",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=False,
        wrap_packet_in_payload=True,
    )
    repo_root = tmp_path / "repo_custody"
    repo_root.mkdir()
    _write_frontier_pointer(repo_root, 0.18)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="member size differs"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=wrapped_packet,
            candidate_g_archive_member_path="payload.bin",
            repo_root=repo_root,
        )

    mutated_packet = bytearray(compiled.packet)
    mutated_packet[-1] ^= 1
    nonidentical_member = _make_exact_custody(
        tmp_path,
        name="candidate_nonidentical_member",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
        archive_packet=bytes(mutated_packet),
    )
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="differs from compiled canonical G bytes"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=nonidentical_member,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    archive_mutation = _make_exact_custody(
        tmp_path,
        name="candidate_archive_mutation",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    archive_mutation.archive_path.write_bytes(archive_mutation.archive_path.read_bytes() + b"tamper")
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="report archive bytes differ"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=archive_mutation,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    evaluator_mutation = _make_exact_custody(
        tmp_path,
        name="candidate_evaluator_mutation",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    evaluator_mutation.upstream_evaluate_path.write_text("# tampered evaluator\n", encoding="utf-8")
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="canonical full-tree bytes"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=evaluator_mutation,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    raw_mutation = _make_exact_custody(
        tmp_path,
        name="candidate_raw_mutation",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    (raw_mutation.inflated_root / "0.raw").write_bytes(b"tampered raw output")
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="differs from reopened bytes"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=raw_mutation,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    upstream_mutation = _make_exact_custody(
        tmp_path,
        name="candidate_upstream_mutation",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    (upstream_mutation.upstream_evaluate_path.parent / "modules.py").write_text(
        "# scorer weights/source drifted\n",
        encoding="utf-8",
    )
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="canonical full-tree bytes"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=upstream_mutation,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )

    candidate = _make_exact_custody(
        tmp_path,
        name="candidate_runtime_mutation",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    (candidate.runtime_root / "inflate.py").write_text("# tampered runtime\n", encoding="utf-8")
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="bytes differ from its manifest"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )


def test_exact_admission_recomputes_every_compile_receipt_semantic(tmp_path: Path) -> None:
    state = _state()
    teacher = _teacher()
    compiled = _compile()
    missing = tmp_path / "deliberately_missing"
    dummy_custody = ExactEvalCustodyPathsV1(
        custody_root=missing,
        result_json_path=missing,
        archive_path=missing,
        provenance_json_path=missing,
        report_path=missing,
        runtime_root=missing,
        inflated_root=missing,
        inflated_outputs_manifest_path=missing,
        upstream_evaluate_path=missing,
        video_names_path=missing,
    )

    tested_fields: set[str] = set()
    for field in compiled.receipt.__dataclass_fields__:
        value = getattr(compiled.receipt, field)
        if type(value) is bool:
            forged_value: object = not value
        elif type(value) is int:
            forged_value = value + 1
        elif isinstance(value, CorrectionResourceCountsV1):
            forged_value = replace(value, topology_events=value.topology_events + 1)
        elif isinstance(value, str):
            forged_value = ("0" if value[:1] != "0" else "1") + value[1:] if len(value) == 64 else value + ".forged"
        else:  # pragma: no cover - the receipt ABI is intentionally finite.
            raise AssertionError(f"unhandled receipt field {field}: {type(value)}")
        forged_receipt = replace(compiled.receipt, **{field: forged_value})
        forged = replace(
            compiled,
            receipt=forged_receipt,
            receipt_binding_sha256=forged_receipt.binding_sha256,
        )
        with pytest.raises(GenerativeTaskspaceCorrectionError, match="complete semantic recomputation"):
            admit_by_exact_coupled_score(
                forged,
                state,
                teacher_evidence=teacher,
                baseline_custody=dummy_custody,
                candidate_custody=dummy_custody,
                candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
                repo_root=tmp_path,
            )
        tested_fields.add(field)
    assert tested_fields == set(compiled.receipt.__dataclass_fields__)


def test_missing_full_upstream_snapshot_pin_fails_closed(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline_snapshot_pin",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    candidate = _make_exact_custody(
        tmp_path,
        name="candidate_missing_snapshot_pin",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    provenance = json.loads(candidate.provenance_json_path.read_text(encoding="utf-8"))
    provenance.pop("upstream_snapshot_sha256")
    _write_json(candidate.provenance_json_path, provenance)
    result = json.loads(candidate.result_json_path.read_text(encoding="utf-8"))
    result["provenance"] = provenance
    _write_json(candidate.result_json_path, result)
    repo_root = tmp_path / "repo_snapshot_pin"
    repo_root.mkdir()
    _write_frontier_pointer(repo_root, 0.18)

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="producer lacks canonical"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )


def test_serialized_frontier_cache_must_match_recomputed_constituents(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline_pointer_recompute",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    candidate = _make_exact_custody(
        tmp_path,
        name="candidate_pointer_recompute",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    repo_root = tmp_path / "repo_pointer_recompute"
    repo_root.mkdir()
    pointer_path = _write_frontier_pointer(repo_root, 0.18)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["effective_frontier"]["score"] = 0.000001
    _write_json(pointer_path, pointer)

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="differs from its canonical constituents"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )


def test_fresh_pointer_wrapper_cannot_mask_stale_official_snapshot(tmp_path: Path) -> None:
    state = _state()
    compiled = _compile()
    baseline = _make_exact_custody(
        tmp_path,
        name="baseline_stale_official",
        compiled=compiled,
        d_seg=0.001,
        d_pose=0.001,
        include_packet=False,
        archive_padding=10_000,
    )
    candidate = _make_exact_custody(
        tmp_path,
        name="candidate_stale_official",
        compiled=compiled,
        d_seg=0.0011,
        d_pose=0.0004,
        include_packet=True,
    )
    repo_root = tmp_path / "repo_stale_official"
    repo_root.mkdir()
    pointer_path = _write_frontier_pointer(repo_root, 0.18)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["last_refreshed_utc"] = datetime.now(UTC).isoformat()
    pointer["upstream_leaderboard_snapshot_at_utc"] = "2000-01-01T00:00:00+00:00"
    recomposed = recompute_effective_frontier(CanonicalFrontierPointer.from_dict(pointer))
    assert recomposed is not None
    assert recomposed["snapshot_at_utc"] == pointer["upstream_leaderboard_snapshot_at_utc"]
    pointer["effective_frontier"] = recomposed
    _write_json(pointer_path, pointer)

    with pytest.raises(GenerativeTaskspaceCorrectionError, match="official leaderboard snapshot is stale"):
        admit_by_exact_coupled_score(
            compiled,
            state,
            teacher_evidence=_teacher(),
            baseline_custody=baseline,
            candidate_custody=candidate,
            candidate_g_archive_member_path=_G_ARCHIVE_MEMBER,
            repo_root=repo_root,
        )


def test_palette_only_and_receiver_noop_programs_refuse() -> None:
    palette_only = GenerativeCorrectionProgramV1(realization_profile=_program().realization_profile)
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="palette-only"):
        compile_generative_taskspace_correction(
            _state(),
            palette_only,
            teacher_evidence=_teacher(),
        )

    hidden_event = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(11, "Movable", "death", "box", 1, 0, 0, 2, 2),)
    )
    with pytest.raises(GenerativeTaskspaceCorrectionError, match="receiver-output no-op"):
        compile_generative_taskspace_correction(
            _state(),
            hidden_event,
            teacher_evidence=_teacher(),
        )
