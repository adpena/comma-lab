#!/usr/bin/env python3
"""Measure PF1 program descriptions against identical-content flat controls.

The run is local-only and false-authority.  It first proves byte-identical
description replay for G1, V15, and DV2 under three preregistered formulations,
then rebuilds the V15 archive from the structural PF1 replay and measures that
receiver through the frozen CPU scorer in preserved batches of 16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tac.optimization.ddm_pointfree_program import (  # noqa: E402
    Formulation,
    bundle_rate_row,
    code_compiled_flat,
    code_compiled_program,
    compile_bundle,
    compile_dv2_sentence,
    compile_g1_worldsheet,
    compile_v15_template_bank,
    execute_program,
    frame_flat_sources,
    rate_row,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    EVIDENCE_AXIS,
    REALIZATION_PROFILE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    _decode_lane_knots,
    _decode_lane_programs,
    _decode_realization_profile,
    compile_carrier_compose_archive,
    decode_scorer_solved_template_bank,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    _storage_preflight,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    POINTER_SCORE_TEXT,
    DirectDescriptionError,
    _read_regular_file_once,
    rfc8785_canonicalize,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    _load_models,
    _measure_candidate,
    _publish_immutable,
    _write_checkpoint,
)
from tools.run_ddm_v9_carrier_compose import open_stored_npy_memmap  # noqa: E402

RESULT_SCHEMA: Final = "ddm_pf1_pointfree_program_description_receipt.v2"
RATE_SCHEMA: Final = "ddm_pf1_pointfree_program_rate_matrix.v2"
LANE_ID: Final = "lane_ddm_pf1_pointfree_program_description_20260723"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DDMPF1PointFreeProgramConfigV1(BaseModel):
    """Typed $0 local-advisory contract for the PF1 measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DDMPF1PointFreeProgramConfigV1"] = Field(
        default="DDMPF1PointFreeProgramConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr
    seed: Literal[1234] = 1234
    pair_start: Literal[0] = 0
    pair_count: Literal[600] = 600
    source_archive_path: StrictStr
    source_archive_bytes: StrictInt
    source_archive_sha256: StrictStr
    dv2_typed_path: StrictStr
    dv2_typed_bytes: StrictInt
    dv2_typed_sha256: StrictStr
    dv2_stratum_path: StrictStr
    dv2_stratum_bytes: StrictInt
    dv2_stratum_sha256: StrictStr
    target_cache_path: StrictStr
    target_cache_bytes: StrictInt
    target_cache_sha256: StrictStr
    upstream_root: StrictStr
    scorer_threads: StrictInt = Field(ge=1, le=16)
    scorer_batch_size: Literal[16] = 16
    archive_box_bytes: Literal[160000] = 160000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    paid_dispatch_allowed: Literal[False] = False
    exact_contest_eval_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    training_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMPF1PointFreeProgramConfigV1:
        if not self.run_id.startswith("ddm_pf1_"):
            raise ValueError("PF1 run id must bind the ddm_pf1_ namespace")
        for name in (
            "source_archive_sha256",
            "dv2_typed_sha256",
            "dv2_stratum_sha256",
            "target_cache_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _bound_small_file(path_text: str, expected_bytes: int, expected_sha256: str, label: str) -> bytes:
    path = Path(path_text)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise DirectDescriptionError(f"{label} bytes are unavailable")
    payload = _read_regular_file_once(path)
    if _sha256(payload) != expected_sha256:
        raise DirectDescriptionError(f"{label} SHA-256 differs")
    return payload


def _bind_large_file(path_text: str, expected_bytes: int, expected_sha256: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise DirectDescriptionError(f"{label} bytes are unavailable")
    if _sha256_path(path) != expected_sha256:
        raise DirectDescriptionError(f"{label} SHA-256 differs")
    return path


def _timed_compile(compiler: Any, source: bytes, formulation: Formulation) -> tuple[Any, float]:
    started = time.perf_counter()
    compiled = compiler(source, formulation)
    return compiled, time.perf_counter() - started


def _artifact_row(path: Path, payload: bytes) -> dict[str, Any]:
    _publish_immutable(path, payload)
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _rate_evidence(
    *,
    root: Path,
    source_archive: bytes,
    dv2_typed: bytes,
    dv2_stratum: bytes,
    typed_config_sha256: str,
) -> dict[str, Any]:
    members, _homes = parse_carrier_compose_archive(source_archive)
    g1 = members[WORLDSHEET_G1_MEMBER]
    template = members[SCORER_SOLVED_TEMPLATE_MEMBER]
    inputs = {
        "g1_worldsheet": (g1, compile_g1_worldsheet),
        "v15_template_bank": (template, compile_v15_template_bank),
        "dv2_typed_sentence": (dv2_typed, compile_dv2_sentence),
        "dv2_stratum_sentence": (dv2_stratum, compile_dv2_sentence),
    }
    compiled: dict[tuple[str, Formulation], Any] = {}
    source_by_name = {name: value[0] for name, value in inputs.items()}
    component_rows: list[dict[str, Any]] = []
    programs_root = root / "programs_scope_corrected"

    for name, (source, compiler) in inputs.items():
        for formulation in Formulation:
            program, compile_seconds = _timed_compile(compiler, source, formulation)
            compiled[name, formulation] = program
            if execute_program(program.program) != source:
                raise DirectDescriptionError(f"PF1 {name} public exact replay failed")
            started = time.perf_counter()
            flat_coded = code_compiled_flat(program, source)
            flat_seconds = time.perf_counter() - started
            started = time.perf_counter()
            program_coded = code_compiled_program(program)
            program_seconds = time.perf_counter() - started
            row = rate_row(program, source)
            if (
                row["program_counted_sha256"] != program_coded.framed_sha256
                or row["flat_counted_sha256"] != flat_coded.framed_sha256
            ):
                raise DirectDescriptionError("PF1 timed coder custody differs from the canonical rate row")
            row.update(
                {
                    "description": name,
                    "compile_seconds": f"{compile_seconds:.9f}",
                    "program_coder_seconds": f"{program_seconds:.9f}",
                    "flat_coder_seconds": f"{flat_seconds:.9f}",
                    "wallclock_delta_program_minus_flat_coder_seconds": (
                        f"{program_seconds - flat_seconds:.9f}"
                    ),
                    "program_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.pf1",
                        program.program,
                    ),
                    "program_coded_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.counted",
                        program_coded.payload,
                    ),
                    "flat_coded_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.flat.counted",
                        flat_coded.payload,
                    ),
                }
            )
            component_rows.append(row)

    bundle_specs = {
        "composed_typed": ("g1_worldsheet", "v15_template_bank", "dv2_typed_sentence"),
        "composed_stratum": ("g1_worldsheet", "v15_template_bank", "dv2_stratum_sentence"),
    }
    bundle_rows: list[dict[str, Any]] = []
    for name, child_names in bundle_specs.items():
        sources = tuple(source_by_name[child_name] for child_name in child_names)
        flat = frame_flat_sources(sources)
        for formulation in Formulation:
            children = tuple(compiled[child_name, formulation] for child_name in child_names)
            started = time.perf_counter()
            bundle = compile_bundle(children, formulation, source_replays=sources)
            compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            flat_coded = code_compiled_flat(bundle, flat)
            flat_seconds = time.perf_counter() - started
            started = time.perf_counter()
            program_coded = code_compiled_program(bundle)
            program_seconds = time.perf_counter() - started
            row = bundle_rate_row(bundle, sources)
            row.update(
                {
                    "description": name,
                    "children": list(child_names),
                    "compile_seconds": f"{compile_seconds:.9f}",
                    "program_coder_seconds": f"{program_seconds:.9f}",
                    "flat_coder_seconds": f"{flat_seconds:.9f}",
                    "wallclock_delta_program_minus_flat_coder_seconds": (
                        f"{program_seconds - flat_seconds:.9f}"
                    ),
                    "program_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.pf1",
                        bundle.program,
                    ),
                    "program_coded_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.counted",
                        program_coded.payload,
                    ),
                    "flat_coded_artifact": _artifact_row(
                        programs_root / f"{name}.{formulation.name.lower()}.flat.counted",
                        flat_coded.payload,
                    ),
                }
            )
            bundle_rows.append(row)

    structural_rows = [
        row
        for row in (*component_rows, *bundle_rows)
        if row["formulation"] == Formulation.STRUCTURAL.name
    ]
    formulation_falsifier = {}
    for description in inputs | bundle_specs:
        rows = [
            row
            for row in (*component_rows, *bundle_rows)
            if row["description"] == description
        ]
        scope_eligible = [
            row for row in rows if row["discrete_skeleton_scope_eligible"]
        ]
        formulation_falsifier[description] = {
            "delta_program_minus_flat_by_formulation": {
                row["formulation"]: row["delta_program_minus_flat_bytes"] for row in rows
            },
            "scope_eligible_formulations": [
                row["formulation"] for row in scope_eligible
            ],
            "formulation_closed": len(scope_eligible) >= 3
            and all(row["delta_program_minus_flat_bytes"] >= 0 for row in scope_eligible),
            "closure_rule": (
                "discrete-skeleton rung closes only after >=3 scope-eligible two-typed "
                "formulations are nonnegative; opaque fiber controls never close it"
            ),
        }
    return {
        "schema": RATE_SCHEMA,
        "typed_config_sha256": typed_config_sha256,
        "component_rows": component_rows,
        "bundle_rows": bundle_rows,
        "structural_summary": {
            row["description"]: row["delta_program_minus_flat_bytes"] for row in structural_rows
        },
        "formulation_falsifier": formulation_falsifier,
        "counting_boundary": {
            "generic_interpreter": "FREE_rule118",
            "discrete_skeleton_program": "COUNTED_real_coder",
            "opaque_typed_fibers": "COUNTED_native_analog_coder_not_tokenized",
            "generic_recipe_and_basis_implementation": "FREE_rule118",
        },
        "scope_correction": {
            "directive_utc": "2026-07-24T01:06:43Z",
            "scope": "DISCRETE_SKELETON_ONLY",
            "two_typed_representation": True,
            "continuous_fibers_are_opaque_slots": True,
            "scope_eligible_formulation": "STRUCTURAL",
            "opaque_controls": ["LITERAL", "SHARED_LIBRARY"],
        },
        "semantic_parseback_exact_all_rows": True,
        "score_claim": False,
    }


def _rebuild_archive_from_structural_programs(
    source_archive: bytes,
    root: Path,
) -> tuple[bytes, Any, Any]:
    members, _homes = parse_carrier_compose_archive(source_archive)
    g1_program = _read_regular_file_once(
        root / "programs_scope_corrected/g1_worldsheet.structural.pf1"
    )
    template_program = _read_regular_file_once(
        root / "programs_scope_corrected/v15_template_bank.structural.pf1"
    )
    g1 = execute_program(g1_program)
    template = execute_program(template_program)
    if not isinstance(g1, bytes) or not isinstance(template, bytes):
        raise DirectDescriptionError("PF1 archive program replay returned a bundle")
    program_archive, _rows = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=g1,
        lane_programs=_decode_lane_programs(members.get("predict/lane_periodic_programs.ddlp", b"")),
        lane_knots=_decode_lane_knots(members.get("predict/lane_drift_knots.ddlk", b"")),
        realization_profile=_decode_realization_profile(members.get(REALIZATION_PROFILE_MEMBER, b"")),
        scorer_solved_templates=decode_scorer_solved_template_bank(template),
    )
    if program_archive != source_archive:
        raise DirectDescriptionError("PF1 structural replay changed the V15 archive bytes")
    return (
        program_archive,
        receive_carrier_compose_archive(source_archive),
        receive_carrier_compose_archive(program_archive),
    )


class _ExactPairedReceiver:
    """Render source and PF1-rebuilt receivers together and reject any drift."""

    def __init__(self, source: Any, program: Any, archive_sha256: str) -> None:
        self._source = source
        self._program = program
        self.custody = {
            **dict(program.custody),
            "source_program_archive_byte_identical": True,
            "source_program_archive_sha256": archive_sha256,
            "source_program_camera_equality_checked_before_each_new_scorer_batch": True,
        }

    def render_camera_pairs(self, pair_ids: Any) -> np.ndarray:
        source = self._source.render_camera_pairs(pair_ids)
        program = self._program.render_camera_pairs(pair_ids)
        if not np.array_equal(source, program):
            raise DirectDescriptionError("PF1 source/program receiver camera frames differ")
        return program


def run(
    config: DDMPF1PointFreeProgramConfigV1,
    root: Path,
    semantic_argv: list[str],
) -> Path:
    storage = _storage_preflight(root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = (
        root
        / "ddm_pf1_pointfree_program_description_n600_scope_corrected_receipt.json"
    )
    if receipt_path.exists():
        existing = json.loads(_read_regular_file_once(receipt_path))
        if existing.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed PF1 receipt typed config differs")
        print(json.dumps({"complete": True, "receipt": str(receipt_path), "resumed": True}))
        return receipt_path

    source_archive = _bound_small_file(
        config.source_archive_path,
        config.source_archive_bytes,
        config.source_archive_sha256,
        "source V15 archive",
    )
    dv2_typed = _bound_small_file(
        config.dv2_typed_path,
        config.dv2_typed_bytes,
        config.dv2_typed_sha256,
        "DV2 typed sentence",
    )
    dv2_stratum = _bound_small_file(
        config.dv2_stratum_path,
        config.dv2_stratum_bytes,
        config.dv2_stratum_sha256,
        "DV2 stratum sentence",
    )
    cache_path = _bind_large_file(
        config.target_cache_path,
        config.target_cache_bytes,
        config.target_cache_sha256,
        "frozen n600 target cache",
    )

    rate_path = root / "stage_checkpoints/30_two_typed_rate_matrix.json"
    if not rate_path.exists():
        rate = _rate_evidence(
            root=root,
            source_archive=source_archive,
            dv2_typed=dv2_typed,
            dv2_stratum=dv2_stratum,
            typed_config_sha256=config.typed_config_hash(),
        )
        _write_checkpoint(rate_path, rate)
        print(json.dumps({"complete": False, "measured_stage": "rate_matrix"}))
        return rate_path
    rate = json.loads(_read_regular_file_once(rate_path))
    if rate.get("typed_config_sha256") != config.typed_config_hash():
        raise DirectDescriptionError("PF1 rate checkpoint typed config differs")

    program_archive, source_receiver, program_receiver = _rebuild_archive_from_structural_programs(
        source_archive,
        root,
    )
    archive_sha256 = _sha256(program_archive)
    if archive_sha256 != config.source_archive_sha256:
        raise DirectDescriptionError("PF1 rebuilt archive SHA-256 differs")
    paired_receiver = _ExactPairedReceiver(source_receiver, program_receiver, archive_sha256)
    archive_checkpoint = root / "stage_checkpoints/40_two_typed_receiver_closed_archive.json"
    if not archive_checkpoint.exists():
        _write_checkpoint(
            archive_checkpoint,
            {
                "schema": "ddm_pf1_two_typed_receiver_closed_archive.v2",
                "typed_config_sha256": config.typed_config_hash(),
                "source_archive_bytes": len(source_archive),
                "source_archive_sha256": _sha256(source_archive),
                "program_archive_bytes": len(program_archive),
                "program_archive_sha256": archive_sha256,
                "source_program_archive_byte_identical": True,
                "receiver_custody": paired_receiver.custody,
                "score_claim": False,
            },
        )

    measurement_path = root / "stage_checkpoints/20_frozen_scorer_measurement.json"
    if not measurement_path.exists():
        labels = open_stored_npy_memmap(cache_path, "lstars")
        poses = open_stored_npy_memmap(cache_path, "gt_poses")
        segnet, posenet, scorer_custody = _load_models(config)
        measurement = _measure_candidate(
            name="program_description_structural_receiver",
            archive=program_archive,
            receiver=paired_receiver,
            config=config,
            root=root,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
        )
        measurement["scorer_custody"] = scorer_custody
        measurement["evidence_axis"] = EVIDENCE_AXIS
        measurement["source_program_archive_byte_identical"] = True
        _write_checkpoint(measurement_path, measurement)
        print(json.dumps({"complete": False, "measured_stage": "frozen_scorer_n600"}))
        return measurement_path
    measurement = json.loads(_read_regular_file_once(measurement_path))

    composed = {
        row["description"]: row
        for row in rate["bundle_rows"]
        if row["formulation"] == Formulation.STRUCTURAL.name
    }
    all_falsified = all(
        row["formulation_closed"] for row in rate["formulation_falsifier"].values()
    )
    producer_paths = (
        REPO_ROOT / "src/tac/optimization/ddm_pointfree_program.py",
        REPO_ROOT / "tools/measure_ddm_pf1_pointfree_program.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "producer_custody": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(path)),
            }
            for path in producer_paths
        ],
        "source_custody": {
            "archive": {
                "path": config.source_archive_path,
                "bytes": config.source_archive_bytes,
                "sha256": config.source_archive_sha256,
            },
            "dv2_typed": {
                "path": config.dv2_typed_path,
                "bytes": config.dv2_typed_bytes,
                "sha256": config.dv2_typed_sha256,
            },
            "dv2_stratum": {
                "path": config.dv2_stratum_path,
                "bytes": config.dv2_stratum_bytes,
                "sha256": config.dv2_stratum_sha256,
            },
            "target_cache": {
                "path": config.target_cache_path,
                "bytes": config.target_cache_bytes,
                "sha256": config.target_cache_sha256,
                "mutated": False,
            },
        },
        "rate_matrix": rate,
        "composed_structural_rows": composed,
        "receiver_measurement": measurement,
        "falsifier": {
            "rule": (
                "only scope-eligible two-typed discrete-skeleton formulations count; "
                ">=3 nonnegative formulations are required for closure"
            ),
            "all_descriptions_closed": all_falsified,
            "verdict": (
                "FORMULATION_CLOSED"
                if all_falsified
                else "POSITIVE_DISCRETE_SKELETON_RUNG_SURVIVES_FIBERS_OPEN"
            ),
            "verdict_scope": (
                "PF1 discrete event/template skeleton plus opaque native-coded typed fibers; "
                "continuous-fiber coding families open"
            ),
        },
        "constraint_algebra_fusion": {
            "secondary": True,
            "operator_traces_are_composed_from_measured_basis_only": True,
            "wallclock_deltas_recorded_per_component_and_bundle": True,
            "general_program_synthesis": False,
            "status": "MEASURED_RATE_FACTORIZATION_NOT_AN_OPTIMIZER_CLAIM",
        },
        "route": {
            "pool_key": "program_description",
            "substitutive_against": ["g1", "dv2"],
            "menu1_c1": (
                "admit STRUCTURAL PF1 only where the exact two-typed total delta is "
                "negative; never route opaque controls"
            ),
            "successor_rung": (
                "add two more discrete-skeleton formulations with opaque typed fiber "
                "references; re-measure identical-content two-typed flat controls"
            ),
            "frontier_mutation_authorized": False,
        },
        "fail_closed_mutation_proof": prove_carrier_archive_fail_closed(program_archive),
        "storage_preflight": storage,
        "cleanup": {
            "large_artifacts_created": False,
            "existing_5GB_target_cache": "read_only_memmap",
            "camera_batches": "released_after_each_frozen_scorer_forward",
            "durable_outputs": "small PF1/RC1 programs and JSON checkpoints only",
            "automatic_cleanup_required": False,
            "reason": "no rebuildable bulk or scratch is materialized",
        },
        "resume": {
            "legacy_rate_matrix_checkpoint_preserved": True,
            "two_typed_rate_matrix_checkpoint_preserved": True,
            "two_typed_receiver_archive_checkpoint_preserved": True,
            "per_scorer_batch_checkpoints_preserved": True,
            "batch_size": config.scorer_batch_size,
            "completed_receipt_is_immutable": True,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
            config.source_archive_path,
            config.dv2_typed_path,
            config.dv2_stratum_path,
            config.target_cache_path,
            str(Path(config.upstream_root) / "modules.py"),
            str(Path(config.upstream_root) / "frame_utils.py"),
            str(Path(config.upstream_root) / "evaluate.py"),
        ],
        "scope_correction": {
            "directive_utc": "2026-07-24T01:06:43Z",
            "scope": "DISCRETE_SKELETON_ONLY",
            "necessary_not_sufficient": True,
            "continuous_fibers_not_tokenized": True,
            "typed_fiber_policy": "opaque native-coded slots referenced by skeleton",
            "legacy_receipt_preserved_but_superseded_for_scope_claims": (
                ".omx/research/ddm_pf1_pointfree_program_description_n600_"
                "20260723T235900Z/ddm_pf1_pointfree_program_description_n600_receipt.json"
            ),
        },
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "paid_dispatch_allowed": False,
        "exact_contest_eval_allowed": False,
        "training_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_immutable(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"complete": True, "receipt": str(receipt_path), "resumed": False}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMPF1PointFreeProgramConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    semantic_argv = [
        "tools/measure_ddm_pf1_pointfree_program.py",
        "--config",
        str(args.config),
        "--output-directory",
        str(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
