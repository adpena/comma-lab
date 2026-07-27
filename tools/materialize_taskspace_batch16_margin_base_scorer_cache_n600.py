#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preflight or run the G78/G72 batch16 margin and V15 base-scorer cache.

``--preflight-only`` recursively reopens real G46, G51, and fresh-V15 custody,
rehashes the source/scorer/runtime closure, checks SSD capacity, and writes no
dense data.  ``--materialize`` performs exact global batch-16 CPU forwards and
is admitted only through the governed launcher.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.witness_control.taskspace_batch16_margin_base_scorer_cache_v1 import (  # noqa: E402
    CONFIG_SCHEMA,
    DEFAULT_REQUIRED_FREE_BYTES,
    EVIDENCE_AXIS,
    PREFLIGHT_SCHEMA,
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    PRODUCTION_SCORER_HW,
    PRODUCTION_STAGE_COUNT,
    PRODUCTION_STAGE_PAIRS,
    Batch16MarginBaseScorerCacheError,
    BatchProductsV1,
    MarginBaseScorerCacheLoaderV1,
    PreparedBatchV1,
    file_identity,
    materialize_margin_base_scorer_cache,
    payload_sha256,
    reverify_preflight,
    seal_preflight,
    sha256_array_bytes,
    sha256_file,
    storage_preflight,
    write_immutable_json,
)
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (  # noqa: E402
    FreshScorerPlaneOperandLoaderV1,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    load_compile_ready_materialization_receipt,
    load_json_mapping,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    reverify_preflight as reverify_teacher_preflight,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (  # noqa: E402
    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
    FRESH_V15_BASE_SCORER_CACHE_OWED,
    audit_g72_readiness,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (  # noqa: E402
    verify_v15_semantic_compile_lineage,
)

PACKAGE_DISTRIBUTIONS: Final = (
    "av",
    "einops",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
    "torch",
)
CONFIG_KEYS: Final = {
    "schema",
    "run_id",
    "output_root",
    "pair_count",
    "stage_pairs",
    "scorer_batch_pairs",
    "scorer_hw",
    "class_count",
    "seed",
    "num_threads",
    "required_free_bytes",
    "g46_batch_geometry_audit",
    "g51_aggregate",
    "v15_compile_receipt",
    "v15_semantic_archive",
    "v15_source_config",
    "v15_camera_identity_root",
    "producer_root",
    "truth",
}
IDENTITY_KEYS: Final = {"path", "expected_sha256"}


def _require_identity(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != IDENTITY_KEYS:
        raise Batch16MarginBaseScorerCacheError(f"{label} identity keys differ")
    path = value.get("path")
    expected = value.get("expected_sha256")
    if not isinstance(path, str) or not path or path.strip() != path:
        raise Batch16MarginBaseScorerCacheError(f"{label}.path is invalid")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise Batch16MarginBaseScorerCacheError(f"{label}.expected_sha256 is invalid")
    return {"path": path, "expected_sha256": expected}


def _open_identity(value: Any, label: str) -> Path:
    identity = _require_identity(value, label)
    path = Path(identity["path"]).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != identity["expected_sha256"]:
        raise Batch16MarginBaseScorerCacheError(f"{label} path/type/SHA-256 differs")
    return path


def load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    config = load_json_mapping(resolved)
    if set(config) != CONFIG_KEYS or config.get("schema") != CONFIG_SCHEMA:
        raise Batch16MarginBaseScorerCacheError("typed config keys/schema differ")
    truth = config.get("truth")
    if truth != {
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
    }:
        raise Batch16MarginBaseScorerCacheError("typed config truth boundary differs")
    exact = {
        "pair_count": PRODUCTION_PAIR_COUNT,
        "stage_pairs": PRODUCTION_STAGE_PAIRS,
        "scorer_batch_pairs": PRODUCTION_BATCH_PAIRS,
        "scorer_hw": list(PRODUCTION_SCORER_HW),
        "class_count": 5,
        "required_free_bytes": DEFAULT_REQUIRED_FREE_BYTES,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise Batch16MarginBaseScorerCacheError(
                f"production config {field} differs: {config.get(field)!r} != {expected!r}"
            )
    if (
        not isinstance(config.get("run_id"), str)
        or len(config["run_id"]) < 12
        or not isinstance(config.get("output_root"), str)
        or not isinstance(config.get("producer_root"), str)
        or not isinstance(config.get("v15_camera_identity_root"), str)
    ):
        raise Batch16MarginBaseScorerCacheError("typed config path/run identifiers differ")
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
        raise Batch16MarginBaseScorerCacheError("seed must be an exact integer")
    if (
        not isinstance(config.get("num_threads"), int)
        or isinstance(config["num_threads"], bool)
        or not 1 <= config["num_threads"] <= 32
    ):
        raise Batch16MarginBaseScorerCacheError("num_threads must be in [1,32]")
    for field in (
        "g46_batch_geometry_audit",
        "g51_aggregate",
        "v15_compile_receipt",
        "v15_semantic_archive",
        "v15_source_config",
    ):
        _require_identity(config[field], field)
    return resolved, config


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Batch16MarginBaseScorerCacheError(
                f"required package distribution is unavailable: {distribution}"
            ) from exc
    return versions


def _merge_input_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    roles: dict[str, set[str]] = {}
    for row in rows:
        path = str(row["path"])
        identity = {key: row[key] for key in ("path", "bytes", "sha256")}
        if path in merged and merged[path] != identity:
            raise Batch16MarginBaseScorerCacheError(f"input identity collision: {path}")
        merged[path] = identity
        roles.setdefault(path, set()).add(str(row["role"]))
    return [
        {
            "role": "+".join(sorted(roles[path])),
            **merged[path],
        }
        for path in sorted(merged)
    ]


def _input_row(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, **file_identity(path)}


def _g51_custody(
    aggregate_path: Path,
    expected_sha256: str,
) -> tuple[FreshScorerPlaneOperandLoaderV1, dict[str, Any], list[dict[str, Any]]]:
    loader = FreshScorerPlaneOperandLoaderV1.open(
        aggregate_path,
        expected_sha256=expected_sha256,
    )
    rows: list[dict[str, Any]] = [_input_row("g51_aggregate", aggregate_path)]
    stages: list[dict[str, Any]] = []
    for index, stage_binding in enumerate(loader.receipt["stages"]):
        manifest_path = Path(stage_binding["path"])
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict):
            raise Batch16MarginBaseScorerCacheError("G51 stage manifest is invalid")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise Batch16MarginBaseScorerCacheError("G51 stage files are absent")
        y0 = files["y0_u8"]
        y1 = files["y1_u8"]
        poses = files["gt_poses_f32"]
        rows.extend(
            [
                _input_row(f"g51_stage_{index}_manifest", manifest_path),
                _input_row(f"g51_stage_{index}_y0", Path(y0["path"])),
                _input_row(f"g51_stage_{index}_y1", Path(y1["path"])),
                _input_row(f"g51_stage_{index}_poses_advisory", Path(poses["path"])),
            ]
        )
        stages.append(
            {
                "stage_index": index,
                "pair_range": manifest["pair_range"],
                "stage_receipt_sha256": manifest["stage_receipt_sha256"],
                "manifest": file_identity(manifest_path),
                "y0_u8": dict(y0),
                "y1_u8": dict(y1),
                "gt_poses_f32": dict(poses),
                "y0_y1_rederive_performed_by_g78": False,
                "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
            }
        )
    custody = {
        "aggregate": file_identity(aggregate_path),
        "aggregate_receipt_sha256": loader.receipt["aggregate_receipt_sha256"],
        "stage_digest_chain_sha256": loader.receipt["stage_digest_chain_sha256"],
        "pair_count": loader.pair_count,
        "stage_pairs": loader.stage_pairs,
        "stages": stages,
        "y0_y1_reused_not_rederived": True,
        "candidate_payload_allowed": False,
    }
    return loader, custody, rows


def _v15_identity_checkpoints(
    root: Path,
    *,
    compile_receipt: Mapping[str, Any],
    typed_config_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = root.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise Batch16MarginBaseScorerCacheError("V15 identity root is not a regular directory")
    selected = compile_receipt["solved_template_ladder"][0]
    expected_summary = selected["full_p_camera_identity"]
    expected_names = {
        f"batch_{start:04d}_{min(start + PRODUCTION_BATCH_PAIRS, PRODUCTION_PAIR_COUNT):04d}.json"
        for start in range(0, PRODUCTION_PAIR_COUNT, PRODUCTION_BATCH_PAIRS)
    }
    actual_names = {entry.name for entry in resolved.iterdir() if entry.is_file() and not entry.is_symlink()}
    if actual_names != expected_names:
        raise Batch16MarginBaseScorerCacheError("V15 identity checkpoint set differs")
    chain_material: list[str] = []
    checkpoint_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, PRODUCTION_PAIR_COUNT, PRODUCTION_BATCH_PAIRS)):
        stop = min(start + PRODUCTION_BATCH_PAIRS, PRODUCTION_PAIR_COUNT)
        path = resolved / f"batch_{start:04d}_{stop:04d}.json"
        row = json.loads(path.read_bytes())
        expected_keys = {
            "schema",
            "typed_config_sha256",
            "local_pair_range",
            "base_camera_sha256",
            "final_camera_sha256",
            "byte_identical",
            "camera_bytes_released_after_compare",
            "score_claim",
        }
        if (
            not isinstance(row, dict)
            or set(row) != expected_keys
            or row.get("schema") != "ddm_v15_full_p_camera_identity_batch.v1"
            or row.get("typed_config_sha256") != typed_config_sha256
            or row.get("local_pair_range") != [start, stop]
            or row.get("byte_identical") is not True
            or row.get("camera_bytes_released_after_compare") is not True
            or row.get("score_claim") is not False
            or row.get("base_camera_sha256") != row.get("final_camera_sha256")
        ):
            raise Batch16MarginBaseScorerCacheError(f"V15 identity checkpoint {index} differs")
        chain_material.append(row["base_camera_sha256"] + row["final_camera_sha256"])
        checkpoint_rows.append(
            {
                **file_identity(path),
                "pair_range": [start, stop],
                "camera_sha256": row["base_camera_sha256"],
                "byte_identical": True,
            }
        )
        input_rows.append(_input_row(f"v15_camera_identity_{index}", path))
    chain = hashlib.sha256("".join(chain_material).encode("ascii")).hexdigest()
    if (
        chain != expected_summary["digest_chain_sha256"]
        or expected_summary.get("pair_count") != PRODUCTION_PAIR_COUNT
        or expected_summary.get("batch_size") != PRODUCTION_BATCH_PAIRS
        or expected_summary.get("batch_count") != len(checkpoint_rows)
        or expected_summary.get("all_camera_bytes_identical") is not True
    ):
        raise Batch16MarginBaseScorerCacheError("V15 identity digest chain differs")
    return (
        {
            "root": str(resolved),
            "pair_count": PRODUCTION_PAIR_COUNT,
            "batch_size": PRODUCTION_BATCH_PAIRS,
            "batch_count": len(checkpoint_rows),
            "typed_config_sha256": typed_config_sha256,
            "digest_chain_sha256": chain,
            "checkpoints": checkpoint_rows,
            "all_camera_bytes_identical": True,
        },
        input_rows,
    )


def _module_source(module: str) -> Path | None:
    if not module.startswith("tac"):
        return None
    stem = SRC_ROOT.joinpath(*module.split("."))
    module_path = stem.with_suffix(".py")
    if module_path.is_file():
        return module_path.resolve()
    package_path = stem / "__init__.py"
    return package_path.resolve() if package_path.is_file() else None


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.resolve().relative_to(SRC_ROOT)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _local_imports(path: Path) -> set[str]:
    module, is_package = _module_name(path)
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise Batch16MarginBaseScorerCacheError(f"cannot parse runtime dependency source: {path}") from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("tac"))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package_parts) - (node.level - 1)
            if keep < 0:
                raise Batch16MarginBaseScorerCacheError(f"runtime dependency relative import escapes tac: {path}")
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            imported = ".".join(base_parts)
        else:
            imported = node.module or ""
        if imported.startswith("tac"):
            imports.add(imported)
            for alias in node.names:
                candidate = f"{imported}.{alias.name}"
                if _module_source(candidate) is not None:
                    imports.add(candidate)
    return imports


def _local_runtime_dependency_closure(entry_modules: tuple[str, ...]) -> list[Path]:
    queue = list(entry_modules)
    seen_modules: set[str] = set()
    sources: set[Path] = set()
    while queue:
        module = queue.pop(0)
        if module in seen_modules:
            continue
        seen_modules.add(module)
        source = _module_source(module)
        if source is None:
            continue
        sources.add(source)
        current = source.parent
        while current != SRC_ROOT and current.is_relative_to(SRC_ROOT):
            package_init = current / "__init__.py"
            if package_init.is_file():
                sources.add(package_init.resolve())
                package_module = ".".join(current.relative_to(SRC_ROOT).parts)
                if package_module not in seen_modules:
                    queue.append(package_module)
            current = current.parent
        queue.extend(sorted(_local_imports(source) - seen_modules))
    return sorted(sources, key=lambda path: str(path.relative_to(REPO_ROOT)))


def _runtime_sources() -> list[tuple[str, Path]]:
    direct = [
        (
            "g78_materializer_core",
            REPO_ROOT / "src/tac/witness_control/taskspace_batch16_margin_base_scorer_cache_v1.py",
        ),
        ("g78_materializer_cli", Path(__file__).resolve()),
        (
            "g72_consumer_contract",
            REPO_ROOT / "src/tac/witness_dsl/taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1.py",
        ),
        (
            "v15_lineage_verifier",
            REPO_ROOT / "src/tac/witness_dsl/taskspace_selected_preimage_program_v1.py",
        ),
        (
            "g46_teacher_loader",
            REPO_ROOT / "src/tac/witness_control/taskspace_fresh_teacher_materializer_v1.py",
        ),
        (
            "g51_operand_loader",
            REPO_ROOT / "src/tac/witness_control/taskspace_fresh_scorer_plane_materializer_v1.py",
        ),
    ]
    dependency_sources = _local_runtime_dependency_closure(("tac.optimization.direct_description_carrier_compose",))
    dependency_rows = [
        (
            f"v15_receiver_runtime_dependency:{source.relative_to(REPO_ROOT)}",
            source,
        )
        for source in dependency_sources
    ]
    rows = direct + dependency_rows
    paths = [path.resolve() for _role, path in rows]
    if len(paths) != len(set(paths)):
        raise Batch16MarginBaseScorerCacheError("runtime source closure contains duplicate paths")
    return [(role, path.resolve()) for role, path in rows]


def build_preflight(config_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_config, config = load_config(config_path)
    output_root = Path(config["output_root"]).expanduser().resolve()
    preflight_path = output_root / "00_preflight_receipt.json"
    if preflight_path.exists():
        existing = _load_json_exact(preflight_path, "existing G78 preflight")
        reverify_preflight(existing)
        if (
            existing.get("output_root") != str(output_root)
            or existing.get("config", {}).get("path") != str(resolved_config)
            or existing.get("config", {}).get("sha256") != sha256_file(resolved_config)
        ):
            raise Batch16MarginBaseScorerCacheError(
                "existing immutable preflight names another config/run; choose a new output root"
            )
        return preflight_path, existing
    g46_audit_path = _open_identity(config["g46_batch_geometry_audit"], "G46 audit")
    g51_path = _open_identity(config["g51_aggregate"], "G51 aggregate")
    v15_receipt_path = _open_identity(config["v15_compile_receipt"], "V15 receipt")
    v15_archive_path = _open_identity(config["v15_semantic_archive"], "V15 archive")
    v15_source_config_path = _open_identity(config["v15_source_config"], "V15 source config")
    producer_root = Path(config["producer_root"]).expanduser().resolve(strict=True)

    readiness = audit_g72_readiness(
        semantic_compile_receipt_path=v15_receipt_path,
        semantic_archive_path=v15_archive_path,
        semantic_producer_root=producer_root,
        g46_batch_geometry_audit_path=g46_audit_path,
        g51_operand_aggregate_path=g51_path,
    )
    g46_audit = json.loads(g46_audit_path.read_bytes())
    primary = g46_audit["primary"]
    g46_receipt_path = Path(primary["receipt_file"]["path"])
    g46_receipt = load_compile_ready_materialization_receipt(g46_receipt_path)
    teacher_preflight_path = Path(primary["preflight_file"]["path"])
    teacher_preflight = _load_json_exact(teacher_preflight_path, "G46 teacher preflight")
    reverify_teacher_preflight(teacher_preflight)
    if (
        g46_receipt["receipt_sha256"] != primary["receipt_sha256"]
        or g46_receipt["target_labels"]["sha256"] != primary["target_labels"]["sha256"]
        or g46_receipt["scorer_pair_batch_size"] != PRODUCTION_BATCH_PAIRS
    ):
        raise Batch16MarginBaseScorerCacheError("G46 primary custody differs")

    g51_loader, g51_custody, g51_rows = _g51_custody(
        g51_path,
        config["g51_aggregate"]["expected_sha256"],
    )
    if (
        g51_loader.receipt["target_labels"]["sha256"] != g46_receipt["target_labels"]["sha256"]
        or g51_loader.pair_count != PRODUCTION_PAIR_COUNT
        or g51_loader.stage_pairs != PRODUCTION_STAGE_PAIRS
    ):
        raise Batch16MarginBaseScorerCacheError("G51 does not bind the reopened G46 coordinate")

    v15_receipt_bytes = v15_receipt_path.read_bytes()
    v15_archive = v15_archive_path.read_bytes()
    semantic_identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=v15_receipt_bytes,
        compiled_semantic_archive=v15_archive,
        producer_root=producer_root,
    )
    v15_receipt = json.loads(v15_receipt_bytes)
    source_config = json.loads(v15_source_config_path.read_bytes())
    if hashlib.sha256(rfc8785_canonicalize(source_config)).hexdigest() != semantic_identity.typed_compile_config_sha256:
        raise Batch16MarginBaseScorerCacheError("V15 source config canonical hash differs")
    receiver = receive_carrier_compose_archive(v15_archive, verify_member_effects=True)
    if (
        receiver.archive != v15_archive
        or receiver.custody.get("archive_sha256") != semantic_identity.compiled_semantic_archive_sha256
    ):
        raise Batch16MarginBaseScorerCacheError("V15 strict receiver custody differs")
    v15_identity, v15_identity_rows = _v15_identity_checkpoints(
        Path(config["v15_camera_identity_root"]),
        compile_receipt=v15_receipt,
        typed_config_sha256=semantic_identity.typed_compile_config_sha256,
    )

    storage = storage_preflight(
        output_root,
        required_free_bytes=config["required_free_bytes"],
    )
    input_rows: list[dict[str, Any]] = [
        _input_row("typed_config", resolved_config),
        _input_row("g46_batch_geometry_audit", g46_audit_path),
        _input_row("g46_compile_ready_receipt", g46_receipt_path),
        _input_row("g46_teacher_preflight", teacher_preflight_path),
        _input_row("source_video", Path(teacher_preflight["source_video"]["path"])),
        _input_row("segnet_weights", Path(teacher_preflight["segnet_weights"]["path"])),
        _input_row("g46_target_labels", Path(g46_receipt["target_labels"]["path"])),
        _input_row("v15_compile_receipt", v15_receipt_path),
        _input_row("v15_semantic_archive", v15_archive_path),
        _input_row("v15_source_config", v15_source_config_path),
    ]
    input_rows.extend(g51_rows)
    input_rows.extend(v15_identity_rows)
    for member in teacher_preflight["upstream_closure"]["members"]:
        input_rows.append(_input_row(f"upstream_{member['relative_path']}", Path(member["path"])))
    for role, source in _runtime_sources():
        input_rows.append(_input_row(role, source))
    for producer in v15_receipt["producer_custody"]:
        input_rows.append(_input_row("v15_producer", producer_root / producer["path"]))
    sealed_inputs = _merge_input_rows(input_rows)
    runtime_rows = [{"role": role, **file_identity(source)} for role, source in _runtime_sources()]
    target_binding = {
        **{key: g46_receipt["target_labels"][key] for key in ("path", "bytes", "sha256", "shape", "dtype")},
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }
    run_argv = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        str(resolved_config),
        "--materialize",
    ]
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": config["run_id"],
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "output_root": str(output_root),
        "pair_count": config["pair_count"],
        "stage_pairs": config["stage_pairs"],
        "stage_count": PRODUCTION_STAGE_COUNT,
        "scorer_batch_pairs": config["scorer_batch_pairs"],
        "scorer_hw": config["scorer_hw"],
        "class_count": config["class_count"],
        "seed": config["seed"],
        "num_threads": config["num_threads"],
        "test_only_small_fixture": False,
        "storage_preflight": storage,
        "config": {
            **file_identity(resolved_config),
            "canonical_sha256": payload_sha256(config),
        },
        "source_custody": {
            "source_video": file_identity(teacher_preflight["source_video"]["path"]),
            "source_sequence_length": 2,
            "pairing": "AVVideoDataset_NONOVERLAPPING_CONTIGUOUS_PAIRS",
            "upstream_closure": teacher_preflight["upstream_closure"],
        },
        "scorer_custody": {
            "weights": file_identity(teacher_preflight["segnet_weights"]["path"]),
            "model": "upstream.modules.SegNet",
            "device": "cpu",
            "batch_size": PRODUCTION_BATCH_PAIRS,
            "final_partial_batch_size": 8,
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
            "live_r": "SegNet.preprocess_input(last_frame)->bilinear float32 384x512",
            "package_versions": package_versions(),
        },
        "target_custody": {
            "g46_audit": file_identity(g46_audit_path),
            "g46_audit_sha256": g46_audit["audit_sha256"],
            "g46_receipt": file_identity(g46_receipt_path),
            "g46_receipt_sha256": g46_receipt["receipt_sha256"],
            "target_labels": target_binding,
            "labels_reused_not_rederived": True,
            "target_margins_present_before_g78": False,
        },
        "g51_y0_y1_custody": g51_custody,
        "semantic_custody": {
            "compile_receipt": file_identity(v15_receipt_path),
            "compile_receipt_sha256": semantic_identity.fresh_compile_receipt_sha256,
            "compile_proof_dependency_sha256": (semantic_identity.compile_proof_dependency_sha256),
            "source_config": {
                **file_identity(v15_source_config_path),
                "rfc8785_sha256": semantic_identity.typed_compile_config_sha256,
            },
            "archive": file_identity(v15_archive_path),
            "receiver_source_sha256": semantic_identity.receiver_source_sha256,
            "compiler_source_sha256": semantic_identity.compiler_source_sha256,
            "receiver_contract_id": semantic_identity.receiver_contract_id,
            "executed_receiver_contract_id": (
                "tac.optimization.direct_description_carrier_compose.CarrierComposeReceiverV1.render_camera_pairs.v15"
            ),
            "receiver_strict_parse": True,
            "full_p_camera_identity": v15_identity,
        },
        "runtime_custody": {
            "files": runtime_rows,
            "upstream_closure_sha256": teacher_preflight["upstream_closure"]["closure_sha256"],
            "package_versions": package_versions(),
            "python": sys.version.split()[0],
        },
        "sealed_input_files": sealed_inputs,
        "run_argv": run_argv,
        "resume_contract": {
            "global_batch_checkpoints": 38,
            "global_batch_atomic": True,
            "completed_batches_skip_scorer_forward": True,
            "completed_batch_source_and_scorer_input_rehashed": True,
            "completed_batch_v15_camera_and_live_r_input_rehashed": True,
            "fresh_v15_camera_must_equal_owned_identity_before_forward": True,
            "stage_checkpoints": 5,
            "stage_pairs": 120,
            "stage_atomic": True,
            "stage_immutable": True,
            "batch_to_stage_cross_boundary_fragments_bound": True,
            "stage_dense_bytes_rederived_from_batch_fragments_on_reopen": True,
            "stage_g51_binding_reopened_and_compared": True,
        },
        "blockers_closed_by_successful_aggregate": [
            FRESH_BATCH16_MARGIN_CUSTODY_OWED,
            FRESH_V15_BASE_SCORER_CACHE_OWED,
        ],
    }
    preflight = seal_preflight(body)
    output_root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(preflight_path, preflight)
    reopened = _load_json_exact(preflight_path, "G78 preflight")
    reverify_preflight(reopened)
    if reopened != preflight:
        raise Batch16MarginBaseScorerCacheError("preflight changed across parse-back")
    # This is deliberately retained as a source audit: it still reports both
    # blockers open because only a successful aggregate closes them.
    _ = readiness
    return preflight_path, preflight


def _load_json_exact(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Batch16MarginBaseScorerCacheError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Batch16MarginBaseScorerCacheError(f"{label} is not a JSON object")
    return value


def _configure_determinism(*, seed: int, num_threads: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def _winner_fields(logits):
    import torch

    cells = logits.argmax(dim=1)
    winner = logits.gather(1, cells.unsqueeze(1))
    masked = logits.masked_fill(
        torch.nn.functional.one_hot(cells, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool(),
        -torch.inf,
    )
    runner_up = masked.max(dim=1, keepdim=True).values
    margins = winner - runner_up
    return (
        cells.to(torch.uint8).cpu().numpy(),
        margins[:, 0].to(torch.float32).cpu().numpy(),
    )


class _ProductionBatchPreparer:
    def __init__(self, preflight: Mapping[str, Any]) -> None:
        import torch
        from safetensors.torch import load_file

        upstream_root = Path(preflight["source_custody"]["upstream_closure"]["root"])
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        from frame_utils import AVVideoDataset
        from modules import SegNet

        weights = Path(preflight["scorer_custody"]["weights"]["path"])
        if sha256_file(weights) != preflight["scorer_custody"]["weights"]["sha256"]:
            raise Batch16MarginBaseScorerCacheError("SegNet weights drifted before load")
        self.model = SegNet().eval().to(device=torch.device("cpu"))
        self.model.load_state_dict(load_file(weights, device="cpu"))
        source = Path(preflight["source_custody"]["source_video"]["path"])
        self.source_path = source.resolve()
        self.next_batch_index = 1
        dataset = AVVideoDataset(
            [source.name],
            data_dir=source.parent,
            batch_size=PRODUCTION_BATCH_PAIRS,
            device=torch.device("cpu"),
            num_threads=int(preflight["num_threads"]),
            seed=int(preflight["seed"]),
            prefetch_queue_depth=1,
        )
        dataset.prepare_data()
        self.source_iterator = iter(dataset)
        archive_path = Path(preflight["semantic_custody"]["archive"]["path"])
        archive = archive_path.read_bytes()
        if sha256_file(archive_path) != preflight["semantic_custody"]["archive"]["sha256"]:
            raise Batch16MarginBaseScorerCacheError("V15 semantic archive drifted before run")
        self.receiver = receive_carrier_compose_archive(
            archive,
            verify_member_effects=True,
        )

    def _scorer_input(self, batch: np.ndarray):
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(batch))
        tensor = tensor.permute(0, 1, 4, 2, 3).float().contiguous()
        with torch.inference_mode():
            return self.model.preprocess_input(tensor).contiguous()

    def __call__(self, pair_ids: tuple[int, ...]) -> PreparedBatchV1:
        import torch

        try:
            path, batch_index, batch = next(self.source_iterator)
        except StopIteration as exc:
            raise Batch16MarginBaseScorerCacheError("AVVideoDataset ended before requested n600 population") from exc
        expected_batch_index = pair_ids[0] // PRODUCTION_BATCH_PAIRS + 1
        if (
            Path(path).resolve() != self.source_path
            or int(batch_index) != expected_batch_index
            or int(batch_index) != self.next_batch_index
        ):
            raise Batch16MarginBaseScorerCacheError(
                f"AVVideoDataset source/index differs at requested batch {pair_ids[0]}:{pair_ids[-1] + 1}"
            )
        self.next_batch_index += 1
        source = np.ascontiguousarray(batch.cpu().numpy(), dtype=np.uint8)
        if source.shape[0] != len(pair_ids):
            raise Batch16MarginBaseScorerCacheError(
                f"source batch size differs at {pair_ids[0]}: {source.shape[0]} != {len(pair_ids)}"
            )
        target_input = self._scorer_input(source)
        source_sha = sha256_array_bytes(source)
        target_input_sha = sha256_array_bytes(target_input.cpu().numpy())
        camera = np.ascontiguousarray(
            self.receiver.render_camera_pairs(pair_ids),
            dtype=np.uint8,
        )
        camera_input = self._scorer_input(camera)
        camera_sha = sha256_array_bytes(camera)
        camera_input_sha = sha256_array_bytes(camera_input.cpu().numpy())

        def infer() -> BatchProductsV1:
            with torch.inference_mode():
                target_logits = self.model(target_input)
            target_cells, target_margins = _winner_fields(target_logits)
            with torch.inference_mode():
                described_logits = self.model(camera_input)
            described_cells, described_margins = _winner_fields(described_logits)
            return BatchProductsV1(
                target_cells_u8=np.ascontiguousarray(target_cells, dtype=np.uint8),
                target_margins_f32=np.ascontiguousarray(target_margins, dtype=np.float32),
                described_cells_u8=np.ascontiguousarray(described_cells, dtype=np.uint8),
                described_margins_f32=np.ascontiguousarray(
                    described_margins,
                    dtype=np.float32,
                ),
            )

        return PreparedBatchV1(
            source_pair_batch_sha256=source_sha,
            target_scorer_input_sha256=target_input_sha,
            v15_camera_sha256=camera_sha,
            v15_scorer_input_sha256=camera_input_sha,
            infer=infer,
        )


def run_materialization(config_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_config, config = load_config(config_path)
    preflight_path = Path(config["output_root"]) / "00_preflight_receipt.json"
    if not preflight_path.is_file():
        raise Batch16MarginBaseScorerCacheError("materialization requires an existing strict preflight receipt")
    preflight = _load_json_exact(preflight_path, "G78 preflight")
    reverify_preflight(preflight)
    if preflight["config"]["sha256"] != sha256_file(resolved_config):
        raise Batch16MarginBaseScorerCacheError("typed config changed after preflight")
    _configure_determinism(
        seed=int(preflight["seed"]),
        num_threads=int(preflight["num_threads"]),
    )
    return materialize_margin_base_scorer_cache(
        preflight=preflight,
        prepare_batch=_ProductionBatchPreparer(preflight),
    )


def run_status(config_path: Path) -> tuple[Path, dict[str, Any]]:
    _resolved_config, config = load_config(config_path)
    aggregate_path = Path(config["output_root"]) / "aggregate_receipt.json"
    loader = MarginBaseScorerCacheLoaderV1.open(
        aggregate_path,
        expected_sha256=sha256_file(aggregate_path),
    )
    return aggregate_path, loader.receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight_only:
            path, receipt = build_preflight(args.config)
            kind = "preflight"
            self_hash = receipt["preflight_sha256"]
        elif args.materialize:
            assert_governed_admission(
                "taskspace_batch16_margin_base_scorer_cache_n600",
                on_refuse="raise",
            )
            path, receipt = run_materialization(args.config)
            kind = "aggregate"
            self_hash = receipt["aggregate_receipt_sha256"]
        else:
            path, receipt = run_status(args.config)
            kind = "aggregate_status"
            self_hash = receipt["aggregate_receipt_sha256"]
    except (
        Batch16MarginBaseScorerCacheError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": kind,
                "receipt": file_identity(path),
                "sealed_self_sha256": self_hash,
                "closed_only_after_aggregate": [
                    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
                    FRESH_V15_BASE_SCORER_CACHE_OWED,
                ],
                "pointer_moved": False,
                "score_claim": False,
                "candidate_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
