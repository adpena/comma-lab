# SPDX-License-Identifier: MIT
"""Fail-closed contract for a native-MLX SNeRV train/export/archive adapter.

The current SNeRV execution path is a real receiver-priced CPU advisory lane.
It is useful, but it is not the native MLX train/export/archive adapter required
for long campaigns. This module makes that missing surface machine-checkable so
the runner can discover a real adapter when it lands and refuse fake execution
until the required symbols and signatures exist.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA = "snerv_mlx_native_adapter_contract.v1"
SNERV_MLX_NATIVE_TRAINING_EXPORT_GUARD_SCHEMA = (
    "snerv_mlx_native_training_export_guard.v1"
)
DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE = (
    "tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export"
)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

DEFAULT_REQUIRED_NATIVE_EXPORT_PAIRS = 600


@dataclass(frozen=True)
class RequiredSurface:
    """One required callable surface for the native adapter."""

    surface_id: str
    symbol: str
    required_parameters: tuple[str, ...]
    rationale: str


REQUIRED_SURFACES: tuple[RequiredSurface, ...] = (
    RequiredSurface(
        surface_id="train_export",
        symbol="train_export_snerv_mlx_native",
        required_parameters=(
            "output_dir",
            "num_pairs",
            "source_video_path",
            "modelsize_candidate",
            "scorer_upstream_dir",
        ),
        rationale=(
            "native MLX training must bind selected modelsize bytes, real video, "
            "and scorer custody in the same callable"
        ),
    ),
    RequiredSurface(
        surface_id="archive_export",
        symbol="export_snerv_mlx_archive",
        required_parameters=("model_or_artifact", "output_dir", "repo_root"),
        rationale=(
            "training is not useful until it emits a byte-closed contest archive"
        ),
    ),
    RequiredSurface(
        surface_id="receiver_proof",
        symbol="write_snerv_mlx_receiver_proof",
        required_parameters=("archive_zip_path", "runtime_submission_dir", "output_dir"),
        rationale="every native archive must prove inflate/runtime consumption",
    ),
    RequiredSurface(
        surface_id="local_mlx_prefilter",
        symbol="write_snerv_mlx_prefilter_profile",
        required_parameters=(
            "artifact",
            "archive_bytes",
            "archive_sha256",
            "output_path",
            "upstream_dir",
        ),
        rationale=(
            "CPU replay and exact auth are gated by full-video local MLX "
            "acquisition evidence"
        ),
    ),
)


def _signature_accepts_required_parameters(
    fn: Any,
    required_parameters: tuple[str, ...],
) -> tuple[bool, list[str], str | None]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return False, list(required_parameters), f"signature_unreadable:{exc!s}"
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True, [], None
    missing = [name for name in required_parameters if name not in params]
    return not missing, missing, None


def _resolve_evidence_path(
    value: Any,
    *,
    base_dir: str | Path | None,
) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser() / path
    return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_evidence_row(
    *,
    field_id: str,
    path_value: Any,
    expected_sha256: Any = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = _resolve_evidence_path(path_value, base_dir=base_dir)
    exists = bool(path is not None and path.is_file())
    actual_sha256 = _sha256_file(path) if exists and path is not None else None
    expected = str(expected_sha256) if expected_sha256 else None
    sha256_matches = bool(actual_sha256 and (expected is None or actual_sha256 == expected))
    return {
        "schema": "snerv_mlx_native_file_evidence_row.v1",
        "field_id": str(field_id),
        "path": path.as_posix() if path is not None else None,
        "exists": exists,
        "bytes": int(path.stat().st_size) if exists and path is not None else None,
        "expected_sha256": expected,
        "sha256": actual_sha256,
        "sha256_matches": sha256_matches,
    }


def _load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _merged_artifact_mapping(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    art = dict(artifact or {})
    if art.get("schema") == "snerv_mlx_native_file_backed_evidence.v1":
        report = art.get("report_file") if isinstance(art.get("report_file"), Mapping) else {}
        packet = art.get("packet_file") if isinstance(art.get("packet_file"), Mapping) else {}
        archive = art.get("archive_file") if isinstance(art.get("archive_file"), Mapping) else {}
        proof = (
            art.get("receiver_proof_file")
            if isinstance(art.get("receiver_proof_file"), Mapping)
            else {}
        )
        return {
            "schema": art.get("schema"),
            "num_pairs": art.get("num_pairs"),
            "artifact_report_path": report.get("path"),
            "packet_path": packet.get("path"),
            "packet_sha256": packet.get("sha256"),
            "archive_path": archive.get("path"),
            "archive_sha256": archive.get("sha256"),
            "receiver_proof_path": proof.get("path"),
            "receiver_proof_passed": art.get("receiver_proof_passed"),
            "receiver_contract_satisfied": art.get("receiver_contract_satisfied"),
        }
    nested = art.get("artifact")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update(art)
        return merged
    return art


def _native_training_payload(art: Mapping[str, Any], *, base_dir: str | Path | None) -> dict[str, Any]:
    report_path = (
        art.get("artifact_report_path")
        or art.get("report_path")
        or (
            dict(art.get("report_file") or {}).get("path")
            if isinstance(art.get("report_file"), Mapping)
            else None
        )
    )
    report_payload = _load_json_file(_resolve_evidence_path(report_path, base_dir=base_dir))
    merged = dict(report_payload)
    merged.update({key: value for key, value in art.items() if value is not None})
    nested = merged.get("artifact")
    if isinstance(nested, Mapping):
        nested_payload = dict(nested)
        nested_payload.update(merged)
        merged = nested_payload
    return merged


def build_snerv_mlx_native_training_export_guard(
    artifact: Mapping[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed when a requested native MLX decoder step was rejected.

    The native exporter may still emit the closed-form fallback packet after a
    divergent MLX refinement. That file-backed packet is useful forensic
    evidence, but it is not a learned-training candidate and must not unlock
    planner/export success surfaces.
    """

    art = _native_training_payload(
        _merged_artifact_mapping(artifact),
        base_dir=base_dir,
    )
    training = art.get("native_mlx_hf_decoder_training")
    training = dict(training) if isinstance(training, Mapping) else {}
    source_blockers = [
        str(blocker)
        for blocker in (
            list(training.get("blockers") or ())
            + list(art.get("native_mlx_training_blockers") or ())
        )
        if blocker
    ]
    requested_steps = _int_or_none(
        training.get("requested_steps")
        if training.get("requested_steps") is not None
        else training.get("steps")
    )
    attempted = bool(
        training.get("attempted") is True
        or (requested_steps is not None and requested_steps > 0)
        or art.get("native_mlx_training_requested") is True
    )
    training_executed = bool(
        training.get("executed") is True
        or art.get("native_mlx_training_executed") is True
    )
    loss_worsened = bool(
        training.get("any_loss_worsened") is True
        or "snerv_native_mlx_decoder_loss_worsened" in source_blockers
    )
    final_nonfinite = bool(
        training.get("all_final_losses_finite") is False
        or "snerv_native_mlx_decoder_final_loss_nonfinite" in source_blockers
    )
    accepted_value = training.get("accepted")
    accepted = bool(
        accepted_value is True
        or (
            accepted_value is None
            and training_executed
            and not source_blockers
            and not loss_worsened
            and not final_nonfinite
        )
    )
    blockers: list[str] = []
    if attempted and not accepted:
        if loss_worsened:
            blockers.append("snerv_native_mlx_decoder_loss_worsened_export_blocked")
        if final_nonfinite:
            blockers.append("snerv_native_mlx_decoder_final_loss_nonfinite_export_blocked")
        if not blockers:
            blockers.append("snerv_native_mlx_decoder_training_rejected_export_blocked")
    if attempted and accepted and not training_executed:
        blockers.append("snerv_native_mlx_decoder_training_acceptance_mismatch")
    return {
        "schema": SNERV_MLX_NATIVE_TRAINING_EXPORT_GUARD_SCHEMA,
        "training_attempted": attempted,
        "requested_steps": requested_steps,
        "training_executed": training_executed,
        "training_accepted": accepted,
        "loss_worsened": loss_worsened,
        "final_losses_finite": None if "all_final_losses_finite" not in training else bool(training.get("all_final_losses_finite")),
        "source_training_blockers": list(dict.fromkeys(source_blockers)),
        "worsened_level_subband_rows": list(training.get("worsened_level_subband_rows") or ()),
        "export_guard_passed": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def build_snerv_mlx_native_file_backed_evidence(
    artifact: Mapping[str, Any] | None,
    *,
    required_num_pairs: int = DEFAULT_REQUIRED_NATIVE_EXPORT_PAIRS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate SNeRV native export evidence against files, not booleans.

    The runner and planner both use this helper so a hand-written
    ``receiver_proof_passed=True`` cannot clear the native SNeRV blocker unless
    the referenced packet/archive/proof files exist and their hashes agree.
    """

    art = _merged_artifact_mapping(artifact)
    rows = {
        "report": _file_evidence_row(
            field_id="report",
            path_value=art.get("artifact_report_path") or art.get("report_path"),
            base_dir=base_dir,
        ),
        "packet": _file_evidence_row(
            field_id="packet",
            path_value=art.get("packet_path"),
            expected_sha256=art.get("packet_sha256"),
            base_dir=base_dir,
        ),
        "archive": _file_evidence_row(
            field_id="archive",
            path_value=art.get("archive_path"),
            expected_sha256=art.get("archive_sha256"),
            base_dir=base_dir,
        ),
        "receiver_proof": _file_evidence_row(
            field_id="receiver_proof",
            path_value=art.get("receiver_proof_path"),
            base_dir=base_dir,
        ),
    }
    proof_path = (
        Path(rows["receiver_proof"]["path"])
        if rows["receiver_proof"]["path"]
        else None
    )
    proof_payload = _load_json_file(proof_path)
    proof_contract_satisfied = proof_payload.get("receiver_contract_satisfied") is True
    proof_runtime_passed = (
        proof_payload.get("runtime_consumption_proof_passed") is True
        or proof_payload.get("runtime_consumption_proof_ready") is True
        or proof_payload.get("receiver_proof_valid") is True
    )
    receiver_contract_satisfied = bool(proof_contract_satisfied or proof_runtime_passed)
    receiver_proof_passed = bool(proof_runtime_passed or proof_contract_satisfied)
    num_pairs = _int_or_none(art.get("num_pairs"))
    required_pair_coverage_passed = bool(
        num_pairs is not None and int(num_pairs) >= int(required_num_pairs)
    )
    blockers: list[str] = []
    for field_id, row in rows.items():
        if not row["exists"]:
            blockers.append(f"snerv_mlx_native_{field_id}_file_missing")
        elif row["expected_sha256"] and not row["sha256_matches"]:
            blockers.append(f"snerv_mlx_native_{field_id}_sha256_mismatch")
    if not receiver_proof_passed:
        blockers.append("snerv_mlx_native_receiver_proof_file_not_passing")
    if not receiver_contract_satisfied:
        blockers.append("snerv_mlx_native_receiver_contract_file_not_satisfied")
    if not required_pair_coverage_passed:
        blockers.append("snerv_mlx_native_file_backed_export_not_full600")
    training_guard = build_snerv_mlx_native_training_export_guard(
        art,
        base_dir=base_dir,
    )
    blockers.extend(str(blocker) for blocker in training_guard.get("blockers") or ())
    file_backed_export_proof_passed = bool(
        rows["report"]["exists"]
        and rows["packet"]["sha256_matches"]
        and rows["archive"]["sha256_matches"]
        and rows["receiver_proof"]["exists"]
        and receiver_proof_passed
        and receiver_contract_satisfied
        and training_guard.get("export_guard_passed") is True
    )
    scorer_loop_raw = art.get("scorer_loop_qat")
    scorer_loop_present = isinstance(scorer_loop_raw, Mapping)
    scorer_loop = dict(scorer_loop_raw) if scorer_loop_present else {}
    scorer_loop_qat_attached = (
        scorer_loop.get("executed")
        if scorer_loop_present
        else art.get("scorer_loop_qat_attached")
    )
    scorer_loop_qat_accepted_improvement = (
        scorer_loop.get("accepted_improvement")
        if scorer_loop_present
        else art.get("scorer_loop_qat_accepted_improvement")
    )
    scorer_loop_qat_best_materialized = (
        scorer_loop.get("emitted_packet_uses_scorer_loop_best_decoder")
        if scorer_loop_present
        else art.get("scorer_loop_qat_best_materialized")
    )
    return {
        "schema": "snerv_mlx_native_file_backed_evidence.v1",
        "required_num_pairs": int(required_num_pairs),
        "num_pairs": num_pairs,
        "executed": bool(art.get("executed") is True or art.get("schema")),
        "packet_source": art.get("packet_source"),
        "report_file": rows["report"],
        "packet_file": rows["packet"],
        "archive_file": rows["archive"],
        "receiver_proof_file": rows["receiver_proof"],
        "receiver_proof_passed": receiver_proof_passed,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "reported_receiver_proof_passed": bool(art.get("receiver_proof_passed")),
        "reported_receiver_contract_satisfied": bool(
            art.get("receiver_contract_satisfied")
        ),
        "required_pair_coverage_passed": required_pair_coverage_passed,
        "native_mlx_training_export_guard": training_guard,
        "native_mlx_training_export_guard_passed": bool(
            training_guard.get("export_guard_passed")
        ),
        "file_backed_export_proof_passed": file_backed_export_proof_passed,
        "required_pair_file_backed_export_proof_passed": bool(
            file_backed_export_proof_passed and required_pair_coverage_passed
        ),
        "scorer_loop_qat_attached": bool(scorer_loop_qat_attached),
        "scorer_loop_qat_accepted_improvement": bool(
            scorer_loop_qat_accepted_improvement
        ),
        "scorer_loop_qat_best_materialized": bool(
            scorer_loop_qat_best_materialized
        ),
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_snerv_mlx_native_adapter_contract(
    *,
    module_name: str = DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE,
    extra_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fail-closed native-MLX adapter readiness for SNeRV."""

    blockers: list[str] = []
    surface_rows: list[dict[str, Any]] = []
    module_loaded = False
    module_error: str | None = None
    module: Any | None = None
    try:
        module = importlib.import_module(module_name)
        module_loaded = True
    except Exception as exc:  # pragma: no cover - exact exception is environment-specific
        module_error = repr(exc)
        blockers.append("snerv_mlx_native_train_export_archive_adapter_missing")

    for surface in REQUIRED_SURFACES:
        symbol_present = module_loaded and hasattr(module, surface.symbol)
        callable_present = False
        signature_ok = False
        missing_parameters: list[str] = list(surface.required_parameters)
        signature_error: str | None = None
        if symbol_present:
            value = getattr(module, surface.symbol)
            callable_present = callable(value)
            if callable_present:
                signature_ok, missing_parameters, signature_error = (
                    _signature_accepts_required_parameters(
                        value,
                        surface.required_parameters,
                    )
                )
        if not symbol_present:
            blockers.append(f"snerv_mlx_native_surface_missing:{surface.surface_id}")
        elif not callable_present:
            blockers.append(
                f"snerv_mlx_native_surface_not_callable:{surface.surface_id}"
            )
        elif not signature_ok:
            blockers.append(
                f"snerv_mlx_native_surface_signature_mismatch:{surface.surface_id}"
            )
        surface_rows.append(
            {
                "schema": "snerv_mlx_native_adapter_surface_row.v1",
                "surface_id": surface.surface_id,
                "symbol": surface.symbol,
                "symbol_present": bool(symbol_present),
                "callable": bool(callable_present),
                "signature_ok": bool(signature_ok),
                "required_parameters": list(surface.required_parameters),
                "missing_parameters": missing_parameters,
                "signature_error": signature_error,
                "rationale": surface.rationale,
            }
        )

    surfaces_ready = module_loaded and not blockers
    evidence = dict(extra_evidence or {})
    file_artifact = (
        evidence.get("file_backed_export_artifact")
        if isinstance(evidence.get("file_backed_export_artifact"), Mapping)
        else evidence.get("native_mlx_export_artifact")
    )
    file_backed_evidence = build_snerv_mlx_native_file_backed_evidence(
        file_artifact if isinstance(file_artifact, Mapping) else None,
        required_num_pairs=int(
            evidence.get("required_num_pairs")
            or DEFAULT_REQUIRED_NATIVE_EXPORT_PAIRS
        ),
        base_dir=evidence.get("base_dir"),
    )
    smoke_or_file_proven = bool(
        evidence.get("two_pair_smoke_passed")
        or file_backed_evidence.get("file_backed_export_proof_passed")
    )
    if surfaces_ready and not smoke_or_file_proven:
        blockers.append("snerv_mlx_native_adapter_surfaces_present_but_unproven")

    return {
        "schema": SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA,
        "module_name": str(module_name),
        "module_loaded": bool(module_loaded),
        "module_error": module_error,
        "required_surface_count": len(REQUIRED_SURFACES),
        "ready_surface_count": sum(
            1
            for row in surface_rows
            if row["symbol_present"] and row["callable"] and row["signature_ok"]
        ),
        "surfaces_ready": bool(surfaces_ready),
        "two_pair_smoke_passed": bool(evidence.get("two_pair_smoke_passed")),
        "file_backed_export_proof_passed": bool(
            file_backed_evidence.get("file_backed_export_proof_passed")
        ),
        "required_pair_file_backed_export_proof_passed": bool(
            file_backed_evidence.get("required_pair_file_backed_export_proof_passed")
        ),
        "file_backed_export_evidence": file_backed_evidence,
        "full600_campaign_ready": bool(
            surfaces_ready
            and (
                evidence.get("two_pair_smoke_passed")
                or file_backed_evidence.get(
                    "required_pair_file_backed_export_proof_passed"
                )
            )
        ),
        "surface_rows": surface_rows,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DEFAULT_REQUIRED_NATIVE_EXPORT_PAIRS",
    "DEFAULT_SNERV_MLX_NATIVE_ADAPTER_MODULE",
    "REQUIRED_SURFACES",
    "SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA",
    "SNERV_MLX_NATIVE_TRAINING_EXPORT_GUARD_SCHEMA",
    "RequiredSurface",
    "build_snerv_mlx_native_adapter_contract",
    "build_snerv_mlx_native_file_backed_evidence",
    "build_snerv_mlx_native_training_export_guard",
]
