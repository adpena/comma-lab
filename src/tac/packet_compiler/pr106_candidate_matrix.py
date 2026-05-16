# SPDX-License-Identifier: MIT
"""PR106 PacketIR candidate evidence matrix.

This module joins the PR106/R2 PacketIR evidence that otherwise lives across
identity tests, runtime-consumption manifests, and exact auth-eval directories.
It is an audit surface, not a score-claim surface: every emitted matrix and row
remains non-promotional and axis-labelled.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tac.exact_eval_custody import (
    SCORE_FORMULA_TOLERANCE,
    contest_score,
    extract_runtime_tree_sha256,
    validate_exact_eval_evidence,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    prove_pr106_sidecar_packet_ir_identity,
)
from tac.repo_io import read_json, repo_relative, sha256_file, write_json

PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA = "pr106_packetir_candidate_matrix_v1"
PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_JSON = (
    ".omx/research/pr106_packetir_candidate_matrix_20260516_codex.json"
)
PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_MD = (
    ".omx/research/pr106_packetir_candidate_matrix_20260516_codex.md"
)

ContestAxis = Literal["contest_cpu", "contest_cuda"]
_REQUIRED_EXACT_AXES = ("contest_cpu", "contest_cuda")
_PAIRED_MODAL_DISPATCH_TOOL = "tools/dispatch_modal_paired_auth_eval.py"
_PAIRED_MODAL_PROVIDER = "modal_paired_cpu_cuda"
_LEGACY_ROUNDED_COMPONENT_SCORE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PR106PacketIRCandidateSpec:
    """One archive candidate and its known evidence artifact paths."""

    candidate_id: str
    expected_format_id: str
    archive_path: str
    runtime_consumption_path: str = ""
    exact_eval_paths: Mapping[ContestAxis, str] = field(default_factory=dict)
    notes: str = ""


DEFAULT_PR106_PACKETIR_CANDIDATES: tuple[PR106PacketIRCandidateSpec, ...] = (
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x01_r2_release",
        expected_format_id="0x01",
        archive_path="submissions/pr106_latent_sidecar_r2/archive.zip",
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/"
            "original_r2/runtime_consumption.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json"
            ),
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/"
                "pr106_latent_sidecar_r2_20260511T171453Z/contest_auth_eval.json"
            ),
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x02_pr101_grammar",
        expected_format_id="0x02",
        archive_path="submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip",
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/"
            "pr101_grammar/runtime_consumption.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_latent_sidecar_r2_pr101_grammar_20260511T180000Z/"
                "contest_auth_eval.json"
            ),
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/"
                "pr106_latent_sidecar_r2_pr101_grammar_20260511T200000Z/"
                "contest_auth_eval.json"
            ),
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x04_rank_elided",
        expected_format_id="0x04",
        archive_path=(
            "experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/"
            "pr106_sidecar_rank_elided_format04_candidate.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/"
            "runtime_consumption_format04.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/"
                "contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x05_fixed_meta",
        expected_format_id="0x05",
        archive_path=(
            "experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/"
            "runtime_consumption.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_fixed_meta_rank_elided_exact_cuda_20260514T2359Z/"
                "contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x05_hdm8_fixed_meta",
        expected_format_id="0x05",
        archive_path=(
            "experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/"
            "archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/"
            "runtime_consumption.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm8_fixed_meta_rank_elided_exact_cuda_20260515T002100Z/"
                "contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x06_implicit_len",
        expected_format_id="0x06",
        archive_path=(
            "experiments/results/"
            "pr106_hdm8_implicit_len_fixed_meta_20260515_codex/"
            "emitted_candidates/"
            "pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_hdm8_implicit_len_fixed_meta_20260515_codex/"
            "runtime_consumption_0x06.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm8_fmt06_t4_20260515T000000Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x07_headerless",
        expected_format_id="0x07",
        archive_path=(
            "experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/"
            "emitted_candidates/"
            "pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/"
            "runtime_consumption_0x07.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm8_fmt07_t4_20260515T025608Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x08_inner_headerless",
        expected_format_id="0x08",
        archive_path=(
            "experiments/results/"
            "pr106_hdm8_inner_headerless_fmt08_20260515_codex/"
            "emitted_candidates/"
            "pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x08.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_hdm8_inner_headerless_fmt08_20260515_codex/"
            "runtime_consumption_0x08.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm8_fmt08_t4_20260515T033731Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x09_hdm9",
        expected_format_id="0x09",
        archive_path=(
            "experiments/results/pr106_hdm9_packetir_recode_20260515_codex/"
            "candidates/"
            "pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_hdm9_packetir_recode_20260515_codex/"
            "runtime_consumption_hdm9.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm9_fmt09_t4_20260515T043733Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x0a_hlm3",
        expected_format_id="0x0A",
        archive_path=(
            "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
            "candidates/"
            "pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_sidecar_format_0x0a.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
            "runtime_consumption_hdm10_hlm3.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm10_hlm3_fmt0a_t4_20260515T055017Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x0b_magicless",
        expected_format_id="0x0B",
        archive_path=(
            "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
            "candidates/"
            "pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
            "runtime_consumption_hdm11_magicless.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z/contest_auth_eval.json"
            )
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x0c_exact_radix",
        expected_format_id="0x0C",
        archive_path=(
            "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/"
            "candidates/"
            "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/"
            "runtime_consumption_format0c.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_format0c_exact_radix_paired_20260515T0918Z_cuda/"
                "contest_auth_eval.json"
            ),
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/"
                "pr106_format0c_exact_radix_paired_20260515T0918Z_cpu/"
                "contest_auth_eval.json"
            ),
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="format_0x0d_latent_score_table",
        expected_format_id="0x0D",
        archive_path=(
            "experiments/results/"
            "pr106_format0d_latent_score_table_materialized_20260515_codex/"
            "sidecar_archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_format0d_latent_score_table_materialized_20260515_codex/"
            "runtime_consumption_format0d.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cuda/"
                "contest_auth_eval.json"
            ),
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/"
                "pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cpu/"
                "contest_auth_eval.json"
            ),
        },
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="prefix_top_1_pr101grammar",
        expected_format_id="0x02",
        archive_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_1/archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_1/runtime_consumption.json"
        ),
        notes="semantic prefix candidate; runtime-consumed, exact axes pending",
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="prefix_top_4_pr101grammar",
        expected_format_id="0x02",
        archive_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_4/archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_4/runtime_consumption.json"
        ),
        notes="semantic prefix candidate; runtime-consumed, exact axes pending",
    ),
    PR106PacketIRCandidateSpec(
        candidate_id="prefix_top_16_pr101grammar",
        expected_format_id="0x02",
        archive_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_16/archive.zip"
        ),
        runtime_consumption_path=(
            "experiments/results/"
            "pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/"
            "prefix_top_16/runtime_consumption.json"
        ),
        exact_eval_paths={
            "contest_cuda": (
                "experiments/results/modal_auth_eval/"
                "pr106_component_prefix16_pr101grammar_paired_20260515T191614Z_cuda/"
                "contest_auth_eval.json"
            ),
            "contest_cpu": (
                "experiments/results/modal_auth_eval_cpu/"
                "pr106_component_prefix16_pr101grammar_paired_20260515T191614Z_cpu/"
                "contest_auth_eval.json"
            ),
        },
        notes="semantic prefix candidate with paired exact diagnostic evidence",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_relative_path_text(value: object, repo_root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return text


def _normalize_format_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("0x"):
        return "0x" + text[2:].upper().zfill(2)
    try:
        return f"0x{int(text):02X}"
    except ValueError:
        return text


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        out = float(value)
        if out == out and out not in (float("inf"), float("-inf")):
            return out
    return None


def _exact_archive_sha(payload: Mapping[str, Any]) -> str:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        value = str(provenance.get("archive_sha256") or "").strip().lower()
        if value:
            return value
    return str(payload.get("archive_sha256") or "").strip().lower()


def _exact_score(payload: Mapping[str, Any]) -> float | None:
    for key in ("canonical_score", "score_recomputed_from_components", "final_score"):
        value = _finite_number(payload.get(key))
        if value is not None:
            return value
    return None


def _is_legacy_rounded_component_score_mismatch(
    *,
    payload: Mapping[str, Any],
    validation_row: Mapping[str, Any],
) -> bool:
    """Return true for historical artifacts with rounded exported components."""

    score = _finite_number(validation_row.get("score"))
    recomputed = _finite_number(validation_row.get("expected_score_recomputed"))
    artifact_recomputed = _finite_number(payload.get("score_recomputed_from_components"))
    canonical = _finite_number(payload.get("canonical_score"))
    if score is None or recomputed is None or artifact_recomputed is None:
        return False
    if abs(score - artifact_recomputed) > SCORE_FORMULA_TOLERANCE:
        return False
    if canonical is not None and abs(score - canonical) > SCORE_FORMULA_TOLERANCE:
        return False
    return abs(score - recomputed) <= _LEGACY_ROUNDED_COMPONENT_SCORE_TOLERANCE


def _runtime_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        manifest = provenance.get("inflate_runtime_manifest")
        if isinstance(manifest, Mapping):
            return manifest
    return {}


def _runtime_content_tree_sha(payload: Mapping[str, Any]) -> str:
    value = str(_runtime_manifest(payload).get("runtime_content_tree_sha256") or "")
    return value.strip().lower()


def _provenance_text(payload: Mapping[str, Any], key: str) -> str:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        return str(provenance.get(key) or "").strip()
    return ""


def _load_adjacent_request(path: Path, axis: ContestAxis) -> Mapping[str, Any]:
    fallback: Mapping[str, Any] = {}
    for request_path in sorted(path.parent.glob("*auth_eval_local_request.json")):
        try:
            payload = read_json(request_path)
        except Exception:  # pragma: no cover - defensive malformed request sidecar
            continue
        if not isinstance(payload, Mapping):
            continue
        if not fallback:
            fallback = payload
        request_axis = str(payload.get("axis") or payload.get("score_axis") or "")
        if request_axis == axis:
            return payload
    return fallback


def _adjacent_log_path(path: Path, repo_root: Path) -> str:
    for name in ("contest_auth_eval.stdout.log", "contest_auth_eval.stderr.log"):
        candidate = path.parent / name
        if candidate.is_file():
            return repo_relative(candidate, repo_root)
    return ""


def _exact_eval_validation_row(
    *,
    axis: ContestAxis,
    path: Path,
    payload: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    request = _load_adjacent_request(path, axis)
    score = _exact_score(payload)
    return {
        "axis": axis,
        "archive_sha256": _exact_archive_sha(payload),
        "runtime_tree_sha256": extract_runtime_tree_sha256(payload),
        "score": score,
        "seg_dist": payload.get("avg_segnet_dist"),
        "pose_dist": payload.get("avg_posenet_dist"),
        "archive_bytes": payload.get("archive_size_bytes"),
        "n_samples": payload.get("n_samples"),
        "hardware": _provenance_text(payload, "device")
        if axis == "contest_cpu"
        else _provenance_text(payload, "gpu_model") or _provenance_text(payload, "device"),
        "inflate_device": _provenance_text(payload, "device"),
        "eval_device": _provenance_text(payload, "device"),
        "auth_eval_command": request.get("canonical_path", ""),
        "log_path": _adjacent_log_path(path, repo_root),
        "artifact_path": repo_relative(path, repo_root),
        "expected_score_recomputed": contest_score(
            float(payload.get("avg_segnet_dist") or 0.0),
            float(payload.get("avg_posenet_dist") or 0.0),
            int(payload.get("archive_size_bytes") or 0),
        )
        if score is not None
        else None,
    }


def _load_runtime_consumption(
    *,
    path_text: str,
    repo_root: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    if not path_text.strip():
        return {
            "path": "",
            "exists": False,
            "valid": False,
            "blockers": ["runtime_consumption_path_missing"],
        }
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        return {
            "path": repo_relative(path, repo_root),
            "exists": False,
            "valid": False,
            "blockers": ["runtime_consumption_json_missing"],
        }
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive malformed artifact path
        return {
            "path": repo_relative(path, repo_root),
            "exists": True,
            "valid": False,
            "blockers": [f"runtime_consumption_json_invalid:{type(exc).__name__}"],
        }
    blockers = [str(item) for item in payload.get("blockers", []) if str(item)]
    source_archive_sha = str(payload.get("source_archive_sha256") or "").strip().lower()
    archive_payload = payload.get("archive")
    if not source_archive_sha and isinstance(archive_payload, Mapping):
        source_archive_sha = str(archive_payload.get("sha256") or "").strip().lower()
    if not source_archive_sha:
        blockers.append("runtime_consumption_source_archive_sha_missing")
    elif source_archive_sha != expected_archive_sha256:
        blockers.append("runtime_consumption_source_archive_sha_mismatch")
    runtime_source_manifest = payload.get("runtime_source_manifest")
    runtime_source_tree_sha = ""
    runtime_content_tree_sha = ""
    if isinstance(runtime_source_manifest, Mapping):
        runtime_source_tree_sha = str(
            runtime_source_manifest.get("runtime_source_tree_sha256") or ""
        ).strip().lower()
        runtime_content_tree_sha = str(
            runtime_source_manifest.get("runtime_content_tree_sha256") or ""
        ).strip().lower()
    if not runtime_source_tree_sha or len(runtime_source_tree_sha) != 64:
        blockers.append("runtime_consumption_runtime_source_tree_sha_missing")
    for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if payload.get(key) is not False:
            blockers.append(f"runtime_consumption_{key}_not_false")
    for key in (
        "runtime_sidecar_decode_consumption_claim",
        "runtime_sidecar_apply_consumption_claim",
        "runtime_all_score_affecting_sections_consumed",
        "packet_ir_consumed_byte_accounting_passed",
    ):
        if payload.get(key) is not True:
            blockers.append(f"{key}_not_true")
    valid = not blockers
    return {
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "exists": True,
        "schema": payload.get("schema"),
        "format_id": _normalize_format_id(payload.get("format_id")),
        "sidecar_kind": payload.get("sidecar_kind"),
        "source_archive_sha256": source_archive_sha,
        "runtime_dir": _repo_relative_path_text(payload.get("runtime_dir"), repo_root),
        "runtime_inflate_py_sha256": payload.get("runtime_inflate_py_sha256"),
        "runtime_source_tree_sha256": runtime_source_tree_sha,
        "runtime_content_tree_sha256": runtime_content_tree_sha,
        "runtime_sidecar_decode_consumption_claim": payload.get(
            "runtime_sidecar_decode_consumption_claim"
        ),
        "runtime_sidecar_apply_consumption_claim": payload.get(
            "runtime_sidecar_apply_consumption_claim"
        ),
        "runtime_all_score_affecting_sections_consumed": payload.get(
            "runtime_all_score_affecting_sections_consumed"
        ),
        "packet_ir_consumed_byte_accounting_passed": payload.get(
            "packet_ir_consumed_byte_accounting_passed"
        ),
        "score_claim": payload.get("score_claim") is True,
        "promotion_eligible": payload.get("promotion_eligible") is True,
        "ready_for_exact_eval_dispatch": (
            payload.get("ready_for_exact_eval_dispatch") is True
        ),
        "valid": valid,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _load_exact_eval(
    *,
    axis: ContestAxis,
    path_text: str,
    repo_root: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        return {
            "axis": axis,
            "path": repo_relative(path, repo_root),
            "exists": False,
            "valid": False,
            "blockers": ["exact_eval_json_missing"],
        }
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive malformed artifact path
        return {
            "axis": axis,
            "path": repo_relative(path, repo_root),
            "exists": True,
            "valid": False,
            "blockers": [f"exact_eval_json_invalid:{type(exc).__name__}"],
        }
    blockers: list[str] = []
    observed_axis = str(payload.get("score_axis") or "").strip()
    archive_sha = _exact_archive_sha(payload)
    score = _exact_score(payload)
    validation_row = _exact_eval_validation_row(
        axis=axis,
        path=path,
        payload=payload,
        repo_root=repo_root,
    )
    validation = validate_exact_eval_evidence(
        validation_row,
        expected_axis=axis,
        expected_archive_sha256=expected_archive_sha256,
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_devices=True,
        artifact_base_dir=repo_root,
        annotation_prefix=f"pr106_packetir_{axis}",
    )
    tolerated_legacy_rounding = (
        "score_formula_mismatch" in validation.blockers
        and _is_legacy_rounded_component_score_mismatch(
            payload=payload,
            validation_row=validation_row,
        )
    )
    blockers.extend(
        f"exact_eval_{blocker}"
        for blocker in validation.blockers
        if not (blocker == "score_formula_mismatch" and tolerated_legacy_rounding)
    )
    runtime_content_tree_sha = _runtime_content_tree_sha(payload)
    if not runtime_content_tree_sha or len(runtime_content_tree_sha) != 64:
        blockers.append("exact_eval_runtime_content_tree_sha_invalid")
    return {
        "axis": axis,
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "exists": True,
        "valid": not blockers,
        "blockers": blockers,
        "annotations": [
            *validation.annotations,
            *(
                ["legacy_rounded_component_score_mismatch_tolerated"]
                if tolerated_legacy_rounding
                else []
            ),
        ],
        "score_axis": observed_axis,
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "canonical_score": score,
        "score_formula_recomputed": validation_row["expected_score_recomputed"],
        "score_formula_difference": (
            abs(score - validation_row["expected_score_recomputed"])
            if score is not None
            and validation_row["expected_score_recomputed"] is not None
            else None
        ),
        "score_formula_tolerance": (
            _LEGACY_ROUNDED_COMPONENT_SCORE_TOLERANCE
            if tolerated_legacy_rounding
            else SCORE_FORMULA_TOLERANCE
        ),
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": validation.runtime_tree_sha256,
        "runtime_content_tree_sha256": runtime_content_tree_sha,
        "auth_eval_command": validation_row["auth_eval_command"],
        "log_path": validation_row["log_path"],
        "artifact_path": validation_row["artifact_path"],
        "hardware": validation_row["hardware"],
        "inflate_device": validation_row["inflate_device"],
        "eval_device": validation_row["eval_device"],
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "n_samples": payload.get("n_samples"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "score_claim_in_source_artifact": payload.get("score_claim") is True,
        "promotion_eligible_in_source_artifact": (
            payload.get("promotion_eligible") is True
        ),
    }


def _runtime_with_derived_content_sha(
    runtime: Mapping[str, Any],
    exact_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive consumption content SHA only from matching valid paired axes."""

    out = dict(runtime)
    existing = str(out.get("runtime_content_tree_sha256") or "").strip().lower()
    if len(existing) == 64:
        return out
    exact_content_shas = {
        str(exact_evidence.get(axis, {}).get("runtime_content_tree_sha256") or "")
        .strip()
        .lower()
        for axis in _REQUIRED_EXACT_AXES
        if exact_evidence.get(axis, {}).get("exists") is True
        and exact_evidence.get(axis, {}).get("valid") is True
    }
    exact_content_shas = {sha for sha in exact_content_shas if len(sha) == 64}
    if len(exact_content_shas) != 1:
        return out
    out["runtime_content_tree_sha256"] = next(iter(exact_content_shas))
    out["runtime_content_tree_sha256_source"] = "derived_from_matching_paired_exact_eval"
    out["runtime_content_tree_sha256_derivation_axes"] = list(_REQUIRED_EXACT_AXES)
    return out


def _candidate_status(
    *,
    archive_exists: bool,
    identity_passed: bool,
    identity_blockers: Iterable[str],
    runtime: Mapping[str, Any],
    exact_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    if not archive_exists:
        return "archive_missing"
    if not identity_passed:
        return "packet_ir_identity_blocked"
    if tuple(identity_blockers):
        return "packet_ir_identity_blocked"
    if runtime.get("exists") is not True:
        return "runtime_consumption_missing"
    if runtime.get("valid") is not True:
        return "runtime_consumption_blocked"
    valid_axes = {
        axis for axis, row in exact_evidence.items()
        if row.get("exists") is True and row.get("valid") is True
    }
    if valid_axes == {"contest_cpu", "contest_cuda"}:
        return "paired_exact_measured"
    if valid_axes:
        return "single_axis_exact_measured_needs_pair"
    return "runtime_consumed_needs_paired_exact_eval"


def _paired_exact_status_blockers(
    exact_evidence: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    valid_axes = {
        axis
        for axis, row in exact_evidence.items()
        if row.get("exists") is True and row.get("valid") is True
    }
    missing_axes = [axis for axis in _REQUIRED_EXACT_AXES if axis not in valid_axes]
    if missing_axes:
        blockers.append("paired_exact_eval_missing:" + ",".join(missing_axes))
    if valid_axes == set(_REQUIRED_EXACT_AXES):
        runtime_content_shas = {
            str(exact_evidence[axis].get("runtime_content_tree_sha256") or "")
            for axis in _REQUIRED_EXACT_AXES
        }
        if len(runtime_content_shas) != 1 or len(next(iter(runtime_content_shas))) != 64:
            blockers.append("paired_exact_eval_runtime_content_tree_sha_mismatch")
        runtime_consumption_content_sha = str(
            runtime.get("runtime_content_tree_sha256") or ""
        )
        if len(runtime_consumption_content_sha) != 64:
            blockers.append(
                "paired_exact_eval_runtime_consumption_content_tree_sha_missing"
            )
        elif runtime_content_shas != {runtime_consumption_content_sha}:
            blockers.append(
                "paired_exact_eval_runtime_content_tree_sha_mismatch_with_consumption"
            )
    return blockers


def _valid_exact_axes(
    exact_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[ContestAxis, ...]:
    return tuple(
        axis
        for axis in _REQUIRED_EXACT_AXES
        if exact_evidence.get(axis, {}).get("exists") is True
        and exact_evidence.get(axis, {}).get("valid") is True
    )


def _missing_exact_axes(
    exact_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[ContestAxis, ...]:
    valid_axes = set(_valid_exact_axes(exact_evidence))
    return tuple(axis for axis in _REQUIRED_EXACT_AXES if axis not in valid_axes)


def _axis_blockers(
    *,
    axis: ContestAxis,
    exact_evidence: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    evidence = exact_evidence.get(axis)
    if not isinstance(evidence, Mapping):
        return ["exact_eval_artifact_not_listed"]
    blockers = [str(item) for item in evidence.get("blockers", []) if str(item)]
    if not blockers and evidence.get("exists") is not True:
        blockers.append("exact_eval_json_missing")
    if not blockers and evidence.get("valid") is not True:
        blockers.append("exact_eval_invalid_without_blocker")
    return blockers


def _stable_pair_group_id(row: Mapping[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "candidate")
    archive_sha = str(row.get("archive_sha256") or "").strip().lower()
    suffix = archive_sha[:12] if archive_sha else "archive_sha_missing"
    return f"pair_pr106_packetir_{candidate_id}_{suffix}"


def _stable_lane_id_base(row: Mapping[str, Any]) -> str:
    return f"pr106_packetir_{row.get('candidate_id')}"


def _paired_modal_command_template(
    *,
    archive_path: str,
    runtime_dir: str,
    pair_group_id: str,
    lane_id_base: str,
    execute: bool,
) -> str:
    parts = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/python",
        _PAIRED_MODAL_DISPATCH_TOOL,
        "--archive",
        archive_path,
        "--label",
        lane_id_base,
        "--run-id",
        f"{lane_id_base}_<UTC>",
        "--pair-group-id",
        pair_group_id,
        "--lane-id-base",
        lane_id_base,
        "--output-root",
        "experiments/results",
        "--expected-runtime-tree-sha256",
        "auto",
        "--skip-axis-if-promotable-anchor-exists",
    ]
    if runtime_dir:
        parts.extend(["--submission-dir", runtime_dir, "--inflate-sh", "inflate.sh"])
    if execute:
        parts.append("--execute")
    return " ".join(parts)


def _next_exact_eval_targets(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in rows:
        if row.get("archive_exists") is not True:
            continue
        identity = row.get("packet_ir_identity")
        runtime = row.get("runtime_consumption")
        exact_evidence = row.get("exact_axis_evidence")
        if not isinstance(identity, Mapping) or identity.get("passed") is not True:
            continue
        if not isinstance(runtime, Mapping) or runtime.get("valid") is not True:
            continue
        if not isinstance(exact_evidence, Mapping):
            continue
        missing_axes = _missing_exact_axes(exact_evidence)
        if not missing_axes:
            continue
        runtime_dir = str(runtime.get("runtime_dir") or "")
        archive_path = str(row.get("archive_path") or "")
        pair_group_id = _stable_pair_group_id(row)
        existing_valid_axes = list(_valid_exact_axes(exact_evidence))
        lane_id_base = _stable_lane_id_base(row)
        axis_blockers = {
            axis: _axis_blockers(axis=axis, exact_evidence=exact_evidence)
            for axis in missing_axes
        }
        targets.append(
            {
                "candidate_id": row.get("candidate_id"),
                "format_id": row.get("format_id") or row.get("expected_format_id"),
                "missing_axes": list(missing_axes),
                "missing_axis": ",".join(missing_axes),
                "recommended_provider": _PAIRED_MODAL_PROVIDER,
                "modal_entrypoint": _PAIRED_MODAL_DISPATCH_TOOL,
                "paired_dispatch_tool": _PAIRED_MODAL_DISPATCH_TOOL,
                "paired_dispatch_required": True,
                "archive_path": archive_path,
                "archive_sha256": row.get("archive_sha256"),
                "archive_bytes": identity.get("archive_bytes"),
                "runtime_dir": runtime_dir,
                "inflate_sh": (
                    "inflate.sh" if runtime_dir else "submissions/robust_current/inflate.sh"
                ),
                "runtime_source_tree_sha256": runtime.get(
                    "runtime_source_tree_sha256"
                ),
                "runtime_content_tree_sha256": runtime.get(
                    "runtime_content_tree_sha256"
                ),
                "runtime_inflate_py_sha256": runtime.get(
                    "runtime_inflate_py_sha256"
                ),
                "existing_valid_axes": existing_valid_axes,
                "axis_blockers_by_axis": axis_blockers,
                "axis_blockers": [
                    f"{axis}:{blocker}"
                    for axis, blockers in axis_blockers.items()
                    for blocker in blockers
                ],
                "pair_group_id": pair_group_id,
                "lane_id": lane_id_base,
                "lane_id_base": lane_id_base,
                "instance_job_id_template": f"{lane_id_base}_<UTC>",
                "output_dir_template": (
                    f"experiments/results/modal_auth_eval/{lane_id_base}_<UTC>_cuda; "
                    f"experiments/results/modal_auth_eval_cpu/{lane_id_base}_<UTC>_cpu"
                ),
                "provider_detach_required": True,
                "execute_flag_required_for_provider_launch": True,
                "dispatch_status": (
                    "requires_claim_lane_dispatch_before_provider_launch"
                ),
                "expected_runtime_tree_sha256_policy": (
                    "paired_dispatcher_auto_computes_axis_specific_modal_uploaded_runtime_tree_sha256"
                ),
                "skip_axis_if_promotable_anchor_exists": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "command_template": _paired_modal_command_template(
                    archive_path=archive_path,
                    runtime_dir=runtime_dir,
                    pair_group_id=pair_group_id,
                    lane_id_base=lane_id_base,
                    execute=False,
                ),
                "execute_command_template_after_plan_review": _paired_modal_command_template(
                    archive_path=archive_path,
                    runtime_dir=runtime_dir,
                    pair_group_id=pair_group_id,
                    lane_id_base=lane_id_base,
                    execute=True,
                ),
            }
        )
    return targets


def _row_for_candidate(
    spec: PR106PacketIRCandidateSpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    archive_path = _resolve(spec.archive_path, repo_root)
    row: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "expected_format_id": spec.expected_format_id,
        "archive_path": repo_relative(archive_path, repo_root),
        "archive_exists": archive_path.is_file(),
        "notes": spec.notes,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    identity_blockers: list[str] = []
    packet_format_id = ""
    archive_sha = ""
    if archive_path.is_file():
        try:
            proof = prove_pr106_sidecar_packet_ir_identity(archive_path=archive_path)
        except Exception as exc:  # pragma: no cover - defensive corrupt artifact path
            identity = {
                "passed": False,
                "blockers": [f"packet_ir_identity_exception:{type(exc).__name__}"],
            }
            identity_blockers = list(identity["blockers"])
        else:
            packet = proof.get("packet", {})
            archive = proof.get("archive", {})
            member = proof.get("member", {})
            consumed = packet.get("packet_ir_consumed_byte_proof", {})
            packet_format_id = _normalize_format_id(packet.get("format_id"))
            archive_sha = str(archive.get("sha256") or "").strip().lower()
            identity_blockers = [str(item) for item in proof.get("blockers", [])]
            if packet_format_id != spec.expected_format_id:
                identity_blockers.append("packet_ir_format_id_mismatch")
            identity = {
                "passed": (
                    proof.get("packet_ir_identity_passed") is True
                    and not identity_blockers
                ),
                "blockers": list(dict.fromkeys(identity_blockers)),
                "schema": proof.get("schema"),
                "evidence_axis": proof.get("evidence_axis"),
                "archive_sha256": archive_sha,
                "archive_bytes": archive.get("bytes"),
                "member_name": member.get("name"),
                "member_payload_bytes": member.get("payload_bytes"),
                "member_payload_sha256": member.get("payload_sha256"),
                "format_id": packet_format_id,
                "sidecar_kind": packet.get("sidecar_kind"),
                "payload_byte_identical": (
                    proof.get("emitted_payload", {}).get(
                        "byte_identical_to_source_member"
                    )
                    is True
                ),
                "archive_byte_identical": (
                    proof.get("emitted_archive", {}).get(
                        "byte_identical_to_source_archive"
                    )
                    is True
                ),
                "all_payload_bytes_accounted": (
                    consumed.get("all_payload_bytes_accounted") is True
                )
                if isinstance(consumed, Mapping)
                else False,
                "score_claim": proof.get("score_claim") is True,
                "promotion_eligible": proof.get("promotion_eligible") is True,
                "ready_for_exact_eval_dispatch": (
                    proof.get("ready_for_exact_eval_dispatch") is True
                ),
            }
    else:
        identity = {
            "passed": False,
            "blockers": ["archive_missing"],
        }

    runtime = _load_runtime_consumption(
        path_text=spec.runtime_consumption_path,
        repo_root=repo_root,
        expected_archive_sha256=archive_sha,
    )
    exact_evidence = {
        axis: _load_exact_eval(
            axis=axis,
            path_text=path,
            repo_root=repo_root,
            expected_archive_sha256=archive_sha,
        )
        for axis, path in sorted(spec.exact_eval_paths.items())
    }
    runtime = _runtime_with_derived_content_sha(runtime, exact_evidence)
    status = _candidate_status(
        archive_exists=archive_path.is_file(),
        identity_passed=identity.get("passed") is True,
        identity_blockers=identity_blockers,
        runtime=runtime,
        exact_evidence=exact_evidence,
    )
    status_blockers = _paired_exact_status_blockers(exact_evidence, runtime)
    if status == "paired_exact_measured" and status_blockers:
        status = "paired_exact_blocked"
    row.update(
        {
            "packet_ir_identity": identity,
            "runtime_consumption": runtime,
            "exact_axis_evidence": exact_evidence,
            "status": status,
            "status_blockers": status_blockers,
            "format_id": packet_format_id,
            "archive_sha256": archive_sha,
        }
    )
    return row


def build_pr106_packetir_candidate_matrix(
    *,
    repo_root: str | Path | None = None,
    candidates: Iterable[PR106PacketIRCandidateSpec] = DEFAULT_PR106_PACKETIR_CANDIDATES,
) -> dict[str, Any]:
    """Build the PR106 PacketIR matrix without claiming score or dispatch."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    rows = tuple(_row_for_candidate(spec, repo_root=root) for spec in candidates)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    next_exact_eval_targets = _next_exact_eval_targets(rows)
    return {
        "schema": PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA,
        "proof_scope": (
            "packetir_identity_runtime_consumption_and_axis_label_join_no_score_claim"
        ),
        "candidate_count": len(rows),
        "status_counts": status_counts,
        "next_exact_eval_target_count": len(next_exact_eval_targets),
        "next_exact_eval_targets": next_exact_eval_targets,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_semantics": {
            "contest_cpu": "kept separate from contest_cuda; no conversion",
            "contest_cuda": "kept separate from contest_cpu; no conversion",
        },
        "rows": list(rows),
    }


def render_pr106_packetir_candidate_matrix_markdown(matrix: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown summary."""

    lines = [
        "# PR106 PacketIR candidate matrix",
        "",
        f"Schema: `{matrix.get('schema')}`",
        "",
        "This is an audit artifact only: `score_claim=false`, "
        "`promotion_eligible=false`, and CPU/CUDA axes are not converted.",
        "",
        "| candidate | format | status | archive bytes | exact axes | blockers |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in matrix.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        identity = row.get("packet_ir_identity")
        runtime = row.get("runtime_consumption")
        exact = row.get("exact_axis_evidence")
        archive_bytes = ""
        blockers: list[str] = []
        if isinstance(identity, Mapping):
            archive_bytes = str(identity.get("archive_bytes") or "")
            blockers.extend(str(item) for item in identity.get("blockers", []))
        if isinstance(runtime, Mapping):
            blockers.extend(str(item) for item in runtime.get("blockers", []))
        blockers.extend(str(item) for item in row.get("status_blockers", []))
        axes: list[str] = []
        if isinstance(exact, Mapping):
            for axis, evidence in exact.items():
                if isinstance(evidence, Mapping) and evidence.get("exists") is True:
                    axes.append(
                        f"{axis}:{'valid' if evidence.get('valid') is True else 'blocked'}"
                    )
                    blockers.extend(
                        f"{axis}:{item}" for item in evidence.get("blockers", [])
                    )
        blocker_text = ", ".join(dict.fromkeys(blockers)) or "-"
        lines.append(
            "| {candidate} | `{fmt}` | `{status}` | {bytes_} | {axes} | {blockers} |".format(
                candidate=row.get("candidate_id"),
                fmt=row.get("format_id") or row.get("expected_format_id"),
                status=row.get("status"),
                bytes_=archive_bytes,
                axes=", ".join(axes) or "-",
                blockers=blocker_text,
            )
        )
    targets = matrix.get("next_exact_eval_targets", [])
    if isinstance(targets, list) and targets:
        lines.extend(
            [
                "",
                "## Next exact eval targets",
                "",
                "These are fail-fast dispatch targets only. They still require a "
                "`tools/claim_lane_dispatch.py` claim and Modal recovery before any "
                "score or promotion claim.",
                "",
                "| candidate | missing axis | provider | lane | dispatch status | archive | axis blockers |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for target in targets:
            if not isinstance(target, Mapping):
                continue
            blockers = ", ".join(
                str(item) for item in target.get("axis_blockers", [])
            )
            lines.append(
                "| {candidate} | `{axis}` | `{provider}` | `{lane}` | `{dispatch}` | `{archive}` | {blockers} |".format(
                    candidate=target.get("candidate_id"),
                    axis=target.get("missing_axis"),
                    provider=target.get("recommended_provider"),
                    lane=target.get("lane_id"),
                    dispatch=target.get("dispatch_status"),
                    archive=target.get("archive_path"),
                    blockers=blockers or "-",
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_pr106_packetir_candidate_matrix(
    *,
    output_json: str | Path = PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_JSON,
    output_md: str | Path = PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_MD,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write JSON plus Markdown matrix artifacts."""

    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    matrix = build_pr106_packetir_candidate_matrix(repo_root=root)
    json_path = _resolve(output_json, root)
    md_path = _resolve(output_md, root)
    write_json(json_path, matrix)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_pr106_packetir_candidate_matrix_markdown(matrix),
        encoding="utf-8",
    )
    return {
        **matrix,
        "artifact_paths": {
            "json": repo_relative(json_path, root),
            "md": repo_relative(md_path, root),
        },
        "artifact_sha256": {
            "json": sha256_file(json_path),
            "md": sha256_file(md_path),
        },
    }


__all__ = [
    "DEFAULT_PR106_PACKETIR_CANDIDATES",
    "PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_JSON",
    "PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_MD",
    "PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA",
    "PR106PacketIRCandidateSpec",
    "build_pr106_packetir_candidate_matrix",
    "render_pr106_packetir_candidate_matrix_markdown",
    "write_pr106_packetir_candidate_matrix",
]
