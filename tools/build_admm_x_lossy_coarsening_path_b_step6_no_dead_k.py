#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ADMM_WAIVED:B4-reviewed historical/planning naming; docstrings or delegated coordinator code clarify whether this is Lagrangian, bridge, or actual iterative ADMM.
# ROUNDTRIP_SELF_TEST: _local_smoke_roundtrip imports the staged no-K inflate.py,
# parses the emitted archive, and checks q_i8 * fp16 scale reconstruction.
"""Build the Path B step 6 ADMM × lossy_coarsening byte-closed candidate
archive — variant WITHOUT the dead K side-info (REVIEW-ENG C2 closure).

Background (REVIEW-ENG C2, 2026-05-08)
--------------------------------------
The original Path B step 6 wire format
(``tools/build_admm_x_lossy_coarsening_path_b_step6.py`` +
``experiments/results/admm_x_lossy_coarsening_path_b_step6_<ts>/submission_dir/inflate.py``)
reserves 28 bytes for per-tensor K (uint8 each) but the decoder at
inflate.py L117-121 reads them and then discards them — the comment in the
decoder says::

    # The stream stores the K-coarsened q_i8 value directly. K remains
    # charged side-info for audit/reproducibility; multiplying here would
    # apply the coarsening twice.
    reconstructed_q = chunk

The K coefficients are an audit-only annotation; the rounded int8 values are
already coarsened at encode time (``rounded.clip(-127, 127)``), so the
decoder never multiplies by K. REVIEW-ENG C2 flags this as a 28-byte free
win: drop the K section from the wire format.

This tool is a fork of ``build_admm_x_lossy_coarsening_path_b_step6.py`` that:

1. Emits a new wire format with **no K section**::

       uint32 LE: total_section_bytes (D, including this prefix)
       byte * 56: per_tensor_fp16_scale (LE half)
       byte * (D - 4 - 56): brotli(concat(rounded_int8s))

2. Emits a forked ``inflate.py`` that DOES NOT read K bytes.
3. Records the per-tensor Ks in ``build_manifest.json`` (preserved as audit
   metadata for reproducibility — same Ks vector as the source manifest;
   does NOT charge bytes against the archive).
4. Outputs to ``experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_<ts>/``
   so the original variant remains as a forensic audit trail.

The Ks themselves are unchanged — the only difference is whether they are
serialized into the archive or only kept in the build manifest.

Predicted savings
-----------------
- Original variant: 153,699 B archive (built 2026-05-08, commit 82bfc648).
- This variant: 153,699 - 28 = 153,671 B (predicted).
- Smoke roundtrip: identical reconstructed weights as the original (both
  decoders read the SAME ``rounded_chunks``; only the audit-only K bytes
  differ between archives).

Out-of-scope
------------
- No dispatch (Lightning bootstrap is owned by Subagent BSF in parallel).
- No score change vs original variant — same weights, same fp16 scales.
- C3 dispatch_blocker (apogee_int6 contest-CUDA anchor required first)
  applies to BOTH variants; this tool does not retire it.

CLAUDE.md compliance
--------------------
- ``family_falsified=False``,
  ``falsification_scope="lagrangian_x_continuous_K_no_dead_k_only"``.
- ``ready_for_exact_eval_dispatch=False`` (CPU build never promotes itself).
- ``cuda_eval_worth_testing=True`` (this is a free byte win on the same
  candidate the user approved for CUDA score validation).
- Pure-CPU; never loads a scorer; tags evidence ``[CPU-build]``.
- weights_only=True per REVIEW-ENG C4.

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py

    # Consume a beta-Fisher/Jacobian score-aware planning manifest.
    .venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py \
        --selected-Ks-json reports/raw/beta_fisher_lossy_coarsening_weights/<run>/manifest.json \
        --selected-Ks-rms-target 0.0386

    # Consume the generic Jacobian/Fisher allocator manifest format.
    # This reads allocation.selected_by_tensor[].K after matching
    # allocation.target_distortion to --selected-Ks-rms-target.
    .venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py \
        --selected-Ks-json reports/raw/jacobian_fisher_importance_allocator/<run>/manifest.json \
        --selected-Ks-rms-target 0.0386
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _quantize_tensor,
)

# Reuse staging helpers from the canonical Lightning build script.
_LIGHTNING_BUILDER_PATH = REPO_ROOT / "experiments" / "lossy_coarsening_lightning_cuda_test.py"
_spec = importlib.util.spec_from_file_location("_lossy_coarsening_lightning_cuda_test", _LIGHTNING_BUILDER_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - sanity
    raise SystemExit(f"FATAL: could not load builder from {_LIGHTNING_BUILDER_PATH}")
_lightning_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lightning_builder)

_read_pr101_inner_blob = _lightning_builder._read_pr101_inner_blob
_split_pr101_inner_blob = _lightning_builder._split_pr101_inner_blob
_build_inner_blob = _lightning_builder._build_inner_blob
_write_pr101_archive = _lightning_builder._write_pr101_archive

LANE_ID = "admm_x_lossy_coarsening_path_b_step6_no_dead_k"
SCHEMA_VERSION = "admm_x_lossy_coarsening_path_b_step6_no_dead_k_build.v1"
TOOL_NAME = "tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py"

# Source-of-truth Ks (same vector as the original variant — only the wire
# format changes).
ADMM_PATH_B_STEP6_KS: tuple[int, ...] = (
    2,
    1,
    5,
    1,
    5,
    1,
    5,
    1,
    2,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)
ADMM_PROXY_ARCHIVE_BYTES = 153_639  # original-variant proxy
ADMM_PROXY_REL_ERR = 0.0415393353487541
ADMM_PROXY_LAMBDA = 1_276_154.6884887693
ADMM_RMS_TARGET = 0.0386
ADMM_SOURCE_MANIFEST = "reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_20260508T041303Z/manifest.json"
ORIGINAL_VARIANT_TOOL = "tools/build_admm_x_lossy_coarsening_path_b_step6.py"

DEFAULT_PR101_STATE_DICT = (
    REPO_ROOT / "experiments/results/pr101_codecop_sweep_20260507_codex" / "pr101_decoder_state_dict.pt"
)
DEFAULT_PR101_FRONTIER_ARCHIVE = (
    REPO_ROOT / "experiments/results/public_pr_intake_full" / "public_pr101_intake_20260505_auto" / "archive.zip"
)
DEFAULT_PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "source/submissions/hnerv_ft_microcodec/src"
)
DEFAULT_PREDICTED_BAND = (0.18, 0.22)
DEFAULT_SELECTED_KS_MAX_FP32_SMOKE_REL_ERR = 0.055

CPU_BUILD_SCORE_BLOCKERS = [
    "cpu_build_rel_err_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
    # REVIEW-ENG C3 also applies to this variant (same Ks, same rel_err).
    "apogee_int6_contest_cuda_anchor_required_first",
]

SELECTED_KS_SOURCE_BLOCKERS_CLOSED_BY_BYTE_CLOSED_BUILD = frozenset(
    {
        "selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet",
        "weight_export_only_no_byte_closed_archive",
    }
)


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cpu_build_proxy_guard_fields() -> dict[str, object]:
    """Fail-closed custody fields for the local CPU build artifact.

    Mirrors ``cpu_build_proxy_guard_fields`` in the original variant + adds
    REVIEW-ENG C3 dispatch blocker (apogee_int6 contest-CUDA precondition).
    """
    return {
        "evidence_semantics": "cpu_build_byte_closed_candidate_proxy_no_score_no_dead_k",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "cuda_eval_worth_testing": True,
        "family_falsified": False,
        "falsification_scope": "lagrangian_x_continuous_K_no_dead_k_only",
        "custody_status": "transient-allowed",
        "custody_status_reason": (
            "CPU-build archives are ignored local custody artifacts; durable "
            "signal must be summarized in .omx/research and exact-score "
            "promotion requires contest auth eval on a rebuilt packet."
        ),
        "score_claim_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "dispatch_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
    }


def _guard_fields_with_selected_Ks_source_blockers(
    k_source_metadata: dict[str, object],
    *,
    selected_Ks_fp32_smoke_guard: dict[str, object] | None = None,
) -> dict[str, object]:
    guard = cpu_build_proxy_guard_fields()
    extra: list[str] = []
    if k_source_metadata.get("per_tensor_K_source") == "selected_Ks_json":
        extra.append("selected_Ks_json_cpu_planning_not_score_authority")
        source_semantics = k_source_metadata.get("selected_Ks_source_evidence_semantics")
        if source_semantics:
            extra.append(f"selected_Ks_source_evidence_semantics:{source_semantics}")
        for blocker in k_source_metadata.get("selected_Ks_source_dispatch_blockers") or []:
            blocker_str = str(blocker)
            if blocker_str in SELECTED_KS_SOURCE_BLOCKERS_CLOSED_BY_BYTE_CLOSED_BUILD:
                continue
            extra.append(f"selected_Ks_source_blocker:{blocker_str}")
    if selected_Ks_fp32_smoke_guard is not None and selected_Ks_fp32_smoke_guard.get("verdict") == "rejected":
        guard["cuda_eval_worth_testing"] = False
        for blocker in selected_Ks_fp32_smoke_guard.get("blockers") or []:
            extra.append(str(blocker))
    for key in ("score_claim_blockers", "dispatch_blockers"):
        current = guard.get(key)
        if not isinstance(current, list):  # pragma: no cover - guard shape sanity
            raise TypeError(f"guard field {key!r} must be a list")
        guard[key] = sorted({*current, *extra})
    return guard


def _blend_selected_Ks_with_baseline_additive_cap(
    selected_Ks: list[int],
    *,
    additive_cap: int,
    baseline_Ks: tuple[int, ...] = ADMM_PATH_B_STEP6_KS,
) -> tuple[list[int], dict[str, object]]:
    """Cap externally selected Ks against the default no-dead-K baseline.

    Larger K means more aggressive coarsening.  This blend keeps score-aware
    selections when they are less aggressive than the baseline, but limits
    tensors that jump far beyond the baseline to ``baseline + additive_cap``.
    It is a CPU risk-control prototype for beta/Jacobian-Fisher reactivation:
    preserve some byte pressure while preventing one diagnostic importance
    vector from spending all rel_err budget on a few tensors.
    """

    if isinstance(additive_cap, bool) or additive_cap < 0:
        raise SystemExit("--selected-Ks-additive-baseline-cap must be >= 0")
    if len(selected_Ks) != len(baseline_Ks):
        raise SystemExit(
            f"FATAL: selected_Ks length does not match baseline_Ks length ({len(selected_Ks)} != {len(baseline_Ks)})"
        )
    blended: list[int] = []
    changed: list[dict[str, object]] = []
    for idx, (selected, baseline) in enumerate(zip(selected_Ks, baseline_Ks, strict=True)):
        cap = min(255, int(baseline) + int(additive_cap))
        value = min(int(selected), cap) if int(selected) > int(baseline) else int(selected)
        blended.append(value)
        if value != int(selected):
            changed.append(
                {
                    "tensor_index": idx,
                    "tensor_name": FIXED_STATE_SCHEMA[idx][0],
                    "baseline_K": int(baseline),
                    "selected_K_original": int(selected),
                    "selected_K_after_blend": int(value),
                    "additive_cap": int(additive_cap),
                }
            )
    return blended, {
        "selected_Ks_blend_mode": "baseline_additive_cap",
        "selected_Ks_blend_baseline_source": "ADMM_PATH_B_STEP6_KS",
        "selected_Ks_blend_additive_cap": int(additive_cap),
        "selected_Ks_original_from_json": list(selected_Ks),
        "selected_Ks_after_blend": list(blended),
        "selected_Ks_blend_changed_count": len(changed),
        "selected_Ks_blend_changes": changed,
    }


def _selected_Ks_fp32_smoke_safety_guard(
    *,
    k_source_metadata: dict[str, object],
    smoke: dict[str, object],
    archive_bytes: int,
    max_fp32_smoke_rel_err: float,
) -> dict[str, object]:
    """Classify selected-K vectors using aggregate fp32 smoke rel_err.

    This is deliberately separate from the wire-format integrity smoke.  A
    vector may decode correctly while still being too risky as a score-lowering
    candidate because aggregate fp32 rel_err is far above the baseline trust
    region.
    """

    if k_source_metadata.get("per_tensor_K_source") != "selected_Ks_json":
        return {
            "applied": False,
            "verdict": "not_applicable_builtin_Ks",
            "blockers": [],
        }
    if (
        isinstance(max_fp32_smoke_rel_err, bool)
        or not isinstance(max_fp32_smoke_rel_err, int | float)
        or not np.isfinite(float(max_fp32_smoke_rel_err))
        or float(max_fp32_smoke_rel_err) <= 0.0
    ):
        raise SystemExit("--selected-Ks-max-fp32-smoke-rel-err must be finite and > 0")
    threshold = float(max_fp32_smoke_rel_err)
    blockers: list[str] = []

    rel_err_raw = smoke.get("rel_err_vs_quantized_fp32")
    max_tensor_raw = smoke.get("max_per_tensor_rel_err")
    rel_err = float(rel_err_raw) if isinstance(rel_err_raw, int | float) and not isinstance(rel_err_raw, bool) else None
    max_tensor = (
        float(max_tensor_raw)
        if isinstance(max_tensor_raw, int | float) and not isinstance(max_tensor_raw, bool)
        else None
    )

    if rel_err is None or not np.isfinite(rel_err) or rel_err < 0.0:
        rel_err = None
        blockers.append("selected_Ks_fp32_smoke_rel_err_invalid")
    elif rel_err > threshold:
        blockers.append("selected_Ks_fp32_smoke_rel_err_above_guard")
    if max_tensor is None or not np.isfinite(max_tensor) or max_tensor < 0.0:
        max_tensor = None
        blockers.append("selected_Ks_fp32_smoke_max_tensor_rel_err_invalid")
    verdict = "passed" if not blockers else "rejected"
    return {
        "applied": True,
        "verdict": verdict,
        "aggregate_fp32_smoke_rel_err": rel_err,
        "max_per_tensor_fp32_smoke_rel_err": max_tensor,
        "max_allowed_fp32_smoke_rel_err": threshold,
        "archive_bytes": int(archive_bytes),
        "blockers": blockers,
    }


def _closed_selected_Ks_source_blockers(
    k_source_metadata: dict[str, object],
) -> list[str]:
    if k_source_metadata.get("per_tensor_K_source") != "selected_Ks_json":
        return []
    return sorted(
        str(blocker)
        for blocker in k_source_metadata.get("selected_Ks_source_dispatch_blockers") or []
        if str(blocker) in SELECTED_KS_SOURCE_BLOCKERS_CLOSED_BY_BYTE_CLOSED_BUILD
    )


def _validate_Ks(values: list[object], *, source: str) -> list[int]:
    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(values) != n_tensors:
        raise SystemExit(f"FATAL: {source} selected_Ks length {len(values)} != n_tensors {n_tensors}")
    out: list[int] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"FATAL: {source} selected_Ks[{idx}] is not an integer: {value!r}")
        if value < 1 or value > 255:
            raise SystemExit(f"FATAL: {source} selected_Ks[{idx}] out of [1,255]: {value!r}")
        out.append(int(value))
    return out


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _generic_allocation_selected_Ks(
    payload: dict[str, object],
    path: Path,
    *,
    rms_target: float,
) -> tuple[dict[str, object], str] | None:
    """Read K selections from the generic Jacobian/Fisher allocation schema.

    The generic allocator emits ``allocation.selected_by_tensor`` rather than
    the PR101-specific ``weighted_k_allocations`` rows.  Accept it only when
    the manifest was produced for the requested target-distortion row and the
    row order matches ``FIXED_STATE_SCHEMA``; byte budget allocations are not
    interchangeable with this builder's RMS target.
    """
    schema = payload.get("schema", payload.get("schema_version"))
    if schema != "jacobian_fisher_importance_allocator.v1":
        return None
    allocation = payload.get("allocation")
    if allocation is None:
        return None
    if not isinstance(allocation, dict):
        raise SystemExit(f"FATAL: {path}: allocation must be a JSON object")
    objective = allocation.get("objective")
    if objective != "target_distortion":
        raise SystemExit(
            f"FATAL: {path}: generic allocation objective must be "
            f"'target_distortion' for --selected-Ks-rms-target consumption; "
            f"got {objective!r}"
        )
    target = allocation.get("target_distortion")
    if not _is_number(target) or abs(float(target) - float(rms_target)) > 1e-12:
        raise SystemExit(
            f"FATAL: {path}: allocation.target_distortion={target!r} does not match requested rms_target={rms_target}"
        )
    selected_rows = allocation.get("selected_by_tensor")
    if not isinstance(selected_rows, list):
        raise SystemExit(f"FATAL: {path}: allocation.selected_by_tensor must be a list")
    selected: list[object] = []
    for idx, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            raise SystemExit(f"FATAL: {path}: allocation.selected_by_tensor[{idx}] must be an object")
        tensor_index = row.get("tensor_index")
        if tensor_index != idx:
            raise SystemExit(
                f"FATAL: {path}: allocation.selected_by_tensor[{idx}].tensor_index must be {idx}; got {tensor_index!r}"
            )
        expected_name = FIXED_STATE_SCHEMA[idx][0]
        tensor_name = row.get("tensor_name")
        if tensor_name != expected_name:
            raise SystemExit(
                f"FATAL: {path}: allocation.selected_by_tensor[{idx}].tensor_name "
                f"must be {expected_name!r}; got {tensor_name!r}"
            )
        if "K" not in row:
            raise SystemExit(f"FATAL: {path}: allocation.selected_by_tensor[{idx}] lacks K")
        selected.append(row["K"])
    return (
        {
            "rms_target": target,
            "selected_Ks": selected,
            "total_bytes": allocation.get("total_bytes"),
            "rel_err": allocation.get("weighted_rms_error"),
            "weighted_rms_error": allocation.get("weighted_rms_error"),
            "unweighted_rms_error": allocation.get("unweighted_rms_error"),
            "objective": objective,
        },
        "allocation.selected_by_tensor[].K",
    )


def _load_selected_Ks_from_manifest(
    path: Path,
    *,
    rms_target: float,
) -> tuple[list[int], dict[str, object]]:
    """Load ``weighted_k_allocations[].selected_Ks`` from a planning manifest."""
    if not path.is_file():
        raise SystemExit(f"FATAL: --selected-Ks-json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"FATAL: {path} must contain a JSON object")

    selected_row: dict[str, object] | None = None
    source_field = "weighted_k_allocations[].selected_Ks"
    rows = payload.get("weighted_k_allocations")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_target = row.get("rms_target")
            if isinstance(row_target, int | float) and abs(float(row_target) - rms_target) <= 1e-12:
                selected_row = row
                break
        # Codex adversarial review 2026-05-08 HIGH #1: the prior single-row
        # fallback (``len(rows) == 1`` → accept whatever target it has) silently
        # built archives at the WRONG distortion target when the manifest had a
        # stale or wrong single row. Removed: every row must match the requested
        # rms_target exactly. Operators wanting a single-row override must add
        # an explicit flag in a future change with a hard dispatch blocker.

    if selected_row is None:
        generic = _generic_allocation_selected_Ks(
            payload,
            path,
            rms_target=float(rms_target),
        )
        if generic is not None:
            selected_row, source_field = generic

    if selected_row is None and isinstance(payload.get("selected_Ks"), list):
        selected_row = {
            "rms_target": rms_target,
            "selected_Ks": payload["selected_Ks"],
        }
        source_field = "selected_Ks"

    if selected_row is None:
        raise SystemExit(
            f"FATAL: {path} has no weighted_k_allocations row or generic "
            f"allocation.selected_by_tensor K vector for rms_target={rms_target}"
        )
    selected = selected_row.get("selected_Ks")
    if not isinstance(selected, list):
        raise SystemExit(f"FATAL: selected row in {path} lacks selected_Ks list")

    source = f"{path}:{source_field}"
    Ks = _validate_Ks(selected, source=source)
    metadata = {
        "per_tensor_K_source": "selected_Ks_json",
        "selected_Ks_json": str(path),
        "selected_Ks_json_sha256": _sha256(path.read_bytes()),
        "selected_Ks_rms_target_requested": float(rms_target),
        "selected_Ks_row_rms_target": selected_row.get("rms_target"),
        "selected_Ks_row_total_bytes_proxy": selected_row.get("total_bytes"),
        "selected_Ks_row_rel_err_proxy": selected_row.get("rel_err"),
        "selected_Ks_row_weighted_rms_error": selected_row.get("weighted_rms_error"),
        "selected_Ks_row_unweighted_rms_error": selected_row.get("unweighted_rms_error"),
        "selected_Ks_source_field": source_field,
        "selected_Ks_source_allocation_objective": selected_row.get("objective"),
        "selected_Ks_source_schema": payload.get("schema", payload.get("schema_version")),
        "selected_Ks_source_evidence_semantics": payload.get("evidence_semantics"),
        "selected_Ks_source_dispatch_blockers": payload.get("dispatch_blockers", []),
    }
    return Ks, metadata


def _build_lossy_decoder_section_no_K(
    state_dict_path: Path,
    Ks: list[int],
    *,
    brotli_quality: int = 11,
) -> dict:
    """Apply per-tensor coarsening with FIXED Ks; emit wire format that does
    NOT serialize K (REVIEW-ENG C2 fix). Returns same fields as the original
    variant minus ``K_bytes`` (always 0).

    Wire format::

        uint32 LE: total_section_bytes (D, including this prefix)
        byte * 56: per_tensor_fp16_scale (LE half)
        byte * (D - 4 - 56): brotli(concat(rounded_int8s))
    """
    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")
    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(Ks) != n_tensors:
        raise SystemExit(f"FATAL: Ks length {len(Ks)} != n_tensors {n_tensors} (FIXED_STATE_SCHEMA)")
    for k in Ks:
        if not isinstance(k, int) or k < 1 or k > 255:
            raise SystemExit(f"FATAL: K out of [1,255] range: {k!r}")

    # weights_only=True per REVIEW-ENG C4 — we only need tensor data.
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    scales_fp16: list[float] = []
    rounded_chunks: list[np.ndarray] = []
    abs_orig_total = 0.0
    abs_err_total = 0.0
    n_symbols = 0
    for (name, _shape), K in zip(FIXED_STATE_SCHEMA, Ks, strict=True):
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        scale_fp16 = float(np.float16(qt.scale))
        rounded = np.round(symbols_i32 / K) * K
        rounded_clipped = rounded.clip(-127, 127)
        abs_orig_total += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err_total += float(np.abs(rounded_clipped - symbols_i32).astype(np.float64).sum())
        rounded_chunks.append(rounded_clipped.astype(np.int8))
        scales_fp16.append(scale_fp16)
        n_symbols += int(symbols_i32.size)

    flat = np.concatenate(rounded_chunks).tobytes()
    brotli_payload = brotli.compress(flat, quality=brotli_quality, lgwin=22, lgblock=24)
    rel_err = (
        abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0
    )  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for ADMM-x-lossy_coarsening Path B step6 no-dead-k variant; same form as mainline. See .omx/research/rel_err_inconsistency_audit_20260508_claude.md

    scale_arr = np.array(scales_fp16, dtype=np.float16)
    if not scale_arr.dtype.isnative or sys.byteorder != "little":
        scale_bytes = scale_arr.astype("<f2").tobytes()
    else:
        scale_bytes = scale_arr.tobytes()

    # No K section — only scales + brotli payload.
    section_no_prefix = scale_bytes + brotli_payload
    section_total = 4 + len(section_no_prefix)
    prefix = struct.pack("<I", section_total)
    decoder_bytes = prefix + section_no_prefix
    if len(decoder_bytes) != section_total:
        raise RuntimeError(f"decoder section length mismatch: declared {section_total}, actual {len(decoder_bytes)}")
    return {
        "decoder_bytes": decoder_bytes,
        "per_tensor_K": list(Ks),  # audit-only; not in wire format
        "per_tensor_scale_fp16": scales_fp16,
        "rel_err": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "brotli_payload_bytes": len(brotli_payload),
        "K_bytes_in_wire_format": 0,  # REVIEW-ENG C2: dropped
        "scale_bytes": len(scale_bytes),
        "section_total_bytes": section_total,
    }


# Hardcoded forked inflate.py source (no K read; section layout: prefix +
# scales + brotli). This mirrors the original inflate.py minus the K section.
_FORKED_INFLATE_SRC = '''#!/usr/bin/env python
"""Forked PR101 inflate for lossy_coarsening_analytical archive — variant
without K side-info (REVIEW-ENG C2 fix; same reconstruction as the original
variant; the original\'s K bytes were dead audit metadata).

Wire format (inner blob, single ZIP member \'x\'):
    +--------------------------------------------+
    | uint32 LE: decoder_section_total_bytes (D) |
    +--------------------------------------------+
    | byte * 56: per_tensor_fp16_scale (LE half) |
    +--------------------------------------------+
    | byte * (D - 4 - 56): brotli(int8s)         |
    +--------------------------------------------+
    | byte * 15387: latent_blob (PR101 ORIGINAL) |
    +--------------------------------------------+
    | byte * remaining: sidecar_blob (ORIGINAL)  |
    +--------------------------------------------+

Decoder:
1. Read uint32 prefix = D
2. Read 56 bytes scale_fp16[i]
3. brotli-decode the remaining (D - 60) bytes -> flat int8 array
4. Split flat into per-tensor int8 chunks per FIXED_STATE_SCHEMA shapes
5. Use each chunk directly as the recovered coarsened q_i8 (already rounded
   at encode time; no need for K to dequantize)
6. Apply per-tensor fp16 scale to recover float weights:
   recovered_fp32 = recovered_q_i8.astype(fp32) * scale_fp16

Latent + sidecar use the original PR101 codec functions.
"""
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (  # PR101 originals (vendored)
    LATENT_BLOB_LEN,
    N_PAIRS,
    LATENT_DIM,
    BASE_CHANNELS,
    EVAL_SIZE,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder


def _fixed_state_schema():
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    return tuple((name, tuple(t.shape)) for name, t in probe.state_dict().items())


FIXED_STATE_SCHEMA = _fixed_state_schema()
N_TENSORS = len(FIXED_STATE_SCHEMA)
SCALE_SECTION_BYTES = N_TENSORS * 2  # fp16 = 2 bytes each = 56 bytes
PREFIX_BYTES = 4  # uint32 LE


CAMERA_H, CAMERA_W = 874, 1164


def parse_lossy_archive(archive_bytes):
    if len(archive_bytes) < PREFIX_BYTES + SCALE_SECTION_BYTES + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for lossy_coarsening "
            "no-dead-K format"
        )
    section_total = struct.unpack("<I", archive_bytes[:PREFIX_BYTES])[0]
    if section_total < PREFIX_BYTES + SCALE_SECTION_BYTES:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum "
            f"{PREFIX_BYTES + SCALE_SECTION_BYTES}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    scale_start = PREFIX_BYTES
    scale_end = scale_start + SCALE_SECTION_BYTES
    scales_fp16 = np.frombuffer(archive_bytes[scale_start:scale_end], dtype="<f2")
    if scales_fp16.size != N_TENSORS:
        raise ValueError(
            f"scale section size {scales_fp16.size} != N_TENSORS {N_TENSORS}"
        )

    brotli_start = scale_end
    brotli_end = section_total
    brotli_payload = archive_bytes[brotli_start:brotli_end]
    flat_int8 = np.frombuffer(brotli.decompress(brotli_payload), dtype=np.int8)

    decoder_sd = {}
    cursor = 0
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        nelem = 1
        for d in shape:
            nelem *= d
        if cursor + nelem > flat_int8.size:
            raise ValueError(
                f"flat_int8 underflow at tensor {idx} ({name}): "
                f"need {nelem}, have {flat_int8.size - cursor}"
            )
        chunk = flat_int8[cursor:cursor + nelem].astype(np.int32)
        # Same as the original variant: the stream stores the K-coarsened
        # q_i8 directly. The original variant\'s comment noted K was kept
        # "for audit/reproducibility" but never used in decoding — the
        # no-dead-k variant simply omits those bytes from the archive.
        reconstructed_q = chunk
        # Dequantize: weight_fp32 = q_i8 * scale_fp16
        weight_fp32 = (reconstructed_q.astype(np.float32) * float(scales_fp16[idx]))
        decoder_sd[name] = torch.from_numpy(weight_fp32.reshape(shape).copy())
        cursor += nelem
    if cursor != flat_int8.size:
        raise ValueError(
            f"flat_int8 leftover {flat_int8.size - cursor} bytes after consuming all tensors"
        )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in lossy archive")

    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents, meta


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_lossy_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


def _stage_forked_submission_dir_no_k(
    submission_dir: Path,
    *,
    pr101_source_dir: Path,
) -> None:
    """Stage submission_dir/{inflate.py,inflate.sh,src/codec.py,src/model.py}
    with the no-dead-k inflate.py."""
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Copy PR101 codec.py + model.py (vendored deps for inflate).
    for fname in ("codec.py", "model.py"):
        src_file = pr101_source_dir / fname
        if not src_file.is_file():
            raise SystemExit(f"FATAL: PR101 source missing: {src_file}")
        shutil.copy2(src_file, src_dir / fname)

    # Reuse the original inflate.sh (the runtime contract is identical:
    # contest auth-eval calls inflate.sh <data_dir> <output_dir> <file_list>,
    # and the format-specific decode is in inflate.py itself).
    original_sh = (
        REPO_ROOT
        / "experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z"
        / "submission_dir"
        / "inflate.sh"
    )
    if original_sh.is_file():
        shutil.copy2(original_sh, submission_dir / "inflate.sh")
    else:
        # Minimal fallback if the original artifact is gone: a wrapper that
        # invokes inflate.py with the canonical three-arg contest contract.
        (submission_dir / "inflate.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'DATA_DIR="${1:?data dir required}"\n'
            'OUTPUT_DIR="${2:?output dir required}"\n'
            'FILE_LIST="${3:?file list required}"\n'
            'mkdir -p "$OUTPUT_DIR"\n'
            'INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"\n'
            'INFLATE_TORCH_SPEC="${INFLATE_TORCH_SPEC:-torch==2.5.1+cu124}"\n'
            'INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"\n'
            'UV_BIN="${UV_BIN:-$(command -v uv || echo /usr/local/bin/uv)}"\n'
            'if [ ! -x "$UV_BIN" ]; then\n'
            '  echo "FATAL: uv not on PATH (UV_BIN=$UV_BIN); the canonical inflate-time env requires uv." >&2\n'
            "  exit 1\n"
            "fi\n"
            "UV_WITH_INFLATE_DEPS=(\n"
            '  --with "$INFLATE_BROTLI_SPEC"\n'
            '  --with "$INFLATE_TORCH_SPEC"\n'
            '  --with "$INFLATE_NUMPY_SPEC"\n'
            ")\n"
            "while IFS= read -r line; do\n"
            '  [ -z "$line" ] && continue\n'
            '  BASE="${line%.*}"\n'
            '  SRC="${DATA_DIR}/x"\n'
            '  if [ ! -f "$SRC" ]; then\n'
            '    SRC="${DATA_DIR}/${BASE}.bin"\n'
            "  fi\n"
            '  DST="${OUTPUT_DIR}/${BASE}.raw"\n'
            '  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1\n'
            '  printf "Inflating %s ... " "$line"\n'
            '  "$UV_BIN" run --no-project "${UV_WITH_INFLATE_DEPS[@]}" python "$HERE/inflate.py" "$SRC" "$DST"\n'
            'done < "$FILE_LIST"\n'
        )
        (submission_dir / "inflate.sh").chmod(0o755)

    (submission_dir / "inflate.py").write_text(_FORKED_INFLATE_SRC)


def _remove_python_caches(root: Path) -> list[str]:
    """Remove import-time cache files from a staged runtime tree.

    The CPU smoke imports the staged ``inflate.py`` and vendored ``src`` files,
    which can create ``__pycache__`` directories. They are not runtime inputs
    and should not be available for recursive staging or release packaging.
    """
    removed: list[str] = []
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        if not cache_dir.is_dir():
            continue
        removed.append(str(cache_dir.relative_to(root)))
        shutil.rmtree(cache_dir)
    return sorted(removed)


def _local_smoke_roundtrip(archive_path: Path, *, pr101_state_dict_path: Path, submission_dir: Path) -> dict:
    """Smoke test: parse the lossy archive with the no-K inflate, verify
    every tensor decodes back to the encoder's quantized form within
    ``rel_err`` tolerance.
    """
    spec_path = submission_dir / "inflate.py"
    if not spec_path.is_file():
        raise SystemExit(f"FATAL: forked inflate.py missing for smoke roundtrip: {spec_path}")
    spec = importlib.util.spec_from_file_location("forked_inflate_admm_no_k", spec_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    forked_inflate = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir / "src"))
    try:
        spec.loader.exec_module(forked_inflate)
    finally:
        sys.path.pop(0)

    inner = _read_pr101_inner_blob(archive_path)
    decoder_sd, latents, meta = forked_inflate.parse_lossy_archive(inner)

    sd_ref = torch.load(pr101_state_dict_path, map_location="cpu", weights_only=True)
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        qt_ref = _quantize_tensor(name, sd_ref[name], n_quant=N_QUANT)
        ref_quantized = (qt_ref.q_i8.astype(np.float32) * float(np.float16(qt_ref.scale))).reshape(t_dec.shape)
        denom_q = float(np.abs(ref_quantized).sum())
        err_q = float(np.abs(t_dec - ref_quantized).sum())
        per_tensor.append({"name": name, "rel_err_vs_quantized": (err_q / denom_q) if denom_q > 1e-9 else 0.0})
        abs_orig += denom_q
        abs_err += err_q
    rel_err = (
        abs_err / abs_orig if abs_orig > 1e-9 else 0.0
    )  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for fp32 smoke probe (no-dead-k variant); consistent with mainline form

    n_pairs = int(latents.shape[0]) if hasattr(latents, "shape") else None
    return {
        "rel_err_vs_quantized_fp32": rel_err,
        "n_tensors_compared": len(per_tensor),
        "max_per_tensor_rel_err": max(t["rel_err_vs_quantized"] for t in per_tensor),
        "n_latent_pairs_decoded": n_pairs,
        "latent_dim_meta": meta.get("latent_dim"),
        "base_channels_meta": meta.get("base_channels"),
        "eval_size_meta": meta.get("eval_size"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=DEFAULT_PR101_STATE_DICT,
    )
    p.add_argument(
        "--frontier-archive",
        type=Path,
        default=DEFAULT_PR101_FRONTIER_ARCHIVE,
    )
    p.add_argument(
        "--pr101-source-dir",
        type=Path,
        default=DEFAULT_PR101_SOURCE_DIR,
    )
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "results",
    )
    p.add_argument(
        "--predicted-low",
        type=float,
        default=DEFAULT_PREDICTED_BAND[0],
    )
    p.add_argument(
        "--predicted-high",
        type=float,
        default=DEFAULT_PREDICTED_BAND[1],
    )
    p.add_argument(
        "--selected-Ks-json",
        "--score-weights-json",
        dest="selected_Ks_json",
        type=Path,
        default=None,
        help=(
            "Optional beta-Fisher/Jacobian planning manifest containing "
            "weighted_k_allocations[].selected_Ks or generic "
            "allocation.selected_by_tensor[].K. This changes charged bits after "
            "the archive is rebuilt, but remains CPU-build evidence only."
        ),
    )
    p.add_argument(
        "--selected-Ks-rms-target",
        type=float,
        default=ADMM_RMS_TARGET,
        help="RMS target row to read from --selected-Ks-json.",
    )
    p.add_argument(
        "--selected-Ks-additive-baseline-cap",
        type=int,
        default=None,
        help=(
            "Optional beta/Jacobian-Fisher reactivation risk blend: when an "
            "external selected-Ks vector is more aggressive than the default "
            "no-dead-K baseline, cap that tensor's K at baseline_K + this "
            "integer before building the archive."
        ),
    )
    p.add_argument(
        "--selected-Ks-max-fp32-smoke-rel-err",
        type=float,
        default=DEFAULT_SELECTED_KS_MAX_FP32_SMOKE_REL_ERR,
        help=(
            "Selected-Ks CPU safety guard. External selected-Ks vectors with "
            "aggregate fp32 smoke rel_err above this value are marked "
            "cuda_eval_worth_testing=False and receive a dispatch blocker. "
            "The build still writes custody artifacts for review."
        ),
    )
    args = p.parse_args(argv)
    if args.selected_Ks_additive_baseline_cap is not None and args.selected_Ks_json is None:
        sys.exit("FATAL: --selected-Ks-additive-baseline-cap requires --selected-Ks-json")

    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.frontier_archive.is_file():
        sys.exit(f"FATAL: --frontier-archive not found: {args.frontier_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    build_dir = args.output_root / f"admm_x_lossy_coarsening_path_b_step6_no_dead_k_{timestamp}"
    build_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_dir / "archive.zip"
    submission_dir = build_dir / "submission_dir"
    build_manifest_path = build_dir / "build_manifest.json"

    if args.selected_Ks_json is not None:
        Ks, k_source_metadata = _load_selected_Ks_from_manifest(
            args.selected_Ks_json,
            rms_target=float(args.selected_Ks_rms_target),
        )
        if args.selected_Ks_additive_baseline_cap is not None:
            Ks, blend_metadata = _blend_selected_Ks_with_baseline_additive_cap(
                Ks,
                additive_cap=int(args.selected_Ks_additive_baseline_cap),
            )
            k_source_metadata.update(blend_metadata)
    else:
        Ks = list(ADMM_PATH_B_STEP6_KS)
        k_source_metadata = {
            "per_tensor_K_source": "ADMM_PATH_B_STEP6_KS",
            "selected_Ks_json": None,
            "selected_Ks_json_sha256": None,
            "selected_Ks_rms_target_requested": None,
            "selected_Ks_row_rms_target": ADMM_RMS_TARGET,
            "selected_Ks_row_total_bytes_proxy": ADMM_PROXY_ARCHIVE_BYTES,
            "selected_Ks_row_rel_err_proxy": ADMM_PROXY_REL_ERR,
            "selected_Ks_source_schema": None,
            "selected_Ks_source_evidence_semantics": None,
            "selected_Ks_source_dispatch_blockers": [],
        }
    print(
        "[admm-build-no-dead-k] applying Ks "
        f"(source={k_source_metadata['per_tensor_K_source']}, "
        f"rms_target={k_source_metadata['selected_Ks_row_rms_target']}, "
        f"lambda={ADMM_PROXY_LAMBDA:.0f})"
    )
    print(f"[admm-build-no-dead-k]   Ks (audit-only) = {Ks}")
    section = _build_lossy_decoder_section_no_K(args.state_dict, Ks, brotli_quality=args.brotli_quality)
    print(
        f"[admm-build-no-dead-k]   decoder section bytes={section['section_total_bytes']:,} "
        f"(brotli={section['brotli_payload_bytes']:,}, K_in_wire=0 vs original=28)"
    )
    print(f"[admm-build-no-dead-k]   rel_err vs int8 quantized symbols: {section['rel_err']:.6f}")

    pr101_inner = _read_pr101_inner_blob(args.frontier_archive)
    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    if len(_orig_decoder) != DECODER_BLOB_LEN:
        raise SystemExit(f"FATAL: PR101 frontier decoder length {len(_orig_decoder)} != expected {DECODER_BLOB_LEN}")
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(f"FATAL: PR101 latent_blob length {len(latent_blob)} != expected {LATENT_BLOB_LEN}")
    print(f"[admm-build-no-dead-k]   PR101 latent_blob={len(latent_blob):,} B sidecar_blob={len(sidecar_blob):,} B")

    inner_blob = _build_inner_blob(section["decoder_bytes"], latent_blob, sidecar_blob)
    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[admm-build-no-dead-k] WROTE archive: {archive_path.relative_to(REPO_ROOT)} "
        f"size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    _stage_forked_submission_dir_no_k(submission_dir, pr101_source_dir=args.pr101_source_dir)
    print(f"[admm-build-no-dead-k] WROTE submission dir: {submission_dir.relative_to(REPO_ROOT)}")

    print("[smoke] running CPU roundtrip ...")
    smoke = _local_smoke_roundtrip(
        archive_path,
        pr101_state_dict_path=args.state_dict,
        submission_dir=submission_dir,
    )
    print(
        f"[smoke] rel_err_vs_quantized_fp32={smoke['rel_err_vs_quantized_fp32']:.6f} "
        f"max_per_tensor={smoke['max_per_tensor_rel_err']:.6f} "
        f"n_tensors={smoke['n_tensors_compared']} "
        f"n_latent_pairs={smoke['n_latent_pairs_decoded']}"
    )
    if smoke["rel_err_vs_quantized_fp32"] > section["rel_err"] * 2 + 1e-3:
        sys.exit(
            f"FATAL: roundtrip rel_err {smoke['rel_err_vs_quantized_fp32']:.4f} "
            f">> encoder rel_err {section['rel_err']:.4f}; wire-format bug"
        )
    if smoke["n_latent_pairs_decoded"] != 600:
        sys.exit(
            f"FATAL: smoke decoded n_pairs={smoke['n_latent_pairs_decoded']} != 600 "
            "(PR101 N_PAIRS); latent_blob passthrough broken"
        )
    removed_cache_dirs = _remove_python_caches(submission_dir)
    if removed_cache_dirs:
        print("[smoke] removed import caches from submission_dir: " + ", ".join(removed_cache_dirs))

    selected_Ks_fp32_smoke_guard = _selected_Ks_fp32_smoke_safety_guard(
        k_source_metadata=k_source_metadata,
        smoke=smoke,
        archive_bytes=archive_bytes,
        max_fp32_smoke_rel_err=float(args.selected_Ks_max_fp32_smoke_rel_err),
    )
    if selected_Ks_fp32_smoke_guard.get("applied"):
        fp32_rel_err = selected_Ks_fp32_smoke_guard["aggregate_fp32_smoke_rel_err"]
        fp32_rel_err_text = "invalid" if fp32_rel_err is None else f"{float(fp32_rel_err):.6f}"
        print(
            "[selected-Ks guard] "
            f"verdict={selected_Ks_fp32_smoke_guard['verdict']} "
            f"fp32_rel_err={fp32_rel_err_text} "
            f"threshold={selected_Ks_fp32_smoke_guard['max_allowed_fp32_smoke_rel_err']:.6f}"
        )

    guard_fields = _guard_fields_with_selected_Ks_source_blockers(
        k_source_metadata,
        selected_Ks_fp32_smoke_guard=selected_Ks_fp32_smoke_guard,
    )
    selected_Ks_guard_applied = bool(selected_Ks_fp32_smoke_guard.get("applied"))
    rel_err_actual_fp32_smoke = (
        selected_Ks_fp32_smoke_guard["aggregate_fp32_smoke_rel_err"]
        if selected_Ks_guard_applied
        else smoke["rel_err_vs_quantized_fp32"]
    )
    max_per_tensor_rel_err_fp32_smoke = (
        selected_Ks_fp32_smoke_guard["max_per_tensor_fp32_smoke_rel_err"]
        if selected_Ks_guard_applied
        else smoke["max_per_tensor_rel_err"]
    )
    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "lane_id": LANE_ID,
        "built_at_utc": _utc_now_iso(),
        "source_admm_manifest": ADMM_SOURCE_MANIFEST,
        "original_variant_tool": ORIGINAL_VARIANT_TOOL,
        **k_source_metadata,
        "selected_Ks_source_blockers_closed_by_this_build": (_closed_selected_Ks_source_blockers(k_source_metadata)),
        "review_eng_finding_closed": "C2_drop_dead_K_bytes_28_byte_savings",
        "wire_format_diff_vs_original": (
            "removed K_section (28 bytes uint8 per-tensor) since the original "
            "decoder discarded these bytes (inflate.py L117-121); rounded "
            "int8 stream + fp16 scales are sufficient for reconstruction"
        ),
        "admm_rms_target": ADMM_RMS_TARGET,
        "admm_lambda": ADMM_PROXY_LAMBDA,
        "admm_proxy_archive_bytes_original_variant": ADMM_PROXY_ARCHIVE_BYTES,
        "admm_proxy_rel_err": ADMM_PROXY_REL_ERR,
        "rel_err_actual_int8": section["rel_err"],
        "rel_err_actual_fp32_smoke": rel_err_actual_fp32_smoke,
        "max_per_tensor_rel_err_fp32_smoke": max_per_tensor_rel_err_fp32_smoke,
        "selected_Ks_fp32_smoke_safety_guard": selected_Ks_fp32_smoke_guard,
        "brotli_quality": args.brotli_quality,
        "archive_relpath": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "submission_dir_relpath": str(submission_dir.relative_to(REPO_ROOT)),
        "submission_dir_import_cache_dirs_removed": removed_cache_dirs,
        "input_state_dict": str(args.state_dict),
        "input_frontier_archive": str(args.frontier_archive),
        "input_pr101_source_dir": str(args.pr101_source_dir),
        "section_total_bytes": section["section_total_bytes"],
        "section_brotli_payload_bytes": section["brotli_payload_bytes"],
        "section_K_bytes_in_wire_format": section["K_bytes_in_wire_format"],  # 0
        "n_tensors": section["n_tensors"],
        "n_symbols": section["n_symbols"],
        "per_tensor_K_audit_only": section["per_tensor_K"],
        "per_tensor_scale_fp16": section["per_tensor_scale_fp16"],
        "smoke_n_latent_pairs_decoded": smoke["n_latent_pairs_decoded"],
        "smoke_latent_dim_meta": smoke["latent_dim_meta"],
        "smoke_base_channels_meta": smoke["base_channels_meta"],
        "smoke_eval_size_meta": smoke["eval_size_meta"],
        "predicted_band": [args.predicted_low, args.predicted_high],
        "evidence_grade": "[CPU-build]",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        **guard_fields,
    }
    build_manifest_path.write_text(json.dumps(build_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"[admm-build-no-dead-k] manifest: "
        f"{build_manifest_path.relative_to(REPO_ROOT)} "
        f"(archive={archive_bytes:,} B, sha256={archive_sha[:16]}...)"
    )
    print(
        "[admm-build-no-dead-k] DONE. CPU build complete. "
        "ready_for_exact_eval_dispatch=False; "
        "cuda_eval_worth_testing=True. "
        "REVIEW-ENG C3 dispatch_blocker apogee_int6_contest_cuda_anchor_required_first "
        "applies to BOTH variants."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
