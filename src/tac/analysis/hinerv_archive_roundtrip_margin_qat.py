# SPDX-License-Identifier: MIT
"""Aggregate the four HiNeRV birth surfaces into one decisive QAT receipt.

The accepted hard-birth target margin is measured at four receiver surfaces:

    live -> fakequant -> archive_roundtrip_shadow -> parseback

``archive_roundtrip_shadow`` (latent-quantizer isolation: latents routed through
the exact HIV1 int16 archive decode, decoder weights live) is the bridge surface
that decides whether the parse-back scorer-effect collapse is caused by the
latent quantizer specifically.  This module composes the per-surface survival
rows into ``hi_nerv_archive_roundtrip_margin_qat.v1`` and runs the mechanical
decision table so the next operator (backend QAT vs sidecar lowering vs audit)
is named by the artifact, not assumed.

It is false-authority by design: no score/rank/promotion claim.  The exact
contest objective that ultimately arbitrates any fix is
``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``; a birth that does
not survive the archive interpreter buys zero ``d_seg`` and only adds rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)
from tac.substrates.hi_nerv.birth_survival import (
    MIN_SCORER_EFFECT_RETENTION_FOR_SURVIVAL,
)

HI_NERV_ARCHIVE_ROUNDTRIP_MARGIN_QAT_SCHEMA = "hi_nerv_archive_roundtrip_margin_qat.v1"
_MARGIN_KEYS = ("target_margin_min", "target_margin_p10", "target_margin_p50", "target_margin_mean")


class ArchiveRoundtripMarginQatError(ValueError):
    """Raised when surface rows cannot be aggregated into a QAT receipt."""


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _surface_wrong_to_target(row: Mapping[str, Any], *aliases: str) -> int | None:
    return _int_or_none(
        _first(
            *(row.get(alias) for alias in aliases),
            row.get("surface_wrong_to_target_count"),
            row.get("region_hard_won_count"),
        )
    )


def _surface_margins(row: Mapping[str, Any]) -> dict[str, float | None]:
    cert = _mapping(row.get("target_margin_certificate"))
    return {key: _first(_float_or_none(row.get(key)), _float_or_none(cert.get(key))) for key in _MARGIN_KEYS}


def _retention(surface_wrong: int | None, live_wrong: int | None) -> float | None:
    if surface_wrong is None or live_wrong is None or live_wrong <= 0:
        return None
    return float(surface_wrong) / float(live_wrong)


def _surface_state(retention: float | None, floor: float) -> str:
    """Classify a surface as high / low / unknown against the retention floor."""

    if retention is None:
        return "unknown"
    return "high" if retention >= floor else "low"


def _classify(
    *,
    fakequant: str,
    shadow: str,
    parseback: str,
) -> tuple[str, str, str]:
    """Return (interpretation_case, recommended_lowering, next_operator).

    Mechanical decision table (GPT/operator-agreed); the artifact decides, the
    QAT opt-in is NOT assumed.  ``recommended_lowering`` is advisory only.
    """

    if fakequant == "high" and shadow == "low" and parseback == "low":
        return (
            "A_latents_fine_quantizer_mismatch_confirmed",
            "backend_qat",
            "implement latents_fine HIV1 archive-roundtrip QAT opt-in with target-margin floor",
        )
    if fakequant == "high" and shadow == "high" and parseback == "low":
        return (
            "B_latent_quantizer_not_the_cause",
            "audit_export_selection_or_decoder_sections",
            "shadow survives but parse-back collapses: audit export/selection/head_rgb/parseback tensors",
        )
    if fakequant == "high" and shadow == "low" and parseback == "high":
        return (
            "C_shadow_too_pessimistic_or_wrong",
            "compare_shadow_vs_parseback_decoded_tensors",
            "shadow collapses but parse-back survives: shadow forward differs from archive decode; reconcile",
        )
    if "low" not in (fakequant, shadow, parseback) and "unknown" not in (
        fakequant,
        shadow,
        parseback,
    ):
        return (
            "D_all_surfaces_survive",
            "proceed_to_gate_and_lowering_race",
            "blocker cleared for this surface: route to launch gate + lowering race",
        )
    if fakequant == "low" and shadow == "low" and parseback == "low":
        return (
            "E_all_low_identity_or_stale",
            "audit_action_identity_and_custody",
            "all surfaces low including fakequant: audit action_id/support identity + row staleness",
        )
    return (
        "U_incomplete_surfaces",
        "emit_missing_surface_rows",
        "one or more surface rows missing/unknown retention: emit the in-loop live shadow + parse-back rows",
    )


def build_hinerv_archive_roundtrip_margin_qat_receipt(
    *,
    live_birth_payload: Mapping[str, Any] | None = None,
    fakequant_survival: Mapping[str, Any] | None = None,
    archive_roundtrip_shadow_survival: Mapping[str, Any] | None = None,
    parseback_survival: Mapping[str, Any] | None = None,
    section: str = "latents_fine",
    retention_floor: float | None = None,
) -> dict[str, Any]:
    """Compose the four birth surfaces into the decisive QAT receipt.

    Each survival row must be the canonical ``hi_nerv_target_region_birth_survival.v1``
    shape (or its blocked sibling).  The receipt flattens per-surface
    ``wrong_to_target`` and ``target_margin_*`` and runs the decision table to
    name ``first_failed_surface`` / ``recommended_lowering`` / ``next_operator``.
    """

    floor = float(MIN_SCORER_EFFECT_RETENTION_FOR_SURVIVAL if retention_floor is None else retention_floor)
    birth = _mapping(live_birth_payload)
    fq = _mapping(fakequant_survival)
    shadow = _mapping(archive_roundtrip_shadow_survival)
    pb = _mapping(parseback_survival)
    for label, row in (("fakequant", fq), ("archive_roundtrip_shadow", shadow), ("parseback", pb)):
        if row:
            require_no_truthy_authority_fields(row, context=f"{label}_survival_row")

    action_id = _first(
        birth.get("action_id"),
        fq.get("action_id"),
        shadow.get("action_id"),
        pb.get("action_id"),
    )
    # Same-action proof: every present surface row must carry the SAME action_id.
    present_ids = {
        str(row.get("action_id"))
        for row in (fq, shadow, pb)
        if row.get("action_id") is not None
    }
    action_id_consistent = len(present_ids) <= 1

    live_wrong = _int_or_none(
        _first(
            birth.get("live_wrong_to_target"),
            birth.get("live_wrong_to_target_count"),
            fq.get("live_wrong_to_target"),
            shadow.get("live_wrong_to_target"),
            pb.get("live_wrong_to_target"),
        )
    )
    fq_wrong = _surface_wrong_to_target(fq, "fakequant_wrong_to_target", "fakequant_wrong_to_target_count")
    shadow_wrong = _surface_wrong_to_target(
        shadow,
        "archive_roundtrip_shadow_wrong_to_target",
        "archive_roundtrip_shadow_wrong_to_target_count",
    )
    pb_wrong = _surface_wrong_to_target(pb, "parseback_wrong_to_target", "parseback_wrong_to_target_count")

    fq_ret = _first(_float_or_none(fq.get("scorer_effect_retention_ratio")), _retention(fq_wrong, live_wrong))
    shadow_ret = _first(
        _float_or_none(shadow.get("scorer_effect_retention_ratio")),
        _retention(shadow_wrong, live_wrong),
    )
    pb_ret = _first(_float_or_none(pb.get("scorer_effect_retention_ratio")), _retention(pb_wrong, live_wrong))

    fq_state = _surface_state(fq_ret, floor)
    shadow_state = _surface_state(shadow_ret, floor)
    pb_state = _surface_state(pb_ret, floor)
    interpretation_case, recommended_lowering, next_operator = _classify(
        fakequant=fq_state, shadow=shadow_state, parseback=pb_state
    )

    # first_failed_surface = the earliest surface in pipeline order that collapsed.
    first_failed_surface = None
    for name, state in (
        ("fakequant_mlx", fq_state),
        ("archive_roundtrip_shadow", shadow_state),
        ("parseback_mlx", pb_state),
    ):
        if state == "low":
            first_failed_surface = name
            break

    shadow_meta = _mapping(shadow.get("surface_meta"))
    section_delta = _mapping(shadow_meta.get("section_hiv1_roundtrip_max_abs_delta"))

    receipt: dict[str, Any] = {
        "schema": HI_NERV_ARCHIVE_ROUNDTRIP_MARGIN_QAT_SCHEMA,
        "family": "hinerv",
        "action_id": None if action_id is None else str(action_id),
        "action_id_consistent_across_surfaces": bool(action_id_consistent),
        "support_sha256": _first(
            birth.get("support_sha256"),
            fq.get("support_sha256"),
            shadow.get("support_sha256"),
            pb.get("support_sha256"),
        ),
        "section": str(section),
        "retention_floor": floor,
        "live_wrong_to_target": live_wrong,
        "fakequant_wrong_to_target": fq_wrong,
        "archive_roundtrip_shadow_wrong_to_target": shadow_wrong,
        "parseback_wrong_to_target": pb_wrong,
        "fakequant_retention": fq_ret,
        "archive_roundtrip_shadow_retention": shadow_ret,
        "parseback_retention": pb_ret,
        "fakequant_surface_state": fq_state,
        "archive_roundtrip_shadow_surface_state": shadow_state,
        "parseback_surface_state": pb_state,
        "section_hiv1_roundtrip_max_abs_delta": section_delta or None,
        "first_failed_surface": first_failed_surface,
        "first_failed_section": (str(section) if first_failed_surface == "archive_roundtrip_shadow" else None),
        "interpretation_case": interpretation_case,
        "recommended_lowering": recommended_lowering,
        "next_operator": next_operator,
        "surfaces": {
            "live": {"wrong_to_target": live_wrong, **_surface_margins(birth)},
            "fakequant_mlx": {"wrong_to_target": fq_wrong, "retention": fq_ret, **_surface_margins(fq)},
            "archive_roundtrip_shadow": {
                "wrong_to_target": shadow_wrong,
                "retention": shadow_ret,
                **_surface_margins(shadow),
            },
            "parseback_mlx": {"wrong_to_target": pb_wrong, "retention": pb_ret, **_surface_margins(pb)},
        },
        "blockers": _aggregate_blockers(
            action_id_consistent=action_id_consistent,
            first_failed_surface=first_failed_surface,
            shadow_present=bool(shadow),
        ),
        "human_visual_fidelity_objective": False,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    # Flatten per-surface margin p10 (the gate's certificate percentile).
    for name, row in (
        ("live", birth),
        ("fakequant", fq),
        ("archive_roundtrip_shadow", shadow),
        ("parseback", pb),
    ):
        for key, value in _surface_margins(row).items():
            receipt[f"{name}_{key}"] = value
    return receipt


def _aggregate_blockers(
    *,
    action_id_consistent: bool,
    first_failed_surface: str | None,
    shadow_present: bool,
) -> list[str]:
    blockers: list[str] = [
        "hi_nerv_archive_roundtrip_margin_qat_is_local_false_authority",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not action_id_consistent:
        blockers.append("archive_roundtrip_margin_qat_action_id_mismatch_across_surfaces")
    if not shadow_present:
        blockers.append("archive_roundtrip_shadow_surface_row_missing")
    if first_failed_surface is not None:
        blockers.append(f"hinerv_birth_first_failed_surface:{first_failed_surface}")
    return blockers
