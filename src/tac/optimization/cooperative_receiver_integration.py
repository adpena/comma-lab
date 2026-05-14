# SPDX-License-Identifier: MIT
"""Unified solver integration for cooperative-receiver campaigns.

The campaign queue is the score-lowering plan; this module makes it useful to
the rest of the solver stack. It emits one proxy-safe manifest that threads the
same rows into autopilot candidates, meta-Lagrangian atoms, Pareto constraints,
continual-learning policy, xray signatures, packet-compiler stages, and
magic-codec awareness.

No score authority is created here. Predicted rows remain blocked until a
byte-closed archive/runtime packet and exact eval artifacts exist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.codec_magic_registry import all_entries as codec_magic_entries
from tac.optimization.cooperative_receiver_campaigns import (
    GENERATED_AT_STABLE,
    build_campaign_queue,
)
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    validate_proxy_candidate,
)
from tac.packet_compiler.cooperative_receiver_grammars import compiler_hook_rows
from tac.packet_compiler.deterministic_compiler import (
    COMPILER_MODES,
    TARGET_PROFILES,
)
from tac.packet_compiler.deterministic_compiler import (
    SCHEMA_VERSION as DETERMINISTIC_COMPILER_SCHEMA,
)

INTEGRATION_SCHEMA = "tac_cooperative_receiver_solver_integration_v1"
BASE_POSE_DIST_FOR_PLANNING_ONLY_ATOMS = 0.0


def _sha256_path(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows_from_queue(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("top_k")
    if not isinstance(rows, list):
        raise ValueError("campaign queue top_k must be a list")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("campaign queue rows must be objects")
        row_dict = dict(row)
        violations = validate_proxy_candidate(row_dict)
        if violations:
            raise ValueError(f"{row_dict.get('campaign_id')}: proxy violations {violations}")
        out.append(row_dict)
    return out


def build_autopilot_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return CandidateRow-compatible planning rows for the autopilot surface."""

    out: list[dict[str, Any]] = []
    for row in rows:
        candidate = {
            "candidate_id": str(row["candidate_id"]),
            "campaign_id": str(row["campaign_id"]),
            "lane_id": str(row["lane_id"]),
            "lane_class": str(row.get("lane_class") or ""),
            "campaign_tier": str(row.get("campaign_tier") or ""),
            "family": "cooperative_receiver_campaign",
            "predicted_score_delta": float(row["predicted_score_delta"]),
            "expected_information_gain": abs(float(row["predicted_score_delta"])),
            "estimated_dispatch_cost_usd": float(row["estimated_dispatch_cost_usd"]),
            "estimated_cost_usd_band": list(row.get("estimated_cost_usd_band") or []),
            "cost_metadata": dict(row.get("cost_metadata") or {}),
            "expected_horizon_weeks": str(row.get("expected_horizon_weeks") or ""),
            "timeline_metadata": dict(row.get("timeline_metadata") or {}),
            "dependency_gate": str(row.get("dependency_gate") or ""),
            "operator_decision_required": bool(row.get("operator_decision_required") or False),
            "blockers": list(row.get("dispatch_blockers") or []),
            "notes": (
                "[predicted; cooperative-receiver planning-only] "
                f"{row['hypothesis']} promotion_gate={row['promotion_gate']}"
            ),
            "source_campaign_id": str(row["campaign_id"]),
            "timing_smoke_command": str(row["timing_smoke_command"]),
            "target_axis": str(row["target_axis"]),
        }
        out.append(apply_proxy_evidence_boundary(candidate))
    return out


def _campaign_row_to_atom(row: Mapping[str, Any]) -> dict[str, Any]:
    predicted_total_delta = float(row["predicted_score_delta"])
    return {
        "atom_id": f"cooperative_receiver:{row['campaign_id']}",
        "family": "cooperative_receiver",
        "family_group": str(row["target_axis"]),
        "pareto_scope": f"cooperative_receiver:{row['target_axis']}",
        "byte_delta": 0,
        # Meta allocator consumes component deltas, so translate the total
        # score prediction through the SegNet score term. This is explicitly
        # planning-only and proxy-blocked below.
        "expected_seg_dist_delta": predicted_total_delta / 100.0,
        "expected_pose_dist_delta": 0.0,
        "confidence": 0.35,
        "evidence_grade": "planning",
        "proxy_row": True,
        "expected_information_gain_nats": abs(predicted_total_delta),
        "interaction_assumptions": [
            "cooperative_receiver_campaigns_are_not_additive_until_exact_eval",
            "predicted_delta_is_cross_domain_prior_not_score_evidence",
            "byte_closed_packet_required_before_pareto_binding",
        ],
        "conflicts_with_families": [],
        "conflicts_with_atoms": [],
        "evidence_source_path": str(row["source_memo"]),
        "evidence_source_sha256": _sha256_path(str(row["source_memo"])),
        "dispatch_blockers": list(row.get("dispatch_blockers") or []),
        "campaign_id": str(row["campaign_id"]),
        "lane_id": str(row["lane_id"]),
        "lane_class": str(row.get("lane_class") or ""),
        "campaign_tier": str(row.get("campaign_tier") or ""),
        "timeline_metadata": dict(row.get("timeline_metadata") or {}),
        "cost_metadata": dict(row.get("cost_metadata") or {}),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_meta_lagrangian_ledger(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return canonical meta-Lagrangian atom ledger for campaign rows."""

    atoms = [_campaign_row_to_atom(row) for row in rows]
    return build_atom_ledger(
        atoms,
        base_pose_dist=BASE_POSE_DIST_FOR_PLANNING_ONLY_ATOMS,
        source="cooperative_receiver_integration",
    )


def build_pareto_constraint_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return non-binding Pareto rows explaining why proxy rows cannot bind."""

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "campaign_id": str(row["campaign_id"]),
                "candidate_id": str(row["candidate_id"]),
                "lane_id": str(row["lane_id"]),
                "lane_class": str(row.get("lane_class") or ""),
                "campaign_tier": str(row.get("campaign_tier") or ""),
                "timeline_metadata": dict(row.get("timeline_metadata") or {}),
                "cost_metadata": dict(row.get("cost_metadata") or {}),
                "pareto_scope": f"cooperative_receiver:{row['target_axis']}",
                "pareto_eligible": False,
                "pareto_frontier": False,
                "non_binding_rationale": (
                    "prediction/proxy row lacks verified byte-closed archive manifest "
                    "and exact eval evidence"
                ),
                "objectives": {
                    "predicted_score_delta": float(row["predicted_score_delta"]),
                    "estimated_dispatch_cost_usd": float(row["estimated_dispatch_cost_usd"]),
                    "ev_per_dollar_proxy": float(row["ev_per_dollar_proxy"]),
                },
                "next_binding_gate": str(row["promotion_gate"]),
            }
        )
    return out


def build_continual_learning_policy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return fail-closed posterior update policy for prediction-only rows."""

    return {
        "schema": "cooperative_receiver_continual_learning_policy_v1",
        "posterior_update_allowed": False,
        "reason": "campaign queue rows are predictions/proxy planning artifacts",
        "rows_blocked_from_posterior": [str(row["campaign_id"]) for row in rows],
        "required_before_update": [
            "byte_closed_archive_sha256",
            "runtime_tree_sha256",
            "contest_cuda_auth_eval_json",
            "contest_cpu_axis_companion_or_explicit_axis_gap_record",
            "terminal_lane_dispatch_claim",
        ],
        "canonical_update_path": "tac.continual_learning.posterior_update_locked",
        "score_claim": False,
    }


def build_magic_codec_rows() -> list[dict[str, object]]:
    """Return canonical magic-codec registry rows for operator visibility."""

    return [
        {
            "magic_hex": entry.magic.hex(),
            "magic_ascii": entry.magic.decode("ascii", errors="replace"),
            "name": entry.name,
            "decode_module": entry.decode_module,
            "encode_module": entry.encode_module,
            "description": entry.description,
        }
        for entry in codec_magic_entries()
    ]


def build_packet_compiler_hook() -> dict[str, object]:
    """Return deterministic compiler policy threaded into the manifest."""

    return {
        "schema": DETERMINISTIC_COMPILER_SCHEMA,
        "canonical_module": "tac.packet_compiler.deterministic_compiler",
        "allowed_modes": list(COMPILER_MODES),
        "target_profiles": list(TARGET_PROFILES),
        "cooperative_receiver_packet_grammars": compiler_hook_rows(),
        "required_order": [
            "representation",
            "prediction",
            "quantization",
            "hyperprior",
            "arithmetic",
            "pack",
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_integration_manifest(
    *,
    queue: Mapping[str, Any] | None = None,
    generated_at_utc: str = GENERATED_AT_STABLE,
) -> dict[str, Any]:
    """Build the canonical cooperative-receiver solver integration manifest."""

    campaign_queue = dict(queue or build_campaign_queue(generated_at_utc=generated_at_utc))
    rows = _rows_from_queue(campaign_queue)
    autopilot_rows = build_autopilot_rows(rows)
    meta_lagrangian = build_meta_lagrangian_ledger(rows)
    pareto_rows = build_pareto_constraint_rows(rows)
    continual_learning = build_continual_learning_policy(rows)
    manifest = {
        "schema": INTEGRATION_SCHEMA,
        "generated_at_utc": generated_at_utc,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "campaign_queue_schema": campaign_queue["schema"],
        "campaign_count": len(rows),
        "autopilot_dispatch_hook": {
            "candidate_row_compatible": True,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "rows": autopilot_rows,
        },
        "meta_lagrangian_hook": meta_lagrangian,
        "pareto_constraint_hook": {
            "binding": False,
            "reason": "all rows are proxy/planning until byte-closed exact eval",
            "rows": pareto_rows,
        },
        "continual_learning_hook": continual_learning,
        "xray_hook": {
            "static_archive_classifier": "tools/xray_substrate_classifier.py",
            "cooperative_receiver_packet_grammars": compiler_hook_rows(),
        },
        "magic_codec_hook": {
            "canonical_registry": "tac.codec_magic_registry",
            "entries": build_magic_codec_rows(),
        },
        "packet_compiler_hook": build_packet_compiler_hook(),
        "score_axis_contract": {
            "contest_cuda": "required_before_promotion",
            "contest_cpu": "required_as_companion_axis_or_explicit_axis_gap_record",
            "macos_cpu_advisory": "proxy_only",
            "score_claim": False,
        },
    }
    _validate_manifest_proxy_safety(manifest)
    return manifest


def _validate_manifest_proxy_safety(manifest: Mapping[str, Any]) -> None:
    if manifest.get("score_claim") is not False:
        raise ValueError("integration manifest must not claim score")
    rows = manifest.get("autopilot_dispatch_hook", {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError("autopilot rows must be a list")
    for row in rows:
        violations = validate_proxy_candidate(row)
        if violations:
            raise ValueError(f"{row.get('candidate_id')}: proxy violations {violations}")


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact operator-readable integration summary."""

    autopilot = manifest["autopilot_dispatch_hook"]
    rows = autopilot["rows"]
    lines = [
        "# Cooperative-Receiver Solver Integration",
        "",
        f"- schema: `{manifest['schema']}`",
        f"- campaign_count: `{manifest['campaign_count']}`",
        f"- score_claim: `{str(manifest['score_claim']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(manifest['ready_for_exact_eval_dispatch']).lower()}`",
        f"- meta_lagrangian_rows: `{len(manifest['meta_lagrangian_hook']['rows'])}`",
        f"- xray_grammars: `{len(manifest['xray_hook']['cooperative_receiver_packet_grammars'])}`",
        "",
        "| rank | campaign | predicted delta | cost | compiler/xray surface | next gate |",
        "|---:|---|---:|---:|---|---|",
    ]
    grammar_by_campaign = {
        str(row["campaign_id"]): row
        for row in manifest["packet_compiler_hook"]["cooperative_receiver_packet_grammars"]
    }
    for idx, row in enumerate(rows, start=1):
        grammar = grammar_by_campaign.get(str(row["source_campaign_id"]))
        surface = (
            f"`{grammar['magic_ascii']}` / `{grammar['substrate_class']}`"
            if grammar
            else "`planning_only_no_packet_magic_yet`"
        )
        lines.append(
            "| {idx} | `{campaign}` | `{delta:.6f}` | `${cost:.2f}` | {surface} | {gate} |".format(
                idx=idx,
                campaign=row["source_campaign_id"],
                delta=float(row["predicted_score_delta"]),
                cost=float(row["estimated_dispatch_cost_usd"]),
                surface=surface,
                gate=row["notes"].split("promotion_gate=", 1)[-1],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_integration_manifest(
    output: str | Path,
    *,
    markdown_output: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON and optional markdown integration artifacts."""

    manifest = build_integration_manifest()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if markdown_output is not None:
        md_path = Path(markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


__all__ = [
    "INTEGRATION_SCHEMA",
    "build_autopilot_rows",
    "build_continual_learning_policy",
    "build_integration_manifest",
    "build_magic_codec_rows",
    "build_meta_lagrangian_ledger",
    "build_packet_compiler_hook",
    "build_pareto_constraint_rows",
    "render_markdown",
    "write_integration_manifest",
]
