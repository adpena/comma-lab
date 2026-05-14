#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe the zen-floor interpretation from Z1 MDL output.

The probe arbitrates between two council-approved interpretations:

* static floor: the floor is mostly a source plus scorer property;
* substrate engineering scope: the floor moves with better substrate binding.

Without true byte-to-scorer feature bindings, this remains a proxy planning
artifact. It never emits a score claim or dispatch authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ImportError:  # pragma: no cover - direct execution from tools/
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, sha256_file  # noqa: E402

EXPECTED_Z1_SCHEMA = "tac_scorer_conditional_mdl_ablation_v1"


class ZenFloorProbeError(ValueError):
    """Raised when the zen-floor probe cannot safely consume an input."""


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _default_output_dir(repo_root: Path) -> Path:
    return repo_root / "experiments" / "results" / f"zen_floor_disambiguator_{_utc_stamp()}"


def _resolve_outputs(args: argparse.Namespace, repo_root: Path) -> tuple[Path, Path]:
    output_dir = args.output_dir or _default_output_dir(repo_root)
    output_json = args.output_json or output_dir / "zen_floor_disambiguator.json"
    output_md = args.output_md or output_dir / "zen_floor_disambiguator.md"
    return Path(output_json), Path(output_md)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ZenFloorProbeError(f"JSON input missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ZenFloorProbeError(f"JSON input is not an object: {path}")
    return payload


def _require_fail_closed_source(payload: dict[str, Any]) -> None:
    if payload.get("schema") != EXPECTED_Z1_SCHEMA:
        raise ZenFloorProbeError(
            f"expected Z1 schema {EXPECTED_Z1_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if payload.get(key) is not False:
            raise ZenFloorProbeError(f"unsafe Z1 input: {key} must be false")
    if payload.get("dispatch_attempted") is not False:
        raise ZenFloorProbeError("unsafe Z1 input: dispatch_attempted must be false")


def _feature_binding_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "evidence_grade": "proxy_planning_only",
            "blockers": [
                "true_byte_to_scorer_feature_binding_missing",
                "penultimate_saliency_or_component_response_map_required",
            ],
        }
    payload = _read_json(path)
    available = bool(
        payload.get("true_scorer_feature_binding_available")
        or payload.get("byte_to_scorer_feature_binding_ready")
    )
    return {
        "available": available,
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "evidence_grade": (
            "true_scorer_feature_bound_planning" if available else "proxy_planning_only"
        ),
        "blockers": []
        if available
        else [
            "feature_binding_file_did_not_assert_ready",
            "penultimate_saliency_or_component_response_map_required",
        ],
    }


def _int_at(payload: dict[str, Any], *keys: str) -> int:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return 0
        cur = cur.get(key)
    if isinstance(cur, bool):
        return 0
    try:
        return int(cur)
    except (TypeError, ValueError):
        return 0


def _build_probe(
    *,
    z1_path: Path,
    z1: dict[str, Any],
    feature_binding: dict[str, Any],
) -> dict[str, Any]:
    _require_fail_closed_source(z1)
    role_gain = _int_at(
        z1,
        "measurement_layers",
        "parser_role_conditioned",
        "gap_to_current_bytes_ceil",
    )
    proxy_gain = _int_at(
        z1,
        "measurement_layers",
        "scorer_feature_proxy_conditioned",
        "gap_to_current_bytes_ceil",
    )
    current_bytes = _int_at(
        z1,
        "measurement_layers",
        "unconditional_payload",
        "current_bytes",
    )
    gain_ratio = float(role_gain) / float(current_bytes) if current_bytes > 0 else 0.0
    verdict = z1.get("verdict") if isinstance(z1.get("verdict"), dict) else {}
    true_binding_in_z1 = bool(
        z1.get("true_scorer_conditional_entropy_claim")
        or verdict.get("true_scorer_feature_available")
    )
    true_binding_available = bool(feature_binding.get("available") or true_binding_in_z1)

    static_support = max(0.0, 1.0 - min(1.0, gain_ratio / 0.02))
    substrate_support = min(1.0, gain_ratio / 0.02)
    if role_gain >= 4096 or gain_ratio >= 0.02:
        provisional = "substrate_engineering_scope"
    elif role_gain <= 1024 and gain_ratio < 0.005:
        provisional = "static_floor"
    else:
        provisional = "indeterminate_band"

    if true_binding_available:
        selected = provisional
        authority = "true_scorer_feature_bound_planning"
        blockers = [
            "still_no_score_claim",
            "byte_closed_codec_candidate_required_before_dispatch",
            "exact_cuda_eval_required_before_promotion",
        ]
    else:
        selected = f"proxy_{provisional}"
        authority = "proxy_planning_only"
        blockers = [
            "true_byte_to_scorer_feature_binding_missing",
            "do_not_promote_proxy_entropy_to_score_claim",
            "component_response_or_penultimate_saliency_map_required",
        ]

    return {
        "schema": "zen_floor_disambiguator_v1",
        "schema_version": 1,
        "tool": "tools/probe_zen_floor_disambiguator.py",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_z1": {
            "path": str(z1_path),
            "sha256": sha256_file(z1_path),
            "schema": z1.get("schema"),
            "archive_count": z1.get("archive_count"),
        },
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "planning_artifact_only": not true_binding_available,
        "evidence_grade": authority,
        "true_scorer_feature_bindings": feature_binding,
        "observed_mdl_headroom": {
            "parser_role_conditioned_gain_bytes_ceil": role_gain,
            "scorer_feature_proxy_gain_bytes_ceil": proxy_gain,
            "current_payload_bytes": current_bytes,
            "parser_role_gain_ratio": round(gain_ratio, 12),
            "z1_headroom_class": verdict.get("headroom_class"),
        },
        "interpretations": {
            "static_floor": {
                "hypothesis": "zen_floor_is_source_scorer_runtime_property",
                "support": round(static_support, 6),
                "meaning": (
                    "Small measured parser-conditioned headroom suggests the current "
                    "packet grammar is close to its archive-local entropy floor."
                ),
            },
            "substrate_engineering_scope": {
                "hypothesis": "zen_floor_moves_with_better_substrate_and_feature_binding",
                "support": round(substrate_support, 6),
                "meaning": (
                    "Measurable parser-conditioned headroom suggests more substrate "
                    "engineering and true scorer-feature binding may lower MDL."
                ),
            },
        },
        "verdict": {
            "selected_interpretation": selected,
            "decision_authority": authority,
            "true_scorer_feature_binding_available": true_binding_available,
            "blockers": blockers,
            "next_actions": [
                "Attach byte-to-scorer feature maps or component-response curves.",
                "Use Z1 top sensitivity rows only as bit-allocation priors.",
                "Build a byte-closed candidate before any exact-eval dispatch.",
            ],
        },
        "probe_fields": {
            "catalog_125_hook": "probe_disambiguator",
            "input_decision": "Z1",
            "static_vs_substrate_scope_arbitrated": True,
            "proxy_only_unless_true_feature_bindings": not true_binding_available,
        },
        "autopilot_rows": [
            {
                "candidate_id": "lane_zen_floor_probe_disambiguator_20260514",
                "family": "zen_floor_planning_probe",
                "predicted_score_delta": 0.0,
                "expected_information_gain": 2.0,
                "estimated_dispatch_cost_usd": 0.0,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": blockers,
                "notes": f"[proxy] selected_interpretation={selected}",
            }
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    headroom = payload.get("observed_mdl_headroom")
    if not isinstance(headroom, dict):
        headroom = {}
    lines = [
        "# Zen-Floor Disambiguator",
        "",
        f"- score_claim: `{str(payload.get('score_claim')).lower()}`",
        f"- promotion_eligible: `{str(payload.get('promotion_eligible')).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(payload.get('ready_for_exact_eval_dispatch')).lower()}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- selected_interpretation: `{verdict.get('selected_interpretation')}`",
        f"- parser_role_gain_bytes_ceil: `{headroom.get('parser_role_conditioned_gain_bytes_ceil')}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in verdict.get("blockers", []) if isinstance(verdict.get("blockers"), list) else []:
        lines.append(f"- {blocker}")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(json_path: Path, md_path: Path, payload: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text(payload), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "schema": "zen_floor_disambiguator_error_v1",
        "schema_version": 1,
        "tool": "tools/probe_zen_floor_disambiguator.py",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "planning_artifact_only": True,
        "evidence_grade": "failed_closed",
        "error": {
            "class": "fail_closed_probe_input_error",
            "message": message,
        },
        "verdict": {
            "selected_interpretation": "failed_closed",
            "decision_authority": "none",
            "blockers": ["valid_z1_manifest_required"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe static-floor vs substrate-engineering-scope from Z1 output."
    )
    parser.add_argument("--z1-json", type=Path, required=True)
    parser.add_argument(
        "--feature-binding-json",
        type=Path,
        default=None,
        help=(
            "Optional true byte-to-scorer feature binding JSON. Must assert "
            "true_scorer_feature_binding_available or byte_to_scorer_feature_binding_ready."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_json, output_md = _resolve_outputs(args, repo_root)
    try:
        z1_path = Path(args.z1_json)
        z1 = _read_json(z1_path)
        feature_binding = _feature_binding_summary(args.feature_binding_json)
        payload = _build_probe(
            z1_path=z1_path,
            z1=z1,
            feature_binding=feature_binding,
        )
        _write_outputs(output_json, output_md, payload)
        print(f"wrote_json={output_json}")
        print(f"wrote_markdown={output_md}")
        return 0
    except (ZenFloorProbeError, ValueError, OSError, json.JSONDecodeError) as exc:
        payload = _error_payload(str(exc))
        _write_outputs(output_json, output_md, payload)
        print(f"failed_closed_json={output_json}", file=sys.stderr)
        print(f"failed_closed_markdown={output_md}", file=sys.stderr)
        print(f"error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
