#!/usr/bin/env python3
"""Plan rel-error-safe factorized HNeRV rank schedules.

This is a CPU-only planning tool. It loads a HNeRV substrate state_dict,
measures actual post-quantization SVD reconstruction error for candidate
decoder tensors, and emits rank schedules that can be passed to
``tools/build_factorized_hnerv_archive.py --plan-config``.

The output is deliberately fail-closed: isolated per-tensor brotli savings are
not section-level archive evidence, rel_err is not score-domain evidence, and
the tool never marks a row exact-eval dispatch-ready.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.factorized_hnerv_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    FactorizedHnervCodecError,
    estimate_factorized_byte_savings,
)

SCHEMA = "factorized_hnerv_relerr_schedule_plan.v1"
TOOL = "tools/plan_factorized_hnerv_relerr_schedule.py"

DEFAULT_ANCHOR_ARCHIVE_BYTES = 185_578
DEFAULT_CANDIDATE_INDICES = (0, 2, 4, 6, 8, 10, 12)
DEFAULT_RELERR_CAPS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15)
DEFAULT_RECOMMENDED_MAX_RELERR = 0.06


@dataclass(frozen=True)
class ScheduleRow:
    index: int
    name: str
    rank: int
    rel_err: float
    factor_record_brotli_bytes: int
    non_factor_record_brotli_bytes: int
    isolated_savings_bytes_brotli: int

    @property
    def selected(self) -> bool:
        return self.isolated_savings_bytes_brotli > 0

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "rank": self.rank,
            "rel_err": self.rel_err,
            "factor_record_brotli_bytes": self.factor_record_brotli_bytes,
            "non_factor_record_brotli_bytes": self.non_factor_record_brotli_bytes,
            "isolated_savings_bytes_brotli": self.isolated_savings_bytes_brotli,
            "selected_for_plan_config": self.selected,
            "selection_rule": "positive isolated brotli savings at or below relerr cap",
        }


def _load_build_factorized_module():
    path = REPO_ROOT / "tools" / "build_factorized_hnerv_archive.py"
    spec = importlib.util.spec_from_file_location("_factorized_hnerv_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pr107_substrate(archive_path: Path) -> dict[str, torch.Tensor]:
    """Load a PR107/apogee archive through the canonical factorized builder."""
    builder = _load_build_factorized_module()
    state_dict, _latents = builder._load_pr107_substrate(archive_path)  # noqa: SLF001
    return state_dict


def synthetic_low_rank_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """Small deterministic fixture that gives the planner a positive schedule."""
    torch.manual_seed(seed)
    state = {name: torch.randn(*shape) for name, shape in FIXED_STATE_SCHEMA}
    state["stem.weight"] = torch.randn(1728, 6) @ torch.randn(6, 28)
    state["blocks.0.weight"] = (
        torch.randn(144, 8) @ torch.randn(8, 36 * 3 * 3)
    ).reshape(144, 36, 3, 3)
    return state


def _parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _matrixizable_index(idx: int) -> bool:
    if not 0 <= idx < len(FIXED_STATE_SCHEMA):
        return False
    _name, shape = FIXED_STATE_SCHEMA[idx]
    return len(shape) in (2, 4)


def build_schedule_for_cap(
    state_dict: dict[str, torch.Tensor],
    *,
    relerr_cap: float,
    candidate_indices: tuple[int, ...],
    brotli_quality: int,
    anchor_archive_bytes: int,
) -> dict[str, Any]:
    valid_indices = tuple(idx for idx in candidate_indices if _matrixizable_index(idx))
    estimates = estimate_factorized_byte_savings(
        state_dict,
        valid_indices,
        target_rms_err=relerr_cap,
        brotli_quality=brotli_quality,
    )
    rows = []
    for idx in valid_indices:
        row = estimates[idx]
        rows.append(
            ScheduleRow(
                index=idx,
                name=str(row["name"]),
                rank=int(row["rank"]),
                rel_err=float(row["rel_err"]),
                factor_record_brotli_bytes=int(row["factor_record_brotli_bytes"]),
                non_factor_record_brotli_bytes=int(row["non_factor_record_brotli_bytes"]),
                isolated_savings_bytes_brotli=int(row["isolated_savings_bytes_brotli"]),
            )
        )
    selected = [row for row in rows if row.selected and row.rel_err <= relerr_cap]
    total_savings = sum(row.isolated_savings_bytes_brotli for row in selected)
    estimated_archive_bytes = anchor_archive_bytes - total_savings
    return {
        "relerr_cap": relerr_cap,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_packet_build": bool(selected),
        "candidate_rows": [row.to_json() for row in rows],
        "selected_rows": [row.to_json() for row in selected],
        "selected_factorized_indices": [row.index for row in selected],
        "selected_per_index_rank": {str(row.index): row.rank for row in selected},
        "estimated_isolated_brotli_savings_bytes": total_savings,
        "estimated_archive_bytes_if_isolated_savings_realize": estimated_archive_bytes,
        "plan_config": {
            "factorized_indices": [row.index for row in selected],
            "per_index_rank": {str(row.index): row.rank for row in selected},
        },
        "dispatch_blockers": [
            "isolated_per_tensor_estimates_not_section_archive_bytes",
            "rel_err_not_score_domain_evidence",
            "no_packet_built_by_this_tool",
            "exact_cpu_cuda_auth_eval_required_before_score_claim",
        ],
    }


def build_plan(
    state_dict: dict[str, torch.Tensor],
    *,
    substrate_label: str,
    candidate_indices: tuple[int, ...] = DEFAULT_CANDIDATE_INDICES,
    relerr_caps: tuple[float, ...] = DEFAULT_RELERR_CAPS,
    recommended_max_relerr: float = DEFAULT_RECOMMENDED_MAX_RELERR,
    anchor_archive_bytes: int = DEFAULT_ANCHOR_ARCHIVE_BYTES,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    bad = [idx for idx in candidate_indices if not _matrixizable_index(idx)]
    if bad:
        raise FactorizedHnervCodecError(f"non-matrixizable candidate indices: {bad}")
    schedules = [
        build_schedule_for_cap(
            state_dict,
            relerr_cap=cap,
            candidate_indices=candidate_indices,
            brotli_quality=brotli_quality,
            anchor_archive_bytes=anchor_archive_bytes,
        )
        for cap in relerr_caps
    ]
    eligible = [
        row for row in schedules
        if row["ready_for_packet_build"] and row["relerr_cap"] <= recommended_max_relerr
    ]
    recommended = max(
        eligible,
        key=lambda row: int(row["estimated_isolated_brotli_savings_bytes"]),
        default=None,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "substrate_label": substrate_label,
        "candidate_indices": list(candidate_indices),
        "candidate_names": [FIXED_STATE_SCHEMA[idx][0] for idx in candidate_indices],
        "relerr_caps": list(relerr_caps),
        "recommended_max_relerr": recommended_max_relerr,
        "anchor_archive_bytes": anchor_archive_bytes,
        "brotli_quality": brotli_quality,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "recommended_schedule": recommended,
        "recommended_plan_config": (
            recommended["plan_config"] if recommended is not None else None
        ),
        "schedules": schedules,
        "global_blockers": [
            "cpu_planning_only",
            "isolated_per_tensor_brotli_not_archive_section_measurement",
            "rel_err_not_score_domain_evidence",
            "packet_build_and_strict_compliance_required",
            "dual_axis_exact_eval_required_before_score_claim",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Factorized HNeRV Rel-Err Schedule Plan",
        "",
        f"score_claim: `{str(plan['score_claim']).lower()}`",
        f"ready_for_exact_eval_dispatch: `{str(plan['ready_for_exact_eval_dispatch']).lower()}`",
        f"substrate: `{plan['substrate_label']}`",
        f"recommended_max_relerr: `{plan['recommended_max_relerr']}`",
        "",
        "## Recommended Schedule",
        "",
    ]
    recommended = plan.get("recommended_schedule")
    if recommended is None:
        lines.append("No rel-error-capped schedule has positive isolated brotli savings.")
    else:
        lines.extend([
            f"- relerr_cap: `{recommended['relerr_cap']}`",
            f"- selected tensors: `{recommended['selected_factorized_indices']}`",
            f"- isolated savings bytes: `{recommended['estimated_isolated_brotli_savings_bytes']}`",
            f"- estimated archive bytes if isolated savings realize: "
            f"`{recommended['estimated_archive_bytes_if_isolated_savings_realize']}`",
        ])
    lines.extend(["", "## Schedules", ""])
    for schedule in plan["schedules"]:
        lines.append(
            "- cap `{}`: selected={} isolated_savings={} ready_for_packet_build={}".format(
                schedule["relerr_cap"],
                schedule["selected_factorized_indices"],
                schedule["estimated_isolated_brotli_savings_bytes"],
                str(schedule["ready_for_packet_build"]).lower(),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in plan["global_blockers"]:
        lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--substrate-archive", type=Path)
    source.add_argument("--synthetic-low-rank", action="store_true")
    parser.add_argument("--candidate-indices", type=_parse_int_csv, default=DEFAULT_CANDIDATE_INDICES)
    parser.add_argument("--relerr-caps", type=_parse_float_csv, default=DEFAULT_RELERR_CAPS)
    parser.add_argument("--recommended-max-relerr", type=float, default=DEFAULT_RECOMMENDED_MAX_RELERR)
    parser.add_argument("--anchor-archive-bytes", type=int, default=DEFAULT_ANCHOR_ARCHIVE_BYTES)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--plan-config-out", type=Path)
    parser.add_argument("--print-markdown", action="store_true")
    args = parser.parse_args(argv)

    if args.synthetic_low_rank:
        state_dict = synthetic_low_rank_state_dict()
        substrate_label = "synthetic_low_rank"
    else:
        assert args.substrate_archive is not None
        state_dict = load_pr107_substrate(args.substrate_archive)
        substrate_label = str(args.substrate_archive)

    plan = build_plan(
        state_dict,
        substrate_label=substrate_label,
        candidate_indices=args.candidate_indices,
        relerr_caps=args.relerr_caps,
        recommended_max_relerr=args.recommended_max_relerr,
        anchor_archive_bytes=args.anchor_archive_bytes,
        brotli_quality=args.brotli_quality,
    )
    payload = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(plan), encoding="utf-8")
    if args.plan_config_out:
        if plan["recommended_plan_config"] is None:
            raise SystemExit("no recommended_plan_config; refusing to write empty plan")
        args.plan_config_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_config_out.write_text(
            json.dumps(plan["recommended_plan_config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(render_markdown(plan) if args.print_markdown else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
