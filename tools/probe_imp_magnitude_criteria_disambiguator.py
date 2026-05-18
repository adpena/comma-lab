#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane 17 IMP magnitude-criteria disambiguator.

This is a no-spend, local probe for the Catalog #308 binding revision in
``council_per_substrate_symposium_lane_17_imp_20260517.md``. It compares
multiple IMP mask criteria on the same renderer anchor and emits a JSON packet
that is deliberately not score authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.imps_renderer_archive import encode_imps_archive  # noqa: E402
from tac.iterative_magnitude_pruning import (  # noqa: E402
    apply_mask_to_model,
    iter_prunable_parameters,
)

SCHEMA = "lane17_imp_magnitude_criteria_disambiguator_v1"
LANE_ID = "lane_17_imp_10cycle"
SUBSTRATE_ID = "lane_17_imp"
DEFAULT_ANCHOR = (
    Path("experiments") / "results" / "lane_g_v3_landed" / "iter_0" / "renderer.bin"
)
DEFAULT_OUTPUT = (
    Path(".omx")
    / "research"
    / "lane17_imp_magnitude_criteria_disambiguator_20260518_codex.json"
)
DEFAULT_CYCLES = 10
DEFAULT_SPARSITY_INCREMENT = 0.20
AUTHORITY_FALSE = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_paid_dispatch": False,
    "rank_or_kill_eligible": False,
}


ScoreFn = Callable[[str, torch.Tensor, dict[str, Any]], torch.Tensor]


@dataclass(frozen=True)
class CriterionSpec:
    criterion_id: str
    label: str
    methodology: str
    pruning_scope: str
    saliency_source: str
    status_without_sidecar: str
    score_fn: ScoreFn | None = None


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fan_in(param: torch.Tensor) -> int:
    if param.ndim >= 2:
        return max(1, int(param[0].numel()))
    return max(1, int(param.numel()))


def _global_abs_score(
    _name: str, param: torch.Tensor, _context: dict[str, Any]
) -> torch.Tensor:
    return param.detach().cpu().float().abs()


def _obd_hessian_trace_proxy_score(
    _name: str, param: torch.Tensor, _context: dict[str, Any]
) -> torch.Tensor:
    # OBD saliency is 0.5 * H_ii * w_i^2. Without a real Hessian sidecar this
    # local probe uses a transparent fan-in curvature proxy and keeps the output
    # advisory-only.
    scale = 1.0 / math.sqrt(float(_fan_in(param)))
    return param.detach().cpu().float().pow(2).mul(scale)


def _score_gradient_saliency_score(
    name: str, param: torch.Tensor, context: dict[str, Any]
) -> torch.Tensor:
    saliency = context.get("score_gradient_saliency") or {}
    scale = float(saliency.get(name, 1.0))
    return param.detach().cpu().float().abs().mul(scale)


CRITERIA = (
    CriterionSpec(
        criterion_id="l1_per_tensor_canonical_frankle",
        label="L1 per-tensor canonical Frankle 2019",
        methodology="per_tensor_lowest_abs_weight",
        pruning_scope="per_tensor",
        saliency_source="weight_magnitude_non_score_aware",
        status_without_sidecar="ready_local_byte_mask_proxy",
        score_fn=None,
    ),
    CriterionSpec(
        criterion_id="global_l1_canonical_frankle",
        label="Global L1 magnitude control",
        methodology="global_lowest_abs_weight",
        pruning_scope="global",
        saliency_source="weight_magnitude_non_score_aware",
        status_without_sidecar="ready_local_byte_mask_proxy",
        score_fn=_global_abs_score,
    ),
    CriterionSpec(
        criterion_id="hessian_trace_per_tensor_obd_proxy",
        label="Hessian-trace per-tensor OBD proxy",
        methodology="global_lowest_weight2_fan_in_curvature_proxy",
        pruning_scope="global",
        saliency_source="hessian_trace_proxy_without_score_loss",
        status_without_sidecar="advisory_proxy_requires_score_loss_hessian_for_authority",
        score_fn=_obd_hessian_trace_proxy_score,
    ),
    CriterionSpec(
        criterion_id="score_gradient_saliency_catalog123",
        label="Score-gradient saliency per Catalog #123",
        methodology="global_lowest_abs_weight_times_score_gradient_saliency",
        pruning_scope="global",
        saliency_source="score_gradient_param_saliency",
        status_without_sidecar=(
            "blocked_missing_score_gradient_saliency_sidecar_for_authority"
        ),
        score_fn=_score_gradient_saliency_score,
    ),
)


def load_anchor_renderer(anchor_path: Path) -> tuple[nn.Module, bytes, str]:
    raw = anchor_path.read_bytes()
    magic = raw[:4]
    if magic == b"FP4A":
        from tac.renderer_export import load_asymmetric_checkpoint_fp4

        return load_asymmetric_checkpoint_fp4(raw, device="cpu"), raw, magic.decode()
    if magic == b"ASYM":
        from tac.renderer_export import load_asymmetric_checkpoint

        return load_asymmetric_checkpoint(raw, device="cpu"), raw, magic.decode()
    raise ValueError(
        f"unsupported anchor magic {magic!r}; only FP4A and ASYM Lane-G anchors "
        "are valid for this no-spend IMP probe"
    )


def _initial_mask(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(param, dtype=torch.bool, device="cpu")
        for name, param in iter_prunable_parameters(model)
    }


def _kth_prune_mask(
    values: torch.Tensor,
    old_mask: torch.Tensor,
    sparsity_increment: float,
) -> torch.Tensor:
    surviving = values[old_mask]
    n_surviving = int(surviving.numel())
    if n_surviving <= 1:
        return old_mask.clone()
    n_to_prune = int(round(n_surviving * sparsity_increment))
    n_to_prune = max(1, min(n_to_prune, n_surviving - 1))
    threshold = torch.kthvalue(surviving.flatten(), n_to_prune).values.item()
    return (values > threshold) & old_mask


def build_per_tensor_l1_masks(
    model: nn.Module,
    *,
    cycles: int,
    sparsity_increment: float,
) -> dict[str, torch.Tensor]:
    masks = _initial_mask(model)
    params = iter_prunable_parameters(model)
    for _cycle in range(cycles):
        next_masks: dict[str, torch.Tensor] = {}
        for name, param in params:
            values = param.detach().cpu().float().abs()
            next_masks[name] = _kth_prune_mask(
                values,
                masks[name],
                sparsity_increment,
            )
        masks = next_masks
    return masks


def build_global_score_masks(
    model: nn.Module,
    *,
    cycles: int,
    sparsity_increment: float,
    score_fn: ScoreFn,
    context: dict[str, Any],
) -> dict[str, torch.Tensor]:
    masks = _initial_mask(model)
    params = iter_prunable_parameters(model)
    for _cycle in range(cycles):
        scored: dict[str, torch.Tensor] = {}
        surviving_scores: list[torch.Tensor] = []
        for name, param in params:
            score = score_fn(name, param, context).detach().cpu().float()
            if score.shape != param.shape:
                raise ValueError(
                    f"criterion returned shape {tuple(score.shape)} for {name}, "
                    f"expected {tuple(param.shape)}"
                )
            scored[name] = score
            surviving_scores.append(score[masks[name]])
        flat = torch.cat([item.flatten() for item in surviving_scores])
        if flat.numel() <= 1:
            break
        n_to_prune = int(round(int(flat.numel()) * sparsity_increment))
        n_to_prune = max(1, min(n_to_prune, int(flat.numel()) - 1))
        threshold = torch.kthvalue(flat, n_to_prune).values.item()
        masks = {
            name: (score > threshold) & masks[name]
            for name, score in scored.items()
        }
    return masks


def mask_stats(mask: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = 0
    kept = 0
    tensors: list[dict[str, Any]] = []
    for name in sorted(mask):
        tensor = mask[name].detach().cpu().bool()
        count = int(tensor.numel())
        kept_count = int(tensor.sum().item())
        total += count
        kept += kept_count
        tensors.append(
            {
                "name": name,
                "value_count": count,
                "kept": kept_count,
                "pruned": count - kept_count,
                "sparsity": (count - kept_count) / count if count else 0.0,
            }
        )
    pruned = total - kept
    top_pruned = max((int(t["pruned"]) for t in tensors), default=0)
    return {
        "tensor_count": len(tensors),
        "value_count": total,
        "kept": kept,
        "pruned": pruned,
        "sparsity": pruned / total if total else 0.0,
        "top_tensor_pruned_share": top_pruned / pruned if pruned else 0.0,
        "tensors": tensors,
        "mask_sha256": mask_sha256(mask),
    }


def mask_sha256(mask: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(mask):
        tensor = mask[name].detach().cpu().bool().contiguous()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(",".join(str(int(dim)) for dim in tensor.shape).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def pairwise_mask_jaccard(
    criteria: list[dict[str, Any]],
    masks_by_id: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ids = [
        item["criterion_id"]
        for item in criteria
        if item["criterion_id"] in masks_by_id
    ]
    for idx, left_id in enumerate(ids):
        for right_id in ids[idx + 1 :]:
            left = masks_by_id[left_id]
            right = masks_by_id[right_id]
            intersection = 0
            union = 0
            for name in sorted(set(left) | set(right)):
                if name not in left or name not in right:
                    continue
                l_pruned = ~left[name].detach().cpu().bool()
                r_pruned = ~right[name].detach().cpu().bool()
                intersection += int((l_pruned & r_pruned).sum().item())
                union += int((l_pruned | r_pruned).sum().item())
            out.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "pruned_set_jaccard": intersection / union if union else 1.0,
                    "intersection_pruned": intersection,
                    "union_pruned": union,
                }
            )
    return out


def _archive_measurement(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    *,
    anchor_bytes: int | None,
) -> dict[str, Any]:
    model.load_state_dict(base_state, strict=True)
    apply_mask_to_model(model, mask)
    blob = encode_imps_archive(model=model, masks=mask)
    return {
        "imps_archive_bytes": len(blob),
        "imps_archive_sha256": sha256_bytes(blob),
        "delta_vs_anchor_bytes": (
            len(blob) - int(anchor_bytes) if anchor_bytes is not None else None
        ),
        "savings_vs_anchor_pct": (
            (int(anchor_bytes) - len(blob)) / int(anchor_bytes) * 100.0
            if anchor_bytes
            else None
        ),
    }


def _load_saliency_json(path: Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None:
        return {}, {
            "path": None,
            "loaded": False,
            "blocker": "score_gradient_saliency_json_not_supplied",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "saliency" in data and isinstance(data["saliency"], dict):
        data = data["saliency"]
    if not isinstance(data, dict):
        raise ValueError(f"saliency JSON must be an object, got {type(data).__name__}")
    saliency = {str(k): float(v) for k, v in data.items()}
    return saliency, {
        "path": str(path),
        "loaded": True,
        "tensor_count": len(saliency),
        "sha256": sha256_bytes(path.read_bytes()),
    }


def build_probe_payload(
    *,
    model: nn.Module,
    anchor_path: str,
    anchor_bytes: bytes | None,
    anchor_magic: str,
    cycles: int = DEFAULT_CYCLES,
    sparsity_increment: float = DEFAULT_SPARSITY_INCREMENT,
    score_gradient_saliency: dict[str, float] | None = None,
    score_gradient_saliency_metadata: dict[str, Any] | None = None,
    measure_archive: bool = True,
    created_utc: str | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")
    if not (0.0 < sparsity_increment < 1.0):
        raise ValueError(
            f"sparsity_increment must be in (0, 1), got {sparsity_increment}"
        )
    prunable = iter_prunable_parameters(model)
    if not prunable:
        raise ValueError("model has no IMP-prunable Conv2d/ConvTranspose2d weights")

    saliency = score_gradient_saliency or {}
    context = {"score_gradient_saliency": saliency}
    base_state = copy.deepcopy(model.state_dict())
    criteria_results: list[dict[str, Any]] = []
    masks_by_id: dict[str, dict[str, torch.Tensor]] = {}
    anchor_size = len(anchor_bytes) if anchor_bytes is not None else None

    for spec in CRITERIA:
        model.load_state_dict(base_state, strict=True)
        status = spec.status_without_sidecar
        score_gradient_without_sidecar = (
            spec.criterion_id == "score_gradient_saliency_catalog123" and not saliency
        )
        if score_gradient_without_sidecar:
            criteria_results.append(
                {
                    "criterion_id": spec.criterion_id,
                    "label": spec.label,
                    "methodology": spec.methodology,
                    "pruning_scope": spec.pruning_scope,
                    "saliency_source": spec.saliency_source,
                    "status": status,
                    "target_cycles": cycles,
                    "sparsity_increment": sparsity_increment,
                    "target_sparsity_formula": (
                        f"1 - (1 - {sparsity_increment})**{cycles}"
                    ),
                    "mask_stats": None,
                    "archive_measurement": None,
                    "measurement_blockers": [
                        "requires_catalog123_score_gradient_saliency_sidecar",
                        "no_global_l1_surrogate_emitted_for_score_gradient_branch",
                    ],
                }
            )
            continue
        if spec.pruning_scope == "per_tensor":
            mask = build_per_tensor_l1_masks(
                model,
                cycles=cycles,
                sparsity_increment=sparsity_increment,
            )
        else:
            assert spec.score_fn is not None
            mask = build_global_score_masks(
                model,
                cycles=cycles,
                sparsity_increment=sparsity_increment,
                score_fn=spec.score_fn,
                context=context,
            )
        stats = mask_stats(mask)
        if spec.criterion_id == "score_gradient_saliency_catalog123" and saliency:
            status = "ready_with_supplied_score_gradient_saliency_sidecar"
        result = {
            "criterion_id": spec.criterion_id,
            "label": spec.label,
            "methodology": spec.methodology,
            "pruning_scope": spec.pruning_scope,
            "saliency_source": spec.saliency_source,
            "status": status,
            "target_cycles": cycles,
            "sparsity_increment": sparsity_increment,
            "target_sparsity_formula": f"1 - (1 - {sparsity_increment})**{cycles}",
            "mask_stats": stats,
        }
        if measure_archive:
            result["archive_measurement"] = _archive_measurement(
                model,
                base_state,
                mask,
                anchor_bytes=anchor_size,
            )
            model.load_state_dict(base_state, strict=True)
        criteria_results.append(result)
        masks_by_id[spec.criterion_id] = mask

    model.load_state_dict(base_state, strict=True)

    missing_score_gradient = not bool(saliency)
    blockers = [
        "not_score_authority",
        "not_promotion_authority",
        "requires_cycle0_empirical_regression_ratio_before_dispatch",
        "requires_train_distill_swap_and_pcc3_wall_clock_evidence_refresh",
        "requires_quantizr_composition_plan_before_frontier_score_claim",
    ]
    if missing_score_gradient:
        blockers.append("requires_catalog123_score_gradient_saliency_sidecar")

    payload = {
        "schema": SCHEMA,
        "created_utc": created_utc or now_utc(),
        "lane_id": LANE_ID,
        "substrate_id": SUBSTRATE_ID,
        "evidence_axis": "[local-IMP-mask-byte-proxy advisory]",
        **AUTHORITY_FALSE,
        "anchor": {
            "path": anchor_path,
            "magic": anchor_magic,
            "bytes": anchor_size,
            "sha256": sha256_bytes(anchor_bytes) if anchor_bytes is not None else None,
        },
        "catalog_308": {
            "required_alternative_methodologies": [
                "l1_per_tensor_canonical_frankle",
                "hessian_trace_per_tensor_obd",
                "score_gradient_saliency_per_catalog_123",
            ],
            "three_magnitude_criteria_callable": True,
            "score_gradient_helper": (
                "tac.score_gradient_param_saliency.compute_score_gradient_param_saliency"
            ),
            "score_gradient_saliency_metadata": score_gradient_saliency_metadata or {},
        },
        "criteria": criteria_results,
        "pairwise_mask_jaccard": pairwise_mask_jaccard(
            criteria_results,
            masks_by_id,
        ),
        "blockers": blockers,
        "summary": {
            "criteria_count": len(criteria_results),
            "callable_criteria": [item["criterion_id"] for item in criteria_results],
            "catalog308_disambiguator_landed": True,
            "score_gradient_sidecar_supplied": not missing_score_gradient,
            "verdict": (
                "catalog308_disambiguator_landed_pending_cycle0_empirical_runs"
            ),
            "recipe_blocker_replacement": (
                "lane_17_imp_requires_catalog308_cycle0_empirical_"
                "regression_ratio_disambiguation"
            ),
        },
    }
    return payload


def build_probe_from_anchor(
    *,
    anchor_path: Path,
    saliency_json: Path | None,
    cycles: int,
    sparsity_increment: float,
    created_utc: str | None = None,
) -> dict[str, Any]:
    model, raw, magic = load_anchor_renderer(anchor_path)
    saliency, saliency_meta = _load_saliency_json(saliency_json)
    return build_probe_payload(
        model=model,
        anchor_path=str(anchor_path),
        anchor_bytes=raw,
        anchor_magic=magic,
        cycles=cycles,
        sparsity_increment=sparsity_increment,
        score_gradient_saliency=saliency,
        score_gradient_saliency_metadata=saliency_meta,
        measure_archive=True,
        created_utc=created_utc,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-renderer",
        type=Path,
        default=DEFAULT_ANCHOR,
        help="FP4A/ASYM renderer.bin anchor to mask locally.",
    )
    parser.add_argument(
        "--score-gradient-saliency-json",
        type=Path,
        default=None,
        help=(
            "Optional dict from tac.score_gradient_param_saliency. Without it, "
            "the Catalog #123 criterion is emitted as callable but blocked for "
            "authority."
        ),
    )
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument(
        "--sparsity-increment",
        type=float,
        default=DEFAULT_SPARSITY_INCREMENT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON artifact path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor = args.anchor_renderer
    if not anchor.is_file():
        raise SystemExit(f"anchor renderer missing: {anchor}")
    payload = build_probe_from_anchor(
        anchor_path=anchor,
        saliency_json=args.score_gradient_saliency_json,
        cycles=args.cycles,
        sparsity_increment=args.sparsity_increment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(payload))
    print(f"wrote {args.output}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
