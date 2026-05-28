# SPDX-License-Identifier: MIT
"""P19/P18 scorer-region artifacts for queue-owned rate/distortion cascades."""

from __future__ import annotations

import ast
import json
import math
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FEC8_MAGIC,
    FECA_MAGIC,
    FecaSelectorReparameterizationError,
    _decode_selector_payload,
    _load_feca_module,
    _load_markov_module,
    _read_single_member,
    split_fp11_member,
)
from tac.repo_io import sha256_file

P19_POSENET_NULL_PAIRS_SCHEMA = "p19_posenet_null_pair_detection.v1"
P18_SEGNET_REGION_WATERFILL_SCHEMA = "p18_segnet_region_waterfill.v1"
DISTORTION_BUDGET_ATTACK_PLAN_SCHEMA = "receiver_closed_distortion_budget_attack_plan.v1"
FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA = "frame1_region_waterfill_runtime_patch.v1"


class ScorerRegionWaterfillError(ValueError):
    """Raised when scorer-region artifact construction is unsafe."""


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _artifact_record(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionWaterfillError(f"artifact missing: {path}")
    return {
        "path": _repo_rel(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionWaterfillError(f"JSON artifact missing: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerRegionWaterfillError(f"JSON artifact must be an object: {path}")
    return payload


def _literal_tuple_from_python(path: Path, name: str) -> tuple[Any, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, tuple):
                raise ScorerRegionWaterfillError(f"{name} in {path} must be a tuple")
            return value
    raise ScorerRegionWaterfillError(f"{name} not found in {path}")


def active_selector_mode_ids(source_submission_dir: str | Path) -> tuple[str, ...]:
    """Read the active K=16 selector palette from a PR110-family runtime."""

    source_dir = Path(source_submission_dir)
    inflate_py = source_dir / "inflate.py"
    if not inflate_py.is_file():
        raise ScorerRegionWaterfillError(f"missing inflate.py: {inflate_py}")
    values = _literal_tuple_from_python(inflate_py, "FEC6_FIXED_K16_MODE_IDS")
    mode_ids = tuple(str(item) for item in values)
    if not mode_ids:
        raise ScorerRegionWaterfillError("active selector palette is empty")
    return mode_ids


def decode_selector_codes(source_submission_dir: str | Path) -> list[int]:
    """Decode the archive selector codes without executing inflate-time scorers."""

    source_dir = Path(source_submission_dir)
    archive = source_dir / "archive.zip"
    encoder_dir = source_dir / "encoder"
    if not archive.is_file():
        raise ScorerRegionWaterfillError(f"missing source archive: {archive}")
    try:
        _info, member_payload = _read_single_member(archive)
        parts = split_fp11_member(
            member_payload,
            allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC),
        )
        feca_module = _load_feca_module(encoder_dir, module_suffix="p19_decode")
        markov_module = _load_markov_module(encoder_dir, module_suffix="p19_decode_markov")
        return list(
            _decode_selector_payload(
                feca_module=feca_module,
                markov_module=markov_module,
                payload=parts["selector_payload"],
            )
        )
    except FecaSelectorReparameterizationError as exc:
        raise ScorerRegionWaterfillError(str(exc)) from exc


def _pose_null_mode_scores(payload: Mapping[str, Any]) -> dict[str, float]:
    analysis = payload.get("analysis")
    rows = analysis.get("ranked_top_n_by_abs_pose") if isinstance(analysis, Mapping) else None
    if not isinstance(rows, Sequence):
        rows = analysis.get("pose_null_decile") if isinstance(analysis, Mapping) else []
    scores: dict[str, float] = {"none": 0.0}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        mode_id = str(row.get("mode_id") or "").strip()
        if not mode_id:
            continue
        try:
            scores[mode_id] = abs(float(row.get("abs_pose_delta", row.get("pose_delta", 1.0))))
        except (TypeError, ValueError):
            continue
    return scores


def build_p19_posenet_null_pairs(
    *,
    repo_root: str | Path,
    source_submission_dir: str | Path,
    pose_null_modes_artifact: str | Path,
    null_fraction: float = 0.10,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Materialize pair ids whose current selector modes sit in PoseNet-null space."""

    source_dir = _resolve(source_submission_dir, repo_root)
    mode_scores = _pose_null_mode_scores(
        _read_json(pose_null_modes_artifact, repo_root=repo_root)
    )
    if not include_identity:
        mode_scores.pop("none", None)
    active_modes = active_selector_mode_ids(source_dir)
    codes = decode_selector_codes(source_dir)
    if any(code < 0 or code >= len(active_modes) for code in codes):
        raise ScorerRegionWaterfillError("selector code outside active mode palette")
    n_pairs = len(codes)
    target_count = max(1, math.ceil(float(null_fraction) * n_pairs))
    rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []
    for pair_id, code in enumerate(codes):
        mode_id = active_modes[int(code)]
        score = float(mode_scores.get(mode_id, 1.0))
        row = {
            "pair_id": pair_id,
            "selector_code": int(code),
            "mode_id": mode_id,
            "abs_pose_delta_proxy": score,
            "axis_tag": "[macOS-CPU advisory]",
        }
        if mode_id in mode_scores:
            all_candidate_rows.append(row)
        rows.append(row)
    selected = sorted(rows, key=lambda row: (float(row["abs_pose_delta_proxy"]), int(row["pair_id"])))[:target_count]
    selected_pair_ids = [int(row["pair_id"]) for row in selected]
    histogram = Counter(active_modes[int(code)] for code in codes)
    payload = {
        "schema": P19_POSENET_NULL_PAIRS_SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_submission_dir": _repo_rel(source_dir, repo_root),
        "source_archive": _artifact_record(source_dir / "archive.zip", repo_root=repo_root),
        "pose_null_modes_artifact": _artifact_record(pose_null_modes_artifact, repo_root=repo_root),
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "selector_mode_proxy_detection_only",
        "n_pairs": n_pairs,
        "null_fraction": float(null_fraction),
        "target_count": target_count,
        "selected_pair_count": len(selected_pair_ids),
        "selected_pair_ids": selected_pair_ids,
        "detected_null_candidate_pair_count": len(all_candidate_rows),
        "detected_null_candidate_pair_ids": [int(row["pair_id"]) for row in all_candidate_rows],
        "selected_rows": selected,
        "active_mode_ids": list(active_modes),
        "selector_mode_histogram": dict(sorted(histogram.items())),
        "null_mode_scores": dict(sorted(mode_scores.items())),
        "blockers": [
            "p19_pair_detection_is_local_proxy_not_score_authority",
            "receiver_closed_distortion_budget_materializer_required_before_score_use",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="p19_posenet_null_pairs")
    return payload


def _load_softmax(path: str | Path, *, repo_root: str | Path) -> np.ndarray:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise ScorerRegionWaterfillError(f"softmax artifact missing: {path}")
    array = np.load(resolved)
    if array.ndim != 3 or array.shape[2] < 2:
        raise ScorerRegionWaterfillError(
            f"expected softmax shape (pairs, regions, classes), got {array.shape}"
        )
    return np.asarray(array, dtype=np.float64)


def _region_box(region: int, region_count: int, *, width: int, height: int) -> dict[str, float]:
    side = round(math.sqrt(region_count))
    if side * side != region_count:
        raise ScorerRegionWaterfillError(f"region_count must be square, got {region_count}")
    y, x = divmod(int(region), side)
    return {
        "x0": x / side,
        "y0": y / side,
        "x1": (x + 1) / side,
        "y1": (y + 1) / side,
        "pixel_x0": round(width * x / side),
        "pixel_y0": round(height * y / side),
        "pixel_x1": round(width * (x + 1) / side),
        "pixel_y1": round(height * (y + 1) / side),
    }


def _region_scores(softmax: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(softmax, 1e-12, 1.0)
    entropy = -(clipped * np.log2(clipped)).sum(axis=2)
    top2 = np.sort(clipped, axis=2)[:, :, -2:]
    margin = top2[:, :, 1] - top2[:, :, 0]
    entropy_norm = entropy / math.log2(float(clipped.shape[2]))
    vulnerability = entropy_norm * (1.0 - margin)
    class_id = np.argmax(clipped, axis=2)
    return entropy, margin, vulnerability, class_id


def build_p18_segnet_region_waterfill(
    *,
    repo_root: str | Path,
    posenet_null_pairs: str | Path,
    segnet_softmax_16: str | Path,
    segnet_softmax_256: str | Path,
    top_regions_per_pair: int = 4,
    image_width: int = 512,
    image_height: int = 384,
) -> dict[str, Any]:
    """Materialize SegNet-region waterfill rows for P19-selected pairs."""

    p19 = _read_json(posenet_null_pairs, repo_root=repo_root)
    if p19.get("schema") != P19_POSENET_NULL_PAIRS_SCHEMA:
        raise ScorerRegionWaterfillError("posenet_null_pairs schema mismatch")
    require_no_truthy_authority_fields(p19, context="p18_waterfill_p19_input")
    pair_ids = [int(item) for item in p19.get("selected_pair_ids") or []]
    if not pair_ids:
        raise ScorerRegionWaterfillError("posenet_null_pairs contains no selected_pair_ids")
    soft16 = _load_softmax(segnet_softmax_16, repo_root=repo_root)
    soft256 = _load_softmax(segnet_softmax_256, repo_root=repo_root)
    if soft16.shape[0] != soft256.shape[0]:
        raise ScorerRegionWaterfillError("softmax pair counts differ")
    n_pairs = int(soft256.shape[0])
    if any(pair_id < 0 or pair_id >= n_pairs for pair_id in pair_ids):
        raise ScorerRegionWaterfillError("selected P19 pair id outside softmax cache")
    entropy256, margin256, vulnerability256, class256 = _region_scores(soft256)
    entropy16, margin16, vulnerability16, class16 = _region_scores(soft16)
    top_k = max(1, int(top_regions_per_pair))
    rows: list[dict[str, Any]] = []
    selector_region_symbols: list[int] = []
    for pair_id in pair_ids:
        order = np.argsort(-vulnerability256[pair_id])[:top_k]
        coarse_region = int(np.argmax(vulnerability16[pair_id]))
        selector_region_symbols.append(coarse_region)
        regions: list[dict[str, Any]] = []
        for region in order:
            rid = int(region)
            regions.append(
                {
                    "region_id": rid,
                    "class_id": int(class256[pair_id, rid]),
                    "entropy_bits": float(entropy256[pair_id, rid]),
                    "top2_margin": float(margin256[pair_id, rid]),
                    "waterfill_priority": float(vulnerability256[pair_id, rid]),
                    "box": _region_box(
                        rid,
                        int(soft256.shape[1]),
                        width=int(image_width),
                        height=int(image_height),
                    ),
                }
            )
        rows.append(
            {
                "pair_id": int(pair_id),
                "coarse_region16": coarse_region,
                "coarse_region16_class_id": int(class16[pair_id, coarse_region]),
                "coarse_region16_entropy_bits": float(entropy16[pair_id, coarse_region]),
                "coarse_region16_top2_margin": float(margin16[pair_id, coarse_region]),
                "coarse_region16_waterfill_priority": float(vulnerability16[pair_id, coarse_region]),
                "regions256": regions,
            }
        )
    class_histogram = Counter()
    region_histogram = Counter()
    for row in rows:
        for region in row["regions256"]:
            class_histogram[str(region["class_id"])] += 1
            region_histogram[str(region["region_id"])] += 1
    payload = {
        "schema": P18_SEGNET_REGION_WATERFILL_SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "cached_segnet_softmax_region_waterfill_proxy",
        "source_p19_posenet_null_pairs": _artifact_record(posenet_null_pairs, repo_root=repo_root),
        "segnet_softmax_16": _artifact_record(segnet_softmax_16, repo_root=repo_root),
        "segnet_softmax_256": _artifact_record(segnet_softmax_256, repo_root=repo_root),
        "n_pairs_available": n_pairs,
        "selected_pair_count": len(pair_ids),
        "top_regions_per_pair": top_k,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "region_count_16": int(soft16.shape[1]),
        "region_count_256": int(soft256.shape[1]),
        "rows": rows,
        "selector_region_bits": {
            "schema": "selector_region_bits_proxy.v1",
            "region_count": int(soft16.shape[1]),
            "pair_ids": pair_ids,
            "symbols": selector_region_symbols,
            "symbol_histogram": dict(sorted(Counter(selector_region_symbols).items())),
        },
        "class_histogram_top_regions": dict(sorted(class_histogram.items())),
        "region_histogram_top_regions": dict(sorted(region_histogram.items())),
        "blockers": [
            "p18_waterfill_uses_cached_region_softmax_not_exact_auth_score",
            "receiver_closed_materializer_required_before_score_use",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="p18_segnet_region_waterfill")
    return payload


def build_receiver_closed_distortion_budget_attack_plan(
    *,
    repo_root: str | Path,
    chain_report: str | Path,
    posenet_null_pairs: str | Path,
    segnet_region_waterfill: str | Path,
    score_bytes_denominator: int = 37_545_489,
) -> dict[str, Any]:
    """Convert rate-only savings into a receiver-closed distortion budget plan."""

    report = _read_json(chain_report, repo_root=repo_root)
    p19 = _read_json(posenet_null_pairs, repo_root=repo_root)
    p18 = _read_json(segnet_region_waterfill, repo_root=repo_root)
    require_no_truthy_authority_fields(report, context="distortion_budget_chain_report")
    require_no_truthy_authority_fields(p19, context="distortion_budget_p19")
    require_no_truthy_authority_fields(p18, context="distortion_budget_p18")
    if p19.get("schema") != P19_POSENET_NULL_PAIRS_SCHEMA:
        raise ScorerRegionWaterfillError("p19 schema mismatch")
    if p18.get("schema") != P18_SEGNET_REGION_WATERFILL_SCHEMA:
        raise ScorerRegionWaterfillError("p18 schema mismatch")
    try:
        saved_bytes = int(report.get("cumulative_rate_saved_bytes_vs_source") or 0)
    except (TypeError, ValueError):
        saved_bytes = 0
    score_credit = 25.0 * max(0, saved_bytes) / float(score_bytes_denominator)
    rows: list[dict[str, Any]] = []
    p19_ids = {int(item) for item in p19.get("selected_pair_ids") or []}
    for row in p18.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        pair_id = int(row.get("pair_id", -1))
        if pair_id not in p19_ids:
            continue
        regions = row.get("regions256") if isinstance(row.get("regions256"), Sequence) else []
        rows.append(
            {
                "pair_id": pair_id,
                "receiver_operator_family": "frame1_segnet_region_waterfill",
                "p19_pose_null_pair": True,
                "candidate_region_count": len(regions),
                "regions256": regions,
                "estimated_score_credit_available": score_credit / max(1, len(p19_ids)),
                "receiver_closure_required": [
                    "runtime_consumes_selector_region_bits",
                    "inflated_full_frame_output_changes_when_budget_spent",
                    "local_mlx_or_cpu_component_spot_check",
                    "exact_auth_eval_before_score_claim",
                ],
            }
        )
    blockers = ordered_unique(
        [
            *[str(item) for item in report.get("readiness_blockers") or [] if str(item)],
            "distortion_budget_plan_is_not_materialized_frame_delta",
            "receiver_runtime_patch_required_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ]
    )
    payload = {
        "schema": DISTORTION_BUDGET_ATTACK_PLAN_SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chain_report": _artifact_record(chain_report, repo_root=repo_root),
        "posenet_null_pairs": _artifact_record(posenet_null_pairs, repo_root=repo_root),
        "segnet_region_waterfill": _artifact_record(segnet_region_waterfill, repo_root=repo_root),
        "score_bytes_denominator": int(score_bytes_denominator),
        "rate_saved_bytes": saved_bytes,
        "rate_score_credit": score_credit,
        "budget_pair_count": len(rows),
        "budget_rows": rows,
        "receiver_contract_target": "frame1_region_waterfill_runtime_patch_v1",
        "packetir_generalization_targets": [
            "packetir.selector_region_bits",
            "hnerv.receiver_sidecar.region_waterfill",
            "boostnerv.residual_region_budget",
            "nerv_family.frame_pair_region_operator",
            "non_nerv.archive_bound_region_transform",
        ],
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="distortion_budget_attack_plan")
    return payload


def _copy_submission_tree(source_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ScorerRegionWaterfillError(f"output submission dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(
        source_dir,
        output_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def _runtime_patch_rows(
    p18: Mapping[str, Any],
    *,
    max_pairs: int,
    regions_per_pair: int,
) -> list[tuple[int, list[tuple[float, float, float, float]]]]:
    rows: list[tuple[int, list[tuple[float, float, float, float]]]] = []
    for row in p18.get("rows") or []:
        if len(rows) >= max_pairs:
            break
        if not isinstance(row, Mapping):
            continue
        pair_id = int(row.get("pair_id", -1))
        boxes: list[tuple[float, float, float, float]] = []
        for region in (row.get("regions256") or [])[:regions_per_pair]:
            if not isinstance(region, Mapping):
                continue
            box = region.get("box")
            if not isinstance(box, Mapping):
                continue
            boxes.append(
                (
                    float(box["x0"]),
                    float(box["y0"]),
                    float(box["x1"]),
                    float(box["y1"]),
                )
            )
        if pair_id >= 0 and boxes:
            rows.append((pair_id, boxes))
    if not rows:
        raise ScorerRegionWaterfillError("no runtime patch rows could be built")
    return rows


def _region_patch_module_source(
    rows: Sequence[tuple[int, list[tuple[float, float, float, float]]]],
    *,
    rgb_delta: tuple[int, int, int],
) -> str:
    return (
        "# SPDX-License-Identifier: MIT\n"
        '"""Archive-bound frame-1 region waterfill receiver patch."""\n\n'
        "from __future__ import annotations\n\n"
        f"REGION_WATERFILL_ROWS = {list(rows)!r}\n"
        f"RGB_DELTA = {tuple(int(v) for v in rgb_delta)!r}\n\n"
        "def apply_region_waterfill(frames_bchw, *, pair_start: int):\n"
        "    if frames_bchw.shape[0] % 2 != 0:\n"
        "        raise ValueError('region waterfill expects complete frame pairs')\n"
        "    out = frames_bchw.clone()\n"
        "    n_pairs = int(frames_bchw.shape[0] // 2)\n"
        "    _b, _c, height, width = out.shape\n"
        "    for pair_id, boxes in REGION_WATERFILL_ROWS:\n"
        "        local = int(pair_id) - int(pair_start)\n"
        "        if local < 0 or local >= n_pairs:\n"
        "            continue\n"
        "        frame_offset = local * 2 + 1\n"
        "        for x0, y0, x1, y1 in boxes:\n"
        "            xx0 = max(0, min(width, round(float(x0) * width)))\n"
        "            yy0 = max(0, min(height, round(float(y0) * height)))\n"
        "            xx1 = max(0, min(width, round(float(x1) * width)))\n"
        "            yy1 = max(0, min(height, round(float(y1) * height)))\n"
        "            if xx1 <= xx0 or yy1 <= yy0:\n"
        "                continue\n"
        "            out[frame_offset, 0, yy0:yy1, xx0:xx1].add_(RGB_DELTA[0])\n"
        "            out[frame_offset, 1, yy0:yy1, xx0:xx1].add_(RGB_DELTA[1])\n"
        "            out[frame_offset, 2, yy0:yy1, xx0:xx1].add_(RGB_DELTA[2])\n"
        "    return out.clamp_(0.0, 255.0).round_()\n\n"
        "__all__ = ['apply_region_waterfill']\n"
    )


def _patch_inflate_source(source: str) -> str:
    import_anchor = "from model import HNeRVDecoder  # type: ignore[import-not-found]\n"
    if "from region_waterfill_patch import apply_region_waterfill" not in source:
        if import_anchor not in source:
            raise ScorerRegionWaterfillError("inflate.py import anchor not found")
        source = source.replace(
            import_anchor,
            import_anchor
            + "from region_waterfill_patch import apply_region_waterfill  # type: ignore[import-not-found]\n",
            1,
        )
    call_anchor = (
        "            rounded = apply_pr101_selector_to_frames(\n"
        "                rounded,\n"
        "                selector_kind,\n"
        "                selector_codes,\n"
        "                selector_specs,\n"
        "                pair_start=i,\n"
        "            )\n"
    )
    call_insert = call_anchor + "            rounded = apply_region_waterfill(rounded, pair_start=i)\n"
    if "apply_region_waterfill(rounded, pair_start=i)" not in source:
        if call_anchor not in source:
            raise ScorerRegionWaterfillError("inflate.py selector call anchor not found")
        source = source.replace(call_anchor, call_insert, 1)
    return source


def build_frame1_region_waterfill_runtime_patch(
    *,
    repo_root: str | Path,
    source_submission_dir: str | Path,
    segnet_region_waterfill: str | Path,
    output_submission_dir: str | Path,
    candidate_archive: str | Path | None = None,
    candidate_archive_source: str | None = None,
    max_pairs: int = 12,
    regions_per_pair: int = 1,
    rgb_delta: tuple[int, int, int] = (-1, -1, -1),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a receiver-closed frame-1 region patch from P18/P19 waterfill rows."""

    source_dir = _resolve(source_submission_dir, repo_root)
    output_dir = _resolve(output_submission_dir, repo_root)
    candidate_archive_path = (
        _resolve(candidate_archive, repo_root) if candidate_archive is not None else None
    )
    p18 = _read_json(segnet_region_waterfill, repo_root=repo_root)
    require_no_truthy_authority_fields(p18, context="frame1_region_patch_p18_input")
    if p18.get("schema") != P18_SEGNET_REGION_WATERFILL_SCHEMA:
        raise ScorerRegionWaterfillError("p18 schema mismatch")
    if not (source_dir / "inflate.py").is_file() or not (source_dir / "archive.zip").is_file():
        raise ScorerRegionWaterfillError("source submission must contain inflate.py and archive.zip")
    if len(rgb_delta) != 3:
        raise ScorerRegionWaterfillError("rgb_delta must contain exactly 3 values")
    if candidate_archive_path is not None and not candidate_archive_path.is_file():
        raise ScorerRegionWaterfillError(
            f"candidate archive override missing: {candidate_archive_path}"
        )

    rows = _runtime_patch_rows(
        p18,
        max_pairs=max(1, int(max_pairs)),
        regions_per_pair=max(1, int(regions_per_pair)),
    )
    _copy_submission_tree(source_dir, output_dir, overwrite=overwrite)
    if candidate_archive_path is not None:
        shutil.copy2(candidate_archive_path, output_dir / "archive.zip")
    patch_path = output_dir / "src" / "region_waterfill_patch.py"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(
        _region_patch_module_source(rows, rgb_delta=rgb_delta),
        encoding="utf-8",
    )
    inflate_path = output_dir / "inflate.py"
    inflate_path.write_text(
        _patch_inflate_source(inflate_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    payload = {
        "schema": FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_submission_dir": _repo_rel(source_dir, repo_root),
        "output_submission_dir": _repo_rel(output_dir, repo_root),
        "source_archive": _artifact_record(source_dir / "archive.zip", repo_root=repo_root),
        "candidate_archive": _artifact_record(output_dir / "archive.zip", repo_root=repo_root),
        "candidate_archive_override": (
            _artifact_record(candidate_archive_path, repo_root=repo_root)
            if candidate_archive_path is not None
            else None
        ),
        "candidate_archive_source": (
            str(candidate_archive_source).strip()
            if candidate_archive_source is not None
            and str(candidate_archive_source).strip()
            else "source_submission_archive"
        ),
        "segnet_region_waterfill": _artifact_record(segnet_region_waterfill, repo_root=repo_root),
        "runtime_patch": _artifact_record(patch_path, repo_root=repo_root),
        "patched_inflate": _artifact_record(inflate_path, repo_root=repo_root),
        "patched_pair_count": len(rows),
        "regions_per_pair": max(1, int(regions_per_pair)),
        "rgb_delta": [int(v) for v in rgb_delta],
        "receiver_contract_target": "frame1_region_waterfill_runtime_patch_v1",
        "runtime_consumption_proof_present": False,
        "inflated_output_change_proof_present": False,
        "blockers": [
            "runtime_consumption_proof_required_before_exact_eval",
            "inflated_output_change_proof_required_before_budget_spend_claim",
            "local_mlx_or_cpu_component_spot_check_required",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="frame1_region_waterfill_runtime_patch")
    return payload


__all__ = [
    "DISTORTION_BUDGET_ATTACK_PLAN_SCHEMA",
    "FRAME1_REGION_WATERFILL_RUNTIME_PATCH_SCHEMA",
    "P18_SEGNET_REGION_WATERFILL_SCHEMA",
    "P19_POSENET_NULL_PAIRS_SCHEMA",
    "ScorerRegionWaterfillError",
    "active_selector_mode_ids",
    "build_frame1_region_waterfill_runtime_patch",
    "build_p18_segnet_region_waterfill",
    "build_p19_posenet_null_pairs",
    "build_receiver_closed_distortion_budget_attack_plan",
    "decode_selector_codes",
]
