# SPDX-License-Identifier: MIT
"""Fail-closed pivot audit for compact VQ NeRV carrier rows."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

COMPACT_VQ_PIVOT_AUDIT_SCHEMA = "compact_vq_pivot_audit.v1"
COMPACT_VQ_MISMATCH_STATUS = "pivot_or_rebuild_vq_before_more_long_run_spend"
COMPACT_VQ_RETAIN_STATUS = "retain_vq_residual_token_lane_after_rebuild"
HPRC_MLX_COMPONENT_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY = 0.5

_FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def build_compact_vq_pivot_audit(
    *,
    repo_root: str | Path,
    upstream_dir: str | Path,
    mlx_profile_paths: list[str | Path] | tuple[str | Path, ...] = (),
    family: str = "pact_nerv_vq",
    max_mlx_score_for_local_replay: float = DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY,
) -> dict[str, Any]:
    """Build an executable audit for whether compact VQ deserves more spend.

    The audit separates two facts that were getting blurred:

    * VQ/RT-NeRV research supports residual tokenization of shallow/inter-frame
      features with codebook-utilization repair.
    * The local PACT/VQ implementation is a per-pair latent codebook carrier.

    If the local row is also terrible under full-video MLX scorer replay, the
    durable next action is to pivot spend toward PR95/HiNeRV/SNeRV or rebuild
    VQ as a residual-token bolt-on.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    upstream = Path(upstream_dir).expanduser().resolve(strict=False)
    scorer_contract = _upstream_scorer_contract(upstream)
    impl = _implementation_contract(root)
    profiles = [_load_profile(path, root=root) for path in mlx_profile_paths]
    profile_signal = _profile_signal(
        profiles,
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )

    residual_token_ready = (
        impl["residual_tokenization_present"]
        and impl["shallow_interframe_feature_path_present"]
        and impl["codebook_utilization_repair_present"]
    )
    terrible_full_video_score = (
        profile_signal["best_full_video_mlx_score"] is not None
        and float(profile_signal["best_full_video_mlx_score"])
        >= float(max_mlx_score_for_local_replay)
    )
    if not residual_token_ready or terrible_full_video_score:
        verdict = COMPACT_VQ_MISMATCH_STATUS
    else:
        verdict = COMPACT_VQ_RETAIN_STATUS

    blockers: list[str] = ["contest_cpu_cuda_exact_eval_not_executed"]
    if not residual_token_ready:
        blockers.append("compact_vq_is_per_pair_latent_not_residual_tokenization")
        if not impl["shallow_interframe_feature_path_present"]:
            blockers.append("compact_vq_shallow_interframe_feature_path_missing")
        if not impl["codebook_utilization_repair_present"]:
            blockers.append("compact_vq_codebook_utilization_repair_missing")
    if not profile_signal["has_full_video_mlx_profile"]:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    elif terrible_full_video_score:
        blockers.append("full_video_mlx_score_above_local_replay_threshold")

    return {
        "schema": COMPACT_VQ_PIVOT_AUDIT_SCHEMA,
        "family": family,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "upstream_dir": upstream.as_posix(),
        "scorer_contract": scorer_contract,
        "implementation_contract": impl,
        "profile_signal": profile_signal,
        "research_basis": {
            "schema": "compact_vq_research_basis.v1",
            "rt_nerv_arxiv": "https://arxiv.org/abs/2403.12401",
            "rt_nerv_requirement": (
                "residual tokenization of shallow/inter-frame features, "
                "residual-aware codebook learning, and token utilization repair"
            ),
            "hinerv_arxiv": "https://arxiv.org/abs/2306.09818",
            "hinerv_requirement": (
                "hierarchical representation plus pruning/quantization codec "
                "pipeline, not only a tiny unfit decoder"
            ),
            "snerv_arxiv": "https://arxiv.org/abs/2501.01681",
            "snerv_requirement": (
                "frequency split with LF carriage and learned HF restoration; "
                "useful if score-aware decoder fitting solves distortion"
            ),
        },
        "verdict": verdict,
        "spend_recommendation": (
            "route_compact_training_budget_to_pr95_hinerv_snerv_stage8_or_"
            "rebuild_vq_as_rt_residual_token_bolton"
            if verdict == COMPACT_VQ_MISMATCH_STATUS
            else "retain_compact_vq_for_receiver_proven_full_video_replay"
        ),
        "next_actions": [
            (
                "stop promoting per-pair latent VQ as a primary carrier until "
                "it implements residual-tokenized shallow/inter-frame features"
            ),
            (
                "prioritize PR95 Stage-8 faithful continuation and HiNeRV/SNeRV "
                "score-aware decoder-weight fitting under archive byte ceilings"
            ),
            (
                "if VQ is retained, make it a residual-token bolt-on with "
                "codebook-utilization metrics and full-video scorer replay gates"
            ),
        ],
        "blockers": _dedupe(blockers),
        **_FALSE_AUTHORITY,
    }


def write_compact_vq_pivot_audit(
    *,
    output_path: str | Path,
    audit: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _upstream_scorer_contract(upstream: Path) -> dict[str, Any]:
    evaluate = upstream / "evaluate.py"
    modules = upstream / "modules.py"
    frame_utils = upstream / "frame_utils.py"
    constants = _frame_utils_constants(frame_utils)
    evaluate_text = evaluate.read_text(encoding="utf-8")
    modules_text = modules.read_text(encoding="utf-8")
    return {
        "schema": "upstream_contest_eval_contract.v1",
        "evaluate_py_path": evaluate.as_posix(),
        "evaluate_py_sha256": _sha256_file(evaluate),
        "modules_py_path": modules.as_posix(),
        "modules_py_sha256": _sha256_file(modules),
        "frame_utils_py_path": frame_utils.as_posix(),
        "frame_utils_py_sha256": _sha256_file(frame_utils),
        "seq_len": constants.get("seq_len"),
        "camera_size": constants.get("camera_size"),
        "segnet_model_input_size": constants.get("segnet_model_input_size"),
        "score_formula_observed": (
            "100 * segnet_dist" in evaluate_text
            and "math.sqrt(posenet_dist * 10)" in evaluate_text
            and "25 * rate" in evaluate_text
        ),
        "segnet_last_frame_only_observed": "x = x[:, -1, ...]" in modules_text,
        "posenet_yuv6_pair_input_observed": (
            "rgb_to_yuv6" in modules_text and "b (t c) h w" in modules_text
        ),
        "pose_first_six_dims_observed": "[..., : h.out // 2]" in modules_text,
        "rate_authority": "archive.zip_bytes_over_uncompressed_video_bytes",
    }


def _implementation_contract(root: Path) -> dict[str, Any]:
    architecture = root / "src/tac/substrates/pact_nerv_vq/architecture.py"
    mlx_renderer = root / "src/tac/substrates/pact_nerv_vq/mlx_renderer.py"
    archive = root / "src/tac/substrates/pact_nerv_vq/archive.py"
    arch_text = architecture.read_text(encoding="utf-8")
    mlx_text = mlx_renderer.read_text(encoding="utf-8")
    archive_text = archive.read_text(encoding="utf-8")
    all_text = "\n".join([arch_text, mlx_text, archive_text]).lower()
    return {
        "schema": "compact_vq_implementation_contract.v1",
        "architecture_py_path": architecture.as_posix(),
        "architecture_py_sha256": _sha256_file(architecture),
        "mlx_renderer_py_path": mlx_renderer.as_posix(),
        "mlx_renderer_py_sha256": _sha256_file(mlx_renderer),
        "archive_py_path": archive.as_posix(),
        "archive_py_sha256": _sha256_file(archive),
        "per_pair_single_vector_vq_present": (
            "torch.empty(cfg.num_pairs, cfg.latent_dim)" in arch_text
            and "z_e.dim() != 2" in arch_text
            and "(num_pairs,) uint16 codebook indices" in archive_text
        ),
        "archive_ships_codebook_and_indices": (
            "CODEBOOK_BLOB_LEN" in archive_text and "INDICES_BLOB_LEN" in archive_text
        ),
        "residual_tokenization_present": (
            "residual tokenizer" in all_text or "residual_token" in all_text
        ),
        "shallow_interframe_feature_path_present": (
            ("shallow" in all_text and "inter-frame" in all_text)
            or ("shallow" in all_text and "inter_frame" in all_text)
        ),
        "codebook_utilization_repair_present": (
            "dead code" in all_text
            or "dead_code" in all_text
            or "codebook utilization" in all_text
            or "k-means" in all_text
            or "kmeans" in all_text
        ),
        "current_vq_role": (
            "primary_pair_latent_carrier"
            if "torch.empty(cfg.num_pairs, cfg.latent_dim)" in arch_text
            else "unknown"
        ),
    }


def _profile_signal(
    profiles: list[dict[str, Any]],
    *,
    max_mlx_score_for_local_replay: float,
) -> dict[str, Any]:
    full_profiles = [
        profile
        for profile in profiles
        if _profile_has_full_video_coverage(profile["payload"])
    ]
    scores = [
        score
        for profile in full_profiles
        if (score := _profile_score_estimate_from_loaded_profile(profile)) is not None
    ]
    best_score = min(scores) if scores else None
    return {
        "schema": "compact_vq_profile_signal.v1",
        "profile_count": len(profiles),
        "has_full_video_mlx_profile": bool(full_profiles),
        "full_video_profile_paths": [profile["path"] for profile in full_profiles],
        "best_full_video_mlx_score": best_score,
        "max_mlx_score_for_local_replay": float(max_mlx_score_for_local_replay),
        "local_replay_threshold_passed": (
            None if best_score is None else best_score < max_mlx_score_for_local_replay
        ),
    }


def _load_profile(path: str | Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile JSON must be an object: {resolved}")
    return {"path": resolved.as_posix(), "sha256": _sha256_file(resolved), "payload": payload}


def _profile_has_full_video_coverage(profile: dict[str, Any]) -> bool:
    return (
        profile.get("schema") == HPRC_MLX_COMPONENT_PROFILE_SCHEMA
        and _profile_full_video_scope(profile) == "executed"
        and int(_profile_pair_count(profile) or 0) >= 600
        and _profile_batch_pairs(profile) == 1
    )


def _profile_pair_count(profile: dict[str, Any]) -> int | None:
    counts: list[int] = []
    for container in _profile_containers(profile):
        for key in (
            "max_pairs",
            "num_pairs",
            "n_samples",
            "candidate_cache_pairs",
            "reference_cache_pairs",
        ):
            value = _nonnegative_int(container.get(key))
            if value is not None:
                counts.append(value)
    return max(counts) if counts else None


def _profile_batch_pairs(profile: dict[str, Any]) -> int | None:
    for container in _profile_containers(profile):
        for key in ("scorer_batch_pairs", "batch_pairs"):
            value = _nonnegative_int(container.get(key))
            if value is not None:
                return value
    return None


def _profile_full_video_scope(profile: dict[str, Any]) -> str:
    scope = profile.get("scope_status")
    if not isinstance(scope, dict):
        return "missing_scope_status"
    marker = scope.get("full_video")
    if marker == "executed" or marker is True:
        return "executed"
    if isinstance(marker, str) and marker:
        return marker
    return "missing_full_video_scope"


def _profile_score_estimate(profile: dict[str, Any]) -> float | None:
    for container in _profile_containers(profile):
        for key in (
            "canonical_score",
            "recomputed_total_score",
            "score_recomputed_from_components",
            "local_score_estimate",
        ):
            value = _finite_float(container.get(key))
            if value is not None:
                return value
    return None


def _profile_score_estimate_from_loaded_profile(profile: dict[str, Any]) -> float | None:
    payload = profile["payload"]
    direct = _profile_score_estimate(payload)
    if direct is not None:
        return direct
    profile_path = Path(profile["path"])
    for row in payload.get("variant_rows") or []:
        if not isinstance(row, dict) or row.get("variant_id") != "baseline":
            continue
        response_path = row.get("mlx_response")
        if not isinstance(response_path, str) or not response_path:
            continue
        resolved = _resolve(response_path, base=profile_path.parent)
        if not resolved.is_file():
            continue
        try:
            response = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(response, dict):
            score = _profile_score_estimate(response)
            if score is not None:
                return score
    return None


def _profile_containers(profile: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = [profile]
    for key in ("score_components", "mlx_response_summary", "response_metadata"):
        value = profile.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        out = float(value)
        if out == out and out not in (float("inf"), float("-inf")):
            return out
    return None


def _frame_utils_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"seq_len", "camera_size", "segnet_model_input_size"}:
            continue
        constants[target.id] = ast.literal_eval(node.value)
    return constants


def _resolve(path: str | Path, *, base: Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else (base / raw).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _utc_stamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
