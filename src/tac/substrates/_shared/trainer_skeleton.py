# SPDX-License-Identifier: MIT
"""Canonical substrate-trainer utilities.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + the
2026-05-13 dedup audit (`feedback_canon_dedup_1_LANDED_20260513.md`), the
14 substrate trainers under ``experiments/train_substrate_*.py`` share a
small set of truly substrate-agnostic helpers. The dominant variants of
each helper are reproduced here as the canonical implementation.

Scope:
    Substrate-agnostic byte-faithful utilities only. Substrate-specific
    archive grammar, runtime emission, and architecture stay in each
    trainer (per HNeRV parity discipline lessons 3 + 4 + 5).

Compatibility:
    All 14 substrate trainers under ``experiments/train_substrate_*.py``
    import from this module (migration completed 2026-05-13).

Exception-type policy:
    Helper invariants (missing upstream frame_utils.py, pyav unavailable,
    insufficient decoded frames) raise plain ``RuntimeError`` rather than a
    dedicated ``SubstrateError`` subclass. Per R6-1 (2026-05-13 recursive
    review): there is no architectural reason to invent a dedicated
    exception class for canonical-helper invariants — these signal
    operator-environment misconfiguration (substrate's call site cannot
    recover and should fail loud). ``FileNotFoundError`` is used for
    missing files where the standard exception type fits exactly.
    Sibling ``tac.preflight`` raises ``PreflightError`` /
    ``MetaBugViolation`` because those are CI-gating violations with
    structured suppression semantics; this module's failures are
    environment-level. Cross-module catch should use ``Exception``
    rather than ``PreflightError`` when wrapping substrate trainer calls.

Cross-refs:
    - CLAUDE.md "Beauty, simplicity, and developer experience"
    - Catalog #146 (Phase 1 trainer runtime contract — substrate-specific
      ``_write_runtime`` stays per-trainer)
    - Catalog #164 (scorer preprocess — substrate score-aware loss stays
      per-trainer; the canonical helper is ``score_aware_common.py``)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

EVAL_HW: tuple[int, int] = (384, 512)

TRAINER_PROXY_AXIS_LABEL = "[trainer-proxy; not contest-CPU or contest-CUDA]"
TRAINER_PROXY_PROMOTION_REQUIREMENT = (
    "[contest-CUDA] auth eval required before score/rank/promotion claims"
)

OPTIMIZATION_FLAGS_MANIFEST: dict[str, dict[str, Any]] = {
    "--enable-autocast-fp16": {
        "env": "ENABLE_AUTOCAST_FP16",
        "default": False,
        "rationale": (
            "Catalog #172 Tier-1 speed path; wraps training forwards in "
            "torch.autocast without changing inflate/runtime semantics"
        ),
    },
    "--enable-torch-compile": {
        "env": "ENABLE_TORCH_COMPILE",
        "default": False,
        "rationale": (
            "Catalog #179 Tier-1 speed path; wraps substrate model with "
            "torch.compile via canonical fallback helper"
        ),
    },
    "--enable-gt-scorer-cache": {
        "env": "ENABLE_GT_SCORER_CACHE",
        "default": True,
        "rationale": (
            "F3/GTScorerCache Tier-1 speed path; precomputes frozen GT "
            "PoseNet+SegNet targets once and indexes them per batch"
        ),
    },
}


@dataclass(frozen=True)
class OptimizedTrainingContext:
    """Canonical Tier-1 optimization bundle for substrate trainer hot loops.

    The context is deliberately score-claim neutral: it may speed up trainer
    proxy losses, but it never changes the contest auth-eval axis. Trainers
    should continue to tag proxy losses as non-authoritative and require a
    byte-closed [contest-CUDA] auth eval before promotion language.
    """

    gt_cache: Any | None
    substrate_model: Any
    score_fn: Any
    autocast_cfg: Any
    eval_axis_label: str = TRAINER_PROXY_AXIS_LABEL
    promotion_requirement: str = TRAINER_PROXY_PROMOTION_REQUIREMENT


def merge_optimization_flags(
    trainer_tier_1_manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge canonical Tier-1 optimization flags into a trainer manifest.

    Trainer-specific entries intentionally win on key collision so a lane can
    tighten a rationale/default while still inheriting the canonical surface.
    """

    return {**OPTIMIZATION_FLAGS_MANIFEST, **trainer_tier_1_manifest}


def _resolve_scorer_pair(
    scorers: Any | None,
    *,
    seg_scorer: Any | None,
    pose_scorer: Any | None,
) -> tuple[Any, Any]:
    """Resolve ``(seg_scorer, pose_scorer)`` from common trainer shapes."""

    if seg_scorer is not None and pose_scorer is not None:
        return seg_scorer, pose_scorer
    if isinstance(scorers, dict):
        seg = scorers.get("seg_scorer") or scorers.get("segnet") or scorers.get("seg")
        pose = (
            scorers.get("pose_scorer")
            or scorers.get("posenet")
            or scorers.get("pose")
        )
        if seg is not None and pose is not None:
            return seg, pose
    if isinstance(scorers, (tuple, list)) and len(scorers) == 2:
        first, second = scorers
        first_name = type(first).__name__.lower()
        second_name = type(second).__name__.lower()
        if "seg" in first_name or "pose" in second_name:
            return first, second
        return second, first
    raise ValueError(
        "build_optimized_training_context needs scorers as "
        "(pose_scorer, seg_scorer), (seg_scorer, pose_scorer), a dict with "
        "seg/pose keys, or explicit seg_scorer= and pose_scorer=."
    )


def build_optimized_training_context(
    args: Any,
    scorers: Any | None = None,
    gt_pairs: Any | None = None,
    substrate_model: Any | None = None,
    device: Any | None = None,
    *,
    seg_scorer: Any | None = None,
    pose_scorer: Any | None = None,
    target_pixels: Any | None = None,
) -> OptimizedTrainingContext:
    """Build the canonical Tier-1 optimization context for trainers.

    ``gt_pairs``/``target_pixels`` must be shaped ``(N, 2, 3, H, W)`` when
    GT caching is enabled. The returned cache stays on CPU and is looked up
    by pair index inside trainer loops.
    """

    import functools

    import torch

    from tac.substrates.score_aware_common import score_pair_components_dispatch
    from tac.training_optimization import (
        AutocastConfig,
        build_gt_scorer_cache,
        compile_with_fallback,
        resolve_autocast_dtype,
    )

    if device is None:
        raise ValueError("build_optimized_training_context requires device")
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    seg, pose = _resolve_scorer_pair(
        scorers, seg_scorer=seg_scorer, pose_scorer=pose_scorer
    )
    target = target_pixels if target_pixels is not None else gt_pairs

    gt_cache = None
    if getattr(args, "enable_gt_scorer_cache", True):
        if target is None:
            raise ValueError(
                "enable_gt_scorer_cache=True requires gt_pairs/target_pixels"
            )
        gt_cache = build_gt_scorer_cache(
            target_pixels=target,
            posenet=pose,
            segnet=seg,
            device=resolved_device,
            segmentation_temperature=float(
                getattr(args, "segmentation_temperature", 1.0)
            ),
            cache_chunk_size=int(getattr(args, "gt_scorer_cache_chunk_size", 16)),
        )

    optimized_model = substrate_model
    if substrate_model is not None and getattr(args, "enable_torch_compile", False):
        optimized_model = compile_with_fallback(
            substrate_model,
            enabled=True,
            mode=str(getattr(args, "torch_compile_mode", "default")),
            fallback_on_error=bool(getattr(args, "torch_compile_fallback", True)),
            dynamic=getattr(args, "torch_compile_dynamic", None),
        )

    autocast_dtype = resolve_autocast_dtype(
        getattr(args, "autocast_dtype", "fp16")
    )
    autocast_cfg = AutocastConfig(
        enabled=bool(getattr(args, "enable_autocast_fp16", False)),
        dtype=autocast_dtype,
        device_type=resolved_device.type,
    )
    score_fn = functools.partial(
        score_pair_components_dispatch,
        seg_scorer=seg,
        pose_scorer=pose,
    )

    return OptimizedTrainingContext(
        gt_cache=gt_cache,
        substrate_model=optimized_model,
        score_fn=score_fn,
        autocast_cfg=autocast_cfg,
    )

# Canonical (axis, gpu_token) -> hardware_substrate map, aligned with
# `tac.continual_learning.TAG_HARDWARE_REQUIREMENT` accepted-substrate set.
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #127
# (`check_authoritative_tag_requires_custody_metadata`): the posterior write
# MUST record the actual GPU substrate the dispatch ran on, not a default
# placeholder. The 14 substrate trainers previously hardcoded
# `"linux_x86_64_t4"` regardless of the dispatched GPU — for A100/4090/H100
# dispatches that produces a silent custody mislabel.
_GPU_TOKEN_TO_SUBSTRATE: dict[str, str] = {
    "t4": "linux_x86_64_t4",
    "rtx_4090": "linux_x86_64_4090",
    "4090": "linux_x86_64_4090",
    "a100": "linux_x86_64_a100",
    "a100-40gb": "linux_x86_64_a100",
    "a100-80gb": "linux_x86_64_a100",
    "h100": "linux_x86_64_h100",
    "a10g": "linux_x86_64_a10g",
    "l40s": "linux_x86_64_l40s",
}


def pin_seeds(seed: int) -> None:
    """Deterministic seed pinning (torch + python + numpy if present)."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def torch_version_string() -> str:
    """Return torch version (or '<unknown>' on import failure)."""
    try:
        import torch

        return f"{torch.__version__}"
    except Exception:
        return "<unknown>"


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _clean_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, inner in value.items()
            for cleaned in (_clean_none(inner),)
            if cleaned is not None
        }
    if isinstance(value, list):
        return [cleaned for inner in value for cleaned in (_clean_none(inner),) if cleaned is not None]
    return value


def _mapping_or_empty(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def build_representation_training_probe_manifest(
    *,
    candidate_id: str,
    representation_family: str,
    substrate_family: str,
    schema: str = "representation_training_probe_manifest_v1",
    lane_id: str | None = None,
    lane_class: str | None = None,
    candidate_family: str | None = None,
    profile: str = "representation_training_probe",
    param_schema: str = "representation_training_manifest_params_v1",
    training_signal_kind: str = "local_representation_training_optimizer_schedule_probe",
    seed: int | None = None,
    device_requested: str | None = None,
    device_selected: str | None = None,
    source_tree_sha256: str | None = None,
    runtime_tree_sha256: str | None = None,
    output_dir: str | None = None,
    stages: list[Mapping[str, Any]] | None = None,
    results: list[Mapping[str, Any]] | None = None,
    stage_count: int | None = None,
    training_recipe: Mapping[str, Any] | None = None,
    optimizer_recipe: Mapping[str, Any] | None = None,
    scheduler_recipe: Mapping[str, Any] | None = None,
    candidate_params: Mapping[str, Any] | None = None,
    archive_zip: Mapping[str, Any] | None = None,
    auth_eval_bridge: Mapping[str, Any] | None = None,
    dispatch_blockers: list[str] | None = None,
    evidence_grade: str = "local_training_probe_advisory",
    source_anchor: str | None = None,
    score_lowering_hypothesis: str | None = None,
    variant_axes: list[str] | None = None,
    paired_modes: list[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic representation-training manifest with no score authority.

    The helper is intentionally trainer-agnostic. HNeRV, broader NeRV-family,
    SIREN, learned-codec, and future representation trainers can keep their
    substrate-specific provenance while also emitting one canonical sidecar
    consumed by candidate queues, learned sweeps, and cathedral consumers.
    """

    if not candidate_id.strip():
        raise ValueError("candidate_id is required")
    if not representation_family.strip():
        raise ValueError("representation_family is required")
    if not substrate_family.strip():
        raise ValueError("substrate_family is required")

    stage_rows = [dict(stage) for stage in (stages or [])]
    result_rows = [dict(result) for result in (results or [])]
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "dispatch_packet_ready": False,
    }
    manifest: dict[str, Any] = {
        "schema": schema,
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "lane_class": lane_class,
        "candidate_family": candidate_family,
        "representation_family": representation_family,
        "substrate_family": substrate_family,
        "profile": profile,
        "param_schema": param_schema,
        "training_signal_kind": training_signal_kind,
        "seed": seed,
        "device_requested": device_requested,
        "device_selected": device_selected,
        "source_tree_sha256": source_tree_sha256,
        "runtime_tree_sha256": runtime_tree_sha256,
        "output_dir": output_dir,
        "stage_count": stage_count if stage_count is not None else len(stage_rows),
        "stages": stage_rows,
        "results": result_rows,
        "training_recipe": _mapping_or_empty(training_recipe),
        "optimizer_recipe": _mapping_or_empty(optimizer_recipe),
        "scheduler_recipe": _mapping_or_empty(scheduler_recipe),
        "candidate_params": _mapping_or_empty(candidate_params),
        "archive_zip": _mapping_or_empty(archive_zip),
        "auth_eval_bridge": {
            **_mapping_or_empty(auth_eval_bridge),
            **{
                key: False
                for key in false_authority
                if key not in _mapping_or_empty(auth_eval_bridge)
            },
        },
        "dispatch_blockers": list(dispatch_blockers or []),
        "evidence_grade": evidence_grade,
        "source_anchor": source_anchor,
        "score_lowering_hypothesis": score_lowering_hypothesis,
        "variant_axes": list(variant_axes or []),
        "paired_modes": list(paired_modes or []),
        **false_authority,
    }
    if extra_fields:
        for key, value in extra_fields.items():
            if key in false_authority and value is not False:
                raise ValueError(f"{key} must remain false in representation training manifests")
            manifest[str(key)] = value

    return _clean_none(manifest)


def write_representation_training_probe_manifest(
    path: Path,
    *,
    validate: bool = True,
    **manifest_kwargs: Any,
) -> dict[str, Any]:
    """Write a canonical representation-training manifest sidecar as JSON."""

    manifest = build_representation_training_probe_manifest(**manifest_kwargs)
    if validate:
        from tac.optimization.representation_training_probe_integration import (
            validate_representation_training_manifest,
        )

        validate_representation_training_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def git_head_sha(repo_root: Path | None = None) -> str:
    """Return git HEAD sha (or '<unknown>' on failure)."""
    root = repo_root if repo_root is not None else REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "<unknown>"


def utc_now_iso() -> str:
    """UTC ISO-8601 timestamp suitable for provenance/stage logs."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def vendor_shared_inflate_runtime(
    submission_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Copy the shared raw-output inflate helper into a submission runtime tree.

    Substrate archives vendor only the minimal inflate-time package. Any
    substrate runtime importing ``tac.substrates._shared.inflate_runtime`` must
    call this helper from its trainer ``_write_runtime`` path, otherwise local
    source-tree tests pass while contest auth eval fails with an import error.

    OVERNIGHT-GG vendor-stub fix 2026-05-21: explicit ``os.utime(dst, None)``
    after the copy refreshes the destination mtime to the current time. This
    is a defense-in-depth against any mtime-based harvester filter (sister of
    the META fix in ``experiments/modal_train_lane.py`` that bypasses the
    mtime_floor for ``output/submission/`` paths). ``shutil.copy2`` preserves
    the source's mtime, which is the old local-repo mtime carried through
    Modal's ``copytree(symlinks=True)`` mount staging. The OVERNIGHT-CC
    99d06f967 incident proved this is a real failure mode: 8 vendored .py
    modules dropped by harvester → ModuleNotFoundError on Modal worker re-fire.
    """

    root = repo_root if repo_root is not None else REPO_ROOT
    shared_src = root / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py"
    if not shared_src.is_file():
        raise FileNotFoundError(f"shared inflate runtime helper missing: {shared_src}")
    shared_dst = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dst.mkdir(parents=True, exist_ok=True)
    (shared_dst / "__init__.py").write_text("", encoding="utf-8")
    dst_file = shared_dst / "inflate_runtime.py"
    shutil.copy2(shared_src, dst_file)
    # OVERNIGHT-GG defense-in-depth: refresh mtime to current time so
    # mtime-based harvester filters cannot silently drop the vendored body.
    os.utime(dst_file, None)


def vendor_module_with_fresh_mtime(
    src_path: Path,
    dst_path: Path,
) -> None:
    """Copy a substrate module body with current-time mtime stamping.

    Canonical helper for substrate trainer ``_write_runtime`` paths that
    vendor module bodies via ``shutil.copy2``. The default ``shutil.copy2``
    preserves the source's mtime, which can collide with the Modal
    harvester's ``mtime_floor`` filter (set at lane start; source files
    from ``copytree(symlinks=True)`` retain old local-repo mtimes). This
    helper combines ``shutil.copy2`` (data + metadata) with explicit
    ``os.utime(dst, None)`` to ensure the destination's mtime reflects
    the vendor-emission time, making it robust against the
    ``output/submission/`` harvester filter regardless of whether the
    META layer fix (Catalog #360 in ``experiments/modal_train_lane.py``)
    is in place.

    Source: OVERNIGHT-GG bug class fix 2026-05-21 per OVERNIGHT-CC
    99d06f967 IMPLEMENTATION-LEVEL falsification (Catalog #307).
    """

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    os.utime(dst_path, None)


def detect_hardware_substrate(
    *,
    axis: str = "cuda",
    substrate_tag: str,
    provenance_path: Path | None = None,
    env_var_candidates: tuple[str, ...] = (),
) -> str:
    """Resolve the canonical ``hardware_substrate`` token for posterior writes.

    Per CLAUDE.md SIREN audit (2026-05-13) CRITICAL #1 + Catalog #190
    (``check_substrate_trainer_does_not_hardcode_hardware_substrate``): the
    14 substrate trainers used to hardcode ``"linux_x86_64_t4"`` regardless
    of the dispatched GPU. This helper resolves the substrate dynamically
    from (1) the remote driver's ``provenance.json``, (2) environment-var
    candidates (typically ``<SUBSTRATE>_GPU`` then ``MODAL_GPU``), or
    (3) live ``nvidia-smi`` query. Falls back to
    ``"linux_x86_64_unknown_cuda"`` with a stderr-warning if all sources
    are silent — never silently mislabels.

    Args:
        axis: ``"cuda"`` (default) or ``"cpu"``. Drives the lookup table.
        substrate_tag: Short label used in the warning banner (e.g. ``"siren"``).
        provenance_path: Optional path to a substrate-emitted
            ``provenance.json`` carrying a ``gpu_name`` field (typically from
            ``scripts/remote_lane_substrate_<id>.sh``).
        env_var_candidates: Ordered tuple of env var names to consult in
            priority order (e.g. ``("SIREN_GPU", "MODAL_GPU")``).

    Returns:
        Canonical substrate token from
        ``tac.continual_learning.TAG_HARDWARE_REQUIREMENT``: e.g.
        ``"linux_x86_64_a100"``, ``"linux_x86_64_t4"``, ``"linux_x86_64_4090"``,
        ``"linux_x86_64_unknown_cuda"`` (fallback). For ``axis="cpu"``,
        returns ``"linux_x86_64_modal_cpu"`` (Linux x86_64 non-GHA) or
        ``"unknown_cpu"``.
    """
    if axis == "cpu":
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Darwin":
            return "macos_arm64" if machine in {"arm64", "aarch64"} else "macos_x86_64"
        if system == "Linux" and machine in {"x86_64", "amd64"}:
            if any(name.startswith("MODAL_") for name in os.environ):
                return "linux_x86_64_modal_cpu"
            if os.environ.get("GITHUB_ACTIONS"):
                return "linux_x86_64_gha_cpu"
            if os.environ.get("VAST_CONTAINERLABEL") or os.environ.get("VASTAI_INSTANCE_ID"):
                return "linux_x86_64_vast_cpu"
            if os.environ.get("LIGHTNING_CLOUD_PROJECT_ID") or os.environ.get("LIGHTNING_USER_ID"):
                return "linux_x86_64_lightning_cpu"
            return "linux_x86_64_cpu"
        return "unknown_cpu"
    if axis != "cuda":
        return "unknown"

    gpu_token = ""

    # (1) Prefer provenance.json (remote driver writes the actual GPU name).
    if provenance_path is not None:
        try:
            if provenance_path.is_file():
                prov = json.loads(provenance_path.read_text())
                gpu_name = str(prov.get("gpu_name") or "").strip().lower()
                gpu_token = gpu_name
        except Exception:
            gpu_token = ""

    # (2) Environment-var ladder (operator wrapper / Modal env_overrides).
    if not gpu_token:
        for env_name in env_var_candidates:
            value = os.environ.get(env_name)
            if value:
                gpu_token = value.strip().lower()
                break

    # (3) Live nvidia-smi probe (CUDA-only).
    if not gpu_token:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_token = proc.stdout.strip().splitlines()[0].lower()
        except Exception:
            gpu_token = ""

    # Map the token to a canonical substrate.
    for key, substrate in _GPU_TOKEN_TO_SUBSTRATE.items():
        if key in gpu_token:
            return substrate

    # All sources silent or unrecognized GPU.
    import sys as _sys

    print(
        f"[{substrate_tag}] WARN: hardware_substrate detection found no GPU "
        f"token (provenance={provenance_path}, env_candidates={env_var_candidates}, "
        f"resolved={gpu_token!r}); falling back to 'linux_x86_64_unknown_cuda'. "
        "Posterior write will record this fallback explicitly per CLAUDE.md "
        "forbidden-empirical-claim-without-evidence-tag discipline.",
        file=_sys.stderr,
    )
    return "linux_x86_64_unknown_cuda"


class StageLog:
    """Append-only stage tracker for provenance ``stage_log`` blocks.

    Each substrate trainer's ``_full_main`` builds a stage_log dict-list
    consumed by the provenance.json writer. This helper canonicalizes the
    pattern.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def stage(self, name: str) -> None:
        """Append a stage marker keyed to the current UTC time."""
        self._entries.append({"stage": name, "at": utc_now_iso()})

    def entries(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the recorded stage entries."""
        return list(self._entries)


def device_or_die(
    name: str,
    *,
    smoke: bool,
    substrate_tag: str,
    allow_full_cpu: bool = False,
):
    """Resolve compute device or raise SystemExit.

    Args:
        name: One of {'cuda', 'cpu'}.
        smoke: True iff this is the smoke path (CPU permitted).
        substrate_tag: Short label used in error messages (e.g. 'cool_chic').
        allow_full_cpu: Explicit advisory-only exception for trainers that
            implement a coupled ``--full-cpu`` + waiver path.

    Per CLAUDE.md "MPS auth eval is NOISE" + "EMA — non-negotiable":
    cuda is the default for full training, cpu is permitted only with
    --smoke or an explicit trainer-owned advisory exception, mps is FORBIDDEN.
    """
    import torch

    if name == "cpu":
        if not smoke and not allow_full_cpu:
            raise SystemExit(
                f"[{substrate_tag}] --device cpu is permitted only with "
                "--smoke per CLAUDE.md 'MPS auth eval is NOISE' + 'EMA — "
                "non-negotiable' + full-training-needs-CUDA convention. "
                "Use --device cuda for promotion-grade training. CPU smoke is "
                "allowed only when deterministic-bytes acceptable."
            )
        if allow_full_cpu and smoke:
            raise SystemExit(
                f"[{substrate_tag}] allow_full_cpu=True is only valid for "
                "full advisory runs, not smoke."
            )
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                f"[{substrate_tag}] --device cuda requested but cuda not available"
            )
        # Canonical substrate-trainer fast-math policy. Catalog #178 forbids
        # each trainer from rediscovering or silently omitting this Ampere/
        # Hopper speed path. TF32 affects CUDA matmul/convolution kernels only;
        # exact score authority still comes from archive/runtime auth eval.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    raise SystemExit(f"[{substrate_tag}] unknown --device {name!r}")


def require_contest_cuda_auth_eval_claim(
    auth_eval_result_path: Path,
    *,
    archive_sha256: str,
    substrate_tag: str,
) -> tuple[Any, dict[str, Any]]:
    """Load an auth-eval JSON file and require a real contest-CUDA score claim.

    This is the canonical substrate-trainer boundary between "auth eval
    produced some finite diagnostics" and "this run may update the CUDA-axis
    posterior." A finite component-coherent score is insufficient: the JSON
    must also carry the contest-CUDA custody fields checked by
    ``parse_auth_eval_score_claim``.
    """

    if not auth_eval_result_path.is_file():
        raise RuntimeError(
            f"[{substrate_tag}] auth eval JSON missing: {auth_eval_result_path}"
        )
    try:
        payload = json.loads(auth_eval_result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"[{substrate_tag}] could not parse auth eval JSON "
            f"{auth_eval_result_path}: {exc}"
        ) from exc

    from tac.auth_eval_result import (
        parse_auth_eval_score_claim,
        parse_finite_auth_eval_score,
    )

    parsed_score = parse_finite_auth_eval_score(
        payload,
        require_component_recompute=True,
    )
    if parsed_score is None:
        raise RuntimeError(
            f"[{substrate_tag}] auth eval JSON lacks a finite, "
            "component-coherent score; refusing contest-CUDA claim."
        )
    claim = parse_auth_eval_score_claim(
        payload,
        required_score_axis="contest_cuda",
        require_component_recompute=True,
    )
    if claim is None:
        raise RuntimeError(
            f"[{substrate_tag}] auth eval score is finite but is not a "
            "valid [contest-CUDA] claim "
            f"(score_axis={payload.get('score_axis')!r}, "
            f"score_claim={payload.get('score_claim')!r}, "
            f"score_claim_valid={payload.get('score_claim_valid')!r}, "
            f"exact_cuda_eval_complete={payload.get('exact_cuda_eval_complete')!r}, "
            f"evidence_grade={payload.get('evidence_grade')!r}, "
            f"archive_sha256={archive_sha256})."
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    scored_archive_sha256 = payload.get("archive_sha256") or provenance.get(
        "archive_sha256"
    )
    if archive_sha256 and scored_archive_sha256 and scored_archive_sha256 != archive_sha256:
        raise RuntimeError(
            f"[{substrate_tag}] auth eval scored archive SHA "
            f"{scored_archive_sha256} but trainer expected {archive_sha256}; "
            "refusing to attach a contest-CUDA claim to the wrong scored object."
        )
    return claim, payload


def load_upstream_yuv420_to_rgb(*, substrate_tag: str, repo_root: Path | None = None):
    """Load upstream/frame_utils.py's ``yuv420_to_rgb`` without patching upstream.

    Per CLAUDE.md "Non-Negotiable Upstream Rule": upstream is the source of
    truth; we re-use the canonical contest-faithful decode path (BT.601 /
    no in-place ops) without modifying upstream files.

    Args:
        substrate_tag: Short label used for the importlib spec name to
            avoid collisions when multiple substrates share a process.
        repo_root: Optional override; defaults to repo root.
    """
    import importlib.util

    root = repo_root if repo_root is not None else REPO_ROOT
    frame_utils_path = root / "upstream" / "frame_utils.py"
    if not frame_utils_path.is_file():
        raise FileNotFoundError(
            f"upstream/frame_utils.py not found at {frame_utils_path}; "
            "verify --upstream-dir is correct."
        )
    spec = importlib.util.spec_from_file_location(
        f"pact_{substrate_tag}_upstream_frame_utils", frame_utils_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"unable to load upstream frame_utils.py from {frame_utils_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    substrate_tag: str,
    max_pairs: int | None = None,
    repo_root: Path | None = None,
):
    """Decode real contest pairs (0,1), (2,3), ... at EVAL_HW (384, 512).

    Returns:
        torch.Tensor shape ``(N, 2, 3, 384, 512)`` float32 in ``[0, 255]``.

    Raises:
        FileNotFoundError: ``video_path`` is missing.
        RuntimeError: pyav not installed, or video yielded fewer frames
            than ``n_pairs * 2`` (with ``max_pairs`` accounted for).

    Per CLAUDE.md Catalog #114 + the HNeRV parity lesson L1 (score-aware
    substrate trains against real contest video, NOT synthetic data).
    """
    import torch
    import torch.nn.functional as F

    if not video_path.is_file():
        raise FileNotFoundError(
            f"real target video not found: {video_path}. Non-smoke "
            "training requires upstream/videos/0.mkv."
        )
    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            f"pyav (`av`) is required for non-smoke {substrate_tag} "
            "training; run `uv pip install av`"
        ) from exc

    yuv420_to_rgb = load_upstream_yuv420_to_rgb(
        substrate_tag=substrate_tag, repo_root=repo_root
    )
    target_pairs = n_pairs if max_pairs is None else min(n_pairs, max_pairs)
    frames_needed = target_pairs * 2
    frames_chw: list = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb_hwc = yuv420_to_rgb(frame)
            rgb_chw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(
                rgb_chw, size=EVAL_HW, mode="bilinear", align_corners=False
            )
            frames_chw.append(resized.squeeze(0).contiguous())
            if len(frames_chw) >= frames_needed:
                break
    finally:
        container.close()
    if len(frames_chw) < frames_needed:
        raise RuntimeError(
            f"{video_path} yielded {len(frames_chw)} frame(s), "
            f"need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


__all__ = [
    "EVAL_HW",
    "OPTIMIZATION_FLAGS_MANIFEST",
    "REPO_ROOT",
    "TRAINER_PROXY_AXIS_LABEL",
    "TRAINER_PROXY_PROMOTION_REQUIREMENT",
    "OptimizedTrainingContext",
    "StageLog",
    "build_optimized_training_context",
    "build_representation_training_probe_manifest",
    "decode_real_pairs",
    "detect_hardware_substrate",
    "device_or_die",
    "git_head_sha",
    "load_upstream_yuv420_to_rgb",
    "merge_optimization_flags",
    "pin_seeds",
    "require_contest_cuda_auth_eval_claim",
    "sha256_bytes",
    "torch_version_string",
    "utc_now_iso",
    "write_representation_training_probe_manifest",
]
