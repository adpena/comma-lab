# SPDX-License-Identifier: MIT
"""Boyd-style dispatch feasibility protocol.

The dispatch protocol is the runtime umbrella over many narrower preflight
gates. Individual catalog checks can prove one fact, but a paid dispatch needs
the conjunction:

``dispatch_protocol_complete = tier1_engineering
                               AND tier2_hardware_correctness
                               AND tier3_substrate_correctness``

This module is intentionally lightweight and side-effect free so provider
actuators can call it immediately before lane-claim creation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DispatchProtocolError",
    "DispatchProtocolReport",
    "DispatchProtocolTierReport",
    "evaluate_dispatch_protocol_complete",
    "require_dispatch_protocol_complete",
]

DISPATCH_PROTOCOL_SCHEMA = "pact.dispatch_protocol_complete.v1"

LEGAL_NATIVE_PLATFORMS = frozenset({"modal", "vastai", "vast", "local"})
LEGAL_VIDEO_INPUT_STRATEGIES = frozenset(
    {"per_dispatch_local_copy", "readonly_mmap", "shared_volume_no_contention_expected"}
)
LEGAL_PYAV_DECODE_STRATEGIES = frozenset(
    {"cpu_thread_async_upload", "cuda_nvdec", "cpu_blocking_upload", "not_applicable"}
)
LEGAL_TARGET_MODES = frozenset(
    {
        "contest_one_video_replay",
        "contest_generalized",
        "production_generalized",
        "production_edge_adaptive",
        "research_substrate",
    }
)
LEGAL_CANARY_STATUS = frozenset(
    {"canary", "post_canary_dependent", "independent_substrate"}
)
LEGAL_GPU_ORDER = {
    "cpu": -1,
    "T4": 0,
    "L4": 1,
    "A10G": 2,
    "L40S": 3,
    "A100": 4,
    "H100": 5,
}
REQUIRED_MODAL_ENV_TOKENS = (
    "DALI_DISABLE_NVML",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
)


class DispatchProtocolError(RuntimeError):
    """Raised when a native dispatch attempts to fire with an empty feasibility intersection."""


@dataclass(frozen=True)
class DispatchProtocolTierReport:
    """One tier in the dispatch feasibility conjunction."""

    name: str
    passed: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class DispatchProtocolReport:
    """Structured result for one recipe/trainer dispatchability check."""

    recipe_name: str
    native_dispatch: bool
    dispatch_protocol_complete: bool
    tiers: tuple[DispatchProtocolTierReport, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        out: list[str] = []
        for tier in self.tiers:
            out.extend(f"{tier.name}:{blocker}" for blocker in tier.blockers)
        return tuple(out)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DISPATCH_PROTOCOL_SCHEMA,
            "recipe_name": self.recipe_name,
            "native_dispatch": self.native_dispatch,
            "dispatch_protocol_complete": self.dispatch_protocol_complete,
            "tiers": [tier.to_dict() for tier in self.tiers],
            "blockers": list(self.blockers),
        }


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if not text or text.startswith("${") or "://" in text:
        return None
    return text


def _resolve_repo_path(repo_root: Path, value: Any) -> Path | None:
    text = _path_value(value)
    if text is None:
        return None
    path = Path(text)
    return path if path.is_absolute() else repo_root / path


def _resolve_env_fallback(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(.+)\}", text)
    if match:
        return match.group(1)
    return text


def _read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _recipe_name(raw_recipe: Mapping[str, Any], recipe_path: Path | None) -> str:
    name = str(raw_recipe.get("name") or "").strip()
    if name:
        return name
    if recipe_path is not None:
        return recipe_path.stem
    return "<unknown>"


def _remote_driver_path(
    raw_recipe: Mapping[str, Any],
    repo_root: Path,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        return _resolve_repo_path(repo_root, explicit)
    modal_cfg = raw_recipe.get("modal", {}) or {}
    modal_lane = modal_cfg.get("lane_script") if isinstance(modal_cfg, Mapping) else None
    return _resolve_repo_path(repo_root, modal_lane or raw_recipe.get("remote_driver"))


def _trainer_path(
    raw_recipe: Mapping[str, Any],
    repo_root: Path,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        return _resolve_repo_path(repo_root, explicit)
    modal_cfg = raw_recipe.get("modal", {}) or {}
    modal_trainer = (
        modal_cfg.get("cost_band_trainer") if isinstance(modal_cfg, Mapping) else None
    )
    return _resolve_repo_path(
        repo_root, modal_trainer or raw_recipe.get("required_input_files_trainer")
    )


def _tier(name: str, blockers: Sequence[str]) -> DispatchProtocolTierReport:
    return DispatchProtocolTierReport(
        name=name,
        passed=not blockers,
        blockers=tuple(blockers),
    )


def _tier1_engineering(
    raw_recipe: Mapping[str, Any],
    *,
    platform: str,
    remote_driver: Path | None,
    trainer: Path | None,
) -> DispatchProtocolTierReport:
    blockers: list[str] = []
    if not _as_bool(raw_recipe.get("dispatch_enabled"), default=True):
        blockers.append("dispatch_enabled_false")
    for field in ("dispatch_blockers", "pre_promotion_blockers"):
        values = [str(item) for item in _as_list(raw_recipe.get(field)) if str(item)]
        if values:
            blockers.append(f"{field}_present:{','.join(values[:4])}")
    lane_id = str(raw_recipe.get("lane_id") or "")
    if not re.fullmatch(r"lane_[a-z0-9_]+_\d{8}", lane_id):
        blockers.append("lane_id_missing_or_noncanonical")
    if platform not in LEGAL_NATIVE_PLATFORMS:
        blockers.append(f"native_platform_not_supported:{platform or '<missing>'}")
    cost_band = raw_recipe.get("cost_band", {}) or {}
    epochs = _as_int(cost_band.get("epochs") if isinstance(cost_band, Mapping) else None)
    if epochs is None or epochs <= 0:
        blockers.append("cost_band_epochs_missing_or_nonpositive")
    if remote_driver is None or not remote_driver.is_file():
        blockers.append("remote_driver_missing")
    if trainer is None or not trainer.is_file():
        blockers.append("required_input_files_trainer_missing")
    return _tier("tier1_engineering", blockers)


def _tier2_hardware_correctness(
    raw_recipe: Mapping[str, Any],
    *,
    platform: str,
    repo_root: Path,
    remote_driver: Path | None,
) -> DispatchProtocolTierReport:
    blockers: list[str] = []
    min_vram = _as_int(raw_recipe.get("min_vram_gb"))
    if min_vram is None or min_vram < 1:
        blockers.append("catalog_170_min_vram_gb_missing")
    if raw_recipe.get("video_input_strategy") not in LEGAL_VIDEO_INPUT_STRATEGIES:
        blockers.append("catalog_171_video_input_strategy_missing_or_illegal")
    pyav_strategy = raw_recipe.get("pyav_decode_strategy")
    if pyav_strategy not in LEGAL_PYAV_DECODE_STRATEGIES:
        blockers.append("catalog_181_pyav_decode_strategy_missing_or_illegal")
    targets = [str(item) for item in _as_list(raw_recipe.get("target_modes"))]
    if not targets:
        blockers.append("catalog_182_target_modes_missing")
    else:
        illegal = sorted(set(targets) - LEGAL_TARGET_MODES)
        if illegal:
            blockers.append(f"catalog_182_target_modes_illegal:{','.join(illegal)}")
    canary = raw_recipe.get("canary_status")
    if canary not in LEGAL_CANARY_STATUS:
        blockers.append("catalog_173_canary_status_missing_or_illegal")
    if canary == "post_canary_dependent" and not raw_recipe.get("canary_dependency"):
        blockers.append("catalog_173_canary_dependency_missing")
    min_smoke_gpu = str(raw_recipe.get("min_smoke_gpu") or "").strip()
    if min_smoke_gpu not in LEGAL_GPU_ORDER or min_smoke_gpu == "cpu":
        blockers.append("catalog_215_min_smoke_gpu_missing_or_illegal")
    cost_band = raw_recipe.get("cost_band", {}) or {}
    full_gpu = ""
    if isinstance(cost_band, Mapping):
        full_gpu = str(cost_band.get("gpu_key") or "").strip()
    if not full_gpu:
        full_gpu = _resolve_env_fallback(raw_recipe.get("gpu"))
    if (
        full_gpu in LEGAL_GPU_ORDER
        and min_smoke_gpu in LEGAL_GPU_ORDER
        and LEGAL_GPU_ORDER[full_gpu] < LEGAL_GPU_ORDER[min_smoke_gpu]
    ):
        blockers.append(
            f"cost_band_gpu_below_min_smoke_gpu:{full_gpu}<{min_smoke_gpu}"
        )
    if platform == "modal":
        text = _read_text(remote_driver)
        missing = [tok for tok in REQUIRED_MODAL_ENV_TOKENS if tok not in text]
        if missing:
            rel = (
                remote_driver.relative_to(repo_root).as_posix()
                if remote_driver and remote_driver.is_relative_to(repo_root)
                else str(remote_driver)
            )
            blockers.append(
                "catalog_244_modal_env_hygiene_missing:"
                + ",".join(missing)
                + f":{rel}"
            )
    return _tier("tier2_hardware_correctness", blockers)


def _tier3_substrate_correctness(
    raw_recipe: Mapping[str, Any],
    *,
    trainer: Path | None,
) -> DispatchProtocolTierReport:
    blockers: list[str] = []
    text = _read_text(trainer)
    if not text:
        return _tier("tier3_substrate_correctness", ("trainer_unreadable",))
    if "--enable-autocast-fp16" not in text and "AUTOCAST_FP16_WAIVED:" not in text:
        blockers.append("catalog_172_autocast_fp16_missing_or_unwaived")
    if (
        "allow_tf32" not in text
        and "--enable-tf32" not in text
        and "TF32_WAIVED:" not in text
    ):
        blockers.append("catalog_178_tf32_missing_or_unwaived")
    if (
        "--enable-torch-compile" not in text
        and "torch.compile" not in text
        and "TORCH_COMPILE_WAIVED:" not in text
    ):
        blockers.append("catalog_179_torch_compile_missing_or_unwaived")
    if (
        "torch.no_grad" not in text
        and "torch.inference_mode" not in text
        and "NO_GRAD_WAIVED:" not in text
    ):
        blockers.append("catalog_180_no_grad_eval_missing_or_unwaived")
    targets = {str(item) for item in _as_list(raw_recipe.get("target_modes"))}
    research_only = _as_bool(raw_recipe.get("research_only"), default=False)
    promotion_surface = bool(
        targets
        & {
            "contest_one_video_replay",
            "contest_generalized",
            "production_generalized",
            "production_edge_adaptive",
        }
    )
    if (
        promotion_surface
        and not research_only
        and "gate_auth_eval_call" not in text
        and "auth_eval_renderer" not in text
    ):
        blockers.append("catalog_226_auth_eval_canonical_helper_missing")
    return _tier("tier3_substrate_correctness", blockers)


def evaluate_dispatch_protocol_complete(
    raw_recipe: Mapping[str, Any],
    *,
    repo_root: str | Path,
    recipe_path: str | Path | None = None,
    trainer_path: str | Path | None = None,
    remote_driver_path: str | Path | None = None,
    native_dispatch: bool = True,
) -> DispatchProtocolReport:
    """Evaluate the full dispatch feasibility conjunction for one recipe.

    ``native_dispatch=False`` reports a skipped, non-blocking no-op surface so
    plan-only, release-only, and bespoke legacy actions are not misclassified
    as provider dispatches.
    """

    root = Path(repo_root).resolve()
    rpath = Path(recipe_path).resolve() if recipe_path is not None else None
    name = _recipe_name(raw_recipe, rpath)
    if not native_dispatch:
        skipped = _tier(
            "non_native_dispatch",
            (),
        )
        return DispatchProtocolReport(
            recipe_name=name,
            native_dispatch=False,
            dispatch_protocol_complete=True,
            tiers=(skipped,),
        )
    platform = str(raw_recipe.get("platform") or "").strip().lower()
    driver = _remote_driver_path(raw_recipe, root, remote_driver_path)
    trainer = _trainer_path(raw_recipe, root, trainer_path)
    tiers = (
        _tier1_engineering(
            raw_recipe,
            platform=platform,
            remote_driver=driver,
            trainer=trainer,
        ),
        _tier2_hardware_correctness(
            raw_recipe,
            platform=platform,
            repo_root=root,
            remote_driver=driver,
        ),
        _tier3_substrate_correctness(raw_recipe, trainer=trainer),
    )
    return DispatchProtocolReport(
        recipe_name=name,
        native_dispatch=True,
        dispatch_protocol_complete=all(tier.passed for tier in tiers),
        tiers=tiers,
    )


def require_dispatch_protocol_complete(
    raw_recipe: Mapping[str, Any],
    *,
    repo_root: str | Path,
    recipe_path: str | Path | None = None,
    trainer_path: str | Path | None = None,
    remote_driver_path: str | Path | None = None,
    native_dispatch: bool = True,
) -> DispatchProtocolReport:
    """Return the report or raise ``DispatchProtocolError`` before dispatch."""

    report = evaluate_dispatch_protocol_complete(
        raw_recipe,
        repo_root=repo_root,
        recipe_path=recipe_path,
        trainer_path=trainer_path,
        remote_driver_path=remote_driver_path,
        native_dispatch=native_dispatch,
    )
    if not report.dispatch_protocol_complete:
        raise DispatchProtocolError(
            "dispatch_protocol_complete=false for "
            f"{report.recipe_name}: "
            + "; ".join(report.blockers[:12])
        )
    return report
