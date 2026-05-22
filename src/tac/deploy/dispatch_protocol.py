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
    "LEGAL_DISPATCH_KINDS",
    "LEGAL_NATIVE_PLATFORMS",
    "LOCAL_RESEARCH_SIGNAL_PLATFORMS",
    "TOOL_DISPATCH_LEGAL_GPU_TOKENS",
    "DispatchProtocolError",
    "DispatchProtocolReport",
    "DispatchProtocolTierReport",
    "evaluate_dispatch_protocol_complete",
    "is_local_research_signal_dispatch",
    "is_tool_dispatch",
    "require_dispatch_protocol_complete",
]

DISPATCH_PROTOCOL_SCHEMA = "pact.dispatch_protocol_complete.v1"

# Catalog #270 scope clarification (2026-05-17 per lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517):
# The dispatch protocol's Tier 2/3 fields below are SCOPED to substrate trainers
# (``experiments/train_substrate_*.py``). Tool dispatches (``tools/*.py``) are
# categorically NOT subject to:
#   - Catalog #172 ``--enable-autocast-fp16`` (FP16 autocast is a substrate trainer
#     primitive; tool one-shot inference does not benefit and may be CPU-only).
#   - Catalog #178 TF32 (CUDA matmul-only; inapplicable to CPU tool dispatches).
#   - Catalog #179 ``--enable-torch-compile`` (substrate training primitive; one-shot
#     tool inference does not benefit from compile overhead).
#   - Catalog #226 ``gate_auth_eval_call`` (substrate auth-eval routing; tools that
#     are not contest_auth_eval invocations do not produce contest-CUDA score claims).
#   - Catalog #215 ``min_smoke_gpu`` GPU class (tool dispatches can be CPU-only).
# Tool dispatches DO still get the GENUINELY hardware-correctness Tier 1/2 fields:
#   - lane_id / dispatch_enabled / cost_band / driver-and-trainer existence (T1)
#   - min_vram_gb / video_input_strategy / pyav_decode_strategy / target_modes /
#     canary_status / Modal NVML env block (T2 — env hygiene is universal)
#   - no_grad / eval-roundtrip discipline (T3 — when applicable)
# Detection is via two surfaces (either short-circuits ``_is_tool_dispatch``):
#   1. Explicit: ``dispatch_kind: tool`` in the recipe frontmatter.
#   2. Implicit: trainer_path matches ``tools/*.py`` (not ``experiments/train_substrate_*.py``).
# Sister of the runtime scope-clarification CLAUDE.md row under "Production-hardened
# dispatch optimization protocol".
LEGAL_DISPATCH_KINDS = frozenset(
    {"substrate", "tool", "local_research_signal", "hf_jobs_research_surrogate"}
)
TOOL_DISPATCH_LEGAL_GPU_TOKENS = frozenset({"cpu", "CPU", "Cpu"})
# Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + operator directive
# 2026-05-17 ("Deploying to local MPS versus modal should be super easy to
# configure, like one arg in a func"): ``local_mps`` and ``local_cpu`` are
# first-class dispatch targets for non-authoritative research-signal generation.
# They route through ``mps_research_signal`` and ``macos_cpu_advisory_signal``
# canonical manifests (Catalog #1 + #192 sister discipline). The
# ``local_research_signal`` dispatch_kind tells the protocol gates to skip
# substrate-only Tier 2/3 checks (sister of Catalog #270 scope clarification
# for tool dispatches; same precedent applied to a different kind).
# Slot 13 (2026-05-19) wired ``_dispatch_hf_jobs`` into ``tools/operator_authorize.py``
# but the legal-platforms enum here was NOT extended in the same commit batch
# — slot 26's HF Jobs T4 dispatch was therefore refused by
# ``evaluate_dispatch_protocol_complete`` with
# ``platform 'hf_jobs' not in LEGAL_NATIVE_PLATFORMS``. Per CLAUDE.md "Bugs must
# be permanently fixed AND self-protected against" + the slot 26 handoff,
# extending the enum here closes the dispatch-blocked gap so HF Jobs vision
# training jobs (per ``feedback_hf_jobs_segnet_surrogate_per_pixel_sister_lane_landed_20260519.md``)
# can route through the canonical operator-authorize 30s harness.
LEGAL_NATIVE_PLATFORMS = frozenset(
    {"modal", "vastai", "vast", "local", "local_mps", "local_cpu", "hf_jobs"}
)
LOCAL_RESEARCH_SIGNAL_PLATFORMS = frozenset({"local_mps", "local_cpu"})
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
_MODE_ENV_SUFFIXES = (
    "TRAINER_MODE",
    "DISPATCH_MODE",
    "RUN_MODE",
    "SMOKE_MODE",
    "FULL_MODE",
    "MODE_FULL",
    "SMOKE_ONLY",
)
_DEVICE_ARG_ENV_RE = re.compile(
    r"--device\s+(?:[\"']?\$\{?([A-Z0-9_]*DEVICE[A-Z0-9_]*)\}?[\"']?)"
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


def _env_overrides(raw_recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    env = raw_recipe.get("env_overrides")
    return env if isinstance(env, Mapping) else {}


def _mode_value_forces_full(env_var: str, value: Any) -> bool:
    low = _clean_env_text(value).lower()
    if not low:
        return False
    if "smoke" in env_var.lower():
        return low in {"0", "false", "no", "off", "full"}
    return low in {"full", "0", "false", "no", "off"}


def _clean_env_text(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _mode_prefix(env_var: str) -> str:
    for suffix in _MODE_ENV_SUFFIXES:
        if env_var.endswith(suffix):
            return env_var[: -len(suffix)]
    return ""


def _driver_device_env_vars(driver_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _DEVICE_ARG_ENV_RE.finditer(driver_text):
        var = match.group(1)
        if var in seen:
            continue
        seen.add(var)
        ordered.append(var)
    return tuple(ordered)


def _driver_env_default_value(driver_text: str, env_var: str) -> str | None:
    pattern = re.compile(
        rf'^\s*{re.escape(env_var)}\s*=\s*["\']?\$\{{{re.escape(env_var)}:-([^}}]*)\}}',
        re.MULTILINE,
    )
    match = pattern.search(driver_text)
    if not match:
        return None
    value = match.group(1).strip().strip('"').strip("'")
    return value or None


def _full_mode_device_cpu_blockers(
    raw_recipe: Mapping[str, Any],
    *,
    remote_driver: Path | None,
) -> list[str]:
    env = _env_overrides(raw_recipe)
    full_modes = {
        str(key): value
        for key, value in env.items()
        if any(str(key).endswith(suffix) for suffix in _MODE_ENV_SUFFIXES)
        and _mode_value_forces_full(str(key), value)
    }
    if not full_modes:
        return []
    driver_text = _read_text(remote_driver)
    device_vars = _driver_device_env_vars(driver_text)
    if not device_vars:
        return []
    full_prefixes = {_mode_prefix(var) for var in full_modes}
    blockers: list[str] = []
    for device_var in device_vars:
        device_prefix = (
            device_var[: -len("DEVICE")] if device_var.endswith("DEVICE") else ""
        )
        if full_prefixes and "" not in full_prefixes and device_prefix not in full_prefixes:
            continue
        effective_device = env.get(device_var)
        if effective_device is None:
            effective_device = _driver_env_default_value(driver_text, device_var)
        if _clean_env_text(effective_device).lower() != "cpu":
            continue
        mode_pairs = ",".join(
            f"{var}={_clean_env_text(value)}"
            for var, value in sorted(full_modes.items())
            if _mode_prefix(var) in {"", device_prefix}
        )
        blockers.append(
            "full_mode_device_cpu_bug_class:"
            f"{mode_pairs or 'full_mode'}:{device_var}=cpu"
        )
    return blockers


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


def _is_tool_dispatch(
    raw_recipe: Mapping[str, Any],
    *,
    trainer_path: Path | None,
    repo_root: Path,
) -> bool:
    """Return True when this dispatch is a TOOL invocation (not a substrate trainer).

    Two acceptance surfaces (either short-circuits to True):
      1. Explicit: ``dispatch_kind: tool`` in the recipe frontmatter.
      2. Implicit: trainer path is under ``tools/`` AND not under
         ``experiments/train_substrate_*.py``.

    Tool dispatches are categorically NOT subject to substrate-trainer-only
    Tier 2/3 fields (Catalogs #172/#178/#179/#226/#215-CPU). See the
    ``LEGAL_DISPATCH_KINDS`` docstring above.

    Per CLAUDE.md "Production-hardened dispatch optimization protocol"
    Catalog #270 scope clarification (2026-05-17).
    """

    explicit_kind = str(raw_recipe.get("dispatch_kind") or "").strip().lower()
    if explicit_kind == "tool":
        return True
    if explicit_kind == "hf_jobs_research_surrogate":
        return str(raw_recipe.get("platform") or "").strip().lower() == "hf_jobs"
    if explicit_kind in {"substrate", "local_research_signal"}:
        return False
    # Implicit detection by trainer path.
    if trainer_path is None:
        return False
    try:
        rel = trainer_path.resolve().relative_to(repo_root.resolve())
    except (OSError, ValueError):
        return False
    posix = rel.as_posix()
    # Substrate trainer surface is canonical: ``experiments/train_substrate_*.py``.
    if posix.startswith("experiments/train_substrate_") and posix.endswith(".py"):
        return False
    # Tool dispatch surface is canonical: ``tools/*.py``.
    return posix.startswith("tools/") and posix.endswith(".py")


# Public alias for ``_is_tool_dispatch`` per the module's ``__all__`` contract.
# Test callers should use ``is_tool_dispatch`` directly; runtime callers in this
# module use the private name to keep the call surface small and intentional.
is_tool_dispatch = _is_tool_dispatch


def _is_local_research_signal_dispatch(
    raw_recipe: Mapping[str, Any],
) -> bool:
    """Return True when this dispatch is a LOCAL RESEARCH-SIGNAL invocation.

    Two acceptance surfaces (either short-circuits to True):
      1. Explicit: ``dispatch_kind: local_research_signal`` in the recipe.
      2. Implicit: ``platform`` ∈ ``LOCAL_RESEARCH_SIGNAL_PLATFORMS``
         (``local_mps`` or ``local_cpu``).

    Local-research-signal dispatches are categorically NOT subject to
    substrate-trainer-only Tier 2/3 fields (sister of Catalog #270 scope
    clarification for tool dispatches; same precedent applied to a different
    dispatch_kind). They route through the canonical
    ``tac.optimization.mps_research_signal`` (for ``local_mps``) or
    ``tac.optimization.macos_cpu_advisory_signal`` (for ``local_cpu``) helpers
    so the canonical posterior is structurally protected from non-authoritative
    results.

    Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1 +
    Catalog #192 + operator directive 2026-05-17 ("Deploying to local MPS
    versus modal should be super easy to configure, like one arg in a func").
    """

    explicit_kind = str(raw_recipe.get("dispatch_kind") or "").strip().lower()
    if explicit_kind == "local_research_signal":
        return True
    if explicit_kind in {"substrate", "tool"}:
        return False
    platform = str(raw_recipe.get("platform") or "").strip().lower()
    return platform in LOCAL_RESEARCH_SIGNAL_PLATFORMS


# Public alias mirroring ``is_tool_dispatch``.
is_local_research_signal_dispatch = _is_local_research_signal_dispatch


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
    # Local research-signal dispatches (local_mps / local_cpu) need NOT declare
    # cost_band.epochs because they run on the operator's machine at $0. The
    # remote_driver requirement is also relaxed because the dispatch is a
    # direct shell-out, not a provider script. Sister exemption to Catalog
    # #270's tool-dispatch scope-fix; same precedent applied here.
    is_local_research_signal = platform in LOCAL_RESEARCH_SIGNAL_PLATFORMS
    if not is_local_research_signal:
        cost_band = raw_recipe.get("cost_band", {}) or {}
        epochs = _as_int(cost_band.get("epochs") if isinstance(cost_band, Mapping) else None)
        if epochs is None or epochs <= 0:
            blockers.append("cost_band_epochs_missing_or_nonpositive")
    if remote_driver is None or not remote_driver.is_file():
        blockers.append("remote_driver_missing")
    if trainer is None or not trainer.is_file():
        blockers.append("required_input_files_trainer_missing")
    dispatch_kind = str(raw_recipe.get("dispatch_kind") or "").strip().lower()
    if dispatch_kind == "hf_jobs_research_surrogate":
        if platform != "hf_jobs":
            blockers.append("hf_jobs_research_surrogate_platform_not_hf_jobs")
        if _as_bool(raw_recipe.get("research_only"), default=False) is not True:
            blockers.append("hf_jobs_research_surrogate_requires_research_only_true")
        if raw_recipe.get("score_claim") is not False:
            blockers.append("hf_jobs_research_surrogate_requires_score_claim_false")
        if raw_recipe.get("promotion_eligible") is not False:
            blockers.append(
                "hf_jobs_research_surrogate_requires_promotion_eligible_false"
            )
        if raw_recipe.get("ready_for_exact_eval_dispatch") is not False:
            blockers.append(
                "hf_jobs_research_surrogate_requires_ready_for_exact_eval_dispatch_false"
            )
        hf_cfg = raw_recipe.get("hf_jobs", {}) or {}
        expected_axis = (
            str(hf_cfg.get("expected_axis") or raw_recipe.get("expected_axis") or "")
            .strip()
            .lower()
        )
        if expected_axis != "advisory":
            blockers.append("hf_jobs_research_surrogate_expected_axis_not_advisory")
    return _tier("tier1_engineering", blockers)


def _tier2_hardware_correctness(
    raw_recipe: Mapping[str, Any],
    *,
    platform: str,
    repo_root: Path,
    remote_driver: Path | None,
    is_tool: bool = False,
) -> DispatchProtocolTierReport:
    blockers: list[str] = []
    # Local research-signal dispatches (local_mps / local_cpu) run on the
    # operator's machine; the substrate-only hardware fields (min_vram_gb,
    # video_input_strategy, pyav_decode_strategy, target_modes, canary_status,
    # min_smoke_gpu, NVML env block) do not apply because the dispatch never
    # touches a paid provider. Sister of the Catalog #270 scope clarification
    # for tool dispatches.
    is_local_research_signal = platform in LOCAL_RESEARCH_SIGNAL_PLATFORMS
    if is_local_research_signal:
        return _tier("tier2_hardware_correctness", ())
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
    min_smoke_gpu_raw = str(raw_recipe.get("min_smoke_gpu") or "").strip()
    # Catalog #270 scope fix (2026-05-17): tool dispatches MAY declare ``CPU``
    # (case-insensitive) for ``min_smoke_gpu`` because they are not GPU-bound.
    # Substrate trainers still get the GPU-class enforcement so a CPU smoke
    # cannot stand in for a CUDA-required substrate dispatch.
    min_smoke_gpu_is_cpu = min_smoke_gpu_raw in TOOL_DISPATCH_LEGAL_GPU_TOKENS
    if is_tool and min_smoke_gpu_is_cpu:
        # Tool CPU dispatch: accept; do not enforce GPU class.
        pass
    elif min_smoke_gpu_raw not in LEGAL_GPU_ORDER or min_smoke_gpu_raw == "cpu":
        blockers.append("catalog_215_min_smoke_gpu_missing_or_illegal")
    cost_band = raw_recipe.get("cost_band", {}) or {}
    full_gpu = ""
    if isinstance(cost_band, Mapping):
        full_gpu = str(cost_band.get("gpu_key") or "").strip()
    if not full_gpu:
        full_gpu = _resolve_env_fallback(raw_recipe.get("gpu"))
    # Tool CPU dispatch: also accept ``CPU`` as the full ``cost_band.gpu_key`` /
    # ``gpu`` to short-circuit the substrate-only GPU-ordering check.
    full_gpu_is_cpu = full_gpu in TOOL_DISPATCH_LEGAL_GPU_TOKENS
    if (
        not (is_tool and (full_gpu_is_cpu or min_smoke_gpu_is_cpu))
        and full_gpu in LEGAL_GPU_ORDER
        and min_smoke_gpu_raw in LEGAL_GPU_ORDER
        and LEGAL_GPU_ORDER[full_gpu] < LEGAL_GPU_ORDER[min_smoke_gpu_raw]
    ):
        blockers.append(
            f"cost_band_gpu_below_min_smoke_gpu:{full_gpu}<{min_smoke_gpu_raw}"
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
    blockers.extend(
        _full_mode_device_cpu_blockers(raw_recipe, remote_driver=remote_driver)
    )
    return _tier("tier2_hardware_correctness", blockers)


def _tier3_substrate_correctness(
    raw_recipe: Mapping[str, Any],
    *,
    trainer: Path | None,
    is_tool: bool = False,
) -> DispatchProtocolTierReport:
    blockers: list[str] = []
    text = _read_text(trainer)
    if not text:
        return _tier("tier3_substrate_correctness", ("trainer_unreadable",))
    # Catalog #270 scope fix (2026-05-17): tool dispatches are categorically
    # NOT subject to substrate-trainer Tier 3 fields (Catalogs #172/#178/#179/
    # #226). These checks enforce substrate training primitives (autocast,
    # TF32, torch.compile, canonical auth-eval routing) that do not apply to
    # one-shot tool inference. Tool dispatches still get the no_grad/
    # inference_mode check because eval-time memory hygiene IS universally
    # applicable when torch is in use. If a tool doesn't use torch at all the
    # no_grad check is also vacuous; we keep it advisory-only via the existing
    # NO_GRAD_WAIVED token mechanism.
    if not is_tool:
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
    if not is_tool:
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
    # Catalog #270 scope clarification (2026-05-17): substrate-trainer Tier 2/3
    # fields are skipped for ``dispatch_kind: tool`` recipes (or ``tools/*.py``
    # trainer paths) per ``_is_tool_dispatch`` detection. See module docstring.
    # Sister 2026-05-17 (lane_one_arg_local_mps_vs_modal_dispatch_switch_20260517):
    # ``dispatch_kind: local_research_signal`` (or ``platform: local_mps``/
    # ``local_cpu``) gets the SAME skip-set per the same precedent — these are
    # non-substrate-trainer dispatches that route through the canonical
    # mps_research_signal / macos_cpu_advisory_signal manifests instead of
    # contest auth_eval. The ``is_tool`` variable below carries the BROADER
    # semantic "non-substrate dispatch — skip substrate-only checks".
    is_tool = _is_tool_dispatch(raw_recipe, trainer_path=trainer, repo_root=root)
    is_local_research_signal = _is_local_research_signal_dispatch(raw_recipe)
    skip_substrate_only_checks = is_tool or is_local_research_signal
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
            is_tool=skip_substrate_only_checks,
        ),
        _tier3_substrate_correctness(
            raw_recipe,
            trainer=trainer,
            is_tool=skip_substrate_only_checks,
        ),
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
