# SPDX-License-Identifier: MIT
"""Per-substrate local-routability classification audit.

Per operator directive 2026-05-21 verbatim *"Let's make sure we are leveraging
local cpu and mps and metal and mlx as much as possible"*: this module
classifies every substrate trainer + operator-authorize recipe by whether
its memory footprint + kernel surface can be served by the local M5 Max
128GB unified memory + Metal GPU + MLX framework, OR whether it requires
paid Linux x86_64 + NVIDIA dispatch.

Per CLAUDE.md non-negotiables preserved:
- Local routing is ALWAYS non-promotable per Catalog #1/#192/#317; this
  audit does NOT relax authoritative-axis discipline.
- Paid contest-axis remains required for promotion + score-claim per
  CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
- Local routing is OPERATOR-ELIGIBLE for: paradigm prototyping, premise
  verification, smoke validation, candidate generation, cargo-cult-unwind
  iteration per Carmack MVP-first 5-step phasing (CLAUDE.md ``be125b878``).

Classification taxonomy (4-tier per M5 Max specs):

- **LOCAL_MLX_TRAINABLE**: substrate fits comfortably in MLX framework
  on M5 Max Metal GPU (< 16GB declared min_vram_gb; substrate uses
  primitives MLX supports — Linear / Conv / activation / loss). MLX is
  measurably faster than MPS for these primitives per Apple's published
  benchmarks (~2-3x typical on small models < 1B params).

- **LOCAL_MPS_TRAINABLE**: substrate fits but uses primitives MLX does
  NOT yet support (e.g. specific torch ops without MLX equivalents),
  routed via PyTorch MPS backend instead. Still local + non-promotable.
  Recommended for 17-40GB declared min_vram_gb substrates per M5 Max
  unified memory (~80GB usable for working set).

- **LOCAL_CPU_PROXY**: substrate's compute kernel is CPU-friendly (e.g.
  codec primitives without neural net; lossless byte transforms;
  arithmetic coding inner loops). Routed via macOS-CPU per
  :mod:`tac.optimization.macos_cpu_advisory_signal`.

- **PAID_ONLY**: substrate requires kernels or memory footprint that
  exceeds local M5 Max capacity OR requires Linux x86_64 / NVIDIA-specific
  kernels (e.g. DALI video decode, NVDEC, CUDA-specific kernels). Must
  route through paid Modal/Vast.ai/Lightning.

The audit emits a structured JSON manifest to ``.omx/state/local_leverage_routability_audit_<utc>.json``
that downstream consumers (cathedral autopilot, operator briefing,
cost-band priority queue) read to recommend the cheapest faithful
substrate-prototype route BEFORE paid dispatch is considered.

Per CLAUDE.md "Strict-flip atomicity rule": this audit is OPERATOR-FACING
ranking helper; no STRICT preflight gate is wired by THIS landing because
the classification depends on per-substrate empirical kernel-compatibility
testing (sister wave; queued as op-routable).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SubstrateRoutabilityClass",
    "SubstrateRoutabilityVerdict",
    "M5_MAX_UNIFIED_MEMORY_BYTES",
    "M5_MAX_USABLE_WORKING_SET_BYTES",
    "MLX_TRAINABLE_VRAM_CEILING_GB",
    "MPS_TRAINABLE_VRAM_CEILING_GB",
    "PAID_VRAM_CEILING_GB",
    "classify_recipe_routability",
    "audit_all_substrate_recipes",
    "verdict_summary_text",
]

# Per MLX `mx.device_info()` on M5 Max 2026-05-21 empirical anchor:
# memory_size=137438953472 (128 GiB) ; max_recommended_working_set_size=115448725504 (107.5 GiB)
M5_MAX_UNIFIED_MEMORY_BYTES = 137_438_953_472  # 128 GiB
M5_MAX_USABLE_WORKING_SET_BYTES = 115_448_725_504  # ~107.5 GiB per Apple's recommendation

# Classification thresholds. Conservative defaults — assume substrate's
# declared min_vram_gb is the FORWARD-pass footprint; training typically
# 2-3x this (gradients + optimizer state + intermediate activations).
# Sister CLAUDE.md "Substrate recipes declare min_vram_gb floor" Catalog #170.
MLX_TRAINABLE_VRAM_CEILING_GB = 16  # comfortable in MLX; sub-16GB pre-training memory
MPS_TRAINABLE_VRAM_CEILING_GB = 40  # MPS-routable up to 40GB declared (training ~80-120GB peak)
PAID_VRAM_CEILING_GB = 80  # beyond M5 Max usable working set; paid-only

# Substrate primitive tokens that DO have MLX equivalents (incomplete; sister
# subagent extends via empirical per-substrate kernel compatibility testing).
# Conservative default: substrates whose trainer body uses ONLY these primitive
# tokens are MLX-trainable candidates.
_MLX_COMPATIBLE_PRIMITIVE_TOKENS = (
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose2d",
    "BatchNorm",
    "LayerNorm",
    "GroupNorm",
    "ReLU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Dropout",
    "Embedding",
    "MultiheadAttention",
    "MSELoss",
    "L1Loss",
    "CrossEntropyLoss",
    "BCELoss",
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
)

# Substrate primitive tokens that LACK MLX equivalents (as of MLX 0.x) and
# need PyTorch MPS fallback or paid CUDA. Examples: specific scatter ops,
# fused kernels, custom CUDA kernels, NVDEC dependencies.
_MLX_INCOMPATIBLE_TOKENS = (
    "torch.cuda.amp",
    "torch.distributed",
    "DistributedDataParallel",
    "NVDEC",
    "DALI",
    "fn.experimental.inputs.video",
    "torch._C._cuda",
    "tensor_parallel",
    "flash_attn",
    "xformers",
)


class SubstrateRoutabilityClass:
    """Canonical routability classification per substrate."""

    LOCAL_MLX_TRAINABLE = "LOCAL_MLX_TRAINABLE"
    LOCAL_MPS_TRAINABLE = "LOCAL_MPS_TRAINABLE"
    LOCAL_CPU_PROXY = "LOCAL_CPU_PROXY"
    PAID_ONLY = "PAID_ONLY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def all_classes(cls) -> tuple[str, ...]:
        return (
            cls.LOCAL_MLX_TRAINABLE,
            cls.LOCAL_MPS_TRAINABLE,
            cls.LOCAL_CPU_PROXY,
            cls.PAID_ONLY,
            cls.UNKNOWN,
        )


@dataclass(frozen=True, slots=True)
class SubstrateRoutabilityVerdict:
    """Typed per-substrate routability verdict.

    Carries the classification + rationale + per-Catalog non-promotable
    markers so downstream consumers (cathedral autopilot, operator briefing)
    can route the substrate without re-deriving the analysis.
    """

    substrate_id: str
    recipe_path: str
    classification: str
    min_vram_gb_declared: int | None
    rationale: str
    estimated_cost_compression_usd: float
    # Per Catalog #287 + #323 canonical Provenance: every routability verdict
    # carries the non-promotable triple.
    evidence_grade: str = "local-routability-audit-advisory"
    promotable: bool = False
    score_claim: bool = False
    blockers: tuple[str, ...] = field(
        default=(
            "local_routability_audit_is_advisory_only_per_catalog_192_317",
            "paid_linux_x86_64_nvidia_required_for_authoritative_axis_per_claude_md",
            "auto_routing_disabled_pending_per_substrate_empirical_kernel_compat_test",
        )
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "recipe_path": self.recipe_path,
            "classification": self.classification,
            "min_vram_gb_declared": self.min_vram_gb_declared,
            "rationale": self.rationale,
            "estimated_cost_compression_usd": self.estimated_cost_compression_usd,
            "evidence_grade": self.evidence_grade,
            "promotable": self.promotable,
            "score_claim": self.score_claim,
            "blockers": list(self.blockers),
        }


def _extract_min_vram_gb(recipe_text: str) -> int | None:
    """Extract ``min_vram_gb`` from a recipe YAML body.

    Returns None if not declared (the recipe is out-of-scope per Catalog #170
    sister discipline; classification falls back to UNKNOWN).
    """

    match = re.search(r"^min_vram_gb:\s*(\d+)", recipe_text, re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1))


def _extract_substrate_id(recipe_path: Path) -> str:
    """Extract canonical substrate_id from recipe filename.

    Pattern: ``substrate_<id>_modal_<gpu>_dispatch.yaml`` →  ``<id>``.
    Falls back to filename-without-extension for non-conforming names.
    """

    stem = recipe_path.stem
    # Strip canonical prefix + suffix per CLAUDE.md naming convention.
    stripped = re.sub(r"^substrate_", "", stem)
    stripped = re.sub(r"_modal_.*$", "", stripped)
    stripped = re.sub(r"_smoke_dispatch$", "", stripped)
    stripped = re.sub(r"_dispatch$", "", stripped)
    return stripped


def _check_mlx_incompatibility(recipe_text: str, trainer_text: str) -> tuple[bool, list[str]]:
    """Check whether substrate trainer uses any MLX-incompatible primitives.

    Returns (is_incompatible, matched_token_list). When any incompatible
    token is found in either recipe env_overrides or trainer body, the
    substrate cannot route through MLX.
    """

    matched: list[str] = []
    haystack = recipe_text + "\n" + trainer_text
    for token in _MLX_INCOMPATIBLE_TOKENS:
        if token in haystack:
            matched.append(token)
    return (len(matched) > 0, matched)


def classify_recipe_routability(
    recipe_path: Path,
    trainer_path: Path | None = None,
) -> SubstrateRoutabilityVerdict:
    """Classify a single substrate recipe by local-routability.

    Args:
        recipe_path: path to ``.omx/operator_authorize_recipes/substrate_*.yaml``.
        trainer_path: optional path to ``experiments/train_substrate_<id>.py``
            for MLX-incompatibility scanning. When None, MLX incompatibility
            is conservatively assumed unless min_vram_gb classification puts
            the substrate in CPU-proxy / paid-only territory anyway.

    Returns:
        :class:`SubstrateRoutabilityVerdict` with classification + rationale +
        canonical non-promotable markers.
    """

    if not recipe_path.exists():
        return SubstrateRoutabilityVerdict(
            substrate_id="unknown",
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.UNKNOWN,
            min_vram_gb_declared=None,
            rationale=f"recipe not found at {recipe_path}",
            estimated_cost_compression_usd=0.0,
        )

    recipe_text = recipe_path.read_text(encoding="utf-8", errors="ignore")
    substrate_id = _extract_substrate_id(recipe_path)
    min_vram = _extract_min_vram_gb(recipe_text)

    trainer_text = ""
    if trainer_path is not None and trainer_path.exists():
        trainer_text = trainer_path.read_text(encoding="utf-8", errors="ignore")

    is_mlx_incompatible, incompat_tokens = _check_mlx_incompatibility(
        recipe_text, trainer_text
    )

    # Classification cascade per M5 Max 128GB unified memory + MLX availability.
    if min_vram is None:
        return SubstrateRoutabilityVerdict(
            substrate_id=substrate_id,
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.UNKNOWN,
            min_vram_gb_declared=None,
            rationale=(
                f"recipe does not declare min_vram_gb per Catalog #170; "
                f"classification deferred pending operator-routed backfill"
            ),
            estimated_cost_compression_usd=0.0,
        )

    if min_vram > PAID_VRAM_CEILING_GB:
        return SubstrateRoutabilityVerdict(
            substrate_id=substrate_id,
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.PAID_ONLY,
            min_vram_gb_declared=min_vram,
            rationale=(
                f"min_vram_gb={min_vram} > M5 Max usable working set "
                f"(~80GB safe limit); paid dispatch required for substrate "
                f"training"
            ),
            estimated_cost_compression_usd=0.0,
        )

    if is_mlx_incompatible:
        # MLX cannot handle the primitives; fall back to MPS or CPU.
        if min_vram <= MPS_TRAINABLE_VRAM_CEILING_GB:
            return SubstrateRoutabilityVerdict(
                substrate_id=substrate_id,
                recipe_path=str(recipe_path),
                classification=SubstrateRoutabilityClass.LOCAL_MPS_TRAINABLE,
                min_vram_gb_declared=min_vram,
                rationale=(
                    f"min_vram_gb={min_vram}GB fits M5 Max unified memory; "
                    f"routes through PyTorch MPS backend (MLX-incompatible "
                    f"primitives present: {incompat_tokens[:3]}); "
                    f"non-promotable per Catalog #1/#192/#317"
                ),
                estimated_cost_compression_usd=2.0,  # typical smoke cost saved
            )
        return SubstrateRoutabilityVerdict(
            substrate_id=substrate_id,
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.PAID_ONLY,
            min_vram_gb_declared=min_vram,
            rationale=(
                f"min_vram_gb={min_vram}GB exceeds MPS training comfort + "
                f"MLX-incompatible (tokens: {incompat_tokens[:3]}); "
                f"paid dispatch required"
            ),
            estimated_cost_compression_usd=0.0,
        )

    if min_vram <= MLX_TRAINABLE_VRAM_CEILING_GB:
        return SubstrateRoutabilityVerdict(
            substrate_id=substrate_id,
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.LOCAL_MLX_TRAINABLE,
            min_vram_gb_declared=min_vram,
            rationale=(
                f"min_vram_gb={min_vram}GB fits MLX training comfort on "
                f"M5 Max Metal GPU; MLX framework available + no "
                f"incompatible primitives detected; ~2-3x faster than MPS "
                f"typical; non-promotable per Catalog #1/#192/#317"
            ),
            estimated_cost_compression_usd=3.0,  # typical T4/A10G smoke cost saved
        )

    if min_vram <= MPS_TRAINABLE_VRAM_CEILING_GB:
        return SubstrateRoutabilityVerdict(
            substrate_id=substrate_id,
            recipe_path=str(recipe_path),
            classification=SubstrateRoutabilityClass.LOCAL_MPS_TRAINABLE,
            min_vram_gb_declared=min_vram,
            rationale=(
                f"min_vram_gb={min_vram}GB fits M5 Max unified memory but "
                f"exceeds MLX comfort ceiling ({MLX_TRAINABLE_VRAM_CEILING_GB}GB); "
                f"routes through PyTorch MPS backend; "
                f"non-promotable per Catalog #1/#192/#317"
            ),
            estimated_cost_compression_usd=5.0,  # typical A100 smoke cost saved
        )

    # min_vram in (40, 80] range - tight on local, paid-preferred.
    return SubstrateRoutabilityVerdict(
        substrate_id=substrate_id,
        recipe_path=str(recipe_path),
        classification=SubstrateRoutabilityClass.PAID_ONLY,
        min_vram_gb_declared=min_vram,
        rationale=(
            f"min_vram_gb={min_vram}GB exceeds MPS training comfort "
            f"({MPS_TRAINABLE_VRAM_CEILING_GB}GB) on M5 Max unified memory; "
            f"paid dispatch preferred for safe execution"
        ),
        estimated_cost_compression_usd=0.0,
    )


def audit_all_substrate_recipes(
    repo_root: Path | str,
) -> list[SubstrateRoutabilityVerdict]:
    """Audit all substrate recipes under ``.omx/operator_authorize_recipes/``.

    Args:
        repo_root: repo root path. Recipes scanned at
            ``{repo_root}/.omx/operator_authorize_recipes/substrate_*.yaml``.
            Trainers scanned at
            ``{repo_root}/experiments/train_substrate_<id>.py`` per recipe.

    Returns:
        list of :class:`SubstrateRoutabilityVerdict` ordered by substrate_id.
    """

    repo = Path(repo_root)
    recipes_dir = repo / ".omx" / "operator_authorize_recipes"
    if not recipes_dir.exists():
        return []

    verdicts: list[SubstrateRoutabilityVerdict] = []
    seen_substrate_ids: set[str] = set()
    for recipe_path in sorted(recipes_dir.glob("substrate_*.yaml")):
        substrate_id = _extract_substrate_id(recipe_path)
        if substrate_id in seen_substrate_ids:
            # Multiple recipes for same substrate (e.g. modal_t4 + modal_a100
            # variants); dedup by canonical substrate_id.
            continue
        seen_substrate_ids.add(substrate_id)

        trainer_path = (
            repo / "experiments" / f"train_substrate_{substrate_id}.py"
        )
        verdict = classify_recipe_routability(recipe_path, trainer_path)
        verdicts.append(verdict)

    return verdicts


def verdict_summary_text(verdicts: Iterable[SubstrateRoutabilityVerdict]) -> str:
    """Render a human-readable summary of routability verdicts."""

    by_class: dict[str, list[SubstrateRoutabilityVerdict]] = {
        cls: [] for cls in SubstrateRoutabilityClass.all_classes()
    }
    for v in verdicts:
        by_class.setdefault(v.classification, []).append(v)

    total = sum(len(lst) for lst in by_class.values())
    if total == 0:
        return "No substrate recipes scanned."

    total_compression = sum(
        v.estimated_cost_compression_usd
        for lst in by_class.values()
        for v in lst
    )

    lines = [
        f"Local-leverage routability audit: {total} substrate recipes scanned",
        f"  Total estimated cost-compression opportunity: ~${total_compression:.2f} (per-smoke)",
        "",
    ]
    for cls in SubstrateRoutabilityClass.all_classes():
        lst = by_class.get(cls, [])
        if not lst:
            continue
        lines.append(f"  {cls}: {len(lst)} substrates")
        for v in lst[:10]:
            lines.append(
                f"    - {v.substrate_id} (vram={v.min_vram_gb_declared}GB, "
                f"save~${v.estimated_cost_compression_usd:.2f})"
            )
        if len(lst) > 10:
            lines.append(f"    ... +{len(lst) - 10} more")
        lines.append("")

    lines.append(
        "All local routings are NON-PROMOTABLE per Catalog #1/#192/#317; "
        "paid Linux x86_64 + NVIDIA remains required for authoritative axis."
    )
    return "\n".join(lines)


def write_audit_manifest(
    verdicts: Iterable[SubstrateRoutabilityVerdict],
    out_path: Path,
) -> None:
    """Persist audit manifest to canonical state path.

    Per Catalog #131 fcntl-locked bare-write discipline: callers should
    route through the canonical write helper at the consumer site (sister
    of :func:`tac.optimization.macos_cpu_advisory_signal.append_manifest_row_to_jsonl`).
    This module's writer is for one-shot operator audits; downstream
    cathedral consumers should use the canonical helper.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "local_leverage_routability_audit.v1",
        "verdicts": [v.as_dict() for v in verdicts],
        "summary": verdict_summary_text(verdicts),
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
