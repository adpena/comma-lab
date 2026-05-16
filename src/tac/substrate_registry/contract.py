# SPDX-License-Identifier: MIT
"""SubstrateContract — canonical schema for the META layer.

Per `.omx/research/substrate_meta_layer_design_20260515.md`, this is the
single source-of-truth contract a substrate must declare via
``@register_substrate(...)``. The contract drives:

  1. Decoration-time validation (refuses to import inconsistent substrates).
  2. Auto-wire query helpers (the 6 Catalog #125 hooks read from the registry).
  3. Auto-generation of recipes + remote drivers + (future) lane-registry
     entries + (future) e2e pytest fixtures.

The schema is implemented as a dataclass with explicit validators. We avoid
Pydantic as a hard dependency to keep the META layer importable in any
context (pyav-only, CPU-only, archive-builder-only). The validation API is
"Pydantic-style" — field validators raise ``SubstrateContractError`` with a
field-level error message at construction time.

Per CLAUDE.md "Beauty, simplicity, and developer experience — non-negotiable"
the API is narrow: 36 canonical fields, one error class, one factory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

__all__ = [
    "KNOWN_CATALOG_COMPLIANCE_TOKENS",
    "LEGAL_CANARY_STATUS",
    "LEGAL_DEPLOYMENT_TARGETS",
    "LEGAL_EXPORT_FORMATS",
    "LEGAL_GPU_KEY",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_MIN_SMOKE_GPU",
    "LEGAL_OPERATIONAL_STATUS",
    "LEGAL_PLATFORM_KEY",
    "LEGAL_PYAV_DECODE_STRATEGY",
    "LEGAL_SCORE_AWARE_LOSS",
    "LEGAL_TARGET_MODES",
    "LEGAL_VIDEO_INPUT_STRATEGY",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "SubstrateContract",
    "SubstrateContractError",
]


class SubstrateContractError(ValueError):
    """Raised at decoration time when a SubstrateContract is invalid."""


# ---------------------------------------------------------------------------
# Enum-like legal value sets. Stored as frozensets for O(1) membership
# (per the MACKAY-2 immutability hygiene pattern that auto-closed alongside
# Catalog #233 token sets — see Catalog #239 entry).
# ---------------------------------------------------------------------------

LEGAL_TARGET_MODES: frozenset[str] = frozenset(
    {
        "contest_exact_eval",
        "contest_one_video_replay",
        "contest_generalized",
        "production_generalized",
        "production_edge_adaptive",
        "research_substrate",
    }
)

LEGAL_DEPLOYMENT_TARGETS: frozenset[str] = frozenset(
    {
        "t4_contest_runtime",
        "comma_ai_production",
        "openpilot_edge",
        "desktop_research",
        "device_learning_optional",
    }
)

LEGAL_EXPORT_FORMATS: frozenset[str] = frozenset(
    {
        "fp4_brotli",
        "fp16_brotli",
        "int8_arith",
        "int4_lsq",
        "fp4_packed_sorted_keys_ibps1",
        "custom",
    }
)

LEGAL_SCORE_AWARE_LOSS: frozenset[str] = frozenset(
    {
        "scorer_loss_terms_btchw",
        "kl_distill",
        "custom",
    }
)

NOT_APPLICABLE_WITH_RATIONALE = "not_applicable_with_rationale"

LEGAL_OPERATIONAL_STATUS: frozenset[str] = frozenset(
    {
        "OPERATIONAL",
        "PRE_BUILD_SUBSTRATE_ENGINEERING",
        "RESEARCH_ONLY",
        "SCAFFOLD_DEFERRED_INTEGRATION",
    }
)

LEGAL_MIN_SMOKE_GPU: frozenset[str] = frozenset(
    {"T4", "L4", "A10G", "L40S", "A100", "H100"}
)

LEGAL_PYAV_DECODE_STRATEGY: frozenset[str] = frozenset(
    {
        "cpu_thread_async_upload",
        "cuda_nvdec",
        "cpu_blocking_upload",
        "not_applicable",
    }
)

LEGAL_CANARY_STATUS: frozenset[str] = frozenset(
    {"canary", "post_canary_dependent", "independent_substrate"}
)

LEGAL_VIDEO_INPUT_STRATEGY: frozenset[str] = frozenset(
    {
        "per_dispatch_local_copy",
        "readonly_mmap",
        "shared_volume_no_contention_expected",
    }
)

LEGAL_HOOK_SENSITIVITY: frozenset[str] = frozenset(
    {
        "scorer_conditional_entropy_map_v1",
        "mdl_density_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_PARETO: frozenset[str] = frozenset(
    {
        "rate_distortion_v1",
        "cost_band_envelope_v1",
        "custom",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_BIT_ALLOCATOR: frozenset[str] = frozenset(
    {
        "per_tensor_uniform",
        "per_channel_lsq",
        "ibps_kkt",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_HOOK_CONTINUAL_LEARNING: frozenset[str] = frozenset(
    {
        "paired_axis",
        "cuda_only",
        "cpu_only",
        "macos_cpu_advisory",
        NOT_APPLICABLE_WITH_RATIONALE,
    }
)

LEGAL_GPU_KEY: frozenset[str] = LEGAL_MIN_SMOKE_GPU
LEGAL_PLATFORM_KEY: frozenset[str] = frozenset(
    {"modal", "vastai", "lightning", "kaggle"}
)

# Known catalog-compliance tokens. Unknown tokens warn but do not refuse
# (substrates can declare new catalog hooks as they land).
KNOWN_CATALOG_COMPLIANCE_TOKENS: frozenset[str] = frozenset(
    {
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_153_canonical_mount_builder_used",
        "catalog_163_remote_lane_sentinel_used",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_165_modal_mtime_stability_check_active",
        "catalog_166_source_parity_honored",
        "catalog_167_smoke_before_full_routed",
        "catalog_170_min_vram_gb_declared",
        "catalog_171_video_input_strategy_declared",
        "catalog_172_autocast_fp16_declared",
        "catalog_173_canary_status_declared",
        "catalog_178_tf32_supported",
        "catalog_181_pyav_decode_strategy_declared",
        "catalog_182_target_modes_declared",
        "catalog_191_modal_sentinel_files_threaded",
        "catalog_205_select_inflate_device_used",
        "catalog_215_min_smoke_gpu_consistent",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    }
)


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LANE_ID_PATTERN = re.compile(r"^lane_[a-z0-9_]+_\d{8}$")


@dataclass(frozen=True)
class SubstrateContract:
    """Canonical contract declared by every substrate via ``@register_substrate``.

    See `.omx/research/substrate_meta_layer_design_20260515.md` for the full
    field-by-field rationale and the failure-mode taxonomy.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": the contract
    is a single dataclass with field-level validators, no inheritance, no
    hidden state. Frozen (immutable) so the registered contract cannot drift
    after import.
    """

    # 2.1 Identity & lifecycle (5)
    id: str
    lane_id: str
    target_modes: tuple[str, ...]
    deployment_target: str
    council_verdict_provenance: str | None

    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar: str
    parser_section_manifest: Mapping[str, str]
    inflate_runtime_loc_budget: int
    runtime_dep_closure: tuple[str, ...]
    export_format: str
    score_aware_loss: str
    bolt_on_loc_budget: int
    no_op_detector_planned: bool

    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added: str | None
    score_improvement_mechanism_status: str
    runtime_overlay_consumed: bool

    # 2.4 Recipe schema (8)
    recipe_smoke_only: bool
    recipe_research_only: bool
    recipe_min_smoke_gpu: str
    recipe_min_vram_gb: int
    recipe_pyav_decode_strategy: str
    recipe_canary_status: str
    recipe_video_input_strategy: str
    recipe_canary_dependency: str | None

    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs: int
    cost_band_gpu_key: str
    cost_band_platform_key: str
    cost_band_p50_usd: float

    # 2.6 6-hook wire-in (6 per Catalog #125)
    hook_sensitivity_contribution: str
    hook_pareto_constraint: str
    hook_bit_allocator_class: str
    hook_autopilot_ranker_class_shift_token: str | None
    hook_continual_learning_anchor_kind: str
    hook_probe_disambiguator: str | None

    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations: tuple[str, ...] = field(default_factory=tuple)
    hook_not_applicable_rationale: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Identity
        if not isinstance(self.id, str) or not _ID_PATTERN.match(self.id):
            raise SubstrateContractError(
                f"id={self.id!r} must match /^[a-z][a-z0-9_]*$/"
            )
        if not isinstance(self.lane_id, str) or not _LANE_ID_PATTERN.match(self.lane_id):
            raise SubstrateContractError(
                f"lane_id={self.lane_id!r} must match /^lane_[a-z0-9_]+_\\d{{8}}$/"
            )
        if not self.target_modes:
            raise SubstrateContractError("target_modes must be non-empty")
        for mode in self.target_modes:
            if mode not in LEGAL_TARGET_MODES:
                raise SubstrateContractError(
                    f"target_modes contains illegal value {mode!r}; legal={sorted(LEGAL_TARGET_MODES)}"
                )
        if self.deployment_target not in LEGAL_DEPLOYMENT_TARGETS:
            raise SubstrateContractError(
                f"deployment_target={self.deployment_target!r} not in {sorted(LEGAL_DEPLOYMENT_TARGETS)}"
            )

        # Architecture
        if not self.archive_grammar:
            raise SubstrateContractError("archive_grammar must be non-empty")
        if not isinstance(self.parser_section_manifest, Mapping):
            raise SubstrateContractError("parser_section_manifest must be a mapping")
        if self.inflate_runtime_loc_budget <= 0:
            raise SubstrateContractError("inflate_runtime_loc_budget must be > 0")
        if not self.runtime_dep_closure:
            raise SubstrateContractError("runtime_dep_closure must list at least one dep")
        if self.export_format not in LEGAL_EXPORT_FORMATS:
            raise SubstrateContractError(
                f"export_format={self.export_format!r} not in {sorted(LEGAL_EXPORT_FORMATS)}"
            )
        if self.score_aware_loss not in LEGAL_SCORE_AWARE_LOSS:
            raise SubstrateContractError(
                f"score_aware_loss={self.score_aware_loss!r} not in {sorted(LEGAL_SCORE_AWARE_LOSS)}"
            )
        if self.bolt_on_loc_budget <= 0:
            raise SubstrateContractError("bolt_on_loc_budget must be > 0")
        if not isinstance(self.no_op_detector_planned, bool):
            raise SubstrateContractError("no_op_detector_planned must be bool")

        # Operational mechanism (Catalog #220 mirror)
        if self.score_improvement_mechanism_status not in LEGAL_OPERATIONAL_STATUS:
            raise SubstrateContractError(
                f"score_improvement_mechanism_status={self.score_improvement_mechanism_status!r} "
                f"not in {sorted(LEGAL_OPERATIONAL_STATUS)}"
            )
        if (
            self.score_improvement_mechanism_status == "OPERATIONAL"
            and not self.runtime_overlay_consumed
        ):
            raise SubstrateContractError(
                "score_improvement_mechanism_status=OPERATIONAL requires runtime_overlay_consumed=True "
                "(Catalog #220 invariant)"
            )
        if (
            self.score_improvement_mechanism_status != "OPERATIONAL"
            and self.runtime_overlay_consumed
        ):
            raise SubstrateContractError(
                "runtime_overlay_consumed=True requires score_improvement_mechanism_status=OPERATIONAL"
            )
        if (
            self.archive_bytes_added is not None
            and self.score_improvement_mechanism_status
            == "SCAFFOLD_DEFERRED_INTEGRATION"
        ):
            # Mirror of Catalog #220: a SCAFFOLD lane that adds bytes >1 KB
            # without an operational mechanism is the research-substrate trap.
            kb_match = re.search(r"(\d+)\s*KB", self.archive_bytes_added)
            if kb_match and int(kb_match.group(1)) > 1:
                raise SubstrateContractError(
                    f"archive_bytes_added={self.archive_bytes_added!r} declares >1 KB "
                    "but score_improvement_mechanism_status=SCAFFOLD_DEFERRED_INTEGRATION "
                    "(Catalog #220 forbidden anti-pattern). Use OPERATIONAL or RESEARCH_ONLY."
                )

        # Recipe
        if not isinstance(self.recipe_smoke_only, bool):
            raise SubstrateContractError("recipe_smoke_only must be bool")
        if not isinstance(self.recipe_research_only, bool):
            raise SubstrateContractError("recipe_research_only must be bool")
        if self.recipe_min_smoke_gpu not in LEGAL_MIN_SMOKE_GPU:
            raise SubstrateContractError(
                f"recipe_min_smoke_gpu={self.recipe_min_smoke_gpu!r} not in {sorted(LEGAL_MIN_SMOKE_GPU)}"
            )
        if self.recipe_min_vram_gb < 1:
            raise SubstrateContractError("recipe_min_vram_gb must be >= 1")
        if self.recipe_pyav_decode_strategy not in LEGAL_PYAV_DECODE_STRATEGY:
            raise SubstrateContractError(
                f"recipe_pyav_decode_strategy={self.recipe_pyav_decode_strategy!r} not in "
                f"{sorted(LEGAL_PYAV_DECODE_STRATEGY)}"
            )
        if self.recipe_canary_status not in LEGAL_CANARY_STATUS:
            raise SubstrateContractError(
                f"recipe_canary_status={self.recipe_canary_status!r} not in {sorted(LEGAL_CANARY_STATUS)}"
            )
        if self.recipe_video_input_strategy not in LEGAL_VIDEO_INPUT_STRATEGY:
            raise SubstrateContractError(
                f"recipe_video_input_strategy={self.recipe_video_input_strategy!r} not in "
                f"{sorted(LEGAL_VIDEO_INPUT_STRATEGY)}"
            )
        if self.recipe_canary_status == "post_canary_dependent" and not self.recipe_canary_dependency:
            raise SubstrateContractError(
                "recipe_canary_status=post_canary_dependent requires recipe_canary_dependency"
            )
        if self.recipe_canary_status != "post_canary_dependent" and self.recipe_canary_dependency:
            raise SubstrateContractError(
                "recipe_canary_dependency only legal when recipe_canary_status=post_canary_dependent"
            )
        if self.recipe_smoke_only and self.cost_band_epochs > 100:
            raise SubstrateContractError(
                f"recipe_smoke_only=True is inconsistent with cost_band_epochs={self.cost_band_epochs} > 100; "
                "either flip recipe_smoke_only OR drop epochs to a smoke band"
            )

        # Cost band
        if self.cost_band_epochs < 1:
            raise SubstrateContractError("cost_band_epochs must be >= 1")
        if self.cost_band_gpu_key not in LEGAL_GPU_KEY:
            raise SubstrateContractError(
                f"cost_band_gpu_key={self.cost_band_gpu_key!r} not in {sorted(LEGAL_GPU_KEY)}"
            )
        if self.cost_band_platform_key not in LEGAL_PLATFORM_KEY:
            raise SubstrateContractError(
                f"cost_band_platform_key={self.cost_band_platform_key!r} not in {sorted(LEGAL_PLATFORM_KEY)}"
            )
        if self.cost_band_p50_usd < 0.0:
            raise SubstrateContractError("cost_band_p50_usd must be >= 0.0")

        # 6-hook wire-in
        hook_validators = {
            "hook_sensitivity_contribution": (self.hook_sensitivity_contribution, LEGAL_HOOK_SENSITIVITY),
            "hook_pareto_constraint": (self.hook_pareto_constraint, LEGAL_HOOK_PARETO),
            "hook_bit_allocator_class": (self.hook_bit_allocator_class, LEGAL_HOOK_BIT_ALLOCATOR),
            "hook_continual_learning_anchor_kind": (
                self.hook_continual_learning_anchor_kind,
                LEGAL_HOOK_CONTINUAL_LEARNING,
            ),
        }
        for fname, (value, legal) in hook_validators.items():
            if value not in legal:
                raise SubstrateContractError(
                    f"{fname}={value!r} not in {sorted(legal)}"
                )
            if value == NOT_APPLICABLE_WITH_RATIONALE:
                rationale = self.hook_not_applicable_rationale.get(fname)
                if not rationale or not rationale.strip():
                    raise SubstrateContractError(
                        f"{fname}=not_applicable_with_rationale requires a non-empty entry in "
                        f"hook_not_applicable_rationale[{fname!r}]"
                    )
        # Hook #6 probe-disambiguator: if None, demand a rationale; if a string, must look like a path.
        if self.hook_probe_disambiguator is None:
            rationale = self.hook_not_applicable_rationale.get("hook_probe_disambiguator")
            if not rationale or not rationale.strip():
                raise SubstrateContractError(
                    "hook_probe_disambiguator=None requires entry in "
                    "hook_not_applicable_rationale['hook_probe_disambiguator']"
                )
        else:
            if not isinstance(self.hook_probe_disambiguator, str) or not self.hook_probe_disambiguator.strip():
                raise SubstrateContractError("hook_probe_disambiguator must be a non-empty path or None")

        # Hook #4: token may be None (within-class) or non-empty string.
        if self.hook_autopilot_ranker_class_shift_token is not None and not isinstance(
            self.hook_autopilot_ranker_class_shift_token, str
        ):
            raise SubstrateContractError(
                "hook_autopilot_ranker_class_shift_token must be str or None"
            )

        # Hook-rationale dict keys must match hook field names if present.
        legal_rationale_keys = {
            "hook_sensitivity_contribution",
            "hook_pareto_constraint",
            "hook_bit_allocator_class",
            "hook_continual_learning_anchor_kind",
            "hook_probe_disambiguator",
        }
        for k in self.hook_not_applicable_rationale:
            if k not in legal_rationale_keys:
                raise SubstrateContractError(
                    f"hook_not_applicable_rationale has illegal key {k!r}; legal={sorted(legal_rationale_keys)}"
                )

        # ------------------------------------------------------------------
        # Adversarial-review cross-field invariants (2026-05-15).
        # See ``feedback_meta_layer_adversarial_review_round_1_2_landed_20260515.md``
        # for the per-finding rationale.
        # ------------------------------------------------------------------

        # Finding F1 (Fridrich MEDIUM): a contract that claims a monolithic
        # archive grammar but lists multiple parser sections is internally
        # inconsistent. Detect by token + section count.
        if (
            "monolithic" in self.archive_grammar.lower()
            and "single_file" in self.archive_grammar.lower()
            and len(self.parser_section_manifest) > 1
        ):
            # Allow the substrate to declare logical sections within the SAME
            # monolithic file (e.g. header / weights / latents) — the rule
            # fires only when the section *names* look like distinct files
            # (suffixes ``.bin`` / ``.pt`` / ``.zip`` / ``.mkv`` / ``.br``).
            file_like = [
                k
                for k in self.parser_section_manifest
                if isinstance(k, str)
                and any(
                    k.lower().endswith(ext)
                    for ext in (".bin", ".pt", ".zip", ".mkv", ".br", ".mp4", ".pkl")
                )
            ]
            if len(file_like) > 1:
                raise SubstrateContractError(
                    f"archive_grammar={self.archive_grammar!r} claims monolithic single-file "
                    f"but parser_section_manifest declares multiple file-like sections "
                    f"{file_like!r}. Per HNeRV parity discipline lesson 3 a monolithic "
                    "single-file archive must not enumerate distinct file members. Either "
                    "rename the sections to logical labels (header/weights/latents) or "
                    "change archive_grammar to a multi-file token."
                )

        # Finding M1 (MacKay MEDIUM): cost_band_gpu_key must be at least as
        # capable as recipe_min_smoke_gpu, otherwise the cost-band budgets
        # for a class the smoke literally cannot run on. Order is the
        # observed Modal/Vast.ai capability ladder: T4 < L4 < A10G < L40S < A100 < H100.
        _GPU_CAPABILITY_RANK = {
            "T4": 0, "L4": 1, "A10G": 2, "L40S": 3, "A100": 4, "H100": 5,
        }
        full_rank = _GPU_CAPABILITY_RANK.get(self.cost_band_gpu_key, -1)
        smoke_rank = _GPU_CAPABILITY_RANK.get(self.recipe_min_smoke_gpu, -1)
        if full_rank >= 0 and smoke_rank >= 0 and full_rank < smoke_rank:
            raise SubstrateContractError(
                f"cost_band_gpu_key={self.cost_band_gpu_key!r} is cheaper than "
                f"recipe_min_smoke_gpu={self.recipe_min_smoke_gpu!r}. The cost band "
                "cannot budget for a class less capable than the smoke requires; the "
                "smoke would never run. Promote cost_band_gpu_key OR demote "
                "recipe_min_smoke_gpu so cost_band >= smoke."
            )

        # Finding Q1 (Quantizr LOW): catalog_compliance_declarations is
        # advisory metadata; substrates can declare new catalog hooks as they
        # land. We do NOT refuse unknown tokens (that would race with the
        # canonical catalog table updates). But we raise when the tuple
        # contains a value that is not a string, since that is always wrong.
        for tok in self.catalog_compliance_declarations:
            if not isinstance(tok, str) or not tok.strip():
                raise SubstrateContractError(
                    f"catalog_compliance_declarations contains non-string or empty entry: {tok!r}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (sorted keys for byte-stable downstream use)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Mapping):
                out[f.name] = dict(v)
            elif isinstance(v, tuple):
                out[f.name] = list(v)
            else:
                out[f.name] = v
        return out
