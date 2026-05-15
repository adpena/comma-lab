# SPDX-License-Identifier: MIT
"""Auto-generate ``substrate_<id>_modal_<gpu>_dispatch.yaml`` from a contract.

Pure function: ``generate_recipe_yaml(contract) -> str`` returns YAML bytes
suitable for writing to
``.omx/operator_authorize_recipes/substrate_<id>_modal_<gpu>_dispatch.yaml``.

The generator emits the canonical schema that ``tools/operator_authorize.py``
+ ``tools/run_modal_smoke_before_full.py`` consume. Generated recipes honor:

  - Catalog #170 (``min_vram_gb``)
  - Catalog #171 (``video_input_strategy``)
  - Catalog #173 (``canary_status``)
  - Catalog #181 (``pyav_decode_strategy``)
  - Catalog #182 (``target_modes``)
  - Catalog #215 (``min_smoke_gpu`` consistent with full GPU class)

Per CLAUDE.md "Beauty, simplicity, and developer experience" the output is
deterministic byte-for-byte from the contract — no clock, no random ids, no
host-machine specifics.

This module is READ-ONLY for existing artifacts; the migration subagent
diffs ``generate_recipe_yaml(extracted_contract)`` against the existing
recipe to surface drift.
"""

from __future__ import annotations

from tac.substrate_registry.contract import SubstrateContract

__all__ = [
    "generate_recipe_yaml",
    "default_recipe_relpath",
]


def default_recipe_relpath(contract: SubstrateContract) -> str:
    """Canonical recipe path under ``.omx/operator_authorize_recipes/``."""
    return (
        f".omx/operator_authorize_recipes/substrate_{contract.id}_"
        f"{contract.cost_band_platform_key}_{contract.cost_band_gpu_key.lower()}_dispatch.yaml"
    )


def _yaml_str(value: str | None) -> str:
    """Render a YAML scalar string with quoting suitable for downstream parsers."""
    if value is None:
        return "null"
    if "\n" in value:
        # Use literal block scalar
        indented = "\n".join("  " + line for line in value.split("\n"))
        return "|\n" + indented
    if any(ch in value for ch in (":", "#", "{", "}", "[", "]", "&", "*", "!", "|", ">", "%", "@", "`", "\"", "'")):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f"\"{escaped}\""
    return value


def _yaml_list(values: tuple[str, ...] | list[str]) -> str:
    if not values:
        return "[]"
    lines = [""]
    for v in values:
        lines.append(f"  - {_yaml_str(v)}")
    return "\n".join(lines)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def generate_recipe_yaml(contract: SubstrateContract) -> str:
    """Emit canonical recipe YAML bytes for ``contract``.

    Output is deterministic: same contract → same bytes.
    """
    sentinel_files = (
        f"src/tac/substrates/{contract.id}/__init__.py",
        f"src/tac/substrates/{contract.id}/architecture.py",
        f"src/tac/substrates/{contract.id}/archive.py",
        f"src/tac/substrates/{contract.id}/inflate.py",
        "src/tac/substrates/_shared/trainer_skeleton.py",
        "src/tac/substrates/_shared/inflate_runtime.py",
        "src/tac/substrates/score_aware_common.py",
    )
    canary_dependency_field = (
        f"\ncanary_dependency: {_yaml_str(contract.recipe_canary_dependency)}"
        if contract.recipe_canary_status == "post_canary_dependent"
        else ""
    )

    lines: list[str] = [
        "schema_version: 1",
        f"name: substrate_{contract.id}_{contract.cost_band_platform_key}_"
        f"{contract.cost_band_gpu_key.lower()}_dispatch",
        f"lane_id: {contract.lane_id}",
        "summary: |",
        f"  Auto-generated from SubstrateContract via",
        f"  src/tac/substrate_registry/recipe_generator.py.",
        "",
        f"  Substrate id: {contract.id}",
        f"  Archive grammar: {contract.archive_grammar}",
        f"  Council verdict provenance: {contract.council_verdict_provenance or 'N/A'}",
        f"  Operational status: {contract.score_improvement_mechanism_status}",
        "",
        f"platform: {contract.cost_band_platform_key}",
        f"gpu: \"{contract.cost_band_gpu_key}\"",
        f"min_vram_gb: {contract.recipe_min_vram_gb}",
        f"min_smoke_gpu: \"{contract.recipe_min_smoke_gpu}\"",
        f"video_input_strategy: {contract.recipe_video_input_strategy}",
        f"pyav_decode_strategy: {contract.recipe_pyav_decode_strategy}",
        "target_modes:" + _yaml_list(contract.target_modes),
        f"canary_status: {contract.recipe_canary_status}{canary_dependency_field}",
        "",
        f"smoke_only: {_bool_str(contract.recipe_smoke_only)}",
        f"research_only: {_bool_str(contract.recipe_research_only)}",
        "",
        "cost_band:",
        f"  epochs: {contract.cost_band_epochs}",
        "  all_flags_on: true",
        f"  hand_calibrated_fallback_p50_usd: {contract.cost_band_p50_usd:.2f}",
        f"  platform_key: {contract.cost_band_platform_key}",
        f"  gpu_key: {contract.cost_band_gpu_key}",
        "",
        f"remote_driver: scripts/remote_lane_substrate_{contract.id}.sh",
        "timeout_hours: 4.0",
        "",
        "required_input_files:",
        "  - flag: --video-path",
        "    default_path: upstream/videos/0.mkv",
        "",
        "modal:",
        f"  lane_script: scripts/remote_lane_substrate_{contract.id}.sh",
        f"  cost_band_trainer: experiments/train_substrate_{contract.id}.py",
        f"  cost_band_epochs: {contract.cost_band_epochs}",
        "  cost_band_all_flags_on: true",
        "",
        f"required_input_files_trainer: experiments/train_substrate_{contract.id}.py",
        "",
        "# Per Catalog #191 (sentinel-files-per-Catalog-#166) source-parity sentinels.",
        "sentinel_files:" + _yaml_list(sentinel_files),
        "",
        "catalog_compliance_declarations:" + _yaml_list(contract.catalog_compliance_declarations),
        "",
        "# 6-hook Catalog #125 wire-in declarations (auto-injected from contract).",
        f"hook_sensitivity_contribution: {contract.hook_sensitivity_contribution}",
        f"hook_pareto_constraint: {contract.hook_pareto_constraint}",
        f"hook_bit_allocator_class: {contract.hook_bit_allocator_class}",
        f"hook_autopilot_ranker_class_shift_token: {_yaml_str(contract.hook_autopilot_ranker_class_shift_token)}",
        f"hook_continual_learning_anchor_kind: {contract.hook_continual_learning_anchor_kind}",
        f"hook_probe_disambiguator: {_yaml_str(contract.hook_probe_disambiguator)}",
        "",
        "notes: |",
        "  This recipe was auto-generated from a SubstrateContract declared",
        f"  in experiments/train_substrate_{contract.id}.py via",
        "  @register_substrate(...). DO NOT EDIT BY HAND — regenerate via",
        "  the recipe_generator if the contract changes.",
        "",
        "  Any drift between this recipe and the substrate's declared",
        "  contract will be caught by Catalog #242",
        "  (check_register_substrate_contract_fields_canonical) at preflight",
        "  time.",
    ]
    return "\n".join(lines) + "\n"
