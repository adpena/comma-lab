# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260526_path_3_g_nirvana_cascading_nerv_hierarchical_residual_decoder_cascade_fresh_design_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_and_substrate_contract_canonical_helper_adoption_when_nirvana1_archive_grammar_per_level_residual_blob_field_is_added_to_contract_schema
"""nirvana_cascading_nerv — NIRVANA cascading NeRV hierarchical residual decoder cascade (L0 SCAFFOLD).

Path 3 candidate #G per operator directive 2026-05-26 verbatim:
*"the MLX first requirement might also force us out of the issue we were
having before where we had great ideas but we're building them as Boltons
to the same substrates over and over again; we want to design the
substrate and curriculum and then optimize the design the whole stack
around it for extreme optimization and performance and optimal score
lowering"*

NEW operator binding directive 2026-05-26 directive #3 verbatim:
*"we also need adversarial review against all landing recursive for math
and scientific and engineering rigor and for MLX drift minimization and
portability via numpy"*

This substrate is a FRESH DESIGN, NOT extension of existing
``src/tac/substrates/nirvana/`` (which is Maiya CVPR 2024 patch-wise +
adaptive scheduling — a paradigm-DISTINCT substrate). The shared
"NIRVANA" name reflects paradigm-family-level kinship (neural implicit
video rendering with adaptive components) but the architectural class
is structurally distinct.

Paradigm anchor: NIRVANA-style cascading NeRV with hierarchical residual
decoder cascade. Each layer in the decoder produces a coarse estimate
and the residual to a finer estimate; the next layer refines via
wavelet-pyramid-style upsampling + residual addition.

Distinct from sister candidates by decomposition principle:
- A=DreamerV3 RSSM: categorical latent dynamics (G×K group-categorical alphabet)
- E=BoostNeRV-against-PR110: iterative boosting (frozen-base + iterative residual)
- G=NIRVANA cascading NeRV (this): hierarchical residual decoder cascade
  (multi-scale wavelet-pyramid decoder with per-level residual; single
  learned model trained end-to-end)
- F=Z8 hierarchical predictive coding: canonical quadruple (Rao-Ballard
  + Mallat + Hafner + Wyner-Ziv)

Architecture (council-approved L0 SCAFFOLD 2026-05-26):

    Per-pair latent z in R^16
       |
       v
    Level 0 decoder: produces RGB at 48×64 (base coarse estimate)
       |
       +--> store base_rgb
       v
    Bilinear upsample ×2 → 96×128
       |
       v
    Level 1 residual decoder: produces residual_1 at 96×128
       |
       +--> store residual_1 (int8 quantized for archive)
       v
    base_rgb (upsampled) + residual_1 → rgb_at_96×128
       |
       v
    Bilinear upsample ×2 → 192×256
       |
       v
    Level 2 residual decoder: produces residual_2 at 192×256
       |
       +--> store residual_2 (int8 quantized for archive)
       v
    rgb_at_96×128 (upsampled) + residual_2 → rgb_at_192×256
       |
       v
    Bilinear upsample ×2 → 384×512
       |
       v
    Level 3 residual decoder: produces residual_3 at 384×512
       |
       +--> store residual_3 (int8 quantized for archive)
       v
    rgb_at_192×256 (upsampled) + residual_3 → final_rgb at 384×512

Archive grammar (NIRVANA1):
    27-byte header: magic b"NIR1\\x00" (5) + version u8 (1) + NUM_LEVELS u8 (1) +
        PER_PAIR_LATENT_DIM u8 (1) + BASE_H u16 (2) + BASE_W u16 (2) +
        DECODER_BLOB_LEN u32 (4) + RESIDUAL_BLOB_LEN u32 (4) +
        LATENTS_BLOB_LEN u32 (4) + META_BLOB_LEN u32 (4)
    DECODER_BLOB_LEN bytes: brotli(q=9) compressed per-level decoder
        state_dict (fp16; canonical sister substrate pattern)
    RESIDUAL_BLOB_LEN bytes: brotli(q=9) compressed per-level int8
        quantized residuals (4 levels concatenated with per-level length
        prefix)
    LATENTS_BLOB_LEN bytes: brotli(q=9) compressed per-pair latent z
        (int16 quantized)
    META_BLOB_LEN bytes: sorted-keys JSON utf-8 (gumbel_temperature,
        num_pairs, eval_hw, num_levels, ...)

CLAUDE.md compliance:
- No silent device defaults (MLX explicit; PyTorch export path uses canonical
  ``tac.substrates._shared.inflate_runtime.select_inflate_device`` per Catalog #205)
- No scorer load at inflate time (only per-level decoder forward + bilinear
  upsample + int8 dequantize + residual addition + uint8 cast)
- No /tmp paths in persisted artifacts (Catalog #113 forbidden-path)
- Every file reviewable in 30 seconds per HNeRV parity L12
- ``_full_main`` raises NotImplementedError per Catalog #240 L0 SCAFFOLD posture
- numpy reference implementation per axis 3 portability (operator directive #3)
- MLX↔PyTorch + MLX↔numpy parity tests per axis 2 drift minimization

See ``.omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md``
for the full design memo with Catalog #290 canonical-vs-unique decisions per
layer, Catalog #294 9-dim checklist, Catalog #296 predicted-band Dykstra-
feasibility check + Shannon R(D) bound, Catalog #303 cargo-cult audit per
assumption, Catalog #305 observability surface, Catalog #309 horizon_class
declaration, and the NEW operator-directive-#3 sections:
- "Math + scientific + engineering rigor per layer" (10/10 HARD-EARNED)
- "MLX drift minimization per primitive" (7 primitives, 3 KNOWN-DRIFT-RISK)
- "Portability via numpy per primitive" (7/7 numpy reference implementations)
"""

from __future__ import annotations

# Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them).
# DO NOT remove without operator review per Catalog #229 premise verification.
ARCHIVE_GRAMMAR_FIELDS: dict[str, str] = {
    "archive_grammar": "monolithic_single_file_0bin_nirvana1",
    "parser_section_manifest": (
        "nirvana1_header + decoder_blob + residual_blob + latents_blob + meta_blob"
    ),
    "inflate_runtime_loc_budget": (
        "le_200_loc_substrate_engineering_waiver_per_hnerv_parity_l7"
    ),
    "runtime_dep_closure": "torch_brotli_only_per_hnerv_parity_l4",
    "export_format": "nirvana1_monolithic_single_zip_member_0bin",
    "score_aware_loss": (
        "pending_phase_2_trainer_canonical_helper_score_pair_components_catalog_164"
    ),
    "bolt_on_loc_budget": (
        "substrate_engineering_lane_class_per_hnerv_parity_l7_waiver"
    ),
    "no_op_detector_planned": (
        "byte_mutation_gate_catalog_139_plus_105_plus_272_distinguishing_feature_"
        "per_level_residual_blob"
    ),
}

# Archive grammar constants (Catalog #146 inflate runtime contract; fixed
# offsets declared in source per HNeRV parity L3 monolithic-single-file 0.bin
# pattern).
ARCHIVE_MAGIC: bytes = b"NIR1\x00"  # 5 bytes: NIR=NIRVANA, 1=v1
"""NIRVANA cascading NeRV variant 1 archive magic."""

ARCHIVE_VERSION: int = 1  # u8
"""Schema version byte. Bump when grammar changes."""

# Header layout per __init__ docstring:
# MAGIC(5) + VERSION(1) + NUM_LEVELS(1) + PER_PAIR_LATENT_DIM(1) +
# BASE_H(2) + BASE_W(2) + DECODER_LEN(4) + RESIDUAL_LEN(4) +
# LATENTS_LEN(4) + META_LEN(4) = 5+1+1+1+2+2+4+4+4+4 = 28 bytes
NIRVANA1_HEADER_FMT: str = "<5sBBBHHIIII"
NIRVANA1_HEADER_LEN: int = 28

# Default config (CARGO-CULTED at L0 per Catalog #303 audit in design memo).
DEFAULT_NUM_LEVELS: int = 4
"""Default wavelet-pyramid depth (48×64 → 96×128 → 192×256 → 384×512)."""

DEFAULT_PER_PAIR_LATENT_DIM: int = 16
"""Default per-pair latent dimension (NeRV-family canonical)."""

DEFAULT_BASE_H: int = 48
"""Default base coarse-level height (level 0 decoder output)."""

DEFAULT_BASE_W: int = 64
"""Default base coarse-level width (level 0 decoder output)."""

# Sister substrate citation: paradigm-distinct from sister substrates per
# Catalog #290 §"Canonical-vs-unique decision per layer".
SISTER_SUBSTRATES: tuple[str, ...] = (
    "dreamer_v3_rssm",            # A: categorical latent dynamics
    "boost_nerv_pr110_residual",  # E: iterative boosting vs frozen base
    # F (Z8 hierarchical predictive coding) — queued; not yet landed
)


# Lazy import to keep top-level import cheap (MLX may not be installed on
# every consumer machine; the architecture module pulls MLX lazily as well).
def _load_config():
    """Lazy import escape hatch for NirvanaCascadingNervConfig."""
    from .mlx_renderer import NirvanaCascadingNervConfig as _Config

    return _Config


# Sentinel re-export via __getattr__ so `from tac.substrates.nirvana_cascading_nerv import NirvanaCascadingNervConfig`
# works without forcing MLX at top-level import time.
def __getattr__(name: str):  # noqa: D401 — module-level escape
    if name == "NirvanaCascadingNervConfig":
        return _load_config()
    raise AttributeError(
        f"module 'nirvana_cascading_nerv' has no attribute {name!r}"
    )


# Public API surface. Catalog #335 contract for cathedral consumer auto-
# discovery: narrow + explicit per HNeRV parity L12.
__all__ = [
    "ARCHIVE_GRAMMAR_FIELDS",
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "DEFAULT_BASE_H",
    "DEFAULT_BASE_W",
    "DEFAULT_NUM_LEVELS",
    "DEFAULT_PER_PAIR_LATENT_DIM",
    "NIRVANA1_HEADER_FMT",
    "NIRVANA1_HEADER_LEN",
    "NirvanaCascadingNervConfig",
    "SISTER_SUBSTRATES",
]
