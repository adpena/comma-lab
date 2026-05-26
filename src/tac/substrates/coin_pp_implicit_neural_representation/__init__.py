# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260526_path_3_k_coin_pp_implicit_neural_representation_meta_learned_modulated_coord_mlp_fresh_design_per_operator_directives_substrate_design_not_bolt_on_plus_cargo_cult_pass_plus_3_axis_recursive_adversarial_review_discipline_mlx_first_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_and_substrate_contract_canonical_helper_adoption_when_coinpp1_archive_grammar_per_pair_modulation_blob_field_is_added_to_contract_schema
"""coin_pp_implicit_neural_representation — COIN++ meta-learned modulated coord-MLP (L0 SCAFFOLD).

Path 3 candidate #K per operator directives 2026-05-26 verbatim:

*"The MLX first requirement might also force us out of the issue we were
having before where we had great ideas but we're building them as Boltons
to the same substrates over and over again; we want to design the
substrate and curriculum and then optimize the design the whole stack
around it for extreme optimization and performance and optimal score
lowering"*

*"Never simply extend unless a rigorous adversarial cargo cult pass has
been done first"*

*"we also need adversarial review against all landing recursive for math
and scientific and engineering rigor and for MLX drift minimization and
portability via numpy"*

This substrate is a FRESH DESIGN FROM FIRST PRINCIPLES (2-phase
methodology) honoring all three 2026-05-26 binding directives at design
time. NOT a bolt-on extension of sister ``src/tac/substrates/coin_plus_plus/``
(2026-05-20 prior sketch by sister subagent; predates MLX-first + 3-axis
discipline; preserved per Catalog #110/#113 HISTORICAL_PROVENANCE).

Paradigm anchor: Dupont, Loya, Bronstein 2021/2022 "COIN++: Neural
Compression across Modalities" (arXiv:2201.12904 / ICML 2022). Architecture
= meta-learned shared base coord-MLP F_phi(x, y, t) -> rgb + per-pair compact
modulation m_i in R^MOD_DIM via FiLM-style scale+shift (Perez et al. 2017)
on hidden layers. The base MLP is amortized over all pairs; per-pair cost
is O(MOD_DIM) bytes after int8 quantization + brotli compression.

Distinct from sister candidates by decomposition principle:
- A=DreamerV3 RSSM (landed): categorical latent dynamics (G×K alphabet)
- E=BoostNeRV-against-PR110 (landed): iterative boosting vs frozen base
- G=NIRVANA cascading NeRV (in-flight): hierarchical residual decoder cascade
- F=Z8 hierarchical predictive coding (in-flight): Rao-Ballard quadruple
- K=COIN++ (this): meta-learned modulated coord-MLP via FiLM

Architecture (council-approved L0 SCAFFOLD 2026-05-26):

    Per-pair modulation m_i in R^MOD_DIM (default 64; int8 quantized in archive)
       |
       v
    Shared coord-MLP base F_phi:
        Input: (x, y, t) -- normalized pixel coord + frame_index (0 or 1)
            Sinusoidal positional encoding: coord -> R^(POS_DIM * 2 * 3)
        Hidden layers: 3 layers x HIDDEN_DIM=64 with FiLM modulation from m_i:
            h <- sin( linear(h) * scale_i + shift_i )
            scale_i, shift_i = split( linear_film_proj(m_i), HIDDEN_DIM x 2 )
        Output: linear(h) -> R^3 -> sigmoid -> rgb in [0, 1]^3
       |
       v
    For each pixel (x, y) in [0, 384) x [0, 512):
        rgb_t(x, y) = F_phi_mod_m_i(x, y, t)
       |
       v
    Stack into rgb_0, rgb_1: (B, 3, 384, 512)

Archive grammar (COINPP1):

    32-byte header: magic b"CPP1\\x00" (5) + version u8 (1) + MOD_DIM u8 (1) +
        POS_DIM u8 (1) + HIDDEN_DIM u16 (2) + NUM_HIDDEN_LAYERS u8 (1) +
        NUM_PAIRS u16 (2) + EVAL_H u16 (2) + EVAL_W u16 (2) +
        BASE_BLOB_LEN u32 (4) + MOD_BLOB_LEN u32 (4) + META_BLOB_LEN u32 (4) +
        reserved u8 x 3
    BASE_BLOB_LEN bytes: brotli(q=9) compressed base coord-MLP state_dict
        (fp16; canonical sister substrate pattern)
    MOD_BLOB_LEN bytes: brotli(q=9) compressed per-pair int8 modulation
        vectors (num_pairs x MOD_DIM bytes)
    META_BLOB_LEN bytes: sorted-keys JSON utf-8 (modulation_scale, num_pairs,
        eval_hw, schema_version, ...)

CLAUDE.md compliance:
- No silent device defaults (MLX explicit; PyTorch export path uses canonical
  ``tac.substrates._shared.inflate_runtime.select_inflate_device`` per Catalog #205)
- No scorer load at inflate time (only coord-MLP forward + sigmoid + bicubic upscale to camera HW + uint8 cast)
- No /tmp paths in persisted artifacts (Catalog #113 forbidden-path)
- Every file reviewable in 30 seconds per HNeRV parity L12
- ``_full_main`` raises NotImplementedError per Catalog #240 L0 SCAFFOLD posture
- numpy reference implementation per axis 3 portability (operator directive #3)
- MLX↔PyTorch + MLX↔numpy parity tests per axis 2 drift minimization

See ``.omx/research/path_3_k_coin_pp_substrate_design_20260526.md`` for the
full design memo with Catalog #290 canonical-vs-unique decisions per layer,
Catalog #294 9-dim checklist, Catalog #296 predicted-band Shannon R(D)
bound + Dykstra-feasibility commentary, Catalog #303 cargo-cult audit per
assumption, Catalog #305 observability surface, Catalog #309 horizon_class
declaration, and the NEW operator-directive-#3 sections:
- "Math + scientific + engineering rigor per layer" (10/10 HARD-EARNED with
  explicit CARGO-CULTED carve-out for MOD_DIM=64 specific choice)
- "MLX drift minimization per primitive" (9 primitives, 2 MEDIUM-DRIFT-RISK
  mitigated by canonical helpers)
- "Portability via numpy per primitive" (9/9 numpy reference implementations)
"""

from __future__ import annotations

# Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them).
# DO NOT remove without operator review per Catalog #229 premise verification.
ARCHIVE_GRAMMAR_FIELDS: dict[str, str] = {
    "archive_grammar": "monolithic_single_file_0bin_coinpp1",
    "parser_section_manifest": (
        "coinpp1_header + base_blob + modulation_blob + meta_blob"
    ),
    "inflate_runtime_loc_budget": (
        "le_200_loc_substrate_engineering_waiver_per_hnerv_parity_l7"
    ),
    "runtime_dep_closure": "torch_brotli_only_per_hnerv_parity_l4",
    "export_format": "coinpp1_monolithic_single_zip_member_0bin",
    "score_aware_loss": (
        "pending_phase_2_trainer_canonical_helper_score_pair_components_catalog_164"
    ),
    "bolt_on_loc_budget": (
        "substrate_engineering_lane_class_per_hnerv_parity_l7_waiver"
    ),
    "no_op_detector_planned": (
        "byte_mutation_gate_catalog_139_plus_105_plus_272_distinguishing_feature_"
        "per_pair_modulation_blob"
    ),
}

# Archive grammar constants (Catalog #146 inflate runtime contract; fixed
# offsets declared in source per HNeRV parity L3 monolithic-single-file 0.bin
# pattern).
ARCHIVE_MAGIC: bytes = b"CPP1\x00"  # 5 bytes: CPP=COIN++, 1=v1
"""COIN++ implicit neural representation variant 1 archive magic."""

ARCHIVE_VERSION: int = 1  # u8
"""Schema version byte. Bump when grammar changes."""

# Header layout per __init__ docstring:
# MAGIC(5) + VERSION(1) + MOD_DIM(1) + POS_DIM(1) + HIDDEN_DIM(2) +
# NUM_HIDDEN_LAYERS(1) + NUM_PAIRS(2) + EVAL_H(2) + EVAL_W(2) +
# BASE_LEN(4) + MOD_LEN(4) + META_LEN(4) + reserved(3)
# = 5+1+1+1+2+1+2+2+2+4+4+4+3 = 32 bytes
COINPP1_HEADER_FMT: str = "<5sBBBHBHHHIII3s"
COINPP1_HEADER_LEN: int = 32

# Default config (CARGO-CULTED at L0 per Catalog #303 audit in design memo).
DEFAULT_MOD_DIM: int = 64
"""Default per-pair modulation dimension (CARGO-CULTED at L0; sweep at L1)."""

DEFAULT_POS_DIM: int = 32
"""Default sinusoidal positional encoding frequency count (Mildenhall NeRF 2020 default)."""

DEFAULT_HIDDEN_DIM: int = 64
"""Default coord-MLP hidden dim (COIN++ paper canonical small-INR width)."""

DEFAULT_NUM_HIDDEN_LAYERS: int = 3
"""Default coord-MLP depth (COIN++ paper canonical)."""

DEFAULT_EVAL_H: int = 384
"""Default contest scorer-resolution height."""

DEFAULT_EVAL_W: int = 512
"""Default contest scorer-resolution width."""

# Sister substrate citation: paradigm-distinct from sister substrates per
# Catalog #290 §"Canonical-vs-unique decision per layer".
SISTER_SUBSTRATES: tuple[str, ...] = (
    "dreamer_v3_rssm",            # A: categorical latent dynamics
    "boost_nerv_pr110_residual",  # E: iterative boosting vs frozen base
    "nirvana_cascading_nerv",     # G: hierarchical residual decoder cascade
    "coin_plus_plus",             # 2026-05-20 prior sketch; HISTORICAL_PROVENANCE
)


# Lazy import to keep top-level import cheap (MLX may not be installed on
# every consumer machine; the architecture module pulls MLX lazily as well).
def _load_config():
    """Lazy import escape hatch for CoinPPImplicitNeuralRepresentationConfig."""
    from .mlx_renderer import CoinPPImplicitNeuralRepresentationConfig as _Config

    return _Config


# Sentinel re-export via __getattr__ so
# `from tac.substrates.coin_pp_implicit_neural_representation import CoinPPImplicitNeuralRepresentationConfig`
# works without forcing MLX at top-level import time.
def __getattr__(name: str):  # noqa: D401 — module-level escape
    if name == "CoinPPImplicitNeuralRepresentationConfig":
        return _load_config()
    raise AttributeError(
        f"module 'coin_pp_implicit_neural_representation' has no attribute {name!r}"
    )


# Public API surface. Catalog #335 contract for cathedral consumer auto-
# discovery: narrow + explicit per HNeRV parity L12.
__all__ = [
    "ARCHIVE_GRAMMAR_FIELDS",
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "COINPP1_HEADER_FMT",
    "COINPP1_HEADER_LEN",
    "CoinPPImplicitNeuralRepresentationConfig",
    "DEFAULT_EVAL_H",
    "DEFAULT_EVAL_W",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_MOD_DIM",
    "DEFAULT_NUM_HIDDEN_LAYERS",
    "DEFAULT_POS_DIM",
    "SISTER_SUBSTRATES",
]
