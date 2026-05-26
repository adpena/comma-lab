# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260526_path_3_e_boost_nerv_against_pr110_residual_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_and_substrate_contract_canonical_helper_adoption_when_pr110_base_archive_sha_prefix_field_is_added_to_contract_schema
"""boost_nerv_pr110_residual — BoostNeRV against PR110 fec6 frontier (L0 SCAFFOLD).

Path 3 candidate #E per operator directive 2026-05-26:
*"We should add boostnerv to the priority list too, maybe against PR110,
because it seems like it could be free gains if done right."*

Binding strategic reframing 2026-05-26 (verbatim):
*"design the substrate and curriculum and then optimize the design the
whole stack around it for extreme optimization and performance and
optimal score lowering"*

This substrate is a SISTER to `src/tac/substrates/boost_nerv/` (generic
boosting NeRV), NOT an extension of it. The "against PR110" framing means
PR110's HNeRV+fec6+Huffman-k=16 archive is the FROZEN base learner; we
add an iterative residual codec via brotli-compressed sidecar that the
inflate runtime applies on top of PR110-produced frames.

Architecture (council-approved L0 SCAFFOLD 2026-05-26):

    Stage 0: cache PR110 base reconstructions (subprocess inflate; one-time)
       |
       v
    Stage 1: per-pair residual_target = GT - PR110_base_reconstruction
       |
       v
    Stage 2: MLX residual learner warm-up (L2 loss; ~10 epochs)
       |
       v
    Stage 3: MLX score-aware fine-tune (Lagrangian; eval_roundtrip; EMA; ~50 epochs)
       |
       v
    Stage 4: archive = BPR1_sidecar (residual EMA-shadow + int8 quant + brotli) || PR110_base
              + Catalog #1265 contest-equivalence gate (MANDATORY before paid dispatch)
       |
       v
    [optional L1+] Stage 5: round-2 residual on top of round-1

Archive grammar (BPR1):
    24-byte header: magic b"BPR1\\x00" (5) + version u8 (1) + NUM_ROUNDS u8 (1) +
        PR110_BASE_SHA256_PREFIX[16] u128 (16) + RESIDUAL_BLOB_LEN u32 (4) +
        reserved u8 (1, set to 0)
    RESIDUAL_BLOB_LEN bytes: brotli-quality9 compressed int8 residual learner
        state_dict + per-pair latent z_pr110 reference
    Then PR110_BASE_ARCHIVE_BYTES inline (preserves PR110 fec6+Huffman-k=16 bytes
        unchanged; bound by PR110_BASE_SHA256_PREFIX at runtime)

CLAUDE.md compliance:
- No silent device defaults (MLX explicit; PyTorch export path uses canonical
  `tac.substrates._shared.inflate_runtime.select_inflate_device`)
- No scorer load at inflate time (only PR110-base inflate + brotli decode +
  per-pair int8 dequantize + residual MLP forward)
- No /tmp paths in persisted artifacts
- Every file reviewable in 30 seconds per HNeRV parity L12
- `_full_main` raises NotImplementedError per Catalog #240 L0 SCAFFOLD posture

See `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
for the full design memo with Catalog #290 canonical-vs-unique decisions,
Catalog #294 9-dim checklist, Catalog #303 cargo-cult audit, Catalog #305
observability surface, Catalog #296 predicted-band Dykstra-feasibility check.
"""

from __future__ import annotations

# Public API surface. The canonical pattern (per Catalog #335 contract for
# cathedral consumer auto-discovery sister discipline) is narrow + explicit.
__all__ = [
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "BPR1_HEADER_FMT",
    "BPR1_HEADER_LEN",
    "DEFAULT_NUM_BOOSTING_ROUNDS",
    "DEFAULT_RESIDUAL_BUDGET_BYTES",
    "BoostNervPr110ResidualConfig",
]

# Archive grammar constants (Catalog #146 inflate runtime contract; fixed
# offsets declared in source per HNeRV parity L3 monolithic-single-file 0.bin
# pattern). The PR110_BASE_SHA256_PREFIX[16] field is the structural-extinction
# primitive that prevents the sidecar from being silently mis-applied to a
# non-PR110 base archive (per Catalog #139 byte-mutation discipline).
ARCHIVE_MAGIC = b"BPR1\x00"  # 5 bytes: B=Boost, PR=PR110, 1=v1
ARCHIVE_VERSION = 1  # u8
BPR1_HEADER_FMT = "<5sBBB16sIB"  # magic(5s) + version(B) + num_rounds(B) +
#                                  reserved_alignment(B) + sha_prefix(16s) +
#                                  residual_blob_len(I) + reserved_tail(B)
BPR1_HEADER_LEN = 29  # bytes — sum of "<5sBBB16sIB" with little-endian no padding

# Default budgets (CARGO-CULTED at L0 per Catalog #303 audit in design memo).
DEFAULT_NUM_BOOSTING_ROUNDS = 1
DEFAULT_RESIDUAL_BUDGET_BYTES = 8192  # ~+0.00546 contest-units rate cost


# Lazy import to keep top-level import cheap (MLX may not be installed on
# every consumer machine; the architecture module pulls MLX lazily as well).
def _load_config():
    """Lazy import escape hatch for BoostNervPr110ResidualConfig."""
    from .architecture import BoostNervPr110ResidualConfig as _Config

    return _Config


# Sentinel re-export via __getattr__ so `from tac.substrates.boost_nerv_pr110_residual import BoostNervPr110ResidualConfig`
# works without forcing MLX at top-level import time.
def __getattr__(name: str):  # noqa: D401 — module-level escape
    if name == "BoostNervPr110ResidualConfig":
        return _load_config()
    raise AttributeError(f"module 'boost_nerv_pr110_residual' has no attribute {name!r}")
