# SPDX-License-Identifier: MIT
"""Z3 Ballé hyperprior bolt-on substrate package.

Across-class staircase Step 1 per the zen-floor council
(`feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`) + long-term
campaign roadmap (`feedback_long_term_multi_year_campaigns_landed_20260514.md`):
the cheapest $2 validation that Ballé-2018 scale hyperprior side-info
reduces bytes on the frozen A1 base.

Re-exports the public API of the architecture + archive + inflate modules
so callers can ``from tac.substrates.z3_balle_hyperprior_bolton import ...``.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
Tagged ``research_only=true`` until empirical smoke + full-run anchors land.

Versioning + supersession (SELFCOMP-1, R2 MEDIUM, 2026-05-15):

* **v1 (``archive`` / ``inflate`` / ``score_aware_loss``)** is the LEGACY
  append-only Z3HP1 sidecar grammar. Council omnibus Decision 3 (commit
  ``7872c9f4b``, 2026-05-14) resolved that v1 is **DEPRECATED-pending-v2-
  empirical-confirmation**: v2 is the operational latent-replacement path,
  but v1 is preserved as the production default until v2 has a
  ``[contest-CUDA]`` empirical anchor on Modal/Vast.ai. Per CLAUDE.md
  "Forbidden premature KILL" + "KILL is LAST RESORT": v1 stays LIVE with a
  deprecation warning at import; reactivation/retirement criterion =
  v2 paired Modal A100 anchor lands.
* **v2 (``archive_v2`` / ``inflate_v2`` / ``score_aware_loss_v2``)** is the
  Z3HV2 latent-replacement archive grammar that REPLACES A1's
  ``latent_blob`` in-place. Trainer opt-in via
  ``--enable-v2-latent-replacement``.

To suppress the v1 deprecation warning during legitimate v1 use (forensic
replay, regression tests, etc.), set the environment variable
``Z3_BALLE_USE_V1=1`` before importing. The warning is a one-shot per
process; disable warnings entirely via ``warnings.filterwarnings("ignore")``
if a downstream tool floods the operator console.
"""
from __future__ import annotations

import os as _os
import warnings as _warnings

from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_BASE_CHANNELS,
    A1_CAMERA_H,
    A1_CAMERA_W,
    A1_EVAL_H,
    A1_EVAL_W,
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
    conditional_gaussian_rate_bits,
    factorized_uniform_rate_bits,
    total_balle_rate_bits,
)


def _emit_v1_deprecation_warning() -> bool:
    """One-shot DeprecationWarning when v1 surfaces are imported without opt-in.

    Returns True if a warning was emitted, False if suppressed via the
    ``Z3_BALLE_USE_V1=1`` environment opt-in (set when the operator
    legitimately wants v1 — forensic replay, regression test, etc.).

    Per SELFCOMP-1 (R2 MEDIUM, 2026-05-15) + CLAUDE.md "Bugs must be
    permanently fixed AND self-protected against": surface drift between
    code reality (v1 LIVE) and commit-message claims (v1 retired) is now
    a runtime warning so future trainer authors do not import v1 by
    accident. Cross-ref ``feedback_recursive_review_r2_wave_a_*`` SELFCOMP-1
    + Council omnibus Decision 3 (commit ``7872c9f4b``).
    """
    if _os.environ.get("Z3_BALLE_USE_V1", "").strip() == "1":
        return False
    _warnings.warn(
        "Z3 Ballé hyperprior bolt-on v1 (Z3HP1 append-only sidecar grammar) "
        "is DEPRECATED-pending-v2-empirical-confirmation per Council omnibus "
        "Decision 3 (commit 7872c9f4b, 2026-05-14). v2 (Z3HV2 latent-replacement) "
        "is the operational path; v1 is preserved as production default until "
        "v2 has a [contest-CUDA] empirical anchor. To suppress this warning "
        "for legitimate v1 use (forensic replay, regression tests), set "
        "Z3_BALLE_USE_V1=1 in the environment. Use v2 by passing "
        "--enable-v2-latent-replacement to the Z3 trainer.",
        DeprecationWarning,
        stacklevel=2,
    )
    return True


_emit_v1_deprecation_warning()


from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    Z3_APPEND_ONLY_CONTRACT_BLOCKER,
    Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER,
    Z3CompositionArchiveContract,
    Z3HP1_HEADER_STRUCT,
    Z3HP1_MAGIC,
    Z3HP1_VERSION,
    Z3HP1SidecarMeta,
    build_composition_archive_contract,
    decode_z3hp1_sidecar,
    dequantize_int8_with_scale,
    encode_z3hp1_sidecar,
    pack_composition_archive,
    quantize_int8_with_scale,
    split_composition_archive,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_BLOB_LEN,
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    A1_SECTION_TOTAL_PREFIX_LEN,
    Z3HV2_HEADER_STRUCT,
    Z3HV2_MAGIC,
    Z3HV2_PER_DIM_AFFINE_LEN,
    Z3HV2_VERSION,
    Z3HV2SectionMeta,
    Z3V2CompositionArchiveContract,
    build_z3v2_composition_archive_contract,
    build_z3v2_payload_bytes,
    decode_z3hv2_section,
    encode_z3hv2_section,
    split_z3v2_payload_bytes,
)

__all__ = [
    "_emit_v1_deprecation_warning",
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_DECODER_BLOB_LEN",
    "A1_DECODER_SECTION_TOTAL",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_BLOB_LEN",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "A1_SECTION_TOTAL_PREFIX_LEN",
    "Z3HV2_HEADER_STRUCT",
    "Z3HV2_MAGIC",
    "Z3HV2_PER_DIM_AFFINE_LEN",
    "Z3HV2_VERSION",
    "Z3HV2SectionMeta",
    "Z3V2CompositionArchiveContract",
    "Z3_APPEND_ONLY_CONTRACT_BLOCKER",
    "Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER",
    "Z3CompositionArchiveContract",
    "Z3HP1_HEADER_STRUCT",
    "Z3HP1_MAGIC",
    "Z3HP1_VERSION",
    "Z3HP1SidecarMeta",
    "Z3HyperpriorConfig",
    "Z3HyperpriorMLP",
    "build_composition_archive_contract",
    "build_z3v2_composition_archive_contract",
    "build_z3v2_payload_bytes",
    "conditional_gaussian_rate_bits",
    "decode_z3hp1_sidecar",
    "decode_z3hv2_section",
    "dequantize_int8_with_scale",
    "encode_z3hp1_sidecar",
    "encode_z3hv2_section",
    "factorized_uniform_rate_bits",
    "pack_composition_archive",
    "quantize_int8_with_scale",
    "split_composition_archive",
    "split_z3v2_payload_bytes",
    "total_balle_rate_bits",
]
