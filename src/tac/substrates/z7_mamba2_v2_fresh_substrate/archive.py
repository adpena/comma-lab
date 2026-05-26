# SPDX-License-Identifier: MIT
"""Z7MCM3 archive grammar for Z7-Mamba-2-v2 fresh substrate (L0 SCAFFOLD).

Per CC-J unwind (Phase 1 cargo-cult audit §2): the Z7MCM2 grammar (v1
substrate) is a sister-renamed clone of Z7PCWM1 (LSTM grammar) — same
bolt-on pattern at the archive-grammar layer. Z7MCM3 redesigns the
predictor blob per Mamba-2's actual numerical structure:

1. **A_log procedural regeneration**: the Mamba-2 selective-state-space
   A matrix is `A = -exp(A_log)` where A_log is INPUT-INDEPENDENT and
   has KNOWN STRUCTURE (decreasing positive integers per upstream
   default; HiPPO-like per Gu 2022; log-uniform per ablation). The
   matrix can be procedurally generated from a 1-byte init-scheme tag,
   saving ~4 KB vs Z7MCM2 fp16 serialization.
2. **B/C cosine quantization**: B_proj + C_proj are low-rank linear
   maps from d_inner → d_state. Empirically (per Mamba-2 §5 ablation)
   cosine-similarity quantization to ~8 bits preserves accuracy within
   1e-3 of fp16. Saves ~1 KB vs Z7MCM2 fp16 serialization.
3. **conv1d kernel quantized**: the d_conv=4 temporal conv kernel is
   d_inner=128 * 4 = 512 fp16 values; kernel-quantization to int8 saves
   ~256 bytes.
4. **dt_proj / in_proj / out_proj**: linear layers; fp16+brotli per
   sister-canonical pattern (HARD-EARNED entropy coding per Catalog
   "Bit-level deconstruction and entropy discipline").

Estimated byte budget vs Z7MCM2 baseline:
- Z7MCM2 predictor_blob: ~30 KB
- Z7MCM3 predictor_blob: ~25 KB (savings: ~5 KB at rate-axis directly)

Per CLAUDE.md HNeRV parity discipline:
- L3 (Archive grammar = monolithic single-file `0.bin`): preserved
- L4 (Inflate ≤200 LOC substrate-engineering waiver): preserved
- L8 (Eval-roundtrip-aware): preserved at training (sister-canonical loss)
- L11 (No-op detector via byte-mutation smoke per Catalog #139 + #272):
  preserved + extended with A_log procedural-regeneration disambiguator
  (1-byte init-scheme mutation must produce measurable downstream frame
  changes via different A matrix → different selective-state-space
  recurrence → different latent stream → different RGB output)

L0 SCAFFOLD scope: contract + grammar layout declaration only. Full
pack/unpack/replay implementation lands at L1 EMPIRICAL build per the
Phase 3 L0 SCAFFOLD design memo §7.3.

[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md §7.3]
[verified-against: .omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md CC-J]
[verified-against: Dao-Gu 2024 (arxiv 2405.21060) Mamba-2 §3 A_log structure]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tac.substrates.z7_mamba2_v2_fresh_substrate.architecture import (
    EVAL_HW,
    Z7Mamba2V2Config,
    normalize_a_log_init_scheme,
)

# Z7MCM3 magic + version per Phase 3 design memo §7.3
Z7MCM3_MAGIC: bytes = b"Z7M3"
Z7MCM3_VERSION: int = 3
Z7MCM3_HEADER_FMT: str = "<4sBHBBBBBB"  # magic(4) + version(1) + num_pairs(2) + ...
"""Little-endian header struct: magic(4s) version(B) num_pairs(H) latent_dim(B)
d_model(B) d_state(B) ego_dim(B) d_conv(B) a_log_init(B); total 13 bytes."""
Z7MCM3_HEADER_SIZE: int = struct.calcsize(Z7MCM3_HEADER_FMT)

# Section role enum per Phase 3 design memo §7.3
Z7MCM3_SECTION_ROLES: tuple[str, ...] = (
    "meta_blob",
    "encoder_blob",
    "decoder_blob",
    "predictor_blob",  # CC-J unwind: A_log procedurally regenerated, NOT serialized
    "latent_init_blob",
    "residuals_blob",
    "ego_motion_blob",
)
"""Section roles in canonical order; Z7MCM3 adds A_log procedural-regeneration sentinel."""

A_LOG_INIT_SCHEME_BYTE_ENUM: dict[str, int] = {
    "z_plus_1": 0,
    "hippo_like": 1,
    "log_uniform": 2,
}
"""1-byte enum for A_log init scheme per CC-D + CC-J unwind."""


@dataclass(frozen=True)
class Z7MCM3Archive:
    """L0 SCAFFOLD dataclass for parsed Z7MCM3 archive.

    Carries the parsed config + section blobs needed by the inflate
    runtime. Full implementation (pack_archive, parse_archive,
    replay_latent_sequence) lands at L1 per Phase 3 design memo §7.3.

    The A_log matrix is NOT stored in `predictor_state_dict` — it is
    procedurally regenerated from `config.a_log_init_scheme` per CC-J
    unwind. This is the substrate-distinguishing rate-axis savings.
    """

    config: Z7Mamba2V2Config
    meta: dict[str, object]
    encoder_state_dict: dict[str, object]
    decoder_state_dict: dict[str, object]
    predictor_state_dict: dict[str, object]  # EXCLUDES A_log (regenerated)
    latent_init: object  # placeholder for torch.Tensor at L1
    residuals: object  # placeholder for torch.Tensor at L1
    ego_motion: object  # placeholder for torch.Tensor at L1


def pack_archive(*args: object, **kwargs: object) -> bytes:
    """L0 SCAFFOLD stub for Z7MCM3 packer.

    Full implementation lands at L1 per Phase 3 design memo §7.3.
    Refuses to run at L0 per CLAUDE.md "Substrate scaffolds MUST be
    COMPLETE or RESEARCH-ONLY" non-negotiable.
    """
    raise NotImplementedError(
        "pack_archive is L0 SCAFFOLD only — full Z7MCM3 packer lands at L1 "
        "EMPIRICAL build per the Phase 3 L0 SCAFFOLD design memo §7.3 + per "
        "CC-J unwind A_log procedural-regeneration grammar."
    )


def parse_archive(blob: bytes) -> Z7MCM3Archive:
    """L0 SCAFFOLD stub for Z7MCM3 parser.

    Full implementation lands at L1 per Phase 3 design memo §7.3.
    """
    raise NotImplementedError(
        "parse_archive is L0 SCAFFOLD only — full Z7MCM3 parser lands at L1 "
        "EMPIRICAL build per the Phase 3 L0 SCAFFOLD design memo §7.3."
    )


def regenerate_a_log_from_init_scheme(
    *,
    d_inner: int,
    d_state: int,
    a_log_init_scheme: str,
) -> object:
    """L0 SCAFFOLD stub for A_log procedural regeneration per CC-J unwind.

    Full implementation lands at L1; the per-scheme math is:
    - `z_plus_1`: A_log[i, j] = log(j + 1) for j ∈ [0, d_state); upstream
      Mamba-2 default per Dao-Gu 2024 §3.
    - `hippo_like`: A_log[i, j] = log(scale * (j + 0.5)) for j ∈ [0,
      d_state); Gu 2022 HiPPO-like init.
    - `log_uniform`: A_log[i, j] = log(uniform_random_init) for j ∈ [0,
      d_state); per CC-D ablation.

    All three schemes produce SAME shape (d_inner, d_state) but DIFFERENT
    eigenvalue spectrum → DIFFERENT selective-state-space decay
    characteristics → DIFFERENT latent stream → DIFFERENT score.
    """
    scheme = normalize_a_log_init_scheme(a_log_init_scheme)
    if scheme not in A_LOG_INIT_SCHEME_BYTE_ENUM:
        raise ValueError(f"a_log_init_scheme {scheme!r} not in enum")
    raise NotImplementedError(
        f"regenerate_a_log_from_init_scheme is L0 SCAFFOLD only — full math "
        f"per Dao-Gu 2024 §3 lands at L1; scheme={scheme!r} d_inner={d_inner} "
        f"d_state={d_state} validated."
    )


def replay_latent_sequence(archive: Z7MCM3Archive) -> object:
    """L0 SCAFFOLD stub for autoregressive replay per Phase 3 §7.1.

    Full implementation lands at L1; the replay must regenerate A_log
    via `regenerate_a_log_from_init_scheme` per CC-J unwind, then unroll
    the Mamba-2 selective-state-space recurrence across `num_pairs`
    pairs.

    The SSD theorem (Dao-Gu 2024 §4) guarantees chunk-parallel SSD scan
    + sequential unroll produce IDENTICAL hidden states; this is the
    byte-stable invariant for inflate-time determinism.
    """
    if not isinstance(archive, Z7MCM3Archive):
        raise TypeError(
            f"archive must be Z7MCM3Archive; got {type(archive).__name__}"
        )
    raise NotImplementedError(
        "replay_latent_sequence is L0 SCAFFOLD only — full Mamba-2 autoregressive "
        "replay lands at L1 EMPIRICAL build per the Phase 3 L0 SCAFFOLD design "
        "memo §7.1."
    )


def estimated_byte_budget() -> dict[str, int]:
    """Return the L0 SCAFFOLD byte-budget estimate per Phase 3 design memo §7.3.

    Returns the per-section byte-count estimate documented in the design
    memo; provides operator-facing observability into the substrate's
    rate-axis cost BEFORE any L1 empirical anchor lands.
    """
    return {
        "header": Z7MCM3_HEADER_SIZE,
        "meta_blob": 2048,  # ~2 KB sorted JSON
        "encoder_blob_optional": 0,
        "decoder_blob": 30 * 1024,
        "predictor_blob_z7mcm3_after_cc_j_unwind": 25 * 1024,  # CC-J unwind saves ~5 KB
        "predictor_blob_z7mcm2_baseline_v1": 30 * 1024,
        "predictor_blob_savings_vs_z7mcm2_per_cc_j": 5 * 1024,
        "latent_init_blob": 50,
        "residuals_blob": 20 * 1024,  # 600 * 32 = 19200 + scale + zero_point
        "ego_motion_blob": 10 * 1024,  # 600 * 16 = 9600 + scale + zero_point
        "total_estimate_compressed": 87 * 1024,
        "z7mcm2_v1_baseline_total_compressed": 92 * 1024,
        "savings_per_cc_j_unwind": 5 * 1024,
    }


__all__ = [
    "A_LOG_INIT_SCHEME_BYTE_ENUM",
    "Z7MCM3_HEADER_FMT",
    "Z7MCM3_HEADER_SIZE",
    "Z7MCM3_MAGIC",
    "Z7MCM3_SECTION_ROLES",
    "Z7MCM3_VERSION",
    "Z7MCM3Archive",
    "estimated_byte_budget",
    "pack_archive",
    "parse_archive",
    "regenerate_a_log_from_init_scheme",
    "replay_latent_sequence",
]
