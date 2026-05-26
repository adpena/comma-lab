# SPDX-License-Identifier: MIT
"""Z7-Mamba-2-v2 fresh substrate architecture — L0 SCAFFOLD skeleton.

This file is the L0 SCAFFOLD design-level architecture surface. Per the
Phase 3 L0 SCAFFOLD design memo
(`.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`), this
substrate is designed from FIRST PRINCIPLES around Mamba-2's selective
state-space math, NOT extended from the existing
`tac.substrates.time_traveler_l5_z7_mamba2` bolt-on scaffold (which
inherited Z6 decoder + Z7-LSTM/GRU latent + Z7PCWM1 archive grammar).

The 4 orthogonal axes of UNIQUE-FORK per Phase 1 audit (CC-A through CC-J):

1. **Decoder axis (CC-A unwind):** `Mamba2TemporalDecoder` with Conv1D
   temporal pre-stage matching Mamba-2's `d_conv=4` window. Replaces
   the canonical Z6 PixelShuffle decoder that did NOT consume Mamba-2's
   distinguishing temporal structure.
2. **Latent dimensionality axis (CC-B + CC-C unwind):** `latent_dim=32`
   default (was 24) + `ego_motion_dim=16` default (was 8). Curriculum
   sweep ready for L1 ablation.
3. **Training-pathway axis (CC-D + CC-G unwind):** chunk-parallel SSD-scan
   (CUDA backend via upstream mamba_ssm); sequential reference
   (MLX/MPS/reference_torch). A_log init scheme configurable
   ∈ {Z+1 default, HiPPO-like, log-uniform} per CC-D unwind.
4. **Archive grammar axis (CC-J unwind):** Z7MCM3 grammar with
   procedurally-regenerable A_log (~4 KB savings) + cosine-quantized
   B/C projection matrices (~1 KB savings). Lives in archive.py.

L0 SCAFFOLD scope: this file lands the Config + Substrate class SKELETON
with `NotImplementedError` on `_full_main` per CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable. Full
trainer + MLX-native cell + SSD-scan integration lands at L1 EMPIRICAL
build per the canonical 6-step per-substrate symposium contract
(Catalog #325).

Per CLAUDE.md "Forbidden premature KILL": existing
`time_traveler_l5_z7_mamba2` is PRESERVED as v1 historical sister; this
v2 scaffold is the substrate-class-shift candidate per HNeRV parity L7
substrate-engineering split.

[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md §3 + §7]
[verified-against: .omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md §2 CC-A through CC-J]
[verified-against: Dao-Gu 2024 (arxiv 2405.21060) Mamba-2 §3 + §4 SSD]
"""

from __future__ import annotations

from dataclasses import dataclass

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width); HARD-EARNED non-negotiable."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair); HARD-EARNED non-negotiable."""

A_LOG_INIT_SCHEMES: tuple[str, ...] = ("z_plus_1", "hippo_like", "log_uniform")
"""Per CC-D unwind: A_log init scheme is configurable, NOT inherited from upstream language default."""

TRAINING_BACKENDS: tuple[str, ...] = ("ssd_scan_cuda", "sequential_reference_torch", "mlx_native")
"""Per CC-G + CC-F unwind: training pathway is substrate-design choice, not inherited from GRU sequential autoregress."""


def normalize_a_log_init_scheme(value: str) -> str:
    """Normalize + validate the A_log init scheme per CC-D unwind."""
    scheme = str(value).strip().lower().replace("-", "_")
    if scheme not in A_LOG_INIT_SCHEMES:
        allowed = ", ".join(A_LOG_INIT_SCHEMES)
        raise ValueError(f"a_log_init_scheme must be one of: {allowed}; got {value!r}")
    return scheme


def normalize_training_backend(value: str) -> str:
    """Normalize + validate the training backend per CC-G + CC-F unwind."""
    backend = str(value).strip().lower().replace("-", "_")
    if backend not in TRAINING_BACKENDS:
        allowed = ", ".join(TRAINING_BACKENDS)
        raise ValueError(f"training_backend must be one of: {allowed}; got {value!r}")
    return backend


@dataclass(frozen=True)
class Z7Mamba2V2Config:
    """Static design-time parameters for Z7-Mamba-2-v2 substrate.

    Defaults match Phase 3 design memo §7 architectural specification.
    Every default that differs from the existing
    `tac.substrates.time_traveler_l5_z7_mamba2.Z7Mamba2PredictiveCodingConfig`
    encodes an explicit UNIQUE-FORK per the Phase 1 cargo-cult audit.

    Args:
        latent_dim: per-pair latent dimensionality. **DEFAULT 32**
            (was 24 in v1 per CC-B cargo-cult). Curriculum sweep ready.
        ego_motion_dim: ego-motion projection dim. **DEFAULT 16**
            (was 8 in v1 per CC-C cargo-cult).
        d_model: Mamba-2 internal model dim (default 64; same as v1).
        d_state: Mamba-2 selective state-space dim (default 16;
            CC-9 sweep candidate per parent design memo).
        expand: Mamba-2 expansion factor (default 2; upstream canonical).
        d_conv: Mamba-2 conv1d kernel size (default 4; upstream canonical
            AND the temporal-conv window for the NEW decoder per CC-A unwind).
        a_log_init_scheme: per CC-D unwind; ∈ {"z_plus_1" (Mamba-2
            upstream default), "hippo_like", "log_uniform"}.
        training_backend: per CC-G + CC-F unwind; ∈ {"ssd_scan_cuda",
            "sequential_reference_torch", "mlx_native"}.
        stateful: per CC-7 HARD-EARNED Wyner-Ziv pattern (canonical-adopt
            from v1) + CC-E HARD-EARNED-PARTIAL channel-size disambiguator
            at L1 (warn at substrate construction; ablation at L1).
        identity_predictor: probe-disambiguator control per Catalog #125
            hook #6; sister-canonical from v1.
        beta_ib: β-IB Lagrangian; default 1.0 inherited from v1; ib_scale
            forked per CC-H HARD-EARNED-PARTIAL.
        ib_scale: per CC-H unwind; **DEFAULT 5e-4** (was 1e-3 in v1);
            substrate-forked because Mamba-2 SSM produces smoother
            latents than GRU → latent_smoothness penalty is more
            redundant for Mamba-2.
        num_pairs: contest pair count (600); HARD-EARNED non-negotiable.
        decoder_*: Mamba2TemporalDecoder canonical parameters
            (decoder is UNIQUE-FORK per CC-A; sister field names match
            v1 _Z6Decoder for paired-comparison observability per
            Catalog #305 facet 3).
        output_height/output_width: HARD-EARNED EVAL_HW.
        latent_init_std: standard init scale (canonical-adopt from v1).
    """

    # --- UNIQUE-FORK per Phase 1 audit ---
    latent_dim: int = 32  # CC-B unwind: was 24 in v1
    ego_motion_dim: int = 16  # CC-C unwind: was 8 in v1
    a_log_init_scheme: str = "z_plus_1"  # CC-D unwind: explicitly configurable
    training_backend: str = "mlx_native"  # CC-G + CC-F unwind: MLX-first per binding directive #1
    ib_scale: float = 5e-4  # CC-H unwind: was 1e-3 in v1
    # --- CANONICAL-ADOPT (sister-parity with v1) ---
    d_model: int = 64
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    stateful: bool = True  # CC-7 HARD-EARNED Wyner-Ziv pattern
    identity_predictor: bool = False
    beta_ib: float = 1.0
    num_pairs: int = NUM_PAIRS
    # --- UNIQUE-IMPL decoder skeleton (Mamba2TemporalDecoder; CC-A unwind) ---
    decoder_embed_dim: int = 32
    decoder_initial_grid_h: int = 24
    decoder_initial_grid_w: int = 32
    decoder_channels: tuple[int, ...] = (32, 24, 16, 12)
    decoder_num_upsample_blocks: int = 4
    decoder_temporal_conv_enabled: bool = True  # CC-A unwind: enable Conv1D temporal pre-stage
    # --- HARD-EARNED non-negotiables ---
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    latent_init_std: float = 0.02

    @property
    def d_inner(self) -> int:
        """Mamba-2 inner dimension after expansion."""
        return self.expand * self.d_model

    @property
    def predictor_input_dim(self) -> int:
        """Concat dim for (z_prev, ego_motion) input."""
        return self.latent_dim + self.ego_motion_dim

    def __post_init__(self) -> None:
        """Validate cargo-cult-unwound defaults per Phase 1 audit invariants."""
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {self.latent_dim}")
        if self.ego_motion_dim <= 0:
            raise ValueError(f"ego_motion_dim must be positive; got {self.ego_motion_dim}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive; got {self.d_state}")
        if self.expand <= 0:
            raise ValueError(f"expand must be positive; got {self.expand}")
        if self.d_conv <= 0:
            raise ValueError(f"d_conv must be positive; got {self.d_conv}")
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive; got {self.num_pairs}")
        if self.ib_scale < 0:
            raise ValueError(f"ib_scale must be non-negative; got {self.ib_scale}")
        if self.beta_ib < 0:
            raise ValueError(f"beta_ib must be non-negative; got {self.beta_ib}")
        if self.latent_init_std < 0:
            raise ValueError(
                f"latent_init_std must be non-negative; got {self.latent_init_std}"
            )
        # Validate configurable schemes per CC-D + CC-G + CC-F unwinds
        normalize_a_log_init_scheme(self.a_log_init_scheme)
        normalize_training_backend(self.training_backend)


class Z7Mamba2V2Substrate:
    """Z7-Mamba-2-v2 substrate L0 SCAFFOLD skeleton.

    L0 SCAFFOLD scope: this class declares the substrate's structural
    contract (Config + per-axis distinguishing-feature surfaces). Full
    architecture implementation (Mamba2V2Cell + Mamba2TemporalDecoder +
    SSD-scan integration + MLX-native predictor) is DEFERRED to L1
    EMPIRICAL build per the canonical 6-step per-substrate symposium
    contract (Catalog #325) AND per the canonical-vs-unique decision
    per layer table in the Phase 3 L0 SCAFFOLD design memo §3.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable: ``research_only=True`` declared at package
    `__init__.py`; this class raises ``NotImplementedError`` from
    ``__init__`` to make the scaffold-only nature structurally explicit
    so any caller that tries to instantiate fails-loud BEFORE GPU spend.

    Per Phase 1 audit CC-F + Phase 2 design decision §3 op-routable #1:
    L0 SCAFFOLD is design+skeleton+memo only ($0 GPU; NO paid dispatch).
    """

    def __init__(self, config: Z7Mamba2V2Config) -> None:
        """L0 SCAFFOLD skeleton: declares contract, refuses instantiation."""
        # Validate config (this works even at L0)
        if not isinstance(config, Z7Mamba2V2Config):
            raise TypeError(
                f"config must be Z7Mamba2V2Config; got {type(config).__name__}"
            )
        self.config = config
        raise NotImplementedError(
            "Z7Mamba2V2Substrate is L0 SCAFFOLD only — full implementation "
            "(Mamba2V2Cell + Mamba2TemporalDecoder + SSD-scan integration + "
            "MLX-native predictor) lands at L1 EMPIRICAL build per the "
            "Phase 3 L0 SCAFFOLD design memo "
            "(.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md) §7 + "
            "per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' "
            "non-negotiable. Validated config (latent_dim=%d / ego_motion_dim=%d / "
            "training_backend=%s / a_log_init=%s / ib_scale=%.0e). "
            "Use the existing time_traveler_l5_z7_mamba2 (v1) for runnable "
            "Z7-Mamba-2 work until L1 lands."
            % (
                config.latent_dim,
                config.ego_motion_dim,
                config.training_backend,
                config.a_log_init_scheme,
                config.ib_scale,
            )
        )


class Mamba2TemporalDecoder:
    """L0 SCAFFOLD skeleton for the CC-A UNIQUE-FORK decoder.

    Mamba2TemporalDecoder consumes the (num_pairs=600, latent_dim=32)
    latent stream with a Conv1D temporal pre-stage matching Mamba-2's
    d_conv=4 selective-state-space temporal window, then applies the
    sister-canonical PixelShuffle spatial decode per pair.

    Per Phase 3 design memo §7.2 architectural spec::

        Mamba2TemporalDecoder(latent_dim=32, embed_dim=32, num_pairs=600)
        forward(z_stream) -> (rgb_0_stream, rgb_1_stream):
          # z_stream: (num_pairs, latent_dim)
          z_temporal = Conv1D(d_conv=4, padding=3, in_ch=latent_dim,
                              out_ch=embed_dim)(z_stream.T).T
          # Per-pair PixelShuffle decode (sister to Z6 spatial decoder)
          for t in range(num_pairs):
            rgb_0, rgb_1 = spatial_decoder(z_temporal[t])
          return stack(rgb_0_list), stack(rgb_1_list)

    L0 SCAFFOLD: contract-only; refuses instantiation per Phase 2
    op-routable #1.
    """

    def __init__(self, config: Z7Mamba2V2Config) -> None:
        if not isinstance(config, Z7Mamba2V2Config):
            raise TypeError(
                f"config must be Z7Mamba2V2Config; got {type(config).__name__}"
            )
        self.config = config
        raise NotImplementedError(
            "Mamba2TemporalDecoder is L0 SCAFFOLD only — full implementation "
            "lands at L1 EMPIRICAL build per the Phase 3 L0 SCAFFOLD design "
            "memo §7.2. The temporal Conv1D pre-stage IS the substrate's "
            "distinguishing-feature surface per CC-A unwind."
        )


class Mamba2V2Cell:
    """L0 SCAFFOLD skeleton for the CC-D UNIQUE-FORK selective-state-space cell.

    Mamba2V2Cell is the MLX-native Mamba-2 selective state-space cell
    with A_log init scheme configurable per CC-D unwind. The CUDA path
    uses the upstream mamba_ssm SSD-scan kernel per CC-G unwind; the
    MLX-native path uses a pure-MLX sequential reference (predecessor
    `ae2fa302fbbf5ffa4` state_dict-key-parity work verifies MLX↔PyTorch
    is byte-stable at the math layer).

    Per Phase 3 design memo §7.1 architectural spec::

        Mamba2V2Cell(d_model=64, d_state=16, expand=2, d_conv=4,
                     a_log_init_scheme="z_plus_1")
        forward(x_t, h_prev) -> (y_t, h_t):
          # Selective state-space step per Dao-Gu 2024
          ...

    L0 SCAFFOLD: contract-only; refuses instantiation per Phase 2
    op-routable #1.
    """

    def __init__(self, config: Z7Mamba2V2Config) -> None:
        if not isinstance(config, Z7Mamba2V2Config):
            raise TypeError(
                f"config must be Z7Mamba2V2Config; got {type(config).__name__}"
            )
        self.config = config
        self.a_log_init_scheme = normalize_a_log_init_scheme(config.a_log_init_scheme)
        self.training_backend = normalize_training_backend(config.training_backend)
        raise NotImplementedError(
            "Mamba2V2Cell is L0 SCAFFOLD only — full MLX-native + "
            "SSD-scan-CUDA implementation lands at L1 EMPIRICAL build per "
            "the Phase 3 L0 SCAFFOLD design memo §7.1 + per the predecessor "
            "state_dict-key-parity work as research input."
        )


__all__ = [
    "A_LOG_INIT_SCHEMES",
    "EVAL_HW",
    "NUM_PAIRS",
    "TRAINING_BACKENDS",
    "Mamba2TemporalDecoder",
    "Mamba2V2Cell",
    "Z7Mamba2V2Config",
    "Z7Mamba2V2Substrate",
    "normalize_a_log_init_scheme",
    "normalize_training_backend",
]
