# SPDX-License-Identifier: MIT
"""faiss_ivf_pq_residual.mlx_renderer — MLX-native PQ codebook gather + tile reassemble.

FIX-WAVE-R1''-I (2026-05-26): R1'' adversarial review (memo
`.omx/research/path_3_i_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
CRITICAL finding I-R1''-1) caught that the prior L0 SCAFFOLD version of this
module declared "MLX-first" in the design memo but shipped NO actual MLX
primitives — only `FaissIVFPQResidualConfig` + cost estimators + `_full_main`
NotImplementedError + an MLX import guard. The "MLX↔numpy parity test
PASSES byte-identical" claim in landing memo §4 referred to a test that uses
`mx.take` DIRECTLY in `tests/test_basic.py:318-326`, NOT through any
substrate-package MLX module. Per Catalog #307 IMPLEMENTATION-LEVEL
falsification (paradigm INTACT; renderer just needs actual MLX
implementation): this FIX-WAVE adds the canonical 5 MLX primitives mirroring
the numpy_reference sister so the Catalog #1265 MLX-first contest-equivalence
gate has substrate-package surfaces to gate.

MLX primitives shipped (canonical mirrors of `numpy_reference.py` sister):

1. ``mlx_pq_codebook_gather(codebook_mx, indices_mx)`` — per-sub-quantizer
   ``mx.take`` gather; byte-identical vs numpy reference (integer index +
   exact float copy; expected drift = 0).
2. ``mlx_pq_reconstruct_tile_vectors(codebook_mx, indices_mx)`` — composition
   of gather + ``mx.reshape`` to flat per-tile vectors.
3. ``mlx_tiles_to_frame_nhwc(tiles_mx, frame_h, frame_w, tile_h, tile_w)``
   — per-pair tile reassemble via ``mx.reshape`` + ``mx.concatenate``;
   deterministic.
4. ``mlx_to_uint8(x_mx)`` — canonical Catalog #205 sister rounding via
   ``mx.round`` + ``mx.clip`` + cast.
5. ``mlx_decode_per_pair_residual(codebook_mx, codewords_mx, tile_h, tile_w,
   residual_scale)`` — full per-pair decode composition mirroring
   ``inflate._decode_per_pair_residual`` for MLX↔PyTorch parity verification.

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable: every
artifact this module produces is tagged ``[macOS-MLX research-signal]`` and
carries ``score_claim=False`` + ``promotion_eligible=False`` +
``ready_for_exact_eval_dispatch=False``. The PROMOTION path is MLX
state_dict → PyTorch via canonical
``tac.local_acceleration.mlx_to_pytorch_export`` (#1251) → FAISSPQ1 archive
via ``tac.substrates.faiss_ivf_pq_residual.archive.build_archive_bytes`` →
Catalog #1265 contest-equivalence gate → operator routes paid CUDA dispatch.

L0 SCAFFOLD module: declares the substrate config + MLX primitives + cost
estimators. MLX is OPTIONAL at top-level import time per Catalog #1
device-selection-defaults discipline + axis 3 portability per operator
directive #3 2026-05-26.

For the numpy-only / non-Apple-Silicon test path, route through
``tac.substrates.faiss_ivf_pq_residual.numpy_reference`` instead.

Per axis 2 MLX drift minimization (operator directive #3 2026-05-26) +
R1'' empirical anchor (M-series MPS matmul drift O(1e-2) abs / O(1e-3) rel
canonical floor) + the MLX-first doctrine baseline:

- PQ codebook gather uses ``mx.take`` per sub-quantizer; integer-index +
  float copy is exact (drift = 0 vs numpy reference)
- Per-tile reassemble uses ``mx.reshape`` + ``mx.concatenate``; deterministic
- ``mlx_to_uint8`` uses ``mx.round`` (banker's rounding identical to numpy
  reference); deterministic
- No bilinear upsample required at the per-pair residual decode (substrate's
  decoder output is already at EVAL_HW; upsample to camera 874×1164 happens
  at inflate.py PyTorch boundary per Catalog #146 contract)
- No PyTorch-style matmul in the substrate's MLX primitives (PQ decode is
  pure gather + reshape + concatenate); the R1'' M-series matmul drift
  bound does NOT apply to this substrate's primitives

Per Catalog #146 + #205 inflate runtime contract:

- Substrate-engineering LOC budget ≤200 for inflate runtime (the MLX
  renderer here is COMPRESS-TIME / EXPORT-TIME only; the canonical inflate
  runtime at ``inflate.py`` is PyTorch-only and stays within budget)
- Device selection at INFLATE-time via canonical
  ``tac.substrates._shared.inflate_runtime.select_inflate_device``
- No scorer load at inflate time
- No /tmp paths in persisted artifacts

Per Catalog #240 L0 SCAFFOLD posture: ``_full_main`` raises
NotImplementedError until PHASE 2 Catalog #325 per-substrate symposium clears
for paid dispatch.

Cross-references
----------------

- Canonical sister G=NIRVANA scaffold pattern:
  ``src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py``
- Canonical sister Z6 MLX renderer with actual primitives:
  ``src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py``
- Canonical MLX-PR95 helpers (CONSOLIDATE-OP-1):
  ``tac.local_acceleration.pr95_hnerv_mlx``
- Canonical numpy-PR95 portable references (CONSOLIDATE-OP-1):
  ``tac.local_acceleration.pr95_hnerv_numpy_reference``
- Substrate numpy reference primitives (sister parity target):
  ``tac.substrates.faiss_ivf_pq_residual.numpy_reference``
- PyTorch baseline (canonical algorithm source-of-truth):
  ``tac.substrates.faiss_ivf_pq_residual.inflate._decode_per_pair_residual``
- R1'' review memo:
  ``.omx/research/path_3_i_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.framework_agnostic import require_mlx_core

# Module-level config. MLX imports are lazy inside primitive functions to keep
# top-level import cheap on non-Apple-Silicon CI (sister substrates' canonical
# pattern per Catalog #1 device-selection-defaults discipline + axis-3
# portability per operator directive #3 2026-05-26).
EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution; final per-pair RGB residual shape (height, width)."""


# Non-promotable canonical contract per CLAUDE.md "MLX portable-local-substrate
# authority" + Catalog #341 routing-markers + Catalog #323 canonical Provenance.
SCHEMA_VERSION = "faiss_ivf_pq_residual_mlx_renderer_v1"
"""Canonical schema version for MLX state_dict export manifests."""

LANE_ID = "lane_path_3_fix_wave_r1_prime_prime_i_faiss_ivf_pq_residual_20260526"
"""Canonical lane id for #325 sister-symposium chain + Catalog #126 pre-registration."""

EVIDENCE_GRADE = "macOS-MLX research-signal"
"""Per CLAUDE.md 'MLX portable-local-substrate authority' non-promotable tag."""

EVIDENCE_TAG = "[macOS-MLX research-signal]"
"""Per CLAUDE.md axis-tag-format non-promotable label."""


@dataclass(frozen=True)
class FaissIVFPQResidualConfig:
    """Substrate configuration; immutable per Catalog #229 PV discipline."""

    m_sub_quantizers: int = 4
    """PQ M parameter — number of sub-quantizers per tile vector.

    Each sub-quantizer carries log2(ksub) bits per tile per sub. Total
    bits-per-tile = m * log2(ksub). M=4 ksub=256 = 32 bits/tile = 4 bytes/tile.
    """

    ksub_codebook_size: int = 256
    """PQ ksub parameter — codebook size per sub-quantizer.

    Per Jégou-Douze-Schmid 2011 canonical: ksub=256 is the SIFT-1M anchor;
    larger ksub = richer codebook but larger codebook bytes (M*ksub*dim*4 raw).
    """

    tile_h: int = 96
    """Spatial tile height. EVAL_HW[0] must be divisible by tile_h."""

    tile_w: int = 128
    """Spatial tile width. EVAL_HW[1] must be divisible by tile_w."""

    num_pairs: int = 600
    """Contest video pair count."""

    residual_quant_bits: int = 16
    """Per-tile residual quantization bits BEFORE PQ encoding (int16 canonical).

    Per-tile vector ∈ [-1, 1] residual range; int16 maps to 32767 levels.
    PQ codebook is trained on int16-quantized residuals.
    """

    residual_scale: float = 0.5
    """Per-pair residual scale factor; residual ∈ [-residual_scale, residual_scale]."""

    def __post_init__(self) -> None:
        if self.m_sub_quantizers < 1 or self.m_sub_quantizers > 16:
            raise ValueError(
                f"m_sub_quantizers={self.m_sub_quantizers} out of [1, 16] range"
            )
        if self.ksub_codebook_size < 2 or self.ksub_codebook_size > 65536:
            raise ValueError(
                f"ksub_codebook_size={self.ksub_codebook_size} out of [2, 65536] range"
            )
        if self.tile_h < 1 or self.tile_w < 1:
            raise ValueError(
                f"tile_h={self.tile_h} tile_w={self.tile_w} must be >= 1"
            )
        if EVAL_HW[0] % self.tile_h != 0:
            raise ValueError(
                f"EVAL_HW[0]={EVAL_HW[0]} must be divisible by tile_h={self.tile_h}"
            )
        if EVAL_HW[1] % self.tile_w != 0:
            raise ValueError(
                f"EVAL_HW[1]={EVAL_HW[1]} must be divisible by tile_w={self.tile_w}"
            )
        tile_dim = self.tile_h * self.tile_w * 3
        if tile_dim % self.m_sub_quantizers != 0:
            raise ValueError(
                f"tile_dim={tile_dim} must be divisible by "
                f"m_sub_quantizers={self.m_sub_quantizers}"
            )
        if self.residual_quant_bits not in {8, 16}:
            raise ValueError(
                f"residual_quant_bits={self.residual_quant_bits} must be in {{8, 16}}"
            )
        if self.residual_scale <= 0.0 or self.residual_scale > 1.0:
            raise ValueError(
                f"residual_scale={self.residual_scale} must be in (0, 1]"
            )

    @property
    def tiles_per_pair(self) -> int:
        """Number of tiles per per-pair RGB residual frame."""
        return (EVAL_HW[0] // self.tile_h) * (EVAL_HW[1] // self.tile_w)

    @property
    def tile_dim(self) -> int:
        """Per-tile vector dimension = tile_h × tile_w × 3."""
        return self.tile_h * self.tile_w * 3

    @property
    def sub_dim(self) -> int:
        """Per-sub-quantizer dimension = tile_dim // m_sub_quantizers."""
        return self.tile_dim // self.m_sub_quantizers

    @property
    def bits_per_tile(self) -> int:
        """Total bits per tile after PQ encoding."""
        # log2(ksub) per sub-quantizer × M sub-quantizers
        import math

        return int(self.m_sub_quantizers * math.ceil(math.log2(self.ksub_codebook_size)))

    @property
    def bytes_per_pair_codeword_stream_raw(self) -> int:
        """Per-pair codeword stream raw bytes (uncompressed)."""
        return (self.tiles_per_pair * self.bits_per_tile + 7) // 8


def _ensure_mlx_available() -> Any:
    """Lazy import MLX. Raises with actionable message if not installed.

    Sister substrates' canonical pattern per Catalog #1 device-selection-
    defaults discipline + axis-3 portability per operator directive #3
    2026-05-26.
    """
    return require_mlx_core()


# ---------------------------------------------------------------------------
# MLX primitive 1: PQ codebook gather
# ---------------------------------------------------------------------------


def mlx_pq_codebook_gather(codebook_mx: Any, indices_mx: Any) -> Any:
    """Gather PQ centroids from codebook by integer indices (MLX primitive).

    Canonical MLX mirror of
    ``tac.substrates.faiss_ivf_pq_residual.numpy_reference.pq_codebook_gather``.

    Args:
        codebook_mx: shape (M, ksub, sub_dim) float32 MLX array
            (per-sub-quantizer centroids)
        indices_mx: shape (..., M) int32 MLX array (per-tile codeword indices;
            caller MUST cast uint16/uint8 numpy → int32 before passing to MLX
            because ``mx.take`` requires integer indices)

    Returns:
        Per-tile reconstructed vector shape (..., M, sub_dim) float32 MLX
        array; stack of per-sub-quantizer gathered centroids along axis -2.

    Drift bound vs numpy reference: 0 (byte-identical; integer-index gather
    + exact float copy; no matmul / no fp32 accumulation involved).

    Per CLAUDE.md "MLX portable-local-substrate authority" non-promotable
    contract: caller MUST tag downstream artifacts as
    ``[macOS-MLX research-signal]`` with ``score_claim=False`` +
    ``promotion_eligible=False`` per Catalog #341 routing-markers.
    """
    mx = _ensure_mlx_available()
    # codebook_mx shape: (M, ksub, sub_dim); indices_mx shape: (..., M)
    M = int(codebook_mx.shape[0])
    if int(indices_mx.shape[-1]) != M:
        raise ValueError(
            f"indices last-dim {int(indices_mx.shape[-1])} != codebook M {M}"
        )
    # Per-sub-quantizer take: for each m, indices_mx[..., m] is the (...,)
    # shaped int32 index array into codebook_mx[m] (shape (ksub, sub_dim));
    # mx.take returns (..., sub_dim).
    per_sub_gathered = []
    for m in range(M):
        # codebook_mx[m] is shape (ksub, sub_dim); indices_mx[..., m] is (...,)
        m_indices = indices_mx[..., m]  # type: ignore[index]
        sub_gathered = mx.take(codebook_mx[m], m_indices, axis=0)  # (..., sub_dim)
        per_sub_gathered.append(sub_gathered)
    # Stack to (..., M, sub_dim) along axis -2
    return mx.stack(per_sub_gathered, axis=-2)


# ---------------------------------------------------------------------------
# MLX primitive 2: per-tile vector reassemble (PQ decode → flat tile)
# ---------------------------------------------------------------------------


def mlx_pq_reconstruct_tile_vectors(
    codebook_mx: Any,
    indices_mx: Any,
) -> Any:
    """Reconstruct per-tile flat vectors from PQ codebook + indices (MLX).

    Canonical MLX mirror of
    ``tac.substrates.faiss_ivf_pq_residual.numpy_reference.pq_reconstruct_tile_vectors``.

    Args:
        codebook_mx: shape (M, ksub, sub_dim) float32 MLX array
        indices_mx: shape (num_tiles, M) int32 MLX array

    Returns:
        Per-tile flat vector shape (num_tiles, M * sub_dim) float32 MLX
        array; composition of mlx_pq_codebook_gather + mx.reshape.

    Drift bound vs numpy reference: 0 (byte-identical; integer-index gather
    + exact reshape).
    """
    mx = _ensure_mlx_available()
    gathered = mlx_pq_codebook_gather(codebook_mx, indices_mx)
    # gathered shape: (num_tiles, M, sub_dim)
    num_tiles = int(gathered.shape[0])
    M = int(gathered.shape[1])
    sub_dim = int(gathered.shape[2])
    return mx.reshape(gathered, (num_tiles, M * sub_dim))


# ---------------------------------------------------------------------------
# MLX primitive 3: tile reassemble (flat tiles → per-pair RGB)
# ---------------------------------------------------------------------------


def mlx_tiles_to_frame_nhwc(
    tiles_mx: Any,
    *,
    frame_h: int,
    frame_w: int,
    tile_h: int,
    tile_w: int,
) -> Any:
    """Reassemble per-pair tile array into frame NHWC layout (MLX primitive).

    Canonical MLX mirror of
    ``tac.substrates.faiss_ivf_pq_residual.numpy_reference.tiles_to_frame_nhwc``.

    Tile ordering: row-major (idx 0 = top-left, idx grid_w = below top-left).
    Uses MLX-native reshape + concatenate to avoid the Python-loop scatter
    pattern of the numpy reference; result is byte-identical because the
    underlying tile layout is the same row-major contiguous reshape.

    Args:
        tiles_mx: shape (num_tiles, tile_h * tile_w * 3) float32 MLX array
        frame_h: output frame height (must equal grid_h * tile_h)
        frame_w: output frame width (must equal grid_w * tile_w)
        tile_h: per-tile height
        tile_w: per-tile width

    Returns:
        Frame array shape (frame_h, frame_w, 3) float32 MLX; channel-last
        NHWC layout (no batch dim — matches numpy reference signature).

    Drift bound vs numpy reference: 0 (reshape/concatenate are exact).
    """
    mx = _ensure_mlx_available()
    if frame_h % tile_h != 0 or frame_w % tile_w != 0:
        raise ValueError(
            f"frame_h={frame_h} frame_w={frame_w} must be divisible by "
            f"tile_h={tile_h}, tile_w={tile_w}"
        )
    grid_h = frame_h // tile_h
    grid_w = frame_w // tile_w
    num_tiles_expected = grid_h * grid_w
    if int(tiles_mx.shape[0]) != num_tiles_expected:
        raise ValueError(
            f"tiles count {int(tiles_mx.shape[0])} != grid_h*grid_w = "
            f"{num_tiles_expected}"
        )
    if int(tiles_mx.shape[1]) != tile_h * tile_w * 3:
        raise ValueError(
            f"tiles dim {int(tiles_mx.shape[1])} != tile_h*tile_w*3 = "
            f"{tile_h * tile_w * 3}"
        )
    # Step 1: reshape (num_tiles, tile_h*tile_w*3) → (grid_h, grid_w, tile_h, tile_w, 3)
    # tile g is at (g // grid_w, g % grid_w); reshape first to (grid_h, grid_w, ...)
    reshaped = mx.reshape(tiles_mx, (grid_h, grid_w, tile_h, tile_w, 3))
    # Step 2: transpose to (grid_h, tile_h, grid_w, tile_w, 3) so the
    # subsequent reshape collapses (grid_h, tile_h) → frame_h and
    # (grid_w, tile_w) → frame_w in the correct row-major order
    transposed = mx.transpose(reshaped, (0, 2, 1, 3, 4))
    # Step 3: reshape to (frame_h, frame_w, 3)
    return mx.reshape(transposed, (frame_h, frame_w, 3))


# ---------------------------------------------------------------------------
# MLX primitive 4: uint8 cast at output (canonical Catalog #205 sister rounding)
# ---------------------------------------------------------------------------


def mlx_to_uint8(x_mx: Any) -> Any:
    """Canonical uint8 cast: clip [0, 255] + round + cast (MLX primitive).

    Canonical MLX mirror of
    ``tac.substrates.faiss_ivf_pq_residual.numpy_reference.to_uint8``.

    Per Catalog #205 sister rounding discipline; deterministic.

    Args:
        x_mx: float MLX array (any shape)

    Returns:
        uint8 MLX array (same shape) with values clipped to [0, 255] and
        rounded.

    Drift bound vs numpy reference: 0 expected for inputs that round
    unambiguously; per CLAUDE.md sister "banker's rounding" semantics MLX's
    ``mx.round`` uses round-half-to-even matching ``np.round``.
    """
    mx = _ensure_mlx_available()
    return mx.clip(mx.round(x_mx), 0, 255).astype(mx.uint8)


# ---------------------------------------------------------------------------
# MLX primitive 5: per-pair residual decode (full composition)
# ---------------------------------------------------------------------------


def mlx_decode_per_pair_residual(
    codebook_mx: Any,
    codewords_mx: Any,
    *,
    tile_h: int,
    tile_w: int,
    residual_scale: float,
) -> Any:
    """PQ decode + tile reassemble for one pair (MLX primitive).

    Canonical MLX mirror of
    ``tac.substrates.faiss_ivf_pq_residual.inflate._decode_per_pair_residual``
    (the canonical PyTorch baseline algorithm source-of-truth).

    Composes mlx_pq_reconstruct_tile_vectors + mlx_tiles_to_frame_nhwc +
    multiplicative residual scaling.

    Args:
        codebook_mx: shape (M, ksub, sub_dim) float32 MLX array (shared
            across pairs)
        codewords_mx: shape (tiles_per_pair, M) int32 MLX array (per-pair
            indices; caller MUST cast uint16/uint8 numpy → int32 before
            passing to MLX)
        tile_h: per-tile height
        tile_w: per-tile width
        residual_scale: per-pair residual scale factor

    Returns:
        Per-pair RGB residual shape (EVAL_HW[0], EVAL_HW[1], 3) float32
        MLX array in [-residual_scale, residual_scale].

    Drift bound vs numpy reference + PyTorch baseline: 0 expected for the
    structural primitives (gather + reshape + concatenate + scalar multiply
    are exact in fp32); end-to-end MLX vs PyTorch parity is verified via
    sister test ``test_mlx_pytorch_decode_per_pair_residual_parity``.
    """
    _ensure_mlx_available()
    # Step 1: PQ decode → flat tile vectors
    tile_vectors_mx = mlx_pq_reconstruct_tile_vectors(codebook_mx, codewords_mx)
    # Step 2: tile reassemble → per-pair frame at EVAL_HW
    frame_mx = mlx_tiles_to_frame_nhwc(
        tile_vectors_mx,
        frame_h=EVAL_HW[0],
        frame_w=EVAL_HW[1],
        tile_h=tile_h,
        tile_w=tile_w,
    )
    # Step 3: scale to residual range
    return frame_mx * float(residual_scale)


# ---------------------------------------------------------------------------
# Cost estimators (Dykstra-feasibility surface; no MLX needed)
# ---------------------------------------------------------------------------


def estimate_per_pair_codeword_bytes_raw(cfg: FaissIVFPQResidualConfig) -> int:
    """Estimate per-pair codeword stream raw bytes (uncompressed) for a config.

    Used for Dykstra-feasibility check at design time (Catalog #296).
    """
    return cfg.bytes_per_pair_codeword_stream_raw


def estimate_archive_bytes(cfg: FaissIVFPQResidualConfig) -> int:
    """Estimate FAISSPQ1 archive size in bytes for a given config.

    Used for Dykstra-feasibility check at design time (Catalog #296) without
    instantiating MLX or actually packing.

    Breakdown per design memo §"Predicted ΔS band":
    - PQ codebook raw: M × ksub × sub_dim × 4 (float32)
    - PQ codebook brotli-compressed: ~30% of raw (high redundancy)
    - Per-pair codeword stream raw: num_pairs × bytes_per_pair_codeword_stream_raw
    - Per-pair codeword stream brotli-compressed: ~60% of raw (medium entropy)
    - Header + meta JSON: ~256 bytes
    """
    codebook_raw = (
        cfg.m_sub_quantizers * cfg.ksub_codebook_size * cfg.sub_dim * 4
    )  # float32
    codebook_compressed = int(codebook_raw * 0.3)

    codewords_raw = cfg.num_pairs * cfg.bytes_per_pair_codeword_stream_raw
    codewords_compressed = int(codewords_raw * 0.6)

    header_meta = 256

    return codebook_compressed + codewords_compressed + header_meta


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240: full main raises NotImplementedError.

    The L0 scaffold ships design + MLX-native primitives + numpy reference +
    PyTorch inflate + archive grammar + tests + smoke trainer stub. The full
    MLX training path is gated by Phase 2 council symposium per Catalog
    #325 + operator-authorized paid-CUDA dispatch eligibility per Catalog
    #1265 MLX-first contest-equivalence gate.
    """
    raise NotImplementedError(
        "faiss_ivf_pq_residual full main NOT YET IMPLEMENTED — L0 SCAFFOLD "
        "posture per Catalog #240. Phase 2 council symposium per Catalog "
        "#325 + Catalog #1265 MLX↔PyTorch parity gate REQUIRED before any "
        "paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md "
        "for the Phase 2+ roadmap."
    )


__all__ = [
    "EVAL_HW",
    "EVIDENCE_GRADE",
    "EVIDENCE_TAG",
    "LANE_ID",
    "SCHEMA_VERSION",
    "FaissIVFPQResidualConfig",
    "_ensure_mlx_available",
    "_full_main",
    "estimate_archive_bytes",
    "estimate_per_pair_codeword_bytes_raw",
    "mlx_decode_per_pair_residual",
    "mlx_pq_codebook_gather",
    "mlx_pq_reconstruct_tile_vectors",
    "mlx_tiles_to_frame_nhwc",
    "mlx_to_uint8",
]
