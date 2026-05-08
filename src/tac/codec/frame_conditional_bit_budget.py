"""Frame-conditional bit-budget allocator (Track 1 council prescription, Decision 5).

Idea
────
Different frames in the contest video have DIFFERENT contest-distortion
sensitivity. High-motion / textured / occluded frames yield more contest-score
improvement per allocated bit than low-motion / low-texture frames. This
module exposes pure-CPU allocators that distribute a fixed total bit budget
across N frames proportionally to a per-frame complexity proxy.

Two-step pipeline
─────────────────
1. ``compute_per_frame_complexity(video_path, n_frames)`` reads frames from a
   contest-shape mkv via pyav and returns a 1-D numpy array of complexity
   scalars (edge density × pixel variance × frame-to-frame difference). All
   scalars are non-negative. Pure CPU + numpy + pyav; no torch, no GPU.

2. ``allocate_per_frame_bits(complexities, total_bit_budget, eta=1.0,
   floor=0.5, cap=2.0)`` returns the same-shape budget vector that sums to
   ``total_bit_budget`` (within rounding tolerance) where each entry is
   proportional to ``complexities[i]**eta`` clamped between
   ``floor*avg`` and ``cap*avg``.

Mathematics
───────────
Let ``c_i = complexities[i]``, ``w_i = c_i**eta``. Without floor/cap the
proportional allocation is::

    b_i = total * w_i / sum(w_j)

The floor/cap clamps after normalisation, then re-distributes the residual
mass evenly across the *unclamped* frames. We iterate up to a small fixed
number of times so that all clamps are honoured AND ``sum(b_i) == total``
within ±0.5 bits.

Edge cases
──────────
* ``eta=0`` → all weights equal → uniform allocation regardless of
  complexity.
* ``n_frames=1`` → single-element vector containing ``total``.
* identical complexities → uniform regardless of ``eta``.
* zero complexity total (every frame zero) → uniform fallback (prevents
  divide-by-zero; honours floor by symmetry).

CLAUDE.md compliance
────────────────────
* No torch import → no MPS / CUDA path.
* No scorer load → CPU-prep only.
* Output is a budget vector. Score impact requires per-frame score-marginal
  evidence which this module does NOT supply (dispatch_blocker:
  ``awaiting_per_frame_score_marginal``).

Cross-references
────────────────
* ``tac.score_geometry``: closed-form contest score formula. The frame-axis
  marginal is *one of three* (seg / pose / rate); this allocator targets
  the rate axis (charged bytes per frame).
* Memory ``feedback_pr101_frame_conditional_bit_budget_*_20260508.md``: the
  empirical anchor on PR101's monolithic latent stream.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA",
    "FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS",
    "ComplexityComponents",
    "allocate_per_frame_bits",
    "apply_frame_conditional_q_bits",
    "build_frame_conditional_wire_contract",
    "compute_per_frame_complexity",
    "pack_frame_conditional_latent_codes",
    "pack_frame_conditional_q_bits",
    "unpack_frame_conditional_latent_codes",
    "unpack_frame_conditional_q_bits",
]


FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA = "tac_frame_conditional_latent_wire.v1"
FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS = 3
FRAME_CONDITIONAL_Q_BITS_MIN = 1
FRAME_CONDITIONAL_Q_BITS_MAX = 8
FRAME_CONDITIONAL_WIRE_CONTRACT_CLEARED_BLOCKERS = (
    "per_pair_bit_width_schema_change_requires_inflate_path_update",
)
FRAME_CONDITIONAL_WIRE_CONTRACT_REMAINING_BLOCKERS = (
    "frame_conditional_packet_runtime_patch_not_built",
    "frame_conditional_runtime_consumption_proof_missing",
)


@dataclass(frozen=True)
class ComplexityComponents:
    """The three multiplicative factors of the per-frame complexity proxy.

    Each array has length ``n_frames``. The top-level complexity is
    ``edge_density * pixel_variance * frame_difference`` element-wise.
    """

    edge_density: np.ndarray
    pixel_variance: np.ndarray
    frame_difference: np.ndarray

    @property
    def complexity(self) -> np.ndarray:
        return self.edge_density * self.pixel_variance * self.frame_difference


# ─────────────────────────────────────────────────────────────────────────
# Runtime side-info wire contract
# ─────────────────────────────────────────────────────────────────────────


def _normalise_q_bits_per_pair(
    q_bits_per_pair: Sequence[float] | np.ndarray,
    *,
    expected_pairs: int | None = None,
) -> np.ndarray:
    arr = np.asarray(q_bits_per_pair, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"q_bits_per_pair must be 1-D, got shape {arr.shape}")
    if expected_pairs is not None and arr.size != expected_pairs:
        raise ValueError(
            f"q_bits_per_pair length {arr.size} != expected_pairs {expected_pairs}"
        )
    if arr.size == 0:
        raise ValueError("q_bits_per_pair must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("q_bits_per_pair must contain only finite values")
    if (arr < FRAME_CONDITIONAL_Q_BITS_MIN).any() or (
        arr > FRAME_CONDITIONAL_Q_BITS_MAX
    ).any():
        raise ValueError(
            "q_bits_per_pair values must be in "
            f"[{FRAME_CONDITIONAL_Q_BITS_MIN}, {FRAME_CONDITIONAL_Q_BITS_MAX}]"
        )
    q_bits = np.floor(arr).astype(np.uint8)
    if (q_bits < FRAME_CONDITIONAL_Q_BITS_MIN).any() or (
        q_bits > FRAME_CONDITIONAL_Q_BITS_MAX
    ).any():
        raise ValueError("floored q_bits_per_pair values are outside uint3 range")
    return q_bits


def _validate_q_codes_pair_first(q_pair_first: np.ndarray) -> np.ndarray:
    q = np.asarray(q_pair_first)
    if q.ndim != 2:
        raise ValueError(f"q_pair_first must be 2-D, got shape {q.shape}")
    if q.size == 0:
        raise ValueError("q_pair_first must be non-empty")
    if not np.issubdtype(q.dtype, np.integer):
        raise ValueError(f"q_pair_first must have integer dtype, got {q.dtype}")
    if (q < 0).any() or (q > 255).any():
        raise ValueError("q_pair_first values must be uint8-compatible")
    return q.astype(np.uint8, copy=False)


def _append_msb_bits(out: bytearray, bit_pos: int, value: int, width: int) -> int:
    if width < 0:
        raise ValueError(f"bit width must be non-negative, got {width}")
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value {value} does not fit in {width} bits")
    for shift in range(width - 1, -1, -1):
        if value & (1 << shift):
            out[bit_pos // 8] |= 1 << (7 - (bit_pos % 8))
        bit_pos += 1
    return bit_pos


def _read_msb_bits(data: bytes, bit_pos: int, width: int) -> tuple[int, int]:
    value = 0
    for _ in range(width):
        if bit_pos >= len(data) * 8:
            raise ValueError("bitstream truncated")
        value = (value << 1) | ((data[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1)
        bit_pos += 1
    return value, bit_pos


def _assert_zero_padding(data: bytes, bit_pos: int) -> None:
    while bit_pos < len(data) * 8:
        if (data[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1:
            raise ValueError("non-zero padding bits in frame-conditional bitstream")
        bit_pos += 1


def pack_frame_conditional_q_bits(
    q_bits_per_pair: Sequence[float] | np.ndarray,
) -> bytes:
    """Pack per-pair q-bit widths as fixed 3-bit MSB-first values.

    The wire value is ``q_bits - 1`` so the valid decoded range is exactly
    ``1..8``. For PR101's 600 frame-pairs this is 600 * 3 / 8 = 225 bytes.
    The payload is intentionally headerless: the surrounding A5 packet schema
    fixes the pair count, and this function fails closed on length mismatch
    during decode.
    """
    q_bits = _normalise_q_bits_per_pair(q_bits_per_pair)
    total_bits = q_bits.size * FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS
    out = bytearray((total_bits + 7) // 8)
    bit_pos = 0
    for value in q_bits:
        bit_pos = _append_msb_bits(
            out,
            bit_pos,
            int(value) - 1,
            FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS,
        )
    return bytes(out)


def unpack_frame_conditional_q_bits(
    data: bytes,
    *,
    n_pairs: int,
) -> np.ndarray:
    """Decode the fixed 3-bit A5 q-bit side-info stream."""
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    expected_bits = n_pairs * FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS
    expected_bytes = (expected_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"q-bit side-info length {len(data)} != expected {expected_bytes}"
        )
    out = np.empty(n_pairs, dtype=np.uint8)
    bit_pos = 0
    for i in range(n_pairs):
        code, bit_pos = _read_msb_bits(
            data,
            bit_pos,
            FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS,
        )
        out[i] = code + 1
    _assert_zero_padding(data, bit_pos)
    return out


def apply_frame_conditional_q_bits(
    q_pair_first: np.ndarray,
    q_bits_per_pair: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply per-pair q-bit truncation to PR101 uint8 latent codes.

    This is the decode-side semantic transform that A5's side-info authorizes:
    pair ``i`` retains the top ``q_bits[i]`` bits of each uint8 symbol and
    zeroes the low bits. It is idempotent and deterministic.
    """
    q = _validate_q_codes_pair_first(q_pair_first).copy()
    q_bits = _normalise_q_bits_per_pair(q_bits_per_pair, expected_pairs=q.shape[0])
    for i, bits in enumerate(q_bits):
        bit_count = int(bits)
        if bit_count == FRAME_CONDITIONAL_Q_BITS_MAX:
            continue
        shift = FRAME_CONDITIONAL_Q_BITS_MAX - bit_count
        q[i] = ((q[i].astype(np.uint16) >> shift) << shift).astype(np.uint8)
    return q


def pack_frame_conditional_latent_codes(
    q_pair_first: np.ndarray,
    q_bits_per_pair: Sequence[float] | np.ndarray,
) -> bytes:
    """Pack PR101 latent uint8 codes using per-pair variable-width q-bits.

    The q-bit side-info is not advisory here: the decoder needs it to know
    where each pair's symbol boundaries are in this bitstream.
    """
    q = _validate_q_codes_pair_first(q_pair_first)
    q_bits = _normalise_q_bits_per_pair(q_bits_per_pair, expected_pairs=q.shape[0])
    total_bits = int(q_bits.astype(np.uint64).sum()) * q.shape[1]
    out = bytearray((total_bits + 7) // 8)
    bit_pos = 0
    for pair_idx, bits in enumerate(q_bits):
        bit_count = int(bits)
        shift = FRAME_CONDITIONAL_Q_BITS_MAX - bit_count
        for symbol in q[pair_idx]:
            code = int(symbol) >> shift
            bit_pos = _append_msb_bits(out, bit_pos, code, bit_count)
    return bytes(out)


def unpack_frame_conditional_latent_codes(
    data: bytes,
    q_bits_per_pair: Sequence[float] | np.ndarray,
    *,
    latent_dim: int,
) -> np.ndarray:
    """Decode A5 variable-width latent codes into uint8 PR101 code space."""
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    q_bits = _normalise_q_bits_per_pair(q_bits_per_pair)
    expected_bits = int(q_bits.astype(np.uint64).sum()) * latent_dim
    expected_bytes = (expected_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"latent bitstream length {len(data)} != expected {expected_bytes}"
        )
    out = np.empty((q_bits.size, latent_dim), dtype=np.uint8)
    bit_pos = 0
    for pair_idx, bits in enumerate(q_bits):
        bit_count = int(bits)
        shift = FRAME_CONDITIONAL_Q_BITS_MAX - bit_count
        for dim_idx in range(latent_dim):
            code, bit_pos = _read_msb_bits(data, bit_pos, bit_count)
            out[pair_idx, dim_idx] = (code << shift) & 255
    _assert_zero_padding(data, bit_pos)
    return out


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_frame_conditional_wire_contract(
    q_bits_per_pair: Sequence[float] | np.ndarray,
    *,
    latent_dim: int,
    q_pair_first: np.ndarray | None = None,
) -> dict[str, object]:
    """Return no-score A5 wire-contract metadata for manifests/guards.

    This contract proves that side-info bytes are typed and decoder-consumed by
    local helpers. It does not claim an inflate/runtime packet exists; the
    returned blockers keep exact-eval readiness fail-closed until a packet-local
    PR101 inflate patch and runtime-consumption proof land.
    """
    q_bits = _normalise_q_bits_per_pair(q_bits_per_pair)
    sideinfo = pack_frame_conditional_q_bits(q_bits)
    decoded_q_bits = unpack_frame_conditional_q_bits(sideinfo, n_pairs=q_bits.size)
    q_bits_roundtrip_passed = bool(np.array_equal(decoded_q_bits, q_bits))
    record: dict[str, object] = {
        "schema": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_decoder_helper": (
            "tac.codec.frame_conditional_bit_budget."
            "unpack_frame_conditional_latent_codes"
        ),
        "sideinfo_decoder_helper": (
            "tac.codec.frame_conditional_bit_budget."
            "unpack_frame_conditional_q_bits"
        ),
        "wire_encoding": {
            "q_bits_per_pair": "3bit_msb_q_bits_minus_1",
            "latent_codes": "per_pair_msb_variable_width_uint8_top_bits",
            "latent_dim": int(latent_dim),
            "n_pairs": int(q_bits.size),
        },
        "q_bits_sideinfo": {
            "bytes": len(sideinfo),
            "bits_per_pair": FRAME_CONDITIONAL_Q_BITS_PER_PAIR_BITS,
            "sha256": _sha256_bytes(sideinfo),
            "q_bits_min": int(q_bits.min()),
            "q_bits_max": int(q_bits.max()),
            "q_bits_mean": float(q_bits.mean()),
        },
        "q_bits_roundtrip": {
            "passed": q_bits_roundtrip_passed,
            "decoded_sha256": _sha256_bytes(decoded_q_bits.tobytes()),
        },
        "decoder_helper_consumes_sideinfo_bytes": True,
        "cleared_blockers": list(FRAME_CONDITIONAL_WIRE_CONTRACT_CLEARED_BLOCKERS),
        "remaining_blockers": list(FRAME_CONDITIONAL_WIRE_CONTRACT_REMAINING_BLOCKERS),
    }
    if q_pair_first is not None:
        q = _validate_q_codes_pair_first(q_pair_first)
        if q.shape[0] != q_bits.size:
            raise ValueError(
                f"q_pair_first pair count {q.shape[0]} != q_bits length {q_bits.size}"
            )
        if q.shape[1] != latent_dim:
            raise ValueError(
                f"q_pair_first latent_dim {q.shape[1]} != requested {latent_dim}"
            )
        latent_payload = pack_frame_conditional_latent_codes(q, q_bits)
        decoded = unpack_frame_conditional_latent_codes(
            latent_payload,
            q_bits,
            latent_dim=latent_dim,
        )
        expected = apply_frame_conditional_q_bits(q, q_bits)
        record["latent_wire_payload"] = {
            "bytes": len(latent_payload),
            "sha256": _sha256_bytes(latent_payload),
            "source_q_codes_sha256": _sha256_bytes(q.tobytes()),
            "decoded_q_codes_sha256": _sha256_bytes(decoded.tobytes()),
            "score_affecting_payload_changed": bool(not np.array_equal(q, decoded)),
        }
        record["latent_decode_roundtrip"] = {
            "passed": bool(np.array_equal(decoded, expected)),
            "expected_truncated_q_codes_sha256": _sha256_bytes(expected.tobytes()),
        }
    return record


# ─────────────────────────────────────────────────────────────────────────
# Per-frame complexity proxy
# ─────────────────────────────────────────────────────────────────────────


def _luma_from_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Return BT.601 luma as float32 in [0, 255]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got shape {rgb.shape}")
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edge_density(luma: np.ndarray) -> float:
    """Mean absolute gradient magnitude — cheap Sobel-like proxy.

    Uses simple finite differences (no scipy dependency) so this stays in
    pure numpy. The mean absolute gradient is a non-negative scalar; it is
    bounded above by 255 / sqrt(2).
    """
    dx = np.abs(np.diff(luma, axis=1))
    dy = np.abs(np.diff(luma, axis=0))
    return float(0.5 * (dx.mean() + dy.mean()))


def _pixel_variance(luma: np.ndarray) -> float:
    """Per-frame luma variance (single non-negative scalar)."""
    return float(luma.var())


def _frame_difference(luma_curr: np.ndarray, luma_prev: np.ndarray | None) -> float:
    """Mean absolute difference vs previous luma. First frame: 0.0."""
    if luma_prev is None:
        return 0.0
    return float(np.abs(luma_curr - luma_prev).mean())


def compute_per_frame_complexity(
    video_path: str | Path,
    n_frames: int,
    *,
    return_components: bool = False,
) -> np.ndarray | ComplexityComponents:
    """Read up to ``n_frames`` frames and compute a per-frame complexity.

    Parameters
    ----------
    video_path : Path-like
        Path to a contest-shape video readable by pyav (e.g. ``upstream/videos/0.mkv``).
    n_frames : int
        Number of frames to read from the start of the stream.
    return_components : bool
        If True, return the underlying :class:`ComplexityComponents`. Default
        returns the multiplied 1-D complexity array.

    Returns
    -------
    np.ndarray or ComplexityComponents
        Length-``n_frames`` non-negative float64 array, OR the components
        dataclass if ``return_components=True``.

    Notes
    -----
    The first frame's frame-difference is 0.0 (no predecessor). To avoid
    that frame collapsing to zero complexity (and thus being clamped to the
    floor), we replace the first-frame difference with the median of the
    remaining differences before multiplying. This is a conventional
    convention for first-frame motion proxies.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    import av  # local import; keeps this module importable for unit tests

    edge = np.zeros(n_frames, dtype=np.float64)
    var = np.zeros(n_frames, dtype=np.float64)
    diff = np.zeros(n_frames, dtype=np.float64)

    prev_luma: np.ndarray | None = None
    captured = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if captured >= n_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            luma = _luma_from_rgb_uint8(rgb)
            edge[captured] = _edge_density(luma)
            var[captured] = _pixel_variance(luma)
            diff[captured] = _frame_difference(luma, prev_luma)
            prev_luma = luma
            captured += 1

    if captured < n_frames:
        raise ValueError(
            f"video {video_path} yielded only {captured} frames, requested {n_frames}"
        )

    # Convention: first-frame difference replaced with median of rest so it
    # is not artificially zero (which would multiply the whole row to zero).
    if n_frames > 1:
        diff[0] = float(np.median(diff[1:]))

    components = ComplexityComponents(
        edge_density=edge, pixel_variance=var, frame_difference=diff
    )
    if return_components:
        return components
    return components.complexity


# ─────────────────────────────────────────────────────────────────────────
# Bit budget allocator
# ─────────────────────────────────────────────────────────────────────────


def _clamp_and_redistribute(
    weights: np.ndarray,
    total: float,
    floor: float,
    cap: float,
    *,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> np.ndarray:
    """Distribute ``total`` proportional to ``weights`` then iteratively clamp.

    Algorithm
    ---------
    1. Initial proportional allocation: ``b_i = total * w_i / sum(w_j)``.
    2. Compute uniform mean ``mu = total / n``.
    3. Define ``lo = floor * mu`` and ``hi = cap * mu``.
    4. Clamp every entry to ``[lo, hi]``; track which entries were clamped.
    5. The clamped entries consume a fixed amount; the residual must be
       redistributed across the unclamped entries proportional to their
       (un-normalised) weights.
    6. Repeat until no entry is clamped in an iteration OR ``max_iter`` hit.

    The ``floor`` and ``cap`` are expressed as multiples of the uniform
    average (e.g. ``floor=0.5`` means every frame gets ≥ 50 % of the average
    bit count; ``cap=2.0`` means no frame gets > 200 % of the average).
    """
    n = weights.size
    mu = total / n
    lo = floor * mu
    hi = cap * mu

    if lo > mu or hi < mu:
        # Floor/cap interval excludes the mean — the constraint is
        # infeasible (every frame must equal mu in the limit). Snap to mu.
        return np.full(n, mu, dtype=np.float64)

    # Pre-conditions: floor ≤ 1 ≤ cap and total bits must satisfy
    # n * lo ≤ total ≤ n * hi (always true by construction since
    # n * lo = n * floor * mu = floor * total ≤ total and same for cap).

    # Indices that are NOT yet locked at floor/cap.
    locked_low = np.zeros(n, dtype=bool)
    locked_high = np.zeros(n, dtype=bool)
    allocation = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        active = ~(locked_low | locked_high)
        if not active.any():
            break

        active_weights = weights[active]
        active_weight_sum = float(active_weights.sum())

        # Bits already consumed by locked entries.
        consumed = float(allocation[locked_low].sum() + allocation[locked_high].sum())
        remaining = total - consumed

        if active_weight_sum <= 0.0:
            # Active entries have zero weight: split remaining evenly.
            allocation[active] = remaining / active.sum() if active.sum() else 0.0
        else:
            allocation[active] = remaining * active_weights / active_weight_sum

        # Lock at most ONE side per iteration so the residual mass on the
        # unlocked side can flow to the other side. Locking both sides in
        # one step can shrink the unlocked pool to empty before the
        # residual is redistributed (causing sum != total).
        new_lock_high = (~locked_low) & (~locked_high) & (allocation > hi + tol)
        if new_lock_high.any():
            allocation[new_lock_high] = hi
            locked_high |= new_lock_high
            continue

        new_lock_low = (~locked_low) & (~locked_high) & (allocation < lo - tol)
        if new_lock_low.any():
            allocation[new_lock_low] = lo
            locked_low |= new_lock_low
            continue

        break

    return allocation


def allocate_per_frame_bits(
    complexities: Sequence[float] | np.ndarray,
    total_bit_budget: float,
    *,
    eta: float = 1.0,
    floor: float = 0.5,
    cap: float = 2.0,
) -> np.ndarray:
    """Allocate ``total_bit_budget`` across frames proportional to complexity.

    Parameters
    ----------
    complexities : 1-D array-like of non-negative floats
        Per-frame complexity proxy (e.g. the output of
        :func:`compute_per_frame_complexity`).
    total_bit_budget : float
        Total number of bits to distribute. Must be ≥ 0.
    eta : float, default 1.0
        Concentration exponent. ``eta=0`` collapses to uniform allocation;
        ``eta=1`` is purely proportional; ``eta>1`` concentrates more bits
        on high-complexity frames; ``eta<0`` is permitted but unusual
        (allocates more bits to low-complexity frames). Must be finite.
    floor : float, default 0.5
        Minimum per-frame budget as a multiple of the uniform mean
        ``mu = total_bit_budget / n``. Must satisfy ``0 ≤ floor ≤ 1``.
    cap : float, default 2.0
        Maximum per-frame budget as a multiple of the uniform mean. Must
        satisfy ``cap ≥ 1``.

    Returns
    -------
    np.ndarray, shape (n,), float64
        Per-frame budget. The vector sums to ``total_bit_budget`` within
        ±0.5 (rounding tolerance).
    """
    arr = np.asarray(complexities, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"complexities must be 1-D, got shape {arr.shape}")
    n = arr.size
    if n == 0:
        raise ValueError("complexities must be non-empty")
    if total_bit_budget < 0:
        raise ValueError(f"total_bit_budget must be non-negative, got {total_bit_budget}")
    if (arr < 0).any():
        raise ValueError("complexities must be non-negative")
    if not np.isfinite(eta):
        raise ValueError(f"eta must be finite, got {eta}")
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if cap < 1.0:
        raise ValueError(f"cap must be >= 1, got {cap}")

    if n == 1:
        return np.array([float(total_bit_budget)], dtype=np.float64)

    if total_bit_budget == 0.0:
        return np.zeros(n, dtype=np.float64)

    # Compute weights = complexity^eta. eta=0 → uniform regardless of input.
    if eta == 0.0:
        weights = np.ones(n, dtype=np.float64)
    elif arr.sum() == 0.0:
        # All zero complexities → uniform fallback (still honours floor by
        # symmetry — every entry equals mu, which is ≥ lo by definition).
        weights = np.ones(n, dtype=np.float64)
    else:
        # Avoid 0**negative_eta by treating exact zeros as a tiny epsilon
        # relative to the smallest positive weight.
        if eta < 0:
            positive_min = float(arr[arr > 0].min()) if (arr > 0).any() else 1.0
            safe = np.where(arr > 0, arr, positive_min * 1e-6)
            weights = safe ** eta
        else:
            weights = arr ** eta

    return _clamp_and_redistribute(weights, float(total_bit_budget), floor, cap)
