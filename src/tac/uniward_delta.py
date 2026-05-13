"""Lane C: UNIWARD δ-injection on rendered frames at compress-time.

Detector-informed embedding in the Yousfi-2022 sense, on a contest where the
scorer IS the detector. Adds a sparse, variance-weighted, L∞-bounded
perturbation δ to renderer output BEFORE the camera-resolution upscale.
δ is optimized at compress-time against the actual PoseNet (and optional
SegNet) loss; at inflate-time it is a pure additive lookup table — NO
scorer is loaded. This satisfies Yousfi PR#35 strict-scorer-rule.

THEORY (Holub & Fridrich 2014, "Universal distortion function for
steganography in an arbitrary domain"):

    The S-UNIWARD distortion ρ(i, j) for embedding ±1 at pixel (i, j) is
    inversely proportional to the local wavelet residual energy:

        ρ(i, j) = Σ_d Σ_(u,v) | W_d(I)(u, v) |^{-1} * | K_d(i-u, j-v) |

    where W_d are directional wavelet filters (1st-level Daubechies
    decomposition) and K_d are the corresponding synthesis kernels.
    The intuition: textured pixels have high wavelet residual → low
    embedding cost → safe to perturb. Smooth pixels have near-zero
    wavelet residual → high embedding cost → DO NOT perturb.

    For Lane C we substitute the full Daubechies bank with directional
    Haar (already present in tac.fridrich._haar_wavelet_filters), which
    is the same approximation Yousfi 2017 ("Detector-Informed JPEG
    Steganography") uses in the digital-side application, and is what
    fridrich.compute_pixel_cost_map(method='uniward') already implements.

    The cost map gives PER-PIXEL L∞ budget:

        |δ(i, j, c)| ≤ L∞_BUDGET * (1 - cost_normalised(i, j))

    plus a global L∞ cap of 4/255 ≈ 0.0157 in [0, 1] units, equivalently
    4 in [0, 255] units. The Fridrich council bet is that the EfficientNet-B2
    SegNet stride-2 stem is BLIND to perturbations bounded this way, while
    PoseNet (FastViT-T12 on YUV6) sees the pose-error-canceling component.

DETERMINISTIC REPRODUCIBILITY (CLAUDE.md non-negotiable):

    Same renderer + masks + poses + seed + (compress device) → same δ bytes.

    This is enforced by:
    1. ``torch.manual_seed(seed)`` + ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
       BEFORE any cuBLAS/cuDNN call.
    2. Adam optimizer over a deterministic batched loop (sorted pair
       indices, fixed batch size).
    3. Top-K sparsity selection by ``torch.topk`` is deterministic
       (CUDA topk has tie-breaking by lower index).
    4. Quantization to int8 via ``round-to-nearest-even`` (PyTorch default).
    5. The packed bytes are written via ``zlib.compress(level=9)`` which
       is deterministic given identical input bytes.

    Mac-MPS vs CUDA WILL produce different δ bytes because the underlying
    PoseNet softmax + YUV6 chroma kernels differ (CLAUDE.md drift table).
    Lane C MUST run on CUDA — both for the bytes-stable provenance and to
    avoid the 23x PoseNet drift on MPS.

WIRE FORMAT (delta.bin):

    [4 bytes]   magic = b"UWD1"
    [4 bytes]   header_len (uint32 LE)
    [N bytes]   header_json (utf-8) with:
                  schema_version, n_frames, H, W, n_kept, l_inf_budget,
                  quantize_bits (always 8), scale, seed, sha256s of
                  (renderer, masks, poses) for provenance.
    [n_kept*4]  flat pixel indices (uint32 LE), shape =
                  frame_idx * H * W * 3 + (y * W + x) * 3 + channel
                  i.e. "fully flat in (frame, y, x, c)" order.
                  Channel-flat means the apply path is one scatter_add
                  with no fan-out per pixel (matches Hotz R1 #1 layout
                  used in gradient_corrections).
    [n_kept]    quantized δ values (int8 LE), dequant = v * scale / 127.

The whole thing is zlib-compressed at level 9 (matches gradient_corrections
codec). For a 1% sparsity at 1200x384x512x3 ≈ 7M pixel-channels, n_kept ≈
70k → header(~250B) + 70k*5 = ~350KB raw → ~100KB compressed; budget says
≤5KB total which forces sparsity ≤0.014% of pixel-channels (~1k slots) and
hard top-K selection. The optimizer below enforces the budget by
``--target-bytes`` knob on the packer.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


__all__ = [
    "MAGIC",
    "SCHEMA_VERSION",
    "COMPLIANCE_PENDING",
    "COMPLIANCE_APPROVED",
    "COMPLIANCE_REJECTED",
    "compute_uniward_cost_map",
    "pack_sparse_delta",
    "unpack_sparse_delta",
    "apply_delta_to_frame",
    "DeltaSpec",
    "DETECTOR_COST_SCHEMA_VERSION",
    "build_detector_cost_manifest",
]


MAGIC = b"UWD1"  # UniWard Delta v1
SCHEMA_VERSION = 2  # bumped from 1 in the Codex R5 HIGH fix below.
DETECTOR_COST_SCHEMA_VERSION = 1

# ── Compliance gate (Codex R5 HIGH — silent contest-noncompliance) ──────
#
# Lane C δ.bin is a SCORER-DERIVED artifact (compress-time PoseNet+SegNet
# gradients on the GT video). Yousfi PR #35 strict-scorer-rule may class
# this as "derived asset" → contest-noncompliant. Until the council issues
# a binding ruling, every produced δ.bin must be marked PENDING_RULING in
# the wire-format header. The archive builder hard-fails when bundling
# such a δ unless --allow-pending-compliance is passed (and recorded into
# provenance). The inflate path prints a [lane-c-pending-ruling] banner
# whenever a pending-ruling δ is loaded so eval logs carry the audit trail.
#
# Allowed values: "pending_ruling", "approved", "rejected". The first is
# the safe default for any newly-built δ. "approved" requires explicit
# council action. "rejected" is reserved for archived experiments that
# turned out to violate the rule — kept only so we can re-load and inspect
# them without crashing.
COMPLIANCE_PENDING = "pending_ruling"
COMPLIANCE_APPROVED = "approved"
COMPLIANCE_REJECTED = "rejected"
_VALID_COMPLIANCE_STATUSES = frozenset({
    COMPLIANCE_PENDING, COMPLIANCE_APPROVED, COMPLIANCE_REJECTED,
})


@dataclass(frozen=True)
class DeltaSpec:
    """Decoded sparse-δ payload, ready for fast on-device application.

    Attributes:
        n_frames: total renderer-output frames the δ targets.
        H, W: renderer-side spatial dims (typically 384 x 512).
        per_frame_local_idx: list of length ``n_frames``. Each entry is
            either ``None`` (no δ for that frame) or a 1-D int64 tensor of
            FLAT (y * W * 3 + x * 3 + c) indices into the (H, W, 3) frame.
        per_frame_dequant: parallel list of float32 tensors, same length
            as the index tensor for each frame. Already dequantized
            (multiplied by scale / 127).
        l_inf_budget: the per-pixel L∞ bound used to clip during pack.
            Stored only for provenance / sanity checks at apply time.
        scale: int8 dequant scale (max |δ| in float units).
        n_kept: total non-zero pixel-channels across all frames.
        any_delta: ``False`` ⇒ caller can skip the apply path entirely.
        compliance_status: one of ``"pending_ruling"`` (default for any new
            δ until council ruling lands), ``"approved"`` (explicit council
            sign-off recorded), or ``"rejected"`` (kept for forensics only,
            never bundle). Legacy v1 blobs that lack the field decode as
            ``"pending_ruling"`` so old-format δ also fail closed in the
            archive builder.
    """

    n_frames: int
    H: int
    W: int
    per_frame_local_idx: list[torch.Tensor | None]
    per_frame_dequant: list[torch.Tensor | None]
    l_inf_budget: float
    scale: float
    n_kept: int
    any_delta: bool
    compliance_status: str = COMPLIANCE_PENDING


# ── 1. Cost map (re-uses tac.fridrich.compute_pixel_cost_map for parity) ──


def compute_uniward_cost_map(
    frames_bchw: torch.Tensor,
    *,
    sigma: float = 1e-4,
) -> torch.Tensor:
    """S-UNIWARD per-pixel embedding cost on ALREADY-RENDERED frames.

    Direct wrapper around ``tac.fridrich._uniward_cost`` so we don't
    duplicate the directional-Haar wavelet math. Returns the unnormalized
    cost (higher = more textured = SAFER to embed).

    NOTE on the inversion convention used downstream:
    ``compute_pixel_cost_map`` (the public Fridrich API) returns
    "high cost = preserve" because it's used inside augmented Lagrangian
    optimization where you multiply by ``(1 - cost)`` to push perturbations
    INTO low-cost (smooth) pixels. For Lane C we want the OPPOSITE
    convention — high "embedding capacity" in textured regions — because
    PoseNet+SegNet are robust to textured perturbations (Fridrich
    inverse-steganalysis principle #1: UNIWARD textured regions are
    undetectable). We therefore use the raw ``_uniward_cost`` (which IS
    "high in textured regions") DIRECTLY, without inverting.

    Args:
        frames_bchw: (B, 3, H, W) float frames in [0, 255].
        sigma: stabilizer for low-energy regions. Larger → smoother map,
            less aggressive sparsity. Default 1e-4 matches fridrich.

    Returns:
        (B, H, W) float tensor on the same device as input. Unnormalized.
        Caller normalizes per-frame to derive the per-pixel L∞ budget.
    """
    if frames_bchw.ndim != 4 or frames_bchw.shape[1] != 3:
        raise ValueError(
            f"frames_bchw must be (B, 3, H, W); got {tuple(frames_bchw.shape)}"
        )
    # Lazy import — fridrich.py pulls torch.compile etc. and we want
    # `import tac.uniward_delta` to stay fast for CLI introspection.
    from tac.fridrich import _uniward_cost
    return _uniward_cost(frames_bchw, sigma=sigma)


# ── 2. Pack / unpack ────────────────────────────────────────────────────


def _flatten_pixel_channel_index(
    frame_idx: int, y: int | torch.Tensor, x: int | torch.Tensor,
    c: int | torch.Tensor, H: int, W: int,
) -> int | torch.Tensor:
    """Produce the GLOBAL (frame, y, x, c) flat index used in the wire
    format. Inverse of the unpack path's local-index extraction.

    Layout is contiguous over (c) innermost, then (x), then (y), then frame.
    Equivalent to ``np.ravel_multi_index((frame_idx, y, x, c), (n_frames, H, W, 3))``.
    """
    return frame_idx * (H * W * 3) + y * (W * 3) + x * 3 + c


def pack_sparse_delta(
    delta_bchw: torch.Tensor,
    cost_map_bhw: torch.Tensor,
    *,
    l_inf_budget: float,
    target_bytes: int | None = None,
    seed: int = 1234,
    extra_provenance: dict[str, Any] | None = None,
    compliance_status: str = COMPLIANCE_PENDING,
    _internal_promotion_token: str | None = None,
) -> bytes:
    """Pack a dense δ tensor into the sparse UWD1 wire format.

    The function keeps the top-K pixel-channels (ranked by |δ| weighted
    inversely by the LOCAL cost-map intensity — i.e. textured pixels are
    BOOSTED in the ranking because Fridrich said they're invisible to
    the scorer), clips each kept value to the per-pixel L∞ budget,
    quantizes int8 with a global scale = ``l_inf_budget`` (so resolution
    is l_inf_budget/127 per step), and zlib-compresses.

    Top-K is chosen so that the final compressed bytes are <= target_bytes.
    Because compression is non-trivial to model analytically we use a
    binary search over n_kept (deterministic, integer-bounded — typically
    converges in ≤6 iterations).

    Args:
        delta_bchw: (B, 3, H, W) float δ in [0, 255] units (i.e. on the
            same scale as the rendered frame). Will be clipped and
            quantized in place semantically but the input is NOT mutated.
        cost_map_bhw: (B, H, W) UNIWARD cost on the SAME frames; used
            ONLY for ranking (textured pixels rank higher). Does NOT
            change the dequantized δ values.
        l_inf_budget: hard cap on |δ| per channel, in [0, 255] units.
            Council recommends 4.0 (4/255 ≈ 0.0157 in normalized units).
        target_bytes: if set, top-K is chosen so compressed payload
            (including header + indices + values) ≤ target_bytes.
            Council target: ≤5000 bytes (~5KB) for Lane C. If None,
            we keep the natural top-1% sparsity.
        seed: stored in header for provenance only (does NOT affect
            packing — packing is fully deterministic given inputs).
        extra_provenance: dict merged into the header before serialization.
            Use this to record renderer/masks/poses sha256 from the caller.

    Returns:
        ``bytes`` ready to write to ``delta.bin``. Includes the 4-byte
        ``UWD1`` magic. Round-trip stable: ``unpack_sparse_delta(pack_sparse_delta(x))``
        returns the SAME quantized δ values.

    Raises:
        ValueError: shape mismatch or l_inf_budget <= 0.
    """
    if delta_bchw.ndim != 4 or delta_bchw.shape[1] != 3:
        raise ValueError(
            f"delta_bchw must be (B, 3, H, W); got {tuple(delta_bchw.shape)}"
        )
    if cost_map_bhw.ndim != 3:
        raise ValueError(
            f"cost_map_bhw must be (B, H, W); got {tuple(cost_map_bhw.shape)}"
        )
    if delta_bchw.shape[0] != cost_map_bhw.shape[0]:
        raise ValueError(
            f"frame count mismatch: δ has {delta_bchw.shape[0]}, "
            f"cost map has {cost_map_bhw.shape[0]}"
        )
    if delta_bchw.shape[2] != cost_map_bhw.shape[1] or delta_bchw.shape[3] != cost_map_bhw.shape[2]:
        raise ValueError(
            f"spatial mismatch: δ HxW={tuple(delta_bchw.shape[2:])}, "
            f"cost HxW={tuple(cost_map_bhw.shape[1:])}"
        )
    if l_inf_budget <= 0.0:
        raise ValueError(f"l_inf_budget must be > 0; got {l_inf_budget}")
    if compliance_status not in _VALID_COMPLIANCE_STATUSES:
        raise ValueError(
            f"compliance_status must be one of {sorted(_VALID_COMPLIANCE_STATUSES)}; "
            f"got {compliance_status!r}. Default is {COMPLIANCE_PENDING!r}; only "
            f"set {COMPLIANCE_APPROVED!r} via the dedicated promotion tool "
            f"(tools/promote_lane_c_to_approved.py)."
        )
    # Codex R5-3 #2 fix (2026-04-27): writing compliance_status='approved'
    # is no longer reachable from any caller other than the dedicated
    # promotion tool. The promotion tool is the only place where the
    # internal token is supplied. Tests that need to forge an approved
    # blob for negative-path testing also pass the token explicitly —
    # this is intentional because the test exercises the SAME codepath
    # the promotion tool uses, so the gate cannot be bypassed by a
    # library refactor that quietly drops the token check.
    if compliance_status == COMPLIANCE_APPROVED:
        from tac.lane_c_compliance import verify_internal_promotion_token
        if not verify_internal_promotion_token(_internal_promotion_token):
            raise ValueError(
                f"Codex R5-3 #2: compliance_status={COMPLIANCE_APPROVED!r} is "
                "no longer accepted from library callers. Use the dedicated "
                "promotion tool which verifies a trust-rooted attestation "
                "before re-emitting the δ.bin with approved status:\n"
                "  python tools/promote_lane_c_to_approved.py \\\n"
                "      --pending-delta-bin <pending.bin> \\\n"
                "      --attestation <.omx/state/.../<sha>.json>\n"
                "Tests that need to forge an approved blob for negative-"
                "path testing must import INTERNAL_PROMOTION_TOKEN from "
                "tac.lane_c_compliance and pass it as _internal_promotion_token."
            )

    n_frames, _, H, W = delta_bchw.shape
    device = delta_bchw.device

    # Move to CPU for deterministic packing (CUDA argsort/topk over
    # millions of values is deterministic in modern torch but the bytes
    # we ship are CPU-bound by the file write anyway).
    delta_cpu = delta_bchw.detach().to("cpu", dtype=torch.float32)
    cost_cpu = cost_map_bhw.detach().to("cpu", dtype=torch.float32)

    # 1. Per-pixel L∞ clip. Channels share the budget independently
    #    (Fridrich principle: channel-independent perturbation models
    #    photon noise + JPEG-style chroma noise both of which are
    #    detector-invisible at the scorer's stride-2 stem).
    delta_cpu = delta_cpu.clamp(min=-l_inf_budget, max=l_inf_budget)

    # 2. Compute ranking score = |δ| * (1 + normalized_cost). Textured
    #    pixels (high cost) get rank-boosted → top-K prefers them.
    #    Normalize cost per-frame (prevents one bright frame dominating).
    cost_per_frame_max = cost_cpu.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    cost_norm = cost_cpu / cost_per_frame_max  # (B, H, W) in [0, 1]
    # Broadcast cost across 3 channels; reshape to (B, 3, H, W)
    rank_score = delta_cpu.abs() * (1.0 + cost_norm.unsqueeze(1))

    flat_score = rank_score.reshape(-1)              # (B*3*H*W,)
    flat_delta = delta_cpu.reshape(-1)               # (B*3*H*W,)
    n_total = flat_score.numel()

    # The wire-format flat index is (frame, y, x, c) order; PyTorch's
    # default reshape on a (B, 3, H, W) tensor lays it out as (b, c, y, x).
    # We need to translate the linear position from (b, c, y, x) ⇒
    # (b, y, x, c). Pre-compute the permutation index ONCE.
    bcyx_to_byxc = _build_bcyx_to_byxc_index(n_frames, H, W)

    def _pack_for_n_kept(n_kept: int) -> bytes:
        """Inner: pack with exactly n_kept non-zero entries."""
        if n_kept <= 0:
            kept_local = np.zeros((0,), dtype=np.uint32)
            kept_q = np.zeros((0,), dtype=np.int8)
        else:
            # Top-K by score; ties broken by lower flat-index (torch.topk
            # returns the first occurrence on equal scores → deterministic).
            top = torch.topk(flat_score, k=min(n_kept, n_total), largest=True, sorted=True)
            kept_pos_bcyx = top.indices  # in (b, c, y, x) layout

            # Translate to wire-format (b, y, x, c) layout.
            kept_pos_byxc = bcyx_to_byxc[kept_pos_bcyx]

            # Sort by wire-format position so unpack's per-frame partition
            # via searchsorted (mirrors gradient_corrections layout) is fast.
            sort_idx = torch.argsort(kept_pos_byxc, stable=True)
            kept_pos_byxc = kept_pos_byxc[sort_idx]
            kept_pos_bcyx_sorted = kept_pos_bcyx[sort_idx]

            # Quantize δ to int8 with scale = l_inf_budget.
            scale = float(l_inf_budget)
            kept_vals = flat_delta[kept_pos_bcyx_sorted]
            quant = torch.round(kept_vals / scale * 127.0).clamp(-127, 127).to(torch.int8)

            kept_local = kept_pos_byxc.numpy().astype(np.uint32, copy=False)
            kept_q = quant.numpy().astype(np.int8, copy=False)

        header: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "n_frames": int(n_frames),
            "H": int(H),
            "W": int(W),
            "n_kept": int(kept_local.size),
            "n_total": int(n_total),
            "l_inf_budget": float(l_inf_budget),
            "quantize_bits": 8,
            "scale": float(l_inf_budget),
            "seed": int(seed),
            "layout": "frame_y_x_c",
            # Codex R5 HIGH fix: machine-readable compliance gate. Defaults
            # to PENDING for any newly-built δ; the archive builder fails
            # closed on this value unless --allow-pending-compliance is set.
            "compliance_status": str(compliance_status),
        }
        if extra_provenance:
            header["provenance"] = extra_provenance

        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        body = (
            MAGIC
            + struct.pack("<I", len(header_bytes))
            + header_bytes
            + kept_local.tobytes()
            + kept_q.tobytes()
        )
        return zlib.compress(body, level=9)

    # 3. Choose n_kept — explicit budget or natural top-1%.
    if target_bytes is None:
        # M1 fix: the natural ~1% top-K explodes the rate term. For a
        # 1200-frame 384x512 input n_total=7M → n_kept=70k → ~100KB
        # compressed, which wipes any score win Lane C could give.
        # Library callers (chiefly tests) may still want this codepath,
        # so we keep it but emit a loud warning. Production callers MUST
        # pass an explicit target_bytes. The CLI script enforces this.
        import warnings
        n_kept = max(1, n_total // 100)
        warnings.warn(
            "pack_sparse_delta called with target_bytes=None — defaulting "
            f"to n_kept={n_kept} (~1% sparsity). On full Lane C inputs this "
            "produces ~100KB blobs that wipe the score win. Pass an explicit "
            "target_bytes (council target ≤5000 B) for any production run.",
            stacklevel=2,
        )
        return _pack_for_n_kept(n_kept)

    # Binary search for the largest n_kept that fits in target_bytes.
    lo, hi = 0, n_total
    best_blob = _pack_for_n_kept(0)  # all-zero δ — always fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = _pack_for_n_kept(mid)
        if len(candidate) <= target_bytes:
            best_blob = candidate
            lo = mid
        else:
            hi = mid - 1
    return best_blob


def _build_bcyx_to_byxc_index(n_frames: int, H: int, W: int) -> torch.Tensor:
    """Return a 1-D int64 tensor of length n_frames*3*H*W mapping
    a flat index in (b, c, y, x) layout (PyTorch default) to the
    corresponding flat index in (b, y, x, c) wire-format layout.

    Pure index arithmetic — no allocations of size n_total inside the
    hot pack loop. Computed ONCE per pack_sparse_delta call.
    """
    # Use np to keep memory off-GPU (we are in CPU-deterministic land).
    b = np.arange(n_frames, dtype=np.int64)[:, None, None, None]
    c = np.arange(3, dtype=np.int64)[None, :, None, None]
    y = np.arange(H, dtype=np.int64)[None, None, :, None]
    x = np.arange(W, dtype=np.int64)[None, None, None, :]
    byxc = b * (H * W * 3) + y * (W * 3) + x * 3 + c
    # Flatten in (b, c, y, x) order — same as flat_delta.reshape(-1).
    return torch.from_numpy(byxc.reshape(-1))


def unpack_sparse_delta(
    blob: bytes,
    *,
    device: str | torch.device = "cpu",
) -> DeltaSpec:
    """Decode a UWD1 blob into a per-frame DeltaSpec ready for fast apply.

    Round-trip-stable inverse of ``pack_sparse_delta``.

    Args:
        blob: raw bytes read from ``delta.bin``.
        device: target device for the per-frame index/value tensors.

    Returns:
        DeltaSpec — see class docstring.

    Raises:
        ValueError: bad magic, version mismatch, or truncated payload.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"blob must be bytes-like, got {type(blob).__name__}")

    raw = zlib.decompress(bytes(blob))
    if len(raw) < 8 or raw[:4] != MAGIC:
        raise ValueError(
            f"bad UWD magic: expected {MAGIC!r}, got {bytes(raw[:4])!r}"
        )
    header_len = struct.unpack("<I", raw[4:8])[0]
    if 8 + header_len > len(raw):
        raise ValueError("UWD payload truncated in header")
    header = json.loads(raw[8:8 + header_len].decode("utf-8"))

    # Codex R5 HIGH backward-compat: v1 blobs (no compliance_status field)
    # are accepted and DECODE AS PENDING_RULING — fail-safe. Anything newer
    # than the current SCHEMA_VERSION is rejected so a future-format δ
    # cannot be silently mis-parsed.
    blob_schema = int(header.get("schema_version", -1))
    if blob_schema not in (1, SCHEMA_VERSION):
        raise ValueError(
            f"UWD schema mismatch: got v{header.get('schema_version')}, "
            f"this build supports v1 (legacy) and v{SCHEMA_VERSION}"
        )

    n_frames = int(header["n_frames"])
    H = int(header["H"])
    W = int(header["W"])
    n_kept = int(header["n_kept"])
    scale = float(header["scale"])
    quantize_bits = int(header["quantize_bits"])
    if quantize_bits != 8:
        raise ValueError(
            f"unsupported quantize_bits={quantize_bits}; only 8 is wire-stable"
        )

    # Default to PENDING_RULING when the field is absent (v1 blobs predate
    # the gate; treat them as untrusted just like a freshly-built δ).
    compliance_status = str(
        header.get("compliance_status", COMPLIANCE_PENDING)
    )
    if compliance_status not in _VALID_COMPLIANCE_STATUSES:
        # Unknown values are coerced to PENDING (fail-safe). Don't crash
        # the loader; the archive builder will refuse to bundle anyway.
        compliance_status = COMPLIANCE_PENDING

    off = 8 + header_len
    idx_bytes = n_kept * 4  # uint32
    val_bytes = n_kept       # int8
    if off + idx_bytes + val_bytes > len(raw):
        raise ValueError("UWD payload truncated in body")

    indices = np.frombuffer(
        raw[off:off + idx_bytes], dtype=np.uint32,
    ).astype(np.int64, copy=True)
    off += idx_bytes
    values_q = np.frombuffer(
        raw[off:off + val_bytes], dtype=np.int8,
    ).astype(np.float32, copy=True)
    dequant = values_q * (scale / 127.0)

    if n_kept == 0:
        return DeltaSpec(
            n_frames=n_frames, H=H, W=W,
            per_frame_local_idx=[None] * n_frames,
            per_frame_dequant=[None] * n_frames,
            l_inf_budget=float(header["l_inf_budget"]),
            scale=scale, n_kept=0, any_delta=False,
            compliance_status=compliance_status,
        )

    # Indices are pre-sorted by pack; verify (cheap O(n)) so corrupt
    # archives can't smuggle out-of-order entries that would silently
    # break searchsorted.
    if not np.all(indices[:-1] <= indices[1:]):
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        dequant = dequant[order]

    # Per-frame partition via searchsorted on the (frame, y, x, c) flat layout.
    pixels_per_frame = H * W * 3
    frame_starts = np.searchsorted(
        indices, np.arange(n_frames + 1) * pixels_per_frame, side="left",
    )

    per_frame_local_idx: list[torch.Tensor | None] = [None] * n_frames
    per_frame_dequant: list[torch.Tensor | None] = [None] * n_frames
    for f in range(n_frames):
        s, e = int(frame_starts[f]), int(frame_starts[f + 1])
        if s == e:
            continue
        local = (indices[s:e] - f * pixels_per_frame).astype(np.int64)
        per_frame_local_idx[f] = torch.from_numpy(local).to(device, non_blocking=True)
        per_frame_dequant[f] = torch.from_numpy(dequant[s:e]).to(device, non_blocking=True)

    return DeltaSpec(
        n_frames=n_frames, H=H, W=W,
        per_frame_local_idx=per_frame_local_idx,
        per_frame_dequant=per_frame_dequant,
        l_inf_budget=float(header["l_inf_budget"]),
        scale=scale, n_kept=n_kept, any_delta=True,
        compliance_status=compliance_status,
    )


# ── 3. Apply (inflate-time, NO scorer) ──────────────────────────────────


def apply_delta_to_frame(
    frame_hwc: torch.Tensor,
    spec: DeltaSpec,
    frame_index: int,
) -> torch.Tensor:
    """Add the sparse δ to a single rendered frame.

    Pure additive lookup — no scorer, no learning, no FP magic. Returns
    the frame with δ added and clamped to [0, 255]. ``frame_hwc`` is
    cloned (does not mutate the renderer output buffer).

    Args:
        frame_hwc: (H, W, 3) float tensor on the same device as the spec
            tensors. Same units as the renderer output ([0, 255]).
        spec: parsed DeltaSpec from ``unpack_sparse_delta``.
        frame_index: which frame of the (n_frames,) δ stack to apply.

    Returns:
        (H, W, 3) tensor with δ added, clamped to [0, 255]. Same dtype
        as input (typically float32).

    Raises:
        IndexError: frame_index out of range.
        ValueError: frame size doesn't match the spec.
    """
    if not (0 <= frame_index < spec.n_frames):
        raise IndexError(
            f"frame_index={frame_index} out of range [0, {spec.n_frames})"
        )
    if frame_hwc.shape != (spec.H, spec.W, 3):
        raise ValueError(
            f"frame shape {tuple(frame_hwc.shape)} != spec ({spec.H}, {spec.W}, 3). "
            f"δ was packed at the renderer's native resolution; apply BEFORE the "
            f"camera-resolution upscale."
        )
    local_idx = spec.per_frame_local_idx[frame_index]
    if local_idx is None:
        return frame_hwc  # no δ this frame — pass through (zero-cost)

    dequant = spec.per_frame_dequant[frame_index]
    flat = frame_hwc.reshape(-1).clone().float()
    # local_idx is the FLAT (y * W * 3 + x * 3 + c) position; matches the
    # natural memory layout of an HWC frame view.
    flat.scatter_add_(0, local_idx, dequant.to(flat.dtype))
    # CRITICAL (C3 fix): only clamp at the modified positions. The previous
    # implementation called ``flat.clamp_(0, 255)`` over the WHOLE frame,
    # which mutates non-perturbed pixels whenever the renderer ever produces
    # a value slightly outside [0, 255] (e.g. mild overshoot from bilinear
    # interpolation on the upstream side). That created a sharp δ-on vs
    # δ-off behaviour difference even on frames with n_kept=0 entries —
    # equivalent to a hidden global clip whose presence depended on the
    # δ codepath being installed. The downstream uint8 conversion clamps
    # the full frame anyway, so the apply path only needs to clamp the
    # specific positions it actually touched.
    flat[local_idx] = flat[local_idx].clamp(0.0, 255.0)
    return flat.reshape(spec.H, spec.W, 3).to(frame_hwc.dtype)


# ── 4. Planning-only Fridrich detector-cost atom ranking ────────────────


def build_detector_cost_manifest(
    atoms: Iterable[Mapping[str, Any]],
    *,
    source_label: str = "manual",
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Rank charged atom candidates by detector capacity and score evidence.

    This is optimizer feedback for stack search, not a score path. The ranking
    uses only fields present on the input atoms:

    ``priority = positive_scorer_sensitivity * detector_capacity / charged_bytes``

    where ``detector_capacity`` is either an explicit UNIWARD/Fridrich capacity
    field, ``1 / (1 + detector_cost)``, or an HNeRV section entropy proxy
    ``entropy_bits_per_byte / 8``. Missing evidence keeps the row in the
    manifest, but assigns priority zero and adds a risk reason. The manifest is
    always non-dispatchable until a charged archive consumes the atom bytes and
    exact CUDA auth eval measures the resulting archive.
    """
    rows = [_detector_cost_row(atom) for atom in atoms]
    rows.sort(
        key=lambda row: (
            -float(row["allocation_priority"]),
            int(row["charged_bytes"]),
            str(row["atom_id"]),
        )
    )
    if max_atoms is not None:
        if max_atoms < 0:
            raise ValueError("max_atoms must be nonnegative")
        rows = rows[:max_atoms]
    role_counts = Counter(str(row["stream_role"]) for row in rows)
    blockers = [
        "fridrich_detector_cost_is_optimizer_feedback_only",
        "requires_charged_archive_consumption",
        "requires_archive_manifest_preflight",
        "requires_exact_cuda_auth_eval_on_candidate",
    ]
    return {
        "schema_version": DETECTOR_COST_SCHEMA_VERSION,
        "tool": "tac.uniward_delta.build_detector_cost_manifest",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": blockers,
        "source_label": str(source_label),
        "ranking_formula": (
            "positive_scorer_sensitivity * detector_capacity / charged_bytes"
        ),
        "atom_count": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "rows": rows,
    }


def _detector_cost_row(atom: Mapping[str, Any]) -> dict[str, Any]:
    atom_id = str(atom.get("atom_id") or atom.get("id") or "")
    if not atom_id:
        raise ValueError("detector-cost atom missing atom_id")
    charged_bytes = _positive_int(atom.get("charged_bytes") or atom.get("bytes"))
    stream_role = str(
        atom.get("stream_role")
        or atom.get("optimization_role")
        or atom.get("atom_kind")
        or "unknown"
    )
    detector_capacity, detector_source, detector_risks = _detector_capacity(atom)
    sensitivity, sensitivity_source, sensitivity_risks = _scorer_sensitivity(atom)
    risk_reasons = [*detector_risks, *sensitivity_risks]
    if charged_bytes <= 0:
        risk_reasons.append("charged_bytes_must_be_positive")
    evidence_grade = str(atom.get("evidence_grade") or atom.get("evidence") or "planning")
    if evidence_grade not in {"A++", "A", "B", "empirical", "derivation", "planning"}:  # CUSTODY_VALIDATOR_OK: fail-closed atom risk annotation, not score promotion
        risk_reasons.append(f"non_promotable_evidence_grade:{evidence_grade}")
    priority = 0.0
    if charged_bytes > 0 and detector_capacity > 0.0 and sensitivity > 0.0:
        priority = sensitivity * detector_capacity / charged_bytes
    return {
        "atom_id": atom_id,
        "atom_kind": str(atom.get("atom_kind") or stream_role),
        "stream_role": stream_role,
        "member": str(atom.get("member") or ""),
        "charged_bytes": charged_bytes,
        "detector_capacity": round(detector_capacity, 12),
        "detector_capacity_source": detector_source,
        "positive_scorer_sensitivity": round(sensitivity, 12),
        "scorer_sensitivity_source": sensitivity_source,
        "allocation_priority": round(priority, 15),
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "risk_reasons": sorted(set(risk_reasons)),
    }


def _detector_capacity(atom: Mapping[str, Any]) -> tuple[float, str, list[str]]:
    for key in (
        "detector_capacity",
        "uniward_capacity",
        "fridrich_capacity",
        "texture_capacity",
    ):
        value = _finite_float(atom.get(key))
        if value is not None:
            return max(0.0, value), key, []
    detector_cost = _finite_float(atom.get("detector_cost"))
    if detector_cost is not None:
        return 1.0 / (1.0 + max(0.0, detector_cost)), "1/(1+detector_cost)", []
    entropy_bits_per_byte = _finite_float(atom.get("entropy_bits_per_byte"))
    if entropy_bits_per_byte is not None:
        return max(0.0, min(1.0, entropy_bits_per_byte / 8.0)), "entropy_bits_per_byte/8", []
    return 0.0, "missing", ["missing_detector_capacity"]


def _scorer_sensitivity(atom: Mapping[str, Any]) -> tuple[float, str, list[str]]:
    for key in (
        "positive_scorer_sensitivity",
        "score_sensitivity",
        "component_sensitivity",
        "expected_score_saved",
        "rate_score_gain_if_save_1pct",
    ):
        value = _finite_float(atom.get(key))
        if value is not None:
            if value > 0.0:
                return value, key, []
            return 0.0, key, ["non_positive_scorer_sensitivity"]
    return 0.0, "missing", ["missing_positive_scorer_sensitivity"]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _positive_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ── 5. Convenience: hash the inputs that determine δ output ─────────────


def fingerprint_inputs(*paths: Path) -> str:
    """SHA256 over a list of input files. Used in pack_sparse_delta's
    extra_provenance so a future re-run can detect input drift.

    Returns the hex digest of the concatenation:
        sha256(path1) || sha256(path2) || ...
    """
    h = hashlib.sha256()
    for p in paths:
        sub = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                sub.update(chunk)
        h.update(sub.digest())
    return h.hexdigest()
