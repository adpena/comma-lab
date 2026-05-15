# SPDX-License-Identifier: MIT
"""D1 L2 INTEGRATION overlay — per-pixel polytope-aware noise application.

The L1 SCAFFOLD landed the encoder, archive grammar, and runtime custody
check. L2 INTEGRATION wires the inflate-time noise overlay so the archive
bytes ACTUALLY change the rendered frames (not just sit as dead bytes paying
a rate-term penalty).

Per the deep-math memo §3.6, the SegNet logit margin map ``m(x, y)``
defines a per-pixel safe-perturbation polytope. The polytope encoder
allocated noise levels via reverse-water-fill (closed-form, deterministic);
at inflate time we re-derive the same allocation from the dequantized
margin map and apply the recovered noise levels to frame_1 RGB pixels at
camera resolution.

This module is loaded ONLY by D1's ``inflate.py`` consumer. It contains NO
score claims, NO scorer load (CLAUDE.md strict-scorer-rule), NO /tmp paths,
and NO subagent state. The overlay is byte-deterministic and applies
``delta_rgb \\in {-2, -1, 0, +1, +2}`` per pixel based on the noise lattice
from the encoder.

Key invariants (preserved by construction):

* Polytope-interior invariant: pixels with margin = 0 (decision boundary)
  receive zero noise level by the encoder; the overlay applies a no-op.
* Argmax stability: since ``noise_levels`` came from the encoder's
  per-pixel safe budget, applying them as-is preserves SegNet argmax
  by construction (assuming the encoder's Jacobian bound ``L`` was valid).
* Byte determinism: the upsample mode, clamp, and dtype-cast are fixed so
  two runs with the same margin map produce byte-identical .raw output.

NO score claim — empirical verification still requires
``[contest-CUDA] AND [contest-CPU]`` paired auth eval on 1:1 contest-CI
hardware per CLAUDE.md "Submission auth eval" non-negotiable.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Contest camera resolution per ``submissions/a1/inflate.py:34``.
# Matches ``upstream/constants.py``; recorded here so the overlay does not
# import from upstream (keeps the inflate runtime dependency closure
# minimal per HNeRV parity L9).
_CAMERA_H: int = 874
_CAMERA_W: int = 1164

# Each frame is RGB uint8 at camera resolution.
_FRAME_BYTES: int = _CAMERA_H * _CAMERA_W * 3

# Contest expects 600 pairs (= 1200 frames) per video.
_N_PAIRS_PER_VIDEO: int = 600
D1_OVERLAY_CHANNEL_POLICIES: tuple[str, ...] = (
    "rgb",
    "neg_rgb",
    "red",
    "green",
    "blue",
    "neg_green",
    "rb_pos_g_neg",
)

D1_OVERLAY_SIGN_POLICIES: tuple[str, ...] = (
    "payload",
    "negate_payload",
    "alternating_pairs",
    "pair_mask",
)

D1_OVERLAY_AMPLITUDE_SCALES: tuple[float, ...] = (0.0, 0.5, 1.0)


def channel_policy_weights(channel_policy: str) -> np.ndarray:
    """Return integer RGB weights for a D1 scalar overlay policy."""
    policy = channel_policy.strip().lower()
    weights_by_policy = {
        "rgb": (1, 1, 1),
        "neg_rgb": (-1, -1, -1),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "neg_green": (0, -1, 0),
        "rb_pos_g_neg": (1, -1, 1),
    }
    if policy not in weights_by_policy:
        raise ValueError(
            f"unsupported D1 overlay_channel_policy={channel_policy!r}; "
            f"expected one of {sorted(weights_by_policy)}"
        )
    return np.asarray(weights_by_policy[policy], dtype=np.int16)


def normalize_overlay_amplitude_scale(amplitude_scale: float) -> float:
    """Validate D1 overlay attenuation.

    Values are constrained to ``[0, 1]`` so metadata-only policy variants can
    attenuate the decoded safe-polytope lattice but cannot amplify it beyond
    the encoder-certified ``[-2, 2]`` perturbation budget.
    """
    scale = float(amplitude_scale)
    if not np.isfinite(scale):
        raise ValueError(
            f"overlay_amplitude_scale must be finite; got {amplitude_scale!r}"
        )
    if scale < 0.0 or scale > 1.0:
        raise ValueError(
            f"overlay_amplitude_scale={scale} out of range [0, 1]"
        )
    return scale


def _pack_pair_sign_mask_bytes(signs: Sequence[int]) -> bytes:
    """Pack per-pair D1 signs into deterministic 2-bit bytes.

    Values are ``0`` = disabled, ``1`` = payload sign, ``-1`` = negated sign.
    Four signs fit in one byte. This keeps a 600-pair selector to 150 bytes
    before JSON/ZIP compression.
    """
    if len(signs) == 0:
        raise ValueError("pair sign mask must not be empty")
    out = bytearray()
    acc = 0
    nbits = 0
    code_by_sign = {0: 0, 1: 1, -1: 2}
    for raw in signs:
        sign = int(raw)
        if sign not in code_by_sign:
            raise ValueError(f"pair sign mask value must be -1, 0, or 1; got {raw!r}")
        acc |= code_by_sign[sign] << nbits
        nbits += 2
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc)
    return bytes(out)


def pack_pair_sign_mask(signs: Sequence[int]) -> str:
    """Pack per-pair D1 signs into canonical base85 2-bit bytes.

    Pair selectors are charged archive bytes, so D1 uses the shortest safe
    stdlib ASCII armoring for raw 2-bit selector bytes. Python's b85 alphabet
    avoids JSON quotes/backslashes for these payloads and is 12 bytes shorter
    than base64 for the 600-pair contest selector.
    """
    return base64.b85encode(_pack_pair_sign_mask_bytes(signs)).decode("ascii")


def unpack_pair_sign_mask(mask_b85: str, *, n_pairs: int) -> tuple[int, ...]:
    """Unpack a canonical base85 D1 2-bit pair-sign mask from metadata."""
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be > 0; got {n_pairs}")
    try:
        payload = base64.b85decode(str(mask_b85).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("pair sign mask is not valid base85") from exc
    expected_len = (int(n_pairs) * 2 + 7) // 8
    if len(payload) != expected_len:
        raise ValueError(
            f"pair sign mask byte length {len(payload)} != expected {expected_len}"
        )
    signs: list[int] = []
    sign_by_code = {0: 0, 1: 1, 2: -1}
    for byte in payload:
        for shift in (0, 2, 4, 6):
            if len(signs) >= n_pairs:
                break
            code = (byte >> shift) & 0b11
            if code not in sign_by_code:
                raise ValueError("pair sign mask contains reserved 2-bit code 3")
            signs.append(sign_by_code[code])
    return tuple(signs)


def pair_sign_mask_from_meta(
    meta: Mapping[str, object],
    *,
    default_n_pairs: int = _N_PAIRS_PER_VIDEO,
) -> tuple[int, ...] | None:
    """Decode the canonical D1 pair-mask metadata, or return None."""
    policy = str(meta.get("overlay_sign_policy", "payload")).strip().lower()
    if policy != "pair_mask":
        return None
    mask_b85 = meta.get("pair_mask_b85")
    n_pairs = int(meta.get("pair_mask_n", default_n_pairs))
    if not isinstance(mask_b85, str) or not mask_b85:
        raise ValueError("D1 pair_mask policy missing pair_mask_b85")
    return unpack_pair_sign_mask(mask_b85, n_pairs=n_pairs)


def overlay_sign_for_pair(
    sign_policy: str,
    pair_idx: int,
    pair_sign_mask: Sequence[int] | None = None,
) -> int:
    """Return the scalar sign multiplier for a pair under a D1 sign policy."""
    policy = sign_policy.strip().lower()
    if policy == "payload":
        return 1
    if policy == "negate_payload":
        return -1
    if policy == "alternating_pairs":
        return 1 if int(pair_idx) % 2 == 0 else -1
    if policy == "pair_mask":
        if pair_sign_mask is None:
            raise ValueError("overlay_sign_policy='pair_mask' requires pair_sign_mask")
        if pair_idx < 0 or pair_idx >= len(pair_sign_mask):
            raise ValueError(
                f"pair_idx {pair_idx} outside pair_sign_mask length {len(pair_sign_mask)}"
            )
        sign = int(pair_sign_mask[pair_idx])
        if sign not in (-1, 0, 1):
            raise ValueError(
                f"pair_sign_mask[{pair_idx}] must be -1, 0, or 1; got {sign}"
            )
        return sign
    raise ValueError(
        f"unsupported D1 overlay_sign_policy={sign_policy!r}; "
        f"expected one of {sorted(D1_OVERLAY_SIGN_POLICIES)}"
    )


def attenuate_overlay_levels(
    overlay_hw: np.ndarray,
    *,
    amplitude_scale: float,
) -> np.ndarray:
    """Attenuate integer overlay levels while preserving the D1 lattice."""
    if overlay_hw.dtype != np.int8:
        raise ValueError(
            f"attenuate_overlay_levels expects int8 overlay; got {overlay_hw.dtype}"
        )
    scale = normalize_overlay_amplitude_scale(amplitude_scale)
    if scale == 1.0:
        return overlay_hw.copy()
    values = overlay_hw.astype(np.int16)
    magnitude = np.floor(np.abs(values).astype(np.float32) * scale + 0.5)
    signed = np.sign(values).astype(np.int16) * magnitude.astype(np.int16)
    return np.clip(signed, -2, 2).astype(np.int8)


def validate_polytope_margin_contract(
    *,
    noise_levels_flat: np.ndarray,
    margin_map_int8: np.ndarray,
    margin_map_scale: float,
    archive_jacobian_lipschitz: float,
    payload_jacobian_lipschitz: float,
    boundary_eps: float = 1e-6,
) -> dict[str, int | float]:
    """Validate that decoded D1 overlay bytes match the margin/L contract.

    This is an inflate-time safety guard, not a score claim. The current D1
    lattice uses integer RGB deltas, while the scorer margin lives in logit
    units, so this guard enforces the conservative invariants that are
    dimensionally meaningful today: payload/header Lipschitz agreement, shape
    agreement, valid lattice range, and no nonzero noise on quantized
    boundary pixels.
    """
    if noise_levels_flat.dtype != np.int8:
        raise ValueError(
            f"D1 contract expects int8 noise levels; got {noise_levels_flat.dtype}"
        )
    if margin_map_int8.dtype != np.int8:
        raise ValueError(
            f"D1 contract expects int8 margin map; got {margin_map_int8.dtype}"
        )
    if noise_levels_flat.ndim != 1:
        raise ValueError(
            f"D1 contract expects flat noise levels; got {noise_levels_flat.shape}"
        )
    margin_flat = margin_map_int8.reshape(-1)
    if margin_flat.size != noise_levels_flat.size:
        raise ValueError(
            "D1 contract shape mismatch: "
            f"noise={noise_levels_flat.size} margin={margin_flat.size}"
        )
    if margin_map_scale <= 0:
        raise ValueError(f"D1 margin_map_scale must be > 0; got {margin_map_scale}")
    if archive_jacobian_lipschitz <= 0 or payload_jacobian_lipschitz <= 0:
        raise ValueError(
            "D1 jacobian_lipschitz values must be > 0; got "
            f"archive={archive_jacobian_lipschitz} payload={payload_jacobian_lipschitz}"
        )
    if not np.isclose(
        float(archive_jacobian_lipschitz),
        float(payload_jacobian_lipschitz),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "D1 jacobian_lipschitz mismatch: "
            f"archive={archive_jacobian_lipschitz} payload={payload_jacobian_lipschitz}"
        )
    lattice_violation_count = int(np.count_nonzero(np.abs(noise_levels_flat) > 2))
    margin_float = margin_flat.astype(np.float32) * float(margin_map_scale)
    boundary_mask = margin_float <= boundary_eps
    boundary_violation_count = int(
        np.count_nonzero((noise_levels_flat != 0) & boundary_mask)
    )
    safe_budget = margin_float / float(archive_jacobian_lipschitz)
    max_safe_abs = np.floor(safe_budget + 1e-6).astype(np.int16)
    unsafe_nonzero_count = int(
        np.count_nonzero(np.abs(noise_levels_flat.astype(np.int16)) > max_safe_abs)
    )
    if lattice_violation_count:
        raise ValueError(
            f"D1 lattice violation: {lattice_violation_count} levels outside [-2,2]"
        )
    if boundary_violation_count:
        raise ValueError(
            "D1 boundary violation: "
            f"{boundary_violation_count} nonzero noise levels on zero-margin pixels"
        )
    if unsafe_nonzero_count:
        raise ValueError(
            "D1 safe-budget violation: "
            f"{unsafe_nonzero_count} noise levels exceed floor(margin/L)"
        )
    return {
        "noise_pixels": int(noise_levels_flat.size),
        "nonzero_noise_pixels": int(np.count_nonzero(noise_levels_flat)),
        "boundary_pixels": int(np.count_nonzero(boundary_mask)),
        "boundary_violation_count": boundary_violation_count,
        "lattice_violation_count": lattice_violation_count,
        "unsafe_nonzero_count": unsafe_nonzero_count,
        "archive_jacobian_lipschitz": float(archive_jacobian_lipschitz),
        "payload_jacobian_lipschitz": float(payload_jacobian_lipschitz),
    }


def _upsample_int8_levels_to_camera(
    noise_levels_2d: np.ndarray,
    *,
    camera_h: int = _CAMERA_H,
    camera_w: int = _CAMERA_W,
) -> np.ndarray:
    """Upsample the 2D noise-level grid to camera resolution.

    Args:
        noise_levels_2d: int8 array shape (H, W) with values in
            {-2, -1, 0, +1, +2}; H, W are the encoder-side noise grid
            (typically 96x128 shrunk or 384x512 full).
        camera_h: Output height (default 874).
        camera_w: Output width (default 1164).

    Returns:
        int8 array shape (camera_h, camera_w) with the same value range.
        Uses nearest-neighbor upsample to preserve the discrete lattice
        values (bilinear/bicubic would smear lattice levels into
        fractional values that would round inconsistently).
    """
    if noise_levels_2d.dtype != np.int8:
        raise ValueError(
            f"_upsample_int8_levels_to_camera expects int8; got "
            f"{noise_levels_2d.dtype}"
        )
    if noise_levels_2d.ndim != 2:
        raise ValueError(
            "_upsample_int8_levels_to_camera expects 2D (H, W); got "
            f"shape {noise_levels_2d.shape}"
        )
    if noise_levels_2d.shape == (camera_h, camera_w):
        return noise_levels_2d.copy()
    # Convert to float for nearest-neighbor (avoids int8 unsupported in
    # F.interpolate on some torch builds); cast back to int8.
    t = torch.from_numpy(noise_levels_2d.astype(np.float32))
    up = F.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=(camera_h, camera_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return up.round().clamp(-2, 2).to(torch.int8).numpy().copy()


def _build_camera_resolution_overlay(
    *,
    noise_levels_flat: np.ndarray,
    encoder_grid_h: int,
    encoder_grid_w: int,
) -> np.ndarray:
    """Build the camera-resolution overlay delta from the encoder's flat noise levels.

    Args:
        noise_levels_flat: 1D int8 from the polytope payload (length
            == encoder_grid_h * encoder_grid_w).
        encoder_grid_h: Margin-map height (96 shrunk / 384 full).
        encoder_grid_w: Margin-map width (128 shrunk / 512 full).

    Returns:
        int8 array shape (camera_h, camera_w) — broadcast across all 3
        RGB channels at apply-time.
    """
    expected = encoder_grid_h * encoder_grid_w
    if noise_levels_flat.size != expected:
        raise ValueError(
            f"noise_levels_flat size {noise_levels_flat.size} != expected "
            f"H*W={expected} for grid {(encoder_grid_h, encoder_grid_w)}"
        )
    noise_2d = noise_levels_flat.reshape(encoder_grid_h, encoder_grid_w)
    return _upsample_int8_levels_to_camera(noise_2d)


def apply_polytope_overlay_inplace(
    raw_path: Path,
    *,
    noise_levels_flat: np.ndarray,
    encoder_grid_h: int,
    encoder_grid_w: int,
    n_pairs: int = _N_PAIRS_PER_VIDEO,
    camera_h: int = _CAMERA_H,
    camera_w: int = _CAMERA_W,
    channel_policy: str = "rgb",
    amplitude_scale: float = 1.0,
    sign_policy: str = "payload",
    pair_sign_mask: Sequence[int] | None = None,
) -> dict[str, int | float | str]:
    """Apply the polytope noise overlay to every frame_1 in the .raw video.

    Per ``upstream/modules.py:108``, SegNet operates on frame_1 only
    (``x[:, -1, ...]`` slicing code-discards frame_0). So the polytope
    overlay is applied ONLY to frame_1 pixels — frame_0 stays byte-identical
    to the base substrate's rendering. PoseNet sees BOTH frames so the
    polytope-interior invariant guarantees the pose component is unchanged
    only if the per-pixel noise stays inside the SegNet polytope (which
    the encoder enforces by construction).

    The overlay is additive: ``frame_1_modified = clamp(frame_1 + noise, 0, 255)``
    where ``noise`` is the camera-resolution upsample of the encoder's
    int8 noise levels in ``{-2, -1, 0, +1, +2}``.

    Args:
        raw_path: Path to the per-video .raw file produced by the base
            substrate's inflate. The file is mutated in-place.
        noise_levels_flat: 1D int8 from the D1 polytope payload.
        encoder_grid_h: Margin-map height at encode time.
        encoder_grid_w: Margin-map width at encode time.
        n_pairs: Expected number of (frame_0, frame_1) pairs per video
            (default 600).
        camera_h: Frame height (default 874).
        camera_w: Frame width (default 1164).

    Returns:
        Diagnostic dict ``{pairs_modified, bytes_changed, frame_bytes,
        nonzero_overlay_pixels}`` so the inflate driver can log the no-op
        detector signal (per Catalog #105 / #139).

    Raises:
        FileNotFoundError: raw_path missing.
        ValueError: .raw file size doesn't match expected pair count.
    """
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw_path missing: {raw_path}")
    overlay_hw = _build_camera_resolution_overlay(
        noise_levels_flat=noise_levels_flat,
        encoder_grid_h=encoder_grid_h,
        encoder_grid_w=encoder_grid_w,
    )
    overlay_hw = attenuate_overlay_levels(
        overlay_hw, amplitude_scale=amplitude_scale
    )
    nonzero_overlay_pixels = int(np.count_nonzero(overlay_hw))
    frame_bytes = camera_h * camera_w * 3
    expected_bytes = n_pairs * 2 * frame_bytes
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"raw_path {raw_path} size {actual_bytes} != expected "
            f"{expected_bytes} (n_pairs={n_pairs} camera={camera_h}x"
            f"{camera_w})"
        )
    weights = channel_policy_weights(channel_policy)
    overlay_sign_for_pair(sign_policy, 0, pair_sign_mask)
    if nonzero_overlay_pixels == 0:
        # No-op overlay — write nothing, but report so the no-op detector
        # surfaces the dead-bytes condition.
        return {
            "pairs_modified": 0,
            "bytes_changed": 0,
            "frame_bytes": frame_bytes,
            "nonzero_overlay_pixels": 0,
            "channel_policy": channel_policy,
            "overlay_amplitude_scale": normalize_overlay_amplitude_scale(
                amplitude_scale
            ),
            "overlay_sign_policy": sign_policy,
        }
    overlay_hwc = overlay_hw[:, :, np.newaxis].astype(np.int16) * weights
    overlay_flat_positive = overlay_hwc.reshape(-1)  # length frame_bytes
    overlay_flat_negative = -overlay_flat_positive

    pairs_modified = 0
    bytes_changed = 0
    with open(raw_path, "r+b") as fp:
        for pair_idx in range(n_pairs):
            sign = overlay_sign_for_pair(sign_policy, pair_idx, pair_sign_mask)
            if sign == 0:
                continue
            overlay_flat = (
                overlay_flat_positive if sign > 0 else overlay_flat_negative
            )
            # Frame layout: [frame_0 bytes][frame_1 bytes] per pair.
            frame_1_offset = (2 * pair_idx + 1) * frame_bytes
            fp.seek(frame_1_offset)
            frame_1_bytes = fp.read(frame_bytes)
            if len(frame_1_bytes) != frame_bytes:
                raise ValueError(
                    f"short read at pair {pair_idx} offset "
                    f"{frame_1_offset}; got {len(frame_1_bytes)} bytes"
                )
            frame_1_arr = np.frombuffer(
                frame_1_bytes, dtype=np.uint8
            ).copy()
            new_vals = np.clip(
                frame_1_arr.astype(np.int16) + overlay_flat, 0, 255
            ).astype(np.uint8)
            bytes_changed_pair = int(
                np.count_nonzero(new_vals != frame_1_arr)
            )
            if bytes_changed_pair > 0:
                fp.seek(frame_1_offset)
                fp.write(new_vals.tobytes())
                bytes_changed += bytes_changed_pair
                pairs_modified += 1
    return {
        "pairs_modified": pairs_modified,
        "bytes_changed": bytes_changed,
        "frame_bytes": frame_bytes,
        "nonzero_overlay_pixels": nonzero_overlay_pixels,
        "channel_policy": channel_policy,
        "overlay_amplitude_scale": normalize_overlay_amplitude_scale(
            amplitude_scale
        ),
        "overlay_sign_policy": sign_policy,
    }


def apply_l2_overlay_for_video_list(
    *,
    output_dir: Path,
    video_names: list[str],
    polytope_payload: bytes,
    encoder_grid_h: int,
    encoder_grid_w: int,
    margin_map_int8: np.ndarray | None = None,
    margin_map_scale: float | None = None,
    archive_jacobian_lipschitz: float | None = None,
    channel_policy: str = "rgb",
    amplitude_scale: float = 1.0,
    sign_policy: str = "payload",
    pair_sign_mask: Sequence[int] | None = None,
    n_pairs_per_video: int | None = None,
) -> dict[str, int | float | str]:
    """Apply the L2 polytope overlay across every video's .raw output.

    Walks the ``video_names`` list, locates each video's ``.raw`` file
    under ``output_dir`` (case-insensitive basename match; falls back to
    sole .raw if name not present), and applies the polytope-interior
    noise overlay via :func:`apply_polytope_overlay_inplace`.

    Args:
        output_dir: Directory containing per-video .raw files.
        video_names: List of contest video names (one per line in the
            contest file_list).
        polytope_payload: Bytes from the D1POLY1 archive's polytope_payload
            section (brotli-compressed lattice levels).
        encoder_grid_h: Margin-map height at encode time.
        encoder_grid_w: Margin-map width at encode time.
        n_pairs_per_video: Number of (frame_0, frame_1) pairs per video.
            Defaults to None which auto-derives from the .raw file size
            (allows tests + non-contest videos with different pair counts).
            Passing an explicit value is fail-closed: a size mismatch
            raises ValueError.

    Returns:
        Diagnostic dict ``{total_pairs_modified, total_bytes_changed,
        videos_processed}`` for the no-op detector.

    Raises:
        FileNotFoundError: when no .raw file can be located.
        ValueError: when ``n_pairs_per_video`` is provided and the .raw
            file size doesn't match.
    """
    from tac.substrates.d1_segnet_margin_polytope.polytope_encoder import (
        decode_polytope_payload,
    )

    polytope_result = decode_polytope_payload(polytope_payload)
    contract_diag: dict[str, int | float] = {}
    if (
        margin_map_int8 is not None
        and margin_map_scale is not None
        and archive_jacobian_lipschitz is not None
    ):
        contract_diag = validate_polytope_margin_contract(
            noise_levels_flat=polytope_result.noise_levels,
            margin_map_int8=margin_map_int8,
            margin_map_scale=float(margin_map_scale),
            archive_jacobian_lipschitz=float(archive_jacobian_lipschitz),
            payload_jacobian_lipschitz=float(polytope_result.jacobian_lipschitz),
        )
    total_pairs_modified = 0
    total_bytes_changed = 0
    videos_processed = 0
    for video_name in video_names:
        basename = (
            video_name.rsplit(".", 1)[0] if "." in video_name else video_name
        )
        raw_path = output_dir / f"{basename}.raw"
        if not raw_path.is_file():
            raw_candidates = sorted(output_dir.glob("*.raw"))
            if len(raw_candidates) == 1:
                raw_path = raw_candidates[0]
            else:
                raise FileNotFoundError(
                    f"D1 overlay cannot locate .raw for "
                    f"video={video_name!r} under {output_dir}; got "
                    f"{raw_candidates}"
                )
        # Auto-derive pair count from file size if not provided. The size
        # MUST be a multiple of 2 * frame_bytes (one pair = 2 frames).
        frame_bytes = _CAMERA_H * _CAMERA_W * 3
        actual_size = raw_path.stat().st_size
        if n_pairs_per_video is None:
            if actual_size % (2 * frame_bytes) != 0:
                raise ValueError(
                    f"D1 overlay: {raw_path} size {actual_size} is not a "
                    f"multiple of 2*frame_bytes={2 * frame_bytes}; cannot "
                    "auto-derive pair count"
                )
            pairs = actual_size // (2 * frame_bytes)
        else:
            pairs = int(n_pairs_per_video)
        diag = apply_polytope_overlay_inplace(
            raw_path,
            noise_levels_flat=polytope_result.noise_levels,
            encoder_grid_h=encoder_grid_h,
            encoder_grid_w=encoder_grid_w,
            n_pairs=pairs,
            channel_policy=channel_policy,
            amplitude_scale=amplitude_scale,
            sign_policy=sign_policy,
            pair_sign_mask=pair_sign_mask,
        )
        total_pairs_modified += diag["pairs_modified"]
        total_bytes_changed += diag["bytes_changed"]
        videos_processed += 1
    return {
        "total_pairs_modified": total_pairs_modified,
        "total_bytes_changed": total_bytes_changed,
        "videos_processed": videos_processed,
        "contract_nonzero_noise_pixels": int(
            contract_diag.get("nonzero_noise_pixels", 0)
        ),
        "contract_boundary_pixels": int(contract_diag.get("boundary_pixels", 0)),
        "channel_policy": channel_policy,
        "overlay_amplitude_scale": normalize_overlay_amplitude_scale(
            amplitude_scale
        ),
        "overlay_sign_policy": sign_policy,
    }


__all__ = [
    "D1_OVERLAY_AMPLITUDE_SCALES",
    "D1_OVERLAY_CHANNEL_POLICIES",
    "D1_OVERLAY_SIGN_POLICIES",
    "apply_l2_overlay_for_video_list",
    "apply_polytope_overlay_inplace",
    "attenuate_overlay_levels",
    "channel_policy_weights",
    "normalize_overlay_amplitude_scale",
    "overlay_sign_for_pair",
    "pack_pair_sign_mask",
    "pair_sign_mask_from_meta",
    "unpack_pair_sign_mask",
    "validate_polytope_margin_contract",
]
