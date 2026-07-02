# SPDX-License-Identifier: MIT
"""Track-A DISTORTION finishing-kit -- inflate-side, zero/near-zero-byte bolt-ons.

This is the polished, wired, tested realization of the Track-A distortion
finishing-kit (the inflate-side frame postprocessing half of the bolt-on
stacking plan, ``bolton_inventory_and_stacking_plan_20260612.md``). It is applied
to the CONVERGED distortion-arm decoder's rendered frames; it is DISJOINT from the
weight/latent codec the rate-lever surface edits (per the scope split):

  * **PR98** (L28) -- decode-side per-(frame-parity, channel) constant bias that
    cancels the substrate's systematic color-space bias. ``0 archive bytes`` (the
    constants ride in a ~tens-of-bytes section). RE-FIT per base.
  * **T10** -- a richer per-(frame-parity, channel) AFFINE (scale + bias) fitted to
    the exact ``frame_utils.yuv420_to_rgb`` CPU GT, the 2nd-order generalization of
    PR98. Same tiny section; CPU-axis fit (CUDA needs its own).
  * **S12** -- resize-null preimage. On a RENDER-based substrate (this one stores
    decoder+latents, NOT frames) the byte savings do NOT materialize (there is no
    stored-frame section to make compressible); the certified zero-distortion
    invisibility mask is still carried as a SAFE-PERTURB region for the in-frame
    levers. So S12 here is the certification, not a byte lever (honest scope).

2026-06-13 under-power audit correction: the old mid-basin ``PR98+T10 = -0.058``
headline was a decoder-convergence artifact. On the converged ep2120 decoder it
shrinks to about ``-0.003`` advisory distortion-score, and the cross-slice transfer
prefers the PR98-only residual ``[[1,0,1],[0,0,0]]`` over the same-slice T10 affine.
Therefore the shipped default fixture is residual-only PR98; T10 remains a refit
search capability, not a banked default assumption.

Authority: this module APPLIES the kit; the FIT (the constants) is produced by the
real-scorer probe ``experiments/probe_track_a_distortion_finishing_kit.py`` and is
``[contest-CPU advisory] NON-PROMOTABLE`` until a byte-closed ``upstream/evaluate.py``
row lands. The kit is **default-OFF / byte-identical when disabled** so the live
distortion arm + basin are unperturbed if they resume onto this code.

PORTABILITY (CLAUDE.md inflate ≤100 LOC + numpy-portable + pristine-vendored):
  * the postproc is PURE NUMPY (no torch, no scorer) — it runs on the flat raw
    uint8 ``(N, 874, 1164, 3)`` tensor the vendored ``inflate.py`` produces, AFTER
    it, via the substrate's OWN ``inflate.sh`` wrapper. The vendored PR95
    ``inflate.py`` is NOT edited.
  * pair k -> frames ``2k`` (frame_0) and ``2k+1`` (frame_1); the kit indexes by
    frame PARITY so it matches the scorer's (frame_0 SegNet-blind, frame_1 seg+pose).
  * the section is a fixed-layout little-endian blob (magic + version + flags +
    6 affine (scale,bias) pairs) so the numpy inflate decodes it with struct only.

Hooks per Catalog #125 6-hook wire-in declaration:
  * #1 sensitivity-map = ACTIVE (the per-(frame,channel) fit IS the measured
    color-bias sensitivity; S12 mask is the per-pixel scorer-invisibility prior).
  * #2 Pareto constraint = ACTIVE (PR98/T10 move the distortion vertex at 0 rate;
    S12 is a rate lever ONLY on frame-carrying sections -- N/A on this render base).
  * #3 bit-allocator = N/A (zero/near-zero archive bytes; no bit allocation).
  * #4 cathedral autopilot = N/A (an inflate-side postproc applied at export, not a
    dispatch surface; the converged-checkpoint finishing pass is the consumer).
  * #5 continual-learning = ACTIVE (the re-fit constants + measured gain reseed the
    judge on whether the substrate's color-bias is a live distortion axis).
  * #6 probe-disambiguator = ACTIVE (the real-scorer probe IS the disambiguator
    between PR101's canonical constants and the base_ch=20-specific re-fit).

Cross-references:
  * ``tac.codec.pr98_channel_balance_zero_byte_bolt_on`` (the canonical L28 anchor
    this generalizes — same mechanism, re-fit constants + affine extension).
  * ``tac.optimization.resize_null_preimage`` (S12 the certified invisibility mask).
  * ``experiments/probe_track_a_distortion_finishing_kit.py`` (the fit + measure).
  * ``bolton_inventory_and_stacking_plan_20260612.md`` (the inventory; rows PR98/T10/S12/LeverD).
  * ``witness_seg_boundary_decisive_probe_20260612.md`` (the LeverD NO-GO flip-count crux).
  * ``.omx/research/finishing_kit_convergence_revalidation_RESULT.json`` (the
    converged-decoder audit that demotes mid-basin PR98/T10 assumptions).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# Section magic + version (decoded by the numpy inflate with struct only).
DISTORTION_SECTION_MAGIC = b"TADK"  # Track-A Distortion Kit
DISTORTION_SECTION_VERSION = 1
# 6 (frame-parity, channel) slots: frame parity 0 (frame_0) / 1 (frame_1) x R/G/B.
_N_SLOTS = 6
CAMERA_H, CAMERA_W = 874, 1164

CONVERGED_RESIDUAL_PR98_BIAS: tuple[
    tuple[float, float, float], tuple[float, float, float]
] = ((1.0, 0.0, 1.0), (0.0, 0.0, 0.0))
CONVERGED_RESIDUAL_PR98_PROVENANCE = (
    "[contest-CPU advisory] NON-PROMOTABLE; 2026-06-13 converged ep2120 "
    "under-power audit: mid-basin PR98/T10 shrank, cross-slice transfer kept "
    "only PR98 frame0 R/B residual."
)


@dataclass(frozen=True)
class DistortionKitConfig:
    """The fit constants for the inflate-side distortion finishing-kit.

    The transform applied to each camera-res float frame BEFORE the eval round()/
    uint8 cast is, per (frame-parity ``fr`` in {0,1}, channel ``ch`` in {0,1,2})::

        x_out = scale[fr, ch] * x_in - bias[fr, ch]

    PR98 is the special case ``scale == 1`` (pure bias). T10 fits ``scale != 1``.
    The DEFAULT is disabled identity (scale=1, bias=0). A disabled kit serializes
    no section bytes and applies no transform, so a converged checkpoint exported
    WITHOUT a fit is unchanged.

    ``enabled``: master switch. When False the kit serializes NO section and the
    postproc is a pure no-op (the default-OFF daemon-safety contract).
    ``s12_invisibility_certified``: informational flag — S12's certified zero-
    distortion mask is available for the in-frame levers; it carries NO bytes on a
    render-based substrate.
    """

    enabled: bool = False
    # (2, 3) per (frame-parity, channel); identity default.
    scale: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )
    bias: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    s12_invisibility_certified: bool = False
    provenance: str = "[contest-CPU advisory] NON-PROMOTABLE"

    def __post_init__(self) -> None:
        for name, mat in (("scale", self.scale), ("bias", self.bias)):
            arr = np.asarray(mat, dtype=np.float64)
            if arr.shape != (2, 3):
                raise ValueError(f"{name} must be shape (2, 3); got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")

    @property
    def is_identity(self) -> bool:
        """True iff the transform is the byte-identical no-op (scale=1, bias=0)."""
        return bool(
            np.allclose(np.asarray(self.scale), 1.0)
            and np.allclose(np.asarray(self.bias), 0.0)
        )

    def scale_array(self) -> np.ndarray:
        return np.asarray(self.scale, dtype=np.float64)

    def bias_array(self) -> np.ndarray:
        return np.asarray(self.bias, dtype=np.float64)

    @classmethod
    def from_pr98_bias(
        cls,
        bias_frame_channel: np.ndarray,
        *,
        s12_certified: bool = False,
        provenance: str = "[contest-CPU advisory] NON-PROMOTABLE",
    ) -> DistortionKitConfig:
        """Build a PR98-only kit (scale=1) from a fitted (2,3) bias matrix."""
        b = np.asarray(bias_frame_channel, dtype=np.float64)
        if b.shape != (2, 3):
            raise ValueError(f"bias must be (2,3); got {b.shape}")
        return cls(
            enabled=True,
            scale=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            bias=tuple(tuple(float(v) for v in row) for row in b),
            s12_invisibility_certified=s12_certified,
            provenance=provenance,
        )

    @classmethod
    def from_affine(
        cls,
        scale_frame_channel: np.ndarray,
        bias_frame_channel: np.ndarray,
        *,
        s12_certified: bool = False,
        provenance: str = "[contest-CPU advisory] NON-PROMOTABLE",
    ) -> DistortionKitConfig:
        """Build a T10 affine kit from fitted (2,3) scale + bias matrices."""
        s = np.asarray(scale_frame_channel, dtype=np.float64)
        b = np.asarray(bias_frame_channel, dtype=np.float64)
        if s.shape != (2, 3) or b.shape != (2, 3):
            raise ValueError("scale/bias must be (2,3)")
        return cls(
            enabled=True,
            scale=tuple(tuple(float(v) for v in row) for row in s),
            bias=tuple(tuple(float(v) for v in row) for row in b),
            s12_invisibility_certified=s12_certified,
            provenance=provenance,
        )

    @classmethod
    def from_converged_residual_pr98(
        cls, *, s12_certified: bool = False
    ) -> DistortionKitConfig:
        """Build the only residual kit that survived the converged under-power audit.

        This is intentionally not the old mid-basin PR98+T10/full-affine kit. The
        2026-06-13 converged ep2120 revalidation retained only a small PR98-only
        residual on frame_0 red/blue; full T10 affine stays available through
        ``from_affine`` for future final-decoder refits.
        """
        return cls.from_pr98_bias(
            np.asarray(CONVERGED_RESIDUAL_PR98_BIAS, dtype=np.float64),
            s12_certified=s12_certified,
            provenance=CONVERGED_RESIDUAL_PR98_PROVENANCE,
        )


# ---------------------------------------------------------------------------
# Section (de)serialization — fixed little-endian layout (struct-only decode).
#   MAGIC(4) | u8 version | u8 flags | 6 x (f32 scale, f32 bias)  => 6 + 48 = 54 B
# flags bit0 = s12_invisibility_certified.
# ---------------------------------------------------------------------------
_SECTION_STRUCT = struct.Struct("<4sBB" + "ff" * _N_SLOTS)


def serialize_distortion_section(config: DistortionKitConfig) -> bytes:
    """Serialize the fit constants into the fixed-layout section blob.

    Returns ``b""`` when the kit is disabled (the default-OFF contract: a disabled
    kit adds ZERO bytes to the archive). When enabled it is a fixed 54-byte blob.
    """
    if not config.enabled:
        return b""
    s = config.scale_array().reshape(-1)
    b = config.bias_array().reshape(-1)
    flags = 0x01 if config.s12_invisibility_certified else 0x00
    vals: list[float] = []
    for i in range(_N_SLOTS):
        vals.append(float(s[i]))
        vals.append(float(b[i]))
    return _SECTION_STRUCT.pack(
        DISTORTION_SECTION_MAGIC, DISTORTION_SECTION_VERSION, flags, *vals
    )


def parse_distortion_section(blob: bytes) -> DistortionKitConfig:
    """Parse a section blob back into a config.

    An EMPTY blob -> a DISABLED identity config (the default-OFF round-trip). A
    malformed/wrong-magic blob raises (fail-closed; no silent wrong-transform)."""
    if not blob:
        return DistortionKitConfig(enabled=False)
    if len(blob) != _SECTION_STRUCT.size:
        raise ValueError(
            f"distortion section must be {_SECTION_STRUCT.size} bytes; got {len(blob)}"
        )
    unpacked = _SECTION_STRUCT.unpack(blob)
    magic, version, flags = unpacked[0], unpacked[1], unpacked[2]
    if magic != DISTORTION_SECTION_MAGIC:
        raise ValueError(f"bad distortion section magic {magic!r}")
    if version != DISTORTION_SECTION_VERSION:
        raise ValueError(f"unsupported distortion section version {version}")
    rest = unpacked[3:]
    scale = np.empty((2, 3), dtype=np.float64)
    bias = np.empty((2, 3), dtype=np.float64)
    for i in range(_N_SLOTS):
        fr, ch = divmod(i, 3)
        scale[fr, ch] = rest[2 * i]
        bias[fr, ch] = rest[2 * i + 1]
    return DistortionKitConfig.from_affine(
        scale, bias, s12_certified=bool(flags & 0x01)
    )


# ---------------------------------------------------------------------------
# The inflate-side numpy postprocessor (pure numpy — runs AFTER the vendored
# inflate produces the flat raw uint8 frames).
# ---------------------------------------------------------------------------
def apply_distortion_kit_to_raw_frames(
    raw_frames: np.ndarray, config: DistortionKitConfig
) -> np.ndarray:
    """Apply the affine distortion kit to the flat raw frames in place-equivalent.

    ``raw_frames``: ``(N, H, W, 3)`` uint8 — the vendored inflate output. Pair k ->
    frames ``2k`` (frame_0, parity 0) and ``2k+1`` (frame_1, parity 1). The kit
    applies ``round(clip(scale*x - bias, 0, 255))`` per (frame-parity, channel).

    DEFAULT-OFF / NO-OP: a disabled OR identity config returns the input
    UNCHANGED (byte-identical) — the daemon-safety contract.

    Returns a uint8 array (a copy when the transform is non-identity; the SAME
    array object when it is a no-op, so callers can rely on `is` for the no-op)."""
    if (not config.enabled) or config.is_identity:
        return raw_frames
    x = np.asarray(raw_frames)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"raw_frames must be (N, H, W, 3); got {x.shape}")
    scale = config.scale_array()
    bias = config.bias_array()
    out = x.astype(np.float32)
    n = x.shape[0]
    parity = np.arange(n) % 2  # 0=frame_0, 1=frame_1
    for fr in (0, 1):
        idx = np.where(parity == fr)[0]
        if idx.size == 0:
            continue
        for ch in range(3):
            s = float(scale[fr, ch])
            b = float(bias[fr, ch])
            if s == 1.0 and b == 0.0:
                continue
            out[idx, :, :, ch] = out[idx, :, :, ch] * s - b
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Torch-side application (for the export-time eval / measurement parity) — applies
# the kit to camera FLOAT frames (B, 2, H, W, 3) before the clamp/round/uint8 cast,
# 1:1 with the scorer's input contract. Used by the export faithful-eval path.
# ---------------------------------------------------------------------------
def apply_distortion_kit_to_camera_float(camera_float, config: DistortionKitConfig):
    """Apply the kit to camera FLOAT pairs ``(B, 2, H, W, 3)`` (torch tensor).

    Operates on the continuous signal BEFORE round() — the same point the scorer's
    cast snaps. Returns the input UNCHANGED for a disabled/identity config."""
    if (not config.enabled) or config.is_identity:
        return camera_float
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch required for camera-float kit application") from exc
    if not torch.is_tensor(camera_float) or camera_float.ndim != 5 or camera_float.shape[-1] != 3:
        raise ValueError("camera_float must be a (B, 2, H, W, 3) torch tensor")
    scale = config.scale_array()
    bias = config.bias_array()
    out = camera_float.clone()
    for fr in (0, 1):
        for ch in range(3):
            s = float(scale[fr, ch])
            b = float(bias[fr, ch])
            if s == 1.0 and b == 0.0:
                continue
            out[:, fr, :, :, ch] = out[:, fr, :, :, ch] * s - b
    return out


@dataclass
class LeverDVerdict:
    """The measured GO/NO-GO verdict for LeverD (margin-conditional seg-repair).

    Carried as a NON-byte-bearing record (the verdict gates whether a seg-repair
    SIDECAR is admitted; on a basin/converged base the witness-probe flip-count
    crux makes it NO-GO — the d_seg win is banked IN TRAINING via the
    margin-weighted seg loss at zero added bytes)."""

    go: bool
    conditional_bytes_per_flip: float
    waterline_bytes_per_flip: float
    flips_per_pair: float
    scaled_residual_bytes_600pairs: float
    net_delta_s_full_sidecar_vs_frontier: float
    rationale: str = ""

    @property
    def verdict(self) -> str:
        return "GO" if self.go else "NO-GO"


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CONVERGED_RESIDUAL_PR98_BIAS",
    "CONVERGED_RESIDUAL_PR98_PROVENANCE",
    "DISTORTION_SECTION_MAGIC",
    "DISTORTION_SECTION_VERSION",
    "DistortionKitConfig",
    "LeverDVerdict",
    "apply_distortion_kit_to_camera_float",
    "apply_distortion_kit_to_raw_frames",
    "parse_distortion_section",
    "serialize_distortion_section",
]
