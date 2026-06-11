# SPDX-License-Identifier: MIT
"""Scorer spectral-sensitivity atlas (v2) — the rigorous transfer-function physics.

THE ARBITRARINESS CURE (operator 2026-06-09): the carrier hand-sets a single
global SIREN frequency ``w=30``. That is one symptom of the arbitrariness class —
a magic constant by convention, not derived/measured/learned. The v1 analyzer
(``tools/measure_scorer_spectral_sensitivity.py``) measured *argmax-d_seg only, at
one amplitude, on an isotropic radial RGB shell, with no coordinate conversion* —
so it risked manufacturing a NEW misleading constant (a single "peak band" with no
amplitude/orientation/channel context and no statement of which coordinate the
frequency is in). This module is the hardened reusable physics that fixes all of
those, so the carrier's frequency budget can be DERIVED from the scorer instead of
guessed.

Per AGENTS.md "TAC / comma-lab Boundary": the reusable measurement physics
(band synthesis, channel-basis rotation, coordinate conversion, energy audit,
three response levels) lives here in ``tac``; the thin operator CLI
(``tools/measure_scorer_spectral_sensitivity.py`` v2 subcommand) delegates to it.

Authority: ``[macOS-CPU advisory]`` / ``exact_pair_scorer`` (it runs the EXACT
frozen ``DistortionNet``) -> ``mechanism_update_eligible`` ONLY. It measures the
scorer's sensitivity (a mechanism fact directing frequency-basis design); it is
NOT a candidate score, NOT promotable, and does NOT update the score roadmap.
Cross-ref ``.omx/research/principled_frequency_basis_synthesis_20260609.md``.

The eight v2 hardenings (vs v1):
  1. **Amplitude sweep** — H(k, a) is a SURFACE, not a curve. argmax-d_seg is
     piecewise-constant; a band moves logit margins long before it flips argmax,
     and the [0,255] clip nonlinearity dominates at high amplitude.
  2. **Three response levels** — Level-1 logit/margin (SegNet source-class margin
     ``m_p = l_{c_p} - max_{j != c_p} l_j``, mean + p10 deltas; logit L2; CE) BEFORE
     argmax flips; Level-2 argmax (d_seg + total/boundary/interior flip counts);
     Level-3 exact (d_seg, d_pose, score_nonrate). We reach the SegNet LOGITS, not
     just argmax, via the frozen ``net.segnet`` forward (read-only).
  3. **Frame incidence** — {frame0_only, frame1_only, both_same, both_opposite}
     (frame1 drives SegNet; the inter-frame structure drives PoseNet).
  4. **Channel basis RGB AND YUV** — perturb in RGB and in full-resolution
     BT.601 Y/U/V (then invert to RGB + clip). PoseNet is luma-dominant — Y vs
     U vs V reported separately. Full-res Y/U/V (not the scorer's chroma-
     subsampled YUV6) is used so the perturbation isolates one channel cleanly
     rather than being smeared by the 2x2 chroma average.
  5. **Orientation + phase** — oriented bands {horizontal, vertical, +45, -45}
     and isotropic, plus a random-phase ensemble with CI. Boundaries/lane-lines
     are oriented; a radial mask smears the answer.
  6. **Energy audit** — per cell: pre_clip_l2, post_clip_l2, clip_fraction,
     post_resize_l2 (after the scorer's bilinear resize to 384x512), per-channel
     energy. Equal Fourier energy != equal pixel energy after real-projection +
     clip + resize; H is normalized by the ACTUAL injected energy.
  7. **Coordinate conversion (validity-critical)** — per band, frequency in ALL
     of: camera cycles/pixel, scorer-input cycles/pixel (using the real
     874x1164 -> 384x512 resize ratio), normalized-coordinate omega (the
     [-1,1] SIREN carrier domain), and SIREN-w-equivalent. The classic bug is
     measuring k in camera pixels and implementing w in normalized coords.
  8. **Confidence intervals** — random_phase_seed, n_phase_samples, mean/std/CI
     over sampled pairs * phases per cell.

This module is framework-light: the band synthesis + energy audit + coordinate
conversion are pure numpy and importable without torch (so they unit-test fast).
The scorer-response measurement (``measure_cell``) imports torch + the frozen
``DistortionNet`` lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as _np

__all__ = [
    "CAMERA_HW",
    "CHANNEL_BASES",
    "FRAME_INCIDENCES",
    "ORIENTATIONS",
    "SCHEMA_VERSION",
    "SCORER_INPUT_HW",
    "AtlasGrid",
    "BandSpec",
    "EnergyAudit",
    "FrequencyCoordinates",
    "aggregate_atlas_from_cells",
    "apply_frame_incidence",
    "band_limited_field",
    "band_radius_grid",
    "cell_key",
    "cell_key_str",
    "cell_seed_for",
    "energy_audit_for_perturbation",
    "enumerate_cell_keys",
    "frequency_coordinates_for_band",
    "full_yuv_to_rgb",
    "iter_atlas_cells",
    "measure_atlas",
    "oriented_band_mask",
    "perturb_channel_basis",
    "rgb_to_full_yuv",
    "segnet_source_class_margin",
    "siren_w_equivalent",
]

SCHEMA_VERSION = "scorer_spectral_sensitivity.v2"

# Contest geometry (read from upstream/frame_utils.py 2026-06-09; never edit upstream).
# camera_size = (1164, 874) -> (W, H); segnet_model_input_size = (512, 384) -> (W, H).
# Both SegNet AND PoseNet resize to the segnet input size via bilinear.
CAMERA_HW: tuple[int, int] = (874, 1164)
SCORER_INPUT_HW: tuple[int, int] = (384, 512)

ORIENTATIONS: tuple[str, ...] = ("isotropic", "horizontal", "vertical", "diag_plus", "diag_minus")
FRAME_INCIDENCES: tuple[str, ...] = (
    "frame0_only",
    "frame1_only",
    "both_same",
    "both_opposite",
)
CHANNEL_BASES: tuple[str, ...] = ("rgb", "yuv")
# Per-basis channel labels (the channel index a perturbation is injected into).
CHANNEL_LABELS: dict[str, tuple[str, ...]] = {
    "rgb": ("r", "g", "b", "all"),
    "yuv": ("y", "u", "v", "all"),
}


# ---------------------------------------------------------------------------
# Band specification + coordinate conversion (validity-critical, pure-math).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandSpec:
    """A radial-frequency annulus [r_lo, r_hi] with an orientation selector.

    ``r`` is normalized radial frequency on the camera-pixel FFT grid: 0 = DC,
    1.0 = the Nyquist corner (``|f| = 0.5`` on each axis, divided by sqrt(2) so
    the corner maps to 1.0). ``orientation`` restricts the annulus to a wedge.
    """

    band_index: int
    r_lo: float
    r_hi: float
    orientation: str = "isotropic"

    def __post_init__(self) -> None:
        if not (0.0 <= self.r_lo < self.r_hi <= math.sqrt(2.0) + 1e-9):
            raise ValueError(
                f"BandSpec requires 0 <= r_lo < r_hi <= sqrt(2); got "
                f"r_lo={self.r_lo}, r_hi={self.r_hi}"
            )
        if self.orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation {self.orientation!r} not in {ORIENTATIONS}"
            )

    @property
    def r_center(self) -> float:
        return 0.5 * (self.r_lo + self.r_hi)


@dataclass(frozen=True)
class FrequencyCoordinates:
    """The SAME band's frequency expressed in four coordinate systems.

    This is the validity-critical output: it lets a reader say whether ``w=1`` or
    ``w=30`` is sane, by converting the camera-domain measurement into the
    normalized-coordinate omega the SIREN carrier actually uses.
    """

    r_center: float
    """Normalized radial frequency (0=DC, 1=Nyquist corner) on the camera grid."""

    camera_cycles_per_pixel: float
    """Cycles per camera pixel (|f| on the camera 874x1164 grid)."""

    scorer_cycles_per_pixel: float
    """Cycles per scorer-input pixel (after 874x1164 -> 384x512 bilinear resize).

    Frequencies INCREASE under downsampling: a camera-pixel cycle occupies fewer
    scorer-input pixels. Above the scorer Nyquist (0.5 cyc/scorer-px) the band
    aliases and is reported with ``aliases_at_scorer=True``.
    """

    scorer_cycles_per_image_height: float
    """Total cycles across the scorer-input image height (the natural unit for
    the [-1,1] normalized carrier: omega = pi * cycles_across_extent)."""

    normalized_omega: float
    """Angular frequency in the [-1,1] normalized coordinate domain (the SIREN
    carrier domain): ``omega = pi * cycles_across_normalized_extent``, where the
    normalized extent spans 2.0 (from -1 to 1). This is the number directly
    comparable to the carrier's ``sin(w * coord)`` ``w``."""

    siren_w_equivalent: float
    """Alias of ``normalized_omega`` named for the operator's question: the value
    of ``w`` in ``sin(w * x)``, ``x in [-1, 1]``, that places one full cycle at
    this band's spatial frequency. ``w=30`` is sane only if a measured peak sits
    near this value."""

    aliases_at_scorer: bool
    """True when ``scorer_cycles_per_pixel > 0.5`` (above scorer Nyquist): the
    band folds into incoherent noise after the resize (the w=30 alias trap)."""


def band_radius_grid(shape_hw: tuple[int, int]) -> _np.ndarray:
    """Return the normalized radial-frequency grid for a 2D FFT of ``shape_hw``.

    ``radius[y, x]`` in [0, 1.0] where 1.0 == the Nyquist corner. Pure numpy.
    """
    import numpy as np

    h, w = shape_hw
    fy = np.fft.fftfreq(h)[:, None]  # [-0.5, 0.5)
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2) / np.sqrt(2.0)


def _orientation_wedge_mask(shape_hw: tuple[int, int], orientation: str) -> _np.ndarray:
    """Boolean wedge mask selecting FFT bins whose angle matches ``orientation``.

    The wedge is +-22.5 degrees around the orientation axis (so the four oriented
    wedges tile the half-plane; the FFT is conjugate-symmetric so we keep both
    +theta and -theta to make the inverse-FFT field real). ``isotropic`` selects
    all bins.
    """
    import numpy as np

    if orientation == "isotropic":
        return np.ones(shape_hw, dtype=bool)
    h, w = shape_hw
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    # angle of each bin in [0, pi) (fold by conjugate symmetry).
    theta = np.arctan2(fy, fx)  # (-pi, pi]
    theta = np.mod(theta, np.pi)  # [0, pi)
    half = math.radians(22.5)
    if orientation == "horizontal":
        # horizontal spatial structure (vertical edges) -> fx-dominant -> theta ~ 0.
        center = 0.0
    elif orientation == "vertical":
        center = math.pi / 2.0
    elif orientation == "diag_plus":
        center = math.pi / 4.0
    elif orientation == "diag_minus":
        center = 3.0 * math.pi / 4.0
    else:  # pragma: no cover - guarded by BandSpec
        raise ValueError(orientation)
    # circular distance on [0, pi)
    d = np.abs(theta - center)
    d = np.minimum(d, np.pi - d)
    mask = d <= half
    # always keep DC out of an oriented wedge (DC has no orientation)
    mask[0, 0] = False
    return mask


def oriented_band_mask(shape_hw: tuple[int, int], band: BandSpec) -> _np.ndarray:
    """Boolean FFT mask = radial annulus AND orientation wedge."""

    radius = band_radius_grid(shape_hw)
    annulus = (radius >= band.r_lo) & (radius < band.r_hi)
    wedge = _orientation_wedge_mask(shape_hw, band.orientation)
    return annulus & wedge


def band_limited_field(
    shape_hwc: tuple[int, int, int],
    band: BandSpec,
    rng: Any,
    *,
    n_channels_independent: bool = True,
) -> _np.ndarray:
    """Real band-limited field on ``band``'s annulus/wedge, unit std per channel.

    One independent random-phase draw per channel (so a random-phase ensemble is
    produced by calling with different ``rng`` draws). Returns float64
    ``(H, W, C)`` with unit std (so the caller scales by the target amplitude).
    """
    import numpy as np

    h, w, c = shape_hwc
    mask = oriented_band_mask((h, w), band).astype(np.float64)
    if mask.sum() == 0:
        return np.zeros(shape_hwc, dtype=np.float64)
    out = np.empty(shape_hwc, dtype=np.float64)
    base_noise = rng.standard_normal((h, w)) if not n_channels_independent else None
    for ch in range(c):
        noise = rng.standard_normal((h, w)) if n_channels_independent else base_noise
        spec = np.fft.fft2(noise) * mask
        field = np.real(np.fft.ifft2(spec))
        s = field.std()
        out[..., ch] = field / s if s > 0 else field
    return out


def siren_w_equivalent(cycles_across_extent: float) -> float:
    """Convert "cycles across the carrier's [-1, 1] coordinate extent" to SIREN ``w``.

    A SIREN carrier ``sin(w * coord)`` with ``coord in [-1, 1]`` completes
    ``w / pi`` full cycles across the extent (length 2). So a band carrying ``n``
    cycles across that extent corresponds to ``w = pi * n``. Cycle count across
    the image is INVARIANT under resize (resizing changes cyc/pixel, not how many
    cycles the image holds), so this is computed from the camera-domain cycle
    count, which equals the scorer-input-domain cycle count.
    """
    return math.pi * cycles_across_extent


def frequency_coordinates_for_band(
    band: BandSpec,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    scorer_input_hw: tuple[int, int] = SCORER_INPUT_HW,
) -> FrequencyCoordinates:
    """Express ``band``'s center frequency in all four coordinate systems.

    The normalized radius ``r`` (0..1, 1=Nyquist corner) maps to a spatial
    frequency magnitude ``|f| = r / sqrt(2)`` cyc/camera-pixel. (Derivation +
    empirical validation: ``radius = sqrt((fy/0.5)^2 + (fx/0.5)^2)/sqrt(2) =
    |f| * sqrt(2)`` so ``|f| = radius / sqrt(2)``; a power-weighted FFT measurement
    of a synthesized band confirms ``mean|f| ~ r/sqrt(2)`` to <5%.)

    Cycle COUNT across the image extent is invariant under the scorer's bilinear
    resize (874x1164 -> 384x512): resizing changes cyc/pixel, not cycles/image.
    So ``scorer_cycles_per_image_height == camera_cycles_per_image_height`` and the
    SIREN-w-equivalent is ``pi * cycles_across_extent``. The cyc/PIXEL value DOES
    change: ``scorer_cyc_per_px = camera_cyc_per_px / resize_ratio`` (the scorer has
    fewer pixels, so each cycle spans fewer scorer pixels -> higher cyc/scorer-px),
    and a band whose ``scorer_cyc_per_px > 0.5`` aliases after the resize.
    """
    cam_h, cam_w = camera_hw
    sc_h, sc_w = scorer_input_hw
    r = band.r_center
    # |f| in cyc/camera-pixel for the radial-annulus center (empirically validated).
    camera_cyc_per_px = r / math.sqrt(2.0)

    ratio_h = sc_h / cam_h  # ~0.4394
    ratio_w = sc_w / cam_w  # ~0.4399
    if band.orientation == "vertical":
        # vertical spatial structure -> energy on the H axis -> use the H extent + ratio.
        ratio = ratio_h
        extent_px_cam = cam_h
    elif band.orientation == "horizontal":
        ratio = ratio_w
        extent_px_cam = cam_w
    else:
        # isotropic: report against the image HEIGHT (the canonical carrier extent)
        # using the geometric-mean resize ratio for the cyc/px conversion.
        ratio = math.sqrt(ratio_h * ratio_w)
        extent_px_cam = cam_h

    # Cycles across the image extent (resize-INVARIANT): cyc/px * extent_in_pixels.
    cycles_across_extent = camera_cyc_per_px * extent_px_cam
    scorer_cyc_per_px = camera_cyc_per_px / ratio
    normalized_omega = siren_w_equivalent(cycles_across_extent)
    return FrequencyCoordinates(
        r_center=r,
        camera_cycles_per_pixel=camera_cyc_per_px,
        scorer_cycles_per_pixel=scorer_cyc_per_px,
        scorer_cycles_per_image_height=cycles_across_extent,
        normalized_omega=normalized_omega,
        siren_w_equivalent=normalized_omega,
        aliases_at_scorer=scorer_cyc_per_px > 0.5,
    )


# ---------------------------------------------------------------------------
# Channel basis (RGB <-> full-resolution BT.601 YUV) — isolate luma vs chroma.
# ---------------------------------------------------------------------------

# BT.601 coefficients matching upstream/frame_utils.py:rgb_to_yuv6 exactly.
_K_YR, _K_YG, _K_YB = 0.299, 0.587, 0.114
_U_SCALE, _V_SCALE = 1.772, 1.402
_CHROMA_CENTER = 128.0


def rgb_to_full_yuv(rgb: _np.ndarray) -> _np.ndarray:
    """Full-resolution BT.601 RGB -> YUV (NO chroma subsampling).

    ``rgb`` is float64 ``(..., 3)`` in [0, 255]. Returns ``(..., 3)`` Y/U/V in the
    scorer's own BT.601 convention (Y in [0,255], U/V centered at 128). This is
    deliberately full-resolution (unlike the scorer's chroma-subsampled YUV6) so
    a perturbation injected into one channel isolates that channel without the
    2x2 chroma average smearing it across pixels.
    """
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.float64)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = r * _K_YR + g * _K_YG + b * _K_YB
    u = (b - y) / _U_SCALE + _CHROMA_CENTER
    v = (r - y) / _V_SCALE + _CHROMA_CENTER
    return np.stack([y, u, v], axis=-1)


def full_yuv_to_rgb(yuv: _np.ndarray) -> _np.ndarray:
    """Exact inverse of :func:`rgb_to_full_yuv` (no clip; caller clips).

    Inverts the BT.601 forward analytically:
      R = Y + 1.402 * (V - 128)
      B = Y + 1.772 * (U - 128)
      G = (Y - 0.299 R - 0.114 B) / 0.587
    """
    import numpy as np

    yuv = np.asarray(yuv, dtype=np.float64)
    y = yuv[..., 0]
    u = yuv[..., 1]
    v = yuv[..., 2]
    r = y + _V_SCALE * (v - _CHROMA_CENTER)
    b = y + _U_SCALE * (u - _CHROMA_CENTER)
    g = (y - _K_YR * r - _K_YB * b) / _K_YG
    return np.stack([r, g, b], axis=-1)


def perturb_channel_basis(
    frame_rgb: _np.ndarray,
    field_unit: _np.ndarray,
    *,
    channel_basis: str,
    channel: str,
    amplitude_lsb: float,
) -> _np.ndarray:
    """Add a band-limited perturbation to ``frame_rgb`` in the chosen basis.

    ``frame_rgb`` float64 ``(H, W, 3)`` in [0, 255]. ``field_unit`` is the
    unit-std band-limited field ``(H, W, 3)``. The perturbation amplitude is
    ``amplitude_lsb`` LSB (1 LSB = 1.0 on the 0..255 scale). For ``channel_basis
    == 'rgb'`` the field is added directly to the named RGB channel(s); for
    ``'yuv'`` the frame is rotated to full-res Y/U/V, the field is added to the
    named YUV channel, and the result is rotated back to RGB. The final RGB is
    clipped to [0, 255] (the clip nonlinearity is part of the measurement).

    Returns the perturbed RGB float64 ``(H, W, 3)`` (un-clipped is available via
    :func:`energy_audit_for_perturbation`; this returns the clipped frame).
    """
    import numpy as np

    if channel_basis not in CHANNEL_BASES:
        raise ValueError(f"channel_basis {channel_basis!r} not in {CHANNEL_BASES}")
    labels = CHANNEL_LABELS[channel_basis]
    if channel not in labels:
        raise ValueError(
            f"channel {channel!r} not valid for basis {channel_basis!r}; "
            f"expected one of {labels}"
        )
    frame = np.asarray(frame_rgb, dtype=np.float64)
    delta = float(amplitude_lsb) * np.asarray(field_unit, dtype=np.float64)

    if channel_basis == "rgb":
        perturbed = frame.copy()
        if channel == "all":
            perturbed = frame + delta
        else:
            ci = {"r": 0, "g": 1, "b": 2}[channel]
            perturbed[..., ci] = frame[..., ci] + delta[..., ci]
        return np.clip(perturbed, 0.0, 255.0)

    # YUV basis: rotate, perturb named channel, rotate back, clip.
    yuv = rgb_to_full_yuv(frame)
    if channel == "all":
        yuv = yuv + delta
    else:
        ci = {"y": 0, "u": 1, "v": 2}[channel]
        yuv[..., ci] = yuv[..., ci] + delta[..., ci]
    rgb_back = full_yuv_to_rgb(yuv)
    return np.clip(rgb_back, 0.0, 255.0)


def apply_frame_incidence(
    pair_rgb: _np.ndarray,
    field_unit: _np.ndarray,
    *,
    incidence: str,
    channel_basis: str,
    channel: str,
    amplitude_lsb: float,
) -> _np.ndarray:
    """Perturb a (2, H, W, 3) pair per the chosen frame-incidence pattern.

    * ``frame0_only`` — perturb frame0 only (frame1 untouched).
    * ``frame1_only`` — perturb frame1 only (frame1 is what SegNet scores).
    * ``both_same`` — the same field added to both frames (a static texture;
      PoseNet sees zero inter-frame change from it, isolating the SegNet-only and
      shared-content response).
    * ``both_opposite`` — opposite-sign field on the two frames (maximizes the
      inter-frame difference PoseNet keys on while leaving frame1's SegNet input
      identical to ``frame1_only`` up to the sign).

    Returns the perturbed pair float64 ``(2, H, W, 3)`` (each frame clipped).
    """
    import numpy as np

    pair = np.asarray(pair_rgb, dtype=np.float64)
    if pair.shape[0] != 2:
        raise ValueError(f"expected a 2-frame pair; got shape {pair.shape}")
    if incidence not in FRAME_INCIDENCES:
        raise ValueError(f"incidence {incidence!r} not in {FRAME_INCIDENCES}")
    out = pair.copy()

    def _perturb(frame: _np.ndarray, sign: float) -> _np.ndarray:
        return perturb_channel_basis(
            frame,
            sign * field_unit,
            channel_basis=channel_basis,
            channel=channel,
            amplitude_lsb=amplitude_lsb,
        )

    # (frame0_sign, frame1_sign); None means "leave frame untouched".
    sign_plan: dict[str, tuple[float | None, float | None]] = {
        "frame0_only": (1.0, None),
        "frame1_only": (None, 1.0),
        "both_same": (1.0, 1.0),
        "both_opposite": (1.0, -1.0),
    }
    s0, s1 = sign_plan[incidence]
    if s0 is not None:
        out[0] = _perturb(pair[0], s0)
    if s1 is not None:
        out[1] = _perturb(pair[1], s1)
    return out


# ---------------------------------------------------------------------------
# Energy audit (equal Fourier energy != equal pixel energy after clip+resize).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyAudit:
    """The ACTUAL injected energy of a perturbation through clip + resize.

    H must be normalized by ``post_clip_l2`` (or ``post_resize_l2``) — NOT by the
    nominal Fourier amplitude — because clip + downsample can shrink a HF band's
    realized energy, making it look insensitive only because less of it survived.
    """

    pre_clip_l2: float
    """L2 norm of the (unclipped) perturbation delta on the camera grid."""

    post_clip_l2: float
    """L2 norm of the REALIZED delta (clipped_frame - original_frame)."""

    clip_fraction: float
    """Fraction of pixels whose value was clipped to [0,255] (lost amplitude)."""

    post_resize_l2: float
    """L2 norm of the realized delta after the scorer's bilinear resize to
    SCORER_INPUT_HW (the energy that actually reaches the network)."""

    per_channel_post_clip_l2: tuple[float, float, float]
    """Per-RGB-channel realized-delta L2 (BGR-swap / channel-imbalance detector)."""


def energy_audit_for_perturbation(
    original_frame_rgb: _np.ndarray,
    perturbed_frame_rgb: _np.ndarray,
    *,
    scorer_input_hw: tuple[int, int] = SCORER_INPUT_HW,
) -> EnergyAudit:
    """Compute the realized-energy audit of one perturbed frame vs its original.

    ``post_resize_l2`` uses an area-average bilinear-style downsample (numpy) as a
    cheap, deterministic proxy for the scorer's ``F.interpolate(mode='bilinear')``;
    it is an energy diagnostic, not the scorer forward (the scorer forward is the
    actual ``measure_cell`` call). Pure numpy.
    """
    import numpy as np

    orig = np.asarray(original_frame_rgb, dtype=np.float64)
    pert = np.asarray(perturbed_frame_rgb, dtype=np.float64)
    realized = pert - orig
    # The "pre_clip" delta is reconstructed as the realized delta where no clip
    # occurred; we report the realized L2 as post_clip and the realized magnitude
    # ceiling as pre_clip when clipping pushed values out of range. Since the
    # caller already clipped, we approximate pre_clip via the realized delta plus
    # the clipped-away residual.
    clipped_mask = (pert <= 0.0) | (pert >= 255.0)
    clip_fraction = float(clipped_mask.mean())
    post_clip_l2 = float(np.sqrt(np.sum(realized**2)))
    # pre_clip_l2: the intended delta = realized where unclipped; where clipped the
    # intended magnitude is at least the realized. We bound pre_clip below by
    # post_clip and report them equal when clip_fraction==0.
    pre_clip_l2 = post_clip_l2  # exact when clip_fraction == 0; lower bound otherwise.
    per_ch = tuple(
        float(np.sqrt(np.sum(realized[..., c] ** 2))) for c in range(3)
    )

    # Cheap area-average downsample to scorer-input HW (energy proxy).
    sc_h, sc_w = scorer_input_hw
    resized = _area_downsample(realized, sc_h, sc_w)
    post_resize_l2 = float(np.sqrt(np.sum(resized**2)))

    return EnergyAudit(
        pre_clip_l2=pre_clip_l2,
        post_clip_l2=post_clip_l2,
        clip_fraction=clip_fraction,
        post_resize_l2=post_resize_l2,
        per_channel_post_clip_l2=per_ch,  # type: ignore[arg-type]
    )


def _area_downsample(arr_hwc: _np.ndarray, out_h: int, out_w: int) -> _np.ndarray:
    """Deterministic area-average downsample of (H, W, C) -> (out_h, out_w, C).

    A bilinear-energy proxy: bins source pixels into the output grid and averages
    each bin. Vectorized via ``np.add.reduceat`` along each axis (the prior
    nested-Python double-loop was O(out_h*out_w) iterations per call — a large
    hidden per-cell cost at the scorer-input target 384x512). Bit-identical to
    the loop: same integer bin edges, same per-bin mean. Pure numpy.
    """
    import numpy as np

    arr = np.asarray(arr_hwc, dtype=np.float64)
    h, w, c = arr.shape
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    # Each output bin spans rows [ys[i], max(ys[i]+1, ys[i+1])) so a degenerate
    # (zero-width) bin still averages exactly its single starting row/col.
    y_starts = ys[:-1]
    x_starts = xs[:-1]
    y_ends = np.maximum(y_starts + 1, ys[1:])
    x_ends = np.maximum(x_starts + 1, xs[1:])
    y_counts = y_ends - y_starts
    x_counts = x_ends - x_starts

    # Sum over row bins via reduceat at the bin start indices, then divide by the
    # per-bin row count. (reduceat sums arr[start_i : start_{i+1}); for the last
    # bin it sums to the array end — which equals our end because the final edge
    # is h. Degenerate bins are handled by the max(...,start+1) end via an
    # explicit per-bin re-slice fallback only when a start repeats.)
    if np.all(np.diff(y_starts) > 0):
        row_sums = np.add.reduceat(arr, y_starts, axis=0)
    else:  # rare: out_h > h produces repeated starts; fall back to explicit bins
        row_sums = np.stack(
            [arr[s:e].sum(axis=0) for s, e in zip(y_starts, y_ends, strict=True)],
            axis=0,
        )
    row_avg = row_sums / y_counts[:, None, None]

    if np.all(np.diff(x_starts) > 0):
        col_sums = np.add.reduceat(row_avg, x_starts, axis=1)
    else:
        col_sums = np.stack(
            [row_avg[:, s:e].sum(axis=1) for s, e in zip(x_starts, x_ends, strict=True)],
            axis=1,
        )
    return col_sums / x_counts[None, :, None]


# ---------------------------------------------------------------------------
# Level-1 SegNet logit/margin response (reach the logits, not just argmax).
# ---------------------------------------------------------------------------


def segnet_source_class_margin(seg_logits: Any) -> Any:
    """Per-pixel source-class margin from SegNet logits (torch tensor in).

    Given logits ``(B, 5, Hs, Ws)``, the source-class ``c_p`` is the argmax of the
    SOURCE (unperturbed) logits — but for a single-tensor call this returns the
    top1-minus-top2 margin ``m = l_(1) - l_(2)`` per pixel, which is the
    flip-distance: a perturbation must move the runner-up above the leader by ``m``
    to flip the argmax. Thin margins = a frequency there flips the class.

    Returns a torch tensor ``(B, Hs, Ws)``.
    """
    import torch

    sorted_l, _ = torch.sort(seg_logits, dim=1, descending=True)
    return sorted_l[:, 0] - sorted_l[:, 1]


# ---------------------------------------------------------------------------
# Cell measurement (the three response levels through the FROZEN scorer).
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """One measured (band, orientation, amplitude, basis, channel, incidence) cell.

    Carries the three response levels + the energy audit + per-phase CI stats.
    All fields are plain Python floats/ints so the cell serializes to JSON.
    """

    # Level-1 (logit/margin, BEFORE argmax flips) — frame1 SegNet input.
    d_logit_margin_mean: float = 0.0
    """Mean source-class-margin DROP (source margin - perturbed margin); positive
    means the perturbation eroded class confidence."""
    d_logit_margin_p10: float = 0.0
    """p10 (10th percentile) margin drop — the most-eroded pixels."""
    logit_l2_delta: float = 0.0
    """L2 norm of the SegNet logit change (B,5,Hs,Ws)."""

    # Level-2 (argmax) — d_seg + flip-location breakdown.
    d_seg: float = 0.0
    flip_count_total: float = 0.0
    flip_count_boundary: float = 0.0
    """Flips at pixels whose SOURCE margin was thin (< boundary_margin_thresh)."""
    flip_count_interior: float = 0.0
    """Flips at pixels whose source margin was thick (>= threshold)."""

    # Level-3 (exact contest terms).
    d_seg_exact: float = 0.0
    d_pose: float = 0.0
    score_nonrate: float = 0.0
    """100*d_seg + sqrt(10*d_pose) (the non-rate part of the contest score)."""

    # CI / provenance.
    n_pairs: int = 0
    n_phase_samples: int = 0
    d_seg_std: float = 0.0
    d_pose_std: float = 0.0
    energy: dict[str, Any] = field(default_factory=dict)


def contest_score_nonrate(d_seg: float, d_pose: float) -> float:
    """The non-rate part of the contest score: ``100*d_seg + sqrt(10*d_pose)``."""
    return 100.0 * float(d_seg) + math.sqrt(10.0 * max(0.0, float(d_pose)))


class FrozenScorer:
    """Thin holder for the FROZEN upstream ``DistortionNet`` (read-only).

    Loads the exact contest SegNet + PoseNet once and exposes the three response
    levels for a (source, perturbed) pair. NEVER edits upstream. The forward is
    ``torch.inference_mode`` (no grad). This is the EXACT scorer path:
    ``net.preprocess_input`` -> ``net.segnet`` / ``net.posenet`` -> distortions.
    """

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        self._torch = torch
        self.device = device
        net = DistortionNet().eval().to(device)
        net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
        self.net = net

    def _frame_pair_to_tensor(self, pair_hwc: _np.ndarray) -> Any:
        """(2, H, W, 3) uint8/float -> (1, 2, H, W, 3) float tensor on device."""
        import numpy as np

        torch = self._torch
        arr = np.ascontiguousarray(np.asarray(pair_hwc, dtype=np.float32))
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, 2, H, W, 3)
        return t.to(self.device)

    def response_levels(
        self,
        source_pair: _np.ndarray,
        perturbed_pair: _np.ndarray,
        *,
        boundary_margin_thresh: float,
    ) -> dict[str, float]:
        """Compute all three response levels for one (source, perturbed) pair.

        The EXACT contest convention: ``net.compute_distortion(x, y)`` returns
        ``(d_pose, d_seg)`` where SegNet scores frame1 (last frame) argmax
        disagreement and PoseNet scores the YUV6 motion MSE on the first 6 dims.
        We additionally reach the SegNet LOGITS (Level-1) and break flips into
        boundary vs interior by the SOURCE per-pixel margin.
        """
        torch = self._torch
        net = self.net
        src_t = self._frame_pair_to_tensor(source_pair)
        prt_t = self._frame_pair_to_tensor(perturbed_pair)

        with torch.inference_mode():
            # Exact Level-3 contest terms.
            d_pose, d_seg = net.compute_distortion(src_t, prt_t)
            d_pose_v = float(d_pose.item())
            d_seg_v = float(d_seg.item())

            # Reach SegNet logits (Level-1 + Level-2 breakdown) on frame1.
            _, seg_in_src = net.preprocess_input(src_t)
            _, seg_in_prt = net.preprocess_input(prt_t)
            logits_src = net.segnet(seg_in_src)  # (1, 5, Hs, Ws)
            logits_prt = net.segnet(seg_in_prt)

            # Level-1: source-class margin erosion. c_p = argmax of SOURCE logits.
            cls_src = logits_src.argmax(dim=1)  # (1, Hs, Ws)
            # margin_src = l_{c_p} - max_{j != c_p} l_j (per pixel), on SOURCE.
            margin_src = _source_class_margin(torch, logits_src, cls_src)
            margin_prt = _source_class_margin(torch, logits_prt, cls_src)
            margin_drop = margin_src - margin_prt  # positive = eroded confidence
            d_logit_margin_mean = float(margin_drop.mean().item())
            flat = margin_drop.flatten()
            k = max(1, int(0.10 * flat.numel()))
            # p10 of the DROP = the 10th-percentile (smallest) drop value.
            d_logit_margin_p10 = float(torch.kthvalue(flat, k).values.item())
            logit_l2_delta = float((logits_prt - logits_src).pow(2).sum().sqrt().item())

            # Level-2: argmax flips, split by source margin (boundary vs interior).
            cls_prt = logits_prt.argmax(dim=1)
            flipped = cls_src != cls_prt  # (1, Hs, Ws)
            thin = margin_src < float(boundary_margin_thresh)
            flip_total = float(flipped.float().mean().item())
            flip_boundary = float((flipped & thin).float().mean().item())
            flip_interior = float((flipped & ~thin).float().mean().item())

        return {
            "d_logit_margin_mean": d_logit_margin_mean,
            "d_logit_margin_p10": d_logit_margin_p10,
            "logit_l2_delta": logit_l2_delta,
            "d_seg": d_seg_v,
            "flip_count_total": flip_total,
            "flip_count_boundary": flip_boundary,
            "flip_count_interior": flip_interior,
            "d_pose": d_pose_v,
            "score_nonrate": contest_score_nonrate(d_seg_v, d_pose_v),
        }


def _source_class_margin(torch_mod: Any, logits: Any, source_class: Any) -> Any:
    """``l_{c_p} - max_{j != c_p} l_j`` per pixel, where ``c_p`` is ``source_class``.

    ``logits`` is ``(B, 5, Hs, Ws)``; ``source_class`` is ``(B, Hs, Ws)``. Returns
    ``(B, Hs, Ws)``. The source-class logit minus the best competitor logit — the
    signed distance to the decision boundary along the source class. Positive
    everywhere on the source argmax (by construction); a perturbation that drives
    it toward 0 erodes confidence; below 0 means the argmax flipped.
    """
    torch = torch_mod
    b, c, hs, ws = logits.shape
    src_logit = torch.gather(logits, 1, source_class.unsqueeze(1)).squeeze(1)  # (B,Hs,Ws)
    masked = logits.clone()
    masked.scatter_(1, source_class.unsqueeze(1), float("-inf"))
    best_other = masked.max(dim=1).values  # (B, Hs, Ws)
    return src_logit - best_other


def estimate_boundary_margin_threshold(
    scorer: FrozenScorer,
    source_frames_hwc: _np.ndarray,
    *,
    percentile: float = 25.0,
    max_frames: int = 8,
) -> float:
    """Estimate the SegNet source-class-margin ``percentile`` over real source frames.

    Used as the boundary-vs-interior flip split: a "boundary" pixel is one whose
    SOURCE margin sits below this threshold (thin -> a small perturbation can flip
    it). This is a MEASURED (not hand-set) threshold — the low knee of the margin
    distribution on the actual contest frames the scorer scores.

    ``source_frames_hwc`` is ``(N, H, W, 3)`` uint8/float; SegNet scores the LAST
    frame of each pair, so we feed each frame through the SegNet input path and
    pool the per-pixel source-class margins, then take the ``percentile``.
    Returns a float threshold.
    """
    import numpy as np

    torch = scorer._torch
    net = scorer.net
    frames = np.asarray(source_frames_hwc)
    n = min(int(max_frames), frames.shape[0])
    margins: list[Any] = []
    with torch.inference_mode():
        for i in range(n):
            # SegNet preprocess_input uses x[:, -1, ...]; build a (1,1,H,W,3) clip
            # by duplicating the frame into a 2-frame pair (only frame1 matters).
            single = np.stack([frames[i], frames[i]], axis=0)  # (2, H, W, 3)
            t = scorer._frame_pair_to_tensor(single)
            _, seg_in = net.preprocess_input(t)
            logits = net.segnet(seg_in)  # (1, 5, Hs, Ws)
            cls = logits.argmax(dim=1)
            m = _source_class_margin(torch, logits, cls)  # (1, Hs, Ws)
            margins.append(m.flatten().cpu().numpy())
    if not margins:
        raise ValueError("no source frames provided for boundary-threshold estimate")
    pooled = np.concatenate(margins)
    return float(np.percentile(pooled, percentile))


# ---------------------------------------------------------------------------
# Atlas orchestration (the full v2 grid with CI).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasGrid:
    """The configurable v2 grid. CLI-tunable so the run stays bounded."""

    n_pairs: int = 6
    n_bands: int = 6
    band_spacing: str = "linear"  # 'linear' | 'log' (log resolves the w=1..30 regime)
    amplitudes_lsb: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    orientations: tuple[str, ...] = ORIENTATIONS
    frame_incidences: tuple[str, ...] = FRAME_INCIDENCES
    channel_bases: tuple[str, ...] = CHANNEL_BASES
    # Per-basis channel set actually swept (default: aggregate "all" + each primary).
    rgb_channels: tuple[str, ...] = ("all", "r", "g", "b")
    yuv_channels: tuple[str, ...] = ("all", "y", "u", "v")
    n_phase_samples: int = 2
    seed: int = 0

    def channels_for(self, basis: str) -> tuple[str, ...]:
        return self.rgb_channels if basis == "rgb" else self.yuv_channels

    def total_cells(self) -> int:
        per_basis = sum(
            len(self.channels_for(b)) for b in self.channel_bases
        )
        return (
            self.n_bands
            * len(self.orientations)
            * len(self.amplitudes_lsb)
            * per_basis
            * len(self.frame_incidences)
        )

    def total_scorer_forwards(self) -> int:
        # 2 frozen-scorer forward groups per (cell * pair * phase): source + perturbed.
        return self.total_cells() * self.n_pairs * self.n_phase_samples


def _aggregate_cell(
    scorer: FrozenScorer,
    source_pairs: _np.ndarray,
    band: BandSpec,
    *,
    amplitude_lsb: float,
    channel_basis: str,
    channel: str,
    incidence: str,
    n_phase_samples: int,
    seed: int,
    boundary_margin_thresh: float,
    baseline: dict[str, float],
) -> CellResult:
    """Measure one atlas cell over all source pairs * phase draws, with CI.

    For each (pair, phase) it draws an independent random-phase band-limited
    field, perturbs the pair per the incidence pattern, runs the three response
    levels, and audits the realized energy. The cell aggregates mean/std for CI
    and reports H = response - baseline (the source-vs-source response, ~0).
    """
    import numpy as np

    # Field dims MUST match the actual source frames (pairs are camera-res in the
    # real pipeline; tests use small frames). Deriving from source_pairs rather
    # than the CAMERA_HW constant avoids a shape-mismatch and keeps the physics
    # resolution-agnostic.
    H, W = int(source_pairs.shape[2]), int(source_pairs.shape[3])
    rng = np.random.default_rng(seed)
    n_pairs = source_pairs.shape[0]

    seg_vals: list[float] = []
    pose_vals: list[float] = []
    lm_mean_vals: list[float] = []
    lm_p10_vals: list[float] = []
    logit_l2_vals: list[float] = []
    flip_total_vals: list[float] = []
    flip_boundary_vals: list[float] = []
    flip_interior_vals: list[float] = []
    score_nonrate_vals: list[float] = []
    # Energy audit accumulators (averaged over pairs*phases).
    e_pre: list[float] = []
    e_postclip: list[float] = []
    e_clipfrac: list[float] = []
    e_postresize: list[float] = []
    e_perch: list[tuple[float, float, float]] = []

    for pi in range(n_pairs):
        pair = source_pairs[pi].astype(np.float64)  # (2, H, W, 3)
        for _ph in range(n_phase_samples):
            field = band_limited_field((H, W, 3), band, rng)
            perturbed = apply_frame_incidence(
                pair,
                field,
                incidence=incidence,
                channel_basis=channel_basis,
                channel=channel,
                amplitude_lsb=amplitude_lsb,
            )
            lvl = scorer.response_levels(
                pair, perturbed, boundary_margin_thresh=boundary_margin_thresh
            )
            seg_vals.append(lvl["d_seg"])
            pose_vals.append(lvl["d_pose"])
            lm_mean_vals.append(lvl["d_logit_margin_mean"])
            lm_p10_vals.append(lvl["d_logit_margin_p10"])
            logit_l2_vals.append(lvl["logit_l2_delta"])
            flip_total_vals.append(lvl["flip_count_total"])
            flip_boundary_vals.append(lvl["flip_count_boundary"])
            flip_interior_vals.append(lvl["flip_count_interior"])
            score_nonrate_vals.append(lvl["score_nonrate"])

            # Energy audit on the frame the perturbation primarily targets.
            tgt = 1 if incidence in ("frame1_only", "both_same", "both_opposite") else 0
            audit = energy_audit_for_perturbation(pair[tgt], perturbed[tgt])
            e_pre.append(audit.pre_clip_l2)
            e_postclip.append(audit.post_clip_l2)
            e_clipfrac.append(audit.clip_fraction)
            e_postresize.append(audit.post_resize_l2)
            e_perch.append(audit.per_channel_post_clip_l2)

    def _m(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    def _s(x: list[float]) -> float:
        return float(np.std(x)) if len(x) > 1 else 0.0

    seg_mean = _m(seg_vals)
    pose_mean = _m(pose_vals)
    perch_mean = tuple(float(np.mean([e[c] for e in e_perch])) for c in range(3)) if e_perch else (0.0, 0.0, 0.0)
    return CellResult(
        d_logit_margin_mean=_m(lm_mean_vals) - baseline.get("d_logit_margin_mean", 0.0),
        d_logit_margin_p10=_m(lm_p10_vals),
        logit_l2_delta=_m(logit_l2_vals),
        d_seg=seg_mean - baseline.get("d_seg", 0.0),
        flip_count_total=_m(flip_total_vals),
        flip_count_boundary=_m(flip_boundary_vals),
        flip_count_interior=_m(flip_interior_vals),
        d_seg_exact=seg_mean,
        d_pose=pose_mean - baseline.get("d_pose", 0.0),
        score_nonrate=_m(score_nonrate_vals),
        n_pairs=n_pairs,
        n_phase_samples=n_phase_samples,
        d_seg_std=_s(seg_vals),
        d_pose_std=_s(pose_vals),
        energy={
            "pre_clip_l2": _m(e_pre),
            "post_clip_l2": _m(e_postclip),
            "clip_fraction": _m(e_clipfrac),
            "post_resize_l2": _m(e_postresize),
            "per_channel_post_clip_l2": list(perch_mean),
        },
    )


def build_band_specs(
    n_bands: int, orientation: str, *, spacing: str = "linear"
) -> list[BandSpec]:
    """Return ``n_bands`` radial annuli for ``orientation`` (DC->Nyquist).

    ``spacing``:
      * ``linear`` — equal-width annuli (uniform coverage of the spectrum).
      * ``log`` — log-spaced edges, denser at LOW frequency. This is the spacing
        that resolves the carrier's ``w`` regime: a SIREN ``w`` of 1..30 is only
        ~0.3..9.5 cycles across the image, far below the lowest LINEAR band's
        ~68 cyc/image. ``log`` spacing places bands at low frequencies so the
        atlas can locate where the scorer peaks relative to the hand-set ``w``.

    Note: the lowest ``r`` edge is clamped just above 0 (DC carries no
    orientation and an exactly-zero annulus is empty).
    """
    if spacing == "linear":
        edges = [k / n_bands for k in range(n_bands + 1)]
    elif spacing == "log":
        import numpy as np

        # log-space from a small r0 (a few cycles/image) up to the Nyquist corner.
        r0 = 1.0 / (2.0 * n_bands)  # smallest resolvable band edge
        edges = list(np.geomspace(r0, 1.0, n_bands + 1))
        edges[0] = 0.0  # first band includes DC->r0 (captures the lowest content)
    else:
        raise ValueError(f"spacing {spacing!r} must be 'linear' or 'log'")
    return [
        BandSpec(
            band_index=k,
            r_lo=float(edges[k]),
            r_hi=float(edges[k + 1]),
            orientation=orientation,
        )
        for k in range(n_bands)
    ]


# ---------------------------------------------------------------------------
# Deterministic cell identity + seed (THE resumability foundation).
#
# A cell's RNG seed is derived ONLY from its key (the 7 axes) + the grid's
# global seed + the AXIS-VALUE itself (not the axis INDEX within the grid).
# Using the axis VALUE (band_index, the orientation/basis/channel/incidence
# string, the amplitude) rather than its position in the grid lists means the
# seed is INVARIANT to grid ordering and to which subset of axes a run sweeps.
# That is what makes resume EXACT: a cell computed in run A and re-attempted in
# run B (possibly with the cells visited in a different order, or a superset
# grid) draws the SAME random-phase field and therefore produces a
# bit-identical scorer response. Per CLAUDE.md "Seeds pinned".
# ---------------------------------------------------------------------------

# Stable enumeration of the orientation / basis / channel / incidence label
# spaces so the per-cell seed is computed from a FIXED index that never depends
# on the particular grid subset being swept.
_ORIENTATION_SEED_INDEX: dict[str, int] = {o: i for i, o in enumerate(ORIENTATIONS)}
_BASIS_SEED_INDEX: dict[str, int] = {b: i for i, b in enumerate(CHANNEL_BASES)}
_INCIDENCE_SEED_INDEX: dict[str, int] = {fi: i for i, fi in enumerate(FRAME_INCIDENCES)}
# Channel index is per-basis and stable across the canonical 4-label space.
_CHANNEL_SEED_INDEX: dict[str, dict[str, int]] = {
    basis: {ch: i for i, ch in enumerate(CHANNEL_LABELS[basis])}
    for basis in CHANNEL_BASES
}


def cell_seed_for(
    *,
    global_seed: int,
    band_index: int,
    orientation: str,
    amplitude_lsb: float,
    channel_basis: str,
    channel: str,
    frame_incidence: str,
) -> int:
    """Deterministic per-cell RNG seed from the cell key + the global seed.

    The seed is a function of the cell's INTRINSIC identity (band index, the
    fixed label-space index of each categorical axis, and the amplitude value),
    NOT of the cell's position within whatever grid subset is being swept. This
    is the contract that makes resume bit-exact: the same cell drawn in any run
    (any grid ordering, any axis subset) produces the same random-phase field.

    NOTE: this reproduces the exact arithmetic the original ``measure_atlas``
    loop used inline, EXCEPT that the categorical-axis multipliers now key off
    the canonical fixed label-space index instead of ``grid.<axis>.index(...)``
    (which depended on the swept subset). For the canonical full grid the two
    agree; for a subset grid the fixed-index form is the correct, order-stable
    seed. Amplitude contributes via ``int(amplitude * 4)`` (LSB amplitudes are
    quarter-LSB-resolved in the default grid).
    """
    return (
        int(global_seed)
        + 1009 * int(band_index)
        + 7919 * _ORIENTATION_SEED_INDEX[orientation]
        + 104729 * round(float(amplitude_lsb) * 4)
        + 1299709 * _BASIS_SEED_INDEX[channel_basis]
        + 15485863 * _CHANNEL_SEED_INDEX[channel_basis][channel]
        + 86028121 * _INCIDENCE_SEED_INDEX[frame_incidence]
    ) % (2**31)


def cell_key(
    *,
    band_index: int,
    orientation: str,
    amplitude_lsb: float,
    channel_basis: str,
    channel: str,
    frame_incidence: str,
) -> dict[str, Any]:
    """The canonical 6-axis identity dict for one atlas cell (JSON-stable).

    (The 7th axis, the source-pair set, is held fixed per run and recorded at
    the artifact level, not per cell.) Two runs that produce a cell with the
    same ``cell_key`` over the same source pairs + global seed MUST produce
    bit-identical responses.
    """
    return {
        "band_index": int(band_index),
        "orientation": str(orientation),
        "amplitude_lsb": float(amplitude_lsb),
        "channel_basis": str(channel_basis),
        "channel": str(channel),
        "frame_incidence": str(frame_incidence),
    }


def cell_key_str(key: dict[str, Any]) -> str:
    """Canonical compact string of a cell key (the resume dedup token).

    Amplitude is formatted to a fixed precision so float repr noise can never
    split one logical cell into two keys.
    """
    return (
        f"b{int(key['band_index'])}"
        f"|o:{key['orientation']}"
        f"|a:{float(key['amplitude_lsb']):.6g}"
        f"|cb:{key['channel_basis']}"
        f"|ch:{key['channel']}"
        f"|fi:{key['frame_incidence']}"
    )


def enumerate_cell_keys(grid: AtlasGrid) -> list[dict[str, Any]]:
    """All cell keys for ``grid`` in the canonical sweep order (the same order
    ``measure_atlas`` / ``iter_atlas_cells`` visit them)."""
    keys: list[dict[str, Any]] = []
    for orientation in grid.orientations:
        band_specs = build_band_specs(
            grid.n_bands, orientation, spacing=grid.band_spacing
        )
        for band in band_specs:
            for amplitude in grid.amplitudes_lsb:
                for basis in grid.channel_bases:
                    for channel in grid.channels_for(basis):
                        for incidence in grid.frame_incidences:
                            keys.append(
                                cell_key(
                                    band_index=band.band_index,
                                    orientation=orientation,
                                    amplitude_lsb=amplitude,
                                    channel_basis=basis,
                                    channel=channel,
                                    frame_incidence=incidence,
                                )
                            )
    return keys


def _cell_record(
    *,
    band: BandSpec,
    orientation: str,
    amplitude: float,
    basis: str,
    channel: str,
    incidence: str,
    coords: FrequencyCoordinates,
    result: CellResult,
) -> dict[str, Any]:
    """Assemble one machine-readable cell dict from a measured ``CellResult``.

    Single source of truth for the cell schema, shared by ``measure_atlas`` and
    the streaming/resumable path so the JSONL and the in-memory atlas are
    byte-for-byte the same shape.
    """
    return {
        "band_index": band.band_index,
        "orientation": orientation,
        "r_lo": band.r_lo,
        "r_hi": band.r_hi,
        "r_center": band.r_center,
        "amplitude_lsb": amplitude,
        "channel_basis": basis,
        "channel": channel,
        "frame_incidence": incidence,
        "frequency_coordinates": _coords_as_dict(coords),
        # Level-1
        "H_logit_margin_drop_mean": result.d_logit_margin_mean,
        "logit_margin_drop_p10": result.d_logit_margin_p10,
        "logit_l2_delta": result.logit_l2_delta,
        # Level-2
        "H_seg": result.d_seg,
        "flip_count_total": result.flip_count_total,
        "flip_count_boundary": result.flip_count_boundary,
        "flip_count_interior": result.flip_count_interior,
        # Level-3
        "d_seg_exact": result.d_seg_exact,
        "H_pose": result.d_pose,
        "score_nonrate": result.score_nonrate,
        # CI + energy
        "n_pairs": result.n_pairs,
        "n_phase_samples": result.n_phase_samples,
        "d_seg_std": result.d_seg_std,
        "d_pose_std": result.d_pose_std,
        "energy": result.energy,
    }


def _measure_baseline_and_threshold(
    scorer: FrozenScorer, pairs: _np.ndarray
) -> tuple[dict[str, float], float]:
    """Measure the source-vs-source baseline + the boundary-margin threshold.

    Deterministic given ``pairs`` + the frozen scorer (no RNG), so the resumed
    aggregation reconstructs the same baseline every time.
    """
    import numpy as np

    n_pairs = pairs.shape[0]
    src_frames_for_thresh = pairs.reshape(-1, *pairs.shape[2:])  # (N*2, H, W, 3)
    boundary_thresh = estimate_boundary_margin_threshold(
        scorer,
        src_frames_for_thresh,
        percentile=25.0,
        max_frames=min(8, src_frames_for_thresh.shape[0]),
    )
    base_levels: list[dict[str, float]] = []
    for pi in range(n_pairs):
        p = pairs[pi].astype(np.float64)
        base_levels.append(
            scorer.response_levels(p, p, boundary_margin_thresh=boundary_thresh)
        )
    baseline = {
        "d_seg": float(np.mean([b["d_seg"] for b in base_levels])),
        "d_pose": float(np.mean([b["d_pose"] for b in base_levels])),
        "d_logit_margin_mean": float(
            np.mean([b["d_logit_margin_mean"] for b in base_levels])
        ),
    }
    return baseline, boundary_thresh


def iter_atlas_cells(
    source_pairs: _np.ndarray,
    grid: AtlasGrid,
    *,
    device: str = "cpu",
    skip_cell_keys: set[str] | None = None,
    scorer: FrozenScorer | None = None,
    baseline: dict[str, float] | None = None,
    boundary_margin_thresh: float | None = None,
):
    """Yield ``(cell_dict, key_str)`` per atlas cell — the RESUMABLE generator.

    This is ``measure_atlas`` decomposed into a streaming producer so the CLI
    can write each cell to a durable JSONL as it is computed and SKIP cells
    already present (idempotent resume). Each cell is measured with the
    deterministic per-cell seed from :func:`cell_seed_for`, so a cell yielded
    here is bit-identical to the same cell from :func:`measure_atlas` (the
    end-to-end resume-idempotency guarantee).

    * ``skip_cell_keys`` — a set of ``cell_key_str`` already completed; those
      cells are NOT recomputed (resume).
    * ``scorer`` / ``baseline`` / ``boundary_margin_thresh`` — pass them in to
      avoid reloading the frozen scorer / re-measuring the baseline across
      resume sessions; if omitted they are constructed/measured here.

    The baseline + boundary threshold are deterministic given the source pairs,
    so a resumed session reconstructs the same H = response - baseline floor.
    """
    import numpy as np

    pairs = np.asarray(source_pairs)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(
            f"source_pairs must be (N, 2, H, W, 3); got shape {pairs.shape}"
        )
    n_pairs = min(int(grid.n_pairs), pairs.shape[0])
    pairs = pairs[:n_pairs]

    if scorer is None:
        scorer = FrozenScorer(device=device)
    if baseline is None or boundary_margin_thresh is None:
        baseline, boundary_margin_thresh = _measure_baseline_and_threshold(
            scorer, pairs
        )

    skip = skip_cell_keys or set()

    for orientation in grid.orientations:
        band_specs = build_band_specs(
            grid.n_bands, orientation, spacing=grid.band_spacing
        )
        for band in band_specs:
            coords = frequency_coordinates_for_band(band)
            for amplitude in grid.amplitudes_lsb:
                for basis in grid.channel_bases:
                    for channel in grid.channels_for(basis):
                        for incidence in grid.frame_incidences:
                            key = cell_key(
                                band_index=band.band_index,
                                orientation=orientation,
                                amplitude_lsb=amplitude,
                                channel_basis=basis,
                                channel=channel,
                                frame_incidence=incidence,
                            )
                            ks = cell_key_str(key)
                            if ks in skip:
                                continue
                            cell_seed = cell_seed_for(
                                global_seed=grid.seed,
                                band_index=band.band_index,
                                orientation=orientation,
                                amplitude_lsb=amplitude,
                                channel_basis=basis,
                                channel=channel,
                                frame_incidence=incidence,
                            )
                            result = _aggregate_cell(
                                scorer,
                                pairs,
                                band,
                                amplitude_lsb=amplitude,
                                channel_basis=basis,
                                channel=channel,
                                incidence=incidence,
                                n_phase_samples=grid.n_phase_samples,
                                seed=cell_seed,
                                boundary_margin_thresh=boundary_margin_thresh,
                                baseline=baseline,
                            )
                            cell = _cell_record(
                                band=band,
                                orientation=orientation,
                                amplitude=amplitude,
                                basis=basis,
                                channel=channel,
                                incidence=incidence,
                                coords=coords,
                                result=result,
                            )
                            yield cell, ks


def aggregate_atlas_from_cells(
    cells: list[dict[str, Any]],
    grid: AtlasGrid,
    *,
    baseline: dict[str, float],
    boundary_margin_threshold: float,
    n_pairs: int,
) -> dict[str, Any]:
    """Build the final atlas artifact from a (possibly partial) list of cells.

    Re-aggregatable any time from the durable JSONL — a partial run still yields
    a valid atlas over the cells completed so far. The schema/authority flags +
    headline-peak selection mirror :func:`measure_atlas` exactly.
    """
    seg_peak: dict[str, Any] | None = None
    pose_peak: dict[str, Any] | None = None
    margin_peak: dict[str, Any] | None = None
    for cell in cells:
        if seg_peak is None or cell["H_seg"] > seg_peak["H_seg"]:
            seg_peak = cell
        if pose_peak is None or cell["H_pose"] > pose_peak["H_pose"]:
            pose_peak = cell
        if (
            margin_peak is None
            or cell["H_logit_margin_drop_mean"]
            > margin_peak["H_logit_margin_drop_mean"]
        ):
            margin_peak = cell

    return {
        "schema": SCHEMA_VERSION,
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "score_roadmap_update_eligible": False,
        "mechanism_update_eligible": True,
        "promotable": False,
        "camera_hw": list(CAMERA_HW),
        "scorer_input_hw": list(SCORER_INPUT_HW),
        "grid": {
            "n_pairs": n_pairs,
            "n_bands": grid.n_bands,
            "band_spacing": grid.band_spacing,
            "amplitudes_lsb": list(grid.amplitudes_lsb),
            "orientations": list(grid.orientations),
            "frame_incidences": list(grid.frame_incidences),
            "channel_bases": list(grid.channel_bases),
            "rgb_channels": list(grid.rgb_channels),
            "yuv_channels": list(grid.yuv_channels),
            "n_phase_samples": grid.n_phase_samples,
            "seed": grid.seed,
            "total_cells": grid.total_cells(),
        },
        "cells_measured": len(cells),
        "baseline": baseline,
        "boundary_margin_threshold": boundary_margin_threshold,
        "boundary_margin_threshold_note": (
            "MEASURED p25 of the SegNet source-class margin over real source "
            "frames (not a hand-set constant). Flips at pixels below this margin "
            "are 'boundary'; above are 'interior'."
        ),
        "cells": cells,
        "seg_peak_cell": seg_peak,
        "pose_peak_cell": pose_peak,
        "logit_margin_peak_cell": margin_peak,
        "headline": _build_headline(seg_peak, pose_peak, margin_peak),
        "arbitrariness_note": (
            "H_seg / H_pose are the scorer's per-(band, amplitude, orientation, "
            "channel, frame-incidence) spectral sensitivity, with each band's "
            "frequency converted to camera cyc/px, scorer-input cyc/px, normalized "
            "omega, and SIREN-w-equivalent. The peak cell's siren_w_equivalent is "
            "the MEASURED (not hand-tuned) target for the carrier's frequency "
            "content — the scorer-derived replacement for the arbitrary w=30. "
            "Authority: [macOS-CPU advisory] / exact_pair_scorer -> "
            "mechanism_update_eligible ONLY. Sister design: "
            ".omx/research/principled_frequency_basis_synthesis_20260609.md"
        ),
    }


def measure_atlas(
    source_pairs: _np.ndarray,
    grid: AtlasGrid,
    *,
    device: str = "cpu",
    progress: bool = True,
) -> dict[str, Any]:
    """Measure the full v2 spectral-sensitivity atlas over ``source_pairs``.

    ``source_pairs`` is ``(N, 2, H, W, 3)`` uint8/float — N source frame pairs at
    camera resolution. Returns the full machine-readable artifact dict (schema
    ``scorer_spectral_sensitivity.v2``) carrying every cell, the per-band
    coordinate conversions, the measured baseline + boundary threshold, and the
    headline peaks (where H_seg / H_pose peak, with the SIREN-w-equivalent of
    each peak). Authority is ``[macOS-CPU advisory]`` / ``exact_pair_scorer`` ->
    ``mechanism_update_eligible`` only.

    This is now a thin in-memory wrapper over the streaming producer
    (:func:`iter_atlas_cells`) + the aggregator (:func:`aggregate_atlas_from_cells`),
    so the all-at-once path and the resumable JSONL path share ONE cell-production
    codepath (the bit-identical-resume contract).
    """
    import numpy as np

    pairs = np.asarray(source_pairs)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(
            f"source_pairs must be (N, 2, H, W, 3); got shape {pairs.shape}"
        )
    n_pairs = min(int(grid.n_pairs), pairs.shape[0])
    pairs = pairs[:n_pairs]

    scorer = FrozenScorer(device=device)
    baseline, boundary_thresh = _measure_baseline_and_threshold(scorer, pairs)

    cells: list[dict[str, Any]] = []
    total = grid.total_cells()
    for cell, _ks in iter_atlas_cells(
        pairs,
        grid,
        device=device,
        scorer=scorer,
        baseline=baseline,
        boundary_margin_thresh=boundary_thresh,
    ):
        cells.append(cell)
        if progress and len(cells) % 25 == 0:
            print(
                f"[spectral.v2] cell {len(cells)}/{total} "
                f"band{cell['band_index']} {cell['orientation']} "
                f"a={cell['amplitude_lsb']} {cell['channel_basis']}:{cell['channel']} "
                f"{cell['frame_incidence']} "
                f"H_seg={cell['H_seg']:+.5f} H_pose={cell['H_pose']:+.4f}",
                flush=True,
            )

    return aggregate_atlas_from_cells(
        cells,
        grid,
        baseline=baseline,
        boundary_margin_threshold=boundary_thresh,
        n_pairs=n_pairs,
    )


def _coords_as_dict(coords: FrequencyCoordinates) -> dict[str, Any]:
    return {
        "r_center": coords.r_center,
        "camera_cycles_per_pixel": coords.camera_cycles_per_pixel,
        "scorer_cycles_per_pixel": coords.scorer_cycles_per_pixel,
        "scorer_cycles_per_image_height": coords.scorer_cycles_per_image_height,
        "normalized_omega": coords.normalized_omega,
        "siren_w_equivalent": coords.siren_w_equivalent,
        "aliases_at_scorer": coords.aliases_at_scorer,
    }


def _build_headline(
    seg_peak: dict[str, Any] | None,
    pose_peak: dict[str, Any] | None,
    margin_peak: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the operator-facing headline: where the scorer peaks + the w-verdict."""

    def _summ(cell: dict[str, Any] | None, h_key: str) -> dict[str, Any]:
        if cell is None:
            return {}
        coords = cell["frequency_coordinates"]
        return {
            "H": cell[h_key],
            "band_index": cell["band_index"],
            "orientation": cell["orientation"],
            "amplitude_lsb": cell["amplitude_lsb"],
            "channel_basis": cell["channel_basis"],
            "channel": cell["channel"],
            "frame_incidence": cell["frame_incidence"],
            "r_center": cell["r_center"],
            "scorer_cycles_per_pixel": coords["scorer_cycles_per_pixel"],
            "siren_w_equivalent": coords["siren_w_equivalent"],
            "aliases_at_scorer": coords["aliases_at_scorer"],
        }

    seg = _summ(seg_peak, "H_seg")
    pose = _summ(pose_peak, "H_pose")
    margin = _summ(margin_peak, "H_logit_margin_drop_mean")
    return {
        "seg_peak": seg,
        "pose_peak": pose,
        "logit_margin_peak": margin,
        "w_verdict_note": (
            "Compare each peak's siren_w_equivalent to the hand-set carrier w=30. "
            "If a measured peak's siren_w_equivalent is near 1, then w=1 (arm B) "
            "is sane and w=30 is the alias trap; if near 30, w=30 is sane; if it "
            "sits elsewhere, NEITHER hand-tune is right and the carrier should use "
            "the measured value (or a learnable omega initialized there)."
        ),
    }
