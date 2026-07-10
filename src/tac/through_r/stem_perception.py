# SPDX-License-Identifier: MIT
"""stem_perception — read the FROZEN SegNet's texture perceiver, then synthesize the MINIMAL
per-class texture that classifies correctly WITH margin through R.

The task-space thesis at the TEXTURE level (operator 2026-07-10, following the palette finding
``negaudit_retests_c1_e5_20260709`` + :mod:`tac.through_r.palette_realization`): argmax is
TEXTURE-dominated — a constant colour is a degenerate texture, and the 216-tile flat probe found
Road wins only 1/216, Lane 0/216, ~90% -> Undrivable. The trained witness beats the flat-paint floor
BECAUSE it synthesizes texture (it has an ``out_tex`` head + ``palette``). So don't reproduce video
texture — produce SegNet-OPTIMAL texture at MINIMAL rate.

SegNet is FROZEN ⇒ its perceiver is a fixed, readable object. This module:

* **PHASE 1 — read the perceiver.** :func:`extract_stem_filters` pulls the EfficientNet-B2
  ``conv_stem`` (32×3×3×3, stride-2, no bias) + ``bn1`` (BatchNormAct2d, SiLU) straight from the
  frozen ``upstream/modules.py`` SegNet. :func:`characterize_filters` reads each kernel's dominant
  COLOR direction (SVD of the 3×9 channel×space matrix), its COLOR-OPPONENCY class (achromatic /
  R-vs-GB / B-vs-RG / …), its LOW-PASS-vs-EDGE character (DC fraction of the collapsed spatial
  kernel) and its dominant ORIENTATION + anisotropy (3×3 DFT peak). :func:`stem_nyquist` reports the
  stride-2 stem's Nyquist: the finest texture PERIOD that even survives the stem, in seg-input px and
  camera px — the scale below which R + stride-2 make texture INVISIBLE.

* **PHASE 2 — per-class minimal sufficient texture.** :func:`make_flat_tile` /
  :func:`make_stripe_tile` / :func:`make_checker_tile` / :func:`make_gabor_tile` synthesize
  cheapest-first parametric textures; :func:`measure_tile_responses` pushes each through the
  canonical R (:func:`resolution_chain.render_grid_to_camera_uint8`) + frozen SegNet and returns the
  per-class SIGNED margin (``logit_c − max_{k≠c} logit_k``, mean over the tile interior).
  :func:`per_class_price_list` searches the families and, per class, returns the CHEAPEST (fewest
  description-length bits) texture that makes that class win with positive margin — the measured
  answer to "precisely what texture is optimal", as a bits-vs-margin curve.

NO-FAKE / REUSE-not-rederive: the R operator + frozen SegNet forward are the canonical
:mod:`tac.through_r` authorities (``render_grid_to_camera_uint8`` + the chunked SegNet forward the
palette module already uses). This module reads the perceiver + synthesizes texture on top; it
re-derives nothing. Authority: constant/tile probes are CONTEXT-FREE decision-geometry probes
(explicitly NON-n600, like :class:`palette_realization.DecisionGeometry`); a d_seg VERDICT is n600
only through :func:`tac.through_r.measure_through_r`. Every realized number is
``[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]`` — never a score; the
pointer (contest-CPU 0.19110) moves only through byte-closed exact eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.through_r.resolution_chain import (
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    render_grid_to_camera_uint8,
)
from tac.witness_control.perclass_verdict import CLASS_NAMES, N_CLASSES

__all__ = [
    "STEM_STRIDE",
    "FilterCharacterization",
    "StemFilterBank",
    "StemPerceptionError",
    "TextureSpec",
    "TileResponse",
    "characterize_filters",
    "extract_stem_filters",
    "make_checker_tile",
    "make_flat_tile",
    "make_gabor_tile",
    "make_stripe_tile",
    "measure_tile_responses",
    "per_class_price_list",
    "stem_nyquist",
    "summarize_filter_bank",
    "synth_tile",
    "texture_dl_bits",
]

STEM_STRIDE = 2  # EfficientNet-B2 conv_stem stride (verified from upstream SegNet)


class StemPerceptionError(ValueError):
    """Raised on a mis-shaped / toy / non-authority stem-perception input."""


# --------------------------------------------------------------------------- #
# PHASE 1 — read the perceiver.                                                #
# --------------------------------------------------------------------------- #
@dataclass
class StemFilterBank:
    """The frozen SegNet EfficientNet-B2 stem: ``conv_stem`` + ``bn1`` (SiLU), read verbatim.

    ``weight`` ``(out=32, in=3, kh=3, kw=3)`` float64 — the first-layer kernels the entire SegNet
    perception is built on. ``bn_*`` the fused BatchNorm running stats + affine (``bn1`` is a timm
    ``BatchNormAct2d`` = BN then SiLU). ``stride=2`` is what sets the Nyquist (:func:`stem_nyquist`).
    """

    weight: np.ndarray
    bn_weight: np.ndarray
    bn_bias: np.ndarray
    bn_mean: np.ndarray
    bn_var: np.ndarray
    bn_eps: float
    act: str
    stride: int
    out_ch: int
    in_ch: int
    kh: int
    kw: int


def extract_stem_filters(segnet: Any | None = None) -> StemFilterBank:
    """Read the frozen SegNet's ``conv_stem`` + ``bn1`` into a :class:`StemFilterBank`.

    ``segnet=None`` loads the frozen CPU authority (:func:`tac.boundary_math.seg_core.load_real_segnet`).
    Reads verbatim — no mutation of the upstream weights (NO-FAKE / no-upstream-edit).
    """

    if segnet is None:
        from tac.boundary_math.seg_core import load_real_segnet

        segnet = load_real_segnet("cpu")
    m = segnet.encoder.model
    cs = m.conv_stem
    bn = m.bn1
    w = cs.weight.detach().cpu().numpy().astype(np.float64)
    if w.ndim != 4 or w.shape[1] != 3:
        raise StemPerceptionError(f"unexpected conv_stem weight shape {w.shape}; expected (O,3,kh,kw)")
    stride = int(cs.stride[0]) if isinstance(cs.stride, (tuple, list)) else int(cs.stride)
    return StemFilterBank(
        weight=w,
        bn_weight=bn.weight.detach().cpu().numpy().astype(np.float64),
        bn_bias=bn.bias.detach().cpu().numpy().astype(np.float64),
        bn_mean=bn.running_mean.detach().cpu().numpy().astype(np.float64),
        bn_var=bn.running_var.detach().cpu().numpy().astype(np.float64),
        bn_eps=float(bn.eps),
        act=type(bn.act).__name__,
        stride=stride,
        out_ch=int(w.shape[0]),
        in_ch=int(w.shape[1]),
        kh=int(w.shape[2]),
        kw=int(w.shape[3]),
    )


@dataclass
class FilterCharacterization:
    """One stem kernel read as {colour direction, opponency, low-pass-vs-edge, orientation}.

    ``color_dir`` ``(3,)`` unit vector — the dominant RGB direction the kernel responds to (first
    left singular vector of the 3×(kh·kw) channel×space matrix, sign-normalized so the
    largest-magnitude channel is positive). ``opponency`` the human colour-opponency label.
    ``dc_fraction`` = ``|Σg| / Σ|g|`` of the collapsed spatial kernel ``g`` (space-only, after
    projecting onto ``color_dir``): ~1 ⇒ pure low-pass (a colour/blur detector), ~0 ⇒ a
    zero-mean edge/high-pass detector. ``orientation_deg`` + ``anisotropy`` from the 3×3 DFT peak.
    ``kind`` a coarse label derived from ``dc_fraction`` + ``anisotropy``.
    """

    idx: int
    color_dir: np.ndarray
    opponency: str
    dc_fraction: float
    orientation_deg: float
    anisotropy: float
    kind: str
    l2_energy: float


def _opponency_label(color_dir: np.ndarray, *, thresh: float = 0.35) -> str:
    """Human colour-opponency name from a sign-normalized unit RGB direction.

    ``thresh`` is the |component| below which a channel is "not significant". All-positive
    significant ⇒ achromatic/luminance; a mix of + and − channels ⇒ an opponent name.
    """

    a = np.asarray(color_dir, dtype=np.float64).reshape(3)
    names = ("R", "G", "B")
    sig = np.abs(a) >= thresh
    pos = [names[i] for i in range(3) if sig[i] and a[i] > 0]
    neg = [names[i] for i in range(3) if sig[i] and a[i] < 0]
    if not neg:
        if len(pos) == 3:
            return "achromatic (luminance)"
        return "+".join(pos) + " (unipolar)" if pos else "flat"
    plus = "".join(pos) if pos else "0"
    minus = "".join(neg)
    return f"+{plus} -{minus} (opponent)"


def _collapse_and_orientation(kernel_hw3: np.ndarray, color_dir: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Project a (kh,kw,3) kernel onto ``color_dir`` -> spatial ``g`` (kh,kw); return (g, orient_deg, aniso).

    Orientation + anisotropy from the 2-D DFT of ``g``: the strongest non-DC frequency bin's offset
    from the (fftshifted) centre gives the orientation; anisotropy = that peak's magnitude share of
    all non-DC energy (1 ⇒ a single oriented grating, ~0 ⇒ isotropic).
    """

    k = np.asarray(kernel_hw3, dtype=np.float64)  # (kh,kw,3)
    cd = np.asarray(color_dir, dtype=np.float64).reshape(3)
    g = k @ cd  # (kh,kw) spatial response along the dominant colour axis
    kh, kw = g.shape
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    cy, cx = kh // 2, kw // 2
    mag[cy, cx] = 0.0  # drop DC
    total = float(mag.sum())
    if total <= 1e-12:
        return g, 0.0, 0.0
    py, px = np.unravel_index(int(np.argmax(mag)), mag.shape)
    dy, dx = py - cy, px - cx
    orient = float(np.degrees(np.arctan2(dy, dx))) % 180.0
    aniso = float(mag[py, px] / total)
    return g, orient, aniso


def characterize_filters(bank: StemFilterBank) -> list[FilterCharacterization]:
    """Characterize every stem kernel: colour direction, opponency, low-pass-vs-edge, orientation."""

    w = np.asarray(bank.weight, dtype=np.float64)  # (O,3,kh,kw)
    chars: list[FilterCharacterization] = []
    for i in range(bank.out_ch):
        f = w[i]  # (3,kh,kw)
        chan_space = f.reshape(3, -1)  # (3, kh*kw)
        # SVD: dominant colour direction = first left singular vector.
        u, s, _vt = np.linalg.svd(chan_space, full_matrices=False)
        color_dir = u[:, 0].astype(np.float64)
        # sign-normalize so the largest-magnitude channel is positive (canonical, deterministic).
        if color_dir[int(np.argmax(np.abs(color_dir)))] < 0:
            color_dir = -color_dir
        kernel_hw3 = np.transpose(f, (1, 2, 0))  # (kh,kw,3)
        g, orient, aniso = _collapse_and_orientation(kernel_hw3, color_dir)
        dc = float(abs(g.sum()) / (np.abs(g).sum() + 1e-12))
        if dc >= 0.6:
            kind = "low-pass (colour/blur)"
        elif aniso >= 0.45:
            kind = "oriented edge"
        else:
            kind = "high-pass (isotropic)"
        chars.append(
            FilterCharacterization(
                idx=i,
                color_dir=color_dir,
                opponency=_opponency_label(color_dir),
                dc_fraction=dc,
                orientation_deg=orient,
                anisotropy=aniso,
                kind=kind,
                l2_energy=float(np.sqrt((f**2).sum())),
            )
        )
    return chars


def summarize_filter_bank(chars: list[FilterCharacterization]) -> dict[str, Any]:
    """Aggregate a filter-bank characterization: counts by kind / opponency, DC + orientation stats."""

    if not chars:
        raise StemPerceptionError("no filter characterizations to summarize")
    by_kind: dict[str, int] = {}
    by_opp: dict[str, int] = {}
    for c in chars:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        by_opp[c.opponency] = by_opp.get(c.opponency, 0) + 1
    dc = np.array([c.dc_fraction for c in chars])
    aniso = np.array([c.anisotropy for c in chars])
    return {
        "n_filters": len(chars),
        "count_by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "count_by_opponency": dict(sorted(by_opp.items(), key=lambda kv: -kv[1])),
        "dc_fraction_mean": float(dc.mean()),
        "dc_fraction_median": float(np.median(dc)),
        "low_pass_share": float((dc >= 0.6).mean()),
        "edge_share": float((dc < 0.6).mean()),
        "anisotropy_mean": float(aniso.mean()),
        "oriented_share": float((aniso >= 0.45).mean()),
    }


def stem_nyquist() -> dict[str, float]:
    """The stride-2 stem Nyquist: the finest texture PERIOD that survives the stem.

    The stem downsamples seg-input (384×512) by ``STEM_STRIDE=2`` to 192×256; the finest period
    representable at the stem OUTPUT is 2 stem-output-px = ``2·stride`` = 4 seg-INPUT px. Seg-input is
    itself R's bilinear-DOWN of camera 874×1164 -> 384×512 (scale ≈ 874/384 ≈ 2.28 H, 1164/512 ≈
    2.27 W), so 4 seg-input px ≈ 9.1 camera px. Texture finer than this period is ALIASED away before
    SegNet's first MBConv even runs — it cannot carry class-discriminative signal. This is WHY a
    sub-period-4 micro-pattern is destroyed by R (measured in :func:`per_class_price_list`).
    """

    seg_in_period_px = float(2 * STEM_STRIDE)  # 2 stem-out px * stride
    cam_scale_h = CAMERA_H / SEG_H
    cam_scale_w = CAMERA_W / SEG_W
    cam_scale = 0.5 * (cam_scale_h + cam_scale_w)
    return {
        "stem_stride": float(STEM_STRIDE),
        "seg_input_hw": float(SEG_H),  # H; W is SEG_W
        "stem_output_h": float(SEG_H // STEM_STRIDE),
        "stem_output_w": float(SEG_W // STEM_STRIDE),
        "finest_period_seg_input_px": seg_in_period_px,
        "cam_to_seg_scale": float(cam_scale),
        "finest_period_camera_px": float(seg_in_period_px * cam_scale),
        "nyquist_cycles_per_seg_input_px": float(1.0 / seg_in_period_px),
    }


# --------------------------------------------------------------------------- #
# PHASE 2 — texture synthesis + description length.                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextureSpec:
    """A parametric texture generator (cheapest-first). ``family`` selects the synth function.

    * ``flat``: 1 colour (``c_a``). ``k=0`` texture components — the palette baseline.
    * ``stripe``: 2 colours (``c_a``,``c_b``), ``period`` px, ``orientation`` deg, ``duty`` in (0,1).
    * ``checker``: 2 colours, ``period`` px.
    * ``gabor``: ``c_a`` base + ``c_b`` modulation colour, ``period`` px, ``orientation`` deg,
      ``phase`` deg (a single band-limited grating; k=1 Gabor component).
    """

    family: str
    c_a: tuple[float, float, float]
    c_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    period: int = 4
    orientation: float = 0.0
    duty: float = 0.5
    phase: float = 0.0


def _color_bits(c: tuple[float, float, float], *, quant: int = 5) -> int:
    """Description-length of one RGB colour at ``quant`` bits/channel (default 5 => 15 bits)."""
    return 3 * int(quant)


def texture_dl_bits(spec: TextureSpec, *, color_quant: int = 5) -> int:
    """Description length (bits) of a texture generator's PARAMETERS (the rate the class carrier pays).

    Colours at ``color_quant`` bits/channel; period ~4 bits (log2 of the sweep); orientation 2 bits
    (4 directions); phase/duty 3 bits when the family uses them; +1 family-tag bit per non-flat family.
    This is the GENERATOR-parameter rate (rule-118: the deterministic synth code is FREE; only these
    fitted parameters are counted, AND only when video-derived — see the memo's compliance boundary).
    """

    if spec.family == "flat":
        return _color_bits(spec.c_a, quant=color_quant)
    base = 1 + _color_bits(spec.c_a, quant=color_quant) + _color_bits(spec.c_b, quant=color_quant)
    if spec.family == "stripe":
        return base + 4 + 2 + 3  # period + orientation + duty
    if spec.family == "checker":
        return base + 4  # period only
    if spec.family == "gabor":
        return base + 4 + 2 + 3  # period + orientation + phase
    raise StemPerceptionError(f"unknown texture family {spec.family!r}")


def _grid_yx(h: int = SEG_H, w: int = SEG_W) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(np.float64), xx.astype(np.float64)


def make_flat_tile(color: tuple[float, float, float], *, h: int = SEG_H, w: int = SEG_W) -> np.ndarray:
    """Constant-colour tile ``(h,w,3)`` float — the degenerate (k=0) texture (palette baseline)."""
    return np.broadcast_to(np.asarray(color, dtype=np.float64), (h, w, 3)).astype(np.float64).copy()


def _oriented_coord(orientation_deg: float, h: int, w: int) -> np.ndarray:
    """Projection of pixel coords onto the grating normal (deg) -> (h,w) float phase coordinate."""
    yy, xx = _grid_yx(h, w)
    th = np.radians(float(orientation_deg))
    return xx * np.cos(th) + yy * np.sin(th)


def make_stripe_tile(spec: TextureSpec, *, h: int = SEG_H, w: int = SEG_W) -> np.ndarray:
    """Two-colour square-wave stripe tile at ``spec.period`` px / ``spec.orientation`` deg."""
    if spec.period < 1:
        raise StemPerceptionError("stripe period must be >= 1")
    t = _oriented_coord(spec.orientation, h, w)
    phase = np.mod(t / float(spec.period), 1.0)
    mask = (phase < float(spec.duty))[..., None]
    a = np.asarray(spec.c_a, dtype=np.float64)
    b = np.asarray(spec.c_b, dtype=np.float64)
    return np.where(mask, a, b).astype(np.float64)


def make_checker_tile(spec: TextureSpec, *, h: int = SEG_H, w: int = SEG_W) -> np.ndarray:
    """Two-colour checkerboard tile with square size ``spec.period`` px."""
    if spec.period < 1:
        raise StemPerceptionError("checker period must be >= 1")
    yy, xx = _grid_yx(h, w)
    parity = ((np.floor(xx / spec.period) + np.floor(yy / spec.period)).astype(np.int64) % 2)[..., None]
    a = np.asarray(spec.c_a, dtype=np.float64)
    b = np.asarray(spec.c_b, dtype=np.float64)
    return np.where(parity == 0, a, b).astype(np.float64)


def make_gabor_tile(spec: TextureSpec, *, h: int = SEG_H, w: int = SEG_W) -> np.ndarray:
    """Single band-limited grating (k=1 Gabor): ``c_a`` + cos-modulation toward ``c_b``.

    ``value = c_a + 0.5*(1+cos(2π t/period + phase))*(c_b - c_a)`` clipped to [0,255]. A smooth
    (band-limited) 2-colour pattern — the natural next step up from a hard stripe.
    """
    if spec.period < 1:
        raise StemPerceptionError("gabor period must be >= 1")
    t = _oriented_coord(spec.orientation, h, w)
    ph = np.radians(float(spec.phase))
    m = 0.5 * (1.0 + np.cos(2.0 * np.pi * t / float(spec.period) + ph))  # (h,w) in [0,1]
    a = np.asarray(spec.c_a, dtype=np.float64)
    b = np.asarray(spec.c_b, dtype=np.float64)
    return np.clip(a[None, None, :] + m[..., None] * (b - a)[None, None, :], 0.0, 255.0)


def synth_tile(spec: TextureSpec, *, h: int = SEG_H, w: int = SEG_W) -> np.ndarray:
    """Dispatch a :class:`TextureSpec` to its synth function -> render-grid float ``(h,w,3)``."""
    if spec.family == "flat":
        return make_flat_tile(spec.c_a, h=h, w=w)
    if spec.family == "stripe":
        return make_stripe_tile(spec, h=h, w=w)
    if spec.family == "checker":
        return make_checker_tile(spec, h=h, w=w)
    if spec.family == "gabor":
        return make_gabor_tile(spec, h=h, w=w)
    raise StemPerceptionError(f"unknown texture family {spec.family!r}")


# --------------------------------------------------------------------------- #
# PHASE 2 — tile-margin measurement (through R, frozen SegNet).               #
# --------------------------------------------------------------------------- #
@dataclass
class TileResponse:
    """One tile's frozen-SegNet response over the tile interior (through R unless ``through_R=False``).

    ``modal_class`` the most-frequent argmax over the interior. ``signed_margin`` ``(n_classes,)`` the
    mean of ``logit_c − max_{k≠c} logit_k`` over the interior — POSITIVE for exactly the winning class
    (the amount by which it beats the runner-up), negative for the rest. ``win_fraction``
    ``(n_classes,)`` the fraction of interior pixels each class wins the argmax.
    """

    modal_class: int
    signed_margin: np.ndarray
    win_fraction: np.ndarray
    n_classes: int
    through_R: bool
    spec: TextureSpec | None = None

    def margin_for(self, c: int) -> float:
        return float(self.signed_margin[int(c)])


def _interior_mask(h: int, w: int, frac: float) -> np.ndarray:
    """Central-``frac`` box interior mask ``(h,w)`` bool (avoids the tile-border receptive-field ramp)."""
    frac = float(np.clip(frac, 0.1, 1.0))
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[mh : h - mh, mw : w - mw] = True
    return mask


def _segnet_logits_field(segnet: Any, seg_frame_hw3: np.ndarray, *, through_R: bool) -> np.ndarray:
    """Frozen SegNet logits ``(n_classes, SEG_H, SEG_W)`` for one render-grid tile.

    ``through_R=True``: push the render-grid tile through R's first half
    (:func:`render_grid_to_camera_uint8`, bicubic-UP to camera + uint8), then SegNet
    ``preprocess_input`` bilinear-DOWNs it back — the real R the texture must survive. ``False``:
    feed the seg-res tile straight in (``preprocess_input`` at seg-res is exact identity), the
    R-free upper bound. Op-for-op mirror of the campaign verdict forward.
    """

    import torch

    a = np.asarray(seg_frame_hw3, dtype=np.float64)
    if a.shape != (SEG_H, SEG_W, 3):
        raise StemPerceptionError(f"tile must be ({SEG_H},{SEG_W},3); got {a.shape}")
    if through_R:
        cam = render_grid_to_camera_uint8(a)  # (CAMERA_H,CAMERA_W,3) uint8
        arr = np.asarray(cam)[None, None]  # (1,1,H,W,3)
    else:
        arr = a[None, None]  # (1,1,SEG_H,SEG_W,3)
    with torch.inference_mode():
        xp = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).contiguous().float()
        seg_in = segnet.preprocess_input(xp)
        logits = segnet(seg_in)[0].cpu().numpy().astype(np.float64)  # (n_classes, SEG_H, SEG_W)
    return logits


def _response_from_logits(logits: np.ndarray, interior: np.ndarray, n_classes: int) -> tuple[int, np.ndarray, np.ndarray]:
    """(modal_class, signed_margin[n], win_fraction[n]) over the interior from a logit field."""
    lg = logits[:, interior]  # (n_classes, P)
    arg = lg.argmax(axis=0)  # (P,)
    win_fraction = np.array([(arg == c).mean() for c in range(n_classes)], dtype=np.float64)
    signed = np.zeros((n_classes,), dtype=np.float64)
    for c in range(n_classes):
        others = np.delete(lg, c, axis=0).max(axis=0)  # (P,)
        signed[c] = float((lg[c] - others).mean())
    modal = int(win_fraction.argmax())
    return modal, signed, win_fraction


def measure_tile_responses(
    segnet: Any,
    specs: list[TextureSpec],
    *,
    through_R: bool = True,
    interior_frac: float = 0.5,
    n_classes: int = N_CLASSES,
) -> list[TileResponse]:
    """Synthesize each :class:`TextureSpec`, push through R + frozen SegNet, read the interior response.

    CONTEXT-FREE constant-tile probe (a whole-frame texture) — a decision-geometry probe, NOT an n600
    d_seg verdict (labelled as such). Real SegNet is the authority; the constant-tile fills the whole
    frame so the interior reads the texture's native class with minimal boundary contamination.
    """

    interior = _interior_mask(SEG_H, SEG_W, interior_frac)
    out: list[TileResponse] = []
    for spec in specs:
        tile = synth_tile(spec)
        logits = _segnet_logits_field(segnet, tile, through_R=through_R)
        modal, signed, win = _response_from_logits(logits, interior, int(n_classes))
        out.append(
            TileResponse(
                modal_class=modal,
                signed_margin=signed,
                win_fraction=win,
                n_classes=int(n_classes),
                through_R=bool(through_R),
                spec=spec,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# PHASE 2 — per-class price list (bits vs margin).                            #
# --------------------------------------------------------------------------- #
@dataclass
class ClassPricePoint:
    """One (bits, margin) point on a class's price curve: a texture that makes the class win."""

    class_idx: int
    class_name: str
    spec: TextureSpec
    bits: int
    margin: float  # signed_margin for this class (>0 ⇒ wins)
    win_fraction: float
    through_R: bool


@dataclass
class ClassPriceList:
    """Per-class cheapest winning texture + the full bits-vs-margin curve (advisory, NON-PROMOTABLE)."""

    per_class_cheapest: dict[str, ClassPricePoint | None]
    per_class_curve: dict[str, list[ClassPricePoint]]
    flat_best_margin: dict[str, float]  # best flat-tile signed margin per class (the k=0 floor)
    through_R: bool
    n_specs_measured: int
    label: str = "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet . NON-PROMOTABLE]"


def _default_color_grid(levels: tuple[float, ...] = (0.0, 96.0, 160.0, 255.0)) -> list[tuple[float, float, float]]:
    return [(r, g, b) for r in levels for g in levels for b in levels]


def build_default_specs(
    *,
    colors: list[tuple[float, float, float]] | None = None,
    periods: tuple[int, ...] = (2, 4, 8, 16),
    orientations: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    max_color_pairs: int = 24,
    seed: int = 0,
) -> list[TextureSpec]:
    """A cheapest-first texture-spec sweep: all flats + a bounded stripe/checker/gabor grid.

    Colours default to a coarse 4-level RGB grid (64 colours). Flats cover every colour; the 2-colour
    families sample ``max_color_pairs`` distinct ordered colour pairs (deterministic, ``seed``) ×
    ``periods`` × ``orientations`` to keep the through-R forward count bounded.
    """

    colors = colors if colors is not None else _default_color_grid()
    specs: list[TextureSpec] = [TextureSpec(family="flat", c_a=c) for c in colors]
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # deterministic bounded pair sample
    while len(pairs) < max_color_pairs and len(seen) < len(colors) * (len(colors) - 1):
        i, j = int(rng.integers(len(colors))), int(rng.integers(len(colors)))
        if i != j and (i, j) not in seen:
            seen.add((i, j))
            pairs.append((i, j))
    for i, j in pairs:
        ca, cb = colors[i], colors[j]
        for p in periods:
            for o in orientations:
                specs.append(TextureSpec(family="stripe", c_a=ca, c_b=cb, period=p, orientation=o))
            specs.append(TextureSpec(family="checker", c_a=ca, c_b=cb, period=p))
        specs.append(TextureSpec(family="gabor", c_a=ca, c_b=cb, period=8, orientation=0.0))
    return specs


def per_class_price_list(
    segnet: Any,
    *,
    specs: list[TextureSpec] | None = None,
    through_R: bool = True,
    interior_frac: float = 0.5,
    n_classes: int = N_CLASSES,
    color_quant: int = 5,
) -> ClassPriceList:
    """Search texture specs; per class return the CHEAPEST (fewest bits) texture that wins with margin.

    The measured answer to "precisely what texture is optimal per class": for each class ``c`` collect
    every spec whose interior signed margin for ``c`` is > 0 (``c`` wins), sorted into a bits-vs-margin
    curve; the cheapest such spec is the headline price. ``flat_best_margin`` records the k=0 floor
    (best constant colour) so a class that ONLY wins with texture (Lane/Road, per the palette finding)
    is visible as ``flat_best_margin < 0`` while ``per_class_cheapest`` is a textured spec.
    """

    specs = specs if specs is not None else build_default_specs()
    responses = measure_tile_responses(
        segnet, specs, through_R=through_R, interior_frac=interior_frac, n_classes=n_classes
    )
    names = CLASS_NAMES if int(n_classes) == N_CLASSES else tuple(f"class_{c}" for c in range(int(n_classes)))
    curves: dict[str, list[ClassPricePoint]] = {nm: [] for nm in names}
    flat_best: dict[str, float] = {nm: float("-inf") for nm in names}
    for resp in responses:
        assert resp.spec is not None
        bits = texture_dl_bits(resp.spec, color_quant=color_quant)
        for c in range(int(n_classes)):
            m = resp.margin_for(c)
            if resp.spec.family == "flat":
                flat_best[names[c]] = max(flat_best[names[c]], m)
            if m > 0.0:
                curves[names[c]].append(
                    ClassPricePoint(
                        class_idx=c,
                        class_name=names[c],
                        spec=resp.spec,
                        bits=bits,
                        margin=m,
                        win_fraction=float(resp.win_fraction[c]),
                        through_R=bool(through_R),
                    )
                )
    cheapest: dict[str, ClassPricePoint | None] = {}
    for nm in names:
        pts = curves[nm]
        # cheapest bits; tie-break by larger margin.
        pts_sorted = sorted(pts, key=lambda p: (p.bits, -p.margin))
        curves[nm] = sorted(pts, key=lambda p: (p.bits, -p.margin))
        cheapest[nm] = pts_sorted[0] if pts_sorted else None
    return ClassPriceList(
        per_class_cheapest=cheapest,
        per_class_curve=curves,
        flat_best_margin={nm: (flat_best[nm] if flat_best[nm] != float("-inf") else 0.0) for nm in names},
        through_R=bool(through_R),
        n_specs_measured=len(specs),
    )
