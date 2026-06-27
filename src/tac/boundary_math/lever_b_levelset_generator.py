# SPDX-License-Identifier: MIT
"""LEVER-B LEVEL-SET — softmax-of-SDF witness + GENERIC curvelet/shearlet front-end.

Deep-math (operator 2026-06-26 "step and non-linear and non-spectral bias … informed by
deep math and topology and manifolds"). The SegNet argmax is a PARTITION (a cartoon
function). Represent it as ``argmax_k phi_k(x)`` of K=5 SIGNED-DISTANCE fields:

  * partition = ``argmax_k phi_k``  (the topology-matched chart for a piecewise-constant
    argmax target);
  * boundary  = the FLAT zero-crossing ``{phi_i = phi_j}``.

WHY SDF (the R-survival cure). A signed-distance field is 1-LIPSCHITZ (``|grad phi| = 1``):
near a class boundary the *margin* field ``m = phi_top1 - phi_top2`` is ~linear through zero
with unit slope and NO oscillation. Under the contest round-trip R (bicubic up to camera ->
uint8 @ camera -> bilinear down to 384x512) the low-pass shifts the zero-crossing by only
``O(blur)`` and MONOTONICALLY — there is ZERO off-boundary ringing. This is the exact cure
for the Gibbs/R-aliasing that caps the sine/Fourier (spectral-bias) witness: a brick-wall /
periodic basis overshoots the indicator near the edge, and uint8 quantization freezes that
overshoot into argmax flips FAR from the true boundary.

WHY the generic CURVELET/SHEARLET front-end (the byte-closeability crux). The measured
decisive d_seg lever is an ORIENTED (anisotropic) basis matched to the all-class boundary
tangent (-48% d_seg). But the in-tree ``directional_fourier_feats`` orients to the GT SegNet
argmax tangent => GT-DERIVED => NOT byte-closeable (trainer bug #5: no inflate-time SegNet,
the RGB witness emits no class map, and storing the tangent field is image-sized). A
curvelet/shearlet bank instead covers ALL orientations at ALL scales from a PARAMETRIC SEED
(J scales x L_j orientations, parabolic L_j ~ L0*2^(j//2)) — it is the provably ``N^-2``
optimal generic basis for curved C^2 singularities (Candes-Donoho curvelets; Labate-Kutyniok
shearlets) and is regenerated identically at decode from ``(J, L0, f0, base)``. The network
LEARNS which orientations to use per region via its trained weights; the BANK costs ZERO
archive bytes (free generic code in inflate.py, rule 118) and leaks NO GT. This is the
directional benefit made FREE + leak-free.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the MLX module is the fast train path; the numpy forward
is the portability reference (ARGMAX-parity, not bit-exact logits). NO MPS. The d_seg VERDICT
is the FROZEN CPU-torch SegNet argmax on the witness frame rendered THROUGH R — NEVER MLX,
NEVER MPS. Evidence ``[macOS-MLX research-signal]`` for the forward, ``[macOS-CPU advisory]``
once scored by the exact SegNet. promotion_eligible=False until a byte-closed exact-eval row.

NO-FAKE: ``signed_distance_fields`` runs the ACTUAL scipy EDT on the ACTUAL label map (a stub
returning constants FAILS the round-trip assertion ``argmax(sdf) == labels``). The curvelet
bank is the ACTUAL parametric polar frequency grid (a random matrix FAILS the orientation-
coverage assertion). ``r_survival_flip_fraction`` applies the ACTUAL contest R and counts the
ACTUAL argmax disagreement — no proxy stands in for the measured flip set.

BORROWED-SUBSTRATE (CLAUDE.md NO-FAKE #7):
  * BORROWED: curvelets (Candes-Donoho 2004) / shearlets (Labate-Kutyniok 2005); WIRE Gabor
    activation (Saragadam et al. CVPR 2023); HOSC (Serrano 2024); Fourier features (Tancik
    2020); FiLM (Perez 2018); level-set / Chan-Vese length term (Osher-Sethian; Chan-Vese
    2001); scipy EDT; the frozen contest scorers + GT decode + CPU authority.
  * OURS-ORIGINAL: representing the SegNet argmax partition as softmax-of-SDF level sets whose
    1-Lipschitz margin field is R-survival-by-construction, driven by a GENERIC (byte-closeable,
    GT-free) curvelet/shearlet directional bank — the joint resolution of the R-aliasing wall
    AND the directional byte-closeability crux for the task-space witness capstone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── deterministic free-table seed (regenerated identically at inflate time) ──
_LEVELSET_SEED = 0
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Coord grid (shared with the RGB witness convention: row-major, gx in [-1,1]).
# ---------------------------------------------------------------------------
def build_coords(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# GENERIC curvelet / shearlet directional Fourier bank (byte-closeable, GT-free).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurveletBankConfig:
    """Parametric (seed-free) polar frequency grid -> a generic anisotropic bank.

    ``n_scales`` J radial octaves; ``n_orient0`` L0 orientations at the coarsest scale, with
    CURVELET parabolic doubling ``L_j = L0 * 2**(j//2)`` (more orientations at finer scales —
    the anisotropic ``width ~ length^2`` law); ``f0`` base radial frequency (cycles over the
    [-1,1] coord span); ``base`` the radial octave ratio; ``n_iso`` extra low isotropic
    frequencies (coarse / DC-ish bands the oriented atoms miss). Fully determined by these 5
    scalars -> reconstructed identically at decode, ZERO archive bytes, no GT leak.
    """

    n_scales: int = 4
    n_orient0: int = 6
    f0: float = 2.0
    base: float = 2.0
    n_iso: int = 4


def stem_nyquist_max_freq_cycles_per_unit(scorer_w: int = 512, stem_stride: int = 2) -> float:
    """LEVER-2 (stem-Nyquist) — max USEFUL curvelet frequency (cycles per coord unit).

    The contest SegNet is EfficientNet-B2 whose stride-2 stem halves the input immediately, so
    the finest spatial feature it can resolve lives on the post-stem feature map (``scorer_w /
    stem_stride`` px wide ~ 256). Sampling theory: that feature map's Nyquist is ``W_f / 2``
    cycles ACROSS the full width. Our coord grid is ``x in [-1,1]`` (span 2) and the curvelet
    feature is ``sin(2*pi f x)`` which completes ``2 f`` cycles across the full width. Equating::

        2 * f_max  =  W_f / 2  =  scorer_w / (2 * stem_stride)
        ==>  f_max = scorer_w / (4 * stem_stride)

    Default (scorer_w=512, stem_stride=2) -> ``f_max = 64`` cycles/unit. Any basis frequency
    ABOVE this encodes detail SegNet structurally cannot see; under the round-trip R (uint8 @
    camera) it instead ALIASES into off-boundary argmax flips (the d_seg killer). So this is the
    free byte/aliasing budget cap: drop (or never generate) atoms with ``|f| > f_max``.

    NOTE (measured 2026-06-27): the DEFAULT curvelet bank (f0=2, base=2, n_scales=4 -> max
    f=16 cycles/unit) is already ~4x BELOW this Nyquist, so capping the curvelet bank itself is a
    no-op at the default config. The over-Nyquist waste lives in the SELF-ORIENT directional
    feats (``freq_across=32, n_dir_freqs=6`` -> across-edge freqs up to 32*2^5 = 1024 cycles/unit,
    16x over Nyquist): the Nyquist-implied directional config is ``n_dir_freqs<=2`` at
    freq_across=32, or ``freq_across=8, n_dir_freqs=4`` (4 octaves all <= f_max, finer angular
    coverage, fewer params). This helper makes the cap executable + auditable for the next sweep.
    """

    return float(scorer_w) / (4.0 * float(max(stem_stride, 1)))


def curvelet_directional_B(cfg: CurveletBankConfig, max_freq: float | None = None) -> np.ndarray:
    """The (2, n_feats) generic curvelet/shearlet frequency matrix (NOT trained, NOT GT).

    Columns are frequency vectors on a polar grid: J scales x L_j orientations (parabolic
    doubling) + ``n_iso`` low isotropic frequencies. Orientations span ``[0, pi)`` (k and -k
    give redundant sin/cos). The resulting ``sin(2*pi X@B), cos(2*pi X@B)`` features are a
    multi-scale, multi-ORIENTATION, anisotropic basis: a curved boundary at any local tangent
    is covered by SOME atom at SOME scale, so the net synthesizes oriented edges cheaply.

    LEVER-2 (stem-Nyquist) ``max_freq`` (cycles per coord unit; default ``None`` = NO cap =
    current behavior, fully ADDITIVE): when given, atoms whose frequency magnitude exceeds
    ``max_freq`` are DROPPED (they encode detail above the SegNet stem Nyquist that only aliases
    under R; see ``stem_nyquist_max_freq_cycles_per_unit``). At least one (lowest-freq) atom is
    always kept so the bank never empties. Reduces ``in_feat`` -> a smaller ``in_proj`` weight
    (a counted decoder param) AND removes aliasing-prone bands. The cap is a deterministic
    function of the 5 cfg scalars + max_freq, so it regenerates identically at decode (free).
    """

    cols: list[np.ndarray] = []
    for j in range(int(cfg.n_scales)):
        f_j = float(cfg.f0) * (float(cfg.base) ** j)
        l_j = int(cfg.n_orient0) * (2 ** (j // 2))  # parabolic curvelet doubling
        for l in range(l_j):
            theta = np.pi * l / l_j
            cols.append(np.array([f_j * np.cos(theta), f_j * np.sin(theta)], dtype=np.float32))
    # Low isotropic coarse bands (radial only; spread over [0,pi) at f0/2).
    for i in range(int(cfg.n_iso)):
        theta = np.pi * i / max(int(cfg.n_iso), 1)
        f_low = float(cfg.f0) * 0.5
        cols.append(np.array([f_low * np.cos(theta), f_low * np.sin(theta)], dtype=np.float32))
    stacked = np.stack(cols, axis=1).astype(np.float32)  # (2, n_feats)
    if max_freq is not None:
        norms = np.sqrt((stacked.astype(np.float64) ** 2).sum(axis=0))  # (n_feats,) |f|
        keep = norms <= float(max_freq) + 1e-6
        if not keep.any():  # never empty the bank — keep the single lowest-freq atom
            keep = norms <= float(norms.min()) + 1e-6
        stacked = stacked[:, keep]
    return stacked


def curvelet_feats(coords: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``[sin(2*pi X@B), cos(2*pi X@B)]`` -> (P, 2*n_feats). Identical at train + inflate."""

    proj = (2.0 * np.pi) * (np.asarray(coords, np.float64) @ np.asarray(B, np.float64))
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def isotropic_fourier_B(n_fourier: int, sigma: float, seed: int = _LEVELSET_SEED) -> np.ndarray:
    """The CONTROL basis: the existing RANDOM isotropic Gaussian Fourier matrix (2, n_fourier)."""

    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, n_fourier)) * sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Activations (numpy reference). WIRE (Gabor) | HOSC | relu — the non-spectral-bias family.
# ---------------------------------------------------------------------------
def wire_activation(u: np.ndarray, w0: float = 20.0, s0: float = 10.0) -> np.ndarray:
    """Real WIRE Gabor wavelet (Saragadam CVPR 2023): ``cos(w0 u) * exp(-(s0 u)^2)``.

    Spatially+spectrally localized -> sharp, oriented features with NO global Gibbs (each atom
    decays), the natural partner of the SDF level-set boundary."""

    u = np.asarray(u, np.float64)
    return (np.cos(w0 * u) * np.exp(-((s0 * u) ** 2))).astype(np.float32)


def hosc_activation(u: np.ndarray, beta: float = 4.0, omega: float = 1.0) -> np.ndarray:
    """HOSC step-native (Serrano 2024): ``tanh(beta * sin(omega u))`` (beta->inf => step train)."""

    u = np.asarray(u, np.float64)
    return np.tanh(beta * np.sin(omega * u)).astype(np.float32)


def _act(u: np.ndarray, name: str, **kw) -> np.ndarray:
    if name == "wire":
        return wire_activation(u, kw.get("w0", 20.0), kw.get("s0", 10.0))
    if name == "hosc":
        return hosc_activation(u, kw.get("beta", 4.0), kw.get("omega", 1.0))
    return np.maximum(u, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Config + numpy reference forward (K SDF fields). ARGMAX-parity contract.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSetConfig:
    num_pairs: int
    bank: CurveletBankConfig = CurveletBankConfig()
    hidden_dim: int = 96
    n_hidden: int = 4
    mod_dim: int = 32
    n_classes: int = 5
    activation: str = "wire"        # "wire" | "hosc" | "relu"
    wire_w0: float = 20.0
    wire_s0: float = 10.0
    hosc_beta: float = 4.0
    hosc_omega: float = 1.0
    front_end: str = "curvelet"     # "curvelet" | "isotropic"
    iso_n_fourier: int = 48
    iso_sigma: float = 8.0
    # LEVER-2 (stem-Nyquist): optional curvelet-bank frequency cap (cycles/unit); None = no cap
    # = current behavior. See ``stem_nyquist_max_freq_cycles_per_unit``. Additive/default-off.
    max_freq: float | None = None

    def in_feat(self) -> int:
        if self.front_end == "isotropic":
            return 2 * int(self.iso_n_fourier)
        return 2 * curvelet_directional_B(self.bank, max_freq=self.max_freq).shape[1]

    def build_B(self) -> np.ndarray:
        if self.front_end == "isotropic":
            return isotropic_fourier_B(self.iso_n_fourier, self.iso_sigma)
        return curvelet_directional_B(self.bank, max_freq=self.max_freq)


def numpy_levelset_forward(
    params: dict[str, np.ndarray],
    feats: np.ndarray,
    mod_vec: np.ndarray,
    cfg: LevelSetConfig,
) -> np.ndarray:
    """Pure-numpy mirror of the MLX level-set witness (float64). Returns (P, n_classes) SDF.

    FiLM-per-(pair,frame) modulates each hidden layer (scale,shift). The output head is LINEAR
    (SDFs are unbounded; the argmax is invariant to an output affine). The hidden nonlinearity
    is the non-spectral-bias activation (WIRE/HOSC) — the topology-matched chart.
    """

    p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    feats = np.asarray(feats, np.float64)
    mod_vec = np.asarray(mod_vec, np.float64)
    akw = dict(w0=cfg.wire_w0, s0=cfg.wire_s0, beta=cfg.hosc_beta, omega=cfg.hosc_omega)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _act(feats @ p["in_proj.weight"].T + p["in_proj.bias"], cfg.activation, **akw)
        film = (mod_vec @ p["film.weight"].T + p["film.bias"]).reshape(cfg.n_hidden, 2, cfg.hidden_dim)
        for li in range(cfg.n_hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = _act((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * scale + shift,
                     cfg.activation, **akw)
        return (h @ p["out.weight"].T + p["out.bias"]).astype(np.float32)


def levelset_argmax(
    params: dict[str, np.ndarray], cfg: LevelSetConfig, coords: np.ndarray, pair_idx: int, h: int, w: int
) -> np.ndarray:
    """Per-(pair,frame) generator partition (H,W) uint8 = ``argmax_k phi_k`` (numpy-portable)."""

    B = cfg.build_B()
    feats = curvelet_feats(coords, B) if cfg.front_end == "curvelet" else _iso_feats(coords, B)
    sdf = numpy_levelset_forward(params, feats, np.asarray(params["code"])[pair_idx], cfg)
    return sdf.argmax(axis=-1).reshape(h, w).astype(np.uint8)


def _iso_feats(coords: np.ndarray, B: np.ndarray) -> np.ndarray:
    proj = np.asarray(coords, np.float64) @ np.asarray(B, np.float64)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Signed-distance construction (the SDF target / init) + level-set regularizers.
# ---------------------------------------------------------------------------
def signed_distance_fields(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class signed distance ``phi_k`` (H,W,K): +EDT inside class k, -EDT outside.

    ``argmax_k phi_k == labels`` EXACTLY (inside class k: phi_k>0, all others<0). This is the
    IDEAL SDF representation of the partition — the R-survival reference + a from-scratch init.
    """

    from scipy import ndimage

    a = np.asarray(labels)
    h, w = a.shape
    out = np.zeros((h, w, int(n_classes)), np.float32)
    for k in range(int(n_classes)):
        inside = a == k
        if inside.all():
            out[..., k] = float(max(h, w))
            continue
        if not inside.any():
            out[..., k] = -float(max(h, w))
            continue
        d_in = ndimage.distance_transform_edt(inside)
        d_out = ndimage.distance_transform_edt(~inside)
        out[..., k] = (d_in - d_out).astype(np.float32)
    return out


def eikonal_penalty(phi_hwk: np.ndarray) -> float:
    """Mean ``(|grad phi_k| - 1)^2`` over classes (the Eikonal / 1-Lipschitz SDF constraint).

    Drives each field toward unit-gradient (true signed distance) so the margin field's
    zero-crossing shifts monotonically (not chaotically) under R. Pure numpy reference; the
    MLX trainer adds the differentiable twin to the loss.
    """

    phi = np.asarray(phi_hwk, np.float64)
    gy, gx = np.gradient(phi, axis=0), np.gradient(phi, axis=1)
    gmag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean((gmag - 1.0) ** 2))


def boundary_length_penalty(phi_hwk: np.ndarray, eps: float = 1.0) -> float:
    """Chan-Vese / Mumford-Shah boundary-LENGTH term: ``sum_k integral |grad H(phi_k)|``.

    Approximated by the smoothed-Heaviside gradient magnitude (delta_eps * |grad phi|). Penalizes
    boundary perimeter -> short, smooth class boundaries (a topology prior matching the SegNet
    argmax's connected regions; suppresses salt-and-pepper flip islands).
    """

    phi = np.asarray(phi_hwk, np.float64)
    delta = (eps / np.pi) / (eps * eps + phi * phi)  # delta_eps(phi)
    gy, gx = np.gradient(phi, axis=0), np.gradient(phi, axis=1)
    gmag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(delta * gmag))


# ---------------------------------------------------------------------------
# Contest round-trip R (field-level proxy) + R-survival flip metric.
# ---------------------------------------------------------------------------
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)


def apply_R_to_fields(fields_hwk: np.ndarray, camera_hw=CAMERA_HW, scorer_hw=SCORER_HW) -> np.ndarray:
    """Contest-faithful R applied per field channel: bicubic up to camera -> uint8 @ camera ->
    bilinear down to scorer. Returns (h2,w2,K). Channels share ONE global affine to [0,255]
    before the uint8 knife-edge (the quantization is what reveals Gibbs vs the SDF's monotone
    shift). Uses torch interpolate (CPU; align_corners=False, matches the MLX/contest order).
    """

    import torch
    import torch.nn.functional as F

    f = np.asarray(fields_hwk, np.float32)
    lo, hi = float(f.min()), float(f.max())
    scale = 255.0 / max(hi - lo, 1e-6)
    x = (f - lo) * scale  # (h,w,K) in ~[0,255], shared affine
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()  # (1,K,h,w)
    up = F.interpolate(t, size=camera_hw, mode="bicubic", align_corners=False)
    up = torch.clamp(up, 0.0, 255.0).round()  # uint8 knife-edge @ CAMERA res
    down = F.interpolate(up, size=scorer_hw, mode="bilinear", align_corners=False)
    return down[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _boundary_band(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Pixels within ``radius`` of an inter-class boundary (for off-boundary flip attribution)."""

    from scipy import ndimage

    a = np.asarray(labels)
    bnd = np.zeros(a.shape, bool)
    bnd[:-1, :] |= a[:-1, :] != a[1:, :]
    bnd[1:, :] |= a[:-1, :] != a[1:, :]
    bnd[:, :-1] |= a[:, :-1] != a[:, 1:]
    bnd[:, 1:] |= a[:, :-1] != a[:, 1:]
    return ndimage.binary_dilation(bnd, iterations=int(radius))


@dataclass(frozen=True)
class RSurvivalStats:
    pre_R_disagree: float          # argmax(field) vs L* BEFORE R (representation fidelity)
    post_R_disagree: float         # argmax(R(field)) vs L* AFTER R (== field-level d_seg proxy)
    r_added_flips: float           # post_R - pre_R (the flips R itself introduces)
    r_added_off_boundary_frac: float  # fraction of R-added flips that are >radius from any edge

    def to_dict(self) -> dict[str, float]:
        return {
            "pre_R_disagree": self.pre_R_disagree,
            "post_R_disagree": self.post_R_disagree,
            "r_added_flips": self.r_added_flips,
            "r_added_off_boundary_frac": self.r_added_off_boundary_frac,
        }


def r_survival_flip_fraction(fields_hwk: np.ndarray, lstar: np.ndarray, radius: int = 2) -> RSurvivalStats:
    """The DECISIVE R-survival metric. Argmax the fields pre- and post-R vs L*; attribute the
    R-ADDED flips to boundary-band vs off-boundary (Gibbs aliasing shows as OFF-boundary flips).

    ``fields_hwk`` (h,w,K) reproduce ``lstar`` (h,w) pre-R by construction; R is downsampled to
    scorer res so L* is NN-resized to match. NO-FAKE: the disagreement is the ACTUAL popcount.
    """

    lstar = np.asarray(lstar)
    pre = np.asarray(fields_hwk).argmax(axis=-1)
    pre_dis = float(np.count_nonzero(pre != lstar)) / lstar.size
    r = apply_R_to_fields(fields_hwk)
    h2, w2, _ = r.shape
    l2 = _nn_resize_labels(lstar, h2, w2)
    post = r.argmax(axis=-1)
    post_dis = float(np.count_nonzero(post != l2)) / l2.size
    # R-added flips: pixels correct pre-R (at scorer res) but wrong post-R.
    pre2 = _nn_resize_labels(pre.astype(np.int64), h2, w2)
    added = (post != l2) & (pre2 == l2)
    n_added = int(added.sum())
    if n_added == 0:
        off_frac = 0.0
    else:
        band = _nn_resize_labels(_boundary_band(lstar, radius).astype(np.uint8), h2, w2).astype(bool)
        off_frac = float((added & ~band).sum()) / n_added
    return RSurvivalStats(pre_dis, post_dis, post_dis - pre_dis, off_frac)


def _nn_resize_labels(labels: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    a = np.asarray(labels)
    if a.shape[0] == out_h and a.shape[1] == out_w:
        return a
    ys = np.linspace(0, a.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, out_w).round().astype(np.int64)
    return a[np.ix_(ys, xs)]


def spectral_lowpass_fields(labels: np.ndarray, n_classes: int, cutoff_frac: float = 0.12) -> np.ndarray:
    """The SINE/Fourier-feature CONTROL representation: brick-wall low-pass of the one-hot class
    indicators (the function a band-limited spectral basis converges to — WITH Gibbs ringing).

    ``cutoff_frac`` is the radial FFT cutoff as a fraction of the Nyquist radius. argmax of the
    low-passed one-hots ~reproduces L* but overshoots/rings near edges — the spectral-bias wall.
    """

    a = np.asarray(labels)
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rad = np.sqrt(fy * fy + fx * fx)
    mask = (rad <= float(cutoff_frac) * 0.5).astype(np.float64)  # 0.5 = Nyquist in cycles/px
    out = np.zeros((h, w, int(n_classes)), np.float32)
    for k in range(int(n_classes)):
        ind = (a == k).astype(np.float64)
        out[..., k] = np.real(np.fft.ifft2(np.fft.fft2(ind) * mask)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Front-end fit test (curvelet vs isotropic at EQUAL budget) — least-squares, $0.
# ---------------------------------------------------------------------------
def front_end_fit_disagreement(
    coords: np.ndarray, sdf_target_hwk: np.ndarray, B: np.ndarray, h: int, w: int, ridge: float = 1e-3
) -> float:
    """Least-squares fit each class SDF target with the linear feature basis ``feats(B)``, then
    report ``argmax`` disagreement of the fitted field vs the target argmax. A LOWER number =
    the basis represents the partition better at EQUAL feature count (the front-end's value).
    """

    feats = curvelet_feats(coords, B).astype(np.float64)  # (P, F)
    tgt = np.asarray(sdf_target_hwk, np.float64).reshape(h * w, -1)  # (P, K)
    G = feats.T @ feats + ridge * np.eye(feats.shape[1])
    W = np.linalg.solve(G, feats.T @ tgt)  # (F, K)
    fit = (feats @ W).reshape(h, w, -1)
    a_fit = fit.argmax(axis=-1)
    a_tgt = tgt.reshape(h, w, -1).argmax(axis=-1)
    return float(np.count_nonzero(a_fit != a_tgt)) / a_tgt.size


# ---------------------------------------------------------------------------
# Byte-close — int8 + brotli the LEARNED weights only (bank B is free generic code).
# ---------------------------------------------------------------------------
def _int8_symmetric(a: np.ndarray) -> tuple[np.ndarray, float]:
    s = float(np.abs(a).max()) + 1e-8
    q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
    return q, (s / 127.0)


def int8_dequant_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return params after the SAME int8 symmetric round-trip the byte-close applies (the
    eval_roundtrip-for-WEIGHTS, so the verdict/inflate render the DEPLOY weights, not fp32 EMA).
    ``_B``/``B`` (free deterministic bank) pass through unquantized."""

    out: dict[str, np.ndarray] = {}
    for name, arr in params.items():
        a = np.asarray(arr, np.float32)
        if a.size == 0 or name == "B" or name.endswith("_B"):
            out[name] = a
            continue
        q, scale = _int8_symmetric(a)
        out[name] = (q.astype(np.float32) * scale).astype(np.float32)
    return out


def levelset_rgb_forward_numpy(
    params: dict[str, np.ndarray],
    feats: np.ndarray,
    code_row: np.ndarray,
    *,
    n_hidden: int,
    hidden_dim: int,
    n_classes: int,
    activation: str,
    softmax_temp: float,
    wire_w0: float,
    wire_s0: float,
    hosc_beta: float,
    hosc_omega: float,
    chroma: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """THE ONE CODEPATH (numpy fp32) — bit-mirror of the MLX ``LevelSetRGBWitness.__call__``.

    Used by BOTH the trainer's fp32 deploy-faithful verdict AND the byte-close / inflate forward
    (representation = carrier = archive: one forward that cannot diverge). Returns ``(rgb (P,3) in
    [0,255], phi (P,K) SDF)``. ``params`` are the (optionally int8-dequantized) deploy weights;
    ``feats`` is the curvelet coord feature grid (the free bank's output). mlx.nn.Linear is
    ``x @ W.T + b`` (W is (out,in)) — mirrored exactly here in float64 accumulation (argmax-stable).
    """

    p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    feats = np.asarray(feats, np.float64)
    code_row = np.asarray(code_row, np.float64)
    akw = dict(w0=wire_w0, s0=wire_s0, beta=hosc_beta, omega=hosc_omega)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _act(feats @ p["in_proj.weight"].T + p["in_proj.bias"], activation, **akw)
        film = (code_row @ p["film.weight"].T + p["film.bias"]).reshape(n_hidden, 2, hidden_dim)
        for li in range(n_hidden):
            h = _act((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li, 0]) + film[li, 1],
                     activation, **akw)
        phi = h @ p["out_sdf.weight"].T + p["out_sdf.bias"]      # (P, K)
        tex = h @ p["out_tex.weight"].T + p["out_tex.bias"]      # (P, 3)
        z = phi / float(softmax_temp)
        z = z - z.max(axis=-1, keepdims=True)
        soft = np.exp(z)
        soft = soft / soft.sum(axis=-1, keepdims=True)
        base = soft @ p["palette"]                              # (P, 3)
        rgb = (1.0 / (1.0 + np.exp(-(base + tex)))) * 255.0     # (P, 3) deploy-faithful render
        if not chroma:
            luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
    return rgb.astype(np.float32), phi.astype(np.float32)


def quantize_levelset_blob(params: dict[str, np.ndarray]) -> dict[str, Any]:
    """int8 + brotli the learned params (weights + per-frame ``code``). The curvelet bank ``B``
    is EXCLUDED (rule 118: a deterministic free table regenerated at decode from the 5 cfg
    scalars). Returns the MEASURED archive bytes (the counted rate term)."""

    import brotli

    base_chunks: list[bytes] = []
    code_chunk = b""
    n_params = 0
    for name, arr in params.items():
        a = np.asarray(arr, np.float32)
        if a.size == 0 or name.endswith("_B") or name == "B":
            continue  # the bank is free generic code
        n_params += a.size
        q, _ = _int8_symmetric(a)
        if name == "code" or name.endswith("code"):
            code_chunk = q.tobytes()
        else:
            base_chunks.append(q.tobytes())
    base_b = len(brotli.compress(b"".join(base_chunks), quality=11))
    code_b = len(brotli.compress(code_chunk, quality=11)) if code_chunk else 0
    return {
        "n_params": int(n_params),
        "base_int8_brotli_bytes": base_b,
        "code_int8_brotli_bytes": code_b,
        "total_quantized_blob_bytes": base_b + code_b,
    }


def save_levelset_npz(path: Path, params: dict[str, np.ndarray], cfg: LevelSetConfig) -> Path:
    """Persist learned weights + cfg scalars (NOT the bank — free table) to a portable .npz."""

    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"param::{k}": np.asarray(v, np.float32) for k, v in params.items() if not (k == "B" or k.endswith("_B"))}
    flat["cfg::num_pairs"] = np.asarray(cfg.num_pairs)
    flat["cfg::hidden_dim"] = np.asarray(cfg.hidden_dim)
    flat["cfg::n_hidden"] = np.asarray(cfg.n_hidden)
    flat["cfg::mod_dim"] = np.asarray(cfg.mod_dim)
    flat["cfg::n_classes"] = np.asarray(cfg.n_classes)
    flat["cfg::activation"] = np.asarray(cfg.activation)
    flat["cfg::front_end"] = np.asarray(cfg.front_end)
    flat["cfg::bank_n_scales"] = np.asarray(cfg.bank.n_scales)
    flat["cfg::bank_n_orient0"] = np.asarray(cfg.bank.n_orient0)
    flat["cfg::bank_f0"] = np.asarray(cfg.bank.f0)
    flat["cfg::bank_base"] = np.asarray(cfg.bank.base)
    flat["cfg::bank_n_iso"] = np.asarray(cfg.bank.n_iso)
    np.savez(path, **flat)
    return path


# ---------------------------------------------------------------------------
# MLX module (fast train path) — import-guarded so the numpy reference + feasibility
# run with ZERO MLX dependency (the $0 CPU build/feasibility surface).
# ---------------------------------------------------------------------------
def build_levelset_witness_module(cfg: LevelSetConfig):
    """The MLX coord-INR level-set witness: curvelet/iso feats -> WIRE/HOSC hidden -> K SDF.

    Mirrors the RGB witness's FiLM-per-(pair,frame) structure (``code`` has ``2*num_pairs`` rows
    = f0,f1 interleaved) but the head outputs K=5 SDF fields (LINEAR, no sigmoid). The frozen
    curvelet bank ``_B`` is held as a non-trainable constant (free table). The future
    ``train_levelset_witness_realized_through_R_mlx.py`` renders ``argmax_k phi_k`` -> a class
    palette RGB -> R -> frozen CPU-torch SegNet for the d_seg VERDICT.
    """

    import mlx.core as mx
    import mlx.nn as nn

    B_np = cfg.build_B()
    in_feat = cfg.in_feat()

    class LevelSetWitnessMLX(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg_activation = cfg.activation
            self.wire_w0 = float(cfg.wire_w0)
            self.wire_s0 = float(cfg.wire_s0)
            self.hosc_beta = float(cfg.hosc_beta)
            self.hosc_omega = float(cfg.hosc_omega)
            self.n_hidden = cfg.n_hidden
            self.hidden_dim = cfg.hidden_dim
            self._B = mx.array(B_np)  # free deterministic curvelet bank (not trained)
            self.code = mx.zeros((cfg.num_pairs * 2, cfg.mod_dim))
            self.in_proj = nn.Linear(in_feat, cfg.hidden_dim)
            self.film = nn.Linear(cfg.mod_dim, 2 * cfg.hidden_dim * cfg.n_hidden)
            self.hidden = [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_hidden)]
            self.out = nn.Linear(cfg.hidden_dim, cfg.n_classes)

        def _act(self, u):
            if self.cfg_activation == "wire":
                return mx.cos(self.wire_w0 * u) * mx.exp(-((self.wire_s0 * u) ** 2))
            if self.cfg_activation == "hosc":
                return mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))
            return nn.relu(u)

        def feats(self, coords):
            proj = (2.0 * mx.pi) * (coords @ self._B)
            return mx.concatenate([mx.sin(proj), mx.cos(proj)], axis=-1)

        def __call__(self, coord_feats, code_idx):
            h = self._act(self.in_proj(coord_feats))
            film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                h = self._act(layer(h) * (1.0 + film[li, 0]) + film[li, 1])
            return self.out(h)  # (P, n_classes) SDF — LINEAR head

    return LevelSetWitnessMLX()


__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "CurveletBankConfig",
    "LevelSetConfig",
    "RSurvivalStats",
    "apply_R_to_fields",
    "boundary_length_penalty",
    "build_coords",
    "build_levelset_witness_module",
    "curvelet_directional_B",
    "curvelet_feats",
    "eikonal_penalty",
    "front_end_fit_disagreement",
    "isotropic_fourier_B",
    "levelset_argmax",
    "numpy_levelset_forward",
    "quantize_levelset_blob",
    "r_survival_flip_fraction",
    "save_levelset_npz",
    "signed_distance_fields",
    "spectral_lowpass_fields",
    "stem_nyquist_max_freq_cycles_per_unit",
    "wire_activation",
    "hosc_activation",
]
