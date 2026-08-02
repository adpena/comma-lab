# SPDX-License-Identifier: MIT
"""ddm_mp1 -- the LSB-misplacement instrument, and its join against the margin field.

WHAT THIS MEASURES
------------------
The live receiver realizes a 384x512 render ``r`` as camera-resolution uint8 via
``cam = clip(rint(U(r)))`` (``ddm_tr1_runtime.render_frame1_camera_uint8``), and the
frozen SegNet then reads ``D(cam)`` where ``D`` is the upstream bilinear resize
``(874,1164) -> (384,512)`` (``upstream/modules.py:109``).  ``U`` is the receiver's
PyTorch-style bicubic ``align_corners=False, A=-0.75``
(``ddm_tr1_runtime.bicubic_up_to_camera_float``).

The REALIZATION ERROR is what the receiver fails to realize OF ITS OWN INTENT::

    e_real = D(clip(rint(U(r)))) - r

decomposed exactly into two additive parts::

    e_resample = D(U(r))              - r     (U then D is not the identity)
    e_quant    = D(clip(rint(U(r))))  - D(U(r))   (uint8 rounding, D-averaged)
    e_real     = e_resample + e_quant

For the IDEAL render ``r = y* = D(X_gt)`` this reproduces exactly the quantity
``ddm_cg2`` swept at n600 (mean 0.19815 / rms 0.34517 / max 29.844 LSB,
``.omx/research/ddm_cg2_realization_n600_20260802.json``) -- that is this
instrument's cross-instrument positive control.  cg2 landed those 8 scalars and NO
code (task #898), so the per-pixel field was never persisted and the ideal-vs-live
question was never asked.  This module rebuilds the sweep, PERSISTS the per-pixel
field, and adds the two legs cg2 could not:

  * ``--render live``  -- the SAME quantity on the actual v4d receiver's render,
    which answers whether cg2's ideal-render number transfers at all;
  * ``--mode join``    -- the per-pixel misplacement joined against the cached
    frozen-SegNet margin field, which is what decides whether misplacement can
    reach an argmax flip.

AUTHORITY
---------
``[macOS-CPU advisory]`` SCORER-FREE for the sweep and the join (the margin field is
cached in ``gt_n600.npz`` and was produced by the frozen CPU-torch SegNet).  The
optional ``--mode control-scorer`` runs a HANDFUL of single-frame SegNet forwards
purely to validate that this module's ``D`` is the same operator the cached
``lstars`` came from; it is instrument validation, never a measurement.
``score_claim=false``, ``promotion_eligible=false``.

RESUMABILITY / OVERLAP REFUSAL
------------------------------
The sweep is chunked (``--start``/``--end``) and each chunk writes its own npz, so a
harness SIGURG (exit 144, the documented killer of cg2's single long job) costs at
most one chunk.  ``--mode aggregate`` REFUSES overlapping chunk ranges outright
rather than summing them, because summing overlapping ranges silently double-counts
frames and corrupts every mean -- the exact corruption cg2 caught late.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEG_H, SEG_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
N_CHAN = 3
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")  # comma10k order (CLAUDE.md)

DEFAULT_GT = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_V4D_ZIP = Path(
    "/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_refine_celldrop50_archive.zip"
)
DEFAULT_RUN_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mp1_20260802")

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")

# --- The registered first-order gain law (segnet_head_rank4_linear_flipdist_v1 +
# --- tools/adversarial_evasion_fisher_null_probe.py).  ||grad m_p|| is in units of
# --- margin-logits per unit input-L2 over the (384,512,3) 0-255 frame.
ANCHOR_MARGIN = 0.516
ANCHOR_FLIP_L2 = 8.8
G_GAIN = (ANCHOR_MARGIN / ANCHOR_FLIP_L2) * math.cosh(ANCHOR_MARGIN / 2.0)  # ~0.0606


#: The per-frame scalar series every chunk carries (the aggregate's whole input).
_CHUNK_SCALAR_KEYS = (
    "sum_abs",
    "sum_sq",
    "max_abs",
    "cnt_gt_half",
    "cnt_gt_one",
    "resample_sum_sq",
    "quant_sum_sq",
    "clipped_values",
    "direct_payload_residual_max",
)


class Mp1Error(RuntimeError):
    """Instrument refusal -- always fail closed, never degrade silently."""


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise Mp1Error(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Zero-copy frame access into the STORED (uncompressed) npz.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredMember:
    """A single uncompressed .npy member of an npz, mapped without inflating it."""

    path: Path
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype

    def frame(self, index: int) -> np.ndarray:
        """Read row ``index`` of the member.  O(row), never O(member)."""
        if not 0 <= index < self.shape[0]:
            raise Mp1Error(f"frame {index} out of range for {self.shape}")
        row_shape = self.shape[1:]
        count = int(np.prod(row_shape)) if row_shape else 1
        itemsize = self.dtype.itemsize
        with open(self.path, "rb") as handle:
            handle.seek(self.offset + index * count * itemsize)
            raw = handle.read(count * itemsize)
        if len(raw) != count * itemsize:
            raise Mp1Error(f"short read for frame {index}")
        return np.frombuffer(raw, dtype=self.dtype).reshape(row_shape)


def open_stored_npz(path: Path) -> dict[str, StoredMember]:
    """Map every STORED member of an npz.  Refuses compressed members.

    ``np.load(npz)[key]`` materialises the ENTIRE member (``gt_f1`` alone is
    1.83 GB), which is how a per-frame sweep turns into an OOM.  This reads one
    frame at a time instead.
    """

    members: dict[str, StoredMember] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise Mp1Error(
                    f"{info.filename} is compressed (type {info.compress_type}); "
                    "per-frame mapping requires a STORED npz"
                )
            with archive.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
                if fortran:
                    raise Mp1Error(f"{info.filename} is Fortran-ordered; unsupported")
                data_offset = handle._orig_compress_start + handle.tell()
            members[info.filename[: -len(".npy")]] = StoredMember(
                path=path, offset=data_offset, shape=tuple(shape), dtype=dtype
            )
    if not members:
        raise Mp1Error(f"no .npy members found in {path}")
    # The offsets above come from private numpy/zipfile internals.  Validate them
    # STRUCTURALLY so a library change fails closed instead of reading garbage:
    # every member must lie inside the file and no two members may overlap.
    size = path.stat().st_size
    spans = sorted(
        (m.offset, m.offset + int(np.prod(m.shape)) * m.dtype.itemsize, name)
        for name, m in members.items()
    )
    for lo, hi, name in spans:
        if lo < 0 or hi > size:
            raise Mp1Error(f"member {name} span [{lo},{hi}) outside {path} ({size} bytes)")
    for (_, prev_hi, prev), (lo, _, name) in pairwise(spans):
        if lo < prev_hi:
            raise Mp1Error(f"members {prev} and {name} overlap; offset decoding is wrong")
    return members


# ---------------------------------------------------------------------------
# The two operators.  D is the scorer's own resize; U is the receiver's own bicubic.
# ---------------------------------------------------------------------------


def scorer_downsample(frame_hwc: np.ndarray) -> np.ndarray:
    """``D``: the EXACT upstream SegNet resize, (874,1164,3) -> (384,512,3).

    ``upstream/modules.py:109`` is ``F.interpolate(x, size=(384,512), mode='bilinear')``
    on a float32 NCHW tensor produced by ``uint8 -> .float()`` (0-255 scale).  We call
    the same torch op on the same dtype rather than re-deriving a separable matmul, so
    there is no second implementation to be wrong.
    """

    import torch

    array = np.ascontiguousarray(frame_hwc, dtype=np.float32)
    if array.shape != (CAMERA_H, CAMERA_W, N_CHAN):
        raise Mp1Error(f"D input must be (874,1164,3); got {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W) float32
    with torch.inference_mode():
        out = torch.nn.functional.interpolate(tensor, size=(SEG_H, SEG_W), mode="bilinear")
    return np.ascontiguousarray(out[0].permute(1, 2, 0).numpy(), dtype=np.float32)


def receiver_upsample(render_hwc: np.ndarray) -> np.ndarray:
    """``U``: the receiver's own bicubic, imported (never re-implemented)."""

    from tac.optimization.ddm_tr1_runtime import bicubic_up_to_camera_float

    return bicubic_up_to_camera_float(np.ascontiguousarray(render_hwc, dtype=np.float32))


@dataclass(frozen=True)
class RealizationTerms:
    """The exact additive decomposition of one frame's realization error."""

    e_real: np.ndarray  # (384,512,3) float64  D(clip(rint(U(r)))) - r
    e_resample: np.ndarray  # (384,512,3) float64  D(U(r)) - r
    e_quant: np.ndarray  # (384,512,3) float64  D(clip(rint(U(r)))) - D(U(r))
    clipped_values: int  # camera values where U(r) left [0,255] before rint


def realization_terms(render_384: np.ndarray) -> RealizationTerms:
    """Push one 384x512x3 render through the live realize-then-score round trip."""

    render = np.ascontiguousarray(render_384, dtype=np.float32)
    if render.shape != (SEG_H, SEG_W, N_CHAN):
        raise Mp1Error(f"render must be (384,512,3); got {render.shape}")
    up = receiver_upsample(render)
    clipped = int(np.count_nonzero((up < 0.0) | (up > 255.0)))
    cam = np.clip(np.rint(up), 0, 255).astype(np.uint8)
    back_float = scorer_downsample(up).astype(np.float64)  # D(U(r)) -- no quantisation
    back_uint8 = scorer_downsample(cam.astype(np.float32)).astype(np.float64)
    r64 = render.astype(np.float64)
    return RealizationTerms(
        e_real=back_uint8 - r64,
        e_resample=back_float - r64,
        e_quant=back_uint8 - back_float,
        clipped_values=clipped,
    )


# ---------------------------------------------------------------------------
# Render sources.
# ---------------------------------------------------------------------------


class IdealRender:
    """``r = y* = D(X_gt)`` -- the flawless render.  Reproduces cg2's leg exactly."""

    name = "ideal"

    def __init__(self, gt: dict[str, StoredMember]) -> None:
        self._gt_f1 = gt["gt_f1"]

    def render(self, index: int) -> np.ndarray:
        return scorer_downsample(self._gt_f1.frame(index))


class LiveV4dRender:
    """``r`` = the ACTUAL v4d receiver's pre-realization 384 render.

    v4d is our live own-vehicle frontier (360,238 B, 6 members, S = 0.9639878).
    Its render is a 24x32 uint4 token grid expanded by the counted renderer, so it is
    far smoother than ``y*`` -- which is precisely why the ideal-leg number cannot be
    assumed to transfer.
    """

    name = "live_v4d"

    #: The frame0 policy v4d's receiver refuses to run without
    #: (``experiments/inflate_runner_v4d.py`` FRAME0_POLICY).
    FRAME0_POLICY = "warp_two_plane_static_photo_beta_v4d"

    def __init__(self, archive_zip: Path, workdir: Path) -> None:
        from experiments.ddm_r7_token_coder import decode_token_codes
        from tac.optimization.ddm_tr1_runtime import (
            _encode_tokens,
            build_packet,
            parse_packet,
            render_frame1_float,
        )

        self._render_frame1_float = render_frame1_float
        members: dict[str, bytes] = {}
        with zipfile.ZipFile(archive_zip) as archive:
            for name in archive.namelist():
                members[name] = archive.read(name)
        manifest = json.loads(members["manifest.json"])
        if manifest.get("frame0_policy") != self.FRAME0_POLICY:
            raise Mp1Error(
                f"archive frame0_policy={manifest.get('frame0_policy')!r}; "
                f"expected {self.FRAME0_POLICY!r} -- this is not v4d"
            )
        # Byte-for-byte the packet assembly in inflate_runner_v4d.Decoder.__init__
        # (read, not paraphrased).  pose_warp/pose_stub play no part in frame1's
        # render, so the pose receiver is deliberately not imported.
        codes = decode_token_codes(members["state/tokens.dr7t"])
        packet_bytes = build_packet(
            manifest["tr1_metadata"],
            {
                "tokens": _encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8)),
                "lotto_renderer": members["state/renderer.sec"],
                "selector": members["state/selector.sec"],
                "pose_stub": members["state/pose_stub.sec"],
            },
        )
        self.packet = parse_packet(packet_bytes)
        self.n_pairs = int(self.packet.selector["num_pairs"])
        self.archive_sha256 = hashlib.sha256(archive_zip.read_bytes()).hexdigest()
        self.archive_bytes = archive_zip.stat().st_size

    def render(self, index: int) -> np.ndarray:
        return self._render_frame1_float(self.packet, index)

    def control_receiver_parity(self, index: int) -> int:
        """POSITIVE CONTROL: this module's realize step IS the receiver's.

        Returns the number of camera values where ``clip(rint(U(r)))`` computed here
        differs from ``ddm_tr1_runtime.render_frame1_camera_uint8`` -- the function the
        deployed v4d runtime actually calls.  Must be 0.
        """

        from tac.optimization.ddm_tr1_runtime import (
            render_frame1_camera_uint8,
        )

        mine = np.clip(np.rint(receiver_upsample(self.render(index))), 0, 255).astype(np.uint8)
        theirs = render_frame1_camera_uint8(self.packet, index)
        return int(np.count_nonzero(mine != theirs))


# ---------------------------------------------------------------------------
# Sweep (chunked, resumable).
# ---------------------------------------------------------------------------


def _per_pixel_magnitudes(err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate the 3 per-channel errors at a pixel into two scalars.

    ``chan_l2`` = sqrt(sum_c e_c^2) is the PRINCIPLED one for the margin join: the
    first-order margin perturbation is ``dm = <grad m, e>``, so under an unaligned
    (isotropic) gradient the contribution of pixel p enters through ``||e_p||_2``.
    ``chan_maxabs`` is reported alongside because it is what a per-channel threshold
    such as cg2's ">0.5 LSB" fraction actually keys on.
    """

    sq = err.astype(np.float64) ** 2
    return np.sqrt(sq.sum(axis=2)), np.abs(err).max(axis=2)


def run_sweep(
    *,
    gt_path: Path,
    render_kind: str,
    start: int,
    end: int,
    out_dir: Path,
    v4d_zip: Path,
    persist_field: bool,
) -> Path:
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = open_stored_npz(gt_path)
    n_total = gt["gt_f1"].shape[0]
    if not (0 <= start < end <= n_total):
        raise Mp1Error(f"chunk [{start},{end}) invalid for {n_total} frames")

    stem = f"chunk_{render_kind}_{start:04d}_{end:04d}"
    out_path = out_dir / f"{stem}.npz"
    if out_path.exists():
        return out_path  # RESUME: a completed chunk is never recomputed

    source: IdealRender | LiveV4dRender
    if render_kind == "ideal":
        source = IdealRender(gt)
    elif render_kind == "live":
        source = LiveV4dRender(v4d_zip, out_dir)
    else:
        raise Mp1Error(f"unknown render kind {render_kind!r}")

    n = end - start
    scalars = {key: np.zeros(n, dtype=np.float64) for key in _CHUNK_SCALAR_KEYS}
    field_l2 = np.zeros((n, SEG_H, SEG_W), dtype=np.float16) if persist_field else None
    field_max = np.zeros((n, SEG_H, SEG_W), dtype=np.float16) if persist_field else None

    for offset in range(n):
        index = start + offset
        render = source.render(index)
        terms = realization_terms(render)
        err = terms.e_real
        abs_err = np.abs(err)
        scalars["sum_abs"][offset] = abs_err.sum()
        scalars["sum_sq"][offset] = float((err**2).sum())
        scalars["max_abs"][offset] = abs_err.max()
        scalars["cnt_gt_half"][offset] = float(np.count_nonzero(abs_err > 0.5))
        scalars["cnt_gt_one"][offset] = float(np.count_nonzero(abs_err > 1.0))
        scalars["resample_sum_sq"][offset] = float((terms.e_resample**2).sum())
        scalars["quant_sum_sq"][offset] = float((terms.e_quant**2).sum())
        scalars["clipped_values"][offset] = float(terms.clipped_values)
        # POSITIVE CONTROL (tautological, as cg2 flagged): a camera-res uint8 payload
        # realises its own downsample exactly.  Kept because a NONZERO value here would
        # mean this module's D disagrees with itself.
        gt_down = scorer_downsample(gt["gt_f1"].frame(index)).astype(np.float64)
        scalars["direct_payload_residual_max"][offset] = float(np.abs(gt_down - gt_down).max())
        if persist_field:
            l2, mx = _per_pixel_magnitudes(err)
            field_l2[offset] = l2.astype(np.float16)  # type: ignore[index]
            field_max[offset] = mx.astype(np.float16)  # type: ignore[index]

    payload: dict[str, Any] = {
        "start": np.int64(start),
        "end": np.int64(end),
        "render_kind": np.str_(render_kind),
        "values_per_frame": np.int64(SEG_H * SEG_W * N_CHAN),
        **scalars,
    }
    if persist_field:
        payload["field_l2_f16"] = field_l2
        payload["field_maxabs_f16"] = field_max
    tmp = out_dir / f".{stem}.partial"
    with open(tmp, "wb") as handle:  # np.savez would re-suffix a Path; write atomically
        np.savez(handle, **payload)
    tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Aggregate (overlap-refusing).
# ---------------------------------------------------------------------------


def aggregate(out_dir: Path, render_kind: str, n_expected: int) -> dict[str, Any]:
    """Combine chunks.  REFUSES overlaps; asserts contiguous cover of [0, n)."""

    chunks = sorted(out_dir.glob(f"chunk_{render_kind}_*.npz"))
    if not chunks:
        raise Mp1Error(f"no chunks for render_kind={render_kind!r} in {out_dir}")
    loaded = []
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            per_frame = {k: data[k].copy() for k in _CHUNK_SCALAR_KEYS if k in data.files}
            loaded.append((int(data["start"]), int(data["end"]), path, per_frame))
    loaded.sort(key=lambda row: row[0])

    cursor = 0
    for start, end, path, _ in loaded:
        if start < cursor:
            raise Mp1Error(
                f"OVERLAPPING chunk {path.name} [{start},{end}) under cursor {cursor}: "
                "refusing to aggregate (summing overlaps silently double-counts frames)"
            )
        if start > cursor:
            raise Mp1Error(f"GAP in coverage: [{cursor},{start}) missing before {path.name}")
        cursor = end
    if cursor != n_expected:
        raise Mp1Error(f"coverage ends at {cursor}, expected {n_expected}")

    cat = {key: np.concatenate([row[3][key] for row in loaded]) for key in _CHUNK_SCALAR_KEYS}
    values_per_frame = SEG_H * SEG_W * N_CHAN
    denom = float(n_expected * values_per_frame)
    return {
        "render_kind": render_kind,
        "frames_covered": n_expected,
        "contiguous_0_to_n": True,
        "n_chunks": len(loaded),
        "denominator_values": int(denom),
        "mean_abs_err_LSB": float(cat["sum_abs"].sum() / denom),
        "rms_err_LSB": float(math.sqrt(cat["sum_sq"].sum() / denom)),
        "max_abs_err_LSB": float(cat["max_abs"].max()),
        "frac_gt_half_LSB": float(cat["cnt_gt_half"].sum() / denom),
        "frac_gt_one_LSB": float(cat["cnt_gt_one"].sum() / denom),
        "rms_resample_LSB": float(math.sqrt(cat["resample_sum_sq"].sum() / denom)),
        "rms_quant_LSB": float(math.sqrt(cat["quant_sum_sq"].sum() / denom)),
        "clipped_camera_values_total": int(cat["clipped_values"].sum()),
        "direct_camera_payload_residual_max": float(cat["direct_payload_residual_max"].max()),
    }


# ---------------------------------------------------------------------------
# The join: misplacement vs the cached frozen-SegNet margin field.
# ---------------------------------------------------------------------------

MARGIN_EDGES = np.array(
    [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf], dtype=np.float64
)


def _flip_probability(margin: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """First-order argmax-flip probability under the registered gain law.

    A pixel flips only when the margin perturbation is NEGATIVE and larger than the
    margin: ``P = P(dm < -m) = 0.5*erfc(m / (sigma*sqrt(2)))``.  It is ONE-sided --
    the two-sided form would double-count, since a positive ``dm`` only deepens the
    winner's lead.
    """

    from scipy.special import erfc

    safe = np.maximum(sigma, 1e-12)
    return 0.5 * erfc(margin / (safe * math.sqrt(2.0)))


def run_join(
    *,
    gt_path: Path,
    out_dir: Path,
    render_kind: str,
    n_expected: int,
    seed: int,
) -> dict[str, Any]:
    """Join the persisted per-pixel misplacement against margins + lstars.

    MODELLING ASSUMPTION, stated because it is load-bearing: SegNet's receptive field
    is large, so the margin at pixel p does not depend on the input at p alone.  The
    join therefore tests SPATIAL CO-LOCATION of misplacement with small margins.  That
    is the right test because realization error is spatially structured (it tracks
    image detail) and the receptive field is centred at p -- but it is a local proxy,
    not an exact per-pixel sensitivity.
    """

    gt = open_stored_npz(gt_path)
    margins_m = gt["margins"]
    lstars_m = gt["lstars"]
    n_bins = len(MARGIN_EDGES) - 1

    bin_pixels = np.zeros(n_bins, dtype=np.int64)
    bin_sum_q = np.zeros(n_bins, dtype=np.float64)
    bin_sum_q2 = np.zeros(n_bins, dtype=np.float64)
    class_pixels = np.zeros(len(CLASSES), dtype=np.int64)
    class_sum_q = np.zeros(len(CLASSES), dtype=np.float64)
    total_pixels = 0
    total_q = 0.0
    total_q2 = 0.0
    reach_measured = 0.0
    reach_null = 0.0
    reach_uniform = 0.0
    thresh_1sigma = 0
    reach_measured_class = np.zeros(len(CLASSES), dtype=np.float64)
    rng = np.random.default_rng(seed)

    chunks = sorted(out_dir.glob(f"chunk_{render_kind}_*.npz"))
    if not chunks:
        raise Mp1Error(f"no chunks for render_kind={render_kind!r} in {out_dir}")
    cursor = 0
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            start, end = int(data["start"]), int(data["end"])
            if start != cursor:
                raise Mp1Error(f"join requires contiguous chunks; {path.name} breaks at {cursor}")
            if "field_l2_f16" not in data.files:
                raise Mp1Error(f"{path.name} has no persisted per-pixel field (task #898 debt)")
            field = data["field_l2_f16"]
            for offset in range(end - start):
                index = start + offset
                q = field[offset].astype(np.float64)  # (384,512) channel-L2 misplacement
                margin = margins_m.frame(index).astype(np.float64)
                label = lstars_m.frame(index)
                if label.min() < 0 or label.max() >= len(CLASSES):
                    raise Mp1Error(
                        f"frame {index} has labels outside [0,{len(CLASSES) - 1}]; "
                        "the class histogram would silently mis-bin"
                    )
                idx = np.digitize(margin, MARGIN_EDGES) - 1
                np.clip(idx, 0, n_bins - 1, out=idx)
                bin_pixels += np.bincount(idx.ravel(), minlength=n_bins)
                bin_sum_q += np.bincount(idx.ravel(), weights=q.ravel(), minlength=n_bins)
                bin_sum_q2 += np.bincount(idx.ravel(), weights=(q**2).ravel(), minlength=n_bins)
                class_pixels += np.bincount(label.ravel(), minlength=len(CLASSES))
                class_sum_q += np.bincount(label.ravel(), weights=q.ravel(),
                                           minlength=len(CLASSES))
                total_pixels += q.size
                total_q += float(q.sum())
                total_q2 += float((q**2).sum())

                # First-order reach under the registered gain law.  rho_p is the
                # per-ELEMENT rms of the misplacement at p, i.e. ||e_p||_2 / sqrt(3).
                rho = q / math.sqrt(N_CHAN)
                grad = G_GAIN / np.cosh(margin / 2.0)
                sigma = rho * grad
                pflip = _flip_probability(margin, sigma)
                reach_measured += float(pflip.sum())
                reach_measured_class += np.bincount(
                    label.ravel(), weights=pflip.ravel(), minlength=len(CLASSES)
                )
                thresh_1sigma += int(np.count_nonzero(margin < sigma))
                # NULL 1 -- SHUFFLED: identical misplacement magnitudes, permuted across
                # the frame so any margin correlation is destroyed while the marginal
                # distribution is preserved exactly.  This is the control that separates
                # "misplacement is enriched at the boundary" from "there is simply a lot
                # of misplacement".
                shuffled = rng.permutation(rho.ravel()).reshape(rho.shape)
                reach_null += float(_flip_probability(margin, shuffled * grad).sum())
                # NULL 2 -- UNIFORM rho: every pixel gets the frame's rms misplacement.
                # This is exactly the margin-BLIND model of the registered evasion probe,
                # so the two numbers are directly comparable.
                rho_bar = float(np.sqrt((rho**2).mean()))
                reach_uniform += float(_flip_probability(margin, rho_bar * grad).sum())
            cursor = end
    if cursor != n_expected:
        raise Mp1Error(f"join coverage ends at {cursor}, expected {n_expected}")

    mean_q = total_q / total_pixels
    safe_pixels = np.maximum(bin_pixels, 1)
    bin_mean = np.where(bin_pixels > 0, bin_sum_q / safe_pixels, np.nan)
    # The rms alongside the mean says whether an enrichment is a shift of the whole
    # bin or a few heavy-tailed pixels dragging the mean.
    bin_rms = np.where(bin_pixels > 0, np.sqrt(bin_sum_q2 / safe_pixels), np.nan)
    return {
        "render_kind": render_kind,
        "frames": n_expected,
        "pixels": int(total_pixels),
        "channel_aggregation": "chan_l2 = sqrt(sum_c e_c^2) (first-order margin coupling)",
        "mean_misplacement_chanL2_LSB": float(mean_q),
        "rms_misplacement_chanL2_LSB": float(math.sqrt(total_q2 / total_pixels)),
        "margin_bins": [
            {
                "lo": float(MARGIN_EDGES[i]),
                "hi": float(MARGIN_EDGES[i + 1]),
                "pixels": int(bin_pixels[i]),
                "area_frac": float(bin_pixels[i] / total_pixels),
                "mean_misplacement_LSB": float(bin_mean[i]),
                "rms_misplacement_LSB": float(bin_rms[i]),
                "enrichment_vs_global": float(bin_mean[i] / mean_q) if mean_q > 0 else float("nan"),
            }
            for i in range(n_bins)
        ],
        "per_class": [
            {
                "class": CLASSES[c],
                "pixels": int(class_pixels[c]),
                "mean_misplacement_LSB": float(class_sum_q[c] / max(class_pixels[c], 1)),
                "reach_d_seg_share": float(
                    reach_measured_class[c] / reach_measured if reach_measured > 0 else 0.0
                ),
            }
            for c in range(len(CLASSES))
        ],
        "first_order_reach": {
            "law": "P(flip) = Phi(-m_p/sigma_p), sigma_p = (||e_p||_2/sqrt(3)) * "
            f"G/cosh(m_p/2); G = {G_GAIN:.6f}",
            "law_source": "segnet_head_rank4_linear_flipdist_v1 + "
            "tools/adversarial_evasion_fisher_null_probe.py",
            "law_status": "DERIVED first-order, single-anchor gain; NOT scorer-measured",
            "d_seg_measured_field": float(reach_measured / total_pixels),
            "d_seg_null_shuffled": float(reach_null / total_pixels),
            "d_seg_null_uniform_rho": float(reach_uniform / total_pixels),
            "enrichment_vs_shuffled": (
                float(reach_measured / reach_null) if reach_null > 0 else float("nan")
            ),
            "enrichment_vs_uniform": (
                float(reach_measured / reach_uniform) if reach_uniform > 0 else float("nan")
            ),
            "d_seg_threshold_1sigma": float(thresh_1sigma / total_pixels),
            "S_units_measured": float(100.0 * reach_measured / total_pixels),
            "bytes_equivalent_at_W": float(reach_measured * 1.27310821533),
            "W_B_per_flip": 1.27310821533,
        },
    }


def reach_curve(gt_path: Path, n_expected: int, rhos: tuple[float, ...]) -> dict[str, Any]:
    """d_seg vs a UNIFORM per-element misplacement rho, from the cached margins alone.

    This is the dose-response the dither moves along: it converts "the realization
    error drops from rho_now to rho_floor" into a d_seg delta without needing the
    per-pixel field, and it is directly comparable to the registered evasion probe's
    margin-blind table.
    """

    gt = open_stored_npz(gt_path)
    margins_m = gt["margins"]
    totals = np.zeros(len(rhos), dtype=np.float64)
    pixels = 0
    for index in range(n_expected):
        margin = margins_m.frame(index).astype(np.float64)
        grad = G_GAIN / np.cosh(margin / 2.0)
        for slot, rho in enumerate(rhos):
            totals[slot] += float(_flip_probability(margin, rho * grad).sum())
        pixels += margin.size
    return {
        "scope": f"cached frozen-SegNet margins, {n_expected} frames, {pixels} pixels",
        "law": f"P(flip) = Phi(-m/sigma), sigma = rho * G/cosh(m/2); G = {G_GAIN:.6f}",
        "law_status": "DERIVED first-order, single-anchor gain; NOT scorer-measured",
        "curve": [
            {
                "rho_LSB": float(rho),
                "d_seg": float(totals[slot] / pixels),
                "S_units": float(100.0 * totals[slot] / pixels),
            }
            for slot, rho in enumerate(rhos)
        ],
    }


# ---------------------------------------------------------------------------
# How much of the misplacement is RECOVERABLE: the uint8 camera lattice.
# ---------------------------------------------------------------------------


def _separable_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Impulse-probe ``D`` for its 1-D row and column kernels.

    ``F.interpolate(mode='bilinear')`` is a tensor-product interpolation, so ``D`` is
    ``Row @ frame @ Col.T``.  Both kernels are MEASURED by pushing impulse bases
    through the same torch op, never re-derived from the resize formula.
    """

    import torch

    with torch.inference_mode():
        row_probe = torch.eye(CAMERA_H, dtype=torch.float32).reshape(CAMERA_H, 1, CAMERA_H, 1)
        row = torch.nn.functional.interpolate(row_probe, size=(SEG_H, 1), mode="bilinear")
        col_probe = torch.eye(CAMERA_W, dtype=torch.float32).reshape(CAMERA_W, 1, 1, CAMERA_W)
        col = torch.nn.functional.interpolate(col_probe, size=(1, SEG_W), mode="bilinear")
    return (
        row.reshape(CAMERA_H, SEG_H).numpy().T.copy(),  # (384, 874)
        col.reshape(CAMERA_W, SEG_W).numpy().T.copy(),  # (512, 1164)
    )


def _exact_min_residual(weights: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """EXACT ``min_{c in uint8^k} |sum_i w_i c_i - t|`` for every target.

    Splits the taps into two halves, enumerates each half's 256^(k/2) reachable sums,
    sorts one, and binary-searches it -- so the answer is the true minimum over the
    whole 256^k lattice, not a greedy approximation.
    """

    levels = np.arange(256, dtype=np.float64)
    half = (len(weights) + 1) // 2
    left = np.zeros(1, dtype=np.float64)
    for w in weights[:half]:
        left = (left[:, None] + w * levels[None, :]).ravel()
    right = np.zeros(1, dtype=np.float64)
    for w in weights[half:]:
        right = (right[:, None] + w * levels[None, :]).ravel()
    right.sort()
    best = np.empty(targets.shape, dtype=np.float64)
    for index, target in enumerate(np.asarray(targets, dtype=np.float64).ravel()):
        need = target - left  # vectorised over the whole left half
        pos = np.searchsorted(right, need)
        local = np.inf
        for candidate in (pos - 1, pos):
            ok = (candidate >= 0) & (candidate < right.size)
            idx = np.clip(candidate, 0, right.size - 1)
            diff = np.where(ok, np.abs(right[idx] + left - target), np.inf)
            local = min(local, float(diff.min()))
        best.ravel()[index] = local
    return best


def lattice_report(
    *, gt_path: Path, n_weight_sets: int, n_targets: int, seed: int
) -> dict[str, Any]:
    """Measure the camera lattice: block structure, disjointness, achievable residual.

    This is what sizes the RECOVERABLE part of the misplacement.  ``e_resample`` is
    removable in principle iff ``D`` is surjective; what remains is the granularity of
    the uint8 camera lattice, which this measures EXACTLY on sampled blocks.
    """

    import torch

    row, col = _separable_kernels()
    gt = open_stored_npz(gt_path)
    frame = gt["gt_f1"].frame(0).astype(np.float32)
    rows_applied = np.tensordot(row, frame, axes=([1], [0]))  # (384, 1164, 3)
    separable = np.tensordot(rows_applied, col, axes=([1], [1])).transpose(0, 2, 1)
    torch_ref = scorer_downsample(frame)
    sep_err = float(np.abs(separable - torch_ref).max())

    row_taps = (row != 0.0).sum(axis=1)
    col_taps = (col != 0.0).sum(axis=1)
    row_owner = (row != 0.0).sum(axis=0)  # how many outputs each camera row feeds
    col_owner = (col != 0.0).sum(axis=0)
    blind_rows = int(np.count_nonzero(row_owner == 0))
    blind_cols = int(np.count_nonzero(col_owner == 0))

    rng = np.random.default_rng(seed)
    out_rows = rng.integers(0, SEG_H, size=n_weight_sets)
    out_cols = rng.integers(0, SEG_W, size=n_weight_sets)
    targets = rng.uniform(0.0, 255.0, size=n_targets)
    residuals = []
    for r_idx, c_idx in zip(out_rows, out_cols, strict=True):
        rw = row[r_idx][row[r_idx] != 0.0]
        cw = col[c_idx][col[c_idx] != 0.0]
        weights = np.outer(rw, cw).ravel().astype(np.float64)
        residuals.append(_exact_min_residual(weights, targets))
    residuals_arr = np.concatenate(residuals)

    quant_only = np.abs(
        np.rint(targets) - targets
    )  # what plain rint leaves at the same targets
    return {
        "control_separability_maxabs_vs_torch_2d": sep_err,
        "control_note": "D = Row @ frame @ Col.T must reproduce the torch 2-D bilinear",
        "row_kernel_shape": list(row.shape),
        "col_kernel_shape": list(col.shape),
        "denominators": {"output_rows": SEG_H, "output_cols": SEG_W,
                         "camera_rows": CAMERA_H, "camera_cols": CAMERA_W},
        "taps_per_output_row": {"min": int(row_taps.min()), "max": int(row_taps.max())},
        "taps_per_output_col": {"min": int(col_taps.min()), "max": int(col_taps.max())},
        "camera_row_feeds_n_outputs": {"min": int(row_owner.min()), "max": int(row_owner.max())},
        "camera_col_feeds_n_outputs": {"min": int(col_owner.min()), "max": int(col_owner.max())},
        "blocks_disjoint": bool(row_owner.max() <= 1 and col_owner.max() <= 1),
        "blind_camera_rows": blind_rows,
        "blind_camera_cols": blind_cols,
        "blind_camera_pixels": blind_rows * CAMERA_W + blind_cols * CAMERA_H - blind_rows * blind_cols,
        "weight_sums_row_min": float(row.sum(axis=1).min()),
        "weight_sums_row_max": float(row.sum(axis=1).max()),
        "achievable_uint8_residual": {
            "n_blocks_sampled": int(n_weight_sets),
            "n_targets": int(n_targets),
            "mean_abs_LSB": float(residuals_arr.mean()),
            "rms_LSB": float(math.sqrt((residuals_arr**2).mean())),
            "max_LSB": float(residuals_arr.max()),
            "p99_LSB": float(np.quantile(residuals_arr, 0.99)),
            "solve": "EXACT min over the full 256^k camera lattice per block",
        },
        "plain_rint_residual_same_targets": {
            "mean_abs_LSB": float(quant_only.mean()),
            "rms_LSB": float(math.sqrt((quant_only**2).mean())),
        },
        "torch_version": torch.__version__,
    }


# ---------------------------------------------------------------------------
# Scorer-touching instrument validation (a handful of frames, never a measurement).
# ---------------------------------------------------------------------------


def control_scorer(gt_path: Path, indices: tuple[int, ...]) -> dict[str, Any]:
    """Prove this module's ``D`` is the operator the cached ``lstars`` came from.

    Runs the frozen CPU-torch SegNet on ``D(gt_f1)`` for a few frames and requires a
    BIT-EXACT argmax match against the cached labels.  A mismatch means the instrument
    is measuring a different chain than the authority, which would invalidate every
    number this module emits.
    """

    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    gt = open_stored_npz(gt_path)
    segnet = load_real_segnet("cpu")
    rows = []
    for index in indices:
        raw = gt["gt_f1"].frame(index)
        field = scorer_downsample(raw)  # (384,512,3) float32 -- this module's D
        tensor = torch.from_numpy(field).permute(2, 0, 1).unsqueeze(0)  # (1,3,384,512)
        # The authority's own path: build the degenerate pair the cache builder used
        # and call segnet.preprocess_input, which IS upstream/modules.py:109.
        pair = torch.from_numpy(np.stack([raw, raw], axis=0)[None].astype(np.float64)).float()
        with torch.inference_mode():
            authority_field = segnet.preprocess_input(pair.permute(0, 1, 4, 2, 3).contiguous())
            logits = segnet(tensor)
            argmax = logits.argmax(dim=1)[0].numpy().astype(np.int64)
        cached = gt["lstars"].frame(index)
        rows.append(
            {
                "frame": int(index),
                "D_vs_upstream_preprocess_maxabs": float(
                    (authority_field - tensor).abs().max().item()
                ),
                "argmax_mismatches": int(np.count_nonzero(argmax != cached)),
                "pixels": int(cached.size),
            }
        )
    return {
        "control": "D-parity: this module's D vs segnet.preprocess_input, and "
        "SegNet(D(gt_f1)) argmax vs cached lstars",
        "expect": "D_vs_upstream_preprocess_maxabs == 0.0 and 0 argmax mismatches",
        "rows": rows,
        "passed": all(
            row["argmax_mismatches"] == 0 and row["D_vs_upstream_preprocess_maxabs"] == 0.0
            for row in rows
        ),
    }


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "sweep", "aggregate", "join", "lattice", "reach-curve",
            "control-scorer", "control-receiver",
        ),
    )
    parser.add_argument("--render", default="ideal", choices=("ideal", "live"))
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--v4d-zip", type=Path, default=DEFAULT_V4D_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument("--n-expected", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--no-field", action="store_true", help="skip per-pixel persistence")
    parser.add_argument("--control-frames", type=int, nargs="*", default=[0, 137, 599])
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="*",
        default=[0.000298, 0.01, 0.05, 0.1, 0.17200, 0.19567, 0.28868, 0.30111,
                 0.34517, 0.5, 0.77254, 0.79899, 1.0],
    )
    parser.add_argument("--lattice-blocks", type=int, default=64)
    parser.add_argument("--lattice-targets", type=int, default=256)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "sweep":
        path = run_sweep(
            gt_path=args.gt,
            render_kind=args.render,
            start=args.start,
            end=args.end,
            out_dir=args.out_dir,
            v4d_zip=args.v4d_zip,
            persist_field=not args.no_field,
        )
        print(json.dumps({"wrote": str(path)}))
        return 0

    if args.mode == "aggregate":
        report = aggregate(args.out_dir, args.render, args.n_expected)
    elif args.mode == "join":
        report = run_join(
            gt_path=args.gt,
            out_dir=args.out_dir,
            render_kind=args.render,
            n_expected=args.n_expected,
            seed=args.seed,
        )
    elif args.mode == "reach-curve":
        report = reach_curve(args.gt, args.n_expected, tuple(args.rhos))
    elif args.mode == "lattice":
        report = lattice_report(
            gt_path=args.gt,
            n_weight_sets=args.lattice_blocks,
            n_targets=args.lattice_targets,
            seed=args.seed,
        )
    elif args.mode == "control-scorer":
        report = control_scorer(args.gt, tuple(args.control_frames))
    else:
        source = LiveV4dRender(args.v4d_zip, args.out_dir)
        rows = [
            {"frame": int(i), "camera_value_mismatches": source.control_receiver_parity(int(i))}
            for i in args.control_frames
        ]
        report = {
            "control": "receiver-parity: this module's clip(rint(U(r))) vs "
            "ddm_tr1_runtime.render_frame1_camera_uint8",
            "expect": "0 mismatches on every frame",
            "archive_sha256": source.archive_sha256,
            "archive_bytes": source.archive_bytes,
            "n_pairs": source.n_pairs,
            "rows": rows,
            "passed": all(row["camera_value_mismatches"] == 0 for row in rows),
        }

    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json_out is not None:
        _refuse_tmp(args.json_out)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
