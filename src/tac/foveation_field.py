"""Telescopic foveation field — per-frame attention-warp residual sidecar.

Lane: ``lane_pose_axis_telescopic_foveation_field_full_scaffold`` (Phase 3
substrate-engineering; pose-axis target). Per the handoff P3 ledger
``~/Downloads/pact_score_lowering_handoff_2026-05-11.md`` ("Wavelet/foveation/
RAFT/ego-motion" section) and Grand Council Insight 3 (pose-axis lanes have
**2.79× higher EV per byte at PR106 r2 operating point** vs SegNet), the
foveation field is a charged byte stream that the inflate-time consumer reads
and uses to warp the PR106 r2 base decode toward score-sensitive regions
(camera centre + road-region rows that drive PoseNet).

Architecture
============

Per-frame foveation field parameterised as a small grid of Gaussian "lenses"::

    field = {
        "n_frames": int,
        "n_gauss": int,
        "centers":   (n_frames, n_gauss, 2)  float16  in [0, 1]²
        "log_sigma": (n_frames, n_gauss)     float16  log-space stddev
        "log_amp":   (n_frames, n_gauss)     float16  log-space amplitude
    }

The warp at frame ``t`` is a sum of Gaussian displacements applied via
``F.grid_sample`` over an identity grid::

    delta(x, y) = sum_g amp_g * exp(-||(x,y) - center_g||² / (2 sigma_g²)) *
                  (center_g - (x, y))

This pulls pixels toward the centres of high-attention regions. The warp is
fully differentiable so the future trainer can backprop pose-loss into the
field params. Composition is RESIDUAL on top of the PR106 r2 base RGB output,
so HNeRV-parity lesson 5 (FULL renderer / RGB out) is satisfied.

Byte budget
-----------
- ``n_frames=1200``, ``n_gauss=4``: 1200 * 4 * (2+1+1) * 2 = 38,400 bytes raw
- After delta-quantisation + brotli: target **≤ 500 B** (operator constraint;
  enforced by ``compute_foveation_byte_budget``).

The encoded payload uses delta-from-frame-0 + int8 quantisation + brotli, the
same primitive family as ``tac.pose_delta_codec`` (Schmidhuber eureka).

Wire format (foveation field sidecar)
-------------------------------------
::

    magic        : u8  = 0xFC  (FOVEATION_FIELD_MAGIC)
    format_id    : u8  = 0x30  (FOVEATION_FIELD_FORMAT_ID)
    n_frames     : u16_LE
    n_gauss      : u8
    quant_scale  : f32_LE  (delta quantisation scale)
    anchor_centers   : n_gauss * 2 * f16_LE  (frame-0 absolute, in [0,1]²)
    anchor_log_sigma : n_gauss * f16_LE
    anchor_log_amp   : n_gauss * f16_LE
    deltas       : brotli(int8 * (n_frames-1) * n_gauss * 4)
                   [last axis = (cx, cy, log_sigma, log_amp)]

CLAUDE.md compliance
====================
- ``score_claim = False`` permanently until the charged archive consumer +
  exact T4 land. **NO authoritative score claims** in this module.
- ``promotion_eligible = False`` permanently.
- ``ready_for_exact_eval_dispatch = False`` until council deliberation.
- 8 archive-grammar fields declared in module docstring + lane registry.
- ``research_only = False`` because the inflate-time consumer is the
  ``submissions/pr106_foveation_field_sidecar/`` packet (built alongside).
- ``lane_class = substrate_engineering`` (LOC > the bolt-on 350-LOC budget;
  full RGB renderer composition).
- NO scorer load (the warp is RGB → RGB; PoseNet/SegNet only run downstream
  in the contest evaluator).
- NO ``/tmp`` paths; module is pure-Python with brotli + numpy + torch only.
- NO MPS authoritative; ``select_device`` rejects MPS at encode time.
- eval_roundtrip gradient-reachable by construction (grid_sample is
  differentiable; the future trainer will close the loop).

8 archive-grammar fields (Catalog #124)
=======================================
- ``archive_grammar``: monolithic single-file ``0.bin`` (0xFC + 0x30 wrapper)
- ``parser_section_manifest``: see ``FOVEATION_FIELD_FORMAT`` constant below
- ``inflate_runtime_loc_budget``: 200 LOC waiver under substrate_engineering
- ``runtime_dep_closure``: torch + brotli + numpy (all in contest runtime)
- ``export_format``: ``foveation_field_v1`` (this module emits the bytes)
- ``score_aware_loss``: deferred to trainer (research_only encoder right now)
- ``bolt_on_loc_budget``: substrate_engineering (this module is the substrate)
- ``no_op_detector_planned``: warp must shift > 1e-3 norm vs identity; tested

6-hook wire-in declarations
===========================
- Sensitivity-map: foveation_field.predict_pose_delta() returns predicted
  Δpose for the autopilot to consume (N/A — predicted column only)
- Pareto: this lane is a candidate when L2 encoder + dispatch land
- Bit-allocator: ``compute_foveation_byte_budget`` informs allocator
- Cathedral autopilot: register as ``optimize_mode_transforms`` candidate
- Continual-learning posterior update: triggered on exact T4 result
- Probe-disambiguator: foveation-vs-RAFT-vs-LAPose head-to-head (see
  sister modules ``tac.raft_pose_stream`` + ``tac.lapose_motion_atom_allocator``)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

FOVEATION_FIELD_MAGIC = 0xFC
FOVEATION_FIELD_FORMAT_ID = 0x30
FOVEATION_FIELD_VERSION = 1

# Hard byte budget per CLAUDE.md / handoff operator constraint.
MAX_ENCODED_BYTES = 500

# Sensible defaults: 4 Gaussians per frame is enough to model centre-of-attention
# + road-region pull without consuming bytes.
DEFAULT_N_GAUSS = 4

# Quantization scale chosen so that int8 covers [-1.27, 1.27] in normalized
# coordinates (frame param values live in roughly [0, 1] absolute, deltas
# rarely exceed 0.1). 1.0 / 127 ~ 7.87e-3 per quantum.
DEFAULT_DELTA_SCALE = 0.01

# No-op detector threshold: warp displacement L2 norm must move at least this
# many normalized-coordinate units across the frame stack for the sidecar to
# be considered non-trivially consuming bytes.
NO_OP_DISPLACEMENT_THRESHOLD = 1e-3


@dataclass
class FoveationField:
    """Per-frame foveation field parameters.

    All tensors are CPU float32 in this representation. The encoder/decoder
    handle quantization + delta encoding.
    """

    centers: torch.Tensor  # (n_frames, n_gauss, 2) in [0, 1]²
    log_sigma: torch.Tensor  # (n_frames, n_gauss)
    log_amp: torch.Tensor  # (n_frames, n_gauss)

    @property
    def n_frames(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_gauss(self) -> int:
        return int(self.centers.shape[1])

    def validate(self) -> None:
        """Hard-error on shape / dtype / bounds violations."""
        if self.centers.ndim != 3 or self.centers.shape[-1] != 2:
            raise ValueError(
                f"centers must have shape (n_frames, n_gauss, 2); got {tuple(self.centers.shape)}"
            )
        nf, ng = self.centers.shape[:2]
        if self.log_sigma.shape != (nf, ng):
            raise ValueError(
                f"log_sigma shape mismatch: got {tuple(self.log_sigma.shape)}; expected ({nf}, {ng})"
            )
        if self.log_amp.shape != (nf, ng):
            raise ValueError(
                f"log_amp shape mismatch: got {tuple(self.log_amp.shape)}; expected ({nf}, {ng})"
            )
        if not torch.all(torch.isfinite(self.centers)):
            raise ValueError("centers contains non-finite values")
        if not torch.all(torch.isfinite(self.log_sigma)):
            raise ValueError("log_sigma contains non-finite values")
        if not torch.all(torch.isfinite(self.log_amp)):
            raise ValueError("log_amp contains non-finite values")


def _select_device_no_mps(device: str | torch.device) -> torch.device:
    """Resolve device; reject MPS per CLAUDE.md non-negotiable.

    MPS is forbidden because PoseNet drifts 23× on MPS. The downstream consumer
    of the warp is the contest evaluator; the warp itself is computed on
    CUDA or CPU only.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "mps":
        raise ValueError(
            "MPS is forbidden for foveation-field warp computation per CLAUDE.md; "
            "use 'cpu' or 'cuda'"
        )
    return dev


def compute_foveation_warp(
    field: FoveationField,
    decoded_rgb: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Apply per-frame foveation warp to a stack of decoded RGB frames.

    Differentiable end-to-end through ``F.grid_sample`` (bilinear, zero-pad
    boundary) so a future trainer can backprop pose-loss into the field.

    Args:
        field: ``FoveationField`` with ``n_frames`` matching ``decoded_rgb.shape[0]``.
        decoded_rgb: ``(T, 3, H, W)`` float32 in [0, 255]. Contiguous.
        device: Override; defaults to ``decoded_rgb.device``. MPS rejected.

    Returns:
        Warped tensor of the same shape, same dtype, same device.
    """
    field.validate()
    if decoded_rgb.ndim != 4 or decoded_rgb.shape[1] != 3:
        raise ValueError(
            f"decoded_rgb must have shape (T, 3, H, W); got {tuple(decoded_rgb.shape)}"
        )
    if decoded_rgb.shape[0] != field.n_frames:
        raise ValueError(
            f"decoded_rgb T={decoded_rgb.shape[0]} != field.n_frames={field.n_frames}"
        )
    target_device = _select_device_no_mps(device or decoded_rgb.device)
    rgb = decoded_rgb.to(target_device)
    centers = field.centers.to(target_device)
    sigma = field.log_sigma.exp().to(target_device)  # (T, G)
    amp = field.log_amp.exp().to(target_device)  # (T, G)
    t_dim, _, h, w = rgb.shape

    # Build identity grid in normalized coordinates [-1, 1] (grid_sample's
    # convention). We work in [0, 1]² internally for field params then convert.
    ys = torch.linspace(0.0, 1.0, h, device=target_device)
    xs = torch.linspace(0.0, 1.0, w, device=target_device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # both (H, W)
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) in [0,1]²

    # Compute per-Gaussian displacement contributions and sum.
    # Vectorize over T frames and G gaussians.
    # base_grid expanded to (T, H, W, 2); centers to (T, 1, 1, G, 2).
    base_exp = base_grid.unsqueeze(0).expand(t_dim, -1, -1, -1)  # (T, H, W, 2)
    # Pull each pixel toward each Gaussian centre, weighted by amplitude and
    # Gaussian envelope.
    delta = torch.zeros_like(base_exp)  # (T, H, W, 2)
    for g in range(field.n_gauss):
        c = centers[:, g].unsqueeze(1).unsqueeze(1)  # (T, 1, 1, 2)
        s = sigma[:, g].unsqueeze(-1).unsqueeze(-1)  # (T, 1, 1)
        a = amp[:, g].unsqueeze(-1).unsqueeze(-1)  # (T, 1, 1)
        diff = c - base_exp  # (T, H, W, 2)
        sq_dist = (diff ** 2).sum(dim=-1)  # (T, H, W)
        envelope = torch.exp(-sq_dist / (2.0 * s ** 2 + 1e-12))  # (T, H, W)
        delta = delta + (a * envelope).unsqueeze(-1) * diff

    warped_grid = base_exp + delta  # (T, H, W, 2) in [0,1]²
    # Convert to [-1, 1] for grid_sample.
    warped_grid = 2.0 * warped_grid - 1.0

    warped = F.grid_sample(
        rgb,
        warped_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped


def _quantize_int8(values: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric int8 quantization."""
    q = np.round(values / scale).clip(-127, 127).astype(np.int8)
    return q


def _dequantize_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * float(scale)


def encode_foveation_field(
    field: FoveationField,
    *,
    delta_scale: float = DEFAULT_DELTA_SCALE,
    enforce_budget: bool = True,
) -> bytes:
    """Encode a ``FoveationField`` to the wire format.

    Returns the byte string. Raises if ``enforce_budget`` and the encoded
    payload exceeds ``MAX_ENCODED_BYTES`` (per CLAUDE.md / handoff operator
    constraint of ≤ 500 B).
    """
    field.validate()
    nf = field.n_frames
    ng = field.n_gauss
    if nf < 1:
        raise ValueError(f"n_frames must be >= 1; got {nf}")
    if ng < 1 or ng > 255:
        raise ValueError(f"n_gauss must be in [1, 255]; got {ng}")
    if nf > 65535:
        raise ValueError(f"n_frames must fit in u16; got {nf}")

    # Pack frame-0 absolute anchors.
    centers_np = field.centers.detach().cpu().float().numpy().astype(np.float16)
    log_sigma_np = field.log_sigma.detach().cpu().float().numpy().astype(np.float16)
    log_amp_np = field.log_amp.detach().cpu().float().numpy().astype(np.float16)

    anchor_centers = centers_np[0].reshape(-1)  # (ng*2,)
    anchor_log_sigma = log_sigma_np[0].reshape(-1)  # (ng,)
    anchor_log_amp = log_amp_np[0].reshape(-1)  # (ng,)

    # Compute deltas frame-by-frame.
    if nf > 1:
        # Build per-frame (ng, 4) where 4 = (cx, cy, log_sigma, log_amp).
        # Then delta = frame[t] - frame[t-1].
        full = np.empty((nf, ng, 4), dtype=np.float32)
        full[:, :, 0:2] = field.centers.detach().cpu().float().numpy()
        full[:, :, 2] = field.log_sigma.detach().cpu().float().numpy()
        full[:, :, 3] = field.log_amp.detach().cpu().float().numpy()
        deltas = full[1:] - full[:-1]  # (nf-1, ng, 4)
        deltas_q = _quantize_int8(deltas, delta_scale)
        deltas_payload = brotli.compress(deltas_q.tobytes(), quality=11)
    else:
        deltas_payload = b""

    # Pack the wire format.
    out = bytearray()
    out.append(FOVEATION_FIELD_MAGIC)
    out.append(FOVEATION_FIELD_FORMAT_ID)
    out += struct.pack("<H", nf)
    out += struct.pack("<B", ng)
    out += struct.pack("<f", float(delta_scale))
    out += anchor_centers.tobytes()
    out += anchor_log_sigma.tobytes()
    out += anchor_log_amp.tobytes()
    out += struct.pack("<I", len(deltas_payload))
    out += deltas_payload

    encoded = bytes(out)
    if enforce_budget and len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError(
            f"foveation field encoded size {len(encoded)} > budget {MAX_ENCODED_BYTES}; "
            "reduce n_frames, n_gauss, or use a coarser delta_scale"
        )
    return encoded


def decode_foveation_field(blob: bytes) -> FoveationField:
    """Inverse of :func:`encode_foveation_field`."""
    if len(blob) < 8:
        raise ValueError(f"foveation field blob too short: {len(blob)}")
    if blob[0] != FOVEATION_FIELD_MAGIC:
        raise ValueError(
            f"foveation field magic mismatch: 0x{blob[0]:02X} != 0x{FOVEATION_FIELD_MAGIC:02X}"
        )
    if blob[1] != FOVEATION_FIELD_FORMAT_ID:
        raise ValueError(
            f"foveation field format_id mismatch: 0x{blob[1]:02X} != 0x{FOVEATION_FIELD_FORMAT_ID:02X}"
        )
    pos = 2
    (nf,) = struct.unpack_from("<H", blob, pos)
    pos += 2
    (ng,) = struct.unpack_from("<B", blob, pos)
    pos += 1
    (delta_scale,) = struct.unpack_from("<f", blob, pos)
    pos += 4
    anchor_centers = np.frombuffer(blob, dtype=np.float16, count=ng * 2, offset=pos).astype(np.float32).reshape(ng, 2)
    pos += ng * 2 * 2
    anchor_log_sigma = np.frombuffer(blob, dtype=np.float16, count=ng, offset=pos).astype(np.float32)
    pos += ng * 2
    anchor_log_amp = np.frombuffer(blob, dtype=np.float16, count=ng, offset=pos).astype(np.float32)
    pos += ng * 2
    (deltas_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    deltas_payload = blob[pos : pos + deltas_len]
    pos += deltas_len
    if pos != len(blob):
        raise ValueError(f"foveation field trailing bytes: pos={pos} total={len(blob)}")

    # Reconstruct full per-frame params.
    if nf > 1 and deltas_len > 0:
        deltas_raw = brotli.decompress(deltas_payload)
        expected_bytes = (nf - 1) * ng * 4
        if len(deltas_raw) != expected_bytes:
            raise ValueError(
                f"deltas size mismatch: got {len(deltas_raw)}, expected {expected_bytes}"
            )
        deltas_q = np.frombuffer(deltas_raw, dtype=np.int8).reshape(nf - 1, ng, 4)
        deltas = _dequantize_int8(deltas_q, delta_scale)
        full = np.empty((nf, ng, 4), dtype=np.float32)
        full[0, :, 0:2] = anchor_centers
        full[0, :, 2] = anchor_log_sigma
        full[0, :, 3] = anchor_log_amp
        for t in range(1, nf):
            full[t] = full[t - 1] + deltas[t - 1]
        centers = torch.from_numpy(full[:, :, 0:2]).contiguous()
        log_sigma = torch.from_numpy(full[:, :, 2]).contiguous()
        log_amp = torch.from_numpy(full[:, :, 3]).contiguous()
    else:
        centers = torch.from_numpy(anchor_centers).unsqueeze(0).contiguous()
        log_sigma = torch.from_numpy(anchor_log_sigma).unsqueeze(0).contiguous()
        log_amp = torch.from_numpy(anchor_log_amp).unsqueeze(0).contiguous()

    return FoveationField(centers=centers, log_sigma=log_sigma, log_amp=log_amp)


def compute_foveation_byte_budget(field: FoveationField, *, delta_scale: float = DEFAULT_DELTA_SCALE) -> int:
    """Return the encoded byte budget of ``field`` without raising on overflow."""
    blob = encode_foveation_field(field, delta_scale=delta_scale, enforce_budget=False)
    return len(blob)


def is_no_op(field: FoveationField, *, threshold: float = NO_OP_DISPLACEMENT_THRESHOLD) -> bool:
    """Return True if the warp is effectively the identity transform.

    The trainer + the bit-allocator MUST refuse to spend bytes on a no-op
    field (per CLAUDE.md "no-op detector" non-negotiable / HNeRV parity
    discipline lesson 11). We measure max displacement induced by any pixel
    via the closed-form Gaussian sum at the four cardinal points + centre,
    in [0,1]² coordinates. If the max norm < threshold the warp is no-op.
    """
    field.validate()
    nf = field.n_frames
    ng = field.n_gauss
    centers = field.centers.detach().cpu().float().numpy()  # (T, G, 2)
    sigma = np.exp(field.log_sigma.detach().cpu().float().numpy())  # (T, G)
    amp = np.exp(field.log_amp.detach().cpu().float().numpy())  # (T, G)

    # Sample displacement at 9 probe points across the field.
    probes = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.5],
            [0.75, 0.5],
            [0.5, 0.25],
            [0.5, 0.75],
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ],
        dtype=np.float32,
    )  # (P, 2)
    max_disp = 0.0
    for t in range(nf):
        for px, py in probes:
            disp = np.zeros(2, dtype=np.float32)
            for g in range(ng):
                cx, cy = centers[t, g]
                sq_dist = (cx - px) ** 2 + (cy - py) ** 2
                envelope = float(np.exp(-sq_dist / (2.0 * sigma[t, g] ** 2 + 1e-12)))
                disp[0] += amp[t, g] * envelope * (cx - px)
                disp[1] += amp[t, g] * envelope * (cy - py)
            norm = float(np.linalg.norm(disp))
            if norm > max_disp:
                max_disp = norm
    return max_disp < threshold


__all__ = [
    "DEFAULT_DELTA_SCALE",
    "DEFAULT_N_GAUSS",
    "FOVEATION_FIELD_FORMAT_ID",
    "FOVEATION_FIELD_MAGIC",
    "FOVEATION_FIELD_VERSION",
    "FoveationField",
    "MAX_ENCODED_BYTES",
    "NO_OP_DISPLACEMENT_THRESHOLD",
    "compute_foveation_byte_budget",
    "compute_foveation_warp",
    "decode_foveation_field",
    "encode_foveation_field",
    "is_no_op",
]
