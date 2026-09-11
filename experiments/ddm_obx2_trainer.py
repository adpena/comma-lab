# SPDX-License-Identifier: MIT
"""OBX2 trainer: the born QBF generator plus a co-trained edge-local lattice.

The object is one packet.  Sections 1-4 are the born QBF generator (config,
model, latent metadata, per-pair latents) and section 5 is the counted
multiresolution lattice.  The lattice is queried by coordinate and gated by the
generator's OWN decoded signed-interface field, so it is edge-local without
shipping any support map.  Its output is a chroma-first RGB correction applied
to the render BEFORE the camera-size round trip.

Authority discipline.  The training twin runs in torch (CPU or MPS); the verdict
authority is the NumPy-fp32 reference receiver in `experiments/ddm_qbflow_packet`
plus this module's lattice reference, executed on the PARSED packet bytes.  MPS
is a gradient device only and is never a score.  Every admission measurement
reads the parsed packet, never a live tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from tac import obx2_lattice_packet as lat

N = 600
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
CHANNELS = 3
RATE_DENOMINATOR = 37_545_489
PACKET_BYTE_GATE = 122_000
DISTORTION_GATE = 0.04

QBT2B_CONTAINER = Path(
    "/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/"
    "governed_n32_r10/stage_05_same_budget_admission/reencoded/stage_05_end/"
    "reencode_payloads.tar"
)
QBT2B_PACKET_SHA256 = "607abebda2708f00daab79aac7bc6839d314096e6ed5693b642525487a1019f7"
OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction")

# Declared initial lattice geometry.  `(T, H, W)` per level with `channels`
# features each; level 2 is the per-pair level.  Sized from a MEASURED coder
# result, not a guess: one trained epoch on the first geometry produced 101,760
# codes that a real Brotli q11 race coded at 90,415 B (7.11 bits per code), far
# past the budget.  Holding the born model (79,688 B), latents (26,130 B),
# config (488 B), metadata (20 B) and framing, a 122,000 B packet leaves about
# 15,400 B for the lattice, so the geometry below is 15,840 codes.
DEFAULT_LEVELS = ((30, 6, 8), (10, 12, 16), (600, 1, 1))
DEFAULT_CHANNELS = 4
DEFAULT_BITS = (8, 8, 8)
DEFAULT_HIDDEN = 24
DEFAULT_GATE_TAU = 0.35
CONDITION_CHANNELS = qbf1.N_INTERFACES + 1  # tanh(signed interfaces) plus the gate itself
RENDER_OUTPUTS = 2 * CHANNELS  # two frames x RGB
DEFAULT_GATE_KIND = 1


def head_width(gate_kind: int) -> int:
    """Fusion head width: the blend kind emits a near and a far correction."""

    if gate_kind not in lat.GATE_KINDS:
        raise OBX2TrainerError(f"unknown lattice gate kind: {gate_kind}")
    return 2 * RENDER_OUTPUTS if gate_kind == 2 else RENDER_OUTPUTS


class OBX2TrainerError(RuntimeError):
    """Fail-closed refusal for OBX2 model, packet, or receiver-parity drift."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tar_member_bytes(path: Path, member: str) -> bytes:
    with tarfile.open(path, mode="r") as archive:
        stream = archive.extractfile(member)
        if stream is None:
            raise OBX2TrainerError(f"retained tar member is unreadable: {member}")
        return stream.read()


def born_packet() -> bytes:
    payload = tar_member_bytes(QBT2B_CONTAINER, "packet.qbf")
    if sha256_bytes(payload) != QBT2B_PACKET_SHA256:
        raise OBX2TrainerError("pinned QBT2B r10 packet drifted")
    return payload


def born_state(packet: bytes) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    decoded = qbf1.decode_packet(packet)
    params = qbf1.decode_model(decoded.sections[qbf1.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(decoded.sections[qbf1.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(decoded.sections[qbf1.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise OBX2TrainerError("born packet does not carry all 600 latent records")
    boundary = np.stack(
        [qbf1.dequantize(records[i][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)) for i in range(N)]
    )
    interior = np.stack(
        [qbf1.dequantize(records[i][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)) for i in range(N)]
    )
    return params, boundary, interior


def symmetric_scale(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-level symmetric uniform scale; never zero, so dequantization is defined."""

    limit = float((1 << (bits - 1)) - 1)
    peak = values.detach().abs().max()
    return torch.clamp(peak, min=1.0e-8) / limit


def quantize_ste(values: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Round-to-nearest with clamping and a straight-through gradient."""

    limit = float((1 << (bits - 1)) - 1)
    codes = torch.clamp(torch.round(values / scale), -limit - 1.0, limit)
    dequantized = codes * scale
    return values + (dequantized - values).detach()


def condition_from_interfaces(signed: torch.Tensor, gate_tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic decoded-state condition and edge gate; identical to the receiver."""

    nearest = signed.abs().min(dim=-1).values
    gate = torch.exp(-torch.square(nearest / gate_tau.clamp_min(1.0e-6)))
    condition = torch.cat((torch.tanh(signed), gate.unsqueeze(-1)), dim=-1)
    return condition, gate


def _axis_weights(coordinate: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low index, high index, and interpolation weight for one lattice axis."""

    position = ((coordinate + 1.0) * 0.5 * (size - 1)).clamp(0.0, float(size - 1))
    low = position.floor()
    high = torch.clamp(low + 1.0, max=float(size - 1))
    return low.long(), high.long(), position - low


class LatticeTorch(nn.Module):
    """Differentiable twin of the counted OBX2 lattice section."""

    def __init__(self, spec: lat.LatticeSpec, *, seed: int, prequantized: bool = False) -> None:
        super().__init__()
        if spec.condition_channels != CONDITION_CHANNELS or spec.outputs != head_width(spec.gate_kind):
            raise OBX2TrainerError("lattice spec does not match the OBX2 render contract")
        self.spec = spec
        # A receiver rebuilt from a packet already holds DEQUANTIZED grid values;
        # re-deriving a scale and re-quantizing them would be a silent second
        # pass of the quantizer, so the receiver skips it by construction rather
        # than relying on the round trip happening to be the identity.
        self.prequantized = bool(prequantized)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.grids = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros((depth, height, width, spec.channels), dtype=torch.float32)
                )
                for depth, height, width in spec.levels
            ]
        )
        self.hidden_w = nn.Parameter(
            torch.randn((spec.feature_width, spec.hidden), generator=generator) * (1.0 / math.sqrt(spec.feature_width))
        )
        self.hidden_b = nn.Parameter(torch.zeros(spec.hidden))
        # The output layer starts at exactly zero so an untrained lattice is the
        # identity on the born render; Stage 1 proves that byte-for-byte.
        self.out_w = nn.Parameter(torch.zeros((spec.hidden, spec.outputs)))
        self.out_b = nn.Parameter(torch.zeros(spec.outputs))
        self.gate_tau = nn.Parameter(torch.tensor([DEFAULT_GATE_TAU], dtype=torch.float32))

    def quantized_grids(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        grids, scales = [], []
        for grid, bits in zip(self.grids, self.spec.bits, strict=True):
            scale = symmetric_scale(grid, bits)
            grids.append(grid if self.prequantized else quantize_ste(grid, scale, bits))
            scales.append(scale)
        return grids, scales

    def sample(self, grids: Sequence[torch.Tensor], t: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trilinear sample of every level at normalized [-1,1] coordinates.

        Written as eight explicit gathers rather than `grid_sample` for two
        measured reasons: `aten::grid_sampler_3d_backward` has no MPS kernel, so
        the gradient device would be lost; and this form is the same arithmetic,
        in the same order, as `tac.obx2_lattice_packet.trilinear_sample`, which
        keeps the training twin structurally identical to the receiver.
        """

        features = []
        for grid in grids:
            depth, height, width, channels = grid.shape
            flat = grid.reshape(depth * height * width, channels)
            (t0, t1, ft) = _axis_weights(t, depth)
            (y0, y1, fy) = _axis_weights(y, height)
            (x0, x1, fx) = _axis_weights(x, width)
            sampled = torch.zeros((t.shape[0], channels), dtype=grid.dtype, device=grid.device)
            for ti, wt in ((t0, 1.0 - ft), (t1, ft)):
                for yi, wy in ((y0, 1.0 - fy), (y1, fy)):
                    for xi, wx in ((x0, 1.0 - fx), (x1, fx)):
                        index = (ti * height + yi) * width + xi
                        sampled = sampled + flat.index_select(0, index) * (wt * wy * wx).unsqueeze(-1)
            features.append(sampled)
        return torch.cat(features, dim=-1)

    def forward(
        self,
        *,
        t: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        grids, _ = self.quantized_grids()
        features = self.sample(grids, t, y, x)
        stacked = torch.cat((features, condition), dim=-1)
        hidden = torch.tanh(stacked @ self.hidden_w + self.hidden_b)
        return hidden @ self.out_w + self.out_b

    def export(self) -> tuple[list[np.ndarray], list[float], dict[str, np.ndarray]]:
        codes, scales = [], []
        for grid, bits in zip(self.grids, self.spec.bits, strict=True):
            scale = symmetric_scale(grid, bits)
            limit = float((1 << (bits - 1)) - 1)
            code = torch.clamp(torch.round(grid.detach() / scale), -limit - 1.0, limit)
            codes.append(code.cpu().numpy().astype(np.int64))
            scales.append(float(scale.detach().cpu()))
        fusion = {
            "hidden_w": self.hidden_w.detach().cpu().numpy().astype(np.float32),
            "hidden_b": self.hidden_b.detach().cpu().numpy().astype(np.float32),
            "out_w": self.out_w.detach().cpu().numpy().astype(np.float32),
            "out_b": self.out_b.detach().cpu().numpy().astype(np.float32),
            "gate_tau": self.gate_tau.detach().cpu().numpy().astype(np.float32),
        }
        return codes, scales, fusion


class OBX2Module(nn.Module):
    """Born QBF generator plus the co-trained edge-local correction lattice."""

    def __init__(self, base: qbt1.QBFLOWTorch, lattice: LatticeTorch) -> None:
        super().__init__()
        self.base = base
        self.lattice = lattice

    def forward(self, pair_ids: torch.Tensor, *, height: int = EVAL_H, width: int = EVAL_W) -> dict[str, torch.Tensor]:
        outputs = self.base(pair_ids, height=height, width=width)
        batch = int(pair_ids.numel())
        signed = outputs["signed_interfaces"]
        condition, gate = condition_from_interfaces(signed, self.lattice.gate_tau)
        device = signed.device
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pair_t = (-1.0 + 2.0 * pair_ids.float() / float(N - 1)).to(device)
        t_field = pair_t[:, None, None].expand(batch, height, width)
        correction = self.lattice(
            t=t_field.reshape(-1),
            y=grid_y.expand(batch, height, width).reshape(-1),
            x=grid_x.expand(batch, height, width).reshape(-1),
            condition=condition.reshape(-1, condition.shape[-1]),
        )
        gate_kind = self.lattice.spec.gate_kind
        correction = correction.reshape(batch, height, width, -1)
        weights = gate.reshape(batch, height, width, 1)
        if gate_kind == 0:
            blended = correction
        elif gate_kind == 1:
            blended = correction * weights
        else:
            half = correction.shape[-1] // 2
            blended = correction[..., :half] * weights + correction[..., half:] * (1.0 - weights)
        correction = blended.reshape(batch, height, width, 2, CHANNELS)
        base_rgb = outputs["rgb_pair_01"].permute(0, 3, 4, 1, 2)
        corrected = torch.clamp(base_rgb + correction, 0.0, 1.0)
        outputs["rgb_pair_01"] = corrected.permute(0, 3, 4, 1, 2)
        outputs["lattice_correction"] = correction
        outputs["interface_gate"] = gate
        return outputs


def default_spec(
    *,
    levels: Sequence[tuple[int, int, int]] = DEFAULT_LEVELS,
    channels: int = DEFAULT_CHANNELS,
    bits: Sequence[int] = DEFAULT_BITS,
    hidden: int = DEFAULT_HIDDEN,
    gate_kind: int = DEFAULT_GATE_KIND,
) -> lat.LatticeSpec:
    return lat.LatticeSpec(
        levels=tuple(tuple(int(value) for value in level) for level in levels),  # type: ignore[arg-type]
        channels=int(channels),
        bits=tuple(int(value) for value in bits),
        gate_kind=int(gate_kind),
        quantizer_kind=0,
        entropy_model=0,
        condition_channels=CONDITION_CHANNELS,
        hidden=int(hidden),
        outputs=head_width(int(gate_kind)),
    )


def build_module(packet: bytes, spec: lat.LatticeSpec, *, seed: int) -> OBX2Module:
    params, boundary, interior = born_state(packet)
    base = qbt1.QBFLOWTorch(params, boundary, interior)
    return OBX2Module(base, LatticeTorch(spec, seed=seed))


def build_packet(module: OBX2Module) -> tuple[bytes, dict[str, Any], bytes]:
    """Serialize the complete OBX2 object: born sections re-encoded plus the lattice."""

    packet = born_packet()
    decoded = qbf1.decode_packet(packet)
    params, boundary, interior = module.base.packet_state()
    latent_meta, boundary_codes, interior_codes = qbf1.encode_latent_meta(boundary, interior)
    raws = {
        lat.SECTION_CONFIG: decoded.sections[qbf1.SECTION_CONFIG],
        lat.SECTION_MODEL: qbf1.encode_model(params),
        lat.SECTION_LATENT_META: latent_meta,
        lat.SECTION_LATENTS: qbf1.encode_latent_table(range(N), boundary_codes, interior_codes),
        lat.SECTION_LATTICE: lat.encode_lattice_section(
            module.lattice.spec, **dict(zip(("codes", "scales", "fusion"), module.lattice.export(), strict=True))
        ),
    }
    sections = [lat.race_section(section_id, raw) for section_id, raw in raws.items()]
    payload = lat.pack_obx2_packet(sections)
    # The contest charges `archive.zip`, not the packet: the ZIP container adds
    # framing on top of every packet byte, so the gate is reported on both and
    # the rate term is computed from the archive.
    archive = qbf1.deterministic_archive(payload, member_name="0.obx2")
    archive_repeat = qbf1.deterministic_archive(payload, member_name="0.obx2")
    if archive_repeat != archive:
        raise OBX2TrainerError("deterministic archive builder is nondeterministic")
    if qbf1.read_deterministic_archive(archive, member_name="0.obx2") != payload:
        raise OBX2TrainerError("archive receiver does not return the packet it was built from")
    accounting = {
        "packet_bytes": len(payload),
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "archive_framing_bytes": len(archive) - len(payload),
        "archive_byte_headroom": PACKET_BYTE_GATE - len(archive),
        "rate_at_archive_bytes": 25.0 * len(archive) / RATE_DENOMINATOR,
        "sections": {
            lat.SECTION_NAMES[row.section_id]: {
                "raw_bytes": row.raw_bytes,
                "coded_bytes": len(row.payload),
                "codec": lat.CODEC_NAMES[row.codec_id],
                "candidates": dict(row.candidates),
            }
            for row in sections
        },
        "framing_bytes": len(payload) - sum(len(row.payload) for row in sections),
        "packet_sha256": sha256_bytes(payload),
        "packet_byte_gate": PACKET_BYTE_GATE,
        "packet_byte_headroom": PACKET_BYTE_GATE - len(payload),
    }
    return payload, accounting, archive


def receiver_render(packet: bytes, pair_ids: Sequence[int], *, height: int = EVAL_H, width: int = EVAL_W) -> np.ndarray:
    """NumPy-fp32 reference receiver on the PARSED packet: the verdict authority.

    Returns `[len(pair_ids), 2, 3, height, width]` render values in [0,1].
    """

    sections = lat.decode_obx2_packet(packet)
    params = qbf1.decode_model(sections[lat.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(sections[lat.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(sections[lat.SECTION_LATENTS])
    spec, codes, scales, fusion = lat.decode_lattice_section(sections[lat.SECTION_LATTICE])
    grids = lat.lattice_grids(spec, codes, scales)
    tau = float(np.asarray(fusion["gate_tau"]).reshape(-1)[0])

    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    out = np.zeros((len(pair_ids), 2, CHANNELS, height, width), dtype=np.float32)
    for index, pair_id in enumerate(pair_ids):
        boundary = qbf1.dequantize(records[int(pair_id)][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,))
        interior = qbf1.dequantize(records[int(pair_id)][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,))
        reference = qbf1.reference_forward(
            params,
            boundary,
            interior,
            pair_id=int(pair_id),
            num_pairs=N,
            height=height,
            width=width,
        )
        signed = np.asarray(reference["signed_interfaces"], dtype=np.float32).reshape(height, width, qbf1.N_INTERFACES)
        gate = lat.interface_gate(signed, tau)
        condition = np.concatenate((np.tanh(signed), gate[..., None]), axis=-1)
        t_value = np.float32(-1.0 + 2.0 * float(pair_id) / float(N - 1))
        correction = lat.query_lattice_numpy(
            spec,
            grids,
            fusion,
            t=np.full(height * width, t_value, dtype=np.float32),
            y=grid_y.reshape(-1),
            x=grid_x.reshape(-1),
            condition=condition.reshape(-1, spec.condition_channels),
        )
        correction = lat.blend_correction(correction, gate.reshape(-1), spec.gate_kind).reshape(
            height, width, 2, CHANNELS
        )
        rgb = np.asarray(reference["rgb_pair"], dtype=np.float32).reshape(height, width, 2, CHANNELS)
        out[index] = np.clip(rgb + correction, 0.0, 1.0).transpose(2, 3, 0, 1)
    return out


def parsed_module(packet: bytes) -> OBX2Module:
    """Rebuild the object from the PARSED packet bytes alone, for the torch receiver.

    Nothing here reads a live tensor: the generator parameters, the per-pair
    latents, the lattice codes, the per-level scales, and the fusion weights all
    come out of the decoded sections.  This is the receiver that is fast enough
    to ship (measured 0.311 s/pair at batch 16, 187 s for n600 against the
    1,260 s public budget, against 4.32 s/pair and 2,593 s for the portable
    float64 NumPy reference).  The two receivers disagree on 0.0666% of rounded
    uint8 values, so the admission row must be measured through the receiver that
    actually ships; the NumPy reference stays the portability cross-check.
    """

    sections = lat.decode_obx2_packet(packet)
    params = qbf1.decode_model(sections[lat.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(sections[lat.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(sections[lat.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise OBX2TrainerError("parsed packet does not carry all 600 latent records")
    boundary = np.stack(
        [qbf1.dequantize(records[i][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)) for i in range(N)]
    )
    interior = np.stack(
        [qbf1.dequantize(records[i][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)) for i in range(N)]
    )
    spec, codes, scales, fusion = lat.decode_lattice_section(sections[lat.SECTION_LATTICE])
    grids = lat.lattice_grids(spec, codes, scales)
    base = qbt1.QBFLOWTorch(params, boundary, interior)
    lattice = LatticeTorch(spec, seed=0, prequantized=True)
    with torch.no_grad():
        for parameter, grid in zip(lattice.grids, grids, strict=True):
            parameter.copy_(torch.from_numpy(np.ascontiguousarray(grid)))
        lattice.hidden_w.copy_(torch.from_numpy(fusion["hidden_w"]))
        lattice.hidden_b.copy_(torch.from_numpy(fusion["hidden_b"]))
        lattice.out_w.copy_(torch.from_numpy(fusion["out_w"]))
        lattice.out_b.copy_(torch.from_numpy(fusion["out_b"]))
        lattice.gate_tau.copy_(torch.from_numpy(fusion["gate_tau"]))
    module = OBX2Module(base, lattice)
    module.eval()
    return module


def torch_receiver_render(module: OBX2Module, pair_ids: Sequence[int]) -> torch.Tensor:
    """Render the parsed object with the shipping torch receiver."""

    with torch.no_grad():
        outputs = module(torch.tensor(list(pair_ids), dtype=torch.long))
        return outputs["rgb_pair_01"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parallel_receiver_render(packet: bytes, pair_ids: Sequence[int], *, workers: int) -> np.ndarray:
    """Run the NumPy verdict receiver over many pairs across processes."""

    if workers <= 1 or len(pair_ids) <= 1:
        return receiver_render(packet, pair_ids)
    chunks = [list(pair_ids[index::workers]) for index in range(workers)]
    chunks = [chunk for chunk in chunks if chunk]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {pool.submit(receiver_render, packet, chunk): chunk for chunk in chunks}
        rendered: dict[int, np.ndarray] = {}
        for future in futures:
            chunk = futures[future]
            result = future.result()
            for offset, pair_id in enumerate(chunk):
                rendered[int(pair_id)] = result[offset]
    return np.stack([rendered[int(pair_id)] for pair_id in pair_ids])


def teacher_render(stage2a_root: Path, pair_ids: Sequence[int]) -> np.ndarray:
    """Retained scorer-plane-matched 384x512 render: the Stage-2 distillation target."""

    wanted = {int(pair_id) for pair_id in pair_ids}
    collected: dict[int, np.ndarray] = {}
    for path in sorted((stage2a_root / "stage_2a" / "sp_384x512").glob("pairs_*.npz")):
        stem = path.stem.removeprefix("pairs_")
        first, last = (int(value) for value in stem.split("_"))
        if not wanted & set(range(first, last + 1)):
            continue
        with np.load(path, allow_pickle=False) as payload:
            ids = np.asarray(payload["pair_ids_i64"], dtype=np.int64)
            render = np.asarray(payload["render_u8"], dtype=np.uint8)
        for offset, pair_id in enumerate(ids.tolist()):
            if int(pair_id) in wanted:
                collected[int(pair_id)] = render[offset]
    missing = sorted(wanted - set(collected))
    if missing:
        raise OBX2TrainerError(f"retained sp_384x512 teacher render is missing pairs: {missing[:8]}")
    return np.stack([collected[int(pair_id)] for pair_id in pair_ids]).astype(np.float32) / 255.0


def operating_point_pose_weight(d_pose: float) -> float:
    """Exact contest derivative of sqrt(10*d_pose) at the current operating point."""

    return 5.0 / math.sqrt(10.0 * max(d_pose, 1.0e-9))


def stage_checkpoint_path(output: Path, stage: str, tag: str, *, stage_tag: str = "") -> Path:
    """Stage-encoded checkpoint path; a stage tag keeps a re-aimed stage distinct.

    A stage that re-derives its loss weights writes under its own tag so the
    earlier stage's checkpoints are preserved rather than continued over.
    """

    prefix = f"obx2_{stage}" if not stage_tag else f"obx2_{stage}_{stage_tag}"
    return output / "checkpoints" / f"{prefix}_{tag}.pt"


def save_stage_checkpoint(
    path: Path,
    *,
    module: OBX2Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    step: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomic, byte-close-loadable checkpoint; the EMA shadow is the authority."""

    payload = {
        "schema": "ddm_obx2_checkpoint.v1",
        "step": int(step),
        "config": dict(config),
        "model_state": {name: value.detach().cpu() for name, value in module.state_dict().items()},
        "ema_state": {name: value.detach().cpu() for name, value in ema.shadow.items()},
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "history": [dict(row) for row in history],
        "lattice_levels": [list(level) for level in module.lattice.spec.levels],
        "lattice_bits": list(module.lattice.spec.bits),
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    return qbt1.atomic_torch(path, payload)


RESUME_BINDING_KEYS = ("stage", "seed", "lattice_enabled", "chunk_pairs", "lattice_spec")


def assert_resume_compatible(saved: Mapping[str, Any], live: Mapping[str, Any], path: Path) -> None:
    """Refuse a resume whose config differs from the one that wrote the checkpoint.

    Torch reports a shape mismatch here as an opaque optimizer group error; a run
    resumed under a different lattice or stage would otherwise either crash late
    or, worse, continue against a different object.
    """

    differences = {
        key: {"checkpoint": saved.get(key), "live": live.get(key)}
        for key in RESUME_BINDING_KEYS
        if saved.get(key) != live.get(key)
    }
    if differences:
        raise OBX2TrainerError(
            f"resume config differs from the checkpoint that wrote {path}: {json.dumps(differences, default=str)}"
        )


def load_stage_checkpoint(
    path: Path,
    module: OBX2Module,
    optimizer: torch.optim.Optimizer,
    ema: Any,
    *,
    live_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ddm_obx2_checkpoint.v1":
        raise OBX2TrainerError(f"checkpoint schema differs: {path}")
    if live_config is not None:
        assert_resume_compatible(payload.get("config", {}), live_config, path)
    module.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    # A checkpoint is written on CPU, so the shadow must be restored onto the
    # device its module actually lives on; otherwise the first ema.update after
    # a resume mixes an MPS parameter with a CPU shadow and the run dies at the
    # first step of the new stage.
    live = dict(module.state_dict())
    for name, value in payload["ema_state"].items():
        reference = live.get(name)
        restored = value.clone()
        ema.shadow[name] = restored if reference is None else restored.to(reference.device)
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    return payload


def ema_module(module: OBX2Module, ema: Any, spec: lat.LatticeSpec, *, seed: int) -> OBX2Module:
    """A detached copy of the EMA shadow: the object that is byte-closed and scored."""

    shadow = build_module(born_packet(), spec, seed=seed)
    state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    for name, value in ema.shadow.items():
        state[name] = value.detach().cpu().clone()
    shadow.load_state_dict(state)
    shadow.eval()
    return shadow


def pair_order(seed: int, epoch: int) -> list[int]:
    generator = np.random.default_rng(seed * 1_000_003 + epoch)
    order = np.arange(N)
    generator.shuffle(order)
    return order.tolist()


SEG_LOCALITY_MAX_DISTANCE = 4


def seg_error_distance_histogram(argmax: np.ndarray, target: np.ndarray) -> dict[int, int]:
    """Misclassified pixels binned by their distance to a GT argmax boundary.

    The cure for a seg leg depends on where the error lives.  Error at distance 0
    or 1 is boundary jitter: the partition is right and its edge is a cell off.
    Error further in is region-level: a whole area is the wrong class, which no
    edge-local mechanism reaches.  Distance is the four-neighbour chessboard
    hop-count to the nearest GT class change, capped, with everything beyond the
    cap collected in one `SEG_LOCALITY_MAX_DISTANCE + 1` bin.
    """

    if argmax.shape != target.shape or argmax.ndim != 2:
        raise OBX2TrainerError("seg locality needs one pair's 2-D argmax and target")
    boundary = np.zeros(target.shape, dtype=bool)
    boundary[1:] |= target[1:] != target[:-1]
    boundary[:-1] |= target[1:] != target[:-1]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, 1:] != target[:, :-1]
    wrong = argmax != target
    histogram: dict[int, int] = {}
    reached = boundary.copy()
    remaining = wrong & ~reached
    histogram[0] = int((wrong & reached).sum())
    for distance in range(1, SEG_LOCALITY_MAX_DISTANCE + 1):
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        shell = grown & ~reached
        histogram[distance] = int((remaining & shell).sum())
        remaining = remaining & ~shell
        reached = grown
    histogram[SEG_LOCALITY_MAX_DISTANCE + 1] = int(remaining.sum())
    return histogram


def summarize_locality(histogram: Mapping[int, int], seg_errors: int) -> dict[str, Any]:
    """Where the seg errors live, as counts and as fractions of all seg errors."""

    total = sum(int(v) for v in histogram.values())
    if seg_errors and total != seg_errors:
        raise OBX2TrainerError(f"seg locality total {total} differs from the seg error count {seg_errors}")
    fractions = {str(k): (int(v) / total if total else 0.0) for k, v in sorted(histogram.items())}
    boundary_local = sum(int(v) for k, v in histogram.items() if int(k) <= 1)
    return {
        "counts_by_distance": {str(k): int(v) for k, v in sorted(histogram.items())},
        "fractions_by_distance": fractions,
        "boundary_local_fraction_within_1_cell": (boundary_local / total) if total else 0.0,
        "region_level_fraction_beyond_4_cells": (
            int(histogram.get(SEG_LOCALITY_MAX_DISTANCE + 1, 0)) / total if total else 0.0
        ),
        "distance_definition": "four-neighbour hop count to the nearest GT argmax class change",
        "total_seg_errors": total,
    }


def score_parsed_object(
    packet: bytes,
    *,
    pair_ids: Sequence[int],
    gt: np.ndarray,
    pose_target: np.ndarray,
    workers: int,
    render_chunk: int = 50,
    scorer_batch: int = 8,
    receiver: str = "torch",
    archive_bytes: bytes | None = None,
    retain_root: Path | None = None,
) -> dict[str, Any]:
    """Authority advisory row: render the PARSED packet, score on the frozen CPU scorers.

    `receiver` selects which reference decodes the parsed bytes.  They are not
    interchangeable: on a trained lattice they disagree on about 0.09% of rounded
    uint8 values, so a row is only meaningful with its receiver named.
    """

    from tac.scorer import load_differentiable_scorers

    if receiver not in ("numpy", "torch"):
        raise OBX2TrainerError(f"unknown receiver: {receiver}")
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    torch_module = parsed_module(packet) if receiver == "torch" else None
    seg_errors = 0
    seg_pixels = 0
    pose_square = 0.0
    pose_values = 0
    locality: dict[int, int] = {}
    if retain_root is not None:
        retain_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    render_seconds = 0.0
    for start in range(0, len(pair_ids), render_chunk):
        chunk = list(pair_ids[start : start + render_chunk])
        render_started = time.time()
        if torch_module is None:
            render = torch.from_numpy(parallel_receiver_render(packet, chunk, workers=workers))
        else:
            render = torch_receiver_render(torch_module, chunk)
        render_seconds += time.time() - render_started
        # Render in large chunks (one module build) but score in small batches:
        # the frozen scorers at 50 pairs peak past 16 GiB of RSS, which is both
        # wasteful and enough to trip a launch's own memory guard at the very
        # end of a long run.
        for offset in range(0, len(chunk), scorer_batch):
            pairs = chunk[offset : offset + scorer_batch]
            with torch.no_grad():
                camera = qbt1.roundtrip_to_camera_uint8_ste(render[offset : offset + len(pairs)])
                pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
                argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                pose = pose6.cpu().numpy().astype(np.float64)
            target = np.asarray(gt[pairs], dtype=np.uint8)
            seg_errors += int((argmax != target).sum())
            seg_pixels += int(target.size)
            pose_square += float(np.square(pose - np.asarray(pose_target[pairs], dtype=np.float64)).sum())
            pose_values += int(pose.size)
            for offset_index in range(len(pairs)):
                histogram = seg_error_distance_histogram(argmax[offset_index], target[offset_index])
                for distance, count in histogram.items():
                    locality[distance] = locality.get(distance, 0) + count
            if retain_root is not None:
                qbt1.atomic_npz(
                    retain_root / f"scored_{pairs[0]:04d}_{pairs[-1]:04d}.npz",
                    pair_ids_i64=np.asarray(pairs, dtype=np.int64),
                    segnet_argmax_u8=argmax,
                    posenet_pose6_f32=pose.astype("<f4"),
                    target_argmax_u8=target,
                )
    d_seg = seg_errors / seg_pixels
    d_pose = pose_square / pose_values
    distortion = 100.0 * d_seg + math.sqrt(10.0 * d_pose)
    scored_bytes = len(archive_bytes) if archive_bytes is not None else len(packet)
    rate = 25.0 * scored_bytes / RATE_DENOMINATOR
    return {
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "receiver": receiver,
        "render_chunk": render_chunk,
        "scorer_batch": scorer_batch,
        "pairs": len(pair_ids),
        "seg_errors": seg_errors,
        "seg_pixels": seg_pixels,
        "d_seg": d_seg,
        "pose_squared_error_sum": pose_square,
        "pose_values": pose_values,
        "d_pose": d_pose,
        "distortion": distortion,
        "scored_bytes": scored_bytes,
        "rate_at_scored_bytes": rate,
        "advisory_score_at_scored_bytes": rate + distortion,
        "passes_distortion_gate": distortion < DISTORTION_GATE,
        "passes_byte_gate": scored_bytes <= PACKET_BYTE_GATE,
        "seg_error_locality": summarize_locality(locality, seg_errors),
        "render_seconds": render_seconds,
        "total_seconds": time.time() - started,
    }


def run_training(
    output: Path,
    *,
    stage: str,
    device_name: str,
    epochs: int,
    chunk_pairs: int,
    learning_rate: float,
    seed: int,
    lattice_enabled: bool,
    resume_from: Path | None,
    workers: int,
    stage2a_root: Path,
    save_every_epochs: int,
    pose_weight_d_pose: float,
    validate_pairs: int,
    cross_check_pairs: int = 40,
    gate_kind: int = DEFAULT_GATE_KIND,
    stage_tag: str = "",
    pose_weight_derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One governed training stage over the complete 600-pair population."""

    if stage not in ("distill", "joint"):
        raise OBX2TrainerError(f"unsupported training stage: {stage}")
    from experiments import ddm_qbz1_descent_rate_configuration as qbz1
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    spec = default_spec(gate_kind=gate_kind)
    module = build_module(born_packet(), spec, seed=seed).to(device)
    module.train()
    if not lattice_enabled:
        for parameter in module.lattice.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    ema = EMA(module, decay=0.997)
    if pose_weight_derivation is not None:
        pose_weight = float(pose_weight_derivation["pose_weight"])
        pose_weight_d_pose = float(pose_weight_derivation["operating_point_d_pose"])
    else:
        pose_weight = operating_point_pose_weight(pose_weight_d_pose)
    config = {
        "stage": stage,
        "stage_tag": stage_tag,
        "device": device_name,
        "epochs": epochs,
        "chunk_pairs": chunk_pairs,
        "learning_rate": learning_rate,
        "seed": seed,
        "lattice_enabled": lattice_enabled,
        "gate_kind": int(gate_kind),
        "pose_weight": pose_weight,
        "pose_weight_operating_point_d_pose": pose_weight_d_pose,
        "lattice_spec": spec.describe(),
        "ema_decay": 0.997,
        "pose_weight_derivation": pose_weight_derivation,
    }
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume_from is not None and resume_from.is_file():
        payload = load_stage_checkpoint(resume_from, module, optimizer, ema, live_config=config)
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("step", 0))
        module.to(device)

    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    posenet = segnet = None
    if stage == "joint":
        posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=device)
        posenet.eval()
        segnet.eval()
    started = time.time()
    for epoch in range(start_epoch, epochs):
        order = pair_order(seed, epoch)
        epoch_loss = 0.0
        epoch_seg = 0.0
        epoch_pose = 0.0
        batches = 0
        tau = qbt1.tau_for_step(epoch, max(1, epochs))
        for index in range(0, N, chunk_pairs):
            ids = order[index : index + chunk_pairs]
            pair_ids = torch.tensor(ids, dtype=torch.long, device=device)
            outputs = module(pair_ids)
            if stage == "distill":
                target = torch.from_numpy(teacher_render(stage2a_root, ids)).to(device)
                loss = F.mse_loss(outputs["rgb_pair_01"], target)
            else:
                camera = qbt1.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
                pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
                target_seg = torch.from_numpy(np.asarray(gt[ids], dtype=np.int64)).to(device)
                target_pose = torch.from_numpy(np.asarray(pose_target[ids], dtype=np.float32)).to(device)
                seg_loss = qbt1.expected_flip_margin_loss(logits, target_seg, tau)
                pose_loss = F.mse_loss(pose6, target_pose)
                loss = 100.0 * seg_loss + pose_weight * pose_loss
                epoch_seg += float(seg_loss.detach())
                epoch_pose += float(pose_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ema.update(module)
            epoch_loss += float(loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "mean_loss": epoch_loss / max(1, batches),
            # Components, so a flattening total can be attributed to the term
            # that is actually stuck.  Both are surrogates read at the epoch's
            # tau, not scores.
            "mean_expected_flip": epoch_seg / max(1, batches),
            "mean_pose_mse": epoch_pose / max(1, batches),
            "mean_seg_contribution": 100.0 * epoch_seg / max(1, batches),
            "mean_pose_contribution": pose_weight * epoch_pose / max(1, batches),
            "tau": tau,
            "elapsed_seconds": time.time() - started,
            "axis": "[training-surrogate, not a score]",
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if (epoch + 1) % max(1, save_every_epochs) == 0 or epoch + 1 == epochs:
            save_stage_checkpoint(
                stage_checkpoint_path(output, stage, f"epoch_{epoch + 1:05d}", stage_tag=stage_tag),
                module=module,
                ema=ema,
                optimizer=optimizer,
                config=config,
                step=epoch + 1,
                history=history,
            )

    shadow = ema_module(module, ema, spec, seed=seed)
    packet, accounting, archive = build_packet(shadow)
    repeat, _, repeat_archive = build_packet(shadow)
    if repeat != packet or repeat_archive != archive:
        raise OBX2TrainerError("terminal packet or archive encoder is nondeterministic")
    terminal_stem = f"obx2_{stage}{'_' + stage_tag if stage_tag else ''}_terminal"
    packet_path = output / "candidates" / f"{terminal_stem}.packet"
    qbt1.atomic_bytes(packet_path, packet)
    archive_path = output / "candidates" / f"{terminal_stem}.archive.zip"
    qbt1.atomic_bytes(archive_path, archive)
    qbt1.atomic_bytes(output / "candidates" / f"{terminal_stem}.archive.repeat.zip", repeat_archive)
    validation = score_parsed_object(
        packet,
        pair_ids=list(range(min(validate_pairs, N))),
        gt=gt,
        pose_target=pose_target,
        workers=workers,
        receiver="torch",
        archive_bytes=archive,
    )
    cross_pairs = list(range(min(cross_check_pairs, N)))
    cross_check = {
        "pairs": len(cross_pairs),
        "numpy": score_parsed_object(
            packet, pair_ids=cross_pairs, gt=gt, pose_target=pose_target,
            workers=workers, receiver="numpy", archive_bytes=archive,
        ),
        "torch_same_pairs": score_parsed_object(
            packet, pair_ids=cross_pairs, gt=gt, pose_target=pose_target,
            workers=workers, receiver="torch", archive_bytes=archive,
        ),
        "note": "both receivers on the same pairs; a PREFIX, so not a population row",
    }
    cross_check["distortion_delta_numpy_minus_torch"] = (
        cross_check["numpy"]["distortion"] - cross_check["torch_same_pairs"]["distortion"]
    )
    receipt = {
        "schema": "ddm_obx2_training_stage.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "config": config,
        "packet_accounting": accounting,
        "packet_path": str(packet_path),
        "archive_path": str(archive_path),
        "deterministic_repeat": True,
        "validation": validation,
        "portable_receiver_cross_check": cross_check,
        "history_tail": history[-10:],
        "elapsed_seconds": time.time() - started,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / f"STAGE_{stage.upper()}{'_' + stage_tag.upper() if stage_tag else ''}_RESULT.json", receipt)
    return receipt


def stage7_public_timing(output: Path, *, archive_path: Path, receiver: str, workers: int) -> dict[str, Any]:
    """Decode the exact candidate twice: both runs inside 1,260 s and byte-identical."""

    archive = archive_path.read_bytes()
    packet = qbf1.read_deterministic_archive(archive, member_name="0.obx2")
    runs = []
    digests = []
    for attempt in range(2):
        started = time.time()
        frames = []
        for start in range(0, N, 50):
            chunk = list(range(start, min(N, start + 50)))
            if receiver == "torch":
                render = torch_receiver_render(parsed_module(packet), chunk)
            else:
                render = torch.from_numpy(parallel_receiver_render(packet, chunk, workers=workers))
            with torch.no_grad():
                camera = qbt1.roundtrip_to_camera_uint8_ste(render).to(torch.uint8)
            frames.append(hashlib.sha256(camera.cpu().numpy().tobytes(order="C")).hexdigest())
        elapsed = time.time() - started
        digest = hashlib.sha256("".join(frames).encode("ascii")).hexdigest()
        runs.append({"attempt": attempt, "elapsed_seconds": elapsed, "output_sha256": digest})
        digests.append(digest)
    receipt = {
        "schema": "ddm_obx2_stage7_public_timing.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory timing]",
        "score_claim": False,
        "promotable": False,
        "receiver": receiver,
        "archive": {"path": str(archive_path), "bytes": len(archive), "sha256": sha256_bytes(archive)},
        "runs": runs,
        "budget_seconds": 1_260,
        "deterministic_across_runs": digests[0] == digests[1],
        "slowest_seconds": max(row["elapsed_seconds"] for row in runs),
        "passes_timing_budget": max(row["elapsed_seconds"] for row in runs) <= 1_260,
        "note": (
            "this is local macOS timing, not the contest runtime; it bounds the receiver's own cost "
            "and proves decode determinism, it does not certify the contest host"
        ),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / f"STAGE_7_TIMING_{receiver}.json", receipt)
    return receipt


def score_checkpoint(
    output: Path,
    *,
    checkpoint: Path,
    seed: int,
    workers: int,
    validate_pairs: int,
    cross_check_pairs: int,
    zero_lattice: bool = False,
) -> dict[str, Any]:
    """Byte-close an existing stage checkpoint and score the parsed object on n600.

    This reads a checkpoint a live run already wrote; it trains nothing and it
    never touches the run's state.  The lattice geometry and gate kind come from
    the checkpoint's own config, so the object scored is the object that run is
    building.
    """

    from experiments import ddm_qbz1_descent_rate_configuration as qbz1
    from tac.training import EMA

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ddm_obx2_checkpoint.v1":
        raise OBX2TrainerError(f"checkpoint schema differs: {checkpoint}")
    saved = payload.get("config", {})
    spec = default_spec(gate_kind=int(saved.get("gate_kind", DEFAULT_GATE_KIND)))
    if spec.describe() != saved.get("lattice_spec", spec.describe()):
        raise OBX2TrainerError("checkpoint lattice geometry differs from the compiled default spec")
    module = build_module(born_packet(), spec, seed=seed)
    if not bool(saved.get("lattice_enabled", True)):
        for parameter in module.lattice.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([q for q in module.parameters() if q.requires_grad], lr=1.0e-4)
    ema = EMA(module, decay=0.997)
    load_stage_checkpoint(checkpoint, module, optimizer, ema)
    shadow = ema_module(module, ema, spec, seed=seed)
    if zero_lattice:
        # The ablation that attributes the seg leg to its author: with the head
        # zeroed the lattice emits exactly nothing, so whatever the object still
        # achieves is the base generator's work, not the correction's.
        with torch.no_grad():
            shadow.lattice.out_w.zero_()
            shadow.lattice.out_b.zero_()
    packet, accounting, archive = build_packet(shadow)
    repeat, _, repeat_archive = build_packet(shadow)
    if repeat != packet or repeat_archive != archive:
        raise OBX2TrainerError("checkpoint packet or archive encoder is nondeterministic")
    stem = f"{checkpoint.parent.parent.name}_{checkpoint.stem}" + ("_zerolattice" if zero_lattice else "")
    qbt1.atomic_bytes(output / "candidates" / f"{stem}.packet", packet)
    qbt1.atomic_bytes(output / "candidates" / f"{stem}.archive.zip", archive)

    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(qbz1.GT_POSE6, mmap_mode="r", allow_pickle=False)
    validation = score_parsed_object(
        packet,
        pair_ids=list(range(min(validate_pairs, N))),
        gt=gt,
        pose_target=pose_target,
        workers=workers,
        receiver="torch",
        archive_bytes=archive,
        retain_root=output / "scored" / stem,
    )
    cross_check = None
    if cross_check_pairs > 0:
        # Compare the two receivers on the SAME pairs.  A NumPy prefix read against
        # a 600-pair torch row measures prefix bias, not receiver disagreement.
        cross_pairs = list(range(min(cross_check_pairs, N)))
        cross_check = {
            "pairs": len(cross_pairs),
            "numpy": score_parsed_object(
                packet, pair_ids=cross_pairs, gt=gt, pose_target=pose_target,
                workers=workers, receiver="numpy", archive_bytes=archive,
            ),
            "torch_same_pairs": score_parsed_object(
                packet, pair_ids=cross_pairs, gt=gt, pose_target=pose_target,
                workers=workers, receiver="torch", archive_bytes=archive,
            ),
        }
        cross_check["distortion_delta_numpy_minus_torch"] = (
            cross_check["numpy"]["distortion"] - cross_check["torch_same_pairs"]["distortion"]
        )
        cross_check["note"] = (
            "both receivers on the same pairs; this is receiver disagreement, and it is a PREFIX "
            "so it is not a population row"
        )
    receipt = {
        "schema": "ddm_obx2_checkpoint_score.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "checkpoint": {"path": str(checkpoint), "step": int(payload.get("step", 0)), "config": saved},
        "zero_lattice_ablation": bool(zero_lattice),
        "packet_accounting": accounting,
        "validation": validation,
        "portable_receiver_cross_check": cross_check,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / f"CHECKPOINT_SCORE_{stem}.json", receipt)
    return receipt


def derive_pose_weight_from_row(row_path: Path) -> dict[str, Any]:
    """Derive the pose weight from a measured n600 row, pinning every input.

    The weight is the contest Pose term's own derivative at the operating point
    the LIVE object actually occupies, not a constant chosen in advance.
    """

    from tac.canonical_equations.obx2_pose_vs_scorer_plane_rmse_20260911 import derive_pose_weight

    payload = json.loads(row_path.read_text())
    if payload.get("schema") != "ddm_obx2_checkpoint_score.v1":
        raise OBX2TrainerError(f"pose-weight row schema differs: {row_path}")
    validation = payload["validation"]
    archive_path = Path(payload["packet_accounting"].get("archive_path", ""))
    archive_sha = payload["packet_accounting"].get("archive_sha256", "")
    derived = derive_pose_weight(
        d_pose=float(validation["d_pose"]),
        d_seg=float(validation["d_seg"]),
        source_artifact=str(row_path),
        source_sha256=str(archive_sha),
        receiver=str(validation["receiver"]),
        pair_denominator=int(validation["pairs"]),
    )
    derived["source_checkpoint"] = payload["checkpoint"]["path"]
    derived["source_step"] = payload["checkpoint"]["step"]
    derived["source_archive_bytes"] = payload["packet_accounting"]["archive_bytes"]
    derived["superseded_constant"] = payload["checkpoint"]["config"].get("pose_weight")
    if derived["superseded_constant"]:
        derived["constant_overweight_factor"] = float(derived["superseded_constant"]) / derived["pose_weight"]
    if archive_path.name:
        derived["source_archive_path"] = str(archive_path)
    return derived


POINTER_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
POINTER_FIELD_SHA256 = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"


def decompose_seg_error(
    output: Path,
    *,
    checkpoint: Path,
    seed: int,
    pairs: int,
    render_chunk: int = 25,
    scorer_batch: int = 8,
) -> dict[str, Any]:
    """Split the object's seg leg into PARTITION error and RENDER floor.

    The pointer reaches d_seg 0.000103 with EXACT tokens plus a renderer that
    still argmaxes wrong at correct tokens.  This object ships no token field; the
    closest analogue is the generator's own internal class head, whose argmax is
    the partition the generator represents BEFORE the render turns it into RGB.
    So each misclassified scorer pixel is attributed:

      * PARTITION — the generator already represented the wrong class there.  A
        correction applied to the RGB is not aimed at this.
      * RENDER FLOOR — the generator represented the RIGHT class and the
        render-plus-scorer path lost it.  This is what an RGB correction targets.

    Honest limit, stated because it bounds the conclusion: the class head is an
    internal intermediate, not a shipped field, and the training loss acts on the
    SCORER's logits rather than on it.  A wrong internal class is therefore
    strong evidence that the generator does not represent the site, not proof
    that no render could get it right.
    """

    from experiments import ddm_qbz1_descent_rate_configuration as qbz1
    from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})
    spec = default_spec(gate_kind=int(saved.get("gate_kind", DEFAULT_GATE_KIND)))
    module = build_module(born_packet(), spec, seed=seed)
    optimizer = torch.optim.AdamW([q for q in module.parameters() if q.requires_grad], lr=1.0e-4)
    ema = EMA(module, decay=0.997)
    load_stage_checkpoint(checkpoint, module, optimizer, ema)
    shadow = ema_module(module, ema, spec, seed=seed)
    packet, accounting, archive = build_packet(shadow)
    parsed = parsed_module(packet)

    assert_gt_lineage(qbz1.GT_ARGMAX, required=AUTHORITY_LINEAGE, instrument="OBX2 partition split")
    gt = np.load(qbz1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    with POINTER_FIELD.open("rb") as stream:
        field_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    field_fact = {"path": str(POINTER_FIELD), "bytes": POINTER_FIELD.stat().st_size, "sha256": field_digest}
    if field_digest != POINTER_FIELD_SHA256:
        raise OBX2TrainerError("pinned pointer token plane drifted")
    token_plane = np.memmap(POINTER_FIELD, dtype=np.uint8, mode="r", shape=(N, EVAL_H, EVAL_W))
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()

    totals = {
        "pixels": 0,
        "scorer_wrong": 0,
        "partition_wrong": 0,
        "scorer_wrong_and_partition_wrong": 0,
        "scorer_wrong_and_partition_right": 0,
        "partition_vs_pointer_tokens_differ": 0,
        "pointer_tokens_vs_gt_differ": 0,
    }
    locality_partition: dict[int, int] = {}
    locality_render: dict[int, int] = {}
    pair_ids = list(range(min(pairs, N)))
    started = time.time()
    for start in range(0, len(pair_ids), render_chunk):
        chunk = pair_ids[start : start + render_chunk]
        with torch.no_grad():
            outputs = parsed(torch.tensor(chunk, dtype=torch.long))
            partition = outputs["class_logits"].argmax(dim=-1).cpu().numpy().astype(np.uint8)
            render = outputs["rgb_pair_01"]
        for offset in range(0, len(chunk), scorer_batch):
            pairs_slice = chunk[offset : offset + scorer_batch]
            with torch.no_grad():
                camera = qbt1.roundtrip_to_camera_uint8_ste(render[offset : offset + len(pairs_slice)])
                _, logits = qbt1.scorer_forward(camera, posenet, segnet)
                scorer_argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            target = np.asarray(gt[pairs_slice], dtype=np.uint8)
            tokens = np.asarray(token_plane[pairs_slice], dtype=np.uint8)
            implied = partition[offset : offset + len(pairs_slice)]
            scorer_wrong = scorer_argmax != target
            partition_wrong = implied != target
            totals["pixels"] += int(target.size)
            totals["scorer_wrong"] += int(scorer_wrong.sum())
            totals["partition_wrong"] += int(partition_wrong.sum())
            totals["scorer_wrong_and_partition_wrong"] += int((scorer_wrong & partition_wrong).sum())
            totals["scorer_wrong_and_partition_right"] += int((scorer_wrong & ~partition_wrong).sum())
            totals["partition_vs_pointer_tokens_differ"] += int((implied != tokens).sum())
            totals["pointer_tokens_vs_gt_differ"] += int((tokens != target).sum())
            for index in range(len(pairs_slice)):
                for bucket, mask in (
                    (locality_partition, scorer_wrong[index] & partition_wrong[index]),
                    (locality_render, scorer_wrong[index] & ~partition_wrong[index]),
                ):
                    histogram = seg_error_distance_histogram(
                        np.where(mask, 1 - target[index].astype(np.int16), target[index]).astype(np.uint8),
                        target[index],
                    )
                    for distance, count in histogram.items():
                        bucket[distance] = bucket.get(distance, 0) + count

    wrong = max(1, totals["scorer_wrong"])
    receipt = {
        "schema": "ddm_obx2_seg_decomposition.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "checkpoint": {"path": str(checkpoint), "step": int(payload.get("step", 0))},
        "packet_bytes": accounting["packet_bytes"],
        "archive_bytes": accounting["archive_bytes"],
        "pairs": len(pair_ids),
        "receiver": "torch",
        "pointer_token_plane": field_fact,
        "counts": totals,
        "d_seg_scorer": totals["scorer_wrong"] / max(1, totals["pixels"]),
        "d_partition_vs_gt": totals["partition_wrong"] / max(1, totals["pixels"]),
        "d_pointer_tokens_vs_gt": totals["pointer_tokens_vs_gt_differ"] / max(1, totals["pixels"]),
        "d_partition_vs_pointer_tokens": totals["partition_vs_pointer_tokens_differ"] / max(1, totals["pixels"]),
        "share_of_seg_error_that_is_partition_level": totals["scorer_wrong_and_partition_wrong"] / wrong,
        "share_of_seg_error_that_is_render_floor": totals["scorer_wrong_and_partition_right"] / wrong,
        "partition_level_locality": summarize_locality(
            locality_partition, totals["scorer_wrong_and_partition_wrong"]
        ),
        "render_floor_locality": summarize_locality(
            locality_render, totals["scorer_wrong_and_partition_right"]
        ),
        "limit": (
            "the class head is an internal intermediate, not a shipped field, and the loss acts on "
            "the scorer's logits rather than on it; a wrong internal class is strong evidence that "
            "the generator does not represent the site, not proof that no render could recover it"
        ),
        "elapsed_seconds": time.time() - started,
    }
    qbt1.atomic_json(output / f"SEG_DECOMPOSITION_{checkpoint.stem}.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBX2 base+lattice trainer")
    parser.add_argument("stage", choices=("stage1", "distill", "joint", "stage7", "score", "decompose"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--parity-pairs", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--chunk-pairs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--no-lattice", action="store_true", help="freeze the lattice: the base-only A/B control")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stage2a-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--pose-weight-operating-point", type=float, default=1.0e-3)
    parser.add_argument(
        "--pose-weight-from-row",
        type=Path,
        default=None,
        help="a CHECKPOINT_SCORE json: derive the pose weight from ITS measured operating point",
    )
    parser.add_argument("--stage-tag", default="", help="keeps a re-aimed stage's checkpoints distinct")
    parser.add_argument("--validate-pairs", type=int, default=N)
    parser.add_argument("--cross-check-pairs", type=int, default=40)
    parser.add_argument("--gate-kind", type=int, default=DEFAULT_GATE_KIND, choices=(0, 1, 2))
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--zero-lattice",
        action="store_true",
        help="ablation: zero the lattice head so the row attributes the legs to the base generator",
    )
    parser.add_argument("--receiver", default="torch", choices=("torch", "numpy"))
    parser.add_argument("--launch-authorized", action="store_true")
    return parser


def stage1(output: Path, *, seed: int, parity_pairs: int, gate_kind: int = DEFAULT_GATE_KIND) -> dict[str, Any]:
    """Receiver parity: a zero lattice must be the born object, byte for byte."""

    started = time.time()
    spec = default_spec(gate_kind=gate_kind)
    module = build_module(born_packet(), spec, seed=seed)
    module.eval()
    payload, accounting, archive = build_packet(module)
    repeat, repeat_accounting, repeat_archive = build_packet(module)
    if repeat != payload or repeat_archive != archive:
        raise OBX2TrainerError("OBX2 packet or archive encoder is nondeterministic")
    sections = lat.decode_obx2_packet(payload)
    if set(sections) != set(lat.SECTION_NAMES):
        raise OBX2TrainerError("parsed OBX2 packet section set differs")

    pair_ids = list(range(min(parity_pairs, N)))
    parsed = receiver_render(payload, pair_ids)
    params, boundary, interior = born_state(born_packet())
    control = np.zeros_like(parsed)
    for index, pair_id in enumerate(pair_ids):
        reference = qbf1.reference_forward(
            params,
            boundary[pair_id],
            interior[pair_id],
            pair_id=pair_id,
            num_pairs=N,
            height=EVAL_H,
            width=EVAL_W,
        )
        rgb = np.asarray(reference["rgb_pair"], dtype=np.float32).reshape(EVAL_H, EVAL_W, 2, CHANNELS)
        control[index] = rgb.transpose(2, 3, 0, 1)
    max_abs = float(np.abs(parsed - control).max())

    with torch.no_grad():
        torch_out = module(torch.tensor(pair_ids, dtype=torch.long))["rgb_pair_01"].cpu().numpy()
    difference = torch_out - parsed
    max_abs_parity = float(np.abs(difference).max())
    reference_energy = float(np.sqrt(np.square(parsed).sum())) or 1.0
    parity_ratio = 1.0 - float(np.sqrt(np.square(difference).sum())) / reference_energy
    rounded_camera_disagreement = float(
        np.mean(np.round(torch_out * 255.0) != np.round(parsed * 255.0))
    )

    receipt = {
        "schema": "ddm_obx2_stage1_receiver_parity.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory receiver check]",
        "score_claim": False,
        "promotable": False,
        "research_only": True,
        "seed": seed,
        "lattice_spec": spec.describe(),
        "packet_accounting": accounting,
        "repeat_packet_sha256": repeat_accounting["packet_sha256"],
        "deterministic_repeat": True,
        "parity_pairs": pair_ids,
        "zero_lattice_vs_born_max_abs": max_abs,
        "zero_lattice_is_identity": max_abs == 0.0,
        "torch_vs_numpy_max_abs": max_abs_parity,
        "torch_vs_numpy_relative_l2_parity": parity_ratio,
        "torch_vs_numpy_rounded_uint8_disagreement_fraction": rounded_camera_disagreement,
        "parity_note": (
            "the torch twin accumulates in float32 and the receiver in float64 by design "
            "(experiments/ddm_qbflow_packet.reference_forward); torch is the gradient device only "
            "and the NumPy receiver on the parsed packet is the verdict authority"
        ),
        "elapsed_seconds": time.time() - started,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }
    qbt1.atomic_json(output / "checkpoints/stage_01_receiver_parity.json", receipt)
    if max_abs != 0.0:
        raise OBX2TrainerError(f"zero-lattice receiver is not the born object: max_abs={max_abs}")
    if parity_ratio < 0.9997:
        raise OBX2TrainerError(f"torch/NumPy relative-L2 parity below 0.9997: {parity_ratio}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if args.stage == "stage1":
        receipt = stage1(args.output, seed=args.seed, parity_pairs=args.parity_pairs, gate_kind=args.gate_kind)
        print(
            json.dumps(
                {
                    "stage": "stage1",
                    "packet_bytes": receipt["packet_accounting"]["packet_bytes"],
                    "headroom_to_gate": receipt["packet_accounting"]["packet_byte_headroom"],
                    "zero_lattice_is_identity": receipt["zero_lattice_is_identity"],
                    "torch_vs_numpy_relative_l2_parity": receipt["torch_vs_numpy_relative_l2_parity"],
                    "torch_vs_numpy_max_abs": receipt["torch_vs_numpy_max_abs"],
                    "torch_vs_numpy_rounded_uint8_disagreement_fraction": receipt[
                        "torch_vs_numpy_rounded_uint8_disagreement_fraction"
                    ],
                    "sections": {
                        name: row["coded_bytes"] for name, row in receipt["packet_accounting"]["sections"].items()
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.stage == "score":
        if args.checkpoint is None:
            raise OBX2TrainerError("score requires --checkpoint")
        receipt = score_checkpoint(
            args.output,
            checkpoint=args.checkpoint,
            seed=args.seed,
            workers=args.workers,
            validate_pairs=args.validate_pairs,
            cross_check_pairs=args.cross_check_pairs,
            zero_lattice=args.zero_lattice,
        )
        print(json.dumps({"stage": "score", "step": receipt["checkpoint"]["step"],
                          "packet_accounting": {k: receipt["packet_accounting"][k] for k in
                                                ("packet_bytes", "archive_bytes", "archive_byte_headroom")},
                          "zero_lattice_ablation": receipt["zero_lattice_ablation"],
                          "validation": receipt["validation"]}, indent=2))
        return 0
    if args.stage == "decompose":
        if args.checkpoint is None:
            raise OBX2TrainerError("decompose requires --checkpoint")
        receipt = decompose_seg_error(
            args.output, checkpoint=args.checkpoint, seed=args.seed, pairs=args.validate_pairs
        )
        print(json.dumps({k: receipt[k] for k in (
            "pairs", "d_seg_scorer", "d_partition_vs_gt", "d_pointer_tokens_vs_gt",
            "d_partition_vs_pointer_tokens", "share_of_seg_error_that_is_partition_level",
            "share_of_seg_error_that_is_render_floor")}, indent=2))
        return 0
    if args.stage == "stage7":
        if args.archive is None:
            raise OBX2TrainerError("stage7 requires --archive")
        receipt = stage7_public_timing(
            args.output, archive_path=args.archive, receiver=args.receiver, workers=args.workers
        )
        print(json.dumps(receipt, indent=2))
        return 0
    if not args.launch_authorized:
        raise OBX2TrainerError("n600 training requires explicit --launch-authorized")
    derivation = None
    if args.pose_weight_from_row is not None:
        derivation = derive_pose_weight_from_row(args.pose_weight_from_row)
        print(json.dumps({"pose_weight_derivation": derivation}, indent=2), flush=True)
    receipt = run_training(
        args.output,
        stage=args.stage,
        stage_tag=args.stage_tag,
        pose_weight_derivation=derivation,
        device_name=args.device,
        epochs=args.epochs,
        chunk_pairs=args.chunk_pairs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        lattice_enabled=not args.no_lattice,
        resume_from=args.resume_from,
        workers=args.workers,
        stage2a_root=args.stage2a_root,
        save_every_epochs=args.save_every_epochs,
        pose_weight_d_pose=args.pose_weight_operating_point,
        validate_pairs=args.validate_pairs,
        cross_check_pairs=args.cross_check_pairs,
        gate_kind=args.gate_kind,
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "validation": receipt["validation"],
                "portable_receiver_cross_check": receipt["portable_receiver_cross_check"],
                "packet_bytes": receipt["packet_accounting"]["packet_bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
