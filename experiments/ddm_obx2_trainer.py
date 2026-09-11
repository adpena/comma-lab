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
from collections.abc import Sequence
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
# features each.  Level 2 is the per-pair level: it replaces the born object's
# 28-value per-pair latent with 4*channels values per pair.
DEFAULT_LEVELS = ((60, 12, 16), (15, 24, 32), (600, 2, 2))
DEFAULT_CHANNELS = 4
DEFAULT_BITS = (8, 8, 8)
DEFAULT_HIDDEN = 24
DEFAULT_GATE_TAU = 0.35
CONDITION_CHANNELS = qbf1.N_INTERFACES + 1  # tanh(signed interfaces) plus the gate itself
OUTPUTS = 2 * CHANNELS


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


class LatticeTorch(nn.Module):
    """Differentiable twin of the counted OBX2 lattice section."""

    def __init__(self, spec: lat.LatticeSpec, *, seed: int) -> None:
        super().__init__()
        if spec.condition_channels != CONDITION_CHANNELS or spec.outputs != OUTPUTS:
            raise OBX2TrainerError("lattice spec does not match the OBX2 render contract")
        self.spec = spec
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
            grids.append(quantize_ste(grid, scale, bits))
            scales.append(scale)
        return grids, scales

    def sample(self, grids: Sequence[torch.Tensor], t: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trilinear sample of every level at normalized [-1,1] coordinates."""

        features = []
        for grid in grids:
            depth, height, width, channels = grid.shape
            volume = grid.permute(3, 0, 1, 2).unsqueeze(0)
            sample_grid = torch.stack((x, y, t), dim=-1).reshape(1, 1, 1, -1, 3)
            sampled = F.grid_sample(
                volume,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            features.append(sampled.reshape(channels, -1).transpose(0, 1))
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
        correction = correction.reshape(batch, height, width, 2, CHANNELS)
        correction = correction * gate.reshape(batch, height, width, 1, 1)
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
) -> lat.LatticeSpec:
    return lat.LatticeSpec(
        levels=tuple(tuple(int(value) for value in level) for level in levels),  # type: ignore[arg-type]
        channels=int(channels),
        bits=tuple(int(value) for value in bits),
        gate_kind=1,
        quantizer_kind=0,
        entropy_model=0,
        condition_channels=CONDITION_CHANNELS,
        hidden=int(hidden),
        outputs=OUTPUTS,
    )


def build_module(packet: bytes, spec: lat.LatticeSpec, *, seed: int) -> OBX2Module:
    params, boundary, interior = born_state(packet)
    base = qbt1.QBFLOWTorch(params, boundary, interior)
    return OBX2Module(base, LatticeTorch(spec, seed=seed))


def build_packet(module: OBX2Module) -> tuple[bytes, dict[str, Any]]:
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
    accounting = {
        "packet_bytes": len(payload),
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
    return payload, accounting


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
        ).reshape(height, width, 2, CHANNELS)
        correction = correction * gate[..., None, None]
        rgb = np.asarray(reference["rgb_pair"], dtype=np.float32).reshape(height, width, 2, CHANNELS)
        out[index] = np.clip(rgb + correction, 0.0, 1.0).transpose(2, 3, 0, 1)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OBX2 base+lattice trainer")
    parser.add_argument("stage", choices=("stage1",))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--parity-pairs", type=int, default=8)
    return parser


def stage1(output: Path, *, seed: int, parity_pairs: int) -> dict[str, Any]:
    """Receiver parity: a zero lattice must be the born object, byte for byte."""

    started = time.time()
    spec = default_spec()
    module = build_module(born_packet(), spec, seed=seed)
    module.eval()
    payload, accounting = build_packet(module)
    repeat, repeat_accounting = build_packet(module)
    if repeat != payload:
        raise OBX2TrainerError("OBX2 packet encoder is nondeterministic")
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
        receipt = stage1(args.output, seed=args.seed, parity_pairs=args.parity_pairs)
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
    raise OBX2TrainerError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
