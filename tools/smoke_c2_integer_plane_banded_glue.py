#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-base/GT capped smoke for C2 glue; zero-band control, never promotion."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.boundary_math.integer_plane_banded_trainer import (
    LOGICAL_PAIR_COUNT,
    PLANE_SHAPE,
    StagePlan,
    TrainerConfig,
    canonical_json,
    inflate_worker_count,
    sha256_file,
    storage_preflight,
    train_streamed,
)
from tac.boundary_math.integer_plane_emitter import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    factor2_operator,
)
from tac.boundary_math.integer_plane_emitter_byte_close import (
    build_counted_archive,
    compare_capped_archives,
)
from tac.boundary_math.power_diagram_witness import (
    affine_head_to_power_diagram,
    encode_pdw2,
    pdw1_to_pdw2,
    read_frozen_segmentation_head,
)
from tac.witness_dsl.integer_plane_emitter_policy import (
    IntegerPlaneEmitterPolicy,
    PolicyMode,
)
from tools.measure_c2_integer_plane_emitter import _load_real_cache

SCHEMA = "c2_integer_plane_glue_real_capped_smoke.v1"
PDW2_N600_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4))


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project(camera: np.ndarray) -> np.ndarray:
    operator = factor2_operator()
    out = np.empty((camera.shape[0], *PLANE_SHAPE), dtype=np.uint8)
    for pair_index in range(camera.shape[0]):
        for plane_index in range(PLANE_COUNT):
            numerator, denominator = operator.apply_numerators(camera[pair_index, plane_index])
            out[pair_index, plane_index] = np.clip(
                np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0
            ).astype(np.uint8)
    return out


def _pdw2_margin_packet(upstream: Path) -> bytes:
    weight, bias = read_frozen_segmentation_head(upstream / "models/segnet.safetensors")
    target = affine_head_to_power_diagram(weight, bias, adjacency=PDW2_N600_EDGES).target
    return encode_pdw2(pdw1_to_pdw2(target, partition_only=False))


@dataclass(slots=True)
class PrefixRealSource:
    base: np.ndarray
    source: np.ndarray
    base_sha256: str
    source_sha256: str
    band_sha256: str
    pair_count: int = LOGICAL_PAIR_COUNT
    band_mode: str = "zero_radius_control"

    def fetch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.any(indices < 0) or np.any(indices >= len(self.base)):
            raise ValueError("capped real source received an out-of-prefix pair")
        base = np.asarray(self.base[indices], dtype=np.float32)
        source = np.asarray(self.source[indices], dtype=np.float32)
        return base, source, np.zeros(base.shape, dtype=np.float32)


def _decode_base_prefix(
    archive: Path, decoder: Path, root: Path, pair_cap: int
) -> tuple[np.ndarray, dict[str, object]]:
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
        if len(infos) != 1 or infos[0].filename != "0.bin":
            raise ValueError("base archive is not the counted one-packet grammar")
        packet = root / "0.bin"
        packet.write_bytes(handle.read(infos[0]))
    raw = root / "base.raw"
    env = os.environ.copy()
    workers = inflate_worker_count()
    env.update({"INFLATE_MAX_PAIRS": str(pair_cap), "INFLATE_WORKERS": str(workers)})
    command = [sys.executable, str(decoder), str(packet), str(raw)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise ValueError(f"base decoder failed rc={proc.returncode}: {proc.stderr[-1000:]}")
    shape = (pair_cap, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
    expected = int(np.prod(shape))
    if raw.stat().st_size != expected:
        raise ValueError("capped base decoder byte count mismatch")
    camera = np.memmap(raw, mode="r", dtype=np.uint8, shape=shape)
    return _project(camera), {
        "command": command,
        "environment": {"INFLATE_MAX_PAIRS": str(pair_cap), "INFLATE_WORKERS": str(workers)},
        "stdout_tail": proc.stdout[-500:],
        "scratch_raw_bytes": raw.stat().st_size,
        "scratch_rebuildable": True,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_root.expanduser().resolve()
    preflight = storage_preflight(output, required_free_bytes=args.required_free_bytes)
    if not preflight["ok"]:
        raise OSError("insufficient storage for capped real smoke")
    base_archive = args.base_archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    cache_fields, cache_sha = _load_real_cache(args.cache.expanduser().resolve())
    pdw2_packet = _pdw2_margin_packet(args.upstream.expanduser().resolve(strict=True))
    zero_band = {
        "derivation": "derive_margin_rgb_band",
        "scale": 0.0,
        "label": "zero_radius_control",
        "source_cache_sha256": cache_sha,
    }
    band_sha = _sha(canonical_json(zero_band))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c2_real_smoke_", dir=output) as scratch_name:
        scratch = Path(scratch_name)
        base_planes, base_decode = _decode_base_prefix(base_archive, decoder, scratch, args.pair_cap)
        source_camera = np.stack(
            (
                np.asarray(cache_fields["gt_f0"][: args.pair_cap]),
                np.asarray(cache_fields["gt_f1"][: args.pair_cap]),
            ),
            axis=1,
        )
        source_planes = _project(source_camera)
        config = TrainerConfig(
            policy=IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING),
            base_archive_sha256=sha256_file(base_archive),
            base_decoder_sha256=sha256_file(decoder),
            source_sha256=cache_sha,
            band_sha256=band_sha,
            band_mode="zero_radius_control",
            output_dir=output / "checkpoints",
            run_id=args.run_id,
            seed=args.seed,
            pair_batch_size=args.pair_batch_size,
            smoke_pair_cap=args.pair_cap,
            checkpoint_every_steps=args.checkpoint_every_steps,
            stages=(
                StagePlan("warmup", 1, 0.5, 1e-6),
                StagePlan("band_fit", 1, 0.25, 1e-5),
                StagePlan("rate_polish", 1, 0.1, 1e-3),
            ),
        )
        source = PrefixRealSource(
            base_planes,
            source_planes,
            config.base_archive_sha256,
            cache_sha,
            band_sha,
        )
        training = train_streamed(config, source)
        pre_archive = output / "pre_archive.zip"
        post_archive = output / "post_archive.zip"
        pre_build = build_counted_archive(
            base_archive=base_archive,
            checkpoint_path=training["checkpoint_paths"][0],
            output=pre_archive,
            pdw2_packet=pdw2_packet,
        )
        post_build = build_counted_archive(
            base_archive=base_archive,
            checkpoint_path=training["checkpoint_paths"][-1],
            output=post_archive,
            pdw2_packet=pdw2_packet,
        )
        comparison = compare_capped_archives(
            pre_archive=pre_archive,
            post_archive=post_archive,
            base_decoder=decoder,
            cache=args.cache,
            upstream=args.upstream,
            scratch_root=output / "decode_scratch",
            output_root=output / "hard_oracle",
            pair_cap=args.pair_cap,
            cpu_threads=args.cpu_threads,
        )
    return {
        "schema": SCHEMA,
        "authority": "capped macOS-CPU hard-oracle; non-n600; non-score; non-promotion",
        "logical_pair_count": LOGICAL_PAIR_COUNT,
        "executed_pair_cap": args.pair_cap,
        "real_gt_cache_sha256": cache_sha,
        "base_archive_sha256": sha256_file(base_archive),
        "base_decoder_sha256": sha256_file(decoder),
        "band": zero_band,
        "positive_anisotropic_band_active": False,
        "carrier_basis": "quotient_residual_polynomial_basis_v1_control_non_curvelet",
        "ev_metric_active": False,
        "pdw2": {
            "bytes": len(pdw2_packet),
            "sha256": _sha(pdw2_packet),
            "role": "training_only_target_certificate",
            "spatial_receiver_consumed": False,
            "verdict": "TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT",
        },
        "ready_to_fire": False,
        "remaining_blockers": [
            (
                "real n600 positive-anisotropic derive_hyperplane_channel_band artifact with "
                "winner/rival/VJP/pair-norm custody and measured EV ordering is absent"
            ),
            (
                "#553 PDW2 is gauge-fixed and counted, but its scorer-free spatial/RGB pullback "
                "is absent from the #543 factor-2 receiver"
            ),
            (
                "receiver-bound curvelet/shearlet carrier and executable Fisher-margin "
                "first-order+secant+QP EV field are absent; polynomial control is non-promotable"
            ),
        ],
        "storage_preflight": preflight,
        "base_decode": base_decode,
        "training": training,
        "pre_archive": pre_build,
        "post_archive": post_build,
        "hard_oracle": comparison,
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--base-decoder", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-id", default="c2_glue_real_smoke")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--pair-cap", type=int, default=6)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--required-free-bytes", type=int, default=2_000_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        if args.receipt.exists():
            raise ValueError(f"receipt overwrite refused: {args.receipt}")
        args.receipt.write_bytes(canonical_json(receipt))
    except OSError as exc:
        print(f"C2 smoke storage refusal: {exc}", file=sys.stderr)
        return 4
    except (ValueError, RuntimeError) as exc:
        print(f"C2 smoke refusal: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
