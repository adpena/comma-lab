#!/usr/bin/env python3
"""ddm_up1 — an EXACT local pose instrument for the shipping object.

MEASURED VERDICT (this module's own runs; the hypothesis it was built to test is
FALSIFIED and the docstring says so rather than keeping the stale headline).

The live pointer body (archive ``50e56145…``) really does decode to two different
objects -- both raw digests re-derived at source, same 3,662,409,600 B geometry:

===================  ==============  ==============================
inflate device       score axis      raw ``0.raw`` sha256
===================  ==============  ==============================
cpu                  local advisory  ``ccbfa332…``
cuda                 T4 row r1       ``3c810cc4…``
===================  ==============  ==============================

The device is chosen by ``inflate.py:56`` (``"cuda" if torch.cuda.is_available()
else "cpu"``) and the only two lines where ``device.type`` changes COMPUTATION
rather than placement are ``cpr1/inflate.py:312`` (``semantic_batch = 8 if cuda
else 1``) and ``:335`` (``pose_batch = 64 if cuda else 1``) -- batch shape is part
of the forward instrument. That is a deterministic-decode compliance defect.

It is NOT, however, where the pose gap lived. Two hypotheses were tested here and
the first died:

1. FALSIFIED -- "the 19x pose gap is a global photometric shift". Swept at n=64:
   d_pose is MINIMISED at offset 0 (ratio 1.0000); +/-1 LSB costs 9-18% and +/-4
   LSB only 3.6x, in the WRONG direction. A rounding-scale perturbation cannot
   manufacture a 15.7x pose improvement.
2. CONFIRMED -- the gap is the GROUND-TRUTH DECODE LINEAGE, not our frames
   (``ddm_pi2``: 99.9960%). On the SAME cpu-decoded frames and the SAME frozen
   scorer, changing only the GT cache moves d_pose 23.74x at n=64
   (AV/PyAV 1.219851e-04 -> DALI 5.137472e-06), and at full n600 the DALI reading
   is **7.769511e-06 against the T4 row's 7.77e-06 -- ratio 0.9999**.

So the cpu-decoded frames are pose-equivalent to the cuda-decoded ones, and this
module is an exact local stand-in for the contest pose axis: n600 in ~140 s at
$0. The pose axis was never blocked on CUDA compute; it was blocked on a
GT-lineage bug inside our own tooling.

Consequence for solvers: any pose solve driven by an AV/PyAV GT cache is chasing
~23.7x of PHANTOM error that does not exist on the shipping axis. Point every
pose instrument at the DALI lineage before solving.

Authority: frozen CPU-torch PoseNet (``ddm_pk4``); ``ddm_ps1u`` measured PoseNet
CPU == CUDA at 0.9999 on identical frames, so the scorer is device-agnostic.
Numbers here are ``[macOS-CPU advisory]``; ``score_claim=false``,
``promotable=false``. Sampling is seeded-random and NEVER a prefix -- pose
prefixes measure 2.5-4.2x anti-conservatively (memory m88/m96).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"

FRAME_H = 874
FRAME_W = 1164
FRAME_C = 3
FRAME_BYTES = FRAME_H * FRAME_W * FRAME_C
N_PAIRS_TOTAL = 600

# The two decodes of the live pointer body, both re-derived at source by ddm_up1.
CPU_DECODE_RAW_SHA256 = "ccbfa3327d0f2486f8a2d7970fe89c5d56302eb1e04714d05eabff52278f1f9d"
CUDA_DECODE_RAW_SHA256 = "3c810cc4adff01a6783e8727f2cd7161e47d83693acc2aba7941b8ee7b115f6d"
POINTER_ARCHIVE_SHA256 = "50e561454b23026d3870f056747e848a49bd5f2b1e23930155d1281aeee91927"

# n600 receipts for the same archive (see module docstring).
ADVISORY_CPU_DPOSE_N600 = 1.4829e-04
T4_CUDA_DPOSE_N600 = 7.77e-06


class Up1Error(RuntimeError):
    """A precondition of the probe was not met."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class ProbeConfig:
    raw_path: Path
    gt_cache: Path
    output_dir: Path
    pairs: int
    seed: int
    offsets: tuple[float, ...]
    batch_pairs: int
    realize_uint8: bool
    verify_raw_sha: bool
    threads: int


def parse_offsets(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise Up1Error("at least one offset is required")
    if 0.0 not in values:
        raise Up1Error("the sweep must contain 0.0 -- it is the base control")
    return tuple(values)


def select_pairs(pairs: int, seed: int) -> np.ndarray:
    """Seeded-random pair sample. NEVER a prefix (memory m88/m96)."""
    if not 1 <= pairs <= N_PAIRS_TOTAL:
        raise Up1Error(f"pairs must be in [1, {N_PAIRS_TOTAL}], got {pairs}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N_PAIRS_TOTAL, size=pairs, replace=False))


def open_raw(path: Path, *, verify_sha: bool) -> np.memmap:
    if not path.is_file():
        raise Up1Error(f"decoded raw does not exist: {path}")
    size = path.stat().st_size
    expected = 2 * N_PAIRS_TOTAL * FRAME_BYTES
    if size != expected:
        raise Up1Error(f"raw is {size} B, expected {expected} B ({2 * N_PAIRS_TOTAL} frames)")
    if verify_sha:
        observed = _sha256_file(path)
        if observed not in {CPU_DECODE_RAW_SHA256, CUDA_DECODE_RAW_SHA256}:
            raise Up1Error(
                "raw sha256 matches neither known decode of the pointer body: " + observed
            )
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS_TOTAL, FRAME_H, FRAME_W, FRAME_C))


def load_gt_poses(path: Path) -> tuple[np.ndarray, str]:
    """Load GT pose targets and RECORD THEIR LINEAGE.

    ``ddm_pi2`` measured that the GT decode path -- PyAV ``yuv420_to_rgb`` versus
    DALI/nvdec -- accounts for 99.9960% of the advisory-vs-contest pose offset,
    while our own frames contribute at most 0.0040%. An instrument that does not
    know which lineage it is holding is measuring a different object than the one
    that ships. So the lineage is returned, never inferred, and it is stamped on
    every summary.
    """
    if not path.is_file():
        raise Up1Error(f"gt cache does not exist: {path}")
    if path.suffix == ".pt":
        import torch

        cache = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict) or "pose" not in cache:
            raise Up1Error(f"gt .pt cache has no 'pose' key: {path}")
        poses = np.asarray(cache["pose"], dtype=np.float64)
        lineage = "dali" if "dali" in path.name.lower() else "unknown_pt"
    else:
        with np.load(path) as cache:
            if "gt_poses" not in cache.files:
                raise Up1Error(f"gt cache has no gt_poses: {sorted(cache.files)}")
            poses = np.asarray(cache["gt_poses"], dtype=np.float64)
        lineage = "av_pyav"
    if poses.shape != (N_PAIRS_TOTAL, 6):
        raise Up1Error(f"gt poses have shape {poses.shape}, expected ({N_PAIRS_TOTAL}, 6)")
    return poses, lineage


def load_posenet() -> Any:
    """The frozen upstream PoseNet on CPU. Never MPS, never CUDA -- authority instrument."""
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().cpu()
    network.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def apply_offset(frames: np.ndarray, offset: float, *, realize_uint8: bool) -> np.ndarray:
    """Apply a global photometric offset to decoded frames.

    ``realize_uint8`` clips and rounds back to the uint8 lattice the receiver
    actually writes -- that is the REALIZED correction. With it False the offset
    stays in float, which measures the unrealized sensitivity curve (useful for
    locating an optimum finer than 1 LSB, which a receiver-side pre-round bias
    could reach).
    """
    out = frames.astype(np.float32, copy=True)
    if offset != 0.0:
        out += np.float32(offset)
    np.clip(out, 0.0, 255.0, out=out)
    if realize_uint8:
        # round-half-to-even, matching numpy/torch default rounding.
        out = np.rint(out)
    return out


def pose_vectors(posenet: Any, frames: np.ndarray) -> np.ndarray:
    """frames: (B, 2, H, W, C) float32 in [0,255] -> (B, 6) float64 pose."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    # upstream expects (b, t, c, h, w)
    tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
    with torch.inference_mode():
        prepared = posenet.preprocess_input(tensor)
        out = posenet(prepared)
        pose = out["pose"][..., :6]
    return pose.to(torch.float64).cpu().numpy()


def iter_batches(indices: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


@dataclass
class Checkpoint:
    """Resumable per-offset state. P0: every launch resumes from disk."""

    path: Path
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        state = cls(path=path)
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                state.rows[cls.key(row["offset"], row["pair"])] = row
        return state

    @staticmethod
    def key(offset: float, pair: int) -> str:
        return f"{offset:+.6f}|{pair:04d}"

    def has(self, offset: float, pair: int) -> bool:
        return self.key(offset, pair) in self.rows

    def append(self, row: dict[str, Any]) -> None:
        self.rows[self.key(row["offset"], row["pair"])] = row
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    import torch

    torch.set_num_threads(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    retained = config.output_dir / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    raw = open_raw(config.raw_path, verify_sha=config.verify_raw_sha)
    gt_poses, gt_lineage = load_gt_poses(config.gt_cache)
    pairs = select_pairs(config.pairs, config.seed)
    posenet = load_posenet()

    checkpoint = Checkpoint.load(config.output_dir / "rows.jsonl")
    started = time.time()

    for offset in config.offsets:
        pending = [int(p) for p in pairs if not checkpoint.has(offset, int(p))]
        if not pending:
            continue
        vectors: dict[int, np.ndarray] = {}
        for batch in iter_batches(np.asarray(pending, dtype=int), config.batch_pairs):
            stack = np.empty((len(batch), 2, FRAME_H, FRAME_W, FRAME_C), dtype=np.uint8)
            for slot, pair in enumerate(batch):
                stack[slot, 0] = raw[2 * int(pair)]
                stack[slot, 1] = raw[2 * int(pair) + 1]
            shifted = apply_offset(stack, offset, realize_uint8=config.realize_uint8)
            observed = pose_vectors(posenet, shifted)
            for slot, pair in enumerate(batch):
                pair = int(pair)
                delta = observed[slot] - gt_poses[pair]
                checkpoint.append(
                    {
                        "offset": float(offset),
                        "pair": pair,
                        "d_pose": float(np.mean(delta**2)),
                        "pose_observed": [float(v) for v in observed[slot]],
                        "realize_uint8": bool(config.realize_uint8),
                    }
                )
                vectors[pair] = observed[slot]
        if vectors:
            # ALWAYS KEEP THE PAYLOAD: retain every solved pose vector, not just the scalar.
            ordered = np.stack([vectors[p] for p in sorted(vectors)], axis=0)
            tag = f"{offset:+.6f}".replace("+", "p").replace("-", "m").replace(".", "_")
            np.save(retained / f"pose_vectors_offset_{tag}.npy", ordered)

    summary = summarize(checkpoint, pairs, config)
    summary["gt_lineage"] = gt_lineage
    summary["elapsed_seconds"] = time.time() - started
    (config.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def summarize(checkpoint: Checkpoint, pairs: np.ndarray, config: ProbeConfig) -> dict[str, Any]:
    wanted = {int(p) for p in pairs}
    per_offset: dict[float, list[float]] = {}
    for row in checkpoint.rows.values():
        if int(row["pair"]) not in wanted:
            continue
        per_offset.setdefault(float(row["offset"]), []).append(float(row["d_pose"]))

    table: list[dict[str, Any]] = []
    base_mean: float | None = None
    for offset in sorted(per_offset):
        values = np.asarray(per_offset[offset], dtype=np.float64)
        # mean-of-d_pose, NEVER mean-of-ratios (rt1: the sign flips).
        mean = float(values.mean())
        if offset == 0.0:
            base_mean = mean
        table.append(
            {
                "offset": offset,
                "n": int(values.size),
                "mean_d_pose": mean,
                "median_d_pose": float(np.median(values)),
                "pose_score_contribution": float(np.sqrt(10.0 * mean)),
            }
        )
    for row in table:
        if base_mean is not None and base_mean > 0.0:
            row["ratio_vs_base"] = row["mean_d_pose"] / base_mean
            row["delta_pose_score_vs_base"] = row["pose_score_contribution"] - float(
                np.sqrt(10.0 * base_mean)
            )

    best = min(table, key=lambda r: r["mean_d_pose"]) if table else None
    return {
        "schema": "ddm_up1_decode_axis_photometric_probe.v1",
        "axis": "[macOS-CPU advisory frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "pointer_archive_sha256": POINTER_ARCHIVE_SHA256,
        "raw_path": str(config.raw_path),
        "raw_sha256_expected_cpu_decode": CPU_DECODE_RAW_SHA256,
        "gt_cache": str(config.gt_cache),
        "gt_lineage_note": "ddm_pi2: GT decode lineage is 99.9960% of the advisory-vs-contest pose offset",
        "pairs_sampled": [int(p) for p in pairs],
        "n_pairs": len(pairs),
        "seed": config.seed,
        "sampling": "seeded_random_never_prefix",
        "realize_uint8": config.realize_uint8,
        "advisory_cpu_dpose_n600": ADVISORY_CPU_DPOSE_N600,
        "t4_cuda_dpose_n600": T4_CUDA_DPOSE_N600,
        "table": table,
        "best_offset": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="decoded 0.raw to probe")
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--offsets", type=str, default="0")
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--float-offset",
        action="store_true",
        help="keep the offset in float instead of realizing it on the uint8 lattice",
    )
    parser.add_argument("--no-verify-raw-sha", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ProbeConfig(
        raw_path=args.raw,
        gt_cache=args.gt_cache,
        output_dir=args.output_dir,
        pairs=args.pairs,
        seed=args.seed,
        offsets=parse_offsets(args.offsets),
        batch_pairs=args.batch_pairs,
        realize_uint8=not args.float_offset,
        verify_raw_sha=not args.no_verify_raw_sha,
        threads=args.threads,
    )
    summary = run_probe(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
