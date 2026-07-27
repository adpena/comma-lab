#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure rank-4 Laguerre/Bregman closure on real frozen-SegNet tiles.

This is a bounded macOS-CPU advisory measurement, not an archive score.  It
runs four real ``gt_n600`` frame1 images through the frozen SegNet, captures
the final affine-head input, computes its independent rank-four quotient, and
compares 20 deterministic held-out tiles against live argmax labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _entry in (REPO / "src",):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    affine_head_to_power_diagram,
    open_stored_npy_memmap,
)
from tac.canonical_equations.closed_scorer_variational_de_20260721 import (  # noqa: E402
    RATE_PRICE_EXACT,
    bregman_voronoi_labels,
    power_laguerre_labels,
    reachability_certificate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _native_f32_power(points: np.ndarray, sites: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    s = np.asarray(sites, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    dot = np.sum(values[:, None, :] * s[None, :, :], axis=2, dtype=np.float32)
    norm = np.sum(s * s, axis=1, dtype=np.float32)
    scores = np.asarray(np.float32(2.0) * dot + w[None, :] - norm[None, :], dtype=np.float32)
    return np.argmax(scores, axis=1).astype(np.int64, copy=False)


def _extract_live_and_quotient(model: Any, final_head: Any, basis: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    captured: list[Any] = []

    def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or inputs[0].ndim != 4:
            raise RuntimeError("final-head pre-hook geometry drift")
        captured.append(inputs[0].detach())

    hook = final_head.register_forward_pre_hook(capture)
    try:
        batch = torch.from_numpy(np.array(frame, copy=True, order="C")).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        with torch.inference_mode():
            logits = model(model.preprocess_input(batch))
        if len(captured) != 1:
            raise RuntimeError("expected exactly one captured final-head input")
        filters = torch.from_numpy(np.asarray(basis, dtype=np.float64).T.reshape(4, *final_head.weight.shape[1:])).to(
            dtype=captured[0].dtype
        )
        with torch.inference_mode():
            quotient = functional.conv2d(
                captured[0], filters, stride=final_head.stride, padding=final_head.padding,
                dilation=final_head.dilation, groups=final_head.groups,
            )
        return (
            np.ascontiguousarray(logits[0].cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(quotient[0].cpu().numpy(), dtype=np.float32),
        )
    finally:
        hook.remove()


def build_receipt(*, upstream: Path, gt_cache: Path, seed: int, heldout_tiles: int, tile_size: int) -> dict[str, Any]:
    import torch

    from tac.scorer import load_default_segnet

    if heldout_tiles < 20:
        raise ValueError("D1 requires at least 20 held-out tiles")
    if heldout_tiles % 4:
        raise ValueError("heldout_tiles must be divisible across four real frames")
    for required in (upstream / "modules.py", upstream / "frame_utils.py", upstream / "models/segnet.safetensors", gt_cache):
        if not required.is_file():
            raise FileNotFoundError(required)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_default_segnet(upstream, device="cpu")
    final_head = model.segmentation_head[0]
    frozen = affine_head_to_power_diagram(
        final_head.weight.detach().cpu().numpy(), final_head.bias.detach().cpu().numpy()
    )
    if frozen.target.rank != 4:
        raise RuntimeError(f"frozen final-head rank drift: {frozen.target.rank}")

    frames = open_stored_npy_memmap(gt_cache, "gt_f1")
    cached_labels = open_stored_npy_memmap(gt_cache, "lstars")
    if frames.shape[0] != 600 or cached_labels.shape != (600, 384, 512):
        raise RuntimeError("gt_n600 geometry drift")
    rng = np.random.default_rng(seed)
    frame_ids = np.sort(rng.choice(600, size=4, replace=False)).astype(int)
    tiles_per_frame = heldout_tiles // 4
    pixels = 0
    native_disagreements = 0
    generic_disagreements = 0
    bregman_disagreements = 0
    cache_disagreements = 0
    rows: list[dict[str, Any]] = []

    for frame_id in frame_ids:
        live_logits, quotient = _extract_live_and_quotient(
            model, final_head, frozen.quotient_basis, np.asarray(frames[frame_id])
        )
        live_labels = np.argmax(live_logits, axis=0).astype(np.int64, copy=False)
        points = np.moveaxis(quotient, 0, -1)
        logits_hwc = np.moveaxis(live_logits, 0, -1)
        origins: set[tuple[int, int]] = set()
        while len(origins) < tiles_per_frame:
            origins.add(
                (int(rng.integers(0, 384 - tile_size + 1)), int(rng.integers(0, 512 - tile_size + 1)))
            )
        for top, left in sorted(origins):
            sl = np.s_[top : top + tile_size, left : left + tile_size]
            q = points[sl].reshape(-1, 4)
            z = logits_hwc[sl].reshape(-1, 5)
            live = live_labels[sl].reshape(-1)
            cached = np.asarray(cached_labels[frame_id][sl], dtype=np.int64).reshape(-1)
            native = _native_f32_power(q, frozen.target.sites, frozen.target.weights)
            generic = power_laguerre_labels(
                q.astype(np.float64), frozen.target.sites, frozen.target.weights
            )
            bregman = bregman_voronoi_labels(z)
            row_pixels = int(live.size)
            counts = {
                "native_f32_power_vs_live": int(np.count_nonzero(native != live)),
                "generic_f64_power_vs_live": int(np.count_nonzero(generic != live)),
                "bregman_voronoi_vs_live": int(np.count_nonzero(bregman != live)),
                "cached_target_vs_live": int(np.count_nonzero(cached != live)),
            }
            pixels += row_pixels
            native_disagreements += counts["native_f32_power_vs_live"]
            generic_disagreements += counts["generic_f64_power_vs_live"]
            bregman_disagreements += counts["bregman_voronoi_vs_live"]
            cache_disagreements += counts["cached_target_vs_live"]
            rows.append({"frame_id": int(frame_id), "top": top, "left": left, "size": tile_size, "pixels": row_pixels, **counts})

    rates = {
        "native_f32_power_vs_live_argmax_disagreement_rate": native_disagreements / pixels,
        "generic_f64_power_vs_live_argmax_disagreement_rate": generic_disagreements / pixels,
        "bregman_voronoi_vs_live_argmax_disagreement_rate": bregman_disagreements / pixels,
        "cached_target_vs_live_argmax_disagreement_rate": cache_disagreements / pixels,
    }
    reachability = reachability_certificate().to_dict()
    return {
        "schema": "closed_scorer_variational_de_fidelity.v1",
        "written_at_utc": _utc(),
        "authority": "[macOS-CPU advisory real frozen-SegNet tiles; NON-PROMOTABLE]",
        "seed": seed,
        "pointer_moved": False,
        "score_claim_valid": False,
        "source_custody": {
            "git_sha_before_landing": _git_sha(),
            "gt_cache": str(gt_cache),
            "gt_cache_bytes": gt_cache.stat().st_size,
            "gt_cache_sha256": _sha256(gt_cache),
            "upstream_modules_sha256": _sha256(upstream / "modules.py"),
            "upstream_frame_utils_sha256": _sha256(upstream / "frame_utils.py"),
            "segnet_weights_sha256": _sha256(upstream / "models/segnet.safetensors"),
            "verification_tool_sha256": _sha256(Path(__file__).resolve()),
            "equation_module_sha256": _sha256(
                REPO / "src/tac/canonical_equations/closed_scorer_variational_de_20260721.py"
            ),
            "factorization_module_sha256": _sha256(
                REPO / "src/tac/boundary_math/power_diagram_witness.py"
            ),
            "torch_version": torch.__version__,
            "torch_threads": 1,
            "official_n600_batch_geometry": 32,
            "measurement_batch_geometry": 1,
            "batch_geometry_scope": "algebraic closure at captured live features; cached-target disagreement is diagnostic only",
        },
        "d1_fidelity": {
            "frame_ids": frame_ids.tolist(),
            "heldout_tiles": len(rows),
            "tile_size": tile_size,
            "heldout_pixels": pixels,
            **rates,
            "max_normalized_residual": max(
                rates["native_f32_power_vs_live_argmax_disagreement_rate"],
                rates["bregman_voronoi_vs_live_argmax_disagreement_rate"],
            ),
            "rows": rows,
        },
        "interpretation": {
            "confirmed": "rank-4 native-f32 Laguerre and negative-entropy Bregman cells reproduce the live frozen final-head argmax on measured tiles",
            "not_confirmed": "archive-feasible inverse image, decoder reachability, exact byte rate, contest score, or V9 model factorization",
            "v9_gauge_status": "GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED",
        },
        "d2_stationarity": {
            "objective_rate_price_exact": f"{RATE_PRICE_EXACT.numerator}/{RATE_PRICE_EXACT.denominator}",
            "objective_rate_price_float": float(RATE_PRICE_EXACT),
            "hard_cap_multiplier": "mu_B; distinct from the fixed objective rate price",
            "seg_differential": "viscosity/subgradient Hamilton-Jacobi force on Laguerre cell interfaces",
            "pose_differential": "SE(3) geodesic xi force with measured decoder pullback owed",
            "rate_differential": "entropy/MDL code-length gradient with exact-byte residual gate",
        },
        "d3_constrained": {
            **reachability,
            "numeric_infimum": None,
            "numeric_infimum_status": "REFUSED_UNKNOWN_DESCRIPTION_LANGUAGE_AND_NO_SUB015_BYTE_CLOSED_WITNESS",
            "s_floor_0_118_verdict": "REFUTE_AS_ESTABLISHED_EXACT_MINIMUM",
            "s_floor_0_118_retained_scope": "EMPIRICAL_ACHIEVER_RATE_REFERENCE_AT_177169_BYTES; OUTSIDE_154600_BYTE_CAP",
            "known_under_cap_advisory_incumbent": {
                "source": ".omx/research/einstein_kolmogorov_crux_v3_20260720.json",
                "archive_bytes": 91062,
                "score": 35.955425463668846,
                "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
            },
        },
        "d4_routing": {
            "einstein_kolmogorov_ultra.U1": "closed_scorer_viscosity_kkt_stationarity_v1",
            "einstein_kolmogorov_ultra.U2": "closed_scorer_archive_reachability_bound_v1 lower-bound relaxation",
            "einstein_kolmogorov_ultra.U3": "byte-closed S-star/reachability witness and exact replay",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--heldout-tiles", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if str(out).startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit("refusing transient durable evidence path")
    payload = build_receipt(
        upstream=args.upstream.resolve(), gt_cache=args.gt_cache.resolve(), seed=args.seed,
        heldout_tiles=args.heldout_tiles, tile_size=args.tile_size,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "d1_fidelity": payload["d1_fidelity"] | {"rows": "omitted"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
