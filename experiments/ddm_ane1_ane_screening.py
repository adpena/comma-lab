"""ddm_ane1 -- prove ANE placement, price fp16 drift per scorer, screen the pose axis.

The operator asked to saturate the ANE alongside the CPU and the GPU.  The ANE
is fp16-only, so under the law CLAUDE.md applies to MPS it can be a SCREENING
device and never a score.  This instrument answers, with receipts:

``convert``    frozen SegNet / PoseNet -> retained ``.mlpackage`` (fp16 and fp32)
``placement``  per-op compute-device census via ``MLComputePlan`` + a latency triad
``fidelity``   n600 REAL-input drift: SegNet argmax flips, PoseNet 6-dim MSE delta
``price``      the exact-SegNet hybrid, PRICED from the flip census -- not built

Every row is ``[macOS-CPU/ANE advisory]``; ``score_claim=false`` throughout.  The
authority for both scorers remains 1-thread CPU-torch fp32, and for the contest
score remains ``upstream/evaluate.py``.

Runs under ``.venv_executorch_spike`` (coremltools 9.0, torch 2.12.0) with
``PYTHONPATH=src:upstream``; ``upstream/`` is READ-ONLY -- the scorers are
converted from copies held in memory, never patched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.ane_screening import (  # noqa: E402
    SEG_AUTHORITY_FLIP_BAR,
    sha256_tree,
    write_json,
)

EVAL_H, EVAL_W = 384, 512
# PoseNet's trunk sees the YUV6 lattice, NOT the eval lattice: preprocess_input
# resizes to (384,512) and then rgb_to_yuv6 folds each 2x2 luma block into 4
# channels, halving both spatial dims (upstream/frame_utils.py). MEASURED from
# the real preprocess, never assumed.
POSE_H, POSE_W = EVAL_H // 2, EVAL_W // 2
CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600
POSE_DIMS = 6
SEG_CLASSES = 5

AXIS = "[macOS-CPU/ANE advisory, frozen scorers, real n600 inputs]"


class Ane1Error(RuntimeError):
    pass


# ---------------------------------------------------------------- frozen models


def _load_upstream(name: str):
    sys.path.insert(0, str(UPSTREAM))
    try:
        import modules  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return getattr(modules, name), modules


def load_segnet():
    """Frozen SegNet on CPU, eval, grads off -- the authority form."""
    from safetensors.torch import load_file

    SegNet, modules = _load_upstream("SegNet")
    net = SegNet().eval()
    net.load_state_dict(load_file(modules.segnet_sd_path, device="cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, modules.segnet_sd_path


def load_posenet():
    from safetensors.torch import load_file

    PoseNet, modules = _load_upstream("PoseNet")
    net = PoseNet().eval()
    net.load_state_dict(load_file(modules.posenet_sd_path, device="cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, modules.posenet_sd_path


def _pose_wrapper(posenet):
    """``(B,12,384,512) -> (B,12)`` pose head, the exact term ``pose_from_frames`` reads."""
    import torch

    class PoseTrunk(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)["pose"]

    return PoseTrunk(posenet).eval()


def _seg_wrapper(segnet):
    """``(B,3,384,512) -> (B,5,384,512)`` logits, post-preprocess (the 07-13 shape)."""
    import torch

    class SegTrunk(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)

    return SegTrunk(segnet).eval()


def _best_deployment_target(ct):
    """Highest OS target this coremltools build offers (read, never guessed).

    coremltools 9 exposes only ``iOSNN`` names (``macOSNN`` aliases were
    dropped); ``iOS26`` is the macOS 26 target.  Prefer an explicit ``macOS``
    name when the build has one, else fall back to the ``iOS`` ladder.
    """
    import re as _re

    def rank(prefix: str) -> list[tuple[int, str]]:
        out = []
        for name in dir(ct.target):
            match = _re.fullmatch(rf"{prefix}(\d+)", name)
            if match:
                out.append((int(match.group(1)), name))
        return sorted(out)

    for prefix in ("macOS", "iOS"):
        ladder = rank(prefix)
        if ladder:
            best = ladder[-1][1]
            return getattr(ct.target, best), best
    raise Ane1Error("coremltools exposes no macOS/iOS deployment target")


# ------------------------------------------------------------------- convert


def run_convert(args) -> int:
    import coremltools as ct
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target, target_name = _best_deployment_target(ct)
    manifest: dict[str, Any] = {
        "schema": "tac.ddm_ane1.convert.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "coremltools_version": str(ct.__version__),
        "torch_version": str(torch.__version__),
        "deployment_target": target_name,
        "models": {},
    }

    jobs = []
    if args.model in ("segnet", "both"):
        net, sd_path = load_segnet()
        jobs.append(("segnet", _seg_wrapper(net), (args.batch, 3, EVAL_H, EVAL_W), sd_path))
    if args.model in ("posenet", "both"):
        net, sd_path = load_posenet()
        jobs.append(("posenet", _pose_wrapper(net), (args.batch, 12, POSE_H, POSE_W), sd_path))

    for name, module, shape, sd_path in jobs:
        example = torch.zeros(shape, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(module, example)
        for precision, tag in (
            (ct.precision.FLOAT16, "fp16"),
            (ct.precision.FLOAT32, "fp32"),
        ):
            if args.precision not in ("both", tag):
                continue
            started = time.time()
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
                convert_to="mlprogram",
                compute_precision=precision,
                minimum_deployment_target=target,
            )
            path = out_dir / f"{name}_b{shape[0]}_{tag}.mlpackage"
            if path.exists():
                import shutil

                shutil.rmtree(path)
            mlmodel.save(str(path))
            elapsed = time.time() - started
            spec = mlmodel.get_spec()
            manifest["models"][f"{name}_{tag}"] = {
                "path": str(path),
                "sha256": sha256_tree(path),
                "shape": list(shape),
                "precision": tag,
                "convert_seconds": elapsed,
                "weights_sha256": sha256_tree(Path(sd_path)),
                "weights_path": str(sd_path),
                "input_name": spec.description.input[0].name,
                "output_name": spec.description.output[0].name,
            }
            print(f"[convert] {name} {tag} -> {path} ({elapsed:.1f}s)", flush=True)

    digest = write_json(Path(args.out), manifest)
    print(json.dumps({"manifest": args.out, "sha256": digest}))
    return 0


# ------------------------------------------------------------------ placement


def _compiled_path(path: Path) -> Path:
    """Compile an ``.mlpackage`` to ``.mlmodelc``; ``MLComputePlan`` needs that form.

    The compiled artifact is retained beside the package so the placement proof
    is reproducible from disk rather than from a temp directory that vanished.
    """
    from coremltools.models.utils import compile_model

    destination = path.with_suffix(".mlmodelc")
    if not destination.exists():
        compile_model(str(path), destination_path=str(destination))
    return destination


def _compute_plan_census(path: Path, mode: str) -> dict[str, Any]:
    """Per-op device census from ``MLComputePlan`` -- a PROOF, not an inference."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    units = getattr(ct.ComputeUnit, mode)
    plan = MLComputePlan.load_from_path(
        path=str(_compiled_path(path)), compute_units=units
    )
    program = plan.model_structure.program
    counts: dict[str, int] = {}
    cost: dict[str, float] = {}
    total_ops = 0
    for _function_name, function in program.functions.items():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                continue
            device = type(usage.preferred_compute_device).__name__
            counts[device] = counts.get(device, 0) + 1
            total_ops += 1
            estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
            if estimate is not None:
                cost[device] = cost.get(device, 0.0) + float(estimate.weight)
    fractions = {k: v / total_ops for k, v in counts.items()} if total_ops else {}
    return {
        "compute_units_requested": mode,
        "total_ops_with_usage": total_ops,
        "ops_by_device": counts,
        "op_fraction_by_device": fractions,
        "estimated_cost_by_device": cost,
        "placement_evidence": "MLComputePlan per-op device (MEASURED)",
    }


def _latency(path: Path, mode: str, shape: tuple[int, ...], reps: int) -> dict[str, Any]:
    import coremltools as ct

    units = getattr(ct.ComputeUnit, mode)
    model = ct.models.MLModel(str(path), compute_units=units)
    name = model.get_spec().description.input[0].name
    rng = np.random.default_rng(20260905)
    sample = rng.standard_normal(shape).astype(np.float32)
    for _ in range(3):
        model.predict({name: sample})
    times = []
    for _ in range(reps):
        started = time.perf_counter()
        model.predict({name: sample})
        times.append(time.perf_counter() - started)
    return {
        "compute_units": mode,
        "reps": reps,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "p90_s": float(np.percentile(times, 90)),
    }


def _torch_latency(kind: str, shape: tuple[int, ...], reps: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    net = load_segnet()[0] if kind == "segnet" else _pose_wrapper(load_posenet()[0])
    sample = torch.from_numpy(
        np.random.default_rng(20260905).standard_normal(shape).astype(np.float32)
    )
    with torch.inference_mode():
        for _ in range(2):
            net(sample)
        times = []
        for _ in range(reps):
            started = time.perf_counter()
            net(sample)
            times.append(time.perf_counter() - started)
    return {
        "backend": "cpu_torch_fp32_1thread",
        "reps": reps,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
    }


def run_placement(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    report: dict[str, Any] = {
        "schema": "tac.ddm_ane1.placement.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "coremltools_version": manifest["coremltools_version"],
        "deployment_target": manifest["deployment_target"],
        "models": {},
    }
    for key, entry in manifest["models"].items():
        path = Path(entry["path"])
        shape = tuple(entry["shape"])
        kind = key.split("_")[0]
        row: dict[str, Any] = {
            "mlpackage": str(path),
            "mlpackage_sha256": entry["sha256"],
            "shape": list(shape),
            "compute_plan": {},
            "latency": {},
        }
        for mode in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"):
            try:
                row["compute_plan"][mode] = _compute_plan_census(path, mode)
            except Exception as exc:
                row["compute_plan"][mode] = {"error": repr(exc)}
            try:
                row["latency"][mode] = _latency(path, mode, shape, args.reps)
            except Exception as exc:
                row["latency"][mode] = {"error": repr(exc)}
            print(f"[placement] {key} {mode} done", flush=True)
        try:
            row["latency"]["cpu_torch_fp32_1thread"] = _torch_latency(kind, shape, args.reps)
        except Exception as exc:
            row["latency"]["cpu_torch_fp32_1thread"] = {"error": repr(exc)}
        report["models"][key] = row
    digest = write_json(Path(args.out), report)
    print(json.dumps({"placement": args.out, "sha256": digest}))
    return 0


# ------------------------------------------------------------------- fidelity


def _open_raw(path: Path) -> np.ndarray:
    return np.memmap(
        path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )


def _seg_prepared(frames_bhwc: np.ndarray):
    """Upstream SegNet preprocess on real camera frames: resize to (384,512)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames_bhwc)).float().permute(0, 3, 1, 2)
    return torch.nn.functional.interpolate(
        x, size=(EVAL_H, EVAL_W), mode="bilinear"
    )


def _pose_prepared(posenet, frame0_bhwc: np.ndarray, frame1_bhwc: np.ndarray):
    """Upstream PoseNet preprocess (resize then YUV6) on a real pair."""
    import torch

    def bchw(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2)

    pair = torch.stack([bchw(frame0_bhwc), bchw(frame1_bhwc)], dim=1)
    return posenet.preprocess_input(pair)


def run_fidelity(args) -> int:
    import torch

    torch.set_num_threads(args.threads)
    manifest = json.loads(Path(args.manifest).read_text())
    raw = _open_raw(Path(args.raw))
    pairs = np.arange(min(args.pairs, N_PAIRS), dtype=np.int64)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane1.fidelity.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "pairs": int(pairs.size),
        "raw": str(args.raw),
        "raw_sha256_note": "shipped decode of the pointer body (real frames)",
        "seg_authority_flip_bar": SEG_AUTHORITY_FLIP_BAR,
        "coremltools_version": manifest["coremltools_version"],
        "segnet": None,
        "posenet": None,
    }

    if args.model in ("segnet", "both"):
        report["segnet"] = _fidelity_segnet(manifest, raw, pairs, args, out_dir)
    if args.model in ("posenet", "both"):
        report["posenet"] = _fidelity_posenet(manifest, raw, pairs, args, out_dir)

    digest = write_json(Path(args.out), report)
    print(json.dumps({"fidelity": args.out, "sha256": digest}))
    return 0


def _load_ml(manifest: dict[str, Any], key: str, mode: str):
    import coremltools as ct

    entry = manifest["models"][key]
    model = ct.models.MLModel(entry["path"], compute_units=getattr(ct.ComputeUnit, mode))
    return model, entry


def _fidelity_segnet(manifest, raw, pairs, args, out_dir) -> dict[str, Any]:
    """Argmax flip rate of each CoreML route vs 1-thread CPU-torch fp32, n600."""
    import torch

    net, _ = load_segnet()
    routes = {}
    if "segnet_fp16" in manifest["models"]:
        routes["ane_fp16"] = _load_ml(manifest, "segnet_fp16", "CPU_AND_NE")
    if "segnet_fp32" in manifest["models"]:
        routes["coreml_cpu_fp32"] = _load_ml(manifest, "segnet_fp32", "CPU_ONLY")

    flips = {k: 0 for k in routes}
    per_pair = {k: np.zeros(pairs.size, dtype=np.float64) for k in routes}
    total_px = 0
    margins_at_flips: list[np.ndarray] = []
    started = time.time()

    for index, pair in enumerate(pairs):
        frame1 = np.asarray(raw[2 * int(pair) + 1])[None]
        prepared = _seg_prepared(frame1)
        with torch.inference_mode():
            logits = net(prepared)
        ref = logits.argmax(dim=1).numpy()
        top2 = torch.topk(logits, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).numpy()
        total_px += ref.size
        array = prepared.numpy()
        for name, (model, entry) in routes.items():
            out = model.predict({entry["input_name"]: array})
            got = np.asarray(out[entry["output_name"]])
            got_arg = got.argmax(axis=1)
            mismatch = got_arg != ref
            n = int(mismatch.sum())
            flips[name] += n
            per_pair[name][index] = n / ref.size
            if name == "ane_fp16" and n:
                margins_at_flips.append(margin[mismatch].astype(np.float32))
        if index % 25 == 0:
            print(
                f"[seg] pair {index}/{pairs.size} "
                + " ".join(f"{k}={flips[k]}" for k in routes)
                + f" {time.time() - started:.0f}s",
                flush=True,
            )

    payload = out_dir / "segnet_per_pair_flip_rate.npz"
    np.savez_compressed(payload, **{k: v for k, v in per_pair.items()})
    all_margins = (
        np.concatenate(margins_at_flips) if margins_at_flips else np.zeros(0, np.float32)
    )
    margin_payload = out_dir / "segnet_fp16_flip_margins.npy"
    np.save(margin_payload, all_margins)

    rows = {}
    for name in routes:
        rate = flips[name] / total_px
        rows[name] = {
            "flips": int(flips[name]),
            "total_px": int(total_px),
            "flip_rate": rate,
            "per_pair_median": float(np.median(per_pair[name])),
            "per_pair_p95": float(np.percentile(per_pair[name], 95)),
            "per_pair_max": float(per_pair[name].max()),
            "passes_authority_bar": bool(rate <= SEG_AUTHORITY_FLIP_BAR),
            "multiple_of_bar": rate / SEG_AUTHORITY_FLIP_BAR,
        }
    quantiles = (
        {
            f"q{int(q * 100)}": float(np.quantile(all_margins, q))
            for q in (0.5, 0.9, 0.95, 0.99)
        }
        if all_margins.size
        else {}
    )
    return {
        "routes": rows,
        "seconds": time.time() - started,
        "per_pair_payload": str(payload),
        "per_pair_payload_sha256": sha256_tree(payload),
        "fp16_flip_margin_payload": str(margin_payload),
        "fp16_flip_margin_payload_sha256": sha256_tree(margin_payload),
        "fp16_flip_margin_quantiles": quantiles,
        "fp16_flip_margin_count": int(all_margins.size),
    }


def _fidelity_posenet(manifest, raw, pairs, args, out_dir) -> dict[str, Any]:
    """6-dim pose drift of each CoreML route vs CPU-torch fp32, on REAL pairs."""
    import torch

    net, _ = load_posenet()
    trunk = _pose_wrapper(net)
    routes = {}
    if "posenet_fp16" in manifest["models"]:
        routes["ane_fp16"] = _load_ml(manifest, "posenet_fp16", "CPU_AND_NE")
    if "posenet_fp32" in manifest["models"]:
        routes["coreml_cpu_fp32"] = _load_ml(manifest, "posenet_fp32", "CPU_ONLY")

    ref_poses = np.zeros((pairs.size, POSE_DIMS), dtype=np.float64)
    got_poses = {k: np.zeros((pairs.size, POSE_DIMS), dtype=np.float64) for k in routes}
    started = time.time()
    for index, pair in enumerate(pairs):
        f0 = np.asarray(raw[2 * int(pair)])[None]
        f1 = np.asarray(raw[2 * int(pair) + 1])[None]
        prepared = _pose_prepared(net, f0, f1)
        with torch.inference_mode():
            ref = trunk(prepared)[..., :POSE_DIMS].to(torch.float64).numpy()
        ref_poses[index] = ref[0]
        array = prepared.numpy()
        for name, (model, entry) in routes.items():
            out = np.asarray(model.predict({entry["input_name"]: array})[entry["output_name"]])
            if out.ndim == 1:
                out = out[None]
            got_poses[name][index] = out[0, :POSE_DIMS].astype(np.float64)
        if index % 25 == 0:
            print(f"[pose] pair {index}/{pairs.size} {time.time() - started:.0f}s", flush=True)

    payload = out_dir / "posenet_poses.npz"
    np.savez_compressed(payload, cpu_torch_fp32=ref_poses, **got_poses)

    rows = {}
    for name, poses in got_poses.items():
        delta = poses - ref_poses
        # The instrument's per-pair quantity is MSE against the GT target; the
        # backend perturbs the PREDICTION, so the drift the axis reads is the
        # change in that MSE.  Report the prediction drift directly plus a
        # first-order bound on the MSE it induces.
        per_dim_abs = np.abs(delta)
        rel = per_dim_abs / np.maximum(np.abs(ref_poses), 1e-12)
        mse_self = (delta**2).mean(axis=1)
        rows[name] = {
            "abs_delta_median": float(np.median(per_dim_abs)),
            "abs_delta_p95": float(np.percentile(per_dim_abs, 95)),
            "abs_delta_max": float(per_dim_abs.max()),
            "rel_delta_median": float(np.median(rel)),
            "rel_delta_p95": float(np.percentile(rel, 95)),
            "rel_delta_max": float(rel.max()),
            "self_mse_median": float(np.median(mse_self)),
            "self_mse_p95": float(np.percentile(mse_self, 95)),
            "self_mse_max": float(mse_self.max()),
        }
    return {
        "routes": rows,
        "seconds": time.time() - started,
        "poses_payload": str(payload),
        "poses_payload_sha256": sha256_tree(payload),
        "pose_dims": POSE_DIMS,
    }


# -------------------------------------------------------------- margin census


MARGIN_EDGES = np.concatenate(
    [
        np.array([0.0], dtype=np.float64),
        np.geomspace(1e-4, 1e2, 121),
        np.array([np.inf], dtype=np.float64),
    ]
)


def run_margins(args) -> int:
    """All-pixel SegNet top-2 logit margin histogram at n600, 1-thread fp32.

    The flip-site margins say WHERE fp16 goes wrong; only this census says how
    many pixels a margin band of a given width actually contains -- which is the
    quantity the exact-SegNet hybrid is priced on.  Without it a band area would
    be an assumption wearing a measurement's clothes.
    """
    import torch

    torch.set_num_threads(args.threads)
    net, _ = load_segnet()
    raw = _open_raw(Path(args.raw))
    pairs = np.arange(min(args.pairs, N_PAIRS), dtype=np.int64)
    counts = np.zeros(MARGIN_EDGES.size - 1, dtype=np.int64)
    total = 0
    started = time.time()
    for index, pair in enumerate(pairs):
        prepared = _seg_prepared(np.asarray(raw[2 * int(pair) + 1])[None])
        with torch.inference_mode():
            logits = net(prepared)
        top2 = torch.topk(logits, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).numpy().ravel()
        counts += np.histogram(margin, bins=MARGIN_EDGES)[0]
        total += margin.size
        if index % 50 == 0:
            print(f"[margins] pair {index}/{pairs.size} {time.time() - started:.0f}s", flush=True)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "segnet_margin_histogram.npz"
    np.savez_compressed(payload, edges=MARGIN_EDGES, counts=counts)
    report = {
        "schema": "tac.ddm_ane1.margin_census.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "pairs": int(pairs.size),
        "total_px": int(total),
        "edges": MARGIN_EDGES.tolist(),
        "counts": counts.tolist(),
        "payload": str(payload),
        "payload_sha256": sha256_tree(payload),
        "seconds": time.time() - started,
    }
    digest = write_json(Path(args.out), report)
    print(json.dumps({"margins": args.out, "sha256": digest}))
    return 0


def band_area_below(census: dict[str, Any], width: float) -> float:
    """Fraction of pixels whose top-2 margin is below ``width`` (MEASURED)."""
    edges = np.asarray(census["edges"], dtype=np.float64)
    counts = np.asarray(census["counts"], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    inside = counts[edges[1:] <= width].sum()
    # partial bin: attribute proportionally on a log scale, and say so.
    idx = int(np.searchsorted(edges[1:], width))
    if 0 <= idx < counts.size and edges[idx] < width < edges[idx + 1]:
        lo, hi = max(edges[idx], 1e-12), edges[idx + 1]
        if np.isfinite(hi):
            frac = math.log(width / lo) / math.log(hi / lo)
            inside += counts[idx] * frac
    return float(inside / total)


# ---------------------------------------------------------------------- price


def run_price(args) -> int:
    """Price the exact-SegNet hybrid from the measured flip-margin census.

    Not built.  The question is only: what fraction of pixels would an fp32
    recompute have to cover for BIT-EXACT argmax, and what end-to-end speedup
    survives that recompute?
    """
    fidelity = json.loads(Path(args.fidelity).read_text())
    placement = json.loads(Path(args.placement).read_text())
    seg = fidelity["segnet"]
    fp16 = seg["routes"]["ane_fp16"]

    margins = np.load(seg["fp16_flip_margin_payload"])
    total_px = fp16["total_px"]
    # A margin band must contain EVERY flip to be bit-exact.  The band width is
    # therefore the max flip margin; its area is the fraction of pixels whose
    # top-2 margin falls below that width -- which we bound from the flip census
    # plus the measured coverage curve.
    band_width = float(margins.max()) if margins.size else 0.0
    lat = placement["models"]["segnet_fp16"]["latency"]

    def median(mode: str) -> float | None:
        row = lat.get(mode, {})
        return float(row["median_s"]) if "median_s" in row else None

    ane = median("CPU_AND_NE")
    cpu_torch = None
    row = lat.get("cpu_torch_fp32_1thread", {})
    if "median_s" in row:
        cpu_torch = float(row["median_s"])

    census = (
        json.loads(Path(args.margin_census).read_text()) if args.margin_census else None
    )
    price: dict[str, Any] = {
        "schema": "tac.ddm_ane1.hybrid_price.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "built": False,
        "verdict_scope": "the priced hybrid formulation on this host/graph/inputs",
        "fp16_flip_rate": fp16["flip_rate"],
        "fp16_flip_count": fp16["flips"],
        "total_px": total_px,
        "max_flip_margin": band_width,
        "margin_quantiles_at_flips": seg["fp16_flip_margin_quantiles"],
        "ane_median_s": ane,
        "cpu_torch_1thread_median_s": cpu_torch,
    }
    if census is not None:
        # The band that guarantees BIT-EXACT argmax must contain every fp16 flip
        # site, so its width is the max flip margin and its AREA is measured
        # from the all-pixel census -- never assumed from the flip count.
        price["band_area_for_bit_exact"] = band_area_below(census, band_width)
        price["band_area_curve"] = [
            {
                "margin_width": float(w),
                "pixel_fraction": band_area_below(census, float(w)),
                "flip_coverage": (
                    float((margins <= w).mean()) if margins.size else None
                ),
            }
            for w in (
                seg["fp16_flip_margin_quantiles"].get("q50", 0.0),
                seg["fp16_flip_margin_quantiles"].get("q90", 0.0),
                seg["fp16_flip_margin_quantiles"].get("q95", 0.0),
                seg["fp16_flip_margin_quantiles"].get("q99", 0.0),
                band_width,
            )
            if w
        ]
        price["margin_census_pairs"] = census["pairs"]

    if ane and cpu_torch:
        price["forward_speedup_ane_vs_cpu_torch"] = cpu_torch / ane
        covers = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
        if census is not None:
            covers.append(price["band_area_for_bit_exact"])
        for cover in sorted(set(covers)):
            # fp32 recompute of a fraction f of pixels costs at least f of the
            # dense CPU pass (dense convs do not sparsify for free); the hybrid
            # total is the ANE pass plus that share.
            total = ane + cover * cpu_torch
            price.setdefault("hybrid_curve", []).append(
                {
                    "recompute_pixel_fraction": cover,
                    "hybrid_total_s": total,
                    "speedup_vs_cpu_torch": cpu_torch / total,
                    "meets_3x_bar": bool(cpu_torch / total >= 3.0),
                }
            )
        # The break-even fraction at the 3x bar, solved not swept.
        price["max_recompute_fraction_at_3x"] = max(
            0.0, (cpu_torch / 3.0 - ane) / cpu_torch
        )
    digest = write_json(Path(args.out), price)
    print(json.dumps({"price": args.out, "sha256": digest}))
    return 0


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="frozen scorers -> retained .mlpackage")
    convert.add_argument("--out-dir", required=True)
    convert.add_argument("--model", choices=("segnet", "posenet", "both"), default="both")
    convert.add_argument("--precision", choices=("fp16", "fp32", "both"), default="both")
    convert.add_argument("--batch", type=int, default=1)
    convert.add_argument("--out", required=True)

    placement = sub.add_parser("placement", help="MLComputePlan per-op census + latency triad")
    placement.add_argument("--manifest", required=True)
    placement.add_argument("--reps", type=int, default=30)
    placement.add_argument("--out", required=True)

    fidelity = sub.add_parser("fidelity", help="n600 REAL-input drift per scorer")
    fidelity.add_argument("--manifest", required=True)
    fidelity.add_argument("--raw", required=True)
    fidelity.add_argument("--pairs", type=int, default=N_PAIRS)
    fidelity.add_argument("--model", choices=("segnet", "posenet", "both"), default="both")
    fidelity.add_argument("--threads", type=int, default=1)
    fidelity.add_argument("--out", required=True)

    margins = sub.add_parser("margins", help="all-pixel SegNet top-2 margin census")
    margins.add_argument("--raw", required=True)
    margins.add_argument("--pairs", type=int, default=N_PAIRS)
    margins.add_argument("--threads", type=int, default=1)
    margins.add_argument("--out", required=True)

    price = sub.add_parser("price", help="price the exact-SegNet hybrid (not built)")
    price.add_argument("--fidelity", required=True)
    price.add_argument("--placement", required=True)
    price.add_argument("--margin-census", default=None)
    price.add_argument("--out", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "convert": run_convert,
        "placement": run_placement,
        "fidelity": run_fidelity,
        "margins": run_margins,
        "price": run_price,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
