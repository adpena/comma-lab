"""ddm_ane2 -- ENGINEER the fp16 drift out of the ANE scorer path.

ane1 proved placement (fp16 = 100% of ops on the ANE, fp32 = 0%) and then
measured both scorers' fp16 drift as INADMISSIBLE: SegNet 1.46x its argmax bar,
PoseNet 1,448x its pose axis.  Per the runtime-lift grant an unengineered drift
is a to-do, not a closure.  This instrument asks whether SELECTIVE PRECISION --
some ops fp16 on the ANE, the rest fp32 -- reaches either bar, and prices what
it costs.

``enumerate``    MIL compute-op inventory (the identity every split point uses)
``reference``    cache the 1-thread CPU-torch fp32 authority (argmax + margin, poses)
``sensitivity``  flip ONE contiguous op group to fp16 at a time vs the fp32 model
``ladder``       fp16 prefix / fp32 suffix at k = last {1,2,4,...} compute ops
``selective``    hold a named minimal op set at fp32, everything else fp16
``hybrid``       REALIZED crop-batched fp32 recompute over the fp16 margin band

Every row is ``[macOS-CPU/ANE advisory]``; ``score_claim=false`` throughout.
Authority for both scorers is 1-thread CPU-torch fp32; for the contest score it
remains ``upstream/evaluate.py``.  ``upstream/`` is READ-ONLY: the scorers are
converted from copies held in memory, never patched.

Runs under ``.venv_executorch_spike`` (coremltools 9.0, torch 2.12.0) with
``PYTHONPATH=src:upstream``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.ane_precision import (
    POSE_D_POSE_ADVISORY_BASE,
    POSE_D_POSE_T4_EXACT,
    POSE_PER_DIM_TOLERANCE,
    AnePrecisionError,
    assert_op_sequence_stable,
    compute_op_names,
    dilate_bool,
    fixed_crop_boxes,
    group_ranges,
    hybrid_speedup,
    margin_band_mask,
    occupied_tiles,
    pose_drift_verdict,
    seg_flip_verdict,
    selective_fp32_names,
    selector_from_names,
    split_backend_name,
    split_fp16_names,
)
from tac.ane_screening import SEG_AUTHORITY_FLIP_BAR, sha256_tree, write_json

EVAL_H, EVAL_W = 384, 512
POSE_H, POSE_W = EVAL_H // 2, EVAL_W // 2
CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600
POSE_DIMS = 6

AXIS = "[macOS-CPU/ANE advisory, frozen scorers, real n600 inputs]"

MODEL_SHAPES = {
    "segnet": (1, 3, EVAL_H, EVAL_W),
    "posenet": (1, 12, POSE_H, POSE_W),
}


class Ane2Error(RuntimeError):
    pass


# ---------------------------------------------------------------- frozen models


def _load_upstream(name: str):
    sys.path.insert(0, str(UPSTREAM))
    try:
        import modules  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return getattr(modules, name), modules


def _traced(model: str):
    """``(traced_module, shape, weights_path)`` for one scorer -- the authority form."""
    import torch
    from safetensors.torch import load_file

    if model == "segnet":
        SegNet, modules = _load_upstream("SegNet")
        net = SegNet().eval()
        net.load_state_dict(load_file(modules.segnet_sd_path, device="cpu"))
        sd_path = modules.segnet_sd_path

        class Trunk(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.net = inner

            def forward(self, x):
                return self.net(x)

    elif model == "posenet":
        PoseNet, modules = _load_upstream("PoseNet")
        net = PoseNet().eval()
        net.load_state_dict(load_file(modules.posenet_sd_path, device="cpu"))
        sd_path = modules.posenet_sd_path

        class Trunk(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.net = inner

            def forward(self, x):
                return self.net(x)["pose"]

    else:
        raise Ane2Error(f"unknown model {model!r}")

    for parameter in net.parameters():
        parameter.requires_grad_(False)
    module = Trunk(net).eval()
    shape = MODEL_SHAPES[model]
    with torch.no_grad():
        traced = torch.jit.trace(module, torch.zeros(shape, dtype=torch.float32))
    return traced, shape, Path(sd_path)


def _torch_net(model: str):
    """The live torch module for the authority reference (not traced)."""
    from safetensors.torch import load_file

    if model == "segnet":
        SegNet, modules = _load_upstream("SegNet")
        net = SegNet().eval()
        net.load_state_dict(load_file(modules.segnet_sd_path, device="cpu"))
    else:
        PoseNet, modules = _load_upstream("PoseNet")
        net = PoseNet().eval()
        net.load_state_dict(load_file(modules.posenet_sd_path, device="cpu"))
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def _target(ct):
    import re

    ladder = sorted(
        (int(m.group(1)), name)
        for name in dir(ct.target)
        if (m := re.fullmatch(r"iOS(\d+)", name))
    )
    if not ladder:
        raise Ane2Error("coremltools exposes no iOS deployment target")
    return getattr(ct.target, ladder[-1][1]), ladder[-1][1]


# ------------------------------------------------------------- mixed conversion


def _convert_mixed(traced, shape, fp16_names, out_path: Path, expected_ops=None):
    """Convert with exactly ``fp16_names`` at fp16; PROVE the op sequence first.

    Returns ``(mlmodel, observed_compute_names, seconds, raw_records)``.  The
    op-sequence assertion is what makes an ordinal split reproducible: if the
    trace emitted a different graph this raises rather than silently
    relabelling every rung.
    """
    import coremltools as ct

    target, _ = _target(ct)
    selector = selector_from_names(fp16_names)
    started = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=selector),
        minimum_deployment_target=target,
    )
    elapsed = time.time() - started
    records = list(selector.observed)
    observed = compute_op_names(records)
    if expected_ops is not None:
        assert_op_sequence_stable(expected_ops, observed, context=str(out_path.name))
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return mlmodel, observed, elapsed, records


def _io_names(path: Path) -> tuple[str, str]:
    import coremltools as ct

    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    return spec.description.input[0].name, spec.description.output[0].name


def _compiled(path: Path) -> Path:
    """Compile to ``.mlmodelc`` (``MLComputePlan`` needs that form), never stale.

    The compiled tree is retained beside the package so a placement proof is
    reproducible from disk.  It is rebuilt whenever the package is newer: a
    stale ``.mlmodelc`` would silently prove the placement of a DIFFERENT graph.
    """
    import shutil

    from coremltools.models.utils import compile_model

    destination = path.with_suffix(".mlmodelc")
    if destination.exists():
        if destination.stat().st_mtime >= path.stat().st_mtime:
            return destination
        shutil.rmtree(destination)
    compile_model(str(path), destination_path=str(destination))
    return destination


def _placement(path: Path, mode: str) -> dict[str, Any]:
    """Per-op device census -- a PROOF (MLComputePlan), never inferred from latency."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        path=str(_compiled(path)), compute_units=getattr(ct.ComputeUnit, mode)
    )
    counts: dict[str, int] = {}
    total = 0
    for _name, function in plan.model_structure.program.functions.items():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                continue
            device = type(usage.preferred_compute_device).__name__
            counts[device] = counts.get(device, 0) + 1
            total += 1
    ane = counts.get("MLNeuralEngineComputeDevice", 0)
    return {
        "compute_units_requested": mode,
        "total_ops_with_usage": total,
        "ops_by_device": counts,
        "ane_ops": ane,
        "ane_op_fraction": (ane / total) if total else 0.0,
        "placement_evidence": "MLComputePlan per-op device (MEASURED)",
    }


def _latency(path: Path, mode: str, shape, reps: int) -> dict[str, Any]:
    import coremltools as ct

    model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, mode))
    name = model.get_spec().description.input[0].name
    sample = np.random.default_rng(20260905).standard_normal(shape).astype(np.float32)
    for _ in range(3):
        model.predict({name: sample})
    times = []
    for _ in range(reps):
        started = time.perf_counter()
        model.predict({name: sample})
        times.append(time.perf_counter() - started)
    return {
        "compute_units": mode,
        "reps": int(reps),
        "median_ms": float(np.median(times) * 1e3),
        "min_ms": float(np.min(times) * 1e3),
        "p90_ms": float(np.percentile(times, 90) * 1e3),
    }


# ------------------------------------------------------------------ real inputs


class FrameSource:
    """Real n600 frames from either a shipped-body decode or the GT cache.

    ane1 read ``.../ddm_to1/.../inflated/0.raw`` -- a decode of the pointer body
    that no longer exists on disk.  ``gt_n600.npz`` is the durable, PyAV-decoded
    GT cache the born trainer pins, so this arm reads THAT and re-measures its
    own all-fp16 endpoint on it.  Re-measuring the endpoint is not redundancy:
    it makes every rung of the ladder comparable to a baseline on the SAME
    inputs, and it happens to answer ane1's owed GT-frame control.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.suffix == ".npz":
            self.kind = "gt_npz"
            self._data = np.load(self.path, mmap_mode="r")
            if "gt_f0" not in self._data or "gt_f1" not in self._data:
                raise Ane2Error(f"{self.path} has no gt_f0/gt_f1 arrays")
            self.pairs_available = int(self._data["gt_f0"].shape[0])
        else:
            self.kind = "raw_decode"
            self._data = np.memmap(
                self.path,
                dtype=np.uint8,
                mode="r",
                shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3),
            )
            self.pairs_available = N_PAIRS

    def frame0(self, pair: int) -> np.ndarray:
        if self.kind == "gt_npz":
            return np.asarray(self._data["gt_f0"][int(pair)])[None]
        return np.asarray(self._data[2 * int(pair)])[None]

    def frame1(self, pair: int) -> np.ndarray:
        if self.kind == "gt_npz":
            return np.asarray(self._data["gt_f1"][int(pair)])[None]
        return np.asarray(self._data[2 * int(pair) + 1])[None]

    def describe(self) -> dict[str, Any]:
        return {"path": str(self.path), "kind": self.kind, "pairs": self.pairs_available}


def _open_raw(path: Path) -> FrameSource:
    return FrameSource(path)


def _pairs(count: int) -> np.ndarray:
    """STRATIFIED sample of the 600 pairs -- never a prefix ([[m88]])."""
    if count >= N_PAIRS:
        return np.arange(N_PAIRS, dtype=np.int64)
    if count <= 0:
        raise Ane2Error(f"pair count must be positive, got {count}")
    stride = N_PAIRS / count
    return np.unique((np.arange(count) * stride).astype(np.int64))


def _seg_prepared(frame_bhwc: np.ndarray):
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frame_bhwc)).float().permute(0, 3, 1, 2)
    return torch.nn.functional.interpolate(x, size=(EVAL_H, EVAL_W), mode="bilinear")


def _pose_prepared(posenet, f0: np.ndarray, f1: np.ndarray):
    import torch

    def bchw(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float().permute(0, 3, 1, 2)

    return posenet.preprocess_input(torch.stack([bchw(f0), bchw(f1)], dim=1))


# ------------------------------------------------------------------- reference


def run_reference(args) -> int:
    """Cache the CPU-torch fp32 authority so every rung is scored against ONE reference."""
    import torch

    torch.set_num_threads(args.threads)
    raw = _open_raw(Path(args.raw))
    pairs = _pairs(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.reference.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "pairs": [int(p) for p in pairs],
        "pair_count": int(pairs.size),
        "sampling": "stratified stride over 600 (never a prefix)",
        "frames": raw.describe(),
        "threads": int(args.threads),
    }

    if args.model in ("segnet", "both"):
        net = _torch_net("segnet")
        argmax = np.zeros((pairs.size, EVAL_H, EVAL_W), dtype=np.uint8)
        margin = np.zeros((pairs.size, EVAL_H, EVAL_W), dtype=np.float32)
        started = time.time()
        for index, pair in enumerate(pairs):
            prepared = _seg_prepared(raw.frame1(pair))
            with torch.inference_mode():
                logits = net(prepared)
            argmax[index] = logits.argmax(dim=1).numpy()[0].astype(np.uint8)
            top2 = torch.topk(logits, 2, dim=1).values
            margin[index] = (top2[:, 0] - top2[:, 1]).numpy()[0]
            if index % 50 == 0:
                print(f"[ref seg] {index}/{pairs.size} {time.time() - started:.0f}s", flush=True)
        seg_path = out_dir / "reference_segnet.npz"
        np.savez(seg_path, argmax=argmax, margin=margin, pairs=pairs)
        report["segnet"] = {
            "payload": str(seg_path),
            "sha256": sha256_tree(seg_path),
            "seconds": time.time() - started,
            "total_px": int(argmax.size),
            "median_forward_ms": (time.time() - started) / pairs.size * 1e3,
        }

    if args.model in ("posenet", "both"):
        net = _torch_net("posenet")
        poses = np.zeros((pairs.size, POSE_DIMS), dtype=np.float64)
        prepared_store = np.zeros((pairs.size, 12, POSE_H, POSE_W), dtype=np.float32)
        started = time.time()
        for index, pair in enumerate(pairs):
            prepared = _pose_prepared(net, raw.frame0(pair), raw.frame1(pair))
            with torch.inference_mode():
                poses[index] = net(prepared)["pose"][0, :POSE_DIMS].to(torch.float64).numpy()
            prepared_store[index] = prepared.numpy()[0]
            if index % 50 == 0:
                print(f"[ref pose] {index}/{pairs.size} {time.time() - started:.0f}s", flush=True)
        pose_path = out_dir / "reference_posenet.npz"
        np.savez(pose_path, poses=poses, prepared=prepared_store, pairs=pairs)
        report["posenet"] = {
            "payload": str(pose_path),
            "sha256": sha256_tree(pose_path),
            "seconds": time.time() - started,
            "median_forward_ms": (time.time() - started) / pairs.size * 1e3,
            "prepared_cached": True,
        }

    digest = write_json(Path(args.out), report)
    print(json.dumps({"reference": args.out, "sha256": digest}))
    return 0


def _load_reference(path: Path, eval_pairs: int | None = None) -> dict[str, Any]:
    """Materialize the cached authority ONCE, optionally SUBSAMPLED stratified.

    ``np.load`` ignores ``mmap_mode`` for ``.npz``, so indexing the archive
    inside an evaluation loop would re-read the whole array per pair.  Every
    array is read once here; ``eval_pairs`` then takes a stratified stride (never
    a prefix, [[m88]]) so a cheap n120 rung and an n600 finalist score against
    the SAME reference rows.
    """
    with np.load(path) as data:
        store = {key: np.asarray(data[key]) for key in data.files}
    total = int(store["pairs"].size)
    if eval_pairs is None or eval_pairs >= total:
        store["selected_rows"] = np.arange(total, dtype=np.int64)
        return store
    if eval_pairs <= 0:
        raise Ane2Error(f"eval pairs must be positive, got {eval_pairs}")
    rows = np.unique((np.arange(eval_pairs) * (total / eval_pairs)).astype(np.int64))
    for key, value in list(store.items()):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total:
            store[key] = value[rows]
    store["selected_rows"] = rows
    return store


# ------------------------------------------------------------------- evaluation


def _eval_segnet(
    package: Path, mode: str, raw, pairs, ref, payload_path: Path | None = None
) -> dict[str, Any]:
    """Argmax flips of one CoreML package against the cached CPU-torch reference."""
    import coremltools as ct

    model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, mode))
    in_name, out_name = _io_names(package)
    ref_argmax = ref["argmax"]
    if ref_argmax.shape[0] != pairs.size:
        raise Ane2Error(
            f"reference has {ref_argmax.shape[0]} rows but {pairs.size} pairs were asked for"
        )
    flips = 0
    total = 0
    per_pair = np.zeros(pairs.size, dtype=np.float64)
    started = time.time()
    for index, pair in enumerate(pairs):
        prepared = _seg_prepared(raw.frame1(pair)).numpy()
        got = np.asarray(model.predict({in_name: prepared})[out_name]).argmax(axis=1)[0]
        mismatch = got.astype(np.uint8) != np.asarray(ref_argmax[index])
        n = int(mismatch.sum())
        flips += n
        total += mismatch.size
        per_pair[index] = n / mismatch.size
    row = seg_flip_verdict(flips, total, SEG_AUTHORITY_FLIP_BAR, per_pair)
    row["eval_seconds"] = time.time() - started
    row["compute_units"] = mode
    if payload_path is not None:
        # ALWAYS KEEP THE PAYLOAD: the per-pair flip rates are the measurement,
        # the summary quantiles are only a reading of them.
        np.save(payload_path, per_pair)
        row["per_pair_payload"] = str(payload_path)
        row["per_pair_payload_sha256"] = sha256_tree(payload_path)
    return row


def _eval_posenet(
    package: Path, mode: str, ref, payload_path: Path | None = None
) -> dict[str, Any]:
    import coremltools as ct

    model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, mode))
    in_name, out_name = _io_names(package)
    prepared = ref["prepared"]
    reference = np.asarray(ref["poses"])
    got = np.zeros_like(reference)
    started = time.time()
    for index in range(reference.shape[0]):
        out = np.asarray(
            model.predict({in_name: np.asarray(prepared[index])[None]})[out_name]
        )
        if out.ndim == 1:
            out = out[None]
        got[index] = out[0, :POSE_DIMS].astype(np.float64)
    row = pose_drift_verdict(reference, got)
    row["advisory_base_multiple"] = float(
        row["self_mse_median"] / POSE_D_POSE_ADVISORY_BASE
    )
    row["eval_seconds"] = time.time() - started
    row["compute_units"] = mode
    if payload_path is not None:
        np.save(payload_path, got)
        row["poses_payload"] = str(payload_path)
        row["poses_payload_sha256"] = sha256_tree(payload_path)
    return row, got


# ------------------------------------------------------------------ enumerate


def run_enumerate(args) -> int:
    """MIL compute-op inventory: the identity every later split point indexes."""
    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.enumerate.v1",
        "axis": AXIS,
        "score_claim": False,
        "models": {},
    }
    for model in ([args.model] if args.model != "both" else ["segnet", "posenet"]):
        traced, shape, weights = _traced(model)
        out_path = Path(args.out_dir) / f"{model}_enumerate_probe.mlpackage"
        _mlmodel, observed, seconds, records = _convert_mixed(
            traced, shape, frozenset(), out_path
        )
        from collections import Counter

        types_by_name = {name: op_type for op_type, name in records}
        ordered = [
            {"ordinal": index, "name": name, "op_type": types_by_name.get(name, "?")}
            for index, name in enumerate(observed)
        ]
        report["models"][model] = {
            "shape": list(shape),
            "weights_sha256": sha256_tree(weights),
            "compute_ops": len(observed),
            "const_ops": len(records) - len(observed),
            "convert_seconds": seconds,
            "probe_mlpackage": str(out_path),
            "ops": ordered,
            "op_type_counts": dict(Counter(op["op_type"] for op in ordered)),
        }
        print(f"[enumerate] {model}: {len(observed)} compute ops", flush=True)
    digest = write_json(Path(args.out), report)
    print(json.dumps({"enumerate": args.out, "sha256": digest}))
    return 0


def _enumerated(path: Path, model: str) -> tuple[str, ...]:
    report = json.loads(Path(path).read_text())
    entry = report["models"].get(model)
    if entry is None:
        raise Ane2Error(f"enumerate report has no {model!r}")
    return tuple(op["name"] for op in entry["ops"])


# ---------------------------------------------------------------- sensitivity


def run_sensitivity(args) -> int:
    """Flip ONE contiguous op group to fp16 at a time -- which ops carry the drift?"""
    import torch

    torch.set_num_threads(args.threads)
    names = _enumerated(Path(args.enumerate_json), args.model)
    ranges = group_ranges(len(names), args.groups)
    traced, shape, _weights = _traced(args.model)
    raw = _open_raw(Path(args.raw))
    ref = _load_reference(Path(args.reference), args.eval_pairs)
    pairs = np.asarray(ref["pairs"])
    out_dir = Path(args.out_dir)

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.sensitivity.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "model": args.model,
        "compute_ops": len(names),
        "groups": args.groups,
        "pairs": int(pairs.size),
        "reference": str(args.reference),
        "seg_bar": SEG_AUTHORITY_FLIP_BAR,
        "pose_per_dim_tolerance": POSE_PER_DIM_TOLERANCE,
        "rows": [],
    }

    for gi, (lo, hi) in enumerate(ranges):
        fp16 = frozenset(names[lo:hi])
        package = out_dir / f"{args.model}_sens_g{gi:02d}.mlpackage"
        _model, observed, convert_s, _rec = _convert_mixed(
            traced, shape, fp16, package, expected_ops=names
        )
        row: dict[str, Any] = {
            "group": gi,
            "ordinal_lo": lo,
            "ordinal_hi": hi,
            "ops_in_group": hi - lo,
            "first_op": names[lo],
            "last_op": names[hi - 1],
            "convert_seconds": convert_s,
            "mlpackage": str(package),
            "mlpackage_sha256": sha256_tree(package),
        }
        payload = out_dir / f"{args.model}_sens_g{gi:02d}_payload.npy"
        if args.model == "segnet":
            row["fidelity"] = _eval_segnet(
                package, args.compute_units, raw, pairs, ref, payload
            )
        else:
            verdict, _got = _eval_posenet(package, args.compute_units, ref, payload)
            row["fidelity"] = verdict
        if args.placement:
            row["placement"] = _placement(package, "CPU_AND_NE")
        report["rows"].append(row)
        key = (
            f"flip_rate={row['fidelity']['flip_rate']:.6e}"
            if args.model == "segnet"
            else f"self_mse={row['fidelity']['self_mse_median']:.6e}"
        )
        print(f"[sens] g{gi:02d} [{lo},{hi}) {key} ({convert_s:.1f}s conv)", flush=True)
        if not args.keep_packages:
            import shutil

            shutil.rmtree(package, ignore_errors=True)
            shutil.rmtree(package.with_suffix(".mlmodelc"), ignore_errors=True)
            row["mlpackage"] = "deleted after measurement (sha256 retained)"

    digest = write_json(Path(args.out), report)
    print(json.dumps({"sensitivity": args.out, "sha256": digest}))
    return 0


# --------------------------------------------------------------------- ladder


def run_ladder(args) -> int:
    """fp16 prefix on the ANE + fp32 suffix: the split-point ladder."""
    import torch

    torch.set_num_threads(args.threads)
    names = _enumerated(Path(args.enumerate_json), args.model)
    traced, shape, _weights = _traced(args.model)
    raw = _open_raw(Path(args.raw))
    ref = _load_reference(Path(args.reference), args.eval_pairs)
    pairs = np.asarray(ref["pairs"])
    out_dir = Path(args.out_dir)

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.ladder.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "model": args.model,
        "compute_ops": len(names),
        "pairs": int(pairs.size),
        "reference": str(args.reference),
        "seg_bar": SEG_AUTHORITY_FLIP_BAR,
        "pose_d_pose_t4_exact": POSE_D_POSE_T4_EXACT,
        "pose_per_dim_tolerance": POSE_PER_DIM_TOLERANCE,
        "rungs": [],
    }

    for k in args.splits:
        if k > len(names):
            print(f"[ladder] k={k} exceeds {len(names)} compute ops -- skipped", flush=True)
            continue
        fp16 = split_fp16_names(names, k)
        backend = split_backend_name(k)
        package = out_dir / f"{args.model}_{backend}.mlpackage"
        _model, _observed, convert_s, _rec = _convert_mixed(
            traced, shape, fp16, package, expected_ops=names
        )
        rung: dict[str, Any] = {
            "k_fp32_tail_ops": int(k),
            "backend": backend,
            "fp16_ops": len(fp16),
            "fp16_op_fraction": len(fp16) / len(names),
            "convert_seconds": convert_s,
            "mlpackage": str(package),
            "mlpackage_sha256": sha256_tree(package),
            "placement": {},
            "latency": {},
        }
        for mode in args.modes:
            try:
                rung["placement"][mode] = _placement(package, mode)
            except Exception as exc:  # placement failure is data, not a crash
                rung["placement"][mode] = {"error": repr(exc)}
            try:
                rung["latency"][mode] = _latency(package, mode, shape, args.reps)
            except Exception as exc:
                rung["latency"][mode] = {"error": repr(exc)}
        payload = out_dir / f"{args.model}_{backend}_payload.npy"
        if args.model == "segnet":
            rung["fidelity"] = _eval_segnet(
                package, args.compute_units, raw, pairs, ref, payload
            )
            head = f"flips={rung['fidelity']['flips']} rate={rung['fidelity']['flip_rate']:.4e}"
        else:
            verdict, _got = _eval_posenet(package, args.compute_units, ref, payload)
            rung["fidelity"] = verdict
            head = f"self_mse={verdict['self_mse_median']:.4e}"
        ane = rung["placement"].get("CPU_AND_NE", {}).get("ane_op_fraction", 0.0)
        lat = rung["latency"].get("CPU_AND_NE", {}).get("median_ms", float("nan"))
        report["rungs"].append(rung)
        print(f"[ladder] k={k} ane={ane:.1%} {lat:.2f}ms {head}", flush=True)

    digest = write_json(Path(args.out), report)
    print(json.dumps({"ladder": args.out, "sha256": digest}))
    return 0


# ------------------------------------------------------------------ selective


def run_selective(args) -> int:
    """Hold a NAMED minimal op set at fp32; everything else fp16 on the ANE."""
    import torch

    torch.set_num_threads(args.threads)
    names = _enumerated(Path(args.enumerate_json), args.model)
    traced, shape, _weights = _traced(args.model)
    raw = _open_raw(Path(args.raw))
    ref = _load_reference(Path(args.reference), args.eval_pairs)
    pairs = np.asarray(ref["pairs"])
    out_dir = Path(args.out_dir)

    sets: list[tuple[str, list[int]]] = []
    for spec in args.fp32_set:
        label, _, body = spec.partition("=")
        ordinals: list[int] = []
        for chunk in body.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                lo, _, hi = chunk.partition(":")
                ordinals.extend(range(int(lo), int(hi)))
            else:
                ordinals.append(int(chunk))
        sets.append((label, ordinals))

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.selective.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "model": args.model,
        "compute_ops": len(names),
        "pairs": int(pairs.size),
        "seg_bar": SEG_AUTHORITY_FLIP_BAR,
        "pose_per_dim_tolerance": POSE_PER_DIM_TOLERANCE,
        "rows": [],
    }
    for label, ordinals in sets:
        fp16 = selective_fp32_names(names, ordinals)
        package = out_dir / f"{args.model}_sel_{label}.mlpackage"
        _model, _observed, convert_s, _rec = _convert_mixed(
            traced, shape, fp16, package, expected_ops=names
        )
        row: dict[str, Any] = {
            "label": label,
            "fp32_ordinals": sorted(set(ordinals)),
            "fp32_ops": len(set(ordinals)),
            "fp16_ops": len(fp16),
            "fp16_op_fraction": len(fp16) / len(names),
            "fp32_op_names": [names[o] for o in sorted(set(ordinals))],
            "convert_seconds": convert_s,
            "mlpackage": str(package),
            "mlpackage_sha256": sha256_tree(package),
            "placement": {},
            "latency": {},
        }
        for mode in args.modes:
            try:
                row["placement"][mode] = _placement(package, mode)
            except Exception as exc:
                row["placement"][mode] = {"error": repr(exc)}
            try:
                row["latency"][mode] = _latency(package, mode, shape, args.reps)
            except Exception as exc:
                row["latency"][mode] = {"error": repr(exc)}
        payload = out_dir / f"{args.model}_sel_{label}_payload.npy"
        if args.model == "segnet":
            row["fidelity"] = _eval_segnet(
                package, args.compute_units, raw, pairs, ref, payload
            )
            head = f"rate={row['fidelity']['flip_rate']:.4e}"
        else:
            verdict, _got = _eval_posenet(package, args.compute_units, ref, payload)
            row["fidelity"] = verdict
            head = f"self_mse={verdict['self_mse_median']:.4e}"
        report["rows"].append(row)
        ane = row["placement"].get("CPU_AND_NE", {}).get("ane_op_fraction", 0.0)
        print(f"[selective] {label} ane={ane:.1%} {head}", flush=True)

    digest = write_json(Path(args.out), report)
    print(json.dumps({"selective": args.out, "sha256": digest}))
    return 0


# --------------------------------------------------------------------- hybrid


def run_hybrid(args) -> int:
    """REALIZED exact-argmax hybrid: fp16 ANE dense pass + crop-batched fp32 recompute.

    Two things make this a REALIZATION rather than a price:

    * the band is selected from the **fp16** top-2 margin -- the quantity an
      inference-time decoder actually has.  Selecting on the fp32 reference
      margin would be a fake: the runtime never sees it.
    * the fp32 recompute runs on real crops through a real CoreML model
      converted for the crop's exact shape, so its cost is measured wall-clock,
      not a proportional-area model.  ane1's price assumed proportionality; the
      2026-07-13 lane measured the opposite.

    A U-Net crop is NOT a window on the full-frame result: EfficientNet-B2's 23
    squeeze-excitation blocks average over the whole input, so the crop's global
    gates differ.  ``crop_vs_fullframe_argmax_disagreements`` measures exactly
    that, and it is reported whether or not the timing passes.
    """
    import coremltools as ct
    import torch

    torch.set_num_threads(args.threads)
    raw = _open_raw(Path(args.raw))
    ref = _load_reference(Path(args.reference), args.eval_pairs)
    pairs = np.asarray(ref["pairs"])
    ref_argmax = ref["argmax"]
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16_model = ct.models.MLModel(
        str(args.fp16_package), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    fp16_in, fp16_out = _io_names(Path(args.fp16_package))

    size = args.tile + 2 * args.halo
    crop_package = out_dir / f"segnet_crop{size}_fp32.mlpackage"
    traced, _shape, _weights = _traced("segnet")
    _m, _obs, crop_convert_s, _rec = _convert_mixed(
        traced, (1, 3, size, size), frozenset(), crop_package
    )
    crop_model = ct.models.MLModel(str(crop_package), compute_units=ct.ComputeUnit.CPU_ONLY)
    crop_in, crop_out = _io_names(crop_package)

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane2.hybrid.v2",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "pairs": int(pairs.size),
        "band_width": args.band,
        "tile": args.tile,
        "halo": args.halo,
        "crop_shape": [1, 3, size, size],
        "crop_mlpackage": str(crop_package),
        "crop_mlpackage_sha256": sha256_tree(crop_package),
        "crop_convert_seconds": crop_convert_s,
        "band_selected_from": "fp16 top-2 margin at inference (not the fp32 reference)",
        "seg_bar": SEG_AUTHORITY_FLIP_BAR,
    }

    flips_dense = 0
    flips_hybrid = 0
    total_px = 0
    ane_time = 0.0
    recompute_time = 0.0
    crop_fractions: list[float] = []
    tile_occupancy: list[float] = []
    crops_per_frame: list[int] = []
    band_fractions: list[float] = []
    dilated_fractions: list[float] = []
    corrected = 0
    crop_disagreements = 0
    crop_band_px = 0
    per_pair_hybrid = np.zeros(pairs.size, dtype=np.float64)
    started = time.time()

    for index, pair in enumerate(pairs):
        prepared = _seg_prepared(raw.frame1(pair)).numpy()
        t0 = time.perf_counter()
        logits16 = np.asarray(fp16_model.predict({fp16_in: prepared})[fp16_out])[0]
        ane_time += time.perf_counter() - t0
        arg16 = logits16.argmax(axis=0).astype(np.uint8)
        order = np.sort(logits16, axis=0)
        margin16 = (order[-1] - order[-2]).astype(np.float32)
        reference = np.asarray(ref_argmax[index])
        flips_dense += int((arg16 != reference).sum())
        total_px += reference.size

        band = margin_band_mask(margin16, args.band)
        band_fractions.append(float(band.mean()))
        dilated_fractions.append(
            float(dilate_bool(band, args.halo).mean()) if args.halo else float(band.mean())
        )
        occupied, tiles = occupied_tiles(band, args.tile)
        tile_occupancy.append(occupied / tiles if tiles else 0.0)

        merged = arg16.copy()
        pairs_of_boxes = fixed_crop_boxes(band, args.tile, args.halo)
        crops_per_frame.append(len(pairs_of_boxes))
        crop_fractions.append(len(pairs_of_boxes) * size * size / float(EVAL_H * EVAL_W))
        t1 = time.perf_counter()
        for (cy0, cy1, cx0, cx1), (by0, by1, bx0, bx1) in pairs_of_boxes:
            crop = np.ascontiguousarray(prepared[:, :, by0:by1, bx0:bx1])
            crop_logits = np.asarray(crop_model.predict({crop_in: crop})[crop_out])[0]
            crop_arg = crop_logits.argmax(axis=0).astype(np.uint8)
            inner = crop_arg[cy0 - by0 : cy1 - by0, cx0 - bx0 : cx1 - bx0]
            sub = band[cy0:cy1, cx0:cx1]
            merged[cy0:cy1, cx0:cx1] = np.where(sub, inner, merged[cy0:cy1, cx0:cx1])
            crop_band_px += int(sub.sum())
            crop_disagreements += int(
                (inner[sub] != reference[cy0:cy1, cx0:cx1][sub]).sum()
            )
        recompute_time += time.perf_counter() - t1

        corrected += int(((arg16 != reference) & band).sum())
        pair_flips = int((merged != reference).sum())
        flips_hybrid += pair_flips
        per_pair_hybrid[index] = pair_flips / reference.size
        if index % 20 == 0:
            print(
                f"[hybrid] {index}/{pairs.size} dense={flips_dense} hybrid={flips_hybrid} "
                f"crops={np.mean(crops_per_frame):.1f} {time.time() - started:.0f}s",
                flush=True,
            )

    payload = out_dir / "hybrid_per_pair_flip_rate.npy"
    np.save(payload, per_pair_hybrid)
    reference_report = json.loads(Path(args.reference_report).read_text())
    ref_seconds = float(reference_report["segnet"]["seconds"])
    ref_pairs = int(reference_report["pair_count"])
    ref_per_pair = ref_seconds / ref_pairs

    report["dense"] = seg_flip_verdict(flips_dense, total_px, SEG_AUTHORITY_FLIP_BAR)
    report["hybrid"] = seg_flip_verdict(
        flips_hybrid, total_px, SEG_AUTHORITY_FLIP_BAR, per_pair_hybrid
    )
    report["hybrid"]["per_pair_payload"] = str(payload)
    report["hybrid"]["per_pair_payload_sha256"] = sha256_tree(payload)
    report["band_geometry"] = {
        "band_pixel_fraction_median": float(np.median(band_fractions)),
        "band_dilated_by_halo_fraction_median": float(np.median(dilated_fractions)),
        "crop_area_fraction_median": float(np.median(crop_fractions)),
        "crop_area_fraction_mean": float(np.mean(crop_fractions)),
        "tile_occupancy_median": float(np.median(tile_occupancy)),
        "tile_occupancy_max": float(np.max(tile_occupancy)),
        "crops_per_frame_median": float(np.median(crops_per_frame)),
        "crops_per_frame_max": int(np.max(crops_per_frame)),
        "band_pixels_recomputed": int(crop_band_px),
        "crop_vs_fullframe_argmax_disagreements": int(crop_disagreements),
        "crop_exactness_note": (
            "a crop is NOT a window on the full-frame result: EfficientNet-B2's 23 "
            "squeeze-excitation blocks pool globally, so a nonzero count here is the "
            "CROP REALIZATION failing exactness, not fp32 drift"
        ),
    }
    report["timing"] = hybrid_speedup(
        ane_s=ane_time / pairs.size,
        recompute_s=recompute_time / pairs.size,
        reference_s=ref_per_pair,
    )
    report["timing"]["reference_source"] = (
        "cached CPU-torch fp32 1-thread reference run (dense pass, per pair)"
    )
    report["flips_inside_band"] = int(corrected)
    report["flip_coverage_by_band"] = float(corrected / flips_dense) if flips_dense else 1.0
    report["seconds"] = time.time() - started

    digest = write_json(Path(args.out), report)
    print(json.dumps({"hybrid": args.out, "sha256": digest}))
    return 0


# ------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enumerate_p = sub.add_parser("enumerate", help="MIL compute-op inventory")
    enumerate_p.add_argument("--model", choices=("segnet", "posenet", "both"), default="both")
    enumerate_p.add_argument("--out-dir", required=True)
    enumerate_p.add_argument("--out", required=True)
    enumerate_p.set_defaults(func=run_enumerate)

    reference_p = sub.add_parser("reference", help="cache the CPU-torch fp32 authority")
    reference_p.add_argument("--model", choices=("segnet", "posenet", "both"), default="both")
    reference_p.add_argument("--raw", required=True)
    reference_p.add_argument("--pairs", type=int, default=600)
    reference_p.add_argument("--threads", type=int, default=1)
    reference_p.add_argument("--out-dir", required=True)
    reference_p.add_argument("--out", required=True)
    reference_p.set_defaults(func=run_reference)

    sens_p = sub.add_parser("sensitivity", help="per-op-group fp16 sensitivity profile")
    sens_p.add_argument("--model", choices=("segnet", "posenet"), required=True)
    sens_p.add_argument("--enumerate-json", required=True)
    sens_p.add_argument("--reference", required=True)
    sens_p.add_argument("--raw", required=True)
    sens_p.add_argument("--groups", type=int, default=16)
    sens_p.add_argument("--threads", type=int, default=1)
    sens_p.add_argument("--compute-units", default="CPU_AND_NE")
    sens_p.add_argument("--placement", action="store_true")
    sens_p.add_argument("--keep-packages", action="store_true")
    sens_p.add_argument("--out-dir", required=True)
    sens_p.add_argument(
        "--eval-pairs",
        type=int,
        default=None,
        help="stratified subsample of the cached reference rows (default: all of them)",
    )
    sens_p.add_argument("--out", required=True)
    sens_p.set_defaults(func=run_sensitivity)

    ladder_p = sub.add_parser("ladder", help="fp16 prefix / fp32 suffix split ladder")
    ladder_p.add_argument("--model", choices=("segnet", "posenet"), required=True)
    ladder_p.add_argument("--enumerate-json", required=True)
    ladder_p.add_argument("--reference", required=True)
    ladder_p.add_argument("--raw", required=True)
    ladder_p.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ladder_p.add_argument("--modes", nargs="+", default=["CPU_AND_NE"])
    ladder_p.add_argument("--reps", type=int, default=20)
    ladder_p.add_argument("--threads", type=int, default=1)
    ladder_p.add_argument("--compute-units", default="CPU_AND_NE")
    ladder_p.add_argument("--out-dir", required=True)
    ladder_p.add_argument(
        "--eval-pairs",
        type=int,
        default=None,
        help="stratified subsample of the cached reference rows (default: all of them)",
    )
    ladder_p.add_argument("--out", required=True)
    ladder_p.set_defaults(func=run_ladder)

    sel_p = sub.add_parser("selective", help="named minimal fp32 set, rest fp16")
    sel_p.add_argument("--model", choices=("segnet", "posenet"), required=True)
    sel_p.add_argument("--enumerate-json", required=True)
    sel_p.add_argument("--reference", required=True)
    sel_p.add_argument("--raw", required=True)
    sel_p.add_argument(
        "--fp32-set",
        nargs="+",
        required=True,
        help="label=ordinals, e.g. tail8=278:286 or head=0,1,2",
    )
    sel_p.add_argument("--modes", nargs="+", default=["CPU_AND_NE"])
    sel_p.add_argument("--reps", type=int, default=20)
    sel_p.add_argument("--threads", type=int, default=1)
    sel_p.add_argument("--compute-units", default="CPU_AND_NE")
    sel_p.add_argument("--out-dir", required=True)
    sel_p.add_argument(
        "--eval-pairs",
        type=int,
        default=None,
        help="stratified subsample of the cached reference rows (default: all of them)",
    )
    sel_p.add_argument("--out", required=True)
    sel_p.set_defaults(func=run_selective)

    hyb_p = sub.add_parser("hybrid", help="realized crop-batched exact-argmax hybrid")
    hyb_p.add_argument("--fp16-package", required=True)
    hyb_p.add_argument("--reference", required=True)
    hyb_p.add_argument("--reference-report", required=True)
    hyb_p.add_argument("--raw", required=True)
    hyb_p.add_argument("--band", type=float, default=0.4456)
    hyb_p.add_argument("--tile", type=int, default=64)
    hyb_p.add_argument("--halo", type=int, default=32)
    hyb_p.add_argument("--threads", type=int, default=1)
    hyb_p.add_argument(
        "--eval-pairs",
        type=int,
        default=None,
        help="stratified subsample of the cached reference rows (default: all of them)",
    )
    hyb_p.add_argument("--out", required=True)
    hyb_p.set_defaults(func=run_hybrid)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (Ane2Error, AnePrecisionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
