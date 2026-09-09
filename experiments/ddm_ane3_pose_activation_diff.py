"""ddm_ane3 ITEM 2: localize PoseNet's fixed-path fp16 dim-0 drift.

The instrument exposes the tensor output of each of PoseNet compute operations
0:18 in two otherwise matched Core ML packages (all fp32 versus all fp16),
retains both raw activation banks for a stratified sample, and then converts 18
causal rescue packages.  Rescue ``i`` keeps operation ``i`` fp32 while every
other compute op remains fp16; its reduction of the pinned dim-0 error is the
strict attribution test.  No latency or device request is used as placement
evidence: ``MLComputePlan`` supplies that census, including an honest CPU
fallback when the ANE compiler is unavailable.

Every artifact is ``[macOS-CPU/CoreML advisory]`` and ``score_claim=false``.
The frozen cpu_torch PoseNet cache is read, not regenerated, so this instrument
does not enter the scorer lane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import ddm_ane2_engineer_precision_drift as ane2

from tac.ane_precision import pose_drift_verdict
from tac.ane_screening import sha256_tree, write_json

AXIS = "[macOS-CPU/CoreML advisory, frozen PoseNet, stratified real inputs]"
BOUNDARY_LO = 0
BOUNDARY_HI = 18
ATTRIBUTION_BAR = 0.80


class Ane3ActivationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _compute_ops(spec) -> list[Any]:
    function = spec.mlProgram.functions["main"]
    block = function.block_specializations[function.opset]
    return [op for op in block.operations if op.type not in ("const", "cast")]


def _debug_package(source: Path, destination: Path, count: int) -> tuple[Path, list[dict[str, Any]]]:
    """Expose the first ``count`` compute-op tensors and retain the package."""
    import coremltools as ct
    from coremltools.proto import FeatureTypes_pb2, Model_pb2

    if destination.exists() and destination.with_suffix(".mlmodelc").exists():
        retained = ct.models.MLModel(str(destination), skip_model_load=True).get_spec()
        rows = []
        for ordinal, op in enumerate(_compute_ops(retained)[:count]):
            output = op.outputs[0]
            rows.append(
                {
                    "ordinal": ordinal,
                    "op_type": op.type,
                    "output_name": output.name,
                    "mil_dtype": int(output.type.tensorType.dataType),
                }
            )
        if len(rows) != count:
            raise Ane3ActivationError(
                f"retained debug package {destination} exposes only {len(rows)} compute ops"
            )
        return destination, rows

    base = ct.models.MLModel(str(source), skip_model_load=True)
    spec = copy.deepcopy(base.get_spec())
    ops = _compute_ops(spec)
    if len(ops) < count:
        raise Ane3ActivationError(f"{source} has only {len(ops)} compute ops")
    function = spec.mlProgram.functions["main"]
    block = function.block_specializations[function.opset]
    existing = {item.name for item in spec.description.output}
    rows: list[dict[str, Any]] = []
    mil_to_feature = {
        10: FeatureTypes_pb2.ArrayFeatureType.FLOAT16,
        11: FeatureTypes_pb2.ArrayFeatureType.FLOAT32,
    }
    for ordinal, op in enumerate(ops[:count]):
        if len(op.outputs) != 1:
            raise Ane3ActivationError(
                f"compute op {ordinal} ({op.type}) has {len(op.outputs)} outputs; "
                "the one-tensor boundary contract is not satisfied"
            )
        output = op.outputs[0]
        if output.type.WhichOneof("type") != "tensorType":
            raise Ane3ActivationError(f"op {ordinal} output is not a tensor")
        dtype = int(output.type.tensorType.dataType)
        feature_dtype = mil_to_feature.get(dtype)
        if feature_dtype is None:
            raise Ane3ActivationError(f"op {ordinal} has unsupported MIL dtype {dtype}")
        name = output.name
        if name not in block.outputs:
            block.outputs.append(name)
        if name not in existing:
            description = Model_pb2.FeatureDescription()
            description.name = name
            description.type.multiArrayType.dataType = feature_dtype
            spec.description.output.append(description)
            existing.add(name)
        rows.append(
            {
                "ordinal": ordinal,
                "op_type": op.type,
                "output_name": name,
                "mil_dtype": dtype,
            }
        )

    if destination.exists():
        shutil.rmtree(destination)
    if destination.with_suffix(".mlmodelc").exists():
        shutil.rmtree(destination.with_suffix(".mlmodelc"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    ct.models.utils.save_spec(
        spec,
        str(destination),
        weights_dir=base.weights_dir,
    )
    ane2._compiled(destination)
    return destination, rows


def _load_retained_activation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _activation_metrics(
    fp32_payloads: list[Path],
    fp16_payloads: list[Path],
    rows32: list[dict[str, Any]],
    rows16: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accumulators = [
        {"abs_sum": 0.0, "sq_sum": 0.0, "ref_sq_sum": 0.0, "count": 0, "max": 0.0}
        for _ in rows32
    ]
    for path32, path16 in zip(fp32_payloads, fp16_payloads, strict=True):
        bank32 = _load_retained_activation(path32)
        bank16 = _load_retained_activation(path16)
        for ordinal, accumulator in enumerate(accumulators):
            a = np.asarray(bank32[f"op_{ordinal:03d}"], dtype=np.float64)
            b = np.asarray(bank16[f"op_{ordinal:03d}"], dtype=np.float64)
            if a.shape != b.shape:
                raise Ane3ActivationError(
                    f"op {ordinal} activation shape mismatch: {a.shape} vs {b.shape}"
                )
            delta = b - a
            accumulator["abs_sum"] += float(np.abs(delta).sum())
            accumulator["sq_sum"] += float(np.square(delta).sum())
            accumulator["ref_sq_sum"] += float(np.square(a).sum())
            accumulator["count"] += int(delta.size)
            accumulator["max"] = max(accumulator["max"], float(np.abs(delta).max()))

    result: list[dict[str, Any]] = []
    for ordinal, accumulator in enumerate(accumulators):
        count = accumulator["count"]
        result.append(
            {
                "ordinal": ordinal,
                "op_type": rows32[ordinal]["op_type"],
                "fp32_output_name": rows32[ordinal]["output_name"],
                "fp16_output_name": rows16[ordinal]["output_name"],
                "fp32_mil_dtype": rows32[ordinal]["mil_dtype"],
                "fp16_mil_dtype": rows16[ordinal]["mil_dtype"],
                "elements": count,
                "mean_abs_delta": accumulator["abs_sum"] / count,
                "rmse": (accumulator["sq_sum"] / count) ** 0.5,
                "max_abs_delta": accumulator["max"],
                "relative_l2": (
                    accumulator["sq_sum"] / accumulator["ref_sq_sum"]
                )
                ** 0.5
                if accumulator["ref_sq_sum"]
                else 0.0,
            }
        )
    return result


def _predict_activation_banks(
    fp32_debug: Path,
    fp16_debug: Path,
    rows32: list[dict[str, Any]],
    rows16: list[dict[str, Any]],
    reference: dict[str, Any],
    payload_dir: Path,
    compute_units: str,
) -> tuple[list[dict[str, Any]], list[Path], list[Path]]:
    import coremltools as ct

    mode = getattr(ct.ComputeUnit, compute_units)
    model32 = ct.models.CompiledMLModel(str(fp32_debug.with_suffix(".mlmodelc")), compute_units=mode)
    model16 = ct.models.CompiledMLModel(str(fp16_debug.with_suffix(".mlmodelc")), compute_units=mode)
    input32, final32 = ane2._io_names(fp32_debug)
    input16, final16 = ane2._io_names(fp16_debug)
    pairs = np.asarray(reference["pairs"])
    prepared = np.asarray(reference["prepared"])
    manifest: list[dict[str, Any]] = []
    fp32_payloads: list[Path] = []
    fp16_payloads: list[Path] = []
    for index, pair in enumerate(pairs):
        path32 = payload_dir / "activation_fp32" / f"pair_{int(pair):03d}.npz"
        path16 = payload_dir / "activation_fp16" / f"pair_{int(pair):03d}.npz"
        if not path32.exists():
            got32 = model32.predict({input32: np.asarray(prepared[index])[None]})
            arrays32 = {
                f"op_{ordinal:03d}": np.asarray(got32[row["output_name"]])
                for ordinal, row in enumerate(rows32)
            }
            arrays32["pose_output"] = np.asarray(got32[final32])
            receipt32 = _write_npz_atomic(path32, arrays32)
        else:
            receipt32 = {
                "path": str(path32),
                "bytes": path32.stat().st_size,
                "sha256": _sha256_file(path32),
            }
        if not path16.exists():
            got16 = model16.predict({input16: np.asarray(prepared[index])[None]})
            arrays16 = {
                f"op_{ordinal:03d}": np.asarray(got16[row["output_name"]])
                for ordinal, row in enumerate(rows16)
            }
            arrays16["pose_output"] = np.asarray(got16[final16])
            receipt16 = _write_npz_atomic(path16, arrays16)
        else:
            receipt16 = {
                "path": str(path16),
                "bytes": path16.stat().st_size,
                "sha256": _sha256_file(path16),
            }
        manifest.append(
            {
                "pair": int(pair),
                "fp32": receipt32,
                "fp16": receipt16,
            }
        )
        fp32_payloads.append(path32)
        fp16_payloads.append(path16)
        print(f"[activation] {index + 1}/{pairs.size} pair={int(pair)}", flush=True)
    return manifest, fp32_payloads, fp16_payloads


def _pose_row_from_retained(reference: dict[str, Any], payload: Path) -> dict[str, Any]:
    got = np.load(payload)
    row = pose_drift_verdict(np.asarray(reference["poses"]), np.asarray(got))
    row["compute_units"] = "retained payload replay"
    row["poses_payload"] = str(payload)
    row["poses_payload_sha256"] = sha256_tree(payload)
    return row


def _evaluate_package(
    package: Path,
    reference: dict[str, Any],
    payload: Path,
    compute_units: str,
) -> dict[str, Any]:
    if payload.exists():
        return _pose_row_from_retained(reference, payload)
    row, _got = ane2._eval_posenet(package, compute_units, reference, payload)
    return row


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    payload_dir = Path(args.payload_dir)
    report_path = Path(args.out)
    partial_path = report_path.with_name(report_path.stem + "_partial.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    names = ane2._enumerated(Path(args.enumerate_json), "posenet")
    traced, shape, _weights = ane2._traced("posenet")

    fp32_package = out_dir / "posenet_fixed_all_fp32.mlpackage"
    fp16_package = out_dir / "posenet_fixed_all_fp16.mlpackage"
    if not fp32_package.exists():
        ane2._convert_mixed(
            traced,
            shape,
            frozenset(),
            fp32_package,
            expected_ops=names,
        )
    if not fp16_package.exists():
        ane2._convert_mixed(
            traced,
            shape,
            frozenset(names),
            fp16_package,
            expected_ops=names,
        )
    ane2._compiled(fp32_package)
    ane2._compiled(fp16_package)

    fp32_debug, rows32 = _debug_package(
        fp32_package,
        out_dir / "posenet_fixed_all_fp32_debug_0_18.mlpackage",
        BOUNDARY_HI,
    )
    fp16_debug, rows16 = _debug_package(
        fp16_package,
        out_dir / "posenet_fixed_all_fp16_debug_0_18.mlpackage",
        BOUNDARY_HI,
    )
    activation_reference = ane2._load_reference(
        Path(args.reference), args.activation_pairs
    )
    manifest, payloads32, payloads16 = _predict_activation_banks(
        fp32_debug,
        fp16_debug,
        rows32,
        rows16,
        activation_reference,
        payload_dir,
        args.compute_units,
    )
    activation_rows = _activation_metrics(payloads32, payloads16, rows32, rows16)
    activation_base_fp32 = _evaluate_package(
        fp32_package,
        activation_reference,
        payload_dir / "debug_parity" / "base_fp32_poses.npy",
        args.compute_units,
    )
    activation_base_fp16 = _evaluate_package(
        fp16_package,
        activation_reference,
        payload_dir / "debug_parity" / "base_fp16_poses.npy",
        args.compute_units,
    )
    debug_pose32 = np.concatenate(
        [_load_retained_activation(path)["pose_output"] for path in payloads32],
        axis=0,
    )[:, : ane2.POSE_DIMS]
    debug_pose16 = np.concatenate(
        [_load_retained_activation(path)["pose_output"] for path in payloads16],
        axis=0,
    )[:, : ane2.POSE_DIMS]
    base_pose32 = np.load(activation_base_fp32["poses_payload"])
    base_pose16 = np.load(activation_base_fp16["poses_payload"])
    debug_parity = {
        "fp32_max_abs_final_output_delta": float(np.max(np.abs(debug_pose32 - base_pose32))),
        "fp16_max_abs_final_output_delta": float(np.max(np.abs(debug_pose16 - base_pose16))),
        "contract": (
            "the model with exposed intermediate outputs must reproduce the retained "
            "base package's final pose output on the same rows"
        ),
    }

    causal_reference = ane2._load_reference(Path(args.reference), args.causal_pairs)
    uniform_payload = payload_dir / "causal" / "uniform_fp16_poses.npy"
    fp32_payload = payload_dir / "causal" / "uniform_fp32_poses.npy"
    uniform = _evaluate_package(
        fp16_package, causal_reference, uniform_payload, args.compute_units
    )
    fixed_fp32 = _evaluate_package(
        fp32_package, causal_reference, fp32_payload, args.compute_units
    )
    pinned = float(uniform["per_dim_mean_abs_delta"][0])
    if not 0.150 <= pinned <= 0.154:
        pinned_scope = (
            "fixed-output path did not reproduce the predecessor's 0.150-0.154 "
            "dim-0 interval on this execution surface"
        )
    else:
        pinned_scope = "fixed-output path reproduced the 0.150-0.154 pinned interval"

    report: dict[str, Any] = {
        "schema": "tac.ddm_ane3.pose_activation_diff.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "compute_units_requested": args.compute_units,
        "boundary": [BOUNDARY_LO, BOUNDARY_HI],
        "activation_pairs": [int(x) for x in activation_reference["pairs"]],
        "activation_sampling": "stratified over the frozen n600 reference; never a prefix",
        "causal_pairs": int(np.asarray(causal_reference["pairs"]).size),
        "causal_sampling": "stratified over the frozen n600 reference; never a prefix",
        "attribution_bar": ATTRIBUTION_BAR,
        "fp32_package": str(fp32_package),
        "fp32_package_sha256": sha256_tree(fp32_package),
        "fp16_package": str(fp16_package),
        "fp16_package_sha256": sha256_tree(fp16_package),
        "fp32_debug_package": str(fp32_debug),
        "fp32_debug_package_sha256": sha256_tree(fp32_debug),
        "fp16_debug_package": str(fp16_debug),
        "fp16_debug_package_sha256": sha256_tree(fp16_debug),
        "base_placement": {
            "fp32": ane2._placement(fp32_package, args.compute_units),
            "fp16": ane2._placement(fp16_package, args.compute_units),
        },
        "activation_payload_manifest": manifest,
        "activation_rows": activation_rows,
        "debug_output_parity": debug_parity,
        "uniform_fp16": uniform,
        "uniform_fp32": fixed_fp32,
        "pinned_dim0_mean_abs_delta": pinned,
        "pinned_interval_verdict": pinned_scope,
        "causal_rescues": [],
    }
    write_json(partial_path, report)

    all_fp16 = frozenset(names)
    for ordinal in range(BOUNDARY_LO, BOUNDARY_HI):
        op_name = names[ordinal]
        package = out_dir / f"posenet_rescue_op_{ordinal:03d}.mlpackage"
        if not package.exists():
            ane2._convert_mixed(
                traced,
                shape,
                all_fp16 - {op_name},
                package,
                expected_ops=names,
            )
        ane2._compiled(package)
        payload = payload_dir / "causal" / f"rescue_op_{ordinal:03d}_poses.npy"
        fidelity = _evaluate_package(
            package, causal_reference, payload, args.compute_units
        )
        error = float(fidelity["per_dim_mean_abs_delta"][0])
        explained = 1.0 - error / pinned if pinned else 0.0
        row = {
            "ordinal": ordinal,
            "op_name": op_name,
            "op_type": rows32[ordinal]["op_type"],
            "intervention": "this op fp32; every other compute op fp16; model output fp32",
            "dim0_mean_abs_delta": error,
            "fraction_of_pinned_dim0_delta_removed": explained,
            "meets_80pct_attribution_bar": explained >= ATTRIBUTION_BAR,
            "mlpackage": str(package),
            "mlpackage_sha256": sha256_tree(package),
            "fidelity": fidelity,
        }
        report["causal_rescues"].append(row)
        write_json(partial_path, report)
        print(
            f"[rescue] op={ordinal:02d} {op_name} dim0={error:.6g} "
            f"removed={explained:.1%}",
            flush=True,
        )

    winners = [
        row for row in report["causal_rescues"] if row["meets_80pct_attribution_bar"]
    ]
    measured_ane_fraction = float(report["base_placement"]["fp16"]["ane_op_fraction"])
    answers_pinned_ane_question = (
        measured_ane_fraction > 0.0 and 0.150 <= pinned <= 0.154
    )
    report["attribution_verdict"] = {
        "status": "named_op_found" if winners else "scoped_negative",
        "named_ops": [row["op_name"] for row in winners],
        "answers_pinned_ane_question": answers_pinned_ane_question,
        "measured_ane_op_fraction": measured_ane_fraction,
        "scope": (
            "single-op fp32 rescue within fixed-output all-fp16 Core ML PoseNet, "
            f"ops {BOUNDARY_LO}:{BOUNDARY_HI}, stratified n={args.causal_pairs}, "
            f"requested {args.compute_units}, MLComputePlan ANE fraction "
            f"{measured_ane_fraction:.6f}"
        ),
        "reason": (
            "at least one one-op rescue removed >=80% of pinned dim-0 error"
            if winners
            else (
                "no one-op rescue removed >=80% of current-surface dim-0 error; "
                "the predecessor's 0.150-0.154 ANE error remains unlocalized when "
                "measured ANE placement is zero"
            )
        ),
    }
    report["elapsed_seconds"] = time.time() - args.started
    digest = write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "sha256": digest}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumerate-json", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--activation-pairs", type=int, default=32)
    parser.add_argument("--causal-pairs", type=int, default=120)
    parser.add_argument("--compute-units", default="CPU_AND_NE")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.started = time.time()
    try:
        return run(args)
    except (Ane3ActivationError, ane2.Ane2Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
