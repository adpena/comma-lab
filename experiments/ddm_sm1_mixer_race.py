"""Disk-resumable SM1 feature fit and five retained lossless coding variants."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import brotli
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ddm_sm1_semantic_bound as b
from experiments import ddm_sm1_semantic_mixer_codec as c


def array_file(path, array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return b.retain(path, buffer.getvalue())


def binding(root):
    return {
        "inputs": b.fact(root / "INPUTS.json"),
        "bound": b.fact(root / "BOUND_SUPPORTED.json"),
        "codec": b.fact(c.__file__),
        "base_codec": b.fact(c.rc1.__file__),
        "fixed_math": b.fact(c.fixed.__file__),
    }


def gate(root):
    result = json.loads((root / "BOUND_SUPPORTED.json").read_text())
    if not result["gate_pass"]:
        raise RuntimeError("closed-form gate failed; coder construction forbidden")
    for key in ("input_binding", "source_code", "decoder_trace"):
        if b.fact(result[key]["path"]) != result[key]:
            raise RuntimeError(f"bound provenance drift: {key}")
    inputs = json.loads((root / "INPUTS.json").read_text())
    for source in inputs["source_files"]:
        if b.fact(source["path"]) != source:
            raise RuntimeError("source-runtime drift")
    if b.fact(root / "retained/source/body.sm3r") != result["body"]:
        raise RuntimeError("source-body drift")
    for item in inputs["census"].values():
        if b.fact(item["path"]) != item:
            raise RuntimeError("source-section drift")
    source_sections, header = b.sections(root / "source_runtime/archive.zip")
    restored_rider = brotli.decompress(source_sections["semantic"])
    restored_rider = b.layout_tools.ck2_uninterleave(restored_rider) if header[4] & 2 else restored_rider
    if restored_rider != (root / "retained/source/rider.rc1s").read_bytes():
        raise RuntimeError("source-rider drift")
    return result


def prepare(root):
    bound = gate(root)
    target = root / "FEATURES.json"
    if target.exists():
        result = json.loads(target.read_text())
        if result["binding"] != binding(root):
            raise RuntimeError("feature binding drift")
        for key in ("x", "truth"):
            if b.fact(result[key]["path"]) != result[key]:
                raise RuntimeError("retained feature drift")
        return result
    _, shipped, template = b.load_source(root)
    # The fallback helper copies must match the actual shipped decoder code.
    if Path(shipped.__file__).read_bytes() != Path(c.rc1.__file__).read_bytes():
        raise RuntimeError("RC1 fallback differs from shipped code")
    if (root / "source_runtime/runtime/rc2_hpac_semistatic_mixing.py").read_bytes() != Path(
        c.fixed.__file__
    ).read_bytes():
        raise RuntimeError("fixed-point fallback differs from shipped code")
    body = (root / "retained/source/body.sm3r").read_bytes()
    metadata, groups = c.split_body(body, template)
    b.retain(root / "retained/fit/metadata.bin", metadata)
    _, desc = c.plan_metadata(metadata, template)
    initial = np.zeros(24, dtype=np.int8)
    initial[0] = 32
    _, decoded, events, truth = c.walk(desc, initial, source=groups, observe=True)
    if any(not np.array_equal(a, z) for a, z in zip(groups, decoded, strict=True)):
        raise RuntimeError("feature trace changed values")
    x, y = np.asarray(events, dtype=np.int32), np.asarray(truth, dtype=np.uint8)
    if len(y) != bound["coded_bits"]:
        raise RuntimeError("incomplete event matrix")
    source = np.load(root / "retained/decoder_events.npy", allow_pickle=False)
    if not np.array_equal(y, source[:, 2]):
        raise RuntimeError("truth differs from shipped observed decoder")
    baseline_p = np.asarray([c.fixed.squash(int(v)) for v in x[:, 0]])
    observed_p = np.where(source[:, 2], 4096 - source[:, 3], source[:, 3])
    if not np.array_equal(baseline_p, observed_p):
        raise RuntimeError("baseline expert differs from shipped predictor")
    result = {
        "binding": binding(root),
        "x": array_file(root / "retained/fit/x.npy", x),
        "truth": array_file(root / "retained/fit/truth.npy", y),
        "events": len(y),
        "all_source_probabilities_identical": True,
        "weight_count": 24,
        "model_family": "scale_geometry_history_shared",
        "scope": "one shared vector; six named context families plus adaptation timescale and count priors",
    }
    b.save(target, result)
    return result


def fixed_loss(x, y, weights):
    score = np.einsum("ij,j->i", x, weights.astype(np.int64), optimize=False)
    logits = np.where(score >= 0, (score + 16) // 32, -((-score + 16) // 32))
    table = np.asarray(c.fixed.STRETCH, dtype=np.int64)
    index = np.searchsorted(table, logits)
    lo, hi = np.clip(index - 1, 0, 4094), np.clip(index, 0, 4094)
    p = np.where(logits - table[lo] <= table[hi] - logits, lo + 1, hi + 1)
    return float(-np.log2(np.where(y == 0, p, 4096 - p) / 4096).sum())


def fit(root):
    prepared = prepare(root)
    inputs = {"features": b.fact(root / "FEATURES.json"), "code": b.fact(__file__)}
    target = root / "FIT.json"
    if target.exists():
        result = json.loads(target.read_text())
        if result["binding"] != inputs:
            raise RuntimeError("fit binding drift")
        return result
    b.save(root / "FIT_INPUTS.json", inputs)
    x = np.load(prepared["x"]["path"], allow_pickle=False)
    y = np.load(prepared["truth"]["path"], allow_pickle=False)
    design = x.astype(np.float64) * (math.log(2) / 4096)
    zero = (y == 0).astype(np.float64)
    current = np.zeros(24)
    current[0] = 1
    folder = root / "retained/fit/iterations"
    folder.mkdir(parents=True, exist_ok=True)
    records = sorted(folder.glob("ITER_*.json"))
    start = 0
    if records:
        record = json.loads(records[-1].read_text())
        current = np.asarray(record["weights"])
        start = record["iteration"]
        if record["improvement_nats"] < 1e-7:
            start = 40

    def objective(weights):
        z = np.einsum("ij,j->i", design, weights, optimize=False)
        return float(np.sum(np.logaddexp(0, z) - zero * z))

    initial_loss = objective(np.eye(24)[0])
    for iteration in range(start, 40):
        z = np.einsum("ij,j->i", design, current, optimize=False)
        p = 1 / (1 + np.exp(-z))
        loss = objective(current)
        grad = np.einsum("ij,i->j", design, p - zero, optimize=False)
        hess = np.einsum("ij,ik,i->jk", design, design, p * (1 - p), optimize=False)
        direction = np.linalg.solve(hess + 1e-6 * np.eye(24), -grad)
        trials, accepted = [], False
        for power in range(16):
            trial = current + direction / (2**power)
            # Stay strictly in the int8 representable interior. No hidden optimizer state.
            if np.max(np.abs(trial)) > 3.95:
                continue
            value = objective(trial)
            trials.append({"power": power, "weights": trial.tolist(), "loss_nats": value})
            if value <= loss + 1e-4 * float(grad @ (trial - current)):
                accepted = True
                break
        b.save(folder / f"TRIALS_{iteration + 1:03d}.json", trials)
        if not accepted:
            raise RuntimeError("fit did not descend within declared int8 box")
        current = trial
        b.save(
            folder / f"ITER_{iteration + 1:03d}.json",
            {
                "iteration": iteration + 1,
                "weights": current.tolist(),
                "loss_nats": value,
                "max_gradient": float(np.max(np.abs(grad))),
                "improvement_nats": loss - value,
            },
        )
        if loss - value < 1e-7:
            break
    b.save(
        root / "FIT_CONTINUOUS.json",
        {
            "binding": inputs,
            "weights": current.tolist(),
            "terminal_iteration": iteration + 1 if start < 40 else record["iteration"],
            "complete": True,
        },
    )
    weights = np.rint(current * 32).astype(np.int8)
    quantized_initial = weights.copy()
    # Exact integer objective, two complete coordinate passes, own tuning optimum.
    for cycle in range(2):
        for j in range(24):
            choices = []
            for delta in (-1, 0, 1):
                v = int(weights[j]) + delta
                if -127 <= v <= 127:
                    trial = weights.copy()
                    trial[j] = v
                    choices.append((fixed_loss(x, y, trial), abs(delta), delta, trial))
            winner = min(choices, key=lambda t: t[:3])
            weights = winner[3]
            b.save(
                root / f"retained/fit/coordinate_{cycle}_{j:02d}.json",
                {"weights": weights.tolist(), "loss_bits": winner[0]},
            )
    baseline_fixed_bits = fixed_loss(x, y, np.asarray([32] + [0] * 23, dtype=np.int8))
    fitted_fixed_bits = fixed_loss(x, y, weights)
    result = {
        "binding": inputs,
        "continuous_weights": current.tolist(),
        "initial_quantized_weights": quantized_initial.tolist(),
        "weights": weights.tolist(),
        "baseline_fixed_bits": baseline_fixed_bits,
        "fitted_fixed_bits": fitted_fixed_bits,
        "continuous_gain_bytes": (initial_loss - objective(current)) / math.log(2) / 8,
        "counted_parameters_bytes": 24,
        "header_increment_bytes": 2,
        "fixed_net_ideal_bytes": (baseline_fixed_bits - fitted_fixed_bits) / 8 - 26,
        "limitation": "fit and integer log-loss only; no realized container gain until RACE",
    }
    b.save(target, result)
    return result


def race(root):
    fitted = fit(root)
    renderer, shipped, template = b.load_source(root)
    body = (root / "retained/source/body.sm3r").read_bytes()
    source_rider = (root / "retained/source/rider.rc1s").read_bytes()
    source_container = (root / "retained/source/semantic.bin").read_bytes()
    # Historical raw-body/container controls are materialized and retained.
    controls = {}
    for name, data in [("raw_body", body), ("rc1_rider", source_rider)]:
        transformed = b.layout_tools.ck2_interleave(data)
        b.retain(root / f"retained/controls/{name}.ck2", transformed)
        coded = brotli.compress(transformed, quality=11, lgwin=24)
        controls[name] = b.retain(root / f"retained/controls/{name}.br", coded)
        if brotli.decompress(coded) != transformed:
            raise RuntimeError("Brotli control failed")
    if (root / "retained/controls/rc1_rider.br").read_bytes() != source_container:
        raise RuntimeError("current RC1 container control is not byte-identical")
    rng = np.random.default_rng(b.SEED)
    weights = np.asarray(fitted["weights"], dtype=np.int8)
    variants = [weights]
    for _ in range(4):
        offset = np.zeros(24, dtype=np.int16)
        chosen = rng.choice(24, size=4, replace=False)
        offset[chosen] = rng.choice([-1, 1], size=4)
        variants.append(np.clip(weights.astype(np.int16) + offset, -127, 127).astype(np.int8))
    rows = []
    for index, vector in enumerate(variants):
        folder = root / f"retained/variants/sample_{index}"
        bind = {"fit": b.fact(root / "FIT.json"), "codecs": binding(root), "weights": vector.tolist()}
        target = folder / "RESULT.json"
        if target.exists():
            row = json.loads(target.read_text())
            if row["binding"] != bind or any(b.fact(v["path"]) != v for v in row["artifacts"]):
                raise RuntimeError("variant resume drift")
            rows.append(row)
            continue
        artifacts = [b.retain(folder / "weights.int8", vector.tobytes())]
        rider, payload, metadata = c.encode(body, template, vector)
        rider_fact = b.retain(folder / "rider.sm1s", rider)
        artifacts += [rider_fact, b.retain(folder / "range.bin", payload), b.retain(folder / "metadata.bin", metadata)]
        twin, twin_payload, twin_metadata = c.encode(body, template, vector)
        artifacts += [
            b.retain(folder / "rider.repeat.sm1s", twin),
            b.retain(folder / "range.repeat.bin", twin_payload),
            b.retain(folder / "metadata.repeat.bin", twin_metadata),
        ]
        if twin != rider:
            raise RuntimeError("twin encode differs")
        decoded = c.restore_semantic(rider, template)
        artifacts.append(b.retain(folder / "decoded.sm3r", decoded))
        if decoded != body:
            raise RuntimeError("decoded body differs")
        transformed = b.layout_tools.ck2_interleave(rider)
        artifacts.append(b.retain(folder / "container_input.ck2", transformed))
        container = brotli.compress(transformed, quality=11, lgwin=24)
        semantic_container = b.retain(folder / "semantic.br", container)
        artifacts.append(semantic_container)
        if brotli.decompress(container) != transformed:
            raise RuntimeError("container did not restore rider")
        row = {
            "binding": bind,
            "sample": index,
            "weights": vector.tolist(),
            "rider": rider_fact,
            "ck2": True,
            "q": 11,
            "lgwin": 24,
            "semantic_container": semantic_container,
            "artifacts": artifacts,
            "payload_bytes": len(payload),
            "parameter_bytes": 24,
            "header_increment_bytes": 2,
            "body_identity": True,
            "twins_identical": True,
            "net_container_bytes": len(container) - len(source_container),
        }
        b.save(target, row)
        print(
            json.dumps(
                {
                    "sample": index,
                    "range_bytes": len(payload),
                    "semantic_bytes": len(container),
                    "delta": row["net_container_bytes"],
                }
            ),
            flush=True,
        )
        rows.append(row)
    result = {
        "axis": b.AXIS,
        "score_claim": False,
        "bound": b.fact(root / "BOUND_SUPPORTED.json"),
        "fit": b.fact(root / "FIT.json"),
        "rows": rows,
        "controls": controls,
        "winner": min(rows, key=lambda r: (r["semantic_container"]["bytes"], r["sample"])),
        "variant_design": "one fitted vector and four seeded +/-1 perturbations of four shared coefficients; all lossless; q11/w24 fixed",
        "seed": b.SEED,
        "families_built": 1,
    }
    b.save(root / "RACE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "race"), required=True)
    parser.add_argument("--root", type=Path, default=b.ROOT)
    parser.add_argument("--resume-from", type=Path)
    from comma_lab.instrument_gates import add_coder_gate_argument, check_experimental_coder

    add_coder_gate_argument(parser)
    args = parser.parse_args()
    instrument_gate = check_experimental_coder((args.resume_from or args.root) / "source_runtime", rationale=args.coder_differs_because)
    b.save((args.resume_from or args.root) / "INSTRUMENT_GATE.json", instrument_gate)
    result = globals()[args.stage](args.resume_from or args.root)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("continuous_gain_bytes", "fixed_net_ideal_bytes", "weights", "events")
                if k in result
            }
        )
    )
