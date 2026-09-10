#!/usr/bin/env python3
"""Independent causal decoder for the charged BND2 research archive.

Only the candidate's own model bytes and charged segment/residual payload feed
decoding. Retained source rows are compared AFTER recomputing each probability
row; they never supply symbols, model features, or decoder state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_bnd2_causal_trace as trace
from experiments import ddm_bnd2_segment_encode as segment
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_tc1_mixer_codec as tc1


def read_candidate(residual, archive: Path):
    """Adapt only the new counted rider; cmp2's actual reader parses all models."""
    from runtime import tc1_shared_mixer

    original = tc1_shared_mixer.unpack_rider
    result = {}

    def unpack_rider(payload):
        if payload[:5] != b"BND2\x01" or len(payload) <= 40:
            raise ValueError("candidate must contain the BND2 counted rider")
        weights = payload[5:40]
        envelope = b"BND2" + payload[40:] + b"\0"
        supplied, raw_envelope = segment.split_composite(envelope)
        length = int.from_bytes(envelope[4:8], "little")
        _same_supplied, edges = segment.unpack_segments(envelope[8 : 8 + length], with_edges=True)
        if len(supplied) != 600:
            raise ValueError("candidate does not carry 600 segment planes")
        result.update(supplied=supplied, residual=raw_envelope, edges=edges)
        return weights, raw_envelope[4:-1]

    tc1_shared_mixer.unpack_rider = unpack_rider
    try:
        parts = residual.read_residual_archive(archive)
    finally:
        tc1_shared_mixer.unpack_rider = original
    if not result:
        raise ValueError("candidate reader did not consume BND2 rider")
    return parts, result["supplied"], result["residual"], result["edges"]


def decode(candidate: Path, stop: int) -> dict:
    import torch

    root = trace.ROOT / "generation2"
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root) or candidate.name != "archive.zip":
        raise ValueError("candidate is outside this arm's retained store")
    work = candidate.parent / "causal_decode"
    trace.storage(work / "checkpoints")
    source = segment.complete_trace()
    library = Path(source["build"]["library"]["path"])
    if jg2.file_fact(library) != source["build"]["library"]:
        raise ValueError("decoder native library changed")
    route = jg2.load_route_b()
    residual, renderer, renderer_dir = jg2.load_runtime(root / "runtime")
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(trace.SEED)
    np.random.seed(trace.SEED)
    random.seed(trace.SEED)
    torch.use_deterministic_algorithms(True)
    parts, supplied, raw_envelope, edges = read_candidate(residual, candidate)
    inputs = json.loads((root / "INPUTS.json").read_text())
    for fact in inputs["sources"].values():
        if jg2.file_fact(Path(fact["path"])) != fact:
            raise ValueError("read-only runtime source changed")
    binding = {
        "producer": jg2.sha256_file(Path(__file__)),
        "candidate": jg2.file_fact(candidate),
        "segment_code": jg2.sha256_file(Path(segment.__file__)),
        "trace_code": jg2.sha256_file(Path(trace.__file__)),
        "route": jg2.sha256_file(Path(route.__file__)),
        "jg2": jg2.sha256_file(Path(jg2.__file__)),
        "tc1": jg2.sha256_file(Path(tc1.__file__)),
        "source_trace": jg2.sha256_file(root / "trace/RESULT.json"),
        "runtime_inputs": jg2.sha256_file(root / "INPUTS.json"),
        "library": source["build"]["library"],
    }
    model = renderer.load_hpac(residual.materialize_ihs1(parts.hpac_blob, renderer), torch.device("cpu"))
    sparse = residual._sparse_class(renderer_dir)(model, 384, 512)
    corrector, cold = FreeCorrector(384 * 512), FreeCorrector(384 * 512)
    mixer = tc1.SharedMixer(parts.tc1_weights)
    state, start = None, 0
    latest = work / "LATEST.json"
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt["binding"] != binding or jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("causal decoder checkpoint binding changed")
        with np.load(receipt["payload"]["path"], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state["frame"][0])
        jg2.load_corrector_state(corrector, {k[2:]: v for k, v in state.items() if k.startswith("c_")})
        mixer.restore({k[2:]: v for k, v in state.items() if k.startswith("m_")})
        if mixer.frame != start:
            raise ValueError("restored mixer stage differs")
    if not 0 <= start <= stop <= 600:
        raise ValueError("invalid decoder frame range")
    decoder = route.NativeRc64Decoder(library, raw_envelope if state is None else state["decoder"].tobytes())
    previous = torch.zeros((1, 384, 512), dtype=torch.long)
    if state is not None:
        previous[0] = torch.from_numpy(state["previous"].astype(np.int64))
    groups = [np.flatnonzero(m.cpu().numpy().reshape(-1)) for m in renderer.group_masks(torch.device("cpu"))]
    started = time.monotonic()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for frame in range(start, stop):
            reference = segment.load_frame(frame)  # comparisons only, never fed to decoder
            previous_cpu = None if frame == 0 else previous[0].numpy().astype(np.uint8)
            boundary = (
                np.full(384 * 512, 4, dtype=np.uint8)
                if frame == 0
                else residual._boundary_buckets(previous_cpu).reshape(-1)
            )
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            corrector.begin_frame(boundary)
            mixer.begin_frame()
            supplied_symbols = np.full(384 * 512, -1, dtype=np.int32)
            for pos, symbol in supplied[frame].items():
                supplied_symbols[pos] = symbol
            for group, positions in enumerate(groups):
                logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = logits.argmax(axis=1).astype(np.int64)
                feature = boundary[positions].astype(np.int64) * 5 + predicted
                probability = residual._probability_table(
                    logits + parts.table.values[feature], renderer.HPAC_LOGIT_PRECISION
                )
                cs = corrector.group_state(probability, predicted, positions)
                coding = mixer.coding(
                    corrector.coding_row(cs), positions, current[0].numpy().astype(np.uint8), previous_cpu
                )
                if not np.array_equal(coding, reference["rows"][positions]):
                    raise ValueError(f"causal probability trajectory differs at {frame}/{group}")
                symbols = supplied_symbols[positions].copy()
                missing = symbols < 0
                if missing.any():
                    symbols[missing] = decoder.decode(None, coding[missing])
                corrector.observe(cs, symbols.astype(np.int64))
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(symbols.astype(np.int64))
            plane = current[0].numpy().astype(np.uint8)
            fact = trace.blob(work / "decoded" / f"frame_{frame:04d}.u8", plane.tobytes())
            if not np.array_equal(plane, reference["tokens"]):
                raise ValueError("independent causal decode differs from shipped tokens")
            segment.verify_crack_sides(plane, edges[frame])
            trace.record(
                work / "decoded" / f"frame_{frame:04d}.json",
                {"payload": fact, "binding": binding, "all_group_probability_rows_recomputed_identical": True},
            )
            corrector.end_frame(plane.reshape(-1))
            mixer.end_frame(plane, previous_cpu)
            previous = current
            if (frame + 1) % 20 == 0 or frame + 1 == stop:
                captured = jg2.corrector_state(corrector)
                if jg2.uncaptured_divergent_state(corrector, cold, set(captured)):
                    raise ValueError("decoder checkpoint omits corrector state")
                values = {"c_" + k: v for k, v in captured.items()}
                values.update({"m_" + k: v for k, v in mixer.snapshot().items()})
                values.update(
                    frame=np.array([frame + 1]),
                    previous=plane,
                    decoder=np.frombuffer(decoder.get_compressed().tobytes(), dtype=np.uint8),
                )
                fact = trace.arrays(work / "checkpoints" / f"stage_{frame + 1:04d}.npz", values)
                receipt = {"binding": binding, "frame": frame + 1, "payload": fact}
                trace.record(work / f"STAGE_{frame + 1:04d}.json", receipt)
                trace.record(latest, receipt)
            print(
                json.dumps(
                    {"stage": "independent_causal_decode", "frame": frame + 1, "elapsed_s": time.monotonic() - started}
                ),
                flush=True,
            )
    digest = hashlib.sha256()
    for frame in range(stop):
        path = work / "decoded" / f"frame_{frame:04d}.u8"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt["binding"] != binding or jg2.file_fact(path) != receipt["payload"]:
            raise ValueError("decoded field custody changed")
        digest.update(path.read_bytes())
    if stop == 600 and digest.hexdigest() != trace.FIELD_SHA:
        raise ValueError("full decoded field SHA differs")
    return trace.record(
        work / ("RESULT.json" if stop == 600 else f"PARTIAL_{stop:04d}.json"),
        {
            "schema": "ddm_bnd2_independent_causal_decode.v1",
            "binding": binding,
            "frames": stop,
            "resumed_from_frame": start,
            "field_sha256": digest.hexdigest(),
            "axis": trace.AXIS,
            "score_claim": False,
            "all_causal_rows_recomputed_identical": True,
            "decoder_input": "candidate archive only; source rows/labels used exclusively as post-computation comparisons",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != trace.ROOT.resolve():
        raise ValueError("resume root outside owned store")
    print(json.dumps(decode(args.candidate, args.stop_after), sort_keys=True))


if __name__ == "__main__":
    main()
