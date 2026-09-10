#!/usr/bin/env python3
"""Retain the move-42 TC1 trajectory and real source-control twins, scorer-free.

Research only. No fitted prices, no score claims. The trace is an encoder-side
cache, never free decoder side information. Every completed frame is retained;
complete corrector, mixer and arithmetic-encoder checkpoints occur every 20.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_tc1_mixer_codec as tc1

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42")
LIVE = Path("/Volumes/VertigoDataTier/pact/ddm_rp1_round2/candidate/candidate_runtime")
FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rp1_round2/parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
ARCHIVE_SHA = "f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f"
FIELD_SHA = "d5248c775e49d8ac4299d2b0b60ffe5cffbeaafe348ef57986a169d3f118322b"
AXIS = "[exact bytes, scorer-free macOS-CPU]"
SEED = 20260910
RESERVE = 16 * 1024**3


def storage(path: Path, need: int = 16 * 1024**2) -> None:
    """Route writes to this arm and preserve the arm's 16 GiB reserve plus every pending write."""
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"write outside owned store: {path}")
    if shutil.disk_usage(ROOT.parent).free < RESERVE + need:
        raise RuntimeError("STORAGE_RESERVE: retain all existing bytes and stop")
    seen, used = set(), 0
    for item in ROOT.rglob("*"):
        if item.is_file():
            stat = item.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                used += stat.st_size
    if used + need > 8 * 1024**3:
        raise RuntimeError("STORAGE_CAP: retain every payload; no certified deletion")
    path.parent.mkdir(parents=True, exist_ok=True)


def blob(path: Path, payload: bytes) -> dict:
    storage(path, len(payload))
    jg2.persist_immutable_bytes(path, payload, label="tc3 retained payload")
    return jg2.file_fact(path)


def arrays(path: Path, values: dict) -> dict:
    storage(path, sum(v.nbytes for v in values.values()))
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError(f"immutable array payload changed: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".new")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    return jg2.file_fact(path)


def record(path: Path, value: dict) -> dict:
    storage(path)
    jg2.atomic_json(path, value)
    return value


def prepare() -> dict:
    """Pin the actual move42 reader and source bytes in the owned tree."""
    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=REPO)
    if committed != Path(__file__).read_bytes():
        raise RuntimeError("PRODUCER_SOURCE_NOT_COMMITTED: land exact source before production")
    pointer_payload = (REPO / ".omx/state/canonical_frontier_pointer.json").read_bytes()
    pointer = json.loads(pointer_payload)
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != ARCHIVE_SHA:
        raise RuntimeError("POINTER_MOVED: rebind before a new launch")
    if jg2.sha256_file(LIVE / "archive.zip") != ARCHIVE_SHA or jg2.sha256_file(FIELD) != FIELD_SHA:
        raise ValueError("shipped archive/field custody changed")
    root = ROOT / "move42"
    blob(root / "pointer_snapshots" / (hashlib.sha256(pointer_payload).hexdigest() + ".json"), pointer_payload)
    sources = {}
    for path in sorted(LIVE.rglob("*")):
        if (
            path.is_file()
            and path.suffix in (".py", ".c", ".sh", ".zip", ".json", ".md", ".txt")
            and "__pycache__" not in path.parts
        ):
            relative = path.relative_to(LIVE)
            sources[str(relative)] = blob(root / "source_runtime" / relative, path.read_bytes())
    residual, _, _ = jg2.load_runtime(root / "source_runtime")
    parts = residual.read_residual_archive(root / "source_runtime/archive.zip")
    blob(ROOT / "retained/tc1_weights.bin", bytes(parts.tc1_weights))
    blob(ROOT / "retained/shipped_token_stream.rc64", parts.token_stream)
    envelope = b"R6D1" + parts.token_stream
    envelope += b"\0" * (-len(envelope) % 4)
    blob(ROOT / "retained/rank_control.envelope", envelope)
    # Field copy is a durable checkpoint input; never write into the live tree.
    field = blob(root / "field.u8", FIELD.read_bytes())
    base = {
        "schema": "ddm_tc3_trace_inputs.v1",
        "archive_sha256": ARCHIVE_SHA,
        "field": field,
        "sources": sources,
        "seed": SEED,
        "axis": AXIS,
        "score_claim": False,
        "upstream_evaluate": jg2.file_fact(REPO / "upstream/evaluate.py"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "retention": "all payloads and stage checkpoints kept; certify-or-block; 8 GiB store cap",
        "producer_source": jg2.file_fact(Path(__file__)),
        "free_bytes": shutil.disk_usage(ROOT.parent).free,
        "reserve_bytes": RESERVE,
    }
    path = root / "INPUTS.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("archive_sha256", "field", "sources", "seed", "upstream_evaluate", "producer_source"):
            if previous[key] != base[key]:
                raise ValueError(f"input binding changed: {key}")
        return previous
    return record(path, base)


def run(stage: str, stop: int) -> dict:
    """Full unchanged-field causal encode with a twin source-stream control."""
    import torch

    if stage != "trace":
        raise ValueError("TC3 only retains the unchanged source trajectory")
    inputs = prepare()
    root = ROOT / "move42"
    work = root / stage
    storage(work / "build")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    route = jg2.load_route_b()
    build_path = work / "BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        for key in ("base_source", "generated", "library"):
            if jg2.file_fact(Path(build[key]["path"])) != build[key]:
                raise ValueError("retained native build changed")
        library = Path(build["library"]["path"])
    else:
        library, build = jg2.compile_rc64(work / "build", route, "rlc3_" + stage)
        record(build_path, build)
    residual, renderer, renderer_dir = jg2.load_runtime(root / "source_runtime")
    from runtime import tc1_receiver_checkpoint as native_state
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.native_free_corrector import NativeFreeCorrector

    from experiments import ddm_tc1_public_proof as native_support

    parts = residual.read_residual_archive(root / "source_runtime/archive.zip")
    weights = (ROOT / "retained/tc1_weights.bin").read_bytes()
    if parts.token_stream != (ROOT / "retained/shipped_token_stream.rc64").read_bytes() or parts.tc1_weights != weights:
        raise ValueError("move42 reader changed the shipped stream or counted weights")
    target = jg2.load_tokens(Path(inputs["field"]["path"]))
    model = renderer.load_hpac(residual.materialize_ihs1(parts.hpac_blob, renderer), torch.device("cpu"))
    sparse = residual._sparse_class(renderer_dir)(model, 384, 512)
    native_build = native_support.build_libraries(root / "source_runtime", work / "corrector_native")
    corrector = NativeFreeCorrector(384 * 512, Path(os.environ["F26_CORRECTOR_NATIVE_LIBRARY"]))
    if corrector.library.f26_corrector_abi_version() != 1:
        raise ValueError("native corrector ABI changed")
    if "#define N_FAMILIES 23 " not in (root / "source_runtime/runtime/f26_corrector_native.c").read_text():
        raise ValueError("native corrector family layout changed")
    holder = native_state.ReceiverCheckpoint.__new__(native_state.ReceiverCheckpoint)
    holder.corrector = corrector
    holder.c = ctypes.cast(corrector.handle, ctypes.POINTER(native_state.CorrectorPrefix)).contents
    if holder.c.plane != 384 * 512:
        raise ValueError("native plane changed")

    def capture_corrector():
        if holder.c.group_open:
            raise ValueError("native group remains open at frame checkpoint")
        values = {k: v.copy() for k, v in holder.arrays().items()}
        values["have_prev"] = np.array([holder.c.have_prev], dtype=np.int64)
        return values

    def restore_corrector(values):
        if holder.c.group_open:
            raise ValueError("restore requires a closed fresh native handle")
        views = holder.arrays()
        if set(values) != set(views) | {"have_prev"}:
            raise ValueError("native checkpoint table census changed")
        for name, view in views.items():
            value = values[name]
            if value.dtype != view.dtype or value.shape != view.shape:
                raise ValueError("native checkpoint table shape or dtype changed")
            view[:] = value
        have_prev = values["have_prev"]
        if have_prev.shape != (1,) or have_prev.dtype != np.int64 or int(have_prev[0]) != 1:
            raise ValueError("completed nonempty frame must retain one previous plane")
        holder.c.have_prev = int(have_prev[0])

    mixer = tc1.SharedMixer(weights)
    binding = {
        "inputs_sha": jg2.sha256_file(root / "INPUTS.json"),
        "producer": jg2.sha256_file(Path(__file__)),
        "codec": jg2.sha256_file(Path(tc1.__file__)),
        "jg2": jg2.sha256_file(Path(jg2.__file__)),
        "stage": stage,
        "route_wrapper_sha": jg2.sha256_file(jg2.ROUTE_B),
        "library_sha": jg2.sha256_file(library),
        "native_build": native_build,
        "native_support": jg2.file_fact(Path(native_support.__file__)),
        "native_state_source": jg2.file_fact(Path(native_state.__file__)),
    }
    state, start = None, 0
    latest = work / "LATEST.json"
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt["binding"] != binding or jg2.file_fact(Path(receipt["payload"]["path"])) != receipt["payload"]:
            raise ValueError("checkpoint binding or bytes changed")
        with np.load(receipt["payload"]["path"], allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        start = int(state["frame"][0])
        restore_corrector({k[2:]: v for k, v in state.items() if k.startswith("c_")})
        mixer.restore({k[2:]: v for k, v in state.items() if k.startswith("m_")})
        if mixer.frame != start:
            raise ValueError("mixer/frame checkpoint mismatch")
    if not 0 <= start <= stop <= 600:
        raise ValueError("invalid frame range")
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[k].tobytes())
        for k in ("encoder0", "encoder1")
    ]
    previous = torch.zeros((1, 384, 512), dtype=torch.long)
    if state is not None:
        previous[0] = torch.from_numpy(state["previous"].astype(np.int64))
    groups = [np.flatnonzero(m.cpu().numpy().reshape(-1)) for m in renderer.group_masks(torch.device("cpu"))]
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(384 * 512)):
        raise ValueError("group plan is not a partition")
    blob(work / "group_order.i32", np.concatenate(groups).astype("<i4").tobytes())
    started = time.monotonic()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for frame in range(start, stop):
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
            truth = np.asarray(target[frame]).reshape(-1).copy()
            rows = np.empty((384 * 512, 5), dtype=np.float32)
            raw_rows = np.empty_like(rows)
            best = np.empty(384 * 512, dtype=np.uint8)
            for group, positions in enumerate(groups):
                logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = logits.argmax(axis=1).astype(np.int64)
                feature = boundary[positions].astype(np.int64) * 5 + predicted
                probability = residual._probability_table(
                    logits + parts.table.values[feature], renderer.HPAC_LOGIT_PRECISION
                )
                cs = corrector.group_state(probability, predicted, positions)
                raw_rows[positions] = corrector.coding_row(cs)
                coding = mixer.coding(raw_rows[positions], positions, current[0].numpy().astype(np.uint8), previous_cpu)
                symbols = truth[positions].astype(np.int32)
                rows[positions] = coding
                best[positions] = tc1.frequencies(coding).argmax(axis=1).astype(np.uint8)
                for encoder in encoders:
                    encoder.encode(symbols, coding)
                corrector.observe(cs, symbols.astype(np.int64))
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(symbols.astype(np.int64))
            plane = current[0].numpy().astype(np.uint8)
            np.testing.assert_array_equal(plane.reshape(-1), truth)
            corrector.end_frame(plane.reshape(-1))
            mixer.end_frame(plane, previous_cpu)
            previous = current
            values = {"best": best, "miss_pos": np.flatnonzero(best != truth).astype(np.int32), "tokens": plane}
            if stage == "trace":
                values["rows"] = rows
                values["raw_rows"] = raw_rows
            path = work / "frames" / f"frame_{frame:04d}.npz"
            fact = arrays(path, values)
            record(
                path.with_suffix(".json"),
                {
                    "payload": fact,
                    "frame": frame,
                    "source_field_sha256": FIELD_SHA,
                    "plane_sha256": hashlib.sha256(plane.tobytes()).hexdigest(),
                    "stage": stage,
                    "misses": len(values["miss_pos"]),
                    "binding": binding,
                },
            )
            if (frame + 1) % 20 == 0 or frame + 1 == stop:
                captured = capture_corrector()
                values = {"c_" + k: v for k, v in captured.items()}
                values.update({"m_" + k: v for k, v in mixer.snapshot().items()})
                values.update(
                    frame=np.array([frame + 1]),
                    previous=plane,
                    encoder0=np.frombuffer(encoders[0].snapshot(), dtype=np.uint8),
                    encoder1=np.frombuffer(encoders[1].snapshot(), dtype=np.uint8),
                )
                fact = arrays(work / "checkpoints" / f"stage_{frame + 1:04d}.npz", values)
                receipt = {"binding": binding, "frame": frame + 1, "payload": fact}
                record(work / f"STAGE_{frame + 1:04d}.json", receipt)
                record(latest, receipt)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "frame": frame + 1,
                        "misses": int((best != truth).sum()),
                        "elapsed_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    payloads = [encoder.finish() for encoder in encoders]
    facts = [
        blob(work / f"{tag}_{stop:04d}.envelope", value)
        for tag, value in zip(("primary", "repeat"), payloads, strict=False)
    ]
    if payloads[0] != payloads[1]:
        raise ValueError("independent arithmetic encoders differ")
    identity = payloads[0] == (ROOT / "retained/rank_control.envelope").read_bytes()
    if stage == "trace" and stop == 600 and not identity:
        raise ValueError("full shipped-field identity control failed")
    stage_digest = hashlib.sha256()
    for frame in range(stop):
        frame_path = work / "frames" / f"frame_{frame:04d}.npz"
        frame_receipt = json.loads(frame_path.with_suffix(".json").read_text())
        if jg2.file_fact(frame_path) != frame_receipt["payload"]:
            raise ValueError("completed stage frame custody changed")
        with np.load(frame_path, allow_pickle=False) as data:
            stage_digest.update(data["tokens"].tobytes())
    result = {
        "schema": "ddm_tc3_trace.v1",
        "axis": AXIS,
        "score_claim": False,
        "stage": stage,
        "frames": stop,
        "binding": binding,
        "build": build,
        "payloads": facts,
        "twin_byte_identical": True,
        "full_control_byte_identical": identity,
        "source_field_sha256": FIELD_SHA,
        "field_sha256": stage_digest.hexdigest(),
        "resumed_from_frame": start,
        "interpretation": "unchanged move42 field, before/after TC1 source rows",
    }
    return record(work / ("RESULT.json" if stop == 600 else f"PARTIAL_{stop:04d}.json"), result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("trace",))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("resume root differs from assigned arm store")
    print(json.dumps(run(args.stage, args.stop_after), sort_keys=True))


if __name__ == "__main__":
    main()
