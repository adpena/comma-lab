#!/usr/bin/env python3
"""RLC1 retained, resumable source control and counted fixed-point twin encode.

Scorer-free research only. Writes only this arm's SSD store, capped at 8 GiB.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as io
from experiments import ddm_rlc1_geometry as geo
from experiments import ddm_rlc1_mixer as mix

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime")
TRACE = Path("/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40")
ARCHIVE_SHA = "986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857"
FIELD_SHA = "b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5"
ORDER = np.concatenate(geo.POSITIONS)
SEED = 20260910


def fact(path):
    return io.file_fact(Path(path))


def storage(need=0):
    used = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    if used + need > 8 * 1024**3 or shutil.disk_usage(ROOT).free < need + 16 * 1024**3:
        raise RuntimeError("STORAGE_BLOCK: retain all payloads; no certified deletion")


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.atomic_json(path, value)
    return value


def blob(path, payload):
    storage(len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    io.persist_immutable_bytes(path, payload, label="RLC1 retained payload")
    return fact(path)


def save(path, values):
    storage(sum(v.nbytes for v in values.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                raise ValueError("immutable checkpoint drift")
    else:
        temp = path.with_suffix(".new")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    return fact(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def prepare():
    storage(128 * 1024**2)
    if fact(SOURCE / "archive.zip")["sha256"] != ARCHIVE_SHA or fact(TRACE / "field.u8")["sha256"] != FIELD_SHA:
        raise ValueError("move40 archive/field custody changed")
    # Use the explicit submittable adjudication; canonical JSON has retracted move41.
    hot = (REPO / ".omx/state/main_hot_state.md").read_bytes()
    if ARCHIVE_SHA.encode() not in hot or b"SUBMITTABLE" not in hot:
        raise ValueError("live board no longer names the assigned submittable base")
    blob(ROOT / "retained/hot_state_at_start.md", hot) if not (
        ROOT / "retained/hot_state_at_start.md"
    ).exists() else None
    residual, _, _ = io.load_runtime(SOURCE)
    parts = residual.read_residual_archive(SOURCE / "archive.zip")
    old = bytes(parts.tc1_weights)
    fitted = (TRACE / "fit/A/weights_i8.bin").read_bytes()
    # ALL content-selected or uncertain constants are counted, not receiver defaults.
    cfg = geo.FORMAT.pack(1, 128, 320, 16, 4, 4, 36, 2, 2, 3, 24, 0, 1, 2, 4, 8, 16)
    geo.parse_config(cfg)
    blob(ROOT / "retained/geometry_config.bin", cfg)
    blob(ROOT / "retained/config.bin", bytes([1]) + old + fitted + cfg)
    blob(ROOT / "retained/source.rc64", parts.token_stream)
    sources = [fact(Path(m.__file__)) for m in (io, geo, mix, mix.base)]
    sources += [fact(Path(__file__)), fact(REPO / "experiments/ddm_rlc1_geometry.c")]
    value = {
        "archive": fact(SOURCE / "archive.zip"),
        "field": fact(TRACE / "field.u8"),
        "config": fact(ROOT / "retained/config.bin"),
        "sources": sources,
        "native": fact(ROOT / "retained/geometry.dylib"),
        "upstream": fact(REPO / "upstream/evaluate.py"),
        "seed": SEED,
        "score_claim": False,
        "axis": "[exact bytes; macOS-CPU scorer-free]",
        "source_trace_result": fact(TRACE / "trace/RESULT.json"),
    }
    path = ROOT / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("RLC1 input/source binding changed; preserve and use new generation")
    record(path, value)
    return value


def encode(stop):
    pin = prepare()
    work = ROOT / "encode"
    work.mkdir(exist_ok=True)
    route = io.load_route_b()
    build_path = work / "BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        for key in ("base_source", "generated", "library"):
            if fact(build[key]["path"]) != build[key]:
                raise ValueError("native encoder changed")
        library = Path(build["library"]["path"])
    else:
        library, build = io.compile_rc64(work, route, "rlc1")
        record(build_path, build)
    os.environ["RLC1_GEOMETRY_LIBRARY"] = str(ROOT / "retained/geometry.dylib")
    mixer = mix.LaneMixer((ROOT / "retained/config.bin").read_bytes())
    latest = work / "LATEST.json"
    start, state = 0, None
    if latest.exists():
        last = json.loads(latest.read_text())
        if last["binding"] != pin or fact(last["state"]["path"]) != last["state"]:
            raise ValueError("encoder resume drift")
        state = arrays(last["state"]["path"])
        start = int(state["frame"][0])
        mixer.restore({k[2:]: v for k, v in state.items() if k.startswith("m_")})
    if start > stop:
        raise ValueError("stop precedes retained state")
    encoders = [
        route.NativeRc64Encoder(library, None if state is None else state[f"enc{i}"].tobytes()) for i in range(3)
    ]
    field = np.memmap(TRACE / "field.u8", mode="r", dtype=np.uint8, shape=(600, geo.H, geo.W))
    started = time.monotonic()
    for frame in range(start, stop):
        path = TRACE / "trace/frames" / f"frame_{frame:04d}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        if fact(path) != receipt["payload"] or receipt["source_field_sha256"] != FIELD_SHA:
            raise ValueError("source frame custody mismatch")
        data = arrays(path)
        plane, previous = field[frame], None if frame == 0 else field[frame - 1]
        np.testing.assert_array_equal(data["tokens"], plane)
        rows = np.empty_like(data["raw_rows"])
        mixer.begin_frame()
        for positions in geo.POSITIONS:
            rows[positions] = mixer.coding(data["raw_rows"][positions], positions, plane, previous)
            mixer.observe(positions, plane.reshape(-1)[positions])
        mixer.end_frame(plane, previous)
        truth = plane.reshape(-1)[ORDER].astype(np.int32)
        for enc in encoders[:2]:
            enc.encode(truth, rows[ORDER])
        encoders[2].encode(truth, data["rows"][ORDER])
        state = {"frame": np.array([frame + 1]), **{"m_" + k: v for k, v in mixer.snapshot().items()}}
        state.update({f"enc{i}": np.frombuffer(enc.snapshot(), dtype=np.uint8) for i, enc in enumerate(encoders)})
        payload = save(work / "states" / f"frame_{frame + 1:04d}.npz", state)
        record(latest, {"binding": pin, "state": payload, "frame": frame + 1})
        print(json.dumps({"frame": frame + 1, "elapsed": time.monotonic() - started}), flush=True)
    if stop != 600:
        return record(work / f"PARTIAL_{stop:04d}.json", {"binding": pin, "frame": stop, "score_claim": False})
    raw_streams = []
    for i, enc in enumerate(encoders):
        envelope = enc.finish()
        blob(work / f"twin{i}.envelope", envelope)
        raw = ctypes.string_at(
            enc.library.rc64_encoder_data(enc.context), int(enc.library.rc64_encoder_size(enc.context))
        )
        blob(work / f"twin{i}.rc64", raw)
        raw_streams.append(raw)
    if raw_streams[0] != raw_streams[1] or raw_streams[2] != (ROOT / "retained/source.rc64").read_bytes():
        raise ValueError("candidate twins or source stream reconstruction failed")
    cfg = (ROOT / "retained/config.bin").read_bytes()
    for i in range(2):
        blob(work / f"twin{i}.rider", mix.MAGIC + cfg + raw_streams[i])
    return record(
        work / "RESULT.json",
        {
            "binding": pin,
            "full_n600": True,
            "positions": 600 * geo.H * geo.W,
            "twins_identical": True,
            "source_stream_identical": True,
            "streams": [fact(work / f"twin{i}.rc64") for i in range(3)],
            "score_claim": False,
            "geometry_config_bytes": geo.FORMAT.size,
        },
    )


def pack(member, path, method, level):
    storage(len(member) * 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if io.read_archive_member(path) != member:
            raise ValueError("existing retained archive member differs")
        return
    if method == "stored":
        io.pack_archive(member, path)
    else:
        temp = path.with_suffix(".new")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as z:
            info = zipfile.ZipInfo("p", date_time=io.SHIPPED_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = io.SHIPPED_ZIP_EXTERNAL_ATTR
            info.create_system = io.SHIPPED_ZIP_CREATE_SYSTEM
            z.writestr(info, member, compresslevel=level)
        temp.replace(path)
    if io.read_archive_member(path) != member:
        raise ValueError("archive parse back mismatch")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("receiver replacement not unique: " + old[:80])
    return text.replace(old, new)


def stage():
    import brotli

    from experiments import ddm_tc1_public_proof as proof

    pin = prepare()
    result = json.loads((ROOT / "encode/RESULT.json").read_text())
    if result["binding"] != pin or not result["twins_identical"] or not result["source_stream_identical"]:
        raise ValueError("full bound twins required")
    member = io.read_archive_member(SOURCE / "archive.zip")
    pack(member, ROOT / "retained/source_control.zip", "stored", None)
    if fact(ROOT / "retained/source_control.zip")["sha256"] != ARCHIVE_SHA:
        raise ValueError("source archive reconstruction failed")
    section = io.split_member(member)
    prefix = b"".join(section[k] for k in ("header", "hpac", "semantic", "carrier")) + section["tail"][:96]
    config, raw = mix.unpack_rider((ROOT / "encode/twin0.rider").read_bytes())
    alternatives = [("plain", mix.MAGIC + config + raw)]
    for quality in (9, 10, 11):
        for window in (22, 23, 24):
            packed = brotli.compress(raw, quality=quality, lgwin=window)
            blob(ROOT / "containers" / f"q{quality}_w{window}.br", packed)
            alternatives.append((f"q{quality}_w{window}", mix.MAGIC + bytes([config[0] | 16]) + config[1:] + packed))
    census = []
    for label, rider in alternatives:
        blob(ROOT / "containers" / (label + ".rider"), rider)
        if mix.unpack_rider(rider) != (config, raw):
            raise ValueError("rider parse-back mismatch")
        for method, level in (("stored", None), ("deflate", 1), ("deflate", 6), ("deflate", 9)):
            path = ROOT / "containers" / f"{label}_{method}_{level}.zip"
            pack(prefix + rider, path, method, level)
            census.append({"archive": fact(path), "rider": fact(ROOT / "containers" / (label + ".rider"))})
    winner = min(census, key=lambda v: (v["archive"]["bytes"], v["archive"]["path"]))
    saved = 180233 - winner["archive"]["bytes"]
    record(
        ROOT / "CONTAINERS.json",
        {"samples": census, "winner": winner, "net_saved_bytes": saved, "config_bytes": 19, "score_claim": False},
    )
    if saved < 30:
        return record(
            ROOT / "NO_SEAL.json",
            {
                "reason": "counted fixed-point candidate saves fewer than 30 bytes",
                "measured_saved": saved,
                "verdict_scope": "INSTANCE",
                "score_claim": False,
            },
        )
    candidate = ROOT / "candidate_runtime"
    if candidate.exists():
        old = json.loads((ROOT / "CANDIDATE.json").read_text())
        if old["identity"] != proof.identity(candidate):
            raise ValueError("candidate runtime drift")
        return old
    candidate.mkdir()
    for path in SOURCE.rglob("*"):
        if path.is_file() and path.suffix not in (".pyc", ".so", ".dylib") and "__pycache__" not in path.parts:
            dest = candidate / path.relative_to(SOURCE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
    (candidate / "archive.zip").write_bytes(Path(winner["archive"]["path"]).read_bytes())
    for name in ("geometry.py", "geometry.c", "mixer.py"):
        text = (REPO / "experiments" / ("ddm_rlc1_" + name)).read_text()
        text = text.replace(
            "from experiments import ddm_tc1_mixer_codec as base", "from . import tc1_shared_mixer as base"
        )
        text = text.replace("from experiments.ddm_rlc1_geometry import", "from .rlc1_geometry import")
        (candidate / "runtime" / ("rlc1_" + name)).write_text(text)
    path = candidate / "runtime/residual_archive.py"
    text = path.read_text()
    text = replace_once(
        text,
        "        from .tc1_shared_mixer import unpack_rider\n        tc1_weights, tokens = unpack_rider(tokens)",
        '        if tokens[:4] == b"RLC1":\n            from .rlc1_mixer import unpack_rider\n        else:\n            from .tc1_shared_mixer import unpack_rider\n        tc1_weights, tokens = unpack_rider(tokens)',
    )
    text = replace_once(
        text,
        "    mixer = None if parts.tc1_weights is None else SharedMixer(parts.tc1_weights)",
        "    rlc1 = parts.tc1_weights is not None and len(parts.tc1_weights) == 60\n    if rlc1:\n        from .rlc1_mixer import LaneMixer\n        mixer = LaneMixer(parts.tc1_weights)\n    else:\n        mixer = None if parts.tc1_weights is None else SharedMixer(parts.tc1_weights)",
    )
    text = replace_once(
        text,
        "                symbols = decoder.decode(coding).astype(np.int64)\n                corrector.observe(state, symbols)",
        "                symbols = decoder.decode(coding).astype(np.int64)\n                if rlc1:\n                    mixer.observe(flat_positions, symbols)\n                corrector.observe(state, symbols)",
    )
    text = replace_once(
        text,
        '"tail_mixer": "none" if mixer is None else "TC1M-35-int8",',
        '"tail_mixer": "RLC1-counted-Q16" if rlc1 else ("none" if mixer is None else "TC1M-35-int8"),',
    )
    path.write_text(text)
    path = candidate / "inflate.sh"
    text = path.read_text()
    text = replace_once(
        text,
        'mkdir -p "$OUTPUT_DIR"',
        '"$CCBIN" -O3 -std=c11 -shared -fPIC "$HERE/runtime/rlc1_geometry.c" -o "$BUILD_DIR/rlc1_geometry.so"\nexport RLC1_GEOMETRY_LIBRARY="$BUILD_DIR/rlc1_geometry.so"\n\nmkdir -p "$OUTPUT_DIR"',
    )
    # Keep the actual shell compiling geometry; never override it for proof.
    path.write_text(text)
    path = candidate / "inflate.py"
    text = (
        path.read_text()
        .replace(ARCHIVE_SHA, winner["archive"]["sha256"])
        .replace("ARCHIVE_BYTES = 180233", f"ARCHIVE_BYTES = {winner['archive']['bytes']}")
    )
    text = replace_once(
        text,
        "    if not torch.cuda.is_available():",
        '    advisory_cpu = os.environ.get("RLC1_ADVISORY_CPU") == "1"\n    if not advisory_cpu and not torch.cuda.is_available():',
    )
    text = replace_once(text, '    device_name = "cuda"', '    device_name = "cpu" if advisory_cpu else "cuda"')
    text = replace_once(
        text, '    os.environ[_name] = "4"', '    os.environ[_name] = os.environ.get("RLC1_PROOF_BLAS_THREADS", "4")'
    )
    text = replace_once(
        text,
        "    print(json.dumps(report, sort_keys=True), flush=True)",
        '    if advisory_cpu:\n        report["axis"] = "[macOS-CPU advisory]"\n        report["score_claim"] = False\n    print(json.dumps(report, sort_keys=True), flush=True)',
    )
    path.write_text(text)
    for name in ("header", "hpac", "semantic", "carrier"):
        if io.split_member(io.read_archive_member(candidate / "archive.zip"))[name] != section[name]:
            raise ValueError("untargeted archive section changed")
    changed = sorted(
        str(p.relative_to(candidate))
        for p in candidate.rglob("*")
        if p.is_file()
        and (
            not (SOURCE / p.relative_to(candidate)).exists()
            or p.read_bytes() != (SOURCE / p.relative_to(candidate)).read_bytes()
        )
    )
    return record(
        ROOT / "CANDIDATE.json",
        {
            "identity": proof.identity(candidate),
            "changed_paths": changed,
            "source_archive": fact(SOURCE / "archive.zip"),
            "archive": fact(candidate / "archive.zip"),
            "net_saved_bytes": saved,
            "score_claim": False,
            "binding": pin,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "encode", "stage"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.stop_after <= 600:
        raise ValueError("incorrect explicit resume root/stop")
    result = {"prepare": prepare, "encode": lambda: encode(args.stop_after), "stage": stage}[args.stage]()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
