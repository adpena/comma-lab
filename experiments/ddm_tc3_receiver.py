#!/usr/bin/env python3
"""Stage exactly one TC3 receiver candidate and prove public CPU byte identity.

No scorer or remote dispatch. CPU mode is explicitly advisory; default public
inflation retains the source CUDA gate. MAIN alone consumes the final seal.
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments import ddm_tc1_public_proof as proof
from experiments import ddm_tc3_geometry as geometry
from experiments import ddm_tc3_mixer as codec
from experiments import ddm_tc3_trace as ref

io = ref.jg2
ROOT = ref.ROOT / "move39"
POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"


def receiver_binding():
    inputs = json.loads((ROOT / "INPUTS.json").read_text())
    for item in [*list(inputs["sources"].values()), inputs["field"]]:
        if io.file_fact(Path(item["path"])) != item:
            raise ValueError("source runtime manifest or field changed")
    return {
        "inputs": io.file_fact(ROOT / "INPUTS.json"),
        "sources": [io.file_fact(Path(m.__file__)) for m in (ref, io, codec, geometry, proof)],
        "producer": io.file_fact(Path(__file__)),
    }


def pin(stage):
    """No runtime import here: candidate subprocesses must import their own tree."""
    payload = POINTER.read_bytes()
    pointer = json.loads(payload)
    ref.blob(ROOT / "pointer_snapshots" / (stage + "_" + hashlib.sha256(payload).hexdigest() + ".json"), payload)
    if pointer["effective_frontier"]["archive_sha256"] != ref.ARCHIVE_SHA:
        raise RuntimeError("POINTER_MOVED: rebind explicitly before this receiver stage")
    if io.sha256_file(ROOT / "source_runtime/archive.zip") != ref.ARCHIVE_SHA:
        raise ValueError("source runtime archive changed")
    current = receiver_binding()
    if (ROOT / "CANDIDATE.json").exists():
        candidate = json.loads((ROOT / "CANDIDATE.json").read_text())
        if candidate["receiver_binding"] != current or candidate["runtime_identity"] != proof.identity(
            ROOT / "candidate_runtime"
        ):
            raise ValueError("staged receiver code or runtime changed; preserve and explicitly rebind")
    return pointer


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("receiver rewrite is not unique: " + before[:100])
    return text.replace(before, after)


def attempt(parent):
    """Preserve every failed child log and incomplete staging tree on retry."""
    ref.storage(parent / "marker")
    parent.mkdir(exist_ok=True)
    for number in range(10000):
        path = parent / f"attempt_{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("attempt namespace exhausted")


def stable_libraries(runtime, parent):
    """Select a complete build atomically before any receiver checkpoint exists."""
    ref.storage(parent / "marker")
    selection = parent / "SELECTED.json"
    identity = proof.identity(runtime)
    if selection.exists():
        saved = json.loads(selection.read_text())
        work = Path(saved["directory"])
        if saved["runtime_identity"] != identity or not work.resolve().is_relative_to(parent.resolve()):
            raise ValueError("native selection changed runtime or left its owned directory")
        result = proof.build_libraries(runtime, work)
        if saved["build"] != result:
            raise ValueError("selected complete native build changed")
        return result
    work = attempt(parent / "attempts")
    result = proof.build_libraries(runtime, work)
    ref.record(selection, {"directory": str(work), "runtime_identity": identity, "build": result})
    return result


def pack(member, target, method, level):
    ref.storage(target, len(member) * 2)
    if method == "stored":
        io.pack_archive(member, target)
    else:
        temp = target.with_suffix(".zip.new")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as z:
            info = zipfile.ZipInfo("p", date_time=io.SHIPPED_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = io.SHIPPED_ZIP_EXTERNAL_ATTR
            info.create_system = io.SHIPPED_ZIP_CREATE_SYSTEM
            z.writestr(info, member, compresslevel=level)
        temp.replace(target)
    if io.read_archive_member(target) != member:
        raise ValueError("container changed its member")


def stage():
    import brotli

    pointer = pin("stage")
    source = ROOT / "source_runtime"
    member = io.read_archive_member(source / "archive.zip")
    work = ROOT / "containers"
    control = work / "source_control.zip"
    pack(member, control, "stored", None)
    if io.sha256_file(control) != ref.ARCHIVE_SHA:
        raise ValueError("control archive did not reproduce the pointer SHA")
    section = io.split_member(member)
    prefix = section["header"] + section["hpac"] + section["semantic"] + section["carrier"] + section["tail"][:96]
    census = []
    winners = {}
    for variant in ("A", "B"):
        folder = ROOT / "encode" / variant
        result = json.loads((folder / "RESULT.json").read_text())
        parent_binding = result["binding"]
        for item in parent_binding["sources"] + [parent_binding["producer"], parent_binding["native_wrapper"]]:
            if io.file_fact(Path(item["path"])) != item:
                raise ValueError("encoder dependency changed before staging")
        if parent_binding["archive"] != ref.ARCHIVE_SHA or parent_binding["field"]["sha256"] != ref.FIELD_SHA:
            raise ValueError("encoder measured another field")
        for item in parent_binding["extra"].values():
            if io.file_fact(Path(item["path"])) != item:
                raise ValueError("encoder fit or weights changed before staging")
        rider = (folder / "twin0.rider").read_bytes()
        if (
            not result["full_n600"]
            or not result["twin_byte_identical"]
            or rider != (folder / "twin1.rider").read_bytes()
        ):
            raise ValueError("full n600 twin result required")
        if io.file_fact(folder / "twin0.rider") != result["candidate_rider"]:
            raise ValueError("counted rider changed")
        config, raw = codec.unpack_rider(rider)
        alternatives = [("plain", rider)]
        # Predeclared container samples, same finite menu as TC1. No adaptive search.
        for quality in (9, 10, 11):
            for window in (22, 23, 24):
                packed = brotli.compress(raw, quality=quality, lgwin=window)
                alt = codec.MAGIC + bytes([config[0] | 16]) + config[1:] + packed
                alternatives.append((f"q{quality}_w{window}", alt))
        for label, alt in alternatives:
            rider_path = work / variant / (label + ".rider")
            ref.blob(rider_path, alt)
            if codec.unpack_rider(alt) != (config, raw):
                raise ValueError("sampled rider parse-back differs")
            for method, level in (("stored", None), ("deflate", 1), ("deflate", 6), ("deflate", 9)):
                archive = work / variant / f"{label}_{method}_{level}.zip"
                pack(prefix + alt, archive, method, level)
                census.append(
                    {
                        "variant": variant,
                        "label": label,
                        "method": method,
                        "level": level,
                        "archive": io.file_fact(archive),
                        "rider": io.file_fact(rider_path),
                    }
                )
        winners[variant] = min(
            (x for x in census if x["variant"] == variant), key=lambda x: (x["archive"]["bytes"], x["archive"]["path"])
        )
    saving_b_over_a = winners["A"]["archive"]["bytes"] - winners["B"]["archive"]["bytes"]
    selected = "B" if saving_b_over_a >= 300 else "A"
    winner = winners[selected]
    net_saved = (source / "archive.zip").stat().st_size - winner["archive"]["bytes"]
    comparison = {
        "axis": ref.AXIS,
        "score_claim": False,
        "pointer": pointer["effective_frontier"],
        "source_control": io.file_fact(control),
        "samples": census,
        "winners": winners,
        "selected": selected,
        "b_over_a_archive_saved_bytes": saving_b_over_a,
        "net_archive_saved_bytes": net_saved,
        "selection_rule": "B only if >=300 actual archive bytes better than A",
    }
    ref.record(ROOT / "CONTAINERS.json", comparison)
    if net_saved < 30:
        return ref.record(
            ROOT / "NO_SEAL.json", dict(**comparison, reason="charter receiver candidate byte falsifier: net <30 B")
        )
    candidate = ROOT / "candidate_runtime"
    if candidate.exists():
        completed = ROOT / "CANDIDATE.json"
        old = json.loads((completed if completed.exists() else ROOT / "STAGING_READY.json").read_text())
        if (
            old["archive"] != io.file_fact(candidate / "archive.zip")
            or old["archive"]["sha256"] != winner["archive"]["sha256"]
            or old["runtime_identity"] != proof.identity(candidate)
            or old["receiver_binding"] != receiver_binding()
        ):
            raise ValueError("existing candidate is not the completed staged object")
        if not completed.exists():
            ref.record(completed, old)
        return old
    candidate = attempt(ROOT / "staging_attempts") / "runtime"
    shutil.copytree(source, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib"))
    (candidate / "archive.zip").write_bytes(Path(winner["archive"]["path"]).read_bytes())
    mixer = (REPO / "experiments/ddm_tc3_mixer.py").read_text()
    mixer = replace_once(
        mixer, "from experiments import ddm_tc1_mixer_codec as base", "from . import tc1_shared_mixer as base"
    )
    mixer = replace_once(mixer, "from experiments.ddm_tc3_geometry import", "from .tc3_geometry import")
    (candidate / "runtime/tc3_mixer.py").write_text(mixer)
    (candidate / "runtime/tc3_geometry.py").write_bytes((REPO / "experiments/ddm_tc3_geometry.py").read_bytes())
    path = candidate / "runtime/residual_archive.py"
    text = path.read_text()
    text = replace_once(
        text,
        "        from .tc1_shared_mixer import unpack_rider\n        tc1_weights, tokens = unpack_rider(tokens)",
        '        if tokens[:4] == b"TC3M":\n            from .tc3_mixer import unpack_rider\n        else:\n            from .tc1_shared_mixer import unpack_rider\n        tc1_weights, tokens = unpack_rider(tokens)',
    )
    text = replace_once(
        text,
        "    mixer = None if parts.tc1_weights is None else SharedMixer(parts.tc1_weights)",
        "    tc3 = parts.tc1_weights is not None and len(parts.tc1_weights) == 41\n    if tc3:\n        from .tc3_mixer import LaneMixer\n        mixer = LaneMixer(parts.tc1_weights)\n    else:\n        mixer = None if parts.tc1_weights is None else SharedMixer(parts.tc1_weights)",
    )
    text = replace_once(
        text,
        "                symbols = decoder.decode(coding).astype(np.int64)\n                corrector.observe(state, symbols)",
        "                symbols = decoder.decode(coding).astype(np.int64)\n                if tc3:\n                    mixer.observe(flat_positions, symbols)\n                corrector.observe(state, symbols)",
    )
    text = replace_once(
        text,
        '"tail_mixer": "none" if mixer is None else "TC1M-35-int8",',
        '"tail_mixer": "TC3M-40-int8" if tc3 else ("none" if mixer is None else "TC1M-35-int8"),',
    )
    compile(text, str(path), "exec")
    path.write_text(text)
    path = candidate / "inflate.sh"
    text = path.read_text()
    before = (
        '"$CCBIN" -O3 -std=c11 -shared -fPIC \\\n'
        '  "$HERE/runtime/entropy/rc64_backend.c" \\\n'
        '  -o "$BUILD_DIR/rc64_backend.so"\n'
        'export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"'
    )
    after = (
        'if [[ -n "${CPR1_RC64_LIBRARY:-}" ]]; then\n'
        '  [[ -f "$CPR1_RC64_LIBRARY" ]] || { echo "missing RC64 library" >&2; exit 69; }\n'
        "else\n" + before + "\nfi"
    )
    path.write_text(replace_once(text, before, after))
    path = candidate / "inflate.py"
    text = path.read_text()
    text = replace_once(text, ref.ARCHIVE_SHA, winner["archive"]["sha256"])
    text = replace_once(text, "ARCHIVE_BYTES = 180186", f"ARCHIVE_BYTES = {winner['archive']['bytes']}")
    text = replace_once(
        text,
        "    if not torch.cuda.is_available():",
        '    advisory_cpu = os.environ.get("TC3_ADVISORY_CPU", "0") == "1"\n    if not advisory_cpu and not torch.cuda.is_available():',
    )
    text = replace_once(text, '    device_name = "cuda"', '    device_name = "cpu" if advisory_cpu else "cuda"')
    text = replace_once(
        text,
        "    print(json.dumps(report, sort_keys=True), flush=True)",
        '    if advisory_cpu:\n        report["axis"] = "[macOS-CPU advisory / public receiver identity only]"\n        report["score_claim"] = False\n    print(json.dumps(report, sort_keys=True), flush=True)',
    )
    compile(text, str(path), "exec")
    path.write_text(text)
    changed = sorted(
        str(p.relative_to(candidate))
        for p in candidate.rglob("*")
        if p.is_file()
        and (
            not (source / p.relative_to(candidate)).exists()
            or p.read_bytes() != (source / p.relative_to(candidate)).read_bytes()
        )
    )
    expected = [
        "archive.zip",
        "inflate.py",
        "inflate.sh",
        "runtime/residual_archive.py",
        "runtime/tc3_geometry.py",
        "runtime/tc3_mixer.py",
    ]
    if changed != expected:
        raise ValueError("unexpected runtime change census: " + repr(changed))
    for name in ("header", "hpac", "semantic", "carrier"):
        if io.split_member(io.read_archive_member(candidate / "archive.zip"))[name] != section[name]:
            raise ValueError("untargeted archive section changed: " + name)
    destination = ROOT / "candidate_runtime"
    identity = proof.identity(candidate)
    identity.update(runtime_path=str(destination), archive_path=str(destination / "archive.zip"))
    archive = io.file_fact(candidate / "archive.zip")
    archive["path"] = str(destination / "archive.zip")
    result = dict(
        **comparison,
        archive=archive,
        receiver_binding=receiver_binding(),
        runtime_identity=identity,
        runtime=str(destination),
        changed_paths=changed,
        predictor_code_free=True,
        counted_weights=40,
        no_per_frame_fitted_table=True,
        measured_score=None,
        projected_score=pointer["effective_frontier"]["score"] - net_saved * 25 / 37545489,
        projection_condition="identical public receiver output; contest-CUDA n600 owed to MAIN",
    )
    ref.record(ROOT / "STAGING_READY.json", result)
    candidate.rename(destination)
    return ref.record(ROOT / "CANDIDATE.json", result)


def probe(role, work=None):
    pin("probe_" + role)
    runtime = ROOT / ("candidate_runtime" if role == "candidate" else "source_runtime")
    parent = ROOT / "public_smoke" / role
    if work is None:
        work = attempt(parent)
    elif not work.resolve().is_relative_to(parent.resolve()):
        raise ValueError("probe work must remain in its owned smoke role directory")
    ref.storage(work / "native")
    build = stable_libraries(runtime, work / "native")
    sys.path.insert(0, str(runtime))
    import torch
    from runtime import f26_inflate as public

    if Path(public.__file__).resolve() != (runtime / "runtime/f26_inflate.py").resolve():
        raise ValueError("probe imported another runtime tree")
    torch.manual_seed(ref.SEED)
    torch.use_deterministic_algorithms(True)
    identity = proof.identity(runtime)
    identity["digest_definition"] = "tac.candidate_seal.measure_runtime_digest"
    reached = False

    class ProbeReached(Exception):
        pass

    def observer(*args, **kwargs):
        nonlocal reached
        reached = True
        raise ProbeReached()

    public.decode_production_tokens = observer
    os.environ["F26_TOKEN_DECODER"] = "python"
    for key in (
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "TC1_RECEIVER_CHECKPOINT_DIR",
        "TC1_RECEIVER_STOP_AFTER",
    ):
        os.environ.pop(key, None)
    started = time.monotonic()
    try:
        public.inflate_archive(
            runtime / "archive.zip",
            work / "not_rendered.raw",
            renderer_dir=runtime / "cpr1",
            device_name="cpu",
            num_threads=4,
            checkpoint_dir=work / "checkpoint",
        )
    except ProbeReached:
        pass
    if not reached:
        raise RuntimeError("actual public token stage was not reached")
    return ref.record(
        work / "DIRECT.json",
        dict(
            **identity,
            outcome="REACHED_TOKEN_DECODE",
            exception_class=None,
            exception_message="",
            public_entrypoint="runtime.f26_inflate.inflate_archive",
            reached_function="runtime.residual_archive.decode_production_tokens",
            seconds=time.monotonic() - started,
            build=build,
            scope="observer before token decoder, after real semantic/carrier setup; full decode is separately measured",
        ),
    )


def shell_inputs(runtime, work):
    for name in ("data", "output", "scratch"):
        ref.storage(work / name / "marker")
        (work / name).mkdir(exist_ok=True)
    member = io.read_archive_member(runtime / "archive.zip")
    ref.blob(work / "data/p", member)
    ref.blob(work / "file_list.txt", b"0.hevc\n")
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ.get("PATH", ""),
        TMPDIR=str(work / "scratch"),
        F26_TOKEN_DECODER="python",
    )
    for key in (
        "TC3_ADVISORY_CPU",
        "TC1_RECEIVER_CHECKPOINT_DIR",
        "TC1_RECEIVER_STOP_AFTER",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "F26_ADVISORY_RENDER_WORKERS",
        "F26_ADVISORY_RENDER_RSS_BYTES",
    ):
        env.pop(key, None)
    command = [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ]
    return command, env


def smoke():
    pin("smoke")
    from tac.candidate_seal import _public_smoke_problems

    block = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": 180,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for role in ("candidate", "frontier"):
        runtime = ROOT / ("candidate_runtime" if role == "candidate" else "source_runtime")
        work = attempt(ROOT / "public_smoke" / role)
        process = proof.retained_child(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "probe",
                "--role",
                role,
                "--resume-from",
                str(ref.ROOT),
                "--probe-work",
                str(work),
            ],
            work,
            dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            180,
        )
        if process["returncode"]:
            raise RuntimeError("public probe failed: " + str(work / "run.log"))
        direct = json.loads((work / "DIRECT.json").read_text())
        direct["process"] = process
        block["public_path_probes"][role] = direct
        shell_work = work / "shell"
        command, env = shell_inputs(runtime, shell_work)
        process = proof.retained_child(command, shell_work, env, 180)
        errors = [
            line[len("RuntimeError: ") :]
            for line in (shell_work / "run.log").read_text().splitlines()
            if line.startswith("RuntimeError: ")
        ]
        if process["returncode"] == 0 or not errors:
            raise RuntimeError("shell did not reach the source CUDA gate")
        block["inflate_sh_smokes"][role] = dict(
            **proof.identity(runtime),
            **process,
            digest_definition="tac.candidate_seal.measure_runtime_digest",
            outcome="REACHED_CUDA_GATE",
            exception_class="RuntimeError",
            exception_message=errors[-1],
        )
    problems, observed = _public_smoke_problems(
        block,
        candidate_runtime_dir=ROOT / "candidate_runtime",
        candidate_archive_path=ROOT / "candidate_runtime/archive.zip",
        pointer_archive_sha256=ref.ARCHIVE_SHA,
    )
    ref.record(ROOT / "PUBLIC_SMOKE.json", block)
    ref.record(ROOT / "PUBLIC_SMOKE_VALIDATOR.json", {"problems": problems, "observed": observed})
    if problems:
        raise RuntimeError("public smoke rejected: " + repr(problems))
    return block


def public(stop, proof_name):
    if not 0 < stop <= 600:
        raise ValueError("invalid public checkpoint boundary")
    pin("public_" + proof_name + "_" + str(stop))
    runtime = ROOT / "candidate_runtime"
    work = ROOT / proof_name
    if stop == 600 and (ROOT / "PUBLIC_IDENTITY.json").exists():
        prior = json.loads((ROOT / "PUBLIC_IDENTITY.json").read_text())
        if prior["runtime_identity"] != proof.identity(runtime):
            raise ValueError("completed public receipt names another runtime")
        for key in ("decoded_field", "original_raw", "candidate_raw", "source_parseback_receipt"):
            if io.file_fact(Path(prior[key]["path"])) != prior[key]:
                raise ValueError("completed public identity artifact changed: " + key)
        return prior
    command, env = shell_inputs(runtime, work)
    output = work / "output"
    if (output / "0.raw").exists():
        # A child may finish before its parent records success. Preserve that raw
        # and reuse the complete frame checkpoint with a fresh render destination.
        output = attempt(work / "recovered_outputs")
        command[-2] = str(output)
    ref.storage(output / "0.raw", 5 * 1024**3)
    env.update(
        TC3_ADVISORY_CPU="1",
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER=str(stop),
    )
    build = stable_libraries(runtime, work / "native")
    for key in ("CPR1_RC64_LIBRARY", "F26_CORRECTOR_NATIVE_LIBRARY"):
        env[key] = os.environ[key]
    launch_binding = {"runtime": proof.identity(runtime), "receiver": receiver_binding(), "build": build}
    launch_manifest = work / "RUN_BINDING.json"
    if launch_manifest.exists() and json.loads(launch_manifest.read_text()) != launch_binding:
        raise ValueError("public resume changed its runtime, producer or native build")
    ref.record(launch_manifest, launch_binding)
    run_work = attempt(work / f"stage_{stop:04d}")
    latest_path = work / "frame_checkpoints/LATEST.json"
    if latest_path.exists():
        last = json.loads(latest_path.read_text())
        checkpoint_fact = io.file_fact(Path(last["path"]))
        if any(checkpoint_fact[k] != last[k] for k in ("bytes", "sha256")):
            raise ValueError("public frame checkpoint changed")
        if last["frame"] > stop:
            raise ValueError("requested public stop precedes the durable checkpoint")
        if last["frame"] == stop and stop < 600:
            for path, digest in last["binding"]["sources"].items():
                if io.sha256_file(Path(path)) != digest:
                    raise ValueError("completed partial checkpoint source changed")
            libraries = last["binding"]["libraries"]
            if any(io.sha256_file(Path(env[k])) != digest for k, digest in libraries.items()):
                raise ValueError("completed partial checkpoint native build changed")
            return ref.record(
                run_work / "RESULT.json",
                {"checkpoint": last, "build": build, "recovered_complete_stage": True, "score_claim": False},
            )
    process = proof.retained_child(command, run_work, env, 2400)
    if stop < 600:
        if process["returncode"] == 0 or "TC1_RECEIVER_STAGE_COMPLETE" not in (run_work / "run.log").read_text():
            raise RuntimeError("public partial stage failed before its complete checkpoint")
        last = json.loads((work / "frame_checkpoints/LATEST.json").read_text())
        if last["frame"] != stop:
            raise ValueError("public checkpoint stopped at wrong frame")
        return ref.record(
            run_work / "RESULT.json",
            {
                "process": process,
                "checkpoint": last,
                "build": build,
                "scope": "public implementation/resume control only",
                "score_claim": False,
            },
        )
    if process["returncode"]:
        raise RuntimeError("public inflate.sh failed: " + str(run_work / "run.log"))
    reports = [
        json.loads(line)
        for line in (run_work / "run.log").read_text().splitlines()
        if line.startswith('{"archive_bytes"')
    ]
    if len(reports) != 1:
        raise ValueError("expected one complete public inflate report")
    report = reports[0]
    archive = io.file_fact(runtime / "archive.zip")
    if (
        report["archive_sha256"] != archive["sha256"]
        or report["archive_bytes"] != archive["bytes"]
        or report["pair_count"] != 600
    ):
        raise ValueError("public report archive or n600 custody mismatch")
    original_receipt = ref.LIVE.parents[1] / "parseback/PARSEBACK_RESULT.json"
    original_record = json.loads(original_receipt.read_text())
    if original_record["archive"]["sha256"] != ref.ARCHIVE_SHA:
        raise ValueError("source parse-back receipt is not the current pointer archive")
    original = ref.LIVE.parents[1] / "parseback/0.raw"
    actual = output / "0.raw"
    original_fact, actual_fact = io.file_fact(original), io.file_fact(actual)
    if original_fact != original_record["rendered_raw"]:
        raise ValueError("source raw differs from its retained parse-back receipt")
    if report["raw_bytes"] != actual_fact["bytes"] or report["raw_sha256"] != actual_fact["sha256"]:
        raise ValueError("public report output digest mismatch")
    checkpoint = output / ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    field = io.file_fact(checkpoint)
    if field["sha256"] != ref.FIELD_SHA or field["bytes"] != 117964800:
        raise ValueError("candidate public parse-back field differs")
    if any(original_fact[k] != actual_fact[k] for k in ("bytes", "sha256")):
        raise ValueError("candidate's own public RGB/YUV witness differs from source")
    result = {
        "axis": "[macOS-CPU advisory / actual inflate.sh -> inflate.py, n600]",
        "score_claim": False,
        "process": process,
        "report": report,
        "archive": archive,
        "runtime_identity": proof.identity(runtime),
        "decoded_field": field,
        "original_raw": original_fact,
        "source_parseback_receipt": io.file_fact(original_receipt),
        "candidate_raw": actual_fact,
        "all600_field_identity": True,
        "all600_seg_pose_input_byte_identity": True,
        "no_scorer_ran": True,
        "proof": "all output bytes identical; scorer outputs follow for the same deterministic scorer; no new distortion measurement",
        "contest_cuda_identity_and_score": "not measured here; MAIN exact eval remains required",
    }
    ref.record(run_work / "RESULT.json", result)
    return ref.record(ROOT / "PUBLIC_IDENTITY.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage", "probe", "smoke", "public"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--role", choices=("candidate", "frontier"), default="candidate")
    parser.add_argument("--probe-work", type=Path)
    parser.add_argument("--stop-after", type=int, default=600)
    parser.add_argument("--proof-name", choices=("public_identity", "resume_control"), default="public_identity")
    args = parser.parse_args()
    if args.resume_from.resolve() != ref.ROOT.resolve():
        raise ValueError("wrong receiver resume root")
    result = {
        "stage": stage,
        "probe": lambda: probe(args.role, args.probe_work),
        "smoke": smoke,
        "public": lambda: public(args.stop_after, args.proof_name),
    }[args.stage]()
    print(json.dumps(result, sort_keys=True))
