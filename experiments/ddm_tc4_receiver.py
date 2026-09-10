#!/usr/bin/env python3
"""Seal preparation and literal public receiver identity for selected TC4 bytes.

Scorer-free. MAIN alone fires exact evaluation. The source/candidate runtime,
actual shell invocation, checkpoints, raw output and structured smoke are pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments import ddm_tc1_public_proof as proof
from experiments import ddm_tc4_price as price

ROOT = price.ROOT
SOURCE = price.SOURCE / "candidate_runtime"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("nonunique runtime port: " + old)
    return text.replace(old, new)


def binding(stage):
    current = {
        "inputs": price.pin("receiver_price_binding"),
        "producer": price.fact(Path(__file__)),
        "proof_helper": price.fact(Path(proof.__file__)),
        "fast": price.fact(REPO / "experiments/ddm_tc4_fast.py"),
        "source_public_receipt": price.fact(price.SOURCE / "PUBLIC_IDENTITY.json"),
        "selection": price.fact(ROOT / "SELECTION.json"),
    }
    path = ROOT / "receiver" / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise ValueError("receiver source binding changed")
    price.record(path, current)
    if (ROOT / "CANDIDATE.json").exists():
        candidate = json.loads((ROOT / "CANDIDATE.json").read_text())
        if candidate["binding"] != current or candidate["identity"] != proof.identity(ROOT / "candidate_runtime"):
            raise ValueError("candidate runtime binding changed")
    price.record(ROOT / "receiver" / (stage + ".json"), {"binding": current})
    return current


def stage():
    pin = binding("stage")
    selection = json.loads((ROOT / "SELECTION.json").read_text())
    if selection["selected_saved_bytes"] < 20:
        raise ValueError("no admitted receiver candidate")
    mask = selection["selected_mask"]
    result = json.loads((ROOT / "encode" / str(mask) / "RESULT.json").read_text())
    if result["binding"] != pin["inputs"] or result["frames"] != 600 or not result["twin_identical"]:
        raise ValueError("selected encode custody mismatch")
    for item in result["riders"] + result["archives"]:
        if price.fact(item["path"]) != item:
            raise ValueError("retained twin artifact drift")
    if result["archive_saved_bytes"] != selection["selected_saved_bytes"]:
        raise ValueError("selection price mismatch")
    destination = ROOT / "candidate_runtime"
    if destination.exists():
        ready_path = ROOT / ("CANDIDATE.json" if (ROOT / "CANDIDATE.json").exists() else "STAGING_READY.json")
        ready = json.loads(ready_path.read_text())
        if (
            ready["binding"] != pin
            or ready["identity"] != proof.identity(destination)
            or ready["mask"] != mask
            or any(ready["archive"][k] != result["archives"][0][k] for k in ("bytes", "sha256"))
        ):
            raise ValueError("completed staging recovery drift")
        return price.record(ROOT / "CANDIDATE.json", ready)
    candidate = ROOT / "staging_runtime"
    price.storage(candidate, 4 * 1024**2)
    shutil.copytree(
        SOURCE, candidate, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib")
    )
    (candidate / "archive.zip").write_bytes(Path(result["archives"][0]["path"]).read_bytes())
    for name in ("maps", "fast"):
        text = (REPO / f"experiments/ddm_tc4_{name}.py").read_text()
        if name == "fast":
            text = (
                "from __future__ import annotations\n\n"
                + text[text.index("import numpy as np") : text.index("\ndef validate(")]
            )
        text = text.replace(
            "from experiments import ddm_tc1_mixer_codec as base", "from . import tc1_shared_mixer as base"
        )
        text = text.replace("from experiments.ddm_tc3_mixer import LaneMixer", "from .tc3_mixer import LaneMixer")
        text = text.replace("from experiments import ddm_tc4_maps as maps", "from . import tc4_maps as maps")
        text = text.replace("from experiments.ddm_tc4_maps import", "from .tc4_maps import")
        if "from experiments" in text:
            raise ValueError("unported research dependency")
        compile(text, f"tc4_{name}.py", "exec")
        (candidate / f"runtime/tc4_{name}.py").write_text(text)
    path = candidate / "runtime/residual_archive.py"
    text = path.read_text()
    text = replace_once(
        text,
        '        if tokens[:4] == b"TC3M":',
        '        if tokens[:4] == b"TC4M":\n            from .tc4_maps import unpack_rider\n        elif tokens[:4] == b"TC3M":',
    )
    text = replace_once(
        text,
        "    tc3 = parts.tc1_weights is not None and len(parts.tc1_weights) == 41\n    if tc3:",
        "    tc4 = parts.tc1_weights is not None and len(parts.tc1_weights) > 41\n    tc3 = parts.tc1_weights is not None and len(parts.tc1_weights) >= 41\n    if tc4:\n        from .tc4_fast import FastContextMixer\n        mixer = FastContextMixer(parts.tc1_weights)\n    elif tc3:",
    )
    text = replace_once(
        text,
        '"tail_mixer": "TC3M-40-int8" if tc3 else',
        '"tail_mixer": "TC4M-counted-int8" if tc4 else "TC3M-40-int8" if tc3 else',
    )
    compile(text, str(path), "exec")
    path.write_text(text)
    path = candidate / "inflate.py"
    text = replace_once(path.read_text(), price.ARCHIVE_SHA, result["archives"][0]["sha256"])
    text = replace_once(text, "ARCHIVE_BYTES = 180154", f"ARCHIVE_BYTES = {result['archive_bytes']}")
    path.write_text(text)
    changed = sorted(
        str(p.relative_to(candidate))
        for p in candidate.rglob("*")
        if p.is_file()
        and (
            not (SOURCE / p.relative_to(candidate)).exists()
            or p.read_bytes() != (SOURCE / p.relative_to(candidate)).read_bytes()
        )
    )
    if changed != [
        "archive.zip",
        "inflate.py",
        "runtime/residual_archive.py",
        "runtime/tc4_fast.py",
        "runtime/tc4_maps.py",
    ]:
        raise ValueError("unexpected runtime changes: " + repr(changed))
    old_parts = price.io.split_member(price.io.read_archive_member(SOURCE / "archive.zip"))
    new_parts = price.io.split_member(price.io.read_archive_member(candidate / "archive.zip"))
    for name in ("header", "hpac", "semantic", "carrier"):
        if old_parts[name] != new_parts[name]:
            raise ValueError("untargeted section changed: " + name)
    if old_parts["tail"][:96] != new_parts["tail"][:96]:
        raise ValueError("fixed tail prefix changed")
    identity = proof.identity(candidate)
    identity.update(runtime_path=str(destination), archive_path=str(destination / "archive.zip"))
    archive = price.fact(candidate / "archive.zip")
    archive["path"] = str(destination / "archive.zip")
    ready = {
        "binding": pin,
        "identity": identity,
        "archive": archive,
        "mask": mask,
        "changed_paths": changed,
        "score_claim": False,
        "counted_weights": result["fitted_weight_bytes"],
    }
    price.record(ROOT / "STAGING_READY.json", ready)
    candidate.rename(destination)
    return price.record(ROOT / "CANDIDATE.json", ready)


def work_attempt(parent):
    price.storage(parent)
    parent.mkdir(exist_ok=True)
    for i in range(10000):
        p = parent / f"attempt_{i:04d}"
        try:
            p.mkdir()
            return p
        except FileExistsError:
            continue
    raise RuntimeError("attempt namespace exhausted")


def shell_inputs(runtime, work):
    for name in ("data", "output", "scratch"):
        price.storage(work / name)
        (work / name).mkdir(exist_ok=True)
    price.blob(work / "data/p", price.io.read_archive_member(runtime / "archive.zip"))
    price.blob(work / "file_list.txt", b"0.hevc\n")
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
        "CPR1_RC64_LIBRARY",
        "F26_CORRECTOR_NATIVE_LIBRARY",
    ):
        env.pop(key, None)
    return [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ], env


def probe(role, work):
    binding("probe_" + role)
    runtime = ROOT / "candidate_runtime" if role == "candidate" else SOURCE
    price.storage(work, 32 * 1024**2)
    build = proof.build_libraries(runtime, work / "native")
    sys.path.insert(0, str(runtime))
    import torch
    from runtime import f26_inflate as public

    if Path(public.__file__).resolve() != (runtime / "runtime/f26_inflate.py").resolve():
        raise ValueError("wrong runtime import")
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)

    class Reached(Exception):
        pass

    def observer(*args, **kwargs):
        raise Reached()

    public.decode_production_tokens = observer
    for key in (
        "TC1_RECEIVER_CHECKPOINT_DIR",
        "TC1_RECEIVER_STOP_AFTER",
        "F26_ADVISORY_PAIR_LIMIT",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
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
    except Reached:
        pass
    else:
        raise RuntimeError("public token stage not reached")
    return price.record(
        work / "DIRECT.json",
        dict(
            **proof.identity(runtime),
            digest_definition="tac.candidate_seal.measure_runtime_digest",
            outcome="REACHED_TOKEN_DECODE",
            exception_class=None,
            exception_message="",
            public_entrypoint="runtime.f26_inflate.inflate_archive",
            reached_function="runtime.residual_archive.decode_production_tokens",
            seconds=time.monotonic() - started,
            build=build,
        ),
    )


def smoke():
    binding("smoke")
    from tac.candidate_seal import _public_smoke_problems

    block = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": 180,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for role in ("candidate", "frontier"):
        runtime = ROOT / "candidate_runtime" if role == "candidate" else SOURCE
        work = work_attempt(ROOT / "public_smoke" / role)
        child = proof.retained_child(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "probe",
                "--role",
                role,
                "--work",
                str(work),
                "--resume-from",
                str(ROOT),
            ],
            work,
            dict(os.environ),
            180,
        )
        if child["returncode"]:
            raise RuntimeError("public probe failed: " + str(work))
        block["public_path_probes"][role] = dict(**json.loads((work / "DIRECT.json").read_text()), process=child)
        shell = work / "shell"
        command, env = shell_inputs(runtime, shell)
        child = proof.retained_child(command, shell, env, 180)
        errors = [
            line[14:] for line in (shell / "run.log").read_text().splitlines() if line.startswith("RuntimeError: ")
        ]
        if child["returncode"] == 0 or not errors:
            raise ValueError("expected source CUDA gate")
        block["inflate_sh_smokes"][role] = dict(
            **proof.identity(runtime),
            **child,
            digest_definition="tac.candidate_seal.measure_runtime_digest",
            outcome="REACHED_CUDA_GATE",
            exception_class="RuntimeError",
            exception_message=errors[-1],
        )
    problems, observed = _public_smoke_problems(
        block,
        candidate_runtime_dir=ROOT / "candidate_runtime",
        candidate_archive_path=ROOT / "candidate_runtime/archive.zip",
        pointer_archive_sha256=price.ARCHIVE_SHA,
    )
    price.record(ROOT / "PUBLIC_SMOKE.json", block)
    price.record(ROOT / "PUBLIC_SMOKE_VALIDATOR.json", {"problems": problems, "observed": observed})
    if problems:
        raise RuntimeError("structured smoke rejected: " + repr(problems))
    return block


def public(stop, name):
    pin = binding("public_" + name + "_" + str(stop))
    runtime, work = ROOT / "candidate_runtime", ROOT / name
    completion = work / "PUBLIC_IDENTITY.json"
    if stop == 600 and completion.exists():
        value = json.loads(completion.read_text())
        for key in ("candidate_raw", "original_raw", "decoded_field"):
            if price.fact(value[key]["path"]) != value[key]:
                raise ValueError("completed identity payload drift")
        if value["binding"] != pin:
            raise ValueError("completed public binding drift")
        if name == "public_identity":
            price.record(ROOT / "PUBLIC_IDENTITY.json", value)
        return value
    command, env = shell_inputs(runtime, work)
    build = proof.build_libraries(runtime, work / "native")
    for key in ("CPR1_RC64_LIBRARY", "F26_CORRECTOR_NATIVE_LIBRARY"):
        env[key] = os.environ[key]
    env.update(
        TC3_ADVISORY_CPU="1",
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER=str(stop),
    )
    launch_binding = {"binding": pin, "runtime": proof.identity(runtime), "build": build}
    manifest = work / "RUN_BINDING.json"
    if manifest.exists() and json.loads(manifest.read_text()) != launch_binding:
        raise ValueError("public source/build resume drift")
    price.record(manifest, launch_binding)
    latest = work / "frame_checkpoints/LATEST.json"
    if latest.exists():
        last = json.loads(latest.read_text())
        f = price.fact(last["path"])
        if f["sha256"] != last["sha256"] or f["bytes"] != last["bytes"] or last["frame"] > stop:
            raise ValueError("public resume state or boundary mismatch")
        if last["frame"] == stop and stop < 600:
            return {"checkpoint": last, "recovered_complete_stage": True}
    if (work / "output/0.raw").exists():
        completed = sorted((work / f"stage_{stop:04d}").glob("attempt_*/PROCESS.json"))
        if stop != 600 or not completed:
            raise RuntimeError("unreceipted raw retained without completed child; inspect before rerender")
        child = json.loads(completed[-1].read_text())
        log = completed[-1].parent
        if child["argv"] != command or child["log"] != price.fact(log / "run.log"):
            raise ValueError("public completion recovery custody mismatch")
    else:
        price.storage(work / "output/0.raw", 4 * 1024**3 if stop == 600 else 64 * 1024**2)
        log = work_attempt(work / f"stage_{stop:04d}")
        child = proof.retained_child(command, log, env, 3600)
    if stop < 600:
        last = json.loads(latest.read_text())
        if (
            child["returncode"] == 0
            or "TC1_RECEIVER_STAGE_COMPLETE" not in (log / "run.log").read_text()
            or last["frame"] != stop
        ):
            raise RuntimeError("partial receiver failed before requested checkpoint")
        return price.record(log / "RESULT.json", {"binding": pin, "checkpoint": last, "process": child})
    if child["returncode"]:
        raise RuntimeError("literal public inflation failed: " + str(log))
    reports = [
        json.loads(line) for line in (log / "run.log").read_text().splitlines() if line.startswith('{"archive_bytes"')
    ]
    if len(reports) != 1:
        raise ValueError("missing literal public report")
    report = reports[0]
    archive = price.fact(runtime / "archive.zip")
    source_proof = json.loads((price.SOURCE / "PUBLIC_IDENTITY.json").read_text())
    if source_proof["archive"]["sha256"] != price.ARCHIVE_SHA:
        raise ValueError("source public proof is not the pinned archive")
    original, actual = price.fact(source_proof["candidate_raw"]["path"]), price.fact(work / "output/0.raw")
    field = price.fact(work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
    if original != source_proof["candidate_raw"]:
        raise ValueError("source actual public raw custody changed")
    if any(original[k] != actual[k] for k in ("bytes", "sha256")):
        raise ValueError("literal output identity failed")
    if field["sha256"] != price.FIELD_SHA or field["bytes"] != 117964800:
        raise ValueError("public decoded field mismatch")
    if (
        report["archive_sha256"] != archive["sha256"]
        or report["archive_bytes"] != archive["bytes"]
        or report["pair_count"] != 600
    ):
        raise ValueError("public report archive/population mismatch")
    if report["raw_sha256"] != actual["sha256"] or report["raw_bytes"] != actual["bytes"]:
        raise ValueError("public raw report mismatch")
    result = price.record(
        completion,
        {
            "binding": pin,
            "process": child,
            "report": report,
            "archive": archive,
            "candidate_raw": actual,
            "original_raw": original,
            "decoded_field": field,
            "source_public_receipt": price.fact(price.SOURCE / "PUBLIC_IDENTITY.json"),
            "selection": price.fact(ROOT / "SELECTION.json"),
            "all600_field_identity": True,
            "all600_seg_pose_input_byte_identity": True,
            "score_claim": False,
            "axis": "[macOS-CPU advisory literal public output identity, n600]",
            "no_scorer_ran": True,
        },
    )
    if name == "public_identity":
        price.record(ROOT / "PUBLIC_IDENTITY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage", "probe", "smoke", "public"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--role", choices=("candidate", "frontier"), default="candidate")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--stop-after", type=int, default=600)
    parser.add_argument("--name", choices=("public_identity", "public_fresh2"), default="public_identity")
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.stop_after <= 600:
        raise ValueError("invalid receiver root/boundary")
    if args.stage == "probe" and (args.work is None or not args.work.resolve().is_relative_to(ROOT.resolve())):
        raise ValueError("probe work outside owned root")
    result = (
        stage()
        if args.stage == "stage"
        else probe(args.role, args.work)
        if args.stage == "probe"
        else smoke()
        if args.stage == "smoke"
        else public(args.stop_after, args.name)
    )
    print(json.dumps(result), flush=True)
