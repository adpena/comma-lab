#!/usr/bin/env python3
"""Retained RC3 container sweep, isolated staging, and bounded public smoke.

No scorer or Modal dispatch. Each command is a disk-resumable stage. The two
public smokes run sequentially, each in its own process group with a 60 s bound.
Only an observed function-entry marker licenses REACHED_TOKEN_DECODE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import brotli
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ddm_rc3_shared_mixer_codec as codec
from experiments import ddm_rc3_shared_mixer_pricing as p


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer_guard() -> dict:
    value = json.loads((p.REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if value["effective_frontier"]["archive_sha256"] != p.PIN:
        raise RuntimeError("Live pointer moved; rebase only HPAC against its new tree before staging/sealing")
    assert sha(p.LIVE / "archive.zip") == p.PIN
    return value["our_local_frontier_contest_cuda"]


def sweep() -> dict:
    target = p.STORE / "CONTAINER_SWEEP.json"
    if target.exists() and not (p.STORE / "CONTAINER_INPUTS.json").exists():
        preserve_incomplete_receipt(target)
    if target.exists():
        r = json.loads(target.read_text())
        binding = json.loads((p.STORE / "CONTAINER_INPUTS.json").read_text())
        if binding != container_binding():
            raise ValueError("container inputs changed since completed sweep")
        for c in r["cells"]:
            assert sha(Path(c["payload"]["path"])) == c["payload"]["sha256"]
        return r
    race = json.loads((p.STORE / "RACE.json").read_text())
    riders = [("rc2_base", p.STORE / "retained/source/rider.rc2h")]
    riders += [(r["slug"], Path(r["artifacts"]["rider"]["path"])) for r in race["rows"]]
    cells = []
    for slug, path in riders:
        rider = path.read_bytes()
        for ck2 in (False, True):
            transformed = p.prior.rc1.ck2_interleave(rider) if ck2 else rider
            p.retain(p.STORE / "retained/grid" / slug / f"input_ck{int(ck2)}.bin", transformed)
            for q in (9, 10, 11):
                for window in (22, 23, 24):
                    raw = brotli.compress(transformed, quality=q, lgwin=window)
                    fact = p.retain(p.STORE / "retained/grid" / slug / f"ck{int(ck2)}_q{q}_w{window}.br", raw)
                    assert brotli.decompress(raw) == transformed
                    cells.append(
                        {
                            "slug": slug,
                            "ck2": ck2,
                            "q": q,
                            "lgwin": window,
                            "bytes": len(raw),
                            "payload": fact,
                            "shippable": ck2,
                            "identical_to_shipped": raw == (p.STORE / "retained/source/hpac.br").read_bytes(),
                        }
                    )
    controls = [c for c in cells if c["slug"] == "rc2_base" and c["identical_to_shipped"]]
    assert controls and all(c["bytes"] == 12112 for c in controls)
    shipped = min(controls, key=lambda c: (c["q"], c["lgwin"]))

    def key(c):
        return (c["bytes"], c["q"] != shipped["q"], c["lgwin"] != shipped["lgwin"], c["lgwin"], c["slug"])

    winner = min((c for c in cells if c["shippable"]), key=key)
    r = {
        "axis": p.AXIS,
        "score_claim": False,
        "cells": cells,
        "winner": winner,
        "selection": "archive-sized HPAC; ck2=true only; size ties prefer byte-identity-proven shipped shape",
        "shipped_shape": {k: shipped[k] for k in ("ck2", "q", "lgwin")},
        "ck2_false_disposition": "measurement only: no unused unambiguous RX1 flag",
        "control_byte_identity": True,
    }
    p.save_json(target, r)
    p.save_json(p.STORE / "CONTAINER_INPUTS.json", container_binding())
    return r


def container_binding() -> dict:
    race = json.loads((p.STORE / "RACE.json").read_text())
    return {
        "race_sha256": sha(p.STORE / "RACE.json"),
        "base_rider_sha256": sha(p.STORE / "retained/source/rider.rc2h"),
        "riders": {r["slug"]: sha(Path(r["artifacts"]["rider"]["path"])) for r in race["rows"]},
    }


def stage_binding() -> dict:
    from tac.candidate_seal import measure_runtime_digest

    root = p.STORE / "candidate_runtime"
    return {
        "runtime_sha256": measure_runtime_digest(root).sha256,
        "live_runtime_sha256": measure_runtime_digest(p.LIVE).sha256,
        "codec_sha256": sha(Path(codec.__file__)),
        "base_codec_sha256": sha(Path(codec.base.__file__)),
        "race_sha256": sha(p.STORE / "RACE.json"),
        "container_sweep_sha256": sha(p.STORE / "CONTAINER_SWEEP.json"),
        "source_sha256": sha(p.STORE / "retained/source/body.ihs1"),
        "staged_receipt_sha256": sha(p.STORE / "STAGED.json"),
    }


def preserve_partial_stage(root: Path) -> None:
    """Certify and retain an interrupted stage; resume from the complete race."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = p.STORE / "retained/interrupted_stage" / stamp
    dest.mkdir(parents=True)
    facts = [
        {"relative_path": str(f.relative_to(root)), "bytes": f.stat().st_size, "sha256": sha(f)}
        for f in sorted(root.rglob("*"))
        if f.is_file()
    ]
    p.save_json(
        dest / "CUSTODY.json",
        {
            "original_path": str(root),
            "destination": str(dest / "runtime"),
            "files": facts,
            "reason": "incomplete stage retained intact; resume from RACE.json",
            "command": sys.argv,
            "score_claim": False,
        },
    )
    os.rename(root, dest / "runtime")


def preserve_incomplete_receipt(path: Path) -> None:
    """A receipt without its binding is an interrupted stage, never a completion."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = p.STORE / "retained/interrupted_receipts" / stamp
    dest.mkdir(parents=True)
    p.save_json(
        dest / "CUSTODY.json",
        {
            "original_path": str(path),
            "destination": str(dest / path.name),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "command": sys.argv,
            "reason": "completion receipt lacks input binding; retained before replaying stage",
            "score_claim": False,
        },
    )
    os.rename(path, dest / path.name)


def stage() -> dict:
    pointer = pointer_guard()
    target = p.STORE / "STAGED.json"
    if target.exists() and not (p.STORE / "STAGED_INPUTS.json").exists():
        preserve_incomplete_receipt(target)
    if target.exists():
        r = json.loads(target.read_text())
        assert sha(Path(r["archive"]["path"])) == r["archive"]["sha256"]
        if json.loads((p.STORE / "STAGED_INPUTS.json").read_text()) != stage_binding():
            raise ValueError("completed stage source/runtime inputs drifted")
        return r
    winner = sweep()["winner"]
    if winner["bytes"] >= 12112:
        raise RuntimeError("no smaller container to stage")
    root = p.STORE / "candidate_runtime"
    if root.exists():
        preserve_partial_stage(root)
    p.prior.LIVE_ROOT = p.LIVE
    p.prior.copy_live_tree(root)
    shutil.copy2(Path(codec.__file__), root / "runtime/rc3_shared_mixer.py")
    patch = p.prior.patch_once
    patch(
        root / "runtime/residual_archive.py",
        "        from .rc2_hpac_semistatic_mixing import MAGIC as RC2_HPAC_MAGIC",
        "        from .rc2_hpac_semistatic_mixing import MAGIC as RC2_HPAC_MAGIC\n        from .rc3_shared_mixer import MAGIC as RC3_HPAC_MAGIC",
    )
    patch(
        root / "runtime/residual_archive.py",
        "hpac.startswith((RC1_HPAC_MAGIC, RC2_HPAC_MAGIC))",
        "hpac.startswith((RC1_HPAC_MAGIC, RC2_HPAC_MAGIC, RC3_HPAC_MAGIC))",
    )
    patch(
        root / "runtime/ihs2.py",
        "from .rc2_hpac_semistatic_mixing import restore_hpac as restore_rc2_hpac",
        "from .rc2_hpac_semistatic_mixing import restore_hpac as restore_rc2_hpac\nfrom .rc3_shared_mixer import MAGIC as RC3_HPAC_MAGIC\nfrom .rc3_shared_mixer import restore_hpac as restore_rc3_hpac",
    )
    patch(
        root / "runtime/ihs2.py",
        "    if blob.startswith(RC2_HPAC_MAGIC):",
        "    if blob.startswith(RC3_HPAC_MAGIC):\n        return restore_rc3_hpac(blob, layout_from_runtime(runtime).row_counts)\n    if blob.startswith(RC2_HPAC_MAGIC):",
    )
    # Build twice AFTER staging; the runtime codec bytes must equal the encoder source.
    assert (root / "runtime/rc3_shared_mixer.py").read_bytes() == Path(codec.__file__).read_bytes()
    race = json.loads((p.STORE / "RACE.json").read_text())
    row = next(r for r in race["rows"] if r["slug"] == winner["slug"])
    rider = Path(row["artifacts"]["rider"]["path"]).read_bytes()
    counts = json.loads((p.STORE / "retained/source/layout.json").read_text())["row_counts"]
    body = (p.STORE / "retained/source/body.ihs1").read_bytes()
    weights = np.frombuffer(Path(row["artifacts"]["parameters"]["path"]).read_bytes()[1:], dtype=np.int8)
    family = next(k for k, v in codec.FAMILIES.items() if v == row["family"])
    parts = p.prior.jg2.split_member((p.STORE / "retained/source/member.rx1").read_bytes())
    source = {"sections": parts, "header": p.prior.RX1_HEADER.unpack(parts["header"])}
    archives = []
    for trial in (1, 2):
        built, details = codec.encode(body, counts, family, weights, row["learning_shift"])
        retained = p.STORE / "retained/staged_twins" / str(trial)
        p.retain(retained / "rider.rc3h", built)
        p.retain(retained / "range.bin", details["payload"])
        p.retain(retained / "parameters.bin", details["parameters"])
        assert built == rider
        outer = p.prior.rc1.ck2_interleave(built)
        p.retain(retained / "rider.ck2", outer)
        container = brotli.compress(outer, quality=winner["q"], lgwin=winner["lgwin"])
        p.retain(retained / "hpac.br", container)
        assert container == Path(winner["payload"]["path"]).read_bytes()
        member = p.prior.build_member(source, container)
        p.retain(retained / "member.rx1", member)
        archive = retained / "archive.zip"
        p.prior.jg2.pack_archive(member, archive)
        archives.append(archive)
    assert archives[0].read_bytes() == archives[1].read_bytes()
    shutil.copy2(archives[0], root / "archive.zip")
    archive = (root / "archive.zip").read_bytes()
    p.prior.repin_receiver(root, archive)
    p.prior.regenerate_manifest(root)
    new = p.prior.jg2.split_member(p.prior.jg2.read_archive_member(root / "archive.zip"))
    census = {
        k: {
            "identical": parts[k] == new[k],
            "source_bytes": len(parts[k]),
            "candidate_bytes": len(new[k]),
            "source_sha256": hashlib.sha256(parts[k]).hexdigest(),
            "candidate_sha256": hashlib.sha256(new[k]).hexdigest(),
        }
        for k in ("semantic", "carrier", "tail")
    }
    assert all(c["identical"] for c in census.values())
    decode_parent = p.STORE / "retained/staged_public_decode"
    decode_parent.mkdir(parents=True, exist_ok=True)
    work = decode_parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    work.mkdir()
    decode_script = """import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]);sys.path[:0]=[str(root/'cpr1'),str(root)]
import inflate
from runtime.residual_archive import read_residual_archive
from runtime.ihs2 import materialize_ihs1
body=materialize_ihs1(read_residual_archive(root/'archive.zip').hpac_blob,inflate)
with out.open('xb') as f: f.write(body)
"""
    p.retain(work / "probe.py", decode_script.encode())
    proof = bounded(
        [sys.executable, "-B", str(work / "probe.py"), str(root), str(work / "decoded.ihs1")],
        work,
        {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        "decode",
    )
    if proof["returncode"] != 0 or (work / "decoded.ihs1").read_bytes() != body:
        raise RuntimeError("staged public receiver did not restore exact retained IHS1")
    decoded = {
        "bytes": len(body),
        "sha256": sha(work / "decoded.ihs1"),
        "identity": True,
        "retained_path": str(work / "decoded.ihs1"),
        "process": proof,
    }
    changed = []
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(root)
            if not (p.LIVE / rel).is_file() or path.read_bytes() != (p.LIVE / rel).read_bytes():
                changed.append(str(rel))
    assert set(changed) == {
        "archive.zip",
        "MANIFEST.sha256",
        "inflate.py",
        "runtime/ihs2.py",
        "runtime/residual_archive.py",
        "runtime/rc3_shared_mixer.py",
    }
    r = {
        "axis": p.AXIS,
        "score_claim": False,
        "archive": p.retain(root / "archive.zip", archive),
        "runtime": str(root),
        "winner": winner,
        "twin_identical": True,
        "model_decode": decoded,
        "section_census": census,
        "runtime_changed_files": sorted(changed),
        "live_pointer": pointer,
        "net_archive_bytes": len(archive) - 181414,
        "projected_score": pointer["score"] + (len(archive) - 181414) * 25 / 37545489,
        "projection_only": True,
        "cleanup": "all bytes retained; runtime is under owned SSD root",
    }
    p.save_json(target, r)
    p.save_json(p.STORE / "STAGED_INPUTS.json", stage_binding())
    return r


def bounded(argv: list[str], work: Path, env: dict, label: str) -> dict:
    """Retain stdout/stderr and kill the entire process group at the bound."""
    started = time.monotonic()
    with (work / (label + ".stdout")).open("wb") as stdout, (work / (label + ".stderr")).open("wb") as stderr:
        proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    return {
        "seconds": time.monotonic() - started,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "argv": argv,
        "stdout": str(work / (label + ".stdout")),
        "stderr": str(work / (label + ".stderr")),
        "timeout_policy": "60s; SIGKILL entire new process group; wait/reap",
    }


def smoke(role: str) -> dict:
    pointer_guard()
    root = p.LIVE if role == "frontier" else p.STORE / "candidate_runtime"
    sys.path.insert(0, str(p.REPO / "src"))
    from tac.candidate_seal import measure_runtime_digest

    tree_before = measure_runtime_digest(root).sha256
    receipt = p.STORE / f"SMOKE_{role}.json"
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r["direct"]["tree_sha256"] == tree_before and r["direct"]["archive_sha256"] == sha(root / "archive.zip")
        return r
    parent = p.STORE / "smoke" / role
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent / f"attempt_{attempt:03d}").exists():
        attempt += 1
    work = parent / f"attempt_{attempt:03d}"
    work.mkdir()
    (work / "scratch").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(work / "scratch"),
        "PATH": f"{Path(sys.executable).parent}" + os.pathsep + os.environ["PATH"],
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    }
    lib = work / "rc64_backend.so"
    compile_result = bounded(
        ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(root / "runtime/entropy/rc64_backend.c"), "-o", str(lib)],
        work,
        env,
        "compile",
    )
    if compile_result["returncode"] != 0:
        raise RuntimeError("smoke compiler failed; retained compiler logs identify the failure")
    env["CPR1_RC64_LIBRARY"] = str(lib)
    script = """import json,sys
from pathlib import Path
root,work=map(Path,sys.argv[1:]);sys.path.insert(0,str(root))
from runtime.f26_inflate import inflate_archive
def observe(frame,event,arg):
 if event=='call' and frame.f_code.co_name=='decode_production_tokens' and Path(frame.f_code.co_filename).resolve()==root/'runtime/residual_archive.py':
  (work/'TOKEN_DECODE_ENTERED.json').write_text(json.dumps({'filename':frame.f_code.co_filename,'function':frame.f_code.co_name,'attempt':str(work),'runtime':str(root)}))
  sys.setprofile(None)
sys.setprofile(observe)
inflate_archive(root/'archive.zip',work/'0.raw',renderer_dir=root/'cpr1',device_name='cpu',num_threads=4,checkpoint_dir=work/'.ckpt')
"""
    p.retain(work / "direct_probe.py", script.encode())
    direct = bounded([sys.executable, "-B", str(work / "direct_probe.py"), str(root), str(work)], work, env, "direct")
    marker = work / "TOKEN_DECODE_ENTERED.json"
    if not marker.exists() or not direct["timed_out"]:
        raise RuntimeError(f"public token entry not proven: {direct}")
    observed = json.loads(marker.read_text())
    assert observed["attempt"] == str(work) and observed["runtime"] == str(root)
    assert Path(observed["filename"]).resolve() == root / "runtime/residual_archive.py"
    direct.update(
        outcome="REACHED_TOKEN_DECODE",
        exception_class=None,
        exception_message="",
        marker=str(marker),
        observation="actual Python call event, then no exception until group timeout",
    )
    extracted = work / "extracted"
    extracted.mkdir(exist_ok=True)
    p.retain(extracted / "p", p.prior.jg2.read_archive_member(root / "archive.zip"))
    p.retain(work / "file_list.txt", b"0.hevc\n")
    shell = bounded(
        ["bash", str(root / "inflate.sh"), str(extracted), str(work / "out"), str(work / "file_list.txt")],
        work,
        env,
        "shell",
    )
    stderr = Path(shell["stderr"]).read_text()
    if shell["timed_out"] or shell["returncode"] == 0 or "requires CUDA inflation" not in stderr:
        raise RuntimeError(f"public shell CUDA gate not proven: {shell}")
    message = next(
        line.removeprefix("RuntimeError: ") for line in stderr.splitlines() if line.startswith("RuntimeError:")
    )
    shell.update(outcome="REACHED_CUDA_GATE", exception_class="RuntimeError", exception_message=message)
    assert tree_before == measure_runtime_digest(root).sha256
    identity = {
        "runtime_path": str(root),
        "archive_path": str(root / "archive.zip"),
        "tree_sha256": tree_before,
        "archive_sha256": sha(root / "archive.zip"),
    }
    direct.update(identity)
    shell.update(identity)
    r = {
        "direct": direct,
        "shell": shell,
        "axis": p.AXIS,
        "score_claim": False,
        "processes_at_a_time": 1,
        "threads": 4,
        "thread_reason": "public receiver mandates four; charter caps processes at two",
        "retention": "all payloads and smoke logs retained; shell deletes only its own trivial compiler scratch",
    }
    p.save_json(receipt, r)
    return r


def combine() -> dict:
    pointer_guard()
    from tac.candidate_seal import _public_smoke_problems

    roles = {r: json.loads((p.STORE / f"SMOKE_{r}.json").read_text()) for r in ("candidate", "frontier")}
    block = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": 90,
        "public_path_probes": {r: v["direct"] for r, v in roles.items()},
        "inflate_sh_smokes": {r: v["shell"] for r, v in roles.items()},
    }
    errors, _ = _public_smoke_problems(
        block,
        candidate_runtime_dir=p.STORE / "candidate_runtime",
        candidate_archive_path=p.STORE / "candidate_runtime/archive.zip",
        pointer_archive_sha256=p.PIN,
    )
    if errors:
        raise RuntimeError(errors)
    p.save_json(p.STORE / "PUBLIC_ENTRYPOINT_SMOKE.json", block)
    return {"public_smoke_problems": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage", "smoke-candidate", "smoke-frontier", "combine"), required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    a = parser.parse_args()
    if a.resume_from.resolve() != (p.STORE / "RACE.json").resolve():
        raise ValueError("resume-from must name the completed race receipt")
    if shutil.disk_usage(p.STORE).free < 256 * 1024**2:
        raise RuntimeError("SSD free-space preflight refuses")
    sys.path.insert(0, str(p.REPO / "src"))
    if a.stage == "stage":
        result = stage()
    elif a.stage == "combine":
        result = combine()
    else:
        result = smoke(a.stage.removeprefix("smoke-"))
    print(json.dumps(result, indent=2))
