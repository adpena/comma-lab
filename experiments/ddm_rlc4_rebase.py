#!/usr/bin/env python3
"""Rebase the landed RLC1 counted cure on the retained real move42 trace.

Scorer-free, research-only. Reuses RLC1's actual resumable triple encoder;
both candidate encoders and the unchanged-stream control process all symbols.
All payloads stay in the assigned SSD store. No seal or dispatch is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_rlc1_run as prior
from experiments import ddm_rlc3_move42_trace as trace
from tac.candidate_seal import measure_runtime_digest, runtime_digest_skip_reason
from tac.decode_wall_clock import measure_receiver_digest

ROOT = trace.ROOT
SOURCE = trace.LIVE
REFERENCE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure")
EXPECTED_RECEIVER = "b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d"
BASE_BYTES = 180238
MAX_ARCHIVE_BYTES = 180207
RECEIVER_PATHS = ("inflate.py", "inflate.sh", "runtime/residual_archive.py",
                  "runtime/rlc1_geometry.c", "runtime/rlc1_geometry.py", "runtime/rlc1_mixer.py")


def fact(path):
    return prior.fact(Path(path))


def record(path, value):
    return prior.record(path, value)


def blob(path, payload):
    return trace.blob(path, payload)


def check_base():
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != trace.ARCHIVE_SHA:
        raise ValueError("POINTER_MOVED: no production on a stale base")
    if fact(SOURCE / "archive.zip")["sha256"] != trace.ARCHIVE_SHA:
        raise ValueError("source archive drift")
    if measure_receiver_digest(REFERENCE / "candidate_runtime") != EXPECTED_RECEIVER:
        raise ValueError("timed RLC1 reference drift")


def prepare():
    check_base()
    result_path = ROOT / "move42/trace/RESULT.json"
    result = json.loads(result_path.read_text())
    if not (result["frames"] == 600 and result["full_control_byte_identical"]
            and result["twin_byte_identical"] and result["field_sha256"] == trace.FIELD_SHA):
        raise ValueError("full source trace control required")
    if fact(ROOT / "move42/field.u8")["sha256"] != trace.FIELD_SHA:
        raise ValueError("source field drift")
    residual, _, _ = prior.io.load_runtime(SOURCE)
    parts = residual.read_residual_archive(SOURCE / "archive.zip")
    cfg = (REFERENCE / "retained/config.bin").read_bytes()
    if len(cfg) != 60 or cfg[1:36] != bytes(parts.tc1_weights):
        raise ValueError("counted reference config is not the move42 inherited mixer")
    if hashlib.sha256(cfg).hexdigest() != "76f10171e42d27e6a1966a5a4a0a530c18541415b2a60a5fe9f8c2bb8af3d91a":
        raise ValueError("reference config changed")
    prior.geo.parse_config(cfg[41:])
    blob(ROOT / "retained/config.bin", cfg)
    blob(ROOT / "retained/geometry_config.bin", cfg[41:])
    blob(ROOT / "retained/source.rc64", parts.token_stream)
    native_record = json.loads((REFERENCE / "retained/build_geometry.json").read_text())
    library = REFERENCE / "retained/geometry.dylib"
    if (fact(library)["sha256"] != native_record["library_sha256"]
            or fact(REPO / "experiments/ddm_rlc1_geometry.c")["sha256"] != native_record["source_sha256"]):
        raise ValueError("retained native source/build drift")
    blob(ROOT / "retained/geometry.dylib", library.read_bytes())
    blob(ROOT / "retained/geometry_build_source.json", (REFERENCE / "retained/build_geometry.json").read_bytes())
    binding = {
        "archive": fact(SOURCE / "archive.zip"), "field": fact(ROOT / "move42/field.u8"),
        "config": fact(ROOT / "retained/config.bin"), "source_trace_result": fact(result_path),
        "native": fact(ROOT / "retained/geometry.dylib"), "seed": trace.SEED,
        "sources": [fact(Path(m.__file__)) for m in (prior, prior.io, prior.geo, prior.mix, prior.mix.base, trace)]
                   + [fact(Path(__file__))],
        "upstream": fact(REPO / "upstream/evaluate.py"),
        "score_claim": False, "axis": "[exact bytes; macOS-CPU scorer-free]",
    }
    path = ROOT / "RLC4_INPUTS.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["binding"] != binding:
            raise ValueError("rebase input/source drift; retain checkpoints")
        return old
    return record(path, {"binding": binding,
        "production_started_at_utc": datetime.now(UTC).isoformat(),
        "producer_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_custody": "landed RLC1 mechanism and RLC3 trace; charter-authorized unlanded rebase wrapper bundled at final harvest"})


def encode(stop):
    # Explicit parameter binding of the unchanged landed encoder; no source rewrite.
    prior.ROOT, prior.SOURCE, prior.TRACE = ROOT, SOURCE, ROOT / "move42"
    prior.ARCHIVE_SHA, prior.FIELD_SHA = trace.ARCHIVE_SHA, trace.FIELD_SHA
    prior.prepare = prepare
    return prior.encode(stop)


def stage():
    pin = prepare()
    result = json.loads((ROOT / "encode/RESULT.json").read_text())
    if result["binding"] != pin or not (result["twins_identical"] and result["source_stream_identical"]):
        raise ValueError("full bound candidate twins required")
    member = prior.io.read_archive_member(SOURCE / "archive.zip")
    section = prior.io.split_member(member)
    prefix = b"".join(section[k] for k in ("header", "hpac", "semantic", "carrier")) + section["tail"][:96]
    prior.ROOT = ROOT
    prior.pack(member, ROOT / "retained/source_control.zip", "stored", None)
    if fact(ROOT / "retained/source_control.zip")["sha256"] != trace.ARCHIVE_SHA:
        raise ValueError("full source archive reconstruction differs")
    for i in range(2):
        rider = (ROOT / f"encode/twin{i}.rider").read_bytes()
        config, stream = prior.mix.unpack_rider(rider)
        if config != (ROOT / "retained/config.bin").read_bytes() or stream != (ROOT / f"encode/twin{i}.rc64").read_bytes():
            raise ValueError("rider parse-back mismatch")
        blob(ROOT / f"retained/twin{i}.member", prefix + rider)
        prior.pack(prefix + rider, ROOT / f"retained/archive.twin{i}.zip", "stored", None)
    if (ROOT / "retained/archive.twin0.zip").read_bytes() != (ROOT / "retained/archive.twin1.zip").read_bytes():
        raise ValueError("full archive twins differ")
    archive = fact(ROOT / "retained/archive.twin0.zip")
    if archive["bytes"] > MAX_ARCHIVE_BYTES:
        return record(ROOT / "BYTE_GATE_STOP.json", {"archive": archive, "maximum_bytes": MAX_ARCHIVE_BYTES,
            "verdict": "STOP", "verdict_scope": "INSTANCE", "score_claim": False})
    candidate = ROOT / "candidate_runtime"
    # Copy move42 first, then exactly RLC1's six receiver paths. All payload pins are measured.
    for path in sorted(SOURCE.rglob("*")):
        if (path.is_file() and path.name not in ("archive.zip", "MANIFEST.sha256")
                and not runtime_digest_skip_reason(path.relative_to(SOURCE).as_posix())
                and path.relative_to(SOURCE).as_posix() not in RECEIVER_PATHS):
            blob(candidate / path.relative_to(SOURCE), path.read_bytes())
    for relative in RECEIVER_PATHS:
        payload = (REFERENCE / "candidate_runtime" / relative).read_bytes()
        if relative == "inflate.py":
            text = payload.decode()
            text = prior.replace_once(text, 'ARCHIVE_SHA256 = "8c2eaefa944ca8acba3db80829a5cd7a3d06a17b1fdb7ed5f39ce5a6fd9a124d"',
                                      f'ARCHIVE_SHA256 = "{archive["sha256"]}"')
            text = prior.replace_once(text, "ARCHIVE_BYTES = 180173", f'ARCHIVE_BYTES = {archive["bytes"]}')
            payload = text.encode()
        blob(candidate / relative, payload)
    blob(candidate / "archive.zip", (ROOT / "retained/archive.twin0.zip").read_bytes())
    rows = [(p.relative_to(candidate).as_posix(), fact(p)) for p in sorted(candidate.rglob("*"))
            if p.is_file() and p.name not in ("archive.zip", "MANIFEST.sha256")
            and not runtime_digest_skip_reason(p.relative_to(candidate).as_posix())]
    if len(rows) != 49:
        raise ValueError(f"dependency census differs: {len(rows)}")
    manifest = "".join(f'{f["sha256"]}  {rel}\n' for rel, f in sorted(rows)).encode()
    blob(ROOT / "retained/MANIFEST.sha256.generated_outside", manifest)
    blob(candidate / "MANIFEST.sha256", manifest)
    for line in manifest.decode().splitlines():
        sha, rel = line.split("  ", 1)
        if fact(candidate / rel)["sha256"] != sha:
            raise ValueError("independent manifest hash failed")
    changed = sorted(p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_file()
        and (not (SOURCE / p.relative_to(candidate)).exists() or p.read_bytes() != (SOURCE / p.relative_to(candidate)).read_bytes()))
    expected = sorted(["archive.zip", "MANIFEST.sha256", *RECEIVER_PATHS])
    if changed != expected:
        raise ValueError(f"unexpected delta: {changed}")
    runtime = measure_runtime_digest(candidate)
    receiver = measure_receiver_digest(candidate)
    return record(ROOT / "RLC4_CANDIDATE.json", {
        "archive": fact(candidate / "archive.zip"), "runtime": runtime.to_dict(), "receiver_sha256": receiver,
        "expected_receiver_sha256": EXPECTED_RECEIVER, "receiver_gate_pass": receiver == EXPECTED_RECEIVER,
        "changed_paths": changed, "manifest_rows": 49, "all_manifest_hashes_passed": True,
        "source_archive": fact(SOURCE / "archive.zip"), "source_runtime": measure_runtime_digest(SOURCE).to_dict(),
        "binding": pin, "net_saved_bytes": BASE_BYTES - archive["bytes"], "score_claim": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "encode", "stage"))
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.stop_after <= 600:
        raise ValueError("wrong resume root or stage boundary")
    print(json.dumps({"prepare": prepare, "encode": lambda: encode(args.stop_after), "stage": stage}[args.stage]()))


if __name__ == "__main__":
    main()
