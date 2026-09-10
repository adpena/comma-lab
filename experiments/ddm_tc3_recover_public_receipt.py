"""Recover the completed TC3 public comparison after a certified source move.

The literal shell already finished successfully. This does not render, decode,
score, or relabel a failed child as successful. Both retained byte objects and
the original child log, runtime, source receipt and move record are verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments import ddm_tc3_receiver as receiver

ref, io, proof, ROOT = receiver.ref, receiver.io, receiver.proof, receiver.ROOT


def recover():
    receiver.pin("recover_public_cold_store")
    runtime = ROOT / "candidate_runtime"
    work = ROOT / "public_identity"
    attempt = work / "stage_0600/attempt_0000"
    process_path = attempt / "PROCESS.json"
    process = json.loads(process_path.read_text())
    log = attempt / "run.log"
    if process["returncode"] != 0 or process["log"] != io.file_fact(log):
        raise ValueError("recovery requires a successful, byte-bound public child")
    expected_command = [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ]
    if process["argv"] != expected_command:
        raise ValueError("recovery is not the literal candidate public shell")
    launch_binding = json.loads((work / "RUN_BINDING.json").read_text())
    if (
        launch_binding["runtime"] != proof.identity(runtime)
        or launch_binding["receiver"] != receiver.receiver_binding()
    ):
        raise ValueError("completed child runtime or original producer changed")
    for item in launch_binding["build"]:
        for key in ("source", "library"):
            if item[key] != io.file_fact(Path(item[key]["path"])):
                raise ValueError("completed child native build changed")
    reports = [json.loads(x) for x in log.read_text().splitlines() if x.startswith('{"archive_bytes"')]
    if len(reports) != 1:
        raise ValueError("exactly one complete public report required")
    report = reports[0]
    archive = io.file_fact(runtime / "archive.zip")
    if report["pair_count"] != 600 or any(report["archive_" + key] != archive[key] for key in ("sha256", "bytes")):
        raise ValueError("completed child archive or population mismatch")
    original_receipt = ref.LIVE.parents[1] / "parseback/PARSEBACK_RESULT.json"
    source = json.loads(original_receipt.read_text())
    if source["archive"]["sha256"] != ref.ARCHIVE_SHA:
        raise ValueError("original raw receipt names another source archive")
    original_path = Path(source["rendered_raw"]["path"])
    moved_path = original_path.with_name(original_path.name + ".MOVED.json")
    moved = json.loads(moved_path.read_text())
    cold = Path(moved["moved_to"])
    if not cold.resolve().is_relative_to(Path("/Volumes/APDataStore/pact").resolve()):
        raise ValueError("unexpected source cold-store tier")
    original = io.file_fact(cold)
    actual = io.file_fact(work / "output/0.raw")
    for key in ("bytes", "sha256"):
        if not original[key] == moved[key] == source["rendered_raw"][key] == actual[key] == report["raw_" + key]:
            raise ValueError("raw or certified move identity mismatch: " + key)
    field = io.file_fact(work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
    if field["bytes"] != 117964800 or field["sha256"] != ref.FIELD_SHA:
        raise ValueError("full public field differs")
    if report["token_decoder"]["decoded_token_sha256"] != field["sha256"]:
        raise ValueError("token report does not match persisted full field")
    result = {
        "axis": "[macOS-CPU advisory / actual inflate.sh -> inflate.py, n600]",
        "score_claim": False,
        "process": process,
        "report": report,
        "archive": archive,
        "runtime_identity": proof.identity(runtime),
        "decoded_field": field,
        "original_raw": original,
        "original_raw_before_move": source["rendered_raw"],
        "source_parseback_receipt": io.file_fact(original_receipt),
        "source_move_receipt": io.file_fact(moved_path),
        "candidate_raw": actual,
        "all600_field_identity": True,
        "all600_seg_pose_input_byte_identity": True,
        "no_scorer_ran": True,
        "proof": "all output bytes identical; no new distortion measurement",
        "contest_cuda_identity_and_score": "not measured here; MAIN exact eval remains required",
        "recovery": {
            "reason": "source raw cold-stored during successful public child; original wrapper failed at old-path stat",
            "producer": io.file_fact(Path(__file__)),
            "completed_child_receipt": io.file_fact(process_path),
            "original_failed_wrapper": io.file_fact(ref.ROOT / "launch_public600/safe_run.json"),
            "rerendered": False,
        },
    }
    ref.record(ROOT / "PUBLIC_IDENTITY_RECOVERED.json", result)
    return ref.record(ROOT / "PUBLIC_IDENTITY.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ref.ROOT.resolve():
        raise ValueError("wrong recovery root")
    print(json.dumps(recover(), sort_keys=True))
