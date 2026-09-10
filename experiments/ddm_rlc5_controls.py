#!/usr/bin/env python3
"""Run the real amended intent producer and refusal-only normal CLI controls.

No first-measurement authorization or fire is exposed. The normal --seal
negative control installs a Python audit hook that records and blocks any
process launch; success requires no attempted launch, not merely a blocked one.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from tac import candidate_seal as seal

OUT = REPO / ".omx/research/ddm_rlc5_20260910"
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43")
INTENT = OUT / "CANDIDATE_PREFIRE_INTENT.json"


def write(name, value):
    path = OUT / name
    if path.exists():
        raise ValueError(f"refuse to overwrite evidence: {path}")
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return seal.prefire_file_reference(path)


def deduplicate_raw():
    """Certify full public identity before replacing only this arm's raw by a hardlink."""
    proof_path = OUT / "RAW_IDENTITY_N600.json"
    proof = json.loads(proof_path.read_text())
    candidate, source = (Path(proof[k]["path"]) for k in ("candidate_raw", "pointer_raw"))
    if not candidate.resolve().is_relative_to(ROOT.resolve()) or source == candidate:
        raise ValueError("raw deduplication custody paths differ")
    if (proof["n_samples"] != 600 or proof["checkpoint_resume"] is not False
            or proof["token_cache_status"] != "DISABLED"):
        raise ValueError("cold public identity proof required before deduplication")
    actual = [seal.prefire_file_reference(p) for p in (candidate, source)]
    if actual != [proof["candidate_raw"], proof["pointer_raw"]]:
        raise ValueError("raw proof custody drift; keep both payloads")
    if (actual[0]["bytes"], actual[0]["sha256"]) != (actual[1]["bytes"], actual[1]["sha256"]):
        raise ValueError("raw bytes differ; keep both payloads")
    stats = [p.stat() for p in (candidate, source)]
    if stats[0].st_dev != stats[1].st_dev:
        raise ValueError("cross-device deduplication refused; keep both payloads")
    certificate = {"schema": "ddm_rlc5_lossless_raw_hardlink.v1", "original": actual[0],
        "cold_store_source": actual[1], "raw_identity_proof": seal.prefire_file_reference(proof_path),
        "candidate_archive": seal.prefire_file_reference(ROOT / "candidate_runtime/archive.zip"),
        "public_result": seal.prefire_file_reference(ROOT / "public_rlc4/RESULT.json"),
        "command": proof["command"], "original_inodes": [s.st_ino for s in stats],
        "reason": "Full cold public execution compared every byte before certification; replacing the duplicate owned raw retains both paths and identical bytes on one inode.",
        "score_claim": False}
    write("RAW_HARDLINK_CERTIFICATE.json", certificate)
    temporary = candidate.with_name("0.raw.rlc5_hardlink")
    os.link(source, temporary)
    os.replace(temporary, candidate)
    after = [p.stat() for p in (candidate, source)]
    if (after[0].st_dev, after[0].st_ino) != (after[1].st_dev, after[1].st_ino):
        raise ValueError("hardlink verification failed")
    write("RAW_HARDLINK_RESULT.json", {"certificate": seal.prefire_file_reference(OUT / "RAW_HARDLINK_CERTIFICATE.json"),
        "same_inode": True, "bytes_per_path": after[0].st_size, "retained_paths": [str(candidate), str(source)],
        "score_claim": False})


def emit():
    """Parameterize the landed RLC4 real CLI and retain its exact exit/output."""
    deduplicate_raw()
    old = json.loads((REPO / ".omx/research/ddm_rlc4_20260910/INTENT_PRODUCER_ARGV.json").read_text())
    argv = []
    i = 0
    while i < len(old):
        if old[i] == "--falsifier":
            i += 2
            continue
        value = old[i].replace("/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42", str(ROOT))
        value = value.replace(".omx/research/ddm_rlc4_20260910", ".omx/research/ddm_rlc5_20260910")
        value = value.replace("ddm_rlc2_counted_cure_move42_rlc4", "ddm_rlc2_counted_cure_move43_rlc5")
        argv.append(value)
        i += 1
    # Retain the RLC2 family identity: the frozen validator explicitly binds this
    # historical diagnostic chain by hash; this is the chartered successor chain.
    prereg = json.loads((OUT / "FALSIFIERS_PREREGISTERED.json").read_text())
    for falsifier in prereg["falsifiers"]:
        argv.extend(["--falsifier", falsifier])
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    write("INTENT_PRODUCER_ARGV.json", argv)
    write("INTENT_PRODUCER_ENV.json", {k: env[k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")})
    with (OUT / "INTENT_PRODUCER_STDOUT.txt").open("xb") as stdout, (OUT / "INTENT_PRODUCER_STDERR.txt").open("xb") as stderr:
        result = subprocess.run(argv, cwd=REPO, env=env, stdout=stdout, stderr=stderr, check=False)
    write("INTENT_PRODUCER_STATUS.json", {"returncode": result.returncode,
        "intent_exists": INTENT.exists(), "score_claim": False})
    if result.returncode:
        raise RuntimeError(f"REAL_PRODUCER_REFUSED rc={result.returncode}; STOP, never patch around the contract")
    document = seal.validate_prefire_intent(INTENT, require_committed=False, repo=REPO)
    reference = seal.prefire_file_reference(INTENT)
    digest = seal.prefire_digest(document, "intent_sha256")
    if digest != document["intent_sha256"] or reference != seal.prefire_file_reference(INTENT):
        raise ValueError("intent changed on re-read")
    write("INTENT_IDENTITY.json", {"path": reference["path"], "file_sha256": reference["sha256"],
        "bytes": reference["bytes"], "canonical_digest": digest, "self_validation": "PASS",
        "require_committed": False, "score_claim": False})


def controls():
    """Exercise normal validators on the real positive intent, without dispatch."""
    document = seal.validate_prefire_intent(INTENT, require_committed=False, repo=REPO)
    identity = seal.prefire_file_reference(INTENT)
    verdict = seal.validate_seal(INTENT, require_decode_wall_clock=True)
    write("NORMAL_VALIDATE_SEAL_REFUSAL.json", verdict.to_dict())
    if verdict.verdict != "PREFIRE_INTENT_SCHEMA_REFUSED":
        raise ValueError(f"wrong normal seal refusal: {verdict.verdict}")
    argv = [str(REPO / "tools/fire_modal_auth_eval.py"), "--seal", str(INTENT),
            "--output-dir", str(ROOT / "normal_seal_refusal"),
            "--lane-id", "ddm_rlc5_counted_rider_move43_20260910",
            "--pair-group-id", "ddm_rlc5_normal_seal_refusal_control",
            "--instance-job-id", "ddm_rlc5_normal_seal_refusal_control"]
    write("NORMAL_FIRE_ARGV.json", [sys.executable, *argv])
    attempted = []

    def audit(event, args):
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork", "os.forkpty"}:
            attempted.append({"event": event, "args": repr(args)})
            raise RuntimeError("NORMAL_SEAL_CONTROL_SUBPROCESS_REACHED")

    sys.addaudithook(audit)
    previous = sys.argv
    rc, error = None, None
    try:
        sys.argv = argv
        with (OUT / "NORMAL_FIRE_STDOUT.txt").open("x") as stdout, (OUT / "NORMAL_FIRE_STDERR.txt").open("x") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as exc:
                rc = exc.code
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv = previous
    retained = list((ROOT / "normal_seal_refusal").glob("*.json"))
    sibling = INTENT.with_name(INTENT.name + ".REFUSED.json")
    if sibling.exists():
        retained.append(sibling)
    receipts = [seal.prefire_file_reference(p) for p in sorted(retained)]
    matching = [p for p in retained if json.loads(p.read_text()).get("code") == "PREFIRE_INTENT_SCHEMA_REFUSED"]
    result = {"returncode": rc, "exception": error, "attempted_subprocesses": attempted,
        "refusal_code": "PREFIRE_INTENT_SCHEMA_REFUSED" if matching else None,
        "consumer_execution": "unmodified tools/fire_modal_auth_eval.py via runpy __main__ with literal --seal argv",
        "receipts": receipts, "intent": identity, "intent_canonical_digest": document["intent_sha256"],
        "consumer_source": seal.prefire_file_reference(Path(argv[0])), "score_claim": False}
    write("NORMAL_FIRE_REFUSAL.json", result)
    if rc != 7 or error is not None or attempted or not matching:
        raise ValueError("normal fire control failed; retain real refusal and stop")
    if identity != seal.prefire_file_reference(INTENT):
        raise ValueError("intent changed during refusal controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("emit", "controls"))
    args = parser.parse_args()
    {"emit": emit, "controls": controls}[args.stage]()


if __name__ == "__main__":
    main()
