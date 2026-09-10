"""Resume one real public receiver frame from a retained checkpoint and stop.

The preserved native build cache closes the Mach-O temporary-path hash hazard.
No renderer/scorer is reached and the source public proof remains untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments.ddm_rlc1_public import ROOT, fact, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("wrong resume control root")
    work = ROOT / "public_resume_control"
    runtime = ROOT / "candidate_runtime"
    for name in ("data", "output", "scratch", "frame_checkpoints"):
        (work / name).mkdir(parents=True, exist_ok=True)
    source = ROOT / "public_threads1_g2/frame_checkpoints/stage_0025.json"
    before = json.loads(source.read_text())
    payload = fact(Path(before["path"]))
    if any(payload[k] != before[k] for k in ("bytes", "sha256")):
        raise ValueError("source checkpoint drift")
    latest = work / "frame_checkpoints/LATEST.json"
    done = work / "RESULT.json"
    if done.exists():
        receipt = json.loads(done.read_text())
        if fact(Path(receipt["resumed_state"]["path"])) != receipt["resumed_state"]:
            raise ValueError("resume-control retained state drift")
        print(json.dumps(receipt))
        return
    if latest.exists() and json.loads(latest.read_text()) != before:
        raise ValueError("partial control exists: recover receipt without overwriting it")
    record(latest, before)
    with zipfile.ZipFile(runtime / "archive.zip") as z:
        (work / "data/p").write_bytes(z.read("p"))
    (work / "file_list.txt").write_text("0.hevc\n")
    env = dict(
        os.environ,
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "scratch"),
        RLC1_ADVISORY_CPU="1",
        RLC1_PROOF_BLAS_THREADS="1",
        RLC1_NATIVE_CACHE=str(ROOT / "public_threads1_g2/retained_native"),
        CC=str(REPO / "experiments/ddm_rlc1_cached_cc.py"),
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER="26",
        F26_TOKEN_DECODER="python",
    )
    for key in (
        "CPR1_RC64_LIBRARY",
        "F26_CORRECTOR_NATIVE_LIBRARY",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "F26_ADVISORY_RENDER_WORKERS",
    ):
        env.pop(key, None)
    command = [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ]
    record(
        work / "COMMAND.json",
        {
            "argv": command,
            "source_checkpoint": fact(source),
            "native_cache": fact(ROOT / "public_threads1_g2/retained_native/MANIFEST.json"),
            "env": {
                k: env[k] for k in ("CC", "RLC1_NATIVE_CACHE", "TC1_RECEIVER_STOP_AFTER", "RLC1_PROOF_BLAS_THREADS")
            },
        },
    )
    with (work / "run.log").open("wb") as log:
        child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if child.returncode == 0 or "TC1_RECEIVER_STAGE_COMPLETE" not in (work / "run.log").read_text():
        raise ValueError("resume did not stop at requested stage")
    after = json.loads(latest.read_text())
    if after["frame"] != 26 or after["binding"] != before["binding"]:
        raise ValueError("resume binding or stop changed")
    state = fact(Path(after["path"]))
    if any(state[k] != after[k] for k in ("bytes", "sha256")):
        raise ValueError("resumed state digest differs")
    source_field = Path("/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/field.u8")
    field = np.memmap(source_field, mode="r", dtype=np.uint8, shape=(600, 384, 512))
    with np.load(after["path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["tokens"], field[:26])
    result = {
        "source_checkpoint": payload,
        "resumed_state": state,
        "from_frame": 25,
        "to_frame": 26,
        "restored_and_decoded_next_frame": True,
        "all_retained_tokens_equal_source": True,
        "native_libraries_match_original": after["binding"]["libraries"] == before["binding"]["libraries"],
        "axis": "macOS-CPU advisory",
        "score_claim": False,
        "public_entrypoint": str(runtime / "inflate.sh"),
        "returncode": child.returncode,
        "expected_exception": "TC1_RECEIVER_STAGE_COMPLETE",
    }
    record(done, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
