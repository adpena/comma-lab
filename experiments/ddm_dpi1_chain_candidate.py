"""ddm_dpi1 -- fire the candidate price as soon as its two prerequisites land.

WHY THIS EXISTS.  The candidate row needs BOTH the ``control48`` falsifier (the rail must
reproduce move 48's archive byte-identically before any move-48 price is admissible) and
the depth-restored refit's terminal checkpoint.  The two run concurrently and finish tens
of minutes apart.  This chainer waits on their ARTIFACTS -- never on a clock, never on the
process table -- re-checks the control's verdict itself, and then execs the rail.  It adds
no science: the rail's own gates still decide, and this refuses loudly rather than
proceeding on a control it has not read.

Axis ``[macOS-CPU advisory; exact bytes, scorer-free]``; ``score_claim=false``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_dpi1")
CONTROL_PRICE = STORE / "price/control48/PRICE.json"
CHECKPOINT = STORE / "train/depths/qat.checkpoints/qat_stage_end_epoch_0060.pt"
TRAIN_RESULT = STORE / "train/depths/result.json"
CANDIDATE = "retrain_depths_frame_even"
#: A generous ceiling on the two prerequisites, well past their measured rates
#: (control48 ~3.1 s/frame over 600 frames; the refit ~60 s/epoch over 60 epochs).
DEADLINE_SECONDS = 4 * 3600
POLL_SECONDS = 30


class ChainError(RuntimeError):
    """A prerequisite is absent, or the control did not do what it must."""


def live_pointer_sha() -> str:
    document = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    return document["our_local_frontier_contest_cuda"]["archive_sha256"]


def wait_for(paths: list[Path]) -> None:
    """Block until every artifact exists, bounded by a deadline that is not the signal."""
    started = time.time()
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if time.time() - started > DEADLINE_SECONDS:
            raise ChainError(f"prerequisites still absent after {DEADLINE_SECONDS}s: {missing}")
        time.sleep(POLL_SECONDS)


def verify_control() -> dict:
    """Read the control's own verdict; refuse unless it reproduced the live pointer."""
    price = json.loads(CONTROL_PRICE.read_text())
    twins = price["twins"]
    pointer = live_pointer_sha()
    if twins[0]["sha256"] != twins[1]["sha256"]:
        raise ChainError("control48 twins differ")
    if twins[0]["sha256"] != pointer:
        raise ChainError(f"control48 produced {twins[0]['sha256']}, not the live pointer {pointer}")
    if price["delta_bytes_vs_base"] != 0:
        raise ChainError(f"control48 delta is {price['delta_bytes_vs_base']}, not 0")
    if not price["output_lossless"]:
        raise ChainError("control48 decode is not output-lossless")
    return price


def main() -> int:
    wait_for([CONTROL_PRICE, TRAIN_RESULT, CHECKPOINT])
    control = verify_control()
    print(
        json.dumps(
            {
                "control48_archive_bytes": control["twins"][0]["bytes"],
                "control48_sha256": control["twins"][0]["sha256"],
                "control48_delta": control["delta_bytes_vs_base"],
                "verdict": "CONTROL48_PASSED; firing the candidate",
            }
        ),
        flush=True,
    )
    argv = [
        sys.executable,
        str(REPO / "experiments/ddm_hpr1_shape_price.py"),
        "--treatment",
        CANDIDATE,
        "--store-root",
        "dpi1",
        "--checkpoint",
        str(CHECKPOINT),
        "--resume-from",
        str(STORE / "price" / CANDIDATE),
    ]
    print(json.dumps({"argv": argv}), flush=True)
    return subprocess.run(argv, cwd=str(REPO), env=os.environ.copy(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
