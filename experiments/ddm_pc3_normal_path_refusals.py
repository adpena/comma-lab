"""ddm_pc3 -- prove BOTH normal seal paths refuse the intent, before any subprocess.

An intent is a promise that nothing downstream will treat it as a completed seal.  The
only way to know that is to hand it to the real consumers and watch them refuse -- which
is what this does, on the unmodified tools, with the literal argv a caller would type.

TWO DOORS
---------
1. ``tac.candidate_seal.validate_seal`` -- the seal reader. An intent is
   ``candidate_prefire_intent.v1``, not ``candidate_seal.v3``, so the reader must return
   verdict ``PREFIRE_INTENT_SCHEMA_REFUSED``.
2. ``tools/fire_modal_auth_eval.py --seal <intent>`` -- the FIRE path, the one that spends
   money. It must refuse with the same typed code and it must do so BEFORE starting any
   consumer subprocess.

The second claim is the load-bearing one, and asserting it from the exit code alone would
be worthless: a tool that spawned a dispatch and then refused would still have spent.  So
``subprocess.run`` / ``Popen`` / ``call`` / ``check_output`` and ``os.system`` are replaced
with recorders that raise on use.  If the tool tries to start ANYTHING, the attempt is
recorded and the control fails -- the proof is "no subprocess was attempted", not "the
tool said no".

Nothing here authorizes, fires, or completes anything. ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from tac.candidate_seal import prefire_digest, validate_seal


class Pc3RefusalError(RuntimeError):
    """A ddm_pc3 normal-path control failed. A door that does not shut is not a door."""


def fact(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as stream:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def write(path: Path, document: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fire-output-dir", type=Path, required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--pair-group-id", required=True)
    parser.add_argument("--instance-job-id", required=True)
    args = parser.parse_args()

    intent_path = Path(args.intent).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    intent = json.loads(intent_path.read_text())

    # DOOR 1 -- the seal reader.
    validation = validate_seal(intent_path)
    door_one = {
        "seal_path": str(intent_path),
        "verdict": validation.verdict,
        "ok": bool(validation.ok),
        "problems": list(validation.problems),
        "candidate_id": validation.candidate_id,
        "axis": validation.axis,
        "observed": dict(validation.observed),
    }
    write(out / "NORMAL_VALIDATE_SEAL_REFUSAL.json", door_one)
    if validation.verdict != "PREFIRE_INTENT_SCHEMA_REFUSED" or validation.ok:
        raise Pc3RefusalError(f"seal reader did not refuse the intent: {door_one}")

    # DOOR 2 -- the fire path, with every subprocess entry point disarmed and recorded.
    fire_tool = REPO / "tools/fire_modal_auth_eval.py"
    argv = [
        sys.executable, str(fire_tool),
        "--seal", str(intent_path),
        "--output-dir", str(Path(args.fire_output_dir)),
        "--lane-id", args.lane_id,
        "--pair-group-id", args.pair_group_id,
        "--instance-job-id", args.instance_job_id,
    ]
    write(out / "NORMAL_FIRE_ARGV.json", argv)

    attempted: list[dict[str, Any]] = []

    def refuse(name):
        def recorder(*call_args, **call_kwargs):
            attempted.append({"api": name, "args": repr(call_args)[:400]})
            raise Pc3RefusalError(
                f"the fire path attempted a consumer subprocess via {name}; "
                "the control is that it refuses BEFORE spending, so this is a failure"
            )
        return recorder

    saved = {
        "subprocess.run": subprocess.run,
        "subprocess.Popen": subprocess.Popen,
        "subprocess.call": subprocess.call,
        "subprocess.check_output": subprocess.check_output,
        "subprocess.check_call": subprocess.check_call,
        "os.system": os.system,
    }
    for name in saved:
        module, attribute = name.split(".")
        setattr(subprocess if module == "subprocess" else os, attribute, refuse(name))

    stdout, stderr = io.StringIO(), io.StringIO()
    exception: str | None = None
    exit_code: int | None = None
    saved_argv = sys.argv[:]
    try:
        sys.argv = argv[1:]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            runpy.run_path(str(fire_tool), run_name="__main__")
    except SystemExit as exit_error:
        exit_code = exit_error.code if isinstance(exit_error.code, int) else 1
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
    finally:
        sys.argv = saved_argv
        for name, original in saved.items():
            module, attribute = name.split(".")
            setattr(subprocess if module == "subprocess" else os, attribute, original)

    (out / "NORMAL_FIRE_STDOUT.txt").write_text(stdout.getvalue())
    (out / "NORMAL_FIRE_STDERR.txt").write_text(stderr.getvalue())
    receipts = [
        fact(p)
        for p in sorted(Path(args.fire_output_dir).rglob("*"))
        if p.is_file()
    ]
    refusal_file = intent_path.with_name(intent_path.name + ".REFUSED.json")
    if refusal_file.is_file():
        receipts.append(fact(refusal_file))
    door_two = {
        "consumer_execution": (
            "unmodified tools/fire_modal_auth_eval.py via runpy __main__ with the literal "
            "--seal argv; subprocess.run/Popen/call/check_output/check_call and os.system "
            "replaced by recorders that raise, so a spend attempt cannot pass unseen"
        ),
        "consumer_source": fact(fire_tool),
        "intent": fact(intent_path),
        "intent_canonical_digest": prefire_digest(intent, "intent_sha256"),
        "argv": argv,
        "exit_code": exit_code,
        "exception": exception,
        "attempted_subprocesses": attempted,
        "stderr_tail": stderr.getvalue().strip().splitlines()[-8:],
        "receipts": receipts,
        "score_claim": False,
        "promotion_eligible": False,
    }
    write(out / "NORMAL_FIRE_REFUSAL.json", door_two)
    if attempted:
        raise Pc3RefusalError(f"fire path attempted a subprocess: {attempted}")

    typed = [
        json.loads(Path(r["path"]).read_text())
        for r in receipts
        if r["path"].endswith(".json")
    ]
    codes = {
        document.get("code")
        for document in typed
        if isinstance(document, dict) and document.get("code")
    }
    door_two["typed_refusal_codes"] = sorted(codes)
    write(out / "NORMAL_FIRE_REFUSAL.json", door_two)
    if "PREFIRE_INTENT_SCHEMA_REFUSED" not in codes:
        raise Pc3RefusalError(
            "the fire path refused, but not with the typed PREFIRE_INTENT_SCHEMA_REFUSED "
            f"code; observed {sorted(codes)}. An untyped refusal is not this control."
        )
    print(json.dumps({
        "door_1_validate_seal": validation.verdict,
        "door_2_fire_codes": sorted(codes),
        "attempted_subprocesses": len(attempted),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
