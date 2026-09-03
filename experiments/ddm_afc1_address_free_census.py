#!/usr/bin/env python3
"""Price the address-free ``tile48 x groupbin8`` interaction on exact lb1.

The candidate adds one zero-payload mixer family to a staged copy of the pinned
lb1 Python receiver.  The existing resumable ``ddm_jg2_tail_reencode`` runner
performs the full-n600 physical control and candidate encodes; this module owns
only custody, the staged-copy patch, and byte-exact repeat adjudication.

No scorer is used.  The decoded token field is unchanged by construction, and
the jg2 control/receipt is the authority that the physical encoder follows the
same causal trajectory as the staged receiver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_fcd1_field_for_coder_diagonal import stage_runtime
from tac.candidate_seal import CONSISTENT, check_pin_consistency

AP_ROOT = Path("/Volumes/APDataStore/pact")
STORE = AP_ROOT / "ddm_afc1_address_free_census" / "tile48_groupbin8"
MEASUREMENT = STORE / "measurement_v1"
RUN_STORE = STORE / "physical_v1"
CONTROL_RUNTIME = (
    REPO / ".omx" / "tmp" / "codex_runs" / "ddm_afc1_tile48_groupbin8_control_runtime"
)
RUNTIME = REPO / ".omx" / "tmp" / "codex_runs" / "ddm_afc1_tile48_groupbin8_runtime"
RETAINED_RUNTIME_SURFACE = STORE / "retained" / "runtime_surface"
LB1_ROOT = AP_ROOT / "ddm_lb1_banked_lossless_joint_collect"
LB1_RUNTIME = LB1_ROOT / "runtime_candidate_native"
LB1_ARCHIVE = LB1_ROOT / "retained" / "candidate_lb1_joint22_patch192.zip"
LB1_RECEIPT = LB1_ROOT / "retained" / "S1_encode_lb1_joint22_patch192.json"
TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/"
    "retained/input/dx2_tokens_decoded.u8"
)

LB1_ARCHIVE_BYTES = 180_083
LB1_ARCHIVE_SHA256 = "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9"
LB1_STREAM_BYTES = 113_492
LB1_STREAM_SHA256 = "8838e44f6498cd9b94f480ae04d9ea12d89b7020ff3c6f215ff83de177a3eac2"
TOKENS_BYTES = 117_964_800
TOKENS_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
MINIMUM_FREE_BYTES = 6 << 30
ADMISSION_BAR_BYTES = 30
EXPECTED_IDENTITY_RAW_BYTES = 3_662_409_600
EXPECTED_IDENTITY_TOKEN_BYTES = 117_964_800
IDENTITY_RESERVE_BYTES = 4 << 30
IDENTITY_SCRATCH_BYTES = 64 << 20
IDENTITY_REQUIRED_FREE_BYTES = (
    EXPECTED_IDENTITY_RAW_BYTES
    + EXPECTED_IDENTITY_TOKEN_BYTES
    + IDENTITY_RESERVE_BYTES
    + IDENTITY_SCRATCH_BYTES
)
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"

CONTROL_RECEIPT = RUN_STORE / "retained" / "S1_control_600.json"
CANDIDATE_RECEIPT = RUN_STORE / "retained" / "S1_encode_afc1_tile48_groupbin8.json"
REPEAT_RECEIPT = RUN_STORE / "retained" / "S1_encode_afc1_tile48_groupbin8_repeat.json"


class Afc1Error(RuntimeError):
    """A custody, staged-runtime, control, or repeat gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise Afc1Error(f"missing pinned input: {path}")
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise Afc1Error(
            f"pin mismatch for {path}: {observed['bytes']}/{observed['sha256']} "
            f"!= {expected_bytes}/{expected_sha256}"
        )
    return observed


def replace_once(text: str, before: str, after: str, *, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise Afc1Error(f"runtime patch anchor {label!r} occurs {count} times")
    return text.replace(before, after, 1)


def stage_preflight() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MINIMUM_FREE_BYTES:
        raise Afc1Error(
            f"storage preflight failed: {free} B free < {MINIMUM_FREE_BYTES} B"
        )
    archive = require_file(LB1_ARCHIVE, LB1_ARCHIVE_BYTES, LB1_ARCHIVE_SHA256)
    tokens = require_file(TOKENS, TOKENS_BYTES, TOKENS_SHA256)
    receipt = json.loads(LB1_RECEIPT.read_text())
    if (
        receipt.get("archive_bytes_candidate") != LB1_ARCHIVE_BYTES
        or receipt.get("token_stream_bytes_candidate") != LB1_STREAM_BYTES
        or receipt.get("tokens_changed") != 0
        or receipt.get("stream", {}).get("sha256") != LB1_STREAM_SHA256
        or not receipt.get("delta_trustworthy")
    ):
        raise Afc1Error("lb1 receipt no longer proves the pinned lossless source row")
    runtime_verdict = check_pin_consistency(LB1_RUNTIME)
    if runtime_verdict.verdict != CONSISTENT:
        raise Afc1Error(f"lb1 runtime is pin-inconsistent: {runtime_verdict.summary()}")
    payload = {
        "schema": "ddm_afc1_tile48_groupbin8_preflight.v1",
        "axis": "[custody/storage preflight / no score claim]",
        "score_claim": False,
        "storage": {
            "path": str(STORE),
            "free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
        },
        "inputs": {
            "lb1_archive": archive,
            "lb1_receipt": file_fact(LB1_RECEIPT),
            "tokens": tokens,
            "lb1_runtime": str(LB1_RUNTIME),
            "pin_consistency": runtime_verdict.verdict,
        },
        "candidate": {
            "name": "tile48_groupbin8",
            "cells": 48 * 8,
            "receiver_expression": (
                "tile48=((y//64)*8+(x//64)); "
                "groupbin8=(((x%64)+2*(y%64))*8)//190; "
                "index=tile48*8+groupbin8"
            ),
            "stored_bytes": 0,
        },
    }
    atomic_json(MEASUREMENT / "PREFLIGHT.json", payload)
    return payload


FEATURE_ANCHOR = '            "patch192": (\n'
FEATURE_PATCH = (
    '            "tile48": (\n'
    "                ((flat // WIDTH) // 64) * (WIDTH // 64)\n"
    "                + ((flat % WIDTH) // 64)\n"
    "            ),\n"
)
SPEC_ANCHOR = "    def groupbin8_only(f):\n"
SPEC_PATCH = (
    "    def tile48_groupbin8(f):\n"
    '        return f["tile48"] * GROUP_BINS + f["groupbin8"]\n\n'
    '    specs.update({"tile48_groupbin8": (48 * GROUP_BINS, tile48_groupbin8)})\n\n'
)
MEMBER_ANCHOR = '        "patch192_only",\n'


def patch_runtime(runtime: Path) -> dict[str, Any]:
    target = runtime / "runtime" / "fx2_model_axis_corrector.py"
    before = file_fact(target)
    text = target.read_text()
    if '"tile48_groupbin8"' not in text:
        for anchor, replacement, label in (
            (FEATURE_ANCHOR, FEATURE_PATCH + FEATURE_ANCHOR, "tile48 feature"),
            (SPEC_ANCHOR, SPEC_PATCH + SPEC_ANCHOR, "interaction family"),
            (
                MEMBER_ANCHOR,
                MEMBER_ANCHOR + '        "tile48_groupbin8",\n',
                "ordered configuration member",
            ),
        ):
            text = replace_once(text, anchor, replacement, label=label)
        temporary = target.with_suffix(".py.partial")
        temporary.write_text(text)
        os.replace(temporary, target)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                f"sys.path.insert(0,{str(runtime)!r}); "
                "from runtime.fx2_model_axis_corrector import fx2_family_specs; "
                "from runtime.free_corrector import SHIPPED_CONFIG; "
                "s=fx2_family_specs(); "
                "print(json.dumps({'families':list(SHIPPED_CONFIG['families']), "
                "'cells':int(s['tile48_groupbin8'][0])}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise Afc1Error(f"patched runtime import failed:\n{probe.stderr}")
    wiring = json.loads(probe.stdout.strip().splitlines()[-1])
    if (
        len(wiring["families"]) != 23
        or wiring["families"][-1] != "tile48_groupbin8"
        or wiring["cells"] != 384
    ):
        raise Afc1Error(f"interaction wiring mismatch: {wiring}")
    return {"before": before, "after": file_fact(target), "wiring": wiring}


def jg2_command(stage: str, tag: str | None = None) -> list[str]:
    runtime = CONTROL_RUNTIME if stage == "control" else RUNTIME
    command = [
        ".venv/bin/python",
        "experiments/ddm_jg2_tail_reencode.py",
        "--stage",
        stage,
        "--store",
        str(RUN_STORE),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(LB1_ARCHIVE),
        "--expect-pointer-sha256",
        LB1_ARCHIVE_SHA256,
        "--tokens",
        str(TOKENS),
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
        "--resume",
    ]
    if tag is not None:
        command.extend(("--tag", tag))
    return command


def stage_prepare() -> dict[str, Any]:
    preflight_path = MEASUREMENT / "PREFLIGHT.json"
    if not preflight_path.is_file():
        raise Afc1Error("run preflight before preparing the staged runtime")
    control_staged = stage_runtime(LB1_RUNTIME, LB1_ARCHIVE, CONTROL_RUNTIME)
    control_verdict = check_pin_consistency(CONTROL_RUNTIME)
    if control_verdict.verdict != CONSISTENT:
        raise Afc1Error(
            f"prepared control runtime is pin-inconsistent: {control_verdict.summary()}"
        )
    staged = stage_runtime(LB1_RUNTIME, LB1_ARCHIVE, RUNTIME)
    patched = patch_runtime(RUNTIME)
    retained_corrector = RETAINED_RUNTIME_SURFACE / "fx2_model_axis_corrector.py"
    retained_corrector.parent.mkdir(parents=True, exist_ok=True)
    retained_partial = retained_corrector.with_suffix(".py.partial")
    shutil.copy2(
        RUNTIME / "runtime" / "fx2_model_axis_corrector.py", retained_partial
    )
    os.replace(retained_partial, retained_corrector)
    verdict = check_pin_consistency(RUNTIME)
    if verdict.verdict != CONSISTENT:
        raise Afc1Error(f"prepared runtime is pin-inconsistent: {verdict.summary()}")
    payload = {
        "schema": "ddm_afc1_tile48_groupbin8_prepare.v1",
        "axis": "[build artifact / no score claim]",
        "score_claim": False,
        "control_runtime": control_staged,
        "control_pin_consistency": control_verdict.verdict,
        "runtime": staged,
        "patch": patched,
        "retained_patched_corrector": file_fact(retained_corrector),
        "pin_consistency": verdict.verdict,
        "commands": {
            "control": jg2_command("control"),
            "candidate": jg2_command("encode", "afc1_tile48_groupbin8"),
            "repeat": jg2_command("encode", "afc1_tile48_groupbin8_repeat"),
        },
        "retention": (
            "jg2 atomically checkpoints all RC64/corrector state every 25 frames and "
            "retains the control, both streams, both archives, per-frame ledgers, "
            "receipts, and RC64 sources/builds under this store"
        ),
    }
    atomic_json(MEASUREMENT / "RUNTIME_PREPARE.json", payload)
    return payload


def load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Afc1Error(f"missing required full-n600 receipt: {path}")
    payload = json.loads(path.read_text())
    if payload.get("frames") != 600:
        raise Afc1Error(f"receipt is not full n600: {path}")
    return payload


def stage_adjudicate() -> dict[str, Any]:
    control = load_receipt(CONTROL_RECEIPT)
    candidate = load_receipt(CANDIDATE_RECEIPT)
    repeat = load_receipt(REPEAT_RECEIPT)
    if (
        control.get("emitted_bytes") != LB1_STREAM_BYTES
        or control.get("emitted_sha256") != LB1_STREAM_SHA256
        or not control.get("byte_identical")
    ):
        raise Afc1Error("fresh control did not reproduce the lb1 token stream byte-identically")
    for name, receipt in (("candidate", candidate), ("repeat", repeat)):
        if receipt.get("tokens_changed") != 0 or not receipt.get("delta_trustworthy"):
            raise Afc1Error(f"{name} is not a lossless trustworthy physical row")
    candidate_stream = candidate["stream"]
    repeat_stream = repeat["stream"]
    candidate_archive = candidate["candidate_archive"]
    repeat_archive = repeat["candidate_archive"]
    deterministic = (
        candidate_stream["bytes"] == repeat_stream["bytes"]
        and candidate_stream["sha256"] == repeat_stream["sha256"]
        and candidate_archive["bytes"] == repeat_archive["bytes"]
        and candidate_archive["sha256"] == repeat_archive["sha256"]
    )
    if not deterministic:
        raise Afc1Error("candidate and repeat payloads are not byte-identical")
    saving = LB1_ARCHIVE_BYTES - int(candidate["archive_bytes_candidate"])
    if saving >= ADMISSION_BAR_BYTES:
        disposition = "ADMIT_NATIVE_PORT_AND_RECEIVER_IDENTITY_OWED"
    elif saving > 0:
        disposition = "BANKED_BELOW_30B_ADMISSION_BAR"
    else:
        disposition = "CLOSED_NONPOSITIVE_EXACT_RATE"
    invalid_stream = STORE / "work" / "tail_control_600.bin"
    invalid_checkpoint = STORE / "work" / "encode_control_600.checkpoint.npz"
    invalid_attempt = None
    if invalid_stream.is_file() and invalid_checkpoint.is_file():
        invalid_attempt = {
            "status": "EXCLUDED_INSTRUMENTATION_ERROR",
            "reason": (
                "the first wrapper version pointed the null-control stage at the patched "
                "23-family candidate runtime; its 113411 B output was therefore a candidate "
                "trace, not an unmodified lb1 control"
            ),
            "stream": file_fact(invalid_stream),
            "checkpoint": file_fact(invalid_checkpoint),
            "accepted_as_measurement": False,
            "payload_retained": True,
        }
    payload = {
        "schema": "ddm_afc1_tile48_groupbin8_adjudication.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "base": {
            "archive": file_fact(LB1_ARCHIVE),
            "token_stream_bytes": LB1_STREAM_BYTES,
            "token_stream_sha256": LB1_STREAM_SHA256,
        },
        "control": file_fact(CONTROL_RECEIPT),
        "candidate_receipt": file_fact(CANDIDATE_RECEIPT),
        "repeat_receipt": file_fact(REPEAT_RECEIPT),
        "candidate_stream": candidate_stream,
        "candidate_archive": candidate_archive,
        "repeat_stream": repeat_stream,
        "repeat_archive": repeat_archive,
        "archive_saving_bytes": saving,
        "delta_S_rate": -saving * 25.0 / 37_545_489.0,
        "tokens_changed": 0,
        "deterministic_repeat": True,
        "stored_candidate_parameter_bytes": 0,
        "admission_bar_bytes": ADMISSION_BAR_BYTES,
        "disposition": disposition,
        "invalid_attempt": invalid_attempt,
    }
    atomic_json(MEASUREMENT / "ADJUDICATION.json", payload)
    return payload


def stage_identity_preflight() -> dict[str, Any]:
    adjudication_path = MEASUREMENT / "ADJUDICATION.json"
    if not adjudication_path.is_file():
        raise Afc1Error("identity preflight requires the exact-rate adjudication")
    adjudication = json.loads(adjudication_path.read_text())
    if adjudication.get("disposition") != "ADMIT_NATIVE_PORT_AND_RECEIVER_IDENTITY_OWED":
        raise Afc1Error("candidate did not clear the native receiver admission bar")
    free = shutil.disk_usage(STORE).free
    enough = free >= IDENTITY_REQUIRED_FREE_BYTES
    payload = {
        "schema": "ddm_afc1_tile48_groupbin8_identity_preflight.v1",
        "axis": "[storage/native-receiver preflight / no score claim]",
        "score_claim": False,
        "candidate_archive": adjudication["candidate_archive"],
        "storage": {
            "path": str(STORE),
            "observed_free_bytes": free,
            "required_free_bytes": IDENTITY_REQUIRED_FREE_BYTES,
            "shortfall_bytes": max(IDENTITY_REQUIRED_FREE_BYTES - free, 0),
            "expected_retained_raw_bytes": EXPECTED_IDENTITY_RAW_BYTES,
            "expected_retained_token_checkpoint_bytes": EXPECTED_IDENTITY_TOKEN_BYTES,
            "scratch_allowance_bytes": IDENTITY_SCRATCH_BYTES,
            "post_run_reserve_bytes": IDENTITY_RESERVE_BYTES,
            "status": "PASS" if enough else "BLOCKED_STORAGE",
        },
        "native_port_status": "OWED",
        "receiver_identity_status": "READY_TO_FIRE" if enough else "NOT_STARTED",
        "payload_materialized_by_this_stage": False,
        "disposition": (
            "FIRE_NATIVE_PORT_AND_FULL_RECEIVER_IDENTITY"
            if enough
            else "QUEUED_AFTER_APDATASTORE_IDENTITY_FLOOR"
        ),
    }
    atomic_json(MEASUREMENT / "IDENTITY_PREFLIGHT.json", payload)
    return payload


def stage_manifest() -> dict[str, Any]:
    entries = []
    manifest_path = MEASUREMENT / "MANIFEST.json"
    for path in sorted(STORE.rglob("*")):
        if path.is_file() and path != manifest_path and not path.name.endswith(".partial"):
            entries.append(
                {
                    "path": str(path.relative_to(STORE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema": "ddm_afc1_tile48_groupbin8_manifest.v1",
        "root": str(STORE),
        "entries": entries,
        "total_bytes": sum(int(row["bytes"]) for row in entries),
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "prepare", "adjudicate", "identity-preflight", "manifest"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = {
        "preflight": stage_preflight,
        "prepare": stage_prepare,
        "adjudicate": stage_adjudicate,
        "identity-preflight": stage_identity_preflight,
        "manifest": stage_manifest,
    }[args.stage]()
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
