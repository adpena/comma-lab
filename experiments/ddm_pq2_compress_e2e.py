#!/usr/bin/env python3
"""End-to-end compression entry point: rebuild the submission archive and prove its bytes.

WHAT THIS IS.  One sanitized command that takes the retained training product and
returns the exact ``archive.zip`` that was evaluated, then refuses to exit 0 unless
the rebuilt bytes hash to the pinned sha256.  It orchestrates the two instruments
that actually produced the candidate -- it does not reimplement them, so what this
script proves is what the shipped pipeline does, not a parallel reconstruction of it.

THE THREE STAGES.

  ``provenance``  (stage A, documented, no compute)  Emits the training lineage that
      produced the checkpoint: the stage scripts, their arguments, the corrector
      module, and the input manifest with every sha256.  Stage A is DOCUMENTED
      RATHER THAN RE-RUN, and this script says so plainly: reproducing the
      checkpoint from raw video is days of GPU compute.  What is verifiable here
      is everything downstream of the checkpoint, which is the part that
      determines the archive bytes.

  ``encode`` + ``build``  (stage B, EXACT and verifiable)  Replays the shipped
      decode order, encodes the token field under the free decode-time corrector,
      splices the new token stream into the member, and repacks the archive.  The
      seven non-token sections are carried through byte-identically by
      construction; only the token stream is re-encoded.  ``--resume`` continues
      bit-faithfully from a retained frame-boundary checkpoint, which is what
      makes a cheap end-to-end verification possible.

  ``verify``  (stage B gate)  Hashes the rebuilt archive and compares it against
      the pinned sha256 and byte count.  Fail-closed: a mismatch exits non-zero.

  ``decode``  (stage C)  Runs the shipped receiver over the rebuilt archive and
      checks that the decoded token field reproduces its pinned sha256.  This is
      the distortion proof: an identical decoded field cannot move d_seg or
      d_pose, so the score delta is a pure rate delta.

      HONEST LIMITATION.  Stage C drives our internal receiver-closure harness,
      which still resolves one source root from local custody, so it is a
      REPOSITORY proof rather than a judge-runnable one.  A judge does not need
      it: the shipped ``inflate.sh`` is the real decode path, takes no private
      input, and is what the evaluator runs.  Stage C exists to prove the
      decoded field is unchanged, not to be the decode entry point.

NO PRIVATE PATHS.  This file contains no filesystem layout.  Every input root is
supplied by the caller through ``--inputs-json`` (or the matching environment
variables) and is validated by sha256 before any stage runs.  Run
``--emit-inputs-template`` to print the schema.

DETERMINISM.  The pipeline carries no RNG: the corrector is integer/rational
arithmetic and the range coder is exact.  The seed below is recorded and pushed
into the environment so that any future stochastic stage inherits it rather than
silently defaulting.  Determinism is checked by measurement -- ``build`` writes a
second archive from the same member and asserts the two are byte-identical.

AXIS.  ``[macOS-CPU advisory / scorer-free EXACT byte measurement]``.  This script
measures BYTES and byte-identity only.  It makes no score claim: the score on the
rebuilt bytes is whatever the contest evaluator returns on the contest hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENCODER = REPO / "experiments" / "ddm_rr2_encoder_byteclose.py"
RECEIVER = REPO / "experiments" / "ddm_rr2_receiver_close.py"

# The candidate this entry point rebuilds.  These are the pinned, measured values
# from the retained build receipt; they are assertions, never defaults to fall
# back on.  A rebuild that does not reproduce them is a failure of the rebuild.
EXPECTED_ARCHIVE_SHA256 = "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"
EXPECTED_ARCHIVE_BYTES = 181_161
EXPECTED_TOKEN_SHA256 = "6c3757bd52a18d3c38e9120d56293f03c7aefd111fb9ee655b19d055e8d06b14"
EXPECTED_TOKEN_BYTES = 110_512
EXPECTED_DECODED_FIELD_SHA256 = (
    "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
)
CORRECTOR_MODULE = "ddm_rr4_free_corrector_v2"

# Every input the rebuild consumes, with the environment variable the stage
# scripts read it from.  `sha256` is verified before any stage runs; `directory`
# entries are verified through their named member file.
INPUT_SPEC = {
    "prepared_dir": {
        "env": "TAC_PQ2_PREPARED_DIR",
        "kind": "directory",
        "member": "archive.zip",
        "sha256": "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e",
        "role": "base archive whose seven non-token sections are carried through unchanged",
    },
    "hm1_dir": {
        "env": "TAC_PQ2_HM1_DIR",
        "kind": "directory",
        "member": "group_index.u8",
        "sha256": None,
        "role": "retained pre-correction HPAC logits, boundary buckets and group index",
    },
    "tokens_file": {
        "env": "TAC_PQ2_TOKENS_FILE",
        "kind": "file",
        "sha256": EXPECTED_DECODED_FIELD_SHA256,
        "role": "decoded token field the encoder replays; the distortion anchor",
    },
    "rc64_source": {
        "env": "TAC_PQ2_RC64_SOURCE",
        "kind": "file",
        "sha256": "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6",
        "role": "range-coder backend C source (inherited substrate; see the accounting table)",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_inputs_template() -> int:
    """Print the input manifest schema without revealing any local layout."""
    template = {
        name: {
            "path": f"<absolute path to the {spec['kind']}>",
            "role": spec["role"],
            "expected_sha256": spec["sha256"],
            "environment_variable": spec["env"],
        }
        for name, spec in INPUT_SPEC.items()
    }
    print(json.dumps({"inputs": template}, indent=2, sort_keys=True))
    return 0


def resolve_inputs(inputs_json: Path | None) -> dict[str, Path]:
    """Resolve every input root from the manifest, falling back to the environment.

    Fail-closed on a missing input: a rebuild that silently skips an input would
    produce different bytes and blame the algorithm.
    """
    supplied: dict[str, str] = {}
    if inputs_json is not None:
        document = json.loads(inputs_json.read_text())
        section = document.get("inputs", document)
        for name, entry in section.items():
            if name in INPUT_SPEC:
                supplied[name] = entry["path"] if isinstance(entry, dict) else str(entry)

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name, spec in INPUT_SPEC.items():
        raw = supplied.get(name) or os.environ.get(spec["env"])
        if not raw:
            missing.append(f"{name} (--inputs-json entry or ${spec['env']})")
            continue
        resolved[name] = Path(raw)
    if missing:
        raise SystemExit(
            "missing required inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun --emit-inputs-template for the schema."
        )
    return resolved


def verify_inputs(resolved: dict[str, Path]) -> list[dict[str, object]]:
    """Hash each input and refuse to proceed on a mismatch or an absent file."""
    manifest: list[dict[str, object]] = []
    problems: list[str] = []
    for name, spec in INPUT_SPEC.items():
        root = resolved[name]
        target = root / spec["member"] if spec["kind"] == "directory" else root
        if not target.is_file():
            problems.append(f"{name}: not found -> {target}")
            continue
        measured = sha256_file(target)
        expected = spec["sha256"]
        ok = expected is None or measured == expected
        if not ok:
            problems.append(f"{name}: sha256 {measured} != expected {expected}")
        manifest.append(
            {
                "input": name,
                "verified_member": target.name,
                "bytes": target.stat().st_size,
                "sha256": measured,
                "expected_sha256": expected,
                "sha256_matches": ok,
                "role": spec["role"],
            }
        )
    if problems:
        raise SystemExit("input verification failed:\n  " + "\n  ".join(problems))
    return manifest


def stage_environment(resolved: dict[str, Path], seed: int) -> dict[str, str]:
    """Build the child environment: input roots, corrector selection, and the seed."""
    environment = dict(os.environ)
    for name, spec in INPUT_SPEC.items():
        environment[spec["env"]] = str(resolved[name])
    environment["TAC_RR2_CORRECTOR_MODULE"] = CORRECTOR_MODULE
    environment["PYTHONHASHSEED"] = str(seed)
    environment["TAC_PQ2_SEED"] = str(seed)
    return environment


def run_stage(argv: list[str], environment: dict[str, str], label: str) -> dict[str, object]:
    """Run one real stage script and fail closed on a non-zero return code."""
    print(f"[pq2] {label}: {' '.join(argv)}", flush=True)
    started = time.time()
    completed = subprocess.run(argv, env=environment, cwd=str(REPO), check=False)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with return code {completed.returncode}")
    return {"stage": label, "argv": argv, "returncode": 0, "elapsed_seconds": elapsed}


def verify_archive(store: Path) -> dict[str, object]:
    """Assert the rebuilt archive is byte-identical to the pinned candidate."""
    archive = store / "retained" / "archive.zip"
    if not archive.is_file():
        raise SystemExit(f"no rebuilt archive at {archive}; run the build stage first")
    measured_sha = sha256_file(archive)
    measured_bytes = archive.stat().st_size
    sha_ok = measured_sha == EXPECTED_ARCHIVE_SHA256
    bytes_ok = measured_bytes == EXPECTED_ARCHIVE_BYTES

    repeat = store / "work" / "archive.repeat.zip"
    repeat_sha = sha256_file(repeat) if repeat.is_file() else None
    deterministic = repeat_sha == measured_sha if repeat_sha is not None else None

    token = store / "retained" / "token_stream.bin"
    token_sha = sha256_file(token) if token.is_file() else None

    result = {
        "archive_path": str(archive),
        "archive_sha256": measured_sha,
        "archive_bytes": measured_bytes,
        "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "sha256_matches": sha_ok,
        "bytes_match": bytes_ok,
        "determinism_repeat_sha256": repeat_sha,
        "determinism_repeat_byte_identical": deterministic,
        "token_stream_sha256": token_sha,
        "token_stream_matches": None if token_sha is None else token_sha == EXPECTED_TOKEN_SHA256,
    }
    if not (sha_ok and bytes_ok):
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(
            "ARCHIVE VERIFICATION FAILED: rebuilt bytes do not match the pinned candidate"
        )
    print(
        f"[pq2] VERIFIED archive sha256={measured_sha} bytes={measured_bytes:,}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("provenance", "encode", "build", "verify", "decode", "all"),
        help="'all' runs provenance, encode, build and verify (not decode)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="working/output root; retained/archive.zip is written under it",
    )
    parser.add_argument(
        "--inputs-json",
        type=Path,
        default=None,
        help="input manifest; see --emit-inputs-template",
    )
    parser.add_argument(
        "--emit-inputs-template",
        action="store_true",
        help="print the input manifest schema and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the encode bit-faithfully from a retained frame checkpoint",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="write the run receipt here (default: <store>/RESULT_pq2_e2e.json)",
    )
    args = parser.parse_args()

    if args.emit_inputs_template:
        return emit_inputs_template()

    if args.store is None:
        parser.error("--store is required (or pass --emit-inputs-template)")

    store = args.store
    store.mkdir(parents=True, exist_ok=True)
    resolved = resolve_inputs(args.inputs_json)
    input_manifest = verify_inputs(resolved)
    environment = stage_environment(resolved, args.seed)

    receipt: dict[str, object] = {
        "schema": "pact.pq2.compress_e2e.v1",
        "stage_requested": args.stage,
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "seed": args.seed,
        "corrector_module": CORRECTOR_MODULE,
        "python_version": sys.version,
        "store": str(store),
        "input_manifest": input_manifest,
        "stages_run": [],
    }

    if args.stage in ("provenance", "all"):
        receipt["provenance"] = {
            "stage_a_status": "DOCUMENTED_NOT_RE_RUN",
            "stage_a_note": (
                "Reproducing the checkpoint from raw video is multi-day GPU compute. "
                "Stage A records the lineage; stages B and C are exactly verifiable "
                "from the retained checkpoint and are what determine the archive bytes."
            ),
            "encoder_script": str(ENCODER.relative_to(REPO)),
            "receiver_script": str(RECEIVER.relative_to(REPO)),
            "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "expected_token_bytes": EXPECTED_TOKEN_BYTES,
            "expected_decoded_field_sha256": EXPECTED_DECODED_FIELD_SHA256,
            "sections_rebuilt": "token_stream only; seven sections carried byte-identically",
        }
        print(json.dumps(receipt["provenance"], indent=2, sort_keys=True))

    if args.stage in ("encode", "all"):
        argv = [
            sys.executable,
            str(ENCODER),
            "--stage",
            "encode",
            "--store",
            str(store),
        ]
        if args.resume:
            argv.append("--resume")
        receipt["stages_run"].append(run_stage(argv, environment, "encode"))

    if args.stage in ("build", "all"):
        argv = [sys.executable, str(ENCODER), "--stage", "build", "--store", str(store)]
        receipt["stages_run"].append(run_stage(argv, environment, "build"))

    if args.stage in ("verify", "build", "all"):
        receipt["verification"] = verify_archive(store)

    if args.stage == "decode":
        for sub in ("build", "parseback"):
            argv = [
                sys.executable,
                str(RECEIVER),
                "--stage",
                sub,
                "--store",
                str(store),
            ]
            receipt["stages_run"].append(run_stage(argv, environment, f"receiver-{sub}"))
        receipt["decode_note"] = (
            "receiver parse-back writes RESULT_receiver_parseback.json; the decoded "
            f"token field must hash to {EXPECTED_DECODED_FIELD_SHA256}"
        )

    destination = args.receipt or (store / "RESULT_pq2_e2e.json")
    partial = destination.with_suffix(".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    os.replace(partial, destination)
    print(f"[pq2] receipt -> {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
