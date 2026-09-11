"""ddm_hpr1 -- cold public parse-back of the RETRAIN CONTROL candidate on move 45.

`ddm_hpr1_shape_price.py` priced `retrain` at 179,359 B (−887 B) by twin encodes. That
price is only a candidate once the SHIPPED receiver, run the way the contest runs it,
decodes the new archive -- and, because this lever changes only the arithmetic coder's
PRIOR, decodes it to raw bytes IDENTICAL to the pointer's own retained decode. That
identity is the whole distortion argument: if the raw is byte-identical, d_seg and d_pose
are not "expected to be unchanged", they ARE unchanged, and no scorer needs to run.

WHY THIS FILE EXISTS instead of a flag on ntb2's prover. `ddm_ntb2_public.py` is the
landed proof and its logic is right; but its `--treatment` is an ntb2 enum that it writes
verbatim into every receipt it emits, so proving THIS arm's bytes under one of those names
would stamp a false label into custody. Editing a sister arm's producer while it runs is
out of this arm's boundaries. So the pure, load-bearing helpers are IMPORTED from it --
above all `regenerate_manifest`, whose rule was falsified against move 44's shipped
manifest before use -- and only the labels, the roots and the bindings are this arm's.
Nothing about the decode, the environment clearing or the raw comparison is re-derived.

Axis: [macOS-CPU advisory]. No scorer runs. No score is claimed. No seal is written here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments import ddm_hpr1_shape_price as price_producer
from experiments import ddm_ntb2_public as base
from experiments import ddm_rlc1_run as landed
from experiments.ddm_rlc4_rebase import measure_receiver_digest, measure_runtime_digest

#: This arm's own store, on Vertigo. MAIN routed NEW payloads here on 2026-09-11 and
#: APDataStore is never written by this producer.
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/public")
#: The PROMOTED move-45 receiver tree, which ships its own MANIFEST.sha256. Copied, never
#: edited: it is sealed.
PROMOTED = price_producer.PROMOTED45
CANDIDATE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/price/retrain/retained/archive.twin0.zip")
PRICE_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/price/retrain/PRICE.json")
#: The pointer's own retained cold decode. Move 45's raw is byte-identical to move 44's
#: (both `2b762eba…`) because move 45 moved the carrier member and not the frames.
SOURCE_RAW = base.SOURCE_RAW
SOURCE_RAW_SHA = base.SOURCE_RAW_SHA
RAW_BYTES = base.RAW_BYTES
RESERVE_BYTES = base.RESERVE_BYTES
#: The truthful name of what is being proven. It is this arm's treatment, so it is this
#: arm's word, and it is the same string the price receipt carries.
TREATMENT = "retrain_control"


def retain(path: Path, payload: bytes) -> dict:
    """Persist a receipt inside THIS producer's store, under the same reserve law."""
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside the public-proof store")
    # The bulk gate lives at the decode (RAW_BYTES + RESERVE_BYTES, checked before a byte
    # is written); a kilobyte receipt written AFTER a multi-hour proof must not be the
    # thing that throws that proof away.
    if shutil.disk_usage(ROOT.parent).free < (1 << 30) + len(payload):
        raise RuntimeError("STORAGE_BLOCK: keep all existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ddm_hpr1 public proof receipt")
    return landed.fact(path)


def record(path: Path, value: dict) -> dict:
    retain(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    return value


def stage_runtime() -> Path:
    """The PROMOTED move-45 tree with this candidate's archive, its pin, and a rebuilt manifest.

    `inflate.py` carries `ARCHIVE_SHA256`/`ARCHIVE_BYTES` as a self-check on the artifact
    it was promoted with, so a new archive cannot decode until they name it, and the
    manifest lists `inflate.py`'s raw hash so it follows. The assertion is therefore on
    the FILE SET that moved: exactly `archive.zip`, `inflate.py` and `MANIFEST.sha256`,
    and nothing else -- which is also the proof that this candidate is NOT a receiver
    change.
    """
    runtime = ROOT / TREATMENT / "candidate_runtime"
    if not (PROMOTED / "MANIFEST.sha256").is_file():
        raise SystemExit(f"promoted tree missing its manifest: {PROMOTED}")
    if (runtime / "MANIFEST.sha256").is_file():
        # Idempotent on resume: re-staging a tree the receiver has already checkpointed
        # against makes it refuse with "receiver checkpoint binding or bytes changed".
        return runtime
    if runtime.exists():
        shutil.rmtree(runtime)
    shutil.copytree(
        PROMOTED, runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store")
    )
    if not CANDIDATE_ARCHIVE.is_file():
        raise SystemExit(f"no priced archive: {CANDIDATE_ARCHIVE}")
    shutil.copy2(CANDIDATE_ARCHIVE, runtime / "archive.zip")
    sha = base.sha256_file(runtime / "archive.zip")
    size = (runtime / "archive.zip").stat().st_size
    inflate = runtime / "inflate.py"
    text = inflate.read_text()
    for name, value in (("ARCHIVE_SHA256", f'"{sha}"'), ("ARCHIVE_BYTES", str(size))):
        marker = f"{name} = "
        if text.count(marker) != 1:
            raise SystemExit(f"{name} is not a single explicit assignment in inflate.py")
        head, rest = text.split(marker, 1)
        text = head + marker + value + rest[rest.index("\n") :]
    inflate.write_text(text)
    (runtime / "MANIFEST.sha256").write_bytes(base.regenerate_manifest(runtime))
    moved = sorted(
        path.relative_to(runtime).as_posix()
        for path in base.shipped_files(runtime)
        if (
            not (PROMOTED / path.relative_to(runtime)).is_file()
            or path.read_bytes() != (PROMOTED / path.relative_to(runtime)).read_bytes()
        )
    )
    if moved != ["MANIFEST.sha256", "archive.zip", "inflate.py"]:
        raise SystemExit(f"RECEIVER CHANGED beyond archive + pin + manifest: {moved}")
    return runtime


def run(timeout: int) -> dict:
    work = ROOT / TREATMENT
    for name in ("data", "output", "scratch", "frame_checkpoints", "attempts"):
        (work / name).mkdir(parents=True, exist_ok=True)
    price = json.loads(PRICE_RECEIPT.read_text())
    if price["twins"][0]["sha256"] != price["twins"][1]["sha256"]:
        raise SystemExit("twins disagree; nothing to parse back")
    if not price["output_lossless"]:
        raise SystemExit("price receipt does not claim output-losslessness")
    runtime = stage_runtime()
    archive = landed.fact(runtime / "archive.zip")
    if archive["sha256"] != price["twins"][0]["sha256"]:
        raise SystemExit("staged archive is not the priced archive")
    binding = {
        "treatment": TREATMENT,
        "runtime_sha256": measure_runtime_digest(runtime).sha256,
        "receiver_sha256": measure_receiver_digest(runtime),
        "archive": archive,
        "base_archive_sha256": price_producer.POINTER45_SHA,
        "source_raw_sha256": SOURCE_RAW_SHA,
        "price": landed.fact(PRICE_RECEIPT),
        "producer": landed.fact(Path(__file__)),
        "price_producer": landed.fact(REPO / "experiments/ddm_hpr1_shape_price.py"),
        "forked_from": landed.fact(REPO / "experiments/ddm_ntb2_public.py"),
        "score_claim": False,
    }
    identity = {k: binding[k] for k in (
        "treatment",
        "runtime_sha256",
        "receiver_sha256",
        "archive",
        "base_archive_sha256",
        "source_raw_sha256",
    )}
    inputs = work / "INPUTS.json"
    if inputs.exists() and json.loads(inputs.read_text()) != identity:
        raise SystemExit("public proof source/config drift")
    record(inputs, identity)

    done = work / "RESULT.json"
    if done.exists():
        return json.loads(done.read_text())
    output = work / "output/0.raw"
    if output.exists() and not (work / "output/.f26_decode_checkpoints").exists():
        raise SystemExit("unreceipted raw: preserve and recover the log, never re-render over it")
    if shutil.disk_usage(ROOT.parent).free < RAW_BYTES + RESERVE_BYTES:
        raise RuntimeError("STORAGE_BLOCK: keep all bytes")

    with zipfile.ZipFile(runtime / "archive.zip") as bundle:
        (work / "data/p").write_bytes(bundle.read("p"))
    (work / "file_list.txt").write_bytes(b"0.hevc\n")
    env = dict(
        os.environ,
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "scratch"),
        RLC1_ADVISORY_CPU="1",
        RLC1_PROOF_BLAS_THREADS="4",
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER="600",
        F26_TOKEN_DECODER="python",
    )
    # Let `inflate.sh` invoke the real compiler exactly as the contest would.
    env.pop("CC", None)
    # These would let a stale library or a CACHED decode stand in for the real one, which
    # is precisely what this proof exists to rule out.
    for key in (
        "CPR1_RC64_LIBRARY",
        "F26_CORRECTOR_NATIVE_LIBRARY",
        "F26_HPAC_NATIVE_LIBRARY",
        "RLC1_GEOMETRY_LIBRARY",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "F26_ADVISORY_RENDER_WORKERS",
        "F26_ADVISORY_RENDER_RSS_BYTES",
    ):
        env.pop(key, None)
    command = [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ]
    attempts = work / "attempts"
    attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
    attempt.mkdir()
    resumed = (work / "frame_checkpoints/LATEST.json").exists()
    record(
        attempt / "COMMAND.json",
        {
            "argv": command,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith(("RLC", "TC1", "F26", "OMP", "MKL", "OPENBLAS", "VECLIB"))
                or k in ("CC", "TMPDIR", "PATH", "PYTHONDONTWRITEBYTECODE")
            },
            "binding": binding,
            "resumed": resumed,
            "score_claim": False,
        },
    )
    started = time.perf_counter()
    with (attempt / "run.log").open("wb") as log:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
        )
    wall = time.perf_counter() - started
    record(
        attempt / "PROCESS.json",
        {
            "returncode": completed.returncode,
            "wall_seconds": wall,
            "resumed": resumed,
            "axis": "[macOS-CPU advisory diagnostic; NOT a contest decode-time receipt]",
            "score_claim": False,
        },
    )
    if completed.returncode:
        raise RuntimeError(f"public shell failed rc={completed.returncode}; log {attempt / 'run.log'}")

    candidate_raw = landed.fact(output)
    pointer_raw = landed.fact(SOURCE_RAW)
    if pointer_raw["sha256"] != SOURCE_RAW_SHA:
        raise SystemExit("pointer retained raw drift")
    identical = base.compare_bytes(output, SOURCE_RAW)
    result = {
        "schema": "ddm_hpr1_public.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "binding": binding,
        "treatment": TREATMENT,
        "candidate_raw": candidate_raw,
        "pointer_raw": pointer_raw,
        "raw_byte_identical_to_pointer": identical,
        "decoded_field_sha256": price["decoded_field_sha256"],
        "n_samples": 600,
        "cold_start": not resumed,
        "command": command,
        "stdout": landed.fact(attempt / "run.log"),
        "wall_seconds": wall,
        "no_scorer_ran": True,
        "receiver_change": False,
        "distortion_argument": (
            "the archive changes only the arithmetic coder's PRIOR, so a byte-identical raw IS "
            "the d_seg and d_pose proof: the frames the scorer would see are the same frames"
        ),
    }
    if not identical or candidate_raw["bytes"] != RAW_BYTES:
        result["status"] = "RAW_IDENTITY_FAILED"
        record(work / "RAW_IDENTITY_FAILED.json", result)
        raise RuntimeError("raw identity failed; every byte retained")
    if binding["runtime_sha256"] != measure_runtime_digest(runtime).sha256:
        raise SystemExit("runtime changed during the proof")
    return record(done, result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=36000)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--public-root", type=Path, default=None)
    args = parser.parse_args(argv)
    global ROOT
    if args.public_root is not None:
        ROOT = args.public_root.resolve()
    if str(ROOT).startswith("/Volumes/APDataStore"):
        raise SystemExit("APDataStore is not this producer's tier")
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None and args.resume_from.resolve() != ROOT.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    result = run(args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k != "binding"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
