"""Cold public parse-back of ntb2's winning HPAC candidate, and its raw identity proof.

`ddm_ntb2_hpac.py` priced `frame_even` at a joint −245 B by twin encodes. That price is
only a candidate once the SHIPPED receiver, run the way the contest runs it, decodes the
new archive — and, because this lever only changes the arithmetic coder's PRIOR, decodes
it to bytes IDENTICAL to move 44's own retained decode. That identity is the whole
distortion argument: if the raw is byte-identical, d_seg and d_pose are not "expected to
be unchanged", they ARE unchanged, and no scorer needs to run at all.

Structurally this follows the landed `ddm_rlc4_public.py` proof: same `inflate.sh`
invocation, same cleared library environment, same native-compile cache shim, same
attempt/checkpoint custody, same refusal to re-render over an unreceipted output.

Axis: [macOS-CPU advisory]. No scorer runs. No score is claimed. No seal is written here.
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments import ddm_ntb2_control as control
from experiments import ddm_ntb2_hpac as hpac
from experiments import ddm_rlc1_run as landed
from experiments.ddm_rlc4_rebase import measure_receiver_digest, measure_runtime_digest

# The storage waterfall, not a loosened guard: VertigoDataTier fell to 28 GiB under five
# live arms while this proof needs 3.66 GB plus headroom, and APDataStore has 63 GiB. Both
# are sanctioned SSD tiers; `--public-root` chooses, and the reserve is unchanged.
DEFAULT_ROOT = control.ROOT.parent / "public"
ROOT = DEFAULT_ROOT
# move 44's own cold public decode, from ddm_rlc5's retained proof.
SOURCE_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw"
)
SOURCE_RAW_SHA = "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
RAW_BYTES = 3_662_409_600
RESERVE_BYTES = 24 << 30


def retain(path: Path, payload: bytes) -> dict:
    """Persist a receipt inside THIS producer's store.

    `hpac.retain` refuses anything outside `hpac_v3/`, which is correct for it and wrong
    here: the parse-back's receipts belong to the parse-back. Same reserve, same
    immutable-write helper, different root.
    """
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside the public-proof store")
    # A RECEIPT is kilobytes and is written AFTER the multi-gigabyte decode it describes.
    # Guarding it with the same 40 GiB reserve that gates the decode would throw away a
    # completed multi-hour proof to save 3 KB, so the bulk gate lives at the decode
    # (RAW_BYTES + RESERVE_BYTES, checked before a byte is written) and this one only
    # refuses when the tier is genuinely out of room.
    if shutil.disk_usage(ROOT.parent).free < (1 << 30) + len(payload):
        raise RuntimeError("STORAGE_BLOCK: keep all existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ntb2 public proof receipt")
    return landed.fact(path)


def record(path: Path, value: dict) -> dict:
    retain(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    return value


def compare_bytes(left: Path, right: Path) -> bool:
    """Byte equality of two multi-GB files without holding either in memory."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk_a = a.read(1 << 22)
            chunk_b = b.read(1 << 22)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shipped_files(root: Path):
    """Every file the receiver tree actually ships.

    APDataStore is ExFAT, so `shutil.copytree` leaves AppleDouble `._*` stubs beside every
    entry. They are filesystem metadata, never receiver content, and they must not reach
    the manifest, the census, or the moved-file guard -- the same stub genus that bites
    arm bundles.
    """
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and not path.name.startswith("._")
            and path.name != ".DS_Store"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            yield path


def regenerate_manifest(root: Path) -> bytes:
    """Rebuild `MANIFEST.sha256` the way the promoted tree's own was built.

    MEASURED rule, falsified against move 44's shipped manifest before use: every file
    except `MANIFEST.sha256` ITSELF and `archive.zip`, sorted, `sha  relpath` per line.
    Regenerating move 44's tree by this rule reproduces its shipped 49-row manifest byte
    for byte; including `archive.zip` gives 50 rows and does not.
    """
    rows = "".join(
        f"{sha256_file(path)}  {path.relative_to(root)}\n"
        for path in shipped_files(root)
        if path.name not in ("MANIFEST.sha256", "archive.zip")
    )
    return rows.encode()


def stage_runtime(treatment: str) -> Path:
    """The PROMOTED tree with the treatment's archive, its pin, and a rebuilt manifest.

    The source is move 44's actually promoted receiver tree, not this arm's working copy:
    the working copy inherited from the codex arm is byte-identical on all 50 files it
    holds but is MISSING `MANIFEST.sha256`, which the promoted tree ships. Staging from
    the working copy produced a tree one file short of the shipped lineage and made this
    arm's first receiver-digest comparison an equality between two trees that both lacked
    the file -- which is how MAIN's warned row came to look absent here.

    `inflate.py` carries `ARCHIVE_SHA256`/`ARCHIVE_BYTES` as a self-check on the artifact
    it was promoted with, so a new archive cannot decode until they name it, and the
    manifest lists `inflate.py`'s raw hash so it follows. This function therefore asserts
    the FILE SET that moved, not a digest equality: exactly `archive.zip`, `inflate.py`
    and `MANIFEST.sha256`, and nothing else.
    """
    runtime = ROOT / treatment / "candidate_runtime"
    source = control.ROOT / "promoted_runtime"
    if not (source / "MANIFEST.sha256").is_file():
        raise SystemExit(f"promoted tree missing its manifest: {source}")
    if (runtime / "MANIFEST.sha256").is_file():
        # Idempotent on resume. Re-staging a tree the receiver has already checkpointed
        # against makes it refuse with "receiver checkpoint binding or bytes changed".
        return runtime
    if runtime.exists():
        shutil.rmtree(runtime)
    shutil.copytree(
        source, runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", ".DS_Store")
    )
    archive = hpac.ROOT / treatment / "retained/archive.twin0.zip"
    if not archive.is_file():
        raise SystemExit(f"no priced archive for {treatment}: {archive}")
    shutil.copy2(archive, runtime / "archive.zip")
    sha = sha256_file(runtime / "archive.zip")
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
    (runtime / "MANIFEST.sha256").write_bytes(regenerate_manifest(runtime))
    moved = sorted(
        path.relative_to(runtime).as_posix()
        for path in shipped_files(runtime)
        if (
            not (source / path.relative_to(runtime)).is_file()
            or path.read_bytes() != (source / path.relative_to(runtime)).read_bytes()
        )
    )
    if moved != ["MANIFEST.sha256", "archive.zip", "inflate.py"]:
        raise SystemExit(f"RECEIVER CHANGED beyond archive + pin + manifest: {moved}")
    return runtime


def run(treatment: str, timeout: int) -> dict:
    work = ROOT / treatment
    for name in ("data", "output", "scratch", "frame_checkpoints", "retained_native", "attempts"):
        (work / name).mkdir(parents=True, exist_ok=True)
    price = json.loads((hpac.ROOT / treatment / "PRICE.json").read_text())
    if price["twins"][0]["sha256"] != price["twins"][1]["sha256"]:
        raise SystemExit("twins disagree; nothing to parse back")
    runtime = stage_runtime(treatment)
    archive = landed.fact(runtime / "archive.zip")
    if archive["sha256"] != price["twins"][0]["sha256"]:
        raise SystemExit("staged archive is not the priced archive")
    binding = {
        "treatment": treatment,
        "runtime_sha256": measure_runtime_digest(runtime).sha256,
        "receiver_sha256": measure_receiver_digest(runtime),
        "archive": archive,
        "base_archive_sha256": control.BASE_SHA,
        "source_raw_sha256": SOURCE_RAW_SHA,
        "price": landed.fact(hpac.ROOT / treatment / "PRICE.json"),
        "producer": landed.fact(Path(__file__)),
        "hpac_producer": landed.fact(REPO / "experiments/ddm_ntb2_hpac.py"),
        "score_claim": False,
    }
    # INPUTS.json holds the IDENTITY of the object under proof and nothing else, so it is
    # stable across producer edits: the receipts are written immutably, and a binding that
    # carried this file's own sha would refuse its own resume the moment the producer was
    # fixed. The full binding, producer sha included, is recorded per ATTEMPT below, where
    # a new value is a new file rather than a conflict.
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
    # rlc4 wrapped CC in a compile CACHE keyed to ITS run root, which refuses a request
    # from any other arm's tree ("unrecognized owned compile request") and additionally
    # requires the compile destination to land under TMPDIR -- which pc3 MEASURED macOS
    # mktemp to ignore. Dropping the shim lets `inflate.sh` invoke the real compiler
    # exactly as the contest would, which is more faithful, not less; the only thing lost
    # is compile reuse across resumed attempts.
    env.pop("CC", None)
    # These would let a stale library or a cached decode stand in for the real one.
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
        raise SystemExit("move 44 retained raw drift")
    identical = compare_bytes(output, SOURCE_RAW)
    result = {
        "schema": "ddm_ntb2_public.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "binding": binding,
        "treatment": treatment,
        "candidate_raw": candidate_raw,
        "pointer_raw": pointer_raw,
        "raw_byte_identical_to_move44": identical,
        "n_samples": 600,
        "cold_start": not resumed,
        "command": command,
        "stdout": landed.fact(attempt / "run.log"),
        "wall_seconds": wall,
        "no_scorer_ran": True,
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
    parser.add_argument("--treatment", default="frame_even", choices=hpac.TREATMENTS[1:])
    parser.add_argument("--timeout", type=int, default=36000)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=None,
        help="tier to run the proof on; defaults to this arm's own store",
    )
    args = parser.parse_args(argv)
    if args.public_root is not None:
        global ROOT
        ROOT = args.public_root.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None and args.resume_from.resolve() != ROOT.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    result = run(args.treatment, args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k != "binding"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
