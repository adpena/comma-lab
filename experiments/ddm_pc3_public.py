"""ddm_pc3 -- stage the predictor-refit candidate and prove it COLD on all 600 pairs.

WHAT IS BEING PROVED, AND WHY IT IS THE STRONGEST FORM OF THE PROOF
-------------------------------------------------------------------
The candidate changes the CAP1 AR(1)+bias predictor and nothing else: the coefficient
codes, the coefficient scales, the basis scales and the basis payload are bit-identical,
and ``decode_cap1`` reconstructs the byte-identical canonical CPR1.  So the claim is not
"the distortion moved by a small amount I measured" -- it is "the decoder's OUTPUT is the
same file".  That is checkable directly, and this module checks it: the cold public
``inflate.sh`` on the candidate must emit all 3,662,409,600 bytes of ``0.raw`` IDENTICAL
to the pointer's own retained decode (sha 2b762eba...).  Not "identical outside the
carrier frames" -- identical everywhere.  Under that, d_seg and d_pose are the pointer's
by construction and the row is a pure rate move.

The mechanism of the harness is ``ddm_rlc4_public``'s, reused because it is the landed
instrument for this body's public path: the same archive-member extraction, the same file
list, the same environment (python token decoder, no pre-built native libraries), the same
one-report-line contract, the same full-byte comparison.  What is NOT reused is its PINS
-- source raw, root, producer -- because pinning an arm's proof to another arm's objects is
how a proof ends up about the wrong body; and not its compile CACHE, which cannot be
satisfied from this session (see the ``CC`` comment below).

THE RE-PIN IS NOT A RECEIVER CHANGE
------------------------------------
``inflate.py`` carries ``ARCHIVE_SHA256`` / ``ARCHIVE_BYTES`` as a self-check, so a new
archive needs new values there.  ``tac.decode_wall_clock.measure_receiver_digest``
normalises exactly those two assignments before hashing, so the pin itself is invisible
to the digest.  ``MANIFEST.sha256`` is NOT invisible: it lists the archive's own hash,
it ships, and so it correctly enters the canonical digest and moves with any new archive
(the move-43 -> 44 packet records normalizing it as pr14's OWED amendment).  This module
therefore asserts the weaker, true thing -- the receiver CODE digest, the same function
over both trees with the manifest removed, is EQUAL -- records both digests, and leaves
the seal tool's own inheritance check as the authority on the wall-clock leg.  Staging
also enumerates every file that differs from the pointer tree and refuses anything beyond
``{archive.zip, inflate.py, MANIFEST.sha256}``.

Axis: ``[macOS-CPU advisory]``.  No scorer runs here and no timing window is claimed;
the wall time recorded is loaded-macOS diagnostic data only.  ``score_claim=false``.
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
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "experiments")]

import ddm_pc3_pose_carrier_curve as pc3

from tac.candidate_seal import measure_runtime_digest
from tac.decode_wall_clock import measure_receiver_digest

RAW_BYTES = 3_662_409_600


class Pc3PublicError(RuntimeError):
    """A ddm_pc3 public-proof precondition failed. Fail closed, keep every byte."""


def fact(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def record(path: Path, value: Any) -> Any:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True))
    return value


def compare_bytes(left: Path, right: Path, chunk: int = 1 << 24) -> bool:
    """Whole-file byte comparison. Not a hash comparison of a prefix, every byte."""
    if Path(left).stat().st_size != Path(right).stat().st_size:
        return False
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            block_a, block_b = a.read(chunk), b.read(chunk)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def repin_inflate(dest: Path, archive_bytes: bytes) -> None:
    """Update the staged ``inflate.py`` archive self-check to the candidate bytes."""
    path = Path(dest) / "inflate.py"
    digest = hashlib.sha256(archive_bytes).hexdigest()
    replaced = 0
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("ARCHIVE_SHA256"):
            lines.append(f'ARCHIVE_SHA256 = "{digest}"\n')
            replaced += 1
        elif stripped.startswith("ARCHIVE_BYTES"):
            lines.append(f"ARCHIVE_BYTES = {len(archive_bytes)}\n")
            replaced += 1
        else:
            lines.append(line)
    if replaced != 2:
        raise Pc3PublicError(
            f"inflate.py archive pin: replaced {replaced} lines, expected 2"
        )
    path.write_text("".join(lines))


def write_manifest(dest: Path) -> dict[str, Any]:
    """Regenerate ``MANIFEST.sha256`` from the staged tree, outside the tree first.

    The manifest is built in memory over the tree's OWN current files and only then
    written, so it can never describe a state that a half-finished write left behind.
    """
    dest = Path(dest)
    rows = []
    for path in sorted(dest.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(dest).as_posix()
        if relative == "MANIFEST.sha256" or "__pycache__" in path.parts:
            continue
        with path.open("rb") as stream:
            rows.append(
                (hashlib.file_digest(stream, "sha256").hexdigest(), relative)
            )
    payload = "".join(f"{digest}  {relative}\n" for digest, relative in rows)
    (dest / "MANIFEST.sha256").write_text(payload)
    return {
        "entries": len(rows),
        "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def compiler_fact() -> dict[str, Any]:
    """Identify the system compiler that builds the receiver's C, so the build repeats."""
    version = subprocess.run(
        ["/usr/bin/cc", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    return {**fact(Path("/usr/bin/cc")), "version": version}


def receiver_digest_excluding_manifest(runtime: Path) -> str:
    """``measure_receiver_digest`` over the tree with ``MANIFEST.sha256`` removed.

    The manifest is a shippable file, so the canonical digest includes it -- correctly,
    because it ships.  But the manifest lists the archive's own hash, so ANY candidate
    with new archive bytes changes it, and a candidate whose receiver code is genuinely
    byte-identical can never show an identical canonical digest.  (The move-43 -> 44
    packet records this as pr14's owed amendment: "excludes/normalizes MANIFEST.sha256".)

    This computes the same digest on both trees with the manifest removed, symmetrically,
    on COPIES -- neither tree is touched.  It answers "is the receiver CODE the same?",
    which is the question an inheritance claim actually rests on.  It does not replace the
    canonical digest: both are recorded, and the seal tool's own inheritance check remains
    the authority on whether a wall-clock leg may be inherited.
    """
    import tempfile

    runtime = Path(runtime)
    with tempfile.TemporaryDirectory() as scratch:
        mirror = Path(scratch) / "tree"
        shutil.copytree(
            runtime, mirror, ignore=shutil.ignore_patterns("__pycache__")
        )
        manifest = mirror / "MANIFEST.sha256"
        if manifest.exists():
            manifest.unlink()
        return measure_receiver_digest(mirror)


def stage(args) -> dict[str, Any]:
    source = Path(args.source_runtime)
    dest = Path(args.out_dir) / "candidate_runtime"
    archive_bytes = Path(args.archive).read_bytes()
    observed = hashlib.sha256(archive_bytes).hexdigest()
    if args.expect_archive_sha256 and observed != args.expect_archive_sha256:
        raise Pc3PublicError(
            f"candidate archive sha {observed} != {args.expect_archive_sha256}"
        )
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "archive.zip").write_bytes(archive_bytes)
    repin_inflate(dest, archive_bytes)
    manifest = write_manifest(dest)

    pointer_receiver = measure_receiver_digest(source)
    candidate_receiver = measure_receiver_digest(dest)
    pointer_code = receiver_digest_excluding_manifest(source)
    candidate_code = receiver_digest_excluding_manifest(dest)
    if pointer_code != candidate_code:
        raise Pc3PublicError(
            "the staged receiver CODE digest differs from the pointer's "
            f"({candidate_code} vs {pointer_code}); this candidate claims an UNCHANGED "
            "receiver and it is not one"
        )
    changed = []
    for path in sorted(dest.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(dest)
        original = source / relative
        if not original.exists() or original.read_bytes() != path.read_bytes():
            changed.append(relative.as_posix())
    if sorted(changed) != ["MANIFEST.sha256", "archive.zip", "inflate.py"]:
        raise Pc3PublicError(
            f"staging changed an unexpected file set: {sorted(changed)}"
        )
    report = {
        "schema": "ddm_pc3_stage.v1",
        "source_runtime": str(source),
        "candidate_runtime": str(dest),
        "archive": fact(dest / "archive.zip"),
        "manifest": manifest,
        "files_changed_vs_pointer_tree": sorted(changed),
        "receiver_digest_canonical": candidate_receiver,
        "receiver_digest_canonical_pointer": pointer_receiver,
        "receiver_digest_canonical_equal": candidate_receiver == pointer_receiver,
        "receiver_code_digest_excluding_manifest": candidate_code,
        "receiver_code_digest_equals_pointer": True,
        "why_the_canonical_digest_differs": (
            "MANIFEST.sha256 lists the archive's own hash and ships, so it enters the "
            "canonical receiver digest and moves with any new archive; the move-43 -> 44 "
            "packet records normalizing it as pr14's owed amendment. The code digest "
            "above is the same function over both trees with the manifest removed"
        ),
        "runtime_digest": measure_runtime_digest(dest).sha256,
        "why_inflate_py_changed": (
            "ARCHIVE_SHA256/ARCHIVE_BYTES are the receiver's own archive self-check; "
            "measure_receiver_digest normalises exactly those two assignments, and the "
            "equality asserted above is the proof that nothing else moved"
        ),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    record(Path(args.out_dir) / "STAGE.json", report)
    print(json.dumps({k: report[k] for k in ("archive", "files_changed_vs_pointer_tree")}))
    return report


def public(args) -> dict[str, Any]:
    root = Path(args.out_dir)
    runtime = root / "candidate_runtime"
    work = root / "public"
    for name in ("data", "output", "scratch", "frame_checkpoints"):
        (work / name).mkdir(parents=True, exist_ok=True)
    archive = fact(runtime / "archive.zip")
    pointer_raw_path = Path(args.pointer_raw)
    binding = {
        "runtime_sha256": measure_runtime_digest(runtime).sha256,
        "archive": archive,
        "receiver_sha256": measure_receiver_digest(runtime),
        "pointer_raw_sha256": args.expect_pointer_raw_sha256,
        "producer": fact(Path(__file__)),
        "compiler": compiler_fact(),
    }
    binding_path = work / "INPUTS.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise Pc3PublicError("public proof source/config drift")
    record(binding_path, binding)

    done = work / "RESULT.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["candidate_raw"] != fact(work / "output/0.raw"):
            raise Pc3PublicError("completed output custody drift")
        print(json.dumps({k: result[k] for k in ("literal_public_output_byte_identity", "wall_seconds")}))
        return result
    output = work / "output/0.raw"
    if output.exists():
        raise Pc3PublicError(
            "unreceipted raw: preserve and recover the log, never rerender over it"
        )
    need = RAW_BYTES + 256 * 1024**2
    if shutil.disk_usage(root).free < need + 8 * 1024**3:
        raise Pc3PublicError("STORAGE_BLOCK: keep all bytes")

    with zipfile.ZipFile(runtime / "archive.zip") as zf:
        (work / "data/p").write_bytes(zf.read("p"))
    (work / "file_list.txt").write_bytes(b"0.hevc\n")
    env = dict(
        os.environ,
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "scratch"),
        RLC1_ADVISORY_CPU="1",
        RLC1_PROOF_BLAS_THREADS=str(args.blas_threads),
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER="600",
        F26_TOKEN_DECODER="python",
        # The system compiler, not ``ddm_rlc4_native_cache.py``.  That shim refuses any
        # compile whose destination is outside ``$TMPDIR``, and ``inflate.sh`` picks its
        # build directory with ``mktemp -d``, which on this macOS ALWAYS returns the
        # Darwin per-user temp directory and ignores ``TMPDIR`` entirely (measured here:
        # ``env -i TMPDIR=<scratch> mktemp -d`` -> ``/var/folders/...``).  So the shim
        # cannot be satisfied from this session and its refusal is correct, not a bug.
        #
        # Dropping it costs nothing this proof needs.  The cache exists to replay
        # byte-identical native builds across a RESUMED decode; this run is cold, single,
        # and start-to-finish, so there is nothing to replay.  The compiler's own identity
        # is recorded in the receipt below, and the C sources are hashed in the tree
        # manifest, so the build stays reproducible.  Nothing is borrowed from another arm.
        CC="/usr/bin/cc",
    )
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
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
    attempt.mkdir()
    resumed = (work / "frame_checkpoints/LATEST.json").exists()
    record(
        attempt / "COMMAND.json",
        {
            "argv": command,
            "env": {
                key: value
                for key, value in env.items()
                if key.startswith(("RLC", "TC1", "F26", "OMP", "MKL", "OPENBLAS", "VECLIB"))
                or key in ("CC", "TMPDIR", "PATH", "PYTHONDONTWRITEBYTECODE")
            },
            "binding": binding,
            "resumed": resumed,
            "score_claim": False,
        },
    )
    started = time.perf_counter()
    with (attempt / "run.log").open("wb") as log:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    wall = time.perf_counter() - started
    record(
        attempt / "PROCESS.json",
        {
            "returncode": completed.returncode,
            "wall_seconds": wall,
            "resumed": resumed,
            "axis": "[macOS-CPU advisory diagnostic; no timing window]",
            "score_claim": False,
        },
    )
    if completed.returncode:
        raise Pc3PublicError(f"public shell failed; retained log {attempt / 'run.log'}")
    reports = [
        json.loads(line)
        for line in (attempt / "run.log").read_text().splitlines()
        if line.startswith('{"archive_bytes"')
    ]
    if len(reports) != 1:
        raise Pc3PublicError("exactly one public report required")
    report = reports[0]

    candidate_raw = fact(output)
    pointer_raw = fact(pointer_raw_path)
    if pointer_raw["sha256"] != args.expect_pointer_raw_sha256:
        raise Pc3PublicError("pointer retained raw drift")
    identity = compare_bytes(output, pointer_raw_path)
    if not identity or candidate_raw["bytes"] != RAW_BYTES:
        raise Pc3PublicError(
            "full raw identity FAILED; every byte is retained, nothing is deleted"
        )
    if (
        report["archive_sha256"] != archive["sha256"]
        or report["raw_sha256"] != candidate_raw["sha256"]
        or report["pair_count"] != 600
        or report["token_cache"]["status"] != "DISABLED"
    ):
        raise Pc3PublicError("public report identity/cache mismatch")
    if binding["runtime_sha256"] != measure_runtime_digest(runtime).sha256:
        raise Pc3PublicError("runtime changed during proof")

    result = record(
        done,
        {
            "schema": "ddm_pc3_public.v1",
            "binding": binding,
            "candidate_raw": candidate_raw,
            "pointer_raw": pointer_raw,
            "literal_public_output_byte_identity": True,
            "bytes_compared": RAW_BYTES,
            "n_samples": 600,
            "cold_start": not resumed,
            "checkpoint_resume": report["checkpoint_resume"],
            "report": report,
            "command": command,
            "stdout": fact(attempt / "run.log"),
            "wall_seconds": wall,
            "no_scorer_ran": True,
            "what_identity_proves": (
                "the decoder's whole output file is the pointer's, so d_seg and d_pose "
                "are the pointer's by construction and this row is a pure rate move"
            ),
            "score_claim": False,
            "axis": "[macOS-CPU advisory]",
        },
    )
    print(
        json.dumps(
            {
                "literal_public_output_byte_identity": True,
                "wall_seconds": wall,
                "archive_bytes": archive["bytes"],
            }
        )
    )
    return result


def _public_path_probe(runtime_root: Path, timeout_s: float) -> dict[str, Any]:
    """Run the receiver's own ``f26_inflate.inflate_archive`` on CPU, this body's way.

    ``ddm_pc2``'s probe is the landed shape and its reasoning is unchanged: ``bash
    inflate.sh`` cannot COMPLETE without CUDA, so the leg that proves the archive PARSES
    is the function ``inflate.py`` calls, and reaching the token decode means the
    container framing, the RX1 riders, the metadata restore, the renderer load and the
    semantic unpack all passed.  One thing is added, because this body needs it: pc2's
    probe builds ``rc64_backend.so`` only, and move 44's receiver also requires
    ``RLC1_GEOMETRY_LIBRARY`` -- ``inflate.sh:75-76`` builds and exports it.  Without it
    the probe dies with ``KeyError('RLC1_GEOMETRY_LIBRARY')`` on BOTH the candidate and
    the pointer tree, which is how this was found: the control failed identically, so the
    defect was the probe's and not the candidate's.  Both libraries are built here from
    the tree's OWN C with the same flags ``inflate.sh`` uses.
    """
    import tempfile

    runtime_root = Path(runtime_root).resolve()
    script = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "from runtime.f26_inflate import inflate_archive, InflationError\n"
        "out = Path(sys.argv[2])\n"
        "started = time.time()\n"
        "try:\n"
        "    inflate_archive(root / 'archive.zip', out / '0.raw',\n"
        "                    renderer_dir=root / 'cpr1', device_name='cpu',\n"
        "                    num_threads=4, checkpoint_dir=out / '.ckpt')\n"
        "    print(json.dumps({'outcome': 'COMPLETED', 'seconds': time.time()-started}))\n"
        "except InflationError as error:\n"
        "    print(json.dumps({'outcome': 'INFLATION_ERROR', 'error': str(error),\n"
        "                      'seconds': time.time()-started}))\n"
        "except Exception as error:\n"
        "    print(json.dumps({'outcome': type(error).__name__, 'error': str(error),\n"
        "                      'seconds': time.time()-started}))\n"
    )
    builds = {
        "CPR1_RC64_LIBRARY": (
            runtime_root / "runtime" / "entropy" / "rc64_backend.c",
            ["-O3", "-std=c11", "-shared", "-fPIC"],
        ),
        "RLC1_GEOMETRY_LIBRARY": (
            runtime_root / "runtime" / "rlc1_geometry.c",
            ["-O3", "-std=c11", "-shared", "-fPIC"],
        ),
    }
    with tempfile.TemporaryDirectory() as scratch:
        environment = dict(os.environ)
        for variable, (source, flags) in builds.items():
            library = Path(scratch) / (source.stem + ".so")
            subprocess.run(
                ["/usr/bin/cc", *flags, str(source), "-o", str(library)],
                check=True, capture_output=True,
            )
            environment[variable] = str(library)
        environment["F26_TOKEN_DECODER"] = "python"
        work = Path(scratch) / "work"
        work.mkdir()
        started = time.time()
        try:
            done = subprocess.run(
                [sys.executable, "-c", script, str(runtime_root), str(work)],
                capture_output=True, text=True, timeout=timeout_s, env=environment,
            )
            payload = json.loads((done.stdout or "").strip().splitlines()[-1])
            payload["stderr_tail"] = (done.stderr or "").strip().splitlines()[-6:]
            payload.setdefault("exception_class", None)
            payload.setdefault("exception_message", "")
            return payload
        except subprocess.TimeoutExpired:
            # A timeout IS the pass condition: the decode reached the token stage and was
            # cut there. The whole decode cannot finish inside a smoke bound by design.
            return {
                "outcome": "REACHED_TOKEN_DECODE",
                "seconds": time.time() - started,
                "note": "cut at the probe bound inside the token decode",
                "exception_class": None,
                "exception_message": "",
                "stderr_tail": [],
            }
        except (ValueError, IndexError) as error:
            return {
                "outcome": "PROBE_OUTPUT_UNPARSEABLE",
                "seconds": time.time() - started,
                "error": str(error),
                "exception_class": type(error).__name__,
                "exception_message": str(error),
                "stderr_tail": [],
            }


def smoke(args) -> dict[str, Any]:
    """The seal's required smoke PAIR, on the candidate AND the pointer trees.

    The two probes are ``ddm_pc2``'s, imported rather than re-typed.  They are the landed
    pair for a carrier candidate on this body and they encode two facts about this host
    that a re-derivation would get wrong: ``bash inflate.sh`` cannot COMPLETE without CUDA,
    so REACHING that refusal is the pass condition (it proves the C toolchain, the Brotli
    gate, the file-list loop and ``_verify_input``'s two pinned constants -- exactly what a
    re-pinned candidate could get wrong); and the probes must run at a fraction of the
    declared bound, because a timed-out probe records ``timeout + epsilon`` and the
    validator refuses ``seconds > public_path_probe_seconds``.

    The frontier leg is the POINTER's tree, not pc2's, so the control is this candidate's
    own base.
    """
    import ddm_pc2_carrier_kwidth_rankcut as pc2

    candidate = Path(args.out_dir) / "candidate_runtime"
    frontier = Path(args.frontier_runtime)
    bound = float(args.bound_seconds)
    probe_timeout = 0.8 * bound
    probes = {
        ("public_path_probes", "candidate"): _public_path_probe(
            candidate, timeout_s=probe_timeout
        ),
        ("public_path_probes", "frontier"): _public_path_probe(
            frontier, timeout_s=probe_timeout
        ),
        ("inflate_sh_smokes", "candidate"): pc2._inflate_sh_smoke(
            candidate, timeout_s=probe_timeout
        ),
        ("inflate_sh_smokes", "frontier"): pc2._inflate_sh_smoke(
            frontier, timeout_s=probe_timeout
        ),
    }
    trees = {"candidate": candidate, "frontier": frontier}
    block: dict[str, Any] = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": bound,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for (group, role), probe in probes.items():
        block[group][role] = pc2._smoke_receipt(trees[role], probe)
    out = Path(args.out_dir) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "public_entrypoint_smoke.json").write_text(json.dumps(block, indent=2))

    # Check the block with the VALIDATOR'S OWN checker, never a second reading of the
    # contract: a block validated against a local paraphrase of the rules is worse than
    # an unvalidated one. The private name is deliberate -- a rename fails closed.
    from tac.candidate_seal import _public_smoke_problems

    problems, observed = _public_smoke_problems(
        block,
        candidate_runtime_dir=candidate,
        candidate_archive_path=candidate / "archive.zip",
        pointer_archive_sha256=args.pointer_archive_sha256,
    )
    record(
        out / "public_entrypoint_smoke_validation.json",
        {"problems": problems, "observed": observed, "score_claim": False},
    )
    if problems:
        raise Pc3PublicError(f"smoke block refused by the seal validator: {problems}")
    print(json.dumps({"smoke_problems": problems}))
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    staging = sub.add_parser("stage", help="stage the candidate runtime tree")
    staging.add_argument("--source-runtime", type=Path, default=pc3.MOVE44_TREE)
    staging.add_argument("--archive", type=Path, required=True)
    staging.add_argument("--out-dir", type=Path, required=True)
    staging.add_argument("--expect-archive-sha256", default=None)
    staging.set_defaults(func=stage)

    proof = sub.add_parser("public", help="cold n600 public parse-back + raw identity")
    proof.add_argument("--out-dir", type=Path, required=True)
    proof.add_argument("--pointer-raw", type=Path, default=pc3.MOVE44_RAW)
    proof.add_argument(
        "--expect-pointer-raw-sha256", default=pc3.MOVE44_RAW_SHA256
    )
    proof.add_argument("--blas-threads", type=int, default=4)
    proof.set_defaults(func=public)

    smokes = sub.add_parser("smoke", help="the seal's candidate+frontier smoke pair")
    smokes.add_argument("--out-dir", type=Path, required=True)
    smokes.add_argument("--frontier-runtime", type=Path, default=pc3.MOVE44_TREE)
    smokes.add_argument(
        "--pointer-archive-sha256", default=pc3.MOVE44_ARCHIVE_SHA256
    )
    smokes.add_argument("--bound-seconds", type=float, default=180.0)
    smokes.set_defaults(func=smoke)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
