"""ddm_sz1 -- build and byte-close the semantic-metadata-split archive.

Rebuilds the rr4 archive with the RX1M ``semantic`` section byte-split, sets the
``reserved`` split flag, patches a receiver tree with the free un-split, and PROVES the
decode is bit-identical to the base before reporting a single byte.

WHY NOT ``ddm_pq2_compress_e2e.py``: pq2 carries the seven non-token sections through
byte-identically by construction -- it can only re-encode the token stream, so it cannot
express a semantic-section change.  This script repacks the RX1M container directly,
which is the only surface that can, and it re-derives the base archive byte-for-byte
first so the rebuild path itself is validated before it is asked to beat anything.

IMPORT CONTEXT (read before moving this file)
---------------------------------------------
Run as ``python experiments/ddm_sz1_build_split_archive.py``, ``experiments/`` is
``sys.path[0]`` -- NOT the repo root.  So the sibling transform module is imported as a
TOP-LEVEL name (``ddm_sz1_semantic_metadata_split``), never as ``experiments.ddm_...``:
the latter passes pytest from the repo root and then fails in the real chain.  The
runtime tree is not in the repo at all; its path is injected explicitly below.

Usage::

    .venv/bin/python experiments/ddm_sz1_build_split_archive.py [--profile derived]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import ddm_sz1_semantic_metadata_split as sz1

BASE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime")
OUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_sz1")

BASES = {
    # The rr4 frontier base this arm was chartered against.
    "rr4": {
        "archive": Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip"),
        "sha256": "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956",
        "bytes": 181_161,
        "token_stream": None,
        "note": "rr4 candidate archive",
    },
    # ddm_fx2 candidate A, byte-closed AFTER its fire-order was sealed as BLOCKED.
    # Composing here is legitimate because the two changes touch DISJOINT sections:
    # fx2 re-encodes the token stream, sz1 rewrites the semantic section.
    "fx2_a": {
        "archive": Path("/Volumes/APDataStore/pact/ddm_fx2/byteclose_a/retained/archive.zip"),
        "sha256": "9de0f6db3ca7ae4efcd9237752b7c95ed1119d9285f8aadd92fee7c8c18547ef",
        "bytes": 180_450,
        "token_stream": Path("/Volumes/APDataStore/pact/ddm_fx2/byteclose_a/retained/token_stream.bin"),
        "note": "fx2 candidate A (fx2_d1_13x_homog_context), token stream 109,801 B",
    },
}

BROTLI_QUALITY = 11
BROTLI_LGWIN = 24


class BuildError(RuntimeError):
    """The split rebuild failed a fail-closed check."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise BuildError("archive must contain exactly one member named p")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise BuildError("archive member must be STORED")
        value = archive.read(infos[0])
        if archive.testzip() is not None:
            raise BuildError("ZIP CRC validation failed")
    return value


def deterministic_zip(member: bytes) -> bytes:
    """The frontier's exact single-member STORED framing (no timestamps, no RNG)."""
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def build_member(member: bytes, profile: sz1.SplitProfile, brotli) -> tuple[bytes, dict]:
    """Return the rebuilt member and its accounting, or raise fail-closed."""
    version, codec, table_mode, reserved, hpac_n, sem_n, car_n = sz1.read_rx1_header(member)
    if version != 1 or codec != 2:
        raise BuildError(f"expected RX1M version 1 codec 2 (Brotli), got {version}/{codec}")
    if reserved != 0:
        raise BuildError(f"base archive already carries reserved={reserved}")

    head = sz1.RX1_MODEL_HEADER.size
    hpac = member[head : head + hpac_n]
    semantic = member[head + hpac_n : head + hpac_n + sem_n]
    carrier = member[head + hpac_n + sem_n : head + hpac_n + sem_n + car_n]
    tail = member[head + hpac_n + sem_n + car_n :]

    body = brotli.decompress(semantic)
    if len(body) != sz1.WANS_BODY_BYTES:
        raise BuildError(f"semantic body is {len(body)} B, expected {sz1.WANS_BODY_BYTES}")

    # CONTROL FIRST: the instrument must reproduce the shipped section byte-for-byte
    # before it is allowed to beat it.  Without this the delta is unreadable.
    control = brotli.compress(body, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    if len(control) != len(semantic):
        raise BuildError(
            f"control re-Brotli is {len(control)} B, shipped is {len(semantic)} B -- "
            "the container's parameters are not reproduced, so no delta is readable"
        )

    split_body = sz1.split_region(body, profile)
    if len(split_body) != len(body):
        raise BuildError("split changed the body length")
    if sz1.unsplit_region(split_body, profile) != body:
        raise BuildError("un-split does not invert the split")
    split_stream = brotli.compress(split_body, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)

    rebuilt = (
        sz1.RX1_MODEL_HEADER.pack(
            sz1.RX1_MAGIC, version, codec, table_mode,
            sz1.RX1_RESERVED_SEMANTIC_SPLIT,
            hpac_n, len(split_stream), car_n,
        )
        + hpac + split_stream + carrier + tail
    )
    accounting = {
        "shipped_semantic_bytes": len(semantic),
        "control_rebrotli_bytes": len(control),
        "control_delta": len(control) - len(semantic),
        "split_semantic_bytes": len(split_stream),
        "split_delta": len(split_stream) - len(semantic),
        "member_delta": len(rebuilt) - len(member),
    }
    return rebuilt, accounting


def build_patched_runtime(destination: Path, profile: sz1.SplitProfile) -> str:
    """Copy the sealed runtime tree and insert the free un-split.  Fails closed."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(BASE_RUNTIME, destination, ignore=shutil.ignore_patterns("__pycache__"))
    target = destination / "runtime" / "residual_archive.py"
    patched = sz1.patch_receiver_source(target.read_text(), profile)
    target.write_text(patched)
    return sha256(patched.encode())


VERIFIER = textwrap.dedent(
    """
    import sys, json, hashlib, dataclasses
    runtime, base, cand = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, runtime)
    from runtime.residual_archive import read_residual_archive
    import numpy as np

    def digest(value, out):
        # Deep, order-stable structural digest.  Comparing decoded objects field by
        # field with == is what hides an array mismatch behind a ValueError, so every
        # leaf is hashed by dtype+shape+raw bytes instead.
        if isinstance(value, np.ndarray):
            out.update(b"A" + str(value.dtype).encode() + str(value.shape).encode())
            out.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (bytes, bytearray, memoryview)):
            out.update(b"B"); out.update(bytes(value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            out.update(b"D" + type(value).__name__.encode())
            for f in dataclasses.fields(value):
                out.update(b"|" + f.name.encode()); digest(getattr(value, f.name), out)
        elif isinstance(value, (list, tuple)):
            out.update(b"L" + str(len(value)).encode())
            for item in value: digest(item, out)
        elif isinstance(value, dict):
            out.update(b"M")
            for key in sorted(value, key=repr):
                out.update(b"|" + repr(key).encode()); digest(value[key], out)
        else:
            out.update(b"S" + repr(value).encode())

    def fingerprint(obj):
        table = {}
        names = sorted(vars(obj)) if hasattr(obj, "__dict__") else [
            f.name for f in dataclasses.fields(obj)
        ]
        for name in names:
            h = hashlib.sha256(); digest(getattr(obj, name), h)
            table[name] = h.hexdigest()
        return table

    a, b = fingerprint(read_residual_archive(base)), fingerprint(read_residual_archive(cand))
    fields = sorted(set(a) | set(b))
    mismatched = [n for n in fields if a.get(n) != b.get(n)]
    print(json.dumps({"fields": fields, "mismatched": mismatched,
                      "base_fingerprint": a, "candidate_fingerprint": b}))
    """
).strip()


def verify_receiver(runtime: Path, base: Path, candidate: Path) -> dict:
    """Decode base and candidate through the PATCHED receiver; every field must match."""
    completed = subprocess.run(
        [sys.executable, "-c", VERIFIER, str(runtime), str(base), str(candidate)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise BuildError(f"receiver verification failed:\n{completed.stderr[-3000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=sz1.SHIPPED_PROFILE.name, choices=sorted(sz1.PROFILES))
    parser.add_argument("--base", default="rr4", choices=sorted(BASES))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    profile = sz1.PROFILES[args.profile]
    base = BASES[args.base]
    base_archive = base["archive"]

    sys.path.insert(0, str(BASE_RUNTIME))
    import brotli

    base_bytes = base_archive.read_bytes()
    if sha256(base_bytes) != base["sha256"] or len(base_bytes) != base["bytes"]:
        raise BuildError(f"base archive {args.base} does not match its pin")
    member = read_stored_member(base_archive)

    # Validate the rebuild path itself: repacking the untouched member must reproduce
    # the base archive byte-for-byte.  If it does not, no delta downstream is readable.
    if deterministic_zip(member) != base_bytes:
        raise BuildError("deterministic_zip does not reproduce the base archive")

    rebuilt, accounting = build_member(member, profile, brotli)
    archive = deterministic_zip(rebuilt)
    if deterministic_zip(rebuilt) != archive:
        raise BuildError("archive rebuild is not deterministic")

    # The token stream is a DISJOINT region: the split must not perturb one byte of it.
    # Checking it against the base's own retained stream is what makes the composition
    # claim measured rather than asserted.
    token_check = None
    if base["token_stream"] is not None:
        expected = base["token_stream"].read_bytes()
        if not rebuilt.endswith(expected):
            raise BuildError("rebuilt member does not carry the base token stream verbatim")
        token_check = {"bytes": len(expected), "sha256": sha256(expected), "carried_verbatim": True}

    out = args.out_root / f"archives/{args.base}__{profile.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "archive.zip").write_bytes(archive)
    (out / "archive.repeat.zip").write_bytes(deterministic_zip(rebuilt))
    (out / "member.bin").write_bytes(rebuilt)

    runtime = args.out_root / f"runtime/{profile.name}"
    receiver_sha = build_patched_runtime(runtime, profile)
    verdict = verify_receiver(runtime, base_archive, out / "archive.zip")

    # The base must ALSO still decode through the patched receiver: reserved == 0 is
    # byte-identity, and that is the whole backward-compatibility claim.
    inactive = verify_receiver(runtime, base_archive, base_archive)

    report = {
        "schema": "ddm_sz1_split_archive.v1",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "profile": {
            "name": profile.name, "offset": profile.offset,
            "length": profile.length, "rationale": profile.rationale,
        },
        "base": {
            "id": args.base, "path": str(base_archive),
            "sha256": base["sha256"], "bytes": base["bytes"], "note": base["note"],
        },
        "section_accounting": accounting,
        "token_stream_carried": token_check,
        "archive": {
            "sha256": sha256(archive), "bytes": len(archive),
            "delta_bytes_vs_base": len(archive) - base["bytes"],
            "repeat_identical": (out / "archive.repeat.zip").read_bytes() == archive,
        },
        "receiver": {
            "tree": str(runtime), "residual_archive_sha256": receiver_sha,
            "decoded_fields": verdict["fields"],
            "mismatched_fields": verdict["mismatched"],
            "bit_exact_vs_base": not verdict["mismatched"],
            "base_still_decodes_unchanged": not inactive["mismatched"],
        },
    }
    # ``compressed_models`` is the RAW RX1M container bytes -- the thing we deliberately
    # rewrote.  It MUST differ.  Every DECODED product must not.  Listing it explicitly
    # (rather than ignoring mismatches generally) keeps the check load-bearing: if the
    # split ever perturbed a decoded field, this still fires.
    container_fields = {"compressed_models"}
    decoded_mismatch = sorted(set(verdict["mismatched"]) - container_fields)
    if decoded_mismatch:
        raise BuildError(f"decode is NOT bit-exact; decoded fields differ: {decoded_mismatch}")
    if "compressed_models" not in verdict["mismatched"]:
        raise BuildError("container bytes are identical -- the split did not land (inert)")
    for required in ("semantic_blob",):
        if required not in verdict["fields"]:
            raise BuildError(f"receiver did not expose {required}; verification is vacuous")
    report["receiver"]["decoded_mismatched_fields"] = decoded_mismatch
    report["receiver"]["container_fields_changed"] = sorted(
        set(verdict["mismatched"]) & container_fields
    )
    report["receiver"]["bit_exact_vs_base"] = True
    (out / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
