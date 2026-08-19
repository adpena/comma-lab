# SPDX-License-Identifier: MIT
"""ddm_to1 -- carry ``ddm_ma1``'s within-miss law onto the LIVE ck2 pointer body.

WHAT WAS MISSING, STRUCTURALLY.  ``ddm_ck2`` is the live pointer (S 0.156664512 @
176,525 B, contest-CUDA T4).  It is built by ``ddm_sa3_rebase_sz1``, which
re-serialises the semantic and carrier sections and then **borrows sz1's token
tail verbatim** (``build_candidate``: ``... + carrier_stream + sz1["tail"]``).
There has never been a build step that substitutes a *re-encoded* tail, so every
rate win on the token stream has been unreachable from the pointer body.  This
module is that step.

WHY THE SUBSTITUTION IS IDENTITY, NOT TRANSFER.  ``ddm_ma1`` measured a
**-104.584 B** code-length win from the within-miss relative law and projected
**-105 B** on the archive, on both the ck1 and ck2 bodies -- but could not
byte-close it (its blocker was the ``rc64_source`` pin).  ``ddm_rc1x`` cleared
that blocker with the two-role rc64 recipe and closed the law end-to-end on the
rr4/D1 lineage: archive 180,345 B, with the inflated ``0.raw`` byte-identical to
the base.  A rate-only re-encode of the same tokens.

The question this module had to answer is whether ma1's tail re-encodes *the same
object* the ck2 pointer ships.  It does, and the proof is byte equality rather
than argument (``tail_identity_gate``):

    fx2 D1 tail   109,897 B   sha 59cc27c9...
    ck2 tail      109,897 B   sha 59cc27c9...   <- EQUAL, so the same object
    ma1 tail      109,792 B   sha 4bc30d3f...   <- -105 B, the law realised

ck2's borrowed tail IS ``ddm_fx2``'s D1 tail, byte for byte, and ma1's tail is
that exact object re-encoded under the within-miss law.  So this is the
``ddm_sa3`` identity-gate pattern (asserted in code at build time, per qs5), not
the qs4 cross-lattice transfer disaster.

WHY A TAIL SPLICE IS EXACTLY A COMPOSER RE-RUN.  The RX1M member is

    HEADER(14) + hpac + semantic + carrier + tail

and the header stores ``hb``/``sb``/``cb`` only -- the tail is implicit
(``sections()``: ``outer[offset:]``).  Substituting the tail therefore changes
**no other field**, and this module asserts that: the spliced member's
header+hpac+semantic+carrier prefix must be byte-identical to ck2's.  The
sister hook ``ddm_sa3_rebase_sz1.build_candidate(tail_override=...)`` carries the
same substitution on the full composer path.

AXIS ``[macOS-CPU exact byte/container]``.  This module measures BYTES exactly
and runs no scorer.  It is not a score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_rx1_rate_representation_attack import deterministic_zip
from experiments.ddm_sa3_rebase_sz1 import MEMBER_NAME, RX1_HEADER, sections

__all__ = [
    "CK2_ARCHIVE",
    "CK2_ARCHIVE_BYTES",
    "CK2_ARCHIVE_SHA256",
    "EXPECTED_TAIL_DELTA_BYTES",
    "FX2_D1_ARCHIVE",
    "MA1_ARCHIVE",
    "RATE_DENOMINATOR",
    "To1Error",
    "build_override",
    "identity_control_off",
    "rewrite_corrector_to_relative_imports",
    "splice_tail",
    "tail_identity_gate",
]


class To1Error(RuntimeError):
    """Refusal. Every gate in this module fails closed."""


# --- pinned inputs, each verified from disk before use --------------------------------

CK2_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_ck2/generations/ck2_plane2_r1/archive.zip"
)
CK2_ARCHIVE_SHA256: Final = (
    "0aa1cada2ca79ad43a11bfa72be69a5240315e35cf5b4c94665d60d0c3583933"
)
CK2_ARCHIVE_BYTES: Final = 176_525

FX2_D1_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_fx2/byteclose_a/retained/archive.zip"
)
FX2_D1_ARCHIVE_SHA256: Final = (
    "9de0f6db3ca7ae4efcd9237752b7c95ed1119d9285f8aadd92fee7c8c18547ef"
)
FX2_D1_ARCHIVE_BYTES: Final = 180_450

MA1_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_rc1x/byteclose_ma1/retained/archive.zip"
)
MA1_ARCHIVE_SHA256: Final = (
    "a0b2bdb1cd300177563b113ae7dec3db006d76bc869f3ce115e0dee05e7bc9d1"
)
MA1_ARCHIVE_BYTES: Final = 180_345

CK2_RUNTIME: Final = CK2_ARCHIVE.parent

MA1_CORRECTOR_SOURCE: Final = REPO / "experiments/ddm_ma1_within_miss_corrector.py"

EXPECTED_TAIL_DELTA_BYTES: Final = -105
"""``ddm_ma1``'s projected archive delta.  MEASURED code-length delta was
-104.584 B; the realised container delta is asserted to be exactly this."""

RATE_DENOMINATOR: Final = 37_545_489
"""``upstream/evaluate.py``: rate term = 25 * archive_bytes / this."""

S_PER_BYTE: Final = 25.0 / RATE_DENOMINATOR
"""6.6586e-07 S per archive byte."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_member(path: Path) -> bytes:
    """Read the single stored member, refusing any other archive shape."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
            raise To1Error(f"{path}: member layout is not a single {MEMBER_NAME!r}")
        payload = archive.read(infos[0])
        if archive.testzip() is not None:
            raise To1Error(f"{path}: ZIP CRC validation failed")
    return payload


def require_archive(path: Path, *, digest: str | None, size: int) -> dict[str, Any]:
    """Refuse unless the archive on disk is exactly the pinned object."""
    if not path.is_file():
        raise To1Error(f"required archive is absent: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise To1Error(f"{path}: bytes differ ({actual_size} != {size})")
    actual = sha256_file(path)
    if digest is not None and actual != digest:
        raise To1Error(f"{path}: sha256 differs ({actual} != {digest})")
    return {"path": str(path), "sha256": actual, "bytes": actual_size}


# --------------------------------------------------------------------------
# THE TAIL IDENTITY GATE -- the sa3/qs5 pattern, applied to the token section
# --------------------------------------------------------------------------


def tail_identity_gate() -> dict[str, Any]:
    """Refuse to splice ma1's tail unless it re-encodes ck2's OWN tail object.

    Three facts must hold, and each is byte equality rather than argument:

    1. ``fx2_D1.tail == ck2.tail`` -- ck2's verbatim borrow IS fx2's D1 tail, so
       ma1's re-encode of the fx2 tail is a re-encode of the bytes ck2 ships.
       Without this the splice would be a cross-lineage transfer (qs4).
    2. ``hpac`` is byte-identical across all three.  The corrector's context
       state is seeded from the HPAC prior; a differing prior would mean ma1's
       encoder and ck2's receiver disagree about the model, and the arithmetic
       decode desyncs SILENTLY (the ck1 S=79.4 T4 refusal).
    3. The realised delta is exactly ``EXPECTED_TAIL_DELTA_BYTES``.  ma1
       PRE-REGISTERED -105 B; a different number means the object being spliced
       is not the law that was measured, whatever its name says.
    """
    ck2 = require_archive(CK2_ARCHIVE, digest=CK2_ARCHIVE_SHA256, size=CK2_ARCHIVE_BYTES)
    ma1 = require_archive(MA1_ARCHIVE, digest=MA1_ARCHIVE_SHA256, size=MA1_ARCHIVE_BYTES)
    # fx2 is pinned by sha too, not merely by the tail check below.  The tail
    # check would catch a WRONG fx2, but only after the module had already
    # claimed the file was the D1 build -- and a pin that is implied by a later
    # assertion is a pin that moves when the later assertion is edited.
    fx2 = require_archive(
        FX2_D1_ARCHIVE, digest=FX2_D1_ARCHIVE_SHA256, size=FX2_D1_ARCHIVE_BYTES
    )

    s_ck2, s_ma1, s_fx2 = (
        sections(CK2_ARCHIVE), sections(MA1_ARCHIVE), sections(FX2_D1_ARCHIVE)
    )

    failures: list[str] = []
    if s_fx2["tail"] != s_ck2["tail"]:
        failures.append(
            "ck2's borrowed tail is NOT fx2's D1 tail "
            f"({sha256_bytes(s_ck2['tail'])[:16]} vs {sha256_bytes(s_fx2['tail'])[:16]}); "
            "ma1's re-encode would be a cross-lineage transfer, not an identity"
        )
    if not (s_fx2["hpac"] == s_ck2["hpac"] == s_ma1["hpac"]):
        failures.append("the hpac prior is not byte-identical across fx2/ck2/ma1")

    delta = len(s_ma1["tail"]) - len(s_ck2["tail"])
    if delta != EXPECTED_TAIL_DELTA_BYTES:
        failures.append(
            f"realised tail delta is {delta:+d} B, not the pre-registered "
            f"{EXPECTED_TAIL_DELTA_BYTES:+d} B"
        )
    if failures:
        raise To1Error("tail identity gate REFUSED: " + "; ".join(failures))

    return {
        "verdict": "IDENTITY",
        "ck2": ck2,
        "ma1": ma1,
        "fx2_d1": fx2,
        "ck2_tail_bytes": len(s_ck2["tail"]),
        "ck2_tail_sha256": sha256_bytes(s_ck2["tail"]),
        "fx2_tail_sha256": sha256_bytes(s_fx2["tail"]),
        "ma1_tail_bytes": len(s_ma1["tail"]),
        "ma1_tail_sha256": sha256_bytes(s_ma1["tail"]),
        "tail_delta_bytes": delta,
        "hpac_sha256": sha256_bytes(s_ck2["hpac"]),
        "note": (
            "ck2's borrowed tail is byte-identical to ddm_fx2's D1 tail, and ma1's "
            "tail is that exact object re-encoded under the within-miss law. The "
            "substitution is valid by IDENTITY, not by transfer."
        ),
    }


# --------------------------------------------------------------------------
# the splice -- provably a composer re-run with tail_override
# --------------------------------------------------------------------------


def splice_tail(base_archive: Path, tail: bytes) -> bytes:
    """Rebuild ``base_archive``'s member with ``tail`` in place of its own.

    The RX1M header carries ``hb``/``sb``/``cb`` only; the tail is
    ``outer[offset:]``.  So nothing but the tail moves, and the caller asserts
    exactly that against the base member's prefix.
    """
    outer = read_member(base_archive)
    _, _, _, _, _, hb, sb, cb = RX1_HEADER.unpack_from(outer)
    offset = RX1_HEADER.size + hb + sb + cb
    if offset > len(outer):
        raise To1Error(f"{base_archive}: declared section lengths exceed the member")
    member = outer[:offset] + tail
    archive = deterministic_zip(member)

    # Re-PARSE the assembled archive rather than re-checking the slice we just
    # built.  Asserting ``member[:offset] == outer[:offset]`` would be vacuous --
    # it is true by construction and could not fail -- and a gate that cannot
    # fail is not a gate.  Parsing the round-tripped bytes instead exercises the
    # header arithmetic, the zip writer, and the reader that the receiver uses.
    rebuilt, base = read_member_sections(archive), sections(base_archive)
    for field in ("magic", "version", "codec", "table_mode", "reserved",
                  "hpac", "semantic", "carrier"):
        if rebuilt[field] != base[field]:
            raise To1Error(f"splice moved the {field} field; this is not a tail-only edit")
    if rebuilt["tail"] != tail:
        raise To1Error("the round-tripped tail is not the tail that was spliced in")
    return archive


def read_member_sections(archive_bytes: bytes) -> dict[str, Any]:
    """``sections()`` over in-memory archive bytes, so nothing needs a temp file."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as handle:
        outer = handle.read(MEMBER_NAME)
    magic, version, codec, table_mode, reserved, hb, sb, cb = RX1_HEADER.unpack_from(outer)
    offset = RX1_HEADER.size
    hpac, offset = outer[offset : offset + hb], offset + hb
    semantic, offset = outer[offset : offset + sb], offset + sb
    carrier, offset = outer[offset : offset + cb], offset + cb
    return {
        "magic": magic, "version": version, "codec": codec,
        "table_mode": table_mode, "reserved": reserved,
        "hpac": hpac, "semantic": semantic, "carrier": carrier,
        "tail": outer[offset:],
    }


def _price(archive_bytes: int) -> dict[str, Any]:
    """Price the candidate, and SAY which legs are measured and which are asserted.

    The rate leg is exact -- it is a byte count.  The distortion legs are 0.0 by
    the identity argument (same tokens in, same decode out), and that argument is
    only DISCHARGED by the decode.  Reporting them as measured before the decode
    lands would be the surrogate-is-not-authority failure, so the receipt carries
    the distinction instead of burying it.
    """
    delta_bytes = archive_bytes - CK2_ARCHIVE_BYTES
    d_rate = 25.0 * delta_bytes / RATE_DENOMINATOR
    return {
        "archive_bytes": archive_bytes,
        "delta_bytes_vs_pointer": delta_bytes,
        "rate_term": 25.0 * archive_bytes / RATE_DENOMINATOR,
        "pointer_rate_term": 25.0 * CK2_ARCHIVE_BYTES / RATE_DENOMINATOR,
        "dS_rate": d_rate,
        "dS_rate_status": "MEASURED (exact byte count)",
        "dS_seg": 0.0,
        "dS_pose": 0.0,
        "dS_distortion_status": (
            "ASSERTED from the tail identity gate; DISCHARGED only by a decode "
            "whose output is byte-identical to the pointer's"
        ),
        "net_dS": d_rate,
        "net_dS_is_pure_rate_pending_decode_identity": True,
        "admit_bar": -3.5e-06,
        "admit_bar_multiple": abs(d_rate / -3.5e-06),
    }


def identity_control_off(output: Path | None = None) -> dict[str, Any]:
    """Control (a): splicing ck2's OWN tail must reproduce the pointer exactly.

    This is not ceremony.  It proves the splice path itself introduces zero
    bytes, so the ON variant's delta is attributable to the tail alone.  A
    vacuous control -- one that could pass without exercising the splice -- is
    the failure this avoids: the path here is the SAME ``splice_tail`` the ON
    variant uses, differing only in which tail it is handed.
    """
    tail = sections(CK2_ARCHIVE)["tail"]
    rebuilt = splice_tail(CK2_ARCHIVE, tail)
    digest = sha256_bytes(rebuilt)
    matches = digest == CK2_ARCHIVE_SHA256 and len(rebuilt) == CK2_ARCHIVE_BYTES
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rebuilt)
    if not matches:
        raise To1Error(
            "identity control FAILED: splicing ck2's own tail did not reproduce the "
            f"pointer archive ({digest} @ {len(rebuilt)} B != {CK2_ARCHIVE_SHA256} @ "
            f"{CK2_ARCHIVE_BYTES} B)"
        )
    return {
        "control": "override_OFF",
        "verdict": "REPRODUCES_POINTER_BYTE_IDENTICALLY",
        "archive_sha256": digest,
        "archive_bytes": len(rebuilt),
        "expected_sha256": CK2_ARCHIVE_SHA256,
        "path": str(output) if output else None,
    }


def build_override(output: Path) -> dict[str, Any]:
    """Splice ma1's tail onto the ck2 pointer body, with a determinism repeat."""
    tail = sections(MA1_ARCHIVE)["tail"]
    archive = splice_tail(CK2_ARCHIVE, tail)
    repeat = splice_tail(CK2_ARCHIVE, tail)
    if archive != repeat:
        raise To1Error("double-compile is NOT deterministic")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(archive)

    member = read_member(output)
    base_member = read_member(CK2_ARCHIVE)
    s_new, s_ck2 = sections(output), sections(CK2_ARCHIVE)
    for field in ("hpac", "semantic", "carrier"):
        if s_new[field] != s_ck2[field]:
            raise To1Error(f"the {field} section moved; this is not a tail-only edit")
    if s_new["reserved"] != s_ck2["reserved"]:
        raise To1Error("the reserved flag moved; this is not a tail-only edit")

    return {
        "control": "override_ON",
        "archive_path": str(output),
        "archive_sha256": sha256_bytes(archive),
        "archive_bytes": len(archive),
        "member_bytes": len(member),
        "member_sha256": sha256_bytes(member),
        "base_member_bytes": len(base_member),
        "determinism_repeat_byte_identical": True,
        "determinism_repeat_sha256": sha256_bytes(repeat),
        "tail_bytes": len(s_new["tail"]),
        "tail_sha256": sha256_bytes(s_new["tail"]),
        "sections_unchanged_vs_pointer": ["hpac", "semantic", "carrier", "reserved"],
        "price": _price(len(archive)),
    }


# --------------------------------------------------------------------------
# the receiver -- ma1 is a MODEL change, so the decoder must carry it
# --------------------------------------------------------------------------

_IMPORT_REWRITES: Final = (
    (
        "from experiments.ddm_fx2_model_axis_corrector import (",
        "from .fx2_model_axis_corrector import (",
    ),
    (
        "from experiments.ddm_rr4_free_corrector_v2 import GroupState, NUM_CLASSES",
        "from .rr4_free_corrector import GroupState, NUM_CLASSES",
    ),
)

_DYNAMIC_IMPORT: Final = '''SHIPPED_CONFIG: dict = dict(
    __import__(
        "experiments.ddm_fx2_model_axis_corrector", fromlist=["SHIPPED_CONFIG"]
    ).SHIPPED_CONFIG
)'''

_DYNAMIC_REPLACEMENT: Final = '''from .fx2_model_axis_corrector import SHIPPED_CONFIG as _FX2_SHIPPED_CONFIG

SHIPPED_CONFIG: dict = dict(_FX2_SHIPPED_CONFIG)'''


def rewrite_corrector_to_relative_imports(source: str) -> str:
    """Stage ``ddm_ma1`` as receiver code with NO repo-absolute imports.

    ``ddm_rc1x`` staged this module verbatim, leaving ``from experiments....``
    in the shipped tree; its parse-back passed only because the driver put the
    repo on ``PYTHONPATH``.  That tree is not self-contained and would fail at
    contest decode.  This rewrite fails closed on every pattern and then
    re-parses, so a silently-missed rewrite cannot ship.
    """
    out = source
    for needle, replacement in _IMPORT_REWRITES:
        if needle not in out:
            raise To1Error(f"expected import not found in ma1 source: {needle!r}")
        out = out.replace(needle, replacement)
    if _DYNAMIC_IMPORT not in out:
        raise To1Error("ma1's dynamic SHIPPED_CONFIG import was not found verbatim")
    out = out.replace(_DYNAMIC_IMPORT, _DYNAMIC_REPLACEMENT)

    if "experiments." in out:
        leaked = [
            line.strip()
            for line in out.splitlines()
            if "experiments." in line and not line.lstrip().startswith(("#", "*"))
        ]
        if leaked:
            raise To1Error(f"repo-absolute import survived the rewrite: {leaked[:3]}")
    ast.parse(out)  # a tree that does not parse is a tree that cannot decode
    return out


def stage_receiver(dest: Path) -> dict[str, Any]:
    """Copy the ck2 generation tree and swap fx2 -> ma1 as ``free_corrector``.

    ``runtime/residual_archive.py`` does ``from .free_corrector import
    FreeCorrector``, and ma1 exports exactly that drop-in, so the swap is a
    rename plus one new module.  fx2 is KEPT (as ``fx2_model_axis_corrector``)
    because ma1 subclasses it and inherits its frozen ``SHIPPED_CONFIG``.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("cpr1", "runtime"):
        shutil.copytree(
            CK2_RUNTIME / name, dest / name,
            ignore=shutil.ignore_patterns("._*", "__pycache__", "*.pyc"),
            copy_function=shutil.copy,
        )
    for name in ("inflate.py", "inflate.sh"):
        shutil.copy(CK2_RUNTIME / name, dest / name)

    runtime = dest / "runtime"
    fx2_live = runtime / "free_corrector.py"
    fx2_kept = runtime / "fx2_model_axis_corrector.py"
    if not fx2_live.is_file():
        raise To1Error("ck2's runtime has no free_corrector.py to preserve")
    fx2_sha = sha256_file(fx2_live)
    shutil.move(str(fx2_live), str(fx2_kept))

    staged = rewrite_corrector_to_relative_imports(MA1_CORRECTOR_SOURCE.read_text())
    fx2_live.write_text(staged)

    return {
        "generation": str(dest),
        "fx2_preserved_as": "runtime/fx2_model_axis_corrector.py",
        "fx2_sha256": fx2_sha,
        "ma1_source": str(MA1_CORRECTOR_SOURCE),
        "ma1_source_sha256": sha256_file(MA1_CORRECTOR_SOURCE),
        "ma1_staged_as": "runtime/free_corrector.py",
        "ma1_staged_sha256": sha256_bytes(staged.encode()),
        "ma1_staged_bytes": len(staged.encode()),
    }


def pin_generation_archive(dest: Path, archive: Path) -> dict[str, Any]:
    """Copy the archive in and re-pin ``inflate.py`` BY PATTERN, failing closed."""
    shutil.copy(archive, dest / "archive.zip")
    digest = sha256_file(dest / "archive.zip")
    size = (dest / "archive.zip").stat().st_size

    inflate = dest / "inflate.py"
    source = inflate.read_text()
    replaced = re.sub(
        r'(ARCHIVE_SHA256\s*=\s*")[0-9a-f]{64}(")', rf"\g<1>{digest}\g<2>", source
    )
    replaced = re.sub(r"(ARCHIVE_BYTES\s*=\s*)[0-9_]+", rf"\g<1>{size:_}", replaced)
    if digest not in replaced or f"{size:_}" not in replaced:
        raise To1Error("inflate.py archive pin did not move; the tree would self-refuse")
    inflate.write_text(replaced)

    for residue in list(dest.rglob("._*")) + list(dest.rglob("__pycache__")):
        if residue.is_dir():
            shutil.rmtree(residue)
        else:
            residue.unlink()
    unparseable = [
        str(p.relative_to(dest))
        for p in sorted(dest.rglob("*.py"))
        if not _parses(p)
    ]
    if unparseable:
        raise To1Error(f"staged tree has unparseable python: {unparseable}")
    return {"archive_sha256": digest, "archive_bytes": size, "inflate_pin_updated": True}


def _parses(path: Path) -> bool:
    try:
        ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path, default=Path("/Volumes/APDataStore/pact/ddm_to1")
    )
    parser.add_argument("--no-stage", action="store_true")
    args = parser.parse_args(argv)

    compile_dir = args.store / "compile"
    gate = tail_identity_gate()
    control = identity_control_off(compile_dir / "control_override_off.zip")
    candidate = build_override(compile_dir / "candidate_tail_override.zip")

    receipt: dict[str, Any] = {
        "arm": "ddm_to1_tail_override_20260819",
        "axis": "[macOS-CPU exact byte/container]",
        "score_claim": False,
        "promotable": False,
        "pointer_at_build": {
            "archive_sha256": CK2_ARCHIVE_SHA256,
            "archive_bytes": CK2_ARCHIVE_BYTES,
            "score": 0.1566645120483069,
            "axis": "contest_cuda",
        },
        "tail_identity_gate": gate,
        "identity_control_off": control,
        "candidate": candidate,
        "s_per_byte": S_PER_BYTE,
    }
    if not args.no_stage:
        generation = args.store / "generations/to1_tail_override_r1"
        staged = stage_receiver(generation)
        staged.update(
            pin_generation_archive(generation, Path(candidate["archive_path"]))
        )
        receipt["staged_generation"] = staged

    receipt["retention"] = write_retention_manifest(args.store)

    out = compile_dir / "TO1_TAIL_OVERRIDE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def write_retention_manifest(store: Path) -> dict[str, Any]:
    """Record sha256 + bytes for EVERY retained payload, winners and controls alike.

    ALWAYS KEEP THE PAYLOAD is a precondition for running, not a post-hoc tidy.
    Both archives are kept -- the control as well as the candidate -- because the
    control is what makes the candidate's -105 B attributable to the tail.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(store.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.name == "TO1_RETENTION_MANIFEST.json":
            continue
        rows.append({
            "path": str(path.relative_to(store)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "arm": "ddm_to1_tail_override_20260819",
        "store": str(store),
        "files": rows,
        "total_files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    (store / "TO1_RETENTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return {"total_files": len(rows), "total_bytes": manifest["total_bytes"]}


if __name__ == "__main__":
    raise SystemExit(main())
