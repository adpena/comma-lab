"""ddm_sa3 -- re-base the sa2 pose-compensated semantic edit onto the sz1 pointer.

WHY THIS ARM EXISTS
-------------------
``ddm_sa2`` proved a real mechanism: in-compile frame-0 Schur compensation cancels
99.942% of the S2 semantic edit's pose damage for +36 B.  But it built its candidate
on **rr4** (S 0.158533 @ 181,161 B) while the live pointer moved to **sz1**
(S 0.15771357797660338 @ 179,930 B).  sa2's projected 0.157896 is WORSE than the live
pointer, so the candidate is obsolete even though the mechanism is not.  That is the
delta-without-its-baseline law firing on our own arm.

TRANSFER vs IDENTITY -- the distinction this module is built around
-------------------------------------------------------------------
Carrying a Schur compensation across a *different* lattice is the qs4 disaster
(+2.396e-4 pose, ~100x the seg win).  So this module does **not** assume transfer.
It MEASURES whether sz1's decoded state equals rr4's, and refuses to build unless
it does.  Measured result (``assert_decoded_identity``):

    carrier lattice  max|delta| = 0        semantic body  byte-identical
    carrier body     byte-identical        hpac body      byte-identical

sz1's two moves (fx2 token re-encode, semantic byte-plane split) are both **container**
transforms over an unchanged decoded state.  So the sa2 compensation is valid here by
IDENTITY, not by transfer -- and the identity is asserted in code at build time, per
the qs5 pattern, not argued in prose.

WHAT IS ACTUALLY REBUILT
------------------------
The rate side is rebuilt from scratch, because rate does NOT carry:

* ``tail``   -- sz1's fx2 token stream (109,897 B vs rr4's 110,608).  MEASURED here to
  be byte-identical across rr4/S2/sa2, i.e. independent of both the semantic edit and
  the carrier compensation, so it is borrowable verbatim.  -711 B.
* ``semantic`` -- the S2-edited body, optionally byte-plane split.  The split's credit
  on the *base* body was -520 B; on the *edited* body it must be re-measured, because
  the S2 edit changes the body length (36,040 -> 38,514 B) and two credits over the
  same redundancy do not add (the union-is-not-the-sum-of-its-legs law).
* ``carrier`` -- the compensated lattice, re-encoded CPR1 -> CAP1 -> packed metadata.

AXIS ``[macOS-CPU exact byte/container + receiver parse-back]``.  This module measures
BYTES exactly.  It is not a score and it runs no scorer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_sa2_compile_candidate import (
    _imports,
    brotli_stream,
    carrier_surface,
    encode_carrier_body,
    patch_runtime_tree,
)

# --- the three pinned inputs, each verified from disk before use ----------------------

SZ1_GENERATION: Final = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet"
    "/generations/gen3_sz1_composed_split"
)
SZ1_ARCHIVE_SHA256: Final = (
    "debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a"
)
SZ1_ARCHIVE_BYTES: Final = 179_930

RR4_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime"
)
SA1: Final = Path("/Volumes/APDataStore/pact/ddm_sa1")
S2_ARCHIVE: Final = SA1 / "generations/S2_film23_q2_top3_q3/archive.zip"
S2_ARCHIVE_SHA256: Final = (
    "a36890b6541cf259b3f662996f8c3a935d0648aa977d02d30992aaa1e4feae29"
)

# Every sa1 row is the SAME container with a different semantic section, so one builder
# byte-closes all of them.  Each is pinned by sha: an edited archive is the object the
# compensation was solved against, and a silent substitution is the qs4 failure again.
EDITED_ROWS: Final = {
    "S2_film23_q2_top3_q3": {
        "archive": S2_ARCHIVE,
        "sha256": S2_ARCHIVE_SHA256,
        "solve_root": SA1 / "retained/sa2/n600",
        "d_seg_delta_cpu": 1.72e-06,
        "d_pose_uncompensated_cpu": 9.865438e-04,
    },
    "sm3r_keep01": {
        "archive": SA1 / "generations/sm3r_keep01/archive.zip",
        "sha256": "67422cf0d49b7af0e241780f5935b4d735f565f13fa365bc36ac9c0f0ff95953",
        "solve_root": Path(
            "/Volumes/APDataStore/pact/ddm_sa3/keep01_authority/sm3r_keep01"
        ),
        "d_seg_delta_cpu": 4.77e-06,
        "d_pose_uncompensated_cpu": 3.87399e-03,
    },
    # ck1: keep01's FiLM row-prune composed with S2's surviving legs (frame_embed q3 +
    # blocks.0.film q3) via SM3R mode 6.  LINEAGE LESSON (the S=79.4 T4 refusal,
    # fc-01M0BGYV): staging this row from the ck1 GENERATION tree paired sz1's
    # fx1/rr4-encoded token bytes with ck1's PRE-fx1 free_corrector — arithmetic
    # decode desyncs SILENTLY into garbage (parse-back is structure-only and cannot
    # see it).  The correct composition is sz1's runtime (matching token chain) with
    # ck1's RECEIVER overlaid (the only file that knows SM3R mode 6).
    "ck1_composed": {
        "archive": Path(
            "/Volumes/APDataStore/pact/ddm_ck1/generations/ck1_composed/archive.zip"
        ),
        "sha256": "71026ec4df1918f6c3bb1ff48bc8e7ad03006cd8af15941638a2854de89a7ffa",
        "solve_root": Path("/Volumes/APDataStore/pact/ddm_ck1/authority/ck1_composed"),
        # advisory n600: ck1 d_seg 0.00043336 - base 0.00042714 (attempt_0002 receipts)
        "d_seg_delta_cpu": 6.22e-06,
        # advisory n600 uncompensated d_pose (== AUTHORITY.json subset.mean_uncompensated)
        "d_pose_uncompensated_cpu": 3.99327e-03,
        "runtime_overlays": (
            (
                Path("/Volumes/APDataStore/pact/ddm_ck1/generations/ck1_composed"),
                "cpr1/ddm_mp2_semantic_receiver.py",
            ),
        ),
        # sz1's PINNED split constants (offset 49, length 8284) were fitted to sz1's
        # body layout; the mode-6 row-prune moves that layout and they measure +59 B --
        # a LOSS, and the cross-regime-constant-transfer law firing on our own tool.
        "supports_semantic_split": False,
    },
    # ck2: the ELEVENTH move.  Same sections, same decoded state, same compensated
    # lattice as ck1_composed -- only the container serialization changes, so d_seg and
    # d_pose are unchanged BY CONSTRUCTION and the whole delta is exact rate.  Requires
    # the ck2 receiver overlay, which is what teaches reserved bits 0x02/0x04.
    "ck2_plane2": {
        "archive": Path(
            "/Volumes/APDataStore/pact/ddm_ck1/generations/ck1_composed/archive.zip"
        ),
        "sha256": "71026ec4df1918f6c3bb1ff48bc8e7ad03006cd8af15941638a2854de89a7ffa",
        "solve_root": Path("/Volumes/APDataStore/pact/ddm_ck1/authority/ck1_composed"),
        "d_seg_delta_cpu": 6.22e-06,
        "d_pose_uncompensated_cpu": 3.99327e-03,
        "runtime_overlays": (
            (
                Path("/Volumes/APDataStore/pact/ddm_ck1/generations/ck1_composed"),
                "cpr1/ddm_mp2_semantic_receiver.py",
            ),
            (
                Path("/Volumes/APDataStore/pact/ddm_ck2/overlay"),
                "runtime/residual_archive.py",
            ),
        ),
        "supports_semantic_split": False,
        # Identity first (it is the ck1 pointer archive and the control every credit is
        # read against), then the two parameter-free container transforms.
        "container_variants": (
            ("none", False),
            ("plane2", False),
            ("plane2", True),
        ),
    },
}
SA2_SOLVE_ROOT: Final = SA1 / "retained/sa2/n600"

# --- the live pointer's measured components ------------------------------------------
# Source: gen3_sz1_composed_split/{archive_manifest.json, report.txt}
# [contest-CUDA T4, n600].  Recomputed from components, never the 2 dp display.
SZ1_SCORE: Final = 0.15771357797660338
SZ1_SEG_S: Final = 0.029611
SZ1_POSE_S: Final = 0.008294576541331089
SZ1_RATE_S: Final = 0.11980800143527229
SZ1_D_SEG: Final = SZ1_SEG_S / 100.0
SZ1_D_POSE: Final = (SZ1_POSE_S**2) / 10.0

PAIR_COUNT: Final = 600
RATE_DENOMINATOR: Final = 37_545_489
ADMIT_BAR_S: Final = -3.5e-6

# --- sa2's measured same-instrument pose legs (frozen CPU-torch PoseNet, n600) --------
# These are NOT re-measured here; they are valid by the decoded-state identity this
# module asserts.  Their transfer to T4 is UNMEASURED -- see the memo.
SA2_D_POSE_BASE_CPU: Final = 1.474661e-04
SA2_D_POSE_COMPENSATED_CPU: Final = 1.479554e-04
SA2_D_POSE_UNCOMPENSATED_CPU: Final = 9.865438e-04
SA2_D_SEG_DELTA_CPU: Final = 1.72e-06  # S2's seg cost, in d_seg units
SA2_ADVISORY_D_SEG_BASE_CPU: Final = 4.2714e-04

# --- the shipped sz1 semantic split, used at its EXACT shipped constants --------------
SZ1_RESERVED_SEMANTIC_SPLIT: Final = 0x01
SZ1_SPLIT_OFFSET: Final = 49
SZ1_SPLIT_LENGTH: Final = 8_284

# --- ck2's parameter-free whole-section split (DDM_CK2_WHOLE_SECTION_PLANE2_V1) -------
# Only a runtime carrying the ck2 receiver overlay decodes these bits; rows without that
# overlay must not list plane2 variants, and the header's unknown-bit guard refuses them
# fail-closed if they ever do.
CK2_RESERVED_SEMANTIC_PLANE2: Final = 0x02
CK2_RESERVED_CARRIER_PLANE2: Final = 0x04

RX1_HEADER: Final = struct.Struct("<4sBBBBHHH")
MEMBER_NAME: Final = "p"
AXIS: Final = "[macOS-CPU exact byte/container + receiver parse-back]"


class SA3Error(RuntimeError):
    """A re-base precondition, identity gate, or parse-back control failed."""


# --------------------------------------------------------------------------
# input verification -- nothing is used before it is verified from disk
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sz1_base() -> dict[str, Any]:
    """Refuse unless the live pointer's archive is exactly what the charter names."""
    archive = SZ1_GENERATION / "archive.zip"
    if not archive.exists():
        raise SA3Error(f"sz1 archive missing: {archive}")
    digest, size = sha256_file(archive), archive.stat().st_size
    if digest != SZ1_ARCHIVE_SHA256:
        raise SA3Error(f"sz1 archive sha differs: {digest} != {SZ1_ARCHIVE_SHA256}")
    if size != SZ1_ARCHIVE_BYTES:
        raise SA3Error(f"sz1 archive bytes differ: {size} != {SZ1_ARCHIVE_BYTES}")
    members = zipfile.ZipFile(archive).infolist()
    if len(members) != 1 or members[0].filename != MEMBER_NAME:
        raise SA3Error("sz1 archive member layout differs")
    recomputed = SZ1_SEG_S + SZ1_POSE_S + SZ1_RATE_S
    if abs(recomputed - SZ1_SCORE) > 1e-12:
        raise SA3Error(f"sz1 components do not recompute the score: {recomputed}")
    return {
        "archive": str(archive),
        "sha256": digest,
        "bytes": size,
        "member": members[0].filename,
        "member_bytes": members[0].file_size,
        "score_recomputed_from_components": recomputed,
        "seg_s": SZ1_SEG_S,
        "pose_s": SZ1_POSE_S,
        "rate_s": SZ1_RATE_S,
        "d_seg": SZ1_D_SEG,
        "d_pose": SZ1_D_POSE,
    }


def sections(archive: Path) -> dict[str, Any]:
    """Split an RX1M member into its four sections plus the reserved flag byte."""
    outer = zipfile.ZipFile(archive).read(MEMBER_NAME)
    magic, version, codec, table_mode, reserved, hb, sb, cb = RX1_HEADER.unpack_from(outer)
    offset = RX1_HEADER.size
    hpac, offset = outer[offset : offset + hb], offset + hb
    semantic, offset = outer[offset : offset + sb], offset + sb
    carrier, offset = outer[offset : offset + cb], offset + cb
    return {
        "magic": magic, "version": version, "codec": codec,
        "table_mode": table_mode, "reserved": reserved,
        "hpac": hpac, "semantic": semantic, "carrier": carrier,
        "tail": outer[offset:], "archive_bytes": archive.stat().st_size,
    }


# --------------------------------------------------------------------------
# THE IDENTITY GATE -- the qs4 lesson, in code
# --------------------------------------------------------------------------


_DECODE_PROBE: Final = r'''
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root, repo = sys.argv[1], sys.argv[2]
book = "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src"
sys.path.insert(0, repo); sys.path.insert(0, book)
sys.path.insert(0, root); sys.path.insert(0, root + "/cpr1")
import carrier_codec as cc
import runtime.residual_archive as ra
from runtime.carrier_repack import materialize_cpr1
archive = Path(root) / "archive.zip"
parts = ra.read_residual_archive(archive)
canonical = materialize_cpr1(parts.carrier_blob, SimpleNamespace(N=600, CARRIER_DIM=12))
_bs, _bc, _cs, enc = cc.decode_compact_carrier(
    canonical, basis_count=12 * 3 * 24 * 32, frames=600, dimensions=12)
d = (enc.astype(np.int64) >> 1) ^ -(enc.astype(np.int64) & 1)
codes = np.cumsum(d, axis=0) & 0xFFF
codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
print(json.dumps({
    "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
    "semantic_body_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
    "canonical_cpr1_sha256": hashlib.sha256(bytes(canonical)).hexdigest(),
}))
'''


def _decode_through_own_runtime(root: Path) -> dict[str, str]:
    """Decode an archive through the runtime tree that SHIPS with it.

    Each root gets a fresh interpreter: rr4's ``residual_archive`` refuses
    ``reserved != 0`` and so cannot read sz1 at all, and a shared interpreter
    would silently serve one root's modules to the other.
    """
    done = subprocess.run(
        [sys.executable, "-B", "-c", _DECODE_PROBE, str(root), str(REPO)],
        capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    if done.returncode != 0:
        raise SA3Error(f"decode probe failed under {root}: {done.stderr.strip()[-500:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def assert_decoded_identity(work: Path) -> dict[str, Any]:
    """Refuse to build unless sz1's decoded state EQUALS the one sa2 solved against.

    The compensation is a function of (base carrier lattice, base frame_1, edited
    frame_1).  If sz1 decodes the same lattice and the same semantic body as rr4,
    those three objects are identical and the solution is valid by identity.  If any
    of them differs, this is a cross-regime transfer and it must be re-solved from
    scratch -- so this refuses rather than proceeding.
    """
    rr4 = _decode_through_own_runtime(scratch_runtime_copy(RR4_RUNTIME, work, "rr4"))
    sz1 = _decode_through_own_runtime(scratch_runtime_copy(SZ1_GENERATION, work, "sz1"))

    lattice_delta = 0 if rr4["codes_sha256"] == sz1["codes_sha256"] else -1
    rr4_semantic, sz1_semantic = rr4["semantic_body_sha256"], sz1["semantic_body_sha256"]
    rr4_carrier, sz1_carrier = rr4["canonical_cpr1_sha256"], sz1["canonical_cpr1_sha256"]

    failures: list[str] = []
    if lattice_delta != 0:
        failures.append(
            f"carrier lattice differs ({rr4['codes_sha256']} vs {sz1['codes_sha256']})"
        )
    if rr4_semantic != sz1_semantic:
        failures.append(f"semantic body differs ({rr4_semantic} vs {sz1_semantic})")
    if rr4_carrier != sz1_carrier:
        failures.append("canonical CPR1 carrier body differs")
    if failures:
        raise SA3Error(
            "sz1 is NOT decode-identical to the object sa2 solved against; a "
            "compensation may not be carried across a different lattice (the qs4 "
            "disaster). Re-solve in-compile instead. Failures: " + "; ".join(failures)
        )
    return {
        "verdict": "IDENTITY",
        "carrier_lattice_sha256": sz1["codes_sha256"],
        "semantic_body_sha256": sz1_semantic,
        "canonical_cpr1_sha256": sz1_carrier,
        "note": (
            "sz1's decoded state is byte-identical to rr4's; the sa2 compensation "
            "is valid here by identity, not by transfer."
        ),
    }


def assert_tail_independence() -> dict[str, Any]:
    """Refuse to borrow sz1's fx2 tail unless the tail is provably edit-independent."""
    rr4 = sections(RR4_RUNTIME / "archive.zip")
    s2 = sections(S2_ARCHIVE)
    if rr4["tail"] != s2["tail"]:
        raise SA3Error(
            "the token tail is NOT independent of the S2 semantic edit; sz1's fx2 "
            "tail cannot be borrowed and fx2 must be re-run on S2's tokens"
        )
    if rr4["hpac"] != s2["hpac"]:
        raise SA3Error("the hpac section is not independent of the S2 semantic edit")
    sz1 = sections(SZ1_GENERATION / "archive.zip")
    return {
        "verdict": "INDEPENDENT",
        "rr4_tail_bytes": len(rr4["tail"]),
        "s2_tail_bytes": len(s2["tail"]),
        "sz1_tail_bytes": len(sz1["tail"]),
        "tail_delta_bytes": len(sz1["tail"]) - len(rr4["tail"]),
        "tail_sha256_rr4": hashlib.sha256(rr4["tail"]).hexdigest(),
        "tail_sha256_sz1": hashlib.sha256(sz1["tail"]).hexdigest(),
    }


# --------------------------------------------------------------------------
# THE PACKET IS A CUSTODY ARTIFACT, NOT A WORKSPACE
# --------------------------------------------------------------------------

_SCRATCH_ROOTS: dict[Path, Path] = {}


_COPY_KWARGS: Final = {"copy_function": shutil.copy}
"""Content-only copy.  ``copy2``/``copystat`` carries xattrs, and on the ExFAT SSD
macOS materializes those as AppleDouble ``._*`` companion files -- including ``._*.py``,
which ``contest_auth_eval.py`` then AST-parses and dies on (UnicodeDecodeError at the
AppleDouble magic).  Metadata is worthless here; the bytes are the artifact."""

_RESIDUE_GLOBS: Final = ("._*", "__pycache__", "*.pyc")


def strip_import_and_appledouble_residue(root: Path) -> list[str]:
    """Remove every residue class from a tree we own, and return what was removed."""
    removed: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.name.startswith("._") or path.name == "__pycache__" or path.suffix == ".pyc":
            removed.append(str(path.relative_to(root)))
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    return removed


def scratch_runtime_copy(root: Path, work: Path, label: str) -> Path:
    """Copy a runtime tree into scratch so nothing ever imports IN the packet.

    An in-place ``import`` of the packet's receiver writes CPython bytecode (and, on
    this filesystem, AppleDouble twins) INSIDE the hashed runtime tree.  That breaks
    the pinned runtime-tree sha cited by report.txt/README/receipt r5, and the .pyc
    embed absolute local paths -- a public-hygiene violation in a submission artifact.

    This arm CAUSED exactly that at 2026-08-18T13:16:25Z: an ad-hoc identity probe
    imported the sz1 receiver in place and wrote 15 .pyc plus 18 AppleDouble twins.
    MAIN removed them (manifest ``gen3_receipts/PYCACHE_CONTAMINATION_REMOVED_20260818.json``,
    33 files, removed 13:27:21Z) and proved restoration.  ``PYTHONDONTWRITEBYTECODE``
    alone would have prevented it, but a copy removes the whole class: the packet is
    never on any interpreter's import path at all.
    """
    cached = _SCRATCH_ROOTS.get(root)
    if cached is not None and cached.exists():
        return cached
    dest = work / f"runtime_{label}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("cpr1", "runtime"):
        shutil.copytree(
            root / name, dest / name,
            ignore=shutil.ignore_patterns(*_RESIDUE_GLOBS), **_COPY_KWARGS,
        )
    for name in ("archive.zip", "inflate.py", "inflate.sh"):
        if (root / name).is_file():
            shutil.copy(root / name, dest / name)
    strip_import_and_appledouble_residue(dest)
    _SCRATCH_ROOTS[root] = dest
    return dest


def assert_tree_pristine(root: Path) -> dict[str, Any]:
    """Report every residue class in a tree, with its denominator.

    All THREE classes, always.  A predicate that covers only ``.pyc`` and
    ``__pycache__`` reports a clean tree that still crashes ``contest_auth_eval.py``
    on an AppleDouble ``._*.py`` -- which is exactly how this arm burned a launch.
    """
    found = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.name.startswith("._") or p.name == "__pycache__" or p.suffix == ".pyc"
    )
    # The residue names are a proxy; THIS is the property contest_auth_eval.py needs --
    # it AST-parses every *.py in the runtime tree and dies on the first non-UTF-8 byte.
    unparseable: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(path.read_text())
        except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
            unparseable.append(f"{path.relative_to(root)}: {type(exc).__name__}")
    return {
        "root": str(root),
        "residue_count": len(found),
        "residue": found,
        "py_files": sum(1 for _ in root.rglob("*.py")),
        "unparseable_py": unparseable,
        "total_files": sum(1 for p in root.rglob("*") if p.is_file()),
        "pristine": not found,
    }


# --------------------------------------------------------------------------
# the semantic byte-plane split, at sz1's EXACT shipped constants
# --------------------------------------------------------------------------


def split_region(body: bytes) -> bytes:
    """Group the pinned region's byte planes (high plane, then low plane).

    Uses sz1's SHIPPED constants unchanged, so the shipped receiver un-splits this
    with zero receiver edits and zero new fitted parameters.  The transform is a pure
    byte permutation, so it restores byte-exactly whatever the region contains --
    correctness does not depend on the region being aligned to the metadata.  Only
    the *profit* does, and the profit is measured, never assumed.
    """
    start, end = SZ1_SPLIT_OFFSET, SZ1_SPLIT_OFFSET + SZ1_SPLIT_LENGTH
    if len(body) < end:
        raise SA3Error("semantic body too short for the pinned split region")
    region = np.frombuffer(body[start:end], dtype=np.uint8)
    return body[:start] + region[1::2].tobytes() + region[0::2].tobytes() + body[end:]


def unsplit_region(body: bytes) -> bytes:
    """Exact inverse of :func:`split_region` -- mirrors the shipped receiver."""
    start, end = SZ1_SPLIT_OFFSET, SZ1_SPLIT_OFFSET + SZ1_SPLIT_LENGTH
    planes = np.frombuffer(body[start:end], dtype=np.uint8)
    half = SZ1_SPLIT_LENGTH // 2
    region = np.empty(SZ1_SPLIT_LENGTH, dtype=np.uint8)
    region[1::2] = planes[:half]
    region[0::2] = planes[half:]
    return body[:start] + region.tobytes() + body[end:]


# Container variants are labelled by these suffixes.  ``none``/``pinned`` keep their
# PRE-ck2 spellings on purpose: SA3_REBASE.json is read by
# ``experiments/ddm_sa3_adjudicate_advisory.py`` and cited by the packet's borrowed-
# substrate accounting, both by the literal key ``candidate_nosplit``.  Renaming the
# label would have been a silent dangling-callsite break.
_VARIANT_LABEL: Final = {"none": "nosplit", "pinned": "split", "plane2": "plane2"}


def _variant_label(mode: str, carrier_plane2: bool) -> str:
    return f"{_VARIANT_LABEL[mode]}{'_carrier2' if carrier_plane2 else ''}"


def plane2(body: bytes) -> bytes:
    """Whole-section 2-plane de-interleave: even plane, odd plane, odd tail untouched.

    DDM_CK2_WHOLE_SECTION_PLANE2_V1.  Unlike :func:`split_region` this fits NOTHING --
    there is no offset and no length to carry across a body-layout change, which is
    exactly why it survives ck1's row-prune where sz1's pinned constants do not.

    The receiver's inverse is a GENERATED text artifact (it ships inside the archive
    runtime and cannot import this module), so this pair is deliberately duplicated in
    ``experiments/ddm_ck2_build_receiver_overlay.py``.  The two copies are not trusted
    to agree by inspection: parse-back decodes the built archive through the generated
    receiver and compares the recovered codes and semantic sha, so a divergence fails
    the build rather than shipping.
    """
    span = len(body) & ~1
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    return planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]


def unplane2(body: bytes) -> bytes:
    """Exact inverse of :func:`plane2` -- mirrors the ck2 receiver overlay."""
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]


def semantic_stream_for(body: bytes, *, mode: str) -> tuple[bytes, int]:
    """Return (brotli stream, reserved bits) for a semantic body under one mode.

    ``none``   -- byte identity, reserved 0 (what a pre-sz1 receiver reads).
    ``pinned`` -- sz1's fitted-region split at its EXACT shipped constants.
    ``plane2`` -- ck2's parameter-free whole-section split.

    Every non-identity mode proves its own inverse here before the stream is used, so a
    permutation bug cannot reach the archive.
    """
    if mode == "none":
        return brotli_stream(body), 0
    if mode == "pinned":
        permuted = split_region(body)
        if unsplit_region(permuted) != body:
            raise SA3Error("semantic split is not exactly invertible")
        return brotli_stream(permuted), SZ1_RESERVED_SEMANTIC_SPLIT
    if mode == "plane2":
        permuted = plane2(body)
        if unplane2(permuted) != body:
            raise SA3Error("semantic plane2 split is not exactly invertible")
        return brotli_stream(permuted), CK2_RESERVED_SEMANTIC_PLANE2
    raise SA3Error(f"unknown semantic container mode: {mode!r}")


def carrier_stream_for(body: bytes, *, plane2_split: bool) -> tuple[bytes, int]:
    """Return (brotli stream, reserved bits) for a carrier body.

    The permutation happens BEFORE brotli and is undone BEFORE the packed-CAP1 framing
    arithmetic runs, so the decoded lattice is bit-identical and the compensated pose
    solution is untouched by construction.
    """
    if not plane2_split:
        return brotli_stream(body), 0
    permuted = plane2(body)
    if unplane2(permuted) != body:
        raise SA3Error("carrier plane2 split is not exactly invertible")
    return brotli_stream(permuted), CK2_RESERVED_CARRIER_PLANE2


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


_VERIFIER: Final = r'''
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root, archive, expected = sys.argv[1], Path(sys.argv[2]), np.load(sys.argv[3])
sys.path.insert(0, root); sys.path.insert(0, root + "/cpr1")
import carrier_codec as cc
import runtime.residual_archive as ra
from runtime.carrier_repack import materialize_cpr1
parts = ra.read_residual_archive(archive)
canonical = materialize_cpr1(parts.carrier_blob, SimpleNamespace(N=600, CARRIER_DIM=12))
_bs, _bc, _cs, enc = cc.decode_compact_carrier(
    canonical, basis_count=12 * 3 * 24 * 32, frames=600, dimensions=12)
d = (enc.astype(np.int64) >> 1) ^ -(enc.astype(np.int64) & 1)
codes = np.cumsum(d, axis=0) & 0xFFF
codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
print(json.dumps({
    "codes_match": bool(np.array_equal(codes, expected)),
    "max_abs_code_deviation": int(np.max(np.abs(codes - expected))),
    "semantic_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
    "runtime_root": root,
}))
'''


def verify_parse_back(
    *, archive: Path, runtime_root: Path, expected_codes: np.ndarray,
    expected_semantic_sha256: str,
) -> dict[str, Any]:
    """Read the built archive back through the SHIPPING runtime, fresh interpreter."""
    codes_path = archive.with_suffix(".expected_codes.npy")
    np.save(codes_path, np.ascontiguousarray(expected_codes))
    done = subprocess.run(
        [sys.executable, "-B", "-c", _VERIFIER, str(runtime_root), str(archive), str(codes_path)],
        capture_output=True, text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    if done.returncode != 0:
        raise SA3Error(f"parse-back failed under {runtime_root}: {done.stderr.strip()[-500:]}")
    report = json.loads(done.stdout.strip().splitlines()[-1])
    if not report["codes_match"]:
        raise SA3Error(
            "parse-back carrier codes differ from the solved lattice "
            f"(max abs deviation {report['max_abs_code_deviation']})"
        )
    if report["semantic_sha256"] != expected_semantic_sha256:
        raise SA3Error("parse-back semantic section differs from the packed one")
    report["status"] = "PASS"
    return report


def build_candidate(
    *, codes: np.ndarray, semantic_mode: str, carrier_plane2: bool, drop_overlay: bool,
    output: Path, runtime_root: Path, mod: SimpleNamespace,
    edited_archive: Path = S2_ARCHIVE,
) -> dict[str, Any]:
    """Assemble sz1-tail + sz1-hpac + edited-semantic + compensated-carrier."""
    s2 = carrier_surface(edited_archive, mod)
    sz1 = sections(SZ1_GENERATION / "archive.zip")

    semantic_body = mod.ra._decompress_brotli(s2["semantic_stream"])
    semantic_stream, semantic_reserved = semantic_stream_for(
        semantic_body, mode=semantic_mode
    )

    overlay = b"" if drop_overlay else s2["overlay"]
    encoded = encode_carrier_body(codes, s2, mod, overlay)
    carrier_stream, carrier_reserved = carrier_stream_for(
        encoded["body"], plane2_split=carrier_plane2
    )
    reserved = semantic_reserved | carrier_reserved

    member = (
        RX1_HEADER.pack(
            sz1["magic"], sz1["version"], s2["codec"], s2["table_mode"], reserved,
            len(sz1["hpac"]), len(semantic_stream), len(carrier_stream),
        )
        + sz1["hpac"] + semantic_stream + carrier_stream + sz1["tail"]
    )
    archive_bytes = mod.rx1.deterministic_zip(member)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(archive_bytes)

    verify = verify_parse_back(
        archive=output, runtime_root=runtime_root,
        expected_codes=np.asarray(codes, dtype=np.int32),
        expected_semantic_sha256=hashlib.sha256(s2["parts"].semantic_blob).hexdigest(),
    )
    return {
        "archive_bytes": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_path": str(output),
        "reserved": reserved,
        "semantic_mode": semantic_mode,
        "semantic_split": semantic_mode == "pinned",
        "carrier_plane2": carrier_plane2,
        "semantic_body_bytes": len(semantic_body),
        "semantic_stream_bytes": len(semantic_stream),
        "carrier_stream_bytes": len(carrier_stream),
        "tail_bytes": len(sz1["tail"]),
        "hpac_stream_bytes": len(sz1["hpac"]),
        "overlay_dropped": bool(drop_overlay),
        "changed_coordinates": int(
            np.count_nonzero(np.asarray(codes, dtype=np.int32) != s2["codes"])
        ),
        "delta_bytes_vs_sz1": len(archive_bytes) - SZ1_ARCHIVE_BYTES,
        "parse_back": verify,
    }


# --------------------------------------------------------------------------
# the admit arithmetic, against the LIVE pointer
# --------------------------------------------------------------------------


def load_solved_lattice(
    solve_root: Path, base_codes: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Apply every retained per-pair solve onto the base lattice.

    Accepts both retained layouts -- sa2 nests results under ``pairs/``, this arm's
    authority probe does not -- because a loader that silently found ZERO results would
    return the BASE lattice and build an uncompensated archive wearing a compensated
    name.  That is the vacuity-equals-pass failure, so it refuses on an empty solve.
    """
    codes = np.asarray(base_codes, dtype=np.int32).copy()
    found = sorted(solve_root.glob("pairs/pair_*/RESULT.json")) or sorted(
        solve_root.glob("pair_*/RESULT.json")
    )
    if not found:
        raise SA3Error(f"no per-pair solves under {solve_root}; refusing to build")
    solved: list[int] = []
    for path in found:
        row = json.loads(path.read_text())
        pair = int(row["pair"])
        codes[pair] = np.asarray(row["codes"]["final"], dtype=np.int32)
        solved.append(pair)
    return codes, sorted(solved)


def admit_arithmetic(
    delta_bytes: int, *, compensated: bool, d_seg_delta_override: float | None = None,
    d_pose_compensated_override: float | None = None,
    d_pose_uncompensated_override: float | None = None,
) -> dict[str, Any]:
    """Price an archive against sz1's measured components.

    ``compensated`` is NOT cosmetic.  The zero-compensation controls exist only to
    price the compensation's marginal byte cost; they carry the FULL uncompensated
    pose collapse (d_pose 9.865e-4 vs 1.475e-4 base).  Pricing a control with the
    compensated pose model would make an archive that sa1 already refused look like
    the best row in the table -- which is exactly what happened on the first run of
    this module before this flag existed.

    Both pose-transfer models are reported because they differ by 21x and the
    campaign has not measured which one holds across the CPU->T4 boundary:

    * ABSOLUTE -- the compensation's residual d_pose damage carries in absolute
      units.  Conservative here, because sqrt(10*x) has a steeper marginal at
      sz1's lower operating point (5/sqrt(10*x) is 4.63x larger at T4's d_pose
      than at the CPU instrument's).
    * RELATIVE -- the damage carries as a fraction of the base d_pose.

    The ABSOLUTE model is the one this arm reports as its verdict.
    """
    rate_s = 25.0 * delta_bytes / RATE_DENOMINATOR

    # seg: the compensation touches frame_0 only; SegNet reads frame_1
    # (upstream/modules.py:109 -> x[:, -1, ...]), so the ONLY seg cost is S2's own.
    seg_absolute = 100.0 * (
        SA2_D_SEG_DELTA_CPU if d_seg_delta_override is None else d_seg_delta_override
    )
    seg_relative = seg_absolute * (SZ1_D_SEG / SA2_ADVISORY_D_SEG_BASE_CPU)

    # Each row is priced with ITS OWN measured legs.  Defaulting keep01 to S2's
    # residual would over-price it by ~290 uS; defaulting a row with a LARGER residual
    # to S2's would under-price it, which is the direction that ships a refusing
    # archive.  Both are wrong, so the caller supplies the row's own numbers.
    if compensated:
        d_pose_after = (
            SA2_D_POSE_COMPENSATED_CPU if d_pose_compensated_override is None
            else d_pose_compensated_override
        )
    else:
        d_pose_after = (
            SA2_D_POSE_UNCOMPENSATED_CPU if d_pose_uncompensated_override is None
            else d_pose_uncompensated_override
        )
    base_pose_s = (10.0 * SZ1_D_POSE) ** 0.5
    damage_absolute = d_pose_after - SA2_D_POSE_BASE_CPU
    pose_absolute = (10.0 * (SZ1_D_POSE + damage_absolute)) ** 0.5 - base_pose_s
    ratio = d_pose_after / SA2_D_POSE_BASE_CPU
    pose_relative = (10.0 * SZ1_D_POSE * ratio) ** 0.5 - base_pose_s

    rows = {}
    for name, seg_s, pose_s in (
        ("absolute", seg_absolute, pose_absolute),
        ("relative", seg_relative, pose_relative),
    ):
        net = rate_s + seg_s + pose_s
        rows[name] = {
            "rate_s": rate_s, "seg_s": seg_s, "pose_s": pose_s, "net_delta_s": net,
            "projected_score": SZ1_SCORE + net,
            "admits": bool(net < ADMIT_BAR_S),
            "margin_vs_bar": net - ADMIT_BAR_S,
        }
    return {
        "base_score": SZ1_SCORE,
        "base_d_seg": SZ1_D_SEG,
        "base_d_pose": SZ1_D_POSE,
        "delta_bytes": delta_bytes,
        "compensated": compensated,
        "admit_bar_s": ADMIT_BAR_S,
        "cpu_pose_cancellation_fraction": (
            1.0
            - (d_pose_after - SA2_D_POSE_BASE_CPU)
            / (
                (
                    SA2_D_POSE_UNCOMPENSATED_CPU if d_pose_uncompensated_override is None
                    else d_pose_uncompensated_override
                )
                - SA2_D_POSE_BASE_CPU
            )
        ),
        "models": rows,
        "verdict_model": "absolute",
        "transfer_caveat": (
            "The compensation was fitted on the frozen CPU-torch PoseNet; the base "
            "leg shows 21.4x CPU-vs-T4 d_pose drift on identical bytes. Transfer to "
            "T4 is UNMEASURED. Both models are reported; neither is a score."
        ),
    }


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def assert_not_in_use(dest: Path) -> None:
    """Refuse to destroy a tree a live process is reading.

    This arm rm -rf'd its own staged generation while an advisory eval was inflating
    FROM that tree.  The restored tree was byte-identical so nothing was corrupted, but
    that was luck, not design: a re-stage that changed a byte would have silently mixed
    two runtimes inside one measurement, and the receipt would have looked clean.
    A destructive re-stage is only safe when nothing is reading it.
    """
    try:
        listing = subprocess.run(
            ["/bin/ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return  # cannot enumerate: do not block the build on a missing ps
    mine = str(os.getpid())
    holders = [
        line.strip() for line in listing.splitlines()
        if str(dest) in line and line.split(maxsplit=1)[0] not in {mine}
    ]
    if holders:
        raise SA3Error(
            f"refusing to re-stage {dest}: {len(holders)} live process(es) reference it "
            f"(e.g. {holders[0][:160]}). Stage to a different --generation instead."
        )


def stage_generation(dest: Path, archive: Path, source_runtime: Path) -> dict[str, Any]:
    """Copy the sz1 runtime tree, apply the carrier patch, re-pin inflate.py."""
    if dest.exists():
        assert_not_in_use(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("cpr1", "runtime"):
        shutil.copytree(
            source_runtime / name, dest / name,
            ignore=shutil.ignore_patterns(*_RESIDUE_GLOBS), **_COPY_KWARGS,
        )
    for name in ("inflate.py", "inflate.sh"):
        shutil.copy(source_runtime / name, dest / name)
    patch = patch_runtime_tree(dest)
    shutil.copy(archive, dest / "archive.zip")

    digest, size = sha256_file(dest / "archive.zip"), (dest / "archive.zip").stat().st_size
    inflate = dest / "inflate.py"
    source = inflate.read_text()
    # Re-pin by ASSIGNMENT PATTERN, not by substituting a known base literal:
    # a non-sz1 source runtime pins ITS OWN generation sha, which a literal
    # replace silently misses -- the staged inflate then refuses its own
    # archive at decode (the ck1 r2 incident).  Fail closed if no pin moved.
    replaced = re.sub(
        r'(ARCHIVE_SHA256\s*=\s*")[0-9a-f]{64}(")', rf"\g<1>{digest}\g<2>", source
    )
    replaced = re.sub(
        r"(ARCHIVE_BYTES\s*=\s*)[0-9_]+", rf"\g<1>{size:_}", replaced
    )
    if digest not in replaced or f"{size:_}" not in replaced:
        raise SA3Error(
            "inflate.py archive pin was not updated -- the staged tree would "
            "refuse its own archive at decode"
        )
    inflate.write_text(replaced)

    # LAST mutation wins: strip only after every write, then fail closed.
    stripped = strip_import_and_appledouble_residue(dest)
    pristine = assert_tree_pristine(dest)
    if not pristine["pristine"] or pristine["unparseable_py"]:
        raise SA3Error(
            "the staged generation is not clean after stripping: "
            f"residue={pristine['residue']} unparseable={pristine['unparseable_py']}"
        )
    return {
        "generation": str(dest),
        "archive_sha256": digest,
        "archive_bytes": size,
        "runtime_patch": patch,
        "inflate_pin_updated": replaced != source,
        "residue_stripped": stripped,
        "pristine": assert_tree_pristine(dest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path,
                        default=Path("/Volumes/APDataStore/pact/ddm_sa3/build"))
    parser.add_argument("--generation", type=Path,
                        default=Path("/Volumes/APDataStore/pact/ddm_sa3/generations/sa3_rebased_sz1"))
    parser.add_argument("--no-stage", action="store_true")
    parser.add_argument(
        "--row", default="S2_film23_q2_top3_q3", choices=sorted(EDITED_ROWS),
        help="which sa1 edited-semantic row to compensate and byte-close",
    )
    args = parser.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    base = verify_sz1_base()
    pristine_before = assert_tree_pristine(SZ1_GENERATION)
    if not pristine_before["pristine"]:
        raise SA3Error(
            "the sz1 packet already carries import/AppleDouble residue; refusing "
            f"to work against a contaminated custody artifact: {pristine_before['residue']}"
        )
    mod = _imports()
    identity = assert_decoded_identity(args.work)
    tail = assert_tail_independence()
    row_spec = EDITED_ROWS[args.row]
    edited_archive = Path(row_spec["archive"])
    digest = sha256_file(edited_archive)
    if digest != row_spec["sha256"]:
        raise SA3Error(
            f"{args.row} archive sha differs ({digest} != {row_spec['sha256']}); the "
            "compensation was solved against a different object -- refusing to compile"
        )
    s2 = carrier_surface(edited_archive, mod)
    base_codes = s2["codes"]
    codes, solved = load_solved_lattice(Path(row_spec["solve_root"]), base_codes)
    # DISTINCT pairs: two directories resolving to the same index (pair_0007 / pair_7)
    # would reach 600 files with 599 real pairs, shipping one pair UNCOMPENSATED behind
    # a compensated claim -- exactly what this gate exists to prevent.
    if len(set(solved)) != PAIR_COUNT:
        raise SA3Error(
            f"{args.row}: expected {PAIR_COUNT} DISTINCT solved pairs, found "
            f"{len(set(solved))} distinct across {len(solved)} result files. A partial "
            "solve must not be byte-closed -- the unsolved pairs would ship "
            "UNCOMPENSATED while the archive claims otherwise."
        )

    # The compensated lattice's Rice residual stream has a different length, so the
    # SHIPPED reader -- which dispatches the carrier on a pinned section length --
    # refuses it.  sa2's variable-length patch is what makes it parse.  Stage the
    # patched tree FIRST so every parse-back runs against the runtime that would
    # actually ship with the candidate, never against a convenient stand-in.
    # The row's own measured compensated d_pose, from its own solve's AGGREGATE.json
    # when the probe wrote one; otherwise fall back to sa2's S2 legs and SAY SO.
    # FINDING-3 FIX.  The compensated pose leg must come from THIS row's own solve.
    # The previous code fell back to S2's constant whenever the summary was missing or
    # stale, which for keep01 turns a REFUSE (+1.16e-2 at 99.0% cancellation) into an
    # ADMIT (-8.11e-4) and byte-closes an archive that does not clear the bar.  A missing
    # or non-n600 summary now REFUSES; only S2 may use sa2's own measured legs, because
    # for S2 they ARE this row's legs.
    measured_residual_d_pose: float | None = None
    residual_source: str
    if args.row == "S2_film23_q2_top3_q3":
        residual_source = "sa2's own n600 solve for THIS row (600/600 pairs)"
    else:
        aggregate = Path(row_spec["solve_root"]) / "AUTHORITY.json"
        if not aggregate.is_file():
            raise SA3Error(
                f"{args.row}: no AUTHORITY.json at {aggregate}. Refusing to price this "
                "row with another row's residual -- run the authority solve first."
            )
        agg = json.loads(aggregate.read_text())
        n_pairs = int(agg.get("n_pairs", -1))
        if n_pairs != PAIR_COUNT:
            raise SA3Error(
                f"{args.row}: AUTHORITY.json summarizes n={n_pairs}, not {PAIR_COUNT}. A "
                "subset estimate must not price a byte-closed archive (this arm measured "
                "the estimate drift adversely from n=16 to n=70). Re-run at --pairs 600."
            )
        solve_id = agg.get("solved_pairs_sha256")
        measured_residual_d_pose = SA2_D_POSE_BASE_CPU + agg["subset"]["mean_residual"]
        residual_source = f"{args.row} own n600 solve (AUTHORITY.json, n={n_pairs})"
        del solve_id

    # Per-row source runtime + overlays: the staged/parse-back tree must decode BOTH
    # the token sections (lineage-matched coder chain — mixing lineages desyncs the
    # arithmetic decode SILENTLY, the ck1 S=79.4 T4 refusal) AND the row's semantic
    # body (receiver mode coverage).  Base = sz1's shipped runtime; per-row
    # ``runtime_overlays`` copies named files from another tree over the base.
    source_generation = Path(row_spec.get("source_runtime", SZ1_GENERATION))
    shipped_runtime = scratch_runtime_copy(source_generation, args.work, "src")
    for overlay_tree, overlay_rel in row_spec.get("runtime_overlays", ()):
        overlay_src = Path(overlay_tree) / overlay_rel
        overlay_dst = shipped_runtime / overlay_rel
        if not overlay_src.is_file():
            raise SA3Error(f"runtime overlay source missing: {overlay_src}")
        if not overlay_dst.is_file():
            raise SA3Error(
                f"runtime overlay target absent in base tree (refusing to ADD "
                f"files blind): {overlay_dst}"
            )
        shutil.copy(overlay_src, overlay_dst)
    patched_runtime = args.work / "runtime_sz1_patched"
    if patched_runtime.exists():
        shutil.rmtree(patched_runtime)
    shutil.copytree(
        shipped_runtime, patched_runtime,
        ignore=shutil.ignore_patterns("__pycache__", "._*", "*.pyc"),
    )
    runtime_patch = patch_runtime_tree(patched_runtime)

    # Split variants set RX1 header reserved bit 0 (DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1),
    # which only the sz1-lineage residual_archive.py decodes; a runtime without that
    # extension refuses the header outright ("invalid RX1 model metadata").  Rows on
    # such runtimes declare supports_semantic_split=False and build nosplit-only.
    supports_split = bool(row_spec.get("supports_semantic_split", True))
    # A container variant is (semantic_mode, carrier_plane2).  Rows declare the set they
    # can actually decode; the default reproduces the pre-ck2 plan exactly, so legacy
    # rows rebuild byte-identically.
    container_variants = row_spec.get(
        "container_variants", (("none", False), ("pinned", False))
    )
    if not supports_split:
        container_variants = tuple(
            v for v in container_variants if v[0] != "pinned"
        )
    variant_plan = [
        (f"{stage}_{_variant_label(mode, car)}", use_codes, mode, car, root)
        for stage, use_codes, root in (
            ("control", base_codes, shipped_runtime),
            ("candidate", codes, patched_runtime),
        )
        for mode, car in container_variants
    ]

    results: dict[str, Any] = {}
    for label, use_codes, mode, car, root in variant_plan:
        row = build_candidate(
            codes=use_codes, semantic_mode=mode, carrier_plane2=car, drop_overlay=True,
            output=args.work / f"{label}.zip",
            runtime_root=root, mod=mod, edited_archive=edited_archive,
        )
        row["parse_back_runtime"] = (
            "sz1_shipped_copy" if root is shipped_runtime else "sz1_copy+sa2_carrier_patch"
        )
        row["compensated"] = label.startswith("candidate")
        row["admit"] = admit_arithmetic(
            row["delta_bytes_vs_sz1"], compensated=row["compensated"],
            d_seg_delta_override=row_spec["d_seg_delta_cpu"],
            d_pose_compensated_override=measured_residual_d_pose,
            d_pose_uncompensated_override=row_spec["d_pose_uncompensated_cpu"],
        )
        results[label] = row
        print(
            f"{label:20s} {row['archive_bytes']:7d} B  "
            f"d={row['delta_bytes_vs_sz1']:+6d}  sem={row['semantic_stream_bytes']:6d}  "
            f"car={row['carrier_stream_bytes']:6d}  parse-back {row['parse_back']['status']}"
        )

    # Only a COMPENSATED archive is a candidate.  The controls are byte instruments,
    # not shippable rows -- see admit_arithmetic's ``compensated`` docstring.
    best = min(
        (r for r in results.values() if r["compensated"]),
        key=lambda r: r["admit"]["models"]["absolute"]["net_delta_s"],
    )
    if not best["admit"]["models"]["absolute"]["admits"]:
        raise SA3Error(
            "no compensated candidate clears the admit bar under the conservative "
            f"model (best net {best['admit']['models']['absolute']['net_delta_s']:+.6e})"
        )
    report = {
        "schema": "ddm_sa3_rebase.v1",
        "row": args.row,
        "measured_residual_d_pose_cpu": measured_residual_d_pose,
        "residual_source": residual_source,
        "edited_archive": str(edited_archive),
        "edited_archive_sha256": digest,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "base": base,
        "identity_gate": identity,
        "tail_gate": tail,
        "carrier_runtime_patch": runtime_patch,
        "solved_pairs": len(solved),
        "results": results,
        "best": best["archive_path"],
        "supports_semantic_split": supports_split,
        "container_variants": [list(v) for v in container_variants],
        # The compensation's byte cost, read per container variant: candidate minus its
        # OWN control, never across variants (two credits over the same redundancy do
        # not add -- the union-is-not-the-sum-of-its-legs law).
        "compensation_marginal_bytes": {
            _variant_label(mode, car): (
                results[f"candidate_{_variant_label(mode, car)}"]["archive_bytes"]
                - results[f"control_{_variant_label(mode, car)}"]["archive_bytes"]
            )
            for mode, car in container_variants
        },
        # Container credit is measured against this row's OWN identity variant, so it is
        # a property of THIS body rather than a constant carried from another lineage.
        "container_credit_bytes_vs_identity": {
            _variant_label(mode, car): (
                results[f"candidate_{_variant_label(mode, car)}"]["archive_bytes"]
                - results["candidate_nosplit"]["archive_bytes"]
            )
            for mode, car in container_variants
        }
        if ("none", False) in tuple(tuple(v) for v in container_variants)
        else {"skipped": "row declares no identity variant to measure against"},
    }
    if not args.no_stage:
        report["staged"] = stage_generation(
            args.generation, Path(best["archive_path"]), shipped_runtime
        )
    report["packet_pristine_before"] = pristine_before
    report["packet_pristine_after"] = assert_tree_pristine(SZ1_GENERATION)
    out = args.work / "SA3_REBASE.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
