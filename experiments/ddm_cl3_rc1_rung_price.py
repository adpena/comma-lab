#!/usr/bin/env python3
"""ddm_cl3 -- exact price of one HPAC capacity rung on the historical pc1/rc1 control object.

WHY THIS EXISTS.  ddm_cl2 priced its ladder against the fs2 object, where the hpac
MODEL section reached the archive as ``brotli(raw IHS1 body)``.  Since then the object
changed twice: ddm_rc1 replaced the generic Brotli byte coder on both MODEL sections
with an adaptive per-group binary-tree range coder (the ``RC1H`` / ``RC1S`` riders), and
ddm_pc1 shrank the carrier.  **A Brotli-packed IHS1 body no longer describes the object**,
so cl2's ``pack_model`` would price a section the receiver would refuse to read.  This
module re-roots the SAME ladder on the live tree and prices the model section through
rc1's coder, reusing cl2's and rc1's functions rather than reimplementing either.

THE CONTAINER RECIPE (MEASURED against the then-live pointer, 2026-09-05).  For the hpac
section the historical pc1/rc1 tree carries::

    brotli(ck2_interleave(apply_hpac(raw_ihs1, row_counts, shift=5)), q11, lgwin24)

Driving cl2's retained lambda=1.0 control raw body (17,770 B, sha ``81728190...``) through
exactly that recipe reproduces the historical control's hpac section **byte for byte** --
12,343 B, sha-identical -- which is the instrument control that licenses every rung priced
here, and which also proves that object carries cl2's control weights re-coded.

WHAT MOVES AND WHAT IS HELD.  The token FIELD is held bit-identical, so d_seg and d_pose are
held BY CONSTRUCTION and only two numbers move: the hpac container bytes and the RC64 token
stream bytes.  ``J = hpac_container + token_stream``; the live pointer's J is
12,343 + 113,419 = 125,762 B.  A rung pays iff its J is lower.

Axis: ``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; ``score_claim=false``.
No scorer runs here and no Modal is dispatched.

Usage::

  python experiments/ddm_cl3_rc1_rung_price.py control
  python experiments/ddm_cl3_rc1_rung_price.py price --rung lambda_2p0 \
      --checkpoint <qat_stage_end_epoch_0060.pt>
  python experiments/ddm_cl3_rc1_rung_price.py report
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cl2_hpac_prior_capacity_ladder as cl2
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_rc1_adaptive_section_codec as rc1codec
from experiments import ddm_rc1_model_section_adaptive_recode as rc1rec
from tac.candidate_seal import SealContractError
from tac.receiver_manifest import rebind_receiver_manifest, require_receiver_manifest_rows

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_cl3_hpac_smaller_prior_and_seed_selection")
#: Historical ladder control: pc1's x16 rung on rc1's coded model sections.  It was
#: the live frontier when ddm_cl3 ran, but the current pointer is the sj1 object below.
LIVE_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained/v3x16_on_rc1_candidate_runtime"
)
LIVE_ARCHIVE = LIVE_RUNTIME / "archive.zip"
LIVE_ARCHIVE_BYTES = 174_786
LIVE_ARCHIVE_SHA256 = "1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629"
LIVE_HPAC_BYTES = 12_343
LIVE_STREAM_BYTES = 113_419
LIVE_STREAM_SHA256 = "e07274caeacbb3a6ce00e26b42d7032671af5c8109a2cee3a0697b116ac125cf"
LIVE_JOINT_BYTES = LIVE_HPAC_BYTES + LIVE_STREAM_BYTES  # 125,762
LIVE_SCORE = 0.14411787458634504
#: The CURRENT score pointer is a successor of the pc1 object above.  The ladder rows
#: restore pc1's older token field, so their smaller archives are NOT evidence that they
#: beat this pointer: distortion is not held across that comparison.  ``stage_report``
#: verifies these bytes from the retained source archive before writing the report.
CURRENT_POINTER_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
    "/candidate_pass3/candidate_runtime"
)
CURRENT_POINTER_ARCHIVE = CURRENT_POINTER_RUNTIME / "archive.zip"
CURRENT_POINTER_ARCHIVE_BYTES = 181_645
CURRENT_POINTER_ARCHIVE_SHA256 = "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
CURRENT_POINTER_SCORE = 0.13900437796841966
CURRENT_POINTER_HPAC_BYTES = 12_343
CURRENT_POINTER_STREAM_BYTES = 120_225
CL2_CONTROL_RESULT = (
    Path("/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder")
    / "rungs/lambda_1p0/RUNG_RESULT.json"
)
CL2_CONTROL_VERIFY = CL2_CONTROL_RESULT.with_name("VERIFY_RESULT.json")
CL3_CONTROL_RESULT = STORE / "control_rc1/CONTROL_RESULT.json"
#: The seed the live pointer's weights were trained under (cl2's lambda = 1.0 control).
CONTROL_SEED = 20260716
#: The raw IHS1 body of the weights the live pointer carries (cl2's lambda=1.0 control).
CONTROL_RAW_IHS1 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/model/hpac.ihs1.raw"
)
S_PER_BYTE = 25.0 / 37_545_489
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
N = 600


class Cl3Error(RuntimeError):
    pass


def rung_root(rung: str) -> Path:
    return STORE / "rungs" / rung


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def row_admissibility(row: dict[str, Any]) -> dict[str, Any]:
    """Return the three independent legs required for an admissible ladder row."""

    legs = {
        "twin_encode_byte_identical": bool(row.get("two_encodes_identical")),
        "receiver_copy_decode_identity": bool(row.get("decoded_identity")),
        "section_census_only_model_and_stream_moved": bool(row.get("only_model_and_stream_moved")),
    }
    return {"legs": legs, "admissible": all(legs.values())}


def source_verified_current_pointer() -> dict[str, Any]:
    """Re-measure the current pointer's container facts from its retained source archive."""

    archive_bytes = CURRENT_POINTER_ARCHIVE.read_bytes()
    archive_sha256 = sha256_bytes(archive_bytes)
    if len(archive_bytes) != CURRENT_POINTER_ARCHIVE_BYTES or archive_sha256 != CURRENT_POINTER_ARCHIVE_SHA256:
        raise Cl3Error(
            "current pointer archive drifted; re-derive its score and section constants before reporting"
        )
    sections = jg2.split_member(jg2.read_archive_member(CURRENT_POINTER_ARCHIVE))
    stream = sections["tail"][jg2.RESIDUAL_COMPACT_BYTES :]
    hpac_bytes = len(sections["hpac"])
    stream_bytes = len(stream)
    if hpac_bytes != CURRENT_POINTER_HPAC_BYTES or stream_bytes != CURRENT_POINTER_STREAM_BYTES:
        raise Cl3Error(
            "current pointer section census drifted: "
            f"hpac={hpac_bytes} stream={stream_bytes}"
        )
    return {
        "score": CURRENT_POINTER_SCORE,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha256,
        "source_archive": str(CURRENT_POINTER_ARCHIVE),
        "hpac_bytes": hpac_bytes,
        "stream_bytes": stream_bytes,
        "joint_bytes": hpac_bytes + stream_bytes,
        "verified_at_source": True,
    }


def source_verified_ladder_control() -> dict[str, Any]:
    """Verify the historical pc1 control's three admissibility legs from retained receipts."""

    result = json.loads(CL2_CONTROL_RESULT.read_text(encoding="utf-8"))
    verify = json.loads(CL2_CONTROL_VERIFY.read_text(encoding="utf-8"))
    rc1_control = json.loads(CL3_CONTROL_RESULT.read_text(encoding="utf-8"))
    encodes = result.get("encodes", [])
    stream_shas = [entry.get("stream", {}).get("sha256") for entry in encodes]
    twin = len(stream_shas) == 2 and bool(stream_shas[0]) and len(set(stream_shas)) == 1
    control_row = {
        "two_encodes_identical": twin,
        "decoded_identity": result.get("decoded_identity"),
        "only_model_and_stream_moved": verify.get("only_model_and_stream_moved"),
    }
    admission = row_admissibility(control_row)
    if not admission["admissible"]:
        raise Cl3Error(f"historical ladder control is not admissible at source: {admission}")
    live_archive = rc1_control.get("live_archive", {})
    live_hpac = rc1_control.get("live_hpac", {})
    live_stream = rc1_control.get("live_stream", {})
    expected = {
        "archive_bytes": LIVE_ARCHIVE_BYTES,
        "archive_sha256": LIVE_ARCHIVE_SHA256,
        "hpac_bytes": LIVE_HPAC_BYTES,
        "stream_bytes": LIVE_STREAM_BYTES,
        "stream_sha256": LIVE_STREAM_SHA256,
    }
    observed = {
        "archive_bytes": live_archive.get("bytes"),
        "archive_sha256": live_archive.get("sha256"),
        "hpac_bytes": live_hpac.get("bytes"),
        "stream_bytes": live_stream.get("bytes"),
        "stream_sha256": live_stream.get("sha256"),
    }
    if observed != expected or not rc1_control.get("hpac_bytes_match") or not rc1_control.get(
        "hpac_sha_match"
    ):
        raise Cl3Error(
            "historical pc1/rc1 ladder control drifted at source: "
            f"expected={expected} observed={observed}"
        )
    return {
        "rate_lambda": 1.0,
        "seed": CONTROL_SEED,
        "hpac_container_bytes": LIVE_HPAC_BYTES,
        "stream_bytes": LIVE_STREAM_BYTES,
        "joint_bytes": LIVE_JOINT_BYTES,
        "joint_delta_vs_ladder_control": 0,
        "candidate_archive_bytes": LIVE_ARCHIVE_BYTES,
        "status": "MEASURED_CONTROL",
        "verified_at_source": True,
        **admission,
    }


def rc1_hpac_container(raw: bytes) -> dict[str, Any]:
    """The live tree's hpac container recipe, applied to one raw IHS1 body.

    Returns the container bytes plus every intermediate, with the losslessness of the
    rider PROVED by a fresh restore rather than asserted.
    """

    row_counts = rc1rec.hpac_row_counts(raw)
    rider = rc1codec.apply_hpac(raw, row_counts, rc1rec.HPAC_SHIFT)
    if rc1codec.restore_hpac(rider, row_counts) != raw:
        raise Cl3Error("RC1H rider does not restore the raw IHS1 body")
    interleaved = rc1rec.ck2_interleave(rider)
    container = brotli.compress(interleaved, quality=11, lgwin=24)
    return {
        "row_counts_channels": len(row_counts),
        "row_counts_params": int(sum(row_counts)),
        "shift": rc1rec.HPAC_SHIFT,
        "rider_bytes": len(rider),
        "rider_sha256": sha256_bytes(rider),
        "rider_lossless": True,
        "container_bytes": len(container),
        "container_sha256": sha256_bytes(container),
        "container_recipe": "brotli(ck2_interleave(apply_hpac(raw, row_counts, shift)), q11, lgwin24)",
        "_container": container,
        "_rider": rider,
    }


def brotli_basis_bytes(raw: bytes) -> dict[str, Any]:
    """cl2's OLD container (the Brotli q0..q11 race) on the same raw body.

    Advisory only -- the live object no longer uses it.  It is computed so ddm_cl3's
    PRE-REGISTERED predictions, which were written in cl2's Brotli currency before rc1
    changed the coder, stay scoreable in the currency they were written in.  The DECISION
    is always taken on the rc1 basis, which is the object that actually ships.
    """

    sizes = {q: len(brotli.compress(raw, quality=q, lgwin=24)) for q in range(12)}
    best_q = min(sizes, key=lambda q: (sizes[q], q))
    return {"per_quality_bytes": sizes, "winner_quality": best_q, "winner_bytes": sizes[best_q]}


def patch_inflate_pins_live(runtime_root: Path, archive_sha256: str, archive_bytes: int) -> dict[str, Any]:
    """Patch BOTH archive pins in a copy of the LIVE tree's ``inflate.py``.

    cl2's ``patch_inflate_pins`` cannot be reused here: it searches for fs2's sha and for
    ``ARCHIVE_BYTES = 180_023`` written with an underscore separator, while the live pc1 tree pins
    ``1de6c5d7...`` and writes ``ARCHIVE_BYTES = 174786`` plain.  Reusing it fails closed with
    "pins are absent or ambiguous" -- which it did, AFTER an hour of encoding.  Both pins are
    patched together (jf2 #1237: a half-updated pin is the bug this shape prevents), and the patch
    must land exactly once or this refuses.
    """

    path = runtime_root / "inflate.py"
    text = path.read_text(encoding="utf-8")
    try:
        require_receiver_manifest_rows(runtime_root, ("inflate.py",))
    except SealContractError as exc:
        raise Cl3Error(str(exc)) from exc
    old_sha = f'ARCHIVE_SHA256 = "{LIVE_ARCHIVE_SHA256}"'
    old_bytes = f"ARCHIVE_BYTES = {LIVE_ARCHIVE_BYTES}"
    new_sha = f'ARCHIVE_SHA256 = "{archive_sha256}"'
    new_bytes = f"ARCHIVE_BYTES = {archive_bytes}"
    if text.count(old_sha) != 1 or text.count(old_bytes) != 1:
        if text.count(new_sha) == 1 and text.count(new_bytes) == 1:
            try:
                manifest = rebind_receiver_manifest(runtime_root, ("inflate.py",))
            except SealContractError as exc:
                raise Cl3Error(str(exc)) from exc
            return {**cl2.file_fact(path), "manifest_rebind": manifest}
        raise Cl3Error(
            "live receiver copy inflate.py pins are absent or ambiguous: "
            f"sha_hits={text.count(old_sha)} bytes_hits={text.count(old_bytes)}"
        )
    original_manifest = (runtime_root / "MANIFEST.sha256").read_bytes()
    patched = text.replace(old_sha, new_sha).replace(old_bytes, new_bytes)
    path.write_text(patched, encoding="utf-8")
    try:
        if patched.count(new_sha) != 1 or patched.count(new_bytes) != 1:
            raise Cl3Error("inflate.py pin patch did not land exactly once")
        manifest = rebind_receiver_manifest(runtime_root, ("inflate.py",))
    except Exception:
        path.write_text(text, encoding="utf-8")
        (runtime_root / "MANIFEST.sha256").write_bytes(original_manifest)
        raise
    return {**cl2.file_fact(path), "manifest_rebind": manifest}


def stage_control(args: argparse.Namespace) -> dict[str, Any]:
    """Instrument control: the live pointer's own hpac section, rebuilt from its raw body.

    This is the row that licenses every priced rung.  It also proves, byte for byte, that the
    live pointer carries cl2's lambda=1.0 control weights.
    """

    raw = CONTROL_RAW_IHS1.read_bytes()
    packed = rc1_hpac_container(raw)
    live_sections = jg2.split_member(jg2.read_archive_member(LIVE_ARCHIVE))
    live_hpac = live_sections["hpac"]
    live_stream = live_sections["tail"][jg2.RESIDUAL_COMPACT_BYTES :]
    result = {
        "schema": "ddm_cl3_rc1_control.v1",
        "axis": AXIS,
        "score_claim": False,
        "control_raw_ihs1": {
            "path": str(CONTROL_RAW_IHS1),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        },
        "rebuilt": {k: v for k, v in packed.items() if not k.startswith("_")},
        "live_hpac": {"bytes": len(live_hpac), "sha256": sha256_bytes(live_hpac)},
        "live_stream": {"bytes": len(live_stream), "sha256": sha256_bytes(live_stream)},
        "live_joint_bytes": len(live_hpac) + len(live_stream),
        "hpac_bytes_match": packed["container_bytes"] == len(live_hpac),
        "hpac_sha_match": packed["container_sha256"] == sha256_bytes(live_hpac),
        "live_archive": {"bytes": LIVE_ARCHIVE.stat().st_size, "sha256": cl2.sha256_file(LIVE_ARCHIVE)},
    }
    root = STORE / "control_rc1"
    cl2.atomic_json(root / "CONTROL_RESULT.json", result)
    cl2.persist_bytes(root / "retained" / "control_hpac_container.bin", packed["_container"], label="control hpac")
    cl2.persist_bytes(root / "retained" / "control_hpac_rider.rc1h", packed["_rider"], label="control rider")
    if not (result["hpac_bytes_match"] and result["hpac_sha_match"]):
        raise Cl3Error("CONTROL FAILED: the rc1 recipe does not reproduce the live hpac section")
    print(json.dumps({k: v for k, v in result.items() if k != "rebuilt"}, indent=2, sort_keys=True))
    return result


def stage_price(args: argparse.Namespace) -> dict[str, Any]:
    rung = args.rung
    if rung not in cl2.RUNG_LAMBDA:
        raise Cl3Error(f"unknown rung {rung}; admitted: {sorted(cl2.RUNG_LAMBDA)}")
    root = rung_root(rung)
    result_path = root / "RC1_RUNG_RESULT.json"
    if result_path.is_file() and not args.force:
        raise Cl3Error(f"rung already priced at {result_path}; pass --force to redo")
    retained = root / "retained_rc1"
    retained.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    ck = cl2.load_checkpoint_facts(checkpoint, args.expected_profile)
    if ck["rate_lambda"] != cl2.RUNG_LAMBDA[rung]:
        raise Cl3Error(f"checkpoint rate_lambda {ck['rate_lambda']} does not match rung {rung}")
    if ck["seed"] != cl2.RUNG_SEED[rung]:
        raise Cl3Error(f"checkpoint seed {ck['seed']} does not match rung {rung} (expects {cl2.RUNG_SEED[rung]})")
    if cl2.sha256_file(LIVE_ARCHIVE) != LIVE_ARCHIVE_SHA256:
        raise Cl3Error("live pointer archive drifted; re-derive the constants before pricing")
    cl2.atomic_json(
        root / "RC1_RUNG_START.json",
        {"schema": "ddm_cl3_rc1_rung_start.v1", "rung": rung, "checkpoint": ck, "live_joint_bytes": LIVE_JOINT_BYTES},
    )

    # 1. pack the model: raw IHS1 (cl2's packer) then rc1's container recipe -----------
    packed_raw = cl2.rx2._pack_terminal_ihs1(checkpoint, retained / "model")
    raw = Path(packed_raw["raw"]["path"]).read_bytes()
    model = rc1_hpac_container(raw)
    hpac_section = model.pop("_container")
    rider = model.pop("_rider")
    cl2.persist_bytes(retained / "model" / "hpac.rc1h.rider", rider, label="rung rider")
    hpac_fact = cl2.persist_bytes(retained / "model" / "hpac.rc1h.container", hpac_section, label="rung hpac container")

    # 2. stage the LIVE tree carrying the new model section -----------------------------
    staged = retained / "model_only_runtime"
    if not staged.exists():
        shutil.copytree(LIVE_RUNTIME, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
    base_member = jg2.read_archive_member(LIVE_ARCHIVE)
    model_member = cl2.replace_hpac_section(base_member, hpac_section)
    model_archive = retained / "model_only_runtime_archive.zip"
    jg2.pack_archive(model_member, model_archive)
    shutil.copyfile(model_archive, staged / "archive.zip")
    env = jg2._prepare(SimpleNamespace(store=str(root), runtime_root=str(staged)), f"cl3_{rung}")
    if env["sections"]["hpac"] != hpac_section:
        raise Cl3Error("staged archive does not carry the packed model section")
    target = jg2.load_tokens(cl2.FIELD)

    # 3. encode twice -------------------------------------------------------------------
    # A resumed run reuses streams the previous attempt already finished.  The encodes cost ~50 min
    # each; re-running them to recover from a post-encode failure would be pure waste, and the
    # bytes are what the row is made of, so reuse is byte-identical by construction.
    encodes: list[dict[str, Any]] = []
    if args.reuse_encodes:
        work = root / "work"
        first, repeat = work / f"tail_cl3_{rung}.bin", work / f"tail_cl3_{rung}_repeat.bin"
        if not first.is_file():
            raise Cl3Error(f"--reuse-encodes: no finished stream at {first}")
        encodes.append({"stream": cl2.file_fact(first), "reused_from_previous_attempt": True})
        if repeat.is_file():
            encodes.append({"stream": cl2.file_fact(repeat), "reused_from_previous_attempt": True})
        cl2.progress({"stage": "encode", "rung": rung, "reused": len(encodes)})
    # NOTE the guard is on the FLAG, not on ``encodes`` being non-empty.  Guarding on ``encodes``
    # looks equivalent and is not: on a fresh run the list becomes non-empty after pass 0, so the
    # SECOND encode never runs and the row silently loses its determinism proof.  That regression
    # shipped once (ddm_cl3 lambda_1p0_s18, first attempt) -- it is why this comment exists.
    for pass_index, tag in enumerate(() if args.reuse_encodes else (f"cl3_{rung}", f"cl3_{rung}_repeat")):
        if pass_index == 1 and args.skip_repeat:
            break
        encodes.append(
            jg2.encode_tail(
                residual=env["residual"],
                renderer=env["renderer"],
                renderer_dir=env["renderer_dir"],
                parts=env["parts"],
                target=target,
                library=env["library"],
                route_b=env["route_b"],
                work=env["work"],
                tag=tag,
                frames=N,
                checkpoint_every=20,
                resume=True,
            )
        )
        cl2.progress({"stage": "encode", "rung": rung, "pass": pass_index, "bytes": encodes[-1]["stream"]["bytes"]})
    emitted = Path(encodes[0]["stream"]["path"]).read_bytes()
    two_encodes_identical = len(encodes) == 2 and Path(encodes[1]["stream"]["path"]).read_bytes() == emitted
    stream_fact = cl2.persist_bytes(retained / "token_stream.rc64.bin", emitted, label="rung stream")

    # 4. candidate archive + receiver-copy decode ----------------------------------------
    sections = dict(env["sections"])
    sections["tail"] = sections["residual_compact"] + emitted
    candidate = retained / "candidate_archive.zip"
    jg2.pack_archive(jg2.join_member(sections), candidate)
    candidate_fact = cl2.file_fact(candidate)
    receiver_copy = retained / "receiver_copy_runtime"
    if not receiver_copy.exists():
        shutil.copytree(LIVE_RUNTIME, receiver_copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
    shutil.copyfile(candidate, receiver_copy / "archive.zip")
    pin_fact = patch_inflate_pins_live(receiver_copy, candidate_fact["sha256"], candidate_fact["bytes"])
    decoded, decode_report, decode_seconds = cl2.decode_with_receiver(env, receiver_copy / "archive.zip")
    identity = bool(np.array_equal(decoded, np.asarray(target)))
    decoded_fact = cl2.persist_bytes(retained / "decoded_tokens.u8", decoded.tobytes(order="C"), label="decoded field")

    # 5. section census: prove ONLY the model section and the stream moved -------------
    live_sections = jg2.split_member(jg2.read_archive_member(LIVE_ARCHIVE))
    cand_sections = jg2.split_member(jg2.read_archive_member(candidate))
    census: dict[str, Any] = {}
    for name in ("semantic", "carrier"):
        census[name] = {
            "bytes": len(cand_sections[name]),
            "identical_to_live": cand_sections[name] == live_sections[name],
        }
    census["residual_table"] = {
        "bytes": jg2.RESIDUAL_COMPACT_BYTES,
        "identical_to_live": cand_sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES]
        == live_sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES],
    }
    census["hpac"] = {
        "bytes": len(cand_sections["hpac"]),
        "identical_to_live": cand_sections["hpac"] == live_sections["hpac"],
    }
    only_model_and_stream_moved = (
        census["semantic"]["identical_to_live"]
        and census["carrier"]["identical_to_live"]
        and census["residual_table"]["identical_to_live"]
        and not census["hpac"]["identical_to_live"]
    )
    if not only_model_and_stream_moved:
        raise Cl3Error(f"{rung}: a section other than the model and the stream moved; the field-held claim is void")

    joint = int(hpac_fact["bytes"]) + int(stream_fact["bytes"])
    result = {
        "section_census": census,
        "only_model_and_stream_moved": bool(only_model_and_stream_moved),
        "schema": "ddm_cl3_rc1_rung_price.v1",
        "axis": AXIS,
        "score_claim": False,
        "rung": rung,
        "rate_lambda": cl2.RUNG_LAMBDA[rung],
        "seed": cl2.RUNG_SEED[rung],
        "checkpoint": ck,
        "model": model,
        "brotli_basis_advisory": brotli_basis_bytes(raw),
        "model_raw_ihs1_bytes": len(raw),
        "model_raw_ihs1_sha256": sha256_bytes(raw),
        "hpac_container": hpac_fact,
        "hpac_delta_vs_live_12343": int(hpac_fact["bytes"]) - LIVE_HPAC_BYTES,
        "stream": stream_fact,
        "stream_delta_vs_live_113419": int(stream_fact["bytes"]) - LIVE_STREAM_BYTES,
        "joint_bytes": joint,
        "joint_delta_vs_live_125762": joint - LIVE_JOINT_BYTES,
        "candidate_archive": candidate_fact,
        "candidate_archive_delta_vs_live_174786": candidate_fact["bytes"] - LIVE_ARCHIVE_BYTES,
        "candidate_rate_only_delta_S": (candidate_fact["bytes"] - LIVE_ARCHIVE_BYTES) * S_PER_BYTE,
        "two_encodes_identical": two_encodes_identical,
        "decoded": decoded_fact,
        "decoded_identity": identity,
        "decode_report": decode_report,
        "decode_wall_clock_seconds": decode_seconds,
        "receiver_copy_runtime": {"path": str(receiver_copy), "inflate_pins": pin_fact},
        "encodes": encodes,
        "distortion": "HELD by construction (field bit-identical); NOT re-measured",
    }
    cl2.atomic_json(result_path, result)
    if not identity:
        raise Cl3Error(f"{rung}: the receiver did not decode the candidate to the exact field")
    print(json.dumps({k: v for k, v in result.items() if k not in ("encodes", "model", "checkpoint")}, indent=2))
    return result


def stage_report(args: argparse.Namespace) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for rung in cl2.RUNG_LAMBDA:
        path = rung_root(rung) / "RC1_RUNG_RESULT.json"
        if path.is_file():
            rows[rung] = json.loads(path.read_text(encoding="utf-8"))
    # The ladder's lambda = 1.0 control was the live pointer when cl3 ran -- it carries cl2's
    # control weights re-coded by rc1 -- so it has no cl3 row of its own.  Without injecting
    # that historical control, the secant this arm exists to measure would not appear.
    control = {
        "hpac_container": {"bytes": LIVE_HPAC_BYTES},
        "stream": {"bytes": LIVE_STREAM_BYTES},
        "joint_bytes": LIVE_JOINT_BYTES,
    }
    priced = {"live_pointer_control_lambda_1p0": control, **rows}
    ladder = [r for r in ("live_pointer_control_lambda_1p0", "lambda_2p0", "lambda_4p0") if r in priced]
    slopes = []
    for left, right in itertools.pairwise(ladder):
        d_model = priced[right]["hpac_container"]["bytes"] - priced[left]["hpac_container"]["bytes"]
        d_stream = priced[right]["stream"]["bytes"] - priced[left]["stream"]["bytes"]
        slopes.append(
            {
                "from": left,
                "to": right,
                "delta_model_bytes": d_model,
                "delta_stream_bytes": d_stream,
                "delta_joint_bytes": d_model + d_stream,
                "pays": bool(d_model + d_stream < 0),
            }
        )
    # Seed rungs hold lambda fixed, so they are NOT ladder steps (a secant over them would divide
    # by ~0 model bytes).  Their spread against the control is what says whether the lever lives.
    seed_rows = {r: v for r, v in rows.items() if v["rate_lambda"] == 1.0}
    seed_spread = None
    if seed_rows:
        joints = [LIVE_JOINT_BYTES, *(v["joint_bytes"] for v in seed_rows.values())]
        seed_spread = {
            "seeds_priced": [CONTROL_SEED, *(v["seed"] for v in seed_rows.values())],
            "joint_bytes": joints,
            "spread_bytes": max(joints) - min(joints),
            "best_is_the_incumbent_control": min(joints) == LIVE_JOINT_BYTES,
            "fixed_law_noise_floor_bytes": 0,
            "note": "the fixed-law noise floor is 0 B (cl2's twin was byte-identical at every layer), so the spread is pure seed effect; with n=3 the incumbent leading is chance, so the SCALE is the claim, not the ranking",
        }
    report_rows: dict[str, dict[str, Any]] = {
        "live_pointer_control_lambda_1p0": source_verified_ladder_control()
    }
    for rung in ("lambda_2p0", "lambda_1p0_s17", "lambda_1p0_s18"):
        if rung not in rows:
            continue
        row = rows[rung]
        admission = row_admissibility(row)
        report_rows[rung] = {
            "rate_lambda": row["rate_lambda"],
            "seed": row["seed"],
            "hpac_container_bytes": row["hpac_container"]["bytes"],
            "stream_bytes": row["stream"]["bytes"],
            "joint_bytes": row["joint_bytes"],
            "joint_delta_vs_ladder_control": row["joint_bytes"] - LIVE_JOINT_BYTES,
            "candidate_archive_bytes": row["candidate_archive"]["bytes"],
            "status": "MEASURED",
            **admission,
        }
    report_rows["lambda_4p0"] = {
        "rate_lambda": 4.0,
        "seed": CONTROL_SEED,
        "hpac_container_bytes": None,
        "stream_bytes": None,
        "joint_bytes": None,
        "joint_delta_vs_ladder_control": None,
        "candidate_archive_bytes": None,
        "status": "NOT_RUN_BECAUSE_FALSIFIED",
        "legs": {
            "twin_encode_byte_identical": None,
            "receiver_copy_decode_identity": None,
            "section_census_only_model_and_stream_moved": None,
        },
        "admissible": False,
    }
    admissible_rows = [
        (rung, row)
        for rung, row in report_rows.items()
        if row["admissible"] and row["joint_bytes"] is not None
    ]
    best_rung, best = min(admissible_rows, key=lambda item: item[1]["joint_bytes"])
    current_pointer = source_verified_current_pointer()
    report = {
        "schema": "ddm_cl3_rc1_ladder_report.v2",
        "axis": AXIS,
        "score_claim": False,
        "ladder_control": {
            "score": LIVE_SCORE,
            "archive_bytes": LIVE_ARCHIVE_BYTES,
            "archive_sha256": LIVE_ARCHIVE_SHA256,
            "hpac_bytes": LIVE_HPAC_BYTES,
            "stream_bytes": LIVE_STREAM_BYTES,
            "joint_bytes": LIVE_JOINT_BYTES,
            "role": "historical pc1/rc1 control whose token field every rung restores",
        },
        "current_pointer": current_pointer,
        "rows": report_rows,
        "adjacent_slopes": slopes,
        "seed_spread": seed_spread,
        "best_rung": best_rung,
        "best_beats_ladder_control": bool(best["joint_bytes"] < LIVE_JOINT_BYTES),
        "best_beats_live_pointer": False,
        "best_beats_live_pointer_reason": (
            "no ladder row holds the current sj1 pointer's distortion: each row restores the older "
            "pc1 control field, so smaller archive bytes are not a score comparison; no scorer ran"
        ),
    }
    out = Path(args.out) if args.out else STORE / "RC1_LADDER_REPORT.json"
    cl2.atomic_json(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("control", help="rebuild the live pointer's hpac section from its raw body")
    price = sub.add_parser("price", help="exact price of one rung on the live rc1-coded object")
    price.add_argument("--rung", required=True, choices=sorted(cl2.RUNG_LAMBDA))
    price.add_argument("--checkpoint", required=True, type=Path)
    price.add_argument("--expected-profile", default="cl2_shipped_ladder", choices=("cl2_shipped_ladder",))
    price.add_argument("--skip-repeat", action="store_true", help="skip the second encode (row is NOT admissible)")
    price.add_argument(
        "--reuse-encodes",
        action="store_true",
        help="reuse streams a previous attempt already finished, instead of re-encoding (~50 min each)",
    )
    price.add_argument("--force", action="store_true")
    report = sub.add_parser("report", help="ladder table against the live pointer")
    report.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "control":
        stage_control(args)
    elif args.stage == "price":
        stage_price(args)
    else:
        stage_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
