#!/usr/bin/env python3
"""ddm_iv1 — persist the #826 repack rungs as rebuildable bytes + a SHA-256 manifest.

WHY THIS FILE EXISTS
--------------------
``ddm_iv1``'s memo (``.omx/research/ddm_iv1_inventory_drain_20260803.md`` §4.8)
reported a rung ladder (A/B/C/D) that existed only as prose: the archives were
never written to disk.  A successor could not re-derive them without redoing the
whole section.  That is the record-censoring class applied to bytes.  This script
is the cure: it REBUILDS every rung from the two immutable source archives and
emits a manifest carrying bytes + sha256 + the exact rebuild command.

THE MEASURED FACT THE LADDER RESTS ON
-------------------------------------
``gr1``'s token codes and the live-best (``pu2``) token codes are **bit-identical**
(0 of 1,843,200 uint8 entries differ, shape ``(600, 24, 32, 4)``).  Re-encoding
gr1's decoded codes with ``ix2.encode_token_frame`` reproduces pu2's bulk section
**byte-for-byte**.  So gr1's ``state/tokens.dr7t`` is 5,183 B larger than pu2's
bulk purely because of CODER GENERATION (``DR7T`` -> ``IX2TOK01``), not content.

THE RUNGS
---------
A  shipped-equivalent : DR7T bulk, gr1's 4 joint sections, single-member container
B  A + ix2 token recode                       (the coder-generation swap alone)
C  B + pu2-shaped config section
D  C + gr1's two pose sections merged into one (pu2 ships a single pose section)

AUTHORITY / SCOPE
-----------------
``research_only``.  This script measures CONTAINER BYTES ONLY.  It performs zero
scorer forwards and makes NO score claim: the seg leg of #826 is cross-instrument
(~1.4 ppm) and is owed one exact ``upstream/evaluate.py`` n600 row, and receiver
acceptance is a separate verdict produced by ``--verify-receiver``.  A smaller
archive is not a score.

Sources are READ-ONLY; nothing here mutates them.

Rebuild:
    .venv/bin/python experiments/ddm_iv1_repack_rungs.py \
        --out-dir /Volumes/VertigoDataTier/pact/ddm_iv1_20260803
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import shutil
import sys
import zipfile
from typing import Any

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))

import ddm_r7_token_coder as r7

from tac.optimization import ddm_ix2_archive_container as ix2

GR1 = pathlib.Path(
    "/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_cell_drop50_archive.zip"
)
PU2 = pathlib.Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/archive.zip"
)

# gr1 member order is load-bearing: it is the joint-section order the rungs keep.
GR1_RENDERER = "state/renderer.sec"
GR1_SELECTOR = "state/selector.sec"
GR1_POSE_STUB = "state/pose_stub.sec"
GR1_POSE_WARP = "state/pose_warp.stp"
GR1_TOKENS = "state/tokens.dr7t"

# Sizes ddm_iv1's memo §4.8 published. Rebuild must reproduce them or report a delta.
MEMO_EXPECTED = {"A": 357161, "B": 351978, "C": 352048, "D": 352021}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _members(path: pathlib.Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info.filename) for info in archive.infolist()}


def load_sources() -> dict[str, Any]:
    """Read both source archives and establish the bit-identity fact.

    Returns the decoded gr1 codes, pu2's parsed sections, and the identity receipt.
    Raises if the archives are missing so the caller fails closed rather than
    silently emitting rungs built on nothing.
    """

    for path in (GR1, PU2):
        if not path.is_file():
            raise SystemExit(f"REFUSE: source archive absent: {path}")

    gr1 = _members(GR1)
    pu2_payload = _members(PU2)["0.bin"]
    pu2_bulk, pu2_joint = ix2.parse_payload(pu2_payload)

    codes_gr1 = r7.decode_token_codes(gr1[GR1_TOKENS])
    codes_pu2 = ix2.decode_token_frame(pu2_bulk)

    if codes_gr1.shape != codes_pu2.shape:
        raise SystemExit(
            f"REFUSE: code shape mismatch gr1={codes_gr1.shape} pu2={codes_pu2.shape}"
        )
    differing = int((codes_gr1 != codes_pu2).sum())

    recoded = ix2.encode_token_frame(codes_gr1)
    if not np.array_equal(ix2.decode_token_frame(recoded), codes_gr1):
        raise SystemExit("REFUSE: ix2 token round-trip is not lossless")

    return {
        "gr1": gr1,
        "pu2_bulk": pu2_bulk,
        "pu2_joint": pu2_joint,
        "codes": codes_gr1,
        "recoded_bulk": recoded,
        "identity": {
            "code_shape": list(codes_gr1.shape),
            "elements_total": int(codes_gr1.size),
            "elements_differing": differing,
            "bit_identical": differing == 0,
            "dr7t_bytes": len(gr1[GR1_TOKENS]),
            "ix2_recode_bytes": len(recoded),
            "format_excess_bytes": len(gr1[GR1_TOKENS]) - len(recoded),
            "recode_byte_identical_to_pu2_bulk": recoded == pu2_bulk,
            "ix2_roundtrip_lossless": True,
        },
    }


def build_rungs(src: dict[str, Any]) -> dict[str, bytes]:
    """Construct each rung as a complete single-member archive (bytes)."""

    gr1 = src["gr1"]
    renderer = gr1[GR1_RENDERER]
    selector = gr1[GR1_SELECTOR]
    pose_stub = gr1[GR1_POSE_STUB]
    pose_warp = gr1[GR1_POSE_WARP]
    config = src["pu2_joint"][0]  # pu2-shaped config section, verbatim

    gr1_joint = [renderer, selector, pose_stub, pose_warp]

    rungs: dict[str, bytes] = {}
    rungs["A"] = ix2.build_single_member_zip(
        ix2.build_payload(gr1[GR1_TOKENS], gr1_joint)
    )
    rungs["B"] = ix2.build_single_member_zip(
        ix2.build_payload(src["recoded_bulk"], gr1_joint)
    )
    rungs["C"] = ix2.build_single_member_zip(
        ix2.build_payload(src["recoded_bulk"], [config, *gr1_joint])
    )
    rungs["D"] = ix2.build_single_member_zip(
        ix2.build_payload(
            src["recoded_bulk"],
            [config, renderer, selector, pose_stub + pose_warp],
        )
    )
    return rungs


def verify_rung_parses(rung: bytes, *, expect_sections: int) -> dict[str, Any]:
    """Re-parse a built rung and confirm it closes exactly on its final byte."""

    with zipfile.ZipFile(io.BytesIO(rung)) as archive:
        names = archive.namelist()
        payload = archive.read(names[0])
    bulk, joint = ix2.parse_payload(payload)
    return {
        "members": names,
        "parses": True,
        "bulk_bytes": len(bulk),
        "joint_section_count": len(joint),
        "joint_section_bytes": [len(s) for s in joint],
        "section_count_matches": len(joint) == expect_sections,
    }


PU2_SUBMISSION = pathlib.Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2"
)
SPILLOVER = (
    pathlib.Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/gd3_CONTROL_identity_rebuild.zip"),
    pathlib.Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip"),
)


def verify_receiver(rungs: dict[str, bytes], work: pathlib.Path) -> dict[str, Any]:
    """Run the LIVE-BEST submission's own receiver against each rung.

    Typed verdicts only. The control (pu2's own archive) MUST be accepted or the
    harness itself is untrusted -- that is the positive control, and a receiver
    test that cannot return a negative is not a test.
    """

    import importlib.util

    runner = PU2_SUBMISSION / "inflate_runner.py"
    if not runner.is_file():
        return {"status": "SKIPPED_RECEIVER_ABSENT", "path": str(runner)}
    sys.path.insert(0, str(PU2_SUBMISSION))
    spec = importlib.util.spec_from_file_location("iv1_inflate_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases: dict[str, bytes] = {"ctl_pu2_live_best": (PU2_SUBMISSION / "archive.zip").read_bytes()}
    cases.update({f"rung{k}": v for k, v in rungs.items()})
    cases["gr1_original_6member"] = GR1.read_bytes()

    out: dict[str, Any] = {}
    for label, blob in cases.items():
        case_dir = work / label
        # rmtree, not a hand-rolled reverse-rglob walk: gr1's 6-member archive
        # extracts a nested ``state/`` directory, and the hand-rolled version
        # only removes nested dirs if the walk happens to visit them last.
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "_in.zip").write_bytes(blob)
        from tac.submission_archive import safe_extract_zip

        safe_extract_zip(case_dir / "_in.zip", case_dir)
        (case_dir / "_in.zip").unlink()
        try:
            decoder = module.Decoder(case_dir)
            out[label] = {
                "verdict": "ACCEPTED_CONSTRUCT",
                "n_pairs": int(decoder.n_pairs),
            }
        except BaseException as exc:  # SystemExit is not an Exception
            out[label] = {
                "verdict": "REJECTED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            }
    out["_positive_control_ok"] = (
        out.get("ctl_pu2_live_best", {}).get("verdict") == "ACCEPTED_CONSTRUCT"
    )
    # SELF-PROTECT (round-2 review finding): the submission VENDORS its own
    # ``ddm_r7_token_coder`` and it is NOT byte-identical to the repo copy
    # (39,404 B vs 61,695 B). ``inflate_runner`` imports it by bare name, so
    # whichever copy reached ``sys.modules`` first wins -- and this script
    # imports the repo copy at module load. Byte-identity of ``ix2`` means the
    # container verdicts below are unaffected, but a future caller reaching the
    # LEGACY (DR7T) decode path would silently exercise the wrong coder. Report
    # which module actually bound rather than letting the harness stay quiet.
    bound = sys.modules.get("ddm_r7_token_coder")
    vendored = PU2_SUBMISSION / "ddm_r7_token_coder.py"
    out["_module_binding"] = {
        "r7_bound_from": getattr(bound, "__file__", None),
        "r7_vendored_copy": str(vendored),
        "r7_bound_is_vendored": (
            bool(bound) and pathlib.Path(getattr(bound, "__file__", "")) == vendored
        ),
        "r7_repo_and_vendored_identical": (
            vendored.is_file()
            and _sha256(vendored.read_bytes())
            == _sha256((_REPO / "experiments" / "ddm_r7_token_coder.py").read_bytes())
        ),
        "affects_verdicts_below": False,
        "why_not": (
            "every rung verdict is produced inside the ix2 single-member path, which "
            "does not call the r7 coder; the ix2 module IS byte-identical between repo "
            "and submission. The one legacy-path case (gr1_original_6member) is refused "
            "at the manifest frame0_policy check, before any token decode."
        ),
    }
    return out


def pose_normalisation(src: dict[str, Any], rungs: dict[str, bytes]) -> dict[str, Any]:
    """The apples-to-apples check the byte ladder alone cannot make.

    gr1 predates the pose work that the live best carries.  Comparing raw archive
    sizes across vehicles with DIFFERENT pose content is the classic
    apples-to-apples failure, so normalise for it explicitly.
    """

    gr1 = src["gr1"]
    gr1_pose = len(gr1[GR1_POSE_STUB]) + len(gr1[GR1_POSE_WARP])
    pu2_pose = len(src["pu2_joint"][3])
    live_best = PU2.stat().st_size
    deficit = pu2_pose - gr1_pose
    rows = {}
    for name, blob in rungs.items():
        raw = len(blob) - live_best
        rows[name] = {
            "raw_vs_live_best_bytes": raw,
            "pose_normalised_vs_live_best_bytes": raw + deficit,
            "still_a_win_after_normalisation": (raw + deficit) < 0,
        }
    return {
        "gr1_pose_bytes": gr1_pose,
        "gr1_pose_magic": gr1[GR1_POSE_WARP][:8].decode("latin1"),
        "pu2_pose_bytes": pu2_pose,
        "pu2_pose_magic": src["pu2_joint"][3][:8].decode("latin1"),
        "gr1_pose_content_deficit_bytes": deficit,
        "per_rung": rows,
    }


def spillover() -> dict[str, Any]:
    """Do the CURRENT-generation archives still carry the DR7T format excess?"""

    pu2_codes = ix2.decode_token_frame(
        ix2.parse_payload(_members(PU2)["0.bin"])[0]
    )
    out: dict[str, Any] = {}
    for path in SPILLOVER:
        if not path.is_file():
            out[path.name] = {"status": "ABSENT"}
            continue
        members = _members(path)
        if GR1_TOKENS not in members:
            out[path.name] = {"status": "NO_DR7T_MEMBER"}
            continue
        dr7t = members[GR1_TOKENS]
        codes = r7.decode_token_codes(dr7t)
        recoded = ix2.encode_token_frame(codes)
        out[path.name] = {
            "archive_bytes": path.stat().st_size,
            "dr7t_bytes": len(dr7t),
            "ix2_recode_bytes": len(recoded),
            "dead_format_bytes": len(dr7t) - len(recoded),
            "codes_bit_identical_to_live_best": bool(
                codes.shape == pu2_codes.shape and int((codes != pu2_codes).sum()) == 0
            ),
            "renderer_magic": members[GR1_RENDERER][:8].decode("latin1"),
            "pose_magic": members[GR1_POSE_WARP][:8].decode("latin1"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("/Volumes/VertigoDataTier/pact/ddm_iv1_20260803"),
        help="SSD cold-store destination for the rung archives + manifest.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Measure and report; write nothing."
    )
    parser.add_argument(
        "--verify-receiver",
        action="store_true",
        help="Also run the live-best receiver against every rung and report typed verdicts.",
    )
    args = parser.parse_args(argv)

    src = load_sources()
    identity = src["identity"]
    print("=== token-code identity (the fact the ladder rests on) ===")
    for key, value in identity.items():
        print(f"  {key:38s} {value}")
    if not identity["bit_identical"]:
        print("\nWARNING: codes are NOT bit-identical; the ladder's premise does not hold.")

    rungs = build_rungs(src)
    expect_sections = {"A": 4, "B": 4, "C": 5, "D": 4}
    live_best = PU2.stat().st_size

    print("\n=== rung ladder (rebuilt) ===")
    print(f"  live best (pu2 archive.zip)            = {live_best} B")
    rows: dict[str, Any] = {}
    all_match = True
    for name in ("A", "B", "C", "D"):
        blob = rungs[name]
        size = len(blob)
        expected = MEMO_EXPECTED[name]
        delta = size - expected
        all_match &= delta == 0
        parse = verify_rung_parses(blob, expect_sections=expect_sections[name])
        rows[name] = {
            "bytes": size,
            "sha256": _sha256(blob),
            "memo_expected_bytes": expected,
            "delta_vs_memo_bytes": delta,
            "reproduces_memo": delta == 0,
            "vs_live_best_bytes": size - live_best,
            "parse_back": parse,
        }
        flag = "OK " if delta == 0 else "DIFF"
        print(
            f"  rung {name}: {size:8d} B  memo={expected:8d}  delta={delta:+6d} [{flag}] "
            f"vs_live_best={size - live_best:+6d}  sections={parse['joint_section_count']}"
        )

    print(
        "\n  ALL RUNGS REPRODUCE THE MEMO"
        if all_match
        else "\n  *** AT LEAST ONE RUNG DIFFERS FROM THE MEMO — this is a FINDING ***"
    )

    receiver: dict[str, Any] = {"status": "NOT_RUN (pass --verify-receiver)"}
    if args.verify_receiver:
        receiver = verify_receiver(rungs, args.out_dir / "recv")
        print("\n=== receiver acceptance (live-best submission's own inflate_runner) ===")
        for label, row in receiver.items():
            if label.startswith("_"):
                continue
            if row["verdict"] == "ACCEPTED_CONSTRUCT":
                print(f"  {label:24s} ACCEPTED_CONSTRUCT  n_pairs={row['n_pairs']}")
            else:
                print(f"  {label:24s} REJECTED  [{row['error_type']}] {row['error'][:70]}")
        print(f"  positive control OK: {receiver.get('_positive_control_ok')}")

    posenorm = pose_normalisation(src, rungs)
    print("\n=== apples-to-apples: pose-content normalisation ===")
    print(
        f"  gr1 pose {posenorm['gr1_pose_bytes']} B ({posenorm['gr1_pose_magic']})  vs  "
        f"live-best pose {posenorm['pu2_pose_bytes']} B ({posenorm['pu2_pose_magic']})"
    )
    print(f"  gr1 POSE-CONTENT DEFICIT = {posenorm['gr1_pose_content_deficit_bytes']} B")
    for name, row in posenorm["per_rung"].items():
        print(
            f"    rung {name}: raw {row['raw_vs_live_best_bytes']:+6d} B -> "
            f"pose-normalised {row['pose_normalised_vs_live_best_bytes']:+6d} B   "
            f"still a win: {row['still_a_win_after_normalisation']}"
        )

    spill = spillover()
    print("\n=== spillover: current-generation archives ===")
    for name, row in spill.items():
        if "dead_format_bytes" not in row:
            print(f"  {name}: {row.get('status')}")
            continue
        print(
            f"  {name}: total={row['archive_bytes']} dr7t={row['dr7t_bytes']} "
            f"dead_format={row['dead_format_bytes']} B  codes_identical="
            f"{row['codes_bit_identical_to_live_best']}  ren={row['renderer_magic']} "
            f"pose={row['pose_magic']}"
        )

    manifest = {
        "schema": "ddm_iv1_repack_rung_manifest.v1",
        "receiver_acceptance": receiver,
        "pose_normalisation": posenorm,
        "spillover": spill,
        "arm": "ddm_iv1",
        "date_utc": "2026-08-03",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "evidence_axis": "[macOS-CPU advisory; container-byte measurement, ZERO scorer forwards]",
        "rebuild_command": (
            ".venv/bin/python experiments/ddm_iv1_repack_rungs.py "
            f"--out-dir {args.out_dir}"
        ),
        "sources": {
            "gr1_archive": {
                "path": str(GR1),
                "bytes": GR1.stat().st_size,
                "sha256": _sha256(GR1.read_bytes()),
            },
            "pu2_live_best_archive": {
                "path": str(PU2),
                "bytes": live_best,
                "sha256": _sha256(PU2.read_bytes()),
            },
        },
        "token_code_identity": identity,
        "rungs": rows,
        "all_rungs_reproduce_memo": all_match,
        "caveat": (
            "Container bytes only. Receiver acceptance is a SEPARATE verdict "
            "(see the receiver-acceptance section of the ddm_iv1 memo). The seg leg "
            "of #826 remains cross-instrument (~1.4 ppm) and is owed one exact "
            "upstream/evaluate.py n600 row. A smaller archive is NOT a score."
        ),
    }

    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in rungs.items():
        (args.out_dir / f"ddm_iv1_rung{name}_archive.zip").write_bytes(blob)
    manifest_path = args.out_dir / "ddm_iv1_rung_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote 4 rung archives + manifest -> {args.out_dir}")
    print(f"manifest sha256 = {_sha256(manifest_path.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
