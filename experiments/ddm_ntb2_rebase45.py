"""Rebase ntb2's `frame_even` HPAC candidate from move 44 onto the move-45 pointer.

Move 45 (`ddm_pc3_cap1_predictor_refit…`) changed the pose-CARRIER member only. MEASURED
here before anything is built: move 45's `hpac` (11,911 B), `tail` (119,909 B) and
`semantic` (29,862 B) members are BYTE-IDENTICAL to move 44's, and only `carrier`
(18,610 → 18,450 B) and the `header` moved. That is what makes this a rebase rather than a
re-measurement: the 600-frame RLC1 re-encode this arm already ran consumed the token field
and the HPAC prior, both unchanged, so its retained `hpac.twin*.br` and `tail.twin*.rider`
payloads are still the right bytes and are REUSED rather than recomputed.

Assembly mirrors `ddm_ntb2_hpac.encode` exactly -- same member split, same
`tail[:96] + rider` composition, same header length field -- and the CONTROL is the
falsifier: reassembling move 45's OWN members must reproduce its archive byte-identically
before any treatment archive is admitted.

Axis: [macOS-CPU advisory; exact bytes, output-lossless by construction]. No score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments import ddm_ntb2_control as control
from experiments import ddm_ntb2_hpac as hpac
from experiments import ddm_rlc1_run as landed

ROOT = control.ROOT.parent / "rebase45"
PROMOTED45 = control.ROOT / "promoted_runtime_move45"
POINTER45_SHA = "145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a"
POINTER45_BYTES = 180_246
# The members this lever is allowed to move, and the ones it must leave alone.
CHANGED_MEMBERS = ("hpac", "tail", "header")
UNTOUCHED_MEMBERS = ("semantic", "carrier")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def retain(path: Path, payload: bytes) -> dict:
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside the rebase store")
    if shutil.disk_usage(ROOT.parent).free < (2 << 30) + len(payload):
        raise RuntimeError("STORAGE_BLOCK: keep all existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ntb2 rebase45 payload")
    return landed.fact(path)


def record(path: Path, value: dict) -> dict:
    retain(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def assemble(member: dict, hpac_section: bytes, rider: bytes) -> bytes:
    """move 45's members with this lever's HPAC prior and its re-encoded tail.

    Mirrors `ddm_ntb2_hpac.encode`: the tail keeps its 96-byte prefix and takes the new
    rider, and the header's HPAC length field follows the new section.
    """
    from runtime import residual_archive as rx

    changed = dict(member)
    changed["hpac"] = hpac_section
    changed["tail"] = member["tail"][:96] + rider
    fields = list(rx.RX1_MODEL_HEADER.unpack(changed["header"]))
    fields[5] = len(hpac_section)
    changed["header"] = rx.RX1_MODEL_HEADER.pack(*fields)
    return landed.io.join_member(changed)


def run(treatment: str) -> dict:
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    live = pointer["our_local_frontier_contest_cuda"]
    if live["archive_sha256"] != POINTER45_SHA:
        raise SystemExit(f"pointer is not move 45: {live['archive_sha256']}")
    ROOT.mkdir(parents=True, exist_ok=True)
    work = ROOT / treatment
    runtime45 = PROMOTED45
    archive45 = runtime45 / "archive.zip"
    if landed.fact(archive45)["sha256"] != POINTER45_SHA:
        raise SystemExit("staged move-45 tree is not the pointer archive")

    sys.path.insert(0, str(runtime45))
    try:
        member45 = landed.io.split_member(landed.io.read_archive_member(archive45))
        member44 = landed.io.split_member(
            landed.io.read_archive_member(control.ROOT / "promoted_runtime/archive.zip")
        )
        # The rebase premise, measured, not assumed.
        premise = {
            key: {
                "move45_bytes": len(member45[key]),
                "move44_bytes": len(member44[key]),
                "identical": member45[key] == member44[key],
            }
            for key in sorted(set(member45) | set(member44))
        }
        for key in ("hpac", "tail", "semantic"):
            if not premise[key]["identical"]:
                raise SystemExit(f"REBASE PREMISE FALSE: move 45 moved {key}")

        landed.ROOT = ROOT
        archives = []
        for twin in range(2):
            if treatment == "control":
                section = member45["hpac"]
                rider = member45["tail"][96:]
            else:
                retained = hpac.ROOT / treatment / "retained"
                section = (retained / f"hpac.twin{twin}.br").read_bytes()
                rider = (retained / f"tail.twin{twin}.rider").read_bytes()
            body = assemble(member45, section, rider)
            retain(work / f"member.twin{twin}.bin", body)
            out = work / f"archive.twin{twin}.zip"
            landed.pack(body, out, "stored", None)
            from runtime import residual_archive as rx

            parsed = rx.read_residual_archive(out)
            base = rx.read_residual_archive(archive45)
            for field in ("semantic_blob", "carrier_blob", "tc1_weights", "residual_payload"):
                if getattr(parsed, field) != getattr(base, field):
                    raise SystemExit(f"rebase moved an unauthorized section: {field}")
            archives.append(landed.fact(out))
        if archives[0]["sha256"] != archives[1]["sha256"]:
            raise SystemExit("rebase twins differ")
        if treatment == "control" and archives[0]["sha256"] != POINTER45_SHA:
            raise SystemExit(
                "REBASE CONTROL FAILED: reassembling move 45's own members does not "
                "reproduce its archive; no treatment archive is admissible"
            )
    finally:
        sys.path.pop(0)

    result = {
        "schema": "ddm_ntb2_rebase45.v1",
        "axis": "[macOS-CPU advisory; exact bytes, output-lossless by construction]",
        "score_claim": False,
        "promotion_eligible": False,
        "treatment": treatment,
        "pointer": {"archive_sha256": POINTER45_SHA, "archive_bytes": POINTER45_BYTES,
                    "score": live["score"], "lane_id": live["lane_id"]},
        "rebase_premise": premise,
        "reused_encode": (
            "the 600-frame RLC1 re-encode consumed the token field and the HPAC prior, both "
            "byte-identical between move 44 and move 45, so its retained payloads are reused"
            if treatment != "control" else "n/a"),
        "twins": archives,
        "archive_bytes": archives[0]["bytes"],
        "delta_bytes_vs_move45": archives[0]["bytes"] - POINTER45_BYTES,
        "delta_s": (archives[0]["bytes"] - POINTER45_BYTES) * 25 / 37_545_489,
        "projected_score": live["score"] + (archives[0]["bytes"] - POINTER45_BYTES) * 25 / 37_545_489,
        "distortion_argument": (
            "the HPAC section is the arithmetic coder's PRIOR, so decoded symbols are unchanged "
            "and d_seg/d_pose are unchanged by construction; the cold parse-back turns that into "
            "a receipt"),
    }
    return record(work / "REBASE.json", result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", default="frame_even")
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.resume_from is not None and args.resume_from.resolve() != ROOT.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    result = run(args.treatment)
    print(json.dumps({k: v for k, v in result.items() if k != "rebase_premise"},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
