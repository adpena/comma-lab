"""Reproduce move44 with the landed RLC1 encoder, retaining every checkpoint.

Research only. The inherited third encoder independently checks the move43
pre-rider stream; the two RLC1 encoders must reproduce move44 itself. Cached
pre-mixer rows are valid only for this unchanged-model control, never pruning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_rlc1_run as encoder
from experiments import ddm_rlc4_rebase as rebase

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ntb2_non_tail_lossy_levers/control")
ANCESTOR = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43")
BASE_SHA = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"


def retain(destination, payload):
    """Keep all bytes and refuse insufficient storage; no cleanup discards evidence."""
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside owned store")
    if shutil.disk_usage(ROOT.parent).free < (40 << 30) + len(payload):
        raise RuntimeError("STORAGE_BLOCK: preserve bytes")
    return encoder.blob(destination, payload)


def copy_bound(source, destination):
    """Copy immutable inputs and verify their content hashes."""
    src = encoder.fact(source)
    dst = retain(destination, source.read_bytes())
    if any(src[k] != dst[k] for k in ("bytes", "sha256")):
        raise ValueError("input copy mismatch")
    return {"source": src, "copy": dst}


def prepare():
    """Copy and bind the shipped runtime and all 600 source trace frames."""
    ROOT.mkdir(parents=True, exist_ok=True)
    encoder.ROOT = ROOT
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["our_local_frontier_contest_cuda"]["archive_sha256"] != BASE_SHA:
        raise ValueError("POINTER_MOVED: rebind before launch")
    receipt_path = ANCESTOR / "encode/RESULT.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt["full_n600"] or not receipt["twins_identical"]:
        raise ValueError("ancestor encode incomplete")
    bound = receipt["binding"]["binding"]
    for item in [bound["field"], bound["config"], bound["native"], *bound["sources"]]:
        if encoder.fact(item["path"]) != item:
            raise ValueError("ancestor source or payload drift: " + item["path"])
    pins = {"encoder_receipt": encoder.fact(receipt_path), "source_files": {}}
    for source in sorted((ANCESTOR / "candidate_runtime").rglob("*")):
        if source.is_file() and source.suffix in (".py", ".c", ".sh", ".zip", ".json", ".txt", ".md"):
            rel = source.relative_to(ANCESTOR / "candidate_runtime")
            pins["source_files"][str(rel)] = copy_bound(source, ROOT / "source_runtime" / rel)
    if encoder.fact(ROOT / "source_runtime/archive.zip")["sha256"] != BASE_SHA:
        raise ValueError("wrong base archive")
    for name in ("config.bin", "geometry.dylib", "source.rc64"):
        pins[name] = copy_bound(ANCESTOR / "retained" / name, ROOT / "retained" / name)
    pins["field"] = copy_bound(ANCESTOR / "move42/field.u8", ROOT / "trace_input/field.u8")
    if pins["field"]["copy"]["sha256"] != FIELD_SHA:
        raise ValueError("wrong source field")
    frame_receipts = []
    for frame in range(600):
        src = ANCESTOR / "move42/trace/frames" / f"frame_{frame:04d}.npz"
        original = json.loads(src.with_suffix(".json").read_text())
        if encoder.fact(src) != original["payload"] or original["source_field_sha256"] != FIELD_SHA:
            raise ValueError("trace custody mismatch")
        dest = ROOT / "trace_input/trace/frames" / src.name
        copied = copy_bound(src, dest)
        updated = {**original, "payload": copied["copy"], "original_receipt": encoder.fact(src.with_suffix(".json"))}
        retain(dest.with_suffix(".json"), (json.dumps(updated, sort_keys=True, indent=2) + "\n").encode())
        frame_receipts.append(copied)
    pins.update(
        frames=frame_receipts,
        producer=encoder.fact(Path(__file__)),
        landed_encoder=encoder.fact(Path(encoder.__file__)),
        landed_rebase=encoder.fact(Path(rebase.__file__)),
        upstream=encoder.fact(REPO / "upstream/evaluate.py"),
        seed=encoder.SEED,
        axis="[macOS-CPU advisory; exact bytes, scorer-free]",
        score_claim=False,
        cleanup="KEEP all payloads; per-frame immutable state; 40 GiB reserve; inherited 8 GiB arm cap",
    )
    retain(ROOT / "INPUTS.json", (json.dumps(pins, sort_keys=True, indent=2) + "\n").encode())
    return pins


def run():
    """Invoke the landed triple encoder and compare RLC1 twins to shipped bytes."""
    pins = prepare()
    encoder.ROOT, encoder.SOURCE, encoder.TRACE = ROOT, ROOT / "source_runtime", ROOT / "trace_input"
    encoder.ARCHIVE_SHA, encoder.FIELD_SHA = BASE_SHA, FIELD_SHA
    # Same explicit binding mechanism as ddm_rlc4_rebase.encode; no coder rewrite.
    encoder.prepare = lambda: pins
    result = encoder.encode(600)
    io = encoder.io
    member = io.read_archive_member(ROOT / "source_runtime/archive.zip")
    sections = io.split_member(member)
    for twin in range(2):
        rider = (ROOT / f"encode/twin{twin}.rider").read_bytes()
        if rider != sections["tail"][96:]:
            raise ValueError("RLC1_REPRODUCTION_FAILED: no pruning prices allowed")
        rebuilt = b"".join(sections[k] for k in ("header", "hpac", "semantic", "carrier"))
        rebuilt += sections["tail"][:96] + rider
        retain(ROOT / f"retained/member.twin{twin}.bin", rebuilt)
        path = ROOT / f"retained/archive.twin{twin}.zip"
        encoder.pack(rebuilt, path, "stored", None)
        if encoder.fact(path)["sha256"] != BASE_SHA:
            raise ValueError("full archive reproduction failed")
    proof = {
        "encoder_result": result,
        "base_archive": encoder.fact(ROOT / "source_runtime/archive.zip"),
        "twins": [encoder.fact(ROOT / f"retained/archive.twin{i}.zip") for i in range(2)],
        "source_rider_byte_identical": True,
        "full_archive_byte_identical": True,
        "n": 600,
        "score_claim": False,
    }
    retain(ROOT / "REPRODUCTION.json", (json.dumps(proof, sort_keys=True, indent=2) + "\n").encode())
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("wrong resume root")
    print(json.dumps(run()), flush=True)


if __name__ == "__main__":
    main()
