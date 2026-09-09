"""Source custody and fail-closed launch preflight for bnd2; this is NOT an encoder.

Read the cmp2 tree's actual archive reader and preserve every extracted payload.
No scorer, native forward, field materialization, or new coded candidate runs here.
The priority probe is deliberately required: EPERM is a blocker, not best effort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as jg2

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code")
LIVE = Path("/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime")
RP1 = Path("/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion")
FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/retained/source/field.u8")
POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
AXIS = "[macOS-CPU scorer-free source-custody preflight]"
RESERVE = 40 * 1024**3


def fact(path: Path) -> dict:
    """Hash an actual file with bounded memory."""
    return jg2.file_fact(path)


def preserve(path: Path, payload: bytes) -> dict:
    """Persist once without overwriting a changed or stranded artifact."""
    if not path.resolve().is_relative_to(STORE):
        raise ValueError("write outside assigned bnd2 store")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable payload differs: {path}")
        return fact(path)
    if shutil.disk_usage(STORE.parent).free < RESERVE + len(payload):
        raise RuntimeError("40 GiB storage reserve would be crossed")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".{os.getpid()}.partial")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"concurrent immutable payload differs: {path}") from None
    # Only success-only scratch with identical published bytes is removed.
    # The returned path/size/hash and recorded argv retain its reproduction proof.
    if path.read_bytes() != payload:
        raise ValueError("published bytes changed; retain partial for diagnosis")
    partial.unlink()
    return fact(path)


def encode_json(value: object) -> bytes:
    """Canonical metadata only; no coding-price claim."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def run(root: Path) -> dict:
    """Verify source identity and record blockers without starting an experiment."""
    if root.resolve() != STORE:
        raise ValueError("--resume-from must name the assigned bnd2 store")
    free = shutil.disk_usage(STORE.parent).free
    if free < RESERVE + 4 * 1024**2:
        raise RuntimeError("insufficient storage for bounded source preflight")
    root.mkdir(parents=True, exist_ok=True)
    pointer_bytes = POINTER.read_bytes()
    pointer = json.loads(pointer_bytes)
    anchor = pointer["our_local_frontier_contest_cuda"]
    archive = fact(LIVE / "archive.zip")
    if archive["sha256"] != anchor["archive_sha256"]:
        raise RuntimeError("configured cmp2 tree is no longer the pointer")
    retained = root / "retained"
    copies = {"pointer.json": preserve(retained / "pointer.json", pointer_bytes)}
    for name, path in (
        ("archive.zip", LIVE / "archive.zip"),
        ("rank.json", RP1 / "rank_mixer/RANK.json"),
        ("filtered_candidates.npz", RP1 / "rank_mixer/candidates.npz"),
    ):
        copies[name] = preserve(retained / name, path.read_bytes())

    # Import read-only source, never the mutable sister experiment module.
    residual, _, _ = jg2.load_runtime(LIVE)
    if Path(residual.__file__).resolve() != LIVE / "runtime/residual_archive.py":
        raise RuntimeError("wrong runtime module imported")
    if copies["archive.zip"]["sha256"] != archive["sha256"]:
        raise RuntimeError("source archive changed during the copy")
    parts = residual.read_residual_archive(retained / "archive.zip")
    decoded_components = {}
    for name in (
        "semantic_blob",
        "carrier_blob",
        "hpac_blob",
        "residual_payload",
        "compressed_models",
        "compensation_blob",
    ):
        value = getattr(parts, name)
        if value is not None:
            decoded_components[name] = preserve(retained / f"reader_{name}.bin", value)
    stream = preserve(retained / "shipped_token_stream.rc64", parts.token_stream)
    weights = preserve(retained / "tc1_weights.bin", bytes(parts.tc1_weights))
    envelope_bytes = (RP1 / "rank_mixer/tail_rp1_mixer_control.bin").read_bytes()
    envelope = preserve(retained / "rank_control.envelope", envelope_bytes)
    expected = b"R6D1" + parts.token_stream
    expected += b"\0" * ((-len(expected)) % 4)
    reconstructed = preserve(retained / "reader_rewrapped.envelope", expected)
    if envelope_bytes != expected:
        raise RuntimeError("RP1 envelope is not the shipped reader stream plus R6D1 padding")
    rank = json.loads((retained / "rank.json").read_text())
    field = fact(FIELD)
    # This hash identifies the complete pass-4 field, not the ancestor TC1 field.
    if field["sha256"] != "361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94":
        raise RuntimeError("pass-4 source field changed")
    control = rank["identity_control"]
    if rank["schema"] != "ddm_rp1_rank_mixer.v1" or control["byte_identical"] is not True:
        raise RuntimeError("retained RP1 schema/control is invalid")
    if control["frames_encoded"] != 600:
        raise RuntimeError("retained RP1 control is not n600")
    for prefix in ("emitted", "cmp1_stream"):
        if control[prefix + "_bytes"] != envelope["bytes"] or control[prefix + "_sha256"] != envelope["sha256"]:
            raise RuntimeError("rank control is not bound to retained envelope")
    if any(rank["stream"][key] != envelope[key] for key in ("bytes", "sha256")):
        raise RuntimeError("rank stream custody differs")
    census = rank["census"]
    if (
        census["total_tokens"] != 600 * 384 * 512
        or not 0 <= census["n_symbol_is_coder_argmax"] <= census["total_tokens"]
        or sum(census["bin_counts"]) != census["total_tokens"]
    ):
        raise RuntimeError("rank census denominator differs")
    with np.load(retained / "filtered_candidates.npz", allow_pickle=False) as data:
        columns = {key: data[key] for key in data.files}
    frame, pos, sym, best = (columns[key] for key in ("frame", "pos", "sym", "best"))
    count = len(frame)
    if (
        any(value.shape != (count,) for value in columns.values())
        or count != rank["candidates_dump"]["rows"]
        or any(value.dtype.kind not in "ui" for value in (frame, pos, sym, best))
        or np.any((frame < 0) | (frame >= 600))
        or np.any((pos < 0) | (pos >= 384 * 512))
        or np.any((sym < 0) | (sym >= 5) | (best < 0) | (best >= 5) | (sym == best))
    ):
        raise RuntimeError("filtered candidate schema or coordinates differ")
    flat_ids = frame.astype(np.int64) * (384 * 512) + pos
    if np.unique(flat_ids).size != count:
        raise RuntimeError("duplicate filtered candidate locations")
    mapped = np.memmap(FIELD, dtype=np.uint8, mode="r")
    if mapped.size != census["total_tokens"] or not np.array_equal(mapped[flat_ids], sym):
        raise RuntimeError("filtered symbols do not join to the shipped pass-4 field")
    misses = census["total_tokens"] - census["n_symbol_is_coder_argmax"]
    if count > misses:
        raise RuntimeError("filtered rows exceed total coder misses")

    priority = {"required": 10, "before": os.getpriority(os.PRIO_PROCESS, 0)}
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
        priority["after"] = os.getpriority(os.PRIO_PROCESS, 0)
        priority["satisfied"] = priority["after"] == 10
    except OSError as error:
        priority.update(after=os.getpriority(os.PRIO_PROCESS, 0), satisfied=False, errno=error.errno, error=str(error))
    blockers = [] if priority["satisfied"] else ["NICE_10_DENIED"]
    # No production encoder is claimed merely because source preflight succeeded.
    result = {
        "schema": "ddm_bnd2_source_preflight.v1",
        "axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "charter_complete": False,
        "source_preflight_passed": True,
        "launch_constraints_satisfied": not blockers,
        "encoder_implemented": False,
        "new_encode_rows": 0,
        "new_scorer_pairs": 0,
        "new_misprediction_locations": 0,
        "new_shipped_field_decode": False,
        "retained_locations_revalidated": count,
        "source_total_mispredictions": misses,
        "unlocated_mispredictions": misses - count,
        "recovered_census_complete": False,
        "blockers": blockers,
        "priority": priority,
        "archive": archive,
        "field": field,
        "retained": copies,
        "raw_stream": stream,
        "weights": weights,
        "control_envelope": envelope,
        "reconstructed_envelope": reconstructed,
        "envelope_identity": True,
        "envelope_overhead_bytes": len(expected) - len(parts.token_stream),
        "reader_components": decoded_components,
        "retained_rank_census": rank["census"],
        "source_reader": fact(LIVE / "runtime/residual_archive.py"),
        "imported_jg2": fact(Path(jg2.__file__)),
        "producer": fact(Path(__file__)),
        "upstream_evaluate": fact(REPO / "upstream/evaluate.py"),
        "archive_bytes": archive["bytes"],
        "pointer_anchor": anchor,
        "free_bytes_before": free,
        "storage_reserve_bytes": RESERVE,
        "retention_policy": "all extracted bytes kept; no deletion or movement",
        "argv": sys.argv,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "seed": None,
        "randomness": "none",
    }
    mirror = json.loads((REPO / anchor["source_path"]).read_text())
    recomputed = (
        100 * mirror["avg_segnet_dist"]
        + math.sqrt(10 * mirror["avg_posenet_dist"])
        + 25 * archive["bytes"] / 37_545_489
    )
    if mirror["archive_sha256"] != archive["sha256"] or abs(recomputed - anchor["score"]) > 1e-14:
        raise RuntimeError("existing exact-anchor component arithmetic or identity differs")
    result["existing_anchor_recomputed_score"] = recomputed
    encoded = encode_json(result)
    receipt_id = hashlib.sha256(encoded).hexdigest()[:16]
    preserve(retained / f"PREFLIGHT_{receipt_id}.json", encoded)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["launch_constraints_satisfied"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
