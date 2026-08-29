#!/usr/bin/env python3
"""Build and price pose-free field-for-coder edits on the live jt21 body.

The screen is deliberately narrow.  It starts from positions where the shipped
token is wrong against the pinned GT argmax field and the shipped DX2 HPAC coding
argmax is right.  Those edits are label-level BENEFIT with HARM=0 by exact B/H/W
classification; their realised SegNet and PoseNet effects remain unmeasured until
the receiver and both frozen scorers run.

No entropy estimate is accepted as a byte price.  ``prepare`` writes three
disjoint, seeded, transition-and-time-stratified edit payloads plus their union.
The caller must pass every payload through ``ddm_jg2_tail_reencode.py`` against
the staged jt21 runtime.  ``summarize`` refuses unless the full n600 inverse-coder
control is byte-identical and every candidate receipt is a real joint re-encode.

All payloads are retained on APDataStore.  This module does not load a scorer and
does not claim distortion or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tac.candidate_seal import CONSISTENT, check_pin_consistency, repin_receiver

AP_ROOT = Path("/Volumes/APDataStore/pact")
REPO = Path(__file__).resolve().parents[1]
STORE = AP_ROOT / "ddm_fcd1_field_for_coder_diagonal"
SOURCE_RUNTIME = AP_ROOT / "ddm_gb1_groupbin8_conditioning" / "runtime_joint21"
JT21_ARCHIVE = AP_ROOT / "ddm_gb1_groupbin8_conditioning" / "retained" / "candidate_gb1_joint21.zip"
TOKENS = (
    AP_ROOT
    / "ddm_tb2_token_bit_attribution"
    / "measurement_v1"
    / "retained"
    / "fields"
    / "decoded_tokens_instrumented.u8"
)
CODING_ARGMAX = (
    AP_ROOT / "ddm_df1_dddb_field" / "measurement_v1" / "retained" / "fields" / "position_coding_argmax.u8.bin"
)
GT_ARGMAX = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")

PINS = {
    "jt21_archive": (180_192, "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3"),
    "tokens": (117_964_800, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "coding_argmax": (117_964_800, "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e"),
    "gt_argmax": (117_964_928, "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"),
}

N, H, W = 600, 384, 512
PLANE = H * W
RECEIVER_RAW_BYTES = 2 * N * 874 * 1164 * 3
SEED = 1_295
BATCHES = ("batch0", "batch1", "batch2")
ALL_CANDIDATES = (*BATCHES, "union")
MINIMUM_FREE_BYTES = 8 << 30
AXIS = "[macOS-CPU advisory / scorer-free exact B/H labels and real re-encode bytes]"
S_PER_BYTE = 25.0 / 37_545_489.0
REALIZED_FLIP_BREAKEVEN_PER_BYTE = 0.785
PYTHON_CORRECTOR_MARKER = "# ddm_fcd1: generation-21 corrector is Python-bound until its C port lands."
NATIVE_CORRECTOR_BUILD = """if [[ -n "${F26_CORRECTOR_NATIVE_LIBRARY:-}" ]]; then
  [[ -f "$F26_CORRECTOR_NATIVE_LIBRARY" ]] || { echo "missing F26 corrector library" >&2; exit 69; }
elif "${CC:-cc}" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\
       "$HERE/runtime/f26_corrector_native.c" -lm \\
       -o "$BUILD_DIR/f26_corrector_native.so" 2>/dev/null; then
  export F26_CORRECTOR_NATIVE_LIBRARY="$BUILD_DIR/f26_corrector_native.so"
else
  echo "f26 corrector native build unavailable; using the python corrector" >&2
fi
"""
PYTHON_CORRECTOR_SELECTION = f"""{PYTHON_CORRECTOR_MARKER}
# The retained jt21 stream was encoded by FreeCorrector with the 21-family Python
# configuration.  Its older native C table intentionally refuses that configuration.
# Force the semantically matching public fallback instead of silently compiling a
# different generation.  This costs runtime only; the real receiver proof decides use.
unset F26_CORRECTOR_NATIVE_LIBRARY
echo "ddm_fcd1 generation-21 runtime: using the Python corrector" >&2
"""


class Fcd1Error(RuntimeError):
    """A custody, identity, or real-price gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def require_pin(path: Path, key: str) -> dict[str, Any]:
    size, digest = PINS[key]
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise Fcd1Error(f"custody pin failed for {key}: {path}")
    return file_fact(path)


def preflight(store: Path) -> dict[str, Any]:
    if not store.resolve().is_relative_to(AP_ROOT.resolve()):
        raise Fcd1Error(f"store must remain under {AP_ROOT}: {store}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise Fcd1Error(f"storage preflight: {free} B free < {MINIMUM_FREE_BYTES} B")
    if not (SOURCE_RUNTIME / "inflate.py").is_file():
        raise Fcd1Error(f"jt21 runtime missing: {SOURCE_RUNTIME}")
    return {
        "storage": {
            "path": str(store),
            "free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
        },
        "inputs": {
            "jt21_archive": require_pin(JT21_ARCHIVE, "jt21_archive"),
            "tokens": require_pin(TOKENS, "tokens"),
            "coding_argmax": require_pin(CODING_ARGMAX, "coding_argmax"),
            "gt_argmax": require_pin(GT_ARGMAX, "gt_argmax"),
        },
    }


def bind_generation21_python_corrector(runtime: Path) -> dict[str, Any]:
    """Make a copied jt21 runtime select the exact corrector used by its encoder."""
    inflate = runtime / "inflate.sh"
    before = file_fact(inflate)
    text = inflate.read_text()
    changed = False
    if PYTHON_CORRECTOR_MARKER not in text:
        if text.count(NATIVE_CORRECTOR_BUILD) != 1:
            raise Fcd1Error(f"native corrector build block drifted in {inflate}")
        text = text.replace(NATIVE_CORRECTOR_BUILD, PYTHON_CORRECTOR_SELECTION)
        temporary = inflate.with_suffix(".sh.partial")
        temporary.write_text(text)
        temporary.chmod(inflate.stat().st_mode)
        os.replace(temporary, inflate)
        changed = True
    if text.count(PYTHON_CORRECTOR_MARKER) != 1 or NATIVE_CORRECTOR_BUILD in text:
        raise Fcd1Error(f"generation-21 Python-corrector binding failed in {inflate}")
    return {
        "backend": "runtime.free_corrector.FreeCorrector",
        "reason": "jt21 encoder used the 21-family Python config; its older native C generation refuses it",
        "before": before,
        "after": file_fact(inflate),
        "changed": changed,
    }


def stage_runtime(source: Path, archive: Path, destination: Path) -> dict[str, Any]:
    """Copy, stage, and re-pin without mutating either source object."""
    if destination.exists():
        verdict = check_pin_consistency(destination)
        if verdict.verdict != CONSISTENT:
            raise Fcd1Error(f"existing runtime is not pin-consistent: {verdict.summary()}")
        if sha256_file(destination / "archive.zip") != sha256_file(archive):
            raise Fcd1Error(f"existing runtime names different archive bytes: {destination}")
        corrector = bind_generation21_python_corrector(destination)
        return {
            "runtime": str(destination),
            "archive": file_fact(destination / "archive.zip"),
            "pin_consistency": verdict.verdict,
            "corrector": corrector,
            "resumed": True,
        }

    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise Fcd1Error(f"partial runtime exists; inspect before retry: {temporary}")
    source_inflate = file_fact(source / "inflate.sh")
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    shutil.copy2(archive, temporary / "archive.zip")
    corrector = bind_generation21_python_corrector(temporary)
    repin = repin_receiver(temporary)
    verdict = check_pin_consistency(temporary)
    if verdict.verdict != CONSISTENT:
        raise Fcd1Error(f"repinned runtime refused: {verdict.summary()}")
    os.replace(temporary, destination)
    corrector["before"] = source_inflate
    corrector["after"] = file_fact(destination / "inflate.sh")
    return {
        "runtime": str(destination),
        "archive": file_fact(destination / "archive.zip"),
        "pin_consistency": verdict.verdict,
        "corrector": corrector,
        "repin_changed": repin.changed,
        "resumed": False,
    }


def classify_pool(tokens: np.ndarray, argmax: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    changed = tokens != argmax
    return {
        "benefit": changed & (tokens != gt) & (argmax == gt),
        "harm": changed & (tokens == gt) & (argmax != gt),
        "wash": changed & (tokens != gt) & (argmax != gt),
    }


def stratified_assign(coords: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Seeded round-robin within (60-frame block, old->new) strata."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise Fcd1Error(f"coords must have shape (n,3), got {coords.shape}")
    assignment = np.full(coords.shape[0], -1, dtype=np.int8)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (frame, _row, _column) in enumerate(coords):
        groups[(int(frame) // 60, int(old[index]), int(new[index]))].append(index)
    for stratum, indices in sorted(groups.items()):
        local_seed = SEED + stratum[0] * 101 + stratum[1] * 17 + stratum[2]
        rng = np.random.default_rng(local_seed)
        order = np.asarray(indices, dtype=np.int64)
        rng.shuffle(order)
        offset = int(rng.integers(0, len(BATCHES)))
        assignment[order] = (np.arange(order.size, dtype=np.int64) + offset) % len(BATCHES)
    if np.any(assignment < 0):
        raise Fcd1Error("stratified assignment left coordinates unassigned")
    return assignment


def write_candidate_payload(
    *,
    name: str,
    selected: np.ndarray,
    coords: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    tokens: np.ndarray,
    retained: Path,
) -> dict[str, Any]:
    chosen = np.flatnonzero(selected)
    chosen_coords = coords[chosen].astype(np.int32, copy=False)
    chosen_old = old[chosen].astype(np.uint8, copy=False)
    chosen_new = new[chosen].astype(np.uint8, copy=False)
    coord_path = retained / "coordinates" / f"{name}.frame_y_x_old_new.npz"
    atomic_npz(coord_path, coords=chosen_coords, old=chosen_old, new=chosen_new)

    by_pair: dict[int, list[int]] = defaultdict(list)
    for local, row in enumerate(chosen_coords):
        by_pair[int(row[0])].append(local)
    edit_planes: dict[str, np.ndarray] = {}
    for pair, local_indices in sorted(by_pair.items()):
        plane = np.asarray(tokens[pair], dtype=np.uint8).copy()
        local = np.asarray(local_indices, dtype=np.int64)
        locations = chosen_coords[local]
        plane[locations[:, 1], locations[:, 2]] = chosen_new[local]
        edit_planes[str(pair)] = plane
    edit_path = retained / "edits" / f"{name}.edits.npz"
    atomic_npz(edit_path, **edit_planes)

    field_path = retained / "fields" / f"{name}.tokens.u8"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = field_path.with_suffix(field_path.suffix + ".partial")
    with temporary.open("wb") as handle:
        for pair in range(N):
            plane = edit_planes.get(str(pair))
            if plane is None:
                plane = np.asarray(tokens[pair], dtype=np.uint8)
            handle.write(np.ascontiguousarray(plane, dtype=np.uint8).tobytes())
    os.replace(temporary, field_path)
    if field_path.stat().st_size != N * PLANE:
        raise Fcd1Error(f"candidate field has wrong size: {field_path}")

    transitions = Counter(f"{int(a)}->{int(b)}" for a, b in zip(chosen_old, chosen_new, strict=True))
    return {
        "name": name,
        "tokens_changed": int(chosen.size),
        "active_pairs": len(by_pair),
        "B_benefit_exact": int(chosen.size),
        "H_harm_exact": 0,
        "W_wash_exact": 0,
        "temporal_blocks_touched": sorted({int(row[0]) // 60 for row in chosen_coords}),
        "class_transitions": dict(sorted(transitions.items())),
        "coordinates": file_fact(coord_path),
        "edits": file_fact(edit_path),
        "candidate_field": file_fact(field_path),
        "distortion_status": "UNMEASURED_PENDING_REAL_RECEIVER_AND_FROZEN_SCORERS",
    }


def run_prepare(args: argparse.Namespace) -> int:
    store = Path(args.store)
    pre = preflight(store)
    retained = store / "retained"
    tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))
    argmax = np.memmap(CODING_ARGMAX, dtype=np.uint8, mode="r", shape=(N, H, W))
    gt = np.load(GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, H, W) or gt.dtype != np.uint8:
        raise Fcd1Error(f"GT field shape/dtype drifted: {gt.shape} {gt.dtype}")

    totals = Counter()
    coordinates: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    for pair in range(N):
        classes = classify_pool(tokens[pair], argmax[pair], gt[pair])
        for key, mask in classes.items():
            totals[key] += int(np.count_nonzero(mask))
        yy, xx = np.nonzero(classes["benefit"])
        if yy.size:
            coordinates.append(
                np.stack(
                    [np.full(yy.size, pair, dtype=np.int32), yy.astype(np.int32), xx.astype(np.int32)],
                    axis=1,
                )
            )
            old_parts.append(np.asarray(tokens[pair][yy, xx], dtype=np.uint8))
            new_parts.append(np.asarray(argmax[pair][yy, xx], dtype=np.uint8))
    coords = np.concatenate(coordinates, axis=0)
    old = np.concatenate(old_parts)
    new = np.concatenate(new_parts)
    assignment = stratified_assign(coords, old, new)

    pool_path = retained / "coordinates" / "benefit_pool.frame_y_x_old_new_assignment.npz"
    atomic_npz(pool_path, coords=coords, old=old, new=new, assignment=assignment)
    rows = []
    for index, name in enumerate(BATCHES):
        rows.append(
            write_candidate_payload(
                name=name,
                selected=assignment == index,
                coords=coords,
                old=old,
                new=new,
                tokens=tokens,
                retained=retained,
            )
        )
    rows.append(
        write_candidate_payload(
            name="union",
            selected=np.ones(coords.shape[0], dtype=bool),
            coords=coords,
            old=old,
            new=new,
            tokens=tokens,
            retained=retained,
        )
    )

    base_runtime = stage_runtime(SOURCE_RUNTIME, JT21_ARCHIVE, store / "runtimes" / "base_jt21")
    result = {
        "schema": "ddm_fcd1_prepare.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "seed": SEED,
        "implementation": {
            "field_generator": file_fact(Path(__file__)),
            "real_reencoder": file_fact(REPO / "experiments" / "ddm_jg2_tail_reencode.py"),
            "fresh_schur_compiler": file_fact(REPO / "experiments" / "ddm_fcd1_incompile_schur.py"),
        },
        "preflight": pre,
        "selection": {
            "law": "token!=GT and coding_argmax==GT",
            "source_coding_row": "DX2 DF1 full-field shipped coding argmax; final prices come only from jt21 real re-encodes",
            "population_B": int(totals["benefit"]),
            "population_H": int(totals["harm"]),
            "population_W": int(totals["wash"]),
            "candidate_pool": file_fact(pool_path),
            "batching": "disjoint seeded round-robin within (60-frame block, old->new) strata",
        },
        "base_runtime": base_runtime,
        "candidates": rows,
        "next_stage": "run one n600 jg2 control and one n600 joint re-encode per candidate",
    }
    atomic_json(store / "PREPARE.json", result)
    print(
        json.dumps(
            {
                "B": totals["benefit"],
                "H": totals["harm"],
                "W": totals["wash"],
                "candidates": [(r["name"], r["tokens_changed"], r["active_pairs"]) for r in rows],
            },
            indent=2,
        )
    )
    return 0


def run_summarize(args: argparse.Namespace) -> int:
    store = Path(args.store)
    preflight(store)
    prepare_path = store / "PREPARE.json"
    if not prepare_path.is_file():
        raise Fcd1Error("PREPARE.json missing")
    prepare = json.loads(prepare_path.read_text())
    reencode = store / "reencode" / "retained"
    control_path = reencode / "S1_control_600.json"
    if not control_path.is_file():
        raise Fcd1Error(f"n600 inverse-coder control missing: {control_path}")
    control = json.loads(control_path.read_text())
    if not control.get("byte_identical"):
        raise Fcd1Error("n600 inverse-coder control is not byte-identical")

    source_by_name = {row["name"]: row for row in prepare["candidates"]}
    rows = []
    for name in ALL_CANDIDATES:
        receipt_path = reencode / f"S1_encode_fcd1_{name}.json"
        if not receipt_path.is_file():
            raise Fcd1Error(f"real re-encode receipt missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("delta_trustworthy") or receipt.get("frames") != N:
            raise Fcd1Error(f"real re-encode is not trustworthy n600: {receipt_path}")
        pointer = receipt.get("pointer_archive", {})
        if pointer.get("sha256") != PINS["jt21_archive"][1] or pointer.get("bytes") != PINS["jt21_archive"][0]:
            raise Fcd1Error(f"real re-encode was not based on the pinned jt21 bank: {receipt_path}")
        if receipt.get("tokens_changed") != source_by_name[name]["tokens_changed"]:
            raise Fcd1Error(f"edit count drift for {name}")
        archive = Path(receipt["candidate_archive"]["path"])
        if file_fact(archive) != receipt["candidate_archive"]:
            raise Fcd1Error(f"retained candidate archive drifted for {name}")
        runtime = stage_runtime(
            store / "runtimes" / "base_jt21",
            archive,
            store / "runtimes" / f"candidate_{name}",
        )
        delta = int(receipt["archive_delta_bytes"])
        exact_b = int(source_by_name[name]["B_benefit_exact"])
        rows.append(
            {
                "name": name,
                "independent_batch": name in BATCHES,
                "tokens_changed": int(receipt["tokens_changed"]),
                "active_pairs": source_by_name[name]["active_pairs"],
                "B_benefit_exact": exact_b,
                "H_harm_exact": 0,
                "real_joint_reencode_delta_bytes_vs_jt21": delta,
                "real_joint_reencode_delta_S_rate": delta * S_PER_BYTE,
                "exact_label_B_per_byte_credit": exact_b / -delta if delta < 0 else None,
                "realized_flip_breakeven_per_byte": REALIZED_FLIP_BREAKEVEN_PER_BYTE,
                "breakeven_status": "UNMEASURED_UNTIL_REALIZED_SEGNET_FLIPS",
                "candidate_archive": receipt["candidate_archive"],
                "runtime": runtime,
                "realized_d_seg": None,
                "compensated_d_pose": None,
                "net_delta_S": None,
                "disposition": "QUEUED_FOR_IN_COMPILE_SCHUR_AND_N600_SCORERS",
            }
        )
    fire_order = [
        row["name"]
        for row in sorted(rows, key=lambda item: item["real_joint_reencode_delta_bytes_vs_jt21"])
        if row["real_joint_reencode_delta_bytes_vs_jt21"] < 0
    ]
    receiver_proofs: dict[str, Any] = {}
    union_row = next(row for row in rows if row["name"] == "union")
    expected_receivers = {
        "base_jt21": (PINS["jt21_archive"][1], PINS["tokens"][1]),
        "union": (
            union_row["candidate_archive"]["sha256"],
            source_by_name["union"]["candidate_field"]["sha256"],
        ),
    }
    for receiver_name, (expected_archive_sha, expected_token_sha) in expected_receivers.items():
        decode_dir = store / "decode" / receiver_name
        receipt_path = decode_dir / "DECODE.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("score_claim") is not False:
            raise Fcd1Error(f"receiver receipt has an authority claim: {receipt_path}")
        if receipt.get("archive", {}).get("sha256") != expected_archive_sha:
            raise Fcd1Error(f"receiver archive binding drifted: {receipt_path}")
        raw = Path(receipt.get("raw", {}).get("path", ""))
        if not raw.is_file() or raw.stat().st_size != RECEIVER_RAW_BYTES or file_fact(raw) != receipt.get("raw"):
            raise Fcd1Error(f"receiver raw payload drifted: {receipt_path}")
        checkpoint_path = decode_dir / "inflated" / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("binding", {}).get("archive_sha256") != expected_archive_sha:
            raise Fcd1Error(f"receiver token checkpoint archive drifted: {checkpoint_path}")
        token_decoder = checkpoint.get("token_decoder", {})
        if token_decoder.get("free_corrector") != "FreeCorrector":
            raise Fcd1Error(f"receiver did not use the generation-21 Python corrector: {checkpoint_path}")
        if token_decoder.get("decoded_token_sha256") != expected_token_sha:
            raise Fcd1Error(f"receiver decoded the wrong token field: {checkpoint_path}")
        receiver_proofs[receiver_name] = {
            "decode_receipt": file_fact(receipt_path),
            "raw": receipt["raw"],
            "token_checkpoint": file_fact(checkpoint_path),
            "corrector": "FreeCorrector",
            "elapsed_seconds": receipt.get("elapsed_seconds"),
        }
    receiver_status = "PASS" if set(receiver_proofs) == set(expected_receivers) else "PENDING"
    boundary_correction_path = store / "RECEIVER_BOUNDARY_CORRECTION.json"
    atomic_json(
        boundary_correction_path,
        {
            "schema": "ddm_fcd1_receiver_boundary_correction.v1",
            "applies_to": {
                name: proof["decode_receipt"] for name, proof in receiver_proofs.items()
            },
            "incorrect_prior_axis_phrase": "real receiver + R + uint8 payload, no scorer",
            "correct_boundary": "real public receiver to retained uint8 raw; scorer preprocessing R and both frozen scorers were not run",
            "score_claim": False,
        },
    )
    result = {
        "schema": "ddm_fcd1_byte_only_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "control": file_fact(control_path),
        "rows": rows,
        "fire_order": fire_order,
        "receiver_status": receiver_status,
        "receiver_proofs": receiver_proofs,
        "receiver_boundary": "public receiver to retained uint8 raw passed; scorer preprocessing R and frozen scorers not run",
        "receiver_boundary_correction": file_fact(boundary_correction_path),
        "owner": "MAIN",
        "consumer_store": str(store),
        "fire_trigger": (
            "qbt2b r10 scorer claim terminal; then fresh in-compile Schur solve with pose gate and n600 both frozen scorers"
            if receiver_status == "PASS"
            else "base/union receiver proofs pass and qbt2b r10 scorer claim is terminal"
        ),
        "closure_status": "NOT_AVAILABLE_WITHOUT_N600_REALIZED_DISTORTION",
    }
    atomic_json(store / "BYTE_ONLY_RESULT.json", result)
    print(
        json.dumps(
            {
                "rows": [{"name": r["name"], "dB": r["real_joint_reencode_delta_bytes_vs_jt21"]} for r in rows],
                "fire_order": fire_order,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=STORE)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare").set_defaults(func=run_prepare)
    sub.add_parser("summarize").set_defaults(func=run_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
