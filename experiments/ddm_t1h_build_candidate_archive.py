#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""t1h -- byte-close the re-solved carrier into a real archive, and prove it parses back.

The headroom sweep says what the re-solved codes buy; the pricer says what they cost.  This
builds the actual ``archive.zip`` so the claim stops being arithmetic about sections and
becomes bytes on disk that a receiver can read.

WHAT IS REPLACED, AND WHAT IS NOT.  Exactly one thing changes: the Rice residual payload
(and its two header fields) inside the packed CAP1 carrier.  The semantic weights, the HPAC
model, the token stream, the residual table, the 14 B frame-0 selector and the 36 B ``Q2C1``
compensation overlay are all carried through byte-identically.  ``d_seg`` is therefore
invariant by construction and not merely expected to be small: SegNet reads only the odd
frame (``upstream/modules.py:108``) and no odd frame is touched.

THE PROOF OBLIGATIONS, ALL ENFORCED HERE
----------------------------------------
1. **Parse-back.** Re-open the built archive with the shipped receiver, redo the carrier
   decode INCLUDING the selector split and the compensation overlay, and require the
   resulting signed-int12 receiver lattice to equal the intended candidate lattice exactly.
   A byte-close that the receiver reads differently is not a byte-close.
2. **Determinism.** Build twice into different paths and require identical sha256.
3. **Invariance.** Require every non-carrier section to be byte-identical to the shipped
   archive, by sha, and report the carrier delta explicitly.

None of this is a score.  It produces the bytes an exact evaluation would consume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COMPOSE_SOURCE = REPO / "experiments/ddm_t1h_compose_pass1.py"
PRICER_SOURCE = REPO / "experiments/ddm_t1h_carrier_byte_pricer.py"

ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip"
)
RUNTIME = Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime")
OUT = Path("/Volumes/APDataStore/pact/ddm_t1h/candidate")

N_PAIRS = 600
CARRIER_DIM = 12


def _load(source: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def zip_template(archive: Path) -> zipfile.ZipInfo:
    """Reuse the shipped member's ZipInfo so the container carries no timestamp of ours."""
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if names != ["p"]:
            raise SystemExit(f"expected a single member 'p', found {names}")
        info = handle.getinfo("p")
    fresh = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    fresh.compress_type = info.compress_type
    fresh.external_attr = info.external_attr
    fresh.internal_attr = info.internal_attr
    fresh.create_system = info.create_system
    fresh.create_version = info.create_version
    fresh.extract_version = info.extract_version
    fresh.flag_bits = info.flag_bits
    return fresh


def build_member(archive: Path, runtime: Path, candidate: np.ndarray, pricer, compose):
    """Splice the re-solved Rice payload into the member and return the new member bytes."""
    (carrier_repack, _cap1, predictor, _carrier_blob, _selector, _info, model,
     _base_codes, _compensation) = pricer.load_shipped(archive, runtime)
    residuals = pricer.forward_ar1(candidate, model, predictor)
    ks, payload, bits = carrier_repack._rice_encode(
        carrier_repack._zigzag(residuals), 1
    )
    priced = compose.counted_carrier_bytes(
        archive, runtime, payload, int(bits), ks.reshape(-1)
    )
    if not priced["packable"]:
        raise SystemExit("candidate is not packable: " + priced["blocker"])

    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from runtime import residual_archive as ra

    member = priced["member"]
    header = ra.RX1_MODEL_HEADER
    magic, version, codec, table_mode, reserved, hpac_b, sem_b, _car_b = header.unpack_from(
        member
    )
    offset = priced["stream_offset"]
    model_end = offset + priced["stream_bytes_before"]
    new_stream = priced["new_stream"]
    new_member = (
        header.pack(
            magic, version, codec, table_mode, reserved, hpac_b, sem_b, len(new_stream)
        )
        + member[header.size : offset]
        + new_stream
        + member[model_end:]
    )
    return new_member, priced


def parse_back(archive: Path, runtime: Path, expected_receiver_codes: np.ndarray) -> dict:
    """Read the built archive with the shipped receiver and compare the decoded lattice."""
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier
    from runtime.compensation_overlay import apply_compensation_overlay
    from runtime.residual_archive import read_residual_archive

    fx1_tree = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
    if str(fx1_tree) not in sys.path:
        sys.path.insert(0, str(fx1_tree))
    spec = importlib.util.spec_from_file_location(
        "fx1_renderer_t1h", fx1_tree / "inflate.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    sys.modules["fx1_renderer_t1h"] = renderer
    spec.loader.exec_module(renderer)

    parts = read_residual_archive(archive)
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical = materialize_cpr1(carrier_blob, renderer)
    basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
    _, _, _scales, encoded = renderer.decode_compact_carrier(
        canonical, basis_count=basis_count, frames=renderer.N,
        dimensions=renderer.CARRIER_DIM,
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    codes = np.cumsum(delta, axis=0) & 0xFFF
    codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
    if parts.compensation_blob is not None:
        codes = apply_compensation_overlay(codes, parts.compensation_blob)
    return {
        "receiver_lattice_matches_intent": bool(
            np.array_equal(codes, expected_receiver_codes)
        ),
        "mismatched_coordinates": int(
            np.count_nonzero(codes != expected_receiver_codes)
        ),
        "selector_bytes": 0 if selector_blob is None else len(selector_blob),
        "compensation_bytes": (
            0 if parts.compensation_blob is None else len(parts.compensation_blob)
        ),
        "sections": {
            "semantic_blob": _sha(parts.semantic_blob),
            "hpac_blob": _sha(parts.hpac_blob),
            "token_stream": _sha(parts.token_stream),
            "residual_payload": _sha(parts.residual_payload),
            "compensation_blob": (
                None if parts.compensation_blob is None
                else _sha(parts.compensation_blob)
            ),
        },
    }


def run(args) -> int:
    pricer = _load(PRICER_SOURCE, "t1h_pricer_build")
    compose = _load(COMPOSE_SOURCE, "t1h_compose_build")

    candidate = np.load(Path(args.candidate_base_codes)).astype(np.int32)
    receiver_codes = np.load(Path(args.candidate_receiver_codes)).astype(np.int32)
    if candidate.shape != (N_PAIRS, CARRIER_DIM):
        raise SystemExit(f"candidate has shape {candidate.shape}, expected (600, 12)")

    archive = Path(args.archive)
    runtime = Path(args.runtime)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    template = zip_template(archive)

    shipped = parse_back(archive, runtime, receiver_codes)  # sections of the SHIPPED archive
    member, priced = build_member(archive, runtime, candidate, pricer, compose)

    built = []
    for index in (1, 2):
        path = out / ("archive.zip" if index == 1 else "archive.repeat.zip")
        tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w") as handle:
            handle.writestr(template, member)
        tmp.replace(path)
        payload = path.read_bytes()
        built.append({"path": str(path), "bytes": len(payload), "sha256": _sha(payload)})

    # STANDALONE SWAPPABLE SECTION.  The packed CAP1 carrier section is the unit another
    # arm can drop into its own byte-close: it is byte-LENGTH-identical to the shipped one
    # (the receiver dispatches on that exact length), so composing it changes no offset.

    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))

    # DDM_UP3 FIX: the pinned-length read silently misparses any CK2-interleaved body
    # (to1/ck2 lineage).  The canonical helper un-interleaves and derives the portion.
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ddm_up3_carrier_splice import carrier_section_from_archive

    def packed_section(path: Path) -> bytes:
        return carrier_section_from_archive(path, runtime).packed

    section = {}
    for label, source in (("shipped", archive), ("candidate", out / "archive.zip")):
        packed = packed_section(source)
        path = out / f"carrier_section_{label}.bin"
        path.write_bytes(packed)
        section[label] = {"path": str(path), "bytes": len(packed), "sha256": _sha(packed)}
    section["byte_length_identical"] = (
        section["shipped"]["bytes"] == section["candidate"]["bytes"]
    )
    section["note"] = (
        "drop-in replacement for the packed CAP1 carrier section; same length, so every "
        "downstream offset is unchanged and only the Rice payload and its two header "
        "fields differ"
    )

    deterministic = built[0]["sha256"] == built[1]["sha256"]
    verified = parse_back(out / "archive.zip", runtime, receiver_codes)
    invariant = {
        name: shipped["sections"][name] == verified["sections"][name]
        for name in shipped["sections"]
    }

    receipt = {
        "schema": "ddm_t1h_candidate_archive.v1",
        "axis": "[EXACT bytes; no scorer was run by this tool]",
        "score_claim": False,
        "promotable": False,
        "base_archive": {"path": str(archive), "sha256": _sha(archive.read_bytes()),
                         "bytes": archive.stat().st_size},
        "candidate_archive": built[0],
        "determinism_repeat": {"path": built[1]["path"], "sha256": built[1]["sha256"],
                               "byte_identical": deterministic},
        "archive_delta_bytes": built[0]["bytes"] - archive.stat().st_size,
        "carrier_counted": {
            key: value for key, value in priced.items()
            if not isinstance(value, (bytes, bytearray))
        },
        "standalone_carrier_section": section,
        "parse_back": verified,
        "non_carrier_sections_byte_identical": invariant,
        "all_proofs_green": bool(
            deterministic
            and verified["receiver_lattice_matches_intent"]
            and all(invariant.values())
        ),
    }
    (out / "T1H_CANDIDATE_ARCHIVE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["all_proofs_green"]:
        raise SystemExit("BYTE-CLOSE FAILED: see the receipt above; not fireable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument(
        "--candidate-base-codes", type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_t1h/candidate_base_codes.int32.npy"),
    )
    parser.add_argument(
        "--candidate-receiver-codes", type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_t1h/candidate_receiver_codes.int32.npy"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
