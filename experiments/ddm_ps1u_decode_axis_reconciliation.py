#!/usr/bin/env python3
"""ddm_ps1u — CPU-vs-CUDA decode reconciliation for the hv1 frontier archive.

THE QUESTION
------------
Archive ``80d9c8c6…`` @182,759 B scores d_pose **6.88e-06** through the T4 chain
(``inflate_device_policy=auto`` ⇒ inflate ran on CUDA) and **1.4747e-04** through the
advisory CPU chain (``inflate`` on CPU).  ddm_ps1u eliminated scorer numerics (CPU≡CUDA
PoseNet, per-pair correlation 1.0000 on identical frames, GT bit-identical) and the metric
definition (``upstream/modules.py:82-84`` is a plain 6-dim MSE).  The surviving explanation
is that **the CPU and CUDA inflate paths emit different frames**.

That is not merely a scoring question.  Our deterministic-decode non-negotiable requires
"same ``archive.zip`` → bit-identical inflate output every run/host".  If decode is
device-dependent, that is a compliance-grade vehicle defect independent of which axis is
"right".

WHAT THIS MODULE DOES
---------------------
``manifest``  — per-frame SHA-256 over the raw uint8 decode (1,200 frames → 600 pairs of
                f0/f1), plus the aggregate raw SHA-256 and full geometry pins.  Hashing the
                RAW FRAMES (not preprocessed scorer tensors) is deliberate: the raw decode
                is exactly the object the determinism rule governs, and it keeps scorer
                preprocessing out of the comparison as a possible confound.
``diff``      — adjudicates two manifests.  IDENTICAL ⇒ decode is deterministic across the
                axes and the discrepancy must be re-hunted elsewhere.  DIFFERENT ⇒
                device-dependent decode CONFIRMED; the report enumerates which pairs and
                which frame (f0/f1) diverge and, for a sample, the first divergent byte
                offset with its pixel coordinate and both values.
``spec``      — emits the STAGED dispatch spec for the CUDA half.  It fires nothing.

AXIS.  ``[local-CPU $0 decode-identity manifest]``.  No scorer runs here, no score is
claimed, nothing here is promotable.  The manifest is an identity artifact, not a metric.

ALWAYS KEEP THE PAYLOAD.  The manifest IS the retained payload for a 3.66 GB decode: the
raw frames are deterministically rebuildable from the pinned archive by the pinned runtime,
and the aggregate ``raw_sha256`` proves which decode produced these hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

AXIS = "[local-CPU $0 decode-identity manifest]"
SCORE_CLAIM = False
SCHEMA = "ddm_ps1u_decode_frame_hash_manifest.v1"
DIFF_SCHEMA = "ddm_ps1u_decode_axis_diff.v1"

CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
FRAME_BYTES = CAMERA_H * CAMERA_W * CHANNELS
PAIR_COUNT = 600
FRAME_COUNT = 2 * PAIR_COUNT
RAW_BYTES = FRAME_COUNT * FRAME_BYTES

HV1_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
HV1_ARCHIVE_BYTES = 182_759
CPU_RAW_SHA256_PREFIX = "e5539653"


class ReconciliationError(RuntimeError):
    """A retained input, geometry pin, or manifest invariant differed."""


def _file_sha256(path: Path, *, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _producer_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }


def build_manifest(
    raw_path: Path,
    *,
    archive_sha256: str | None = None,
    label: str = "",
    device: str = "",
) -> dict[str, Any]:
    """Per-frame SHA-256 over a raw decode, streamed (never loads 3.66 GB)."""
    size = raw_path.stat().st_size
    if size != RAW_BYTES:
        raise ReconciliationError(
            f"raw geometry differs: {size} B != {RAW_BYTES} B "
            f"({FRAME_COUNT}x{CAMERA_H}x{CAMERA_W}x{CHANNELS})"
        )
    started = time.time()
    raw = np.memmap(
        raw_path, dtype=np.uint8, mode="r"
    ).reshape(FRAME_COUNT, CAMERA_H, CAMERA_W, CHANNELS)
    frame_hashes: list[str] = []
    aggregate = hashlib.sha256()
    for index in range(FRAME_COUNT):
        payload = np.ascontiguousarray(raw[index]).tobytes()
        frame_hashes.append(hashlib.sha256(payload).hexdigest())
        aggregate.update(payload)
    pairs = [
        {"pair": p, "f0": frame_hashes[2 * p], "f1": frame_hashes[2 * p + 1]}
        for p in range(PAIR_COUNT)
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": False,
        "label": label,
        "decode_device": device,
        "source": str(raw_path),
        "archive_sha256": archive_sha256,
        "archive_bytes": HV1_ARCHIVE_BYTES if archive_sha256 == HV1_ARCHIVE_SHA256 else None,
        "raw_sha256": _file_sha256(raw_path),
        "raw_bytes": size,
        "frames_concat_sha256": aggregate.hexdigest(),
        "frame_shape_hwc": [CAMERA_H, CAMERA_W, CHANNELS],
        "frame_count": FRAME_COUNT,
        "pair_count": PAIR_COUNT,
        "pair_frame_sha256": pairs,
        "producer_environment": _producer_environment(),
        "elapsed_seconds": time.time() - started,
        "hash_only": True,
        "omitted_payload_reason": (
            "raw frames are deterministically rebuildable from the pinned archive by the "
            "pinned runtime; raw_sha256 pins which decode produced these hashes"
        ),
    }
    return manifest


def diff_manifests(a: dict[str, Any], b: dict[str, Any], *, sample: int = 3) -> dict[str, Any]:
    """Adjudicate two decode manifests."""
    for side, m in (("a", a), ("b", b)):
        if m.get("schema") != SCHEMA:
            raise ReconciliationError(f"manifest {side} is not {SCHEMA}")
        if int(m.get("pair_count", -1)) != PAIR_COUNT:
            raise ReconciliationError(f"manifest {side} pair_count differs")
    if a.get("archive_sha256") != b.get("archive_sha256"):
        raise ReconciliationError(
            "manifests describe DIFFERENT archives — the comparison would be meaningless "
            f"({a.get('archive_sha256')} vs {b.get('archive_sha256')})"
        )
    rows_a = {int(r["pair"]): r for r in a["pair_frame_sha256"]}
    rows_b = {int(r["pair"]): r for r in b["pair_frame_sha256"]}
    differing: list[dict[str, Any]] = []
    for pair in range(PAIR_COUNT):
        ra, rb = rows_a[pair], rows_b[pair]
        f0 = ra["f0"] != rb["f0"]
        f1 = ra["f1"] != rb["f1"]
        if f0 or f1:
            differing.append(
                {
                    "pair": pair,
                    "f0_differs": f0,
                    "f1_differs": f1,
                    "a_f0": ra["f0"], "b_f0": rb["f0"],
                    "a_f1": ra["f1"], "b_f1": rb["f1"],
                }
            )
    identical = not differing
    verdict = (
        "DECODE_IDENTICAL_ACROSS_AXES" if identical else "DEVICE_DEPENDENT_DECODE_CONFIRMED"
    )
    report: dict[str, Any] = {
        "schema": DIFF_SCHEMA,
        "axis": "[decode-identity adjudication]",
        "score_claim": False,
        "verdict": verdict,
        "archive_sha256": a.get("archive_sha256"),
        "a": {k: a.get(k) for k in ("label", "decode_device", "raw_sha256", "source")},
        "b": {k: b.get(k) for k in ("label", "decode_device", "raw_sha256", "source")},
        "raw_sha256_equal": a.get("raw_sha256") == b.get("raw_sha256"),
        "pairs_differing": len(differing),
        "frames_differing": sum(
            int(d["f0_differs"]) + int(d["f1_differs"]) for d in differing
        ),
        "f0_only": sum(1 for d in differing if d["f0_differs"] and not d["f1_differs"]),
        "f1_only": sum(1 for d in differing if d["f1_differs"] and not d["f0_differs"]),
        "both": sum(1 for d in differing if d["f0_differs"] and d["f1_differs"]),
        "differing_pairs": [d["pair"] for d in differing],
        "differing_detail_sample": differing[:sample],
        "adjudication": (
            "IDENTICAL manifests ⇒ the inflate output is bit-identical on both axes, so the "
            "d_pose discrepancy is NOT a decode difference; re-hunt it at the scorer-input "
            "preprocessing boundary (resize/YUV6 path) and at the evaluate.py batching/"
            "device-reduction surface, in that order."
            if identical
            else "DIFFERENT manifests ⇒ device-dependent decode CONFIRMED. This violates the "
            "deterministic-decode non-negotiable independently of scoring axis. Route to the "
            "receiver's device-adaptive block; the byte-offset sample localizes the first "
            "divergence for the decode-path suspect hunt."
        ),
    }
    return report


def adjudicate_aggregate(
    cpu_manifest: dict[str, Any],
    *,
    cuda_raw_sha256: str,
    cuda_raw_bytes: int,
    cuda_source: str,
    cuda_provenance: str,
) -> dict[str, Any]:
    """Aggregate-level verdict when only the CUDA side's raw sha is available.

    This is the $0 path: the T4 run's own ``inflated_outputs_manifest`` already
    records the CUDA-decoded ``0.raw`` sha, so the decode-identity question is
    answerable without buying a dispatch. It yields the VERDICT but not the
    per-pair localization — that rides the next T4 row for free."""
    if cpu_manifest.get("schema") != SCHEMA:
        raise ReconciliationError(f"cpu manifest is not {SCHEMA}")
    cpu_sha = str(cpu_manifest["raw_sha256"])
    cpu_bytes = int(cpu_manifest["raw_bytes"])
    if cpu_bytes != cuda_raw_bytes:
        raise ReconciliationError(
            f"raw geometry differs across axes ({cpu_bytes} vs {cuda_raw_bytes}) — "
            "the comparison would be confounded by a shape change, not a decode change"
        )
    identical = cpu_sha == cuda_raw_sha256
    return {
        "schema": "ddm_ps1u_decode_axis_aggregate_verdict.v1",
        "axis": "[decode-identity adjudication; $0 from retained receipts]",
        "score_claim": False,
        "promotion_eligible": False,
        "verdict": (
            "DECODE_IDENTICAL_ACROSS_AXES"
            if identical
            else "DEVICE_DEPENDENT_DECODE_CONFIRMED"
        ),
        "archive_sha256": cpu_manifest.get("archive_sha256"),
        "archive_bytes": cpu_manifest.get("archive_bytes"),
        "raw_bytes_both_axes": cpu_bytes,
        "cpu": {
            "raw_sha256": cpu_sha,
            "source": cpu_manifest.get("source"),
            "decode_device": cpu_manifest.get("decode_device"),
            "frames_concat_sha256": cpu_manifest.get("frames_concat_sha256"),
        },
        "cuda": {
            "raw_sha256": cuda_raw_sha256,
            "source": cuda_source,
            "decode_device": "cuda",
            "provenance": cuda_provenance,
        },
        "localization_available": False,
        "localization_plan": (
            "per-pair localization requires the CUDA-side per-frame hashes; it rides the "
            "NEXT T4 row for free by enabling the hash-request fields and running this "
            "module's `manifest` step remotely — no dedicated dispatch is warranted"
        ),
        "compliance": (
            "VIOLATION of the deterministic-decode non-negotiable ('same archive.zip -> "
            "bit-identical inflate output every run/host'): one archive, two devices, two "
            "different raw decodes. This is a vehicle defect independent of which scoring "
            "axis is 'right'."
            if not identical
            else "deterministic-decode non-negotiable HOLDS across these two axes"
        ),
        "cure_direction": (
            "port the decode to the portable native/runtime-rs program so ONE deterministic "
            "implementation runs on every host. Engineer it to PRESERVE the CUDA-favorable "
            "frames -- the frontier rides them -- rather than naively CPU-pinning, which "
            "would lock in the degraded decode."
        ),
    }


def locate_first_divergence(
    raw_a: Path, raw_b: Path, pairs: list[int], *, limit: int = 3
) -> list[dict[str, Any]]:
    """First divergent byte offset + pixel coordinate for a sample of pairs.

    Only callable when BOTH raws are on local disk; the CUDA half returns hashes
    only, so this is the follow-up step after a DIFFERENT verdict."""
    a = np.memmap(raw_a, dtype=np.uint8, mode="r").reshape(
        FRAME_COUNT, CAMERA_H, CAMERA_W, CHANNELS
    )
    b = np.memmap(raw_b, dtype=np.uint8, mode="r").reshape(
        FRAME_COUNT, CAMERA_H, CAMERA_W, CHANNELS
    )
    out: list[dict[str, Any]] = []
    for pair in pairs[:limit]:
        for which, frame in (("f0", 2 * pair), ("f1", 2 * pair + 1)):
            fa = np.asarray(a[frame])
            fb = np.asarray(b[frame])
            if np.array_equal(fa, fb):
                continue
            flat = np.flatnonzero(fa.reshape(-1) != fb.reshape(-1))
            first = int(flat[0])
            y, rem = divmod(first, CAMERA_W * CHANNELS)
            x, c = divmod(rem, CHANNELS)
            delta = fa.astype(np.int16) - fb.astype(np.int16)
            out.append(
                {
                    "pair": pair,
                    "frame": which,
                    "differing_values": int(flat.size),
                    "differing_fraction": float(flat.size) / FRAME_BYTES,
                    "first_offset": first,
                    "first_pixel_yxc": [y, x, c],
                    "a_value": int(fa[y, x, c]),
                    "b_value": int(fb[y, x, c]),
                    "max_abs_delta": int(np.abs(delta).max()),
                    "mean_abs_delta_over_differing": float(
                        np.abs(delta).reshape(-1)[flat].mean()
                    ),
                }
            )
    return out


def dispatch_spec() -> dict[str, Any]:
    """The STAGED CUDA half. MAIN fires it; this module fires nothing."""
    return {
        "schema": "ddm_ps1u_cuda_decode_manifest_dispatch_spec.v1",
        "status": "STAGED_NOT_FIRED",
        "owner_to_fire": "MAIN",
        "purpose": (
            "Inflate archive 80d9c8c6… on a T4 with the CUDA-adaptive receiver and return a "
            "per-pair raw-frame hash manifest (hashes only, ~120 KB) — no 3.66 GB egress."
        ),
        "archive": {
            "sha256": HV1_ARCHIVE_SHA256,
            "bytes": HV1_ARCHIVE_BYTES,
            "staged_generation": (
                "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
                "hv1_ep0634_s1p25_c1p0_brotli_q10"
            ),
            "runtime_tree_sha256": (
                "70ec7bb1a673dcc4b828b7d826603e365092d524fc74ee0e8c4f2ad66bfcf6e8"
            ),
        },
        "reuse_not_rebuild": {
            "transport": "experiments/ddm_qs1_modal_t4_dual_axis.py::main "
            "(the durable sealed-request T4 transport; gpu='T4', "
            "timeout=substrate.CONTEST_LIMIT_SECONDS, memory=16384, volume-backed)",
            "prior_shape": "experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/"
            "MODAL_REMOTE_RESULT.json (same archive, inflate_device_policy=auto, "
            "scorer_device=cuda, Tesla T4, 421.6 s, PASSED)",
            "hash_return_precedent": "that result already carries the "
            "scorer_input_cache_hashes_requested / _hash_batch_pairs fields (both currently "
            "False/8) — the hash-only return channel EXISTS on the canonical auth-eval path "
            "and does not need inventing",
            "canonical_aggregate_producer": "tac.local_acceleration.mlx_preprocess."
            "write_scorer_input_cache_hash_manifest_from_raw_file (aggregate-only; this "
            "module's per-pair manifest is the localizing superset and emits "
            "frames_concat_sha256 for cross-checking against it)",
        },
        "remote_step": (
            "after inflate completes and 0.raw exists in the run root, call "
            "experiments/ddm_ps1u_decode_axis_reconciliation.py manifest "
            "--raw <run_root>/inflated/0.raw --out <run_root>/CUDA_DECODE_MANIFEST.json "
            f"--archive-sha256 {HV1_ARCHIVE_SHA256} --label cuda_t4 --device cuda"
        ),
        "return_payload_bytes_estimate": 120_000,
        "expected_cost_usd": "~0.15-0.20 (T4, ~7 min: the r2 precedent ran 421.6 s wall)",
        "adjudication_command": (
            ".venv/bin/python experiments/ddm_ps1u_decode_axis_reconciliation.py diff "
            "--a <cpu_manifest.json> --b <cuda_manifest.json> --out DECODE_AXIS_VERDICT.json"
        ),
        "decode_path_suspects_if_different": [
            "the receiver's device-adaptive block (inflate.py 'cuda' if available) — the "
            "ONLY known intentional CPU/CUDA branch in the shipped runtime; the hv1 fire memo "
            "records it as UNCHANGED from the incumbent, so it is the prime suspect",
            "torch bicubic/bilinear interpolate on CUDA vs CPU (the frame0 carrier render "
            "upsamples 384x512 -> 874x1164 with mode='bicubic'); CUDA and CPU kernels are not "
            "bit-identical and the result is round()ed to uint8 — a half-ULP straddle flips a "
            "pixel",
            "clamp/round ordering and fp32 accumulation order in the HPAC/neural render",
            "threading/worker count in the native decoder (the CPU lift runs 4 workers)",
        ],
        "fires_nothing": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="per-pair raw-frame hash manifest")
    m.add_argument("--raw", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--archive-sha256", default=None)
    m.add_argument("--label", default="")
    m.add_argument("--device", default="")

    d = sub.add_parser("diff", help="adjudicate two manifests")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--sample", type=int, default=3)
    d.add_argument("--raw-a", default=None, help="optional: locate first divergent bytes")
    d.add_argument("--raw-b", default=None)

    s = sub.add_parser("spec", help="emit the STAGED CUDA dispatch spec (fires nothing)")
    s.add_argument("--out", required=True)

    g = sub.add_parser("verdict-aggregate", help="$0 verdict from a retained CUDA raw sha")
    g.add_argument("--cpu-manifest", required=True)
    g.add_argument("--cuda-raw-sha256", required=True)
    g.add_argument("--cuda-raw-bytes", type=int, required=True)
    g.add_argument("--cuda-source", required=True)
    g.add_argument("--cuda-provenance", required=True)
    g.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.command == "manifest":
        manifest = build_manifest(
            Path(args.raw),
            archive_sha256=args.archive_sha256,
            label=args.label,
            device=args.device,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, out)
        print(
            f"[ps1u-recon] {AXIS}\n"
            f"[ps1u-recon] raw_sha256={manifest['raw_sha256']}\n"
            f"[ps1u-recon] frames={manifest['frame_count']} pairs={manifest['pair_count']} "
            f"concat={manifest['frames_concat_sha256'][:16]}… "
            f"{manifest['elapsed_seconds']:.1f}s -> {out}"
        )
        return 0

    if args.command == "diff":
        a = json.loads(Path(args.a).read_text())
        b = json.loads(Path(args.b).read_text())
        report = diff_manifests(a, b, sample=args.sample)
        if report["verdict"] == "DEVICE_DEPENDENT_DECODE_CONFIRMED" and args.raw_a and args.raw_b:
            report["first_divergence_sample"] = locate_first_divergence(
                Path(args.raw_a), Path(args.raw_b), report["differing_pairs"]
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[ps1u-recon] VERDICT {report['verdict']}")
        print(
            f"[ps1u-recon] pairs differing {report['pairs_differing']}/{PAIR_COUNT} "
            f"frames {report['frames_differing']}/{FRAME_COUNT} "
            f"(f0-only {report['f0_only']} f1-only {report['f1_only']} both {report['both']})"
        )
        print(f"[ps1u-recon] -> {out}")
        return 0

    if args.command == "verdict-aggregate":
        report = adjudicate_aggregate(
            json.loads(Path(args.cpu_manifest).read_text()),
            cuda_raw_sha256=args.cuda_raw_sha256,
            cuda_raw_bytes=args.cuda_raw_bytes,
            cuda_source=args.cuda_source,
            cuda_provenance=args.cuda_provenance,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[ps1u-recon] VERDICT {report['verdict']}")
        print(f"[ps1u-recon]   cpu  {report['cpu']['raw_sha256']}")
        print(f"[ps1u-recon]   cuda {report['cuda']['raw_sha256']}")
        print(f"[ps1u-recon] -> {out}")
        return 0

    spec = dispatch_spec()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(f"[ps1u-recon] STAGED (fires nothing) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
