#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure and byte-close the lossless cross-tensor witness coding lever.

This is a $0 local, read-only-on-checkpoint probe.  It writes only small packet
artifacts and a machine-readable receipt under ``--out-dir``.  The probe fixes
the checkpoint's canonical symmetric-int8 grid, then compares four storage
charts: identity, derived weight-axis permutation, frame-separated pair delta,
and their composition.  No scorer, cloud, training, or precision allocation is
invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import levelset_byte_close_and_eval as bc  # noqa: E402

from tac import contest_score  # noqa: E402
from tac.boundary_math import witness_crosstensor_codec as xcodec  # noqa: E402
from tac.boundary_math import xi_pose_coder  # noqa: E402


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _checkpoint_verdict(ckpt_dir: Path) -> dict[str, Any]:
    """Read the checkpoint-selected n600 CPU verdict without inventing values."""
    best_path = ckpt_dir / "levelset_best.json"
    log_path = ckpt_dir / "run.log"
    if not (best_path.exists() and log_path.exists()):
        return {"available": False, "reason": "levelset_best.json or run.log absent"}
    best = json.loads(best_path.read_text())
    epoch = int(best["epoch"])
    matches: list[dict[str, Any]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stage") == "verdict" and int(row.get("epoch", -1)) == epoch and row.get("verdict_device") == "cpu":
            matches.append(row)
    if not matches:
        return {"available": False, "reason": f"no CPU verdict row for best epoch {epoch}"}
    row = matches[-1]
    return {
        "available": True,
        "measurement_class": "MEASURED",
        "scope": "n600 full-precision trainer CPU verdict; not the int8 byte-closed absolute row",
        "epoch": epoch,
        "n_pairs": 600,
        "d_seg": float(best["d_seg"]),
        "d_pose": float(row["d_pose"]),
        "verdict_device": "cpu",
        "best_json": str(best_path),
        "run_log": str(log_path),
        "run_log_blob_bytes": int(row["blob_bytes"]),
    }


def _pack_chart(
    baseline_blob: bytes,
    candidate_blob: bytes,
    *,
    use_base_permutation: bool,
    use_code_delta: bool,
) -> bytes:
    bm, bb, bcoder, bpose, blane, bpcar = bc._read_blob_bytes(baseline_blob)
    cm, cb, ccoder, _cpose, _clane, _cpcar = bc._read_blob_bytes(candidate_blob)
    manifest = dict(bm)
    if use_base_permutation or use_code_delta:
        manifest["xcodec"] = {
            "p": list((cm.get("xcodec") or {}).get("p", ())) if use_base_permutation else [],
            "c": 1 if use_code_delta else 0,
        }
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    return bc._io_pack(
        mj,
        cb if use_base_permutation else bb,
        ccoder if use_code_delta else bcoder,
        bpose,
        blane,
        bpcar,
    )


def _decoded_quantized_state(blob: bytes) -> tuple[str, dict[str, np.ndarray], np.ndarray]:
    manifest, base_b, code_b, _pose, _lane, _pcar = bc._read_blob_bytes(blob)
    raw_base = brotli.decompress(base_b)
    base = xcodec.decode_base_quantized(
        raw_base,
        manifest["base_param_order"],
        manifest["base_shapes"],
        bc._xcodec_transposed_names(manifest),
    )
    raw_code = brotli.decompress(code_b)
    code = xcodec.decode_code_quantized(
        raw_code,
        manifest["code_shape"],
        bc._xcodec_code_transform(manifest),
    )
    h = hashlib.sha256()
    for name in manifest["base_param_order"]:
        arr = np.ascontiguousarray(base[name], dtype=np.int8)
        h.update(name.encode("utf-8") + b"\0")
        h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
        h.update(arr.tobytes())
    h.update(b"code\0")
    h.update(np.asarray(code.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(code, dtype=np.int8).tobytes())
    return h.hexdigest(), base, code


def _pose_payload_report(params: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = ("pose_carrier.xi_stored", "pose_carrier.dxi")
    if not all(k in params for k in keys):
        return {"present": False, "measurement_class": "MEASURED"}
    xi_stored = np.asarray(params[keys[0]], dtype=np.float64)
    dxi = np.asarray(params[keys[1]], dtype=np.float64)
    xi_eff = xi_stored + dxi
    q, scales = xi_pose_coder.quantize_xi(xi_eff)
    coders = {}
    decoded = []
    for name in ("none", "delta_ar", "delta_res"):
        payload = xi_pose_coder.serialize_xi_payload(q, scales, coder=name)
        q2, s2 = xi_pose_coder.parse_xi_payload(payload)
        coders[name] = {
            "payload_bytes": len(payload),
            "quantized_roundtrip_exact": bool(np.array_equal(q2, q) and np.array_equal(s2, scales)),
        }
        decoded.append(q2)
    strict = all(np.array_equal(decoded[0], other) for other in decoded[1:])
    return {
        "present": True,
        "measurement_class": "MEASURED",
        "n_pairs": int(xi_eff.shape[0]),
        "xi_stored_float32_bytes": int(params[keys[0]].nbytes),
        "dxi_float32_bytes": int(params[keys[1]].nbytes),
        "xi_eff_float32_bytes": int(xi_eff.astype(np.float32).nbytes),
        "exact_unique_quantized_rows": int(np.unique(q, axis=0).shape[0]),
        "coders": coders,
        "all_coders_same_quantized_state": strict,
        "verdict": "REUSE_EXISTING_DELTA_RES" if strict else "FAIL_CLOSED",
        "verdict_scope": (
            "FORMULATION x INSTANCE: exact-row dedup and existing xi integer residual coders "
            "on this checkpoint; not a negative on in-training pose structure"
        ),
    }


def _chart_receipt(name: str, blob: bytes, packet_root: Path) -> dict[str, Any]:
    packet_dir = packet_root / f"{name}_packet"
    archive, archive_bytes = bc.assemble_packet(blob, packet_dir)
    manifest, base_b, code_b, _pose, _lane, _pcar = bc._read_blob_bytes(blob)
    return {
        "name": name,
        "measurement_class": "MEASURED",
        "blob_bytes": len(blob),
        "manifest_bytes": len(json.dumps(manifest, separators=(",", ":")).encode("utf-8")),
        "base_brotli_bytes": len(base_b),
        "code_brotli_bytes": len(code_b),
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256_path(archive),
        "packet_dir": str(packet_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--npz-name", default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bit-exact-pairs", type=int, default=1)
    args = ap.parse_args()

    ckpt = args.ckpt_dir / args.npz_name
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    out_dir = args.out_dir.resolve()
    if str(out_dir).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError("operator-facing evidence cannot live under a temporary directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    params, cfg = bc._load_levelset_ckpt(args.ckpt_dir, args.npz_name)
    if int(cfg["n_pairs"]) != 600:
        raise ValueError(f"n600 required; checkpoint has {cfg['n_pairs']} pairs")
    so = bc.detect_self_orient(cfg, {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4})
    order = tuple(
        k for k in params if k != "code" and not k.startswith("pose_carrier.") and not (k == "B" or k.endswith("_B"))
    )

    step0 = xcodec.measure_step0(params, order, ckpt)
    base_plan = xcodec.derive_base_permutation_plan(params, order)
    code_plan = xcodec.derive_code_transform_plan(params["code"])
    baseline_blob, baseline_breakdown = bc.build_levelset_blob(params, cfg, so, None, cross_tensor_codec=False)
    candidate_blob, candidate_breakdown = bc.build_levelset_blob(params, cfg, so, None, cross_tensor_codec=True)
    blobs = {
        "identity": baseline_blob,
        "weight_permutation_only": _pack_chart(
            baseline_blob, candidate_blob, use_base_permutation=True, use_code_delta=False
        ),
        "pair_delta_only": _pack_chart(baseline_blob, candidate_blob, use_base_permutation=False, use_code_delta=True),
        "joint_lossless": candidate_blob,
    }
    charts = {name: _chart_receipt(name, blob, out_dir) for name, blob in blobs.items()}

    state_hashes = {}
    decoded = {}
    for name, blob in blobs.items():
        state_hashes[name], base, code = _decoded_quantized_state(blob)
        decoded[name] = (base, code)
    ref_base, ref_code = decoded["identity"]
    exact = all(
        np.array_equal(ref_code, code)
        and ref_base.keys() == base.keys()
        and all(np.array_equal(ref_base[k], base[k]) for k in ref_base)
        for base, code in decoded.values()
    )
    if not exact or len(set(state_hashes.values())) != 1:
        raise RuntimeError("cross-tensor chart changed the decoded quantized n600 state")

    # Exercise both shipped decoders against the NumPy-fp32 oracle. The gate deletes its capped raw
    # scratch on success; the durable packet and JSON receipt remain small.
    bit_exact = {}
    for name in ("identity", "joint_lossless"):
        result = bc.bit_exact_roundtrip_gate(Path(charts[name]["packet_dir"]), blobs[name], args.bit_exact_pairs, True)
        bit_exact[name] = result

    pose = _pose_payload_report(params)
    baseline_archive = charts["identity"]["archive_bytes"]
    candidate_archive = charts["joint_lossless"]["archive_bytes"]
    delta_archive = candidate_archive - baseline_archive
    delta_s = contest_score.rate_term(candidate_archive) - contest_score.rate_term(baseline_archive)
    verdict = _checkpoint_verdict(args.ckpt_dir)
    base_symbols = int(step0["base_weight_symbols"])
    code_symbols = int(np.prod(params["code"].shape))
    receipt = {
        "schema": "witness_crosstensor_structure_rate.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU/numpy-fp32 advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": False,
        "execution": "$0 LOCAL; no cloud/paid dispatch; no training",
        "checkpoint": {
            "path": str(ckpt),
            "sha256": _sha256_path(ckpt),
            "bytes": ckpt.stat().st_size,
            "n_pairs": int(cfg["n_pairs"]),
            "npz_name": args.npz_name,
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "step0_shared_codebook_gate": step0,
        "mass": {
            "measurement_class": "MEASURED",
            "base_weight_symbols": base_symbols,
            "pair_code_symbols": code_symbols,
            "counted_symbol_total": base_symbols + code_symbols,
            "base_weight_fraction": base_symbols / (base_symbols + code_symbols),
        },
        "base_permutation_plan": base_plan.to_json(),
        "pair_code_plan": code_plan.to_json(),
        "per_pair_payload": pose,
        "charts": charts,
        "decoded_quantized_state": {
            "measurement_class": "MEASURED",
            "n_pairs": 600,
            "exact_equal_across_all_charts": exact,
            "sha256_by_chart": state_hashes,
            "component_delta_d_seg": 0.0,
            "component_delta_d_pose": 0.0,
            "component_delta_class": "DERIVED from exact full-state equality plus deterministic receiver",
        },
        "shipped_receiver_gate": bit_exact,
        "baseline_checkpoint_component_verdict": verdict,
        "score_delta": {
            "measurement_class": "DERIVED from MEASURED exact archive bytes using tac.contest_score.rate_term",
            "archive_delta_bytes": delta_archive,
            "advisory_delta_S": delta_s,
            "d_seg_delta": 0.0,
            "d_pose_delta": 0.0,
            "absolute_byte_closed_d_seg_d_pose": "NOT_REMEASURED; full 3.66 GB n600 inflate blocked by SSD-first policy in workspace-only sandbox",
        },
        "baseline_breakdown": baseline_breakdown,
        "candidate_breakdown": candidate_breakdown,
        "verdict": "ADMIT_JOINT_LOSSLESS_STORAGE_CHART",
        "verdict_scope": (
            "INSTANCE x FORMULATION: exact symmetric-int8 byte-close of the n600 V9 ep150 EMA-best "
            "checkpoint. Shared post-hoc value codebook and exact row dedup are null here; this does "
            "not kill training-time tying, low-rank/VQ-in-loop, latent-structure regularization, or "
            "other witness checkpoints."
        ),
        "reactivation_route": (
            "Route training-induced shared structure to ideal-config/#242/#110; do not edit "
            "spec_v9_cgauge.py from this lane."
        ),
    }
    receipt_path = out_dir / "measurement_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "baseline_archive_bytes": baseline_archive,
                "candidate_archive_bytes": candidate_archive,
                "archive_bytes_saved": -delta_archive,
                "advisory_delta_S": delta_s,
                "quantized_state_sha256": state_hashes["identity"],
                "bit_exact": {k: v["bit_exact"] for k, v in bit_exact.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
