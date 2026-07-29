#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P2a — EU1-FD2-QDBS discrete terminal finisher on the t3 TR1 endpoint.

Runs the committed ``run_fd2_qdbs_terminal`` harness (16 signed singletons + 8
grouped proposals + 24 precommitted matched controls, <=48 candidate evals + 1
shared base) on the TR1 token lattice with an incremental-EXACT full-population
evaluator:

* theta = the flat int64 view of the deployed token codes [600,24,32,4]
  (16-level lattice) — the counted description coordinates of this vehicle.
* compile = re-encode the token section via the committed runtime framing
  (``_encode_tokens`` + ``build_packet`` + deterministic stored ZIP); all other
  sections are the byte-identical base payloads.  Base compile is asserted
  byte-identical to the P1 exporter archive.
* evaluate = render ONLY the pairs whose codes differ from base through the
  committed receiver, verdict them with the frozen CPU-torch SegNet/PoseNet,
  and recompute the exact full-n600 means from the P1 per-pair cache.  d_seg
  mean is a per-pair average, so single-pair re-verdicts compose EXACTLY.

Authority: ``STALE_REHEARSAL`` mode is used DELIBERATELY — the module's
``ContestAxis`` has no advisory member and our verdicts are
``[macOS-CPU advisory]``; claiming ``[contest-CPU]`` custody would be a fake
axis label.  The evaluations themselves are full-population on the deployed
bytes.  Any strict winner is re-confirmed by an independent non-incremental
full verdict (``--confirm-winner``).

Proposal recipe (pre-registered, deterministic from P1 caches):
  rank coordinate (pair,gh,gw,ch) by  cell_flips[pair,gh,gw] * |quant_resid|,
  where quant_resid = continuous EMA lattice position - rounded code, sign
  toward the second-nearest level; legality: base code in [1,14] so +/-1 stays
  on the lattice.  Top 16 distinct -> singletons; next 16 -> 8 groups of 2.

score_claim=false; promotion_eligible=false; pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pb1_p2a_qdbs_tr1_receipt.v1"
LEVELS = 16
SHAPE = (600, 24, 32, 4)
SIGNAL_LABEL = "p1_cell_flips_x_quant_resid_v1"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--pose-targets", required=True, type=Path)
    ap.add_argument("--p1-dir", required=True, type=Path,
                    help="out dir holding p1_chunks/ from the P1 base verdict")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--confirm-winner", action="store_true",
                    help="independent full non-incremental verdict of winner")
    return ap.parse_args()


def _load_p1(p1_dir: Path):
    paths = sorted((p1_dir / "p1_chunks").glob("chunk_*.npz"))
    if not paths:
        raise SystemExit("no P1 chunks found; run P1 first")
    idxs, dsegs, dposes, cells = [], [], [], []
    for p in paths:
        z = np.load(p)
        idxs.append(z["idxs"])
        dsegs.append(z["dsegs"])
        dposes.append(z["dposes"])
        cells.append(z["cell_flips"])
    idx = np.concatenate(idxs)
    order = np.argsort(idx)
    if not np.array_equal(idx[order], np.arange(SHAPE[0])):
        raise SystemExit("P1 cache does not cover exactly pairs 0..599")
    ds = np.concatenate(dsegs)[order]
    dp = np.concatenate(dposes)[order]
    cf = np.concatenate(cells)[order]
    if not np.isfinite(dp).all():
        raise SystemExit("P1 cache lacks pose rows; rerun P1 with --pose")
    return ds, dp, cf


def _quant_residuals(checkpoint: Path) -> np.ndarray:
    z = np.load(checkpoint)
    base = np.asarray(z["ema::tokens_base"], dtype=np.float32)
    delta = np.asarray(z["ema::tokens_delta"], dtype=np.float32)
    combined = np.clip(base[None] + delta, -1.0, 1.0)
    scaled = (combined + 1.0) * 0.5 * (LEVELS - 1)
    return scaled - np.rint(scaled)


def build_proposals(cell_flips: np.ndarray, resid: np.ndarray,
                    base_codes: np.ndarray):
    from tac.optimization.fd2_qdbs_terminal import (
        DescriptionDelta,
        DescriptionProposal,
        ProposalClass,
    )

    weight = cell_flips.astype(np.float64)[..., None] * np.abs(resid)
    legal = (base_codes >= 1) & (base_codes <= LEVELS - 2)
    weight = np.where(legal, weight, 0.0)
    flat = weight.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    picked = [int(i) for i in order[:32] if flat[i] > 0.0]
    if len(picked) < 32:
        raise SystemExit(f"only {len(picked)} legal signal coords; need 32")
    resid_flat = resid.reshape(-1)

    def _sign(i: int) -> int:
        return 1 if resid_flat[i] > 0 else -1

    singles = tuple(
        DescriptionProposal(
            identity=f"single_{k:02d}_idx{picked[k]}",
            proposal_class=ProposalClass.SCORER_SINGLETON,
            deltas=(DescriptionDelta(picked[k], _sign(picked[k])),),
            signal_label=SIGNAL_LABEL,
            signal_value=float(flat[picked[k]]),
        )
        for k in range(16)
    )
    groups = []
    for g in range(8):
        a, b = picked[16 + 2 * g], picked[16 + 2 * g + 1]
        deltas = tuple(sorted(
            (DescriptionDelta(a, _sign(a)), DescriptionDelta(b, _sign(b)))))
        groups.append(DescriptionProposal(
            identity=f"group_{g:02d}_idx{a}_{b}",
            proposal_class=ProposalClass.SCORER_GROUP,
            deltas=deltas,
            signal_label=SIGNAL_LABEL,
            signal_value=float(flat[a] + flat[b]),
        ))
    active = np.nonzero(
        (np.broadcast_to(cell_flips[..., None] > 0, SHAPE) & legal).reshape(-1)
    )[0]
    return singles, tuple(groups), active


class TR1Oracle:
    """Callbacks for run_fd2_qdbs_terminal over the TR1 archive grammar."""

    def __init__(self, base_archive: bytes, base_dsegs, base_dposes,
                 gt_cache: Path, pose_targets: Path):
        from train_witness_realized_through_R_mlx import (
            cpu_verdict_d_pose_batch,
            cpu_verdict_d_seg_argmax_batch,
        )

        from tac.boundary_math.power_diagram_witness import (
            open_stored_npy_memmap,
        )
        from tac.optimization import ddm_tr1_runtime as rt

        self.rt = rt
        self._seg_batch = cpu_verdict_d_seg_argmax_batch
        self._pose_batch = cpu_verdict_d_pose_batch
        self.base_archive = base_archive
        self.base_parsed = rt.parse_archive(base_archive)
        self.base_codes = np.asarray(
            self.base_parsed.packet.token_codes, dtype=np.int64)
        self.base_dsegs = np.asarray(base_dsegs, dtype=np.float64)
        self.base_dposes = np.asarray(base_dposes, dtype=np.float64)
        self.lstars = open_stored_npy_memmap(gt_cache, "lstars")
        rows = json.loads(pose_targets.read_text())["rows"]
        self.targets = [np.asarray(r["center"], dtype=np.float64)
                        for r in rows]
        seg_cpu, posenet_cpu = _load_scorers()
        self.seg_cpu = seg_cpu
        self.posenet_cpu = posenet_cpu
        self.eval_count = 0
        self.eval_wall = 0.0
        self._compile_cache: dict[str, bytes] = {}

    # -- compile ---------------------------------------------------------
    def compile_archive(self, theta: np.ndarray, proposal) -> bytes:
        codes = np.asarray(theta, dtype=np.int64).reshape(SHAPE)
        if codes.min() < 0 or codes.max() >= LEVELS:
            raise ValueError("candidate code escaped the declared lattice")
        cache_key = hashlib.sha256(
            np.ascontiguousarray(codes).tobytes()).hexdigest()
        cached = self._compile_cache.get(cache_key)
        if cached is not None:
            return cached
        rt = self.rt
        token_payload = rt._encode_tokens(codes.astype(np.uint8))
        payloads = {
            "tokens": token_payload,
            "lotto_renderer": self.base_parsed.packet.section_payloads[1],
            "selector": self.base_parsed.packet.section_payloads[2],
            "pose_stub": self.base_parsed.packet.section_payloads[3],
        }
        packet = rt.build_packet(self.base_parsed.packet.metadata, payloads)
        manifest = rt._archive_manifest(packet)
        archive = rt._deterministic_stored_zip({
            "manifest.json": rt._canonical_json(dict(manifest)),
            "state/tr1.ddt1": packet,
        })
        if len(self._compile_cache) >= 8:
            self._compile_cache.pop(next(iter(self._compile_cache)))
        self._compile_cache[cache_key] = archive
        return archive

    # -- parse -----------------------------------------------------------
    def parse_archive(self, archive_bytes: bytes):
        from tac.optimization.fd2_qdbs_terminal import (
            ParsedDescriptionCandidate,
        )

        parsed = self.rt.parse_archive(archive_bytes)
        theta = np.asarray(parsed.packet.token_codes, dtype=np.int64).reshape(-1)
        return ParsedDescriptionCandidate(
            realized_theta=theta,
            archive_sha256=_sha256_bytes(archive_bytes),
            exact_parseback=True,
            value=parsed,
        )

    # -- consume ----------------------------------------------------------
    def consume_archive(self, parsed_candidate):
        from tac.optimization.fd2_qdbs_terminal import (
            ConsumedDescriptionCandidate,
        )

        parsed = parsed_candidate.value
        codes = np.asarray(parsed.packet.token_codes, dtype=np.int64)
        touched = np.nonzero(
            (codes != self.base_codes).any(axis=(1, 2, 3)))[0]
        check_pairs = [*[int(p) for p in touched[:4]], 0]
        for p in check_pairs:
            grid = self.rt.decode_token_grid(parsed.packet, int(p))
            if not np.array_equal(np.asarray(grid, dtype=np.int64),
                                  codes[int(p)]):
                raise ValueError("decode_token_grid mismatch")
        return ConsumedDescriptionCandidate(
            realized_theta=parsed_candidate.realized_theta,
            archive_sha256=parsed_candidate.archive_sha256,
            exact_consumption=True,
            value=parsed,
        )

    # -- evaluate ----------------------------------------------------------
    def evaluate_joint_action_idempotent(self, consumed, idempotency_key: str):
        from tac.optimization.fd2_qdbs_terminal import RealizedJointAction

        t0 = time.time()
        parsed = consumed.value
        codes = np.asarray(parsed.packet.token_codes, dtype=np.int64)
        touched = [int(p) for p in np.nonzero(
            (codes != self.base_codes).any(axis=(1, 2, 3)))[0]]
        dsegs = self.base_dsegs.copy()
        dposes = self.base_dposes.copy()
        if touched:
            frames = [self.rt.render_frame1_camera_uint8(parsed.packet, p)
                      for p in touched]
            gts = [np.asarray(self.lstars[p], dtype=np.int64)
                   for p in touched]
            ds, _ = self._seg_batch(self.seg_cpu, frames, gts)
            zeros = np.zeros_like(frames[0])
            dp = self._pose_batch(
                self.posenet_cpu, [zeros] * len(frames), frames,
                [self.targets[p] for p in touched])
            for j, p in enumerate(touched):
                dsegs[p] = ds[j]
                dposes[p] = dp[j]
        archive_bytes = self.compile_archive(
            codes.reshape(-1), None)
        self.eval_count += 1
        self.eval_wall += time.time() - t0
        return RealizedJointAction(
            d_seg=float(dsegs.mean()),
            d_pose=float(dposes.mean()),
            archive_sha256=_sha256_bytes(archive_bytes),
            archive_bytes=len(archive_bytes),
            sample_count=600,
            authority_marker="FULL_N600_INCREMENTAL_EXACT_MACOS_ADVISORY",
            custody_digest=None,
            evaluation_idempotency_key=idempotency_key,
            realized=True,
        )


def _load_scorers():
    sys.path.insert(0, str(REPO / "upstream"))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    from tac.boundary_math.seg_core import load_real_segnet

    seg_cpu = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False
    return seg_cpu, posenet_cpu


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    from tac.optimization.fd2_qdbs_terminal import (
        DescriptionHardOracleCallbacks,
        QDBSAuthorityMode,
        run_fd2_qdbs_terminal,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_archive = args.archive.read_bytes()
    base_dsegs, base_dposes, cell_flips = _load_p1(args.p1_dir)
    resid = _quant_residuals(args.checkpoint)

    oracle = TR1Oracle(base_archive, base_dsegs, base_dposes,
                       args.gt_cache, args.pose_targets)
    rebuilt = oracle.compile_archive(oracle.base_codes.reshape(-1), None)
    if rebuilt != base_archive:
        raise SystemExit(
            "base recompile is NOT byte-identical to the exporter archive")
    print("[determinism] base recompile byte-identical", flush=True)

    singles, groups, active = build_proposals(
        cell_flips, resid, oracle.base_codes)
    print(f"[proposals] 16 singletons + 8 groups; active set {active.size}",
          flush=True)

    callbacks = DescriptionHardOracleCallbacks(
        compile_archive=oracle.compile_archive,
        parse_archive=oracle.parse_archive,
        consume_archive=oracle.consume_archive,
        evaluate_joint_action_idempotent=(
            oracle.evaluate_joint_action_idempotent),
    )
    t0 = time.time()
    result = run_fd2_qdbs_terminal(
        oracle.base_codes.reshape(-1),
        singles,
        groups,
        callbacks,
        active_indices=[int(i) for i in active],
        seed=args.seed,
        authority_mode=QDBSAuthorityMode.STALE_REHEARSAL,
    )
    wall = time.time() - t0

    payload = result.to_payload() if hasattr(result, "to_payload") else {
        "repr": repr(result)}

    winner_info = None
    best_id = result.best_strict_improvement_identity
    if best_id is not None:
        winner_prop = None
        for cand in result.schedule.candidates:
            if cand.identity == best_id:
                winner_prop = cand
                break
        if winner_prop is None:
            raise SystemExit(f"winner {best_id!r} missing from schedule")
        theta = oracle.base_codes.reshape(-1).copy()
        for d in winner_prop.deltas:
            theta[d.index] += d.delta
        winner_archive = oracle.compile_archive(theta, None)
        winner_path = args.out_dir / "p2a_winner_archive.zip"
        winner_path.write_bytes(winner_archive)
        winner_info = {
            "identity": best_id,
            "deltas": [d.to_payload() for d in winner_prop.deltas],
            "archive_sha256": _sha256_bytes(winner_archive),
            "archive_bytes": len(winner_archive),
            "path": str(winner_path),
        }
    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "authority_mode_note": (
            "STALE_REHEARSAL chosen because ContestAxis has no advisory"
            " member; evaluations are full-population incremental-exact on"
            " the deployed bytes, [macOS-CPU advisory]"),
        "proposal_recipe": SIGNAL_LABEL,
        "base_archive_sha256": _sha256_bytes(base_archive),
        "base_d_seg_mean": float(base_dsegs.mean()),
        "base_d_pose_mean": float(base_dposes.mean()),
        "base_joint_action": float(
            100.0 * base_dsegs.mean()
            + float(np.sqrt(10.0 * base_dposes.mean()))
            + 25.0 * len(base_archive) / 37_545_489.0),
        "seed": args.seed,
        "active_indices_count": int(active.size),
        "evaluations": oracle.eval_count,
        "evaluation_wall_seconds": oracle.eval_wall,
        "total_wall_seconds": wall,
        "result": payload,
        "winner": winner_info,
        "generated_by": "tools/pb1_qdbs_tr1.py",
    }
    out = args.out_dir / "p2a_qdbs_receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True,
                              default=str) + "\n")
    print(f"[done] wall {wall:.0f}s evals {oracle.eval_count}; "
          f"receipt {out}", flush=True)


if __name__ == "__main__":
    main()
