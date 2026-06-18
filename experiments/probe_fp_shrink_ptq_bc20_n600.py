#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""FP-SHRINK rate-lever $0 PTQ smoke on the TRUSTWORTHY bc20 n600 basin (task #136).

THE DECISIVE $0 QUESTION (CLAUDE.md "THE GOAL — SUB-0.15"): the bc20 small-basis basin
ships its decoder weights as per-tensor symmetric INT8 (the shipped codec). The rate term
is ~0.0594 (25 * 89,136 / 37,545,489) and the decoder weights dominate the 89KB. FP-shrink
= re-quantize the decoder weights to fewer bits to cut bytes. The rate saving is MECHANICAL;
the OPEN question is the DISTORTION-HOLD: does d_seg/d_pose hold under fp8/fp6/fp5/fp4 weight
quantization? If a sub-8-bit grid HOLDS d_seg/d_pose ≈ int8 baseline while cutting bytes →
real (near-)free rate win toward sub-0.15. If naive PTQ collapses d_seg → the verdict is
"QAT/LSQ needed" (NOT built here; that is the next gated step).

WHY THIS IS NOT A DUPLICATE of ``probe_fp4_dseg_hold_smoke.py`` / ``probe_fp4_pctl_retest.py``
(SEARCH-AND-FAMILIARIZE, CLAUDE.md):
  * Those ran on the ``from0_ab_v2_n96`` basin (76,592 B, 96-pair MEMORIZED operating point).
    This runs on ``torch_vehicle_full_mps_basin_bc20_n600`` (89,136 B) — the G3-VALIDATED
    basin where local-CPU d_seg/d_pose ≈ exact contest-CPU to ~0.001% (trustworthy).
  * Those ESTIMATED int-N bytes via a rough nbits/8 scaling of the decoder blob ("ignores
    brotli interplay" — their own docstring). This MEASURES the real archive bytes: it
    re-encodes each shrunk-weight state dict through the SAME vendored ``build_archive`` and
    reads the actual brotli output size (the honest "ship int-N via the existing int8 codec"
    rate). Latents + meta are byte-identical across modes (only the decoder grid changes).
  * Full 600-pair authority eval (not a 96-pair subset), via the canonical
    ``RealScorerContext.exact_eval`` (the vendored ``evaluate_decoder`` + ``compute_score``).

REUSE (no scorer / codec / quant reimplemented):
  * ``RealScorerContext.exact_eval`` — the canonical full-600-pair d_seg/d_pose/rate/score.
  * vendored ``codec.parse_archive`` / ``codec.build_archive`` — the real archive bytes.
  * ``tac.post_hoc_weight_shrink.requantize_decoder_state_dict`` — the reusable qdq grids.

AUTHORITY: ``[contest-CPU advisory] NON-PROMOTABLE``. Local CPU ≈ contest-CPU per G3, but
this is a FEASIBILITY smoke, NOT a byte-closed dual-exact row. The frontier pointer is
UNMOVED. NO score/frontier/promotion claim. CPU only (NO MPS — MPS owns the live train and
2x-corrupts SegNet / 23x PoseNet). $0 — no GPU dispatch, no paid spend.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

_RATE_DENOM = 37_545_489
_BASIN = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best")
# int8 first (codec baseline), then descending bit-width, then the FP4 grids.
_MODES = ("fp32", "int8", "int7", "int6", "int5", "int4", "fp4_mixed", "fp4_all")


def _measure_archive_bytes(codec, dec_sd: dict, latents: torch.Tensor, meta: dict) -> int:
    """Re-encode the (possibly shrunk) decoder + IDENTICAL latents/meta through the SHIPPED
    vendored ``build_archive`` and return the real archive byte length. The codec int8-stores
    + zigzag + brotli's; feeding int-N-qdq'd weights means the int8 store holds only the
    sparser int-N levels → the measured brotli size IS the honest int-N rate."""
    archive = codec.build_archive(dec_sd, latents, meta)
    return len(archive)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basin", default=str(_BASIN))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--modes", nargs="*", default=list(_MODES))
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))

    from tac.post_hoc_weight_shrink import requantize_decoder_state_dict
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    model_mod = import_vendored("model")
    codec = import_vendored("codec")
    vp = import_vendored("data").get_default_video_path()

    basin = Path(args.basin)
    arch0 = (basin / "best_archive.bin").read_bytes()
    dec_sd, latents, meta = codec.parse_archive(arch0)
    latents = latents.float()
    n_pairs = int(latents.shape[0])
    latent_dim = int(meta["latent_dim"])
    base_channels = int(meta["base_channels"])
    eval_size = tuple(meta["eval_size"])

    # meta for build_archive must match what parse_archive expects on re-parse.
    build_meta = {
        "n_pairs": n_pairs,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "eval_size": list(eval_size),
    }

    print(f"[basin] {basin}  archive0_bytes={len(arch0)}  n_pairs={n_pairs} "
          f"latent_dim={latent_dim} base_channels={base_channels}", flush=True)

    # parse-back baseline bytes via build_archive (sanity: should ≈ arch0 for int8-equivalent).
    rebuilt0 = _measure_archive_bytes(codec, dec_sd, latents, build_meta)
    print(f"[parse-back rebuild bytes] {rebuilt0}  (vs on-disk {len(arch0)})", flush=True)

    ctx = RealScorerContext(
        vp, device="cpu", train_device="cpu", split_by_head=False,
        max_pairs=n_pairs, targets_cache="experiments/results/capstone_gt_targets_cache",
    )

    def build_decoder(shrunk_sd: dict):
        d = model_mod.HNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels, eval_size=eval_size
        ).eval()
        d.load_state_dict(shrunk_sd)
        return d

    rows: dict[str, dict] = {}
    for mode in args.modes:
        t0 = time.time()
        shrunk = requantize_decoder_state_dict(dec_sd, mode)
        # MEASURED bytes: re-encode the shrunk weights through the real shipping codec.
        bytes_meas = _measure_archive_bytes(codec, shrunk, latents, build_meta)
        # full 600-pair authority eval on the shrunk decoder.
        res = ctx.exact_eval(build_decoder(shrunk), latents, bytes_meas)
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        rate = 25.0 * bytes_meas / _RATE_DENOM
        score = 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + rate
        rows[mode] = {
            "d_seg": d_seg, "d_pose": d_pose, "archive_bytes": bytes_meas,
            "rate_term": rate, "score": score, "wall_s": round(time.time() - t0, 1),
        }
        print(f"[{mode:9}] d_seg={d_seg:.6f}  d_pose={d_pose:.6f}  bytes={bytes_meas:>6}  "
              f"rate={rate:.4f}  S={score:.4f}  ({rows[mode]['wall_s']}s)  "
              f"[contest-CPU advisory]", flush=True)

    # ── verdict ──────────────────────────────────────────────────────────────
    i8 = rows.get("int8")
    report = {
        "probe": "fp_shrink_ptq_bc20_n600",
        "task_id": 136,
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "basin": str(basin),
        "archive0_bytes_on_disk": len(arch0),
        "parse_back_rebuild_bytes": rebuilt0,
        "n_pairs": n_pairs,
        "rate_denom": _RATE_DENOM,
        "rows": rows,
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Bytes are MEASURED (vendored build_archive re-encode + real brotli), not nbits/8 "
            "estimated. d_seg/d_pose are full-600-pair exact SegNet argmax-flip / PoseNet MSE "
            "via RealScorerContext.exact_eval. Local CPU ≈ contest-CPU per G3."
        ),
    }
    if i8:
        i8_seg = i8["d_seg"]
        i8_pose = i8["d_pose"]
        i8_score = i8["score"]
        # PTQ collapse threshold: lowest bit-width holding d_seg within +10% AND d_pose within +10% of int8.
        ranked = sorted(rows.items(), key=lambda kv: kv[1]["score"])
        s_min_mode, s_min = ranked[0]
        holds = {}
        for m, r in rows.items():
            if m in ("fp32",):
                continue
            seg_rel = (r["d_seg"] - i8_seg) / max(i8_seg, 1e-12)
            pose_rel = (r["d_pose"] - i8_pose) / max(i8_pose, 1e-12)
            holds[m] = {
                "seg_rel_vs_int8_pct": round(100 * seg_rel, 1),
                "pose_rel_vs_int8_pct": round(100 * pose_rel, 1),
                "delta_S_vs_int8": round(r["score"] - i8_score, 5),
                "holds_dseg_within_10pct": bool(seg_rel <= 0.10),
                "holds_dpose_within_10pct": bool(pose_rel <= 0.10),
            }
        # The ONLY criterion for a real win is a LOWER net S (ΔS < 0). The ±10%
        # relative d_seg/d_pose hold gate is NOT a win criterion: the 100× d_seg
        # weight means even a small relative d_seg spill can raise S more than the
        # rate saving (NO FAKE: a "hold within 10%" is not a score win).
        intn_candidates = ("int4", "int5", "int6", "int7")
        all_shrink_modes = (*intn_candidates, "fp4_mixed", "fp4_all")
        # net-S winner: any sub-8-bit grid with ΔS < 0 vs int8 (none expected).
        s_winner = None
        for m in all_shrink_modes:
            if (
                m in rows
                and (rows[m]["score"] - i8_score) < -1e-9
                and (s_winner is None or rows[m]["score"] < rows[s_winner]["score"])
            ):
                s_winner = m
        # collapse threshold: lowest bit-width with ΔS still < a small slack (the
        # last grid that is "int8-class" on S, not a win, just not-yet-collapsed).
        near_int8 = None
        for m in ("int7", "int6", "int5", "int4"):
            if m in rows and (rows[m]["score"] - i8_score) <= 0.02:
                near_int8 = m
        report["verdict"] = {
            "int8_baseline": {"d_seg": i8_seg, "d_pose": i8_pose, "score": i8_score,
                              "bytes": i8["archive_bytes"]},
            "S_min_mode": s_min_mode,
            "S_min": s_min["score"],
            "S_min_delta_vs_int8": round(s_min["score"] - i8_score, 5),
            "holds_vs_int8": holds,
            "net_S_winner_vs_int8": s_winner,  # None => no bit-width lowers S
            "best_intn_near_int8_dS_le_0p02": near_int8,
            # byte axis (real, mechanical): int4 cuts the most bytes.
            "byte_axis": {
                m: {
                    "archive_bytes": rows[m]["archive_bytes"],
                    "delta_bytes_vs_int8": rows[m]["archive_bytes"] - i8["archive_bytes"],
                    "delta_rate_vs_int8": round(rows[m]["rate_term"] - i8["rate_term"], 5),
                }
                for m in all_shrink_modes if m in rows
            },
        }
        if s_winner is not None:
            fr = rows[s_winner]
            report["verdict"]["headline"] = (
                f"PTQ-WINS: {s_winner} LOWERS net S (ΔS {fr['score'] - i8_score:+.5f}; "
                f"d_seg {fr['d_seg']:.6f}, bytes {fr['archive_bytes']} vs {i8['archive_bytes']}) "
                f"→ a real NO-QAT rate win. NEXT: byte-close + dual exact eval."
            )
        else:
            int4 = rows.get("int4", {})
            report["verdict"]["headline"] = (
                "PTQ-COLLAPSES: NO naive sub-8-bit grid lowers net S — every bit-width "
                f"raises S (int7 ΔS {rows['int7']['score'] - i8_score:+.5f}, "
                f"int6 {rows['int6']['score'] - i8_score:+.5f}, int5 {rows['int5']['score'] - i8_score:+.5f}, "
                f"int4 {int4.get('score', 0) - i8_score:+.5f}) because the 100× d_seg weight "
                "outweighs the rate saving. The RATE win is REAL on bytes (int4 "
                f"{int4.get('archive_bytes', 0)} vs {i8['archive_bytes']} = "
                f"Δrate {int4.get('rate_term', 0) - i8['rate_term']:+.4f}, matching the −0.022/−0.029 "
                "estimate) but the un-QAT'd d_seg/d_pose spill buries it. QAT/LSQ is the unblock "
                "(the next gated step, NOT built here)."
            )
        print("\n=== FP-SHRINK PTQ VERDICT (bc20 n600, MEASURED bytes) ===")
        print(f"  {report['verdict']['headline']}")
        print(f"  S-min mode: {s_min_mode}  S={s_min['score']:.4f}  (int8 S={i8_score:.4f})  "
              f"net_S_winner={s_winner}")

    text = json.dumps(report, indent=2)
    print()
    print(text)
    out = Path(args.out_json) if args.out_json else Path("reports/fp_shrink_ptq_bc20_n600.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\n[wrote] {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
