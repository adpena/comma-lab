# SPDX-License-Identifier: MIT
"""#425 dash-phase carrier — $0 n600 RATE + RECOVERY measurement ([macOS-CPU advisory]).

RATE: encodes the carrier on the frozen SegNet-argmax cache (gt_n600.npz lstars + gt_poses)
and reports the section bytes against (a) the world-frame per-dash-anchor budget band
0.9–1.8 KB (`lane_channel_deep_refactorization_20260716.md` §5, DERIVED), (b) the naive
per-frame partition baseline 1,003 B/frame (`temporal_advection_stratified_20260715.md`,
MEASURED zlib-9), (c) a lane-only naive baseline measured here (zlib-9 of the per-frame
Lane packbits mask), and (d) the naive per-frame anchor stream (n_dashes × anchor bytes).

RECOVERY (label-space, HONEST SCOPE): the through-R d_seg A/B needs the trainer render
plumbing against the live-run checkpoint (run dir SACRED; 60 GiB trainer live) — NOT run
here. Instead we measure at the PARTITION/argmax level, and SAY SO:
  * centroid-offset distributions (d=0 / ≤1 / ≤2 / >2 px) of the TRANSPORT-ONLY prediction
    vs the PHASE-CORRECT decode against the observed GT dash centroids — the direct
    curve-domain analogue of the memo's separatrix jitter prior;
  * Lane-layer raster substitution on matched tracks: previous-frame observed island pixels
    placed by (persist | transport-only | phase-correct) vs the GT Lane mask (XOR px count)
    — a SHAPE-PERSISTENCE approximation (dash pixels persist, only phase moves), which is
    exactly the carrier's semantics.
NEVER a score: `score_claim=false`, `promotable=false`; the pointer moves only through
``upstream/evaluate.py`` on exact archive bytes.

Usage:
  .venv/bin/python tools/measure_dash_phase_carrier_n600.py \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --out experiments/results/dash_phase_carrier_n600_20260717/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from tac.boundary_math.dash_phase_carrier import (  # noqa: E402
    DashPhaseConfig,
    encode_dash_phase_carrier,
    extract_dash_observations,
)
from tac.boundary_math.warp_real_luma_frame0 import xi_from_pose_calibration  # noqa: E402

# MEASURED references (cited, not re-derived here)
NAIVE_PARTITION_BYTES_PER_FRAME = 1003.0  # temporal_advection_stratified n600, zlib-9, ALL classes
ANCHOR_BUDGET_BAND_BYTES = (900, 1800)    # lane_channel refactor §5 world-frame anchors (DERIVED)
BIRTH_ANCHOR_BYTES = 5.5                  # u16+u16+4b tilt+~varint area ≈ 44 bits (this codec's anchor)


def _offset_hist(dists: np.ndarray) -> dict[str, float]:
    if dists.size == 0:
        return {"n": 0, "d0": 0.0, "le1": 0.0, "le2": 0.0, "gt2": 0.0, "mean_px": 0.0}
    return {
        "n": int(dists.size),
        "d0": float(np.mean(dists < 0.5)),
        "le1": float(np.mean(dists <= 1.0)),
        "le2": float(np.mean(dists <= 2.0)),
        "gt2": float(np.mean(dists > 2.0)),
        "mean_px": float(dists.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--out", default="experiments/results/dash_phase_carrier_n600_20260717/results.json")
    ap.add_argument("--match-radius", type=float, default=6.0)
    ap.add_argument("--dormant-max", type=int, default=30)
    ap.add_argument("--pitch", type=float, default=-0.01,
                    help="ground-plane pitch (the grok/advection-memo calibration default -0.01).")
    ap.add_argument("--s-t", type=float, default=-0.00322,
                    help="pose->xi translation calibration (MEASURED advection-memo fit -0.00322).")
    ap.add_argument("--s-r", type=float, default=0.0,
                    help="pose->xi rotation calibration (MEASURED advection-memo fit 0.0).")
    args = ap.parse_args()

    z = np.load(args.gt_cache, allow_pickle=False)
    P = int(min(args.n_pairs, z["lstars"].shape[0]))
    lstars = np.asarray(z["lstars"])[:P]
    gt_poses = np.asarray(z["gt_poses"])[:P]
    cfg = DashPhaseConfig(match_radius_px=args.match_radius, dormant_max_frames=args.dormant_max,
                          pitch=args.pitch, s_t=args.s_t, s_r=args.s_r)

    xi = np.stack([
        xi_from_pose_calibration(gt_poses[p], s_t=cfg.s_t, s_r=cfg.s_r, pitch=cfg.pitch)
        for p in range(P)
    ])

    telemetry: list[dict] = []
    section, report, _dec = encode_dash_phase_carrier(lstars, xi, cfg, telemetry=telemetry)

    # ------------------------- observations (deterministic re-extraction) ---------------
    observations = [extract_dash_observations(lstars[p], cfg) for p in range(P)]
    n_obs_total = sum(len(o) for o in observations)
    obs_per_frame = n_obs_total / P

    # ------------------------- RATE table ------------------------------------------------
    lane_naive_bytes = 0
    for p in range(P):
        lane_naive_bytes += len(zlib.compress(np.packbits(lstars[p] == cfg.lane_class).tobytes(), 9))
    naive_anchor_stream = n_obs_total * BIRTH_ANCHOR_BYTES

    rate = {
        "section_bytes_total": report.section_bytes,
        "section_bytes_excl_xi": report.section_bytes_excl_xi,
        "xi_bytes_in_section": report.xi_bytes,
        "xi_marginal_note": "dxi already banked for d_pose (L68, 7.2KB) -> 0 marginal in composition",
        "per_frame_bytes_excl_xi": report.section_bytes_excl_xi / P,
        "vs_anchor_budget_band_bytes": list(ANCHOR_BUDGET_BAND_BYTES),
        "over_budget_factor_vs_1800B": report.section_bytes_excl_xi / ANCHOR_BUDGET_BAND_BYTES[1],
        "naive_partition_bytes_600f_allclass_cited": NAIVE_PARTITION_BYTES_PER_FRAME * P,
        "lane_only_naive_packbits_zlib9_bytes_measured": lane_naive_bytes,
        "naive_per_frame_anchor_stream_bytes": naive_anchor_stream,
        "wins_vs_lane_only_naive": report.section_bytes_excl_xi < lane_naive_bytes,
        "wins_vs_naive_anchor_stream": report.section_bytes_excl_xi < naive_anchor_stream,
        "delta_stream_prior_code_bytes": report.prior_code_delta_bytes,
        "delta_stream_zlib9_bytes": report.zlib9_delta_stream_bytes,
        "prior_code_beats_zlib9": report.prior_code_delta_bytes <= report.zlib9_delta_stream_bytes,
        "bit_breakdown": {
            "alive_bits": report.alive_bits, "delta_bits": report.delta_bits,
            "birth_bits": report.birth_bits, "rebirth_bits": report.rebirth_bits,
        },
    }

    # ------------------------- RECOVERY (label-space; honest scope) ----------------------
    # centroid-offset distributions
    m_ev = [e for e in telemetry if e["kind"] in ("match", "rebirth")]
    d_transport = np.array([
        float(np.hypot(e["pred_rc"][0] - e["obs_rc"][0], e["pred_rc"][1] - e["obs_rc"][1])) for e in m_ev
    ])
    d_phase = np.array([
        float(np.hypot(e["dec_rc"][0] - e["obs_rc"][0], e["dec_rc"][1] - e["obs_rc"][1])) for e in m_ev
    ])
    births_after0 = [e for e in telemetry if e["kind"] == "birth" and e["frame"] > 0]
    coverage = {
        "observed_dashes_after_frame0": n_obs_total - len(observations[0]),
        "transport_covered_fraction": len(m_ev) / max(1, n_obs_total - len(observations[0])),
        "event_coded_birth_fraction": len(births_after0) / max(1, n_obs_total - len(observations[0])),
        "note": "transport-only leaves births UNCOVERED; the carrier's event codes cover 100% "
                "(births are anchor-coded, so phase-correct coverage = 1.0 by construction)",
    }

    # Lane-layer raster substitution (matched tracks; shape-persistence approximation)
    xor_persist = xor_transport = xor_phase = 0
    gt_lane_px = 0
    n_sub_frames = 0
    track_last: dict[int, tuple[int, int]] = {}
    events_sorted = sorted(telemetry, key=lambda e: (e["frame"],))
    ptr = 0
    for p in range(P):
        frame_events = []
        while ptr < len(events_sorted) and events_sorted[ptr]["frame"] == p:
            frame_events.append(events_sorted[ptr])
            ptr += 1
        if p > 0:
            Hh, Ww = lstars[p].shape
            gt_mask = lstars[p] == cfg.lane_class
            rec_persist = np.zeros_like(gt_mask)
            rec_transport = np.zeros_like(gt_mask)
            rec_phase = np.zeros_like(gt_mask)
            any_sub = False
            for e in frame_events:
                if e["kind"] not in ("match", "rebirth") or e["track_id"] not in track_last:
                    continue
                pf, poi = track_last[e["track_id"]]
                prev_o = observations[pf][poi]
                prev_c = prev_o.centroid_rc
                for rec, tgt in (
                    (rec_persist, prev_c),
                    (rec_transport, e["pred_rc"]),
                    (rec_phase, e["dec_rc"]),
                ):
                    dr = int(round(tgt[0] - prev_c[0]))
                    dc = int(round(tgt[1] - prev_c[1]))
                    rr = prev_o.pixel_rows + dr
                    cc = prev_o.pixel_cols + dc
                    ok = (rr >= 0) & (rr < Hh) & (cc >= 0) & (cc < Ww)
                    rec[rr[ok], cc[ok]] = True
                any_sub = True
            if any_sub:
                n_sub_frames += 1
                # restrict the GT reference to the SAME matched-track footprint scope:
                # XOR against the union of (GT pixels of the matched observations)
                gt_scope = np.zeros_like(gt_mask)
                for e in frame_events:
                    if e["kind"] in ("match", "rebirth"):
                        o = observations[p][e["obs_index"]]
                        gt_scope[o.pixel_rows, o.pixel_cols] = True
                gt_lane_px += int(gt_scope.sum())
                xor_persist += int(np.logical_xor(rec_persist, gt_scope).sum())
                xor_transport += int(np.logical_xor(rec_transport, gt_scope).sum())
                xor_phase += int(np.logical_xor(rec_phase, gt_scope).sum())
        # update history AFTER using previous state
        for e in frame_events:
            if e["kind"] in ("match", "rebirth", "birth"):
                track_last[e["track_id"]] = (p, e["obs_index"])

    recovery = {
        "authority": "[macOS-CPU advisory] LABEL-SPACE (partition/argmax level). NOT through-R, NOT a "
                     "score: through-R d_seg A/B on the c2 ep725 EMA is OWED (trainer render plumbing + "
                     "live-run memory gate; run dir SACRED).",
        "centroid_offset_transport_only": _offset_hist(d_transport),
        "centroid_offset_phase_correct": _offset_hist(d_phase),
        "memo_jitter_prior_screw_cited": {"d0": 0.404, "le1": 0.723, "le2": 0.798, "gt2": 0.202},
        "coverage": coverage,
        "lane_layer_raster_substitution_matched_tracks": {
            "scope": "matched/rebirth tracks only; shape-persistence approximation (prev observed island "
                     "pixels placed at target centroid); GT reference = matched observations' own pixels",
            "n_frames_with_substitution": n_sub_frames,
            "gt_scope_lane_px": gt_lane_px,
            "xor_px_persist": xor_persist,
            "xor_px_transport_only": xor_transport,
            "xor_px_phase_correct": xor_phase,
            "xor_rate_persist": xor_persist / max(1, gt_lane_px),
            "xor_rate_transport_only": xor_transport / max(1, gt_lane_px),
            "xor_rate_phase_correct": xor_phase / max(1, gt_lane_px),
        },
    }

    out = {
        "utc": "2026-07-17",
        "authority": "[macOS-CPU advisory] research-signal; score_claim=false; promotable=false",
        "tool": "tools/measure_dash_phase_carrier_n600.py",
        "gt_cache": str(args.gt_cache),
        "n_frames": P,
        "config": {
            "lane_class": cfg.lane_class, "min_area": cfg.min_area, "border_px": cfg.border_px,
            "match_radius_px": cfg.match_radius_px, "q_px": cfg.q_px,
            "dormant_max_frames": cfg.dormant_max_frames, "pitch": cfg.pitch,
            "s_t": cfg.s_t, "s_r": cfg.s_r,
        },
        "dash_stats": {
            "n_obs_total": n_obs_total, "obs_per_frame": obs_per_frame,
            "n_tracks_total": report.n_tracks_total,
            "n_matched": report.n_matched, "n_births": report.n_births,
            "n_rebirths": report.n_rebirths, "n_deaths": report.n_deaths,
            "blink_back_fraction": report.blink_back_fraction,
            "memo_cited_islands_per_frame": 20.6,
            "memo_cited_births_per_step_upper": 9.43,
        },
        "code": {
            "expected_bits_per_dash_prior_preregistered": report.expected_bits_per_dash_prior,
            "measured_bits_per_matched_dash": report.measured_bits_per_matched_dash,
            "esc_rate": report.esc_rate,
            "symbol_histogram": report.symbol_histogram,
            "mean_abs_delta_px": report.mean_abs_delta_px,
        },
        "rate": rate,
        "recovery": recovery,
        "reconstruction_bit_identical": report.reconstruction_bit_identical,
        "section_sha_note": "deterministic: same cache + config -> same bytes",
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[dash-phase-carrier n600] wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
