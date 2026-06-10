#!/usr/bin/env python
"""Cross-pair waterfilled POSE corrector exact-scorer smoke (task #54).

THE NEAR-TERM EXACT-ROW TEST (operator spec): d_pose is a GLOBAL pooled budget. This wires the real
frozen CPU-torch PoseNet observer to ``tac.optimization.cross_pair_waterfilled_corrector`` and runs the
cross-pair waterfiller on the CURRENT frontier carrier:

  * BASE = the 177,169 B frontier archive's decoded comp pairs WITH its existing FEC6 K=16 frame-0
    selector applied (the live selector). Per-pair base d_pose measured on the EXACT DistortionNet
    PoseNet (GT via ``frame_utils.yuv420_to_rgb`` ONLY -- NEVER PyAV rgb24, NEVER MPS).
  * CANDIDATE actions = for each pair, every alternative FEC6 K=16 frame-0 mode. A frame-0 change is
    SegNet-blind by construction (SegNet reads frame1 only via ``x[:, -1, ...]``), so d_seg is EXACTLY
    untouched -- the corrector is a pure pose-axis move.
  * BYTE COST = the FEC6 fixed-Huffman code-length DELTA of switching the pair's selector index, in
    bytes (the selector is a 600-pair bitstream; switching a pair's mode changes only its code length).
  * The waterfiller equalizes the marginal at lambda* = 25/D = 6.66e-7 against the GLOBAL pooled
    d_pose (the marginal grows as d_pose falls).

THE HONEST VERDICT (operator + the #55 honesty rule): the row reports
  {pooled d_pose before->after, delta_bytes, new_bad (pairs worsened), net delta_score, beats_base}.
``new_bad`` MUST be 0 for a correct allocator (it never admits a pair-worsening action). The net
delta_score is the verdict -- did cross-pair waterfilled allocation move the EXACT score below the
frontier (NET-positive after the byte cost)?

Authority: ``[local CPU-torch advisory]`` -- non-promotable. $0, no GPU, no MPS. The verdict is a
candidate-generator signal; a real frontier move requires paired CPU+CUDA exact eval on the byte-closed
archive (gated on beats_base).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
_RUNTIME_SRC = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/runtime/src"
_SUBMISSION = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/submission"
_UPSTREAM = REPO_ROOT / "upstream"
for _p in (str(_HARNESS), str(_RUNTIME_SRC), str(_UPSTREAM), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.optimization.cross_pair_waterfilled_corrector import (  # noqa: E402
    CrossPairPoseWaterfiller,
    constant_correction_result,
)

DEVICE = torch.device("cpu")

# FEC6 K=16 active palette + fixed-Huffman code lengths (from submission/inflate.py).
FEC6_K16_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)
FEC6_K16_CODE_BITS = (
    "00", "1100", "01", "111010", "11010", "111011", "111100", "100",
    "111101", "11011", "1111110", "111110", "11111110", "101", "11100", "11111111",
)
MODE_TO_K16 = {m: i for i, m in enumerate(FEC6_K16_MODE_IDS)}
MODE_CODE_LEN_BITS = {m: len(FEC6_K16_CODE_BITS[i]) for i, m in enumerate(FEC6_K16_MODE_IDS)}


class FrontierPoseObserver:
    """Exact CPU-torch PoseNet observer over the frontier carrier (SegNet-blind frame-0 modes).

    Measures, for the sampled pairs, the base per-pair d_pose (frontier render + live selector) and
    the d_pose under each alternative FEC6 K=16 frame-0 mode. The byte cost of switching a pair's mode
    is the Huffman code-length delta (bits/8). All exact: DistortionNet PoseNet, GT via
    ``yuv420_to_rgb``, NEVER MPS.
    """

    def __init__(self, pair_indices, *, candidate_modes):
        import render_and_score_lib as L
        from frame_selector import apply_frame0_mode

        self._L = L
        self._apply_frame0_mode = apply_frame0_mode
        self.pair_indices = list(pair_indices)
        self.candidate_modes = tuple(candidate_modes)
        self._scorer = L.ExactScorer()

        # decode GT pairs + render frontier comp pairs (NO selector -- we apply modes explicitly).
        self._gts = L.decode_gt_pairs(self.pair_indices)  # pi -> (2,H,W,3) uint8
        self._comps = L.FrontierRenderer().render_baseline_pairs(self.pair_indices)  # pi -> (2,3,H,W)

        # the live selector: which mode the frontier currently assigns each sampled pair.
        self._live_modes = self._extract_live_selector_modes()

        # cache base residuals + per-(pair,mode) residuals (one exact eval each).
        self._idx_of = {pi: k for k, pi in enumerate(self.pair_indices)}
        self._base = np.zeros(len(self.pair_indices), dtype=np.float64)
        self._mode_cache: dict[tuple[int, str], float] = {}
        self._measure_base()

    # --- live selector extraction (the frontier's current per-pair mode) ---
    def _extract_live_selector_modes(self) -> dict[int, str]:
        """Decode the live FEC6 selector codes for the sampled pairs (best-effort; default none)."""

        try:
            from tac.optimization.scorer_region_waterfill import decode_selector_codes

            codes = decode_selector_codes(_SUBMISSION)
            out = {}
            for pi in self.pair_indices:
                if pi < len(codes):
                    idx = int(codes[pi])
                    out[pi] = FEC6_K16_MODE_IDS[idx] if 0 <= idx < len(FEC6_K16_MODE_IDS) else "none"
                else:
                    out[pi] = "none"
            return out
        except Exception:
            # fall back: treat the rendered baseline as the base (selector already baked in render
            # path is NOT -- render_baseline_pairs excludes selector; so default to "none").
            return dict.fromkeys(self.pair_indices, "none")

    def _gt_pair(self, pi):
        g = self._gts[pi]
        return torch.stack([g[0], g[1]]).float()  # (2,H,W,3)

    def _comp_under_mode(self, pi, mode_id):
        """Frontier comp pair with frame-0 ``mode_id`` applied (clamped+rounded)."""

        comp = self._comps[pi].clone().float()  # (2,3,H,W)
        if mode_id != "none":
            comp[0] = self._apply_frame0_mode(comp[0], mode_id)
        return comp.clamp_(0.0, 255.0).round_()

    def _score_pose(self, pi, comp_pair):
        gt = self._gt_pair(pi).unsqueeze(0)  # (1,2,H,W,3)
        comp_bthwc = comp_pair.permute(0, 2, 3, 1).unsqueeze(0)  # (1,2,H,W,3)
        pose, _seg = self._scorer.score_batch(gt, comp_bthwc)
        return float(pose[0])

    def _measure_base(self):
        for k, pi in enumerate(self.pair_indices):
            live = self._live_modes.get(pi, "none")
            self._base[k] = self._score_pose(pi, self._comp_under_mode(pi, live))

    # --- observer protocol ---
    def base_pose_residuals(self) -> np.ndarray:
        return self._base.copy()

    def pose_residual_under_mode(self, pair_index: int, mode_id: str) -> float:
        key = (pair_index, mode_id)
        if key not in self._mode_cache:
            pi = self.pair_indices[pair_index]
            self._mode_cache[key] = self._score_pose(pi, self._comp_under_mode(pi, mode_id))
        return self._mode_cache[key]

    def mode_byte_cost(self, pair_index: int, mode_id: str) -> float:
        """Huffman code-length delta of switching this pair from its live mode to ``mode_id`` (bytes).

        The selector is a single concatenated bitstream; switching one pair's code changes the total
        bit length by ``len(code(mode_id)) - len(code(live_mode))``. We charge the byte-equivalent
        (bits/8) so the rate term is in the same units as the archive size. A switch to a longer code
        costs positive bytes; to a shorter code frees bytes (negative cost is allowed)."""

        pi = self.pair_indices[pair_index]
        live = self._live_modes.get(pi, "none")
        live_bits = MODE_CODE_LEN_BITS.get(live, 4)
        new_bits = MODE_CODE_LEN_BITS.get(mode_id, 4)
        return (new_bits - live_bits) / 8.0


def _archive_bytes() -> int:
    p = _SUBMISSION / "archive.zip"
    return p.stat().st_size if p.exists() else 177_169


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=40, help="number of pairs to sample (smoke).")
    ap.add_argument("--all-600", action="store_true", help="use all 600 pairs (heavy, ~minutes).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT / "experiments/results/cross_pair_waterfilled_corrector_20260610/pose_smoke.json"),
    )
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.all_600:
        pair_indices = list(range(600))
    else:
        rng = np.random.default_rng(args.seed)
        pair_indices = sorted(int(x) for x in rng.choice(600, size=min(args.n_pairs, 600), replace=False))

    t0 = time.time()
    candidate_modes = FEC6_K16_MODE_IDS
    obs = FrontierPoseObserver(pair_indices, candidate_modes=candidate_modes)

    bytes_before = _archive_bytes()
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=candidate_modes)
    result = wf.run(bytes_before=bytes_before)

    # constant control: the single best-on-average alternative mode applied to every sampled pair.
    # find the alt mode with lowest mean residual across sampled pairs.
    best_const_mode, best_const_mean = None, float("inf")
    for m in candidate_modes:
        if m == "none":
            continue
        vals = [obs.pose_residual_under_mode(k, m) for k in range(len(pair_indices))]
        mean_v = float(np.mean(vals))
        if mean_v < best_const_mean:
            best_const_mean, best_const_mode = mean_v, m
    const_byte_cost = float(np.mean([obs.mode_byte_cost(k, best_const_mode) for k in range(len(pair_indices))]))
    const_result = constant_correction_result(
        obs, best_const_mode, bytes_before=bytes_before, byte_cost_per_pair=const_byte_cost
    )

    # DIAGNOSTIC (the honest "why unmoved"): per sampled pair, the best available alternative-mode
    # improvement (negative = a pair the live selector left improvable) + its value-per-byte vs
    # lambda*. Characterizes whether the frontier selector is already per-pair-optimal (no improvable
    # pairs) OR there are improvements but all under the water level.
    from tac.optimization.cross_pair_waterfilled_corrector import WATER_LEVEL_LAMBDA_STAR

    base = obs.base_pose_residuals()
    improvable = 0
    best_improvements = []  # (pair_local, delta_pair, byte_cost, vpb_at_base_pool)
    pooled_base = float(base.mean())
    from tac.optimization.cross_pair_waterfilled_corrector import pose_score_term

    for k in range(len(pair_indices)):
        best_dp = 0.0
        best_m = None
        best_bc = 0.0
        for m in candidate_modes:
            if m == "none":
                continue
            dp = obs.pose_residual_under_mode(k, m) - float(base[k])
            if dp < best_dp:
                best_dp, best_m, best_bc = dp, m, obs.mode_byte_cost(k, m)
        if best_dp < 0.0:
            improvable += 1
            pooled_after = pooled_base + best_dp / len(pair_indices)
            ds = pose_score_term(max(pooled_after, 0.0)) - pose_score_term(pooled_base)
            ds += 25.0 * best_bc / 37_545_489
            vpb = (-ds / best_bc) if best_bc > 0 else float("inf")
            best_improvements.append(
                {"pair_local": k, "best_mode": best_m, "delta_pair_dpose": best_dp,
                 "byte_cost": best_bc, "value_per_byte": vpb,
                 "pays_rent_at_lambda_star": vpb > WATER_LEVEL_LAMBDA_STAR}
            )
    diagnostic = {
        "n_improvable_pairs": improvable,
        "n_pairs": len(pair_indices),
        "frac_improvable": improvable / max(len(pair_indices), 1),
        "improvable_paying_rent": sum(1 for b in best_improvements if b["pays_rent_at_lambda_star"]),
        "best_improvements": best_improvements[:40],
        "interpretation": (
            "n_improvable_pairs=0 => frontier FEC6 selector is already per-pair pose-optimal over "
            "the K=16 palette (no remaining cross-pair re-allocation pays rent). improvable but "
            "0 paying rent => improvements exist but sit below the water level lambda*."
        ),
    }

    elapsed = time.time() - t0
    out = {
        "schema": "cross_pair_pose_waterfill_smoke.v1",
        "task": 54,
        "base": "frontier_archive",
        "frontier_bytes": bytes_before,
        "n_pairs_sampled": len(pair_indices),
        "all_600": bool(args.all_600),
        "candidate_modes": list(candidate_modes),
        "pose_waterfill": result.to_row(),
        "constant_control": {
            "mode": best_const_mode,
            "mean_residual": best_const_mean,
            "row": const_result.to_row(),
        },
        "waterfill_beats_constant": result.net_delta_score <= const_result.net_delta_score + 1e-15,
        "diagnostic_why_unmoved": diagnostic,
        "elapsed_seconds": elapsed,
        "authority": "[local CPU-torch advisory]",
        "promotable": False,
        "score_claim": False,
        "note": (
            "SAMPLED-PAIR d_pose is the MEAN over the sampled subset, not the full-600 contest "
            "d_pose; the net_delta_score is the score delta on the sampled pose pool (advisory). "
            "The full-600 exact pose pool requires --all-600. new_bad MUST be 0."
        ),
    }
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))

    r = result
    print(f"[cross-pair-pose-waterfill] n_pairs={len(pair_indices)} elapsed={elapsed:.1f}s")
    print(f"  pooled d_pose: {r.pooled_d_pose_before:.6e} -> {r.pooled_d_pose_after:.6e}")
    print(f"  admitted={r.admitted}  new_bad={r.new_bad}  delta_bytes={r.bytes_after - r.bytes_before}")
    print(f"  NET delta_score (sampled pose pool) = {r.net_delta_score:+.6e}  beats_base={r.beats_base}")
    print(f"  constant control ({best_const_mode}): net={const_result.net_delta_score:+.6e}")
    print(f"  waterfill beats constant: {out['waterfill_beats_constant']}")
    print(
        f"  WHY: improvable_pairs={diagnostic['n_improvable_pairs']}/{diagnostic['n_pairs']} "
        f"  paying_rent={diagnostic['improvable_paying_rent']}"
    )
    print(f"  -> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
