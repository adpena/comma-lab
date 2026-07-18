# SPDX-License-Identifier: MIT
"""Realization-vs-gradient regime classifier (organ upgrade B, 2026-07-17).

Answers the live decision the organ could not: WHY the run plateaus — is the remaining
d_seg flip mass still reachable by more training (``gradient_limited``), or does its
correction require render-space moves BELOW what the uint8 realization path can express
(``realization_limited`` — more epochs cannot help; the terminal SOLVE / Dykstra
projection #341/#342 is the correct actuation)?

Mechanism (all REAL inputs; convention IDENTICAL to the canonical measured law
``realization_necessity_preimage_per_stratum_v1`` / the necessity solver's VJP stage):

  1. snapshot the LIVE EMA checkpoint's remaining flips + exact pairwise margins
     (``factorized_features.snapshot_witness_margins`` — real decode through R, real
     frozen SegNet, bit-exact GT lstars);
  2. per sampled flip pixel, full-chain VJP: camera-res witness frame -> the EXACT shared
     bilinear resize (parity-asserted, bitwise, against ``segnet.preprocess_input`` —
     fail-closed on any mismatch) -> frozen SegNet -> pairwise margin
     ``m = z_wrong - z_gt``; gradient ``g = dm/dx_camera``;
  3. the min-norm camera displacement that crosses the margin is ``delta* = -m g/||g||^2``;
     its largest per-coordinate amplitude is ``a_max = m * max|g| / ||g||^2`` (0-255
     units).  ``sub-LSB iff a_max < 0.5`` — the min-norm correcting move rounds away under
     uint8 (the solver's exact convention, mirrored at
     ``tools/necessity_inverse_factorization_solver.py:498-511``).
  4. aggregate the flip-MASS-weighted sub-LSB fraction (each sampled pixel weighted by its
     oriented-pair stratum's flip count / sampled count) -> regime + a crisp
     ``terminal_solve_admissible`` boolean.

Classification convention (DERIVED choice, stated — the continuous fraction is the
primary output): ``realization_limited`` iff the majority (>= 0.5) of remaining flip mass
is sub-LSB (training through uint8-R cannot realize most of what remains);
``gradient_limited`` iff <= 0.25; ``mixed`` between.

HONESTY: sub-LSB of the MIN-NORM move is a necessary-side indicator — wider-support
dithered moves (>=1 LSB per pixel, signed, spread over the tap support) may still realize
a sub-min-norm-LSB margin change; exactly as in the canonical equation, the fraction
bounds what amplitude-style training can express, it does not prove impossibility.  The
snapshot is a labeled stride subset.  Advisory ``[macOS-CPU advisory] NON-PROMOTABLE``;
``score_claim=False``; read-only against the SACRED run dir (state rows land in
``.omx/state/witness_realization_regime.jsonl``, never in the run dir).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tac.witness_control.factorized_features import (
    AXIS_TAG,
    SCORER_HW,
    MarginSnapshot,
    load_frozen_segnet_cpu,
    locked_append_jsonl,
    oriented_key,
    snapshot_witness_margins,
    utc_stamp,
)

_REPO = Path(__file__).resolve().parents[3]
REGIME_JSONL = _REPO / ".omx" / "state" / "witness_realization_regime.jsonl"

#: classification convention (DERIVED, documented above; the fraction is primary)
REALIZATION_LIMITED_MIN_FRAC = 0.5
GRADIENT_LIMITED_MAX_FRAC = 0.25
REALIZATION_REGIMES = ("realization_limited", "mixed", "gradient_limited")
SUB_LSB_MAX_COORD = 0.5  # uint8 LSB convention, identical to the necessity solver


def classify_fraction(sub_lsb_frac: float) -> str:
    if sub_lsb_frac >= REALIZATION_LIMITED_MIN_FRAC:
        return "realization_limited"
    if sub_lsb_frac <= GRADIENT_LIMITED_MAX_FRAC:
        return "gradient_limited"
    return "mixed"


def min_norm_crossing_max_coord(margin: float, grad: np.ndarray) -> tuple[float, float]:
    """EXACT closed form for the min-norm margin-crossing displacement's largest
    per-coordinate amplitude: ``a_max = m * max|g| / ||g||^2`` (and the L2 flip distance
    ``m/||g||``).  Raises on a zero gradient (no first-order path — fail closed)."""
    g = np.asarray(grad, dtype=np.float64)
    gn = float(np.sqrt((g * g).sum()))
    if gn <= 0.0:
        raise ValueError("zero camera gradient at flip pixel — no first-order crossing path")
    m = float(margin)
    return m * float(np.abs(g).max()) / (gn * gn), m / gn


def _assert_preprocess_parity(segnet_cpu, x_cam_f32) -> None:
    """Fail-closed parity: our differentiable resize MUST be bitwise-identical to the real
    ``segnet.preprocess_input``.  (The upstream preprocess is not modified — we re-express
    the same op differentiably and PROVE equality on the live tensor.)"""
    import torch
    import torch.nn.functional as tfun

    ours = tfun.interpolate(x_cam_f32.detach(), size=SCORER_HW, mode="bilinear")
    ref = segnet_cpu.preprocess_input(x_cam_f32.detach()[:, None])  # (B,T=1,C,H,W) -> last frame
    if not torch.equal(ours, ref):
        raise AssertionError(
            "differentiable resize != segnet.preprocess_input (bitwise) — convention drift; refusing"
        )


def stratified_flip_sample(snapshot: MarginSnapshot, n_pixels: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Sample flip-pixel indices stratified by oriented pair, proportional to flip mass
    (>=1 per non-empty stratum), capped at ``n_pixels`` total.  Returns
    {oriented_key: indices into the snapshot's flip arrays}."""
    rng = np.random.default_rng(seed)
    groups: dict[str, np.ndarray] = {}
    for w in range(5):
        for g in range(5):
            if w == g:
                continue
            sel = np.nonzero((snapshot.flip_wrong == w) & (snapshot.flip_gt == g))[0]
            if sel.size:
                groups[oriented_key(w, g)] = sel
    total = sum(int(v.size) for v in groups.values())
    if total == 0:
        return {}
    out: dict[str, np.ndarray] = {}
    budget = max(int(n_pixels), len(groups))
    for key, sel in sorted(groups.items()):
        take = max(1, round(budget * sel.size / total))
        take = min(take, sel.size)
        out[key] = rng.choice(sel, size=take, replace=False)
    return out


@dataclass
class RealizationRegimeResult:
    run_ref: str
    ema_epoch: int
    generated_at: str
    n_pairs_sampled: int
    n_flips_total: int
    n_pixels_vjp: int
    sub_lsb_frac_mass_weighted: float
    sub_lsb_frac_unweighted: float
    regime: str
    terminal_solve_admissible: bool
    d_seg_sample: float
    per_pair: dict = field(default_factory=dict)
    convention: str = (
        "min-norm crossing displacement max-coordinate < 0.5 uint8-LSB "
        "(realization_necessity_preimage_per_stratum_v1 convention)"
    )
    axis_tag: str = AXIS_TAG
    score_claim: bool = False

    def to_row(self) -> dict:
        return {
            "schema": "witness_realization_regime.v1",
            "run_ref": self.run_ref,
            "ema_epoch": self.ema_epoch,
            "generated_at": self.generated_at,
            "n_pairs_sampled": self.n_pairs_sampled,
            "n_flips_total": self.n_flips_total,
            "n_pixels_vjp": self.n_pixels_vjp,
            "sub_lsb_frac_mass_weighted": self.sub_lsb_frac_mass_weighted,
            "sub_lsb_frac_unweighted": self.sub_lsb_frac_unweighted,
            "regime": self.regime,
            "terminal_solve_admissible": self.terminal_solve_admissible,
            "d_seg_sample": self.d_seg_sample,
            "per_pair": self.per_pair,
            "convention": self.convention,
            "thresholds": {
                "realization_limited_min_frac": REALIZATION_LIMITED_MIN_FRAC,
                "gradient_limited_max_frac": GRADIENT_LIMITED_MAX_FRAC,
                "sub_lsb_max_coord": SUB_LSB_MAX_COORD,
            },
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
        }


def vjp_sub_lsb_over_snapshot(
    snapshot: MarginSnapshot,
    segnet_cpu,
    *,
    n_pixels: int = 120,
    seed: int = 0,
    torch_threads: int = 4,
) -> RealizationRegimeResult:
    """The measurement: full-chain per-pixel VJP over a stratified sample of the
    snapshot's REAL remaining flips.  Requires ``snapshot.frames1`` (the realized witness
    camera frames) — fail-closed otherwise (no surrogate frames)."""
    import torch
    import torch.nn.functional as tfun

    if not snapshot.frames1:
        raise ValueError("snapshot lacks frames1 — build it with keep_frames=True (no surrogates)")
    torch.set_num_threads(max(1, int(torch_threads)))

    strata = stratified_flip_sample(snapshot, n_pixels, seed=seed)
    if not strata:
        raise ValueError("no remaining flips in the snapshot — nothing to classify")

    # group sampled pixels by frame so each frame builds ONE autograd graph
    by_frame: dict[int, list[tuple[str, int]]] = {}
    for key, idxs in strata.items():
        for i in idxs:
            by_frame.setdefault(int(snapshot.flip_pair_idx[i]), []).append((key, int(i)))

    stratum_counts = {
        oriented_key(w, g): int(np.count_nonzero((snapshot.flip_wrong == w) & (snapshot.flip_gt == g)))
        for w in range(5) for g in range(5) if w != g
    }
    per_pair_rows: dict[str, dict] = {
        k: {"n_flips": stratum_counts.get(k, 0), "n_sampled": 0, "n_sub_lsb": 0,
            "a_max_values": []} for k in strata
    }

    n_done = 0
    for fi, items in sorted(by_frame.items()):
        x_cam = torch.from_numpy(
            np.ascontiguousarray(snapshot.frames1[fi])
        ).permute(2, 0, 1).float().unsqueeze(0)          # (1,3,874,1164), 0-255
        x_cam.requires_grad_(True)
        _assert_preprocess_parity(segnet_cpu, x_cam)
        x_s = tfun.interpolate(x_cam, size=SCORER_HW, mode="bilinear")
        logits = segnet_cpu(x_s)[0]                       # (5,384,512), grad-connected
        for key, i in items:
            y, x = int(snapshot.flip_y[i]), int(snapshot.flip_x[i])
            w, g = int(snapshot.flip_wrong[i]), int(snapshot.flip_gt[i])
            m_t = logits[w, y, x] - logits[g, y, x]
            m = float(m_t.item())
            # exactness guard: the graph margin must equal the snapshot margin (same
            # frames, same net) — a drift here means the snapshot is stale/fake.
            if abs(m - float(snapshot.flip_margin[i])) > 1e-3:
                raise AssertionError(
                    f"VJP margin {m:.6f} != snapshot margin {float(snapshot.flip_margin[i]):.6f} "
                    f"at pair-slot {fi} px ({y},{x}) — stale/mismatched snapshot; refusing"
                )
            if x_cam.grad is not None:
                x_cam.grad = None
            m_t.backward(retain_graph=True)
            gr = x_cam.grad[0].detach().numpy()
            a_max, _flipdist_l2 = min_norm_crossing_max_coord(m, gr)
            row = per_pair_rows[key]
            row["n_sampled"] += 1
            row["a_max_values"].append(float(a_max))
            if a_max < SUB_LSB_MAX_COORD:
                row["n_sub_lsb"] += 1
            n_done += 1
        del logits, x_s, x_cam

    # aggregate: mass-weighted by stratum flip counts; unweighted over sampled pixels
    tot_mass = sum(r["n_flips"] for r in per_pair_rows.values() if r["n_sampled"] > 0)
    mass_frac = 0.0
    n_sub_total = 0
    n_samp_total = 0
    for _key, r in per_pair_rows.items():
        if r["n_sampled"] == 0:
            continue
        frac = r["n_sub_lsb"] / r["n_sampled"]
        r["sub_lsb_frac"] = frac
        vals = np.asarray(r.pop("a_max_values"), dtype=np.float64)
        r["a_max_med"] = float(np.median(vals))
        r["a_max_p10"] = float(np.percentile(vals, 10))
        mass_frac += frac * (r["n_flips"] / tot_mass)
        n_sub_total += r["n_sub_lsb"]
        n_samp_total += r["n_sampled"]

    regime = classify_fraction(mass_frac)
    return RealizationRegimeResult(
        run_ref=snapshot.run_ref,
        ema_epoch=snapshot.ema_epoch,
        generated_at=utc_stamp(),
        n_pairs_sampled=len(snapshot.pair_indices),
        n_flips_total=snapshot.n_flips,
        n_pixels_vjp=n_done,
        sub_lsb_frac_mass_weighted=float(mass_frac),
        sub_lsb_frac_unweighted=(n_sub_total / n_samp_total) if n_samp_total else float("nan"),
        regime=regime,
        terminal_solve_admissible=(regime == "realization_limited"),
        d_seg_sample=snapshot.d_seg_sample,
        per_pair=per_pair_rows,
    )


def classify_realization_regime(
    ema_ckpt: str | Path,
    gt_cache: str | Path,
    *,
    pair_indices: list[int] | None = None,
    n_pixels: int = 120,
    seed: int = 0,
    run_ref: str | None = None,
    append_state_row: bool = True,
    state_path: str | Path | None = None,
) -> RealizationRegimeResult:
    """End-to-end: snapshot the live EMA checkpoint -> VJP sample -> regime.  All real
    inputs; fail-closed on any missing artifact.  Appends the result row to the
    ``.omx/state`` ledger (never the run dir) unless disabled."""
    segnet = load_frozen_segnet_cpu()
    snap = snapshot_witness_margins(
        ema_ckpt, gt_cache, pair_indices, segnet_cpu=segnet, keep_frames=True, run_ref=run_ref
    )
    res = vjp_sub_lsb_over_snapshot(snap, segnet, n_pixels=n_pixels, seed=seed)
    if append_state_row:
        locked_append_jsonl(state_path or REGIME_JSONL, res.to_row())
    return res


def format_regime_line(row: dict, age_s: float | None = None) -> str:
    """Digest line (pure formatter).  Crisp: the fraction, the regime, the actuation."""
    frac = row.get("sub_lsb_frac_mass_weighted")
    frac_s = f"{100 * frac:.0f}%" if isinstance(frac, (int, float)) else "?"
    reg = str(row.get("regime", "?")).upper()
    adm = row.get("terminal_solve_admissible")
    act = ("terminal SOLVE #341/#342 admissible — more epochs cannot realize the majority"
           if adm else "keep training (amplitude path still open)")
    head = (f"realization-regime: ema ep{row.get('ema_epoch', '?')} — {frac_s} of remaining "
            f"flip mass needs sub-LSB moves -> {reg} ({act})")
    if age_s is not None:
        head += f" [{age_s / 3600:.1f}h old]"
    return head + " [advisory NON-PROMOTABLE]"


def latest_regime_row(state_path: str | Path | None = None, run_prefix: str | None = None) -> dict | None:
    """Last persisted regime row (optionally filtered to a run-name prefix). Fail-open None."""
    p = Path(state_path or REGIME_JSONL)
    if not p.is_file():
        return None
    last = None
    for ln in p.read_text(errors="replace").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if run_prefix and not str(row.get("run_ref", "")).startswith(run_prefix):
            continue
        last = row
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ema-ckpt", required=True, help="live EMA npz (read-only)")
    ap.add_argument("--gt-cache", required=True, help="bit-exact GT cache npz (lstars)")
    ap.add_argument("--pairs", type=int, default=24, help="stride-sampled pair count")
    ap.add_argument("--pixels", type=int, default=120, help="VJP flip-pixel sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-ref", default=None,
                    help="run label for the state row (default: the ckpt's parent dir name)")
    ap.add_argument("--no-state-row", action="store_true")
    args = ap.parse_args(argv)

    from tac.witness_control.factorized_features import default_pair_sample, load_witness_ema

    manifest, _p, _c = load_witness_ema(args.ema_ckpt)
    pairs = default_pair_sample(int(manifest["n_pairs"]), args.pairs)
    res = classify_realization_regime(
        args.ema_ckpt, args.gt_cache, pair_indices=pairs, n_pixels=args.pixels,
        seed=args.seed, run_ref=args.run_ref, append_state_row=not args.no_state_row,
    )
    print(json.dumps(res.to_row(), indent=1, sort_keys=True))
    print(format_regime_line(res.to_row()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRADIENT_LIMITED_MAX_FRAC",
    "REALIZATION_LIMITED_MIN_FRAC",
    "REALIZATION_REGIMES",
    "REGIME_JSONL",
    "SUB_LSB_MAX_COORD",
    "RealizationRegimeResult",
    "classify_fraction",
    "classify_realization_regime",
    "format_regime_line",
    "latest_regime_row",
    "min_norm_crossing_max_coord",
    "stratified_flip_sample",
    "vjp_sub_lsb_over_snapshot",
]
