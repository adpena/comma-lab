"""ddm_hg1 -- the SIGNED ring-0 margin hinge: re-derivation, exact prize, and binding proof.

Parent: `.omx/research/ddm_rn1_render_boundary_mechanism_20260816.md` (sister arm ddm_rn1).

rn1 closed the whole DECODE-SIDE operator family with one measured coefficient --
`rho(delta) = correct ring-0 px at risk per fixable flip`, with `rho(0.01) = 0.985`, a fair
coin -- and named the reason a TRAINING-SIDE version escapes the same limit: the decoder
cannot know the SIGN of its own error (SegNet weights are 38,502,892 B = S 25.64 if shipped,
2,672x the remaining gap, and forbidden regardless), but TRAINING has GT and therefore has the
sign.  rn1 then priced the training prize at `+0.1 logits -> 49.7% of the seg axis = 0.014711 S
= 1.53x the whole remaining gap`, labelled DERIVED / UPPER BOUND.

THIS TOOL RE-DERIVES THAT NUMBER INDEPENDENTLY, AND ATTACKS IT.

The attack has a specific target.  rn1's ladder is built on `gap = top1 - top2` -- the distance
from our (wrong) argmax to the runner-up.  That equals the deficit a signed hinge must close
ONLY IF the GT class IS the runner-up.  rt1 measured that at 98.3%, so rn1's ladder is
optimistic by an unmeasured amount.  The EXACT quantity a hinge acts on is the SIGNED margin

    m(px) = logit[GT] - max_{c != GT} logit[c]

which is negative exactly at a flip (deficit = -m) and positive at a correct pixel (headroom
= m).  A signed hinge that raises m by +delta everywhere on the ring recovers every flip with
deficit < delta and breaks nothing, because it moves correct pixels AWAY from the boundary.
This tool computes `m` directly, reports rn1's `gap` alongside it on the same pixels, and
measures the discrepancy -- so the prize is re-priced on the exact quantity rather than on a
proxy for it.

Stages
------
  rederive   the signed-margin ladder, rho, per-class shares, and the gap-vs-margin
             discrepancy.  Retains the per-pair signed margin field (payload, not just its
             length) so the hinge's binding proof consumes bytes it can verify by sha256.
  binding    replays the retained margin field against a hinge weight schedule and reports
             the fraction of ring-0 pixels the hinge is ACTIVE on and the gradient share the
             term would carry -- the inert-lever falsifier, run before any launch.

Axis: [macOS-CPU advisory] frozen CPU-torch SegNet, batch = 1 pair, upstream preprocess
verbatim -- the same instrument pins as rt1/rn1, so the rows are leg-to-leg comparable.
NEVER a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"

# --- provenance pins, inherited from the rn1/rt1 charter (do NOT re-derive) -----------------
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_S = 0.15959729295498598        # hv1 ep0634 [contest-CUDA T4 n600]
GAP_TO_015 = BASE_S - 0.15          # +0.0095973 that must be removed
BASE_FLIPS_ADVISORY = 34938         # rt1 RT1_INSTRUMENT_CHECK, n600 advisory

FRAMES = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W           # 117,964,800
N_CLASSES = 5
S_PER_FLIP = 100.0 / SCORED_PX               # 8.4771e-07, td1's derived law

# rn1's own ladder, quoted here ONLY so the re-derivation can be diffed against it in-tool.
RN1_RHO_RING0 = {"0.01": 0.9851485148514851, "0.03": 1.233644859813084,
                 "0.1": 2.137841832963784, "0.3": 7.058344058344058}
RN1_HINGE_SHARE = {"0.01": 0.07415, "0.03": 0.19640, "0.1": 0.49670, "0.3": 0.85573}

DELTAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_hg1_ring0_margin_hinge_20260816")


class Hg1Error(RuntimeError):
    """Fail-closed error for instrument / custody violations."""


# ============================================================================================
# instrument -- pinned to rt1/rn1 so rows are leg-to-leg comparable
# ============================================================================================
class Instrument:
    """Frozen CPU-torch SegNet, batch-1, upstream preprocess verbatim."""

    def __init__(self, threads: int) -> None:
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        import einops
        from modules import SegNet, segnet_sd_path
        from safetensors.torch import load_file

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self._torch, self._einops = torch, einops
        self.threads = threads
        seg = SegNet().eval()
        seg.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
        self.seg = seg

    def seg_logits(self, frame1_cam_u8: np.ndarray) -> np.ndarray:
        """(5,384,512) float32 logits from one camera-resolution uint8 frame."""
        torch = self._torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(frame1_cam_u8))[None, None]
            x = self._einops.rearrange(t, "b t h w c -> b t c h w").float()
            return self.seg(self.seg.preprocess_input(x))[0].numpy()


# ============================================================================================
# io helpers
# ============================================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while (b := fh.read(chunk)):
            h.update(b)
    return h.hexdigest()


def open_raw(raw: Path) -> np.memmap:
    n = raw.stat().st_size // (CAM_H * CAM_W * 3)
    if n != 2 * FRAMES:
        raise Hg1Error(f"{raw} holds {n} frames, expected {2 * FRAMES}")
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(n, CAM_H, CAM_W, 3))


def open_tokens(tokens: Path) -> np.memmap:
    if tokens.stat().st_size != SCORED_PX:
        raise Hg1Error(f"{tokens} is {tokens.stat().st_size} B, expected {SCORED_PX}")
    return np.memmap(tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))


def write_receipt(work: Path, name: str, payload: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p


def retain_payload(work: Path, name: str, arr: np.ndarray) -> dict:
    """ALWAYS KEEP THE PAYLOAD: persist the bytes, then record sha256 AND length."""
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{name}.npy"
    np.save(p, arr)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p),
            "shape": list(arr.shape), "dtype": str(arr.dtype)}


def seeded_pairs(n: int, seed: int) -> list[int]:
    """SEEDED RANDOM pairs -- never a prefix (m88/m96)."""
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(FRAMES, size=n, replace=False))


def boundary_ring0(lab: np.ndarray) -> np.ndarray:
    """Scorer-lattice pixels adjacent to a label change (rt1's ring 0, the free support)."""
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


def signed_margin(lg: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """m = logit[GT] - max_{c != GT} logit[c].

    NEGATIVE exactly at a flip (its magnitude is the deficit a hinge must close); POSITIVE at a
    correct pixel (its value is the headroom before that pixel breaks).  This is the EXACT
    quantity a signed ring-0 hinge acts on -- rn1's `top1 - top2` equals it only where the GT
    class is the runner-up.
    """
    own = np.take_along_axis(lg, gt[None].astype(np.intp), axis=0)[0]
    masked = lg.copy()
    np.put_along_axis(masked, gt[None].astype(np.intp), -np.inf, axis=0)
    return (own - masked.max(axis=0)).astype(np.float32)


# ============================================================================================
# stage: rederive -- the signed-margin ladder, rho, and the gap-vs-margin discrepancy
# ============================================================================================
def stage_rederive(args: argparse.Namespace) -> int:
    raw = open_raw(args.raw)
    tok = open_tokens(args.tokens)
    gt_all = np.load(args.gt, mmap_mode="r")
    pairs = seeded_pairs(args.pairs, args.seed)
    inst = Instrument(args.threads)

    nd = len(DELTAS)
    n_flip = n_ring_ours = n_ring_gt = n_corr_ring_ours = n_corr_ring_gt = 0
    cum_flip_m = np.zeros(nd, dtype=np.int64)          # deficit from SIGNED margin
    cum_flip_gap = np.zeros(nd, dtype=np.int64)        # deficit from rn1's top1-top2
    cum_corr_ours = np.zeros(nd, dtype=np.int64)       # ring-0 from OUR tokens (rn1's def)
    cum_corr_gt = np.zeros(nd, dtype=np.int64)         # ring-0 from GT (training's def)
    cls_flip = np.zeros((N_CLASSES, nd), dtype=np.int64)
    cls_tot = np.zeros(N_CLASSES, dtype=np.int64)
    n_runner_up = 0                                    # flips where GT IS the runner-up
    med_flip_m: list[float] = []
    med_corr_m: list[float] = []
    margins = np.zeros((len(pairs), SEG_H, SEG_W), dtype=np.float32)

    for k, p in enumerate(pairs):
        lg = inst.seg_logits(np.asarray(raw[2 * p + 1]))
        gtp = np.asarray(gt_all[p]).astype(np.uint8)
        am = lg.argmax(0).astype(np.uint8)
        flip = am != gtp

        m = signed_margin(lg, gtp)
        margins[k] = m
        srt = np.sort(lg, axis=0)
        gap = (srt[-1] - srt[-2]).astype(np.float32)   # rn1's quantity, same pixels

        # GT is the runner-up exactly where the deficit equals the top1-top2 gap.
        rn = flip & np.isclose(-m, gap, rtol=0, atol=1e-5)
        n_runner_up += int(rn.sum())

        ring_ours = boundary_ring0(np.asarray(tok[p]))
        ring_gt = boundary_ring0(gtp)
        corr_ours = (~flip) & ring_ours
        corr_gt = (~flip) & ring_gt

        n_flip += int(flip.sum())
        n_ring_ours += int(ring_ours.sum())
        n_ring_gt += int(ring_gt.sum())
        n_corr_ring_ours += int(corr_ours.sum())
        n_corr_ring_gt += int(corr_gt.sum())

        deficit = -m[flip]
        for i, d in enumerate(DELTAS):
            cum_flip_m[i] += int((deficit < d).sum())
            cum_flip_gap[i] += int((gap[flip] < d).sum())
            cum_corr_ours[i] += int((m[corr_ours] < d).sum())
            cum_corr_gt[i] += int((m[corr_gt] < d).sum())
        for c in range(N_CLASSES):
            msk = flip & (gtp == c)
            cls_tot[c] += int(msk.sum())
            dc = -m[msk]
            for i, d in enumerate(DELTAS):
                cls_flip[c, i] += int((dc < d).sum())
        if flip.any():
            med_flip_m.append(float(np.median(deficit)))
        med_corr_m.append(float(np.median(m[corr_ours])))
        print(f"  pair {p:3d}  flips={int(flip.sum()):5d}  ring0(ours)={int(ring_ours.sum()):6d}"
              f"  ring0(gt)={int(ring_gt.sum()):6d}  median deficit={med_flip_m[-1]:.4f}"
              f"  median headroom={med_corr_m[-1]:.4f}", flush=True)

    def ladder(cum: np.ndarray) -> dict:
        return {f"{d}": int(cum[i]) for i, d in enumerate(DELTAS)}

    def rho(corr: np.ndarray, flips: np.ndarray) -> dict:
        return {f"{d}": (float(corr[i]) / flips[i] if flips[i] else float("inf"))
                for i, d in enumerate(DELTAS)}

    share_m = {f"{d}": (float(cum_flip_m[i]) / n_flip if n_flip else None)
               for i, d in enumerate(DELTAS)}
    share_gap = {f"{d}": (float(cum_flip_gap[i]) / n_flip if n_flip else None)
                 for i, d in enumerate(DELTAS)}

    payload = retain_payload(args.work, f"HG1_MARGIN_FIELD_{args.tag}", margins)

    rec = {
        "schema": "ddm_hg1_rederive.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False, "verdict_scope": "INSTANCE",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "pair_selection": "seeded random (never a prefix; m88/m96)",
        "definition_signed_margin": (
            "m = logit[GT] - max_{c != GT} logit[c].  NEGATIVE exactly at a flip (deficit = -m); "
            "POSITIVE at a correct pixel (headroom = m).  This is the EXACT quantity a signed "
            "ring-0 hinge acts on.  rn1's `top1 - top2` equals it only where GT is the runner-up."),
        "counts": {
            "flips": n_flip,
            "ring0_px_ours": n_ring_ours, "ring0_px_gt": n_ring_gt,
            "correct_on_ring0_ours": n_corr_ring_ours,
            "correct_on_ring0_gt": n_corr_ring_gt,
            "flips_where_gt_is_runner_up": n_runner_up,
            "share_gt_is_runner_up": (float(n_runner_up) / n_flip if n_flip else None),
        },
        "median_deficit_at_flips": float(np.median(med_flip_m)) if med_flip_m else None,
        "median_headroom_at_correct_ring0": float(np.median(med_corr_m)),
        "cum_flips_below_delta_SIGNED_MARGIN": ladder(cum_flip_m),
        "cum_flips_below_delta_rn1_top1_minus_top2": ladder(cum_flip_gap),
        "cum_correct_ring0_below_delta_ours": ladder(cum_corr_ours),
        "cum_correct_ring0_below_delta_gt": ladder(cum_corr_gt),
        "rho_ring0_ours": rho(cum_corr_ours, cum_flip_m),
        "rho_ring0_gt": rho(cum_corr_gt, cum_flip_m),
        "hinge_share_SIGNED_MARGIN": share_m,
        "hinge_share_rn1_top1_minus_top2": share_gap,
        "hinge_prize_SIGNED_MARGIN": {
            f"{d}": {
                "share_of_flips_recoverable": share_m[f"{d}"],
                "S_units_n600_equiv": (share_m[f"{d}"] * BASE_FLIPS_ADVISORY * S_PER_FLIP
                                       if share_m[f"{d}"] is not None else None),
                "multiple_of_gap_to_015": (share_m[f"{d}"] * BASE_FLIPS_ADVISORY * S_PER_FLIP
                                           / GAP_TO_015 if share_m[f"{d}"] is not None else None),
            } for d in DELTAS},
        "per_gt_class_flips": {str(c): int(cls_tot[c]) for c in range(N_CLASSES)},
        "per_gt_class_share_below_delta": {
            str(c): {f"{d}": (float(cls_flip[c, i]) / cls_tot[c] if cls_tot[c] else None)
                     for i, d in enumerate(DELTAS)}
            for c in range(N_CLASSES)},
        "class_order": "canonical comma10k: 0 Road, 1 Lane, 2 Undrivable, 3 Movable, 4 MyCar",
        "rn1_diff": {
            "rn1_rho_ring0": RN1_RHO_RING0,
            "rn1_hinge_share": RN1_HINGE_SHARE,
            "note": ("rn1 reproduces where this tool's rho column (built on the SIGNED margin "
                     "and OUR-token ring-0) matches its published table; any gap between "
                     "hinge_share_SIGNED_MARGIN and hinge_share_rn1_top1_minus_top2 is the "
                     "runner-up assumption's cost and re-prices the prize downward."),
        },
        "retained_payload": payload,
        "reading": ("The signed-margin ladder is the exact prize a ring-0 hinge can claim; the "
                    "top1-top2 ladder is an upper bound on it.  rho remains the undirected "
                    "exchange rate and does NOT bound the signed hinge, because a signed shift "
                    "moves correct pixels AWAY from the boundary."),
    }
    write_receipt(args.work, f"HG1_REDERIVE_{args.tag}", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "pairs"}, indent=2, sort_keys=True))
    return 0


# ============================================================================================
# stage: binding -- the inert-lever falsifier, run on the retained margin field
# ============================================================================================
def stage_binding(args: argparse.Namespace) -> int:
    """How much of the ring-0 support does a hinge at `--target` actually ACTIVATE?

    A loss term that is active on ~0 pixels is INERT and its run is confounded (CLAUDE.md
    confound self-protection L1: `adaptive_eps_INERT`).  This stage reads the retained margin
    field -- by sha256, never recomputed from a remembered number -- and reports the active
    fraction and the mean hinge value per delta, on BOTH ring-0 definitions.
    """
    field = Path(args.field)
    if not field.exists():
        raise Hg1Error(f"retained margin field not found: {field}")
    sha = sha256_file(field)
    margins = np.load(field)
    tok = open_tokens(args.tokens)
    gt_all = np.load(args.gt, mmap_mode="r")
    pairs = seeded_pairs(args.pairs, args.seed)
    if margins.shape[0] != len(pairs):
        raise Hg1Error(f"field holds {margins.shape[0]} pairs, seed/pairs give {len(pairs)}")

    rows = []
    for target in args.target:
        act_ours = act_gt = tot_ours = tot_gt = 0
        sum_ours = sum_gt = 0.0
        for k, p in enumerate(pairs):
            m = margins[k]
            ring_ours = boundary_ring0(np.asarray(tok[p]))
            ring_gt = boundary_ring0(np.asarray(gt_all[p]).astype(np.uint8))
            for ring, key in ((ring_ours, "ours"), (ring_gt, "gt")):
                h = np.maximum(0.0, target - m[ring])       # relu(target - m), the hinge
                if key == "ours":
                    act_ours += int((h > 0).sum())
                    tot_ours += int(ring.sum())
                    sum_ours += float(h.sum())
                else:
                    act_gt += int((h > 0).sum())
                    tot_gt += int(ring.sum())
                    sum_gt += float(h.sum())
        rows.append({
            "target_logits": target,
            "ring0_ours": {"active_px": act_ours, "support_px": tot_ours,
                           "active_fraction": act_ours / tot_ours if tot_ours else None,
                           "mean_hinge_over_support": sum_ours / tot_ours if tot_ours else None},
            "ring0_gt": {"active_px": act_gt, "support_px": tot_gt,
                         "active_fraction": act_gt / tot_gt if tot_gt else None,
                         "mean_hinge_over_support": sum_gt / tot_gt if tot_gt else None},
        })
        print(f"  target={target:<5}  ours active={rows[-1]['ring0_ours']['active_fraction']:.6f}"
              f"  gt active={rows[-1]['ring0_gt']['active_fraction']:.6f}", flush=True)

    rec = {
        "schema": "ddm_hg1_binding.v1", "axis": "[macOS-CPU advisory]",
        "score_claim": False, "promotable": False, "verdict_scope": "INSTANCE",
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "consumed_field": {"path": str(field), "sha256": sha, "bytes": field.stat().st_size},
        "n_pairs": len(pairs), "seed": args.seed, "pairs": pairs,
        "hinge": "loss = mean over ring-0 of relu(target - m), m = logit[GT] - max_{c!=GT} logit",
        "rows": rows,
        "falsifier": ("A hinge whose active fraction is ~0 is INERT: its run cannot be "
                      "attributed and its verdict is confounded.  A hinge whose active fraction "
                      "is ~1 is not a hinge, it is a global margin push and will fight rate."),
    }
    write_receipt(args.work, f"HG1_BINDING_{args.tag}", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "pairs"}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["rederive", "binding"])
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--field", type=Path, default=None,
                    help="binding stage: retained HG1_MARGIN_FIELD_*.npy")
    ap.add_argument("--target", type=float, action="append", default=[],
                    help="binding stage: hinge target in logits. Repeatable.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--tag", default="n96")
    args = ap.parse_args(argv)
    if args.stage == "binding":
        if args.field is None:
            ap.error("--field is required for the binding stage")
        if not args.target:
            args.target = [0.01, 0.03, 0.1, 0.3, 1.0]
    return {"rederive": stage_rederive, "binding": stage_binding}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
