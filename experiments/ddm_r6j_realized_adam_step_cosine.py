#!/usr/bin/env python3
"""R6 judge repair, leg 3 -- the realized-AdamW step cosine from retained moments.

rg1b §6.6 item 3, verbatim: "The realized-AdamW cosine from the retained optimizer
moments -- one desk computation, closes the last gap between 'gradient rotated' and
'step rotated'."

WHY THIS IS THE DECISIVE LEG, not a nicety.  rg1b measured that the band objective
ROTATES THE GRADIENT (`cos` < 0.90 on the Adam-relevant metric in 9/9 cells) and yet
the band arm's flips land ON the 5-arm displacement law at -0.871 sigma -- a smaller
residual than the worst stock arm.  Two readings survive that pair of facts:

  (J) JUDGE-DEAD  -- the objective steers, but a 600-step peak-flip probe cannot see
      direction, so the probe design is invalid.  This is rg1b's adjudication.
  (M) MECHANISM-DEAD -- AdamW's per-coordinate normalisation `m_hat/(sqrt(v_hat)+eps)`
      DIVIDES OUT the reweighting.  A per-pixel loss reweight that scales gradient
      MAGNITUDES without changing their sign pattern is largely undone by dividing by
      sqrt(v_hat), because v_hat carries the same scaling.  Under (M) the objective
      cannot steer THIS optimiser at all, the flip law is a consequence rather than a
      coincidence, and fixing the judge buys nothing.

Prior evidence that (M) operates on this stack: task #903, where the upsample-VJP
scatter interacted with Adam's sign behaviour so that the loss SCALAR was identical
while 40 of 41 arrays diverged.  Adam's normalisation demonstrably dominates direction
here.  (M) has never been tested and it is free to test.

THE MEASUREMENT.  Both arms retain full AdamW state (`training_state.optimizer.state`
carries `exp_avg` = m and `exp_avg_sq` = v) at matched steps.  The realized update
direction is reconstructible from the checkpoint ALONE -- no re-forward, no scorer:

    u = m_hat / (sqrt(v_hat) + eps),   m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)

so we can compare, at each matched step:

    cos_m(A2, band) -- do the MOMENTUM directions differ?   (rg1b says yes, <0.90)
    cos_u(A2, band) -- do the REALIZED STEP directions differ?

FALSIFIER, pre-registered.  Let g = cos_m and s = cos_u at the same matched step.
  * MECHANISM-DEAD (M) is SUPPORTED iff s > g materially and s is near 1 -- concretely
    `s >= 0.95 AND s - g >= 0.05` at a majority of matched steps.  Adam collapsed two
    different objectives onto one step direction; R6's objective cannot steer AdamW,
    and the repair is an OPTIMISER change, not a judge change.
  * JUDGE-DEAD (J) SURVIVES iff s stays materially below 0.95 (the step really is
    rotated) -- then rg1b's reading stands, the objective steers, and the judge/window
    is the only blocker.
  * MIXED/INDETERMINATE otherwise; report it as such and do not force a side.

SCOPE CEILING, stated up front.  A2 and band_a1 share init/cache/seed/steps/lr but by
step t they sit at DIFFERENT points in weight space (measured ||dw||_100 0.047400 vs
0.055976, an 18% gap).  So a cross-arm cosine conflates "different objective" with
"different location".  That confound INFLATES apparent rotation -- it can only push s
DOWN, never up.  Therefore a HIGH s is sound evidence for (M) despite the confound,
while a LOW s is NOT by itself sound evidence for (J).  The test is one-sided and this
file says so rather than claiming a clean two-sided discriminator.

The per-arm control `cos(m, u)` -- how far Adam's own normalisation rotates that arm's
momentum -- has NO location confound and is reported alongside.

axis: [macOS-CPU advisory] -- read-back of retained payloads.  NEVER a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

AXIS = "[macOS-CPU advisory]"
A2_DIR = Path("/Volumes/APDataStore/pact/ddm_lr1/A2")
BAND_DIR = Path("/Volumes/APDataStore/pact/ddm_rg1/band_a1")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_r6j_realized_adam_cosine_20260816")

# rg1b's matched-step grid; both arms wrote a full_state checkpoint at each.
STEPS = (100, 200, 300, 400, 500, 600)


class R6jError(RuntimeError):
    """Fail closed; never silently substitute a different object."""


def _ckpt_path(arm_dir: Path, step: int) -> Path:
    """Locate the full_state checkpoint for `step`, whatever stage tag it carries.

    A2 writes `checkpoints.*`, band_a1 writes `ckpt.*` -- the stage suffix varies with
    the curriculum phase (periodic / stage-ce / stage-softplus_margin / ...), so match
    on the step token rather than assuming a name.  Never invent a path.
    """
    token = f"step{step:06d}"
    hits = sorted(
        p for p in arm_dir.glob("*full_state.pt")
        if token in p.name and not p.name.startswith("._")
    )
    if not hits:
        raise R6jError(f"no full_state checkpoint for step {step} under {arm_dir}")
    if len(hits) > 1:
        raise R6jError(f"ambiguous checkpoints for step {step}: {[p.name for p in hits]}")
    return hits[0]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_moments(path: Path) -> tuple[dict, dict, dict]:
    """Return (m_by_name, v_by_name, meta) from a retained full_state checkpoint.

    The optimizer state is keyed by PARAMETER INDEX; the name mapping comes from the
    param_groups order against the model's own state_dict key order.  We rebuild that
    mapping explicitly rather than assuming index==enumeration of state_dict, and we
    fail closed if the counts disagree.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    ts = d.get("training_state")
    if not isinstance(ts, dict):
        raise R6jError(f"{path}: no training_state")
    opt = ts.get("optimizer")
    if not isinstance(opt, dict) or "state" not in opt or "param_groups" not in opt:
        raise R6jError(f"{path}: optimizer state absent -- the moments were not retained")
    msd = ts.get("model_state_dict") or d.get("state_dict")
    if not isinstance(msd, dict):
        raise R6jError(f"{path}: no model_state_dict")

    # Trainable-parameter order: param_groups list indices in order of appearance.
    idx_order: list[int] = []
    for g in opt["param_groups"]:
        idx_order.extend(int(i) for i in g["params"])
    names = [k for k in msd.keys()]
    if len(idx_order) != len(names):
        raise R6jError(
            f"{path}: optimizer holds {len(idx_order)} params but state_dict has "
            f"{len(names)} entries -- refusing to guess the mapping"
        )

    state = opt["state"]
    m: dict[str, np.ndarray] = {}
    v: dict[str, np.ndarray] = {}
    steps_seen: set[int] = set()
    for pos, pidx in enumerate(idx_order):
        st = state.get(pidx) if pidx in state else state.get(str(pidx))
        if st is None:
            raise R6jError(f"{path}: no optimizer state for param index {pidx}")
        if "exp_avg" not in st or "exp_avg_sq" not in st:
            raise R6jError(f"{path}: param {pidx} lacks exp_avg/exp_avg_sq (not AdamW?)")
        name = names[pos]
        m[name] = st["exp_avg"].detach().to(torch.float64).cpu().numpy().ravel()
        v[name] = st["exp_avg_sq"].detach().to(torch.float64).cpu().numpy().ravel()
        s = st.get("step")
        if s is not None:
            steps_seen.add(int(s.item()) if hasattr(s, "item") else int(s))

    g0 = opt["param_groups"][0]
    meta = {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "recorded_step": int(ts.get("step", -1)),
        "phase": ts.get("phase"),
        "betas": [float(b) for b in g0.get("betas", (0.9, 0.999))],
        "eps": float(g0.get("eps", 1e-8)),
        "lr": float(g0.get("lr", float("nan"))),
        "adam_step_counters": sorted(steps_seen)[:4],
        "n_tensors": len(m),
        "n_coords": int(sum(a.size for a in m.values())),
    }
    return m, v, meta


def _realized_update(m: dict, v: dict, betas: tuple[float, float], eps: float,
                     t: int) -> dict[str, np.ndarray]:
    """u = m_hat/(sqrt(v_hat)+eps) -- the AdamW direction actually applied."""
    b1, b2 = betas
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    if bc1 <= 0 or bc2 <= 0:
        raise R6jError(f"degenerate bias correction at t={t}")
    out = {}
    for k in m:
        mh = m[k] / bc1
        vh = v[k] / bc2
        out[k] = mh / (np.sqrt(vh) + eps)
    return out


def _cos(a: dict, b: dict) -> float:
    """Global cosine over the concatenated parameter vector (shared keys only)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise R6jError("no shared tensors between the two arms")
    num = sum(float(np.dot(a[k], b[k])) for k in keys)
    na = math.sqrt(sum(float(np.dot(a[k], a[k])) for k in keys))
    nb = math.sqrt(sum(float(np.dot(b[k], b[k])) for k in keys))
    if na == 0.0 or nb == 0.0:
        raise R6jError("zero-norm direction vector")
    return num / (na * nb)


def _per_tensor_cos(a: dict, b: dict) -> dict[str, float]:
    out = {}
    for k in sorted(set(a) & set(b)):
        na = math.sqrt(float(np.dot(a[k], a[k])))
        nb = math.sqrt(float(np.dot(b[k], b[k])))
        out[k] = float(np.dot(a[k], b[k]) / (na * nb)) if na and nb else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a2-dir", type=Path, default=A2_DIR,
                    help="stock control arm (ddm_lr1/A2)")
    ap.add_argument("--band-dir", type=Path, default=BAND_DIR,
                    help="band-objective arm (ddm_rg1/band_a1)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, nargs="*", default=list(STEPS))
    ap.add_argument("--s-bar", type=float, default=0.95,
                    help="pre-registered step-cosine bar for MECHANISM-DEAD")
    ap.add_argument("--gap-bar", type=float, default=0.05,
                    help="pre-registered minimum s-g gap for MECHANISM-DEAD")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for step in args.steps:
        pa = _ckpt_path(args.a2_dir, step)
        pb = _ckpt_path(args.band_dir, step)
        ma, va, meta_a = _load_moments(pa)
        mb, vb, meta_b = _load_moments(pb)
        if meta_a["betas"] != meta_b["betas"] or meta_a["eps"] != meta_b["eps"]:
            raise R6jError(
                f"step {step}: optimiser hyperparameters differ "
                f"({meta_a['betas']}/{meta_a['eps']} vs {meta_b['betas']}/{meta_b['eps']}) "
                "-- the arms are not matched and the cosine would be a wrong-object read"
            )
        betas = tuple(meta_a["betas"])
        eps = meta_a["eps"]
        ua = _realized_update(ma, va, betas, eps, step)
        ub = _realized_update(mb, vb, betas, eps, step)

        g = _cos(ma, mb)            # momentum (gradient-EMA) cosine
        s = _cos(ua, ub)            # REALIZED AdamW step cosine
        rows.append({
            "step": step,
            "cos_momentum_cross_arm": g,
            "cos_realized_step_cross_arm": s,
            "s_minus_g": s - g,
            "cos_m_to_u_A2": _cos(ma, ua),      # per-arm control: Adam's own rotation
            "cos_m_to_u_band": _cos(mb, ub),
            "a2": meta_a,
            "band": meta_b,
        })
        # ALWAYS KEEP THE PAYLOAD: the per-tensor vectors, not only the scalars.
        np.savez_compressed(
            args.out / f"per_tensor_cos_step{step:06d}.npz",
            names=np.array(sorted(set(ma) & set(mb))),
            cos_m=np.array([_per_tensor_cos(ma, mb)[k]
                            for k in sorted(set(ma) & set(mb))]),
            cos_u=np.array([_per_tensor_cos(ua, ub)[k]
                            for k in sorted(set(ua) & set(ub))]),
        )
        print(f"step {step:>4}  cos_m {g:+.6f}  cos_u {s:+.6f}  "
              f"delta {s - g:+.6f}", flush=True)

    n_mech = sum(1 for r in rows
                 if r["cos_realized_step_cross_arm"] >= args.s_bar
                 and r["s_minus_g"] >= args.gap_bar)
    n_judge = sum(1 for r in rows if r["cos_realized_step_cross_arm"] < args.s_bar)
    majority = len(rows) / 2.0
    if n_mech > majority:
        verdict = "MECHANISM_DEAD_ADAM_NORMALISATION_COLLAPSES_THE_OBJECTIVE"
    elif n_judge > majority:
        verdict = "JUDGE_DEAD_SURVIVES_STEP_IS_GENUINELY_ROTATED"
    else:
        verdict = "INDETERMINATE"

    receipt = {
        "schema": "ddm_r6j_realized_adam_cosine.v1",
        "axis": AXIS + " -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "question": "does AdamW's per-coordinate normalisation divide out the band "
                    "objective's gradient rotation before it reaches the weights?",
        "source": "rg1b §6.6 item 3 (the named, never-run leg)",
        "pre_registered_falsifier": {
            "mechanism_dead": f"cos_u >= {args.s_bar} AND (cos_u - cos_m) >= "
                              f"{args.gap_bar} at a majority of matched steps",
            "judge_dead_survives": f"cos_u < {args.s_bar} at a majority of steps",
            "one_sidedness": "A2 and band sit at different weight-space points by step "
                             "t (||dw||_100 0.047400 vs 0.055976), which can only push "
                             "cos_u DOWN; so HIGH cos_u is sound evidence for "
                             "mechanism-dead, LOW cos_u is NOT sound evidence for "
                             "judge-dead on its own",
        },
        "rows": rows,
        "n_steps": len(rows),
        "n_mechanism_dead_steps": n_mech,
        "n_judge_dead_steps": n_judge,
        "verdict": verdict,
        "verdict_scope": "INSTANCE -- two arms (stock A2, band alpha=1), 600 steps, "
                         "this init, MPS-trained checkpoints read back on CPU",
        "wall_s": time.time() - t0,
    }
    out_json = args.out / "R6J_REALIZED_ADAM_COSINE.json"
    out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "n_mechanism_dead_steps", "n_judge_dead_steps",
                       "n_steps")}, indent=2))
    print(f"receipt: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
