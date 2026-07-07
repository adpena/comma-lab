#!/usr/bin/env python
"""#341 QUADRATIC BASIN FINISHER probe (council draft §16.1) — staged $0 measurement.

[macOS-CPU/MLX advisory] NON-PROMOTABLE — NOT a score claim. Pointer contest-CPU 0.19110 UNMOVED.

Measures, on the mod32cap tau-saturated EMA-best checkpoint (ep650, d_seg 0.0033662 logged n600):
  STAGE 0 (linear HEAD re-solve): damped Newton-CG restricted to the output-head parameters that
    are EXACTLY affine given frozen trunk features h(x): ``out_sdf.{weight,bias}`` (phi = W h + b),
    ``out_tex.{weight,bias}`` (tex = W h + b), ``palette`` (pre-sigmoid base = softmax(phi/T) @ palette
    is linear in palette). The through-R scorer loss is nonlinear, so the head solve is IRLS/GN
    (damped Newton-CG on the affine head coordinates), not one-shot closed-form. FiLM gains are NOT
    affine (they multiply inside activations) and are excluded.
  STAGE 1 (the finisher): damped Newton-CG on the FULL parameter vector (Morse-lemma basin solve),
    HVPs via mx.vjp of mx.grad (forward-over-reverse jvp fails: MLX SliceUpdate JVP unimplemented;
    the Hessian is symmetric so vjp-of-grad == HVP).

The SOLVE loss is THE loss the run was crawling at the tau stage (launch.sh + run log):
  L_pair = 100 * mean(tau*softplus(-signed/tau)), tau=0.3, signed = realized through-R SegNet margin
           + 0.001 * ChanVese-length(phi0 on frame0)          (eikonal weight 0; w_pose 0 => omitted)
at the checkpoint's OWN schedule point (softmax_temp / hosc_beta read from the ckpt cfg, never args).

VERDICT authority: numpy-fp32 deploy forward (levelset_rgb_forward_numpy on int8_dequant params —
the run's realized_verdict ONE CODEPATH) -> _torch_R_to_camera_uint8 -> frozen CPU-torch SegNet,
chunked (vbatch<=32). d_pose is NOT measured (w_pose=0: pose is not this witness's job; the A/B
metric is d_seg). NO Stiefel projection (mod32cap does not use --film-stiefel — projecting anyway
was a confound in the earlier lane_share probe).

Self-orient dir feats are a CO-EVOLVED STATE not persisted in checkpoints; this tool reconstructs
them by the trainer's own fixed-point map (numpy deploy forward -> argmax -> directional feats),
seedable from GT lstar or zeros; the reconstruction gap vs the run's logged verdict is itself a
measured, reported quantity (stage ``verdict`` on the unmodified checkpoint).

EXECUTION LAWS honored: chunked resumable foreground (every stage persists per-unit state and
exits within --max-seconds); mx.eval + mx.clear_cache every CG step / pair; RAM floor check before
each sweep; durable outputs under experiments/results/basin_finisher_probe_20260707 (never /tmp).

Usage (each invocation does bounded work and exits; rerun until "done"):
  .venv/bin/python tools/quadratic_basin_finisher_probe.py feats   --tag gt1 --seed-from gt --iters 1 --pairs all
  .venv/bin/python tools/quadratic_basin_finisher_probe.py verdict --tag gt1 --params <npz> --out baseline
  .venv/bin/python tools/quadratic_basin_finisher_probe.py solve   --tag gt1 --mask head --k-pairs 8 --subset-seed 0
  .venv/bin/python tools/quadratic_basin_finisher_probe.py apply   --mask head
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO / "experiments/results/basin_finisher_probe_20260707"
DEFAULT_CKPT = PROBE_DIR / "ema_best_snapshot.npz"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
RAM_FLOOR_GIB = 20.0
TAU = 0.3
W_SEG = 100.0
LEN_W = 0.001
HEAD_KEYS = ("out_sdf.weight", "out_sdf.bias", "out_tex.weight", "out_tex.bias", "palette")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without the checkpoint/scorers).
# ---------------------------------------------------------------------------
def flatten_masked(arrs: dict[str, np.ndarray], mask_keys: list[str]) -> np.ndarray:
    """Concatenate the masked arrays (sorted key order) into one fp32 vector."""
    return np.concatenate([np.asarray(arrs[k], np.float32).ravel() for k in mask_keys])


def unflatten_masked(vec: np.ndarray, shapes: dict[str, tuple], mask_keys: list[str]) -> dict[str, np.ndarray]:
    """Inverse of flatten_masked."""
    out, off = {}, 0
    for k in mask_keys:
        n = int(np.prod(shapes[k]))
        out[k] = np.asarray(vec[off:off + n], np.float32).reshape(shapes[k])
        off += n
    if off != vec.size:
        raise ValueError(f"vector size {vec.size} != mask size {off}")
    return out


def cg_step(x: np.ndarray, r: np.ndarray, p: np.ndarray, hp_plus_lam_p: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """One damped-CG iteration for (H+lam I) x = b, given (H+lam I) p.

    Returns (x', r', p', pAp, negcurv). negcurv True when p^T(H+lam I)p <= 0 (caller bumps lam)."""
    pAp = float(np.dot(p, hp_plus_lam_p))
    if pAp <= 0.0:
        return x, r, p, pAp, True
    rr = float(np.dot(r, r))
    alpha = rr / pAp
    x2 = x + alpha * p
    r2 = r - alpha * hp_plus_lam_p
    beta = float(np.dot(r2, r2)) / rr
    p2 = r2 + beta * p
    return x2, r2, p2, pAp, False


def lm_accept(l0: float, l1: float, pred_red: float) -> tuple[bool, float]:
    """Levenberg accept/reject: returns (accept, rho). pred_red = predicted REDUCTION (>0 good)."""
    actual_red = l0 - l1
    rho = actual_red / pred_red if pred_red > 0 else float("-inf")
    return (actual_red > 0.0 and rho > 0.05), rho


def choose_subset_pairs(n: int, k: int, seed: int) -> list[int]:
    """Deterministic solve subset: rng(seed).choice without replacement, sorted."""
    return sorted(int(x) for x in np.random.default_rng(seed).choice(n, size=k, replace=False))


def ram_floor_ok(floor_gib: float = RAM_FLOOR_GIB) -> tuple[bool, float]:
    import psutil

    avail = psutil.virtual_memory().available / 2**30
    return avail >= floor_gib, avail


# ---------------------------------------------------------------------------
# Heavy context (lazy: only built by the stage runners).
# ---------------------------------------------------------------------------
class Ctx:
    """Loads checkpoint + curvelet frontend + gt lstars (lstars ONLY — memory-light) once."""

    def __init__(self, ckpt: Path):
        import sys
        for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
        from tac.boundary_math.lever_b_levelset_generator import (
            CurveletBankConfig,
            build_coords,
            curvelet_directional_B,
            curvelet_feats,
        )

        z = np.load(ckpt, allow_pickle=True)
        self.params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
        self.cfg = {k: (z[k][()] if z[k].shape == () else np.asarray(z[k])) for k in z.files if k.startswith("__")}
        self.h = int(self.cfg["__render_hw"][0])
        self.w = int(self.cfg["__render_hw"][1])
        self.keys = sorted(self.params.keys())
        self.shapes = {k: tuple(self.params[k].shape) for k in self.keys}
        bank = CurveletBankConfig(
            n_scales=int(self.cfg["__bank_n_scales"]), n_orient0=int(self.cfg["__bank_n_orient0"]),
            f0=float(self.cfg["__bank_f0"]), base=float(self.cfg["__bank_base"]),
            n_iso=int(self.cfg["__bank_n_iso"]))
        max_bf = float(self.cfg["__cfg_max_bank_freq"])
        B = curvelet_directional_B(bank, max_freq=(None if max_bf < 0 else max_bf))
        self.coords = build_coords(self.h, self.w)
        self.curv = curvelet_feats(self.coords, B).astype(np.float32)
        gtz = np.load(GT_CACHE)
        self.n_pairs = int(gtz["n_pairs"])
        self.lstars = np.asarray(gtz["lstars"], np.int64)  # (P, h, w) — lstars only, ~236MB int64
        self._fwd_kw = {
            "n_hidden": int(self.cfg["__cfg_n_hidden"]), "hidden_dim": int(self.cfg["__cfg_hidden_dim"]),
            "n_classes": 5, "activation": str(self.cfg["__cfg_activation"]),
            "softmax_temp": float(self.cfg["__cfg_softmax_temp"]),
            "wire_w0": float(self.cfg["__cfg_wire_w0"]), "wire_s0": float(self.cfg["__cfg_wire_s0"]),
            "hosc_beta": float(self.cfg["__cfg_hosc_beta"]), "hosc_omega": float(self.cfg["__cfg_hosc_omega"]),
            "chroma": bool(int(self.cfg["__cfg_chroma"])),
        }

    def dir_feats(self, argmax_hw: np.ndarray) -> np.ndarray:
        from tac.boundary_math.lever_b_generator import self_orientation_directional_feats

        return self_orientation_directional_feats(
            self.coords, np.asarray(argmax_hw, np.int64),
            n_freqs=int(self.cfg["__cfg_n_dir_freqs"]),
            freq_across=float(self.cfg["__cfg_freq_across"]),
            freq_along=float(self.cfg["__cfg_freq_along"])).astype(np.float32)

    def feats_for(self, argmax_hw: np.ndarray) -> np.ndarray:
        return np.concatenate([self.curv, self.dir_feats(argmax_hw)], axis=-1).astype(np.float32)

    def deploy(self, params: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        from tac.boundary_math.lever_b_levelset_generator import int8_dequant_params

        return int8_dequant_params(dict(params if params is not None else self.params))

    def fwd_numpy(self, deploy: dict, feats: np.ndarray, code_row: np.ndarray):
        from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

        return levelset_rgb_forward_numpy(deploy, feats, code_row, **self._fwd_kw)


def _pairs_arg(ctx: Ctx, spec: str) -> list[int]:
    if spec == "all":
        return list(range(ctx.n_pairs))
    if spec.startswith("stride:"):
        s = int(spec.split(":")[1])
        return list(range(0, ctx.n_pairs, s))
    return [int(x) for x in spec.split(",")]


# ---------------------------------------------------------------------------
# STAGE feats: fixed-point self-orient reconstruction (chunked, resumable).
# ---------------------------------------------------------------------------
def stage_feats(args) -> int:
    ok, avail = ram_floor_ok()
    if not ok:
        print(json.dumps({"stage": "refuse_ram", "avail_gib": round(avail, 1)}))
        return 4
    ctx = Ctx(args.params)
    pairs = _pairs_arg(ctx, args.pairs)
    st_path = PROBE_DIR / f"feats_state_{args.tag}.npz"
    if st_path.exists():
        st = np.load(st_path)
        argmax_prev = np.asarray(st["argmax_prev"], np.int8)
        argmax_cur = np.asarray(st["argmax_cur"], np.int8)
        iters_done, cursor = int(st["iters_done"]), int(st["cursor"])
        assert list(st["pairs"]) == pairs, "pairs spec changed mid-state"
    else:
        if args.seed_from == "gt":
            argmax_prev = np.stack([ctx.lstars[pi].astype(np.int8) for pi in pairs])
        else:
            argmax_prev = np.zeros((len(pairs), ctx.h, ctx.w), np.int8)
        argmax_cur = argmax_prev.copy()
        iters_done, cursor = 0, 0
    deploy = ctx.deploy()
    t0 = time.time()
    flips = 0
    px = 0
    while iters_done < args.iters:
        while cursor < len(pairs):
            if time.time() - t0 > args.max_seconds:
                _save_feats_state(st_path, pairs, argmax_prev, argmax_cur, iters_done, cursor)
                print(json.dumps({"stage": "feats_partial", "tag": args.tag, "iter": iters_done,
                                  "cursor": cursor, "secs": round(time.time() - t0, 1)}))
                return 0
            pi = pairs[cursor]
            feats = ctx.feats_for(argmax_prev[cursor])
            _rgb, phi = ctx.fwd_numpy(deploy, feats, deploy["code"][2 * pi + 1])
            am = phi.argmax(-1).reshape(ctx.h, ctx.w).astype(np.int8)
            flips += int((am != argmax_prev[cursor]).sum())
            px += am.size
            argmax_cur[cursor] = am
            cursor += 1
        iters_done += 1
        cursor = 0
        argmax_prev = argmax_cur.copy()
        print(json.dumps({"stage": "feats_iter_done", "tag": args.tag, "iters_done": iters_done,
                          "flip_frac": flips / max(px, 1), "secs": round(time.time() - t0, 1)}))
        flips = px = 0
    _save_feats_state(st_path, pairs, argmax_prev, argmax_cur, iters_done, cursor)
    print(json.dumps({"stage": "feats_done", "tag": args.tag, "iters_done": iters_done,
                      "n_pairs": len(pairs), "secs": round(time.time() - t0, 1)}))
    return 0


def _save_feats_state(path: Path, pairs, argmax_prev, argmax_cur, iters_done, cursor) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pairs=np.asarray(pairs, np.int32), argmax_prev=argmax_prev,
                        argmax_cur=argmax_cur, iters_done=iters_done, cursor=cursor)
    tmp.replace(path)


def _feats_state_argmax(tag: str) -> tuple[list[int], np.ndarray]:
    st = np.load(PROBE_DIR / f"feats_state_{tag}.npz")
    return [int(x) for x in st["pairs"]], np.asarray(st["argmax_prev"], np.int8)


# ---------------------------------------------------------------------------
# STAGE verdict: chunked realized d_seg on given params (torch-CPU authority path).
# ---------------------------------------------------------------------------
def stage_verdict(args) -> int:
    ok, avail = ram_floor_ok()
    if not ok:
        print(json.dumps({"stage": "refuse_ram", "avail_gib": round(avail, 1)}))
        return 4
    import sys
    for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8, cpu_verdict_d_seg_batch

    from tac.boundary_math.seg_core import load_real_segnet

    ctx = Ctx(args.params)
    pairs, argmax_state = _feats_state_argmax(args.tag)
    if args.pairs != "state":
        want = _pairs_arg(ctx, args.pairs)
        missing = [p for p in want if p not in set(pairs)]
        if missing:
            raise SystemExit(f"pairs {missing[:5]}... not in feats state {args.tag}")
        sel = [pairs.index(p) for p in want]
    else:
        want, sel = pairs, list(range(len(pairs)))
    jl = PROBE_DIR / f"verdict_{args.out}.jsonl"
    done = set()
    if jl.exists():
        for line in jl.read_text().splitlines():
            done.add(int(json.loads(line)["pair"]))
    todo = [(p, s) for p, s in zip(want, sel, strict=True) if p not in done]
    if not todo:
        return _verdict_summary(args, jl, want)
    deploy = ctx.deploy()
    seg_cpu = load_real_segnet("cpu")
    t0 = time.time()
    vb = int(args.vbatch)
    for s0 in range(0, len(todo), vb):
        if time.time() - t0 > args.max_seconds:
            print(json.dumps({"stage": "verdict_partial", "out": args.out, "done": len(done),
                              "secs": round(time.time() - t0, 1)}))
            return 0
        ok, avail = ram_floor_ok()
        if not ok:
            print(json.dumps({"stage": "refuse_ram_mid", "avail_gib": round(avail, 1)}))
            return 4
        chunk = todo[s0:s0 + vb]
        f1s, ls = [], []
        for pi, si in chunk:
            feats = ctx.feats_for(argmax_state[si])
            rgb, _phi = ctx.fwd_numpy(deploy, feats, deploy["code"][2 * pi + 1])
            f1s.append(_torch_R_to_camera_uint8(rgb.reshape(ctx.h, ctx.w, 3)))
            ls.append(ctx.lstars[pi])
        ds = cpu_verdict_d_seg_batch(seg_cpu, f1s, ls)
        with jl.open("a") as fh:
            for (pi, _si), d in zip(chunk, ds, strict=True):
                fh.write(json.dumps({"pair": int(pi), "d_seg": float(d)}) + "\n")
                done.add(pi)
        print(json.dumps({"stage": "verdict_chunk", "done": len(done), "of": len(want),
                          "secs": round(time.time() - t0, 1)}), flush=True)
    return _verdict_summary(args, jl, want)


def _verdict_summary(args, jl: Path, want: list[int]) -> int:
    rows = {int(r["pair"]): float(r["d_seg"]) for r in map(json.loads, jl.read_text().splitlines())}
    if set(want) - set(rows):
        print(json.dumps({"stage": "verdict_incomplete", "missing": len(set(want) - set(rows))}))
        return 0
    d = float(np.mean([rows[p] for p in want]))
    out = {"stage": "verdict_done", "out": args.out, "params": str(args.params), "tag": args.tag,
           "n_pairs": len(want), "d_seg": d,
           "axis": "[macOS-CPU advisory] NON-PROMOTABLE (numpy-fp32 deploy fwd + frozen CPU-torch SegNet)"}
    (PROBE_DIR / f"verdict_{args.out}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return 0


# ---------------------------------------------------------------------------
# STAGE solve: damped Newton-CG (Levenberg) on head|full mask, resumable.
# ---------------------------------------------------------------------------
class SolveCtx:
    """MLX loss/grad/HVP context for the solve subset (built per invocation)."""

    def __init__(self, ctx: Ctx, tag: str, k_pairs: int, subset_seed: int, theta: dict[str, np.ndarray]):
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        from train_levelset_witness_realized_through_R_mlx import (
            _eikonal_length_mlx,
            build_levelset_rgb_witness,
        )
        from train_witness_realized_through_R_mlx import render_through_R_mlx

        from tac.local_acceleration.mlx_scorer_adapters import (
            load_mlx_distortion_scorer_adapter_from_upstream,
        )

        mx.set_default_device(mx.cpu)
        self.mx = mx
        self.ctx = ctx
        self.subset = choose_subset_pairs(ctx.n_pairs, k_pairs, subset_seed)
        pairs, argmax_state = _feats_state_argmax(tag)
        idx = {p: i for i, p in enumerate(pairs)}
        missing = [p for p in self.subset if p not in idx]
        if missing:
            raise SystemExit(f"solve subset pairs {missing} not in feats state {tag}")
        self.model = build_levelset_rgb_witness(
            num_pairs=ctx.n_pairs, in_feat=int(ctx.cfg["__cfg_in_feat"]),
            hidden_dim=int(ctx.cfg["__cfg_hidden_dim"]), n_hidden=int(ctx.cfg["__cfg_n_hidden"]),
            mod_dim=int(theta["code"].shape[1]), n_classes=5,
            activation=str(ctx.cfg["__cfg_activation"]), softmax_temp=float(ctx.cfg["__cfg_softmax_temp"]),
            wire_w0=float(ctx.cfg["__cfg_wire_w0"]), wire_s0=float(ctx.cfg["__cfg_wire_s0"]),
            hosc_beta=float(ctx.cfg["__cfg_hosc_beta"]), hosc_omega=float(ctx.cfg["__cfg_hosc_omega"]),
            chroma=bool(int(ctx.cfg["__cfg_chroma"])))
        self._tree_unflatten = tree_unflatten
        self._render = render_through_R_mlx
        self._eik_len = _eikonal_length_mlx
        self.adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        self.cf = {}
        self.oh = {}
        for pi in self.subset:
            self.cf[pi] = mx.array(ctx.feats_for(argmax_state[idx[pi]]))
            oh = np.zeros((1, ctx.h, ctx.w, 5), np.float32)
            for k in range(5):
                oh[0, :, :, k] = ctx.lstars[pi] == k
            self.oh[pi] = mx.array(oh)
        self.keys = ctx.keys

    def _pair_loss(self, pi: int, *arrs):
        mx = self.mx
        self.model.update(self._tree_unflatten([(k, a) for k, a in zip(self.keys, arrs, strict=True)]))
        f1 = self._render(self.model, self.cf[pi], 2 * pi + 1, self.ctx.h, self.ctx.w)
        logits = self.adapter.segnet(f1)
        gt_logit = mx.sum(logits * self.oh[pi], axis=-1)
        runner = mx.max(logits + self.oh[pi] * (-1e9), axis=-1)
        signed = gt_logit - runner
        zz = -signed / TAU
        seg_l = mx.mean(TAU * mx.logaddexp(mx.zeros_like(zz), zz))
        phi0 = self.model.sdf(self.cf[pi], 2 * pi)
        _eik, length, _ = self._eik_len(phi0, self.ctx.h, self.ctx.w)
        return W_SEG * seg_l + LEN_W * length

    def loss_at(self, theta: dict[str, np.ndarray]) -> float:
        mx = self.mx
        arrs = [mx.array(theta[k]) for k in self.keys]
        tot = 0.0
        for pi in self.subset:
            val = self._pair_loss(pi, *arrs)
            mx.eval(val)
            tot += float(val)
            mx.clear_cache()
        return tot / len(self.subset)

    def grad_pair(self, theta: dict[str, np.ndarray], pi: int) -> dict[str, np.ndarray]:
        mx = self.mx
        arrs = [mx.array(theta[k]) for k in self.keys]
        g = mx.grad(lambda *a: self._pair_loss(pi, *a), argnums=list(range(len(self.keys))))(*arrs)
        mx.eval(g)
        out = {k: np.asarray(gi, np.float32) for k, gi in zip(self.keys, g, strict=True)}
        mx.clear_cache()
        return out

    def hvp_pair(self, theta: dict[str, np.ndarray], v: dict[str, np.ndarray], pi: int) -> dict[str, np.ndarray]:
        """Hessian-vector product via vjp of grad (H symmetric; MLX jvp lacks SliceUpdate)."""
        mx = self.mx
        arrs = [mx.array(theta[k]) for k in self.keys]
        vec = [mx.array(v.get(k, np.zeros(self.ctx.shapes[k], np.float32))) for k in self.keys]

        def gradfn(*a):
            return mx.grad(lambda *b: self._pair_loss(pi, *b), argnums=list(range(len(self.keys))))(*a)

        _, hv = mx.vjp(gradfn, arrs, vec)
        mx.eval(hv)
        out = {k: np.asarray(h, np.float32) for k, h in zip(self.keys, hv, strict=True)}
        mx.clear_cache()
        return out


def _mask_keys(ctx: Ctx, mask: str) -> list[str]:
    if mask == "head":
        return [k for k in ctx.keys if k in HEAD_KEYS]
    return list(ctx.keys)


def stage_solve(args) -> int:
    ok, avail = ram_floor_ok()
    if not ok:
        print(json.dumps({"stage": "refuse_ram", "avail_gib": round(avail, 1)}))
        return 4
    ctx = Ctx(args.params)
    mask_keys = _mask_keys(ctx, args.mask)
    st_path = PROBE_DIR / f"solve_state_{args.mask}.npz"
    log_path = PROBE_DIR / f"solve_log_{args.mask}.jsonl"

    def log(row: dict) -> None:
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if st_path.exists():
        st = dict(np.load(st_path, allow_pickle=False))
        theta = {k: np.asarray(st[f"theta__{k}"], np.float32) for k in ctx.keys}
        scal = json.loads(str(st["scal_json"]))
        g = np.asarray(st["g"], np.float32) if "g" in st else None
        x = np.asarray(st["x"], np.float32) if "x" in st else None
        r = np.asarray(st["r"], np.float32) if "r" in st else None
        p = np.asarray(st["p"], np.float32) if "p" in st else None
        g_partial = np.asarray(st["g_partial"], np.float32) if "g_partial" in st else None
    else:
        theta = dict(ctx.params)
        scal = {"phase": "grad", "lam": float(args.lam0), "round": 0, "cg_iter": 0,
                "grad_cursor": 0, "L0": None, "cg_iters_total": 0,
                "lam_schedule": [float(args.lam0)], "done": False,
                "k_pairs": int(args.k_pairs), "subset_seed": int(args.subset_seed)}
        g = x = r = p = g_partial = None
    if scal.get("done"):
        log({"stage": "solve_already_done", "mask": args.mask})
        return 0
    sc = SolveCtx(ctx, args.tag, scal["k_pairs"], scal["subset_seed"], theta)
    t0 = time.time()

    def save() -> None:
        arrays = {f"theta__{k}": theta[k] for k in ctx.keys}
        arrays["scal_json"] = np.asarray(json.dumps(scal))
        for nm, arr in (("g", g), ("x", x), ("r", r), ("p", p), ("g_partial", g_partial)):
            if arr is not None:
                arrays[nm] = arr
        tmp = st_path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(st_path)

    def timeup() -> bool:
        return time.time() - t0 > args.max_seconds

    while not scal.get("done"):
        if scal["phase"] == "grad":
            if scal["L0"] is None:
                scal["L0"] = sc.loss_at(theta)
                log({"stage": "subset_loss_theta", "L0": scal["L0"], "round": scal["round"],
                     "subset": sc.subset})
            if g_partial is None:
                g_partial = np.zeros(sum(int(np.prod(ctx.shapes[k])) for k in mask_keys), np.float32)
            while scal["grad_cursor"] < len(sc.subset):
                if timeup():
                    save()
                    log({"stage": "solve_partial", "phase": "grad", "cursor": scal["grad_cursor"]})
                    return 0
                pi = sc.subset[scal["grad_cursor"]]
                gp = sc.grad_pair(theta, pi)
                g_partial = g_partial + flatten_masked(gp, mask_keys) / len(sc.subset)
                scal["grad_cursor"] += 1
            g = g_partial
            g_partial = None
            x = np.zeros_like(g)
            r = -g.copy()
            p = r.copy()
            scal["phase"] = "cg"
            scal["cg_iter"] = 0
            log({"stage": "grad_done", "round": scal["round"], "gnorm_masked": float(np.linalg.norm(g))})
            save()
        elif scal["phase"] == "cg":
            while scal["cg_iter"] < args.cg_iters:
                if timeup():
                    save()
                    log({"stage": "solve_partial", "phase": "cg", "cg_iter": scal["cg_iter"]})
                    return 0
                if float(np.linalg.norm(r)) <= args.cg_tol * float(np.linalg.norm(g)):
                    log({"stage": "cg_converged", "cg_iter": scal["cg_iter"],
                         "rnorm": float(np.linalg.norm(r))})
                    break
                vd = unflatten_masked(p, ctx.shapes, mask_keys)
                hp = np.zeros_like(p)
                for pi in sc.subset:
                    hp = hp + flatten_masked(sc.hvp_pair(theta, vd, pi), mask_keys) / len(sc.subset)
                hp_lam = hp + scal["lam"] * p
                x2, r2, p2, pAp, negcurv = cg_step(x, r, p, hp_lam)
                if negcurv:
                    # p^T (H+lam I) p <= 0: bump lam and restart CG from x=0 (cheap + safe;
                    # recomputing (H+lam)x for a warm restart would cost another K HVPs).
                    scal["lam"] *= 10.0
                    scal["lam_schedule"].append(scal["lam"])
                    log({"stage": "negcurv_bump", "lam": scal["lam"], "pAp": pAp,
                         "cg_iter": scal["cg_iter"]})
                    x = np.zeros_like(g)
                    r = -g.copy()
                    p = r.copy()
                    scal["cg_iter"] = 0
                    save()
                    continue
                x, r, p = x2, r2, p2
                scal["cg_iter"] += 1
                scal["cg_iters_total"] += 1
                log({"stage": "cg_iter", "round": scal["round"], "cg_iter": scal["cg_iter"],
                     "rnorm": float(np.linalg.norm(r)), "xnorm": float(np.linalg.norm(x)),
                     "pAp": pAp, "lam": scal["lam"]})
                save()
            scal["phase"] = "check"
            save()
        elif scal["phase"] == "check":
            delta = unflatten_masked(x, ctx.shapes, mask_keys)
            theta_c = dict(theta)
            for k in mask_keys:
                theta_c[k] = theta[k] + delta[k]
            l1 = sc.loss_at(theta_c)
            # predicted reduction under the damped quadratic: -(g.x + 0.5 x.(H+lam)x); at CG iterate,
            # (H+lam)x = -r - ... use pred = -(g.x) - 0.5*x.((-r) - g)  since r = -g - (H+lam)x.
            hx = -r - g  # (H+lam I) x  — exact from the CG residual definition
            pred_red = -(float(np.dot(g, x)) + 0.5 * float(np.dot(x, hx)))
            accept, rho = lm_accept(scal["L0"], l1, pred_red)
            log({"stage": "tr_check", "round": scal["round"], "L0": scal["L0"], "L1": l1,
                 "pred_red": pred_red, "rho": rho, "accept": accept, "lam": scal["lam"],
                 "step_norm": float(np.linalg.norm(x))})
            if accept:
                for k in mask_keys:
                    theta[k] = theta_c[k]
                scal["L0"] = l1
                scal["lam"] = max(scal["lam"] / 3.0, 1e-6)
            else:
                scal["lam"] *= 10.0
            scal["lam_schedule"].append(scal["lam"])
            scal["round"] += 1
            if scal["round"] >= args.rounds or (not accept and scal["lam"] > 1e5):
                scal["done"] = True
                g = x = r = p = None
                save()
                solved = PROBE_DIR / f"solved_theta_{args.mask}.npz"
                arrays = {k: theta[k] for k in ctx.keys}
                zsrc = np.load(args.params, allow_pickle=True)
                for ck in zsrc.files:
                    if ck.startswith("__"):
                        arrays[ck] = zsrc[ck]
                np.savez(solved, **arrays)
                log({"stage": "solve_done", "mask": args.mask, "solved": str(solved),
                     "rounds": scal["round"], "cg_iters_total": scal["cg_iters_total"],
                     "lam_schedule": scal["lam_schedule"], "final_subset_loss": scal["L0"]})
                return 0
            scal["phase"] = "grad"
            scal["grad_cursor"] = 0
            g = x = r = p = g_partial = None
            save()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--params", type=Path, default=DEFAULT_CKPT)
    common.add_argument("--tag", type=str, default="gt1")
    common.add_argument("--max-seconds", type=float, default=420.0)

    f = sub.add_parser("feats", parents=[common])
    f.add_argument("--seed-from", choices=["gt", "zero"], default="gt")
    f.add_argument("--iters", type=int, default=1)
    f.add_argument("--pairs", type=str, default="all")

    v = sub.add_parser("verdict", parents=[common])
    v.add_argument("--pairs", type=str, default="state")
    v.add_argument("--out", type=str, required=True)
    v.add_argument("--vbatch", type=int, default=12)

    s = sub.add_parser("solve", parents=[common])
    s.add_argument("--mask", choices=["head", "full"], required=True)
    s.add_argument("--k-pairs", type=int, default=8)
    s.add_argument("--subset-seed", type=int, default=0)
    s.add_argument("--cg-iters", type=int, default=20, help="max CG iterations per LM round")
    s.add_argument("--cg-tol", type=float, default=0.05, help="relative residual stop")
    s.add_argument("--rounds", type=int, default=3, help="LM rounds")
    s.add_argument("--lam0", type=float, default=0.1)

    args = ap.parse_args(argv)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "feats":
        return stage_feats(args)
    if args.stage == "verdict":
        return stage_verdict(args)
    if args.stage == "solve":
        return stage_solve(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
