#!/usr/bin/env python3
"""ddm_p956 — MHAR row-2 probe: per-class gradient-subspace disagreement on TR1.

Task #956 (crosswalk `.omx/research/ddm_mhar1_crosswalk_20260805.md` row 2): before
building v8-lite per-class carriers/heads or the MHAR row-4 depth-mixing race, MEASURE
whether the 5 SegNet classes actually FIGHT over the shared TR1 trunk on TRAINED
weights (the paper's probe-the-driver methodology). Per class c: masked-CE gradient
of adapter.segnet(_apply_R(model.render_frame(idx))) over pixels with lstar==c,
accumulated over strided probe pairs; then pairwise cosine per parameter tensor and
per group (TRUNK conv stack vs TOKEN fields).

HIGH trunk disagreement (classes pulling in conflicting directions) = the
per-class-split premise has measured support (row-4 races via the row-1
identity-preserving entry protocol); LOW = shared trunk suffices, split dominated.

DUAL-METRIC (operator 2026-08-05 "plain cosine is pretty much never optimal" +
[[cosine_is_hardly_ever_optimal_prefer_fisher_derived_basis_20260803]] + m65
Euclid-vs-Fisher SIGN-FLIP law): every cosine is reported in BOTH metrics —
plain Euclidean AND diagonal-Fisher-whitened (per-coordinate 1/sqrt(E[g^2])
over all class x pair gradient samples), never one alone. The whitened metric
kills the plain-cosine pathology where a few large-magnitude coordinates
dominate the angle; a sign flip between the two metrics is itself a finding.

Axis: [macOS-CPU frozen-scorer advisory], score_claim=false. n=8 STRIDED pairs
(m88/m96: prefixes are biased; strided is the mitigation) — a DIRECTION probe for
routing, never a population-scale finding. #855 caveat: the MLX conv adapter has a
known 76-pixel argmax drift vs CPU-torch on real frames; this probe is COMPARATIVE
(all 5 class gradients share the same adapter), so the drift cancels to first order
for cosine purposes; recorded in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")  # canonical comma10k order
DEFAULT_PAIRS = "0,75,150,225,300,375,450,525"  # strided, never [:8] (m88/m96)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="winner checkpoint .npz (the TP1 boundary winner endpoint)")
    ap.add_argument("--config", type=Path, required=True,
                    help="tr1_config.json from the SAME run dir as --ckpt")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--gt-cache", type=Path,
                    default=Path("/Users/adpena/Projects/pact/experiments/results/"
                                 "mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--ema", choices=("on", "off"), default="on",
                    help="on = probe the EMA shadow (the gate basis)")
    ap.add_argument("--out", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_tp1_20260805/"
                                 "p956_class_grad_disagreement.json"))
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    mx.set_default_device(mx.cpu)  # dt1: CPU is run-to-run bit-identical

    import dataclasses

    from experiments.train_tr1_partition_renderer_mlx import (
        TR1Config, build_module, ema_snapshot_swap, load_checkpoint,
    )
    from experiments.train_witness_realized_through_R_mlx import _apply_R
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    cfg_json = json.loads(args.config.read_text())
    field_names = {f.name for f in dataclasses.fields(TR1Config)}
    cfg = TR1Config(**{k: v for k, v in cfg_json.items() if k in field_names})
    model = build_module(cfg)
    ck = load_checkpoint(args.ckpt, model)
    if args.ema == "on" and ck["ema"]:
        ema_snapshot_swap(model, ck["ema"])  # probe process exits; no restore needed

    upstream_root = str(REPO / "upstream")
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream_root, device="cpu")
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")

    pairs = [int(s) for s in str(args.pairs).split(",") if s.strip()]

    def class_ce(mdl, idx: int, cls: int, mask_mx, n_px: float):
        f1 = _apply_R(mdl.render_frame(int(idx)))          # (1,384,512,3) scorer plane
        logits = adapter.segnet(f1)                        # (1,384,512,5)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -(logp[0, :, :, cls] * mask_mx).sum() / n_px

    lvag = nn.value_and_grad(model, class_ce)

    # Accumulate per-class flat gradients (name -> float64 np array) + the
    # per-coordinate second moment over ALL class x pair samples (the empirical
    # diagonal Fisher for the whitened metric).
    acc: list[dict[str, np.ndarray]] = [dict() for _ in range(5)]
    acc_sq: dict[str, np.ndarray] = {}
    n_samples = 0
    skipped: list[dict] = []
    t0 = time.time()
    receipt: dict = {
        "schema": "p956_class_grad_disagreement.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] n8 strided pairs, NON-PROMOTABLE",
        "score_claim": False,
        "ckpt": str(args.ckpt), "config": str(args.config),
        "ema_basis": args.ema, "pairs": pairs,
        "adapter_caveat_855": ("MLX conv adapter ~76px argmax drift vs CPU-torch; "
                               "comparative probe — shared drift cancels to first order"),
        "method": "per-class masked CE grads via nn.value_and_grad; pairwise cosine "
                  "per tensor + per group (trunk vs tokens)",
        "status": "running",
    }
    for pi, idx in enumerate(pairs):
        lstar = np.asarray(lstars[idx], dtype=np.int64)
        for cls in range(5):
            mask = (lstar == cls)
            n_px = float(mask.sum())
            if n_px < 1.0:
                skipped.append({"pair": idx, "class": CLASS_NAMES[cls], "reason": "0 px"})
                continue
            loss, grads = lvag(model, idx, cls, mx.array(mask.astype(np.float32)), n_px)
            mx.eval(loss, grads)  # never let the lazy graph accumulate (#205 lesson)
            n_samples += 1
            for name, g in tree_flatten(grads):
                a = np.asarray(g, dtype=np.float64).ravel()
                if name in acc[cls]:
                    acc[cls][name] += a
                else:
                    acc[cls][name] = a
                if name in acc_sq:
                    acc_sq[name] += a * a
                else:
                    acc_sq[name] = a * a
        receipt["progress"] = {"pairs_done": pi + 1, "of": len(pairs),
                               "elapsed_s": round(time.time() - t0, 1)}
        args.out.write_text(json.dumps(receipt, indent=1))  # resumable-in-spirit partial

    def cos(a: np.ndarray, b: np.ndarray) -> float | None:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(a, b) / (na * nb))

    names = sorted(set().union(*[set(a.keys()) for a in acc if a]))
    trunk = [n for n in names if not n.startswith("tokens")]
    tokens = [n for n in names if n.startswith("tokens")]
    # Diagonal-Fisher whitening: per-coordinate 1/sqrt(E[g^2]) over all class x pair
    # samples. eps = 1e-3 of the tensor's own RMS floor (scale-free, no bare constant).
    whiten: dict[str, np.ndarray] = {}
    for n, sq in acc_sq.items():
        rms = np.sqrt(sq / max(1, n_samples))
        eps = 1e-3 * float(rms.mean()) if float(rms.mean()) > 0 else 1.0
        whiten[n] = 1.0 / (rms + eps)

    def group_matrix(group: list[str], whitened: bool) -> list[list[float | None]]:
        vecs = []
        for cls in range(5):
            parts = [(acc[cls][n] * whiten[n]) if whitened else acc[cls][n]
                     for n in group if n in acc[cls]]
            vecs.append(np.concatenate(parts) if parts else np.zeros(1))
        return [[cos(vecs[i], vecs[j]) for j in range(5)] for i in range(5)]

    def tensor_matrix(n: str, whitened: bool) -> list[list[float | None]]:
        def vec(cls: int) -> np.ndarray:
            a = acc[cls].get(n)
            if a is None:
                return np.zeros(1)
            return a * whiten[n] if whitened else a
        return [[cos(vec(i), vec(j)) for j in range(5)] for i in range(5)]

    def off_diag_mean(m) -> float | None:
        vals = [m[i][j] for i in range(5) for j in range(5) if i != j and m[i][j] is not None]
        return float(np.mean(vals)) if vals else None

    trunk_plain, trunk_fisher = group_matrix(trunk, False), group_matrix(trunk, True)
    tok_plain = group_matrix(tokens, False) if tokens else None
    tok_fisher = group_matrix(tokens, True) if tokens else None
    per_tensor = {n: {"plain": tensor_matrix(n, False), "fisher_whitened": tensor_matrix(n, True)}
                  for n in names}
    od_plain, od_fisher = off_diag_mean(trunk_plain), off_diag_mean(trunk_fisher)
    sign_flips = [(CLASS_NAMES[i], CLASS_NAMES[j], trunk_plain[i][j], trunk_fisher[i][j])
                  for i in range(5) for j in range(i + 1, 5)
                  if trunk_plain[i][j] is not None and trunk_fisher[i][j] is not None
                  and (trunk_plain[i][j] > 0) != (trunk_fisher[i][j] > 0)]

    # Pre-registered reading bands (probe-scoped interpretation aid, NOT a law;
    # registered before results were seen, per MHAR row-2). VERDICT keys off the
    # FISHER-WHITENED metric per the dual-metric law; plain is reported alongside.
    def band(od: float | None) -> str:
        if od is None:
            return "NO_SIGNAL"
        if od < 0.2:
            return "HIGH_DISAGREEMENT_split_premise_supported"
        if od > 0.6:
            return "LOW_DISAGREEMENT_shared_trunk_suffices"
        return "AMBIGUOUS_between_preregistered_bands"

    receipt.update({
        "status": "complete",
        "class_names": list(CLASS_NAMES),
        "trunk_tensors": trunk, "token_tensors": tokens,
        "n_grad_samples": n_samples,
        "trunk_cosine_5x5": {"plain": trunk_plain, "fisher_whitened": trunk_fisher},
        "token_cosine_5x5": ({"plain": tok_plain, "fisher_whitened": tok_fisher}
                             if tokens else None),
        "per_tensor_cosine_5x5": per_tensor,
        "trunk_offdiag_mean": {"plain": od_plain, "fisher_whitened": od_fisher},
        "token_offdiag_mean": {"plain": off_diag_mean(tok_plain) if tok_plain else None,
                               "fisher_whitened": off_diag_mean(tok_fisher) if tok_fisher else None},
        "trunk_pairwise_sign_flips_plain_vs_fisher": sign_flips,
        "skipped": skipped,
        "preregistered_bands": {"high": "<0.2", "low": ">0.6",
                                "verdict_metric": "fisher_whitened (dual-metric law m65)"},
        "verdict_advisory": band(od_fisher),
        "verdict_plain_for_signflip_watch": band(od_plain),
        "routing": ("HIGH -> MHAR row-4 depth-mixing + per-class routing race via row-1 "
                    "identity-preserving entry; LOW -> per-class split DOMINATED at this "
                    "endpoint (INSTANCE scope: this ckpt, these pairs); any plain-vs-fisher "
                    "sign flip is a finding in its own right"),
        "elapsed_s": round(time.time() - t0, 1),
    })
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("trunk_offdiag_mean", "token_offdiag_mean", "verdict_advisory",
                       "elapsed_s", "status")}, indent=1))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
