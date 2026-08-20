#!/usr/bin/env python3
"""ddm_rg1b: the WEIGHT-SPACE cosine between the stock and band-weighted gradients.

THE QUESTION
------------
The rg1 probe rotated the PIXEL-space objective from 2.16% to 99.29% of its loss
mass on the 1-px label band -- an 83.3 degree rotation, telemetry-confirmed to have
fired on all 600 steps -- and the trajectory did not move (peak 29,747 px vs A2's
27,170; ``improved_over_init=False``; ``best_step=0``).  Two mechanisms explain that,
and they route to opposite places:

* **PARAMETRIZATION COLLINEARITY** -- the decoder's Jacobian cannot express
  band-selective motion, so the pixel-space rotation collapses in weight space.
  Then the pixel-reweighting FAMILY is dead on this vehicle and the lever must be
  architectural (per-edge / local DOF) or solve-based.
* **MAGNITUDE-DOMINATED DIFFUSION** -- the realized update WAS rotated, but flip
  response in this displacement regime is direction-insensitive.  Then short-window
  probes cannot test objectives at all, and the judge/window must change first.

``cos(g_stock, g_band)`` in WEIGHT space discriminates them.  Pre-registered:
``>= 0.95`` collinearity, ``< 0.90`` diffusion, between = both partly open.

WHAT IS REPRODUCED EXACTLY
--------------------------
The trainer's real gradient path, not a stand-in: the ``EditabilityLevers.applied``
QAT route with ``--weight-qat-q3q4`` (mixed q3/q4, straight-through), the exact-path
render through ``render_float``, the frozen SegNet, and the same curriculum phases.
Both gradients come from ONE shared forward per pair (``retain_graph``), so the
cosine is not contaminated by forward nondeterminism.

SAMPLING (m96: seeded RANDOM, never a prefix)
---------------------------------------------
Pairs are drawn with a recorded seed from all 600.  ``n`` is not asserted -- the
per-pair gradients yield EXACT Gram matrices, so the sampling CI is computed by
bootstrap over pairs in closed form, and a cosine-vs-n curve is reported alongside.

AXIS: ``[macOS advisory]`` gradient geometry at a fixed checkpoint.  NEVER a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr130_lift.band_objective import (
    band_weight_field,
    band_weight_table_sha256,
    curriculum_loss_weighted,
    load_band_weight_table,
)
from tac.pr130_lift.checkpoint_schema import (
    architecture_config_from_checkpoint,
)
from tac.pr130_lift.editability_levers import (
    EditabilityLeverConfig,
    EditabilityLevers,
)

N_TOTAL_PAIRS = 600
#: The three curriculum phases, sampled at the step the trainer would be in.
PHASE_STEPS: tuple[tuple[int, str], ...] = ((0, "ce"), (400, "softplus_margin"), (560, "expected_flip"))


class GradCosineError(RuntimeError):
    """Fail-closed error for instrument or custody violations."""


# ---------------------------------------------------------------------------
# parameter grouping
# ---------------------------------------------------------------------------
def parameter_group(name: str) -> str:
    """Functional group for a renderer parameter, from the real module names.

    ``blocks.N.dw`` / ``blocks.N.pw`` are the depthwise / pointwise convolutions,
    ``blocks.N.film`` is the per-pair FiLM conditioning (ns1's pose-critical
    subspace), ``blocks.N.norm`` the normalisation affine, plus the two embeddings,
    the coordinate mixer and the output head.
    """

    if ".film." in name:
        return "film"
    if ".dw." in name or ".pw." in name:
        return "conv"
    if ".norm." in name:
        return "norm"
    if name.startswith("token_embed"):
        return "token_embed"
    if name.startswith("frame_embed"):
        return "frame_embed"
    if name.startswith("coord_mix"):
        return "coord_mix"
    if name.startswith("head"):
        return "head"
    return "other"


def _load_lifted_qat() -> Any:
    import importlib.util

    lifted = REPO_ROOT / "src" / "tac" / "pr130_lift" / "lifted"
    spec = importlib.util.spec_from_file_location(
        "tac.pr130_lift.dynamic.train_semantic_quantized",
        lifted / "train_semantic_quantized.py",
    )
    if spec is None or spec.loader is None:
        raise GradCosineError("cannot load the lifted QAT trainer")
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    sys.path.insert(0, str(lifted))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_weights(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The LIVE training weights (never the EMA shadow) from any checkpoint form."""

    state = payload.get("training_state")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and all(
            torch.is_tensor(v) for v in candidate.values()
        ):
            return candidate
    if all(torch.is_tensor(v) for v in payload.values()):
        return payload  # bare state dict
    raise GradCosineError(f"cannot locate live weights; keys={list(payload)[:12]}")


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def measure(
    *,
    checkpoint: Path,
    label: str,
    challenge_root: Path,
    cache: Path,
    init_for_architecture: Path,
    pair_ids: Sequence[int],
    alpha: float,
    bits: int,
    device: torch.device,
) -> dict[str, Any]:
    qat = _load_lifted_qat()
    sys.path.insert(0, str(challenge_root.resolve()))
    import modules

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = architecture_config_from_checkpoint(
        torch.load(init_for_architecture, map_location="cpu", weights_only=False),
        consumer="ddm_rg1b_grad_cosine",
    )
    model = qat.SemanticTokenRenderer(
        width=architecture["width"],
        blocks=architecture["blocks"],
        frame_dim=architecture["frame_dim"],
        num_pairs=architecture["num_pairs"],
        num_tokens=architecture["num_tokens"],
        phase_y=architecture["phase_y"],
        phase_x=architecture["phase_x"],
        temporal_radius=architecture["temporal_radius"],
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in _live_weights(payload).items()})
    model.train()

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    tokens = torch.load(cache, map_location="cpu", weights_only=False)["seg"].long()
    if tokens.shape[0] != N_TOTAL_PAIRS:
        raise GradCosineError("semantic cache must contain all 600 pairs")

    # EXACTLY the trainer's lever route for --weight-qat-q3q4.
    levers = EditabilityLevers(
        EditabilityLeverConfig(
            weight_qat_q3q4=True, weight_qat_low_bits=3, weight_qat_high_bits=bits
        )
    )
    table = load_band_weight_table()

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    sizes = [p.numel() for _, p in model.named_parameters() if p.requires_grad]
    offsets = np.cumsum([0, *sizes])
    groups = [parameter_group(n) for n in names]
    total = int(offsets[-1])

    def flat_grad() -> torch.Tensor:
        out = torch.zeros(total, dtype=torch.float64)
        for index, (_, parameter) in enumerate(
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ):
            if parameter.grad is not None:
                out[offsets[index] : offsets[index + 1]] = (
                    parameter.grad.detach().reshape(-1).cpu().double()
                )
        return out

    # per-phase, per-variant per-pair gradients
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        phase: {"stock": [], "band": []} for _, phase in PHASE_STEPS
    }
    started = time.time()
    for position, pair_id in enumerate(pair_ids):
        batch_ids = torch.tensor([int(pair_id)], dtype=torch.long)
        idx = batch_ids.to(device)
        conditioning = qat.gather_conditioning(
            tokens, batch_ids, model.temporal_radius
        ).to(device)
        target = tokens[batch_ids].to(device)

        # ONE shared forward; both gradients are read from the same graph.
        with levers.applied(model, base_bits=bits):
            frame = qat.render_float(model, conditioning, idx, exact_path=True)
        logits = segnet(frame)
        weight = band_weight_field(target, table, alpha)

        for step, phase in PHASE_STEPS:
            for variant, weight_argument in (("stock", None), ("band", weight)):
                model.zero_grad(set_to_none=True)
                loss, realised = curriculum_loss_weighted(
                    logits, target, step, 600, 0.50, 0.85, weight_argument
                )
                if realised != phase:
                    raise GradCosineError(f"phase drift: {realised} != {phase}")
                loss.backward(retain_graph=True)
                collected[phase][variant].append(flat_grad())
        model.zero_grad(set_to_none=True)
        del logits, frame
        if (position + 1) % 10 == 0:
            elapsed = time.time() - started
            print(
                f"  [{label}] {position + 1}/{len(pair_ids)} pairs "
                f"({elapsed:.0f}s, {elapsed / (position + 1):.2f}s/pair)",
                file=sys.stderr,
                flush=True,
            )

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "n_pairs": len(pair_ids),
        "wall_s": time.time() - started,
        "analysis": _analyse(collected, names, groups, offsets, total),
    }


def _cosine(gram_ab: np.ndarray, gram_aa: np.ndarray, gram_bb: np.ndarray,
            selection: np.ndarray | None = None) -> float:
    if selection is None:
        num = gram_ab.sum()
        den = np.sqrt(gram_aa.sum() * gram_bb.sum())
    else:
        counts = np.bincount(selection, minlength=gram_ab.shape[0]).astype(np.float64)
        num = counts @ gram_ab @ counts
        den = np.sqrt((counts @ gram_aa @ counts) * (counts @ gram_bb @ counts))
    return float(num / den) if den > 0 else float("nan")


def _analyse(
    collected: dict[str, dict[str, list[torch.Tensor]]],
    names: list[str],
    groups: list[str],
    offsets: np.ndarray,
    total: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260816)
    out: dict[str, Any] = {}
    for phase, variants in collected.items():
        stock = torch.stack(variants["stock"]).numpy()
        band = torch.stack(variants["band"]).numpy()
        n = stock.shape[0]
        # FAIL CLOSED on the physics.  numpy 1.26 + Accelerate emits spurious
        # divide-by-zero / overflow / invalid RuntimeWarnings from *any* wide
        # float64 matmul on Apple silicon -- reproduced here on pure random
        # finite input, with the result matching np.einsum to 2.8e-14 relative.
        # So the warnings are suppressed narrowly around the Gram products
        # ONLY, and the actual gradients are checked for finiteness HERE, where
        # a real non-finite value must refuse rather than be swallowed.
        for name, block in (("stock", stock), ("band", band)):
            if not np.isfinite(block).all():
                raise GradCosineError(
                    f"{phase}/{name}: non-finite gradient entries "
                    f"({int((~np.isfinite(block)).sum())} of {block.size})"
                )

        def grams(
            columns: slice | np.ndarray,
            stock: np.ndarray = stock,
            band: np.ndarray = band,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            a, b = stock[:, columns], band[:, columns]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                return a @ b.T, a @ a.T, b @ b.T

        gram_ab, gram_aa, gram_bb = grams(slice(None))
        full = _cosine(gram_ab, gram_aa, gram_bb)

        # bootstrap over PAIRS, exact from the Gram matrices (no re-forward)
        boots = np.array(
            [
                _cosine(gram_ab, gram_aa, gram_bb, rng.integers(0, n, size=n))
                for _ in range(2000)
            ]
        )
        # cosine as a function of n, to show the estimate has converged
        curve = {}
        for size in sorted({4, 8, 16, 32, 64, min(n, 96), n}):
            if size > n:
                continue
            values = [
                _cosine(
                    gram_ab, gram_aa, gram_bb,
                    rng.choice(n, size=size, replace=False),
                )
                for _ in range(200)
            ]
            curve[str(size)] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if size > 1 else 0.0,
            }

        # per-parameter-group
        per_group: dict[str, Any] = {}
        for group in sorted(set(groups)):
            columns = np.concatenate(
                [
                    np.arange(offsets[i], offsets[i + 1])
                    for i, g in enumerate(groups)
                    if g == group
                ]
            )
            g_ab, g_aa, g_bb = grams(columns)
            per_group[group] = {
                "cosine": _cosine(g_ab, g_aa, g_bb),
                "n_params": int(columns.size),
                "stock_norm_share": float(
                    np.sqrt(g_aa.sum()) / np.sqrt(gram_aa.sum())
                ),
                "band_norm_share": float(np.sqrt(g_bb.sum()) / np.sqrt(gram_bb.sum())),
            }

        # per-tensor, so a single collinear block cannot hide inside a group
        per_tensor = {}
        for index, name in enumerate(names):
            columns = np.arange(offsets[index], offsets[index + 1])
            t_ab, t_aa, t_bb = grams(columns)
            per_tensor[name] = {
                "cosine": _cosine(t_ab, t_aa, t_bb),
                "group": groups[index],
                "stock_norm_share": float(np.sqrt(t_aa.sum()) / np.sqrt(gram_aa.sum())),
            }

        # ADAM-LIMIT PROXY.  The trainer optimises with AdamW, whose update is
        # per-coordinate normalised (~ m / sqrt(v)).  So cos(g_stock, g_band) is
        # the REALISED-update rotation only under SGD; under Adam the magnitude
        # is divided out and what survives is closer to sign(g).  Report the
        # sign-space cosine and the raw sign-agreement fraction as well, so the
        # adjudication is about the optimiser the run actually used.
        sum_stock, sum_band = stock.sum(axis=0), band.sum(axis=0)
        sign_stock, sign_band = np.sign(sum_stock), np.sign(sum_band)
        denominator = np.linalg.norm(sign_stock) * np.linalg.norm(sign_band)
        cosine_sign = float(sign_stock @ sign_band / denominator) if denominator else float("nan")
        nonzero = (sign_stock != 0) & (sign_band != 0)
        agreement = float(
            (sign_stock[nonzero] == sign_band[nonzero]).mean()
        ) if nonzero.any() else float("nan")

        per_group_sign = {}
        for group in sorted(set(groups)):
            columns = np.concatenate(
                [
                    np.arange(offsets[i], offsets[i + 1])
                    for i, g in enumerate(groups)
                    if g == group
                ]
            )
            a_sign, b_sign = sign_stock[columns], sign_band[columns]
            group_denominator = np.linalg.norm(a_sign) * np.linalg.norm(b_sign)
            group_nonzero = (a_sign != 0) & (b_sign != 0)
            per_group_sign[group] = {
                "cosine_sign": (
                    float(a_sign @ b_sign / group_denominator)
                    if group_denominator
                    else float("nan")
                ),
                "sign_agreement": (
                    float((a_sign[group_nonzero] == b_sign[group_nonzero]).mean())
                    if group_nonzero.any()
                    else float("nan")
                ),
            }

        out[phase] = {
            "cosine_global": full,
            "cosine_sign_adam_limit": cosine_sign,
            "sign_agreement_fraction": agreement,
            "per_group_sign": per_group_sign,
            "bootstrap_ci95": [
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ],
            "bootstrap_sd": float(boots.std(ddof=1)),
            "angle_degrees": float(np.degrees(np.arccos(np.clip(full, -1, 1)))),
            "norm_ratio_band_over_stock": float(
                np.sqrt(gram_bb.sum()) / np.sqrt(gram_aa.sum())
            ),
            "cosine_vs_n": curve,
            "per_group": per_group,
            "per_tensor": per_tensor,
            "n_parameters": total,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-root", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/"
            "gt_cache_600_official_ada.pt"  # GT_LINEAGE_OK: default bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
        ),
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/"
            "artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
        ),
    )
    parser.add_argument(
        "--also",
        type=Path,
        nargs="*",
        default=[],
        help="extra checkpoints (state-dependence check), as label=path",
    )
    parser.add_argument("--pairs", type=int, default=120)
    parser.add_argument("--sample-seed", type=int, default=20260816)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    # m96: seeded RANDOM sample, never a contiguous prefix.
    rng = np.random.default_rng(args.sample_seed)
    pair_ids = sorted(
        rng.choice(N_TOTAL_PAIRS, size=args.pairs, replace=False).tolist()
    )
    device = torch.device(args.device)

    targets: list[tuple[str, Path]] = [("init", args.init)]
    for entry in args.also:
        text = str(entry)
        label, _, path = text.partition("=")
        targets.append((label, Path(path)))

    results = []
    for label, checkpoint in targets:
        print(f"[ddm_rg1b] measuring {label}: {checkpoint}", file=sys.stderr, flush=True)
        results.append(
            measure(
                checkpoint=checkpoint,
                label=label,
                challenge_root=args.challenge_root,
                cache=args.cache,
                init_for_architecture=args.init,
                pair_ids=pair_ids,
                alpha=args.alpha,
                bits=args.bits,
                device=device,
            )
        )

    payload = {
        "schema": "ddm_rg1b_weight_space_gradient_cosine.v1",
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS advisory] gradient geometry at fixed checkpoints -- NEVER a score",
        "question": (
            "does the 83.3-degree PIXEL-space rotation survive into WEIGHT space? "
            ">=0.95 => parametrization collinearity; <0.90 => magnitude-dominated diffusion"
        ),
        "sampling": {
            "law": "m96 seeded RANDOM over all 600 pairs, never a contiguous prefix",
            "seed": args.sample_seed,
            "n": args.pairs,
            "pair_ids": pair_ids,
        },
        "config": {
            "alpha": args.alpha,
            "bits": args.bits,
            "device": str(device),
            "weight_qat_q3q4": True,
            "band_table_sha256": band_weight_table_sha256(),
            "cache_sha256": _sha256(args.cache),
            "init_sha256": _sha256(args.init),
            "phase_steps": [list(p) for p in PHASE_STEPS],
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[ddm_rg1b] wrote {args.out}", file=sys.stderr)
    for result in results:
        for phase, block in result["analysis"].items():
            print(
                f"  {result['label']:>10} {phase:<16} cos={block['cosine_global']:.6f} "
                f"CI95=[{block['bootstrap_ci95'][0]:.4f},{block['bootstrap_ci95'][1]:.4f}] "
                f"angle={block['angle_degrees']:.2f}deg "
                f"|band|/|stock|={block['norm_ratio_band_over_stock']:.3g} "
                f"cos_sign={block['cosine_sign_adam_limit']:.4f} "
                f"agree={block['sign_agreement_fraction']:.4f}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
