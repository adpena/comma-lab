#!/usr/bin/env python
"""ddm_pa1r (#793) — $0 SEAL + ticket builder for the Pool-A race (D16 warm-tail realization).

SCRATCHPAD measurement scaffold (rebuildable from B checkpoint + QA80 field + committed code;
NOT a committed module).  Grid-compatible realization of pa1b's Pool-A race: the arms are D16
WARM TAILS from B (dw1 control ep440) — NOT D8 fresh burns — because (a) NO D8/rowband checkpoint
exists anywhere (every tr1 ckpt is D16), so pa1b's D8 arms would need fresh full curriculum burns;
(b) the task frames the delta-sparsity arm as a WARM RESUME from B, from_step_0 (post-knee);
(c) delta group-sparsity + margin-coupled quant are GRID-AGNOSTIC (work on B's D16 24x32 grid);
(d) a D16 warm tail is directly comparable to the QA24 D16 joint-descent slope (FEED-reanchor).
The D8-rowband FOVEATION arm is a separate, expensive fresh-burn question (deferred).

Emits: the seal verdict (matched-SMEVR-bytes + argv-diff vs B's launch receipt) + 4 sealed tickets.
Pointer 0.1910828242 [contest-CPU] UNMOVED; score_claim=False.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.witness_dsl.ax1_pool_a_levers_20260730 import (
    QA80_FIELD_CUSTODY,
    apply_per_cell_quant_np,
    load_qa80_cell_field,
    margin_coupled_level_map,
)
from tac.witness_dsl.ax1_pool_a_race_20260730 import seal_matched_bytes_race
from tac.witness_dsl.curriculum_dsl import Lever
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TR1RendererProgramV1,
    lever_delta_group_sparsity,
    lever_token_quant_margin_coupling,
)

B_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/control")
B_CKPT = B_DIR / "checkpoints" / "stage_seg_trunk_tau_final.npz"
B_TICKET = Path("/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/tickets/control_ticket.json")
CELL_MASK = Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy")
GT_CACHE = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_pa1r_20260730")
LEVELS = 16
DELTA_W = 3e-2  # shrinkage strength commensurate w/ the existing rate force (w_rate 0.05);
#                 watched at fire — bump if bytes don't move, lower if d_seg spikes (single-weight
#                 discriminator of the exchange DIRECTION; a full weight-sweep is the follow-up)
TAIL_EPOCHS = 500  # resume B ep440 -> ~59-epoch warm tail
TAIL_WALL = 45.0


def b_uint8_codes() -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct B's FULL per-frame quantized token field EXACTLY as the trainer byte-closes it.

    Uses the EMA SHADOW (``ema::`` — the trainer ships the shadow at gate/eval per the EMA
    non-negotiable, NOT the live params); field = (base + delta) * keep; q = round((clip(field)
    +1)/2*(L-1)) uint8.  VERIFIED: ema+clip = 255,907 = nv1's authoritative anchor + B's own
    telemetry tokens_bytes_smevr.  (Live params drift to absmax ~1.9 and give 269,437 — the shadow
    is the shipped state.)"""
    z = np.load(B_CKPT)
    base = z["ema::tokens_base"].astype(np.float32)            # (24,32,4)
    delta = z["ema::tokens_delta"].astype(np.float32)          # (600,24,32,4)
    keep = np.load(CELL_MASK).astype(np.float32)               # (24,32) bool -> float
    if keep.ndim == 2:
        keep = keep[..., None]
    field = (base[None] + delta) * keep                        # (600,24,32,4)
    return np.round((np.clip(field, -1, 1) + 1.0) * 0.5 * (LEVELS - 1)).astype(np.uint8), field


def base_levers_from_b() -> list[Lever]:
    t = json.loads(B_TICKET.read_text())
    levers: list[Lever] = []
    for lv in t["levers"]:
        ov = dict(lv["overrides"])
        if lv["name"] == "tr1_window_ep441":                  # swap to the warm-tail window
            ov = {"--epochs": str(TAIL_EPOCHS), "--max-wall-minutes": str(TAIL_WALL),
                  "--batch-pairs": ov.get("--batch-pairs", "8"), "--lr": ov.get("--lr", "0.002")}
        levers.append(Lever(name=lv["name"], overrides=ov, notes=lv.get("notes", "")))
    return levers


def build_arm(name: str, extra: list[Lever]) -> TR1RendererProgramV1:
    levers = tuple(base_levers_from_b() + extra)
    prog = TR1RendererProgramV1(
        levers=levers, num_pairs=600, out_dir=str(OUT_ROOT / name), seed=0,
        gt_cache=GT_CACHE, resume_from=str(B_CKPT), full_confirm=True)
    prog.validate()
    return prog


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tickets").mkdir(exist_ok=True)

    # ---- reconstruct B's field + codes; price the control ----
    codes, field = b_uint8_codes()
    from experiments.ddm_r7_token_coder import encode_token_codes
    ctrl_bytes = len(encode_token_codes(np.ascontiguousarray(codes), levels=LEVELS, codec="smevr"))
    print(f"[seal] B SMEVR token bytes = {ctrl_bytes} (nv1 anchor 255,907; drift "
          f"{ctrl_bytes - 255907:+d})")

    # ---- margin level map at D16 (24,32) from the QA80 field ----
    agg = load_qa80_cell_field(24, 32, downsample=16, field_custody=QA80_FIELD_CUSTODY)
    level_map = margin_coupled_level_map(agg.flip_mass, base_levels=LEVELS, min_levels=LEVELS // 4)
    mq_field = apply_per_cell_quant_np(field, level_map[None])   # (600,24,32,4)
    mq_codes = np.round((np.clip(mq_field, -1, 1) + 1.0) * 0.5 * (LEVELS - 1)).astype(np.uint8)
    print(f"[seal] margin-quant level map: min={int(level_map.min())} max={int(level_map.max())} "
          f"distinct={np.unique(level_map).size}")

    # ---- the 4 D16 warm-tail arms ----
    coupling = Lever(name="tr1_coupling_field_only",
                     overrides={"--token-quant-coupling-field": QA80_FIELD_CUSTODY},
                     notes="bare QA80 field for xi_informed delta-sparsity (no margin-coupling)")
    dsparse = lever_delta_group_sparsity(DELTA_W, engage="from_step_0", weight_field="xi_informed")
    mquant = lever_token_quant_margin_coupling(QA80_FIELD_CUSTODY)

    arms_progs = {
        "control_tail": build_arm("control_tail", []),
        "delta_sparsity_tail": build_arm("delta_sparsity_tail", [coupling, dsparse]),
        "margin_quant_tail": build_arm("margin_quant_tail", [mquant]),
        "joint_tail": build_arm("joint_tail", [mquant, dsparse]),
    }

    # starting codes per arm (deterministic transform of B's field; in-loop levers start matched).
    arm_codes = {
        "control_tail": codes,
        "delta_sparsity_tail": codes,        # force is in-loop => starts identical to control
        "margin_quant_tail": mq_codes,       # per-cell level map => different starting budget
        "joint_tail": mq_codes,
    }

    # B's launch argv (the diff-law baseline).
    b_argv = json.loads((B_DIR / "launch_receipt.json").read_text())["argv"]

    arms_for_seal = {name: (p.compile_trainer_argv(), arm_codes[name])
                     for name, p in arms_progs.items()}
    verdict = seal_matched_bytes_race(b_argv, codes, arms_for_seal, levels=LEVELS, tol=0.01)

    # write tickets + the seal receipt
    for name, prog in arms_progs.items():
        tk = prog.sealed_ticket()
        (OUT_ROOT / "tickets" / f"{name}_ticket.json").write_text(
            json.dumps(tk, indent=2, sort_keys=True) + "\n")

    # argv-diff vs B for each arm (pre-fire law): the delta MUST be exactly the arm's lever set
    # (+ the window/out-dir/resume changes).
    seal_out = verdict.to_dict()
    seal_out["b_launch_argv_baseline"] = True
    seal_out["arm_starting_bytes"] = {
        n: len(encode_token_codes(np.ascontiguousarray(c), levels=LEVELS, codec="smevr"))
        for n, c in arm_codes.items()}
    (OUT_ROOT / "seal_verdict.json").write_text(json.dumps(seal_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
