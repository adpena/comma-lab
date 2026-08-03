#!/usr/bin/env python3
"""ddm_mt1 — assign every in-scope choice point to a triage class, and report the histogram.

The class assignment is a HAND JUDGEMENT recorded as data, so the memo's per-class counts are
GENERATED rather than transcribed.  Hand arithmetic over overlapping sections is exactly where a
denominator silently goes wrong; this module removes that failure mode.

Classes (the rule, validated on all three signs by pw1 / pb2 / dc1):

  AT_A_BOUND         occupancy piled at an admissible-set endpoint AND the next stage admits
                     beyond it  =>  freeing may pay
  DIRECTION          a direction resolved by a myopic probe; freeing may HURT (pb2)
  DEGENERATE         imposes no limit -- another shipped coordinate spans the same manifold, or
                     the row is a duplicate copy of another row (DOF below arity)
  NO_OCCUPANCY_DATA  no receipt records which value fired
  INTERIOR_CLOSED    occupancy measured and strictly interior, or the menu was raced exhaustively
  NO_DOF             fail-closed structural validation / dispatch arm: it passes or the program
                     dies; no setting of it trades score

Axis: [macOS-CPU $0] NON-PROMOTABLE. score_claim=false.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mt1_20260802/mt1_inventory.json"
)

AT_A_BOUND = "AT_A_BOUND"
DIRECTION = "DIRECTION"
DEGENERATE = "DEGENERATE"
NO_DATA = "NO_OCCUPANCY_DATA"
INTERIOR = "INTERIOR_CLOSED"
NO_DOF = "NO_DOF"

# Key = (file basename, symbol/comparison name, occurrence index within that file).
#
# NOT the row_id, and NOT the line number.  Both are POSITIONAL and both drifted under me while
# this arm was running: a sister arm edited ddm_v4c_resolve.py and ddm_v4d_resolve.py mid-flight,
# shifting every downstream line number AND every sequential row_id.  A key that moves when an
# unrelated line is inserted is an instrument that silently mis-attributes.  This key survives
# insertions, deletions and reordering; only a rename or a change in the number of same-named
# sites in one file can disturb it, and either of those makes the tool REFUSE rather than guess.
#
# Anything NOT listed defaults to NO_DOF and is reported as the residue, with its own denominator.
ASSIGN: dict[tuple[str, str, int], tuple[str, str]] = {
    # ---- AT_A_BOUND -------------------------------------------------------------------------
    ('train_tr1_partition_renderer_mlx.py', 'grid_downsample', 0): (AT_A_BOUND, "grid_downsample choices=[8,16], shipped 16 = coarse endpoint; receiver admits 32"),
    ('train_tr1_partition_renderer_mlx.py', 'code_width', 0): (AT_A_BOUND, "code_width choices=[2,4,6]; 3-point sample of an integer continuum, receiver minimum=1"),
    ('ddm_tr1_runtime.py', 'levels > 256', 0): (AT_A_BOUND, "token_quant_levels<=256 general guard; shipped 16 == the SMEVR ceiling"),
    ('ddm_tr1_runtime.py', 'levels > _R7_SMEVR_MAX_LEVELS', 0): (AT_A_BOUND, "SMEVR levels ceiling 16 == the shipped default (the default IS the bound)"),
    ('train_tr1_partition_renderer_mlx.py', 'lane_guard', 1): (AT_A_BOUND, "lane_guard: lambda=0 on 64/64 gates, budget one value all run (bs2)"),
    ('train_tr1_partition_renderer_mlx.py', 'args.verdict_chunk > 120', 0): (AT_A_BOUND, "verdict_chunk>120 hard refuse; score-neutral measurement-side cap"),
    ('ddm_r7_token_coder.py', 'max(lengths) > 15', 0): (AT_A_BOUND, "Huffman max code length 15; inert while huffman_nibble loses the race by +50%"),
    ('ddm_v4d_build_composed_archive.py', 'inherited != vendored', 0): (AT_A_BOUND, "beta-table uint8 ceiling 256; live tables 13 (pw1) / 44 (mq1), not near it"),
    # ---- DIRECTION --------------------------------------------------------------------------
    # NOTE (ddm_na1, step-0 pre-screen): E3/E4 act on pose6[0], a member of degeneracy class A
    # (t = s_t*[p2,p1,p0]).  They stay DIRECTION -- the myopic-probe defect is real and pb2
    # measured it -- but their bound-side reading is INSTANCE-of-search-reach, not axis-level.
    ('ddm_v4d_resolve.py', 'd < best_d', 0): (DIRECTION, "dim0 entry probe: for sign in (1.0,-1.0) ... break; -1 never evaluated if +1 improves [class-A degenerate coordinate]"),
    ('ddm_v4d_resolve.py', 'direction == 0.0', 0): (DIRECTION, "dim0 doubling gated on direction!=0 set by the myopic entry probe [class-A degenerate coordinate]"),
    ('ddm_v4d_resolve.py', 'd < best_d', 1): (DIRECTION, "beta entry probe, identical structure [class-B partial: overall s_r scale absorbable, ratio+sign are not]"),
    ('ddm_v4d_resolve.py', 'direction == 0.0', 1): (DIRECTION, "beta doubling gated on the same myopic entry [class-B partial]"),
    ('ddm_v4d_resolve.py', 'd >= best_d', 0): (DIRECTION, "d>=best_d first-failure stop on a possibly non-unimodal line [class-B partial]"),
    ('ddm_pfs1_ep_warp_pose_solve.py', 'cval < cur', 0): (DIRECTION, "strict-improvement accept: monotone-safe BY CONSTRUCTION; do NOT free"),
    ('ddm_v4c_resolve.py', 'cv < cur', 0): (DIRECTION, "strict-improvement accept; do NOT free"),
    ('ddm_v4c_resolve.py', 'cv < cur', 1): (DIRECTION, "strict-improvement accept; do NOT free"),
    ('ddm_v4c_resolve.py', 'cv < curB', 0): (DIRECTION, "strict-improvement accept; do NOT free"),
    # ---- DEGENERATE -------------------------------------------------------------------------
    ('ddm_pfs1_ep_warp_pose_solve.py', 'ST_GRID', 0): (DEGENERATE, "ST_GRID: exactly multiplicatively degenerate with the shipped translation triple (dc1, 4.539e-16)"),
    ('pfs1_warp_receiver.py', 'ST_GRID', 0): (DEGENERATE, "ST_GRID duplicate copy of A1 -- DOF 1 not 2 (#907 class)"),
    ('ddm_v4d_resolve.py', 'BETA_MAGS', 0): (DEGENERATE, "BETA_MAGS seed; pw1 CURED it, manifest ships 13 entries"),
    ('ddm_v4d_build_composed_archive.py', 'BETA_MAGS', 0): (DEGENERATE, "BETA_MAGS duplicate copy of E1 -- and STALE vs the shipped 13-entry table"),
    ('inflate_runner_v4d.py', 'DEFAULT_BETA_MAGS', 0): (DEGENERATE, "DEFAULT_BETA_MAGS duplicate copy of E1 -- STALE; receiver reads the manifest at :127"),
    # ---- INTERIOR / raced closed --------------------------------------------------------------
    ('inflate_runner_v4d.py', 'sel == 0', 0): (INTERIOR, "selector 376/224 n600 (dc1); both values carry real mass, no blend representable"),
    ('ddm_r7_token_coder.py', 'AUTO_CODECS', 0): (INTERIOR, "AUTO_CODECS: all 9 CODEC_IDS raced byte-exact, smevr wins by 51,546 B (bs2)"),
    ('ddm_r7_token_coder.py', 'CODEC_IDS', 0): (INTERIOR, "CODEC_IDS: same race, closed"),
    ('ddm_v4c_resolve.py', 'RS_GLOBAL_G', 0): (INTERIOR, "RS_GLOBAL_G 415/84/101: saturated SEED, not a binding bound; 22/101 escape downstream (lg2)"),
    ('train_tr1_partition_renderer_mlx.py', '_COMMA10K_LUMA_ANCHORS', 0): (INTERIOR, "comma10k luma anchors: a documented reference span, NOT used as a class-index order (checked negative)"),
    # ---- NO_OCCUPANCY_DATA --------------------------------------------------------------------
    ('train_tr1_partition_renderer_mlx.py', 'token_ste', 0): (NO_DATA, "token_ste {round,dither}: never swept anywhere on this vehicle"),
    ('train_tr1_partition_renderer_mlx.py', 'token_temporal_mode', 0): (NO_DATA, "token_temporal_mode {shared_base,independent}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'token_init_mode', 0): (NO_DATA, "token_init_mode {zero,solve_project}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'adam_bias_correction', 1): (NO_DATA, "adam_bias_correction {off,on}: self-described reset-race ARM SELECTOR, never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'margin_weighted_loss', 0): (NO_DATA, "margin_weighted_loss: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'token_quant_anneal', 0): (NO_DATA, "token_quant_anneal {off,at_knee}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'token_quant_margin_coupling', 0): (NO_DATA, "token_quant_margin_coupling: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'token_delta_group_sparsity', 0): (NO_DATA, "token_delta_group_sparsity: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'delta_sparsity_engage', 0): (NO_DATA, "delta_sparsity_engage: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'delta_sparsity_weight_field', 0): (NO_DATA, "delta_sparsity_weight_field: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'head_range_relax', 0): (NO_DATA, "head_range_relax: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'basin_handoff', 0): (NO_DATA, "basin_handoff: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'boundary_probe', 0): (NO_DATA, "boundary_probe: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'rate_model', 0): (NO_DATA, "rate_model {entropy,smevr_surrogate}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'byte_ledger_coder', 0): (NO_DATA, "byte_ledger_coder {smevr,zlib}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'renderer_head_mode', 0): (NO_DATA, "renderer_head_mode: class_field collapses the head to ONE channel (topology-matched); never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'distill_form', 0): (NO_DATA, "distill_form {kd_logits,margin_field,argmax_ce}: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'seg_form_start', 0): (NO_DATA, "seg_form_start 4-way: never swept as a menu"),
    ('train_tr1_partition_renderer_mlx.py', 'lane_guard_ratchet', 1): (NO_DATA, "lane_guard_ratchet: built by bs2, DEFAULT-OFF, never run in a real trainer"),
    ('train_tr1_partition_renderer_mlx.py', 'seg_spike_reweight', 0): (NO_DATA, "seg_spike_reweight dataclass field: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'lane_guard', 0): (NO_DATA, "lane_guard dataclass field: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'lane_guard_ratchet', 0): (NO_DATA, "lane_guard_ratchet dataclass field: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'adam_bias_correction', 0): (NO_DATA, "adam_bias_correction dataclass field: never fired"),
    ('train_tr1_partition_renderer_mlx.py', 'seg_spike_reweight', 1): (NO_DATA, "seg_spike_reweight argparse: never fired"),
    ('ddm_v4c_resolve.py', 'AB_START_POLICIES', 0): (NO_DATA, "AB_START_POLICIES WELDS how-many restarts to which-restarts (lg2 B10)"),
    ('ddm_v4c_resolve.py', 'two_all', 0): (NO_DATA, "two_all: no receipt"),
    ('ddm_v4c_resolve.py', 'pose_source', 0): (NO_DATA, "pose_source {shiptable,resolve}: no receipt"),
    ('ddm_v4c_resolve.py', 'mode', 0): (NO_DATA, "v4c --mode 3-way: ab_stop census returns {'ABSENT': 600} -- re-run --mode photo populates it"),
    ('ddm_v4d_resolve.py', 'mode', 0): (NO_DATA, "v4d --mode 5-way: ab_trace bound at :372 and referenced NOWHERE (produced-and-discarded)"),
    ('train_tr1_partition_renderer_mlx.py', 'variant', 0): (NO_DATA, "variant {plain,lotto}: architecture-pinned, occupancy undefined at $0"),
    ('train_tr1_partition_renderer_mlx.py', 'mlx_device', 0): (NO_DATA, "mlx_device {gpu,cpu}: score-neutral"),
    ('train_tr1_partition_renderer_mlx.py', 'deterministic_r', 0): (NO_DATA, "deterministic_r: score-neutral"),
    ('train_tr1_partition_renderer_mlx.py', 'full_confirm', 0): (NO_DATA, "full_confirm: score-neutral"),
    ('train_tr1_partition_renderer_mlx.py', 'telemetry_v9_port', 0): (NO_DATA, "telemetry_v9_port: score-neutral (observability, should default ON)"),
    ('ddm_pfs1_ep_warp_pose_solve.py', 'mode', 0): (NO_DATA, "pfs1 --mode {solve,price}: run-control"),
    ('train_tr1_partition_renderer_mlx.py', 'TR1_LOSS_TERM_KEYS', 0): (NO_DATA, "TR1_LOSS_TERM_KEYS {seg,rate,delta_sparsity}: the loss-term set itself"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    inv = json.loads(args.inventory.read_text())
    in_scope = [r for r in inv["rows"] if not r["exclusion"]]

    # rebuild the drift-stable key for every in-scope row
    seen: Counter[tuple[str, str]] = Counter()
    keyed: list[tuple[tuple[str, str, int], dict]] = []
    for r in in_scope:
        base = r["file"].split("/")[-1]
        k2 = (base, r["name"])
        keyed.append(((base, r["name"], seen[k2]), r))
        seen[k2] += 1
    present = {k for k, _ in keyed}

    # FAIL CLOSED, never guess: an assignment that no longer resolves means the source moved
    # under us and the class would be silently mis-attributed.
    unknown = sorted(set(ASSIGN) - present)
    if unknown:
        raise SystemExit(
            "assignment keys no longer resolve against the inventory (source drifted); "
            f"refusing rather than mis-attributing: {unknown}"
        )

    hist: Counter[str] = Counter()
    rows_out = []
    for key, r in keyed:
        cls, why = ASSIGN.get(
            key, (NO_DOF, "fail-closed guard or dispatch arm: hand read found no numeric DOF")
        )
        hist[cls] += 1
        rows_out.append({**r, "triage_key": list(key), "triage_class": cls, "triage_reason": why})

    total = len(in_scope)
    print(f"in-scope denominator: {total}")
    for cls, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")
    print(f"  {'-- explicitly assigned':20s} {sum(hist.values()) - hist[NO_DOF]:4d}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "schema": "ddm_mt1_triage_assignment.v1",
                    "axis": "[macOS-CPU $0] NON-PROMOTABLE",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "pointer_moved": False,
                    "in_scope_denominator": total,
                    "class_histogram": dict(hist),
                    "explicitly_assigned": sum(hist.values()) - hist[NO_DOF],
                    "residue_defaulted_to_NO_DOF": hist[NO_DOF],
                    "rows": rows_out,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
