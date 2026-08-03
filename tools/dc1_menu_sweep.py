#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_dc1 -- sweep EVERY discrete menu in the shipped stack for the ms8 defect.

WHY THIS EXISTS
---------------
``ddm_ms8`` refit ONE menu (``ST_GRID``, the per-pair ``s_t`` scale) and moved
the own-vehicle line 0.9476092 -> 0.8984335 for +51 bytes.  Its measured
discriminator was **NOT** mode share (the RD-optimal codebook also piles 51.5%
on one entry) but the **DEAD-CODEWORD FRACTION**: ST_GRID spent 7 of 11
codewords where the measured mass is EXACTLY zero, covering the live support
with four.  At fixed K the placement debt was -0.041841 S versus -0.007300 from
re-selection alone -- 5.7x.

This tool applies that discriminator to the REST of the stack.

  ``--mode inventory``  $0.  For every discrete menu whose index or value
      reaches the shipped archive or the receiver, measure occupancy on the
      REAL n600 solution and report the dead-codeword fraction.  Reports the
      DENOMINATOR (how many menus were found, how many were priced, how many
      could not be reached and why) because an empty scope is VACUOUS, never a
      PASS.

  ``--mode selcurves``  the per-pair ``selector in {0,1}`` sweep ms8 named as
      owed and left unmeasured ("~1 line of the same harness").  For each pair
      it evaluates the realized ``d_pose`` of the shipped reconstruction over

          s_t     in {shipped, ms8-fitted}
          (sel, pose) in {(0, p_ship), (1, p_ship)}
                        u {(0, p_v4c_single), (1, p_v4c_two)} where v4c solved
                          both branches

      with ``a``, ``b`` and ``beta`` held at the shipped value, through the same
      frozen-CPU-PoseNet path ``inflate_runner_v4d.Decoder.f0`` runs.  One f1
      render per pair, reused for every configuration.

  ``--mode seldesign``  consumes the curves and races selector arms, each
      priced through the REAL shipped encoders (``brotli(packbits(sel), q=11)``
      for the selector stream and ``encode_kl1_field`` for the pose member).

POSITIVE CONTROL (mandatory, ABORTS on failure)
-----------------------------------------------
The shipped configuration ``(s_shipped, sel_shipped, p_ship)`` is itself a
column of the evaluated grid, so the canary is read off the same array as every
arm -- there is no second code path that could agree for the wrong reason.  Its
tolerance is ms8's MEASURED instrument floor on this vehicle (max abs err
1.085e-05, 578/600 exact); ``seldesign`` REFUSES if the canary exceeds it.

AXIS
----
``[macOS-CPU frozen-PoseNet advisory]``, ``score_claim=false``,
``promotion_eligible=false``.  No training, no paid dispatch, no pointer
mutation.  Any composed-S column is a PREDICTION whose fidelity anchor on this
exact vehicle is the QA78 v4d gate residual (1.8e-6) and the pw1 gate residual
(2.5e-6).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_dc1_menu_sweep.v1"

N_PAIRS = 600
ARCHIVE_DENOM = 37_545_489.0

#: ms8's MEASURED instrument floor on this exact vehicle (its §3 canary).  A
#: canary above this means the harness is not reproducing the shipped decode
#: and NO verdict from it is admissible.
CANARY_MAX_ABS_ERR = 1.2e-05

#: The ms8 fitted s_t codebook, so this sweep can ask whether a selector
#: re-selection COMPOSES with the landed ms8 move or is absorbed by it.
MS8_OVERRIDE = Path("/Volumes/VertigoDataTier/pact/ddm_ms8_20260802/"
                    "ms8_st_override.json")

#: ms8's MEASURED archive delta (360,374 - 360,323 B): the widened s_t index
#: stream.  It lives outside the members this tool re-encodes, so any arm that
#: inherits the ms8 codebook must be charged it explicitly or it is under-priced.
MS8_ARCHIVE_DELTA_BYTES = 51

LIVE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
                    "v4d_composed_pw1_archive.zip")
FINAL_JSONL = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/"
                   "final_pw1.jsonl")
V4C_SOLVE = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/"
                 "solve_celldrop50.partial.jsonl")


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * float(d_pose_mean)))


def _load_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["pair"])] = row
    return rows


def dead_fraction(counts) -> tuple[int, int, float]:
    """(dead, K, dead/K) where dead = codewords with EXACTLY zero mass."""
    arr = np.asarray(counts, np.int64)
    k = int(arr.size)
    dead = int((arr == 0).sum())
    return dead, k, (dead / k if k else float("nan"))


# --------------------------------------------------------------------------- #
# mode: inventory
# --------------------------------------------------------------------------- #
def run_inventory(args: argparse.Namespace) -> None:
    """Occupancy + dead-codeword fraction for every reachable discrete menu.

    Every row states HOW its occupancy was obtained.  A menu that cannot be
    measured from the shipped artifacts is recorded as ``unreached`` WITH the
    reason -- it is never silently dropped, because a shrunken denominator is
    the vacuity genus (an empty scope emitting the same symbol as a clean one).
    """
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    from ddm_r7_token_coder import AUTO_CODECS, CODEC_IDS, decode_token_codes

    zf = zipfile.ZipFile(args.archive)
    manifest = json.loads(zf.read("manifest.json"))
    selector = json.loads(zf.read("state/selector.sec"))
    final = _load_jsonl(args.final_jsonl)
    if len(final) != N_PAIRS:
        raise SystemExit(f"final jsonl has {len(final)} rows, expected {N_PAIRS}")

    menus: list[dict] = []

    def add(name, site, kind, counts, *, values=None, note="",
            reached=True, reason="") -> None:
        if reached:
            dead, k, frac = dead_fraction(counts)
            mode_share = float(max(counts)) / float(sum(counts)) if sum(counts) else 0.0
        else:
            dead, k, frac, mode_share = None, (len(values) if values else None), None, None
        menus.append({
            "menu": name, "site": site, "kind": kind,
            "reached": bool(reached), "unreached_reason": reason,
            "K": k,
            "values": None if values is None else [float(v) if isinstance(v, (int, float)) else v
                                                  for v in values],
            "occupancy": None if not reached else [int(c) for c in counts],
            "dead_codewords": dead,
            "dead_fraction": frac,
            "mode_share": mode_share,
            "note": note,
        })

    # 1. s_t codebook -- INDEX coded into state/pose_warp.stp section 2.
    st_grid = manifest["st_grid"]
    st_idx = np.asarray(
        decode_token_codes(_pose_warp_section(zf, 1)), np.int64).reshape(-1)[:N_PAIRS]
    add("st_grid (s_t ground-plane translation scale)",
        "manifest['st_grid'] <- inflate_runner_v4d.py:156; index in "
        "state/pose_warp.stp sec2 (r7 token codes)",
        "index-coded per pair",
        np.bincount(st_idx, minlength=len(st_grid)),
        values=st_grid,
        note="ms8 REFIT this menu: 7/11 dead -> fitted 11/11 live, "
             "dS -0.049177 byte-closed at +51 B.")

    # 2. per-pair selector -- BIT coded into pose_warp.stp section 3.
    sel = np.asarray([int(final[i]["selector"]) for i in range(N_PAIRS)], np.int64)
    add("selector (single-plane vs two-plane frame_0 compose)",
        "inflate_runner_v4d.py:179-183; bits in state/pose_warp.stp sec3 "
        "(brotli(packbits))",
        "index-coded per pair",
        np.bincount(sel, minlength=2),
        values=[0, 1],
        note="A 2-entry menu has NO placement degree of freedom; the only open "
             "question is SELECTION (#882). Measured by --mode selcurves.")

    # 3. rolling-shutter beta magnitude table -- VALUES in the manifest,
    #    INDEX coded into pose_warp.stp section 5.
    # The row's own ``beta_idx`` is a SENTINEL for magnitudes outside the
    # 3-entry seed menu (ddm_v4d_resolve.py:379), so the shipped index comes
    # from the builder's own ``derive_beta_table`` -- the real path.  The
    # manifest's ``beta_idx_counts`` is an INDEPENDENT positive control on it.
    from ddm_v4d_build_composed_archive import derive_beta_table
    beta_mags_derived, beta_idx_u8 = derive_beta_table(final, N_PAIRS)
    beta_mags = manifest["rs_beta_mags"]
    if [float(x) for x in beta_mags_derived] != [float(x) for x in beta_mags]:
        raise SystemExit("derived beta table differs from the shipped manifest")
    beta_idx = np.asarray(beta_idx_u8, np.int64)
    beta_counts = np.bincount(beta_idx, minlength=len(beta_mags))
    if beta_counts.tolist() != list(manifest["beta_idx_counts"]):
        raise SystemExit(
            f"beta occupancy {beta_counts.tolist()} differs from the manifest's "
            f"own beta_idx_counts {manifest['beta_idx_counts']}")
    add("rs_beta_mags (rolling-shutter row-shear magnitude)",
        "manifest['rs_beta_mags'] <- inflate_runner_v4d.py:127; index in "
        "state/pose_warp.stp sec5 (brotli uint8)",
        "value-coded table + index-coded per pair",
        beta_counts, values=beta_mags,
        note="DEAD FRACTION IS 0 BY CONSTRUCTION: "
             "ddm_v4d_build_composed_archive.derive_beta_table builds the table "
             "as sorted(set(chosen magnitudes)), so every entry has >=1 user "
             "and the quantization error is exactly zero. This menu is already "
             "fitted; the ms8 defect cannot occur here.")

    # 4. token quantization lattice -- the 96.2%-of-bytes menu.  The codes ARE
    #    in the shipped archive, so its occupancy is measurable at $0 even
    #    though its DEQUANT values are hardcoded generic.
    codes = np.asarray(decode_token_codes(zf.read("state/tokens.dr7t")))
    levels = int(selector["token_quant_levels"])
    tok_counts = np.bincount(codes.reshape(-1), minlength=levels)
    dq = [float(2.0 * c / (levels - 1) - 1.0) for c in range(levels)]
    add("token_quant_levels (uniform dequant lattice on [-1,1])",
        "ddm_tr1_runtime.decode_token_grid:1224-1227 (HARDCODED uniform "
        "v = 2*code/(L-1) - 1); codes in state/tokens.dr7t",
        "index-coded per token (346,478 B = 96.2% of the archive)",
        tok_counts, values=dq,
        note="The dequant VALUES are a generic uniform lattice hardcoded in the "
             "receiver -- no table ships, so this menu has never been fitted. "
             "Occupancy measured from the SHIPPED codes.")

    # 4b. token-grid CELL constancy.  Not a codeword menu, but the same
    #     question one level up: how many of the 3072 grid cells spend their
    #     per-pair budget on a value that never changes?
    P = codes.shape[0]
    flat = codes.reshape(P, -1)
    n_uniq = np.asarray([np.unique(flat[:, j]).size
                         for j in range(flat.shape[1])], np.int64)
    const_cells = int((n_uniq == 1).sum())

    # 5. r7 SMEVR temporal mode-base -- a per-cell MODAL codeword table that IS
    #    value-coded into the shipped DR7T frame (lzma-packed nibbles).
    mode_base = np.asarray(
        [np.bincount(flat[:, j], minlength=levels).argmax()
         for j in range(flat.shape[1])], np.int64)
    add("r7 SMEVR temporal mode-base (per-cell modal codeword)",
        "ddm_r7_token_coder.py:268 factor_mode_delta; value-coded into the "
        "DR7T base_stream inside state/tokens.dr7t",
        "value-coded table (3072 entries over a 16-symbol alphabet)",
        np.bincount(mode_base, minlength=levels),
        values=list(range(levels)),
        note=f"{const_cells}/{flat.shape[1]} grid cells ({100 * const_cells / flat.shape[1]:.1f}%) "
             "are CONSTANT across all 600 pairs, so their residual is always 0 "
             "and the mode base carries them exactly. This is a property of the "
             "TRAINED token field, not a menu defect.")

    # 6. AUTO_CODECS -- the codec search menu.  Offline: its index is written
    #    into the token frame header, but the choice affects only rate.
    add("AUTO_CODECS (token codec searched by 'auto')",
        "ddm_r7_token_coder.py:55 AUTO_CODECS; chosen id in the DR7T header",
        "offline search menu (1 winner coded)",
        None, values=list(AUTO_CODECS), reached=False,
        reason=f"a search menu, not a per-pair codebook: {len(AUTO_CODECS)} of "
               f"{len(CODEC_IDS)} registered codecs are searched and exactly ONE "
               "id is coded. 'Dead codeword' is not defined for a 1-of-N "
               "offline choice; bs2 measured argmin over 9 == argmin over the 2 "
               "searched.")

    # 6. DEFLATE_MEMBERS -- which zip members are deflated.
    add("DEFLATE_MEMBERS (per-member zip compression choice)",
        "ddm_v4d_build_composed_archive.py:47",
        "offline per-member binary choice",
        None, values=sorted({"manifest.json", "state/selector.sec"}),
        reached=False,
        reason="a per-member binary, not a codebook indexed by data; ms8 "
               "re-tested every member byte-exact at level 9 (only pose_stub.sec "
               "is 6 B better) -- CLOSED as an honest negative.")

    # 7. selector.sec architectural fields -- pinned single-value menus.
    pinned = {k: v for k, v in selector.items()
              if k in {"activation", "arch", "bank_algorithm", "frame_role",
                       "output_activation", "token_encoding", "token_ste",
                       "token_temporal_mode", "schema"}}
    add("selector.sec architectural fields (activation/arch/bank/...)",
        "ddm_tr1_runtime._validate_selector (_SELECTOR_KEYS:110-130)",
        "architectural menus, one value each",
        None, values=sorted(pinned), reached=False,
        reason=f"{len(pinned)} fields, each pinned to a single admissible value "
               "by the validator. Racing them changes the TRAINED model, so "
               "occupancy is undefined at $0 and a refit needs training -- "
               "out of this arm's authority (NO heavy/paid launch).")

    n_total = len(menus)
    n_reached = sum(1 for m in menus if m["reached"])
    flagged = [m for m in menus
               if m["reached"] and m["dead_fraction"] is not None
               and m["dead_fraction"] >= args.dead_threshold]

    receipt = {
        "schema": SCHEMA, "mode": "inventory", "utc": _utc(),
        "axis": "[macOS-CPU $0 static occupancy] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "archive": str(args.archive),
        "n_pairs": N_PAIRS,
        "denominator": {
            "menus_found": n_total,
            "menus_occupancy_measured": n_reached,
            "menus_unreached_with_reason": n_total - n_reached,
            "dead_fraction_threshold": args.dead_threshold,
            "menus_over_threshold": len(flagged),
            # HONESTY: this tool enumerates the menus a per-pair occupancy
            # question is DEFINED for.  An independent repo-wide sweep of the
            # receiver + builder import closure (ddm_dc1, 2026-08-02) found a
            # LARGER candidate set; the difference is architectural /
            # single-value / adaptive-context menus for which "dead codeword"
            # is not a defined quantity.  Recorded so the denominator here is
            # never mistaken for the whole population.
            "repo_wide_sweep_candidate_menus": 36,
            "repo_wide_sweep_archive_or_receiver_reaching": 19,
            "not_enumerated_here": [
                "A6-A9/A11-A19: architectural, single-value, or "
                "seed-regenerated menus (SECTION_CONTRACT, MEMBER_ORDER, "
                "lotto bank alphabet, code_width, grid_downsample, token_ste, "
                "token_temporal_mode, token_codec, pose_stub) -- each has "
                "cardinality but no per-pair occupancy",
                "B12-B15: SMEVR adaptive context/value tables -- decoder-"
                "derived, zero table bytes, so no codeword can be dead",
            ],
        },
        "menus": menus,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "dc1_inventory_receipt.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")

    print(f"=== dc1 menu inventory  [{n_reached}/{n_total} occupancy-measured] ===")
    hdr = f"{'menu':<52}{'K':>4}{'dead':>6}{'dead%':>8}{'mode%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in menus:
        if not m["reached"]:
            print(f"{m['menu'][:52]:<52}{m['K'] or '-'!s:>4}"
                  f"{'  UNREACHED':>6}")
            continue
        print(f"{m['menu'][:52]:<52}{m['K']:>4}{m['dead_codewords']:>6}"
              f"{100 * m['dead_fraction']:>7.1f}%{100 * m['mode_share']:>7.1f}%")
    print(f"\nover dead-fraction threshold {args.dead_threshold:.0%}: "
          f"{[m['menu'] for m in flagged] or 'NONE'}")
    print(f"receipt -> {out}")


def _pose_warp_section(zf: zipfile.ZipFile, index: int) -> bytes:
    """Return section ``index`` (0-based) of ``state/pose_warp.stp``."""
    import struct
    payload = zf.read("state/pose_warp.stp")
    if payload[:8] != b"PFS1WPD1":
        raise SystemExit("pose_warp magic differs")
    off = 12
    secs = []
    for _ in range(5):
        (ln,) = struct.unpack_from("<I", payload, off)
        off += 4
        secs.append(payload[off:off + ln])
        off += ln
    if off != len(payload):
        raise SystemExit("pose_warp has unconsumed bytes")
    return secs[index]


# --------------------------------------------------------------------------- #
# mode: selcurves
# --------------------------------------------------------------------------- #
#: The evaluated configuration grid.  ``pose`` is one of ``ship`` (the pw1
#: refined pose, which is what ships) or ``v4c`` (that BRANCH's own v4c GN
#: pose, available only where v4c solved both branches).  ``st`` is one of
#: ``ship`` or ``ms8`` (the landed fitted codebook), so the composition
#: question is answered by the same sweep rather than assumed.
CONFIG_KEYS: tuple[str, ...] = (
    "st_ship__sel0__pose_ship", "st_ship__sel1__pose_ship",
    "st_ms8__sel0__pose_ship", "st_ms8__sel1__pose_ship",
    "st_ship__sel0__pose_v4c", "st_ship__sel1__pose_v4c",
    "st_ms8__sel0__pose_v4c", "st_ms8__sel1__pose_v4c",
)


def run_selcurves(args: argparse.Namespace) -> None:
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_tv] = "1"
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shipped = _load_jsonl(args.final_jsonl)
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")
    ms8 = json.loads(MS8_OVERRIDE.read_text())
    ms8_grid = [float(x) for x in ms8["st_grid"]]
    ms8_idx = [int(x) for x in ms8["st_idx"]]
    v4c_rows = _load_jsonl(V4C_SOLVE) if V4C_SOLVE.exists() else {}

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """Realized d_pose -- VERBATIM from ``tools/ms8_st_codebook_race.py``'s
        ``run_curves.score``, which mirrors ``inflate_runner_v4d.Decoder.f0``
        and ``ddm_v4d_resolve._beta_select`` exactly.  Copied rather than
        re-derived so the two arms are provably on the same path; the canary
        below is the check that it really is."""
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    seq = [p for p in range(min(args.pairs, N_PAIRS))
           if p % args.nshards == args.shard]
    jl = args.out_dir / f"dc1_selcurves_shard{args.shard}.jsonl"
    cache = {int(json.loads(ln)["pair"]) for ln in
             (jl.read_text().splitlines() if jl.exists() else []) if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[dc1 selcurves] shard {args.shard}/{args.nshards} pairs={len(seq)} "
          f"cached={len(cache)} v4c_rows={len(v4c_rows)}", flush=True)

    done = 0
    for pidx in seq:
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[dc1 selcurves] wall cap at pair {pidx}; rerun to resume",
                  flush=True)
            break
        sh = shipped[pidx]
        p_ship = np.asarray(sh["p"], np.float64)
        a, b = float(sh["a"]), float(sh["b"])
        sel_ship = int(sh["selector"])
        g = float(sh["beta_mag"])
        s_ship = float(v4c._d2_row(pidx)["s_t"])
        s_ms8 = float(ms8_grid[ms8_idx[pidx]])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)

        vr = v4c_rows.get(pidx)
        pose_v4c: dict[int, np.ndarray] = {}
        if vr is not None:
            if "p_single_static" in vr:
                pose_v4c[0] = np.asarray(vr["p_single_static"], np.float64)
            if "p_two_static" in vr:
                pose_v4c[1] = np.asarray(vr["p_two_static"], np.float64)

        vals: dict[str, float | None] = {}
        for st_tag, s in (("st_ship", s_ship), ("st_ms8", s_ms8)):
            for sel in (0, 1):
                vals[f"{st_tag}__sel{sel}__pose_ship"] = score(
                    p_ship, s, sel, f1_u8, f1_f, tp, a, b, g)
                key = f"{st_tag}__sel{sel}__pose_v4c"
                vals[key] = (None if sel not in pose_v4c else
                             score(pose_v4c[sel], s, sel, f1_u8, f1_f, tp, a, b, g))

        # POSITIVE CONTROL: the shipped configuration is a column of this grid.
        ctrl_key = f"st_ship__sel{sel_ship}__pose_ship"
        d_ctrl = float(vals[ctrl_key])
        rec = {
            "pair": int(pidx),
            "sel_shipped": sel_ship,
            "s_shipped": s_ship, "s_ms8": s_ms8,
            "d_shipped_reported": float(sh["d_final"]),
            "d_ctrl": d_ctrl,
            "canary_abs_err": abs(d_ctrl - float(sh["d_final"])),
            "has_v4c_branch_poses": sorted(pose_v4c),
            "p_v4c": {str(k): [float(x) for x in v] for k, v in pose_v4c.items()},
            "vals": {k: (None if vals[k] is None else float(vals[k]))
                     for k in CONFIG_KEYS},
            "beta_mag": g, "a": a, "b": b,
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        done += 1
        if done % 10 == 0 or done <= 3:
            el = time.time() - t0
            print(f"[dc1 s{args.shard} {done}/{len(seq)}] pair {pidx} "
                  f"ctrl {d_ctrl:.6f} canary {rec['canary_abs_err']:.2e} "
                  f"flip {vals[f'st_ship__sel{1 - sel_ship}__pose_ship']:.6f} "
                  f"({el:.0f}s, {el / max(done, 1):.1f}s/pair)", flush=True)
    fj.close()
    print(f"[dc1 selcurves] shard {args.shard} done={done} "
          f"total={len(cache) + done} {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# mode: seldesign
# --------------------------------------------------------------------------- #
def sel_stream_bytes(sel: np.ndarray) -> int:
    """Bytes the REAL shipped encoder spends on this selector bit stream."""
    import brotli
    return len(brotli.compress(
        np.packbits(np.ascontiguousarray(sel, np.uint8)).tobytes(), quality=11))


def pose_member_bytes(poses: np.ndarray, dim0_offset: float | None) -> int:
    """Bytes the REAL shipped encoder spends on this pose member."""
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_v4d_build_composed_archive import encode_kl1_field
    store = np.asarray(poses, np.float64).copy()
    if dim0_offset is not None:
        store[:, 0] = store[:, 0] - float(dim0_offset)
    return len(encode_kl1_field(store.astype(np.float16)))


def run_seldesign(args: argparse.Namespace) -> None:
    rows: dict[int, dict] = {}
    for path in sorted(args.out_dir.glob("dc1_selcurves_shard*.jsonl")):
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[int(r["pair"])] = r
    missing = [i for i in range(N_PAIRS) if i not in rows]
    if missing and not args.allow_partial:
        raise SystemExit(f"selcurves incomplete: {len(rows)}/{N_PAIRS}, "
                         f"first missing {missing[:5]}")
    pairs = sorted(rows)
    n = len(pairs)

    canary = np.asarray([rows[p]["canary_abs_err"] for p in pairs])
    # MANDATORY POSITIVE CONTROL -- ABORTS.  An instrument that does not
    # reproduce the shipped decode cannot adjudicate anything.
    if float(canary.max()) > CANARY_MAX_ABS_ERR:
        raise SystemExit(
            f"POSITIVE CONTROL FAILED: canary max {canary.max():.3e} exceeds "
            f"the measured instrument floor {CANARY_MAX_ABS_ERR:.3e}. NO "
            "verdict from this run is admissible.")

    sel_ship = np.asarray([rows[p]["sel_shipped"] for p in pairs], np.int64)
    d_ctrl = np.asarray([rows[p]["d_ctrl"] for p in pairs])
    base_mean = float(d_ctrl.mean())
    base_contrib = contribution(base_mean)

    def col(key: str) -> np.ndarray:
        return np.asarray([np.nan if rows[p]["vals"][key] is None
                           else rows[p]["vals"][key] for p in pairs])

    grid = {k: col(k) for k in CONFIG_KEYS}
    p_ship = None
    if args.final_jsonl.exists():
        final = _load_jsonl(args.final_jsonl)
        p_ship = np.asarray([final[p]["p"] for p in pairs], np.float64)

    zf = zipfile.ZipFile(args.archive)
    manifest = json.loads(zf.read("manifest.json"))
    dim0 = manifest.get("pose_dim0_offset")
    bytes_sel_ship = sel_stream_bytes(sel_ship)
    bytes_pose_ship = None if p_ship is None else pose_member_bytes(p_ship, dim0)

    arms: list[dict] = []

    def price(label, *, chosen_sel, d, extra_pose=None, extra_bytes=0,
              note="") -> dict:
        d_mean = float(np.mean(d))
        d_contrib = contribution(d_mean)
        b_sel = sel_stream_bytes(chosen_sel)
        # ``extra_bytes`` carries a MEASURED archive delta this arm inherits
        # from a change outside the selector/pose members (e.g. ms8's widened
        # s_t index stream).  Omitting it would under-price the arm.
        d_bytes = b_sel - bytes_sel_ship + int(extra_bytes)
        if extra_pose is not None and bytes_pose_ship is not None:
            d_bytes += pose_member_bytes(extra_pose, dim0) - bytes_pose_ship
        rec = {
            "arm": label,
            "occupancy": np.bincount(chosen_sel, minlength=2).tolist(),
            "pairs_whose_selector_moves": int((chosen_sel != sel_ship).sum()),
            "d_pose_mean": d_mean,
            "pose_contribution": d_contrib,
            "delta_S_pose": d_contrib - base_contrib,
            "selector_stream_bytes": b_sel,
            "delta_bytes_vs_shipped": int(d_bytes),
            "delta_S_rate": 25.0 * d_bytes / ARCHIVE_DENOM,
            "delta_S_total": (d_contrib - base_contrib) + 25.0 * d_bytes / ARCHIVE_DENOM,
            "pairs_that_regress": int((d > d_ctrl + 0.0).sum()),
            "note": note,
        }
        arms.append(rec)
        return rec

    # CTRL -- must reproduce the shipped mean by construction.
    price("CTRL_shipped", chosen_sel=sel_ship, d=d_ctrl,
          note="the shipped configuration, read off the same grid as every arm")

    # ARM 1 -- the #882 move ms8 named as owed: same 2-entry alphabet, the
    # index re-picked per pair against the FINAL composed objective, at the
    # SHIPPED s_t and the SHIPPED (refined) pose.  Nothing else changes.
    s0 = grid["st_ship__sel0__pose_ship"]
    s1 = grid["st_ship__sel1__pose_ship"]
    stack = np.stack([s0, s1])
    pick = np.nanargmin(stack, axis=0).astype(np.int64)
    price("RESEL_sel_at_shipped_st", chosen_sel=pick,
          d=stack[pick, np.arange(n)],
          note="re-select the selector against the final objective, shipped "
               "pose and shipped s_t held. The pose was GN-solved for the "
               "SHIPPED branch, so the flip is evaluated with a mismatched "
               "pose -- an ADVERSE bias; a win here is a floor.")

    # ARM 2 -- the same move ON TOP of the landed ms8 codebook, so the
    # composition question is measured rather than assumed.
    m0 = grid["st_ms8__sel0__pose_ship"]
    m1 = grid["st_ms8__sel1__pose_ship"]
    mstack = np.stack([m0, m1])
    mpick = np.nanargmin(mstack, axis=0).astype(np.int64)
    d_ms8_ctrl = mstack[sel_ship, np.arange(n)]
    price("MS8_only", chosen_sel=sel_ship, d=d_ms8_ctrl,
          extra_bytes=MS8_ARCHIVE_DELTA_BYTES,
          note="the landed ms8 fitted s_t codebook at the SHIPPED selector -- "
               "this arm's reproduction of ms8's own d_pose is a cross-check. "
               f"Priced with ms8's MEASURED +{MS8_ARCHIVE_DELTA_BYTES} B "
               "archive delta (the widened s_t index stream), which lives "
               "OUTSIDE the members this tool re-encodes.")
    price("RESEL_sel_on_ms8_st", chosen_sel=mpick,
          d=mstack[mpick, np.arange(n)],
          extra_bytes=MS8_ARCHIVE_DELTA_BYTES,
          note="selector re-selection stacked on the ms8 codebook")

    # ARM 3 -- the flip evaluated with THAT BRANCH's own v4c GN pose, where
    # v4c solved both.  This changes the pose member too, so it is priced.
    if p_ship is not None:
        for st_tag in ("st_ship", "st_ms8"):
            base_d = (d_ctrl if st_tag == "st_ship" else d_ms8_ctrl)
            cand_sel = sel_ship.copy()
            cand_d = base_d.copy()
            cand_pose = p_ship.copy()
            moved = 0
            for i, p in enumerate(pairs):
                flip = 1 - int(sel_ship[i])
                v = rows[p]["vals"][f"{st_tag}__sel{flip}__pose_v4c"]
                pv = rows[p]["p_v4c"].get(str(flip))
                if v is None or pv is None or not (v < base_d[i]):
                    continue
                cand_sel[i] = flip
                cand_d[i] = v
                cand_pose[i] = np.asarray(pv, np.float64)
                moved += 1
            price(f"BRANCHPOSE_flip_{st_tag}", chosen_sel=cand_sel, d=cand_d,
                  extra_pose=cand_pose,
                  extra_bytes=(MS8_ARCHIVE_DELTA_BYTES
                               if st_tag == "st_ms8" else 0),
                  note=f"{moved} pairs adopt the OTHER branch together with that "
                       "branch's own v4c GN pose (available only where v4c "
                       "solved both). The v4c pose is UN-refined by pw1, so "
                       "this is still adverse-biased.")

    n_v4c = sum(1 for p in pairs if rows[p]["has_v4c_branch_poses"])
    receipt = {
        "schema": SCHEMA, "mode": "seldesign", "utc": _utc(), "n_pairs": n,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "canary_max_abs_err": float(canary.max()),
        "canary_exact_pairs": int((canary == 0.0).sum()),
        "canary_tolerance": CANARY_MAX_ABS_ERR,
        "positive_control": "PASS",
        "ctrl_d_pose_mean": base_mean,
        "ctrl_pose_contribution": base_contrib,
        "shipped_selector_occupancy": np.bincount(sel_ship, minlength=2).tolist(),
        "selector_dead_codewords": int(
            (np.bincount(sel_ship, minlength=2) == 0).sum()),
        "pairs_with_v4c_branch_poses": n_v4c,
        "shipped_selector_stream_bytes": bytes_sel_ship,
        "shipped_pose_member_bytes": bytes_pose_ship,
        "arms": arms,
    }
    out = args.out_dir / "dc1_seldesign_receipt.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n")

    print(f"\n=== dc1 selector race, n={n} "
          f"[macOS-CPU frozen-PoseNet advisory] ===")
    print(f"POSITIVE CONTROL PASS: canary max {canary.max():.3e} "
          f"({int((canary == 0.0).sum())}/{n} EXACT, tol {CANARY_MAX_ABS_ERR:.1e})")
    print(f"CTRL d_pose {base_mean:.8f}  contribution {base_contrib:.6f}")
    print(f"shipped selector occupancy {receipt['shipped_selector_occupancy']} "
          f"dead={receipt['selector_dead_codewords']}/2  "
          f"({n_v4c}/{n} pairs have a v4c counterfactual pose)")
    hdr = (f"{'arm':<30}{'d_pose':>13}{'dS_pose':>11}{'moved':>7}"
           f"{'dBytes':>8}{'dS_rate':>11}{'dS_total':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for a in sorted(arms, key=lambda r: r["delta_S_total"]):
        print(f"{a['arm']:<30}{a['d_pose_mean']:>13.8f}{a['delta_S_pose']:>11.6f}"
              f"{a['pairs_whose_selector_moves']:>7d}"
              f"{a['delta_bytes_vs_shipped']:>8d}{a['delta_S_rate']:>11.6f}"
              f"{a['delta_S_total']:>11.6f}")
    print(f"\nreceipt -> {out}")


# --------------------------------------------------------------------------- #
# mode: degentest -- is ms8's win a QUANTIZER win or a SEARCH win?
# --------------------------------------------------------------------------- #
def build_rescaled_poses(final: dict[int, dict], ms8_doc: dict,
                         s_ship: dict[int, float]):
    """Fold ms8's chosen ``s*`` INTO the pose, keeping the shipped ``s_t``.

    ``pfs1_warp_receiver.pose_to_homography`` uses the pose only as
    ``t = s_t * [p2, p1, p0]`` and ``R = expmap(s_r * [p3, p4, p5])``, so
    scaling ``(p0,p1,p2)`` by ``k`` and ``s_t`` by ``1/k`` leaves the
    homography INVARIANT (ddm_mq1 §2 measured the identity at 5.98e-16 over
    200 pairs x 4 scale factors x 3 rotation scales).  Therefore ms8's fitted
    codebook is reachable at the INCUMBENT ``s_t`` with no menu change at all.

    The one place the identity is NOT exact is STORAGE: ``p1``/``p2`` are plain
    f16 (relative precision, so rescaling is benign) but ``p0`` ships as an f16
    RESIDUAL off a manifest offset (QA65), whose ABSOLUTE quantum depends on the
    residual magnitude.  The offset is re-derived here exactly as the builder
    does (``--dim0-offset auto``), which is one manifest float and free.
    """
    grid = [float(x) for x in ms8_doc["st_grid"]]
    idx = [int(x) for x in ms8_doc["st_idx"]]
    n = len(idx)
    if sorted(final) != list(range(n)):
        raise SystemExit(
            f"the fitted index stream covers {n} pairs but the final JSONL "
            f"has {len(final)} rows keyed {min(final)}..{max(final)}; a "
            "silent truncation here would fold the WRONG scale into the pose")
    if any(not 0 <= j < len(grid) for j in idx):
        raise SystemExit("fitted st_idx escapes its own st_grid")
    poses = np.asarray([final[i]["p"] for i in range(n)], np.float64)
    k = np.asarray([grid[idx[i]] / s_ship[i] for i in range(n)], np.float64)
    resc = poses.copy()
    resc[:, 0:3] = poses[:, 0:3] * k[:, None]
    # the builder's own auto offset, on the RESCALED column
    offset = float(np.float16(resc[:, 0].mean()))
    store = resc.copy()
    store[:, 0] = resc[:, 0] - offset
    q = store.astype(np.float16).astype(np.float64)
    q[:, 0] = q[:, 0] + offset
    return resc, q, k, offset


def run_degentest(args: argparse.Namespace) -> None:
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_tv] = "1"
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = _load_jsonl(args.final_jsonl)
    ms8_doc = json.loads(MS8_OVERRIDE.read_text())
    s_ship = {i: float(v4c._d2_row(i)["s_t"]) for i in range(N_PAIRS)}
    resc, qpose, kfac, offset = build_rescaled_poses(final, ms8_doc, s_ship)

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)

    # ALGEBRAIC LEG -- free, no scorer.  If the two homographies are identical
    # the rendered f0 is bit-identical and d_pose follows BY CONSTRUCTION.
    # Uses the SAME receiver object the scorer path uses, so there is no second
    # implementation that could agree for the wrong reason.
    K, Kinv = oracle.K, oracle.Kinv
    p2h = comp.recv.pose_to_homography
    grid = [float(x) for x in ms8_doc["st_grid"]]
    midx = [int(x) for x in ms8_doc["st_idx"]]
    rel = []
    for i in range(N_PAIRS):
        p = np.asarray(final[i]["p"], np.float64)
        h_ms8 = p2h(p, K, Kinv, float(grid[midx[i]]), 1.0, 0.0)
        h_fold = p2h(resc[i], K, Kinv, s_ship[i], 1.0, 0.0)
        rel.append(float(np.abs(h_ms8 - h_fold).max()
                         / max(np.abs(h_ms8).max(), 1e-300)))
    rel_max = float(max(rel))
    print(f"[dc1 degentest] ALGEBRAIC leg: max relative homography difference "
          f"over {N_PAIRS} pairs = {rel_max:.3e}  "
          f"(exact-degeneracy claim: ddm_mq1 §2)", flush=True)

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    seq = [p for p in range(min(args.pairs, N_PAIRS))
           if p % args.nshards == args.shard]
    jl = args.out_dir / f"dc1_degen_shard{args.shard}.jsonl"
    cache = {int(json.loads(ln)["pair"]) for ln in
             (jl.read_text().splitlines() if jl.exists() else []) if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[dc1 degentest] shard {args.shard}/{args.nshards} pairs={len(seq)} "
          f"cached={len(cache)} dim0_offset={offset}", flush=True)

    done = 0
    for pidx in seq:
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[dc1 degentest] wall cap at pair {pidx}; rerun to resume",
                  flush=True)
            break
        sh = final[pidx]
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        g = float(sh["beta_mag"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        # SHIPPABLE leg: the rescaled pose AFTER the real f16 storage roundtrip,
        # decoded at the INCUMBENT s_t -- zero menu change, zero index widening.
        d_fold_q = score(qpose[pidx], s_ship[pidx], sel, f1_u8, f1_f, tp, a, b, g)
        rec = {
            "pair": int(pidx), "k": float(kfac[pidx]),
            "s_shipped": s_ship[pidx], "s_ms8": float(grid[midx[pidx]]),
            "d_shipped_reported": float(sh["d_final"]),
            "d_fold_quantized_at_shipped_st": d_fold_q,
            "rel_homography_err": rel[pidx],
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        done += 1
        if done % 20 == 0 or done <= 2:
            el = time.time() - t0
            print(f"[dc1 degen s{args.shard} {done}/{len(seq)}] pair {pidx} "
                  f"k {kfac[pidx]:.4f} d_fold {d_fold_q:.6f} "
                  f"({el:.0f}s, {el / max(done, 1):.1f}s/pair)", flush=True)
    fj.close()
    meta = args.out_dir / "dc1_degen_meta.json"
    meta.write_text(json.dumps({
        "schema": SCHEMA, "mode": "degentest", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "dim0_offset_rederived": offset,
        "algebraic_max_rel_homography_err": rel_max,
        "k_min": float(kfac.min()), "k_max": float(kfac.max()),
        "pairs_with_k_ne_1": int((kfac != 1.0).sum()),
        "pose_member_bytes_shipped": pose_member_bytes(
            np.asarray([final[i]["p"] for i in range(N_PAIRS)], np.float64),
            json.loads(zipfile.ZipFile(args.archive).read(
                "manifest.json")).get("pose_dim0_offset")),
        "pose_member_bytes_folded": pose_member_bytes(resc, offset),
    }, indent=1) + "\n")
    print(f"[dc1 degentest] shard {args.shard} done={done} "
          f"{time.time() - t0:.0f}s  meta -> {meta}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode",
                    choices=("inventory", "selcurves", "seldesign", "degentest"),
                    required=True)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--archive", type=Path, default=LIVE_ARCHIVE)
    ap.add_argument("--final-jsonl", type=Path, default=FINAL_JSONL)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_dc1_20260802"))
    # n600 IS the evidence bar (CLAUDE.md "allergic to non-n600-scale"); the
    # default is the full set and a subset must be asked for explicitly.
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dead-threshold", type=float, default=0.10,
                    help="dead-codeword fraction at or above which a menu is "
                         "flagged for refit (pre-registered falsifier: 10%%)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inventory":
        run_inventory(args)
    elif args.mode == "selcurves":
        run_selcurves(args)
    elif args.mode == "degentest":
        run_degentest(args)
    else:
        run_seldesign(args)


if __name__ == "__main__":
    main()
