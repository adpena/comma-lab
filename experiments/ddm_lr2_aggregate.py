# SPDX-License-Identifier: MIT
"""ddm_lr2 aggregator -- per-rung pooled eta, ACTUAL carrier bytes, the rung's OWN bar, verdict.

Reads the ladder receipts (transport / response / solve / solve0) and emits one table where
every rung is priced against ITS OWN recomputed bar (never bz1's 0.426 reused):

    rate_total(n600) = offsets_bytes + extra_payload_bytes + pose_stream_bytes   [x RATE/BYTE]
    net_dS           = rate_total - eta_pooled x gross(n600)                     [<0 = BANKS]

Subset->n600 projections are labelled: per-pair payloads are extrapolated x(600/n_pairs) and
the flips-representativeness of the pair subset is printed with every projection (m96/m88).

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=False.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RATE_PER_BYTE = 25.0 / 37_545_489.0
S_PER_FLIP = 100.0 / (600 * 384 * 512)
GROSS_N600_S = 0.1803885565863715      # et1 re-solved block16 label ceiling, our field
OFFSET_LZMA1_N600 = 57_809             # bz1-measured LZMA1, label-solved n600 offset field
POSE_STREAM_N600 = 57_600              # 96 B/pair k=4 frame_0 repair stream (bz1 G1/G2)
POP_MEAN_FLIPS = 508_640 / 600.0
LIVE_BEST_S = 0.7910689

RECEIPTS = Path("/Volumes/VertigoDataTier/pact/ddm_lr2_20260804")


def pooled(rows, get_fa, get_nd=lambda r: r["n_described"]):
    fb = sum(r["flips_before"] for r in rows)
    nd = sum(get_nd(r) for r in rows)
    fa = sum(get_fa(r) for r in rows)
    return (fb - fa) / nd if nd else float("nan"), fb, fa, nd


def verdict_row(name, eta, offsets_b, extra_b, note="", pose_b=POSE_STREAM_N600, dpx=None):
    total_b = offsets_b + extra_b + pose_b
    rate = total_b * RATE_PER_BYTE
    bar = rate / GROSS_N600_S
    net = rate - eta * GROSS_N600_S
    return {"rung": name, "eta_pooled": round(eta, 4),
            "carrier_bytes_n600": int(total_b),
            "rate_S": round(rate, 6), "own_bar_eta": round(bar, 4),
            "net_dS": round(net, 6),
            "d_pose_ratio_subset_median": (round(dpx, 2) if dpx is not None else None),
            "verdict": "BANKS" if net < 0 else "LOSES", "note": note}


def med_dpx(rows, get):
    v = sorted(get(r)["d_pose_ratio_vs_shipped"] for r in rows)
    return v[len(v) // 2]


def main() -> int:
    n_out = []
    j = lambda f: json.load(open(RECEIPTS / f))  # noqa: E731

    tr = j("lr2_transport_n8.json")["rows"]
    n_pairs = len(tr)
    scale = 600.0 / n_pairs
    subset_mean_flips = sum(r["flips_before"] for r in tr) / n_pairs
    print(f"pairs n={n_pairs}; subset mean flips/pair {subset_mean_flips:.1f} vs population "
          f"{POP_MEAN_FLIPS:.1f} -> representativeness {subset_mean_flips/POP_MEAN_FLIPS:.4f}")
    print(f"gross(n600) {GROSS_N600_S:.6f} S | offsets LZMA1 {OFFSET_LZMA1_N600} B | "
          f"pose stream {POSE_STREAM_N600} B | live best {LIVE_BEST_S}\n")

    # naive baseline (bz1, for the table)
    n_out.append(verdict_row("bz1 naive camera translate [bz1 receipt]", 0.1192,
                             OFFSET_LZMA1_N600, 0, "instance-scoped negative, inherited"))

    e1, *_ = pooled(tr, lambda r: r["A1_token_resample"]["flips_after"])
    e2, *_ = pooled(tr, lambda r: r["A2_native_translate"]["flips_after"])
    n_out.append(verdict_row("A1 token-resample + re-render", e1, OFFSET_LZMA1_N600, 0,
                             dpx=med_dpx(tr, lambda r: r["A1_token_resample"])))
    n_out.append(verdict_row("A2 pre-R native translate", e2, OFFSET_LZMA1_N600, 0,
                             dpx=med_dpx(tr, lambda r: r["A2_native_translate"])))

    try:
        rp = j("lr2_response_n8.json")["rows"]
        for s in ("tok", "rgb"):
            es, *_ = pooled(rp, lambda r, s=s: r[f"{s}_composite"]["flips_after"],
                            lambda r: r["n_described_label"])
            lab_b = sum(r["offsets_label_lzma1"] for r in rp)
            rsp_b = sum(r[f"{s}_offsets_response_lzma1"] for r in rp)
            # ratio-price the response field against the label field's joint-n600 price
            off_b = OFFSET_LZMA1_N600 * (rsp_b / lab_b)
            n_out.append(verdict_row(f"A3-{s} response-solved offsets", es, off_b, 0,
                                     f"offsets ratio-priced {rsp_b}/{lab_b} of label",
                                     dpx=med_dpx(rp, lambda r, s=s: r[f"{s}_composite"])))
            bound = sum(r[f"{s}_blockwise_bound_flips_fixed"] for r in rp)
            ndl = sum(r["n_described_label"] for r in rp)
            print(f"A3-{s}: blockwise interference-free bound eta {bound/ndl:.4f} "
                  f"(realized composite {es:.4f})")
    except FileNotFoundError:
        print("response receipt missing")

    try:
        sv = j("lr2_solve_n8.json")["rows"]
        ea2, *_ = pooled(sv, lambda r: r["A2_transport"]["flips_after"])
        for M in ("M16", "M32"):
            eC, *_ = pooled(sv, lambda r, M=M: r["C_arms"][M]["flips_after"])
            pb = sum(r["C_arms"][M]["payload_lzma1"] for r in sv) * scale
            n_out.append(verdict_row(f"C transport + per-block shifts {M}", eC,
                                     OFFSET_LZMA1_N600, pb,
                                     f"{pb/600:.0f} B/pair params",
                                     dpx=med_dpx(sv, lambda r, M=M: r["C_arms"][M])))
        mall = [k for k in sv[0]["C_arms"] if k not in ("M16", "M32")][0]
        eC, *_ = pooled(sv, lambda r: [v for k, v in r["C_arms"].items()
                                       if k not in ("M16", "M32")][0]["flips_after"])
        pb = sum([v for k, v in r["C_arms"].items() if k not in ("M16", "M32")][0]["payload_lzma1"]
                 for r in sv) * scale
        n_out.append(verdict_row(f"C transport + shifts ALL({mall})", eC, OFFSET_LZMA1_N600, pb))
        for K in ("K32", "K64", "K128", "K256"):
            eB, *_ = pooled(sv, lambda r, K=K: r["B_arms"][K]["flips_after"])
            pb = sum(r["B_arms"][K]["payload_lzma1"] for r in sv) * scale
            n_out.append(verdict_row(f"B transport + sparse residual {K}", eB,
                                     OFFSET_LZMA1_N600, pb,
                                     "selector=|delta| (INSTRUMENT-limited)"))
        eD, *_ = pooled(sv, lambda r: r["B_dense_band"]["flips_after"])
        print(f"B dense-band eta (unpayable ceiling of this solve): {eD:.4f}")
    except FileNotFoundError:
        print("solve receipt missing")

    try:
        s0 = j("lr2_solve0_n8.json")["rows"]
        keys = sorted(s0[0]["arms"], key=lambda k: (k.endswith("_null"),
                                                    int(k.split("_")[0][1:])))
        for K in keys:
            e0, *_ = pooled(s0, lambda r, K=K: r["arms"][K]["flips_after"])
            pb = sum(r["arms"][K]["payload_lzma1"] for r in s0) * scale
            dpx = med_dpx(s0, lambda r, K=K: r["arms"][K])
            null = K.endswith("_null")
            n_out.append(verdict_row(
                f"C0 NO-offset block paint {K}", e0, 0, pb,
                f"{pb/600:.0f} B/pair"
                + ("; POSE-NULL actuator, pose stream EXCLUDED" if null else ""),
                pose_b=(0 if null else POSE_STREAM_N600), dpx=dpx))
    except FileNotFoundError:
        print("solve0 receipt missing")

    for fo1_file in ("lr2_fo1_m32_n8.json", "lr2_fo1_m64_n8.json"):
        try:
            f1 = j(fo1_file)["rows"]
        except FileNotFoundError:
            continue
        cells = sorted(f1[0]["cells"])
        for cell in cells:
            M = int(cell.split("_")[0][1:])
            arm = cell.split("_")[1]
            addr_b = 2 * M                        # ONE static block list, shipped once
            pose_b = POSE_STREAM_N600 if arm == "U" else 0
            n_cap = sum(1 for r in f1 if r["cells"][cell]["stop_reason"] == "cap")
            for depth in ("int8", "step2", "step4"):
                eF, *_ = pooled(f1, lambda r, cell=cell, depth=depth:
                                r["cells"][cell]["depths"][depth]["flips_after"])
                pb = sum(r["cells"][cell]["depths"][depth]["params_lzma1"] for r in f1) * scale
                dpx = med_dpx(f1, lambda r, cell=cell, depth=depth:
                              r["cells"][cell]["depths"][depth])
                n_out.append(verdict_row(
                    f"FO1 static {cell} {depth}", eF, addr_b, pb,
                    f"{pb/600:.0f} B/pair; {n_cap}/8 cap-hit"
                    + ("; NO pose stream (AC pose-null)" if arm == "AC" else ""),
                    pose_b=pose_b, dpx=dpx))

    try:
        ky = j("lr2_keys_n8.json")["rows"]
        for key, addr_b, label in (
                ("gt", 64 * 600, "per-pair top-M indices (64 B/pair)"),
                ("static", 64, "ONE static block list (~64 B total)"),
                ("proxy", 0, "0-B decoder-derived edge-energy rank")):
            ek, *_ = pooled(ky, lambda r, key=key: r["arms"][key]["flips_after"])
            pb = sum(r["arms"][key]["params_lzma1"] for r in ky) * scale
            cap = sum(r["arms"][key]["capture_of_pair_flips"] for r in ky) / len(ky)
            n_out.append(verdict_row(
                f"C0-keys M32 addr={key}", ek, addr_b, pb,
                f"{label}; capture {cap:.2f}",
                dpx=med_dpx(ky, lambda r, key=key: r["arms"][key])))
    except FileNotFoundError:
        print("keys receipt missing")

    print(f"\n{'rung':44s} {'eta':>8s} {'dpx~':>7s} {'bytes':>9s} {'rate S':>9s} {'bar':>7s} "
          f"{'net dS':>9s} verdict")
    for r in n_out:
        dpx = r.get("d_pose_ratio_subset_median")
        print(f"{r['rung']:44s} {r['eta_pooled']:8.4f} "
              f"{(f'{dpx:7.1f}' if dpx is not None else '      -')} "
              f"{r['carrier_bytes_n600']:9d} "
              f"{r['rate_S']:9.6f} {r['own_bar_eta']:7.4f} {r['net_dS']:+9.6f} "
              f"{r['verdict']}  {r['note']}")
    out = RECEIPTS / "lr2_ladder_table.json"
    json.dump({"schema": "ddm_lr2_ladder_table.v1",
               "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
               "score_claim": False,
               "denominators": {"gross_n600_S": GROSS_N600_S, "rate_per_byte": RATE_PER_BYTE,
                                "offsets_lzma1_n600": OFFSET_LZMA1_N600,
                                "pose_stream_n600": POSE_STREAM_N600,
                                "live_best_S": LIVE_BEST_S},
               "rows": n_out}, open(out, "w"), indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
