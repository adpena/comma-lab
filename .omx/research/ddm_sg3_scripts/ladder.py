"""ddm_sg3 granularity ladder: MEASURED real-coder ADDRESS cost per granularity.

LOGIC NOTE (bz1 mirage law): every number here is DESCRIPTION-ONLY -- the bytes to
DESCRIBE a GT-derived set at a given granularity. It is a LOWER BOUND on any counted-GT
artifact at that granularity, because a realizer can only ADD bytes. Therefore:
  address_cost > W  =>  the row is DEAD for every realizer  (bankable KILL)
  address_cost < W  =>  says NOTHING about profitability    (never a win)
"""
import numpy as np, zlib, lzma, json
try:
    import brotli
    HAVE_B = True
except Exception:
    HAVE_B = False

CACHE = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
OUT = "/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3"
NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]
DEN = 37_545_489


def coded(buf_bool):
    b = np.packbits(np.asarray(buf_bool).reshape(-1)).tobytes()
    o = {"zlib": len(zlib.compress(b, 9)),
         "lzma": len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))}
    if HAVE_B:
        o["brotli"] = len(brotli.compress(b, quality=11))
    k = min(o, key=o.get)
    return o[k], k, len(b)


def main():
    gt = np.load(f"{CACHE}/gt_argmax_n600.npy", mmap_mode="r")
    pxa = np.load(f"{CACHE}/cx1_argmax_n600.npy", mmap_mode="r")
    F, H, W = gt.shape
    SLOTS = F * H * W
    Wb = 4 * DEN / SLOTS

    gtv = np.ascontiguousarray(gt)
    pv = np.ascontiguousarray(pxa)
    flip = gtv != pv
    TOT = int(flip.sum())
    print(f"# frames={F} H={H} W={W} slots={SLOTS:,} W={Wb:.6f} B/flip total_flips={TOT:,}")
    print(f"# seg budget at W = {TOT*Wb:,.0f} B ; live archive 353,805 B")

    res = {"meta": {"F": F, "H": H, "W": W, "slots": SLOTS, "W_bytes_per_flip": Wb,
                    "total_flips": TOT, "budget_B": TOT * Wb}}
    rows = []

    def emit(level, name, px_count, flips, byts, coder, note=""):
        bpf = byts / max(flips, 1)
        rows.append(dict(level=level, name=name, px=int(px_count), flips=int(flips),
                         bytes=int(byts), b_per_flip=bpf, vs_W=bpf / Wb, coder=coder, note=note))
        print(f"{level:<12}{name:<30}{int(px_count):>11,}{int(flips):>10,}"
              f"{int(byts):>10,}{bpf:>9.4f}{bpf/Wb:>7.2f}x  {coder}")

    print(f"\n{'level':<12}{'object':<30}{'px in set':>11}{'flips':>10}{'bytes':>10}{'B/flip':>9}{'vs W':>7}  coder")

    # ---- L6 exact per-(frame,pixel) ----
    b, k, raw = coded(flip)
    emit("L6 slot", "exact flip set (all)", flip.sum(), TOT, b, k, "payload-free upper bound")

    # ---- L0 static risk map (2D, one frame) ----
    static = flip.any(axis=0)
    b, k, raw = coded(static)
    emit("L0 static", "static risk map (2D)", static.sum(), TOT, b, k, "addresses 100%, precision 1.94%")

    # ---- L1 class: GT class mask (payload-free, fixes both directions) ----
    for i in range(5):
        m = gtv == i
        fixed = int((flip & ((gtv == i) | (pv == i))).sum())
        b, k, raw = coded(m)
        emit("L1 class", f"GT mask {NAMES[i]}", m.sum(), fixed, b, k, "full-volume mask")

    # ---- L2 interface (undirected pair), full volume ----
    inter = {}
    for i in range(5):
        for j in range(i + 1, 5):
            m = ((gtv == i) & (pv == j)) | ((gtv == j) & (pv == i))
            n = int(m.sum())
            if n < 500:
                continue
            b, k, raw = coded(m)
            inter[(i, j)] = m
            emit("L2 iface", f"{NAMES[i]}<->{NAMES[j]} full-vol", n, n, b, k, "payload-free: pair implies label")

    # ---- L3 interface CROPS (below class -- the marginal-value surface) ----
    from scipy import ndimage
    print()
    for (i, j), m in sorted(inter.items(), key=lambda kv: -int(kv[1].sum())):
        n = int(m.sum())
        crops = []
        nis = 0
        bbox_bits = 0
        sizes = []
        for f in range(F):
            lab, k_ = ndimage.label(m[f])
            if k_ == 0:
                continue
            objs = ndimage.find_objects(lab)
            for idx, sl in enumerate(objs):
                sub = (lab[sl] == idx + 1)
                a = int(sub.sum())
                if a == 0:
                    continue
                nis += 1
                sizes.append(a)
                crops.append(np.packbits(sub.reshape(-1)))
                # bbox: y0,x0 (9+9) + h,w (9+10) + frame delta (~10) = 47 bits
                bbox_bits += 47
        cat = np.concatenate(crops) if crops else np.zeros(0, np.uint8)
        cb = min(len(zlib.compress(cat.tobytes(), 9)),
                 len(lzma.compress(cat.tobytes(), preset=9 | lzma.PRESET_EXTREME)))
        if HAVE_B:
            cb = min(cb, len(brotli.compress(cat.tobytes(), quality=11)))
        tot_b = cb + bbox_bits / 8
        sizes = np.array(sizes)
        emit("L3 crop", f"{NAMES[i]}<->{NAMES[j]} islands", n, n, tot_b, "crop+coded",
             f"{nis:,} islands, median {np.median(sizes):.0f}px, hdr {bbox_bits/8:,.0f}B")
        res[f"islands_{NAMES[i]}_{NAMES[j]}"] = {
            "n_islands": nis, "flips": n,
            "median_px": float(np.median(sizes)), "mean_px": float(sizes.mean()),
            "p90_px": float(np.percentile(sizes, 90)), "max_px": int(sizes.max()),
            "hdr_B": bbox_bits / 8, "mask_B": cb, "total_B": tot_b,
            "top_island_share": float(np.sort(sizes)[::-1][:max(1, nis // 100)].sum() / n),
        }

    res["rows"] = rows
    json.dump(res, open(f"{OUT}/ladder_addr.json", "w"), indent=1)
    np.save(f"{OUT}/static.npy", static)
    print(f"\n# wrote {OUT}/ladder_addr.json")


if __name__ == "__main__":
    main()
