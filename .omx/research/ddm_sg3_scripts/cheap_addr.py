"""sg3 part 3 -- CHEAP-ADDRESSING LADDER (operator correction 2026-08-04).

KILL-1's 198,468 B floor was computed on 47-bit ABSOLUTE BBOX headers = a generic/naive basis.
This re-prices the SAME flip sets under receiver-derived addressing bases.

BASIS NOTE (must be read with every number): the receiver cannot run SegNet (strict scorer rule).
The enumeration/boundary objects below are computed here from cx1's SegNet argmax as a STAND-IN
for the receiver's own deterministic class field. That is legitimate for an ADDRESS because an
address only needs ENCODER/DECODER AGREEMENT on a deterministic enumeration, not correctness --
and our task-space generator emits a 5-class field natively. But the exact byte counts inherit
that field's granularity, so every rung below is stamped ASSUMPTION-SCOPED(receiver-field).
Closing it = re-run against the live generator's own class field.
"""
import numpy as np, zlib, lzma, json
from scipy import ndimage
try:
    import brotli; HAVE_B = True
except Exception: HAVE_B = False

CACHE = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
OUT = "/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3"
NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]
DEN = 37_545_489


def cb(b):
    o = {"zlib": len(zlib.compress(b, 9)), "lzma": len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))}
    if HAVE_B: o["brotli"] = len(brotli.compress(b, quality=11))
    k = min(o, key=o.get); return o[k], k


def pk(a): return np.packbits(np.asarray(a).reshape(-1)).tobytes()


gt = np.ascontiguousarray(np.load(f"{CACHE}/gt_argmax_n600.npy", mmap_mode="r"))
pv = np.ascontiguousarray(np.load(f"{CACHE}/cx1_argmax_n600.npy", mmap_mode="r"))
F, H, W = gt.shape; SLOTS = F * H * W; Wb = 4 * DEN / SLOTS
flip = gt != pv; TOT = int(flip.sum())
res = {"meta": dict(F=F, H=H, W=W, slots=SLOTS, W_bytes_per_flip=Wb, total_flips=TOT)}

# ---------- rung (a) DECODER-DERIVED COMPONENT INDEX ----------
print("=== (a) DECODER-DERIVED COMPONENT INDEX: what does naming a component cost? ===")
ncomp = []
for f in range(0, F, 10):
    n_f = 0
    for c in range(5):
        _, k = ndimage.label(pv[f] == c)
        n_f += k
    ncomp.append(n_f)
ncomp = np.array(ncomp)
idx_bits = np.log2(ncomp.mean())
print(f"receiver-field components/frame: mean {ncomp.mean():.0f} median {np.median(ncomp):.0f} max {ncomp.max()}")
print(f"  -> index cost = log2({ncomp.mean():.0f}) = {idx_bits:.2f} bits vs 47-bit absolute bbox "
      f"= {47/idx_bits:.2f}x cheaper PER ADDRESS")
res["rung_a"] = dict(comps_per_frame=float(ncomp.mean()), index_bits=float(idx_bits),
                     vs_bbox_47bit=float(47 / idx_bits))

# ---------- rung (b)+(d) BOUNDARY-BAND RESTRICTED ADDRESSING ----------
print("\n=== (b)+(d) BOUNDARY-BAND ADDRESSING: support shrinks to a receiver-derived band ===")
print("    (band = pixels within k of the cx1 interface -> FREE to the receiver, so only the")
print("     flip/no-flip bits AT band positions are counted)")
st3 = ndimage.generate_binary_structure(2, 2)


def band_for(i, j, k, f):
    a = pv[f] == i; b = pv[f] == j
    return ndimage.binary_dilation(a, st3, k) & ndimage.binary_dilation(b, st3, k)


pairs = [(0, 1, 235148, 198468), (0, 2, 89545, 58936), (0, 3, 57225, 37128),
         (0, 4, 63027, 31658), (2, 3, 61892, 39769)]
print(f"\n{'interface':<18}{'k':>3}{'band px':>12}{'capture':>9}{'bytes':>9}{'B/flip':>9}{'vs W':>7}{'vs bbox':>9}")
for (i, j, nflip, naive_B) in pairs:
    tgt = ((gt == i) & (pv == j)) | ((gt == j) & (pv == i))
    for k in (1, 2, 3):
        bits = []; cap = 0; bpx = 0
        for f in range(F):
            bd = band_for(i, j, k, f)
            v = tgt[f][bd]
            bits.append(np.packbits(v))
            cap += int(v.sum()); bpx += int(bd.sum())
        buf = np.concatenate(bits).tobytes()
        B, coder = cb(buf)
        print(f"{NAMES[i]+'<->'+NAMES[j]:<18}{k:>3}{bpx:>12,}{cap/nflip*100:>8.1f}%{B:>9,}"
              f"{B/max(cap,1):>9.4f}{B/max(cap,1)/Wb:>6.2f}x{naive_B/B:>8.2f}x")
        res[f"band_{NAMES[i]}_{NAMES[j]}_k{k}"] = dict(band_px=bpx, capture=cap / nflip, flips=cap,
                                                       bytes=B, b_per_flip=B / max(cap, 1),
                                                       vs_W=B / max(cap, 1) / Wb, vs_naive=naive_B / B,
                                                       coder=coder)

# ---------- rung (c) TEMPORAL DELTA OF THE BAND SIGNAL ----------
print("\n=== (c) TEMPORAL DELTA of the band-restricted address signal (Road<->Lane, k=2) ===")
i, j, k = 0, 1, 2
tgt = ((gt == i) & (pv == j)) | ((gt == j) & (pv == i))
prev = None; raw = []; dlt = []
for f in range(F):
    bd = band_for(i, j, k, f)
    cur = np.zeros((H, W), bool); cur[bd] = tgt[f][bd]
    raw.append(np.packbits(cur.reshape(-1)))
    dlt.append(np.packbits((cur ^ prev).reshape(-1)) if prev is not None else np.packbits(cur.reshape(-1)))
    prev = cur
Braw, cr = cb(np.concatenate(raw).tobytes())
Bdlt, cd = cb(np.concatenate(dlt).tobytes())
print(f"  full-frame band mask, raw   : {Braw:,} B [{cr}]")
print(f"  full-frame band mask, XOR-dt: {Bdlt:,} B [{cd}]  -> {Braw/Bdlt:.2f}x "
      f"({'delta WINS' if Bdlt < Braw else 'delta LOSES: addresses are NOT temporally redundant'})")
res["rung_c"] = dict(raw_B=Braw, delta_B=Bdlt, ratio=Braw / Bdlt)

json.dump(res, open(f"{OUT}/cheap_addr.json", "w"), indent=1)
print(f"\n# wrote {OUT}/cheap_addr.json")
