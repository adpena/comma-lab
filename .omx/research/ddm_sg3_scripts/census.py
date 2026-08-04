import numpy as np, json, sys
CACHE="/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
gt = np.load(f"{CACHE}/gt_argmax_n600.npy", mmap_mode='r')
px = np.load(f"{CACHE}/cx1_argmax_n600.npy", mmap_mode='r')
print("shapes", gt.shape, px.shape, gt.dtype, file=sys.stderr)
F,H,W = gt.shape
PX = F*H*W
DEN=37_545_489
Wb = 4*DEN/PX

NAMES=["Road","Lane","Undrivable","Movable","MyCar"]
conf = np.zeros((5,5), dtype=np.int64)      # [gt, pred]
per_frame = np.zeros(F, dtype=np.int64)
# per-pixel flip count across frames (static structure)
static_count = np.zeros((H,W), dtype=np.int32)
# per-frame per-edge
per_frame_edge = np.zeros((F,5,5), dtype=np.int32)

for f in range(F):
    g = np.asarray(gt[f]); p = np.asarray(px[f])
    d = g!=p
    per_frame[f] = d.sum()
    static_count += d
    gi = g[d].astype(np.int64); pi = p[d].astype(np.int64)
    np.add.at(conf, (gi,pi), 1)
    c2 = np.zeros((5,5),dtype=np.int64); np.add.at(c2,(gi,pi),1)
    per_frame_edge[f] = c2
    if f%100==0: print("f",f, file=sys.stderr)

total = int(per_frame.sum())
print(json.dumps({
 "frames":F,"H":H,"W":W,"PX":PX,"W_bytes_per_flip":Wb,
 "total_flips":total,
 "d_seg":total/PX, "seg_leg_S":100*total/PX,
}, indent=1))
np.save("/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3/static_count.npy", static_count)
np.save("/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3/per_frame.npy", per_frame)
np.save("/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3/per_frame_edge.npy", per_frame_edge)
np.save("/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3/conf.npy", conf)
