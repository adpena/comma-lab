import time, json, sys
import numpy as np
import mlx.core as mx
from tac.local_acceleration.mlx_scorer_response import _load_upstream_distortion_net, _resolve_upstream_dir, load_scorer_input_cache
from tac.local_acceleration.mlx_scorer_adapters import torch_distortion_net_to_mlx, run_mlx_distortion_scorer_nchw, temporary_mlx_device

CACHE='experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600'
dev=sys.argv[1]; BATCH=int(sys.argv[2]) if len(sys.argv)>2 else 8; ITERS=int(sys.argv[3]) if len(sys.argv)>3 else 10
cache=load_scorer_input_cache(CACHE)
pose=np.asarray(cache.posenet_yuv6_pair[:BATCH],dtype=np.float32)
seg=np.asarray(cache.segnet_last_rgb[:BATCH],dtype=np.float32)
from pathlib import Path
dist=_load_upstream_distortion_net(_resolve_upstream_dir(Path('.').resolve()))
with temporary_mlx_device(dev):
    adapter=torch_distortion_net_to_mlx(dist)
    # warmup
    _=run_mlx_distortion_scorer_nchw(adapter,pose,seg); mx.eval(mx.array([0.0]))
    t0=time.time()
    for _ in range(ITERS):
        out=run_mlx_distortion_scorer_nchw(adapter,pose,seg)
    dt=time.time()-t0
total_pairs=BATCH*ITERS
print('FWD_'+dev.upper()+' '+json.dumps({'device':dev,'batch':BATCH,'iters':ITERS,'total_pairs':total_pairs,'elapsed_s':round(dt,3),'pairs_per_s':round(total_pairs/dt,2),'ms_per_batch':round(1000*dt/ITERS,1)}))
