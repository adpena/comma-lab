import json, time, sys
from tac.local_acceleration.mlx_scorer_torch_parity import build_mlx_scorer_torch_parity_sweep_manifest

CACHE='experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600'
N=int(sys.argv[2]) if len(sys.argv)>2 else 100
WIN=4
dev=sys.argv[1]
allow=(dev=='gpu')
t0=time.time()
m = build_mlx_scorer_torch_parity_sweep_manifest(
    cache_dir=CACHE, repo_root='.', device_type=dev,
    start_pair=0, max_pairs=N, window_pairs=WIN, stride_pairs=WIN,
    run_id=f'real_{dev}_{N}pairs', allow_gpu_research_signal=allow,
    progress_every=5)
dt=time.time()-t0
s=m['summary']
def dist(k): 
    v=s.get(k,{}); 
    return {'max':v.get('max'),'mean':v.get('mean'),'p95':v.get('p95'),'count':v.get('count')}
out={
    'device': dev, 'pairs': N, 'windows': m['window_count'], 'elapsed_s': round(dt,2),
    'pairs_per_s': round(N/dt,3), 'verdict': m['verdict'], 'passed': m['passed'],
    'seg_argmax_diff_pixels': dist('segnet_argmax_diff_pixels'),
    'seg_argmax_mismatch_pixels_total': s.get('segnet_argmax_mismatch_pixels_total'),
    'seg_argmax_diff_fraction': dist('segnet_argmax_diff_fraction'),
    'seg_mismatch_min_top2_margin': dist('segnet_argmax_mismatch_min_top2_margin'),
    'seg_logit_abs_max': dist('segnet_logit_abs_max'),
    'pose_output_abs_max': dist('posenet_output_abs_max'),
    'pose_component_abs_max': dist('posenet_component_abs_max'),
    'passed_windows': s.get('passed_windows'), 'failed_windows': s.get('failed_windows'),
    'blockers': m['blockers'][:8],
}
print('RESULT_'+dev.upper()+' '+json.dumps(out))
import os
os.makedirs('.omx/tmp/mlx_drift_out', exist_ok=True)
with open(f'.omx/tmp/mlx_drift_out/{dev}_{N}.json','w') as f: json.dump(out,f,indent=2)
