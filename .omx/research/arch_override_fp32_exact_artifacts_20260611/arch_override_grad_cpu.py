import json, sys, numpy as np, torch
sys.path.insert(0,"src/tac/mlx_pr95_port/tests")
from test_mlx_gpu_score_bridge import _build_real_setup, _cos
from tac.mlx_pr95_port.mlx_gpu_score_bridge import MLXGpuScorerBridge
from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
net, seg_t, pose_t, render, idx_t = _build_real_setup(n_pairs=8)
tb=TorchScorerBridge(net,seg_t,pose_t,seg_loss_form="ce_seg_loss",seg_weight=100.0,pose_weight=1.0,eval_roundtrip=True)
tr=tb.loss_and_pixel_grad(render,idx_t); tg=np.asarray(tr.pixel_cotangent,dtype=np.float64)
cb=MLXGpuScorerBridge(net,seg_t,pose_t,seg_loss_form="ce_seg_loss",seg_weight=100.0,pose_weight=1.0,eval_roundtrip=True,device_type="cpu")
cr=cb.loss_and_pixel_grad(render,idx_t); cg=np.asarray(cr.pixel_cotangent,dtype=np.float64)
d=cg-tg
print("CPU_GRAD_RESULT "+json.dumps({"grad_cosine":float(_cos(cg,tg)),"grad_rel_l2":float(np.linalg.norm(d)/(np.linalg.norm(tg)+1e-12)),"loss_rel_err":abs(cr.loss_value-tr.loss_value)/(abs(tr.loss_value)+1e-9),"seg_loss_abs_delta":abs(cr.seg_loss_value-tr.seg_loss_value)}))
