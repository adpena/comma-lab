# Level-set witness trainer (realized through R) — landed + $0 CPU smoke (make-or-break)

UTC 2026-06-27. Axis: training-gradient `[macOS-MLX]`; VERDICT `[macOS-CPU advisory]` (frozen
CPU-torch SegNet argmax + PoseNet MSE). `promotion_eligible=false`, `score_claim=false`, pointer
UNMOVED. SANDBOX parallel arm (hosc probe owns the GPU). NO GPU training arm — $0 CPU build + smoke.

## What landed
- `experiments/train_levelset_witness_realized_through_R_mlx.py` (NEW; a7660df3's
  `train_witness_realized_through_R_mlx.py` NOT edited — IMPORTED). Composition:
  curvelet/shearlet front-end (GT-free, byte-closeable; from `lever_b_levelset_generator`) ->
  FiLM WIRE/HOSC trunk -> (a) K SDF fields phi (level-set partition) + (b) per-(pair,frame) RGB
  texture; **RGB = sigmoid(softmax(phi/T)@palette + texture)*255** (POSE-LEGAL, not flat palette).
  REALIZED d_seg = frozen CPU-torch SegNet argmax of `render -> _torch_R_to_camera_uint8` vs L*
  (NOT a field proxy). REALIZED d_pose = frozen CPU-torch PoseNet vs stored GT pose. Curriculum
  (ce->tau_softplus->l7), Eikonal + Chan-Vese length reg, EMA, spike-guard, byte-close.
- IMPORTED (no duplication): `make_loss_fn`, `render_*_through_R_mlx`, `_render_rgb_render_res`,
  `_torch_R_to_camera_uint8`, `cpu_verdict_d_{seg,pose}_batch`, `MlxEMA`, `precompute_gt`,
  `quantize_witness_blob`, `implied_score_from_verdict` (from the RGB-witness trainer);
  `self_orientation_directional_feats` (byte-closeable directional, from `lever_b_generator`).

## Pose-legal RGB (the coordinator's make-or-break)
A flat `softmax(phi/T)@palette` frame is pose-blind. Fix: the additive per-(pair,frame) `out_tex`
head restores luma/chroma detail PoseNet's YUV6 needs, while the palette term pins the SegNet
argmax to the 1-Lipschitz SDF partition (so the GO'd -587x R-survival transfers to the realized
SegNet argmax). SegNet reads the SDF-pinned color; PoseNet reads the texture.

## $0 CPU SMOKE — REALIZED d_seg through R + CPU-torch SegNet (the decisive measure)
Pipeline RUNS end-to-end (render -> R -> frozen CPU-torch SegNet/PoseNet), MLX-CPU.
- n2 / 2ep (render 48x64): realized d_seg **0.79 -> 0.51 in ONE epoch**; d_pose 192->180.
- n6 / 15ep (render 64x86, CE-only, wire): realized d_seg 0.528->0.507 (then PLATEAUS);
  d_pose 187.9->165; training loss MONOTONE 277->257->238->215; blob ~125 KB.
- **GO verdict:** (1) realized d_seg (through R + SegNet, NOT proxy) RESPONDS + descends;
  (2) the RGB is POSE-RESPONSIVE — d_pose descends + loss monotone-drops (texture carries pose),
  categorically not a structurally pose-blind flat frame. The ~0.507 d_seg plateau is the
  DOCUMENTED CE plateau (CLAUDE.md PR95 digest: tau_softplus is "THE primary d_seg drop") — broken
  by the curriculum on the n96 GPU run, NOT a representational ceiling. Absolute d_seg/d_pose are
  UNDERTRAINED (15ep n6); NO frontier claim — the frontier descent needs the n96 GPU run + full
  curriculum + convergence. Pose at deploy rides the solved Quantizr stored-pose sidecar.

## n96 GPU launch (full run — for the coordinator/operator to dispatch later)
```
.venv/bin/python -u experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n96_<utc> --num-pairs 96 --epochs 1500 \
  --render-h 192 --render-w 256 --hidden-dim 128 --n-hidden 4 --mod-dim 48 \
  --activation wire --softmax-temp 0.1 --curriculum \
  --tau-softplus-start-epoch 300 --l7-start-epoch 900 \
  --eikonal-weight 0.01 --length-weight 0.001 --w-seg 100 --w-pose 1 --score-domain-loss \
  --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --verdict-pairs 96 \
  --mlx-device gpu --gt-cache <shared n96+ gt cache> --eval-every 25
```
Byte-close: `tools/witness_byte_close_and_eval.py` on `levelset_witness_live_mlx.npz` (curvelet
bank free; weights+code int8+brotli counted) -> exact-eval row.
"""
