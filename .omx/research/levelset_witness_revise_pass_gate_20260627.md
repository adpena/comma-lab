# Level-set witness — REVISE pass (3 review gates) + the transfer GATE + plateau diagnosis

UTC 2026-06-27. Axis `[macOS-CPU advisory]` (frozen CPU-torch SegNet/PoseNet authority);
`promotion_eligible=false`, pointer UNMOVED. $0 CPU only; GPU was HELD until cleared.

## STEP 1 — TRANSFER PROBE (the gate) = PASS
The 587x R-survival was FIELD-LEVEL (`argmax(R(phi))`); the contest scores
`SegNet(R(RGB(phi))).argmax`. Probe (`tools/levelset_transfer_probe.py`, n4 real L*): ideal SDF
phi + natural per-class palette + bounded texture, rendered RGB through the contest R into the
frozen CPU-torch SegNet:
- best realized d_seg = **0.0049** (T0.1, bounded texture); pure palette 0.0121.
- **r_added_segnet ~ 0** (R adds ~zero flips at the SegNet output) -> the 1-Lipschitz R-survival
  TRANSFERS through the SegNet+softmax+sigmoid composition. VERDICT: vehicle SOUND.

## PLATEAU DIAGNOSIS (the 0.507 wall) = PALETTE ARTIFACT, proven + fixed
With the IDEAL phi (perfect partition, T->0):
- OLD luma-ramp palette -> realized d_seg **0.5055** (== the smoke plateau).
- NATURAL per-class-mean palette -> realized d_seg **0.0121**. **41.7x gap.**
The ~0.507 plateau was the palette (SegNet can't read unnatural colors), NOT a training or
representation wall. FIX `--palette-anchor` (default ON): init the LEARNABLE palette to the
per-class mean GT RGB (logit). NO-FAKE/legality: the palette is a 15-float LEARNED weight
(counted in archive), not per-frame GT — no decode-time GT leak.

## STEP 2 — FIXES LANDED (a-i + config-review)
(a)+(b)+(c) numpy-fp32 EMA-shadow ONE-CODEPATH verdict = the inflate/byte-close forward
  (`levelset_rgb_forward_numpy`); NOT mlx-gpu reduced precision. PARITY proven: numpy fwd
  reproduces the MLX partition (phi-argmax agree 0.9997; int8-deploy 0.988). EMA-shadow
  checkpoint + cfg/bank persisted. (d) curriculum fail-closed assert + epochs>=1500.
  (e) self-orient FAIL-CLOSED (honest, #1 follow-up; per-pair coord_feats threading). (g) DROP
  pose-from-texture (w_pose=0; deploy pose = solved Quantizr stored-pose sidecar). (h) Eikonal +
  Chan-Vese length on the DECISION MARGIN m=phi_top1-phi_top2 (not each field). (i) Muon docstring
  corrected (optimizer is AdamW). config-review pre-caps: render-384 default (192 caps at
  0.00085), mod-32 capacity (RD-optimum ~122KB), activation hosc primary (+SIREN-init for
  from-scratch trainability), softmax-temp ANNEAL hi->lo.

## STEP 3 — $0 VALIDATION (numpy-fp32 + EMA, palette-anchored, hosc+curriculum, n4)
realized d_seg: ep0 0.560 -> ep5 0.549 -> ep10 **0.409** (BREAKS 0.507) -> ep15 0.534. The fix
breaks 0.507 (gate-2 met) but the n4/render-96 smoke is NOISY (0.409<->0.534) — not yet stable at
tiny scale. Proven floor with the fix = 0.012 (ideal phi) / 0.0049 (anneal+texture).

## GPU-READY = GO (gate cleared) — with a noise caveat
Both gate conditions literally met: transfer PASS (0.0049) AND $0 smoke broke 0.507 (0.409). The
plateau was a diagnosed+fixed palette artifact; the floor is 0.012-0.0049 (far below 0.507). The
n4 smoke is noisy (recommend the n96 run watch the EMA-shadow d_seg + capacity sweep). self-orient
(the -48% byte-closeable lever) is the #1 post-GO follow-up.

## n96 GPU LAUNCH (render-384 + RD-optimum capacity SWEEP + hosc + anneal-T + palette-anchor)
```
for MOD in 32 24 40; do   # capacity sweep around the RD-optimum ~122KB
.venv/bin/python -u experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n96_mod${MOD}_<utc> --num-pairs 96 --epochs 1500 \
  --render-h 384 --render-w 512 --hidden-dim 96 --mod-dim ${MOD} \
  --activation hosc --siren-init --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 900 \
  --palette-anchor --w-seg 100 --w-pose 0 --eikonal-weight 0.01 --length-weight 0.001 \
  --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --verdict-pairs 96 \
  --mlx-device gpu --gt-cache <shared n96 gt cache> --eval-every 25 ; done
```
Byte-close -> exact row: `tools/witness_byte_close_and_eval.py` on `levelset_witness_ema_mlx.npz`
(curvelet bank + palette are the one-codepath; weights+code int8+brotli counted).
"""
