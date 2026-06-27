# Softmax-of-SDF level-set witness + generic curvelet/shearlet front-end — $0 CPU feasibility

UTC 2026-06-27. Axis `[macOS-CPU advisory]` field-level R-survival PROXY; `promotion_eligible=false`,
`score_claim=false`, pointer UNMOVED. Authority for L* = frozen CPU-torch SegNet argmax. SANDBOX
parallel arm (the hosc probe owns the GPU). NO GPU training arm here — build + $0 feasibility only.

## What was built (new module, NO collision with sister lever_b_generator / trainer edits)
- `src/tac/boundary_math/lever_b_levelset_generator.py` — softmax-of-SDF witness:
  K=5 SDF coord-INR (`argmax_k phi_k` = partition; boundary = flat zero-crossing `{phi_i=phi_j}`),
  WIRE(Gabor)/HOSC/relu activation, GENERIC curvelet/shearlet front-end, Eikonal + Chan-Vese
  length regularizers, numpy reference forward (argmax-parity), MLX module (import-guarded),
  int8+brotli byte-close (bank excluded), R-survival metric. numpy-portable.
- `tools/levelset_curvelet_feasibility_smoke.py` — the $0 CPU feasibility (this ledger's source).

## Deep-math (why this is not a PR95/sine reskin)
- SDF is 1-Lipschitz -> the margin field `m = phi_top1 - phi_top2` is ~linear-through-zero, unit
  slope, NO oscillation -> contest R (bicubic up -> uint8 @ camera -> bilinear down) shifts the
  zero-crossing O(blur) MONOTONICALLY, ZERO off-boundary ringing. Cures the Gibbs/R-aliasing wall.
- GENERIC curvelet/shearlet bank (J scales x L_j parabolic orientations + low iso, from 5 scalars)
  = the N^-2-optimal generic basis for curved singularities, regenerated at decode -> FREE in
  inflate.py (rule 118), NO GT leak. Resolves the directional byte-closeability crux (trainer
  bug #5: `directional_fourier_feats` orients to GT SegNet argmax tangent => NOT byte-closeable).

## MEASURED ($0 CPU, n6 real SegNet L* @ 384x512; field-level R-survival proxy)
| rep | pre-R disagree | post-R disagree | off-boundary R-flips |
|---|---|---|---|
| **SDF level-set** | 0.000000 | **1.27e-5** | 0.0 |
| spectral/sine (band-limited indicator) | 0.00745 | 7.46e-3 | 0.0 |
- **SDF post-R is ~587x lower** than spectral (ratio 0.0017). The spectral basis can't even
  represent the sharp real partition (0.0075 disagreement BEFORE R = Gibbs in the rep itself);
  the SDF rep is EXACT pre-R and stays ~0 post-R. Synthetic-cartoon control: spectral off-boundary
  R-flip-frac 0.02 vs SDF 0.0 (Gibbs aliasing is off-boundary; SDF shift is boundary-local).
- Front-end fit (least-squares, EQUAL 40-col budget): curvelet argmax-disagree 0.555 vs isotropic
  0.692 = **0.80x** (curvelet 20% better basis at equal bytes, GT-FREE). Absolute high (linear
  40-feat fit; a trained nonlinear net does far better) — the RATIO is the basis-quality signal.
- Byte-close (n96, hidden128/nh4/mod48, RANDOM init worst-case): 136,453 params ->
  base 109,218 B + code 8,306 B = **117,524 B counted** (bank B free/excluded). Trained int8
  compresses far below random-init.

## Honest limits (NO-FAKE)
- This is the FIELD-LEVEL R-survival proxy, NOT the SegNet-authority realized d_seg. The authority
  number (render SDF partition -> palette RGB -> R -> frozen SegNet argmax) needs the TRAINING arm
  (an untrained flat-palette frame is uninformative to a natural-image SegNet). Feasibility = GO.
- The front-end ratio is a linear-budget proxy; the achievable d_seg comes from the trained net.

## Next (GO): the level-set through-R training arm (future GPU)
Build `experiments/train_levelset_witness_realized_through_R_mlx.py` mirroring the RGB witness
trainer but: head = K-SDF (linear), front_end=curvelet, activation=wire, loss += Eikonal +
length reg, render = softmax(phi/T)@palette -> R -> frozen CPU-torch SegNet d_seg VERDICT.
"""
