---
title: DDM V15 grammar-parametrized scorer-solve canonical equations
utc: 2026-07-23T00:06:05Z
tasks: [578, 603, 613]
research_only: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
main_landing_review_required: true
---

# DDM V15 grammar-parametrized scorer solve

Let `g_t` be the inherited v14 grammar state at pair `t`, `M_{r,b}(g_t)` the receiver-visible
support for semantic role `r` and scorer row band `b`, and `z_{r,b} in {0,...,255}^{h x w x 3}`
the counted shared RGB template. The extended receiver is

```text
x_t(z) = Paint_v14(g_t; z),
u_t(z) = R(x_t(z)),
q_t(z) = SegNet(u_t(z)).
```

`Paint_v14` retains the v14 semantic order. A template support subtracts every later role's
support, so the encode-side optimizer and the decoder consume exactly the same visible cells.
The archive contains `z` and grammar bytes only: no scorer weights, logits, gradients, or target
argmax table.

For top-1/top-2 margin `m`, the registered categorical Fisher scalar is

```text
F(m) = tr Fisher = 1/2 sech^2(m/2).
```

Cells are ranked by small winner-rival margin and `F(m)`. The exact bilinear `R` chain is inside
autodiff. A continuous projected step `d` is tested only after the uint8 lattice and a realized
secant:

```text
z'(alpha) = clip_uint8(round(z + alpha d)),  alpha in {1/4,1/2,1}.
```

For target role `r`, define realized target gain `G_r(z')` and harmful collateral

```text
H_r(z') = count{p outside M_r : baseline correct at p and z' makes p wrong}.
```

The hard feasibility gate is `G_r > 0 and H_r = 0`. For counted byte increment `Delta B`, the
reverse-waterfill admission is

```text
Delta S / Delta B >= 25 / 37,545,489 = 6.65858953122e-7 score units/byte.
```

The preregistered n64 projected-gradient/secant search admitted zero steps. The smallest harmful
collateral among otherwise improving proposals was 13 cells for Movable and 23 for Lane. This is a
`FORMULATION x preregistered eight-island development set` negative, not a family verdict. No QP
or joint predictor update was evaluated.

The SHA-bound n600 receiver produced exact camera-byte identity to v14 for all 600 pairs in 38
preserved batches. Therefore the inherited frozen-scorer row is `DERIVED` rather than remeasured:

```text
archive = 133,941 bytes
d_seg = 0.027470296224
d_pose = 163.061327281443
Movable conditional = 0.291615222639
Lane conditional = 0.435195521828.
```

The fork `Movable <= 0.05 and archive <= 160,000` fails. The remaining named formulation is #366
joint predictor/template training under the same receiver and custody constraints. Whitened AR(1)
innovations remain `BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY`.

Executable law anchor:
`src/tac/canonical_equations/ddm_v15_grammar_parametrized_scorer_solve_20260723.py`.

CONSUMED-BY: MAIN review; #366 only after a fresh lane claim. Pointer
`0.1910828242 [contest-CPU]` unchanged.
