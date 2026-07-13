# share_{≥2} linearity gate for the frozen-SegNet costate attack (#449 / YOPO) — 2026-07-12

**Source** `[OPERATOR-SUPPLIED]`: two equations sent as an image (IMG_6984) 2026-07-12, operator asked "Useful?".
Provenance of the equations is UNKNOWN (not attributed to a paper; do NOT claim one). The interpretation below is
`[DERIVED]` from the notation, NOT confirmed — it rests on `a_i, β_i, τ, p, e_k, c` meaning per-layer Jacobian
gains / adjoint / softness-temperature / basis. Reference/design-input only — **pointer 0.18804 UNMOVED**,
score_claim=false.

## The two equations (verbatim)
1. `W_k = c^2 [ Π_i (1 - a_i p)^2 ] e_k`,  hence computable in **O(L) forward passes**.
2. `share_{≥2} = 1 - τ^2 / ( Π_i (1 + β_i) - 1 )`.

## `[DERIVED]` interpretation
- **Eq 1 = the O(L) efficiency claim of YOPO made precise.** The deep input→output sensitivity factorizes as a
  PRODUCT over the L layers (`Π_i (1 - a_i p)` is the chain-rule Jacobian; `p` the adjoint/costate; `a_i` per-layer
  gain). Because it is a product, it is obtainable in O(L) FORWARD passes instead of a full backward — exactly YOPO's
  core: bank the deep costate once, recompute the cheap layer-product between anchors. This is the mechanism yopo_449
  is porting to the frozen SegNet.
- **Eq 2 = the go/no-go gate for linearizing the costate.** `Π_i(1+β_i)-1` is the total all-order interaction gain;
  `τ^2` is the first-order (linear) piece; so `share_{≥2}` is the fraction of the signal in SECOND-ORDER-AND-HIGHER
  (nonlinear, cross-layer) interactions. It is the closed form of the empirical fork the goldmine hunt named: "a
  learned surrogate can be nonlinear ⇒ faithful across the trajectory (unlike a linearization)."

## The gate (what to build/measure — $0, O(L) forwards on the frozen SegNet)
- Measure per-layer `a_i, β_i` off the frozen EfficientNet-B2 SegNet in O(L) forward passes; compute `share_{≥2}(τ)`
  at the witness operating τ.
- **SMALL share_{≥2}** ⇒ the banked LINEAR costate (YOPO) is faithful → YOPO wins, INSTANT low-rank has headroom.
- **LARGE share_{≥2}** ⇒ linearization fails → the attack MUST go nonlinear-surrogate (#449/#428 distilled surrogate);
  INSTANT's low-rank projection has a hard ceiling; YOPO's banked-linear costate goes stale between anchors.
- `share_{≥2}(τ)` predicts AT WHICH τ the linear costate stops being trustworthy (τ = witness softness → the argmax
  boundary's nonlinearity). This turns yopo_449/instant's empirical build into a closed-form go/no-go.

## Routing
Folded into the master #451 OSS-reconciliation pass as the "share_{≥2} linearity gate": measure it on the frozen
SegNet, and let it DECIDE linear-costate-suffices vs nonlinear-surrogate-required BEFORE committing the YOPO/INSTANT
build. Honest caveats travel with it (provenance unknown; `a_i/β_i/τ/p` definitions DERIVED — verify against the
SegNet before the gate is load-bearing). DAG FEED appended. `[no-triality]` reference/routing until measured.
