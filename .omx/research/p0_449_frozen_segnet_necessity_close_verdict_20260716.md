# p0_449 — Frozen-SegNet necessity/optimality: CLOSE VERDICT (one page)

**Date:** 2026-07-16 · **Arm:** #515 $0 P0 burn-down · **Authority:** training-loop engineering only
(`score_claim=false`, `promotion_eligible=false`; pointer 0.19108 UNMOVED). This is the CLOSE-OUT
extraction of the already-landed fleet — nothing re-run. Sources (settled, not re-derived):
`.omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md` (memo, 35.6K),
`codex_findings_frozen_segnet_exact_forward_20260713_codex.md` (#456),
`GO_PACKET_p0_backward_k2_inloop_timer_20260713.md`.

## NECESSITY verdict

| surface | necessary | replaceable |
|---|---|---|
| exact verdict / score | real frozen SegNet forward + exact R + exact bytes | **nothing** — `d_seg` IS the frozen-SegNet argmax disagreement; no substitute authority |
| full-teacher training anchor | real forward on current candidate at refresh/verify boundaries + every score claim | refresh FREQUENCY (a gate can re-anchor less often) |
| intermediate training step | a signal correlated with the current scorer decision geometry | full teacher forward AND backward may both be amortized |
| renderer update | `J_x(θ)ᵀ λ` (input costate λ = dL/dx) | HOW λ is obtained (backward is only the present estimator) |

- **Forward every step:** NOT necessary. Periodically re-anchored student / exact local response model /
  teacher-derived costate can supply intermediate signal; source-side argmax is immutable → compute once,
  content-address. Full forward stays mandatory at refresh/verify + every score claim.
- **Backward every step:** NOT necessary. Backward is not in the metric definition; it is the estimator of
  `dL_relaxed/dx`. Replaceable by a faithful input-costate estimate (renderer grad = `J_xᵀ λ`). This is the
  primary structural target (the #426 costate organ / K2 exact-costate reuse).

## OPTIMALITY verdict

Load-bearing kernels already strong: last-frame-only forward is exact; fp32 is the measured fast path;
custom Metal grouped/depthwise backward default-on, measured **16.9×** with witness-parameter gradient
cosine **1.0** (after the historical mismatched-init artifact was corrected). Loop NOT globally optimal:
(1) base loss + surgical raw-margin levers can call SegNet twice on the same composed frame-1 → share the
raw logits, apply the loss-only class offset afterward; (2) witness-alone frame-1 vs temporal frame-0 calls
are semantically distinct → not cacheable as exact duplicates; (3) activation checkpointing is a memory
trade, not a speed win here; (4) no distilled student / #426 costate / #36 Atlas / #141 saliency cache
currently replaces the live per-step frame gradient in this trainer; (5) the naive forward-distilled student
already FAILED descent equivalence despite high argmax agreement → only gradient-aware / on-policy
formulations remain admissible.

## ALTERNATIVES / P0 build outcome

The ELM/INR streaming closed-form affine-SDF-head seed + the fail-closed periodic student/costate/cache
contract are BUILT. ELM seed is consumable by #341 via its `--params` surface; it does NOT solve the
non-affine trunk, FiLM, texture head, palette, or exact argmax objective. Its first pair-0 slice is a scoped
negative: the semantic-head proxy improves but frozen-SegNet disagreement WORSENS before #341 polish (both
before and after R) → seed built but **NOT admitted as a default terminal initializer**.
`verdict_scope: ELM-affine-head-seed formulation, pair-0` — not a family kill.

## 95%-forward-kill wave (#455/#456/#462/#465/#486/#487) — settled

#456 GO for ONE registered formulation: 1-thread frozen Torch-fp32 SegNet forward → matched **2.995×**
speedup, **66.6%** forward-time reduction (**81.1%** anchor-relative). **NEEDS-MORE** for the requested
95% forward-cost kill and for any training/backward claim. `verdict_scope: forward-formulation only`.
Canonical equation `segnet_exact_forward_cpu_thread_control_v1`.

## Remaining sub-item — OPERATOR-GO / launch-blocked (names the fwd-vs-bwd contradiction)

The in-loop component-timer confirmation (`GO_PACKET_p0_backward_k2_inloop_timer_20260713.md`) is
**OPERATOR-GO REQUIRED** — a governed training/timing launch. It settles the fwd-vs-bwd share
CONTRADICTION: the diagnostic harness reported `f=0.1785` forward-share / `82.15%` backward-share, but that
harness ran ~12× the live path and the sparse flagship observed the OPPOSITE component ordering; **neither
ratio has in-loop authority**. The D-A component-timer surface is ONE-GO-READY (163/163 parser/DSL/telemetry
checks), but K2 actuation additionally needs a main-owned hot-file costate-provider patch (exposes/persists/
restores+SHA-verifies the exact input-costate beside each checkpoint) — deferred behind the live run + operator
GO. **This is NOT a $0 item and does not block the close.**

## Disposition

Necessity + optimality + alternatives verdicts are settled and extracted here. Task #449 CLOSED. The only
open thread is the operator-GO in-loop timer + costate-provider patch (launch-blocked, named above).
