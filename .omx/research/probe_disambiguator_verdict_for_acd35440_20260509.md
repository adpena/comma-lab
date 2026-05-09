# Probe-disambiguator verdict for in-flight subagent acd35440 (2026-05-09)

<!-- generated_at: 2026-05-09T05:50:00Z, from_state_hash: probe_landed_b2745e55 -->

## Summary

Subagent ad3d7a57 LANDED the $0 T7/T8/T11 sub-additivity disambiguator (commit `b2745e55`). Empirical decision matrix: **all 4 multi-surrogate compositions DEFER**. **Phase 1 trainer wires T8 Sinkhorn-W2 ALONE** as the seg surrogate.

## Implication for acd35440 (T13+T19 trainer integration)

You are in-flight wiring T13 + T19 into `experiments/train_score_gradient_pr101_finetune.py` (or sibling). Per the just-landed probe, your wire-in MUST use **T8 Sinkhorn-W2 alone** as the seg surrogate (NOT T7 alone, NOT T7+T8, NOT T7+T8+T11).

Specifically:
- The trainer's `--seg-surrogate` flag should default to `sinkhorn` (T8), not `fisher_rao` (T7) or `lovasz_hinge` (T11) or any combination
- `tac.losses.segnet_surrogate_per_pixel(..., surrogate="sinkhorn", sinkhorn_blur=0.01, ...)` per a0be36e's T8 implementation (codex HIGH 3 fix already applied)
- Add a CLI flag `--seg-surrogate-multi-experimental` that allows operator opt-in to multi-surrogate stacks (DEFER tag); default OFF

Per CLAUDE.md HNeRV parity discipline lesson 6 (score-domain Lagrangian) and lesson 13 (DEFERRED-not-killed): T7 and T11 stay landed in `tac.losses` + `tac.lovasz_hinge`; the disambiguator did NOT kill them, just deferred multi-composition.

## Operator decision #2: T20 priority ELEVATED

The probe also surfaced: max pairwise cos sim 0.342 ≥ 0.3 means **seg axis is over-attacked**. Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" rule (pose marginal sensitivity 2.71× SegNet's at PR106 frontier), the operator decision is:

**T20 (KL pose-axis loss) should match-or-exceed any new seg-axis lane** until a regime-conditional weighting scheme exists.

This is a directive for sibling subagent a8501386 (T20+T22). They'll see this independently when they read the probe memo.

## Empirical evidence

Cos sim is **REGIME-DEPENDENT**, not constant:

| Regime | T7vsT8 | T7vsT11 | T8vsT11 | Triple |
|---|---:|---:|---:|---:|
| class_mass_swap | +0.7384 | -0.0061 | +0.0694 | +0.2672 |
| interior | +0.0947 | +0.4671 | +0.1537 | +0.2385 |
| near_boundary | +0.2371 | +0.1839 | +0.0528 | +0.1579 |
| sharp_disagreement | -0.0435 | +0.2006 | +0.0025 | +0.0532 |
| soft_disagreement | +0.6844 | +0.7858 | +0.6698 | +0.7133 |

`soft_disagreement` (most common mid-training regime): 0.65-0.78 redundancy. `sharp_disagreement` (late-training): near-zero or negative.

## Codex HIGH 3 status

Codex HIGH 3 fix (Sinkhorn min blur 0.01 + safe iteration floor) is **CONFIRMED CLEAN** by the probe — `class_mass_swap` regime (codex's stress case) returns well-defined non-zero gradients at default blur 0.05. Your T8 wire-in inherits the fix.

## References

- Disambiguator memo: `~/.claude/projects/.../feedback_t7_t8_t11_sub_additivity_disambiguator_landed_20260509.md`
- Tool: `tools/probe_seg_loss_surrogate_disambiguator.py`
- Test: `src/tac/tests/test_probe_seg_loss_surrogate_disambiguator.py`
- T8 implementation (codex HIGH 3 fixed): `src/tac/losses.py:369-397`
