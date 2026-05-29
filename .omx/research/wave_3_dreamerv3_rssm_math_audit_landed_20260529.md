---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hafner, Schmidhuber, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The 1% unimix fix landed this wave is canonical 1:1 with Hafner 2023 §3 and not an over-fit to the dashcam video problem space"
    classification: HARD-EARNED
    rationale: "Hafner 2023 §3 explicitly states the unimix is the canonical robustness mechanism enabling fixed hyperparameters across diverse domains; the dashcam substrate's structural need to preserve STE gradient flow during training is the same need"
  - assumption: "The remaining 4 N/A classifications (symlog / KL balancing / percentile return / symexp twohot) are HARD-EARNED problem-space adaptations, not CARGO-CULTED omissions"
    classification: HARD-EARNED
    rationale: "Each of the 4 N/A elements has a structurally-specific RL world-model rationale in Hafner 2023 that does not apply to per-pair video compression at L0; rationales documented per-element in the per-substrate symposium memo §1"
council_decisions_recorded:
  - "wave-3 op-routable #1: 1% unimix fix landed (Wave 3 deliverable) — CARGO-CULTED omission → HARD-EARNED canonical 1:1"
  - "wave-3 op-routable #2: first empirical anchor on canonical equation categorical_posterior_capacity_vs_continuous_gaussian_v1 landed (Catalog #344 surface)"
  - "wave-3 op-routable #3: 16 new math-fidelity tests + 11 existing tests = 27/27 PASS in 0.69s"
  - "wave-3 op-routable #4: per-substrate symposium memo landed per Catalog #325"
  - "wave-3 op-routable #5: Path B2 PyTorch port + Modal smoke deferred to sister wave (this audit's anchor is closed-form math identity; sister wave's anchor is trained-logits empirical)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519
  - dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z
  - dreamerv3_rssm_per_substrate_symposium_wave_3_20260529
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
catalog_344_compliance: anchor_appended_per_update_equation_with_empirical_anchor
canonical_equation_ids:
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - categorical_blahut_arimoto_rate_distortion_v1
substrate_id: dreamer_v3_rssm
lane_id: lane_wave_3_dreamerv3_rssm_math_audit_20260529
---

# Wave 3 DreamerV3 RSSM math-fidelity audit — landing memo

Per Wave 3 of the 12-wave 15-item math-fidelity audit cascade (operator
blanket approval 2026-05-29). This wave covers Item 7 of 15: DreamerV3
RSSM categorical posterior math audit vs Hafner et al. 2023.

## Headline

The DreamerV3 RSSM L0 SCAFFOLD substrate at `tac.substrates.dreamer_v3_rssm`
audited against Hafner et al. 2023 arXiv:2301.04104 "Mastering Diverse
Domains through World Models". Result: 1 CARGO-CULTED omission fixed
(1% unimix robustness mixture) + 8 elements classified across the 5-axis
documented-adaptation taxonomy + first empirical anchor registered on
canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1`
per Catalog #344.

## Wave 3 deliverables landed

1. **Code fix**: `unimix_alpha` config field + `apply_unimix_to_logits` helper
   + `gumbel_softmax_sample` threads the canonical Hafner 2023 §3 1% unimix
   mixture before Gumbel perturbation. Default `unimix_alpha=0.01` matches
   Hafner 2023 canonical exactly.
2. **Test suite**: 16 new math-fidelity tests in
   `src/tac/substrates/dreamer_v3_rssm/tests/test_hafner_2023_math_fidelity.py`
   covering the unimix mixture identity, STE invariants under unimix, both
   Hafner 32x32 and C6 24x256 G/K configs, ablation surface (`unimix_alpha=0`),
   and the closed-form mixture identity at fp32 precision (atol=1e-6).
3. **Apparatus mutation**: first empirical anchor registered on canonical
   equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` per
   Catalog #344 + first empirical anchor closes the prior 0-anchor state.
4. **Per-substrate symposium memo** at
   `.omx/research/dreamerv3_rssm_per_substrate_symposium_wave_3_20260529.md`
   per Catalog #325 6-step contract.
5. **Lane registry** L0→L1 (impl_complete + memory_entry).

## Per-element classification matrix

Per Slot EEE 6-axis methodology + Catalog #303 cargo-cult audit + the 5-axis
documented-adaptation taxonomy (optimization-to-contest / problem-space /
math / data / video):

| Hafner 2023 element | Our impl | Classification | Adaptation axis |
|---|---|---|---|
| Straight-through gradient estimator | `use_straight_through=True` default | HARD-EARNED CANONICAL 1:1 | n/a |
| Gumbel-Softmax categorical sampling | `gumbel_softmax_sample()` | HARD-EARNED CANONICAL 1:1 | n/a |
| 1% unimix on all categoricals | `unimix_alpha=0.01` default (Wave 3 fix) | HARD-EARNED CANONICAL 1:1 (post-fix) | n/a |
| 32x32 vs 24x256 (G x K) | parameterizable; both validated | DOCUMENTED ADAPTATION | problem-space (RL vs video compression) |
| RSSM = GRU deterministic + categorical stochastic | NO GRU at L0 (per symposium) | DOCUMENTED ADAPTATION | problem-space (per-pair contest scorer) |
| symlog observation squashing | n/a | DOCUMENTED N/A | problem-space (video [0,255] native via sigmoid * 255) |
| KL balancing + free bits | n/a | DOCUMENTED N/A | problem-space (no prior/posterior split at L0) |
| Percentile return normalization | n/a | DOCUMENTED N/A | problem-space (no RL reward) |
| symexp twohot loss for reward/critic | n/a | DOCUMENTED N/A | problem-space (no value/critic heads) |

The single CARGO-CULTED finding (1% unimix) is now fixed; all 8 other elements
are either canonical 1:1 or HARD-EARNED documented adaptations per the
problem-space (video compression vs RL world model).

## Mathematical fidelity verification

The Wave 3 audit produced the first empirical anchor on canonical equation
`categorical_posterior_capacity_vs_continuous_gaussian_v1`. The anchor is
the closed-form mixture identity verified to fp32 precision:

- H(T) = G * log2(K) entropy capacity identity (Shannon Cover & Thomas Theorem 2.6.4)
- C6 Path B2: G=24, K=256, H = 192 bits/sample (verified)
- Hafner canonical: G=32, K=32, H = 160 bits/sample (verified)
- Unimix mixture at α=0.01, K=256: peak prob = 0.9900390625, floor prob = 0.0000390625 (verified atol=1e-6)
- Residual: 0.0 (closed-form math identity)

Sister anchors (sister wave deliverable):

- Path B2 PyTorch port + Modal smoke produces the trained-logits anchor.
- (G, K) sweep probe per symposium op-routable #1a produces the
  K-capacity-vs-G-groups disambiguator anchor.

## Test summary

```
27 passed in 0.69s
```

- 11 existing baseline tests (preserved; verified pass with unimix wired in)
- 16 new math-fidelity tests (this wave's deliverable):
  - 6 unimix math fidelity (closed-form mixture identity)
  - 2 straight-through estimator invariants under unimix
  - 2 documented adaptation classifications (G/K parameterization; no GRU/critic at L0)
  - 2 canonical equation first empirical anchor (formula identity verification)
  - 2 end-to-end training forward (with + without unimix ablation)
  - 2 configuration + manifest validation

## Documented adaptation rationales

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Catalog #287
non-placeholder-rationale discipline, each of the 4 N/A classifications
carries a substantive HARD-EARNED rationale documented in the per-substrate
symposium memo §1:

- **symlog observation squashing**: Hafner 2023 symlog handles diverse RL
  domains' value scales (Atari pixels vs proprioceptive floats). Video
  frames are bound to [0, 255] uint8 natively + the sigmoid * 255 RGB head
  produces values in the same range; symlog is structurally redundant.
- **KL balancing + free bits**: Hafner 2023 KL balance + free bits
  regularize the prior-posterior gap in the RSSM dynamics network. At L0
  we have no separate prior network (per-pair learned logits ARE the
  posterior with no temporal prior); when GRU + dynamics prior is added at
  L1+ this classification must be re-evaluated per Catalog #303.
- **Percentile return normalization**: no RL reward signal; video
  compression uses contest scorer loss directly (per Catalog #164 canonical
  helper routing).
- **symexp twohot loss for reward/critic**: no value/critic heads at L0;
  video compression has rgb_0 + rgb_1 heads via sigmoid * 255 instead.

## Cross-references

- Hafner et al. 2023 arXiv:2301.04104 "Mastering Diverse Domains through World Models" §3 "Robustness"
- Jang et al. 2016 arXiv:1611.01144 "Categorical Reparameterization with Gumbel-Softmax"
- Maddison et al. 2016 arXiv:1611.00712 "The Concrete Distribution"
- Cover & Thomas 2nd ed. Theorem 2.6.4 (entropy capacity)
- Reference impl https://github.com/danijar/dreamerv3
- T3 grand-council symposium 2026-05-19:
  `.omx/research/council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519.md`
- Canonical equation derivation 2026-05-20:
  `.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md`
- Wave 3 per-substrate symposium 2026-05-29:
  `.omx/research/dreamerv3_rssm_per_substrate_symposium_wave_3_20260529.md`

## 6-hook wire-in per Catalog #125

- **hook #1 sensitivity-map**: per-axis decomposition deferred to Path B2 sister wave
- **hook #2 Pareto constraint**: canonical equation #1 (capacity vs continuous Gaussian) feeds Dykstra Pareto solver per Catalog #372
- **hook #3 bit-allocator**: archive grammar declares per-pair int8 packing; bit budget locked at H = G * log2(K) bits/sample per Catalog #344
- **hook #4 cathedral autopilot dispatch**: canonical equation lookup consumer auto-discovers updated anchor count per Catalog #335
- **hook #5 continual-learning posterior**: first empirical anchor registered via `update_equation_with_empirical_anchor` per Catalog #344 + #371 auto-recalibrator threshold (3+ anchors) tracked
- **hook #6 probe-disambiguator**: (G, K) sweep probe queued per symposium op-routable #1a; canonical disambiguator between K-capacity vs G-groups hypotheses

## Mission contribution

`apparatus_maintenance`: this audit closes the prior 0-empirical-anchor state
on canonical equation #344 + lands the canonical Hafner 2023 1% unimix fix
+ produces the canonical fidelity anchor. The empirical contest score impact
is deferred to the sister Path B2 PyTorch port + Modal smoke wave (this
audit's anchor is the closed-form math identity; the trained-logits anchor
is the sister wave's deliverable). The structural foundation is now in place
for future paid Modal dispatches to be credibly grounded in Hafner-canonical
math fidelity.
