# Retroactive sweep for Wave 3 DreamerV3 RSSM math-fidelity audit

Per Catalog #348 4-field contract. Wave 3 of the 12-wave 15-item math-fidelity
audit cascade landed a code-side fix (1% unimix per Hafner 2023 §3) + first
empirical anchor on canonical equation #344
(`categorical_posterior_capacity_vs_continuous_gaussian_v1`). This wave did
NOT introduce a new STRICT preflight gate; the retroactive sweep below
documents the empirical-falsification scope across prior verdicts that the
fix may have invalidated.

## Field 1 — Bug-class symptom signature

The pre-Wave-3 substrate `tac.substrates.dreamer_v3_rssm` claimed in module
docstrings to be the "canonical Hafner 2024 recipe" while the implementation
omitted the canonical Hafner 2023 §3 1% unimix robustness mixture. The
resulting Gumbel-Softmax categorical posterior could collapse to a hard
one-hot under any pathologically peaked logits, breaking gradient flow through
the straight-through estimator. The omission was CARGO-CULTED relative to
Hafner 2023 canonical per the Catalog #303 cargo-cult audit framework.

Detection pattern (post-fix, the bug class is structurally extinct via the
default `unimix_alpha=0.01`):

- Module-level: `from tac.substrates.dreamer_v3_rssm import gumbel_softmax_sample` AND no `apply_unimix_to_logits` call before Gumbel perturbation
- Config-level: `DreamerV3RSSMConfig` without `unimix_alpha` field AND default ≠ 0.01

## Field 2 — Pre-fix window

The pre-fix scope is narrow because the substrate is a single-package L0
SCAFFOLD landed 2026-05-26 (commit per existing landing memo
`council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519`).
The CARGO-CULTED unimix omission existed for ~3 days (2026-05-26 → 2026-05-29).
No paid Modal/Lightning/Vast.ai dispatch fired during this window (substrate
remained at L0 with `dispatch_enabled=false`).

## Field 3 — Historical KILL/DEFER/FALSIFY search results

Search performed across:

- `~/.claude/projects/-Users-adpena-Projects-pact/memory/` for `dreamer` /
  `rssm` / `categorical_posterior` filename matches
- `.omx/research/` for `dreamer` / `rssm` / `categorical_posterior` filename
  matches
- `.omx/state/probe_outcomes.jsonl` for substrate `dreamer_v3_rssm` blocking
  verdicts

Results:

- **0 KILL verdicts** found for the substrate (substrate is L0 SCAFFOLD;
  symposium verdict is PROCEED_WITH_REVISIONS not KILL per Catalog #307
  paradigm-vs-implementation classification).
- **0 DEFER verdicts** found via the canonical probe-outcomes ledger for
  this substrate's `unimix` axis (the unimix axis was never probed pre-fix).
- **1 prior canonical equation derivation** at
  `.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md`
  registered the canonical equation with 0 anchors; Wave 3 lands the first
  empirical anchor (closes the 0-anchor state per Catalog #371 trigger
  `when_3+_new_empirical_anchors_in_domain` — first of 3 needed for
  auto-recalibration).

## Field 4 — Per-finding RE-EVAL-priority assignment

| Finding | RE-EVAL priority | Reason |
|---|---|---|
| Canonical equation #344 first empirical anchor (closed-form mixture identity) | LOW | Closed-form math identity; residual = 0.0; not a contest score claim per Catalog #287/#323 |
| Sister wave Path B2 PyTorch port + Modal smoke (trained-logits anchor) | MEDIUM | Sister wave deliverable per symposium op-routable #3; produces the second of 3 anchors needed for Catalog #371 auto-recalibration |
| (G, K) sweep probe (K-capacity vs G-groups disambiguator) | MEDIUM | Per symposium op-routable #1a; produces the third anchor + canonical disambiguator per Catalog #125 hook #6 |
| Substrate L1+ extension with full RSSM GRU + dynamics prior | LOW | Future work; re-evaluation of KL balancing + free bits classifications per Catalog #303 will be a sister audit wave's deliverable |
| Wave 4 sister Z7-Mamba-2 audit | LOW (orthogonal) | Sister wave 4 of 12 audit cascade; disjoint substrate scope per Catalog #340 sister-checkpoint guard |

No historical KILL or DEFER verdicts are invalidated by this audit (substrate
was at L0 SCAFFOLD with no paid dispatches fired). The audit's primary
deliverable is canonical Hafner 2023 §3 fidelity restoration + first
empirical anchor registration.
