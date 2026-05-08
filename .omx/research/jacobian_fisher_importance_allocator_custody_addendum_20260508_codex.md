# Jacobian/Fisher Importance Allocator Custody Addendum - 2026-05-08

Scope: hardening review for `src/tac/optimization/jacobian_fisher_importance_allocator.py`
and `tools/jacobian_fisher_importance_allocator.py`. No dispatch scripts,
runtime packets, or PR104 artifacts were touched.

## Concrete Gap

The top-level planning manifest already set `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false`. The nested `allocation.metadata` object
did not carry the complete custody contract. A downstream consumer that reads
only `allocation` could therefore miss `score_claim_blockers`,
`score_affecting_payload_changed=false`, `charged_bits_changed=false`, and the
explicit CPU/MPS/proxy non-authority rule.

## Hardening Landed

- Added a shared planning-only metadata block used by both the top-level
  manifest and nested `allocation.metadata`.
- Added a fail-closed score-custody assertion that requires false score/rank/
  promotion/kill flags, `score_authority=none`, disallowed decision uses, and
  exact CUDA archive-custody blockers before returning a manifest.
- Updated the CLI status text so CPU/MPS/proxy allocator evidence is explicitly
  non-score, non-rank, non-promotion, and non-kill evidence.
- Extended tests to cover the top-level manifest, direct allocator plan
  metadata, and CLI-written JSON.

## Evidence Status

Evidence grade remains
`[CPU/MPS/proxy-planning empirical/prediction jacobian-fisher-importance allocator]`.
This allocator output can choose local build order or proposal vectors only.
It cannot promote, rank, kill, dispatch, or claim score movement without a
byte-closed archive, static compliance, active dispatch claim, and full exact
CUDA auth-eval result on the canonical archive path.
