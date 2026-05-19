# Codex Findings - Z7-Mamba-2 Blocker Burndown - 2026-05-19

session_actor: codex
date_utc: 2026-05-19T13:37:15Z
score_claim: false
promotion_eligible: false
dispatch_fired: false
research_only: true

## Scope

Operator directive: burn down all remaining Z7-Mamba blockers.

This pass reviewed the Z7-Mamba-2 trainer, archive grammar, generated inflate
runtime, same-byte static-control probe, exact-eval handoff verifier, remote
driver, operator-authorize recipe, and focused tests. Two xhigh review agents
performed independent read-only adversarial passes; their local implementation
findings were converted into fail-closed code paths and regression tests.

## Blockers Burned Down

1. Exact-eval handoff false authority.
   - `tools/verify_z7_exact_eval_handoff.py` now rejects stats that set
     `ready_for_exact_eval_dispatch=true`.
   - Static-control rows also fail closed if they explicitly claim exact-eval
     readiness.

2. LSTM/GRU identity bleed into Mamba handoff and probe outputs.
   - Handoff identity is now derived from the stats packet and known substrate
     pair-group contracts.
   - The Mamba path emits `lane_z7_as_mamba_2_full_landing_20260518`,
     `time_traveler_l5_z7_mamba2`, and
     `z7_mamba2_temporal_coherence_vs_static_capacity_same_bytes`.
   - Dispatch command labels now use the actual substrate identity instead of
     hard-coded LSTM strings.

3. Probe axis-label fragility.
   - The recurrent-vs-static disambiguator now normalizes `contest_cuda`,
     `contest-CUDA`, and `[contest-CUDA]` while preserving the raw label.

4. Misleading `mamba_ssm` stateful evidence.
   - Stateful `mamba_ssm` now fails closed until a true incremental-state or
     sequence-replay implementation lands.
   - `reference_torch` remains the canonical byte-faithful runtime path.

5. Generated runtime portability.
   - Generated `inflate.sh` uses `${PYTHON:-python3}` rather than bare
     `python`.

6. Predictor-byte consumption.
   - A recurrent archive test now mutates only predictor weights and asserts the
     inflated raw output changes.

7. Remote-driver false-authority guard.
   - Remote completion guard now checks `ready_for_paid_dispatch` in addition to
     score, promotion, and exact-eval readiness fields.
   - Full remote mode refuses to run while the operator-authorize recipe remains
     `research_only: true` or `dispatch_enabled: false`.

8. Stale smoke lane identity.
   - Smoke stats now use the active full-landing lane id and retain the older
     scaffold lane only as historical metadata.

9. Parsed archive config geometry.
   - `parse_archive()` now reconstructs decoder geometry from metadata so
     downstream consumers do not silently see default config geometry.

10. Canonical-scale stability controls.
    - Trainer and remote driver now expose and record `--lr-warmup-steps` and
      `--grad-clip-norm`.
    - Training records per-epoch learning rate and pre-clip gradient norm when
      clipping is active.

## Verification

- Focused Z7-Mamba test bundle:
  `83 passed, 12 warnings`.
- Ruff on all touched Python surfaces:
  `All checks passed`.
- Shell/runtime hygiene:
  `bash -n scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh` and
  `git diff --check` clean.
- Operator authorize dry run:
  refused as expected because the recipe is still research-only, dispatch is
  disabled, and evidence gates remain.
- Tiny real-video CPU packet:
  `experiments/results/z7_mamba2_codex_tiny_e2e_hardened_20260519T133511Z`.
  It exercised the new warmup/clipping fields, generated inflate runtime, static
  same-byte control, archive parsing, and handoff verification. It made no score
  claim.
- Mamba-specific disambiguator artifact:
  `.omx/research/probe_z7_mamba2_temporal_coherence_vs_static_capacity_disambiguator_20260519_codex.json`.

## Remaining Gates

No known local implementation blocker remains in the reviewed Z7-Mamba-2
control path. The remaining blockers are evidence gates:

- Z7-GRU Wave 2 disambiguator outcome.
- Wave N+1 council after Z7-GRU outcome.
- C6 IBPS Phase 2 beta anchor.
- Paired exact-eval JSON for the Mamba recurrent-vs-static probe.
- Same-archive-byte identity disambiguator evidence.
- Reference-torch runtime exact handoff on a ratified packet.

These gates should stay fail-closed until exact same-runtime packet evidence
lands. This pass does not authorize paid dispatch or promotion.


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
