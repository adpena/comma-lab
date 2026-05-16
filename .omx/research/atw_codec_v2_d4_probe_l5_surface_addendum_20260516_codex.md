# ATW Codec V2 D4 Probe L5 Surface Addendum

research_only=true
score_claim=false
promotion_eligible=false
rank_or_kill_eligible=false
ready_for_exact_eval_dispatch=false

## Purpose

Commit `628553bcee4d7820c8efaadc43f4ac2b418a3c76` landed the ATW v2 D4
diagnostic verdict after the T3 batched council memo had already recorded
`D4 PROBE NOT YET RUN`. This addendum preserves the old council text as history
while making the current machine-readable L5/operator surface fail closed from
the landed verdict.

## Current verdict

- verdict artifact:
  `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json`
- Markdown verdict:
  `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`
- axis: `[diagnostic-CPU; H(latent|scorer_class) probe]`
- verdict: `INDEPENDENT`
- `I(latent; scorer_class)`: `0.006385502752311645` bits/symbol
- Wyner-Ziv gain ceiling fraction: `0.0009071575012169224`
- Phase-2 status: `defer_measured_a1_latent_class_conditioning_surface`
- recommended variant: `none`
- next action: `do_not_dispatch_atw_v2_phase2_from_this_signal`

This is a deferral of the measured A1-latent/class-conditioning surface, not a
kill of cooperative-receiver methods. Reactivation requires richer side-info
or trained ATW residuals, then paired CPU/CUDA exact-eval custody before any
score, rank, kill, promotion, or submission claim.

## L5/operator surface changes

- `src/tac/optimization/atw_v2_phase2_gate.py` now loads the verdict
  artifact and returns `atw_codec_v2_phase2_gate_status_v1`.
- `tools/operator_briefing.py` now exposes `atw_v2_phase2_gate_status` in the
  `l5_v2_frontier_readiness` payload so the L5-v2 briefing cannot keep treating
  ATW v2 as probe-pending.
- `src/tac/substrates/atw_codec_v2/__init__.py` now exports the D4 verdict
  constants and keeps `RESEARCH_ONLY=true`.
- `src/tac/substrates/atw_codec_v2/registered_substrate.py` points the probe
  hook at `tools/run_atw_v2_d4_probe_from_a1.py` and records the INDEPENDENT
  verdict in the continual-learning rationale.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
| --- | --- | --- |
| Verdict loader | UNIQUE | ATW v2 has a substrate-specific D4 verdict schema and Phase-2 status semantics. Reusing a generic probe reader would hide the `INDEPENDENT -> defer` consequence. |
| Operator surface | ADOPT canonical false-authority flags | The briefing remains planning-only with `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`. |
| Historical council memo | PRESERVE | No rewrite of the older T3 council text. This addendum is the no-signal-loss supersession layer. |
| Reactivation policy | UNIQUE | ATW v2 reactivation is specific to richer scorer side-info or trained ATW residuals, not a generic L5 architecture-lock condition. |

## 9-dimension success checklist evidence

1. **Score movement:** none claimed; diagnostic-only verdict blocks ATW v2
   Phase-2 lift from this signal.
2. **Axis custody:** explicit `[diagnostic-CPU; H(latent|scorer_class) probe]`;
   no contest-CPU/CUDA promotion language.
3. **Archive/runtime custody:** probe references A1 archive and member hashes in
   the verdict JSON; no new contest archive is emitted here.
4. **Operational mechanism:** current measured side-info mechanism is
   independent on A1 latents, so the dispatch gate is closed.
5. **Canonical-vs-unique:** documented above.
6. **Paired evidence:** absent by design; paired CPU/CUDA exact-eval remains a
   reactivation requirement.
7. **No-op/byte consumption:** not a byte-shipping result; the addendum only
   prevents stale dispatch authority.
8. **Cost/performance:** no spend; blocks an otherwise premature $5-50 ATW v2
   Phase-2 lift from a diagnostic signal that returned independent.
9. **Next action:** do not dispatch ATW v2 Phase 2 from this class signal;
   investigate richer side-info surfaces only if they change the L5 build,
   eval, guard, or dispatch decision.

## 6-hook wire-in declaration

1. Sensitivity-map contribution: deferred; current MI is below independence
   tolerance on the measured class-conditioning surface.
2. Pareto constraint: non-binding for this signal; no score/rank reward.
3. Bit-allocator hook: no allocation from class-conditional WZ surface until a
   richer side-info probe returns meaningful conditioning.
4. Cathedral autopilot dispatch hook: L5/operator briefing consumes
   `atw_v2_phase2_gate_status`; dispatch allowed remains false.
5. Continual-learning posterior update: deferred until paired CPU/CUDA exact
   anchor exists.
6. Probe-disambiguator: `tools/run_atw_v2_d4_probe_from_a1.py` is the current
   ATW-specific disambiguator; richer side-info follow-ups must produce a new
   dated artifact.

## Supersession note

Supersedes only the stale actionable implication in
`.omx/research/grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md`
that ATW v2 D4 is not yet run. The historical deliberation remains valid as a
record of the pre-verdict state.
