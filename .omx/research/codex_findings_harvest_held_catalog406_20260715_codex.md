# Codex findings — held harness + Catalog #406 harvest review

**Mode:** local CPU review/audit only; no launch, GPU, provider, score, pointer, or run-directory mutation.

## Harvested sources

- Harness source A: four commits cherry-picked as `164604b70e`, `f48148142b`, `b060204e21`, and `3c58f2d49d`.
- DSL-hash source B: verified bundle SHA-256 `3125680a5216d4cdf0a37a5ef1096e8c506ba3c4554e2d9efb97ba8ebf8bed6e`; commit `2681d40b9820f30de2fd45f1049f25bd6e009f07` cherry-picked as `3c3f6a07b2`.

## Independent review verdicts

### A — PASS within requested verdict scope

- `STRAND_DOOMED` is limited to a live write-capable legacy/non-isolated arm: the predicate excludes read-only and requires `isolate is not True`; bucket selection applies it only while `alive` (`tools/codex_status.py:88-111,202-224`).
- Drain classification is evidence-bearing and fail-closed: empty current set is `DRAINED`; every survivor needs recent log or an advancing cursor and cannot be strand-doomed for `TIMED_OUT`; any other survivor is `WEDGED`; exit codes are exactly 0/2/3 (`tools/codex_drain_detector.py:120-161,194-220`).
- Optional `.last.txt` and stage-NPZ globs are lexically inspected outside quoted/comment text and require `null_glob`, `nullglob`, `(N)`, or a non-expanding `find(1)` path (`tools/check_dispatch_cli_shell_hazards.py:245-324`).
- Retry custody accepts only the exact delegation key, `in_progress`, positive integer step, and nonempty next action; absent custody refuses rc 20 (`tools/codex_retry_checkpoint.py:19-58`). Retry count is capped at two and the resume prompt carries only the external authority digest/key (`tools/codex_delegate.py:214-225,260-288,362-396`).

**Verdict scope:** requested classification, exit-code, empty-glob, checkpoint-key, and retry-cap invariants. No claim about unrelated harness behavior.

### B — PASS

- Hash volatility excludes only run identity and LawRef observation/resolution/staleness timestamps while binding the canonical typed spec, argv, #332 manifest/hash, and semantic LawRef provenance (`src/tac/v9_provenance_gates.py:45-60,154-218,334-345`).
- Verification reopens exact artifacts, reconstructs `TypedWitnessConfig -> WitnessProgram`, validates it, recompiles argv/constants/LawRefs, independently rebuilds the bijection, and compares the rebuilt document/hash (`src/tac/v9_provenance_gates.py:389-560`).
- Launcher compile/artifact verification precedes governed dispatch; trainer admission refuses a missing/mismatched binding rc 8 (`tools/launch_witness_run.py:543-624,1825-1893`; `src/tac/admission_guard.py:82-188`).
- Governor rejects raw or renamed witness entrypoints, invokes exact artifact verification first in `_do_start`, and reaches `Popen` only afterward (`tools/spawn_durable_daemon.py:525-612,812-829,892-908`). Catalog #406 statically rechecks those callsites and lexical order (`src/tac/preflight.py:86408-86566`).

**NO-FAKE verdict:** this is an active launch + governor + trainer gate, not a marker. Runtime enforcement is fail-closed even while the orchestrator meta-gate remains WARN-only.

## Catalog #406 strict-flip assessment — BLOCKED

Operator-requested probe: `check_config_flag_provenance_bijection_complete(strict=True, verbose=True)`.

Current live count is **3,884**, not zero. Therefore `check_launch_and_governor_require_dsl_compile_hash(strict=False, ...)` is intentionally **not** flipped. The #406 structural checker itself has zero live violations, but the stricter operator prerequisite (#332 complete-chain closure) is not satisfied.

### Exhaustive residual identity map

The 3,884 named residuals are exactly the union of the per-factory coverage-summary rows and `(factory, flag, missing-edge)` tuples below. Lists are exhaustive; the three ideal/core factories have byte-identical residual sets and are grouped without losing identity.

- `v9_cgauge_432`: 899 = 3 coverage summaries + 145 owner + 199 LawRef + 193 compiler-missing + 6 compiler/LawRef mismatch + 154 rung + 199 receipt.
- Each of `v9_cgauge_truly_optimal_core`, `v9_cgauge_ideal_mod19`, and `v9_cgauge_ideal_mod32`: 995 = 3 coverage summaries + 141 owner + 224 LawRef + 218 compiler-missing + 6 compiler/LawRef mismatch + 179 rung + 224 receipt.
- The three coverage summaries for every factory are: `LawRef coverage mismatch`, `compiler-record coverage mismatch`, and `provenance-table coverage mismatch`. The latter also names stale provenance key `schedule`.

#### `v9_cgauge_432`

- missing-or-nonunique Lever owner (145 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--anneal-epochs`, `--annulus-band`
  - `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--async-verdict`, `--cache-gt-skeleton`, `--chroma`
  - `--ckpt-every`, `--cldice-iters`, `--containment-damp`, `--containment-mode`, `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`
  - `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--eikonal-weight`
  - `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hidden-dim`
  - `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`, `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`
  - `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`
  - `--lane-band-start-epoch`, `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`
  - `--lane-render-band`, `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`
  - `--max-bank-freq`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`
  - `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`
  - `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`
  - `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`
  - `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seed-island-eased`, `--seed-islands`
  - `--seed-lr`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`
  - `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`
  - `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-live-mq`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`
  - `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`
  - `--witness-alone-island-loss`

- missing Lever.constant_refs LawRef (199 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--containment-damp`, `--containment-mode`
  - `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`
  - `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--eikonal-weight`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`
  - `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`
  - `--island-dilate-px`, `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`
  - `--jacobian-basin-telemetry`, `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`
  - `--ladder-lane-r0`, `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`
  - `--ladder-release-coeff`, `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`
  - `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`
  - `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`, `--max-bank-freq`
  - `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`
  - `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`
  - `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`
  - `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`
  - `--render-h`, `--render-w`, `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`
  - `--seed-island-eased`, `--seed-islands`, `--seed-lr`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`
  - `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-phase-advect-start-epoch`, `--seg-phase-advect-weight`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`
  - `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`, `--stage-checkpoints`
  - `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`, `--structured-init-sdf-clip`
  - `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-live-mq`, `--tail-lr-prop-tau`
  - `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`
  - `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- missing canonical compiler record (193 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--containment-damp`, `--containment-mode`
  - `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`
  - `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--eikonal-weight`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`
  - `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-omega`, `--island-dilate-px`
  - `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`
  - `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`
  - `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`
  - `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`, `--lane-band-start-event`
  - `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`, `--length-weight`
  - `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-end`, `--max-bank-freq`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`
  - `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`
  - `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`
  - `--polyak-finisher-arm`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`
  - `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`, `--safe-compile-regions`, `--score-domain-loss`
  - `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seed-island-eased`, `--seed-islands`, `--seed-lr`, `--seg-chroma-boundary-margin-band`
  - `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`
  - `--seg-phase-advect-weight`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`
  - `--softmax-temp-start`, `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`
  - `--structured-init-lr`, `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`
  - `--tail-live-mq`, `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`
  - `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`
  - `--witness-alone-island-loss`

- compiler record mismatches owning LawRef (6 flags):
  - `--hosc-beta-end`, `--lr-anneal-epochs`, `--lr-hold-frac`, `--polyak-finisher-start-epoch`, `--seg-phase-advect-start-epoch`, `--softmax-temp-end`

- missing/unknown provenance rung (154 flags):
  - `--accum-pairs`, `--activation`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`, `--area-constraint-birth`
  - `--area-constraint-birth-force`, `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`
  - `--birth-completion-ramp`, `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--curriculum`
  - `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`
  - `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hosc-beta`
  - `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`, `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`
  - `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`, `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`
  - `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`, `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`
  - `--lane-band-softness`, `--lane-band-start-epoch`, `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`
  - `--lane-prior-phi1-mode`, `--lane-render-band`, `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`
  - `--mlx-device`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`
  - `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`
  - `--persistence-warmup-epochs`, `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`, `--pose-carrier-residual-mode`, `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`
  - `--render-aa`, `--render-h`, `--render-w`, `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-island-eased`, `--seed-islands`
  - `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`
  - `--seg-phase-advect-ref`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`
  - `--softmax-temp-end`, `--softmax-temp-start`, `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`
  - `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tau-advance-mode`, `--tau-anneal-shape`
  - `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`
  - `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- missing runtime receipt schema (199 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--containment-damp`, `--containment-mode`
  - `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`
  - `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--eikonal-weight`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`
  - `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`
  - `--island-dilate-px`, `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`
  - `--jacobian-basin-telemetry`, `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`
  - `--ladder-lane-r0`, `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`
  - `--ladder-release-coeff`, `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`
  - `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`
  - `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`, `--max-bank-freq`
  - `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`
  - `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`
  - `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`
  - `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`
  - `--render-h`, `--render-w`, `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`
  - `--seed-island-eased`, `--seed-islands`, `--seed-lr`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`
  - `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-phase-advect-start-epoch`, `--seg-phase-advect-weight`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`
  - `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`, `--stage-checkpoints`
  - `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`, `--structured-init-sdf-clip`
  - `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-live-mq`, `--tail-lr-prop-tau`
  - `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`
  - `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

#### Each of `v9_cgauge_truly_optimal_core`, `v9_cgauge_ideal_mod19`, `v9_cgauge_ideal_mod32`

- missing-or-nonunique Lever owner (141 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--anneal-epochs`, `--annulus-band`
  - `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--async-verdict`, `--cache-gt-skeleton`, `--chroma`
  - `--ckpt-every`, `--cldice-iters`, `--containment-damp`, `--containment-mode`, `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`
  - `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--ema-decay`
  - `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`
  - `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`, `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`
  - `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`
  - `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`
  - `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`, `--max-bank-freq`
  - `--micro-batch-pairs`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`
  - `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`
  - `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`
  - `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`
  - `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seed-island-eased`, `--seed-islands`
  - `--seed-lr`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--siren-init`, `--softmax-temp-end`
  - `--softmax-temp-start`, `--stage-checkpoints`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`, `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`
  - `--structured-init-thresh`, `--tail-live-mq`, `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`
  - `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- missing Lever.constant_refs LawRef (224 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--closed-loop-control`, `--closed-loop-eikonal-bump`
  - `--closed-loop-eikonal-max`, `--closed-loop-max-bumps`, `--closed-loop-min-sustained-windows`, `--closed-loop-stop-after-windows`, `--containment-damp`, `--containment-mode`, `--curriculum`, `--curriculum-event-triggered`
  - `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`
  - `--dseg-aware-taper-strength`, `--eikonal-weight`, `--eikonal-weight-end`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`
  - `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`
  - `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`
  - `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`
  - `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`
  - `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`, `--lane-band-start-event`
  - `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`, `--length-sigma-matrix`
  - `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`, `--margin-saliency-start-epoch`
  - `--margin-saliency-target`, `--margin-saliency-tau`, `--margin-saliency-weight`, `--max-bank-freq`, `--micro-batch-pairs`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`
  - `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`
  - `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`
  - `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`
  - `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`, `--safe-compile-regions`
  - `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seed-island-eased`, `--seed-islands`, `--seed-lr`
  - `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--seg-margin-satisfice-band`, `--seg-margin-satisfice-delta-r`, `--seg-margin-satisfice-headroom`
  - `--seg-margin-satisfice-msafe`, `--seg-margin-satisfice-start-epoch`, `--seg-margin-satisfice-weight`, `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-phase-advect-start-epoch`
  - `--seg-phase-advect-weight`, `--seg-subpix-boundary-start-epoch`, `--seg-subpix-boundary-v-band`, `--seg-subpix-boundary-weight`, `--seg-subpix-edge-weight-path`, `--seg-subpix-edge-weight-source`, `--seg-subpix-ref-domain`, `--seg-temporal-screw-band`
  - `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`
  - `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`
  - `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-live-mq`
  - `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`
  - `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- missing canonical compiler record (218 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--closed-loop-control`, `--closed-loop-eikonal-bump`
  - `--closed-loop-eikonal-max`, `--closed-loop-max-bumps`, `--closed-loop-min-sustained-windows`, `--closed-loop-stop-after-windows`, `--containment-damp`, `--containment-mode`, `--curriculum`, `--curriculum-event-triggered`
  - `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`
  - `--dseg-aware-taper-strength`, `--eikonal-weight`, `--eikonal-weight-end`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`
  - `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-omega`, `--island-dilate-px`, `--jacobian-basin-every`
  - `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`, `--ladder-gate-softness`
  - `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`, `--ladder-max-step-px`
  - `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`, `--ladder-sigma-eff`
  - `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`, `--lane-band-start-event`, `--lane-band-tau`
  - `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`, `--length-sigma-matrix`, `--length-weight`
  - `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-end`, `--margin-saliency-start-epoch`, `--margin-saliency-target`, `--margin-saliency-tau`, `--margin-saliency-weight`
  - `--max-bank-freq`, `--micro-batch-pairs`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`, `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`
  - `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`, `--palette-anchor`
  - `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--polyak-finisher-arm`, `--pose-carrier`, `--pose-carrier-pitch`
  - `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`
  - `--render-aa`, `--render-h`, `--render-w`, `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`
  - `--seed-blend`, `--seed-island-eased`, `--seed-islands`, `--seed-lr`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`
  - `--seg-form-unify-tau`, `--seg-margin-satisfice-band`, `--seg-margin-satisfice-delta-r`, `--seg-margin-satisfice-headroom`, `--seg-margin-satisfice-msafe`, `--seg-margin-satisfice-start-epoch`, `--seg-margin-satisfice-weight`, `--seg-phase-advect-band`
  - `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-phase-advect-weight`, `--seg-subpix-boundary-start-epoch`, `--seg-subpix-boundary-v-band`, `--seg-subpix-boundary-weight`, `--seg-subpix-edge-weight-path`
  - `--seg-subpix-edge-weight-source`, `--seg-subpix-ref-domain`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`
  - `--siren-init`, `--softmax-temp-start`, `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`
  - `--structured-init-include-lane`, `--structured-init-lr`, `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`
  - `--tail-dwell-min`, `--tail-live-mq`, `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`
  - `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`
  - `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- compiler record mismatches owning LawRef (6 flags):
  - `--hosc-beta-end`, `--lr-anneal-epochs`, `--lr-hold-frac`, `--polyak-finisher-start-epoch`, `--seg-phase-advect-start-epoch`, `--softmax-temp-end`

- missing/unknown provenance rung (179 flags):
  - `--accum-pairs`, `--activation`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`, `--area-constraint-birth`
  - `--area-constraint-birth-force`, `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`
  - `--birth-completion-ramp`, `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--closed-loop-control`
  - `--closed-loop-eikonal-bump`, `--closed-loop-eikonal-max`, `--closed-loop-max-bumps`, `--closed-loop-min-sustained-windows`, `--closed-loop-stop-after-windows`, `--curriculum`, `--curriculum-event-triggered`, `--curriculum-min-stage-epochs`
  - `--curriculum-nucleus-guard`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`, `--dseg-aware-taper-strength`, `--eikonal-weight-end`, `--ema-decay`
  - `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`, `--grad-normalize`, `--gt-cache`, `--hosc-beta`, `--hosc-beta-anneal`
  - `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`, `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`
  - `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`, `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`
  - `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`, `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`
  - `--lane-band-start-epoch`, `--lane-band-start-event`, `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`
  - `--lane-render-band`, `--length-sigma-matrix`, `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`
  - `--margin-saliency-start-epoch`, `--margin-saliency-target`, `--margin-saliency-tau`, `--margin-saliency-weight`, `--micro-batch-pairs`, `--mlx-device`, `--muon-lr`, `--muon-lr-final-frac`
  - `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`, `--num-pairs`, `--out-dir`
  - `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`, `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`
  - `--pose-carrier-residual-mode`, `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`
  - `--safe-compile-regions`, `--score-domain-loss`, `--seed`, `--seed-island-eased`, `--seed-islands`, `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`
  - `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--seg-margin-satisfice-band`, `--seg-margin-satisfice-delta-r`, `--seg-margin-satisfice-headroom`, `--seg-margin-satisfice-msafe`, `--seg-margin-satisfice-start-epoch`, `--seg-margin-satisfice-weight`
  - `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-subpix-boundary-start-epoch`, `--seg-subpix-boundary-v-band`, `--seg-subpix-boundary-weight`, `--seg-subpix-edge-weight-path`
  - `--seg-subpix-edge-weight-source`, `--seg-subpix-ref-domain`, `--seg-temporal-screw-band`, `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`
  - `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`, `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`
  - `--structured-init`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tau-advance-mode`
  - `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`, `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`
  - `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

- missing runtime receipt schema (224 flags):
  - `--accum-pairs`, `--activation`, `--adam-beta2`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--amplify-weight`, `--anneal-epochs`
  - `--annulus-band`, `--annulus-bottom-k`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--annulus-telemetry`, `--area-constraint-birth`, `--area-constraint-birth-force`
  - `--area-constraint-classes`, `--area-constraint-tolerance`, `--async-verdict`, `--birth-completion-area-band`, `--birth-completion-classes`, `--birth-completion-event`, `--birth-completion-post-level`, `--birth-completion-ramp`
  - `--birth-completion-ramp-epochs`, `--birth-completion-tau-persist`, `--cache-gt-skeleton`, `--chroma`, `--ckpt-every`, `--cldice-iters`, `--closed-loop-control`, `--closed-loop-eikonal-bump`
  - `--closed-loop-eikonal-max`, `--closed-loop-max-bumps`, `--closed-loop-min-sustained-windows`, `--closed-loop-stop-after-windows`, `--containment-damp`, `--containment-mode`, `--curriculum`, `--curriculum-event-triggered`
  - `--curriculum-min-stage-epochs`, `--curriculum-nucleus-guard`, `--curriculum-nucleus-min-part-frac`, `--curriculum-nucleus-within-flip`, `--curriculum-reanchor-levers`, `--dseg-aware-taper`, `--dseg-aware-taper-floor`, `--dseg-aware-taper-scale`
  - `--dseg-aware-taper-strength`, `--eikonal-weight`, `--eikonal-weight-end`, `--ema-decay`, `--epochs`, `--eval-every`, `--fused-r-kernel`, `--grad-clip`
  - `--grad-normalize`, `--gt-cache`, `--hidden-dim`, `--hosc-beta`, `--hosc-beta-anneal`, `--hosc-beta-end`, `--hosc-omega`, `--island-dilate-px`
  - `--jacobian-basin-every`, `--jacobian-basin-f-basin`, `--jacobian-basin-k-pairs`, `--jacobian-basin-quorum-q`, `--jacobian-basin-sigma-floor`, `--jacobian-basin-stratify-t`, `--jacobian-basin-t0`, `--jacobian-basin-telemetry`
  - `--ladder-gate-softness`, `--ladder-island-homotopy`, `--ladder-lane-anneal-epochs`, `--ladder-lane-birth-epochs`, `--ladder-lane-dash-gate`, `--ladder-lane-hold-epochs`, `--ladder-lane-lambda-gate`, `--ladder-lane-r0`
  - `--ladder-max-step-px`, `--ladder-movable-anneal-epochs`, `--ladder-movable-birth-epochs`, `--ladder-movable-hold-epochs`, `--ladder-movable-lambda-gate`, `--ladder-movable-r0`, `--ladder-refresh-every`, `--ladder-release-coeff`
  - `--ladder-sigma-eff`, `--lane-band-comb-softness-m`, `--lane-band-dash-comb`, `--lane-band-dash-forward-max-m`, `--lane-band-eps`, `--lane-band-softness`, `--lane-band-start-epoch`, `--lane-band-start-event`
  - `--lane-band-tau`, `--lane-band-uncertainty-source`, `--lane-band-weight`, `--lane-prior-phi1`, `--lane-prior-phi1-dash-gate`, `--lane-prior-phi1-mode`, `--lane-render-band`, `--length-sigma-matrix`
  - `--length-weight`, `--logit-adjust-classes`, `--logit-adjust-loss-tau`, `--lr`, `--lr-anneal-epochs`, `--lr-end`, `--lr-hold-frac`, `--margin-saliency-start-epoch`
  - `--margin-saliency-target`, `--margin-saliency-tau`, `--margin-saliency-weight`, `--max-bank-freq`, `--micro-batch-pairs`, `--mlx-device`, `--mod-dim`, `--muon-adamw-lr`
  - `--muon-lr`, `--muon-lr-final-frac`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-epoch`, `--muon-start-event`, `--muon-warm-start-momentum`, `--n-hidden`
  - `--num-pairs`, `--out-dir`, `--palette-anchor`, `--per-group-grad-clip`, `--persistence-classes`, `--persistence-loss-weight`, `--persistence-recall-weight`, `--persistence-warmup-epochs`
  - `--polyak-finisher-arm`, `--polyak-finisher-start-epoch`, `--pose-carrier`, `--pose-carrier-pitch`, `--pose-carrier-residual-mode`, `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`
  - `--pose-carrier-source`, `--pose-finish-engage-on`, `--pose-finish-start-epoch`, `--pose-grad-coeff-max`, `--render-aa`, `--render-h`, `--render-w`, `--safe-compile-regions`
  - `--score-domain-loss`, `--seed`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seed-island-eased`, `--seed-islands`, `--seed-lr`
  - `--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--seg-chroma-boundary-start-event`, `--seg-chroma-boundary-weight`, `--seg-form-unify-tau`, `--seg-margin-satisfice-band`, `--seg-margin-satisfice-delta-r`, `--seg-margin-satisfice-headroom`
  - `--seg-margin-satisfice-msafe`, `--seg-margin-satisfice-start-epoch`, `--seg-margin-satisfice-weight`, `--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-ref`, `--seg-phase-advect-start-epoch`
  - `--seg-phase-advect-weight`, `--seg-subpix-boundary-start-epoch`, `--seg-subpix-boundary-v-band`, `--seg-subpix-boundary-weight`, `--seg-subpix-edge-weight-path`, `--seg-subpix-edge-weight-source`, `--seg-subpix-ref-domain`, `--seg-temporal-screw-band`
  - `--seg-temporal-screw-classes`, `--seg-temporal-screw-start-epoch`, `--seg-temporal-screw-start-event`, `--seg-temporal-screw-weight`, `--seg-temporal-screw-xi-source`, `--siren-init`, `--softmax-temp-end`, `--softmax-temp-start`
  - `--stage-checkpoints`, `--stage-transition-reset-moments`, `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`, `--stage-transition-rewarmup-shape`, `--structured-init`, `--structured-init-include-lane`, `--structured-init-lr`
  - `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--tail-cycle-floor-epochs`, `--tail-cycles-max`, `--tail-dwell-min`, `--tail-live-mq`
  - `--tail-lr-prop-tau`, `--tail-start-epoch`, `--tail-stop-marginal-s`, `--tail-tau-halving`, `--tau-advance-mode`, `--tau-anneal-shape`, `--tau-softplus-tau`, `--verdict-anchor-every`
  - `--verdict-batch`, `--verdict-device`, `--verdict-pairs`, `--w-pose`, `--w-seg`, `--weight-decay`, `--weight-entropy-penalty-lambda`, `--witness-alone-island-loss`

### Required backfill before strict flip

For every tuple above, backfill the missing canonical edge on the owning closed V9 factory: one and only one Lever owner, one substantive LawRef, one compiler record agreeing with that LawRef, a known value-provenance rung, and one runtime receipt schema. Correct the six named compiler/LawRef disagreements, regenerate the LawRef/compiler/provenance coverage tables, and remove or re-own stale provenance key `schedule`. Then rerun the exact strict probe; only literal zero authorizes the atomic #406 meta-gate flip.

## Verification

- Focused tests: 102 passed (`test_codex_apparatus_safety`, `test_codex_delegate_retry`, `test_dispatch_cli_shell_hazards`, `test_dsl_compile_hash_enforcement`, `test_launch_dsl_config_gate`).
- No GPU, provider dispatch, score claim, frontier movement, or experiment/result run-directory access.
- Pointer delta: **UNMOVED**.
