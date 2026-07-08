# Full-stack dynamics telemetry — the INTERACTION layer (task #312) — landed 2026-07-08

[no-triality]  ·  axis: all numbers [macOS advisory] NON-PROMOTABLE  ·  pointer 0.19110 UNMOVED.

Existing telemetry measures STATES (16-term loss values, per-class verdicts, costate λ, annulus
#333, memory, mod-dim pack). This landing adds the INTERACTION layer: how the series move
*together* — gradient synergy/antagonism, curvature, and cross-series lead/lag. Three phases,
each independently landable (committed separately).

## STORES CONSULTED (proactive recall before building)
- `.omx/research/costate_controller_design_20260705.md` — the D-3/4/5 verdict (GN/Fisher SPECTRUM
  as a 2nd-order costate SENSE state; first measurement = $0 HVP-Lanczos on a saved checkpoint) +
  the shadow-controller rec schema (`{action, predicted_dS, rationale, evidence, costate}`) + the
  actuation boundary (advisory-only; no auto-fire).
- `src/tac/witness_control/shadow_controller.py` — `ShadowReport.to_row` / `_recommendations`
  schema my Phase C recs MATCH (so the costate DECIDE surface ingests them).
- The levelset trainer `total_loss_fn` + `terms_out` (#304 item 4 per-term decomposition) + the
  `_loss_terms_for_chunk` no-grad recompute + `LOSS_TERM_KEYS` (~16 terms) — Phase A reuses this
  exact decomposition; Phase B reuses the seg term through the SAME forward.
- CLAUDE.md #312 GradNorm discipline (per-step grad balancing MUTED a canary once) → the grad
  hook OBSERVES only, emits at boundaries, never rescales a live gradient.
- The live checkpoint region already carries `_mdd_governor_ok` + `_mdd_ablation_checkpoint`
  (sibling mod-dim telemetry, in HEAD) — the curvature hook REUSES `_mdd_governor_ok` and fires at
  the same checkpoint cadence. Sibling's UNCOMMITTED mod-dim edits were preserved (patch-file
  commits; my hunks only).

## Phase A — gradient-interaction telemetry (the synergy/antagonism matrix, live)
`src/tac/witness_control/grad_interaction.py` (+ trainer `--grad-interaction-telemetry`, default
OFF). At stage/octave boundaries (and every `--grad-interaction-every` N epochs) a SEPARATE
`value_and_grad` per ACTIVE loss term on a K-pair sample (`--grad-interaction-k-pairs`, default
32) yields per-term gradient vectors → term×term **cosine matrix** (upper triangle), **conflict
pairs** (cos < −0.2, tagged), **dominance shares** (‖g_i‖ / Σ‖g_j‖). Two terms with aligned
gradients are SYNERGISTIC; opposed are ANTAGONISTIC (fighting the optimizer).
- **Score-neutrality (mandatory bit-compare):** `MxRngGuard` snapshots + restores the numpy
  stream exactly and FAILS CLOSED if the measurement touches the global MLX stream. Verified: MLX's
  default RNG has internal state NOT exposed by `mx.random.state` (state assignment does not
  reproduce the next draw), so the contract is "measurement must not draw from the global stream"
  (use a forked key / deterministic ops) — the witness forward is deterministic, so this holds. A
  numpy draw taken after a guarded measurement block is bit-identical to one taken without it (test).
- **Never touches training:** a separate `value_and_grad`; opt/EMA never called; grads only
  flattened to numpy. Default-OFF ⇒ dead branch ⇒ byte-identical. Fail-open (emits an error row,
  never crashes).

## Phase B — curvature spectrum (the D-3/4/5 first measurement)
`src/tac/witness_control/curvature.py`: HVP-Lanczos top-k eigenvalues of a symmetric operator
given only a matrix-vector product `matvec(v)->Hv` (full reorthogonalization) + Hutchinson trace +
`Spectrum{lambda_max, anisotropy λ1/λk, sharpness λ_max/|mean(top_k)|, trace, negative-curvature
saddle flag}`. `make_mlx_hvp` = nested `mx.grad` HVP; `mlx_model_hvp` flattens an `nn.Module`'s
params, builds the matvec, and SNAPSHOTS/RESTORES the module (score-neutral).

- **Standalone $0 tool** `tools/witness_curvature_spectrum.py`: `--self-validate` runs the full
  MLX HVP→Lanczos path on a synthetic witness-shaped MLP and cross-checks Lanczos vs a dense
  Hessian; measured **λ_max ≈ 1.28–1.38, anisotropy ≈ 2.0, trace ≈ 5.0, Lanczos-vs-dense λ_max
  abs-err ≈ 2.7e-4** — the instrument is proven correct end-to-end today.
- **mod32cap ep726 curvature numbers — HONESTLY BLOCKED (NO fake):** the mod32cap checkpoint is a
  SELF-ORIENT levelset witness (`__cfg_self_orient=1`; `out_sdf`/`out_tex`/`palette`/directional
  `__bank_*`). Its through-R seg forward requires the trainer's exact self-orient reorient +
  palette-SDF setup, which is not standalone-reconstructable without risking a wrong (fake)
  number. The tool FAILS CLOSED with the named blocker + resolution, and emits NO curvature number
  for it. The REAL mod32cap spectrum comes from the in-trainer hook below on the next governed run
  resumed from that checkpoint (duty-to-measure), OR from extending the tool with the self-orient
  feature reconstruction.
- **In-trainer hook** `--curvature-telemetry` (default OFF): at CHECKPOINT cadence, HVP-Lanczos on
  the LIVE seg-loss forward via the tested `mlx_model_hvp` — governor-gated (`_mdd_governor_ok`:
  SKIP-with-logged-reason under memory pressure) + fail-open. This is the honest path to the real
  through-R numbers; its HVP core is unit-tested (recovers a known Hessian on a tiny module), the
  through-R wiring is contained by fail-open until a governed validation run exercises it.

## Phase C — cross-series interaction ANALYZER (offline; no trainer changes)
`tools/witness_dynamics_analyzer.py` + `src/tac/witness_control/dynamics_analyzer.py`. Consumes a
run dir (`run.log` loss_terms/verdict + `costate_shadow.jsonl`), builds named series (per-term
losses, d_seg/d_pose/implied_S/bytes, λ_pose, gnorm, softmax_temp/octave, hosc_beta), forward-fills
to a union epoch grid, and computes **windowed lagged cross-correlations** (lag>0 ⇒ first series
LEADS) + **stability** (1 − dispersion of per-window correlation). Emits (i) a machine-readable
synergy report (`--jsonl-out` / `--json`) and (ii) ranked ADVISORY recs matching the shadow rec
schema (`predicted_dS=None` — synergy is directional, never a fabricated ΔS; every rec carries the
evidence chain + `[macOS advisory] NON-PROMOTABLE`). Correlation math is unit-tested against
synthetic known lead/lag series.

### 3 real findings from run-1 (`levelset_n600_witness_mod32cap_20260706T115554Z`, 11 series / 1001 epochs / 37 interactions)
1. **The seg term DOMINATES the gradient.** `gnorm ~ term:seg` r=+0.98 (lag 0), `loss_total ~
   term:seg` r=+1.00 — the ~15 auxiliary terms (eikonal, length, boundary, thin-lane, …) contribute
   negligibly to the gradient norm at this config. Fine-tune implication: reweighting aux terms has
   almost no gradient leverage here; d_seg is essentially a pure-seg descent.
2. **The two schedule knobs are a coordinated PR95-style anneal, hosc_beta LEADING.** `hosc_beta ~
   softmax_temp` r=−0.99 at **lag +8** (hosc_beta leads softmax_temp by 8 epochs). This is the
   curriculum's two-knob anneal visible as a lagged coupling — directly relevant to the "curriculum
   = last PR95 inheritance" concern (the schedule shape is a tunable lever, not physics).
3. **Schedule changes hit gnorm ~2 epochs BEFORE the loss responds.** `hosc_beta ~ gnorm` r=−0.78
   and `softmax_temp ~ gnorm` r=+0.75, both at **lag −2** (schedule leads gnorm), stability 0.71 —
   gnorm is an early-warning signal of a schedule bump's effect on optimization stability, ahead of
   the loss. Cadence-tuning implication: watch gnorm, not loss, to time anneal steps.
   (Sanity anchor, not a finding: `d_pose ~ lambda_d_pose` r=−0.89 confirms the costate math is live
   — λ_pose = 5/√(10·d_pose) rises as d_pose falls.)

## What each surface answers for fine-tuning
- **grad-interaction matrix (A):** *which loss terms are synergistic vs antagonistic right now, and
  which dominates the step* → reweight/recadence antagonistic pairs; know when an aux term is inert.
- **curvature spectrum (B):** *is the current minimum a knife-edge (sharp) or a flat basin, and how
  ill-conditioned (anisotropic)* → set LR / trust-region / when to advance a stage (flat basin ⇒
  marginal value of more epochs ≈ 0; high anisotropy ⇒ slow flat directions need a different lever).
- **cross-series analyzer (C):** *the lead/lag timing structure — which knob's change propagates to
  which distortion, and after how many epochs* → tune anneal cadence and stage-boundary timing off
  the measured lag, not guesswork.

## Landed / verified
- 3 commits (Phase C · Phase A module+trainer · Phase B module+trainer), all via serializer;
  trainer edits via `--patch-file` (shared hot file; sibling mod-dim edits untouched, verified).
- 43 tests pass (14 analyzer · 18 grad-interaction incl e2e MLX RNG-clean bit-compare · 11
  curvature incl MLX HVP vs known Hessian). ruff F clean. Trainer AST OK post-commit; hooks
  default-OFF ⇒ byte-identical.
- Duty-to-measure (named, not orphaned): run the in-trainer `--curvature-telemetry` on a governed
  run to get the real self-orient mod32cap spectrum; that is operator-GO (heavy launch), not
  autonomous.
