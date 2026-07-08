# MOD-DIM DYNAMICS telemetry — score-neutral, default-ON latent-table introspection for v7

**Date:** 2026-07-08 · **Trigger:** operator "will the next run provide sufficient telemetry to
understand the mod dim dynamics and how it's working under the hood and exploitation" (answer was NO;
this lands the YES) + charter amplification "Per dim utilization could be a big one." · **[no-triality]**
(observability, not a lever — no DSL governance entry; code commits carry the triality treatment).

## STORES CONSULTED
- `CLAUDE.md` §"'Off' is a tracked queue" (score-neutral read-only telemetry DEFAULTS ON; byte-identity
  by construction) + §"Max observability" (6-facet) + the annulus-telemetry precedent (`--annulus-telemetry
  default=True`, the exact default-ON companion-row pattern this mirrors).
- MEMORY.md L77 (`quadratic_head_chart_subset_solve_gap_v1`): the mod-32 autopsy anchor — trained latents
  effective rank ~17.8, k90 ~20 ≈ Whitney 2·8+1 = 17. Measured ONCE offline; this makes it CONTINUOUS.
- `src/tac/boundary_math/lever_b_levelset_generator.py`: existing spectral helpers
  (`code_spectrum_participation_ratio`, `film_modulation_participation_ratio`) — REUSED-by-verification
  (my `effective_rank` cross-checks equal to `code_spectrum_participation_ratio` in a test).
- Trainer verdict/checkpoint emission sites (`experiments/train_levelset_witness_realized_through_R_mlx.py`:
  `_emit_verdict_row` async + the SYNC inline path + baseline v0 + periodic checkpoint) — the annulus/
  per-class companion-row pattern I mirror.
- `tools/witness_run_introspect.py` (per-stage readers → `introspect_run` aggregator; presence-gated).
- `tools/system_memory_governor.py` (`read_memory_pressure_level`) for the cut-3 governor gate.
- `.omx/state/deferral_ledger.md` (D18 registration).

## WHAT LANDED

### Pure math — `src/tac/boundary_math/mod_dim_dynamics.py` (MLX/torch-free, $0-unit-testable)
Spectral pack + per-dim + redundancy + exploitation, all pure numpy (read-only; no input mutation; no
`np.random` draw). The trainer wraps every call fail-open so telemetry can NEVER break a run.

### The emitted series and what each ANSWERS

| Series (per `{stage:mod_dim_dynamics}` verdict row) | The operator question it answers |
|---|---|
| `spectrum.effective_rank` (participation ratio) | how many latent dims are actually live — the autopsy 17.8, now continuous |
| `spectrum.k90` / `k99` (energy cutoffs) | intrinsic dimensionality — does k90 sit at ~20 ≈ Whitney 2·8+1? does it SATURATE before dash birth? |
| `spectrum.spectral_entropy_norm` + `top_energy_fracs` | is the code spectrum uniform (all dims live) or collapsing to a few directions? |
| `tau` + `seg_form` (per-octave alignment) | does effective rank TRACK the coarse-to-fine anneal octaves (rank-growth vs τ)? |
| `per_dim.variance` (D-vector) | which dims VARY across pairs |
| `per_dim.film_consumption` (D-vector) | **which dims the network actually READS** (FiLM input-weight column norm) — amplification cut 1 |
| `per_dim.xi_max_r2` + `xi_r2_by_component` (D×6) | per-dim r² vs each twist component — **names the ξ-redundant deletable dims** — amplification cut 2 |
| `latent_xi_cca` (canonical corrs + mean/max) | table-level latent↔ξ redundancy (memory L77), now continuous |

### The EXPLOITATION hooks (the "and exploitation" half)
1. **k90 truncate-bytes estimate** — `k90_truncate_bytes_estimate` = `round(code_bytes_full · k90/mod_dim)`
   on every verdict row: the projected blob savings if the latent table were truncated to k90 columns at
   export (the free-rate lever the autopsy suggested). ESTIMATE only — no behavior change.
2. **Deferral D18 registered** — the real truncate-at-export byte-close A/B (truncate `code` to measured
   k90 → real Δbytes vs Δd_seg/Δd_pose) fires at **v7 stop → byte-close on the FINAL ckpt**, consuming
   this k90 series. Ledger row D18, status ARMED (sensor landed).
3. **Per-dim bit-allocation = #157 waterfill at mod-dim granularity → #336 consumer.** The checkpoint-
   cadence ablation row carries `bit_allocation_hint = normalize(per_dim_util · |Δd_seg|)` — per-dim
   export bit-allocation weights the Catalog #336 master-gradient bit-allocator can consume. The **#299
   mod-dim A/B**'s sensor is this whole series (effective_rank / k90 vs mod_dim sweeps read the same rows).

### Amplification cut 3 — per-dim d_seg ATTRIBUTION (checkpoint cadence, governor-gated)
`{stage:mod_dim_ablation}` at checkpoint cadence: zero-ablate latent dim j **in a COPY of the code**,
re-render a K-pair sample's frame1 through the frozen CPU SegNet, record `Δd_seg` — the direct "which
dims carry score" vector (K default 32, `--mod-dim-ablation-k`). Heavier (mod_dim×K renders), so it is
**governor-gated**: runs only under NORMAL macOS memory pressure (`read_memory_pressure_level()==1`),
else emits a `{stage:mod_dim_ablation, skipped:true, reason:...}` row and the run continues — never adds
load beside the live run (operator "if (3)'s cost is material… governor-gated with the skip-logged path").

## SCORE-NEUTRALITY PROOF (byte-identity by construction)
- Every emission READS snapshots only: `int8_dequant_params(ema_np)` returns a NEW dict; the SVD / per-dim
  math run on centered copies; the ablation zeroes a `.copy()`. Inputs are never mutated (tested:
  `test_row_build_does_not_mutate_inputs_or_consume_rng`, `test_ablation_copies_and_does_not_mutate`).
- NO `np.random` draw anywhere in the path → the seeded training RNG stream is untouched (tested:
  `np.random.get_state()` identical before/after a full row build).
- The rows are companions to `{stage:verdict}`, printed to stdout under the same lock — NEVER appended to
  `history` / `result.json` and NEVER read back into training / parity / resume. `--no-mod-dim-dynamics`
  gives a pure byte-identity A/B. Therefore no trained weight, archive byte, d_seg, or d_pose can change.

## WIRING (all default-ON per "'Off' is a tracked queue"; fail-open)
- `--mod-dim-dynamics` (default ON) → per-verdict row on BOTH the SYNC path (the default; async is
  default-off) AND the async worker AND the pre-loop baseline v0. τ = the snapshot softmax-temp (async)
  / live model.softmax_temp (sync) for octave alignment.
- `--mod-dim-ablation` (default ON, governor-gated) + `--mod-dim-ablation-k 32` → checkpoint-cadence row.
- Dashboard: `tools/witness_run_introspect.py` gains `read_mod_dim_dynamics` (presence-gated, additive;
  surfaces latest spectrum + effective-rank/k90 sparkline series) wired into `introspect_run` — no panel
  redesign; None over pre-2026-07-08 run dirs.

## DSL note
No DSL governance entry: this is OBSERVABILITY, not a score-affecting lever, and I did NOT add an emission-
cadence knob (default = the existing verdict/checkpoint cadence). If a cadence knob is ever added it must
be registered per the "'Off' is a tracked queue" rule; it is not, so there is nothing to register.

## TESTS
`src/tac/tests/test_mod_dim_dynamics.py` (20) + `tools/test_witness_run_introspect.py` (+3 mdd readers) =
23 new. Coverage: spectrum on known matrices (rank-1 collapse, isotropic full-rank, k-energy bounds,
zero-spectrum) · effective-rank ↔ canonical PR-helper cross-check · score-neutrality (no mutation, no RNG
consumption) · ablation copy-safety + fail-open-nan · ξ-CCA (identical→1, independent→small, too-few-safe)
· per-dim variance/FiLM-consumption(quadrature+width-guard)/ξ-r2 · truncate estimate · waterfill hint ·
row assembly/JSON-safety · introspect reader (autopsy anchor 17.8/20, error-row surfaced, absent→None).
