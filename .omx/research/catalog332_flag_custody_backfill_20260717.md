# Catalog #332 flag-custody backfill — 3,862 → 0 bijection residuals (2026-07-17)

**APPARATUS, pointer-neutral: the exact frontier pointer 0.19108 is UNMOVED.** This landing
satisfies the operator-authorized #332 "DSL-as-complete-SoT" provenance debt ("we will have
to satisfy the debt sometime"); it authorizes (but does not perform) the eventual #406/#332
strict-flip. No score/promotion claim may cite it.

## MEASURED before → after (live checker, this repo, 2026-07-17)

- `check_config_flag_provenance_bijection_complete(strict=False)`:
  **3,862 residuals → 0** across the closed live V9 factory set
  (`v9_cgauge_432` 907 → 0; `v9_cgauge_truly_optimal_core` /
  `v9_cgauge_ideal_mod19` / `v9_cgauge_ideal_mod32` 985 → 0 each), with
  **deterministic per-factory bijection hashes** across the checker's double
  compile.
- `check_launch_and_governor_require_dsl_compile_hash(strict=False)`:
  **2 findings → 0** (the two launcher dry-start findings were a STALE AST
  target in the gate — the 07-17 B1 durability wrapper split `_run_dry_start`
  into wrapper + `_run_dry_start_inner`, which still performs the DSL-bound
  launch + typed Lever; the gate now follows the split and additionally
  REQUIRES the wrapper to delegate to the inner body).

## The mechanism (honest provenance — the entire point)

`tac.witness_dsl.spec_v9_cgauge.attach_flag_custody` completes, per factory, on the FINAL
composed program:

1. **Ownership** — ONE value-neutral `v9_flag_custody_rollup` TypedLever owns every flag no
   scientific Lever owns (147 in 432; 139 per ideal arm); existing owning levers are
   extended in place for their own flags. Byte-identity of the compiled argv is asserted
   fail-closed at attach time, and `WitnessProgram.validate` now enforces the composition
   law: custody levers must be value-neutral (a derived config that mutates base/out_dir
   after custody REFUSES loudly instead of being silently shadowed).
2. **LawRef custody, honestly classified per flag**:
   - flags whose canonical compiler record already cites a REAL derivation
     (`hosc_beta_fireband_pin_v1`, `tau_end_knee_launch_v1`, `lr_control_denominator_v1`,
     `lr_hold_frac_no_hold_v1`, `muon_finisher_schedule_warmstart_and_lr_anneal_v1`,
     `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`, taper/msafe lever
     records) get a LawRef RECONSTRUCTED from that record — same equation_id, same
     resolved inputs; never a fabricated new derivation;
   - every other flag (the overwhelming majority — generic config knobs) is custodied via
     the newly REGISTERED non-derivational identity law
     **`dsl_custodied_scalar_identity_v1`** as an explicit **class-4 `hardcoded_waiver`**
     with typed `HardcodedWaiverCustody` (reason / owner / rederivation_trigger /
     battery_arm). Value bytes preserved; ZERO scientific authority claimed. No flag was
     given a fake derivation.
3. **Compiler records** — identity resolutions are installed into the launcher constants
   manifest (snake keys); flags with existing records keep exactly the one record
   (equation match asserted). `refresh_identity_custody_records` regenerates an identity
   record when a LATER composed Lever (launcher dry-start delta, `--dsl-lever` factory,
   ISO treatment) legitimately overrides the flag — single scalar-value owner = emitted
   argv, wired through `CrucibleV7LaunchConfig._rebind_typed`.
4. **Provenance table** — the COMPLETE per-flag rung table (curated reviewed rows win;
   auto rows mirror the flag's record ladder_class or are honest class-4) is registered
   per program and consumed by `_provenance_table_for_program` (the gate READS the DSL's
   table; it never synthesizes provenance).
5. **Receipt schema** — `v9_config_compile.v1` per custodied flag (the existing taper
   precedent).

### Class breakdown (v9_cgauge_432, 201 flags; ideal arms 226 analogous)

- reconstructed-from-real-compiler-record LawRefs: 6 (432) / 11 (ideal, incl. taper x4 +
  msafe already custodied by their levers)
- class-4 identity `hardcoded_waiver` custody: the remaining ~195 (432) / ~210 (ideal)
- curated provenance rows retained verbatim: 47 (432 tables) — incl. the `--mod-dim`
  Whitney derivation row; the mod-32 arm's row honestly labels the incumbent CONTROL.

### Scoped exemption (contract-consistent, not scope-shrink)

`--out-dir` (the gate's own `_VOLATILE_ARGV_VALUE_FLAGS` run-local identity, placeholdered
in EVERY canonical payload) is exempt from the semantic ownership edges: demanding a
scientific LawRef for a per-launch run directory is provenance theater, and ownership of a
volatile value would shadow legitimate derived-config out_dir changes. Token/type/order
custody still applies to it.

## Named finite fixes also landed

- `--seg-form-unify-tau` dual ownership (ideal `unified_tau_eikonal_hold` re-asserted the
  flag the inherited `seg_form_unify_tau` Lever owns) — removed from eik_hold; boolean
  presence added to the factory's fail-closed actuator check.
- stale provenance key `schedule` — re-homed to `V9_CGAUGE_432_SCHEDULE_PROVENANCE`
  (non-flag constant; information preserved verbatim).
- stale ideal compiler-record keys `eikonal_retention_tau_rung` (an ACTUATOR record →
  moved to the DSL program manifest) and `margin_saliency_reachability` (now emitted only
  when `--margin-saliency-reachability` is actually compiled, i.e. the sR treatment).
- `lawref.py` extended to admit **string literals/fallbacks** (per the #351 target
  contract: the identity law "preserves bool/int/float/string bytes"); bool custody stays
  the canonical int-0/1 convention; anchors remain numeric; NaN refused.

## Containment / live-lineage safety (verified, not assumed)

- The LIVE c2 lineage is untouched: `compile_v9_cgauge_ideal_mod19_sR_launch_config`
  defaults `flag_custody=False`; `compile_c2_surgical_warm_launch_config` argv AND
  typed_config_hash verified BYTE-IDENTICAL to pre-change main; c1 optimal-form argv
  byte-identical.
- smoke-regime + iso variants + optimal_basis A/B (`--basis`-only diff preserved) +
  spec_next + micro-batch identity probe all compile green.

## Verdict scope

INSTANCE/apparatus level. The class-4 waivers are a tracked queue (duty-to-measure per the
off-is-orphan rule), not settled science: each flag's waiver retires when a real derivation
law or content-hashed measured anchor is registered. The #332/#406 STRICT flips remain
OWED as a separate reviewed step (this landing produces the live-count-0 precondition).

Triality legs: DSL (`spec_v9_cgauge` custody engine + rollup Lever) · equations
(`dsl_custodied_scalar_identity_v1` registered with evaluator + canonical-equation ledger
row) · this memo (trajectory). Tests: v9 gate / dsl-compile-hash / spec_v9 / lawref /
curriculum-dsl / consumer suites green (pre-existing unrelated failures on main noted:
`test_typed_launcher_dsl_composition` x4 (epochs=5 curriculum feasibility) and
worktree-missing run-dir costate tests — both reproduced at the SAME main HEAD without
these changes).
