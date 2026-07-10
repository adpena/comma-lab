# Design philosophies → LIVE surfaces (#387 extension) — landing memo

**Date:** 2026-07-09 · **Axis:** all numbers [macOS-CPU advisory] NON-PROMOTABLE · **Pointer 0.19110
UNMOVED** (apparatus/means — this makes the campaign harder to fool; it does not move the score).

**Operator GO (2026-07-09):** *"Any other surfaces our design philosophies should be encoded to? Costate
controller?"* + addendum *"fmtools (#259) is available for hard/fuzzy classification ... make it durable in
every composed dispatch prompt"* + addendum *"P9 landed — proxies are poison, use the thing itself; cite
floors in the SAME units as the Δ they gate."*

Sibling of `.omx/research/eightfold_apparatus_build_20260709.md` (the #387 sibling landed the P1/P4 GATES).
This unit landed the LIVE wirings those gates guard. Source discipline: `design_philosophies_eightfold_
20260709` (P1-P8) + `costate_controller_design_20260705.md` + `docs/operating_manual_craft_handoff.md`.

## STORES CONSULTED
- memory `design_philosophies_eightfold_20260709` (the 8 + clauses A/B + apparatus routing)
- `.omx/research/costate_controller_design_20260705.md` (the costate SENSE+DECIDE architecture)
- `docs/operating_manual_craft_handoff.md` (verify-by-re-deriving; label MEASURED/DERIVED; answer first)
- `.omx/state/lever_relative_significance.jsonl` (the live significance store — the real BEFORE/AFTER)
- `.omx/state/canonical_frontier_pointer.json` (0.19110 — read live, never hardcoded)
- canonical_equations `anisotropic_basis_two_regime_allocation_v1` (d_seg 0.00087 floor),
  `independent_flicker_jitter_dseg_floor_smooth_optimal_v1` (#141 label-noise identity),
  `information_theoretic_floor` (S_floor 0.11797 LOOSE) — the floor provenance
- `src/tac/witness_control/sigma_min_plateau.py::canary_suite` (the P4 canary pattern)
- `.omx/research/sub015_DAG_...md` FEED-eightfold-apparatus (the P4 gate live-count-1 = VerdictTrendAlarm)

## The four wirings (each: build + tests + byte-identity where applicable)

### 1. Costate controller (the operator-named surface)
- **1a — P8 floor-aware duty-to-measure ranking.** New module `src/tac/witness_dsl/term_floors.py`:
  `FloorSpec` (value-provenance ladder MEASURED / MEASURED_REGIME_DEPENDENT / LOOSE / OWED_UNMEASURED) +
  `resolve_term_floors()` (d_seg 0.00087 MEASURED conservative bound · rate 0.11797 LOOSE surface-only ·
  d_pose OWED None) + `apply_term_floor()` (AT_FLOOR / HEADROOM_CAPPED / ABOVE_FLOOR /
  FLOOR_KNOWN_CURRENT_UNKNOWN / FLOOR_UNMEASURED). `duty_to_measure_ranked()` gained optional
  `term_current` / `term_floors` / `floor_aware` (default True). **Backward-compatible by construction:**
  with no `term_current`, rel_sig + ordering are UNCHANGED (only floor metadata is added) — the existing
  70 ranking/ledger tests stay green. Only a clean numeric MEASURED floor + a measured current-term value
  caps est / fires AT_FLOOR (NO-FAKE: regime-dependent / LOOSE / owed floors are surfaced, never guessed,
  never change the ranking).

  **MEASURED BEFORE/AFTER top-10 diff (real live store):**

  | rank | BEFORE (floor-unaware) | AFTER live (floor-aware, no live d_seg) | AFTER demo (d_seg AT floor 0.00087) |
  |------|------------------------|------------------------------------------|--------------------------------------|
  | 1 | DsegAwareTaper 73% | DsegAwareTaper 73% `FLOOR_KNOWN_CURRENT_UNKNOWN` | latent_table_truncate 2.4% (rate) |
  | 2 | HorizonWeightedMargin 43.8% | HorizonWeightedMargin 43.8% `…CURRENT_UNKNOWN` | mod32_neutrality 1.2% (rate) |
  | 3 | StepNativeActivation 31.6% | StepNativeActivation 31.6% `…CURRENT_UNKNOWN` | DsegAwareTaper **0.0 AT_FLOOR** |
  | 4 | latent_table_truncate 2.4% | latent_table_truncate 2.4% `FLOOR_UNMEASURED` | HorizonWeightedMargin **0.0 AT_FLOOR** |
  | 5 | mod32_neutrality 1.2% | mod32_neutrality 1.2% `FLOOR_UNMEASURED` | StepNativeActivation **0.0 AT_FLOOR** |

  **Honest consequence:** the LIVE order is UNCHANGED — the witness's current per-term d_seg is an OWED
  measurement when no run is live (exactly the P8 prediction: floors + current values are derived when
  cornered, not free). What changed: every d_seg row now surfaces the MEASURED floor 0.00087 with its
  provenance, every rate row surfaces the LOOSE floor, d_pose is surfaced as OWED — the duty-to-measure-
  the-floor/current is now first-class + queryable. The demo control proves the machinery: the moment a
  measured witness d_seg reaches the floor, d_seg levers auto-derank to ~0. The digest feeds the live
  annulus `overall_d_seg` as `term_current`, so this activates automatically on the next live run.
- **1b — P2 noise-floor on recommendations.** `costate_digest.section_shadow` tags each rec by its
  `predicted_dS_band`: absent → `[INSTANCE — no noise floor]`; band spans 0 → `[INSTANCE — Δ within noise
  floor]`; else `(floor [lo,hi])`. P9: the band gates `predicted_dS` in the SAME ΔS units as the Δ (no
  proxy-unit floor — the R7 category-error lesson). The digest is advisory-only, so "never escalate above
  advisory" holds structurally.
- **1c — P1 canonical-key reads.** Audited: the digest's ONLY significance-store read is
  `duty_to_measure_ranked` → `canonicalize_significance_keys`; no raw-key read exists (documented in the
  section docstring). No change needed.

### 2. Byte-close auto-dedup (clause A live)
`tools/levelset_byte_close_and_eval.py::dedup_audit_section` wires `movable_deshare.pairwise_dedup_audit`
into the report as an automatic per-run geometric-section derivability table (sources `lstars` from
`--gt-cache`, bounded to 4 frames for a $0 read-only pass). **Score-neutral, read-only, fail-open,
byte-identical** — it adds `report["dedup_audit"]`, never mutates archive.zip / the blob / any packet byte.
Observability defaults ON (CLAUDE.md "'Off' is a tracked queue").

### 3. Subagent contract
`tac.subagent_contract.EIGHTFOLD_CLAUSE` (one paragraph: P1-P8 + clauses A/B + the fmtools #259
advisory-availability sentence per the operator addendum — advisory classifier where regex/name heuristics
are uncertain, NEVER sole authority on score-relevant decisions) composed into `standard_contract()`.
`check_subagent_contract_module_integrity` key-phrases + `_SUBAGENT_CONTRACT_REQUIRED_CONSTANTS` extended so
the clause is protected. (Fixed a pre-existing stale block-count assertion — 13 → 15 separators; the
COMMIT_DISCIPLINE block had already made it 14 before this clause.)

### 4. Equations schema (P2 structural)
`EmpiricalAnchor` gains optional `noise_floor` + `noise_floor_provenance` (additive; absent = legacy None;
validated non-negative + requires provenance — NO-FAKE never a guessed floor; emitted in `to_dict` only
when set → byte-stable legacy serialization; round-trips through `registry._equation_from_dict`).
`delta_exceeds_floor(anchor, delta=None)` verdict-clearance helper: `None` when the floor is UNMEASURED
(never silently treated as 0, which would falsely clear every Δ), else `|delta| > noise_floor`.
**Backfill:** the two residual-kit anchors landed today (`dseg_aware_fourier_taper` #121,
`horizon_weighted_margin` #169) keep `noise_floor=None` — their memos state an oracle-ceiling RANGE
(0.012–0.024, a margin-threshold sensitivity) and qualitative "irreducible label-noise", NOT a single-axis
measurement noise floor; our single-seed deterministic spine leaves across-seed variance UNKNOWN (P2). The
floor is an OWED measurement — surfaced-as-None, not fabricated.

### Sibling addendum resolved (P4 canary)
Added `verdict_trend_alarm.canary_suite` + `synthetic_decoupling_verdicts` (MUST fire
TRAIN_VERDICT_DECOUPLING) + `synthetic_codescending_verdicts` (MUST NOT fire) + 3 tests → the #387 P4 gate
`check_witness_control_meters_have_canaries` live count **1 → 0**. (The earlier brief mislabeled
VerdictTrendAlarm a passing exemplar; re-derivation confirmed it carried no canary — now it does.)

## Verification
- 41 new tests (`test_term_floors_ranking.py` 13 · `test_empirical_anchor_noise_floor.py` 6 ·
  `test_subagent_contract_eightfold.py` 3 · `test_byte_close_dedup_section.py` 3 · canary 3 + assorted).
- 238-test regression green across every touched surface (ranking, ledger, witness_control, canonical
  equations, subagent contract, movable_deshare).
- ruff F clean on all touched files (pre-existing registry.py F401/F811 at lines 61-62/503-506 are NOT in
  my diff and left untouched). Contract-integrity + meter-canary strict PASS. Digest `--json` rc=0.

## Triality
- **DAG** = FEED-philosophies-live (this landing's trajectory point + the BEFORE/AFTER diff).
- **DSL** = N/A — `term_floors` / `activation_ledger` are the costate SENSE apparatus, not a trainer Lever
  or curriculum object (nothing to compile to trainer argv).
- **Equations** = the `EmpiricalAnchor.noise_floor` schema extension itself (P2 structural law-carrier
  change; no new S_τ anchor is owed — a floor with no measurement is honestly None).

## Wire-in hooks (Catalog #125)
sensitivity-map N/A · Pareto N/A · bit-allocator N/A · cathedral-autopilot: the floor-aware ranking + the
dedup section ARE consumed by the always-on `costate_digest` (SessionStart hook) + the byte-close report ·
continual-learning: the floors + noise_floor schema become queryable system intelligence · probe-
disambiguator N/A (no 2+ defensible-interpretation fork). Not a score-mover — pointer UNMOVED, stated
plainly.
