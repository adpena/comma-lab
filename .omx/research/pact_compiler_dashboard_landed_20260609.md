# Vehicle-OS compiler dashboard generator + evidence-assigned L0-L7 maturity — LANDED 2026-06-09

UTC 2026-06-09 · claude (subagent `pact_dashboard_maturity_20260609`, executor #42) · per operator
binding directive 2026-06-09 + `docs/vehicle_operating_system.md` "Dashboard discipline"
(*"every turn begins with the dashboard; no stale-memory decisions"*).

## What landed

1. **`src/comma_lab/pact_compiler_dashboard.py`** — the reusable, fail-soft dashboard renderer
   (AGENTS.md "tac stays clean; comma-lab owns research state": research-state / operator-facing
   dashboards live in `comma_lab`, not `tac`). Reads ONLY machine-readable sources:
   - `vehicle_fidelity` (`.omx/state/vehicle_fidelity/*.json`) — claimed-vs-actual mechanism identity;
   - `objective_reachability` (`.omx/state/objective_reachability/*.json`) — VJP/weight reachability;
   - `constants_provenance` (`.omx/state/constants_provenance/*.json`) — `declared_maturity_level`;
   - the canonical frontier pointer — scores are **POINTER-ONLY, never hardcoded** (CLAUDE.md
     "Frontier scores are pointer-only"); a missing pointer yields an explicit `POINTER_MISSING`
     status, not a fabricated score;
   - the latest typed verdict JSONs on the SSD tier (G1b export-binding verdict +
     `candidate_action_evaluation_g1b_*`, pr110pp byte-closure/no-op proofs, the receiver-closed
     ladder) — fail-soft to `AUDIT_PENDING` when the SSD tier is detached;
   - `subagent_progress.jsonl` — the live running daemons/agents (latest-row-wins per id).

2. **`tools/render_pact_compiler_dashboard.py`** — the thin CLI. REPLACED the prior carrier-registry
   prose-only generator (which wrote `.omx/research/pact_compiler_dashboard.md` from
   `composition_carrier_registry`) with a manifest-driven generator that emits
   `pact_compiler_dashboard.{json,md}` at the **repo root** (committed). The prior carrier-registry
   triage + active-heartbeat views are PRESERVED as supplementary appended Markdown sections (no
   signal lost; `--no-triage` to omit). `--print json|md` for dry-run/piping.

3. **`pact_compiler_dashboard.{json,md}`** (repo root, committed) — the generated artifact.

4. **`tests/test_comma_lab_pact_compiler_dashboard.py`** — 21 behavioral tests (NO-FAKE Class-2: each
   would fail if the generator regressed to a constant-emitter): fresh-checkout safety, pointer-only
   scores (change pointer -> change score), missing-pointer explicit status, missing-manifest fail-soft
   AUDIT_PENDING rows, laundering->L0, present-mechanism->L1, zero-mechanism sketch->L0,
   reachability-severance-in-blocker, snerv >=L2 floor (L4 when the SSD CAE row is mounted),
   pr110pp byte-closed->L3, pr110pp-without-candidate < L3, infrastructure n/a-vehicle row,
   allowed-claim<->ladder wiring, live-work latest-row-wins + complete-excluded, corrupt-log fail-soft,
   write-both-files, machine-readable schema_gaps.

## The per-vehicle dashboard rows (maturity FROM EVIDENCE; cite per assignment)

| vehicle | L | allowed_claim | authority_tier | metric_family | evidence / blocker |
|---|---|---|---|---|---|
| **snerv** | **L4** | exact_scored_row_exists | exact_cpu_advisory | exact_pair_scorer | `candidate_action_evaluation_g1b_pathb_ep273.v1.json`: d_seg=0.00247, d_pose=0.00203, bytes=581,583,207, pays_rent=False. Fidelity clean (real MFU/HFR/TUB + orthonormal DWT), reachability clean (both VJPs reach; weights 7.24/7.0). Blocker = 100% rate (581.6MB skip_high float64 LL=99.9996%); export bound, rate chasm -> LF entropy-coding front. **NOT L5** (no contest-axis paired row, pays_rent=false). |
| **hi_nerv** | **L1** | mechanism_present_unit_tested | AUDIT_PENDING | AUDIT_PENDING | vehicle_fidelity present=`['bilinear_skip']` (opt-in, OFF by default); reachability FAILS at **weight** surface (shared MLX harness defaults SegNet/PoseNet weights to 0.0). constants_provenance declares L1. |
| **pact_nerv_vq** | **L1** | mechanism_present_unit_tested | AUDIT_PENDING | AUDIT_PENDING | vehicle_fidelity present=`['codebook_vq']` (genuine STE+EMA+commitment); reachability FAILS at weight surface (MLX-route VJP AUDIT_PENDING, recon-MSE-by-default). |
| **sane_hnerv** | **L0** | research_carrier_sketch | AUDIT_PENDING | AUDIT_PENDING | NAME-LAUNDERING (documentation-fake): docstring advertises bilinear-skip the forward never implements -> `verify()` RAISES. |
| **ff_nerv** | **L0** | research_carrier_sketch | AUDIT_PENDING | AUDIT_PENDING | zero present mechanisms (honest sketch); skip-free + band-limited DCT grid cannot represent boundary HF. |
| **pr110pp** | **L3** | archive_real_byte_closed_consumed | exact_cpu_advisory | advisory_pose_delta | `byte_closure_proof.json` (178,493 B) + `noop_detector.json` (consumption_proven=true, 5.6M differing raw bytes). Blocker = L4 exact row in flight (R1 paired contest-CPU Modal eval dispatched). |
| **atlas_atoms_v3** | **n/a-vehicle** | infrastructure | n/a | n/a | spectral atlas + V3 (frozen_evaluator_contract) are the measured-law + DeltaS-judge kernels every vehicle consumes; not on the L0-L7 ladder. |

## Maturity assignments where evidence DISAGREED with the prompt priors

- **snerv = L4** (prompt prior said L4): AGREES. The exact `CandidateActionEvaluation` row
  (advisory tier, exact_pair_scorer) is the decisive L4 evidence — even though it does not pay rent.
- **hi_nerv**: prompt said L0/L1. Evidence supports **L1** specifically — the vehicle_fidelity manifest
  has one genuine present mechanism (`bilinear_skip`, opt-in) AND constants_provenance explicitly
  declares L1. The strategic memo calls it "L0 sketch"; the dashboard records L1 with the disagreement
  surfaced (the L1 is on the generous edge — a single opt-in mechanism OFF by default + a severed
  objective). No contradiction with the prompt's "L0/L1".
- **pact_nerv_vq = L1** (prompt prior said L1): AGREES (genuine VQ mechanism; MLX-route reachability
  FAILS at weight surface).
- **sane_hnerv = L0** (prompt prior said L0): AGREES (laundering corrected -> the name is not a claim).
- **ff_nerv = L0** (prompt prior said L0): AGREES.
- **pr110pp = L3** (prompt prior said "L3-L4 pending R1"): AGREES — byte-closed candidate + no-op proof
  exist (L3 archive-real); the L4 exact row is dispatched (Modal CPU eval) but not yet a confirmed
  `CandidateActionEvaluation`, so L3 is the evidence-justified level with "L4 in flight" as the blocker.
- **atlas_atoms_v3 = n/a-vehicle** (prompt prior said "infrastructure"): AGREES.

## SCHEMA GAP (task #3 outcome — noted, NOT hand-edited)

The prompt asked to update each `vehicle_fidelity` manifest's `maturity_level` field via the canonical
emitter "where the schema supports it; else note the schema gap." **The `vehicle_fidelity_manifest.v1`
schema has NO `maturity_level` field** (zero `maturity` references in the module; confirmed). So:

- I did NOT hand-edit any manifest JSON (forbidden) and did NOT unilaterally add a schema field (a
  council/operator design decision per CLAUDE.md "Design decisions — non-negotiable" + the manifests'
  APPEND-ONLY HISTORICAL_PROVENANCE adjacency).
- The gap is surfaced as a **machine-readable `schema_gaps` entry** in the dashboard JSON (+ a "Schema
  gaps" Markdown section): remediation = council/operator-approved `maturity_level` field on
  `VehicleFidelityManifest`, OR standardize on `constants_provenance.declared_maturity_level` for ALL
  vehicles (only hi_nerv has a constants manifest today).
- The dashboard therefore DERIVES maturity from the union of fidelity + reachability + constants +
  typed verdict rows, which is the correct evidence-driven behavior until the schema gap is closed.

## Generator command

```bash
.venv/bin/python tools/render_pact_compiler_dashboard.py            # write {json,md} at repo root
.venv/bin/python tools/render_pact_compiler_dashboard.py --print md # dry-run to stdout
```

## 6-hook wire-in (Catalog #125)

1. sensitivity-map — N/A (dashboard is an aggregator, not a score actuator).
2. Pareto constraint — N/A.
3. bit-allocator — N/A.
4. cathedral autopilot dispatch — ACTIVE-adjacent: the JSON is the machine-readable surface the
   autopilot/next-subagent consumes to choose the next vehicle (per OS "Dashboard discipline").
5. continual-learning posterior — N/A (no empirical score landed; pointer-only read).
6. probe-disambiguator — N/A.

`council_predicted_mission_contribution`: `apparatus_maintenance` (the dashboard is the no-stale-memory
substrate that frontier-breaking work routes through; it does not itself move the score).

## Notes / non-fakery

- All scores POINTER-ONLY (never hardcoded); the live dashboard read 0.19199 [contest-CPU] /
  0.20533 [contest-CUDA] from the pointer at generation time.
- All `[macOS-CPU advisory]` / `exact_cpu_advisory` rows are non-promotable (snerv pays_rent=false;
  pr110pp advisory pose gain only) — consistent with the apples-to-apples + MLX-authority discipline.
- One pre-existing unrelated test failure observed in
  `src/tac/substrates/_shared/tests/test_compact_decoder_codec_sweep.py` (`_decoder_state_codec`
  KeyError; a decoder-codec sweep, no reference to my module or the manifests). Out of scope for this
  landing; not introduced by this change.
