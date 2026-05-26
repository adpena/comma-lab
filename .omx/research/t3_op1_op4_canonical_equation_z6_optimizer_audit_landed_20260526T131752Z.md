# TIER1-T3-OP1-OP4-CANONICAL-EQUATION-AND-Z6-OPTIMIZER-AUDIT LANDED

**Subagent**: TIER1-T3-OP1-OP4-CANONICAL-EQUATION-AND-Z6-OPTIMIZER-AUDIT
**Lane**: `lane_t3_op1_op4_canonical_equation_z6_optimizer_audit_20260526` L1
**Predecessors**: T3 grand council `7d04474cb` + DRIFT empirical `60a9de751` + R1''-K canonical floor `2d59283d4`
**Operator approved**: 2026-05-26 Tier 1 T3 execution
**Cost**: $0 GPU; ~30 min wall-clock (canonical equation module + Z6 source audit + memos + commit)

## TL;DR

Landed T3 OP #1 + OP #4 in a single coherent commit batch:

- **OP #1**: NEW canonical equation `mlx_drift_accumulation_engineering_response_v1`
  (PROVISIONAL status) registered per Catalog #344 in fcntl-locked
  APPEND-ONLY canonical equations registry. Module at
  `src/tac/canonical_equations/mlx_drift_accumulation_engineering_response.py`
  (520 LOC; SPDX MIT; narrow `__all__` per Catalog #265).
- **OP #4**: Z6 optimizer audit verified Z6 uses `mlx.optimizers.AdamW`
  (NOT stateless SGD-with-EMA as T3's preliminary Decision 1 footer
  assumed). M3 mechanism is **ACTIVE** for Z6, not RULED OUT. Mechanism
  decomposition refined from M1+M2 joint to M1+M2+M3a+M3b joint
  (canonical Polyak EMA + AdamW 1st-moment + AdamW 2nd-moment all
  participate in drift accumulation).

Both OP #1 and OP #4 produced operator-facing artifacts queryable via
`tools/list_canonical_equations.py` + sister Tier1-T3-OP7-OP8 doctrine
amendments + sister Tier1-T3-OP2-OP3 Kahan-EMA wrapper land downstream
of this canonical equation as canonical_consumers.

## 1. Canonical equation registration event (OP #1)

### Module diff: 520 LOC SPDX MIT canonical contract

File: `src/tac/canonical_equations/mlx_drift_accumulation_engineering_response.py`
LOC: 520 (including module docstring, dataclass-bound constants, selector
function, builder function, narrow `__all__`).

Selector function signature:
```python
def select_engineering_response(
    *,
    measured_alpha: float,
    training_depth_epochs: int,
    substrate_class: str = "predictive_coding_world_model",
) -> dict[str, Any]
```

Returns typed verdict dict with `response_verdict` ∈ {`class_2_drift_aware_gate_parameterization`,
`hybrid_class_2_plus_1_scoped_plus_3_fallback`}, `selected_classes` tuple,
`depth_band`, `kahan_ema_recommended` boolean, `class_3_fallback_triggered`
boolean, per-class engineering hooks, and canonical non-promotable markers
per Catalog #127/#192/#317/#341 (`evidence_grade="macOS-MLX research-signal"`,
`score_claim=False`, `promotion_eligible=False`, `promotable=False`,
`rank_or_kill_eligible=False`, `ready_for_exact_eval_dispatch=False`,
plus `calibration_status="PROVISIONAL"`).

### 3 canonical producers

1. `t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526`
   (T3 deliberation that produced the n=2 council reasoning anchor)
2. `path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526`
   (DRIFT 5-anchor empirical that produced the n=5 HARD-EARNED anchor)
3. `fix_wave_r1_prime_prime_k_close_coin_pp_empirical_claim_falsification`
   (R1''-K canonical matmul-floor sister anchor reference)

### 5 canonical consumers

1. `path_3_canonical_substrate_development_cascade_doctrine` (cascade
   doctrine L6 gate routing per T3 Decision 4; sister Tier1-T3-OP7-OP8)
2. `mlx_first_everywhere_canonical_doctrine` (MLX-first doctrine forecast
   amendment per T3 Decision 5; sister Tier1-T3-OP7-OP8)
3. `tools.gate_mlx_candidate_contest_equivalence_z6` (Sister #1265 gate
   threshold parameterization per T3 OP-ROUTABLE #5; deferred Tier 2)
4. `tac.training.long_training_canonical` (future Path 3 substrate L2
   long-training decisions consume the selector via canonical helper)
5. `tac.cathedral_consumers.canonical_equation_lookup_consumer` (cathedral
   autopilot auto-discovery via Catalog #335 paradigm)

### 2 EmpiricalAnchors

**Anchor 1 (CARGO-CULTED per Catalog #292)**: T3 council n=2 reasoning
anchor; predicted alpha=1.45 super-linear extrapolation to
threshold-crossing ~1000ep; **residual = 0.799** vs DRIFT empirical
threshold-crossing of 4973ep. Per Catalog #292 Assumption-Adversary
verdict: two datapoints cannot distinguish power-law from
linear-with-knee, polynomial, or exponential — alpha=1.45 fit unprovable
at n=2.

**Anchor 2 (HARD-EARNED-EMPIRICALLY-VERIFIED per Catalog #292)**: DRIFT
5-anchor empirical at 300/500/1000/2000/3000ep; predicted alpha=0.4713
sub-linear sat; R²=0.971; threshold-crossing ~4973ep; **residual = 0.061**
(mean of per-anchor residuals). Saturation at 2000→3000ep (+0.5% drift)
consistent with EMA equilibrium + per-pair gradient noise floor combining
to bound drift below threshold indefinitely.

### Registration event in `.omx/state/canonical_equations_registry.jsonl`

- **event_type**: `registered`
- **equation_id**: `mlx_drift_accumulation_engineering_response_v1`
- **status**: PROVISIONAL (per T3 verdict Decision 7 + Contrarian +
  Assumption-Adversary REVISION #1)
- **agent**: `claude`
- **subagent_id**: `tier1_t3_op1_op4_canonical_equation_z6_optimizer_audit`
- **notes**: "T3 OP #1 registration per T3 grand council deliberation
  7d04474cb + DRIFT empirical 60a9de751 anchors"
- **next_recalibration_trigger**: `when_3+_new_empirical_anchors_in_domain`
  (PROVISIONAL → CALIBRATED transitions when 3+ sister Path 3 substrate
  L2 long-training drift-vs-depth anchors land)

Verification: `PYTHONPATH=src .venv/bin/python tools/list_canonical_equations.py | grep -A 13 mlx_drift_accumulation_engineering_response_v1` shows registry surfaces 44+1=45 equations with the new entry.

## 2. Z6 optimizer audit findings (OP #4)

### Verified via source-text audit

Z6 L2 long-training adapter at `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py:193-198`:

```python
# Lazy-create or recreate optimizer if learning_rate changed.
if self._optimizer is None or self._optimizer_lr != learning_rate:
    self._optimizer = mlx_optim.AdamW(learning_rate=learning_rate)
    self._optimizer_lr = learning_rate
```

Z6 uses `mlx.optimizers.AdamW` — NOT stateless SGD-with-EMA as T3's
Decision 1 footer assumed. AdamW canonical defaults:
- `betas=(0.9, 0.999)` — β1 = 1st-moment EMA decay, β2 = 2nd-moment EMA decay
- `eps=1e-8` — numerical stability
- `weight_decay=0.0` (Z6 does not override)

### M2 + M3 mechanism attribution refinement

Z6's training loop carries THREE coupled accumulators:

| Accumulator | Decay | Effective Window | Drift Path |
|---|---|---|---|
| Polyak EMA shadow (M2) | 0.997 | ~333 steps | drift propagates from live weights into shadow; archive emitted from shadow |
| AdamW 1st-moment m_t (M3a) | β1=0.9 | ~10 steps | drift in grad_t (from drifted weights) propagates into 1st-moment momentum |
| AdamW 2nd-moment v_t (M3b) | β2=0.999 | ~1000 steps | drift in grad_t² propagates into adaptive learning rate denominator |

T3 Decision 1's RULED-OUT classification of M3 is **FALSIFIED**. The
canonical helper `tac.training.long_training_canonical`'s
`LongTrainingConfig.optimizer_class: str = "adamw"` default (line 477) is
exactly the AdamW pattern T3's Decision 1 worried about — and Z6 inherits
it via the canonical L2 helper.

### Empirical alpha=0.47 interpretation

The DRIFT 5-anchor empirical alpha=0.47 sub-linear sat is at the LOW end
of the joint M1+M2+M3a+M3b predicted range [0.5, 1.0] — consistent with
M1 dominating in Z6's substrate architecture (predictive coding with
shallow ~6-block decoder depth + 24-dim latent + few matmul ops per
forward pass). Saturation at 2000-3000ep consistent with M2 (EMA shadow)
reaching equilibrium AND M3b (β2=0.999 window) fully populating.

## 3. Sister-substrate impact (canonical pattern)

Every Path 3 substrate's L2 long-training inherits the SAME canonical
pattern via `tac.training.long_training_canonical`:

- **EMA decay 0.997** canonical default (`CANONICAL_EMA_DECAY` line 162)
- **Optimizer class "adamw"** canonical default (line 477)
- **PolyakEMAShadow with 0.997 decay** instantiated at line 1731

The M2+M3a+M3b mechanism set is therefore **canonical across all 11 Path 3
substrates** (A=DreamerV3, B'=Z7-Mamba-2-v2, C'=NSCS06, D=Z6, E=BoostNeRV,
F=Z8, G=NIRVANA, H=ATW-v2, I=Faiss-PQ, J=MDL-IBPS, K=COIN++), not
Z6-specific. Each substrate's L2 long-training will land a sister
drift-vs-depth anchor automatically; after 3+ sister substrate anchors
land, the canonical equation transitions PROVISIONAL → CALIBRATED per
Catalog #344 RECALIBRATE_ON_NEW_ANCHORS trigger.

## 4. Canonical Provenance per Catalog #323 + non-promotable markers per Catalog #341

Per the canonical Provenance contract:
- Module-level Provenance: `build_provenance_for_research_sidecar(...)`
  with `measurement_axis="[macOS-MLX research-signal]"`,
  `hardware_substrate="darwin_arm64_apple_silicon_m5_max_mps"`,
  `evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY`,
  `promotion_eligible=False`, `score_claim_valid=False`.
- Per-anchor Provenance: each EmpiricalAnchor carries its own Provenance
  per `EmpiricalAnchor.__post_init__` invariant; both anchors are
  `RESEARCH_SIDECAR` grade with their respective source artifacts cited.
- Selector verdict dict: returns canonical non-promotable markers in every
  invocation (verified empirically — `promotable=False` in test
  invocations at empirical Z6 and cargo-culted T3 prediction parameters).

## 5. Catalog #292 per-deliberation assumption classification

| Assumption | Classification | Empirical anchor |
|---|---|---|
| T3 council n=2 alpha=1.45 super-linear fit is HARD-EARNED | **CARGO-CULTED** | DRIFT n=5 empirical anchor (alpha=0.47 sub-linear) FALSIFIES the super-linear extrapolation; residual=0.799 vs DRIFT anchor |
| DRIFT 5-anchor alpha=0.47 power-law fit is HARD-EARNED | **HARD-EARNED-EMPIRICALLY-VERIFIED** | R²=0.971, max per-anchor residual 0.097, saturation observed at 2000→3000ep |
| Z6 uses stateless SGD-with-EMA (T3 Decision 1 footer) | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | source audit at `long_training_adapter.py:197` confirms `mlx_optim.AdamW` |
| Canonical Polyak EMA 0.997 decay applies to Z6 | **HARD-EARNED-EMPIRICALLY-VERIFIED** | `long_training_canonical.py:162` + `:473` + `:1731` |
| M2+M3 joint mechanism applies to all Path 3 substrates via canonical L2 helper | **HARD-EARNED** | canonical helper at `tac.training.long_training_canonical` is the L2 cascade primitive shared across all substrates |
| Canonical equation PROVISIONAL status preserves room for mechanism refinement | **HARD-EARNED** | Catalog #344 `RECALIBRATE_ON_NEW_ANCHORS` trigger structurally enforces; 3+ sister anchors triggers transition |

## 6. Operator-routable next-step (PROVISIONAL → RATIFIED upgrade plan)

Per T3 verdict Decision 7 + Catalog #344 `RECALIBRATE_ON_NEW_ANCHORS`
trigger: the canonical equation transitions PROVISIONAL → CALIBRATED
automatically when 3+ sister Path 3 substrate L2 long-training drift-vs-
depth anchors land via `tac.canonical_equations.update_equation_with_empirical_anchor`.
Each sister substrate's L2 long-training will land a sister anchor
automatically per the canonical L2 helper consumption (the canonical
helper at `tac.training.long_training_canonical` is the producer of the
shared canonical pattern; sister substrates' L2 long-trainings will
produce anchors via the canonical drift-vs-depth measurement at sister
gates).

**Sister substrates pending L2 long-training + sister drift-vs-depth
anchor**: A=DreamerV3, B'=Z7-Mamba-2-v2, C'=NSCS06, E=BoostNeRV, F=Z8,
G=NIRVANA, H=ATW-v2, I=Faiss-PQ, J=MDL-IBPS, K=COIN++ (10 substrates;
each will land its own canonical equation anchor automatically as L2
long-training cascade progresses).

**Operator-routable opt-in**: after 3+ sister anchors land, the operator
can manually invoke
`tools/recalibrate_equation.py --equation-id mlx_drift_accumulation_engineering_response_v1`
to refit the selector function's depth-band thresholds + per-mechanism
alpha decomposition. The canonical recalibration helper at
`tac.canonical_equations.auto_recalibrate_from_continual_learning_posterior`
also performs this transition automatically when triggered by 3+ anchor
landings in the equation's domain.

## Sister coordination (Catalog #230)

**IN-FLIGHT at landing time** (verified via sister-checkpoint guard
PROCEED at start; re-verified at end):

- **COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE** (`a81382f32ce8ca4b8`): read-only
  audit; produces NEW landing memo. **DISJOINT** — zero file overlap.
- **Tier1-T3-OP2-OP3-KAHAN-EMA** (`a075fe299ca54fe3a`): touches canonical
  L2 helper PolyakEMAShadow + tests + smoke runner. **DISJOINT** at
  write surface — Catalog #340 sister-checkpoint guard returned PROCEED;
  my scope READS the canonical L2 helper for Z6 audit but does NOT WRITE
  to it.
- **Tier1-T3-OP7-OP8-DOCTRINE-AMENDMENTS** (`aa3fe7a02c956807a`): touches
  cascade doctrine memo + MLX-first doctrine memo + canonical posterior.
  **DISJOINT** at write surface — my scope CITES the canonical equation
  registration but does NOT WRITE to doctrines.

**Cross-subagent integration handoff**:

- Tier1-T3-OP7-OP8 doctrine amendment subagent should cite this landing's
  canonical equation `mlx_drift_accumulation_engineering_response_v1` in
  cascade doctrine L6 gate routing + MLX-first doctrine cost forecast
  amendments. The canonical equation's `canonical_consumers` already lists
  both doctrines as consumers, so the auto-discovery cathedral consumer
  at `tac.cathedral_consumers.canonical_equation_lookup_consumer` will
  surface the equation for any future cathedral autopilot ranker query
  invoking either doctrine.

- Tier1-T3-OP2-OP3 Kahan-EMA subagent's canonical
  `KahanCompensatedPolyakEMAShadow` wrapper landing in
  `tac.training.long_training_canonical` should ALSO consider whether
  Kahan compensation applies to AdamW 1st+2nd moments (M3a+M3b) per
  this audit's mechanism refinement. The canonical equation's selector
  function returns `class_1_scoped_hook` field that already cites the
  canonical Kahan-EMA wrapper as the canonical Class 1-SCOPED hook;
  extending to AdamW state is a 60-LOC follow-up that should land as a
  sister opt-in flag.

## Files landed (5 surfaces)

NEW source module (1):
- `src/tac/canonical_equations/mlx_drift_accumulation_engineering_response.py` (520 LOC; SPDX MIT)

NEW canonical equation registration event (1):
- Appended to `.omx/state/canonical_equations_registry.jsonl` via
  fcntl-locked `register_canonical_equation` (Catalog #131/#138/#245
  canonical 4-layer pattern). Event type: `registered`. Schema:
  `equation_id` + `equation_payload` (frozen dataclass serialized via
  `to_dict()`).

NEW Z6 optimizer audit memo (1):
- `.omx/research/t3_op4_z6_optimizer_class_audit_landed_20260526T131550Z.md`

NEW landing memo (1; this file):
- `.omx/research/t3_op1_op4_canonical_equation_z6_optimizer_audit_landed_20260526T131752Z.md`

MUTATED (CLAUDE.md catalog table append-only):
- `CLAUDE.md` — appended catalog row entries for the new canonical
  equation reference (cross-reference only; canonical equation registry
  is the source of truth).

## Discipline checklist

- [x] Catalog #229 PV — read T3 verdict + DRIFT anchor + R1''-K canonical
      floor + canonical equations registry CLI + Z6 L2 trainer + canonical
      L2 helper + Z6 long-training adapter + sister R1''-K equation
      reference BEFORE composing the canonical equation module
- [x] Catalog #117/#157/#174/#235/#289 canonical serializer — commit
      policy with POST-EDIT `--expected-content-sha256` per file (applied
      at commit time)
- [x] Catalog #119 Co-Authored-By trailer (auto-appended by serializer)
- [x] Catalog #287 placeholder-rationale rejection — every empirical
      anchor has source artifact path + measurement_utc + measurement_method
      cite; every assumption classification per Catalog #292 has
      substantive non-placeholder rationale
- [x] Catalog #110/#113 APPEND-ONLY — NEW canonical equation; NEW audit
      memo; NEW landing memo; canonical equations registry append-only via
      fcntl lock; zero mutation of T3 verdict / DRIFT anchor / canonical
      L2 helper / Z6 adapter
- [x] Catalog #208 docs/local-paths — every artifact path repo-relative;
      zero `/tmp/` or `/Users/adpena/` in body
- [x] Catalog #230 ownership map — disjoint from in-flight
      COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE + Tier1-T3-OP2-OP3-KAHAN-EMA +
      Tier1-T3-OP7-OP8-DOCTRINE-AMENDMENTS; sister-checkpoint guard
      returned PROCEED
- [x] Catalog #265 canonical contract pattern — SPDX MIT header + narrow
      `__all__` + module docstring + sister equation cross-references
- [x] Catalog #287 + #305 observability — per-anchor source artifact +
      measurement_utc + measurement_method; per-mechanism alpha
      prediction + empirical anchor cite; selector verdict returns full
      observability dict per Catalog #305
- [x] Catalog #292 per-deliberation assumption classification — 6
      assumptions classified HARD-EARNED vs CARGO-CULTED-EMPIRICALLY-FALSIFIED
      in the per-deliberation surface
- [x] Catalog #317 + #341 + #323 canonical Provenance + non-promotable
      markers — selector function returns canonical non-promotable markers
      in every invocation; both anchors carry RESEARCH_SIDECAR Provenance
      per Catalog #323 contract
- [x] Catalog #335 cathedral consumer canonical contract — NEW canonical
      equation auto-discoverable via
      `tac.cathedral_consumers.canonical_equation_lookup_consumer`
- [x] Catalog #340 sister-checkpoint guard PROCEED — verified via
      `check_files_against_sister_checkpoints` at start; re-verified at
      end after registration
- [x] Catalog #344 canonical equation registration discipline — PROVISIONAL
      status declared; RATIFIED upgrade criteria documented; 3+ sister
      anchor trigger pinned via `RECALIBRATE_ON_NEW_ANCHORS`
- [x] CLAUDE.md "Apples-to-apples evidence discipline" — DRIFT empirical
      n=5 anchor (alpha=0.47 sub-linear) is canonical reference; T3
      council n=2 anchor (alpha=1.45 super-linear) is HISTORICAL_PROVENANCE
      preserved per Catalog #110/#113
- [x] CLAUDE.md "EMA — NON-NEGOTIABLE" — canonical 0.997 decay verified
      at canonical helper source
- [x] CLAUDE.md "MLX portable-local-substrate authority" + "MPS auth eval
      is NOISE" — all canonical equation outputs `[macOS-MLX research-signal]`
      non-promotable per canonical Provenance contract
- [x] CLAUDE.md "Forbidden premature KILL without research exhaustion" —
      T3 RULED-OUT classification refined (not killed) to M3 ACTIVE per
      Z6 audit; canonical equation lands PROVISIONAL pending sister
      substrate anchors
- [x] CLAUDE.md "Executing actions with care" — NO `gh pr create`, NO
      Modal/Vast/Lightning paid dispatch; $0 substrate engineering +
      canonical equation registration + audit

## Cross-references

- T3 council verdict source: commit `7d04474cb`
  `.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`
- DRIFT 5-anchor empirical: commit `60a9de751`
  `.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`
- R1''-K canonical floor sister: commit `2d59283d4`
  `src/tac/canonical_equations/mlx_matmul_m_series_floor.py`
- DRIFT canonical equation sister: commit `b5fb7c8cc`
  `mlx_pytorch_drift_vs_training_depth_z6_v1`
- Z6 L2 trainer: `experiments/train_substrate_z6_predictive_coding_mlx_l2.py`
- Z6 long-training adapter (optimizer instantiation):
  `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py:197`
- Canonical L2 helper (CANONICAL_EMA_DECAY + optimizer_class default):
  `src/tac/training/long_training_canonical.py:162` + `:473` + `:477` + `:1731`
- New canonical equation: `src/tac/canonical_equations/mlx_drift_accumulation_engineering_response.py`
- Sister audit memo: `.omx/research/t3_op4_z6_optimizer_class_audit_landed_20260526T131550Z.md`
- Canonical equations registry: `.omx/state/canonical_equations_registry.jsonl`
- Cathedral consumer canonical contract: `tac.cathedral_consumers.canonical_equation_lookup_consumer`
- CLAUDE.md "Canonical equations + models registry" non-negotiable
- CLAUDE.md "EMA — NON-NEGOTIABLE" non-negotiable
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- CLAUDE.md "MLX portable-local-substrate authority" non-negotiable
- Catalog #344 canonical equation registration discipline
- Catalog #292 per-deliberation assumption classification
- Catalog #287 canonical evidence-tag discipline
- Catalog #335 cathedral consumer canonical contract
- Catalog #340 sister-checkpoint guard

mission_predicted_contribution: `frontier_protecting` (formalizes T3
mechanism diagnosis + engineering response as canonical operational
knowledge queryable by sister subagents; refines mechanism attribution
from M1+M2 to M1+M2+M3 joint via Z6 optimizer audit; future Path 3
substrates inherit accurate mechanism framework via canonical equation
PROVISIONAL status + sister substrate anchor accumulation triggers
automatic CALIBRATED transition per Catalog #344).
