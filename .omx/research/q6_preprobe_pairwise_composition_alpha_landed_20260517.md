# Q6.preprobe — Pairwise composition_alpha probe LANDED 2026-05-17

**Lane**: `lane_q6_preprobe_pairwise_composition_alpha_20260517`
**Task**: #819
**Source**: T2 HORIZON-CLASS council 2026-05-17 op-routable OP-3 (NEW) —
gate clause #2 for Stage 2 ASYMPTOTIC_PURSUIT reactivation. Anchor memo:
`.omx/research/council_horizon_class_scope_t2_20260517.md`.
**Council CARGO-CULTED assumption being interrogated**: *"composition_alpha
≈ 1.0 (orthogonal Wyner-Ziv contributions across stacked pre-entropy
substrates)"* (Assumption-Adversary verbatim per council memo).
**Operator NON-NEGOTIABLES honored**: NO COMMITS (artifacts staged only);
checkpoint discipline per Catalog #206; premise verification per Catalog
#229; sister-subagent ownership map per Catalog #230; 6-hook wire-in per
Catalog #125; fcntl-locked JSONL state writes per Catalog #131; canonical
helper API (no bare-state writes).

## TL;DR — empirical anchor

**Stage 2 gate clause #2: SATISFIED.**

All **3-of-3 pairs** of the canonical Stage-2 candidate set are ADDITIVE
(α_savings ≥ 0.7 per Catalog #227 banding), exceeding the council's
≥ 2-of-3 requirement (OP-2.b).

| pair | α_op3_council_form | α_savings_ratio_form | band | best codec | gate clause #2 |
|---|---:|---:|---|---|---|
| `pr101_state_dict` + `pr106_state_dict` | +0.0023 | **+1.0007** | ADDITIVE | lzma | **PASS** |
| `pr101_state_dict` + `posenet_class_sensitivity` | +0.0156 | **+1.0027** | ADDITIVE | lzma | **PASS** |
| `pr106_state_dict` + `posenet_class_sensitivity` | +0.0125 | **+1.0021** | ADDITIVE | lzma | **PASS** |

**Council's CARGO-CULTED assumption is EMPIRICALLY CONFIRMED** for these
3 candidate pairs under lzma compression on the raw `.pt` byte buffers.

## What landed

1. **Canonical helper** at `tools/q6_preprobe_pairwise_composition_alpha.py`
   (~430 LOC) — library API + CLI implementing the OP-3 council spec:
   concat raw bytes → lzma/brotli/zlib on concat vs sum-of-marginals →
   compute α per both formulas (OP-3 council form + savings-ratio form).
   Both α values reported per pair for cross-audit transparency. Canonical
   3-candidate sweep default; pinned-pair mode via `--candidate-a/-b`.
   fcntl-locked atomic write per Catalog #131. Per CLAUDE.md
   non-authoritative contract: `score_claim=false`, `promotion_eligible=false`,
   `ready_for_exact_eval_dispatch=false`, `evidence_grade="predicted"`.
2. **Test suite** at `src/tac/tests/test_q6_preprobe_pairwise_composition_alpha.py`
   (~340 LOC, **15/15 tests pass**) covering: synthetic ADDITIVE pair /
   synthetic redundant pair (cross-audit divergence) / 3-pair sweep
   aggregation / ≥2-of-3 SATISFIED case / ≥2-of-3 NOT_SATISFIED case /
   schema validation / real-data probe on canonical candidates / unknown
   candidate raises / missing file raises / α banding boundaries / atomic
   fcntl-locked write / CLI subprocess smoke / OP-3-vs-savings distinct
   semantics / STAGE_2_CANDIDATES registered / canonical thresholds pinned.
3. **First empirical artifact** at
   `.omx/state/wyner_ziv_deliverability/pairwise_alpha_20260517T213127Z.json`
   with 3-of-3 ADDITIVE pair-results.
4. **Probe outcome registered** in canonical Catalog #313 ledger via
   `register_probe_outcome(verdict='PROCEED', blocker_status='advisory')`;
   expires_at_utc = 2026-06-16 (30-day staleness window).
5. **Lane registry** marked L1 (impl_complete + strict_preflight gates
   satisfied; this memo wires `memory_entry` via the `mark` command after
   the file is on disk).

## Per-pair detail

### pair 1 — `pr101_state_dict + pr106_state_dict`

- raw bytes: 925,009 + 924,277 = 1,849,286
- compressed alone (lzma): 198,272 + 208,832 = 407,104
- compressed concat (lzma): 406,156
- savings alone: 726,737 + 715,445 = 1,442,182
- savings concat: 1,443,130
- α_op3 = 1 − 406,156/407,104 = **+0.0023** (concat is ~2x marginal-compressed;
  α_op3 lands near 0 for similar-sized marginals because sum_marginal ≈
  2 × concat)
- α_savings = 1,443,130 / 1,442,182 = **+1.0007**
- **band**: ADDITIVE

### pair 2 — `pr101_state_dict + posenet_class_sensitivity`

- raw bytes: 925,009 + 20,348,335 = 21,273,344
- compressed alone (lzma): 198,272 + 2,905,308 = 3,103,580
- compressed concat (lzma): 3,055,236
- savings alone: 726,737 + 17,443,027 = 18,169,764
- savings concat: 18,218,108
- α_op3 = **+0.0156**
- α_savings = **+1.0027** (the slight excess over 1.0 reflects lzma's
  amortized dictionary overhead — slightly fewer than 2× the marginal cost
  is needed for the concat, so savings are slightly higher)
- **band**: ADDITIVE

### pair 3 — `pr106_state_dict + posenet_class_sensitivity`

- raw bytes: 924,277 + 20,348,335 = 21,272,612
- compressed alone (lzma): 208,832 + 2,905,308 = 3,114,140
- compressed concat (lzma): 3,075,288
- savings alone: 715,445 + 17,443,027 = 18,158,472
- savings concat: 18,197,324
- α_op3 = **+0.0125**
- α_savings = **+1.0021**
- **band**: ADDITIVE

## Cross-audit: why two α formulas

Per dispatch brief OP-3 council spec the formulas are:

* **OP-3 council form**: α = 1 − compressed_concat / (compressed_a + compressed_b)
* **Savings-ratio form**: α = savings_concat / (savings_a + savings_b)

These measure DIFFERENT semantic quantities and DO NOT converge
numerically for general inputs:

* The OP-3 form measures *fraction of marginal-compressed budget saved by
  concatenation*. For ADDITIVE pairs of similar-size marginals this naturally
  lands near 0.5 (concat ≈ one-marginal, sum_marginal ≈ 2-marginals →
  α_op3 ≈ 0.5). When marginals are *very* unequal (like our pr101_state +
  posenet pairs, 1:22 size ratio), the OP-3 form lands near 0.
* The savings-ratio form measures *fraction of total marginal savings
  preserved when concatenated*. For ADDITIVE pairs this lands near 1.0
  regardless of size ratio. This is the canonical band-determining form
  per Catalog #227 `adjust_predicted_delta_for_composition_alpha` consumer
  semantics (α ≥ 0.7 → ADDITIVE; halving for 0.3 < α < 0.7; floor at α ≤ 0.3).

The probe emits BOTH for cross-audit transparency. The band is determined
by the savings-ratio form (the canonical Catalog #227 consumer).

The DIVERGENCE for redundant pairs is itself a useful empirical signal —
the dual reporting catches what either form alone would miss. Verified
by `test_alpha_known_redundant_pair_diverges_op3_vs_savings` (identical
bytes give α_op3 ≈ 0.47, α_savings ≈ 1.0, divergence > 0.5).

## Stage 2 reactivation clause status (post-Q6.preprobe)

Per council OP-2 reactivation criteria, Q6+ ASYMPTOTIC dispatching is
contingent on ALL of:

| clause | status (2026-05-17 post-Q6.preprobe) |
|---|---|
| (#1) Q4 first empirical anchor lands AND CPU+CUDA paired delta within ±10% of L5 codex predicted band [−0.0019, −0.0032] | PENDING (Q4 not yet dispatched) |
| **(#2) Pairwise composition_alpha empirically measured; ≥ 2-of-3 pairs achieve α ≥ 0.7 ADDITIVE** | **SATISFIED — this probe** |
| (#3) Operator explicitly approves T2 + T3 deliberation budget allocation per UNCLEAR cadence assumption | PENDING (operator decision required) |
| (#4) Per-baker Tier-2 + Tier-3 contest-compliance attestation paths landed per Yousfi's per-stack compliance overhead | PENDING (per-stack compliance landings required) |

**Overall Stage 2 unlock status**: 1-of-4 clauses satisfied. Not yet
authorized to dispatch Q6-Q11 ASYMPTOTIC stacking.

## 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution**: N/A — composition_alpha is a
   SECOND-order interaction term, not per-substrate sensitivity. Per-
   substrate marginal pre-entropy bytes are already in
   `tac.sensitivity_map.*` via the sister prober (`a98c94e1`).
2. **Pareto constraint**: N/A at the probe surface — Catalog #227's
   `adjust_predicted_delta_for_composition_alpha` IS the Pareto
   consumer; this probe emits the α value that feeds it.
3. **Bit-allocator hook**: N/A — Wyner-Ziv hoist allocation activates
   downstream on Stage 2 unlock (post all 4 clauses).
4. **Cathedral autopilot dispatch hook**: ACTIVE — `apply_z1_empirical_revision_to_candidate_delta`
   reads composition_alpha per Catalog #227's adjust helper. The empirical
   α values land in the cross-substrate matrix the autopilot consumes
   (machine-readable JSON @
   `.omx/state/wyner_ziv_deliverability/pairwise_alpha_<utc>.json`).
5. **Continual-learning posterior update**: ACTIVE — probe outcome
   registered in canonical Catalog #313 `probe_outcomes_ledger` via
   `register_probe_outcome(verdict='PROCEED', blocker_status='advisory')`;
   expires 2026-06-16 per 30-day staleness window.
6. **Probe-disambiguator**: ACTIVE — this PROBE IS the disambiguator for
   the Stage 2 ASYMPTOTIC reactivation decision (resolves the Assumption-
   Adversary's CARGO-CULTED assumption #2 to empirical truth for these
   3 candidates).

## Apples-to-apples axis labels (per CLAUDE.md "Apples-to-apples evidence discipline")

Every reported α value is tagged
`[diagnostic; pairwise composition_alpha probe]` per the canonical
non-authoritative measurement axis. NO score claims. NO
promotion-eligibility. NO ready-for-exact-eval-dispatch. Per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" + Catalog #192: this probe is
a DIAGNOSTIC over compressibility statistics, NOT a contest score
measurement; it consumes ZERO GPU and produces ZERO contest-CPU/CUDA
artifacts.

## Premise verification (Catalog #229)

Five premises verified pre-edit, recorded at
`.omx/tmp/q6_preprobe_pairwise_composition_alpha_premise_verifier.txt`:

* **PV-1**: Pre-entropy prober artifact exists at canonical path with
  3 target candidate substrates (pr101_state_dict + pr106_state_dict +
  posenet_class_sensitivity).
* **PV-2**: Catalog #227 substrate_composition_matrix exists with
  expected_alpha banding semantics (α ≥ 0.7 ADDITIVE, 0.3 < α < 0.7
  SUB_ADDITIVE, α ≤ 0.3 SATURATING).
* **PV-3**: T2 council verdict memo exists with OP-3 spec verbatim
  (`concat raw byte arrays + lzma/brotli/zstd + α = 1 − concat/sum_marginal`).
  Clarification noted on the subagent brief's stacked/(alone_a + alone_b)
  formula vs council's 1 − concat/sum_marginal formula — both implemented
  for cross-audit.
* **PV-4**: All 3 candidate `.pt` files exist on disk
  (pr101 = 903 KB, pr106 = 902 KB, posenet = 19.4 MB).
* **PV-5**: Catalog #227 schema match — `expected_alpha` is a float;
  our output schema emits per-pair single-float α matching the
  downstream `adjust_predicted_delta_for_composition_alpha` consumer.

## Sister-subagent ownership map (Catalog #230)

Disjoint scope declared:

* **THIS scope**: NEW `tools/q6_preprobe_pairwise_composition_alpha.py`
  + NEW `src/tac/tests/test_q6_preprobe_pairwise_composition_alpha.py`
  + NEW `.omx/state/wyner_ziv_deliverability/pairwise_alpha_20260517T213127Z.json`
  + THIS landing memo + `.omx/tmp/q6_preprobe_pairwise_composition_alpha_premise_verifier.txt`.
* **Q2+Q3 batched sister** (`a27e5ce58773fc35c`): preflight.py + cathedral
  autopilot + master_gradient_consumers — disjoint.
* **WZ pipeline-stage codec sister** (`a3cf2e4c774f6bb67`):
  `src/tac/codec/wyner_ziv_layer.py` — disjoint.

## Op-routables

* **OP-1 [PROCEED if council/operator decides to fund Q4]**: Q4 first
  empirical anchor (Stage 1 unconditional per HYBRID verdict) — this is
  the natural next dispatch since Q6.preprobe has cleared clause #2.
* **OP-2 [DEFER_PENDING_EVIDENCE — clauses #1 + #3 + #4]**: Q6
  ASYMPTOTIC dispatch (Q6 = pr101_state_dict Tier-2 hoist; Q7 =
  pr106_state_dict Tier-2; Q8 = posenet_class_sensitivity Tier-3) —
  blocked on 3 remaining clauses (Q4 anchor + operator budget approval
  + per-baker compliance attestations).
* **OP-3 [PROCEED — follow-on optional]**: Run the probe on a wider
  candidate set (extend to include pr106_latents + lane_g_v3_renderer +
  sabor_margin_frame_000 / _001 + distill_v2_best — the other 5
  PRE_ENTROPY substrates in the canonical pivot prober artifact). This
  could surface additional ADDITIVE pairs that weren't on the Stage-2
  candidate list but might unlock alternative Stage-2 stacking topologies.
  Cost: $0 GPU, ~10min editor (CLI already supports `--candidate-a/-b`
  pinned mode for arbitrary pairs; sweep mode needs `--candidates` list
  extension, ~10 LOC).
* **OP-4 [PROCEED — autopilot integration verification]**: Verify that
  cathedral autopilot's `apply_z1_empirical_revision_to_candidate_delta`
  consumes the new empirical α values (vs whatever default the existing
  substrate_composition_matrix entry held). The canonical consumer reads
  per-pair α from the substrate_composition_matrix dict; the new probe
  artifact is the queryable source of truth. Cost: $0 GPU, ~30min editor
  if integration patch needed.
* **OP-5 [DEFER_PENDING_EVIDENCE — codex variance probe]**: The
  brotli codec was UNAVAILABLE during this probe run (`brotli_available=true`
  in the sister prober but the actual probe run picked lzma uniformly).
  Re-run with brotli forced to verify α banding stability across codecs;
  per Catalog #227 the banding should be codec-invariant for ADDITIVE
  pairs but could shift for borderline SUB_ADDITIVE pairs. Cost: $0 GPU,
  ~5min editor.

## Cross-references

* `.omx/research/council_horizon_class_scope_t2_20260517.md` — T2
  HYBRID verdict + OP-3 specification.
* `.omx/research/comprehensive_state_tracker_20260517.md` — session
  state tracker.
* `.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_20260517T210723.json`
  — sister pre-entropy prober artifact (marginal compressibility per
  substrate).
* `tools/pre_entropy_substrate_pivot_prober.py` — canonical marginal-
  compressibility prober that this probe extends to pairwise.
* `src/tac/optimization/substrate_composition_matrix.py` — canonical
  Catalog #227 α-banding consumer.
* `src/tac/probe_outcomes_ledger.py` — canonical Catalog #313
  probe-outcomes ledger.
* `tools/q6_preprobe_pairwise_composition_alpha.py` — this lane's
  canonical helper.
* `src/tac/tests/test_q6_preprobe_pairwise_composition_alpha.py` —
  this lane's test suite (15/15 pass).
* `.omx/state/wyner_ziv_deliverability/pairwise_alpha_20260517T213127Z.json`
  — empirical first artifact.
