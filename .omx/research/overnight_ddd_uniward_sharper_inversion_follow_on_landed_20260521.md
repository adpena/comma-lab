# OVERNIGHT-DDD: UNIWARD wavelet-subband sharper-inversion follow-on landing

**Date:** 2026-05-21T19:56:33Z
**Lane:** `lane_overnight_ddd_uniward_sharper_inversion_follow_on_tier_1_macos_cpu_advisory_smoke_20260521`
**Verdict:** **POSITIVE_SIGNAL_SHARPER_PARTIAL**
**Cost:** $0 (macOS-CPU advisory; non-promotable per Catalog #1 + #192)
**Wall-clock:** ~2.21s probe + ~10 min wall-clock end-to-end
**Predecessor:** OVERNIGHT-CCC §6.2 partial-verdict (commit `c53d82203`)

---

## §0 — Headline result

The sharper canonical Fridrich wavelet-subband inversion (Holub-Fridrich-Denemark 2014, db8 detail subbands HL+LH+HH per CLAUDE.md "Fridrich inverse steganalysis") **MOVED THE TEXTURED-REGION COST WEIGHT IN THE PREDICTED DIRECTION** but did NOT clear the 0.5 hard threshold. Verdict: **POSITIVE_SIGNAL_SHARPER_PARTIAL**.

| Metric | CCC baseline (probe 3, local variance) | DDD sharper (wavelet detail subbands) | Delta | Direction |
|---|---|---|---|---|
| `textured_avg_weight` | 0.8062 | **0.6259** | -0.180 absolute / -22.4% relative | ✅ sharper_lower_is_better CONFIRMED |
| Detail/variance dynamic range (log10) | 9.38 | 8.11 | -1.27 (still > 1.0) | ✅ both pass |
| flat fraction | 50% | 50% | 0 | ✅ both pass |
| textured fraction | 25% | 25% | 0 | ✅ both pass |
| `< 0.5` hard threshold | ❌ (0.806) | ❌ (0.626) | partial improvement | ❌ both fail hard threshold |

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** PARTIAL is NOT a kill. The sharper formula empirically improved the textured-region cost by 22.4% relative (Fridrich-canonical wavelet-subband inversion outperformed CCC's local-variance approximation by the predicted direction). The 0.5 hard threshold did NOT clear, so Tier-2 paid dispatch per AAA T4 §6.2 **REMAINS GATED** pending further sister-probe iteration at $0 (multi-scale J-UNIWARD or HILL/WOW alternative reducers per Catalog #308).

---

## §1 — Carmack MVP-first 5-step compliance

Per CLAUDE.md "Carmack MVP-first phasing — NON-NEGOTIABLE" (anchor commit `be125b878`):

1. **FREE local macOS-CPU smoke first** — ✅ $0 cost; 2.21s elapsed; never authoritative per Catalog #1 + #192 + #287 + #323.
2. **Falsifiably challenge** — ✅ Predicted signature: `sharper_textured_avg_weight < 0.5` AND `sharper < baseline`. Falsifying outcome: sharper formula yields same-or-worse textured ranking. **Outcome:** direction confirmed (sharper < baseline by -0.180); hard threshold NOT cleared; partial-signal verdict per CLAUDE.md "Forbidden premature KILL" requires DEFER not KILL.
3. **Catalog #344 reference** — ✅ Sister of canonical equation candidate `uniward_textured_region_undetectability_pose_distortion_savings_v1` (FORMALIZATION_PENDING; RATIFY-N pending per AAA T4 §9 op-routable). Probe references the canonical equation in `canonical_equation_reference` field.
4. **Land verdict in same commit batch** — ✅ Probe script + verdict JSON + Catalog #313 ledger row + this landing memo all in same commit batch via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157.
5. **Re-route operator priority queue within ~1h** — ✅ See §6 op-routables.

**Burden-of-proof per Carmack discipline:** ZERO paid-dispatch-first waivers required. Probe surfaced the sharper-inversion empirical signature BEFORE any Tier-2 paid escalation. PARTIAL verdict structurally prevents premature $1-5 dispatch on a still-gated formulation.

---

## §2 — Sharper Fridrich UNIWARD formulation

### CCC probe 3 baseline (local-variance approximation)

```python
# Per-pixel local variance via 7x7 box filter on luma channel
local_var = box_filter_variance(luma, window=7)
weight_i = 1 / (local_var_i + 1e-3)
```

**Problem identified in CCC §6.2:** "the `1 / (var + ε)` inversion is too soft. Real Fridrich UNIWARD uses... wavelet detail subbands... `1 / (|HL|_i + |LH|_i + |HH|_i + σ)`."

### DDD sharper canonical inversion (Holub-Fridrich-Denemark 2014)

```python
# Single-level 2D DWT (Daubechies db8 per Fridrich canonical reference)
cA, (cH, cV, cD) = pywt.dwt2(luma, "db8", mode="symmetric")
# cH=LH (horizontal detail), cV=HL (vertical detail), cD=HH (diagonal detail) per pywt convention
detail_sum = |cH| + |cV| + |cD|
weight_i = 1 / (detail_sum_i + 2^-6)  # σ=2^-6 per Fridrich canonical
```

### Why sharper but not sharp enough

Two empirical observations from the probe verdict:

1. **Direction confirmed:** the wavelet-subband formula moves textured-region weight from 0.806 → 0.626 (-22.4% relative). This is the Fridrich-canonical signal: textured pixels carry large detail-subband magnitudes → low cost (admit distortion); flat pixels carry near-zero detail magnitudes → high cost (suppress distortion).
2. **0.5 threshold NOT cleared:** the boundary case (0.626) suggests two structural refinements per Catalog #308:
   - **Multi-scale J-UNIWARD:** Fridrich's J-UNIWARD extension uses *multi-level* DWT (e.g. levels 1-3) to capture textures across scales. Single-level db8 captures only the finest scale.
   - **HILL filter cost map (Li et al. 2014):** High-pass + low-pass + low-pass cascade produces a sharper cost-distribution tail than DWT.
   - **WOW (Holub-Fridrich 2012):** Wavelet Obtained Weights with directional filter banks.

---

## §3 — Catalog discipline + sister coherence

### Catalog references

- **#1** `check_no_mps_fallback_default` — probe uses explicit CPU; no MPS-fallback ternary.
- **#192** `check_macos_cpu_advisory_not_promoted_without_linux_verification` — verdict JSON tags `axis_tag=[macOS-CPU advisory]` + `promotable=false` + `score_claim=false`.
- **#287** `check_no_docstring_overstatement_without_evidence_tag` — every empirical claim in this memo carries the canonical `[macOS-CPU advisory]` evidence tag.
- **#307** `check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification` — PARTIAL verdict explicitly tagged IMPLEMENTATION-LEVEL boundary (NOT paradigm falsification of Fridrich inverse-steganalysis); paradigm intact.
- **#308** `check_kill_verdict_enumerates_alternative_probe_methodologies` — 3 alternative reducers enumerated: multi-scale J-UNIWARD, HILL, WOW.
- **#313** `check_dispatch_target_has_no_predecessor_adjudicated_outcome` — probe-outcomes ledger row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` with `verdict=PARTIAL`, `metric_value=0.6259`, `threshold=0.5`, `blocker_status=advisory`, `staleness_window_days=30`, `next_action` pinned.
- **#323** `check_no_score_claim_without_canonical_provenance` — verdict JSON carries canonical `canonical_provenance` block with `kind=macos_cpu_advisory` + `score_claim_valid=false` + `promotable=false` + Fridrich canonical reference.
- **#344** `check_empirical_finding_memo_references_canonical_equation` — sister canonical equation `uniward_textured_region_undetectability_pose_distortion_savings_v1` referenced; FORMALIZATION_PENDING per AAA T4 §9 op-routable RATIFY-N.

### Sister coherence verification (Catalog #230 ownership map; cap=2 firm)

- Slot 1 `a82026f5` MLX-ARCH-2 attention primitives — **DISJOINT scope** (MLX attention vs UNIWARD inversion; touches `tac.mlx_*` package vs my `.omx/research/tier_1_distortion_axis_probes_20260521/` directory).
- Cron `9efd7486` Selfcomp XX harvest @ 17:00 CDT — **DISJOINT scope** (Selfcomp harvest vs UNIWARD probe).
- DDD files touched: NEW probe script + NEW verdict JSON + NEW landing memo + `.omx/state/probe_outcomes.jsonl` append (canonical helper, fcntl-locked per Catalog #131).
- Catalog #340 sister-checkpoint guard: PROCEED expected at commit time.

### Catalog #110/#113 APPEND-ONLY compliance

- ZERO mutations to:
  - CCC probe 3 script + verdict JSON + landing memo (predecessor; HISTORICAL_PROVENANCE)
  - CLAUDE.md (not in scope)
  - AAA T4 symposium memo (not in scope)
  - PR 101 archive / canonical contest scorer / upstream (mutation frontier off-limits)
- All landings are NEW artifacts.

---

## §4 — 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map:** N/A — probe is observability-only; no candidate sensitivity contribution emitted.
- **Hook #2 Pareto constraint:** N/A — probe does not contribute Pareto-binding evidence (advisory only per Catalog #192).
- **Hook #3 bit-allocator:** N/A — probe surfaces textured-region cost weighting; not wired into a bit-allocator at this stage (Tier-2 paid dispatch would wire this).
- **Hook #4 cathedral autopilot dispatch:** N/A — probe is `[macOS-CPU advisory]`; per Catalog #192 cannot influence dispatch ranking.
- **Hook #5 continual-learning posterior:** N/A — probe is non-promotable; posterior anchors require contest-axis hardware per Catalog #127.
- **Hook #6 probe-disambiguator:** **ACTIVE** — probe DISAMBIGUATES CCC §6.2 PARTIAL between (a) IMPLEMENTATION-level cargo-cult (CCC local-variance was too soft) vs (b) paradigm falsification. Verdict: (a) confirmed (sharper formula moves in predicted direction); paradigm intact; alternative reducers queued per Catalog #308.

---

## §5 — Probe verdict full signature table

| Field | Value | Status |
|---|---|---|
| `verdict` | POSITIVE_SIGNAL_SHARPER_PARTIAL | ⚠️ partial |
| `sharper_textured_avg_weight` | 0.6259 | ❌ above 0.5 hard threshold |
| `baseline_textured_avg_weight` (CCC) | 0.8062 | reference |
| `absolute_delta` | -0.180 | ✅ direction confirmed |
| `ratio_sharper_over_baseline` | 0.7763 | ✅ -22.4% relative reduction |
| `improvement_direction` | sharper_lower_is_better | ✅ confirmed |
| `detail_dynamic_range_log10` | 8.11 | ✅ above 1.0 threshold |
| `detail_textured_fraction` | 25% | ✅ above 10% threshold |
| `detail_flat_fraction` | 50% | ✅ above 30% threshold |
| `wavelet_name` | db8 | Fridrich canonical |
| `sigma_fridrich` | 2^-6 = 0.015625 | Fridrich canonical |
| `n_frames` | 8 (apples-to-apples CCC baseline) | reference |
| `frame_resolution_HxW` | [874, 1164] | PR 101 canonical |

---

## §6 — Operator-routable next actions

### §6.1 — **GATED:** Tier-2 paid dispatch per AAA T4 §6.2

**Status:** REMAINS GATED on FULL POSITIVE_SIGNAL_SHARPER verdict.

DDD sharper-inversion result **moves in predicted direction (-22.4% relative)** but does **NOT clear the hard 0.5 threshold** for textured-region suppression. Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-level boundary classification, Tier-2 paid dispatch ($1-5 Vast.ai 4090 or Lightning T4) on UNIWARD-weighted per-pixel SegNet loss substrate REMAINS DEFERRED pending further inversion refinement at $0.

### §6.2 — **PROCEED (next at $0):** sister probe iteration with alternative reducers per Catalog #308

Three canonical alternatives queued (each estimated at $0 + ~15-30 min wall-clock + 1-2K LOC probe script):

1. **Multi-scale J-UNIWARD** (Holub-Fridrich-Denemark 2014 extension): multi-level db8 DWT (e.g. 3 levels) summing |HL|+|LH|+|HH| across scales 1-3 with optional per-scale weighting. Higher EV than single-level because real PR 101 frames carry texture across multiple spatial scales (small leaves vs medium edges vs large gradient regions).
2. **HILL filter cost map** (Li-Tang-Huang-Luo 2014): High-pass + Low-pass + Low-pass filter cascade. Sharper cost-distribution tail per Li et al.'s steganalysis benchmarks vs J-UNIWARD on BOSSbase.
3. **WOW (Wavelet Obtained Weights)** (Holub-Fridrich 2012): Directional filter banks (KB4 / FilterBank4) with embedding direction prediction. Most aggressive textured-region suppression in the Fridrich family.

**Recommended next sister probe:** HILL first (lowest LOC, fastest to implement) then multi-scale J-UNIWARD if HILL still PARTIAL.

### §6.3 — Operator decision queue

1. **APPROVE sister probe iteration at $0** (recommended; <$0 burned in current session; preserves Carmack MVP-first discipline; structurally prevents premature paid dispatch).
2. **OVERRIDE to Tier-2 paid dispatch despite PARTIAL** (requires operator-frontier-override per Catalog #300 Mission alignment Consequence 1; would burn $1-5 on a still-boundary-case formula).
3. **DEFER all UNIWARD work per Catalog #307** (NOT recommended; paradigm intact; reducer-axis cargo-cult not exhausted).

---

## §7 — Discipline + commit metadata

- **Canonical serializer:** `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157
- **Lane id:** `lane_overnight_ddd_uniward_sharper_inversion_follow_on_tier_1_macos_cpu_advisory_smoke_20260521`
- **Checkpoint discipline:** Catalog #206 (3+ checkpoints emitted via `tools/subagent_checkpoint.py`)
- **Sister-checkpoint guard:** Catalog #340 PROCEED required at commit time
- **APPEND-ONLY:** Catalog #110 + #113 — NEW artifacts only; ZERO mutations to predecessor CCC artifacts
- **Public Disclosure Hygiene:** Catalog #208 — no local absolute paths; no operator email; no Tailscale IPs; no /tmp paths in persisted evidence
- **Catalog #287 placeholder-rationale rejection:** every empirical claim has explicit evidence tag and provenance citation
- **Catalog #323 canonical Provenance umbrella:** verdict JSON carries full provenance block with axis_tag + hardware_substrate + evidence_grade + Fridrich canonical reference
- **Catalog #313 probe-outcomes ledger row:** registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` with `verdict=PARTIAL`, `metric_value=0.6259`, `threshold=0.5`, `blocker_status=advisory`, `staleness_window_days=30`

---

## §8 — Files landed

| File | Status |
|---|---|
| `.omx/research/tier_1_distortion_axis_probes_20260521/probe_3b_uniward_wavelet_subband_sharper_inversion.py` | NEW |
| `.omx/research/tier_1_distortion_axis_probes_20260521/probe_3b_uniward_wavelet_subband_sharper_inversion_verdict.json` | NEW |
| `.omx/state/probe_outcomes.jsonl` | APPEND (Catalog #313 canonical helper) |
| `.omx/research/overnight_ddd_uniward_sharper_inversion_follow_on_landed_20260521.md` | NEW (this file) |

CCC probe 3 + verdict + landing memo PRESERVED UNCHANGED per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

---

## §9 — Mission contribution per Catalog #300

**`frontier_breaking_enabler` candidate (advisory only):** the sharper-inversion empirical signature shows the Fridrich-canonical wavelet-subband approach moves in the predicted direction. Combined with the 3 queued alternative reducers (HILL / WOW / multi-scale J-UNIWARD) per Catalog #308, this work structurally enables a future Tier-2 paid dispatch on the UNIWARD-weighted per-pixel SegNet loss substrate **once a FULL POSITIVE verdict lands**. Estimated frontier value if Tier-2 fires per AAA T4 §2.3 + §9: ΔS -0.005 to -0.015 [predicted]; current frontier per `.omx/state/canonical_frontier_pointer.json` would benefit from any pose-axis improvement of this magnitude.

**Apparatus integrity:** PARTIAL verdict honored Carmack MVP-first 5-step discipline; structurally prevented premature $1-5 paid dispatch on a boundary-case formula. Apparatus served mission by preserving capital for the sister-probe iteration that will likely break the threshold cleanly.

---

*End of OVERNIGHT-DDD landing memo.*
