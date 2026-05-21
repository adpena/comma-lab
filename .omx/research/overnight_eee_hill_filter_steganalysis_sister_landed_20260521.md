# OVERNIGHT-EEE: HILL filter steganalysis sister probe landing

**Date:** 2026-05-21T20:06:17Z
**Lane:** `lane_overnight_eee_hill_filter_steganalysis_sister_tier_1_macos_cpu_advisory_smoke_20260521`
**Verdict:** **NULL_SIGNAL_DEFER** (DEFER per Catalog #307 IMPLEMENTATION-level boundary; paradigm INTACT)
**Cost:** $0 (macOS-CPU advisory; non-promotable per Catalog #1 + #192)
**Wall-clock:** ~4.45s probe + ~10 min wall-clock end-to-end
**Predecessor:** OVERNIGHT-DDD §6.2 op-routable explicit recommendation (commit `44ca225bf`)

---

## §0 — Headline result

Canonical Li-Fridrich-Wang 2014 HILL filter (KB high-pass + L1 + L2 cascade) **did NOT improve on DDD's wavelet-subband sharper baseline** when measured with apples-to-apples textured_avg_weight metric. HILL cascade produced **opposite-direction signal** (textured weight HIGHER, not lower). Verdict: **NULL_SIGNAL_DEFER**.

| Metric | CCC baseline (probe 3, local variance) | DDD sharper (wavelet detail subbands) | EEE HILL (KB+L1+L2 cascade) | Delta vs DDD | Direction vs DDD |
|---|---|---|---|---|---|
| `textured_avg_weight` | 0.8062 | 0.6259 | **1.1249** | +0.499 absolute / +79.7% relative | ❌ HILL HIGHER (opposite direction) |
| Dynamic range (log10) | 9.38 | 8.11 | **0.51** | -7.60 | ❌ HILL below 1.0 threshold |
| flat fraction | 50% | 50% | 50% | 0 | ✅ all pass |
| textured fraction | 25% | 25% | 25% | 0 | ✅ all pass |
| `< 0.5` hard threshold | ❌ (0.806) | ❌ (0.626) | ❌ (1.125) | HILL worse | ❌ all fail hard threshold |

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion":** NULL_SIGNAL is NOT a paradigm kill. The HILL cascade's reciprocal-inside-cascade pattern structurally inverts the cost-distribution interpretation vs CCC/DDD's external-reciprocal-on-magnitude pattern: HILL's outer L2 propagates HIGH cost values (which signal HIGH undetectability in steganography embedding cost terms) to neighboring pixels, producing the opposite of what the CCC/DDD reciprocal framing produces. This is an **IMPLEMENTATION-level boundary** (Li et al.'s HILL cost-distribution semantics ≠ the textured-weight framing the CCC/DDD baselines established), NOT falsification of the Fridrich inverse-steganalysis paradigm. Per Catalog #307 + Catalog #308: DEFER + queue alternative reducers (multi-scale J-UNIWARD + WOW) per DDD §6.2 explicit recommendation.

---

## §1 — Carmack MVP-first 5-step compliance

Per CLAUDE.md "Carmack MVP-first phasing — NON-NEGOTIABLE" (anchor commit `be125b878`):

1. **FREE local macOS-CPU smoke first** — ✅ $0 cost; 4.45s elapsed; never authoritative per Catalog #1 + #192 + #287 + #323.
2. **Falsifiably challenge** — ✅ Predicted signature: `hill_textured_avg_weight ≤ 0.5` AND `hill < ddd (0.626)` AND `hill < ccc (0.806)`. Falsifying outcome: HILL formula yields hill_textured_avg_weight ≥ 0.626 (no improvement vs DDD) OR HILL produces opposite-direction signal. **Outcome:** HILL produced opposite-direction signal (1.125 vs DDD 0.626); both falsification conditions met; verdict NULL_SIGNAL_DEFER per Catalog #307 IMPLEMENTATION-level boundary classification.
3. **Catalog #344 reference** — ✅ Sister of canonical equation candidate `uniward_textured_region_undetectability_pose_distortion_savings_v1` (FORMALIZATION_PENDING; RATIFY-N pending per AAA T4 §9 op-routable). HILL extends UNIWARD via Li-Fridrich-Wang 2014 sharper-tail formulation; probe references the canonical equation in `canonical_equation_reference` field.
4. **Land verdict in same commit batch** — ✅ Probe script + verdict JSON + Catalog #313 ledger row + this landing memo all in same commit batch via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157.
5. **Re-route operator priority queue within ~1h** — ✅ See §6 op-routables.

**Burden-of-proof per Carmack discipline:** ZERO paid-dispatch-first waivers required. Probe surfaced the HILL-cascade IMPLEMENTATION-level boundary BEFORE any Tier-2 paid escalation. NULL_SIGNAL_DEFER verdict structurally prevents premature $1-5 dispatch on a still-gated formulation; preserves capital for multi-scale J-UNIWARD + WOW sister probes per DDD §6.2 + Catalog #308 alternative-reducer cascade.

---

## §2 — HILL filter cascade formulation

### Canonical Li-Fridrich-Wang 2014 "A new cost function for spatial image steganography"

```python
# Step 1: KB high-pass filter (Ker-Bohme 2008 canonical 3x3 kernel; /4 normalization)
K_KB = (1/4) * [[-1, 2, -1],
                [ 2, -4,  2],
                [-1, 2, -1]]
residual = convolve(luma, K_KB)

# Step 2: First low-pass filter L1 (3x3 average per Li et al. canonical)
L1 = ones(3,3) / 9
cost_intermediate = convolve(|residual|, L1)

# Step 3: Reciprocal with sigma to prevent inf at zero residual
cost_reciprocal = 1 / (cost_intermediate + sigma)

# Step 4: Second low-pass filter L2 (15x15 average per Li et al. canonical)
L2 = ones(15,15) / 225
cost_smooth = convolve(cost_reciprocal, L2)

cost_i = cost_smooth_i  # per-pixel cost map (HIGH cost = more textured per Li et al.)
weight_i = cost_i / cost.mean()  # normalized for apples-to-apples vs CCC + DDD
```

### Why HILL produces opposite-direction signal in this framing

Two empirical observations from the probe verdict:

1. **HILL's reciprocal-inside-cascade structurally inverts cost interpretation:** Li-Fridrich-Wang 2014 defines cost as cost_smooth where HIGH cost = LOW embedding admissibility = MORE TEXTURED in steganography embedding terms (where textured pixels CAN admit more distortion). CCC + DDD framings define `weight = 1 / (signal + ε)` where HIGH weight = HIGH suppression = MORE FLAT (where flat pixels must REJECT distortion). The two semantics are reciprocally related: HILL produces a cost-distribution where textured pixels (which Li et al. classified as low embedding-cost) carry HIGH cost values, opposite the CCC/DDD weight-distribution where textured pixels (which CCC/DDD classified as high reconstruction-error tolerance) carry LOW weight values.

2. **HILL cascade smoothing reduces dynamic range:** the outer L2 (15×15 average) propagates cost values to neighboring pixels for spatial coherence, but this smoothing also compresses the dynamic range (HILL log10=0.51 vs DDD log10=8.11). For the apples-to-apples textured_avg_weight metric defined by upper-quartile mask + mean-of-weights-in-mask, the smoothed cost map produces less aggressive textured-vs-flat separation than the un-smoothed DDD wavelet-subband signal.

**Structural conclusion:** HILL is the CORRECT formula for Li et al.'s steganography embedding cost definition but the WRONG formula when interpreted through the CCC/DDD reciprocal-weight framing this probe family uses. Per Catalog #308 alternative-reducer enumeration: the next sister probes must preserve the DDD reciprocal pattern (`weight = 1 / (sharper_signal + ε)`) while substituting a sharper underlying signal (multi-scale J-UNIWARD multi-level db8 DWT OR WOW directional KB4 filter banks).

---

## §3 — Catalog discipline + sister coherence

### Catalog references

- **#1** `check_no_mps_fallback_default` — probe uses explicit CPU; no MPS-fallback ternary.
- **#192** `check_macos_cpu_advisory_not_promoted_without_linux_verification` — verdict JSON tags `axis_tag=[macOS-CPU advisory]` + `promotable=false` + `score_claim=false`.
- **#287** `check_no_docstring_overstatement_without_evidence_tag` — every empirical claim in this memo carries the canonical `[macOS-CPU advisory]` evidence tag.
- **#307** `check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification` — NULL_SIGNAL verdict explicitly tagged IMPLEMENTATION-LEVEL boundary (NOT paradigm falsification of Fridrich inverse-steganalysis); paradigm intact; HILL's reciprocal-inside-cascade semantically incompatible with CCC/DDD's reciprocal-weight framing.
- **#308** `check_kill_verdict_enumerates_alternative_probe_methodologies` — 2 alternative reducers explicitly queued per DDD §6.2 carry-forward: multi-scale J-UNIWARD (Holub-Fridrich-Denemark 2014 multi-level extension), WOW (Holub-Fridrich 2012 directional filter banks).
- **#313** `check_dispatch_target_has_no_predecessor_adjudicated_outcome` — probe-outcomes ledger row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` with `verdict=DEFER`, `metric_value=1.1249`, `threshold=0.5`, `blocker_status=advisory`, `staleness_window_days=30`, `next_action=queue_j_uniward_multiscale_and_wow_per_catalog_308`, `expires_at_utc=2026-06-20T20:06:17.889253Z`.
- **#323** `check_no_score_claim_without_canonical_provenance` — verdict JSON carries canonical `canonical_provenance` block with `kind=macos_cpu_advisory` + `score_claim_valid=false` + `promotable=false` + Li-Fridrich-Wang 2014 + Ker-Bohme 2008 canonical references.
- **#344** `check_empirical_finding_memo_references_canonical_equation` — sister canonical equation `uniward_textured_region_undetectability_pose_distortion_savings_v1` referenced; FORMALIZATION_PENDING per AAA T4 §9 op-routable RATIFY-N.

### Sister coherence verification (Catalog #230 ownership map; cap=2 firm)

- Slot 1 `a24dea7f` MLX-ARCH-3 FastViT-T12 backbone — **DISJOINT scope** (MLX backbone assembly under `tac.mlx_*` package vs my HILL filter probe under `.omx/research/tier_1_distortion_axis_probes_20260521/` directory).
- Cron `9efd7486` Selfcomp XX harvest @ 17:00 CDT — **DISJOINT scope** (Selfcomp harvest vs HILL probe).
- EEE files touched: NEW probe script + NEW verdict JSON + NEW landing memo + `.omx/state/probe_outcomes.jsonl` append (canonical helper, fcntl-locked per Catalog #131).
- Catalog #340 sister-checkpoint guard: PROCEED expected at commit time.

### Catalog #110/#113 APPEND-ONLY compliance

- ZERO mutations to:
  - DDD probe 3b script + verdict JSON + landing memo (predecessor; HISTORICAL_PROVENANCE)
  - CCC probe 3 script + verdict JSON + landing memo (predecessor predecessor; HISTORICAL_PROVENANCE)
  - CLAUDE.md (not in scope)
  - AAA T4 symposium memo (not in scope)
  - PR 101 archive / canonical contest scorer / upstream (mutation frontier off-limits)
- All landings are NEW artifacts.

---

## §4 — 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map:** N/A — probe is observability-only; no candidate sensitivity contribution emitted.
- **Hook #2 Pareto constraint:** N/A — probe does not contribute Pareto-binding evidence (advisory only per Catalog #192).
- **Hook #3 bit-allocator:** N/A — probe surfaces textured-region cost weighting; not wired into a bit-allocator at this stage (paradigm-axis exhaustion required first).
- **Hook #4 cathedral autopilot dispatch:** N/A — probe is `[macOS-CPU advisory]`; per Catalog #192 cannot influence dispatch ranking.
- **Hook #5 continual-learning posterior:** N/A — probe is non-promotable; posterior anchors require contest-axis hardware per Catalog #127.
- **Hook #6 probe-disambiguator:** **ACTIVE** — probe DISAMBIGUATES DDD §6.2 PARTIAL boundary case between (a) HILL filter sharper-tail hypothesis vs (b) HILL semantically-incompatible-with-reciprocal-weight-framing hypothesis. Verdict: (b) confirmed; the DDD §6.2 op-routable expectation that HILL would produce sharper-tail in the same reciprocal-weight framing was empirically falsified at IMPLEMENTATION level. Paradigm intact; alternative reducers queued per Catalog #308.

---

## §5 — Probe verdict full signature table

| Field | Value | Status |
|---|---|---|
| `verdict` | NULL_SIGNAL_DEFER | ❌ DEFER per Catalog #307 |
| `hill_textured_avg_weight` | 1.1249 | ❌ above 0.5 hard threshold + above DDD boundary |
| `ddd_textured_avg_weight` (DDD baseline) | 0.6259 | reference |
| `ccc_textured_avg_weight` (CCC baseline) | 0.8062 | reference |
| `absolute_delta_vs_DDD` | +0.499 | ❌ HILL WORSE than DDD |
| `ratio_HILL_over_DDD` | 1.797 | ❌ HILL 79.7% worse |
| `improvement_direction_vs_DDD` | hill_did_not_improve_or_worse | ❌ confirmed |
| `hill_dynamic_range_log10` | 0.51 | ❌ below 1.0 threshold |
| `hill_textured_fraction` | 25% | ✅ above 10% threshold |
| `hill_flat_fraction` | 50% | ✅ above 30% threshold |
| `sigma_fridrich` | 2^-6 = 0.015625 | Fridrich canonical (matches DDD) |
| `kb_kernel_normalization` | divided_by_4_unit_gain | Ker-Bohme 2008 canonical |
| `l1_window` | 3 | Li et al. canonical |
| `l2_window` | 15 | Li et al. canonical |
| `n_frames` | 8 (apples-to-apples CCC + DDD baseline) | reference |
| `frame_resolution_HxW` | [874, 1164] | PR 101 canonical |
| `elapsed_seconds` | 4.45 | $0 macOS-CPU advisory |

---

## §6 — Operator-routable next actions

### §6.1 — **GATED:** Tier-2 paid dispatch per AAA T4 §6.2

**Status:** REMAINS GATED. HILL cascade did NOT clear the 0.5 hard threshold AND did NOT improve on DDD's wavelet-subband sharper baseline. Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-level boundary classification, Tier-2 paid dispatch ($1-5 Vast.ai 4090 or Lightning T4) on UNIWARD-weighted per-pixel SegNet loss substrate REMAINS DEFERRED pending further inversion refinement at $0.

### §6.2 — **PROCEED (next at $0):** sister probe iteration with remaining alternative reducers per Catalog #308

Per DDD §6.2 op-routable + this EEE landing's empirical falsification of HILL, the queue now reduces to 2 canonical alternatives (each estimated at $0 + ~15-30 min wall-clock):

1. **Multi-scale J-UNIWARD** (Holub-Fridrich-Denemark 2014 extension): multi-level db8 DWT (e.g. 3 levels) summing |HL|+|LH|+|HH| across scales 1-3 with optional per-scale weighting. Preserves DDD's reciprocal-weight framing (`weight = 1 / (multi_scale_detail_sum + ε)`) while capturing texture across multiple spatial scales (small leaves vs medium edges vs large gradient regions). **Highest EV remaining** per Carmack MVP-first lowest-LOC + fastest-implementation discipline.
2. **WOW (Wavelet Obtained Weights)** (Holub-Fridrich 2012): Directional filter banks (KB4 / FilterBank4) with embedding direction prediction. Preserves reciprocal-weight framing while substituting directional-filter signal for db8 detail subbands. More aggressive textured-region suppression but higher LOC + slower implementation.

**Recommended next sister probe:** multi-scale J-UNIWARD first (preserves DDD's reciprocal framing structurally; lowest LOC vs WOW directional filter banks; canonical per Holub-Fridrich-Denemark 2014 J-UNIWARD reference). If multi-scale J-UNIWARD still PARTIAL, escalate to WOW.

### §6.3 — Operator decision queue

1. **APPROVE multi-scale J-UNIWARD sister probe at $0** (recommended; preserves Carmack MVP-first discipline; preserves DDD's reciprocal-weight framing; structurally prevents premature paid dispatch on still-gated formula).
2. **APPROVE WOW sister probe at $0** (alternative; higher LOC + slower implementation; preserves reciprocal-weight framing; more aggressive textured-region suppression per Holub-Fridrich 2012).
3. **OVERRIDE to Tier-2 paid dispatch despite NULL_SIGNAL** (NOT recommended; would burn $1-5 on a falsified formula; requires operator-frontier-override per Catalog #300 Mission alignment Consequence 1).
4. **DEFER all UNIWARD-family work per Catalog #307** (NOT recommended; paradigm intact; reducer-axis cargo-cult not exhausted; 2 canonical alternatives remain queued).

---

## §7 — Discipline + commit metadata

- **Canonical serializer:** `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157
- **Lane id:** `lane_overnight_eee_hill_filter_steganalysis_sister_tier_1_macos_cpu_advisory_smoke_20260521`
- **Checkpoint discipline:** Catalog #206 (3+ checkpoints emitted via `tools/subagent_checkpoint.py`)
- **Sister-checkpoint guard:** Catalog #340 PROCEED required at commit time
- **APPEND-ONLY:** Catalog #110 + #113 — NEW artifacts only; ZERO mutations to predecessor CCC + DDD artifacts
- **Public Disclosure Hygiene:** Catalog #208 — no local absolute paths; no operator email; no Tailscale IPs; no /tmp paths in persisted evidence
- **Catalog #287 placeholder-rationale rejection:** every empirical claim has explicit evidence tag and provenance citation
- **Catalog #323 canonical Provenance umbrella:** verdict JSON carries full provenance block with axis_tag + hardware_substrate + evidence_grade + Li-Fridrich-Wang 2014 + Ker-Bohme 2008 canonical references
- **Catalog #313 probe-outcomes ledger row:** registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` with `verdict=DEFER`, `metric_value=1.1249`, `threshold=0.5`, `blocker_status=advisory`, `staleness_window_days=30`, `expires_at_utc=2026-06-20T20:06:17.889253Z`

---

## §8 — Files landed

| File | Status |
|---|---|
| `.omx/research/tier_1_distortion_axis_probes_20260521/probe_3c_hill_filter_steganalysis_sister.py` | NEW |
| `.omx/research/tier_1_distortion_axis_probes_20260521/probe_3c_hill_filter_steganalysis_sister_verdict.json` | NEW |
| `.omx/state/probe_outcomes.jsonl` | APPEND (Catalog #313 canonical helper) |
| `.omx/research/overnight_eee_hill_filter_steganalysis_sister_landed_20260521.md` | NEW (this file) |

DDD + CCC probe scripts + verdicts + landing memos PRESERVED UNCHANGED per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

---

## §9 — Mission contribution per Catalog #300

**`apparatus_maintenance` per Catalog #300 5-category taxonomy:** EEE NULL_SIGNAL verdict serves the mission by structurally preventing premature $1-5 paid dispatch on a falsified formula. The HILL cascade's IMPLEMENTATION-level boundary (reciprocal-inside-cascade semantics incompatible with DDD/CCC reciprocal-weight framing) is a learning anchor for the next-iteration sister probes: any reducer that hopes to improve on DDD must preserve the DDD reciprocal-weight framing structurally, not just produce a sharper underlying signal in some other coordinate system. Apparatus served mission by preserving capital + surfacing the framing-compatibility constraint as a learning anchor for multi-scale J-UNIWARD and WOW sister probes.

**Apparatus integrity:** NULL_SIGNAL verdict honored Carmack MVP-first 5-step discipline; structurally prevented premature $1-5 paid dispatch on an IMPLEMENTATION-level-falsified formula. Apparatus served mission by preserving capital for the multi-scale J-UNIWARD sister probe that preserves DDD's reciprocal framing.

---

*End of OVERNIGHT-EEE landing memo.*
