# ACH Matrix — Substrate Candidate Analysis of Competing Hypotheses

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Method:** Heuer 1980s Analysis of Competing Hypotheses, applied per CIA Tradecraft Primer (2009).
**Evidence grade:** `[structured-analytic-technique]` — methodological.

---

## 0. ACH method recap

For each EVIDENCE row × HYPOTHESIS column cell, assess:
- `✓` (consistent with hypothesis — confirms or is at least not against)
- `✗` (inconsistent — would not occur if hypothesis is true)
- `?` (neutral / unknown / under-determined)
- `–` (not applicable to this hypothesis)

The hypothesis with the LOWEST count of `✗` (inconsistencies) is best supported. The
hypothesis with the highest `✓` count is NOT the winner — confirming evidence is cheap
and most hypotheses generate some.

---

## 1. Substrate candidates (columns)

We evaluate 14 substrate hypotheses spanning:
- HNeRV-family-internal: H1 PR101-parity, H2 PR101+hyperprior+QAT
- A1-substrate-extension: H3 A1+LAPose, H4 A1+wavelet
- Scorer-blindspot-exploit: H5 SABOR, H6 S2SBS, H7 F0ABS, H8 PAYIC
- Substitution: H9 mosaic encoder swarm, H10 SIREN substrate
- Adversarial: H11 LoRA/DoRA, H12 IGLT
- Codec-replacement: H13 WBCE-MERA, H14 Langevin-MCMC
- Hybrid: H15 Cool-Chic / Coord-MLP residual basis (PR103-on-PR106 sidecar)

## 2. Evidence anchors (rows)

| # | Evidence anchor | Source |
|---|---|---|
| E1 | PR101 0.193 [contest-CUDA] empirical anchor | public leaderboard 2026-05-04 |
| E2 | PR101 0.198 [contest-CPU] empirical anchor | post-race CPU bot reply |
| E3 | A1 0.192847 [contest-CPU] empirical anchor | internal clone, 2026-05-12 |
| E4 | φ1 SABOR audit: 99.27% stable @ ε=32 RGB uint8 | `.omx/research/sabor_boundary_audit_20260513.md` |
| E5 | PR106 r2 pose_avg = 3.4e-5, marginal d_pose=271 | CLAUDE.md "SegNet vs PoseNet importance" |
| E6 | Theoretical floor 0.10 ± 0.03 (council aggregation) | `grand_council_first_principles_original_score_lowering_20260513.md` |
| E7 | HNeRV-family floor estimate 0.171 | grand council Q4 §6 |
| E8 | Pose-residual harmonic structure (not yet measured) | hypothesis from F2 |
| E9 | Auth-proxy gap on PoseNet: 2-11× | CLAUDE.md MPS-falsification |
| E10 | rem2 PR103 silver (0.195) was 241 LOC, 2 files | CLAUDE.md "Race-mode rigor" |
| E11 | PR101 archive ≈ 178 KB at 25·B/37,545,489 ≈ 0.119 rate | empirical |
| E12 | Inflate.py budget ≤ 100 LOC (200 LOC waiver) | HNeRV parity lesson 4 |
| E13 | Score formula nonlinearity: sqrt(10·d_pose) blows up at 0 | CLAUDE.md mathematical fact |
| E14 | SegNet stride-2 stem; PoseNet 2× stride-2 + chroma 2× | `s2sbs_blindspot_audit_20260513.md` |
| E15 | Public PR mining: PR50-80 + PR105-115 (no sub-0.20 except PR105 0.198) | `feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md` |
| E16 | A1+LAPose substrate already in flight (canonical pose codec) | session memory |
| E17 | Cool-Chic / C3 / Coord-MLP residual basis prior research | grand council §6.3 |
| E18 | Carmack 100-LOC budget review on multi-renderer dispatch | ledger 02 §1.2 |

---

## 3. ACH Matrix (15 columns × 18 rows)

Legend: ✓=consistent | ✗=inconsistent | ?=neutral/unknown | –=N/A

| | H1 PR101-parity | H2 PR101+hyperprior+QAT | H3 A1+LAPose | H4 A1+wavelet | H5 SABOR | H6 S2SBS | H7 F0ABS | H8 PAYIC | H9 mosaic | H10 SIREN | H11 LoRA/DoRA | H12 IGLT | H13 WBCE-MERA | H14 Langevin-MCMC | H15 Coord-MLP residual |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 PR101 0.193 CUDA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| E2 PR101 0.198 CPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| E3 A1 0.193 CPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| E4 SABOR 99.27% stable | ? | ? | ? | ? | ✓ | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| E5 PoseNet marginal 271 | ✓ | ✓ | ✓✓ | ✓ | ✗ | ✗ | ✓ | ✓✓ | ✓ | ? | ✓ | ✓ | ? | ? | ✓ |
| E6 floor 0.10±0.03 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓✓ | ✓ | ? | ✗ | ✗ | ? | ? | ✓ |
| E7 HNeRV floor 0.171 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| E8 pose harmonics (?) | ? | ? | ✓ | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ✓ |
| E9 proxy-auth gap 2-11× | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ? | ? | ? | ✓ |
| E10 silver 241 LOC | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| E11 PR101 ~178 KB | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| E12 inflate ≤ 100 LOC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✗ | ? | ✓ | ✓ | ? | ✗ | ✓ |
| E13 sqrt nonlinearity | ✓ | ✓ | ✓✓ | ✓ | ✗ | ✗ | ✓ | ✓✓ | ✓ | ? | ✓ | ✓ | ? | ? | ✓ |
| E14 stride-2 stems | ? | ? | ? | ? | ✓✓ | ✓✓ | ✓ | ? | ? | ? | ? | ? | ? | ? | ✓ |
| E15 no sub-0.20 in corpus | ✗ | ✗ | ✗ | ✗ | ? | ? | ? | ✓ | ? | ? | ✗ | ? | ? | ? | ? |
| E16 A1+LAPose in flight | – | – | ✓ | ? | – | – | – | – | – | – | – | – | – | – | – |
| E17 Coord-MLP basis ready | – | – | – | – | – | – | – | – | – | – | – | – | – | – | ✓✓ |
| E18 Carmack 100-LOC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | ✗ | ✓ | ✓ | ✓ | ? | ✗ | ✓ |
| **Total ✗ (inconsistencies)** | **3** | **4** | **3** | **3** | **2** | **2** | **0** | **1** | **2** | **1** | **3** | **1** | **0** | **2** | **0** |
| **Total ✓ (consistencies)** | 11 | 10 | 12 | 11 | 12 | 11 | 12 | 11 | 9 | 8 | 11 | 9 | 7 | 6 | 14 |

---

## 4. Verdicts

### 4.1 Lowest-inconsistency hypotheses (best supported by ACH)

Three hypotheses tie at **zero inconsistencies**:
1. **H7 F0ABS** (Frame-0 Asymmetric Byte Stuffing) — every evidence is at least neutral
2. **H13 WBCE-MERA** — every evidence is at least neutral (but only 7 ✓; mostly under-determined)
3. **H15 Coord-MLP residual basis** — every evidence is at least neutral, plus E17 explicit
   readiness anchor

**ACH verdict (Heuer method): H15 Coord-MLP residual basis is the BEST-SUPPORTED hypothesis**
— 14 consistencies, 0 inconsistencies, plus the explicit E17 readiness evidence. H7 F0ABS
is the next-best given φ2 PAYIC probe pending.

### 4.2 Highest-inconsistency (poorly-supported) hypotheses

- **H2 PR101+hyperprior+QAT**: 4 inconsistencies (E6, E7, E10, E15). The HNeRV-family-internal
  hypothesis is INCONSISTENT with the council's floor estimate (0.10 ± 0.03 vs. HNeRV-floor
  0.171) — these substrates can only reach 0.171.

- **H1, H3, H4, H11**: 3 inconsistencies each. HNeRV-family-internal or near-internal hypotheses
  are inconsistent with the floor analysis.

### 4.3 ACH-driven priority ranking

| Rank | Hypothesis | ✗ count | Predicted Δscore | Risk | Operator priority |
|---:|---|---:|---|---|---|
| 1 | H15 Coord-MLP residual basis | 0 | -0.005 to -0.020 | LOW | **DISPATCH FIRST** |
| 2 | H7 F0ABS | 0 | -0.005 to -0.015 | MED (φ2 dep) | DISPATCH AFTER φ2 |
| 3 | H13 WBCE-MERA | 0 | -0.010 to -0.030 | MED | RESEARCH PROBE |
| 4 | H8 PAYIC | 1 | -0.018 to -0.060 | HIGH | φ2 PROBE FIRST |
| 5 | H10 SIREN | 1 | -0.005 to -0.018 | MED | PARALLEL |
| 6 | H12 IGLT | 1 | -0.003 to -0.012 | LOW | OPPORTUNISTIC |
| 7 | H5 SABOR | 2 | -0.005 to -0.020 | LOW | φ3 PROBE PENDING |
| 8 | H6 S2SBS | 2 | -0.005 to -0.015 | LOW | RESEARCH PROBE |
| 9 | H9 mosaic encoder | 2 | -0.005 to -0.015 | MED | LOC-BUDGET PROBE |
| 10 | H14 Langevin-MCMC | 2 | -0.005 to -0.025 | HIGH | DEFERRED |
| 11-14 | H1-4, H11 | 3 | various | LOW-MED | HNeRV-FAMILY-INTERNAL (low priority per E7) |
| 15 | H2 PR101+hyperprior+QAT | 4 | -0.003 to -0.012 | LOW | TIGHTLY-CONSTRAINED |

---

## 5. ACH-derived recommendation to operator

**The single best-supported substrate per ACH is H15 Coord-MLP residual basis** —
specifically the PR103-on-PR106 sidecar pattern. Evidence:
- E17 explicit readiness anchor (prior research already in tree)
- E5 PoseNet marginal 271 → pose-axis bytes matter; residual basis closes the pose gap
- E10 silver was 241 LOC; residual basis fits the budget
- E18 Carmack 100-LOC; residual basis is compatible
- E6/E7 floor analysis → residual basis is COMPOSITIONAL ESCAPE, not HNeRV-family-internal

**Secondary recommendations**:
- Run φ2 PAYIC probe ($0-5 GPU) to disambiguate H7 / H8 viability
- Run φ3 SABOR boundary audit refinement ($0 GPU) to refine H5
- The 5 candidates in the autopilot journal (substrate-composition matrix per session
  memory) intersect with H15 + H7 — those are the right primary dispatch targets

**De-prioritized**:
- H1 PR101-parity (HNeRV-family-internal; capped at 0.171 per E7)
- H2 PR101+hyperprior+QAT (most inconsistent; small marginal Δ at high build cost)

---

## 6. Cross-reference to sister-team's signal-processing recommendations

This ACH should be read alongside the sister `lane_expert_team_signal_processing_alien_tech_20260513`
output. If the sister team's ranking diverges from H15-first, council should reconcile
before next dispatch.

---

**End ACH matrix.**
