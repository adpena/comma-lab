# Ledger 03 — CIA Structured Analytic Techniques (Heuer-Pherson 2010)

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Evidence grade:** `[structured-analytic-technique]` — methodological, not empirical.

---

## 0. Persona discipline

Richards J. Heuer Jr. (CIA, *Psychology of Intelligence Analysis* 1999) and Randolph H.
Pherson (CIA Sherman Kent School) formalized 50 structured analytic techniques (SATs) in
*Structured Analytic Techniques for Intelligence Analysis* (CQ Press, 2010,
ISBN 978-1608710188). The SATs counter cognitive biases — anchoring, confirmation bias,
groupthink, mirror-imaging — that have caused intelligence-community failures: Pearl
Harbor (1941), Bay of Pigs (1961), Tet (1968), Yom Kippur (1973), Iran (1978), Iraq WMD
(2002), the COVID-origins debate, and many others.

The contest analog: our council is a 10-15-member analytic group prone to the same biases.
Decisions like "is TRIPLET E worth firing?" or "should we KILL Lane 12 NeRV mask codec?"
are intelligence judgments under uncertainty. Applying SATs structurally to these decisions
extincts known failure modes.

---

## 1. Analysis of Competing Hypotheses (ACH) — Heuer 1980s

### 1.1 Method

ACH is a matrix of EVIDENCE rows × HYPOTHESIS columns. For each (evidence, hypothesis)
cell, the analyst assesses CONSISTENCY: ✓ (consistent), ✗ (inconsistent), or ? (neutral).
A hypothesis is supported by the COUNT of inconsistencies (✗) against it being LOW, not
the count of consistencies being HIGH. The least-inconsistent hypothesis wins.

This inversion is critical: confirmatory evidence is cheap (most hypotheses generate at
least some); disconfirming evidence is expensive (you must show no compatible explanation
exists). ACH avoids the confirmation-bias trap.

### 1.2 Contest application

See the **ACH MATRIX** deliverable, this team's primary output (separate file:
`.omx/research/expert_team_aerospace_stealth_analytic_ach_matrix_20260513.md`).

### 1.3 Derived technique C1 — ACH-driven substrate ranking

**Formulation.** Build a matrix:
- ROWS: empirical evidence anchors (PR101 0.193 [contest-CUDA], PR101 0.198 [contest-CPU],
  φ1 SABOR audit 99.27% stable, PR106 r2 pose_avg=3.4e-5, etc.)
- COLUMNS: substrate hypotheses (A1+LAPose, A1+wavelet, SABOR, S2SBS, F0ABS, mosaic, etc.)

For each cell, mark ✓/✗/?. Rank columns by count of ✗ (lower is better).

**Predicted Δscore:** N/A (methodology, not substrate).
**Build cost:** N/A (deliverable, this session).
**Output:** see ACH matrix file.

---

## 2. Key Assumption Check — Heuer-Pherson 2010

### 2.1 Method

Before any analytic conclusion, list the assumptions the conclusion rests on. For each:
(a) is it explicitly testable?; (b) is it likely to fail?; (c) what evidence would
disconfirm it?; (d) if disconfirmed, how does the conclusion change?

Common analytic assumption-failures:
- "We are not facing a deception campaign" (Pearl Harbor)
- "Our adversary thinks like we do" (Yom Kippur)
- "The current measurement instrument is accurate" (Iraq WMD)

### 2.2 Contest application

Our council's working assumptions include:
1. PR101's 0.193 [contest-CUDA] score is the true frontier.
2. HNeRV-family achievable floor is 0.171.
3. SegNet's argmax is stable under ±32-RGB-noise (99.27% per φ1).
4. PoseNet has a 6-dim quotient (existence per Tao).
5. Our internal CUDA measurement faithfully reproduces GHA Linux x86_64.
6. The 600 pairs of `upstream/videos/0.mkv` are representative of the contest video.
7. Brotli-15 is the best entropy coder for our archive class.

### 2.3 Derived technique C2 — Key-assumption-check matrix

**Formulation.** For each working assumption, score:
| Assumption | Testable? | Likely-failure prob | Disconfirming evidence | Impact if fails |
|---|---|---:|---|---|
| 1. PR101 frontier | YES | 0.05 | replay disagreement | minor; PR103 might be true frontier |
| 2. HNeRV ceiling 0.171 | NO (forward-looking) | 0.40 | sub-0.171 [contest-CUDA] | major; floor lower |
| 3. SegNet argmax stable | YES | 0.05 | φ1 retest with K=10 | minor; redo at K=10 |
| 4. PoseNet 6-dim quotient | PARTIAL (φ2) | 0.30 | φ2 PAYIC probe fails | major; A6 F0ABS dies |
| 5. CUDA-CPU faithful | YES | 0.15 | side-by-side eval drift | major; revisit all CUDA promotions |
| 6. video 0.mkv representative | NO | 0.20 | unseen video performs differently | medium; contest scope |
| 7. Brotli-15 optimal | YES | 0.40 | zstd-22 or LZMA-9 better | minor; swap coder |

Assumption 4 is the highest-risk: if PoseNet's 6-dim quotient doesn't have a low-byte
inverse, technique A6 F0ABS is dead. PRIORITY: run φ2 PAYIC probe immediately.

**Predicted impact:** Surfaces assumption 4 as a blocker on A6/PAYIC investment.

---

## 3. Devil's Advocacy — formal dissent role

### 3.1 Method

Assign one council member to argue AGAINST the consensus. The role is rotated so no member
becomes "the negative person." The devil's advocate's job is to construct the strongest
possible case for the dissenting view — even if they don't personally hold it.

### 3.2 Contest application

Our council already has a Contrarian role. Per CLAUDE.md "Council conduct — non-negotiable":
*"The Contrarian's role is to challenge WEAK arguments, not BOLD ones."* This is a refined
version of devil's advocacy.

### 3.3 Derived technique C3 — Rotating devil's-advocate assignment

**Formulation.** For each council deliberation, ROTATE the devil's-advocate role through
the 10 inner-council members (Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Quantizr,
Hotz, Selfcomp, MacKay, Ballé) so each member adopts the contrarian posture once per 10
deliberations. This prevents the Contrarian role from becoming a single member's identity
and forces every member to construct dissenting cases.

**Predicted impact:** ~10-20% increase in council dissent-discovery rate per the McKinsey
"Bias Busters: Premortems" finding.

---

## 4. What-If Analysis — Heuer-Pherson 2010

### 4.1 Method

Assume a low-probability/high-impact scenario has occurred. Work backward to identify:
(a) what conditions would have produced it; (b) what early indicators would have warned;
(c) what mitigations would have prevented it.

### 4.2 Contest application

The IF: "Our final submission scores 0.225, BELOW PR101 but ABOVE the medal band." That is
the gold-medal-but-not-bronze failure mode.

Working backward:
- 0.225 is at the 11th-place band (April 30 contest).
- Conditions: our internal CUDA score matched, but the bot's CPU eval drifted by +0.020.
- Early indicators: CPU-eval-drift between proxy and contest in earlier auth eval.
- Mitigations: dual-eval mandate on every shippable archive (CLAUDE.md non-negotiable).

### 4.3 Derived technique C4 — Standardized what-if analysis for shipping

**Formulation.** For every candidate archive that crosses the "submit?" threshold, run a
what-if matrix:
| Scenario | Probability | Impact | Mitigation pre-shipped? |
|---|---:|---|---|
| CPU eval drifts +0.020 from CUDA | 0.25 | major | Linux x86_64 CPU pre-replay |
| Brotli version drift host vs CI | 0.05 | medium | Brotli pinned in inflate.py |
| Archive bytes corrupted in upload | 0.01 | catastrophic | SHA-256 verify pre-PR |
| Council misjudged predicted Δ | 0.40 | minor (deferred re-rank) | continual-learning posterior |

**Predicted impact:** Closes the 2026-04-30 May 4 race-failure-mode loophole (we shipped
0.229 internally; medal band was 0.193-0.196 [contest-CPU]).

---

## 5. Premortem Analysis — Klein 2007

### 5.1 Method

Imagine the project failed. Work backward to identify failure modes. Klein's 2007 HBR
article showed pre-mortems increase failure-mode identification by ~30% over standard
risk assessments because the COMMITMENT BIAS is inverted.

### 5.2 Contest application

See the **PRE-MORTEM ANALYSIS** deliverable, this team's secondary primary output
(separate file: `.omx/research/expert_team_aerospace_stealth_analytic_pre_mortem_20260513.md`).

### 5.3 Derived technique C5 — Pre-mortem before every dispatch

**Operational rule.** Before any > $1 GPU dispatch, the dispatching team must produce a
pre-mortem listing the top 5 ways the dispatch could fail-loudly OR fail-silently. The
pre-mortem is a CLAUDE.md non-negotiable artifact analogous to the "Submission auth eval"
section.

---

## 6. Red Team Analysis — extended devil's advocacy

### 6.1 Method

A dedicated red team adopts the adversary's PERSPECTIVE: their goals, constraints, biases,
information advantages. The red team plans operations FROM THE ADVERSARY'S SIDE and reports
findings back to the blue team.

### 6.2 Contest application

The "adversary" is the OTHER CONTESTANTS — Brady Meighan, rem2, EthanYangTW, Quantizr.
Question: if you were Quantizr (already at 0.33 with 88K-param FiLM-conditioned depthwise-
separable CNN), what would your NEXT score-lowering move be?

Quantizr's own assessment (per CLAUDE.md "Quantizr intelligence"): *"sub 0.30 is possible
just by sweeping conv dims."* He stopped optimizing because the EFFORT exceeded the
EXPECTED gain. His next move would be: sweep conv dims at low cost; if Δ > 0.005, re-engage.

### 6.3 Derived technique C6 — Red-team competitor profiling

**Formulation.** For each leaderboard competitor (top 10), build a profile:
| Competitor | Last score | Architecture | Stop reason (inferred) | Likely next move | Threat level |
|---|---:|---|---|---|---|
| Brady Meighan | 0.195 PR100 | dense+entropy | exhausted ideas | re-engage if 0.18 lands | HIGH |
| rem2 (silver) | 0.195 PR103 | residual-mask | 241 LOC budget | re-engage on novel coder | MED |
| EthanYangTW | 0.195 PR102 | residual-pose | ??? | unknown | MED |
| Quantizr | 0.33 | FiLM CNN | sub-0.30 reachable but unmotivated | sweep conv dims | LOW |

**Predicted impact:** Surfaces Brady Meighan as the highest-threat-of-resurgence competitor.

---

## 7. Quadrant Crunching — 2D strategic axes

### 7.1 Method

Identify two orthogonal strategic axes. Build a 2×2 matrix. Analyze each quadrant's
implications.

### 7.2 Contest application

Axis 1: HNeRV-family-internal vs HNeRV-external substrate.
Axis 2: Bytes-axis-focused vs Components-axis-focused.

|             | Bytes-axis focused | Components-axis focused |
|-------------|--------------------|-----------------------|
| HNeRV-internal | PR101+Brotli optimization, hyperprior, QAT | PR101+score-domain-training (S1 recovery) |
| HNeRV-external | SABOR + S2SBS + mosaic encoder | PAYIC + active cancellation |

Each quadrant represents a class of substrates. The council's TRIPLET φ recommendation
covers all 4 quadrants modulo replacement-substrate, which is HNeRV-external + components.
The grand-council aggregation (post-Q4): "HNeRV-external + components is the highest-EV
direction." Empirical validation pending φ2 PAYIC probe.

### 7.3 Derived technique C7 — Quadrant-crunching for substrate-class diversification

**Operational rule.** When the dispatch matrix is built (the 5 candidates from B1 + the
sister 5 from C8 etc.), ensure at least one candidate from each of the 4 quadrants is
included. Avoids concentrating effort in a single quadrant (e.g. all 5 candidates being
HNeRV-internal byte-axis would be the kitchen-sink anti-pattern).

---

## 8. Council adversarial review

- **Heuer (memorial seat).** "ACH is the heart of structured analysis; you must run it
  before any major substrate-direction decision. Don't anchor on the first plausible
  hypothesis." → **STRONG ENDORSE C1**.

- **Pherson (memorial seat).** "Pre-mortems and what-if are complementary; pre-mortem is
  for what already exists in your plan; what-if is for what is coming at you. Run both."
  → **STRONG ENDORSE C4 + C5**.

- **Yousfi.** "Red team analysis (C6) is the steganalysis discipline — you can't beat the
  scorer without modeling its blindspots from its OWN team's perspective." → **ENDORSE C6**.

- **Hotz.** "All these analytic techniques have a fixed time cost. If we're racing, ACH
  matrix takes 1 hour we don't have. Use them in PEACETIME (now), not in race window." →
  **AGREE.** Per CLAUDE.md "Race-mode rigor inversion", SAT discipline applies pre-leader-
  shift; in active race, the smallest credible bolt-on wins.

- **Contrarian.** "Structured analytic techniques are a CYA tool: they make decisions
  defensible, not BETTER. Did Iraq WMD ACH save anyone?" → **CHALLENGE.** Iraq WMD failure
  was NOT use of ACH; it was the analytic-community's REFUSAL to run ACH despite the
  Robert Gates / Sherman Kent guidance. The Heuer-Pherson book exists precisely because
  the IC did not systematically apply these techniques pre-Iraq. SATs WORK when applied;
  they fail when bypassed.

---

## 9. Reactivation criteria — for any deferred SAT

All SATs are process-only; reactivation is by operator directive in council deliberations.
No empirical reactivation criteria.

---

## 10. Citations (CIA structured analytic techniques)

- Heuer, R. J. Jr. *Psychology of Intelligence Analysis* (CIA, 1999): <https://www.cia.gov/static/9a5f1162fd0932c29bfed1c030edf4ae/Pyschology-of-Intelligence-Analysis.pdf>
- Heuer, R. J. Jr. & Pherson, R. H. *Structured Analytic Techniques for Intelligence Analysis* (CQ Press, 2010, ISBN 978-1608710188): <https://www.amazon.com/Structured-Analytic-Techniques-Intelligence-Analysis/dp/1608710181>
- CIA *Tradecraft Primer: Structured Analytic Techniques for Improving Intelligence Analysis* (Apr 2009): <https://www.cia.gov/resources/csi/static/Tradecraft-Primer-apr09.pdf>
- Klein, Gary. "Performing a Project Premortem" — Harvard Business Review (Sept 2007): <https://hbr.org/2007/09/performing-a-project-premortem>
- McKinsey "Bias Busters: Premortems": <https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/bias-busters-premortems-being-smart-at-the-start>
- "5 Structured Analytic Techniques for Better Decision-Making" — The Mind Collection: <https://themindcollection.com/structured-analytic-techniques/>
- "Improving your Intelligence Analysis with Structured Analytic Techniques" — Maltego: <https://www.maltego.com/blog/improving-your-intelligence-analysis-with-structured-analytic-techniques/>

---

## 11. Wire-in (Catalog #125)

1. Sensitivity-map: C1 ACH matrix's row-evidence-anchors feed sensitivity-prior on substrate
   selection.
2. Pareto: N/A (process-only).
3. Bit-allocator: N/A.
4. Cathedral autopilot dispatch: C4 what-if + C5 pre-mortem gate every $1+ dispatch
   (wire into autopilot's pre-dispatch journal).
5. Continual-learning posterior: every ACH row is a continual-learning anchor.
6. Probe-disambiguator: C7 quadrant-crunching forces multi-quadrant probe portfolios.

---

**End ledger 03.**
