---
council_tier: T3
council_attendees:
- Shannon
- Dykstra
- Rudin
- Daubechies
- Yousfi
- Fridrich
- Contrarian
- Assumption-Adversary
- MacKay
- Schmidhuber
- Atick
- Redlich
- Tishby
- Rao
- Ballard
- Carmack
- Hotz
- PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
- member: Contrarian
  verbatim: "the 2026-05-29 symposium already ruled the UNIWARD 5th/6th nulls reactivation territory and the Z6-v2 pose floor a genuine ceiling. Re-auditing those verbatim is the cargo-culted ritual I flagged last time. This audit is admissible ONLY because the 2026-05-31 Z8 + range-coder cluster is genuinely fresh and was not in scope on 2026-05-29. I VETO any verdict row that merely restates a prior adjudication without new empirical grounding."
- member: Assumption-Adversary
  verbatim: "the operator's standing question 'were the negatives given OPTIMAL FORM?' (the #1585/#1586 cargo-cult-unwind lens) cuts BOTH ways. Surfacing false reactivation hope on a genuine ceiling is itself a cargo-cult — it burns the next dispatch re-confirming a known floor. Two of today's negatives are GENUINE CEILINGS at the surface measured (the range-coder DEFER, the dead-zone-alone gap) and I will not let the council manufacture reactivation theater for them."
- member: Carmack
  verbatim: "stop confusing the distortion paradigm being intact with the lane being viable. Z8 is 546x from frontier and 99.5% of that is one blob of raw float32. The pose collapse is real and beautiful and irrelevant until the rate blob is quantized. The ONLY honest reactivation is the per-subband delta operating point. Everything else on Z8 is polishing a corpse."
council_assumption_adversary_verdict:
- assumption: "a negative recorded at lifted-trainer form is reactivation territory, not a kill"
  classification: HARD-EARNED
  rationale: "NSCS06 v6->v7 44% improvement via cargo-cult-unwind is the canonical empirical anchor (Catalog #315 source). Applies to lifted-trainer negatives; does NOT apply to optimal-form-tested ceilings."
- assumption: "the Z8 detail-coefficient blob is THE rate lever and quantize+entropy-code is the only path to close the 546x gap"
  classification: HARD-EARNED
  rationale: "Measured on REAL Z8HPC1 archive (z8_detail_entropy_headroom_20260531T185438Z.json, evidence_grade=advisory_cpu): blob is ~99.5% of archive bytes; raw-f32->brotli pays 3.2-3.7 B/coeff for |c|~0.014-0.036; stored 1000-4000x above Shannon floor. This is source-inspection + empirical-anchor verified, not inferred."
- assumption: "the constriction range coder is already per-subband-optimal so the '5-10% better entropy coder' recovery does not exist"
  classification: HARD-EARNED
  rationale: "Empirically measured: forcing a single global range coder is -3.9%/-0.5% WORSE than the per-subband split; v2 codec within 5-10% of per-subband Shannon floor (sometimes below via cross-subband context). This is a genuine local-optimum, not a cargo-cult."
- assumption: "categorical hierarchical-PC closes the pose axis (99.94% reduction is a real win)"
  classification: HARD-EARNED
  rationale: "Catalog #382 currency check: token categorical_posterior_capacity_vs_continuous_gaussian_v1 returns None (NOT flipped to FALSIFIED/PHANTOM). 6 anchors. The 99.94% pose reduction is current. NOTE: the equation's stored residual (predicted 0.192 vs empirical 0.9353 = 0.7433) is a CALIBRATION drift not a falsification — the predicted band was the SCORE-form 0.192 frontier, mistakenly compared against the EFFECT-SIZE-form 0.9353 pose-reduction fraction. Unit-mismatch in the anchor, not a paradigm failure. PROVISIONAL flag raised."
- assumption: "the UNIWARD 5th/6th nulls were cargo-culted (not optimal form) and are reactivation territory"
  classification: CARGO-CULTED (the re-audit, not the finding)
  rationale: "ALREADY adjudicated 2026-05-29 (council_negative_findings_falsifications_extreme_rigor_audit_20260529.md): 5th/6th IMPLEMENTATION-LEVEL, applied before the entropy-coded sidecar surface existed, reactivation territory. Re-classifying them today is the Contrarian's flagged ritual. CITED, not re-litigated."
council_decisions_recorded:
- "op-routable #1 (TOP EV): per-subband delta operating-point water-fill (joint P18/P19, tasks #1591/#1592) is the ONLY honest Z8 reactivation. The dead-zone actuator (ad73c2863) + the per-subband RD curve (z8_detail_entropy_headroom JSON) are BOTH landed; the solver has its inputs. Predicted cost: $0 macOS-CPU/MLX (no paid GPU). This is NOT a re-test of a negative; it is the FIRST optimal-form attempt at the rate axis. Highest EV/cost."
- "op-routable #2: re-anchor categorical_posterior_capacity_vs_continuous_gaussian_v1 to fix the predicted-vs-empirical UNIT MISMATCH (0.192 score-form vs 0.9353 effect-size-form). The 0.7433 residual is a recalibration artifact, not a paradigm failure — Catalog #371 auto-recalibrator should re-derive from the 6 landed anchors in effect-size units. Predicted cost: $0 (registry edit). Prevents a phantom-residual being read as a falsification by a future Catalog #382 consumer."
- "op-routable #3: matched-300ep tau-anneal ablation on DreamerV3 RSSM v2 to isolate the 2x-epoch + cosine-LR confound from the 1.1% tau-anneal gain. The full-budget 99.21% claim is CONFOUNDED per the v2 memo's own honest verdict. Predicted cost: $0 MLX-LOCAL. Disambiguates a PROVISIONAL claim."
- "VERDICT: the range-coder DEFER (df9cd8bec / today) is a GENUINE LOCAL OPTIMUM at the entropy-coder surface, NOT reactivation territory. Per-subband constriction is optimal; the global-coder 'recovery' was empirically -3.9%/-0.5% worse. Do NOT manufacture reactivation hope. The rate lever moved UPSTREAM to the per-subband delta (op-routable #1), not to a better coder."
- "VERDICT: the dead-zone-ALONE gap (ad73c2863, Z8 still ~24x from frontier) is IMPLEMENTATION-LEVEL paradigm-intact. The actuator works and joint P18/P19 beats magnitude at every keep fraction; dead-zone alone cannot close 24x because surviving coeffs are still stored as raw quantized values without an arithmetic coder operating-point. Reactivation = op-routable #1 (the water-fill is the missing operating-point lens, NOT a different actuator)."
- "VERDICT: the seg structural ceiling (~2.68, STILL DESCENDING at 2000ep) is a GENUINE distortion ceiling for the current Z8 decoder under 100*d_seg dominance. NOT a kill (the curve is still descending = more epochs/decoder capacity could move it). Reactivation = decoder/SegNet lever (per the long-run + dreamer-v2 memos), but this is LOWER EV than the rate axis because seg is only ~2.9 of 104.94 (the rate term is 102)."
- "re-audit cadence (re-affirmed from 2026-05-29): convene ONLY when a NEW negative cluster lands, not on a timer. Today's cluster (Z8 + range-coder + dead-zone, all 2026-05-31) is the qualifying new cluster."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "need a grand council symposium subagent to review all negative findings and results and falsifications, extreme rigor audit"
self_reflection_round: 3
self_reflection_findings:
- "Round 1->2 reclassification: the categorical_posterior 0.7433 residual was initially read as a near-falsification; Round 2 source-inspection (Catalog #382 None + anchor unit inspection) re-classified it as a unit-mismatch CALIBRATION artifact -> verdict status PROVISIONAL-PENDING-VERIFICATION until op-routable #2 re-anchors it."
- "Round 2->3 SEAL: zero material unverified-assumption findings remain. All 5 assumption-adversary verdicts are HARD-EARNED (source-inspection or empirical-anchor) except the one CARGO-CULTED re-audit flag (which is a procedural finding, not a substrate claim). The PROVISIONAL flag on the categorical equation is explicitly recorded and routed to op-routable #2."
---

# Grand Council Symposium — Negative Findings & Falsifications Extreme Rigor Audit (2026-05-31)

## Operator override (Catalog #300 Consequence 1)

> need a grand council symposium subagent to review all negative findings and results and falsifications, extreme rigor audit

This is an operator-frontier-override authorizing an over-cadence T3 grand
council. Roster validated COMPLETE per Catalog #346
(`tac.canonical_council_roster.validate_council_dispatch_roster`:
`complete=True`, `missing_co_leads=None`): 4 co-leads (Shannon LEAD /
Dykstra / Rudin / Daubechies) + sextet (Yousfi / Fridrich / Contrarian /
Assumption-Adversary + Shannon + Dykstra) + topical PC specialists
(Atick / Redlich / Tishby / Rao / Ballard) + MDL specialists
(MacKay / Schmidhuber) + engineering-ceiling specialists (Carmack / Hotz)
+ PR95Author.

## Scope discipline (per the 2026-05-29 Contrarian binding dissent)

The prior symposium (`council_negative_findings_falsifications_extreme_rigor_audit_20260529.md`)
comprehensively adjudicated: UNIWARD cascade (5th/6th/7th order),
Z5/Z6-v2 pose floors, V14 PR-substitution, C6 IBPS 22x, MPS-drift. Its
Contrarian recorded a BINDING dissent: *"re-auditing the same
falsifications each session is itself a cargo-culted ritual unless each
audit surfaces a NEW reactivation that was not visible before."*

This audit therefore CITES those prior verdicts and builds forward ONLY on
the genuinely-fresh **2026-05-31 Z-stack + range-coder + dead-zone**
cluster, which was NOT in scope on 2026-05-29. Prior verdicts are
preserved per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

## Per-finding adjudication (EXTREME-RIGOR CONTRACT)

Each row carries: (a) Catalog #307 paradigm-vs-implementation class;
(b) Catalog #311 substrate-compatibility evidence; (c) OPTIMAL-FORM-vs-
CARGO-CULTED verdict (#1585/#1586 lens); (d) Catalog #382 currency check;
(e) reactivation paths; (f) Catalog #363 empirical_verification_status.

### Finding 1 — Z8 600-pair byte-closed contest score 104.94 [contest-CPU advisory], ~546x from 0.192 frontier

- **(a) #307 class:** IMPLEMENTATION-LEVEL. The hierarchical-PC PARADIGM is
  intact — pose collapses 99.94%, seg descends to a structural ceiling
  ~2.68. The 546x gap is ENTIRELY the rate term, and the rate term is
  ENTIRELY raw-float32 detail-coefficient storage (~99.5% of archive). This
  is a rate-encoding implementation problem, NOT a distortion paradigm
  failure.
- **(b) #311 compat-evidence:** measured against the REAL Z8HPC1 archive
  byte-closed through the canonical inflate path (NOT an incompatible
  grammar, NOT a synthetic fixture). The frontier comparison uses the
  canonical pointer (0.1920282830, lane dqs1_rank021). COMPATIBLE.
- **(c) OPTIMAL-FORM verdict:** the DISTORTION axis was given optimal form
  (2000ep, real Hinton-distilled SegNet KL T=2.0 + PoseNet teacher, EMA,
  tau-anneal). The RATE axis was NEVER given optimal form — the archive
  stores raw f32 detail coeffs with NO quantize/entropy-code operating
  point. This negative is RATE-AXIS-CARGO-CULTED: the rate lever was never
  attempted at optimal form. REACTIVATION TERRITORY on the rate axis.
- **(d) #382 currency:** the load-bearing pose claim token
  `categorical_posterior_capacity_vs_continuous_gaussian_v1` returns None
  (not flipped). Current.
- **(e) reactivation:** (1) per-subband delta operating-point water-fill
  (joint P18/P19, #1591/#1592) — $0; (2) per-subband joint thresholds; (3)
  full 600-pair operating-point keep sweep.
- **(f) #363 status:** VERIFIED_VIA_EMPIRICAL_ANCHOR (byte-closed 600-pair
  measurement + REAL archive entropy-headroom report).

### Finding 2 — Z8 joint P18/P19 dead-zone rate attack DEFER (ad73c2863), Z8 still ~24x from frontier

- **(a) #307 class:** IMPLEMENTATION-LEVEL. The actuator works (real SegNet
  boundary saliency + real PoseNet pixel-Jacobian pushed to Mallat detail
  domain via exact-adjoint analysis DWT). JOINT beats MAGNITUDE at EVERY
  keep fraction (J-mag -0.049 @keep=0.02 ... -0.841 aggressive). P19
  pose-protection is REAL and GROWING (+0.66 d_pose preserved @keep=0.02).
- **(b) #311 compat-evidence:** 14 NO-FAKE tests verify coeffs ACTUALLY
  zeroed, brotli rate ACTUALLY shrinks, splice byte-identical, real scorer
  saliencies vary + sparse. Tested on the REAL Z8 archive (Catalog #213).
  COMPATIBLE.
- **(c) OPTIMAL-FORM verdict:** the dead-zone actuator IS optimal form for
  what it is (the honest mid-lane design correction — multiply |coeff| by
  saliency rather than replace — is the canonical RD-energy-aware keep). BUT
  dead-zone ALONE was never the full lever: surviving coeffs are still
  stored as raw quantized values without an arithmetic-coder operating
  point. The 24x residual gap is because the OPERATING POINT (per-subband
  delta) is unsolved, not because the actuator is wrong. REACTIVATION =
  the water-fill solver (op-routable #1), NOT a different actuator.
- **(d) #382 currency:** no flipped token. Current.
- **(e) reactivation:** (1) quantize + entropy-code surviving coeffs (the
  dead-zone gives the sparsity prior; the arithmetic-coder operating point
  is the missing lever); (2) per-subband joint thresholds; (3) full
  600-pair operating-point keep sweep.
- **(f) #363 status:** VERIFIED_VIA_EMPIRICAL_ANCHOR (8-pair smoke + 14
  NO-FAKE behavioral tests).

### Finding 3 — TODAY's range-coder DEFER (df9cd8bec / z8_detail_entropy_headroom report footer)

- **(a) #307 class:** PARADIGM-LEVEL for the SPECIFIC claim "a 5-10% better
  entropy coder exists." It does NOT. The constriction range coder is
  ALREADY per-subband-optimal; forcing a single global range coder is
  -3.9%/-0.5% WORSE. The v2 codec is within 5-10% of the per-subband
  Shannon floor (sometimes below via cross-subband context). This is a
  GENUINE LOCAL OPTIMUM at the entropy-coder surface.
- **(b) #311 compat-evidence:** measured on the REAL Z8HPC1 archive's
  detail blob (`z8_detail_entropy_headroom_20260531T185438Z.json`,
  evidence_grade=advisory_cpu, axis=advisory_cpu, REAL archive). Decode
  fits the 30-min window trivially (0.92s). COMPATIBLE.
- **(c) OPTIMAL-FORM verdict:** GENUINE CEILING at the surface measured. The
  entropy coder WAS given optimal form (per-subband constriction beats
  every alternative tested incl. global range coder + lossless byte-shuffle
  which is a measured DEAD END at 3.36-3.40 B/coeff). Do NOT manufacture
  reactivation hope for "a better coder." The rate lever moved UPSTREAM to
  the per-subband DELTA (the quantization operating point), not to the
  coder. This is the Assumption-Adversary's explicit guard against
  reactivation theater.
- **(d) #382 currency:** no flipped token. Current.
- **(e) reactivation:** NOT the coder. The ONLY open knob is the per-subband
  delta operating point (= op-routable #1 water-fill). The report supplies
  the per-subband RD curve as that solver's input.
- **(f) #363 status:** VERIFIED_VIA_EMPIRICAL_ANCHOR (REAL archive RD curve
  + global-vs-per-subband measurement).

### Finding 4 — Z8 seg structural ceiling (~2.68, STILL DESCENDING at 2000ep)

- **(a) #307 class:** IMPLEMENTATION-LEVEL (decoder-capacity ceiling), NOT
  paradigm-level. The curve is STILL DESCENDING (56.98% reduction 6.45 ->
  2.68 and not flat) = the ceiling is the current decoder/epoch budget, not
  a fundamental limit.
- **(b) #311 compat-evidence:** measured under real Hinton-distilled SegNet
  KL T=2.0 teacher (the long-run memo confirms m12a_score_binding=
  real_segnet_posenet_hinton_t2, pose[ep0]=104.6 non-zero = NOT mock per
  Catalog #322). COMPATIBLE — real scorer, not phantom.
- **(c) OPTIMAL-FORM verdict:** PARTIALLY optimal form (real teacher, 2000ep)
  but the seg axis is LOWER EV than the rate axis — seg is only ~2.9 of
  104.94 while the rate term is 102. Reactivation exists (decoder/SegNet
  lever) but is dominated by op-routable #1.
- **(d) #382 currency:** no flipped token. Current.
- **(e) reactivation:** decoder/SegNet capacity lever (per long-run +
  dreamer-v2 memos). LOWER priority — fix the rate axis first.
- **(f) #363 status:** VERIFIED_VIA_EMPIRICAL_ANCHOR (2000ep 600-pair real
  teacher run).

### Finding 5 — DreamerV3 RSSM v2 tau-anneal full-budget 99.21% claim (CONFOUNDED)

- **(a) #307 class:** PARADIGM_INTACT IMPLEMENTATION-LEVEL improvement (per
  the v2 memo's own honest verdict). The matched-budget @ep299 gain is a
  SMALL real ~1.1% rel (pose 7.0036 -> 6.9266). The full-budget 99.21%
  (pose 0.85) is CONFOUNDED by 2x epochs + cosine LR.
- **(b) #311 compat-evidence:** 18 NO-FAKE tests incl. a Class-2 guard
  (`test_tau_actually_changes_per_epoch_not_static`) that FAILS if forward
  reverts to a static read. The tau-anneal closes a real comment-only-
  contract bug. COMPATIBLE.
- **(c) OPTIMAL-FORM verdict:** the matched-budget ablation IS optimal form
  (apples-to-apples @ep299). The full-budget claim is NOT apples-to-apples
  (confound). The 1.1% matched gain is the honest number; the 99.21% is a
  confounded artifact. NOT a kill (the matched gain is real); the
  full-budget number must be DISAMBIGUATED.
- **(d) #382 currency:** the 5th anchor on the categorical equation carries
  the SAME unit-mismatch residual artifact (predicted 0.192 vs empirical
  0.9353 = 0.7433). This is a CALIBRATION artifact (score-form vs
  effect-size-form unit mismatch), NOT a falsification flip (#382 returns
  None for the token). PROVISIONAL flag -> op-routable #2.
- **(e) reactivation:** matched-300ep tau-anneal ablation to isolate the
  2x-epoch + cosine-LR confound from the 1.1% tau-anneal gain (op-routable
  #3). $0 MLX-LOCAL.
- **(f) #363 status:** PROVISIONAL-PENDING-VERIFICATION (the matched-budget
  gain is VERIFIED_VIA_EMPIRICAL_ANCHOR; the full-budget 99.21% is
  ASSUMED_AWAITING_VERIFICATION pending the confound-isolating ablation).

## Recursive self-reflection (Catalog #363)

- **Round 1 (deliberate):** all 5 findings classified with #307 / #311 /
  optimal-form / #382 / reactivation rows. The categorical equation's
  0.7433 residual was initially flagged as a near-falsification.
- **Round 2 (self-reflect + re-classify):** source-inspection (Catalog #382
  returns None for the token + anchor-unit inspection) re-classified the
  0.7433 residual as a UNIT-MISMATCH calibration artifact (score-form 0.192
  vs effect-size-form 0.9353), NOT a falsification. Verdict status on
  Finding 5 downgraded to PROVISIONAL-PENDING-VERIFICATION and routed to
  op-routable #2. The range-coder DEFER (Finding 3) was re-confirmed as a
  GENUINE LOCAL OPTIMUM (the Assumption-Adversary's anti-reactivation-
  theater guard) rather than reactivation territory.
- **Round 3 (resolve / SEAL):** zero material unverified-assumption findings
  remain. The one PROVISIONAL claim (Finding 5 full-budget) is explicitly
  recorded + routed. SEAL conditions met (3 consecutive clean passes on the
  assumption-classification axis).

## The honest bottom line (no manufactured hope)

Of the 5 fresh negatives:
- **2 are GENUINE CEILINGS at the surface measured** (Finding 3 range-coder
  local optimum; Finding 4 seg decoder ceiling — though Finding 4's curve
  is still descending so it is a soft ceiling). Do NOT re-test the entropy
  coder. Do NOT manufacture reactivation theater for "a better coder."
- **2 are RATE-AXIS-CARGO-CULTED** (Finding 1 + Finding 2): the DISTORTION
  axis was given optimal form but the RATE axis (per-subband delta
  operating point) was NEVER attempted at optimal form. This is the genuine,
  NEW, high-EV reactivation that was not visible as a single coherent lever
  until today's entropy-headroom report supplied the per-subband RD curve.
- **1 is CONFOUNDED-PROVISIONAL** (Finding 5): the matched gain is real
  (1.1%); the full-budget 99.21% needs a confound-isolating ablation.

The unifying NEW insight: the entropy-headroom report (today) + the
dead-zone actuator (ad73c2863) together convert "Z8 has a 546x rate gap"
from a vague negative into a SOLVABLE per-subband delta water-fill with
both inputs landed. That is the single op-routable that turns a corpus of
Z8 negatives into a frontier-breaking actuator.

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: ACTIVE (the per-subband RD curve IS the rate
  sensitivity map; routed to op-routable #1).
- hook #2 Pareto: ACTIVE (the per-subband rate-vs-distortion frontier is the
  Pareto polytope the water-fill solves; Dykstra co-leads this).
- hook #3 bit-allocator: ACTIVE (per-subband delta = per-subband bit
  allocation; op-routable #1 IS a bit-allocator hook).
- hook #4 cathedral autopilot: N/A — all Z8 findings are
  [advisory]/[research-signal] non-promotable; no contest-archive dispatch
  is gated by this council memo (it is READ + MEMO only per Catalog #340).
- hook #5 continual-learning posterior: ACTIVE (this deliberation appends a
  CouncilDeliberationRecord anchor; op-routable #2 re-anchors the
  categorical equation to fix the unit-mismatch residual).
- hook #6 probe-disambiguator: ACTIVE (the council disambiguated
  genuine-ceiling vs rate-axis-cargo-culted vs confounded-provisional —
  preventing reactivation theater on the 2 genuine ceilings).

## Cross-references

- `council_negative_findings_falsifications_extreme_rigor_audit_20260529.md`
  (prior symposium — UNIWARD 5th/6th/7th + Z5/Z6-v2 + V14 + C6 + MPS-drift;
  CITED not re-litigated).
- `feedback_z8_600pair_byte_closed_contest_score_advisory_landed_20260531.md`
  (Finding 1 anchor).
- `feedback_z8_joint_p18_p19_deadzone_rate_attack_landed_20260531.md`
  (Finding 2 anchor; commit ad73c2863).
- `feedback_z8_detail_entropy_headroom_report_landed_20260531.md` +
  `z8_detail_entropy_headroom_20260531T185438Z.json` (Finding 3 anchor;
  REAL archive RD curve).
- `feedback_z8_hier_pc_full_stack_longrun_landed_20260531.md` (Finding 4
  anchor; seg ceiling).
- `feedback_dreamer_v3_rssm_v2_tau_anneal_landed_20260531.md` (Finding 5
  anchor; confounded full-budget).
- CLAUDE.md "Forbidden premature KILL without research exhaustion" +
  "KILL/FALSIFIED memory verdicts" + Catalog #307/#311/#315/#325/#346/#363/
  #382 (the rigor framework this audit enforces).
