---
council_tier: T3
council_attendees: [Schmidhuber_LEAD, Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Katsavounidis_invited, Elad_invited, Li_RC_invited, Caruana_invited, Kaplan_invited]
council_quorum_met: true
council_verdict: PROCEED
council_override_invoked: true
council_override_rationale: "operator 2026-07-30 verbatim: 'What would schmidhuber and the pantheon and the pantheon of pantheons and all minds they would invite or want to consult suggest and consider and wonder and pursue from here' — Schmidhuber LEAD retained from gc1/gc5/gc6 override lineage"
council_predicted_mission_contribution: frontier_breaking
council_dissent:
  - member: Contrarian
    verbatim: "the hyperbola-slide reading of the product law is a heuristic, not a constraint surface — do not let 2*sqrt(c) become a quoted bound without the label; and do not let the projection probe's L2-fit understate the floor (fit in the margin metric or state the metric gap)"
  - member: Hotz
    verbatim: "wr1 Gate B is a 34-minute measurement that has been staged for two days — run it before convening anything else ever again"
council_assumption_adversary_verdict:
  - assumption: "two endpoints at seg+rate ~= 0.77 means a seg+rate WALL"
    classification: CARGO-CULTED
    rationale: "the SUM framing hides that the PRODUCT c=(100*d_seg)*(rate) improved x0.60-0.68 burn-1 -> QA24 (0.129-0.148 -> 0.0881 SMEVR-priced). The wall claim is premature; the measured object is a per-burn product-improvement rate, and the honest question is whether burn-3's levers change its slope."
  - assumption: "the 25.58x amortization gap is attackable by training (96.1% typed attackable)"
    classification: CARGO-CULTED
    rationale: "'attackable' was typed against the exact-solve TEACHER's flip set (margin-decile membership), never against this renderer-class's capacity at this granularity. The capacity-vs-objective split is UNMEASURED — the projection probe (op-routable 2) is the classification's missing measurement."
  - assumption: "pose contribution ~0.03 is achievable on the selected parent"
    classification: CARGO-CULTED
    rationale: "su2's own table: every banked non-ideal pose term exceeds the bar by itself; 0.03 is ASSUMED pending the su2 stage-2/3 program. All bar arithmetic below carries this label."
council_decisions_recorded:
  - "op-routable 1: fire wr1 Gate-B staged exact gate (34 min) FIRST — su2 stage-1, the primary parent's unknowns"
  - "op-routable 2: PROJECTION PROBE — freeze QA24 endpoint renderer, fit tokens to the C1 solve frames (scorer-free), one n600 realized gate -> capacity-floor split of the 25.58x"
  - "op-routable 3: register product-law observable c=(100*d_seg)*rate as canonical equation + burn-3 seal input"
  - "op-routable 4: burn-3 adjudication DEFERRED until op-routables 1+2 land (~1 day) — the fork is measurable, not opinable"
related_deliberation_ids: [council_gc5_micro_macro_20260728, ddm_gc6_from_endpoint_20260729, ddm_gc8_postreversal_20260729, ddm_pn1_completion_20260728, ddm_ph3_hybrid_adaptive_20260731, ddm_su2_pose_endgame_20260730]
---

# ddm_gc9 — from-here convocation over the TWO-ENDPOINT state (11th; 2026-07-30)

Pointer honesty FIRST: submittable **0.1910828242 [contest-CPU] UNMOVED**. Official bar 0.172141
[contest-CUDA, external row]. T_3 = 0.15. Every number below [macOS-CPU advisory] unless marked;
composed predictions are gate-grade only via the fidelity law (3 anchors, residual 1.8e-6).

STORES CONSULTED: gc6/gc8/pn1/ph2/ph3 op-routable tables (no row repeated below unless its premise
changed — each such row says which premise) · su2 program memo (merged 6813679636) · sg1 QA74 typing
(da493fad26) · ja1 joint tables (ddm_ja1_joint_waterfill_table_20260731.json) · r7 S4 floor receipt ·
tr1_window_receipt.json + p1_receiver_realized_verdict.json (archive sha e7640dee9c3cf41d) ·
co9 SENSE laws · b2p/b2b prep memos.

## §0 The new facts the minds deliberate over (receipts)

- **E1 (burn-1)**: d_seg 0.0038892 n600 EMA @ pfs1-D1 archive 569,996 B (rate 0.37952); zero-init
  alt 499,587 B (rate 0.33262). MEASURED.
- **E2 (QA24)**: receiver-realized d_seg **0.0052766** (deploy parity −1e-6 vs trainer confirm;
  max 0.01556 @ pair 517) @ masked exporter archive **358,209 B** (rate 0.23850); SMEVR-priced token
  section **250,898 B** → rate 0.16705. MEASURED (p1_receiver_realized_verdict.json).
- **su2 Gate-B frame** (MEASURED bytes, never-fired distortion): Gate B = 174,578 B, rate 0.1162443;
  bar 0.172141 leaves 100·d_seg + √(10·d_pose) < **0.0558967** → d_seg < 5.59e-4 at pose 0;
  vs 0.15: < 0.0337557 → d_seg < 3.38e-4.
- **Existence proof**: exact C1 solve realizes d_seg **1.52e-4** through the real path — INSIDE the
  Gate-B seg budget with 3.7× slack at pose 0. The scorer-side target is feasible; the open question
  is REALIZING it inside Gate-B bytes with a renderer, i.e. rate-constrained amortization.
- **QA74 typing** (sg1): endpoint flips = ≥96.1% attackable gap (25.58× over the teacher), ≤3.9%
  exact-solve floor; 100% in the bottom GT-margin decile; Lane 38.7% of flips at 69.5× over its floor.

## §1 THE PRODUCT LAW (new; Schmidhuber LEAD + Shannon) — the two-endpoint measurement is a slope, not a wall

Define the joint observable **c = (100·d_seg) · (25·B/37,545,489)** (seg-term × rate-term).

| endpoint | seg term | rate term | c |
|---|---|---|---|
| E1 (D1 shipped) | 0.38892 | 0.37952 | **0.14760** |
| E1 (zero-init alt) | 0.38892 | 0.33262 | 0.12936 |
| E2 QA24 (SMEVR-priced) | 0.52766 | 0.16705 | **0.08815** |
| E2 QA24 (exporter-Brotli today) | 0.52766 | 0.23850 | 0.12585 |

Measured per-burn product improvement: **×0.60–0.68** (E1→E2, config-reachable SMEVR row).

Bar arithmetic (EXACT given c; the *slide* along a c-level curve is a heuristic — Contrarian label):
min(seg+rate | seg·rate=c) = 2√c at seg=rate=√c. With pose contribution 0.03 (**ASSUMED**, su2
stages 2–3 unproven): bar 0.172141 needs seg+rate < 0.14214 ⇒ **c ≤ 5.05e-3** (min-sum point:
d_seg 7.1e-4 @ ~106.8 KB — same order as the Gate-B frame, independent cross-check). T_3 0.15
needs c ≤ 3.60e-3. From E2's 0.08815 that is **×17.5** (bar) / **×24.5** (0.15). At the measured
×0.6/burn slope: ~5.6–6.3 more burns — serially infeasible at ~7 h/burn UNLESS burn-3's levers are
qualitatively stronger than QA24's form-fixes. Honesty labels: c rows MEASURED; 2√c bound EXACT-given-c;
hyperbola-slide CONJECTURE (formulation scope); pose 0.03 ASSUMED.

**Schmidhuber (compression-progress lens):** the campaign's progress scalar is Δlog c per unit
compute. QA24 bought Δlog c = −0.44 for ~7 h. The next compute unit should buy MEASUREMENTS that
change the SLOPE, not another point on the same slope. Ranked by information/minute: (1) wr1 Gate-B
gate — 34 min converts the primary parent's distortion from UNKNOWN to MEASURED; (2) the projection
probe (§2) — splits the 25.58× into capacity-floor × optimization-gap for ~hours of scorer-free fit;
(3) S2 ν-audit ($0, pre-capacity re-route, pivot [0.55,0.75] UNMEASURED); (4) only then a burn-3
decision. A burn fired before (1)+(2) is compute spent on an unmeasured slope.

## §2 THE PROJECTION PROBE (new; rank-1 NEW measurement) — split the 25.58× before spending a burn on it

The QA74 "96.1% attackable" typing classifies flips by *teacher* margin membership — it never
measured whether THIS renderer class at THIS granularity can express the teacher's content. The
cheapest split: **freeze the QA24 endpoint renderer (EMA), fit ONLY the tokens (cell-mask applied,
quant-engaged STE) to reproduce the C1 solve frames** (targets already materialized by b2p's QA75
prep), scorer-free per-pair regression; then ONE n600 realized gate on the fitted tokens.
Output: **f = the renderer-class conditional capacity floor** at current granularity/rate.

- f ≤ ~5.6e-4-class ⇒ the vehicle can EXPRESS Gate-B fidelity; the 25.58× is dominated by the
  training objective (GT-infeasible targets) ⇒ **QA75 distill burn-3 is well-founded** — its
  gradient replaces unreachable GT margins with the realizable solve configuration.
- f ≥ ~2.6e-3-class (≥half the endpoint) ⇒ renderer-class/granularity CAPACITY wall at
  INSTANCE(QA24 geometry) scope ⇒ burn-3-as-configured cannot buy 10×; route to the capacity fork
  (ν re-route / granularity re-race / renderer-class change) with the floor as the design number.
- Between: mixed; the distill-window probe (§3 row 5) arbitrates.

Contrarian rider (recorded): fit in a margin-aware metric or report the L2↔margin metric gap
alongside f. Fridrich rider: fit tokens THROUGH the deployed quant path (uint8/STE engaged), else
f inherits the deploy gap.

## §3 SUGGEST — ranked pursue-next (S-arithmetic first; existing-owned rows cited, not duplicated)

| # | mechanism (receipt) | next MEASUREMENT | falsifier | S-arithmetic vs 0.172141 / 0.15 | consumer |
|---|---|---|---|---|---|
| 1 | wr1 Gate-B staged gate (su2 stage-1; 174,578 B MEASURED, distortion UNKNOWN) — EXISTING-OWNED #766, premise changed: now the PRIMARY parent per su2 | the staged ~34-min exact n600 gate | realized seg-term at Gate B ≫ 0.056 ⇒ Gate B not target-capable on burn-1 tokens → burn-3 must produce the parent | Gate B is the ONLY measured byte-frame where bar arithmetic closes (d_seg<5.59e-4 at pose 0) | #766→#782; su2 stage 2 |
| 2 | **PROJECTION PROBE** (§2; NEW) | scorer-free token fit to C1 solve frames + 1 gate | f ≥ 2.6e-3 ⇒ capacity wall INSTANCE(QA24-geometry) | decides whether ANY burn at this class can reach c ≤ 5.05e-3 | burn-3 seal; §4 fork |
| 3 | **PRODUCT LAW registration** (§1; NEW) | $0 — canonical equation + per-burn Δlog c telemetry | — (observable) | the ×17.5 / ×24.5 target is the burn-3 GO bar | canonical_equations; burn-3 seal; costate SENSE |
| 4 | S2 ν-audit — EXISTING-OWNED (pn1), premise changed: now a burn-3 PRE-CONDITION | $0 re-route of sealed burn pre-capacity; ν-pivot [0.55,0.75] | ν in-band confirmed ⇒ no re-route | reroutes capacity before the expensive fork | pn1 S2; burn-3 config |
| 5 | QA75 distill-WINDOW probe (NEW discriminator; gated on row 2 low-f) | 30–60 ep resume from E2 w/ real QA75 lever; slope ratio vs plain continuation at matched steps | slope ratio ≈1× ⇒ distill does not unlock the gap at this capacity (formulation scope) | if ≥3× early: 10×-in-one-burn plausible; else fork | ph3 §10 QA75; burn-3 seal |
| 6 | su2 pose program stages 2–5 on the SELECTED parent — EXISTING-OWNED #775/#782/QA43 (merged, build-ready) | nested warp-tail k=56→112→200 + TT1 + QA66 refit | whole-action >600 B/admitted pair | converts pose 0.03 from ASSUMED to MEASURED — every bar row depends on it | #782 chain |
| 7 | QA24-tail correction re-grade (NEW premise: E2 max 0.01556, fatter tail than E1) | band classifier + white-jitter re-price on E2's top-24 tail pairs | corrections still break-even at tail base | co9's break-even was measured at E1's base; E2's tail may flip per-pair corrections positive | co9; menu1 round-2 |
| 8 | (d_seg, rate) CONVEX HULL chart over ALL measured rows (Katsavounidis lens; NEW, $0 apparatus) | assemble from existing receipts; mark hull vs dominated rows | — | prevents pursuing interior points; the burns must move the hull, not fill it | costate SENSE; burn-3 seal |

## §4 CONSIDER — the central fork, adjudicated to a measurement (not an opinion)

Fork: burn-3 ph3-§10 stack vs ν/capacity re-route vs renderer-class change vs compose-forward.
**Council verdict: NOT ADJUDICABLE TODAY — and that is itself the finding.** The fork is spanned by
two cheap orthogonal measurements: row 1 (what does the 2× rate cut cost in seg on current tokens)
and row 2 (what is the renderer-class seg floor at this rate class). Decision table:

- row2 f low AND row1 Gate-B seg-cost small → burn-3 (QA75>QA81>QA80 + QA83/84/86) at Gate-B-target
  rate, seal requires projected Δlog c ≥ log(17.5)/1-ish honesty check against §1.
- row2 f low AND row1 cost large → burn-3 must RE-PRODUCE the parent (rate-in-loss at Gate-B target;
  Li λ-domain form, §5) rather than post-hoc waterfill.
- row2 f high → capacity fork: ν re-route (row 4) + granularity re-race (gc6 row 10 stays BLOCKED
  until this) + renderer-class change enters design; QA75 alone cannot buy 10× (formulation scope).
- compose-forward on the own-vehicle line (0.9639878) continues REGARDLESS as the local-line
  refinement — it is not in tension with the fork; it just does not reach 0.172 (su2 arithmetic).

QA75 derivation honesty (asked-for): distill removes the TARGET-INFEASIBILITY component of the
25.58× (GT margins unrealizable through R/uint8 — 100% of flips in the bottom GT-margin decile is
the signature), leaves the CAPACITY component. Whether that buys 10× at ≤200 KB is exactly the
f-split — the derivation cannot say without row 2. DERIVED, not measured.

## §5 INVITE — pantheon-of-pantheons consultees for a seg×rate product wall (one row each; lessons-only where banned-lineage)

| mind | what they see that we missed | next measurement |
|---|---|---|
| Katsavounidis/Aaron (per-title encoding) | we optimize configs, not the HULL; per-content ladders pick the hull knee per title — we have one title and dozens of measured (d_seg, rate) rows never assembled | §3 row 8 hull chart ($0) |
| Elad/Aharon (K-SVD) | the token dictionary should be learned from the SOLVE frames (feasible configurations), not emerge from GT descent; atoms = solve-token patterns | K-SVD on row-2's fitted token field; price dictionary+codes vs SMEVR |
| Li (λ-domain rate control) | our w_rate is a derived CONSTANT (0.0768); λ-domain RC ties it to a BYTES TARGET — w_rate(t) solved so the running coded size lands on 174,578 B | recompute b2b w_rate law in bytes-target form; burn-3 lever |
| Caruana (born-again distill) | one distill generation is rarely the last; generation-2 (student re-distilled with teacher+self) historically beats gen-1 at fixed capacity | QA75 gen-2 window after row 5, same falsifier |
| Kaplan/Hestness (scaling laws) | a 10× ask with zero measured capacity-scaling points is blind; two capacities at matched schedule give the local seg-vs-params exponent | 2-point capacity curve inside the burn-3 budget (small arm) |
| Marpe (CABAC) — lessons-only | context-mixing over token neighborhoods | already covered: SMEVR contexts + QA08 MIX pool (gc8 row 8); no new row |

## §6 WONDER — typed open questions (measurement per row)

- W1 (OPEN, decisive): capacity vs objective split of the 25.58× — answered by §2. MEASURABLE-NOW.
- W2 (OPEN): does the Gate-B token subset retain the separatrix-annulus content? — row 1's gate.
- W3 (OPEN): ν-pivot [0.55,0.75] — S2 audit ($0).
- W4 (OPEN): is the correction break-even premise base-dependent at E2's fatter tail? — row 7.
- W5 (OPEN, the product law's own question): is c-slope a config property or a class property —
  does ANY same-class burn beat ×0.6/burn? Only burn-3-after-rows-1+2 answers; register Δlog c
  telemetry now so the answer is free. CONJECTURE until then.
- W6 (OPEN, honesty): pose 0.03 assumption — su2 stages 2–5 (row 6). Every bar row inherits it.

## §7 Per-member operating-within assumptions (sextet + leads; one line each)

Schmidhuber_LEAD: "compute buys slope-changing measurements first" — operating within: Δlog c is the
right progress scalar (HARD-EARNED: two measured endpoints). Shannon: bar decompositions are exact
identities (HARD-EARNED). Dykstra: Gate-B frame = feasible-set intersection; XOR discipline from su2
(HARD-EARNED). Rudin: every §3 row is a falling rule with falsifier (HARD-EARNED). Daubechies:
annulus sparsity ⇒ the content is codeable below GT-pixel rate (HARD-EARNED: 100% bottom-decile).
Yousfi: margin-decile membership is detector-side truth (HARD-EARNED). Fridrich: deploy path
(uint8/STE) inside every probe (HARD-EARNED: the 0.443 cell-mask incident). Contrarian + A-A:
frontmatter verbatims. Quantizr: pose rides a stored-target-shaped stream, values joint-solved
(HARD-EARNED L68). Hotz: run the 34-minute thing (verbatim). MacKay: the product law is MDL in
disguise — c is a two-part code cost cross-term (CARGO-CULTED until registered with an evaluator).

## §8 Routing (defer-at-source)

Rows 1/4/6 = EXISTING-OWNED (#766, pn1-S2, #775/#782) — elevated order only, no new arms. NEW rows
2/3/5/7/8 route to: row 2+5 → burn-3 pre-seal chain (MAIN slot decisions); row 3 → canonical
equation at next equations-touching landing; rows 7/8 → $0 batch at next free analysis window.
NO launches from this memo; actuation stays with MAIN.

Pointer delta: **UNMOVED** (0.1910828242 [contest-CPU]). This memo is means, not end.
