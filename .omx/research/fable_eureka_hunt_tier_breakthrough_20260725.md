# FABLE EUREKA HUNT — full-spectrum map for tier score lowering (2026-07-25)

`research_only=true` · `score_claim=false` · `promotion_eligible=false` · pointer
`0.1910828242 [contest-CPU]` UNMOVED · effective competitive frontier `0.172 [official
leaderboard display]` (PR130, per routing card §6) · every number below tagged
MEASURED / DERIVED / CONJECTURE / OPEN-QUESTION. Operator directives honored: comprehensive
view ("suggest everything necessary for frontier score lowering. Even if it's not immediately
measurable"), free-rein method, honesty-on-scores, the two permanent bans.

## §0 Method + coverage (what was actually enumerated this session)

- Durable inventory counts (run live this session): **407 canonical equations**
  (`tools/list_canonical_equations.py --json`) · **363 mapped + 75 unmapped levers**
  (`lever_registry.completeness()`; the 75 unmapped flag names listed in session log) ·
  **1,218 unique FEED-* tags / 2,152 FEED mentions** in the DAG · **6,689 md files** in
  `.omx/research/` (~9,346 entries incl. dirs/JSON) · **2,007 memory files** ·
  task ledger `canonical_task_status.jsonl` = **84 rows** (37 completed / 33 pending /
  9 blocked / 5 in_progress; NOTE: the ledger's own row marks it SUPERSEDED for witness
  tasks — the live TaskList + DAG is SoT, so "700+ rows" from the charter was not
  reproducible here; recorded honestly).
- Two parallel sweep agents: (1) `.omx/research` orphaned-positives + scoped-negatives
  (24 items returned, all receipt-cited); (2) memory-dir negative→cure adjacencies +
  orphaned laws (19 files deep-read; 8 cure pairs + 10 orphan pools returned).
- MAIN-thread primary-artifact verifications I ran myself (not delegated): the ev1
  receipt histogram custody, the ev2 seven-home lineage table, the m6/m7/joint-575
  lineage trace, the routing card §1–§6, the optimal-start card §1–§23, negative
  register auditor-A in full, is1 DAG FEED, rd1 receipt inventory, costate digest.

## §1 The game state in three measured facts (the frame every item below sits in)

1. **Distortion is solved; description is not.** Exact uint8-lattice solve: d_seg
   1.51969e-4 (17,927/117,964,800 errors), d_pose 1.0184e-4, survives uint8+real-R
   [MEASURED, is1/rd1 receipts]. Raw realization = 409,526,925 B (rate-dead). The
   compact-vs-raw description span is ~2,000×, measured at both ends (routing card §2).
2. **Sub-0.15 is a byte number.** At the solved distortion, S<0.15 ⟺ B ≤ 154,522
   [DERIVED from evaluate.py score law, m6/is1 receipts]. PR130 proves the reachable
   set contains (d_seg 2.966e-4, d_pose 2.331e-5, 191,052 B) = 0.1721 [contest-CUDA
   external, existence proof only, quarantined lineage].
3. **The two lines wall on the same layer.** DESCRIBE is blocked at finite-price
   MATERIALIZATION (0/162 per-cell prices lawfully null per ev2; 0/37 MS4D buckets);
   DESCENT at the PC1 tube-finish + unknown decay laws (G2 INDETERMINATE). Both are
   PRECONDITION-scope, paradigm intact [MEASURED, card §§2–5, §11, §19–23].

---

## §2 STRATUM A — NOW: verified eurekas + stale blocks (highest confidence)

### A1. The seven-home stream-level waterfill is ev2's OWN named lawful successor (a) — and it is not being run; PF3 is successor (b)
- MEASURED: ev2 (`codex_findings_ddm_ev2_per_pair_allocation_producer_20260725_codex.md`)
  proved 0/134,211 C1 bytes are pair×cell separable (100% UNALLOCATED) and sealed the
  coarsest lawful partition = **7 stream homes** (manifest 3,345 · v15 predictor
  100,099 · G1 worldsheet 29,878 · receiver profile 85 · solved template 151 ·
  ZIP CD/EOCD 383 · lane seed 270). Its closing line names TWO lawful successors:
  "(a) a coarser seven-home stream-level waterfill, or (b) a newly constructed C1
  object with independently coded per-pair/per-cell sections."
- The live pf3 arm charter = successor (b) (bind coordinate → receiver builder →
  realized uint8 quantum → new same-object prices). **Successor (a) has no owner.**
  And 1–2 rows of the 7-row price table already exist measured: cc3's whole-stream
  lossless delta (−3,422 B at zero distortion delta, integration overhead zero,
  135/135 parse-backs, FEED-603-cc3 merged 06845c4582) is exactly a stream-level
  price row; ev1's three dual-byte reconciliations (16 B / 962 B / 409,388,124 B)
  are same-family evidence.
- CONSEQUENCE (DERIVED): the describe line's "0/162 actionable" wall is partly an
  artifact of demanding prices at a granularity ev2 proved lawfully empty. A 7-row
  stream-level price table is measurable NOW on the existing C1 object with existing
  tooling ($0, local, no new coordinate families) and gives ms2r-class waterfills a
  finite domain — a parallel unblock that does not wait for PF3's new object.
- Label: MEASURED components, DERIVED composition. Named test: re-pose the ms4d/ms2r
  waterfill preflight at the 7-home schema; populate with whole-stream deltas
  (cc2/cc3 protocol) per home.

### A2. R2 (Laughlin histogram-matched per-dim quanta) is UNBLOCKED — the sy1 withhold is stale
- sy1 §14 (07-25) WITHHELD R2 at INSTANCE scope "pending real histogram custody,"
  citing rd1's findings ("per-dimension receiver uint8 absolute-step histogram
  EXPLICITLY RECORDED AS MISSING").
- VERIFIED THIS SESSION (primary artifact): the ev1 receipt
  (`ddm_ev1_campaign_evidence_joins_20260724T191623Z/ddm_ev1_campaign_evidence_join_receipt.json`,
  landed 07-24T19:16 — BEFORE sy1) contains
  `rd1_evidence/bucket_rows[*]/receiver_uint8_abs_step_histogram` (256-bin) plus a
  `histogram_coder`; its FEED states "EV1 owns … 162 receiver histograms."
- CONSEQUENCE: R2 — the quantization-gate row attacking the measured
  minimal-writes-die-at-uint8 wall (realization law, memory 2026-07-20) — can fire
  as a $0 derivation against ev1's custody. The quantization/realization gate is the
  named R1/R2 gap the round-3 council said the card lacked.
- Label: MEASURED (histograms exist; the withhold's cited blocker is gone).
  CAVEAT: sy1 may have judged ev1's histograms insufficient for R2's exact needs —
  first action is a one-paragraph adjudication, then the derivation.

### A3. The pose leg's open door: frame-0 carrier by DESCENT (not linearized solve)
- p1's negative (card §23) is FORMULATION-scoped to the **shared low-rank linearized
  solve** channel: best rank-1 d_pose 19.895 vs bar 5e-5; rank-6 WORSE than matched
  Rademacher control (= the uint8 trust-region crossing); the memo itself lists what
  the negative does NOT close: "nonlinear, pair-conditioned, higher-rank, or
  scorer-solved frame-0 quotient generators."
- The three composable measured pieces: (i) frame_0 is seg-free — OUR frozen-space
  fact (Δd_seg = 0 by construction for any frame-0 carrier; p1 verified frame-1 byte
  identity); (ii) pc2 PROVED descent machinery works and pays jointly (16/16 accepted,
  joint ΔS −0.2475, ratio 14.02 [macOS-CPU advisory exact n600]); (iii) PR130's
  ~23 KB neutral-gray low-rank frame_0 carrier reaches d_pose 2.33e-5-class
  [contest-CUDA external existence proof — frozen-space reachability fact ONLY;
  no design/constants transfer].
- EUREKA (CONJECTURE, existence-proof-backed): descend p1's own carrier
  parameterization (packet = 15+3505r bytes, builder already in-tree) through the
  real PoseNet with uint8-STE in-loop, from the W_seg-family parent — i.e. replace
  p1's one-shot linearized solve with pc2's accept-loop. Target: pose leg
  2.3e-5–5e-5-class at ≤23 KB → pose contribution 0.015–0.022 vs the composing
  triple's requirement (2.94e-5-class → 0.0172). This is the M4 knot-race member
  the intake memo called fork (b) "proven-feasible-in-frozen-space"; the gap is
  composition, not physics (never-weaker-state memory).
- Why it matters at tier level: pose is THE #1 crux of the T_1 composing triple
  (card §2); PC1-descent's measured slope (163.045→160.900 in 16 steps) needs a
  DERIVED 1,216 accepted steps at constant slope — an unmeasured decay law. A
  by-construction pose-legible carrier is the hedge that does not depend on that law.

### A4. Vehicle B's support-renewal inventory ALREADY EXISTS: the v8/v9 per-class carrier table — blocked only on byte-close
- MEASURED: db1 (card §11) proved the fixed atlas CANNOT close the box even as an
  all-beneficial oracle (short 192,020 px) — only live descent or richer description
  families can; Vehicle B's job is support renewal, not pixel correction.
- ORPHAN (memory sweep, `per_class_carriers_culminated_in_v8_v9…20260721`): the #503
  nine-dimension recursive-fractal carrier table + SPEC_v8 §2 per-class carriers
  (MyCar static mask 0.1–0.5 KB, IoU 0.994 · Lane analytic ground band 1–2 KB at
  lane-floor d_seg 0.00087-class · Road/Undrivable bulk-boundary field · Movable
  sparse islands) are polished, derived-optimal, built-default-OFF — and EVERY row
  carries `NO_VERDICT` on the byte measurement (`V9_INTEGRATION_BLOCKED_OWNER`).
  The single richest built-but-never-byte-closed pool in the program, and its job
  description is literally db1's shortfall (Lane = 53% of the d_seg enemy).
- Label: DERIVED (carrier numbers), MEASURED (the shortfall + the IoU/floor
  anchors). Named test: byte-close ONE carrier (MyCar static mask, the cheapest)
  through the proven e5a adapter → fresh n600 → the first Vehicle-B measured row.
- Fold: also recall the witness-era Lane cures as description-family primitives —
  paint-then-SDF measured FN nucleation 0.0058→0.0019 (3×), and the #268 through-R
  sR reachability weight (built, `gt_n600_sR.npz` staged, never fired) as E1/E3
  proposal-ranking signal. These were trainer levers; their transfer target now is
  Vehicle B / the campaign scheduler, not the dead witness trainer.

### A5. cc3's lossless stream transform: a guaranteed −0.00228 S on every future export
- MEASURED (FEED-603-cc3, merged): −3,422 B receiver-closed lossless
  (139,538→136,116 B), scorer-surface byte-identical outputs, zero distortion delta,
  inflate 489.7 s < 30-min budget, rate costate −0.0022785693. Inheritable by the
  R6/e-chain; other stream shapes must re-measure. Small, certain, compounding.

### A6. Wall-clock is a tier lever: the campaign's 33.66 h ETA can plausibly become ~1 order cheaper
- MEASURED: 302.9 s/step (costate digest); E3-SPRT ladder targets the ~437 s/proposal
  exact tax ~10–20× (card §8, producer g3's subset→full validity r) [DERIVED].
- Never-fired compute rows: D4 micro-batch (bit-identity impossible, but the 1.56×
  gate rests on a possibly unnecessary bit-identity REQUIREMENT — the named bounded
  n600 d_seg A/B was never run) · deferral D48–D51 (YOPO cadence, native projected
  execution, megakernel, micro-batch) · task #494 max-throughput authority ladder
  (pending). Rows/day is the meta-metric that multiplies every other stratum.

### A7. Honest exclusion enforced (adversarial pass on my own sweep's #1)
- The research sweep returned "m7 relaxed receiver S=0.18965 advisory, byte-closed,
  one Modal dispatch from an exact row" as its top item. I traced lineage in primary
  artifacts: `joint_optimum_575_xhigh_DAG_FEED_20260720.md` shows the archive is the
  **PR110-lineage compact archive** (b4689726… → cb6cf0ba…, 177,169 B; the same
  object as m6's FP11/CTXR anatomy incl. the PR101-style 607-B sidecar). That is the
  operator-banned borrowed-incumbent surface (quarantine-enforced). **REJECTED — not
  a eureka, and the routing-card §4(c) "verify before proposing" caution resolves as:
  fork option (c) relaxed-path-direct is NOT ours; it should be struck.**

---

## §3 STRATUM B — reactivations whose cures LANDED (scoped negatives, cure status verified where possible)

| negative (scope) | named cure | cure status |
|---|---|---|
| attempt-5 "phantom regression" + j9/j10 stop (INSTANCE, apparatus) | derived-EMA + dual live/EMA verdicts + degeneracy guard | **LANDED + exonerated** (card §19: live state was descending the whole time) |
| ev2 per-cell waterfill misposed (FORMULATION) | seven-home stream waterfill (successor a) | **LANDED as law; unowned as a run** → A1 |
| R2 quanta withheld (INSTANCE, producer citation) | ev1 162 receiver histograms | **LANDED** (verified this session) → A2 |
| p1 frame-0 pose carrier (FORMULATION: linearized shared-basis) | nonlinear/scorer-solved descent of same carrier | machinery LANDED (pc2 loop); arm unowned → A3 |
| at1 λ-ranking "BACKTESTED-FAIL" (INSTANCE) | at1 8-pair replay measured ρ 0.9027 / NDCG@4 0.9269 positive; blocked `LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED` (7-version drift) | cure named: locked-source materializer + n600 gaze Jacobian — buildable |
| cc1 Q8/C3/v5 continuation + hinge routes (#91–96: d_seg 0.0084 @3,082 B still descending; #98–99 hinge ~29% @1,029 B) | J8F descriptor-to-counted-application operator | **NOT landed** — the single shared blocker for 4 measured-composed routes → C1 below |
| hosc/gauss activation death (INSTANCE: n24 + wrong-lr) | β-anneal (built FEED-fe) / FINER++ init | built-never-fired; witness-era — fold only if a trainer-class vehicle returns |
| Lever-D flicker coder >0.65 B/flip (FORMULATION, ckpt-conditioned) | re-run on coherent (islands-active) ckpt | conditional; dormant until such a ckpt exists |
| M2 tie-aware preimage (+0.047 S expected) | — | **anti-cure: MEASURED NO-OP** (fp32-exact 0/117.96M); retire citations (v10 memo's "M2 widens box" is stale) |

---

## §4 STRATUM C — missing apparatus / instrumentation (buildable; each unlocks a family)

1. **J8F descriptor→counted-application operator** — four measured-composed cc1 routes
   are ROUTED-PENDING on this one operator. Highest fan-in single build on the ledger.
2. **at1 locked-source factor materializer** (+ n600 gaze Jacobian) — converts a
   measured ρ 0.90 λ-ranking instrument from blocked to live; feeds E1/E3 scheduling.
3. **Predictor-stream structural replacement** — v15 predictor zip = 100,099 B =
   75% of the C1 describe object (ev2 lineage table). The la1 follow-on is queued
   post-refoundation; it is the single largest rate home on the original describe line.
   [MEASURED home size; replacement value OPEN.]
4. **The #669c type×layer re-homing ledger** (SKELETON/CONNECTION/FIBER/GAUGE/RESIDUAL
   × L1–L5) — queued, never run. It is the instrument that converts "RESIDUAL" from
   unclassified remainder into priced homes; is1 declared inventing the telescoping
   allocation without it a Directive-3 violation. Without it, all inherited exchange
   rates stay [upper-bound, proposal-search channel] (economics law 07-24).
5. **The seg SECANT curve** (bytes vs d_seg through the real coder) — the unfilled
   axis of every waterfill; arm existed (`seg_secant_rd_curve_20260719`), curve unfinished.
6. **Pose decay-law instrument** — P2's ten-interval trace schema extended to pose
   rows (card §15 names this); until it lands, every pose-horizon number is
   constant-slope DERIVED.

---

## §5 STRATUM D — paradigm-level reformulations + theory questions (CONJECTURE / OPEN-QUESTION; not gated on immediacy)

1. **Solve IN description coordinates (realization-aware solve).** Today the program
   solves in RGB space (409 MB object) then tries to describe it — and v14 measured
   the paint loss (low-error mask → high-error after fixed RGB projection). The dual
   failure: descent explores realizable-but-weak moves; solve finds strong-but-
   unrealizable ones. CONJECTURE: run the Gauss-Newton/CG solve machinery (rank-4
   SegNet head law + ≤6-dim Pose quadratic, both custodied) DIRECTLY over the
   parameters of a compact generator (so every iterate is by-construction ~130 KB and
   receiver-closed), with uint8-STE inside the solve. E8 (solved-plane distillation)
   is the SGD shadow of this; is1 path (d) "score-quotient functional, #366/J5 may be
   the fitting engine" is its named home — ranked prospective-#1 there and never built.
   This dissolves describe-then-realize into one optimization and is, I believe, the
   single most promising unbuilt vehicle in the record.
2. **The description language against the 114 KB benchmark.** PR130's generic
   masked-conv AR prior prices the exact 5-class partition stream at ~114 KB
   [external MEASURED, harvest-signal]. Our describe family's entire reason to exist
   is the scene structure a generic prior ignores: BEV staticity, ξ-advection,
   worldsheet events, per-class carriers (A4). OPEN-QUESTION with a now-concrete
   number: what is OUR bpp on the solved partition vs generic-AR? (Their measured
   warning transfers as a lesson: raw label-diff temporal factorization was 3.5×
   WORSE — temporal structure must ride geometry.)
3. **Box-tolerance economics.** The solve has 7.6× error headroom inside the box
   (136,839 allowed vs 17,931 exact) [MEASURED, card §4]. Almost all prior describe
   pricing targeted the exact solved point. The R(D) object that matters is the
   CHEAPEST box member, not the solved point; rd1's λ-continuation should be re-read
   as "which tolerance to sell for which bytes" with the E7 dual-currency rule.
4. **The true seg-debt currency: H(flip-field | free decoder context).** The 405.5
   B/error greedy price is a channel upper bound (economics law). E7/R5/R6 name the
   real object: context-MIXED entropy + syndrome-coded transmit realization ($0 race
   on the v19c 104-admission set, preregistered, unrun). If H is several× below the
   greedy price, byte-payable seg debt expands proportionally — this is a tier-moving
   scalar nobody has measured. OPEN-QUESTION.
5. **The rate-only endgame theorem.** PR86 measured seg 0.00 / pose <1e-9 at
   unconstrained rate [external]; our exact solve is the same fact at 409 MB. So
   S_min = 25·K/37,545,489 where K = the Kolmogorov-style cost of a scorer-equivalent
   witness under the free-interpreter rule (rule 118: generic code free, video-derived
   bytes counted). The 10-year program IS estimating and approaching K. Current
   brackets: MDL(MS) upper bound ~236 KB (contour+Brotli, tightenable via power
   diagrams #539) vs sub-0.15-by-rate line 225,272 B vs the 154.5 KB strict line vs
   S_floor 0.11797 ⇒ K ≈ 177 KB-class if the floor is tight. Closing that bracket —
   from both sides — is the cleanest formal statement of the whole campaign. [DERIVED
   frame; brackets MEASURED at their cited channels.]
6. **G2's fork is the program's biggest branch point.** If the live decay law (P2
   trace) shows descent cannot reach 8.7e-4-class in-horizon, ALL seg weight shifts
   to Vehicle B + dual-currency residual, and the engine becomes a finisher. Budget
   allocation across A-vs-B should be pre-planned for both outcomes rather than
   re-litigated on landing. OPEN-QUESTION (closes on the j11/P2 receipt).

---

## §6 STRATUM E — long-horizon arcs (10-year program; CLAUDE.md's long-bet clause invoked deliberately)

1. **Compile-the-generator end-state.** The doctrine already in CLAUDE.md, now with
   measured supports: inflate.py carries the entire deterministic scene model (BEV
   static world + lane geometry + worldsheet event grammar + pose-legible frame-0
   constructor) for FREE; archive.zip carries only the ~8-dim ξ trajectory (AR-coded,
   hundreds of bytes), sparse event tokens, and the irreducible residual. Every
   stratum above is a station on this arc; the K-bracket (D5) is its scoreboard.
2. **Amortized scorer inversion.** We now hold analytic custody of both scorers
   (rank-4 head, pose quadratic, margin-Fisher atlases). The long bet: a trained
   amortized INVERSE (target margins → RGB) as a reusable decoder head, making
   description families directly decodable and killing the paint loss class once.
   (#211 per-video amortized-init lineage; is1 family-(d) is its first customer.)
3. **Throughput as compounding capital.** Exact rows/day is the rate limit on the
   whole research loop; megakernel + micro-batch + SPRT + prefilter compose to
   ~1-order more measurements per week at the same honesty bar. Treat as
   infrastructure with a named consumer (the campaign), per THE GOAL's rule 3.

## §7 Pool/conflict notes
- Pose legs COMPETE (one pool): PC1-descent (live, j12) vs frame-0-descent (A3) vs
  R3 precision ladder — the M4 knot interface race decides on bytes; do not sum.
- Seg debt: steps (Vehicle A) vs bytes (Vehicle B residual) — E7 D* arbitrates; A4's
  carriers and D4's H(flips|context) are the SAME pool as the 405.5 B/error greedy
  channel (never add their savings).
- Rate: A1/A5/C3 act on the same describe-line object; stream-level prices make them
  commensurable — measure jointly, not additively.

## §8 Honest residual (what I could not verify this session)
- Whether sy1's R2 withhold considered-and-rejected ev1's histograms (I found no
  text doing so; adjudication owed before firing R2).
- The v8/v9 carrier table's byte-close cost through the e5a adapter (numbers are
  design-derived; NO_VERDICT stands until a packet lands).
- The PC1 −2.76 S rehome (j12 §3) vs pc2's −0.2475: both receipt-cited in the routing
  card but the j12 worktree receipt itself was not re-opened here.
- Task-ledger completeness: the 84-row canonical ledger is superseded for witness
  tasks; a full live-TaskList sweep was not possible from this seat.
- Nothing here is a score claim; the pointer moves only via a byte-closed
  `upstream/evaluate.py` n600 row.

STORES CONSULTED: routing card `council_coherent_optimal_path_routing_20260725.md` §1–§6 ·
`optimal_start_card_366_refoundation_20260725.md` §1–§23 ·
`negative_findings_register_20260709/auditor_A_dag_research.md` (51 findings + TOP-10) ·
`negative_cure_join_table_20260710.md` · `ddm_is1_full_inverse_solve_to_the_end_DAG_FEED_20260724.md` ·
`codex_findings_ddm_m6_close_22645_byte_gap_20260723_codex.md` + receipt ·
`codex_session_summary_20260723_ddm_m7_relaxed_receiver_codex.md` ·
`joint_optimum_575_xhigh_DAG_FEED_20260720.md` (lineage trace) ·
`codex_findings_ddm_ev2_per_pair_allocation_producer_20260725_codex.md` ·
`ddm_ev1_campaign_evidence_joins_DAG_FEED_20260724T191623Z.md` + receipt (histogram custody
verified) · `public_pr129_132_intake_20260725.md` · rd1 receipt inventory · costate digest
(SessionStart SENSE) · canonical-equations count · lever-registry completeness · task ledger ·
memory: MEMORY.md + established-findings cluster + 19 deep-read memories via sweep agent ·
two sweep-agent reports (research-dir + memory-dir).
