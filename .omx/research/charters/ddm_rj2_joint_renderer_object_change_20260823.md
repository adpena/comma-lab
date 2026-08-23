# ddm_rj2_joint_renderer_object_change — build the governed joint renderer solve on the exact DX2 object (in-compile compensation + carrier re-solve), and report which closed legs the move RE-PRICES (owning memo: ddm_rj1_renderer_joint_move_20260823.md)

## MANDATE

Routed finding, `ddm_rj1_renderer_joint_move_20260823.md` (commit `f0bf2651cf`, memo sha
`88814e9f06b42cec…`), NEXT_IF_RESUMED fire-order row 1 verbatim: *"QUEUED-WITH-A-FIRE-ORDER —
mechanism completion. Owner: a MAIN-designated RJ1 successor. … Fire trigger: JF1's live reference
fits are terminal and harvested, the successor has a governed resumable/checkpointed renderer
trainer, and storage preflight passes. Action: jointly optimize each retained representation on the
exact DX2 object, solve compensation in-compile against each final packet/render pair, re-solve and
re-encode the carrier, retain every primary/repeat archive, and fold any row that fails a required
mechanism gate before scoring."*

Storage preflight NOW PASSES (APDataStore 208 GiB free as of 2026-08-23, after an
operator-authorized certified reclaim). JF1 is still live and holds the Metal slot, so **this arm
builds and smokes on CPU only** — it does not take the slot and does not fire a scorer.

`ddm_mf1_manufactured_seg_repair_20260823.md` (commit `b0c2869ce4`, memo sha `4ec2d9b3799e1dbe…`)
independently names this same arm as its consumer: *"fold MF1's boundary/margin map into the
already-queued RJ1 exact-object joint renderer solve, not into a shipped mask."* Two arms converged
on one object; this charter is that object.

**WHY THIS AND NOT A CARRIER OR A CONTAINER.** Three independent contest-grade measurements now say
the same thing — **PoseNet scores the FRAMES, so the RENDERER is the pose carrier**: RJ1 refused
renderer re-representation 3.51× with d_pose **97.70%** of the loss (`ddm_rj1_renderer_joint_move_20260823.md`);
`ddm_mf1_manufactured_seg_repair_20260823.md` measured its zero-byte boundary pull at ΔS
**+0.77834455** with pose **95.37%** of the loss — a MEASURED PERTURBATION DELTA, not a frontier
score; and MF1's oracle mask pull at α=0.25 genuinely improved seg (net 26 fewer errors,
d_seg −4.1325887e-6) while d_pose went 5.4316097e-6 → 7.2015297e-5 (**13.3×**, +0.01946572 S).
No post-hoc render edit can buy seg.
And the container route is byte-dead on both horns: ET1's exact midpoint-BSP 535,761 B, HG1's
heterogeneous analytic 460,408 B (commit `1eb31298ec`, memo sha `1ea85c9d0f0be6cb…`) — whose own
table shows the *generators* cost only 47,667 B while the unique-home residual costs 359,280 B.
**The learned DX2 renderer stores the same partition, end to end, in 180,368 B — 2.55× better than
the best measured analytic container.** The learned family is the right family. The only open
question inside it is whether a jointly-solved renderer object beats the converged one.

## SCOPE

1. **Build the governed joint solver** on the exact DX2 object (archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B; categorical field
   sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`). Required mechanism
   properties, none droppable: (a) joint optimization of the moved renderer against **both** frozen
   scorers, (b) compensation solved **IN-COMPILE against each final packet/render pair** — never
   carried from another object, (c) carrier **re-solved and re-encoded** after the move, (d) real
   coders on real payloads, (e) receiver closure with parse-back proof, (f) primary + repeat archive
   retained with matching hashes.
2. **Resumability is P0** (CLAUDE.md, operator binding): resumable-from-disk (`--resume-from`), a
   complete byte-close-loadable checkpoint at the END OF EACH STAGE under a distinct
   stage-encoded filename (never overwrite the prior stage), periodic intra-stage saves, EMA shadow
   saved (not live weights), atomic tmp+rename. Loop-end-only saving is FORBIDDEN. A trainer that
   cannot resume is not a deliverable.
3. **Consume MF1's boundary/margin map as a training-time input**, per MF1's own routing — as a
   weighting/conditioning signal inside the joint objective, NOT as a shipped mask and NOT as a
   post-hoc edit (that formulation is measured dead, above).
4. **Bounded CPU smoke on the cheapest rung only.** RJ1's own ranking of its retained
   representations: single-FiLM W96 is the lowest-distortion-risk move (preserves full width and
   full pointwise spatial mixing; 1,078 B cut). Smoke that one. Declare the smoke's SCOPE reduction
   explicitly (pair count, step count, precision) — SCOPE reductions are legal, MECHANISM reductions
   are not.
5. **THE OBJECT-CHANGE RE-PRICING TABLE — the deliverable that makes this worth a Metal slot.**
   `ddm_sy2_composition_synergy_deep_pass_20260823.md` (commit `fe2ba12dc2`, memo sha
   `32fc8fcc206bf76c…`) established: *a closed leg survives only when another leg first CHANGES THE
   OBJECT it was priced on.* For each moved renderer object this arm produces, emit a typed row per
   previously-banked-or-closed leg — at minimum QS2 (−4.375e-6, +34 B), RE1 (−1.207e-6, 0 B), EC1,
   LD1, AE1, OE1, and the HPAC sharp-optimum rows — stating whether the move changes the object that
   leg was priced on, and if so what specifically changed (field? model? both?). A leg whose object
   is unchanged stays closed and must be reported as such. **Do not re-open a leg by assertion.**
6. **Report in BOTH currencies** (`ddm_tl1_teacher_ledger_20260822.md`): ≤137,986 B at current
   distortion (shed 42,382 B), OR shed 150 B at zero distortion. Exchange rate **6.658590e-07 S/B**
   — CITE `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (commit `19522460a5`, memo sha
   `4bf730e5e5d3958f…`), do NOT re-derive. Decompose against AR1B's exact zero-remainder residue
   (`ddm_ar1b_archive_residue_purchase_20260822.md`, memo sha `388185a6c283359e…` — UNCOMMITTED at
   charter time, pin by content sha and say so): renderer 30,856 · carrier 22,010 · HPAC model
   13,515 · compact residual 96 · framing 114 · token stream 113,777.
7. **No scorer claim from the arm.** Every distortion number is either MEASURED on the stated axis
   with its label, or explicitly **UNMEASURED**. Recompute S from its three components
   (rate 25·B/37545489 + seg 100·d_seg + pose √(10·d_pose)); the rounded `final_score` printed by
   `upstream/evaluate.py` is a DISPLAY value and is never the citable quantity — see the
   rounded-display rule recorded in `ddm_tx1_toolbox_crosswalk_20260819.md` lineage and enforced in
   live code.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- **NO Metal slot.** JF1 holds it. CPU build + bounded CPU smoke only. If the arm concludes the
  smoke needs Metal, it STOPS and emits a sealed MAIN-owned fire-order instead of taking the slot.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000); bulky receipts to
  `/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/`. **Vertigo is FULL — do not
  write there.** sha256 + bytes on every persisted payload; a scalar-only artifact when bytes exist
  in memory is forbidden at the typing moment.
- File-ownership: JF1's receipts under `.omx/tmp/arm_receipts_local/ddm_jf1_*` are SACRED.
  RJ1's retained tree is READ-ONLY custody — reuse its harnesses, do not mutate them. Note RJ1's
  consumer store is named on Vertigo (full); re-home to APDataStore and record the re-homing.
- Every negative-existence claim states its SEARCH SCOPE or is not made (m53).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rj1_renderer_joint_move_20260823.md` — renderer re-representation REFUSED **3.51×**, d_pose
  97.70% of it. Its exact WD4 warm-lineage salience-pruned dense W64 instance is CLOSED
  (`d_seg=0.03182023`, `d_pose=13.43292999`, S≈14.8829) — do not rerun it. SVD-r32 is dead
  unconditionally. Its three current archives are NOT score candidates.
- `ddm_qs4` — carrying a compensation solved for ANOTHER object cost **+2.396e-4** pose damage.
  `ddm_qs5` — in-compile compensation on the final object measured d_pose BELOW base. Constraint (b)
  in SCOPE 1 exists because of these two rows, not as style.
- `ddm_mf1_manufactured_seg_repair_20260823.md` — post-hoc render edits (boundary pull AND oracle
  mask pull) are FORMULATION-CLOSED. New addressing is rate-dominated: MF1's address mask measured
  35,969 B Brotli-q11 vs the 21,537.2 B perfect-repair ceiling (**1.670× larger**).
- `ddm_et1` (535,761 B) + `ddm_hg1` (460,408 B) — the container route, byte-dead on both horns at
  FORMULATION scope. `ddm_nr1` (349×) + `ddm_ni1` (247.69×) — whole-body lossy, CLOSED on two
  authority rows. `ddm_ld1` — every lossy Lane rung makes the archive BIGGER.
- **THE SHARP-OPTIMUM LAW** (`ddm_oe1_*`, `ddm_ld1_*`, `ddm_ae1_*`, `ddm_ni1_*`, `ddm_wj1_*` — five
  concordant arms): the HPAC model and the field are jointly at a local optimum, SHARP in every
  measured direction. This charter's entire premise is that a joint solve is the one move that is
  not a perturbation of that optimum — it changes the object. If the smoke shows it behaving like a
  perturbation, say so in those words.
- `ddm_na12_post_sy2_negative_regrade_20260823.md` (commit `b6e60fa7e6`, memo sha
  `4e2b7fe543c9a358…`) — W96 is REOPENED but **tie-only**: winning it needs 55,188 B beyond the
  displayed construction. **Do not charter W96 as a byte-shrink route.** Its 1,078 B cut buys
  7.18e-4 S of distortion headroom = 2.5% of the 0.028220 gap AT BEST, and only if distortion-neutral.
  It is fired here for a BETTER BASIS (SCOPE 5), never for shrink.
- `ddm_jf1` (LIVE) — its mandatory positive control FAILED by 7,554 B. Any JF1 figure quoted in this
  arm inherits that caveat EXPLICITLY, on the same line as the number.
- Amplification exponent **~16.7**: token/parameter/field agreement is a near-useless d_seg
  predictor. Never interpolate distortion between rungs; measure it or mark it UNMEASURED.

## OPTIMAL FORM

- **Family exemplar / reference form:** the RJ1 joint-move representation set at
  `.omx/research/ddm_rj1_renderer_joint_move_20260823.md`, commit `f0bf2651cf`, memo sha256 prefix
  `88814e9f06b42cec` — this is the **reference** the mechanism must match. Mechanism properties that
  may NOT be dropped: joint optimization against both frozen scorers · in-compile exact-object
  compensation · carrier re-solve + re-encode · real coders · receiver closure with parse-back ·
  primary+repeat retention. Dropping any one is a MECHANISM reduction and requires an explicit
  TOY-BRACKET declaration — after which the row CANNOT produce a family verdict (the NY1 lens: MAIN
  closed a family from mechanism-reduced rows this week; do not repeat it).
- SCOPE reductions LEGAL and declared per row: pair subset, step budget, a single representation
  rather than all three, bounding rather than exactly solving the carrier fit. Each row states what
  it omits.
- **PRIOR-LAW PREDICTION (falsifiable, deliberately not optimistic).** The sharp-optimum law (five
  concordant arms) plus the three-measurement pose law plus NA12's W96 arithmetic together predict:
  joint optimization with in-compile compensation will recover the MAJORITY of RJ1's 3.51× refusal —
  because QS5 proved in-compile compensation reverses stale-object pose harm — but will **NOT reach
  a negative joint ΔS on any single rung**, because the renderer carries pose and the byte cut on
  offer (1,078 B ≈ 7.18e-4 S) is 2.5% of the 0.028220 gap. I predict **REFUTED for a pointer move,
  CONFIRMED for mechanism recovery.** FALSIFIER: a retained, receiver-closed archive from the
  bounded smoke whose joint ΔS, recomputed from components, is **negative**. If that lands it is the
  campaign's first live sub-0.12-direction route from the current object and MAIN seals the scorer
  row. If it does not land, the decision-relevant output is SCOPE 5's re-pricing table: state
  plainly whether the joint move changes the object enough to re-open ANY closed leg, because if it
  does not, then the renderer family is converged too and the honest campaign fact is that no
  surviving route to sub-0.12 exists from the current object.

## DELIVERABLE

`.omx/research/ddm_rj2_joint_renderer_object_change_20260823.md` — typed rows: the solver's
mechanism-gate table {joint-opt · in-compile compensation · carrier re-solve · real coder · receiver
closure · primary/repeat hashes} each {PASS | FAIL | NOT-REACHED} with its receipt path; the bounded
smoke's rows with SCOPE reductions declared and every distortion column MEASURED-or-UNMEASURED; the
SCOPE 5 object-change re-pricing table, one row per closed/banked leg, {RE-PRICED | UNCHANGED} with
what changed; totals vs 137,986 B in BOTH currencies; the prior-law prediction adjudicated
CONFIRMED/REFUTED per clause with its number; `verdict_scope` on the narrowest rung the evidence
supports; a `STORES CONSULTED:` line (the contract's literal key) naming what was loaded, honestly
including "none" where none; `NEXT_IF_RESUMED` with an owner and a fire trigger per row. Commit via
the serializer. End with the own-vehicle frontier line.
