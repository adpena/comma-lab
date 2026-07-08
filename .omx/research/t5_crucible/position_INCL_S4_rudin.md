# POSITION — SEAT S4 (Rudin, evidence-class audit lens) — T3 v7 INCLUSION SYMPOSIUM

BLIND. Did not read sibling position_INCL_S* files. Advisory means only — **pointer contest-CPU
0.19110 UNMOVED; nothing below moves it. Only a byte-closed n600 evaluate.py row does.**

**Operating-within assumption (#363, stated at top):** I audit whether each item's DOCKET-claimed
evidence CLASS matches the ACTUAL evidence artifact (test/measurement/anchor), and independently
re-derive the class. My core assumption: **bit-identity is a CODE property (provable at any scale
where the code path is exercised); a d_seg/d_pose/rate SCORE verdict is an EVIDENCE claim that the
n600-or-not rule binds.** The inclusion boundary lives exactly on that line. Classification (this
assumption): VERIFIED_VIA_SOURCE_INSPECTION (CLAUDE.md NO-FAKE #8 surrogate-vs-authority +
n600-or-not-evidence + the #330/#348 bit-identity artifacts read directly).

Evidence-status taxonomy per #363: VERIFIED_VIA_SOURCE_INSPECTION · VERIFIED_VIA_EMPIRICAL_ANCHOR ·
INFERRED_FROM_DOMAIN_LITERATURE · ASSUMED_AWAITING_VERIFICATION.

Classes ∈ {IN-v7 · v7.1-ARM · REGISTERED-duty-to-measure}. My anti-laundering bar: **a lever with
no anchor cannot be IN-v7; a "bit-identical" claim must name a real test; a "measured speedup" must
cite a real bench.**

---

## ITEMS 1–2 — OPERATOR-DECIDED (I audit only citation accuracy)

**Item 1 (Basis rebalance — IN-v7):** citation ACCURATE. `793631e00` exists and is
`basis-integration v7: DirectionalBasisRebalance(lane_offloaded) IN-v7 as 4th DSL lever`
(VERIFIED_VIA_SOURCE_INSPECTION, git log). Waterfill peak 71.54 GiB claim consistent with the
ledger's mem-preflight rows (67.61 GiB base + the basis's +3.93 GiB). No audit objection.

**Item 2 (Event mode — new baseline):** citation ACCURATE. Operator verbatim IS in the ledger
(ORCHESTRATION_LEDGER lines 268–279, "we want to transition to event based now and accept the
risk … a new baseline, not clean"). Recorded as risk-accepted, attribution-via-per-stage-ckpts.
VERIFIED_VIA_SOURCE_INSPECTION. No audit objection.

---

## ITEM 3 — Micro-batch-pairs → **v7.1-ARM** (matches docket)

- Evidence: routing LANDED `bd6219a0a` (VERIFIED_VIA_SOURCE_INSPECTION). Equivalence derivation in
  the commit: logit-adjust offset `o` is **bit-EXACT per pair** (per-class constant broadcast);
  live-τ L_τ bit-exact vs serial. **But: "only the existing mean-over-B is a reorder."**
- AUDIT: the docket's "logit-adjust bit-exact" is a NARROW-TRUE sub-claim (the offset). The lever
  **as a whole is NOT bit-identical** — the mean-over-B fp reorder is trajectory-affecting. Correct
  gate = n600 trajectory A/B. The docket does not over-claim; class v7.1-ARM is right.
- No discrepancy. (Note the "waterfill B pinned 1 UNMEASURED" caveat is honestly carried.)

## ITEM 4 — Safe-compile certified regions → **v7.1-ARM** (matches docket; but the docket STAT is misleading)

- **AUDIT QUESTION (mine): is the hosc `_act` flag-flip admissible at v7 LAUNCH under the per-chip
  fingerprint gate, or only after stop-time GPU re-cert? ANSWER: only after stop-time GPU re-cert.**
- Evidence (mlx_safe_compile_v2_finish_20260708.md, VERIFIED_VIA_SOURCE_INSPECTION):
  1. **The flag-flip lever IS `hosc_activation` — "the exact activation run-1 uses" (memo §90.2).**
  2. On **CPU: hosc FAILS bit-equality, max|Δ|=5.96e-8 (1 ULP)** → auto-routed to a kernel candidate.
     So the compiled hosc is affirmatively NON-bit-identical on CPU.
  3. On **GPU: hosc certified bit-equal only at n=32** — and the memo itself calls this
     "coverage-limited (a finite input sample, NOT a proof) … GPU hosc warrants BROADER/adversarial
     coverage before the score-bearing run trusts it." (= surrogate, not authority — NO-FAKE #8.)
  4. The **canonical GPU manifest is device=gpu, fingerprint-ABSENT (legacy-unscoped)**; the b2
     launcher gate `resolve_enabled_regions(enforce_fingerprint,…)` REFUSES rc=4 an
     absent/stale-fingerprint manifest before spawn. GPU cross-process re-cert WITH fingerprint is a
     **v3 residual, governor-deferred to run-1 stop.**
- ⇒ Admissibility: launch-flip is inadmissible on TWO independent grounds — the fingerprint gate
  refuses the fingerprint-absent GPU manifest, AND the only GPU cert (n=32) is coverage-limited and
  CPU's clean-pass-then-fail-on-unsampled-inputs is the live proof that n=32 can lie. Gate = the
  run-1 stop-time GPU re-cert (fingerprinted) + broader coverage + whole-step B=8 bench (D17).
- **DISCREPANCY D-2 (LOUD):** the DOCKET STATUS phrase **"8/9 bit-eq=0 CPU"** is a misleading
  aggregate for the inclusion question — **the 1/9 that FAILS is hosc, the exact flag-flip lever.**
  Reading "8/9 bit-eq" as evidence for the hosc flip laundries the one region that decides it.
  Class unchanged (v7.1-ARM) but the evidence STAT must be restated as "hosc = the failing region;
  not launch-admissible pre-GPU-re-cert."

## ITEM 5 — D16 Metal kernels → **v7.1-ARM** (I DISPUTE the docket's stated REASON)

- Evidence REAL (VERIFIED_VIA_EMPIRICAL_ANCHOR): `68ed00ba2`, `test_metal_persistence_pool.py`
  (139 lines) — bit-identity **max|Δ|=0 MEASURED vs the numpy authority `_pool3x3_np`** on real
  shapes (M∈{1,2,4,8}, 384×512, borders); **N=5 cross-process bit-identical** via
  `mlx_gpu_determinism_probe.py::persistence_pool`; speedup **1.9–4× measured**; default-OFF; the
  kernel is forward-only stop-grad (no VJP path → score-neutral by construction). All claims trace
  to real artifacts. No laundering.
- **DISCREPANCY D-1 (LOUD):** the docket says *"the consuming loss term is default-off in v7."*
  **This is FALSE.** run-1/crucible_v6 launch.sh carries `--persistence-loss-weight 1.0`,
  `--persistence-recall-weight 1.0`, `--persistence-warmup-epochs 275`, `--cache-gt-skeleton`,
  `--length-weight 0.001` — the persistence-topology consumer is **ACTIVE at weight 1.0** (and v7 is
  "all other flags per-flag byte-identical" vs v6 per the v7 authoring memo → carries these ON).
  What is default-off is the **kernel-DISPATCH flag**, not the loss term. So D16 is a STRONGER
  IN-v7-adjacent candidate than the docket credits: a bit-identical, deterministic, measured
  forward-only speedup on an **active** term.
- Class: **v7.1-ARM** — but the gate is the **whole-step B=8 bench** (the "finished speedup number"
  is an explicit v3 residual; the 1.9–4× is the isolated-kernel figure, not in-loop), NOT
  "REGISTERED because the consumer is off." Correct class, wrong docket rationale.

## ITEM 6 — #330 verdict memory reclaim → **v7.1-ARM** (docket over-reaches with "Candidate for IN-v7")

- **AUDIT QUESTION (mine): is gt_n24 bit-identity sufficient for IN-v7 on a score-neutral-by-
  construction claim, or does n600-scale bind? WHERE is the boundary?**
- Evidence (`4ba4058e1`, `tools/measure_verdict_memory_reclaim.py`, VERIFIED_VIA_SOURCE_INSPECTION):
  bit-identity measured at **real gt_n24, vbatch 8, macOS M5 Max** — subprocess d_seg/d_pose ==
  in-process, **per-pair array_equal AND mean bit-equal**; the child re-uses the SAME
  `cpu_verdict_*` primitives + SAME `--verdict-batch`.
- **THE BOUNDARY (precise):** bit-identity here is a **CODE property** — "does relocating the verdict
  computation into a killpg subprocess perturb its output?" The answer is structurally NO (identical
  inputs, identical code, identical batch chunking). That is scale-free once the code path is
  exercised; the n600-or-not rule binds SCORE/d_seg **verdicts that inform decisions**, not a
  code-transform-equivalence demonstration. **So gt_n24 IS sufficient for the score-neutrality
  claim.** The n600 rule does NOT bind the bit-identity leg.
- **BUT IN-v7-ON is inadmissible on a DIFFERENT axis (operational, not score):** the memo's OWN
  recommendation is **"keep DEFAULT OFF until a governed n600 run confirms the disk-hop"** — the
  ~7 GiB transient npz PER n600 verdict (SSD serialization) is a **new operational behavior
  UNMEASURED at n600** (the n24 npz is trivially small, so n24 never exercised the disk-hop / IO
  cost / memory-pressure interaction beside a live run). And the docket's "the memory it frees funds
  the basis raise" rationale is **moot** — the basis raise was already waterfill-ADMITTED at
  71.54 GiB WITHOUT #330's reclaim.
- ⇒ v7 launch state = flag OFF = byte-identical (reclaim inactive). Class **v7.1-ARM** (gate =
  governed n600 disk-hop confirm) — or REGISTERED-with-trigger (activate on n600 RSS pressure). The
  docket's "Candidate for IN-v7" over-reaches vs the artifact's own default-OFF-until-n600 line.
  **DISCREPANCY D-3 (MEDIUM).**

## ITEM 7 — Adaptive-ε → **REGISTERED-duty-to-measure** (matches docket; INDEPENDENTLY VERIFIED)

- **My independent verification of "never-fired-at-n600":**
  1. run-1 launch.sh = 107 flags, **ZERO `adaptive-*` token** (grep clean; it carries
     `--eikonal-weight 0.01` + `--lane-band-eps 0.35`, NOT `--eikonal-viscosity-adaptive`).
     VERIFIED_VIA_SOURCE_INSPECTION.
  2. Trainer default `adaptive=False` (train_levelset…:3932). Built + flag `--eikonal-viscosity-
     adaptive` present (:9460).
  3. **Equation status:** `adaptive_eps_cfl_edge_tracking_v1` REGISTERED; byte-identity+parity anchor
     VERIFIED at gt_n6 (advisory/[macOS-MLX] NON-PROMOTABLE); **n600 cure =
     ASSUMED_AWAITING_VERIFICATION**; the `8` FORMALIZATION_PENDING (DAG FEED-06c, verbatim).
  → n600 A/B NEVER RAN, confirmed on three independent surfaces. Class REGISTERED is correct.
- ADD (not a discrepancy — a duty-to-measure sharpener): the memo's HONEST CAVEAT is that at the
  launch η/λ, ε **clamps at FLOOR 0.3 for >90% of epochs** (interior |c_a|~10 ≪ the ~80 needed to
  rise) — i.e. adaptive-ε is likely **INERT** unless sharpness explodes. The built
  `adaptive_eps_INERT` confound alarm already covers this. The duty-to-measure trigger ("eikonal
  re-entry signature") should ALSO watch for INERT-when-fired, else a null A/B mis-reads as "no
  effect" when the real cause is the clamp. (req-T anchor: the 0.3/0.7 clamps bind >90% of epochs.)

## ITEM 8 — R-7 finishers → **v7.1-ARM** (the three residuals do NOT block IN-v7; they bind the A/B gate)

- Evidence (`c1738b5bd` memo + `7790261f6`/`3d44fd51c` wiring, VERIFIED_VIA_SOURCE_INSPECTION): both
  levers **default-OFF byte-identical DSL levers** (finisher-1 β2-rewarmup = DSL-only, the ramp
  mechanism pre-existed; finisher-2 Polyak = new `--polyak-finisher-arm`). Byte-identity verified as
  **CODE-CORRECTNESS only** (real trainer helpers round-tripped GPU-free; None-guard makes OFF
  structural) — honestly NO gt_n6 smoke, and NO training-benefit claim. **No A/B evidence exists.**
- **AUDIT ANSWER to the three named residuals — do they block IN-v7?** No, because **R-7 is not an
  IN-v7 candidate in the first place** (default-OFF, zero A/B). The residuals bind at the v7.1-ARM
  A/B gate, not at v7 launch:
  1. `steps_per_epoch=75` config-specific → **matches the n600 crucible** ⇒ CORRECT for v7; only a
     footgun for a DIFFERENT config. Not launch-blocking.
  2. `start_epoch=0` arms Polyak from run START = whole-run average, **NOT a tail finisher** —
     a documented footgun; the helper `polyak_finisher_window_provenance` derives the true value.
     This is A/B-CONFOUNDING (a naive arm measures a whole-run mean, not the basin-center finisher
     the lever names) → must be sized before a clean v7.1 arm.
  3. β2-window sizing law = **INFERRED_FROM_DOMAIN_LITERATURE / PROVISIONAL** (floor=0.1/cosine
     underived; no isolated β2-sweep) → an armed β2-rewarmup A/B measures lever+unverified-law
     jointly.
- Class **v7.1-ARM**, gate = (isolated β2-sweep to promote finisher-1's law past PROVISIONAL) +
  (correctly-sized `start_epoch` for finisher-2). Residuals 2 & 3 are the real gate content;
  residual 1 is v7-benign. Could equally sit REGISTERED-duty-to-measure. No over-scope; honest.

## ITEM 9 — Resume-registry (event-gates fired-state persistence) → **IN-v7** (matches docket; landed+tested)

- Evidence REAL (VERIFIED_VIA_EMPIRICAL_ANCHOR): `2b7332f4b` (canonical `ResumeRegistry` + `Resumable`
  protocol) + `8d349088d` (ALL 3 latching gates routed, was muon-only) + `0295659e1` fold;
  **`test_resume_registry.py` = 320 lines, 16 tests incl `crash-resume bit-equality` + static
  gate-coverage.** Byte-identical for cap-only/legacy. Persists lane_band/chroma fire-state +
  chroma-detector history previously UNPERSISTED.
- AUDIT: this is hardening (not a lever), and it is **load-bearing FOR v7 specifically** — v7 is the
  operator-decided EVENT baseline (item 2), so its event gates' fire-state persistence is required
  for correct v7 crash-resume. The docket class "IN-v7 once landed+tested" is now SATISFIED. The
  live run-1 (v6, frozen launch.sh) does not carry it — correct: IN-v7 refers to the v7 RESTART, not
  the interim run. No discrepancy.

## ITEM 10 — GPU-verdict hybrid → **REGISTERED-duty-to-measure** (matches docket)

- Evidence (`4487d0e58`): default `--verdict-device cpu` (byte-identical); gpu+anchor stays
  `council_pending`, gated on the n600 GPU-vs-CPU agreement probe that the governor correctly
  REFUSED beside the live run (154.2 > 117.8 GiB ceiling — P0 held, not bypassed). Not in run-1.
  Determinism measured 9/9 fwd ops N=5, but promoting on determinism alone = surrogate-not-authority
  (correctly refused). Class REGISTERED is right; the agreement table is the promotion evidence.

## ITEM 11 — fp16 cf-feats → **REGISTERED-duty-to-measure** (matches docket)

- Not active in run-1. The basis raise consumed +3.93 GiB (peak 71.54 GiB) of the same envelope
  fp16-feats' memory-for-capacity trade competes for → needs a FRESH waterfill AGAINST the basis
  raise before a class can be set. REGISTERED pending that waterfill is correct; IN-v7 would be
  premature (no measured post-basis envelope room). No discrepancy.

---

## MY CLASS ASSIGNMENTS (independent) vs DOCKET

| # | Item | My class | Docket expected | Agree? |
|---|---|---|---|---|
| 1 | Basis | IN-v7 (op-decided) | IN-v7 | ✓ citation accurate |
| 2 | Event mode | (op-decided baseline) | op-decided | ✓ citation accurate |
| 3 | Micro-batch | v7.1-ARM | v7.1-ARM | ✓ |
| 4 | Safe-compile | v7.1-ARM | v7.1-ARM | ✓ (stat misleading, D-2) |
| 5 | D16 kernels | v7.1-ARM | v7.1-ARM / REGISTERED | ✓ class, ✗ REASON (D-1) |
| 6 | #330 reclaim | v7.1-ARM (NOT IN-v7-on) | "Candidate for IN-v7" | ✗ over-reach (D-3) |
| 7 | Adaptive-ε | REGISTERED | REGISTERED | ✓ verified independently |
| 8 | R-7 finishers | v7.1-ARM | (wave-2) | ✓ residuals don't block |
| 9 | Resume-registry | IN-v7 | IN-v7 | ✓ landed+tested |
| 10 | GPU-verdict | REGISTERED | REGISTERED | ✓ |
| 11 | fp16-feats | REGISTERED | REGISTERED | ✓ |

**The COMPOSED IN-v7 set (evidence-admissible NOW):** items 1 (basis, op-decided) + 9
(resume-registry, hardening, landed+tested). Everything else is v7.1-ARM (3,4,5,6,8) or
REGISTERED-duty-to-measure (7,10,11). **No lever without an anchor is IN-v7.**

## DISCREPANCIES (loudest first)
- **D-1 (LOUD) — item 5:** docket "consuming loss term is default-off in v7" is FALSE;
  `--persistence-loss-weight 1.0` + `--length-weight 0.001` are ACTIVE in run-1/v7. Only the kernel
  DISPATCH flag is off. D16 is a stronger candidate than credited; gate = whole-step B=8 bench.
- **D-2 (LOUD) — item 4:** "8/9 bit-eq=0 CPU" laundries the decision — the 1/9 FAIL (hosc, 5.96e-8)
  IS the flag-flip lever. GPU cert = n=32 coverage-limited + fingerprint-absent → launch-flip
  inadmissible; gate = stop-time GPU re-cert.
- **D-3 (MEDIUM) — item 6:** docket "Candidate for IN-v7" contradicts the artifact's own
  "default-OFF until governed n600 disk-hop confirm"; the "funds the basis raise" rationale is moot
  (basis already waterfill-admitted). n600-rule does NOT bind the bit-identity leg, but DOES bind the
  unmeasured 7-GiB/verdict disk-hop.
- **D-4 (MINOR/STALE) — docket text:** the DOCKET STATUS UPDATE marks items 8 (R-7) and 9
  (resume-registry) "builder IN FLIGHT" — both have LANDED (`c1738b5bd`/`8a23732e4`;
  `2b7332f4b`/`0295659e1`). Convene precondition (8+9 land) is SATISFIED; the update text lags HEAD.

**Could-not-verify:** none load-bearing. (The GPU whole-step B=8 bench for items 4 & 5 is an
explicit, honestly-declared v3 residual — not a claim I failed to verify, but the un-run gate.)

Pointer 0.19110 UNMOVED — all above is MEANS (evidence-class audit); only a byte-closed n600
evaluate.py row is the END.
