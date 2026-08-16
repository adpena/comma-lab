# ddm_rc2 — the lr prerequisite probe, and the unified seg+editability regime charter

Date: 2026-08-16 · Arm: `ddm_rc2` (REGIME-CHARTER COMPOSER, build-and-seal only; MAIN fires)
Axis of every number below: `[macOS-CPU advisory]` / `[contest-CUDA T4 n600]` as labelled at each
row. Nothing here is a score claim. This arm launched nothing, spent no Modal, and touched no
Metal slot.

Frontier at composition time: **hv1 ep0634 S 0.15959729295498598 @ 182,759 B**
`[contest-CUDA T4 n600]`, archive sha256
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
Gap to sub-0.15: **−0.0095973** (seg 0.029611 · pose 0.0082946 · rate 0.1216917).

---

## ANSWER FIRST

**Three of the four assumptions in my own charter brief are wrong at source. I checked the code
instead of the memos, and the corrections change what the charter should build.**

1. **b2e's NEXT #3 — "put the edit operator inside the forward pass" — is ALREADY BUILT, and it
   was ALREADY ON for the edit that was refused.** `EditabilityLevers.applied()`
   (`src/tac/pr130_lift/editability_levers.py:331`) is a straight-through in-forward operator: the
   forward sees the perturbed/regridded weight, the gradient reaches the clean parameter. The b2e
   window ran `--weight-qat-q3q4` (F2), so for `mixed_q3q4` the edit *was* in the loop and collapse
   was still 0.945. NEXT #3 is therefore **half-discharged**, and the honest re-statement is:
   *at lr 2e-7 the window did not train, so F2-on and F2-off are indistinguishable.* The lr probe
   is strictly prior to every other question, exactly as b2e's own NEXT #2 said.
2. **The trainer does NOT refuse non-uniform QAT grids.** `--bits` must be 4 because the deployed
   packer is int4-only (`train_semantic_quantized_resumable.py:820`), but F2 requests the mixed
   q3/q4 grid through **its own flag** and owns the quantization inside the forward. **No trainer
   extension is needed for mixed-grid QAT.** My brief said otherwise; it was wrong.
3. **`_live_margin_weight` (#925) is not on this vehicle.** It lives in
   `experiments/train_witness_realized_through_R_mlx.py:1086` — the MLX witness trainer. There is
   **no** `hinge_weight` anywhere in `src/tac/pr130_lift/`. So "compose with #925 + #888" is a
   **PORT, not a compose**, and I am not going to pretend otherwise. The pr130 seg objective is
   `curriculum_loss` (`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:181`), which returns a
   **scalar** — already `.mean()`-reduced. A band-weighted objective needs a per-pixel variant.
   That is the one real code delta the charter owes.
4. **Wall-clock is already MEASURED — do not re-take the timing smoke.** The b2e 50-step governed
   smoke landed **166.30 s / 50 steps end-to-end** (≤3.326 s/step, an upper bound because it
   amortizes two full-600-pair evals over only 50 steps), **peak RSS 2,779 MiB under a 12 GiB
   admission**, ×2 receipts, through `tools/safe_run.py`. Same trainer, same device, same lever
   set. The probe inherits it as a planning bound and re-measures its own split.

**The single never-fired lever is F3 (`--film-row-dropout`).** Two of b2e's three refused edits are
FiLM row prunes, and the lever that would train for them was OFF. That is the largest untested
surface in the whole editability thesis, and it costs one flag.

**Routing.** Fire the bounded lr probe (§1) before anything else. The charter (§2) is drafted and
sealed but **must not fire until the probe answers**, because every one of its levers is a
multiplier on a training signal the probe is checking exists at all.

---

## §1 — PREREQUISITE PROBE TICKET (sealed; MAIN fires)

**Name:** `ddm_lr1` — does any learning rate move the burn-2 base?
**Owner of the FIRE:** MAIN (governed Metal slot). **Owner of the build:** none — no new code.
**Status:** SEALED, READY_TO_FIRE. Every flag below is verified against the real argparse
(`train_semantic_quantized_resumable.py:721-836`). No flag is invented.

### The question, stated so it can be answered NO

b2e measured ΔS_adv **+0.000336** and a **9-byte** weight-entropy change across 3,000 steps. Either
(a) the object is at a genuine optimum that no lr escapes, or (b) lr 2e-7 was ~2 orders too small.
These have opposite consequences for the charter, and no receipt distinguishes them.

### The instrument — free, in-loop, no advisory row

The trainer already emits `quantized_exact_seg` at step 0 (line 934, when `--resume-from` is
absent) and at every `--eval-every` step: the EMA shadow, fake-quantized at the deployment grid,
rendered through the exact path, read back through the frozen SegNet, over the **full 600 pairs**.
That is the judge. **No 900-second advisory archive row is needed for this gate**, which is why it
is cheap enough to be a prerequisite.

Instrument honesty: `quantized_exact_seg` is measured against the transmitted tokens at 384×512
(`EVAL_H, EVAL_W`). It is **not** contest `d_seg` and its magnitude does not transfer to the score.
It is used here only for **within-probe deltas against a matched control**, which is what the
question needs.

### Arms — one variable, four values

Everything except `--lr` is held at b2e's exact setting, so this is a single-variable ladder and not
a 2×2 on the diagonal.

| arm | `--lr` | rationale (provenance ladder) |
|---|---|---|
| **C0** | `2e-7` | matched control — b2e's exact lr. Supplies the floor `F`. Class: measured-prior. |
| **A1** | `2e-6` | 10× — first rung. Class: derived (ladder spacing). |
| **A2** | `2e-5` | 100× — **the trainer's own declared default** (`argparse` line 738). Class: code default, not invented. |
| **A3** | `2e-4` | 1000× — upper bracket; answers "does it break" as well as "does it move". Class: derived (ladder spacing). |

### Command shape (single arm; substitute `--lr` and the run dir)

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 12288 --timeout 4200 \
  --label ddm_lr1_<arm> \
  --status-receipt /Volumes/APDataStore/pact/ddm_lr1/<arm>/safe_run_status.json \
  --child-pidfile  /Volumes/APDataStore/pact/ddm_lr1/<arm>/child.pid \
  -- \
  .venv/bin/python -m tac.pr130_lift.train_semantic_quantized_resumable \
  --challenge-root upstream \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init  /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/\
checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --bits 4 \
  --weight-qat-q3q4 \
  --steps 600 --lr <ARM_LR> --float-warmup-steps 0 \
  --ce-fraction 0.50 --softplus-fraction 0.85 \
  --eval-every 100 --checkpoint-every 100 \
  --device mps --seed 20260715 \
  --out  /Volumes/APDataStore/pact/ddm_lr1/<arm>/result.json \
  --save /Volumes/APDataStore/pact/ddm_lr1/<arm>/checkpoints
```

**Pins (re-hash at fire time; do not trust these strings as current).** `--init` sha256
`3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647` (== sd1
`EXPECTED_CHECKPOINT_SHA256`); `--cache` sha256
`382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`; `upstream/modules.py` sha256
`065961ba…` (== sd1 `EXPECTED_UPSTREAM_MODULES_SHA256`). All three resolved sha-exact by MAIN on
08-16 for b2e. **The trainer file itself changed when the levers landed — re-hash it at fire time
rather than citing any pin recorded before 08-16.**

`--weight-qat-q3q4` stays ON in every arm, including the control, so the probe measures lr inside
the regime the charter will actually run.

**Why the wrapper flags are not optional — a defect I caught in my own first draft.**
`tools/safe_run.py` defaults are `--rss-mb 2048` and **`--timeout 30.0`** (a wall-clock kill,
line 33/532, exit 124). My first draft wrote a bare `safe_run.py -- CMD`, which would have killed
every arm at **30 seconds** and at a **2 GiB** cap against a **measured 2,779 MiB** peak — two
independent silent kills. `--rss-mb 12288` matches b2e's admitted footprint (~4.4× headroom over
the measured peak); `--timeout 4200` is ~2.1× the 1,996 s upper-bound arm. `--status-receipt` and
`--child-pidfile` are the watchers-at-launch requirement, and the pidfile is how the arm is killed
— never by pattern-matching the wrapped argv.

### What differs from b2e, declared

**SCOPE reduction only.** `--steps` 3,000 → 600 and `--eval-every` 250 → 100. No mechanism changes.
Consequence to state rather than hide: `--steps` also sets the cosine `T_max`
(line 918) **and** the EMA decay through the run-geometry LawRef `ema_decay_run_geometry_v1`
(line 921). So C0 is **not** a bit-replay of b2e's first 600 steps; it is a **matched control inside
this probe**, which is exactly what a self-measured floor requires. Absolute magnitudes from this
600-step geometry do **not** transfer to a 3,000-step window — only the ordering and the
moved/not-moved verdict do.

### The pre-registered bar

For each arm let `Δ = max over eval points of |quantized_exact_seg(step) − quantized_exact_seg(0)|`
and let `F = Δ` of the C0 control.

> **MOVED** iff `Δ_arm > 3·F` **and** `Δ_arm > 1e-5` (raw, this instrument).

The `3·F` leg is the matched-control leg (m85). The absolute leg is **derived, not picked**: the
rate exchange on the retained rt1 ledger is `6.658589531221714e-07` S per byte, and the smallest
byte pool anyone has actually measured as harvestable on this object is b2e's best edit at
**968 B** → `968 × 6.6586e-7 = 6.446e-4` S of rate → a permissible seg movement of
`6.446e-6` in raw d_seg terms. `1e-5` is ~1.5× that. ⚠ **Order-of-magnitude anchor only** — the
in-loop metric is not contest d_seg, so this fixes the scale, not an exchange rate. Say so in the
receipt.

### Firing order and the trajectory stop

Fire **A2 → C0 → A1 → A3**. A2 is the trainer's own default and the most likely to move; C0 then
supplies the floor that interprets it.

- **Stop after two arms** if A2 clears the bar by ≥10× and C0 is flat: the gate is answered, and
  A1/A3 become refinement rather than gate. Record the stop (cap-stop / trajectory-stopping law).
- **Run all four** if A2 is ambiguous, or if A2 moves *upward* (seg worse) — then A1 tests whether a
  smaller step descends and A3 tests whether the move is a scale artifact.

### Budget, from the measured rate

Upper bound per arm = `600 × 3.326 s ≈ 1,996 s ≈ 33 min`; four arms ≈ **2.2 h serial**. The bound
over-states, because 3.326 s/step already amortizes two full evals over 50 steps. **Arm 1's receipt
must report the train-step/eval split from its own history timestamps**, and MAIN confirms the
remaining budget from that measured split before firing arms 2–4. I did not invent a split.

Memory: peak RSS 2,779 MiB at a 12 GiB admission — one arm at a time, governed, well inside the
single-Metal-fire rule.

### Resumability (P0) — satisfied by construction

`--checkpoint-every 100` over 600 steps = 6 per-stage checkpoints per arm; the resume payload
carries model + EMA + optimizer + scheduler + generator + order/cursor + RNG + history + best-EMA,
and the deployment state written is the EMA shadow.

### Payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_lr1/<arm>/` — **APDataStore, not VertigoDataTier**, which has
**954 MiB free** (measured 08-16). Retain per arm: every checkpoint, `result.json` with the full
`history` list (the eval trajectory *is* the measurement), the `safe_run` status, and the launcher
tree. The trajectory is the payload here; a scalar-only receipt would be the measure-and-discard
defect.

### What the probe CANNOT conclude

- Nothing about the contest score. It never builds an archive.
- Nothing about **editability** — collapse factors need the b2e admission harness, which is not run
  here.
- A NO does **not** refute the charter's objective term. It says *this trainer/init/curriculum
  cannot be moved by lr alone*, which routes the charter to the re-route in §2.5.

---

## §2 — THE UNIFIED SEG + EDITABILITY REGIME CHARTER (draft; DO NOT FIRE until §1 answers)

**Proposed name:** `ddm_rg1` (MAIN may rename). **One burn, composing the seg-axis supplier that
rt1 measured with the editability levers that ns1 located and b2e half-tested.**

### §2.0 The thesis, in one line

rt1 measured that 99.22% of the seg axis sits on a curve one pixel wide that the decoder already
owns for free, and that 98.3% of those flips are a **tie** the render loses by a median **0.105**
logits. An objective that spends its gradient budget in proportion to the measured per-edge debt on
exactly that curve is the only carrier-free supplier rt1 left standing. Editability rides the same
burn because it is free to add and because ns1's screen shows the rate half cannot be harvested any
other way.

### §2.1 (a) The edge-weighted Road↔Lane 1-px-band objective term

**The band.** Ring-0 of the transmitted label boundary — city-block (4-neighbour) distance, `0` = on
the boundary — recomputed per batch on device from the loss target field:

```
band = (t != up(t)) | (t != down(t)) | (t != left(t)) | (t != right(t))     # edge-replicated
```

This is verbatim rt1's `ring_definition`. It is a deterministic function of the tokens the decoder
already parses, so it costs **zero archive bytes**.

**Pre-registered instrument control (free, exact, before any training):** the n600 union of `band`
must reproduce rt1's `ring_population[0] = 2,551,464` px exactly, checked against the retained
`free_band_mask.npy`
(`/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/free_band_mask.npy`, 117,964,928 B).
A mismatch means the band operator or the target field is not the one rt1 measured, and the burn
does not start.

**The weight.** Per-edge-type, **debt-proportional** — proportional to marginal S per band pixel of
that edge class:

```
W_e  ∝  flips_e / band_px_e
```

`flips_e` is derivable **today** from the retained `RT1_EDGESHAPE.json::confusion_pred_to_gt`
(symmetrised over the unordered pair). Worked from that receipt: Road↔Lane = `0->1` 8,491 +
`1->0` 6,687 = **15,178** of 34,938 = **43.44%** (reproduces rt1's 43.4%); Road↔Undrivable 6,975;
Undrivable↔Movable 5,936; the three sum to 80.4%, also reproducing rt1.

`band_px_e` is **not** in any retained receipt — `ring_population` is aggregate. It is a **$0 desk
recomputation** from the retained band mask plus the label field, and it is **step (0) of this
charter's build**. Until it is computed the weight table does not exist; I am not going to
substitute flip-share alone and call it debt-proportional, because that would silently weight by
edge *length* as well as by debt.

**No class index is hardcoded.** The table is keyed by the unordered class pair present at each band
pixel and resolved at config time from the retained receipt, honoring the self-detect rule.

**Hard requirement — scale neutrality.** Normalize so `mean(w) == 1` over the field. Otherwise the
band weight silently rescales the loss, which rescales the effective learning rate, and confounds
the exact quantity §1 just measured. This is a build-time assertion, not a convention.

**The code delta (the one real build this charter owes).** `curriculum_loss` returns a scalar.
Add a **new** module `src/tac/pr130_lift/band_objective.py` exposing
`curriculum_loss_weighted(logits, target, step, total_steps, ce_fraction, softplus_fraction,
weight=None) -> (loss, phase)` that reproduces the three phases per-pixel and reduces with `weight`.
Requirements, all testable:

1. **Do not edit `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py`.** It is the lifted PR130
   reference and the trainer asserts phase parity against it
   (`train_semantic_quantized_resumable.py:1063`).
2. **Positive control:** `curriculum_loss_weighted(..., weight=None)` must return a value
   **exactly equal** to `curriculum_loss(...)` on the same inputs, and the same phase string. This
   is the no-op proof, and it is the test that makes the term admissible.
3. **Byte-identity when off:** with the band term disabled the trainer must be byte-identical to
   today, including RNG consumption — the same discipline the b2e levers already honor.
4. New flags follow the existing default-off lever pattern (`--band-objective-weight` default
   `0.0`, inert at 0), with the activation counter reported in the receipt so a held-but-never-fired
   term is detectable rather than silent.

**What this term is NOT.** It is not a port of `_live_margin_weight`. That consumer is on the MLX
witness vehicle and weights by margin **magnitude**; this term weights by **measured per-edge S
debt on a geometrically-defined band**. If someone later wants the margin-magnitude weight too,
that is a second, separate variable and it does not ride this burn.

### §2.2 (b) Edit-operator-in-forward QAT

All four levers already exist, straight-through, default-off, 17 tests pinning the resume guard.

| lever | flag | this burn | why |
|---|---|---|---|
| **F2** mixed q3/q4 grid | `--weight-qat-q3q4` | **ON** | already exercised by b2e; trains the grid we deploy onto |
| **F3** FiLM row dropout | `--film-row-dropout`, `--film-row-dropout-protect-top` | **ON** | **NEVER FIRED.** Two of b2e's three refused edits are FiLM row prunes and the lever that trains for them was off. Largest untested surface in the thesis. |
| **F1** weight perturbation | `--weight-perturb-robustness`, `--film-critical-multiplier` | **ON** | multiplier default is `sqrt(93.7) ≈ 9.68`, **derived** from ns1 §A's measured ~94× sensitivity spread — not a borrowed constant |
| **F4** rank penalty | `--carrier-rank-penalty`, `--carrier-tensors` | **OFF** | no rank edit is in the deployment distribution. Off **with a recorded reason**, per the tracked-queue law; reactivates if a rank edit enters the edit set. |

`--film-row-dropout-protect-top` is the trainer-side realization of ns1's **FiLM protection list**
(`blocks_1` FiLM rows excluded from coarsening). Its value is derived at config time from
`editability_levers.film_row_order` over `POSE_CRITICAL_TENSORS`, not typed by hand.

**On confounding, stated rather than dodged.** Three levers at once is not an attribution design.
This burn's job is to **produce a better object**, and attribution is already owned by a built
instrument: `experiments/ddm_b2e_edit_replay_admission.py` measures **per-edit** collapse against
the pinned 50× bar. So the composed regime runs; the admission harness attributes. Any claim of the
form "F3 did it" requires an F3-off arm and is out of scope here.

### §2.3 (c) Vehicle decision — the semantic trainer, not the wd3 stack

**Decision: `src/tac/pr130_lift/train_semantic_quantized_resumable.py`.** From the code and the
receipts, not from preference:

*For the semantic trainer.* It trains the object the frontier actually ships — its
`deployed_argmax_parity` gate validates the real `pack_semantic` export, and the b2e smoke confirmed
the packed export at **40,252 B == sd1 `EXPECTED_BASE_SEMANTIC_BYTES`**. All four editability levers
live there. Resume, per-stage checkpoints, EMA-shadow deployment, the run-geometry EMA LawRef, and
the governed-admission assertion are all already wired and smoke-proven.

*Against the wd3 stack.* It is a **different object with its own receiver**
(`experiments/ddm_wd3_student_receiver.py`) and its own compiled-config DSL. Its fresh-init family
is **FAMILY-NEGATIVE at 65ep**, confirmed on seeded nonprefix n120: D56 **2.01×** and F64 **2.41×**
the matched baseline's `hard_d_seg`. Its one live rung — warm-lineage width change — is a **rate**
play (W0_warm's semantic packet 21,807 B ⇒ ≈16 KB saved) whose **seg half projects ~6× over its
byte-derived bar** and is TRAJECTORY-STOPPED on that arithmetic. Routing a **seg-lowering** burn
into a stack carrying a measured 6× seg deficit is the wrong instrument for the axis.

*The tradeoff, honestly.* The wd3 stack owns the larger single byte pool (~16 KB vs b2e's best
measured 968 B edit). This charter gives that up on purpose, because rt1 measured that the byte
half cannot close the gap alone (the correction channel is break-even at +0.00029 S) while the seg
half is 0.029611 S with a supplier standing. The wd3 warm-lineage rung stays **PARKED in the rate
lane** with its own recorded reactivation criterion — any distillation arm whose projected Δd_seg
lands within ~1.5× of its byte bar fires the n600 same-instrument row first. **No rival lane.**

### §2.4 (d) The pose leg — what this burn does NOT touch

This burn does not optimize pose and must not claim to. Pose routes to **P3 / js8**: the uncapped,
per-pair, realized-acceptance GN solve on the hv1 base with seg-hold, whose named prerequisite is
the **#850 iteration-cap lift** (every pose GN in the corpus stopped at a cap while still descending
13–23%/iter). That is the js8 successor's first concrete row, not a lane this charter opens.

**The interface between them, so they compose instead of colliding:**

1. This burn inherits ns1's screen as a **guardrail**: `Δd_pose_budget ≈ 5.1e-9 · ΔB` at this
   operating point. A stage whose pose damage exceeds it has bought bytes at a loss and is
   reported as such.
2. This burn inherits the **FiLM protection list** (§2.2), which is the trained-side form of the
   same finding.
3. This burn **reports d_pose at every stage boundary** so js8 receives a clean, pinned base rather
   than a moving one.
4. Pose contribution is **0.0082946** of the −0.0095973 gap — 86% if driven to zero. That is js8's
   to claim, and this charter explicitly does not double-count it.

### §2.5 Re-route if the probe says NO

If no lr moves the base beyond the §1 bar, the charter does **not** fire as written. It re-routes,
in this order:

1. **The objective, not the step size.** A converged object under a *different* objective is not at
   the same optimum. Fire the band-weighted term at the arm's own best lr as the single new
   variable — the term changes the loss surface, which is a different question from the one §1
   asked. This is the cheapest re-route and it keeps the seg supplier alive.
2. **The init.** The b2e init is itself a 6k-step lr-2e-7 tail; a genuinely stuck object may need an
   earlier lineage checkpoint. Named, not assumed — requires a checkpoint inventory first.
3. **Only then** the wd3 warm-lineage rung, in the rate lane, under its own reactivation criterion.

A NO is a finding about **this trainer, init, and curriculum**, `verdict_scope: formulation`. It is
not a verdict on train-for-editability, on the band objective (never tested), or on the family.

### §2.6 OPTIMAL FORM

**Reference form.** The PR130 semantic QAT trainer at its shipped geometry: `w96 / b4`, uniform
int4 deployment packer, `--steps 12000` default, `--lr 2e-5` default, `--ce-fraction 0.50`,
`--softplus-fraction 0.85`, EMA decay derived from run geometry via `ema_decay_run_geometry_v1`,
EMA shadow as the deployment state, `deployed_argmax_parity` over 600 pairs.
Receipt: the shipped init `semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, sha256
`3948ccfc…`, == sd1's `EXPECTED_CHECKPOINT_SHA256`.

**Declared deltas.**

| delta | class | note |
|---|---|---|
| step budget set from §1's measured s/step and the governed slot | **SCOPE** | legal; the budget is a slot decision, not a mechanism |
| eval/checkpoint cadence | **SCOPE** | legal |
| **M1** band-weighted objective term | **MECHANISM** | the regime under test; verdict is scoped to the composed regime |
| **M2** F1 + F3 active alongside F2 | **MECHANISM** | as above |
| F4 off | recorded non-activation | tracked-queue law; reason and reactivation condition on record |

**This is not a toy bracket.** Every delta is either a scope reduction or the mechanism the charter
exists to test. n600 throughout — the in-loop eval, the parity gate, and the admission rows all run
the full 600 pairs. No prefix appears anywhere, so the m88/m96 subset laws are satisfied by
construction rather than by argument.

**Provenance pins.** `--init`, `--cache`, `upstream/modules.py`, the trainer file, and
`editability_levers.py` are all re-hashed at fire time and recorded in the run receipt. The rt1
receipts (`RT1_EDGESHAPE.json`, `RT1_GEOMETRY.json`, `RT1_LEDGER.json`, `free_band_mask.npy`) are
hashed and pinned as the weight table's inputs, so the derived `W_e` is reproducible from named
bytes.

### §2.7 Build order (nothing here launches)

0. **$0 desk:** per-edge band-pixel populations `band_px_e` from the retained band mask + label
   field → the `W_e` table, with its inputs hashed. *(Blocking: the weight does not exist without
   it.)*
1. `src/tac/pr130_lift/band_objective.py` + the `weight=None` exact-equality positive control +
   byte-identity-when-off tests. Two review passes, ruff clean, serializer commit.
2. Wire the flags into the trainer, default-off, activation counter in the receipt.
3. Band instrument control against `ring_population[0] = 2,551,464`.
4. Only then: the sealed launch ticket for MAIN.

---

## STORES CONSULTED

- `CLAUDE.md` (NO-FAKE supreme rule · THE GOAL · ALWAYS KEEP THE PAYLOAD · SSD tier + certify-or-block
  · resumability + per-stage checkpoints P0 · never-invent-flags · n600 allergy · charter-time
  OPTIMAL FORM law · MPS is a gradient device never an authority · verdict_scope ladder ·
  upstream read-only) and `AGENTS.md`.
- `.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md` §2.3–§2.8, §3.1–§3.3, §4, §5, §6.4.
- `.omx/research/ddm_b2e_edit_replay_admission_verdict_20260816.md` (full), and
  `ddm_b2e_sealed_launch_ticket_20260816.md` (command shape, resolved pins, the 50-step smoke
  receipt, the window derivation), `ddm_b2e_train_for_editability_burn2_charter_20260816.md`.
- `.omx/research/ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` §A, §C P1–P5, §D.
- `.omx/research/ddm_wd3_n120_family_disposition_20260816.md` (family verdict, W0_warm byte
  arithmetic, the trajectory stop, the reactivation ladder).
- Code, read at source rather than from memos: `src/tac/pr130_lift/train_semantic_quantized_resumable.py`
  (argparse 721-836 · loss callsite 1040-1090 · eval + history 1090-1155 · step-0 baseline 934 ·
  optimizer/scheduler/EMA 917-925 · `resolve_ema_policy` 104 · `deployed_argmax_parity` 517 ·
  `_evaluate_semantic_pairs` 655); `src/tac/pr130_lift/editability_levers.py`
  (`EditabilityLevers` 331 · `_quantize` · `_perturb` · `mixed_bit_allocation` 125 ·
  `film_row_order` 143 · `DEFAULT_FILM_CRITICAL_MULTIPLIER` 122);
  `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` (`target_margin` 174 · `curriculum_loss`
  181 · `render_for_seg` · `EVAL_H, EVAL_W = 384, 512`); `src/tac/admission_guard.py`;
  `experiments/ddm_b2e_edit_replay_admission.py` (stages `replay` / `pairs` / `admit`);
  `experiments/ddm_wd3_scorer_aware_width_distillation.py` (compiled-config CLI surface);
  `experiments/train_witness_realized_through_R_mlx.py:1086` (`_live_margin_weight`, the vehicle
  boundary).
- Retained payloads, read not cited: `/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/`
  (`RT1_GEOMETRY.json`, `RT1_MARGIN.json`, `RT1_LEDGER.json`, `RT1_EDGESHAPE.json`,
  `free_band_mask.npy`); `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/` (verdict inputs).
- Disk state measured 08-16: APDataStore **255 GiB free**, VertigoDataTier **954 MiB free**.
- Memories: `m04` (own-vehicle frontier), `m85` (matched-base control), `m88`/`m96` (subset and
  prefix-bias laws), `m94` (a negative measures the instrument), `wd3_fresh_topology_pose_carry…`
  (judge at the seg asymptote), `default_off_is_orphaned_signal…` (tracked-queue),
  `caps_genus_trajectory_stopping`, `first_attempt_wall_clock_is_not_a_family_verdict`,
  `cross-regime-constant-transfer`, `naive_first_pass_born_at_charter_time_optimal_form_law`.

---

## NEXT_IF_RESUMED

1. **Fire `ddm_lr1` (§1) before anything else.** A2 first, then C0. Four arms ≈2.2 h serial upper
   bound; arm 1's receipt supplies the true train/eval split and MAIN re-confirms the budget from it.
   Governed via `tools/safe_run.py` — a raw launch is refused by the admission guard, correctly
   (b2e's first attempt hit exactly that and the guard was right).
2. **Do not re-take the timing smoke.** 166.30 s / 50 steps, peak RSS 2,779 MiB @ 12 GiB, ×2
   receipts, same trainer / device / lever set. Re-taking it is rediscovery.
3. **Correct the b2e NEXT #3 line at source.** The edit operator is already inside the forward pass
   and was already ON for `mixed_q3q4`. The live untested lever is **F3 `--film-row-dropout`**, and
   the live open question is the lr, not the placement. A successor who reads the b2e memo without
   this correction will rebuild something that exists.
4. **Charter step (0) is blocking and is $0:** per-edge band-pixel populations from the retained
   band mask → the `W_e` table. The band objective has no weight table until it runs. It needs no
   GPU and no launch.
5. **`band_objective.py` is NOT built.** I deliberately did not build it: the probe gates the whole
   charter, and building a training term before knowing the object trains is infrastructure ahead of
   its row. The spec in §2.1 is complete enough to build against — new module, `weight=None` exact
   equality to `curriculum_loss` as the positive control, mean-weight-1 normalization asserted, lifted
   oracle untouched.
6. **Pose stays with js8.** #850's cap lift is its prerequisite. This charter's only pose duty is to
   report d_pose at stage boundaries and to stay inside the `5.1e-9·ΔB` screen.
7. **Nothing in this unit moved the pointer, and it was not permitted to.** Own-vehicle frontier
   unchanged: **hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`**.
