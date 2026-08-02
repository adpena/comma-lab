# ddm_lr1 — REBASE THE LATTICE SOLVE ONTO THE CURRENT VEHICLE: **REFUTED at the teacher-content level** (2026-08-02)

**Arm ddm_lr1, operator-originated 2026-08-02.** Pointer honesty FIRST: submittable
**0.1910828242 [contest-CPU] UNMOVED**; own-vehicle frontier **v4d 0.9639878** UNMOVED. Every number
below is **[macOS-CPU advisory]**, `score_claim=false`, `research_only=true`. This unit is means,
not end. **Scorer slot: RELEASED UNUSED** — see §8.

---

## ANSWER (lead)

**The rebase premise is REFUTED — and it cost zero scorer forwards to refute.**

The mismatch the charter names is **REAL and SHA-bound at the pixel level** (§2): teacher vehicle
`tac.witness_dsl.v10_production_receiver.v1`, student vehicle `tr1_lotto_combined_ema_v1`, and the
solve frames are **33-unique-value, ~30%-exact-zero, mean-15–21 uint8 lattice constructions** —
MEASURED by me, not borrowed — which TR1's `sigmoid(·)*255` head structurally cannot reach.

**But that mismatch does not propagate into the teacher's scorer-space content, which is the only
thing a rebase could cure.** MEASURED over 600/600 pairs, 117,964,800 pixels (§3): the teacher's
SegNet margin field is **pixel-wise 0.9867-correlated with the free GT margin field**, with residual
RMSE only **16.6% of the GT margin σ**; its argmax is GT's argmax minus **0.116%**.

⇒ **The teacher is informationally DOMINATED by a free reference.** It is a strictly *worse* copy
(noisier soft field, 0.116%-wrong argmax) of `gt_n600.npz`, which the TR1 trainer **already memmaps
at `experiments/train_tr1_partition_renderer_mlx.py:1628`** and already consumes through
`--margin-weighted-loss` and the `exp(-GT_margin/temp)` attack weighting.

**A rebase cannot fix this, in principle (§6).** The solve is *targeted at GT's argmax* under a
tolerance cap; any tolerance-capped solve against GT is at best a noisy copy of GT's field. Rebasing
changes *which pixels realize* the field — it cannot manufacture supervisory content the objective
never contained. Separately, the exact finite-family DP is structurally non-transferable to a
learned decoder's image (§6b).

**Charter step 2 satisfied by REFUTATION** ("a clean refutation is a real result and kills the arm
honestly"). Steps 3–5 are MOOT and were deliberately NOT run (§8).

---

## §1 STORES CONSULTED (recall receipts; path + sha where load-bearing)

- `.omx/research/ddm_dw1_qa75_distill_window_20260730.md` — the QA75 distill-window DEAD-END.
- `.omx/research/ddm_pj1_projection_probe_20260730.md` — the photometric CONFOUND.
- `.omx/research/ddm_fp1_class_field_projection_20260731.md` — records `teacher~GT agree 0.99884`.
- `.omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md` — the q1 object's frames were
  **never persisted** ("constructor-realizable but never materialized").
- `.omx/research/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/{receipt.json,.done,DAG_FEED.md}` —
  the S=194.42556 control point; `d_seg=0.001159998575846354`, `d_pose=0.01663315390825408`,
  archive `e3d0581f…`, 291,205,400 B, `receiver_contract=tac.witness_dsl.v10_production_receiver.v1`.
- `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames/{manifest.json,materialize_receipt.json}`.
- `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache/cache_manifest.json`.
- `.omx/research/ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md` (the pose-axis
  analogue: base-relative, matched-control, REFUTED) + `src/tac/canonical_equations/ddm_fs1_coordinate_fit_staleness_20260802.py`.
- `.omx/research/ddm_deferral_queue_ledger_20260729.md` row QA75; `.omx/state/operator_p0_ledger.jsonl`;
  `.omx/state/canonical_task_status.jsonl` (#799).
- `experiments/train_tr1_partition_renderer_mlx.py`; `tools/ddm_b2b_segnet_field_pass.py`;
  `tools/run_ddm_ms2r_r3_box_tolerance_solve.py`.

**Prior-art sweep (denominators):** a dedicated read-only recall sweep over `.omx/research/**`
(36,550 files; 6,946 depth-1 `*.md`), `.omx/state/**` (21,372 files), the 96-row deferral ledger,
`CLAUDE.md`, `MEMORY.md`, `docs/` found **no prior proposal, queue row, or run** of "re-solve the
ms2r/C1 lattice solve on the TR1 vehicle." 143 files match `rebase`; all 19 that also mention
qa75/ms2r/distill/teacher are git-rebase or frontier-pointer-rebaseline senses. **This is an
explicit scoped negative over the scopes listed — not a claim that no such proposal exists
anywhere.** Nearest existing ideas, none of them this: gc10 row (6) SAME-VEHICLE **self**-teacher
(student's own EMA); gc12 Assumption-Adversary staleness **re-check** (not re-solve); dw1
reformulation row (3) "GT-margin-feasible projected targets"; and the executed **pose-axis** analogue
`ddm_cr2r` (REFUTED 2026-08-02).

---

## §2 The mismatch IS real, and it is SHA-bound (charter step 2)

| leg | binding | value |
|---|---|---|
| teacher archive | `materialize_receipt.json::source_archive_sha256` | `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e` (291,205,400 B) |
| teacher packet | `…::packet_sha256` | `daf1e1db6314e8cdbf63347afa35899e9891e3068428d42dc5a2fca235bb5295` |
| teacher receiver contract | ms2r receipt `candidate.receiver_contract` | **`tac.witness_dsl.v10_production_receiver.v1`** |
| student receiver arch | dw1 §4 / trainer | **`tr1_lotto_combined_ema_v1`** (TR1) |
| student endpoint (dw1) | `…/ddm_bc1_20260731/burn_out/checkpoints/stage_seg_trunk_tau_final.npz` | sha `e51178c01d3d8062…`, d_seg 0.0052766 |
| student endpoint (CURRENT) | `window_03/checkpoints/intra_seg_trunk_tau_ep00854.npz` | d_seg **0.003943024** |

**Pixel-level disjointness — MEASURED by me directly from `qa75_solve_frames` (re-derived, not
borrowed from pj1):**

| pair | frame1 mean | std | min | max | unique values | exact-zero fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.673 | 22.922 | 0 | 255 | **33** | 0.2970 |
| 133 | 17.783 | 22.095 | 0 | 255 | — | — |
| 300 | 15.277 | 19.423 | 0 | 255 | — | — |
| 599 | 17.579 | 19.801 | 0 | 255 | — | — |

My pair-0/300 means (20.673 / 15.277) reproduce pj1's R1 (20.7 / 15.3) to the stated precision, so
pj1's measurement is REPRODUCIBLE. The **33 unique values** is new here and is the sharper statement:
these are not "dark images", they are coarse lattice preimage constructions.

**So the charter's step-2 question answers: mismatch CONFIRMED, SHA-bound.** What follows is why it
does not license the rebase.

---

## §3 THE REFUTATION — the teacher's content ≈ the free GT field (MEASURED, n600, zero scorer forwards)

Producer: `tools/ddm_lr1_teacher_information_probe.py` (landed this unit).
Receipt: `.omx/research/ddm_lr1_teacher_information_probe_receipt_20260802.json`.
Inputs: the dw1 teacher logit cache (`distill_logits.f16.npy`, 1,179,648,128 B, sha
`4bb40f01620276b58f5b5b403a9d44f6369787f2537f7606b0d2a6c2b324fd0c`) and `gt_n600.npz`.
**Denominator: 600/600 pairs, 117,964,800 pixels. No subset.**

**(a) Argmax — the teacher is WORSE than GT, which is free and exact.**

```
teacher realized d_seg vs GT lstar = 0.0011601681   (per-pair max 0.00263468 @ pair 178)
```

This reproduces the ms2r receipt's `0.001159998575846354` to **1.7e-7 absolute (1.5e-4 relative)** —
residual attributable to recomputing argmax from the fp16 logit cache. Per-class error rate
(denominator = GT pixels of that class): Road `0.00198734` · Lane `0.03007360` · Undrivable
`0.00041896` · Movable `0.00734015` · MyCar `0.00088119`.

GT `lstars` has d_seg **0 by definition** — it *is* the reference. So as a hard-label target the
teacher is strictly dominated. Its only possible added value is the **soft** field.

**(b) Soft field — the teacher is a 16.6%-of-σ noisy copy of GT's soft field.**

| comparison | corr | RMSE | RMSE / σ(GT margin) |
|---|---:|---:|---:|
| teacher margin vs **GT margin, same pair** | **0.986669** | 0.247577 | **0.16564** |
| teacher margin vs GT margin, **shuffled pair** (POSITIVE CONTROL) | 0.560625 | 1.402206 | 0.93811 |

σ(GT margin) = 1.494707. Distributionally the two fields are indistinguishable (teacher mean 5.5502
/ std 1.4822 vs GT 5.5767 / 1.4858; every percentile p1→p99 within 2%) — but that alone proves
nothing, which is why the **pixel-wise** comparison above is the load-bearing one.

**⇒ the teacher carries no supervisory content the student does not already have for free.**

**(c) And the trainer already reads that free field.** `experiments/train_tr1_partition_renderer_mlx.py:1628`:
`margins = open_stored_npy_memmap(args.gt_cache, "margins")` — feeding `--margin-weighted-loss`
(`margin_weighted=True`, "100% of flips at small GT-margin") and `--distill-attack-temp`
(`exp(-GT_margin/temp)`). **dw1's own winning attack weighting was computed from GT margins, not
teacher margins.**

---

## §4 Instrument validation (no vacuous passes)

A near-identity readout is exactly the shape a dead instrument produces, so every leg is controlled:

| control | expectation | MEASURED | verdict |
|---|---|---|---|
| shuffled-pair control | must lose correlation | corr 0.9867 → **0.5606**, RMSE ×5.66 | **FIRES** |
| GT-margin cache semantics: recompute SegNet top1−top2 on `gt_f1`, 2 pairs, real frozen CPU-torch scorer | must match cache | max\|diff\| **0.0**, RMSE **0.0**, argmax agree **1.0** | **EXACT** |
| same recompute on `gt_f0` (wrong frame) — negative control | must NOT match | max\|diff\| 5.46 / 4.64; argmax agree 0.9916 / 0.9925 | **FIRES** |
| identity teacher (feed GT logits as the teacher) | d_seg 0, corr 1 | d_seg **0.0000000000**, corr **1.000000** | **CALIBRATED** |
| two independent margin implementations (`np.sort` scratch vs `np.partition` in the landed tool) | must agree | identical to all printed digits | **AGREE** |
| guard controls: zero-pairs / pair-count mismatch / degenerate stride | must REFUSE | all three raise | **REFUSE** |

The GT-cache check is the one my whole conclusion rests on, and it returned **exact 0.0** on the
scored frame while differing on the other frame — so the cache is SegNet top1−top2 on `gt_f1` by the
same path as `ddm_b2b_segnet_field_pass.py::real_segnet_field_fn`, and the comparison in §3 is
apples-to-apples.

---

## §5 THREE CORRECTIONS TO THE RECORD (all checkable)

**5a — the teacher in custody is the BOX solve, not the 1.52e-4 object. The record is wrong by 7.63×.**
`ddm_dw1_qa75_distill_window_20260730.md:17-18` states the distill field is "the precomputed SegNet
scorer response on the EXACT C1 solve frames (realized d_seg ~1.52e-4)." **MEASURED: those frames
realize 0.0011601681**, 7.63× worse. The 1.52e-4 belongs to the q1 exact control (409,526,925 B, sha
`e4cd154f…`) whose frames `ddm_rp1` confirms were **never materialized**. Two independent routes
agree: my direct n600 measurement, and fp1's already-recorded `teacher~GT agree 0.99884`
(1 − 0.99884 = 0.00116). This is a **borrowed number inside a primary artifact** — the class
`ddm_fs1` and the operating manual both name.
*Consequence:* the motivating gap is **3.40×** (ep854 0.003943024 / 0.00116), not the 25.58× the
QA75 row and gc9 fork carry — those were computed against an object that cannot be a teacher because
it does not exist on disk.

**5b — the charter's two quotes are pj1's, not dw1's.** "the premise is UNusable for this pair
because the C1 solve is a DIFFERENT VEHICLE's output" and "fit-to-solve-frames objective is
mis-specified for the (TR1-renderer, C1-solve) pair" are both from
`ddm_pj1_projection_probe_20260730.md` (§2 / lead), describing the **photometric** probe. dw1's own
named mechanism is different: a **realization gap** — "the teacher's dark-knowledge directions do not
survive the deploy path uint8-STE/R round-trip at this near-floor operating point" (dw1 §5). dw1
explicitly chose scorer space *because* it believed that route was vehicle-agnostic. The operator's
underlying intuition (cross-vehicle contamination reaching the scorer-space teacher too) was the
right thing to test — §3 tests it and finds a different, stronger answer.

**5c — a SECOND, unnamed mismatch: dw1's verdict is STALE.** dw1 measured against student endpoint
E2 0.0052766 (control B 0.0051147); the current vehicle is ep854 **0.003943024** — the partner moved
**23% below dw1's control**. Per `ddm_fs1_coordinate_fit_staleness_20260802`, dw1's window result is
a fit against a moved partner. **The §3 refutation is staleness-IMMUNE** — it compares teacher to GT,
neither of which moved — so it holds at any endpoint, whereas dw1's window numbers do not.

---

## §6 Why the rebase is blocked, on two independent grounds (DERIVED)

**6a — Content (the binding one).** The ms2r solve's objective is "match GT argmax to within a
tolerance cap, minimizing coded bytes." Its supervisory content is therefore **bounded above by GT's**
by construction, for any realization. Rebasing re-chooses which pixels realize the field; it cannot
add information the objective never sought. Formally: for any tolerance-capped solve `T` against GT
`G`, `I(T; d_seg-optimal-student) ≤ I(G; ·)`, with equality only at zero tolerance — at which point
`T = G` and the solve is redundant with a free cache. **This argument is realization-independent and
therefore covers every possible rebase, not just the one I did not run.**

**6b — Machinery (independent).** The ms2r exactness rests on the camera plane being a **free uint8
lattice** with a `DisjointResizeOperator` making the resize preimage separable, so the finite-family
DP is exact (`tools/run_ddm_ms2r_r3_box_tolerance_solve.py` → `uint8_lattice_feasibility`,
`solve_binary_pair_lattice`). TR1's reachable set is the image of a learned decoder over a
24×32/384-cell/code-width-4/16-level token space — **not a lattice and not separable**. Constraining
the solve to it destroys the DP; any "rebased solve" is gradient descent in token space, i.e. it *is
training*, under a new name. Building that would be the recorded
`built_new_machinery_instead_of_paying_identified_debt` poison.

---

## §7 A cleaner mechanism for dw1's negative (re-explanation, MEASURED-grounded)

dw1 attributed window A's reversal to soft directions not surviving the deploy round-trip. §3 gives a
simpler mechanism that predicts the same signature: since teacher argmax ≈ GT argmax (99.884%) and
teacher margin ≈ GT margin (corr 0.9867), **the KD term at `w=100` was a duplicate — a 16.6%-noisier
copy of the field the base loss was already weighted by.** Window A was not chasing a new target; it
was *double-weighting a noisier copy of its existing one*. That predicts exactly what dw1 observed:
the smooth KD loss falls monotonically (it is fitting the added noise) while realized d_seg rises.
It also explains why `argmax_ce` (CE to a target 99.884% identical to GT) landed mid-pack rather than
catastrophically, and why the attack weighting — computed from **GT** margins — was what made any
form realize at all.

**Falsifiable prediction registered (NOT measured here):** a GT-soft-margin KD term at comparable
weight on a converged endpoint should *also* hurt, for the same duplicate-lever reason — being exact
rather than noisy does not make it non-redundant. This retires the expected value of dw1
reformulation row (3)'s "GT-margin-feasible projected targets" *as a distillation teacher*, unless
someone first shows the base loss is not already consuming that field.

---

## §8 What I did NOT do, and why (anti-goldplating + slot honesty)

- **Charter step 3 (re-solve on the current vehicle): NOT RUN.** Moot under §6a, structurally blocked
  under §6b.
- **Charter step 4 (re-run dw1's distill window + pj1's capacity fit at n600 against a rebased
  teacher): NOT RUN.** There is no admissible rebased teacher to run them against.
- **Charter step 5 (matched control, old vs rebased teacher, same pairs/machinery):** run **in the
  domain that settles the question for free** — teacher-vs-GT on identical pairs with a shuffled-pair
  control (§3/§4). The training-window form is moot for the same reason as step 4.
- **SCORER SLOT: RELEASED UNUSED.** Total scorer compute consumed: **2 pairs × 2 frames** for the
  §4 GT-cache validation. The n600 slot is free for another arm. Holding a scarce resource I no
  longer need is the recorded fleet-cap poison; MAIN should re-route it.
- **Observed apparatus warning (NOT mine, reported not chased):** importing with the repo root on
  `sys.path` triggers module-scope execution in `tools/mq1_joint_pose_refine_emit.py` — it prints a
  histogram and `LAW claims … MATCH: False` **at import time, without raising**. A law check that
  runs and fails silently at import is a confound-class surface. It did not affect any number here
  (the §4 GT check returned exact 0.0 afterwards), but it belongs to whoever owns mq1/pose.

---

## §9 Verdict + verdict_scope

**VERDICT: the "rebase the lattice solve onto the current vehicle" premise is REFUTED.** The
cross-vehicle mismatch is real at the pixel level and irrelevant at the content level; the teacher is
informationally dominated by a free field the trainer already consumes; and no realization change can
alter that.

- **verdict_scope: FAMILY** (broader than dw1's FORMULATION) — this closes **every solve-derived
  teacher whose objective is "match GT argmax under a tolerance cap," at any realization, on any
  vehicle**, including all rebases. The argument in §6a is realization-independent, so this is not a
  one-formulation negative.
- **NOT closed by this unit:** teachers whose objective is *not* GT-argmax-matching (e.g. a
  self/born-again teacher at a *different* capacity, gc10 row (6); cross-class KD, dw1 row (5)); the
  capacity question itself; the TR1 renderer paradigm; and the ms2r solve's value as a **rate-side**
  object (§10).
- **Labels:** §2 pixel stats, §3 all numbers, §4 all controls = **MEASURED**. §6a, §6b, §7 mechanism
  = **DERIVED**. §7 prediction = **PREDICTION, unmeasured**.

---

## §10 LIVE-HYPOTHESES / DEAD-ENDS / NEXT-IF-RESUMED

- **DEAD-END (verdict_scope: FAMILY):** solve-derived distillation teachers built by
  tolerance-capped GT-argmax matching — including any rebase onto TR1. Do not re-open by changing the
  realization; only a teacher with a *different objective* escapes §6a.
- **REFRAME (the constructive residue):** the ms2r solve is a **RATE object, not a distortion oracle**.
  Its distortion (0.00116) is strictly worse than the free GT reference (0); what it actually proves
  is *how many bytes* it costs to realize near-GT argmax through the v10 receiver (291,205,400 B,
  99.7% rate). Every future consumption should treat it as an upper bound on the rate side and never
  as a distortion teacher. This is consistent with the `ddm_is1` operator framing (typed diff
  Δ = solved_archive − base) and with the r6cal rate verdict.
- **LIVE-HYPOTHESIS (untested, cheap):** because dw1's student moved 23% (§5c), dw1's *window*
  numbers are stale. If anyone wants the plain-continuation dividend re-priced at ep854, that is a
  training question independent of any teacher.
- **NEXT-IF-RESUMED:** (1) correct the QA75 ledger row + gc9/gc12 fork tables to cite **0.00116 /
  3.40×**, not 1.52e-4 / 25.58× (§5a); (2) run `tools/ddm_lr1_teacher_information_probe.py` as the
  **admission gate on any future teacher** before spending a window on it — it is n600 and costs no
  scorer forwards; (3) release the mq1 import-time law failure to its owner (§8).

---

## §11 Custody (durable paths + shas; no `/tmp`)

- Probe (landed, this unit): `tools/ddm_lr1_teacher_information_probe.py`.
- Receipt (durable, committed): `.omx/research/ddm_lr1_teacher_information_probe_receipt_20260802.json`
  — schema `ddm_lr1_teacher_information_probe.v1`, `scope=FULL`, 600 pairs / 117,964,800 px,
  `control_fired=true`.
- Teacher logit cache (READ-ONLY, untouched, REBUILDABLE via `tools/ddm_dw1_build_distill_field_cache.py`):
  `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache/distill_logits.f16.npy`
  (1,179,648,128 B, sha `4bb40f01620276b58f5b5b403a9d44f6369787f2537f7606b0d2a6c2b324fd0c`).
- Solve frames (READ-ONLY, untouched): `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames/`
  (600 × `(2,874,1164,3)` uint8; from archive sha `e3d0581f…`).
- GT authority cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`.
- **No artifact was created, moved, or deleted on the SSD by this unit; no run dir was touched;
  `experiments/ddm_v4c_resolve.py` untouched.**

Pointer delta: **UNMOVED** (submittable 0.1910828242 [contest-CPU]; own-vehicle v4d 0.9639878).
