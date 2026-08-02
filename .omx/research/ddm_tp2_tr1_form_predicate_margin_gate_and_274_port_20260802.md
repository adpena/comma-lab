# ddm_tp2 — two live TR1 bugs fixed, and #274's producer moved onto the live vehicle (2026-08-02)

**Evidence axis: `[macOS-CPU/MLX advisory]`. No scorer forward in anything landed here, no race fired,
canonical effective-frontier pointer UNMOVED (0.1910828242 [contest-CPU]).**

Three commits, one mechanism each:

| commit | row | mechanism |
|---|---|---|
| `a689f8ad09` | 1 | basin predicate keys on the loss FORM, not the launch-flag-dependent stage LABEL |
| `59a1793b44` | 2 | `--margin-weighted-loss` REFUSES when inert for a reachable seg form |
| `ee300ff82f` | 3 | #274 spike/coherent PRODUCER ported into TR1, cross-validated at n600 |

---

## ROW 1 — a stage predicate that could never be satisfied

**MEASURED at `experiments/train_tr1_partition_renderer_mlx.py:983` before fixing.**
`basin_entry_fires` compared the DISPLAY label by exact string, `x["stage"] == "seg_trunk_tau"`.
`stage` is a label whose text varies with `--seg-form-start`:

| launch | label | can the predicate fire? |
|---|---|---|
| `--seg-form-start ce` | `seg_trunk_ce`, rewritten to `seg_trunk_tau` by the knee event | yes |
| `--seg-form-start tau_softplus` | `seg_trunk_tau_softplus`, **forever** | **never** |

In the second case `knee_switched = (stage != "seg_trunk_ce")` is True from epoch 0, so BOTH
form-switch blocks are guarded off and nothing can ever rewrite the label. Such a run trains the
IDENTICAL `tau_softplus` loss and is structurally incapable of entering the basin.

**FIX:** key on `state_form["form"]` — the state machine the switch events actually mutate, which is
`"tau_softplus"` in both launch paths. The `basin_window` row now carries `form`; `stage` is retained
as a human label only and is never compared. `initial_stage_label()` becomes the sole owner of the
label convention (anti-drift test pins it).

**Both failure directions verified against the OLD comparison rather than assumed:**

```
regression window label = 'seg_trunk_tau_softplus'
  OLD -> False  (the bug)          NEW -> True
dual: ce-FORM window mislabelled 'seg_trunk_tau'
  OLD -> True   (a FALSE basin entry) NEW -> False
```

**BLAST RADIUS: ZERO recorded verdicts.** All four TR1 windows ran `basin_handoff='off'`, so
`basin_entry_fires` was never consumed; and all four reached `seg_trunk_tau` anyway (window_01 via the
ce knee, 02/03 via resume re-anchor). **The bug is LATENT and fires on the NEXT burn** — specifically
if it is launched `--seg-form-start tau_softplus`, the natural warm-start choice from a tau endpoint.

---

## ROW 2 — a flag that is ON and INERT for the stage that actually ran

**MEASURED by parsing the canonical loss source** (`train_witness_realized_through_R_mlx.py::make_loss_fn`):

| seg form | reads `apply_mw`? |
|---|---|
| `tau_softplus` | **IGNORES** |
| `l7_softplus` | **IGNORES** (deliberate — it carries its own hard-pixel weight, documented) |
| `margin_hinge` / `unify_tau` / `ce` | HONORS |

`tau_softplus` is a bare `mean(tau*softplus(-m/tau))` with no weight of its own, so nothing absorbs
the flag: it is declared-ON and dead.

**BLAST RADIUS (MEASURED from live telemetry, all four windows):**

| run | `seg_form_start` | `margin_weighted_loss` | form actually run | effect |
|---|---|---|---|---|
| `ddm_b4s_20260731/window_01` | ce | **on** | `seg_trunk_tau` 38/38 rows | none |
| `ddm_b4s_20260731/window_02` | ce | **on** | `seg_trunk_tau` 199/199 | none |
| `ddm_b4s_20260731/window_03` | ce | **on** | `seg_trunk_tau` 199/199 (ep807–945) | none |
| `ddm_r1c_20260731/window_01` | ce | **on** | `seg_trunk_tau` 171/171 | none |

window_03 additionally emits `resume_form_reanchor` pinning `form='tau_softplus'` from its first
epoch. **The "§3.2 boundary-annulus form fix" was inert for every epoch of the burn lineage.** Any
verdict conditioned on margin weighting being active on those windows is void; the runs' actual
physics is `margin_weighted_loss='off'`.

**DECISION: REFUSE, not wire.** Justification:
1. Wiring changes the live vehicle's **training loss**. That is a score-affecting lever, and levers are
   RACED, never adopted by citation — it would also make the next window incomparable to 01–03.
2. `l7_softplus` ignores the flag **by design**, so "wire it everywhere" would contradict an existing
   design decision. The honoring set is a real surface, not a uniform oversight.
3. Refusing is fail-closed and turns a silent artifact LOUD.
4. The remedy is free: because the flag was inert, **dropping it reproduces the burn lineage
   byte-for-byte**. No third state.

**The check is on REACHABLE forms, which is the part that matters.** `ce` honors the flag, but the knee
event — whose F2 midpoint fallback makes the switch unconditional — moves ce → tau_softplus mid-run, so
a ce-start run goes silently inert *at the knee*. That is exactly how all four windows died. Hence
`reachable_seg_forms('ce') == {ce, tau_softplus}` REFUSES; terminal starts `unify_tau` / `margin_hinge`
are allowed.

Anti-drift: the honoring-set constant is asserted EQUAL to the set of branches that actually reference
`apply_mw`, parsed from the real if/elif chain with `ast`. Adding or removing any guard fails the test.

---

## ROW 3 — the #274 producer, ported (not rebuilt)

The lever was already BUILT at `train_levelset_witness_realized_through_R_mlx.py:9467-9494` with a DSL
Lever and a gauge default, its activation-ledger row reading `"ever_fired": false`. Its **producer** had
no TR1 counterpart; the **consumer** already did (`seg_pixel_w`). Only the producer moved. No parallel
surface was created.

**Why it is admissible (ddm_ti1, MEASURED n600 / 598 interior pairs / 117,571,584 px / ZERO scorer
forwards).** TR1's per-pixel seg loss is **per-pair separable** — for pair `t` it reads only `lstars[t]`,
`margins[t]` and the student's own logits — so every weight it can express is measurable w.r.t.
σ(class, GT margin). A **cross-pair** field built from `lstars[t±1]` is outside that σ-algebra **by
construction**. That is the non-redundancy frontier, and why lr1's per-pair teacher was structurally
doomed.

**CROSS-VALIDATION (the port's positive control, n600, real frozen-authority GT):**

| quantity | this port | ddm_ti1 / ddm_fl1 |
|---|---:|---:|
| SPIKE px | **625,297** | **625,297** (two independent impls) |
| interior px | 117,571,584 | 117,571,584 |
| union frac | 1.9595% | "1.96% of pixels" |

Landed as a test, so **the instrument travels with the port** rather than being loose scalars.

**ASYMMETRIC PRICING (ti1 §3a).** Nothing in the existing #274 record priced the knobs apart — the gauge
carries one default and both flags default to a symmetric inert 1.0. They are priced by **different
rules** and must be raced **separately**:

* **COHERENT is RISK-PROPORTIONAL** — race start `1.2988561739557636` **IS** its measured stratified MH
  lift. The number is the measurement; no arithmetic invented on top.
* **SPIKE is CONCESSION-priced, deliberately NOT risk-proportional** — its lift is *higher* (1.7574) but
  ~88.6% of that set is irreducible single-frame appearance change, so it is conceded at the inherited
  0.25 rather than chased.

Both trainer flags still **default to 1.0 (inert)**; these are race starting values, not defaults.

**Implementation notes.** Field stored as `(P,H,W)` uint8 codes (118 MB at n600) with a 3-entry LUT
applied at use time, not float maps (472 MB) — 4× smaller than the levelset copy because TR1 shares its
memory ceiling with the GT/gate caches. Folded **multiplicatively** into `w_np` *after* the additive
class/lane-guard accumulation, so it scales the composed weight. A value flag without its gate is
REFUSED (same genus as row 2).

---

## REPORTED, NOT CHASED — this trainer is not run-to-run bit-deterministic on this host

I tried to confirm row 3's OFF-path byte-identity by checkpoint diff and **the control failed**, so I
checked the instrument instead of patching around it.

| control (n6, 2 epochs) | identical arrays | differing | worst \|Δ\| |
|---|---:|---:|---:|
| SAME script, SAME argv, run A vs run B | 1 / 30 | 29 | **9.1e+01** |
| second control pair (kernel unset attempt) | 1 / 30 | 28 | **2.1e-03** |

`custom_grouped_backward` reported `active: true` in every run. **Checkpoint diff is therefore an
INVALID byte-identity instrument on this host**, and my earlier baseline-vs-ported comparison was
measuring trainer noise, not my change. Row 3's OFF-path identity is asserted **structurally** instead
(None sentinel; all-ones LUT even if armed inert), with both legs tested.

* `verdict_scope`: **INSTANCE** — one host, n6/2ep, two control pairs. Not a claim about other scales,
  hosts, or the mechanism. The custom-kernel attribution is **INFERRED, not measured** (my `env -u`
  attempt did not actually deactivate it).
* **Consequence for the preregistered race:** a small d_seg difference cannot be attributed to the lever
  until this run-to-run floor is measured **at the race's own scale**. dw1's 2.99e-5 in-window
  gate-residual std is a *within-run* figure and does not cover this. **MAIN should size the
  run-to-run floor before sealing any arm.** This is the one thing that could make the ti1 race
  unattributable, and it was not visible before this unit.
* Not mine to fix; collides with the deterministic-reproducibility non-negotiable and needs an owner.

---

## READY_TO_FIRE — the preregistered race (ti1 §7). NOT FIRED; MAIN fires.

Verified end-to-end at n6: producer builds, `seg_spike_reweight_ready` telemetry emits (score-neutral,
default-on), training completes. Three matched governed windows from
`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_03/checkpoints/intra_seg_trunk_tau_ep00854.npz`
under dw1's 8-guard discipline, raced **separately**:

```
B (control) plain continuation
A  --seg-spike-reweight --seg-spike-downweight 0.25 --seg-coherent-upweight 1.0
C  --seg-spike-reweight --seg-spike-downweight 1.0  --seg-coherent-upweight 1.2988561739557636
```

**FALSIFIER (preregistered):** if neither A nor C beats B's endpoint realized n600 d_seg by more than
B's own in-window gate-residual std, the temporal-instability teacher is **FORMULATION-dead as a
per-pixel seg reweight at this endpoint**, and the vein's remaining live form is representational (a
carrier that resolves temporal phase) rather than a loss weight. A seg win bought with token entropy is
not a win — bytes matched, or reported jointly.

### Honest limits carried forward, not laundered

* The flip atlas is **ep399**; the live endpoint is **ep854**. The **predictiveness** half is
  **INSTANCE-scoped and must be re-measured at ep854**. Only the **non-redundancy** half (structural,
  endpoint-free) transfers for free.
* Lift **ATTENUATES** with finer conditioning (spike 2.22 → 1.76 at 10 → 40 bins). The true value is at
  or below the 40-bin number. **Do not quote 2.22.**
* The direction leg's raw 98.17% neighbour-label match carries a locally-binary tautology risk. The
  defensible statement is the **31-point excess over the 67.16% shuffled control**, never 98.17%.
* No ΔS and no predicted band are claimed. The 0.2721 S of error mass on the unstable set is a **mass**,
  not a recoverable quantity.

---

## Inventoried, deliberately NOT built here

**`ddm_sn1`'s per-pixel `STRUCTURALLY_HARD_IRREDUCIBLE` tensor** (635,011 px, n600;
`.omx/research/ddm_sn1_error_source_tensor_n600_20260723/`) is consumed by **no training code** — the
same orphan shape #274 had before this unit.

* **Owner:** unassigned — needs one.
* **Fire condition:** port only **after** the ti1 race returns. If the race is falsified, a second
  per-pixel reweight from the same family is dominated and should not be built; if it pays, sn1's tensor
  is the next producer to move, through the same `seg_pixel_w` consumer.
* Not ported in this landing (one mechanism per commit; and porting a second reweight before the first
  is adjudicated would be building instead of measuring).

## Custody

* Code: `experiments/train_tr1_partition_renderer_mlx.py`, tests
  `src/tac/tests/test_ddm_tp2_tr1_form_predicate_and_margin_gate.py` (30 tests; existing TR1 suites
  unaffected). Verified against the **committed** HEAD in a clean detached worktree: 48 passed.
* Evidence read (READ-ONLY, untouched): `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/`,
  `/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/window_01/`,
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  `.omx/research/ddm_ti1_nonredundancy_probe_receipt_bins40_20260802.json`.
* All three commits staged via `--patch-file` intent-manifest mode: the trainer is a shared hot file
  carrying uncommitted sister hunks (`ddm_bs2` #871 `lane_guard_ratchet`). Each patch was replayed onto a
  clean `git show HEAD:` copy, AST-checked for duplicate top-level definitions, and asserted to contain
  zero sister tokens. Verified post-commit: **0 sister lines in all three diffs**, and bs2's work is
  preserved uncommitted in the working tree.
* No artifact created, moved, or deleted outside my own scratch; no other arm's run dir touched; no
  training run launched beyond n6 CPU/MLX smokes.
