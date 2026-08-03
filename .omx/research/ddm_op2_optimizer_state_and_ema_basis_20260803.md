# ddm_op2 — the two defects `ddm_gd5` owned-and-deferred are FIXED, and the fix is proven not to touch the live run

**Arm:** `ddm_op2`, closing `.omx/research/ddm_gd5_ds32_windows_20260803.md` §8.1b (i) and (ii)
(`010cb087fe`, `f695fac005`, `fa4b3724b0`, `a125c34b6c`).
**Axis:** `[macOS-CPU/MLX advisory]` for every measurement below.
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`.

---

## §0 POINTER HONESTY

**Exact contest pointer UNMOVED at `0.1910828242 [contest-CPU]`.** No archive built, no candidate
gated, no scorer slot taken, no paid dispatch. **This arm is APPARATUS, not a score claim** — it
buys epochs and repairs an instrument; it does not move S by itself.

LIVE BEST is `ddm_cx1`, and I recomputed it from components rather than reading the rounded
`Final score` field (which prints `0.83` for both `pj2` and `cx1` and cannot tell them apart):

```
seg 0.4311790 + pose 0.1597320 + rate 0.2355862 = S 0.8264972
rate cross-check: 25 * 353808 / 37545489        =   0.2355862   ✓
gap to the 0.172141 bar                          =   0.6543562
```

| deliverable | state |
|---|---|
| **OP2-1** optimizer state persists | **BUILT** (`--persist-optimizer-state`, args-only, DEFAULT OFF). MEASURED positive control: a restored boundary reproduces an uninterrupted run to **max abs param diff 0.0**; the reset path diverges **5.5e-2**. |
| **OP2-2** `ema_basis_held` | **FIXED** — now requires `parent_gate_basis == first_gate_basis` as well as the decay match, with both legs reported separately and an L1 runtime alarm on a basis switch. |
| **Two-landing rule** | **DONE** — `check_checkpoint_saves_do_not_silently_drop_optimizer_state`, STRICT from byte one at **live-count 0**, with a registered POSITIVE CONTROL (coverage ratchet 8 → 9; the uncovered ceiling does not move). |
| **Live run** | **UNDISTURBED, and proven so on real production bytes** (§3). Chain advanced 09 → 10 → 11 while I worked. |
| **gd4 resume guard** | **STILL REFUSES** the real `b4s` ds=16 ep945 checkpoint against the live ds=32 model, and the new opt payload goes THROUGH it, not around it (§4). |
| **Not mine** | one PRE-EXISTING red test, proven identical at HEAD, reported not papered over (§6). |

---

## §1 OP2-1 — WHAT WAS WRONG, AND WHAT IT COST

Six `save_checkpoint(...)` callsites in `experiments/train_tr1_partition_renderer_mlx.py` passed
the bare literal `opt_state_flat={}`. No checkpoint on disk therefore carried optimizer state, so
every `--resume-from` constructed a fresh `optim.Adam` with both moments zeroed — the
pre-registered `#824` reset-operator **arm B** (`what='both', to='zero', structure='uniform'`).

The trainer already shipped the price in its own `optimizer_arm` row —
`boundary_impulse_epochs_per_reset: 16.167` — and `ddm_gd5` §3.6 corroborated it from a
**completely different channel**: window_02's LIVE training signal jumped `1.912 → 14.846` across
the boundary and took ~17 epochs to return. Two independent routes to the same number.

At the MEASURED ~46 epochs per 30-minute window, a 666-epoch run pays that at ~13.5 boundaries:

| | epochs |
|---|---:|
| sealed budget | 666 |
| spent re-converging a deliberately reset Adam | **~218 (33%)** |
| **effective training** | **~448** |

`#824` scoped arms A/C out as *"a BUILD, not a port"* on the MEASURED ground that `opt_flat` had
one repo-wide hit that nothing read and nothing wrote. **That scoping was correct as a scoping of
that race, and it is preserved verbatim.** What changed is that gd5's measurement made the BUILD
load-bearing — and gd5's hypothesis, which I can neither confirm nor refute here, is that this is
plausibly why the incumbent lineage only ever existed as a continuation.

### The fix, and the one design decision inside it

`optimizer_state_to_flat` / `restore_optimizer_state` / `opt_state_param_path`, wired behind
`--persist-optimizer-state {off,on}`. Three things are worth stating because they are decisions,
not defaults:

1. **`learning_rate` is NOT restored.** MLX's `Optimizer.state` carries it, so a naive
   round-trip would make a window that changed `--lr` silently inherit the parent's — a
   silent-wrong of exactly the class the resume-geometry guard exists to refuse. The live
   `cfg.lr` wins; the difference is reported on the row.
2. **`optimizer.init(model.trainable_parameters())` is required before restore.** MLX creates
   per-parameter state lazily on the first `update`, so a freshly constructed optimizer has only
   the two scalars and there is nowhere for the moments to land. VERIFIED against mlx 0.31.2.
3. **A moment-free payload is REFUSED, not silently accepted.** Restoring `{step: ...}` alone
   would restore nothing while the run reported arm C — the fake this whole fix exists to close.

### The MEASURED positive control

Against the real `mlx.optimizers.Adam`, comparing actual trained parameters (not markers):

```
6 straight steps                                   -> reference
3 steps, snapshot, rebuild, RESTORE, 3 more steps   -> max abs diff  0.0
3 steps, snapshot, rebuild, RESET,   3 more steps   -> max abs diff  5.5e-2
```

**A boundary with persistence is a no-op on trained bytes.** That is the entire claim, and both
legs are asserted in the test — a restore that silently did nothing would pass the first and fail
the second, so neither can be faked alone.

---

## §2 OP2-2 — THE FLAG THAT CERTIFIED TWO DIFFERENT OBJECTS AS ONE

`boundary_jump_row` computed

```python
held = parent_ema_decay is not None and abs(parent_ema_decay - child_ema_decay) <= 1e-12
```

while the very same row separately recorded `parent_gate_basis` and `first_gate_basis`. **It had
the data and did not use it.** On the window_02 boundary it therefore stamped `ema_basis_held:
true` where the parent reading came off **LIVE weights** (`live_ema_warmup` — a fresh window's
`global_step` is below the EMA warmup) and the child's off the **EMA shadow** (`ema_shadow` —
every resumed window sets `global_step = ema_warmup_updates`), directly beside a
`boundary_dseg_delta` of **+0.496** (0.0157596 → 0.5118894, a 32× apparent collapse).

The magnitude is fully explained by the sealed decay and needs no training regression at all:
`ema_decay = 0.9999199` ⇒ time constant `1/(1−d) = 12,487 updates = 166.5 epochs`, so at ep46 the
shadow still carries `exp(−3450/12487) = 76%` weight on **initialization**.

**Matching decays are necessary and not sufficient. Two shadows of equal averaging length read
off different weight sets are still two different quantities.**

### The fix

`ema_basis_held` now requires **both** legs, and the row reports which one failed so no reader has
to re-derive it from the two basis strings:

| field | meaning |
|---|---|
| `ema_decay_held` | same averaging LENGTH (the original condition, unweakened) |
| `gate_basis_held` | same OBJECT (live weights vs EMA shadow) |
| `ema_basis_held` | **both** — the only flag that licenses a cross-boundary comparison |

Plus: **fail-closed on an unverifiable basis** (an absent basis field cannot *certify*
commensurability, only fail to), and an **L1 `confound_alarm` of kind `gate_basis_switch`** so the
switch is LOUD in telemetry rather than inferable only by a careful reader.

### A second-order finding I had to fix in my own edit

The same key name `ema_basis_held` is stamped at **three** sites, and the other two
(the resume `tlog` and the window receipt) run **before any gate has read anything** — they
*cannot* know the basis. Leaving them as-is would have reproduced the exact bug one level over:
one scalar name meaning two different things. They now carry `ema_decay_held` /
`gate_basis_held` / `held_scope` explicitly, and the **receipt** reports the strict verdict once
the first post-resume gate has observed the basis. I also caught, in round-1 self-review, that my
first version fed the receipt's `gate_basis_held` the *strict* (decay∧basis) value instead of the
basis leg — corrected before commit.

**What I did NOT do: soften the A1 thresholds.** gd5 §3.7 showed the window_02 `a1_realization_
gap_refuse` was a false positive built out of these two defects composing. Tuning the alarm would
have disabled a real confound guard to hide a real confound. The cause is fixed; the guard is
untouched.

---

## §3 THE LIVE RUN IS UNDISTURBED — PROVEN, NOT ASSERTED

This was the binding constraint, so it is measured on **real production bytes**, not argued:

```
P1  config_hash recomputed from window_09's checkpoint cfg
      == config_hash recorded inside that checkpoint            TRUE
    'persist' among TR1Config fields                            FALSE
P2  opt:: keys written with the flag OFF                        0
P3  real b4s ds=16 ep945 ckpt -> live ds=32 model               REFUSED (guard intact)
P4  real ds=32 stage_seg_trunk_tau_final -> ds=32 model         ACCEPTED, new_params=[]
```

**P1 is the load-bearing one.** The flag is threaded via `args` ONLY, never `TR1Config` — the
`--telemetry-v9-port` precedent, which the trainer states verbatim (*"never TR1Config =>
config_hash + every checkpoint stay flag-invariant"*). So `config_hash` is unchanged, the sealed
ticket still validates, `derive_ema_decay` is untouched, and the chain's `tr1_config.json`
byte-identity across windows — the thing that makes gd5's windows internally commensurable at all
— survives. **P2** closes it from the other side: with the flag off, `save_checkpoint` writes zero
`opt::` keys, so checkpoint bytes are identical to every pre-OP2-1 run.

**Why the default is OFF rather than ON, stated plainly:** a sealed chain is mid-flight, and
switching the reset arm underneath it would make its own windows incommensurable — the same class
of error as the `--epochs`-per-window drift `boundary_jump_row`'s docstring already indicts. It is
worth ~218 of 666 epochs, and it should be turned ON for the next **from-scratch** run, where the
whole chain runs one arm.

### Chain state while I worked (freshest artifact, mtime-checked — gd5's §5 lesson)

| window | ep | d_seg | × bar | Lane β₀ | gates | wall (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 09 | 359 | 0.01252605 | 2.108 | 434 | 10 | 30.14 |
| 10 | 409 | **0.01043913** | **1.757** | 437 | 10 | 30.56 |
| 11 | running (pid 52363) | — | — | — | — | — |

Two things gd5 could not yet see. **(a) The false-positive refuse has stopped firing** — windows
08/09/10 each ran a FULL ~30.5-minute window with 9–10 gates, against window_02's 13 epochs. The
boundary impulse shrinks as the run matures, exactly as gd5 predicted from window_03. **(b) The
falsifier is still closing**: 2.108× → 1.757× bar. Both readings are `ema_shadow` basis, so this
is a legitimate same-basis comparison under the very rule OP2-2 enforces.

Liveness was read as **telemetry row-growth + mtime** (window_11: 43 rows, mtime 8.5 s old), never
a pattern probe — `m50` and gd5 §3.8 between them record this instrument failing in **both**
directions.

---

## §4 THE PAYLOAD GOES THROUGH THE GUARD, NOT AROUND IT

`assert_resume_geometry_compatible` gained an optional `ckpt_opt_shapes` argument rather than a
new parallel check. Every `<param>.m` / `<param>.v` carries the parameter's own shape, so a ds=16
moment tree is exactly as inadmissible in a ds=32 model as the params are — and
`optimizer.state` assignment is the same silent-reshape surface `Module.update` is. Scalars
(`step`, `learning_rate`) are geometry-free and exempt. The default `None` reproduces the
pre-OP2-1 guard byte-for-byte, so no existing caller changes.

**The guard still refuses** (P3 above, on the real 14.3 MB `b4s` ds=16 ep945 checkpoint) — the
thing the brief asked me to verify, because a guard that stops refusing is worse than no guard.

---

## §5 THE SECOND LANDING

`check_checkpoint_saves_do_not_silently_drop_optimizer_state` (`src/tac/confound_gates.py`),
**STRICT from byte one at live-count 0**, AST-based, with a declared denominator
(`N file(s) mentioning opt_state_flat scanned, M keyword callsite(s) considered`).

It refuses the **bare literal** `opt_state_flat={}` only — the shape an omission actually takes.
Passing `{}` through a resolver or through `no_opt_state("<rationale>")` is fine. **The cure is
not "always persist"** (the default must stay off for byte-identity); it is that a callsite must
SAY which it is. `no_opt_state` returns the same empty mapping and rejects a placeholder reason.

Registered with a **POSITIVE CONTROL** carrying the exact pre-fix shape, so coverage ratchets
8 → 9 and the uncovered ceiling (17) does not move — landing a REFUSE-capable gate without a
control is the arithmetic `ddm_gh1`'s ceiling exists to close.

**One judgement call worth naming.** The gate initially fired on **7 test-fixture callsites** in
three sibling test files. The tempting fix was to exclude `src/tac/tests/` from the scan — which
is precisely the **narrowing/vacuity genus** the gate itself is built to guard against, and would
have shrunk the denominator to make a number look clean. Instead each file carries a stated
`# OPT_STATE_DROP_OK:` waiver ("fixture checkpoints, never resumed by a live run"), which is
exactly the state-your-intent discipline the gate enforces. Denominator preserved; live count 0
earned rather than arranged.

---

## §6 ONE PRE-EXISTING RED TEST — NOT MINE, AND NOT PAPERED OVER

`test_real_repo_live_count_bounded[check_levelset_hosc_requires_beta_end]` fails: live count **10**
against a bound of **9**. I touched no `launch.sh` and no hosc surface. **Proven by re-running
HEAD's OWN gate against the current tree: 10, identical.** Some sibling arm's `launch.sh` crossed
the bound. Raising the bound to make the suite green would be the wrong move; it is reported here
for whoever owns it.

The two failures that WERE mine (`CONFOUND_GATES` length 25 → 26 and a missing bounds entry) are
fixed, with the bound pinned at 0 and a comment saying not to raise it.

### An instrument catch of my own, and it nearly cost me the right answer

To test whether the hosc failure was pre-existing I loaded HEAD's `confound_gates.py` from a temp
directory and ran the gate. **It returned 0** — and 0 would have meant "my change caused the
failure." It was VACUOUS: `REPO_ROOT = Path(__file__).resolve().parent.parent.parent`, so a module
loaded from `/tmp/op2_head_pkg/` resolved its repo root to `/private` and scanned an empty tree.
An empty scan and a clean scan emit the identical symbol. Passing `repo_root=` explicitly gave
HEAD 10 and current 10 — the true answer, and the opposite verdict. This is `m50`'s genus arriving
in a third form (after gd5's `ps` truncation false-negative): **a probe whose denominator I did
not check nearly made me blame my own change.**

---

## §7 WHAT I DID NOT DO

* **No scorer slot, no n600 eval, no byte-close, no archive, no composition.** Every number here
  is `[macOS-CPU/MLX advisory]` and the pointer is untouched.
* **Did not turn `--persist-optimizer-state` ON**, and did not touch the live chain, its ticket,
  its driver, or its config. §3 is the proof that landing it mid-flight is safe *because* it is
  off, not a claim that flipping it mid-flight would be.
* **Did not soften the A1 thresholds** (§2) — the false positive is cured at its cause.
* **Did not run the B-vs-C A/B.** Arm C is now available; whether persisting moments improves
  **final d_seg** — as opposed to removing a loss re-convergence transient — is NOT settled by
  this landing, and `#824` is explicit that the reset is a studied arm rather than a defect.
  Claiming "fewer resets is better" from `ep_loss` alone would be the surrogate-for-authority
  substitution this campaign keeps paying for.
* **Did not fix the hosc bound** (§6) — not my defect.
* **Did not re-derive the 666-epoch budget or the ~450 effective-epoch scope.** gd5 pre-registered
  it; OP2-1 changes the arithmetic only for a run that turns the flag ON.

---

## §8 NEXT-IF-RESUMED

1. **Turn `--persist-optimizer-state on` for the next FROM-SCRATCH ds=32 run.** It is worth ~218
   of 666 epochs and it is now one flag. Do NOT flip it mid-chain: the current chain's windows are
   commensurable precisely because every one of them runs the same arm.
2. **Pre-registerable A/B, cheap and clean:** persist ON vs OFF from scratch, same seed, same
   schedule, read on **final d_seg** — not on `ep_loss` re-convergence. Pairs naturally with
   gd5 §8.4b's window-length A/B (`max_wall_minutes` is not in `TR1Config`, so neither perturbs
   `config_hash` or `ema_decay`). Both target the same quantity — boundary count × cost per
   boundary — so run them as a 2×2, and note `m52`: do not sample only the diagonal.
3. **Watch for `gate_basis_switch` alarms** in telemetry now that they exist. Every resumed
   window's first gate should fire exactly one on a live→shadow transition and none thereafter;
   more than one per window means something else is moving the basis.
4. **The falsifier still belongs on an n600 byte-closed reading**, per gd5 §3.4 — unchanged by
   this landing. The gate scalar is a 36-pair training-time steering signal, and OP2-2 makes it
   *comparable* across boundaries; it does not make it the bar.
5. **Still OWED from gd4/gd5, unclaimed by me:** the per-geometry launcher memory floor
   (`MEASURED_T2_PEAK_RSS_GIB` is a D16 constant standing in for every geometry) and a STRICT
   preflight sister gate for the resume-geometry class.
6. **§6's hosc bound** needs an owner.

**No candidate gated. No pointer moved. `score_claim=false` on every row above.**
