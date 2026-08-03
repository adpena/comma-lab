# ddm_gd5 — the ds=32 run under supervision: window_01 MEASURED, the chain is governed, and the falsifier cannot legally be adjudicated on the scalar everyone has been quoting

**Arm:** `ddm_gd5`, supervising the run `ddm_gd4` sealed (`.omx/research/ddm_gd4_ds32_training_seal_20260802.md`, `c6a83481f6`).
**Axis:** `[macOS-CPU/MLX advisory — real trainer, real n600 gt cache, real DSL-compiled ticket, real governed launcher]`.
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`.

---

## §0 POINTER HONESTY + THE ANSWER

**Exact contest pointer UNMOVED at `0.1910828242 [contest-CPU]`. No archive built, no candidate
gated, no scorer slot taken, no paid dispatch. LIVE BEST is unchanged at `pj2` S = 0.8308905.**

| deliverable | state |
|---|---|
| **GD5-1 windows** | **window_01 COMPLETE + ADJUDICATED; windows 02–03 fired; chain driver landed, self-tested, running (pid 29389).** 46 epochs / 30 min = **39.1 s/epoch**. **But window_02 delivered only 13 epochs — the trainer refused itself (§3.7), and that refuse is a false positive built out of §3.5 + §3.6.** |
| **GD5-1 budget** | **DERIVED, and the answer is "hold 666" — for a reason nobody had named.** `epochs` has exactly one load-bearing consumer beyond the loop bound, and changing it mid-chain is a *known* confound the trainer already documents (§2). **But the budget buys less than it says: ~218 of the 666 epochs (33%) are spent re-converging a deliberately reset Adam at window boundaries — §3.6.** |
| **GD5-2 falsifier** | **REACHABLE, not yet met.** ep44 d_seg **0.0157596 = 2.65× the bar**, `COUPLED_DESCENT`, descent *accelerating* (−12.45% last gate). **Two findings say the scalar everyone quotes cannot carry the verdict: §3.4 (the bar and the gate are different populations) and §3.5 (the gate silently changes which object it reads at every resume).** |
| **GD5-3 guard** | **VERIFIED ON PRODUCTION BYTES.** It REFUSED a real `b4s` ds=16 ep945 checkpoint against the live ds=32 model, and ACCEPTED the legitimate ds=32 resume with `new_params=[]`. First refusal seen on real bytes rather than a fixture. §4. |
| **GD5-4 endpoint** | Not reached. Pose re-solve NOT run (needs the scorer slot; `ddm_cx1` holds it). §6. |
| **Instrument catches** | **Two, both of the "stale view" genus, both caught by measuring rather than believing.** §5. |

---

## §1 WHAT window_01 ACTUALLY MEASURED

Clean exit: `stop_reason: max_wall_minutes`, `epochs_ran: 46`, 9 gates, `weights_stepped: true`
on every epoch, no `ep_loss == 0.0`, no frozen-epoch or term-domination alarm.

`realized_gate_dseg_mean` and its **exact per-GT-class partition** (`ddm_bs3` #909 — the partition
sums to the scalar; I checked, it does):

| ep | d_seg | class | seg form | Road | Lane | Undriv | Movable | MyCar | Lane β₀ | tok B |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0335213 | FIRST_GATE | ce | 0.003987 | **0.006050** | 0.008242 | 0.013765 | 0.001477 | **0** | 45279 |
| 9 | 0.0239330 | COUPLED | ce | 0.004106 | **0.006050** | 0.004204 | 0.007946 | 0.001626 | 2 | 46177 |
| 14 | 0.0222483 | COUPLED | ce | 0.002824 | **0.006050** | 0.004739 | 0.007324 | 0.001311 | 0 | 47027 |
| 19 | 0.0207853 | COUPLED | ce | 0.003396 | **0.006050** | 0.003522 | 0.006635 | 0.001182 | 0 | 47368 |
| 24 | 0.0204215 | FIRST_GATE | **tau** | 0.003971 | 0.005710 | 0.003456 | 0.006114 | 0.001170 | 78 | 48594 |
| 29 | 0.0217395 | **ALARM** | tau | 0.006142 | 0.005443 | 0.003978 | 0.004843 | 0.001333 | 162 | 49087 |
| 34 | 0.0199215 | COUPLED | tau | 0.009115 | 0.005011 | 0.001999 | 0.002934 | 0.000864 | 237 | 49632 |
| 39 | 0.0179999 | COUPLED | tau | — | — | — | — | — | — | 50xxx |
| 44 | **0.0157596** | COUPLED | tau | 0.005332 | 0.004590 | 0.002200 | 0.002455 | 0.001184 | **313** | 51151 |

Three things are worth stating precisely, because two of them are traps I walked into.

**(a) The CE stage cannot birth Lane at all, and while it doesn't, Lane's contribution is a
constant — which I can now name exactly.** Across ep4–19 the Lane entry is
`0.006049968578197337`, **bit-identical to 16 digits**. That is not a coincidence and not a stuck
counter: with zero GT-lane recall, the class's error contribution collapses to the class's own GT
area fraction, which is a property of the ground truth alone. MEASURED confirmation — the 36-pair
gate set's GT Lane area fraction is **0.0060500**, matching the observed constant to **0.001%**.
So "Lane fully erased" has an exactly-known price on this instrument, and it is `0.0060500`.

**(b) The `tau_softplus` knee is what births Lane.** β₀ goes `0 → 78 → 162 → 237 → 313` of 985 GT
components across ep24–44, and Lane's contribution falls `0.006050 → 0.004590` (−24%). The
incumbent ds=16 lineage carries 500–576 born components at its endpoint, so ds=32 at ep44 is
~55–63% of the way to the incumbent's Lane topology, 44 epochs into a 666-epoch budget.

**(c) The one ALARM was a stage transient, and my first read of it was wrong.** At ep29 the run
fired `A1_REALIZATION_GAP_ALARM` and d_seg *rose*. Reading ep19→29→34 I wrote that Road was
"rising monotonically" (0.003396 → 0.006142 → 0.009115) and was about to record a bad Road↔Lane
trade as a structural finding — the exact Road-hub exchange `m91` predicts. **ep44 refutes it:
Road fell back to 0.005332.** The Road excursion is a knee transient (the CE→tau form switch plus
the `at_knee` STE engagement land in the same window), not a trade. Three points spanning a stage
boundary are not a trend; I had them and still nearly shipped the wrong mechanism.

---

## §2 GD5-1 — THE EPOCH BUDGET, DERIVED

`ddm_gd4` §7 correctly flagged 666 as inherited: the incumbent's `window_01` was a *continuation*
(`r1c` starts at ep504), so its epoch count is a borrowed constant for a from-scratch run. I traced
what the constant actually does. `cfg.epochs` has **exactly five consumers**, and only one is
load-bearing:

| consumer | line | effect |
|---|---|---|
| `total_updates = epochs × steps_per_epoch` → `derive_ema_decay` | 2003 | **LOAD-BEARING** — 666×75 = U=49,950 ⇒ `ema_decay = 0.9999199` |
| loop bound `range(start_epoch, cfg.epochs)` | 2608 | how long the run goes |
| F2 knee midpoint fallback `epoch >= cfg.epochs // 2` | 2692 | would fire at ep333 — **did not fire**; the knee fired on the EVENT at ~ep20 |
| final-epoch gate | 2735 | cosmetic |
| config record | 2014 | provenance |

So the budget's real content is **the EMA horizon**, and CLAUDE.md makes the EMA shadow the
inference checkpoint — it is not a scheduling detail, it is the artifact.

**And the trainer already knows changing it mid-chain is a confound.** `boundary_jump_row`
(lines 1414–1436) exists precisely for this and says so verbatim:

> *"if the parent and child resolved DIFFERENT `ema_decay` the shadow's own averaging length moved
> underneath the measurement and the two readings are not commensurable … **the burn ran
> U=49,950/60,450/70,950 ⇒ a different decay at EVERY boundary**."*

That parenthetical is a measured indictment of the incumbent burn: **it changed `--epochs` at every
window boundary, so its own cross-window d_seg readings are not strictly commensurable.**

**Verdict — hold 666 for the whole chain, and here is the honest accounting:**

1. It is an **inherited** number. I am not claiming it was derived for a from-scratch ds=32 run.
2. Holding it fixed buys something the incumbent never had: window_02's resume row reports
   `ema_basis_held: true`, `parent_ema_decay == child_ema_decay == 0.9999199199199199`, and the two
   `tr1_config.json` files are **byte-identical**. Our window chain is internally commensurable.
3. Re-deriving it now costs window_01 — a new budget changes `ema_decay`, which changes the trained
   artifact, so it is a from-scratch restart, not a parameter tweak.
4. **Directionally 666 is more likely too SHORT than too long**: the incumbent lineage reached
   **ep945**, and ours is from scratch at a 4× coarser description. **Therefore an endpoint that
   misses the falsifier carries a live alternative explanation — "under-trained at 666" — and the
   verdict must be scoped to it, not reported as "ds=32 is worse."** Pre-registering that scope now,
   before the number exists, is the point.

`ddm_tl1`'s surface (`tools/ddm_tl1_scored_dims_reductions.py`) is a **scored-dimension** sizing
tool (pose `rank(J) ≤ 6`, seg head affine rank-4) — it bounds how many dimensions the scorer can
see, not how many epochs SGD needs to reach them. It does not size an epoch budget, and I am not
going to pretend it does.

---

## §3 GD5-2 — THE FALSIFIER WATCH

### 3.1 Where the run stands

Pre-registered bar (`ddm_gd4` §6, `any` mask, **not softened here**):
**endpoint realized `Δd_seg` above base `0.00431179` must be `< 1.633e-03`**, i.e. **d_seg < 0.00594179**.

ep44 = **0.0157596 = 2.65× the bar.**

### 3.2 Is it reachable? — yes, on the evidence so far

Per-gate relative drops over the tau stage: **−1.75%, +6.45% (the knee transient), −8.36%, −9.65%,
−12.45%.** Descent is *accelerating*, not plateauing. Held at the last observed 12.45%/gate, the
2.65× closes in ~9 gates (~ep90); at a heavily decayed 3%/gate it closes in ~33 gates (~ep210).
Both sit well inside the 666 budget. **Nothing here supports an early stop**, and stopping at ep44
— 6.6% of the budget, mid-descent — would have been the cheap wrong call.

### 3.3 Confound alarms — all clean in window_01

`weights_stepped: true` on all 46 epochs · no `ep_loss == 0.0` · no term domination
(`sum_minus_total: 0.0`, terms seg 2.213 / rate 0.166 / delta_sparsity 0.0) · `accepted_frac: 1.0`
(zero skips) · gnorm 2.9–4.9 against no clip event · the positive-control canary passed at ds=32.
One `A1_REALIZATION_GAP_ALARM` at ep29, adjudicated in §1(c) as a stage transient and cleared by
ep34/39/44 `COUPLED_DESCENT`.

### 3.4 **THE FINDING: the bar and the scalar are two different populations**

`realized_gate_dseg_mean` is computed on a **pre-registered 36-pair gate set**
(`resolve_gate_ids`: block 447–450 + 32 `rng(0)`-sampled off-block pairs). The falsifier's base
`0.00431179` is `dc1_fold`'s **n600 archive-level** d_seg. **These are not the same population**, and
the trainer's own gate set is measurably skewed against n600:

| class | n600 area frac | 36-pair gate area frac | gate / n600 |
|---|---:|---:|---:|
| Road | 0.2323324 | 0.2275796 | 0.980 |
| **Lane** | 0.0058546 | 0.0060500 | **1.033** |
| Undrivable | 0.4951755 | 0.4948815 | 0.999 |
| **Movable** | 0.0123793 | 0.0173882 | **1.405** |
| MyCar | 0.2542581 | 0.2541007 | 0.999 |

**Movable is 40.5% over-represented and Lane 3.3% over-represented in the gate set.** Per `m88`
(a prefix/subset of a skewed population is a *different* population — the 5.1× `bp2` incident),
a ratio ≠ 1 on the governing quantity is the whole test, and this fails it on two classes.

The sign of this actually **flips a claim** I was one step from making. While Lane was fully erased
I had "Lane's contribution alone (0.0060500) exceeds the entire bar (0.0059418), so the run cannot
pass unless Lane births" — true **on the gate set** (+1.8%). On n600 the Lane area fraction is
**0.0058546**, which is **1.5% BELOW the bar**. Same physical statement, opposite verdict,
purely from which population you evaluate it on.

**Consequence, binding on GD5-4 and on anyone who inherits this run:** the endpoint falsifier
**must be adjudicated on an n600 byte-closed reading**, never on `realized_gate_dseg_mean`. The gate
scalar is a training-time steering signal on a deliberately hard 36-pair instrument; it is not the
quantity the bar was written against. I have not attempted to estimate a correction factor between
them — that would be a surrogate standing in for the measurement, which is the thing to avoid.

### 3.5 **SECOND FINDING: the gate silently changes WHICH OBJECT it reads at every resume — and `ema_basis_held` certifies it as commensurable anyway**

window_02's first gate read **d_seg 0.5118894** against window_01's ep44 **0.0157596** — a 32×
apparent collapse, five epochs later, on an identical config. It is **not** a training regression.
The trainer's own `boundary_jump` row states the mechanism:

```json
{"parent_gate_epoch": 44, "parent_gate_dseg": 0.01575964, "parent_gate_basis": "live_ema_warmup",
 "first_gate_epoch": 49, "first_gate_dseg": 0.51188942, "first_gate_basis": "ema_shadow",
 "gate_basis_mode": "resumed_warm_shadow", "boundary_dseg_delta": 0.49612978,
 "ema_basis_held": true,
 "caveat": "... commensurable with the parent reading ONLY when ema_basis_held is true"}
```

A from-scratch window gates on **LIVE weights** (`global_step` below the EMA warmup); every
**resumed** window sets `global_step = ema_warmup_updates` ("resume ⇒ warm shadow") and gates on the
**EMA shadow**. Two different objects, one scalar name.

**The magnitude is fully explained by the sealed decay, and this is §2's inherited constant coming
due.** `ema_decay = 0.9999199` ⇒ time constant `1/(1−d) = 12,487 updates = **166.5 epochs**`. At
ep46 the run has completed `3,450/12,487 = 0.276` time constants, so the shadow still carries
`exp(−0.276) = **76%** weight on initialization`. A shadow that is three-quarters init reading
d_seg ≈ 0.51 is exactly right. The sealed 666-epoch budget is only **4.0 time constants** long.

**The apparatus gap.** `ema_basis_held` checks only that the decay *value* matched
(`parent_ema_decay == child_ema_decay`). It does **not** check
`parent_gate_basis == first_gate_basis`. So the row carries both basis fields, has everything it
needs to detect the mismatch, and instead stamps `true` — certifying as "commensurable" a pair of
readings taken from different objects, directly beside a `boundary_dseg_delta` of +0.496. This is
the L1 "instrument reads the wrong quantity" class the trainer's own
`a1_smooth_excluding_delta_penalty` docstring was written about, recurring one level up.

**Fix (exact, one condition):** `ema_basis_held` must additionally require
`parent_gate_basis == first_gate_basis`, with a test that a live→shadow boundary reports
`False`. **Deferred, not orphaned — owner `ddm_gd5`.** Blocker: `boundary_jump_row` runs at the
first gate of every window, so an untested edit to it mid-chain would crash all ~13 remaining
windows of a live 7 h run. Fire condition: the chain reaching its stop or its endpoint; then I land
it with the test.

**Consequences, which are the load-bearing part:**

1. **d_seg is NOT comparable across a differing `gate_basis`.** window_01's trajectory
   (0.0335 → 0.0158, live) and window_02+'s readings (shadow) are separate series.
   `tools/ddm_gd5_ds32_window_chain.sh` now records `gate_basis` per ledger row and its stop rules
   only compare **same-basis** windows.
2. **The shadow series will fall fast for ~100+ epochs from EMA convergence alone, not learning.**
   That is a confound in the *flattering* direction — the more dangerous one — and it is why the
   plateau rule is same-basis-scoped rather than tuned.
3. **This constrains the endpoint.** CLAUDE.md makes the EMA shadow the inference checkpoint, so the
   shipped artifact is the shadow. At the sealed ep666 the shadow is 4.0 time constants in
   (init weight `e⁻⁴` ≈ 1.8%) and is sound. **A run stopped early would ship a shadow still
   contaminated by init, and its byte-closed d_seg would be far worse than the live weights
   suggest.** Any early stop must therefore be reported on the live weights *and* explicitly
   flagged as not-shippable-as-is — not read off the shadow gate.

### 3.6 **THIRD FINDING: the window boundary costs ~16 epochs of re-convergence, so the sealed 666-epoch budget buys only ~450 effective epochs**

§3.5's basis switch is a *readout* artifact. This one is not. `ep_loss` contains no EMA at all — it
is the live-weight training signal — and it jumped across the boundary:

```
window_01  ep44 1.840019   ep45 1.912496          <- 46 epochs of descent
window_02  ep47 14.845633  <- 7.8x JUMP on the LIVE signal
           ep49  5.262828   ep54 2.856726
           ep59  2.267794  <- 13 epochs later, still ABOVE window_01's ep45
```

**Cause, MEASURED, and it is designed behaviour rather than a bug.** No checkpoint on disk carries
optimizer state: `stage_seg_trunk_tau_final.npz` holds `{param: 23, ema: 23, meta: 2}` and **zero
`opt::` keys**, because **all six `save_checkpoint` callsites pass `opt_state_flat={}`**. Every
resume therefore constructs a fresh `optim.Adam` with both moments zeroed. `reset_arm_for` names
this exactly — it is the pre-registered `#824` **reset-operator arm B**
(`what='both', to='zero', structure='uniform'`, `bias_correction=False`), and the trainer's
`optimizer_arm` telemetry row even ships the cost:

```json
{"arm": "B", "boundary_impulse_epochs_per_reset": 16.167649176608975, "requires_persistence": false,
 "note": "arm B (bias_correction False) IS MLX's Adam default => trained bytes identical to every
          pre-#824 run; arm B' removes the eta(t) reset impulse"}
```

**My measurement corroborates the prediction independently:** window_02 needs ~17 epochs to return
to window_01's ep45 loss (2.268 at ep59, falling ~4%/epoch), against the predicted **16.17**.

**The consequence for the budget (this is the part that is new).** At the MEASURED 46 epochs per
30-minute window, a 666-epoch run is ~14.5 windows ⇒ **~13.5 boundaries × 16.17 = ~218 epochs spent
re-converging Adam**. The loop still terminates at `cfg.epochs = 666`, so that time is not added —
**it is taken out of the budget**:

| | epochs |
|---|---:|
| sealed budget | 666 |
| spent re-converging Adam at ~13.5 boundaries | **~218 (33%)** |
| **effective training** | **~448** |

So §2's "666 is more likely too short than too long" is stronger than I could show there: the
inherited budget, executed in 30-minute windows, delivers roughly **450 effective epochs** against
an incumbent lineage that reached **945**. Any endpoint miss must be scoped against ~450, not 666.

**The obvious lever is longer windows** — halving the boundary count would return ~110 epochs to
training at zero risk on memory (15.0 GiB peak vs 74–96 GiB free), and `max_wall_minutes` is **not**
in `TR1Config`, so it changes neither `config_hash` nor `ema_decay` and would not break window
comparability. **I did not pull it, and the reason matters:** `#824` treats the reset as a *studied
arm*, not a defect — arm B′ exists specifically to remove the `eta(t)` reset impulse — so whether
the boundary impulse costs or helps **final d_seg** is not settled by a loss-re-convergence
measurement. Claiming "fewer resets is better" from `ep_loss` alone would be exactly the
surrogate-for-authority substitution this campaign keeps paying for. It is a clean, cheap,
pre-registerable A/B for whoever owns the next ds=32 run.

### 3.7 The three findings COMPOSE into a false-positive that stops windows early — the trainer refuses itself

**window_02 did not hit the wall clock. The trainer stopped it: `stop_reason:
a1_realization_gap_refuse`, at ep59 — 13 epochs into a 46-epoch window.**

The cascade is §3.6 → §3.5 → the A1 predicate, and every link is measured:

1. Adam's moments are zeroed at the boundary (§3.6) ⇒ the live weights take a large excursion
   (`ep_loss` 1.912 → 14.846) and then re-converge fast (→ 2.268 in 13 epochs).
2. The EMA shadow **absorbs that excursion**, so shadow-basis d_seg *rises*:
   0.5119 → 0.6134 → 0.7294, with Lane falling back to fully erased (985/985).
3. The A1 gate reads the **shadow** on a resumed window (§3.5) while its smooth input is `ep_loss`
   on the **live** weights. Its alarm predicate is *"smooth fell ≥ threshold while realized fell
   < threshold"* — which is **exactly** what a re-converging optimizer plus a lagging shadow
   produces. Two consecutive alarms ⇒ `a1_realization_gap_refuse`.

So the refuse is a **FALSE POSITIVE**: the instrument is built to catch "the loss improves but the
argmax doesn't", and at every resume boundary it instead catches "Adam is recovering while the
shadow lags". It is the same L1 *instrument-reads-the-wrong-quantity* class the trainer's own
`a1_smooth_excluding_delta_penalty` docstring documents — the smooth and realized channels are
being read off **different weight sets**.

**Degraded, not dead — and the evidence is window_03.** Its first gate reads **0.275569**, down
from window_02's 0.729438, and its boundary impulse was far smaller (`ep_loss` 2.268 → ~2.9 vs
1.912 → 14.846). The shadow is catching up and the impulse shrinks as the run matures. But the tax
is real: window_02 delivered 13 epochs where window_01 delivered 46.

**Verdict for the chain: keep running, do not re-tune.** The driver treats a refuse-exit as a normal
window exit and continues; the run is recovering; and the honest fix is upstream (§3.5's basis
check, §3.6's optimizer persistence), not a threshold tweak on a gate that is reading the wrong
object. Softening the A1 thresholds to stop the false positives would disable a real confound guard
to hide a real confound.

### 3.8 A fourth instance of the probe genus — this one mine, and it caused a duplicate launch

`ps -axo pid,command | grep -F 'ddm_gd5_ds32_window_chain.sh'` returned **empty** while the driver
was alive (pid 29389, 14 min elapsed), so I concluded it had died and relaunched — briefly running
**two** chain drivers. Reproduced cleanly on the trainer as well:

```
ps -axo pid,command | grep -c 'train_tr1_partition_renderer_mlx.py'   ->  0
pgrep -f              'train_tr1_partition_renderer_mlx.py' | wc -l   ->  1     (it IS running)
```

`ps -axo command` truncates to terminal width, so a pattern living deep in a long argv is invisible.
`m50` records this instrument over-matching (a loose pattern counts your own monitors); this is the
**same instrument under-matching** — a false negative on a real process. Both directions come from
text-scanning a *truncated view* instead of issuing an anchored query. **`ps -p <pid>` and
`pgrep -f` were correct throughout; the grep-over-`ps` was not.** The duplicate was removed; one
driver (29389) and one trainer (40440) remain. The driver itself never used this pattern — it polls
`ps -p "$PID"` on the exact pid — so the chain was never at risk from it; only my manual check was.

**Root cause of the relaunch:** the first driver was started as `nohup bash script &` **without
`< /dev/null`**, which CLAUDE.md's detach Pattern A requires. It survived, but the missing stdin
redirect is a real defect in how I launched it; the relaunch used the full Pattern A form.

---

## §4 GD5-3 — THE RESUME GUARD, TESTED ON REAL BYTES

`ddm_gd4` built `assert_resume_geometry_compatible` after MEASURING that `mlx.nn.Module.update`
silently assigns wrong-shaped arrays, with **672 real ds=16 checkpoints on disk**. A guard nobody
has seen refuse is an untested guard, so I ran it against production bytes — not a fixture:

```
live model = ds=32 (tokens_base (12,16,4))

ds32 ckpt  window_01/checkpoints/stage_seg_trunk_tau_final.npz
    ACCEPTED   new_params_since_ckpt=[]

ds16 ckpt  ddm_b4s_20260731/window_03/checkpoints/intra_seg_trunk_tau_ep00945.npz
    REFUSED    ResumeGeometryMismatch
    shape conflicts: ['tokens_base: ckpt(24,32,4) != model(12,16,4)',
                      'tokens_delta: ckpt(600,24,32,4) != model(600,12,16,4)']
```

The hazard is confirmed real in the same breath: the ds=16 checkpoint has **no `up4` params at
all** (`s_up4/g_up4/b_up4` absent), so absent the shape conflict the existing `backfilled` path
would have absorbed them as "new params since the checkpoint" and logged a truthful-looking
`ema_backfilled_new_params` line — a clean-looking warm start onto a silently wrong geometry,
exactly as gd4 predicted.

**Every continuation in this chain resumes by SHAPE, never by filename.** The driver's picker
requires `tokens_base == (12,16,4)` before a file is eligible. window_02's actual resume row:
`resume_from … stage_seg_trunk_tau_final.npz`, `epoch: 47`, `ema_backfilled_new_params: []`,
`ema_basis_held: true`, `quant_engaged: true`.

---

## §5 TWO STALE-INSTRUMENT CATCHES

Both are the same genus and both were caught only by measuring the live artifact.

**(1) `run.log` lags `telemetry.jsonl`.** `run.log` carried gates only to ep29 — whose last reading
was the ALARM with d_seg *rising* and Lane at 0 born. Working from it I had assembled a
stop-early case ("Lane-erasure floor exceeds the bar; unreachable"). `telemetry.jsonl` already held
ep34 with **Lane at 237 components born and d_seg falling**. The stop-early verdict was wrong and
the file I had reached for first was the reason. **`telemetry.jsonl` is the authority; `run.log` is
a lagged tail.**

**(2) The brief said window_01 "has already exited cleanly"; it had not.** My window_02 dry-run was
REFUSED by G4 (scorer slot busy, pid 79056). The tempting reads were "stale ps" or "governor bug".
Measured instead: `ps -p 79056` ⇒ **alive, `ELAPSED 29:31`, RSS 14.5 GiB** — 29 minutes into a
30-minute cap, and its RSS independently reproduces gd4's certified 15.014 GiB peak.
**The governor was right and the brief was stale.** A REFUSE is information. It also would have
been trivially easy to "fix" this by reaching for a force flag and firing two n600 trainers at once.
Waiting cost 90 seconds.

---

## §6 THE CHAIN DRIVER

`tools/ddm_gd5_ds32_window_chain.sh` — fires continuation windows **through the governed launcher**
(never around it), so G1 seal-freshness / G2 import custody / G3 memory / G4 slot / G5
detached+receipted adjudicate **every** window, not just the first.

* **Resume point by TENSOR SHAPE**, ordered by **write time** — not by the epoch digits in the
  filename. This is a bug I wrote and caught in self-test: `stage_seg_trunk_tau_final.npz` (the
  terminal on-exit save, EMA shadow inside, written *last*) carries no epoch digits, so digit
  sorting would have silently resumed from `ep00039` and discarded epochs 40–46.
* **A launcher `rc=4` REFUSE is waited on, never forced** (up to ~60 min), and the driver **never
  kills anything**.
* **Liveness anchored on the exact pid**, never a loose `pgrep` pattern (`m50` — three
  self-matching-probe incidents on record).
* Appends a ledger row per window (`gd5_chain_ledger.jsonl`) with d_seg, the per-class partition,
  Lane β₀, bytes, and the liveness flags, each stamped `score_claim=false`.
* **Bounded stop rules**, which do not touch the falsifier: STOP on a frozen-epoch /
  `weights_stepped` alarm; STOP on plateau-above-bar (3 consecutive windows each < 1% relative
  d_seg drop); BAR_MET when d_seg ≤ 0.00594179 — subject to §3.4, which says the *real*
  adjudication is n600 byte-closed, so BAR_MET is a hand-off trigger, not a verdict.
* `GD5_SELF_TEST=1` exercises both helpers against the live tree without launching.

Fired detached. window_02 = pid 29425, all gates PASS, resume verified.

---

## §7 WHAT I DID NOT DO

* **No scorer-slot work.** `ddm_cx1` holds it. No n600 eval, no `ddm_v4d_resolve --mode refine`,
  no byte-close, no composition. Every number here is `[macOS-CPU/MLX advisory]`.
* **No endpoint verdict on the falsifier.** ep44 is 6.6% of the budget.
* **No correction factor between the 36-pair gate and n600** (§3.4) — that would be a surrogate
  standing in for the measurement the verdict actually needs.
* **Did not re-derive the epoch budget** (§2) — I state plainly that 666 is inherited, name its one
  load-bearing consumer, and pre-register the scope limit that follows.
* **Did not land the OWED launcher memory re-anchor** (gd4 §8.1: `MEASURED_T2_PEAK_RSS_GIB` is a D16
  constant standing in for every geometry; the ds=32 floor should be 30.0 GiB not 25.6 GiB). Still
  passing on a 74–96 GiB-free host; still the borrowed-constant genus. Unclaimed.

---

## §8 NEXT-IF-RESUMED

1. **Watch the chain.** `gd5_chain_ledger.jsonl` (one row per window, now carrying `gate_basis`)
   and `gd5_chain_driver.log`. Driver pid 29389, trainer pid 40440 at hand-off. The driver stops
   itself on plateau/liveness alarms and waits on a governor REFUSE rather than forcing.
   **Expect short windows** until §3.7's false-positive refuse is addressed upstream.
1b. **The two upstream fixes, in priority order, both OWNED by `ddm_gd5` and deferred only behind
   the live chain:** (i) `ema_basis_held` must also require `parent_gate_basis == first_gate_basis`
   (§3.5) — one condition plus a test; (ii) persist optimizer state (§3.6) — `#824` scoped arms A/C
   out because `opt_flat` has one repo-wide hit and nothing reads or writes it, calling C "a BUILD,
   not a port". That BUILD is now load-bearing: without it every windowed **from-scratch** run pays
   ~16 epochs and a false refuse per boundary, which is plausibly why the incumbent lineage only
   ever existed as a continuation.
2. **Adjudicate the falsifier on n600 byte-closed, NOT on the gate scalar — for two independent
   reasons.** (§3.4) the bar `d_seg < 0.00594179` was written against an n600 archive-level base
   while `realized_gate_dseg_mean` is a 36-pair reading on a set 40.5% heavier in Movable and 3.3%
   heavier in Lane; and (§3.5) that scalar is taken from **live weights** in window_01 and from the
   **EMA shadow** in every resumed window, so it is not even one series. Byte-close the endpoint and
   read d_seg from the real receiver.
2b. **Do not read the window_02+ shadow series as learning.** It will fall steeply for ~100+ epochs
   purely as the EMA converges (166.5-epoch time constant, 76% init weight at ep46). Compare
   same-basis only; the ledger now carries `gate_basis` on every row.
3. **Then the pose re-solve** — `ddm_v4d_resolve --mode refine`, which takes the n600 scorer slot,
   so sequence it through MAIN against `ddm_cx1`/`ddm_pj2`. The token change makes the fitted
   carrier stale by construction (`f0 := a·warp(f1)+b` with `(a,b)`/`s_t`/selector/`beta` all fitted
   to the old decoded frame_1). Interrogate `_refine_dim0`'s `±0.048/±0.006` bounds; **do not**
   chase `#850`'s GN cap — `ddm_pw1` measured that path dead on this vehicle.
4. **Scope any miss.** If the endpoint misses the bar, "under-trained" is a live alternative to
   "ds=32 is worse", and §3.6 sharpens it: the sealed 666 delivers only **~450 effective epochs**
   after ~218 are spent re-converging Adam at ~13.5 boundaries, against an incumbent lineage that
   reached **945**. Scope the verdict against ~450, not 666. §2 + §3.6 pre-register this.
4b. **Pre-registerable A/B for the next ds=32 run:** window length. `max_wall_minutes` is not in
   `TR1Config` (no `config_hash` / `ema_decay` change), memory has 5–6× headroom, and halving the
   boundary count returns ~110 epochs to training. Not pulled here because `#824` treats the reset
   as a studied arm (B vs B′) whose effect on **final d_seg** — not on `ep_loss` re-convergence — is
   unsettled.
5. **Still OWED from gd4:** the per-geometry launcher memory floor, and a STRICT preflight sister
   gate for the resume-geometry class (the runtime guard is now verified on production bytes; the
   static gate is the second landing CLAUDE.md's two-landing rule asks for).

**No candidate gated. No pointer moved. `score_claim=false` on every row above.**
