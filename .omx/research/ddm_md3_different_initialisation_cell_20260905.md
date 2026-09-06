---
title: "The born vehicle has no init seed to flip — build_initial_state copies r10's EMA shadow verbatim, so the question was answered a cheaper way: the step-0 wrong POOL moves enormously across eight legitimately-produced starts (Jaccard 0.99 → 0.12) while the UNREACHABLE sites inside it do not — over the four comparable-quality starts 83.13% of unreachable sites are wrong at ALL of them against 19.78% of the sites the optimizer DID reach (4.20×, reproducing 4.21× on the second lineage); and the data-order control lands PERSISTENT 61.606% with J 0.8536, a fourth instance within 1.35 pp that moves the sites LESS than the two-lever schedule does"
arm: ddm_md3
charter: .omx/research/charters/ddm_md3_different_initialisation_cell_20260905.md
parent_arms: [ddm_md1, ddm_md2, ddm_ng5, ddm_qbr1, ddm_gs4]
utc: 2026-09-05T14:40:00Z
verdict_scope: "[macOS-CPU advisory . step-0 argmax reconstructed by md1's instrument UNCHANGED from retained starting states . frozen CPU-torch SegNet+PoseNet . QBF1-born vehicle . n32 sealed selection . eight born-gate-eligible starts, ALL descending from root init seed 20260827 . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval by this arm]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_md3 — is the born vehicle's unreachable error set fixed by its START?

## VERDICT: **DATA-ANCHORED**, at FORMULATION scope, on the evidence this arm could buy for $0

The charter's falsifier is stated on a burned different-init cell's persistent-set Jaccard
(`J >= 0.70` → DATA-ANCHORED). **That cell is sealed and armed but has not fired** — the Metal was
held by cl2's λ ladder for this arm's whole window — so the pre-registered number is OWED and this
verdict rests on the two measurements this arm could buy for $0. The verdict word carries that
limit, and §7 says exactly what would overturn it.

**Three things are settled, all $0:**

1. **The data-order control (§5) removes the confound the charter asked to remove first.**
   PERSISTENT **61.606%** on the shadow — a fourth instance, all four within 1.35 pp — and the
   sites overlap the cold control's at **J 0.8536**, *higher* than ng5's two-lever schedule
   (0.8069). Varying the order the pairs arrive in perturbs the unreachable set **less** than
   changing the whole schedule does, and neither moves it.
2. **The step-0 wrong pool is NOT fixed** (§2). Across eight legitimately-produced starts it moves
   from J 0.99 to J 0.12. md2's bit-identical pools were an artifact of every cell declaring one
   `initial_state` file, not a property of the vehicle.
3. **The unreachable sites inside that moving pool ARE fixed** (§3). Over the four starts whose
   quality is comparable to the incumbent's, **83.13%** of the incumbent's unreachable sites are
   wrong at all four, against **19.78%** of the sites the optimizer did reach — **4.20×**,
   reproducing at 4.21× on the second GT lineage, with both groups drawn from the same pool so
   size artifacts cancel.

---

## 0. The premise correction this arm owes first (MEASURED, and it changes the experiment)

The charter says to seal a cell "byte-identical except the initialisation … a pinned init file/seed
in the config". **There is no init seed in any burn cell's config.**

`experiments/ddm_qbr1_born_fairform_burn_prep.py:199-220` (`build_initial_state`) takes **no
argument at all**. It loads r10's `stage_03_end.pt`, copies `checkpoint["ema"]["shadow"]` verbatim,
and writes it as `qbr1_from_r10_ema_state.pt` (sha `991a1cc6…`, 398,687 B). The born vehicle's
random initialisation is `ddm_qbflow_packet.initialize_params(SEED)` /
`initialize_latents(SEED, N)` with `SEED = 20260827`
(`experiments/ddm_qbflow_rate_first_rung.py:44`, consumed at `:456-457`) — behind the **entire
qbt1 r1…r10 training chain**.

| claim | verified at | label |
|---|---|---|
| `build_initial_state` takes no seed and copies r10's EMA shadow verbatim | source, `ddm_qbr1_born_fairform_burn_prep.py:199-220` | MEASURED |
| the true init is `packet.initialize_params(20260827)` behind r1…r10 | source, `ddm_qbflow_rate_first_rung.py:44,456-457` | MEASURED |
| **my rebuilt `initialize_params(20260827)` reproduces the pinned `initialized_float_params.npz` byte-for-byte**, and the latents likewise | executed positive control | MEASURED |
| `seed_20260903`'s sealed config declares init sha `991a1cc6…` and `schedule.seed 20260903` with a different `schedule.sha256_u8` | the three wc3 sealed configs | MEASURED (confirms md2 §7) |
| md1's instrument reads **exactly one** field from a cell config — `config["initial_state"]["path"]` | `ddm_md1_micro_to_macro.py:431`, sole `config[` reference | MEASURED |

**Consequence, stated plainly: a different RANDOM INITIALISATION is not purchasable in this arm's
budget.** md2 §8 priced it at "a fresh QBF1 init (or an r10 re-derivation) + one 5,000-update burn
≈ 2.8 h"; the r10 re-derivation is the whole r1…r10 chain, not one burn. What this arm varies is
the **STARTING POINT** — a state some governed run actually reached and retained, copied verbatim,
never hand-edited. Every verdict below carries that scope: **different start, same root init seed.**

That last line in md1's instrument is also why the whole thing became cheap. Because the sweep
takes the step-0 field from the config's `initial_state` file and nothing else, **any candidate
start's step-0 wrong pool costs one forward pass** — and the pool is the object md2 showed the
persistent class is drawn from.

---

## 1. Free result #1: the prior cells' identical step-0 pool is construction, not finding

MEASURED, from md1's and md2's own retained excursion payloads (`trajectory_code_u8 & 1`):

| lineage / forward | cold pool | warm pool | ng5 pool | J(cold, warm) | J(cold, ng5) |
|---|---:|---:|---:|---:|---:|
| dali / shadow | 16,553 | 16,553 | 16,553 | **1.000000** | **1.000000** |
| dali / live | 16,553 | 16,553 | 16,553 | 1.000000 | 1.000000 |
| pyav / shadow | 16,327 | 16,327 | 16,327 | 1.000000 | 1.000000 |
| pyav / live | 16,327 | 16,327 | 16,327 | 1.000000 | 1.000000 |

md2 reported this identity for two cells and called it definitional. It is definitional for a
sharper reason than shared pins: **md1's instrument evaluates step 0 from the config's
`initial_state` file for every cell** (`ddm_md1_micro_to_macro.py:431,441`), so three cells
declaring one file can only ever produce one pool.

**I first read the warm cell as a hole in that and I was wrong; the check that proves it is worth
recording.** The warm cell's config carries `resume_from = .../warm_seed_mps.pt`, so its step-0
weights looked like they could differ from the pinned file — which would have made md1/md2's warm
rows a mis-read. They do not. `ddm_ng1_warm_transition_burn_prep.verify_warm_seed` loads the seed
exactly as `run_config` would and **REFUSES if `_load_checkpoint` moves a single weight** — *"warm
seed changed the start weights; the transition must carry optimizer state only"*. The warm
transition carries optimizer state, never weights. So the identical pool is correct rather than an
artifact, and the ng1 refusal is a positive control on it. No known-boundary item is owed here.

---

## 2. The ladder: eight legitimately-produced starts, and what the pool does

`experiments/ddm_md3_alternate_initial_states.py` (committed `4abab1d99`) writes each candidate in
the same `ddm_qbt2b_initialized_qbf1_state.v1` schema and through the same strict tensor-key round
trip `build_initial_state` enforces. Sources: r6–r10 `stage_03_end` (all 44 tensors, identical
keys and shapes, all at `stage_03_joint_boundary_interior_birth_end`), r10 periodic checkpoints,
and `packet.initialize_params(seed)` as a **diagnostic null anchor only** (untrained; the birth
gate forbids it as a cell start). Each was swept by **md1's instrument, UNCHANGED,
`--mode sweep --max-step 0`**.

**Positive controls, executed before any row below was read:** the incumbent's reconstruction
reproduces md2's published integer bridge exactly — weighted numerator **301,470**, denominator
**117,964,800**, `d_seg_hat` **0.0025555929**, pool **16,553** (DALI) / **16,327** (PyAV),
persistent **11,842** / **11,621**. The tool refuses if the retained PERSISTENT class map is not a
subset of the reconstructed pool; it did not refuse. The near-identity rung
(`r10_periodic_009999`, rel L2 7.6e-6) returns J 0.9909 and containment 99.89%, as it must.

### EMA shadow, DALI authority (n32; the same reading md1/md2 call authoritative)

| start | rel L2 from incumbent | pool sites | step-0 `d_seg_hat` | J(pool, incumbent) | incumbent PERSISTENT still wrong | × chance |
|---|---:|---:|---|---:|---:|---:|
| *(incumbent `991a1cc6…`)* | 0 | 16,553 | 0.0025555929 | 1.0000 | — | — |
| **r10_live** | 0.0084 | 17,151 | 0.0026658376 | **0.6288** | **91.01%** | 334× |
| r10_periodic_004950 | 0.0350 | 19,902 | 0.0031047821 | 0.6089 | 95.89% | 303× |
| r10_periodic_001650 | 0.0596 | 22,344 | 0.0035010020 | 0.5536 | 95.36% | 269× |
| r10_periodic_000033 | 0.0725 | 54,240 | 0.0085718791 | 0.2421 | 92.16% | 107× |
| r9_shadow | 0.0726 | 19,958 | 0.0030919393 | 0.5781 | 92.23% | 291× |
| r8_shadow | 0.1467 | 29,682 | 0.0046138763 | 0.4037 | 90.78% | 192× |
| r7_shadow | 0.2662 | 82,648 | 0.0131431580 | 0.1506 | 86.62% | 66× |
| r6_shadow | 0.3024 | 56,887 | 0.0090760549 | 0.1175 | **49.89%** | 55× |
| *root_init 20260827* (null anchor, born-gate INELIGIBLE) | 0.3424 | 3,150,870 | 0.5020847321 | 0.0045 | 87.96% | 1.76× |
| *root_init 20260905* (null anchor, a genuinely different random init) | 1.2435 | 3,171,971 | 0.5047678630 | 0.0045 | 87.96% | 1.76× |

`× chance` is the observed containment over `pool_sites / 6,291,456`, the containment a pool of
that size would achieve by drawing uniformly. The PyAV block agrees throughout (r10_live J 0.6246,
containment 90.83%).

**Read it in two halves, because they say opposite-looking things and both are the finding.**

* **The POOL moves, enormously.** Jaccard falls 0.99 → 0.12 across the ladder. A different start
  really does paint a different set of initially-wrong sites; md2's bit-identical pools were an
  artifact of every cell declaring one file, not a property of the vehicle.
* **The incumbent's UNREACHABLE sites do not.** Their containment stays 86.6–95.9% out to
  rel L2 0.266 (r7), where the pool Jaccard has already collapsed to 0.151 — 55–334× chance
  everywhere on the ladder.

**The honest wrinkle: r6 breaks the monotone at 49.89%.** r6 sits farthest out (rel L2 0.302) and
its pool is 3.4× the incumbent's, yet only half the incumbent's unreachable sites are wrong there —
while r7, which is *worse* by step-0 `d_seg` (0.0131 vs 0.0091), holds 86.62%. So r6 is wrong in a
genuinely *different* place, not merely wrong more. One rung is not a mechanism; it is recorded, not
explained.

**And the ladder confounds two variables, which I should not let the table hide.** The runs did not
train equally: r6 and r7 stop at **step 5,010**, r9 and r10 at **10,010**, r8 at **15,010**
(MEASURED from each `stage_03_end.pt`). So "farther in weight space" and "less trained" move
together on the low rungs, and the pool-size column tracks training amount at least as well as it
tracks distance. That is why §3 does not read the ladder as a dose-response at all: it restricts to
the four starts whose step-0 `d_seg_hat` is comparable to the incumbent's and uses a
within-experiment contrast instead.

**The null anchors behave as null anchors and that is their whole value.** Both untrained inits —
the incumbent's own root seed and a genuinely different one — are wrong on 50.1% of the frame, and
their containment collapses to 1.76× chance. Identical to four significant figures for both seeds,
which is what one expects before training. They put a floor under the containment scale and answer
nothing about reachability; naming them as if they did would be the substitution this memo just
corrected in §0.

---

## 3. Free result #2 — the decisive one, because it carries its own control

Containment can be inflated by a bigger pool. So compare, inside one cell and one lineage, two
groups that are **both** step-0-wrong sites of the incumbent:

* **UNREACHABLE** = the PERSISTENT class (11,842 DALI) — the optimizer did not fix them in 5,000
  updates;
* **REACHED** = step-0 wrong but *not* persistent (4,711 DALI) — the optimizer did fix them.

Then ask, of each group, what fraction is wrong at **all four** starts whose quality is comparable
to the incumbent's (r10_live, r9, r10@4950, r10@1650 — step-0 `d_seg_hat` 0.00267–0.00350 against
the incumbent's 0.00256):

| group | n | wrong at ALL four comparable starts | share |
|---|---:|---:|---:|
| **UNREACHABLE (persistent)** | 11,842 | **9,844** | **83.13%** |
| REACHED (step-0 wrong, optimizer fixed it) | 4,711 | 932 | 19.78% |
| | | | **ratio 4.20×** |

PyAV reproduces it: 83.11% vs 19.76%, **ratio 4.21×**.

**Being unreachable-by-the-optimizer predicts being wrong under every other comparable start, 4.2×
over the sites the optimizer did reach.** Both groups are drawn from the same 16,553-site pool of
the same cell, so any pool-size or scorer-difficulty artifact hits both and cancels in the ratio.
That is the sharpest statement this arm can make, and it is a within-experiment contrast rather
than a comparison against a modelled null.

Across all eight eligible starts, **4,397 of 11,842 (37.13%)** unreachable sites are wrong at every
one, and only **8 of 11,842** are correct at every one. (An independence null across starts would
put the first figure at ~1e-16, but the starts are *not* independent — they share a lineage — so
that null is unrealistic and is reported only to show the direction, never as a p-value.)

---

## 4. The cell: sealed, validated inside its own firing tree, armed — and NOT fired

`experiments/ddm_md3_different_init_cell.py` (committed `905c278ad`) recompiles ng5's composition
**in this tree** and swaps the start, nothing else.

| artifact | value |
|---|---|
| cell id | `seed_20260902_different_init_r10_live_control_native100` |
| sealed config | sha `010f8450a609f12b…`, 13,109 B |
| **re-rooted config (the one that fires)** | sha `8b726db112ba5b2e…`, 13,589 B |
| authorized config | sha `1474f9de4fb9325b…`, 13,588 B |
| sealed source tree | `/Volumes/VertigoDataTier/pact/ddm_md3_different_initialisation/sealed_source_17193c34a9` at revision `17193c34a9c5302325fdcb5fb596332018ee1cf6` |
| queue spec | sha `6a147cc2b585a022…`, 3,439 B |
| new start | `md3_r10_live_state.pt`, sha `414b7701dffce74e…` (incumbent `991a1cc653c786af…`) |
| **`differing_keys` vs ng5's cell** | **exactly `[cell_id, initial_state, output]`** — the allowed set, nothing else |
| lever legs vs ng5's SEALED config on disk | `tau_band`, `margin_dual`, `expected_flip_tau_start/end`, `initial_lambdas` — all **identical** |
| **in-tree validation** | **PASS** — τ `[0.04376363754272461, 0.021881818771362305]`, `msafe_band`, `r10_continuation` |
| pin re-root | 20 pins, **content-identical**, paths rooted in the firing tree |
| burn-path files vs ng5's sealed tree AND md3's own | all four **byte-identical** (`8de4112c…`, `68f29774…`, `cdf90d1a…`, `8947ceec…`) |
| seal wall clock | 132.4 s, $0 |

**The no-op detector is INVERTED here and is already satisfied by measurement.** For this cell the
step-0 field must *differ* from the incumbent's — that is the lever. MEASURED: pool Jaccard 0.6288
(DALI), 13,012 sites shared, **7,680 differ** (4,139 wrong only under the new start, 3,541 only
under the incumbent). The seal gate reads that from the retained receipt and refuses if the pools
ever agree.

### Why r10_live, and not a farther rung — the gate that chose it

The pre-registered falsifier is `J(persistent_new, persistent_cold) >= 0.70`. A cell's persistent
set is a subset of its own step-0 pool, so the pool sizes **cap J before the burn starts**. DERIVED
from the measured pool sizes and containments:

| rung | max attainable J | can the 0.70 falsifier fire? |
|---|---:|---|
| **r10_live** | **0.8082** | yes, with headroom |
| r10_periodic_004950 | 0.7711 | yes, thin |
| r9_shadow | 0.7187 | barely — 0.019 of headroom |
| r8_shadow | 0.4815 | **no** |
| r7_shadow | 0.1689 | **no** |
| r6_shadow | 0.1267 | **no** |

Firing r8 or below would install a gate that **cannot fire by arithmetic** — ng5's own lesson about
a gate wearing the name of one that works. r9 is the scientifically more attractive rung (an
independent training run) but its falsifier has 0.019 of headroom, which is a gate that fires on
size, not on physics. r10_live is also the maximally comparable rung: same source checkpoint, same
step, step-0 `d_seg_hat` within 4.3% of the incumbent's.

### Why it did not fire

`cell_queue_driver.py run --dry-run` returns **`ready: true`**, `blockers: []`, seal
content-identical, 20 pins, storage 52.9 GB against an 8.6 GB reserve, peak resolved
`FROM_LEDGER` = 49.572 GiB (`SOLE_CELL_INFERRED_FROM_LEDGER`). **The driver would have fired it.**
Its admission is an `ADVISORY_GATE` and admits on memory grounds — cl2's declared peak is 1.657 GiB.

I did not fire, because **cl2's λ=0.5 trainer is live on `--device mps`** (pid 21860, plus its
`price` sibling), and the binding rule is one Metal cell at a time, not one memory-admissible cell.
Instead a waiter is armed (`wait_for_metal_then_fire.sh`, launched through
`tools/launch_detached_process.py`, receipt `md3_waiter_fire`) that gates on the **process table**
— the actor — rather than on a receipt that may never be written, requires the Metal to read clear
**3/3 polls a minute apart**, logs a `--dry-run` plan, then fires. It refuses after 6 h rather than
hold a slot forever.

**That the driver said `ready` while a Metal cell was live is itself worth recording.** The
admission gate is honest about being advisory; a reader who takes `ready: true` as "the Metal is
free" would fire two cells. The waiter is the cure at this arm's scope; a driver-level Metal
exclusivity check is named as a follow-on and NOT built here.

---

## 5. The data-order control

**STATUS: LANDED** (launched 14:01:15Z, all four stages rc=0 by 14:53:21Z). md1's instrument UNCHANGED on the retained `wc3 seed_20260903` control
(313 checkpoints; same init sha `991a1cc6…`, `schedule.seed` 20260903 with a different
`schedule.sha256_u8` — **data order only**), both analyses, both overlaps against md1's own
cold store. This is the fourth cell on one initialisation and the first that varies nothing
but the order the pairs arrive in.

### The partition, EMA shadow, DALI authority

| class | data-order sites | terminal-wrong | share | cold share | ng5 share |
|---|---:|---:|---:|---:|---:|
| **PERSISTENT** | **11,776** | **11,355** | **61.606%** | 62.011% | 62.954% |
| CHURN | 10,332 | 4,290 | 24.393% | 24.606% | 24.687% |
| NEW_PERSISTENT | 1,906 | 1,906 | 10.828% | 10.470% | 9.142% |
| HEALED | 1,512 | 556 | 3.174% | 2.913% | 3.217% |
| TRANSIENT_BORN | 7,012 | 0 | 0.000% | 0.000% | 0.000% |

* **dali** — shadow PERSISTENT **61.606%** (cold 62.011%, ng5 62.954%); live PERSISTENT 40.562% (cold 35.779%); terminal `d_seg_hat` 0.0028207143 (cold 0.0028065999).
* **pyav** — shadow PERSISTENT **61.289%** (cold 61.670%, ng5 62.346%); live PERSISTENT 40.275% (cold 35.409%); terminal `d_seg_hat` 0.0027788798 (cold 0.0027610779).

**Reachability.** Persistent floor `0.0017377218` = **12.734×** the sub-0.12 accuracy corner `d_seg = 1.3646784205e-4` (cold 12.753×, ng5 11.671×). Integer calibration gate: max |Σ classes − total| = 0 at every checkpoint, in integers.

### The site overlap — a different DATA ORDER against the cold control

| forward / lineage | data-order persistent | cold persistent | intersection | **Jaccard** | data-order ⊂ cold |
|---|---:|---:|---:|---:|---:|
| **shadow / dali** | 11,776 | 11,842 | 10,876 | **0.8536** | 92.36% |
| live / dali | 7,478 | 7,396 | 6,906 | 0.8667 | 92.35% |
| shadow / pyav | 11,558 | 11,621 | 10,664 | 0.8521 | 92.27% |
| live / pyav | 7,344 | 7,253 | 6,781 | 0.8676 | 92.33% |

**Data order alone moves the persistent sites about as little as the schedule does.** J = 0.8536 on the shadow/DALI reading, against md2's schedule-lever J 0.8069 for ng5 vs the same cold control. Both cells here share the step-0 pool by construction (§1), so the within-pool null applies to this number exactly as it did to md2's, and the comparison that matters is J-to-J: **changing the data order and changing the whole two-lever schedule perturb the unreachable set by a similar amount, and neither is anywhere near moving it.**

**One number here breaks a pattern md2 reported, and it is a finding rather than noise.** md2 said
the LIVE-forward persistent share agrees across cells within 0.44 pp (35.779% cold / 35.336% warm /
35.439% ng5). The data-order cell lands **40.562%** (PyAV 40.275%) — **+4.78 pp**, an order of
magnitude outside that spread, while its SHADOW share sits inside the shadow spread at 61.606%. So
the live forward is materially more sensitive to data order than to any schedule lever tried,
and the shadow forward is not. That is consistent with md1's own reason for preferring the
shadow — the live weights are noisy checkpoint-to-checkpoint and CHURN absorbs most of their
terminal error — but it is a **new** fact: the live-forward agreement md2 recorded was an
agreement across three cells that shared a data order, and it does not survive varying it. The
shipped object is the shadow, so no conclusion in md1/md2/md3 moves; the live-forward
cross-cell-stability claim should be re-scoped.

**What this control does and does not buy.** It removes the data-order confound from md1/md2's persistent share — the share now has four instances on one initialisation — and it is the measurement the charter asked for first. It cannot speak to initialisation at all: the init sha is `991a1cc6…` here as everywhere else, which is precisely md2 §7's correction and the reason §2 exists.

---

## 6. What this does and does not answer

**Answers.** The step-0 wrong pool is *not* a fixed property of this vehicle — it moves from
J 0.99 to J 0.12 as the start moves 0.8% → 30% in relative weight distance. And within that moving
pool, the sites the optimizer cannot reach are the sites other comparable starts also get wrong,
4.20× over the sites it can reach. The data-order control adds the fourth leg: order alone moves
the unreachable SITES (J 0.8536) *less* than the two-lever schedule does (0.8069), and the share
holds at 61.606% — four instances within 1.35 pp. On the evidence available for $0, **the
unreachable set behaves as a property of the DATA (the frozen scorer's hard sites) far more than of
the START, and the two nuisance variables that could have explained it — schedule and data order —
are both now measured and both fail to move it.**

**Does not answer.** A site wrong at step 0 under start X is not thereby *unreachable* from X.
The pre-registered falsifier — does a burned cell from a different start leave those same sites
persistent — is exactly the gap, and it is the cell that is sealed and armed. My verdict is a
prediction about that cell: given 91.01% containment and a 0.8082 ceiling, I expect
`J >= 0.70` and the falsifier to FIRE. **That is stated before the burn and is falsifiable.**

**Scope, welded on.** Eight starts, all descending from root init seed 20260827. One vehicle, one
seed, one n32 selection. Nothing here tests a different random initialisation; §0 explains why
that is not purchasable here and §7 prices it honestly.

**The n32 caveat, as md1 and md2 stated it.** All 32 pairs of the sealed no2 stratified
Horvitz–Thompson selection, integer weights `(15.0,)*24 + (30.0,)*8`, pair ids `(4, 31, 49, 52, 62,
90, 100, 113, 128, 148, 173, 179, 186, 187, 214, 236, 256, 260, 268, 278, 326, 328, 341, 352, 368,
382, 444, 456, 483, 508, 563, 573)`. The HT estimator estimates the n600 population and **n32 →
n600 transfer is untested on this vehicle**. The selection is stratified across the video, not a
contiguous prefix, so the prefix-bias genus (`[[m88]]`) is not the applicable caveat; n32 → n600
transfer is.

---

## 7. Priced next step

**If the armed cell fires and `J >= 0.70`:** the born accuracy corner is closed at FAMILY scope for
this generator form, gs4 §5(b) is answered, and no further start-variation is worth buying. The
route needs a different generator (gc1/gf2 territory). Cost already sunk: the waiter fires itself.

**If it fires and `J <= 0.45`:** my §6 prediction is wrong, initialisation is a live lever, and the
next unit is an init *search* — for which the honest price is now known and is **not** md2's 2.8 h.
A genuinely different random initialisation requires re-running `initialize_params(new_seed)`
through the entire qbt1 r1…r10 chain (r10's stage 03 alone is 10,010 updates; r8's is 15,010).
Nobody has measured that chain's wall clock end-to-end; **pricing it is the prerequisite**, not the
run. `[[m118]]` — price the ceiling before chartering the arm.

**Either way, one free thing is worth doing first:** the eight starts' step-0 payloads are retained.
The 4,397 sites wrong at *every* eligible start are an enumerated, persisted object. Any arm that
wants to price an explicit representation of the scorer-hard core can read them directly rather
than re-derive the set — the same offer md2 made for its 11,019.

**Not built here, and named:** a Metal-exclusivity check inside `cell_queue_driver` (§4). This
arm's waiter cures the symptom for one cell; the next reader of `ready: true` gets no such cure.

---

## 8. Equations leg (`tac.canonical_equations`)

**A FOURTH empirical anchor is appended** to `checkpoint_trajectory_error_partition_v1`
(`src/tac/canonical_equations/checkpoint_trajectory_error_partition_20260904.py`):
`md3_data_order_control_seed_20260903_shadow_trajectory_partition_20260905`. It is a legitimate
fourth cell on the law's own terms — same vehicle, same cadence, same authority, the full
five-class integer partition with residual 0 — and it is the first that varies the DATA ORDER.
It carries terminal numerator 332,745, persistent numerator 204,990, share 0.6160573412072308,
floor 12.734× the corner, live share 0.4056160088528219, and site overlap J 0.8535551718725475
(intersection 10,876) against md2's schedule-lever 0.8069 on the same cold control. Its
pre-registered band was [0.55, 0.65]; measured 61.606%; residual 0.0039. Every value is read from
the receipt, none retyped — a review pass asserts each against
`ANALYSIS_data_order_control_seed_20260903_dali.json` and the overlap JSON.

**The law's `known_boundary` is rewritten, in both directions.** It now records four cells and
that **data-order-invariance of the shadow share is MEASURED** alongside schedule-invariance —
and that the **live** forward is *not* order-invariant (40.562% against the 35.3–35.8% of the
three cells that shared an order), so the earlier live-forward cross-cell agreement must not be
transferred. It also records that **init-invariance remains UNMEASURED and is not cheaply
purchasable** for the §0 reason, with the step-0 half (pool J 0.9909 → 0.1175; 83.13% vs 19.78%)
recorded as anchor context rather than as a term of the law — it is a step-0 statement, not a
trajectory partition. The ng5 anchor's `reactivation_criteria` named *"a cell with a DIFFERENT
INITIALISATION"*; this arm reports that the named experiment is **not purchasable as stated**,
which is a correction to the criterion rather than a satisfaction of it, and the new anchor's own
criteria name the sealed cell and the r1…r10 pricing that must precede any init search.

**Also consumed, unchanged:** `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — the per-site
scalar the wrong/correct indicator is built on. No refinement claimed.

**Not registered, deliberately, and each with its reason.**
* *The start-dependence of the step-0 pool.* One vehicle, one root init seed, eight starts on one
  lineage. A law needs the root seed varied, which §7 prices.
* *The unreachable-vs-reached contrast (4.20×).* The sharpest number here, and still one arm on one
  cell. It becomes a law when the armed cell measures the same contrast on a different start's own
  partition — which is exactly what that cell produces.
* *The subset-constrained Jaccard ceiling* (§4: when both sets are subsets of measurable pools, the
  pool sizes cap J before the experiment runs, so a pre-registered set-overlap falsifier must be
  checked against that ceiling or it can be structurally unfireable). This is DERIVED arithmetic,
  it already changed a decision in this arm, and it generalises past this vehicle — the strongest
  registration candidate of the three. It is left unregistered only because a law landed on the
  same day it was first used, by the arm that used it, has no independent consumer yet. **Named as
  owed** rather than dropped.

---

## 9. Custody

`$0`. CPU only. **0 Metal, 0 Modal, 0 contest eval by this arm.** `upstream/` untouched. No `/tmp`
path in any artifact. Every long step launched through `tools/launch_detached_process.py` with a
declared measured peak; the step-0 probes and the data-order partition each ran as a single
process at `--threads 4`, peak declared 15.4 GiB from md2's realized 15.367 GiB.

Store `/Volumes/APDataStore/pact/ddm_md3_different_initialisation/`:

| artifact | what it holds |
|---|---|
| `alternate_initial_states/md3_*_state.pt` | the eleven candidate starting states, schema `ddm_qbt2b_initialized_qbf1_state.v1`, each with source-checkpoint provenance |
| `probe_configs/md3_*_probe.json` | the minimal per-candidate configs md1's instrument consumes |
| `step0_probes/payloads/md3_start_*/shadow_step_000000.npz` | **the retained step-0 argmax and δ_R band for every candidate** — the payload every §2/§3 row derives from |
| `ALTERNATE_INITIAL_STATES.json` | the ladder index: sha, provenance, weight-space distance, born-gate eligibility |
| `STEP0_POOL_OVERLAP.json` | §2: pool sizes, `d_seg_hat`, Jaccards, chance and attainable-max nulls, containments, both lineages |
| `GOOD_START_CONTRAST.json` | §3: the 4.20× unreachable-vs-reached contrast, both lineages |
| `PERSISTENT_WRONG_AT_EVERY_START.json` | §3: the 4,397 sites wrong at all eight eligible starts |
| `sealed_configs/`, `authorized_configs/`, `QUEUE_SPEC.json`, `SEAL_RECEIPT.json` | §4: the sealed cell, its re-root receipt and in-tree validation |
| `sweep_rows_…_seed_20260903.jsonl`, `ANALYSIS_…`, `site_classes_…`, `excursion_…`, `OVERLAP_…` | §5: the data-order control's full partition |
| `md3_*_driver.sh`, `wait_for_metal_then_fire.sh`, `launch*/` | the exact recipes and every launch manifest + safe_run receipt |

Sealed tree `/Volumes/VertigoDataTier/pact/ddm_md3_different_initialisation/sealed_source_17193c34a9`.
Instruments `experiments/ddm_md1_micro_to_macro.py` and `experiments/ddm_md2_persistent_site_overlap.py`
**unmodified**; the new modules are `ddm_md3_alternate_initial_states.py` (`4abab1d99`),
`ddm_md3_step0_pool_overlap.py` (`17193c34a`) and `ddm_md3_different_init_cell.py` (`905c278ad`),
each with two review-gate passes, ruff clean.

## NEXT_IF_RESUMED

1. **Harvest the armed cell** (`md3_waiter_fire` → `md3_different_init_DONE.json`), then run md1's
   instrument + `ddm_md2_persistent_site_overlap.py` on its run root against md1's cold store. The
   pre-registered falsifier is `J >= 0.70`; the ceiling is 0.8082; my prediction is FIRE.
2. **The §7 pricing** — the wall clock of the full `initialize_params(new_seed)` → r1…r10 chain —
   before any charter proposes an init search.
3. **The Metal-exclusivity gap in `cell_queue_driver`** (§4), which this arm worked around rather
   than fixed.

`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`

---

# Terminal read (md4)

**Arm:** ddm_md4 · **UTC:** 2026-09-06T00:45Z · **Tokens:** `[no-triality] [p0-ledger-ok]`
**verdict_scope:** `[macOS-CPU advisory . site sets reconstructed by ddm_md1_micro_to_macro from the cell's own retained 16-step checkpoints . frozen CPU-torch SegNet+PoseNet . QBF1-born vehicle . n32 sealed stratified selection . ONE burned cell from ONE different STARTING POINT, same root init seed 20260827 . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval by this arm]`
**Append-only.** Nothing above this line is rewritten. This section reports what the fired cell measured.

## VERDICT: **DATA-ANCHORED** — the pre-registered falsifier FIRED

`J(persistent_different-start, persistent_cold) = **0.7279606507756338**` on the EMA shadow / DALI
authority, against the charter's `J >= 0.70 -> DATA-ANCHORED`. PyAV reproduces at **0.7242097147262915**.
MEASURED. The `<= 0.45` INIT-ANCHORED branch is not close: the measured value clears it by 0.28.

md3 §6 wrote its expectation **before** the burn — *"given 91.01% containment and a 0.8082 ceiling, I
expect `J >= 0.70` and the falsifier to FIRE"*. That prediction is now MEASURED CORRECT. md3's §0-§5
formulation-scope verdict is upgraded from *"on the evidence this arm could buy for $0"* to a verdict
carrying its own pre-registered burned cell. The scope word stays **FORMULATION**, not FAMILY: see §6.

## The md1 table — `seed_20260902_different_init_r10_live_control_native100`, EMA shadow, DALI

| class | sites | terminal wrong | terminal d_seg contribution | share of terminal error |
|---|---:|---:|---:|---:|
| ALWAYS_CORRECT | 6,261,145 | 0 | 0 | 0.000% |
| CHURN | 9,756 | 3,557 | 0.000562032 | 22.931% |
| **PERSISTENT** | **10,993** | **10,468** | **0.0015900929768880208** | **64.877%** |
| NEW_PERSISTENT | 1,294 | 1,294 | 0.000204086 | 8.327% |
| TRANSIENT_BORN | 6,130 | 0 | 0 | 0.000% |
| HEALED | 2,138 | 600 | 9.47316e-05 | 3.865% |

Integer calibration gate `max_t |Σ classes − total| = **0**` at every checkpoint, in integers, on both
forwards and both lineages. Terminal `d_seg_hat` **0.0024509429931640625**; persistent floor
**11.652×** the sub-0.12 accuracy corner `1.3646784205e-4`. **Birth:** over-paint birth step
`Lane 32` (shadow) / `Lane 16, Movable 16` (live); excursion peak at step **1728** (shadow) / **16**
(live). **Lane:** lane-touching share of terminal wrong **64.387%**, GT=Lane enrichment **54.578×**.
PyAV: PERSISTENT 10,742 sites, share **64.568%**, floor 0.0015501658121744792.

Live forward, DALI: PERSISTENT 6,682 sites, share **35.237%**, terminal `d_seg_hat` 0.002825164794921875.
That share sits inside md2's three-cell live band (35.3–35.8%) and **outside** md3 §5's data-order
outlier (40.562%) — so the live forward's sensitivity that md3 found is to DATA ORDER, not to the START.

### The five instances of the persistent share (shadow, DALI)

| cell | lever vs the cold control | PERSISTENT share | floor / corner |
|---|---|---:|---:|
| cold control seed_20260902 | — | 62.011% | 12.753× |
| ng1 warm transition | AdamW moments carried in | 59.009% | — |
| ng5 τ-band × carried duals | two schedule levers | 62.954% | 11.671× |
| md3 data-order control | pair order only | 61.606% | 12.734× |
| **md4 different START (this cell)** | **r10 live weights instead of r10 EMA shadow** | **64.877%** | **11.652×** |

## Both Jaccards, with the null the charter asked for

The pools DIFFER here, so the charter required the pools' own Jaccard first and a within-pool null.
**Step-0 pools: 17,151 (new) vs 16,553 (cold), intersection 13,012, `J_pool = 0.6288`** (MEASURED by
md3 §2 and unchanged). The within-pool null generalises md2's shared-pool null to differing pools —
`E[|A∩B|] = |pool_a ∩ pool_b| · (n_a/|pool_a|) · (n_b/|pool_b|)`, which reduces to md2's
`n_a·n_b/|pool|` exactly when the pools coincide (the general form reproduces md2's published
0.5263033946906265 to 1e-15; regression-tested).

| comparison | forward | measured J | within-pool null | measured / null | intersection |
|---|---|---:|---:|---:|---:|
| **vs COLD control** | shadow | **0.7279606507756338** | 0.3537 | **2.058×** | 9,620 |
| vs COLD control | live | 0.7917780323278605 | 0.1917 | 4.129× | 6,221 |
| **vs DATA-ORDER control** | shadow | **0.7379589344324861** | 0.3524 | **2.094×** | — |
| vs DATA-ORDER control | live | 0.7960426179604262 | 0.1929 | 4.126× | — |

Containment: **87.51%** of this cell's persistent sites are also cold-persistent; **81.24%** of the
cold cell's are also persistent here.

**Against the ceiling.** md3's `0.8082` is RE-DERIVED here from the measured step-0 pools rather than
copied: `10,777 / (11,842 + 12,269 − 10,777) = **0.8082345882705865**`, matching to 4 dp — but that
ceiling assumed the new cell's persistent set would be the same *fraction of its pool* as the cold
cell's (71.54%). It is not: **10,993/17,151 = 64.10%**. With the now-measured size the true ceiling is
**0.8937634765301045**. So the measured 0.7280 is **90.07%** of the pre-registered ceiling and
**81.45%** of the real one. The gate was fireable with genuine headroom in both readings — it did not
fire on size.

## The number that closes md3's own named gap

md3 §6 stated the gap precisely: *"A site wrong at step 0 under start X is not thereby unreachable
from X."* It is now measured. Of the **10,777** cold-persistent sites that are step-0-wrong at the new
start, **9,620 stayed persistent through 5,000 updates** — a step-0 → trajectory transfer rate of
**89.26%**. The step-0 probe was not merely correlated with unreachability; it predicted it at
nine-in-ten.

## The mechanism behind the share, stated so it is not misread

64.877% is the highest of the five instances and sits at the very top edge of the pre-registered
[0.55, 0.65] band. **The floor did not rise — the reachable part shrank.** Against ng5, the comparator
that is byte-identical except the start:

* persistent floor **0.0015900929768880208** vs ng5's **0.0015927632649739582** → **−0.17%**;
* optimizer-reachable error **0.00086085** vs ng5's **0.00093727** → **−8.15%**;
* terminal `d_seg_hat` **0.0024509** vs ng5's **0.0025300** → **−3.13%**.

So the different start bought a real but small terminal improvement, took it **entirely out of the
reachable class**, and left the floor within 0.17% — 11.652× the corner against 11.671×. A share that
went UP is here evidence that the start helped only where the optimizer was already winning.

## What this does and does not answer

**Answers.** The trajectory half of md3 §3 is now measured, not inferred: a burned cell from a
genuinely different starting point — one whose step-0 wrong pool overlaps the incumbent's at only
J 0.6288 — converges on the *same* unreachable sites at J 0.7280, 2.06× the within-pool null and 81%
of the attainable ceiling, and leaves a floor 0.17% from the incumbent's. The two nuisance variables
md1–md3 could vary (schedule, data order) and this third one (starting point) all fail to move the
unreachable set.

**Does not answer — the scope word is FORMULATION, and here is exactly why.** All four cells and all
eight probed starts descend from **root init seed 20260827** through the qbt1 r1…r10 chain. This cell
varies the STARTING POINT (r10's live weights instead of r10's EMA shadow, rel L2 0.0084); it does not
vary the random INITIALISATION, which md3 §0 established is not purchasable without re-running that
whole chain. A start 0.84% away in weight space is a weak lever for this question even though its
step-0 *pool* moved a lot. Calling the born accuracy corner closed at **FAMILY** scope would need
either a genuinely different root init seed or a start far enough out that its pool Jaccard approaches
the r7/r6 regime — and md3 §4 showed those rungs cannot fire a 0.70 gate by arithmetic, so that
experiment needs a *different* falsifier, not a farther rung.

**The n32 caveat, unchanged from md1/md2/md3.** All 32 pairs of the sealed no2 stratified
Horvitz–Thompson selection, integer weights `(15.0,)*24 + (30.0,)*8`. n32 → n600 transfer is untested
on this vehicle; the selection is stratified across the video, not a contiguous prefix, so `[[m88]]`
prefix bias is not the applicable caveat.

## The resume is a measured non-confound, not an assumed one

The cell was killed by a wall-clock cap at step 4,552 and resumed from `periodic_004544.pt`, so the
partition is read off checkpoints that straddle a restart. MEASURED
(`terminal_read/RESUME_BOUNDARY_CONTROL.json`):

* `config_identity_sha256` **identical** on both sides (`89887c6d9e8a8ffb…`), all nine checkpoints
  4,480…4,608;
* EMA update counter monotone 4,480…4,608 — the replayed steps 4,545–4,552 are not double-counted;
* EMA-shadow displacement across the boundary **1.215436e-04** against an interior range of
  1.2136–1.2185e-04 → ratio **0.99954**;
* live-weight displacement **1.049793e-03**, sitting on the window's own upward trend
  (0.893–0.962e-03 before, 1.107–1.163e-03 after) → ratio **1.027** to the pooled interior mean;
* `history.jsonl` monotone 1…5,000 with exactly one row per step; `RESULT.json` `complete: true`,
  `completed_steps: 5000`, `metal_invocations: 1`.

**Sensitivity, stated honestly:** the EMA shadow at decay 0.99908 attenuates a live discontinuity by
~10³, so the shadow ratio alone is a weak detector. The live ratio is the sensitive channel — a lost
optimizer moment would show as a large boundary step, and 1.027× is ordinary. This is a control, not a
bit-identity proof; bit identity would require re-running the killed segment.

Also verified: the config that fired differs from md3's authorized config **only** in
`metal_lane.claim_id/claimed`, `scorer_lane.claim_id/claimed` and (for the resume config) `resume_from`.
Every physics field — `initial_state` sha `414b7701…`, τ band, `initial_lambdas`, schedule — is
unchanged. The authorized-config sha md3 §4 recorded (`1474f9de…`, 13,588 B) is now the file the driver
tombstoned as `.unbound_by_driver.20260905T154454Z.json`; the config that actually fired is
`fb3773f57b2c6813…` (13,649 B) and the resume config is `d2e62e05e640bc29…`.

**Wall clock, for custody:** segment 1 ran at ~11 min per 16 steps under cl2's Metal contention;
segment 2 at ~35 s per 16 steps with the Metal free — a 19× starvation, visible in the checkpoint
mtimes and in nothing else. The burn was starved, never broken.

## An apparatus finding, recorded because it is a vacuity class

`tools/review_tracker.py mark-file` does **not** rescan the file. A function added to an
already-scanned module is therefore marked **vacuously** — this arm's module reported "11 entities
reviewed" twice while the new function was invisible to the tracker, and only after
`review_tracker.py scan` did it report 12. A reader who takes the mark-file receipt as coverage gets a
PASS whose denominator excluded the new code (`[[m50]]`: VACUITY==PASS — report the denominator). The
cure at this arm's scope was to run `scan` before marking; a `mark-file`-implies-rescan change is
NAMED as a follow-on and NOT built here.

## Priced next step

The charter's §7 branch is now selected: **the falsifier fired**, so *"no further start-variation is
worth buying"* on this generator form, and gs4 §5(b) is answered at FORMULATION scope. The route to the
accuracy corner needs a different GENERATOR (gc1/gf2 territory), not a different start — the floor sat
at 11.65–12.75× the corner across all five cells and moved by 0.17% under the strongest start lever
available.

Two things stay owed and neither is this arm's to spend:

1. **FAMILY scope needs a different root init seed**, whose honest price is the wall clock of the full
   `initialize_params(new_seed)` → r1…r10 chain (r10 stage 03 alone is 10,010 updates). Nobody has
   measured that chain end to end; **pricing it is the prerequisite**, not the run (`[[m118]]`).
2. **The subset-constrained Jaccard ceiling** that md3 §8 named as owed is now used a second time, by a
   second arm, and it changed a reading again (the pre-registered 0.8082 was conservative by 0.086
   because it assumed a persistent-share-of-pool that the burn did not honour). It now has an
   independent consumer and is a live registration candidate.

**Free and already persisted:** every site set behind the table above is retained. The 9,620 sites that
are persistent in BOTH the incumbent and a different start are an enumerated object any arm can read
rather than re-derive.

## Equations leg (`tac.canonical_equations`)

A **FIFTH** empirical anchor is appended to `checkpoint_trajectory_error_partition_v1`:
`md4_different_start_r10_live_seed_20260902_shadow_trajectory_partition_20260906`. It is the cell the
law's own `known_boundary` named as owed — *"The trajectory half of that statement still needs a burned
cell from a different start"* — and that clause is rewritten to MEASURED. The anchor carries terminal
numerator 289,125, persistent numerator 187,575, share 0.6487678339818417, floor 11.652× the corner,
live share 0.3523719506706274, `J` 0.7279606507756338 against the cold control and 0.7379589344324861
against the data-order control, both with their within-pool nulls, and the 89.26% step-0 → trajectory
transfer rate. Its pre-registered band was [0.55, 0.65]; measured 64.877%; residual 0.0288 against the
0.62 prior. Every value is read from the receipt, none retyped.

**Also consumed, unchanged:** `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`. No refinement
claimed.

**Not registered, deliberately.** The subset-constrained Jaccard ceiling (§ above) — now twice-used and
consumer-independent, so its registration blocker is gone, but registering a law inside the arm that
harvests it, on the same day, repeats exactly what md3 declined to do. Named as owed, with the note
that the blocker md3 cited no longer applies.

## Custody

`$0`. CPU only, `--threads 4`, `nice 10`. **0 Metal, 0 Modal, 0 contest eval by this arm.**
`upstream/` untouched. No `/tmp` path in any artifact. Instruments
`experiments/ddm_md1_micro_to_macro.py` and `experiments/ddm_md2_persistent_site_overlap.py`
**unmodified**; the new module is `experiments/ddm_md4_terminal_read.py` (+18 tests, ruff clean, two
review-gate passes after a rescan, mutation-tested).

Store `/Volumes/APDataStore/pact/ddm_md3_different_initialisation/terminal_read/`:

| artifact | what it holds |
|---|---|
| `sweep_rows_seed_20260902_different_init_r10_live_control_native100.jsonl` | 141 forwards over the 71-checkpoint cadence |
| `payloads/…/{shadow,live}_step_*.npz` | the retained per-checkpoint argmax + δ_R band — every row above derives from these |
| `ANALYSIS_…_{dali,pyav}.json` | md1's partition, both lineages, both forwards |
| `site_classes_…npz`, `excursion_…npz` | the per-site class codes and trajectory codes |
| `OVERLAP_…_vs_cold_control_seed_20260902_{dali,pyav}.json` | the pre-registered Jaccard |
| `OVERLAP_…_vs_data_order_control_seed_20260903_{dali,pyav}.json` | the second comparator |
| `TERMINAL_READ_{dali,pyav}.json` | the verdict arithmetic, nulls, and re-derived ceiling |
| `RESUME_BOUNDARY_CONTROL.json` | the two-segment resume control |
| `TABLES_{dali,pyav}.md` | md1's rendered tables |
| `md4_terminal_read_driver.sh`, `launch/` | the exact recipe and the launch manifest + safe_run receipt |
