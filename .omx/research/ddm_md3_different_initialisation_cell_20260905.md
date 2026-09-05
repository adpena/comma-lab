---
title: "The born vehicle has no init seed to flip — build_initial_state copies r10's EMA shadow verbatim; so the question was answered a cheaper way: across eight legitimately-produced starts spanning 0.8%–30% relative weight distance, the step-0 wrong POOL moves enormously (Jaccard 0.99 → 0.12) while the incumbent's UNREACHABLE sites stay wrong at 55–334× chance, and over the four comparable-quality starts 83.13% of unreachable sites are wrong at ALL of them against 19.78% of the sites the optimizer DID reach — a 4.20× within-experiment contrast that reproduces at 4.21× on the second GT lineage"
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
held by cl2's λ rungs for this arm's whole window — so the pre-registered number is OWED and this
verdict rests on a different, cheaper measurement that the charter's own step 3 asked for first.
The verdict word is stated with that limit welded on, and §7 says exactly what would overturn it.

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

## 1. Free result #1: the step-0 pool is bit-identical across all three prior cells *by
construction*, not by finding

MEASURED, from md1's and md2's own retained excursion payloads (`trajectory_code_u8 & 1`):

| lineage / forward | cold pool | warm pool | ng5 pool | J(cold, warm) | J(cold, ng5) |
|---|---:|---:|---:|---:|---:|
| dali / shadow | 16,553 | 16,553 | 16,553 | **1.000000** | **1.000000** |
| dali / live | 16,553 | 16,553 | 16,553 | 1.000000 | 1.000000 |
| pyav / shadow | 16,327 | 16,327 | 16,327 | 1.000000 | 1.000000 |
| pyav / live | 16,327 | 16,327 | 16,327 | 1.000000 | 1.000000 |

md2 reported this identity for two cells and called it definitional. It is definitional for a
sharper reason than shared pins: **md1's instrument evaluates step 0 from the config's
`initial_state` file for every cell**, so three cells declaring one file can only ever produce one
pool. The **warm** cell is the case that matters — it *resumes from a checkpoint*, so its TRUE
step-0 weights are not the pinned file, yet its pool is read as the cold one's. That is a
known-boundary item for md1/md2's warm rows, recorded here and not otherwise acted on.

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

**STATUS: RUNNING at the time of writing** — `md3_step1_data_order`, launched 14:01:15Z, md1's
instrument UNCHANGED on the retained `wc3 seed_20260903` control (313 checkpoints; same init sha
`991a1cc6…`, `schedule.seed 20260903`, different `schedule.sha256_u8` — data order only). Numbers
are filled in below when the sweep, both analyses and both overlaps land; if this section still
reads RUNNING, the charter's item 1 is **incomplete and is reported as such**, not inferred.

<!-- MD3_DATA_ORDER_SECTION -->

---

## 6. What this does and does not answer

**Answers.** The step-0 wrong pool is *not* a fixed property of this vehicle — it moves from
J 0.99 to J 0.12 as the start moves 0.8% → 30% in relative weight distance. And within that moving
pool, the sites the optimizer cannot reach are the sites other comparable starts also get wrong,
4.20× over the sites it can reach. On the evidence available for $0, **the unreachable set behaves
as a property of the DATA (the frozen scorer's hard sites) far more than of the START.**

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

**Consumed, unchanged:** `checkpoint_trajectory_error_partition_v1`
(`src/tac/canonical_equations/checkpoint_trajectory_error_partition_20260904.py`). This arm is the
reactivation the ng5 anchor's `reactivation_criteria` names — *"a cell with a DIFFERENT
INITIALISATION"* — and it reports that the named experiment is **not purchasable as stated** (§0),
which is a correction to the criterion rather than a satisfaction of it. The law's
`known_boundary` clause *"schedule-invariance is measured and init-invariance is NOT"* is now
**partially narrowed and not closed**: start-invariance is measured **at step 0 only**, across
eight starts, and the trajectory half still needs the armed cell.

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
