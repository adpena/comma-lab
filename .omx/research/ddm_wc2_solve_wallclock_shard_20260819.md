# ddm_wc2 -- sharding the ddm_jg3 n600 joint solve for wall clock

**Axis:** `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false`.
Nothing here moves a pointer. This arm buys WALL CLOCK on the binding path.

**Arm:** `ddm_wc2` · **date:** 2026-08-19 · **subject:** the `ddm_jg3` n600 joint
solve, formerly pid 81175.

---

## 0. Headline

**SWAPPED.** The single live solver was terminated cleanly at 27 banked pairs
(zero pair loss) and replaced by **6 governed shards** over the 573 remaining
pairs, LPT-balanced on predicted flip count (load imbalance **1.0024**).

Two findings, one of which contradicts the charter's premise:

1. **Sharding is structurally licensed.** Per-pair acceptance is exactly
   independent across pairs -- six legs, section 1, read out of the solver rather
   than assumed. No solver edit was needed; the shards ride its existing
   `--pair-list` reproduction seam.
2. **Sharding is NOT decision-identical, and that claim is WITHDRAWN.** The
   charter predicted "equivalence is exact or the shard wrapper is wrong." The
   replay measured a third possibility the charter did not consider: the wrapper
   is right, and the solver is not reproducible across processes with different
   computation histories.

---

## 1. The independence analysis (why sharding is licensed)

The charter asked whether the in-loop rate estimate for pair *k* depends on edits
accepted for pairs already processed. **It does not.**

| # | Possible coupling | What the code does | Verdict |
|---|---|---|---|
| 1 | site-subsample RNG shared across pairs | `solve_pair` builds `np.random.default_rng(site_seed + pair)` FRESH per call (`:588`) | per-pair, order-free |
| 2 | rate ranker sees prior edits | `LogitPrice.bits_for` reads `self._memmap[pair]` (read-only, size-checked) and prices against `tokens_pair`, the BASE plane (`:405-425`) | static per-pair lookup |
| 3 | acceptance price is context-dependent | winner chosen on `cost_bits = tokens_here * 4.1379` (`:807`) -- a flat measured constant, deliberately NOT the logit sum | no cross-pair term |
| 4 | inputs carry cross-pair state | `tokens[pair]`, `base_argmax[pair]`, `gt[pair]` | none |
| 5 | model state mutates | `SegNet().eval()` + `requires_grad_(False)`; BatchNorm reads running stats | none |
| 6 | edit accumulator feeds back | `cmd_solve` writes `edits[str(pair)]`, never reads it into `solve_pair` | write-only |

Leg 3 is the exact term the charter worried about. The solver's own comment
records that the hm1 logits charge 1.91 bits/token where `ddm_jg2` MEASURED
4.1379 through a byte-identical encoder -- a 2.2x under-price -- so the author
already demoted the context-dependent price to a within-site RANKER and made the
absolute cost a constant. There is no cross-pair term to perturb.

Also noted: `max_sites` defaults to 0, so the site-subsample RNG never fires at
all. **Floating-point in the forward pass is the only possible source of
divergence between two runs.**

**The campaign's one real cross-pair coupling is not in this loop.** `ddm_jg2`
measured rate superposing at union/sum = 1.0258 (2.6%) at the ARCHIVE layer. That
is resolved at merge by the authority re-encoder over the union of accepted
edits, and the union is identical whether one process or six produced it.

---

## 2. The build

`tools/ddm_wc2_shard_solve.py`. The live solver is **not edited**.
Subcommands: `plan` · `launch` · `verify` · `compare` · `merge`.
Pins: `src/tac/tests/test_ddm_wc2_shard_solve.py` (32 tests).

`merge` unions shard checkpoints, **refuses to pick a winner on conflict** (a
disagreement falsifies the claim; resolving it silently would be the fake), and
reconstructs the payload npz from base tokens + each pair's `accepted` list --
lossless per the solver's own docstring -- with sha256. Payload reconstruction is
the DEFAULT per ALWAYS KEEP THE PAYLOAD. **Proven on real data before the swap:**
6 pairs from 3 shards, 0 conflicts, payload 8456 B sha `f89dc586433a5816`.

**Partition key (coordinator-directed).** Per-pair cost is predictable before any
pair is solved: it tracks the base flip count `base_argmax != gt` at **r = 0.967**
(jg3's own read: 0.945). Shards are LPT-bin-packed on that key, then each bucket
is RE-SORTED into seeded-permutation order -- because LPT emits heaviest-first,
and a heaviest-first prefix is exactly the biased-population shape `ddm_bp2` /
`ddm_na2` measured. Balancing decides WHICH pairs a shard owns; the permutation
decides in WHAT ORDER it visits them, so no shard prefix is biased.

Realized at the swap: shard sizes 96/96/96/95/95/95, flip loads
5705/5705/5705/5678/5678/5678, **imbalance max/mean = 1.0024**.

---

## 3. Decision equivalence -- what the replay found

Replay pairs were selected as the highest-packing-residual completed pairs, i.e.
the DENSEST frames where sites interact (coordinator point 3). Pair 519 carries
residual 22 against 1-6 typical.

### 3.1 The determinism matrix (pair 519)

| run | process | thread env | repaired | flips_after | evals | len(best) | accepted sha8 |
|---|---|---|---:|---:|---:|---:|---|
| LIVE | warm, 13th pair | unset | 62 | 85 | 564 | 266 | `b710afc5` |
| VERIFY | fresh, cold | `OMP=6` | 60 | 87 | 563 | 261 | `98c13421` |
| CTL-A | fresh, cold | `OMP=6` | 60 | 87 | 563 | 261 | `98c13421` |
| CTL-B | fresh, cold | unset | 60 | 87 | 563 | 261 | `98c13421` |

1. **Fresh processes are exactly reproducible** -- byte-identical accepted sets
   across three independent runs.
2. **My leading hypothesis was FALSIFIED.** I flagged before launching that
   `--derive-resource-budgets` sets `OMP_NUM_THREADS` while the live process
   leaves it unset, and that this was a real instrument change (`ddm_et4`).
   CTL-B ran with the environment UNSET and still produced `98c13421`. The thread
   environment is **not** the discriminator. Testing it instead of arguing it is
   the only reason this memo is not wrong.
3. **A fresh process does not reproduce a warm one.**

### 3.2 Mechanism

Not in the partition arithmetic: both runs agree exactly on
`screened_candidates` = 2928 and `packing_residual_max` = 22 -- same sites, same
candidates, same packing. What differs is downstream:

* `len(best)` 266 -> 261: **5 of 266 sites lost their Lagrangian-qualifying
  candidate**;
* unique trial signatures 23 -> 22, hence `evaluations` 564 -> 563;
* a different accepted set of the same size, at neighbouring coordinates -- live
  `(176, 98, 3)` against replay `(176, 99, 3)`.

Every screened site is by construction a pixel where the base argmax DISAGREES
with GT: a low-margin near-tie. A ~1e-6 logit perturbation flips such a pixel,
changing that site's `repaired`, hence `gain`, hence `best[]`, hence
`select_separated` and the winner. **The solver screens exactly at the argmax
decision boundary, so it amplifies any inter-process numerical difference into a
different local optimum.** This is a property of the SOLVER, not of sharding; it
would appear between any two processes with different histories.

---

## 4. Admissibility -- the measured bound

There is no canonical "correct" accepted set: the solver is a rate-aware GREEDY
descent and acceptance is REALIZED (joint render + re-segment against the frozen
scorer). Both outcomes are honestly measured. So the question is not identity but
**bias**, and it must be read in the solver's OWN objective.

**A flaw in my first metric, caught by the data.** I initially compared on
`repaired` alone. Pair 17 refuted that: the fresh run banked MORE repair (19 vs
18) using FEWER tokens (16 vs 20) -- strictly better on both axes -- which a
repair-only comparator scores as a loss. Repair is bought with tokens and tokens
cost rate, so the comparison is now scored in net delta S,
`-repaired * S_PER_SEG_CELL + (tokens * 4.1379 / 8) * S_PER_ARCHIVE_BYTE`.

**Result (`.omx/research/ddm_wc2_equivalence_receipt_20260819.json`):**

| quantity | value |
|---|---|
| pairs compared (paired, same pair both sides) | **13** |
| shard wins / losses / ties | 4 / 8 / 1 |
| sign test, two-sided | **p = 0.3877** |
| bias established at .05 | **NO** |
| repaired: live vs shard | 445 vs 428 (**-3.82%**) |
| shard advantage, 13 pairs | **-5.11e-06 S** |
| projected over 573 pairs | **-2.25e-04 S** (EXTRAPOLATION) |

Two honesty notes that both cut against overstating the penalty:

* **The sample is deliberately adversarial.** Pairs were selected for highest
  packing residual and highest cost -- the frames most susceptible to the
  cascade. This is an UPPER bound on the effect, not a field average.
* **The estimate shrank as n grew**: -1.04e-6 S/pair at n=6, -4.7e-7 at n=11,
  -3.93e-7 at n=13 -- the signature of an early sample inflated by the hardest
  pairs regressing toward the mean.

**Verdict scope.** "Sharding changes no decision" is FALSE and withdrawn. What
holds: (a) per-pair acceptance is structurally independent, so sharding adds no
ALGORITHMIC coupling; (b) the shard fleet is internally deterministic and exactly
reproducible; (c) the yield penalty is **not established** (p = 0.39) and is
bounded above at ~-2.25e-4 S, ~1.4% of the seg leg, on an adversarially-selected
sample.

---

## 5. Wall clock -- and what may NOT be quoted

**Coordinator correction, accepted in full.** Machine load is part of the
wall-clock instrument. Absolute per-pair seconds measured while several solves
shared the machine are NOT quotable -- and **three of those competing solves were
my own verification legs**, so this contamination is mine.

Explicitly withdrawn: the "17-19.6 h" log ETA, the "21-28 h" relay, and my own
intermediate means (138.87 / 169.3 / 150.3 s per pair) as absolute rates. The
only single-solver window was the live run's first ~16 pairs, before I launched
anything; even that is a small sample of a heavy-tailed distribution
(MEASURED 55.7 s to 657.0 s per pair), so no point ETA is defensible.

**What survives, because it is a RATIO and not an absolute time:**

* the machine was ~86% idle under one solver (~260% CPU of 1800%) -- itself
  measured with one solver plus a pytest arm and the dashboard resident, so
  label it *lightly contended*, not quiet;
* the render must run at batch 1 (`ddm_up2` sec.6: batch 8 is BYTE-CHANGING), so
  the serial half is structural;
* **6 workers on a machine that one worker left ~86% idle**, with a near-exact
  balance (1.0024), is the speedup claim. Sharding divides whatever the true base
  rate is; it does not need that rate to be known.

Realized contention will hold the fleet below a linear 6x (6 x ~260% = ~15.6 of
18 cores). The honest expectation is **~4-5x**, and the true post-swap rate
should be re-measured from the shard checkpoints in the now-steady 6-worker
window rather than projected from the contaminated one.

---

## 6. Profile decomposition (secondary deliverable)

Fitted over the completed pairs from the checkpoint, no new run required:

```
seconds = 3.59 + 0.5123 * evaluations        R2 = 0.9890
```

**Contamination caveat:** the pairs behind this fit span both single-solver and
contended windows, so the SLOPE (0.512 s/evaluation) is inflated by an unknown
amount and is not quotable as an absolute per-evaluation cost.

**What survives the caveat**, because both are ratios or correlations:

* **97.9%** of per-pair wall clock is evaluation-driven; fixed cost is 2.1%. The
  dominant term is the evaluation COUNT.
* `evaluations` correlates with `flips_before` at **r = 0.967** -- the property
  the shard balancer actually uses.
* Within one evaluation, jg1's own numbers put the batch-1 render at 0.228 s
  against SegNet 0.206 s at batch 8, i.e. the render is ~44.5% and is
  IRREDUCIBLE, because byte-identity requires batch 1.

**Routed recommendation, NOT adopted.** The knobs that cut evaluations are
`--max-candidates-per-site`, `--site-budget`, `--max-sites`. The solver's own
comment already argues the first is principled: the per-move cost distribution is
wide (p10 -2.344, p25 +0.902, p90 +12.443 bits), so the expensive tail can almost
never clear the Lagrangian test. A candidate cap priced in accept-margin terms is
the natural next wall-clock lever.

**Deliberately out of scope** (coordinator point 4): it is a SCOPE reduction with
an unmeasured yield cost, whereas sharding is the pure wall-clock lever. Keeping
them separate is what makes this result interpretable. Any future adoption owes a
measured yield-vs-wall-clock trade at the n=3 rung before n600.

---

## 7. The bias0/1/2 exit -- owed account

**They did not fail. I killed them, and I under-narrated it.**

| evidence | reading |
|---|---|
| `rc = 241` | `241` = `-15` as an unsigned byte = **SIGTERM** |
| receipts `generated_utc` all `14:46:31Z` | one simultaneous kill |
| launches staggered `14:37:35 / :38 / :41` | elapsed 536/533/530 s mirrors the 6 s/3 s launch stagger exactly |
| safe_run `status: ok`, `timeout_s: 7200` unreached | safe_run did NOT cap, time out, or kill |
| my kill loop printed `SIGTERM bias0 -> pid 49830` etc. | `ps` found all three ALIVE at that instant |

"Lockstep at ~530 s" is the signature of ONE simultaneous SIGTERM against
staggered launches, not three independent deaths. I stopped them on purpose: they
were re-solving pairs that were already banked -- useful only as paired controls,
worthless as progress -- and were taking CPU from the live solve while it was
still the only thing making forward progress. Having gathered n=13, their marginal
evidence no longer justified the contention.

**Governor admission.** Yes, admitted. Each of the three, and each of the six
production shards, launched through
`tools/launch_detached_process.py --derive-resource-budgets`, which enforces the
canonical 116 GiB host ceiling through `safe_run` and derives the thread budget
from a measured need. All returned rc=0 at launch; all `safe_run` statuses read
`status: ok`. No governor refusal occurred, so no admitted-count constraint was
imposed on the shard count. Memory was never near the ceiling: 90.7 GiB available
with 5 solvers resident.

---

## 8. Swap receipt

* live pid 81175 SIGTERM'd; **exited cleanly**; checkpoint **27/27 rows parse
  valid**; 726 cells repaired banked; **zero pair loss** (an interrupted pair was
  never written and simply returns to the remaining set).
* 6 shards launched, all rc=0, all confirmed running at 162-221% CPU with
  distinct tags and disjoint pair lists.
* new pids recorded for MAIN in
  `/Volumes/APDataStore/pact/ddm_jg3/logs/SWAP_NOTE.txt`:
  `n600_wc2s0=67213 · s1=67277 · s2=67344 · s3=67430 · s4=67541 · s5=67661`.
* merge when the fleet drains:

```
.venv/bin/python tools/ddm_wc2_shard_solve.py merge \
  --store /Volumes/APDataStore/pact/ddm_jg3 \
  --tags n600,n600_wc2s0,n600_wc2s1,n600_wc2s2,n600_wc2s3,n600_wc2s4,n600_wc2s5 \
  --out-tag n600_merged
```

## 9. Receipts

* orchestrator `tools/ddm_wc2_shard_solve.py` · pins
  `src/tac/tests/test_ddm_wc2_shard_solve.py` (32 passing)
* admissibility receipt `.omx/research/ddm_wc2_equivalence_receipt_20260819.json`
* SSD tier: `/Volumes/APDataStore/pact/ddm_jg3/logs/wc2_*`, checkpoints
  `seg_solve_n600_wc2s{0..5}.jsonl`, controls `seg_solve_ctlA_pinned6.jsonl`,
  `seg_solve_ctlB_liveenv.jsonl`, `seg_solve_bias{0,1,2}.jsonl`,
  `seg_solve_n600_wc2verify.jsonl`
