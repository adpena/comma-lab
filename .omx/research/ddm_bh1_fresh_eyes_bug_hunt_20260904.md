# ddm_bh1 — fresh-eyes bug hunt over the surfaces that produced this wave's closures

Owner: Opus arm (ddm_bh1). Charter: `.omx/research/charters/ddm_bh1_fresh_eyes_bug_hunt_20260904.md` (5b696e9e5).
Reference commit `820db413ea1186ec525629875c7805389d0e9f0c`. Tokens `[no-triality] [p0-ledger-ok]`.
Axis: source reads + $0 CPU arithmetic on retained artifacts. **No scorer forward, no training, no pointer move.**
Operator's words: *"Likely bugs in a lot of places."* Six surfaces hunted; **15 findings, 2 fixed and committed**.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED** (a hunt cannot move it).

---

## 1. The table

| # | surface | finding | path:line | severity | reproducing check | fix-or-charter |
|---|---|---|---|---|---|---|
| **1** | GT lineage | **The born trainer TRAINS and SCORES against the PyAV table — and the seal PINS it by sha.** `GT_CACHE` is the only GT constant in the module; `_target_arrays` is the only loader; the milestone evaluator uses the same call. | `ddm_qbt1_qbflow_trainer.py:123` (const), `:246` (`PINNED_SHA256["gt_cache"]=cf8d8360…`), `:2067-2073` (loader); milestone `ddm_qbr1_born_fairform_burn_prep.py:615` | **measurement-corrupting (bounded)** | `shasum -a 256 experiments/results/mlx_fleet_gt_cache/gt_n600.npz` = the pinned sha; `ddm_ft1_identity_gate_and_caches.py:92-101` labels that file PyAV | **CHARTER** — cure below is $0 |
| **2** | GT lineage | **The lineage preflight gate cannot see it.** Both artifact regexes require `.npy`/`.pt`; no `.npz` can match, and `gt_n600` does not start with `gt_cache_`. The gate is also WARN-ONLY. Its own rule chain names exactly this harm. | `src/tac/preflight.py:2466-2469` (patterns), `:2863` (`strict=False`, "deliberately NOT raised") | **measurement-corrupting (apparatus)** | I ran both regexes against the path: `[False, False]` | **CHARTER** (widening it floods ~372 historical consumers; needs a live-count plan first) |
| **3** | population | **The n32 selection understates the lineage fork 26–40×, then HT-expands to an n600 claim.** DALI-vs-PyAV `d_seg` offset is 1.011–1.017× on the n32 selection vs **1.4425×** on n600. HT weights correct the *stratum* design; they cannot correct *lineage-fork composition*. | `.omx/research/ddm_sd1_surrogate_exact_decoupling_20260904.md:294-296` (sd1's own caveat) | **verdict-changing** | read sd1 :294-296 beside `qbr1:643-647` (`population_n: 600`) | **CHARTER** |
| **4** | objective | **The dual constraint measures an UNWEIGHTED quantity but penalises an HT-WEIGHTED one.** `realized_within_class_error` pools pixels with no sample weights; the penalty it drives (`per_class_expected_flip_margin_loss`) is HT-weighted. The bound is a population target; the driver is a composition-biased estimate. | `ddm_qbt1_qbflow_trainer.py:598-612` (unweighted) vs `:568-596` (weighted), joined at `ddm_qbr1…:729-742` | **verdict-changing** | §3 below — I MEASURED the heavy-stratum share gap: **1.60× Lane, 1.39× Movable** | **CHARTER** (ng2/ng3 are live; changing it moves their trajectory) |
| **5** | apparatus | **The δ_R producer still defaulted to the n96 prefix that MADE the retired constant.** ql2/ql3 cured every consumer; the producer's `--gt-npz`/`--n` defaults stayed on the prefix, so one flagless re-run regenerated `0.019590163230895963`. | `tools/measure_delta_R_noise_floor.py:99,102` (pre-fix) | **verdict-changing** | 2 new tests fail pre-fix, pass post-fix | **FIXED** `524d7ff13` |
| **6** | fire/seal | **The fire scripts' memory guard uses `free + ALL inactive` as "reclaimable" — the exact basis the repo's CLASS-1 fix refuses.** It re-implements `psutil.available` in bash, outside the scanned tree, and it is the gate that ADMITS concurrent Metal cells. | `/Volumes/APDataStore/pact/ddm_ng3_tau_band/fire/fire_ng3_tau_cell.sh:7` (and the ng2 twin); law `tools/mem_basis.py:1-24` | **verdict-changing (live risk)** | §4 below — MEASURED over-trust **1.204×** now; the law's own anchor measured **4.2×** under load | **CHARTER** (cannot touch live arms' custody) |
| **7** | GT lineage | Axis strings on `qbt1`/`qbr1`/`ng1`/`ng3` name device and sample but **never the GT lineage**. ng2 alone declares it. | `qbt1:2189`; `qbr1:644`; `ng2…:203-204` (the good example) | **verdict-changing** | grep the `"axis"` literals | **CHARTER** — one-line each |
| **8** | fire/seal | **A pin that can never fail.** `verify_pins()` synthesizes the `wd3_reference` row with a fixed sha and a hardcoded `bytes: 145_956` when the file is ABSENT, so `run_training`'s `config["source_pins"] != pins` check is vacuous for that row, and the reseal tool's content-identity claim covers it without verifying it. | `ddm_qbt1_qbflow_trainer.py:310-319`; consumed by `ddm_qbr1…:141-142`, then `ddm_reseal…:reroot` | hygiene (provenance) | delete/rename the WD3 file → `verify_pins()` still returns an identical row | **CHARTER** (changing the row's shape would break the live cells' pin equality) |
| **9** | fire/seal | **The reseal receipt attested to bytes it never read.** `config_in` was re-read for its sha AFTER `config_out` was written, and nothing refused `--config-out == --config-in`. | `ddm_reseal_pins_inside_sealed_tree.py` (pre-fix) | hygiene (provenance-corrupting) | §2 below — demonstrated: in-place, the receipt's `config_in.sha256` **equals** the output sha | **FIXED** `803e7315a` |
| **10** | closure arithmetic | lb1's `5.5513×` applies md1's **born-field** 1.61× credit ceiling to a **post-oracle** field whose error composition has changed (post-oracle reachable share 47.37%, not 37.99%). | `.omx/research/ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md:244`; `SUMMARY.json.residual_after_oracle_and_schedule_levers_x` | hygiene | `8.937533224486373/5.551262872351784 = 1.61` exactly; the sister row implies 1.9002 | **CHARTER** — see §5, the two rows are correctly *ordered*, the constant is the wrong regime |
| **11** | closure arithmetic | md1's Lane enrichment **51.50×** mixes an n32 numerator with an n600 denominator (like-for-like 51.315×); **Movable 1.62× is not reproducible** (recompute 1.864×; no documented denominator yields 1.62). | `ddm_md1_micro_to_macro_dynamics_20260904.md:73` (hand division; not computed in the script) | hygiene / one unreproducible | recompute from `gt_class_histogram` + `classes.PERSISTENT` | **CHARTER** |
| **12** | closure arithmetic | gm1's `bottom-k@0.05` "**EXACTLY** inert / 0.0% everywhere" is **0.042%** max relative removal (its own table already prints "+0.1%"). | `ddm_gm1_…md:2,55,237,418,431` | hygiene | recompute `row1_grad` vs `grad` over the 45 bins | **CHARTER** |
| **13** | closure arithmetic | md1's "45.7% by painting Lane over Road" is the **pred-Lane share** (45.69%); the actual Road→Lane **edge** is 45.18%. The memo's own table at `:233` has them as separate columns. | `ddm_md1_…md:438` | hygiene | `born_pred_class_histogram_at_peak` vs `born_edges_at_peak` | **CHARTER** |
| **14** | closure arithmetic | pr1 payload custody: `sha256` hashes `array.tobytes()` (4,800 B) while `bytes` is the `.npy` file size (4,928 B) — a verifier who hashes the file gets a mismatch. | `ddm_pr1_pose_resolve_on_renderer_change.py:103-104` | hygiene | `sha256sum` the `.npy` vs the recorded sha | **CHARTER** |
| **15** | milestone | **Reading caveat, not a defect:** the EMA is sealed to CONSTANT decay (warmup OFF), `decay = 0.9990793899844618`, time constant **1086.24 updates** — so a milestone's shadow still carries the common r10 init at **39.8% / 15.8% / 6.3% / 1.0%** of weight at steps 1k/2k/3k/5k. Between-cell *differences* are init-free (it cancels), but absolute `S_hat` at early milestones is pulled toward the start. | `qbt1:2232-2258` (`resolve_ema_law`), `src/tac/training.py:504-557` | hygiene (interpretation) | §6 below | **record only** — this confirms md1's "1,086-update low-pass" exactly |

---

## 2. The reseal receipt lie — MEASURED, then fixed (`803e7315a`)

Pre-fix, `reroot` read `config_in` a second time for the receipt sha **after** `config_out.write_text`, and nothing
refused an in-place re-root. Run against `HEAD~` with a stubbed sealed tree:

```
true input sha        : 9e9213bf8f5ec55e1a6a0176070041ac5168d2edb499381d6cd090ee1535a424
receipt config_in sha : 9752ff60658aa1e3ca2865d334581eb6d67f9376de7a31de51b20e3b2e55761e
receipt config_out sha: 9752ff60658aa1e3ca2865d334581eb6d67f9376de7a31de51b20e3b2e55761e
RECEIPT LIES ABOUT INPUT: True | config_in==config_out in receipt: True
```

The receipt was internally self-contradictory — it listed `paths_rerooted` as non-empty while claiming input and
output were the same bytes. **Cure:** input read ONCE before any write; the three paths must be pairwise distinct on
their **resolved** form (so two spellings cannot alias); both writes atomic (tmp + `os.replace`) with **UTF-8 pinned**
(bare `write_text` encodes with the process locale while the receipt's sha is taken over `text.encode()` — on a
non-UTF-8 host the receipt would attest to bytes the file does not hold; that second defect was in my own first patch
and the second review pass caught it). **17 tests; 6 fail against the pre-fix file.** The tool had no tests at all.

## 3. The dual-constraint estimator gap — MEASURED

`realized_within_class_error` returns `mean(predicted[target_mask] != c)` — pixels pooled with **no** sample weights.
The λ it drives multiplies `per_class_expected_flip_margin_loss`, which **is** HT-weighted. Measuring the heavy
stratum's share of each estimator over the two sealed training chunks (a $0 memmap read of `lstars`, no scorer):

| class | chunk | heavy-stratum share, UNWEIGHTED (the driver) | heavy-stratum share, HT (the penalty) | gap |
|---|---|---:|---:|---:|
| Lane | 0 | 0.2384 | 0.3850 | **1.615×** |
| Lane | 1 | 0.2522 | 0.4028 | **1.597×** |
| Movable | 0 | 0.4420 | 0.6131 | **1.387×** |
| Movable | 1 | 0.4354 | 0.6067 | **1.393×** |

The 8 heavy pairs carry weight 30 and represent 240/600 = **40%** of the population; the driver gives them 24–25%
(Lane). The dual therefore chases a different estimator of the same population quantity than the one it penalises.
It propagates: ng2's `measured_birth_force` is `median(100·λ^dual)` over the control's history
(`ddm_ng2_area_cap_cell.py:92-135`), so λ_Lane 2799.8 / λ_Movable 7587.4 inherit the bias. **Not fixed** — ng2 and ng3
are live on the Metal and this would move their trajectory.

## 4. The memory guard — MEASURED

`fire_ng3_tau_cell.sh:7` computes `RECLAIM = free + inactive` and admits when `RECLAIM ≥ peak + 16`. The repo's
canonical law says verbatim (`tools/mem_basis.py:5-11`): *"on macOS `psutil.virtual_memory().available` = (free +
inactive) — it counts DIRTY ANONYMOUS pages parked in the inactive queue as 'available' even though evicting them
needs swap … MEASURED live 2026-07-17 … `.available` = 57.3 GiB but the truly reclaimable-without-swap figure = 13.7
GiB"*, and *"the legacy raw psutil basis is what the CLASS-1 preflight gate refuses outside this module."* The bash
one-liner is that refused basis, re-implemented where the Python gate cannot reach it. Measured on this host during
the live burn:

```
free 2.82  inactive 15.70  file-backed 11.77  purgeable 0.79  anon 15.50  (GiB, 16384-B pages)
FIRE SCRIPT basis  free+inactive          = 18.51 GiB
CANONICAL basis    free+file_backed+purge = 15.38 GiB   (governor agrees: 15.3729)
over-trust = +3.14 GiB (1.204x);  7.32 GiB already resident in the compressor
```

Two compounding defects: (a) the basis over-trusts dirty inactive anon; (b) the guard adds a flat 16 GiB margin and
never subtracts the **resident footprint of the already-running cell** — and a running cell's dirty anon pages parked
on the inactive queue are counted by this guard as headroom for the next one. That is precisely the concurrency
decision the script's own comment authorizes. **Cure:** route the guard through
`tools/mem_basis.conservative_free_gib`, and subtract concurrent cells' peaks rather than adding a flat margin.
Correct on the small stuff: the 16384-B page size is right for arm64, the awk field index is right, and integer
truncation errs toward refusing. Sister: the n205 OOM (a 5,000-step run died with no checkpoint).

## 5. Closure arithmetic — every load-bearing number reproduces

Recomputed independently from the stored artifacts with our own code (no import of the arms' instruments):

* **md1 persistent fraction 62.011% → 62.010692%** (denominator `0.0028065999348958334`, bit-exact); 11,842 sites,
  64.79%, 33.69%, floor 0.0017403920, 12.75×, 4.233%, 2.21×, 24,336, 1.29558, 1.61× — **all reproduce**.
* **lb1 0.909 / 0.0996 / 93.5% / 0.5651 / 63.12% / 8.94× / 4.70× / 2,832 B / +0.008481 S** — all reproduce, and the
  break-even formula was **derived independently from first principles and is CORRECT**: for a lever that overrides
  the incumbent on the set it claims, `p* = b/(a+b)` with `a = P(inc wrong | C)`, `b = P(inc correct | ¬C)`, matching
  the published 0.9092466319717002 to 16 digits. (Caveat on *use*, not on the number: `a`,`b` are population rates,
  not claim-set-conditional, so 0.909 is conservative against lb1's own negative — and the verdict rests on 0/162
  configurations improving, a direct measurement.)
* **pr1 13.82 / 228.45 / 16.42× / 9.43e-4 / 598 / +125 B / 21.55×** — all reproduce; the **−1.032e-4 selector
  projection reproduces to 10 s.f.** (`−1.0321261e-04 → 0.14787295862215796` vs published `…740366`, diff 5.2e-12),
  its parts check out (ΔS_pose −1.271836e-04, ΔS_rate +2.397092e-05, ΔB +36 B), and the 17-digit S base decomposes
  exactly. **The 39/600 gains were MEASURED on all 600 pairs, not extrapolated** — no subset-scaled-to-population
  error.
* **gm1 77.715% and the 45.6 / 77.7 / 96.8% band** — reproduce exactly.

Corrections are #10–#14 above: four hand-computed side numbers and three prose overstatements. **None changes a
verdict.** On #10 specifically — I checked the sub-arm's framing and it overreached: 5.5513× (carrier + schedule
ceiling) and 4.7036× (carrier + *perfect* optimization) are correctly **ordered**, so they are not contradictory
answers to one question. The real defect is narrower and is genus `[[m143]]`: the 1.61× was derived on the born field
(reachable 37.99%) and applied to the post-oracle field (reachable 47.37%) — the wrong regime's constant, which
*understates* the credit.

## 6. Verified CLEAN — recorded so nobody re-checks

`tau_for_step` has **no** off-by-one (step 0 → `start`, step `T-1` → `end`) · the camera STE
(`roundtrip_to_camera_uint8_ste`) and the area STE (`realized_class_area_ste`) both carry the exact value with the
right soft gradient · `build_initial_state` loads r10's EMA shadow into **both** the live weights and the shadow ·
`schedule_for_seed` is balanced (both chunks per 2-step epoch) and `training_chunks` are equal-mass (300 each) · the
milestone evaluates under `ema_scope` and `reencode_inference_state` takes `state=ema.shadow` explicitly — **no
live/shadow mismatch** · **the retained argmax is taken from the float32 logits before the float16 cast**, so sd1's
storage caveat does not corrupt it (`qbt1:2120-2121`) · scorer preprocessing is upstream's own, so the
`interpolate → rgb_to_yuv6` order is correct by construction · the HT estimators are right (`_weighted_mean` divides
by N with Σw = 600 exactly; `b_var_hat` is a **total**, correctly not divided) · the HT weights are applied to the
loss as a *normalised* weighted mean, which re-aims the objective without inflating gradient scale · `no2_gate` fails
closed and labels its unweighted fallback · `_maximum_rss_bytes` handles Darwin-bytes vs Linux-KiB correctly ·
`MEMORY_CEILING_BYTES` = 116.0 GiB as documented · ng3's band takes effect at **step 0** and `m_safe = 2·δ_R` exactly
(0.04376363754272461 / 0.021881818771362305) · **ng2 already disclosed its cap-gradient caveat honestly** (memo
:251-269: cap/recall 0.0125, parity multiplier ×80.02–80.05, "this lowers my own prior") — I re-derived the mechanism
and found no defect beyond what ng2 states · the δ_R census by VALUE finds **no live surface** on the retired
constant and **no third value** (one stale comment, `spec_v9_cgauge.py:1130`).

---

## 7. What this invalidates, and what must be re-measured

**Finding #1 touches ng1 / ng2 / ng3 and every QBR1 excursion reading** — their `d_seg_hat`, `S_hat`, and the
falsifier thresholds are PyAV-target, n32-composition numbers. Because each cell is a **relative** comparison against
a control on the **same** target, the lineage error largely cancels for the **sign** of each verdict. What does *not*
transfer: (a) absolute `d_seg_hat` / `S_hat` levels, which must never be read against the 0.12 target or against a
DALI row; (b) the falsifiers 0.425149 / 0.485677, valid inside the PyAV/n32 frame they were pre-registered in and
invalid as progress-to-target; (c) any n32→n600 lineage transfer, blocked by #3.

**Unaffected:** qn1 and br2 (they assert `AUTHORITY_LINEAGE` and score on DALI), lb1 (DALI via md1), md1
(dual-lineage by construction), gc1 and gf2 (no GT reference at all — rate/form arms).

**The cheapest honest cure is not a re-run.** `_retain_eval_outputs` already persists `segnet_argmax_u8` and
`target_argmax_u8` per pair per milestone (`qbt1:2109-2140`), so a DALI `d_seg_hat` for every existing milestone is a
**$0 CPU re-read of retained bytes** — no re-training, no scorer forward. That converts #1 from an open exposure into
a measured number. It is the single highest-value follow-on this hunt found.

## 8. Fixed and committed

| commit | what |
|---|---|
| `803e7315a` | reseal tool: receipt read-after-write, in-place/aliased refusal, atomic UTF-8 writes; **17 tests** (6 fail pre-fix) |
| `524d7ff13` | δ_R producer: defaults moved off the n96 prefix to the n600 population; **3 tests** (2 fail pre-fix) |
| `5ee24191a` | equations leg (below) |

## 9. Equations leg (`tac.canonical_equations`)

`annulus_restricted_prefix_bias_detector_v1` gains a second predicate and a second anchor:

* **`producer_default_reinfects_cured_constant(consumers_cured, producer_default_cohort_is_prefix)`** — THE
  RE-INFECTION SCREEN. A retired prefix constant is re-infectable whenever the consumers were cured while the
  **producer** that measured it still defaults to the prefix cohort. A constant census answers *"what do the consumers
  hold?"*; it cannot answer *"what will the producer emit next time?"* — two independent screens, and the cure is not
  finished until both return False.
* **Anchor `bh1_producer_default_still_the_prefix_after_consumer_cure_20260904`** — predicted 0 producers still on the
  prefix, **measured 1**, residual **1.0**; falsifier fired. Cure landed in the same wave with three regression tests.

**Owed:** the ledger append to `.omx/state/canonical_equations_registry.jsonl` — NOT taken, deliberately. That file
carries uncommitted changes from live arms, and staging it would absorb their in-flight work (the absorption bug
class). The law itself is in git, tested, and reviewed; MAIN should append when the ledger is clean.

## 10. Verdict scope

Every finding is scoped to the **instance** it was measured on. #4 and #6 are measured on this vehicle and this host;
#1, #2, #5 are source facts, not statistical claims. **No finding here is a family verdict**, and none of them moves
the pointer. The two fixes are apparatus, not score. Said plainly per the means/ends firewall: **this arm bought no
exact row.**

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED.**
