# DDM NG3 — the tau band cell: the loss stops paying for structure the round trip erases

**Date:** 2026-09-04
**Arm:** `ddm_ng3_tau_band_cell`
**Axis:** `[seal + bounded macOS-CPU mechanism smoke only; no Metal, no Modal, no contest eval]`
**Disposition:** **SEALED / BOUNDED-SMOKE-PASS ×2 / BURN-NOT-FIRED — MAIN fires.**

## Result first

The cell is sealed and the lever is exactly two numbers. Three things I had to correct on the way,
and each is a correction to something this arm was TOLD, not to something it found convenient.

1. **The tau gate the charter told me to extend does not guard this path.** ng1 recorded that "tau
   geometry is structurally frozen (any other refused)" at the trainer's own validator, and the
   charter inherited that. MEASURED: `ddm_qbr1_born_fairform_burn_prep.py` **never calls**
   `qbt.validate_config` — it validates through its own `validate_config` (`:303`), whose field set
   is disjoint from qbt's. So until this landing **any** tau pair reached `tau_for_step` unchecked.
   That is a hole, not a gate. The band would have run through it silently. I closed it
   (`validate_tau_band_block`) **and** widened the literal pin the charter named, so the two
   validators do not contradict each other.

2. **The charter's falsifier 3 imports a number from outside the band's own range.** It asks for a
   Lane gradient-share drop of "1.6–2.1×", citing gm1's headline. Read at source, that headline is
   gm1's own sentence *"as tau falls 0.15 → **0.5 δ_R**"* — the 0.5·δ_R column. **This band runs
   2·δ_R → 1·δ_R and never reaches 0.5·δ_R.** Recomputed from gm1's own table at the temperatures
   this band visits: **1.281–1.592× at τ_start, 1.496–1.903× at τ_end.** The correction runs in the
   direction that FLATTERS this cell (the band is gentler on Lane than pre-registered), which is
   exactly why it is in the source, asserted by a test, and stated here before the burn.

3. **My own seal was not byte-stable, and my own check caught it — twice.** The first seal produced
   two shas from two identical compiles: the LawRef manifest carried a `resolved_at` timestamp
   inside the config. Cured at the DSL compile boundary. The determinism check then found a
   **second** volatile field, `ema.lawref.resolved_at`, which the QBR1 lineage has always kept
   inside the config — and for which `qbt.stable_ema_law_identity` is already the sanctioned
   comparator. I used the lineage's existing rule rather than inventing a second one, and the seal
   receipt now REPORTS the residual instead of claiming a property that does not hold: **the config
   sha is a FILE hash MAIN verifies with `shasum`, never by recompiling.**

The lever is proven to be exactly the temperature: at a shared τ = 0.15 the band cell's objective
and the control's are **bit-for-bit identical in every component** (`loss_total`
1.0765775442123413 both), while at the band's own τ it is 0.6409480571746826 — so the test is not
passing because the objective is tau-blind.

## Verified at source (every premise carries `path:line`)

| claim | evidence | label |
|---|---|---|
| the seg term is `sigmoid(-margin/tau)`, pixel-mean, HT-weighted | `experiments/ddm_qbt1_qbflow_trainer.py:544-565` | MEASURED (gm1, re-read) |
| `tau_for_step` anneals linearly and takes `start`/`end` as arguments | `ddm_qbt1_qbflow_trainer.py:685-688` (was `:643` pre-landing) | MEASURED |
| the QBR1 cell reads the band from its CONFIG and passes it to `tau_for_step` — the mechanism needs NO trainer change | `ddm_qbr1_born_fairform_burn_prep.py:722-728` (train) and `:908-914` (resume smoke) | MEASURED |
| the trainer's tau check pins the literal pair | `ddm_qbt1_qbflow_trainer.py:2522-2526` (was `:2481-2483`); resolver at `:649` | MEASURED |
| **that check is NEVER called on a QBR1 cell** — `qbr1` calls only its own `validate_config`; zero `qbt.validate_config` callers outside the trainer's own tests | `ddm_qbr1_born_fairform_burn_prep.py:273,667,987,1358` all resolve to its own `:405`; repo-wide grep finds no `qbt.validate_config` caller outside the trainer's own tests | MEASURED (this arm) |
| `qbr1.validate_config` checked arm/pairs/schedule/EMA/initial-state/pins/area-cap and **nothing about tau** | `ddm_qbr1_born_fairform_burn_prep.py:405-441`, pre-landing body | MEASURED (this arm) |
| `m_safe = headroom·δ_R` resolved THROUGH the law: `delta_r 0.021881818771362305`, `headroom 2.0`, `m_safe 0.04376363754272461`, `n_frames 600`, artifact `reports/delta_R_noise_floor_n600.json`, `artifact_fallback_used False`, `lawref_fallback_used False` | `tac.canonical_equations.margin_band_satisficing_threshold_20260712.resolve_margin_band_threshold`, live call | MEASURED |
| `qbt.stable_ema_law_identity` already pops `lawref.resolved_at` — the lineage's own volatile-field rule | `ddm_qbt1_qbflow_trainer.py:2261-2264` |  MEASURED (this arm) |
| `config_identity` ignores `action/output/resume_from/launch_authorized/scorer_lane/metal_lane`, so `tau_band` and both tau scalars ARE part of the identity | `ddm_qbr1_born_fairform_burn_prep.py:174-183` | MEASURED |
| the cold control of record (six milestones) | ng1 memo `.omx/research/ddm_ng1_warm_transition_burn_design_20260904.md`, read back from `runs/seed_20260902/control_native100/milestones/step_*/MILESTONE.json` | TRANSFERRED (ng1, read-only) |
| 77.7% of the seg gradient at τ=0.15 is correct-and-outside `m_safe`; the band removes 45.6% (2δ_R) / 77.7% (1δ_R) of it | gm1 memo §1, §5 | TRANSFERRED (gm1) |
| Lane spends 87.9–89.8% of its gradient outside `m_safe` and holds ~90.1% of the rate demand | gm1 §2, `[[m131]]` | TRANSFERRED |
| the fixed-τ telemetry row already exists in the working tree (ng2), and is byte-neutral | `ddm_qbr1_born_fairform_burn_prep.py` `fairform_objective` tail; ng2 memo | MEASURED (re-measured here) |

## The band, derived rather than picked

| | shipped (legacy) | ng3 (msafe_band) |
|---|---|---|
| τ_start | 0.15 | **0.04376363754272461** = `m_safe` = 2·δ_R |
| τ_end | 0.05 | **0.021881818771362305** = δ_R |
| in δ_R units | 6.855006047134966 → 2.2850020157116555 | **2.0 → 1.0** |
| provenance | two free literals; nothing derives them | `margin_band_satisficing_threshold_v1`, resolved at compile time |

Why these two ends (DERIVED, gm1 §5, re-stated because the numbers must travel with the choice):

* **τ_start = `m_safe`.** At τ = `m_safe` the surrogate's soft band is MATCHED to the satisficing
  band instead of being 4–12× wider than it. Outside `m_safe` the uint8 round trip cannot flip the
  pixel, so gradient spent there buys nothing the score can see.
* **τ_end = δ_R.** δ_R is the physical floor of "decided". Below it the loss optimizes inside the
  band where the round trip's own noise picks the class.
* **The ratio stays 2:1**, so the coarse→fine anneal SURVIVES; only its scale is re-based.

**No decimal is retyped anywhere.** `test_no_arm_source_carries_a_delta_r_or_m_safe_literal_in_CODE`
parses each touched file, strips docstrings, and fails if any of the four historical δ_R / `m_safe`
decimals (n600 or the retired n96 pair) appears in executable code. The arm's own script is held to
the stricter rule: not even in a docstring.

## How the band enters, and the gate it now passes

Three layers, and the middle one did not exist before this landing:

1. **DSL (the lever).** `ExpectedFlipTauBandMsafe(mode="msafe_band")` in
   `src/tac/witness_dsl/curriculum_dsl.py` resolves the law and emits `expected_flip_tau.*`
   overrides; `compile_qbr1_tau_band_config` turns them into the `tau_band` provenance block plus
   the two top-level scalars the trainer reads. `mode="legacy"` emits the historical pair from the
   same factory, so a control is never a hand-written config. Sister of ng2's
   `AreaCapBornRareClass` / `compile_qbr1_area_cap_config` — same vehicle, same compile target.
2. **The QBR1 gate (new).** `validate_tau_band_block` (`ddm_qbr1_born_fairform_burn_prep.py:321`, called from `validate_config` at `:405`) admits exactly two states: **no block ⇒ the
   legacy literal pair** (so every sealed cell predating ng3, including the live chain's, validates
   unchanged), or **a block ⇒ every scalar re-derived through the law**, `headroom` pinned to the
   law's DERIVED default, `n_frames` matched, the WAIVER fallback refused, and the two scalars the
   trainer reads required to equal the block's own endpoints. A hand-edited temperature its stated
   law does not imply is refused, exactly as `validate_area_cap_block` refuses a hand-edited
   stiffness.
3. **The trainer gate (widened, not bypassed).** `admissible_expected_flip_tau_bands()` returns the
   legacy pair AND the law-resolved band. Widened by exactly one point, and that point is not a new
   literal — it is the law's live output. A third band is still refused. The composition is
   fail-closed in the right direction: this gate is the permissive one and caches the resolution;
   the QBR1 gate re-resolves fresh on every call and adds the fallback/population checks, so a
   regenerated artifact makes the strict gate refuse, never the permissive one wave something
   through.

**A class fix fell out of this.** `--dsl-lever NAME` composes onto trainer ARGV, and the registry
decided composability by "does the factory take required arguments?". ng2's config-surface lever is
excluded only ACCIDENTALLY, by having three required args; mine has all-default parameters, slipped
straight into the composable set, and failed the CI parse-test with an `unrecognized arguments`
error naming neither the lever nor the reason. `_composability_check` now excludes by the SHAPE of
what a lever emits — a Lever whose overrides are not `--flags` targets a compiled-JSON cell and
cannot be argv at all. MEASURED: the set drops 108 → 107, exactly this lever; ng2's stays excluded
now for the right reason; a lever with no overrides at all stays composable.

## The single lever, proven

Both configs are compiled fresh through `qbr1.compile_cell` in this tree, so their pins are this
tree's pins and only the lever separates them.

```
differing keys: ["cell_id", "expected_flip_tau_end", "expected_flip_tau_start", "output", "tau_band"]
allowed       : the same five
```

Held identical and asserted: `objective` (native_interface_weight 100), `ema`, `schedule`,
`initial_state`, `learning_rate` 2e-4, `margin_constraints`, `pair_ids`, `selection_weights`,
`total_steps` 5,000, `milestones`, `seed` 20260902, `chunk_pairs` 16, `device`, `source_pins`,
and `resume_from` = null (**COLD** — the same transition the control took, so the pair isolates the
band and never composes it with ng1's warm lever). **`area_cap` is `None` and the seal refuses a
cell that carries one**: ng2's cap and this band are separate levers, and union is not the sum of
legs (`[[m164]]`, 3.705×).

### The pin delta — and why a same-pins twin was never available

| pin | sealed control (`106d0dd0…`) | ng3 |
|---|---|---|
| `qbt_trainer` | `6eda9c20…` | `c8ff9dbd…` |
| `qbt_packet_schema` | `5405ccd4…` | `7fe5285f…` |

The charter offered "seal from the working tree as a same-pins twin of the cold control **if** the
band is config-only". The band IS config-only on the mechanism side — but the premise fails for a
reason that predates this arm: **ng2 already moved the trainer pin** (a new loss term), and MAIN's
`4a7ae5ca0` already moved the packet-schema pin. A same-pins twin was off the table before ng3
touched anything. ng3 moves the trainer pin once more, by widening the tau check. Everything else
is byte-identical; `uncommitted_diffstat_at_seal` is empty (sealed at a clean committed revision).

**What licenses the comparison anyway is a MEASUREMENT, not an argument** — see the smoke's second
row below.

## Bounded CPU smoke — PASS, run twice, identical to the last digit

The Metal lane is held by the live QBR1 chain (`pgrep -f ddm_qbr1_cell_chain` → 95296/95299/95317
throughout), so both segments ran on **CPU**; `_run_resume_smoke_segment` forces `device="cpu"`
itself. This arm did **not** touch `.omx/state/active_lane_dispatch_claims.md` — the chain reads it
every poll and a malformed edit raises `CLAIMS_UNREADABLE` and refuses an 18-hour burn. 0 Metal /
0 Modal / 0 contest-eval invocations. Governed through `tools/launch_detached_process.py` (the
launcher REFUSED the first attempt for a done-receipt name already reserved by the live chain —
the governor working, and the reason the receipt is named `NG3_SMOKE_DONE.json`).

| check | result |
|---|---|
| **NO-OP DETECTOR** — band step-1 vs control step-1 live state | **DIFFERENT** (`723f1c44ff73d375…` vs `27f514180db2b4cd…`) |
| — and their re-encoded archives differ | yes, 106,709 B (`f50abfca5aff2d21…`) vs 106,676 B (`236b9e0135f89d9f…`) |
| **THE TRAINING PATH IS UNMOVED BY THIS LANDING** — control step-1 vs ng1's PRE-telemetry cold reference | **BIT-IDENTICAL** (`27f514180db2b4cda57289bbeb4be5ca8daf64e874921c92ba5c08d613c30973`) |
| **DIFFERENTIAL** — band vs control objective at a shared τ = 0.15, all components | **BIT-IDENTICAL** (`loss_total` 1.0765775442123413 both, 16 real pairs) |
| — and the objective is not tau-blind | band at its own τ: **0.6409480571746826** |
| — the fixed-τ ruler reads the same under both schedules | 0.0029314844869077206 in **both** arms |
| first-update displacement `‖θ₁−θ₀‖₂`, control | 0.05588674077623233 (ng2 measured 0.05588674077623233; ng1 0.055886740188786026) |
| first-update displacement, band arm | 0.05588611132474989 |
| wall / peak RSS | 61.1 s and 65.1 s; **41.48 GiB** |

**The second row is the strongest receipt in this memo and it is a measurement, not an argument.**
ng1 ran this exact one-update cold segment BEFORE ng2's telemetry row and before ng3's validator
work existed. The trained state after all of it is bit-identical. So every byte landed since ng1 —
ng2's area-cap code (inert here), ng2's telemetry row, MAIN's re-pin, and this arm's two validators
— is score-neutral on the training path. **That is what lets a band cell built on a moved pin be
compared against a control that ran under the older trainer.**

The smoke ran **twice** at the final seal and reproduced every hash, every displacement and every
objective value to the last digit — a free determinism receipt on top of the mechanism one.

**A hazard for whoever re-runs this:** the probe's high-water mark is **41.48 GiB**, because two
real B=16 updates and the differential share one address space. It completed with ~59 GiB free
while the Metal chain ran, but budget it.

### A prediction turned into a measurement, on a different instrument ($0)

gm1 DERIVED that the band cuts sd1's τ-schedule reporting artefact 4.8×. I re-measured the same
quantity through the **trainer's own** `expected_flip_margin_loss` on the payload the smoke already
retained — a different instrument and a different sample from gm1's n32 milestone read:

| band | surrogate at τ_start → τ_end | schedule leg | gm1 predicted |
|---|---|---:|---:|
| legacy 0.15 → 0.05 | 0.005018208 → 0.002931596 | **−41.58%** | −41.30% |
| ng3 2δ_R → 1δ_R | 0.002840060 → 0.002591794 | **−8.74%** | −8.57% |
| | artefact reduction | **4.757×** | 4.819× |

Agreement within 0.3 pp on both legs. **Scope: this is the SCHEDULE leg on one frozen 16-pair
chunk.** It confirms the mechanism; it says nothing about where step 5,000 lands.

## Pre-registered falsifiers (fixed before the burn)

1. **PRIMARY — the band must act on the excursion.**
   `S_hat(5,000) < 0.42514878445269977` **AND** `S_hat(2,000) < 0.48567677825279465` (the cold
   control's endpoint and its over-paint peak).
   *If it fails:* gm1's static gradient-mass read does not imply a trajectory effect, and the τ row
   is refuted at FORMULATION scope for the born object — not the family.
2. **The fixed-τ telemetry must be faithful in-loop.** `seg_expected_flip_realized_tau_ref`
   (τ_ref = 0.05) **and** the annealed `seg_expected_flip_realized` at the band's own τ must BOTH
   peak at the same milestone as `d_seg_hat`. Read from `<run>/history.jsonl` under `objective`,
   **not** from `MILESTONE.json`, which does not carry it.
   *If it fails:* sd1's fixed-τ faithfulness does not survive inside the loop; the telemetry, not
   the lever, is wrong.
3. **Lane's share must fall as gm1 measured — AT THIS BAND'S OWN TEMPERATURES.** Lane's share of
   the seg gradient at step 0 falls **1.281×** at τ_start and **1.496×** at τ_end relative to
   τ = 0.15, and stays inside **1.281–1.592× / 1.496–1.903×** across the three milestones gm1
   measured. **This supersedes the charter's 1.6–2.1×**, which is gm1's 0.5·δ_R column and lies
   outside this band's range at every step.
   *If it fails:* gm1's static read did not transfer to the live loss. *Caveat recorded before the
   burn:* gm1 measured this by re-weighting a FROZEN field; the burn measures it on a field the
   band itself has moved, so a miss here is weaker evidence than a miss on falsifier 1.

**Read the DECOMPOSITION at every milestone, never the composite.** The control's damage is 91.20%
d_seg; a band that "fixed" S_hat by moving bytes or pose would be a different finding.

## What the band costs, stated plainly

Lowering τ de-prioritizes **Lane** — 0.59% of area, 33.56% of the model bits, ~90.1% of the rate
demand. gm1 MEASURED the cost at this band's own endpoints (§ falsifier 3), and MEASURED that a
GLOBAL `m_safe` cap's Lane over-push rises from 2.76% at τ = 0.15 to 11.23% at 2δ_R and 17.45% at
1δ_R. **The band CREATES the per-class-cap race it would otherwise have made unnecessary**
(`[[m148]]`: it changes the object the cap acts on). gm1 folded the per-class cap INTO this race.
**This cell deliberately does not carry it** — the charter specifies one lever and a pair of
`{band, measured cold control}`, and composing two unmeasured legs is the union-≠-sum trap. The
per-class cap is the first conditional follow-on below, with its trigger.

## MAIN fire command

Preconditions MAIN owns: the QBR1 chain has released the Metal, ng1's and ng2's cells have been
adjudicated (three separate levers must not compose before each is read), a live scorer claim and a
live Metal claim exist, and the sealed tree is unchanged. MAIN copies the sealed config to
`authorized_configs/`, sets `launch_authorized: true`, and binds both claim IDs — all three fields
are in `config_identity`'s ignored set, so binding them does not disturb the sealed identity.

**Verify the config first by hashing the FILE, never by recompiling** (a recompile legitimately
moves `ema.lawref.resolved_at`):

```bash
shasum -a 256 /Volumes/APDataStore/pact/ddm_ng3_tau_band/sealed_configs/seed_20260902_tau_band_control_native100.json
# expect 62739a88748e378c8632dcde4845a2c1eb427476189497339246c5f96fd459ef  (12,222 B)
```

```bash
SRC=/Volumes/VertigoDataTier/pact/ddm_ng3_tau_band/sealed_source_eed6f963c4
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng3_tau_band/launch/seed_20260902_tau_band_control_native100 \
  --cwd $SRC \
  --purpose "NG3 tau-band cell seed_20260902_tau_band_control_native100" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 2.3959503173828125 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt NG3_CELL_DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config /Volumes/APDataStore/pact/ddm_ng3_tau_band/authorized_configs/seed_20260902_tau_band_control_native100.json
```

**Fire from the SEALED tree.** Its `verify_inputs()` was run inside it as part of the snapshot and
returned PASS on all pinned inputs; `upstream/` and `experiments/results/mlx_fleet_gt_cache` are
symlinks to the repo (the arrangement `sealed_source_106d0dd0_v2` uses), because `git archive`
carries only tracked files and those two are the large sha-pinned inputs. **Use a done-receipt name
no other launch has reserved** — the arm's first smoke attempt was refused for exactly that.

Cost: one cell, **~2.95 h** measured on the identical object, **~1.375 GB** retained. No control
re-burn — the control is the already-measured seed-20260902 row.

**A scheduling fact MAIN should have:** at seal time the chain still held the Metal, and ng1's and
ng2's cells are ahead of this one in the queue. APDataStore holds **22 GiB** free against an 8 GiB
reserve; the remaining chain cells plus ng1 + ng2 + ng3 project ≈ 9.6 GB, which clears it but not
by much.

## Custody (ALWAYS KEEP THE PAYLOAD)

| artifact | path |
|---|---|
| law resolution receipt (band, provenance, FULL LawRef manifest incl. the volatile field) | `/Volumes/APDataStore/pact/ddm_ng3_tau_band/RESOLUTION.json` |
| seal receipt (single-lever diff, pin delta, recompile-determinism report, falsifiers) | `…/ddm_ng3_tau_band/SEAL_RECEIPT.json` |
| **sealed band-cell config** — sha256 `62739a88748e378c8632dcde4845a2c1eb427476189497339246c5f96fd459ef`, 12,222 B | `…/ddm_ng3_tau_band/sealed_configs/seed_20260902_tau_band_control_native100.json` |
| matched control (reference recompile, marked do-not-fire) | `…/sealed_configs/matched_control_of_record_seed_20260902_control_native100.reference.json` |
| schedule-leg receipt (the cross-instrument confirmation) | `…/ddm_ng3_tau_band/SCHEDULE_LEG_RECEIPT.json` |
| sealed source manifest | `…/ddm_ng3_tau_band/SEALED_SOURCE_MANIFEST.json` |
| **sealed source tree** (rev `eed6f963c4…`, pins verify inside) | `/Volumes/VertigoDataTier/pact/ddm_ng3_tau_band/sealed_source_eed6f963c4/` |
| bounded smoke result + both arms' retained payloads (139 MB) | `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng3_tau_band/bounded_smoke/` |
| governed launch manifests + run logs | `…/ng3_tau_band/smoke_launch/`, `…/smoke_launch2/` |
| run output root (empty; MAIN's cell writes here) | `…/ng3_tau_band/runs/seed_20260902_tau_band_control_native100` |
| code + 55 tests | `experiments/ddm_ng3_tau_band_cell.py`, `src/tac/tests/test_ddm_ng3_tau_band_cell.py` |

`authorized_configs/` is **not** written by this arm, and a test asserts the arm's source contains
no code path that could write it. Nothing was written under the live chain's `runs/`,
`authorized_configs/` or `CHAIN_LEDGER.jsonl`; the claims ledger was not touched; the
seed-20260902 control run was opened read-only.

**Two superseded sealed trees were removed** (`sealed_source_28bbb25fb4`, `sealed_source_c386702d46`,
2.0 GB each). Both were sealed at earlier commits during this arm's own fix rounds, neither was ever
fired, and each is exactly reproducible by `git archive <revision>` — the revision IS the
certificate, which is what makes the delete lossless rather than a discard.

## Equations leg (`tac.canonical_equations`)

**`margin_band_satisficing_threshold_v1` — CONSUMED, IN-DOMAIN, and consumed the RIGHT WAY.** Its
`domain_of_validity.included` names "SegNet signed-margin units measured by the delta_R artifact",
which is the unit the surrogate's temperature lives in. The band is the law's two outputs and
nothing else: the DSL factory calls `resolve_margin_band_threshold()` at compile time, the QBR1
validator re-derives it at validate time, the trainer's admissible set resolves it live, and a test
fails if any of the four historical decimals appears in executable code. That is the `[[m107]]`
split-banks cure at the point of use, and it is why this arm automatically reads dr1's n600 repoint
(`fallback_used False`, `n_frames 600`) rather than the retired n96 value.

**No anchor appended.** The law's `empirical_output` is δ_R / `m_safe`; this arm measures neither.
It uses `m_safe` as a TEMPERATURE, which is a new consumer of the same quantity, not a new
measurement of it. An anchor whose output is a schedule endpoint would teach the posterior a
quantity the law does not predict. Whether the band LOWERS d_seg is unmeasured until the cell burns;
anchoring the law on a design is the thing this campaign keeps extincting.

**`scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — CONSUMED as the premise.** It is why
`margin` in `sigmoid(-margin/tau)` is the exact signed distance to a flip, and therefore why
comparing τ to δ_R is a comparison of two quantities in the SAME units rather than an analogy. Both
gm1's split and this band's derivation rest on it. **No anchor appended** — the law's
`domain_of_validity.vehicle` is `softmax_of_sdf_levelset_witness` + `frozen_contest_segnet`, and
QBF1-born is a different vehicle sharing only the frozen scorer (gm1 and sd1 declined for the same
reason; `[[m21]]`, `[[m143]]`, `[[L18]]`). gm1's MEASURED refinement travels with it: "GT is the
runner-up on 98.018% of flips" is **90.138%** on this vehicle.

**`ema_decay_run_geometry_v1`** is consumed unchanged and IN-DOMAIN: the cell inherits the control's
sealed decay 0.9990793899844618 and the strict `check_ema_executable_law_matches_sealed_law` gate
sees no change. Its `lawref.resolved_at` is the volatile field discussed above.

**FORMALIZATION_PENDING** — the law this band's headline would need does not exist:

> *For a temperature-annealed sigmoid surrogate over an argmax field passed through a lossy
> round trip R, the surrogate's temperature and the round trip's noise floor are measured in the
> same units, and gradient mass placed at margins above the satisficing threshold `m_safe` is
> unrecoverable by the score. The schedule is therefore not a free pair of constants: its scale is
> determined by `m_safe`, and its span controls a reporting artefact of size
> `(L(τ_end) − L(τ_start))/L(τ_start)` on a frozen field.*

MEASURED on this arm's own field: that artefact is **−41.58%** on the shipped band and **−8.74%** on
the derived one. It should be registered once a band cell has burned, so it anchors on a measurement
of the cure rather than on this design.

## Scope and limits (these travel with the numbers)

* **Axis.** Every `S_hat` quoted is `[macOS-MPS n32 stratified advisory]` (ng1's read of the control).
  The derivation, smoke and schedule-leg are `[macOS-CPU advisory]`. **No score claim, nothing
  promotable, the pointer is untouched.**
* **GT lineage.** The vehicle pins the **PyAV** `gt_n600.npz`
  (`[[gt_n600_npz_is_pyav_lineage_train_on_dali_20260903]]`). Both arms sit on the identical
  lineage, so the comparison is internally valid; the **absolute** d_seg values are not
  DALI-authority numbers. Changing the GT would have been a second lever and was not done.
* **δ_R is an n600 constant applied to an n32 field.** dr1 measured it over 600 PyAV frames; this
  cell trains on 32 sealed pairs. The n32 selection's own δ_R is not measured, and `[[m88]]` cuts
  both ways. gm1 carried the same exposure and named it; it is inherited here unchanged.
* **The smoke is a MECHANISM check, not a verdict.** One update is not d_seg. It says the seed is
  consumed, the bytes change, the lever is exactly τ, and how the loss reads — nothing about where
  step 5,000 lands.
* **The schedule-leg receipt is the SCHEDULE leg only**, on one frozen 16-pair chunk. It confirms
  sd1's and gm1's mechanism on a second instrument; it is not evidence about the trajectory.
* **n = 1 seed, one cell.** A single-lever race on seed 20260902. It can move the design; it cannot
  close the family. Seeds 20260903 / 20260904 are the sign-repeat.
* **`EXPECTED_FLIP_TAU_REFERENCE` stays at 0.05 and now sits ABOVE the band's whole range.** That is
  deliberate and tested: ng2's row exists to compare cells across DIFFERENT schedules, so moving the
  ruler with the band would destroy the comparability it was added for. The smoke MEASURED both arms
  reading it identically (0.0029314844869077206).

## NEXT_IF_RESUMED — every row carries a disposition, an owner and a fire condition

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **`SEALED-AWAITING-MAIN-METAL-CLAIM`** — copy sealed → authorized, bind claims, fire the command above | **SEALED, ready** | MAIN | the chain has released the Metal AND ng1 + ng2 are adjudicated (three levers, read separately, `[[m164]]`) |
| 2 | **`CONDITIONAL-PER-CLASS-MSAFE-CAP`** — gm1 MEASURED that a GLOBAL cap over-pushes 11.23% of Lane's gradient at 2δ_R and 17.45% at 1δ_R; dr1's `m_safe_Lane = 0.025712` is already derived | **QUEUED-WITH-FIRE-ORDER** | MAIN to assign | fires if falsifier 1 PASSES and falsifier 3's Lane cost lands at or above its predicted range — the band is then working and Lane is the thing to protect |
| 3 | **`CONDITIONAL-DEEPER-BAND`** — gm1 MEASURED 96.8% waste removal at 0.5δ_R vs 77.7% at 1δ_R | **QUEUED, no fire order** | unowned; MAIN to assign or close | fires ONLY if falsifier 1 passes and #2 has landed; below ~1.5δ_R the Lane over-push is the binding cost and a deeper band without the per-class cap is choosing to waste it |
| 4 | **`TAU-INVARIANT-REPORTING-CELL` (sd1's)** — still worth firing even though this band cuts the schedule leg 4.76×; it is ~1 sigmoid per update and makes falsifier 2 readable live | **QUEUED-WITH-FIRE-ORDER** | MAIN | the next QBR1-lineage burn; note the live chain's remaining cells run from a tree that predates ng2's row, so read their histories with sd1's decomposition, never at face value |
| 5 | **`REGISTER-THE-TEMPERATURE-SCALE-LAW`** — the FORMALIZATION_PENDING statement above | **QUEUED** | whoever harvests the band cell | fires when the band cell returns, so it anchors on the cure |
| 6 | **`TAU-GATE-RETROFIT-AUDIT`** — this arm found a QBR1-level parameter with NO validator. `qbr1.validate_config` checks arm/pairs/schedule/EMA/initial-state/pins and two lever blocks; every OTHER config key it does not read is the same shape of hole | **QUEUED, needs a census first** | unowned — naming it without owning it would be the deferral scatter this repo extincted (`[[m36]]`) | fires if a second QBR1 cell proposes moving a config key no validator reads |

## DEAD-ENDS

* **"The trainer's τ-geometry check guards the QBR1 cell" is CLOSED as a premise** — MEASURED, it is
  never called on this path. ng1's memo line and the charter that inherited it are corrected here.
* **"The band cell is a same-pins twin of its control" is CLOSED as a framing** — the trainer pin had
  already moved at ng2, before this arm existed. The reproducible statement is same-START /
  same-schedule / same-EMA / same-selection, plus the MEASURED proof that everything landed since
  ng1 leaves the trained bytes identical.
* **"Lane loses 1.6–2.1× under the band" is CLOSED as a transfer** — that is gm1's 0.5·δ_R column,
  outside this band's range. The in-band numbers are 1.281–1.592× / 1.496–1.903×.
* **"A sealed QBR1 config's sha is reproducible by recompiling" is CLOSED** — `ema.lawref.resolved_at`
  is a dated observation the lineage keeps inside the config. The sha is a FILE property. This also
  bounds what ng2's quoted sha means, and MAIN's `shasum` check is unaffected.
* **Composing the area cap with the band in this cell is CLOSED** — one lever, and the seal refuses
  a cell that carries both. It is conditional row #2, after both have been read alone.
* **Re-measuring the gradient split to design this band is CLOSED as unnecessary** — gm1 had already
  measured it at $0 from sd1's retained logits, and this arm's only new measurement (the schedule
  leg) also came from an already-retained payload.

---

**Own-vehicle frontier: NOT MOVED** — this arm designs, gates and seals; it trained nothing,
byte-closed nothing, and could not move the pointer.
`afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED`.
