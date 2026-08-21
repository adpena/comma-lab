# ddm_ec2p — the ec1 conditioning price, re-pinned on rc2, and the realized measurement that already exists

**arm** `ddm_ec2p` (ec1 prep) · **date** 2026-08-21 · **git** `16b8706bc3f865c521d163e2d07159f98077b853`
**This arm fired no scorer, no trainer, no paid GPU job.** It read one Modal volume, re-encoded bytes
through rc2's own coder, and built candidate ZIPs locally. `score_claim=false`,
`promotion_eligible=false`. **Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B
`[contest-CUDA T4, n600]`, archive `df7fd266…` — UNMOVED.**

STORES CONSULTED: `.omx/research/ddm_eq1_equations_lineages_vs_rc2_20260821.md` ·
`ddm_ec1_implicit_edge_conditioning_20260814.md` · `ddm_ec2_oriented_adapter_trainer_20260814.md` ·
`.omx/state/canonical_frontier_pointer.json` · `.omx/state/canonical_task_status.jsonl` ·
the ec1 retained store on VertigoDataTier · the Modal volume `comma-ddm-js1b-argmax-retained` ·
rc2's shipped runtime tree at `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/`.

---

## ANSWER FIRST

**Re-pinned price: 1,176 B, not 1,707 B.** MEASURED as an actual archive delta on rc2's body
(180,456 → 181,632 B), using rc2's own coder — brotli q11 lgwin24, which I verified reproduces rc2's
shipped semantic stream byte-for-byte. That is **531 B cheaper, a 31.1% cut**, and it drops the
break-even from 5.644% of rc2's flips to **3.888% — 923.7 flips**.

**And the premise I was given is wrong. The realized Δd_seg HAS been measured.** My charter, the eq1
audit, and the ec1 memo all say "NEVER MEASURED." The oriented adapter was trained on T4 on 2026-08-14
and its full-n600 endpoint returned. I recovered the run and verified its artifacts byte-for-byte
against their receipt SHAs. It did not miss the bar. **It went 2.17× the wrong way:**

| MEASURED, `[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY`, CP135 body | value |
|---|---:|
| base instrument flips | 34,970 |
| candidate flips | 75,749 |
| base errors it FIXED | **12,075 = 34.53% of the base** |
| errors it INTRODUCED | **52,854 = 4.377× its fixes** |
| net flip reduction | **−40,779** |
| break-even it needed | +1,155 |

**Read the decomposition, not the headline.** The targeting worked. A 34.53% gross recovery clears the
CP135 break-even (3.30%) by **10.5×** — the 808× selectivity did convert into real fixes through the
real receiver. **The mechanism failed on collateral alone.** At the re-pinned rc2 price, and holding
that gross rate, collateral must fall **4.93×** for the family to clear break-even. That is a specific,
attackable number. It is not a family verdict.

**Status of the realization measurement: BLOCKED, and not on compute.** The sealed trainer is built,
tested, and on disk. Two of its three pinned inputs are CP135-bodied and do not exist for rc2.

---

## 1. The orphan, and how it happened

The 08-14 dispatch fired on the fourth attempt (`fc-01M0073TSNJEKW2BA4XTGF950X`, git
`bcaec56e6f12`). It completed: `returncode 0`, `training_complete: true`, all 1,800 optimizer steps,
EMA updates 1,800, 543.6 s elapsed. The result landed at
`…/ddm_ec1_20260814/main_cuda/poller_run/EC2_T4_REMOTE_RESULT.json`.

Then three things failed at once, and none of them was the science:

1. The local poller crashed on a dependency install —
   `pip install 'pydantic>=2.0,<3'` returned non-zero (`poller.failed`).
2. Every dispatch claim stayed ACTIVE. `run.log` carries four repeated
   `⛔ BLOCKER … claims file still ACTIVE for terminal call_id` lines. The terminal rows were never
   appended.
3. **No arm ever wrote the result into a memo.** `ddm_ec2_oriented_adapter_trainer_20260814.md` is the
   pre-fire seal document and still reads `QUEUED-WITH-A-FIRE-ORDER` / `modal_dispatched=false`.

So the corpus kept saying "never measured" while a −40,779 flip endpoint sat on an SSD. Today's eq1
audit ranked ec1 the largest un-cashed claim on exactly that premise. This is the harvest-orphan class
in its purest form: the payload was retained, the conclusion was lost.

**I verified the recovery rather than trusting the JSON.** I downloaded the selected module and its
archive from the Modal volume and hashed them:

| artifact | bytes | sha256 (first 16) | matches receipt |
|---|---:|---|---|
| `ec1_latent.int8.br` (stage-30 EMA, selected) | 1,369 | `9559c2ab5128f193` | yes |
| `archive.zip` (CP135 + ec1) | 187,723 | `3fcef97c9857123f` | yes |

Its member list is `['p', 'ec1_latent.br']` at 186,152 + 1,369 B, so the CP135 delta of **+1,471 B** is
real and reproducible.

---

## 2. Deliverable 1 — the re-pinned price, MEASURED

### 2.1 First: as-designed packaging is structurally REFUSED on rc2

rc2 ships **one** stored ZIP member, `p`. Two independent guards in its own runtime refuse anything
else:

- `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/inflate.py:36` —
  `if archive.namelist() != ["p"]: raise`
- `…/runtime/residual_archive.py:459` — the same check inside `read_residual_archive`

ec1's design ships a second member. **On rc2 that is not a +102 B choice; it does not decode.** Any ec1
landing must fold into `p`. This was not a known constraint before this arm — ec1 was designed against
CP135, which has the same guard but was never asked the question.

### 2.2 The coder, identified and verified

rc2's `p` is an RX1 container: a 14 B header plus three brotli streams plus the token tail.

| section | in-archive bytes | decoded body |
|---|---:|---:|
| RX1 header | 14 | — |
| hpac stream | 13,515 | — |
| **semantic stream** | **30,856** | 36,130 |
| carrier stream | 22,028 | — |
| residual + RC64 tokens | 113,943 | — |
| **sum** | **180,356** = member `p` | |

I recompressed the decoded semantic body and searched the parameter grid.
**`brotli(body, quality=11, lgwin=24)` reproduces the shipped 30,856 B stream byte-for-byte.** That is
the real coder, not an assumed one. Every price below is measured through it.

### 2.3 The two payload forms

The ec1 module raw is 2,173 B: 9 B magic, a **645 B JSON header**, and 1,528 B of int8 weights and
fp16 biases. The header carries schema strings, tensor names, shapes, dtypes and byte counts — all
**generic receiver code** under rule 118. Only three fp32 scales are video-derived. The minimal form
keeps 12 B of scales plus the 1,528 B of tensors = **1,540 B**, and moves 633 B of header into
inflate.py where it is free.

### 2.4 The measured table

Built as real ZIPs and measured with `os.path.getsize`. **Writer control: rebuilding rc2's own `p`
gives exactly 180,456 B, delta 0**, so the writer adds no size of its own. (The control's SHA differs
from `df7fd266…` because my writer normalizes timestamps; it is size-neutral, not byte-identical, and I
label it that way.)

| placement | archive | **delta** | rate ΔS | legal on rc2? |
|---|---:|---:|---:|---|
| A — second ZIP member, as-designed | 181,927 | **+1,471** | +0.000979479 | **no** — namelist guard |
| B — folded into semantic stream, as-designed | 181,865 | +1,409 | +0.000938195 | yes |
| **B — folded, minimal payload** | **181,632** | **+1,176** | **+0.000783050** | **yes** |

**Instrument validation, twice.** Placement A on the *seeded design* module reproduces the ec1 memo's
1,707 B exactly. Placement A on the *trained* module reproduces the T4 receipt's 1,471 B exactly. The
re-pin agrees with both prior measurements where it should, then departs where the body actually
differs.

### 2.5 The price is a tight band, not a point

I re-pinned all three stage-boundary EMA modules:

| training state | coded B | rc2 fold-minimal delta | break-even flips | % of 23,757 |
|---|---:|---:|---:|---:|
| stage 10 `target_birth` | 1,380 | 1,174 | 922.2 | 3.882% |
| stage 20 `balanced_descent` | 1,360 | 1,191 | 935.5 | 3.938% |
| stage 30 `collateral_finish` (selected) | 1,369 | **1,176** | **923.7** | **3.888%** |

**Band 1,174–1,191 B, spread 17 B = 1.4%.** The price barely moves with the weights, so MAIN can budget
1,176 B and treat ±17 B as noise. Use `ceil(0.7855 × actual_delta_bytes)` at package time regardless.

---

## 3. Deliverable 2 — ceiling and break-even, re-derived

rc2 recomputed from components: `100·0.00020139 + √(10·6.37e-6) + 25·180,456/37,545,489`
= `0.020139000 + 0.007981228 + 0.120158243` = **0.14827847122030852**, matching the pointer to 17 dp.
Exchange rates: **S/flip = 8.477105e-07 (1.2731 B-eq)**, **S/B = 6.658590e-07**, **flip pool = 23,757**.

At the re-pinned 1,176 B:

| quantity | value | vs charter (1,707 B) |
|---|---:|---:|
| rate price | +0.000783050 S | +0.001136621 |
| **ceiling** (d_seg → 0, net of price) | **−0.019355950 S** | −0.019002379 |
| **break-even** | **923.7 flips = 3.888%** = Δd_seg **7.8305e-06** | 1,340.8 = 5.644% |
| clears the −3.5e-6 admission bar | **927.9 flips = 3.906%** | 1,344.9 = 5.661% |
| clears 10× that bar (−3.5e-5) | **965.0 flips = 4.062%** | 1,382.1 = 5.818% |

**The required recovery share, stated plainly: 3.906% of rc2's flips to be admitted at all, 4.062% to
clear the bar by 10×.** The re-pin cut both by about 1.75 percentage points.

**The ceiling still exceeds the gap.** −0.019356 against a 0.018278 gap to 0.13 is 105.9%. ec1 remains
the only single mechanism in the inventory that can close that gap alone — and it remains a ceiling, not
a projection. A reader who treats −0.019356 as an expected value has misread this memo.

### The number that actually matters now

The 08-14 endpoint fixed 34.53% of base errors. Carry that gross rate to rc2's 23,757-flip pool:

- gross fixes ≈ **8,203**
- net required ≥ **924**
- therefore introduced must be ≤ **7,279 = 0.887× fixes**
- measured collateral was **4.377× fixes**

**Collateral must fall 4.93× at constant gross recovery.** That, not the AUROC and not the ceiling, is
the quantity the next run has to move.

---

## 4. Deliverable 3 — the realization plan

### 4.1 What already exists

`experiments/ddm_ec2_oriented_adapter_trainer_worker.py` (45,440 B) and
`experiments/ddm_ec2_modal_oriented_adapter_trainer.py` (39,342 B) are in-tree, tested (12 passing),
and sealed. The injection site survives on rc2 unchanged: `…/candidate_runtime_composed/cpr1/inflate.py`
lines 92–130 hold the same `SemanticTokenRenderer(96)` with `token_embed → coord_mix →` four
`TokenBlock`s at dilations (1,1,2,4). **ec1 injects between `coord_mix` and the blocks, and that seam
exists on rc2's body.** No architectural port is needed.

### 4.2 The blocking prep — two pinned fields are CP135-bodied

The trainer pins three inputs. Two are wrong for rc2:

| input | state | action |
|---|---|---|
| GT argmax field `91d3ff11…` | **reusable** — GT does not depend on our body | none |
| base error field (CP135, 34,970 flips) | **wrong body** — rc2 has 23,757 | regenerate: one T4 n600 field pass on rc2 |
| `decoded_tokens_n600.npy` (CP135) | **wrong body** — rc2 ships a different token stream and HPAC | regenerate from rc2's archive |

I checked `/Volumes/APDataStore/pact/ddm_rc2/`: the rc2 T4 row is **scalars only**
(`MODAL_REMOTE_RESULT.json`, d_seg 0.00020139). No per-pixel argmax field was retained. This is the same
retention gap that produced the orphan above, one layer down.

### 4.3 The instrument defect that made the 08-14 verdict unattributable

I pulled all three `STAGE_PACKAGE.json` files. They carry byte accounting and checkpoints — and **no
flip count**. The n600 field pass ran **once**, at the end. So the run cannot distinguish "the mechanism
is harmful" from "training was stopped at the wrong place." The collateral trajectory was never
observable.

**Fix it in the config, not in the reading.** Run the n600 field pass at every stage boundary. The
measured endpoint cost is small — the whole 08-14 run was 543.6 s against a 10,800 s cap — so three
extra passes are affordable inside the same cap. Without this, the next run buys the same
un-attributable scalar.

### 4.4 The config, at optimal form

Changes from the 08-14 recipe, each tied to a measured cause:

1. **Re-pin all three fields to rc2** (§4.2). Refuse to start on a CP135 field.
2. **Measure flips at every stage boundary** (§4.3).
3. **Put the collateral term in the objective, not in a schedule.** 08-14 used a fixed weight ramp:
   stage 10 ran error-weight 3,373.31 against correct-weight 0.25 — a **13,493:1 per-pixel ratio**. It
   bought 34.53% gross recovery and 4.377× collateral. The target is a ratio ≤ 0.887×, so the run needs
   a constraint on introduced errors, not a hand-set ramp — a Lagrangian on introduced-count with the
   multiplier driven by the measured ratio.
4. **Add a PoseNet leg.** 08-14 was `COMPONENT-ONLY`: SegNet only, no pose at all. ec1 changes rendered
   RGB, and PoseNet reads rendered frames, so d_pose *will* move. It is currently **entirely
   unmeasured** for this mechanism, and §5 shows the gate is brutal.
5. **Keep** what worked: full-population 20-stratum permutation (no prefix), serialized parse-back
   endpoint, per-stage live+EMA checkpoints, full payload retention.
6. **Package via placement B** (fold into the semantic stream, minimal payload) and re-pin rc2's
   runtime `ARCHIVE_SHA256` / `ARCHIVE_BYTES` constants. Placement A does not decode (§2.1).

### 4.5 Preflight

**Memory — MEASURED, not projected.** The 08-14 run is the runnability anchor: T4 16 GB, batch 16, full
n600 endpoint, `returncode 0`, 543.6 s. rc2 changes the token source, not the tensor shapes, so peak is
unchanged at batch 16. Its own storage preflight recorded 40 GiB expected retention against 409.6 GB
free on the Modal volume; that budget stands, plus roughly 3 field-pass fields.

**Local storage — a real constraint.** `/Volumes/APDataStore` has **35 GiB free (99% full)** and
`/Volumes/VertigoDataTier` has 69 GiB. This arm retained 3.4 MB. **Do not harvest the 40 GiB of Modal
training payload to APDataStore.** Route bulk to VertigoDataTier or leave it on the volume with a
manifest.

**Wall clock and cost.** 08-14 measured 543.6 s inside a 10,800 s cap. Adding three field passes and a
pose leg stays well inside it. At $0.60/T4-hour the 08-14 run cost about $0.09 against its $1.80
projection — the 3× autograd allowance was conservative by roughly 18×.

### 4.6 Status: **BLOCKED — READY_TO_FIRE not asserted**

I am not emitting a sealed launch ticket, and I will say why rather than ship one that lies. The sealed
08-14 fire order points at CP135 fields. Firing it re-runs the same CP135 experiment. The two field
regenerations in §4.2 and the objective change in §4.4 must land first. Those are MAIN's calls, and the
second one is a design decision, not a parameter.

**Fire order for MAIN, in dependency order:**

1. Close the four orphaned `ddm_ec2_oriented_adapter_trainer` dispatch claims (P0, one command, and it
   is already four blockers overdue).
2. Regenerate rc2's base argmax field + `decoded_tokens_n600.npy` and retain both.
3. Land the collateral-constrained objective and stage-boundary flip measurement in the worker.
4. Re-seal against rc2 fields, fire, then package via placement B and run the RE1T/JS1B component chain
   plus `upstream/evaluate.py` for the exact row.

---

## 5. Deliverable 4 — the falsifier, PRE-REGISTERED

Registered **before** any measurement. Thresholds derive from §3 at the re-pinned 1,176 B. Instrument:
rc2's own n600 T4 field, base **23,757 flips**, all 117,964,800 scorer pixels, no prefix. `net` = base
flips − candidate flips.

| band | condition | verdict |
|---|---|---|
| **LIVE** | net ≥ **965** (4.062%) | clears the admission bar by 10×. Proceed to pose + exact row. |
| **MARGINAL** | **924** ≤ net < 965 | pays its rate but sits inside the bar. Not admissible alone; hold for composition. |
| **CLOSED-neutral** | **0** ≤ net < 924 | mechanism works, does not pay its bytes. INSTANCE-closed at this price. |
| **CLOSED-harmful** | net < 0 | INSTANCE-closed. 08-14 landed here at −40,779. |

**Pose gate (refuses admission in every band).** Δd_pose must not cost more than 2.25e-7 S. At rc2,
`dS/d(d_pose) = 626.47`, so the gate is **d_pose ≤ 6.370359e-06, an allowed rise of 3.59e-10 —
0.00564% relative.** That is effectively "pose must not move." I flag it as the harshest gate in the
plan: ec1 perturbs rendered RGB and PoseNet reads rendered RGB, so this is a live risk, not a formality,
and it has never been measured for this mechanism.

**Secondary diagnostic, recorded whatever the band.** Report `fixed`, `introduced`, and
`introduced/fixed` at every stage boundary. If gross recovery holds near 34.53% and the ratio falls
below 0.887×, the family is alive even when a given run lands CLOSED.

**Verdict scope.** Any outcome is **INSTANCE-level** — this objective, this schedule, this body. Per
ec1's own pre-registration and the paradigm-vs-implementation rule, routing Seg to #978 requires the
optimal-form capacity/family ladder to fail the same test, not one run.

---

## 6. My own round-1 adversarial review

1. **Is the re-pin honest about what it measured?** The 1,176 B is an archive delta built with a
   size-neutral writer control. But the placement-B archive I built **does not parse** —
   `read_residual_archive` length-checks the semantic body. I measured the coding cost of ec1's bytes in
   rc2's real coder, not a decodable candidate. The bytes are right; the runtime work to make it decode
   is unbuilt, and I have not priced that work.
2. **Am I transferring the CP135 flip result to rc2?** The 34.53% gross rate is CP135's. rc2 has a 1.47×
   smaller flip pool on a different body. I use it once, explicitly labelled, to derive the 4.93×
   collateral target — which is a *design target*, not a prediction. It could be wrong in either
   direction and the run must re-measure it.
3. **Did I over-claim the minimal payload?** The 633 B header saving assumes shapes, names and dtypes
   move into inflate.py as generic code. That is legal under rule 118 and consistent with how the rest
   of this receiver is built, but it is unimplemented. If MAIN keeps the JSON header, the price is
   1,409 B and break-even rises to 4.659%.
4. **Is "the realized measurement exists" itself over-claimed?** I verified two artifact SHAs against
   the receipt and read the three stage packages. I did **not** re-run the endpoint or download the
   118 MB argmax field, so I am trusting the run's own flip arithmetic. It is internally consistent
   (75,749 − 34,970 = 40,779; 12,075 fixed and 52,854 introduced reconcile) and its price fields
   reproduce byte-exactly under my independent re-encode, which is decent corroboration but not a
   re-measurement.
5. **Negative-existence claims.** "No arm wrote up the ec2 result" is scoped to `rg` over
   `.omx/research/` and `.omx/state/` for `ddm_ec2`, `net_flip_reduction`, and `40779` — the ec2 memo
   exists but is the *pre-fire* seal. "No rc2 argmax field is retained" is scoped to
   `/Volumes/APDataStore/pact/ddm_rc2/` and a maxdepth-3 sweep of both SSD roots. Both are scoped
   searches, not universal claims.
6. **What would change my mind fastest?** A stage-boundary flip trajectory. If collateral was still
   falling monotonically at step 1,800, the 08-14 verdict is a stopping-rule artifact and the cure is
   cheap. If it had flattened, the 4.93× target needs a mechanism change. **We cannot tell, and that is
   the single most valuable thing the next run buys.**

---

## 7. Retention

`/Volumes/APDataStore/pact/ddm_ec2p/retained/` — 17 files, 3.4 MB, every one with bytes + sha256 in
`RETENTION_MANIFEST.json`.

| artifact | bytes | sha256 (16) |
|---|---:|---|
| `ec1_latent.int8.trained.br` (selected, recovered from Modal) | 1,369 | `9559c2ab5128f193` |
| `ec1_latent.ema.10_target_birth.br` | 1,380 | `eccb85b91cae60e0` |
| `ec1_latent.ema.20_balanced_descent.br` | 1,360 | `252db8994ac40b1f` |
| `ec1_latent.trained.raw` / `.minimal.bin` | 2,173 / 1,540 | `2e5b077f72856001` / `4f0c57d890bc80c7` |
| `ec2_trained_cp135.archive.zip` | 187,723 | `3fcef97c9857123f` |
| `candidate_B_fold_minimal_trained.zip` (**the re-pin**) | 181,632 | `057675ae6853e8b7` |
| `candidate_B_fold_raw_trained.zip` | 181,865 | `16574d1ab8af1167` |
| `candidate_A_member_trained.zip` | 181,927 | `6b733ceec3227c43` |
| `control_rc2_rebuild.zip` (size-neutral writer control) | 180,456 | `90de6b60315c3929` |
| `EC2_BATCH_RECEIPTS.jsonl` + 3 `EC2_STAGE_*.json` | 67,055 + ~23.5 K | see manifest |

---

## 8. Owed, with owners

1. **Close the four ACTIVE `ddm_ec2_oriented_adapter_trainer` dispatch claims.** P0, overdue since
   2026-08-14. **MAIN.**
2. **Correct the corpus.** `ddm_eq1_…_20260821.md` §3 and `ddm_ec1_…_20260814.md` both say ec1's
   realized Δd_seg was never measured. It was, on 2026-08-14, at −40,779 flips. Append-only supersession
   per Catalog #110/#113 — do not mutate the originals. **MAIN.**
3. **Regenerate rc2's base argmax field and decoded token field, and retain them.** The blocking prep.
   **MAIN.**
4. **Decide the collateral-constrained objective** (§4.4 item 3). A design decision, not a parameter.
   **MAIN / council.**
5. **Retain the per-pixel argmax field on every future T4 candidate row.** The rc2 row kept scalars
   only, which is why §4.2 exists. Sister of the always-keep-the-payload rule at the field level.

`verdict_scope`: the re-pin is **MEASURED-EXACT** on rc2's real coder and real ZIP arithmetic. The
economics are **DERIVED-EXACT** from it. The 08-14 flip result is **MEASURED**
`[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY` on the **CP135 body**, INSTANCE-scoped. No
mechanism is closed at the family level. Every ΔS is priced against
**baseline rc2 = 0.14827847122030852**.
