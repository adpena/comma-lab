# ddm_oa2 — does the frontier trainer's objective point where its score does?

**Arm:** `ddm_oa2_frontier_objective_alignment` · 2026-08-17 · axis `[macOS-CPU advisory]` ·
`score_claim: false` · `promotable: false`
**Receipt:** `/Volumes/APDataStore/pact/ddm_oa2/oa2_selector_seg_proxy_cost.json`
(14,358 B, sha256 `89287d6811516521188301b5db7e7fe93c5d1bc6785459513fd5776f57c38b07`)

---

## ANSWER

**Yes. The frontier trainer's objective is aligned with its score — exactly, and analytically.**
Not "well aligned". Aligned by construction, because **rate is the only axis its parameters can
move**, and cross-entropy plus weight-bits *is* the rate.

The alignment question as posed has **no measurable answer on this vehicle**, and that is the
finding. `∂d_seg/∂θ ≡ 0` — identically the zero vector, not a small one. So
`cos(g_CE, g_d_seg)` is `0/0`: **undefined, not low.** There is no alignment object to measure and
no noise floor to quote, because there is no estimator. Manufacturing one would be theatre.

**cw1's finding #1 is CONFIRMED as a code fact and REFUTED as an inference.** The code reads
exactly as cw1 quoted it. The conclusion drawn from it — that the frontier's seg debt is a
by-product of a misaimed objective, and that aiming `rg1b`'s probe there would say so — does not
hold. Absence of a seg term here is **correct**, not a defect.

**On the cheap lever: the charter's premise is backwards, and my own first measurement was wrong.**
My charter said the selector is `argmin(joint_bytes)` and asked me to price making it seg-aware.
The selector is **already seg-aware**, and that term has zero score effect while carrying **2.18×
the weight of the term that is the score**. But on the **frontier's actual selection it cost
nothing**: the authoritative e960 receipt (81 candidates) picked epoch 634, which is the argmin on
*both* criteria. **Measured seg-proxy cost on the frontier: 0 B.** The term is a live hazard that
did not fire here. See §4 — including the correction to my own first pass.

---

## 1. What I verified at source

| cw1 claim | verdict | evidence |
|---|---|---|
| `:1234` is `F.cross_entropy(logits, target)` + rate term | **CONFIRMED** | read at source |
| `:1265` selects `best` on `estimated_joint_bytes` | **CONFIRMED** | read at source |
| "SegNet never enters the loop" | **CONFIRMED, literally** | no SegNet forward anywhere |
| "The vehicle carrying 0.029611 S of seg debt optimises token density" | **REFUTED as inference** | see §2 |
| "aim rg1b's `cos(sign g)` at the frontier trainer" = the cheap probe | **REFUTED** | undefined, §3 |

cw1 scoped itself honestly — it wrote *"The consequence for the frontier's d_seg is INFERRED and
un-measured."* That caveat is what the correction lands on. The code facts are exactly right.

**Two things cw1 did not check, both load-bearing:**

1. **The training target *is* the SegNet argmax field.** `:962` `raw_tokens = cache_payload["seg"]`,
   verified at `:866–884` as CPU uint8 `(600,384,512)` with values in `[0,4]`, pinned by
   `EXPECTED_RX2_SPATIAL_TOKEN_SHA256`. Seg is not "absent from the loop" in the sense of being
   unrepresented — it is the *label set*. What is absent is a *scorer forward*, and §2 explains why
   that absence is correct.
2. **`tools/select_hpac_checkpoint.py` is not `argmin(bytes)`.** It minimises
   `(25/37545489)*estimated_joint_bytes + 100*top1_error`. cw1 quoted only the trainer's in-loop
   `best`; the tool that actually picks the shipped checkpoint has a seg proxy in it already.
   *(This also corrects my own charter, which asserted the selector was argmin-bytes.)*

---

## 2. Why `∂d_seg/∂θ` is exactly zero — MEASURED, by four independent arms

The frontier is `lane_ddm_hv1_ep0634`, S `0.15959729295498598` @ 182,759 B `[contest-CUDA T4]`.
It is a **coder recode**, not a retrain. Its own fire record
(`ddm_hv1_ep0634_t4_fire_execution_20260815.md:57`) states components as
*"seg 0.029611 (**identical decode**) · pose 0.0082946 (**identical decode**)"* — only rate moved,
by −743 B.

Three convergent facts settle it:

- **The label field is frozen.** `ddm_sp2` probe metadata, `settled_constraints`:
  *"fixed MC36 token labels make RX2 **rate-only**"*.
- **The codec is lossless.** Same probe: *"native RC64 decoded spatial-token SHA-256 is the
  **losslessness authority**"*. The decoder reproduces the exact token field; `top1_error` is the
  model's mode-accuracy diagnostic and never touches the reconstruction.
- **Measured across the lineage.** `ddm_fb1:58` — *"`seg + pose` is decode-identical across the
  entire `cp135 → MC36 → e480b v2 → hv1` lineage — I measured this to 1e-15 — so only rate moves"*.
  Independently reached by `hv2`, and restated at `ddm_rx1:150` and `ddm_rx2:246`.

The mechanism is plain once stated. The model emits a *probability* over a fixed field; CE is that
field's arithmetic-coding length. Rate `= CE`. Distortion is **not a function of θ at all** — every
checkpoint decodes the same bytes to the same field, so SegNet reads the same input and returns the
same `d_seg`. The trainer cannot move seg because *nothing in this vehicle can*.

**Consequence for the goal.** The gap to sub-0.15 is not a seg problem *on this vehicle*. It is an
absolute byte invariant: **archive ≤ 168,345.6 B**, a cut of **14,413.4 B** from 182,759 B
(`ddm_fb1`, re-derived off four bases). Seg 0.029611 is 3.09× the remaining gap in *magnitude*, but
its **derivative on this vehicle is zero**, so it is not addressable here at any price. Magnitude
ranks axes; derivatives decide where to spend. Ranking by magnitude alone is what pointed cw1's
rung 3 at a vehicle that cannot move.

---

## 3. Why the probe cannot be aimed here

`experiments/ddm_rg1b_weight_space_gradient_cosine.py` binds to `tac.pr130_lift`
(`band_objective`, `editability_levers`) — the **semantic renderer**: 228,958 params, FiLM,
frozen SegNet differentiably in-loop. It measures `cos` between *two candidate objectives* on a
vehicle that renders RGB.

The HPAC vehicle emits **token logits over a fixed label field**. It never materialises an image, so
there is no differentiable path to SegNet and no second gradient to take a cosine against. This is
not a porting cost. `g_d_seg` is the zero vector; the cosine's denominator is zero.

`tools/local_endpoint_close.py` confirms the design intent independently: it *"never invokes a
scorer, provider, identity race, archive compiler, or auth evaluation."* Scorer-free is the
architecture, not an oversight.

---

## 4. The selector's seg term — priced on the right receipt, after I priced the wrong one

### 4.0 Correction to my own first pass (same turn)

My first measurement used
`/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/checkpoint_selection_e960_endpoint.json` and I
called it "the real e960 receipt". **It is not.** Despite the filename, its `checkpoint_dir` points
at `…/training/checkpoints/mc36_hpac_best_ema.checkpoints/periodic` and it holds **31** candidates
over epochs 1–60 — an earlier, different run. The e960 endpoint has **81** candidates over epochs
482–642. The filename is a live trap for anyone reading by name; nothing downstream appears to
consume it, but it should be renamed.

I am correcting the headline rather than footnoting it, because a stale headline surviving a
corrected body is a failure genus this repo has hit repeatedly — and here I was the one committing
it.

### 4.1 The authoritative receipt (the one that produced the frontier)

`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/endpoint_closure/checkpoint_selection.json`
— 81 candidates, bound by sha to the run log, and cited by both `PREFLIGHT.json` and the governed
early-stop receipt.

| | epoch | est. joint bytes | top1_error |
|---|---:|---:|---:|
| **selector picked** | 634 | 130,393 | 0.00189454 |
| **rate optimum** | **634** | **130,393** | — |
| **cost of the seg proxy** | | **0 B** | **0.0 S** |

**The criteria agreed.** Epoch 634 is the argmin on both axes, so the spurious term changed nothing
on the shipped selection. That is the honest answer for the frontier, and it is a *fortunate
coincidence*, not a property of the criterion.

### 4.2 The hazard is real and it does fire

Same tool, the earlier 31-candidate receipt — a genuine `hpac_checkpoint_joint_proxy_selection.v1`
run:

| | epoch | est. joint bytes | top1_error |
|---|---:|---:|---:|
| selector picked | 46 | 134,539 | 0.00192072 |
| rate optimum | 54 | **134,453** | 0.00192657 |
| **cost** | | **+86 B** | **+5.73e-5 S** |

So the term **can and does** override the score axis; it simply did not on the frontier's pool.

### 4.3 Why it is a hazard at all — the structural defect, exact on both receipts

**The spurious term outweighs the real one by 2.18×** on the authoritative receipt
(`seg proxy = 100 × 0.00189454 = 0.18945` versus `rate = 6.659e-7 × 130,393 = 0.08683`), and 2.15×
on the other. A quantity with **zero score effect** carries ~69% of the criterion's mass. Whether it
flips the pick is then left to luck about how the candidates happen to line up.

**The modelling error, named.** The coefficient `100.0` is the S formula's seg coefficient — the
tool treats `top1_error` *as* `d_seg`. It is not. True `d_seg` at the frontier is
`0.029611/100 = 2.9611e-4`; `top1_error` at ep634 reads `1.89454e-3`, **6.40× larger**, and it
*moves with the checkpoint while true d_seg does not move at all*. Two different quantities in two
different charts, joined by a shared coefficient.

**Scope of the numbers.** Both are on the **estimated** byte axis
(`byte_authority: ADVISORY_ESTIMATE_NOT_SERIALIZED`) — deterministic given weights, so no
statistical noise floor applies, but an estimate can order candidates differently than serialized
bytes would. The **term-dominance ratio is exact**; the 86 B magnitude is advisory. Candidate byte
spread across the e960 pool is worth checking before any future selection, since the criterion
governs it.

---

## 5. What I landed, and what I deliberately did not

**Landed (score-neutral observability, default-on):** `tools/select_hpac_checkpoint.py` now emits
`rate_only_optimum` in every receipt — the rate-axis argmin, whether it differs from the pick, and
`seg_proxy_cost_bytes` / `seg_proxy_cost_score`. **The selection is unchanged.** Two tests added
(7 pass, ruff F clean).

**Deliberately NOT done:** changing the selection criterion. That is a **design decision** under
CLAUDE.md, it sits on a live fire path MAIN consumes, and the existing test
`test_selects_joint_proxy_argmin_not_latest_or_byte_minimum` encodes the current behaviour as
*intended* — it asserts the selector does **not** pick the byte minimum. Overriding a documented
design intent unilaterally is not mine to do. The receipt now carries the evidence so it can be
adjudicated rather than argued.

**Recommended (cheapest honest intervention, for MAIN/council):** set `SEG_PROXY_COEFFICIENT = 0.0`
— one constant, $0, no retraining — and update that test. **Expected gain on the frontier's own
selection: 0 B.** This is *insurance*, not a win: the term is worth 2.18× the score term and fired
on one of the two real receipts I checked (86 B advisory). Buy it because the next selection may not
be as lucky, not because it recovers anything already lost. Pre-condition if a magnitude is ever
claimed: one serialization pass confirming estimated-byte ordering matches real-byte ordering.

**Also owed (housekeeping, not mine to land mid-arm):** rename
`ddm_hv1_harvest_compose/checkpoint_selection_e960_endpoint.json` — it is not the e960 endpoint and
it cost me a wrong headline. Related: `local_endpoint_close.py` writes
`checkpoint_joint_proxy_selection.json`, but the landed authoritative file is
`checkpoint_selection.json`; no `*joint_proxy*` file exists in either store, so that receipt came
from a differently-named invocation than the current code path.

---

## 6. Where the seg axis actually lives

Seg 0.029611 is real and it is 3.09× the remaining gap. It is created **upstream of this vehicle**,
in the frozen MC36 label field and the receiver's realization — the known "realization in the IMAGE
chart" crux. It is addressable on the **semantic renderer**, which is exactly where `rg1b`'s probe
already runs and where the "92.7% CONFIGURATION" result was measured. Per
`ddm_drain_vehicle_split_and_lever2_payoff_20260817.md` those are different architectures
(37 tensors / 39,375 params, no FiLM vs 38 / 66,339 with 8 FiLM tensors), so no number crosses. The
**question** transfers; the numbers do not.

---

## 7. Labels

- **MEASURED:** seg-proxy cost **0 B on the frontier's authoritative 81-candidate receipt**; 86 B /
  5.726e-5 S on the earlier 31-candidate receipt; 2.18× / 2.15× term dominance; 6.40×
  top1-vs-d_seg ratio at ep634. All recomputed by me from the two named receipts.
- **CORRECTED IN-TURN:** my first pass priced the wrong receipt (misnamed file) and reported 86 B as
  the frontier's cost. The frontier's cost is 0 B. §4.0.
- **MEASURED (by other arms, re-verified at their sources):** decode-identity of seg+pose to 1e-15
  across the lineage; losslessness via decoded spatial-token SHA-256; fixed MC36 labels.
- **VERIFIED_VIA_SOURCE_INSPECTION:** every line-number claim about the trainer and the selector.
- **DERIVED:** `∂d_seg/∂θ ≡ 0` follows from lossless coding of a frozen label field. Deductive, not
  measured — its premises are measured.
- **ASSUMED:** none load-bearing. I did not run the RC64 decoder myself; I relied on the recorded
  losslessness authority plus four convergent arms. A direct two-checkpoint decode-and-compare would
  close that last inch and costs one local run.

**verdict_scope:** `FORMULATION` for "no seg lever exists here" — it binds the HPAC/RX2 fixed-label
lossless-coder formulation, not the paradigm. Any vehicle that renders RGB, or any change making the
label field trainable, reopens it immediately. **`INSTANCE`** for each byte figure: 0 B binds the
e960 receipt, 86 B binds the 31-candidate receipt. Neither generalises; the 2.18× term dominance is
the part that does, because it is a property of the formula rather than of a candidate pool.

---

## 8. Provenance note on the vehicle

"hv1" is a **harvest/composition arm**, not a training run. The checkpoint `epoch_0634.pt` came from
`rx2_wc2_full_mps_e960`, launched via `tools/train_ddm_cl1_hpac_capacity_mps.py`, which imports
`tools/train_ddm_cl1_hpac_capacity.py` by pinned content hash (`run_identity` embedded in the
checkpoint records both, `port_mode: full-mps-e960`). So cw1's file attribution is right, reached
through the MPS port. hv1 then byte-closes an archive from it — `measured_delta_distortion: 0.0`,
`decoded_token_identity: true`, raw decode byte-identical to the MC36 CPU decode — which is the
composition-side confirmation of §2's invariance, independent of the lineage memos.
