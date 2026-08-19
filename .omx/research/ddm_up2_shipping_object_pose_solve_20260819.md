# ddm_up2 — the pose carrier is not rate-limited and not lattice-limited. It is basis-limited.

- **arm** `ddm_up2` (up1 fire-order 1–4 / the first pose solve aimed at the object that ships)
- **date** 2026-08-19
- **axis** `[macOS-CPU advisory, frozen CPU-torch PoseNet]` · `score_claim=false` ·
  `promotable=false`. **Pointer UNMOVED** at contest-CUDA `0.15659459685822907`.
  This arm fired no Modal job. MAIN owns the T4 slot.
- **cost** $0.
- **code** `experiments/ddm_up2_shipping_pose_solve.py` (+57 tests) · commits `72a829bafe`,
  `39a04c3214`
- **store** `/Volumes/APDataStore/pact/ddm_up2/`

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` · `ddm_up1_uncapped_pose_solve_20260819.md`
(§1/§3/§6/§7, the arm I inherit) · `ddm_pi2_pose_axis_attribution_20260816.md` (its unbuilt
fire-order 2 is built here) · `ddm_pv1_pose_floor_and_admission_bar_20260816.md` ·
`ddm_ps1u_uncapped_pose_solve_20260816.md` · `ddm_t1h_pose_coeff_resolve_headroom_20260817.md`
(+ its byte pricer, reused) · `ddm_qs2_compensation_overlay_runtime.py` ·
`/Volumes/APDataStore/pact/ddm_to1/{t4_row_r1/MODAL_REMOTE_RESULT.json,advisory/attempt_0002/**,
generations/to1_tail_override_r1/**}` · `upstream/{evaluate.py,frame_utils.py,modules.py}` (read at
source, not quoted from memory).

---

## ANSWER FIRST

1. **The instrument premise SURVIVES, and now on structure rather than on a coincidence.**
   `upstream/evaluate.py:31-42` binds the GT dataset class to the device: `cuda ->
   DaliVideoDataset`, else `AVVideoDataset`, consumed at `:58`. The two classes assert
   their device at `frame_utils.py:113` and `:188`, so the binding is bijective and
   cannot silently fall back. The to1 T4 receipt records `--device cuda`,
   `provenance_device=cuda`, `gpu_t4_match=true`. **The shipping row is scored against
   DALI GT.** up1's 0.9999x agreement was real, and it is now backed by the code path.
2. **NEW, and it is a bigger deal than the instrument: `[contest-CUDA]` and
   `[contest-CPU]` are scored against DIFFERENT GROUND TRUTH.** Not a hardware drift —
   a different decoder, by construction, in the same file. This is the standing open
   question CLAUDE.md flags on the PR102 CUDA−CPU gap ("mechanism attribution remains
   open: DALI/NVDEC-vs-PyAV ground-truth decode..."). up1's 23.74x same-frames
   measurement plus this code path answers it: **the dominant term is the GT decoder.**
   Consequence nobody has priced: a pose gain earned on DALI GT is not owed on the
   public CPU leaderboard, and may invert there.
3. **The solve ran, uncapped, to a convergence PROOF on all 600 pairs — and the carrier
   was already within 1.5% of its own optimum.** d_pose **7.769484e-06 → 7.649247e-06**
   (ratio **0.98452**), 429 pairs improved, **0 worsened**, at **0 archive bytes** and
   **0 d_seg**. Net **ΔS = −6.846805e-05**: past the −3.5e-06 bar by 19.6x and past the
   summed 8dp report bound by 12.0x, so it is a real and resolvable row. It also closes
   only **1.04%** of the gap to sub-0.15. Every prior claim of pose headroom on this
   vehicle was measured against the wrong GT; this is the first one that was not, and its
   answer is **the pose carrier is not where sub-0.15 comes from**.
4. **The wall is NOT rate.** Coefficient bytes are nearly free: perturbing **all 7,200**
   coefficients by ±4 costs **+5 bytes**; 1,000 coefficients at ±1 costs **+6 bits (0
   bytes)**. Measured with `ddm_t1h`'s pricer, whose re-encode reproduces the shipped
   78,065-bit Rice payload exactly, and which is proven sensitive by a monotone
   perturbation ladder. Any future claim that the pose carrier is rate-blocked is false.
5. **The wall is NOT the int12 lattice either.** Per-coordinate scale-shrink headroom is
   **1.000–1.020x** — the int12 range is already fully used (max |code| 2044–2048 on all
   12 coordinates). There is no finer lattice to buy.
6. **The wall IS the 12-dimensional basis.** The pose residual sits almost entirely in
   the carrier Jacobian's *smallest* singular direction (σ_min ≈ 0.011–0.029 against
   σ_max ≈ 10–14.5; residual share on the last direction 0.94–0.995). To null the
   residual the rendered frame must move a median **4.1 LSB RMS** through the shipped
   basis, but only **0.87 LSB RMS** if the basis is relaxed to full 24×32 freedom at the
   same spatial band-limit — a **6.4x median (up to 18.3x) penalty attributable to the
   basis**, not to resolution, not to rate, not to quantization. §5.
7. **CORRECTION to up1 — its two-line decode mechanism is half right.** Rendering the
   shipped body on ONE device at both batch settings: `semantic_batch` (`cpr1/inflate.py:312`)
   **is** byte-changing — 1,326 pixels flip by ±1 from batch shape alone. `pose_batch`
   (`:335`) is **byte-NEUTRAL** — 0 of 1.83e9 pixels change. up1 proposed this as its
   cheapest experiment; run, it falsifies half its own claim. §6.

---

## 1. Structural confirmation — which GT does the shipping row score against?

Not inferred from the 0.9999x agreement. Read from source:

```
upstream/evaluate.py:31    if device.type == "cuda":
upstream/evaluate.py:39        DefaultDatasetClass = DaliVideoDataset
upstream/evaluate.py:40    else:
upstream/evaluate.py:42        DefaultDatasetClass = AVVideoDataset
upstream/evaluate.py:58    ds_gt = DefaultDatasetClass(test_video_names, data_dir=args.uncompressed_dir, ...)
upstream/frame_utils.py:113    assert self.device.type == 'cuda'   # DaliVideoDataset
upstream/frame_utils.py:188    assert self.device.type != 'cuda'   # AVVideoDataset
```

The compressed side is `TensorVideoDataset` on BOTH axes (`evaluate.py:67`) — it reads
our inflated `.raw`. So **only the GT side changes with the device.** That is exactly the
variable up1 isolated, and it is why the same frames scored 23.74x apart.

The to1 T4 row's own receipt: `canonical_path = archive.zip -> inflate.sh ->
upstream/evaluate.py --device cuda`, `scorer_device=cuda`, `provenance_device=cuda`,
`gpu_model=Tesla T4`, `gpu_t4_match=true`, `score_axis=contest_cuda`. **CUDA branch taken
⇒ DALI GT.** Premise confirmed; the arm proceeds.

**The consequence up1 did not draw.** `AXIS_GT_LINEAGE` is now a table in code:
`contest_cuda -> dali`, `contest_cpu -> av_pyav`. These are two different objectives.
d_pose is not one quantity. Optimising one is not neutral on the other, and nothing in
the repo has ever priced that. Named as an owed measurement in §8.

## 2. The vehicle, and why the seg-hold is free

`cpr1/inflate.py:312-328` renders frame `2p+1` from the semantic tokens; `:335-352`
renders frame `2p` from 12 coefficients against a shared 12×3×24×32 basis. And
`upstream/modules.py:108` is `x = x[:, -1, ...]  # Use only last frame`.

**SegNet never sees frame 0.** The seg-hold is not a measurement I have to defend — it is
a slice. The carrier has 12 free coefficients per pair, 6 pose equations per pair, all 600
pairs independent, and zero d_seg obligation.

**Forward-model control (the thing that makes every later number mean something).**
Re-rendering frame 0 from the coefficients and diffing against the shipped decode:
**byte-exact, max |delta| = 0 over 33,572,088 pixels**, across 10 pairs including all 5
selector-special pairs (ROLL, CHANNEL, TILE). Receipt:
`retained/forward_model_control.json`. The float selector is separately proven
bit-identical to the receiver's integer `apply_pixel_mode` across all 8 modes including
saturating inputs (tests).

## 3. What the solve is, and why it uses no surrogate

Uncapped greedy descent on the **realized** objective: every candidate is rendered
through the exact receiver path (real `round`, real selector), scored by the frozen CPU
PoseNet against the DALI targets, and accepted only if d_pose actually falls. Termination
is a convergence *proof* — a full sweep of the lattice neighbourhood finding no improving
move — not an iteration cap.

**Execution note (method, not result).** The single-process driver measured at ~245% of
an 1800% machine and was solving ~2 pairs/min with 15 cores idle. Pairs are independent
(`cpr1/inflate.py:338` is per-pair), so the tail of the field was sharded across four
detached workers at 4 threads each, ~7 pairs/min per shard. Identical per-pair math,
identical GT, identical lineage gate — only which pairs a process walks changes. Rows are
keyed by pair and the solve is deterministic, so the merge REFUSES any pair whose two
sources disagree rather than taking last-wins.

That refusal is a *mechanism*, and I nearly reported it as a *result*: I first wrote that
the main run and the first shard overlap, giving a free cross-process determinism check.
They do not — the main run was stopped at pair 199 and the shards begin at 200, so the
merge had nothing to compare and the check never fired. A deliberate overlap was therefore
re-run as an explicit control: **10 seeded-random already-solved pairs re-solved in a
fresh process, spanning 4 different origin processes — `final_d_pose` bit-identical,
solved codes identical, and even the pass count identical, on all 10.** Receipt
`retained/determinism_control.json`. That is the deterministic-reproducibility leg
actually exercised, not merely designed.

A gradient solve was built first and **abandoned on measurement**, which is itself the
finding. With STE through the receiver's two `round`s the Jacobian is correct (it agrees
with a finite-difference Jacobian at cos 0.96–0.99 for h=32 code units), but every
Levenberg-Marquardt step made the realized d_pose **worse**, at every damping and every
step scale tried, down to sub-lattice moves:

| damping | α | predicted MSE | realized MSE | vs base | max Δcode |
|---|---|---:|---:|---:|---:|
| 1e-1 | 0.03 | 2.838e-06 | 2.883e-06 | **1.015** | 0.02 |
| 1e-1 | 1.0 | 2.808e-06 | 3.763e-06 | 1.325 | 0.62 |
| 1e-2 | 1.0 | 2.738e-06 | 1.203e-05 | 4.236 | 5.65 |
| 1e-3 | 1.0 | 2.457e-06 | 6.268e-05 | **22.074** | 35.88 |

base 2.839e-06 (n=8). The linear model never predicts more than a ~7% gain and the
realized value never improves. Reading this correctly is §5.

## 4. The n600 result

**The instrument reproduces the T4 row at full field, independently.** My own n600
measurement of the SHIPPED codes against DALI GT is **7.76948388629175e-06** against the
T4 row's **7.77e-06** — **0.99993x**. up1 established this; this is a second, independent
n600 confirmation on the merged field.

**The solve, run uncapped to a convergence PROOF on every one of 600 pairs:**

| | value |
|---|---:|
| d_pose before (DALI, n600) | 7.769484e-06 |
| d_pose after (DALI, n600) | **7.649247e-06** |
| ratio | **0.98452** |
| pairs improved | **429 / 600** |
| pairs worsened | **0** |
| coefficients changed | 970 / 7,200 |
| **archive bytes** | **0** (Rice 78,065 → 78,072 bits; payload 9,759 B unchanged) |

**Priced against the live pointer** (`S = 0.15659459685822907`, d_pose 7.77e-06, 176,420 B):

| term | ΔS |
|---|---:|
| pose | **−6.846805e-05** |
| rate | 0.0 |
| seg | 0.0 |
| **net** | **−6.846805e-05** |

* vs the **−3.5e-06 admission bar**: clears it by **19.6x**.
* vs the **summed two-row 8dp report bound** (5.6945e-06, bounds ADD and the pose bound
  GROWS as d_pose falls): **12.0x the bound**, so the T4 receipt can resolve it.
* d_pose stays at 7.65e-06, far above the **5e-09** floor below which the 8dp report
  prints `0.00000000` and cannot resolve an improvement at all. No caveat needed here.
* Projected pointer: **0.15652612881042932**.

**Read this honestly.** It is a real, byte-free, seg-neutral, resolvable improvement — and
it closes **1.04%** of the 0.0065946 gap to sub-0.15. The carrier is now within 1.5% of its
own optimum on the objective that ships. **The pose carrier is not where sub-0.15 comes
from, and this arm is the measurement that establishes that rather than assuming it.**

### 4b. Two shipping routes, both priced — and the sparse one loses

The receiver offers a sparse compensation section (`runtime/compensation_overlay.py`)
that looks purpose-built for exactly this: apply small code deltas without touching the
entropy stream. It is capped by its own format at **15 pairs** (count packs into 4 header
bits, `:52`) with deltas in **[-3, 4]** (`:58`). So a 600-pair solve cannot ship that way
at all, and the best 15-pair subset is exactly priceable.

Both routes at **n600**:

| route | changed | bytes | ΔS pose | ΔS rate | **net ΔS** | verdict |
|---|---:|---:|---:|---:|---:|---|
| full re-encode | 970 coords | **0** | −6.847e-05 | 0.0 | **−6.847e-05** | **admissible, 19.6x the bar** |
| 15-pair overlay | 42 coords | +66 | −1.212e-05 | +4.395e-05 | **+3.183e-05** | **REFUSE** |

**The sparse overlay is strictly dominated by re-encoding everything.** That is
counterintuitive and it is the opposite of the usual rate instinct. The reason is §ANSWER-4:
the CAP1/AR1 Rice stream absorbs small coefficient perturbations almost for free — the
full re-encode here came in **one byte SMALLER** than shipped — while the overlay pays a
fixed 10 bits of pair index plus 12 bits of support mask per pair before it encodes a
single delta. Any future arm reaching for the compensation section to carry a broad,
small correction should read this row first.

My byte formula for the overlay is not asserted: it is tested against the receiver's own
`encode_compensation_overlay` at supports 1/2/7/15 and matches exactly.

### 4c. The CUDA gain costs the CPU axis — measured, and nobody had priced it

Same solved frames, same PoseNet, both GT caches, **n600**:

| objective | GT lineage | d_pose before | d_pose after | ratio |
|---|---|---:|---:|---:|
| `contest_cuda` (the pointer) | DALI | 7.7695e-06 | 7.6492e-06 | **0.98452** |
| `contest_cpu` (public board) | PyAV | 1.482928e-04 | 1.483534e-04 | **1.00041** |

**Improving the shipping axis by 1.55% makes the public-leaderboard axis 0.041% worse.**
The magnitudes are wildly asymmetric but the SIGN is what matters: the two objectives pull
in **opposite directions** under this actuator, so "improve d_pose" has been an ambiguous
instruction for the whole campaign. The measured `av / dali` ratio on the shipped codes is
**19.0866x** — which is the same 19.1x that separates this body's advisory `d_pose`
(0.00014829) from its T4 `d_pose` (7.77e-06). Two independent routes to the same constant;
the advisory/T4 pose gap **is** the GT-lineage gap, not hardware.

The cost here is small enough to accept (−6.85e-05 on CUDA for +6.1e-09 d_pose on CPU,
which is +2.5e-08 in CPU-axis score units), but it is a cost, it was never priced before,
and on a larger pose move it would not stay negligible.

### 4d. The local advisory eval is NOT a valid gate for a CUDA-axis pose candidate

My own charter set the promotion gate as "advisory `d_seg` identical AND advisory `d_pose`
IMPROVED vs 0.00014829". **That gate is unsound, and §4c is why.** The local advisory runs
`upstream/evaluate.py --device cpu`, which takes the `else` branch at `evaluate.py:40` and
scores against **PyAV** GT. So the advisory measures the *contest-CPU* objective, while the
candidate is solved for the *contest-CUDA* objective. The receipt shows exactly this: the
same body reads `d_pose = 0.00014829` on the advisory and `7.77e-06` on the T4 — a 19.1x
lineage gap, not noise.

A CUDA-axis pose candidate is therefore **expected to look slightly WORSE on the advisory**,
and gating on "advisory d_pose improved" would refuse a correct candidate. Every prior arm
that used the advisory as a pose gate was reading the wrong instrument.

**Replacement gate, pre-registered here rather than chosen after the fact:**
1. advisory `d_seg` **bit-identical** to 0.00043336 (the seg-hold, §7.6);
2. advisory `d_pose` lands within the band my CPU-axis cross-price predicts
   (0.00014829 x the measured n600 `av_ratio`), which is a *quantitative prediction* my
   model can fail;
3. archive bytes and parse-back as declared.

Leg 2 turns the advisory from a broken gate into a genuine falsifier of §4c.

## 5. Why the carrier cannot reach its own residual — the measured mechanism

Three candidate walls. Two are measured shut.

**Not the lattice.** Per-coordinate max |code| is 2044/2006/2017/2040/2048/2026/2048/2048/
2030/2039/2047/2048 against an int12 rail of 2048. Scale-shrink headroom
`2047/max|code|` = **1.000–1.020x**. The lattice is already as fine as this dynamic range
permits. Separately, a full single-coordinate scan (12 coordinates × {±1, ±2}) finds the
shipped codes at or beside a local minimum on 6 sampled pairs: 0–5 improving moves out of
48, best ratio 0.930.

**Not the rate.** §ANSWER-4. Nearly free.

**The basis.** Per-pair SVD of `d(pose)/d(coeff)`:

| pair | σ (6 values) | share of residual on the LAST direction |
|---|---|---:|
| 61 | 13.51 … 0.0112 | **0.995** |
| 82 | 14.51 … 0.0291 | **0.941** |
| 70 | 10.70 … 0.0115 | 0.610 |

The carrier moves pose *hugely* in one direction (σ≈10–14, almost certainly the global
photometric one) and *barely at all* in the direction the residual actually occupies. An
exact linear solve therefore demands **366–10,141 code units** — up to 5x the entire
representable int12 range — which is precisely why every LM step left the linear regime
and lost.

Relaxing only the basis, at the same 24×32 spatial band-limit, and asking for the
minimum-norm step that nulls the residual:

| pair | 12-dim basis (LSB rms) | code units | 24×32 relaxed (LSB rms) | basis penalty |
|---|---:|---:|---:|---:|
| 102 | 0.68 | 76 | 0.278 | 2.5x |
| 227 | 2.11 | 224 | 1.158 | 1.8x |
| 553 | 2.72 | 214 | 0.591 | 4.6x |
| 582 | 13.87 | 1,619 | 1.705 | 8.1x |
| 589 | 30.60 | 1,544 | 1.674 | **18.3x** |
| 595 | 5.51 | 536 | 0.548 | 10.1x |
| **median** | **4.11** | | **0.875** | **6.4x** |

**The 24×32 spatial resolution is not the wall — 0.87 LSB rms is a small, representable
image change. The 12-dim basis is.** It forces a 6.4x larger frame perturbation to buy the
same pose correction, and that oversized perturbation is what the rounding and the
nonlinearity then destroy.

Scope: linearized (Jacobian) analysis, n=6 seeded-random pairs, `verdict_scope:
formulation`. It bounds the *demanded step*, not the realized gain of a re-fit basis.

## 6. CORRECTION to ddm_up1 — the decode mechanism is half falsified

up1 localized the device-dependent decode to two lines and named the confirming
experiment as its fire-order 1. Run here on ONE device, both batch settings, whole field:

| line | setting | result |
|---|---|---|
| `cpr1/inflate.py:335` `pose_batch` | 1 vs 64 | **bit-identical**, 0 / 1,831,204,800 pixels changed |
| `cpr1/inflate.py:312` `semantic_batch` | 1 vs 8 | **differs**, 1,326 pixels at max \|Δ\|=1 |

Batch-1 reproduces the shipped CPU raw on both halves; batch-8 does not. So batch shape
alone *does* flip bytes, through the `clamp/round` at `:323-324` — but only on the
semantic half. **`pose_batch` is falsified as a divergence source.** This does not show
that `semantic_batch` explains the *whole* CPU-vs-CUDA gap; that still needs a CUDA-side
render to compare against raw sha `3c810cc4…`. Receipt:
`retained/decode_axis_batch_shape_probe.json`.

## 7. My own round-1 adversarial review

1. **Is the "converged" claim neighbourhood-limited?** Yes, and it is labelled so. The
   solve proves no improving *single-coordinate* ±1/±2 lattice move exists. A richer
   neighbourhood (coordinate pairs, larger offsets) is not excluded, and since bytes are
   nearly free there is no rate reason not to try it. Named in §8.
2. **Is the byte pricer inert?** Controlled against exactly that failure. It reproduces
   the shipped 78,065-bit payload, and a perturbation ladder shows monotone growth
   (+0 / +2 / +6 / +7 / +47 / +226 bits). A function that ignored its input would have
   returned 0 throughout; it does not.
3. **Is my STE gradient the reason the LM steps failed?** No — falsified by the
   finite-difference control (cos 0.96–0.99 at h=32 codes). The gradient is right; the
   realized landscape is what refuses.
4. **Am I quoting a prefix?** No. Every reported sub-n600 measurement uses a seeded
   RANDOM sample; `select_pairs` now refuses to return a prefix below n600, because
   `ddm_na2`/`ddm_bp2` measured pose prefixes 2.54–4.21x HARDER than the population.
   (This was a live bug in my own driver, found in review and fixed before the landing.)
   **The dry run independently re-measures that law:** the first 118 pairs read
   `d_pose = 1.744e-05` against the field's `7.77e-06` — **2.25x harder**, inside the
   published band, from data collected for a different purpose. The §4b/§4c prefix rows
   are therefore labelled non-quotable and are used only for the SIGN and the MECHANISM,
   never for a magnitude.
5. **Does the DALI GT cache equal what a T4 actually decodes?** The cache's lineage is
   *declared* and verified against the axis, and §1 proves which class the axis uses. It
   is not proof that this particular cached tensor was produced by that class on that
   hardware. Residual risk, stated; the 0.9999x n600 agreement with the T4 row is the
   evidence that it was.
6. **Is the seg-hold really free?** Structurally yes (`modules.py:108`), and now
   empirically too: on 8 pairs carrying real solved code changes, frame 0 moved (max
   |Δ|=2) while SegNet argmax stayed bit-identical over 1,572,864 pixels and
   `compute_distortion` returned exactly 0.0. Receipt `retained/seg_hold_control.json`.
7. **THE TRANSFER RISK, and it is the one I would attack first.** This solve optimises
   against the **CPU-decoded** body (`0.raw` sha `ccbfa332…`); the T4 scores the
   **CUDA-decoded** body (sha `3c810cc4…`). Both frames enter PoseNet. up1's n600
   agreement bounds the aggregate decode effect at roughly the report half-ULP (~5e-9 in
   d_pose), which is ~20x smaller than the gain in §4 — so the gain should survive in
   aggregate. **But the mechanism is uncomfortable:** the solve earns its gain by
   exploiting fine sub-LSB rounding structure, and fine rounding structure is exactly
   what differs between the two decodes (§6 measures 1,326 such pixels from batch shape
   alone). Per-pair transfer is therefore NOT guaranteed even though the aggregate bound
   is small. Any T4 fire on this candidate must treat "gain does not transfer" as a live
   falsifier, not a formality. This is the cross-regime-constant-transfer genus, and I am
   flagging it against my own result.

## 8. Owed, with owners

1. **Price the CPU axis.** A DALI-GT pose gain has never been checked for what it does to
   the PyAV-GT (public leaderboard) objective. Both caches are on disk; this is a
   re-scoring, not a re-solve. **Nobody owns the two-objective question.** (§ANSWER-2.)
2. **Re-fit the 12-dim basis to the measured pose-residual subspace.** The single
   highest-value follow-on: 6.4x median demanded-step reduction, coefficients nearly
   free. Requires pricing the basis section (Huffman-coded 5-bit codes) which is NOT free.
3. **Richer solve neighbourhood** on top of the converged codes (resumable, additive).
4. **Empirical seg-hold control** on the solved codes (argmax identity), before any seal.
5. **GT-lineage gate → STRICT catalog wire-in.** Built here as a canonical helper
   (`verify_gt_lineage` / `required_lineage_for_axis` / `AXIS_GT_LINEAGE`, 8 tests, fails
   closed inside the solve driver). The numbered-catalog wire-in is **BLOCKED** by the
   Catalog #299 quota brake (next number 408 > 400, no operator waiver in CLAUDE.md's
   first 200 lines). That refusal is information: the brake exists to force a
   stop-and-consolidate pause, so it was not bypassed. Operator decision owed.
6. **Retain the 1,326 changed-pixel coordinates** from §6 (digested and discarded; re-derives
   deterministically from retained inputs).
7. **Byte-close the candidate. BLOCKED, diagnosed, minutes of work once cleared.**
   The byte ACCOUNTING is finished and validated — the candidate re-encodes to the
   IDENTICAL Rice parameters, the IDENTICAL bit count and the IDENTICAL payload length as
   shipped, so `delta_bytes = 0` and the shipped container carries it with no header or
   k-field change. Writing the `archive.zip` is blocked by two things, both recorded in
   `identity_control/BYTE_CLOSE_BLOCKER.json`:
   - `ddm_t1h_compose_pass1.py:108` reads `k_base = packed[139]`, a **body-specific
     hardcoded offset**. On the to1 body that byte is 177, so the tool refuses every
     candidate — **including the shipped codes themselves**. That failed identity control
     is the proof it is a tool bug and not a container limit.
   - The Rice payload is verbatim in the CAP1 blob (offset 12467, uniquely) but is **not
     byte-locatable in the packed carrier section** the archive actually stores, so a
     content-addressed same-length splice cannot find its target. Closing it needs the
     CAP1↔packed field mapping, which depends on `encode_cap1` **off-repo** on
     VertigoDataTier. I wrote a narrow splicer (`retained/byte_close_up2.py`); it refuses
     correctly at this point rather than mis-splicing.
   No re-solve and no new measurement is needed — only packaging.
8. **The advisory-as-pose-gate defect (§4d) is apparatus-wide**, not local to this arm.
   Any tool that gates a CUDA-axis pose claim on a macOS/CPU advisory `d_pose` is reading
   the other objective. The `verify_gt_lineage` helper landed here is the fix; wiring it
   into those tools is owed.

## 9. Retained payload

`/Volumes/APDataStore/pact/ddm_up2/` — `solve_n600_dali/` (`rows.jsonl` with per-pair
start/final d_pose, full descent history, converged flag, and the solved codes for every
pair; `SUMMARY.json`; launcher manifest + watcher logs), `retained/`
(`forward_model_control.json`, `decode_axis_batch_shape_probe.json`, `base_codes.npy`,
and the three probe scripts: basis conditioning, byte-pricer sensitivity, trust region).
