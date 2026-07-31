---
title: ua1 — read-only inventory of the frozen scorer weight FILES (segnet/posenet safetensors)
utc: 2026-07-31
lane_id: lane_ddm_ua1_scorer_weight_file_inventory_20260731
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
evidence_axis: "[macOS-CPU advisory]"
pointer_moved: false
pointer: 0.1910828242 [contest-CPU] UNMOVED
upstream_mutated: false
machine_readable: .omx/research/ddm_ua1_scorer_weight_file_inventory_20260731/tensor_inventory.jsonl
---

# ua1 — the frozen oracle, read as two files

`$0`. No training, no scorer slot, no dispatch. Every byte of `upstream/` read-only;
nothing inside `upstream/` was created, edited, moved, or deleted.

## 0. Denominator and custody

| | tensors inspected | params inspected | unreachable |
|---|---|---|---|
| `upstream/models/segnet.safetensors` | **562 / 562 (100%)** | 9,610,645 / 9,610,645 | 0 |
| `upstream/models/posenet.safetensors` | **510 / 510 (100%)** | 13,943,652 / 13,943,652 | 0 |

`sha256(segnet.safetensors) = 68956e328d4c5d87…` (matches the pin cited by #725/hb1)
`sha256(posenet.safetensors) = 0f3a0874c5c387f9…`

Machine-readable per-tensor inventory (name · shape · dtype · params · **byte offsets** ·
min/max/mean/std/zero-fraction/all-equal) is committed beside this memo as
`ddm_ua1_scorer_weight_file_inventory_20260731/tensor_inventory.jsonl` (1,072 rows) with
`file_summary.json`. Every number below is queryable from those two files.

MAIN's ground truth reproduced exactly (tensor counts, param counts, dtype split, header
lengths, metadata dicts, and the 5-class order). No disagreement found.

**Search scope for every "did not find" statement below**, stated once: filename scan of
`.omx/research/` (~all entries), full-text grep of
`.omx/research/ddm_hb1_hope_bn_capacity_findings_20260727.md`,
`.omx/research/codex_findings_ddm_pa1_posenet_amplitude_twin_20260723_codex.md`, and
`grep -rn` over `src/ tools/ experiments/*.py` for `num_batches_tracked` and
`final_layer.pose`. I did **not** grep the full `.omx/research/` corpus by content (one
attempt exceeded the tool budget). Statements are therefore "did not find in that scope",
never "does not exist".

---

## 1. SURFACE: *read-vs-present* — the 4-way partition of a weight file

**Coordinates.** Each file key lands in exactly one cell of
`{present in file} × {in model.state_dict()} × {read by the eval-mode forward}`.
MEASURED by instantiating the real `upstream/modules.py` `SegNet()`/`PoseNet()` and
diffing key sets, then `load_state_dict(sd)` at its default `strict=True`.

| cell | SegNet | PoseNet |
|---|---|---|
| present in file, **absent** from `state_dict()` | **0** | **0** |
| present in file, loaded, **read** in eval forward | 484 | 510 |
| present in file, loaded, **never read** in eval forward | **78** (`num_batches_tracked`) | 0 |
| **absent** from file, silently back-filled at load | 0 | **88** (`num_batches_tracked`) |
| file keys that are `nn.Parameter` / buffer | 328 / 234 | 332 / 178 |

**Where the language flips.** The charter asked for "PRESENT but NOT LOADED". On the
strict-load axis that set is **empty in both files** — `load_state_dict` returns
`<All keys matched successfully>` for both, and 100% of file bytes are consumed. The free
information is not on the *loaded* axis; it is one level in, on the *read* axis:

`torch/nn/modules/batchnorm.py::_BatchNorm.forward` reads `num_batches_tracked` only
inside `if self.training and self.track_running_stats:` (source-verified, quoted below).
The contest scorer runs `.eval()`. **MEASURED: SegNet's 78 `num_batches_tracked` are
loaded into memory and never read by any forward pass we run.** 78 params, 624 bytes of
pure provenance we have been carrying and not reading.

```
if self.training and self.track_running_stats:
    if self.num_batches_tracked is not None:
        self.num_batches_tracked.add_(1)
```

**What moves this level set.** Only two things: (a) running the scorer in `.train()` mode
(we never do — it would also mutate `running_mean`/`running_var`), or (b) a torch version
that reads the counter in eval. Neither is in our control path.

**Which way it falls.** Toward "free": the 78 counters cost us nothing to read and are
inert with respect to every score we compute.

**Reported, not acted on (upstream is immutable).** PoseNet's 88 `num_batches_tracked`
are **absent from the file** yet `strict=True` succeeds. The mechanism is
`_NormBase._load_from_state_dict`, which back-fills the counter with `0` when the
state-dict carries no `_metadata` version — and `safetensors.torch.load_file` returns a
bare `dict` with no `_metadata`. So `strict=True` is *silently non-strict* for this one
buffer class on this exact load path. This is a latent fragility in the pinned upstream
loader, not a defect we may touch. It also means: **PoseNet's file carries no training-
schedule provenance at all**, while SegNet's does.

---

## 2. SURFACE: *training-schedule provenance* — the value of the 78 counters

**Coordinates.** One integer per BN unit: the number of batches that unit's running
statistics were updated over, at the moment the checkpoint was written.

MEASURED, and the distribution is **perfectly bimodal with zero exceptions**:

| value | count | which units |
|---|---|---|
| **1,947,423** | **68** | every `encoder.model.*` BN (incl. the stem BN `encoder.model.bn1`) |
| **148,480** | **10** | every `decoder.blocks.{0..4}.conv{1,2}.1` BN |

DERIVED (exact integer identity, not a fit): `1,947,423 − 148,480 = 1,798,943`.
Ratio `1,947,423 / 148,480 = 13.116`.

**Where the language flips.** If the two values were *equal*, the honest reading would be
"trained end-to-end from scratch, single phase". They are not equal, and the encoder is
the *larger* one.

- A **frozen** encoder cannot be the explanation: a module in `.eval()` does not increment
  its counter, so a frozen encoder would read *lower* than the decoder. It reads higher.
- INFERRED (one assumption: during the joint phase all BN units see every batch — true for
  a U-Net with no frozen sub-tree): the encoder counters were **inherited at 1,798,943**
  from a prior checkpoint whose counters were never reset, and encoder+decoder were then
  trained together for **148,480** batches. The decoder was born at the start of that
  joint phase.
- This is the first hard evidence in our custody that the frozen SegNet is a **two-phase
  artifact**, not a from-scratch model — despite `SegNet.__init__` passing
  `encoder_weights=None` (which governs *this* constructor, not the checkpoint's history).

**What moves this level set.** Nothing we can do; the file is pinned. What *changes our
reading* is finding a public `tu-efficientnet_b2` checkpoint whose BN counter equals
1,798,943 — that would convert the inference into an identification.

**Which way it falls.** Toward "the encoder is a long-pretrained backbone, the decoder is
short-trained". CONJECTURE, labelled as such: the `eid` metadata
`ef2db93d-2a06-49fd-b119-993e52cb5a44/144` most plausibly encodes *(experiment UUID)/(run
or epoch index 144)*. It does **not** let us conclude the dataset, the batch size, the
epoch count, the optimizer, or whether 144 indexes epochs vs checkpoints — a UUID and an
integer are not a schedule. `148,480 = 1024 × 145` and `= 32 × 4,640` are both exact
factorisations; neither is evidence for a batch size, and I decline to pick one.
PoseNet has **`metadata: None`**, so none of this exists for the pose half.

**Prior-work position.** #725/hb1 inventoried exactly these **78 BN units** (68 encoder +
10 decoder) and used their `running_*` statistics for capacity. Its unit count and mine
agree exactly — that is the control, and hb1 is SOUND on this. hb1's memo contains **zero**
occurrences of `num_batches_tracked` (grep count = 0), and the repo-wide grep found the
symbol only inside vendored torch. So: the *unit set* was known; the *counter values* are
new in the scope stated in §0.

---

## 3. SURFACE: *BN dead capacity* — per-channel input-invariance

**Coordinates.** Per BN channel: `(γ, running_var)`, and the derived forward gain
`|γ| / sqrt(running_var + 1e-3)` (`BN_EPS = 0.001`, `upstream/modules.py:19`). A channel
with `γ → 0` emits a constant `β` regardless of input: no perturbation we make to the
frames can move it, and no gradient returns through it.

| | channels | `var < 1e-8` | `\|γ\| < 1e-3` | `\|gain\| < 1e-3` | min gain |
|---|---|---|---|---|---|
| SegNet (78 BN) | 33,368 | **0** | **0 (0.000%)** | **0** | 1.50e-03 |
| PoseNet (88 BN) | 20,744 | **59** | **681 (3.283%)** | 207 | **1.86e-37** |

**Where the language flips.** At `|γ| < 1e-3` the channel's input-dependent output falls
below fp32 noise relative to its own `β`. SegNet sits **entirely on the live side** — not
one channel crosses. PoseNet sits with 3.283% across. PoseNet's `running_var` minimum is
`5.6e-45`, the smallest fp32 subnormal: MEASURED, that channel's pre-activation was
*constant* over the whole training set.

**Localisation** (the actionable coordinate, not a ranking): the dead mass is not diffuse.
It concentrates in `token_mixer.mixer.conv_kxk.0.bn` of the deep FastViT stages —
stage 2 blocks run 18.8–22.7% dead, stage 3 blocks 23.0% and **25.8%**. The `conv_scale.bn`
siblings are near-live (0.4–3.7%) but carry almost all the subnormal-variance channels.
Stem and stage 0 are ≤3.1%.

**Degenerate-baseline control (required).** A naive "all_equal" scan reports 34 degenerate
PoseNet tensors — but 32 of those are 1-element tensors from `AllNorm` (`BatchNorm1d(1)`),
for which "all elements equal" is **true by construction**; the trivial baseline is 100%.
Excluding size-1 tensors, PoseNet has exactly **2** all-equal tensors and both are the
constant buffers `_mean`/`_std`. SegNet's 78 all-equal tensors are exactly the 78 scalar
counters — same trivial baseline. **Neither file contains a single all-zero or all-equal
weight tensor of size > 12.** Global exact-zero count is **0 of 9,610,567** (SegNet) and
**0 of 13,943,652** (PoseNet); non-finite count 0 in both. There is no sparsity to exploit
and no pruning already baked in.

**What moves this level set.** The dead-channel fraction is a property of the frozen file
and cannot move. What moves is *our* use of it: it bounds how much of PoseNet's
representation any input-side carrier can ever address.

**Which way it falls.** Asymmetrically. SegNet is fully live — every channel is reachable
from the input, so there is no free invariance there. PoseNet carries 3.28% of its BN
channels as input-invariant. This *extends* #725/hb1 (which measured SegNet capacity and
found the same 78 units) onto the pose half, where hb1 did not go.

---

## 4. SURFACE: *head rank* — verifying a CLAUDE.md MEASURED claim

CLAUDE.md records "SegNet head = EXACT rank-4 linear". **Verified against the file, and it
is SOUND** — with a correction to how it may be read.

**Coordinates.** `segmentation_head.0.weight` is `(5,16,3,3)` = 720 params + 5 bias = 725,
the entire head. Flatten to `A ∈ R^{5×144}`. `argmax` is invariant to adding a per-pixel
constant across the 5 logits, so the *argmax-effective* operator is `D = P A` with
`P = I − 11ᵀ/5`.

| operator | singular values | rank |
|---|---|---|
| `A` (5×144) | 3.2847, 2.2074, 2.0481, 1.8814, 1.5353 | **5** |
| `D = P A` (argmax-effective) | 3.1284, 2.1543, 2.0247, 1.7963, **3.03e-16** | **4** |
| all 10 pairwise `A_i − A_j` (10×144) | 6.995, 4.817, 4.527, 4.017, 1.3e-15 | **4** |

**Degenerate-baseline control (required).** The maximum possible rank of a sum-zero
projection of any 5-row matrix is **4**. So rank-4 is the **ceiling**, not a discovered
deficiency — it would hold identically for a randomly initialised 5-class head. The trained
weights are *full* rank 5 with a healthy smallest singular value 1.535 (ratio to largest
2.14:1). **The claim is exactly true and it is true by construction.** It must not be read
as "the trained head collapsed a dimension" — nothing was lost in training.

**Where the language flips.** It would flip to a *finding* only if `σ₄(D)` were small
relative to `σ₁(D)`. It is not: 1.796 vs 3.128, ratio 1.74. All four argmax-effective
directions are live.

**What moves it.** Class count. With 5 classes the argmax-effective rank is 4, period.

Per-class-pair `‖Δw‖_F` and `Δbias` (the flip-distance denominators, agreeing with the
form hb1 registered as `segnet_head_rank4_linear_flipdist_v1`) are in the artifact dir;
the widest pair is 1–3 (`‖Δw‖ = 4.007`, Lane↔Movable), the narrowest 0–2
(`‖Δw‖ = 2.602`, Road↔Undrivable). Head bias = `[−0.0327, −0.0272, +0.1016, −0.0822, +0.0504]`.

---

## 5. SURFACE: *scored-output concentration* — where d_pose actually lives

The charter asked how many parameters feed only-unscored outputs. The exact answer is
small; the surface it sits on is not.

**Coordinates.** `hydra.final_layer.pose` is `Linear(32 → 12)`: `weight (12,32)` +
`bias (12)` = 396 params. `compute_distortion` (`upstream/modules.py`) slices
`[..., : h.out // 2]` = the **first 6** outputs.

**MEASURED dead set:** rows 6–11 of weight (192) + biases 6–11 (6) = **198 params =
792 bytes**, exactly **50.00%** of the final layer and **1.420e-05** of PoseNet.

**Where the language flips — and it does not propagate.** I tested whether any of the 32
hidden units feeds *only* the unscored rows: the minimum column-norm of the **scored**
block is 0.149 (max-normalised, well clear of zero), so **zero** hidden units are
dead-exclusive. The invariance is confined to those 198 params and buys no upstream
pruning. That is a NEGATIVE and it is the honest ceiling on this route.

**The larger coordinate on the same surface.** `W_scored` is `(6,32)` of rank 6, so
`dim ker(W_scored) = 26` — **81.2% of the final 32-dim hidden space is d_pose-invisible**.
And within the visible 6 dimensions the metric is extremely anisotropic:

| | σ₁ | σ₂ | σ₃ | σ₄ | σ₅ | σ₆ | cond | σ₁²/σ₆² |
|---|---|---|---|---|---|---|---|---|
| `W_scored` | 8.7686 | 0.9877 | 0.6996 | 0.5121 | 0.4198 | 0.3533 | **24.82** | **615.9** |
| MEASURED control, 200× iid N(0,1) `(6,32)` | | | | | | | median **2.00**, p95 2.46 | median **4.0** |

The anisotropy is 12.4× the random-matrix condition number and 154× its gain ratio. This
is not a random-matrix artifact.

Per-output share of `d_pose` under an isotropic hidden perturbation
(`‖w_i‖² / Σ‖w_j‖²`; trivial baseline = 16.667% each):

| dim0 | dim1 | dim2 | dim3 | dim4 | dim5 |
|---|---|---|---|---|---|
| **97.363%** | 1.229% | 0.462% | 0.266% | 0.238% | 0.442% |

Effective dimension `exp(H)` of that share distribution = **1.172 of 6** (trivial
baseline 6.000); spectral effective-rank **1.166 of 6**.

**Which way it falls.** `d_pose` is, at the last layer, a **rank-≈1 functional**: one
output direction carries 97.4% of the response to a generic perturbation, and 81.2% of
the incoming hidden space is exactly invisible to it. Note the unscored rows are *larger*
(‖row₆‖ = 9.71 > ‖row₀‖ = 8.77) and 93.5% of the unscored head's Frobenius mass lies in
`ker(W_scored)` — the network spends real capacity on outputs the contest never reads.

**Prior-work position.** This is CONSISTENT with, and a file-level *explanation* of, the
already-banked empirical result recorded in MEMORY as "e_p RANK-1 ~2KB MEASURED-CLOSED":
a rank-1 pose-error carrier closes because the scored pose functional is itself
effective-rank 1.17. That prior result is SOUND and I am not re-deriving it — I am
supplying the mechanism from the weights. Within the scope stated in §0 I did not find a
row-norm / rank / dim-0-share analysis of `hydra.final_layer.pose`; the repo references
that layer only in the CPU/CUDA x-ray drift tooling
(`tools/cpu_cuda_xray_posenet_layer_drift.py:169`, `src/tac/analysis/scorer_native_diff.py:64`).

---

## 6. SURFACE: *file-byte closure* — is there anything hidden in these files?

**Coordinates.** Bytes of the file not addressed by any header `data_offsets` interval.

MEASURED, both files:

| | `8 + header` | tensor data span | interior gaps | tail after last tensor | sums to file? |
|---|---|---|---|---|---|
| segnet | 60,000 | 38,442,892 | **0** (0 bytes) | **0** | 38,502,892 ✓ |
| posenet | 60,952 | 55,774,608 | **0** (0 bytes) | **0** | 55,835,560 ✓ |

Header JSON trailing-space padding: 1 byte (segnet), 2 bytes (posenet). Tensor intervals
are contiguous and non-overlapping. **Every byte of both files is accounted for by a
declared tensor or the header.** There is no unaddressed region, no tail payload, no
overlap aliasing.

`_mean` and `_std` (PoseNet, 12 elements each, 96 bytes total) are MEASURED bit-exactly
equal to `float32(255/2)` and `float32(255/4)` — they are *used* in the forward
(`(x − _mean)/_std`) but carry zero learned information; they are re-derivable constants
occupying 96 file bytes. SegNet has no such buffers.

**Which way it falls.** Closed. Any future claim of "hidden state in the scorer files" has
to contend with a zero-gap byte account.

---

## 7. Round-1 adversarial review (what I tried to break, and what survived)

| I tried to refute | outcome |
|---|---|
| "the 78 I64 are `num_batches_tracked`" — could be indices/masks | **Survived.** All 78 are shape `[]` scalars named `*.num_batches_tracked`; they load into `nn.BatchNorm` buffers on the real instantiated model. |
| "the counters split by encoder/decoder" — maybe stem or head differs | **Survived.** Exhaustive: 68/68 encoder incl. `encoder.model.bn1`, 10/10 decoder. Zero exceptions. |
| "encoder was frozen, hence different counts" | **Refuted by the sign.** A frozen module's counter does not increment; it would read *lower*, not 13.1× higher. |
| "PoseNet is missing tensors → strict load must fail" | **Refuted.** Load succeeds; located the exact back-fill in `_NormBase._load_from_state_dict` keyed on absent `_metadata` version. Downgraded from "bug" to "reported latent fragility in pinned upstream". |
| "rank-4 head is a discovered property" | **Refuted as a finding, retained as a check.** Rank-4 is the structural ceiling for 5-class argmax; would hold at random init. Claim SOUND, reading corrected. |
| "PoseNet dead channels are just small-γ noise" | **Survived.** 59 channels have `running_var < 1e-8` incl. one at the fp32 subnormal floor `5.6e-45`, and min effective gain is 1.86e-37 — beyond any noise interpretation. |
| "198 dead params extend upstream to whole hidden units" | **Refuted.** Min scored column-norm 0.149; zero dead-exclusive hidden units. Reported as a negative. |
| "pose-head anisotropy is a random-matrix artifact" | **Refuted with a measured control.** 200 iid gaussian `(6,32)`: cond median 2.00 (p95 2.46) vs measured 24.82. |
| "34 degenerate PoseNet tensors is a real finding" | **Refuted by the trivial baseline.** 32 of 34 are 1-element `AllNorm` tensors where all-equal is automatic. True count 2, both constants. |

## 8. What I did not do

No forward pass, no n600 measurement, no scorer slot, no training, no dispatch, no exact
eval. Every row here is a property of two files and of `upstream/modules.py` +
vendored torch source. Nothing here is a score. The pointer **0.1910828242
[contest-CPU] is UNMOVED**, and nothing in this memo claims otherwise.
