# ADDENDUM — PR130's tokens ARE the GT labels · the DALI/AV provenance confound · the compute split

Supersedes both of MAIN's earlier GT-vs-VEH framings in `PR130_BASE_ROADMAP.md` (2026-08-09).

## 1. MEASURED AT SOURCE: PR130's tokens are the GT SegNet argmax

`repro_repo/code/build_gt_cache_official.py:50-58` — loads the official `DistortionNet` with
`posenet_sd_path`/`segnet_sd_path`, runs it over the ORIGINAL videos, stores
`segment.argmax(1).to(torch.uint8)`. That is `lstars`.

`repro_repo/code/train_semantic_full.py`:
```
:70   tokens = torch.load(args.cache)["seg"].long()      # the GT argmax cache
:116  target = tokens[batch_ids]
:124  frame  = render_for_seg(model, target, idx, exact_path=True)   # renders FROM GT labels
:36   pred   = segnet(frame).argmax(1)
:37   mismatches += int((pred != target).sum())                      # this IS d_seg
```

**Consequences:**
- PR130 already ships ACTUAL GT + LEARNED PRIOR. That is their vehicle, not a proposal.
- Their `d_seg = 0.00028609` is PURE REALIZATION ERROR — zero label error by construction.
- "GT labels vs VEH labels" was a FALSE DICHOTOMY. VEH labels (our renderer's own argmax) cap
  d_seg at our renderer's error with NO path below it — a rate datum, not a vehicle.
- The 15.05× (our 0.004305420 vs their 0.00028609) is **entirely a RENDERER gap on the same
  target**. Their renderer is 40,252 B.
- **The learned prior is the enabling mechanism** (operator 2026-08-09 "The learned prior is
  crucial"): raw dense GT labels measure ~713,345 B (tk1); HPAC codes the same content to
  ~137 KB. That ~5× is what converts an unaffordable exact target into an affordable one. Once
  the target is exact, d_seg collapses from a representation problem to a realization problem.

## 2. THE PROVENANCE CONFOUND (operator 2026-08-09 "our using pyav was likely a source of
confounding perhaps we should be optimizing against upstream")

`upstream/evaluate.py:33-42` — the dataset class is chosen BY DEVICE, not by preference:
```
if <cuda>:  DefaultDatasetClass = DaliVideoDataset    # nvdec
else:       DefaultDatasetClass = AVVideoDataset      # PyAV
```
The contest leaderboard runs CUDA ⇒ **DALI**. Our local CPU runs ⇒ **AV**.
PR130 built their cache `--dataset dali` (`repro_repo/scripts/e2e.py:304`).

Our PyAV use is NOT a deviation per se: `upstream/frame_utils.py:159-183` is deliberately
engineered to match — docstring *"yuv420 to rgb matching nvdec output"*, bilinear chroma
upsampling + BT.601 limited range — and our standing rule (GT decodes ONLY via
`frame_utils.yuv420_to_rgb`, never PyAV `rgb24`) already avoided the ~100× phantom-pose version.

**But "matches nvdec" is an INTENT, not proven bit-identity.** nvdec is a hardware decoder;
`frame_utils.py:175` upsamples chroma `mode='bilinear', align_corners=False` = CENTERED siting,
while nvdec sites chroma left/MPEG-2 style. That is exactly **#906 CHROMA SITING**, filed
DEFERRED for want of CUDA+DALI.

**What changed is the SCALE.** At our d_seg 0.004305 the disagreement was a curiosity the gap
dwarfed. Against PR130's 0.00028609 it is potentially the size of the ENTIRE remaining seg term.
Consequences if the delta is non-trivial:
- our GT-HPAC 135,732 B and their 137,159 B are NOT the same object (different label content),
  so "we are 1,427 B below them" is UNSAFE until checked;
- every seg measurement taken against AV-GT inherits the error silently;
- we would be training toward a target the authority does not score.

### 2b. MEASURED 2026-08-09 — the siting sensitivity is 79.66% of PR130's ENTIRE seg term

`tools/measure_chroma_siting_argmax_sensitivity.py` (afa34a0860), n=120 stratified-random,
frozen CPU-torch SegNet, `[macOS-CPU advisory]`. Receipt:
`.omx/research/ddm_rm1_20260808/chroma_siting_sensitivity.json`.

| quantity | value |
|---|---|
| **pooled argmax disagreement (centered vs left-sited chroma)** | **2.2790696885850695e-4** |
| in S units (100·d_seg) | **0.022791 S** |
| PR130's ENTIRE d_seg | 2.8609e-4 → 0.028609 S |
| **ratio** | **79.66%** |
| per-pair max / min | 4.3233e-4 (151% of their seg term) / 8.6466e-5 (30%) |
| pairs with ANY disagreement | **120 / 120** — systematic, not a tail |
| positive control | PASSED — our centered path is BYTE-IDENTICAL to upstream `yuv420_to_rgb` (pair 3) |

~45 argmax pixels per frame move under a half-luma-pixel horizontal chroma shift. That is a small
number — but PR130's realization error is ALSO small (~56 px/frame), so these are the same order.

**What this DOES establish (MEASURED):** the frozen SegNet argmax is siting-sensitive at ~80% of
the bar's seg term. Any renderer program aimed below ~1e-3 d_seg is carrying an unmeasured label
uncertainty of that size. At OUR 0.004305 it is 5.3% (ignorable); at the BAR it is not.

**What this does NOT establish (still UNMEASURED):** which convention nvdec/DALI actually emits.
Upstream *intends* `yuv420_to_rgb` to match nvdec ("yuv420 to rgb matching nvdec output"); this
probe cannot confirm or refute that — it prices the consequence IF they differ. That is exactly
what the one qualifying Modal job (§3) resolves.

**Routing:** the DALI-vs-AV dispatch moves from "worth doing" to **prerequisite** for any
renderer work targeting the bar's seg term, and for the 135,732-vs-137,159 B comparison. #906
stays OPEN with the sensitivity now priced.

## 3. THE COMPUTE SPLIT (operator binding 2026-08-09)

> *"We can use Modal, but only for what's absolutely necessary, and all long runs / training
> needs to happen on local."*

- **Modal (CUDA) = ONLY what physically cannot run here, and SHORT.** DALI/nvdec decode is the
  canonical example: it requires CUDA and cannot be reproduced locally at all.
- **ALL training and all long runs = LOCAL Metal/MLX.** No exceptions for convenience.
- Corollary that makes this efficient: a Modal dispatch should buy a DURABLE ASSET, not a
  service. One short CUDA job that emits the authoritative DALI GT cache is consumed by every
  subsequent local training run forever. Buy the artifact once; train against it locally.

**The one qualifying job:** run `build_gt_cache_official.py` at `--dataset dali` AND
`--dataset av` on the same videos, diff the argmax, report the disagreement rate in units
directly comparable to 2.86e-4. Minutes of GPU. Outputs: (a) the DALI↔AV delta = the #906 answer;
(b) the authoritative DALI GT cache as a durable local asset. Everything downstream — HPAC prior
training, renderer training, burns — stays on Metal.

## Honesty labels

MEASURED: all file:line facts above (read at source 2026-08-09), the byte figures, the two d_seg
values. UNMEASURED: the DALI↔AV label delta itself — that is the job. DERIVED: the chroma-siting
mechanism candidate (from the centered-bilinear line vs nvdec convention), not yet confirmed.
